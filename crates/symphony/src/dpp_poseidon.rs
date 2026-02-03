//! Poseidon-permutation arithmetization (prime-field dR1CS skeleton).
//!
//! This is an incremental step toward arithmetizing the full WE-facing verifier relation:
//! we first arithmetize the Poseidon permutation(s) used by the transcript.

use ark_ff::{BigInteger, PrimeField};
use ark_serialize::CanonicalSerialize;
use rayon::prelude::*;
use core::ops::Range;
use std::path::Path;

use crate::poseidon_trace::{permute_in_place, permute_with_round_trace, PoseidonReplayError};
use crate::poseidon_trace::{replay_ops, PoseidonReplayError as ReplayErr, PoseidonSpongeReplayResult};
use crate::transcript::PoseidonTraceOp;
use crate::file_backed_dr1cs::{
    fast_prepare_out_dir, FileBackedLayout, FileBackedSparseDr1csInstance, SparseDr1csFileWriter,
};
#[cfg(unix)]
use crate::file_backed_dr1cs::FileBackedRangeWriter;

#[derive(Debug)]
enum PoseidonInstance<F: PrimeField> {
    InMemory(SparseDr1csInstance<F>),
    FileBacked(FileBackedSparseDr1csInstance<F>),
}


#[inline]
fn poseidon_profile_on() -> bool {
    match std::env::var("LF_PLUS_PROFILE") {
        Ok(v) => v != "0",
        Err(_) => false,
    }
}

fn escape_json_str(input: &str) -> String {
    input
        .chars()
        .flat_map(|c| match c {
            '\\' => "\\\\".chars().collect::<Vec<_>>(),
            '"' => "\\\"".chars().collect::<Vec<_>>(),
            '\n' => "\\n".chars().collect::<Vec<_>>(),
            '\r' => "\\r".chars().collect::<Vec<_>>(),
            '\t' => "\\t".chars().collect::<Vec<_>>(),
            _ => vec![c],
        })
        .collect()
}

fn debug_log(hypothesis_id: &str, location: &str, message: &str, data_json: &str) {
    use std::io::Write;
    let timestamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0);
    let id = format!(
        "log_{}_{}",
        timestamp,
        location
            .chars()
            .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
            .collect::<String>()
    );
    let payload = format!(
        "{{\"id\":\"{}\",\"timestamp\":{},\"location\":\"{}\",\"message\":\"{}\",\"data\":{},\"sessionId\":\"debug-session\",\"runId\":\"run1\",\"hypothesisId\":\"{}\"}}",
        escape_json_str(&id),
        timestamp,
        escape_json_str(location),
        escape_json_str(message),
        data_json,
        escape_json_str(hypothesis_id),
    );
    if let Ok(mut f) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open("/tmp/debug.log")
    {
        let _ = writeln!(f, "{payload}");
    }
}

#[derive(Clone, Debug)]
pub struct Constraint {
    pub a: Range<usize>,
    pub b: Range<usize>,
    pub c: Range<usize>,
}

#[derive(Clone, Debug)]
pub struct SparseDr1csInstance<F: PrimeField> {
    pub nvars: usize,
    pub constraints: Vec<Constraint>,
    pub a_terms: Vec<(F, usize)>,
    pub b_terms: Vec<(F, usize)>,
    pub c_terms: Vec<(F, usize)>,
}

impl<F: PrimeField> SparseDr1csInstance<F> {
    pub fn eval_lc(terms: &[(F, usize)], assignment: &[F]) -> F {
        // IMPORTANT: keep this sequential.
        //
        // `check()` already parallelizes across constraints. Parallelizing inside each linear
        // combination can introduce nested Rayon parallelism, which is high-overhead and can
        // trigger stack overflows on large instances.
        terms
            .iter()
            .fold(F::ZERO, |acc, (c, idx)| acc + (*c * assignment[*idx]))
    }

    pub fn check(&self, assignment: &[F]) -> Result<(), String> {
        if assignment.len() != self.nvars {
            return Err(format!(
                "assignment length mismatch: expected {}, got {}",
                self.nvars,
                assignment.len()
            ));
        }

        let failed = self
            .constraints
            .par_iter()
            .enumerate()
            .find_any(|(_, row)| {
                let a = Self::eval_lc(&self.a_terms[row.a.clone()], assignment);
                let b = Self::eval_lc(&self.b_terms[row.b.clone()], assignment);
                let c = Self::eval_lc(&self.c_terms[row.c.clone()], assignment);
                a * b != c
            });

        if let Some((i, _)) = failed {
            return Err(format!("constraint {i} failed"));
        }
        Ok(())
    }

    /// Convert to the dense dR1CS format used by the prototype RS FLPCP.
    pub fn to_dense(&self) -> dpp::dr1cs_flpcp::Dr1csInstance<F> {
        let n = self.nvars;
        
        let rows: Vec<(Vec<F>, Vec<F>, Vec<F>)> = self.constraints.par_iter()
            .map(|row| {
            let mut ra = vec![F::ZERO; n];
            let mut rb = vec![F::ZERO; n];
            let mut rc = vec![F::ZERO; n];
            for (coeff, idx) in &self.a_terms[row.a.clone()] {
                ra[*idx] += *coeff;
            }
            for (coeff, idx) in &self.b_terms[row.b.clone()] {
                rb[*idx] += *coeff;
            }
            for (coeff, idx) in &self.c_terms[row.c.clone()] {
                rc[*idx] += *coeff;
            }
                (ra, rb, rc)
            })
            .collect();
        
        let (a, b, c): (Vec<_>, Vec<_>, Vec<_>) = rows.into_iter()
            .map(|(ra, rb, rc)| (ra, rb, rc))
            .fold((Vec::new(), Vec::new(), Vec::new()), |(mut a, mut b, mut c), (ra, rb, rc)| {
            a.push(ra);
            b.push(rb);
            c.push(rc);
                (a, b, c)
            });
        
        dpp::dr1cs_flpcp::Dr1csInstance { a, b, c }
    }
}

/// Merge multiple sparse dR1CS instances into one, sharing variable 0 as the constant-1 slot.
///
/// Each part is assumed to have `assignment[0] = 1`. In the merged instance:
/// - var 0 is shared across all parts
/// - all other variables are appended, and constraints are re-indexed accordingly
pub fn merge_sparse_dr1cs_share_one<F: PrimeField + Copy + Send + Sync>(
    parts: Vec<(SparseDr1csInstance<F>, Vec<F>)>,
) -> Result<(SparseDr1csInstance<F>, Vec<F>), String> {
    if parts.is_empty() {
        return Err("merge_sparse_dr1cs_share_one: empty parts".to_string());
    }

    // First pass: validate parts + compute prefix offsets (so we can fill in parallel).
    let mut total_constraints: usize = 0;
    let mut total_assignment_tail: usize = 0; // excludes the shared var0
    let mut total_terms_a: usize = 0;
    let mut total_terms_b: usize = 0;
    let mut total_terms_c: usize = 0;

    let mut var_offsets: Vec<usize> = Vec::with_capacity(parts.len());
    let mut a_offsets: Vec<usize> = Vec::with_capacity(parts.len());
    let mut b_offsets: Vec<usize> = Vec::with_capacity(parts.len());
    let mut c_offsets: Vec<usize> = Vec::with_capacity(parts.len());
    let mut row_offsets: Vec<usize> = Vec::with_capacity(parts.len());

    for (inst, asg) in parts.iter() {
        if asg.is_empty() || asg[0] != F::ONE {
            return Err("merge_sparse_dr1cs_share_one: each part must have assignment[0]=1".to_string());
        }
        if inst.nvars != asg.len() {
            return Err("merge_sparse_dr1cs_share_one: inst/assignment length mismatch".to_string());
        }

        var_offsets.push(total_assignment_tail);
        a_offsets.push(total_terms_a);
        b_offsets.push(total_terms_b);
        c_offsets.push(total_terms_c);
        row_offsets.push(total_constraints);

        total_constraints = total_constraints
            .checked_add(inst.constraints.len())
            .ok_or_else(|| "merge_sparse_dr1cs_share_one: constraint count overflow".to_string())?;
        total_assignment_tail = total_assignment_tail
            .checked_add(asg.len().saturating_sub(1))
            .ok_or_else(|| "merge_sparse_dr1cs_share_one: assignment size overflow".to_string())?;
        total_terms_a = total_terms_a
            .checked_add(inst.a_terms.len())
            .ok_or_else(|| "merge_sparse_dr1cs_share_one: a_terms size overflow".to_string())?;
        total_terms_b = total_terms_b
            .checked_add(inst.b_terms.len())
            .ok_or_else(|| "merge_sparse_dr1cs_share_one: b_terms size overflow".to_string())?;
        total_terms_c = total_terms_c
            .checked_add(inst.c_terms.len())
            .ok_or_else(|| "merge_sparse_dr1cs_share_one: c_terms size overflow".to_string())?;
    }

    let merged_nvars = 1usize
        .checked_add(total_assignment_tail)
        .ok_or_else(|| "merge_sparse_dr1cs_share_one: merged_nvars overflow".to_string())?;

    // Allocate output buffers without initializing (we will write all entries).
    let mut merged_assignment: Vec<F> = Vec::with_capacity(merged_nvars);
    let mut merged_constraints: Vec<Constraint> = Vec::with_capacity(total_constraints);
    let mut merged_a_terms: Vec<(F, usize)> = Vec::with_capacity(total_terms_a);
    let mut merged_b_terms: Vec<(F, usize)> = Vec::with_capacity(total_terms_b);
    let mut merged_c_terms: Vec<(F, usize)> = Vec::with_capacity(total_terms_c);
    unsafe {
        merged_assignment.set_len(merged_nvars);
        merged_constraints.set_len(total_constraints);
        merged_a_terms.set_len(total_terms_a);
        merged_b_terms.set_len(total_terms_b);
        merged_c_terms.set_len(total_terms_c);
    }
    merged_assignment[0] = F::ONE;

    // Raw pointers for parallel, disjoint writes.
    //
    // Rayon requires captured state to be `Sync`; raw pointers are not `Sync` by default.
    // We wrap them and assert (by construction) that each thread writes to disjoint ranges.
    #[derive(Copy, Clone)]
    struct SyncPtr<T>(*mut T);
    unsafe impl<T> Send for SyncPtr<T> {}
    unsafe impl<T> Sync for SyncPtr<T> {}
    impl<T> SyncPtr<T> {
        #[inline]
        unsafe fn add(&self, off: usize) -> *mut T {
            self.0.add(off)
        }
        #[inline]
        unsafe fn write(&self, off: usize, v: T) {
            core::ptr::write(self.0.add(off), v);
        }
    }

    let out_asg: SyncPtr<F> = SyncPtr(merged_assignment.as_mut_ptr());
    let out_rows: SyncPtr<Constraint> = SyncPtr(merged_constraints.as_mut_ptr());
    let out_a: SyncPtr<(F, usize)> = SyncPtr(merged_a_terms.as_mut_ptr());
    let out_b: SyncPtr<(F, usize)> = SyncPtr(merged_b_terms.as_mut_ptr());
    let out_c: SyncPtr<(F, usize)> = SyncPtr(merged_c_terms.as_mut_ptr());

    // Fill all output buffers in parallel, one part per task.
    parts
        .into_par_iter()
        .enumerate()
        .for_each(move |(i, (inst, asg))| unsafe {
            // Assignment tail copy.
            let vo = var_offsets[i];
            let dst = out_asg.add(1 + vo);
            core::ptr::copy_nonoverlapping(asg.as_ptr().add(1), dst, asg.len().saturating_sub(1));

            // Terms with var-index remap.
            let ao = a_offsets[i];
            for (j, (coeff, idx)) in inst.a_terms.iter().copied().enumerate() {
                let mut new_idx = idx;
                if new_idx != 0 {
                    new_idx += vo;
                }
                out_a.write(ao + j, (coeff, new_idx));
            }
            let bo = b_offsets[i];
            for (j, (coeff, idx)) in inst.b_terms.iter().copied().enumerate() {
                let mut new_idx = idx;
                if new_idx != 0 {
                    new_idx += vo;
                }
                out_b.write(bo + j, (coeff, new_idx));
            }
            let co = c_offsets[i];
            for (j, (coeff, idx)) in inst.c_terms.iter().copied().enumerate() {
                let mut new_idx = idx;
                if new_idx != 0 {
                    new_idx += vo;
                }
                out_c.write(co + j, (coeff, new_idx));
            }

            // Constraints with term-range remap.
            let ro = row_offsets[i];
            for (j, row) in inst.constraints.iter().enumerate() {
                out_rows.write(ro + j, Constraint {
                    a: (ao + row.a.start)..(ao + row.a.end),
                    b: (bo + row.b.start)..(bo + row.b.end),
                    c: (co + row.c.start)..(co + row.c.end),
                });
            }
        });

    Ok((
        SparseDr1csInstance {
            nvars: merged_nvars,
            constraints: merged_constraints,
            a_terms: merged_a_terms,
            b_terms: merged_b_terms,
            c_terms: merged_c_terms,
        },
        merged_assignment,
    ))
}

/// Merge multiple sparse dR1CS instances (sharing var 0) and add *glue* equality constraints
/// between variables belonging to different parts.
///
/// `glue` entries are `(part_a, var_a, part_b, var_b)` in **local** indices.
/// The merged instance enforces `var_a == var_b` for each glue entry.
pub fn merge_sparse_dr1cs_share_one_with_glue<F: PrimeField>(
    parts: Vec<(SparseDr1csInstance<F>, Vec<F>)>,
    glue: &[(usize, usize, usize, usize)],
) -> Result<(SparseDr1csInstance<F>, Vec<F>), String> {
    merge_sparse_dr1cs_share_one_with_glue_impl(parts, glue, true)
}

/// Same as `merge_sparse_dr1cs_share_one_with_glue`, but **does not** require glued variables
/// to have identical witness values in the provided assignments.
///
/// This is useful for *shape-only* / arm-time builds that use dummy witness values: the merged
/// instance (constraints + variable identification) is what matters, not the particular dummy
/// assignment used during construction.
pub fn merge_sparse_dr1cs_share_one_with_glue_relaxed<F: PrimeField>(
    parts: Vec<(SparseDr1csInstance<F>, Vec<F>)>,
    glue: &[(usize, usize, usize, usize)],
) -> Result<(SparseDr1csInstance<F>, Vec<F>), String> {
    merge_sparse_dr1cs_share_one_with_glue_impl(parts, glue, false)
}

fn merge_sparse_dr1cs_share_one_with_glue_impl<F: PrimeField>(
    parts: Vec<(SparseDr1csInstance<F>, Vec<F>)>,
    glue: &[(usize, usize, usize, usize)],
    check_assignment_consistency: bool,
) -> Result<(SparseDr1csInstance<F>, Vec<F>), String> {
    if parts.is_empty() {
        return Err("merge_sparse_dr1cs_share_one_with_glue: empty parts".to_string());
    }

    // Fast path: no glue => standard merge.
    if glue.is_empty() {
        return merge_sparse_dr1cs_share_one(parts);
    }

    // Compute offsets for each part (how much its non-const vars are shifted by in merged space).
    let mut offsets: Vec<usize> = Vec::with_capacity(parts.len());
    let mut merged_assignment: Vec<F> = vec![F::ONE];
    let total_constraints: usize = parts.iter().map(|(inst, _)| inst.constraints.len()).sum();
    let total_assignment_len: usize = parts.iter().map(|(_, asg)| asg.len()).sum();
    // #region agent log
    debug_log(
        "H1",
        "dpp_poseidon.rs:merge_sparse_dr1cs_share_one_with_glue_impl:entry",
        "merge entry",
        &format!(
            "{{\"parts_len\":{},\"glue_len\":{},\"check_assignment_consistency\":{},\"total_constraints\":{},\"total_assignment_len\":{}}}",
            parts.len(),
            glue.len(),
            check_assignment_consistency,
            total_constraints,
            total_assignment_len
        ),
    );
    // #endregion
    for (inst, asg) in parts.iter() {
        if asg.is_empty() || asg[0] != F::ONE {
            return Err("merge_sparse_dr1cs_share_one_with_glue: each part must have assignment[0]=1".to_string());
        }
        if inst.nvars != asg.len() {
            return Err("merge_sparse_dr1cs_share_one_with_glue: inst/assignment length mismatch".to_string());
        }
        offsets.push(merged_assignment.len() - 1);
        merged_assignment.extend_from_slice(&asg[1..]);
    }

    let remap_global = |part_idx: usize, local: usize, offsets: &[usize]| -> usize {
        if local == 0 { 0 } else { local + offsets[part_idx] }
    };

    // --------------------------------------------------------------------
    // Variable unification for glue:
    // Instead of adding explicit equality constraints, we *identify* glued variables
    // into a single variable (like an R1CS variable rename), shrinking nvars and also
    // eliminating glue constraints.
    //
    // This is safe because glue is only used to equate variables that are intended to be
    // exactly equal (e.g. Poseidon squeeze vars == verifier coin vars, or shared witness
    // variables across sub-circuits). If two glued vars have different assignments, we
    // return an error (the caller constructed an inconsistent witness).
    // --------------------------------------------------------------------
    use std::collections::HashMap;

    // Collect all *global* indices that appear in glue.
    let mut idx_map: HashMap<usize, usize> = HashMap::with_capacity(glue.len() * 2);
    let mut idxs: Vec<usize> = Vec::with_capacity(glue.len() * 2);
    let mut glue_pairs: Vec<(usize, usize)> = Vec::with_capacity(glue.len());

    let get_id = |g: usize, idx_map: &mut HashMap<usize, usize>, idxs: &mut Vec<usize>| -> usize {
        if let Some(&id) = idx_map.get(&g) {
            id
        } else {
            let id = idxs.len();
            idxs.push(g);
            idx_map.insert(g, id);
            id
        }
    };

    for &(pa, xa, pb, xb) in glue {
        if pa >= parts.len() || pb >= parts.len() {
            return Err("merge_sparse_dr1cs_share_one_with_glue: glue part idx out of range".to_string());
        }
        let ga = remap_global(pa, xa, &offsets);
        let gb = remap_global(pb, xb, &offsets);
        let ia = get_id(ga, &mut idx_map, &mut idxs);
        let ib = get_id(gb, &mut idx_map, &mut idxs);
        glue_pairs.push((ia, ib));
    }
    // #region agent log
    debug_log(
        "H2",
        "dpp_poseidon.rs:merge_sparse_dr1cs_share_one_with_glue_impl:glue_map",
        "glue mapping built",
        &format!(
            "{{\"idxs_len\":{},\"glue_pairs_len\":{},\"offsets_len\":{}}}",
            idxs.len(),
            glue_pairs.len(),
            offsets.len()
        ),
    );
    // #endregion

    // Union-find over the glued variable set.
    let m = idxs.len();
    let mut parent: Vec<usize> = (0..m).collect();
    let mut rank: Vec<u8> = vec![0u8; m];

    let find = |mut x: usize, parent: &mut [usize]| -> usize {
        // Path compression
        let mut root = x;
        while parent[root] != root {
            root = parent[root];
        }
        while parent[x] != x {
            let p = parent[x];
            parent[x] = root;
            x = p;
        }
        root
    };

    let union = |a: usize, b: usize, parent: &mut [usize], rank: &mut [u8]| {
        let ra = find(a, parent);
        let rb = find(b, parent);
        if ra == rb {
            return;
        }
        let (mut ra, mut rb) = (ra, rb);
        if rank[ra] < rank[rb] {
            core::mem::swap(&mut ra, &mut rb);
        }
        parent[rb] = ra;
        if rank[ra] == rank[rb] {
            rank[ra] = rank[ra].saturating_add(1);
        }
    };

    for (a, b) in glue_pairs {
        union(a, b, &mut parent, &mut rank);
    }

    // For each UF root, choose a representative global index (min global index).
    let mut rep_global_for_root: Vec<usize> = vec![usize::MAX; m];
    for local_id in 0..m {
        let r = find(local_id, &mut parent);
        let g = idxs[local_id];
        let slot = &mut rep_global_for_root[r];
        if *slot == usize::MAX || g < *slot {
            *slot = g;
        }
    }

    // Map each glued global index -> its representative global index.
    let mut rep_of_global: HashMap<usize, usize> = HashMap::with_capacity(m);
    for local_id in 0..m {
        let r = find(local_id, &mut parent);
        let rep_g = rep_global_for_root[r];
        let g = idxs[local_id];
        rep_of_global.insert(g, rep_g);
    }

    // Build a compacted assignment by dropping non-representative glued vars.
    let old_nvars = merged_assignment.len();
    let mut new_index: Vec<usize> = vec![usize::MAX; old_nvars];
    let mut new_assignment: Vec<F> = Vec::with_capacity(old_nvars);

    for i in 0..old_nvars {
        if let Some(&rep) = rep_of_global.get(&i) {
            if i != rep {
                // Consistency check: glued assignments must match exactly (witness-time safety).
                // For shape-only builds, callers may intentionally use arbitrary dummy values,
                // so we optionally skip this check.
                if check_assignment_consistency && merged_assignment[i] != merged_assignment[rep] {
                    let locate = |g: usize| -> Option<(usize, usize)> {
                        if g == 0 {
                            return Some((0, 0));
                        }
                        for (pi, (inst, _asg)) in parts.iter().enumerate() {
                            let off = offsets[pi];
                            let start = off + 1;
                            let end = off + inst.nvars;
                            if g >= start && g < end {
                                return Some((pi, g - off));
                            }
                        }
                        None
                    };
                    let (li_part, li_local) = locate(i).unwrap_or((usize::MAX, usize::MAX));
                    let (lr_part, lr_local) = locate(rep).unwrap_or((usize::MAX, usize::MAX));
                    // #region agent log
                    debug_log(
                        "H3",
                        "dpp_poseidon.rs:merge_sparse_dr1cs_share_one_with_glue_impl:inconsistent",
                        "inconsistent glued assignment",
                        &format!(
                            "{{\"g\":{},\"rep\":{},\"g_part\":{},\"g_local\":{},\"rep_part\":{},\"rep_local\":{},\"g_val\":\"{}\",\"rep_val\":\"{}\"}}",
                            i,
                            rep,
                            li_part,
                            li_local,
                            lr_part,
                            lr_local,
                            escape_json_str(&format!("{}", merged_assignment[i])),
                            escape_json_str(&format!("{}", merged_assignment[rep]))
                        ),
                    );
                    // #endregion
                    return Err(format!(
                        "merge_sparse_dr1cs_share_one_with_glue: inconsistent glued assignment ({} != {})",
                        i, rep
                    ));
                }
                continue; // drop this var
            }
        }
        new_index[i] = new_assignment.len();
        new_assignment.push(merged_assignment[i]);
    }
    // Fill indices for dropped vars: map to representative's new index.
    for (&g, &rep) in rep_of_global.iter() {
        if g != rep {
            new_index[g] = new_index[rep];
        }
    }
    if new_index[0] != 0 || new_assignment.get(0).copied().unwrap_or(F::ZERO) != F::ONE {
        return Err("merge_sparse_dr1cs_share_one_with_glue: internal error (const-1 slot)".to_string());
    }

    let total_constraints: usize = parts.iter().map(|(inst, _)| inst.constraints.len()).sum::<usize>();
    let total_terms_a: usize = parts.iter().map(|(inst, _)| inst.a_terms.len()).sum();
    let total_terms_b: usize = parts.iter().map(|(inst, _)| inst.b_terms.len()).sum();
    let total_terms_c: usize = parts.iter().map(|(inst, _)| inst.c_terms.len()).sum();

    let mut merged_constraints: Vec<Constraint> = Vec::with_capacity(total_constraints);
    let mut merged_a_terms: Vec<(F, usize)> = Vec::with_capacity(total_terms_a);
    let mut merged_b_terms: Vec<(F, usize)> = Vec::with_capacity(total_terms_b);
    let mut merged_c_terms: Vec<(F, usize)> = Vec::with_capacity(total_terms_c);

    // Merge constraints with remapped indices directly into pooled term arrays.
    for (part_idx, (inst, _asg)) in parts.iter().enumerate() {
        let offset = offsets[part_idx];
        let remap_idx = |idx: usize| -> usize {
            let g = if idx == 0 { 0 } else { idx + offset };
            new_index[g]
        };
        for row in &inst.constraints {
            let a0 = merged_a_terms.len();
            for (c, i) in &inst.a_terms[row.a.clone()] {
                merged_a_terms.push((*c, remap_idx(*i)));
            }
            let a1 = merged_a_terms.len();

            let b0 = merged_b_terms.len();
            for (c, i) in &inst.b_terms[row.b.clone()] {
                merged_b_terms.push((*c, remap_idx(*i)));
            }
            let b1 = merged_b_terms.len();

            let c0 = merged_c_terms.len();
            for (c, i) in &inst.c_terms[row.c.clone()] {
                merged_c_terms.push((*c, remap_idx(*i)));
            }
            let c1 = merged_c_terms.len();

            merged_constraints.push(Constraint { a: a0..a1, b: b0..b1, c: c0..c1 });
        }
    }

    // #region agent log
    debug_log(
        "H4",
        "dpp_poseidon.rs:merge_sparse_dr1cs_share_one_with_glue_impl:exit",
        "merge exit",
        &format!(
            "{{\"old_nvars\":{},\"new_nvars\":{},\"merged_constraints\":{}}}",
            old_nvars,
            new_assignment.len(),
            merged_constraints.len()
        ),
    );
    // #endregion
    Ok((
        SparseDr1csInstance {
            nvars: new_assignment.len(),
            constraints: merged_constraints,
            a_terms: merged_a_terms,
            b_terms: merged_b_terms,
            c_terms: merged_c_terms,
        },
        new_assignment,
    ))
}

#[derive(Debug)]
struct Dr1csBuilder<F: PrimeField> {
    assignment: Vec<F>,
    rows: Vec<Constraint>,
    a_terms: Vec<(F, usize)>,
    b_terms: Vec<(F, usize)>,
    c_terms: Vec<(F, usize)>,
    file_sink: Option<PoseidonFileSink<F>>,
    file_rows: u64,
    file_a_terms: u64,
    file_b_terms: u64,
    file_c_terms: u64,
    // Var index remap for direct-to-merged sharded writing.
    // Global var idx = 0 if local=0 else local + file_var_tail_off.
    file_var_tail_off: u32,
    // File-backed fast path: stage blocks in memory and flush via raw-block APIs.
    fb_modulus: u16,
    fb_stage_bytes: usize,
    fb_stage_limit_bytes: usize,
    fb_a_coeffs: Vec<u8>,
    fb_a_idx: Vec<u32>,
    fb_b_coeffs: Vec<u8>,
    fb_b_idx: Vec<u32>,
    fb_c_coeffs: Vec<u8>,
    fb_c_idx: Vec<u32>,
    fb_row_lens: Vec<u32>, // 3*u32 per row: a_len,b_len,c_len
}

#[derive(Debug)]
enum PoseidonFileSink<F: PrimeField> {
    Append(SparseDr1csFileWriter<F>),
    #[cfg(unix)]
    Range(FileBackedRangeWriter),
    /// Count-only: maintain `file_*` counters but do not write pools/rows.
    Count,
}

#[derive(Clone, Debug)]
pub struct RangeWriteResult {
    pub ckpts: Vec<(u64, u64, u64, u64)>,
    pub rows: u64,
    pub a_terms: u64,
    pub b_terms: u64,
    pub c_terms: u64,
}

impl<F: PrimeField + CanonicalSerialize> Dr1csBuilder<F> {
    #[inline]
    fn map_idx_u32(&self, idx: usize) -> u32 {
        if idx == 0 {
            return 0;
        }
        let add = self.file_var_tail_off as u64;
        let v = (idx as u64).saturating_add(add);
        v.try_into()
            .unwrap_or_else(|_| panic!("file-backed dr1cs: mapped var idx overflow u32 (idx={idx} add={add})"))
    }

    fn new() -> Self {
        #[inline]
        fn stage_limit_bytes() -> usize {
            // Keep staging bounded to avoid huge memory spikes in sharded mode.
            // Override with LFP_POSEIDON_FILE_BACKED_STAGE_MB.
            let mb: usize = std::env::var("LFP_POSEIDON_FILE_BACKED_STAGE_MB")
                .ok()
                .and_then(|s| s.parse().ok())
                .filter(|&v| v > 0)
                .unwrap_or(512);
            mb.saturating_mul(1024 * 1024)
        }
        // For file-backed mode we only support small prime fields; compute modulus once.
        let modulus_big = F::MODULUS;
        let limbs = modulus_big.as_ref();
        let fb_modulus: u16 = limbs.get(0).copied().unwrap_or(0).min(u16::MAX as u64) as u16;
        // var 0 is the constant-1 slot
        Self {
            assignment: vec![F::ONE],
            rows: Vec::new(),
            a_terms: Vec::new(),
            b_terms: Vec::new(),
            c_terms: Vec::new(),
            file_sink: None,
            file_rows: 0,
            file_a_terms: 0,
            file_b_terms: 0,
            file_c_terms: 0,
            file_var_tail_off: 0,
            fb_modulus,
            fb_stage_bytes: 0,
            fb_stage_limit_bytes: stage_limit_bytes(),
            fb_a_coeffs: Vec::new(),
            fb_a_idx: Vec::new(),
            fb_b_coeffs: Vec::new(),
            fb_b_idx: Vec::new(),
            fb_c_coeffs: Vec::new(),
            fb_c_idx: Vec::new(),
            fb_row_lens: Vec::new(),
        }
    }

    fn new_file_backed(dir: impl AsRef<std::path::Path>) -> Result<Self, String> {
        let mut b = Self::new();
        b.file_sink = Some(PoseidonFileSink::Append(SparseDr1csFileWriter::<F>::create(dir)?));
        Ok(b)
    }

    #[cfg(unix)]
    fn new_file_backed_range(writer: FileBackedRangeWriter, var_tail_off: u32) -> Self {
        let mut b = Self::new();
        b.file_var_tail_off = var_tail_off;
        b.file_sink = Some(PoseidonFileSink::Range(writer));
        b
    }

    fn new_count_only() -> Self {
        let mut b = Self::new();
        b.file_sink = Some(PoseidonFileSink::Count);
        b
    }

    #[inline]
    fn fb_coeff_u16(&self, coef: F) -> u16 {
        // Same convention as file-backed writer: canonical integer representative in [0, p-1].
        if coef == F::ZERO {
            0
        } else if coef == F::ONE {
            1
        } else if coef == -F::ONE {
            self.fb_modulus.wrapping_sub(1)
        } else {
            let big = coef.into_bigint();
            let limbs = big.as_ref();
            limbs.get(0).copied().unwrap_or(0) as u16
        }
    }

    fn fb_maybe_flush(&mut self) {
        if self.file_sink.is_none() {
            return;
        }
        if self.fb_stage_bytes < self.fb_stage_limit_bytes {
            return;
        }
        self.fb_flush().expect("file-backed dr1cs flush failed");
    }

    fn fb_flush(&mut self) -> Result<(), String> {
        let Some(sink) = self.file_sink.as_mut() else { return Ok(()); };
        if self.fb_row_lens.is_empty() {
            return Ok(());
        }
        // Write term pools first, then row lens block.
        match sink {
            PoseidonFileSink::Append(w) => {
                w.push_a_terms_raw_block(&self.fb_a_coeffs, &self.fb_a_idx)?;
                w.push_b_terms_raw_block(&self.fb_b_coeffs, &self.fb_b_idx)?;
                w.push_c_terms_raw_block(&self.fb_c_coeffs, &self.fb_c_idx)?;
                w.push_constraint_lens_block(&self.fb_row_lens)?;
            }
            #[cfg(unix)]
            PoseidonFileSink::Range(w) => {
                w.push_a_terms_raw_block(&self.fb_a_coeffs, &self.fb_a_idx)?;
                w.push_b_terms_raw_block(&self.fb_b_coeffs, &self.fb_b_idx)?;
                w.push_c_terms_raw_block(&self.fb_c_coeffs, &self.fb_c_idx)?;
                w.push_constraint_lens_block(&self.fb_row_lens)?;
            }
            PoseidonFileSink::Count => {
                // No-op: counters updated below.
            }
        }

        self.file_a_terms = self.file_a_terms.saturating_add(self.fb_a_idx.len() as u64);
        self.file_b_terms = self.file_b_terms.saturating_add(self.fb_b_idx.len() as u64);
        self.file_c_terms = self.file_c_terms.saturating_add(self.fb_c_idx.len() as u64);
        self.file_rows = self.file_rows.saturating_add((self.fb_row_lens.len() / 3) as u64);

        self.fb_a_coeffs.clear();
        self.fb_a_idx.clear();
        self.fb_b_coeffs.clear();
        self.fb_b_idx.clear();
        self.fb_c_coeffs.clear();
        self.fb_c_idx.clear();
        self.fb_row_lens.clear();
        self.fb_stage_bytes = 0;
        Ok(())
    }

    #[inline]
    fn is_file_backed(&self) -> bool {
        self.file_sink.is_some()
    }

    fn one(&self) -> usize {
        0
    }

    fn new_var(&mut self, value: F) -> usize {
        let idx = self.assignment.len();
        self.assignment.push(value);
        idx
    }

    fn add_constraint(&mut self, a: Vec<(F, usize)>, b: Vec<(F, usize)>, c: Vec<(F, usize)>) {
        if matches!(self.file_sink, Some(PoseidonFileSink::Count)) {
            self.file_rows = self.file_rows.saturating_add(1);
            self.file_a_terms = self.file_a_terms.saturating_add(a.len() as u64);
            self.file_b_terms = self.file_b_terms.saturating_add(b.len() as u64);
            self.file_c_terms = self.file_c_terms.saturating_add(c.len() as u64);
            return;
        }
        if self.file_sink.is_some() {
            // Stage term bytes/indices and row lengths, then flush in large blocks.
            self.fb_row_lens
                .extend_from_slice(&[(a.len() as u32), (b.len() as u32), (c.len() as u32)]);
            self.fb_stage_bytes = self.fb_stage_bytes.saturating_add(12);

            let a_len = a.len();
            self.fb_a_coeffs.reserve(a_len * 2);
            self.fb_a_idx.reserve(a_len);
            for (coef, idx) in a.into_iter() {
                let vv = self.fb_coeff_u16(coef);
                self.fb_a_coeffs.extend_from_slice(&vv.to_le_bytes());
                self.fb_a_idx.push(self.map_idx_u32(idx));
            }
            self.fb_stage_bytes = self.fb_stage_bytes.saturating_add(a_len * 6);

            let b_len = b.len();
            self.fb_b_coeffs.reserve(b_len * 2);
            self.fb_b_idx.reserve(b_len);
            for (coef, idx) in b.into_iter() {
                let vv = self.fb_coeff_u16(coef);
                self.fb_b_coeffs.extend_from_slice(&vv.to_le_bytes());
                self.fb_b_idx.push(self.map_idx_u32(idx));
            }
            self.fb_stage_bytes = self.fb_stage_bytes.saturating_add(b_len * 6);

            let c_len = c.len();
            self.fb_c_coeffs.reserve(c_len * 2);
            self.fb_c_idx.reserve(c_len);
            for (coef, idx) in c.into_iter() {
                let vv = self.fb_coeff_u16(coef);
                self.fb_c_coeffs.extend_from_slice(&vv.to_le_bytes());
                self.fb_c_idx.push(self.map_idx_u32(idx));
            }
            self.fb_stage_bytes = self.fb_stage_bytes.saturating_add(c_len * 6);

            self.fb_maybe_flush();
        } else {
            let a0 = self.a_terms.len();
            self.a_terms.extend(a);
            let a1 = self.a_terms.len();
            let b0 = self.b_terms.len();
            self.b_terms.extend(b);
            let b1 = self.b_terms.len();
            let c0 = self.c_terms.len();
            self.c_terms.extend(c);
            let c1 = self.c_terms.len();
            self.rows.push(Constraint { a: a0..a1, b: b0..b1, c: c0..c1 });
        }
    }

    fn enforce_lc_times_one_eq_var(&mut self, lc: Vec<(F, usize)>, out: usize) {
        self.add_constraint(lc, vec![(F::ONE, self.one())], vec![(F::ONE, out)]);
    }

    fn enforce_var_eq_const(&mut self, x: usize, c: F) {
        // x * 1 = c
        self.add_constraint(vec![(F::ONE, x)], vec![(F::ONE, self.one())], vec![(c, self.one())]);
    }

    fn enforce_var_eq_var(&mut self, x: usize, y: usize) {
        self.enforce_lc_times_one_eq_var(vec![(F::ONE, x)], y);
    }

    fn enforce_mul(&mut self, x: usize, y: usize, out: usize) {
        self.add_constraint(vec![(F::ONE, x)], vec![(F::ONE, y)], vec![(F::ONE, out)]);
    }

    fn enforce_pow_u64(&mut self, base: usize, alpha: u64) -> usize {
        // Build constraints for base^alpha over variables.
        //
        // IMPORTANT:
        // This gadget is used heavily by Poseidon (S-box exponentiation). For common small exponents,
        // we use short addition chains to minimize multiplication constraints.
        if alpha == 0 {
            return self.new_var(F::ONE);
        }
        if alpha == 1 {
            return base;
        }
        // Common Poseidon alphas.
        // - alpha=3: x^3 = (x^2)*x (2 muls)
        // - alpha=5: x^5 = (x^2)^2 * x (3 muls)
        // - alpha=7: x^7 = (x^2)^2 * x^2 * x (4 muls)
        if alpha == 3 {
            let x2_val = self.assignment[base] * self.assignment[base];
            let x2 = self.new_var(x2_val);
            self.enforce_mul(base, base, x2);
            let x3_val = x2_val * self.assignment[base];
            let x3 = self.new_var(x3_val);
            self.enforce_mul(x2, base, x3);
            return x3;
        }
        if alpha == 5 {
            let x = base;
            let x_val = self.assignment[x];
            let x2_val = x_val * x_val;
            let x2 = self.new_var(x2_val);
            self.enforce_mul(x, x, x2);
            let x4_val = x2_val * x2_val;
            let x4 = self.new_var(x4_val);
            self.enforce_mul(x2, x2, x4);
            let x5_val = x4_val * x_val;
            let x5 = self.new_var(x5_val);
            self.enforce_mul(x4, x, x5);
            return x5;
        }
        if alpha == 7 {
            let x = base;
            let x_val = self.assignment[x];
            let x2_val = x_val * x_val;
            let x2 = self.new_var(x2_val);
            self.enforce_mul(x, x, x2);
            let x4_val = x2_val * x2_val;
            let x4 = self.new_var(x4_val);
            self.enforce_mul(x2, x2, x4);
            let x6_val = x4_val * x2_val;
            let x6 = self.new_var(x6_val);
            self.enforce_mul(x4, x2, x6);
            let x7_val = x6_val * x_val;
            let x7 = self.new_var(x7_val);
            self.enforce_mul(x6, x, x7);
            return x7;
        }

        // current = base
        let mut cur_var = base;
        let mut cur_val = self.assignment[base];

        // acc = 1
        let mut acc_var = self.new_var(F::ONE);
        let mut acc_val = F::ONE;

        let mut e = alpha;
        while e > 0 {
            if (e & 1) == 1 {
                let out_val = acc_val * cur_val;
                let out_var = self.new_var(out_val);
                self.enforce_mul(acc_var, cur_var, out_var);
                acc_var = out_var;
                acc_val = out_val;
            }
            e >>= 1;
            if e == 0 {
                break;
            }
            // square cur
            let sq_val = cur_val * cur_val;
            let sq_var = self.new_var(sq_val);
            self.enforce_mul(cur_var, cur_var, sq_var);
            cur_var = sq_var;
            cur_val = sq_val;
        }

        acc_var
    }

    fn into_sparse_instance(self) -> (SparseDr1csInstance<F>, Vec<F>) {
        if self.file_sink.is_some() {
            panic!("into_sparse_instance called on file-backed Dr1csBuilder; use into_file_backed_instance");
        }
        let nvars = self.assignment.len();
        let inst = SparseDr1csInstance {
            nvars,
            constraints: self.rows,
            a_terms: self.a_terms,
            b_terms: self.b_terms,
            c_terms: self.c_terms,
        };
        (inst, self.assignment)
    }

    fn into_file_backed_instance(self) -> Result<(FileBackedSparseDr1csInstance<F>, Vec<F>), String> {
        let mut me = self;
        me.fb_flush()?;
        let sink = me
            .file_sink
            .take()
            .ok_or_else(|| "into_file_backed_instance called on in-memory Dr1csBuilder".to_string())?;
        let PoseidonFileSink::Append(sink) = sink else {
            return Err("into_file_backed_instance called on range-writer Dr1csBuilder".to_string());
        };
        let inst = sink.finish(me.assignment.len())?;
        Ok((inst, me.assignment))
    }

    fn into_count_result(self) -> Result<(Vec<F>, RangeWriteResult), String> {
        let mut me = self;
        me.fb_flush()?;
        let sink = me
            .file_sink
            .take()
            .ok_or_else(|| "into_count_result called on in-memory Dr1csBuilder".to_string())?;
        let PoseidonFileSink::Count = sink else {
            return Err("into_count_result called on non-count Dr1csBuilder".to_string());
        };
        Ok((
            me.assignment,
            RangeWriteResult {
                ckpts: Vec::new(),
                rows: me.file_rows,
                a_terms: me.file_a_terms,
                b_terms: me.file_b_terms,
                c_terms: me.file_c_terms,
            },
        ))
    }

    #[cfg(unix)]
    fn into_range_result(self) -> Result<(Vec<F>, RangeWriteResult), String> {
        let mut me = self;
        me.fb_flush()?;
        let sink = me
            .file_sink
            .take()
            .ok_or_else(|| "into_range_result called on in-memory Dr1csBuilder".to_string())?;
        let PoseidonFileSink::Range(mut w) = sink else {
            return Err("into_range_result called on append-writer Dr1csBuilder".to_string());
        };
        let ckpts = w.take_ckpts();
        Ok((
            me.assignment,
            RangeWriteResult {
                ckpts,
                rows: me.file_rows,
                a_terms: me.file_a_terms,
                b_terms: me.file_b_terms,
                c_terms: me.file_c_terms,
            },
        ))
    }
}

/// Build a sparse dR1CS instance for a single Poseidon permutation, given an input state.
///
/// Returns `(instance, assignment, out_state_var_indices)`.
pub fn poseidon_permutation_dr1cs<F: PrimeField + CanonicalSerialize>(
    cfg: &ark_crypto_primitives::sponge::poseidon::PoseidonConfig<F>,
    before_state: &[F],
) -> Result<(SparseDr1csInstance<F>, Vec<F>, Vec<usize>), PoseidonReplayError> {
    let t = cfg.rate + cfg.capacity;
    if before_state.len() != t {
        return Err(PoseidonReplayError::Invalid(format!(
            "bad state length: expected {}, got {}",
            t,
            before_state.len()
        )));
    }

    // Compute round states (ground truth for witness materialization).
    let (after_state, round_states) = permute_with_round_trace(cfg, before_state)?;

    let mut b = Dr1csBuilder::<F>::new();

    // Materialize the input state variables.
    let mut state_vars: Vec<usize> = Vec::with_capacity(t);
    for &s in before_state {
        state_vars.push(b.new_var(s));
    }

    let full_rounds_over_2 = cfg.full_rounds / 2;
    let total_rounds = cfg.full_rounds + cfg.partial_rounds;
    assert_eq!(round_states.len(), total_rounds);

    // For each round: ARK -> SBOX -> MDS.
    for r in 0..total_rounds {
        let is_full = r < full_rounds_over_2 || r >= (full_rounds_over_2 + cfg.partial_rounds);

        // ARK (affine shift).
        let mut ark_vars: Vec<usize> = Vec::with_capacity(t);
        for i in 0..t {
            let val = b.assignment[state_vars[i]] + cfg.ark[r][i];
            let v = b.new_var(val);
            // (state_i + ark*r,i*one) * 1 = v
            b.enforce_lc_times_one_eq_var(
                vec![(F::ONE, state_vars[i]), (cfg.ark[r][i], b.one())],
                v,
            );
            ark_vars.push(v);
        }

        // S-box.
        let mut sbox_vars: Vec<usize> = Vec::with_capacity(t);
        for i in 0..t {
            if is_full || i == 0 {
                let out_var = b.enforce_pow_u64(ark_vars[i], cfg.alpha);
                sbox_vars.push(out_var);
            } else {
                // identity: materialize as a fresh var with equality constraint for uniformity
                let v = b.new_var(b.assignment[ark_vars[i]]);
                b.enforce_var_eq_var(ark_vars[i], v);
                sbox_vars.push(v);
            }
        }

        // MDS.
        let mut next_vars: Vec<usize> = Vec::with_capacity(t);
        for i in 0..t {
            let mut lc: Vec<(F, usize)> = Vec::with_capacity(t);
            let mut val = F::ZERO;
            for j in 0..t {
                let coeff = cfg.mds[i][j];
                lc.push((coeff, sbox_vars[j]));
                val += coeff * b.assignment[sbox_vars[j]];
            }
            let v = b.new_var(val);
            b.enforce_lc_times_one_eq_var(lc, v);
            next_vars.push(v);
        }

        // Sanity: witness should match ground-truth round state.
        // (Recorded round state is after ARK+SBOX+MDS).
        let expected = &round_states[r];
        for i in 0..t {
            if b.assignment[next_vars[i]] != expected[i] {
                return Err(PoseidonReplayError::Mismatch(format!(
                    "round {r} state mismatch at i={i}"
                )));
            }
        }

        state_vars = next_vars;
    }

    // Final state must match.
    for i in 0..t {
        if b.assignment[state_vars[i]] != after_state[i] {
            return Err(PoseidonReplayError::Mismatch(format!(
                "final state mismatch at i={i}"
            )));
        }
    }

    let (inst, assignment) = b.into_sparse_instance();
    Ok((inst, assignment, state_vars))
}

/// Byte squeeze witness info (for later byte-decomposition constraints).
#[derive(Clone, Debug)]
pub struct ByteSqueezeWitness {
    pub n: usize,
    pub usable_bytes: usize,
    pub src_elems: Vec<usize>, // variable indices of squeezed field elements
    pub out: Vec<u8>,          // recorded bytes
}

/// Wiring information for a Poseidon transcript dR1CS instance.
///
/// This is intended for *higher-level verifier arithmetizations* that need to reference
/// specific squeezed field elements (e.g., Fiat–Shamir challenges) as variables.
#[derive(Clone, Debug, Default)]
pub struct PoseidonDr1csWiring {
    /// Flattened variable indices for all absorbed field elements (BF variables) in trace order.
    pub absorb_vars: Vec<usize>,
    /// For each `Absorb` op, `(start, len)` into `absorb_vars`.
    pub absorb_ranges: Vec<(usize, usize)>,
    /// Flattened variable indices for all `SqueezeField` outputs in trace order.
    pub squeeze_field_vars: Vec<usize>,
    /// For each `SqueezeField` op, `(start, len)` into `squeeze_field_vars`.
    pub squeeze_field_ranges: Vec<(usize, usize)>,
}

/// Wiring for `SqueezeBytes` outputs.
#[derive(Clone, Debug, Default)]
pub struct PoseidonByteWiring {
    /// Flattened variable indices for all squeezed bytes (BF variables) in trace order.
    pub squeeze_byte_vars: Vec<usize>,
    /// For each `SqueezeBytes` op, `(start, len)` into `squeeze_byte_vars`.
    pub squeeze_byte_ranges: Vec<(usize, usize)>,
}

/// Build a dR1CS instance for the *entire* Poseidon sponge transcript trace, including:
/// - permutation constraints,
/// - absorb updates (linear constraints),
/// - squeeze-field outputs (linear constraints).
///
/// For `SqueezeBytes`, we constrain the squeezed **field elements** and return `ByteSqueezeWitness`
/// entries so byte-decomposition constraints can be added in a later step.
pub fn poseidon_sponge_dr1cs_from_trace<F: PrimeField + CanonicalSerialize>(
    cfg: &ark_crypto_primitives::sponge::poseidon::PoseidonConfig<F>,
    ops: &[PoseidonTraceOp<F>],
) -> Result<
    (
        SparseDr1csInstance<F>,
        Vec<F>,
        PoseidonSpongeReplayResult<F>,
        Vec<ByteSqueezeWitness>,
    ),
    ReplayErr,
> {
    poseidon_sponge_dr1cs_from_trace_with_wiring(cfg, ops)
        .map(|(inst, asg, replay, bytes, _wiring)| (inst, asg, replay, bytes))
}

/// Same as `poseidon_sponge_dr1cs_from_trace`, but also returns `PoseidonDr1csWiring` describing
/// where each `SqueezeField` output element lives in the dR1CS assignment vector.
pub fn poseidon_sponge_dr1cs_from_trace_with_wiring<F: PrimeField + CanonicalSerialize>(
    cfg: &ark_crypto_primitives::sponge::poseidon::PoseidonConfig<F>,
    ops: &[PoseidonTraceOp<F>],
) -> Result<
    (
        SparseDr1csInstance<F>,
        Vec<F>,
        PoseidonSpongeReplayResult<F>,
        Vec<ByteSqueezeWitness>,
        PoseidonDr1csWiring,
    ),
    ReplayErr,
> {
    poseidon_sponge_dr1cs_from_trace_impl(cfg, ops, false, PoseidonArithMode::ReplayFixed).map(
        |(inst, asg, replay, bytes, wiring, _bw)| (inst, asg, replay, bytes, wiring),
    )
}

/// Like `poseidon_sponge_dr1cs_from_trace_with_wiring_and_bytes`, but suitable for
/// **arm-before-proof WE**: it does **not** bake recorded trace IO values into constraints, and
/// it does **not** require replay consistency.
///
/// This makes the dR1CS instance depend only on the operation schedule (lengths) and Poseidon
/// parameters, not on a specific transcript realization.
pub fn poseidon_sponge_dr1cs_from_ops_with_wiring_and_bytes<F: PrimeField + CanonicalSerialize>(
    cfg: &ark_crypto_primitives::sponge::poseidon::PoseidonConfig<F>,
    ops: &[PoseidonTraceOp<F>],
) -> Result<
    (
        SparseDr1csInstance<F>,
        Vec<F>,
        PoseidonSpongeReplayResult<F>,
        Vec<ByteSqueezeWitness>,
        PoseidonDr1csWiring,
        PoseidonByteWiring,
    ),
    ReplayErr,
> {
    poseidon_sponge_dr1cs_from_trace_impl(cfg, ops, true, PoseidonArithMode::WeWitness)
}

/// WE/arm-before-proof mode, but **without** arithmetizing `SqueezeBytes`.
///
/// This is useful for estimating the marginal constraint cost of the `SqueezeBytes` byte
/// canonicalization gadget (which can be large).
pub fn poseidon_sponge_dr1cs_from_ops_with_wiring_no_bytes<F: PrimeField + CanonicalSerialize>(
    cfg: &ark_crypto_primitives::sponge::poseidon::PoseidonConfig<F>,
    ops: &[PoseidonTraceOp<F>],
) -> Result<(SparseDr1csInstance<F>, Vec<F>, PoseidonDr1csWiring), ReplayErr> {
    poseidon_sponge_dr1cs_from_trace_impl(cfg, ops, false, PoseidonArithMode::WeWitness)
        .map(|(inst, asg, _replay, _bytes, wiring, _bw)| (inst, asg, wiring))
}

/// Like `poseidon_sponge_dr1cs_from_trace_with_wiring`, but also **arithmetizes `SqueezeBytes`**:
/// - allocates byte variables,
/// - constrains each byte is 8-bit,
/// - links bytes to the underlying squeezed field elements via radix-256 decomposition.
pub fn poseidon_sponge_dr1cs_from_trace_with_wiring_and_bytes<F: PrimeField + CanonicalSerialize>(
    cfg: &ark_crypto_primitives::sponge::poseidon::PoseidonConfig<F>,
    ops: &[PoseidonTraceOp<F>],
) -> Result<
    (
        SparseDr1csInstance<F>,
        Vec<F>,
        PoseidonSpongeReplayResult<F>,
        Vec<ByteSqueezeWitness>,
        PoseidonDr1csWiring,
        PoseidonByteWiring,
    ),
    ReplayErr,
> {
    poseidon_sponge_dr1cs_from_trace_impl(cfg, ops, true, PoseidonArithMode::ReplayFixed)
}

/// File-backed WE/arm-before-proof mode (with bytes): streams constraints/terms to `out_dir`.
pub fn poseidon_sponge_dr1cs_from_ops_with_wiring_and_bytes_file_backed<F: PrimeField + CanonicalSerialize>(
    cfg: &ark_crypto_primitives::sponge::poseidon::PoseidonConfig<F>,
    ops: &[PoseidonTraceOp<F>],
    out_dir: impl AsRef<Path>,
) -> Result<
    (
        FileBackedSparseDr1csInstance<F>,
        Vec<F>,
        PoseidonSpongeReplayResult<F>,
        Vec<ByteSqueezeWitness>,
        PoseidonDr1csWiring,
        PoseidonByteWiring,
    ),
    ReplayErr,
> {
    // Canonical file-backed WE mode for LF+:
    // - We want to use multiple cores (Poseidon arith is otherwise single-threaded).
    // - For the LF+ tiny-gate pipeline, we do NOT consume Poseidon's `SqueezeBytes` byte outputs,
    //   so we avoid arithmetizing the very large byte-canonicalization gadget.
    //
    // Therefore: always use the sharded Poseidon builder when Rayon has >1 thread, and run it
    // with `with_bytes=false` internally. `PoseidonByteWiring` will remain empty.
    let out_dir = out_dir.as_ref();
    let n_threads = rayon::current_num_threads().max(1);
    if poseidon_profile_on() {
        eprintln!(
            "[poseidon_arith] start ops={} threads={} file_backed=true",
            ops.len(),
            n_threads
        );
    }
    if n_threads > 1 {
        // Sharding is the only practical parallelization strategy for Poseidon arith.
        //
        // Important: too-small shards create many parts and the *merge rewrites the entire DR1CS
        // multiple times*, which can easily reach 100GB+ of IO. Therefore pick shard size so the
        // number of shards is small (≈ O(threads), not O(permutes)).
        let total_permutes = crate::poseidon_trace::count_permutes_for_ops(cfg, ops);
        // More shards => more parallel merge pairs, but also more total IO during merge.
        // Keep this bounded to avoid exploding rewrite volume on large traces.
        let target_shards = n_threads.min(16).max(2);
        let shard_permutes = (total_permutes + target_shards - 1) / target_shards;
        // Keep shards reasonably coarse to avoid hammering the filesystem/page-cache.
        // Empirically this keeps downstream CM phases faster than very small shards.
        let shard_permutes = shard_permutes.max(1024);
        return poseidon_sponge_dr1cs_from_ops_with_wiring_and_bytes_file_backed_sharded(
            cfg,
            ops,
            out_dir,
            shard_permutes,
        );
    }

    let (inst_any, asg, replay, bytes, wiring, bw) =
        poseidon_sponge_dr1cs_from_trace_impl_any(cfg, ops, false, PoseidonArithMode::WeWitness, Some(out_dir))?;
    match inst_any {
        PoseidonInstance::FileBacked(inst) => {
            if poseidon_profile_on() {
                eprintln!(
                    "[poseidon_arith] done nvars={} constraints={}",
                    asg.len(),
                    inst.layout.nconstraints
                );
            }
            Ok((inst, asg, replay, bytes, wiring, bw))
        }
        PoseidonInstance::InMemory(_) => Err(ReplayErr::Invalid("expected file-backed instance, got in-memory".to_string())),
    }
}

/// Count-only sharded WE/arm-before-proof mode **without bytes**.
///
/// This computes the **exact Poseidon dR1CS variable layout** (assignment + wiring) and exact
/// row/term counts, but does **not** write any constraints/term pools to disk.
///
/// This is used as Pass0 for whole-pipeline direct-to-merged builds.
pub fn poseidon_sponge_dr1cs_from_ops_with_wiring_no_bytes_count_sharded<
    F: PrimeField + CanonicalSerialize,
>(
    cfg: &ark_crypto_primitives::sponge::poseidon::PoseidonConfig<F>,
    ops: &[PoseidonTraceOp<F>],
    shard_permutes: usize,
) -> Result<(Vec<F>, PoseidonDr1csWiring, RangeWriteResult), ReplayErr> {
    use ark_crypto_primitives::sponge::DuplexSpongeMode;

    let t = cfg.rate + cfg.capacity;
    if t == 0 {
        return Err(ReplayErr::Invalid("invalid poseidon t=0".to_string()));
    }
    if shard_permutes == 0 {
        return Err(ReplayErr::Invalid("shard_permutes must be >0".to_string()));
    }

    #[derive(Clone)]
    struct ShardPlan<F: PrimeField> {
        start: usize,
        end: usize,
        init_state: Vec<F>,
        init_mode: DuplexSpongeMode,
        constrain_init: bool,
    }

    // Pass 0 (cheap): simulate sponge schedule to pick shard boundaries and initial states.
    let mut plans: Vec<ShardPlan<F>> = Vec::new();
    let mut state: Vec<F> = vec![F::ZERO; t];
    let mut mode: DuplexSpongeMode = DuplexSpongeMode::Absorbing { next_absorb_index: 0 };
    let mut permutes_since_start: usize = 0;
    plans.push(ShardPlan {
        start: 0,
        end: 0,
        init_state: state.clone(),
        init_mode: mode.clone(),
        constrain_init: true,
    });

    #[inline]
    fn permute_counted<F: PrimeField>(
        cfg: &ark_crypto_primitives::sponge::poseidon::PoseidonConfig<F>,
        state: &mut [F],
        permutes_since_start: &mut usize,
    ) {
        permute_in_place(cfg, state);
        *permutes_since_start = permutes_since_start.saturating_add(1);
    }

    for (op_idx, op) in ops.iter().enumerate() {
        match op {
            PoseidonTraceOp::Absorb(elems) => {
                for &e in elems {
                    let mut absorb_index = match mode {
                        DuplexSpongeMode::Absorbing { next_absorb_index } => next_absorb_index,
                        DuplexSpongeMode::Squeezing { .. } => {
                            permute_counted(cfg, &mut state, &mut permutes_since_start);
                            0
                        }
                    };
                    if absorb_index == cfg.rate {
                        permute_counted(cfg, &mut state, &mut permutes_since_start);
                        absorb_index = 0;
                    }
                    let pos = cfg.capacity + absorb_index;
                    state[pos] += e;
                    mode = DuplexSpongeMode::Absorbing {
                        next_absorb_index: absorb_index + 1,
                    };
                }
            }
            PoseidonTraceOp::SqueezeField(out) => {
                if !out.is_empty() {
                    let mut squeeze_index = match mode {
                        DuplexSpongeMode::Absorbing { .. } => {
                            permute_counted(cfg, &mut state, &mut permutes_since_start);
                            0
                        }
                        DuplexSpongeMode::Squeezing { next_squeeze_index } => next_squeeze_index,
                    };
                    if squeeze_index == cfg.rate {
                        permute_counted(cfg, &mut state, &mut permutes_since_start);
                        squeeze_index = 0;
                    }
                    let mut produced = 0usize;
                    while produced < out.len() {
                        let take = core::cmp::min(cfg.rate - squeeze_index, out.len() - produced);
                        produced += take;
                        squeeze_index += take;
                        if produced < out.len() && squeeze_index == cfg.rate {
                            permute_counted(cfg, &mut state, &mut permutes_since_start);
                            squeeze_index = 0;
                        }
                    }
                    mode = DuplexSpongeMode::Squeezing {
                        next_squeeze_index: squeeze_index,
                    };
                }
            }
            PoseidonTraceOp::SqueezeBytes { .. } => {
                return Err(ReplayErr::Invalid(
                    "PoseidonTraceOp::SqueezeBytes not supported in no-bytes count sharded path".to_string(),
                ));
            }
        }

        if permutes_since_start >= shard_permutes && op_idx + 1 < ops.len() {
            plans.last_mut().unwrap().end = op_idx + 1;
            plans.push(ShardPlan {
                start: op_idx + 1,
                end: op_idx + 1,
                init_state: state.clone(),
                init_mode: mode.clone(),
                constrain_init: false,
            });
            permutes_since_start = 0;
        }
    }
    plans.last_mut().unwrap().end = ops.len();
    plans.retain(|p| p.start < p.end);
    if plans.is_empty() {
        return Err(ReplayErr::Invalid("poseidon shard plan produced empty result".to_string()));
    }

    #[derive(Clone)]
    struct ShardMeta<F: PrimeField> {
        wiring: PoseidonDr1csWiring,
        counts: RangeWriteResult,
        asg: Vec<F>,
    }

    // Build each shard in parallel (count-only sink).
    let shard_metas: Vec<ShardMeta<F>> = plans
        .par_iter()
        .map(|p| -> Result<ShardMeta<F>, ReplayErr> {
            let slice = &ops[p.start..p.end];
            let prebuilt = Some(Dr1csBuilder::<F>::new_count_only());
            let (inst_any, asg, _replay, _bytes, wiring, _bw, _init_state_vars, _final_state_vars, counts) =
                poseidon_sponge_dr1cs_from_trace_impl_any_internal(
                    cfg,
                    slice,
                    false,
                    PoseidonArithMode::WeWitness,
                    prebuilt,
                    None,
                    &p.init_state,
                    p.init_mode.clone(),
                    p.constrain_init,
                )?;
            if !matches!(inst_any, PoseidonInstance::InMemory(_)) {
                return Err(ReplayErr::Invalid("expected in-memory placeholder in count-only shard".to_string()));
            }
            let Some(counts) = counts else {
                return Err(ReplayErr::Invalid("count-only shard missing counts payload".to_string()));
            };
            Ok(ShardMeta::<F> {
                wiring,
                counts,
                asg,
            })
        })
        .collect::<Result<Vec<_>, _>>()?;

    let n_shards = shard_metas.len();

    // Merge assignment, stitch wiring, and sum counts in a single pass.
    //
    // This lets us drop each shard assignment immediately after appending its tail, reducing peak RSS.
    let mut merged_asg: Vec<F> = Vec::new();
    merged_asg.push(F::ONE);
    let mut wiring = PoseidonDr1csWiring::default();
    let mut var_tail_off: usize = 0;
    let mut absorb_off: usize = 0;
    let mut squeeze_off: usize = 0;
    let mut tot_rows: u64 = 0;
    let mut tot_a: u64 = 0;
    let mut tot_b: u64 = 0;
    let mut tot_c: u64 = 0;
    for mut sm in shard_metas.into_iter() {
        if sm.asg.is_empty() || sm.asg[0] != F::ONE {
            return Err(ReplayErr::Invalid("count-only shard assignment missing var0=1".to_string()));
        }
        // Stitch wiring by offsetting per-shard var indices.
        let map_var = |idx: usize| -> usize { if idx == 0 { 0 } else { idx + var_tail_off } };
        for &v in &sm.wiring.absorb_vars {
            wiring.absorb_vars.push(map_var(v));
        }
        for &(st, ln) in &sm.wiring.absorb_ranges {
            wiring.absorb_ranges.push((absorb_off + st, ln));
        }
        absorb_off += sm.wiring.absorb_vars.len();
        for &v in &sm.wiring.squeeze_field_vars {
            wiring.squeeze_field_vars.push(map_var(v));
        }
        for &(st, ln) in &sm.wiring.squeeze_field_ranges {
            wiring.squeeze_field_ranges.push((squeeze_off + st, ln));
        }
        squeeze_off += sm.wiring.squeeze_field_vars.len();

        // Counts.
        tot_rows = tot_rows.saturating_add(sm.counts.rows);
        tot_a = tot_a.saturating_add(sm.counts.a_terms);
        tot_b = tot_b.saturating_add(sm.counts.b_terms);
        tot_c = tot_c.saturating_add(sm.counts.c_terms);

        // Append tail vars by move (avoid keeping multiple full copies alive).
        var_tail_off = var_tail_off.saturating_add(sm.asg.len().saturating_sub(1));
        let mut tail = sm.asg.split_off(1);
        merged_asg.append(&mut tail);
    }
    let eq_rows = (n_shards.saturating_sub(1) as u64).saturating_mul(t as u64);
    let counts = RangeWriteResult {
        ckpts: Vec::new(),
        rows: tot_rows.saturating_add(eq_rows),
        a_terms: tot_a.saturating_add(eq_rows),
        b_terms: tot_b.saturating_add(eq_rows),
        c_terms: tot_c.saturating_add(eq_rows),
    };
    Ok((merged_asg, wiring, counts))
}

/// Sharded Poseidon WE/arm-before-proof mode (**no bytes**), writing directly into preallocated
/// merged files via `pwrite` ranges (unix only).
///
/// This is the Pass1 companion to `poseidon_sponge_dr1cs_from_ops_with_wiring_no_bytes_count_sharded`.
/// Callers must:
/// - preallocate the destination files to at least the needed sizes
/// - provide base offsets (in terms/rows) where this Poseidon part begins in the merged instance
///
/// Returns:
/// - Poseidon-local assignment (var0 shared, then tail vars)
/// - stitched Poseidon wiring (in Poseidon-local var indices)
/// - `RangeWriteResult` containing **global** checkpoints and local counts
#[cfg(unix)]
pub fn poseidon_sponge_dr1cs_from_ops_with_wiring_no_bytes_range_sharded_into_files<
    F: PrimeField + CanonicalSerialize,
>(
    cfg: &ark_crypto_primitives::sponge::poseidon::PoseidonConfig<F>,
    ops: &[PoseidonTraceOp<F>],
    shard_permutes: usize,
    out_fc_a: &std::fs::File,
    out_fi_a: &std::fs::File,
    out_fc_b: &std::fs::File,
    out_fi_b: &std::fs::File,
    out_fc_c: &std::fs::File,
    out_fi_c: &std::fs::File,
    out_rows: &std::fs::File,
    base_a_terms: u64,
    base_b_terms: u64,
    base_c_terms: u64,
    base_rows: u64,
    base_var_tail_off: u32,
) -> Result<(Vec<F>, PoseidonDr1csWiring, RangeWriteResult), ReplayErr> {
    use ark_crypto_primitives::sponge::DuplexSpongeMode;

    let t = cfg.rate + cfg.capacity;
    if t == 0 {
        return Err(ReplayErr::Invalid("invalid poseidon t=0".to_string()));
    }
    if shard_permutes == 0 {
        return Err(ReplayErr::Invalid("shard_permutes must be >0".to_string()));
    }

    #[derive(Clone)]
    struct ShardPlan<F: PrimeField> {
        start: usize,
        end: usize,
        init_state: Vec<F>,
        init_mode: DuplexSpongeMode,
        constrain_init: bool,
    }

    // Pass 0a: simulate sponge schedule to pick shard boundaries and initial states.
    let mut plans: Vec<ShardPlan<F>> = Vec::new();
    let mut state: Vec<F> = vec![F::ZERO; t];
    let mut mode: DuplexSpongeMode = DuplexSpongeMode::Absorbing { next_absorb_index: 0 };
    let mut permutes_since_start: usize = 0;
    plans.push(ShardPlan {
        start: 0,
        end: 0,
        init_state: state.clone(),
        init_mode: mode.clone(),
        constrain_init: true,
    });

    #[inline]
    fn permute_counted<F: PrimeField>(
        cfg: &ark_crypto_primitives::sponge::poseidon::PoseidonConfig<F>,
        state: &mut [F],
        permutes_since_start: &mut usize,
    ) {
        permute_in_place(cfg, state);
        *permutes_since_start = permutes_since_start.saturating_add(1);
    }

    for (op_idx, op) in ops.iter().enumerate() {
        match op {
            PoseidonTraceOp::Absorb(elems) => {
                for &e in elems {
                    let mut absorb_index = match mode {
                        DuplexSpongeMode::Absorbing { next_absorb_index } => next_absorb_index,
                        DuplexSpongeMode::Squeezing { .. } => {
                            permute_counted(cfg, &mut state, &mut permutes_since_start);
                            0
                        }
                    };
                    if absorb_index == cfg.rate {
                        permute_counted(cfg, &mut state, &mut permutes_since_start);
                        absorb_index = 0;
                    }
                    let pos = cfg.capacity + absorb_index;
                    state[pos] += e;
                    mode = DuplexSpongeMode::Absorbing {
                        next_absorb_index: absorb_index + 1,
                    };
                }
            }
            PoseidonTraceOp::SqueezeField(out) => {
                if !out.is_empty() {
                    let mut squeeze_index = match mode {
                        DuplexSpongeMode::Absorbing { .. } => {
                            permute_counted(cfg, &mut state, &mut permutes_since_start);
                            0
                        }
                        DuplexSpongeMode::Squeezing { next_squeeze_index } => next_squeeze_index,
                    };
                    if squeeze_index == cfg.rate {
                        permute_counted(cfg, &mut state, &mut permutes_since_start);
                        squeeze_index = 0;
                    }
                    let mut produced = 0usize;
                    while produced < out.len() {
                        let take = core::cmp::min(cfg.rate - squeeze_index, out.len() - produced);
                        produced += take;
                        squeeze_index += take;
                        if produced < out.len() && squeeze_index == cfg.rate {
                            permute_counted(cfg, &mut state, &mut permutes_since_start);
                            squeeze_index = 0;
                        }
                    }
                    mode = DuplexSpongeMode::Squeezing {
                        next_squeeze_index: squeeze_index,
                    };
                }
            }
            PoseidonTraceOp::SqueezeBytes { .. } => {
                return Err(ReplayErr::Invalid(
                    "PoseidonTraceOp::SqueezeBytes is not supported in the sharded no-bytes range-write path".to_string(),
                ));
            }
        }

        if permutes_since_start >= shard_permutes && op_idx + 1 < ops.len() {
            plans.last_mut().unwrap().end = op_idx + 1;
            plans.push(ShardPlan {
                start: op_idx + 1,
                end: op_idx + 1,
                init_state: state.clone(),
                init_mode: mode.clone(),
                constrain_init: false,
            });
            permutes_since_start = 0;
        }
    }
    plans.last_mut().unwrap().end = ops.len();
    plans.retain(|p| p.start < p.end);
    if plans.is_empty() {
        return Err(ReplayErr::Invalid("poseidon shard plan produced empty result".to_string()));
    }

    // Pass 0b (structural): count each shard cheaply to compute exact disjoint write ranges.
    //
    // IMPORTANT: this must exactly match the WE/no-bytes arithmetizer shape, including absorb
    // state-update constraints and the optimized full/partial round gadgets.
    let total_rounds = cfg.full_rounds + cfg.partial_rounds;
    let mut ark_is_zero: Vec<Vec<bool>> = vec![vec![false; t]; total_rounds];
    for r in 0..total_rounds {
        for i in 0..t {
            ark_is_zero[r][i] = cfg.ark[r][i].is_zero();
        }
    }
    let mut partial_const_is_zero: Vec<Vec<bool>> = vec![vec![true; t]; total_rounds];
    let full_rounds_over_2 = cfg.full_rounds / 2;
    for r in 0..total_rounds {
        let is_full = r < full_rounds_over_2 || r >= (full_rounds_over_2 + cfg.partial_rounds);
        if is_full {
            continue;
        }
        for i in 0..t {
            let mut const_term = F::ZERO;
            for j in 1..t {
                const_term += cfg.mds[i][j] * cfg.ark[r][j];
            }
            partial_const_is_zero[r][i] = const_term.is_zero();
        }
    }

    #[derive(Clone, Copy, Debug, Default)]
    struct ShardCounts {
        nvars: u64,
        nconstraints: u64,
        a_terms: u64,
        b_terms: u64,
        c_terms: u64,
    }

    #[derive(Clone, Debug)]
    struct PoseidonShardCounter<F: PrimeField> {
        // var0 is the constant-1 slot
        nvars: usize,
        counts: ShardCounts,
        mode: DuplexSpongeMode,
        state_vars: Vec<usize>,
        init_state_vars: Vec<usize>,
        _pd: core::marker::PhantomData<F>,
    }

    impl<F: PrimeField> PoseidonShardCounter<F> {
        fn new(t: usize, init_mode: DuplexSpongeMode, constrain_init: bool) -> Self {
            let mut s = Self {
                nvars: 1, // var0
                counts: ShardCounts::default(),
                mode: init_mode,
                state_vars: Vec::with_capacity(t),
                init_state_vars: Vec::with_capacity(t),
                _pd: core::marker::PhantomData,
            };
            for _ in 0..t {
                let v = s.new_var();
                if constrain_init {
                    s.enforce_var_eq_const(v);
                }
                s.state_vars.push(v);
                s.init_state_vars.push(v);
            }
            s
        }
        #[inline]
        fn new_var(&mut self) -> usize {
            let v = self.nvars;
            self.nvars += 1;
            self.counts.nvars = self.nvars as u64;
            v
        }
        #[inline]
        fn add_constraint_lens(&mut self, a_len: usize, b_len: usize, c_len: usize) {
            self.counts.nconstraints = self.counts.nconstraints.saturating_add(1);
            self.counts.a_terms = self.counts.a_terms.saturating_add(a_len as u64);
            self.counts.b_terms = self.counts.b_terms.saturating_add(b_len as u64);
            self.counts.c_terms = self.counts.c_terms.saturating_add(c_len as u64);
        }
        #[inline]
        fn enforce_mul(&mut self, _x: usize, _y: usize, _out: usize) {
            self.add_constraint_lens(1, 1, 1);
        }
        #[inline]
        fn enforce_lc_times_one_eq_var(&mut self, lc_len: usize, _out: usize) {
            self.add_constraint_lens(lc_len, 1, 1);
        }
        #[inline]
        fn enforce_var_eq_const(&mut self, _x: usize) {
            self.add_constraint_lens(1, 1, 1);
        }
        fn enforce_pow_u64(&mut self, base: usize, alpha: u64) -> usize {
            if alpha == 0 {
                return self.new_var();
            }
            if alpha == 1 {
                return base;
            }
            if alpha == 3 {
                let x2 = self.new_var();
                self.enforce_mul(base, base, x2);
                let x3 = self.new_var();
                self.enforce_mul(x2, base, x3);
                return x3;
            }
            if alpha == 5 {
                let x2 = self.new_var();
                self.enforce_mul(base, base, x2);
                let x4 = self.new_var();
                self.enforce_mul(x2, x2, x4);
                let x5 = self.new_var();
                self.enforce_mul(x4, base, x5);
                return x5;
            }
            if alpha == 7 {
                let x2 = self.new_var();
                self.enforce_mul(base, base, x2);
                let x4 = self.new_var();
                self.enforce_mul(x2, x2, x4);
                let x6 = self.new_var();
                self.enforce_mul(x4, x2, x6);
                let x7 = self.new_var();
                self.enforce_mul(x6, base, x7);
                return x7;
            }
            // Fallback: square-and-multiply (shape only).
            let mut cur_var = base;
            let mut acc_var = self.new_var(); // acc = 1
            let mut e = alpha;
            while e > 0 {
                if (e & 1) == 1 {
                    let out_var = self.new_var();
                    self.enforce_mul(acc_var, cur_var, out_var);
                    acc_var = out_var;
                }
                e >>= 1;
                if e == 0 {
                    break;
                }
                let sq_var = self.new_var();
                self.enforce_mul(cur_var, cur_var, sq_var);
                cur_var = sq_var;
            }
            acc_var
        }
        fn apply_perm_counts(
            &mut self,
            cfg: &ark_crypto_primitives::sponge::poseidon::PoseidonConfig<F>,
            ark_is_zero: &[Vec<bool>],
            partial_const_is_zero: &[Vec<bool>],
        ) {
            let t = cfg.rate + cfg.capacity;
            let full_rounds_over_2 = cfg.full_rounds / 2;
            let total_rounds = cfg.full_rounds + cfg.partial_rounds;
            for r in 0..total_rounds {
                let is_full = r < full_rounds_over_2 || r >= (full_rounds_over_2 + cfg.partial_rounds);
                let mut next_vars: Vec<usize> = Vec::with_capacity(t);
                if is_full {
                    let mut sbox_vars: Vec<usize> = Vec::with_capacity(t);
                    for i in 0..t {
                        let lc_len = if ark_is_zero[r][i] { 1 } else { 2 };
                        let out = match cfg.alpha {
                            3 => {
                                let x2 = self.new_var();
                                self.add_constraint_lens(lc_len, lc_len, 1);
                                let x3 = self.new_var();
                                self.add_constraint_lens(1, lc_len, 1);
                                let _ = x2;
                                x3
                            }
                            5 => {
                                let x2 = self.new_var();
                                self.add_constraint_lens(lc_len, lc_len, 1);
                                let x4 = self.new_var();
                                self.enforce_mul(x2, x2, x4);
                                let x5 = self.new_var();
                                self.add_constraint_lens(1, lc_len, 1);
                                let _ = (x2, x4);
                                x5
                            }
                            7 => {
                                let x2 = self.new_var();
                                self.add_constraint_lens(lc_len, lc_len, 1);
                                let x4 = self.new_var();
                                self.enforce_mul(x2, x2, x4);
                                let x6 = self.new_var();
                                self.enforce_mul(x4, x2, x6);
                                let x7 = self.new_var();
                                self.add_constraint_lens(1, lc_len, 1);
                                let _ = (x2, x4, x6);
                                x7
                            }
                            _ => {
                                let v = self.new_var();
                                self.enforce_lc_times_one_eq_var(2, v);
                                self.enforce_pow_u64(v, cfg.alpha)
                            }
                        };
                        sbox_vars.push(out);
                    }
                    for _i in 0..t {
                        let v = self.new_var();
                        self.enforce_lc_times_one_eq_var(t, v);
                        next_vars.push(v);
                    }
                    let _ = sbox_vars;
                } else {
                    let ark0 = self.new_var();
                    self.enforce_lc_times_one_eq_var(2, ark0);
                    let sbox0 = self.enforce_pow_u64(ark0, cfg.alpha);
                    for i in 0..t {
                        let mut lc_len = 1 + (t - 1);
                        if !partial_const_is_zero[r][i] {
                            lc_len += 1;
                        }
                        let v = self.new_var();
                        self.enforce_lc_times_one_eq_var(lc_len, v);
                        next_vars.push(v);
                    }
                    let _ = sbox0;
                }
                self.state_vars = next_vars;
            }
        }
    }

    fn count_shard<F: PrimeField>(
        cfg: &ark_crypto_primitives::sponge::poseidon::PoseidonConfig<F>,
        ops: &[PoseidonTraceOp<F>],
        init_mode: DuplexSpongeMode,
        constrain_init: bool,
        ark_is_zero: &[Vec<bool>],
        partial_const_is_zero: &[Vec<bool>],
    ) -> ShardCounts {
        let t = cfg.rate + cfg.capacity;
        let mut c = PoseidonShardCounter::<F>::new(t, init_mode, constrain_init);
        for op in ops {
            match op {
                PoseidonTraceOp::Absorb(elems) => {
                    if elems.is_empty() {
                        continue;
                    }
                    for _e in elems {
                        if matches!(c.mode, DuplexSpongeMode::Squeezing { .. }) {
                            c.apply_perm_counts(cfg, ark_is_zero, partial_const_is_zero);
                            c.mode = DuplexSpongeMode::Absorbing { next_absorb_index: 0 };
                        }
                        let mut absorb_index = match c.mode {
                            DuplexSpongeMode::Absorbing { next_absorb_index } => next_absorb_index,
                            _ => unreachable!(),
                        };
                        if absorb_index == cfg.rate {
                            c.apply_perm_counts(cfg, ark_is_zero, partial_const_is_zero);
                            absorb_index = 0;
                        }
                        // e_var
                        let _e_var = c.new_var();
                        // new state slot var + constraint: state[pos] + e == new
                        let new_state = c.new_var();
                        c.enforce_lc_times_one_eq_var(2, new_state);
                        let pos = cfg.capacity + absorb_index;
                        c.state_vars[pos] = new_state;
                        c.mode = DuplexSpongeMode::Absorbing {
                            next_absorb_index: absorb_index + 1,
                        };
                    }
                }
                PoseidonTraceOp::SqueezeField(out) => {
                    if out.is_empty() {
                        continue;
                    }
                    let mut squeeze_index = match c.mode {
                        DuplexSpongeMode::Absorbing { .. } => {
                            c.apply_perm_counts(cfg, ark_is_zero, partial_const_is_zero);
                            0
                        }
                        DuplexSpongeMode::Squeezing { next_squeeze_index } => next_squeeze_index,
                    };
                    if squeeze_index == cfg.rate {
                        c.apply_perm_counts(cfg, ark_is_zero, partial_const_is_zero);
                        squeeze_index = 0;
                    }
                    let mut produced = 0usize;
                    while produced < out.len() {
                        let take = core::cmp::min(cfg.rate - squeeze_index, out.len() - produced);
                        produced += take;
                        squeeze_index += take;
                        if produced < out.len() && squeeze_index == cfg.rate {
                            c.apply_perm_counts(cfg, ark_is_zero, partial_const_is_zero);
                            squeeze_index = 0;
                        }
                    }
                    c.mode = DuplexSpongeMode::Squeezing {
                        next_squeeze_index: squeeze_index,
                    };
                }
                PoseidonTraceOp::SqueezeBytes { .. } => unreachable!(),
            }
        }
        c.counts
    }

    let shard_counts: Vec<ShardCounts> = plans
        .iter()
        .map(|p| count_shard::<F>(
            cfg,
            &ops[p.start..p.end],
            p.init_mode.clone(),
            p.constrain_init,
            &ark_is_zero,
            &partial_const_is_zero,
        ))
        .collect();

    // Prefix sums for shard var/row/term offsets (relative to part start).
    let mut var_tail_off: Vec<u32> = Vec::with_capacity(plans.len());
    let mut row_off: Vec<u64> = Vec::with_capacity(plans.len());
    let mut a_off: Vec<u64> = Vec::with_capacity(plans.len());
    let mut b_off: Vec<u64> = Vec::with_capacity(plans.len());
    let mut c_off: Vec<u64> = Vec::with_capacity(plans.len());
    let mut cur_var_tail: u64 = 0;
    let mut cur_rows: u64 = 0;
    let mut cur_a: u64 = 0;
    let mut cur_b: u64 = 0;
    let mut cur_c: u64 = 0;
    for sc in &shard_counts {
        var_tail_off.push(
            (cur_var_tail + (base_var_tail_off as u64))
                .try_into()
                .map_err(|_| ReplayErr::Invalid("poseidon range: var_tail_off overflow u32".to_string()))?,
        );
        row_off.push(base_rows.saturating_add(cur_rows));
        a_off.push(base_a_terms.saturating_add(cur_a));
        b_off.push(base_b_terms.saturating_add(cur_b));
        c_off.push(base_c_terms.saturating_add(cur_c));
        cur_var_tail = cur_var_tail.saturating_add(sc.nvars.saturating_sub(1));
        cur_rows = cur_rows.saturating_add(sc.nconstraints);
        cur_a = cur_a.saturating_add(sc.a_terms);
        cur_b = cur_b.saturating_add(sc.b_terms);
        cur_c = cur_c.saturating_add(sc.c_terms);
    }

    // Boundary equality constraints: t per boundary, each adds 1 term to A/B/C and 1 row.
    let n_boundaries = plans.len().saturating_sub(1);
    let eq_rows: u64 = (n_boundaries as u64).saturating_mul(t as u64);
    let total_rows = cur_rows.saturating_add(eq_rows);
    let total_a_terms = cur_a.saturating_add(eq_rows);
    let total_b_terms = cur_b.saturating_add(eq_rows);
    let total_c_terms = cur_c.saturating_add(eq_rows);

    // Pass 1: build shards in parallel, writing directly into provided merged files.
    #[derive(Clone)]
    struct ShardBuilt<F: PrimeField> {
        asg: Vec<F>,
        init_state_vars: Vec<usize>,
        final_state_vars: Vec<usize>,
        wiring: PoseidonDr1csWiring,
        ckpts: Vec<(u64, u64, u64, u64)>,
    }
    let mut shard_built: Vec<ShardBuilt<F>> = plans
        .par_iter()
        .enumerate()
        .map(|(shard_idx, p)| -> Result<ShardBuilt<F>, ReplayErr> {
            let slice = &ops[p.start..p.end];
            let writer = FileBackedRangeWriter::new(
                out_fc_a
                    .try_clone()
                    .map_err(|e| ReplayErr::Invalid(format!("clone a_coeffs failed: {e}")))?,
                out_fi_a
                    .try_clone()
                    .map_err(|e| ReplayErr::Invalid(format!("clone a_idx failed: {e}")))?,
                out_fc_b
                    .try_clone()
                    .map_err(|e| ReplayErr::Invalid(format!("clone b_coeffs failed: {e}")))?,
                out_fi_b
                    .try_clone()
                    .map_err(|e| ReplayErr::Invalid(format!("clone b_idx failed: {e}")))?,
                out_fc_c
                    .try_clone()
                    .map_err(|e| ReplayErr::Invalid(format!("clone c_coeffs failed: {e}")))?,
                out_fi_c
                    .try_clone()
                    .map_err(|e| ReplayErr::Invalid(format!("clone c_idx failed: {e}")))?,
                out_rows
                    .try_clone()
                    .map_err(|e| ReplayErr::Invalid(format!("clone constraints failed: {e}")))?,
                a_off[shard_idx],
                b_off[shard_idx],
                c_off[shard_idx],
                row_off[shard_idx],
            );
            let prebuilt = Some(Dr1csBuilder::<F>::new_file_backed_range(writer, var_tail_off[shard_idx]));
            let (inst_any, asg, _replay, _bytes, wiring, _bw, init_state_vars, final_state_vars, ckpts) =
                poseidon_sponge_dr1cs_from_trace_impl_any_internal(
                    cfg,
                    slice,
                    false,
                    PoseidonArithMode::WeWitness,
                    prebuilt,
                    None,
                    &p.init_state,
                    p.init_mode.clone(),
                    p.constrain_init,
                )?;
            if !matches!(inst_any, PoseidonInstance::InMemory(_)) {
                return Err(ReplayErr::Invalid("expected range-writer build to return placeholder instance".to_string()));
            }
            let Some(range_res) = ckpts else {
                return Err(ReplayErr::Invalid("range-writer build missing checkpoint payload".to_string()));
            };
            let exp = shard_counts
                .get(shard_idx)
                .ok_or_else(|| ReplayErr::Invalid("shard_counts index OOB".to_string()))?;
            if range_res.rows != exp.nconstraints
                || range_res.a_terms != exp.a_terms
                || range_res.b_terms != exp.b_terms
                || range_res.c_terms != exp.c_terms
            {
                return Err(ReplayErr::Invalid(format!(
                    "poseidon range write: count mismatch for shard {shard_idx}: \
rows got {} exp {}, a_terms got {} exp {}, b_terms got {} exp {}, c_terms got {} exp {}",
                    range_res.rows,
                    exp.nconstraints,
                    range_res.a_terms,
                    exp.a_terms,
                    range_res.b_terms,
                    exp.b_terms,
                    range_res.c_terms,
                    exp.c_terms
                )));
            }
            Ok(ShardBuilt::<F> {
                asg,
                init_state_vars,
                final_state_vars,
                wiring,
                ckpts: range_res.ckpts,
            })
        })
        .collect::<Result<Vec<_>, _>>()?;

    // Merge assignment by concatenating shard assignments (share var0).
    let mut merged_asg: Vec<F> = Vec::new();
    merged_asg.push(F::ONE);
    let mut tail_lens: Vec<usize> = Vec::with_capacity(shard_built.len());
    for sb in shard_built.iter_mut() {
        if sb.asg.is_empty() || sb.asg[0] != F::ONE {
            return Err(ReplayErr::Invalid("shard assignment missing var0=1".to_string()));
        }
        let tail_len = sb.asg.len().saturating_sub(1);
        tail_lens.push(tail_len);
        let mut asg = core::mem::take(&mut sb.asg);
        let mut tail = asg.split_off(1);
        merged_asg.append(&mut tail);
    }

    // Stitch wiring by offsetting per-shard var indices (Poseidon-local).
    let mut wiring = PoseidonDr1csWiring::default();
    let mut var_tail_local: usize = 0;
    let mut absorb_off: usize = 0;
    let mut squeeze_off: usize = 0;
    for (i, sb) in shard_built.iter().enumerate() {
        let map_var = |idx: usize| -> usize { if idx == 0 { 0 } else { idx + var_tail_local } };
        for &v in &sb.wiring.absorb_vars {
            wiring.absorb_vars.push(map_var(v));
        }
        for &(st, ln) in &sb.wiring.absorb_ranges {
            wiring.absorb_ranges.push((absorb_off + st, ln));
        }
        absorb_off += sb.wiring.absorb_vars.len();
        for &v in &sb.wiring.squeeze_field_vars {
            wiring.squeeze_field_vars.push(map_var(v));
        }
        for &(st, ln) in &sb.wiring.squeeze_field_ranges {
            wiring.squeeze_field_ranges.push((squeeze_off + st, ln));
        }
        squeeze_off += sb.wiring.squeeze_field_vars.len();
        var_tail_local = var_tail_local.saturating_add(*tail_lens.get(i).unwrap_or(&0));
    }

    // Boundary equalities: final_state(shard i) == init_state(shard i+1) (in merged-file var space).
    let map_var_global = |shard_idx: usize, local: usize| -> u32 {
        if local == 0 {
            0
        } else {
            (local as u64)
                .saturating_add(var_tail_off[shard_idx] as u64)
                .try_into()
                .unwrap()
        }
    };
    let mut boundary_eqs: Vec<(u32, u32)> = Vec::new();
    for i in 0..shard_built.len().saturating_sub(1) {
        let left = &shard_built[i];
        let right = &shard_built[i + 1];
        for (x, y) in left.final_state_vars.iter().zip(right.init_state_vars.iter()) {
            boundary_eqs.push((map_var_global(i, *x), map_var_global(i + 1, *y)));
        }
    }

    // Append boundary equalities into the reserved tail range.
    let mut eq_writer = FileBackedRangeWriter::new(
        out_fc_a
            .try_clone()
            .map_err(|e| ReplayErr::Invalid(format!("clone a_coeffs failed: {e}")))?,
        out_fi_a
            .try_clone()
            .map_err(|e| ReplayErr::Invalid(format!("clone a_idx failed: {e}")))?,
        out_fc_b
            .try_clone()
            .map_err(|e| ReplayErr::Invalid(format!("clone b_coeffs failed: {e}")))?,
        out_fi_b
            .try_clone()
            .map_err(|e| ReplayErr::Invalid(format!("clone b_idx failed: {e}")))?,
        out_fc_c
            .try_clone()
            .map_err(|e| ReplayErr::Invalid(format!("clone c_coeffs failed: {e}")))?,
        out_fi_c
            .try_clone()
            .map_err(|e| ReplayErr::Invalid(format!("clone c_idx failed: {e}")))?,
        out_rows
            .try_clone()
            .map_err(|e| ReplayErr::Invalid(format!("clone constraints failed: {e}")))?,
        base_a_terms.saturating_add(cur_a),
        base_b_terms.saturating_add(cur_b),
        base_c_terms.saturating_add(cur_c),
        base_rows.saturating_add(cur_rows),
    );
    {
        // each boundary eq is x*1==y (A:1,B:1,C:1) with tiny_u16_u32 coeff=1
        let one_u16: u16 = 1;
        let a_coeffs: Vec<u8> = (0..boundary_eqs.len())
            .flat_map(|_| one_u16.to_le_bytes())
            .collect();
        let b_coeffs = a_coeffs.clone();
        let c_coeffs = a_coeffs.clone();
        let a_idx: Vec<u32> = boundary_eqs.iter().map(|(x, _)| *x).collect();
        let b_idx: Vec<u32> = vec![0u32; boundary_eqs.len()];
        let c_idx: Vec<u32> = boundary_eqs.iter().map(|(_, y)| *y).collect();
        let lens: Vec<u32> = (0..boundary_eqs.len()).flat_map(|_| [1u32, 1u32, 1u32]).collect();
        eq_writer.push_a_terms_raw_block(&a_coeffs, &a_idx).map_err(ReplayErr::Invalid)?;
        eq_writer.push_b_terms_raw_block(&b_coeffs, &b_idx).map_err(ReplayErr::Invalid)?;
        eq_writer.push_c_terms_raw_block(&c_coeffs, &c_idx).map_err(ReplayErr::Invalid)?;
        eq_writer.push_constraint_lens_block(&lens).map_err(ReplayErr::Invalid)?;
    }
    if (boundary_eqs.len() as u64) != eq_rows {
        return Err(ReplayErr::Invalid(format!(
            "poseidon range write: boundary eq count mismatch (got {}, expected {})",
            boundary_eqs.len(),
            eq_rows
        )));
    }

    // Gather checkpoints from shards + boundary eq tail.
    let mut ckpts_all: Vec<(u64, u64, u64, u64)> = Vec::new();
    for sb in &shard_built {
        ckpts_all.extend_from_slice(&sb.ckpts);
    }
    ckpts_all.extend(eq_writer.take_ckpts().into_iter());
    ckpts_all.sort_by_key(|(row_idx, _, _, _)| *row_idx);
    ckpts_all.dedup_by_key(|(row_idx, _, _, _)| *row_idx);
    // Ensure a checkpoint at the start of this part.
    if ckpts_all.first().map(|x| x.0) != Some(base_rows) {
        ckpts_all.insert(0, (base_rows, base_a_terms, base_b_terms, base_c_terms));
    }

    Ok((
        merged_asg,
        wiring,
        RangeWriteResult {
            ckpts: ckpts_all,
            rows: total_rows,
            a_terms: total_a_terms,
            b_terms: total_b_terms,
            c_terms: total_c_terms,
        },
    ))
}

fn poseidon_sponge_dr1cs_from_ops_with_wiring_and_bytes_file_backed_sharded<F: PrimeField + CanonicalSerialize>(
    cfg: &ark_crypto_primitives::sponge::poseidon::PoseidonConfig<F>,
    ops: &[PoseidonTraceOp<F>],
    out_dir: impl AsRef<Path>,
    shard_permutes: usize,
) -> Result<
    (
        FileBackedSparseDr1csInstance<F>,
        Vec<F>,
        PoseidonSpongeReplayResult<F>,
        Vec<ByteSqueezeWitness>,
        PoseidonDr1csWiring,
        PoseidonByteWiring,
    ),
    ReplayErr,
> {
    use ark_crypto_primitives::sponge::DuplexSpongeMode;
    use std::fs::create_dir_all;
    use std::fs::OpenOptions;
    use std::io::{Seek, SeekFrom};
    use std::io::Write;

    #[derive(Clone, Copy, Debug, Default)]
    struct ShardCounts {
        nvars: u64,
        nconstraints: u64,
        a_terms: u64,
        b_terms: u64,
        c_terms: u64,
    }

    /// Count-only builder for Poseidon arithmetization.
    ///
    /// This mirrors the constraint *shape* emitted by `poseidon_sponge_dr1cs_from_trace_impl_any_internal`
    /// in `PoseidonArithMode::WeWitness` with `with_bytes=false`:
    /// - no replay checks
    /// - no byte-canonicalization gadget
    ///
    /// IMPORTANT: counts must be exact; they are used to pre-size files and compute disjoint write ranges.
    #[derive(Clone, Debug)]
    struct PoseidonShardCounter<F: PrimeField> {
        // var0 is the constant-1 slot
        nvars: usize,
        counts: ShardCounts,
        // current sponge mode and state var indices (local)
        mode: DuplexSpongeMode,
        state_vars: Vec<usize>,
        // record the first t state vars, and the final state vars at end
        init_state_vars: Vec<usize>,
        _pd: core::marker::PhantomData<F>,
    }

    impl<F: PrimeField> PoseidonShardCounter<F> {
        fn new(t: usize, init_mode: DuplexSpongeMode, constrain_init: bool) -> Self {
            let mut s = Self {
                nvars: 1, // var0
                counts: ShardCounts::default(),
                mode: init_mode,
                state_vars: Vec::with_capacity(t),
                init_state_vars: Vec::with_capacity(t),
                _pd: core::marker::PhantomData,
            };
            // Allocate initial state variables (t vars).
            for _ in 0..t {
                let v = s.new_var();
                // In the real builder, constrain_init adds an equality-to-const constraint here.
                if constrain_init {
                    s.enforce_var_eq_const(v);
                }
                s.state_vars.push(v);
                s.init_state_vars.push(v);
            }
            s
        }

        #[inline]
        fn new_var(&mut self) -> usize {
            let v = self.nvars;
            self.nvars += 1;
            self.counts.nvars = self.nvars as u64;
            v
        }

        #[inline]
        fn add_constraint_lens(&mut self, a_len: usize, b_len: usize, c_len: usize) {
            self.counts.nconstraints = self.counts.nconstraints.saturating_add(1);
            self.counts.a_terms = self.counts.a_terms.saturating_add(a_len as u64);
            self.counts.b_terms = self.counts.b_terms.saturating_add(b_len as u64);
            self.counts.c_terms = self.counts.c_terms.saturating_add(c_len as u64);
        }

        // Constraint helpers (shape-only).
        #[inline]
        fn enforce_mul(&mut self, _x: usize, _y: usize, _out: usize) {
            // (1*x) * (1*y) = (1*out)
            self.add_constraint_lens(1, 1, 1);
        }
        #[inline]
        fn enforce_lc_times_one_eq_var(&mut self, lc_len: usize, _out: usize) {
            // lc * 1 = out
            self.add_constraint_lens(lc_len, 1, 1);
        }
        #[inline]
        fn enforce_var_eq_const(&mut self, _x: usize) {
            // x * 1 = c  (C is a const term on var0)
            self.add_constraint_lens(1, 1, 1);
        }

        fn enforce_pow_u64(&mut self, base: usize, alpha: u64) -> usize {
            if alpha == 0 {
                return self.new_var();
            }
            if alpha == 1 {
                return base;
            }
            if alpha == 3 {
                let x2 = self.new_var();
                self.enforce_mul(base, base, x2);
                let x3 = self.new_var();
                self.enforce_mul(x2, base, x3);
                return x3;
            }
            if alpha == 5 {
                let x2 = self.new_var();
                self.enforce_mul(base, base, x2);
                let x4 = self.new_var();
                self.enforce_mul(x2, x2, x4);
                let x5 = self.new_var();
                self.enforce_mul(x4, base, x5);
                return x5;
            }
            if alpha == 7 {
                let x2 = self.new_var();
                self.enforce_mul(base, base, x2);
                let x4 = self.new_var();
                self.enforce_mul(x2, x2, x4);
                let x6 = self.new_var();
                self.enforce_mul(x4, x2, x6);
                let x7 = self.new_var();
                self.enforce_mul(x6, base, x7);
                return x7;
            }

            // Generic square-and-multiply as in the builder.
            let mut cur_var = base;
            let mut acc_var = self.new_var(); // acc = 1
            let mut e = alpha;
            while e > 0 {
                if (e & 1) == 1 {
                    let out_var = self.new_var();
                    self.enforce_mul(acc_var, cur_var, out_var);
                    acc_var = out_var;
                }
                e >>= 1;
                if e == 0 {
                    break;
                }
                let sq_var = self.new_var();
                self.enforce_mul(cur_var, cur_var, sq_var);
                cur_var = sq_var;
            }
            acc_var
        }

        fn apply_perm_counts(
            &mut self,
            cfg: &ark_crypto_primitives::sponge::poseidon::PoseidonConfig<F>,
            ark_is_zero: &[Vec<bool>],
            partial_const_is_zero: &[Vec<bool>],
        ) {
            let t = cfg.rate + cfg.capacity;
            let full_rounds_over_2 = cfg.full_rounds / 2;
            let total_rounds = cfg.full_rounds + cfg.partial_rounds;
            for r in 0..total_rounds {
                let is_full = r < full_rounds_over_2 || r >= (full_rounds_over_2 + cfg.partial_rounds);
                let mut next_vars: Vec<usize> = Vec::with_capacity(t);
                if is_full {
                    // Full rounds: S-box all lanes.
                    let mut sbox_vars: Vec<usize> = Vec::with_capacity(t);
                    for i in 0..t {
                        // lc_in length depends on ark[r][i] being zero in the optimized code-path.
                        let lc_len = if ark_is_zero[r][i] { 1 } else { 2 };
                        let out = match cfg.alpha {
                            3 => {
                                let x2 = self.new_var();
                                // enforce_mul_lc_lc(lc_in, lc_in): lens (lc_len, lc_len, 1)
                                self.add_constraint_lens(lc_len, lc_len, 1);
                                let x3 = self.new_var();
                                // enforce_mul_var_lc(x2, lc_in): lens (1, lc_len, 1)
                                self.add_constraint_lens(1, lc_len, 1);
                                let _ = x2;
                                x3
                            }
                            5 => {
                                let x2 = self.new_var();
                                self.add_constraint_lens(lc_len, lc_len, 1);
                                let x4 = self.new_var();
                                self.enforce_mul(x2, x2, x4);
                                let x5 = self.new_var();
                                self.add_constraint_lens(1, lc_len, 1);
                                let _ = (x2, x4);
                                x5
                            }
                            7 => {
                                let x2 = self.new_var();
                                self.add_constraint_lens(lc_len, lc_len, 1);
                                let x4 = self.new_var();
                                self.enforce_mul(x2, x2, x4);
                                let x6 = self.new_var();
                                self.enforce_mul(x4, x2, x6);
                                let x7 = self.new_var();
                                self.add_constraint_lens(1, lc_len, 1);
                                let _ = (x2, x4, x6);
                                x7
                            }
                            _ => {
                                // Fallback path materializes ark var and uses enforce_pow_u64.
                                let v = self.new_var();
                                // lc has two terms (state + ark*one) in the code.
                                self.enforce_lc_times_one_eq_var(2, v);
                                self.enforce_pow_u64(v, cfg.alpha)
                            }
                        };
                        sbox_vars.push(out);
                    }
                    // MDS: t linear constraints, each LC has length t.
                    for _i in 0..t {
                        let v = self.new_var();
                        self.enforce_lc_times_one_eq_var(t, v);
                        next_vars.push(v);
                    }
                    let _ = sbox_vars;
                } else {
                    // Partial rounds: lane 0 S-boxed, other lanes linear with folded constants.
                    let ark0 = self.new_var();
                    // code always uses 2-term LC (state0 + ark*one) even if ark is zero
                    self.enforce_lc_times_one_eq_var(2, ark0);
                    let sbox0 = self.enforce_pow_u64(ark0, cfg.alpha);
                    for i in 0..t {
                        let mut lc_len = 1 /* sbox0 */ + (t - 1) /* linear lanes */;
                        if !partial_const_is_zero[r][i] {
                            lc_len += 1; // const term on var0
                        }
                        let v = self.new_var();
                        self.enforce_lc_times_one_eq_var(lc_len, v);
                        next_vars.push(v);
                    }
                    let _ = sbox0;
                }
                self.state_vars = next_vars;
            }
        }
    }

    fn count_shard<F: PrimeField>(
        cfg: &ark_crypto_primitives::sponge::poseidon::PoseidonConfig<F>,
        ops: &[PoseidonTraceOp<F>],
        init_mode: DuplexSpongeMode,
        constrain_init: bool,
        ark_is_zero: &[Vec<bool>],
        partial_const_is_zero: &[Vec<bool>],
    ) -> ShardCounts {
        let t = cfg.rate + cfg.capacity;
        let usable_bytes = ((F::MODULUS_BIT_SIZE - 1) / 8) as usize;
        let mut c = PoseidonShardCounter::<F>::new(t, init_mode, constrain_init);
        for op in ops {
            match op {
                PoseidonTraceOp::Absorb(elems) => {
                    if elems.is_empty() {
                        continue;
                    }
                    for _e in elems {
                        if matches!(c.mode, DuplexSpongeMode::Squeezing { .. }) {
                            c.apply_perm_counts(cfg, ark_is_zero, partial_const_is_zero);
                            c.mode = DuplexSpongeMode::Absorbing { next_absorb_index: 0 };
                        }
                        let mut absorb_index = match c.mode {
                            DuplexSpongeMode::Absorbing { next_absorb_index } => next_absorb_index,
                            _ => unreachable!(),
                        };
                        if absorb_index == cfg.rate {
                            c.apply_perm_counts(cfg, ark_is_zero, partial_const_is_zero);
                            absorb_index = 0;
                        }
                        // e_var
                        let e_var = c.new_var();
                        let _ = e_var;
                        // new state slot var + constraint: state[pos] + e == new
                        let _new_state = c.new_var();
                        c.enforce_lc_times_one_eq_var(2, _new_state);
                        // update bookkeeping (pos doesn't matter for counts)
                        let pos = cfg.capacity + absorb_index;
                        c.state_vars[pos] = _new_state;
                        c.mode = DuplexSpongeMode::Absorbing {
                            next_absorb_index: absorb_index + 1,
                        };
                    }
                }
                PoseidonTraceOp::SqueezeField(out) => {
                    if out.is_empty() {
                        continue;
                    }
                    let mut squeeze_index = match c.mode {
                        DuplexSpongeMode::Absorbing { .. } => {
                            c.apply_perm_counts(cfg, ark_is_zero, partial_const_is_zero);
                            0
                        }
                        DuplexSpongeMode::Squeezing { next_squeeze_index } => next_squeeze_index,
                    };
                    if squeeze_index == cfg.rate {
                        c.apply_perm_counts(cfg, ark_is_zero, partial_const_is_zero);
                        squeeze_index = 0;
                    }
                    let mut produced = 0usize;
                    while produced < out.len() {
                        let take = core::cmp::min(cfg.rate - squeeze_index, out.len() - produced);
                        produced += take;
                        squeeze_index += take;
                        if produced < out.len() && squeeze_index == cfg.rate {
                            c.apply_perm_counts(cfg, ark_is_zero, partial_const_is_zero);
                            squeeze_index = 0;
                        }
                    }
                    c.mode = DuplexSpongeMode::Squeezing {
                        next_squeeze_index: squeeze_index,
                    };
                }
                PoseidonTraceOp::SqueezeBytes { n, .. } => {
                    if usable_bytes == 0 {
                        // This is invalid in the real builder too.
                        continue;
                    }
                    let num_elements = (*n + usable_bytes - 1) / usable_bytes;
                    if num_elements == 0 {
                        continue;
                    }
                    let mut squeeze_index = match c.mode {
                        DuplexSpongeMode::Absorbing { .. } => {
                            c.apply_perm_counts(cfg, ark_is_zero, partial_const_is_zero);
                            0
                        }
                        DuplexSpongeMode::Squeezing { next_squeeze_index } => next_squeeze_index,
                    };
                    if squeeze_index == cfg.rate {
                        c.apply_perm_counts(cfg, ark_is_zero, partial_const_is_zero);
                        squeeze_index = 0;
                    }
                    let mut produced = 0usize;
                    while produced < num_elements {
                        let take = core::cmp::min(cfg.rate - squeeze_index, num_elements - produced);
                        produced += take;
                        squeeze_index += take;
                        if produced < num_elements && squeeze_index == cfg.rate {
                            c.apply_perm_counts(cfg, ark_is_zero, partial_const_is_zero);
                            squeeze_index = 0;
                        }
                    }
                    c.mode = DuplexSpongeMode::Squeezing {
                        next_squeeze_index: squeeze_index,
                    };
                }
            }
        }
        c.counts
    }

    let out_dir = out_dir.as_ref();
    create_dir_all(out_dir).map_err(|e| ReplayErr::Invalid(format!("create_dir_all failed: {e}")))?;

    let n_threads = rayon::current_num_threads().max(1);
    let t = cfg.rate + cfg.capacity;
    if t == 0 {
        return Err(ReplayErr::Invalid("invalid poseidon t=0".to_string()));
    }
    if shard_permutes == 0 {
        return Err(ReplayErr::Invalid("shard_permutes must be >0".to_string()));
    }

    #[derive(Clone)]
    struct ShardPlan<F: PrimeField> {
        start: usize,
        end: usize,
        init_state: Vec<F>,
        init_mode: DuplexSpongeMode,
        constrain_init: bool,
    }

    // ---------------------------------------------------------------------
    // Pass 0 (cheap): simulate sponge schedule to pick shard boundaries and
    // compute the correct initial state/mode for each shard.
    // ---------------------------------------------------------------------
    let mut plans: Vec<ShardPlan<F>> = Vec::new();
    let mut state: Vec<F> = vec![F::ZERO; t];
    let mut mode: DuplexSpongeMode = DuplexSpongeMode::Absorbing { next_absorb_index: 0 };
    let mut permutes_since_start: usize = 0;

    // initial shard begins from the fixed zero state; constrain it.
    plans.push(ShardPlan {
        start: 0,
        end: 0, // filled later
        init_state: state.clone(),
        init_mode: mode.clone(),
        constrain_init: true,
    });

    #[inline]
    fn permute_counted<F: PrimeField>(
        cfg: &ark_crypto_primitives::sponge::poseidon::PoseidonConfig<F>,
        state: &mut [F],
        permutes_since_start: &mut usize,
    ) {
        permute_in_place(cfg, state);
        *permutes_since_start = permutes_since_start.saturating_add(1);
    }

    // Arkworks squeeze-bytes uses: take (MODULUS_BIT_SIZE-1)/8 bytes from each squeezed field element.
    let usable_bytes = ((F::MODULUS_BIT_SIZE - 1) / 8) as usize;

    for (op_idx, op) in ops.iter().enumerate() {
        match op {
            PoseidonTraceOp::Absorb(elems) => {
                for &e in elems {
                    let mut absorb_index = match mode {
                        DuplexSpongeMode::Absorbing { next_absorb_index } => next_absorb_index,
                        DuplexSpongeMode::Squeezing { .. } => {
                            permute_counted(cfg, &mut state, &mut permutes_since_start);
                            0
                        }
                    };
                    if absorb_index == cfg.rate {
                        permute_counted(cfg, &mut state, &mut permutes_since_start);
                        absorb_index = 0;
                    }
                    // state[cap + absorb_index] += e
                    let pos = cfg.capacity + absorb_index;
                    state[pos] += e;
                    mode = DuplexSpongeMode::Absorbing {
                        next_absorb_index: absorb_index + 1,
                    };
                }
            }
            PoseidonTraceOp::SqueezeField(out) => {
                if out.is_empty() {
                    // no-op for mode; keep consistent with arithmetizer.
                } else {
                    let mut squeeze_index = match mode {
                        DuplexSpongeMode::Absorbing { .. } => {
                            permute_counted(cfg, &mut state, &mut permutes_since_start);
                            0
                        }
                        DuplexSpongeMode::Squeezing { next_squeeze_index } => next_squeeze_index,
                    };
                    if squeeze_index == cfg.rate {
                        permute_counted(cfg, &mut state, &mut permutes_since_start);
                        squeeze_index = 0;
                    }
                    let mut produced = 0usize;
                    while produced < out.len() {
                        let take = core::cmp::min(cfg.rate - squeeze_index, out.len() - produced);
                        produced += take;
                        squeeze_index += take;
                        if produced < out.len() && squeeze_index == cfg.rate {
                            permute_counted(cfg, &mut state, &mut permutes_since_start);
                            squeeze_index = 0;
                        }
                    }
                    mode = DuplexSpongeMode::Squeezing {
                        next_squeeze_index: squeeze_index,
                    };
                }
            }
            PoseidonTraceOp::SqueezeBytes { n, .. } => {
                if usable_bytes == 0 {
                    return Err(ReplayErr::Invalid("usable_bytes computed as 0".to_string()));
                }
                let num_elements = (*n + usable_bytes - 1) / usable_bytes;
                if num_elements == 0 {
                    // leave mode unchanged
                } else {
                    let mut squeeze_index = match mode {
                        DuplexSpongeMode::Absorbing { .. } => {
                            permute_counted(cfg, &mut state, &mut permutes_since_start);
                            0
                        }
                        DuplexSpongeMode::Squeezing { next_squeeze_index } => next_squeeze_index,
                    };
                    if squeeze_index == cfg.rate {
                        permute_counted(cfg, &mut state, &mut permutes_since_start);
                        squeeze_index = 0;
                    }
                    let mut produced = 0usize;
                    while produced < num_elements {
                        let take = core::cmp::min(cfg.rate - squeeze_index, num_elements - produced);
                        produced += take;
                        squeeze_index += take;
                        if produced < num_elements && squeeze_index == cfg.rate {
                            permute_counted(cfg, &mut state, &mut permutes_since_start);
                            squeeze_index = 0;
                        }
                    }
                    mode = DuplexSpongeMode::Squeezing {
                        next_squeeze_index: squeeze_index,
                    };
                }
            }
        }

        // Cut shards on permute count boundaries (after finishing this op).
        if permutes_since_start >= shard_permutes && op_idx + 1 < ops.len() {
            let next_start = op_idx + 1;
            // Close current plan.
            if let Some(last) = plans.last_mut() {
                last.end = next_start;
            }
            permutes_since_start = 0;
            plans.push(ShardPlan {
                start: next_start,
                end: 0, // filled later
                init_state: state.clone(),
                init_mode: mode.clone(),
                constrain_init: false,
            });
        }
    }
    // Close final plan.
    if let Some(last) = plans.last_mut() {
        last.end = ops.len();
    }

    if poseidon_profile_on() {
        eprintln!(
            "[poseidon_arith] start sharded ops={} shard_permutes={} shards={} threads={}",
            ops.len(),
            shard_permutes,
            plans.len(),
            n_threads
        );
    }

    // ---------------------------------------------------------------------
    // Direct-to-merged (heavy): count -> prealloc merged -> parallel write into merged.
    // We stitch shards together by adding equality constraints on boundary state vars.
    // ---------------------------------------------------------------------
    #[derive(Clone)]
    struct ShardMeta {
        wiring: PoseidonDr1csWiring,
        bw: PoseidonByteWiring,
        bytes: Vec<ByteSqueezeWitness>,
        nvars: usize,
        ckpts: Vec<(u64, u64, u64, u64)>,
    }

    #[derive(Clone)]
    struct Group<F: PrimeField> {
        asg: Vec<F>,
        init_state_vars: Vec<usize>,
        final_state_vars: Vec<usize>,
    }

    struct ShardBuilt<F: PrimeField> {
        group: Group<F>,
        meta: ShardMeta,
    }

    // Precompute booleans for exact counting with the optimized arithmetization.
    let t = cfg.rate + cfg.capacity;
    let total_rounds = cfg.full_rounds + cfg.partial_rounds;
    let mut ark_is_zero: Vec<Vec<bool>> = vec![vec![false; t]; total_rounds];
    for r in 0..total_rounds {
        for i in 0..t {
            ark_is_zero[r][i] = cfg.ark[r][i].is_zero();
        }
    }
    let mut partial_const_is_zero: Vec<Vec<bool>> = vec![vec![true; t]; total_rounds];
    let full_rounds_over_2 = cfg.full_rounds / 2;
    for r in 0..total_rounds {
        let is_full = r < full_rounds_over_2 || r >= (full_rounds_over_2 + cfg.partial_rounds);
        if is_full {
            continue;
        }
        for i in 0..t {
            let mut const_term = F::ZERO;
            for j in 1..t {
                const_term += cfg.mds[i][j] * cfg.ark[r][j];
            }
            partial_const_is_zero[r][i] = const_term.is_zero();
        }
    }

    // Pass 0: structural count per shard to preallocate merged files.
    let shard_counts: Vec<ShardCounts> = plans
        .iter()
        .map(|p| {
            let slice = &ops[p.start..p.end];
            count_shard::<F>(
                cfg,
                slice,
                p.init_mode.clone(),
                p.constrain_init,
                &ark_is_zero,
                &partial_const_is_zero,
            )
        })
        .collect();
    if shard_counts.is_empty() {
        return Err(ReplayErr::Invalid("poseidon shard plan produced empty result".to_string()));
    }

    // Prefix sums for disjoint write regions (exclude shared var0 in the global var packing).
    let mut var_tail_off: Vec<u32> = Vec::with_capacity(shard_counts.len());
    let mut row_off: Vec<u64> = Vec::with_capacity(shard_counts.len());
    let mut a_off: Vec<u64> = Vec::with_capacity(shard_counts.len());
    let mut b_off: Vec<u64> = Vec::with_capacity(shard_counts.len());
    let mut c_off: Vec<u64> = Vec::with_capacity(shard_counts.len());
    let mut cur_var_tail: u64 = 0;
    let mut cur_rows: u64 = 0;
    let mut cur_a: u64 = 0;
    let mut cur_b: u64 = 0;
    let mut cur_c: u64 = 0;
    for sc in &shard_counts {
        var_tail_off.push(cur_var_tail as u32);
        row_off.push(cur_rows);
        a_off.push(cur_a);
        b_off.push(cur_b);
        c_off.push(cur_c);
        cur_var_tail = cur_var_tail.saturating_add(sc.nvars.saturating_sub(1));
        cur_rows = cur_rows.saturating_add(sc.nconstraints);
        cur_a = cur_a.saturating_add(sc.a_terms);
        cur_b = cur_b.saturating_add(sc.b_terms);
        cur_c = cur_c.saturating_add(sc.c_terms);
    }

    // Boundary equality constraints: t per boundary, each adds 1 term to A/B/C and 1 row.
    let n_boundaries = plans.len().saturating_sub(1);
    let eq_rows: u64 = (n_boundaries as u64).saturating_mul(t as u64);
    let total_rows = cur_rows.saturating_add(eq_rows);
    let total_a_terms = cur_a.saturating_add(eq_rows);
    let total_b_terms = cur_b.saturating_add(eq_rows);
    let total_c_terms = cur_c.saturating_add(eq_rows);
    let total_nvars: usize = 1usize.saturating_add(cur_var_tail as usize);

    // Preallocate merged files.
    let merged_dir = out_dir.join("poseidon_merged");
    fast_prepare_out_dir(&merged_dir).map_err(ReplayErr::Invalid)?;
    create_dir_all(&merged_dir).map_err(|e| ReplayErr::Invalid(format!("create merged dir failed: {e}")))?;

    #[cfg(not(unix))]
    {
        return Err(ReplayErr::Invalid("direct-to-merged sharded poseidon requires unix".to_string()));
    }
    #[cfg(unix)]
    let (fc_a, fi_a, fc_b, fi_b, fc_c, fi_c, f_rows) = {
        let open_rw = |p: std::path::PathBuf| -> Result<std::fs::File, ReplayErr> {
            OpenOptions::new()
                .create(true)
                .read(true)
                .write(true)
                .truncate(true)
                .open(&p)
                .map_err(|e| ReplayErr::Invalid(format!("open {p:?} failed: {e}")))
        };
        let pa_c = merged_dir.join("a_coeffs.bin");
        let pa_i = merged_dir.join("a_idx.bin");
        let pb_c = merged_dir.join("b_coeffs.bin");
        let pb_i = merged_dir.join("b_idx.bin");
        let pc_c = merged_dir.join("c_coeffs.bin");
        let pc_i = merged_dir.join("c_idx.bin");
        let prow = merged_dir.join("constraints.bin");
        let mut fc_a = open_rw(pa_c)?;
        let mut fi_a = open_rw(pa_i)?;
        let mut fc_b = open_rw(pb_c)?;
        let mut fi_b = open_rw(pb_i)?;
        let mut fc_c = open_rw(pc_c)?;
        let mut fi_c = open_rw(pc_i)?;
        let mut f_rows = open_rw(prow)?;

        fc_a.set_len(total_a_terms.saturating_mul(2) as u64)
            .map_err(|e| ReplayErr::Invalid(format!("set_len a_coeffs failed: {e}")))?;
        fi_a.set_len(total_a_terms.saturating_mul(4) as u64)
            .map_err(|e| ReplayErr::Invalid(format!("set_len a_idx failed: {e}")))?;
        fc_b.set_len(total_b_terms.saturating_mul(2) as u64)
            .map_err(|e| ReplayErr::Invalid(format!("set_len b_coeffs failed: {e}")))?;
        fi_b.set_len(total_b_terms.saturating_mul(4) as u64)
            .map_err(|e| ReplayErr::Invalid(format!("set_len b_idx failed: {e}")))?;
        fc_c.set_len(total_c_terms.saturating_mul(2) as u64)
            .map_err(|e| ReplayErr::Invalid(format!("set_len c_coeffs failed: {e}")))?;
        fi_c.set_len(total_c_terms.saturating_mul(4) as u64)
            .map_err(|e| ReplayErr::Invalid(format!("set_len c_idx failed: {e}")))?;
        f_rows
            .set_len(total_rows.saturating_mul(12) as u64)
            .map_err(|e| ReplayErr::Invalid(format!("set_len constraints failed: {e}")))?;

        // Seek to end so the file length is visible on some filesystems immediately.
        let _ = fc_a.seek(SeekFrom::End(0));
        let _ = fi_a.seek(SeekFrom::End(0));
        let _ = fc_b.seek(SeekFrom::End(0));
        let _ = fi_b.seek(SeekFrom::End(0));
        let _ = fc_c.seek(SeekFrom::End(0));
        let _ = fi_c.seek(SeekFrom::End(0));
        let _ = f_rows.seek(SeekFrom::End(0));
        (fc_a, fi_a, fc_b, fi_b, fc_c, fi_c, f_rows)
    };

    // Pass 1 (heavy): build shards in parallel, writing directly into merged files.
    let shard_built: Vec<ShardBuilt<F>> = plans
        .par_iter()
        .enumerate()
        .map(|(shard_idx, p)| -> Result<ShardBuilt<F>, ReplayErr> {
            let slice = &ops[p.start..p.end];

            #[cfg(unix)]
            let writer = FileBackedRangeWriter::new(
                fc_a.try_clone().map_err(|e| ReplayErr::Invalid(format!("clone a_coeffs failed: {e}")))?,
                fi_a.try_clone().map_err(|e| ReplayErr::Invalid(format!("clone a_idx failed: {e}")))?,
                fc_b.try_clone().map_err(|e| ReplayErr::Invalid(format!("clone b_coeffs failed: {e}")))?,
                fi_b.try_clone().map_err(|e| ReplayErr::Invalid(format!("clone b_idx failed: {e}")))?,
                fc_c.try_clone().map_err(|e| ReplayErr::Invalid(format!("clone c_coeffs failed: {e}")))?,
                fi_c.try_clone().map_err(|e| ReplayErr::Invalid(format!("clone c_idx failed: {e}")))?,
                f_rows.try_clone().map_err(|e| ReplayErr::Invalid(format!("clone constraints failed: {e}")))?,
                a_off[shard_idx],
                b_off[shard_idx],
                c_off[shard_idx],
                row_off[shard_idx],
            );

            #[cfg(unix)]
            let prebuilt = Some(Dr1csBuilder::<F>::new_file_backed_range(writer, var_tail_off[shard_idx]));
            #[cfg(not(unix))]
            let prebuilt: Option<Dr1csBuilder<F>> = None;

            let (inst_any, asg, _replay, bytes, wiring, bw, init_state_vars, final_state_vars, ckpts) =
                poseidon_sponge_dr1cs_from_trace_impl_any_internal(
                    cfg,
                    slice,
                    false,
                    PoseidonArithMode::WeWitness,
                    prebuilt,
                    None,
                    &p.init_state,
                    p.init_mode.clone(),
                    p.constrain_init,
                )?;
            if !matches!(inst_any, PoseidonInstance::InMemory(_)) {
                return Err(ReplayErr::Invalid("expected range-writer build to return placeholder instance".to_string()));
            }
            let Some(range_res) = ckpts else {
                return Err(ReplayErr::Invalid("range-writer build missing checkpoint payload".to_string()));
            };
            // Validate that the structural counter matched the actual emitted shape.
            let exp = shard_counts
                .get(shard_idx)
                .ok_or_else(|| ReplayErr::Invalid("shard_counts index OOB".to_string()))?;
            if range_res.rows != exp.nconstraints
                || range_res.a_terms != exp.a_terms
                || range_res.b_terms != exp.b_terms
                || range_res.c_terms != exp.c_terms
            {
                return Err(ReplayErr::Invalid(format!(
                    "poseidon sharded direct write: count mismatch for shard {shard_idx}: \
rows got {} exp {}, a_terms got {} exp {}, b_terms got {} exp {}, c_terms got {} exp {}",
                    range_res.rows,
                    exp.nconstraints,
                    range_res.a_terms,
                    exp.a_terms,
                    range_res.b_terms,
                    exp.b_terms,
                    range_res.c_terms,
                    exp.c_terms
                )));
            }
            let nvars = asg.len();
            Ok(ShardBuilt {
                group: Group {
                    asg,
                    init_state_vars,
                    final_state_vars,
                },
                meta: ShardMeta {
                    wiring,
                    bw,
                    bytes,
                    nvars,
                    ckpts: range_res.ckpts,
                },
            })
        })
        .collect::<Result<Vec<_>, _>>()?;

    // Unpack shard outputs.
    let mut shard_metas: Vec<ShardMeta> = Vec::with_capacity(shard_built.len());
    let mut groups: Vec<Group<F>> = Vec::with_capacity(shard_built.len());
    for sb in shard_built {
        groups.push(sb.group);
        shard_metas.push(sb.meta);
    }

    // Build merged assignment by concatenating shard assignments (share var0).
    let mut merged_asg: Vec<F> = Vec::with_capacity(total_nvars);
    merged_asg.push(F::ONE);
    for g in &groups {
        if g.asg.is_empty() || g.asg[0] != F::ONE {
            return Err(ReplayErr::Invalid("shard assignment missing var0=1".to_string()));
        }
        merged_asg.extend_from_slice(&g.asg[1..]);
    }
    if merged_asg.len() != total_nvars {
        return Err(ReplayErr::Invalid(format!(
            "poseidon sharded direct write: merged assignment length mismatch (got {}, expected {})",
            merged_asg.len(),
            total_nvars
        )));
    }

    // Boundary equalities: final_state(shard i) == init_state(shard i+1) (in merged space).
    let map_var = |shard_idx: usize, local: usize| -> u32 {
        if local == 0 {
            0
        } else {
            (local as u64)
                .saturating_add(var_tail_off[shard_idx] as u64)
                .try_into()
                .unwrap()
        }
    };
    let mut boundary_eqs: Vec<(u32, u32)> = Vec::new();
    for i in 0..groups.len().saturating_sub(1) {
        let left = &groups[i];
        let right = &groups[i + 1];
        for (x, y) in left.final_state_vars.iter().zip(right.init_state_vars.iter()) {
            boundary_eqs.push((map_var(i, *x), map_var(i + 1, *y)));
        }
    }

    // Append boundary equalities into the reserved tail range.
    #[cfg(unix)]
    let mut eq_writer = FileBackedRangeWriter::new(
        fc_a.try_clone().map_err(|e| ReplayErr::Invalid(format!("clone a_coeffs failed: {e}")))?,
        fi_a.try_clone().map_err(|e| ReplayErr::Invalid(format!("clone a_idx failed: {e}")))?,
        fc_b.try_clone().map_err(|e| ReplayErr::Invalid(format!("clone b_coeffs failed: {e}")))?,
        fi_b.try_clone().map_err(|e| ReplayErr::Invalid(format!("clone b_idx failed: {e}")))?,
        fc_c.try_clone().map_err(|e| ReplayErr::Invalid(format!("clone c_coeffs failed: {e}")))?,
        fi_c.try_clone().map_err(|e| ReplayErr::Invalid(format!("clone c_idx failed: {e}")))?,
        f_rows.try_clone().map_err(|e| ReplayErr::Invalid(format!("clone constraints failed: {e}")))?,
        cur_a,
        cur_b,
        cur_c,
        cur_rows,
    );
    #[cfg(unix)]
    {
        let one_u16: u16 = 1;
        let a_coeffs: Vec<u8> = (0..boundary_eqs.len())
            .flat_map(|_| one_u16.to_le_bytes())
            .collect();
        let b_coeffs = a_coeffs.clone();
        let c_coeffs = a_coeffs.clone();
        let a_idx: Vec<u32> = boundary_eqs.iter().map(|(x, _)| *x).collect();
        let b_idx: Vec<u32> = vec![0u32; boundary_eqs.len()];
        let c_idx: Vec<u32> = boundary_eqs.iter().map(|(_, y)| *y).collect();
        let lens: Vec<u32> = (0..boundary_eqs.len()).flat_map(|_| [1u32, 1u32, 1u32]).collect();
        eq_writer
            .push_a_terms_raw_block(&a_coeffs, &a_idx)
            .map_err(ReplayErr::Invalid)?;
        eq_writer
            .push_b_terms_raw_block(&b_coeffs, &b_idx)
            .map_err(ReplayErr::Invalid)?;
        eq_writer
            .push_c_terms_raw_block(&c_coeffs, &c_idx)
            .map_err(ReplayErr::Invalid)?;
        eq_writer.push_constraint_lens_block(&lens).map_err(ReplayErr::Invalid)?;
    }
    if (boundary_eqs.len() as u64) != eq_rows {
        return Err(ReplayErr::Invalid(format!(
            "poseidon sharded direct write: boundary eq count mismatch (got {}, expected {})",
            boundary_eqs.len(),
            eq_rows
        )));
    }

    // Gather and write checkpoints.
    let mut ckpts_all: Vec<(u64, u64, u64, u64)> = Vec::new();
    for s in &shard_metas {
        ckpts_all.extend_from_slice(&s.ckpts);
    }
    ckpts_all.extend(eq_writer.take_ckpts().into_iter());
    ckpts_all.sort_by_key(|(row_idx, _, _, _)| *row_idx);
    ckpts_all.dedup_by_key(|(row_idx, _, _, _)| *row_idx);
    // Always include row 0 checkpoint if missing.
    if ckpts_all.first().map(|x| x.0) != Some(0) {
        ckpts_all.insert(0, (0, 0, 0, 0));
    }
    for (row_idx, a0, b0, c0) in &ckpts_all {
        if *row_idx >= total_rows {
            return Err(ReplayErr::Invalid(format!(
                "poseidon sharded direct write: ckpt row_idx out of range (row_idx={row_idx} total_rows={total_rows})"
            )));
        }
        if *a0 > total_a_terms || *b0 > total_b_terms || *c0 > total_c_terms {
            return Err(ReplayErr::Invalid(format!(
                "poseidon sharded direct write: ckpt term offset out of range (row_idx={row_idx} a0={a0}/{total_a_terms} b0={b0}/{total_b_terms} c0={c0}/{total_c_terms})"
            )));
        }
    }
    {
        let mut f = std::io::BufWriter::new(
            std::fs::File::create(merged_dir.join("rows_ckpt.bin"))
                .map_err(|e| ReplayErr::Invalid(format!("create rows_ckpt failed: {e}")))?,
        );
        for (row_idx, a0, b0, c0) in &ckpts_all {
            f.write_all(&row_idx.to_le_bytes())
                .map_err(|e| ReplayErr::Invalid(format!("write ckpt failed: {e}")))?;
            f.write_all(&a0.to_le_bytes())
                .map_err(|e| ReplayErr::Invalid(format!("write ckpt failed: {e}")))?;
            f.write_all(&b0.to_le_bytes())
                .map_err(|e| ReplayErr::Invalid(format!("write ckpt failed: {e}")))?;
            f.write_all(&c0.to_le_bytes())
                .map_err(|e| ReplayErr::Invalid(format!("write ckpt failed: {e}")))?;
        }
    }

    // Write meta.txt matching the append writer format.
    let modulus_bigint = F::MODULUS;
    let limbs = modulus_bigint.as_ref();
    if !(limbs.len() == 1 && limbs[0] > 1 && limbs[0] <= 65535) {
        return Err(ReplayErr::Invalid(format!("tiny file-backed format requires modulus<=65535 (got limbs={limbs:?})")));
    }
    let modulus = limbs[0];
    {
        let mut f = std::io::BufWriter::new(
            std::fs::File::create(merged_dir.join("meta.txt"))
                .map_err(|e| ReplayErr::Invalid(format!("create meta failed: {e}")))?,
        );
        use std::io::Write;
        writeln!(f, "nvars={}", total_nvars).ok();
        writeln!(f, "constraints={}", total_rows).ok();
        writeln!(f, "a_terms={}", total_a_terms).ok();
        writeln!(f, "b_terms={}", total_b_terms).ok();
        writeln!(f, "c_terms={}", total_c_terms).ok();
        writeln!(f, "coeff_size=2").ok();
        writeln!(f, "idx_size=4").ok();
        writeln!(f, "row_size=12").ok();
        writeln!(f, "row_ckpt_stride={}", 1u64 << 20).ok();
        writeln!(f, "format=tiny_u16_u32_rows_len_u32_v1").ok();
        writeln!(f, "modulus={}", modulus).ok();
    }

    let merged_inst = FileBackedSparseDr1csInstance::<F>::new(
        total_nvars,
        FileBackedLayout {
            dir: merged_dir.clone(),
            coeff_size: 2,
            idx_size: 4,
            row_size: 12,
            nconstraints: total_rows,
            a_terms: total_a_terms,
            b_terms: total_b_terms,
            c_terms: total_c_terms,
        },
    );

    // ---------------------------------------------------------------------
    // Stitch global wiring/byte wiring by offsetting per-shard var indices.
    // ---------------------------------------------------------------------
    let mut wiring = PoseidonDr1csWiring::default();
    let mut bw = PoseidonByteWiring::default();
    let mut bytes_all: Vec<ByteSqueezeWitness> = Vec::new();

    let mut var_tail_off: usize = 0; // excludes var0
    let mut absorb_off: usize = 0;
    let mut squeeze_off: usize = 0;
    let mut squeeze_byte_off: usize = 0;

    for s in shard_metas.iter() {
        // var mapping for this shard in the final merged instance.
        let map_var = |idx: usize| -> usize {
            if idx == 0 { 0 } else { idx + var_tail_off }
        };

        for &v in &s.wiring.absorb_vars {
            wiring.absorb_vars.push(map_var(v));
        }
        for &(st, ln) in &s.wiring.absorb_ranges {
            wiring.absorb_ranges.push((absorb_off + st, ln));
        }
        absorb_off += s.wiring.absorb_vars.len();

        for &v in &s.wiring.squeeze_field_vars {
            wiring.squeeze_field_vars.push(map_var(v));
        }
        for &(st, ln) in &s.wiring.squeeze_field_ranges {
            wiring.squeeze_field_ranges.push((squeeze_off + st, ln));
        }
        squeeze_off += s.wiring.squeeze_field_vars.len();

        for &v in &s.bw.squeeze_byte_vars {
            bw.squeeze_byte_vars.push(map_var(v));
        }
        for &(st, ln) in &s.bw.squeeze_byte_ranges {
            bw.squeeze_byte_ranges.push((squeeze_byte_off + st, ln));
        }
        squeeze_byte_off += s.bw.squeeze_byte_vars.len();

        for w in &s.bytes {
            let src_elems = w.src_elems.iter().copied().map(map_var).collect();
            bytes_all.push(ByteSqueezeWitness {
                n: w.n,
                usable_bytes: w.usable_bytes,
                src_elems,
                out: w.out.clone(),
            });
        }

        var_tail_off = var_tail_off.saturating_add(s.nvars.saturating_sub(1));
    }

    // WE mode: replay is not used; keep a minimal placeholder.
    let replay = PoseidonSpongeReplayResult {
        final_state: vec![F::ZERO; cfg.rate + cfg.capacity],
        permutes: Vec::new(),
    };

    if poseidon_profile_on() {
        eprintln!(
            "[poseidon_arith] done sharded nvars={} constraints={}",
            merged_asg.len(),
            merged_inst.layout.nconstraints
        );
    }
    Ok((merged_inst, merged_asg, replay, bytes_all, wiring, bw))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PoseidonArithMode {
    /// “Replay mode”: enforce Absorb/Squeeze outputs match the recorded trace values and run
    /// sanity checks against a replay.
    ReplayFixed,
    /// “WE mode”: do not bake recorded trace values into constraints.
    WeWitness,
}

fn poseidon_sponge_dr1cs_from_trace_impl_any<F: PrimeField + CanonicalSerialize>(
    cfg: &ark_crypto_primitives::sponge::poseidon::PoseidonConfig<F>,
    ops: &[PoseidonTraceOp<F>],
    with_bytes: bool,
    arith_mode: PoseidonArithMode,
    out_dir: Option<&Path>,
) -> Result<
    (
        PoseidonInstance<F>,
        Vec<F>,
        PoseidonSpongeReplayResult<F>,
        Vec<ByteSqueezeWitness>,
        PoseidonDr1csWiring,
        PoseidonByteWiring,
    ),
    ReplayErr,
> {
    let t = cfg.rate + cfg.capacity;
    let init_state = vec![F::ZERO; t];
    let init_mode = ark_crypto_primitives::sponge::DuplexSpongeMode::Absorbing {
        next_absorb_index: 0,
    };
    let (inst, asg, replay, bytes, wiring, bw, _init_state_vars, _final_state_vars, _ckpts) =
        poseidon_sponge_dr1cs_from_trace_impl_any_internal(
            cfg,
            ops,
            with_bytes,
            arith_mode,
            None,
            out_dir,
            &init_state,
            init_mode,
            true,
        )?;
    Ok((inst, asg, replay, bytes, wiring, bw))
}

fn poseidon_sponge_dr1cs_from_trace_impl_any_internal<F: PrimeField + CanonicalSerialize>(
    cfg: &ark_crypto_primitives::sponge::poseidon::PoseidonConfig<F>,
    ops: &[PoseidonTraceOp<F>],
    with_bytes: bool,
    arith_mode: PoseidonArithMode,
    mut prebuilt: Option<Dr1csBuilder<F>>,
    out_dir: Option<&Path>,
    init_state: &[F],
    init_mode: ark_crypto_primitives::sponge::DuplexSpongeMode,
    constrain_init: bool,
) -> Result<
    (
        PoseidonInstance<F>,
        Vec<F>,
        PoseidonSpongeReplayResult<F>,
        Vec<ByteSqueezeWitness>,
        PoseidonDr1csWiring,
        PoseidonByteWiring,
        Vec<usize>,
        Vec<usize>,
        Option<RangeWriteResult>,
    ),
    ReplayErr,
> {
    // In WE mode we must not require replay consistency (arming has no concrete trace yet),
    // and we must not bake recorded trace outputs into constraints.
    let replay = if arith_mode == PoseidonArithMode::ReplayFixed {
        replay_ops(cfg, ops)?
    } else {
        PoseidonSpongeReplayResult {
            final_state: vec![F::ZERO; cfg.rate + cfg.capacity],
            permutes: Vec::new(),
        }
    };

    let t = cfg.rate + cfg.capacity;
    if init_state.len() != t {
        return Err(ReplayErr::Invalid(format!(
            "init_state length mismatch: expected {t}, got {}",
            init_state.len()
        )));
    }
    let mut b = if let Some(b) = prebuilt.take() {
        b
    } else if let Some(dir) = out_dir {
        Dr1csBuilder::<F>::new_file_backed(dir).map_err(ReplayErr::Invalid)?
    } else {
        Dr1csBuilder::<F>::new()
    };
    let one = b.one();

    // Initial state: either fixed zeros (single-shard) or carried-in witness state (sharded mode).
    let mut state_vars = Vec::with_capacity(t);
    for i in 0..t {
        let v = b.new_var(init_state[i]);
        if constrain_init {
            b.enforce_var_eq_const(v, init_state[i]);
        }
        state_vars.push(v);
    }
    let init_state_vars = state_vars.clone();

    let mut mode = init_mode;

    let mut byte_witnesses: Vec<ByteSqueezeWitness> = Vec::new();
    let mut wiring = PoseidonDr1csWiring::default();
    let mut byte_wiring = PoseidonByteWiring::default();

    // Helper: apply a Poseidon permutation to the current `state_vars`.
    let mut permute_ptr: usize = 0;
    let mut apply_perm = |b: &mut Dr1csBuilder<F>, state_vars: &mut Vec<usize>| -> Result<(), ReplayErr> {
        let (after_state, round_states) = if arith_mode == PoseidonArithMode::ReplayFixed {
            if permute_ptr >= replay.permutes.len() {
                return Err(ReplayErr::Invalid("permute ptr out of range".to_string()));
            }
            let before = replay.permutes[permute_ptr].before.clone();
            let (after_state, round_states) = permute_with_round_trace(cfg, &before)?;

            // Ensure the current state witness matches `before` (sanity; not a constraint).
            for i in 0..t {
                if b.assignment[state_vars[i]] != before[i] {
                    return Err(ReplayErr::Mismatch(format!(
                        "state mismatch before permute #{permute_ptr} at i={i}"
                    )));
                }
            }
            (after_state, round_states)
        } else {
            // WE mode: no replay-derived sanity checks.
            let before = (0..t).map(|i| b.assignment[state_vars[i]]).collect::<Vec<_>>();
            permute_with_round_trace(cfg, &before)?
        };

        let full_rounds_over_2 = cfg.full_rounds / 2;
        let total_rounds = cfg.full_rounds + cfg.partial_rounds;
        assert_eq!(round_states.len(), total_rounds);

        for r in 0..total_rounds {
            let is_full = r < full_rounds_over_2 || r >= (full_rounds_over_2 + cfg.partial_rounds);

            let mut next_vars: Vec<usize> = Vec::with_capacity(t);
            if is_full {
                // Full rounds: ARK -> SBOX on all lanes -> MDS.
                //
                // Optimization (safe): avoid materializing `ark_vars[i] = state[i] + ark[r][i]`
                // as separate linear constraints/variables. Instead, feed the affine form
                // `(state[i] + const)` directly into the S-box exponentiation constraints as
                // a linear combination.
                //
                // This reduces ~t linear constraints per full round (and their vars), which is
                // a meaningful fraction of Poseidon cost for large permute counts.

                #[inline]
                fn eval_lc<F: PrimeField>(b: &Dr1csBuilder<F>, lc: &[(F, usize)]) -> F {
                    lc.iter()
                        .fold(F::ZERO, |acc, (c, idx)| acc + (*c * b.assignment[*idx]))
                }

                #[inline]
                fn enforce_mul_lc_lc<F: PrimeField>(
                    b: &mut Dr1csBuilder<F>,
                    a: Vec<(F, usize)>,
                    bb: Vec<(F, usize)>,
                ) -> usize {
                    let out_val = eval_lc::<F>(b, &a) * eval_lc::<F>(b, &bb);
                    let out = b.new_var(out_val);
                    b.add_constraint(a, bb, vec![(F::ONE, out)]);
                    out
                }

                #[inline]
                fn enforce_mul_var_lc<F: PrimeField>(
                    b: &mut Dr1csBuilder<F>,
                    x: usize,
                    lc: Vec<(F, usize)>,
                ) -> usize {
                    let out_val = b.assignment[x] * eval_lc::<F>(b, &lc);
                    let out = b.new_var(out_val);
                    b.add_constraint(vec![(F::ONE, x)], lc, vec![(F::ONE, out)]);
                    out
                }

                // S-box outputs per lane.
                let mut sbox_vars: Vec<usize> = Vec::with_capacity(t);
                for i in 0..t {
                    let ark = cfg.ark[r][i];
                    // lc_in = state[i] + ark
                    let lc_in = if ark.is_zero() {
                        vec![(F::ONE, state_vars[i])]
                    } else {
                        vec![(F::ONE, state_vars[i]), (ark, one)]
                    };

                    let out = match cfg.alpha {
                        3 => {
                            // x^3 = (x^2)*x
                            let x2 = enforce_mul_lc_lc::<F>(b, lc_in.clone(), lc_in.clone());
                            enforce_mul_var_lc::<F>(b, x2, lc_in)
                        }
                        5 => {
                            // x^5 = (x^2)^2 * x
                            let x2 = enforce_mul_lc_lc::<F>(b, lc_in.clone(), lc_in.clone());
                            let x4_val = b.assignment[x2] * b.assignment[x2];
                            let x4 = b.new_var(x4_val);
                            b.enforce_mul(x2, x2, x4);
                            enforce_mul_var_lc::<F>(b, x4, lc_in)
                        }
                        7 => {
                            // x^7 = (x^2)^2 * x^2 * x
                            let x2 = enforce_mul_lc_lc::<F>(b, lc_in.clone(), lc_in.clone());
                            let x4_val = b.assignment[x2] * b.assignment[x2];
                            let x4 = b.new_var(x4_val);
                            b.enforce_mul(x2, x2, x4);
                            let x6_val = x4_val * b.assignment[x2];
                            let x6 = b.new_var(x6_val);
                            b.enforce_mul(x4, x2, x6);
                            enforce_mul_var_lc::<F>(b, x6, lc_in)
                        }
                        _ => {
                            // Fallback: materialize ARK as a var and use the generic exponentiation gadget.
                            let val = b.assignment[state_vars[i]] + ark;
                            let v = b.new_var(val);
                            b.enforce_lc_times_one_eq_var(vec![(F::ONE, state_vars[i]), (ark, one)], v);
                            b.enforce_pow_u64(v, cfg.alpha)
                        }
                    };
                    sbox_vars.push(out);
                }

                for i in 0..t {
                    let mut lc: Vec<(F, usize)> = Vec::with_capacity(t);
                    let mut val = F::ZERO;
                    for j in 0..t {
                        let coeff = cfg.mds[i][j];
                        lc.push((coeff, sbox_vars[j]));
                        val += coeff * b.assignment[sbox_vars[j]];
                    }
                    let v = b.new_var(val);
                    b.enforce_lc_times_one_eq_var(lc, v);
                    next_vars.push(v);
                }
            } else {
                // Partial rounds: only lane 0 is S-boxed.
                //
                // Optimization: avoid materializing ARK vars (and equality vars) for the identity lanes.
                // We fold their ARK constants directly into the MDS linear constraints:
                //   input[j] = state[j] + ark[r][j] for j>0, and input[0] = SBOX(state[0] + ark[r][0]).
                let ark0_val = b.assignment[state_vars[0]] + cfg.ark[r][0];
                let ark0 = b.new_var(ark0_val);
                b.enforce_lc_times_one_eq_var(vec![(F::ONE, state_vars[0]), (cfg.ark[r][0], one)], ark0);
                let sbox0 = b.enforce_pow_u64(ark0, cfg.alpha);

                for i in 0..t {
                    // y_i = Σ_j mds[i][j] * input[j]
                    let mut lc: Vec<(F, usize)> = Vec::with_capacity(t + 1);
                    let mut val = F::ZERO;
                    // j=0 nonlinear lane
                    let c0 = cfg.mds[i][0];
                    lc.push((c0, sbox0));
                    val += c0 * b.assignment[sbox0];

                    // j>0 linear lanes: use state var directly, and fold constants into the LC.
                    let mut const_term = F::ZERO;
                    for j in 1..t {
                        let coeff = cfg.mds[i][j];
                        lc.push((coeff, state_vars[j]));
                        val += coeff * b.assignment[state_vars[j]];
                        const_term += coeff * cfg.ark[r][j];
                    }
                    if !const_term.is_zero() {
                        lc.push((const_term, one));
                        val += const_term;
                    }
                    let v = b.new_var(val);
                    b.enforce_lc_times_one_eq_var(lc, v);
                    next_vars.push(v);
                }
            }

            if arith_mode == PoseidonArithMode::ReplayFixed {
                // Sanity against traced round state.
                let expected = &round_states[r];
                for i in 0..t {
                    if b.assignment[next_vars[i]] != expected[i] {
                        return Err(ReplayErr::Mismatch(format!(
                            "round {r} state mismatch at i={i} for permute #{permute_ptr}"
                        )));
                    }
                }
            }
            *state_vars = next_vars;
        }

        if arith_mode == PoseidonArithMode::ReplayFixed {
            // Final sanity against permute-after.
            for i in 0..t {
                if b.assignment[state_vars[i]] != after_state[i] {
                    return Err(ReplayErr::Mismatch(format!(
                        "after state mismatch at i={i} for permute #{permute_ptr}"
                    )));
                }
            }
        }
        permute_ptr += 1;

        Ok(())
    };

    for op in ops {
        match op {
            PoseidonTraceOp::Absorb(elems) => {
                if elems.is_empty() {
                    continue;
                }
                let range_start = wiring.absorb_vars.len();
                let mut range_len = 0usize;
                for &e in elems {
                    // If we were squeezing, permute first.
                    if matches!(mode, ark_crypto_primitives::sponge::DuplexSpongeMode::Squeezing { .. }) {
                        apply_perm(&mut b, &mut state_vars)?;
                        mode = ark_crypto_primitives::sponge::DuplexSpongeMode::Absorbing { next_absorb_index: 0 };
                    }
                    let mut absorb_index = match mode {
                        ark_crypto_primitives::sponge::DuplexSpongeMode::Absorbing { next_absorb_index } => next_absorb_index,
                        _ => unreachable!(),
                    };
                    if absorb_index == cfg.rate {
                        apply_perm(&mut b, &mut state_vars)?;
                        absorb_index = 0;
                    }

                    let e_var = b.new_var(e);
                    if arith_mode == PoseidonArithMode::ReplayFixed {
                        // Replay mode: fix absorb inputs to the recorded trace values.
                        b.enforce_var_eq_const(e_var, e);
                    }
                    wiring.absorb_vars.push(e_var);
                    range_len += 1;

                    // Update one state slot: state[cap + absorb_index] += e
                    let pos = cfg.capacity + absorb_index;
                    let new_val = b.assignment[state_vars[pos]] + b.assignment[e_var];
                    let new_var = b.new_var(new_val);
                    b.enforce_lc_times_one_eq_var(vec![(F::ONE, state_vars[pos]), (F::ONE, e_var)], new_var);
                    state_vars[pos] = new_var;

                    mode = ark_crypto_primitives::sponge::DuplexSpongeMode::Absorbing {
                        next_absorb_index: absorb_index + 1,
                    };
                }
                wiring.absorb_ranges.push((range_start, range_len));
            }
            PoseidonTraceOp::SqueezeField(out) => {
                if out.is_empty() {
                    continue;
                }
                // If we were absorbing, permute first.
                let mut squeeze_index = match mode {
                    ark_crypto_primitives::sponge::DuplexSpongeMode::Absorbing { .. } => {
                        apply_perm(&mut b, &mut state_vars)?;
                        0
                    }
                    ark_crypto_primitives::sponge::DuplexSpongeMode::Squeezing { next_squeeze_index } => next_squeeze_index,
                };
                if squeeze_index == cfg.rate {
                    apply_perm(&mut b, &mut state_vars)?;
                    squeeze_index = 0;
                }

                let mut produced = 0usize;
                let range_start = wiring.squeeze_field_vars.len();
                while produced < out.len() {
                    let take = core::cmp::min(cfg.rate - squeeze_index, out.len() - produced);
                    for j in 0..take {
                        let pos = cfg.capacity + squeeze_index + j;
                        if arith_mode == PoseidonArithMode::ReplayFixed {
                            let expected = out[produced + j];
                            let v = b.new_var(expected);
                            b.enforce_var_eq_const(v, expected);
                            // v == state[pos]
                            b.enforce_var_eq_var(state_vars[pos], v);
                            wiring.squeeze_field_vars.push(v);
                        } else {
                            // WE mode: expose the state element as the squeeze output var.
                            wiring.squeeze_field_vars.push(state_vars[pos]);
                        }
                    }
                    produced += take;
                    squeeze_index += take;
                    if produced < out.len() && squeeze_index == cfg.rate {
                        apply_perm(&mut b, &mut state_vars)?;
                        squeeze_index = 0;
                    }
                }
                wiring.squeeze_field_ranges.push((range_start, out.len()));

                mode = ark_crypto_primitives::sponge::DuplexSpongeMode::Squeezing {
                    next_squeeze_index: squeeze_index,
                };
            }
            PoseidonTraceOp::SqueezeBytes { n, out } => {
                let usable_bytes = ((F::MODULUS_BIT_SIZE - 1) / 8) as usize;
                let num_elements = (*n + usable_bytes - 1) / usable_bytes;

                // Squeeze native field elements and constrain them (like SqueezeField),
                // then check the bytes in Rust and return them for later constraints.
                let mut squeeze_index = match mode {
                    ark_crypto_primitives::sponge::DuplexSpongeMode::Absorbing { .. } => {
                        apply_perm(&mut b, &mut state_vars)?;
                        0
                    }
                    ark_crypto_primitives::sponge::DuplexSpongeMode::Squeezing { next_squeeze_index } => next_squeeze_index,
                };
                if squeeze_index == cfg.rate {
                    apply_perm(&mut b, &mut state_vars)?;
                    squeeze_index = 0;
                }

                let mut src_vars: Vec<usize> = Vec::with_capacity(num_elements);
                let mut produced = 0usize;
                while produced < num_elements {
                    let take = core::cmp::min(cfg.rate - squeeze_index, num_elements - produced);
                    for j in 0..take {
                        let pos = cfg.capacity + squeeze_index + j;
                        // Use the state var directly as the squeezed element source.
                        src_vars.push(state_vars[pos]);
                    }
                    produced += take;
                    squeeze_index += take;
                    if produced < num_elements && squeeze_index == cfg.rate {
                        apply_perm(&mut b, &mut state_vars)?;
                        squeeze_index = 0;
                    }
                }

                if arith_mode == PoseidonArithMode::ReplayFixed {
                    // Compute bytes from witness values and check they match recorded.
                    let mut bytes: Vec<u8> = Vec::with_capacity(usable_bytes * num_elements);
                    for &v in &src_vars {
                        let elem_bytes = b.assignment[v].into_bigint().to_bytes_le();
                        bytes.extend_from_slice(&elem_bytes[..usable_bytes]);
                    }
                    bytes.truncate(*n);
                    if &bytes != out {
                        return Err(ReplayErr::Mismatch("SqueezeBytes bytes mismatch".to_string()));
                    }
                }

                byte_witnesses.push(ByteSqueezeWitness {
                    n: *n,
                    usable_bytes,
                    src_elems: src_vars.clone(),
                    out: out.clone(),
                });

                if with_bytes {
                    // Allocate and constrain byte vars + link to src elements.
                    //
                    // We allocate the **full** usable_bytes*num_elements bytes (even if n truncates
                    // mid-element) so that every src element is linked to its low bytes.
                    let range_start = byte_wiring.squeeze_byte_vars.len();
                    let full_len = usable_bytes * num_elements;

                    // IMPORTANT (soundness): in WE mode, the squeezed bytes must be a *function*
                    // of the sponge state, otherwise FS coins derived from bytes are forgeable.
                    //
                    // Therefore we always derive the initial byte witness from the current
                    // squeezed field elements (like replay mode). The constraints below then
                    // enforce that these bytes are the *canonical* low bytes of the field element
                    // (i.e. of its integer representative in [0, p)).
                    let mut full_bytes: Vec<u8> = Vec::with_capacity(full_len);
                    for &v in &src_vars {
                        let elem_bytes = b.assignment[v].into_bigint().to_bytes_le();
                        full_bytes.extend_from_slice(&elem_bytes[..usable_bytes]);
                    }
                    debug_assert_eq!(full_bytes.len(), full_len);

                    // Allocate byte variables in the same order as `full_bytes`.
                    // We will expose the first `n` bytes as transcript output vars (linked to state).
                    let mut op_byte_vars: Vec<usize> = Vec::with_capacity(full_len);

                    // Constants for radix-256 recomposition.
                    let mut pow256: Vec<F> = Vec::with_capacity(usable_bytes);
                    let mut acc = F::ONE;
                    let base = F::from(256u64);
                    for _ in 0..usable_bytes {
                        pow256.push(acc);
                        acc *= base;
                    }
                    let pow256k = acc; // 256^{usable_bytes}

                    // Modulus decomposition: p = p0 + 256^k * p_hi, where k = usable_bytes.
                    //
                    // We enforce canonicality of the byte decomposition by constraining that the
                    // reconstructed integer representation is < p. Since 256^k < p (by choice of
                    // usable_bytes), this makes the low bytes uniquely determined by the field element.
                    let mut p_bytes = F::MODULUS.to_bytes_le();
                    // Ensure we have enough bytes for indexing.
                    if p_bytes.len() < usable_bytes + 2 {
                        p_bytes.resize(usable_bytes + 2, 0u8);
                    }
                    let p0_bytes = &p_bytes[..usable_bytes];
                    let p0_minus1_val = {
                        // p0 is the low k bytes of p (an integer < 256^k < p), so subtraction by 1 is safe.
                        // Compute p0 as a field element via radix-256 evaluation, then subtract 1 in F.
                        let mut p0 = F::ZERO;
                        for (i, &bb) in p0_bytes.iter().enumerate() {
                            p0 += pow256[i] * F::from(bb as u64);
                        }
                        p0 - F::ONE
                    };
                    let p_hi_u16: u16 = {
                        // p_hi fits in at most 16 bits because `usable_bytes = floor((bits-1)/8)` implies
                        // 8*k >= bits-8, hence p >> (8*k) < 2^8.
                        let mut v: u16 = 0;
                        let mut shift = 0u16;
                        for i in 0..2 {
                            v |= (p_bytes[usable_bytes + i] as u16) << shift;
                            shift += 8;
                        }
                        v
                    };
                    let p_hi_f = F::from(p_hi_u16 as u64);

                    for e in 0..num_elements {
                        // Allocate byte vars for this element.
                        let mut byte_vars: Vec<usize> = Vec::with_capacity(usable_bytes);
                        let mut byte_bits: Vec<[usize; 8]> = Vec::with_capacity(usable_bytes);
                        for i in 0..usable_bytes {
                            let bval = F::from(full_bytes[e * usable_bytes + i] as u64);
                            let bv = b.new_var(bval);
                            if arith_mode == PoseidonArithMode::ReplayFixed {
                                b.enforce_var_eq_const(bv, bval);
                            } else {
                                // WE mode: enforce byte is canonical (0..255) via bit decomposition.
                                let mut bits: [usize; 8] = [0usize; 8];
                                for bi in 0..8 {
                                    let bit = ((full_bytes[e * usable_bytes + i] >> bi) & 1) as u64;
                                    let vbit = b.new_var(if bit == 1 { F::ONE } else { F::ZERO });
                                    // boolean: vbit * (1 - vbit) = 0
                                    b.add_constraint(
                                        vec![(F::ONE, vbit)],
                                        vec![(F::ONE, one), (-F::ONE, vbit)],
                                        vec![(F::ZERO, one)],
                                    );
                                    bits[bi] = vbit;
                                }
                                // bv == Σ 2^j * bits[j]
                                let mut lc: Vec<(F, usize)> = Vec::with_capacity(8);
                                let mut p2 = F::ONE;
                                for &vbit in bits.iter() {
                                    lc.push((p2, vbit));
                                    p2 = p2.double();
                                }
                                b.enforce_lc_times_one_eq_var(lc, bv);
                                byte_bits.push(bits);
                            }
                            byte_vars.push(bv);
                            op_byte_vars.push(bv);
                        }

                        // Link src element to its canonical low bytes:
                        //   src = low + 256^k * high
                        // and enforce that the reconstructed integer (low + 256^k*high) is < p.
                        //
                        // Without the <p canonicality check, the low bytes are not uniquely
                        // determined by the field element (bytes could be chosen arbitrarily with
                        // a compensating high), which breaks FS binding in WE mode.
                        let src = src_vars[e];
                        let mut low = F::ZERO;
                        for i in 0..usable_bytes {
                            low += pow256[i] * b.assignment[byte_vars[i]];
                        }
                        // Materialize `low` as a variable.
                        let mut low_lc: Vec<(F, usize)> = Vec::with_capacity(usable_bytes);
                        for i in 0..usable_bytes {
                            low_lc.push((pow256[i], byte_vars[i]));
                        }
                        let low_var = b.new_var(low);
                        b.enforce_lc_times_one_eq_var(low_lc, low_var);

                        // Compute and allocate `high` from the current witness values.
                        let high_val = (b.assignment[src] - b.assignment[low_var]) * pow256k.inverse().unwrap();
                        let high = b.new_var(high_val);

                        // src - Σ 256^i*byte_i - 256^k*high = 0
                        let mut lc: Vec<(F, usize)> = Vec::with_capacity(2 + usable_bytes);
                        lc.push((F::ONE, src));
                        for i in 0..usable_bytes {
                            lc.push((-pow256[i], byte_vars[i]));
                        }
                        lc.push((-pow256k, high));
                        let z = b.new_var(F::ZERO);
                        b.enforce_var_eq_const(z, F::ZERO);
                        b.enforce_lc_times_one_eq_var(lc, z);

                        if arith_mode != PoseidonArithMode::ReplayFixed {
                            // Canonicality constraints for WE mode.
                            //
                            // 1) Range-check high as an 8-bit integer via bit decomposition.
                            //    (In practice `high` is < 256 because k = floor((bits-1)/8).)
                            let mut high_bits: [usize; 8] = [0usize; 8];
                            for bi in 0..8 {
                                let bit = ((p0_minus1_val.into_bigint().to_bytes_le()[0] >> bi) & 1) as u64;
                                // Use the computed high witness to initialize bits.
                                let hb = ((high_val.into_bigint().to_bytes_le()[0] >> bi) & 1) as u64;
                                let vbit = b.new_var(if hb == 1 { F::ONE } else { F::ZERO });
                                b.add_constraint(
                                    vec![(F::ONE, vbit)],
                                    vec![(F::ONE, one), (-F::ONE, vbit)],
                                    vec![(F::ZERO, one)],
                                );
                                high_bits[bi] = vbit;
                                let _ = bit; // silence unused (kept for potential debugging)
                            }
                            // high == Σ 2^j * high_bits[j]
                            let mut lc_h: Vec<(F, usize)> = Vec::with_capacity(8);
                            let mut p2 = F::ONE;
                            for &vbit in high_bits.iter() {
                                lc_h.push((p2, vbit));
                                p2 = p2.double();
                            }
                            b.enforce_lc_times_one_eq_var(lc_h, high);

                            // 2) Enforce high <= p_hi by introducing an 8-bit slack:
                            //    high + slack = p_hi.
                            let slack_val_u64 = (p_hi_u16 as i64 - (high_val.into_bigint().to_bytes_le()[0] as i64)).max(0) as u64;
                            let slack = b.new_var(F::from(slack_val_u64));
                            // Range-check slack as a byte.
                            let mut slack_bits: [usize; 8] = [0usize; 8];
                            for bi in 0..8 {
                                let sb = ((slack_val_u64 >> bi) & 1) as u64;
                                let vbit = b.new_var(if sb == 1 { F::ONE } else { F::ZERO });
                                b.add_constraint(
                                    vec![(F::ONE, vbit)],
                                    vec![(F::ONE, one), (-F::ONE, vbit)],
                                    vec![(F::ZERO, one)],
                                );
                                slack_bits[bi] = vbit;
                            }
                            // slack == Σ 2^j * slack_bits[j]
                            let mut lc_s: Vec<(F, usize)> = Vec::with_capacity(8);
                            let mut p2s = F::ONE;
                            for &vbit in slack_bits.iter() {
                                lc_s.push((p2s, vbit));
                                p2s = p2s.double();
                            }
                            b.enforce_lc_times_one_eq_var(lc_s, slack);
                            // high + slack == p_hi
                            let lc_hs = vec![(F::ONE, high), (F::ONE, slack), (-p_hi_f, one)];
                            let z2 = b.new_var(F::ZERO);
                            b.enforce_var_eq_const(z2, F::ZERO);
                            b.enforce_lc_times_one_eq_var(lc_hs, z2);

                            // 3) If slack == 0 (i.e. high == p_hi), enforce low <= p0-1.
                            // Compute is_eq = Π_i (1 - slack_bit_i), which is 1 iff slack == 0.
                            let mut is_eq = b.one();
                            for &sb in slack_bits.iter() {
                                // t = 1 - sb
                                let t_val = F::ONE - b.assignment[sb];
                                let t = b.new_var(t_val);
                                b.enforce_lc_times_one_eq_var(vec![(F::ONE, one), (-F::ONE, sb)], t);
                                // is_eq = is_eq * t
                                let prod_val = b.assignment[is_eq] * b.assignment[t];
                                let prod = b.new_var(prod_val);
                                b.enforce_mul(is_eq, t, prod);
                                is_eq = prod;
                            }

                            // diff = (p0-1 - low) * is_eq, and diff must be in [0, 256^k).
                            let tmp_val = p0_minus1_val - b.assignment[low_var];
                            let tmp = b.new_var(tmp_val);
                            b.enforce_lc_times_one_eq_var(vec![(p0_minus1_val, one), (-F::ONE, low_var)], tmp);
                            let diff_val = b.assignment[tmp] * b.assignment[is_eq];
                            let diff = b.new_var(diff_val);
                            b.enforce_mul(tmp, is_eq, diff);

                            // Range-check `diff` by decomposing it into `usable_bytes` bytes.
                            let mut diff_byte_vars: Vec<usize> = Vec::with_capacity(usable_bytes);
                            let diff_bytes_le = diff_val.into_bigint().to_bytes_le();
                            for i in 0..usable_bytes {
                                let bb = diff_bytes_le.get(i).copied().unwrap_or(0u8);
                                let bval = F::from(bb as u64);
                                let bv = b.new_var(bval);
                                // byte range check via bits
                                let mut bits: [usize; 8] = [0usize; 8];
                                for bi in 0..8 {
                                    let bit = ((bb >> bi) & 1) as u64;
                                    let vbit = b.new_var(if bit == 1 { F::ONE } else { F::ZERO });
                                    b.add_constraint(
                                        vec![(F::ONE, vbit)],
                                        vec![(F::ONE, one), (-F::ONE, vbit)],
                                        vec![(F::ZERO, one)],
                                    );
                                    bits[bi] = vbit;
                                }
                                let mut lc: Vec<(F, usize)> = Vec::with_capacity(8);
                                let mut p2 = F::ONE;
                                for &vbit in bits.iter() {
                                    lc.push((p2, vbit));
                                    p2 = p2.double();
                                }
                                b.enforce_lc_times_one_eq_var(lc, bv);
                                diff_byte_vars.push(bv);
                            }
                            // diff == Σ 256^i * diff_byte_i
                            let mut lc_d: Vec<(F, usize)> = Vec::with_capacity(usable_bytes);
                            let mut val = F::ZERO;
                            for i in 0..usable_bytes {
                                lc_d.push((pow256[i], diff_byte_vars[i]));
                                val += pow256[i] * b.assignment[diff_byte_vars[i]];
                            }
                            let diff_recomp = b.new_var(val);
                            b.enforce_lc_times_one_eq_var(lc_d, diff_recomp);
                            // Enforce diff == diff_recomp
                            b.enforce_var_eq_var(diff, diff_recomp);

                            // `byte_bits` is kept for potential future strengthening/debug; silence warnings.
                            let _ = &byte_bits;
                        }
                    }

                    // Expose only the first n bytes in trace order (truncated), as transcript output vars.
                    // These are already linked to the squeezed field elements through the constraints above.
                    for i in 0..*n {
                        byte_wiring.squeeze_byte_vars.push(op_byte_vars[i]);
                    }
                    byte_wiring.squeeze_byte_ranges.push((range_start, *n));
                }

                mode = ark_crypto_primitives::sponge::DuplexSpongeMode::Squeezing {
                    next_squeeze_index: squeeze_index,
                };
            }
        }
    }

    if arith_mode == PoseidonArithMode::ReplayFixed && permute_ptr != replay.permutes.len() {
        return Err(ReplayErr::Invalid(format!(
            "permute count mismatch: used {permute_ptr}, replay has {}",
            replay.permutes.len()
        )));
    }

    let (inst_any, assignment, ckpts) = if b.is_file_backed() {
        #[cfg(unix)]
        {
            if matches!(b.file_sink, Some(PoseidonFileSink::Range(_))) {
                let (asg, ckpts) = b.into_range_result().map_err(ReplayErr::Invalid)?;
                let inst = SparseDr1csInstance {
                    nvars: asg.len(),
                    constraints: Vec::new(),
                    a_terms: Vec::new(),
                    b_terms: Vec::new(),
                    c_terms: Vec::new(),
                };
                (PoseidonInstance::InMemory(inst), asg, Some(ckpts))
            } else if matches!(b.file_sink, Some(PoseidonFileSink::Count)) {
                // Count-only: no IO, but we still computed full assignment (and counts in b.file_*).
                let (asg, counts) = b.into_count_result().map_err(ReplayErr::Invalid)?;
                let inst = SparseDr1csInstance {
                    nvars: asg.len(),
                    constraints: Vec::new(),
                    a_terms: Vec::new(),
                    b_terms: Vec::new(),
                    c_terms: Vec::new(),
                };
                (PoseidonInstance::InMemory(inst), asg, Some(counts))
            } else {
                let (inst, asg) = b.into_file_backed_instance().map_err(ReplayErr::Invalid)?;
                (PoseidonInstance::FileBacked(inst), asg, None)
            }
        }
        #[cfg(not(unix))]
        {
            let (inst, asg) = b.into_file_backed_instance().map_err(ReplayErr::Invalid)?;
            (PoseidonInstance::FileBacked(inst), asg, None)
        }
    } else {
        let (inst, asg) = b.into_sparse_instance();
        (PoseidonInstance::InMemory(inst), asg, None)
    };
    let final_state_vars = state_vars;

    Ok((
        inst_any,
        assignment,
        replay,
        byte_witnesses,
        wiring,
        byte_wiring,
        init_state_vars,
        final_state_vars,
        ckpts,
    ))
}

fn poseidon_sponge_dr1cs_from_trace_impl<F: PrimeField + CanonicalSerialize>(
    cfg: &ark_crypto_primitives::sponge::poseidon::PoseidonConfig<F>,
    ops: &[PoseidonTraceOp<F>],
    with_bytes: bool,
    arith_mode: PoseidonArithMode,
) -> Result<
    (
        SparseDr1csInstance<F>,
        Vec<F>,
        PoseidonSpongeReplayResult<F>,
        Vec<ByteSqueezeWitness>,
        PoseidonDr1csWiring,
        PoseidonByteWiring,
    ),
    ReplayErr,
> {
    let (inst_any, asg, replay, bytes, wiring, bw) =
        poseidon_sponge_dr1cs_from_trace_impl_any(cfg, ops, with_bytes, arith_mode, None)?;
    match inst_any {
        PoseidonInstance::InMemory(inst) => Ok((inst, asg, replay, bytes, wiring, bw)),
        PoseidonInstance::FileBacked(_) => Err(ReplayErr::Invalid(
            "expected in-memory instance, got file-backed".to_string(),
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_ff::UniformRand;
    use cyclotomic_rings::rings::FrogPoseidonConfig as PC;
    use stark_rings::cyclotomic_ring::models::frog_ring::RqPoly as R;
    use stark_rings::PolyRing;

    #[test]
    fn test_poseidon_perm_dr1cs_satisfies() {
        type BF = <<R as PolyRing>::BaseRing as ark_ff::Field>::BasePrimeField;
        let cfg = <PC as cyclotomic_rings::rings::GetPoseidonParams<BF>>::get_poseidon_config();

        let t = cfg.rate + cfg.capacity;
        let mut rng = ark_std::test_rng();
        let before = (0..t).map(|_| BF::rand(&mut rng)).collect::<Vec<_>>();

        let (inst, assignment, _out_vars) =
            poseidon_permutation_dr1cs::<BF>(&cfg, &before).unwrap();
        inst.check(&assignment).unwrap();
    }

    #[test]
    fn test_poseidon_sponge_dr1cs_from_real_trace_satisfies_constraints() {
        use ark_crypto_primitives::sponge::{
            poseidon::PoseidonSponge, CryptographicSponge, FieldBasedCryptographicSponge,
        };
        use stark_rings::PolyRing;

        type BF = <<R as PolyRing>::BaseRing as ark_ff::Field>::BasePrimeField;
        let cfg = <PC as cyclotomic_rings::rings::GetPoseidonParams<BF>>::get_poseidon_config();

        // Build a small synthetic ops trace directly from PoseidonSponge.
        let mut rng = ark_std::test_rng();
        let mut sponge = PoseidonSponge::<BF>::new(&cfg);
        let mut ops: Vec<PoseidonTraceOp<BF>> = Vec::new();

        let absorb1 = (0..(cfg.rate + 3)).map(|_| BF::rand(&mut rng)).collect::<Vec<_>>();
        sponge.absorb(&absorb1);
        ops.push(PoseidonTraceOp::Absorb(absorb1.clone()));

        let out1 = sponge.squeeze_field_elements::<BF>(5);
        ops.push(PoseidonTraceOp::SqueezeField(out1.clone()));
        sponge.absorb(&out1);
        ops.push(PoseidonTraceOp::Absorb(out1));

        let bytes = sponge.squeeze_bytes(17);
        ops.push(PoseidonTraceOp::SqueezeBytes { n: 17, out: bytes });

        let absorb2 = (0..7).map(|_| BF::rand(&mut rng)).collect::<Vec<_>>();
        sponge.absorb(&absorb2);
        ops.push(PoseidonTraceOp::Absorb(absorb2));

        let out2 = sponge.squeeze_native_field_elements(3);
        ops.push(PoseidonTraceOp::SqueezeField(out2.clone()));

        let (inst, assignment, _replay, _bytes) =
            poseidon_sponge_dr1cs_from_trace::<BF>(&cfg, &ops).expect("build dr1cs failed");
        inst.check(&assignment).unwrap();
    }

    #[test]
    fn test_we_mode_squeeze_bytes_are_constrained_by_state() {
        use ark_crypto_primitives::sponge::{
            poseidon::PoseidonSponge, CryptographicSponge,
        };
        use ark_ff::Field;
        use stark_rings::PolyRing;

        type BF = <<R as PolyRing>::BaseRing as ark_ff::Field>::BasePrimeField;
        let cfg = <PC as cyclotomic_rings::rings::GetPoseidonParams<BF>>::get_poseidon_config();

        // Build a small synthetic ops schedule and include a SqueezeBytes.
        let mut rng = ark_std::test_rng();
        let mut sponge = PoseidonSponge::<BF>::new(&cfg);
        let mut ops: Vec<PoseidonTraceOp<BF>> = Vec::new();

        let absorb = (0..(cfg.rate + 2)).map(|_| BF::rand(&mut rng)).collect::<Vec<_>>();
        sponge.absorb(&absorb);
        ops.push(PoseidonTraceOp::Absorb(absorb));

        let bytes = sponge.squeeze_bytes(17);
        ops.push(PoseidonTraceOp::SqueezeBytes { n: 17, out: bytes.clone() });

        // WE/arm-before-proof mode: IO is not fixed, but bytes must still be derived from state.
        let (inst, mut assignment, _replay, _byte_wit, _wiring, byte_wiring) =
            poseidon_sponge_dr1cs_from_ops_with_wiring_and_bytes::<BF>(&cfg, &ops)
                .expect("build we-mode dr1cs failed");
        inst.check(&assignment).unwrap();

        // Flip one squeezed byte var: should break satisfaction (bytes are no longer free).
        let v0 = *byte_wiring.squeeze_byte_vars.first().expect("missing squeeze bytes");
        assignment[v0] += BF::ONE;
        assert!(inst.check(&assignment).is_err());
    }
}

