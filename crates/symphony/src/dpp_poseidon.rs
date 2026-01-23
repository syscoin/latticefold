//! Poseidon-permutation arithmetization (prime-field dR1CS skeleton).
//!
//! This is an incremental step toward arithmetizing the full WE-facing verifier relation:
//! we first arithmetize the Poseidon permutation(s) used by the transcript.

use ark_ff::{BigInteger, PrimeField};
use rayon::prelude::*;

use crate::poseidon_trace::{permute_with_round_trace, PoseidonReplayError};
use crate::poseidon_trace::{replay_ops, PoseidonReplayError as ReplayErr, PoseidonSpongeReplayResult};
use crate::transcript::PoseidonTraceOp;

#[derive(Clone, Debug)]
pub struct Constraint<F: PrimeField> {
    pub a: Vec<(F, usize)>,
    pub b: Vec<(F, usize)>,
    pub c: Vec<(F, usize)>,
}

#[derive(Clone, Debug)]
pub struct SparseDr1csInstance<F: PrimeField> {
    pub nvars: usize,
    pub constraints: Vec<Constraint<F>>,
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
                let a = Self::eval_lc(&row.a, assignment);
                let b = Self::eval_lc(&row.b, assignment);
                let c = Self::eval_lc(&row.c, assignment);
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
            for (coeff, idx) in &row.a {
                ra[*idx] += *coeff;
            }
            for (coeff, idx) in &row.b {
                rb[*idx] += *coeff;
            }
            for (coeff, idx) in &row.c {
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
pub fn merge_sparse_dr1cs_share_one<F: PrimeField>(
    parts: &[(SparseDr1csInstance<F>, Vec<F>)],
) -> Result<(SparseDr1csInstance<F>, Vec<F>), String> {
    if parts.is_empty() {
        return Err("merge_sparse_dr1cs_share_one: empty parts".to_string());
    }

    let mut merged_assignment: Vec<F> = vec![F::ONE];
    let mut merged_constraints: Vec<Constraint<F>> = Vec::new();

    for (inst, asg) in parts {
        if asg.is_empty() || asg[0] != F::ONE {
            return Err("merge_sparse_dr1cs_share_one: each part must have assignment[0]=1".to_string());
        }
        if inst.nvars != asg.len() {
            return Err("merge_sparse_dr1cs_share_one: inst/assignment length mismatch".to_string());
        }

        // Map part var0 -> merged var0, and shift the rest by current offset.
        let offset = merged_assignment.len() - 1;
        let remap_idx = |idx: usize| -> usize { if idx == 0 { 0 } else { idx + offset } };

        for row in &inst.constraints {
            let remap_lc = |lc: &[(F, usize)]| -> Vec<(F, usize)> {
                lc.iter().map(|(c, i)| (*c, remap_idx(*i))).collect()
            };
            merged_constraints.push(Constraint {
                a: remap_lc(&row.a),
                b: remap_lc(&row.b),
                c: remap_lc(&row.c),
            });
        }

        // Append assignment sans constant slot.
        merged_assignment.extend_from_slice(&asg[1..]);
    }

    Ok((
        SparseDr1csInstance {
            nvars: merged_assignment.len(),
            constraints: merged_constraints,
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
    parts: &[(SparseDr1csInstance<F>, Vec<F>)],
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
    parts: &[(SparseDr1csInstance<F>, Vec<F>)],
    glue: &[(usize, usize, usize, usize)],
) -> Result<(SparseDr1csInstance<F>, Vec<F>), String> {
    merge_sparse_dr1cs_share_one_with_glue_impl(parts, glue, false)
}

fn merge_sparse_dr1cs_share_one_with_glue_impl<F: PrimeField>(
    parts: &[(SparseDr1csInstance<F>, Vec<F>)],
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
    for (inst, asg) in parts {
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
    let mut merged_constraints: Vec<Constraint<F>> = Vec::with_capacity(total_constraints);

    // Merge constraints with remapped indices.
    //
    // Default is a single-pass, allocation-friendly merge.
    //
    // For very large gates (multi-million constraints), this merge can become a wall-time
    // bottleneck and appear “single-core bound” in system monitors. In that case, we allow a
    // parallel remap that preserves constraint order and does not allocate *extra* constraints
    // beyond the final merged list (it still must allocate the final `merged_constraints`).
    let use_parallel_merge = total_constraints >= 2_000_000;

    if use_parallel_merge {
        let remapped_parts: Vec<Vec<Constraint<F>>> = parts
            .par_iter()
            .enumerate()
            .map(|(part_idx, (inst, _asg))| {
                let offset = offsets[part_idx];
                let remap_idx = |idx: usize| -> usize {
                    let g = if idx == 0 { 0 } else { idx + offset };
                    new_index[g]
                };
                let remap_lc = |lc: &[(F, usize)]| -> Vec<(F, usize)> {
                    let mut out = Vec::with_capacity(lc.len());
                    for (c, i) in lc {
                        out.push((*c, remap_idx(*i)));
                    }
                    out
                };
                inst.constraints
                    .par_iter()
                    .map(|row| Constraint {
                        a: remap_lc(&row.a),
                        b: remap_lc(&row.b),
                        c: remap_lc(&row.c),
                    })
                    .collect::<Vec<_>>()
            })
            .collect();

        for v in remapped_parts {
            merged_constraints.extend(v);
        }
    } else {
        for (part_idx, (inst, _asg)) in parts.iter().enumerate() {
            let offset = offsets[part_idx];
            let remap_idx = |idx: usize| -> usize {
                let g = if idx == 0 { 0 } else { idx + offset };
                new_index[g]
            };
            for row in &inst.constraints {
                let remap_lc = |lc: &[(F, usize)]| -> Vec<(F, usize)> {
                    let mut out = Vec::with_capacity(lc.len());
                    for (c, i) in lc {
                        out.push((*c, remap_idx(*i)));
                    }
                    out
                };
                merged_constraints.push(Constraint {
                    a: remap_lc(&row.a),
                    b: remap_lc(&row.b),
                    c: remap_lc(&row.c),
                });
            }
        }
    }

    Ok((
        SparseDr1csInstance {
            nvars: new_assignment.len(),
            constraints: merged_constraints,
        },
        new_assignment,
    ))
}

#[derive(Clone, Debug)]
struct Dr1csBuilder<F: PrimeField> {
    assignment: Vec<F>,
    rows: Vec<Constraint<F>>,
}

impl<F: PrimeField> Dr1csBuilder<F> {
    fn new() -> Self {
        // var 0 is the constant-1 slot
        Self {
            assignment: vec![F::ONE],
            rows: Vec::new(),
        }
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
        self.rows.push(Constraint { a, b, c });
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
        let nvars = self.assignment.len();
        let inst = SparseDr1csInstance {
            nvars,
            constraints: self.rows,
        };
        (inst, self.assignment)
    }
}

/// Build a sparse dR1CS instance for a single Poseidon permutation, given an input state.
///
/// Returns `(instance, assignment, out_state_var_indices)`.
pub fn poseidon_permutation_dr1cs<F: PrimeField>(
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
pub fn poseidon_sponge_dr1cs_from_trace<F: PrimeField>(
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
pub fn poseidon_sponge_dr1cs_from_trace_with_wiring<F: PrimeField>(
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
pub fn poseidon_sponge_dr1cs_from_ops_with_wiring_and_bytes<F: PrimeField>(
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

/// Like `poseidon_sponge_dr1cs_from_trace_with_wiring`, but also **arithmetizes `SqueezeBytes`**:
/// - allocates byte variables,
/// - constrains each byte is 8-bit,
/// - links bytes to the underlying squeezed field elements via radix-256 decomposition.
pub fn poseidon_sponge_dr1cs_from_trace_with_wiring_and_bytes<F: PrimeField>(
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PoseidonArithMode {
    /// “Replay mode”: enforce Absorb/Squeeze outputs match the recorded trace values and run
    /// sanity checks against a replay.
    ReplayFixed,
    /// “WE mode”: do not bake recorded trace values into constraints.
    WeWitness,
}

fn poseidon_sponge_dr1cs_from_trace_impl<F: PrimeField>(
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
    let mut b = Dr1csBuilder::<F>::new();
    let one = b.one();

    // Initial state is all zeros.
    let mut state_vars = Vec::with_capacity(t);
    for _ in 0..t {
        let v = b.new_var(F::ZERO);
        b.enforce_var_eq_const(v, F::ZERO);
        state_vars.push(v);
    }

    let mut mode = ark_crypto_primitives::sponge::DuplexSpongeMode::Absorbing {
        next_absorb_index: 0,
    };

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

                            // If high == p_hi, then we must have low <= p0-1 (i.e. low < p0).
                            //
                            // We enforce this with a bytewise comparator against the *constant*
                            // bound (p0-1) using the already-existing per-byte bit decompositions,
                            // instead of introducing a fresh `diff` value and re-decomposing it.
                            // This is substantially cheaper and keeps soundness.
                            //
                            // Build constant bytes for (p0 - 1) in little-endian order.
                            let mut bound_bytes: Vec<u8> = p0_bytes.to_vec();
                            // subtract 1 (since p0 > 0)
                            let mut carry: i16 = -1;
                            for bb in bound_bytes.iter_mut() {
                                let v = (*bb as i16) + carry;
                                if v < 0 {
                                    *bb = 255u8;
                                    carry = -1;
                                } else {
                                    *bb = v as u8;
                                    carry = 0;
                                }
                            }

                            // Compare the little-endian byte vector `byte_vars` (and their bits)
                            // to `bound_bytes` in big-endian order.
                            let mut eq_prefix = b.one(); // 1 iff all more-significant bytes equal
                            b.enforce_var_eq_const(eq_prefix, F::ONE);
                            let mut lt_total = b.new_var(F::ZERO);
                            b.enforce_var_eq_const(lt_total, F::ZERO);
                            for bi in (0..usable_bytes).rev() {
                                // Per-byte eq and lt against constant bound_bytes[bi].
                                let cb = bound_bytes[bi];
                                let cb_bits = (0..8).map(|j| ((cb >> j) & 1) as u8).collect::<Vec<_>>();

                                // eq_byte = Π_j (bit_j == cb_j)
                                let mut eq_byte = b.one();
                                b.enforce_var_eq_const(eq_byte, F::ONE);
                                for j in 0..8 {
                                    let bit = byte_bits[bi][j];
                                    let eq_bit = if cb_bits[j] == 1 {
                                        // eq_bit = bit
                                        bit
                                    } else {
                                        // eq_bit = 1 - bit
                                        let t_val = F::ONE - b.assignment[bit];
                                        let t = b.new_var(t_val);
                                        b.enforce_lc_times_one_eq_var(vec![(F::ONE, one), (-F::ONE, bit)], t);
                                        t
                                    };
                                    let prod_val = b.assignment[eq_byte] * b.assignment[eq_bit];
                                    let prod = b.new_var(prod_val);
                                    b.enforce_mul(eq_byte, eq_bit, prod);
                                    eq_byte = prod;
                                }

                                // lt_byte: standard “first differing bit” check from MSB down.
                                let mut lt_byte = b.new_var(F::ZERO);
                                b.enforce_var_eq_const(lt_byte, F::ZERO);
                                let mut eq_bit_prefix = b.one();
                                b.enforce_var_eq_const(eq_bit_prefix, F::ONE);
                                for j in (0..8).rev() {
                                    let bit = byte_bits[bi][j];
                                    let cbit = ((cb >> j) & 1) as u8;
                                    if cbit == 1 {
                                        // term = eq_bit_prefix * (1 - bit)
                                        let om_val = F::ONE - b.assignment[bit];
                                        let om = b.new_var(om_val);
                                        b.enforce_lc_times_one_eq_var(vec![(F::ONE, one), (-F::ONE, bit)], om);
                                        let term_val = b.assignment[eq_bit_prefix] * b.assignment[om];
                                        let term = b.new_var(term_val);
                                        b.enforce_mul(eq_bit_prefix, om, term);
                                        let sum_val = b.assignment[lt_byte] + b.assignment[term];
                                        let sum = b.new_var(sum_val);
                                        b.enforce_lc_times_one_eq_var(vec![(F::ONE, lt_byte), (F::ONE, term)], sum);
                                        lt_byte = sum;
                                        // update prefix: eq_bit_prefix *= bit
                                        let newp_val = b.assignment[eq_bit_prefix] * b.assignment[bit];
                                        let newp = b.new_var(newp_val);
                                        b.enforce_mul(eq_bit_prefix, bit, newp);
                                        eq_bit_prefix = newp;
                                    } else {
                                        // cbit == 0: update prefix *= (1 - bit)
                                        let om_val = F::ONE - b.assignment[bit];
                                        let om = b.new_var(om_val);
                                        b.enforce_lc_times_one_eq_var(vec![(F::ONE, one), (-F::ONE, bit)], om);
                                        let newp_val = b.assignment[eq_bit_prefix] * b.assignment[om];
                                        let newp = b.new_var(newp_val);
                                        b.enforce_mul(eq_bit_prefix, om, newp);
                                        eq_bit_prefix = newp;
                                    }
                                }

                                // lt_total += eq_prefix * lt_byte
                                let term_val = b.assignment[eq_prefix] * b.assignment[lt_byte];
                                let term = b.new_var(term_val);
                                b.enforce_mul(eq_prefix, lt_byte, term);
                                let sum_val = b.assignment[lt_total] + b.assignment[term];
                                let sum = b.new_var(sum_val);
                                b.enforce_lc_times_one_eq_var(vec![(F::ONE, lt_total), (F::ONE, term)], sum);
                                lt_total = sum;

                                // eq_prefix *= eq_byte
                                let new_eq_val = b.assignment[eq_prefix] * b.assignment[eq_byte];
                                let new_eq = b.new_var(new_eq_val);
                                b.enforce_mul(eq_prefix, eq_byte, new_eq);
                                eq_prefix = new_eq;
                            }

                            // ok = (low <= bound) = lt_total + eq_prefix (disjoint).
                            let ok_val = b.assignment[lt_total] + b.assignment[eq_prefix];
                            let ok = b.new_var(ok_val);
                            b.enforce_lc_times_one_eq_var(vec![(F::ONE, lt_total), (F::ONE, eq_prefix)], ok);

                            // Enforce: if is_eq==1 then ok==1  ⇔  is_eq * (1 - ok) = 0.
                            let one_minus_ok_val = F::ONE - b.assignment[ok];
                            let one_minus_ok = b.new_var(one_minus_ok_val);
                            b.enforce_lc_times_one_eq_var(vec![(F::ONE, one), (-F::ONE, ok)], one_minus_ok);
                            let viol_val = b.assignment[is_eq] * b.assignment[one_minus_ok];
                            let viol = b.new_var(viol_val);
                            b.enforce_mul(is_eq, one_minus_ok, viol);
                            b.enforce_var_eq_const(viol, F::ZERO);
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

    let (inst, assignment) = b.into_sparse_instance();
    Ok((inst, assignment, replay, byte_witnesses, wiring, byte_wiring))
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

