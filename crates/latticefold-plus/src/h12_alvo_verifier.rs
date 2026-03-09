//! ALVO verifier compiler for the research H12 redesign.
//!
//! The current prototype keeps the cryptographic block-opening verification outside AADP and
//! compiles the *affine* local verifier relation over the verified opened packed `pi0` blocks.

use std::collections::{BTreeMap, BTreeSet};

use ark_ff::Field;
use dpp::dr1cs_flpcp::Dr1csProofLayoutInfo;
use dpp::theorem43::Theorem43AlvoLocalCheckSurface;
use latticefold::transcript::poseidon::F257;
use ark_std::Zero;

use crate::aadp_we::{AadpConstraintSystem, AadpLinearForm, AadpMulConstraint};
use crate::h12_pi_commit::{f257_to_u16, H12PiBlockOpening};

#[derive(Clone, Debug)]
pub struct H12AlvoCompiledConstraintSystem {
    pub pack_d: usize,
    pub pi0_len: usize,
    pub opened_packed_pi_blocks: Vec<usize>,
    pub opened_packed_block_lens: Vec<usize>,
    pub cs: AadpConstraintSystem<F257>,
}

pub fn compile_alvo_constraint_system(
    surfaces: &[Theorem43AlvoLocalCheckSurface<F257>],
    pack_d: usize,
) -> Result<H12AlvoCompiledConstraintSystem, String> {
    if surfaces.is_empty() {
        return Err("H12 ALVO requires at least one selected local check".to_string());
    }
    if pack_d == 0 {
        return Err("H12 ALVO requires pack_d >= 1".to_string());
    }

    let layout = require_consistent_layout(surfaces)?;
    let mut opened_blocks = BTreeSet::<usize>::new();
    for s in surfaces {
        for block in s.packed_pi_blocks_conservative(pack_d) {
            opened_blocks.insert(block);
        }
    }
    let opened_packed_pi_blocks: Vec<usize> = opened_blocks.into_iter().collect();
    let opened_packed_block_lens: Vec<usize> = opened_packed_pi_blocks
        .iter()
        .map(|&block_idx| block_len_for_index(layout.pi0_len, pack_d, block_idx))
        .collect();

    let mut pos_to_var = BTreeMap::<usize, usize>::new();
    let mut next_var = 0usize;
    for (&block_idx, &block_len) in opened_packed_pi_blocks
        .iter()
        .zip(opened_packed_block_lens.iter())
    {
        let block_start = block_idx
            .checked_mul(pack_d)
            .ok_or_else(|| "H12 ALVO block_start overflow".to_string())?;
        for off in 0..block_len {
            pos_to_var.insert(block_start + off, next_var);
            next_var += 1;
        }
    }

    fn mk_form(
        constant: F257,
        pi_terms: &[(usize, F257)],
        w_terms: &[(usize, F257)],
        w_eval_block_pi_offset: usize,
        pos_to_var: &BTreeMap<usize, usize>,
    ) -> AadpLinearForm<F257> {
        let mut terms = Vec::with_capacity(pi_terms.len() + w_terms.len());
        for &(pi_idx, coeff) in pi_terms {
            if coeff.is_zero() {
                continue;
            }
            if let Some(&var_idx) = pos_to_var.get(&pi_idx) {
                terms.push((var_idx, coeff));
            }
        }
        for &(pos, coeff) in w_terms {
            if coeff.is_zero() {
                continue;
            }
            let pi_idx = w_eval_block_pi_offset + pos;
            if let Some(&var_idx) = pos_to_var.get(&pi_idx) {
                terms.push((var_idx, coeff));
            }
        }
        AadpLinearForm { constant, terms }
    }

    fn mk_h_form(
        constant: F257,
        w_terms: &[(usize, F257)],
        w_eval_block_pi_offset: usize,
        pos_to_var: &BTreeMap<usize, usize>,
    ) -> AadpLinearForm<F257> {
        let mut terms = Vec::with_capacity(w_terms.len());
        for &(pos, coeff) in w_terms {
            if coeff.is_zero() {
                continue;
            }
            let pi_idx = w_eval_block_pi_offset + pos;
            if let Some(&var_idx) = pos_to_var.get(&pi_idx) {
                terms.push((var_idx, coeff));
            }
        }
        AadpLinearForm { constant, terms }
    }

    let h_constraints_n: usize = surfaces.iter().map(|s| s.h_w_eval_constraints.len()).sum();
    let mut constraints = Vec::with_capacity(surfaces.len() + h_constraints_n);
    for s in surfaces {
        let base = &s.local_check;
        let alpha = mk_form(
            base.q1_x_dot,
            base.q1_pi_terms.as_slice(),
            base.q1_w_terms.as_slice(),
            base.w_eval_block_pi_offset,
            &pos_to_var,
        );
        let beta = mk_form(
            base.q2_x_dot,
            base.q2_pi_terms.as_slice(),
            base.q2_w_terms.as_slice(),
            base.w_eval_block_pi_offset,
            &pos_to_var,
        );
        let gamma = mk_form(
            base.q3_x_dot_sparse,
            base.q3_pi_sparse_terms.as_slice(),
            base.q3_w_terms.as_slice(),
            base.w_eval_block_pi_offset,
            &pos_to_var,
        );
        constraints.push(AadpMulConstraint {
            a: alpha,
            b: beta,
            c: gamma,
            d: AadpLinearForm {
                constant: F257::ONE,
                terms: Vec::new(),
            },
        });
        for lc in &s.h_w_eval_constraints {
            let h_form = mk_h_form(
                lc.constant,
                lc.terms.as_slice(),
                base.w_eval_block_pi_offset,
                &pos_to_var,
            );
            constraints.push(AadpMulConstraint {
                a: h_form,
                b: AadpLinearForm {
                    constant: F257::ONE,
                    terms: Vec::new(),
                },
                c: AadpLinearForm {
                    constant: F257::ZERO,
                    terms: Vec::new(),
                },
                d: AadpLinearForm {
                    constant: F257::ONE,
                    terms: Vec::new(),
                },
            });
        }
    }

    Ok(H12AlvoCompiledConstraintSystem {
        pack_d,
        pi0_len: layout.pi0_len,
        opened_packed_pi_blocks,
        opened_packed_block_lens,
        cs: AadpConstraintSystem {
            num_variables: next_var,
            constraints,
        },
    })
}

impl H12AlvoCompiledConstraintSystem {
    pub fn witness_from_pi0(&self, pi0: &[F257]) -> Result<Vec<F257>, String> {
        if pi0.len() != self.pi0_len {
            return Err(format!(
                "H12 ALVO pi0 length mismatch: got={} expected={}",
                pi0.len(),
                self.pi0_len
            ));
        }
        let mut out = Vec::with_capacity(self.cs.num_variables);
        for (&block_idx, &block_len) in self
            .opened_packed_pi_blocks
            .iter()
            .zip(self.opened_packed_block_lens.iter())
        {
            let start = block_idx * self.pack_d;
            out.extend_from_slice(&pi0[start..start + block_len]);
        }
        Ok(out)
    }

    pub fn witness_from_openings(&self, openings: &[H12PiBlockOpening]) -> Result<Vec<F257>, String> {
        let mut by_block = BTreeMap::<usize, Vec<F257>>::new();
        for opening in openings {
            let block_idx = opening.block_index as usize;
            if by_block.contains_key(&block_idx) {
                return Err(format!("H12 ALVO duplicate opening for block {block_idx}"));
            }
            let expected_len = block_len_for_index(self.pi0_len, self.pack_d, block_idx);
            if opening.values.len() != expected_len {
                return Err(format!(
                    "H12 ALVO opening length mismatch: block={} got={} expected={}",
                    block_idx,
                    opening.values.len(),
                    expected_len
                ));
            }
            by_block.insert(
                block_idx,
                opening
                    .values
                    .iter()
                    .copied()
                    .map(|x| F257::from((x % 257) as u64))
                    .collect(),
            );
        }
        let mut out = Vec::with_capacity(self.cs.num_variables);
        for (&block_idx, &block_len) in self
            .opened_packed_pi_blocks
            .iter()
            .zip(self.opened_packed_block_lens.iter())
        {
            let vals = by_block
                .get(&block_idx)
                .ok_or_else(|| format!("H12 ALVO missing opening for block {block_idx}"))?;
            if vals.len() != block_len {
                return Err(format!(
                    "H12 ALVO opening block length mismatch after decode: block={} got={} expected={}",
                    block_idx,
                    vals.len(),
                    block_len
                ));
            }
            out.extend_from_slice(vals.as_slice());
        }
        Ok(out)
    }
}

pub fn verify_alvo_linear_h_j(
    surface: &Theorem43AlvoLocalCheckSurface<F257>,
    w_eval_block: &[F257],
) -> Result<(), String> {
    if w_eval_block.len() != surface.local_check.w_eval_block_len {
        return Err(format!(
            "H12 ALVO H_j block length mismatch: got={} expected={}",
            w_eval_block.len(),
            surface.local_check.w_eval_block_len
        ));
    }
    for (ci, lc) in surface.h_w_eval_constraints.iter().enumerate() {
        let mut acc = lc.constant;
        for &(pos, coeff) in &lc.terms {
            let v = w_eval_block.get(pos).ok_or_else(|| {
                format!("H12 ALVO H_j term out of range: constraint={ci} pos={pos}")
            })?;
            acc += coeff * *v;
        }
        if !acc.is_zero() {
            return Err(format!("H12 ALVO H_j constraint failed at index {ci}"));
        }
    }
    Ok(())
}

pub fn verify_alvo_surface_relation(
    surface: &Theorem43AlvoLocalCheckSurface<F257>,
    opening_blocks: &BTreeMap<usize, Vec<F257>>,
) -> Result<(), String> {
    let block_values = reconstruct_pi0_slice_from_openings(
        opening_blocks,
        surface.local_check.w_eval_block_pi_offset,
        surface.local_check.w_eval_block_len,
        crate::h12_rcap::H12_RCAP_PACK_D,
    )?;
    verify_alvo_linear_h_j(surface, block_values.as_slice())?;
    let compiled =
        compile_alvo_constraint_system(std::slice::from_ref(surface), crate::h12_rcap::H12_RCAP_PACK_D)?;
    let mut openings = Vec::new();
    for &need_block in &compiled.opened_packed_pi_blocks {
        let vals = opening_blocks
            .get(&need_block)
            .ok_or_else(|| format!("H12 ALVO missing opening block {need_block}"))?;
        openings.push(H12PiBlockOpening {
            block_index: need_block as u32,
            values: vals.iter().copied().map(f257_to_u16).collect(),
        });
    }
    let witness = compiled.witness_from_openings(openings.as_slice())?;
    compiled.cs.check_witness(witness.as_slice())
}

fn reconstruct_pi0_slice_from_openings(
    opening_blocks: &BTreeMap<usize, Vec<F257>>,
    start_pi_idx: usize,
    len: usize,
    pack_d: usize,
) -> Result<Vec<F257>, String> {
    if pack_d == 0 {
        return Err("H12 ALVO reconstruct slice requires pack_d >= 1".to_string());
    }
    let mut out = Vec::with_capacity(len);
    for rel in 0..len {
        let pi_idx = start_pi_idx + rel;
        let block_idx = pi_idx / pack_d;
        let pos = pi_idx % pack_d;
        let block = opening_blocks
            .get(&block_idx)
            .ok_or_else(|| format!("H12 ALVO missing opening block {block_idx} for pi_idx={pi_idx}"))?;
        let v = block.get(pos).ok_or_else(|| {
            format!(
                "H12 ALVO opening block too short for pi_idx={} block={} pos={} block_len={}",
                pi_idx,
                block_idx,
                pos,
                block.len()
            )
        })?;
        out.push(*v);
    }
    Ok(out)
}

fn require_consistent_layout(
    surfaces: &[Theorem43AlvoLocalCheckSurface<F257>],
) -> Result<Dr1csProofLayoutInfo, String> {
    let first = surfaces
        .first()
        .ok_or_else(|| "H12 ALVO missing surfaces".to_string())?
        .proof_layout
        .clone();
    for s in surfaces.iter().skip(1) {
        let l = &s.proof_layout;
        if l.n != first.n
            || l.n_total != first.n_total
            || l.z_w_len != first.z_w_len
            || l.pi0_len != first.pi0_len
            || l.blocks != first.blocks
            || l.ell_local != first.ell_local
            || l.k_star != first.k_star
            || l.witness_positions_star != first.witness_positions_star
        {
            return Err("H12 ALVO surfaces disagree on proof layout".to_string());
        }
    }
    Ok(first)
}

fn block_len_for_index(pi0_len: usize, pack_d: usize, block_index: usize) -> usize {
    let start = block_index.saturating_mul(pack_d);
    if start >= pi0_len {
        return 0;
    }
    (pi0_len - start).min(pack_d)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_ff::Field;
    use dpp::dr1cs_flpcp::Dr1csProofLayoutInfo;
    use dpp::theorem43::{Theorem43AlvoLocalCheckSurface, Theorem43CapsuleLocalCheckSurface, Theorem43Coins};

    fn mk_surface(block_id: usize, rep_id: u64) -> Theorem43AlvoLocalCheckSurface<F257> {
        let local = Theorem43CapsuleLocalCheckSurface {
            block_id,
            rep_id,
            coins: Theorem43Coins {
                idx: 0,
                lambda: F257::ONE,
                rho: F257::ONE,
                sigma: F257::ONE,
                c_hit: F257::ONE,
            },
            q1_pi_terms: vec![(1, F257::ONE)],
            q2_pi_terms: vec![(2, F257::ONE)],
            q3_pi_sparse_terms: vec![(3, F257::ONE)],
            q1_x_dot: F257::ZERO,
            q2_x_dot: F257::ZERO,
            q3_x_dot_sparse: F257::ZERO,
            q1_w_terms: vec![(0, F257::ONE)],
            q2_w_terms: vec![(1, F257::ONE)],
            q3_w_terms: vec![(2, F257::ONE)],
            w_eval_block_pi_offset: 8,
            w_eval_block_len: 4,
            requires_dense_q3_w_eval: true,
        };
        Theorem43AlvoLocalCheckSurface {
            touched_pi_positions_exact: local.touched_pi_positions(false),
            touched_pi_positions_conservative: local.touched_pi_positions(true),
            local_check: local,
            proof_layout: Dr1csProofLayoutInfo {
                n: 4,
                n_total: 8,
                z_w_len: 8,
                pi0_len: 16,
                blocks: 2,
                ell_local: 4,
                k_star: 4,
                low_cube_len: 2,
                witness_positions_star: vec![0, 1, 2, 3],
            },
            h_w_eval_constraints: Vec::new(),
        }
    }

    #[test]
    fn test_compile_alvo_constraint_system_extracts_block_witness() {
        let surfaces = vec![mk_surface(0, 7)];
        let compiled = compile_alvo_constraint_system(surfaces.as_slice(), 4).expect("compile alvo");
        assert_eq!(compiled.opened_packed_pi_blocks, vec![0, 2]);
        let pi0: Vec<F257> = (0u64..16u64).map(F257::from).collect();
        let witness = compiled.witness_from_pi0(pi0.as_slice()).expect("witness from pi0");
        assert_eq!(witness.len(), compiled.cs.num_variables);
        assert_eq!(witness, vec![F257::from(0), F257::from(1), F257::from(2), F257::from(3), F257::from(8), F257::from(9), F257::from(10), F257::from(11)]);
    }

    #[test]
    fn test_verify_alvo_surface_relation_reconstructs_full_w_eval_across_packed_blocks() {
        let local = Theorem43CapsuleLocalCheckSurface {
            block_id: 0,
            rep_id: 9,
            coins: Theorem43Coins {
                idx: 0,
                lambda: F257::ONE,
                rho: F257::ONE,
                sigma: F257::ONE,
                c_hit: F257::ONE,
            },
            q1_pi_terms: Vec::new(),
            q2_pi_terms: Vec::new(),
            q3_pi_sparse_terms: Vec::new(),
            q1_x_dot: F257::ZERO,
            q2_x_dot: F257::ZERO,
            q3_x_dot_sparse: F257::ZERO,
            q1_w_terms: Vec::new(),
            q2_w_terms: Vec::new(),
            q3_w_terms: Vec::new(),
            w_eval_block_pi_offset: 62,
            w_eval_block_len: 4,
            requires_dense_q3_w_eval: true,
        };
        let surface = Theorem43AlvoLocalCheckSurface {
            touched_pi_positions_exact: vec![62, 63, 64, 65],
            touched_pi_positions_conservative: vec![62, 63, 64, 65],
            local_check: local,
            proof_layout: Dr1csProofLayoutInfo {
                n: 4,
                n_total: 62,
                z_w_len: 62,
                pi0_len: 66,
                blocks: 1,
                ell_local: 4,
                k_star: 4,
                low_cube_len: 2,
                witness_positions_star: vec![0, 1, 2, 3],
            },
            h_w_eval_constraints: vec![
                dpp::dr1cs_flpcp::Dr1csBlockLinearConstraint {
                    constant: F257::ZERO,
                    terms: vec![(3, F257::ONE), (0, -F257::ONE)],
                },
            ],
        };
        let mut opening_blocks = BTreeMap::<usize, Vec<F257>>::new();
        let mut block0 = (0u64..64u64).map(F257::from).collect::<Vec<_>>();
        block0[62] = F257::from(11u64);
        block0[63] = F257::from(12u64);
        opening_blocks.insert(0, block0);
        opening_blocks.insert(1, vec![F257::from(13u64), F257::from(11u64)]);
        verify_alvo_surface_relation(&surface, &opening_blocks)
            .expect("full w_eval block reconstructed across packed blocks");
    }
}
