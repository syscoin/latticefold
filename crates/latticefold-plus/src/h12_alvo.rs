//! Shared ALVO-stage metadata for the research H12 redesign.

use dpp::theorem43::{Theorem43AlvoLocalCheckSurface, Theorem43Coins};
use latticefold::transcript::poseidon::F257;
use sha2::Digest;

use crate::{h12_alvo_germ::H12GermProof, h12_pi_commit::f257_to_u16};

#[derive(Clone, Debug)]
pub struct H12AlvoClaim {
    pub alpha: u16,
    pub beta: u16,
    pub gamma: u16,
    pub alpha_pi_sparse: u16,
    pub beta_pi_sparse: u16,
    pub gamma_pi_sparse: u16,
}

#[derive(Clone, Debug)]
pub struct H12AlvoAnchorProjection {
    pub y_anchor_stream: u16,
    pub alpha_sparse: u16,
    pub beta_sparse: u16,
    pub delta_gamma_pi: u16,
}

#[derive(Clone, Debug)]
pub struct H12AlvoRepSurfaceBundle {
    pub rep_id: u64,
    pub anchor_block_id: usize,
    pub anchor_hint_blocks: Vec<usize>,
    pub anchor_projection: H12AlvoAnchorProjection,
    pub poison_blocks: usize,
    pub poison_gate: u16,
}

#[derive(Clone, Debug)]
pub struct H12AlvoLogicalLockAugmentation {
    pub share_index: u32,
    pub r_cap_reps: u16,
    pub stage1_root: [u8; 32],
    pub stage2_root: [u8; 32],
    pub schedule_digest: [u8; 32],
    pub surfaces: Vec<Theorem43AlvoLocalCheckSurface<F257>>,
    pub germ_proof: Option<H12GermProof>,
    pub rep_bundles: Vec<H12AlvoRepSurfaceBundle>,
}

pub fn digest_rep_bundle_hashes(rep_bundle_hashes: &[[u8; 32]], pack_d: usize) -> [u8; 32] {
    let mut h = sha2::Sha256::new();
    h.update(b"LFP_H12_ALVO_REP_BUNDLES_V2");
    h.update((pack_d as u32).to_le_bytes());
    h.update((rep_bundle_hashes.len() as u32).to_le_bytes());
    for hash in rep_bundle_hashes {
        h.update(hash);
    }
    h.finalize().into()
}

pub fn digest_alvo_schedule(
    surfaces: &[Theorem43AlvoLocalCheckSurface<F257>],
    pack_d: usize,
) -> [u8; 32] {
    let mut h = sha2::Sha256::new();
    h.update(b"LFP_H12_ALVO_SCHEDULE_V1");
    h.update((pack_d as u32).to_le_bytes());
    h.update((surfaces.len() as u32).to_le_bytes());
    for s in surfaces {
        digest_surface_into(&mut h, s, pack_d);
    }
    h.finalize().into()
}

pub fn derive_stage1_root(
    stmt_digest: &[F257; 32],
    lock_coin_seed: &[u8; 32],
    share_index: u32,
    r_cap_reps: u16,
    schedule_digest: &[u8; 32],
) -> [u8; 32] {
    let mut h = sha2::Sha256::new();
    h.update(b"LFP_H12_ALVO_STAGE1_ROOT_V1");
    for f in stmt_digest {
        h.update(f257_to_u16(*f).to_le_bytes());
    }
    h.update(lock_coin_seed);
    h.update(share_index.to_le_bytes());
    h.update(r_cap_reps.to_le_bytes());
    h.update(schedule_digest);
    h.finalize().into()
}

pub fn derive_stage2_root(lock: &H12AlvoLogicalLockAugmentation) -> [u8; 32] {
    let mut h = sha2::Sha256::new();
    h.update(b"LFP_H12_ALVO_STAGE2_ROOT_V5");
    h.update(lock.share_index.to_le_bytes());
    h.update(lock.r_cap_reps.to_le_bytes());
    h.update(lock.stage1_root);
    h.update(lock.schedule_digest);
    match &lock.germ_proof {
        Some(proof) => {
            h.update([1u8]);
            digest_germ_proof_into(&mut h, proof);
        }
        None => h.update([0u8]),
    }
    h.update((lock.rep_bundles.len() as u32).to_le_bytes());
    for rep in &lock.rep_bundles {
        h.update(rep.rep_id.to_le_bytes());
        h.update(rep.anchor_block_id.to_le_bytes());
        h.update((rep.anchor_hint_blocks.len() as u32).to_le_bytes());
        for &block_idx in &rep.anchor_hint_blocks {
            h.update(block_idx.to_le_bytes());
        }
        digest_anchor_projection_into(&mut h, &rep.anchor_projection);
        h.update(rep.poison_blocks.to_le_bytes());
        h.update(rep.poison_gate.to_le_bytes());
    }
    h.finalize().into()
}

fn digest_surface_into(
    h: &mut sha2::Sha256,
    surface: &Theorem43AlvoLocalCheckSurface<F257>,
    pack_d: usize,
) {
    let base = &surface.local_check;
    h.update(base.block_id.to_le_bytes());
    h.update(base.rep_id.to_le_bytes());
    digest_coins_into(h, &base.coins);
    digest_sparse_terms_into(h, base.q1_pi_terms.as_slice());
    digest_sparse_terms_into(h, base.q2_pi_terms.as_slice());
    digest_sparse_terms_into(h, base.q3_pi_sparse_terms.as_slice());
    h.update(f257_to_u16(base.q1_x_dot).to_le_bytes());
    h.update(f257_to_u16(base.q2_x_dot).to_le_bytes());
    h.update(f257_to_u16(base.q3_x_dot_sparse).to_le_bytes());
    digest_sparse_terms_into(h, base.q1_w_terms.as_slice());
    digest_sparse_terms_into(h, base.q2_w_terms.as_slice());
    digest_sparse_terms_into(h, base.q3_w_terms.as_slice());
    h.update(base.w_eval_block_pi_offset.to_le_bytes());
    h.update(base.w_eval_block_len.to_le_bytes());
    h.update([base.requires_dense_q3_w_eval as u8]);
    h.update(surface.proof_layout.n.to_le_bytes());
    h.update(surface.proof_layout.n_total.to_le_bytes());
    h.update(surface.proof_layout.z_w_len.to_le_bytes());
    h.update(surface.proof_layout.pi0_len.to_le_bytes());
    h.update(surface.proof_layout.blocks.to_le_bytes());
    h.update(surface.proof_layout.ell_local.to_le_bytes());
    h.update(surface.proof_layout.k_star.to_le_bytes());
    h.update(surface.proof_layout.low_cube_len.to_le_bytes());
    h.update((surface.proof_layout.witness_positions_star.len() as u32).to_le_bytes());
    for &pos in &surface.proof_layout.witness_positions_star {
        h.update(pos.to_le_bytes());
    }
    h.update((surface.h_w_eval_constraints.len() as u32).to_le_bytes());
    for lc in &surface.h_w_eval_constraints {
        h.update(f257_to_u16(lc.constant).to_le_bytes());
        h.update((lc.terms.len() as u32).to_le_bytes());
        for &(pos, coeff) in &lc.terms {
            h.update(pos.to_le_bytes());
            h.update(f257_to_u16(coeff).to_le_bytes());
        }
    }
    h.update((surface.touched_pi_positions_exact.len() as u32).to_le_bytes());
    for &pi_idx in &surface.touched_pi_positions_exact {
        h.update(pi_idx.to_le_bytes());
    }
    h.update((surface.touched_pi_positions_conservative.len() as u32).to_le_bytes());
    for &pi_idx in &surface.touched_pi_positions_conservative {
        h.update(pi_idx.to_le_bytes());
    }
    let exact_blocks = surface.packed_pi_blocks_exact(pack_d);
    h.update((exact_blocks.len() as u32).to_le_bytes());
    for block in exact_blocks {
        h.update(block.to_le_bytes());
    }
    let cons_blocks = surface.packed_pi_blocks_conservative(pack_d);
    h.update((cons_blocks.len() as u32).to_le_bytes());
    for block in cons_blocks {
        h.update(block.to_le_bytes());
    }
}

fn digest_coins_into(h: &mut sha2::Sha256, coins: &Theorem43Coins<F257>) {
    h.update(coins.idx.to_le_bytes());
    h.update(f257_to_u16(coins.lambda).to_le_bytes());
    h.update(f257_to_u16(coins.rho).to_le_bytes());
    h.update(f257_to_u16(coins.sigma).to_le_bytes());
    h.update(f257_to_u16(coins.c_hit).to_le_bytes());
}

fn digest_sparse_terms_into(h: &mut sha2::Sha256, terms: &[(usize, F257)]) {
    h.update((terms.len() as u32).to_le_bytes());
    for &(idx, coeff) in terms {
        h.update(idx.to_le_bytes());
        h.update(f257_to_u16(coeff).to_le_bytes());
    }
}

fn digest_anchor_projection_into(h: &mut sha2::Sha256, proj: &H12AlvoAnchorProjection) {
    h.update(proj.y_anchor_stream.to_le_bytes());
    h.update(proj.alpha_sparse.to_le_bytes());
    h.update(proj.beta_sparse.to_le_bytes());
    h.update(proj.delta_gamma_pi.to_le_bytes());
}

fn digest_germ_proof_into(h: &mut sha2::Sha256, proof: &H12GermProof) {
    h.update(proof.params.pack_d.to_le_bytes());
    h.update(proof.params.commit_rows.to_le_bytes());
    h.update(proof.params.blind_len.to_le_bytes());
    h.update((proof.local_view_values.len() as u32).to_le_bytes());
    for &v in &proof.local_view_values {
        h.update(v.to_le_bytes());
    }
    h.update((proof.blind_values.len() as u32).to_le_bytes());
    for &v in &proof.blind_values {
        h.update(v.to_le_bytes());
    }
    h.update((proof.ajtai_commitment_rows.len() as u32).to_le_bytes());
    for row in &proof.ajtai_commitment_rows {
        h.update((row.len() as u32).to_le_bytes());
        for &v in row {
            h.update(v.to_le_bytes());
        }
    }
    h.update((proof.verifier_payload.global_err_packed_values.len() as u32).to_le_bytes());
    for &v in &proof.verifier_payload.global_err_packed_values {
        h.update(v.to_le_bytes());
    }
    h.update((proof.verifier_payload.designated_challenge.len() as u32).to_le_bytes());
    for &v in &proof.verifier_payload.designated_challenge {
        h.update(v.to_le_bytes());
    }
    h.update((proof.verifier_payload.opening_projection_residuals.len() as u32).to_le_bytes());
    for &v in &proof.verifier_payload.opening_projection_residuals {
        h.update(v.to_le_bytes());
    }
    h.update((proof.verifier_payload.h_projection_residuals.len() as u32).to_le_bytes());
    for &v in &proof.verifier_payload.h_projection_residuals {
        h.update(v.to_le_bytes());
    }
    h.update((proof.verifier_payload.germ_linear_fingerprints.len() as u32).to_le_bytes());
    for fp in &proof.verifier_payload.germ_linear_fingerprints {
        for &v in fp {
            h.update(v.to_le_bytes());
        }
    }
    h.update(proof.verifier_payload.germ_mul_sumcheck.nvars.to_le_bytes());
    h.update((proof.verifier_payload.germ_mul_sumcheck.rounds.len() as u32).to_le_bytes());
    for round in &proof.verifier_payload.germ_mul_sumcheck.rounds {
        for eval in &round.evaluations {
            for &v in eval {
                h.update(v.to_le_bytes());
            }
        }
    }
    for &v in &proof.verifier_payload.germ_mul_sumcheck.opening.a_eval {
        h.update(v.to_le_bytes());
    }
    for &v in &proof.verifier_payload.germ_mul_sumcheck.opening.b_eval {
        h.update(v.to_le_bytes());
    }
    for &v in &proof.verifier_payload.germ_mul_sumcheck.opening.c_eval {
        h.update(v.to_le_bytes());
    }
    for &v in &proof.verifier_payload.germ_mul_sumcheck.opening.d_eval {
        h.update(v.to_le_bytes());
    }
}

#[cfg(test)]
mod tests {
    use ark_ff::Field;
    use dpp::{
        dr1cs_flpcp::Dr1csProofLayoutInfo,
        theorem43::{Theorem43AlvoLocalCheckSurface, Theorem43CapsuleLocalCheckSurface},
    };

    use super::*;
    use crate::h12_alvo_germ::{
        H12GermMulOpening, H12GermMulSumcheckProof, H12GermMulSumcheckRound, H12GermParams,
    };

    fn dummy_surface(rep_id: u64) -> Theorem43AlvoLocalCheckSurface<F257> {
        let local_check = Theorem43CapsuleLocalCheckSurface {
            block_id: 0,
            rep_id,
            coins: Theorem43Coins {
                idx: 1,
                lambda: F257::from(2u64),
                rho: F257::from(3u64),
                sigma: F257::from(4u64),
                c_hit: F257::from(5u64),
            },
            q1_pi_terms: vec![(0, F257::ONE)],
            q2_pi_terms: vec![(1, F257::ONE)],
            q3_pi_sparse_terms: vec![(2, F257::ONE)],
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
            touched_pi_positions_exact: local_check.touched_pi_positions(false),
            touched_pi_positions_conservative: local_check.touched_pi_positions(true),
            local_check,
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

    fn dummy_germ_proof(tag: u8) -> H12GermProof {
        H12GermProof {
            params: H12GermParams {
                pack_d: 64,
                commit_rows: 16,
                blind_len: 16,
            },
            local_view_values: vec![tag as u16, (tag as u16) + 1],
            blind_values: vec![(tag as u16) + 2],
            ajtai_commitment_rows: vec![vec![tag as u64, (tag as u64) + 1]],
            verifier_payload: crate::h12_alvo_germ::H12GermVerifierPayload {
                global_err_packed_values: vec![(tag as u16) + 3],
                designated_challenge: vec![(tag as u16) + 4; 16],
                opening_projection_residuals: vec![(tag as u16) + 5; 4],
                h_projection_residuals: vec![(tag as u16) + 6; 4],
                germ_linear_fingerprints: vec![[tag as u16; 16], [(tag as u16) + 1; 16]],
                germ_mul_sumcheck: H12GermMulSumcheckProof {
                    nvars: 1,
                    rounds: vec![H12GermMulSumcheckRound {
                        evaluations: [
                            [tag as u16; 16],
                            [(tag as u16) + 1; 16],
                            [(tag as u16) + 2; 16],
                            [(tag as u16) + 3; 16],
                        ],
                    }],
                    opening: H12GermMulOpening {
                        a_eval: [(tag as u16) + 4; 16],
                        b_eval: [(tag as u16) + 5; 16],
                        c_eval: [(tag as u16) + 6; 16],
                        d_eval: [(tag as u16) + 7; 16],
                    },
                },
            },
        }
    }

    #[test]
    fn test_stage1_root_changes_with_schedule() {
        let stmt_digest = [F257::from(7u64); 32];
        let lock_coin_seed = [9u8; 32];
        let a = vec![dummy_surface(1)];
        let b = vec![dummy_surface(2)];
        let a_digest = digest_alvo_schedule(a.as_slice(), 64);
        let b_digest = digest_alvo_schedule(b.as_slice(), 64);
        let ra = derive_stage1_root(&stmt_digest, &lock_coin_seed, 3, 1, &a_digest);
        let rb = derive_stage1_root(&stmt_digest, &lock_coin_seed, 3, 1, &b_digest);
        assert_ne!(ra, rb);
    }

    #[test]
    fn test_stage2_root_changes_with_germ_proof_payload() {
        let surface = dummy_surface(5);
        let mut base = H12AlvoLogicalLockAugmentation {
            share_index: 1,
            r_cap_reps: 1,
            stage1_root: [1u8; 32],
            stage2_root: [0u8; 32],
            schedule_digest: digest_alvo_schedule(std::slice::from_ref(&surface), 64),
            surfaces: vec![surface],
            germ_proof: Some(dummy_germ_proof(9)),
            rep_bundles: Vec::new(),
        };
        let root_a = derive_stage2_root(&base);
        base.germ_proof = Some(dummy_germ_proof(10));
        let root_b = derive_stage2_root(&base);
        assert_ne!(root_a, root_b);
    }
}
