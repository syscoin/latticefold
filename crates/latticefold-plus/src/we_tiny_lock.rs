//! Tiny-field (Theorem 4.3) lock helpers for WE arming.
//!
//! This module provides an **arm-before-proof** arming helper that binds to a statement digest
//! and an armer-private secret, using the Theorem-4.3 tiny-field DPP lockable query generator.
//!
//! Notes:
//! - This is intended for tiny fields (e.g., F257) where the accepting set is `{1,2}`.
//! - The lock artifact contains *public coins* and keeps the hidden query inside (as toxic waste),
//!   serving as a stand-in for LWE hints in the real lock layer.

use ark_ff::{FftField, PrimeField};
#[cfg(test)]
use sha2::{Digest, Sha256};

use dpp::dr1cs_flpcp::{ChunkedMulCodeDr1csNpFlpcpSparse, Dr1csInstanceSparse, MulCode, TensorRsMulCode};
use dpp::theorem43::{Theorem43Coins, Theorem43Dpp, Theorem43LockArtifact};
use dpp::dr1cs_flpcp::Dr1csQueryScratch;
use dpp::SparseVec;
use symphony::dpp_poseidon::SparseDr1csInstance as SymDr1cs;

use crate::lockable_ringlwe::{arm_ringlwe_lock, RingLweLockArtifact, RingLweParams};
use crate::lockable_ringlwe::QueryBlockAccumulator;

pub use crate::we_statement::arm_theorem43_from_statement;

#[cfg(feature = "we_gate")]
use crate::we_gate_arith;
#[cfg(feature = "we_gate")]
use crate::we_statement::WeParams;
#[cfg(feature = "we_gate")]
use latticefold::transcript::poseidon::F257;
#[cfg(feature = "we_gate")]
use stark_rings::{CoeffRing, OverField, PolyRing, Zq};

/// Extract the public coins from a lock artifact (convenience).
pub fn public_coins<F: PrimeField>(art: &Theorem43LockArtifact<F>) -> Theorem43Coins<F> {
    art.coins.clone()
}

// NOTE: test-only helpers live in the test module.

/// Arm a Theorem-4.3 tiny-field lock and wrap it in a Ring-LWE backend.
///
/// This produces a compact public lock artifact that does not reveal the hidden query.
pub(crate) fn arm_theorem43_ringlwe_from_statement<F: PrimeField, C: MulCode<F> + Sync>(
    dpp: &Theorem43Dpp<F, ChunkedMulCodeDr1csNpFlpcpSparse<F, C>>,
    stmt_digest: [u8; 32],
    x: &[F],
    armer_seed: [u8; 32],
    lock_j: u64,
    block_id: usize,
    rep_id: u64,
    params: RingLweParams,
    rng: &mut impl rand::RngCore,
    scratch: &mut Dr1csQueryScratch<F>,
    acc: &mut QueryBlockAccumulator,
) -> Result<RingLweLockArtifact<F>, String> {
    let c_stmt = crate::we_statement::digest32_to_bits_field::<F>(stmt_digest);
    let armer_secret = crate::we_statement::derive_armer_secret::<F>(armer_seed, stmt_digest, lock_j, 4);
    let art = dpp.arm(&c_stmt, x, &armer_secret, block_id, rep_id)?;
    let pi_len = dpp.proof_len();
    let mut err: Option<String> = None;
    let offset_f = dpp
        .stream_query_terms_for_pi(x, &art.coins, &art.coeffs, scratch, &mut |pi_idx, coeff| {
            if err.is_some() {
                return;
            }
            if let Err(e) = acc.add_term(&coeff, pi_idx) {
                err = Some(e);
            }
        })?;
    if let Some(e) = err {
        return Err(e);
    }
    let q_blocks = acc.into_blocks();
    arm_ringlwe_lock(
        c_stmt,
        art.accepting_set,
        art.coins,
        offset_f,
        x.len(),
        pi_len,
        q_blocks,
        params,
        rng,
    )
}

/// Streaming arming helper that keeps the chunked FLPCP backend available for proof streaming.
pub struct WeRingLweStreamingContext<F: PrimeField + FftField> {
    pub lock: RingLweLockArtifact<F>,
    pub dpp: Theorem43Dpp<F, ChunkedMulCodeDr1csNpFlpcpSparse<F, TensorRsMulCode<F>>>,
}

impl<F: PrimeField + FftField> WeRingLweStreamingContext<F> {
    pub fn prove_stream(
        &self,
        x: &[F],
        z_w: &[F],
        on_chunk: &mut dyn FnMut(Vec<F>),
    ) -> Result<(), String> {
        self.dpp.prove_for_query_stream(x, z_w, &self.lock.coins, on_chunk)
    }

    pub fn proof_len(&self) -> usize {
        self.dpp.proof_len()
    }
}

/// Arm a Ring-LWE lock and return a streaming context for chunked proof generation.
pub(crate) fn arm_we_ringlwe_from_dr1cs_streaming<F: PrimeField + FftField>(
    dr1cs: SymDr1cs<F>,
    public_len: usize,
    stmt_digest: [u8; 32],
    x: &[F],
    armer_seed: [u8; 32],
    lock_j: u64,
    block_id: usize,
    rep_id: u64,
    params: RingLweParams,
    rng: &mut impl rand::RngCore,
    scratch: &mut Dr1csQueryScratch<F>,
    acc: &mut QueryBlockAccumulator,
) -> Result<WeRingLweStreamingContext<F>, String> {
    if x.len() != public_len {
        return Err("arm_we_ringlwe_from_dr1cs_streaming: x length != public_len".to_string());
    }
    let inst = dr1cs_from_symphony(dr1cs);
    let code = TensorRsMulCode::<F>::new(48, 3)?;
    let k_block = code.dim_k();
    let blocks = chunk_dr1cs_sparse(inst, k_block);
    let flpcp = ChunkedMulCodeDr1csNpFlpcpSparse::<F, _>::new(blocks, public_len, code)?;
    let dpp = Theorem43Dpp::<F, _>::new(flpcp)?;
    let lock = arm_theorem43_ringlwe_from_statement(
        &dpp,
        stmt_digest,
        x,
        armer_seed,
        lock_j,
        block_id,
        rep_id,
        params,
        rng,
        scratch,
        acc,
    )?;
    Ok(WeRingLweStreamingContext { lock, dpp })
}

/// Arm the **LF+ tiny-field WE gate** (Poseidon(F257) + CM-coin surfaces) as a Theorem-4.3 lock.
///
/// This is the main wiring from:
/// - `we_gate_arith::build_we_dr1cs_for_plus_proof_shape_tiny` (arm-time instance construction)
/// into:
/// - `arm_we_ringlwe_from_dr1cs_streaming` (Theorem-4.3 + Ring-LWE wrapper).
///
/// Notes:
/// - The statement binding is carried by `stmt_digest` via `c_stmt` inside `arm_theorem43_from_statement`.
/// - `x` is currently a minimal public prefix (typically just `[1]`) unless/until the tiny gate
///   exports additional public inputs.
#[cfg(feature = "we_gate")]
pub(crate) fn arm_lfplus_we_gate_tiny_ringlwe_streaming<R>(
    params: &WeParams,
    public_inputs_len: usize,
    n_lin_proofs: usize,
    mlen_mats: usize,
    pairs: &[(usize, usize)],
    stmt_digest: [u8; 32],
    armer_seed: [u8; 32],
    lock_j: u64,
    block_id: usize,
    rep_id: u64,
    ringlwe_params: RingLweParams,
    rng: &mut impl rand::RngCore,
    scratch: &mut Dr1csQueryScratch<F257>,
    acc: &mut QueryBlockAccumulator,
) -> Result<WeRingLweStreamingContext<F257>, String>
where
    R: OverField + CoeffRing + PolyRing,
    R::BaseRing: Zq + ark_ff::Field + ark_ff::PrimeField,
{
    let shape = we_gate_arith::build_we_dr1cs_for_plus_proof_shape_tiny::<R>(
        params,
        public_inputs_len,
        n_lin_proofs,
        mlen_mats,
        pairs,
    )?;

    let x: Vec<F257> = match shape.public_len {
        1 => vec![F257::from(1u64)],
        // If/when the tiny gate includes the standard params prefix, use the canonical encoding.
        11 => crate::we_statement::encode_public_x::<F257>(params, &[]),
        other => {
            return Err(format!(
                "arm_lfplus_we_gate_tiny_ringlwe_streaming: unsupported public_len={other}"
            ))
        }
    };

    arm_we_ringlwe_from_dr1cs_streaming::<F257>(
        shape.inst,
        shape.public_len,
        stmt_digest,
        &x,
        armer_seed,
        lock_j,
        block_id,
        rep_id,
        ringlwe_params,
        rng,
        scratch,
        acc,
    )
}

fn dr1cs_from_symphony<F: PrimeField>(inst: SymDr1cs<F>) -> Dr1csInstanceSparse<F> {
    let mut a = Vec::with_capacity(inst.constraints.len());
    let mut b = Vec::with_capacity(inst.constraints.len());
    let mut c = Vec::with_capacity(inst.constraints.len());
    for mut row in inst.constraints {
        a.push(SparseVec::new(std::mem::take(&mut row.a)));
        b.push(SparseVec::new(std::mem::take(&mut row.b)));
        c.push(SparseVec::new(std::mem::take(&mut row.c)));
    }
    Dr1csInstanceSparse { n: inst.nvars, a, b, c }
}

fn chunk_dr1cs_sparse<F: PrimeField>(inst: Dr1csInstanceSparse<F>, k_block: usize) -> Vec<Dr1csInstanceSparse<F>> {
    if k_block == 0 {
        return vec![inst];
    }
    let total = inst.k();
    let mut blocks = Vec::new();
    let mut i = 0usize;
    while i < total {
        let end = usize::min(i + k_block, total);
        let mut a = inst.a[i..end].to_vec();
        let mut b = inst.b[i..end].to_vec();
        let mut c = inst.c[i..end].to_vec();
        // Pad with zero rows if needed.
        while a.len() < k_block {
            a.push(SparseVec::new(Vec::new()));
            b.push(SparseVec::new(Vec::new()));
            c.push(SparseVec::new(Vec::new()));
        }
        blocks.push(Dr1csInstanceSparse { n: inst.n, a, b, c });
        i = end;
    }
    if blocks.is_empty() {
        blocks.push(inst);
    }
    blocks
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_ff::{Field, Fp64, MontBackend, MontConfig};
    use rand::{rngs::StdRng, SeedableRng};
    use dpp::dr1cs_flpcp::{ChunkedMulCodeDr1csNpFlpcpSparse, TensorRsMulCode};
    fn tiny_mul_chunked_dpp<F: PrimeField>() -> Theorem43Dpp<F, ChunkedMulCodeDr1csNpFlpcpSparse<F, TensorRsMulCode<F>>> {
        let n_total = 3usize;
        let a_row = SparseVec::new(vec![(F::ONE, 0)]);
        let b_row = SparseVec::new(vec![(F::ONE, 1)]);
        let c_row = SparseVec::new(vec![(F::ONE, 2)]);
        let inst = Dr1csInstanceSparse::<F> {
            n: n_total,
            a: vec![a_row],
            b: vec![b_row],
            c: vec![c_row],
        };
        let code = TensorRsMulCode::<F>::new(2, 1).expect("tensor code");
        let k_block = code.dim_k();
        let blocks = chunk_dr1cs_sparse(inst, k_block);
        let flpcp = ChunkedMulCodeDr1csNpFlpcpSparse::<F, _>::new(blocks, 1, code)
            .expect("chunked flpcp");
        Theorem43Dpp::<F, _>::new(flpcp).expect("theorem43 new")
    }

    fn collect_streamed_pi<F: PrimeField>(
        dpp: &Theorem43Dpp<F, ChunkedMulCodeDr1csNpFlpcpSparse<F, TensorRsMulCode<F>>>,
        x: &[F],
        z_w: &[F],
        coins: &Theorem43Coins<F>,
    ) -> Vec<F> {
        let mut pi = Vec::new();
        dpp.prove_for_query_stream(x, z_w, coins, &mut |chunk| {
            pi.extend_from_slice(&chunk);
        })
        .expect("prove_for_query_stream");
        pi
    }


    #[derive(MontConfig)]
    #[modulus = "257"]
    #[generator = "3"]
    pub struct F257Config;
    type F257 = Fp64<MontBackend<F257Config, 1>>;

    #[test]
    fn test_tiny_lock_arm_before_proof_roundtrip() {
        let dpp = tiny_mul_chunked_dpp::<F257>();

        let z0 = F257::from(2u64);
        let z1 = F257::from(5u64);
        let z2 = z0 * z1;
        let x = vec![z0];
        let z_w = vec![z1, z2];

        let stmt_digest: [u8; 32] = Sha256::digest(b"LFP_TINY_LOCK_STMT_V1").into();
        let armer_seed = [7u8; 32];
        let lock_j = 0u64;

        let art = arm_theorem43_from_statement::<F257>(
            &dpp,
            stmt_digest,
            &x,
            armer_seed,
            lock_j,
            0,
            0,
        )
        .expect("arm_theorem43_from_statement");
        assert_eq!(art.accepting_set, [F257::ONE, F257::from(2u64)]);
        assert_eq!(art.len, x.len() + dpp.proof_len());

        let pi = collect_streamed_pi(&dpp, &x, &z_w, &art.coins);
        assert_eq!(pi.len(), dpp.proof_len());

        let _a_full = dpp
            .answer_for_stream(&art, &x, &pi)
            .expect("answer_for_stream");
    }

    #[test]
    fn test_tiny_lock_ringlwe_roundtrip() {
        let dpp = tiny_mul_chunked_dpp::<F257>();

        let z0 = F257::from(2u64);
        let z1 = F257::from(5u64);
        let z2 = z0 * z1;
        let x = vec![z0];
        let z_w = vec![z1, z2];

        let stmt_digest: [u8; 32] = Sha256::digest(b"LFP_TINY_LOCK_STMT_RINGLWE_V1").into();
        let armer_seed = [9u8; 32];
        let lock_j = 0u64;

        let params = RingLweParams {
            // Use zero-noise parameters for a strict functional test.
            binomial_k: 0,
            noise_bound: 0,
            ..RingLweParams::default()
        };
        let mut rng = StdRng::seed_from_u64(42);
        let mut scratch = dpp.query_scratch();
        let mut acc = QueryBlockAccumulator::new(dpp.proof_len()).expect("acc");
        let lock = arm_theorem43_ringlwe_from_statement(
            &dpp,
            stmt_digest,
            &x,
            armer_seed,
            lock_j,
            0,
            0,
            params,
            &mut rng,
            &mut scratch,
            &mut acc,
        )
        .expect("arm_theorem43_ringlwe_from_statement");

        let pi = collect_streamed_pi(&dpp, &x, &z_w, &lock.coins);
        let a = lock.decap_answer(&x, &pi).expect("decap_answer");
        assert!(a == F257::ONE || a == F257::from(2u64));

        // Negative check: tweak proof and ensure decap fails.
        let mut pi_bad = pi.clone();
        pi_bad[0] += F257::ONE;
        assert!(lock.decap_answer(&x, &pi_bad).is_err());
    }

    #[test]
    fn test_tiny_lock_ringlwe_roundtrip_streaming() {
        let dpp = tiny_mul_chunked_dpp::<F257>();

        let z0 = F257::from(2u64);
        let z1 = F257::from(5u64);
        let z2 = z0 * z1;
        let x = vec![z0];
        let z_w = vec![z1, z2];

        let stmt_digest: [u8; 32] = Sha256::digest(b"LFP_TINY_LOCK_STMT_RINGLWE_STREAM_V1").into();
        let armer_seed = [11u8; 32];
        let lock_j = 0u64;

        let params = RingLweParams {
            binomial_k: 0,
            noise_bound: 0,
            ..RingLweParams::default()
        };
        let mut rng = StdRng::seed_from_u64(7);
        let mut scratch = dpp.query_scratch();
        let mut acc = QueryBlockAccumulator::new(dpp.proof_len()).expect("acc");
        let lock = arm_theorem43_ringlwe_from_statement(
            &dpp,
            stmt_digest,
            &x,
            armer_seed,
            lock_j,
            0,
            0,
            params,
            &mut rng,
            &mut scratch,
            &mut acc,
        )
        .expect("arm_theorem43_ringlwe_from_statement");

        let mut chunks = Vec::new();
        dpp.prove_for_query_stream(&x, &z_w, &lock.coins, &mut |chunk| chunks.push(chunk))
            .expect("prove_for_query_stream");
        let a = lock
            .decap_answer_stream(&x, dpp.proof_len(), chunks)
            .expect("decap_answer_stream");
        assert!(a == F257::ONE || a == F257::from(2u64));
    }

    #[test]
    #[ignore]
    fn test_tiny_lock_ringlwe_roundtrip_with_noise_stats() {
        let dpp = tiny_mul_chunked_dpp::<F257>();

        let z0 = F257::from(2u64);
        let z1 = F257::from(5u64);
        let z2 = z0 * z1;
        let x = vec![z0];
        let z_w = vec![z1, z2];

        let stmt_digest: [u8; 32] = Sha256::digest(b"LFP_TINY_LOCK_STMT_RINGLWE_V1").into();
        let armer_seed = [9u8; 32];
        let lock_j = 0u64;

        let params = RingLweParams {
            binomial_k: 12,
            noise_bound: 48,
            ..RingLweParams::default()
        };

        let trials = 100usize;
        let mut ok_tight = 0usize;
        let mut ok_loose = 0usize;
        let mut rng = StdRng::seed_from_u64(12345);
        let mut acc = QueryBlockAccumulator::new(dpp.proof_len()).expect("acc");
        for _ in 0..trials {
            let mut scratch = dpp.query_scratch();
            let lock = arm_theorem43_ringlwe_from_statement(
                &dpp,
                stmt_digest,
                &x,
                armer_seed,
                lock_j,
                0,
                0,
                params.clone(),
                &mut rng,
                &mut scratch,
                &mut acc,
            )
            .expect("arm_theorem43_ringlwe_from_statement");

            let pi = collect_streamed_pi(&dpp, &x, &z_w, &lock.coins);
            let mut lock_tight = lock.clone();
            lock_tight.params.noise_bound = 48;
            if lock_tight.decap_answer(&x, &pi).is_ok() {
                ok_tight += 1;
            }

            let mut lock_loose = lock.clone();
            lock_loose.params.noise_bound = lock_tight.params.noise_bound * 2;
            if lock_loose.decap_answer(&x, &pi).is_ok() {
                ok_loose += 1;
            }
        }

        println!(
            "ringlwe noisy decap success: tight={ok_tight}/{trials}, loose={ok_loose}/{trials}"
        );

        // Sanity: loose bound should strictly dominate tight bound.
        assert!(ok_loose > ok_tight, "expected ok_loose > ok_tight");
        assert!(ok_loose > 0, "loose bound should succeed sometimes");
    }
}
