//! Tiny-field (Theorem 4.3) lock helpers for WE arming.
//!
//! This module provides an **arm-before-proof** arming helper that binds to a statement digest
//! and an armer-private secret, using the Theorem-4.3 tiny-field DPP lockable query generator.
//!
//! Notes:
//! - This is intended for tiny fields (e.g., F257) where the accepting set is `{0,1}`.
//! - The lock artifact contains *public coins* and keeps the hidden query inside (as toxic waste),
//!   serving as a stand-in for LWE hints in the real lock layer.

use ark_crypto_primitives::sponge::{poseidon::PoseidonSponge, CryptographicSponge};
use ark_ff::{BigInteger, FftField, PrimeField};
use sha2::{Digest, Sha256};

use dpp::dr1cs_flpcp::{Dr1csInstanceSparse, RsDr1csNpFlpcpSparse};
use dpp::theorem43::{Theorem43Coins, Theorem43Dpp, Theorem43LockArtifact};
use dpp::SparseVec;

use cyclotomic_rings::rings::{FrogPoseidonConfig, GetPoseidonParams};
use stark_rings::cyclotomic_ring::models::frog_ring::Fq;


pub use crate::we_statement::arm_theorem43_from_statement;

/// Extract the public coins from a lock artifact (convenience).
pub fn public_coins<F: PrimeField>(art: &Theorem43LockArtifact<F>) -> Theorem43Coins<F> {
    art.coins.clone()
}

/// Build a minimal NP-style RS FLPCP instance for a 1-constraint multiply relation.
///
/// Constraint: z0 * z1 = z2 over `n_total = 3` variables.
/// Set `l_public` to control how many leading variables are public.
pub fn toy_mul_flpcp<F: PrimeField + FftField>(l_public: usize) -> RsDr1csNpFlpcpSparse<F> {
    let n_total = 3usize;
    assert!(l_public <= n_total);
    let a_row = SparseVec::new(vec![(F::ONE, 0)]);
    let b_row = SparseVec::new(vec![(F::ONE, 1)]);
    let c_row = SparseVec::new(vec![(F::ONE, 2)]);
    let inst = Dr1csInstanceSparse::<F> {
        n: n_total,
        a: vec![a_row],
        b: vec![b_row],
        c: vec![c_row],
    };
    let k_rows = inst.k();
    let ell = 2 * k_rows;
    RsDr1csNpFlpcpSparse::<F>::new(inst, l_public, ell)
}

/// Arm a Theorem-4.3 tiny-field lock using **Frog Poseidon** to derive coins and UV bits.
///
/// This emulates the Frog Poseidon sponge outside the tiny field and feeds the derived
/// coins/UV bits into the tiny-field Theorem-4.3 lock.
pub fn arm_theorem43_from_statement_frog_emulated<F: PrimeField>(
    dpp: &Theorem43Dpp<F>,
    stmt_digest: [u8; 32],
    x: &[F],
    armer_seed: [u8; 32],
    lock_j: u64,
) -> Result<Theorem43LockArtifact<F>, String> {
    // Statement-binding commitment in the tiny field.
    let c_stmt = crate::we_statement::digest32_to_bits_field::<F>(stmt_digest);

    // Frog Poseidon sponge.
    let cfg = FrogPoseidonConfig::get_poseidon_config();
    let mut sponge = PoseidonSponge::<Fq>::new(&cfg);
    sponge.absorb(&Fq::from(43u64)); // domain sep (matches theorem43)

    // Absorb c_stmt and x, mapped into Frog base field.
    for b in &c_stmt {
        let fq = Fq::from(b.into_bigint().to_bytes_le().get(0).copied().unwrap_or(0) as u64);
        sponge.absorb(&fq);
    }
    for xi in x {
        let fq = Fq::from_le_bytes_mod_order(&xi.into_bigint().to_bytes_le());
        sponge.absorb(&fq);
    }

    // Squeeze public coins in Frog, then map into the tiny field.
    let idx_fq = sponge.squeeze_field_elements::<Fq>(1)[0];
    let idx_u = fq_to_u64(idx_fq);
    let idx = (idx_u as usize) % dpp.ell();
    let lambda = fq_to_small::<F>(sponge.squeeze_field_elements::<Fq>(1)[0]);
    let rho = fq_to_small::<F>(sponge.squeeze_field_elements::<Fq>(1)[0]);
    let sigma = fq_to_small::<F>(sponge.squeeze_field_elements::<Fq>(1)[0]);
    let coins = Theorem43Coins { idx, lambda, rho, sigma };

    // Mix in armer-private secret for hidden-query derivation.
    let armer_secret = crate::we_statement::derive_armer_secret::<F>(armer_seed, stmt_digest, lock_j, 4);
    for s in &armer_secret {
        let fq = Fq::from_le_bytes_mod_order(&s.into_bigint().to_bytes_le());
        sponge.absorb(&fq);
    }

    // Derive UV bits from Frog sponge output.
    let q_minus_1 = field_modulus_u64::<F>()?
        .saturating_sub(1) as usize;
    let bits_src = sponge.squeeze_field_elements::<Fq>(q_minus_1);
    let mut q_bits: Vec<u8> = Vec::with_capacity(q_minus_1);
    for (i, b) in bits_src.into_iter().enumerate() {
        let mut u = fq_to_u64(b);
        u ^= i as u64;
        q_bits.push((u & 1) as u8);
    }

    dpp.arm_with_coins_and_uv_bits(c_stmt, x, coins, &q_bits)
}

fn fq_to_u64(x: Fq) -> u64 {
    let bytes = x.into_bigint().to_bytes_le();
    let mut buf = [0u8; 8];
    let n = bytes.len().min(8);
    buf[..n].copy_from_slice(&bytes[..n]);
    u64::from_le_bytes(buf)
}

fn fq_to_small<F: PrimeField>(x: Fq) -> F {
    F::from_le_bytes_mod_order(&x.into_bigint().to_bytes_le())
}

fn field_modulus_u64<F: PrimeField>() -> Result<u64, String> {
    let bytes = F::MODULUS.to_bytes_le();
    if bytes.len() > 8 {
        return Err("field modulus does not fit u64".to_string());
    }
    let mut buf = [0u8; 8];
    buf[..bytes.len()].copy_from_slice(&bytes);
    Ok(u64::from_le_bytes(buf))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_ff::{Field, Fp64, MontBackend, MontConfig};

    #[derive(MontConfig)]
    #[modulus = "257"]
    #[generator = "3"]
    pub struct F257Config;
    type F257 = Fp64<MontBackend<F257Config, 1>>;

    #[test]
    fn test_tiny_lock_arm_before_proof_roundtrip() {
        let flpcp = toy_mul_flpcp::<F257>(1);
        let dpp = Theorem43Dpp::<F257>::new(flpcp.clone()).expect("theorem43 new");

        let z0 = F257::from(2u64);
        let z1 = F257::from(5u64);
        let z2 = z0 * z1;
        let x = vec![z0];
        let z_w = vec![z1, z2];

        let stmt_digest: [u8; 32] = Sha256::digest(b"LFP_TINY_LOCK_STMT_V1").into();
        let armer_seed = [7u8; 32];
        let lock_j = 0u64;

        let art = arm_theorem43_from_statement_frog_emulated::<F257>(
            &dpp,
            stmt_digest,
            &x,
            armer_seed,
            lock_j,
        )
        .expect("arm_theorem43_from_statement_frog_emulated");
        assert_eq!(art.accepting_set, [F257::ZERO, F257::ONE]);
        assert_eq!(art.len, x.len() + dpp.proof_len());

        let pi = dpp.prove_for_query(&x, &z_w, &art.coins).expect("prove_for_query");
        assert_eq!(pi.len(), dpp.proof_len());

        let a_full = art.answer_for(&x, &pi).expect("answer_for");
        let (q_x, q_pi) = art.split_query(x.len(), pi.len()).expect("split_query");
        assert_eq!(a_full, q_x.dot(&x) + q_pi.dot(&pi));
    }
}
