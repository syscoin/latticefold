//! Helpers to bind WE locks to a Symphony statement.
//!
//! In Architecture‑T, the armer derives per-lock queries from a **statement hash** that binds:
//! - the SP1 verifier key / program id (vk hash),
//! - the public statement values,
//! - and the public folding-layer statement artifacts (commitments).
//!
//! This module provides a canonical `sha256` hash for the **public** portion of `R_WE`.

use ark_ff::{BigInteger, Field, PrimeField};
use sha2::{Digest, Sha256};
use stark_rings::OverField;

/// SHA256 hash of the public statement for `R_WE` (hetero‑M).
///
/// This intentionally does **not** include the folding proof (`PiFoldBatchedProof`) nor auxiliary
/// witness messages (`aux`) nor reduced witness (`ro_witness`): those are witnesses to the relation.
///
/// Callers should include:
/// - `vk_hash`: a 32-byte digest identifying the SP1 shrink verifier circuit/version.
/// - `r1cs_digest`: a 32-byte digest identifying the SP1 R1CS being folded (from SP1 R1CS header).
/// - `public_inputs`: the public inputs absorbed by the Poseidon transcript.
/// - `cm_f`: commitment vectors (one per chunk/instance).
/// - `cfs_had_u`, `cfs_mon_b`: CP transcript-message commitment vectors (one per chunk/instance).
pub fn we_statement_hash_hetero_m<R: OverField>(
    vk_hash: [u8; 32],
    r1cs_digest: [u8; 32],
    public_inputs: &[R::BaseRing],
    cm_f: &[Vec<R>],
    cfs_had_u: &[Vec<R>],
    cfs_mon_b: &[Vec<R>],
) -> [u8; 32]
where
    R::BaseRing: Field,
{
    let mut h = Sha256::new();
    h.update(b"SYMPHONY_WE_STATEMENT_V1");
    h.update(&vk_hash);
    h.update(&r1cs_digest);

    // Public inputs (base field elements).
    h.update(&(public_inputs.len() as u64).to_le_bytes());
    for x in public_inputs {
        absorb_field_elem::<R>(x, &mut h);
    }

    // Commitment vectors; bind lengths to avoid ambiguity.
    absorb_vecvec_ring::<R>(b"cm_f", cm_f, &mut h);
    absorb_vecvec_ring::<R>(b"cfs_had_u", cfs_had_u, &mut h);
    absorb_vecvec_ring::<R>(b"cfs_mon_b", cfs_mon_b, &mut h);

    h.finalize().into()
}

fn absorb_vecvec_ring<R: OverField>(tag: &[u8], v: &[Vec<R>], h: &mut Sha256)
where
    R::BaseRing: Field,
{
    h.update(tag);
    h.update(&(v.len() as u64).to_le_bytes());
    for row in v {
        h.update(&(row.len() as u64).to_le_bytes());
        for x in row {
            absorb_ring_elem::<R>(x, h);
        }
    }
}

fn absorb_ring_elem<R: OverField>(x: &R, h: &mut Sha256)
where
    R::BaseRing: Field,
{
    // Canonicalize by hashing the base-prime-field decomposition of all coefficients.
    for c in x.coeffs() {
        for fp in c.to_base_prime_field_elements() {
            h.update(fp.into_bigint().to_bytes_le());
        }
    }
}

fn absorb_field_elem<R: OverField>(x: &R::BaseRing, h: &mut Sha256)
where
    R::BaseRing: Field,
{
    for fp in x.to_base_prime_field_elements() {
        h.update(fp.into_bigint().to_bytes_le());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use stark_rings::{cyclotomic_ring::models::frog_ring::RqPoly as R, PolyRing, Ring};

    #[test]
    fn test_statement_hash_deterministic() {
        let vk = [1u8; 32];
        let r1cs = [2u8; 32];
        let public_inputs: Vec<<R as PolyRing>::BaseRing> = vec![<R as PolyRing>::BaseRing::from(5u128)];
        let cm_f: Vec<Vec<R>> = vec![vec![R::ONE; 2]];
        let cfs_had_u: Vec<Vec<R>> = vec![vec![R::ONE; 3]];
        let cfs_mon_b: Vec<Vec<R>> = vec![vec![R::ONE; 4]];

        let a = we_statement_hash_hetero_m::<R>(vk, r1cs, &public_inputs, &cm_f, &cfs_had_u, &cfs_mon_b);
        let b = we_statement_hash_hetero_m::<R>(vk, r1cs, &public_inputs, &cm_f, &cfs_had_u, &cfs_mon_b);
        assert_eq!(a, b);
    }

    #[test]
    fn test_statement_hash_changes_on_public_inputs() {
        let vk = [1u8; 32];
        let r1cs = [2u8; 32];
        let cm_f: Vec<Vec<R>> = vec![vec![R::ONE; 2]];
        let cfs_had_u: Vec<Vec<R>> = vec![vec![R::ONE; 3]];
        let cfs_mon_b: Vec<Vec<R>> = vec![vec![R::ONE; 4]];

        let p0: Vec<<R as PolyRing>::BaseRing> = vec![<R as PolyRing>::BaseRing::from(5u128)];
        let p1: Vec<<R as PolyRing>::BaseRing> = vec![<R as PolyRing>::BaseRing::from(6u128)];
        let a = we_statement_hash_hetero_m::<R>(vk, r1cs, &p0, &cm_f, &cfs_had_u, &cfs_mon_b);
        let b = we_statement_hash_hetero_m::<R>(vk, r1cs, &p1, &cm_f, &cfs_had_u, &cfs_mon_b);
        assert_ne!(a, b);
    }
}

