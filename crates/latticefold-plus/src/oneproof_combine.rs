//! OneProof "global combine" utilities (hash-combine only).
//!
//! This module intentionally contains the *single* canonical combine rule used by the
//! SP1 OneProof WE-gate path. Keeping it in one place prevents subtle drift between:
//! - production arming / decapsulation code
//! - tests and benchmarks

use ark_ff::PrimeField;
use latticefold::transcript::poseidon::F257;
use sha2::{Digest, Sha256};

fn stmt_digest_to_bytes64(stmt_digest: &[F257; 32]) -> [u8; 64] {
    let mut out = [0u8; 64];
    for (i, e) in stmt_digest.iter().enumerate() {
        let v = (e.into_bigint().as_ref()[0] % 257) as u16;
        let b = v.to_le_bytes();
        out[2 * i] = b[0];
        out[2 * i + 1] = b[1];
    }
    out
}

/// Canonical non-linear combine of per-lock 32-byte secrets.
///
/// Domain-separated and statement-bound to prevent cross-statement / cross-package mixing.
pub fn oneproof_hash_combine_shares_v1(
    stmt_digest: &[F257; 32],
    lock_coin_seed: &[u8; 32],
    shares: &[(u32, [u8; 32])],
) -> [u8; 32] {
    const DST: &[u8] = b"LFP_ONEPROOF_COMBINE_V1";
    let mut h = Sha256::new();
    h.update(DST);
    h.update(stmt_digest_to_bytes64(stmt_digest));
    h.update(lock_coin_seed);

    // Canonicalize by share index.
    let mut sorted: Vec<(u32, [u8; 32])> = shares.to_vec();
    sorted.sort_by_key(|(idx, _v)| *idx);
    for (idx, v) in sorted {
        h.update(idx.to_le_bytes());
        h.update(v);
    }
    let out = h.finalize();
    let mut arr = [0u8; 32];
    arr.copy_from_slice(&out[..32]);
    arr
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hash_combine_order_invariant_by_index() {
        let stmt_digest = [F257::from(7u64); 32];
        let lock_coin_seed = [9u8; 32];
        let s1 = (2u32, [2u8; 32]);
        let s0 = (1u32, [1u8; 32]);

        let a = oneproof_hash_combine_shares_v1(&stmt_digest, &lock_coin_seed, &[s0, s1]);
        let b = oneproof_hash_combine_shares_v1(&stmt_digest, &lock_coin_seed, &[s1, s0]);
        assert_eq!(a, b, "combine should be canonicalized by index");
    }

    #[test]
    fn test_hash_combine_binds_index() {
        let stmt_digest = [F257::from(7u64); 32];
        let lock_coin_seed = [9u8; 32];
        let v = [5u8; 32];

        let a = oneproof_hash_combine_shares_v1(&stmt_digest, &lock_coin_seed, &[(1u32, v)]);
        let b = oneproof_hash_combine_shares_v1(&stmt_digest, &lock_coin_seed, &[(2u32, v)]);
        assert_ne!(a, b, "share index must be bound into combine hash");
    }
}

