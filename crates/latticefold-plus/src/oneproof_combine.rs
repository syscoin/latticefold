//! OneProof "global combine" utilities (hash-combine only).
//!
//! This module intentionally contains the *single* canonical combine rule used by the
//! SP1 OneProof WE-gate path. Keeping it in one place prevents subtle drift between:
//! - production arming / decapsulation code
//! - tests and benchmarks

use ark_ff::PrimeField;
use latticefold::transcript::poseidon::F257;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};

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

#[inline]
fn xor32(a: &[u8; 32], b: &[u8; 32]) -> [u8; 32] {
    let mut out = [0u8; 32];
    for i in 0..32 {
        out[i] = a[i] ^ b[i];
    }
    out
}

/// Reconstruct per-index *logical* candidate shares from per-rep candidates via XOR.
///
/// This matches the rep-unlinkability arming policy where each logical share is split into
/// `R` rep plaintexts \(v_0, ..., v_{R-1}\) such that `v_0 ⊕ ... ⊕ v_{R-1} = master`.
///
/// Input format is the canonical decap output: a flat list of `(share_index, candidates_for_rep)`,
/// with one record per rep.
///
/// Output is one record per `share_index`, where the candidate list contains all distinct XOR
/// combinations across that index's reps (bounded by `max_candidates_per_index`).
pub fn oneproof_xor_reconstruct_logical_candidates_v1(
    per_rep_candidates: &[(u32, Vec<[u8; 32]>)],
    max_candidates_per_index: usize,
) -> Result<Vec<(u32, Vec<[u8; 32]>)>, String> {
    if max_candidates_per_index == 0 {
        return Err("oneproof_xor_reconstruct_logical_candidates_v1: max_candidates_per_index=0".to_string());
    }

    let mut by_idx: BTreeMap<u32, Vec<&Vec<[u8; 32]>>> = BTreeMap::new();
    for (idx, cands) in per_rep_candidates {
        by_idx.entry(*idx).or_default().push(cands);
    }

    let mut out: Vec<(u32, Vec<[u8; 32]>)> = Vec::with_capacity(by_idx.len());
    for (idx, rep_lists) in by_idx {
        // Start from the neutral element for XOR.
        let mut acc: BTreeSet<[u8; 32]> = BTreeSet::new();
        acc.insert([0u8; 32]);
        for rep_cands in rep_lists {
            if rep_cands.is_empty() {
                return Err(format!(
                    "oneproof_xor_reconstruct_logical_candidates_v1: empty candidate list for idx={}",
                    idx
                ));
            }
            let mut next: BTreeSet<[u8; 32]> = BTreeSet::new();
            for a in acc.iter() {
                for b in rep_cands {
                    next.insert(xor32(a, b));
                    if next.len() > max_candidates_per_index {
                        return Err(format!(
                            "oneproof_xor_reconstruct_logical_candidates_v1: candidate explosion at idx={} (cap={})",
                            idx, max_candidates_per_index
                        ));
                    }
                }
            }
            acc = next;
        }
        out.push((idx, acc.into_iter().collect()));
    }
    Ok(out)
}

/// Global-only resolver: search for a tuple of shares whose hash-combine equals `target`.
///
/// This is intentionally exponential in the candidate set sizes. Use only when each index has
/// a small candidate list (e.g., after proof-induced narrowing), and cap work via `max_tries`.
pub fn oneproof_find_selection_for_target_combined_key_v1(
    stmt_digest: &[F257; 32],
    lock_coin_seed: &[u8; 32],
    logical_candidates: &[(u32, Vec<[u8; 32]>)],
    target: &[u8; 32],
    max_tries: u64,
) -> Result<Option<Vec<(u32, [u8; 32])>>, String> {
    let mut items: Vec<(u32, Vec<[u8; 32]>)> = logical_candidates.to_vec();
    items.sort_by_key(|(idx, _)| *idx);
    for (idx, c) in &items {
        if c.is_empty() {
            return Err(format!(
                "oneproof_find_selection_for_target_combined_key_v1: empty candidate list for idx={}",
                idx
            ));
        }
    }

    let mut tries: u64 = 0;
    let mut chosen: Vec<(u32, [u8; 32])> = Vec::with_capacity(items.len());

    fn dfs(
        i: usize,
        items: &[(u32, Vec<[u8; 32]>)],
        stmt_digest: &[F257; 32],
        lock_coin_seed: &[u8; 32],
        target: &[u8; 32],
        max_tries: u64,
        tries: &mut u64,
        chosen: &mut Vec<(u32, [u8; 32])>,
    ) -> Option<Vec<(u32, [u8; 32])>> {
        if *tries >= max_tries {
            return None;
        }
        if i == items.len() {
            *tries = tries.saturating_add(1);
            let combined = oneproof_hash_combine_shares_v1(stmt_digest, lock_coin_seed, chosen.as_slice());
            if &combined == target {
                return Some(chosen.clone());
            }
            return None;
        }

        let (idx, cands) = &items[i];
        for cand in cands {
            if *tries >= max_tries {
                return None;
            }
            chosen.push((*idx, *cand));
            if let Some(sol) = dfs(i + 1, items, stmt_digest, lock_coin_seed, target, max_tries, tries, chosen) {
                return Some(sol);
            }
            chosen.pop();
        }
        None
    }

    Ok(dfs(
        0,
        items.as_slice(),
        stmt_digest,
        lock_coin_seed,
        target,
        max_tries.max(1),
        &mut tries,
        &mut chosen,
    ))
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

    #[test]
    fn test_xor_reconstruct_logical_candidates() {
        let a = [1u8; 32];
        let b = [2u8; 32];
        let c = [3u8; 32];
        let per_rep = vec![(1u32, vec![a, b]), (1u32, vec![c])];
        let logical = oneproof_xor_reconstruct_logical_candidates_v1(&per_rep, 16).unwrap();
        assert_eq!(logical.len(), 1);
        assert_eq!(logical[0].0, 1u32);
        let mut got = logical[0].1.clone();
        got.sort();
        let mut expect = vec![xor32(&a, &c), xor32(&b, &c)];
        expect.sort();
        assert_eq!(got, expect);
    }

    #[test]
    fn test_global_resolver_finds_target_combined_key() {
        let stmt_digest = [F257::from(7u64); 32];
        let lock_coin_seed = [9u8; 32];

        let s1_good = [1u8; 32];
        let s1_bad = [2u8; 32];
        let s2_good = [3u8; 32];
        let s2_bad = [4u8; 32];

        let target = oneproof_hash_combine_shares_v1(
            &stmt_digest,
            &lock_coin_seed,
            &[(1u32, s1_good), (2u32, s2_good)],
        );
        let logical = vec![
            (1u32, vec![s1_bad, s1_good]),
            (2u32, vec![s2_bad, s2_good]),
        ];
        let found = oneproof_find_selection_for_target_combined_key_v1(
            &stmt_digest,
            &lock_coin_seed,
            &logical,
            &target,
            1024,
        )
        .unwrap()
        .expect("solver should find solution");
        let combined = oneproof_hash_combine_shares_v1(&stmt_digest, &lock_coin_seed, found.as_slice());
        assert_eq!(combined, target);
    }
}

