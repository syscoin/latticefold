//! **Proof-Derived Key (PDK)** lock layer for F257 DPP witness encryption.
//!
//! # The innovation
//!
//! Previous designs (lockable_ringlwe, lockable_lwe) all tried to ENCRYPT a
//! pre-chosen payload inside the lock. This fails because the armer must encrypt
//! the same payload under both accepting-set branches (for completeness), and the
//! attacker who recovers s₀ decrypts both → zero ambiguity.
//!
//! PDK flips the model: **the lock doesn't encrypt anything.** Instead, each lock
//! produces a branch-dependent RANDOM VALUE (a "share"), and the proof determines
//! which share is selected. The final key = Hash(selected shares across all locks).
//!
//! The armer doesn't know the final key (they don't have the proof). "Encryption"
//! happens externally — e.g., a Bitcoin transaction to the proof-derived address.
//!
//! # Security
//!
//! Each lock contributes 2 candidate shares (one per branch). The attacker sees
//! both (they're derivable from s₀, s₁ and the accepting set). Only the proof
//! determines which is selected.
//!
//! - **Per lock**: 2 candidates, no local oracle (both shares are random, no tags)
//! - **L locks**: 2^L candidate final keys
//! - **MITM** (checkable output): 2^(L/2) → **L=256 for 128-bit PQ**
//!
//! # Lock compactness
//!
//! No ciphertexts at all. The lock is just: hints + reduction hints + accepting set.
//! Size: ~20 KB per lock at nnz=1000. Total for L=256: **~5 MB**.
//!
//! # How the shares are derived (not stored)
//!
//! `share = PRF(s₀, s₁, a, lock_index, statement_hash)` where `a` is the
//! accepting-set element selected by the proof. Both candidate shares are
//! derivable by anyone who knows `s₀, s₁` (which the attacker recovers from
//! `G₀, G₁`). The shares don't need to be stored in the lock.

use rand::RngCore;

const GOLDILOCKS_P: u64 = 18446744069414584321;

// -- Goldilocks arithmetic --------------------------------------------------

#[inline]
fn add_g(a: u64, b: u64) -> u64 {
    ((a as u128 + b as u128) % (GOLDILOCKS_P as u128)) as u64
}

#[inline]
fn mul_g(a: u64, b: u64) -> u64 {
    ((a as u128 * b as u128) % (GOLDILOCKS_P as u128)) as u64
}

#[inline]
fn inv_g(a: u64) -> u64 { pow_g(a, GOLDILOCKS_P - 2) }

fn pow_g(mut base: u64, mut exp: u64) -> u64 {
    let mut r: u64 = 1;
    base %= GOLDILOCKS_P;
    while exp > 0 {
        if exp & 1 == 1 { r = mul_g(r, base); }
        exp >>= 1;
        base = mul_g(base, base);
    }
    r
}

#[inline]
fn from_i64_g(x: i64) -> u64 {
    if x >= 0 { (x as u64) % GOLDILOCKS_P }
    else {
        let t = ((-x) as u64) % GOLDILOCKS_P;
        if t == 0 { 0 } else { GOLDILOCKS_P - t }
    }
}

#[inline]
fn centered_f257(v: u16) -> i16 {
    debug_assert!(v <= 256);
    if v <= 128 { v as i16 } else { (v as i16) - 257 }
}

// -- Crypto -----------------------------------------------------------------

fn sha256(chunks: &[&[u8]]) -> [u8; 32] {
    use sha2::Digest;
    let mut h = sha2::Sha256::new();
    for c in chunks { h.update(c); }
    h.finalize().into()
}

/// Derive the branch share: a random 32-byte value determined by (s₀, s₁, a, lock_index, stmt).
fn derive_branch_share(
    s0: u64, s1: u64, a_f257: u16,
    lock_index: u64, stmt: &[u8; 32],
) -> [u8; 32] {
    sha256(&[
        b"LFP_PDK_SHARE_V1",
        stmt,
        &lock_index.to_le_bytes(),
        &s0.to_le_bytes(),
        &s1.to_le_bytes(),
        &(a_f257 as u16).to_le_bytes(),
    ])
}

fn xor_crypt(key: &[u8; 32], nonce: &[u8; 12], data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    let mut ctr = 0u32;
    let mut pos = 0usize;
    while pos < data.len() {
        let blk = sha256(&[b"LFP_PDK_XOR_V1", key, nonce, &ctr.to_le_bytes()]);
        let take = (data.len() - pos).min(32);
        for j in 0..take { out.push(data[pos + j] ^ blk[j]); }
        pos += take;
        ctr += 1;
    }
    out
}

// -- Lock types -------------------------------------------------------------

/// PDK lock artifact. No ciphertexts — shares are derived, not stored.
#[derive(Clone, Debug)]
pub struct PdkLockArtifact {
    pub statement_hash: [u8; 32],
    pub lock_index: u64,
    /// Sparse hints: `(proof_index, h0 = s0·q_centered, h1 = s1·q_centered)`.
    pub hints: Vec<(u32, u64, u64)>,
    /// Reduction hints for integer-lift recovery.
    pub g0: u64,
    pub g1: u64,
    /// Accepting set (F257 values).
    pub accepting_set: [u16; 2],
    /// Bound on |⟨q,π⟩_Z|.
    pub ip_bound: u64,
}

/// The share selected by the proof from one lock.
#[derive(Clone, Debug)]
pub struct PdkShare {
    pub lock_index: u64,
    pub branch: u8,
    pub value: [u8; 32],
}

// -- Arm --------------------------------------------------------------------

/// Create a PDK lock. No payload, no ciphertexts — just the hint structure.
pub fn arm_pdk_lock(
    query_sparse: &[(u32, u16)],
    accepting_set: [u16; 2],
    statement_hash: [u8; 32],
    lock_index: u64,
    rng: &mut impl RngCore,
) -> Result<PdkLockArtifact, String> {
    let nnz = query_sparse.len();
    if nnz == 0 { return Err("empty query".into()); }

    let s0 = sample_nonzero_g(rng);
    let s1 = sample_nonzero_g(rng);

    let mut hints: Vec<(u32, u64, u64)> = Vec::with_capacity(nnz);
    for &(idx, q_f257) in query_sparse {
        if q_f257 > 256 { return Err(format!("bad F257 value {q_f257}")); }
        let q_g = from_i64_g(centered_f257(q_f257) as i64);
        hints.push((idx, mul_g(s0, q_g), mul_g(s1, q_g)));
    }

    Ok(PdkLockArtifact {
        statement_hash,
        lock_index,
        hints,
        g0: mul_g(s0, 257),
        g1: mul_g(s1, 257),
        accepting_set,
        ip_bound: (nnz as u64).saturating_mul(16384),
    })
}

// -- Decap ------------------------------------------------------------------

/// Evaluate a PDK lock with a proof. Returns the proof-selected share.
pub fn decap_pdk_lock(
    lock: &PdkLockArtifact,
    proof_f257: &[u16],
    proof_len: usize,
) -> Result<PdkShare, String> {
    let (y0, y1) = accumulate_ip(lock, proof_f257, proof_len)?;
    derive_share(lock, y0, y1)
}

/// Streaming decap for large proofs.
pub struct PdkDecapStream<'a> {
    lock: &'a PdkLockArtifact,
    y0: u64, y1: u64,
    filled: usize, proof_len: usize,
    hint_map: std::collections::HashMap<u32, (u64, u64)>,
}

impl<'a> PdkDecapStream<'a> {
    pub fn new(lock: &'a PdkLockArtifact, proof_len: usize) -> Self {
        let mut hm = std::collections::HashMap::with_capacity(lock.hints.len());
        for &(idx, h0, h1) in &lock.hints { hm.insert(idx, (h0, h1)); }
        Self { lock, y0: 0, y1: 0, filled: 0, proof_len, hint_map: hm }
    }

    pub fn absorb_chunk(&mut self, chunk: &[u16]) {
        for (j, &v) in chunk.iter().enumerate() {
            let gi = (self.filled + j) as u32;
            if let Some(&(h0, h1)) = self.hint_map.get(&gi) {
                let pi = from_i64_g(centered_f257(v) as i64);
                self.y0 = add_g(self.y0, mul_g(h0, pi));
                self.y1 = add_g(self.y1, mul_g(h1, pi));
            }
        }
        self.filled += chunk.len();
    }

    pub fn finish(self) -> Result<PdkShare, String> {
        if self.filled != self.proof_len {
            return Err(format!("expected {} elems, got {}", self.proof_len, self.filled));
        }
        derive_share(self.lock, self.y0, self.y1)
    }
}

// -- Global combine ---------------------------------------------------------

/// Combine L proof-derived shares into the final key.
pub fn combine_pdk_shares(shares: &[PdkShare]) -> [u8; 32] {
    use sha2::Digest;
    let mut h = sha2::Sha256::new();
    h.update(b"LFP_PDK_FINAL_KEY_V1");
    for s in shares {
        h.update(&s.lock_index.to_le_bytes());
        h.update(&[s.branch]);
        h.update(&s.value);
    }
    h.finalize().into()
}

/// Derive a nonce for the outer ciphertext from the statement.
pub fn outer_nonce(statement_hash: &[u8; 32]) -> [u8; 12] {
    let h = sha256(&[b"LFP_PDK_OUTER_NONCE_V1", statement_hash]);
    let mut n = [0u8; 12];
    n.copy_from_slice(&h[..12]);
    n
}

/// Encrypt under the final key (PRF-based stream, no auth tag).
pub fn outer_encrypt(final_key: &[u8; 32], stmt: &[u8; 32], msg: &[u8]) -> Vec<u8> {
    xor_crypt(final_key, &outer_nonce(stmt), msg)
}

/// Decrypt under the final key.
pub fn outer_decrypt(final_key: &[u8; 32], stmt: &[u8; 32], ct: &[u8]) -> Vec<u8> {
    outer_encrypt(final_key, stmt, ct)
}

/// Enumerate ALL possible final keys for the attacker (for security analysis).
/// Returns the total count (2^L) and the correct one for reference.
pub fn attacker_candidate_count(num_locks: usize) -> u128 {
    1u128 << (num_locks as u128).min(127)
}

// -- Internals --------------------------------------------------------------

fn accumulate_ip(
    lock: &PdkLockArtifact, proof: &[u16], proof_len: usize,
) -> Result<(u64, u64), String> {
    let mut y0: u64 = 0;
    let mut y1: u64 = 0;
    for &(idx, h0, h1) in &lock.hints {
        let i = idx as usize;
        if i >= proof_len { return Err("hint index oob".into()); }
        let v = if i < proof.len() { proof[i] } else { 0u16 };
        let pi = from_i64_g(centered_f257(v) as i64);
        y0 = add_g(y0, mul_g(h0, pi));
        y1 = add_g(y1, mul_g(h1, pi));
    }
    Ok((y0, y1))
}

fn derive_share(lock: &PdkLockArtifact, y0: u64, y1: u64) -> Result<PdkShare, String> {
    if lock.g0 == 0 || lock.g1 == 0 { return Err("g=0".into()); }

    // Recover s₀ = G₀ / 257 and T_Z = y₀ / s₀.
    let s0 = mul_g(lock.g0, inv_g(257));
    let s1 = mul_g(lock.g1, inv_g(257));

    // T_Z recovery: y₀ · inv(s₀) = T_Z in the field.
    let t_z_field = mul_g(y0, inv_g(s0));

    let t_z_signed: i64 = if t_z_field == 0 {
        0
    } else if t_z_field <= lock.ip_bound {
        t_z_field as i64
    } else if (GOLDILOCKS_P - t_z_field) <= lock.ip_bound {
        -((GOLDILOCKS_P - t_z_field) as i64)
    } else {
        return Err("T_Z out of range".into());
    };

    let residue = (((t_z_signed % 257) + 257) % 257) as u16;

    for j in 0u8..2u8 {
        let a = lock.accepting_set[j as usize];
        let a_centered = centered_f257(a) as i64;
        let a_pos = (((a_centered % 257) + 257) % 257) as u16;
        if residue != a_pos { continue; }

        let share = derive_branch_share(s0, s1, a, lock.lock_index, &lock.statement_hash);
        return Ok(PdkShare { lock_index: lock.lock_index, branch: j, value: share });
    }

    Err("proof doesn't match any accepting-set element".into())
}

fn sample_nonzero_g(rng: &mut impl RngCore) -> u64 {
    loop {
        let mut buf = [0u8; 8];
        rng.fill_bytes(&mut buf);
        let v = u64::from_le_bytes(buf) % GOLDILOCKS_P;
        if v != 0 { return v; }
    }
}

pub fn lock_size_bytes(nnz: usize) -> usize {
    nnz * 20 + 32 + 8 + 16 + 4 + 8 // hints + stmt + idx + g0/g1 + accept + bound
}

// -- Tests ------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{rngs::StdRng, SeedableRng};

    #[test]
    fn test_single_lock_roundtrip() {
        let mut rng = StdRng::seed_from_u64(42);
        let query = vec![(0u32, 3u16), (1, 7), (2, 250), (3, 1)];
        let accept = [5u16, 6u16];
        let stmt = [0xABu8; 32];

        let lock = arm_pdk_lock(&query, accept, stmt, 0, &mut rng).unwrap();
        let share = decap_pdk_lock(&lock, &[1, 0, 0, 2], 4).unwrap();

        // T_Z = 3+0+0+2 = 5, 5 mod 257 = 5 ∈ {5,6} → branch 0
        assert_eq!(share.branch, 0);
        eprintln!("share value: {:02x?}", &share.value[..8]);
        eprintln!("lock size: {} bytes", lock_size_bytes(query.len()));
    }

    #[test]
    fn test_negative_ip() {
        let mut rng = StdRng::seed_from_u64(99);
        let query = vec![(0u32, 256u16), (1, 255)];
        let accept = [252u16, 253u16];
        let stmt = [0xCDu8; 32];

        let lock = arm_pdk_lock(&query, accept, stmt, 1, &mut rng).unwrap();
        // π=[3,1] → T_Z = -3 + -2 = -5, -5 mod 257 = 252
        let share = decap_pdk_lock(&lock, &[3, 1], 2).unwrap();
        assert_eq!(share.branch, 0); // 252 is accepting_set[0]
    }

    #[test]
    fn test_large_ip_wraparound() {
        let mut rng = StdRng::seed_from_u64(123);
        let query = vec![(0u32, 128u16)];
        let accept = [193u16, 194u16];
        let stmt = [0x33u8; 32];

        let lock = arm_pdk_lock(&query, accept, stmt, 2, &mut rng).unwrap();
        // π=[128] → T_Z = 128*128 = 16384, 16384 mod 257 = 193
        let share = decap_pdk_lock(&lock, &[128], 1).unwrap();
        assert_eq!(share.branch, 0);
    }

    #[test]
    fn test_wrong_proof_fails() {
        let mut rng = StdRng::seed_from_u64(7);
        let query = vec![(0u32, 10u16), (1, 20)];
        let accept = [30u16, 31u16];
        let stmt = [0x11u8; 32];

        let lock = arm_pdk_lock(&query, accept, stmt, 3, &mut rng).unwrap();

        // Valid: π=[1,1] → IP=30 ∈ {30,31} ✓
        assert!(decap_pdk_lock(&lock, &[1, 1], 2).is_ok());

        // Invalid: π=[2,2] → IP=60, 60 mod 257 = 60 ∉ {30,31}
        assert!(decap_pdk_lock(&lock, &[2, 2], 2).is_err());
    }

    #[test]
    fn test_attacker_sees_both_shares_but_different() {
        let mut rng = StdRng::seed_from_u64(55);
        let query = vec![(0u32, 10u16), (1, 20)];
        let accept = [30u16, 31u16];
        let stmt = [0x22u8; 32];

        let lock = arm_pdk_lock(&query, accept, stmt, 4, &mut rng).unwrap();

        // Attacker recovers s₀, s₁ from g₀, g₁.
        let s0 = mul_g(lock.g0, inv_g(257));
        let s1 = mul_g(lock.g1, inv_g(257));

        // Attacker computes BOTH branch shares.
        let share_a0 = derive_branch_share(s0, s1, accept[0], lock.lock_index, &stmt);
        let share_a1 = derive_branch_share(s0, s1, accept[1], lock.lock_index, &stmt);

        // The two shares MUST be different (otherwise no ambiguity).
        assert_ne!(share_a0, share_a1, "branch shares must differ for ambiguity");

        // Honest decapper gets exactly one of them.
        let honest = decap_pdk_lock(&lock, &[1, 1], 2).unwrap();
        assert!(honest.value == share_a0 || honest.value == share_a1);
    }

    #[test]
    fn test_multi_lock_e2e() {
        let mut rng = StdRng::seed_from_u64(2025);
        let l = 8; // 8 locks (production: L=256)
        let stmt = [0x99u8; 32];
        let query = vec![(0u32, 10u16), (1, 20)];
        let accept = [30u16, 31u16];
        let proof: Vec<u16> = vec![1, 1]; // IP = 30

        // Create L locks.
        let mut locks: Vec<PdkLockArtifact> = Vec::new();
        for i in 0..l {
            locks.push(arm_pdk_lock(&query, accept, stmt, i as u64, &mut rng).unwrap());
        }

        // Honest decapper evaluates all locks.
        let mut shares: Vec<PdkShare> = Vec::new();
        for lock in &locks {
            shares.push(decap_pdk_lock(lock, &proof, 2).unwrap());
        }
        let final_key = combine_pdk_shares(&shares);

        // Encrypt a message externally.
        let message = b"bitcoin_address_placeholder_32by";
        let outer_ct = outer_encrypt(&final_key, &stmt, message);
        let recovered = outer_decrypt(&final_key, &stmt, &outer_ct);
        assert_eq!(recovered, message);

        // Attacker: can derive 2 candidate shares per lock → 2^L candidate keys.
        // Verify: wrong branch combination gives a DIFFERENT final key.
        let mut wrong_shares = shares.clone();
        // Flip one branch.
        let s0 = mul_g(locks[0].g0, inv_g(257));
        let s1 = mul_g(locks[0].g1, inv_g(257));
        let other_branch = if wrong_shares[0].branch == 0 { 1u8 } else { 0u8 };
        wrong_shares[0].value = derive_branch_share(
            s0, s1, accept[other_branch as usize], 0, &stmt,
        );
        wrong_shares[0].branch = other_branch;
        let wrong_key = combine_pdk_shares(&wrong_shares);
        assert_ne!(wrong_key, final_key, "wrong branch must give different final key");

        let wrong_message = outer_decrypt(&wrong_key, &stmt, &outer_ct);
        assert_ne!(wrong_message.as_slice(), message.as_slice(),
            "wrong key must produce garbage");

        let total = l * lock_size_bytes(query.len());
        eprintln!(
            "L={l} locks, per-lock={} bytes, total={} bytes ({:.1} KB), attacker candidates=2^{l}",
            lock_size_bytes(query.len()), total, total as f64 / 1024.0,
        );
    }
}
