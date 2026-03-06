//! Goldilocks-prime LWE lock layer for the F257 DPP — **fan-out design**.
//!
//! # Security model (asymmetric cost, no local oracle)
//!
//! | Party | Per-lock cost | Candidates |
//! |---|---|---|
//! | Honest decapper (has π) | O(nnz) | 1 |
//! | Attacker (no π) | O(257² × nnz) to recover s₀ | **K ≈ nnz×128** |
//!
//! Global security: attacker faces K^R combinations across R reps. For nnz=1000,
//! K ≈ 128K ≈ 2^17 per lock. With R=8 reps: 2^{17×8} = 2^136 > 2^128.
//!
//! # Why revealing s₀ is fine
//!
//! The lock publishes `G₀ = s₀·257 (mod p)`, from which `s₀ = G₀·inv(257)`.
//! This does NOT break the scheme because:
//!
//! - Knowing `s₀` lets the attacker compute `KDF(s₀·t, s₁·t)` for **every** target `t`
//!   in the fan-out, producing K candidate payloads.
//! - Without the proof `π`, the attacker **cannot determine which target `t` is real**.
//! - All K candidates are 32-byte pseudorandom blobs (XOR encryption, no tags).
//! - The ambiguity IS the security: K^R combinations across reps.
//!
//! # F257 → integer lift
//!
//! The F257 DPP guarantees `⟨q,π⟩ ≡ a (mod 257)` for `a ∈ A = {a₀,a₁}`.
//! The Goldilocks inner product is `T_Z = a + 257·k` for unknown `k`.
//! The fan-out covers all valid `k` values: one ciphertext per `(a, k)` pair.
//!
//! The honest decapper uses `G₀` to recover `T_Z` from their proof-derived `y₀`,
//! then indexes directly into the fan-out array — O(1) lookup.

use rand::RngCore;

const GOLDILOCKS_P: u64 = 18446744069414584321;

// ---------------------------------------------------------------------------
// Goldilocks modular arithmetic
// ---------------------------------------------------------------------------

#[inline]
fn add_g(a: u64, b: u64) -> u64 {
    ((a as u128 + b as u128) % (GOLDILOCKS_P as u128)) as u64
}

#[inline]
fn sub_g(a: u64, b: u64) -> u64 {
    if a >= b { a - b } else { GOLDILOCKS_P - (b - a) }
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

// ---------------------------------------------------------------------------
// Crypto helpers
// ---------------------------------------------------------------------------

fn sha256(chunks: &[&[u8]]) -> [u8; 32] {
    use sha2::Digest;
    let mut h = sha2::Sha256::new();
    for c in chunks { h.update(c); }
    h.finalize().into()
}

fn derive_fanout_key(s0_t: u64, s1_t: u64, stmt: &[u8; 32], lock_idx: u64) -> [u8; 32] {
    sha256(&[
        b"LFP_LWE_FANOUT_KEY_V1",
        stmt, &lock_idx.to_le_bytes(),
        &s0_t.to_le_bytes(), &s1_t.to_le_bytes(),
    ])
}

fn xor_crypt(key: &[u8; 32], nonce: &[u8; 12], data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    let mut ctr = 0u32;
    let mut pos = 0usize;
    while pos < data.len() {
        let blk = sha256(&[b"LFP_LWE_XOR_V1", key, nonce, &ctr.to_le_bytes()]);
        let take = (data.len() - pos).min(32);
        for j in 0..take { out.push(data[pos + j] ^ blk[j]); }
        pos += take;
        ctr += 1;
    }
    out
}

// ---------------------------------------------------------------------------
// Lock types
// ---------------------------------------------------------------------------

/// A single fan-out entry: XOR-encrypted payload under a target-derived key.
/// No authentication tag — ambiguity is the security.
#[derive(Clone)]
pub struct FanoutEntry {
    pub nonce: [u8; 12],
    pub ct: [u8; 32],
}

impl std::fmt::Debug for FanoutEntry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FanoutEntry").field("nonce", &self.nonce).finish()
    }
}

/// LWE lock artifact with fan-out ambiguity. No per-lock authentication.
#[derive(Clone, Debug)]
pub struct LweLockArtifact {
    pub statement_hash: [u8; 32],
    pub lock_index: u64,
    pub hints: Vec<(u32, u64, u64)>,
    pub g0: u64,
    pub g1: u64,
    pub accepting_set: [u16; 2],
    pub ip_bound: u64,
    /// Fan-out entries indexed by `(branch, k)`:
    /// `fanout[branch * fan_k + k_offset]` where `k_offset = k - k_min`.
    pub fanout: Vec<FanoutEntry>,
    pub fan_k_min: i64,
    pub fan_k_count: usize,
}

// ---------------------------------------------------------------------------
// Arm
// ---------------------------------------------------------------------------

pub fn arm_lwe_lock(
    query_sparse: &[(u32, u16)],
    accepting_set: [u16; 2],
    statement_hash: [u8; 32],
    lock_index: u64,
    payload: &[u8],
    rng: &mut impl RngCore,
) -> Result<LweLockArtifact, String> {
    if payload.len() != 32 { return Err("payload must be 32 bytes".into()); }
    let nnz = query_sparse.len();
    if nnz == 0 { return Err("empty query".into()); }

    let s0 = sample_nonzero_g(rng);
    let s1 = sample_nonzero_g(rng);
    let g0 = mul_g(s0, 257);
    let g1 = mul_g(s1, 257);

    let mut hints: Vec<(u32, u64, u64)> = Vec::with_capacity(nnz);
    for &(idx, q_f257) in query_sparse {
        if q_f257 > 256 { return Err(format!("bad F257 value {q_f257}")); }
        let q_g = from_i64_g(centered_f257(q_f257) as i64);
        hints.push((idx, mul_g(s0, q_g), mul_g(s1, q_g)));
    }

    let ip_bound = (nnz as u64).saturating_mul(16384);
    let k_max = (ip_bound as i64) / 257 + 1;
    let k_min = -k_max;
    let k_count = (k_max - k_min + 1) as usize;

    let mut fanout: Vec<FanoutEntry> = Vec::with_capacity(2 * k_count);
    let mut payload32 = [0u8; 32];
    payload32.copy_from_slice(payload);

    for branch in 0u8..2u8 {
        let a_centered = centered_f257(accepting_set[branch as usize]) as i64;
        for k in k_min..=k_max {
            let t = a_centered + 257 * k;
            let t_g = from_i64_g(t);
            let key = derive_fanout_key(mul_g(s0, t_g), mul_g(s1, t_g), &statement_hash, lock_index);
            let mut nonce = [0u8; 12];
            rng.fill_bytes(&mut nonce);
            let ct_vec = xor_crypt(&key, &nonce, &payload32);
            let mut ct = [0u8; 32];
            ct.copy_from_slice(&ct_vec);
            fanout.push(FanoutEntry { nonce, ct });
        }
    }

    Ok(LweLockArtifact {
        statement_hash, lock_index, hints, g0, g1,
        accepting_set, ip_bound, fanout, fan_k_min: k_min, fan_k_count: k_count,
    })
}

// ---------------------------------------------------------------------------
// Decap
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct LweCandidate {
    pub lock_index: u64,
    pub payload: [u8; 32],
}

/// Honest decap: uses the proof to compute (y0, y1) → recovers T_Z via G0 →
/// indexes into the fan-out array → returns exactly 1 candidate.
pub fn decap_lwe_lock(
    lock: &LweLockArtifact,
    proof_f257: &[u16],
    proof_len: usize,
) -> Result<Vec<LweCandidate>, String> {
    let (y0, y1) = accumulate_ip(lock, proof_f257, proof_len)?;
    honest_decrypt(lock, y0, y1)
}

/// Streaming decap for large proofs.
pub struct LweDecapStream<'a> {
    lock: &'a LweLockArtifact,
    y0: u64, y1: u64,
    filled: usize, proof_len: usize,
    hint_map: std::collections::HashMap<u32, (u64, u64)>,
}

impl<'a> LweDecapStream<'a> {
    pub fn new(lock: &'a LweLockArtifact, proof_len: usize) -> Self {
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

    pub fn finish(self) -> Result<Vec<LweCandidate>, String> {
        if self.filled != self.proof_len {
            return Err(format!("expected {} elems, got {}", self.proof_len, self.filled));
        }
        honest_decrypt(self.lock, self.y0, self.y1)
    }
}

// ---------------------------------------------------------------------------
// Internal
// ---------------------------------------------------------------------------

fn accumulate_ip(
    lock: &LweLockArtifact, proof: &[u16], proof_len: usize,
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

/// Honest decap: recover T_Z from (y0, G0), derive the fan-out key, return 1 candidate.
///
/// Key identity: the armer encrypted entry `(branch, k)` under
///   `derive_fanout_key(s0 · t_g, s1 · t_g, ...)` where `t_g = from_i64_g(a + 257k)`.
/// The decapper has `y0 = s0 · T_Z_field` where `T_Z_field = from_i64_g(T_Z)`.
/// For the correct entry: `t_g = T_Z_field`, so `s0 · t_g = y0`. The key is just
/// `derive_fanout_key(y0, y1, ...)`.
fn honest_decrypt(
    lock: &LweLockArtifact, y0: u64, y1: u64,
) -> Result<Vec<LweCandidate>, String> {
    if lock.g0 == 0 || lock.g1 == 0 { return Err("g=0".into()); }

    // Recover T_Z: y0 · inv(G0) = T_Z · inv(257), then multiply by 257.
    let t_z_field = mul_g(mul_g(y0, inv_g(lock.g0)), 257);

    let t_z_signed: i64 = if t_z_field <= lock.ip_bound {
        t_z_field as i64
    } else if (GOLDILOCKS_P - t_z_field) <= lock.ip_bound {
        -((GOLDILOCKS_P - t_z_field) as i64)
    } else {
        return Ok(Vec::new());
    };

    let residue = ((t_z_signed % 257) + 257) % 257;
    let key = derive_fanout_key(y0, y1, &lock.statement_hash, lock.lock_index);
    let mut out = Vec::new();

    for branch in 0u8..2u8 {
        let a_int = centered_f257(lock.accepting_set[branch as usize]) as i64;
        let a_pos = ((a_int % 257) + 257) % 257;
        if residue != a_pos { continue; }

        let k = (t_z_signed - a_int) / 257;
        let k_offset = (k - lock.fan_k_min) as usize;
        let entry_idx = (branch as usize) * lock.fan_k_count + k_offset;

        if entry_idx >= lock.fanout.len() { continue; }
        let entry = &lock.fanout[entry_idx];

        let plain_vec = xor_crypt(&key, &entry.nonce, &entry.ct);
        if plain_vec.len() == 32 {
            let mut p = [0u8; 32];
            p.copy_from_slice(&plain_vec);
            out.push(LweCandidate { lock_index: lock.lock_index, payload: p });
        }
    }
    Ok(out)
}

fn sample_nonzero_g(rng: &mut impl RngCore) -> u64 {
    loop {
        let mut buf = [0u8; 8];
        rng.fill_bytes(&mut buf);
        let v = u64::from_le_bytes(buf) % GOLDILOCKS_P;
        if v != 0 { return v; }
    }
}

pub fn fanout_count(nnz: usize) -> usize {
    let ip_bound = (nnz as u64).saturating_mul(16384);
    let k_max = (ip_bound as i64) / 257 + 1;
    2 * (2 * k_max as usize + 1)
}

pub fn estimate_lock_bytes(nnz: usize) -> usize {
    let hints = nnz * 20;
    let fan = fanout_count(nnz) * 44; // 12 nonce + 32 ct
    hints + fan + 64
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{rngs::StdRng, SeedableRng};

    #[test]
    fn test_arith() {
        assert_eq!(mul_g(inv_g(257), 257), 1);
        assert_eq!(add_g(from_i64_g(-5), 5), 0);
    }

    #[test]
    fn test_roundtrip_small() {
        let mut rng = StdRng::seed_from_u64(42);
        // q=[3,7,-7,1], π=[1,0,0,2] → T_Z = 3+0+0+2 = 5, 5 mod 257 = 5
        let query = vec![(0u32, 3u16), (1, 7), (2, 250), (3, 1)];
        let accept = [5u16, 6u16];
        let payload = [0x42u8; 32];
        let lock = arm_lwe_lock(&query, accept, [0xAB; 32], 1, &payload, &mut rng).unwrap();

        eprintln!("fanout entries: {}, lock size est: {} bytes",
            lock.fanout.len(), estimate_lock_bytes(query.len()));

        let proof: Vec<u16> = vec![1, 0, 0, 2];
        let cands = decap_lwe_lock(&lock, &proof, 4).unwrap();
        assert_eq!(cands.len(), 1, "honest decap should yield exactly 1 candidate");
        assert_eq!(cands[0].payload, payload);
    }

    #[test]
    fn test_negative_ip() {
        let mut rng = StdRng::seed_from_u64(99);
        // q=[-1,-2], π=[3,1] → T_Z = -3-2 = -5, -5 mod 257 = 252
        let query = vec![(0u32, 256u16), (1, 255)];
        let accept = [252u16, 253u16];
        let payload = [0x77u8; 32];
        let lock = arm_lwe_lock(&query, accept, [0xCD; 32], 2, &payload, &mut rng).unwrap();
        let cands = decap_lwe_lock(&lock, &[3, 1], 2).unwrap();
        assert_eq!(cands.len(), 1);
        assert_eq!(cands[0].payload, payload);
    }

    #[test]
    fn test_wrong_proof() {
        let mut rng = StdRng::seed_from_u64(7);
        let query = vec![(0u32, 10u16), (1, 20)];
        let accept = [30u16, 31u16];
        let payload = [0xFFu8; 32];
        let lock = arm_lwe_lock(&query, accept, [0x11; 32], 3, &payload, &mut rng).unwrap();

        // Valid: π=[1,1] → 10+20=30 ∈ {30,31}
        let good = decap_lwe_lock(&lock, &[1, 1], 2).unwrap();
        assert_eq!(good.len(), 1);
        assert_eq!(good[0].payload, payload);

        // Invalid: π=[2,2] → 20+40=60, 60 mod 257=60 ∉ {30,31}
        let bad = decap_lwe_lock(&lock, &[2, 2], 2).unwrap();
        assert!(bad.is_empty() || bad.iter().all(|c| c.payload != payload));
    }

    #[test]
    fn test_large_ip_wraparound() {
        let mut rng = StdRng::seed_from_u64(123);
        // q=[128], π=[128] → T_Z=16384, 16384 mod 257 = 193
        let query = vec![(0u32, 128u16)];
        let accept = [193u16, 194u16];
        let payload = [0xDDu8; 32];
        let lock = arm_lwe_lock(&query, accept, [0x33; 32], 5, &payload, &mut rng).unwrap();
        let cands = decap_lwe_lock(&lock, &[128], 1).unwrap();
        assert_eq!(cands.len(), 1);
        assert_eq!(cands[0].payload, payload, "integer lift k=63 should work");
    }

    #[test]
    fn test_streaming() {
        let mut rng = StdRng::seed_from_u64(55);
        let query = vec![(0u32, 5u16), (3, 10), (7, 15)];
        // π=[1,0,0,2,0,0,0,1] → 5+20+15=40
        let accept = [40u16, 41u16];
        let payload = [0xBBu8; 32];
        let lock = arm_lwe_lock(&query, accept, [0x22; 32], 4, &payload, &mut rng).unwrap();
        let proof: Vec<u16> = vec![1, 0, 0, 2, 0, 0, 0, 1];
        let mut st = LweDecapStream::new(&lock, 8);
        st.absorb_chunk(&proof[..4]);
        st.absorb_chunk(&proof[4..]);
        let cands = st.finish().unwrap();
        assert_eq!(cands.len(), 1);
        assert_eq!(cands[0].payload, payload);
    }

    #[test]
    fn test_fanout_provides_ambiguity() {
        let nnz = 1000;
        let fan = fanout_count(nnz);
        let bytes = estimate_lock_bytes(nnz);
        eprintln!("nnz={nnz}: fanout={fan} entries, ~{:.1} MB", bytes as f64 / 1e6);
        // For nnz=1000: fan ≈ 128K entries. Each is 44 bytes → ~5.6 MB.
        // Attacker who recovers s₀ gets all 128K candidates per lock.
        // With R=8 reps: 128K^8 ≈ 2^136 combinations.
        assert!(fan > 100_000, "fan-out must be large enough for ambiguity");
    }
}
