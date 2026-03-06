//! Goldilocks-prime LWE lock layer — **multi-lock XOR-share design**.
//!
//! # Security model
//!
//! L independent locks (from one or more armers), each with 2 branch ciphertexts
//! (one per accepting-set element). The honest decapper (with proof π) recovers
//! exactly 1 payload per lock; the attacker (without π) gets 2 candidates per lock.
//!
//! Final key = XOR of all L lock payloads. Outer ciphertext uses a PRF-based
//! stream cipher (SHA-256 in counter mode) — **not** a true one-time pad.
//! No authentication tag on the outer ciphertext.
//!
//! | Party | Per lock | Global (L locks) |
//! |---|---|---|
//! | Honest decapper | 1 candidate, O(nnz) | 1 final key |
//! | Attacker (no π) | 2 candidates | 2^L candidate final keys |
//!
//! # Security analysis
//!
//! Security is **computational** (PRF/random-oracle model on SHA-256), not
//! information-theoretic. The XOR-share combine is vulnerable to meet-in-the-middle
//! (MITM) when the output is checkable (e.g., the final key decrypts to a known-format
//! value like a Bitcoin address).
//!
//! | Output type | Best attack | L for 128-bit PQ |
//! |---|---|---|
//! | Non-checkable (random seed) | Brute force 2^L | L = 128 |
//! | Checkable (address, key format) | MITM / Grover 2^(L/2) | **L = 256** |
//!
//! Total lock size at L=256, nnz=1000: **~5 MB**.
//!
//! # No per-lock oracle
//!
//! The lock publishes G₀ = s₀·257, which reveals s₀. This is by design: the
//! attacker CAN decrypt both branches per lock, producing 2 candidate payloads.
//! But without π they cannot determine which is correct. There are no tags, no
//! structural distinguishers, and no probes — the per-lock recognizer is eliminated.
//!
//! # Integer lift
//!
//! The F257 DPP gives ⟨q,π⟩ ≡ a (mod 257) but the Goldilocks inner product
//! is T_Z = a + 257·k. The reduction hint G₀ = s₀·257 lets the honest decapper
//! recover T_Z and select the correct branch in O(1). The attacker recovers s₀
//! from G₀ (by design) but still gets both branches — the ambiguity comes from
//! not knowing which branch the proof selects.

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

// -- Crypto helpers ---------------------------------------------------------

fn sha256(chunks: &[&[u8]]) -> [u8; 32] {
    use sha2::Digest;
    let mut h = sha2::Sha256::new();
    for c in chunks { h.update(c); }
    h.finalize().into()
}

fn derive_branch_key(s0a: u64, s1a: u64, stmt: &[u8; 32], lock_idx: u64, branch: u8) -> [u8; 32] {
    sha256(&[
        b"LFP_LWE_OTP_KEY_V1",
        stmt, &lock_idx.to_le_bytes(), &[branch],
        &s0a.to_le_bytes(), &s1a.to_le_bytes(),
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

// -- Lock types -------------------------------------------------------------

/// Branch ciphertext (one per accepting-set element). No authentication tag.
#[derive(Clone, Debug)]
pub struct BranchCt {
    pub nonce: [u8; 12],
    pub ct: [u8; 32],
}

/// OTP lock artifact from a single armer. Compact: just hints + 2 branch cts.
#[derive(Clone, Debug)]
pub struct OtpLockArtifact {
    pub statement_hash: [u8; 32],
    pub lock_index: u64,
    /// Sparse LWE hints: `(proof_index, h0, h1)` where `h_j = s_j · q_i_centered`.
    pub hints: Vec<(u32, u64, u64)>,
    /// Reduction hints for integer-lift recovery.
    pub g0: u64,
    pub g1: u64,
    /// Two branch ciphertexts. `cts[j]` encrypts payload under branch j's key.
    pub cts: [BranchCt; 2],
    /// Accepting set (F257 values).
    pub accepting_set: [u16; 2],
    /// Bound on |⟨q,π⟩_Z| (centered representation).
    pub ip_bound: u64,
}

/// Candidate payload from one lock. No local authentication — correctness is
/// determined by global OTP reconstruction across all armers.
#[derive(Clone, Debug)]
pub struct OtpCandidate {
    pub lock_index: u64,
    pub branch: u8,
    pub payload: [u8; 32],
}

// -- Arm --------------------------------------------------------------------

/// Arm a single OTP lock. Called once per armer per statement.
///
/// `query_sparse`: `(proof_index, f257_coeff)` for nonzero combined-query entries.
/// `accepting_set`: two F257 values (shifted by the public offset).
/// `payload`: exactly 32 bytes (this armer's XOR share of the final key).
pub fn arm_otp_lock(
    query_sparse: &[(u32, u16)],
    accepting_set: [u16; 2],
    statement_hash: [u8; 32],
    lock_index: u64,
    payload: &[u8],
    rng: &mut impl RngCore,
) -> Result<OtpLockArtifact, String> {
    if payload.len() != 32 { return Err("payload must be 32 bytes".into()); }
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

    let ip_bound = (nnz as u64).saturating_mul(16384);

    let mut cts: [BranchCt; 2] = [
        BranchCt { nonce: [0u8; 12], ct: [0u8; 32] },
        BranchCt { nonce: [0u8; 12], ct: [0u8; 32] },
    ];
    for j in 0u8..2u8 {
        let a_g = from_i64_g(centered_f257(accepting_set[j as usize]) as i64);
        let key = derive_branch_key(
            mul_g(s0, a_g), mul_g(s1, a_g),
            &statement_hash, lock_index, j,
        );
        let mut nonce = [0u8; 12];
        rng.fill_bytes(&mut nonce);
        let ct_vec = xor_crypt(&key, &nonce, payload);
        let mut ct = [0u8; 32];
        ct.copy_from_slice(&ct_vec);
        cts[j as usize] = BranchCt { nonce, ct };
    }

    Ok(OtpLockArtifact {
        statement_hash, lock_index, hints,
        g0: mul_g(s0, 257), g1: mul_g(s1, 257),
        cts, accepting_set, ip_bound,
    })
}

// -- Decap ------------------------------------------------------------------

/// Decap a single OTP lock. Returns exactly 1 candidate for honest decapper.
pub fn decap_otp_lock(
    lock: &OtpLockArtifact,
    proof_f257: &[u16],
    proof_len: usize,
) -> Result<Vec<OtpCandidate>, String> {
    let (y0, y1) = accumulate_ip(lock, proof_f257, proof_len)?;
    decrypt_branch(lock, y0, y1)
}

/// Streaming decap for large proofs.
pub struct OtpDecapStream<'a> {
    lock: &'a OtpLockArtifact,
    y0: u64,
    y1: u64,
    filled: usize,
    proof_len: usize,
    hint_map: std::collections::HashMap<u32, (u64, u64)>,
}

impl<'a> OtpDecapStream<'a> {
    pub fn new(lock: &'a OtpLockArtifact, proof_len: usize) -> Self {
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

    pub fn finish(self) -> Result<Vec<OtpCandidate>, String> {
        if self.filled != self.proof_len {
            return Err(format!("expected {} elems, got {}", self.proof_len, self.filled));
        }
        decrypt_branch(self.lock, self.y0, self.y1)
    }
}

// -- Global OTP combine -----------------------------------------------------

/// Combine L lock payloads into the final key via XOR.
///
/// Each lock contributes exactly 1 candidate (selected by the honest decapper).
/// Returns the 32-byte final key.
pub fn combine_xor_shares(shares: &[[u8; 32]]) -> [u8; 32] {
    let mut key = [0u8; 32];
    for s in shares {
        for i in 0..32 { key[i] ^= s[i]; }
    }
    key
}

/// Derive a per-statement nonce for the outer ciphertext.
///
/// Using a deterministic nonce is safe because the final key is single-use
/// (each WE instance / statement produces a unique key). Deriving from the
/// statement hash prevents accidental reuse across instances.
pub fn outer_nonce(statement_hash: &[u8; 32]) -> [u8; 12] {
    let h = sha256(&[b"LFP_OUTER_NONCE_V1", statement_hash]);
    let mut n = [0u8; 12];
    n.copy_from_slice(&h[..12]);
    n
}

/// Encrypt a message under the final key. PRF-based stream cipher (SHA-256 CTR),
/// **not** a true one-time pad. No authentication tag — the lack of authentication
/// is load-bearing for the no-local-oracle property.
pub fn outer_encrypt(final_key: &[u8; 32], statement_hash: &[u8; 32], message: &[u8]) -> Vec<u8> {
    xor_crypt(final_key, &outer_nonce(statement_hash), message)
}

/// Decrypt a message under the final key.
pub fn outer_decrypt(final_key: &[u8; 32], statement_hash: &[u8; 32], ciphertext: &[u8]) -> Vec<u8> {
    outer_encrypt(final_key, statement_hash, ciphertext)
}

// -- Internals --------------------------------------------------------------

fn accumulate_ip(
    lock: &OtpLockArtifact, proof: &[u16], proof_len: usize,
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

fn decrypt_branch(
    lock: &OtpLockArtifact, y0: u64, y1: u64,
) -> Result<Vec<OtpCandidate>, String> {
    if lock.g0 == 0 || lock.g1 == 0 { return Err("g=0".into()); }

    let t_z_field = mul_g(mul_g(y0, inv_g(lock.g0)), 257);

    let t_z_signed: i64 = if t_z_field <= lock.ip_bound {
        t_z_field as i64
    } else if (GOLDILOCKS_P - t_z_field) <= lock.ip_bound {
        -((GOLDILOCKS_P - t_z_field) as i64)
    } else {
        return Ok(Vec::new());
    };

    let residue = ((t_z_signed % 257) + 257) % 257;
    let mut out = Vec::new();

    for j in 0u8..2u8 {
        let a_int = centered_f257(lock.accepting_set[j as usize]) as i64;
        let a_pos = ((a_int % 257) + 257) % 257;
        if residue != a_pos { continue; }

        // Recover s0·a = y0 − G0·k and s1·a = y1 − G1·k.
        let k = (t_z_signed - a_int) / 257;
        let k_g = from_i64_g(k);
        let s0a = add_g(y0, GOLDILOCKS_P - mul_g(lock.g0, k_g));
        let s1a = add_g(y1, GOLDILOCKS_P - mul_g(lock.g1, k_g));

        let key = derive_branch_key(s0a, s1a, &lock.statement_hash, lock.lock_index, j);
        let plain = xor_crypt(&key, &lock.cts[j as usize].nonce, &lock.cts[j as usize].ct);
        if plain.len() == 32 {
            let mut p = [0u8; 32];
            p.copy_from_slice(&plain);
            out.push(OtpCandidate { lock_index: lock.lock_index, branch: j, payload: p });
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

/// Lock size in bytes (hints + 2 branch cts + metadata).
pub fn lock_size_bytes(nnz: usize) -> usize {
    nnz * 20 + 2 * 44 + 32 + 8 + 16 + 4 + 8
}

// -- Tests ------------------------------------------------------------------

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
    fn test_roundtrip_single_armer() {
        let mut rng = StdRng::seed_from_u64(42);
        let query = vec![(0u32, 3u16), (1, 7), (2, 250), (3, 1)];
        let accept = [5u16, 6u16];
        let payload = [0x42u8; 32];

        let lock = arm_otp_lock(&query, accept, [0xAB; 32], 1, &payload, &mut rng).unwrap();
        eprintln!("lock size: {} bytes", lock_size_bytes(query.len()));

        // π=[1,0,0,2] → T_Z = 3+0+0+2 = 5, 5 mod 257 = 5 ∈ {5,6} ✓
        let cands = decap_otp_lock(&lock, &[1, 0, 0, 2], 4).unwrap();
        assert_eq!(cands.len(), 1);
        assert_eq!(cands[0].payload, payload);
    }

    #[test]
    fn test_negative_ip() {
        let mut rng = StdRng::seed_from_u64(99);
        let query = vec![(0u32, 256u16), (1, 255)];
        let accept = [252u16, 253u16];
        let payload = [0x77u8; 32];

        let lock = arm_otp_lock(&query, accept, [0xCD; 32], 2, &payload, &mut rng).unwrap();
        let cands = decap_otp_lock(&lock, &[3, 1], 2).unwrap();
        assert_eq!(cands.len(), 1);
        assert_eq!(cands[0].payload, payload);
    }

    #[test]
    fn test_large_ip_wraparound() {
        let mut rng = StdRng::seed_from_u64(123);
        let query = vec![(0u32, 128u16)];
        let accept = [193u16, 194u16];
        let payload = [0xDDu8; 32];

        let lock = arm_otp_lock(&query, accept, [0x33; 32], 5, &payload, &mut rng).unwrap();
        let cands = decap_otp_lock(&lock, &[128], 1).unwrap();
        assert_eq!(cands.len(), 1);
        assert_eq!(cands[0].payload, payload);
    }

    #[test]
    fn test_wrong_proof_no_payload() {
        let mut rng = StdRng::seed_from_u64(7);
        let query = vec![(0u32, 10u16), (1, 20)];
        let accept = [30u16, 31u16];
        let payload = [0xFFu8; 32];

        let lock = arm_otp_lock(&query, accept, [0x11; 32], 3, &payload, &mut rng).unwrap();

        let good = decap_otp_lock(&lock, &[1, 1], 2).unwrap();
        assert_eq!(good.len(), 1);
        assert_eq!(good[0].payload, payload);

        let bad = decap_otp_lock(&lock, &[2, 2], 2).unwrap();
        assert!(bad.is_empty() || bad.iter().all(|c| c.payload != payload));
    }

    #[test]
    fn test_streaming() {
        let mut rng = StdRng::seed_from_u64(55);
        let query = vec![(0u32, 5u16), (3, 10), (7, 15)];
        let accept = [40u16, 41u16];
        let payload = [0xBBu8; 32];

        let lock = arm_otp_lock(&query, accept, [0x22; 32], 4, &payload, &mut rng).unwrap();
        let proof: Vec<u16> = vec![1, 0, 0, 2, 0, 0, 0, 1];
        let mut st = OtpDecapStream::new(&lock, 8);
        st.absorb_chunk(&proof[..4]);
        st.absorb_chunk(&proof[4..]);
        let cands = st.finish().unwrap();
        assert_eq!(cands.len(), 1);
        assert_eq!(cands[0].payload, payload);
    }

    #[test]
    fn test_multi_lock_xor_e2e() {
        let mut rng = StdRng::seed_from_u64(2025);
        let message = b"this is the secret message!!!!!!"; // exactly 32 bytes
        assert_eq!(message.len(), 32);

        let l = 8; // 8 locks for this test (production: L=256 for 128-bit PQ checkable)
        let stmt_hash = [0x99u8; 32];
        let query = vec![(0u32, 10u16), (1, 20)];
        let accept = [30u16, 31u16];
        let proof: Vec<u16> = vec![1, 1]; // IP = 10+20 = 30 ∈ {30,31}

        // Generate L XOR shares whose XOR = final_key.
        let mut shares: Vec<[u8; 32]> = Vec::new();
        let mut xor_acc = [0u8; 32];
        for _ in 0..(l - 1) {
            let mut s = [0u8; 32];
            rng.fill_bytes(&mut s);
            for i in 0..32 { xor_acc[i] ^= s[i]; }
            shares.push(s);
        }
        let mut final_key = [0u8; 32];
        rng.fill_bytes(&mut final_key);
        let mut last_share = [0u8; 32];
        for i in 0..32 { last_share[i] = final_key[i] ^ xor_acc[i]; }
        shares.push(last_share);
        assert_eq!(combine_xor_shares(&shares), final_key);

        // Encrypt message under the final key (PRF-based, statement-bound nonce).
        let outer_ct = outer_encrypt(&final_key, &stmt_hash, message);

        // Create L locks, each with independent (s0, s1) and its XOR share.
        let mut locks: Vec<OtpLockArtifact> = Vec::new();
        for (i, share) in shares.iter().enumerate() {
            let lock = arm_otp_lock(
                &query, accept, stmt_hash, i as u64, share, &mut rng,
            ).unwrap();
            locks.push(lock);
        }

        // Honest decapper: decap all locks -> combine -> decrypt.
        let mut recovered_shares: Vec<[u8; 32]> = Vec::new();
        for lock in &locks {
            let cands = decap_otp_lock(lock, &proof, 2).unwrap();
            assert_eq!(cands.len(), 1);
            recovered_shares.push(cands[0].payload);
        }
        let recovered_key = combine_xor_shares(&recovered_shares);
        assert_eq!(recovered_key, final_key);

        let recovered_message = outer_decrypt(&recovered_key, &stmt_hash, &outer_ct);
        assert_eq!(recovered_message, message);

        let total_bytes = l * lock_size_bytes(query.len());
        eprintln!(
            "L={l} locks, nnz={}, per-lock={} bytes, total={} bytes ({:.1} KB)",
            query.len(), lock_size_bytes(query.len()), total_bytes, total_bytes as f64 / 1024.0
        );
    }
}
