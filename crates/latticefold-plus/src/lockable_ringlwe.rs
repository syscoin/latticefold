//! Lock layer for Theorem-4.3 WE arming (mod-257 linear hints + amplification).
//!
//! # Architecture
//!
//! This module provides the lock layer for the arm-before-proof WE scheme:
//! - arming publishes sparse hint blocks over F257 (one hint vector per accepting-set branch)
//!   and two ciphertexts per lock (one per element of the shifted accepting set A').
//! - decapsulation is streaming: absorb proof chunks, then `finish_decrypt_candidates()`.
//!
//! # Security model
//!
//! Per-lock security is **not** standalone 128-bit. In this deterministic mod-257 design, the lock
//! provides only a small per-lock ambiguity (on the order of ~8–9 bits), and security is achieved
//! through amplification (many locks per armer, no per-lock verification oracle).
//!
//! The lock encryption is **unauthenticated** (no per-lock MAC/tag). This prevents a per-lock
//! verification oracle, forcing the adversary to guess all T locks jointly.

use ark_ff::PrimeField;
use rand::RngCore;
use std::collections::BTreeMap;

/// Block packing size for sparse hints and streamed π.
const PACK_D: usize = 64;

/// Prime modulus for the tiny field.
const MOD_257: u16 = 257;

#[inline]
fn f_to_u64<F: PrimeField>(f: &F) -> u64 {
    f.into_bigint().as_ref()[0]
}

#[inline]
fn add_mod257(a: u16, b: u16) -> u16 {
    let s = a + b;
    if s >= MOD_257 { s - MOD_257 } else { s }
}

// NOTE: keep `sub_mod257` only if we reintroduce a subtraction-based packing.

#[inline]
fn mul_mod257(a: u16, b: u16) -> u16 {
    ((a as u32 * b as u32) % (MOD_257 as u32)) as u16
}

#[inline]
fn pow_mod257(mut a: u16, mut e: u16) -> u16 {
    // MOD_257 is prime (257), so inverse is a^(255) for a!=0.
    let mut acc = 1u16;
    while e > 0 {
        if (e & 1) != 0 {
            acc = mul_mod257(acc, a);
        }
        a = mul_mod257(a, a);
        e >>= 1;
    }
    acc
}

#[inline]
fn inv_mod257(a: u16) -> Result<u16, String> {
    if a % MOD_257 == 0 {
        return Err("inv_mod257: inverse of 0".to_string());
    }
    Ok(pow_mod257(a % MOD_257, 255))
}

/// Sparse encoding of up to 64 coefficients in F257.
///
/// Each entry is 2 bytes:
/// - `pos_flags`: low 6 bits = position in 0..63; bit6 = "is 256" flag; bit7 reserved.
/// - `coeff`: `v % 256` (0..255). If bit6 is set, the value is interpreted as 256.
///
/// We omit zero coefficients, so this representation is much smaller than dense `[u8;64]` when
/// within-block sparsity is high.
#[derive(Clone, Debug, Default)]
pub struct PackedF257SparseBlock64 {
    pub entries: Vec<(u8, u8)>,
}

impl PackedF257SparseBlock64 {
    #[inline]
    pub fn from_dense_u16s(vals: &[u16; PACK_D]) -> Self {
        let mut entries: Vec<(u8, u8)> = Vec::new();
        for i in 0..PACK_D {
            let v = vals[i] % MOD_257;
            if v == 0 {
                continue;
            }
            if v == 256 {
                entries.push(((i as u8) | (1u8 << 6), 0u8));
            } else {
                entries.push((i as u8, v as u8));
            }
        }
        Self { entries }
    }

    #[inline]
    pub fn iter(&self) -> impl Iterator<Item = (usize, u16)> + '_ {
        self.entries.iter().map(|&(pos_flags, coeff)| {
            let pos = (pos_flags & 0x3f) as usize;
            let is_256 = ((pos_flags >> 6) & 1) != 0;
            let v = if is_256 { 256u16 } else { coeff as u16 };
            (pos, v)
        })
    }

    #[inline]
    pub fn scale_mod257(&self, s: u16) -> Self {
        let mut out: Vec<(u8, u8)> = Vec::with_capacity(self.entries.len());
        for &(pos_flags, coeff) in &self.entries {
            let pos = pos_flags & 0x3f;
            let is_256 = ((pos_flags >> 6) & 1) != 0;
            let v = if is_256 { 256u16 } else { coeff as u16 };
            let w = mul_mod257(v, s);
            if w == 0 {
                continue;
            }
            if w == 256 {
                out.push((pos | (1u8 << 6), 0u8));
            } else {
                out.push((pos, w as u8));
            }
        }
        Self { entries: out }
    }
}

/// Pack a row of d coefficients (in 0..257) into the block format used by `dot_row_mod257`.
///
/// Direct (non-ring) convention:
/// - out[i] = row[i] for i=0..d-1, with implicit zero-padding.
fn query_row_to_packed_f257(row: &[u16]) -> PackedF257SparseBlock64 {
    let mut tmp = [0u16; PACK_D];
    if !row.is_empty() {
        let lim = PACK_D.min(row.len());
        for i in 0..lim {
            tmp[i] = row[i] % MOD_257;
        }
    }
    PackedF257SparseBlock64::from_dense_u16s(&tmp)
}

/// Direct dot product for packed F257 blocks:
///
/// `acc = Σ_{i=0..d-1} h[i]*row[i]` over F257 (with implicit zero-padding).
#[inline]
fn dot_row_mod257(h: &PackedF257SparseBlock64, row: &[u16]) -> u16 {
    let mut acc = 0u16;
    for (pos, v) in h.iter() {
        if pos < row.len() {
            acc = add_mod257(acc, mul_mod257(v, row[pos]));
        }
    }
    acc
}

// ---------------------------------------------------------------------------
// Amplification parameters
// ---------------------------------------------------------------------------

/// Amplification parameters for the T-of-R threshold lockset.
#[derive(Clone, Debug)]
pub struct AmplificationParams {
    pub r: usize,
    pub t: usize,
    /// Per-lock brute-force candidates (conservative bound).
    /// This is a heuristic estimate for the nonstandard distribution used here.
    /// The actual per-lock hardness is NOT reduced to standard RLWE; this bound should
    /// be set conservatively and the system security validated via `security_bits_*()`.
    pub candidates_per_lock: u64,
}

impl Default for AmplificationParams {
    fn default() -> Self {
        Self {
            r: 64,
            t: 7,
            candidates_per_lock: 1 << 20, // heuristic; nonstandard distribution, not a BKZ claim
        }
    }
}

impl AmplificationParams {
    pub fn security_bits_single_armer(&self) -> f64 {
        let t = self.t as f64;
        let bits_per_lock = (self.candidates_per_lock as f64).log2();
        let binom = ln_binom(self.r, self.t) / std::f64::consts::LN_2;
        t * bits_per_lock - binom
    }

    pub fn security_bits_system(&self, n_armers: usize) -> f64 {
        self.security_bits_single_armer() * (n_armers as f64)
    }

    pub fn security_bits_outer_threshold(&self, n_armers: usize, t_armers: usize) -> f64 {
        let inner = self.security_bits_single_armer();
        let binom_outer = ln_binom(n_armers, t_armers) / std::f64::consts::LN_2;
        (t_armers as f64) * inner - binom_outer
    }

    pub fn print_summary(&self, n_armers: usize) {
        let bits_1 = self.security_bits_single_armer();
        let bits_n = self.security_bits_system(n_armers);
        eprintln!(
            "[amplification] R={}, T={}, cands/lock=2^{:.0}, bits/armer={:.1}, bits/system(N={})={:.1}",
            self.r, self.t, (self.candidates_per_lock as f64).log2(), bits_1, n_armers, bits_n
        );
    }
}

fn ln_binom(n: usize, k: usize) -> f64 {
    if k > n {
        return f64::NEG_INFINITY;
    }
    let k = k.min(n - k);
    let mut acc = 0.0f64;
    for i in 0..k {
        acc += ((n - i) as f64).ln() - ((i + 1) as f64).ln();
    }
    acc
}

// ---------------------------------------------------------------------------
// Lock parameters and artifact types
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct RingLweParams {
    /// Reserved for future non-deterministic / non-mod257 designs.
    ///
    /// In the current deterministic mod-257 lock, these are not used for security.
    pub _reserved0: u32,
    /// Domain separation label.
    pub domain_label: [u8; 32],
}

impl Default for RingLweParams {
    fn default() -> Self {
        Self {
            // Single canonical encoding: sparse-in-block hints.
            _reserved0: 1,
            domain_label: *b"LFP_RINGLWE_V2_00000000000000000",
        }
    }
}

#[derive(Clone, Debug)]
pub struct RingLweLockArtifact<F: PrimeField> {
    pub c_stmt: Vec<F>,
    pub accepting_set: [F; 2],
    pub offset: F,
    pub x_len: usize,
    pub pi_len: usize,
    pub len: usize,
    pub coins: dpp::theorem43::Theorem43Coins<F>,
    pub params: RingLweParams,
    /// Hint material stored sparsely as `(block_idx, packed_coeffs)`, where each block has
    /// `PACK_D=64` coefficients over F257 packed as 1 byte + an `is_256` mask.
    pub hints: BranchHints,
    /// Single unauthenticated ciphertext.
    ///
    /// IMPORTANT: we intentionally publish **only one** ciphertext. Publishing two ciphertexts of
    /// the same payload under keys from a tiny space (mod-257) creates a per-lock equality oracle
    /// (meet-in-the-middle intersection), which collapses amplification.
    pub ct: LockCiphertext,
}

/// Hint material: one hint vector per lock.
#[derive(Clone, Debug)]
pub struct BranchHints {
    pub hint_blocks_sparse: Vec<(usize, PackedF257SparseBlock64)>,
}

/// Per-branch ciphertext: unauthenticated stream cipher (XOR) under a derived key.
///
/// Critical: there is **no per-lock authentication tag**, to avoid a per-lock verification oracle
/// that would collapse threshold amplification.
#[derive(Clone, Debug)]
pub struct LockCiphertext {
    pub nonce: [u8; 12],
    pub ct: Vec<u8>,
}

// ---------------------------------------------------------------------------
// Cryptographic helpers
// ---------------------------------------------------------------------------


fn sha256_32(chunks: &[&[u8]]) -> [u8; 32] {
    use sha2::Digest;
    let mut h = sha2::Sha256::new();
    for c in chunks {
        h.update(c);
    }
    h.finalize().into()
}

/// Derive a 32-byte payload key from two 8-bit seeds and context binding.
fn derive_payload_key_bytes<F: PrimeField>(
    domain_label: &[u8; 32],
    c_stmt: &[F],
    coins: &dpp::theorem43::Theorem43Coins<F>,
    y_mod257: u16,
) -> [u8; 32] {
    let mut coins_bytes = Vec::with_capacity(8 * 4);
    coins_bytes.extend_from_slice(&(coins.idx as u64).to_le_bytes());
    coins_bytes.extend_from_slice(&f_to_u64(&coins.lambda).to_le_bytes());
    coins_bytes.extend_from_slice(&f_to_u64(&coins.rho).to_le_bytes());
    coins_bytes.extend_from_slice(&f_to_u64(&coins.sigma).to_le_bytes());

    let mut stmt_bytes = Vec::with_capacity(c_stmt.len() * 8);
    for f in c_stmt {
        stmt_bytes.extend_from_slice(&f_to_u64(f).to_le_bytes());
    }

    sha256_32(&[
        b"LFP_RINGLWE_PAYLOAD_KEY_V3",
        domain_label,
        stmt_bytes.as_slice(),
        coins_bytes.as_slice(),
        &y_mod257.to_le_bytes(),
    ])
}

/// Unauthenticated XOR stream cipher under a SHA256-derived keystream.
fn xor_stream_encrypt(key: &[u8; 32], nonce: &[u8; 12], data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    let mut ctr = 0u32;
    let mut pos = 0usize;
    while pos < data.len() {
        let block = sha256_32(&[b"LFP_XOR_STREAM_V1", key, nonce, &ctr.to_le_bytes()]);
        let take = (data.len() - pos).min(32);
        for j in 0..take {
            out.push(data[pos + j] ^ block[j]);
        }
        pos += take;
        ctr += 1;
    }
    out
}

#[inline]
fn xor_stream_decrypt(key: &[u8; 32], nonce: &[u8; 12], ct: &[u8]) -> Vec<u8> {
    xor_stream_encrypt(key, nonce, ct)
}

/// Sample a nonzero secret scalar in `{1,2,...,256}` (viewed mod 257).
#[inline]
fn sample_nonzero_f257_scalar(rng: &mut impl RngCore) -> u16 {
    let v = (rng.next_u32() & 0xFF) as u16; // 0..255
    v + 1 // 1..256
}

// ---------------------------------------------------------------------------
// Streaming decapsulation state
// ---------------------------------------------------------------------------

pub struct RingLweDecapStreamState<'a, F: PrimeField> {
    lock: &'a RingLweLockArtifact<F>,
    d: usize,
    /// Accumulated mod-257 dot product `y = ⟨h, π⟩`.
    ///
    /// If the DPP guarantees `⟨q, π⟩ ∈ {a0, a1}`, and `h = q*s`, then `y = s*⟨q,π⟩ = s*a_true`.
    y: u16,
    sparse: &'a [(usize, PackedF257SparseBlock64)],
    sparse_pos: usize,
    block_idx: usize,
    pos_in_block: usize,
    filled: usize,
    coeffs: Vec<u16>,
}

impl<'a, F: PrimeField> RingLweDecapStreamState<'a, F> {
    fn new(lock: &'a RingLweLockArtifact<F>, x: &[F]) -> Result<Self, String> {
        if x.len() != lock.x_len || x.len() + lock.pi_len != lock.len {
            return Err("ringlwe_decap_state: bad x length".to_string());
        }
        Ok(Self {
            lock,
            d: PACK_D,
            y: 0,
            sparse: lock.hints.hint_blocks_sparse.as_slice(),
            sparse_pos: 0,
            block_idx: 0,
            pos_in_block: 0,
            filled: 0,
            coeffs: Vec::with_capacity(PACK_D),
        })
    }

    #[inline]
    fn next_needed_block(&self) -> Option<usize> {
        self.sparse.get(self.sparse_pos).map(|t| t.0)
    }

    #[inline]
    fn process_current_block(&mut self, row: &[u16]) {
        if self.sparse_pos < self.sparse.len() && self.sparse[self.sparse_pos].0 == self.block_idx
        {
            let h = &self.sparse[self.sparse_pos].1;
            let inc = dot_row_mod257(h, row);
            self.y = add_mod257(self.y, inc);
            self.sparse_pos += 1;
        }
    }

    #[inline]
    fn maybe_process_full_block(&mut self) -> Result<(), String> {
        if self.pos_in_block != self.d {
            return Ok(());
        }
        debug_assert!(self.coeffs.is_empty() || self.coeffs.len() == self.d);
        if !self.coeffs.is_empty() {
            // Avoid borrowing `self` mutably while a slice into `self.coeffs` is live.
            let row = core::mem::take(&mut self.coeffs);
            self.process_current_block(row.as_slice());
        }
        self.block_idx += 1;
        self.pos_in_block = 0;
        Ok(())
    }

    pub fn absorb_chunk(&mut self, chunk: &[F]) -> Result<(), String> {
        let mut i = 0usize;
        while i < chunk.len() {
            if self.filled >= self.lock.pi_len {
                return Err("ringlwe_decap_stream: too many π elements".to_string());
            }
            let need = self.next_needed_block() == Some(self.block_idx);
            let rem_chunk = chunk.len() - i;
            let rem_pi = self.lock.pi_len - self.filled;
            let rem_block = self.d - self.pos_in_block;
            let take = rem_chunk.min(rem_pi).min(rem_block);
            debug_assert!(take > 0);

            if need {
                // Collect coefficients for this block only (hinted blocks).
                for v in &chunk[i..i + take] {
                    let vv = (f_to_u64(v) % (MOD_257 as u64)) as u16;
                    self.coeffs.push(vv);
                }
                self.pos_in_block += take;
                self.filled += take;
                i += take;
                self.maybe_process_full_block()?;
            } else {
                // Skip uninterested blocks in bulk: advance counters without embedding/converting.
                self.pos_in_block += take;
                self.filled += take;
                i += take;
                if self.pos_in_block == self.d {
                    // Fast-path block boundary update (no coeffs to process).
                    self.block_idx += 1;
                    self.pos_in_block = 0;
                    self.coeffs.clear();
                }
            }
        }
        Ok(())
    }

    /// Finalize streaming and return per-branch key material seeds.
    fn finish_y_mod257(mut self) -> Result<u16, String> {
        if self.filled != self.lock.pi_len {
            return Err("ringlwe_decap_stream: bad π length".to_string());
        }
        // Flush remaining partial block.
        if self.pos_in_block != 0 {
            debug_assert_eq!(self.pos_in_block, self.filled % self.d);
            if !self.coeffs.is_empty() {
                // Current block was needed, so we collected its (partial) coefficients.
                // Avoid borrowing `self` mutably while a slice into `self.coeffs` is live.
                let row = core::mem::take(&mut self.coeffs);
                self.process_current_block(row.as_slice());
            }
            self.block_idx += 1;
            self.pos_in_block = 0;
            self.coeffs.clear();
        }
        let nblocks = (self.lock.pi_len + self.d - 1) / self.d;
        if self.block_idx != nblocks {
            return Err("ringlwe_decap_stream: internal block count mismatch".to_string());
        }
        if self.sparse_pos != self.sparse.len() {
            return Err("ringlwe_decap_stream: did not consume all sparse blocks".to_string());
        }
        Ok(self.y)
    }

    /// Finish streaming and return two candidate decryptions (unauthenticated).
    ///
    /// We publish a single ciphertext under a key derived from the hidden scalar `s` (mod 257).
    /// The witness stream yields `y = s * a_true` where `a_true ∈ {a0, a1}`. The decapper tries
    /// both candidates `s0 = y * inv(a0)` and `s1 = y * inv(a1)`.
    pub fn finish_decrypt_candidates(self) -> Result<[Vec<u8>; 2], String> {
        let lock = self.lock;
        let y = self.finish_y_mod257()?;
        let a0 = (f_to_u64(&lock.accepting_set[0]) % 257) as u16;
        let a1 = (f_to_u64(&lock.accepting_set[1]) % 257) as u16;
        let inv0 = inv_mod257(a0)?;
        let inv1 = inv_mod257(a1)?;
        let s0 = mul_mod257(y, inv0);
        let s1 = mul_mod257(y, inv1);

        let key0 =
            derive_payload_key_bytes(&lock.params.domain_label, &lock.c_stmt, &lock.coins, s0);
        let key1 =
            derive_payload_key_bytes(&lock.params.domain_label, &lock.c_stmt, &lock.coins, s1);
        let pt0 = xor_stream_decrypt(&key0, &lock.ct.nonce, &lock.ct.ct);
        let pt1 = xor_stream_decrypt(&key1, &lock.ct.nonce, &lock.ct.ct);
        Ok([pt0, pt1])
    }

}

// ---------------------------------------------------------------------------
// Query accumulator
// ---------------------------------------------------------------------------

pub(crate) struct QueryBlockAccumulator<F: PrimeField> {
    pi_len: usize,
    d: usize,
    blocks: BTreeMap<usize, Vec<u16>>,
    _marker: std::marker::PhantomData<F>,
}

impl<F: PrimeField> QueryBlockAccumulator<F> {
    pub(crate) fn new(pi_len: usize) -> Result<Self, String> {
        Ok(Self {
            pi_len,
            d: PACK_D,
            blocks: BTreeMap::new(),
            _marker: std::marker::PhantomData,
        })
    }

    pub(crate) fn add_term(&mut self, coeff: &F, idx: usize) -> Result<(), String> {
        if idx >= self.pi_len {
            return Err("q_pi index out of range".to_string());
        }
        let block = idx / self.d;
        let pos = idx % self.d;
        let row = self
            .blocks
            .entry(block)
            .or_insert_with(|| vec![0u16; self.d]);
        let v = (f_to_u64(coeff) % (MOD_257 as u64)) as u16;
        row[pos] = add_mod257(row[pos], v);
        Ok(())
    }

    pub(crate) fn into_sparse_blocks(&mut self) -> Vec<(usize, PackedF257SparseBlock64)> {
        let blocks = std::mem::take(&mut self.blocks);
        blocks
            .into_iter()
            .map(|(idx, row)| (idx, query_row_to_packed_f257(row.as_slice())))
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Lock artifact API
// ---------------------------------------------------------------------------

impl<F: PrimeField> RingLweLockArtifact<F> {
    pub fn decap_state<'a>(&'a self, x: &[F]) -> Result<RingLweDecapStreamState<'a, F>, String> {
        RingLweDecapStreamState::new(self, x)
    }
}

/// Arm (create) a lock artifact with deterministic mod-257 hints + unauthenticated XOR.
pub fn arm_ringlwe_lock<F: PrimeField>(
    c_stmt: Vec<F>,
    accepting_set_shifted: [F; 2],
    coins: dpp::theorem43::Theorem43Coins<F>,
    offset: F,
    x_len: usize,
    pi_len: usize,
    q_blocks: Vec<(usize, PackedF257SparseBlock64)>,
    params: RingLweParams,
    payload: &[u8],
    rng: &mut impl RngCore,
) -> Result<RingLweLockArtifact<F>, String> {
    if accepting_set_shifted[0].is_zero() || accepting_set_shifted[1].is_zero() {
        return Err(
            "arm_ringlwe_lock: shifted accepting set contains 0; resample rep_id".to_string(),
        );
    }

    // Single secret scalar per lock.
    let s_u16: u16 = sample_nonzero_f257_scalar(rng);

    // Deterministic hint blocks: h = q * s (mod 257).
    let h_blocks: Vec<(usize, PackedF257SparseBlock64)> = q_blocks
        .iter()
        .map(|(block_idx, q)| (*block_idx, q.scale_mod257(s_u16)))
        .collect();
    let hints = BranchHints {
        hint_blocks_sparse: h_blocks,
    };

    // Encrypt payload once under a key derived from `s` (NOT from `s*a`).
    let key = derive_payload_key_bytes(&params.domain_label, &c_stmt, &coins, s_u16);
    let mut nonce = [0u8; 12];
    rng.fill_bytes(&mut nonce);
    let ct = xor_stream_encrypt(&key, &nonce, payload);
    let ct = LockCiphertext { nonce, ct };

    Ok(RingLweLockArtifact {
        c_stmt,
        accepting_set: accepting_set_shifted,
        offset,
        x_len,
        pi_len,
        len: x_len + pi_len,
        coins,
        params,
        hints,
        ct,
    })
}

// Backward-compat type alias.
pub type DppLockCiphertext = LockCiphertext;

// ---------------------------------------------------------------------------
// Tests (attack/sanity harnesses)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use ark_ff::Field;
    use latticefold::transcript::poseidon::F257;
    use rand::{RngCore, SeedableRng};
    use rand_chacha::ChaCha20Rng;

    /// Attack check: the ciphertext bytes should not equal the payload in the clear.
    ///
    /// This is a very weak negative test (it is not an IND-CPA proof); it simply ensures we are
    /// not publishing the payload directly in a trivially decodable representation.
    #[test]
    fn test_ciphertext_is_not_plaintext_default_params() {
        let mut rng = ChaCha20Rng::from_seed([7u8; 32]);
        let mut payload = [0u8; 48];
        rng.fill_bytes(&mut payload);

        let c_stmt: Vec<F257> = Vec::new();
        let accepting_set_shifted = [F257::from(1u64), F257::from(2u64)];
        let offset = F257::ZERO;
        let x_len = 1usize;
        let pi_len = 1usize;
        let q_blocks: Vec<(usize, PackedF257SparseBlock64)> = Vec::new();
        let params = RingLweParams::default();

        // Dummy coins (not used by ciphertext encoding).
        let coins = dpp::theorem43::Theorem43Coins::<F257> {
            idx: 0,
            lambda: F257::ONE,
            rho: F257::ONE,
            sigma: F257::ONE,
        };

        let lock = arm_ringlwe_lock::<F257>(
            c_stmt,
            accepting_set_shifted,
            coins,
            offset,
            x_len,
            pi_len,
            q_blocks,
            params,
            payload.as_slice(),
            &mut rng,
        )
        .expect("arm_ringlwe_lock");

        assert_ne!(
            lock.ct.ct.as_slice(),
            payload.as_slice(),
            "ciphertext should not equal payload"
        );
    }
}
