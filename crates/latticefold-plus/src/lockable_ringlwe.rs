//! Lock layer for Theorem-4.3 WE arming (mod-257 linear hints + amplification).
//!
//! # Architecture
//!
//! This module provides the lock layer for the arm-before-proof WE scheme:
//! - arming publishes sparse hint blocks over F257 (one hint vector per accepting-set branch)
//!   and a **single** unauthenticated ciphertext per lock (XOR-stream under a derived key).
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

/// Compute the canonical ratio class `min(r, r^{-1})` where `r = a1/a0 (mod 257)`.
///
/// Used to enforce that repetitions within a channel have distinct accepting-set ratios, so the
/// intersection of 2-candidate sets collapses deterministically for an honest decapper (given π).
pub(crate) fn ratio_class_mod257_u16<F: PrimeField>(a0: &F, a1: &F) -> Result<u16, String> {
    let a0u = (f_to_u64(a0) % 257) as u16;
    let a1u = (f_to_u64(a1) % 257) as u16;
    if a0u == 0 || a1u == 0 {
        return Err("ratio_class_mod257_u16: zero accepting element".to_string());
    }
    let r = mul_mod257(a1u, inv_mod257(a0u)?);
    let rinv = inv_mod257(r)?;
    Ok(r.min(rinv))
}

/// Public non-bricking policy check (required for deterministic disambiguation at `R=2`).
///
/// Ensures that the shifted accepting-set ratio class is distinct across repetitions within each
/// channel. If violated, the per-channel intersection of 2-candidate sets can remain ambiguous and
/// blow up the downstream enumeration cap.
pub(crate) fn check_ratio_class_distinctness_per_channel<F: PrimeField>(
    lock: &RingLweLockArtifact<F>,
) -> Result<(), String> {
    let p = lock.p_channels as usize;
    let r = lock.r_reps as usize;
    if p == 0 || r == 0 {
        return Err("ringlwe: invalid (P,R)".to_string());
    }
    if lock.sublocks.len() != p.saturating_mul(r) {
        return Err("ringlwe: sublocks length mismatch".to_string());
    }
    let mut seen: Vec<Vec<u16>> = vec![Vec::with_capacity(r); p];
    for sl in &lock.sublocks {
        let ch = sl.channel_id as usize;
        if ch >= p {
            return Err("ringlwe: sublock channel_id out of range".to_string());
        }
        let rc = ratio_class_mod257_u16(&sl.accepting_set[0], &sl.accepting_set[1])?;
        if seen[ch].contains(&rc) {
            return Err("ringlwe: duplicate accepting-set ratio class within a channel".to_string());
        }
        seen[ch].push(rc);
    }
    Ok(())
}

/// Packed encoding of up to 64 coefficients in F257 (per hinted block).
///
/// We support a dual-format representation:
/// - Sparse: omit zeros; store `(pos_flags, coeff)` pairs.
/// - Dense: store 64 bytes + a 64-bit mask for coefficients equal to 256.
///
/// This reduces the "dense block is worst case for sparse encoding" outliers, which in turn
/// reduces the need for hint-budget-driven rejection sampling (and its distributional bias).
#[derive(Clone, Debug)]
pub enum PackedF257Block64 {
    /// Sparse encoding:
    /// - `pos_flags`: low 6 bits = position in 0..63; bit6 = "is 256" flag; bit7 reserved.
    /// - `coeff`: `v % 256` (0..255). If bit6 is set, the value is interpreted as 256.
    Sparse { entries: Vec<(u8, u8)> },
    /// Dense encoding:
    /// - `vals[i]`: `v % 256` (0..255)
    /// - if bit i in `is256_mask` is set, value is interpreted as 256 (regardless of vals[i])
    Dense { vals: [u8; PACK_D], is256_mask: u64 },
}

impl Default for PackedF257Block64 {
    fn default() -> Self {
        Self::Sparse { entries: Vec::new() }
    }
}

impl PackedF257Block64 {
    // Serialization cost model used for in-memory packing decisions.
    // - Sparse on disk: fmt(1) + nnz(1) + 2*nnz
    // - Dense on disk:  fmt(1) + vals(64) + is256_mask(u64)
    const DENSE_ON_DISK_BYTES: usize = 1 + PACK_D + 8;

    #[inline]
    pub fn nnz(&self) -> usize {
        match self {
            Self::Sparse { entries } => entries.len(),
            Self::Dense { vals, is256_mask } => {
                let mut n = is256_mask.count_ones() as usize;
                // Count nonzero bytes where not overridden by 256-mask.
                let mut m = *is256_mask;
                for i in 0..PACK_D {
                    let masked = (m & 1) != 0;
                    if !masked && vals[i] != 0 {
                        n += 1;
                    }
                    m >>= 1;
                }
                n
            }
        }
    }

    #[inline]
    pub fn on_disk_bytes(&self) -> usize {
        match self {
            Self::Sparse { entries } => 1 + 1 + 2 * entries.len(),
            Self::Dense { .. } => Self::DENSE_ON_DISK_BYTES,
        }
    }

    #[inline]
    pub fn from_dense_u16s(vals: &[u16; PACK_D]) -> Self {
        // Decide sparse vs dense by comparing serialized sizes.
        let mut nnz = 0usize;
        for i in 0..PACK_D {
            if (vals[i] % MOD_257) != 0 {
                nnz += 1;
            }
        }
        let sparse_bytes = 1 + 1 + 2 * nnz;
        if sparse_bytes >= Self::DENSE_ON_DISK_BYTES {
            // Dense wins (or ties): keep fixed size.
            let mut out_vals = [0u8; PACK_D];
            let mut is256_mask: u64 = 0;
            for i in 0..PACK_D {
                let v = vals[i] % MOD_257;
                if v == 256 {
                    is256_mask |= 1u64 << i;
                    out_vals[i] = 0;
                } else {
                    out_vals[i] = (v % 256) as u8;
                }
            }
            Self::Dense {
                vals: out_vals,
                is256_mask,
            }
        } else {
            // Sparse wins.
            let mut entries: Vec<(u8, u8)> = Vec::with_capacity(nnz);
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
            Self::Sparse { entries }
        }
    }

    #[inline]
    fn get_at_dense(vals: &[u8; PACK_D], is256_mask: u64, i: usize) -> u16 {
        let bit = (is256_mask >> i) & 1;
        if bit != 0 {
            256u16
        } else {
            vals[i] as u16
        }
    }

    #[inline]
    pub fn scale_mod257(&self, s: u16) -> Self {
        match self {
            Self::Sparse { entries } => {
                let mut out: Vec<(u8, u8)> = Vec::with_capacity(entries.len());
                for &(pos_flags, coeff) in entries {
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
                Self::Sparse { entries: out }
            }
            Self::Dense { vals, is256_mask } => {
                let mut tmp = [0u16; PACK_D];
                for i in 0..PACK_D {
                    let v = Self::get_at_dense(vals, *is256_mask, i);
                    tmp[i] = mul_mod257(v, s);
                }
                Self::from_dense_u16s(&tmp)
            }
        }
    }
}

/// Pack a row of d coefficients (in 0..257) into the block format used by `dot_row_mod257`.
///
/// Direct (non-ring) convention:
/// - out[i] = row[i] for i=0..d-1, with implicit zero-padding.
fn query_row_to_packed_f257(row: &[u16]) -> PackedF257Block64 {
    let mut tmp = [0u16; PACK_D];
    if !row.is_empty() {
        let lim = PACK_D.min(row.len());
        for i in 0..lim {
            tmp[i] = row[i] % MOD_257;
        }
    }
    PackedF257Block64::from_dense_u16s(&tmp)
}

/// Direct dot product for packed F257 blocks:
///
/// `acc = Σ_{i=0..d-1} h[i]*row[i]` over F257 (with implicit zero-padding).
#[inline]
fn dot_row_mod257(h: &PackedF257Block64, row: &[u16]) -> u16 {
    let mut acc = 0u16;
    match h {
        PackedF257Block64::Sparse { entries } => {
            for &(pos_flags, coeff) in entries {
                let pos = (pos_flags & 0x3f) as usize;
                if pos >= row.len() {
                    continue;
                }
                let is_256 = ((pos_flags >> 6) & 1) != 0;
                let v = if is_256 { 256u16 } else { coeff as u16 };
                acc = add_mod257(acc, mul_mod257(v, row[pos]));
            }
        }
        PackedF257Block64::Dense { vals, is256_mask } => {
            let lim = PACK_D.min(row.len());
            for i in 0..lim {
                let v = PackedF257Block64::get_at_dense(vals, *is256_mask, i);
                if v == 0 {
                    continue;
                }
                acc = add_mod257(acc, mul_mod257(v, row[i]));
            }
        }
    }
    acc
}

// ---------------------------------------------------------------------------
// Amplification parameters
// ---------------------------------------------------------------------------

/// Classical brute-force bits under the global-check-only model.
///
/// Security accounting is driven by the number of independent 8-bit channel secrets that
/// must be guessed jointly before one global check is available:
///   classical_bits = 8 * P * L
/// where P = channels per lock, L = required locks.
pub fn classical_bits_global_check_only(p_channels: u16, required_locks: usize) -> f64 {
    8.0 * (p_channels as f64) * (required_locks as f64)
}

/// Grover bits under the same model:
///   grover_bits = classical_bits / 2 = 4 * P * L
pub fn grover_bits_global_check_only(p_channels: u16, required_locks: usize) -> f64 {
    0.5 * classical_bits_global_check_only(p_channels, required_locks)
}

/// Heuristic PQ-128 condition under Grover-only accounting:
///   P * L >= 32  <=>  grover_bits >= 128
pub fn meets_pq128_grover_global_check_only(p_channels: u16, required_locks: usize) -> bool {
    (p_channels as usize) * required_locks >= 32
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
            // Single canonical encoding: dual-format sparse/dense-in-block hints.
            _reserved0: 2,
            domain_label: *b"LFP_RINGLWE_V2_00000000000000000",
        }
    }
}

#[derive(Clone, Debug)]
pub struct RingLweLockArtifact<F: PrimeField> {
    pub c_stmt: Vec<F>,
    pub x_len: usize,
    pub pi_len: usize,
    pub len: usize,
    pub params: RingLweParams,
    /// Number of independent scalar channels \(P\).
    pub p_channels: u16,
    /// Number of repetitions per channel \(R\).
    pub r_reps: u16,
    /// DPP sublocks (one per `(channel,rep)`), each with its own hidden query.
    pub sublocks: Vec<RingLweSubLock<F>>,
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
    pub hint_blocks_sparse: Vec<(usize, PackedF257Block64)>,
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

/// One DPP instance (hidden query) inside a share-lock.
///
/// Each sublock corresponds to a distinct hidden query \(q^{(channel,rep)}\), with its own
/// accepting set and mod-257 hints. All sublocks within a share-lock share a single ciphertext,
/// keyed by the tuple of per-channel secrets.
#[derive(Clone, Debug)]
pub struct RingLweSubLock<F: PrimeField> {
    /// Channel id in `[0..P)`.
    pub channel_id: u16,
    /// Shifted accepting set (mod 257), must be nonzero.
    pub accepting_set: [F; 2],
    /// Public coins for Theorem-4.3 (prover needs these).
    pub coins: dpp::theorem43::Theorem43Coins<F>,
    /// Deterministic hint blocks `h = q * s_channel (mod 257)`.
    pub hints: BranchHints,
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

/// Derive a 32-byte payload key from the tuple of per-channel secrets and public context binding.
fn derive_payload_key_bytes_multi<F: PrimeField>(
    domain_label: &[u8; 32],
    c_stmt: &[F],
    x_len: usize,
    pi_len: usize,
    len: usize,
    p_channels: u16,
    r_reps: u16,
    sublocks: &[RingLweSubLock<F>],
    s_channels_mod257: &[u16],
) -> [u8; 32] {
    use sha2::Digest;
    let mut stmt_bytes = Vec::with_capacity(c_stmt.len() * 8);
    for f in c_stmt {
        stmt_bytes.extend_from_slice(&f_to_u64(f).to_le_bytes());
    }

    // Bind the key to all public per-sublock coins and accepting sets, to avoid cross-lock keystream
    // reuse even if some channels accidentally repeat.
    let mut bind = sha2::Sha256::new();
    bind.update(b"LFP_RINGLWE_LOCK_BIND_V1");
    bind.update(domain_label);
    bind.update(&(c_stmt.len() as u64).to_le_bytes());
    bind.update(stmt_bytes.as_slice());
    bind.update(&(sublocks.len() as u64).to_le_bytes());
    bind.update(&(x_len as u64).to_le_bytes());
    bind.update(&(pi_len as u64).to_le_bytes());
    bind.update(&(len as u64).to_le_bytes());
    bind.update(&p_channels.to_le_bytes());
    bind.update(&r_reps.to_le_bytes());
    for sl in sublocks {
        bind.update(&sl.channel_id.to_le_bytes());
        bind.update(&f_to_u64(&sl.accepting_set[0]).to_le_bytes());
        bind.update(&f_to_u64(&sl.accepting_set[1]).to_le_bytes());
        bind.update(&(sl.coins.idx as u64).to_le_bytes());
        bind.update(&f_to_u64(&sl.coins.lambda).to_le_bytes());
        bind.update(&f_to_u64(&sl.coins.rho).to_le_bytes());
        bind.update(&f_to_u64(&sl.coins.sigma).to_le_bytes());
    }
    let lock_bind: [u8; 32] = bind.finalize().into();

    let mut s_bytes = Vec::with_capacity(2 * s_channels_mod257.len());
    for &s in s_channels_mod257 {
        s_bytes.extend_from_slice(&s.to_le_bytes());
    }

    sha256_32(&[
        b"LFP_RINGLWE_PAYLOAD_KEY_V4",
        &lock_bind,
        s_bytes.as_slice(),
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
pub(crate) fn sample_nonzero_f257_scalar(rng: &mut impl RngCore) -> u16 {
    let v = (rng.next_u32() & 0xFF) as u16; // 0..255
    v + 1 // 1..256
}

// ---------------------------------------------------------------------------
// Streaming decapsulation state
// ---------------------------------------------------------------------------

pub struct RingLweDecapStreamState<'a, F: PrimeField> {
    lock: &'a RingLweLockArtifact<F>,
    sublock: &'a RingLweSubLock<F>,
    d: usize,
    /// Accumulated mod-257 dot product `y = ⟨h, π⟩`.
    ///
    /// If the DPP guarantees `⟨q, π⟩ ∈ {a0, a1}`, and `h = q*s`, then `y = s*⟨q,π⟩ = s*a_true`.
    y: u16,
    sparse: &'a [(usize, PackedF257Block64)],
    sparse_pos: usize,
    block_idx: usize,
    pos_in_block: usize,
    filled: usize,
    coeffs: Vec<u16>,
}

impl<'a, F: PrimeField> RingLweDecapStreamState<'a, F> {
    fn new(
        lock: &'a RingLweLockArtifact<F>,
        sublock: &'a RingLweSubLock<F>,
        x: &[F],
    ) -> Result<Self, String> {
        if x.len() != lock.x_len || x.len() + lock.pi_len != lock.len {
            return Err("ringlwe_decap_state: bad x length".to_string());
        }
        Ok(Self {
            lock,
            sublock,
            d: PACK_D,
            y: 0,
            sparse: sublock.hints.hint_blocks_sparse.as_slice(),
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

    /// Finish streaming and return the channel id and two candidate scalars `s` (mod 257).
    pub fn finish_s_candidates(self) -> Result<(u16, [u16; 2]), String> {
        let sl = self.sublock;
        let y = self.finish_y_mod257()?;
        let a0 = (f_to_u64(&sl.accepting_set[0]) % 257) as u16;
        let a1 = (f_to_u64(&sl.accepting_set[1]) % 257) as u16;
        let inv0 = inv_mod257(a0)?;
        let inv1 = inv_mod257(a1)?;
        let s0 = mul_mod257(y, inv0);
        let s1 = mul_mod257(y, inv1);
        Ok((sl.channel_id, [s0, s1]))
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

    pub(crate) fn into_sparse_blocks(&mut self) -> Vec<(usize, PackedF257Block64)> {
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
    pub(crate) fn payload_key_bytes(&self, s_channels_mod257: &[u16]) -> Result<[u8; 32], String> {
        if s_channels_mod257.len() != self.p_channels as usize {
            return Err("ringlwe: payload_key_bytes: channel secret length mismatch".to_string());
        }
        Ok(derive_payload_key_bytes_multi(
            &self.params.domain_label,
            &self.c_stmt,
            self.x_len,
            self.pi_len,
            self.len,
            self.p_channels,
            self.r_reps,
            self.sublocks.as_slice(),
            s_channels_mod257,
        ))
    }

    pub(crate) fn decrypt_payload(&self, s_channels_mod257: &[u16]) -> Result<Vec<u8>, String> {
        let key = self.payload_key_bytes(s_channels_mod257)?;
        Ok(xor_stream_decrypt(&key, &self.ct.nonce, &self.ct.ct))
    }

    pub fn decap_states<'a>(
        &'a self,
        x: &[F],
    ) -> Result<Vec<RingLweDecapStreamState<'a, F>>, String> {
        let mut out = Vec::with_capacity(self.sublocks.len());
        for sl in &self.sublocks {
            out.push(RingLweDecapStreamState::new(self, sl, x)?);
        }
        Ok(out)
    }
}

// ---------------------------------------------------------------------------
// Decap helpers (candidate intersection + payload candidate enumeration)
// ---------------------------------------------------------------------------

fn intersect_2cands_across_reps(reps: &[[u16; 2]]) -> Result<Vec<u16>, String> {
    if reps.is_empty() {
        return Err("ringlwe: empty reps list".to_string());
    }
    let mut cur: Vec<u16> = vec![reps[0][0], reps[0][1]];
    cur.sort_unstable();
    cur.dedup();
    for rep in reps.iter().skip(1) {
        let mut nxt: Vec<u16> = vec![rep[0], rep[1]];
        nxt.sort_unstable();
        nxt.dedup();
        let mut inter: Vec<u16> = Vec::new();
        for &a in &cur {
            if nxt.contains(&a) {
                inter.push(a);
            }
        }
        cur = inter;
    }
    Ok(cur)
}

/// Given per-sublock `s` candidates (mod 257), intersect across repetitions within each channel,
/// then decrypt the lock payload under the resulting (unique) per-channel secret tuple.
///
/// Canonical policy: **no branching**. If intersections do not collapse to singletons,
/// this returns an error.
///
/// `sublock_s_candidates` must be in the same order as `lock.sublocks` (one entry per sublock).
pub(crate) fn decrypt_payload_from_sublock_s_candidates<F: PrimeField>(
    lock: &RingLweLockArtifact<F>,
    sublock_s_candidates: &[(u16, [u16; 2])],
) -> Result<Vec<u8>, String> {
    let p = lock.p_channels as usize;
    let r = lock.r_reps as usize;
    if p == 0 || r == 0 {
        return Err("ringlwe: invalid (P,R)".to_string());
    }
    if lock.sublocks.len() != p.saturating_mul(r) {
        return Err("ringlwe: sublocks length mismatch".to_string());
    }
    if sublock_s_candidates.len() != lock.sublocks.len() {
        return Err("ringlwe: sublock_s_candidates length mismatch".to_string());
    }

    // Group 2-candidate sets by channel, preserving the canonical sublock order.
    let mut per_channel_reps: Vec<Vec<[u16; 2]>> = vec![Vec::with_capacity(r); p];
    for (i, (ch, cands)) in sublock_s_candidates.iter().enumerate() {
        let sl = lock
            .sublocks
            .get(i)
            .ok_or_else(|| "ringlwe: internal sublock index mismatch".to_string())?;
        if *ch != sl.channel_id {
            return Err("ringlwe: sublock channel_id mismatch".to_string());
        }
        let ch_usize = *ch as usize;
        if ch_usize >= p {
            return Err("ringlwe: channel_id out of range".to_string());
        }
        per_channel_reps[ch_usize].push(*cands);
    }
    for ch in 0..p {
        if per_channel_reps[ch].len() != r {
            return Err("ringlwe: missing repetitions for some channel".to_string());
        }
    }

    // Intersect across reps within a channel.
    let mut channel_sets: Vec<Vec<u16>> = Vec::with_capacity(p);
    for ch in 0..p {
        let mut cur = intersect_2cands_across_reps(per_channel_reps[ch].as_slice())?;
        cur.sort_unstable();
        cur.dedup();
        if cur.is_empty() {
            return Err("ringlwe: empty intersection for some channel".to_string());
        }
        if cur.len() > 2 {
            return Err("ringlwe: internal error (intersection >2)".to_string());
        }
        channel_sets.push(cur);
    }

    // Enumerate cartesian product over per-channel candidates.
    let mut total: u64 = 1;
    for set in &channel_sets {
        total = total.saturating_mul(set.len() as u64);
    }
    // Canonical policy: no branching. If the per-channel intersections do not collapse to
    // singletons, the lock is not deterministically decapsulatable without an external global
    // disambiguation predicate (which we intentionally avoid at the lock layer).
    //
    if total != 1 {
        let lens: Vec<usize> = channel_sets.iter().map(|s| s.len()).collect();
        return Err(format!(
            "ringlwe: ambiguous per-channel intersections (P={p} R={r} channel_set_lens={lens:?} total_tuples={total}); lock must satisfy ratio-class distinctness policy"
        ));
    }

    let s_channels: Vec<u16> = channel_sets
        .iter()
        .map(|set| {
            debug_assert_eq!(set.len(), 1);
            set[0]
        })
        .collect();
    lock.decrypt_payload(s_channels.as_slice())
}

/// Arm (create) a lock artifact with deterministic mod-257 hints + unauthenticated XOR.
pub fn arm_ringlwe_lock<F: PrimeField>(
    c_stmt: Vec<F>,
    x_len: usize,
    pi_len: usize,
    params: RingLweParams,
    p_channels: u16,
    r_reps: u16,
    sublocks: Vec<RingLweSubLock<F>>,
    payload: &[u8],
    s_channels_mod257: &[u16],
    rng: &mut impl RngCore,
) -> Result<RingLweLockArtifact<F>, String> {
    if p_channels == 0 || r_reps == 0 {
        return Err("arm_ringlwe_lock: invalid (P,R)".to_string());
    }
    if sublocks.len() != (p_channels as usize) * (r_reps as usize) {
        return Err("arm_ringlwe_lock: sublocks length mismatch".to_string());
    }
    if s_channels_mod257.len() != p_channels as usize {
        return Err("arm_ringlwe_lock: channel secrets length mismatch".to_string());
    }
    for &s in s_channels_mod257 {
        if s == 0 || s > 256 {
            return Err("arm_ringlwe_lock: bad channel secret scalar".to_string());
        }
    }
    for sl in &sublocks {
        if sl.accepting_set[0].is_zero() || sl.accepting_set[1].is_zero() {
            return Err("arm_ringlwe_lock: shifted accepting set contains 0".to_string());
        }
        if sl.channel_id >= p_channels {
            return Err("arm_ringlwe_lock: sublock channel_id out of range".to_string());
        }
    }

    // Encrypt payload once under a key derived from the tuple of per-channel secrets.
    let key = derive_payload_key_bytes_multi(
        &params.domain_label,
        &c_stmt,
        x_len,
        pi_len,
        x_len + pi_len,
        p_channels,
        r_reps,
        sublocks.as_slice(),
        s_channels_mod257,
    );
    let mut nonce = [0u8; 12];
    rng.fill_bytes(&mut nonce);
    let ct = xor_stream_encrypt(&key, &nonce, payload);
    let ct = LockCiphertext { nonce, ct };

    Ok(RingLweLockArtifact {
        c_stmt,
        x_len,
        pi_len,
        len: x_len + pi_len,
        params,
        p_channels,
        r_reps,
        sublocks,
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
        let x_len = 1usize;
        let pi_len = 1usize;
        let params = RingLweParams::default();

        // Dummy coins (not used by ciphertext encoding).
        let coins = dpp::theorem43::Theorem43Coins::<F257> {
            idx: 0,
            lambda: F257::ONE,
            rho: F257::ONE,
            sigma: F257::ONE,
        };

        let s = sample_nonzero_f257_scalar(&mut rng);
        let sub = RingLweSubLock::<F257> {
            channel_id: 0,
            accepting_set: [F257::from(1u64), F257::from(2u64)],
            coins,
            hints: BranchHints {
                hint_blocks_sparse: Vec::new(),
            },
        };
        let lock = arm_ringlwe_lock::<F257>(
            c_stmt,
            x_len,
            pi_len,
            params,
            1,
            1,
            vec![sub],
            payload.as_slice(),
            &[s],
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
