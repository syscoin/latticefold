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

/// Block packing size for sparse hints and streamed π.
const PACK_D: usize = 64;

/// Prime modulus for the tiny field.
const MOD_257: u16 = 257;

#[inline]
fn f_to_u64<F: PrimeField>(f: &F) -> u64 {
    f.into_bigint().as_ref()[0]
}

// NOTE: keep `sub_mod257` only if we reintroduce a subtraction-based packing.

#[inline]
pub(crate) fn field_mod257_u16<F: PrimeField>(f: &F) -> u16 {
    (f_to_u64(f) % 257) as u16
}

#[inline]
pub(crate) fn add_mod257_u16(a: u16, b: u16) -> u16 {
    let s = a + b;
    if s >= MOD_257 { s - MOD_257 } else { s }
}

#[inline]
pub(crate) fn sub_mod257_u16(a: u16, b: u16) -> u16 {
    // Returns (a - b) mod 257 with inputs assumed reduced.
    debug_assert!(a < MOD_257 && b < MOD_257);
    if a >= b { a - b } else { a + MOD_257 - b }
}

#[inline]
fn mul_mod257(a: u16, b: u16) -> u16 {
    // Fast reduction mod 257 using 256 ≡ -1 (mod 257).
    //
    // Precondition in the WE pipeline: values are always reduced to 0..=256.
    debug_assert!(a < MOD_257 && b < MOD_257);
    let prod = (a as u32) * (b as u32); // <= 65536
    let low = (prod & 0xFF) as i32; // 0..255
    let high = (prod >> 8) as i32; // 0..256
    let mut r = low - high; // -256..255
    if r < 0 {
        r += 257;
    }
    debug_assert!((0..257).contains(&r));
    r as u16
}

#[inline]
pub(crate) fn mul_mod257_u16(a: u16, b: u16) -> u16 {
    mul_mod257(a, b)
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

/// Compute a mod-257 dot product between a packed block and a 64-wide `u16` row.
#[inline]
pub(crate) fn dot_packed_block_mod257_u16(blk: &PackedF257Block64, row64: &[u16]) -> u16 {
    let mut acc = 0u16;
    match blk {
        PackedF257Block64::Sparse { entries } => {
            for &(pos_flags, coeff) in entries {
                let pos = (pos_flags & 0x3f) as usize;
                if pos >= row64.len() {
                    continue;
                }
                let is_256 = ((pos_flags >> 6) & 1) != 0;
                let v = if is_256 { 256u16 } else { coeff as u16 };
                if v == 0 {
                    continue;
                }
                acc = add_mod257_u16(acc, mul_mod257(v, row64[pos]));
            }
        }
        PackedF257Block64::Dense { vals, is256_mask } => {
            let lim = 64usize.min(row64.len());
            for i in 0..lim {
                let is256 = ((*is256_mask >> i) & 1) != 0;
                let v = if is256 { 256u16 } else { vals[i] as u16 };
                if v == 0 {
                    continue;
                }
                acc = add_mod257_u16(acc, mul_mod257(v, row64[i]));
            }
        }
    }
    acc
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
    /// Number of sublocks per channel.
    ///
    /// In the “hits-per-block” design, this is typically `blocks * hits_per_block` and can be
    /// much larger than `u16`, so we store it as `u32`.
    pub sublocks_per_channel: u32,
    /// DPP sublocks (one per `(channel,hit)`), each with its own hidden query.
    pub sublocks: Vec<RingLweSubLock<F>>,
    /// Single unauthenticated ciphertext.
    ///
    /// IMPORTANT: we intentionally publish **only one** ciphertext. Publishing two ciphertexts of
    /// the same payload under keys from a tiny space (mod-257) creates a per-lock equality oracle
    /// (meet-in-the-middle intersection), which collapses amplification.
    pub ct: LockCiphertext,
}

/// Compressed hint material for one DPP sublock.
///
/// This avoids materializing/storing the full masked query vector `h = q * s` over the enormous
/// Theorem-4.3 proof `π`. Instead, we store only the secret-scaled combination coefficients needed
/// to reconstruct the masked answer:
///
/// - `abg_scales`: `(s*coeff_alpha, s*coeff_beta, s*coeff_gamma)` in mod-257 digits
/// - `offset_scale`: `s * δ(x_arm)` in mod-257 digits (statement-bound offset)
/// - `tail_scales`: 4 packed blocks holding `s*(coeff_mu, coeff_nu, c_3..c_{p-1})`
#[derive(Clone, Debug)]
pub struct BranchHintsCompressed {
    pub abg_scales: [u16; 3],
    pub offset_scale: u16,
    pub tail_scales: [PackedF257Block64; 4],
    /// Optional “global poison” material: secret-scaled per-block weights.
    ///
    /// When present, decap computes an additional masked term:
    ///   y_err = Σ_b err_scales[b] * err_b   (mod 257)
    /// and adds it to the anchor masked scalar before candidate extraction.
    ///
    /// Encoding is canonical packed-in-64s using `PackedF257Block64`.
    ///
    pub poison_blocks: u32,
    pub poison_err_scales: Vec<PackedF257Block64>,
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
    /// Anchor block id in `[0..blocks)` used to derive Theorem-4.3 public coins for this sublock.
    pub anchor_block_id: u32,
    /// Rep-id salt (used to derive Theorem-4.3 public coins and hidden Sq coefficients).
    pub rep_id: u64,
    /// Deterministic compressed hint material.
    pub hints: BranchHintsCompressed,
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
    sublocks_per_channel: u32,
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
    bind.update(&sublocks_per_channel.to_le_bytes());
    for sl in sublocks {
        bind.update(&sl.channel_id.to_le_bytes());
        bind.update(&sl.anchor_block_id.to_le_bytes());
        bind.update(&sl.rep_id.to_le_bytes());
        bind.update(&f_to_u64(&sl.accepting_set[0]).to_le_bytes());
        bind.update(&f_to_u64(&sl.accepting_set[1]).to_le_bytes());
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

/// Compute two candidates for `s` given a mod-257 dot product `y = s * a_true` and a 2-element
/// accepting set `{a0, a1}` (mod 257).
#[inline]
pub(crate) fn s_candidates_from_y_and_accepting_set_mod257<F: PrimeField>(
    accepting_set: &[F; 2],
    y: u16,
) -> Result<[u16; 2], String> {
    let a0 = (f_to_u64(&accepting_set[0]) % 257) as u16;
    let a1 = (f_to_u64(&accepting_set[1]) % 257) as u16;
    let inv0 = inv_mod257(a0)?;
    let inv1 = inv_mod257(a1)?;
    let s0 = mul_mod257(y, inv0);
    let s1 = mul_mod257(y, inv1);
    Ok([s0, s1])
}

/// Compute `y mod 257` from compressed hints and streamed `(alpha,beta,gamma)` plus `tail_dot`.
///
/// This is the core decap scalar relation used for candidate extraction:
/// `y = y_pi + offset_scale  (mod 257)` for the chosen offset flavor.
#[inline]
fn masked_y_mod257_from_compressed_hint_and_tail_with_offset<F: PrimeField>(
    hc: &BranchHintsCompressed,
    abg: &dpp::theorem43::Theorem43AbgTail<F>,
    tail_dot_mod257: u16,
    offset_scale: u16,
) -> Result<u16, String> {
    let alpha = field_mod257_u16(&abg.alpha);
    let beta = field_mod257_u16(&abg.beta);
    let gamma = field_mod257_u16(&abg.gamma);
    let mut y = 0u16;
    y = add_mod257_u16(y, mul_mod257(hc.abg_scales[0], alpha));
    y = add_mod257_u16(y, mul_mod257(hc.abg_scales[1], beta));
    y = add_mod257_u16(y, mul_mod257(hc.abg_scales[2], gamma));
    y = add_mod257_u16(y, tail_dot_mod257 % 257);
    y = add_mod257_u16(y, offset_scale);
    Ok(y)
}

pub(crate) fn masked_y_mod257_main_from_compressed_hint_and_tail<F: PrimeField>(
    hc: &BranchHintsCompressed,
    abg: &dpp::theorem43::Theorem43AbgTail<F>,
    tail_dot_mod257: u16,
) -> Result<u16, String> {
    masked_y_mod257_from_compressed_hint_and_tail_with_offset(
        hc,
        abg,
        tail_dot_mod257,
        hc.offset_scale,
    )
}

#[inline]
pub(crate) fn poison_y_err_mod257_from_abg_full<F: PrimeField>(
    hc: &BranchHintsCompressed,
    abg_full_all_blocks: &[dpp::theorem43::Theorem43AbgFull<F>],
) -> Result<u16, String> {
    let blocks = hc.poison_blocks as usize;
    if blocks == 0 {
        return Ok(0u16);
    }
    if abg_full_all_blocks.len() != blocks {
        return Err("ringlwe: poison abg_full length mismatch".to_string());
    }
    let nblk = (blocks + 63) / 64;
    if hc.poison_err_scales.len() != nblk {
        return Err("ringlwe: poison_err_scales length mismatch".to_string());
    }

    let mut acc = 0u16;
    let mut err64 = [0u16; 64];
    for bi in 0..nblk {
        let start = bi * 64;
        for j in 0..64 {
            let b = start + j;
            if b >= blocks {
                err64[j] = 0u16;
                continue;
            }
            let a = field_mod257_u16(&abg_full_all_blocks[b].alpha);
            let bb = field_mod257_u16(&abg_full_all_blocks[b].beta);
            let g = field_mod257_u16(&abg_full_all_blocks[b].gamma);
            let ab = mul_mod257_u16(a, bb);
            err64[j] = sub_mod257_u16(g, ab);
        }
        let blk = &hc.poison_err_scales[bi];
        let add = dot_packed_block_mod257_u16(blk, &err64);
        acc = add_mod257_u16(acc, add);
    }
    Ok(acc)
}

/// Convenience: compute the 2 candidates for `s` for one sublock under the compressed-hint scheme.
pub(crate) fn sublock_s_candidates_from_abg_tail<F: PrimeField>(
    sl: &RingLweSubLock<F>,
    abg: &dpp::theorem43::Theorem43AbgTail<F>,
    tail_dot_mod257: u16,
) -> Result<[u16; 2], String> {
    let y = masked_y_mod257_main_from_compressed_hint_and_tail(&sl.hints, abg, tail_dot_mod257)?;
    s_candidates_from_y_and_accepting_set_mod257(&sl.accepting_set, y)
}

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
            self.sublocks_per_channel,
            self.sublocks.as_slice(),
            s_channels_mod257,
        ))
    }

    pub(crate) fn decrypt_payload(&self, s_channels_mod257: &[u16]) -> Result<Vec<u8>, String> {
        let key = self.payload_key_bytes(s_channels_mod257)?;
        Ok(xor_stream_decrypt(&key, &self.ct.nonce, &self.ct.ct))
    }
}

// ---------------------------------------------------------------------------
// Decap helpers (candidate intersection + payload candidate enumeration)
// ---------------------------------------------------------------------------

#[inline]
fn bitset257_from_2cands(cands: [u16; 2]) -> [u64; 5] {
    let mut bs = [0u64; 5]; // 5*64 = 320 >= 257
    for &v in &cands {
        let idx = (v % 257) as usize;
        let w = idx >> 6;
        let b = idx & 63;
        bs[w] |= 1u64 << b;
    }
    bs
}

#[inline]
fn bitset257_and_inplace(a: &mut [u64; 5], b: &[u64; 5]) {
    for i in 0..5 {
        a[i] &= b[i];
    }
}

#[inline]
fn bitset257_is_empty(a: &[u64; 5]) -> bool {
    a.iter().all(|&w| w == 0)
}

#[inline]
fn bitset257_popcount(a: &[u64; 5]) -> u32 {
    a.iter().map(|w| w.count_ones()).sum()
}

#[inline]
fn bitset257_singleton_value(a: &[u64; 5]) -> Option<u16> {
    if bitset257_popcount(a) != 1 {
        return None;
    }
    for (wi, &w) in a.iter().enumerate() {
        if w != 0 {
            let tz = w.trailing_zeros() as usize;
            let idx = (wi << 6) | tz;
            if idx <= 256 {
                return Some(idx as u16);
            } else {
                return None;
            }
        }
    }
    None
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
    let r = lock.sublocks_per_channel as usize;
    if p == 0 || r == 0 {
        return Err("ringlwe: invalid (P,sublocks_per_channel)".to_string());
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

    // Intersect across reps within a channel using a 257-bit bitset.
    let mut s_channels: Vec<u16> = Vec::with_capacity(p);
    for ch in 0..p {
        let reps = per_channel_reps[ch].as_slice();
        debug_assert_eq!(reps.len(), r);
        let mut alive = bitset257_from_2cands(reps[0]);
        for rep in reps.iter().skip(1) {
            let m = bitset257_from_2cands(*rep);
            bitset257_and_inplace(&mut alive, &m);
            if bitset257_is_empty(&alive) {
                break;
            }
        }
        if bitset257_is_empty(&alive) {
            return Err(format!(
                "ringlwe: empty intersection for some channel (ch={} reps={:?})",
                ch, per_channel_reps[ch]
            ));
        }
        let s = bitset257_singleton_value(&alive).ok_or_else(|| {
            // Canonical policy: no branching. If intersections do not collapse to singletons,
            // the lock is not deterministically decapsulatable without an external predicate.
            //
            // Report popcount to help parameter tuning.
            let pc = bitset257_popcount(&alive);
            format!(
                "ringlwe: ambiguous per-channel intersections (P={p} sublocks_per_channel={r} ch={ch} popcount={pc}); increase hits per block / total sublocks per channel"
            )
        })?;
        s_channels.push(s);
    }

    // Canonical policy: no branching (already enforced via singleton extraction above).
    if s_channels.len() != p {
        return Err(format!(
            "ringlwe: internal error (s_channels len mismatch: got={} expected={})",
            s_channels.len(),
            p
        ));
    }
    lock.decrypt_payload(s_channels.as_slice())
}

/// Arm (create) a lock artifact with deterministic mod-257 hints + unauthenticated XOR.
pub fn arm_ringlwe_lock<F: PrimeField>(
    c_stmt: Vec<F>,
    x_len: usize,
    pi_len: usize,
    params: RingLweParams,
    p_channels: u16,
    sublocks_per_channel: u32,
    sublocks: Vec<RingLweSubLock<F>>,
    payload: &[u8],
    s_channels_mod257: &[u16],
    rng: &mut impl RngCore,
) -> Result<RingLweLockArtifact<F>, String> {
    if p_channels == 0 || sublocks_per_channel == 0 {
        return Err("arm_ringlwe_lock: invalid (P,sublocks_per_channel)".to_string());
    }
    if sublocks.len() != (p_channels as usize) * (sublocks_per_channel as usize) {
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
            return Err("arm_ringlwe_lock: accepting set contains 0".to_string());
        }
        if sl.channel_id >= p_channels {
            return Err("arm_ringlwe_lock: sublock channel_id out of range".to_string());
        }
    }
    // Public non-bricking policy: within each channel, repetitions must have distinct accepting-set
    // ratio classes so intersections collapse deterministically at small `R`.
    //
    // This is enforced at arming time to prevent publishing ambiguous packages.
    let r = sublocks_per_channel as usize;
    for ch in 0..(p_channels as usize) {
        let mut seen: std::collections::BTreeSet<u16> = std::collections::BTreeSet::new();
        for rep_j in 0..r {
            let si = ch.saturating_mul(r).saturating_add(rep_j);
            let sl = sublocks
                .get(si)
                .ok_or_else(|| "arm_ringlwe_lock: sublock index OOB".to_string())?;
            let rc = ratio_class_mod257_u16(&sl.accepting_set[0], &sl.accepting_set[1])?;
            if !seen.insert(rc) {
                return Err("arm_ringlwe_lock: duplicate accepting-set ratio class within channel".to_string());
            }
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
        sublocks_per_channel,
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
        sublocks_per_channel,
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

        let s = sample_nonzero_f257_scalar(&mut rng);
        let zero64 = [0u16; 64];
        let sub = RingLweSubLock::<F257> {
            channel_id: 0,
            accepting_set: [F257::from(7u64), F257::from(8u64)],
            anchor_block_id: 0,
            rep_id: 0,
            hints: BranchHintsCompressed {
                abg_scales: [0u16; 3],
                offset_scale: 0u16,
                tail_scales: core::array::from_fn(|_| PackedF257Block64::from_dense_u16s(&zero64)),
                poison_blocks: 1,
                poison_err_scales: vec![PackedF257Block64::from_dense_u16s(&zero64)],
            },
        };
        let lock = arm_ringlwe_lock::<F257>(
            c_stmt,
            x_len,
            pi_len,
            params,
            1,
            1u32,
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
