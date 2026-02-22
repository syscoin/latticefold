//! Lock layer for Theorem-4.3 WE arming (sparse mod-257 hint blocks + streaming decap).
//!
//! This is the canonical (non-compressed) hint format:
//! - the package publishes sparse packed blocks of the masked query `h = s * q (mod 257)`
//! - decap streams the proof `π`, accumulates `⟨h, π⟩ (mod 257)` for the tailless anchor
//! - decap returns candidate plaintext bytes for the accepting-set-derived secret candidates
//!
//! Important: this module intentionally does **not** include compressed/tail-coefficient encodings.
//! Those formats reintroduce publicly-evaluable tiny-image probe attacks.

use ark_ff::PrimeField;
use rand::RngCore;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::sync::{Mutex, OnceLock};

/// Block packing size for sparse hints and streamed π.
const PACK_D: usize = 64;

/// Prime modulus for the tiny field.
const MOD_257: u16 = 257;

fn debug_s_map() -> &'static Mutex<HashMap<u64, u16>> {
    static MAP: OnceLock<Mutex<HashMap<u64, u16>>> = OnceLock::new();
    MAP.get_or_init(|| Mutex::new(HashMap::new()))
}

pub(crate) fn debug_record_s_for_rep(rep_id: u64, s: u16) {
    let on = std::env::var("LFP_DEBUG_IDENTITY").ok().as_deref() == Some("1")
        || std::env::var("LFP_DEBUG_ARM").ok().as_deref() == Some("1");
    if on {
        if let Ok(mut g) = debug_s_map().lock() {
            g.insert(rep_id, s);
        }
    }
}

pub(crate) fn debug_get_s_for_rep(rep_id: u64) -> Option<u16> {
    let on = std::env::var("LFP_DEBUG_IDENTITY").ok().as_deref() == Some("1")
        || std::env::var("LFP_DEBUG_ARM").ok().as_deref() == Some("1");
    if !on {
        return None;
    }
    debug_s_map().lock().ok().and_then(|g| g.get(&rep_id).copied())
}

#[inline]
fn f_to_u64<F: PrimeField>(f: &F) -> u64 {
    f.into_bigint().as_ref()[0]
}

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
    debug_assert!(a < MOD_257 && b < MOD_257);
    if a >= b { a - b } else { a + MOD_257 - b }
}

#[inline]
fn mul_mod257(a: u16, b: u16) -> u16 {
    // Fast reduction mod 257 using 256 ≡ -1 (mod 257).
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
fn inv_mod257_u16(a: u16) -> Option<u16> {
    if a == 0 || a >= MOD_257 {
        return None;
    }
    // Extended Euclid over integers.
    let mut t: i32 = 0;
    let mut new_t: i32 = 1;
    let mut r: i32 = MOD_257 as i32;
    let mut new_r: i32 = a as i32;
    while new_r != 0 {
        let q = r / new_r;
        (t, new_t) = (new_t, t - q * new_t);
        (r, new_r) = (new_r, r - q * new_r);
    }
    if r != 1 {
        return None;
    }
    if t < 0 {
        t += MOD_257 as i32;
    }
    Some((t as u16) % MOD_257)
}

#[inline]
fn pow256_mod257_u16(mut a: u16) -> u16 {
    // 256 = 2^8, so compute by 8 squarings.
    a %= MOD_257;
    for _ in 0..8 {
        a = mul_mod257_u16(a, a);
    }
    a
}

#[inline]
fn is_zero_mod257_u16(z: u16) -> u16 {
    // is_zero(z) = 1 - z^256, so:
    // - z==0  => 1
    // - z!=0  => 0
    sub_mod257_u16(1u16, pow256_mod257_u16(z))
}

#[inline]
fn gate_from_mixes_all_mod257_u16(zs: &[u16]) -> u16 {
    let mut g: u16 = 1u16;
    for &z in zs {
        g = mul_mod257_u16(g, is_zero_mod257_u16(z));
        if g == 0 {
            break;
        }
    }
    g
}

/// Backward-compatible helper: apply only the first `k` mixes.
pub(crate) fn gate_from_mixes_mod257_u16(zs: &[u16], k: usize) -> u16 {
    let kk = k.min(zs.len());
    gate_from_mixes_all_mod257_u16(&zs[..kk])
}

/// Canonical repetition selector for per-lock share candidates.
///
/// Input is a list of `(share_index, candidate_list)` records, typically one per lock/rep.
/// For each `share_index`, this picks the candidate that appears in the most records.
/// Ties are broken lexicographically for deterministic behavior.
pub fn select_shares_by_majority(
    share_candidates: &[(u32, Vec<[u8; 32]>)],
) -> Vec<(u32, [u8; 32])> {
    // Score by lock-level presence, not multiplicity within one lock.
    let mut by_idx: BTreeMap<u32, BTreeMap<[u8; 32], usize>> = BTreeMap::new();
    for (idx, cands) in share_candidates {
        let mut uniq: BTreeSet<[u8; 32]> = BTreeSet::new();
        for c in cands {
            uniq.insert(*c);
        }
        let scores = by_idx.entry(*idx).or_default();
        for c in uniq {
            *scores.entry(c).or_insert(0) += 1;
        }
    }

    let mut out: Vec<(u32, [u8; 32])> = Vec::with_capacity(by_idx.len());
    for (idx, scores) in by_idx {
        let mut best_c = [0u8; 32];
        let mut best_score: isize = -1;
        for (cand, score) in scores {
            let s = score as isize;
            if s > best_score || (s == best_score && cand < best_c) {
                best_score = s;
                best_c = cand;
            }
        }
        out.push((idx, best_c));
    }
    out
}

// ---------------------------------------------------------------------------
// Packed F257 blocks (sparse/dense-in-block)
// ---------------------------------------------------------------------------

/// Packed encoding of up to 64 coefficients in F257 (per hinted block).
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

impl PackedF257Block64 {
    pub fn scale_mod257(&self, s: u16) -> PackedF257Block64 {
        debug_assert!(s < MOD_257);
        match self {
            PackedF257Block64::Sparse { entries } => {
                let mut out: Vec<(u8, u8)> = Vec::with_capacity(entries.len());
                for &(pos_flags, coeff) in entries {
                    let pos = pos_flags & 0x3f;
                    let is_256 = ((pos_flags >> 6) & 1) != 0;
                    let v = if is_256 { 256u16 } else { coeff as u16 };
                    let sv = mul_mod257_u16(s, v);
                    if sv == 0 {
                        continue;
                    }
                    let mut pf = pos;
                    let mut c = (sv & 0xFF) as u8;
                    if sv == 256 {
                        pf |= 1u8 << 6;
                        c = 0u8;
                    }
                    out.push((pf, c));
                }
                PackedF257Block64::Sparse { entries: out }
            }
            PackedF257Block64::Dense { vals, is256_mask } => {
                let mut out_vals = [0u8; PACK_D];
                let mut out_mask: u64 = 0;
                for i in 0..PACK_D {
                    let is_256 = ((is256_mask >> i) & 1) != 0;
                    let v = if is_256 { 256u16 } else { vals[i] as u16 };
                    let sv = mul_mod257_u16(s, v);
                    out_vals[i] = (sv & 0xFF) as u8;
                    if sv == 256 {
                        out_mask |= 1u64 << i;
                    }
                }
                PackedF257Block64::Dense { vals: out_vals, is256_mask: out_mask }
            }
        }
    }
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
                acc = add_mod257_u16(acc, mul_mod257_u16(v, row64[pos]));
            }
        }
        PackedF257Block64::Dense { vals, is256_mask } => {
            let n = row64.len().min(PACK_D);
            for i in 0..n {
                let is_256 = ((is256_mask >> i) & 1) != 0;
                let v = if is_256 { 256u16 } else { vals[i] as u16 };
                if v == 0 {
                    continue;
                }
                acc = add_mod257_u16(acc, mul_mod257_u16(v, row64[i]));
            }
        }
    }
    acc
}

#[inline]
fn query_row_to_packed_f257(row: &[u16]) -> PackedF257Block64 {
    debug_assert!(row.len() == PACK_D);
    // Heuristic: use Sparse unless it is too dense (avoids worst-case overhead).
    let nnz = row.iter().filter(|&&x| x != 0).count();
    if nnz <= 16 {
        let mut entries: Vec<(u8, u8)> = Vec::with_capacity(nnz);
        for (i, &v) in row.iter().enumerate() {
            if v == 0 {
                continue;
            }
            let mut pos_flags = (i as u8) & 0x3f;
            let mut coeff = (v & 0xFF) as u8;
            if v == 256 {
                pos_flags |= 1u8 << 6;
                coeff = 0u8;
            }
            entries.push((pos_flags, coeff));
        }
        return PackedF257Block64::Sparse { entries };
    }

    let mut vals = [0u8; PACK_D];
    let mut mask: u64 = 0;
    for i in 0..PACK_D {
        let v = row[i];
        vals[i] = (v & 0xFF) as u8;
        if v == 256 {
            mask |= 1u64 << i;
        }
    }
    PackedF257Block64::Dense { vals, is256_mask: mask }
}

#[inline]
fn coeff0_mul_row_mod257(h: &PackedF257Block64, row: &[u16]) -> u16 {
    dot_packed_block_mod257_u16(h, row)
}

// ---------------------------------------------------------------------------
// Lock parameters and artifact types
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct RingLweParams {
    /// Reserved for future non-deterministic / non-mod257 designs.
    pub _reserved0: u32,
    /// Domain separation label.
    pub domain_label: [u8; 32],
}

impl Default for RingLweParams {
    fn default() -> Self {
        Self { _reserved0: 0, domain_label: *b"LFP_RINGLWE_V2_00000000000000000" }
    }
}

#[derive(Clone, Debug)]
pub struct RingLweLockArtifact<F: PrimeField> {
    pub c_stmt: Vec<F>,
    pub accepting_set: [F; 2],
    pub offset: F,
    /// Public coin-derivation inputs (stored for binding and downstream residual gating).
    pub anchor_block_id: u32,
    pub rep_id: u64,
    /// Number of FLPCP blocks for residual gating (length of `err[b]` vector).
    pub poison_blocks: u32,
    pub x_len: usize,
    pub pi_len: usize,
    pub len: usize,
    pub coins: dpp::theorem43::Theorem43Coins<F>,
    pub params: RingLweParams,
    /// Anchor hints stored sparsely as `(block_idx, packed_coeffs)`.
    pub anchor_hints: BranchHints,
    /// Residual-gate hint material (K mixes over the block residual vector).
    ///
    /// When `k=0`, the gate is disabled and treated as `g=1`.
    pub err_gate_hints: ErrGateHints,
    /// Wrapped key ciphertexts (canonical path uses exactly one entry).
    pub cts: Vec<LockCiphertext>,
    /// Encrypted membership indicator for the hidden UV subset `U ⊆ F257^*`.
    ///
    /// Plaintext is a 256-bit indicator vector over `{1..=256}` (packed into 32 bytes).
    /// Security-critical: this plaintext space is **all** 256-bit strings (no fixed weight),
    /// so brute-forcing `s` does not admit a package-only validity test.
    pub ct_ubits: [u8; 32],
    /// Payload ciphertext encrypted under a random payload key.
    pub payload_ct: LockCiphertext,
}

/// Sparse anchor hint material (one hint vector for the lock).
#[derive(Clone, Debug)]
pub struct BranchHints {
    pub hint_blocks_sparse: Vec<(usize, PackedF257Block64)>,
}

/// Hint material for the residual gate `g(err)`.
///
/// Each mix is a packed vector over `poison_blocks` residual digits (one per FLPCP block):
///   z_i = ⟨h_i, err⟩ (mod 257)
/// and:
///   g = Π_i (1 - z_i^256) ∈ {0,1}.
#[derive(Clone, Debug)]
pub struct ErrGateHints {
    /// Number of independent mixes.
    pub k: u16,
    /// Packed blocks per mix: ceil(poison_blocks / 64).
    pub blocks_per_mix: u32,
    /// Flattened packed blocks, length = k * blocks_per_mix.
    pub mixes: Vec<PackedF257Block64>,
}

/// Unauthenticated stream ciphertext (XOR under a derived key).
#[derive(Clone, Debug)]
pub struct LockCiphertext {
    pub nonce: [u8; 12],
    pub ct: Vec<u8>,
}

// ---------------------------------------------------------------------------
// Cryptographic helpers (scaffold; NOT a production KDF/stream cipher)
// ---------------------------------------------------------------------------

fn sha256_32(chunks: &[&[u8]]) -> [u8; 32] {
    use sha2::Digest;
    let mut h = sha2::Sha256::new();
    for c in chunks {
        h.update(c);
    }
    h.finalize().into()
}

fn derive_wrap_key_bytes<F: PrimeField>(
    domain_label: &[u8; 32],
    c_stmt: &[F],
    coins: &dpp::theorem43::Theorem43Coins<F>,
    s_mod257: u16,
) -> [u8; 32] {
    let mut coins_bytes = Vec::with_capacity(8 * 5);
    coins_bytes.extend_from_slice(&(coins.idx as u64).to_le_bytes());
    coins_bytes.extend_from_slice(&f_to_u64(&coins.lambda).to_le_bytes());
    coins_bytes.extend_from_slice(&f_to_u64(&coins.rho).to_le_bytes());
    coins_bytes.extend_from_slice(&f_to_u64(&coins.sigma).to_le_bytes());
    coins_bytes.extend_from_slice(&f_to_u64(&coins.c_hit).to_le_bytes());

    let mut stmt_bytes = Vec::with_capacity(c_stmt.len() * 8);
    for f in c_stmt {
        stmt_bytes.extend_from_slice(&f_to_u64(f).to_le_bytes());
    }

    sha256_32(&[
        b"LFP_RINGLWE_WRAP_KEY_V1",
        domain_label,
        stmt_bytes.as_slice(),
        coins_bytes.as_slice(),
        &s_mod257.to_le_bytes(),
    ])
}

fn derive_ubits_pad_bytes<F: PrimeField>(
    domain_label: &[u8; 32],
    c_stmt: &[F],
    coins: &dpp::theorem43::Theorem43Coins<F>,
    rep_id: u64,
    s_mod257: u16,
) -> [u8; 32] {
    let mut coins_bytes = Vec::with_capacity(8 * 5);
    coins_bytes.extend_from_slice(&(coins.idx as u64).to_le_bytes());
    coins_bytes.extend_from_slice(&f_to_u64(&coins.lambda).to_le_bytes());
    coins_bytes.extend_from_slice(&f_to_u64(&coins.rho).to_le_bytes());
    coins_bytes.extend_from_slice(&f_to_u64(&coins.sigma).to_le_bytes());
    coins_bytes.extend_from_slice(&f_to_u64(&coins.c_hit).to_le_bytes());
    let mut stmt_bytes = Vec::with_capacity(c_stmt.len() * 8);
    for f in c_stmt {
        stmt_bytes.extend_from_slice(&f_to_u64(f).to_le_bytes());
    }
    sha256_32(&[
        b"LFP_RINGLWE_UBITS_PAD_V1",
        domain_label,
        stmt_bytes.as_slice(),
        coins_bytes.as_slice(),
        &rep_id.to_le_bytes(),
        &s_mod257.to_le_bytes(),
    ])
}

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

#[inline]
pub(crate) fn sample_nonzero_f257_scalar(rng: &mut impl RngCore) -> u16 {
    let v = (rng.next_u32() & 0xFF) as u16; // 0..255
    v + 1 // 1..256
}

/// Dot a packed residual-gate hint vector against the full residual vector `errs` (mod 257 digits).
pub fn dot_err_hint_blocks_mod257_u16(hint_blocks: &[PackedF257Block64], errs: &[u16]) -> u16 {
    let mut acc: u16 = 0u16;
    for (bi, blk) in hint_blocks.iter().enumerate() {
        let start = bi * PACK_D;
        if start >= errs.len() {
            break;
        }
        let end = (start + PACK_D).min(errs.len());
        acc = add_mod257_u16(acc, dot_packed_block_mod257_u16(blk, &errs[start..end]));
    }
    acc
}

/// Evaluate the multiplicative kill-switch gate `g(err)` from per-mix hint blocks.
pub fn eval_err_gate_mod257_u16(hints: &ErrGateHints, errs: &[u16]) -> Result<u16, String> {
    let k = hints.k as usize;
    if k == 0 {
        return Ok(1u16);
    }
    let blocks_per_mix = hints.blocks_per_mix as usize;
    if blocks_per_mix == 0 {
        return Err("eval_err_gate_mod257_u16: blocks_per_mix=0".to_string());
    }
    let expected = k
        .checked_mul(blocks_per_mix)
        .ok_or_else(|| "eval_err_gate_mod257_u16: k*blocks_per_mix overflow".to_string())?;
    if hints.mixes.len() != expected {
        return Err("eval_err_gate_mod257_u16: mixes length mismatch".to_string());
    }
    let mut zs: Vec<u16> = Vec::with_capacity(k);
    for i in 0..k {
        let off = i * blocks_per_mix;
        let z = dot_err_hint_blocks_mod257_u16(&hints.mixes[off..off + blocks_per_mix], errs);
        zs.push(z);
    }
    Ok(gate_from_mixes_all_mod257_u16(zs.as_slice()))
}

/// Intersect per-rep candidate `s` sets across repetitions of the same logical lock.
///
/// - `share_indices[i]` is the logical-lock index for rep `i` (e.g. OneProof share index).
/// - `per_rep_hits[i]` is the candidate list for rep `i` (values in `1..=256`).
///
/// Returns `per_rep_intersected[i]`, the candidate list after intersecting across all reps
/// belonging to `share_indices[i]`.
///
/// Completeness / no local oracle: if a share's intersection is empty, we fall back to the full
/// domain `1..=256` for all reps of that share.
pub fn intersect_s_candidates_across_reps_by_share_index(
    share_indices: &[u32],
    per_rep_hits: &[Vec<u16>],
) -> Result<Vec<Vec<u16>>, String> {
    if share_indices.len() != per_rep_hits.len() {
        return Err("intersect_s_candidates_across_reps_by_share_index: length mismatch".to_string());
    }
    if share_indices.is_empty() {
        return Ok(Vec::new());
    }

    let mut by_share: BTreeMap<u32, Vec<usize>> = BTreeMap::new();
    for (i, &idx) in share_indices.iter().enumerate() {
        by_share.entry(idx).or_default().push(i);
    }

    let mut out: Vec<Vec<u16>> = vec![Vec::new(); share_indices.len()];
    for (_share_idx, reps) in by_share {
        let mut alive = vec![false; 257];
        let mut first = true;
        for &ri in reps.iter() {
            let hits = per_rep_hits.get(ri).map(|v| v.as_slice()).unwrap_or(&[]);
            if first {
                for &s in hits {
                    let si = s as usize;
                    if si >= 1 && si <= 256 {
                        alive[si] = true;
                    }
                }
                first = false;
            } else {
                let mut next = vec![false; 257];
                for &s in hits {
                    let si = s as usize;
                    if si >= 1 && si <= 256 && alive[si] {
                        next[si] = true;
                    }
                }
                alive = next;
            }
        }
        let mut s_intersect: Vec<u16> = Vec::new();
        for s in 1u16..=256u16 {
            if alive[s as usize] {
                s_intersect.push(s);
            }
        }
        if s_intersect.is_empty() {
            s_intersect = (1u16..=256u16).collect();
        }
        for &ri in reps.iter() {
            out[ri] = s_intersect.clone();
        }
    }

    Ok(out)
}

// ---------------------------------------------------------------------------
// Streaming decapsulation state
// ---------------------------------------------------------------------------

pub struct RingLweDecapStreamState<'a, F: PrimeField> {
    lock: &'a RingLweLockArtifact<F>,
    d: usize,
    /// Anchor stream accumulator.
    anchor: BranchAccum<'a>,
    block_idx: usize,
    pos_in_block: usize,
    filled: usize,
    coeffs: Vec<u16>,
}

struct BranchAccum<'a> {
    sparse: &'a [(usize, PackedF257Block64)],
    sparse_pos: usize,
    y: u16,
}

impl<'a, F: PrimeField> RingLweDecapStreamState<'a, F> {
    fn new(lock: &'a RingLweLockArtifact<F>, x: &[F]) -> Result<Self, String> {
        if x.len() != lock.x_len || x.len() + lock.pi_len != lock.len {
            return Err("ringlwe_decap_state: bad x length".to_string());
        }
        Ok(Self {
            lock,
            d: PACK_D,
            anchor: BranchAccum {
                sparse: lock.anchor_hints.hint_blocks_sparse.as_slice(),
                sparse_pos: 0,
                y: 0,
            },
            block_idx: 0,
            pos_in_block: 0,
            filled: 0,
            coeffs: Vec::with_capacity(PACK_D),
        })
    }

    #[inline]
    fn next_needed_block(&self) -> Option<usize> {
        self.anchor.sparse.get(self.anchor.sparse_pos).map(|t| t.0)
    }

    #[inline]
    fn process_current_block(&mut self, row: &[u16]) {
        if self.anchor.sparse_pos < self.anchor.sparse.len()
            && self.anchor.sparse[self.anchor.sparse_pos].0 == self.block_idx
        {
            let h = &self.anchor.sparse[self.anchor.sparse_pos].1;
            let inc = coeff0_mul_row_mod257(h, row);
            self.anchor.y = add_mod257_u16(self.anchor.y, inc);
            self.anchor.sparse_pos += 1;
        }
    }

    #[inline]
    fn maybe_process_full_block(&mut self) -> Result<(), String> {
        if self.pos_in_block != self.d {
            return Ok(());
        }
        debug_assert!(self.coeffs.is_empty() || self.coeffs.len() == self.d);
        if !self.coeffs.is_empty() {
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
                for v in &chunk[i..i + take] {
                    let vv = (f_to_u64(v) % (MOD_257 as u64)) as u16;
                    self.coeffs.push(vv);
                }
                self.pos_in_block += take;
                self.filled += take;
                i += take;
                self.maybe_process_full_block()?;
            } else {
                self.pos_in_block += take;
                self.filled += take;
                i += take;
                if self.pos_in_block == self.d {
                    self.block_idx += 1;
                    self.pos_in_block = 0;
                    self.coeffs.clear();
                }
            }
        }
        Ok(())
    }

    fn finish_key_seed_mod257(mut self) -> Result<u16, String> {
        if self.filled != self.lock.pi_len {
            return Err("ringlwe_decap_stream: bad π length".to_string());
        }
        if self.pos_in_block != 0 {
            debug_assert_eq!(self.pos_in_block, self.filled % self.d);
            if !self.coeffs.is_empty() {
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
        if self.anchor.sparse_pos != self.anchor.sparse.len() {
            return Err("ringlwe_decap_stream: did not consume all sparse blocks".to_string());
        }
        Ok(self.anchor.y)
    }

    pub fn finish_decrypt_candidates_with_gate(
        self,
        gate_mod257: u16,
        y_anchor_override: Option<u16>,
        u_override: Option<u16>,
    ) -> Result<Vec<Vec<u8>>, String> {
        let lock = self.lock;
        if lock.cts.len() != 1 {
            return Err("ringlwe_decap_stream: wrapped key ciphertext count must be 1".to_string());
        }
        let s_sets =
            self.finish_s_candidate_sets_with_gate(gate_mod257, y_anchor_override, u_override, None, None)?;
        let mut seen = std::collections::BTreeSet::<[u8; 32]>::new();
        let mut outs: Vec<Vec<u8>> = Vec::new();
        for s_hits in s_sets {
            for s in s_hits {
                let p = lock.decrypt_payload_with_s_candidate(s);
                if p.len() != 32 {
                    outs.push(p);
                    continue;
                }
                let mut a = [0u8; 32];
                a.copy_from_slice(&p[..32]);
                if seen.insert(a) {
                    outs.push(a.to_vec());
                }
            }
        }
        Ok(outs)
    }

    pub(crate) fn finish_s_candidate_sets_with_gate(
        self,
        gate_mod257: u16,
        y_anchor_override: Option<u16>,
        u_override: Option<u16>,
        gamma_override: Option<u16>,
        alpha_beta_override: Option<(u16, u16)>,
    ) -> Result<Vec<Vec<u16>>, String> {
        let lock = self.lock;
        let y_anchor_stream = self.finish_key_seed_mod257()?;
        let y_anchor = y_anchor_override.unwrap_or(y_anchor_stream) % MOD_257;
        let g = gate_mod257 % MOD_257;

        // We only narrow when the residual gate opens.
        // When g=0, return full-domain to avoid a local reject oracle.
        if g == 0 {
            return Ok(vec![(1u16..=256u16).collect(), (1u16..=256u16).collect()]);
        }

        let a0_u16 = field_mod257_u16(&lock.accepting_set[0]) % MOD_257;
        let a1_u16 = field_mod257_u16(&lock.accepting_set[1]) % MOD_257;

        let true_s_dbg = if std::env::var("LFP_DEBUG_IDENTITY").ok().as_deref() == Some("1") {
            debug_get_s_for_rep(lock.rep_id)
        } else {
            None
        };
        // Debug is best-effort and only prints for the true `s` when available.
        let mut dbg_s_true: Option<u16> = None;
        let mut dbg_eq_line: Option<String> = None;

        // Fixed-u squaring-gadget completion (tailless path).
        //
        // We rely on the proof-derived `u` and `gamma`, but we intentionally do NOT require μ/ν
        // to be queried/published (avoids package-only correctness predicates).
        //
        // Completion identity (cross-only):
        //   c_req == b(u) - c1*u - c2*(2ρσγ)
        // where:
        // - b(u) = 1_{u^{-1} ∈ U} is read from decrypted Ubits(s)
        // - c1,c2 are derived from Ubits(s)
        // - c_req = a - y_div with a ∈ accepting_set_shifted and y_div = y_anchor / s
        // Tailless fixed-u completion needs proof-derived (α,β,γ) to compute:
        //   u = ρ·α + σ·β
        //   lin = c1·u + c2·(2ρσ)·γ
        // and a single membership bit b(u)=1_{u^{-1}∈U}.
        let (alpha_u16, beta_u16, gamma_u16): (u16, u16, u16) =
            match (alpha_beta_override, gamma_override) {
                (Some((a, b)), Some(g)) => (a % MOD_257, b % MOD_257, g % MOD_257),
                _ => {
                    // If caller doesn't provide proof-derived ABG, preserve completeness with full-domain.
                    return Ok(vec![(1u16..=256u16).collect(), (1u16..=256u16).collect()]);
                }
            };
        let rho_u16 = field_mod257_u16(&lock.coins.rho) % MOD_257;
        let sigma_u16 = field_mod257_u16(&lock.coins.sigma) % MOD_257;
        // Evaluation point u := ρ·α + σ·β (mod 257).
        let u_fix = add_mod257_u16(mul_mod257_u16(rho_u16, alpha_u16), mul_mod257_u16(sigma_u16, beta_u16));
        // Keep u_override only for debug printing; the completion uses u_fix.
        let _ = u_override;

        // Fixed-shape scan over s∈F257* (1..=256).
        //
        // For each `s` guess we:
        // - decrypt Ubits(s) and derive (c1,c2) and membership bits
        // - compute y_div = y_anchor / s (mod 257)
        // - for each accepting-set element a ∈ {a0,a1}, compute c_req = a - y_div
        // - complete at the fixed proof-derived evaluation point (u,v) when available
        //   via the identity:
        //      b(u) == c2*v + c1*u + c_req
        //   where b(u) = 1_{u^{-1} ∈ U} is read from Ubits(s).
        let mut hits: Vec<u16> = Vec::new();
        for s_probe in 1u16..=256u16 {
            let pad = derive_ubits_pad_bytes(&lock.params.domain_label, &lock.c_stmt, &lock.coins, lock.rep_id, s_probe);
            let mut ubits = [0u8; 32];
            for i in 0..32 {
                ubits[i] = lock.ct_ubits[i] ^ pad[i];
            }

            // Compute (c1, c2) from Ubits.
            let mut sum1: u16 = 0u16;
            let mut sum2: u16 = 0u16;
            for x in 1u16..=256u16 {
                let idx = (x as usize) - 1;
                let bit = (ubits[idx / 8] >> (idx % 8)) & 1;
                if bit == 0 {
                    continue;
                }
                sum1 = add_mod257_u16(sum1, x);
                let x2 = mul_mod257_u16(x % MOD_257, x % MOD_257);
                sum2 = add_mod257_u16(sum2, x2);
            }
            let c1 = sub_mod257_u16(0u16, sum1);
            let c2 = sub_mod257_u16(0u16, sum2);

            let inv_s = inv_mod257_u16(s_probe).unwrap_or(0u16);
            if inv_s == 0 {
                continue;
            }
            let y_div = mul_mod257_u16(y_anchor, inv_s);

            // Collect a one-line equation check for the true `s` (if known) to help pin down the
            // correct completion identity without changing decapsulation logic.
            if let Some(ts) = true_s_dbg {
                if s_probe == ts {
                    let lambda = field_mod257_u16(&lock.coins.lambda) % MOD_257;
                    let rho = field_mod257_u16(&lock.coins.rho) % MOD_257;
                    let sigma = field_mod257_u16(&lock.coins.sigma) % MOD_257;
                    let u_dbg = u_override.map(|v| v % MOD_257);
                    let gamma_dbg = gamma_override.map(|v| v % MOD_257);

                    let c_req0 = sub_mod257_u16(a0_u16, y_div);
                    let c_req1 = sub_mod257_u16(a1_u16, y_div);

                    // b(u) for the provided u (if any), using the membership bit at u^{-1}.
                    let mut b_u: Option<u16> = None;
                    let mut b_u_dir: Option<u16> = None;
                    let mut expr_lin: Option<u16> = None;
                    let mut expr_sq: Option<u16> = None;
                    let mut expr_sq_dir: Option<u16> = None;
                    let mut expr_sq_u_minus_lambda: Option<u16> = None;
                    let mut expr_sq_u_minus_lambda_dir: Option<u16> = None;
                    let mut expr_cross: Option<u16> = None;
                    let mut expr_fixed_uv: Option<u16> = None;

                    if let Some(u) = u_dbg {
                        // Direct membership bit at `u` (no inversion), for convention debugging.
                        if u != 0 {
                            let idx_dir = (u as usize).saturating_sub(1);
                            let bdir = ((ubits[idx_dir / 8] >> (idx_dir % 8)) & 1) as u16;
                            b_u_dir = Some(bdir);
                        }
                        let inv_u = inv_mod257_u16(u).unwrap_or(0u16);
                        if inv_u != 0 {
                            let idx = (inv_u as usize).saturating_sub(1);
                            let b = ((ubits[idx / 8] >> (idx % 8)) & 1) as u16;
                            b_u = Some(b);
                            let lin = sub_mod257_u16(b, mul_mod257_u16(c1, u));
                            expr_lin = Some(lin);
                            let u2 = mul_mod257_u16(u, u);
                            expr_sq = Some(sub_mod257_u16(lin, mul_mod257_u16(c2, u2)));

                            if let Some(bdir) = b_u_dir {
                                let lin_dir = sub_mod257_u16(bdir, mul_mod257_u16(c1, u));
                                expr_sq_dir = Some(sub_mod257_u16(lin_dir, mul_mod257_u16(c2, u2)));
                            }

                            let u0 = sub_mod257_u16(u, lambda);
                            let u0_2 = mul_mod257_u16(u0, u0);
                            let lin0 = sub_mod257_u16(b, mul_mod257_u16(c1, u0));
                            expr_sq_u_minus_lambda = Some(sub_mod257_u16(lin0, mul_mod257_u16(c2, u0_2)));

                            if let Some(bdir) = b_u_dir {
                                let lin0_dir = sub_mod257_u16(bdir, mul_mod257_u16(c1, u0));
                                expr_sq_u_minus_lambda_dir =
                                    Some(sub_mod257_u16(lin0_dir, mul_mod257_u16(c2, u0_2)));
                            }

                            if let Some(gamma) = gamma_dbg {
                                let two_rho_sigma = mul_mod257_u16(2u16, mul_mod257_u16(rho, sigma));
                                let cross = mul_mod257_u16(two_rho_sigma, gamma);
                                expr_cross = Some(sub_mod257_u16(lin, mul_mod257_u16(c2, cross)));
                            }
                        }
                    }

                    // Fixed-(u,v) completion at the proof-derived evaluation point (if available):
                    //   c_req ?= b(u) - c1*u - c2*v
                    if let Some(u_dbg) = u_override.map(|v| v % MOD_257) {
                        let inv_u = inv_mod257_u16(u_dbg).unwrap_or(0u16);
                        if inv_u != 0 {
                            let idx = (inv_u as usize).saturating_sub(1);
                            let b = ((ubits[idx / 8] >> (idx % 8)) & 1) as u16;
                            let two_rho_sigma = mul_mod257_u16(2u16, mul_mod257_u16(rho, sigma));
                            let cross = mul_mod257_u16(two_rho_sigma, gamma_u16);
                            let lin = sub_mod257_u16(b, mul_mod257_u16(c1, u_dbg));
                            expr_fixed_uv = Some(sub_mod257_u16(lin, mul_mod257_u16(c2, cross)));
                        }
                    }

                    // Does there exist a t such that the Sq-tail identity matches the required
                    // completion `c_req`?
                    //
                    // tail(t) := b(t) - c1*t - c2*t^2, where b(t) is read from Ubits under either:
                    // - b(inv): b(t)=1_{t^{-1}∈U}
                    // - b(dir): b(t)=1_{t∈U}
                    //
                    // We don't assume what `t` is derived from; we just check existence and whether
                    // the provided u_dbg (and u_dbg-lambda) are among the solutions.
                    let tail_solutions_info = {
                        let mut inv0 = 0usize;
                        let mut inv1 = 0usize;
                        let mut dir0 = 0usize;
                        let mut dir1 = 0usize;

                        let mut inv0_has_u = false;
                        let mut inv1_has_u = false;
                        let mut dir0_has_u = false;
                        let mut dir1_has_u = false;

                        let mut inv0_has_u0 = false;
                        let mut inv1_has_u0 = false;
                        let mut dir0_has_u0 = false;
                        let mut dir1_has_u0 = false;

                        let u0_dbg = u_dbg.map(|u| sub_mod257_u16(u, lambda));

                        for t in 1u16..=256u16 {
                            let t2 = mul_mod257_u16(t, t);
                            let inv_t = inv_mod257_u16(t).unwrap_or(0u16);
                            let b_inv = if inv_t == 0 {
                                0u16
                            } else {
                                let idx = (inv_t as usize).saturating_sub(1);
                                ((ubits[idx / 8] >> (idx % 8)) & 1) as u16
                            };
                            let idx_dir = (t as usize).saturating_sub(1);
                            let b_dir = ((ubits[idx_dir / 8] >> (idx_dir % 8)) & 1) as u16;

                            let lin_inv = sub_mod257_u16(b_inv, mul_mod257_u16(c1, t));
                            let tail_inv = sub_mod257_u16(lin_inv, mul_mod257_u16(c2, t2));

                            let lin_dir = sub_mod257_u16(b_dir, mul_mod257_u16(c1, t));
                            let tail_dir = sub_mod257_u16(lin_dir, mul_mod257_u16(c2, t2));

                            if tail_inv == c_req0 {
                                inv0 += 1;
                                if Some(t) == u_dbg {
                                    inv0_has_u = true;
                                }
                                if Some(t) == u0_dbg {
                                    inv0_has_u0 = true;
                                }
                            }
                            if tail_inv == c_req1 {
                                inv1 += 1;
                                if Some(t) == u_dbg {
                                    inv1_has_u = true;
                                }
                                if Some(t) == u0_dbg {
                                    inv1_has_u0 = true;
                                }
                            }
                            if tail_dir == c_req0 {
                                dir0 += 1;
                                if Some(t) == u_dbg {
                                    dir0_has_u = true;
                                }
                                if Some(t) == u0_dbg {
                                    dir0_has_u0 = true;
                                }
                            }
                            if tail_dir == c_req1 {
                                dir1 += 1;
                                if Some(t) == u_dbg {
                                    dir1_has_u = true;
                                }
                                if Some(t) == u0_dbg {
                                    dir1_has_u0 = true;
                                }
                            }
                        }

                        format!(
                            "tail_solutions: inv(c_req0)={} inv(c_req1)={} dir(c_req0)={} dir(c_req1)={} (has_u: inv0={} inv1={} dir0={} dir1={} ; has_u0: inv0={} inv1={} dir0={} dir1={})",
                            inv0,
                            inv1,
                            dir0,
                            dir1,
                            inv0_has_u,
                            inv1_has_u,
                            dir0_has_u,
                            dir1_has_u,
                            inv0_has_u0,
                            inv1_has_u0,
                            dir0_has_u0,
                            dir1_has_u0,
                        )
                    };

                    // Does there exist a u such that the "cross-only" completion matches?
                    //
                    // cross_term := 2*rho*sigma*gamma (mod p) from the streamed ABG state.
                    // completion(u) := b(u) - c1*u - c2*cross_term, under b(inv) or b(dir).
                    let cross_solutions_info = if let Some(gamma) = gamma_dbg {
                        let two_rho_sigma = mul_mod257_u16(2u16, mul_mod257_u16(rho, sigma));
                        let cross_term = mul_mod257_u16(two_rho_sigma, gamma);
                        let c2_cross = mul_mod257_u16(c2, cross_term);

                        let mut inv0 = 0usize;
                        let mut inv1 = 0usize;
                        let mut dir0 = 0usize;
                        let mut dir1 = 0usize;

                        let mut inv0_has_u = false;
                        let mut inv1_has_u = false;
                        let mut dir0_has_u = false;
                        let mut dir1_has_u = false;

                        let u0_dbg = u_dbg.map(|u| sub_mod257_u16(u, lambda));
                        let mut inv0_has_u0 = false;
                        let mut inv1_has_u0 = false;
                        let mut dir0_has_u0 = false;
                        let mut dir1_has_u0 = false;

                        for u in 1u16..=256u16 {
                            let inv_u = inv_mod257_u16(u).unwrap_or(0u16);
                            let b_inv = if inv_u == 0 {
                                0u16
                            } else {
                                let idx = (inv_u as usize).saturating_sub(1);
                                ((ubits[idx / 8] >> (idx % 8)) & 1) as u16
                            };
                            let idx_dir = (u as usize).saturating_sub(1);
                            let b_dir = ((ubits[idx_dir / 8] >> (idx_dir % 8)) & 1) as u16;

                            let lin_inv = sub_mod257_u16(b_inv, mul_mod257_u16(c1, u));
                            let comp_inv = sub_mod257_u16(lin_inv, c2_cross);
                            let lin_dir = sub_mod257_u16(b_dir, mul_mod257_u16(c1, u));
                            let comp_dir = sub_mod257_u16(lin_dir, c2_cross);

                            if comp_inv == c_req0 {
                                inv0 += 1;
                                if Some(u) == u_dbg {
                                    inv0_has_u = true;
                                }
                                if Some(u) == u0_dbg {
                                    inv0_has_u0 = true;
                                }
                            }
                            if comp_inv == c_req1 {
                                inv1 += 1;
                                if Some(u) == u_dbg {
                                    inv1_has_u = true;
                                }
                                if Some(u) == u0_dbg {
                                    inv1_has_u0 = true;
                                }
                            }
                            if comp_dir == c_req0 {
                                dir0 += 1;
                                if Some(u) == u_dbg {
                                    dir0_has_u = true;
                                }
                                if Some(u) == u0_dbg {
                                    dir0_has_u0 = true;
                                }
                            }
                            if comp_dir == c_req1 {
                                dir1 += 1;
                                if Some(u) == u_dbg {
                                    dir1_has_u = true;
                                }
                                if Some(u) == u0_dbg {
                                    dir1_has_u0 = true;
                                }
                            }
                        }

                        format!(
                            "cross_solutions: inv(c_req0)={} inv(c_req1)={} dir(c_req0)={} dir(c_req1)={} (has_u: inv0={} inv1={} dir0={} dir1={} ; has_u0: inv0={} inv1={} dir0={} dir1={})",
                            inv0,
                            inv1,
                            dir0,
                            dir1,
                            inv0_has_u,
                            inv1_has_u,
                            dir0_has_u,
                            dir1_has_u,
                            inv0_has_u0,
                            inv1_has_u0,
                            dir0_has_u0,
                            dir1_has_u0,
                        )
                    } else {
                        "cross_solutions: gamma=None".to_string()
                    };

                    // Compute the "missing μ/ν" candidate term under the current completion formula.
                    let miss_info = if let Some(u_dbg) = u_dbg {
                        let rho2 = mul_mod257_u16(rho, rho);
                        let sigma2 = mul_mod257_u16(sigma, sigma);
                        let alpha2 = mul_mod257_u16(alpha_u16, alpha_u16);
                        let beta2 = mul_mod257_u16(beta_u16, beta_u16);
                        let miss_base = mul_mod257_u16(
                            c2,
                            add_mod257_u16(mul_mod257_u16(rho2, alpha2), mul_mod257_u16(sigma2, beta2)),
                        );
                        let inv_u = inv_mod257_u16(u_dbg).unwrap_or(0u16);
                        let b_inv = if inv_u == 0 {
                            0u16
                        } else {
                            let idx = (inv_u as usize).saturating_sub(1);
                            ((ubits[idx / 8] >> (idx % 8)) & 1) as u16
                        };
                        let idx_dir = (u_dbg as usize).saturating_sub(1);
                        let b_dir = ((ubits[idx_dir / 8] >> (idx_dir % 8)) & 1) as u16;
                        format!(
                            "miss(mu/nu): alpha={} beta={} miss_base={} miss+binv={} miss+bdir={}",
                            alpha_u16,
                            beta_u16,
                            miss_base,
                            add_mod257_u16(miss_base, b_inv),
                            add_mod257_u16(miss_base, b_dir)
                        )
                    } else {
                        "miss(mu/nu): disabled".to_string()
                    };

                    // For the true `s`, find the actual `u` solutions for the cross-only completion.
                    let cross_u_solutions = if let (Some(gamma), Some(_u_dbg)) = (gamma_dbg, u_dbg) {
                        let two_rho_sigma = mul_mod257_u16(2u16, mul_mod257_u16(rho, sigma));
                        let cross_term = mul_mod257_u16(two_rho_sigma, gamma);
                        let c2_cross = mul_mod257_u16(c2, cross_term);
                        let mut sols0: Vec<u16> = Vec::new();
                        let mut sols1: Vec<u16> = Vec::new();
                        for u in 1u16..=256u16 {
                            let inv_u = inv_mod257_u16(u).unwrap_or(0u16);
                            if inv_u == 0 {
                                continue;
                            }
                            let idx = (inv_u as usize).saturating_sub(1);
                            let b = ((ubits[idx / 8] >> (idx % 8)) & 1) as u16;
                            let comp = sub_mod257_u16(sub_mod257_u16(b, mul_mod257_u16(c1, u)), c2_cross);
                            if comp == c_req0 {
                                sols0.push(u);
                            }
                            if comp == c_req1 {
                                sols1.push(u);
                            }
                        }
                        // Print only a prefix to keep logs sane.
                        let p0: Vec<u16> = sols0.iter().copied().take(8).collect();
                        let p1: Vec<u16> = sols1.iter().copied().take(8).collect();
                        format!(
                            "cross_u_solutions: c_req0_len={} c_req0_prefix={:?} c_req1_len={} c_req1_prefix={:?}",
                            sols0.len(),
                            p0,
                            sols1.len(),
                            p1
                        )
                    } else {
                        "cross_u_solutions: disabled".to_string()
                    };

                    dbg_eq_line = Some(format!(
                        "[LF_ID_EQ] rep_id={} g={} s_true={} y_anchor={} y_div={} a0={} a1={} c_req0={} c_req1={} lambda={} rho={} sigma={} u={:?} gamma={:?} c1={} c2={} b_u(inv)={:?} b_u(dir)={:?} expr_lin(inv)={:?} expr_sq(inv)={:?} expr_sq(dir)={:?} expr_sq(u-lam,inv)={:?} expr_sq(u-lam,dir)={:?} expr_cross(inv)={:?} expr_fixed_uv={:?}",
                        lock.rep_id,
                        g,
                        ts,
                        y_anchor,
                        y_div,
                        a0_u16,
                        a1_u16,
                        c_req0,
                        c_req1,
                        lambda,
                        rho,
                        sigma,
                        u_dbg,
                        gamma_dbg,
                        c1,
                        c2,
                        b_u,
                        b_u_dir,
                        expr_lin,
                        expr_sq,
                        expr_sq_dir,
                        expr_sq_u_minus_lambda,
                        expr_sq_u_minus_lambda_dir,
                        expr_cross,
                        expr_fixed_uv,
                    ));
                    dbg_eq_line =
                        dbg_eq_line.map(|l| format!("{} {} {} {} {}", l, tail_solutions_info, cross_solutions_info, miss_info, cross_u_solutions));
                }
            }

            // Tailless narrowing aligned with what the hint stream actually contains.
            //
            // For a candidate `s`, decrypting `Ubits(s)` yields candidate coefficients `(c1,c2)`.
            // The tailless streamed anchor gives:
            //   y_div := y_anchor / s  ==  c1*(ρ·α + σ·β) + c2*(2ρσ)·γ
            // for the correct `s` (and is random-looking for wrong `s`).
            //
            // This gives strong selectivity (~1/257) without any public probe surface.
            let two_rho_sigma = mul_mod257_u16(2u16, mul_mod257_u16(rho_u16, sigma_u16));
            let u_ab = u_fix;
            let y_pred = add_mod257_u16(
                mul_mod257_u16(c1, u_ab),
                mul_mod257_u16(c2, mul_mod257_u16(two_rho_sigma, gamma_u16)),
            );
            let hit = y_div == y_pred;

            if let Some(ts) = true_s_dbg {
                if s_probe == ts {
                    dbg_s_true = Some(ts);
                }
            }
            if hit {
                hits.push(s_probe);
            }
        }

        if let Some(ts) = dbg_s_true {
            let in_hits = hits.iter().any(|&x| x == ts);
            eprintln!(
                "[LF_UBITS] rep_id={} g={} y_anchor={} a0={} a1={} s_true={} in_hits={} hits_len={} (u_override={:?} gamma_override={:?})",
                lock.rep_id,
                g,
                y_anchor,
                a0_u16,
                a1_u16,
                ts,
                in_hits,
                hits.len(),
                u_override,
                gamma_override,
            );
            if let Some(line) = dbg_eq_line {
                eprintln!("{}", line);
            }
        }

        // Completeness / no reject oracle: if no hit, fall back to full domain.
        if hits.is_empty() {
            hits = (1u16..=256u16).collect();
        }

        // Return same hit set for both accepting-set branches (constant-shape outer API).
        Ok(vec![hits.clone(), hits])
    }

    pub fn finish_decrypt_candidates(self) -> Result<Vec<Vec<u8>>, String> {
        self.finish_decrypt_candidates_with_gate(1u16, None, None)
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
        let row = self.blocks.entry(block).or_insert_with(|| vec![0u16; self.d]);
        let v = (f_to_u64(coeff) % (MOD_257 as u64)) as u16;
        row[pos] = add_mod257_u16(row[pos], v);
        Ok(())
    }

    pub(crate) fn into_sparse_blocks(&mut self) -> Vec<(usize, PackedF257Block64)> {
        let blocks = std::mem::take(&mut self.blocks);
        blocks.into_iter().map(|(idx, row)| (idx, query_row_to_packed_f257(row.as_slice()))).collect()
    }
}

// ---------------------------------------------------------------------------
// Lock artifact API
// ---------------------------------------------------------------------------

impl<F: PrimeField> RingLweLockArtifact<F> {
    pub fn decap_state<'a>(&'a self, x: &[F]) -> Result<RingLweDecapStreamState<'a, F>, String> {
        RingLweDecapStreamState::new(self, x)
    }

    pub(crate) fn decrypt_payload_with_s_candidate(&self, s_candidate: u16) -> Vec<u8> {
        let key_wrap = derive_wrap_key_bytes(
            &self.params.domain_label,
            &self.c_stmt,
            &self.coins,
            s_candidate % MOD_257,
        );
        let payload_key_bytes = xor_stream_decrypt(&key_wrap, &self.cts[0].nonce, &self.cts[0].ct);
        let mut payload_key = [0u8; 32];
        if payload_key_bytes.len() == 32 {
            payload_key.copy_from_slice(payload_key_bytes.as_slice());
        } else {
            payload_key = sha256_32(&[
                b"LFP_BAD_WRAP_KEY_LEN_V1",
                payload_key_bytes.as_slice(),
                &s_candidate.to_le_bytes(),
            ]);
        }
        xor_stream_decrypt(&payload_key, &self.payload_ct.nonce, &self.payload_ct.ct)
    }
}

/// Arm (create) a lock artifact with deterministic mod-257 hints + unauthenticated XOR.
pub fn arm_ringlwe_lock<F: PrimeField>(
    c_stmt: Vec<F>,
    accepting_set_shifted: [F; 2],
    anchor_block_id: u32,
    rep_id: u64,
    s_override: Option<u16>,
    poison_blocks: u32,
    coins: dpp::theorem43::Theorem43Coins<F>,
    offset: F,
    x_len: usize,
    pi_len: usize,
    ubits_plain: [u8; 32],
    q_blocks: Vec<(usize, PackedF257Block64)>,
    err_gate_hints_base: ErrGateHints,
    params: RingLweParams,
    payload: &[u8],
    rng: &mut impl RngCore,
) -> Result<RingLweLockArtifact<F>, String> {
    // Canonical oneproof/global-combine model uses 32-byte per-lock shares.
    // Keeping this fixed avoids local format distinguishers across candidate attempts.
    if payload.len() != 32 {
        return Err("arm_ringlwe_lock: payload must be exactly 32 bytes".to_string());
    }

    // Keep accepting-set safety check in the manifest: decap derives candidate secret scalars
    // from y/a over this set.
    for a in &accepting_set_shifted {
        if a.is_zero() {
            return Err("arm_ringlwe_lock: shifted accepting set contains 0; resample rep_id".to_string());
        }
    }

    // Scalar secret for this lock's anchor mask.
    //
    // For logical locks with repetitions, we may intentionally reuse the same `s` across reps
    // (rep intersection strategy). For non-logical single locks, pass `None` to sample fresh.
    let s_u16: u16 = if let Some(s) = s_override {
        let s = s % MOD_257;
        if s == 0 {
            return Err("arm_ringlwe_lock: s_override=0 (must be nonzero mod 257)".to_string());
        }
        s
    } else {
        sample_nonzero_f257_scalar(rng)
    };
    debug_record_s_for_rep(rep_id, s_u16);
    if std::env::var("LFP_DEBUG_ARM").ok().as_deref() == Some("1") {
        eprintln!(
            "[LF_DEBUG_ARM] rep_id={} s={} c_hit={} accepting_set=({}, {}) offset={} anchor_block_id={} poison_blocks={}",
            rep_id,
            s_u16,
            field_mod257_u16(&coins.c_hit),
            field_mod257_u16(&accepting_set_shifted[0]),
            field_mod257_u16(&accepting_set_shifted[1]),
            field_mod257_u16(&offset),
            anchor_block_id,
            poison_blocks
        );
    }
    if std::env::var("LFP_DEBUG_IDENTITY").ok().as_deref() == Some("1") {
        let rep_filter = std::env::var("LFP_DEBUG_REP_ID")
            .ok()
            .and_then(|v| v.parse::<u64>().ok());
        if rep_filter.is_none() || rep_filter == Some(rep_id) {
            eprintln!(
                "[LF_ID_ARM] rep_id={} s_true={} c_hit={} offset={} a_shift=({}, {}) coins(idx={},lambda={},rho={},sigma={}) domain={:02x?} c_stmt_len={}",
                rep_id,
                s_u16,
                field_mod257_u16(&coins.c_hit),
                field_mod257_u16(&offset),
                field_mod257_u16(&accepting_set_shifted[0]),
                field_mod257_u16(&accepting_set_shifted[1]),
                coins.idx,
                field_mod257_u16(&coins.lambda),
                field_mod257_u16(&coins.rho),
                field_mod257_u16(&coins.sigma),
                &params.domain_label[..8],
                c_stmt.len()
            );
        }
    }

    // Deterministic hint blocks: h = q * s (mod 257).
    let h_blocks: Vec<(usize, PackedF257Block64)> =
        q_blocks.iter().map(|(block_idx, q)| (*block_idx, q.scale_mod257(s_u16))).collect();
    let anchor_hints = BranchHints { hint_blocks_sparse: h_blocks };

    // Encrypt the UV membership indicator bits under a low-entropy `s` using a pad that yields
    // *no* publicly checkable structure on decryption (plaintext space is all 256-bit strings).
    let pad = derive_ubits_pad_bytes(
        &params.domain_label,
        &c_stmt,
        &coins,
        rep_id,
        s_u16,
    );
    let mut ct_ubits = [0u8; 32];
    for i in 0..32 {
        ct_ubits[i] = ubits_plain[i] ^ pad[i];
    }

    // Residual gate hints: publish only masked mixes `h = s * w (mod 257)`.
    //
    // This keeps the mix weights hidden, while the gate value remains unchanged because
    // `(s*z)^256 == z^256` for any nonzero `s` in F257.
    let mixes_scaled: Vec<PackedF257Block64> = err_gate_hints_base
        .mixes
        .iter()
        .map(|blk| blk.scale_mod257(s_u16))
        .collect();
    let err_gate_hints = ErrGateHints {
        k: err_gate_hints_base.k,
        blocks_per_mix: err_gate_hints_base.blocks_per_mix,
        mixes: mixes_scaled,
    };

    // Two-phase lock envelope:
    // 1) sample random payload key K_payload and encrypt payload under it,
    // 2) wrap K_payload under armer-known anchor secret s.
    let mut payload_key = [0u8; 32];
    rng.fill_bytes(&mut payload_key);
    let mut payload_nonce = [0u8; 12];
    rng.fill_bytes(&mut payload_nonce);
    let payload_ct = LockCiphertext {
        nonce: payload_nonce,
        ct: xor_stream_encrypt(&payload_key, &payload_nonce, payload),
    };

    // Single wrapped-key ciphertext under armer-known anchor secret `s`.
    let wrap_key = derive_wrap_key_bytes(&params.domain_label, &c_stmt, &coins, s_u16);
    let mut wrap_nonce = [0u8; 12];
    rng.fill_bytes(&mut wrap_nonce);
    let wrap_ct = xor_stream_encrypt(&wrap_key, &wrap_nonce, payload_key.as_slice());
    let cts = vec![LockCiphertext {
        nonce: wrap_nonce,
        ct: wrap_ct,
    }];

    Ok(RingLweLockArtifact {
        c_stmt,
        accepting_set: accepting_set_shifted,
        offset,
        anchor_block_id,
        rep_id,
        poison_blocks,
        x_len,
        pi_len,
        len: x_len + pi_len,
        coins,
        params,
        anchor_hints,
        err_gate_hints,
        cts,
        ct_ubits,
        payload_ct,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use latticefold::transcript::poseidon::F257;
    use rand::{rngs::StdRng, SeedableRng};

    #[test]
    fn test_package_only_rep_intersection_pruning_requires_reused_payload() {
        use std::collections::BTreeSet;
        use rand::RngCore;

        let mut rng = StdRng::seed_from_u64(0xD15EA5E_u64);
        let c_stmt = vec![F257::from(11u64), F257::from(29u64), F257::from(101u64)];
        let accepting_set = [F257::from(5u64), F257::from(6u64)];
        let coins = dpp::theorem43::Theorem43Coins::<F257> {
            idx: 7usize,
            lambda: F257::from(13u64),
            rho: F257::from(9u64),
            sigma: F257::from(17u64),
            c_hit: F257::from(5u64),
        };
        let params = RingLweParams::default();

        // Case 1 (insecure): reuse the same 32-byte payload across reps.
        let payload_master = [0xABu8; 32];
        let lock0 = arm_ringlwe_lock::<F257>(
            c_stmt.clone(),
            accepting_set,
            0u32,
            1u64,
            None,
            1u32,
            coins.clone(),
            F257::from(0u64),
            0usize,
            0usize,
            [0u8; 32],
            Vec::new(),
            ErrGateHints {
                k: 0,
                blocks_per_mix: 1,
                mixes: Vec::new(),
            },
            params.clone(),
            &payload_master,
            &mut rng,
        )
        .expect("arm lock0");
        let lock1 = arm_ringlwe_lock::<F257>(
            c_stmt.clone(),
            accepting_set,
            0u32,
            2u64,
            None,
            1u32,
            coins.clone(),
            F257::from(0u64),
            0usize,
            0usize,
            [0u8; 32],
            Vec::new(),
            ErrGateHints {
                k: 0,
                blocks_per_mix: 1,
                mixes: Vec::new(),
            },
            params.clone(),
            &payload_master,
            &mut rng,
        )
        .expect("arm lock1");

        let set0: BTreeSet<[u8; 32]> = (1u16..=256u16)
            .map(|s| {
                let p = lock0.decrypt_payload_with_s_candidate(s);
                let mut a = [0u8; 32];
                a.copy_from_slice(&p[..32]);
                a
            })
            .collect();
        let set1: BTreeSet<[u8; 32]> = (1u16..=256u16)
            .map(|s| {
                let p = lock1.decrypt_payload_with_s_candidate(s);
                let mut a = [0u8; 32];
                a.copy_from_slice(&p[..32]);
                a
            })
            .collect();
        let inter_same: BTreeSet<[u8; 32]> = set0.intersection(&set1).copied().collect();
        assert!(
            inter_same.contains(&payload_master),
            "reused payload across reps enables package-only intersection pruning"
        );

        // Case 2 (hardened): unlink reps by XOR-splitting the logical payload.
        let mut m = [0u8; 32];
        rng.fill_bytes(&mut m);
        let mut last = payload_master;
        for i in 0..32 {
            last[i] ^= m[i];
        }
        let lock2 = arm_ringlwe_lock::<F257>(
            c_stmt.clone(),
            accepting_set,
            0u32,
            3u64,
            None,
            1u32,
            coins.clone(),
            F257::from(0u64),
            0usize,
            0usize,
            [0u8; 32],
            Vec::new(),
            ErrGateHints {
                k: 0,
                blocks_per_mix: 1,
                mixes: Vec::new(),
            },
            params.clone(),
            &m,
            &mut rng,
        )
        .expect("arm lock2");
        let lock3 = arm_ringlwe_lock::<F257>(
            c_stmt,
            accepting_set,
            0u32,
            4u64,
            None,
            1u32,
            coins,
            F257::from(0u64),
            0usize,
            0usize,
            [0u8; 32],
            Vec::new(),
            ErrGateHints {
                k: 0,
                blocks_per_mix: 1,
                mixes: Vec::new(),
            },
            params,
            &last,
            &mut rng,
        )
        .expect("arm lock3");
        let set2: BTreeSet<[u8; 32]> = (1u16..=256u16)
            .map(|s| {
                let p = lock2.decrypt_payload_with_s_candidate(s);
                let mut a = [0u8; 32];
                a.copy_from_slice(&p[..32]);
                a
            })
            .collect();
        let set3: BTreeSet<[u8; 32]> = (1u16..=256u16)
            .map(|s| {
                let p = lock3.decrypt_payload_with_s_candidate(s);
                let mut a = [0u8; 32];
                a.copy_from_slice(&p[..32]);
                a
            })
            .collect();
        let inter_split: BTreeSet<[u8; 32]> = set2.intersection(&set3).copied().collect();
        assert!(
            !inter_split.contains(&payload_master),
            "unlinkable rep payloads should prevent recovering the logical payload by intersection"
        );
        assert!(
            inter_split.is_empty(),
            "rep unlinkability should make package-only candidate-set intersection empty (negligible collision probability)"
        );
    }
}
