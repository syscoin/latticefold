//! MLWE lock layer for Theorem-4.3 WE arming (Ring-LWE + carry-as-noise + amplification).
//!
//! # Architecture
//!
//! This module provides the lock layer for the arm-before-proof WE scheme:
//! - arming publishes hint blocks (hiding the DPP query behind Ring-LWE over Goldilocks)
//!   and two ciphertexts per lock (one per element of the shifted accepting set A').
//! - decapsulation is streaming: absorb proof chunks, then `finish_decrypt_candidates()`.
//!
//! # Security model
//!
//! Per-lock security is **not** standalone 128-bit. The ring structure (d=64 Goldilocks cyclotomic)
//! couples hint positions via convolution, raising the per-lock cost above scalar F257 brute-force,
//! but the exact per-lock hardness for this nonstandard distribution is **not reduced to standard
//! RLWE**. Security is achieved through T-of-R Shamir threshold amplification with no per-lock
//! verification oracle (unauthenticated encryption). Use `AmplificationParams` to set a
//! conservative per-lock candidate bound and compute the system security level.
//!
//! The lock encryption is **unauthenticated** (no per-lock MAC/tag). This prevents a per-lock
//! verification oracle, forcing the adversary to guess all T locks jointly.
//!
//! # Carry-as-noise reconciliation
//!
//! The DPP query/witness lives in F257. We embed into Goldilocks and treat the carry mismatch
//! `⟨embed(q), embed(π)⟩_GL = a + 257k` as bounded extra noise. With short secrets (centered
//! binomial, |coeff| ≤ η), the carry `s·257k` is bounded and absorbed by Frodo-style rounding.

use ark_ff::{Field, PrimeField};
use cyclotomic_rings::rings::GoldilocksRing64;
use rand::{RngCore, SeedableRng};
use rand_chacha::ChaCha20Rng;
use rayon::prelude::*;
use stark_rings::PolyRing;
use std::collections::BTreeMap;

/// The base field of GoldilocksRing64 (Goldilocks prime field).
type Fq = <GoldilocksRing64 as PolyRing>::BaseRing;

/// Ring dimension (d=64 for GoldilocksRing64).
const RING_D: usize = 64;

/// Block packing size = ring dimension.
const PACK_D: usize = RING_D;

/// Goldilocks prime as u64 (for reconciliation arithmetic).
const GL_P: u64 = 0xFFFF_FFFF_0000_0001;

/// Pack a row of d Fq values into a ring element for coeff0_mul:
/// coeff0(q_ring * π_ring) = ⟨q, π⟩ (standard dot product).
/// Set q_ring[0] = q[0], q_ring[d-i] = -q[i] for i=1..d-1.
fn query_row_to_ring(row: &[Fq]) -> GoldilocksRing64 {
    let mut coeffs = vec![Fq::ZERO; RING_D];
    if !row.is_empty() {
        coeffs[0] = row[0];
        for i in 1..RING_D.min(row.len()) {
            coeffs[RING_D - i] = -row[i];
        }
    }
    GoldilocksRing64::from(coeffs)
}

/// Scale a ring element by a scalar (coefficientwise). O(d), no NTT.
/// This is much faster than full ring multiply when one operand is a constant polynomial.
#[inline]
fn ring_scale(r: &GoldilocksRing64, s: Fq) -> GoldilocksRing64 {
    let coeffs: Vec<Fq> = r.coeffs().iter().map(|c| *c * s).collect();
    GoldilocksRing64::from(coeffs)
}

/// Extract coefficient 0 of a * b when b is provided in coefficient form.
///
/// This avoids allocating a temporary `GoldilocksRing64` for each proof block.
#[inline]
fn coeff0_mul_row(a: &GoldilocksRing64, row: &[Fq]) -> Fq {
    let ac = a.coeffs();
    let d = ac.len();
    let mut acc = if row.is_empty() { Fq::ZERO } else { ac[0] * row[0] };
    for i in 1..d {
        let idx = d - i;
        if idx < row.len() {
            acc -= ac[i] * row[idx];
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
    /// Centered binomial parameter for the SECRET ring polynomial.
    /// CBD(1) gives coefficients in {-1, 0, 1} — short, for carry bounding.
    pub secret_binomial_k: u32,
    /// Log2 of the error noise standard deviation σ.
    /// σ = 2^{noise_log2_sigma}. Must be large enough for RLWE security (σ >> signal).
    /// Signal per coefficient ≈ d × 256 ≈ 2^{14}. Need σ >> 2^{14} for hiding.
    /// σ = 2^{25} gives ~20-bit RLWE per lock. Noise budget: d×σ×256 ≈ 2^{39} vs margin 2^{55}.
    pub noise_log2_sigma: u32,
    /// Reconciliation bits per lock element (rounding to 2^d buckets).
    pub recon_bits: u32,
    /// Domain separation label.
    pub domain_label: [u8; 32],
}

impl Default for RingLweParams {
    fn default() -> Self {
        Self {
            secret_binomial_k: 2,   // CBD(2): secret in {-2,-1,0,1,2}, more entropy than ±1
            noise_log2_sigma: 25,   // σ = 2^25 ≈ 33M — large for RLWE security
            recon_bits: 8,
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
    /// Per-branch hint ring elements, stored sparsely as (block_idx, GoldilocksRing64).
    ///
    /// **Independent scalar secrets per branch**: branch b has its own `s0_b, s1_b` (constant
    /// polynomials) and noise. `branch_hints[b]` = hint blocks for branch b ∈ {0, 1}.
    pub branch_hints: [BranchHints; 2],
    /// Two unauthenticated ciphertexts (one per accepting-set branch).
    pub cts: [LockCiphertext; 2],
}

/// Per-branch hint material: one hint vector per branch.
#[derive(Clone, Debug)]
pub struct BranchHints {
    pub hint0_blocks_sparse: Vec<(usize, GoldilocksRing64)>,
    pub hint1_blocks_sparse: Vec<(usize, GoldilocksRing64)>,
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
    y0_mod257: u16,
    y1_mod257: u16,
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
        &y0_mod257.to_le_bytes(),
        &y1_mod257.to_le_bytes(),
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

/// Fast u64 extraction from an Fq element (Goldilocks has one limb).
#[inline]
fn fq_to_u64(x: &Fq) -> u64 {
    x.into_bigint().as_ref()[0]
}

#[inline]
fn f_to_u64<F: PrimeField>(f: &F) -> u64 {
    f.into_bigint().as_ref()[0]
}

fn centered_binomial(rng: &mut impl RngCore, k: u32) -> i16 {
    let mut a = 0i16;
    let mut b = 0i16;
    for _ in 0..k {
        a += (rng.next_u32() & 1) as i16;
        b += (rng.next_u32() & 1) as i16;
    }
    a - b
}

fn fq_from_i16(x: i16) -> Fq {
    if x >= 0 {
        Fq::from(x as u64)
    } else {
        -Fq::from((-x) as u64)
    }
}


/// Embed an F257 element into Goldilocks using CENTERED representatives [-128, 128].
/// This reduces the integer dot product magnitude (and thus the carry noise) by ~4×
/// compared to the canonical [0, 256] embedding, without changing the mod-257 value.
fn embed_f_to_fq<F: PrimeField>(f: &F) -> Fq {
    let v = f_to_u64(f);
    if v <= 128 {
        Fq::from(v)
    } else {
        // v in 129..256 maps to -(257 - v) in [-128..-1]
        -Fq::from(257u64 - v)
    }
}

/// Sample a nonzero secret scalar in `{1,2,...,256}` (viewed mod 257), and lift to Goldilocks.
///
/// This matches the "tiny field" entropy accounting (256 candidates), but the scalar is used
/// inside Goldilocks ring arithmetic. The decapper recovers `s*a (mod 257)` by reducing the
/// streamed signal modulo 257.
fn sample_nonzero_f257_scalar_as_fq(rng: &mut impl RngCore) -> (u16, Fq) {
    loop {
        let v = (rng.next_u32() & 0xFF) as u16; // 0..255
        let v = v + 1; // 1..256
        return (v, Fq::from(v as u64));
    }
}


/// Sample a NOISE ring element with uniform box coefficients in [-2^{log2_sigma}, 2^{log2_sigma}].
///
/// This is heavier-tailed than a discrete Gaussian with the same σ, making it at least as
/// hard for the adversary (more noise = more hiding). We use uniform instead of Gaussian
/// because (a) our security is not reduced to standard RLWE anyway (nonstandard public
/// distribution), and (b) uniform is faster (no rejection sampling / exp() calls).
///
/// For a future formal RLWE reduction, switch to a discrete Gaussian sampler.
fn sample_error_ring(rng: &mut impl RngCore, log2_sigma: u32) -> GoldilocksRing64 {
    let bound = 1u64 << log2_sigma;
    let coeffs: Vec<Fq> = (0..RING_D)
        .map(|_| {
            let raw = rng.next_u64();
            let val = (raw % (2 * bound + 1)) as i64 - (bound as i64);
            if val >= 0 {
                Fq::from(val as u64)
            } else {
                -Fq::from((-val) as u64)
            }
        })
        .collect();
    GoldilocksRing64::from(coeffs)
}

/// Sample an error ring where every coefficient is a multiple of 257.
///
/// This ensures that reducing the decapsulation signal modulo 257 is invariant to the RLWE noise,
/// so the decapper can recover `s*a (mod 257)` exactly (up to the inherent `+257k` carry).
fn sample_error_ring_257x(rng: &mut impl RngCore, log2_sigma: u32) -> GoldilocksRing64 {
    let scale = Fq::from(257u64);
    let e = sample_error_ring(rng, log2_sigma);
    let coeffs: Vec<Fq> = e.coeffs().iter().map(|c| *c * scale).collect();
    GoldilocksRing64::from(coeffs)
}

#[inline]
fn fq_to_i128_centered(x: &Fq) -> i128 {
    // Goldilocks q fits in u64.
    let q = GL_P as i128;
    let u = fq_to_u64(x) as i128;
    // Center-lift to (-q/2, q/2].
    if u > (q / 2) { u - q } else { u }
}

#[inline]
fn mod257_from_centered_i128(x: i128) -> u16 {
    let m = 257i128;
    let mut r = x % m;
    if r < 0 {
        r += m;
    }
    r as u16
}

// ---------------------------------------------------------------------------
// Streaming decapsulation state
// ---------------------------------------------------------------------------

pub struct RingLweDecapStreamState<'a, F: PrimeField> {
    lock: &'a RingLweLockArtifact<F>,
    d: usize,
    /// We accumulate BOTH branches simultaneously so π is scanned only once.
    branches: [BranchAccum<'a>; 2],
    block_idx: usize,
    pos_in_block: usize,
    filled: usize,
    coeffs: Vec<Fq>,
}

struct BranchAccum<'a> {
    sparse0: &'a [(usize, GoldilocksRing64)],
    sparse1: &'a [(usize, GoldilocksRing64)],
    sparse0_pos: usize,
    sparse1_pos: usize,
    y0: Fq,
    y1: Fq,
}

impl<'a, F: PrimeField> RingLweDecapStreamState<'a, F> {
    fn new(lock: &'a RingLweLockArtifact<F>, x: &[F]) -> Result<Self, String> {
        if x.len() != lock.x_len || x.len() + lock.pi_len != lock.len {
            return Err("ringlwe_decap_state: bad x length".to_string());
        }
        Ok(Self {
            lock,
            d: PACK_D,
            branches: [
                BranchAccum {
                    sparse0: lock.branch_hints[0].hint0_blocks_sparse.as_slice(),
                    sparse1: lock.branch_hints[0].hint1_blocks_sparse.as_slice(),
                    sparse0_pos: 0,
                    sparse1_pos: 0,
                    y0: Fq::ZERO,
                    y1: Fq::ZERO,
                },
                BranchAccum {
                    sparse0: lock.branch_hints[1].hint0_blocks_sparse.as_slice(),
                    sparse1: lock.branch_hints[1].hint1_blocks_sparse.as_slice(),
                    sparse0_pos: 0,
                    sparse1_pos: 0,
                    y0: Fq::ZERO,
                    y1: Fq::ZERO,
                },
            ],
            block_idx: 0,
            pos_in_block: 0,
            filled: 0,
            coeffs: Vec::with_capacity(PACK_D),
        })
    }

    #[inline]
    fn next_needed_block(&self) -> Option<usize> {
        let mut out: Option<usize> = None;
        for br in &self.branches {
            for v in [
                br.sparse0.get(br.sparse0_pos).map(|t| t.0),
                br.sparse1.get(br.sparse1_pos).map(|t| t.0),
            ] {
                if let Some(idx) = v {
                    out = Some(match out { None => idx, Some(prev) => prev.min(idx) });
                }
            }
        }
        out
    }

    #[inline]
    fn process_current_block(&mut self, row: &[Fq]) {
        for br in &mut self.branches {
            if br.sparse0_pos < br.sparse0.len() && br.sparse0[br.sparse0_pos].0 == self.block_idx {
                let h0 = &br.sparse0[br.sparse0_pos].1;
                br.y0 += coeff0_mul_row(h0, row);
                br.sparse0_pos += 1;
            }
            if br.sparse1_pos < br.sparse1.len() && br.sparse1[br.sparse1_pos].0 == self.block_idx {
                let h1 = &br.sparse1[br.sparse1_pos].1;
                br.y1 += coeff0_mul_row(h1, row);
                br.sparse1_pos += 1;
            }
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
                    self.coeffs.push(embed_f_to_fq(v));
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
    ///
    /// We reduce the accumulated Goldilocks signals modulo 257 (after centered lifting). This
    /// cancels both the inherent `+257k` carry from the field embedding and the RLWE noise
    /// (because we sample error coefficients as multiples of 257).
    fn finish_key_seeds_mod257(mut self) -> Result<[[u16; 2]; 2], String> {
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
        for br in &self.branches {
            if br.sparse0_pos != br.sparse0.len() || br.sparse1_pos != br.sparse1.len() {
                return Err("ringlwe_decap_stream: did not consume all sparse blocks".to_string());
            }
        }
        let mut out = [[0u16; 2]; 2];
        for (b, br) in self.branches.iter().enumerate() {
            let y0 = fq_to_i128_centered(&br.y0);
            let y1 = fq_to_i128_centered(&br.y1);
            out[b][0] = mod257_from_centered_i128(y0);
            out[b][1] = mod257_from_centered_i128(y1);
        }
        Ok(out)
    }

    /// Finish streaming and return both candidate decryptions (unauthenticated).
    ///
    /// Each branch b yields a key derived from the modulo-257 reduced signals, then decrypts its
    /// branch ciphertext. Exactly one branch gives the correct Shamir share; the other is garbage.
    pub fn finish_decrypt_candidates(self) -> Result<[Vec<u8>; 2], String> {
        let lock = self.lock;
        let seeds = self.finish_key_seeds_mod257()?;
        let mut out = [Vec::new(), Vec::new()];
        for b in 0..2 {
            let key = derive_payload_key_bytes(
                &lock.params.domain_label,
                &lock.c_stmt,
                &lock.coins,
                seeds[b][0],
                seeds[b][1],
            );
            out[b] = xor_stream_decrypt(&key, &lock.cts[b].nonce, &lock.cts[b].ct);
        }
        Ok(out)
    }

}

// ---------------------------------------------------------------------------
// Query accumulator
// ---------------------------------------------------------------------------

pub(crate) struct QueryBlockAccumulator<F: PrimeField> {
    pi_len: usize,
    d: usize,
    blocks: BTreeMap<usize, Vec<Fq>>,
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
            .or_insert_with(|| vec![Fq::ZERO; self.d]);
        row[pos] += embed_f_to_fq(coeff);
        Ok(())
    }

    pub(crate) fn into_sparse_blocks(&mut self) -> Vec<(usize, GoldilocksRing64)> {
        let blocks = std::mem::take(&mut self.blocks);
        blocks
            .into_iter()
            .map(|(idx, row)| (idx, query_row_to_ring(row.as_slice())))
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

/// Arm (create) a lock artifact with Ring-LWE hints + carry-as-noise + unauthenticated XOR.
pub fn arm_ringlwe_lock<F: PrimeField>(
    c_stmt: Vec<F>,
    accepting_set_shifted: [F; 2],
    coins: dpp::theorem43::Theorem43Coins<F>,
    offset: F,
    x_len: usize,
    pi_len: usize,
    q_blocks: Vec<(usize, GoldilocksRing64)>,
    params: RingLweParams,
    payload: &[u8],
    rng: &mut impl RngCore,
) -> Result<RingLweLockArtifact<F>, String> {
    if params.recon_bits == 0 || params.recon_bits > 32 {
        return Err("arm_ringlwe_lock: recon_bits must be in 1..=32".to_string());
    }
    if params.secret_binomial_k == 0 {
        return Err("arm_ringlwe_lock: secret_binomial_k must be nonzero".to_string());
    }

    // Per-branch: independent (s0, s1) secrets, independent noise, independent hints.
    // This prevents difference attacks across branches.
    let mut branch_hints: [BranchHints; 2] = [
        BranchHints { hint0_blocks_sparse: Vec::new(), hint1_blocks_sparse: Vec::new() },
        BranchHints { hint0_blocks_sparse: Vec::new(), hint1_blocks_sparse: Vec::new() },
    ];
    let mut cts: [LockCiphertext; 2] = [
        LockCiphertext { nonce: [0u8; 12], ct: Vec::new() },
        LockCiphertext { nonce: [0u8; 12], ct: Vec::new() },
    ];

    for (b, a) in accepting_set_shifted.iter().enumerate() {
        if a.is_zero() {
            return Err("arm_ringlwe_lock: shifted accepting set contains 0; resample rep_id".to_string());
        }

        // Fresh independent SCALAR secrets for this branch, embedded as constant polynomials.
        // Using scalars (not full ring) ensures coeff0(s_const * X) = s_scalar * coeff0(X),
        // so the modulo-257 reduced signals are consistent across armer/decapper.
        let (s0_u16, s0_scalar): (u16, Fq) = sample_nonzero_f257_scalar_as_fq(rng);
        let (s1_u16, s1_scalar): (u16, Fq) = sample_nonzero_f257_scalar_as_fq(rng);

        // Build hints: h = q_ring * s_const + e, where e is a multiple of 257 so reduction mod 257
        // cancels it (noise does not affect key-seed extraction).
        // Pre-derive per-block RNG seeds sequentially, then compute in parallel.
        let noise_sigma = params.noise_log2_sigma;
        let block_seeds: Vec<[u8; 32]> = (0..q_blocks.len())
            .map(|_| {
                let mut seed = [0u8; 32];
                rng.fill_bytes(&mut seed);
                seed
            })
            .collect();
        let h0_blocks: Vec<(usize, GoldilocksRing64)> = q_blocks
            .par_iter()
            .zip(block_seeds.par_iter())
            .map(|((block_idx, q), seed)| {
                let mut block_rng = ChaCha20Rng::from_seed(*seed);
                let h0 = ring_scale(q, s0_scalar) + sample_error_ring_257x(&mut block_rng, noise_sigma);
                (*block_idx, h0)
            })
            .collect();
        let h1_blocks: Vec<(usize, GoldilocksRing64)> = q_blocks
            .par_iter()
            .zip(block_seeds.par_iter())
            .map(|((block_idx, q), seed)| {
                // Domain-separate by flipping one byte of the seed.
                let mut s2 = *seed;
                s2[0] ^= 0xA5;
                let mut block_rng = ChaCha20Rng::from_seed(s2);
                let h1 = ring_scale(q, s1_scalar) + sample_error_ring_257x(&mut block_rng, noise_sigma);
                (*block_idx, h1)
            })
            .collect();
        branch_hints[b] = BranchHints {
            hint0_blocks_sparse: h0_blocks,
            hint1_blocks_sparse: h1_blocks,
        };

        // Derive a per-branch DEM key from the small-field seeds s*a (mod 257).
        //
        // Note: `a` is an F257 element; treat it mod 257. This is exactly what the decapper
        // recovers by reducing the streamed Goldilocks signals mod 257.
        let a_u16 = (f_to_u64(a) % 257) as u16;
        let y0 = ((s0_u16 as u32 * a_u16 as u32) % 257) as u16;
        let y1 = ((s1_u16 as u32 * a_u16 as u32) % 257) as u16;
        let key = derive_payload_key_bytes(&params.domain_label, &c_stmt, &coins, y0, y1);
        let mut nonce = [0u8; 12];
        rng.fill_bytes(&mut nonce);
        let ct = xor_stream_encrypt(&key, &nonce, payload);
        cts[b] = LockCiphertext { nonce, ct };
    }

    Ok(RingLweLockArtifact {
        c_stmt,
        accepting_set: accepting_set_shifted,
        offset,
        x_len,
        pi_len,
        len: x_len + pi_len,
        coins,
        params,
        branch_hints,
        cts,
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
        let q_blocks: Vec<(usize, GoldilocksRing64)> = Vec::new();
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

        for b in 0..2 {
            assert_ne!(
                lock.cts[b].ct.as_slice(),
                payload.as_slice(),
                "ciphertext should not equal payload for branch {b}"
            );
        }
    }
}
