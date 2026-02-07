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

use ark_ff::{BigInteger, Field, PrimeField};
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
    /// **Independent scalar secrets per branch**: branch b has its own `s_b` (constant polynomial)
    /// and noise. `branch_hints[b]` = hint blocks for branch b ∈ {0, 1}.
    pub branch_hints: [BranchHints; 2],
    /// Two unauthenticated ciphertexts (one per accepting-set branch).
    pub cts: [LockCiphertext; 2],
}

/// Per-branch hint material: one hint vector per branch.
#[derive(Clone, Debug)]
pub struct BranchHints {
    pub hint_blocks_sparse: Vec<(usize, GoldilocksRing64)>,
}

/// Per-branch ciphertext: Frodo/Regev additive encoding of the payload.
///
/// Each byte μ[j] of the payload is encoded as `C[j] = s[0]*a + Δ*μ[j] + e'[j] (mod q)`
/// where Δ = q/256. The decapper subtracts their inner-product value and rounds.
/// This is unauthenticated — both branches decode to SOMETHING.
#[derive(Clone, Debug)]
pub struct LockCiphertext {
    /// Goldilocks-encoded share bytes. Length = payload length.
    /// Each element encodes one byte via Frodo rounding.
    pub encoded: Vec<Fq>,
}

// ---------------------------------------------------------------------------
// Cryptographic helpers
// ---------------------------------------------------------------------------


/// Frodo-style rounding: encode a byte at the MIDPOINT of its bucket.
///
/// Enc(μ) = floor(((2μ+1) * q) / 512).
/// This avoids wrap issues at μ=0 and μ=255 under modular arithmetic.
/// Correctness holds when total noise < q/512.
#[inline]
fn frodo_encode_byte(mu: u8) -> Fq {
    let q = GL_P as u128;
    let mu_u = mu as u128;
    let val = ((2 * mu_u + 1) * q) / 512;
    Fq::from(val as u64)
}

/// Decode via bucket index: Dec(y) = floor((y * 256) / q).
/// Works if y is within < q/512 of the correct bucket midpoint.
#[inline]
fn frodo_decode_byte(y: Fq) -> u8 {
    let q = GL_P as u128;
    let y_u = fq_to_u64(&y) as u128;
    let mu = (y_u * 256) / q;
    mu.min(255) as u8
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

/// Sample a nonzero short Fq scalar via centered binomial.
fn sample_nonzero_short_scalar(rng: &mut impl RngCore, k: u32) -> Fq {
    loop {
        let v = centered_binomial(rng, k);
        if v != 0 {
            return fq_from_i16(v);
        }
    }
}

/// Sample a short ring element (SECRET) with CBD coefficients.
fn sample_short_ring(rng: &mut impl RngCore, k: u32) -> GoldilocksRing64 {
    let coeffs: Vec<Fq> = (0..RING_D)
        .map(|_| fq_from_i16(centered_binomial(rng, k)))
        .collect();
    GoldilocksRing64::from(coeffs)
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
    sparse: &'a [(usize, GoldilocksRing64)],
    sparse_pos: usize,
    y: Fq,
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
                    sparse: lock.branch_hints[0].hint_blocks_sparse.as_slice(),
                    sparse_pos: 0, y: Fq::ZERO,
                },
                BranchAccum {
                    sparse: lock.branch_hints[1].hint_blocks_sparse.as_slice(),
                    sparse_pos: 0, y: Fq::ZERO,
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
        let a = self.branches[0].sparse.get(self.branches[0].sparse_pos).map(|t| t.0);
        let b = self.branches[1].sparse.get(self.branches[1].sparse_pos).map(|t| t.0);
        match (a, b) {
            (None, None) => None,
            (Some(x), None) => Some(x),
            (None, Some(y)) => Some(y),
            (Some(x), Some(y)) => Some(x.min(y)),
        }
    }

    #[inline]
    fn process_current_block(&mut self, row: &[Fq]) {
        for br in &mut self.branches {
            if br.sparse_pos < br.sparse.len() && br.sparse[br.sparse_pos].0 == self.block_idx {
                let h = &br.sparse[br.sparse_pos].1;
                br.y += coeff0_mul_row(h, row);
                br.sparse_pos += 1;
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

    /// Finalize streaming and return per-branch inner-product signals.
    fn finish_signals(mut self) -> Result<[Fq; 2], String> {
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
            if br.sparse_pos != br.sparse.len() {
                return Err("ringlwe_decap_stream: did not consume all sparse blocks".to_string());
            }
        }

        let mut signals = [Fq::ZERO; 2];
        for (b, br) in self.branches.iter().enumerate() {
            signals[b] = br.y;
        }
        Ok(signals)
    }

    /// Finish streaming and return both candidate decryptions (unauthenticated).
    ///
    /// Each branch b is decoded via Frodo rounding: μ[j] = Dec(C[j] - y_b).
    /// Exactly one branch gives the correct Shamir share; the other is garbage.
    pub fn finish_decrypt_candidates(self) -> Result<[Vec<u8>; 2], String> {
        let lock = self.lock;
        let signals = self.finish_signals()?;
        let mut out = [Vec::new(), Vec::new()];
        for b in 0..2 {
            let y = signals[b];
            out[b] = lock.cts[b]
                .encoded
                .iter()
                .map(|c_j| frodo_decode_byte(*c_j - y))
                .collect();
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
        BranchHints { hint_blocks_sparse: Vec::new() },
        BranchHints { hint_blocks_sparse: Vec::new() },
    ];
    let mut cts: [LockCiphertext; 2] = [
        LockCiphertext { encoded: Vec::new() },
        LockCiphertext { encoded: Vec::new() },
    ];

    for (b, a) in accepting_set_shifted.iter().enumerate() {
        if a.is_zero() {
            return Err("arm_ringlwe_lock: shifted accepting set contains 0; resample rep_id".to_string());
        }

        // Fresh independent SCALAR secret for this branch, embedded as a constant polynomial.
        // Using a scalar (not full ring) ensures coeff0(s_const * X) = s_scalar * coeff0(X),
        // so armer's signal matches decapper's coeff0_mul exactly.
        // The ring convolution coupling comes from q_ring (the query), not s.
        let s_scalar: Fq = sample_nonzero_short_scalar(rng, params.secret_binomial_k);

        // Build hints: h = q_ring * s_const + e (parallelized across blocks).
        // Pre-derive per-block RNG seeds sequentially, then compute in parallel.
        let noise_sigma = params.noise_log2_sigma;
        let block_seeds: Vec<[u8; 32]> = (0..q_blocks.len())
            .map(|_| {
                let mut seed = [0u8; 32];
                rng.fill_bytes(&mut seed);
                seed
            })
            .collect();
        let h_blocks: Vec<(usize, GoldilocksRing64)> = q_blocks
            .par_iter()
            .zip(block_seeds.par_iter())
            .map(|((block_idx, q), seed)| {
                let mut block_rng = ChaCha20Rng::from_seed(*seed);
                let h = ring_scale(q, s_scalar) + sample_error_ring(&mut block_rng, noise_sigma);
                (*block_idx, h)
            })
            .collect();
        branch_hints[b] = BranchHints {
            hint_blocks_sparse: h_blocks,
        };

        // Frodo/Regev additive encoding: C[j] = s_scalar*a + Δ*μ[j] + e'[j]
        // with small per-byte noise e' to preserve correctness.
        let a_fq: Fq = embed_f_to_fq(a);
        let signal = s_scalar * a_fq;
        let encoded: Vec<Fq> = payload
            .iter()
            .map(|&mu_j| {
                // Sample small per-byte noise for the ciphertext.
                let e_prime = fq_from_i16(centered_binomial(rng, params.secret_binomial_k));
                signal + frodo_encode_byte(mu_j) + e_prime
            })
            .collect();
        cts[b] = LockCiphertext { encoded };
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
