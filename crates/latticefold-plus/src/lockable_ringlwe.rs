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
use rand::RngCore;
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

/// Pack a proof chunk into a ring element (coefficients are the embedded F257 values).
fn pi_row_to_ring(row: &[Fq]) -> GoldilocksRing64 {
    let mut coeffs = vec![Fq::ZERO; RING_D];
    for i in 0..RING_D.min(row.len()) {
        coeffs[i] = row[i];
    }
    GoldilocksRing64::from(coeffs)
}

/// Extract coefficient 0 of the negacyclic ring product a * b in O(d).
///
/// For R = Z_q[x]/(x^d+1): coeff0(a·b) = a[0]*b[0] - Σ_{i=1}^{d-1} a[i]*b[d-i].
/// This avoids the full NTT ring multiply (which computes all d coefficients).
fn coeff0_mul(a: &GoldilocksRing64, b: &GoldilocksRing64) -> Fq {
    let ac = a.coeffs();
    let bc = b.coeffs();
    let d = ac.len();
    let mut acc = ac[0] * bc[0];
    for i in 1..d {
        acc -= ac[i] * bc[d - i];
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
    /// This is a heuristic estimate for the nonstandard RLWE distribution used here.
    /// The actual per-lock hardness is NOT reduced to standard RLWE; this bound should
    /// be set conservatively and the system security validated via `security_bits_*()`.
    pub candidates_per_lock: u64,
}

impl Default for AmplificationParams {
    fn default() -> Self {
        Self {
            r: 64,
            t: 7,
            candidates_per_lock: 1 << 19,
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
    /// Centered binomial parameter η: secret/noise coefficients in [-η, η].
    pub binomial_k: u32,
    /// Reconciliation bits per lock element (rounding to 2^d buckets).
    pub recon_bits: u32,
    /// Domain separation label.
    pub domain_label: [u8; 32],
}

impl Default for RingLweParams {
    fn default() -> Self {
        Self {
            binomial_k: 1, // binary secret: coefficients in {-1, 0, 1}
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
    /// **Independent secrets per branch**: branch b has its own (s0_b, s1_b) and noise.
    /// This prevents difference attacks if the cipher is ever upgraded to additive form.
    /// `branch_hints[b]` = (hint0_blocks, hint1_blocks) for branch b ∈ {0, 1}.
    pub branch_hints: [BranchHints; 2],
    /// Two unauthenticated ciphertexts (one per accepting-set branch).
    pub cts: [LockCiphertext; 2],
}

/// Per-branch hint material: two independent hint vectors (for 2× key entropy).
#[derive(Clone, Debug)]
pub struct BranchHints {
    pub hint0_blocks_sparse: Vec<(usize, GoldilocksRing64)>,
    pub hint1_blocks_sparse: Vec<(usize, GoldilocksRing64)>,
}

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

fn derive_payload_key_bytes<F: PrimeField>(
    domain_label: &[u8; 32],
    c_stmt: &[F],
    coins: &dpp::theorem43::Theorem43Coins<F>,
    y0: u64,
    y1: u64,
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
        b"LFP_DPP_PAYLOAD_KEY_V2",
        domain_label,
        stmt_bytes.as_slice(),
        coins_bytes.as_slice(),
        &y0.to_le_bytes(),
        &y1.to_le_bytes(),
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

/// Frodo-style rounding: extract d bits from a Goldilocks value.
/// reconcile(y, d) = round(y * 2^d / q) mod 2^d.
fn reconcile_u64(y: u64, d: u32) -> u64 {
    let q = GL_P as u128;
    let y_u = y as u128;
    let buckets = 1u128 << d;
    let rounded = (y_u * buckets + q / 2) / q;
    (rounded % buckets) as u64
}

fn f_to_u64<F: PrimeField>(f: &F) -> u64 {
    let bytes = f.into_bigint().to_bytes_le();
    let mut acc = 0u64;
    for (i, b) in bytes.iter().take(8).enumerate() {
        acc |= (*b as u64) << (8 * i);
    }
    acc
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

fn fq_to_u64(x: &Fq) -> u64 {
    let bytes = x.into_bigint().to_bytes_le();
    let mut acc = 0u64;
    for (i, b) in bytes.iter().take(8).enumerate() {
        acc |= (*b as u64) << (8 * i);
    }
    acc
}

fn embed_f_to_fq<F: PrimeField>(f: &F) -> Fq {
    Fq::from(f_to_u64(f))
}

fn sample_noise_ring(rng: &mut impl RngCore, k: u32) -> GoldilocksRing64 {
    let coeffs: Vec<Fq> = (0..RING_D)
        .map(|_| fq_from_i16(centered_binomial(rng, k)))
        .collect();
    GoldilocksRing64::from(coeffs)
}

/// Sample a short ring element with CBD coefficients, rejecting the all-zero polynomial.
fn sample_nonzero_short_ring(rng: &mut impl RngCore, k: u32) -> GoldilocksRing64 {
    loop {
        let r = sample_noise_ring(rng, k);
        // Reject if all coefficients are zero (degenerate).
        if r.coeffs().iter().any(|c| *c != Fq::ZERO) {
            return r;
        }
    }
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
                    sparse0_pos: 0, sparse1_pos: 0, y0: Fq::ZERO, y1: Fq::ZERO,
                },
                BranchAccum {
                    sparse0: lock.branch_hints[1].hint0_blocks_sparse.as_slice(),
                    sparse1: lock.branch_hints[1].hint1_blocks_sparse.as_slice(),
                    sparse0_pos: 0, sparse1_pos: 0, y0: Fq::ZERO, y1: Fq::ZERO,
                },
            ],
            block_idx: 0,
            filled: 0,
            coeffs: Vec::with_capacity(PACK_D),
        })
    }

    #[inline]
    fn maybe_process_full_block(&mut self) -> Result<(), String> {
        if self.coeffs.len() != self.d {
            return Ok(());
        }
        let pi_ring = pi_row_to_ring(self.coeffs.as_slice());
        for br in &mut self.branches {
            if br.sparse0_pos < br.sparse0.len() && br.sparse0[br.sparse0_pos].0 == self.block_idx {
                let h0 = &br.sparse0[br.sparse0_pos].1;
                br.y0 += coeff0_mul(h0, &pi_ring);
                br.sparse0_pos += 1;
            }
            if br.sparse1_pos < br.sparse1.len() && br.sparse1[br.sparse1_pos].0 == self.block_idx {
                let h1 = &br.sparse1[br.sparse1_pos].1;
                br.y1 += coeff0_mul(h1, &pi_ring);
                br.sparse1_pos += 1;
            }
        }
        self.block_idx += 1;
        self.coeffs.clear();
        Ok(())
    }

    pub fn absorb_chunk(&mut self, chunk: &[F]) -> Result<(), String> {
        for v in chunk {
            if self.filled >= self.lock.pi_len {
                return Err("ringlwe_decap_stream: too many π elements".to_string());
            }
            self.coeffs.push(embed_f_to_fq(v));
            self.filled += 1;
            self.maybe_process_full_block()?;
        }
        Ok(())
    }

    /// Finalize streaming and return one key per branch.
    fn finish_keys(mut self) -> Result<[[u8; 32]; 2], String> {
        if self.filled != self.lock.pi_len {
            return Err("ringlwe_decap_stream: bad π length".to_string());
        }
        // Flush remaining partial block.
        if !self.coeffs.is_empty() {
            while self.coeffs.len() < self.d {
                self.coeffs.push(Fq::ZERO);
            }
            let pi_ring = pi_row_to_ring(self.coeffs.as_slice());
            for br in &mut self.branches {
                if br.sparse0_pos < br.sparse0.len() && br.sparse0[br.sparse0_pos].0 == self.block_idx {
                    let h0 = &br.sparse0[br.sparse0_pos].1;
                    br.y0 += coeff0_mul(h0, &pi_ring);
                    br.sparse0_pos += 1;
                }
                if br.sparse1_pos < br.sparse1.len() && br.sparse1[br.sparse1_pos].0 == self.block_idx {
                    let h1 = &br.sparse1[br.sparse1_pos].1;
                    br.y1 += coeff0_mul(h1, &pi_ring);
                    br.sparse1_pos += 1;
                }
            }
            self.block_idx += 1;
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

        let rd = self.lock.params.recon_bits;
        let mut keys = [[0u8; 32]; 2];
        for (b, br) in self.branches.iter().enumerate() {
            let y0_r = reconcile_u64(fq_to_u64(&br.y0), rd);
            let y1_r = reconcile_u64(fq_to_u64(&br.y1), rd);
            keys[b] = derive_payload_key_bytes(
                &self.lock.params.domain_label,
                &self.lock.c_stmt,
                &self.lock.coins,
                y0_r,
                y1_r,
            );
        }
        Ok(keys)
    }

    /// Finish streaming and return both candidate decryptions (unauthenticated).
    ///
    /// Each branch b is decrypted with its own key (from its own independent secrets).
    /// Exactly one is the correct Shamir share; the other is garbage.
    pub fn finish_decrypt_candidates(self) -> Result<[Vec<u8>; 2], String> {
        let lock = self.lock;
        let keys = self.finish_keys()?;
        // Branch b's ciphertext is decrypted with branch b's key.
        let pt0 = xor_stream_decrypt(&keys[0], &lock.cts[0].nonce, &lock.cts[0].ct);
        let pt1 = xor_stream_decrypt(&keys[1], &lock.cts[1].nonce, &lock.cts[1].ct);
        Ok([pt0, pt1])
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
    if params.binomial_k == 0 {
        return Err("arm_ringlwe_lock: binomial_k must be nonzero".to_string());
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

        // Fresh independent RING secrets for this branch.
        // s must be a full ring element (not a scalar) so that ring multiplication
        // couples all d=64 positions via negacyclic convolution.
        let s0: GoldilocksRing64 = sample_nonzero_short_ring(rng, params.binomial_k);
        let s1: GoldilocksRing64 = sample_nonzero_short_ring(rng, params.binomial_k);

        // Build hints: h = q_ring * s + e (full ring multiply via NTT).
        let mut h0_blocks = Vec::with_capacity(q_blocks.len());
        let mut h1_blocks = Vec::with_capacity(q_blocks.len());
        for (block_idx, q) in &q_blocks {
            let h0 = (*q * s0) + sample_noise_ring(rng, params.binomial_k);
            let h1 = (*q * s1) + sample_noise_ring(rng, params.binomial_k);
            h0_blocks.push((*block_idx, h0));
            h1_blocks.push((*block_idx, h1));
        }
        branch_hints[b] = BranchHints {
            hint0_blocks_sparse: h0_blocks,
            hint1_blocks_sparse: h1_blocks,
        };

        // Key derivation: coeff₀(s * a_const) where a_const is the accepting-set element
        // embedded as a constant polynomial (scalar ring element).
        // coeff₀(s * a_const) = s[0] * a (since a_const = (a, 0, 0, ..., 0)).
        let a_fq: Fq = embed_f_to_fq(a);
        let k0 = reconcile_u64(fq_to_u64(&(s0.coeffs()[0] * a_fq)), params.recon_bits);
        let k1 = reconcile_u64(fq_to_u64(&(s1.coeffs()[0] * a_fq)), params.recon_bits);
        let key = derive_payload_key_bytes(&params.domain_label, &c_stmt, &coins, k0, k1);
        let mut nonce = [0u8; 12];
        rng.fill_bytes(&mut nonce);
        cts[b] = LockCiphertext { nonce, ct: xor_stream_encrypt(&key, &nonce, payload) };
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
