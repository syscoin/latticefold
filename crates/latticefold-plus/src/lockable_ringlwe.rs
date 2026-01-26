//! Ring-LWE style lock backend for Theorem-4.3 (WE arming).
//!
//! This module is intentionally lightweight: it provides a compact lock artifact that
//! preserves the hidden-query structure without enumerating a large accepting set.
//!
//! Notes:
//! - We currently implement *scalar* secret `s` with coefficient-wise noise, and we
//!   pack coefficient vectors into ring elements for bandwidth/computation benefits.
//! - We compute the dot product using **ring multiplication + coeff-0 extraction**
//!   via a precomputed dual basis for this cyclotomic ring.
//! - Security parameter choices are exposed via `RingLweParams` and must be reviewed.

use ark_ff::{BigInteger, Field, PrimeField};
use ark_std::Zero;
use rand::RngCore;
use stark_rings::cyclotomic_ring::models::goldilocks::{Fq, RqPoly};
use stark_rings::PolyRing;
use std::sync::OnceLock;

use dpp::sparse::SparseVec;

const GOLDILOCKS_Q: u64 = 0xffff_ffff_0000_0001;
const SMALL_P: i128 = 257;

#[derive(Clone, Debug)]
pub struct RingLweParams {
    /// Centered binomial parameter k (sum of k bits minus k bits).
    pub binomial_k: u32,
    /// Noise bound (absolute) in integer lift used for decoding.
    pub noise_bound: i128,
    /// Domain separation label for hashing/PRG expansion (optional).
    pub domain_label: [u8; 32],
}

impl Default for RingLweParams {
    fn default() -> Self {
        Self {
            binomial_k: 16,
            noise_bound: 1 << 20,
            domain_label: *b"LFP_RINGLWE_V1_00000000000000000",
        }
    }
}

#[derive(Clone, Debug)]
pub struct RingLweLockArtifact<F: PrimeField> {
    /// Statement-binding commitment (from Theorem43 arming).
    pub c_stmt: Vec<F>,
    /// Accepting set in the tiny field (e.g., {1,2}).
    pub accepting_set: [F; 2],
    /// Total length of (x || π).
    pub len: usize,
    /// Hidden-query public coins.
    pub coins: dpp::theorem43::Theorem43Coins<F>,
    /// RLWE parameters.
    pub params: RingLweParams,
    /// Public lock vector blocks (packed).
    pub h_blocks: Vec<RqPoly>,
    /// Public scalar s.
    pub s: Fq,
    /// Public offset = <q_x, x> + 1, embedded into Fq.
    pub offset: Fq,
}

fn f_to_u64<F: PrimeField>(f: &F) -> u64 {
    let bytes = f.into_bigint().to_bytes_le();
    let mut acc = 0u64;
    for (i, b) in bytes.iter().take(8).enumerate() {
        acc |= (*b as u64) << (8 * i);
    }
    acc
}

fn embed_f_to_fq<F: PrimeField>(f: &F) -> Fq {
    Fq::from(f_to_u64(f))
}

fn centered_binomial(rng: &mut impl RngCore, k: u32) -> i16 {
    let mut s: i16 = 0;
    for _ in 0..k {
        let b0 = (rng.next_u32() & 1) as i16;
        let b1 = (rng.next_u32() & 1) as i16;
        s += b0 - b1;
    }
    s
}

fn fq_from_i16(x: i16) -> Fq {
    if x >= 0 {
        Fq::from(x as u64)
    } else {
        Fq::ZERO - Fq::from((-x) as u64)
    }
}

fn sample_noise_poly(rng: &mut impl RngCore, k: u32) -> RqPoly {
    let d = RqPoly::dimension();
    let mut coeffs = Vec::with_capacity(d);
    for _ in 0..d {
        let n = centered_binomial(rng, k);
        coeffs.push(fq_from_i16(n));
    }
    RqPoly::from(coeffs)
}

fn pack_vec_to_ring_blocks<F: PrimeField>(v: &[F]) -> Vec<RqPoly> {
    let d = RqPoly::dimension();
    let mut out = Vec::new();
    let mut i = 0;
    while i < v.len() {
        let mut coeffs = Vec::with_capacity(d);
        for j in 0..d {
            if i + j < v.len() {
                coeffs.push(embed_f_to_fq(&v[i + j]));
            } else {
                coeffs.push(Fq::ZERO);
            }
        }
        out.push(RqPoly::from(coeffs));
        i += d;
    }
    out
}

fn coeff0_mul(a: &RqPoly, b: &RqPoly) -> Fq {
    let c = (*a) * (*b);
    c.coeffs()[0]
}

fn center_lift_fq(x: &Fq) -> i128 {
    let mut acc: u128 = 0;
    let bytes = x.into_bigint().to_bytes_le();
    for (i, b) in bytes.iter().take(8).enumerate() {
        acc |= (*b as u128) << (8 * i);
    }
    let q = GOLDILOCKS_Q as i128;
    let mut v = acc as i128;
    if v > q / 2 {
        v -= q;
    }
    v
}

fn centered_mod_257(x: i128) -> i128 {
    let mut r = x.rem_euclid(SMALL_P);
    if r > SMALL_P / 2 {
        r -= SMALL_P;
    }
    r
}

fn fq_mod_257(x: &Fq) -> i128 {
    let t = center_lift_fq(x);
    t.rem_euclid(SMALL_P)
}

fn invert_matrix(m: &[Vec<Fq>]) -> Result<Vec<Vec<Fq>>, String> {
    let n = m.len();
    let mut a = m.to_vec();
    let mut inv = vec![vec![Fq::ZERO; n]; n];
    for i in 0..n {
        inv[i][i] = Fq::ONE;
    }
    for i in 0..n {
        let mut pivot = i;
        while pivot < n && a[pivot][i].is_zero() {
            pivot += 1;
        }
        if pivot == n {
            return Err("dual basis matrix is singular".to_string());
        }
        if pivot != i {
            a.swap(i, pivot);
            inv.swap(i, pivot);
        }
        let inv_p = a[i][i].inverse().ok_or("pivot inverse missing")?;
        for j in 0..n {
            a[i][j] *= inv_p;
            inv[i][j] *= inv_p;
        }
        for r in 0..n {
            if r == i {
                continue;
            }
            let factor = a[r][i];
            if factor.is_zero() {
                continue;
            }
            let row_a = a[i].clone();
            let row_inv = inv[i].clone();
            for c in 0..n {
                a[r][c] -= factor * row_a[c];
                inv[r][c] -= factor * row_inv[c];
            }
        }
    }
    Ok(inv)
}

fn dual_basis_coeff0() -> Result<Vec<RqPoly>, String> {
    let d = RqPoly::dimension();
    let mut basis = Vec::with_capacity(d);
    for i in 0..d {
        let mut coeffs = vec![Fq::ZERO; d];
        coeffs[i] = Fq::ONE;
        basis.push(RqPoly::from(coeffs));
    }
    let mut m = vec![vec![Fq::ZERO; d]; d];
    for i in 0..d {
        for j in 0..d {
            m[i][j] = coeff0_mul(&basis[i], &basis[j]);
        }
    }
    let inv = invert_matrix(&m)?;
    let mut dual = Vec::with_capacity(d);
    for i in 0..d {
        let mut coeffs = vec![Fq::ZERO; d];
        for k in 0..d {
            coeffs[k] = inv[i][k];
        }
        dual.push(RqPoly::from(coeffs));
    }
    Ok(dual)
}

fn dual_basis_coeff0_cached() -> Result<&'static [RqPoly], String> {
    static CACHE: OnceLock<Result<Vec<RqPoly>, String>> = OnceLock::new();
    match CACHE.get_or_init(dual_basis_coeff0) {
        Ok(v) => Ok(v.as_slice()),
        Err(e) => Err(e.clone()),
    }
}

fn pack_query_to_dual_blocks_sparse<F: PrimeField>(
    sv: &SparseVec<F>,
    len: usize,
) -> Result<Vec<RqPoly>, String> {
    let d = RqPoly::dimension();
    let dual = dual_basis_coeff0_cached()?;
    let nblocks = (len + d - 1) / d;
    let mut coeffs = vec![vec![Fq::ZERO; d]; nblocks];

    for (c, idx) in sv.terms.iter().copied() {
        if idx >= len {
            continue;
        }
        let block = idx / d;
        let pos = idx % d;
        let qi = embed_f_to_fq(&c);
        let di = dual[pos].coeffs();
        for k in 0..d {
            coeffs[block][k] += qi * di[k];
        }
    }

    Ok(coeffs.into_iter().map(RqPoly::from).collect())
}

pub(crate) struct QueryBlockAccumulator {
    pi_len: usize,
    d: usize,
    dual: &'static [RqPoly],
    coeffs: Vec<Vec<Fq>>,
}

impl QueryBlockAccumulator {
    pub(crate) fn new(pi_len: usize) -> Result<Self, String> {
        let d = RqPoly::dimension();
        let dual = dual_basis_coeff0_cached()?;
        let nblocks = (pi_len + d - 1) / d;
        let coeffs = vec![vec![Fq::ZERO; d]; nblocks];
        Ok(Self { pi_len, d, dual, coeffs })
    }

    pub(crate) fn add_term<F: PrimeField>(&mut self, coeff: &F, idx: usize) -> Result<(), String> {
        if idx >= self.pi_len {
            return Err("q_pi index out of range".to_string());
        }
        let block = idx / self.d;
        let pos = idx % self.d;
        let qi = embed_f_to_fq(coeff);
        let di = self.dual[pos].coeffs();
        for k in 0..self.d {
            self.coeffs[block][k] += qi * di[k];
        }
        Ok(())
    }

    /// Finalize current blocks and clear internal buffers for reuse.
    ///
    /// This is intentionally `&mut self` (not consuming) so callers can reuse the allocated
    /// buffers across many repetitions `R` when `pi_len` is fixed.
    pub(crate) fn into_blocks(&mut self) -> Vec<RqPoly> {
        let out = self
            .coeffs
            .iter()
            .map(|row| RqPoly::from(row.clone()))
            .collect::<Vec<_>>();
        for row in &mut self.coeffs {
            row.fill(Fq::ZERO);
        }
        out
    }
}

impl<F: PrimeField> RingLweLockArtifact<F> {
    /// Attempt to decapsulate and recover the tiny-field answer.
    pub fn decap_answer(&self, x: &[F], pi: &[F]) -> Result<F, String> {
        if x.len() + pi.len() != self.len {
            return Err("decap_answer: bad (x||pi) length".to_string());
        }
        let pi_blocks = pack_vec_to_ring_blocks(pi);
        let mut t = Fq::ZERO;
        for (h_i, pi_i) in self.h_blocks.iter().zip(pi_blocks.iter()) {
            t += coeff0_mul(h_i, pi_i);
        }
        // Add s * offset so that target is s * (q·pi + offset).
        t += self.s * self.offset;

        // Check against s * accepting_set under small noise, modulo p=257.
        let t_center = center_lift_fq(&t);
        let s_mod = fq_mod_257(&self.s);
        for a in &self.accepting_set {
            let a_i = f_to_u64(a) as i128;
            let target = (s_mod * a_i).rem_euclid(SMALL_P);
            let diff = t_center - target;
            let rem = centered_mod_257(diff);
            if rem.abs() <= self.params.noise_bound {
                return Ok(*a);
            }
        }
        Err("decap_answer: not in accepting set (within noise bound)".to_string())
    }

    /// Streaming decapsulation over proof chunks.
    ///
    /// This avoids materializing the full `pi` vector in memory.
    pub fn decap_answer_stream<I>(&self, x: &[F], pi_len: usize, chunks: I) -> Result<F, String>
    where
        I: IntoIterator<Item = Vec<F>>,
    {
        if x.len() + pi_len != self.len {
            return Err("decap_answer_stream: bad (x||pi) length".to_string());
        }
        let d = RqPoly::dimension();
        let mut t = Fq::ZERO;
        let mut block_idx = 0usize;
        let mut filled = 0usize;
        let mut coeffs: Vec<Fq> = Vec::with_capacity(d);

        for chunk in chunks {
            for v in chunk {
                coeffs.push(embed_f_to_fq(&v));
                filled += 1;
                if coeffs.len() == d {
                    let pi_block = RqPoly::from(coeffs);
                    if block_idx >= self.h_blocks.len() {
                        return Err("decap_answer_stream: too many blocks".to_string());
                    }
                    t += coeff0_mul(&self.h_blocks[block_idx], &pi_block);
                    block_idx += 1;
                    coeffs = Vec::with_capacity(d);
                }
            }
        }
        if filled != pi_len {
            return Err("decap_answer_stream: bad pi_len".to_string());
        }
        if !coeffs.is_empty() {
            while coeffs.len() < d {
                coeffs.push(Fq::ZERO);
            }
            if block_idx >= self.h_blocks.len() {
                return Err("decap_answer_stream: too many blocks".to_string());
            }
            let pi_block = RqPoly::from(coeffs);
            t += coeff0_mul(&self.h_blocks[block_idx], &pi_block);
            block_idx += 1;
        }
        if block_idx != self.h_blocks.len() {
            return Err("decap_answer_stream: bad block count".to_string());
        }

        t += self.s * self.offset;

        let t_center = center_lift_fq(&t);
        let s_mod = fq_mod_257(&self.s);
        for a in &self.accepting_set {
            let a_i = f_to_u64(a) as i128;
            let target = (s_mod * a_i).rem_euclid(SMALL_P);
            let diff = t_center - target;
            let rem = centered_mod_257(diff);
            if rem.abs() <= self.params.noise_bound {
                return Ok(*a);
            }
        }
        Err("decap_answer_stream: not in accepting set (within noise bound)".to_string())
    }
}

pub fn arm_ringlwe_lock<F: PrimeField>(
    c_stmt: Vec<F>,
    accepting_set: [F; 2],
    coins: dpp::theorem43::Theorem43Coins<F>,
    offset_f: F,
    x_len: usize,
    pi_len: usize,
    q_blocks: Vec<RqPoly>,
    params: RingLweParams,
    rng: &mut impl RngCore,
) -> Result<RingLweLockArtifact<F>, String> {
    let offset = embed_f_to_fq(&offset_f);

    // Sample public scalar s and noise.
    let s = if params.binomial_k == 0 {
        Fq::ONE
    } else {
        let mut tries = 0u32;
        loop {
            tries += 1;
            let s_i16 = centered_binomial(rng, params.binomial_k);
            if s_i16 != 0 {
                break fq_from_i16(s_i16);
            }
            if tries > 1024 {
                return Err("arm_ringlwe_lock: failed to sample nonzero s".to_string());
            }
        }
    };
    let mut h_blocks = Vec::with_capacity(q_blocks.len());
    for q in &q_blocks {
        let e = sample_noise_poly(rng, params.binomial_k);
        let mut coeffs = q.coeffs().to_vec();
        for c in &mut coeffs {
            *c *= s;
        }
        let q_scaled = RqPoly::from(coeffs);
        h_blocks.push(q_scaled + e);
    }

    Ok(RingLweLockArtifact {
        c_stmt,
        accepting_set,
        len: x_len + pi_len,
        coins,
        params,
        h_blocks,
        s,
        offset,
    })
}
