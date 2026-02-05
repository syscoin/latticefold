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
use chacha20poly1305::aead::{Aead, KeyInit};
use chacha20poly1305::{ChaCha20Poly1305, Key, Nonce};
use rand::RngCore;
use stark_rings::cyclotomic_ring::models::goldilocks::{Fq, RqPoly};
use stark_rings::PolyRing;
use std::collections::BTreeMap;
use std::sync::OnceLock;

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
    /// Shifted accepting set (two candidates), aligned with `cts`.
    ///
    /// If the underlying Theorem-4.3 accepting set is `A = {a0, a1}` and the public `x`
    /// contributes an additive offset `off`, we store:
    /// - `accepting_set = {a0 - off, a1 - off}`
    /// - `offset = off`
    ///
    /// This makes streaming decapsulation depend only on \(⟨q_π, π⟩\) while still allowing
    /// recovering the original Theorem-4.3 answer as `accepting_set[idx] + offset`.
    pub accepting_set: [F; 2],
    /// Public offset used to shift the accepting set.
    pub offset: F,
    /// Public input length (x length) for sanity checks.
    pub x_len: usize,
    /// Proof length π (field elements) for sanity checks.
    pub pi_len: usize,
    /// Total length of (x || π).
    pub len: usize,
    /// Hidden-query public coins.
    pub coins: dpp::theorem43::Theorem43Coins<F>,
    /// RLWE parameters.
    pub params: RingLweParams,
    /// Public lock vector blocks (packed), stored sparsely.
    ///
    /// Each entry is `(block_index, h_block)` where `block_index` refers to the packing of π
    /// into blocks of size `RqPoly::dimension()`.
    ///
    /// IMPORTANT: `block_index` is with respect to π only (not `(x||π)`), i.e. it indexes
    /// the packed blocks of `pi[0..pi_len]`.
    pub h_blocks_sparse: Vec<(usize, RqPoly)>,
    /// Two ciphertexts, aligned with `accepting_set`.
    ///
    /// Payload can be empty for "answer-only" usage; authenticity still pins the correct branch.
    pub cts: [DppLockCiphertext; 2],
}

#[derive(Clone, Debug)]
pub struct DppLockCiphertext {
    pub nonce: [u8; 12],
    pub ct: Vec<u8>,
}

fn sha256_32(chunks: &[&[u8]]) -> [u8; 32] {
    use sha2::Digest;
    let mut h = sha2::Sha256::new();
    for c in chunks {
        h.update(c);
    }
    h.finalize().into()
}

fn fq_to_8_le(x: &Fq) -> [u8; 8] {
    // Goldilocks field fits in u64; use canonical repr mod q.
    let bytes = x.into_bigint().to_bytes_le();
    let mut out = [0u8; 8];
    for (i, b) in bytes.iter().take(8).enumerate() {
        out[i] = *b;
    }
    out
}

fn derive_payload_key<F: PrimeField>(
    domain_label: &[u8; 32],
    c_stmt: &[F],
    coins: &dpp::theorem43::Theorem43Coins<F>,
    t_fq: &Fq,
) -> Key {
    // Bind key to statement + coins so the same `t` in another context doesn't reuse keys.
    // `t` is the recovered (masked) dot product in Fq (mod q).
    let mut coins_bytes = Vec::with_capacity(8 * 4);
    coins_bytes.extend_from_slice(&(coins.idx as u64).to_le_bytes());
    coins_bytes.extend_from_slice(&f_to_u64(&coins.lambda).to_le_bytes());
    coins_bytes.extend_from_slice(&f_to_u64(&coins.rho).to_le_bytes());
    coins_bytes.extend_from_slice(&f_to_u64(&coins.sigma).to_le_bytes());

    // c_stmt is already a statement digest encoding in-field; serialize as u64 limbs.
    let mut stmt_bytes = Vec::with_capacity(c_stmt.len() * 8);
    for f in c_stmt {
        stmt_bytes.extend_from_slice(&f_to_u64(f).to_le_bytes());
    }

    let t8 = fq_to_8_le(t_fq);
    let k = sha256_32(&[
        b"LFP_DPP_PAYLOAD_KEY_V1",
        domain_label,
        stmt_bytes.as_slice(),
        coins_bytes.as_slice(),
        &t8,
    ]);
    Key::from_slice(&k).to_owned()
}

pub struct RingLweDecapStreamState<'a, F: PrimeField> {
    lock: &'a RingLweLockArtifact<F>,
    d: usize,
    sparse: &'a [(usize, RqPoly)],
    sparse_pos: usize,
    block_idx: usize,
    filled: usize,
    coeffs: Vec<Fq>,
    t: Fq,
}

impl<'a, F: PrimeField> RingLweDecapStreamState<'a, F> {
    fn new(lock: &'a RingLweLockArtifact<F>, x: &[F]) -> Result<Self, String> {
        if x.len() != lock.x_len || x.len() + lock.pi_len != lock.len {
            return Err("ringlwe_decap_state: bad x length".to_string());
        }
        let d = RqPoly::dimension();
        Ok(Self {
            lock,
            d,
            sparse: lock.h_blocks_sparse.as_slice(),
            sparse_pos: 0,
            block_idx: 0,
            filled: 0,
            coeffs: Vec::with_capacity(d),
            t: Fq::ZERO,
        })
    }

    #[inline]
    fn maybe_process_full_block(&mut self) -> Result<(), String> {
        if self.coeffs.len() != self.d {
            return Ok(());
        }
        if self.sparse_pos < self.sparse.len() && self.sparse[self.sparse_pos].0 == self.block_idx {
            let pi_block = RqPoly::from(self.coeffs.clone());
            let h_i = &self.sparse[self.sparse_pos].1;
            self.t += coeff0_mul(h_i, &pi_block);
            self.sparse_pos += 1;
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

    /// Finish streaming and decrypt the payload share.
    ///
    /// Returns `(a_idx, plaintext)` where `a_idx ∈ {0,1}` indexes the lock's `accepting_set/cts`.
    pub fn finish_decrypt(mut self) -> Result<(usize, Vec<u8>), String> {
        if self.filled != self.lock.pi_len {
            return Err("ringlwe_decap_stream: bad π length".to_string());
        }
        if !self.coeffs.is_empty() {
            while self.coeffs.len() < self.d {
                self.coeffs.push(Fq::ZERO);
            }
            if self.sparse_pos < self.sparse.len() && self.sparse[self.sparse_pos].0 == self.block_idx {
                let pi_block = RqPoly::from(self.coeffs.clone());
                let h_i = &self.sparse[self.sparse_pos].1;
                self.t += coeff0_mul(h_i, &pi_block);
                self.sparse_pos += 1;
            }
            self.block_idx += 1;
            self.coeffs.clear();
        }
        let nblocks = (self.lock.pi_len + self.d - 1) / self.d;
        if self.block_idx != nblocks {
            return Err("ringlwe_decap_stream: internal block count mismatch".to_string());
        }
        if self.sparse_pos != self.sparse.len() {
            return Err("ringlwe_decap_stream: did not consume all sparse blocks".to_string());
        }

        let key = derive_payload_key(&self.lock.params.domain_label, &self.lock.c_stmt, &self.lock.coins, &self.t);
        let aead = ChaCha20Poly1305::new(&key);

        let mut ok: Option<(usize, Vec<u8>)> = None;
        for (i, ct) in self.lock.cts.iter().enumerate() {
            let nonce = Nonce::from_slice(&ct.nonce);
            if let Ok(pt) = aead.decrypt(nonce, ct.ct.as_slice()) {
                if ok.is_some() {
                    return Err("ringlwe_decap_stream: multiple ciphertexts authenticated".to_string());
                }
                ok = Some((i, pt));
            }
        }
        ok.ok_or_else(|| "ringlwe_decap_stream: no ciphertext authenticated".to_string())
    }

    /// Finish streaming and return the original Theorem-4.3 answer \(a\in\mathbb{F}\).
    ///
    /// This is the "answer-only" view of the payload lock: it decrypts the payload internally
    /// (to ensure authenticity / uniqueness of the tail) and then maps the authenticated index
    /// back to the unshifted accepting set element.
    pub fn finish(self) -> Result<F, String> {
        let lock = self.lock;
        let (idx, _pt) = self.finish_decrypt()?;
        Ok(lock.accepting_set[idx] + lock.offset)
    }
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

pub(crate) struct QueryBlockAccumulator {
    pi_len: usize,
    d: usize,
    dual: &'static [RqPoly],
    blocks: BTreeMap<usize, Vec<Fq>>,
}

impl QueryBlockAccumulator {
    pub(crate) fn new(pi_len: usize) -> Result<Self, String> {
        let d = RqPoly::dimension();
        let dual = dual_basis_coeff0_cached()?;
        Ok(Self {
            pi_len,
            d,
            dual,
            blocks: BTreeMap::new(),
        })
    }

    pub(crate) fn add_term<F: PrimeField>(&mut self, coeff: &F, idx: usize) -> Result<(), String> {
        if idx >= self.pi_len {
            return Err("q_pi index out of range".to_string());
        }
        let block = idx / self.d;
        let pos = idx % self.d;
        let qi = embed_f_to_fq(coeff);
        let di = self.dual[pos].coeffs();
        let row = self
            .blocks
            .entry(block)
            .or_insert_with(|| vec![Fq::ZERO; self.d]);
        for k in 0..self.d {
            row[k] += qi * di[k];
        }
        Ok(())
    }

    /// Finalize current sparse blocks and clear internal buffers for reuse.
    ///
    /// Returns only the blocks that have any nonzero query mass.
    pub(crate) fn into_sparse_blocks(&mut self) -> Vec<(usize, RqPoly)> {
        let blocks = std::mem::take(&mut self.blocks);
        blocks.into_iter().map(|(i, row)| (i, RqPoly::from(row))).collect()
    }
}

impl<F: PrimeField> RingLweLockArtifact<F> {
    /// Create an incremental streaming decapsulation state.
    ///
    /// This is the preferred API for end-to-end streaming: the prover can call
    /// `state.absorb_chunk(chunk)` as it emits proof chunks, without ever materializing π.
    pub fn decap_state<'a>(&'a self, x: &[F]) -> Result<RingLweDecapStreamState<'a, F>, String> {
        RingLweDecapStreamState::new(self, x)
    }

    // Canonical decapsulation is via `decap_state()` + `RingLweDecapStreamState::absorb_chunk()`,
    // then either:
    // - `finish()`        (answer-only)
    // - `finish_decrypt()` (recover payload share)
}

pub fn arm_ringlwe_lock<F: PrimeField>(
    c_stmt: Vec<F>,
    accepting_set_shifted: [F; 2],
    coins: dpp::theorem43::Theorem43Coins<F>,
    offset: F,
    x_len: usize,
    pi_len: usize,
    q_blocks: Vec<(usize, RqPoly)>,
    params: RingLweParams,
    payload: &[u8],
    rng: &mut impl RngCore,
) -> Result<RingLweLockArtifact<F>, String> {
    if params.binomial_k == 0 {
        return Err("arm_ringlwe_lock: binomial_k must be > 0 (secret mask s)".to_string());
    }

    // Secret scalar s (not published in the lock artifact).
    let mut tries = 0u32;
    let s = loop {
        tries += 1;
        let s_i16 = centered_binomial(rng, params.binomial_k);
        if s_i16 != 0 {
            break fq_from_i16(s_i16);
        }
        if tries > 1024 {
            return Err("arm_ringlwe_lock: failed to sample nonzero s".to_string());
        }
    };
    let mut h_blocks_sparse = Vec::with_capacity(q_blocks.len());
    for (block_idx, q) in &q_blocks {
        // For payload tests / bring-up, allow deterministic key material when `noise_bound==0`.
        // Until reconciliation is implemented, payload recovery is only meaningful with zero noise.
        let e = if params.noise_bound == 0 {
            RqPoly::from(vec![Fq::ZERO; RqPoly::dimension()])
        } else {
            sample_noise_poly(rng, params.binomial_k)
        };
        let mut coeffs = q.coeffs().to_vec();
        for c in &mut coeffs {
            *c *= s;
        }
        let q_scaled = RqPoly::from(coeffs);
        h_blocks_sparse.push((*block_idx, q_scaled + e));
    }

    // Encrypt payload under both candidate keys derived from (s * a_shifted).
    let mut cts: [DppLockCiphertext; 2] = [
        DppLockCiphertext {
            nonce: [0u8; 12],
            ct: Vec::new(),
        },
        DppLockCiphertext {
            nonce: [0u8; 12],
            ct: Vec::new(),
        },
    ];
    for (i, a) in accepting_set_shifted.iter().enumerate() {
        let a_fq = embed_f_to_fq(a);
        let t_i = s * a_fq;
        let key = derive_payload_key(&params.domain_label, &c_stmt, &coins, &t_i);
        let aead = ChaCha20Poly1305::new(&key);
        let mut nonce = [0u8; 12];
        rng.fill_bytes(&mut nonce);
        let ct = aead
            .encrypt(Nonce::from_slice(&nonce), payload)
            .map_err(|_| "arm_ringlwe_lock: encrypt failed".to_string())?;
        cts[i] = DppLockCiphertext { nonce, ct };
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
        h_blocks_sparse,
        cts,
    })
}

// (Payload is now integrated into `arm_ringlwe_lock`; there is no separate payload variant.)
