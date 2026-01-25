//! FLPCP for deterministic R1CS (dR1CS) — prototype (Section 4.1).
//!
//! The paper constructs efficient 3-query FLPCPs for dR1CS using multiplication codes.
//! This module will implement the RS-based instantiation (Corollary 4.8) and expose it
//! through the `BoundedFlpcp` interface so it can be packed into a DPP (Section 5.2).

use ark_ff::{BigInteger, Field, FftField, PrimeField};
use num_bigint::BigInt;
use num_traits::One;
use rand::RngCore;

use rayon::prelude::*;
use rayon::join;

use crate::packing::{BoundedFlpcp, BoundedFlpcpSparse, FlpcpPredicate};
use crate::rs::{barycentric_weights_consecutive, extrapolate_consecutive_next_block, lagrange_coeffs_at};
use crate::sparse::SparseVec;

/// Minimal NP-style dR1CS FLPCP API for lockable Theorem-4.3.
///
/// This allows swapping the RS backend for other multiplication-code backends
/// (e.g., AG-code instantiations from the paper) without changing callers.
pub trait Dr1csNpFlpcpSparseApi<F: PrimeField> {
    /// Number of public variables in `z`.
    fn n(&self) -> usize;
    /// Proof length `m` for π = (z_w || w).
    fn m(&self) -> usize;
    /// Codeword length ℓ (verifier coin index range).
    fn ell(&self) -> usize;
    /// Number of independent blocks/chunks in the instance (1 for non-chunked backends).
    fn blocks(&self) -> usize;
    /// Codeword length per block ℓ_local (equals `ell()` when `blocks()==1`).
    fn ell_local(&self) -> usize;
    /// Prover: given public `x` and private witness `z_w`, output π = (z_w || w).
    fn prove(&self, x: &[F], z_w: &[F]) -> Vec<F>;
    /// Deterministic sparse queries for fixed verifier coins.
    fn queries_for_coins_sparse(
        &self,
        idx: usize,
        lambda: F,
        x: &[F],
    ) -> Result<(Vec<SparseVec<F>>, FlpcpPredicate<F>), String>;
}

/// Multiplication-code interface for small-field FLPCPs (Cor. 4.9).
///
/// This abstracts the code family while preserving the same `(alpha,beta,gamma)` MulEq
/// interface used by the Theorem-4.3 wrapper.
pub trait MulCode<F: PrimeField> {
    /// Base message length k (dR1CS row count).
    fn dim_k(&self) -> usize;
    /// Square-code message length k*.
    fn dim_k_star(&self) -> usize;
    /// Codeword length ℓ.
    fn len_l(&self) -> usize;
    /// Linear functional for E(·)[idx] as coefficients over F^k.
    fn row_e(&self, idx: usize) -> Result<Vec<F>, String>;
    /// Linear functional for E*(·)[idx] as coefficients over the stored `w_eval`.
    fn row_e_star(&self, idx: usize) -> Result<Vec<F>, String>;
    /// Witness positions for the square-code evaluation witness `w_eval` (length k*).
    ///
    /// Semantics (Layout A / evaluation-witness):
    /// - The proof stores `w_eval[j] = E*(w_msg)[S[j]]` for `S = witness_positions_star()`.
    /// - `row_e_star(idx)` returns coefficients `λ*(idx)` such that
    ///   `E*(w_msg)[idx] = <λ*(idx), w_eval>`.
    /// - The **first `k = dim_k()` entries** of `w_eval` MUST correspond to the “low cube”
    ///   positions (i.e., indices where each coordinate is < k0 for tensor-RS), so that
    ///   `row_e_star(idx)[..k]` can be used for the `Cz - w_low || 0` term.
    fn witness_positions_star(&self) -> Result<Vec<usize>, String>;

    /// Stream coefficients for E(·)[idx] without allocating a full vector.
    fn row_e_stream(&self, idx: usize, f: &mut dyn FnMut(usize, F)) -> Result<(), String> {
        let row = self.row_e(idx)?;
        for (i, c) in row.into_iter().enumerate() {
            if !c.is_zero() {
                f(i, c);
            }
        }
        Ok(())
    }

    /// Stream coefficients for E*(·)[idx] without allocating a full vector.
    fn row_e_star_stream(&self, idx: usize, f: &mut dyn FnMut(usize, F)) -> Result<(), String> {
        let row = self.row_e_star(idx)?;
        for (i, c) in row.into_iter().enumerate() {
            if !c.is_zero() {
                f(i, c);
            }
        }
        Ok(())
    }
}

/// Tensor-product RS multiplication code over small fields (Layout A / evaluation-witness).
///
/// Parameters follow the GPT-Pro guidance:
/// - base_n = 256
/// - base_k = 48
/// - t = 3
///
    /// The square-code witness stores evaluations on the `(2k0-1)^t` low grid (Layout A).
#[derive(Clone, Debug)]
pub struct TensorRsMulCode<F: PrimeField> {
    base_n: usize,
    base_k: usize,
    rank: usize,
    points: Vec<F>,
    ws_k: Vec<F>,
    ws_star: Vec<F>,
    lam_k_u16: Vec<u16>,
    lam_star_u16: Vec<u16>,
}

impl<F: PrimeField> TensorRsMulCode<F> {
    pub fn new(base_k: usize, rank: usize) -> Result<Self, String> {
        let base_n = 256usize;
        if base_k == 0 || (2 * base_k - 1) > base_n {
            return Err("base_k must satisfy 1 <= 2*base_k-1 <= 256".to_string());
        }
        if rank == 0 {
            return Err("rank must be >= 1".to_string());
        }
        let points = (0..base_n)
            .map(|i| F::from((i as u64) + 1))
            .collect::<Vec<_>>();
        let ws_k = barycentric_weights_consecutive::<F>(base_k, 1);
        let side = 2 * base_k - 1;
        let ws_star = barycentric_weights_consecutive::<F>(side, 1);
        let points_u16 = (0..base_n).map(|i| (i as u16) + 1).collect::<Vec<_>>();
        let inv_u16 = inv_table_u16();
        let ws_k_u16 = barycentric_weights_consecutive_u16(base_k, &inv_u16);
        let ws_star_u16 = barycentric_weights_consecutive_u16(side, &inv_u16);
        let mut lam_k_u16 = Vec::with_capacity(base_n * base_k);
        let mut lam_star_u16 = Vec::with_capacity(base_n * side);
        for &alpha in points_u16.iter() {
            let lam_k = Self::lagrange_coeffs_u16(
                &points_u16[..base_k],
                &ws_k_u16,
                &inv_u16,
                alpha,
            );
            lam_k_u16.extend_from_slice(&lam_k);
            let lam_star = Self::lagrange_coeffs_u16(
                &points_u16[..side],
                &ws_star_u16,
                &inv_u16,
                alpha,
            );
            lam_star_u16.extend_from_slice(&lam_star);
        }
        Ok(Self {
            base_n,
            base_k,
            rank,
            points,
            ws_k,
            ws_star,
            lam_k_u16,
            lam_star_u16,
        })
    }

    fn pow_usize(base: usize, exp: usize) -> usize {
        let mut out = 1usize;
        for _ in 0..exp {
            out = out.saturating_mul(base);
        }
        out
    }

    fn decompose_index(&self, mut idx: usize) -> Vec<usize> {
        let mut coords = Vec::with_capacity(self.rank);
        for _ in 0..self.rank {
            coords.push(idx % self.base_n);
            idx /= self.base_n;
        }
        coords
    }

    fn compose_index(&self, coords: &[usize]) -> usize {
        let mut idx = 0usize;
        let mut stride = 1usize;
        for &c in coords {
            idx += c * stride;
            stride *= self.base_n;
        }
        idx
    }

    fn is_f257() -> bool {
        let bytes = F::MODULUS.to_bytes_le();
        let mut acc: u64 = 0;
        for (i, b) in bytes.iter().enumerate().take(8) {
            acc |= (*b as u64) << (8 * i);
        }
        acc == 257
    }

    fn lagrange_coeffs_u16(
        points: &[u16],
        weights: &[u16],
        inv: &[u16; 257],
        target: u16,
    ) -> Vec<u16> {
        let n = points.len();
        for (i, &p) in points.iter().enumerate() {
            if p == target {
                let mut out = vec![0u16; n];
                out[i] = 1;
                return out;
            }
        }
        let mut num = vec![0u16; n];
        let mut den = 0u16;
        for i in 0..n {
            let diff = sub_mod(points[i], target);
            let inv_diff = inv[diff as usize];
            let t = mul_mod(weights[i], inv_diff);
            num[i] = t;
            den = add_mod(den, t);
        }
        let den_inv = inv[den as usize];
        num.iter().map(|&t| mul_mod(t, den_inv)).collect()
    }
}

impl<F: PrimeField> MulCode<F> for TensorRsMulCode<F> {
    fn dim_k(&self) -> usize {
        Self::pow_usize(self.base_k, self.rank)
    }

    fn dim_k_star(&self) -> usize {
        Self::pow_usize(2 * self.base_k - 1, self.rank)
    }

    fn len_l(&self) -> usize {
        Self::pow_usize(self.base_n, self.rank)
    }

    fn row_e(&self, idx: usize) -> Result<Vec<F>, String> {
        let mut out = vec![F::ZERO; self.dim_k()];
        self.row_e_stream(idx, &mut |i, c| {
            out[i] = c;
        })?;
        Ok(out)
    }

    fn row_e_star(&self, idx: usize) -> Result<Vec<F>, String> {
        let mut out = vec![F::ZERO; self.dim_k_star()];
        self.row_e_star_stream(idx, &mut |i, c| {
            out[i] = c;
        })?;
        Ok(out)
    }

    fn row_e_stream(&self, idx: usize, f: &mut dyn FnMut(usize, F)) -> Result<(), String> {
        if idx >= self.len_l() {
            return Err("row_e_stream: idx out of range".to_string());
        }
        if Self::is_f257() {
            let coords = self.decompose_index(idx);
            let mut one_dim: Vec<&[u16]> = Vec::with_capacity(self.rank);
            for &c in coords.iter() {
                let start = c * self.base_k;
                let end = start + self.base_k;
                one_dim.push(&self.lam_k_u16[start..end]);
            }
            let total = self.dim_k();
            let base_k = self.base_k;
            for flat in 0..total {
                let mut coeff = 1u16;
                let mut tmp = flat;
                for d in 0..self.rank {
                    let id = tmp % base_k;
                    coeff = mul_mod(coeff, one_dim[d][id]);
                    tmp /= base_k;
                }
                if coeff != 0 {
                    f(flat, F::from(coeff as u64));
                }
            }
        } else {
            let coords = self.decompose_index(idx);
            let mut one_dim = Vec::with_capacity(self.rank);
            for &c in coords.iter() {
                let alpha = self.points[c];
                let lam = lagrange_coeffs_at(&self.points[..self.base_k], &self.ws_k, alpha);
                one_dim.push(lam);
            }
            let total = self.dim_k();
            let base_k = self.base_k;
            for flat in 0..total {
                let mut coeff = F::ONE;
                let mut tmp = flat;
                for d in 0..self.rank {
                    let id = tmp % base_k;
                    coeff *= one_dim[d][id];
                    tmp /= base_k;
                }
                if !coeff.is_zero() {
                    f(flat, coeff);
                }
            }
        }
        Ok(())
    }

    fn row_e_star_stream(&self, idx: usize, f: &mut dyn FnMut(usize, F)) -> Result<(), String> {
        if idx >= self.len_l() {
            return Err("row_e_star_stream: idx out of range".to_string());
        }
        if Self::is_f257() {
            let coords = self.decompose_index(idx);
            let mut one_dim: Vec<&[u16]> = Vec::with_capacity(self.rank);
            let side = 2 * self.base_k - 1;
            for &c in coords.iter() {
                let start = c * side;
                let end = start + side;
                one_dim.push(&self.lam_star_u16[start..end]);
            }
            let total = self.dim_k_star();
            let k = self.dim_k();
            let mut low = 0usize;
            let mut high = k;
            for flat in 0..total {
                let mut coeff = 1u16;
                let mut tmp = flat;
                let mut is_low = true;
                for d in 0..self.rank {
                    let id = tmp % side;
                    coeff = mul_mod(coeff, one_dim[d][id]);
                    if id >= self.base_k {
                        is_low = false;
                    }
                    tmp /= side;
                }
                if is_low {
                    if coeff != 0 {
                        f(low, F::from(coeff as u64));
                    }
                    low += 1;
                } else {
                    if coeff != 0 {
                        f(high, F::from(coeff as u64));
                    }
                    high += 1;
                }
            }
        } else {
            let coords = self.decompose_index(idx);
            let mut one_dim = Vec::with_capacity(self.rank);
            let side = 2 * self.base_k - 1;
            for &c in coords.iter() {
                let alpha = self.points[c];
                let lam = lagrange_coeffs_at(&self.points[..side], &self.ws_star, alpha);
                one_dim.push(lam);
            }
            let total = self.dim_k_star();
            let k = self.dim_k();
            let mut low = 0usize;
            let mut high = k;
            for flat in 0..total {
                let mut coeff = F::ONE;
                let mut tmp = flat;
                let mut is_low = true;
                for d in 0..self.rank {
                    let id = tmp % side;
                    coeff *= one_dim[d][id];
                    if id >= self.base_k {
                        is_low = false;
                    }
                    tmp /= side;
                }
                if is_low {
                    if !coeff.is_zero() {
                        f(low, coeff);
                    }
                    low += 1;
                } else {
                    if !coeff.is_zero() {
                        f(high, coeff);
                    }
                    high += 1;
                }
            }
        }
        Ok(())
    }

    fn witness_positions_star(&self) -> Result<Vec<usize>, String> {
        let k = self.dim_k();
        let k_star = self.dim_k_star();
        let side = 2 * self.base_k - 1;
        let mut out = Vec::with_capacity(k_star);

        // Low cube first (coords < k0 in all dimensions).
        let mut coords = vec![0usize; self.rank];
        loop {
            out.push(self.compose_index(&coords));
            let mut carry = 0usize;
            while carry < self.rank {
                coords[carry] += 1;
                if coords[carry] < self.base_k {
                    break;
                }
                coords[carry] = 0;
                carry += 1;
            }
            if carry == self.rank {
                break;
            }
        }
        if out.len() != k {
            return Err("witness_positions_star: low cube length mismatch".to_string());
        }

        // Then the rest of the (2k0)^t grid.
        let mut coords = vec![0usize; self.rank];
        loop {
            let mut is_low = true;
            for &c in coords.iter() {
                if c >= self.base_k {
                    is_low = false;
                    break;
                }
            }
            if !is_low {
                out.push(self.compose_index(&coords));
            }
            let mut carry = 0usize;
            while carry < self.rank {
                coords[carry] += 1;
                if coords[carry] < side {
                    break;
                }
                coords[carry] = 0;
                carry += 1;
            }
            if carry == self.rank {
                break;
            }
        }
        if out.len() != k_star {
            return Err("witness_positions_star: total length mismatch".to_string());
        }
        Ok(out)
    }
}

/// Multiplication-code FLPCP backend for NP dR1CS (3-query, MulEq predicate).
#[derive(Clone, Debug)]
pub struct MulCodeDr1csNpFlpcpSparse<F: PrimeField, C: MulCode<F>> {
    pub inst: Dr1csInstanceSparse<F>,
    /// Number of public variables in `z` (prefix length).
    pub l: usize,
    pub code: C,
}

/// Chunked multiplication-code FLPCP backend.
///
/// - Splits dR1CS constraints into fixed-size blocks of `k = code.dim_k()`.
/// - Proof layout: `π0 = (z_w || w_eval^(0) || ... || w_eval^(B-1))`.
/// - Verifier coins `idx` encode the block selector: `block_id = idx / ell`.
#[derive(Clone, Debug)]
pub struct ChunkedMulCodeDr1csNpFlpcpSparse<F: PrimeField, C: MulCode<F>> {
    pub blocks: Vec<Dr1csInstanceSparse<F>>,
    pub l: usize,
    pub code: C,
}

impl<F: PrimeField, C: MulCode<F> + Sync> ChunkedMulCodeDr1csNpFlpcpSparse<F, C> {
    pub fn new(blocks: Vec<Dr1csInstanceSparse<F>>, l: usize, code: C) -> Result<Self, String> {
        if blocks.is_empty() {
            return Err("no blocks".to_string());
        }
        let k = code.dim_k();
        for (i, inst) in blocks.iter().enumerate() {
            if inst.k() != k {
                return Err(format!("block {i}: k mismatch"));
            }
            if l > inst.n {
                return Err(format!("block {i}: bad public length"));
            }
        }
        Ok(Self { blocks, l, code })
    }

    fn ell(&self) -> usize {
        self.code.len_l()
    }

    fn k_star(&self) -> usize {
        self.code.dim_k_star()
    }

    pub(crate) fn compute_block_w_eval(
        &self,
        inst: &Dr1csInstanceSparse<F>,
        witness_pos: &[usize],
        x: &[F],
        z_w: &[F],
    ) -> Result<Vec<F>, String> {
        let k = inst.k();
        let k_star = self.k_star();
        if witness_pos.len() != k_star {
            return Err("witness positions length mismatch".to_string());
        }

        let (y_a, y_b) = join(
            || mat_vec_sparse_np(&inst.a, x, z_w, self.l),
            || mat_vec_sparse_np(&inst.b, x, z_w, self.l),
        );
        if y_a.len() != k || y_b.len() != k {
            return Err("bad mat-vec size".to_string());
        }

        let mut w_eval = vec![F::ZERO; k_star];
        if k_star >= 256 {
            w_eval
                .par_iter_mut()
                .enumerate()
                .try_for_each(|(j, out)| -> Result<(), String> {
                    let idx = witness_pos[j];
                    let mut ea = F::ZERO;
                    let mut eb = F::ZERO;
                    self.code.row_e_stream(idx, &mut |i, c| {
                        ea += c * y_a[i];
                        eb += c * y_b[i];
                    })?;
                    *out = ea * eb;
                    Ok(())
                })?;
        } else {
            for (j, &idx) in witness_pos.iter().enumerate() {
                let mut ea = F::ZERO;
                let mut eb = F::ZERO;
                self.code.row_e_stream(idx, &mut |i, c| {
                    ea += c * y_a[i];
                    eb += c * y_b[i];
                })?;
                w_eval[j] = ea * eb;
            }
        }
        Ok(w_eval)
    }

    fn decode_block_idx(&self, idx: usize) -> Result<(usize, usize), String> {
        let ell = self.ell();
        if ell == 0 {
            return Err("ell=0".to_string());
        }
        let block_id = idx / ell;
        let local_idx = idx % ell;
        if block_id >= self.blocks.len() {
            return Err("bad block id".to_string());
        }
        Ok((block_id, local_idx))
    }

    /// Streaming-friendly proof generation.
    ///
    /// This emits each `w_eval` chunk to `on_chunk` as it is computed, while still returning
    /// the full `π0 = (z_w || w^(0) || ... || w^(B-1))` for compatibility.
    pub fn prove_stream(
        &self,
        x: &[F],
        z_w: &[F],
        on_chunk: &mut dyn FnMut(usize, &[F]),
    ) -> Result<Vec<F>, String> {
        let z_w_len = self.blocks[0].n - self.l;
        if x.len() != self.l {
            return Err("bad public input length".to_string());
        }
        if z_w.len() != z_w_len {
            return Err("bad witness length".to_string());
        }
        let mut pi = Vec::with_capacity(self.m());
        pi.extend_from_slice(z_w);

        // This depends only on code parameters, so compute once and reuse across blocks.
        let witness_pos = self.code.witness_positions_star()?;
        if witness_pos.len() != self.k_star() {
            return Err("witness positions length mismatch".to_string());
        }

        for (b, inst) in self.blocks.iter().enumerate() {
            let w_eval = self.compute_block_w_eval(inst, &witness_pos, x, z_w)?;
            on_chunk(b, &w_eval);
            pi.extend_from_slice(&w_eval);
        }
        Ok(pi)
    }

    pub(crate) fn stream_queries_for_coins_sparse(
        &self,
        idx: usize,
        lambda: F,
        x: &[F],
        scratch: &mut Dr1csQueryScratch<F>,
        sink: &mut dyn QuerySink<F>,
    ) -> Result<(), String> {
        if x.len() != self.l {
            return Err("bad input".to_string());
        }
        let (block_id, local_idx) = self.decode_block_idx(idx)?;
        let inst = &self.blocks[block_id];
        let k = inst.k();
        let k_star = self.k_star();
        let z_w_len = inst.n - self.l;
        let base = self.l + z_w_len;
        let block_offset = base + (block_id * k_star);

        scratch.clear();
        let q_a_z = &mut scratch.q_a_z;
        let q_b_z = &mut scratch.q_b_z;
        let q_cx2_z = &mut scratch.q_cx2_z;

        self.code.row_e_stream(local_idx, &mut |i, c| {
            add_scaled_sparse_row_into_acc(q_a_z, &inst.a[i], c);
            add_scaled_sparse_row_into_acc(q_b_z, &inst.b[i], c);
        })?;

        self.code.row_e_star_stream(local_idx, &mut |j, c| {
            if j < k {
                add_scaled_sparse_row_into_acc(q_cx2_z, &inst.c[j], c);
            }
            let coeff = if j < k { c - (lambda * c) } else { c };
            if !coeff.is_zero() {
                sink.on_q3(coeff, block_offset + j);
            }
        })?;

        for (idx, c) in q_a_z.take_terms().into_iter() {
            let (is_pub, j) = map_z_index_to_v(self.l, idx);
            let v_idx = if is_pub { j } else { self.l + j };
            sink.on_q1(c, v_idx);
        }
        for (idx, c) in q_b_z.take_terms().into_iter() {
            let (is_pub, j) = map_z_index_to_v(self.l, idx);
            let v_idx = if is_pub { j } else { self.l + j };
            sink.on_q2(c, v_idx);
        }
        for (idx, c) in q_cx2_z.take_terms().into_iter() {
            let cc = lambda * c;
            if cc.is_zero() {
                continue;
            }
            let (is_pub, j) = map_z_index_to_v(self.l, idx);
            let v_idx = if is_pub { j } else { self.l + j };
            sink.on_q3(cc, v_idx);
        }

        Ok(())
    }
}

impl<F: PrimeField, C: MulCode<F> + Sync> Dr1csNpFlpcpSparseApi<F>
    for ChunkedMulCodeDr1csNpFlpcpSparse<F, C>
{
    fn n(&self) -> usize {
        self.l
    }

    fn m(&self) -> usize {
        let z_w_len = self.blocks[0].n - self.l;
        z_w_len + (self.k_star() * self.blocks.len())
    }

    fn ell(&self) -> usize {
        self.ell() * self.blocks.len()
    }

    fn blocks(&self) -> usize {
        self.blocks.len()
    }

    fn ell_local(&self) -> usize {
        self.ell()
    }

    fn prove(&self, x: &[F], z_w: &[F]) -> Vec<F> {
        let z_w_len = self.blocks[0].n - self.l;
        assert_eq!(x.len(), self.l);
        assert_eq!(z_w.len(), z_w_len);

        let mut pi = Vec::with_capacity(self.m());
        pi.extend_from_slice(z_w);

        // This depends only on code parameters, so compute once and reuse across blocks.
        let witness_pos = self
            .code
            .witness_positions_star()
            .expect("witness positions");
        assert_eq!(witness_pos.len(), self.k_star());

        for inst in self.blocks.iter() {
            let w_eval = self
                .compute_block_w_eval(inst, &witness_pos, x, z_w)
                .expect("block w_eval failed");
            pi.extend_from_slice(&w_eval);
        }
        pi
    }

    fn queries_for_coins_sparse(
        &self,
        idx: usize,
        lambda: F,
        x: &[F],
    ) -> Result<(Vec<SparseVec<F>>, FlpcpPredicate<F>), String> {
        struct VecSink<F: PrimeField> {
            q1: Vec<(F, usize)>,
            q2: Vec<(F, usize)>,
            q3: Vec<(F, usize)>,
        }
        impl<F: PrimeField> QuerySink<F> for VecSink<F> {
            fn on_q1(&mut self, coeff: F, idx: usize) {
                self.q1.push((coeff, idx));
            }
            fn on_q2(&mut self, coeff: F, idx: usize) {
                self.q2.push((coeff, idx));
            }
            fn on_q3(&mut self, coeff: F, idx: usize) {
                self.q3.push((coeff, idx));
            }
        }
        let mut sink = VecSink {
            q1: Vec::new(),
            q2: Vec::new(),
            q3: Vec::new(),
        };
        let mut scratch = Dr1csQueryScratch::<F>::new(self.blocks[0].n);
        self.stream_queries_for_coins_sparse(idx, lambda, x, &mut scratch, &mut sink)?;

        Ok((
            vec![
                SparseVec::new(sink.q1),
                SparseVec::new(sink.q2),
                SparseVec::new(sink.q3),
            ],
            FlpcpPredicate::MulEq,
        ))
    }
}

impl<F: PrimeField, C: MulCode<F> + Sync> MulCodeDr1csNpFlpcpSparse<F, C> {
    pub fn new(inst: Dr1csInstanceSparse<F>, l: usize, code: C) -> Result<Self, String> {
        if inst.k() == 0 {
            return Err("k=0".to_string());
        }
        if l > inst.n {
            return Err("bad public length".to_string());
        }
        if code.dim_k() != inst.k() {
            return Err("code k != dr1cs k".to_string());
        }
        if code.dim_k_star() < inst.k() {
            return Err("code k* < dr1cs k".to_string());
        }
        Ok(Self { inst, l, code })
    }

    pub fn ell(&self) -> usize {
        self.code.len_l()
    }

    pub fn prove_checked(&self, x: &[F], z_w: &[F]) -> Result<Vec<F>, String> {
        if x.len() != self.l {
            return Err("bad public input length".to_string());
        }
        if z_w.len() != self.inst.n - self.l {
            return Err("bad witness length".to_string());
        }
        let k = self.inst.k();
        let k_star = self.code.dim_k_star();
        let witness_pos = self.code.witness_positions_star()?;
        if witness_pos.len() != k_star {
            return Err("witness positions length mismatch".to_string());
        }
        for &idx in witness_pos.iter() {
            if idx >= self.code.len_l() {
                return Err("witness position out of range".to_string());
            }
        }

        let (y_a, y_b) = join(
            || mat_vec_sparse_np(&self.inst.a, x, z_w, self.l),
            || mat_vec_sparse_np(&self.inst.b, x, z_w, self.l),
        );
        if y_a.len() != k || y_b.len() != k {
            return Err("bad mat-vec size".to_string());
        }

        // Build w by evaluating E(Az) and E(Bz) at systematic E* positions.
        let mut w = Vec::with_capacity(k_star);
        for idx in witness_pos {
            let mut ea = F::ZERO;
            let mut eb = F::ZERO;
            self.code.row_e_stream(idx, &mut |i, c| {
                ea += c * y_a[i];
                eb += c * y_b[i];
            })?;
            w.push(ea * eb);
        }

        let mut pi = Vec::with_capacity(z_w.len() + w.len());
        pi.extend_from_slice(z_w);
        pi.extend_from_slice(&w);
        Ok(pi)
    }

    pub fn queries_for_coins_sparse(
        &self,
        idx: usize,
        lambda: F,
        x: &[F],
    ) -> Result<(Vec<SparseVec<F>>, FlpcpPredicate<F>), String> {
        struct VecSink<F: PrimeField> {
            q1: Vec<(F, usize)>,
            q2: Vec<(F, usize)>,
            q3: Vec<(F, usize)>,
        }
        impl<F: PrimeField> QuerySink<F> for VecSink<F> {
            fn on_q1(&mut self, coeff: F, idx: usize) {
                self.q1.push((coeff, idx));
            }
            fn on_q2(&mut self, coeff: F, idx: usize) {
                self.q2.push((coeff, idx));
            }
            fn on_q3(&mut self, coeff: F, idx: usize) {
                self.q3.push((coeff, idx));
            }
        }
        let mut sink = VecSink {
            q1: Vec::new(),
            q2: Vec::new(),
            q3: Vec::new(),
        };
        let mut scratch = Dr1csQueryScratch::<F>::new(self.inst.n);
        self.stream_queries_for_coins_sparse(idx, lambda, x, &mut scratch, &mut sink)?;

        Ok((
            vec![
                SparseVec::new(sink.q1),
                SparseVec::new(sink.q2),
                SparseVec::new(sink.q3),
            ],
            FlpcpPredicate::MulEq,
        ))
    }

    pub(crate) fn stream_queries_for_coins_sparse(
        &self,
        idx: usize,
        lambda: F,
        x: &[F],
        scratch: &mut Dr1csQueryScratch<F>,
        sink: &mut dyn QuerySink<F>,
    ) -> Result<(), String> {
        if x.len() != self.l || self.inst.k() == 0 {
            return Err("bad input".to_string());
        }
        if idx >= self.code.len_l() {
            return Err("bad coin idx".to_string());
        }
        let k = self.inst.k();
        let z_w_len = self.inst.n - self.l;
        let base = self.l + z_w_len;

        scratch.clear();
        let q_a_z = &mut scratch.q_a_z;
        let q_b_z = &mut scratch.q_b_z;
        let q_cx2_z = &mut scratch.q_cx2_z;

        // Stream lam_k coefficients.
        self.code.row_e_stream(idx, &mut |i, c| {
            add_scaled_sparse_row_into_acc(q_a_z, &self.inst.a[i], c);
            add_scaled_sparse_row_into_acc(q_b_z, &self.inst.b[i], c);
        })?;

        // Stream lam_star coefficients; use only low-cube (first k) entries for C.
        self.code.row_e_star_stream(idx, &mut |j, c| {
            if j < k {
                add_scaled_sparse_row_into_acc(q_cx2_z, &self.inst.c[j], c);
            }
            let coeff = if j < k { c - (lambda * c) } else { c };
            if !coeff.is_zero() {
                sink.on_q3(coeff, base + j);
            }
        })?;

        for (idx, c) in q_a_z.take_terms().into_iter() {
            let (is_pub, j) = map_z_index_to_v(self.l, idx);
            let v_idx = if is_pub { j } else { self.l + j };
            sink.on_q1(c, v_idx);
        }
        for (idx, c) in q_b_z.take_terms().into_iter() {
            let (is_pub, j) = map_z_index_to_v(self.l, idx);
            let v_idx = if is_pub { j } else { self.l + j };
            sink.on_q2(c, v_idx);
        }
        for (idx, c) in q_cx2_z.take_terms().into_iter() {
            let cc = lambda * c;
            if cc.is_zero() {
                continue;
            }
            let (is_pub, j) = map_z_index_to_v(self.l, idx);
            let v_idx = if is_pub { j } else { self.l + j };
            sink.on_q3(cc, v_idx);
        }

        Ok(())
    }
}

impl<F: PrimeField, C: MulCode<F> + Sync> Dr1csNpFlpcpSparseApi<F>
    for MulCodeDr1csNpFlpcpSparse<F, C>
{
    fn n(&self) -> usize {
        self.l
    }

    fn m(&self) -> usize {
        (self.inst.n - self.l) + self.code.dim_k_star()
    }

    fn ell(&self) -> usize {
        self.code.len_l()
    }

    fn blocks(&self) -> usize {
        1
    }

    fn ell_local(&self) -> usize {
        self.code.len_l()
    }

    fn prove(&self, x: &[F], z_w: &[F]) -> Vec<F> {
        MulCodeDr1csNpFlpcpSparse::prove_checked(self, x, z_w)
            .expect("mulcode prove failed")
    }

    fn queries_for_coins_sparse(
        &self,
        idx: usize,
        lambda: F,
        x: &[F],
    ) -> Result<(Vec<SparseVec<F>>, FlpcpPredicate<F>), String> {
        self.queries_for_coins_sparse(idx, lambda, x)
    }
}

/// Dense dR1CS instance: check (A x) ⊙ (B x) == (C x).
#[derive(Clone, Debug)]
pub struct Dr1csInstance<F: Field> {
    pub a: Vec<Vec<F>>, // k x n
    pub b: Vec<Vec<F>>, // k x n
    pub c: Vec<Vec<F>>, // k x n
}

impl<F: Field> Dr1csInstance<F> {
    pub fn k(&self) -> usize {
        self.a.len()
    }
    pub fn n(&self) -> usize {
        self.a.first().map(|r| r.len()).unwrap_or(0)
    }
}

/// RS-based 3-query FLPCP for dR1CS (Theorem 4.6 / Corollary 4.8).
///
/// Proof length is `m = 2k` field elements (the systematic prefix of the square code).
#[derive(Clone, Debug)]
pub struct RsDr1csFlpcp<F: PrimeField + FftField> {
    pub inst: Dr1csInstance<F>,
    /// Codeword length ℓ (must satisfy ℓ >= 2k and ℓ <= |F|).
    pub ell: usize,
    /// Evaluation points α_0..α_{ℓ-1}.
    pub points: Vec<F>,
    ws_k: Vec<F>,
    ws_2k: Vec<F>,
}

impl<F: PrimeField + FftField> RsDr1csFlpcp<F> {
    pub fn new(inst: Dr1csInstance<F>, ell: usize) -> Self {
        let k = inst.k();
        assert!(k > 0);
        assert!(ell >= 2 * k);
        // Simple deterministic points: 1,2,...,ell
        let points = (0..ell).map(|i| F::from((i as u64) + 1)).collect::<Vec<_>>();
        // O(k) weights for consecutive points.
        let ws_k = barycentric_weights_consecutive::<F>(k, 1);
        let ws_2k = barycentric_weights_consecutive::<F>(2 * k, 1);
        Self { inst, ell, points, ws_k, ws_2k }
    }

    /// Prover for the RS multiplication-code FLPCP:
    /// computes `w[i] = E(Ax)[i] * E(Bx)[i]` for i in [0..2k).
    pub fn prove(&self, x: &[F]) -> Vec<F> {
        let k = self.inst.k();
        assert_eq!(x.len(), self.inst.n());

        // Compute yA = A x, yB = B x (length k). Independent: run concurrently.
        let (y_a, y_b) = join(|| mat_vec(&self.inst.a, x), || mat_vec(&self.inst.b, x));

        // Fast systematic RS extrapolation on consecutive points:
        // - we already have f(1..k) as y_*
        // - compute f(k+1..2k) in O(k log k) via one convolution
        //
        // Memory note:
        // Extrapolation uses large NTT buffers when k is huge; running multiple extrapolations
        // concurrently can blow up RSS. For large sizes, compute tails sequentially.
        // Two extrapolations is usually an acceptable memory multiplier; keep them parallel.
        let (y_a_tail, y_b_tail) = join(
            || extrapolate_consecutive_next_block::<F>(&y_a),
            || extrapolate_consecutive_next_block::<F>(&y_b),
        );
        debug_assert_eq!(y_a_tail.len(), k);
        debug_assert_eq!(y_b_tail.len(), k);

        let mut w = Vec::with_capacity(2 * k);
        for i in 0..k {
            w.push(y_a[i] * y_b[i]);
        }
        for i in 0..k {
            w.push(y_a_tail[i] * y_b_tail[i]);
        }
        w
    }
}

impl<F: PrimeField + FftField> BoundedFlpcp<F> for RsDr1csFlpcp<F> {
    fn n(&self) -> usize {
        self.inst.n()
    }

    fn m(&self) -> usize {
        2 * self.inst.k()
    }

    fn k(&self) -> usize {
        3
    }

    fn bounds_b(&self) -> Vec<BigInt> {
        // Unbounded FLPCP: return the trivial bound b_i = (n+m) * ((p-1)/2)^2 as in Def 5.2 note.
        // This makes the packing modulus condition fail unless the field is enormous; the bounded
        // embedding (Theorem 5.6) is the intended way to get small b.
        let len = self.n() + self.m();
        let p = BigInt::from_bytes_le(num_bigint::Sign::Plus, &F::MODULUS.to_bytes_le());
        let half = (&p - BigInt::one()) / BigInt::from(2u64);
        let b = BigInt::from(len as u64) * &half * &half;
        vec![b.clone(), b.clone(), b]
    }

    fn sample_queries_and_predicate(
        &self,
        rng: &mut dyn RngCore,
        x: &[F],
    ) -> Result<(Vec<Vec<F>>, FlpcpPredicate<F>), String> {
        let k = self.inst.k();
        if x.len() != self.inst.n() || k == 0 {
            return Err("bad input".to_string());
        }
        // Prover's claimed proof π is w (length 2k).
        // Query schedule (Theorem 4.6):
        // Pick random i in [ℓ], random λ in F; query:
        //  a = E(Ax)[i]
        //  b = E(Bx)[i]
        //  c = E*(w)[i] + λ * E*(Cx - w≤k || 0)[i]
        // Accept if a*b == c.

        let idx = (rng.next_u64() as usize) % self.ell;
        let lambda = F::from(rng.next_u64());
        let alpha = self.points[idx];

        // Lagrange coeffs over first k points (for E(Ax), E(Bx), E(Cx)).
        let lam_k = lagrange_coeffs_at(&self.points[..k], &self.ws_k, alpha);
        // Lagrange coeffs over first 2k points (for E*(w), and for E*(Cx-w||0)).
        let lam_2k = lagrange_coeffs_at(&self.points[..2 * k], &self.ws_2k, alpha);

        // Build query vectors over v = (x || w) of length n + 2k.
        // q_a: coefficients on x to compute dot(lam_k, A x)
        let q_a = lin_combo_rows(&self.inst.a, &lam_k);
        let q_b = lin_combo_rows(&self.inst.b, &lam_k);
        let _q_cx = lin_combo_rows(&self.inst.c, &lam_k); // gives dot(lam_k, Cx)

        let n = self.inst.n();
        let mut q1 = vec![F::zero(); n + 2 * k];
        let mut q2 = vec![F::zero(); n + 2 * k];
        let mut q3 = vec![F::zero(); n + 2 * k];

        // q1 = (q_a || 0)
        q1[..n].copy_from_slice(&q_a);
        // q2 = (q_b || 0)
        q2[..n].copy_from_slice(&q_b);

        // q3 = ( ? || ? )
        // Part 1: E*(w)[i] is linear in w with coeffs lam_2k.
        // Part 2: E*(Cx - w≤k || 0)[i] = dot(lam_2k[..k], (Cx - w_prefix)) since last k entries are 0.
        // => equals dot(lam_2k[..k], Cx) - dot(lam_2k[..k], w_prefix).
        // Combine: c = dot(lam_2k, w) + λ*( dot(lam_2k[..k], Cx) - dot(lam_2k[..k], w_prefix) )
        // = λ * dot(lam_2k[..k], Cx)  + dot(lam_2k, w) - λ * dot(lam_2k[..k], w_prefix)
        // So w coeffs: for j<k: lam_2k[j] - λ*lam_2k[j]; for j>=k: lam_2k[j].
        // x coeffs: λ * (lin_combo_rows(C, lam_2k[..k])).
        // Compute lin combo using lam_2k[..k] over C rows.
        let q_cx2 = lin_combo_rows(&self.inst.c, &lam_2k[..k]);

        for j in 0..n {
            q3[j] = lambda * q_cx2[j];
        }
        for j in 0..2 * k {
            let coeff = if j < k {
                lam_2k[j] - (lambda * lam_2k[j])
            } else {
                lam_2k[j]
            };
            q3[n + j] = coeff;
        }

        Ok((vec![q1, q2, q3], FlpcpPredicate::MulEq))
    }
}

/// Sparse dR1CS instance: check (A x) ⊙ (B x) == (C x) with sparse rows.
#[derive(Clone, Debug)]
pub struct Dr1csInstanceSparse<F: PrimeField> {
    pub n: usize,
    pub a: Vec<SparseVec<F>>, // k rows
    pub b: Vec<SparseVec<F>>, // k rows
    pub c: Vec<SparseVec<F>>, // k rows
}

impl<F: PrimeField> Dr1csInstanceSparse<F> {
    pub fn k(&self) -> usize {
        self.a.len()
    }
}

/// RS-based 3-query FLPCP for sparse dR1CS rows.
///
/// Same protocol as `RsDr1csFlpcp`, but query vectors are emitted as `SparseVec` over v=(x||w).
#[derive(Clone, Debug)]
pub struct RsDr1csFlpcpSparse<F: PrimeField + FftField> {
    pub inst: Dr1csInstanceSparse<F>,
    pub ell: usize,
    pub points: Vec<F>,
    ws_k: Vec<F>,
    ws_2k: Vec<F>,
}

impl<F: PrimeField + FftField> RsDr1csFlpcpSparse<F> {
    pub fn new(inst: Dr1csInstanceSparse<F>, ell: usize) -> Self {
        let k = inst.k();
        assert!(k > 0);
        assert!(ell >= 2 * k);
        let points = (0..ell).map(|i| F::from((i as u64) + 1)).collect::<Vec<_>>();
        let ws_k = barycentric_weights_consecutive::<F>(k, 1);
        let ws_2k = barycentric_weights_consecutive::<F>(2 * k, 1);
        Self { inst, ell, points, ws_k, ws_2k }
    }

    pub fn prove(&self, x: &[F]) -> Vec<F> {
        let k = self.inst.k();
        assert_eq!(x.len(), self.inst.n);
        // A and B mat-vecs are independent: run concurrently.
        let (y_a, y_b) = join(
            || mat_vec_sparse(&self.inst.a, x),
            || mat_vec_sparse(&self.inst.b, x),
        );
        // Extrapolations are independent and often dominate: run concurrently.
        let (y_a_tail, y_b_tail) = join(
            || extrapolate_consecutive_next_block::<F>(&y_a),
            || extrapolate_consecutive_next_block::<F>(&y_b),
        );
        let mut w = Vec::with_capacity(2 * k);
        for i in 0..k {
            w.push(y_a[i] * y_b[i]);
        }
        for i in 0..k {
            w.push(y_a_tail[i] * y_b_tail[i]);
        }
        w
    }
}

impl<F: PrimeField + FftField> BoundedFlpcpSparse<F> for RsDr1csFlpcpSparse<F> {
    fn n(&self) -> usize {
        self.inst.n
    }

    fn m(&self) -> usize {
        2 * self.inst.k()
    }

    fn k(&self) -> usize {
        3
    }

    fn bounds_b(&self) -> Vec<BigInt> {
        // Same unbounded bound as dense prototype.
        let len = self.n() + self.m();
        let p = BigInt::from_bytes_le(num_bigint::Sign::Plus, &F::MODULUS.to_bytes_le());
        let half = (&p - BigInt::one()) / BigInt::from(2u64);
        let b = BigInt::from(len as u64) * &half * &half;
        vec![b.clone(), b.clone(), b]
    }

    fn sample_queries_and_predicate_sparse(
        &self,
        rng: &mut dyn RngCore,
        x: &[F],
    ) -> Result<(Vec<SparseVec<F>>, FlpcpPredicate<F>), String> {
        let k = self.inst.k();
        let n = self.inst.n;
        if x.len() != n || k == 0 {
            return Err("bad input".to_string());
        }

        let idx = (rng.next_u64() as usize) % self.ell;
        let lambda = F::from(rng.next_u64());
        let alpha = self.points[idx];

        let lam_k = lagrange_coeffs_at(&self.points[..k], &self.ws_k, alpha);
        let lam_2k = lagrange_coeffs_at(&self.points[..2 * k], &self.ws_2k, alpha);

        let q_a = lin_combo_rows_sparse(&self.inst.a, n, &lam_k);
        let q_b = lin_combo_rows_sparse(&self.inst.b, n, &lam_k);
        let q_cx2 = lin_combo_rows_sparse(&self.inst.c, n, &lam_2k[..k]);

        // q1 = (q_a || 0), q2 = (q_b || 0)
        let q1 = q_a;
        let q2 = q_b;

        // q3 has x-part and w-part (shifted by n).
        let mut q3_terms: Vec<(F, usize)> = Vec::new();
        for (c, idx) in q_cx2.terms.iter() {
            let cc = lambda * *c;
            if !cc.is_zero() {
                q3_terms.push((cc, *idx));
            }
        }
        for j in 0..2 * k {
            let coeff = if j < k {
                lam_2k[j] - (lambda * lam_2k[j])
            } else {
                lam_2k[j]
            };
            if !coeff.is_zero() {
                q3_terms.push((coeff, n + j));
            }
        }

        Ok((vec![q1, q2, SparseVec::new(q3_terms)], FlpcpPredicate::MulEq))
    }
}

/// RS-based 3-query FLPCP for NP-style dR1CS, where the witness vector is part of the proof.
///
/// - Public input `x` has length `l` (can be 0).
/// - Private witness `z_w` has length `n_total - l`.
/// - The FLPCP proof is `(z_w || w)`, where `w` is the systematic square-code prefix (length 2k).
///
/// This matches the WE/DPP use-case: the statement is public, and the witness is private.
#[derive(Clone, Debug)]
pub struct RsDr1csNpFlpcpSparse<F: PrimeField + FftField> {
    pub inst: Dr1csInstanceSparse<F>,
    /// Number of public variables in `z` (prefix length).
    pub l: usize,
    pub ell: usize,
    pub points: Vec<F>,
    ws_k: Vec<F>,
    ws_2k: Vec<F>,
}

/// Auxiliary RS codeword data for coin-form query evaluation.
///
/// This is NOT part of the published proof; it is prover-side cached data useful for answering
/// many locks (queries) efficiently after proving.
#[derive(Clone, Debug)]
pub struct RsNpCodewords<F: PrimeField> {
    pub y_a: Vec<F>,      // values at points 1..k
    pub y_a_tail: Vec<F>, // values at points k+1..2k
    pub y_b: Vec<F>,
    pub y_b_tail: Vec<F>,
    pub y_c: Vec<F>,
    pub w: Vec<F>, // length 2k: w[i]=y_a_full[i]*y_b_full[i]
}

impl<F: PrimeField + FftField> RsDr1csNpFlpcpSparse<F> {
    pub fn new(inst: Dr1csInstanceSparse<F>, l: usize, ell: usize) -> Self {
        let k = inst.k();
        assert!(k > 0);
        assert!(l <= inst.n);
        assert!(ell >= 2 * k);
        let points = (0..ell).map(|i| F::from((i as u64) + 1)).collect::<Vec<_>>();
        let ws_k = barycentric_weights_consecutive::<F>(k, 1);
        let ws_2k = barycentric_weights_consecutive::<F>(2 * k, 1);
        Self { inst, l, ell, points, ws_k, ws_2k }
    }

    /// Prover: given public `x` and private witness `z_w`, output π = (z_w || w).
    pub fn prove(&self, x: &[F], z_w: &[F]) -> Vec<F> {
        assert_eq!(x.len(), self.l);
        assert_eq!(z_w.len(), self.inst.n - self.l);
        let k = self.inst.k();
        // IMPORTANT: avoid materializing z = (x || z_w) for large instances (multi-million entries).
        // This copy can dominate runtime and look “single-threaded”.
        // A and B mat-vecs are independent: run concurrently.
        let (y_a, y_b) = join(
            || mat_vec_sparse_np(&self.inst.a, x, z_w, self.l),
            || mat_vec_sparse_np(&self.inst.b, x, z_w, self.l),
        );
        // Extrapolations are independent and often dominate: run concurrently.
        let (y_a_tail, y_b_tail) = join(
            || extrapolate_consecutive_next_block::<F>(&y_a),
            || extrapolate_consecutive_next_block::<F>(&y_b),
        );
        let mut w = Vec::with_capacity(2 * k);
        for i in 0..k {
            w.push(y_a[i] * y_b[i]);
        }
        for i in 0..k {
            w.push(y_a_tail[i] * y_b_tail[i]);
        }
        // Output π = (z_w || w) without an extra concat allocation.
        let mut pi = Vec::with_capacity(z_w.len() + w.len());
        pi.extend_from_slice(z_w);
        pi.extend_from_slice(&w);
        pi
    }

    /// Prover + cache: output π = (z_w || w) and also return the RS codeword values needed to
    /// answer verifier coins `(idx, λ)` in **O(1)** time (by indexing), assuming `ell == 2k`.
    pub fn prove_with_codewords(&self, x: &[F], z_w: &[F]) -> (Vec<F>, RsNpCodewords<F>) {
        assert_eq!(x.len(), self.l);
        assert_eq!(z_w.len(), self.inst.n - self.l);
        let k = self.inst.k();
        // Compute y_a, y_b, y_c concurrently.
        let (y_a, (y_b, y_c)) = join(
            || mat_vec_sparse_np(&self.inst.a, x, z_w, self.l),
            || join(
                || mat_vec_sparse_np(&self.inst.b, x, z_w, self.l),
                || mat_vec_sparse_np(&self.inst.c, x, z_w, self.l),
            ),
        );
        // Compute tails.
        //
        // Memory note:
        // For huge k, each extrapolation allocates large NTT buffers. Running 3 extrapolations
        // concurrently can easily exceed machine memory. For large sizes, compute tails sequentially.
        // Memory note:
        // Running 3 huge extrapolations concurrently can explode RSS. Keep bounded parallelism:
        // compute 2 in parallel, then the third.
        let (y_a_tail, y_b_tail) = join(
            || extrapolate_consecutive_next_block::<F>(&y_a),
            || extrapolate_consecutive_next_block::<F>(&y_b),
        );
        // NOTE: we intentionally do **not** compute `y_c_tail`.
        //
        // For the RS-FLPCP query used by the packed lock check, the tail-half query does not use the
        // `C`-part (see coin-form logic in `we_gate_arith::tests::test_large_trace`), so computing
        // `y_c_tail` is wasted work and can be extremely expensive at SP1 scale.

        // Build w (length 2k).
        let mut w = Vec::with_capacity(2 * k);
        for i in 0..k {
            w.push(y_a[i] * y_b[i]);
        }
        for i in 0..k {
            w.push(y_a_tail[i] * y_b_tail[i]);
        }

        // Proof π = (z_w || w).
        let mut pi = Vec::with_capacity(z_w.len() + w.len());
        pi.extend_from_slice(z_w);
        pi.extend_from_slice(&w);

        (
            pi,
            RsNpCodewords {
                y_a,
                y_a_tail,
                y_b,
                y_b_tail,
                y_c,
                w,
            },
        )
    }

    fn map_z_index_to_v(&self, idx: usize) -> (bool, usize) {
        // Returns (is_public, mapped_index)
        if idx < self.l {
            (true, idx)
        } else {
            (false, idx - self.l)
        }
    }

    /// Deterministically build the 3 sparse query vectors for fixed verifier coins `(idx, lambda)`.
    ///
    /// This is useful for higher-level “lockable” compositions that need to index the verifier’s
    /// randomness space, rather than sampling inside this method.
    pub fn queries_for_coins_sparse(
        &self,
        idx: usize,
        lambda: F,
        x: &[F],
    ) -> Result<(Vec<SparseVec<F>>, FlpcpPredicate<F>), String> {
        if x.len() != self.l || self.inst.k() == 0 {
            return Err("bad input".to_string());
        }
        if idx >= self.ell {
            return Err("bad coin idx".to_string());
        }

        let k = self.inst.k();
        let alpha = self.points[idx];

        let lam_k = lagrange_coeffs_at(&self.points[..k], &self.ws_k, alpha);
        let lam_2k = lagrange_coeffs_at(&self.points[..2 * k], &self.ws_2k, alpha);

        // Build sparse linear combos over z indices.
        let q_a_z = lin_combo_rows_sparse(&self.inst.a, self.inst.n, &lam_k);
        let q_b_z = lin_combo_rows_sparse(&self.inst.b, self.inst.n, &lam_k);
        let q_cx2_z = lin_combo_rows_sparse(&self.inst.c, self.inst.n, &lam_2k[..k]);

        // Map z-indices into v=(x || z_w || w) indices.
        let mut q1_terms: Vec<(F, usize)> = Vec::new();
        for (c, idx) in q_a_z.terms {
            let (is_pub, j) = self.map_z_index_to_v(idx);
            let v_idx = if is_pub { j } else { self.l + j };
            q1_terms.push((c, v_idx));
        }
        let mut q2_terms: Vec<(F, usize)> = Vec::new();
        for (c, idx) in q_b_z.terms {
            let (is_pub, j) = self.map_z_index_to_v(idx);
            let v_idx = if is_pub { j } else { self.l + j };
            q2_terms.push((c, v_idx));
        }

        // q3: x/witness part + w part (shift by l + (n-l) = n_total).
        let z_w_len = self.inst.n - self.l;
        let base = self.l + z_w_len;
        let mut q3_terms: Vec<(F, usize)> = Vec::new();
        for (c, idx) in q_cx2_z.terms.iter() {
            let cc = lambda * *c;
            if cc.is_zero() {
                continue;
            }
            let (is_pub, j) = self.map_z_index_to_v(*idx);
            let v_idx = if is_pub { j } else { self.l + j };
            q3_terms.push((cc, v_idx));
        }
        // w part indices are after z_w in the proof: position base + j
        for j in 0..2 * k {
            let coeff = if j < k {
                lam_2k[j] - (lambda * lam_2k[j])
            } else {
                lam_2k[j]
            };
            if !coeff.is_zero() {
                q3_terms.push((coeff, base + j));
            }
        }

        Ok((
            vec![
                SparseVec::new(q1_terms),
                SparseVec::new(q2_terms),
                SparseVec::new(q3_terms),
            ],
            FlpcpPredicate::MulEq,
        ))
    }
}

impl<F: PrimeField + FftField> Dr1csNpFlpcpSparseApi<F> for RsDr1csNpFlpcpSparse<F> {
    fn n(&self) -> usize {
        self.l
    }

    fn m(&self) -> usize {
        (self.inst.n - self.l) + 2 * self.inst.k()
    }

    fn ell(&self) -> usize {
        self.ell
    }

    fn blocks(&self) -> usize {
        1
    }

    fn ell_local(&self) -> usize {
        self.ell
    }

    fn prove(&self, x: &[F], z_w: &[F]) -> Vec<F> {
        RsDr1csNpFlpcpSparse::prove(self, x, z_w)
    }

    fn queries_for_coins_sparse(
        &self,
        idx: usize,
        lambda: F,
        x: &[F],
    ) -> Result<(Vec<SparseVec<F>>, FlpcpPredicate<F>), String> {
        RsDr1csNpFlpcpSparse::queries_for_coins_sparse(self, idx, lambda, x)
    }
}

impl<F: PrimeField + FftField> BoundedFlpcpSparse<F> for RsDr1csNpFlpcpSparse<F> {
    fn n(&self) -> usize {
        self.l
    }

    fn m(&self) -> usize {
        // proof = z_w (n-l) || w (2k)
        (self.inst.n - self.l) + 2 * self.inst.k()
    }

    fn k(&self) -> usize {
        3
    }

    fn bounds_b(&self) -> Vec<BigInt> {
        // Unbounded FLPCP: same trivial bound.
        let len = BoundedFlpcpSparse::n(self) + BoundedFlpcpSparse::m(self);
        let p = BigInt::from_bytes_le(num_bigint::Sign::Plus, &F::MODULUS.to_bytes_le());
        let half = (&p - BigInt::one()) / BigInt::from(2u64);
        let b = BigInt::from(len as u64) * &half * &half;
        vec![b.clone(), b.clone(), b]
    }

    fn sample_queries_and_predicate_sparse(
        &self,
        rng: &mut dyn RngCore,
        x: &[F],
    ) -> Result<(Vec<SparseVec<F>>, FlpcpPredicate<F>), String> {
        let idx = (rng.next_u64() as usize) % self.ell;
        let lambda = F::from(rng.next_u64());
        self.queries_for_coins_sparse(idx, lambda, x)
    }
}

fn mat_vec<F: Field>(m: &[Vec<F>], x: &[F]) -> Vec<F> {
    // Hot path for large dR1CS instances: each row dot is independent.
    if m.len() >= 256 {
        m.par_iter()
            .map(|row| row.iter().zip(x.iter()).fold(F::ZERO, |acc, (a, b)| acc + (*a * *b)))
            .collect()
    } else {
        m.iter()
            .map(|row| row.iter().zip(x.iter()).fold(F::ZERO, |acc, (a, b)| acc + (*a * *b)))
            .collect()
    }
}

fn mat_vec_sparse<F: PrimeField>(m: &[SparseVec<F>], x: &[F]) -> Vec<F> {
    if is_f257_field::<F>() {
        let x_u16 = x.iter().map(|v| f_to_u16(*v)).collect::<Vec<_>>();
        if m.len() >= 256 {
            m.par_iter()
                .map(|row| {
                    let mut acc = 0u16;
                    for (c, idx) in row.terms.iter().copied() {
                        let coeff = f_to_u16(c);
                        acc = add_mod(acc, mul_mod(coeff, x_u16[idx]));
                    }
                    F::from(acc as u64)
                })
                .collect()
        } else {
            m.iter()
                .map(|row| {
                    let mut acc = 0u16;
                    for (c, idx) in row.terms.iter().copied() {
                        let coeff = f_to_u16(c);
                        acc = add_mod(acc, mul_mod(coeff, x_u16[idx]));
                    }
                    F::from(acc as u64)
                })
                .collect()
        }
    } else if m.len() >= 256 {
        m.par_iter().map(|row| row.dot(x)).collect()
    } else {
        m.iter().map(|row| row.dot(x)).collect()
    }
}

fn mat_vec_sparse_np<F: PrimeField>(m: &[SparseVec<F>], x: &[F], z_w: &[F], l: usize) -> Vec<F> {
    debug_assert_eq!(x.len(), l);
    debug_assert_eq!(l + z_w.len(), m.first().map(|_| l + z_w.len()).unwrap_or(l + z_w.len()));
    if is_f257_field::<F>() {
        let x_u16 = x.iter().map(|v| f_to_u16(*v)).collect::<Vec<_>>();
        let z_u16 = z_w.iter().map(|v| f_to_u16(*v)).collect::<Vec<_>>();
        if m.len() >= 256 {
            m.par_iter()
                .map(|row| {
                    let mut acc = 0u16;
                    for (c, idx) in row.terms.iter().copied() {
                        let coeff = f_to_u16(c);
                        let v = if idx < l { x_u16[idx] } else { z_u16[idx - l] };
                        acc = add_mod(acc, mul_mod(coeff, v));
                    }
                    F::from(acc as u64)
                })
                .collect()
        } else {
            m.iter()
                .map(|row| {
                    let mut acc = 0u16;
                    for (c, idx) in row.terms.iter().copied() {
                        let coeff = f_to_u16(c);
                        let v = if idx < l { x_u16[idx] } else { z_u16[idx - l] };
                        acc = add_mod(acc, mul_mod(coeff, v));
                    }
                    F::from(acc as u64)
                })
                .collect()
        }
    } else if m.len() >= 256 {
        m.par_iter()
            .map(|row| {
                row.terms.iter().fold(F::ZERO, |acc, (c, idx)| {
                    let v = if *idx < l { x[*idx] } else { z_w[*idx - l] };
                    acc + (*c * v)
                })
            })
            .collect()
    } else {
        m.iter()
            .map(|row| {
                row.terms.iter().fold(F::ZERO, |acc, (c, idx)| {
                    let v = if *idx < l { x[*idx] } else { z_w[*idx - l] };
                    acc + (*c * v)
                })
            })
            .collect()
    }
}

fn map_z_index_to_v(l: usize, idx: usize) -> (bool, usize) {
    if idx < l {
        (true, idx)
    } else {
        (false, idx - l)
    }
}

struct SparseAccumulator<F: PrimeField> {
    vals: Vec<F>,
    touched: Vec<usize>,
}

impl<F: PrimeField> SparseAccumulator<F> {
    fn new(len: usize) -> Self {
        Self {
            vals: vec![F::ZERO; len],
            touched: Vec::new(),
        }
    }

    fn clear(&mut self) {
        for idx in self.touched.drain(..) {
            self.vals[idx] = F::ZERO;
        }
    }

    fn add_term(&mut self, idx: usize, coeff: F) {
        if coeff.is_zero() {
            return;
        }
        if self.vals[idx].is_zero() {
            self.touched.push(idx);
        }
        self.vals[idx] += coeff;
    }

    fn take_terms(&mut self) -> Vec<(usize, F)> {
        let mut out = Vec::with_capacity(self.touched.len());
        for idx in self.touched.drain(..) {
            let c = self.vals[idx];
            if !c.is_zero() {
                out.push((idx, c));
            }
            self.vals[idx] = F::ZERO;
        }
        out
    }
}

pub struct Dr1csQueryScratch<F: PrimeField> {
    q_a_z: SparseAccumulator<F>,
    q_b_z: SparseAccumulator<F>,
    q_cx2_z: SparseAccumulator<F>,
}

impl<F: PrimeField> Dr1csQueryScratch<F> {
    pub fn new(len: usize) -> Self {
        Self {
            q_a_z: SparseAccumulator::new(len),
            q_b_z: SparseAccumulator::new(len),
            q_cx2_z: SparseAccumulator::new(len),
        }
    }

    fn clear(&mut self) {
        self.q_a_z.clear();
        self.q_b_z.clear();
        self.q_cx2_z.clear();
    }
}

pub trait QuerySink<F: PrimeField> {
    fn on_q1(&mut self, coeff: F, idx: usize);
    fn on_q2(&mut self, coeff: F, idx: usize);
    fn on_q3(&mut self, coeff: F, idx: usize);
}

fn add_scaled_sparse_row_into_acc<F: PrimeField>(
    acc: &mut SparseAccumulator<F>,
    row: &SparseVec<F>,
    coeff: F,
) {
    if coeff.is_zero() {
        return;
    }
    for (c, idx) in row.terms.iter().copied() {
        acc.add_term(idx, c * coeff);
    }
}

fn add_mod(a: u16, b: u16) -> u16 {
    let mut s = a + b;
    if s >= 257 {
        s -= 257;
    }
    s
}

fn sub_mod(a: u16, b: u16) -> u16 {
    if a >= b {
        a - b
    } else {
        a + 257 - b
    }
}

fn mul_mod(a: u16, b: u16) -> u16 {
    let p = 257u32;
    ((a as u32 * b as u32) % p) as u16
}

fn is_f257_field<F: PrimeField>() -> bool {
    let bytes = F::MODULUS.to_bytes_le();
    let mut acc: u64 = 0;
    for (i, b) in bytes.iter().enumerate().take(8) {
        acc |= (*b as u64) << (8 * i);
    }
    acc == 257
}

fn f_to_u16<F: PrimeField>(v: F) -> u16 {
    let bytes = v.into_bigint().to_bytes_le();
    let mut acc: u16 = 0;
    if !bytes.is_empty() {
        acc |= bytes[0] as u16;
    }
    if bytes.len() > 1 {
        acc |= (bytes[1] as u16) << 8;
    }
    acc % 257
}

fn inv_table_u16() -> [u16; 257] {
    let mut inv = [0u16; 257];
    inv[0] = 0;
    for i in 1..257u16 {
        // Fermat: a^(p-2) mod p
        let mut res = 1u16;
        let mut base = i;
        let mut exp = 255u16;
        while exp > 0 {
            if (exp & 1) == 1 {
                res = mul_mod(res, base);
            }
            base = mul_mod(base, base);
            exp >>= 1;
        }
        inv[i as usize] = res;
    }
    inv
}

fn barycentric_weights_consecutive_u16(n: usize, inv: &[u16; 257]) -> Vec<u16> {
    if n == 0 {
        return Vec::new();
    }
    let mut fact = vec![1u16; n];
    for i in 1..n {
        fact[i] = mul_mod(fact[i - 1], (i as u16) % 257);
    }
    let mut inv_fact = vec![1u16; n];
    inv_fact[n - 1] = inv[fact[n - 1] as usize];
    for i in (1..n).rev() {
        inv_fact[i - 1] = mul_mod(inv_fact[i], (i as u16) % 257);
    }
    let mut w = vec![0u16; n];
    for i in 0..n {
        let mut wi = mul_mod(inv_fact[i], inv_fact[n - 1 - i]);
        if ((n - 1 - i) & 1) == 1 {
            wi = if wi == 0 { 0 } else { 257 - wi };
        }
        w[i] = wi;
    }
    w
}

fn lin_combo_rows<F: Field>(m: &[Vec<F>], coeffs: &[F]) -> Vec<F> {
    let n = m.first().map(|r| r.len()).unwrap_or(0);
    let mut out = vec![F::ZERO; n];
    // This is a matrix-transpose times vector. Parallelize over columns (independent outputs).
    if n >= 1024 && m.len() >= 64 {
        out.par_iter_mut().enumerate().for_each(|(j, out_j)| {
            let mut acc = F::ZERO;
            for (row, c) in m.iter().zip(coeffs.iter()) {
                acc += *c * row[j];
            }
            *out_j = acc;
        });
        return out;
    }
    for (row, c) in m.iter().zip(coeffs.iter()) {
        for j in 0..n {
            out[j] += *c * row[j];
        }
    }
    out
}

fn lin_combo_rows_sparse<F: PrimeField>(m: &[SparseVec<F>], n: usize, coeffs: &[F]) -> SparseVec<F> {
    debug_assert_eq!(m.len(), coeffs.len());
    let mut acc: std::collections::BTreeMap<usize, F> = std::collections::BTreeMap::new();
    for (row, c) in m.iter().zip(coeffs.iter()) {
        if c.is_zero() {
            continue;
        }
        for (aij, idx) in row.terms.iter() {
            debug_assert!(*idx < n);
            let entry = acc.entry(*idx).or_insert(F::ZERO);
            *entry += *c * *aij;
        }
    }
    let terms = acc
        .into_iter()
        .filter_map(|(idx, v)| if v.is_zero() { None } else { Some((v, idx)) })
        .collect::<Vec<_>>();
    SparseVec::new(terms)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_ff::{Fp64, MontBackend, MontConfig};
    use rand_chacha::ChaCha20Rng;
    use rand::SeedableRng;
    use crate::packing::{DppFromBoundedFlpcp, DppFromBoundedFlpcpSparse, PackingError, PackedDppParams};

    #[derive(MontConfig)]
    #[modulus = "10007"]
    #[generator = "5"]
    pub struct F10007Config;
    type F = Fp64<MontBackend<F10007Config, 1>>;

    #[test]
    fn test_rs_dr1cs_flpcp_honest_accepts() {
        // Tiny instance: k=2, n=2; choose A,B,C so that relation holds for x=[1,2].
        // Let A = I, B = I, C = I => require x_i^2 == x_i, which doesn't hold, so instead:
        // Set A=I, B=0, C=0 so relation holds trivially.
        let a = vec![vec![F::ONE, F::ZERO], vec![F::ZERO, F::ONE]];
        let b = vec![vec![F::ZERO, F::ZERO], vec![F::ZERO, F::ZERO]];
        let c = vec![vec![F::ZERO, F::ZERO], vec![F::ZERO, F::ZERO]];
        let inst = Dr1csInstance { a, b, c };
        let flpcp = RsDr1csFlpcp::new(inst, 8);
        let x = vec![F::from(3u64), F::from(5u64)];
        let w = flpcp.prove(&x);

        // Sample verifier queries and check predicate holds for honest answers.
        let mut rng = ChaCha20Rng::seed_from_u64(7);
        let (qs, pred) = flpcp.sample_queries_and_predicate(&mut rng, &x).unwrap();
        let v = [x.clone(), w.clone()].concat();
        let ans = qs
            .iter()
            .map(|q| q.iter().zip(v.iter()).fold(F::ZERO, |acc, (a, b)| acc + (*a * *b)))
            .collect::<Vec<_>>();
        assert!(pred.check(&ans));
    }

    #[test]
    fn test_packing_reports_modulus_too_small_for_unbounded_flpcp() {
        // Same as above, but try to pack directly: should usually fail modulus bound check because b is huge.
        let a = vec![vec![F::ONE, F::ZERO], vec![F::ZERO, F::ONE]];
        let b = vec![vec![F::ZERO, F::ZERO], vec![F::ZERO, F::ZERO]];
        let c = vec![vec![F::ZERO, F::ZERO], vec![F::ZERO, F::ZERO]];
        let inst = Dr1csInstance { a, b, c };
        let flpcp = RsDr1csFlpcp::new(inst, 8);
        let dpp = DppFromBoundedFlpcp::<F, _>::new(flpcp.clone(), crate::packing::PackedDppParams { ell: 32 });

        let x = vec![F::from(3u64), F::from(5u64)];
        let pi = flpcp.prove(&x);
        let mut rng = ChaCha20Rng::seed_from_u64(8);
        let res = dpp.verify(&mut rng, &x, &pi);
        assert!(matches!(res, Err(crate::packing::PackingError::ModulusTooSmall)));
    }

    #[test]
    fn test_rs_dr1cs_flpcp_sparse_honest_accepts() {
        let n = 2usize;
        let a = vec![
            SparseVec::new(vec![(F::ONE, 0)]),
            SparseVec::new(vec![(F::ONE, 1)]),
        ];
        let b = vec![SparseVec::default(), SparseVec::default()];
        let c = vec![SparseVec::default(), SparseVec::default()];
        let inst = Dr1csInstanceSparse { n, a, b, c };
        let flpcp = RsDr1csFlpcpSparse::new(inst, 8);

        let x = vec![F::from(3u64), F::from(5u64)];
        let w = flpcp.prove(&x);

        let mut rng = ChaCha20Rng::seed_from_u64(9);
        let (qs, pred) = flpcp.sample_queries_and_predicate_sparse(&mut rng, &x).unwrap();
        let v = [x.clone(), w.clone()].concat();
        let ans = qs.iter().map(|q| q.dot(&v)).collect::<Vec<_>>();
        assert!(pred.check(&ans));
    }

    #[test]
    fn test_sparse_packing_reports_modulus_too_small_for_unbounded_flpcp() {
        let n = 2usize;
        let a = vec![
            SparseVec::new(vec![(F::ONE, 0)]),
            SparseVec::new(vec![(F::ONE, 1)]),
        ];
        let b = vec![SparseVec::default(), SparseVec::default()];
        let c = vec![SparseVec::default(), SparseVec::default()];
        let inst = Dr1csInstanceSparse { n, a, b, c };
        let flpcp = RsDr1csFlpcpSparse::new(inst, 8);
        let dpp = DppFromBoundedFlpcpSparse::<F, _>::new(flpcp.clone(), PackedDppParams { ell: 32 });

        let x = vec![F::from(3u64), F::from(5u64)];
        let mut rng = ChaCha20Rng::seed_from_u64(10);
        let q = dpp.sample_query(&mut rng, &x);
        assert!(matches!(q, Err(PackingError::ModulusTooSmall)));
    }
}


