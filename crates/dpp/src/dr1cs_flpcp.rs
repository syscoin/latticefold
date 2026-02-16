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
use std::collections::HashMap;

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
    /// Total number of variables in `z = (x || z_w)`.
    fn n_total(&self) -> usize;
    /// Private witness length `|z_w|`.
    fn z_w_len(&self) -> usize;
    /// Proof length `m` for π = (z_w || w).
    fn m(&self) -> usize;
    /// Codeword length ℓ (verifier coin index range).
    fn ell(&self) -> usize;
    /// Number of independent blocks/chunks in the instance (1 for non-chunked backends).
    fn blocks(&self) -> usize;
    /// Codeword length per block ℓ_local (equals `ell()` when `blocks()==1`).
    fn ell_local(&self) -> usize;
    /// Square-code witness block length `k*` (length of each `w_eval`).
    fn k_star(&self) -> usize;
    /// Witness positions for `w_eval` (Layout A).
    fn witness_positions_star(&self) -> Result<Vec<usize>, String>;

    /// Stream `w_eval` blocks in order without materializing the full proof.
    ///
    /// Implementations must call `on_block(block_id, w_eval)` for each block_id in `0..blocks()`,
    /// where `w_eval.len() == k_star()`.
    ///
    /// `on_block_hook` is an optional, thread-safe per-block hook intended for heavy computations
    /// that should run inside the backend's internal parallelism (e.g. batched dense q3 dots).
    fn stream_w_eval_blocks(
        &self,
        witness_pos: &[usize],
        x: &[F],
        z_w: &[F],
        x_u16: Option<&[u16]>,
        z_u16: Option<&[u16]>,
        // Optional thread-safe hook that runs per block. Implementations may also provide
        // `w_eval_u16` (same Layout-A block reduced mod 257) to avoid expensive `F -> u16`
        // conversions in downstream hot loops.
        on_block_hook: Option<&(dyn Fn(usize, &[F], &[u16]) -> Result<(), String> + Sync)>,
        on_block: &mut dyn FnMut(usize, &[F]),
    ) -> Result<(), String>;

    /// Stream verifier queries for fixed coins without allocating full query vectors.
    fn stream_queries_for_coins_sparse(
        &self,
        idx: usize,
        lambda: F,
        x: &[F],
        scratch: &mut Dr1csQueryScratch<F>,
        sink: &mut dyn QuerySink<F>,
    ) -> Result<(), String>;

    /// Compute the `q3` contribution against the streamed `w_eval[block]` slice.
    ///
    /// This is a **performance hook** to avoid materializing dense q3 witness terms:
    /// for a given `(idx,lambda)`, `q3` contains a dense component over the `w_eval` coordinates
    /// of the selected block. Implementations with structure (e.g. tensor-RS) should override
    /// this to compute the dot product without allocating/query-term emission.
    ///
    /// Default implementation is correct but may be slow: it reuses `stream_queries_for_coins_sparse`
    /// and accumulates only the q3 terms that land in the current block's `w_eval` slice.
    fn dot_q3_w_eval(
        &self,
        idx: usize,
        lambda: F,
        x: &[F],
        scratch: &mut Dr1csQueryScratch<F>,
        w_eval: &[F],
        w_eval_u16: &[u16],
    ) -> Result<F, String> {
        let blocks = self.blocks();
        let ell_local = self.ell_local();
        if blocks == 0 || ell_local == 0 {
            return Err("dot_q3_w_eval: invalid blocks/ell_local".to_string());
        }
        let block_id = idx / ell_local;
        if block_id >= blocks {
            return Err("dot_q3_w_eval: bad block id".to_string());
        }
        let k_star = self.k_star();
        if w_eval.len() != k_star {
            return Err("dot_q3_w_eval: bad w_eval length".to_string());
        }
        if w_eval_u16.len() != k_star {
            return Err("dot_q3_w_eval: bad w_eval_u16 length".to_string());
        }
        let base = self.n() + self.z_w_len();
        let w_base = base + block_id.saturating_mul(k_star);

        struct DotSink<'a, F: PrimeField> {
            w_base: usize,
            w_eval: &'a [F],
            acc: F,
        }
        impl<'a, F: PrimeField> QuerySink<F> for DotSink<'a, F> {
            fn on_q1(&mut self, _coeff: F, _idx: usize) {}
            fn on_q2(&mut self, _coeff: F, _idx: usize) {}
            fn on_q3(&mut self, coeff: F, idx: usize) {
                if idx < self.w_base {
                    return;
                }
                let j = idx - self.w_base;
                if j < self.w_eval.len() {
                    self.acc += coeff * self.w_eval[j];
                }
            }
        }

        let mut sink = DotSink {
            w_base,
            w_eval,
            acc: F::ZERO,
        };
        self.stream_queries_for_coins_sparse(idx, lambda, x, scratch, &mut sink)?;
        Ok(sink.acc)
    }

    /// Batched version of `dot_q3_w_eval` for a fixed streamed `w_eval[block]`.
    ///
    /// Default implementation calls `dot_q3_w_eval` per coin. Structured backends should
    /// override this to amortize the `w_eval` traversal across many coins (e.g. all hits for a block).
    fn dot_q3_w_eval_many(
        &self,
        idxs: &[usize],
        lambdas: &[F],
        x: &[F],
        scratch: &mut Dr1csQueryScratch<F>,
        w_eval: &[F],
        w_eval_u16: &[u16],
        out: &mut [F],
    ) -> Result<(), String> {
        if idxs.len() != lambdas.len() || idxs.len() != out.len() {
            return Err("dot_q3_w_eval_many: length mismatch".to_string());
        }
        for i in 0..idxs.len() {
            out[i] = self.dot_q3_w_eval(idxs[i], lambdas[i], x, scratch, w_eval, w_eval_u16)?;
        }
        Ok(())
    }
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

    /// Evaluate `E(y)[positions[j]]` for a batch of indices without allocating `row_e`.
    ///
    /// Canonical path for prover hot loop: callers precompute `positions` once (typically
    /// `witness_positions_star()`), then reuse it across blocks.
    ///
    /// Default implementation is correct but may be slow for large batches; code families
    /// with structure (e.g. tensor RS over F257) should override this.
    fn eval_e_at_positions(&self, positions: &[usize], y: &[F]) -> Result<Vec<F>, String>
    where
        Self: Sync,
    {
        let k = self.dim_k();
        if y.len() != k {
            return Err("eval_e_at_positions: bad y length".to_string());
        }
        if positions.len() >= 256 {
            let out = positions
                .par_iter()
                .map(|&idx| -> Result<F, String> {
                    let mut acc = F::ZERO;
                    self.row_e_stream(idx, &mut |i, c| {
                        acc += c * y[i];
                    })?;
                    Ok(acc)
                })
                .collect::<Result<Vec<_>, _>>()?;
            Ok(out)
        } else {
            let mut out = Vec::with_capacity(positions.len());
            for &idx in positions {
                let mut acc = F::ZERO;
                self.row_e_stream(idx, &mut |i, c| {
                    acc += c * y[i];
                })?;
                out.push(acc);
            }
            Ok(out)
        }
    }

    /// Evaluate `E(y)[positions[j]]` into caller-provided storage (no allocation).
    ///
    /// Streaming provers should prefer this API and reuse `out` across blocks to
    /// avoid heap churn.
    ///
    /// Default implementation is correct but may allocate internally via
    /// `eval_e_at_positions` and then copy.
    fn eval_e_at_positions_into(
        &self,
        positions: &[usize],
        y: &[F],
        out: &mut [F],
    ) -> Result<(), String>
    where
        Self: Sync,
    {
        if out.len() != positions.len() {
            return Err("eval_e_at_positions_into: out len != positions len".to_string());
        }
        let tmp = self.eval_e_at_positions(positions, y)?;
        if tmp.len() != out.len() {
            return Err("eval_e_at_positions_into: bad eval length".to_string());
        }
        out.copy_from_slice(&tmp);
        Ok(())
    }

    /// F257-oriented hot path: evaluate `E(y)[positions[j]]` into `u16` residues mod 257.
    ///
    /// This exists to keep the `w_eval` pipeline in `u16` and avoid materializing two huge
    /// `Vec<F>` buffers (and doing field multiplications) on tiny-field instances.
    ///
    /// Default implementation is intentionally **not provided**.
    ///
    /// This API is meant as an explicit, F257-specific fast path. Implementations must override
    /// it (e.g. tensor-RS rank=3 over F257). Callers that need a generic path should use
    /// `eval_e_at_positions_into` instead.
    fn eval_e_at_positions_into_u16(
        &self,
        positions: &[usize],
        y_u16: &[u16],
        out_u16: &mut [u16],
    ) -> Result<(), String>
    where
        Self: Sync,
    {
        let _ = (positions, y_u16, out_u16);
        Err("eval_e_at_positions_into_u16: not implemented for this MulCode; override this method for F257 fast path"
            .to_string())
    }

    /// Fast F257-only helper: for each `idx` in `idxs`, compute the dot products
    /// \(\langle row_e_star(idx), w_eval \rangle\) and \(\langle row_e_star_low(idx), w_eval_low \rangle\) mod 257,
    /// where `w_eval` is in **Layout A** (the output layout of `witness_positions_star()` / `stream_w_eval_blocks()`).
    ///
    /// Implementations with structure (e.g. tensor-RS) should override this to batch many dots
    /// while traversing `w_eval` only once (bandwidth win when `|idxs|` is large, e.g. hits-per-block).
    fn dot_row_e_star_many_mod257_u16(
        &self,
        idxs: &[usize],
        w_eval_u16: &[u16],
        out_star_u16: &mut [u16],
        out_low_u16: &mut [u16],
    ) -> Result<(), String>
    where
        Self: Sync,
    {
        let _ = (idxs, w_eval_u16, out_star_u16, out_low_u16);
        Err("dot_row_e_star_many_mod257_u16: not implemented for this MulCode; override this method for F257 tensor fast path"
            .to_string())
    }

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

#[derive(Default)]
struct TensorRsF257Rank3Scratch {
    t0: Vec<u16>,
    t1: Vec<u16>,
    out_grid: Vec<u16>,
    // Batched q3 dot scratch (Layout A traversal).
    coords: Vec<usize>,
    // lam01 transposed: contiguous in batch for each (c1,c0) position.
    lam01_t: Vec<u16>,
    lam2_vec: Vec<u16>,
    acc_star_u16: Vec<u16>,
    acc_low_u16: Vec<u16>,
    tmp_star_u32: Vec<u32>,
    tmp_low_u32: Vec<u32>,
}

thread_local! {
    static TENSOR_RS_F257_R3_SCRATCH: std::cell::RefCell<TensorRsF257Rank3Scratch> =
        std::cell::RefCell::new(TensorRsF257Rank3Scratch::default());
}

#[inline]
fn reduce_mod257_u32(x: u32) -> u16 {
    // Fast reduction mod 257 using 256 ≡ -1 (mod 257).
    //
    // For x < 2^32, write x = b0 + 256 b1 + 256^2 b2 + 256^3 b3, with 0<=bi<=255.
    // Then x ≡ b0 - b1 + b2 - b3 (mod 257).
    let b0 = (x & 0xFF) as i32;
    let b1 = ((x >> 8) & 0xFF) as i32;
    let b2 = ((x >> 16) & 0xFF) as i32;
    let b3 = ((x >> 24) & 0xFF) as i32;
    let mut r = b0 - b1 + b2 - b3;
    // r is in [-510, 510], so at most two adjustments are needed.
    if r < 0 {
        r += 257;
        if r < 0 {
            r += 257;
        }
    }
    if r >= 257 {
        r -= 257;
        if r >= 257 {
            r -= 257;
        }
    }
    debug_assert!((0..257).contains(&r));
    r as u16
}

#[inline]
pub fn reduce_mod257_u64(x: u64) -> u16 {
    // Same idea as `reduce_mod257_u32`, extended to u64.
    let b0 = (x & 0xFF) as i32;
    let b1 = ((x >> 8) & 0xFF) as i32;
    let b2 = ((x >> 16) & 0xFF) as i32;
    let b3 = ((x >> 24) & 0xFF) as i32;
    let b4 = ((x >> 32) & 0xFF) as i32;
    let b5 = ((x >> 40) & 0xFF) as i32;
    let b6 = ((x >> 48) & 0xFF) as i32;
    let b7 = ((x >> 56) & 0xFF) as i32;
    let mut r = b0 - b1 + b2 - b3 + b4 - b5 + b6 - b7;
    // r is in [-1020, 1020], so at most 4 adjustments are needed.
    if r < 0 {
        r += 257;
    }
    if r < 0 {
        r += 257;
    }
    if r < 0 {
        r += 257;
    }
    if r < 0 {
        r += 257;
    }
    if r >= 257 {
        r -= 257;
    }
    if r >= 257 {
        r -= 257;
    }
    if r >= 257 {
        r -= 257;
    }
    if r >= 257 {
        r -= 257;
    }
    debug_assert!((0..257).contains(&r));
    r as u16
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

    /// Fast F257-only helper: compute \(\langle \lambda_k(idx), values \rangle\) mod 257,
    /// where `values` has length `k = dim_k()`.
    ///
    /// This avoids allocating the full `row_e(idx)` vector and is intended for batched
    /// arming/decap paths that work purely mod 257.
    pub fn dot_row_e_u16(&self, idx: usize, values: &[u16]) -> Result<u16, String> {
        if idx >= self.len_l() {
            return Err("dot_row_e_u16: idx out of range".to_string());
        }
        if !Self::is_f257() {
            return Err("dot_row_e_u16: only supported for F257 fast path".to_string());
        }
        let k = self.dim_k();
        if values.len() != k {
            return Err("dot_row_e_u16: bad values length".to_string());
        }
        let coords = self.decompose_index(idx);
        let mut one_dim: Vec<&[u16]> = Vec::with_capacity(self.rank);
        for &c in coords.iter() {
            let start = c * self.base_k;
            let end = start + self.base_k;
            one_dim.push(&self.lam_k_u16[start..end]);
        }
        let base_k = self.base_k;
        let mut acc: u32 = 0;
        for flat in 0..k {
            let mut coeff = 1u16;
            let mut tmp = flat;
            for d in 0..self.rank {
                let id = tmp % base_k;
                coeff = mul_mod(coeff, one_dim[d][id]);
                tmp /= base_k;
            }
            if coeff != 0 {
                acc = acc.wrapping_add((coeff as u32) * (values[flat] as u32));
            }
        }
        Ok(reduce_mod257_u32(acc))
    }

    /// Fast F257-only helper: compute \(\langle \lambda^*_\\text{low}(idx), values \rangle\) mod 257,
    /// where `values` has length `k = dim_k()`.
    ///
    /// This uses **only the low-cube** portion of `row_e_star(idx)` (the first `k` entries),
    /// and avoids iterating over the full `k* = dim_k_star()` side-cube.
    pub fn dot_row_e_star_low_u16(&self, idx: usize, values: &[u16]) -> Result<u16, String> {
        if idx >= self.len_l() {
            return Err("dot_row_e_star_low_u16: idx out of range".to_string());
        }
        if !Self::is_f257() {
            return Err("dot_row_e_star_low_u16: only supported for F257 fast path".to_string());
        }
        let k = self.dim_k();
        if values.len() != k {
            return Err("dot_row_e_star_low_u16: bad values length".to_string());
        }
        let coords = self.decompose_index(idx);
        let side = 2 * self.base_k - 1;
        let mut one_dim_full: Vec<&[u16]> = Vec::with_capacity(self.rank);
        for &c in coords.iter() {
            let start = c * side;
            let end = start + side;
            one_dim_full.push(&self.lam_star_u16[start..end]);
        }
        let base_k = self.base_k;
        let mut acc: u32 = 0;
        for flat in 0..k {
            let mut coeff = 1u16;
            let mut tmp = flat;
            for d in 0..self.rank {
                let id = tmp % base_k;
                coeff = mul_mod(coeff, one_dim_full[d][id]);
                tmp /= base_k;
            }
            if coeff != 0 {
                acc = acc.wrapping_add((coeff as u32) * (values[flat] as u32));
            }
        }
        Ok(reduce_mod257_u32(acc))
    }

    /// Fast F257-only helper: count the number of nonzero coefficients in `row_e(idx)`.
    ///
    /// For the tensor-product RS code, `row_e(idx)` is the Kronecker product of `rank` one-dimensional
    /// coefficient vectors of length `base_k`. A coefficient is nonzero iff all per-dimension factors
    /// are nonzero, so the nnz is the product of per-dimension nnz counts.
    pub fn nnz_row_e_u16(&self, idx: usize) -> Result<usize, String> {
        if idx >= self.len_l() {
            return Err("nnz_row_e_u16: idx out of range".to_string());
        }
        if !Self::is_f257() {
            return Err("nnz_row_e_u16: only supported for F257 fast path".to_string());
        }
        let coords = self.decompose_index(idx);
        let mut prod: usize = 1;
        for &c in coords.iter() {
            let start = c * self.base_k;
            let end = start + self.base_k;
            let slice = &self.lam_k_u16[start..end];
            let mut nz: usize = 0;
            for &v in slice {
                if v != 0 {
                    nz += 1;
                }
            }
            prod = prod.saturating_mul(nz);
            if prod == 0 {
                return Ok(0);
            }
        }
        Ok(prod)
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

    fn eval_e_at_positions(&self, positions: &[usize], y: &[F]) -> Result<Vec<F>, String>
    where
        Self: Sync,
    {
        if Self::is_f257() && self.rank == 3 {
            let k_star = self.dim_k_star();
            if positions.len() != k_star {
                return Err("eval_e_at_positions: bad positions length".to_string());
            }
            let mut out = vec![F::ZERO; k_star];
            self.eval_e_at_positions_into(positions, y, &mut out)?;
            return Ok(out);
        }

        // Fallback for non-F257 or non-rank-3: correct but potentially slow.
        let k = self.dim_k();
        if y.len() != k {
            return Err("eval_e_at_positions: bad y length".to_string());
        }
        if positions.len() >= 256 {
            let out = positions
                .par_iter()
                .map(|&idx| -> Result<F, String> {
                    let mut acc = F::ZERO;
                    self.row_e_stream(idx, &mut |i, c| {
                        acc += c * y[i];
                    })?;
                    Ok(acc)
                })
                .collect::<Result<Vec<_>, _>>()?;
            Ok(out)
        } else {
            let mut out = Vec::with_capacity(positions.len());
            for &idx in positions {
                let mut acc = F::ZERO;
                self.row_e_stream(idx, &mut |i, c| {
                    acc += c * y[i];
                })?;
                out.push(acc);
            }
            Ok(out)
        }
    }

    fn eval_e_at_positions_into(
        &self,
        positions: &[usize],
        y: &[F],
        out: &mut [F],
    ) -> Result<(), String>
    where
        Self: Sync,
    {
        if Self::is_f257() && self.rank == 3 {
            let k = self.dim_k();
            if y.len() != k {
                return Err("eval_e_at_positions_into: bad y length".to_string());
            }
            let mut y_u16: Vec<u16> = Vec::with_capacity(k);
            for &v in y.iter() {
                y_u16.push(f_to_u16(v));
            }
            let mut out_u16: Vec<u16> = vec![0u16; positions.len()];
            self.eval_e_at_positions_into_u16(positions, &y_u16, &mut out_u16)?;
            if out.len() != out_u16.len() {
                return Err("eval_e_at_positions_into: out len mismatch".to_string());
            }
            let lut: Vec<F> = (0u16..=256u16).map(|d| F::from(d as u64)).collect();
            for (o, &u) in out.iter_mut().zip(out_u16.iter()) {
                *o = lut[u as usize];
            }
            Ok(())
        } else {
            let tmp = <Self as MulCode<F>>::eval_e_at_positions(self, positions, y)?;
            if tmp.len() != out.len() {
                return Err("eval_e_at_positions_into: bad eval length".to_string());
            }
            out.copy_from_slice(&tmp);
            Ok(())
        }
    }

    fn eval_e_at_positions_into_u16(
        &self,
        positions: &[usize],
        y_u16: &[u16],
        out_u16: &mut [u16],
    ) -> Result<(), String>
    where
        Self: Sync,
    {
        if Self::is_f257() && self.rank == 3 {
            // This fast path assumes the canonical Layout A evaluation order, i.e.
            // `positions == witness_positions_star()`.
            if let Ok(expected) = self.witness_positions_star() {
                if positions != expected.as_slice() {
                    return Err(
                        "eval_e_at_positions_into_u16: F257 rank=3 fast path requires positions == witness_positions_star()"
                            .to_string(),
                    );
                }
            }

            let base_k = self.base_k;
            let side = 2 * base_k - 1;
            if side > self.base_n {
                return Err("eval_e_at_positions_into_u16: side out of range".to_string());
            }
            let k = self.dim_k();
            if y_u16.len() != k {
                return Err("eval_e_at_positions_into_u16: bad y length".to_string());
            }
            let k_star = self.dim_k_star();
            if positions.len() != k_star || out_u16.len() != k_star {
                return Err("eval_e_at_positions_into_u16: bad positions/out length".to_string());
            }

            let stride_y1 = base_k;
            let stride_y2 = base_k * base_k;
            let stride_g1 = side;
            let stride_g2 = side * side;

            TENSOR_RS_F257_R3_SCRATCH.with(|cell| {
                let mut s = cell.borrow_mut();
                let s: &mut TensorRsF257Rank3Scratch = &mut *s;

                let t0_len = side * base_k * base_k;
                if s.t0.len() != t0_len {
                    s.t0.resize(t0_len, 0u16);
                }
                // Pass 1: interpolate along dim0.
                let y_u16: &[u16] = y_u16;
                let lam_k_u16: &[u16] = self.lam_k_u16.as_slice();
                let t0: &mut [u16] = &mut s.t0;
                t0.par_chunks_mut(side)
                    .enumerate()
                    .for_each(|(chunk_idx, out_row)| {
                        let i1 = chunk_idx % base_k;
                        let i2 = chunk_idx / base_k;
                        let base_y = i1 * stride_y1 + i2 * stride_y2;
                        for c0 in 0..side {
                            let lam0 = &lam_k_u16[c0 * base_k..(c0 + 1) * base_k];
                            let mut acc: u32 = 0;
                            for i0 in 0..base_k {
                                acc += (lam0[i0] as u32) * (y_u16[base_y + i0] as u32);
                            }
                            out_row[c0] = reduce_mod257_u32(acc);
                        }
                    });

                let t1_len = side * side * base_k;
                if s.t1.len() != t1_len {
                    s.t1.resize(t1_len, 0u16);
                }
                // Pass 2: interpolate along dim1.
                let t0: &[u16] = &s.t0;
                let t1: &mut [u16] = &mut s.t1;
                t1.par_chunks_mut(side)
                    .enumerate()
                    .for_each(|(chunk_idx, out_row)| {
                        let c0 = chunk_idx % side;
                        let i2 = chunk_idx / side;
                        for c1 in 0..side {
                            let lam1 = &lam_k_u16[c1 * base_k..(c1 + 1) * base_k];
                            let mut acc: u32 = 0;
                            for i1 in 0..base_k {
                                let v = t0[c0 + side * (i1 + base_k * i2)];
                                acc += (lam1[i1] as u32) * (v as u32);
                            }
                            out_row[c1] = reduce_mod257_u32(acc);
                        }
                    });

                let grid_len = side * side * side;
                if s.out_grid.len() != grid_len {
                    s.out_grid.resize(grid_len, 0u16);
                }
                // Pass 3: interpolate along dim2.
                let t1: &[u16] = &s.t1;
                let out_grid: &mut [u16] = &mut s.out_grid;
                out_grid
                    .par_chunks_mut(side * side)
                    .enumerate()
                    .for_each(|(c2, out_plane)| {
                        let lam2 = &lam_k_u16[c2 * base_k..(c2 + 1) * base_k];
                        for c1 in 0..side {
                            let row = &mut out_plane[c1 * side..(c1 + 1) * side];
                            for c0 in 0..side {
                                let mut acc: u32 = 0;
                                for i2 in 0..base_k {
                                    let v = t1[c1 + side * (c0 + side * i2)];
                                    acc += (lam2[i2] as u32) * (v as u32);
                                }
                                row[c0] = reduce_mod257_u32(acc);
                            }
                        }
                    });

                // Emit in the exact order of `witness_positions_star()` (Layout A), but as u16.
                let mut w = 0usize;
                for i2 in 0..base_k {
                    for i1 in 0..base_k {
                        for i0 in 0..base_k {
                            let g = i0 + i1 * stride_g1 + i2 * stride_g2;
                            out_u16[w] = s.out_grid[g];
                            w += 1;
                        }
                    }
                }
                if w != k {
                    return Err("eval_e_at_positions_into_u16: low cube length mismatch".to_string());
                }
                for c2 in 0..side {
                    for c1 in 0..side {
                        for c0 in 0..side {
                            if c0 < base_k && c1 < base_k && c2 < base_k {
                                continue;
                            }
                            let g = c0 + c1 * stride_g1 + c2 * stride_g2;
                            out_u16[w] = s.out_grid[g];
                            w += 1;
                        }
                    }
                }
                if w != k_star {
                    return Err("eval_e_at_positions_into_u16: total length mismatch".to_string());
                }
                Ok(())
            })
        } else {
            <Self as MulCode<F>>::eval_e_at_positions_into_u16(self, positions, y_u16, out_u16)
        }
    }

    fn dot_row_e_star_many_mod257_u16(
        &self,
        idxs: &[usize],
        w_eval_u16: &[u16],
        out_star_u16: &mut [u16],
        out_low_u16: &mut [u16],
    ) -> Result<(), String>
    where
        Self: Sync,
    {
        if !Self::is_f257() || self.rank != 3 {
            return <Self as MulCode<F>>::dot_row_e_star_many_mod257_u16(
                self,
                idxs,
                w_eval_u16,
                out_star_u16,
                out_low_u16,
            );
        }
        let batch = idxs.len();
        if out_star_u16.len() != batch || out_low_u16.len() != batch {
            return Err("dot_row_e_star_many_mod257_u16: output length mismatch".to_string());
        }
        if batch == 0 {
            return Ok(());
        }
        let k_star = self.dim_k_star();
        if w_eval_u16.len() != k_star {
            return Err("dot_row_e_star_many_mod257_u16: bad w_eval_u16 length".to_string());
        }
        let ell = self.len_l();
        for &idx in idxs {
            if idx >= ell {
                return Err("dot_row_e_star_many_mod257_u16: idx out of range".to_string());
            }
        }

        let base_k = self.base_k;
        let side = 2 * base_k - 1;
        let side2 = side * side;
        let k = self.dim_k();

        #[inline]
        fn mul_mod257_u16_fast(a: u16, b: u16) -> u16 {
            // 257 = 2^8 + 1, and 256 ≡ -1 (mod 257).
            let prod = (a as u32) * (b as u32); // <= 65536
            let low = (prod & 0xFF) as i32; // 0..255
            let high = (prod >> 8) as i32; // 0..256
            let mut r = low - high; // -256..255
            if r < 0 {
                r += 257;
            }
            r as u16
        }

        #[inline]
        fn flush_chunk(
            batch: usize,
            acc_star_u16: &mut [u16],
            acc_low_u16: &mut [u16],
            tmp_star_u32: &mut [u32],
            tmp_low_u32: &mut [u32],
        ) {
            for i in 0..batch {
                let t = reduce_mod257_u32(tmp_star_u32[i]);
                if t != 0 {
                    acc_star_u16[i] = add_mod(acc_star_u16[i], t);
                }
                tmp_star_u32[i] = 0;
                let t = reduce_mod257_u32(tmp_low_u32[i]);
                if t != 0 {
                    acc_low_u16[i] = add_mod(acc_low_u16[i], t);
                }
                tmp_low_u32[i] = 0;
            }
        }

        TENSOR_RS_F257_R3_SCRATCH.with(|cell| {
            let mut s = cell.borrow_mut();
            let s: &mut TensorRsF257Rank3Scratch = &mut *s;

            s.coords.resize(batch * 3, 0);
            s.lam01_t.resize(batch * side2, 0);
            s.lam2_vec.resize(batch, 0);
            s.acc_star_u16.resize(batch, 0);
            s.acc_low_u16.resize(batch, 0);
            s.tmp_star_u32.resize(batch, 0);
            s.tmp_low_u32.resize(batch, 0);
            for v in s.acc_star_u16.iter_mut() {
                *v = 0;
            }
            for v in s.acc_low_u16.iter_mut() {
                *v = 0;
            }
            for v in s.tmp_star_u32.iter_mut() {
                *v = 0;
            }
            for v in s.tmp_low_u32.iter_mut() {
                *v = 0;
            }

            // Precompute per-idx coordinates and lam01^T(pos,i) = lam0[c0]*lam1[c1] mod 257,
            // stored as contiguous `[pos*batch + i]` to enable vectorization across `i`.
            for (i, &idx) in idxs.iter().enumerate() {
                let coords = self.decompose_index(idx);
                debug_assert_eq!(coords.len(), 3);
                let c0 = coords[0];
                let c1 = coords[1];
                let c2 = coords[2];
                s.coords[i * 3] = c0;
                s.coords[i * 3 + 1] = c1;
                s.coords[i * 3 + 2] = c2;

                let lam0 = &self.lam_star_u16[(c0 * side)..(c0 * side + side)];
                let lam1 = &self.lam_star_u16[(c1 * side)..(c1 * side + side)];
                for j1 in 0..side {
                    let l1 = lam1[j1];
                    for j0 in 0..side {
                        let pos = j1 * side + j0;
                        s.lam01_t[pos * batch + i] = mul_mod257_u16_fast(lam0[j0], l1);
                    }
                }
            }

            // Traverse `w_eval` once in Layout A and update all batch accumulators.
            let mut w_idx = 0usize;
            let mut chunk = 0usize;
            const CHUNK: usize = 1024;

            // Low cube (also contributes to low).
            for c2 in 0..base_k {
                // Precompute lam2 per coin for this c2.
                for i in 0..batch {
                    let base2 = s.coords[i * 3 + 2] * side;
                    s.lam2_vec[i] = self.lam_star_u16[base2 + c2];
                }
                for c1 in 0..base_k {
                    for c0 in 0..base_k {
                        let w = w_eval_u16[w_idx] % 257;
                        if w != 0 {
                            let pos = c1 * side + c0;
                            let lam01_pos = &s.lam01_t[(pos * batch)..(pos * batch + batch)];
                            for i in 0..batch {
                                let coeff01 = lam01_pos[i];
                                if coeff01 != 0 {
                                    let coeff = mul_mod257_u16_fast(coeff01, s.lam2_vec[i]);
                                    if coeff != 0 {
                                        let prod = (coeff as u32) * (w as u32);
                                        s.tmp_star_u32[i] = s.tmp_star_u32[i].wrapping_add(prod);
                                        s.tmp_low_u32[i] = s.tmp_low_u32[i].wrapping_add(prod);
                                    }
                                }
                            }
                        }
                        w_idx += 1;
                        chunk += 1;
                        if chunk == CHUNK {
                            flush_chunk(
                                batch,
                                &mut s.acc_star_u16,
                                &mut s.acc_low_u16,
                                &mut s.tmp_star_u32,
                                &mut s.tmp_low_u32,
                            );
                            chunk = 0;
                        }
                    }
                }
            }
            debug_assert_eq!(w_idx, k);

            // Rest of side-cube, skipping low cube points.
            for c2 in 0..side {
                for i in 0..batch {
                    let base2 = s.coords[i * 3 + 2] * side;
                    s.lam2_vec[i] = self.lam_star_u16[base2 + c2];
                }
                for c1 in 0..side {
                    for c0 in 0..side {
                        if c0 < base_k && c1 < base_k && c2 < base_k {
                            continue;
                        }
                        let w = w_eval_u16[w_idx] % 257;
                        if w != 0 {
                            let pos = c1 * side + c0;
                            let lam01_pos = &s.lam01_t[(pos * batch)..(pos * batch + batch)];
                            for i in 0..batch {
                                let coeff01 = lam01_pos[i];
                                if coeff01 != 0 {
                                    let coeff = mul_mod257_u16_fast(coeff01, s.lam2_vec[i]);
                                    if coeff != 0 {
                                        let prod = (coeff as u32) * (w as u32);
                                        s.tmp_star_u32[i] = s.tmp_star_u32[i].wrapping_add(prod);
                                    }
                                }
                            }
                        }
                        w_idx += 1;
                        chunk += 1;
                        if chunk == CHUNK {
                            flush_chunk(
                                batch,
                                &mut s.acc_star_u16,
                                &mut s.acc_low_u16,
                                &mut s.tmp_star_u32,
                                &mut s.tmp_low_u32,
                            );
                            chunk = 0;
                        }
                    }
                }
            }
            if chunk != 0 {
                flush_chunk(
                    batch,
                    &mut s.acc_star_u16,
                    &mut s.acc_low_u16,
                    &mut s.tmp_star_u32,
                    &mut s.tmp_low_u32,
                );
            }
            if w_idx != k_star {
                return Err("dot_row_e_star_many_mod257_u16: total length mismatch".to_string());
            }

            out_star_u16.copy_from_slice(&s.acc_star_u16[..batch]);
            out_low_u16.copy_from_slice(&s.acc_low_u16[..batch]);
            Ok(())
        })
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
            || mat_vec_sparse_np(&self.inst.a, x, z_w, self.l, None, None),
            || mat_vec_sparse_np(&self.inst.b, x, z_w, self.l, None, None),
        );
        if y_a.len() != k || y_b.len() != k {
            return Err("bad mat-vec size".to_string());
        }

        // Build w by evaluating E(Az) and E(Bz) at systematic E* positions.
        let ea = self.code.eval_e_at_positions(&witness_pos, &y_a)?;
        let eb = self.code.eval_e_at_positions(&witness_pos, &y_b)?;
        if ea.len() != k_star || eb.len() != k_star {
            return Err("bad eval_e_at_positions length".to_string());
        }
        let mut w = Vec::with_capacity(k_star);
        for j in 0..k_star {
            w.push(ea[j] * eb[j]);
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

    fn n_total(&self) -> usize {
        self.inst.n
    }

    fn z_w_len(&self) -> usize {
        self.inst.n - self.l
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

    fn k_star(&self) -> usize {
        self.code.dim_k_star()
    }

    fn witness_positions_star(&self) -> Result<Vec<usize>, String> {
        self.code.witness_positions_star()
    }

    fn stream_w_eval_blocks(
        &self,
        witness_pos: &[usize],
        x: &[F],
        z_w: &[F],
        x_u16: Option<&[u16]>,
        z_u16: Option<&[u16]>,
        on_block_hook: Option<&(dyn Fn(usize, &[F], &[u16]) -> Result<(), String> + Sync)>,
        on_block: &mut dyn FnMut(usize, &[F]),
    ) -> Result<(), String> {
        // Single-block backend: compute one `w_eval` and stream it as block 0.
        let k = self.inst.k();
        let k_star = self.code.dim_k_star();
        if witness_pos.len() != k_star {
            return Err("stream_w_eval_blocks: witness positions length mismatch".to_string());
        }
        let (y_a, y_b) = join(
            || mat_vec_sparse_np(&self.inst.a, x, z_w, self.l, x_u16, z_u16),
            || mat_vec_sparse_np(&self.inst.b, x, z_w, self.l, x_u16, z_u16),
        );
        if y_a.len() != k || y_b.len() != k {
            return Err("stream_w_eval_blocks: bad mat-vec size".to_string());
        }

        // F257 hot path: keep `ea/eb` as `u16` and multiply in `u16` (mod 257),
        // converting to `F` exactly once for the streamed `w_eval`.
        if is_f257_field::<F>() {
            let mut ea_u16 = vec![0u16; k_star];
            let mut eb_u16 = vec![0u16; k_star];
            let y_a_u16 = y_a.iter().copied().map(f_to_u16).collect::<Vec<_>>();
            let y_b_u16 = y_b.iter().copied().map(f_to_u16).collect::<Vec<_>>();
            self.code
                .eval_e_at_positions_into_u16(witness_pos, &y_a_u16, &mut ea_u16)?;
            self.code
                .eval_e_at_positions_into_u16(witness_pos, &y_b_u16, &mut eb_u16)?;

            if k_star >= 256 {
                eb_u16
                    .par_iter_mut()
                    .zip(ea_u16.par_iter())
                    .for_each(|(b, a)| *b = mul_mod(*a, *b));
            } else {
                for j in 0..k_star {
                    eb_u16[j] = mul_mod(ea_u16[j], eb_u16[j]);
                }
            }

            let lut: Vec<F> = (0u16..=256u16).map(|d| F::from(d as u64)).collect();
            let mut w_eval = vec![F::ZERO; k_star];
            if k_star >= 256 {
                w_eval
                    .par_iter_mut()
                    .enumerate()
                    .for_each(|(j, out)| *out = lut[eb_u16[j] as usize]);
            } else {
                for j in 0..k_star {
                    w_eval[j] = lut[eb_u16[j] as usize];
                }
            }
            if let Some(h) = on_block_hook {
                h(0, &w_eval, eb_u16.as_slice())?;
            }
            on_block(0, &w_eval);
            return Ok(());
        }

        // Generic fallback: materialize `ea/eb` in the field and multiply in-field.
        let ea = self.code.eval_e_at_positions(witness_pos, &y_a)?;
        let eb = self.code.eval_e_at_positions(witness_pos, &y_b)?;
        if ea.len() != k_star || eb.len() != k_star {
            return Err("stream_w_eval_blocks: bad eval length".to_string());
        }
        let mut w_eval = vec![F::ZERO; k_star];
        if k_star >= 256 {
            w_eval
                .par_iter_mut()
                .enumerate()
                .for_each(|(j, out)| *out = ea[j] * eb[j]);
        } else {
            for j in 0..k_star {
                w_eval[j] = ea[j] * eb[j];
            }
        }
        if let Some(h) = on_block_hook {
            // Best-effort: provide mod-257 residues of `w_eval`. This is only performance-critical
            // for the F257 fast path above (which provides a zero-cost u16 slice).
            let w_eval_u16: Vec<u16> = w_eval.iter().copied().map(f_to_u16).collect();
            h(0, &w_eval, w_eval_u16.as_slice())?;
        }
        on_block(0, &w_eval);
        Ok(())
    }

    fn stream_queries_for_coins_sparse(
        &self,
        idx: usize,
        lambda: F,
        x: &[F],
        scratch: &mut Dr1csQueryScratch<F>,
        sink: &mut dyn QuerySink<F>,
    ) -> Result<(), String> {
        MulCodeDr1csNpFlpcpSparse::stream_queries_for_coins_sparse(self, idx, lambda, x, scratch, sink)
    }

    fn dot_q3_w_eval(
        &self,
        idx: usize,
        lambda: F,
        _x: &[F],
        _scratch: &mut Dr1csQueryScratch<F>,
        w_eval: &[F],
        w_eval_u16: &[u16],
    ) -> Result<F, String> {
        if idx >= self.code.len_l() {
            return Err("dot_q3_w_eval: bad coin idx".to_string());
        }
        if w_eval.len() != self.code.dim_k_star() {
            return Err("dot_q3_w_eval: bad w_eval length".to_string());
        }
        if w_eval_u16.len() != w_eval.len() {
            return Err("dot_q3_w_eval: bad w_eval_u16 length".to_string());
        }
        let mut star = [0u16; 1];
        let mut low = [0u16; 1];
        self.code
            .dot_row_e_star_many_mod257_u16(&[idx], w_eval_u16, &mut star, &mut low)?;
        let lam_u16 = (lambda.into_bigint().as_ref()[0] % 257) as u16;
        let prod = mul_mod(lam_u16, low[0]);
        let dot_u16 = sub_mod(star[0], prod);
        Ok(F::from(dot_u16 as u64))
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
        MulCodeDr1csNpFlpcpSparse::stream_queries_for_coins_sparse(self, idx, lambda, x, &mut scratch, &mut sink)?;
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
            || mat_vec_sparse_np(&self.inst.a, x, z_w, self.l, None, None),
            || mat_vec_sparse_np(&self.inst.b, x, z_w, self.l, None, None),
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
            || mat_vec_sparse_np(&self.inst.a, x, z_w, self.l, None, None),
            || join(
                || mat_vec_sparse_np(&self.inst.b, x, z_w, self.l, None, None),
                || mat_vec_sparse_np(&self.inst.c, x, z_w, self.l, None, None),
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

    fn n_total(&self) -> usize {
        self.inst.n
    }

    fn z_w_len(&self) -> usize {
        self.inst.n - self.l
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

    fn k_star(&self) -> usize {
        // RS-NP backend stores `w` of length 2k (not a multiplication-code k*).
        2 * self.inst.k()
    }

    fn witness_positions_star(&self) -> Result<Vec<usize>, String> {
        // Identity layout (best-effort; this backend is not used by Theorem-4.3 chunked prover).
        Ok((0..self.k_star()).collect())
    }

    fn stream_w_eval_blocks(
        &self,
        _witness_pos: &[usize],
        x: &[F],
        z_w: &[F],
        _x_u16: Option<&[u16]>,
        _z_u16: Option<&[u16]>,
        on_block_hook: Option<&(dyn Fn(usize, &[F], &[u16]) -> Result<(), String> + Sync)>,
        on_block: &mut dyn FnMut(usize, &[F]),
    ) -> Result<(), String> {
        let pi = RsDr1csNpFlpcpSparse::prove(self, x, z_w);
        if pi.len() < z_w.len() {
            return Err("rs stream_w_eval_blocks: bad proof length".to_string());
        }
        let w = &pi[z_w.len()..];
        if let Some(h) = on_block_hook {
            let w_u16: Vec<u16> = w.iter().copied().map(f_to_u16).collect();
            h(0, w, w_u16.as_slice())?;
        }
        on_block(0, w);
        Ok(())
    }

    fn stream_queries_for_coins_sparse(
        &self,
        idx: usize,
        lambda: F,
        x: &[F],
        _scratch: &mut Dr1csQueryScratch<F>,
        sink: &mut dyn QuerySink<F>,
    ) -> Result<(), String> {
        let (qs, _pred) = RsDr1csNpFlpcpSparse::queries_for_coins_sparse(self, idx, lambda, x)?;
        if qs.len() != 3 {
            return Err("rs stream_queries_for_coins_sparse: expected 3 queries".to_string());
        }
        for (c, i) in qs[0].terms.iter().copied() {
            sink.on_q1(c, i);
        }
        for (c, i) in qs[1].terms.iter().copied() {
            sink.on_q2(c, i);
        }
        for (c, i) in qs[2].terms.iter().copied() {
            sink.on_q3(c, i);
        }
        Ok(())
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

fn mat_vec_sparse_np<F: PrimeField>(
    m: &[SparseVec<F>],
    x: &[F],
    z_w: &[F],
    l: usize,
    x_u16: Option<&[u16]>,
    z_u16: Option<&[u16]>,
) -> Vec<F> {
    debug_assert_eq!(x.len(), l);
    debug_assert_eq!(l + z_w.len(), m.first().map(|_| l + z_w.len()).unwrap_or(l + z_w.len()));
    if is_f257_field::<F>() {
        let (x_u16, z_u16) = match (x_u16, z_u16) {
            (Some(xu), Some(zu)) => {
                debug_assert_eq!(xu.len(), l);
                debug_assert_eq!(zu.len(), z_w.len());
                (xu, zu)
            }
            _ => {
                // Fallback: build local caches.
                let xu = x.iter().copied().map(f_to_u16).collect::<Vec<_>>();
                let zu = z_w.iter().copied().map(f_to_u16).collect::<Vec<_>>();
                // SAFETY: we only use these within this call, so keep them owned.
                // To avoid code duplication below, handle this case separately.
                if m.len() >= 256 {
                    return m
                        .par_iter()
                        .map(|row| {
                            let mut acc = 0u16;
                            for (c, idx) in row.terms.iter().copied() {
                                let coeff = f_to_u16(c);
                                let v = if idx < l { xu[idx] } else { zu[idx - l] };
                                acc = add_mod(acc, mul_mod(coeff, v));
                            }
                            F::from(acc as u64)
                        })
                        .collect();
                }
                return m
                    .iter()
                    .map(|row| {
                        let mut acc = 0u16;
                        for (c, idx) in row.terms.iter().copied() {
                            let coeff = f_to_u16(c);
                            let v = if idx < l { xu[idx] } else { zu[idx - l] };
                            acc = add_mod(acc, mul_mod(coeff, v));
                        }
                        F::from(acc as u64)
                    })
                    .collect();
            }
        };
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

enum SparseAccumulator<F: PrimeField> {
    Dense {
        vals: Vec<F>,
        touched: Vec<usize>,
    },
    Map {
        map: HashMap<usize, F>,
    },
}

impl<F: PrimeField> SparseAccumulator<F> {
    fn new(len: usize) -> Self {
        // For huge variable domains, a dense accumulator is a RAM bomb.
        //
        // We switch to a hash-map-backed accumulator once the dense `vals` vector would become
        // too large. This is on the arming/query side (not prover hot loops), so the asymptotic
        // and constant-factor hit is acceptable compared to OOM.
        //
        // Empirically, a threshold around 1e6 keeps dense mode fast for small instances while
        // avoiding multi-GB allocations for large WE shapes.
        const DENSE_MAX_LEN: usize = 1_000_000;
        if len <= DENSE_MAX_LEN {
            Self::Dense {
                vals: vec![F::ZERO; len],
                touched: Vec::new(),
            }
        } else {
            Self::Map { map: HashMap::new() }
        }
    }

    fn clear(&mut self) {
        match self {
            SparseAccumulator::Dense { vals, touched } => {
                for idx in touched.drain(..) {
                    vals[idx] = F::ZERO;
                }
            }
            SparseAccumulator::Map { map } => {
                map.clear();
            }
        }
    }

    fn add_term(&mut self, idx: usize, coeff: F) {
        if coeff.is_zero() {
            return;
        }
        match self {
            SparseAccumulator::Dense { vals, touched } => {
                if vals[idx].is_zero() {
                    touched.push(idx);
                }
                vals[idx] += coeff;
            }
            SparseAccumulator::Map { map } => {
                let e = map.entry(idx).or_insert(F::ZERO);
                *e += coeff;
                // Keep map small-ish if cancellation happens.
                if e.is_zero() {
                    map.remove(&idx);
                }
            }
        }
    }

    fn take_terms(&mut self) -> Vec<(usize, F)> {
        match self {
            SparseAccumulator::Dense { vals, touched } => {
                let mut out = Vec::with_capacity(touched.len());
                for idx in touched.drain(..) {
                    let c = vals[idx];
                    if !c.is_zero() {
                        out.push((idx, c));
                    }
                    vals[idx] = F::ZERO;
                }
                out
            }
            SparseAccumulator::Map { map } => map.drain().collect(),
        }
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

    /// Clear all internal accumulators.
    ///
    /// This is public so out-of-crate FLPCP backends can implement streaming query generation
    /// without needing access to private fields.
    pub fn clear_all(&mut self) {
        self.clear();
    }

    #[inline]
    pub fn add_q1_term_on_z(&mut self, idx: usize, coeff: F) {
        self.q_a_z.add_term(idx, coeff);
    }

    #[inline]
    pub fn add_q2_term_on_z(&mut self, idx: usize, coeff: F) {
        self.q_b_z.add_term(idx, coeff);
    }

    #[inline]
    pub fn add_q3_cx2_term_on_z(&mut self, idx: usize, coeff: F) {
        self.q_cx2_z.add_term(idx, coeff);
    }

    #[inline]
    pub fn take_q1_terms_on_z(&mut self) -> Vec<(usize, F)> {
        self.q_a_z.take_terms()
    }

    #[inline]
    pub fn take_q2_terms_on_z(&mut self) -> Vec<(usize, F)> {
        self.q_b_z.take_terms()
    }

    #[inline]
    pub fn take_q3_cx2_terms_on_z(&mut self) -> Vec<(usize, F)> {
        self.q_cx2_z.take_terms()
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
    reduce_mod257_u32((a as u32) * (b as u32))
}

pub fn is_f257_field<F: PrimeField>() -> bool {
    let bytes = F::MODULUS.to_bytes_le();
    let mut acc: u64 = 0;
    for (i, b) in bytes.iter().enumerate().take(8) {
        acc |= (*b as u64) << (8 * i);
    }
    acc == 257
}

pub(crate) fn f_to_u16<F: PrimeField>(v: F) -> u16 {
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
}


