//! Streaming MLE implementations for Symphony.
//!
//! This module provides concrete streaming MLE types that compute evaluations
//! on-demand without materializing O(2^n) evaluation tables.
//!
//! Key types:
//! - `SparseMatrixMle`: computes `(M * w)[row]` on demand from sparse matrix
//! - `EqStreamingMle`: computes `eq(bits(index), r)` on demand
//! - `MonomialDigitMle`: computes monomial evaluation `m_j[idx]` on demand

use ark_std::vec::Vec;
use std::sync::Arc;
use stark_rings::{OverField, PolyRing, Ring};
use stark_rings_linalg::SparseMatrix;

use crate::streaming_sumcheck::{DenseStreamingMle, StreamingMle};

/// Streaming MLE backed by a base-ring scalar evaluation vector.
///
/// Stores `evals[index]` in the base ring and lifts to the ambient ring via `R::from`.
/// This is useful when the polynomial is known to be *constant-coefficient* in the ring.
#[derive(Clone)]
pub struct BaseScalarVecMle<R: OverField> {
    evals: Arc<Vec<R::BaseRing>>,
    num_vars: usize,
}

impl<R: OverField> BaseScalarVecMle<R> {
    pub fn new(num_vars: usize, evals: Arc<Vec<R::BaseRing>>) -> Self {
        Self { evals, num_vars }
    }
}

impl<R: OverField> StreamingMle<R> for BaseScalarVecMle<R> {
    fn num_vars(&self) -> usize {
        self.num_vars
    }

    fn eval_at_index(&self, index: usize) -> R {
        R::from(self.evals[index])
    }

    fn fix_variable(&self, r: R) -> Box<dyn StreamingMle<R>> {
        // Materialize the fixed polynomial evaluations to avoid exponential recursion.
        let half = 1 << (self.num_vars - 1);
        let new_evals: Vec<R> = (0..half)
            .map(|i| {
                (R::ONE - r) * R::from(self.evals[i << 1]) + r * R::from(self.evals[(i << 1) | 1])
            })
            .collect();
        Box::new(DenseStreamingMle::new(new_evals))
    }
}

// ============================================================================
// Hadamard-side Streaming MLEs
// ============================================================================

/// Streaming MLE for sparse matrix-vector product y = M * w.
///
/// Computes y[row] on demand from the sparse matrix structure.
pub struct SparseMatrixMle<R: OverField> {
    matrix: Arc<SparseMatrix<R>>,
    witness: Arc<Vec<R>>,
    num_vars: usize,
}

impl<R: OverField> SparseMatrixMle<R> {
    pub fn new(matrix: Arc<SparseMatrix<R>>, witness: Arc<Vec<R>>) -> Self {
        let nrows = matrix.nrows;
        assert!(nrows.is_power_of_two());
        let num_vars = nrows.trailing_zeros() as usize;
        Self { matrix, witness, num_vars }
    }

    /// Compute y[row] = sum_j M[row,j] * w[j]
    #[inline]
    fn compute_row(&self, row: usize) -> R {
        if row >= self.matrix.nrows {
            return R::ZERO;
        }
        let mut sum = R::ZERO;
        for (coeff, col_idx) in &self.matrix.coeffs[row] {
            if *col_idx < self.witness.len() {
                sum += *coeff * self.witness[*col_idx];
            }
        }
        sum
    }
}

impl<R: OverField> StreamingMle<R> for SparseMatrixMle<R> {
    fn num_vars(&self) -> usize {
        self.num_vars
    }

    fn eval_at_index(&self, index: usize) -> R {
        self.compute_row(index)
    }

    fn fix_variable(&self, r: R) -> Box<dyn StreamingMle<R>> {
        // Materialize the fixed polynomial evaluations to avoid exponential recursion.
        // This keeps total work O(m) across rounds (like dense sumcheck), while still
        // avoiding materializing the full length-m vector up front.
        let half = 1 << (self.num_vars - 1);
        let new_evals: Vec<R> = (0..half)
            .map(|i| {
                let f0 = self.compute_row(i << 1);
                let f1 = self.compute_row((i << 1) | 1);
                (R::ONE - r) * f0 + r * f1
            })
            .collect();
        Box::new(DenseStreamingMle::new(new_evals))
    }
}

/// Streaming MLE for eq(x, r) polynomial.
///
/// eq(bits(index), r) = prod_i (bits_i * r_i + (1-bits_i) * (1-r_i))
pub struct EqStreamingMle<R: OverField> {
    r: Vec<R>,
    one_minus_r: Vec<R>,
}

impl<R: OverField> EqStreamingMle<R> {
    pub fn new(r: Vec<R>) -> Self {
        let one_minus_r = r.iter().copied().map(|x| R::ONE - x).collect();
        Self { r, one_minus_r }
    }
}

impl<R: OverField> StreamingMle<R> for EqStreamingMle<R> {
    fn num_vars(&self) -> usize {
        self.r.len()
    }

    fn eval_at_index(&self, index: usize) -> R {
        let mut prod = R::ONE;
        // Slightly faster than recomputing (1 - r_i) each time.
        for i in 0..self.r.len() {
            let bit = ((index >> i) & 1) == 1;
            prod *= if bit { self.r[i] } else { self.one_minus_r[i] };
        }
        prod
    }

    fn fix_variable(&self, r: R) -> Box<dyn StreamingMle<R>> {
        // eq'(b) = (1-r)*eq(0,b,r[1:]) + r*eq(1,b,r[1:])
        //        = ((1-r)*(1-r[0]) + r*r[0]) * eq(b, r[1:])
        //        = eq(r, r[0]) * eq(b, r[1:])
        // So we can just prepend the eq factor and use remaining r values.
        let eq_factor = (R::ONE - r) * self.one_minus_r[0] + r * self.r[0];
        let new_r: Vec<R> = self.r[1..].to_vec();

        if new_r.is_empty() {
            // Single element MLE
            Box::new(crate::streaming_sumcheck::DenseStreamingMle::new(vec![eq_factor]))
        } else {
            Box::new(ScaledEqMle {
                scale: eq_factor,
                inner: EqStreamingMle::new(new_r),
            })
        }
    }
}

struct ScaledEqMle<R: OverField> {
    scale: R,
    inner: EqStreamingMle<R>,
}

impl<R: OverField> StreamingMle<R> for ScaledEqMle<R> {
    fn num_vars(&self) -> usize {
        self.inner.num_vars()
    }

    fn eval_at_index(&self, index: usize) -> R {
        self.scale * self.inner.eval_at_index(index)
    }

    fn fix_variable(&self, r: R) -> Box<dyn StreamingMle<R>> {
        let inner_fixed = self.inner.fix_variable(r);
        Box::new(ScaledMle {
            scale: self.scale,
            inner: inner_fixed,
        })
    }
}

struct ScaledMle<R: OverField> {
    scale: R,
    inner: Box<dyn StreamingMle<R>>,
}

impl<R: OverField> StreamingMle<R> for ScaledMle<R> {
    fn num_vars(&self) -> usize {
        self.inner.num_vars()
    }

    fn eval_at_index(&self, index: usize) -> R {
        self.scale * self.inner.eval_at_index(index)
    }

    fn fix_variable(&self, r: R) -> Box<dyn StreamingMle<R>> {
        Box::new(ScaledMle {
            scale: self.scale,
            inner: self.inner.fix_variable(r),
        })
    }
}

/// Streaming eq MLE specialized to **constant-coefficient ring elements**.
///
/// Stores challenges in the base ring and evaluates `eq(bits(index), r)` in the base ring,
/// then lifts to `R` via `R::from`.
///
/// This is a big constant-factor win vs `EqStreamingMle<R>` when `R` is a polynomial ring
/// and the challenges were sampled from the base ring and embedded as constant terms.
pub struct EqBaseStreamingMle<R: OverField + PolyRing>
where
    R::BaseRing: Ring,
{
    r: Vec<R::BaseRing>,
    one_minus_r: Vec<R::BaseRing>,
}

impl<R: OverField + PolyRing> EqBaseStreamingMle<R>
where
    R::BaseRing: Ring,
{
    pub fn new(r: Vec<R::BaseRing>) -> Self {
        let one_minus_r = r.iter().copied().map(|x| R::BaseRing::ONE - x).collect();
        Self { r, one_minus_r }
    }
}

impl<R: OverField + PolyRing> StreamingMle<R> for EqBaseStreamingMle<R>
where
    R::BaseRing: Ring,
{
    fn num_vars(&self) -> usize {
        self.r.len()
    }

    fn eval_at_index(&self, index: usize) -> R {
        let mut prod = R::BaseRing::ONE;
        for i in 0..self.r.len() {
            let bit = ((index >> i) & 1) == 1;
            prod *= if bit { self.r[i] } else { self.one_minus_r[i] };
        }
        R::from(prod)
    }

    fn fix_variable(&self, r: R) -> Box<dyn StreamingMle<R>> {
        // r is sampled from the transcript as a base-ring element and embedded into R,
        // so extracting coeffs()[0] recovers the scalar.
        let r0 = r.coeffs()[0];
        let eq_factor = (R::BaseRing::ONE - r0) * self.one_minus_r[0] + r0 * self.r[0];
        let new_r: Vec<R::BaseRing> = self.r[1..].to_vec();

        if new_r.is_empty() {
            Box::new(crate::streaming_sumcheck::DenseStreamingMle::new(vec![R::from(eq_factor)]))
        } else {
            Box::new(ScaledEqBaseStreamingMle::<R>::new(eq_factor, new_r))
        }
    }
}

/// Base-ring eq MLE with an extra base-ring scale factor.
struct ScaledEqBaseStreamingMle<R: OverField + PolyRing>
where
    R::BaseRing: Ring,
{
    scale: R::BaseRing,
    r: Vec<R::BaseRing>,
    one_minus_r: Vec<R::BaseRing>,
}

impl<R: OverField + PolyRing> ScaledEqBaseStreamingMle<R>
where
    R::BaseRing: Ring,
{
    fn new(scale: R::BaseRing, r: Vec<R::BaseRing>) -> Self {
        let one_minus_r = r.iter().copied().map(|x| R::BaseRing::ONE - x).collect();
        Self { scale, r, one_minus_r }
    }
}

impl<R: OverField + PolyRing> StreamingMle<R> for ScaledEqBaseStreamingMle<R>
where
    R::BaseRing: Ring,
{
    fn num_vars(&self) -> usize {
        self.r.len()
    }

    fn eval_at_index(&self, index: usize) -> R {
        let mut prod = R::BaseRing::ONE;
        for i in 0..self.r.len() {
            let bit = ((index >> i) & 1) == 1;
            prod *= if bit { self.r[i] } else { self.one_minus_r[i] };
        }
        R::from(self.scale * prod)
    }

    fn fix_variable(&self, r: R) -> Box<dyn StreamingMle<R>> {
        let r0 = r.coeffs()[0];
        let eq_factor = (R::BaseRing::ONE - r0) * self.one_minus_r[0] + r0 * self.r[0];
        let new_scale = self.scale * eq_factor;
        let new_r: Vec<R::BaseRing> = self.r[1..].to_vec();
        if new_r.is_empty() {
            Box::new(crate::streaming_sumcheck::DenseStreamingMle::new(vec![R::from(new_scale)]))
        } else {
            Box::new(ScaledEqBaseStreamingMle::<R>::new(new_scale, new_r))
        }
    }
}

// ============================================================================
// Closure-based Streaming MLE (for monomial and other complex evaluations)
// ============================================================================

/// Generic closure-based streaming MLE.
///
/// This allows wrapping any `Fn(usize) -> R` as a streaming MLE, enabling
/// on-demand evaluation for complex structures like monomial digits.
pub struct ClosureMle<R, F>
where
    R: OverField,
    F: Fn(usize) -> R + Send + Sync,
{
    eval_fn: F,
    num_vars: usize,
    _phantom: std::marker::PhantomData<R>,
}

impl<R, F> ClosureMle<R, F>
where
    R: OverField,
    F: Fn(usize) -> R + Send + Sync,
{
    pub fn new(num_vars: usize, eval_fn: F) -> Self {
        Self {
            eval_fn,
            num_vars,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<R, F> StreamingMle<R> for ClosureMle<R, F>
where
    R: OverField,
    F: Fn(usize) -> R + Send + Sync + Clone + 'static,
{
    fn num_vars(&self) -> usize {
        self.num_vars
    }

    fn eval_at_index(&self, index: usize) -> R {
        (self.eval_fn)(index)
    }

    fn fix_variable(&self, r: R) -> Box<dyn StreamingMle<R>> {
        // Materialize the fixed polynomial evaluations to avoid exponential recursion.
        let half = 1 << (self.num_vars - 1);
        let eval_fn = self.eval_fn.clone();
        let new_evals: Vec<R> = (0..half)
            .map(|i| {
                let f0 = eval_fn(i << 1);
                let f1 = eval_fn((i << 1) | 1);
                (R::ONE - r) * f0 + r * f1
            })
            .collect();
        Box::new(DenseStreamingMle::new(new_evals))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_eq_streaming_mle() {
        use stark_rings::cyclotomic_ring::models::frog_ring::RqPoly as R;
        use stark_rings::Ring;
        let r = vec![R::from(3u128), R::from(5u128), R::from(7u128)];
        let mle = EqStreamingMle::new(r.clone());

        // Brute force expected values
        for idx in 0..(1usize << r.len()) {
            let mut exp = R::ONE;
            for (i, r_i) in r.iter().enumerate() {
                let bit = ((idx >> i) & 1) == 1;
                exp *= if bit { *r_i } else { R::ONE - *r_i };
            }
            assert_eq!(mle.eval_at_index(idx), exp);
        }
    }
}
