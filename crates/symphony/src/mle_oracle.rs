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
use stark_rings::OverField;
use stark_rings_linalg::SparseMatrix;

use crate::streaming_sumcheck::StreamingMle;
use crate::streaming_sumcheck::FixedStreamingMle;

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
        Box::new(FixedStreamingMle::new(Box::new(self.clone()), r))
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
        // Keep it streaming: wrap with an interpolating fixed-variable adapter.
        Box::new(FixedStreamingMle::new(
            Box::new(SparseMatrixMle {
                matrix: self.matrix.clone(),
                witness: self.witness.clone(),
                num_vars: self.num_vars,
            }),
            r,
        ))
    }
}

/// Streaming MLE for eq(x, r) polynomial.
///
/// eq(bits(index), r) = prod_i (bits_i * r_i + (1-bits_i) * (1-r_i))
pub struct EqStreamingMle<R: OverField> {
    r: Vec<R>,
}

impl<R: OverField> EqStreamingMle<R> {
    pub fn new(r: Vec<R>) -> Self {
        Self { r }
    }
}

impl<R: OverField> StreamingMle<R> for EqStreamingMle<R> {
    fn num_vars(&self) -> usize {
        self.r.len()
    }

    fn eval_at_index(&self, index: usize) -> R {
        let mut prod = R::ONE;
        for (i, r_i) in self.r.iter().enumerate() {
            let bit = ((index >> i) & 1) == 1;
            if bit {
                prod *= *r_i;
            } else {
                prod *= R::ONE - *r_i;
            }
        }
        prod
    }

    fn fix_variable(&self, r: R) -> Box<dyn StreamingMle<R>> {
        // eq'(b) = (1-r)*eq(0,b,r[1:]) + r*eq(1,b,r[1:])
        //        = ((1-r)*(1-r[0]) + r*r[0]) * eq(b, r[1:])
        //        = eq(r, r[0]) * eq(b, r[1:])
        // So we can just prepend the eq factor and use remaining r values.
        let eq_factor = (R::ONE - r) * (R::ONE - self.r[0]) + r * self.r[0];
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
        // Keep it streaming: wrap with an interpolating fixed-variable adapter.
        Box::new(FixedStreamingMle::new(
            Box::new(ClosureMle {
                eval_fn: self.eval_fn.clone(),
                num_vars: self.num_vars,
                _phantom: std::marker::PhantomData,
            }),
            r,
        ))
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
