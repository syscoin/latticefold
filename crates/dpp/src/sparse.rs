//! Sparse vectors and dot-products for efficient DPP frontends.

use ark_ff::PrimeField;

/// Sparse vector: list of (coefficient, index) terms.
#[derive(Clone, Debug, Default)]
pub struct SparseVec<F: PrimeField> {
    pub terms: Vec<(F, usize)>,
}

impl<F: PrimeField> SparseVec<F> {
    pub fn new(terms: Vec<(F, usize)>) -> Self {
        Self { terms }
    }

    pub fn is_empty(&self) -> bool {
        self.terms.is_empty()
    }

    pub fn dot(&self, v: &[F]) -> F {
        self.terms
            .iter()
            .fold(F::ZERO, |acc, (c, idx)| acc + (*c * v[*idx]))
    }
}

