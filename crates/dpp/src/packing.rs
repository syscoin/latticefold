//! Query packing transformation for bounded FLPCPs (Construction 5.21 / Theorem 5.7).
//!
//! This compiles a bounded k-query (F)LPCP into a 1-query “dot-product proof” verifier,
//! by sampling a structured weight vector `w` and packing the k queries `Q` into one query
//! `q = Q^T w`.

use ark_ff::{BigInteger, PrimeField};
use num_bigint::{BigInt, BigUint};
use num_traits::{One, Signed, Zero};
use rand::RngCore;
use std::collections::BTreeMap;
use thiserror::Error;

use crate::subset_sum::{decode_bounded_subset_sum, SubsetSumError};
use crate::sparse::SparseVec;

#[derive(Debug, Error)]
pub enum PackingError {
    #[error("invalid parameters")]
    InvalidParams,
    #[error("packed field modulus too small for bounded decoding")]
    ModulusTooSmall,
    #[error("subset-sum decode failed: {0}")]
    Decode(#[from] SubsetSumError),
}

/// Minimal interface of a bounded FLPCP verifier needed for Construction 5.21.
///
/// We model the verifier output as:
/// - `Q`: k linear query vectors (each length n+m) over `F`
/// - `accept`: a deterministic predicate on the k answers (as field elements)
pub trait BoundedFlpcp<F: PrimeField> {
    /// Input length n (field elements).
    fn n(&self) -> usize;
    /// Proof length m (field elements).
    fn m(&self) -> usize;
    /// Query count k.
    fn k(&self) -> usize;

    /// Bounds `b_i` such that honest answers satisfy |[a_i]_p| <= b_i - 1 (integer rep).
    fn bounds_b(&self) -> Vec<BigInt>;

    /// Generate k query vectors `Q` and an accept predicate for a given instance `x`.
    ///
    /// The accept predicate is returned as an enum tag + parameters (to keep this trait object-safe).
    fn sample_queries_and_predicate(
        &self,
        rng: &mut dyn RngCore,
        x: &[F],
    ) -> Result<(Vec<Vec<F>>, FlpcpPredicate<F>), String>;
}

/// Sparse-query interface for a bounded FLPCP verifier.
///
/// Same as `BoundedFlpcp`, but query vectors are represented sparsely as `(coeff, index)` lists.
pub trait BoundedFlpcpSparse<F: PrimeField> {
    /// Input length n (field elements).
    fn n(&self) -> usize;
    /// Proof length m (field elements).
    fn m(&self) -> usize;
    /// Query count k.
    fn k(&self) -> usize;
    /// Bounds `b_i` such that honest answers satisfy |[a_i]_p| <= b_i - 1 (integer rep).
    fn bounds_b(&self) -> Vec<BigInt>;

    fn sample_queries_and_predicate_sparse(
        &self,
        rng: &mut dyn RngCore,
        x: &[F],
    ) -> Result<(Vec<SparseVec<F>>, FlpcpPredicate<F>), String>;
}

/// A small set of built-in predicates for tests and for the dR1CS FLPCP.
#[derive(Clone, Debug)]
pub enum FlpcpPredicate<F: PrimeField> {
    /// Accept iff a0 * a1 == a2.
    MulEq,
    /// Accept iff (a0 * a1 - a2) == 0 mod p_small, where p_small is an integer modulus.
    ///
    /// This is used by the bounded-embedding transformation (Section 5.1), where verification
    /// is performed on answers reduced mod the original smaller field.
    MulEqModP { p_small: BigInt },
    /// Accept iff a0 * a1 == a2 and a3 == 0 (used for 4-query variants).
    MulEqAndZero,
    /// Accept iff all answers equal 0 (useful for toy tests).
    AllZero,
    /// Custom predicate (not supported in packing verifier).
    #[allow(dead_code)]
    Unsupported(core::marker::PhantomData<F>),
    /// Ignore the last `tail_len` answers and apply `inner` to the prefix.
    ///
    /// This is useful for strong-soundness embedding (Claim 5.28), where extra “bound-test”
    /// queries contribute answers that must only be bounded (enforced by decoding), with no
    /// additional predicate constraints.
    IgnoreTail { inner: Box<FlpcpPredicate<F>>, tail_len: usize },
}

impl<F: PrimeField> FlpcpPredicate<F> {
    pub fn check(&self, answers: &[F]) -> bool {
        match self {
            FlpcpPredicate::MulEq => {
                if answers.len() != 3 {
                    return false;
                }
                answers[0] * answers[1] == answers[2]
            }
            FlpcpPredicate::MulEqModP { p_small } => {
                if answers.len() != 3 {
                    return false;
                }
                let a0 = BigInt::from_bytes_le(num_bigint::Sign::Plus, &answers[0].into_bigint().to_bytes_le()) % p_small;
                let a1 = BigInt::from_bytes_le(num_bigint::Sign::Plus, &answers[1].into_bigint().to_bytes_le()) % p_small;
                let a2 = BigInt::from_bytes_le(num_bigint::Sign::Plus, &answers[2].into_bigint().to_bytes_le()) % p_small;
                ((a0 * a1) - a2) % p_small == BigInt::zero()
            }
            FlpcpPredicate::MulEqAndZero => {
                if answers.len() != 4 {
                    return false;
                }
                answers[0] * answers[1] == answers[2] && answers[3].is_zero()
            }
            FlpcpPredicate::AllZero => answers.iter().all(|a| a.is_zero()),
            FlpcpPredicate::Unsupported(_) => false,
            FlpcpPredicate::IgnoreTail { inner, tail_len } => {
                if *tail_len > answers.len() {
                    return false;
                }
                inner.check(&answers[..(answers.len() - *tail_len)])
            }
        }
    }
}

/// Parameters for the packed DPP (Construction 5.21).
#[derive(Clone, Debug)]
pub struct PackedDppParams {
    /// ℓ (denoted `ell` in paper): the per-coordinate sampling range size.
    pub ell: u64,
}

/// A packed DPP verifier built from a bounded FLPCP via Construction 5.21.
#[derive(Clone, Debug)]
pub struct DppFromBoundedFlpcp<F: PrimeField, V: BoundedFlpcp<F>> {
    pub flpcp: V,
    pub params: PackedDppParams,
    _pd: core::marker::PhantomData<F>,
}

/// A sampled packed DPP query (Construction 5.21), plus decoding metadata.
///
/// This is the artifact a WE “armer” wants: a single linear query vector `q` over the
/// concatenated witness vector `v=(x||π)` such that the verifier accepts based on the
/// packed answer `a = <q, v>`.
#[derive(Clone, Debug)]
pub struct PackedDppQuery<F: PrimeField> {
    /// Packed query vector (length n+m).
    pub q: Vec<F>,
    /// Packing weights `w` (integers, length k).
    pub w: Vec<BigInt>,
    /// Bounds `b` (integers, length k).
    pub b: Vec<BigInt>,
    /// Underlying k-answer predicate (on decoded answers).
    pub pred: FlpcpPredicate<F>,
}

/// A sampled packed DPP query in sparse form.
#[derive(Clone, Debug)]
pub struct PackedDppQuerySparse<F: PrimeField> {
    /// Packed query vector (sparse) over v=(x||π), indices in [0..n+m).
    pub q: SparseVec<F>,
    /// Packing weights `w` (integers, length k).
    pub w: Vec<BigInt>,
    /// Bounds `b` (integers, length k).
    pub b: Vec<BigInt>,
    /// Underlying k-answer predicate (on decoded answers).
    pub pred: FlpcpPredicate<F>,
}

impl<F: PrimeField, V: BoundedFlpcp<F>> DppFromBoundedFlpcp<F, V> {
    pub fn new(flpcp: V, params: PackedDppParams) -> Self {
        Self { flpcp, params, _pd: core::marker::PhantomData }
    }

    /// Sample the packed query vector `q` (length n+m) and the packing weights `w`.
    pub fn sample_packed_query(
        &self,
        rng: &mut dyn RngCore,
        x: &[F],
    ) -> Result<(Vec<F>, Vec<BigInt>, FlpcpPredicate<F>, Vec<Vec<F>>), PackingError> {
        let k = self.flpcp.k();
        let b = self.flpcp.bounds_b();
        if k == 0 || b.len() != k || x.len() != self.flpcp.n() {
            return Err(PackingError::InvalidParams);
        }
        let (q_mat, pred) = self
            .flpcp
            .sample_queries_and_predicate(rng, x)
            .map_err(|_| PackingError::InvalidParams)?;
        if q_mat.len() != k || q_mat.iter().any(|row| row.len() != self.flpcp.n() + self.flpcp.m()) {
            return Err(PackingError::InvalidParams);
        }

        // Choose scalar b_max = max_i b_i (as BigInt).
        let mut b_max = BigInt::zero();
        for bi in &b {
            if bi > &b_max {
                b_max = bi.clone();
            }
        }
        if b_max <= BigInt::one() {
            return Err(PackingError::InvalidParams);
        }

        // Sample w as in Construction 5.21 (integer ranges).
        let ell = BigInt::from(self.params.ell);
        if ell.is_zero() {
            return Err(PackingError::InvalidParams);
        }

        let two_b = BigInt::from(2u64) * &b_max;
        let mut w: Vec<BigInt> = Vec::with_capacity(k);
        let mut pow = BigInt::one(); // (2b)^(i-1)
        for _i in 0..k {
            // low = ((pow - 1) * ell + 1), high = pow * ell
            let low = (&pow - BigInt::one()) * &ell + BigInt::one();
            let high = &pow * &ell;
            // Sample uniformly in [low, high] (inclusive).
            let span = (&high - &low) + BigInt::one();
            let span_u = big_int_to_big_uint(&span).ok_or(PackingError::InvalidParams)?;
            let r_u = sample_uniform_below(rng, &span_u);
            let samp = &low + BigInt::from(r_u);
            w.push(samp);
            pow *= &two_b;
        }

        // Ensure the field modulus is large enough so that bounded answers do not wrap mod p.
        // This is the condition in Claim 5.22: p > 2 * <b-1, w>.
        let p_mod = modulus_bigint::<F>();
        let mut bw = BigInt::zero();
        for i in 0..k {
            let bound_i = &b[i] - BigInt::one();
            if bound_i.is_negative() {
                return Err(PackingError::InvalidParams);
            }
            bw += bound_i * &w[i];
        }
        if p_mod <= BigInt::from(2u64) * bw {
            return Err(PackingError::ModulusTooSmall);
        }

        // Compute packed query coefficients: q = Q^T w over integers, then reduce mod p.
        let len = self.flpcp.n() + self.flpcp.m();
        let mut q: Vec<F> = Vec::with_capacity(len);
        for j in 0..len {
            let mut acc = BigInt::zero();
            for i in 0..k {
                let qij = field_to_centered_bigint::<F>(&q_mat[i][j]);
                acc += &w[i] * qij;
            }
            q.push(centered_bigint_to_field::<F>(&acc));
        }

        Ok((q, w, pred, q_mat))
    }

    /// Sample a reusable packed DPP query artifact for this instance `x`.
    pub fn sample_query(&self, rng: &mut dyn RngCore, x: &[F]) -> Result<PackedDppQuery<F>, PackingError> {
        let b = self.flpcp.bounds_b();
        let (q, w, pred, _q_mat) = self.sample_packed_query(rng, x)?;
        Ok(PackedDppQuery { q, w, b, pred })
    }

    /// Verify a packed answer `a = <q, (x||π)>` using the sampled query metadata.
    ///
    /// This is the verifier’s “accepting set” check in algorithmic form:
    /// decode `a` into k bounded answers, then apply the underlying predicate.
    pub fn verify_packed_answer(&self, a: &F, query: &PackedDppQuery<F>) -> Result<bool, PackingError> {
        let a_int = field_to_centered_bigint::<F>(a);
        let ans_int = decode_bounded_subset_sum(&a_int, &query.w, &query.b)?;
        let ans_field = ans_int
            .iter()
            .map(|z| centered_bigint_to_field::<F>(z))
            .collect::<Vec<_>>();
        Ok(query.pred.check(&ans_field))
    }

    /// Verify using a caller-supplied packed query (no resampling).
    pub fn verify_with_query(
        &self,
        x: &[F],
        pi: &[F],
        query: &PackedDppQuery<F>,
    ) -> Result<bool, PackingError> {
        if x.len() != self.flpcp.n() || pi.len() != self.flpcp.m() {
            return Err(PackingError::InvalidParams);
        }
        if query.q.len() != self.flpcp.n() + self.flpcp.m() {
            return Err(PackingError::InvalidParams);
        }
        let v = [x, pi].concat();
        let a = dot::<F>(&query.q, &v);
        self.verify_packed_answer(&a, query)
    }

    /// Verify by computing the packed dot product and decoding it back into k bounded answers.
    ///
    /// This corresponds to running `V'` from Construction 5.21 on the packed answer.
    pub fn verify(
        &self,
        rng: &mut dyn RngCore,
        x: &[F],
        pi: &[F],
    ) -> Result<bool, PackingError> {
        if x.len() != self.flpcp.n() || pi.len() != self.flpcp.m() {
            return Err(PackingError::InvalidParams);
        }
        let (q, w, pred, _q_mat) = self.sample_packed_query(rng, x)?;
        let v = [x, pi].concat();
        let a = dot::<F>(&q, &v);
        let a_int = field_to_centered_bigint::<F>(&a);

        // Decode a_int as <ans, w> with bounds b.
        let b = self.flpcp.bounds_b();
        let ans_int = decode_bounded_subset_sum(&a_int, &w, &b)?;
        let ans_field = ans_int
            .iter()
            .map(|z| centered_bigint_to_field::<F>(z))
            .collect::<Vec<_>>();

        Ok(pred.check(&ans_field))
    }
}

/// Sparse packed DPP verifier built from a sparse bounded FLPCP.
#[derive(Clone, Debug)]
pub struct DppFromBoundedFlpcpSparse<F: PrimeField, V: BoundedFlpcpSparse<F>> {
    pub flpcp: V,
    pub params: PackedDppParams,
    _pd: core::marker::PhantomData<F>,
}

impl<F: PrimeField, V: BoundedFlpcpSparse<F>> DppFromBoundedFlpcpSparse<F, V> {
    pub fn new(flpcp: V, params: PackedDppParams) -> Self {
        Self { flpcp, params, _pd: core::marker::PhantomData }
    }

    pub fn sample_query(
        &self,
        rng: &mut dyn RngCore,
        x: &[F],
    ) -> Result<PackedDppQuerySparse<F>, PackingError> {
        let k = self.flpcp.k();
        let b = self.flpcp.bounds_b();
        if k == 0 || b.len() != k || x.len() != self.flpcp.n() {
            return Err(PackingError::InvalidParams);
        }
        let (q_mat, pred) = self
            .flpcp
            .sample_queries_and_predicate_sparse(rng, x)
            .map_err(|_| PackingError::InvalidParams)?;
        if q_mat.len() != k {
            return Err(PackingError::InvalidParams);
        }

        // Choose scalar b_max = max_i b_i (as BigInt).
        let mut b_max = BigInt::zero();
        for bi in &b {
            if bi > &b_max {
                b_max = bi.clone();
            }
        }
        if b_max <= BigInt::one() {
            return Err(PackingError::InvalidParams);
        }

        // Sample w as in Construction 5.21 (integer ranges).
        let ell = BigInt::from(self.params.ell);
        if ell.is_zero() {
            return Err(PackingError::InvalidParams);
        }
        let two_b = BigInt::from(2u64) * &b_max;
        let mut w: Vec<BigInt> = Vec::with_capacity(k);
        let mut pow = BigInt::one(); // (2b)^(i-1)
        for _i in 0..k {
            let low = (&pow - BigInt::one()) * &ell + BigInt::one();
            let high = &pow * &ell;
            let span = (&high - &low) + BigInt::one();
            let span_u = big_int_to_big_uint(&span).ok_or(PackingError::InvalidParams)?;
            let r_u = sample_uniform_below(rng, &span_u);
            let samp = &low + BigInt::from(r_u);
            w.push(samp);
            pow *= &two_b;
        }

        // Ensure the field modulus is large enough so that bounded answers do not wrap mod p.
        let p_mod = modulus_bigint::<F>();
        let mut bw = BigInt::zero();
        for i in 0..k {
            let bound_i = &b[i] - BigInt::one();
            if bound_i.is_negative() {
                return Err(PackingError::InvalidParams);
            }
            bw += bound_i * &w[i];
        }
        if p_mod <= BigInt::from(2u64) * bw {
            return Err(PackingError::ModulusTooSmall);
        }

        // Compute packed query coefficients q = Σ_i w_i * q_i over integers, then reduce mod p.
        let mut acc: BTreeMap<usize, BigInt> = BTreeMap::new();
        for i in 0..k {
            let wi = &w[i];
            for (coeff, idx) in &q_mat[i].terms {
                let cij = field_to_centered_bigint::<F>(coeff);
                let entry = acc.entry(*idx).or_insert_with(BigInt::zero);
                *entry += wi * cij;
            }
        }
        let mut terms: Vec<(F, usize)> = Vec::with_capacity(acc.len());
        for (idx, z) in acc {
            let f = centered_bigint_to_field::<F>(&z);
            if !f.is_zero() {
                terms.push((f, idx));
            }
        }

        Ok(PackedDppQuerySparse { q: SparseVec::new(terms), w, b, pred })
    }

    pub fn verify_packed_answer(
        &self,
        a: &F,
        query: &PackedDppQuerySparse<F>,
    ) -> Result<bool, PackingError> {
        let a_int = field_to_centered_bigint::<F>(a);
        let ans_int = decode_bounded_subset_sum(&a_int, &query.w, &query.b)?;
        let ans_field = ans_int
            .iter()
            .map(|z| centered_bigint_to_field::<F>(z))
            .collect::<Vec<_>>();
        Ok(query.pred.check(&ans_field))
    }

    pub fn verify_with_query(
        &self,
        x: &[F],
        pi: &[F],
        query: &PackedDppQuerySparse<F>,
    ) -> Result<bool, PackingError> {
        if x.len() != self.flpcp.n() || pi.len() != self.flpcp.m() {
            return Err(PackingError::InvalidParams);
        }
        let v = [x, pi].concat();
        let a = query.q.dot(&v);
        self.verify_packed_answer(&a, query)
    }
}

fn big_int_to_big_uint(x: &BigInt) -> Option<BigUint> {
    if x.is_negative() {
        None
    } else {
        Some(BigUint::from_bytes_le(&x.to_bytes_le().1))
    }
}

/// Unbiased sampling of a uniform integer in [0, n).
fn sample_uniform_below(rng: &mut dyn RngCore, n: &BigUint) -> BigUint {
    if n.is_zero() {
        return BigUint::ZERO;
    }
    let bits = n.bits();
    let nbytes = ((bits + 7) / 8) as usize;
    let max = BigUint::one() << (nbytes * 8);
    let limit = &max - (&max % n); // largest multiple of n below 2^(8*nbytes)

    loop {
        let mut buf = vec![0u8; nbytes];
        rng.fill_bytes(&mut buf);
        let r = BigUint::from_bytes_le(&buf);
        if r < limit {
            return r % n;
        }
    }
}

fn dot<F: PrimeField>(a: &[F], b: &[F]) -> F {
    debug_assert_eq!(a.len(), b.len());
    a.iter().zip(b.iter()).fold(F::zero(), |acc, (x, y)| acc + (*x * *y))
}

/// Map a field element to its centered integer representative in [-(p-1)/2, (p-1)/2].
fn field_to_centered_bigint<F: PrimeField>(x: &F) -> BigInt {
    let p = modulus_bigint::<F>();
    let half = (&p - BigInt::one()) / BigInt::from(2u64);
    let mut t = BigInt::from_bytes_le(num_bigint::Sign::Plus, &x.into_bigint().to_bytes_le());
    // If t > half, interpret as negative: t - p
    if t > half {
        t -= p;
    }
    t
}

/// Reduce an integer (centered or not) into the field.
fn centered_bigint_to_field<F: PrimeField>(z: &BigInt) -> F {
    // Convert via modular reduction into [0,p).
    let p = modulus_bigint::<F>();
    let mut t = z.mod_floor(&p);
    if t.is_negative() {
        t += &p;
    }
    let bytes = t.to_bytes_le().1;
    F::from_le_bytes_mod_order(&bytes)
}

fn modulus_bigint<F: PrimeField>() -> BigInt {
    BigInt::from_bytes_le(num_bigint::Sign::Plus, &F::MODULUS.to_bytes_le())
}

trait ModFloor {
    fn mod_floor(&self, m: &BigInt) -> BigInt;
}

impl ModFloor for BigInt {
    fn mod_floor(&self, m: &BigInt) -> BigInt {
        let r = self % m;
        if r.is_negative() { r + m } else { r }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_ff::{Fp64, MontBackend, MontConfig};
    use rand_chacha::ChaCha20Rng;
    use rand::SeedableRng;

    // Small prime field for tests.
    #[derive(MontConfig)]
    #[modulus = "10007"]
    #[generator = "5"]
    pub struct F97Config;
    type F = Fp64<MontBackend<F97Config, 1>>;

    #[derive(Clone, Debug)]
    struct Toy3Query;

    impl BoundedFlpcp<F> for Toy3Query {
        fn n(&self) -> usize { 1 }
        fn m(&self) -> usize { 0 }
        fn k(&self) -> usize { 3 }
        fn bounds_b(&self) -> Vec<BigInt> { vec![BigInt::from(2), BigInt::from(2), BigInt::from(2)] } // |a_i|<=1
        fn sample_queries_and_predicate(
            &self,
            _rng: &mut dyn RngCore,
            _x: &[F],
        ) -> Result<(Vec<Vec<F>>, FlpcpPredicate<F>), String> {
            // Q answers are simply a0=x, a1=1, a2=x
            Ok((
                vec![vec![F::one()], vec![F::one()], vec![F::one()]],
                FlpcpPredicate::MulEq,
            ))
        }
    }

    #[test]
    fn test_packing_roundtrip_toy() {
        let flpcp = Toy3Query;
        let dpp = DppFromBoundedFlpcp::<F, _>::new(flpcp, PackedDppParams { ell: 32 });
        let mut rng = ChaCha20Rng::seed_from_u64(1);
        let x = vec![F::from(1u64)];
        let ok = dpp.verify(&mut rng, &x, &[]).unwrap();
        assert!(ok);
    }

    #[test]
    fn test_packing_query_object_roundtrip_toy() {
        let flpcp = Toy3Query;
        let dpp = DppFromBoundedFlpcp::<F, _>::new(flpcp, PackedDppParams { ell: 32 });
        let mut rng = ChaCha20Rng::seed_from_u64(2);
        let x = vec![F::from(1u64)];
        let q = dpp.sample_query(&mut rng, &x).unwrap();
        let ok = dpp.verify_with_query(&x, &[], &q).unwrap();
        assert!(ok);
    }

    #[derive(Clone, Debug)]
    struct Toy3QuerySparse;

    impl BoundedFlpcpSparse<F> for Toy3QuerySparse {
        fn n(&self) -> usize { 1 }
        fn m(&self) -> usize { 0 }
        fn k(&self) -> usize { 3 }
        fn bounds_b(&self) -> Vec<BigInt> { vec![BigInt::from(2), BigInt::from(2), BigInt::from(2)] } // |a_i|<=1

        fn sample_queries_and_predicate_sparse(
            &self,
            _rng: &mut dyn RngCore,
            _x: &[F],
        ) -> Result<(Vec<SparseVec<F>>, FlpcpPredicate<F>), String> {
            Ok((
                vec![
                    SparseVec::new(vec![(F::one(), 0)]),
                    SparseVec::new(vec![(F::one(), 0)]),
                    SparseVec::new(vec![(F::one(), 0)]),
                ],
                FlpcpPredicate::MulEq,
            ))
        }
    }

    #[test]
    fn test_packing_sparse_roundtrip_toy() {
        let flpcp = Toy3QuerySparse;
        let dpp = DppFromBoundedFlpcpSparse::<F, _>::new(flpcp, PackedDppParams { ell: 32 });
        let mut rng = ChaCha20Rng::seed_from_u64(3);
        let x = vec![F::from(1u64)];
        let q = dpp.sample_query(&mut rng, &x).unwrap();
        let ok = dpp.verify_with_query(&x, &[], &q).unwrap();
        assert!(ok);
    }
}

