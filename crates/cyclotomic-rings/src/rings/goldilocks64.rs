use ark_ff::{Field, PrimeField};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use stark_rings::balanced_decomposition::Decompose;
use stark_rings::cyclotomic_ring::models::goldilocks::{Fq, FqConfig};
use stark_rings::cyclotomic_ring::Flatten;
use stark_rings::traits::FromRandomBytes;
use stark_rings::traits::MulUnchecked;
use stark_rings::{OverField, PolyRing, Ring};

use super::SuitableRing;
use crate::{
    ark_base::*,
    challenge_set::{error, LatticefoldChallengeSet},
};

/// Goldilocks ring wrapper with **dimension 64** (coefficient representation).
///
/// This is \( \mathbb{F}_{\text{Goldilocks}}[X]/(X^{64}+1) \) in coefficient form.
/// It is intended to be the drop-in analogue of `FrogRing64`, but over the true
/// Goldilocks prime \(p = 2^{64} - 2^{32} + 1\).
///
/// NOTE: This type is *coefficient-form* (not CRT/NTT form). It still benefits LF+/WE
/// because it makes the protocol parameterization and transcript scheduling "pow2-d=64"
/// over Goldilocks explicit; NTT-friendly multiplication can be added later on top of
/// this correct algebraic baseline.
#[repr(transparent)]
#[derive(
    Copy, Clone, Debug, Default, Eq, PartialEq, Hash, CanonicalSerialize, CanonicalDeserialize,
)]
pub struct GoldilocksRing64(
    pub stark_rings::cyclotomic_ring::CyclotomicPolyRingGeneral<Goldilocks64Config, 1, 64>,
);

/// Parameters for \( \mathbb{F}_p[X]/(X^{64}+1) \) over Goldilocks' base prime field.
pub struct Goldilocks64Config;

impl stark_rings::cyclotomic_ring::CyclotomicConfig<1> for Goldilocks64Config {
    type BaseFieldConfig = ark_ff::MontBackend<FqConfig, 1>;
    type BaseCRTField = Fq;
    const CRT_FIELD_EXTENSION_DEGREE: usize = 1;

    fn reduce_in_place(coefficients: &mut Vec<Fq>) {
        // Reduce mod (X^64 + 1): x^{64+i} = -x^i.
        if coefficients.len() > 64 {
            let (left, right) = coefficients.split_at_mut(64);
            for (a, b) in left.iter_mut().zip(right.iter()) {
                *a -= *b;
            }
        }
        coefficients.resize(64, <Fq as Field>::ZERO);
    }

    fn crt_in_place(_coefficients: &mut [Fq]) {}
    fn crt(coefficients: Vec<Fq>) -> Vec<Fq> {
        coefficients
    }
    fn icrt(evaluations: Vec<Fq>) -> Vec<Fq> {
        evaluations
    }
    fn icrt_in_place(_evaluations: &mut [Fq]) {}
}

// ---- Forwarding impls to satisfy `stark_rings::Ring` + `stark_rings::PolyRing` ----
macro_rules! fwd_binop {
    ($trait:ident, $method:ident, $op:tt) => {
        impl core::ops::$trait for GoldilocksRing64 {
            type Output = Self;
            #[inline(always)]
            fn $method(self, rhs: Self) -> Self::Output {
                Self(self.0 $op rhs.0)
            }
        }
        impl<'a> core::ops::$trait<&'a Self> for GoldilocksRing64 {
            type Output = Self;
            #[inline(always)]
            fn $method(self, rhs: &'a Self) -> Self::Output {
                Self(self.0 $op rhs.0)
            }
        }
        impl<'a> core::ops::$trait<&'a mut Self> for GoldilocksRing64 {
            type Output = Self;
            #[inline(always)]
            fn $method(self, rhs: &'a mut Self) -> Self::Output {
                Self(self.0 $op rhs.0)
            }
        }
    };
}
macro_rules! fwd_binop_assign {
    ($trait:ident, $method:ident, $op:tt) => {
        impl core::ops::$trait for GoldilocksRing64 {
            #[inline(always)]
            fn $method(&mut self, rhs: Self) {
                self.0 $op rhs.0;
            }
        }
        impl<'a> core::ops::$trait<&'a Self> for GoldilocksRing64 {
            #[inline(always)]
            fn $method(&mut self, rhs: &'a Self) {
                self.0 $op rhs.0;
            }
        }
        impl<'a> core::ops::$trait<&'a mut Self> for GoldilocksRing64 {
            #[inline(always)]
            fn $method(&mut self, rhs: &'a mut Self) {
                self.0 $op rhs.0;
            }
        }
    };
}

fwd_binop!(Add, add, +);
fwd_binop!(Sub, sub, -);
fwd_binop_assign!(AddAssign, add_assign, +=);
fwd_binop_assign!(SubAssign, sub_assign, -=);

// -----------------------------------------------------------------------------
// GoldilocksRing64 multiplication (performance-critical)
// -----------------------------------------------------------------------------
//
// We implement multiplication in R = Fq[X]/(X^64 + 1) using a negacyclic NTT:
// - pick ψ a primitive 128th root of unity (so ω=ψ^2 is a primitive 64th root)
// - "twist" coefficients by ψ^i
// - do a size-64 cyclic NTT with ω
// - pointwise multiply
// - inverse NTT, scale by 1/64
// - "untwist" by ψ^{-i}
//
// This is the canonical multiply we want for GL64 (no secondary Karatsuba path kept).
impl core::ops::Mul for GoldilocksRing64 {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        const N: usize = 64;
        const LOG_N: u32 = 6;

        #[inline(always)]
        fn pow_u64(mut base: Fq, mut exp: u64) -> Fq {
            let mut acc = <Fq as Field>::ONE;
            while exp != 0 {
                if (exp & 1) == 1 {
                    acc *= base;
                }
                base *= base;
                exp >>= 1;
            }
            acc
        }

        #[inline(always)]
        fn bitrev6(i: usize) -> usize {
            // Reverse 6 bits of i.
            ((i as u32).reverse_bits() >> (32 - LOG_N)) as usize
        }

        #[inline]
        fn ntt_in_place(a: &mut [Fq; N], omega: Fq) {
            // Bit-reversal permutation.
            for i in 0..N {
                let j = bitrev6(i);
                if i < j {
                    a.swap(i, j);
                }
            }

            let mut len = 2usize;
            while len <= N {
                // wlen = omega^(N/len)
                let wlen = pow_u64(omega, (N / len) as u64);
                for i in (0..N).step_by(len) {
                    let mut w = <Fq as Field>::ONE;
                    for j in 0..(len / 2) {
                        let u = a[i + j];
                        let v = a[i + j + len / 2] * w;
                        a[i + j] = u + v;
                        a[i + j + len / 2] = u - v;
                        w *= wlen;
                    }
                }
                len <<= 1;
            }
        }

        #[inline]
        fn intt_in_place(a: &mut [Fq; N], omega_inv: Fq) {
            // Bit-reversal permutation.
            for i in 0..N {
                let j = bitrev6(i);
                if i < j {
                    a.swap(i, j);
                }
            }

            let mut len = 2usize;
            while len <= N {
                let wlen = pow_u64(omega_inv, (N / len) as u64);
                for i in (0..N).step_by(len) {
                    let mut w = <Fq as Field>::ONE;
                    for j in 0..(len / 2) {
                        let u = a[i + j];
                        let v = a[i + j + len / 2] * w;
                        a[i + j] = u + v;
                        a[i + j + len / 2] = u - v;
                        w *= wlen;
                    }
                }
                len <<= 1;
            }
        }

        // Goldilocks base field modulus p as u64.
        let p: u64 = <Fq as PrimeField>::MODULUS.0[0];
        debug_assert_eq!(p, 0xFFFF_FFFF_0000_0001u64);

        // Generator is 7 in the stark-rings Goldilocks field config.
        let g = Fq::from(7u64);
        let psi = pow_u64(g, (p - 1) / 128);
        let omega = psi * psi; // primitive 64th root
        let omega_inv = omega.inverse().expect("omega inverse");
        let psi_inv = psi.inverse().expect("psi inverse");
        let inv_n = Fq::from(N as u64).inverse().expect("inv_n");

        let mut a = [<Fq as Field>::ZERO; N];
        let mut b = [<Fq as Field>::ZERO; N];
        a.copy_from_slice(self.coeffs());
        b.copy_from_slice(rhs.coeffs());

        // Twist by psi^i.
        let mut psi_pow = <Fq as Field>::ONE;
        for i in 0..N {
            a[i] *= psi_pow;
            b[i] *= psi_pow;
            psi_pow *= psi;
        }

        ntt_in_place(&mut a, omega);
        ntt_in_place(&mut b, omega);

        for i in 0..N {
            a[i] *= b[i];
        }

        intt_in_place(&mut a, omega_inv);

        // Scale by 1/N and untwist by psi^{-i}.
        let mut psi_inv_pow = <Fq as Field>::ONE;
        for i in 0..N {
            a[i] *= inv_n;
            a[i] *= psi_inv_pow;
            psi_inv_pow *= psi_inv;
        }

        GoldilocksRing64::from(a.to_vec())
    }
}

impl<'a> core::ops::Mul<&'a Self> for GoldilocksRing64 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: &'a Self) -> Self::Output {
        self * (*rhs)
    }
}
impl<'a> core::ops::Mul<&'a mut Self> for GoldilocksRing64 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: &'a mut Self) -> Self::Output {
        self * (*rhs)
    }
}

impl core::ops::MulAssign for GoldilocksRing64 {
    #[inline(always)]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}
impl<'a> core::ops::MulAssign<&'a Self> for GoldilocksRing64 {
    #[inline(always)]
    fn mul_assign(&mut self, rhs: &'a Self) {
        *self = *self * rhs;
    }
}
impl<'a> core::ops::MulAssign<&'a mut Self> for GoldilocksRing64 {
    #[inline(always)]
    fn mul_assign(&mut self, rhs: &'a mut Self) {
        *self = *self * rhs;
    }
}

impl core::ops::Neg for GoldilocksRing64 {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self::Output {
        Self(-self.0)
    }
}

impl ark_std::fmt::Display for GoldilocksRing64 {
    fn fmt(&self, f: &mut ark_std::fmt::Formatter<'_>) -> ark_std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl ark_std::iter::Sum for GoldilocksRing64 {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::ZERO, |acc, x| acc + x)
    }
}
impl<'a> ark_std::iter::Sum<&'a Self> for GoldilocksRing64 {
    fn sum<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
        iter.fold(Self::ZERO, |acc, x| acc + x)
    }
}
impl ark_std::iter::Product for GoldilocksRing64 {
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::ONE, |acc, x| acc * x)
    }
}
impl<'a> ark_std::iter::Product<&'a Self> for GoldilocksRing64 {
    fn product<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
        iter.fold(Self::ONE, |acc, x| acc * x)
    }
}

impl ark_std::Zero for GoldilocksRing64 {
    #[inline(always)]
    fn zero() -> Self {
        Self(<stark_rings::cyclotomic_ring::CyclotomicPolyRingGeneral<Goldilocks64Config, 1, 64> as Ring>::ZERO)
    }
    #[inline(always)]
    fn is_zero(&self) -> bool {
        self.0.is_zero()
    }
}
impl ark_std::One for GoldilocksRing64 {
    #[inline(always)]
    fn one() -> Self {
        Self(<stark_rings::cyclotomic_ring::CyclotomicPolyRingGeneral<Goldilocks64Config, 1, 64> as Ring>::ONE)
    }
}

impl ark_std::UniformRand for GoldilocksRing64 {
    fn rand<R: rand::Rng + ?Sized>(rng: &mut R) -> Self {
        Self(stark_rings::cyclotomic_ring::CyclotomicPolyRingGeneral::<Goldilocks64Config, 1, 64>::rand(rng))
    }
}

impl FromRandomBytes<Self> for GoldilocksRing64 {
    fn byte_size() -> usize {
        stark_rings::cyclotomic_ring::CyclotomicPolyRingGeneral::<Goldilocks64Config, 1, 64>::byte_size()
    }
    fn try_from_random_bytes(bytes: &[u8]) -> Option<Self> {
        stark_rings::cyclotomic_ring::CyclotomicPolyRingGeneral::<Goldilocks64Config, 1, 64>::try_from_random_bytes(bytes)
            .map(Self)
    }
}

impl Ring for GoldilocksRing64 {
    const ZERO: Self =
        Self(<stark_rings::cyclotomic_ring::CyclotomicPolyRingGeneral<Goldilocks64Config, 1, 64> as Ring>::ZERO);
    const ONE: Self =
        Self(<stark_rings::cyclotomic_ring::CyclotomicPolyRingGeneral<Goldilocks64Config, 1, 64> as Ring>::ONE);
}

impl From<u128> for GoldilocksRing64 {
    fn from(value: u128) -> Self {
        Self(stark_rings::cyclotomic_ring::CyclotomicPolyRingGeneral::<Goldilocks64Config, 1, 64>::from(value))
    }
}
impl From<u64> for GoldilocksRing64 {
    fn from(value: u64) -> Self {
        Self::from(value as u128)
    }
}
impl From<u32> for GoldilocksRing64 {
    fn from(value: u32) -> Self {
        Self::from(value as u128)
    }
}
impl From<u16> for GoldilocksRing64 {
    fn from(value: u16) -> Self {
        Self::from(value as u128)
    }
}
impl From<u8> for GoldilocksRing64 {
    fn from(value: u8) -> Self {
        Self::from(value as u128)
    }
}
impl From<bool> for GoldilocksRing64 {
    fn from(value: bool) -> Self {
        Self::from(value as u128)
    }
}

impl From<Fq> for GoldilocksRing64 {
    fn from(value: Fq) -> Self {
        Self(stark_rings::cyclotomic_ring::CyclotomicPolyRingGeneral::<Goldilocks64Config, 1, 64>::from(value))
    }
}

impl From<Vec<Fq>> for GoldilocksRing64 {
    fn from(value: Vec<Fq>) -> Self {
        Self(stark_rings::cyclotomic_ring::CyclotomicPolyRingGeneral::<Goldilocks64Config, 1, 64>::from(value))
    }
}

impl PolyRing for GoldilocksRing64 {
    type BaseRing = Fq;

    fn coeffs(&self) -> &[Self::BaseRing] {
        self.0.coeffs()
    }
    fn coeffs_mut(&mut self) -> &mut [Self::BaseRing] {
        self.0.coeffs_mut()
    }
    fn into_coeffs(self) -> Vec<Self::BaseRing> {
        self.0.into_coeffs()
    }
    fn dimension() -> usize {
        64
    }
    fn from_scalar(scalar: Self::BaseRing) -> Self {
        Self(stark_rings::cyclotomic_ring::CyclotomicPolyRingGeneral::<Goldilocks64Config, 1, 64>::from_scalar(
            scalar,
        ))
    }
}

impl Flatten for GoldilocksRing64 {}

impl core::ops::Mul<Fq> for GoldilocksRing64 {
    type Output = Self;
    fn mul(self, rhs: Fq) -> Self::Output {
        let mut out = self;
        for c in out.coeffs_mut() {
            *c *= rhs;
        }
        out
    }
}

impl OverField for GoldilocksRing64 {}

impl stark_rings::Cyclotomic for GoldilocksRing64 {
    fn rot(&mut self) {
        let d = <Self as PolyRing>::dimension();
        let mut buf = -self.coeffs()[d - 1];
        for i in 0..d {
            ark_std::mem::swap(&mut buf, &mut self.coeffs_mut()[i]);
        }
    }
}

impl stark_rings::cyclotomic_ring::CRT for GoldilocksRing64 {
    type CRTForm = Self;
    fn crt(self) -> Self::CRTForm {
        self
    }
}
impl stark_rings::cyclotomic_ring::ICRT for GoldilocksRing64 {
    type ICRTForm = Self;
    fn icrt(self) -> Self::ICRTForm {
        self
    }
}

impl SuitableRing for GoldilocksRing64 {
    type CoefficientRepresentation = GoldilocksRing64;
    type PoseidonParams = super::goldilocks::GoldilocksPoseidonConfig;
}

impl<'a> core::ops::MulAssign<&'a u128> for GoldilocksRing64 {
    fn mul_assign(&mut self, rhs: &'a u128) {
        self.0 *= rhs;
    }
}

impl MulUnchecked for GoldilocksRing64 {
    type Output = Self;
    fn mul_unchecked(self, rhs: Self) -> Self::Output {
        // Keep `mul_unchecked` consistent with `Mul` (we're already in coefficient form).
        self * rhs
    }
}

impl Decompose for GoldilocksRing64 {
    fn decompose_to(&self, b: u128, out: &mut [Self]) {
        type Inner =
            stark_rings::cyclotomic_ring::CyclotomicPolyRingGeneral<Goldilocks64Config, 1, 64>;

        use std::cell::RefCell;
        thread_local! {
            static SCRATCH: RefCell<Vec<Inner>> = const { RefCell::new(Vec::new()) };
        }
        SCRATCH.with(|cell| {
            let mut buf = cell.borrow_mut();
            if buf.len() != out.len() {
                buf.resize(out.len(), Inner::ZERO);
            }
            self.0.decompose_to(b, &mut buf[..]);
            for (o, t) in out.iter_mut().zip(buf.iter()) {
                *o = GoldilocksRing64(t.clone());
            }
        });
    }
}

#[derive(Clone)]
pub struct Goldilocks64ChallengeSet;

// Keep the same "[-128, 128)" short-challenge shape as `FrogRing64` to make it easy to
// swap rings without touching higher-level protocol settings (k, bounds, etc.).
impl LatticefoldChallengeSet<GoldilocksRing64> for Goldilocks64ChallengeSet {
    const BYTES_NEEDED: usize = 64;

    fn short_challenge_from_random_bytes(
        bs: &[u8],
    ) -> Result<
        <GoldilocksRing64 as SuitableRing>::CoefficientRepresentation,
        crate::challenge_set::error::ChallengeSetError,
    > {
        if bs.len() != Self::BYTES_NEEDED {
            return Err(error::ChallengeSetError::TooFewBytes(
                bs.len(),
                Self::BYTES_NEEDED,
            ));
        }

        Ok(GoldilocksRing64::from(
            bs.iter()
                .map(|&x| Fq::from(x as i16 - 128))
                .collect::<Vec<Fq>>(),
        ))
    }
}

#[cfg(test)]
mod goldilocks64_tests {
    use super::*;
    use ark_std::{test_rng, UniformRand};
    use rand::RngCore;

    type Inner =
        stark_rings::cyclotomic_ring::CyclotomicPolyRingGeneral<Goldilocks64Config, 1, 64>;

    #[test]
    fn test_goldilocks64_mul_matches_inner_generic() {
        let mut rng = test_rng();
        for _ in 0..256 {
            let a = GoldilocksRing64::rand(&mut rng);
            let b = GoldilocksRing64::rand(&mut rng);

            // Reference: underlying generic ring multiplication.
            let ref_out = Inner::from(a.into_coeffs()) * Inner::from(b.into_coeffs());
            let got = a * b;

            assert_eq!(got.coeffs(), ref_out.coeffs());
        }
    }

    #[test]
    fn test_goldilocks64_mul_by_scalar_in_place() {
        let mut rng = test_rng();
        let a = GoldilocksRing64::rand(&mut rng);
        let s = Fq::rand(&mut rng);
        let got = a * s;

        // Reference via inner generic multiplication by scalar (should match).
        let ref_out = Inner::from(a.into_coeffs()) * s;
        assert_eq!(got.coeffs(), ref_out.coeffs());
    }

    #[test]
    fn test_goldilocks64_decompose_to_matches_inner() {
        let mut rng = test_rng();
        for _ in 0..64 {
            let a = GoldilocksRing64::rand(&mut rng);
            let b = ((rng.next_u64() as u128) | 2) & !1; // even b >= 2
            let len = 8usize;

            let mut out = vec![GoldilocksRing64::ZERO; len];
            a.decompose_to(b, &mut out);

            let mut out_ref = vec![Inner::ZERO; len];
            Inner::from(a.into_coeffs()).decompose_to(b, &mut out_ref);

            for (got, exp) in out.iter().zip(out_ref.iter()) {
                assert_eq!(got.coeffs(), exp.coeffs());
            }
        }
    }

    /// Rough timing smoke test (intentionally **not** ignored).
    ///
    /// This is not a benchmark harness; it’s just a sanity check that `GoldilocksRing64` multiply
    /// stays in a reasonable ballpark.
    #[test]
    fn test_goldilocks64_mul_timing_smoke() {
        use std::time::Instant;
        use ark_std::Zero;
        use stark_rings::Ring;

        let mut rng = test_rng();
        let iters: usize = 50_000;

        let mut acc = GoldilocksRing64::ONE;
        let a = GoldilocksRing64::rand(&mut rng);
        let b = GoldilocksRing64::rand(&mut rng);
        let t0 = Instant::now();
        for _ in 0..iters {
            acc *= a;
            acc *= b;
        }
        let dt = t0.elapsed();

        // Keep the value live.
        assert!(!acc.is_zero());
        eprintln!("[goldilocks64_smoke] iters={} mul64={:?}", iters, dt);
    }
}

