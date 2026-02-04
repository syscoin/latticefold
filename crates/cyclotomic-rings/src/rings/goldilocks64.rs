use ark_ff::{Field, PrimeField};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use stark_rings::balanced_decomposition::Decompose;
use stark_rings::cyclotomic_ring::models::goldilocks::{Fq, FqConfig};
use stark_rings::cyclotomic_ring::Flatten;
use stark_rings::traits::FromRandomBytes;
use stark_rings::traits::MulUnchecked;
use stark_rings::{OverField, PolyRing, Ring};
use std::sync::OnceLock;

use super::SuitableRing;
use super::goldilocks_ntt64 as ntt64;
use crate::{
    ark_base::*,
    challenge_set::{error, LatticefoldChallengeSet},
};

/// Goldilocks ring wrapper with **dimension 64** (coefficient representation).
///
/// This is \( \mathbb{F}_{\text{Goldilocks}}[X]/(X^{64}+1) \) in coefficient form.
/// Goldilocks prime \(p = 2^{64} - 2^{32} + 1\).
///
/// NOTE: This type stores elements in *coefficient form* (not CRT/NTT form). For performance,
/// multiplication is implemented via a negacyclic NTT using the shared `goldilocks_ntt64`
/// schedule (so host + tiny-gate agree on ordering/roots/stage conventions).
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

        // ---------------------------------------------------------------------
        // Fast paths (avoid NTT) for structurally simple operands.
        // ---------------------------------------------------------------------
        //
        // LF+/WE produces many ring elements that are:
        // - constant-coefficient lifts (via `R::from(base_scalar)`), and/or
        // - monomials (e.g. `X^j` or `c*X^j`).
        //
        // For coefficient-form rings, invoking the generic ring×ring `Mul` here would run an NTT
        // convolution, which is *much* slower than the O(d) scalar/shift operations below.
        #[inline(always)]
        fn is_const_coeff(v: &[Fq; N]) -> Option<Fq> {
            for &ci in &v[1..] {
                if ci != <Fq as Field>::ZERO {
                    return None;
                }
            }
            Some(v[0])
        }

        #[inline(always)]
        fn monomial(v: &[Fq; N]) -> Option<(usize, Fq)> {
            let mut idx: Option<usize> = None;
            let mut coeff = <Fq as Field>::ZERO;
            for (i, &ci) in v.iter().enumerate() {
                if ci != <Fq as Field>::ZERO {
                    if idx.is_some() {
                        return None;
                    }
                    idx = Some(i);
                    coeff = ci;
                }
            }
            idx.map(|i| (i, coeff))
        }

        // Pull coefficient arrays once (Copy).
        let mut a0 = [<Fq as Field>::ZERO; N];
        let mut b0 = [<Fq as Field>::ZERO; N];
        a0.copy_from_slice(self.coeffs());
        b0.copy_from_slice(rhs.coeffs());

        // Constant-coeff scalar multiply (O(d)).
        if let Some(c) = is_const_coeff(&b0) {
            return self * c;
        }
        if let Some(c) = is_const_coeff(&a0) {
            return rhs * c;
        }

        // Monomial multiply (O(d)): a(X) * (c*X^j) mod (X^64+1).
        if let Some((j, cj)) = monomial(&b0) {
            if cj == <Fq as Field>::ZERO {
                return GoldilocksRing64::ZERO;
            }
            let mut out = [<Fq as Field>::ZERO; N];
            for i in 0..N {
                let ai = a0[i];
                if ai == <Fq as Field>::ZERO {
                    continue;
                }
                let prod = ai * cj;
                let k = i + j;
                if k < N {
                    out[k] += prod;
                } else {
                    // X^N = -1 in (X^N+1).
                    out[k - N] -= prod;
                }
            }
            return GoldilocksRing64::from(out.to_vec());
        }
        if let Some((i, ci)) = monomial(&a0) {
            if ci == <Fq as Field>::ZERO {
                return GoldilocksRing64::ZERO;
            }
            let mut out = [<Fq as Field>::ZERO; N];
            for j in 0..N {
                let bj = b0[j];
                if bj == <Fq as Field>::ZERO {
                    continue;
                }
                let prod = ci * bj;
                let k = i + j;
                if k < N {
                    out[k] += prod;
                } else {
                    out[k - N] -= prod;
                }
            }
            return GoldilocksRing64::from(out.to_vec());
        }

        #[derive(Clone)]
        struct Precomp {
            bitrev: [usize; N],
            psi_pows: [Fq; N],
            psi_inv_pows: [Fq; N],
            inv_n: Fq,
            // stage twiddles for len = 2,4,8,16,32,64 (only prefix used per stage)
            w2: [Fq; 1],
            w4: [Fq; 2],
            w8: [Fq; 4],
            w16: [Fq; 8],
            w32: [Fq; 16],
            w64: [Fq; 32],
            iw2: [Fq; 1],
            iw4: [Fq; 2],
            iw8: [Fq; 4],
            iw16: [Fq; 8],
            iw32: [Fq; 16],
            iw64: [Fq; 32],
        }

        fn precomp() -> &'static Precomp {
            static PRE: OnceLock<Precomp> = OnceLock::new();
            PRE.get_or_init(|| {
                // Sanity: modulus matches the shared schedule.
                let p: u64 = <Fq as PrimeField>::MODULUS.0[0];
                debug_assert_eq!(p, ntt64::GOLDILOCKS_P_U64);

                #[inline(always)]
                fn fq_arr<const M: usize>(xs: &[u64; M]) -> [Fq; M] {
                    core::array::from_fn(|i| Fq::from(xs[i]))
                }

                let inv_n = Fq::from(ntt64::INV_N_U64);

                // Bitrev table.
                let bitrev = ntt64::BITREV_64;

                // psi^i and psi^{-i}
                let psi_pows = fq_arr::<64>(&ntt64::PSI_POWS_64);
                let psi_inv_pows = fq_arr::<64>(&ntt64::PSI_INV_POWS_64);

                Precomp {
                    bitrev,
                    psi_pows,
                    psi_inv_pows,
                    inv_n,
                    w2: fq_arr::<1>(&ntt64::W_POWS_LEN_2),
                    w4: fq_arr::<2>(&ntt64::W_POWS_LEN_4),
                    w8: fq_arr::<4>(&ntt64::W_POWS_LEN_8),
                    w16: fq_arr::<8>(&ntt64::W_POWS_LEN_16),
                    w32: fq_arr::<16>(&ntt64::W_POWS_LEN_32),
                    w64: fq_arr::<32>(&ntt64::W_POWS_LEN_64),
                    iw2: fq_arr::<1>(&ntt64::IW_POWS_LEN_2),
                    iw4: fq_arr::<2>(&ntt64::IW_POWS_LEN_4),
                    iw8: fq_arr::<4>(&ntt64::IW_POWS_LEN_8),
                    iw16: fq_arr::<8>(&ntt64::IW_POWS_LEN_16),
                    iw32: fq_arr::<16>(&ntt64::IW_POWS_LEN_32),
                    iw64: fq_arr::<32>(&ntt64::IW_POWS_LEN_64),
                }
            })
        }

        #[inline(always)]
        fn bitrev_permute(dst: &mut [Fq; N], src: &[Fq; N], bitrev: &[usize; N]) {
            // dst[bitrev(i)] = src[i]
            for i in 0..N {
                dst[bitrev[i]] = src[i];
            }
        }

        #[inline(always)]
        fn stage<const HALF: usize>(a: &mut [Fq; N], len: usize, w: &[Fq; HALF]) {
            // len = 2*HALF
            debug_assert_eq!(len, 2 * HALF);
            for i in (0..N).step_by(len) {
                for j in 0..HALF {
                    let u = a[i + j];
                    let v = a[i + j + HALF] * w[j];
                    a[i + j] = u + v;
                    a[i + j + HALF] = u - v;
                }
            }
        }

        let mut a = [<Fq as Field>::ZERO; N];
        let mut b = [<Fq as Field>::ZERO; N];

        let pc = precomp();

        // Twist (mul by psi^i) and bitrev permute into working buffers.
        for i in 0..N {
            a0[i] *= pc.psi_pows[i];
            b0[i] *= pc.psi_pows[i];
        }
        bitrev_permute(&mut a, &a0, &pc.bitrev);
        bitrev_permute(&mut b, &b0, &pc.bitrev);

        // Forward NTT (fixed stages).
        stage::<1>(&mut a, 2, &pc.w2);
        stage::<2>(&mut a, 4, &pc.w4);
        stage::<4>(&mut a, 8, &pc.w8);
        stage::<8>(&mut a, 16, &pc.w16);
        stage::<16>(&mut a, 32, &pc.w32);
        stage::<32>(&mut a, 64, &pc.w64);

        stage::<1>(&mut b, 2, &pc.w2);
        stage::<2>(&mut b, 4, &pc.w4);
        stage::<4>(&mut b, 8, &pc.w8);
        stage::<8>(&mut b, 16, &pc.w16);
        stage::<16>(&mut b, 32, &pc.w32);
        stage::<32>(&mut b, 64, &pc.w64);

        for i in 0..N {
            a[i] *= b[i];
        }

        // Inverse NTT: bitrev permute then same stage structure with inverse roots.
        // We already have bitrev order (output of forward). For inverse with this DIT layout,
        // we can run the same butterfly structure with inverse roots on the bitrev-permuted data.
        let mut a_inv_in = [<Fq as Field>::ZERO; N];
        a_inv_in.copy_from_slice(&a);
        bitrev_permute(&mut a, &a_inv_in, &pc.bitrev);

        stage::<1>(&mut a, 2, &pc.iw2);
        stage::<2>(&mut a, 4, &pc.iw4);
        stage::<4>(&mut a, 8, &pc.iw8);
        stage::<8>(&mut a, 16, &pc.iw16);
        stage::<16>(&mut a, 32, &pc.iw32);
        stage::<32>(&mut a, 64, &pc.iw64);

        // Scale by 1/N and untwist by psi^{-i}.
        for i in 0..N {
            a[i] *= pc.inv_n;
            a[i] *= pc.psi_inv_pows[i];
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

    fn monomial(j: usize, c: Fq) -> GoldilocksRing64 {
        assert!(j < 64);
        let mut v = vec![<Fq as Field>::ZERO; 64];
        v[j] = c;
        GoldilocksRing64::from(v)
    }

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
    fn test_goldilocks64_mul_by_constant_hits_fast_path_and_matches_inner() {
        let mut rng = test_rng();
        for _ in 0..256 {
            let a = GoldilocksRing64::rand(&mut rng);
            let c = Fq::rand(&mut rng);
            let b = GoldilocksRing64::from_scalar(c);

            // Reference: underlying generic ring multiplication.
            let ref_out = Inner::from(a.into_coeffs()) * Inner::from(b.into_coeffs());
            let got = a * b;
            assert_eq!(got.coeffs(), ref_out.coeffs());
        }
    }

    #[test]
    fn test_goldilocks64_mul_by_monomial_hits_fast_path_and_matches_inner() {
        let mut rng = test_rng();
        for _ in 0..256 {
            let a = GoldilocksRing64::rand(&mut rng);
            let j = (rng.next_u64() as usize) & 63;
            let c = Fq::rand(&mut rng);
            let b = monomial(j, c);

            // Reference: underlying generic ring multiplication.
            let ref_out = Inner::from(a.into_coeffs()) * Inner::from(b.into_coeffs());
            let got = a * b;
            assert_eq!(got.coeffs(), ref_out.coeffs());

            // Also exercise the other-direction monomial fast path.
            let a2 = GoldilocksRing64::rand(&mut rng);
            let ref_out2 = Inner::from(b.into_coeffs()) * Inner::from(a2.into_coeffs());
            let got2 = b * a2;
            assert_eq!(got2.coeffs(), ref_out2.coeffs());
        }
    }

    #[test]
    fn test_goldilocks64_mul_by_monomial_wrap_cases() {
        // Deterministic wrap-around sanity checks for j near N.
        // In R = Fq[X]/(X^64+1), X^64 = -1, so X^63 * X = -1.
        let one = <Fq as Field>::ONE;
        let x = monomial(1, one);
        let x63 = monomial(63, one);
        let got = x63 * x;

        // Expect constant -1.
        let mut exp = vec![<Fq as Field>::ZERO; 64];
        exp[0] = -one;
        assert_eq!(got.coeffs(), GoldilocksRing64::from(exp).coeffs());

        // And X^63 * X^63 = X^126 = X^(64+62) = -X^62.
        let got2 = x63 * x63;
        let mut exp2 = vec![<Fq as Field>::ZERO; 64];
        exp2[62] = -one;
        assert_eq!(got2.coeffs(), GoldilocksRing64::from(exp2).coeffs());
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

