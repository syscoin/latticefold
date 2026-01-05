//! Shared Symphony transcript coin derivations and common helpers.
//!
//! This module centralizes helper routines that were previously duplicated across
//! `Π_rg`, `Π_gr1cs`, and `Π_fold` implementations.

use latticefold::transcript::Transcript;
use stark_rings::{OverField, PolyRing, Ring};

/// Build tensor weights `ts(r)` matching this codebase's MLE variable ordering.
///
/// `DenseMultilinearExtension::evaluate` in this repo treats the first point coordinate
/// as the least-significant bit, so we reverse the point vector before calling `tensor`.
pub fn ts_weights<F>(point: &[F]) -> Vec<F>
where
    F: Copy + ark_std::One + core::ops::Sub<Output = F> + core::ops::Mul<Output = F>,
{
    let rev = point.iter().rev().copied().collect::<Vec<_>>();
    crate::utils::tensor(&rev)
}

/// Derive a χ-style small projection matrix J ∈ {−1,0,+1}^{λ_pj×ℓ_h} from the transcript.
///
/// Distribution: Pr[0]=1/2, Pr[±1]=1/4 (via 2-bit buckets).
pub fn derive_J<R: OverField>(
    transcript: &mut impl Transcript<R>,
    lambda_pj: usize,
    l_h: usize,
) -> Vec<Vec<R::BaseRing>> {
    // Domain separation tag.
    transcript.absorb_field_element(&R::BaseRing::from(0x4c465053_534a5631u128)); // "LFPS_SJV1"

    let bytes = transcript.squeeze_bytes(lambda_pj * l_h);
    let one = <R::BaseRing as Ring>::ONE;
    let neg_one = -one;

    let mut J = vec![vec![<R::BaseRing as Ring>::ZERO; l_h]; lambda_pj];
    for i in 0..lambda_pj {
        for j in 0..l_h {
            J[i][j] = match bytes[i * l_h + j] & 0b11 {
                0b01 => one,
                0b10 => neg_one,
                _ => <R::BaseRing as Ring>::ZERO,
            };
        }
    }
    J
}

/// Derive χ-style low-norm folding coefficients β ∈ K^ℓ from the transcript.
///
/// Distribution: Pr[0]=1/2, Pr[±1]=1/4 (via 2-bit buckets).
///
/// This is used by Π_fold (Figure 4) as verifier coins.
pub fn derive_beta_chi<R: OverField>(
    transcript: &mut impl Transcript<R>,
    ell: usize,
) -> Vec<R::BaseRing> {
    transcript.absorb_field_element(&R::BaseRing::from(0x4c465053_42455441u128)); // "LFPS_BETA"

    let bytes = transcript.squeeze_bytes(ell);
    let one = <R::BaseRing as Ring>::ONE;
    let neg_one = -one;

    let mut beta = Vec::with_capacity(ell);
    for b in bytes {
        let x = match b & 0b11 {
            0b01 => one,
            0b10 => neg_one,
            _ => <R::BaseRing as Ring>::ZERO,
        };
        // Absorb each β_i so later coins depend on it.
        transcript.absorb_field_element(&x);
        beta.push(x);
    }
    beta
}

/// Evaluate a ring element at x in the base field.
///
/// This matches the polynomial evaluation used by the set-check machinery.
pub fn ev<R: PolyRing>(r: &R, x: R::BaseRing) -> R::BaseRing {
    r.coeffs()
        .iter()
        .fold((R::BaseRing::ZERO, R::BaseRing::ONE), |(mut acc, exp), c| {
            acc += *c * exp;
            (acc, exp * x)
        })
        .0
}
