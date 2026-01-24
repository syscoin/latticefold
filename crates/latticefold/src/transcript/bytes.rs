//! Canonical byte encodings for transcript absorption.
//!
//! Goal: make transcript IO bytes-first and deterministic across prover/verifier.
//! We use fixed-width little-endian encodings for prime fields.

use ark_ff::{BigInteger, Field, PrimeField};
use ark_std::vec::Vec;
use stark_rings::OverField;

/// Fixed-width little-endian encoding of a field element via its base prime field elements.
///
/// For a prime field this is identical to `prime_field_to_bytes_le_fixed`.
/// For an extension field `F`, this encodes `to_base_prime_field_elements()` in order.
#[inline]
pub fn field_to_bytes_le_fixed<F: Field>(x: &F) -> Vec<u8> {
    let bp = x.to_base_prime_field_elements().collect::<Vec<<F as Field>::BasePrimeField>>();
    prime_field_slice_to_bytes_le_fixed::<<F as Field>::BasePrimeField>(&bp)
}

/// Fixed-width little-endian encoding of a prime-field element.
///
/// Width is `ceil(MODULUS_BIT_SIZE/8)` bytes.
#[inline]
pub fn prime_field_to_bytes_le_fixed<F: PrimeField>(x: &F) -> Vec<u8> {
    let nbytes = ((F::MODULUS_BIT_SIZE as usize) + 7) / 8;
    let mut out = vec![0u8; nbytes];
    let le = x.into_bigint().to_bytes_le();
    let m = le.len().min(nbytes);
    out[..m].copy_from_slice(&le[..m]);
    out
}

/// Fixed-width little-endian encoding of a slice of prime-field elements.
#[inline]
pub fn prime_field_slice_to_bytes_le_fixed<F: PrimeField>(xs: &[F]) -> Vec<u8> {
    let nbytes = ((F::MODULUS_BIT_SIZE as usize) + 7) / 8;
    let mut out = Vec::with_capacity(xs.len() * nbytes);
    for x in xs {
        out.extend_from_slice(&prime_field_to_bytes_le_fixed::<F>(x));
    }
    out
}

/// Fixed-width little-endian encoding of a matrix (row-major).
#[inline]
pub fn prime_field_matrix_to_bytes_le_fixed<F: PrimeField>(rows: &[Vec<F>]) -> Vec<u8> {
    let nbytes = ((F::MODULUS_BIT_SIZE as usize) + 7) / 8;
    let mut out = Vec::new();
    for row in rows {
        out.reserve(row.len() * nbytes);
        for x in row {
            out.extend_from_slice(&prime_field_to_bytes_le_fixed::<F>(x));
        }
    }
    out
}

/// Canonical encoding of a ring element by concatenating fixed-width encodings
/// of its base-ring coefficients (little-endian).
#[inline]
pub fn ring_to_bytes_le_fixed<R: OverField>(r: &R) -> Vec<u8>
where
    R::BaseRing: Field,
{
    let nbytes_bp = ((<<R::BaseRing as Field>::BasePrimeField as PrimeField>::MODULUS_BIT_SIZE as usize) + 7) / 8;
    // Coefficient may be an extension field; multiply by its extension degree in base-prime elements.
    let ext_deg = <R::BaseRing as Field>::extension_degree() as usize;
    let mut out = Vec::with_capacity(r.coeffs().len() * ext_deg * nbytes_bp);
    for c in r.coeffs() {
        out.extend_from_slice(&field_to_bytes_le_fixed::<R::BaseRing>(c));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use cyclotomic_rings::rings::GoldilocksRingNTT;
    use stark_rings::cyclotomic_ring::Flatten;
    use stark_rings::PolyRing;

    type R = GoldilocksRingNTT;
    type BR = <R as PolyRing>::BaseRing;
    type BPF = <BR as Field>::BasePrimeField;

    #[test]
    fn test_prime_field_to_bytes_le_fixed_is_fixed_width() {
        let nbytes = ((BPF::MODULUS_BIT_SIZE as usize) + 7) / 8;
        let x = BPF::from(0x0102_0304_0506_0708u64);
        let b = prime_field_to_bytes_le_fixed::<BPF>(&x);
        assert_eq!(b.len(), nbytes);

        // Should be little-endian with zero-padding up to fixed width.
        let le = x.into_bigint().to_bytes_le();
        assert!(le.len() <= nbytes);
        assert_eq!(&b[..le.len()], &le[..]);
        assert!(b[le.len()..].iter().all(|&t| t == 0));
    }

    #[test]
    fn test_prime_field_slice_to_bytes_is_concatenation() {
        let nbytes = ((BPF::MODULUS_BIT_SIZE as usize) + 7) / 8;
        let a = BPF::from(1u64);
        let b = BPF::from(2u64);
        let out = prime_field_slice_to_bytes_le_fixed::<BPF>(&[a, b]);
        assert_eq!(out.len(), 2 * nbytes);
        assert_eq!(&out[..nbytes], &prime_field_to_bytes_le_fixed::<BPF>(&a));
        assert_eq!(&out[nbytes..], &prime_field_to_bytes_le_fixed::<BPF>(&b));
    }

    #[test]
    fn test_field_to_bytes_matches_base_prime_elements() {
        let x = BR::from(0xDEAD_BEEFu64);
        let bp = x
            .to_base_prime_field_elements()
            .collect::<Vec<<BR as Field>::BasePrimeField>>();
        let expected = prime_field_slice_to_bytes_le_fixed::<BPF>(&bp);
        let got = field_to_bytes_le_fixed::<BR>(&x);
        assert_eq!(got, expected);
    }

    #[test]
    fn test_ring_to_bytes_matches_coeff_encodings() {
        let d = R::dimension();
        let mut coeffs = vec![BR::ZERO; d];
        coeffs[0] = BR::from(1u64);
        if d > 1 {
            coeffs[1] = BR::from(2u64);
        }

        let rs = R::promote_from_coeffs(coeffs).expect("promote_from_coeffs failed");
        assert_eq!(rs.len(), 1, "expected a single ring element");
        let r = rs[0];

        let mut expected = Vec::new();
        for c in r.coeffs() {
            expected.extend_from_slice(&field_to_bytes_le_fixed::<BR>(c));
        }
        let got = ring_to_bytes_le_fixed::<R>(&r);
        assert_eq!(got, expected);
    }
}

