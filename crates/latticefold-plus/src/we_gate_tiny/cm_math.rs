use ark_ff::Field;
use latticefold::transcript::poseidon::F257;
use symphony::dpp_sumcheck::Dr1csBuilder;

use super::frog::{
    frog_add_mod_p_from_byte_vars_assume_canonical, frog_mul_mod_p_from_byte_vars_assume_canonical,
    frog_sub_mod_p_from_byte_vars_assume_canonical,
};

/// A ring element whose coefficients are Frog base-field scalars encoded as canonical 8-byte little-endian limbs.
///
/// - Length = ring dimension `d` (one coefficient per ring coefficient).
/// - Each coefficient is `[u8; 8]` represented as 8 F257 vars in `[0,255]`, and is constrained elsewhere to be `< p`.
pub(crate) type RingBytes = Vec<[usize; 8]>;

#[inline]
fn alloc_const_byte(gb: &mut Dr1csBuilder<F257>, v: u8) -> usize {
    let x = gb.new_var(F257::from(v as u64));
    gb.enforce_var_eq_const(x, F257::from(v as u64));
    x
}

#[inline]
pub(crate) fn alloc_const_frog_u64(gb: &mut Dr1csBuilder<F257>, v: u64) -> [usize; 8] {
    let bs = v.to_le_bytes();
    let mut out = [0usize; 8];
    for i in 0..8 {
        out[i] = alloc_const_byte(gb, bs[i]);
    }
    out
}

#[inline]
pub(crate) fn ring_zero_bytes(gb: &mut Dr1csBuilder<F257>, d: usize) -> RingBytes {
    let z = alloc_const_frog_u64(gb, 0u64);
    vec![z; d]
}

#[inline]
pub(crate) fn ring_eq_bytes(gb: &mut Dr1csBuilder<F257>, a: &RingBytes, b: &RingBytes) {
    debug_assert_eq!(a.len(), b.len());
    for (ai, bi) in a.iter().zip(b.iter()) {
        for j in 0..8 {
            gb.enforce_lc_times_one_eq_const(vec![(F257::ONE, ai[j]), (-F257::ONE, bi[j])]);
        }
    }
}

#[inline]
pub(crate) fn ring_add_bytes(gb: &mut Dr1csBuilder<F257>, a: &RingBytes, b: &RingBytes) -> RingBytes {
    debug_assert_eq!(a.len(), b.len());
    let mut out = Vec::with_capacity(a.len());
    for i in 0..a.len() {
        out.push(frog_add_mod_p_from_byte_vars_assume_canonical(gb, &a[i], &b[i]));
    }
    out
}

#[inline]
pub(crate) fn ring_sub_bytes(gb: &mut Dr1csBuilder<F257>, a: &RingBytes, b: &RingBytes) -> RingBytes {
    debug_assert_eq!(a.len(), b.len());
    let mut out = Vec::with_capacity(a.len());
    for i in 0..a.len() {
        out.push(frog_sub_mod_p_from_byte_vars_assume_canonical(gb, &a[i], &b[i]));
    }
    out
}

/// Scale a ring element by a base-field scalar (byte-encoded).
///
/// Since the scalar is in the base field, this is per-coefficient multiplication.
#[inline]
pub(crate) fn ring_scale_bytes(gb: &mut Dr1csBuilder<F257>, a: &RingBytes, s: &[usize; 8]) -> RingBytes {
    let mut out = Vec::with_capacity(a.len());
    for i in 0..a.len() {
        out.push(frog_mul_mod_p_from_byte_vars_assume_canonical(gb, &a[i], s));
    }
    out
}

/// Compute Lagrange basis coefficients (L0,L1,L2) for interpolation of degree-2 sumcheck message
/// polynomials at points 0,1,2 evaluated at `r`.
///
/// This mirrors `we_gate_arith::lagrange_degree2` but over Frog base field, with byte encodings.
pub(crate) fn lagrange_degree2_frog(
    gb: &mut Dr1csBuilder<F257>,
    r: &[usize; 8],
    inv2: &[usize; 8],
    one: &[usize; 8],
    two: &[usize; 8],
) -> ([usize; 8], [usize; 8], [usize; 8]) {
    // t1 = r - 1, t2 = r - 2
    let t1 = frog_sub_mod_p_from_byte_vars_assume_canonical(gb, r, one);
    let t2 = frog_sub_mod_p_from_byte_vars_assume_canonical(gb, r, two);

    // L0 = (r-1)(r-2)/2
    let p = frog_mul_mod_p_from_byte_vars_assume_canonical(gb, &t1, &t2);
    let l0 = frog_mul_mod_p_from_byte_vars_assume_canonical(gb, &p, inv2);

    // L1 = -r(r-2) = 0 - r(r-2)
    let p = frog_mul_mod_p_from_byte_vars_assume_canonical(gb, r, &t2);
    let zero = alloc_const_frog_u64(gb, 0u64);
    let l1 = frog_sub_mod_p_from_byte_vars_assume_canonical(gb, &zero, &p);

    // L2 = r(r-1)/2
    let p = frog_mul_mod_p_from_byte_vars_assume_canonical(gb, r, &t1);
    let l2 = frog_mul_mod_p_from_byte_vars_assume_canonical(gb, &p, inv2);

    (l0, l1, l2)
}

/// Verify a degree-2 sumcheck over ring elements (byte-encoded coefficients) and return the final claim.
///
/// - `claimed_sum`: initial claim (ring)
/// - `msgs`: per-round prover messages, each has 3 ring elements (g(0), g(1), g(2))
/// - `rs`: per-round verifier challenges (base-field scalars), byte-encoded
pub(crate) fn sumcheck_verify_degree2_ring_bytes(
    gb: &mut Dr1csBuilder<F257>,
    mut claimed_sum: RingBytes,
    msgs: &[[RingBytes; 3]],
    rs: &[[usize; 8]],
) -> Result<RingBytes, String> {
    if msgs.len() != rs.len() {
        return Err("sumcheck_verify_degree2_ring_bytes: msgs/rs length mismatch".to_string());
    }
    let d = claimed_sum.len();

    // Constants in Frog field.
    // inv2 is computed in host here as a fixed u64 constant; encoded as bytes.
    // (Caller can reuse these across calls later.)
    let one = alloc_const_frog_u64(gb, 1u64);
    let two = alloc_const_frog_u64(gb, 2u64);
    // inv2 mod p (p is odd): (p+1)/2
    let inv2_u64 = (crate::we_frog_poseidon_f257::FROG_P + 1) / 2;
    let inv2 = alloc_const_frog_u64(gb, inv2_u64);

    for (m, r) in msgs.iter().zip(rs.iter()) {
        if m[0].len() != d || m[1].len() != d || m[2].len() != d {
            return Err("sumcheck_verify_degree2_ring_bytes: ring dimension mismatch".to_string());
        }

        // Check g(0) + g(1) == claimed_sum (coefficient-wise).
        let g01 = ring_add_bytes(gb, &m[0], &m[1]);
        ring_eq_bytes(gb, &g01, &claimed_sum);

        // Update claim = g(r) via Lagrange interpolation.
        let (l0, l1, l2) = lagrange_degree2_frog(gb, r, &inv2, &one, &two);
        let t0 = ring_scale_bytes(gb, &m[0], &l0);
        let t1 = ring_scale_bytes(gb, &m[1], &l1);
        let t2 = ring_scale_bytes(gb, &m[2], &l2);
        let s01 = ring_add_bytes(gb, &t0, &t1);
        claimed_sum = ring_add_bytes(gb, &s01, &t2);
    }
    Ok(claimed_sum)
}

