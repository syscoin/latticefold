use ark_ff::Field;
use latticefold::transcript::poseidon::F257;
use symphony::dpp_sumcheck::Dr1csBuilder;

use super::frog::{
    frog_add_mod_p_from_byte_vars_assume_canonical, frog_mul_mod_p_from_byte_vars_assume_canonical,
    frog_sub_mod_p_from_byte_vars_assume_canonical,
    frog_add_mod_p_digits, frog_mul_mod_p_digits, frog_sub_mod_p_digits, frogscalar_from_u64_bytes_le_digits,
    ring_mul_negacyclic_ntt_goldilocks_d64, FrogScalar,
};

/// A ring element whose coefficients are Frog base-field scalars encoded as canonical 8-byte little-endian limbs.
///
/// - Length = ring dimension `d` (one coefficient per ring coefficient).
/// - Each coefficient is `[u8; 8]` represented as 8 F257 vars in `[0,255]`, and is constrained elsewhere to be `< p`.
pub(crate) type RingBytes = Vec<[usize; 8]>;

/// A ring element whose coefficients are Frog base-field scalars encoded as balanced base-16 digits.
///
/// - Length = ring dimension `d` (one coefficient per ring coefficient).
/// - Each coefficient is a canonical Frog scalar `[usize; 17]` (bal16 digits).
pub(crate) type RingDigits = Vec<FrogScalar>;

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

// -----------------------------------------------------------------------------
// Digit-domain helpers (preferred for heavy CM math + ring multiplication)
// -----------------------------------------------------------------------------

#[inline]
pub(crate) fn frog_bytes_to_digits(gb: &mut Dr1csBuilder<F257>, bytes_le: [usize; 8]) -> FrogScalar {
    frogscalar_from_u64_bytes_le_digits(gb, bytes_le)
}

#[inline]
pub(crate) fn ring_bytes_to_digits(gb: &mut Dr1csBuilder<F257>, a: &RingBytes) -> RingDigits {
    a.iter().copied().map(|x| frog_bytes_to_digits(gb, x)).collect()
}

#[inline]
pub(crate) fn ring_zero_digits(gb: &mut Dr1csBuilder<F257>, d: usize) -> RingDigits {
    let z_bytes = alloc_const_frog_u64(gb, 0u64);
    let z = frog_bytes_to_digits(gb, z_bytes);
    vec![z; d]
}

#[inline]
pub(crate) fn ring_eq_digits(gb: &mut Dr1csBuilder<F257>, a: &RingDigits, b: &RingDigits) {
    debug_assert_eq!(a.len(), b.len());
    for (ai, bi) in a.iter().zip(b.iter()) {
        for j in 0..17 {
            gb.enforce_lc_times_one_eq_const(vec![(F257::ONE, ai[j]), (-F257::ONE, bi[j])]);
        }
    }
}

#[inline]
pub(crate) fn ring_add_digits(gb: &mut Dr1csBuilder<F257>, a: &RingDigits, b: &RingDigits) -> RingDigits {
    debug_assert_eq!(a.len(), b.len());
    a.iter().zip(b.iter()).map(|(ai, bi)| frog_add_mod_p_digits(gb, ai, bi)).collect()
}

#[inline]
pub(crate) fn ring_sub_digits(gb: &mut Dr1csBuilder<F257>, a: &RingDigits, b: &RingDigits) -> RingDigits {
    debug_assert_eq!(a.len(), b.len());
    a.iter().zip(b.iter()).map(|(ai, bi)| frog_sub_mod_p_digits(gb, ai, bi)).collect()
}

#[inline]
pub(crate) fn ring_scale_digits(gb: &mut Dr1csBuilder<F257>, a: &RingDigits, s: &FrogScalar) -> RingDigits {
    a.iter().map(|ai| frog_mul_mod_p_digits(gb, ai, s)).collect()
}

/// Negacyclic ring multiplication for ring_dim=64 over Goldilocks (digit-domain).
pub(crate) fn ring_mul_negacyclic_digits_d64(
    gb: &mut Dr1csBuilder<F257>,
    a: &RingDigits,
    b: &RingDigits,
) -> Result<RingDigits, String> {
    if a.len() != 64 || b.len() != 64 {
        return Err("ring_mul_negacyclic_digits_d64: expected ring_dim=64".to_string());
    }
    let aa: [FrogScalar; 64] = core::array::from_fn(|i| a[i]);
    let bb: [FrogScalar; 64] = core::array::from_fn(|i| b[i]);
    let cc = ring_mul_negacyclic_ntt_goldilocks_d64(gb, &aa, &bb);
    Ok(cc.into_iter().collect())
}

/// Degree-2 Lagrange basis in digit encoding.
pub(crate) fn lagrange_degree2_frog_digits(
    gb: &mut Dr1csBuilder<F257>,
    r: &FrogScalar,
    inv2: &FrogScalar,
    one: &FrogScalar,
    two: &FrogScalar,
) -> (FrogScalar, FrogScalar, FrogScalar) {
    // t1 = r - 1, t2 = r - 2
    let t1 = frog_sub_mod_p_digits(gb, r, one);
    let t2 = frog_sub_mod_p_digits(gb, r, two);

    // L0 = (r-1)(r-2)/2
    let p = frog_mul_mod_p_digits(gb, &t1, &t2);
    let l0 = frog_mul_mod_p_digits(gb, &p, inv2);

    // L1 = -r(r-2)
    let p = frog_mul_mod_p_digits(gb, r, &t2);
    let zero_bytes = alloc_const_frog_u64(gb, 0u64);
    let zero = frog_bytes_to_digits(gb, zero_bytes);
    let l1 = frog_sub_mod_p_digits(gb, &zero, &p);

    // L2 = r(r-1)/2
    let p = frog_mul_mod_p_digits(gb, r, &t1);
    let l2 = frog_mul_mod_p_digits(gb, &p, inv2);

    (l0, l1, l2)
}

/// Verify a degree-2 sumcheck over ring elements (digit-encoded) and return the final claim.
pub(crate) fn sumcheck_verify_degree2_ring_digits(
    gb: &mut Dr1csBuilder<F257>,
    mut claimed_sum: RingDigits,
    msgs: &[[RingDigits; 3]],
    rs: &[FrogScalar],
) -> Result<RingDigits, String> {
    if msgs.len() != rs.len() {
        return Err("sumcheck_verify_degree2_ring_digits: msgs/rs length mismatch".to_string());
    }
    let d = claimed_sum.len();

    let one_bytes = alloc_const_frog_u64(gb, 1u64);
    let two_bytes = alloc_const_frog_u64(gb, 2u64);
    let one = frog_bytes_to_digits(gb, one_bytes);
    let two = frog_bytes_to_digits(gb, two_bytes);
    let inv2_u64 = (crate::we_frog_poseidon_f257::FROG_P + 1) / 2;
    let inv2_bytes = alloc_const_frog_u64(gb, inv2_u64);
    let inv2 = frog_bytes_to_digits(gb, inv2_bytes);

    for (m, r) in msgs.iter().zip(rs.iter()) {
        if m[0].len() != d || m[1].len() != d || m[2].len() != d {
            return Err("sumcheck_verify_degree2_ring_digits: ring dimension mismatch".to_string());
        }

        // g(0) + g(1) == claim
        let g01 = ring_add_digits(gb, &m[0], &m[1]);
        ring_eq_digits(gb, &g01, &claimed_sum);

        // claim = g(r)
        let (l0, l1, l2) = lagrange_degree2_frog_digits(gb, r, &inv2, &one, &two);
        let t0 = ring_scale_digits(gb, &m[0], &l0);
        let t1 = ring_scale_digits(gb, &m[1], &l1);
        let t2 = ring_scale_digits(gb, &m[2], &l2);
        let s01 = ring_add_digits(gb, &t0, &t1);
        claimed_sum = ring_add_digits(gb, &s01, &t2);
    }

    Ok(claimed_sum)
}

// -----------------------------------------------------------------------------
// CM tensor evaluation (t(z)) in digit domain
// -----------------------------------------------------------------------------

#[inline]
fn frog_const_u64_digits(gb: &mut Dr1csBuilder<F257>, v: u64) -> FrogScalar {
    let bytes = alloc_const_frog_u64(gb, v);
    frog_bytes_to_digits(gb, bytes)
}

#[inline]
pub(crate) fn frog_one_minus_digits(gb: &mut Dr1csBuilder<F257>, r: &FrogScalar) -> FrogScalar {
    let one = frog_const_u64_digits(gb, 1u64);
    frog_sub_mod_p_digits(gb, &one, r)
}

/// Tensor-expand a list of verifier challenges `c = [c0..c_{t-1}]` into length-2^t coefficients.
///
/// Convention matches LF+/CM: for each bit, expand `acc` into `[acc*(1-c_i), acc*c_i]`.
pub(crate) fn tensor_frog_scalars_digits(gb: &mut Dr1csBuilder<F257>, c: &[FrogScalar]) -> Vec<FrogScalar> {
    let mut acc: Vec<FrogScalar> = vec![frog_const_u64_digits(gb, 1u64)];
    for ci in c {
        let one_minus = frog_one_minus_digits(gb, ci);
        let mut next: Vec<FrogScalar> = Vec::with_capacity(acc.len() * 2);
        for t in &acc {
            next.push(frog_mul_mod_p_digits(gb, t, &one_minus));
            next.push(frog_mul_mod_p_digits(gb, t, ci));
        }
        acc = next;
    }
    acc
}

/// Lift a tensor-expanded scalar table into constant-coeff ring elements.
pub(crate) fn tensor_frog_ringconst_digits(
    gb: &mut Dr1csBuilder<F257>,
    tensor_scalars: &[FrogScalar],
    ring_dim: usize,
) -> Vec<RingDigits> {
    tensor_scalars
        .iter()
        .map(|s| ring_const_coeff_digits(gb, s, ring_dim))
        .collect()
}

/// Compute powers \([1, x, x^2, ..., x^n]\) in Frog scalar digit encoding.
pub(crate) fn frog_pow_table_digits(
    gb: &mut Dr1csBuilder<F257>,
    x: &FrogScalar,
    n: usize,
) -> Vec<FrogScalar> {
    let mut out: Vec<FrogScalar> = Vec::with_capacity(n + 1);
    let mut acc = frog_const_u64_digits(gb, 1u64);
    out.push(acc);
    for _ in 0..n {
        acc = frog_mul_mod_p_digits(gb, &acc, x);
        out.push(acc);
    }
    out
}

/// Evaluate the multilinear equality polynomial `eq(c, r)` over Frog scalars (digit encoding).
///
/// This matches `we_gate_arith::eq_eval_vars` but over the Frog base field:
/// \(\prod_i (c_i r_i + (1-c_i)(1-r_i))\).
pub(crate) fn eq_eval_frog_digits(
    gb: &mut Dr1csBuilder<F257>,
    c: &[FrogScalar],
    r: &[FrogScalar],
) -> Result<FrogScalar, String> {
    if c.len() != r.len() {
        return Err("eq_eval_frog_digits: length mismatch".to_string());
    }
    let mut acc = frog_const_u64_digits(gb, 1u64);
    for (ci, ri) in c.iter().zip(r.iter()) {
        let one_minus_ci = frog_one_minus_digits(gb, ci);
        let one_minus_ri = frog_one_minus_digits(gb, ri);
        let ci_ri = frog_mul_mod_p_digits(gb, ci, ri);
        let om = frog_mul_mod_p_digits(gb, &one_minus_ci, &one_minus_ri);
        let t = frog_add_mod_p_digits(gb, &ci_ri, &om);
        acc = frog_mul_mod_p_digits(gb, &acc, &t);
    }
    Ok(acc)
}

#[inline]
pub(crate) fn ring_const_coeff_digits(gb: &mut Dr1csBuilder<F257>, c0: &FrogScalar, d: usize) -> RingDigits {
    let z = frog_const_u64_digits(gb, 0u64);
    let mut out = vec![z; d];
    if d > 0 {
        out[0] = *c0;
    }
    out
}

#[inline]
pub(crate) fn ring_unit_monomial_digits(gb: &mut Dr1csBuilder<F257>, idx: usize, d: usize) -> RingDigits {
    let z = frog_const_u64_digits(gb, 0u64);
    let one = frog_const_u64_digits(gb, 1u64);
    let mut out = vec![z; d];
    if idx < d {
        out[idx] = one;
    }
    out
}

/// Evaluate a small MLE table (ring-valued) at point `r` (LSB-first variable order).
///
/// `table.len()` must be a power of two, and `r.len() == log2(table.len())`.
pub(crate) fn eval_small_mle_ring_digits(
    gb: &mut Dr1csBuilder<F257>,
    table: &[RingDigits],
    r: &[FrogScalar],
) -> RingDigits {
    debug_assert!(!table.is_empty());
    debug_assert!(table.len().is_power_of_two());
    debug_assert_eq!(table.len(), 1usize << r.len());

    let mut cur: Vec<RingDigits> = table.to_vec();
    for ri in r.iter() {
        let one_minus = frog_one_minus_digits(gb, ri);
        let mut next: Vec<RingDigits> = Vec::with_capacity(cur.len() / 2);
        for j in 0..(cur.len() / 2) {
            // out = cur[2j]*(1-ri) + cur[2j+1]*ri
            let a = ring_scale_digits(gb, &cur[2 * j], &one_minus);
            let b = ring_scale_digits(gb, &cur[2 * j + 1], ri);
            next.push(ring_add_digits(gb, &a, &b));
        }
        cur = next;
    }
    debug_assert_eq!(cur.len(), 1);
    cur.pop().unwrap()
}

/// Evaluate the CM tensor product t(z) at `r`, mirroring `tensor_eval::eval_t_z_optimized`.
///
/// Expected factor order / bit slicing (LSB-first):
/// - `x_powers` (size = d) uses the lowest `log2(d)` bits
/// - `dpp` (size = ell) uses the next `log2(ell)` bits
/// - `s_prime_flat` (size = k*d) uses the next `log2(k*d)` bits
/// - `tensor_c_ring` (size = kappa) uses the next `log2(kappa)` bits
///
/// Any remaining high bits are padded with the factor \(\prod (1 - r_i)\).
pub(crate) fn eval_t_z_optimized_ring_digits(
    gb: &mut Dr1csBuilder<F257>,
    tensor_c_ring: &[RingDigits],
    s_prime_flat: &[RingDigits],
    dpp: &[RingDigits],
    x_powers: &[RingDigits],
    r: &[FrogScalar],
) -> Result<RingDigits, String> {
    let sizes = [x_powers.len(), dpp.len(), s_prime_flat.len(), tensor_c_ring.len()];
    if sizes.iter().any(|&s| s == 0 || !s.is_power_of_two()) {
        return Err("eval_t_z_optimized_ring_digits: expected power-of-two non-empty factor sizes".to_string());
    }
    let vars4 = sizes.map(|s| ark_std::log2(s) as usize);
    let tensor_vars = vars4.iter().sum::<usize>();
    if r.len() < tensor_vars {
        return Err("eval_t_z_optimized_ring_digits: r too short".to_string());
    }

    // Split r into chunks (innermost to outermost) as in tensor_eval::eval_t_z_optimized.
    let r4 = &r[0..vars4[0]]; // x_powers (lowest bits)
    let r3 = &r[vars4[0]..vars4[0] + vars4[1]];
    let r2 = &r[vars4[0] + vars4[1]..vars4[0] + vars4[1] + vars4[2]];
    let r1 = &r[vars4[0] + vars4[1] + vars4[2]..tensor_vars];

    let v1 = eval_small_mle_ring_digits(gb, tensor_c_ring, r1);
    let v2 = eval_small_mle_ring_digits(gb, s_prime_flat, r2);
    let v3 = eval_small_mle_ring_digits(gb, dpp, r3);
    let v4 = eval_small_mle_ring_digits(gb, x_powers, r4);

    let mut res = ring_mul_negacyclic_digits_d64(gb, &v1, &v2)?;
    res = ring_mul_negacyclic_digits_d64(gb, &res, &v3)?;
    res = ring_mul_negacyclic_digits_d64(gb, &res, &v4)?;

    // Padding factor: Π_{j=tensor_vars..} (1 - r[j]) as scalar.
    let mut pad = frog_const_u64_digits(gb, 1u64);
    for rj in &r[tensor_vars..] {
        let om = frog_one_minus_digits(gb, rj);
        pad = frog_mul_mod_p_digits(gb, &pad, &om);
    }
    Ok(ring_scale_digits(gb, &res, &pad))
}

