use ark_ff::{BigInteger, Field, PrimeField};

use latticefold::transcript::poseidon::F257;
use symphony::dpp_sumcheck::Dr1csBuilder;

use super::gadgets::{alloc_bool, decompose_existing_byte_var_to_bits, ByteVar};

// -----------------------------------------------------------------------------
// Balanced base-16 (nibble) gadgets for *bounded* integer arithmetic in F257.
// -----------------------------------------------------------------------------

const NIBBLE_BASE: i32 = 16;

#[inline]
pub(crate) fn f257_to_i32_bal(x: F257) -> i32 {
    // Interpret F257 element as a signed integer in (-128..128] via centered lift.
    let bytes = x.into_bigint().to_bytes_le();
    let u16v = (bytes.get(0).copied().unwrap_or(0) as u16)
        | ((bytes.get(1).copied().unwrap_or(0) as u16) << 8);
    let u = u16v as i32;
    if u <= 128 { u } else { u - 257 }
}

#[inline]
fn i32_to_f257(x: i32) -> F257 {
    let mut v = x % 257;
    if v < 0 {
        v += 257;
    }
    F257::from(v as u64)
}

fn nibble_from_bits(b: &mut Dr1csBuilder<F257>, bits: [usize; 4], v: u8) -> usize {
    debug_assert!(v < 16);
    let out = b.new_var(F257::from(v as u64));
    // out = Σ 2^i * bits[i]
    b.enforce_lc_times_one_eq_const(vec![
        (F257::ONE, out),
        (-F257::ONE, bits[0]),
        (-F257::from(2u64), bits[1]),
        (-F257::from(4u64), bits[2]),
        (-F257::from(8u64), bits[3]),
    ]);
    out
}

/// Build the low/high nibble vars of a byte var (0..15 each), reusing the existing bit vars.
fn byte_to_nibbles(b: &mut Dr1csBuilder<F257>, byte: &ByteVar) -> (usize, usize) {
    let v8 = b.assignment[byte.byte]
        .into_bigint()
        .to_bytes_le()
        .get(0)
        .copied()
        .unwrap_or(0) as u8;
    let lo = nibble_from_bits(
        b,
        [byte.bits[0], byte.bits[1], byte.bits[2], byte.bits[3]],
        v8 & 0x0f,
    );
    let hi = nibble_from_bits(
        b,
        [byte.bits[4], byte.bits[5], byte.bits[6], byte.bits[7]],
        (v8 >> 4) & 0x0f,
    );
    (lo, hi)
}

/// Convert a 4-bit nibble (0..15) into a balanced digit in [-8,7] using the top bit as the sign.
///
/// digit = nibble - 16 * msb, where msb is the 8's bit (boolean).
fn balanced_digit_from_nibble(b: &mut Dr1csBuilder<F257>, nibble: usize, msb: usize) -> usize {
    // Witness digit value using assignment of nibble/msb.
    let n = b.assignment[nibble]
        .into_bigint()
        .to_bytes_le()
        .get(0)
        .copied()
        .unwrap_or(0) as i32;
    let s = b.assignment[msb]
        .into_bigint()
        .to_bytes_le()
        .get(0)
        .copied()
        .unwrap_or(0) as i32;
    debug_assert!(n < 16);
    debug_assert!(s == 0 || s == 1);
    let d = n - 16 * s; // in [-8,7]
    let out = b.new_var(F257::from(((d % 257 + 257) % 257) as u64));
    b.enforce_lc_times_one_eq_const(vec![
        (F257::ONE, out),
        (-F257::ONE, nibble),
        (F257::from(16u64), msb),
    ]);
    out
}

/// Allocate a balanced base-16 digit variable in [-8,7] from witness `d`.
pub(crate) fn alloc_bal16_digit(b: &mut Dr1csBuilder<F257>, d: i8) -> usize {
    assert!((-8..=7).contains(&d));
    let nib = if d < 0 { (d as i16 + 16) as u8 } else { d as u8 };
    debug_assert!(nib < 16);
    let mut bits4 = [0usize; 4];
    for i in 0..4 {
        bits4[i] = alloc_bool::<F257>(b, ((nib >> i) & 1) == 1);
    }
    let nib_var = nibble_from_bits(b, bits4, nib);
    let msb = bits4[3];
    balanced_digit_from_nibble(b, nib_var, msb)
}

/// Allocate a signed carry `c` by allocating an offset `off = c + 16` in [0,31] and
/// enforcing `off ∈ [5,27]` (i.e., `c ∈ [-11,11]`).
fn alloc_carry_pm11(b: &mut Dr1csBuilder<F257>, c: i32) -> usize {
    assert!((-11..=11).contains(&c));
    let off = (c + 16) as u8; // in [5,27]
    // 5-bit decomposition of off.
    let mut bits = [0usize; 5];
    for i in 0..5 {
        bits[i] = alloc_bool::<F257>(b, ((off >> i) & 1) == 1);
    }
    let off_var = b.new_var(F257::from(off as u64));
    // off = Σ 2^i * bits[i]
    b.enforce_lc_times_one_eq_const(vec![
        (F257::ONE, off_var),
        (-F257::ONE, bits[0]),
        (-F257::from(2u64), bits[1]),
        (-F257::from(4u64), bits[2]),
        (-F257::from(8u64), bits[3]),
        (-F257::from(16u64), bits[4]),
    ]);

    // Enforce off <= 27  <=> NOT(off in {28..31}) <=> NOT(b4 & b3 & b2).
    let t = b.new_var(b.assignment[bits[4]] * b.assignment[bits[3]]);
    b.enforce_mul(bits[4], bits[3], t);
    let u = b.new_var(b.assignment[t] * b.assignment[bits[2]]);
    b.enforce_mul(t, bits[2], u);
    b.enforce_var_eq_const(u, F257::ZERO);

    // Enforce off >= 5:
    // ge5 = b4 OR b3 OR (b2 AND (b1 OR b0)) == 1
    // or01 = b0 OR b1
    let or01 = {
        let v = b.new_var(
            b.assignment[bits[0]] + b.assignment[bits[1]] - b.assignment[bits[0]] * b.assignment[bits[1]],
        );
        // v = b0 + b1 - b0*b1
        let p = b.new_var(b.assignment[bits[0]] * b.assignment[bits[1]]);
        b.enforce_mul(bits[0], bits[1], p);
        b.enforce_lc_times_one_eq_const(vec![
            (F257::ONE, v),
            (-F257::ONE, bits[0]),
            (-F257::ONE, bits[1]),
            (F257::ONE, p),
        ]);
        v
    };
    let and2 = {
        let v = b.new_var(b.assignment[bits[2]] * b.assignment[or01]);
        b.enforce_mul(bits[2], or01, v);
        v
    };
    let or34 = {
        let v = b.new_var(
            b.assignment[bits[3]] + b.assignment[bits[4]] - b.assignment[bits[3]] * b.assignment[bits[4]],
        );
        let p = b.new_var(b.assignment[bits[3]] * b.assignment[bits[4]]);
        b.enforce_mul(bits[3], bits[4], p);
        b.enforce_lc_times_one_eq_const(vec![
            (F257::ONE, v),
            (-F257::ONE, bits[3]),
            (-F257::ONE, bits[4]),
            (F257::ONE, p),
        ]);
        v
    };
    let ge5 = {
        let v = b.new_var(
            b.assignment[or34] + b.assignment[and2] - b.assignment[or34] * b.assignment[and2],
        );
        let p = b.new_var(b.assignment[or34] * b.assignment[and2]);
        b.enforce_mul(or34, and2, p);
        b.enforce_lc_times_one_eq_const(vec![
            (F257::ONE, v),
            (-F257::ONE, or34),
            (-F257::ONE, and2),
            (F257::ONE, p),
        ]);
        v
    };
    b.enforce_var_eq_const(ge5, F257::ONE);

    // Carry var c = off - 16 (in F257 representation).
    let c_var = b.new_var(i32_to_f257(c));
    b.enforce_lc_times_one_eq_const(vec![
        (F257::ONE, c_var),
        (-F257::ONE, off_var),
        (F257::from(16u64), b.one()),
    ]);
    c_var
}

/// Allocate a signed carry `c ∈ [-2,2]` as an F257 variable, with a tight boolean decomposition.
fn alloc_carry_pm2(b: &mut Dr1csBuilder<F257>, c: i32) -> usize {
    assert!((-2..=2).contains(&c));
    // Represent as off = c + 2 in [0,4].
    let off = (c + 2) as u8;
    let b0 = alloc_bool::<F257>(b, (off & 1) == 1);
    let b1 = alloc_bool::<F257>(b, (off & 2) == 2);
    let b2 = alloc_bool::<F257>(b, (off & 4) == 4);
    let off_var = b.new_var(F257::from(off as u64));
    // off = b0 + 2*b1 + 4*b2
    b.enforce_lc_times_one_eq_const(vec![
        (F257::ONE, off_var),
        (-F257::ONE, b0),
        (-F257::from(2u64), b1),
        (-F257::from(4u64), b2),
    ]);
    // c = off - 2
    let c_var = b.new_var(i32_to_f257(c));
    b.enforce_lc_times_one_eq_const(vec![
        (F257::ONE, c_var),
        (-F257::ONE, off_var),
        (F257::from(2u64), b.one()),
    ]);
    c_var
}

/// Add two balanced base-16 digit vectors of the same length.
///
/// Assumes each digit is in [-8,7]. Enforces output digits in [-8,7] and carry in [-2,2].
pub(crate) fn add_bal16_same_len(
    b: &mut Dr1csBuilder<F257>,
    a: &[usize],
    c: &[usize],
) -> (Vec<usize>, usize /* carry_out */) {
    assert_eq!(a.len(), c.len());
    let n = a.len();
    let mut out: Vec<usize> = Vec::with_capacity(n);
    let mut carry_i32: i32 = 0;
    let mut carry = b.new_var(F257::ZERO);
    b.enforce_var_eq_const(carry, F257::ZERO);

    for i in 0..n {
        let ai = f257_to_i32_bal(b.assignment[a[i]]);
        let ci = f257_to_i32_bal(b.assignment[c[i]]);
        let sum = ai + ci + carry_i32;
        // Choose carry_out so remainder in [-8,7].
        let mut carry_next = if sum >= 0 { (sum + 8) / 16 } else { -(((-sum) + 8) / 16) };
        let mut rem = sum - 16 * carry_next;
        while rem > 7 {
            carry_next += 1;
            rem -= 16;
        }
        while rem < -8 {
            carry_next -= 1;
            rem += 16;
        }
        // For inputs in [-8,7], carry stays in [-1,1], but keep [-2,2] margin.
        assert!((-2..=2).contains(&carry_next));
        assert!((-8..=7).contains(&rem));

        let out_digit = alloc_bal16_digit(b, rem as i8);
        let carry_next_var = alloc_carry_pm2(b, carry_next);

        // a_i + c_i + carry - out_i - 16*carry_next = 0
        b.enforce_lc_times_one_eq_const(vec![
            (F257::ONE, a[i]),
            (F257::ONE, c[i]),
            (F257::ONE, carry),
            (-F257::ONE, out_digit),
            (-F257::from(16u64), carry_next_var),
        ]);

        out.push(out_digit);
        carry_i32 = carry_next;
        carry = carry_next_var;
    }

    (out, carry)
}

/// Negate a balanced base-16 digit vector (little-endian), producing digits in [-8,7].
pub(crate) fn neg_bal16_digits(
    b: &mut Dr1csBuilder<F257>,
    x: &[usize],
) -> (Vec<usize>, usize /* carry_out */) {
    let n = x.len();
    let mut out: Vec<usize> = Vec::with_capacity(n);
    let mut carry_i32: i32 = 0;
    let mut carry = b.new_var(F257::ZERO);
    b.enforce_var_eq_const(carry, F257::ZERO);

    for i in 0..n {
        let xi = f257_to_i32_bal(b.assignment[x[i]]);
        let sum = (-xi) + carry_i32;
        let mut carry_next = if sum >= 0 { (sum + 8) / 16 } else { -(((-sum) + 8) / 16) };
        let mut rem = sum - 16 * carry_next;
        while rem > 7 {
            carry_next += 1;
            rem -= 16;
        }
        while rem < -8 {
            carry_next -= 1;
            rem += 16;
        }
        debug_assert!((-2..=2).contains(&carry_next));
        debug_assert!((-8..=7).contains(&rem));

        let out_digit = alloc_bal16_digit(b, rem as i8);
        let carry_next_var = alloc_carry_pm2(b, carry_next);

        // carry - x_i - out_i - 16*carry_next = 0
        b.enforce_lc_times_one_eq_const(vec![
            (F257::ONE, carry),
            (-F257::ONE, x[i]),
            (-F257::ONE, out_digit),
            (-F257::from(16u64), carry_next_var),
        ]);

        out.push(out_digit);
        carry_i32 = carry_next;
        carry = carry_next_var;
    }

    (out, carry)
}

/// Subtract two balanced base-16 digit vectors of the same length: `a - c`.
pub(crate) fn sub_bal16_same_len(
    b: &mut Dr1csBuilder<F257>,
    a: &[usize],
    c: &[usize],
) -> (Vec<usize>, usize /* carry_out */) {
    assert_eq!(a.len(), c.len());
    let (neg_c, carry_neg) = neg_bal16_digits(b, c);
    let _ = carry_neg;
    add_bal16_same_len(b, a, &neg_c)
}

#[inline]
pub(crate) fn mul_bal16_3_by_digits9(
    b: &mut Dr1csBuilder<F257>,
    coeff3: &[usize; 3],
    digits9: &[usize],
) -> [usize; 12] {
    assert_eq!(digits9.len(), 9, "digits9 must have len 9");
    let out = mul_bal16_small(b, coeff3, digits9);
    debug_assert_eq!(out.len(), 12);
    let mut r = [0usize; 12];
    for i in 0..12 {
        r[i] = out[i];
    }
    r
}

#[inline]
pub(crate) fn mul_bal16_3_by_digits18(
    b: &mut Dr1csBuilder<F257>,
    coeff3: &[usize; 3],
    digits18: &[usize],
) -> [usize; 21] {
    assert_eq!(digits18.len(), 18, "digits18 must have len 18");
    let out = mul_bal16_small(b, coeff3, digits18);
    debug_assert_eq!(out.len(), 21);
    let mut r = [0usize; 21];
    for i in 0..21 {
        r[i] = out[i];
    }
    r
}

pub(crate) fn scale_short_coeffs_by_digits9(
    b: &mut Dr1csBuilder<F257>,
    coeffs3: &[[usize; 3]],
    digits9: &[usize],
) -> Vec<[usize; 12]> {
    assert_eq!(digits9.len(), 9);
    coeffs3
        .iter()
        .map(|c3| mul_bal16_3_by_digits9(b, c3, digits9))
        .collect()
}

pub(crate) fn scale_short_coeffs_by_digits18(
    b: &mut Dr1csBuilder<F257>,
    coeffs3: &[[usize; 3]],
    digits18: &[usize],
) -> Vec<[usize; 21]> {
    assert_eq!(digits18.len(), 18);
    coeffs3
        .iter()
        .map(|c3| mul_bal16_3_by_digits18(b, c3, digits18))
        .collect()
}

/// Rebalance the final digit of a `mul_bal16_small` product.
pub(crate) fn rebalance_tail_pm11_to_pm2(b: &mut Dr1csBuilder<F257>, digits: &[usize]) -> Vec<usize> {
    assert!(!digits.is_empty());
    let l = digits.len();
    let tail = f257_to_i32_bal(b.assignment[digits[l - 1]]);
    debug_assert!((-11..=11).contains(&tail));

    let mut carry2 = if tail >= 0 { (tail + 8) / 16 } else { -(((-tail) + 8) / 16) };
    let mut rem = tail - 16 * carry2;
    while rem > 7 {
        carry2 += 1;
        rem -= 16;
    }
    while rem < -8 {
        carry2 -= 1;
        rem += 16;
    }
    debug_assert!((-8..=7).contains(&rem));
    debug_assert!((-2..=2).contains(&carry2));

    let rem_digit = alloc_bal16_digit(b, rem as i8);
    let carry2_var = alloc_carry_pm2(b, carry2);
    b.enforce_lc_times_one_eq_const(vec![
        (F257::ONE, digits[l - 1]),
        (-F257::ONE, rem_digit),
        (-F257::from(16u64), carry2_var),
    ]);

    let mut out = Vec::with_capacity(l + 1);
    out.extend_from_slice(&digits[..l - 1]);
    out.push(rem_digit);
    out.push(carry2_var);
    out
}

fn shift_pad_bal16(digits: &[usize], shift: usize, target_len: usize, zero_digit: usize) -> Vec<usize> {
    assert!(shift <= target_len);
    assert!(digits.len() + shift <= target_len);
    let mut out = Vec::with_capacity(target_len);
    out.extend(std::iter::repeat(zero_digit).take(shift));
    out.extend_from_slice(digits);
    out.extend(std::iter::repeat(zero_digit).take(target_len - shift - digits.len()));
    out
}

pub(crate) fn sum_product_digits_bal16(
    b: &mut Dr1csBuilder<F257>,
    products13: &[[usize; 13]],
    target_len: usize,
) -> Vec<usize> {
    assert!(target_len >= 13);
    let zero = alloc_bal16_digit(b, 0);

    let mut acc = vec![zero; target_len];
    for p13 in products13 {
        let padded = shift_pad_bal16(p13, 0, target_len, zero);
        let (new_acc, carry) = add_bal16_same_len(b, &acc, &padded);
        acc = new_acc;
        b.enforce_var_eq_const(carry, F257::ZERO);
    }
    acc
}

pub(crate) fn sum_product_digits_bal16_22(
    b: &mut Dr1csBuilder<F257>,
    products22: &[[usize; 22]],
    target_len: usize,
) -> Vec<usize> {
    assert!(target_len >= 22);
    let zero = alloc_bal16_digit(b, 0);
    let mut acc = vec![zero; target_len];
    for p22 in products22 {
        let padded = shift_pad_bal16(p22, 0, target_len, zero);
        let (new_acc, carry) = add_bal16_same_len(b, &acc, &padded);
        acc = new_acc;
        b.enforce_var_eq_const(carry, F257::ZERO);
    }
    acc
}

pub(crate) fn sum_bal16_vectors_fixed_len(
    b: &mut Dr1csBuilder<F257>,
    vecs: &[&[usize]],
    len: usize,
) -> Vec<usize> {
    let zero = alloc_bal16_digit(b, 0);
    let mut acc = vec![zero; len];
    for v in vecs {
        assert_eq!(v.len(), len);
        let (new_acc, carry) = add_bal16_same_len(b, &acc, v);
        acc = new_acc;
        b.enforce_var_eq_const(carry, F257::ZERO);
    }
    acc
}

pub(crate) fn sum_products13_coeffwise_fixed_len(
    b: &mut Dr1csBuilder<F257>,
    per_surface_products13: &[&[[usize; 13]]],
    ring_dim: usize,
    out_len: usize,
) -> Vec<Vec<usize>> {
    let zero = alloc_bal16_digit(b, 0);
    let mut out: Vec<Vec<usize>> = Vec::with_capacity(ring_dim);
    for coeff_idx in 0..ring_dim {
        let mut acc = vec![zero; out_len];
        for surf in per_surface_products13 {
            let p13 = &surf[coeff_idx];
            let padded = shift_pad_bal16(p13, 0, out_len, zero);
            let (new_acc, carry) = add_bal16_same_len(b, &acc, &padded);
            acc = new_acc;
            b.enforce_var_eq_const(carry, F257::ZERO);
        }
        out.push(acc);
    }
    out
}

pub(crate) fn sum_products22_coeffwise_fixed_len(
    b: &mut Dr1csBuilder<F257>,
    per_surface_products22: &[&[[usize; 22]]],
    ring_dim: usize,
    out_len: usize,
) -> Vec<Vec<usize>> {
    let zero = alloc_bal16_digit(b, 0);
    let mut out: Vec<Vec<usize>> = Vec::with_capacity(ring_dim);
    for coeff_idx in 0..ring_dim {
        let mut acc = vec![zero; out_len];
        for surf in per_surface_products22 {
            let p22 = &surf[coeff_idx];
            let padded = shift_pad_bal16(p22, 0, out_len, zero);
            let (new_acc, carry) = add_bal16_same_len(b, &acc, &padded);
            acc = new_acc;
            b.enforce_var_eq_const(carry, F257::ZERO);
        }
        out.push(acc);
    }
    out
}

#[inline]
pub(crate) fn rebalance_prod12_to_prod13(
    b: &mut Dr1csBuilder<F257>,
    p12: &[usize; 12],
) -> [usize; 13] {
    let v12 = p12.to_vec();
    let v13 = rebalance_tail_pm11_to_pm2(b, &v12);
    debug_assert_eq!(v13.len(), 13);
    let mut out = [0usize; 13];
    for i in 0..13 {
        out[i] = v13[i];
    }
    out
}

#[inline]
pub(crate) fn rebalance_prod21_to_prod22(
    b: &mut Dr1csBuilder<F257>,
    p21: &[usize; 21],
) -> [usize; 22] {
    let v21 = p21.to_vec();
    let v22 = rebalance_tail_pm11_to_pm2(b, &v21);
    debug_assert_eq!(v22.len(), 22);
    let mut out = [0usize; 22];
    for i in 0..22 {
        out[i] = v22[i];
    }
    out
}

fn mul_bal16_9_by_9_u32ish(b: &mut Dr1csBuilder<F257>, a9: &[usize], b9: &[usize]) -> Vec<usize> {
    assert_eq!(a9.len(), 9);
    assert_eq!(b9.len(), 9);

    let zero_digit = alloc_bal16_digit(b, 0);

    let a0: [usize; 3] = [a9[0], a9[1], a9[2]];
    let a1: [usize; 3] = [a9[3], a9[4], a9[5]];
    let a2: [usize; 3] = [a9[6], a9[7], a9[8]];

    let p0_raw = mul_bal16_small(b, &a0, b9);
    let p0 = rebalance_tail_pm11_to_pm2(b, &p0_raw);
    let p1_raw = mul_bal16_small(b, &a1, b9);
    let p1 = rebalance_tail_pm11_to_pm2(b, &p1_raw);
    let p2_raw = mul_bal16_small(b, &a2, b9);
    let p2 = rebalance_tail_pm11_to_pm2(b, &p2_raw);
    debug_assert_eq!(p0.len(), 13);
    debug_assert_eq!(p1.len(), 13);
    debug_assert_eq!(p2.len(), 13);

    let target_len = p2.len() + 6;
    let s0 = shift_pad_bal16(&p0, 0, target_len, zero_digit);
    let s1 = shift_pad_bal16(&p1, 3, target_len, zero_digit);
    let s2 = shift_pad_bal16(&p2, 6, target_len, zero_digit);

    let (mut t01, carry01) = add_bal16_same_len(b, &s0, &s1);
    t01.push(carry01);

    let mut s2_pad = s2;
    s2_pad.push(zero_digit);
    debug_assert_eq!(t01.len(), s2_pad.len());
    let (mut out, carry) = add_bal16_same_len(b, &t01, &s2_pad);
    out.push(carry);
    out
}

pub(crate) fn mul_u32ish9_to_fixed_bal16(
    b: &mut Dr1csBuilder<F257>,
    a9: &[usize],
    b9: &[usize],
    out_len: usize,
) -> Vec<usize> {
    assert_eq!(a9.len(), 9);
    assert_eq!(b9.len(), 9);
    assert!(out_len >= 16, "u32*u32 fits in 16 nibbles; use >=16 for headroom");

    let zero = alloc_bal16_digit(b, 0);
    let raw = mul_bal16_9_by_9_u32ish(b, a9, b9);
    if raw.len() <= out_len {
        let mut out = raw;
        out.extend(std::iter::repeat(zero).take(out_len - out.len()));
        return out;
    }

    for &dv in &raw[out_len..] {
        b.enforce_var_eq_const(dv, F257::ZERO);
    }
    raw[..out_len].to_vec()
}

pub(crate) fn mul_bal16_long_by_u32ish9(b: &mut Dr1csBuilder<F257>, a: &[usize], b9: &[usize]) -> Vec<usize> {
    assert_eq!(b9.len(), 9);
    if a.is_empty() {
        return vec![alloc_bal16_digit(b, 0)];
    }
    let zero = alloc_bal16_digit(b, 0);
    let blocks = (a.len() + 2) / 3;
    let target_len = 3 * blocks + 13 + 1;
    let mut acc = vec![zero; target_len];

    for blk in 0..blocks {
        let start = blk * 3;
        let end = core::cmp::min(start + 3, a.len());
        let mut coeff3 = [zero; 3];
        for j in 0..(end - start) {
            coeff3[j] = a[start + j];
        }
        let raw = mul_bal16_small(b, &coeff3, b9);
        let reb = rebalance_tail_pm11_to_pm2(b, &raw);
        let shifted = shift_pad_bal16(&reb, blk * 3, target_len, zero);
        let (new_acc, carry) = add_bal16_same_len(b, &acc, &shifted);
        acc = new_acc;
        let top = acc[target_len - 1];
        let (top_sum, top_carry) = add_bal16_same_len(b, &[top], &[carry]);
        acc[target_len - 1] = top_sum[0];
        b.enforce_var_eq_const(top_carry, F257::ZERO);
    }
    acc
}

pub(crate) fn mul_bal16_long_by_long(b: &mut Dr1csBuilder<F257>, a: &[usize], bb: &[usize]) -> Vec<usize> {
    if a.is_empty() || bb.is_empty() {
        return vec![alloc_bal16_digit(b, 0)];
    }
    if a.len().min(bb.len()) <= 3 {
        let raw = mul_bal16_small(b, a, bb);
        return rebalance_tail_pm11_to_pm2(b, &raw);
    }
    let (short, long) = if a.len() <= bb.len() { (a, bb) } else { (bb, a) };

    let zero = alloc_bal16_digit(b, 0);
    let blocks = (short.len() + 2) / 3;

    let per_block_len = long.len() + 5;
    let target_len = per_block_len + 3 * (blocks - 1) + 2;
    let mut acc = vec![zero; target_len];

    for blk in 0..blocks {
        let start = blk * 3;
        let end = core::cmp::min(start + 3, short.len());
        let mut coeff3 = [zero; 3];
        for j in 0..(end - start) {
            coeff3[j] = short[start + j];
        }
        let raw = mul_bal16_small(b, &coeff3, long);
        let reb = rebalance_tail_pm11_to_pm2(b, &raw);
        let shifted = shift_pad_bal16(&reb, blk * 3, target_len, zero);
        let (new_acc, carry) = add_bal16_same_len(b, &acc, &shifted);
        acc = new_acc;
        let top = acc[target_len - 1];
        let (top_sum, top_carry) = add_bal16_same_len(b, &[top], &[carry]);
        acc[target_len - 1] = top_sum[0];
        b.enforce_var_eq_const(top_carry, F257::ZERO);
    }
    acc
}

/// Multiply two balanced base-16 digit vectors (little-endian), specialized for min(len)<=3.
pub(crate) fn mul_bal16_small(b: &mut Dr1csBuilder<F257>, a: &[usize], bb: &[usize]) -> Vec<usize> {
    let la = a.len();
    let lb = bb.len();
    assert!(la > 0 && lb > 0);
    assert!(la.min(lb) <= 3, "mul_bal16_small requires min(len) <= 3");

    let mut out: Vec<usize> = Vec::with_capacity(la + lb);
    let mut carry_i32: i32 = 0;
    let mut carry_var = b.new_var(F257::ZERO);
    b.enforce_var_eq_const(carry_var, F257::ZERO);

    for k in 0..(la + lb - 1) {
        let mut sum: i32 = carry_i32;
        let mut prods: Vec<usize> = Vec::new();
        for i in 0..la {
            let j = k as i32 - i as i32;
            if j < 0 || j >= lb as i32 {
                continue;
            }
            let j = j as usize;
            let aval = f257_to_i32_bal(b.assignment[a[i]]);
            let bval = f257_to_i32_bal(b.assignment[bb[j]]);
            sum += aval * bval;
            let pv = b.new_var(b.assignment[a[i]] * b.assignment[bb[j]]);
            b.enforce_mul(a[i], bb[j], pv);
            prods.push(pv);
        }

        let div_floor = |x: i32, d: i32| -> i32 {
            debug_assert!(d > 0);
            if x >= 0 {
                x / d
            } else {
                -(((-x) + d - 1) / d)
            }
        };
        let mut carry = div_floor(sum + 8, NIBBLE_BASE);
        let mut rem = sum - NIBBLE_BASE * carry;
        while rem > 7 {
            carry += 1;
            rem -= NIBBLE_BASE;
        }
        while rem < -8 {
            carry -= 1;
            rem += NIBBLE_BASE;
        }
        assert!((-8..=7).contains(&rem));
        assert!(
            (-11..=11).contains(&carry),
            "carry out of expected range: {carry} from sum {sum}"
        );

        let digit_var = alloc_bal16_digit(b, rem as i8);
        let carry_out_var = alloc_carry_pm11(b, carry);

        let mut lc: Vec<(F257, usize)> = Vec::new();
        lc.push((F257::ONE, carry_var));
        for &p in &prods {
            lc.push((F257::ONE, p));
        }
        lc.push((-F257::ONE, digit_var));
        lc.push((-F257::from(16u64), carry_out_var));
        b.enforce_lc_times_one_eq_const(lc);

        out.push(digit_var);
        carry_i32 = carry;
        carry_var = carry_out_var;
    }

    out.push(carry_var);
    out
}

/// Convert 4 little-endian byte vars (0..255) into balanced base-16 digits (len 9).
pub(crate) fn u32_bytes_to_bal16_digits(b: &mut Dr1csBuilder<F257>, bytes_le: [usize; 4]) -> Vec<usize> {
    struct Nib {
        d: usize,
        bits: [usize; 4],
        msb: usize,
    }
    let mut nibbles: Vec<Nib> = Vec::with_capacity(8);
    for &bv in &bytes_le {
        let bits8 = decompose_existing_byte_var_to_bits::<F257>(b, bv);
        let tmp = ByteVar { byte: bv, bits: bits8 };
        let (lo, hi) = byte_to_nibbles(b, &tmp);
        nibbles.push(Nib { d: lo, bits: [bits8[0], bits8[1], bits8[2], bits8[3]], msb: bits8[3] });
        nibbles.push(Nib { d: hi, bits: [bits8[4], bits8[5], bits8[6], bits8[7]], msb: bits8[7] });
    }
    debug_assert_eq!(nibbles.len(), 8);

    let mut out: Vec<usize> = Vec::with_capacity(9);
    let mut carry = b.new_var(F257::ZERO);
    b.enforce_var_eq_const(carry, F257::ZERO);
    b.add_constraint(
        vec![(F257::ONE, carry)],
        vec![(F257::ONE, b.one()), (-F257::ONE, carry)],
        vec![(F257::ZERO, b.one())],
    );

    for nib in &nibbles {
        let b0 = nib.bits[0];
        let b1 = nib.bits[1];
        let b2 = nib.bits[2];
        let msb = nib.msb;
        let not_msb = b.new_var(F257::ONE - b.assignment[msb]);
        b.enforce_lc_times_one_eq_const(vec![
            (F257::ONE, not_msb),
            (F257::ONE, msb),
            (-F257::ONE, b.one()),
        ]);

        let t01 = b.new_var(b.assignment[b0] * b.assignment[b1]);
        b.enforce_mul(b0, b1, t01);
        let t012 = b.new_var(b.assignment[t01] * b.assignment[b2]);
        b.enforce_mul(t01, b2, t012);
        let is7 = b.new_var(b.assignment[t012] * b.assignment[not_msb]);
        b.enforce_mul(t012, not_msb, is7);
        b.add_constraint(
            vec![(F257::ONE, is7)],
            vec![(F257::ONE, b.one()), (-F257::ONE, is7)],
            vec![(F257::ZERO, b.one())],
        );

        let carry_is7 = b.new_var(b.assignment[carry] * b.assignment[is7]);
        b.enforce_mul(carry, is7, carry_is7);

        let msb_and = b.new_var(b.assignment[msb] * b.assignment[carry_is7]);
        b.enforce_mul(msb, carry_is7, msb_and);
        let c_out = b.new_var(b.assignment[msb] + b.assignment[carry_is7] - b.assignment[msb_and]);
        b.enforce_lc_times_one_eq_const(vec![
            (F257::ONE, c_out),
            (-F257::ONE, msb),
            (-F257::ONE, carry_is7),
            (F257::ONE, msb_and),
        ]);
        b.add_constraint(
            vec![(F257::ONE, c_out)],
            vec![(F257::ONE, b.one()), (-F257::ONE, c_out)],
            vec![(F257::ZERO, b.one())],
        );

        let d_i = b.assignment[nib.d]
            .into_bigint()
            .to_bytes_le()
            .get(0)
            .copied()
            .unwrap_or(0) as i32;
        let carry_i = b.assignment[carry]
            .into_bigint()
            .to_bytes_le()
            .get(0)
            .copied()
            .unwrap_or(0) as i32;
        let c_i = b.assignment[c_out]
            .into_bigint()
            .to_bytes_le()
            .get(0)
            .copied()
            .unwrap_or(0) as i32;
        let v = d_i + carry_i - 16 * c_i;
        debug_assert!((-8..=7).contains(&v));
        let out_digit = b.new_var(i32_to_f257(v));
        b.enforce_lc_times_one_eq_const(vec![
            (F257::ONE, out_digit),
            (-F257::ONE, nib.d),
            (-F257::ONE, carry),
            (F257::from(16u64), c_out),
        ]);

        out.push(out_digit);
        carry = c_out;
    }

    out.push(carry);
    out
}

/// Convert 8 little-endian byte vars (0..255) into balanced base-16 digits (len 17).
///
/// Output digits are little-endian base-16 with each digit in [-8,7], followed by a final
/// carry digit (also in {0,1} for a canonical u64 byte encoding).
pub(crate) fn u64_bytes_to_bal16_digits(b: &mut Dr1csBuilder<F257>, bytes_le: [usize; 8]) -> Vec<usize> {
    struct Nib {
        d: usize,
        bits: [usize; 4],
        msb: usize,
    }
    let mut nibbles: Vec<Nib> = Vec::with_capacity(16);
    for &bv in &bytes_le {
        let bits8 = decompose_existing_byte_var_to_bits::<F257>(b, bv);
        let tmp = ByteVar { byte: bv, bits: bits8 };
        let (lo, hi) = byte_to_nibbles(b, &tmp);
        nibbles.push(Nib { d: lo, bits: [bits8[0], bits8[1], bits8[2], bits8[3]], msb: bits8[3] });
        nibbles.push(Nib { d: hi, bits: [bits8[4], bits8[5], bits8[6], bits8[7]], msb: bits8[7] });
    }
    debug_assert_eq!(nibbles.len(), 16);

    let mut out: Vec<usize> = Vec::with_capacity(17);
    let mut carry = b.new_var(F257::ZERO);
    b.enforce_var_eq_const(carry, F257::ZERO);
    b.add_constraint(
        vec![(F257::ONE, carry)],
        vec![(F257::ONE, b.one()), (-F257::ONE, carry)],
        vec![(F257::ZERO, b.one())],
    );

    for nib in &nibbles {
        let b0 = nib.bits[0];
        let b1 = nib.bits[1];
        let b2 = nib.bits[2];
        let msb = nib.msb;
        let not_msb = b.new_var(F257::ONE - b.assignment[msb]);
        b.enforce_lc_times_one_eq_const(vec![
            (F257::ONE, not_msb),
            (F257::ONE, msb),
            (-F257::ONE, b.one()),
        ]);

        let t01 = b.new_var(b.assignment[b0] * b.assignment[b1]);
        b.enforce_mul(b0, b1, t01);
        let t012 = b.new_var(b.assignment[t01] * b.assignment[b2]);
        b.enforce_mul(t01, b2, t012);
        let is7 = b.new_var(b.assignment[t012] * b.assignment[not_msb]);
        b.enforce_mul(t012, not_msb, is7);
        b.add_constraint(
            vec![(F257::ONE, is7)],
            vec![(F257::ONE, b.one()), (-F257::ONE, is7)],
            vec![(F257::ZERO, b.one())],
        );

        let carry_is7 = b.new_var(b.assignment[carry] * b.assignment[is7]);
        b.enforce_mul(carry, is7, carry_is7);

        let msb_and = b.new_var(b.assignment[msb] * b.assignment[carry_is7]);
        b.enforce_mul(msb, carry_is7, msb_and);
        let c_out = b.new_var(b.assignment[msb] + b.assignment[carry_is7] - b.assignment[msb_and]);
        b.enforce_lc_times_one_eq_const(vec![
            (F257::ONE, c_out),
            (-F257::ONE, msb),
            (-F257::ONE, carry_is7),
            (F257::ONE, msb_and),
        ]);
        b.add_constraint(
            vec![(F257::ONE, c_out)],
            vec![(F257::ONE, b.one()), (-F257::ONE, c_out)],
            vec![(F257::ZERO, b.one())],
        );

        let d_i = b.assignment[nib.d]
            .into_bigint()
            .to_bytes_le()
            .get(0)
            .copied()
            .unwrap_or(0) as i32;
        let carry_i = b.assignment[carry]
            .into_bigint()
            .to_bytes_le()
            .get(0)
            .copied()
            .unwrap_or(0) as i32;
        let c_i = b.assignment[c_out]
            .into_bigint()
            .to_bytes_le()
            .get(0)
            .copied()
            .unwrap_or(0) as i32;
        let v = d_i + carry_i - 16 * c_i;
        debug_assert!((-8..=7).contains(&v));
        let out_digit = b.new_var(i32_to_f257(v));
        b.enforce_lc_times_one_eq_const(vec![
            (F257::ONE, out_digit),
            (-F257::ONE, nib.d),
            (-F257::ONE, carry),
            (F257::from(16u64), c_out),
        ]);

        out.push(out_digit);
        carry = c_out;
    }

    out.push(carry);
    out
}

