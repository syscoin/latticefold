//! Tiny-field WE gate scaffolding (Theorem 4.3 direction).
//!
//! This module is the starting point for the “single tiny-field NP relation” path:
//! - Poseidon transcript over `F257` (native)
//! - glue from squeezed digits/bytes to verifier coins
//! - verifier arithmetic represented as bounded integers/limbs inside `F257`
//!
//! For now, we expose a **Poseidon-over-F257 arithmetization** builder that produces a dR1CS
//! instance over `F257` from an operation schedule.

#![cfg(feature = "we_gate")]

use ark_crypto_primitives::sponge::poseidon::PoseidonConfig;
use ark_ff::{BigInteger, Field, PrimeField};

use latticefold::transcript::poseidon::{f257_poseidon_config, F257};
use symphony::dpp_sumcheck::Dr1csBuilder;
use symphony::dpp_poseidon::{
    merge_sparse_dr1cs_share_one, poseidon_sponge_dr1cs_from_ops_with_wiring_and_bytes, Constraint,
    PoseidonByteWiring, PoseidonDr1csWiring, SparseDr1csInstance,
};
use symphony::transcript::PoseidonTraceOp;

// -----------------------------------------------------------------------------
// Tiny-field (F257) byte/limb gadgets
// -----------------------------------------------------------------------------

/// Frog base field modulus (fits in u64).
///
/// This is used only for *boundary canonicalization* of 64-bit challenges.
const FROG_P: u64 = 15912092521325583641u64;
const LIMB_BASE_U64: u64 = 128;
const LIMB_BITS: usize = 7;
const LIMBS_U64: usize = 10; // ceil(64/7)=10
const DIGITS_PER_TRY: usize = 8; // base-257 digits per rejection attempt
const DEFAULT_REJECTION_TRIES: usize = 10;
const LIMBS_U32: usize = 5; // ceil(32/7)=5
// SP1 BabyBear prime (31-bit) used by the lift (default/test value).
const BABYBEAR_P_U32: u32 = 2013265921; // 0x78000001

#[derive(Clone, Debug)]
struct ByteVar {
    /// Field var intended to be in {0..255}.
    byte: usize,
    /// Bit-decomposition (little-endian).
    bits: [usize; 8],
}

// -----------------------------------------------------------------------------
// Balanced base-16 (nibble) gadgets for *bounded* integer arithmetic in F257.
//
// We represent signed small integers in F257 via their canonical mod-257 encoding:
//   -x  == 257-x (for 1 <= x <= 256).
//
// For soundness (no wrap ambiguity), every *intermediate* linear combination we constrain must
// remain in a small integer range, strictly smaller than 257 in magnitude. The helpers below are
// designed for the common CM pattern "small coefficient (≈12 bits) times u32 challenge", where
// per-digit convolution has at most 3 terms, keeping carries small.
// -----------------------------------------------------------------------------

const NIBBLE_BASE: i32 = 16;

#[inline]
fn f257_to_i32_bal(x: F257) -> i32 {
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
fn balanced_digit_from_nibble(
    b: &mut Dr1csBuilder<F257>,
    nibble: usize,
    msb: usize,
) -> usize {
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
fn alloc_bal16_digit(b: &mut Dr1csBuilder<F257>, d: i8) -> usize {
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
fn add_bal16_same_len(
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
///
/// Input digits must be in [-8,7]. Output represents `-x` over integers (no wrap),
/// using a carry chain with carry in [-2,2].
fn neg_bal16_digits(b: &mut Dr1csBuilder<F257>, x: &[usize]) -> (Vec<usize>, usize /* carry_out */) {
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

        // -x_i + carry - out_i - 16*carry_next = 0  <=>  carry - x_i - out_i - 16*carry_next = 0
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
///
/// Returns `(digits, carry_out)` where digits are in [-8,7] and carry_out in [-2,2].
fn sub_bal16_same_len(
    b: &mut Dr1csBuilder<F257>,
    a: &[usize],
    c: &[usize],
) -> (Vec<usize>, usize /* carry_out */) {
    assert_eq!(a.len(), c.len());
    let (neg_c, carry_neg) = neg_bal16_digits(b, c);
    // Enforce no overflow in negation (carry should be 0 for typical fixed-width usage).
    // We leave it unconstrained here; caller can decide whether to enforce 0.
    let _ = carry_neg;
    add_bal16_same_len(b, a, &neg_c)
}

/// Multiply a 3-digit balanced base-16 integer by a balanced-u32 (9 digits).
///
/// This matches the dominant CM pattern "small coeff (≈12 bits) × u32 challenge".
/// Output is 12 digits: 11 balanced digits in [-8,7] plus a final carry digit in [-11,11].
#[inline]
fn mul_bal16_3_by_digits9(
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

/// Multiply a 3-digit balanced base-16 integer by an 18-digit balanced base-16 integer.
///
/// Intended for scaling by `u^2` (where `u` is u32-ish), since `u^2` is represented as 18 digits.
/// Output is 21 digits (little-endian): 20 balanced digits in [-8,7] plus a final carry digit in [-11,11].
#[inline]
fn mul_bal16_3_by_digits18(
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

/// Multiply a 3-digit balanced base-16 integer by a balanced-u32 (9 digits).
///
/// Wrapper around `mul_bal16_3_by_digits9` for historical naming.
fn mul_bal16_3_by_u32(
    b: &mut Dr1csBuilder<F257>,
    coeff3: &[usize; 3],
    u32_bal16_9: &[usize],
) -> [usize; 12] {
    mul_bal16_3_by_digits9(b, coeff3, u32_bal16_9)
}

/// Scale a vector of short-challenge coefficients (3 balanced base-16 digits each)
/// by a bounded-u32 challenge (9 balanced digits).
///
/// Returns per-coeff products as 12 base-16 digits.
#[inline]
fn scale_short_coeffs_by_digits9(
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

#[inline]
fn scale_short_coeffs_by_digits18(
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

fn scale_short_coeffs_by_u32(
    b: &mut Dr1csBuilder<F257>,
    coeffs3: &[[usize; 3]],
    u32_digits9: &[usize],
) -> Vec<[usize; 12]> {
    scale_short_coeffs_by_digits9(b, coeffs3, u32_digits9)
}

/// Rebalance the final digit of a `mul_bal16_small` product.
///
/// `mul_bal16_small` returns a vector of length `la+lb` where the last entry is a carry in [-11,11].
/// For chaining additions, it's convenient to represent values with *all* digits in [-8,7] plus
/// a small carry digit in [-2,2] at one higher position.
///
/// Input: digits[0..L-2] in [-8,7], digits[L-1] in [-11,11]
/// Output: digits'[0..L-1] in [-8,7], digits'[L] in [-2,2]
fn rebalance_tail_pm11_to_pm2(b: &mut Dr1csBuilder<F257>, digits: &[usize]) -> Vec<usize> {
    assert!(!digits.is_empty());
    let l = digits.len();
    let tail = f257_to_i32_bal(b.assignment[digits[l - 1]]);
    debug_assert!((-11..=11).contains(&tail));

    // Choose carry2 in [-1,1] so rem in [-8,7].
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
    // tail = rem + 16*carry2
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

/// Pad and shift a balanced base-16 digit vector to a fixed length.
///
/// Returns a vector of length `target_len` representing `digits * 16^shift`.
fn shift_pad_bal16(
    digits: &[usize],
    shift: usize,
    target_len: usize,
    zero_digit: usize,
) -> Vec<usize> {
    assert!(shift <= target_len);
    assert!(digits.len() + shift <= target_len);
    let mut out = Vec::with_capacity(target_len);
    out.extend(std::iter::repeat(zero_digit).take(shift));
    out.extend_from_slice(digits);
    out.extend(std::iter::repeat(zero_digit).take(target_len - shift - digits.len()));
    out
}

/// Multiply two "u32-ish" balanced base-16 integers (9 digits each) using 3×(3×9) partial products.
///
/// This is the intended building block for bounded-integer CM scalar multiplications where both
/// operands are within a 32-bit-ish envelope (as ensured by `b_bound` reasoning).
fn mul_bal16_9_by_9_u32ish(
    b: &mut Dr1csBuilder<F257>,
    a9: &[usize],
    b9: &[usize],
) -> Vec<usize> {
    assert_eq!(a9.len(), 9);
    assert_eq!(b9.len(), 9);

    // One constrained 0 digit for padding/shifts.
    let zero_digit = alloc_bal16_digit(b, 0);

    let a0: [usize; 3] = [a9[0], a9[1], a9[2]];
    let a1: [usize; 3] = [a9[3], a9[4], a9[5]];
    let a2: [usize; 3] = [a9[6], a9[7], a9[8]];

    // Partial products (3×9): 12 digits with tail carry in [-11,11]; rebalance to 13 digits.
    let p0_raw = mul_bal16_small(b, &a0, b9);
    let p0 = rebalance_tail_pm11_to_pm2(b, &p0_raw);
    let p1_raw = mul_bal16_small(b, &a1, b9);
    let p1 = rebalance_tail_pm11_to_pm2(b, &p1_raw);
    let p2_raw = mul_bal16_small(b, &a2, b9);
    let p2 = rebalance_tail_pm11_to_pm2(b, &p2_raw);
    debug_assert_eq!(p0.len(), 13);
    debug_assert_eq!(p1.len(), 13);
    debug_assert_eq!(p2.len(), 13);

    // Align to the maximum shifted length.
    let target_len = p2.len() + 6; // shift by 6 nibbles for the top chunk.
    let s0 = shift_pad_bal16(&p0, 0, target_len, zero_digit);
    let s1 = shift_pad_bal16(&p1, 3, target_len, zero_digit);
    let s2 = shift_pad_bal16(&p2, 6, target_len, zero_digit);

    // Sum s0 + s1.
    let (mut t01, carry01) = add_bal16_same_len(b, &s0, &s1);
    t01.push(carry01); // extend length by 1

    // Pad s2 to match t01 length and add.
    let mut s2_pad = s2;
    s2_pad.push(zero_digit);
    debug_assert_eq!(t01.len(), s2_pad.len());
    let (mut out, carry) = add_bal16_same_len(b, &t01, &s2_pad);
    out.push(carry);
    out
}

/// Multiply two u32-ish balanced base-16 integers (9 digits each) and return a fixed-width output.
///
/// Output length is `out_len` digits (little-endian). If the raw product has fewer digits, we pad with 0.
/// If it has more digits, we *enforce* the truncated high digits are 0 (so the result is truly fixed-width).
fn mul_u32ish9_to_fixed_bal16(
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

    // Enforce dropped high digits are exactly 0.
    for &dv in &raw[out_len..] {
        b.enforce_var_eq_const(dv, F257::ZERO);
    }
    raw[..out_len].to_vec()
}

/// Multiply an arbitrary-length balanced base-16 integer by a 9-digit balanced-u32-ish integer.
///
/// This is implemented by chunking `a` into 3-digit blocks and using the existing `3×9` gadget.
fn mul_bal16_long_by_u32ish9(
    b: &mut Dr1csBuilder<F257>,
    a: &[usize],
    b9: &[usize],
) -> Vec<usize> {
    assert_eq!(b9.len(), 9);
    if a.is_empty() {
        return vec![alloc_bal16_digit(b, 0)];
    }
    let zero = alloc_bal16_digit(b, 0);
    let blocks = (a.len() + 2) / 3;

    // Accumulate in a fixed length with enough headroom.
    // Each block contributes up to 13 digits (after rebalance), shifted by 3*blk.
    let target_len = 3 * blocks + 13 + 1;
    let mut acc = vec![zero; target_len];

    for blk in 0..blocks {
        let start = blk * 3;
        let end = core::cmp::min(start + 3, a.len());
        let mut coeff3 = [zero; 3];
        for j in 0..(end - start) {
            coeff3[j] = a[start + j];
        }
        let raw = mul_bal16_small(b, &coeff3, b9); // len 12
        let reb = rebalance_tail_pm11_to_pm2(b, &raw); // len 13
        let shifted = shift_pad_bal16(&reb, blk * 3, target_len, zero);
        let (new_acc, carry) = add_bal16_same_len(b, &acc, &shifted);
        acc = new_acc;
        // propagate carry into next digit by appending then trimming (keep fixed target_len)
        // carry is in [-2,2], so we can just add it into the top digit with normalization.
        let top = acc[target_len - 1];
        let carry_digit = carry;
        let (top_sum, top_carry) = add_bal16_same_len(b, &[top], &[carry_digit]);
        acc[target_len - 1] = top_sum[0];
        // Enforce overflow beyond target_len is zero.
        b.enforce_var_eq_const(top_carry, F257::ZERO);
    }
    acc
}

/// Multiply two arbitrary-length balanced base-16 integers (little-endian).
///
/// Requires each input digit var to be constrained to `[-8,7]` (as produced by the alloc/helpers).
/// The output is a balanced base-16 digit vector (little-endian) whose length depends on inputs.
///
/// Implementation strategy:
/// - chunk the shorter input into 3-digit blocks
/// - multiply each block (3×|long|) via `mul_bal16_small`
/// - rebalance the tail carry, shift by 3*blk, and accumulate with `add_bal16_same_len`
fn mul_bal16_long_by_long(
    b: &mut Dr1csBuilder<F257>,
    a: &[usize],
    bb: &[usize],
) -> Vec<usize> {
    if a.is_empty() || bb.is_empty() {
        return vec![alloc_bal16_digit(b, 0)];
    }
    if a.len().min(bb.len()) <= 3 {
        let raw = mul_bal16_small(b, a, bb);
        // Normalize tail carry so downstream addition logic can treat it uniformly.
        return rebalance_tail_pm11_to_pm2(b, &raw);
    }

    // Ensure `short` is the shorter input.
    let (short, long) = if a.len() <= bb.len() { (a, bb) } else { (bb, a) };

    let zero = alloc_bal16_digit(b, 0);
    let blocks = (short.len() + 2) / 3;

    // For each block: mul_bal16_small(3 × long_len) returns (long_len + 3 + 1carry) digits,
    // then rebalance adds one more digit.
    let per_block_len = long.len() + 3 + 1 + 1; // = long.len() + 5
    let target_len = per_block_len + 3 * (blocks - 1) + 2; // +2 headroom for accumulation carry
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

        // Fold the per-add carry into the top digit and enforce no overflow beyond target_len.
        let top = acc[target_len - 1];
        let (top_sum, top_carry) = add_bal16_same_len(b, &[top], &[carry]);
        acc[target_len - 1] = top_sum[0];
        b.enforce_var_eq_const(top_carry, F257::ZERO);
    }
    acc
}

/// Multiply two balanced base-16 digit vectors (little-endian), specialized for the case
/// `min(a.len(), b.len()) <= 3` so per-digit convolution has at most 3 terms and carries stay
/// within [-11,11], which we can range-check cheaply.
fn mul_bal16_small(
    b: &mut Dr1csBuilder<F257>,
    a: &[usize],
    bb: &[usize],
) -> Vec<usize> {
    let la = a.len();
    let lb = bb.len();
    assert!(la > 0 && lb > 0);
    assert!(la.min(lb) <= 3, "mul_bal16_small requires min(len) <= 3");

    let mut out: Vec<usize> = Vec::with_capacity(la + lb);
    let mut carry_i32: i32 = 0;
    let mut carry_var = b.new_var(F257::ZERO);
    b.enforce_var_eq_const(carry_var, F257::ZERO);

    for k in 0..(la + lb - 1) {
        // Witness sum in i32 using centered lifts.
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

        // Choose balanced digit in [-8,7] and carry in [-11,11] such that:
        // sum = digit + 16*carry.
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
        assert!((-11..=11).contains(&carry), "carry out of expected range: {carry} from sum {sum}");

        let digit_var = alloc_bal16_digit(b, rem as i8);
        let carry_out_var = alloc_carry_pm11(b, carry);

        // Enforce: carry_in + Σ prods - digit - 16*carry_out == 0
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

    // Final carry (may be nonzero); return as an extra digit (still in [-11,11], so not nibble-range).
    // For the CM use-cases we care about, we keep the carry as a separate var to avoid widening digits.
    // Here, append it as a final base-16 digit by requiring it be in [-8,7] when possible; otherwise
    // just append 0 and leave carry implicit to the caller (tests decode from assignment anyway).
    //
    // In practice: for 3x9 (12-bit * u32_bal) the final carry is small (often 0/1).
    out.push(carry_var);
    out
}

/// Convert a 32-bit unsigned integer represented as 4 little-endian byte vars (each in 0..255)
/// into balanced base-16 digits (little-endian) of length 9.
///
/// Output digits are in [-8,7] for positions 0..7, and the final digit is a carry in {0,1}.
fn u32_bytes_to_bal16_digits(
    b: &mut Dr1csBuilder<F257>,
    bytes_le: [usize; 4],
) -> Vec<usize> {
    // Build 8 nibbles (little-endian).
    struct Nib {
        d: usize,        // nibble var 0..15
        bits: [usize; 4],// bit vars (LE)
        msb: usize,      // bits[3]
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

    // Balance digits with a carry chain.
    let mut out: Vec<usize> = Vec::with_capacity(9);
    let mut carry = b.new_var(F257::ZERO);
    b.enforce_var_eq_const(carry, F257::ZERO);
    // carry is boolean
    b.add_constraint(
        vec![(F257::ONE, carry)],
        vec![(F257::ONE, b.one()), (-F257::ONE, carry)],
        vec![(F257::ZERO, b.one())],
    );

    for nib in &nibbles {
        // is7 = (b0 & b1 & b2 & !msb)
        let b0 = nib.bits[0];
        let b1 = nib.bits[1];
        let b2 = nib.bits[2];
        let msb = nib.msb;
        let not_msb = b.new_var(F257::ONE - b.assignment[msb]);
        b.enforce_lc_times_one_eq_const(vec![(F257::ONE, not_msb), (F257::ONE, msb), (-F257::ONE, b.one())]);

        let t01 = b.new_var(b.assignment[b0] * b.assignment[b1]);
        b.enforce_mul(b0, b1, t01);
        let t012 = b.new_var(b.assignment[t01] * b.assignment[b2]);
        b.enforce_mul(t01, b2, t012);
        let is7 = b.new_var(b.assignment[t012] * b.assignment[not_msb]);
        b.enforce_mul(t012, not_msb, is7);
        // is7 boolean
        b.add_constraint(
            vec![(F257::ONE, is7)],
            vec![(F257::ONE, b.one()), (-F257::ONE, is7)],
            vec![(F257::ZERO, b.one())],
        );

        // carry_is7 = carry & is7
        let carry_is7 = b.new_var(b.assignment[carry] * b.assignment[is7]);
        b.enforce_mul(carry, is7, carry_is7);

        // c_out = msb OR carry_is7
        // c_out = msb + carry_is7 - msb*carry_is7
        let msb_and = b.new_var(b.assignment[msb] * b.assignment[carry_is7]);
        b.enforce_mul(msb, carry_is7, msb_and);
        let c_out = b.new_var(b.assignment[msb] + b.assignment[carry_is7] - b.assignment[msb_and]);
        b.enforce_lc_times_one_eq_const(vec![
            (F257::ONE, c_out),
            (-F257::ONE, msb),
            (-F257::ONE, carry_is7),
            (F257::ONE, msb_and),
        ]);
        // c_out boolean
        b.add_constraint(
            vec![(F257::ONE, c_out)],
            vec![(F257::ONE, b.one()), (-F257::ONE, c_out)],
            vec![(F257::ZERO, b.one())],
        );

        // out_digit = d + carry - 16*c_out
        // Witness value (for downstream gadgets / tests).
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

    // Final carry digit in {0,1}.
    out.push(carry);
    out
}

fn alloc_bool<F: PrimeField>(b: &mut Dr1csBuilder<F>, bit: bool) -> usize {
    let v = b.new_var(if bit { F::ONE } else { F::ZERO });
    // v*(1-v)=0
    b.add_constraint(
        vec![(F::ONE, v)],
        vec![(F::ONE, b.one()), (-F::ONE, v)],
        vec![(F::ZERO, b.one())],
    );
    v
}

fn alloc_byte<F: PrimeField>(b: &mut Dr1csBuilder<F>, v8: u8) -> ByteVar {
    let mut bits = [0usize; 8];
    for i in 0..8 {
        bits[i] = alloc_bool::<F>(b, ((v8 >> i) & 1) == 1);
    }
    let v = b.new_var(F::from(v8 as u64));

    // v = Σ 2^i * bits[i]
    let mut lc = vec![(F::ONE, v)];
    let mut pow = F::ONE;
    for i in 0..8 {
        lc.push((-pow, bits[i]));
        pow *= F::from(2u64);
    }
    b.enforce_lc_times_one_eq_const(lc);

    ByteVar { byte: v, bits }
}

fn alloc_u64_as_bytes_le<F: PrimeField>(b: &mut Dr1csBuilder<F>, x: u64) -> [ByteVar; 8] {
    let mut out: [ByteVar; 8] = [
        alloc_byte::<F>(b, 0),
        alloc_byte::<F>(b, 0),
        alloc_byte::<F>(b, 0),
        alloc_byte::<F>(b, 0),
        alloc_byte::<F>(b, 0),
        alloc_byte::<F>(b, 0),
        alloc_byte::<F>(b, 0),
        alloc_byte::<F>(b, 0),
    ];
    let bytes = x.to_le_bytes();
    for i in 0..8 {
        out[i] = alloc_byte::<F>(b, bytes[i]);
    }
    out
}

fn decompose_existing_byte_var_to_bits<F: PrimeField>(
    b: &mut Dr1csBuilder<F>,
    byte_var: usize,
) -> [usize; 8] {
    // Constrain: byte_var = Σ 2^i * bit_i, with bit_i boolean.
    let mut bits = [0usize; 8];
    let v8 = b.assignment[byte_var]
        .into_bigint()
        .to_bytes_le()
        .get(0)
        .copied()
        .unwrap_or(0) as u16;
    // We expect Poseidon "byte wiring" vars to already be in 0..=255.
    debug_assert!(v8 <= 255);
    for i in 0..8 {
        bits[i] = alloc_bool::<F>(b, ((v8 >> i) & 1) == 1);
    }
    let mut lc = vec![(F::ONE, byte_var)];
    let mut pow = F::ONE;
    for i in 0..8 {
        lc.push((-pow, bits[i]));
        pow *= F::from(2u64);
    }
    b.enforce_lc_times_one_eq_const(lc);
    bits
}

fn alloc_u7<F: PrimeField>(b: &mut Dr1csBuilder<F>, v7: u8) -> usize {
    debug_assert!(v7 < 128);
    let mut bits = [0usize; LIMB_BITS];
    for i in 0..LIMB_BITS {
        bits[i] = alloc_bool::<F>(b, ((v7 >> i) & 1) == 1);
    }
    let v = b.new_var(F::from(v7 as u64));
    let mut lc = vec![(F::ONE, v)];
    let mut pow = F::ONE;
    for i in 0..LIMB_BITS {
        lc.push((-pow, bits[i]));
        pow *= F::from(2u64);
    }
    b.enforce_lc_times_one_eq_const(lc);
    v
}

fn alloc_u2_from_u8<F: PrimeField>(b: &mut Dr1csBuilder<F>, v2: u8) -> usize {
    debug_assert!(v2 <= 2);
    let b0 = alloc_bool::<F>(b, (v2 & 1) == 1);
    let b1 = alloc_bool::<F>(b, (v2 & 2) == 2);
    let v = b.new_var(F::from(v2 as u64));
    b.enforce_lc_times_one_eq_const(vec![
        (F::ONE, v),
        (-F::ONE, b0),
        (-F::from(2u64), b1),
    ]);
    v
}

#[inline]
fn const_zero<F: PrimeField>(b: &mut Dr1csBuilder<F>) -> usize {
    let v = b.new_var(F::ZERO);
    b.enforce_var_eq_const(v, F::ZERO);
    v
}

fn frog_p_base128_digits_le() -> [u8; LIMBS_U64] {
    let mut out = [0u8; LIMBS_U64];
    let mut t = FROG_P;
    for i in 0..LIMBS_U64 {
        out[i] = (t & (LIMB_BASE_U64 - 1)) as u8;
        t >>= LIMB_BITS;
    }
    out
}

fn digit_to_base128_limbs<F: PrimeField>(
    b: &mut Dr1csBuilder<F>,
    d_var: usize,
) -> (usize /* l0 in 0..127 */, usize /* l1 in 0..2 */) {
    let bytes = b.assignment[d_var].into_bigint().to_bytes_le();
    // Digits are in 0..=256, so at most 2 bytes.
    let du16: u16 = (bytes.get(0).copied().unwrap_or(0) as u16)
        | ((bytes.get(1).copied().unwrap_or(0) as u16) << 8);
    debug_assert!(du16 < 257);
    let l0_u8 = (du16 & 127) as u8;
    let l1_u8 = (du16 >> 7) as u8; // 0,1,2
    let l0 = alloc_u7::<F>(b, l0_u8);
    let l1 = alloc_u2_from_u8::<F>(b, l1_u8);
    // Enforce d = l0 + 128*l1 in the field.
    b.enforce_lc_times_one_eq_const(vec![
        (F::ONE, d_var),
        (-F::ONE, l0),
        (-F::from(LIMB_BASE_U64), l1),
    ]);
    (l0, l1)
}

#[inline]
fn digit_u16<F: PrimeField>(b: &Dr1csBuilder<F>, d: usize) -> u16 {
    let bytes = b.assignment[d].into_bigint().to_bytes_le();
    (bytes.get(0).copied().unwrap_or(0) as u16) | ((bytes.get(1).copied().unwrap_or(0) as u16) << 8)
}

/// Map a base-257 digit `d ∈ {0..=256}` to a byte `b ∈ {0..=255}` via the transcript rule:
/// `256 -> 0`, else `b=d`.
///
/// In F257 arithmetic this is: `b = d + is_eq256(d)`, since `256 ≡ -1 (mod 257)`.
fn digit_to_byte_var<F: PrimeField>(b: &mut Dr1csBuilder<F>, d: usize) -> usize {
    let du16 = digit_u16::<F>(b, d);
    debug_assert!(du16 < 257);
    let is256 = alloc_bool::<F>(b, du16 == 256);

    // Enforce is256 <-> (d == 256) via inverse trick.
    let diff = b.new_var(b.assignment[d] - F::from(256u64));
    b.enforce_lc_times_one_eq_const(vec![
        (F::ONE, diff),
        (-F::ONE, d),
        (F::from(256u64), b.one()),
    ]);
    let inv = b.new_var(if du16 == 256 {
        F::ZERO
    } else {
        (b.assignment[d] - F::from(256u64)).inverse().unwrap()
    });
    let prod = b.new_var(b.assignment[diff] * b.assignment[inv]);
    b.enforce_mul(diff, inv, prod);
    // prod = 1 - is256
    b.enforce_lc_times_one_eq_const(vec![
        (F::ONE, prod),
        (F::ONE, is256),
        (-F::ONE, b.one()),
    ]);
    // diff * is256 = 0
    let z = b.new_var(F::ZERO);
    b.enforce_var_eq_const(z, F::ZERO);
    b.enforce_mul(diff, is256, z);

    // byte = d + is256  (since 256 -> 0)
    let byte = b.new_var(b.assignment[d] + b.assignment[is256]);
    b.enforce_lc_times_one_eq_const(vec![
        (F::ONE, byte),
        (-F::ONE, d),
        (-F::ONE, is256),
    ]);

    // Range-check byte to 8 bits.
    let _bits = decompose_existing_byte_var_to_bits::<F>(b, byte);
    byte
}

/// Interpret the transcript's `get_challenge()` digit block as a **bounded u32** and return it as
/// base-128 limbs (little-endian).
///
/// Semantics must match `latticefold-plus/src/transcript.rs:get_challenge`:
/// - consume a fixed 8-digit block (schedule-only)
/// - map the first 4 digits through the byte view (256 -> 0)
/// - pack into a u32, but represent as base-128 limbs (so it embeds soundly in F257)
fn bounded_u32_from_8_digits_base128(
    b: &mut Dr1csBuilder<F257>,
    digits: &[usize; DIGITS_PER_TRY],
) -> ([usize; LIMBS_U32], [usize; 4], Vec<usize>, Vec<usize>) {
    // First 4 digits -> bytes in 0..255.
    let mut bytes = [0usize; 4];
    for i in 0..4 {
        bytes[i] = digit_to_byte_var::<F257>(b, digits[i]);
    }

    // Balanced base-16 digits (len 9) for the same u32 (used by CM mul gadgets).
    let bal16_digits = u32_bytes_to_bal16_digits(b, bytes);
    let bal16_sq_digits = mul_u32ish9_to_fixed_bal16(b, &bal16_digits, &bal16_digits, 18);

    // Bits 0..31 of the u32, little-endian.
    let mut bits32: [usize; 32] = [0usize; 32];
    for i in 0..4 {
        let bits = decompose_existing_byte_var_to_bits::<F257>(b, bytes[i]);
        for j in 0..8 {
            bits32[i * 8 + j] = bits[j];
        }
    }

    // Group bits into 7-bit base-128 limbs.
    let mut limbs = [0usize; LIMBS_U32];
    for li in 0..LIMBS_U32 {
        let start = li * LIMB_BITS;
        let end = usize::min(start + LIMB_BITS, 32);
        // Witness limb value from bits.
        let mut limb_u8: u8 = 0;
        for j in start..end {
            let bit = b.assignment[bits32[j]]
                .into_bigint()
                .to_bytes_le()
                .get(0)
                .copied()
                .unwrap_or(0);
            debug_assert!(bit == 0 || bit == 1);
            limb_u8 |= (bit as u8) << (j - start);
        }
        let limb = b.new_var(F257::from(limb_u8 as u64));
        // Enforce limb = Σ 2^k * bits32[start+k]
        let mut lc = vec![(F257::ONE, limb)];
        for j in start..end {
            let p2 = F257::from(1u64 << (j - start));
            lc.push((-p2, bits32[j]));
        }
        b.enforce_lc_times_one_eq_const(lc);
        limbs[li] = limb;
    }

    (limbs, bytes, bal16_digits, bal16_sq_digits)
}

fn base128_add10<F: PrimeField>(
    b: &mut Dr1csBuilder<F>,
    a: &[usize; LIMBS_U64],
    c: &[usize; LIMBS_U64],
) -> [usize; LIMBS_U64] {
    let mut out = [0usize; LIMBS_U64];
    let mut carry = const_zero::<F>(b);
    for i in 0..LIMBS_U64 {
        let ai = b.assignment[a[i]].into_bigint().to_bytes_le().get(0).copied().unwrap_or(0) as u16;
        let ci = b.assignment[c[i]].into_bigint().to_bytes_le().get(0).copied().unwrap_or(0) as u16;
        let carry_u16 = b.assignment[carry].into_bigint().to_bytes_le().get(0).copied().unwrap_or(0) as u16;
        debug_assert!(ai < 128 && ci < 128 && carry_u16 <= 1);
        let sum = ai + ci + carry_u16; // <= 255
        let out_i = (sum & 127) as u8;
        let carry_next_u8 = (sum >> 7) as u8; // 0 or 1
        let out_var = alloc_u7::<F>(b, out_i);
        let carry_next = if i == LIMBS_U64 - 1 {
            const_zero::<F>(b)
        } else {
            alloc_bool::<F>(b, carry_next_u8 == 1)
        };
        b.enforce_lc_times_one_eq_const(vec![
            (F::ONE, out_var),
            (F::from(LIMB_BASE_U64), carry_next),
            (-F::ONE, a[i]),
            (-F::ONE, c[i]),
            (-F::ONE, carry),
        ]);
        out[i] = out_var;
        carry = carry_next;
    }
    b.enforce_var_eq_const(carry, F::ZERO);
    out
}

fn base128_shift1_10<F: PrimeField>(b: &mut Dr1csBuilder<F>, a: &[usize; LIMBS_U64]) -> [usize; LIMBS_U64] {
    let mut out = [0usize; LIMBS_U64];
    out[0] = const_zero::<F>(b);
    for i in 1..LIMBS_U64 {
        out[i] = a[i - 1];
    }
    out
}

fn base128_mul2_10<F: PrimeField>(b: &mut Dr1csBuilder<F>, a: &[usize; LIMBS_U64]) -> [usize; LIMBS_U64] {
    base128_add10::<F>(b, a, a)
}

fn base128_lt_const10<F: PrimeField>(
    b: &mut Dr1csBuilder<F>,
    x: &[usize; LIMBS_U64],
    c_digits: &[u8; LIMBS_U64],
) -> usize {
    // Borrow chain for x - c. Final borrow = 1 iff x < c.
    let mut borrow = const_zero::<F>(b);
    for i in 0..LIMBS_U64 {
        let xi = b.assignment[x[i]].into_bigint().to_bytes_le().get(0).copied().unwrap_or(0) as i16;
        let bi = b.assignment[borrow].into_bigint().to_bytes_le().get(0).copied().unwrap_or(0) as i16;
        debug_assert!(xi >= 0 && xi < 128);
        debug_assert!(bi == 0 || bi == 1);
        let mut t = xi - (c_digits[i] as i16) - bi;
        let borrow_next_u8 = if t < 0 { 1u8 } else { 0u8 };
        if t < 0 {
            t += LIMB_BASE_U64 as i16;
        }
        let diff = alloc_u7::<F>(b, (t as u8) & 127);
        let borrow_next = alloc_bool::<F>(b, borrow_next_u8 == 1);
        b.enforce_lc_times_one_eq_const(vec![
            (F::ONE, x[i]),
            (-F::from(c_digits[i] as u64), b.one()),
            (-F::ONE, borrow),
            (F::from(LIMB_BASE_U64), borrow_next),
            (-F::ONE, diff),
        ]);
        borrow = borrow_next;
    }
    borrow
}

fn mux_base128_10<F: PrimeField>(
    b: &mut Dr1csBuilder<F>,
    sel: usize, // boolean
    a: &[usize; LIMBS_U64],
    c: &[usize; LIMBS_U64],
) -> [usize; LIMBS_U64] {
    let mut out = [0usize; LIMBS_U64];
    for i in 0..LIMBS_U64 {
        let diff_val = b.assignment[c[i]] - b.assignment[a[i]];
        let diff = b.new_var(diff_val);
        b.enforce_lc_times_one_eq_const(vec![(F::ONE, diff), (-F::ONE, c[i]), (F::ONE, a[i])]);
        let prod = b.new_var(b.assignment[sel] * b.assignment[diff]);
        b.enforce_mul(sel, diff, prod);
        let out_val = b.assignment[a[i]] + b.assignment[prod];
        let out_var = b.new_var(out_val);
        b.enforce_lc_times_one_eq_const(vec![(F::ONE, out_var), (-F::ONE, a[i]), (-F::ONE, prod)]);
        // Range-check limb (0..127).
        let out_u8 = (out_val.into_bigint().to_bytes_le().get(0).copied().unwrap_or(0) as u8) & 127;
        let out_rc = alloc_u7::<F>(b, out_u8);
        b.enforce_lc_times_one_eq_const(vec![(F::ONE, out_var), (-F::ONE, out_rc)]);
        out[i] = out_var;
    }
    out
}

fn compute_u_from_8_digits_base257_in_base128<F: PrimeField>(
    b: &mut Dr1csBuilder<F>,
    digits: &[usize; DIGITS_PER_TRY],
) -> [usize; LIMBS_U64] {
    let mut u = [0usize; LIMBS_U64];
    for i in 0..LIMBS_U64 {
        u[i] = const_zero::<F>(b);
    }
    for i in (0..DIGITS_PER_TRY).rev() {
        let shift = base128_shift1_10::<F>(b, &u);
        let two_shift = base128_mul2_10::<F>(b, &shift);
        let u257 = base128_add10::<F>(b, &u, &two_shift);
        let (l0, l1) = digit_to_base128_limbs::<F>(b, digits[i]);
        let mut d_ext = [0usize; LIMBS_U64];
        d_ext[0] = l0;
        d_ext[1] = l1;
        for j in 2..LIMBS_U64 {
            d_ext[j] = const_zero::<F>(b);
        }
        u = base128_add10::<F>(b, &u257, &d_ext);
    }
    u
}

fn sample_frog_coin_unrolled_rejection_8_digits<F: PrimeField>(
    b: &mut Dr1csBuilder<F>,
    digit_vars: &[usize], // length = tries*8
    tries: usize,
) -> ([usize; LIMBS_U64], usize /* found */) {
    assert_eq!(digit_vars.len(), tries * DIGITS_PER_TRY);
    let p_digits = frog_p_base128_digits_le();

    let mut found = const_zero::<F>(b);
    let mut selected = [0usize; LIMBS_U64];
    for i in 0..LIMBS_U64 {
        selected[i] = const_zero::<F>(b);
    }

    for t in 0..tries {
        let mut d = [0usize; DIGITS_PER_TRY];
        for i in 0..DIGITS_PER_TRY {
            d[i] = digit_vars[t * DIGITS_PER_TRY + i];
        }
        let u = compute_u_from_8_digits_base257_in_base128::<F>(b, &d);
        let lt = base128_lt_const10::<F>(b, &u, &p_digits); // 1 iff u < p

        let not_found = bool_not::<F>(b, found);
        let take = bool_and::<F>(b, not_found, lt);

        selected = mux_base128_10::<F>(b, take, &selected, &u);
        found = bool_or::<F>(b, found, lt);
    }

    (selected, found)
}

fn short_challenge_coeff_from_byte_var<F: PrimeField>(
    b: &mut Dr1csBuilder<F>,
    byte_var: usize,
    u: u64,
) -> usize {
    // Matches `we_gate_arith::short_challenge_coeff_from_byte`:
    // coeff = (byte % u) - (u/2), with u a power of two <= 256.
    debug_assert!(u.is_power_of_two());
    debug_assert!(u <= 256);
    let bits = decompose_existing_byte_var_to_bits::<F>(b, byte_var);

    let logu = u.trailing_zeros() as usize;
    let half = u / 2;

    // low = Σ_{i<logu} 2^i * bits[i]
    let low_u64 = (b.assignment[byte_var]
        .into_bigint()
        .to_bytes_le()
        .get(0)
        .copied()
        .unwrap_or(0) as u64)
        & (u - 1);
    let low = b.new_var(F::from(low_u64));
    let mut lc = vec![(F::ONE, low)];
    let mut p2 = F::ONE;
    for i in 0..logu {
        lc.push((-p2, bits[i]));
        p2 = p2.double();
    }
    b.enforce_lc_times_one_eq_const(lc);

    // coeff = low - half
    // Range note: for u=256, coeff in [-128,127], which embeds uniquely in F257 as integers mod 257.
    let coeff_u64 = ((low_u64 as i64) - (half as i64)).rem_euclid(257) as u64;
    let coeff = b.new_var(F::from(coeff_u64));
    b.enforce_lc_times_one_eq_const(vec![
        (F::ONE, coeff),
        (-F::ONE, low),
        (F::from(half), b.one()),
    ]);
    coeff
}

fn short_challenge_from_bytes<F: PrimeField>(
    b: &mut Dr1csBuilder<F>,
    bytes: &[usize],
    lambda: usize,
    ring_dim: usize,
) -> Vec<usize> {
    // Same parameterization as LF+/WE: u = 2^(lambda / d).
    debug_assert_eq!(bytes.len(), ring_dim);
    let exp = (lambda / ring_dim) as u32;
    let u = 1u64 << exp;
    bytes
        .iter()
        .map(|&by| short_challenge_coeff_from_byte_var::<F>(b, by, u))
        .collect()
}

fn short_challenge_from_digits_128(
    b: &mut Dr1csBuilder<F257>,
    digit_vars: &[usize], // length = ring_dim
    ring_dim: usize,
) -> (Vec<usize> /* byte_vars */, Vec<usize> /* coeff_vars */, Vec<[usize; 3]> /* coeff digits */) {
    debug_assert_eq!(digit_vars.len(), ring_dim);
    let mut byte_vars = Vec::with_capacity(ring_dim);
    for &d in digit_vars {
        byte_vars.push(digit_to_byte_var::<F257>(b, d));
    }
    let coeff_vars = short_challenge_from_bytes::<F257>(b, &byte_vars, 128, ring_dim);

    // Constrain balanced base-16 digits for each coeff (range [-128,127] always holds here).
    let mut coeff_bal16_digits: Vec<[usize; 3]> = Vec::with_capacity(ring_dim);
    let zero_digit = alloc_bal16_digit(b, 0);
    for &cv in &coeff_vars {
        let v = f257_to_i32_bal(b.assignment[cv]); // in [-128,127]
        debug_assert!((-128..=127).contains(&v));
        // Choose d0 in [-8,7] and d1 in [-8,7] such that v = d0 + 16*d1.
        let mut d0 = v % 16;
        if d0 > 7 {
            d0 -= 16;
        }
        if d0 < -8 {
            d0 += 16;
        }
        let d1 = (v - d0) / 16;
        debug_assert!((-8..=7).contains(&d0));
        debug_assert!((-8..=7).contains(&d1));

        let d0v = alloc_bal16_digit(b, d0 as i8);
        let d1v = alloc_bal16_digit(b, d1 as i8);
        // Enforce cv = d0 + 16*d1 (safe: RHS in [-136,119] < 257).
        b.enforce_lc_times_one_eq_const(vec![
            (F257::ONE, cv),
            (-F257::ONE, d0v),
            (-F257::from(16u64), d1v),
        ]);
        coeff_bal16_digits.push([d0v, d1v, zero_digit]);
    }

    (byte_vars, coeff_vars, coeff_bal16_digits)
}

#[inline]
fn bool_not<F: PrimeField>(b: &mut Dr1csBuilder<F>, x: usize) -> usize {
    let v = b.new_var(F::ONE - b.assignment[x]);
    b.add_constraint(
        vec![(F::ONE, b.one()), (-F::ONE, x)],
        vec![(F::ONE, b.one())],
        vec![(F::ONE, v)],
    );
    v
}

#[inline]
fn bool_and<F: PrimeField>(b: &mut Dr1csBuilder<F>, x: usize, y: usize) -> usize {
    let v = b.new_var(b.assignment[x] * b.assignment[y]);
    b.enforce_mul(x, y, v);
    v
}

#[inline]
fn bool_or<F: PrimeField>(b: &mut Dr1csBuilder<F>, x: usize, y: usize) -> usize {
    // x OR y = x + y - x*y  (for boolean x,y)
    let xy = bool_and::<F>(b, x, y);
    let v = b.new_var(b.assignment[x] + b.assignment[y] - b.assignment[xy]);
    b.enforce_lc_times_one_eq_const(vec![
        (F::ONE, v),
        (-F::ONE, x),
        (-F::ONE, y),
        (F::ONE, xy),
    ]);
    v
}

#[inline]
fn bool_eq_bit_to_const<F: PrimeField>(b: &mut Dr1csBuilder<F>, bit: usize, c: bool) -> usize {
    if c { bit } else { bool_not::<F>(b, bit) }
}

fn byte_eq_const<F: PrimeField>(b: &mut Dr1csBuilder<F>, x: &ByteVar, c: u8) -> usize {
    // eq = ∏_i (x_i == c_i)
    let mut eq = b.new_var(F::ONE);
    b.enforce_var_eq_const(eq, F::ONE);
    for i in 0..8 {
        let ci = ((c >> i) & 1) == 1;
        let eqi = bool_eq_bit_to_const::<F>(b, x.bits[i], ci);
        eq = bool_and::<F>(b, eq, eqi);
    }
    eq
}

fn byte_lt_const<F: PrimeField>(b: &mut Dr1csBuilder<F>, x: &ByteVar, c: u8) -> usize {
    // MSB-first bitwise compare for x < c.
    let mut eq_hi = b.new_var(F::ONE);
    b.enforce_var_eq_const(eq_hi, F::ONE);
    let mut lt = b.new_var(F::ZERO);
    b.enforce_var_eq_const(lt, F::ZERO);

    for k in (0..8).rev() {
        let ck = ((c >> k) & 1) == 1;
        if ck {
            let xk0 = bool_not::<F>(b, x.bits[k]);
            let term = bool_and::<F>(b, eq_hi, xk0);
            lt = bool_or::<F>(b, lt, term);
        }
        let eqk = bool_eq_bit_to_const::<F>(b, x.bits[k], ck);
        eq_hi = bool_and::<F>(b, eq_hi, eqk);
    }
    lt
}

fn u64_lt_const_le_bytes<F: PrimeField>(b: &mut Dr1csBuilder<F>, x: &[ByteVar; 8], c: u64) -> usize {
    let c_bytes = c.to_le_bytes();
    let mut eq_hi = b.new_var(F::ONE);
    b.enforce_var_eq_const(eq_hi, F::ONE);
    let mut lt = b.new_var(F::ZERO);
    b.enforce_var_eq_const(lt, F::ZERO);

    // Compare MSB-first: byte 7 down to 0.
    for i in (0..8).rev() {
        let eq_byte = byte_eq_const::<F>(b, &x[i], c_bytes[i]);
        let lt_byte = byte_lt_const::<F>(b, &x[i], c_bytes[i]);
        let term = bool_and::<F>(b, eq_hi, lt_byte);
        lt = bool_or::<F>(b, lt, term);
        eq_hi = bool_and::<F>(b, eq_hi, eq_byte);
    }
    lt
}

fn u32_lt_const_le_bytes<F: PrimeField>(b: &mut Dr1csBuilder<F>, x: &[ByteVar; 4], c: u32) -> usize {
    let c_bytes = c.to_le_bytes();
    let mut eq_hi = b.new_var(F::ONE);
    b.enforce_var_eq_const(eq_hi, F::ONE);
    let mut lt = b.new_var(F::ZERO);
    b.enforce_var_eq_const(lt, F::ZERO);

    // Compare MSB-first: byte 3 down to 0.
    for i in (0..4).rev() {
        let eq_byte = byte_eq_const::<F>(b, &x[i], c_bytes[i]);
        let lt_byte = byte_lt_const::<F>(b, &x[i], c_bytes[i]);
        let term = bool_and::<F>(b, eq_hi, lt_byte);
        lt = bool_or::<F>(b, lt, term);
        eq_hi = bool_and::<F>(b, eq_hi, eq_byte);
    }
    lt
}

/// Wiring for a BabyBear element encoded as 4 little-endian bytes, constrained to be a canonical
/// 31-bit value in `[0, p_bb)`.
#[derive(Clone, Debug)]
pub struct BabyBear31Wiring {
    /// The 4 byte vars (little-endian), each constrained to be 8-bit.
    pub byte_vars: [usize; 4],
    /// The MSB bit (bit 31) of the u32 encoding (must be 0).
    pub bit31: usize,
    /// Balanced base-16 digits (little-endian), length 9, representing the same u32 value.
    pub bal16_digits: Vec<usize>,
}

/// Wiring for a **centered** BabyBear value derived from canonical bytes.
///
/// If the canonical u32 value is `x ∈ [0, p_bb)`, the centered integer is:
/// - `x` if `x <= floor(p_bb/2)`
/// - `x - p_bb` otherwise (negative)
#[derive(Clone, Debug)]
pub struct BabyBearCenteredWiring {
    /// Canonical BabyBear byte encoding (u32 LE), constrained to `[0, p_bb)`.
    pub byte_vars: [usize; 4],
    /// Whether `x > floor(p_bb/2)` (boolean).
    pub is_neg: usize,
    /// Balanced base-16 digits (little-endian), length 9, representing the centered integer.
    pub centered_bal16_digits: Vec<usize>,
}

/// Constrain 4 byte vars to be a canonical SP1 BabyBear element encoding:
/// - u32 little-endian
/// - bit31 == 0 (31-bit)
/// - value < p_bb
///
/// Returns balanced base-16 digits for the same u32 (for downstream bounded integer gadgets).
pub fn babybear31_from_u32_byte_vars_with_modulus(
    b: &mut Dr1csBuilder<F257>,
    byte_vars_le: [usize; 4],
    p_bb: u32,
) -> BabyBear31Wiring {
    debug_assert!(p_bb < (1u32 << 31), "expected 31-bit modulus");
    // Build ByteVar views (with explicit bit decompositions).
    let b0 = ByteVar { byte: byte_vars_le[0], bits: decompose_existing_byte_var_to_bits::<F257>(b, byte_vars_le[0]) };
    let b1 = ByteVar { byte: byte_vars_le[1], bits: decompose_existing_byte_var_to_bits::<F257>(b, byte_vars_le[1]) };
    let b2 = ByteVar { byte: byte_vars_le[2], bits: decompose_existing_byte_var_to_bits::<F257>(b, byte_vars_le[2]) };
    let b3 = ByteVar { byte: byte_vars_le[3], bits: decompose_existing_byte_var_to_bits::<F257>(b, byte_vars_le[3]) };
    let bytes = [b0, b1, b2, b3];

    // Enforce 31-bit encoding: bit 31 (MSB of byte 3) is 0.
    let bit31 = bytes[3].bits[7];
    b.enforce_var_eq_const(bit31, F257::ZERO);

    // Enforce x < p_bb.
    let lt_p = u32_lt_const_le_bytes::<F257>(b, &bytes, p_bb);
    b.enforce_var_eq_const(lt_p, F257::ONE);

    // Produce balanced base-16 digits.
    let bal16_digits = u32_bytes_to_bal16_digits(b, byte_vars_le);

    BabyBear31Wiring { byte_vars: byte_vars_le, bit31, bal16_digits }
}

/// Constrain bytes to a canonical BabyBear encoding and return the **centered** value in balanced base-16.
pub fn babybear_centered_from_u32_byte_vars_with_modulus(
    b: &mut Dr1csBuilder<F257>,
    byte_vars_le: [usize; 4],
    p_bb: u32,
) -> BabyBearCenteredWiring {
    let w = babybear31_from_u32_byte_vars_with_modulus(b, byte_vars_le, p_bb);
    let half: u32 = p_bb / 2;

    // x <= half  <=>  x < half+1
    let bytes = [
        ByteVar { byte: w.byte_vars[0], bits: decompose_existing_byte_var_to_bits::<F257>(b, w.byte_vars[0]) },
        ByteVar { byte: w.byte_vars[1], bits: decompose_existing_byte_var_to_bits::<F257>(b, w.byte_vars[1]) },
        ByteVar { byte: w.byte_vars[2], bits: decompose_existing_byte_var_to_bits::<F257>(b, w.byte_vars[2]) },
        ByteVar { byte: w.byte_vars[3], bits: decompose_existing_byte_var_to_bits::<F257>(b, w.byte_vars[3]) },
    ];
    let le_half = u32_lt_const_le_bytes::<F257>(b, &bytes, half + 1);
    let is_neg = bool_not::<F257>(b, le_half);

    // Build balanced base-16 digits for (-p_bb) in length 9 (host-known, statement-driven).
    let mut neg_p: i128 = -(p_bb as i128);
    let mut neg_p_digits_i8: [i8; 9] = [0i8; 9];
    for i in 0..9 {
        let mut d = (neg_p % 16) as i32; // in [-15,15]
        if d > 7 { d -= 16; }
        if d < -8 { d += 16; }
        neg_p_digits_i8[i] = d as i8;
        neg_p = (neg_p - d as i128) / 16;
    }
    debug_assert_eq!(neg_p, 0, "neg_p did not fit in 9 balanced digits");

    // Masked (-p) digits: either all-zero or (-p) depending on is_neg.
    let zero_digit = alloc_bal16_digit(b, 0);
    let mut masked_neg_p: Vec<usize> = Vec::with_capacity(9);
    for &di in &neg_p_digits_i8 {
        let dvar = alloc_bal16_digit(b, di);
        let mv = b.new_var(b.assignment[dvar] * b.assignment[is_neg]);
        b.enforce_mul(dvar, is_neg, mv);
        masked_neg_p.push(mv);
    }
    // If is_neg=0, masked_neg_p should be all 0; we don't enforce that directly (it follows from mul),
    // but we keep the explicit zero digit for later padding patterns.
    let _ = zero_digit;

    // centered = x + is_neg * (-p)
    let (centered_digits, carry) = add_bal16_same_len(b, &w.bal16_digits, &masked_neg_p);
    // The centered value is guaranteed to fit in 9 digits for 31-bit primes; enforce no overflow.
    b.enforce_var_eq_const(carry, F257::ZERO);

    BabyBearCenteredWiring {
        byte_vars: w.byte_vars,
        is_neg,
        centered_bal16_digits: centered_digits,
    }
}

/// Default BabyBear modulus wiring (SP1).
#[inline]
pub fn babybear31_from_u32_byte_vars(
    b: &mut Dr1csBuilder<F257>,
    byte_vars_le: [usize; 4],
) -> BabyBear31Wiring {
    babybear31_from_u32_byte_vars_with_modulus(b, byte_vars_le, BABYBEAR_P_U32)
}

#[inline]
fn enforce_var_eq<F: PrimeField>(inst: &mut SparseDr1csInstance<F>, x: usize, y: usize) {
    // Enforce (x - y) * 1 = 0
    inst.constraints.push(Constraint {
        a: vec![(F::ONE, x), (-F::ONE, y)],
        b: vec![(F::ONE, 0)],
        c: vec![(F::ZERO, 0)],
    });
}

/// Boundary-only canonicalization gadget:
/// given an unconstrained 64-bit integer `u` as 8 little-endian bytes, produce
/// `(q, z)` such that:
/// - `q ∈ {0,1}`
/// - `u = z + q * p_frog` as an **integer** (no wrap), enforced via base-128 borrows over a
///   bit-derived base-128 limb view of `u`.
///
/// This is the "single subtract" reduction justified by \(2^{64} < 2p\).
/// Takes raw byte variables that are
/// already constrained to be 8-bit (e.g. Poseidon `SqueezeBytes` wiring).
#[allow(dead_code)]
fn reduce_u64_mod_frog_from_byte_vars<F: PrimeField>(
    b: &mut Dr1csBuilder<F>,
    u_byte_vars: &[usize; 8],
) -> (usize /* q bit */, [usize; LIMBS_U64] /* z base-128 limbs */) {
    // Witness compute.
    let mut u_buf = [0u8; 8];
    for i in 0..8 {
        u_buf[i] = b.assignment[u_byte_vars[i]]
            .into_bigint()
            .to_bytes_le()
            .get(0)
            .copied()
            .unwrap_or(0);
    }
    let u = u64::from_le_bytes(u_buf);
    let q_u8: u8 = if u >= FROG_P { 1 } else { 0 };
    let q = alloc_bool::<F>(b, q_u8 == 1);

    // Decompose u_byte_vars into 64 bits.
    let mut u_bits: Vec<usize> = Vec::with_capacity(64);
    for i in 0..8 {
        let bits = decompose_existing_byte_var_to_bits::<F>(b, u_byte_vars[i]);
        u_bits.extend_from_slice(&bits);
    }
    debug_assert_eq!(u_bits.len(), 64);

    // Allocate base-128 limbs for u derived from bits: limb_j = Σ 2^k * bit_{7j+k}.
    let mut u_limbs = [0usize; LIMBS_U64];
    for j in 0..LIMBS_U64 {
        // Witness limb value from native u.
        let limb_val = ((u >> (LIMB_BITS * j)) & (LIMB_BASE_U64 - 1)) as u64;
        let limb_var = b.new_var(F::from(limb_val));
        let mut lc = vec![(F::ONE, limb_var)];
        let mut pow = F::ONE;
        for k in 0..LIMB_BITS {
            let bit_idx = LIMB_BITS * j + k;
            if bit_idx < 64 {
                lc.push((-pow, u_bits[bit_idx]));
            }
            pow *= F::from(2u64);
        }
        b.enforce_lc_times_one_eq_const(lc);
        u_limbs[j] = limb_var;
    }

    // Enforce q matches u >= p by comparing u_limbs (base-128) against p.
    // Comparator: run a base-128 borrow chain on (u - p) with witnessed borrows.
    let p_digits = frog_p_base128_digits_le();
    let mut bor = b.new_var(F::ZERO);
    b.enforce_var_eq_const(bor, F::ZERO);
    for i in 0..LIMBS_U64 {
        let ui = ((u >> (LIMB_BITS * i)) & (LIMB_BASE_U64 - 1)) as i16;
        let pi = p_digits[i] as i16;
        let bi = if i == 0 {
            0i16
        } else {
            b.assignment[bor]
                .into_bigint()
                .to_bytes_le()
                .get(0)
                .copied()
                .unwrap_or(0) as i16
        };
        let mut t = ui - pi - bi;
        let bor_next_u8 = if t < 0 { 1u8 } else { 0u8 };
        if t < 0 {
            t += LIMB_BASE_U64 as i16;
        }
        let diff_i = b.new_var(F::from((t as u64) & (LIMB_BASE_U64 - 1)));
        // Range-check diff_i to 7 bits by reusing bits (cheap): allocate 7 bools.
        let mut dbits = [0usize; LIMB_BITS];
        for k in 0..LIMB_BITS {
            dbits[k] = alloc_bool::<F>(b, (((t as u8) >> k) & 1) == 1);
        }
        let mut lc_diff = vec![(F::ONE, diff_i)];
        let mut pow = F::ONE;
        for k in 0..LIMB_BITS {
            lc_diff.push((-pow, dbits[k]));
            pow *= F::from(2u64);
        }
        b.enforce_lc_times_one_eq_const(lc_diff);

        let bor_next = alloc_bool::<F>(b, bor_next_u8 == 1);
        // u_i - p_i - bor_i + base*bor_{i+1} - diff_i == 0
        b.enforce_lc_times_one_eq_const(vec![
            (F::ONE, u_limbs[i]),
            (-F::from(p_digits[i] as u64), b.one()),
            (-F::ONE, bor),
            (F::from(LIMB_BASE_U64), bor_next),
            (-F::ONE, diff_i),
        ]);
        bor = bor_next;
    }
    // is_ge = 1 - final_borrow
    let is_ge = b.new_var(F::ONE - b.assignment[bor]);
    b.enforce_lc_times_one_eq_const(vec![(F::ONE, is_ge), (F::ONE, bor), (-F::ONE, b.one())]);
    b.enforce_lc_times_one_eq_const(vec![(F::ONE, q), (-F::ONE, is_ge)]);

    // Now compute z limbs as witness and enforce u = z + q*p + base*borrow chain (same as subtraction).
    let z_u64 = if q_u8 == 1 { u - FROG_P } else { u };
    let mut z_limbs = [0usize; LIMBS_U64];
    for i in 0..LIMBS_U64 {
        let zi = ((z_u64 >> (LIMB_BITS * i)) & (LIMB_BASE_U64 - 1)) as u64;
        z_limbs[i] = b.new_var(F::from(zi));
        // Range-check z limb to 7 bits.
        let mut zbits = [0usize; LIMB_BITS];
        for k in 0..LIMB_BITS {
            zbits[k] = alloc_bool::<F>(b, (((zi as u8) >> k) & 1) == 1);
        }
        let mut lc_z = vec![(F::ONE, z_limbs[i])];
        let mut pow = F::ONE;
        for k in 0..LIMB_BITS {
            lc_z.push((-pow, zbits[k]));
            pow *= F::from(2u64);
        }
        b.enforce_lc_times_one_eq_const(lc_z);
    }

    // Base-128 borrow constraints for u - q*p - z = 0, with final borrow=0.
    let mut borrow2 = b.new_var(F::ZERO);
    b.enforce_var_eq_const(borrow2, F::ZERO);
    for i in 0..LIMBS_U64 {
        let ui = ((u >> (LIMB_BITS * i)) & (LIMB_BASE_U64 - 1)) as i16;
        let zi = ((z_u64 >> (LIMB_BITS * i)) & (LIMB_BASE_U64 - 1)) as i16;
        let pi = p_digits[i] as i16;
        let bi = if i == 0 {
            0i16
        } else {
            b.assignment[borrow2]
                .into_bigint()
                .to_bytes_le()
                .get(0)
                .copied()
                .unwrap_or(0) as i16
        };
        let rhs = ui - (q_u8 as i16) * pi - bi - zi;
        let bor_next_u8 = if rhs < 0 { 1u8 } else { 0u8 };
        let bor_next = if i == LIMBS_U64 - 1 {
            let v = b.new_var(F::ZERO);
            b.enforce_var_eq_const(v, F::ZERO);
            v
        } else {
            alloc_bool::<F>(b, bor_next_u8 == 1)
        };
        b.enforce_lc_times_one_eq_const(vec![
            (F::ONE, u_limbs[i]),
            (-F::from(p_digits[i] as u64), q),
            (-F::ONE, borrow2),
            (F::from(LIMB_BASE_U64), bor_next),
            (-F::ONE, z_limbs[i]),
        ]);
        borrow2 = bor_next;
    }
    b.enforce_var_eq_const(borrow2, F::ZERO);

    (q, z_limbs)
}

#[derive(Clone, Debug)]
pub struct FrogRejectionCoinWiring {
    /// Global variable indices of all base-257 digit vars used (tries * 8).
    pub digit_vars: Vec<usize>,
    /// Global variable index of `found` (boolean), enforced to be 1 in the builder.
    pub found_bit: usize,
    /// Global variable indices of the selected coin value `u` as base-128 limbs (little-endian).
    pub coin_limbs: Vec<usize>,
    /// The fixed number of tries assumed by this wiring.
    pub tries: usize,
}

/// Wiring for a bounded scalar challenge (u32) derived from a `get_challenge()` digit block.
#[derive(Clone, Debug)]
pub struct BoundedU32ChallengeWiring {
    /// The 8 digit vars (F257) consumed for this challenge (schedule-only).
    pub digit_vars: Vec<usize>,
    /// The 4 byte-view vars (256 -> 0) packed into the u32.
    pub byte_vars: [usize; 4],
    /// Base-128 limbs (little-endian) representing the u32 value.
    pub limbs: [usize; LIMBS_U32],
    /// Balanced base-16 digits (little-endian), length 9, representing the same u32 value.
    pub bal16_digits: Vec<usize>,
    /// Balanced base-16 digits (little-endian), length 18, representing `u^2` as an integer.
    ///
    /// This is a common derived scalar in CM (e.g. `beta2 = beta*beta`) and avoids going back to
    /// in-field multiplication for u32-ish values.
    pub bal16_sq_digits: Vec<usize>,
}

/// Wiring for short challenges `short_challenge(128)` over a ring of dimension `ring_dim`.
#[derive(Clone, Debug)]
pub struct ShortChallengeWiring {
    /// The digit vars (F257), length = `ring_dim`.
    pub digit_vars: Vec<usize>,
    /// The byte-view vars (256 -> 0), length = `ring_dim`.
    pub byte_vars: Vec<usize>,
    /// The coefficient vars (in F257), length = `ring_dim`.
    pub coeff_vars: Vec<usize>,
    /// Balanced base-16 digits (little-endian) for each coefficient (3 digits per coeff).
    ///
    /// These are constrained to match `coeff_vars[i]` as an integer in the range [-128,127] via:
    ///   coeff = d0 + 16*d1   (and d2 is forced to 0),
    /// where d0,d1 ∈ [-8,7].
    pub coeff_bal16_digits: Vec<[usize; 3]>,
}

/// Helper: number of `short_challenge(128)` blocks used by CM (the `s` and `s_prime` surface).
///
/// Matches `cm.rs`: `s` has 3 blocks, `s_prime` has `k * ring_dim` blocks.
pub fn cm_short_challenge_blocks(ring_dim: usize, k: usize) -> usize {
    3 + k * ring_dim
}

/// Helper: number of bounded scalar challenges (u32) consumed by CM after `absorb_comh`.
///
/// Matches the historical WE arithmetization shape:
/// - `c0`: `log_kappa` challenges
/// - `c1`: `log_kappa` challenges
/// - `rc0`: 1 challenge
/// - `sumcheck_r0`: `nvars_cm` challenges
/// - `rc1`: 1 challenge
/// - `sumcheck_r1`: `nvars_cm` challenges
pub fn cm_bounded_u32_challenges(log_kappa: usize, nvars_cm: usize) -> usize {
    2 * log_kappa + 2 + 2 * nvars_cm
}

/// Explicit wiring of which Poseidon `SqueezeField` ops (by **squeeze-field op index**) are used
/// for each challenge type.
///
/// Indices refer to the order of `PoseidonTraceOp::SqueezeField(..)` occurrences in `ops`.
#[derive(Clone, Debug, Default)]
pub struct TinyCoinOpWiring {
    /// `SqueezeField(len=ring_dim)` op indices for `short_challenge(128)` blocks.
    pub short_squeeze_ops: Vec<usize>,
    /// `SqueezeField(len=8)` op indices for bounded u32 scalar challenges.
    pub u32_squeeze_ops: Vec<usize>,
    /// `SqueezeField(len=8)` op indices for Frog rejection candidates (length must be `n_coins*tries`).
    pub frog_squeeze_ops: Vec<usize>,
}

/// Wiring for a simple “first CM digit-mul surface” stage:
/// multiply one short-challenge block's coefficients by one bounded-u32 challenge using the digit backend.
#[derive(Clone, Debug)]
pub struct CmDigitMulSurfaceWiring {
    pub short_block_idx: usize,
    pub u32_idx: usize,
    /// Per coefficient product digits (len 12 each), in ring coefficient order.
    pub products: Vec<[usize; 12]>,
    /// Same products as `products`, but with the tail carry normalized so all digits are in `[-8,7]`
    /// plus one final carry digit in `[-2,2]`.
    ///
    /// This is the preferred representation for downstream additions/accumulations.
    pub products13: Vec<[usize; 13]>,
    /// Optional accumulated sum of all coefficient products as balanced base-16 digits (little-endian).
    ///
    /// This is the first “real consumption” step beyond per-coeff products: it forces the circuit
    /// to actually add up the products (still purely at the digit level).
    pub sum_digits: Vec<usize>,

    /// Accumulated sum across **all requested digit-mul surfaces** in the batch builder.
    ///
    /// This is a convenience wiring to support “scale then add then add ...” consumption patterns.
    /// All surfaces returned by a single builder call will carry the same `sum_all_pairs_digits`.
    pub sum_all_pairs_digits: Vec<usize>,

    /// Coefficient-wise sum across **all requested digit-mul surfaces**.
    ///
    /// Length = `ring_dim`; each entry is a balanced base-16 digit vector (little-endian) of length 16.
    /// This matches the CM pattern “scale (per coefficient) then add”, and is the next consumption hook
    /// we’ll use to build real CM accumulations.
    pub sum_all_pairs_coeffwise: Vec<Vec<usize>>,
}

/// Like `CmDigitMulSurfaceWiring`, but multiplies a short-challenge block by **u32^2** (18 digits).
#[derive(Clone, Debug)]
pub struct CmDigitMulSqSurfaceWiring {
    pub short_block_idx: usize,
    pub u32_idx: usize,
    /// Per coefficient product digits (len 21 each), in ring coefficient order.
    pub products21: Vec<[usize; 21]>,
    /// Same products as `products21`, but normalized to 22 digits (tail carry split to `[-8,7]` + `[-2,2]`).
    pub products22: Vec<[usize; 22]>,
    /// Sum of all coefficient products (balanced base-16 digits, little-endian), fixed length 24.
    pub sum_digits: Vec<usize>,
    /// Sum across all requested sq-surfaces in the batch, fixed length 24.
    pub sum_all_pairs_digits: Vec<usize>,
    /// Coefficient-wise sum across all requested sq-surfaces, length ring_dim, each fixed length 24.
    pub sum_all_pairs_coeffwise: Vec<Vec<usize>>,
}

#[inline]
fn rebalance_prod12_to_prod13(
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
fn rebalance_prod21_to_prod22(
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

fn sum_product_digits_bal16(
    b: &mut Dr1csBuilder<F257>,
    products13: &[[usize; 13]],
    target_len: usize,
) -> Vec<usize> {
    assert!(target_len >= 13);
    let zero = alloc_bal16_digit(b, 0);

    // Start at zero.
    let mut acc = vec![zero; target_len];

    for p13 in products13 {
        let padded = shift_pad_bal16(p13, 0, target_len, zero);
        let (new_acc, carry) = add_bal16_same_len(b, &acc, &padded);
        acc = new_acc;
        // For the envelope sizes we use (e.g. ring_dim<=64, coeff in [-128,127]), a modest target_len
        // (like 16) is enough so this final carry should be 0.
        b.enforce_var_eq_const(carry, F257::ZERO);
    }
    acc
}

fn sum_product_digits_bal16_22(
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

fn sum_bal16_vectors_fixed_len(
    b: &mut Dr1csBuilder<F257>,
    vecs: &[Vec<usize>],
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

fn sum_products13_coeffwise_fixed_len(
    b: &mut Dr1csBuilder<F257>,
    per_surface_products13: &[Vec<[usize; 13]>],
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

fn sum_products22_coeffwise_fixed_len(
    b: &mut Dr1csBuilder<F257>,
    per_surface_products22: &[Vec<[usize; 22]>],
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

/// Infer the CM coin op wiring from a Poseidon op schedule.
///
/// This follows the CM transcript schedule in `cm.rs`:
/// - first consume `short_need` occurrences of `SqueezeField(len=ring_dim)` for `short_challenge(128)`
/// - then consume `u32_need` occurrences of `SqueezeField(len=8)` for bounded scalar challenges
/// - optionally, treat the next `frog_need` occurrences of `SqueezeField(len=8)` as Frog coin candidates
///
/// IMPORTANT:
/// - This is a **schedule** parser. It does not validate re-absorbs; Poseidon arithmetization handles that.
/// - If your full transcript has other `SqueezeField(len=8)` before CM scalar challenges, pass an
///   `squeeze_field_op_offset` to start counting from the CM segment.
pub fn infer_cm_coin_op_wiring_from_ops(
    ops: &[PoseidonTraceOp<F257>],
    ring_dim: usize,
    k: usize,
    log_kappa: usize,
    nvars_cm: usize,
    squeeze_field_op_offset: usize,
    frog_need: usize,
) -> Result<TinyCoinOpWiring, String> {
    let short_need = cm_short_challenge_blocks(ring_dim, k);
    let u32_need = cm_bounded_u32_challenges(log_kappa, nvars_cm);

    let mut out = TinyCoinOpWiring::default();
    let mut squeeze_field_op_idx = 0usize;

    // First pass: count squeeze-field ops, and select indices in the desired order.
    for op in ops {
        if let PoseidonTraceOp::SqueezeField(v) = op {
            if squeeze_field_op_idx >= squeeze_field_op_offset {
                if v.len() == ring_dim && out.short_squeeze_ops.len() < short_need {
                    out.short_squeeze_ops
                        .push(squeeze_field_op_idx - squeeze_field_op_offset);
                } else if v.len() == DIGITS_PER_TRY && out.u32_squeeze_ops.len() < u32_need {
                    out.u32_squeeze_ops
                        .push(squeeze_field_op_idx - squeeze_field_op_offset);
                } else if v.len() == DIGITS_PER_TRY && out.frog_squeeze_ops.len() < frog_need {
                    out.frog_squeeze_ops
                        .push(squeeze_field_op_idx - squeeze_field_op_offset);
                }
            }
            squeeze_field_op_idx += 1;
        }
        if out.short_squeeze_ops.len() == short_need
            && out.u32_squeeze_ops.len() == u32_need
            && out.frog_squeeze_ops.len() == frog_need
        {
            break;
        }
    }

    if out.short_squeeze_ops.len() != short_need {
        return Err(format!(
            "infer_cm_coin_op_wiring: need {} short squeezes (len={}), got {}",
            short_need,
            ring_dim,
            out.short_squeeze_ops.len()
        ));
    }
    if out.u32_squeeze_ops.len() != u32_need {
        return Err(format!(
            "infer_cm_coin_op_wiring: need {} u32 squeezes (len=8), got {}",
            u32_need,
            out.u32_squeeze_ops.len()
        ));
    }
    if out.frog_squeeze_ops.len() != frog_need {
        return Err(format!(
            "infer_cm_coin_op_wiring: need {} frog squeezes (len=8), got {}",
            frog_need,
            out.frog_squeeze_ops.len()
        ));
    }
    Ok(out)
}

fn squeeze_field_ranges_by_op_index(
    squeeze_field_ranges: &[(usize, usize)],
    op_indices: &[usize],
) -> Result<Vec<(usize, usize)>, String> {
    let mut out = Vec::with_capacity(op_indices.len());
    for &op_idx in op_indices {
        let r = squeeze_field_ranges
            .get(op_idx)
            .copied()
            .ok_or_else(|| format!("squeeze_field op idx {} out of range", op_idx))?;
        out.push(r);
    }
    Ok(out)
}

/// Convert an LF+ recorded transcript trace (stored over an arbitrary base prime field `BF`) into
/// an equivalent op list over `F257`, by interpreting each element as a small integer.
///
/// This is useful when you have a `crate::recording_transcript::TracePoseidonTranscript<R>` trace,
/// but want to arithmetize the transcript layer over the tiny field.
///
/// Requirements:
/// - each traced element must be an integer in `[0,256]` (as used by the F257 byte/digit surface)
pub fn lift_recording_trace_ops_to_f257<BF: PrimeField>(
    ops: &[crate::recording_transcript::PoseidonTraceOp<BF>],
) -> Result<Vec<PoseidonTraceOp<F257>>, String> {
    fn bf_to_u16<BF: PrimeField>(x: &BF) -> u16 {
        let bytes = x.into_bigint().to_bytes_le();
        (bytes.get(0).copied().unwrap_or(0) as u16)
            | ((bytes.get(1).copied().unwrap_or(0) as u16) << 8)
    }
    let mut out: Vec<PoseidonTraceOp<F257>> = Vec::with_capacity(ops.len());
    for op in ops {
        match op {
            crate::recording_transcript::PoseidonTraceOp::Absorb(v) => {
                let mut vv = Vec::with_capacity(v.len());
                for e in v {
                    let d = bf_to_u16::<BF>(e);
                    if d > 256 {
                        return Err(format!("trace element out of range: {d}"));
                    }
                    vv.push(F257::from(d as u64));
                }
                out.push(PoseidonTraceOp::Absorb(vv));
            }
            crate::recording_transcript::PoseidonTraceOp::SqueezeField(v) => {
                let mut vv = Vec::with_capacity(v.len());
                for e in v {
                    let d = bf_to_u16::<BF>(e);
                    if d > 256 {
                        return Err(format!("trace element out of range: {d}"));
                    }
                    vv.push(F257::from(d as u64));
                }
                out.push(PoseidonTraceOp::SqueezeField(vv));
            }
            crate::recording_transcript::PoseidonTraceOp::SqueezeBytes { n, out: bytes } => {
                // Legacy traces may include this op; map it directly.
                out.push(PoseidonTraceOp::SqueezeBytes { n: *n, out: bytes.clone() });
            }
        }
    }
    Ok(out)
}

/// Poseidon(F257) + fixed-tries rejection sampler for one Frog scalar coin.
///
/// This is the *circuit-side* version of Symphony’s `get_challenge` rejection logic, but with a
/// fixed number of tries so the op schedule is deterministic / arithmetizable.
pub fn build_poseidon_f257_with_frog_coin_rejection_glue_from_ops_with_wiring(
    cfg: Option<&PoseidonConfig<F257>>,
    ops: &[PoseidonTraceOp<F257>],
    tries: usize,
) -> Result<(SparseDr1csInstance<F257>, Vec<F257>, FrogRejectionCoinWiring), String> {
    let (pose_inst, pose_asg, wiring, _byte_wiring) =
        build_poseidon_f257_from_ops_with_wiring_and_bytes(cfg, ops)?;

    // Collect the first `tries` SqueezeField ranges of length 8.
    let mut digit_vars_global: Vec<usize> = Vec::with_capacity(tries * DIGITS_PER_TRY);
    for (start, len) in wiring.squeeze_field_ranges.iter().copied() {
        if len != DIGITS_PER_TRY {
            continue;
        }
        for v in &wiring.squeeze_field_vars[start..start + len] {
            digit_vars_global.push(*v);
        }
        if digit_vars_global.len() == tries * DIGITS_PER_TRY {
            break;
        }
    }
    if digit_vars_global.len() != tries * DIGITS_PER_TRY {
        return Err(format!(
            "need {tries} SqueezeField(len=8) ops (got {} digits)",
            digit_vars_global.len()
        ));
    }

    let mut gb = Dr1csBuilder::<F257>::new();
    gb.enforce_var_eq_const(gb.one(), F257::ONE);
    let mut digit_vars_local: Vec<usize> = Vec::with_capacity(digit_vars_global.len());
    for &gv in &digit_vars_global {
        digit_vars_local.push(gb.new_var(pose_asg[gv]));
    }
    let (coin_local, found_local) =
        sample_frog_coin_unrolled_rejection_8_digits::<F257>(&mut gb, &digit_vars_local, tries);
    gb.enforce_var_eq_const(found_local, F257::ONE);

    let (glue_inst, glue_asg) = gb.into_instance();
    let glue_nvars = glue_inst.nvars;

    let (mut inst, asg) = merge_sparse_dr1cs_share_one::<F257>(vec![
        (pose_inst, pose_asg),
        (glue_inst, glue_asg),
    ])
    .map_err(|e| format!("merge poseidon+rejection_glue failed: {e}"))?;

    // Glue digit vars: pose digit == local copied digit.
    let pose_nvars = inst.nvars - (glue_nvars - 1);
    let glue_offset = pose_nvars - 1;
    for i in 0..digit_vars_global.len() {
        let pose_global = digit_vars_global[i];
        let glue_local = digit_vars_local[i];
        let glue_global = if glue_local == 0 { 0 } else { glue_local + glue_offset };
        enforce_var_eq::<F257>(&mut inst, pose_global, glue_global);
    }

    debug_assert_eq!(inst.nvars, asg.len());
    let to_glue_global = |glue_local: usize| -> usize {
        if glue_local == 0 { 0 } else { glue_local + glue_offset }
    };
    let wiring_out = FrogRejectionCoinWiring {
        digit_vars: digit_vars_global,
        found_bit: to_glue_global(found_local),
        coin_limbs: coin_local.iter().copied().map(to_glue_global).collect(),
        tries,
    };
    Ok((inst, asg, wiring_out))
}

/// Poseidon(F257) + fixed-tries rejection sampler for `n_coins` Frog scalar coins.
///
/// Each coin consumes `tries` occurrences of `SqueezeField(len=8)` (i.e. `tries*8` digits) and
/// selects the first candidate `< p_frog`. Enforces `found=1` per coin.
pub fn build_poseidon_f257_with_frog_rejection_coins_from_ops_with_wiring(
    cfg: Option<&PoseidonConfig<F257>>,
    ops: &[PoseidonTraceOp<F257>],
    n_coins: usize,
    tries: usize,
) -> Result<(SparseDr1csInstance<F257>, Vec<F257>, Vec<FrogRejectionCoinWiring>), String> {
    if n_coins == 0 {
        return Err("n_coins must be > 0".to_string());
    }
    let (pose_inst, pose_asg, wiring, _byte_wiring) =
        build_poseidon_f257_from_ops_with_wiring_and_bytes(cfg, ops)?;

    // Collect SqueezeField(len=8) ranges.
    let mut ranges: Vec<(usize, usize)> = wiring
        .squeeze_field_ranges
        .iter()
        .copied()
        .filter(|(_s, l)| *l == DIGITS_PER_TRY)
        .collect();
    if ranges.len() < n_coins * tries {
        return Err(format!(
            "need {} SqueezeField(len=8) ops (got {})",
            n_coins * tries,
            ranges.len()
        ));
    }
    ranges.truncate(n_coins * tries);

    // Build one glue subsystem containing all coins.
    let mut gb = Dr1csBuilder::<F257>::new();
    gb.enforce_var_eq_const(gb.one(), F257::ONE);

    // Local copies of all digit vars in order.
    let mut digit_vars_local: Vec<usize> = Vec::with_capacity(n_coins * tries * DIGITS_PER_TRY);
    let mut digit_vars_global: Vec<usize> = Vec::with_capacity(n_coins * tries * DIGITS_PER_TRY);
    for (start, len) in &ranges {
        for v in &wiring.squeeze_field_vars[*start..*start + *len] {
            digit_vars_global.push(*v);
            digit_vars_local.push(gb.new_var(pose_asg[*v]));
        }
    }
    debug_assert_eq!(digit_vars_global.len(), digit_vars_local.len());

    // Run per-coin sampler over contiguous chunks.
    let mut wirings: Vec<FrogRejectionCoinWiring> = Vec::with_capacity(n_coins);
    let mut coin_limbs_local_all: Vec<[usize; LIMBS_U64]> = Vec::with_capacity(n_coins);
    let mut found_local_all: Vec<usize> = Vec::with_capacity(n_coins);
    for coin_idx in 0..n_coins {
        let off = coin_idx * tries * DIGITS_PER_TRY;
        let digits = &digit_vars_local[off..off + tries * DIGITS_PER_TRY];
        let (coin_local, found_local) =
            sample_frog_coin_unrolled_rejection_8_digits::<F257>(&mut gb, digits, tries);
        gb.enforce_var_eq_const(found_local, F257::ONE);
        coin_limbs_local_all.push(coin_local);
        found_local_all.push(found_local);
    }

    let (glue_inst, glue_asg) = gb.into_instance();
    let glue_nvars = glue_inst.nvars;

    // Merge poseidon + glue.
    let (mut inst, asg) = merge_sparse_dr1cs_share_one::<F257>(vec![
        (pose_inst, pose_asg),
        (glue_inst, glue_asg),
    ])
    .map_err(|e| format!("merge poseidon+rejection_glue failed: {e}"))?;

    // Glue digit vars: pose digit == local copied digit.
    let pose_nvars = inst.nvars - (glue_nvars - 1);
    let glue_offset = pose_nvars - 1;
    for i in 0..digit_vars_global.len() {
        let pose_global = digit_vars_global[i];
        let glue_local = digit_vars_local[i];
        let glue_global = if glue_local == 0 { 0 } else { glue_local + glue_offset };
        enforce_var_eq::<F257>(&mut inst, pose_global, glue_global);
    }

    let to_glue_global = |glue_local: usize| -> usize {
        if glue_local == 0 { 0 } else { glue_local + glue_offset }
    };
    for coin_idx in 0..n_coins {
        let off = coin_idx * tries * DIGITS_PER_TRY;
        let digit_vars = digit_vars_global[off..off + tries * DIGITS_PER_TRY].to_vec();
        let found_bit = to_glue_global(found_local_all[coin_idx]);
        let coin_limbs = coin_limbs_local_all[coin_idx]
            .iter()
            .copied()
            .map(to_glue_global)
            .collect::<Vec<_>>();
        wirings.push(FrogRejectionCoinWiring {
            digit_vars,
            found_bit,
            coin_limbs,
            tries,
        });
    }

    Ok((inst, asg, wirings))
}

/// Build Poseidon(F257) arithmetization + glue that derives:
/// - `n_short` short challenges (each from `SqueezeField(len=ring_dim)`; no reabsorb)
/// - `n_u32` bounded scalar challenges (each from `SqueezeField(len=8)`; reabsorb handled by Poseidon ops)
///
/// This is a **coin surface** builder only: it does not implement CM arithmetic yet.
pub fn build_poseidon_f257_with_cm_coin_surface_from_ops_with_wiring(
    cfg: Option<&PoseidonConfig<F257>>,
    ops: &[PoseidonTraceOp<F257>],
    ring_dim: usize,
    n_short: usize,
    n_u32: usize,
) -> Result<
    (
        SparseDr1csInstance<F257>,
        Vec<F257>,
        Vec<ShortChallengeWiring>,
        Vec<BoundedU32ChallengeWiring>,
    ),
    String,
> {
    let (pose_inst, pose_asg, pose_wiring, _byte_wiring) =
        build_poseidon_f257_from_ops_with_wiring_and_bytes(cfg, ops)?;

    // Identify SqueezeField blocks by length, preserving **appearance order** (short and u32 can interleave).
    #[derive(Clone, Copy)]
    enum BlockKind {
        Short,
        ChalU32,
    }
    let mut blocks: Vec<(BlockKind, usize /*start*/, usize /*len*/)> =
        Vec::with_capacity(n_short + n_u32);
    let mut short_seen = 0usize;
    let mut chal_seen = 0usize;
    for &(start, len) in &pose_wiring.squeeze_field_ranges {
        if len == ring_dim && short_seen < n_short {
            blocks.push((BlockKind::Short, start, len));
            short_seen += 1;
        } else if len == DIGITS_PER_TRY && chal_seen < n_u32 {
            blocks.push((BlockKind::ChalU32, start, len));
            chal_seen += 1;
        }
        if short_seen == n_short && chal_seen == n_u32 {
            break;
        }
    }
    if short_seen != n_short {
        return Err(format!(
            "need {} short SqueezeField(len={}) blocks (got {})",
            n_short, ring_dim, short_seen
        ));
    }
    if chal_seen != n_u32 {
        return Err(format!(
            "need {} challenge SqueezeField(len=8) blocks (got {})",
            n_u32, chal_seen
        ));
    }

    // Build glue circuit.
    let mut gb = Dr1csBuilder::<F257>::new();
    gb.enforce_var_eq_const(gb.one(), F257::ONE);

    // Copy all used digit vars (pose -> local) in deterministic order for gluing.
    let mut local_digits: Vec<usize> = Vec::new();
    let mut global_digits: Vec<usize> = Vec::new();
    for &(_kind, start, len) in &blocks {
        for v in &pose_wiring.squeeze_field_vars[start..start + len] {
            global_digits.push(*v);
            local_digits.push(gb.new_var(pose_asg[*v]));
        }
    }

    // Produce short + u32 challenges in block order (but return separated vectors).
    let mut shorts: Vec<ShortChallengeWiring> = Vec::with_capacity(n_short);
    let mut u32s: Vec<BoundedU32ChallengeWiring> = Vec::with_capacity(n_u32);
    let mut cursor = 0usize;
    for &(kind, _start, len) in &blocks {
        match kind {
            BlockKind::Short => {
                debug_assert_eq!(len, ring_dim);
                let dvars = local_digits[cursor..cursor + ring_dim].to_vec();
                cursor += ring_dim;
                let (bvars, cvars, cdigits) = short_challenge_from_digits_128(&mut gb, &dvars, ring_dim);
                shorts.push(ShortChallengeWiring {
                    digit_vars: dvars,
                    byte_vars: bvars,
                    coeff_vars: cvars,
                    coeff_bal16_digits: cdigits,
                });
            }
            BlockKind::ChalU32 => {
                debug_assert_eq!(len, DIGITS_PER_TRY);
                let mut d = [0usize; DIGITS_PER_TRY];
                for i in 0..DIGITS_PER_TRY {
                    d[i] = local_digits[cursor + i];
                }
                cursor += DIGITS_PER_TRY;
                let (limbs, bytes, bal16_digits, bal16_sq_digits) =
                    bounded_u32_from_8_digits_base128(&mut gb, &d);
                u32s.push(BoundedU32ChallengeWiring {
                    digit_vars: d.to_vec(),
                    byte_vars: bytes,
                    limbs,
                    bal16_digits,
                    bal16_sq_digits,
                });
            }
        }
    }
    debug_assert_eq!(cursor, local_digits.len());

    let (glue_inst, glue_asg) = gb.into_instance();
    let glue_nvars = glue_inst.nvars;

    // Merge poseidon + glue.
    let (mut inst, asg) = merge_sparse_dr1cs_share_one::<F257>(vec![
        (pose_inst, pose_asg),
        (glue_inst, glue_asg),
    ])
    .map_err(|e| format!("merge poseidon+cm_coin_glue failed: {e}"))?;

    // Add explicit equality constraints between pose digit vars and their local copies.
    let pose_nvars = inst.nvars - (glue_nvars - 1);
    let glue_offset = pose_nvars - 1;
    for i in 0..global_digits.len() {
        let pose_global = global_digits[i];
        let glue_local = local_digits[i];
        let glue_global = if glue_local == 0 { 0 } else { glue_local + glue_offset };
        enforce_var_eq::<F257>(&mut inst, pose_global, glue_global);
    }

    // Remap wiring vars to global indices.
    let to_glue_global = |glue_local: usize| -> usize {
        if glue_local == 0 { 0 } else { glue_local + glue_offset }
    };
    let shorts_global = shorts
        .into_iter()
        .map(|w| ShortChallengeWiring {
            digit_vars: w.digit_vars.into_iter().map(to_glue_global).collect(),
            byte_vars: w.byte_vars.into_iter().map(to_glue_global).collect(),
            coeff_vars: w.coeff_vars.into_iter().map(to_glue_global).collect(),
            coeff_bal16_digits: w
                .coeff_bal16_digits
                .into_iter()
                .map(|a| a.map(to_glue_global))
                .collect(),
        })
        .collect::<Vec<_>>();
    let u32s_global = u32s
        .into_iter()
        .map(|w| BoundedU32ChallengeWiring {
            digit_vars: w.digit_vars.into_iter().map(to_glue_global).collect(),
            byte_vars: w.byte_vars.map(to_glue_global),
            limbs: w.limbs.map(to_glue_global),
            bal16_digits: w.bal16_digits.into_iter().map(to_glue_global).collect(),
            bal16_sq_digits: w.bal16_sq_digits.into_iter().map(to_glue_global).collect(),
        })
        .collect::<Vec<_>>();

    Ok((inst, asg, shorts_global, u32s_global))
}

/// Build Poseidon(F257) arithmetization + glue that derives all *CM-facing* coins in one shot:
/// - short challenges (ring coeff vectors)
/// - bounded u32 scalar challenges (base-128 limbs)
/// - Frog scalar coins via fixed-tries rejection (base-128 limbs)
///
/// The caller supplies an explicit `TinyCoinOpWiring` to disambiguate `SqueezeField(len=8)` uses.
pub fn build_poseidon_f257_with_cm_coins_and_frog_rejection_from_ops_with_wiring(
    cfg: Option<&PoseidonConfig<F257>>,
    ops: &[PoseidonTraceOp<F257>],
    ring_dim: usize,
    wiring: &TinyCoinOpWiring,
    n_coins: usize,
    tries: usize,
) -> Result<
    (
        SparseDr1csInstance<F257>,
        Vec<F257>,
        Vec<ShortChallengeWiring>,
        Vec<BoundedU32ChallengeWiring>,
        Vec<FrogRejectionCoinWiring>,
    ),
    String,
> {
    if n_coins == 0 {
        return Err("n_coins must be > 0".to_string());
    }
    if wiring.frog_squeeze_ops.len() != n_coins * tries {
        return Err(format!(
            "frog_squeeze_ops must have length n_coins*tries={} (got {})",
            n_coins * tries,
            wiring.frog_squeeze_ops.len()
        ));
    }

    let (pose_inst, pose_asg, pose_wiring, _byte_wiring) =
        build_poseidon_f257_from_ops_with_wiring_and_bytes(cfg, ops)?;

    // Resolve ranges from op indices.
    let short_ranges = squeeze_field_ranges_by_op_index(&pose_wiring.squeeze_field_ranges, &wiring.short_squeeze_ops)?;
    let u32_ranges = squeeze_field_ranges_by_op_index(&pose_wiring.squeeze_field_ranges, &wiring.u32_squeeze_ops)?;
    let frog_ranges = squeeze_field_ranges_by_op_index(&pose_wiring.squeeze_field_ranges, &wiring.frog_squeeze_ops)?;

    // Validate lengths.
    for &(_s, len) in &short_ranges {
        if len != ring_dim {
            return Err(format!("short squeeze has len {}, expected ring_dim={}", len, ring_dim));
        }
    }
    for &(_s, len) in u32_ranges.iter().chain(frog_ranges.iter()) {
        if len != DIGITS_PER_TRY {
            return Err(format!("len=8 squeeze has len {}, expected 8", len));
        }
    }

    // Build glue circuit.
    let mut gb = Dr1csBuilder::<F257>::new();
    gb.enforce_var_eq_const(gb.one(), F257::ONE);

    // Copy used digit vars (pose -> local) in deterministic order:
    // short blocks, then u32 blocks, then frog candidate blocks (preserving op order within each list).
    let mut local_digits: Vec<usize> = Vec::new();
    let mut global_digits: Vec<usize> = Vec::new();
    for (start, len) in short_ranges.iter().chain(u32_ranges.iter()).chain(frog_ranges.iter()) {
        for v in &pose_wiring.squeeze_field_vars[*start..*start + *len] {
            global_digits.push(*v);
            local_digits.push(gb.new_var(pose_asg[*v]));
        }
    }

    // Short challenges.
    let mut shorts: Vec<ShortChallengeWiring> = Vec::with_capacity(short_ranges.len());
    let mut cursor = 0usize;
    for _ in 0..short_ranges.len() {
        let dvars = local_digits[cursor..cursor + ring_dim].to_vec();
        cursor += ring_dim;
        let (bvars, cvars, cdigits) = short_challenge_from_digits_128(&mut gb, &dvars, ring_dim);
        shorts.push(ShortChallengeWiring {
            digit_vars: dvars,
            byte_vars: bvars,
            coeff_vars: cvars,
            coeff_bal16_digits: cdigits,
        });
    }

    // u32 challenges.
    let mut u32s: Vec<BoundedU32ChallengeWiring> = Vec::with_capacity(u32_ranges.len());
    for _ in 0..u32_ranges.len() {
        let mut d = [0usize; DIGITS_PER_TRY];
        for i in 0..DIGITS_PER_TRY {
            d[i] = local_digits[cursor + i];
        }
        cursor += DIGITS_PER_TRY;
        let (limbs, bytes, bal16_digits, bal16_sq_digits) = bounded_u32_from_8_digits_base128(&mut gb, &d);
        u32s.push(BoundedU32ChallengeWiring {
            digit_vars: d.to_vec(),
            byte_vars: bytes,
            limbs,
            bal16_digits,
            bal16_sq_digits,
        });
    }

    // Frog rejection coins.
    let mut frog_wirings: Vec<FrogRejectionCoinWiring> = Vec::with_capacity(n_coins);
    let mut coin_limbs_local_all: Vec<[usize; LIMBS_U64]> = Vec::with_capacity(n_coins);
    let mut found_local_all: Vec<usize> = Vec::with_capacity(n_coins);
    // Flatten digits for all candidates in op order.
    let frog_digits_local = &local_digits[cursor..cursor + (n_coins * tries * DIGITS_PER_TRY)];
    cursor += n_coins * tries * DIGITS_PER_TRY;
    debug_assert_eq!(cursor, local_digits.len());
    for coin_idx in 0..n_coins {
        let off = coin_idx * tries * DIGITS_PER_TRY;
        let digits = &frog_digits_local[off..off + tries * DIGITS_PER_TRY];
        let (coin_local, found_local) =
            sample_frog_coin_unrolled_rejection_8_digits::<F257>(&mut gb, digits, tries);
        gb.enforce_var_eq_const(found_local, F257::ONE);
        coin_limbs_local_all.push(coin_local);
        found_local_all.push(found_local);
    }

    let (glue_inst, glue_asg) = gb.into_instance();
    let glue_nvars = glue_inst.nvars;

    // Merge poseidon + glue.
    let (mut inst, asg) = merge_sparse_dr1cs_share_one::<F257>(vec![
        (pose_inst, pose_asg),
        (glue_inst, glue_asg),
    ])
    .map_err(|e| format!("merge poseidon+cm+frog_glue failed: {e}"))?;

    // Add equality constraints between pose digit vars and their local copies.
    let pose_nvars = inst.nvars - (glue_nvars - 1);
    let glue_offset = pose_nvars - 1;
    for i in 0..global_digits.len() {
        let pose_global = global_digits[i];
        let glue_local = local_digits[i];
        let glue_global = if glue_local == 0 { 0 } else { glue_local + glue_offset };
        enforce_var_eq::<F257>(&mut inst, pose_global, glue_global);
    }

    let to_glue_global = |glue_local: usize| -> usize {
        if glue_local == 0 { 0 } else { glue_local + glue_offset }
    };

    let shorts_global = shorts
        .into_iter()
        .map(|w| ShortChallengeWiring {
            digit_vars: w.digit_vars.into_iter().map(to_glue_global).collect(),
            byte_vars: w.byte_vars.into_iter().map(to_glue_global).collect(),
            coeff_vars: w.coeff_vars.into_iter().map(to_glue_global).collect(),
            coeff_bal16_digits: w
                .coeff_bal16_digits
                .into_iter()
                .map(|a| a.map(to_glue_global))
                .collect(),
        })
        .collect::<Vec<_>>();
    let u32s_global = u32s
        .into_iter()
        .map(|w| BoundedU32ChallengeWiring {
            digit_vars: w.digit_vars.into_iter().map(to_glue_global).collect(),
            byte_vars: w.byte_vars.map(to_glue_global),
            limbs: w.limbs.map(to_glue_global),
            bal16_digits: w.bal16_digits.into_iter().map(to_glue_global).collect(),
            bal16_sq_digits: w.bal16_sq_digits.into_iter().map(to_glue_global).collect(),
        })
        .collect::<Vec<_>>();

    // Frog wiring: map back to global indices. Each coin has `tries` squeeze ops => `tries*8` digit vars.
    // We output digit vars in the same order as `wiring.frog_squeeze_ops`.
    let frog_digit_vars_global: Vec<usize> = {
        let mut out = Vec::with_capacity(n_coins * tries * DIGITS_PER_TRY);
        for (start, _len) in &frog_ranges {
            out.extend_from_slice(&pose_wiring.squeeze_field_vars[*start..*start + DIGITS_PER_TRY]);
        }
        out
    };
    for coin_idx in 0..n_coins {
        let off = coin_idx * tries * DIGITS_PER_TRY;
        let digit_vars = frog_digit_vars_global[off..off + tries * DIGITS_PER_TRY].to_vec();
        let found_bit = to_glue_global(found_local_all[coin_idx]);
        let coin_limbs = coin_limbs_local_all[coin_idx]
            .iter()
            .copied()
            .map(to_glue_global)
            .collect::<Vec<_>>();
        frog_wirings.push(FrogRejectionCoinWiring { digit_vars, found_bit, coin_limbs, tries });
    }

    Ok((inst, asg, shorts_global, u32s_global, frog_wirings))
}

/// Build Poseidon(F257) + CM coin surface + one digit-mul surface:
/// take `shorts[short_block_idx]` and multiply all its coeffs by `u32s[u32_idx]`.
pub fn build_poseidon_f257_with_cm_coins_frog_and_first_digit_mul_from_ops_with_wiring(
    cfg: Option<&PoseidonConfig<F257>>,
    ops: &[PoseidonTraceOp<F257>],
    ring_dim: usize,
    wiring: &TinyCoinOpWiring,
    n_coins: usize,
    tries: usize,
    short_block_idx: usize,
    u32_idx: usize,
) -> Result<
    (
        SparseDr1csInstance<F257>,
        Vec<F257>,
        Vec<ShortChallengeWiring>,
        Vec<BoundedU32ChallengeWiring>,
        Vec<FrogRejectionCoinWiring>,
        CmDigitMulSurfaceWiring,
    ),
    String,
> {
    let pairs = vec![(short_block_idx, u32_idx)];
    let (inst, asg, shorts, u32s, frogs, surfaces) =
        build_poseidon_f257_with_cm_coins_frog_and_digit_mul_surfaces_from_ops_with_wiring(
            cfg, ops, ring_dim, wiring, n_coins, tries, &pairs,
        )?;
    let surface = surfaces
        .into_iter()
        .next()
        .ok_or_else(|| "expected one surface".to_string())?;
    Ok((inst, asg, shorts, u32s, frogs, surface))
}

/// Build Poseidon(F257) + CM coin surface + Frog coins + a batch of **u32^2 digit-mul surfaces**.
///
/// Each pair `(short_block_idx, u32_idx)` requests multiplying all coeffs in that short block by
/// the squared u32 challenge `u32s[u32_idx]^2` (represented as 18 balanced base-16 digits).
pub fn build_poseidon_f257_with_cm_coins_frog_and_digit_mul_sq_surfaces_from_ops_with_wiring(
    cfg: Option<&PoseidonConfig<F257>>,
    ops: &[PoseidonTraceOp<F257>],
    ring_dim: usize,
    wiring: &TinyCoinOpWiring,
    n_coins: usize,
    tries: usize,
    pairs: &[(usize, usize)],
) -> Result<
    (
        SparseDr1csInstance<F257>,
        Vec<F257>,
        Vec<ShortChallengeWiring>,
        Vec<BoundedU32ChallengeWiring>,
        Vec<FrogRejectionCoinWiring>,
        Vec<CmDigitMulSqSurfaceWiring>,
    ),
    String,
> {
    let (pose_inst, pose_asg, pose_wiring, _byte_wiring) =
        build_poseidon_f257_from_ops_with_wiring_and_bytes(cfg, ops)?;

    let short_ranges =
        squeeze_field_ranges_by_op_index(&pose_wiring.squeeze_field_ranges, &wiring.short_squeeze_ops)?;
    let u32_ranges =
        squeeze_field_ranges_by_op_index(&pose_wiring.squeeze_field_ranges, &wiring.u32_squeeze_ops)?;
    let frog_ranges =
        squeeze_field_ranges_by_op_index(&pose_wiring.squeeze_field_ranges, &wiring.frog_squeeze_ops)?;

    if n_coins == 0 {
        return Err("n_coins must be > 0".to_string());
    }
    if wiring.frog_squeeze_ops.len() != n_coins * tries {
        return Err("frog_squeeze_ops length mismatch".to_string());
    }

    let mut gb = Dr1csBuilder::<F257>::new();
    gb.enforce_var_eq_const(gb.one(), F257::ONE);

    // Copy digit vars in a deterministic order (via a memo map).
    let mut local_map: std::collections::BTreeMap<usize, usize> = std::collections::BTreeMap::new();
    #[inline]
    fn copy_digit(
        gb: &mut Dr1csBuilder<F257>,
        pose_asg: &[F257],
        local_map: &mut std::collections::BTreeMap<usize, usize>,
        gv: usize,
    ) -> usize {
        if let Some(&lv) = local_map.get(&gv) {
            return lv;
        }
        let lv = gb.new_var(pose_asg[gv]);
        local_map.insert(gv, lv);
        lv
    }

    for &(si, ui) in pairs {
        if si >= short_ranges.len() {
            return Err(format!("short_block_idx {si} out of range"));
        }
        if ui >= u32_ranges.len() {
            return Err(format!("u32_idx {ui} out of range"));
        }
    }

    // Materialize required short blocks.
    let mut short_locals: std::collections::BTreeMap<usize, ShortChallengeWiring> = std::collections::BTreeMap::new();
    for &(si, _ui) in pairs {
        if short_locals.contains_key(&si) {
            continue;
        }
        let (s_start, s_len) = short_ranges[si];
        if s_len != ring_dim {
            return Err(format!("short block {si} length mismatch (got {s_len}, expected {ring_dim})"));
        }
        let short_digits_global = &pose_wiring.squeeze_field_vars[s_start..s_start + s_len];
        let short_digits_local: Vec<usize> = short_digits_global
            .iter()
            .copied()
            .map(|gv| copy_digit(&mut gb, &pose_asg, &mut local_map, gv))
            .collect();
        let (short_bvars, short_coeffs, short_coeff_digits) =
            short_challenge_from_digits_128(&mut gb, &short_digits_local, ring_dim);
        short_locals.insert(si, ShortChallengeWiring {
            digit_vars: short_digits_local,
            byte_vars: short_bvars,
            coeff_vars: short_coeffs,
            coeff_bal16_digits: short_coeff_digits,
        });
    }

    // Materialize required u32 blocks.
    let mut u32_locals: std::collections::BTreeMap<usize, BoundedU32ChallengeWiring> = std::collections::BTreeMap::new();
    for &(_si, ui) in pairs {
        if u32_locals.contains_key(&ui) {
            continue;
        }
        let (u_start, u_len) = u32_ranges[ui];
        if u_len != DIGITS_PER_TRY {
            return Err(format!("u32 block {ui} length mismatch (got {u_len}, expected {DIGITS_PER_TRY})"));
        }
        let u_digits_global = &pose_wiring.squeeze_field_vars[u_start..u_start + u_len];
        let mut u_digits_local = [0usize; DIGITS_PER_TRY];
        for i in 0..DIGITS_PER_TRY {
            u_digits_local[i] = copy_digit(&mut gb, &pose_asg, &mut local_map, u_digits_global[i]);
        }
        let (u_limbs, u_bytes, u_bal16, u_bal16_sq) =
            bounded_u32_from_8_digits_base128(&mut gb, &u_digits_local);
        u32_locals.insert(ui, BoundedU32ChallengeWiring {
            digit_vars: u_digits_local.to_vec(),
            byte_vars: u_bytes,
            limbs: u_limbs,
            bal16_digits: u_bal16,
            bal16_sq_digits: u_bal16_sq,
        });
    }

    // Build requested sq digit-mul surfaces.
    let mut surfaces_local: Vec<CmDigitMulSqSurfaceWiring> = Vec::with_capacity(pairs.len());
    for &(si, ui) in pairs {
        let s = short_locals.get(&si).expect("short local present");
        let u = u32_locals.get(&ui).expect("u32 local present");
        let products21 = scale_short_coeffs_by_digits18(&mut gb, &s.coeff_bal16_digits, &u.bal16_sq_digits);
        let products22 = products21
            .iter()
            .map(|p21| rebalance_prod21_to_prod22(&mut gb, p21))
            .collect::<Vec<_>>();
        let sum_digits = sum_product_digits_bal16_22(&mut gb, &products22, 24);
        surfaces_local.push(CmDigitMulSqSurfaceWiring {
            short_block_idx: si,
            u32_idx: ui,
            products21,
            products22,
            sum_digits,
            sum_all_pairs_digits: Vec::new(),
            sum_all_pairs_coeffwise: Vec::new(),
        });
    }

    let all_sum_digits = sum_bal16_vectors_fixed_len(
        &mut gb,
        &surfaces_local.iter().map(|s| s.sum_digits.clone()).collect::<Vec<_>>(),
        24,
    );
    let all_sum_coeffwise = sum_products22_coeffwise_fixed_len(
        &mut gb,
        &surfaces_local.iter().map(|s| s.products22.clone()).collect::<Vec<_>>(),
        ring_dim,
        24,
    );
    for s in &mut surfaces_local {
        s.sum_all_pairs_digits = all_sum_digits.clone();
        s.sum_all_pairs_coeffwise = all_sum_coeffwise.clone();
    }

    // Frog candidates.
    let mut frog_digit_vars_local: Vec<usize> = Vec::with_capacity(n_coins * tries * DIGITS_PER_TRY);
    let mut frog_digit_vars_global: Vec<usize> = Vec::with_capacity(n_coins * tries * DIGITS_PER_TRY);
    for &(start, len) in &frog_ranges {
        if len != DIGITS_PER_TRY {
            return Err("frog squeeze len mismatch".to_string());
        }
        for v in &pose_wiring.squeeze_field_vars[start..start + len] {
            frog_digit_vars_global.push(*v);
            frog_digit_vars_local.push(copy_digit(&mut gb, &pose_asg, &mut local_map, *v));
        }
    }
    let mut frog_wirings_local: Vec<FrogRejectionCoinWiring> = Vec::with_capacity(n_coins);
    let mut coin_limbs_local_all: Vec<[usize; LIMBS_U64]> = Vec::with_capacity(n_coins);
    let mut found_local_all: Vec<usize> = Vec::with_capacity(n_coins);
    for coin_idx in 0..n_coins {
        let off = coin_idx * tries * DIGITS_PER_TRY;
        let digits = &frog_digit_vars_local[off..off + tries * DIGITS_PER_TRY];
        let (coin_local, found_local) =
            sample_frog_coin_unrolled_rejection_8_digits::<F257>(&mut gb, digits, tries);
        gb.enforce_var_eq_const(found_local, F257::ONE);
        coin_limbs_local_all.push(coin_local);
        found_local_all.push(found_local);
    }

    let (glue_inst, glue_asg) = gb.into_instance();
    let glue_nvars = glue_inst.nvars;

    let (mut inst, asg) = merge_sparse_dr1cs_share_one::<F257>(vec![
        (pose_inst, pose_asg),
        (glue_inst, glue_asg),
    ])
    .map_err(|e| format!("merge poseidon+cm+frog+sqmul failed: {e}"))?;

    // Glue copied digit vars.
    let pose_nvars = inst.nvars - (glue_nvars - 1);
    let glue_offset = pose_nvars - 1;
    for (gv, lv) in local_map.iter() {
        let pose_global = *gv;
        let glue_local = *lv;
        let glue_global = if glue_local == 0 { 0 } else { glue_local + glue_offset };
        enforce_var_eq::<F257>(&mut inst, pose_global, glue_global);
    }

    let to_glue_global = |glue_local: usize| -> usize {
        if glue_local == 0 { 0 } else { glue_local + glue_offset }
    };

    let shorts_out = short_locals
        .into_iter()
        .map(|(_idx, w)| ShortChallengeWiring {
            digit_vars: w.digit_vars.into_iter().map(to_glue_global).collect(),
            byte_vars: w.byte_vars.into_iter().map(to_glue_global).collect(),
            coeff_vars: w.coeff_vars.into_iter().map(to_glue_global).collect(),
            coeff_bal16_digits: w
                .coeff_bal16_digits
                .into_iter()
                .map(|a| a.map(to_glue_global))
                .collect(),
        })
        .collect::<Vec<_>>();
    let u32s_out = u32_locals
        .into_iter()
        .map(|(_idx, w)| BoundedU32ChallengeWiring {
            digit_vars: w.digit_vars.into_iter().map(to_glue_global).collect(),
            byte_vars: w.byte_vars.map(to_glue_global),
            limbs: w.limbs.map(to_glue_global),
            bal16_digits: w.bal16_digits.into_iter().map(to_glue_global).collect(),
            bal16_sq_digits: w.bal16_sq_digits.into_iter().map(to_glue_global).collect(),
        })
        .collect::<Vec<_>>();

    for coin_idx in 0..n_coins {
        let off = coin_idx * tries * DIGITS_PER_TRY;
        let digit_vars = frog_digit_vars_global[off..off + tries * DIGITS_PER_TRY].to_vec();
        let found_bit = to_glue_global(found_local_all[coin_idx]);
        let coin_limbs = coin_limbs_local_all[coin_idx]
            .iter()
            .copied()
            .map(to_glue_global)
            .collect::<Vec<_>>();
        frog_wirings_local.push(FrogRejectionCoinWiring { digit_vars, found_bit, coin_limbs, tries });
    }

    let surfaces = surfaces_local
        .into_iter()
        .map(|s| CmDigitMulSqSurfaceWiring {
            short_block_idx: s.short_block_idx,
            u32_idx: s.u32_idx,
            products21: s.products21.into_iter().map(|arr| arr.map(to_glue_global)).collect(),
            products22: s.products22.into_iter().map(|arr| arr.map(to_glue_global)).collect(),
            sum_digits: s.sum_digits.into_iter().map(to_glue_global).collect(),
            sum_all_pairs_digits: s.sum_all_pairs_digits.into_iter().map(to_glue_global).collect(),
            sum_all_pairs_coeffwise: s
                .sum_all_pairs_coeffwise
                .into_iter()
                .map(|v| v.into_iter().map(to_glue_global).collect())
                .collect(),
        })
        .collect::<Vec<_>>();

    Ok((inst, asg, shorts_out, u32s_out, frog_wirings_local, surfaces))
}

/// Build Poseidon(F257) + CM coin surface + Frog coins + a batch of digit-mul surfaces.
///
/// Each pair `(short_block_idx, u32_idx)` requests multiplying all coeffs in that short block by
/// the bounded-u32 challenge at `u32_idx`.
pub fn build_poseidon_f257_with_cm_coins_frog_and_digit_mul_surfaces_from_ops_with_wiring(
    cfg: Option<&PoseidonConfig<F257>>,
    ops: &[PoseidonTraceOp<F257>],
    ring_dim: usize,
    wiring: &TinyCoinOpWiring,
    n_coins: usize,
    tries: usize,
    pairs: &[(usize, usize)],
) -> Result<
    (
        SparseDr1csInstance<F257>,
        Vec<F257>,
        Vec<ShortChallengeWiring>,
        Vec<BoundedU32ChallengeWiring>,
        Vec<FrogRejectionCoinWiring>,
        Vec<CmDigitMulSurfaceWiring>,
    ),
    String,
> {
    let (pose_inst, pose_asg, pose_wiring, _byte_wiring) =
        build_poseidon_f257_from_ops_with_wiring_and_bytes(cfg, ops)?;

    // Resolve ranges from op indices.
    let short_ranges =
        squeeze_field_ranges_by_op_index(&pose_wiring.squeeze_field_ranges, &wiring.short_squeeze_ops)?;
    let u32_ranges =
        squeeze_field_ranges_by_op_index(&pose_wiring.squeeze_field_ranges, &wiring.u32_squeeze_ops)?;
    let frog_ranges =
        squeeze_field_ranges_by_op_index(&pose_wiring.squeeze_field_ranges, &wiring.frog_squeeze_ops)?;

    if n_coins == 0 {
        return Err("n_coins must be > 0".to_string());
    }
    if wiring.frog_squeeze_ops.len() != n_coins * tries {
        return Err("frog_squeeze_ops length mismatch".to_string());
    }

    // Build glue circuit (coins + mul surface).
    let mut gb = Dr1csBuilder::<F257>::new();
    gb.enforce_var_eq_const(gb.one(), F257::ONE);

    // Copy digit vars for shorts, u32s, and frog candidates in appearance order.
    let mut local_map: std::collections::BTreeMap<usize, usize> = std::collections::BTreeMap::new();
    #[inline]
    fn copy_digit(
        gb: &mut Dr1csBuilder<F257>,
        pose_asg: &[F257],
        local_map: &mut std::collections::BTreeMap<usize, usize>,
        gv: usize,
    ) -> usize {
        if let Some(&lv) = local_map.get(&gv) {
            return lv;
        }
        let lv = gb.new_var(pose_asg[gv]);
        local_map.insert(gv, lv);
        lv
    }

    // Determine which short/u32 blocks we need.
    for &(si, ui) in pairs {
        if si >= short_ranges.len() {
            return Err(format!("short_block_idx {si} out of range"));
        }
        if ui >= u32_ranges.len() {
            return Err(format!("u32_idx {ui} out of range"));
        }
    }

    // Materialize required short blocks.
    let mut short_locals: std::collections::BTreeMap<usize, ShortChallengeWiring> = std::collections::BTreeMap::new();
    for &(si, _ui) in pairs {
        if short_locals.contains_key(&si) {
            continue;
        }
        let (s_start, s_len) = short_ranges[si];
        if s_len != ring_dim {
            return Err(format!("short block {si} length mismatch (got {s_len}, expected {ring_dim})"));
        }
        let short_digits_global = &pose_wiring.squeeze_field_vars[s_start..s_start + s_len];
        let short_digits_local: Vec<usize> = short_digits_global
            .iter()
            .copied()
            .map(|gv| copy_digit(&mut gb, &pose_asg, &mut local_map, gv))
            .collect();
        let (short_bvars, short_coeffs, short_coeff_digits) =
            short_challenge_from_digits_128(&mut gb, &short_digits_local, ring_dim);
        short_locals.insert(si, ShortChallengeWiring {
            digit_vars: short_digits_local,
            byte_vars: short_bvars,
            coeff_vars: short_coeffs,
            coeff_bal16_digits: short_coeff_digits,
        });
    }

    // Materialize required u32 blocks.
    let mut u32_locals: std::collections::BTreeMap<usize, BoundedU32ChallengeWiring> = std::collections::BTreeMap::new();
    for &(_si, ui) in pairs {
        if u32_locals.contains_key(&ui) {
            continue;
        }
        let (u_start, u_len) = u32_ranges[ui];
        if u_len != DIGITS_PER_TRY {
            return Err(format!("u32 block {ui} length mismatch (got {u_len}, expected {DIGITS_PER_TRY})"));
        }
        let u_digits_global = &pose_wiring.squeeze_field_vars[u_start..u_start + u_len];
        let mut u_digits_local = [0usize; DIGITS_PER_TRY];
        for i in 0..DIGITS_PER_TRY {
            u_digits_local[i] = copy_digit(&mut gb, &pose_asg, &mut local_map, u_digits_global[i]);
        }
        let (u_limbs, u_bytes, u_bal16, u_bal16_sq) =
            bounded_u32_from_8_digits_base128(&mut gb, &u_digits_local);
        u32_locals.insert(ui, BoundedU32ChallengeWiring {
            digit_vars: u_digits_local.to_vec(),
            byte_vars: u_bytes,
            limbs: u_limbs,
            bal16_digits: u_bal16,
            bal16_sq_digits: u_bal16_sq,
        });
    }

    // Build requested digit-mul surfaces.
    let mut surfaces_local: Vec<CmDigitMulSurfaceWiring> = Vec::with_capacity(pairs.len());
    for &(si, ui) in pairs {
        let s = short_locals.get(&si).expect("short local present");
        let u = u32_locals.get(&ui).expect("u32 local present");
        let products = scale_short_coeffs_by_u32(&mut gb, &s.coeff_bal16_digits, &u.bal16_digits);
        let products13 = products
            .iter()
            .map(|p12| rebalance_prod12_to_prod13(&mut gb, p12))
            .collect::<Vec<_>>();
        let sum_digits = sum_product_digits_bal16(&mut gb, &products13, 16);
        surfaces_local.push(CmDigitMulSurfaceWiring {
            short_block_idx: si,
            u32_idx: ui,
            products,
            products13,
            sum_digits,
            sum_all_pairs_digits: Vec::new(), // filled after we know all surfaces
            sum_all_pairs_coeffwise: Vec::new(), // filled after we know all surfaces
        });
    }

    // Batch-level accumulation across all requested surfaces (sum of sums).
    let all_sum_digits = sum_bal16_vectors_fixed_len(
        &mut gb,
        &surfaces_local.iter().map(|s| s.sum_digits.clone()).collect::<Vec<_>>(),
        16,
    );
    let all_sum_coeffwise = sum_products13_coeffwise_fixed_len(
        &mut gb,
        &surfaces_local.iter().map(|s| s.products13.clone()).collect::<Vec<_>>(),
        ring_dim,
        16,
    );
    for s in &mut surfaces_local {
        s.sum_all_pairs_digits = all_sum_digits.clone();
        s.sum_all_pairs_coeffwise = all_sum_coeffwise.clone();
    }

    // Frog coins (same as existing builder): copy all frog candidate digit vars and run sampler.
    let mut frog_digit_vars_local: Vec<usize> = Vec::with_capacity(n_coins * tries * DIGITS_PER_TRY);
    let mut frog_digit_vars_global: Vec<usize> = Vec::with_capacity(n_coins * tries * DIGITS_PER_TRY);
    for &(start, len) in &frog_ranges {
        if len != DIGITS_PER_TRY {
            return Err("frog squeeze len mismatch".to_string());
        }
        for v in &pose_wiring.squeeze_field_vars[start..start + len] {
            frog_digit_vars_global.push(*v);
            frog_digit_vars_local.push(copy_digit(&mut gb, &pose_asg, &mut local_map, *v));
        }
    }
    let mut frog_wirings_local: Vec<FrogRejectionCoinWiring> = Vec::with_capacity(n_coins);
    let mut coin_limbs_local_all: Vec<[usize; LIMBS_U64]> = Vec::with_capacity(n_coins);
    let mut found_local_all: Vec<usize> = Vec::with_capacity(n_coins);
    for coin_idx in 0..n_coins {
        let off = coin_idx * tries * DIGITS_PER_TRY;
        let digits = &frog_digit_vars_local[off..off + tries * DIGITS_PER_TRY];
        let (coin_local, found_local) =
            sample_frog_coin_unrolled_rejection_8_digits::<F257>(&mut gb, digits, tries);
        gb.enforce_var_eq_const(found_local, F257::ONE);
        coin_limbs_local_all.push(coin_local);
        found_local_all.push(found_local);
        frog_wirings_local.push(FrogRejectionCoinWiring {
            digit_vars: frog_digit_vars_local[off..off + tries * DIGITS_PER_TRY].to_vec(),
            found_bit: found_local,
            coin_limbs: coin_local.to_vec(),
            tries,
        });
    }

    let (glue_inst, glue_asg) = gb.into_instance();
    let glue_nvars = glue_inst.nvars;

    // Merge poseidon + glue.
    let (mut inst, asg) = merge_sparse_dr1cs_share_one::<F257>(vec![
        (pose_inst, pose_asg),
        (glue_inst, glue_asg),
    ])
    .map_err(|e| format!("merge poseidon+cm+frog+mul_glue failed: {e}"))?;

    // Add explicit equality constraints between pose vars and their local copies.
    let pose_nvars = inst.nvars - (glue_nvars - 1);
    let glue_offset = pose_nvars - 1;
    for (&gv, &lv) in local_map.iter() {
        let gg = if lv == 0 { 0 } else { lv + glue_offset };
        enforce_var_eq::<F257>(&mut inst, gv, gg);
    }

    let to_glue_global = |glue_local: usize| -> usize {
        if glue_local == 0 { 0 } else { glue_local + glue_offset }
    };

    // Return only the derived shorts/u32s (those referenced by pairs), in sorted index order.
    let shorts_out = short_locals
        .into_iter()
        .map(|(_idx, w)| ShortChallengeWiring {
            digit_vars: w.digit_vars.into_iter().map(to_glue_global).collect(),
            byte_vars: w.byte_vars.into_iter().map(to_glue_global).collect(),
            coeff_vars: w.coeff_vars.into_iter().map(to_glue_global).collect(),
            coeff_bal16_digits: w
                .coeff_bal16_digits
                .into_iter()
                .map(|a| a.map(to_glue_global))
                .collect(),
        })
        .collect::<Vec<_>>();
    let u32s_out = u32_locals
        .into_iter()
        .map(|(_idx, w)| BoundedU32ChallengeWiring {
            digit_vars: w.digit_vars.into_iter().map(to_glue_global).collect(),
            byte_vars: w.byte_vars.map(to_glue_global),
            limbs: w.limbs.map(to_glue_global),
            bal16_digits: w.bal16_digits.into_iter().map(to_glue_global).collect(),
            bal16_sq_digits: w.bal16_sq_digits.into_iter().map(to_glue_global).collect(),
        })
        .collect::<Vec<_>>();
    let frog_wirings = frog_wirings_local
        .into_iter()
        .map(|w| FrogRejectionCoinWiring {
            digit_vars: w.digit_vars.into_iter().map(to_glue_global).collect(),
            found_bit: to_glue_global(w.found_bit),
            coin_limbs: w.coin_limbs.into_iter().map(to_glue_global).collect(),
            tries: w.tries,
        })
        .collect::<Vec<_>>();

    let surfaces = surfaces_local
        .into_iter()
        .map(|s| CmDigitMulSurfaceWiring {
            short_block_idx: s.short_block_idx,
            u32_idx: s.u32_idx,
            products: s.products.into_iter().map(|arr| arr.map(to_glue_global)).collect(),
            products13: s.products13.into_iter().map(|arr| arr.map(to_glue_global)).collect(),
            sum_digits: s.sum_digits.into_iter().map(to_glue_global).collect(),
            sum_all_pairs_digits: s.sum_all_pairs_digits.into_iter().map(to_glue_global).collect(),
            sum_all_pairs_coeffwise: s
                .sum_all_pairs_coeffwise
                .into_iter()
                .map(|v| v.into_iter().map(to_glue_global).collect())
                .collect(),
        })
        .collect::<Vec<_>>();

    Ok((inst, asg, shorts_out, u32s_out, frog_wirings, surfaces))
}

/// Build Poseidon(F257) + CM coin surface + a batch of digit-mul surfaces (NO Frog coins).
///
/// Each pair `(short_block_idx, u32_idx)` requests multiplying all coeffs in that short block by
/// the bounded-u32 challenge at `u32_idx` **and** (in parallel) by its square `u32^2`.
pub fn build_poseidon_f257_with_cm_coins_and_digit_mul_surfaces_from_ops_with_wiring(
    cfg: Option<&PoseidonConfig<F257>>,
    ops: &[PoseidonTraceOp<F257>],
    ring_dim: usize,
    wiring: &TinyCoinOpWiring,
    pairs: &[(usize, usize)],
) -> Result<
    (
        SparseDr1csInstance<F257>,
        Vec<F257>,
        Vec<ShortChallengeWiring>,
        Vec<BoundedU32ChallengeWiring>,
        Vec<CmDigitMulSurfaceWiring>,
        Vec<CmDigitMulSqSurfaceWiring>,
    ),
    String,
> {
    let (pose_inst, pose_asg, pose_wiring, _byte_wiring) =
        build_poseidon_f257_from_ops_with_wiring_and_bytes(cfg, ops)?;

    // Resolve ranges from op indices.
    let short_ranges =
        squeeze_field_ranges_by_op_index(&pose_wiring.squeeze_field_ranges, &wiring.short_squeeze_ops)?;
    let u32_ranges =
        squeeze_field_ranges_by_op_index(&pose_wiring.squeeze_field_ranges, &wiring.u32_squeeze_ops)?;

    // Build glue circuit (coins + mul surface).
    let mut gb = Dr1csBuilder::<F257>::new();
    gb.enforce_var_eq_const(gb.one(), F257::ONE);

    // Copy digit vars for shorts and u32s in appearance order.
    let mut local_map: std::collections::BTreeMap<usize, usize> = std::collections::BTreeMap::new();
    #[inline]
    fn copy_digit(
        gb: &mut Dr1csBuilder<F257>,
        pose_asg: &[F257],
        local_map: &mut std::collections::BTreeMap<usize, usize>,
        gv: usize,
    ) -> usize {
        if let Some(&lv) = local_map.get(&gv) {
            return lv;
        }
        let lv = gb.new_var(pose_asg[gv]);
        local_map.insert(gv, lv);
        lv
    }

    // Determine which short/u32 blocks we need.
    for &(si, ui) in pairs {
        if si >= short_ranges.len() {
            return Err(format!("short_block_idx {si} out of range"));
        }
        if ui >= u32_ranges.len() {
            return Err(format!("u32_idx {ui} out of range"));
        }
    }

    // Materialize required short blocks.
    let mut short_locals: std::collections::BTreeMap<usize, ShortChallengeWiring> =
        std::collections::BTreeMap::new();
    for &(si, _ui) in pairs {
        if short_locals.contains_key(&si) {
            continue;
        }
        let (s_start, s_len) = short_ranges[si];
        if s_len != ring_dim {
            return Err(format!(
                "short block {si} length mismatch (got {s_len}, expected {ring_dim})"
            ));
        }
        let short_digits_global = &pose_wiring.squeeze_field_vars[s_start..s_start + s_len];
        let short_digits_local: Vec<usize> = short_digits_global
            .iter()
            .copied()
            .map(|gv| copy_digit(&mut gb, &pose_asg, &mut local_map, gv))
            .collect();
        let (short_bvars, short_coeffs, short_coeff_digits) =
            short_challenge_from_digits_128(&mut gb, &short_digits_local, ring_dim);
        short_locals.insert(
            si,
            ShortChallengeWiring {
                digit_vars: short_digits_local,
                byte_vars: short_bvars,
                coeff_vars: short_coeffs,
                coeff_bal16_digits: short_coeff_digits,
            },
        );
    }

    // Materialize required u32 blocks.
    let mut u32_locals: std::collections::BTreeMap<usize, BoundedU32ChallengeWiring> =
        std::collections::BTreeMap::new();
    for &(_si, ui) in pairs {
        if u32_locals.contains_key(&ui) {
            continue;
        }
        let (u_start, u_len) = u32_ranges[ui];
        if u_len != DIGITS_PER_TRY {
            return Err(format!(
                "u32 block {ui} length mismatch (got {u_len}, expected {DIGITS_PER_TRY})"
            ));
        }
        let u_digits_global = &pose_wiring.squeeze_field_vars[u_start..u_start + u_len];
        let mut u_digits_local = [0usize; DIGITS_PER_TRY];
        for i in 0..DIGITS_PER_TRY {
            u_digits_local[i] = copy_digit(&mut gb, &pose_asg, &mut local_map, u_digits_global[i]);
        }
        let (u_limbs, u_bytes, u_bal16, u_bal16_sq) =
            bounded_u32_from_8_digits_base128(&mut gb, &u_digits_local);
        u32_locals.insert(
            ui,
            BoundedU32ChallengeWiring {
                digit_vars: u_digits_local.to_vec(),
                byte_vars: u_bytes,
                limbs: u_limbs,
                bal16_digits: u_bal16,
                bal16_sq_digits: u_bal16_sq,
            },
        );
    }

    // Build requested digit-mul surfaces (u32).
    let mut surfaces_mul_local: Vec<CmDigitMulSurfaceWiring> = Vec::with_capacity(pairs.len());
    for &(si, ui) in pairs {
        let s = short_locals.get(&si).expect("short local present");
        let u = u32_locals.get(&ui).expect("u32 local present");
        let products = scale_short_coeffs_by_u32(&mut gb, &s.coeff_bal16_digits, &u.bal16_digits);
        let products13 = products
            .iter()
            .map(|p12| rebalance_prod12_to_prod13(&mut gb, p12))
            .collect::<Vec<_>>();
        let sum_digits = sum_product_digits_bal16(&mut gb, &products13, 16);
        surfaces_mul_local.push(CmDigitMulSurfaceWiring {
            short_block_idx: si,
            u32_idx: ui,
            products,
            products13,
            sum_digits,
            sum_all_pairs_digits: Vec::new(), // filled after we know all surfaces
            sum_all_pairs_coeffwise: Vec::new(), // filled after we know all surfaces
        });
    }

    // Batch-level accumulation across all requested u32 surfaces (sum of sums).
    let all_sum_digits = sum_bal16_vectors_fixed_len(
        &mut gb,
        &surfaces_mul_local
            .iter()
            .map(|s| s.sum_digits.clone())
            .collect::<Vec<_>>(),
        16,
    );
    let all_sum_coeffwise = sum_products13_coeffwise_fixed_len(
        &mut gb,
        &surfaces_mul_local
            .iter()
            .map(|s| s.products13.clone())
            .collect::<Vec<_>>(),
        ring_dim,
        16,
    );
    for s in &mut surfaces_mul_local {
        s.sum_all_pairs_digits = all_sum_digits.clone();
        s.sum_all_pairs_coeffwise = all_sum_coeffwise.clone();
    }

    // Build requested digit-mul surfaces (u32^2).
    let mut surfaces_sq_local: Vec<CmDigitMulSqSurfaceWiring> = Vec::with_capacity(pairs.len());
    for &(si, ui) in pairs {
        let s = short_locals.get(&si).expect("short local present");
        let u = u32_locals.get(&ui).expect("u32 local present");
        let products21 =
            scale_short_coeffs_by_digits18(&mut gb, &s.coeff_bal16_digits, &u.bal16_sq_digits);
        let products22 = products21
            .iter()
            .map(|p21| rebalance_prod21_to_prod22(&mut gb, p21))
            .collect::<Vec<_>>();
        let sum_digits = sum_product_digits_bal16_22(&mut gb, &products22, 24);
        surfaces_sq_local.push(CmDigitMulSqSurfaceWiring {
            short_block_idx: si,
            u32_idx: ui,
            products21,
            products22,
            sum_digits,
            sum_all_pairs_digits: Vec::new(),
            sum_all_pairs_coeffwise: Vec::new(),
        });
    }

    // Batch-level accumulation across all requested u32^2 surfaces (sum of sums).
    let all_sq_sum_digits = sum_bal16_vectors_fixed_len(
        &mut gb,
        &surfaces_sq_local
            .iter()
            .map(|s| s.sum_digits.clone())
            .collect::<Vec<_>>(),
        24,
    );
    let all_sq_sum_coeffwise = sum_products22_coeffwise_fixed_len(
        &mut gb,
        &surfaces_sq_local
            .iter()
            .map(|s| s.products22.clone())
            .collect::<Vec<_>>(),
        ring_dim,
        24,
    );
    for s in &mut surfaces_sq_local {
        s.sum_all_pairs_digits = all_sq_sum_digits.clone();
        s.sum_all_pairs_coeffwise = all_sq_sum_coeffwise.clone();
    }

    let (glue_inst, glue_asg) = gb.into_instance();
    let glue_nvars = glue_inst.nvars;

    // Merge poseidon + glue.
    let (mut inst, asg) = merge_sparse_dr1cs_share_one::<F257>(vec![
        (pose_inst, pose_asg),
        (glue_inst, glue_asg),
    ])
    .map_err(|e| format!("merge poseidon+cm+mul_glue failed: {e}"))?;

    // Add explicit equality constraints between pose vars and their local copies.
    let pose_nvars = inst.nvars - (glue_nvars - 1);
    let glue_offset = pose_nvars - 1;
    for (&gv, &lv) in local_map.iter() {
        let gg = if lv == 0 { 0 } else { lv + glue_offset };
        enforce_var_eq::<F257>(&mut inst, gv, gg);
    }

    let to_glue_global = |glue_local: usize| -> usize {
        if glue_local == 0 { 0 } else { glue_local + glue_offset }
    };

    // Return only the derived shorts/u32s (those referenced by pairs), in sorted index order.
    let shorts_out = short_locals
        .into_iter()
        .map(|(_idx, w)| ShortChallengeWiring {
            digit_vars: w.digit_vars.into_iter().map(to_glue_global).collect(),
            byte_vars: w.byte_vars.into_iter().map(to_glue_global).collect(),
            coeff_vars: w.coeff_vars.into_iter().map(to_glue_global).collect(),
            coeff_bal16_digits: w
                .coeff_bal16_digits
                .into_iter()
                .map(|a| a.map(to_glue_global))
                .collect(),
        })
        .collect::<Vec<_>>();
    let u32s_out = u32_locals
        .into_iter()
        .map(|(_idx, w)| BoundedU32ChallengeWiring {
            digit_vars: w.digit_vars.into_iter().map(to_glue_global).collect(),
            byte_vars: w.byte_vars.map(to_glue_global),
            limbs: w.limbs.map(to_glue_global),
            bal16_digits: w.bal16_digits.into_iter().map(to_glue_global).collect(),
            bal16_sq_digits: w.bal16_sq_digits.into_iter().map(to_glue_global).collect(),
        })
        .collect::<Vec<_>>();
    let surfaces_out = surfaces_mul_local
        .into_iter()
        .map(|s| CmDigitMulSurfaceWiring {
            short_block_idx: s.short_block_idx,
            u32_idx: s.u32_idx,
            products: s
                .products
                .into_iter()
                .map(|p| p.map(to_glue_global))
                .collect(),
            products13: s
                .products13
                .into_iter()
                .map(|p| p.map(to_glue_global))
                .collect(),
            sum_digits: s.sum_digits.into_iter().map(to_glue_global).collect(),
            sum_all_pairs_digits: s.sum_all_pairs_digits.into_iter().map(to_glue_global).collect(),
            sum_all_pairs_coeffwise: s
                .sum_all_pairs_coeffwise
                .into_iter()
                .map(|v| v.into_iter().map(to_glue_global).collect())
                .collect(),
        })
        .collect::<Vec<_>>();

    let surfaces_sq_out = surfaces_sq_local
        .into_iter()
        .map(|s| CmDigitMulSqSurfaceWiring {
            short_block_idx: s.short_block_idx,
            u32_idx: s.u32_idx,
            products21: s.products21.into_iter().map(|arr| arr.map(to_glue_global)).collect(),
            products22: s.products22.into_iter().map(|arr| arr.map(to_glue_global)).collect(),
            sum_digits: s.sum_digits.into_iter().map(to_glue_global).collect(),
            sum_all_pairs_digits: s.sum_all_pairs_digits.into_iter().map(to_glue_global).collect(),
            sum_all_pairs_coeffwise: s
                .sum_all_pairs_coeffwise
                .into_iter()
                .map(|v| v.into_iter().map(to_glue_global).collect())
                .collect(),
        })
        .collect::<Vec<_>>();

    Ok((inst, asg, shorts_out, u32s_out, surfaces_out, surfaces_sq_out))
}
/// Build the Poseidon transcript subrelation **over F257** from an op schedule.
///
/// This is the correct transcript-layer arithmetization for the Theorem-4.3 tiny-field WE gate:
/// the permutation and IO are all in the tiny field (no Frog Poseidon involved).
pub fn build_poseidon_f257_from_ops(
    cfg: Option<&PoseidonConfig<F257>>,
    ops: &[PoseidonTraceOp<F257>],
) -> Result<(SparseDr1csInstance<F257>, Vec<F257>), String> {
    let default_cfg;
    let cfg = match cfg {
        Some(c) => c,
        None => {
            default_cfg = f257_poseidon_config();
            &default_cfg
        }
    };
    let (inst, asg, _replay, _byte_wit, _wiring, _byte_wiring) =
        poseidon_sponge_dr1cs_from_ops_with_wiring_and_bytes::<F257>(cfg, ops)
            .map_err(|e| format!("poseidon(F257) arith failed: {e}"))?;
    debug_assert_eq!(inst.nvars, asg.len());
    Ok((inst, asg))
}

/// Same as `build_poseidon_f257_from_ops`, but also returns `PoseidonByteWiring` for `SqueezeBytes`.
pub fn build_poseidon_f257_from_ops_with_bytes(
    cfg: Option<&PoseidonConfig<F257>>,
    ops: &[PoseidonTraceOp<F257>],
) -> Result<(SparseDr1csInstance<F257>, Vec<F257>, PoseidonByteWiring), String> {
    let default_cfg;
    let cfg = match cfg {
        Some(c) => c,
        None => {
            default_cfg = f257_poseidon_config();
            &default_cfg
        }
    };
    let (inst, asg, _replay, _byte_wit, _wiring, byte_wiring) =
        poseidon_sponge_dr1cs_from_ops_with_wiring_and_bytes::<F257>(cfg, ops)
            .map_err(|e| format!("poseidon(F257) arith failed: {e}"))?;
    debug_assert_eq!(inst.nvars, asg.len());
    Ok((inst, asg, byte_wiring))
}

/// Same as `build_poseidon_f257_from_ops`, but returns both field wiring and byte wiring.
pub fn build_poseidon_f257_from_ops_with_wiring_and_bytes(
    cfg: Option<&PoseidonConfig<F257>>,
    ops: &[PoseidonTraceOp<F257>],
) -> Result<(SparseDr1csInstance<F257>, Vec<F257>, PoseidonDr1csWiring, PoseidonByteWiring), String> {
    let default_cfg;
    let cfg = match cfg {
        Some(c) => c,
        None => {
            default_cfg = f257_poseidon_config();
            &default_cfg
        }
    };
    let (inst, asg, _replay, _byte_wit, wiring, byte_wiring) =
        poseidon_sponge_dr1cs_from_ops_with_wiring_and_bytes::<F257>(cfg, ops)
            .map_err(|e| format!("poseidon(F257) arith failed: {e}"))?;
    debug_assert_eq!(inst.nvars, asg.len());
    Ok((inst, asg, wiring, byte_wiring))
}

#[cfg(test)]
mod tests {
    use super::*;

    use latticefold::transcript::Transcript;
    use stark_rings::cyclotomic_ring::models::frog_ring::RqPoly as FrogRing;

    fn var_to_u8<F: PrimeField>(b: &Dr1csBuilder<F>, v: usize) -> u8 {
        b.assignment[v]
            .into_bigint()
            .to_bytes_le()
            .get(0)
            .copied()
            .unwrap_or(0)
    }

    fn limbs_to_u64_base128<F: PrimeField>(b: &Dr1csBuilder<F>, limbs: &[usize; LIMBS_U64]) -> u64 {
        let mut acc: u64 = 0;
        for i in (0..LIMBS_U64).rev() {
            let di = b.assignment[limbs[i]]
                .into_bigint()
                .to_bytes_le()
                .get(0)
                .copied()
                .unwrap_or(0) as u64;
            acc <<= LIMB_BITS;
            acc |= di & (LIMB_BASE_U64 - 1);
        }
        acc
    }

    fn limbs_u32_from_base128<F: PrimeField>(b: &Dr1csBuilder<F>, limbs: &[usize; LIMBS_U32]) -> u32 {
        let mut acc: u64 = 0;
        for i in (0..LIMBS_U32).rev() {
            let di = b.assignment[limbs[i]]
                .into_bigint()
                .to_bytes_le()
                .get(0)
                .copied()
                .unwrap_or(0) as u64;
            acc <<= LIMB_BITS;
            acc |= di & (LIMB_BASE_U64 - 1);
        }
        acc as u32
    }

    #[test]
    fn test_babybear31_from_u32_bytes_satisfies_and_decodes() {
        let mut b = Dr1csBuilder::<F257>::new();
        b.enforce_var_eq_const(b.one(), F257::ONE);

        // Some representative values in [0, p_bb).
        let half: u32 = BABYBEAR_P_U32 / 2;
        let vals: [u32; 5] = [0, 1, half, half + 1, BABYBEAR_P_U32 - 1];
        for &x in &vals {
            let xb = x.to_le_bytes();
            let bytes = [
                alloc_byte::<F257>(&mut b, xb[0]).byte,
                alloc_byte::<F257>(&mut b, xb[1]).byte,
                alloc_byte::<F257>(&mut b, xb[2]).byte,
                alloc_byte::<F257>(&mut b, xb[3]).byte,
            ];
            let w = babybear31_from_u32_byte_vars(&mut b, bytes);
            assert_eq!(w.bal16_digits.len(), 9);

            // Decode balanced base-16 digits and check it matches x.
            let mut acc: i128 = 0;
            let mut pow: i128 = 1;
            for &dv in &w.bal16_digits {
                acc += (f257_to_i32_bal(b.assignment[dv]) as i128) * pow;
                pow *= 16;
            }
            assert_eq!(acc as u32, x);
        }

        let (inst, asg) = b.into_instance();
        inst.check(&asg).expect("babybear31 gadget satisfied");
    }

    #[test]
    fn test_babybear31_rejects_noncanonical() {
        // Case 1: x == p_bb is not allowed (must be < p_bb).
        {
            let mut b = Dr1csBuilder::<F257>::new();
            b.enforce_var_eq_const(b.one(), F257::ONE);
            let xb = BABYBEAR_P_U32.to_le_bytes();
            let bytes = [
                alloc_byte::<F257>(&mut b, xb[0]).byte,
                alloc_byte::<F257>(&mut b, xb[1]).byte,
                alloc_byte::<F257>(&mut b, xb[2]).byte,
                alloc_byte::<F257>(&mut b, xb[3]).byte,
            ];
            let _ = babybear31_from_u32_byte_vars_with_modulus(&mut b, bytes, BABYBEAR_P_U32);
            let (inst, asg) = b.into_instance();
            assert!(inst.check(&asg).is_err(), "expected x==p_bb to be rejected");
        }

        // Case 2: bit31 set (>= 2^31) is rejected by the 31-bit constraint.
        {
            let mut b = Dr1csBuilder::<F257>::new();
            b.enforce_var_eq_const(b.one(), F257::ONE);
            let xb = 0x8000_0000u32.to_le_bytes();
            let bytes = [
                alloc_byte::<F257>(&mut b, xb[0]).byte,
                alloc_byte::<F257>(&mut b, xb[1]).byte,
                alloc_byte::<F257>(&mut b, xb[2]).byte,
                alloc_byte::<F257>(&mut b, xb[3]).byte,
            ];
            let _ = babybear31_from_u32_byte_vars(&mut b, bytes);
            let (inst, asg) = b.into_instance();
            assert!(inst.check(&asg).is_err(), "expected bit31-set value to be rejected");
        }
    }

    #[test]
    fn test_babybear_centered_from_u32_bytes_matches_centering() {
        // Check that the centered digits decode to x or x-p depending on x<=p/2.
        let p: u32 = BABYBEAR_P_U32;
        let half: u32 = p / 2;
        let vals: [u32; 6] = [0, 1, half - 1, half, half + 1, p - 1];

        for &x in &vals {
            let mut b = Dr1csBuilder::<F257>::new();
            b.enforce_var_eq_const(b.one(), F257::ONE);

            let xb = x.to_le_bytes();
            let bytes = [
                alloc_byte::<F257>(&mut b, xb[0]).byte,
                alloc_byte::<F257>(&mut b, xb[1]).byte,
                alloc_byte::<F257>(&mut b, xb[2]).byte,
                alloc_byte::<F257>(&mut b, xb[3]).byte,
            ];
            let w = babybear_centered_from_u32_byte_vars_with_modulus(&mut b, bytes, p);
            assert_eq!(w.centered_bal16_digits.len(), 9);

            let (inst, asg) = b.into_instance();
            inst.check(&asg).expect("centered babybear gadget satisfied");

            // Decode centered digits.
            let mut acc: i128 = 0;
            let mut pow: i128 = 1;
            for &dv in &w.centered_bal16_digits {
                acc += (f257_to_i32_bal(asg[dv]) as i128) * pow;
                pow *= 16;
            }

            let expected: i128 = if x <= half { x as i128 } else { (x as i128) - (p as i128) };
            assert_eq!(acc, expected);
        }
    }

    #[test]
    fn test_poseidon_f257_ops_arithmetization_satisfies() {
        // Record a tiny transcript trace in the **actual sponge field** (F257).
        //
        // IMPORTANT: we must not use `crate::recording_transcript::TracePoseidonTranscript`,
        // which lifts F257 digits into the outer base ring. For a tiny-field gate we want the
        // transcript ops directly over F257.
        let mut tr = symphony::transcript::TracePoseidonTranscript::<FrogRing>::empty::<()>();
        tr.absorb_field_element(&<FrogRing as stark_rings::PolyRing>::BaseRing::from(123u64));
        let _c = tr.get_challenge(); // SqueezeField(8) + Absorb(8)
        let _b = tr.squeeze_bytes(17); // SqueezeBytes(17) (no reabsorb)

        let ops: Vec<PoseidonTraceOp<F257>> = tr.trace().ops.clone();

        let (inst, asg) = build_poseidon_f257_from_ops(None, &ops).expect("build_poseidon_f257_from_ops");
        inst.check(&asg).expect("poseidon(F257) dR1CS satisfied");
    }

    #[test]
    fn test_reduce_u64_mod_frog_from_byte_vars_no_wrap() {
        let mut b = Dr1csBuilder::<F257>::new();
        b.enforce_var_eq_const(b.one(), F257::ONE);

        let u = FROG_P + 12345;
        let mut u_byte_vars = [0usize; 8];
        for (i, v) in u.to_le_bytes().into_iter().enumerate() {
            let bv = alloc_byte::<F257>(&mut b, v);
            u_byte_vars[i] = bv.byte;
        }
        let (_q, _z) = reduce_u64_mod_frog_from_byte_vars::<F257>(&mut b, &u_byte_vars);

        let (inst, asg) = b.into_instance();
        inst.check(&asg).expect("reduce_u64_mod_frog_from_byte_vars dR1CS satisfied");
    }

    #[test]
    fn test_reduce_u64_mod_frog_branch_u_lt_p() {
        let mut b = Dr1csBuilder::<F257>::new();
        b.enforce_var_eq_const(b.one(), F257::ONE);

        let u = 42u64;
        let mut u_byte_vars = [0usize; 8];
        for (i, v) in u.to_le_bytes().into_iter().enumerate() {
            let bv = alloc_byte::<F257>(&mut b, v);
            u_byte_vars[i] = bv.byte;
        }
        let (q, z) = reduce_u64_mod_frog_from_byte_vars::<F257>(&mut b, &u_byte_vars);

        // q should be 0, z == u.
        assert_eq!(var_to_u8::<F257>(&b, q), 0);
        assert_eq!(limbs_to_u64_base128::<F257>(&b, &z), u);

        let (inst, asg) = b.into_instance();
        inst.check(&asg).expect("branch u<p satisfied");
    }

    #[test]
    fn test_reduce_u64_mod_frog_branch_u_ge_p() {
        let mut b = Dr1csBuilder::<F257>::new();
        b.enforce_var_eq_const(b.one(), F257::ONE);

        let u = FROG_P + 424242u64;
        let mut u_byte_vars = [0usize; 8];
        for (i, v) in u.to_le_bytes().into_iter().enumerate() {
            let bv = alloc_byte::<F257>(&mut b, v);
            u_byte_vars[i] = bv.byte;
        }
        let (q, z) = reduce_u64_mod_frog_from_byte_vars::<F257>(&mut b, &u_byte_vars);

        // q should be 1, z == u - p.
        assert_eq!(var_to_u8::<F257>(&b, q), 1);
        assert_eq!(limbs_to_u64_base128::<F257>(&b, &z), u - FROG_P);

        let (inst, asg) = b.into_instance();
        inst.check(&asg).expect("branch u>=p satisfied");
    }

    #[test]
    fn test_reduce_u64_mod_frog_rejects_wrong_q() {
        let mut b = Dr1csBuilder::<F257>::new();
        b.enforce_var_eq_const(b.one(), F257::ONE);

        let u = FROG_P + 7u64;
        let mut u_byte_vars = [0usize; 8];
        for (i, v) in u.to_le_bytes().into_iter().enumerate() {
            let bv = alloc_byte::<F257>(&mut b, v);
            u_byte_vars[i] = bv.byte;
        }
        let (q, _z) = reduce_u64_mod_frog_from_byte_vars::<F257>(&mut b, &u_byte_vars);

        let (inst, mut asg) = b.into_instance();
        // Flip q (should be 1 -> set to 0) without adjusting anything else.
        asg[q] = F257::ZERO;
        assert!(inst.check(&asg).is_err(), "flipped q should break constraints");
    }

    #[test]
    fn test_reduce_u64_mod_frog_from_byte_vars_accepts_poseidon_style_bytes() {
        let mut b = Dr1csBuilder::<F257>::new();
        b.enforce_var_eq_const(b.one(), F257::ONE);

        // Create 8 already-range-checked bytes and pass the raw byte vars.
        let u = FROG_P + 123u64;
        let mut u_byte_vars = [0usize; 8];
        for (i, v) in u.to_le_bytes().into_iter().enumerate() {
            let bv = alloc_byte::<F257>(&mut b, v);
            u_byte_vars[i] = bv.byte;
        }
        let (_q, _z) = reduce_u64_mod_frog_from_byte_vars::<F257>(&mut b, &u_byte_vars);

        let (inst, asg) = b.into_instance();
        inst.check(&asg).expect("reduce_u64_mod_frog_from_byte_vars satisfied");
    }

    #[test]
    fn test_poseidon_plus_rejection_coin_demo_satisfies() {
        use ark_crypto_primitives::sponge::{poseidon::PoseidonSponge, CryptographicSponge};

        let cfg = f257_poseidon_config();
        let mut sponge = PoseidonSponge::<F257>::new(&cfg);
        let tries: usize = DEFAULT_REJECTION_TRIES;

        let mut ops: Vec<PoseidonTraceOp<F257>> = Vec::new();
        let a = vec![F257::from(123u64)];
        sponge.absorb(&a);
        ops.push(PoseidonTraceOp::Absorb(a));
        for _ in 0..tries {
            let c = sponge.squeeze_field_elements::<F257>(DIGITS_PER_TRY);
            ops.push(PoseidonTraceOp::SqueezeField(c.clone()));
            sponge.absorb(&c);
            ops.push(PoseidonTraceOp::Absorb(c));
        }

        let (inst, asg, _w) =
            build_poseidon_f257_with_frog_coin_rejection_glue_from_ops_with_wiring(None, &ops, tries)
                .expect("build_poseidon_f257_with_frog_coin_rejection_glue_from_ops_with_wiring");
        inst.check(&asg).expect("poseidon+rejection coin demo satisfied");
    }

    #[test]
    fn test_poseidon_plus_rejection_two_coins_satisfies() {
        use ark_crypto_primitives::sponge::{poseidon::PoseidonSponge, CryptographicSponge};

        let cfg = f257_poseidon_config();
        let mut sponge = PoseidonSponge::<F257>::new(&cfg);
        let tries: usize = DEFAULT_REJECTION_TRIES;
        let n_coins: usize = 2;

        let mut ops: Vec<PoseidonTraceOp<F257>> = Vec::new();
        let a = vec![F257::from(123u64)];
        sponge.absorb(&a);
        ops.push(PoseidonTraceOp::Absorb(a));
        for _ in 0..(n_coins * tries) {
            let c = sponge.squeeze_field_elements::<F257>(DIGITS_PER_TRY);
            ops.push(PoseidonTraceOp::SqueezeField(c.clone()));
            sponge.absorb(&c);
            ops.push(PoseidonTraceOp::Absorb(c));
        }

        let (inst, asg, w) = build_poseidon_f257_with_frog_rejection_coins_from_ops_with_wiring(
            None, &ops, n_coins, tries,
        )
        .expect("build_poseidon_f257_with_frog_rejection_coins_from_ops_with_wiring");
        assert_eq!(w.len(), n_coins);
        inst.check(&asg).expect("poseidon+2 rejection coins satisfied");
    }

    #[test]
    fn test_bounded_u32_from_8_digits_base128_matches_byte_view() {
        let mut b = Dr1csBuilder::<F257>::new();
        b.enforce_var_eq_const(b.one(), F257::ONE);

        // Build a digit block with a 256 in it to exercise byte view (256 -> 0).
        let ds: [u16; 8] = [1, 2, 256, 4, 5, 6, 7, 8];
        let mut dvars = [0usize; 8];
        for i in 0..8 {
            dvars[i] = b.new_var(F257::from(ds[i] as u64));
        }

        let (limbs, bytes, bal16_digits, bal16_sq_digits) =
            bounded_u32_from_8_digits_base128(&mut b, &dvars);

        // Expected u32 from byte view of first 4 digits.
        let bv = |x: u16| if x == 256 { 0u8 } else { x as u8 };
        let exp = u32::from_le_bytes([bv(ds[0]), bv(ds[1]), bv(ds[2]), bv(ds[3])]);

        // Check reconstructed u32 from limbs.
        assert_eq!(limbs_u32_from_base128::<F257>(&b, &limbs), exp);

        // Also check bytes match.
        let to_u8 = |v: usize| var_to_u8::<F257>(&b, v);
        assert_eq!(to_u8(bytes[0]), bv(ds[0]));
        assert_eq!(to_u8(bytes[1]), bv(ds[1]));
        assert_eq!(to_u8(bytes[2]), bv(ds[2]));
        assert_eq!(to_u8(bytes[3]), bv(ds[3]));

        // Check balanced base-16 digits decode to the same u32.
        let mut acc: i128 = 0;
        let mut pow: i128 = 1;
        for &v in &bal16_digits {
            acc += (f257_to_i32_bal(b.assignment[v]) as i128) * pow;
            pow *= 16;
        }
        assert_eq!(acc as u64, exp as u64);

        // Check square digits decode to exp^2.
        let mut sq_acc: i128 = 0;
        let mut pow: i128 = 1;
        for &v in &bal16_sq_digits {
            sq_acc += (f257_to_i32_bal(b.assignment[v]) as i128) * pow;
            pow *= 16;
        }
        assert_eq!(sq_acc, (exp as i128) * (exp as i128));

        let (inst, asg) = b.into_instance();
        inst.check(&asg).expect("bounded u32 gadget satisfied");
    }

    #[test]
    fn test_short_coeff_digits_times_u32_roundtrip() {
        // Build one short coeff (from a single byte var) and one bounded-u32, then multiply using digits.
        let mut b = Dr1csBuilder::<F257>::new();
        b.enforce_var_eq_const(b.one(), F257::ONE);

        // Short challenge coeff from a fixed byte (range ends up in [-128,127]).
        let by = alloc_byte::<F257>(&mut b, 205);
        let coeff = short_challenge_coeff_from_byte_var::<F257>(&mut b, by.byte, 256);
        let v = f257_to_i32_bal(b.assignment[coeff]);
        // Construct its 2-digit balanced base16 representation and enforce equality.
        let mut d0 = v % 16;
        if d0 > 7 { d0 -= 16; }
        if d0 < -8 { d0 += 16; }
        let d1 = (v - d0) / 16;
        let d0v = alloc_bal16_digit(&mut b, d0 as i8);
        let d1v = alloc_bal16_digit(&mut b, d1 as i8);
        b.enforce_lc_times_one_eq_const(vec![
            (F257::ONE, coeff),
            (-F257::ONE, d0v),
            (-F257::from(16u64), d1v),
        ]);
        let coeff3 = [d0v, d1v, alloc_bal16_digit(&mut b, 0)];

        // Bounded u32 from digits.
        let ds: [u16; 8] = [1, 2, 3, 4, 5, 6, 7, 8];
        let mut dvars = [0usize; 8];
        for i in 0..8 {
            dvars[i] = b.new_var(F257::from(ds[i] as u64));
        }
        let (_limbs, _bytes, u32_digits, _u32_sq_digits) = bounded_u32_from_8_digits_base128(&mut b, &dvars);
        assert_eq!(u32_digits.len(), 9);

        let prod = mul_bal16_3_by_u32(&mut b, &coeff3, &u32_digits);

        let (inst, asg) = b.into_instance();
        inst.check(&asg).expect("coeff*u32 digit mul satisfiable");

        // Decode expected u32 directly from the chosen digit block (byte-view has no 256 here).
        let u32v = u32::from_le_bytes([1u8, 2u8, 3u8, 4u8]) as i128;
        let expected = (v as i128) * u32v;

        let mut acc: i128 = 0;
        let mut pow: i128 = 1;
        for &vv in &prod {
            acc += (f257_to_i32_bal(asg[vv]) as i128) * pow;
            pow *= 16;
        }
        assert_eq!(acc, expected);
    }

    #[test]
    fn test_short_coeff_digits_times_digits9_roundtrip() {
        use rand::{Rng, SeedableRng};
        let mut rng = rand::rngs::StdRng::seed_from_u64(0xD16175_9u64);
        let mut b = Dr1csBuilder::<F257>::new();
        b.enforce_var_eq_const(b.one(), F257::ONE);

        // random small coeff in [-128,127]
        let coeff: i32 = rng.gen_range(-128..=127);
        // random 36-ish-bit scalar to keep i128 decode safe and stay "u32-ish envelope"
        let x: i128 = rng.gen_range(-(1i128 << 35)..(1i128 << 35));

        // coeff -> 3 balanced base16 digits
        let mut cur = coeff as i128;
        let mut c3 = [0usize; 3];
        for i in 0..3 {
            let mut r = ((cur % 16) + 16) % 16;
            if r >= 8 {
                r -= 16;
            }
            c3[i] = alloc_bal16_digit(&mut b, r as i8);
            cur = (cur - (r as i128)) / 16;
        }
        debug_assert_eq!(cur, 0);

        // x -> 9 balanced digits (little-endian), with a final carry digit possibly nonzero.
        let x_digits = {
            let mut xx = x;
            let mut out: Vec<usize> = Vec::with_capacity(9);
            for _ in 0..9 {
                let mut r = ((xx % 16) + 16) % 16;
                if r >= 8 {
                    r -= 16;
                }
                out.push(alloc_bal16_digit(&mut b, r as i8));
                xx = (xx - (r as i128)) / 16;
            }
            out
        };

        let prod12 = mul_bal16_3_by_digits9(&mut b, &c3, &x_digits);

        // Decode prod (base16) and compare to native.
        let mut acc: i128 = 0;
        let mut pow: i128 = 1;
        for &dv in &prod12 {
            acc += (f257_to_i32_bal(b.assignment[dv]) as i128) * pow;
            pow *= 16;
        }
        assert_eq!(acc, (coeff as i128) * x);

        let (inst, asg) = b.into_instance();
        inst.check(&asg).expect("digits9 mul satisfied");
    }

    #[test]
    fn test_short_coeff_digits_times_u32_sq_digits18_roundtrip() {
        // Multiply one short coeff by (u32)^2 using the derived 18-digit representation.
        let mut b = Dr1csBuilder::<F257>::new();
        b.enforce_var_eq_const(b.one(), F257::ONE);

        // Short challenge coeff from a fixed byte (range ends up in [-128,127]).
        let by = alloc_byte::<F257>(&mut b, 205);
        let coeff = short_challenge_coeff_from_byte_var::<F257>(&mut b, by.byte, 256);
        let v = f257_to_i32_bal(b.assignment[coeff]);
        let mut d0 = v % 16;
        if d0 > 7 { d0 -= 16; }
        if d0 < -8 { d0 += 16; }
        let d1 = (v - d0) / 16;
        let d0v = alloc_bal16_digit(&mut b, d0 as i8);
        let d1v = alloc_bal16_digit(&mut b, d1 as i8);
        b.enforce_lc_times_one_eq_const(vec![
            (F257::ONE, coeff),
            (-F257::ONE, d0v),
            (-F257::from(16u64), d1v),
        ]);
        let coeff3 = [d0v, d1v, alloc_bal16_digit(&mut b, 0)];

        // Bounded u32 from digits; also returns u^2 digits (len 18).
        let ds: [u16; 8] = [1, 2, 3, 4, 5, 6, 7, 8];
        let mut dvars = [0usize; 8];
        for i in 0..8 {
            dvars[i] = b.new_var(F257::from(ds[i] as u64));
        }
        let (_limbs, _bytes, _u32_digits, u32_sq_digits) =
            bounded_u32_from_8_digits_base128(&mut b, &dvars);
        assert_eq!(u32_sq_digits.len(), 18);

        let prod = mul_bal16_3_by_digits18(&mut b, &coeff3, &u32_sq_digits);

        let (inst, asg) = b.into_instance();
        inst.check(&asg).expect("coeff*u32^2 digit mul satisfiable");

        let u32v = u32::from_le_bytes([1u8, 2u8, 3u8, 4u8]) as i128;
        let expected = (v as i128) * (u32v * u32v);

        let mut acc: i128 = 0;
        let mut pow: i128 = 1;
        for (idx, &vv) in prod.iter().enumerate() {
            acc += (f257_to_i32_bal(asg[vv]) as i128) * pow;
            if idx + 1 < prod.len() {
                pow *= 16;
            }
        }
        assert_eq!(acc, expected);
    }

    #[test]
    fn test_mul_bal16_9_by_9_u32ish_roundtrip() {
        // Multiply two u32 values via their balanced base-16 digit representations (9 digits each).
        use rand::Rng;
        let mut rng = ark_std::test_rng();

        for _ in 0..50 {
            let x: u32 = rng.gen();
            let y: u32 = rng.gen();

            let mut b = Dr1csBuilder::<F257>::new();
            b.enforce_var_eq_const(b.one(), F257::ONE);

            let xb = x.to_le_bytes();
            let yb = y.to_le_bytes();
            let x_bytes = [
                alloc_byte::<F257>(&mut b, xb[0]).byte,
                alloc_byte::<F257>(&mut b, xb[1]).byte,
                alloc_byte::<F257>(&mut b, xb[2]).byte,
                alloc_byte::<F257>(&mut b, xb[3]).byte,
            ];
            let y_bytes = [
                alloc_byte::<F257>(&mut b, yb[0]).byte,
                alloc_byte::<F257>(&mut b, yb[1]).byte,
                alloc_byte::<F257>(&mut b, yb[2]).byte,
                alloc_byte::<F257>(&mut b, yb[3]).byte,
            ];
            let x_digits = u32_bytes_to_bal16_digits(&mut b, x_bytes);
            let y_digits = u32_bytes_to_bal16_digits(&mut b, y_bytes);
            assert_eq!(x_digits.len(), 9);
            assert_eq!(y_digits.len(), 9);

            let prod_digits = mul_bal16_9_by_9_u32ish(&mut b, &x_digits, &y_digits);

            let (inst, asg) = b.into_instance();
            inst.check(&asg).expect("u32ish 9x9 mul satisfiable");

            // Decode digits back to integer and compare (as i128 to avoid overflow).
            let mut acc: i128 = 0;
            let mut pow: i128 = 1;
            for &dv in &prod_digits {
                acc += (f257_to_i32_bal(asg[dv]) as i128) * pow;
                pow *= 16;
            }
            let expected: i128 = (x as i128) * (y as i128);
            assert_eq!(acc, expected);
        }
    }

    #[test]
    fn test_mul_u32ish9_to_fixed_bal16_roundtrip_small() {
        use rand::{Rng, SeedableRng};
        let mut rng = rand::rngs::StdRng::seed_from_u64(0xBEEF_900Du64);
        let mut b = Dr1csBuilder::<F257>::new();
        b.enforce_var_eq_const(b.one(), F257::ONE);

        for _ in 0..50 {
            let x: u32 = rng.gen();
            let y: u32 = rng.gen();

            // u32 -> bytes -> 9 digits (balanced)
            let xb = x.to_le_bytes();
            let yb = y.to_le_bytes();
            let x_vars = xb.map(|by| b.new_var(F257::from(by as u64)));
            let y_vars = yb.map(|by| b.new_var(F257::from(by as u64)));
            let x9 = u32_bytes_to_bal16_digits(&mut b, x_vars.to_vec().try_into().unwrap());
            let y9 = u32_bytes_to_bal16_digits(&mut b, y_vars.to_vec().try_into().unwrap());
            debug_assert_eq!(x9.len(), 9);
            debug_assert_eq!(y9.len(), 9);

            let prod = mul_u32ish9_to_fixed_bal16(&mut b, &x9, &y9, 18);
            assert_eq!(prod.len(), 18);

            // Decode.
            let mut acc: i128 = 0;
            let mut pow: i128 = 1;
            for &dv in &prod {
                acc += (f257_to_i32_bal(b.assignment[dv]) as i128) * pow;
                pow *= 16;
            }
            assert_eq!(acc, (x as i128) * (y as i128));
        }

        let (inst, asg) = b.into_instance();
        inst.check(&asg).expect("u32ish fixed mul satisfied");
    }

    #[test]
    fn test_centered_babybear_times_u32_challenge_roundtrip() {
        // Multiply a centered BabyBear scalar (from canonical bytes) by a bounded-u32 (bytes),
        // using the 9×9 balanced base-16 digit multiplier.
        use rand::Rng;
        let mut rng = ark_std::test_rng();
        let p: u32 = BABYBEAR_P_U32;
        let half: u32 = p / 2;

        for _ in 0..50 {
            // Sample canonical BabyBear element in [0,p).
            let x: u32 = rng.gen_range(0..p);
            // Sample an arbitrary u32 challenge value.
            let y: u32 = rng.gen();

            let mut b = Dr1csBuilder::<F257>::new();
            b.enforce_var_eq_const(b.one(), F257::ONE);

            let xb = x.to_le_bytes();
            let yb = y.to_le_bytes();
            let x_bytes = [
                alloc_byte::<F257>(&mut b, xb[0]).byte,
                alloc_byte::<F257>(&mut b, xb[1]).byte,
                alloc_byte::<F257>(&mut b, xb[2]).byte,
                alloc_byte::<F257>(&mut b, xb[3]).byte,
            ];
            let y_bytes = [
                alloc_byte::<F257>(&mut b, yb[0]).byte,
                alloc_byte::<F257>(&mut b, yb[1]).byte,
                alloc_byte::<F257>(&mut b, yb[2]).byte,
                alloc_byte::<F257>(&mut b, yb[3]).byte,
            ];

            let x_cent = babybear_centered_from_u32_byte_vars_with_modulus(&mut b, x_bytes, p);
            let y_digits = u32_bytes_to_bal16_digits(&mut b, y_bytes);

            assert_eq!(x_cent.centered_bal16_digits.len(), 9);
            assert_eq!(y_digits.len(), 9);

            let prod_digits = mul_bal16_9_by_9_u32ish(&mut b, &x_cent.centered_bal16_digits, &y_digits);

            let (inst, asg) = b.into_instance();
            inst.check(&asg).expect("centered bb * u32 mul satisfiable");

            // Decode product digits.
            let mut acc: i128 = 0;
            let mut pow: i128 = 1;
            for &dv in &prod_digits {
                acc += (f257_to_i32_bal(asg[dv]) as i128) * pow;
                pow *= 16;
            }
            let x_centered: i128 = if x <= half { x as i128 } else { (x as i128) - (p as i128) };
            let expected: i128 = x_centered * (y as i128);
            assert_eq!(acc, expected);
        }
    }

    #[test]
    fn test_poseidon_plus_cm_coin_surface_satisfies_and_matches_values() {
        use ark_crypto_primitives::sponge::{poseidon::PoseidonSponge, CryptographicSponge};

        let cfg = f257_poseidon_config();
        let mut sponge = PoseidonSponge::<F257>::new(&cfg);
        let ring_dim: usize = 64;

        // Build a small op schedule:
        // - absorb 1 elem
        // - squeeze short challenge digits (len=ring_dim), no absorb
        // - squeeze get_challenge digits (len=8), then absorb them
        let mut ops: Vec<PoseidonTraceOp<F257>> = Vec::new();
        let a = vec![F257::from(7u64)];
        sponge.absorb(&a);
        ops.push(PoseidonTraceOp::Absorb(a));

        let short_digits = sponge.squeeze_field_elements::<F257>(ring_dim);
        ops.push(PoseidonTraceOp::SqueezeField(short_digits.clone()));

        let chal_digits = sponge.squeeze_field_elements::<F257>(DIGITS_PER_TRY);
        ops.push(PoseidonTraceOp::SqueezeField(chal_digits.clone()));
        sponge.absorb(&chal_digits);
        ops.push(PoseidonTraceOp::Absorb(chal_digits.clone()));

        let (inst, asg, shorts, u32s) = build_poseidon_f257_with_cm_coin_surface_from_ops_with_wiring(
            None,
            &ops,
            ring_dim,
            1,
            1,
        )
        .expect("build_poseidon_f257_with_cm_coin_surface_from_ops_with_wiring");
        inst.check(&asg).expect("poseidon+cm coin surface satisfied");
        assert_eq!(shorts.len(), 1);
        assert_eq!(u32s.len(), 1);

        // Check short challenge coeffs match the spec `utils::short_challenge(128, ...)`.
        let u: u64 = 1u64 << (128 / ring_dim);
        assert_eq!(u, 4u64);
        let asg_u16 = |v: usize| -> u16 {
            let bytes = asg[v].into_bigint().to_bytes_le();
            (bytes.get(0).copied().unwrap_or(0) as u16)
                | ((bytes.get(1).copied().unwrap_or(0) as u16) << 8)
        };
        for i in 0..ring_dim {
            let du16 = asg_u16(shorts[0].digit_vars[i]);
            let by: u8 = if du16 == 256 { 0u8 } else { du16 as u8 };
            let low = (by as u64) & (u - 1);
            let half = u / 2;
            let coeff = ((low as i64) - (half as i64)).rem_euclid(257) as u64;
            let got = asg_u16(shorts[0].coeff_vars[i]) as u64;
            assert_eq!(got, coeff);
        }

        // Check bounded u32 matches byte-view packing from first 4 digits.
        let mut bs = [0u8; 4];
        for i in 0..4 {
            let du16 = chal_digits[i]
                .into_bigint()
                .to_bytes_le()
                .get(0)
                .copied()
                .unwrap_or(0) as u16;
            bs[i] = if du16 == 256 { 0u8 } else { du16 as u8 };
        }
        let exp_u32 = u32::from_le_bytes(bs);
        // Reconstruct from assignment (base-128 limbs).
        let mut acc: u64 = 0;
        for i in (0..LIMBS_U32).rev() {
            let di = asg[u32s[0].limbs[i]]
                .into_bigint()
                .to_bytes_le()
                .get(0)
                .copied()
                .unwrap_or(0) as u64;
            acc <<= LIMB_BITS;
            acc |= di & (LIMB_BASE_U64 - 1);
        }
        assert_eq!(acc as u32, exp_u32);
    }

    #[test]
    fn test_poseidon_plus_cm_coins_and_frog_rejection_satisfies() {
        use ark_crypto_primitives::sponge::{poseidon::PoseidonSponge, CryptographicSponge};

        let cfg = f257_poseidon_config();
        let mut sponge = PoseidonSponge::<F257>::new(&cfg);
        let ring_dim: usize = 64;

        // Schedule:
        // Absorb(1),
        // Short: SqueezeField(64),
        // u32:   SqueezeField(8) + Absorb(8),
        // frog candidates (tries=2): 2x [SqueezeField(8) + Absorb(8)]
        let mut ops: Vec<PoseidonTraceOp<F257>> = Vec::new();
        let a = vec![F257::from(9u64)];
        sponge.absorb(&a);
        ops.push(PoseidonTraceOp::Absorb(a));

        let _short_digits = sponge.squeeze_field_elements::<F257>(ring_dim);
        ops.push(PoseidonTraceOp::SqueezeField(_short_digits));

        let c0 = sponge.squeeze_field_elements::<F257>(DIGITS_PER_TRY);
        ops.push(PoseidonTraceOp::SqueezeField(c0.clone()));
        sponge.absorb(&c0);
        ops.push(PoseidonTraceOp::Absorb(c0));

        let tries: usize = 2;
        let cand0 = sponge.squeeze_field_elements::<F257>(DIGITS_PER_TRY);
        ops.push(PoseidonTraceOp::SqueezeField(cand0.clone()));
        sponge.absorb(&cand0);
        ops.push(PoseidonTraceOp::Absorb(cand0));
        let cand1 = sponge.squeeze_field_elements::<F257>(DIGITS_PER_TRY);
        ops.push(PoseidonTraceOp::SqueezeField(cand1.clone()));
        sponge.absorb(&cand1);
        ops.push(PoseidonTraceOp::Absorb(cand1));

        // SqueezeField op indices: 0=short, 1=u32, 2=cand0, 3=cand1
        let wiring = TinyCoinOpWiring {
            short_squeeze_ops: vec![0],
            u32_squeeze_ops: vec![1],
            frog_squeeze_ops: vec![2, 3],
        };

        let (inst, asg, shorts, u32s, frogs) =
            build_poseidon_f257_with_cm_coins_and_frog_rejection_from_ops_with_wiring(
                None, &ops, ring_dim, &wiring, 1, tries,
            )
            .expect("build_poseidon_f257_with_cm_coins_and_frog_rejection_from_ops_with_wiring");
        inst.check(&asg).expect("poseidon+cm coins+frog satisfied");
        assert_eq!(shorts.len(), 1);
        assert_eq!(u32s.len(), 1);
        assert_eq!(frogs.len(), 1);
        assert_eq!(frogs[0].tries, tries);
    }

    #[test]
    fn test_poseidon_plus_cm_coins_frog_and_first_digit_mul_surface_satisfies() {
        use ark_crypto_primitives::sponge::{poseidon::PoseidonSponge, CryptographicSponge};

        let cfg = f257_poseidon_config();
        let mut sponge = PoseidonSponge::<F257>::new(&cfg);
        let ring_dim: usize = 64;

        // Same schedule as the existing combined test, but we will also build the digit-mul surface.
        let mut ops: Vec<PoseidonTraceOp<F257>> = Vec::new();
        let a = vec![F257::from(9u64)];
        sponge.absorb(&a);
        ops.push(PoseidonTraceOp::Absorb(a));

        let _short_digits = sponge.squeeze_field_elements::<F257>(ring_dim);
        ops.push(PoseidonTraceOp::SqueezeField(_short_digits));

        let c0 = sponge.squeeze_field_elements::<F257>(DIGITS_PER_TRY);
        ops.push(PoseidonTraceOp::SqueezeField(c0.clone()));
        sponge.absorb(&c0);
        ops.push(PoseidonTraceOp::Absorb(c0));

        let tries: usize = 2;
        let cand0 = sponge.squeeze_field_elements::<F257>(DIGITS_PER_TRY);
        ops.push(PoseidonTraceOp::SqueezeField(cand0.clone()));
        sponge.absorb(&cand0);
        ops.push(PoseidonTraceOp::Absorb(cand0));
        let cand1 = sponge.squeeze_field_elements::<F257>(DIGITS_PER_TRY);
        ops.push(PoseidonTraceOp::SqueezeField(cand1.clone()));
        sponge.absorb(&cand1);
        ops.push(PoseidonTraceOp::Absorb(cand1));

        // SqueezeField op indices: 0=short, 1=u32, 2=cand0, 3=cand1
        let wiring = TinyCoinOpWiring {
            short_squeeze_ops: vec![0],
            u32_squeeze_ops: vec![1],
            frog_squeeze_ops: vec![2, 3],
        };

        let (inst, asg, shorts, u32s, frogs, mul_surface) =
            build_poseidon_f257_with_cm_coins_frog_and_first_digit_mul_from_ops_with_wiring(
                None, &ops, ring_dim, &wiring, 1, tries, 0, 0,
            )
            .expect("build poseidon+cm+frog+mul");
        inst.check(&asg).expect("poseidon+cm+frog+mul satisfied");
        assert_eq!(shorts.len(), 1);
        assert_eq!(u32s.len(), 1);
        assert_eq!(frogs.len(), 1);
        assert_eq!(mul_surface.short_block_idx, 0);
        assert_eq!(mul_surface.u32_idx, 0);
        assert_eq!(mul_surface.products.len(), ring_dim);
        assert_eq!(mul_surface.products13.len(), ring_dim);
        assert_eq!(mul_surface.sum_digits.len(), 16);
        assert_eq!(mul_surface.sum_all_pairs_digits.len(), 16);
        assert_eq!(mul_surface.sum_all_pairs_coeffwise.len(), ring_dim);
        assert_eq!(mul_surface.sum_all_pairs_coeffwise[0].len(), 16);
    }

    #[test]
    fn test_poseidon_plus_cm_coins_frog_and_two_digit_mul_surfaces_satisfies() {
        use ark_crypto_primitives::sponge::{poseidon::PoseidonSponge, CryptographicSponge};

        let cfg = f257_poseidon_config();
        let mut sponge = PoseidonSponge::<F257>::new(&cfg);
        let ring_dim: usize = 64;

        // Schedule:
        // Absorb(1),
        // Short0: SqueezeField(64),
        // Short1: SqueezeField(64),
        // u32_0:  SqueezeField(8)+Absorb(8),
        // u32_1:  SqueezeField(8)+Absorb(8),
        // frog candidates (tries=2): 2x [SqueezeField(8)+Absorb(8)]
        let mut ops: Vec<PoseidonTraceOp<F257>> = Vec::new();
        let a = vec![F257::from(3u64)];
        sponge.absorb(&a);
        ops.push(PoseidonTraceOp::Absorb(a));

        let s0 = sponge.squeeze_field_elements::<F257>(ring_dim);
        ops.push(PoseidonTraceOp::SqueezeField(s0));
        let s1 = sponge.squeeze_field_elements::<F257>(ring_dim);
        ops.push(PoseidonTraceOp::SqueezeField(s1));

        let u0 = sponge.squeeze_field_elements::<F257>(DIGITS_PER_TRY);
        ops.push(PoseidonTraceOp::SqueezeField(u0.clone()));
        sponge.absorb(&u0);
        ops.push(PoseidonTraceOp::Absorb(u0));

        let u1 = sponge.squeeze_field_elements::<F257>(DIGITS_PER_TRY);
        ops.push(PoseidonTraceOp::SqueezeField(u1.clone()));
        sponge.absorb(&u1);
        ops.push(PoseidonTraceOp::Absorb(u1));

        let tries: usize = 2;
        for _ in 0..(1 * tries) {
            let v = sponge.squeeze_field_elements::<F257>(DIGITS_PER_TRY);
            ops.push(PoseidonTraceOp::SqueezeField(v.clone()));
            sponge.absorb(&v);
            ops.push(PoseidonTraceOp::Absorb(v));
        }

        // SqueezeField op indices: 0=s0, 1=s1, 2=u0, 3=u1, 4=cand0, 5=cand1
        let wiring = TinyCoinOpWiring {
            short_squeeze_ops: vec![0, 1],
            u32_squeeze_ops: vec![2, 3],
            frog_squeeze_ops: vec![4, 5],
        };

        let pairs = vec![(0usize, 0usize), (1usize, 1usize)];
        let (inst, asg, shorts, u32s, frogs, surfaces) =
            build_poseidon_f257_with_cm_coins_frog_and_digit_mul_surfaces_from_ops_with_wiring(
                None, &ops, ring_dim, &wiring, 1, tries, &pairs,
            )
            .expect("build poseidon+cm+frog+mul batch");
        inst.check(&asg).expect("poseidon+cm+frog+mul batch satisfied");
        assert_eq!(shorts.len(), 2);
        assert_eq!(u32s.len(), 2);
        assert_eq!(frogs.len(), 1);
        assert_eq!(surfaces.len(), 2);
        for s in &surfaces {
            assert_eq!(s.products.len(), ring_dim);
            assert_eq!(s.products13.len(), ring_dim);
            assert_eq!(s.sum_digits.len(), 16);
            assert_eq!(s.sum_all_pairs_digits.len(), 16);
            assert_eq!(s.sum_all_pairs_coeffwise.len(), ring_dim);
            assert_eq!(s.sum_all_pairs_coeffwise[0].len(), 16);
        }
    }

    #[test]
    fn test_sum_all_pairs_digits_matches_sum_of_surface_sums() {
        use ark_crypto_primitives::sponge::{poseidon::PoseidonSponge, CryptographicSponge};

        let cfg = f257_poseidon_config();
        let mut sponge = PoseidonSponge::<F257>::new(&cfg);
        let ring_dim: usize = 64;

        let mut ops: Vec<PoseidonTraceOp<F257>> = Vec::new();
        let a = vec![F257::from(11u64)];
        sponge.absorb(&a);
        ops.push(PoseidonTraceOp::Absorb(a));

        // Two short blocks and two u32 blocks.
        for _ in 0..2 {
            let short = sponge.squeeze_field_elements::<F257>(ring_dim);
            ops.push(PoseidonTraceOp::SqueezeField(short.clone()));
        }
        for _ in 0..2 {
            let u0 = sponge.squeeze_field_elements::<F257>(DIGITS_PER_TRY);
            ops.push(PoseidonTraceOp::SqueezeField(u0.clone()));
            sponge.absorb(&u0);
            ops.push(PoseidonTraceOp::Absorb(u0.clone()));
        }

        // Frog candidates (tries=1, n_coins=1) just to satisfy API.
        let tries: usize = 1;
        let cand = sponge.squeeze_field_elements::<F257>(DIGITS_PER_TRY);
        ops.push(PoseidonTraceOp::SqueezeField(cand.clone()));
        sponge.absorb(&cand);
        ops.push(PoseidonTraceOp::Absorb(cand));

        // SqueezeField op indices: 0,1 shorts; 2,3 u32s; 4 frog cand
        let wiring = TinyCoinOpWiring {
            short_squeeze_ops: vec![0, 1],
            u32_squeeze_ops: vec![2, 3],
            frog_squeeze_ops: vec![4],
        };

        let pairs = vec![(0usize, 0usize), (1usize, 1usize)];
        let (_inst, asg, _shorts, _u32s, _frogs, surfaces) =
            build_poseidon_f257_with_cm_coins_frog_and_digit_mul_surfaces_from_ops_with_wiring(
                None, &ops, ring_dim, &wiring, 1, tries, &pairs,
            )
            .expect("build");

        let decode16 = |digits: &[usize]| -> i128 {
            let mut acc: i128 = 0;
            let mut pow: i128 = 1;
            for &dv in digits {
                acc += (f257_to_i32_bal(asg[dv]) as i128) * pow;
                pow *= 16;
            }
            acc
        };

        let s0 = decode16(&surfaces[0].sum_digits);
        let s1 = decode16(&surfaces[1].sum_digits);
        let all0 = decode16(&surfaces[0].sum_all_pairs_digits);
        let all1 = decode16(&surfaces[1].sum_all_pairs_digits);
        assert_eq!(all0, s0 + s1);
        assert_eq!(all1, s0 + s1);
    }

    #[test]
    fn test_sum_all_pairs_coeffwise_matches_sum_of_products() {
        use ark_crypto_primitives::sponge::{poseidon::PoseidonSponge, CryptographicSponge};

        let cfg = f257_poseidon_config();
        let mut sponge = PoseidonSponge::<F257>::new(&cfg);
        let ring_dim: usize = 32; // keep coeffs small (u=16) and test fast

        let mut ops: Vec<PoseidonTraceOp<F257>> = Vec::new();
        let a = vec![F257::from(11u64)];
        sponge.absorb(&a);
        ops.push(PoseidonTraceOp::Absorb(a));

        // Two short blocks and two u32 blocks.
        for _ in 0..2 {
            let short = sponge.squeeze_field_elements::<F257>(ring_dim);
            ops.push(PoseidonTraceOp::SqueezeField(short.clone()));
        }
        for _ in 0..2 {
            let u0 = sponge.squeeze_field_elements::<F257>(DIGITS_PER_TRY);
            ops.push(PoseidonTraceOp::SqueezeField(u0.clone()));
            sponge.absorb(&u0);
            ops.push(PoseidonTraceOp::Absorb(u0.clone()));
        }

        // Frog candidates (tries=1, n_coins=1) just to satisfy API.
        let tries: usize = 1;
        let cand = sponge.squeeze_field_elements::<F257>(DIGITS_PER_TRY);
        ops.push(PoseidonTraceOp::SqueezeField(cand.clone()));
        sponge.absorb(&cand);
        ops.push(PoseidonTraceOp::Absorb(cand));

        // SqueezeField op indices: 0,1 shorts; 2,3 u32s; 4 frog cand
        let wiring = TinyCoinOpWiring {
            short_squeeze_ops: vec![0, 1],
            u32_squeeze_ops: vec![2, 3],
            frog_squeeze_ops: vec![4],
        };

        let pairs = vec![(0usize, 0usize), (1usize, 1usize)];
        let (_inst, asg, _shorts, _u32s, _frogs, surfaces) =
            build_poseidon_f257_with_cm_coins_frog_and_digit_mul_surfaces_from_ops_with_wiring(
                None, &ops, ring_dim, &wiring, 1, tries, &pairs,
            )
            .expect("build");

        let decode = |digits: &[usize]| -> i128 {
            let mut acc: i128 = 0;
            let mut pow: i128 = 1;
            for &dv in digits {
                acc += (f257_to_i32_bal(asg[dv]) as i128) * pow;
                pow *= 16;
            }
            acc
        };

        // Check a few coefficient indices.
        for &i in &[0usize, 1, 7, 13, ring_dim - 1] {
            let p0 = decode(&surfaces[0].products13[i]);
            let p1 = decode(&surfaces[1].products13[i]);
            let acci = decode(&surfaces[0].sum_all_pairs_coeffwise[i]);
            assert_eq!(acci, p0 + p1);
        }
    }

    #[test]
    fn test_digit_mul_sq_surface_sum_matches_values_ring64() {
        use ark_crypto_primitives::sponge::{poseidon::PoseidonSponge, CryptographicSponge};

        let cfg = f257_poseidon_config();
        let mut sponge = PoseidonSponge::<F257>::new(&cfg);
        let ring_dim: usize = 64;

        let mut ops: Vec<PoseidonTraceOp<F257>> = Vec::new();
        let a = vec![F257::from(11u64)];
        sponge.absorb(&a);
        ops.push(PoseidonTraceOp::Absorb(a));

        let short = sponge.squeeze_field_elements::<F257>(ring_dim);
        ops.push(PoseidonTraceOp::SqueezeField(short.clone()));
        let u0 = sponge.squeeze_field_elements::<F257>(DIGITS_PER_TRY);
        ops.push(PoseidonTraceOp::SqueezeField(u0.clone()));
        sponge.absorb(&u0);
        ops.push(PoseidonTraceOp::Absorb(u0.clone()));

        // Frog candidates (tries=1, n_coins=1) just to satisfy API.
        let tries: usize = 1;
        let cand = sponge.squeeze_field_elements::<F257>(DIGITS_PER_TRY);
        ops.push(PoseidonTraceOp::SqueezeField(cand.clone()));
        sponge.absorb(&cand);
        ops.push(PoseidonTraceOp::Absorb(cand));

        let wiring = TinyCoinOpWiring {
            short_squeeze_ops: vec![0],
            u32_squeeze_ops: vec![1],
            frog_squeeze_ops: vec![2],
        };
        let pairs = vec![(0usize, 0usize)];

        let (inst, asg, _shorts, _u32s, _frogs, surfaces) =
            build_poseidon_f257_with_cm_coins_frog_and_digit_mul_sq_surfaces_from_ops_with_wiring(
                None, &ops, ring_dim, &wiring, 1, tries, &pairs,
            )
            .expect("build sq surfaces");
        inst.check(&asg).expect("sq surfaces satisfied");

        // Decode u32 from first 4 digits (byte-view 256->0), then square.
        let mut ubytes = [0u8; 4];
        for i in 0..4 {
            let du16 = u0[i]
                .into_bigint()
                .to_bytes_le()
                .get(0)
                .copied()
                .unwrap_or(0) as u16;
            ubytes[i] = if du16 == 256 { 0 } else { du16 as u8 };
        }
        let u32v: i128 = u32::from_le_bytes(ubytes) as i128;
        let u32sq: i128 = u32v * u32v;

        // Decode expected coeffs from `short_challenge(128)` semantics (u=4 for ring_dim=64).
        let u = 1u64 << (128 / ring_dim);
        let half = (u / 2) as i64;
        let mut expected: i128 = 0;
        for i in 0..ring_dim {
            let du16 = short[i]
                .into_bigint()
                .to_bytes_le()
                .get(0)
                .copied()
                .unwrap_or(0) as u16;
            let by: u8 = if du16 == 256 { 0 } else { du16 as u8 };
            let low = (by as u64) & (u - 1);
            let coeff = (low as i64) - half;
            expected += (coeff as i128) * u32sq;
        }

        // Decode computed sum_digits (len 24).
        let sum_digits = &surfaces[0].sum_digits;
        assert_eq!(sum_digits.len(), 24);
        let mut acc: i128 = 0;
        let mut pow: i128 = 1;
        for &dv in sum_digits {
            acc += (f257_to_i32_bal(asg[dv]) as i128) * pow;
            pow *= 16;
        }
        assert_eq!(acc, expected);
    }

    #[test]
    fn test_digit_mul_surface_sum_matches_values_ring64() {
        // Use ring_dim=64 (the production CM setting) so short coeffs are tiny (u=4),
        // and compare the digit-sum against native arithmetic in i128.
        use ark_crypto_primitives::sponge::{poseidon::PoseidonSponge, CryptographicSponge};

        let cfg = f257_poseidon_config();
        let mut sponge = PoseidonSponge::<F257>::new(&cfg);
        let ring_dim: usize = 64;

        let mut ops: Vec<PoseidonTraceOp<F257>> = Vec::new();
        let a = vec![F257::from(11u64)];
        sponge.absorb(&a);
        ops.push(PoseidonTraceOp::Absorb(a));

        let short = sponge.squeeze_field_elements::<F257>(ring_dim);
        ops.push(PoseidonTraceOp::SqueezeField(short.clone()));
        let u0 = sponge.squeeze_field_elements::<F257>(DIGITS_PER_TRY);
        ops.push(PoseidonTraceOp::SqueezeField(u0.clone()));
        sponge.absorb(&u0);
        ops.push(PoseidonTraceOp::Absorb(u0.clone()));

        // Frog candidates (tries=1, n_coins=1) just to satisfy API.
        let tries: usize = 1;
        let cand = sponge.squeeze_field_elements::<F257>(DIGITS_PER_TRY);
        ops.push(PoseidonTraceOp::SqueezeField(cand.clone()));
        sponge.absorb(&cand);
        ops.push(PoseidonTraceOp::Absorb(cand));

        // SqueezeField op indices: 0=short, 1=u32, 2=cand
        let wiring = TinyCoinOpWiring {
            short_squeeze_ops: vec![0],
            u32_squeeze_ops: vec![1],
            frog_squeeze_ops: vec![2],
        };

        let pairs = vec![(0usize, 0usize)];
        let (inst, asg, shorts, u32s, _frogs, surfaces) =
            build_poseidon_f257_with_cm_coins_frog_and_digit_mul_surfaces_from_ops_with_wiring(
                None, &ops, ring_dim, &wiring, 1, tries, &pairs,
            )
            .expect("build poseidon+mul+sum");
        inst.check(&asg).expect("poseidon+mul+sum satisfied");
        assert_eq!(shorts.len(), 1);
        assert_eq!(u32s.len(), 1);
        assert_eq!(surfaces.len(), 1);

        // Decode expected sum from the digits the sponge produced.
        let mut ubytes = [0u8; 4];
        for i in 0..4 {
            let du16 = u0[i]
                .into_bigint()
                .to_bytes_le()
                .get(0)
                .copied()
                .unwrap_or(0) as u16;
            ubytes[i] = if du16 == 256 { 0 } else { du16 as u8 };
        }
        let u32v: i128 = u32::from_le_bytes(ubytes) as i128;
        let u = 1u64 << (128 / ring_dim);
        let half = (u / 2) as i64;

        let mut expected: i128 = 0;
        for i in 0..ring_dim {
            let du16 = short[i]
                .into_bigint()
                .to_bytes_le()
                .get(0)
                .copied()
                .unwrap_or(0) as u16;
            let by: u8 = if du16 == 256 { 0 } else { du16 as u8 };
            let low = (by as u64) & (u - 1);
            let coeff = (low as i64) - half; // in [-half, half-1]
            expected += (coeff as i128) * u32v;
        }

        // Decode computed sum digits (base-16).
        let sum_digits = &surfaces[0].sum_digits;
        let mut acc: i128 = 0;
        let mut pow: i128 = 1;
        for &dv in sum_digits {
            acc += (f257_to_i32_bal(asg[dv]) as i128) * pow;
            pow *= 16;
        }
        assert_eq!(acc, expected);
    }

    #[test]
    fn test_rebalance_prod12_to_prod13_decodes_same() {
        use rand::{Rng, SeedableRng};
        let mut rng = rand::rngs::StdRng::seed_from_u64(0x5EED_13u64);
        let mut b = Dr1csBuilder::<F257>::new();
        b.enforce_var_eq_const(b.one(), F257::ONE);

        // Random coeff in [-128,127] and random scalar in a u32-ish envelope.
        let coeff: i128 = rng.gen_range(-128i128..=127i128);
        let x: i128 = rng.gen_range(-(1i128 << 32)..(1i128 << 32));

        // coeff -> 3 balanced digits
        let mut cur = coeff;
        let mut c3 = [0usize; 3];
        for i in 0..3 {
            let mut r = ((cur % 16) + 16) % 16;
            if r >= 8 { r -= 16; }
            c3[i] = alloc_bal16_digit(&mut b, r as i8);
            cur = (cur - r) / 16;
        }
        debug_assert_eq!(cur, 0);

        // x -> 9 balanced digits
        let x_digits = {
            let mut xx = x;
            let mut out: Vec<usize> = Vec::with_capacity(9);
            for _ in 0..9 {
                let mut r = ((xx % 16) + 16) % 16;
                if r >= 8 { r -= 16; }
                out.push(alloc_bal16_digit(&mut b, r as i8));
                xx = (xx - r) / 16;
            }
            out
        };

        let p12 = mul_bal16_3_by_digits9(&mut b, &c3, &x_digits);
        let p13 = rebalance_prod12_to_prod13(&mut b, &p12);

        let decode = |digits: &[usize]| -> i128 {
            let mut acc: i128 = 0;
            let mut pow: i128 = 1;
            for &dv in digits {
                acc += (f257_to_i32_bal(b.assignment[dv]) as i128) * pow;
                pow *= 16;
            }
            acc
        };
        assert_eq!(decode(&p12), decode(&p13));
        assert_eq!(decode(&p13), coeff * x);

        let (inst, asg) = b.into_instance();
        inst.check(&asg).expect("rebalance prod satisfied");
    }

    #[test]
    fn test_infer_cm_coin_op_wiring_from_ops_smoke() {
        use ark_crypto_primitives::sponge::{poseidon::PoseidonSponge, CryptographicSponge};

        let cfg = f257_poseidon_config();
        let mut sponge = PoseidonSponge::<F257>::new(&cfg);
        let ring_dim: usize = 64;
        let k: usize = 1;
        let log_kappa: usize = 1;
        let nvars_cm: usize = 2;
        let tries: usize = 2;
        let n_coins: usize = 1;
        let frog_need = n_coins * tries;

        let short_need = cm_short_challenge_blocks(ring_dim, k);
        let u32_need = cm_bounded_u32_challenges(log_kappa, nvars_cm);

        let mut ops: Vec<PoseidonTraceOp<F257>> = Vec::new();
        // absorb something
        let a = vec![F257::from(1u64)];
        sponge.absorb(&a);
        ops.push(PoseidonTraceOp::Absorb(a));

        // short blocks
        for _ in 0..short_need {
            let v = sponge.squeeze_field_elements::<F257>(ring_dim);
            ops.push(PoseidonTraceOp::SqueezeField(v));
        }
        // u32 challenges: squeeze(8) + absorb(8)
        for _ in 0..u32_need {
            let v = sponge.squeeze_field_elements::<F257>(DIGITS_PER_TRY);
            ops.push(PoseidonTraceOp::SqueezeField(v.clone()));
            sponge.absorb(&v);
            ops.push(PoseidonTraceOp::Absorb(v));
        }
        // frog candidates (same squeeze+absorb shape)
        for _ in 0..frog_need {
            let v = sponge.squeeze_field_elements::<F257>(DIGITS_PER_TRY);
            ops.push(PoseidonTraceOp::SqueezeField(v.clone()));
            sponge.absorb(&v);
            ops.push(PoseidonTraceOp::Absorb(v));
        }

        let w = infer_cm_coin_op_wiring_from_ops(
            &ops,
            ring_dim,
            k,
            log_kappa,
            nvars_cm,
            0,
            frog_need,
        )
        .expect("infer_cm_coin_op_wiring_from_ops");
        assert_eq!(w.short_squeeze_ops.len(), short_need);
        assert_eq!(w.u32_squeeze_ops.len(), u32_need);
        assert_eq!(w.frog_squeeze_ops.len(), frog_need);

        // Now build the combined surface using inferred wiring.
        let (inst, asg, shorts, u32s, frogs) =
            build_poseidon_f257_with_cm_coins_and_frog_rejection_from_ops_with_wiring(
                None, &ops, ring_dim, &w, n_coins, tries,
            )
            .expect("build combined cm+frog");
        inst.check(&asg).expect("combined cm+frog satisfies");
        assert_eq!(shorts.len(), short_need);
        assert_eq!(u32s.len(), u32_need);
        assert_eq!(frogs.len(), n_coins);
    }

    #[test]
    fn test_lift_recording_trace_ops_to_f257_roundtrip_small() {
        use stark_rings::cyclotomic_ring::models::goldilocks::RqPoly as R;
        use latticefold::transcript::Transcript;
        use stark_rings::PolyRing;
        use stark_rings::Ring;
        use ark_ff::Field;

        // Record a tiny trace in the LF+ recording transcript (base ring is large, but values are digits/bytes).
        let mut rec = crate::recording_transcript::TracePoseidonTranscript::<R>::empty::<()>();
        rec.absorb(&R::ONE);
        let _ = rec.squeeze_bytes(17);
        let _c = rec.get_challenge();
        let tr = rec.trace().clone();

        // Lift ops to F257 and ensure lengths line up.
        type BF = <<R as PolyRing>::BaseRing as Field>::BasePrimeField;
        let ops_f257 =
            lift_recording_trace_ops_to_f257::<BF>(&tr.ops).expect("lift_recording_trace_ops_to_f257");
        assert_eq!(ops_f257.len(), tr.ops.len());

        // Check that every absorbed/squeezed element is in 0..=256.
        for op in ops_f257 {
            match op {
                PoseidonTraceOp::Absorb(v) | PoseidonTraceOp::SqueezeField(v) => {
                    for e in v {
                        let du16 = e.into_bigint().to_bytes_le().get(0).copied().unwrap_or(0) as u16;
                        assert!(du16 < 257);
                    }
                }
                PoseidonTraceOp::SqueezeBytes { .. } => {}
            }
        }
    }

    #[test]
    fn test_mul_bal16_small_3_by_u32_bal_roundtrip() {
        // Multiply a 12-bit-ish value (3 base-16 digits) by a balanced u32 (9 digits),
        // and check the digit decomposition matches native integer multiplication.
        fn to_bal16_u32(mut x: u32) -> Vec<i8> {
            // Start from base-16 digits (0..15), then balance with carry so each digit in [-8,7].
            let mut digs: Vec<i8> = Vec::with_capacity(9);
            let mut carry: i32 = 0;
            for _ in 0..8 {
                let d = (x & 0xF) as i32;
                x >>= 4;
                let mut t = d + carry;
                if t >= 8 {
                    t -= 16;
                    carry = 1;
                } else {
                    carry = 0;
                }
                digs.push(t as i8);
            }
            digs.push(carry as i8);
            digs
        }
        fn from_bal16(digs: &[i8]) -> i128 {
            let mut acc: i128 = 0;
            let mut pow: i128 = 1;
            for &d in digs {
                acc += (d as i128) * pow;
                pow *= 16;
            }
            acc
        }

        let a_u16: u16 = 0x0777; // all nibbles < 8 => already balanced (3 digits)
        let b_u32: u32 = 0xffff_fffe;
        let a_i = a_u16 as i128;
        let b_bal = to_bal16_u32(b_u32);
        let b_i = from_bal16(&b_bal);
        assert_eq!(b_i as u64, b_u32 as u64);
        let prod_i = a_i * b_i;

        let mut b = Dr1csBuilder::<F257>::new();
        b.enforce_var_eq_const(b.one(), F257::ONE);

        // a in balanced base-16 (3 digits) from standard nibbles (no balancing needed due to a<2048).
        let a0 = alloc_bal16_digit(&mut b, (a_u16 & 0xF) as i8);
        let a1 = alloc_bal16_digit(&mut b, ((a_u16 >> 4) & 0xF) as i8);
        let a2 = alloc_bal16_digit(&mut b, ((a_u16 >> 8) & 0xF) as i8);
        let a_digits = vec![a0, a1, a2];

        // b in balanced base-16 (9 digits).
        let mut b_digits: Vec<usize> = Vec::with_capacity(9);
        for &d in &b_bal {
            b_digits.push(alloc_bal16_digit(&mut b, d));
        }

        let out = mul_bal16_small(&mut b, &a_digits, &b_digits);

        // Check satisfiable.
        let (inst, asg) = b.into_instance();
        inst.check(&asg).expect("mul_bal16_small satisfiable");

        // Decode output digits (balanced digits + final carry var) and compare.
        let out_digits_i32: Vec<i32> = out.iter().map(|&v| f257_to_i32_bal(asg[v])).collect();
        let mut out_i: i128 = 0;
        let mut pow: i128 = 1;
        for &di in &out_digits_i32 {
            out_i += (di as i128) * pow;
            pow *= 16;
        }

        // Compute expected base-16 digits from prod_i (standard base-16, then balance with carry).
        let mut exp: Vec<i32> = Vec::new();
        let mut t = prod_i;
        for _ in 0..12 {
            exp.push((t & 0xF) as i32);
            t >>= 4;
        }
        // balance
        let mut carry: i32 = 0;
        for d in exp.iter_mut() {
            let mut v = *d + carry;
            if v >= 8 {
                v -= 16;
                carry = 1;
            } else {
                carry = 0;
            }
            *d = v;
        }
        // If there's a final carry, fold it into the last digit (fits our scenario).
        if carry != 0 {
            exp.push(carry);
        }

        assert_eq!(out_i, prod_i, "decoded integer mismatch");
        // Compare low 12 digits; our `out` is always 12 digits.
        assert_eq!(
            out_digits_i32,
            exp[..out_digits_i32.len()].to_vec(),
            "digit mismatch: out={:?} exp={:?}",
            out_digits_i32,
            exp
        );
    }

    #[test]
    fn test_u32_bytes_to_bal16_digits_roundtrip() {
        let x: u32 = 0xffff_fffe;
        let mut b = Dr1csBuilder::<F257>::new();
        b.enforce_var_eq_const(b.one(), F257::ONE);

        // Allocate bytes as ByteVars, then pass just the byte vars.
        let bytes = x.to_le_bytes();
        let b0 = alloc_byte::<F257>(&mut b, bytes[0]);
        let b1 = alloc_byte::<F257>(&mut b, bytes[1]);
        let b2 = alloc_byte::<F257>(&mut b, bytes[2]);
        let b3 = alloc_byte::<F257>(&mut b, bytes[3]);

        let digs = u32_bytes_to_bal16_digits(&mut b, [b0.byte, b1.byte, b2.byte, b3.byte]);

        let (inst, asg) = b.into_instance();
        inst.check(&asg).expect("u32_bytes_to_bal16_digits satisfiable");

        let mut acc: i128 = 0;
        let mut pow: i128 = 1;
        for &v in &digs {
            let di = f257_to_i32_bal(asg[v]) as i128;
            acc += di * pow;
            pow *= 16;
        }
        assert_eq!(acc as u64, x as u64);
    }

    #[test]
    fn test_add_bal16_same_len_roundtrip() {
        // Check that balanced-base16 addition matches integer addition.
        fn decode(asg: &[F257], digs: &[usize]) -> i128 {
            let mut acc: i128 = 0;
            let mut pow: i128 = 1;
            for &v in digs {
                acc += (super::f257_to_i32_bal(asg[v]) as i128) * pow;
                pow *= 16;
            }
            acc
        }

        let mut b = Dr1csBuilder::<F257>::new();
        b.enforce_var_eq_const(b.one(), F257::ONE);

        // a = 0x0777 (3 digits), b = 0x0001 (3 digits)
        let a = vec![
            alloc_bal16_digit(&mut b, 0x7),
            alloc_bal16_digit(&mut b, 0x7),
            alloc_bal16_digit(&mut b, 0x7),
        ];
        let c = vec![
            alloc_bal16_digit(&mut b, 1),
            alloc_bal16_digit(&mut b, 0),
            alloc_bal16_digit(&mut b, 0),
        ];

        let (sum, carry) = add_bal16_same_len(&mut b, &a, &c);
        let (inst, asg) = b.into_instance();
        inst.check(&asg).expect("add_bal16_same_len satisfiable");

        let a_i = decode(&asg, &a);
        let c_i = decode(&asg, &c);
        let sum_i = decode(&asg, &sum) + (super::f257_to_i32_bal(asg[carry]) as i128) * 16i128.pow(sum.len() as u32);
        assert_eq!(sum_i, a_i + c_i);
    }

    #[test]
    fn test_neg_and_sub_bal16_roundtrip() {
        // Check negation and subtraction decode correctly for random small-ish values.
        use rand::Rng;
        let mut rng = ark_std::test_rng();

        for _ in 0..200 {
            let x: i64 = rng.gen_range(-(1i64 << 31)..(1i64 << 31));
            let y: i64 = rng.gen_range(-(1i64 << 31)..(1i64 << 31));

            // Encode into 9 balanced digits (enough for 32-bit-ish values).
            fn to_bal16(mut v: i64) -> [i8; 9] {
                let mut out = [0i8; 9];
                for i in 0..9 {
                    let mut d = (v % 16) as i32;
                    if d > 7 { d -= 16; }
                    if d < -8 { d += 16; }
                    out[i] = d as i8;
                    v = (v - d as i64) / 16;
                }
                out
            }
            let xd = to_bal16(x);
            let yd = to_bal16(y);

            let mut b = Dr1csBuilder::<F257>::new();
            b.enforce_var_eq_const(b.one(), F257::ONE);
            let xvars: Vec<usize> = xd.iter().map(|&d| alloc_bal16_digit(&mut b, d)).collect();
            let yvars: Vec<usize> = yd.iter().map(|&d| alloc_bal16_digit(&mut b, d)).collect();

            let (nx, c0) = neg_bal16_digits(&mut b, &xvars);
            let (diff, c1) = sub_bal16_same_len(&mut b, &xvars, &yvars);

            // For fixed-width 9-digit inputs, enforce no overflow.
            b.enforce_var_eq_const(c0, F257::ZERO);
            b.enforce_var_eq_const(c1, F257::ZERO);

            let (inst, asg) = b.into_instance();
            inst.check(&asg).expect("neg/sub satisfiable");

            let decode = |ds: &[usize]| -> i128 {
                let mut acc: i128 = 0;
                let mut pow: i128 = 1;
                for &v in ds {
                    acc += (f257_to_i32_bal(asg[v]) as i128) * pow;
                    pow *= 16;
                }
                acc
            };
            assert_eq!(decode(&nx), -(x as i128));
            assert_eq!(decode(&diff), (x as i128) - (y as i128));
        }
    }

    #[test]
    fn test_mul_bal16_long_by_u32ish9_roundtrip() {
        // Multiply a moderately-sized integer by a u32-ish integer via chunking.
        // (Keep decoded values within i128 to avoid overflow in the test.)
        use rand::Rng;
        let mut rng = ark_std::test_rng();

        for _ in 0..50 {
            // Build a random ~48-bit signed integer.
            let mag: i128 = (rng.gen::<u64>() & ((1u64 << 48) - 1)) as i128;
            let sign: i128 = if rng.gen::<bool>() { 1 } else { -1 };
            let a: i128 = sign * mag;
            let b_u32: u32 = rng.gen();
            let b_i: i128 = b_u32 as i128;

            // Encode a into balanced base-16 digits (len 16 is enough for ~64 bits).
            let mut a_tmp = a;
            let mut a_digs: Vec<i8> = Vec::with_capacity(16);
            for _ in 0..16 {
                let mut d = (a_tmp % 16) as i32;
                if d > 7 { d -= 16; }
                if d < -8 { d += 16; }
                a_digs.push(d as i8);
                a_tmp = (a_tmp - d as i128) / 16;
            }
            assert_eq!(a_tmp, 0);

            let mut b = Dr1csBuilder::<F257>::new();
            b.enforce_var_eq_const(b.one(), F257::ONE);
            let a_vars: Vec<usize> = a_digs.iter().map(|&d| alloc_bal16_digit(&mut b, d)).collect();
            let bb = b_u32.to_le_bytes();
            let b_bytes = [
                alloc_byte::<F257>(&mut b, bb[0]).byte,
                alloc_byte::<F257>(&mut b, bb[1]).byte,
                alloc_byte::<F257>(&mut b, bb[2]).byte,
                alloc_byte::<F257>(&mut b, bb[3]).byte,
            ];
            let b9 = u32_bytes_to_bal16_digits(&mut b, b_bytes);
            let prod = mul_bal16_long_by_u32ish9(&mut b, &a_vars, &b9);

            let (inst, asg) = b.into_instance();
            inst.check(&asg).expect("long*u32ish satisfiable");

            let mut acc: i128 = 0;
            let mut pow: i128 = 1;
            for (idx, &dv) in prod.iter().enumerate() {
                acc += (f257_to_i32_bal(asg[dv]) as i128) * pow;
                if idx + 1 < prod.len() {
                    pow *= 16;
                }
            }
            assert_eq!(acc, a * b_i);
        }
    }

    #[test]
    fn test_mul_bal16_long_by_long_roundtrip_small() {
        use rand::{Rng, SeedableRng};
        let mut rng = rand::rngs::StdRng::seed_from_u64(0xA11CE5EED_u64);
        let mut b = Dr1csBuilder::<F257>::new();
        b.enforce_var_eq_const(b.one(), F257::ONE);

        let to_digits = |b: &mut Dr1csBuilder<F257>, mut x: i128, max_len: usize| -> Vec<usize> {
            let zero = alloc_bal16_digit(b, 0);
            if x == 0 {
                return vec![zero];
            }
            let mut out: Vec<usize> = Vec::new();
            for _ in 0..max_len {
                if x == 0 {
                    break;
                }
                // Balanced remainder in [-8,7].
                let mut r = (x % 16) as i32;
                if r < 0 {
                    r += 16;
                }
                if r >= 8 {
                    r -= 16;
                }
                out.push(alloc_bal16_digit(b, r as i8));
                x = (x - (r as i128)) / 16;
            }
            out
        };

        // Keep sizes modest so native product fits in i128 comfortably.
        for _ in 0..50 {
            let a: i128 = rng.gen_range(-(1i128 << 40)..(1i128 << 40));
            let c: i128 = rng.gen_range(-(1i128 << 40)..(1i128 << 40));
            let ad = to_digits(&mut b, a, 20);
            let cd = to_digits(&mut b, c, 20);
            let prod = mul_bal16_long_by_long(&mut b, &ad, &cd);

            // Decode result.
            let mut acc: i128 = 0;
            let mut pow: i128 = 1;
            for &dv in &prod {
                acc += (f257_to_i32_bal(b.assignment[dv]) as i128) * pow;
                pow *= 16;
            }
            assert_eq!(acc, a * c);
        }

        let (inst, asg) = b.into_instance();
        inst.check(&asg).expect("mul_bal16_long_by_long satisfied");
    }

    #[test]
    fn test_mul_bal16_3_by_u32_roundtrip() {
        fn decode(asg: &[F257], digs: &[usize]) -> i128 {
            let mut acc: i128 = 0;
            let mut pow: i128 = 1;
            for &v in digs {
                acc += (super::f257_to_i32_bal(asg[v]) as i128) * pow;
                pow *= 16;
            }
            acc
        }

        let mut b = Dr1csBuilder::<F257>::new();
        b.enforce_var_eq_const(b.one(), F257::ONE);

        // coeff = 0x0777 (positive, already balanced in 3 digits)
        let coeff3 = [
            alloc_bal16_digit(&mut b, 0x7),
            alloc_bal16_digit(&mut b, 0x7),
            alloc_bal16_digit(&mut b, 0x7),
        ];
        // u32 = 0xffff_fffe via byte->bal16 gadget.
        let x: u32 = 0xffff_fffe;
        let bytes = x.to_le_bytes();
        let b0 = alloc_byte::<F257>(&mut b, bytes[0]);
        let b1 = alloc_byte::<F257>(&mut b, bytes[1]);
        let b2 = alloc_byte::<F257>(&mut b, bytes[2]);
        let b3 = alloc_byte::<F257>(&mut b, bytes[3]);
        let u32_bal16 = u32_bytes_to_bal16_digits(&mut b, [b0.byte, b1.byte, b2.byte, b3.byte]);
        assert_eq!(u32_bal16.len(), 9);

        let out12 = mul_bal16_3_by_u32(&mut b, &coeff3, &u32_bal16);

        let (inst, asg) = b.into_instance();
        inst.check(&asg).expect("mul_bal16_3_by_u32 satisfiable");

        let coeff_i = decode(&asg, &coeff3);
        let u_i = decode(&asg, &u32_bal16);
        let out_i = decode(&asg, &out12);
        assert_eq!(out_i, coeff_i * u_i);
    }

    #[test]
    fn test_scale_short_coeffs_by_u32_roundtrip_small_vec() {
        fn decode(asg: &[F257], digs: &[usize]) -> i128 {
            let mut acc: i128 = 0;
            let mut pow: i128 = 1;
            for &v in digs {
                acc += (super::f257_to_i32_bal(asg[v]) as i128) * pow;
                pow *= 16;
            }
            acc
        }

        let mut b = Dr1csBuilder::<F257>::new();
        b.enforce_var_eq_const(b.one(), F257::ONE);

        // Two short coefficients (simulate small CM coeffs).
        // c0 = 0x0777, c1 = -5
        let c0 = [
            alloc_bal16_digit(&mut b, 0x7),
            alloc_bal16_digit(&mut b, 0x7),
            alloc_bal16_digit(&mut b, 0x0),
        ];
        let c1 = [
            alloc_bal16_digit(&mut b, -5),
            alloc_bal16_digit(&mut b, 0),
            alloc_bal16_digit(&mut b, 0),
        ];
        let coeffs = vec![c0, c1];

        // u32 = 0x04030201 (no 256 byte-view edge case)
        let x: u32 = 0x04030201;
        let bytes = x.to_le_bytes();
        let b0 = alloc_byte::<F257>(&mut b, bytes[0]);
        let b1 = alloc_byte::<F257>(&mut b, bytes[1]);
        let b2 = alloc_byte::<F257>(&mut b, bytes[2]);
        let b3 = alloc_byte::<F257>(&mut b, bytes[3]);
        let u32_digits = u32_bytes_to_bal16_digits(&mut b, [b0.byte, b1.byte, b2.byte, b3.byte]);
        assert_eq!(u32_digits.len(), 9);

        let prods = scale_short_coeffs_by_u32(&mut b, &coeffs, &u32_digits);
        assert_eq!(prods.len(), 2);

        let (inst, asg) = b.into_instance();
        inst.check(&asg).expect("scale_short_coeffs_by_u32 satisfiable");

        let u = decode(&asg, &u32_digits);
        for (i, c3) in coeffs.iter().enumerate() {
            let c = decode(&asg, c3);
            let p = decode(&asg, &prods[i]);
            assert_eq!(p, c * u);
        }
    }
}

