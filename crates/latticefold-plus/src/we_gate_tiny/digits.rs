use ark_ff::{BigInteger, Field, PrimeField};

use latticefold::transcript::poseidon::F257;
use symphony::dpp_sumcheck::Dr1csBuilder;

use super::gadgets::{alloc_bool, decompose_existing_byte_var_to_bits};
use super::cm_ir::{
    alloc_bal16_digit_ir, alloc_carry_pm128_ir, alloc_carry_pm2_ir, lower_ir_into_builder, IrBuilder as CmIrBuilder,
    add_bal16_same_len_ir, alloc_carry_pm1_ir, neg_bal16_digits_ir, sub_bal16_same_len_ir,
    u32_bytes_to_bal16_digits_from_bits_ir, u64_bytes_to_bal16_digits_from_bits_ir, VarRef as CmVarRef,
};

// -----------------------------------------------------------------------------
// Balanced base-16 (nibble) gadgets for *bounded* integer arithmetic in F257.
// -----------------------------------------------------------------------------

const NIBBLE_BASE: i32 = 16;

#[inline]
pub(crate) fn f257_to_i32_bal(x: F257) -> i32 {
    // Interpret F257 element as a signed integer in (-128..128] via centered lift.
    // Avoid `to_bytes_le()` allocations: read low limb directly.
    let limb0: u64 = x.into_bigint().as_ref().get(0).copied().unwrap_or(0);
    let u: i32 = (limb0 & 0xFFFF) as i32; // canonical rep in [0,256]
    if u <= 128 { u } else { u - 257 }
}

#[inline]
pub(crate) fn i32_to_f257(x: i32) -> F257 {
    let mut v = x % 257;
    if v < 0 {
        v += 257;
    }
    F257::from(v as u64)
}

/// Allocate a balanced base-16 digit variable in [-8,7] from witness `d`.
pub(crate) fn alloc_bal16_digit(b: &mut Dr1csBuilder<F257>, d: i8) -> usize {
    // IR is the source of truth; lower a tiny IR fragment into this builder.
    let base_one = b.assignment[b.one()];
    let base_asg = [base_one];
    let mut ib = CmIrBuilder::new(&base_asg);
    let out_ir = alloc_bal16_digit_ir(&mut ib, d);
    let lowered = lower_ir_into_builder(b, ib.ir);
    lowered.map_var(out_ir)
}


/// Allocate a signed carry `c ∈ [-2,2]` as an F257 variable, with a tight boolean decomposition.
pub(crate) fn alloc_carry_pm2(b: &mut Dr1csBuilder<F257>, c: i32) -> usize {
    let base_one = b.assignment[b.one()];
    let base_asg = [base_one];
    let mut ib = CmIrBuilder::new(&base_asg);
    let c_ir = alloc_carry_pm2_ir(&mut ib, c);
    let lowered = lower_ir_into_builder(b, ib.ir);
    lowered.map_var(c_ir)
}

/// Allocate a signed carry `c ∈ [-128,127]` as an F257 variable by range-checking an offset.
///
/// We represent `off = c + 128` as an 8-bit value in [0,255] (so it admits a byte decomposition),
/// then enforce `c = off - 128` as a linear relation in F257.
pub(crate) fn alloc_carry_pm128(b: &mut Dr1csBuilder<F257>, c: i32) -> usize {
    let base_one = b.assignment[b.one()];
    let base_asg = [base_one];
    let mut ib = CmIrBuilder::new(&base_asg);
    let c_ir = alloc_carry_pm128_ir(&mut ib, c);
    let lowered = lower_ir_into_builder(b, ib.ir);
    lowered.map_var(c_ir)
}

/// Allocate a signed carry `c ∈ [-64,63]` as an F257 variable by range-checking an offset.
pub(crate) fn alloc_carry_pm64(b: &mut Dr1csBuilder<F257>, c: i32) -> usize {
    assert!((-64..=63).contains(&c));
    let off_u8: u8 = (c + 64) as u8; // in [0,127]
    // 7-bit decomposition of off.
    let mut bits = [0usize; 7];
    for i in 0..7 {
        bits[i] = alloc_bool::<F257>(b, ((off_u8 >> i) & 1) == 1);
    }
    let off_var = b.new_var(F257::from(off_u8 as u64));
    // off = Σ 2^i * bits[i]
    let mut lc = vec![(F257::ONE, off_var)];
    let mut pow = F257::ONE;
    for i in 0..7 {
        lc.push((-pow, bits[i]));
        pow *= F257::from(2u64);
    }
    b.enforce_lc_times_one_eq_const(lc);
    // c = off - 64
    let c_var = b.new_var(i32_to_f257(c));
    b.enforce_lc_times_one_eq_const(vec![
        (F257::ONE, c_var),
        (-F257::ONE, off_var),
        (F257::from(64u64), b.one()),
    ]);
    c_var
}

/// Allocate a signed carry `c ∈ [-32,31]` as an F257 variable by range-checking an offset.
pub(crate) fn alloc_carry_pm32(b: &mut Dr1csBuilder<F257>, c: i32) -> usize {
    assert!((-32..=31).contains(&c));
    let off_u8: u8 = (c + 32) as u8; // in [0,63]
    // 6-bit decomposition of off.
    let mut bits = [0usize; 6];
    for i in 0..6 {
        bits[i] = alloc_bool::<F257>(b, ((off_u8 >> i) & 1) == 1);
    }
    let off_var = b.new_var(F257::from(off_u8 as u64));
    let mut lc = vec![(F257::ONE, off_var)];
    let mut pow = F257::ONE;
    for i in 0..6 {
        lc.push((-pow, bits[i]));
        pow *= F257::from(2u64);
    }
    b.enforce_lc_times_one_eq_const(lc);
    let c_var = b.new_var(i32_to_f257(c));
    b.enforce_lc_times_one_eq_const(vec![
        (F257::ONE, c_var),
        (-F257::ONE, off_var),
        (F257::from(32u64), b.one()),
    ]);
    c_var
}

// NOTE: We intentionally do NOT expose "wide carry" allocators like pm256/pm512/pm1024.
//
// In F257, integer-meaningful constraints must keep carries/digits in an injective lift range
// (typically |x| < 128). Larger carry ranges are too easy to misuse and can reintroduce
// mod-257 aliasing if used in the wrong place.

/// Allocate a signed carry `c ∈ [-16,15]` as an F257 variable by range-checking an offset.
pub(crate) fn alloc_carry_pm16(b: &mut Dr1csBuilder<F257>, c: i32) -> usize {
    assert!((-16..=15).contains(&c));
    let off_u8: u8 = (c + 16) as u8; // in [0,31]
    // 5-bit decomposition of off.
    let mut bits = [0usize; 5];
    for i in 0..5 {
        bits[i] = alloc_bool::<F257>(b, ((off_u8 >> i) & 1) == 1);
    }
    let off_var = b.new_var(F257::from(off_u8 as u64));
    let mut lc = vec![(F257::ONE, off_var)];
    let mut pow = F257::ONE;
    for i in 0..5 {
        lc.push((-pow, bits[i]));
        pow *= F257::from(2u64);
    }
    b.enforce_lc_times_one_eq_const(lc);
    let c_var = b.new_var(i32_to_f257(c));
    b.enforce_lc_times_one_eq_const(vec![
        (F257::ONE, c_var),
        (-F257::ONE, off_var),
        (F257::from(16u64), b.one()),
    ]);
    c_var
}

/// Allocate a signed carry `c ∈ [-8,7]` as an F257 variable by range-checking an offset.
pub(crate) fn alloc_carry_pm8(b: &mut Dr1csBuilder<F257>, c: i32) -> usize {
    assert!((-8..=7).contains(&c));
    let off_u8: u8 = (c + 8) as u8; // in [0,15]
    // 4-bit decomposition of off.
    let mut bits = [0usize; 4];
    for i in 0..4 {
        bits[i] = alloc_bool::<F257>(b, ((off_u8 >> i) & 1) == 1);
    }
    let off_var = b.new_var(F257::from(off_u8 as u64));
    b.enforce_lc_times_one_eq_const(vec![
        (F257::ONE, off_var),
        (-F257::ONE, bits[0]),
        (-F257::from(2u64), bits[1]),
        (-F257::from(4u64), bits[2]),
        (-F257::from(8u64), bits[3]),
    ]);
    let c_var = b.new_var(i32_to_f257(c));
    b.enforce_lc_times_one_eq_const(vec![
        (F257::ONE, c_var),
        (-F257::ONE, off_var),
        (F257::from(8u64), b.one()),
    ]);
    c_var
}

// NOTE: We intentionally do not use a narrower carry range (like pm64) for the streaming
// convolution multipliers. For 17×17 balanced digits, the worst-case carry magnitude can exceed
// 63 (e.g. ≈ 68), so pm64 would reject some valid witnesses. pm128 is safe.

pub(crate) fn alloc_carry_pm1(b: &mut Dr1csBuilder<F257>, c: i32) -> usize {
    // IR is the source of truth; lower a tiny IR fragment into this builder.
    let base_one = b.assignment[b.one()];
    let base_asg = [base_one];
    let mut ib = CmIrBuilder::new(&base_asg);
    let c_ir = alloc_carry_pm1_ir(&mut ib, c);
    let lowered = lower_ir_into_builder(b, ib.ir);
    lowered.map_var(c_ir)
}


/// Add two balanced base-16 digit vectors of the same length.
///
/// Assumes each digit is in [-8,7]. Enforces output digits in [-8,7] and carry in {-1,0,1}.
pub(crate) fn add_bal16_same_len(
    b: &mut Dr1csBuilder<F257>,
    a: &[usize],
    c: &[usize],
) -> (Vec<usize>, usize /* carry_out */) {
    let _prev = b.profile_enter("digits::add_bal16_same_len");
    let (ir, out_ir, carry_ir) = {
        let base_asg: &[F257] = &b.assignment;
        let mut ib = CmIrBuilder::new(base_asg);
        let a_ir: Vec<CmVarRef> = a.iter().copied().map(CmVarRef::Base).collect();
        let c_ir: Vec<CmVarRef> = c.iter().copied().map(CmVarRef::Base).collect();
        let (out_ir, carry_ir) = add_bal16_same_len_ir(&mut ib, &a_ir, &c_ir);
        (ib.ir, out_ir, carry_ir)
    };
    let lowered = lower_ir_into_builder(b, ir);
    let out: Vec<usize> = out_ir.into_iter().map(|v| lowered.map_var(v)).collect();
    let carry = lowered.map_var(carry_ir);
    let res = (out, carry);
    b.profile_exit(_prev);
    res
}

// -----------------------------------------------------------------------------
// Fox #1 (sound variant): loose digits + boundary normalization.
// -----------------------------------------------------------------------------
//
// We often need to sum many base-16 digit vectors. Doing a full carry-propagating
// normalization (`add_bal16_same_len`) after every partial sum is expensive because it
// allocates fresh balanced digits (and carries).
//
// A cheaper approach is:
// 1) keep a *redundant* representation where each digit can be outside [-8,7]
//    (still interpreted as Σ digit[i]*16^i), and update it with digitwise additions
//    (no carries), enforced by a single linear constraint per digit;
// 2) normalize once at the boundary into balanced digits [-8,7] with a carry chain.
//
// Soundness requirement:
// All linear relations used to build the loose digits must be *no-wrap* over F257,
// i.e. each digit value stays in (-128,128) so that equality mod 257 implies equality
// over the integers. We enforce this by only enabling the loose path when the caller
// provides a conservative, statement-derived bound < 128.

/// Add `src` into `acc` digitwise (no carry propagation).
///
/// This preserves the integer value \(\sum_i digit[i]·16^i\) but allows digits to grow
/// outside [-8,7]. The caller must keep a conservative bound on digit magnitudes < 128.
#[inline]
fn add_bal16_loose_in_place(b: &mut Dr1csBuilder<F257>, acc: &mut [usize], src: &[usize]) {
    debug_assert_eq!(acc.len(), src.len());
    for i in 0..acc.len() {
        let v = b.new_var(b.assignment[acc[i]] + b.assignment[src[i]]);
        // v = acc[i] + src[i]
        b.enforce_lc_times_one_eq_const(vec![
            (F257::ONE, v),
            (-F257::ONE, acc[i]),
            (-F257::ONE, src[i]),
        ]);
        acc[i] = v;
    }
}

/// Normalize a loose base-16 digit vector (little-endian) into balanced digits in [-8,7].
///
/// Returns `(out_digits, carry_out)`, where `out_digits.len() == loose.len()`.
/// The caller typically enforces `carry_out == 0` (i.e. the value fits).
fn normalize_bal16_loose_same_len_with_bound(
    b: &mut Dr1csBuilder<F257>,
    loose: &[usize],
    digit_abs_bound: i32,
) -> (Vec<usize>, usize) {
    let _prev = b.profile_enter("digits::normalize_bal16_loose");
    debug_assert!(digit_abs_bound >= 0);
    // Critical for no-wrap soundness when interpreting F257 as integers.
    debug_assert!(digit_abs_bound < 128);

    #[inline]
    fn alloc_carry_with_bound(b: &mut Dr1csBuilder<F257>, c: i32, bound: i32) -> usize {
        debug_assert!(bound >= 0);
        if bound <= 1 {
            alloc_carry_pm1(b, c)
        } else if bound <= 2 {
            alloc_carry_pm2(b, c)
        } else if bound <= 7 {
            alloc_carry_pm8(b, c)
        } else if bound <= 15 {
            alloc_carry_pm16(b, c)
        } else if bound <= 31 {
            alloc_carry_pm32(b, c)
        } else if bound <= 63 {
            alloc_carry_pm64(b, c)
        } else {
            alloc_carry_pm128(b, c)
        }
    }

    // Compute a conservative, statement-derived carry bound schedule from `digit_abs_bound`.
    //
    // If |d_i| <= B and |carry_i| <= C, then |d_i + carry_i| <= B + C, so
    // |carry_{i+1}| <= floor((B + C + 8)/16) + 1.
    let mut carry_bound: i32 = 0;
    let mut carry_bounds: Vec<i32> = Vec::with_capacity(loose.len());
    for _ in 0..loose.len() {
        let max_sum = digit_abs_bound + carry_bound;
        carry_bound = ((max_sum + 8) / 16) + 1;
        carry_bounds.push(carry_bound);
        debug_assert!(carry_bound < 128);
    }

    let mut out: Vec<usize> = Vec::with_capacity(loose.len());
    let mut carry_i32: i32 = 0;
    let mut carry_var = b.zero_var();

    // div_floor(x/16) for possibly-negative x.
    let div_floor = |x: i32, d: i32| -> i32 {
        debug_assert!(d > 0);
        if x >= 0 { x / d } else { -(((-x) + d - 1) / d) }
    };

    for (i, &dv) in loose.iter().enumerate() {
        let di = f257_to_i32_bal(b.assignment[dv]);
        debug_assert!(
            (-digit_abs_bound..=digit_abs_bound).contains(&di),
            "normalize_bal16_loose: digit out of assumed bound (|d|<={}): got {di}",
            digit_abs_bound
        );
        let sum = di + carry_i32;

        let mut carry_next = div_floor(sum + 8, NIBBLE_BASE);
        let mut rem = sum - NIBBLE_BASE * carry_next;
        while rem > 7 {
            carry_next += 1;
            rem -= NIBBLE_BASE;
        }
        while rem < -8 {
            carry_next -= 1;
            rem += NIBBLE_BASE;
        }
        debug_assert!((-8..=7).contains(&rem));
        debug_assert!(
            (-carry_bounds[i]..=carry_bounds[i]).contains(&carry_next),
            "normalize_bal16_loose: carry out of bound at i={i}: {carry_next} (bound={})",
            carry_bounds[i]
        );

        let rem_digit = alloc_bal16_digit(b, rem as i8);
        let carry_next_var = alloc_carry_with_bound(b, carry_next, carry_bounds[i]);

        // loose_i + carry_i - rem_i - 16*carry_{i+1} = 0
        b.enforce_lc_times_one_eq_const(vec![
            (F257::ONE, dv),
            (F257::ONE, carry_var),
            (-F257::ONE, rem_digit),
            (-F257::from(16u64), carry_next_var),
        ]);

        out.push(rem_digit);
        carry_i32 = carry_next;
        carry_var = carry_next_var;
    }

    b.profile_exit(_prev);
    (out, carry_var)
}

/// Add three balanced base-16 digit vectors of the same length.
///
/// Assumes each digit is in [-8,7]. Enforces output digits in [-8,7] and carry in [-2,2].
pub(crate) fn add3_bal16_same_len(
    b: &mut Dr1csBuilder<F257>,
    a: &[usize],
    c: &[usize],
    d: &[usize],
) -> (Vec<usize>, usize /* carry_out */) {
    let _prev = b.profile_enter("digits::add_bal16_same_len");
    assert_eq!(a.len(), c.len());
    assert_eq!(a.len(), d.len());
    let n = a.len();
    let mut out: Vec<usize> = Vec::with_capacity(n);
    let mut carry_i32: i32 = 0;
    let mut carry = b.zero_var();

    for i in 0..n {
        let ai = f257_to_i32_bal(b.assignment[a[i]]);
        let ci = f257_to_i32_bal(b.assignment[c[i]]);
        let di = f257_to_i32_bal(b.assignment[d[i]]);
        let sum = ai + ci + di + carry_i32;

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
        // With 3 inputs in [-8,7], carry stays in [-2,2].
        assert!((-2..=2).contains(&carry_next));
        assert!((-8..=7).contains(&rem));

        let out_digit = alloc_bal16_digit(b, rem as i8);
        // Statement-only arming: do not branch on witness carry value.
        let carry_next_var = alloc_carry_pm2(b, carry_next);

        // a_i + c_i + d_i + carry - out_i - 16*carry_next = 0
        b.enforce_lc_times_one_eq_const(vec![
            (F257::ONE, a[i]),
            (F257::ONE, c[i]),
            (F257::ONE, d[i]),
            (F257::ONE, carry),
            (-F257::ONE, out_digit),
            (-F257::from(16u64), carry_next_var),
        ]);

        out.push(out_digit);
        carry_i32 = carry_next;
        carry = carry_next_var;
    }

    let res = (out, carry);
    b.profile_exit(_prev);
    res
}

/// Negate a balanced base-16 digit vector (little-endian), producing digits in [-8,7].
pub(crate) fn neg_bal16_digits(
    b: &mut Dr1csBuilder<F257>,
    x: &[usize],
) -> (Vec<usize>, usize /* carry_out */) {
    let (ir, out_ir, carry_ir) = {
        let base_asg: &[F257] = &b.assignment;
        let mut ib = CmIrBuilder::new(base_asg);
        let x_ir: Vec<CmVarRef> = x.iter().copied().map(CmVarRef::Base).collect();
        let (out_ir, carry_ir) = neg_bal16_digits_ir(&mut ib, &x_ir);
        (ib.ir, out_ir, carry_ir)
    };
    let lowered = lower_ir_into_builder(b, ir);
    let out: Vec<usize> = out_ir.into_iter().map(|v| lowered.map_var(v)).collect();
    let carry = lowered.map_var(carry_ir);
    (out, carry)
}

/// Subtract two balanced base-16 digit vectors of the same length: `a - c`.
pub(crate) fn sub_bal16_same_len(
    b: &mut Dr1csBuilder<F257>,
    a: &[usize],
    c: &[usize],
) -> (Vec<usize>, usize /* carry_out */) {
    let (ir, out_ir, carry_ir) = {
        let base_asg: &[F257] = &b.assignment;
        let mut ib = CmIrBuilder::new(base_asg);
        let a_ir: Vec<CmVarRef> = a.iter().copied().map(CmVarRef::Base).collect();
        let c_ir: Vec<CmVarRef> = c.iter().copied().map(CmVarRef::Base).collect();
        let (out_ir, carry_ir) = sub_bal16_same_len_ir(&mut ib, &a_ir, &c_ir);
        (ib.ir, out_ir, carry_ir)
    };
    let lowered = lower_ir_into_builder(b, ir);
    let out: Vec<usize> = out_ir.into_iter().map(|v| lowered.map_var(v)).collect();
    let carry = lowered.map_var(carry_ir);
    (out, carry)
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

/// Allocate a signed carry `c ∈ [-14,14]` by allocating an offset `off = c + 16` and
/// enforcing `off ∈ [2,30]`.
///
/// Rationale: `mul_bal16_small` can produce carries outside `[-11,11]` when digits may take
/// the value `-8` (since `(-8)*(-8)=64`), even when `min(len)<=3`.
fn alloc_carry_pm14(b: &mut Dr1csBuilder<F257>, c: i32) -> usize {
    assert!((-14..=14).contains(&c));
    let off = (c + 16) as u8; // in [2,30]

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

    // Enforce off <= 30  <=> off != 31  <=> NOT(all 5 bits are 1).
    let t01 = b.new_var(b.assignment[bits[0]] * b.assignment[bits[1]]);
    b.enforce_mul(bits[0], bits[1], t01);
    let t012 = b.new_var(b.assignment[t01] * b.assignment[bits[2]]);
    b.enforce_mul(t01, bits[2], t012);
    let t0123 = b.new_var(b.assignment[t012] * b.assignment[bits[3]]);
    b.enforce_mul(t012, bits[3], t0123);
    let all1 = b.new_var(b.assignment[t0123] * b.assignment[bits[4]]);
    b.enforce_mul(t0123, bits[4], all1);
    b.enforce_var_eq_const(all1, F257::ZERO);

    // Enforce off >= 2  <=> (b1 OR b2 OR b3 OR b4) == 1.
    // or = 1 - Π (1-bi) for i=1..4
    let one_minus = |b: &mut Dr1csBuilder<F257>, x: usize| -> usize {
        let v = b.new_var(F257::ONE - b.assignment[x]);
        b.enforce_lc_times_one_eq_const(vec![(F257::ONE, v), (F257::ONE, x), (-F257::ONE, b.one())]);
        v
    };
    let om1 = one_minus(b, bits[1]);
    let om2 = one_minus(b, bits[2]);
    let om3 = one_minus(b, bits[3]);
    let om4 = one_minus(b, bits[4]);
    let p12 = b.new_var(b.assignment[om1] * b.assignment[om2]);
    b.enforce_mul(om1, om2, p12);
    let p123 = b.new_var(b.assignment[p12] * b.assignment[om3]);
    b.enforce_mul(p12, om3, p123);
    let p1234 = b.new_var(b.assignment[p123] * b.assignment[om4]);
    b.enforce_mul(p123, om4, p1234);
    // or = 1 - p1234
    let or = b.new_var(F257::ONE - b.assignment[p1234]);
    b.enforce_lc_times_one_eq_const(vec![(F257::ONE, or), (F257::ONE, p1234), (-F257::ONE, b.one())]);
    b.enforce_var_eq_const(or, F257::ONE);

    // Return carry var: c = off - 16
    let carry_var = b.new_var(F257::from(off as u64) - F257::from(16u64));
    b.enforce_lc_times_one_eq_const(vec![
        (F257::ONE, carry_var),
        (-F257::ONE, off_var),
        (F257::from(16u64), b.one()),
    ]);
    carry_var
}

/// Rebalance the final digit of a `mul_bal16_small_const_rhs4` product.
///
/// Input tail is a carry digit in [-16,15]; output is a balanced digit in [-8,7] plus a
/// final carry in {-1,0,1}.
pub(crate) fn rebalance_tail_pm16_to_pm1(b: &mut Dr1csBuilder<F257>, digits: &[usize]) -> Vec<usize> {
    assert!(!digits.is_empty());
    let l = digits.len();
    let tail = f257_to_i32_bal(b.assignment[digits[l - 1]]);
    debug_assert!((-16..=15).contains(&tail));

    let mut carry1 = if tail >= 0 { (tail + 8) / 16 } else { -(((-tail) + 8) / 16) };
    let mut rem = tail - 16 * carry1;
    while rem > 7 {
        carry1 += 1;
        rem -= 16;
    }
    while rem < -8 {
        carry1 -= 1;
        rem += 16;
    }
    debug_assert!((-8..=7).contains(&rem));
    debug_assert!((-1..=1).contains(&carry1));

    let rem_digit = alloc_bal16_digit(b, rem as i8);
    let carry1_var = alloc_carry_pm1(b, carry1);
    b.enforce_lc_times_one_eq_const(vec![
        (F257::ONE, digits[l - 1]),
        (-F257::ONE, rem_digit),
        (-F257::from(16u64), carry1_var),
    ]);

    let mut out = Vec::with_capacity(l + 1);
    out.extend_from_slice(&digits[..l - 1]);
    out.push(rem_digit);
    out.push(carry1_var);
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

#[inline]
fn bal16_zero(b: &mut Dr1csBuilder<F257>) -> usize {
    b.zero_var()
}

pub(crate) fn sum_product_digits_bal16(
    b: &mut Dr1csBuilder<F257>,
    products13: &[[usize; 13]],
    target_len: usize,
) -> Vec<usize> {
    assert!(target_len >= 13);
    let zero = bal16_zero(b);

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
    let zero = bal16_zero(b);
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
    let zero = bal16_zero(b);
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
    let zero = bal16_zero(b);
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
    let zero = bal16_zero(b);
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

    let zero_digit = bal16_zero(b);

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

    let zero = bal16_zero(b);
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
        return vec![bal16_zero(b)];
    }
    let zero = bal16_zero(b);
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
    let _prev = b.profile_enter("digits::mul_bal16_long_by_long");
    if a.is_empty() || bb.is_empty() {
        let out = vec![bal16_zero(b)];
        b.profile_exit(_prev);
        return out;
    }
    if a.len().min(bb.len()) <= 3 {
        let raw = mul_bal16_small(b, a, bb);
        let out = rebalance_tail_pm11_to_pm2(b, &raw);
        b.profile_exit(_prev);
        return out;
    }

    // -------------------------------------------------------------------------
    // Fox #1 (SOUND): Use loose digit accumulation when bounds permit.
    //
    // This path is sound because:
    // 1. mul_bal16_small handles blocks of size 3×n, where sums stay under 257
    // 2. Loose accumulation uses pure linear constraints (no ambiguity)
    // 3. Final normalization has bound < 128, ensuring no F257 wrap-around
    //
    // We check this FIRST (before the streaming path) because it's provably sound
    // for all operand sizes where the bound check passes.
    // -------------------------------------------------------------------------
    let (short, long) = if a.len() <= bb.len() { (a, bb) } else { (bb, a) };
    let zero = bal16_zero(b);
    let blocks = (short.len() + 2) / 3;
    let per_block_len = long.len() + 5;
    let target_len = per_block_len + 3 * (blocks - 1) + 2;

    if long.len() <= 19 && short.len() <= 19 {
        let per_term_bound: i32 = 10; // conservative: digits in [-8,7] plus small tail carry
        let acc_bound: i32 = (blocks as i32) * per_term_bound;
        if acc_bound < 128 {
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
                add_bal16_loose_in_place(b, &mut acc, &shifted);
            }
            let (norm, carry) = normalize_bal16_loose_same_len_with_bound(b, &acc, acc_bound);
            b.enforce_var_eq_const(carry, F257::ZERO);
            b.profile_exit(_prev);
            return norm;
        }
    }

    // Fallback: original carry-normalizing accumulation.
    // This path is used when neither Fox #1 nor streaming applies.
    let zero = bal16_zero(b);
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

    let out = acc;
    b.profile_exit(_prev);
    out
}

/// Multiply two balanced base-16 digit vectors (little-endian), specialized for min(len)<=3.
pub(crate) fn mul_bal16_small(b: &mut Dr1csBuilder<F257>, a: &[usize], bb: &[usize]) -> Vec<usize> {
    let la = a.len();
    let lb = bb.len();
    assert!(la > 0 && lb > 0);
    assert!(la.min(lb) <= 3, "mul_bal16_small requires min(len) <= 3");

    let mut out: Vec<usize> = Vec::with_capacity(la + lb);
    let mut carry_i32: i32 = 0;
    let mut carry_var = b.zero_var();

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
            (-14..=14).contains(&carry),
            "carry out of expected range: {carry} from sum {sum}"
        );

        let digit_var = alloc_bal16_digit(b, rem as i8);
        // Statement-only arming: avoid witness-dependent gadget selection.
        let carry_out_var = alloc_carry_pm14(b, carry);

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

/// Multiply balanced base-16 digits by **constant** balanced digits (little-endian),
/// specialized for `a.len() <= 3` (or `bb_const.len() <= 3` by swapping yourself).
///
/// This avoids any `enforce_mul`: each digit product is a constant scaling inside a linear
/// combination, so the whole step is enforced by a single linear constraint per output digit.
pub(crate) fn mul_bal16_small_const_rhs(
    b: &mut Dr1csBuilder<F257>,
    a: &[usize],
    bb_const: &[i8],
) -> Vec<usize> {
    let la = a.len();
    let lb = bb_const.len();
    assert!(la > 0 && lb > 0);
    assert!(la <= 3, "mul_bal16_small_const_rhs requires a.len() <= 3");

    #[inline]
    fn f257_from_i8(x: i8) -> F257 {
        if x >= 0 {
            F257::from(x as u64)
        } else {
            -F257::from((-x) as u64)
        }
    }

    let mut out: Vec<usize> = Vec::with_capacity(la + lb);
    let mut carry_i32: i32 = 0;
    let mut carry_var = b.zero_var();

    for k in 0..(la + lb - 1) {
        let mut sum: i32 = carry_i32;
        let mut lc: Vec<(F257, usize)> = Vec::new();
        lc.push((F257::ONE, carry_var));

        for i in 0..la {
            let j = k as i32 - i as i32;
            if j < 0 || j >= lb as i32 {
                continue;
            }
            let j = j as usize;
            let aval = f257_to_i32_bal(b.assignment[a[i]]);
            let bval = bb_const[j] as i32;
            sum += aval * bval;
            let cf = f257_from_i8(bb_const[j]);
            if cf != F257::ZERO {
                lc.push((cf, a[i]));
            }
        }

        let div_floor = |x: i32, d: i32| -> i32 {
            debug_assert!(d > 0);
            if x >= 0 { x / d } else { -(((-x) + d - 1) / d) }
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
            (-14..=14).contains(&carry),
            "carry out of expected range: {carry} from sum {sum}"
        );

        let digit_var = alloc_bal16_digit(b, rem as i8);
        // Statement-only arming: avoid witness-dependent gadget selection.
        let carry_out_var = alloc_carry_pm14(b, carry);

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

/// Multiply balanced base-16 digits by **constant** balanced digits (little-endian),
/// specialized for `a.len() <= 4`.
///
/// This is a no-wrap-safe variant for strict mode: each output digit equation contains at most
/// 4 terms of the form `a[i] * const`, so the integer magnitude stays < 257.
fn mul_bal16_small_const_rhs4(
    b: &mut Dr1csBuilder<F257>,
    a: &[usize; 4],
    a_len: usize,
    bb_const: &[i8],
) -> Vec<usize> {
    debug_assert!(a_len >= 1 && a_len <= 4);
    let la = a_len;
    let lb = bb_const.len();

    #[inline]
    fn f257_from_i8(x: i8) -> F257 {
        if x >= 0 { F257::from(x as u64) } else { -F257::from((-x) as u64) }
    }

    let mut out: Vec<usize> = Vec::with_capacity(la + lb);
    let mut carry_i32: i32 = 0;
    let mut carry_var = b.zero_var();

    let div_floor = |x: i32, d: i32| -> i32 {
        debug_assert!(d > 0);
        if x >= 0 { x / d } else { -(((-x) + d - 1) / d) }
    };

    for k in 0..(la + lb - 1) {
        let mut sum: i32 = carry_i32;
        let mut lc: Vec<(F257, usize)> = Vec::new();
        lc.push((F257::ONE, carry_var));

        for i in 0..la {
            let j = k as i32 - i as i32;
            if j < 0 || j >= lb as i32 {
                continue;
            }
            let j = j as usize;
            let aval = f257_to_i32_bal(b.assignment[a[i]]);
            let bval = bb_const[j] as i32;
            sum += aval * bval;
            let cf = f257_from_i8(bb_const[j]);
            if cf != F257::ZERO {
                lc.push((cf, a[i]));
            }
        }

        // With <=4 terms of magnitude <=56 and carry in <=15, |sum| < 257.
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
        debug_assert!((-8..=7).contains(&rem));
        debug_assert!((-16..=15).contains(&carry));

        let digit_var = alloc_bal16_digit(b, rem as i8);
        let carry_out_var = alloc_carry_pm16(b, carry);
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

/// Multiply a long balanced digit vector by a **constant** long balanced digit vector.
///
/// This is the const-RHS analogue of `mul_bal16_long_by_long`, and avoids any digit×digit
/// multiplications by keeping all RHS digits as constants inside linear constraints.
pub(crate) fn mul_bal16_long_by_const_rhs(
    b: &mut Dr1csBuilder<F257>,
    a: &[usize],
    bb_const: &[i8],
) -> Vec<usize> {
    let _prev = b.profile_enter("digits::mul_bal16_long_by_const_rhs");
    if a.is_empty() || bb_const.is_empty() {
        let out = vec![bal16_zero(b)];
        b.profile_exit(_prev);
        return out;
    }
    if a.len() <= 3 {
        let raw = mul_bal16_small_const_rhs(b, a, bb_const);
        let out = rebalance_tail_pm11_to_pm2(b, &raw);
        b.profile_exit(_prev);
        return out;
    }

    // -------------------------------------------------------------------------
    // Block multiplication with Fox #1 loose accumulation (SOUND).
    //
    // This path is sound because:
    // 1. mul_bal16_small_const_rhs4 handles blocks of up to 4 terms, keeping sums < 257
    // 2. Loose accumulation uses pure linear constraints (no ambiguity)
    // 3. Final normalization has bound < 128, ensuring no F257 wrap-around
    //
    // We check this FIRST (before the streaming path) because it's provably sound
    // for all operand sizes where the bound check passes.
    // -------------------------------------------------------------------------
    let zero = bal16_zero(b);
    const BLK: usize = 4;
    let blocks = (a.len() + (BLK - 1)) / BLK;
    let per_block_len = bb_const.len() + BLK + 2;
    let target_len = per_block_len + BLK * (blocks - 1) + 3;

    let per_term_bound: i32 = 10;
    let acc_bound: i32 = (blocks as i32) * per_term_bound;

    if a.len() <= 19 && bb_const.len() <= 17 && acc_bound < 128 {
        // Build all shifted block-products using the sound 4-term multiplier.
        let mut terms: Vec<Vec<usize>> = Vec::with_capacity(blocks);
        for blk in 0..blocks {
            let start = blk * BLK;
            let end = core::cmp::min(start + BLK, a.len());
            let mut coeff4 = [zero; 4];
            for j in 0..(end - start) {
                coeff4[j] = a[start + j];
            }
            let raw = mul_bal16_small_const_rhs4(b, &coeff4, end - start, bb_const);
            let reb = rebalance_tail_pm16_to_pm1(b, &raw);
            let shifted = shift_pad_bal16(&reb, blk * BLK, target_len, zero);
            terms.push(shifted);
        }

        // Fox #1: accumulate as loose digits, normalize once.
        let mut acc = vec![zero; target_len];
        for t in &terms {
            add_bal16_loose_in_place(b, &mut acc, t);
        }
        let (norm, carry) = normalize_bal16_loose_same_len_with_bound(b, &acc, acc_bound);
        b.enforce_var_eq_const(carry, F257::ZERO);
        b.profile_exit(_prev);
        return norm;
    }

    // -------------------------------------------------------------------------
    // Fallback for very large operands (acc_bound >= 128).
    // Uses 3-at-a-time reduction instead of loose accumulation.
    // -------------------------------------------------------------------------
    let mut terms: Vec<Vec<usize>> = Vec::with_capacity(blocks);
    for blk in 0..blocks {
        let start = blk * BLK;
        let end = core::cmp::min(start + BLK, a.len());
        let mut coeff4 = [zero; 4];
        for j in 0..(end - start) {
            coeff4[j] = a[start + j];
        }
        let raw = mul_bal16_small_const_rhs4(b, &coeff4, end - start, bb_const);
        let reb = rebalance_tail_pm16_to_pm1(b, &raw);
        let shifted = shift_pad_bal16(&reb, blk * BLK, target_len, zero);
        terms.push(shifted);
    }

    // Reduce by summing 3-at-a-time (fewer passes than pairwise accumulation).
    let mut stack = terms;
    while stack.len() > 1 {
        if stack.len() >= 3 {
            let d = stack.pop().unwrap();
            let c = stack.pop().unwrap();
            let aa = stack.pop().unwrap();
            let (sum, carry) = add3_bal16_same_len(b, &aa, &c, &d);
            b.enforce_var_eq_const(carry, F257::ZERO);
            stack.push(sum);
        } else {
            let c = stack.pop().unwrap();
            let aa = stack.pop().unwrap();
            let (sum, carry) = add_bal16_same_len(b, &aa, &c);
            b.enforce_var_eq_const(carry, F257::ZERO);
            stack.push(sum);
        }
    }
    let out = stack.pop().unwrap();
    b.profile_exit(_prev);
    out
}

/// Convert 4 little-endian byte vars (0..255) into balanced base-16 digits (len 9).
pub(crate) fn u32_bytes_to_bal16_digits(b: &mut Dr1csBuilder<F257>, bytes_le: [usize; 4]) -> Vec<usize> {
    if let Some(v) = b.u32_bal16_cache.get(&bytes_le) {
        return v.clone();
    }
    let _prev = b.profile_enter("digits::u32_bytes_to_bal16_digits");
    b.profile_exit(_prev);
    let bytes_bits: [[usize; 8]; 4] = core::array::from_fn(|i| decompose_existing_byte_var_to_bits::<F257>(b, bytes_le[i]));

    let (ir, out_ir) = {
        let base_asg: &[F257] = &b.assignment;
        let mut ib = CmIrBuilder::new(base_asg);
        let bits_ir: [[CmVarRef; 8]; 4] =
            core::array::from_fn(|i| core::array::from_fn(|j| CmVarRef::Base(bytes_bits[i][j])));
        let out_ir = u32_bytes_to_bal16_digits_from_bits_ir(&mut ib, &bits_ir);
        (ib.ir, out_ir)
    };
    let lowered = lower_ir_into_builder(b, ir);
    let out: [usize; 9] = core::array::from_fn(|k| lowered.map_var(out_ir[k]));
    let v = out.to_vec();
    b.u32_bal16_cache.insert(bytes_le, v.clone());
    v
}

/// Convert 8 little-endian byte vars (0..255) into balanced base-16 digits (len 17).
///
/// Output digits are little-endian base-16 with each digit in [-8,7], followed by a final
/// carry digit (also in {0,1} for a canonical u64 byte encoding).
pub(crate) fn u64_bytes_to_bal16_digits(b: &mut Dr1csBuilder<F257>, bytes_le: [usize; 8]) -> Vec<usize> {
    if let Some(v) = b.u64_bal16_cache.get(&bytes_le) {
        return v.clone();
    }
    let _prev = b.profile_enter("digits::u64_bytes_to_bal16_digits");
    b.profile_exit(_prev);
    let bytes_bits: [[usize; 8]; 8] = core::array::from_fn(|i| decompose_existing_byte_var_to_bits::<F257>(b, bytes_le[i]));

    let (ir, out_ir) = {
        let base_asg: &[F257] = &b.assignment;
        let mut ib = CmIrBuilder::new(base_asg);
        let bits_ir: [[CmVarRef; 8]; 8] =
            core::array::from_fn(|i| core::array::from_fn(|j| CmVarRef::Base(bytes_bits[i][j])));
        let out_ir = u64_bytes_to_bal16_digits_from_bits_ir(&mut ib, &bits_ir);
        (ib.ir, out_ir)
    };
    let lowered = lower_ir_into_builder(b, ir);
    let out: [usize; 17] = core::array::from_fn(|k| lowered.map_var(out_ir[k]));
    let v = out.to_vec();
    b.u64_bal16_cache.insert(bytes_le, v.clone());
    v
}

