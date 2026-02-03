use ark_ff::{Field, PrimeField};

use latticefold::transcript::poseidon::F257;
use symphony::dpp_sumcheck::Dr1csBuilder;

use super::gadgets::decompose_existing_byte_var_to_bits;
use super::cm_ir::{
    alloc_bal16_digit_ir, lower_ir_into_builder, IrBuilder as CmIrBuilder,
    add_bal16_loose_same_len_ir, add_bal16_same_len_ir, mul_bal16_small_ir, rebalance_tail_pm11_to_pm2_ir,
    u32_bytes_to_bal16_digits_from_bits_ir, u64_bytes_to_bal16_digits_from_bits_ir, VarRef as CmVarRef,
    Bal16CheckedIr as CmBal16CheckedIr,
    Bal16LooseIr as CmBal16LooseIr,
    normalize_bal16_loose_same_len_ir,
};

// -----------------------------------------------------------------------------
// Balanced base-16 (nibble) gadgets for *bounded* integer arithmetic in F257.
// -----------------------------------------------------------------------------

#[inline]
fn enforce_var_eq_const_ir(b: &mut Dr1csBuilder<F257>, x: usize, c: F257) {
    let base_asg: &[F257] = &b.assignment;
    let mut ib = if b.is_count_only() {
        CmIrBuilder::new_count_only(base_asg)
    } else {
        CmIrBuilder::new(base_asg)
    };
    ib.ir.enforce_var_eq_const(CmVarRef::Base(x), c);
    let _lowered = lower_ir_into_builder(b, ib.ir);
}

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
    let mut ib = if b.is_count_only() {
        CmIrBuilder::new_count_only(&base_asg)
    } else {
        CmIrBuilder::new(&base_asg)
    };
    let out_ir = alloc_bal16_digit_ir(&mut ib, d);
    let lowered = lower_ir_into_builder(b, ib.ir);
    lowered.map_var(out_ir)
}


// -----------------------------------------------------------------------------
// Fox #1 (maintainable): explicit checked vs loose digit types
// -----------------------------------------------------------------------------
//
// The core maintenance goal is to make it *obvious in signatures* whether a function expects:
// - **checked** balanced base-16 digits (each digit ∈ [-8,7], proven by bit constraints), or
// - **loose** digits (redundant representation with a static bound |d_i| ≤ M < 128, no bits).
//
// This prevents accidentally feeding a "loose" vector into a gadget that assumes canonical digits,
// and makes conversion points explicit.

#[derive(Clone, Debug)]
pub(crate) struct Bal16Checked(pub(crate) Vec<usize>);

impl Bal16Checked {
    #[inline]
    pub(crate) fn as_slice(&self) -> &[usize] {
        &self.0
    }

    #[inline]
    pub(crate) fn len(&self) -> usize {
        self.0.len()
    }

    #[inline]
    pub(crate) fn into_vec(self) -> Vec<usize> {
        self.0
    }
}

impl core::ops::Deref for Bal16Checked {
    type Target = [usize];
    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

#[derive(Clone, Debug)]
pub(crate) struct Bal16Loose {
    pub(crate) digits: Vec<usize>,
    /// Static bound on digit magnitudes, required to satisfy |d_i| < 128 (no-wrap in F257).
    pub(crate) abs_bound: i32,
}

impl Bal16Loose {
    #[inline]
    pub(crate) fn len(&self) -> usize {
        self.digits.len()
    }
}

/// Add two balanced base-16 digit vectors of the same length.
///
/// Assumes each digit is in [-8,7]. Enforces output digits in [-8,7] and carry in {-1,0,1}.
pub(crate) fn add_bal16_same_len(
    b: &mut Dr1csBuilder<F257>,
    a: &Bal16Checked,
    c: &Bal16Checked,
) -> (Bal16Checked, usize /* carry_out */) {
    let _prev = b.profile_enter("digits::add_bal16_same_len");
    let (ir, out_ir, carry_ir) = {
        let base_asg: &[F257] = &b.assignment;
        let mut ib = if b.is_count_only() {
            CmIrBuilder::new_count_only(base_asg)
        } else {
            CmIrBuilder::new(base_asg)
        };
        let a_ir = CmBal16CheckedIr(a.as_slice().iter().copied().map(CmVarRef::Base).collect());
        let c_ir = CmBal16CheckedIr(c.as_slice().iter().copied().map(CmVarRef::Base).collect());
        let (out_ir, carry_ir) = add_bal16_same_len_ir(&mut ib, &a_ir, &c_ir);
        (ib.ir, out_ir, carry_ir)
    };
    let lowered = lower_ir_into_builder(b, ir);
    let out: Vec<usize> = out_ir.0.into_iter().map(|v| lowered.map_var(v)).collect();
    let carry = lowered.map_var(carry_ir);
    let res = (Bal16Checked(out), carry);
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
fn add_bal16_loose_in_place(b: &mut Dr1csBuilder<F257>, acc: &mut Bal16Loose, src: &Bal16Checked) {
    debug_assert_eq!(acc.len(), src.len());
    let (ir, out_ir) = {
        let base_asg: &[F257] = &b.assignment;
        let mut ib = if b.is_count_only() {
            CmIrBuilder::new_count_only(base_asg)
        } else {
            CmIrBuilder::new(base_asg)
        };
        let acc_ir = CmBal16LooseIr {
            digits: acc.digits.iter().copied().map(CmVarRef::Base).collect(),
            abs_bound: acc.abs_bound,
        };
        // `src` digits are in [-8,7], so abs bound 8 is conservative.
        let src_ir = CmBal16LooseIr { digits: src.0.iter().copied().map(CmVarRef::Base).collect(), abs_bound: 8 };
        let out_ir = add_bal16_loose_same_len_ir(&mut ib, &acc_ir, &src_ir);
        (ib.ir, out_ir)
    };
    let lowered = lower_ir_into_builder(b, ir);
    acc.digits = out_ir.digits.into_iter().map(|v| lowered.map_var(v)).collect();
    // Keep `acc.abs_bound` unchanged: callers already track a conservative bound.
}

/// Normalize a loose base-16 digit vector (little-endian) into balanced digits in [-8,7].
///
/// Returns `(out_digits, carry_out)`, where `out_digits.len() == loose.len()`.
/// The caller typically enforces `carry_out == 0` (i.e. the value fits).
fn normalize_bal16_loose_same_len_with_bound(
    b: &mut Dr1csBuilder<F257>,
    loose: &Bal16Loose,
) -> (Bal16Checked, usize) {
    let _prev = b.profile_enter("digits::normalize_bal16_loose");
    assert!(loose.abs_bound >= 0);
    assert!(loose.abs_bound < 128);

    // Delegate to the IR "source of truth" implementation, then lower into this builder.
    let (ir, out_ir, carry_ir) = {
        let base_asg: &[F257] = &b.assignment;
        let mut ib = if b.is_count_only() {
            CmIrBuilder::new_count_only(base_asg)
        } else {
            CmIrBuilder::new(base_asg)
        };
        let loose_ir = CmBal16LooseIr {
            digits: loose.digits.iter().copied().map(CmVarRef::Base).collect(),
            abs_bound: loose.abs_bound,
        };
        let (out_ir, carry_ir) = normalize_bal16_loose_same_len_ir(&mut ib, &loose_ir);
        (ib.ir, out_ir, carry_ir)
    };
    let lowered = lower_ir_into_builder(b, ir);
    let out: Vec<usize> = out_ir.0.into_iter().map(|v| lowered.map_var(v)).collect();
    let carry = lowered.map_var(carry_ir);

    b.profile_exit(_prev);
    (Bal16Checked(out), carry)
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
    let (ir, out_ir) = {
        let base_asg: &[F257] = &b.assignment;
        let mut ib = if b.is_count_only() {
            CmIrBuilder::new_count_only(base_asg)
        } else {
            CmIrBuilder::new(base_asg)
        };
        let digits_ir: Vec<CmVarRef> = digits.iter().copied().map(CmVarRef::Base).collect();
        let out_ir = rebalance_tail_pm11_to_pm2_ir(&mut ib, &digits_ir);
        (ib.ir, out_ir)
    };
    let lowered = lower_ir_into_builder(b, ir);
    out_ir.into_iter().map(|v| lowered.map_var(v)).collect()
}

fn shift_pad_bal16(digits: &[usize], shift: usize, target_len: usize, zero_digit: usize) -> Bal16Checked {
    assert!(shift <= target_len);
    assert!(digits.len() + shift <= target_len);
    let mut out = Vec::with_capacity(target_len);
    out.extend(std::iter::repeat(zero_digit).take(shift));
    out.extend_from_slice(digits);
    out.extend(std::iter::repeat(zero_digit).take(target_len - shift - digits.len()));
    Bal16Checked(out)
}

#[inline]
fn bal16_zero(b: &mut Dr1csBuilder<F257>) -> usize {
    b.zero_var()
}

pub(crate) fn sum_product_digits_bal16(
    b: &mut Dr1csBuilder<F257>,
    products13: &[[usize; 13]],
    target_len: usize,
) -> Bal16Checked {
    assert!(target_len >= 13);
    let zero = bal16_zero(b);

    // Fox #1: when we sum only a few vectors, accumulate as *loose* digits and normalize once.
    let per_term_bound: i32 = 10; // conservative (matches other Fox #1 paths in this module)
    let acc_bound: i32 = (products13.len() as i32) * per_term_bound;
    if acc_bound < 128 {
        let mut acc = Bal16Loose { digits: vec![zero; target_len], abs_bound: acc_bound };
        for p13 in products13 {
            let padded = shift_pad_bal16(p13, 0, target_len, zero);
            add_bal16_loose_in_place(b, &mut acc, &padded);
        }
        let (norm, carry) = normalize_bal16_loose_same_len_with_bound(b, &acc);
        enforce_var_eq_const_ir(b, carry, F257::ZERO);
        return norm;
    }

    // Fallback: normalize after every addition.
    let mut acc = Bal16Checked(vec![zero; target_len]);
    for p13 in products13 {
        let padded = shift_pad_bal16(p13, 0, target_len, zero);
        let (new_acc, carry) = add_bal16_same_len(b, &acc, &padded);
        acc = new_acc;
        enforce_var_eq_const_ir(b, carry, F257::ZERO);
    }
    acc
}

pub(crate) fn sum_product_digits_bal16_22(
    b: &mut Dr1csBuilder<F257>,
    products22: &[[usize; 22]],
    target_len: usize,
) -> Bal16Checked {
    assert!(target_len >= 22);
    let zero = bal16_zero(b);

    // Fox #1: loose accumulation when bounds permit.
    let per_term_bound: i32 = 10;
    let acc_bound: i32 = (products22.len() as i32) * per_term_bound;
    if acc_bound < 128 {
        let mut acc = Bal16Loose { digits: vec![zero; target_len], abs_bound: acc_bound };
        for p22 in products22 {
            let padded = shift_pad_bal16(p22, 0, target_len, zero);
            add_bal16_loose_in_place(b, &mut acc, &padded);
        }
        let (norm, carry) = normalize_bal16_loose_same_len_with_bound(b, &acc);
        enforce_var_eq_const_ir(b, carry, F257::ZERO);
        return norm;
    }

    let mut acc = Bal16Checked(vec![zero; target_len]);
    for p22 in products22 {
        let padded = shift_pad_bal16(p22, 0, target_len, zero);
        let (new_acc, carry) = add_bal16_same_len(b, &acc, &padded);
        acc = new_acc;
        enforce_var_eq_const_ir(b, carry, F257::ZERO);
    }
    acc
}

pub(crate) fn sum_bal16_vectors_fixed_len(
    b: &mut Dr1csBuilder<F257>,
    vecs: &[&Bal16Checked],
    len: usize,
) -> Bal16Checked {
    let zero = bal16_zero(b);
    // Fox #1: loose accumulation when bounds permit.
    let per_term_bound: i32 = 10;
    let acc_bound: i32 = (vecs.len() as i32) * per_term_bound;
    if acc_bound < 128 {
        let mut acc = Bal16Loose { digits: vec![zero; len], abs_bound: acc_bound };
        for v in vecs {
            assert_eq!(v.len(), len);
            add_bal16_loose_in_place(b, &mut acc, v);
        }
        let (norm, carry) = normalize_bal16_loose_same_len_with_bound(b, &acc);
        enforce_var_eq_const_ir(b, carry, F257::ZERO);
        return norm;
    }

    let mut acc = Bal16Checked(vec![zero; len]);
    for v in vecs {
        assert_eq!(v.len(), len);
        let (new_acc, carry) = add_bal16_same_len(b, &acc, v);
        acc = new_acc;
        enforce_var_eq_const_ir(b, carry, F257::ZERO);
    }
    acc
}

pub(crate) fn sum_products13_coeffwise_fixed_len(
    b: &mut Dr1csBuilder<F257>,
    per_surface_products13: &[&[[usize; 13]]],
    ring_dim: usize,
    out_len: usize,
) -> Vec<Bal16Checked> {
    let zero = bal16_zero(b);
    let per_term_bound: i32 = 10;
    let acc_bound: i32 = (per_surface_products13.len() as i32) * per_term_bound;
    let mut out: Vec<Bal16Checked> = Vec::with_capacity(ring_dim);
    for coeff_idx in 0..ring_dim {
        let mut acc_checked = Bal16Checked(vec![zero; out_len]);
        if acc_bound < 128 {
            let mut acc = Bal16Loose { digits: vec![zero; out_len], abs_bound: acc_bound };
            for surf in per_surface_products13 {
                let p13 = &surf[coeff_idx];
                let padded = shift_pad_bal16(p13, 0, out_len, zero);
                add_bal16_loose_in_place(b, &mut acc, &padded);
            }
            let (norm, carry) = normalize_bal16_loose_same_len_with_bound(b, &acc);
            enforce_var_eq_const_ir(b, carry, F257::ZERO);
            acc_checked = norm;
        } else {
            for surf in per_surface_products13 {
                let p13 = &surf[coeff_idx];
                let padded = shift_pad_bal16(p13, 0, out_len, zero);
                let (new_acc, carry) = add_bal16_same_len(b, &acc_checked, &padded);
                acc_checked = new_acc;
                enforce_var_eq_const_ir(b, carry, F257::ZERO);
            }
        }
        out.push(acc_checked);
    }
    out
}

pub(crate) fn sum_products22_coeffwise_fixed_len(
    b: &mut Dr1csBuilder<F257>,
    per_surface_products22: &[&[[usize; 22]]],
    ring_dim: usize,
    out_len: usize,
) -> Vec<Bal16Checked> {
    let zero = bal16_zero(b);
    let per_term_bound: i32 = 10;
    let acc_bound: i32 = (per_surface_products22.len() as i32) * per_term_bound;
    let mut out: Vec<Bal16Checked> = Vec::with_capacity(ring_dim);
    for coeff_idx in 0..ring_dim {
        let mut acc_checked = Bal16Checked(vec![zero; out_len]);
        if acc_bound < 128 {
            let mut acc = Bal16Loose { digits: vec![zero; out_len], abs_bound: acc_bound };
            for surf in per_surface_products22 {
                let p22 = &surf[coeff_idx];
                let padded = shift_pad_bal16(p22, 0, out_len, zero);
                add_bal16_loose_in_place(b, &mut acc, &padded);
            }
            let (norm, carry) = normalize_bal16_loose_same_len_with_bound(b, &acc);
            enforce_var_eq_const_ir(b, carry, F257::ZERO);
            acc_checked = norm;
        } else {
            for surf in per_surface_products22 {
                let p22 = &surf[coeff_idx];
                let padded = shift_pad_bal16(p22, 0, out_len, zero);
                let (new_acc, carry) = add_bal16_same_len(b, &acc_checked, &padded);
                acc_checked = new_acc;
                enforce_var_eq_const_ir(b, carry, F257::ZERO);
            }
        }
        out.push(acc_checked);
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

fn mul_bal16_9_by_9_u32ish(b: &mut Dr1csBuilder<F257>, a9: &[usize], b9: &[usize]) -> Bal16Checked {
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
    t01.0.push(carry01);

    let mut s2_pad_v = s2.into_vec();
    s2_pad_v.push(zero_digit);
    let s2_pad = Bal16Checked(s2_pad_v);
    debug_assert_eq!(t01.len(), s2_pad.len());
    let (mut out, carry) = add_bal16_same_len(b, &t01, &s2_pad);
    out.0.push(carry);
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
    let raw = mul_bal16_9_by_9_u32ish(b, a9, b9).into_vec();
    if raw.len() <= out_len {
        let mut out = raw;
        out.extend(std::iter::repeat(zero).take(out_len - out.len()));
        return out;
    }

    for &dv in &raw[out_len..] {
        enforce_var_eq_const_ir(b, dv, F257::ZERO);
    }
    raw[..out_len].to_vec()
}

/// Multiply two balanced base-16 digit vectors (little-endian), specialized for min(len)<=3.
pub(crate) fn mul_bal16_small(b: &mut Dr1csBuilder<F257>, a: &[usize], bb: &[usize]) -> Vec<usize> {
    let _prev = b.profile_enter("digits::mul_bal16_small");
    let (ir, out_ir) = {
        let base_asg: &[F257] = &b.assignment;
        let mut ib = if b.is_count_only() {
            CmIrBuilder::new_count_only(base_asg)
        } else {
            CmIrBuilder::new(base_asg)
        };
        let a_ir: Vec<CmVarRef> = a.iter().copied().map(CmVarRef::Base).collect();
        let b_ir: Vec<CmVarRef> = bb.iter().copied().map(CmVarRef::Base).collect();
        let out_ir = mul_bal16_small_ir(&mut ib, &a_ir, &b_ir);
        (ib.ir, out_ir)
    };
    let lowered = lower_ir_into_builder(b, ir);
    let out: Vec<usize> = out_ir.into_iter().map(|v| lowered.map_var(v)).collect();
    b.profile_exit(_prev);
    out
}

/// Multiply balanced base-16 digits by **constant** balanced digits (little-endian),
/// specialized for `a.len() <= 3` (or `bb_const.len() <= 3` by swapping yourself).
///
/// This avoids any `enforce_mul`: each digit product is a constant scaling inside a linear
/// combination, so the whole step is enforced by a single linear constraint per output digit.

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
        let mut ib = if b.is_count_only() {
            CmIrBuilder::new_count_only(base_asg)
        } else {
            CmIrBuilder::new(base_asg)
        };
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
        let mut ib = if b.is_count_only() {
            CmIrBuilder::new_count_only(base_asg)
        } else {
            CmIrBuilder::new(base_asg)
        };
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

