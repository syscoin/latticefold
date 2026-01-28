use ark_ff::{BigInteger, PrimeField};

use symphony::dpp_sumcheck::Dr1csBuilder;

use super::gadgets::{alloc_bool, alloc_u2_from_u8, alloc_u7, bool_and, bool_not, bool_or, const_zero};
use super::params::{DIGITS_PER_TRY, LIMB_BASE_U64, LIMB_BITS, LIMBS_U64};
use crate::we_goldilocks_poseidon_f257::GOLDILOCKS_P;

pub(super) fn goldilocks_p_base128_digits_le() -> [u8; LIMBS_U64] {
    let mut out = [0u8; LIMBS_U64];
    let mut t = GOLDILOCKS_P;
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

fn base128_add10<F: PrimeField>(
    b: &mut Dr1csBuilder<F>,
    a: &[usize; LIMBS_U64],
    c: &[usize; LIMBS_U64],
) -> [usize; LIMBS_U64] {
    let mut out = [0usize; LIMBS_U64];
    let mut carry = const_zero::<F>(b);
    for i in 0..LIMBS_U64 {
        let ai = b.assignment[a[i]]
            .into_bigint()
            .to_bytes_le()
            .get(0)
            .copied()
            .unwrap_or(0) as u16;
        let ci = b.assignment[c[i]]
            .into_bigint()
            .to_bytes_le()
            .get(0)
            .copied()
            .unwrap_or(0) as u16;
        let carry_u16 = b.assignment[carry]
            .into_bigint()
            .to_bytes_le()
            .get(0)
            .copied()
            .unwrap_or(0) as u16;
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

fn base128_lt_const10<F: PrimeField>(b: &mut Dr1csBuilder<F>, x: &[usize; LIMBS_U64], c_digits: &[u8; LIMBS_U64]) -> usize {
    // Borrow chain for x - c. Final borrow = 1 iff x < c.
    let mut borrow = const_zero::<F>(b);
    for i in 0..LIMBS_U64 {
        let xi = b.assignment[x[i]]
            .into_bigint()
            .to_bytes_le()
            .get(0)
            .copied()
            .unwrap_or(0) as i16;
        let bi = b.assignment[borrow]
            .into_bigint()
            .to_bytes_le()
            .get(0)
            .copied()
            .unwrap_or(0) as i16;
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

pub(crate) fn sample_goldilocks_coin_unrolled_rejection_8_digits<F: PrimeField>(
    b: &mut Dr1csBuilder<F>,
    digit_vars: &[usize], // length = tries*8
    tries: usize,
) -> ([usize; LIMBS_U64], usize /* found */) {
    assert_eq!(digit_vars.len(), tries * DIGITS_PER_TRY);
    let p_digits = goldilocks_p_base128_digits_le();

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

#[derive(Clone, Debug)]
pub struct GoldilocksRejectionCoinWiring {
    /// Global variable indices of all base-257 digit vars used (tries * 8).
    pub digit_vars: Vec<usize>,
    /// Global variable index of `found` (boolean), enforced to be 1 in the builder.
    pub found_bit: usize,
    /// Global variable indices of the selected coin value `u` as base-128 limbs (little-endian).
    pub coin_limbs: Vec<usize>,
    /// The fixed number of tries assumed by this wiring.
    pub tries: usize,
}

