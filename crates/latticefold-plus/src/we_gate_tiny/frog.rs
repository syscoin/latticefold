use ark_ff::{BigInteger, Field, PrimeField};

use latticefold::transcript::poseidon::F257;
use symphony::dpp_sumcheck::Dr1csBuilder;

use super::coins::frog_p_base128_digits_le;
use super::digits::{
    add_bal16_same_len, alloc_bal16_digit, mul_bal16_long_by_const_rhs, mul_bal16_long_by_long,
    alloc_carry_pm2, alloc_carry_pm128, alloc_carry_pm512, i32_to_f257, u64_bytes_to_bal16_digits_cached,
};
use super::gadgets::{alloc_bool, decompose_existing_byte_var_to_bits};
use super::params::{LIMB_BASE_U64, LIMB_BITS, LIMBS_U64};
use crate::we_frog_poseidon_f257::FROG_P;

#[inline]
fn strict_nowrap_enabled() -> bool {
    std::env::var("LF_STRICT_NOWRAP").ok().as_deref() == Some("1")
}

fn frog_p_bal16_digits_le_const() -> [i8; 17] {
    // Match the balancing convention used by `u64_bytes_to_bal16_digits`:
    // carry_{i+1} = (nibble_i + carry_i >= 8), out_i = nibble_i + carry_i - 16*carry_{i+1}.
    let mut out = [0i8; 17];
    let mut carry: i16 = 0;
    let mut x = FROG_P;
    for i in 0..16 {
        let nib = (x & 0xF) as i16;
        x >>= 4;
        let v = nib + carry;
        if v >= 8 {
            out[i] = (v - 16) as i8;
            carry = 1;
        } else {
            out[i] = v as i8;
            carry = 0;
        }
    }
    out[16] = carry as i8;
    out
}

fn u64_to_bal16_digits_le_const(mut x: u64) -> [i8; 17] {
    // Match `u64_bytes_to_bal16_digits`: balanced digits in [-8,7] with carry in {0,1}.
    let mut out = [0i8; 17];
    let mut carry: i16 = 0;
    for i in 0..16 {
        let nib = (x & 0xF) as i16;
        x >>= 4;
        let v = nib + carry;
        if v >= 8 {
            out[i] = (v - 16) as i8;
            carry = 1;
        } else {
            out[i] = v as i8;
            carry = 0;
        }
    }
    out[16] = carry as i8;
    out
}

/// Allocate a u64 witness value as balanced base-16 digits (len 17) as F257 variables.
///
/// The digits follow the same convention as `u64_bytes_to_bal16_digits` / `u64_to_bal16_digits_le_const`:
/// - digits[0..16] are in [-8,7]
/// - digits[16] is the final carry in {0,1}
fn alloc_u64_as_bal16_digits_witness(b: &mut Dr1csBuilder<F257>, x: u64) -> Vec<usize> {
    let ds = u64_to_bal16_digits_le_const(x);
    let mut out: Vec<usize> = Vec::with_capacity(17);
    for i in 0..16 {
        // NOTE: This function allocates *witness digits* (for q in mul/lincomb reductions).
        // For statement-only arming, do NOT hard-code witness-derived digits as constants.
        out.push(alloc_bal16_digit(b, ds[i]));
    }
    // Final carry is already in {0,1}.
    out.push(alloc_bool::<F257>(b, ds[16] == 1));
    out
}

/// Boundary-only canonicalization gadget:
/// given an unconstrained 64-bit integer `u` as 8 little-endian bytes, produce
/// `(q, z)` such that:
/// - `q ∈ {0,1}`
/// - `u = z + q * p_frog` as an **integer** (no wrap), enforced via base-128 borrows over a
///   bit-derived base-128 limb view of `u`.
///
/// This is the "single subtract" reduction justified by \(2^{64} < 2p\).
/// Takes raw byte variables that are already constrained to be 8-bit.
pub(super) fn reduce_u64_mod_frog_from_byte_vars<F: PrimeField>(
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
        // Range-check diff_i to 7 bits by allocating 7 bools.
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

    // Now compute z limbs as witness and enforce u = z + q*p + base*borrow chain.
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
    //
    // We use `z_limbs` directly as the per-limb "difference", so we do not need an extra `diff_i`
    // variable (and its 7-bit range-check) per limb.
    let p_digits = frog_p_base128_digits_le();
    let mut borrow = b.new_var(F::ZERO);
    b.enforce_var_eq_const(borrow, F::ZERO);
    for i in 0..LIMBS_U64 {
        let ui = ((u >> (LIMB_BITS * i)) & (LIMB_BASE_U64 - 1)) as i16;
        let zi = ((z_u64 >> (LIMB_BITS * i)) & (LIMB_BASE_U64 - 1)) as i16;
        let pi = p_digits[i] as i16;
        let bi = if i == 0 {
            0i16
        } else {
            b.assignment[borrow]
                .into_bigint()
                .to_bytes_le()
                .get(0)
                .copied()
                .unwrap_or(0) as i16
        };
        let rhs = ui - (q_u8 as i16) * pi - bi - zi;
        let borrow_next_u8 = if rhs < 0 { 1u8 } else { 0u8 };
        let borrow_next = if i == LIMBS_U64 - 1 {
            let v = b.new_var(F::ZERO);
            b.enforce_var_eq_const(v, F::ZERO);
            v
        } else {
            alloc_bool::<F>(b, borrow_next_u8 == 1)
        };
        b.enforce_lc_times_one_eq_const(vec![
            (F::ONE, u_limbs[i]),
            (-F::from(p_digits[i] as u64), q),
            (-F::ONE, borrow),
            (F::from(LIMB_BASE_U64), borrow_next),
            (-F::ONE, z_limbs[i]),
        ]);
        borrow = borrow_next;
    }
    b.enforce_var_eq_const(borrow, F::ZERO);

    (q, z_limbs)
}

/// Interpret 8 little-endian byte vars (0..255) as a **canonical** Frog base-field element.
///
/// This enforces that the represented integer `u` satisfies `u < p_frog` (no reduction),
/// and returns `u` as base-128 limbs.
///
/// Use this for transcript-absorbed base-field elements, which are already encoded canonically
/// by `prime_field_to_bytes_le_fixed` in the transcript.
pub(super) fn frog_u64_canonical_from_byte_vars<F: PrimeField>(
    b: &mut Dr1csBuilder<F>,
    u_byte_vars: &[usize; 8],
) -> [usize; LIMBS_U64] {
    let (q, z) = reduce_u64_mod_frog_from_byte_vars::<F>(b, u_byte_vars);
    // Canonical encoding requires no subtraction (u < p), so q == 0.
    b.enforce_var_eq_const(q, F::ZERO);
    z
}

/// Enforce that an 8-byte little-endian integer `u` satisfies `u < p_frog`.
///
/// This is a *cheaper* alternative to `frog_u64_canonical_from_byte_vars` when you do not need
/// the base-128 limbs of `u` (and do not need to compute the reduced value `z`).
///
/// Internally this:
/// - bit-decomposes the bytes (cached)
/// - packs them into 10 base-128 limbs (7-bit each)
/// - runs a base-128 borrow chain on (u - p)
/// - enforces the final borrow is 1 (i.e., u < p)
pub(super) fn frog_u64_enforce_lt_p_from_byte_vars<F: PrimeField>(
    b: &mut Dr1csBuilder<F>,
    u_byte_vars: &[usize; 8],
) {
    // Witness u.
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

    // Decompose bytes into bits (cached per byte var).
    let mut u_bits: Vec<usize> = Vec::with_capacity(64);
    for i in 0..8 {
        let bits = decompose_existing_byte_var_to_bits::<F>(b, u_byte_vars[i]);
        u_bits.extend_from_slice(&bits);
    }
    debug_assert_eq!(u_bits.len(), 64);

    // Pack into base-128 limbs.
    let mut u_limbs = [0usize; LIMBS_U64];
    for j in 0..LIMBS_U64 {
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

    // Borrow chain for u - p. Final borrow == 1 iff u < p.
    let p_digits = frog_p_base128_digits_le();
    let mut borrow = b.new_var(F::ZERO);
    b.enforce_var_eq_const(borrow, F::ZERO);
    for i in 0..LIMBS_U64 {
        let ui = ((u >> (LIMB_BITS * i)) & (LIMB_BASE_U64 - 1)) as i16;
        let pi = p_digits[i] as i16;
        let bi = if i == 0 {
            0i16
        } else {
            b.assignment[borrow]
                .into_bigint()
                .to_bytes_le()
                .get(0)
                .copied()
                .unwrap_or(0) as i16
        };
        debug_assert!(bi == 0 || bi == 1);
        let mut t = ui - pi - bi;
        let borrow_next_u8 = if t < 0 { 1u8 } else { 0u8 };
        if t < 0 {
            t += LIMB_BASE_U64 as i16;
        }
        let diff_i = b.new_var(F::from((t as u64) & (LIMB_BASE_U64 - 1)));
        // Range-check diff_i to 7 bits.
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

        let borrow_next = alloc_bool::<F>(b, borrow_next_u8 == 1);
        // u_i - p_i - borrow_i + base*borrow_{i+1} - diff_i == 0
        b.enforce_lc_times_one_eq_const(vec![
            (F::ONE, u_limbs[i]),
            (-F::from(p_digits[i] as u64), b.one()),
            (-F::ONE, borrow),
            (F::from(LIMB_BASE_U64), borrow_next),
            (-F::ONE, diff_i),
        ]);
        borrow = borrow_next;
    }

    // Enforce u < p: final borrow must be 1.
    b.enforce_var_eq_const(borrow, F::ONE);
}

/// Enforce that a canonical Frog base-field element `u` (encoded as 8 bytes, u < p_frog)
/// lies in the **centered magnitude** range \(|u| <= bound\), meaning:
/// - u ∈ [0, bound]  (non-negative)
/// - OR u ∈ [p_frog - bound, p_frog - 1]  (negative, in centered lift)
///
/// Returns base-128 limbs of `u` (same as `frog_u64_canonical_from_byte_vars`).
pub(super) fn frog_u64_centered_le_bound_from_byte_vars<F: PrimeField>(
    b: &mut Dr1csBuilder<F>,
    u_byte_vars: &[usize; 8],
    bound: u64,
) -> [usize; LIMBS_U64] {
    // First, enforce canonical encoding and get base-128 limbs for u.
    let u_limbs = frog_u64_canonical_from_byte_vars::<F>(b, u_byte_vars);

    // Witness u as u64 for boolean witnesses.
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

    // Helper: compare u <= c using base-128 borrow chain on (c - u).
    fn le_const_base128<F: PrimeField>(
        b: &mut Dr1csBuilder<F>,
        u_limbs: &[usize; LIMBS_U64],
        u_wit: u64,
        c: u64,
    ) -> usize {
        let le = alloc_bool::<F>(b, u_wit <= c);

        let mut borrow = b.new_var(F::ZERO);
        b.enforce_var_eq_const(borrow, F::ZERO);
        for i in 0..LIMBS_U64 {
            let ui = ((u_wit >> (LIMB_BITS * i)) & (LIMB_BASE_U64 - 1)) as i16;
            let ci = ((c >> (LIMB_BITS * i)) & (LIMB_BASE_U64 - 1)) as i16;
            let bi = if i == 0 {
                0i16
            } else {
                b.assignment[borrow]
                    .into_bigint()
                    .to_bytes_le()
                    .get(0)
                    .copied()
                    .unwrap_or(0) as i16
            };
            let mut t = ci - ui - bi;
            let bor_next_u8 = if t < 0 { 1u8 } else { 0u8 };
            if t < 0 {
                t += LIMB_BASE_U64 as i16;
            }
            let diff_i = b.new_var(F::from((t as u64) & (LIMB_BASE_U64 - 1)));
            // Range-check diff_i to 7 bits.
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
            // c_i - u_i - bor_i + base*bor_{i+1} - diff_i == 0
            b.enforce_lc_times_one_eq_const(vec![
                (F::from(ci as u64), b.one()),
                (-F::ONE, u_limbs[i]),
                (-F::ONE, borrow),
                (F::from(LIMB_BASE_U64), bor_next),
                (-F::ONE, diff_i),
            ]);
            borrow = bor_next;
        }
        // le == 1 - final_borrow
        let one_minus_bor = b.new_var(F::ONE - b.assignment[borrow]);
        b.enforce_lc_times_one_eq_const(vec![
            (F::ONE, one_minus_bor),
            (F::ONE, borrow),
            (-F::ONE, b.one()),
        ]);
        b.enforce_lc_times_one_eq_const(vec![(F::ONE, le), (-F::ONE, one_minus_bor)]);
        le
    }

    // Helper: compare u >= c using base-128 borrow chain on (u - c).
    fn ge_const_base128<F: PrimeField>(
        b: &mut Dr1csBuilder<F>,
        u_limbs: &[usize; LIMBS_U64],
        u_wit: u64,
        c: u64,
    ) -> usize {
        let ge = alloc_bool::<F>(b, u_wit >= c);
        let mut borrow = b.new_var(F::ZERO);
        b.enforce_var_eq_const(borrow, F::ZERO);
        for i in 0..LIMBS_U64 {
            let ui = ((u_wit >> (LIMB_BITS * i)) & (LIMB_BASE_U64 - 1)) as i16;
            let ci = ((c >> (LIMB_BITS * i)) & (LIMB_BASE_U64 - 1)) as i16;
            let bi = if i == 0 {
                0i16
            } else {
                b.assignment[borrow]
                    .into_bigint()
                    .to_bytes_le()
                    .get(0)
                    .copied()
                    .unwrap_or(0) as i16
            };
            let mut t = ui - ci - bi;
            let bor_next_u8 = if t < 0 { 1u8 } else { 0u8 };
            if t < 0 {
                t += LIMB_BASE_U64 as i16;
            }
            let diff_i = b.new_var(F::from((t as u64) & (LIMB_BASE_U64 - 1)));
            // Range-check diff_i to 7 bits.
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
            // u_i - c_i - bor_i + base*bor_{i+1} - diff_i == 0
            b.enforce_lc_times_one_eq_const(vec![
                (F::ONE, u_limbs[i]),
                (-F::from(ci as u64), b.one()),
                (-F::ONE, borrow),
                (F::from(LIMB_BASE_U64), bor_next),
                (-F::ONE, diff_i),
            ]);
            borrow = bor_next;
        }
        // ge == 1 - final_borrow
        let one_minus_bor = b.new_var(F::ONE - b.assignment[borrow]);
        b.enforce_lc_times_one_eq_const(vec![
            (F::ONE, one_minus_bor),
            (F::ONE, borrow),
            (-F::ONE, b.one()),
        ]);
        b.enforce_lc_times_one_eq_const(vec![(F::ONE, ge), (-F::ONE, one_minus_bor)]);
        ge
    }

    let le_bound = le_const_base128::<F>(b, &u_limbs, u, bound);
    let p_minus_bound = FROG_P.saturating_sub(bound);
    let ge_p_minus_bound = ge_const_base128::<F>(b, &u_limbs, u, p_minus_bound);

    // ok = le_bound OR ge_p_minus_bound
    let and = b.new_var(b.assignment[le_bound] * b.assignment[ge_p_minus_bound]);
    b.enforce_mul(le_bound, ge_p_minus_bound, and);
    let ok = b.new_var(b.assignment[le_bound] + b.assignment[ge_p_minus_bound] - b.assignment[and]);
    b.enforce_lc_times_one_eq_const(vec![
        (F::ONE, ok),
        (-F::ONE, le_bound),
        (-F::ONE, ge_p_minus_bound),
        (F::ONE, and),
    ]);
    b.enforce_var_eq_const(ok, F::ONE);

    u_limbs
}

fn alloc_u8_var<F: PrimeField>(b: &mut Dr1csBuilder<F>, v: u8) -> usize {
    let x = b.new_var(F::from(v as u64));
    let _ = decompose_existing_byte_var_to_bits::<F>(b, x);
    x
}

fn pad_bal16(b: &mut Dr1csBuilder<F257>, mut v: Vec<usize>, target_len: usize) -> Vec<usize> {
    if v.len() >= target_len {
        return v;
    }
    let z = b.zero_var();
    v.extend(std::iter::repeat(z).take(target_len - v.len()));
    v
}

fn enforce_bal16_vec_eq(b: &mut Dr1csBuilder<F257>, a: &[usize], c: &[usize]) {
    debug_assert_eq!(a.len(), c.len());
    for (&ai, &ci) in a.iter().zip(c.iter()) {
        b.enforce_lc_times_one_eq_const(vec![(F257::ONE, ai), (-F257::ONE, ci)]);
    }
}

fn frog_p_bytes_le() -> [u8; 8] {
    FROG_P.to_le_bytes()
}

/// Enforce `prod == q*p + r` in balanced base-16 digits *without* materializing `q*p` or `q*p+r`.
///
/// This is a big `alloc_bool` saver in hot paths (mul and mul-by-const), because it avoids
/// allocating and range-checking intermediate digit vectors that are only used for equality.
fn enforce_prod_eq_qp_plus_r_bal16(
    b: &mut Dr1csBuilder<F257>,
    prod_d: &[usize],
    q_d: &[usize],
    p_d_const: &[i8],
    r_d: &[usize],
) {
    let zero = b.zero_var();
    let max_len = prod_d
        .len()
        .max(r_d.len())
        .max(q_d.len().saturating_add(p_d_const.len()).saturating_sub(1))
        + 1; // one digit of headroom for the final carry becoming 0

    // Pad prod and r to max_len with 0 digits.
    let prod_pad = pad_bal16(b, prod_d.to_vec(), max_len);
    let r_pad = pad_bal16(b, r_d.to_vec(), max_len);

    // carry_0 = 0 (fixed, no range-check needed).
    let mut carry_var = b.new_var(F257::ZERO);
    b.enforce_var_eq_const(carry_var, F257::ZERO);
    let mut carry_i32: i32 = 0;

    if strict_nowrap_enabled() {
        // Conservative no-wrap check for the per-digit constraint:
        //
        //   carry + prod_k - r_k - Σ_i (q_i * p_{k-i}) - 16*carry_next == 0   (in F257)
        //
        // Bounds:
        // - carry, carry_next ∈ [-128,127]
        // - prod_k, r_k ∈ [-8,7]
        // - q_i ∈ [-8,7], p_j ∈ [-8,7]  =>  |q_i*p_j| ≤ 64
        //
        // With up to T_k terms in the Σ, a sufficient condition to avoid wrap is:
        //   max_abs_LHS < 257.
        //
        // This is intentionally conservative; if it fails, the constraint may only enforce a
        // mod-257 relation (potential soundness hazard).
        let mut prev = 0i32;
        let ql = q_d.len();
        let pl = p_d_const.len();
        for k in 0..max_len {
            let t = ((k + 1).min(ql)).min(pl) as i32;
            let ck = 128i32; // carry_next is pm128 here
            let max_abs = prev + 8 + 8 + 64 * t + 16 * ck;
            debug_assert!(
                max_abs < 257,
                "LF_STRICT_NOWRAP failed (enforce_prod_eq_qp_plus_r_bal16): k={k} terms={t} prev_bound={prev} carry_next_bound={ck} => max_abs_LHS={max_abs} >= 257"
            );
            prev = ck;
        }
    }

    for k in 0..max_len {
        // Witness the exact carry update from the already-witnessed digits.
        let prod_k = super::digits::f257_to_i32_bal(b.assignment[prod_pad[k]]);
        let r_k = super::digits::f257_to_i32_bal(b.assignment[r_pad[k]]);
        let mut sum: i32 = carry_i32 + prod_k - r_k;

        for i in 0..q_d.len() {
            if i > k {
                break;
            }
            let j = k - i;
            if j >= p_d_const.len() {
                continue;
            }
            let q_i = super::digits::f257_to_i32_bal(b.assignment[q_d[i]]);
            sum -= q_i * (p_d_const[j] as i32);
        }

        debug_assert!(
            sum % 16 == 0,
            "base-16 carry check not divisible: sum={sum} at k={k}"
        );
        let carry_next: i32 = sum / 16;
        debug_assert!(
            (-128..=127).contains(&carry_next),
            "carry out of pm128 bound: {carry_next} at k={k} (sum={sum})"
        );

        // Allocate next carry with the smallest sufficient bound.
        // Statement-only arming: use a fixed pm128 bound (no witness-dependent gadget selection).
        let carry_next_var = alloc_carry_pm128(b, carry_next);

        // Constrain: carry + prod_k - r_k - Σ(q_i * p_{k-i}) - 16*carry_next = 0
        let mut lc: Vec<(F257, usize)> = Vec::with_capacity(4 + q_d.len());
        lc.push((F257::ONE, carry_var));
        lc.push((F257::ONE, prod_pad[k]));
        lc.push((-F257::ONE, r_pad[k]));

        for i in 0..q_d.len() {
            if i > k {
                break;
            }
            let j = k - i;
            if j >= p_d_const.len() {
                continue;
            }
            let coeff = i32_to_f257(-(p_d_const[j] as i32)); // subtract q_i*p_j
            if coeff != F257::ZERO {
                lc.push((coeff, q_d[i]));
            }
        }

        lc.push((-F257::from(16u64), carry_next_var));
        b.enforce_lc_times_one_eq_const(lc);

        carry_var = carry_next_var;
        carry_i32 = carry_next;
    }

    // Final carry must be 0.
    b.enforce_var_eq_const(carry_var, F257::ZERO);
    // Also ensure the padded tails we introduced are actually zero digits (soundness belt+braces).
    b.enforce_var_eq_const(zero, F257::ZERO);
}

/// Enforce `a + c = q*p + r` over balanced base-16 digits with a tight carry bound.
fn enforce_add_mod_p_relation_bal16(
    b: &mut Dr1csBuilder<F257>,
    a_d: &[usize],
    c_d: &[usize],
    r_d: &[usize],
    q: usize,
    q_u8: u8,
    p_d_const: &[i8; 17],
) {
    debug_assert!(q_u8 == 0 || q_u8 == 1);
    let a = pad_bal16(b, a_d.to_vec(), 18);
    let c = pad_bal16(b, c_d.to_vec(), 18);
    let r = pad_bal16(b, r_d.to_vec(), 18);

    let mut carry_var = b.zero_var();
    let mut carry_i32: i32 = 0;

    for k in 0..18 {
        let ak = super::digits::f257_to_i32_bal(b.assignment[a[k]]);
        let ck = super::digits::f257_to_i32_bal(b.assignment[c[k]]);
        let rk = super::digits::f257_to_i32_bal(b.assignment[r[k]]);
        let pk = if k < 17 { p_d_const[k] as i32 } else { 0 };

        let sum = carry_i32 + ak + ck - rk - (q_u8 as i32) * pk;
        debug_assert!(sum % 16 == 0, "add_mod_p carry not divisible: sum={sum} k={k}");
        let carry_next = sum / 16;
        debug_assert!(
            (-2..=2).contains(&carry_next),
            "add_mod_p carry out of pm2: {carry_next} (sum={sum}) at k={k}"
        );
        let carry_next_var = alloc_carry_pm2(b, carry_next);

        let mut lc: Vec<(F257, usize)> = Vec::with_capacity(6);
        lc.push((F257::ONE, carry_var));
        lc.push((F257::ONE, a[k]));
        lc.push((F257::ONE, c[k]));
        lc.push((-F257::ONE, r[k]));
        if k < 17 && p_d_const[k] != 0 {
            lc.push((-i32_to_f257(p_d_const[k] as i32), q));
        }
        lc.push((-F257::from(16u64), carry_next_var));
        b.enforce_lc_times_one_eq_const(lc);

        carry_var = carry_next_var;
        carry_i32 = carry_next;
    }
    b.enforce_var_eq_const(carry_var, F257::ZERO);
}

/// Enforce `a + q*p = c + r` over balanced base-16 digits with a tight carry bound.
fn enforce_sub_mod_p_relation_bal16(
    b: &mut Dr1csBuilder<F257>,
    a_d: &[usize],
    c_d: &[usize],
    r_d: &[usize],
    q: usize,
    q_u8: u8,
    p_d_const: &[i8; 17],
) {
    debug_assert!(q_u8 == 0 || q_u8 == 1);
    let a = pad_bal16(b, a_d.to_vec(), 18);
    let c = pad_bal16(b, c_d.to_vec(), 18);
    let r = pad_bal16(b, r_d.to_vec(), 18);

    let mut carry_var = b.zero_var();
    let mut carry_i32: i32 = 0;

    for k in 0..18 {
        let ak = super::digits::f257_to_i32_bal(b.assignment[a[k]]);
        let ck = super::digits::f257_to_i32_bal(b.assignment[c[k]]);
        let rk = super::digits::f257_to_i32_bal(b.assignment[r[k]]);
        let pk = if k < 17 { p_d_const[k] as i32 } else { 0 };

        let sum = carry_i32 + ak + (q_u8 as i32) * pk - ck - rk;
        debug_assert!(sum % 16 == 0, "sub_mod_p carry not divisible: sum={sum} k={k}");
        let carry_next = sum / 16;
        debug_assert!(
            (-2..=2).contains(&carry_next),
            "sub_mod_p carry out of pm2: {carry_next} (sum={sum}) at k={k}"
        );
        let carry_next_var = alloc_carry_pm2(b, carry_next);

        let mut lc: Vec<(F257, usize)> = Vec::with_capacity(6);
        lc.push((F257::ONE, carry_var));
        lc.push((F257::ONE, a[k]));
        lc.push((-F257::ONE, c[k]));
        lc.push((-F257::ONE, r[k]));
        if k < 17 && p_d_const[k] != 0 {
            lc.push((i32_to_f257(p_d_const[k] as i32), q));
        }
        lc.push((-F257::from(16u64), carry_next_var));
        b.enforce_lc_times_one_eq_const(lc);

        carry_var = carry_next_var;
        carry_i32 = carry_next;
    }
    b.enforce_var_eq_const(carry_var, F257::ZERO);
}

/// General Frog-field multiplication gadget (64-bit prime field) inside the tiny field.
///
/// Inputs are canonical u64 encodings as 8 little-endian byte vars (0..255), representing
/// field elements in \([0, p)\).
///
/// This enforces:
/// \[
///   a \cdot b = q \cdot p + r,\quad 0 \le r < p
/// \]
/// using balanced-base16 digit arithmetic (sound in F257), and returns `r` as 8 byte vars.
pub(super) fn frog_mul_mod_p_from_byte_vars(
    b: &mut Dr1csBuilder<F257>,
    a_bytes: &[usize; 8],
    b_bytes: &[usize; 8],
) -> [usize; 8] {
    // Ensure inputs are canonical (<p).
    frog_u64_enforce_lt_p_from_byte_vars::<F257>(b, a_bytes);
    frog_u64_enforce_lt_p_from_byte_vars::<F257>(b, b_bytes);
    frog_mul_mod_p_from_byte_vars_assume_canonical(b, a_bytes, b_bytes)
}

#[inline]
fn frog_mul_mod_p_from_byte_vars_assume_canonical(
    b: &mut Dr1csBuilder<F257>,
    a_bytes: &[usize; 8],
    b_bytes: &[usize; 8],
) -> [usize; 8] {
    let _prev = b.profile_enter("frog::mul_mod_p");

    // Witness compute.
    let mut ab = [0u8; 8];
    let mut bb = [0u8; 8];
    for i in 0..8 {
        ab[i] = b.assignment[a_bytes[i]]
            .into_bigint()
            .to_bytes_le()
            .get(0)
            .copied()
            .unwrap_or(0);
        bb[i] = b.assignment[b_bytes[i]]
            .into_bigint()
            .to_bytes_le()
            .get(0)
            .copied()
            .unwrap_or(0);
    }
    let a_u = u64::from_le_bytes(ab);
    let b_u = u64::from_le_bytes(bb);
    let prod: u128 = (a_u as u128) * (b_u as u128);
    let q_u: u64 = (prod / (FROG_P as u128)) as u64;
    let r_u: u64 = (prod % (FROG_P as u128)) as u64;

    // Allocate r as byte vars (output). The quotient `q` is internal; we allocate it directly as
    // balanced base-16 digits to avoid the expensive byte->digit conversion.
    let r_bytes_u8 = r_u.to_le_bytes();
    let mut r_bytes = [0usize; 8];
    for i in 0..8 {
        r_bytes[i] = alloc_u8_var::<F257>(b, r_bytes_u8[i]);
    }
    // Enforce r is canonical (<p).
    frog_u64_enforce_lt_p_from_byte_vars::<F257>(b, &r_bytes);

    // Convert a,b,r and p to balanced-base16 digits.
    let a_d = u64_bytes_to_bal16_digits_cached(b, *a_bytes);
    let b_d = u64_bytes_to_bal16_digits_cached(b, *b_bytes);
    let r_d = u64_bytes_to_bal16_digits_cached(b, r_bytes);
    let q_d = alloc_u64_as_bal16_digits_witness(b, q_u);

    // Compute prod_digits = a*b (balanced digits, with headroom/carry already enforced in gadget).
    let prod_d = mul_bal16_long_by_long(b, &a_d, &b_d);
    let p_d_const = frog_p_bal16_digits_le_const();
    enforce_prod_eq_qp_plus_r_bal16(b, &prod_d, &q_d, &p_d_const, &r_d);

    let out = r_bytes;
    b.profile_exit(_prev);
    out
}

/// Multiply a canonical Frog scalar `x` by a **known constant** `c` (as u64 in `[0,p)`),
/// returning a canonical Frog scalar (8-byte little-endian) for `x*c mod p`.
///
/// This is substantially cheaper than `frog_mul_mod_p_from_byte_vars` because it avoids all
/// digit×digit multiplications in the 64-bit integer product checks.
pub(super) fn frog_mul_const_mod_p_from_byte_vars(
    b: &mut Dr1csBuilder<F257>,
    x_bytes: &[usize; 8],
    c: u64,
) -> [usize; 8] {
    assert!(c < FROG_P, "frog_mul_const_mod_p_from_byte_vars requires c < p");
    frog_u64_enforce_lt_p_from_byte_vars::<F257>(b, x_bytes);
    frog_mul_const_mod_p_from_byte_vars_assume_canonical(b, x_bytes, c)
}

#[inline]
fn frog_mul_const_mod_p_from_byte_vars_assume_canonical(
    b: &mut Dr1csBuilder<F257>,
    x_bytes: &[usize; 8],
    c: u64,
) -> [usize; 8] {
    let _prev = b.profile_enter("frog::mul_const_mod_p");

    // Witness compute.
    let mut xb = [0u8; 8];
    for i in 0..8 {
        xb[i] = b.assignment[x_bytes[i]]
            .into_bigint()
            .to_bytes_le()
            .get(0)
            .copied()
            .unwrap_or(0);
    }
    let x_u = u64::from_le_bytes(xb);
    let prod: u128 = (x_u as u128) * (c as u128);
    let q_u: u64 = (prod / (FROG_P as u128)) as u64;
    let r_u: u64 = (prod % (FROG_P as u128)) as u64;

    // Allocate r as byte vars (output). The quotient `q` is internal; we allocate it directly as
    // balanced base-16 digits to avoid the expensive byte->digit conversion.
    let r_bytes_u8 = r_u.to_le_bytes();
    let mut r_bytes = [0usize; 8];
    for i in 0..8 {
        r_bytes[i] = alloc_u8_var::<F257>(b, r_bytes_u8[i]);
    }
    frog_u64_enforce_lt_p_from_byte_vars::<F257>(b, &r_bytes);

    // Convert x,r to bal16 digits (vars). `q` is allocated directly as balanced digits.
    let x_d = u64_bytes_to_bal16_digits_cached(b, *x_bytes);
    let r_d = u64_bytes_to_bal16_digits_cached(b, r_bytes);
    let q_d = alloc_u64_as_bal16_digits_witness(b, q_u);

    // Constant bal16 digits for c and p, computed directly (no variables/constraints).
    let c_d_const = u64_to_bal16_digits_le_const(c);
    let p_d_const = frog_p_bal16_digits_le_const();

    // prod_digits = x*c  (const-RHS)
    let prod_d = mul_bal16_long_by_const_rhs(b, &x_d, &c_d_const);
    enforce_prod_eq_qp_plus_r_bal16(b, &prod_d, &q_d, &p_d_const, &r_d);

    let out = r_bytes;
    b.profile_exit(_prev);
    out
}

/// General Frog-field addition gadget inside F257.
///
/// Enforces `r = (a + c) mod p` for canonical `a,c < p`, returning canonical `r` as 8 bytes.
pub(super) fn frog_add_mod_p_from_byte_vars(
    b: &mut Dr1csBuilder<F257>,
    a_bytes: &[usize; 8],
    c_bytes: &[usize; 8],
) -> [usize; 8] {
    frog_u64_enforce_lt_p_from_byte_vars::<F257>(b, a_bytes);
    frog_u64_enforce_lt_p_from_byte_vars::<F257>(b, c_bytes);
    frog_add_mod_p_from_byte_vars_assume_canonical(b, a_bytes, c_bytes)
}

#[inline]
fn frog_add_mod_p_from_byte_vars_assume_canonical(
    b: &mut Dr1csBuilder<F257>,
    a_bytes: &[usize; 8],
    c_bytes: &[usize; 8],
) -> [usize; 8] {
    let _prev = b.profile_enter("frog::add_mod_p");

    let mut ab = [0u8; 8];
    let mut cb = [0u8; 8];
    for i in 0..8 {
        ab[i] = b.assignment[a_bytes[i]]
            .into_bigint()
            .to_bytes_le()
            .get(0)
            .copied()
            .unwrap_or(0);
        cb[i] = b.assignment[c_bytes[i]]
            .into_bigint()
            .to_bytes_le()
            .get(0)
            .copied()
            .unwrap_or(0);
    }
    let a_u = u64::from_le_bytes(ab);
    let c_u = u64::from_le_bytes(cb);
    let sum = (a_u as u128) + (c_u as u128);
    let q_u8: u8 = if sum >= (FROG_P as u128) { 1 } else { 0 };
    let r_u: u64 = if q_u8 == 1 { (sum - (FROG_P as u128)) as u64 } else { sum as u64 };

    // `q_u8` is a witness-known 0/1 selector; use cached constants instead of allocating a boolean.
    let q = if q_u8 == 1 { b.one() } else { b.zero_var() };
    let r_bytes_u8 = r_u.to_le_bytes();
    let mut r_bytes = [0usize; 8];
    for i in 0..8 {
        r_bytes[i] = alloc_u8_var::<F257>(b, r_bytes_u8[i]);
    }
    frog_u64_enforce_lt_p_from_byte_vars::<F257>(b, &r_bytes);

    let p_d0_const = frog_p_bal16_digits_le_const();
    let a_d0 = u64_bytes_to_bal16_digits_cached(b, *a_bytes);
    let c_d0 = u64_bytes_to_bal16_digits_cached(b, *c_bytes);
    let r_d0 = u64_bytes_to_bal16_digits_cached(b, r_bytes);
    enforce_add_mod_p_relation_bal16(b, &a_d0, &c_d0, &r_d0, q, q_u8, &p_d0_const);

    let out = r_bytes;
    b.profile_exit(_prev);
    out
}

/// General Frog-field subtraction gadget inside F257.
///
/// Enforces `r = (a - c) mod p` for canonical `a,c < p`, returning canonical `r` as 8 bytes.
pub(super) fn frog_sub_mod_p_from_byte_vars(
    b: &mut Dr1csBuilder<F257>,
    a_bytes: &[usize; 8],
    c_bytes: &[usize; 8],
) -> [usize; 8] {
    frog_u64_enforce_lt_p_from_byte_vars::<F257>(b, a_bytes);
    frog_u64_enforce_lt_p_from_byte_vars::<F257>(b, c_bytes);
    frog_sub_mod_p_from_byte_vars_assume_canonical(b, a_bytes, c_bytes)
}

#[inline]
fn frog_sub_mod_p_from_byte_vars_assume_canonical(
    b: &mut Dr1csBuilder<F257>,
    a_bytes: &[usize; 8],
    c_bytes: &[usize; 8],
) -> [usize; 8] {
    let _prev = b.profile_enter("frog::sub_mod_p");

    let mut ab = [0u8; 8];
    let mut cb = [0u8; 8];
    for i in 0..8 {
        ab[i] = b.assignment[a_bytes[i]]
            .into_bigint()
            .to_bytes_le()
            .get(0)
            .copied()
            .unwrap_or(0);
        cb[i] = b.assignment[c_bytes[i]]
            .into_bigint()
            .to_bytes_le()
            .get(0)
            .copied()
            .unwrap_or(0);
    }
    let a_u = u64::from_le_bytes(ab);
    let c_u = u64::from_le_bytes(cb);
    let (q_u8, r_u) = if a_u >= c_u {
        (0u8, a_u - c_u)
    } else {
        (1u8, (a_u as u128 + (FROG_P as u128) - (c_u as u128)) as u64)
    };

    // `q_u8` is a witness-known 0/1 selector; use cached constants instead of allocating a boolean.
    let q = if q_u8 == 1 { b.one() } else { b.zero_var() };
    let r_bytes_u8 = r_u.to_le_bytes();
    let mut r_bytes = [0usize; 8];
    for i in 0..8 {
        r_bytes[i] = alloc_u8_var::<F257>(b, r_bytes_u8[i]);
    }
    frog_u64_enforce_lt_p_from_byte_vars::<F257>(b, &r_bytes);

    let a_d0 = u64_bytes_to_bal16_digits_cached(b, *a_bytes);
    let c_d0 = u64_bytes_to_bal16_digits_cached(b, *c_bytes);
    let r_d0 = u64_bytes_to_bal16_digits_cached(b, r_bytes);
    let p_d0_const = frog_p_bal16_digits_le_const();
    enforce_sub_mod_p_relation_bal16(b, &a_d0, &c_d0, &r_d0, q, q_u8, &p_d0_const);

    let out = r_bytes;
    b.profile_exit(_prev);
    out
}

fn frog_zero_bytes(b: &mut Dr1csBuilder<F257>) -> [usize; 8] {
    // Allocate a single shared 0-byte variable and reuse it for all 8 limbs.
    //
    // Important: we intentionally do NOT eagerly bit-decompose this byte. If/when a downstream
    // gadget truly needs bits (e.g. digit conversion), `decompose_existing_byte_var_to_bits`
    // will do it once and cache it by variable index. Eager decomposition here is pure overhead
    // because `0` is already fixed by a constant constraint.
    let z0 = b.new_var(F257::ZERO);
    b.enforce_var_eq_const(z0, F257::ZERO);
    [z0; 8]
}

fn frog_one_bytes(b: &mut Dr1csBuilder<F257>) -> [usize; 8] {
    let mut o = frog_zero_bytes(b);
    o[0] = b.new_var(F257::ONE);
    b.enforce_var_eq_const(o[0], F257::ONE);
    o
}

fn frog_from_u64_const_bytes(b: &mut Dr1csBuilder<F257>, c: u64) -> [usize; 8] {
    let cb = c.to_le_bytes();
    let mut out = [0usize; 8];
    for i in 0..8 {
        out[i] = alloc_u8_var::<F257>(b, cb[i]);
    }
    frog_u64_enforce_lt_p_from_byte_vars::<F257>(b, &out);
    out
}

fn frog_add_many_mod_p(b: &mut Dr1csBuilder<F257>, terms: &[[usize; 8]]) -> [usize; 8] {
    let mut acc = frog_zero_bytes(b);
    for t in terms {
        acc = frog_add_mod_p_from_byte_vars_assume_canonical(b, &acc, t);
    }
    acc
}

/// Negacyclic ring multiplication for `d=64` (FrogRing64), where coefficients are canonical Frog scalars.
///
/// Returns `c = a*b mod (X^64 + 1)` as 64 canonical Frog scalars (each 8 bytes).
///
/// This uses the same **Toom-4** block structure as the native WE gate (`we_gate_arith.rs`),
/// but implemented over our byte-based Frog-field gadgets. The critical requirement is that
/// interpolation uses `frog_mul_const_mod_p_from_byte_vars` (cheap const-mul), not full mul.
pub(super) fn ring_mul_negacyclic_toom4_d64(
    b: &mut Dr1csBuilder<F257>,
    a: &[[usize; 8]; 64],
    c: &[[usize; 8]; 64],
) -> [[usize; 8]; 64] {
    ring_mul_negacyclic_d64_impl::<false>(b, a, c)
}

/// Negacyclic ring multiplication for `d=64` using a Karatsuba/Toom-2 top-level split.
///
/// This is typically more circuit-friendly than Toom-4 because it uses fewer evaluation points
/// and much simpler interpolation (mostly ±1 coefficients).
pub(super) fn ring_mul_negacyclic_karatsuba_d64(
    b: &mut Dr1csBuilder<F257>,
    a: &[[usize; 8]; 64],
    c: &[[usize; 8]; 64],
) -> [[usize; 8]; 64] {
    ring_mul_negacyclic_d64_impl::<true>(b, a, c)
}

fn ring_mul_negacyclic_d64_impl<const KARATSUBA: bool>(
    b: &mut Dr1csBuilder<F257>,
    a: &[[usize; 8]; 64],
    c: &[[usize; 8]; 64],
) -> [[usize; 8]; 64] {
    let _prev = b.profile_enter("frog::ring_mul_negacyclic_toom4_d64");
    // NOTE: We intentionally do NOT canonical-validate all 128 inputs here.
    //
    // Canonicality should be enforced at the *true* byte-boundaries (transcript absorb / IO),
    // not redundantly inside every internal ring multiplication. This function itself reduces
    // and outputs canonical Frog elements where needed by downstream gadgets.
    // Precomputed inv(Vandermonde(0,±1,±2,±3)) entries as nums/720.
    const NUMS: [[i64; 7]; 7] = [
        [720, 0, 0, 0, 0, 0, 0],
        [0, 540, -540, -108, 108, 12, -12],
        [-980, 540, 540, -54, -54, 4, 4],
        [0, -195, 195, 120, -120, -15, 15],
        [280, -195, -195, 60, 60, -5, -5],
        [0, 15, -15, -12, 12, 3, -3],
        [-20, 15, 15, -6, -6, 1, 1],
    ];
    // inv720 mod p, precomputed in host below (witness-time), but fixed as a constant operand.
    let inv720 = {
        // Compute inv720 in the host field (u64) using extended Euclid on integers.
        fn inv_mod_u64(a: u64, p: u64) -> u64 {
            let mut t0: i128 = 0;
            let mut t1: i128 = 1;
            let mut r0: i128 = p as i128;
            let mut r1: i128 = a as i128;
            while r1 != 0 {
                let q = r0 / r1;
                (t0, t1) = (t1, t0 - q * t1);
                (r0, r1) = (r1, r0 - q * r1);
            }
            debug_assert!(r0 == 1 || r0 == -1);
            let mut t = t0;
            if t < 0 {
                t += p as i128;
            }
            (t as u128 % (p as u128)) as u64
        }
        inv_mod_u64(720u64, FROG_P)
    };

    let pts: [i64; 7] = [0, 1, -1, 2, -2, 3, -3];

    let zero = frog_zero_bytes(b);

    #[inline]
    fn mod_i64_to_u64(x: i64, p: u64) -> u64 {
        let p_i = p as i128;
        let mut r = (x as i128) % p_i;
        if r < 0 {
            r += p_i;
        }
        r as u64
    }

    #[inline]
    fn modmul_i64_u64(x: i64, y: u64, p: u64) -> u64 {
        let xm = mod_i64_to_u64(x, p);
        ((xm as u128) * (y as u128) % (p as u128)) as u64
    }

    #[inline]
    fn div_rem_u192_by_u64(n2: u64, n1: u64, n0: u64, d: u64) -> (u128, u64) {
        debug_assert!(d != 0);
        // Long division in base 2^64 for a 3-limb numerator.
        let mut rem: u128 = 0;
        let limbs = [n2, n1, n0];
        let mut q = [0u64; 3];
        for (i, &limb) in limbs.iter().enumerate() {
            let cur = (rem << 64) | (limb as u128);
            let qi = cur / (d as u128);
            let ri = cur % (d as u128);
            debug_assert!(qi <= (u64::MAX as u128));
            q[i] = qi as u64;
            rem = ri;
        }
        // For our use-cases, the quotient fits in 128 bits (indeed < 2^67).
        debug_assert_eq!(q[0], 0, "quotient unexpectedly needs >128 bits");
        let q_u128 = ((q[1] as u128) << 64) | (q[2] as u128);
        (q_u128, rem as u64)
    }

    fn u128_to_bal16_digits_le_const(mut x: u128, n_nibbles: usize) -> Vec<i8> {
        // Balanced digits in [-8,7], plus a final carry digit in {0,1}.
        let mut out: Vec<i8> = vec![0i8; n_nibbles + 1];
        let mut carry: i16 = 0;
        for i in 0..n_nibbles {
            let nib = (x & 0xF) as i16;
            x >>= 4;
            let v = nib + carry;
            if v >= 8 {
                out[i] = (v - 16) as i8;
                carry = 1;
            } else {
                out[i] = v as i8;
                carry = 0;
            }
        }
        debug_assert!(x == 0, "value does not fit in requested nibble length");
        out[n_nibbles] = carry as i8;
        out
    }

    fn alloc_u128_as_bal16_digits_witness(
        b: &mut Dr1csBuilder<F257>,
        x: u128,
        n_nibbles: usize,
    ) -> Vec<usize> {
        let ds = u128_to_bal16_digits_le_const(x, n_nibbles);
        let mut out: Vec<usize> = Vec::with_capacity(n_nibbles + 1);
        for i in 0..n_nibbles {
            // Statement-only arming: do NOT hard-code witness-derived digits as constants.
            out.push(alloc_bal16_digit(b, ds[i]));
        }
        // Final carry in {0,1}.
        out.push(alloc_bool::<F257>(b, ds[n_nibbles] == 1));
        out
    }

    /// Compute `Σ_i (coeffs[i] * evals[i]) mod p`, where:
    /// - evals[i] are canonical Frog elements (8 bytes each),
    /// - coeffs[i] are u64 constants in [0,p).
    ///
    /// This avoids per-term modular reduction: it accumulates the integer products in bal16 digit
    /// space, then performs a single `S = q*p + r` reduction.
    fn lincomb7_mod_p_from_canonical_evals(
        b: &mut Dr1csBuilder<F257>,
        evals: &[[usize; 8]; 7],
        evals_d: &[Vec<usize>; 7],
        coeffs: &[u64; 7],
    ) -> [usize; 8] {
        // Use the streaming-convolution implementation for N=7.
        lincomb_mod_p_from_canonical_evals::<7>(b, evals, evals_d, coeffs)
    }

    /// Generic linear combination modulo p with one final reduction.
    ///
    /// Like `lincomb7_mod_p_from_canonical_evals` but for a fixed arity `N`.
    fn lincomb_mod_p_from_canonical_evals<const N: usize>(
        b: &mut Dr1csBuilder<F257>,
        evals: &[[usize; 8]; N],
        evals_d: &[Vec<usize>; N],
        coeffs: &[u64; N],
    ) -> [usize; 8] {
        // Witness compute q, r for Σ coeff_i * eval_i.
        let mut lo: u128 = 0;
        let mut hi: u64 = 0;
        for i in 0..N {
            let coeff = coeffs[i];
            if coeff == 0 {
                continue;
            }
            let mut eb = [0u8; 8];
            for j in 0..8 {
                eb[j] = b.assignment[evals[i][j]]
                    .into_bigint()
                    .to_bytes_le()
                    .get(0)
                    .copied()
                    .unwrap_or(0);
            }
            let e_u = u64::from_le_bytes(eb);
            let term = (e_u as u128) * (coeff as u128);
            let (new_lo, carry) = lo.overflowing_add(term);
            lo = new_lo;
            if carry {
                hi += 1;
            }
        }
        debug_assert!(hi <= (N as u64));
        let n2 = hi;
        let n1 = (lo >> 64) as u64;
        let n0 = (lo & (u64::MAX as u128)) as u64;
        let (q_u128, r_u64) = div_rem_u192_by_u64(n2, n1, n0, FROG_P);

        // Allocate r as bytes and enforce canonical.
        let r_bytes_u8 = r_u64.to_le_bytes();
        let mut r_bytes = [0usize; 8];
        for j in 0..8 {
            r_bytes[j] = alloc_u8_var::<F257>(b, r_bytes_u8[j]);
        }
        frog_u64_enforce_lt_p_from_byte_vars::<F257>(b, &r_bytes);

        // Constrain Σ (coeff*eval) == q*p + r in bal16 digit space.
        //
        // Instead of materializing each product vector and summing them with many long adds,
        // stream the digit convolution of the whole sum:
        //   acc = Σ_i (eval_i * coeff_i)  (coeff_i are u64 constants)
        // producing balanced digits in [-8,7] with a bounded carry.
        let p_d_const = frog_p_bal16_digits_le_const();
        const TARGET_LEN: usize = 43;
        let zero_digit = b.zero_var();

        // Precompute constant bal16 digits for each coefficient.
        let mut coeff_ds: [[i8; 17]; N] = [[0i8; 17]; N];
        for i in 0..N {
            coeff_ds[i] = u64_to_bal16_digits_le_const(coeffs[i]);
        }

        // Streaming convolution for acc digits (len up to 33), then expand final carry.
        let la = evals_d[0].len(); // expected 17
        debug_assert!(la <= 17, "expected canonical u64 digit length");
        let lb = 17usize;
        let base_len = la + lb - 1; // <= 33

        let div_floor = |x: i32, d: i32| -> i32 {
            debug_assert!(d > 0);
            if x >= 0 { x / d } else { -(((-x) + d - 1) / d) }
        };

        let mut acc_digits: Vec<usize> = Vec::with_capacity(TARGET_LEN);
        let mut carry_i32: i32 = 0;
        let mut carry_var = b.new_var(F257::ZERO);
        b.enforce_var_eq_const(carry_var, F257::ZERO);

        #[inline]
        fn f257_from_i8(x: i8) -> F257 {
            if x >= 0 { F257::from(x as u64) } else { -F257::from((-x) as u64) }
        }

        for k in 0..base_len {
            let mut sum: i32 = carry_i32;
            let mut lc: Vec<(F257, usize)> = Vec::new();
            lc.push((F257::ONE, carry_var));

            for i in 0..N {
                if coeffs[i] == 0 {
                    continue;
                }
                let e_d = &evals_d[i];
                for j in 0..la {
                    let t = k as i32 - j as i32;
                    if t < 0 || t >= lb as i32 {
                        continue;
                    }
                    let cd = coeff_ds[i][t as usize];
                    if cd == 0 {
                        continue;
                    }
                    let aval = super::digits::f257_to_i32_bal(b.assignment[e_d[j]]);
                    sum += aval * (cd as i32);
                    lc.push((f257_from_i8(cd), e_d[j]));
                }
            }

            let mut carry_next = div_floor(sum + 8, 16);
            let mut rem = sum - 16 * carry_next;
            while rem > 7 {
                carry_next += 1;
                rem -= 16;
            }
            while rem < -8 {
                carry_next -= 1;
                rem += 16;
            }
            debug_assert!((-8..=7).contains(&rem));
            debug_assert!(
                (-512..=511).contains(&carry_next),
                "carry out of pm512 bound: {carry_next} from sum {sum}"
            );

            let digit_var = alloc_bal16_digit(b, rem as i8);
            // Statement-only arming: use a fixed pm512 bound (no witness-dependent gadget selection).
            let carry_out_var = alloc_carry_pm512(b, carry_next);
            lc.push((-F257::ONE, digit_var));
            lc.push((-F257::from(16u64), carry_out_var));
            b.enforce_lc_times_one_eq_const(lc);

            acc_digits.push(digit_var);
            carry_i32 = carry_next;
            carry_var = carry_out_var;
        }

        // Expand remaining carry into balanced digits with a fixed schedule (statement-only arming).
        //
        // carry is always within [-512,511] here (pm512). After one step:
        //   |carry_next| <= floor((|carry|+8)/16) <= 32
        // after two: <= 2, after three: = 0. So 3 digits suffice and are witness-independent.
        for _ in 0..3 {
            let sum = carry_i32;
            let mut carry_next = div_floor(sum + 8, 16);
            let mut rem = sum - 16 * carry_next;
            while rem > 7 {
                carry_next += 1;
                rem -= 16;
            }
            while rem < -8 {
                carry_next -= 1;
                rem += 16;
            }
            debug_assert!((-8..=7).contains(&rem));
            debug_assert!((-512..=511).contains(&carry_next));

            let rem_digit = alloc_bal16_digit(b, rem as i8);
            // Statement-only arming: use a fixed pm512 bound (no witness-dependent gadget selection).
            let carry_next_var = alloc_carry_pm512(b, carry_next);
            b.enforce_lc_times_one_eq_const(vec![
                (F257::ONE, carry_var),
                (-F257::ONE, rem_digit),
                (-F257::from(16u64), carry_next_var),
            ]);
            acc_digits.push(rem_digit);
            carry_i32 = carry_next;
            carry_var = carry_next_var;
        }
        b.enforce_var_eq_const(carry_var, F257::ZERO);

        // Pad acc to TARGET_LEN.
        let acc = pad_bal16(b, acc_digits, TARGET_LEN);

        // q fits within ~log2(N)+64 bits; represent it with 18 nibbles + final carry => 19 digits.
        let q_d = alloc_u128_as_bal16_digits_witness(b, q_u128, 18);
        let qp_d = mul_bal16_long_by_const_rhs(b, &q_d, &p_d_const);
        let qp_d = pad_bal16(b, qp_d, TARGET_LEN);

        let r_d = u64_bytes_to_bal16_digits_cached(b, r_bytes);
        let r_d = pad_bal16(b, r_d, TARGET_LEN);

        let (sum_d, carry_sum) = add_bal16_same_len(b, &qp_d, &r_d);
        b.enforce_var_eq_const(carry_sum, F257::ZERO);
        enforce_bal16_vec_eq(b, &acc, &sum_d);
        r_bytes
    }

    fn eval_deg3_poly_at_t(
        b: &mut Dr1csBuilder<F257>,
        a0: &[usize; 8],
        a1: &[usize; 8],
        a2: &[usize; 8],
        a3: &[usize; 8],
        t: i64,
    ) -> [usize; 8] {
        // coeffs = [1, t, t^2, t^3] mod p
        let t1 = mod_i64_to_u64(t, FROG_P);
        let t2 = mod_i64_to_u64(t * t, FROG_P);
        let t3 = mod_i64_to_u64(t * t * t, FROG_P);
        let coeffs = [1u64, t1, t2, t3];
        let evals = [*a0, *a1, *a2, *a3];
        let evals_d: [Vec<usize>; 4] = [
            u64_bytes_to_bal16_digits_cached(b, *a0),
            u64_bytes_to_bal16_digits_cached(b, *a1),
            u64_bytes_to_bal16_digits_cached(b, *a2),
            u64_bytes_to_bal16_digits_cached(b, *a3),
        ];
        lincomb_mod_p_from_canonical_evals::<4>(b, &evals, &evals_d, &coeffs)
    }

    // Split into 4 blocks of length 16.
    let a0 = &a[0..16];
    let a1 = &a[16..32];
    let a2 = &a[32..48];
    let a3 = &a[48..64];
    let c0 = &c[0..16];
    let c1 = &c[16..32];
    let c2 = &c[32..48];
    let c3 = &c[48..64];

    // Recursive Toom-4 poly mul for length n=16 over Frog scalars (returns len 31).
    fn poly_mul_toom4_len16(
        b: &mut Dr1csBuilder<F257>,
        a: &Vec<[usize; 8]>,
        c: &Vec<[usize; 8]>,
        inv720: u64,
    ) -> Vec<[usize; 8]> {
        debug_assert_eq!(a.len(), 16);
        debug_assert_eq!(c.len(), 16);
        // Base case at n=1 would be frog_mul_mod_p; but for n=16 we do one Toom-4 level (m=4) then base mults at n=4 via schoolbook.

        let zero = frog_zero_bytes(b);

        // Helper: schoolbook convolution for small n with frog_mul_mod_p (n<=4).
        fn schoolbook(
            b: &mut Dr1csBuilder<F257>,
            a: &[[usize; 8]],
            c: &[[usize; 8]],
        ) -> Vec<[usize; 8]> {
            let n = a.len();
            let mut out: Vec<[usize; 8]> = vec![frog_zero_bytes(b); 2 * n - 1];
            for i in 0..n {
                for j in 0..n {
                    let m = frog_mul_mod_p_from_byte_vars_assume_canonical(b, &a[i], &c[j]);
                    out[i + j] = frog_add_mod_p_from_byte_vars_assume_canonical(b, &out[i + j], &m);
                }
            }
            out
        }

        let m = 4;
        let (a0, rest) = a.split_at(m);
        let (a1, rest) = rest.split_at(m);
        let (a2, a3) = rest.split_at(m);
        let (c0, rest) = c.split_at(m);
        let (c1, rest) = rest.split_at(m);
        let (c2, c3) = rest.split_at(m);

        let pts: [i64; 7] = [0, 1, -1, 2, -2, 3, -3];
        let mut w_eval: Vec<Vec<[usize; 8]>> = Vec::with_capacity(7);
        for &t in &pts {
            let mut ae: Vec<[usize; 8]> = Vec::with_capacity(m);
            let mut ce: Vec<[usize; 8]> = Vec::with_capacity(m);
            for i in 0..m {
                ae.push(eval_deg3_poly_at_t(b, &a0[i], &a1[i], &a2[i], &a3[i], t));
                ce.push(eval_deg3_poly_at_t(b, &c0[i], &c1[i], &c2[i], &c3[i], t));
            }
            w_eval.push(schoolbook(b, &ae, &ce)); // len 7 (2m-1)
        }

        // Interpolate (degree<=6) into blocks j=0..6, each length 2m-1=7:
        // block[j][k] = Σ_i inv_v[j][i] * w_eval[i][k], with inv_v = NUMS/720.
        const NUMS: [[i64; 7]; 7] = [
            [720, 0, 0, 0, 0, 0, 0],
            [0, 540, -540, -108, 108, 12, -12],
            [-980, 540, 540, -54, -54, 4, 4],
            [0, -195, 195, 120, -120, -15, 15],
            [280, -195, -195, 60, 60, -5, -5],
            [0, 15, -15, -12, 12, 3, -3],
            [-20, 15, 15, -6, -6, 1, 1],
        ];

        // Compute block contributions block[j][k] (7×7), then batch overlaps.
        //
        // Here m=4 and block length is (2m-1)=7. Shifts by m cause an overlap of length 3:
        // for each j>=1 and k in 0..2, index (j*m+k) receives contributions from
        // block[j][k] and block[j-1][k+m].
        let mut block: Vec<Vec<[usize; 8]>> = vec![vec![zero; 2 * m - 1]; 7];
        for k in 0..(2 * m - 1) {
            // For fixed k, the 7 evaluation values are reused across all 7 j-blocks.
            let mut evals = [zero; 7];
            for i in 0..7 {
                evals[i] = w_eval[i][k];
            }
            let evals_d: [Vec<usize>; 7] =
                core::array::from_fn(|i| u64_bytes_to_bal16_digits_cached(b, evals[i]));

            for j in 0..7 {
                let mut coeffs = [0u64; 7];
                for i in 0..7 {
                    let n = NUMS[j][i];
                    if n != 0 {
                        coeffs[i] = modmul_i64_u64(n, inv720, FROG_P);
                    }
                }
                block[j][k] = lincomb7_mod_p_from_canonical_evals(b, &evals, &evals_d, &coeffs);
            }
        }

        let mut res: Vec<[usize; 8]> = vec![zero; 2 * 16 - 1]; // len 31
        // j=0 has no left-overlap.
        for k in 0..(2 * m - 1) {
            res[k] = block[0][k];
        }
        for j in 1..7 {
            // Overlap k=0..2: sum of current block and previous block's tail.
            for k in 0..(2 * m - 1 - m) {
                let idx = j * m + k;
                res[idx] = frog_add_mod_p_from_byte_vars_assume_canonical(b, &block[j][k], &block[j - 1][k + m]);
            }
            // Non-overlap k=3..6: direct assignment.
            for k in (2 * m - 1 - m)..(2 * m - 1) {
                let idx = j * m + k;
                if idx < res.len() {
                    res[idx] = block[j][k];
                }
            }
        }
        res
    }

    // Top-level multiply: either Toom-4 (existing) or Karatsuba/Toom-2.
    let prod: Vec<[usize; 8]> = if !KARATSUBA {
        // Evaluate at 7 points, multiply (len16), interpolate into convolution len 127, then fold.
        let mut w_eval_top: Vec<Vec<[usize; 8]>> = Vec::with_capacity(7);
        for &t in &pts {
            let mut ae: Vec<[usize; 8]> = Vec::with_capacity(16);
            let mut ce: Vec<[usize; 8]> = Vec::with_capacity(16);
            for i in 0..16 {
                ae.push(eval_deg3_poly_at_t(b, &a0[i], &a1[i], &a2[i], &a3[i], t));
                ce.push(eval_deg3_poly_at_t(b, &c0[i], &c1[i], &c2[i], &c3[i], t));
            }
            w_eval_top.push(poly_mul_toom4_len16(b, &ae, &ce, inv720)); // len 31
        }

        // Interpolate into blocks j=0..6, each len 31, giving conv len 127.
        //
        // With block length 31 and shift 16, overlaps are of length 15:
        // for each j>=1 and k in 0..14, index (16j+k) receives contributions from
        // block[j][k] and block[j-1][k+16].
        let mut block: Vec<Vec<[usize; 8]>> = vec![vec![zero; 31]; 7];
        for k in 0..31 {
            // For fixed k, the 7 evaluation values are reused across all 7 j-blocks.
            let mut evals = [zero; 7];
            for i in 0..7 {
                evals[i] = w_eval_top[i][k];
            }
            let evals_d: [Vec<usize>; 7] =
                core::array::from_fn(|i| u64_bytes_to_bal16_digits_cached(b, evals[i]));

            for j in 0..7 {
                let mut coeffs = [0u64; 7];
                for i in 0..7 {
                    let n = NUMS[j][i];
                    if n != 0 {
                        coeffs[i] = modmul_i64_u64(n, inv720, FROG_P);
                    }
                }
                block[j][k] = lincomb7_mod_p_from_canonical_evals(b, &evals, &evals_d, &coeffs);
            }
        }

        let mut prod: Vec<[usize; 8]> = vec![zero; 2 * 64 - 1]; // len 127
        // j=0 has no left-overlap.
        for k in 0..31 {
            prod[k] = block[0][k];
        }
        for j in 1..7 {
            // Overlap k=0..14: sum of current block and previous block's tail.
            for k in 0..15 {
                let idx = j * 16 + k;
                prod[idx] = frog_add_mod_p_from_byte_vars_assume_canonical(b, &block[j][k], &block[j - 1][k + 16]);
            }
            // Non-overlap k=15..30: direct assignment.
            for k in 15..31 {
                let idx = j * 16 + k;
                if idx < prod.len() {
                    prod[idx] = block[j][k];
                }
            }
        }
        prod
    } else {
        // Karatsuba/Toom-2 for length 64 using Toom-4 length-16 as the base multiplier.
        fn poly_add_same_len(
            b: &mut Dr1csBuilder<F257>,
            x: &[[usize; 8]],
            y: &[[usize; 8]],
        ) -> Vec<[usize; 8]> {
            debug_assert_eq!(x.len(), y.len());
            x.iter()
                .zip(y.iter())
                .map(|(a, c)| frog_add_mod_p_from_byte_vars_assume_canonical(b, a, c))
                .collect()
        }
        fn poly_sub_same_len(
            b: &mut Dr1csBuilder<F257>,
            x: &[[usize; 8]],
            y: &[[usize; 8]],
        ) -> Vec<[usize; 8]> {
            debug_assert_eq!(x.len(), y.len());
            x.iter()
                .zip(y.iter())
                .map(|(a, c)| frog_sub_mod_p_from_byte_vars_assume_canonical(b, a, c))
                .collect()
        }

        fn poly_add_into_shifted(
            b: &mut Dr1csBuilder<F257>,
            acc: &mut Vec<[usize; 8]>,
            src: &Vec<[usize; 8]>,
            shift: usize,
        ) {
            for (i, v) in src.iter().enumerate() {
                let idx = i + shift;
                acc[idx] = frog_add_mod_p_from_byte_vars_assume_canonical(b, &acc[idx], v);
            }
        }

        fn poly_mul_karatsuba_len32(
            b: &mut Dr1csBuilder<F257>,
            a32: &Vec<[usize; 8]>,
            c32: &Vec<[usize; 8]>,
            inv720: u64,
            zero: [usize; 8],
        ) -> Vec<[usize; 8]> {
            debug_assert_eq!(a32.len(), 32);
            debug_assert_eq!(c32.len(), 32);
            let (a0, a1) = a32.split_at(16);
            let (c0, c1) = c32.split_at(16);
            let s_a = poly_add_same_len(b, a0, a1);
            let s_c = poly_add_same_len(b, c0, c1);

            let z0 = poly_mul_toom4_len16(b, &a0.to_vec(), &c0.to_vec(), inv720); // len31
            let z2 = poly_mul_toom4_len16(b, &a1.to_vec(), &c1.to_vec(), inv720); // len31
            let z1_full = poly_mul_toom4_len16(b, &s_a, &s_c, inv720); // len31

            let z0_pad = z0;
            let z2_pad = z2;
            let mut z1 = poly_sub_same_len(b, &z1_full, &z0_pad);
            z1 = poly_sub_same_len(b, &z1, &z2_pad);

            let mut out = vec![zero; 2 * 32 - 1]; // len63
            poly_add_into_shifted(b, &mut out, &z0_pad, 0);
            poly_add_into_shifted(b, &mut out, &z1, 16);
            poly_add_into_shifted(b, &mut out, &z2_pad, 32);
            out
        }

        let a_lo: Vec<[usize; 8]> = a[0..32].to_vec();
        let a_hi: Vec<[usize; 8]> = a[32..64].to_vec();
        let c_lo: Vec<[usize; 8]> = c[0..32].to_vec();
        let c_hi: Vec<[usize; 8]> = c[32..64].to_vec();

        let s_a = poly_add_same_len(b, &a_lo, &a_hi);
        let s_c = poly_add_same_len(b, &c_lo, &c_hi);

        let z0 = poly_mul_karatsuba_len32(b, &a_lo, &c_lo, inv720, zero); // len63
        let z2 = poly_mul_karatsuba_len32(b, &a_hi, &c_hi, inv720, zero); // len63
        let z1_full = poly_mul_karatsuba_len32(b, &s_a, &s_c, inv720, zero); // len63

        let mut z1 = poly_sub_same_len(b, &z1_full, &z0);
        z1 = poly_sub_same_len(b, &z1, &z2);

        let mut conv = vec![zero; 2 * 64 - 1]; // len127
        poly_add_into_shifted(b, &mut conv, &z0, 0);
        poly_add_into_shifted(b, &mut conv, &z1, 32);
        poly_add_into_shifted(b, &mut conv, &z2, 64);
        conv
    };

    // Fold mod X^64+1: out[k] = prod[k] - prod[k+64]
    let mut out = [[0usize; 8]; 64];
    for k in 0..64 {
        let hi = k + 64;
        if hi < prod.len() {
            out[k] = frog_sub_mod_p_from_byte_vars_assume_canonical(b, &prod[k], &prod[hi]);
        } else {
            out[k] = prod[k];
        }
    }
    let res = out;
    b.profile_exit(_prev);
    res
}

