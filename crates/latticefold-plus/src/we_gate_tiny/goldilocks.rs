use ark_ff::{BigInteger, Field, PrimeField};

use latticefold::transcript::poseidon::F257;
use symphony::dpp_sumcheck::Dr1csBuilder;

use super::op_counts::tiny_cm_bump;

use super::coins::goldilocks_p_base128_digits_le;
use super::digits::{
    add_bal16_same_len, alloc_bal16_digit, mul_bal16_long_by_const_rhs, mul_bal16_long_by_long,
    alloc_carry_pm2, alloc_carry_pm32, alloc_carry_pm128, i32_to_f257,
    u64_bytes_to_bal16_digits_cached,
};
use super::gadgets::{alloc_bool, decompose_existing_byte_var_to_bits};
use super::params::{LIMB_BASE_U64, LIMB_BITS, LIMBS_U64};
// NOTE: keep all modulus constants local to this module; do not import from elsewhere.
use cyclotomic_rings::rings::goldilocks_ntt64 as gl_ntt64;

/// Goldilocks prime field modulus: \(2^{64} - 2^{32} + 1\).
///
/// This is NTT-friendly (large 2-adicity), unlike the Goldilocks prime used elsewhere in this module.
pub(crate) const GOLDILOCKS_P: u64 = 0xFFFF_FFFF_0000_0001;

// NOTE: We intentionally do not provide a "fast but potentially wrapping" mode here.
// The pm128 fused carry-chain style relations are not injective over the integers in F257
// (see tests in `we_gate_tiny/tests.rs` demonstrating the 257 = 16^2 + 1 bubble).

fn goldilocks_p_bal16_digits_le_const() -> [i8; 17] {
    // Match the balancing convention used by `u64_bytes_to_bal16_digits`:
    // carry_{i+1} = (nibble_i + carry_i >= 8), out_i = nibble_i + carry_i - 16*carry_{i+1}.
    let mut out = [0i8; 17];
    let mut carry: i16 = 0;
    let mut x = GOLDILOCKS_P;
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
/// - `u = z + q * p_goldilocks` as an **integer** (no wrap), enforced via base-128 borrows over a
///   bit-derived base-128 limb view of `u`.
///
/// This is the "single subtract" reduction justified by \(2^{64} < 2p\).
/// Takes raw byte variables that are already constrained to be 8-bit.
pub(super) fn reduce_u64_mod_goldilocks_from_byte_vars<F: PrimeField>(
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
    let q_u8: u8 = if u >= GOLDILOCKS_P { 1 } else { 0 };
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
    let p_digits = goldilocks_p_base128_digits_le();
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
    let z_u64 = if q_u8 == 1 { u - GOLDILOCKS_P } else { u };
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
    let p_digits = goldilocks_p_base128_digits_le();
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

/// Enforce that an 8-byte little-endian integer `u` satisfies `u < p_goldilocks`.
///
/// This is the preferred "IO boundary" check for transcript-absorbed base-field elements.
///
/// Internally this:
/// - bit-decomposes the bytes (cached)
/// - packs them into 10 base-128 limbs (7-bit each)
/// - runs a base-128 borrow chain on (u - p)
/// - enforces the final borrow is 1 (i.e., u < p)
pub(super) fn goldilocks_u64_enforce_lt_p_from_byte_vars<F: PrimeField>(
    b: &mut Dr1csBuilder<F>,
    u_byte_vars: &[usize; 8],
) {
    let _ = goldilocks_u64_enforce_lt_p_from_byte_vars_and_limbs::<F>(b, u_byte_vars);
}

/// Enforce `u < p_goldilocks` and return `u` packed as base-128 limbs.
///
/// This is the "lt_p only" variant that avoids computing `u mod p` while still materializing the
/// limb view needed by some downstream bounded gadgets.
pub(super) fn goldilocks_u64_enforce_lt_p_from_byte_vars_and_limbs<F: PrimeField>(
    b: &mut Dr1csBuilder<F>,
    u_byte_vars: &[usize; 8],
) -> [usize; LIMBS_U64] {
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
    let p_digits = goldilocks_p_base128_digits_le();
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
    u_limbs
}

/// Enforce that a canonical Goldilocks base-field element `u` (encoded as 8 bytes, u < p_goldilocks)
/// lies in the **centered magnitude** range \(|u| <= bound\), meaning:
/// - u ∈ [0, bound]  (non-negative)
/// - OR u ∈ [p_goldilocks - bound, p_goldilocks - 1]  (negative, in centered lift)
///
/// Returns base-128 limbs of `u`.
pub(super) fn goldilocks_u64_centered_le_bound_from_byte_vars<F: PrimeField>(
    b: &mut Dr1csBuilder<F>,
    u_byte_vars: &[usize; 8],
    bound: u64,
) -> [usize; LIMBS_U64] {
    // First, enforce canonical encoding (u < p) and get base-128 limbs for u.
    //
    // We intentionally avoid the heavier "reduce-then-assert-q=0" style gadget here;
    // for centered-bound checks we only need the limb view + the guarantee `u < p`.
    let u_limbs = goldilocks_u64_enforce_lt_p_from_byte_vars_and_limbs::<F>(b, u_byte_vars);

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
    let p_minus_bound = GOLDILOCKS_P.saturating_sub(bound);
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
    if v.len() > target_len {
        // Enforce truncated tail digits are zero, then truncate.
        for &dv in &v[target_len..] {
            b.enforce_var_eq_const(dv, F257::ZERO);
        }
        v.truncate(target_len);
        return v;
    }
    if v.len() < target_len {
        let z = b.zero_var();
        v.extend(std::iter::repeat(z).take(target_len - v.len()));
    }
    v
}

fn enforce_bal16_vec_eq(b: &mut Dr1csBuilder<F257>, a: &[usize], c: &[usize]) {
    debug_assert_eq!(a.len(), c.len());
    for (&ai, &ci) in a.iter().zip(c.iter()) {
        b.enforce_lc_times_one_eq_const(vec![(F257::ONE, ai), (-F257::ONE, ci)]);
    }
}

fn goldilocks_p_bytes_le() -> [u8; 8] {
    GOLDILOCKS_P.to_le_bytes()
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
    // Sound no-wrap enforcement:
    // materialize `qp = q*p` in bal16 digits (using the safe mul-by-const gadget),
    // add `r`, and enforce equality to `prod` using the bal16 add gadget (pm1 carries).
    let max_len = prod_d
        .len()
        .max(r_d.len())
        .max(q_d.len().saturating_add(p_d_const.len()).saturating_sub(1))
        + 1; // headroom digit

    let prod_pad = pad_bal16(b, prod_d.to_vec(), max_len);
    let r_pad = pad_bal16(b, r_d.to_vec(), max_len);

    let qp = mul_bal16_long_by_const_rhs(b, q_d, p_d_const);
    let qp_pad = pad_bal16(b, qp, max_len);

    let (sum, carry) = add_bal16_same_len(b, &qp_pad, &r_pad);
    b.enforce_var_eq_const(carry, F257::ZERO);

    enforce_bal16_vec_eq(b, &prod_pad, &sum);
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

/// General Goldilocks-field multiplication gadget (64-bit prime field) inside the tiny field.
///
/// Inputs are canonical u64 encodings as 8 little-endian byte vars (0..255), representing
/// field elements in \([0, p)\).
///
/// This enforces:
/// \[
///   a \cdot b = q \cdot p + r,\quad 0 \le r < p
/// \]
/// using balanced-base16 digit arithmetic (sound in F257), and returns `r` as 8 byte vars.
pub(super) fn goldilocks_mul_mod_p_from_byte_vars(
    b: &mut Dr1csBuilder<F257>,
    a_bytes: &[usize; 8],
    b_bytes: &[usize; 8],
) -> [usize; 8] {
    // Ensure inputs are canonical (<p).
    goldilocks_u64_enforce_lt_p_from_byte_vars::<F257>(b, a_bytes);
    goldilocks_u64_enforce_lt_p_from_byte_vars::<F257>(b, b_bytes);
    goldilocks_mul_mod_p_from_byte_vars_assume_canonical(b, a_bytes, b_bytes)
}

#[inline]
pub(super) fn goldilocks_mul_mod_p_from_byte_vars_assume_canonical(
    b: &mut Dr1csBuilder<F257>,
    a_bytes: &[usize; 8],
    b_bytes: &[usize; 8],
) -> [usize; 8] {
    let _prev = b.profile_enter("goldilocks::mul_mod_p");

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
    let q_u: u64 = (prod / (GOLDILOCKS_P as u128)) as u64;
    let r_u: u64 = (prod % (GOLDILOCKS_P as u128)) as u64;

    // Allocate r as byte vars (output). The quotient `q` is internal; we allocate it directly as
    // balanced base-16 digits to avoid the expensive byte->digit conversion.
    let r_bytes_u8 = r_u.to_le_bytes();
    let mut r_bytes = [0usize; 8];
    for i in 0..8 {
        r_bytes[i] = alloc_u8_var::<F257>(b, r_bytes_u8[i]);
    }
    // Enforce r is canonical (<p).
    goldilocks_u64_enforce_lt_p_from_byte_vars::<F257>(b, &r_bytes);

    // Convert a,b,r and p to balanced-base16 digits.
    let a_d = u64_bytes_to_bal16_digits_cached(b, *a_bytes);
    let b_d = u64_bytes_to_bal16_digits_cached(b, *b_bytes);
    let r_d = u64_bytes_to_bal16_digits_cached(b, r_bytes);
    let q_d = alloc_u64_as_bal16_digits_witness(b, q_u);

    // Compute prod_digits = a*b (balanced digits, with headroom/carry already enforced in gadget).
    let prod_d = mul_bal16_long_by_long(b, &a_d, &b_d);
    let p_d_const = goldilocks_p_bal16_digits_le_const();
    enforce_prod_eq_qp_plus_r_bal16(b, &prod_d, &q_d, &p_d_const, &r_d);

    let out = r_bytes;
    b.profile_exit(_prev);
    out
}

/// Multiply a canonical Goldilocks scalar `x` by a **known constant** `c` (as u64 in `[0,p)`),
/// returning a canonical Goldilocks scalar (8-byte little-endian) for `x*c mod p`.
///
/// This is substantially cheaper than `goldilocks_mul_mod_p_from_byte_vars` because it avoids all
/// digit×digit multiplications in the 64-bit integer product checks.
pub(super) fn goldilocks_mul_const_mod_p_from_byte_vars(
    b: &mut Dr1csBuilder<F257>,
    x_bytes: &[usize; 8],
    c: u64,
) -> [usize; 8] {
    assert!(c < GOLDILOCKS_P, "goldilocks_mul_const_mod_p_from_byte_vars requires c < p");
    goldilocks_u64_enforce_lt_p_from_byte_vars::<F257>(b, x_bytes);
    goldilocks_mul_const_mod_p_from_byte_vars_assume_canonical(b, x_bytes, c)
}

#[inline]
pub(super) fn goldilocks_mul_const_mod_p_from_byte_vars_assume_canonical(
    b: &mut Dr1csBuilder<F257>,
    x_bytes: &[usize; 8],
    c: u64,
) -> [usize; 8] {
    let _prev = b.profile_enter("goldilocks::mul_const_mod_p");

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
    let q_u: u64 = (prod / (GOLDILOCKS_P as u128)) as u64;
    let r_u: u64 = (prod % (GOLDILOCKS_P as u128)) as u64;

    // Allocate r as byte vars (output). The quotient `q` is internal; we allocate it directly as
    // balanced base-16 digits to avoid the expensive byte->digit conversion.
    let r_bytes_u8 = r_u.to_le_bytes();
    let mut r_bytes = [0usize; 8];
    for i in 0..8 {
        r_bytes[i] = alloc_u8_var::<F257>(b, r_bytes_u8[i]);
    }
    goldilocks_u64_enforce_lt_p_from_byte_vars::<F257>(b, &r_bytes);

    // Convert x,r to bal16 digits (vars). `q` is allocated directly as balanced digits.
    let x_d = u64_bytes_to_bal16_digits_cached(b, *x_bytes);
    let r_d = u64_bytes_to_bal16_digits_cached(b, r_bytes);
    let q_d = alloc_u64_as_bal16_digits_witness(b, q_u);

    // Constant bal16 digits for c and p, computed directly (no variables/constraints).
    let c_d_const = u64_to_bal16_digits_le_const(c);
    let p_d_const = goldilocks_p_bal16_digits_le_const();

    // prod_digits = x*c  (const-RHS)
    let prod_d = mul_bal16_long_by_const_rhs(b, &x_d, &c_d_const);
    enforce_prod_eq_qp_plus_r_bal16(b, &prod_d, &q_d, &p_d_const, &r_d);

    let out = r_bytes;
    b.profile_exit(_prev);
    out
}

/// General Goldilocks-field addition gadget inside F257.
///
/// Enforces `r = (a + c) mod p` for canonical `a,c < p`, returning canonical `r` as 8 bytes.
pub(super) fn goldilocks_add_mod_p_from_byte_vars(
    b: &mut Dr1csBuilder<F257>,
    a_bytes: &[usize; 8],
    c_bytes: &[usize; 8],
) -> [usize; 8] {
    goldilocks_u64_enforce_lt_p_from_byte_vars::<F257>(b, a_bytes);
    goldilocks_u64_enforce_lt_p_from_byte_vars::<F257>(b, c_bytes);
    goldilocks_add_mod_p_from_byte_vars_assume_canonical(b, a_bytes, c_bytes)
}

#[inline]
pub(super) fn goldilocks_add_mod_p_from_byte_vars_assume_canonical(
    b: &mut Dr1csBuilder<F257>,
    a_bytes: &[usize; 8],
    c_bytes: &[usize; 8],
) -> [usize; 8] {
    let _prev = b.profile_enter("goldilocks::add_mod_p");

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
    let q_u8: u8 = if sum >= (GOLDILOCKS_P as u128) { 1 } else { 0 };
    let r_u: u64 = if q_u8 == 1 { (sum - (GOLDILOCKS_P as u128)) as u64 } else { sum as u64 };

    // `q_u8` is a witness-known 0/1 selector; use cached constants instead of allocating a boolean.
    let q = if q_u8 == 1 { b.one() } else { b.zero_var() };
    let r_bytes_u8 = r_u.to_le_bytes();
    let mut r_bytes = [0usize; 8];
    for i in 0..8 {
        r_bytes[i] = alloc_u8_var::<F257>(b, r_bytes_u8[i]);
    }
    goldilocks_u64_enforce_lt_p_from_byte_vars::<F257>(b, &r_bytes);

    let p_d0_const = goldilocks_p_bal16_digits_le_const();
    let a_d0 = u64_bytes_to_bal16_digits_cached(b, *a_bytes);
    let c_d0 = u64_bytes_to_bal16_digits_cached(b, *c_bytes);
    let r_d0 = u64_bytes_to_bal16_digits_cached(b, r_bytes);
    enforce_add_mod_p_relation_bal16(b, &a_d0, &c_d0, &r_d0, q, q_u8, &p_d0_const);

    let out = r_bytes;
    b.profile_exit(_prev);
    out
}

/// General Goldilocks-field subtraction gadget inside F257.
///
/// Enforces `r = (a - c) mod p` for canonical `a,c < p`, returning canonical `r` as 8 bytes.
pub(super) fn goldilocks_sub_mod_p_from_byte_vars(
    b: &mut Dr1csBuilder<F257>,
    a_bytes: &[usize; 8],
    c_bytes: &[usize; 8],
) -> [usize; 8] {
    goldilocks_u64_enforce_lt_p_from_byte_vars::<F257>(b, a_bytes);
    goldilocks_u64_enforce_lt_p_from_byte_vars::<F257>(b, c_bytes);
    goldilocks_sub_mod_p_from_byte_vars_assume_canonical(b, a_bytes, c_bytes)
}

#[inline]
pub(super) fn goldilocks_sub_mod_p_from_byte_vars_assume_canonical(
    b: &mut Dr1csBuilder<F257>,
    a_bytes: &[usize; 8],
    c_bytes: &[usize; 8],
) -> [usize; 8] {
    let _prev = b.profile_enter("goldilocks::sub_mod_p");

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
        (1u8, (a_u as u128 + (GOLDILOCKS_P as u128) - (c_u as u128)) as u64)
    };

    // `q_u8` is a witness-known 0/1 selector; use cached constants instead of allocating a boolean.
    let q = if q_u8 == 1 { b.one() } else { b.zero_var() };
    let r_bytes_u8 = r_u.to_le_bytes();
    let mut r_bytes = [0usize; 8];
    for i in 0..8 {
        r_bytes[i] = alloc_u8_var::<F257>(b, r_bytes_u8[i]);
    }
    goldilocks_u64_enforce_lt_p_from_byte_vars::<F257>(b, &r_bytes);

    let a_d0 = u64_bytes_to_bal16_digits_cached(b, *a_bytes);
    let c_d0 = u64_bytes_to_bal16_digits_cached(b, *c_bytes);
    let r_d0 = u64_bytes_to_bal16_digits_cached(b, r_bytes);
    let p_d0_const = goldilocks_p_bal16_digits_le_const();
    enforce_sub_mod_p_relation_bal16(b, &a_d0, &c_d0, &r_d0, q, q_u8, &p_d0_const);

    let out = r_bytes;
    b.profile_exit(_prev);
    out
}

fn goldilocks_zero_bytes(b: &mut Dr1csBuilder<F257>) -> [usize; 8] {
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

fn goldilocks_one_bytes(b: &mut Dr1csBuilder<F257>) -> [usize; 8] {
    let mut o = goldilocks_zero_bytes(b);
    o[0] = b.new_var(F257::ONE);
    b.enforce_var_eq_const(o[0], F257::ONE);
    o
}

fn goldilocks_from_u64_const_bytes(b: &mut Dr1csBuilder<F257>, c: u64) -> [usize; 8] {
    let cb = c.to_le_bytes();
    let mut out = [0usize; 8];
    for i in 0..8 {
        out[i] = alloc_u8_var::<F257>(b, cb[i]);
    }
    goldilocks_u64_enforce_lt_p_from_byte_vars::<F257>(b, &out);
    out
}

fn goldilocks_add_many_mod_p(b: &mut Dr1csBuilder<F257>, terms: &[[usize; 8]]) -> [usize; 8] {
    let mut acc = goldilocks_zero_bytes(b);
    for t in terms {
        acc = goldilocks_add_mod_p_from_byte_vars_assume_canonical(b, &acc, t);
    }
    acc
}

/// Goldilocks scalar in balanced base-16 digits (canonical u64 encoding).
///
/// Representation matches `u64_bytes_to_bal16_digits_cached`:
/// - 16 balanced digits in [-8,7]
/// - plus a final carry digit in {0,1}
pub(crate) type GoldilocksScalar = [usize; 17];

#[inline]
fn vec17_to_arr17(v: Vec<usize>) -> GoldilocksScalar {
    debug_assert_eq!(v.len(), 17);
    let mut out = [0usize; 17];
    for i in 0..17 {
        out[i] = v[i];
    }
    out
}

#[inline]
fn digits_to_u64_witness(b: &Dr1csBuilder<F257>, d: &GoldilocksScalar) -> u64 {
    let mut acc: i128 = 0;
    let mut pow: i128 = 1;
    for i in 0..17 {
        let di = super::digits::f257_to_i32_bal(b.assignment[d[i]]) as i128;
        acc += di * pow;
        pow *= 16;
    }
    debug_assert!(acc >= 0);
    acc as u64
}

/// Convert 8 little-endian byte vars (0..255) into a canonical Goldilocks scalar (balanced base-16 digits, len 17).
#[inline]
pub(crate) fn goldilocks_scalar_from_u64_bytes_le_digits(b: &mut Dr1csBuilder<F257>, bytes_le: [usize; 8]) -> GoldilocksScalar {
    vec17_to_arr17(u64_bytes_to_bal16_digits_cached(b, bytes_le))
}

/// Digit-domain Goldilocks addition: `r = a + c (mod p)`.
#[inline]
pub(crate) fn goldilocks_add_mod_p_digits(b: &mut Dr1csBuilder<F257>, a: &GoldilocksScalar, c: &GoldilocksScalar) -> GoldilocksScalar {
    tiny_cm_bump(|cc| cc.scalar_add += 1);
    let p_d = goldilocks_p_bal16_digits_le_const();
    let a_u = digits_to_u64_witness(b, a);
    let c_u = digits_to_u64_witness(b, c);
    let sum = (a_u as u128) + (c_u as u128);
    let q_u8: u8 = if sum >= (GOLDILOCKS_P as u128) { 1 } else { 0 };
    let r_u: u64 = if q_u8 == 1 { (sum - (GOLDILOCKS_P as u128)) as u64 } else { sum as u64 };
    // `q_u8` is witness-known (derived from canonical digits), so use cached constants.
    let q = if q_u8 == 1 { b.one() } else { b.zero_var() };
    let r_d = vec17_to_arr17(alloc_u64_as_bal16_digits_witness(b, r_u));
    enforce_add_mod_p_relation_bal16(b, a, c, &r_d, q, q_u8, &p_d);
    r_d
}

/// Digit-domain Goldilocks subtraction: `r = a - c (mod p)`.
#[inline]
pub(crate) fn goldilocks_sub_mod_p_digits(b: &mut Dr1csBuilder<F257>, a: &GoldilocksScalar, c: &GoldilocksScalar) -> GoldilocksScalar {
    tiny_cm_bump(|cc| cc.scalar_sub += 1);
    let p_d = goldilocks_p_bal16_digits_le_const();
    let a_u = digits_to_u64_witness(b, a);
    let c_u = digits_to_u64_witness(b, c);
    let (q_u8, r_u) = if a_u >= c_u {
        (0u8, a_u - c_u)
    } else {
        (1u8, (a_u as u128 + (GOLDILOCKS_P as u128) - (c_u as u128)) as u64)
    };
    // `q_u8` is witness-known (derived from canonical digits), so use cached constants.
    let q = if q_u8 == 1 { b.one() } else { b.zero_var() };
    let r_d = vec17_to_arr17(alloc_u64_as_bal16_digits_witness(b, r_u));
    enforce_sub_mod_p_relation_bal16(b, a, c, &r_d, q, q_u8, &p_d);
    r_d
}

/// Digit-domain Goldilocks multiplication: `r = a * c (mod p)`.
#[inline]
pub(crate) fn goldilocks_mul_mod_p_digits(b: &mut Dr1csBuilder<F257>, a: &GoldilocksScalar, c: &GoldilocksScalar) -> GoldilocksScalar {
    tiny_cm_bump(|cc| cc.scalar_mul += 1);
    let p_d = goldilocks_p_bal16_digits_le_const();
    let a_u = digits_to_u64_witness(b, a);
    let c_u = digits_to_u64_witness(b, c);
    let prod: u128 = (a_u as u128) * (c_u as u128);
    let q_u: u64 = (prod / (GOLDILOCKS_P as u128)) as u64;
    let r_u: u64 = (prod % (GOLDILOCKS_P as u128)) as u64;
    let q_d = alloc_u64_as_bal16_digits_witness(b, q_u);
    let r_d = vec17_to_arr17(alloc_u64_as_bal16_digits_witness(b, r_u));
    let prod_d = mul_bal16_long_by_long(b, a, c);
    enforce_prod_eq_qp_plus_r_bal16(b, &prod_d, &q_d, &p_d, &r_d);
    r_d
}

/// Negacyclic ring multiplication for `d=64` over **Goldilocks** using an NTT-based method.
///
/// This is a “what it looks like” prototype: it stays in digit arithmetic throughout, uses
/// variable×const twiddle multiplies and only 64 variable×variable multiplications for the
/// pointwise product.
///
/// Returns `c = a*b mod (X^64 + 1)` as 64 canonical u64 digit-encodings (bal16 digits).
pub(super) fn ring_mul_negacyclic_ntt_goldilocks_d64(
    b: &mut Dr1csBuilder<F257>,
    a: &[GoldilocksScalar; 64],
    c: &[GoldilocksScalar; 64],
) -> [GoldilocksScalar; 64] {
    // Shared schedule constants from `cyclotomic-rings` (keeps host + gate in sync).
    let p: u64 = gl_ntt64::GOLDILOCKS_P_U64;
    let omega: u64 = gl_ntt64::OMEGA_U64;
    let omega_inv: u64 = gl_ntt64::OMEGA_INV_U64;
    let psi: u64 = gl_ntt64::PSI_U64;
    let inv_n: u64 = gl_ntt64::INV_N_U64;

    // --- Small host-side helpers for Goldilocks constants.
    #[inline]
    fn mul_mod_u64(a: u64, b: u64, p: u64) -> u64 {
        ((a as u128) * (b as u128) % (p as u128)) as u64
    }
    debug_assert_eq!(p, GOLDILOCKS_P);
    debug_assert_eq!(mul_mod_u64(psi, psi, p), omega);

    let p_d_const = u64_to_bal16_digits_le_const(p);

    // --- Digit-field helpers parameterized by modulus p.
    #[inline]
    fn digits_to_u64_witness(b: &Dr1csBuilder<F257>, d: &GoldilocksScalar) -> u64 {
        let mut acc: i128 = 0;
        let mut pow: i128 = 1;
        for i in 0..17 {
            let di = super::digits::f257_to_i32_bal(b.assignment[d[i]]) as i128;
            acc += di * pow;
            pow *= 16;
        }
        debug_assert!(acc >= 0);
        acc as u64
    }
    #[inline]
    fn vec17_to_arr17(v: Vec<usize>) -> GoldilocksScalar {
        debug_assert_eq!(v.len(), 17);
        let mut out = [0usize; 17];
        for i in 0..17 {
            out[i] = v[i];
        }
        out
    }

    #[inline]
    fn add_mod_p(b: &mut Dr1csBuilder<F257>, a: &GoldilocksScalar, c: &GoldilocksScalar, p_u64: u64, p_d: &[i8; 17]) -> GoldilocksScalar {
        let a_u = digits_to_u64_witness(b, a);
        let c_u = digits_to_u64_witness(b, c);
        let sum = (a_u as u128) + (c_u as u128);
        let q_u8: u8 = if sum >= (p_u64 as u128) { 1 } else { 0 };
        let r_u: u64 = if q_u8 == 1 { (sum - (p_u64 as u128)) as u64 } else { sum as u64 };
        let q = if q_u8 == 1 { b.one() } else { b.zero_var() };
        let r_d = vec17_to_arr17(alloc_u64_as_bal16_digits_witness(b, r_u));
        enforce_add_mod_p_relation_bal16(b, a, c, &r_d, q, q_u8, p_d);
        r_d
    }

    #[inline]
    fn sub_mod_p(b: &mut Dr1csBuilder<F257>, a: &GoldilocksScalar, c: &GoldilocksScalar, p_u64: u64, p_d: &[i8; 17]) -> GoldilocksScalar {
        let a_u = digits_to_u64_witness(b, a);
        let c_u = digits_to_u64_witness(b, c);
        let (q_u8, r_u) = if a_u >= c_u {
            (0u8, a_u - c_u)
        } else {
            (1u8, (a_u as u128 + (p_u64 as u128) - (c_u as u128)) as u64)
        };
        let q = if q_u8 == 1 { b.one() } else { b.zero_var() };
        let r_d = vec17_to_arr17(alloc_u64_as_bal16_digits_witness(b, r_u));
        enforce_sub_mod_p_relation_bal16(b, a, c, &r_d, q, q_u8, p_d);
        r_d
    }

    #[inline]
    fn mul_mod_p(b: &mut Dr1csBuilder<F257>, a: &GoldilocksScalar, c: &GoldilocksScalar, p_u64: u64, p_d: &[i8; 17]) -> GoldilocksScalar {
        let a_u = digits_to_u64_witness(b, a);
        let c_u = digits_to_u64_witness(b, c);
        let prod: u128 = (a_u as u128) * (c_u as u128);
        let q_u: u64 = (prod / (p_u64 as u128)) as u64;
        let r_u: u64 = (prod % (p_u64 as u128)) as u64;
        let q_d = alloc_u64_as_bal16_digits_witness(b, q_u);
        let r_d = vec17_to_arr17(alloc_u64_as_bal16_digits_witness(b, r_u));
        let prod_d = mul_bal16_long_by_long(b, a, c);
        enforce_prod_eq_qp_plus_r_bal16(b, &prod_d, &q_d, p_d, &r_d);
        r_d
    }

    #[inline]
    fn mul_const_mod_p(b: &mut Dr1csBuilder<F257>, x: &GoldilocksScalar, k: u64, p_u64: u64, p_d: &[i8; 17]) -> GoldilocksScalar {
        #[inline]
        fn enforce_prod_const_eq_qp_plus_r_bal16(
            b: &mut Dr1csBuilder<F257>,
            x_d: &GoldilocksScalar,
            k_d_const: &[i8; 17],
            q_d: &[usize],
            p_d_const: &[i8; 17],
            r_d: &GoldilocksScalar,
        ) {
            // Enforce: x*k == q*p + r in base-16 carry chain, without materializing x*k digits.
            //
            // This is the const-RHS analogue of `enforce_prod_eq_qp_plus_r_bal16` and avoids the
            // expensive `digits::mul_bal16_long_by_const_rhs` gadget entirely.
            let max_len = 17usize
                .max(r_d.len())
                .max(q_d.len().saturating_add(p_d_const.len()).saturating_sub(1))
                + 1; // headroom digit

            let mut carry_var = b.new_var(F257::ZERO);
            b.enforce_var_eq_const(carry_var, F257::ZERO);
            let mut carry_i32: i32 = 0;

            // Pad r to max_len with zero digits (constant 0 var).
            let zero = b.zero_var();
            b.enforce_var_eq_const(zero, F257::ZERO);

            for k in 0..max_len {
                let mut sum: i32 = carry_i32;
                let mut lc: Vec<(F257, usize)> = Vec::with_capacity(4 + q_d.len() + 17);
                lc.push((F257::ONE, carry_var));

                // -r_k
                if k < r_d.len() {
                    let rk = super::digits::f257_to_i32_bal(b.assignment[r_d[k]]);
                    sum -= rk;
                    lc.push((-F257::ONE, r_d[k]));
                } else {
                    // r pad is zero
                    lc.push((-F257::ONE, zero));
                }

                // + Σ_i x_i * k_{k-i}
                for i in 0..17 {
                    if i > k {
                        break;
                    }
                    let j = k - i;
                    if j >= 17 {
                        continue;
                    }
                    let kd = k_d_const[j] as i32;
                    if kd == 0 {
                        continue;
                    }
                    let xi = super::digits::f257_to_i32_bal(b.assignment[x_d[i]]);
                    sum += xi * kd;
                    lc.push((i32_to_f257(kd), x_d[i]));
                }

                // - Σ_i q_i * p_{k-i}
                for i in 0..q_d.len() {
                    if i > k {
                        break;
                    }
                    let j = k - i;
                    if j >= 17 {
                        continue;
                    }
                    let pd = p_d_const[j] as i32;
                    if pd == 0 {
                        continue;
                    }
                    let qi = super::digits::f257_to_i32_bal(b.assignment[q_d[i]]);
                    sum -= qi * pd;
                    lc.push((-i32_to_f257(pd), q_d[i]));
                }

                debug_assert!(
                    sum % 16 == 0,
                    "const-mul carry check not divisible: sum={sum} at k={k}"
                );
                let carry_next: i32 = sum / 16;
                debug_assert!(
                    (-128..=127).contains(&carry_next),
                    "const-mul carry out of pm128 bound: {carry_next} at k={k} (sum={sum})"
                );
                let carry_next_var = alloc_carry_pm128(b, carry_next);
                lc.push((-F257::from(16u64), carry_next_var));
                b.enforce_lc_times_one_eq_const(lc);

                carry_var = carry_next_var;
                carry_i32 = carry_next;
            }
            b.enforce_var_eq_const(carry_var, F257::ZERO);
        }

        let x_u = digits_to_u64_witness(b, x);
        let prod: u128 = (x_u as u128) * (k as u128);
        let q_u: u64 = (prod / (p_u64 as u128)) as u64;
        let r_u: u64 = (prod % (p_u64 as u128)) as u64;
        let q_d = alloc_u64_as_bal16_digits_witness(b, q_u);
        let r_d = vec17_to_arr17(alloc_u64_as_bal16_digits_witness(b, r_u));
        let k_d_const = u64_to_bal16_digits_le_const(k);
        enforce_prod_const_eq_qp_plus_r_bal16(b, x, &k_d_const, &q_d, p_d, &r_d);
        r_d
    }

    // --- NTT plumbing (use shared twiddle tables).
    fn ntt_in_place(
        b: &mut Dr1csBuilder<F257>,
        a: &mut [GoldilocksScalar; 64],
        omega: u64,
        p_u64: u64,
        p_d: &[i8; 17],
    ) {
        let zero: GoldilocksScalar = [b.zero_var(); 17];
        // Bit-reversal permutation (purely structural).
        let mut tmp = *a;
        for i in 0..64 {
            tmp[gl_ntt64::BITREV_64[i]] = a[i];
        }
        *a = tmp;

        // Iterative Cooley–Tukey.
        let mut len = 2usize;
        while len <= 64 {
            let half = len / 2;
            for start in (0..64).step_by(len) {
                for j in 0..half {
                    let w: u64 = if omega == gl_ntt64::OMEGA_U64 {
                        match len {
                            2 => gl_ntt64::W_POWS_LEN_2[j],
                            4 => gl_ntt64::W_POWS_LEN_4[j],
                            8 => gl_ntt64::W_POWS_LEN_8[j],
                            16 => gl_ntt64::W_POWS_LEN_16[j],
                            32 => gl_ntt64::W_POWS_LEN_32[j],
                            64 => gl_ntt64::W_POWS_LEN_64[j],
                            _ => unreachable!(),
                        }
                    } else {
                        debug_assert_eq!(omega, gl_ntt64::OMEGA_INV_U64);
                        match len {
                            2 => gl_ntt64::IW_POWS_LEN_2[j],
                            4 => gl_ntt64::IW_POWS_LEN_4[j],
                            8 => gl_ntt64::IW_POWS_LEN_8[j],
                            16 => gl_ntt64::IW_POWS_LEN_16[j],
                            32 => gl_ntt64::IW_POWS_LEN_32[j],
                            64 => gl_ntt64::IW_POWS_LEN_64[j],
                            _ => unreachable!(),
                        }
                    };
                    let u = a[start + j];
                    let v = if w == 1 {
                        a[start + j + half]
                    } else if w == p_u64 - 1 {
                        // v = -x mod p = 0 - x
                        sub_mod_p(b, &zero, &a[start + j + half], p_u64, p_d)
                    } else {
                        mul_const_mod_p(b, &a[start + j + half], w, p_u64, p_d)
                    };
                    a[start + j] = add_mod_p(b, &u, &v, p_u64, p_d);
                    a[start + j + half] = sub_mod_p(b, &u, &v, p_u64, p_d);
                }
            }
            len *= 2;
        }
    }

    fn intt_in_place(
        b: &mut Dr1csBuilder<F257>,
        a: &mut [GoldilocksScalar; 64],
        omega_inv: u64,
        inv_n: u64,
        p_u64: u64,
        p_d: &[i8; 17],
    ) {
        ntt_in_place(b, a, omega_inv, p_u64, p_d);
        // scale by n^{-1}
        for i in 0..64 {
            a[i] = mul_const_mod_p(b, &a[i], inv_n, p_u64, p_d);
        }
    }

    // Negacyclic via twist by ψ^i (ψ is primitive 128th root).
    let mut a_tw = [[b.zero_var(); 17]; 64];
    let mut c_tw = [[b.zero_var(); 17]; 64];
    for i in 0..64 {
        let psi_pow: u64 = gl_ntt64::PSI_POWS_64[i];
        if psi_pow == 1 {
            a_tw[i] = a[i];
            c_tw[i] = c[i];
        } else {
            a_tw[i] = mul_const_mod_p(b, &a[i], psi_pow, p, &p_d_const);
            c_tw[i] = mul_const_mod_p(b, &c[i], psi_pow, p, &p_d_const);
        }
    }

    ntt_in_place(b, &mut a_tw, omega, p, &p_d_const);
    ntt_in_place(b, &mut c_tw, omega, p, &p_d_const);

    // Pointwise multiply.
    for i in 0..64 {
        a_tw[i] = mul_mod_p(b, &a_tw[i], &c_tw[i], p, &p_d_const);
    }

    intt_in_place(b, &mut a_tw, omega_inv, inv_n, p, &p_d_const);

    // Untwist by ψ^{-i}.
    let mut out = [[b.zero_var(); 17]; 64];
    for i in 0..64 {
        let psi_inv_pow: u64 = gl_ntt64::PSI_INV_POWS_64[i];
        out[i] = if psi_inv_pow == 1 {
            a_tw[i]
        } else {
            mul_const_mod_p(b, &a_tw[i], psi_inv_pow, p, &p_d_const)
        };
    }
    out
}

