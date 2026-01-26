use ark_ff::{BigInteger, PrimeField};

use symphony::dpp_sumcheck::Dr1csBuilder;

use super::coins::frog_p_base128_digits_le;
use super::gadgets::{alloc_bool, decompose_existing_byte_var_to_bits};
use super::params::{LIMB_BASE_U64, LIMB_BITS, LIMBS_U64};
use crate::we_frog_poseidon_f257::FROG_P;

/// Boundary-only canonicalization gadget:
/// given an unconstrained 64-bit integer `u` as 8 little-endian bytes, produce
/// `(q, z)` such that:
/// - `q ∈ {0,1}`
/// - `u = z + q * p_frog` as an **integer** (no wrap), enforced via base-128 borrows over a
///   bit-derived base-128 limb view of `u`.
///
/// This is the "single subtract" reduction justified by \(2^{64} < 2p\).
/// Takes raw byte variables that are already constrained to be 8-bit.
#[allow(dead_code)]
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

