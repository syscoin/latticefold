use ark_ff::{BigInteger, Field, PrimeField};

use latticefold::transcript::poseidon::F257;
use symphony::dpp_sumcheck::Dr1csBuilder;

use super::coins::goldilocks_p_base128_digits_le;
use super::digits::u64_bytes_to_bal16_digits_cached;
use super::cm_ir::{
    goldilocks_add_mod_p_digits_ir, goldilocks_mul_const_mod_p_digits_ir,
    goldilocks_mul_mod_p_digits_ir, goldilocks_sub_mod_p_digits_ir, lower_ir_into_builder, IrBuilder as CmIrBuilder,
    VarRef as IrVarRef,
};
use super::gadgets::{alloc_bool, decompose_existing_byte_var_to_bits};
use super::params::{LIMB_BASE_U64, LIMB_BITS, LIMBS_U64};
// NOTE: keep all modulus constants local to this module; do not import from elsewhere.

/// Goldilocks prime field modulus: \(2^{64} - 2^{32} + 1\).
///
/// This is NTT-friendly (large 2-adicity), unlike the Goldilocks prime used elsewhere in this module.
pub(crate) const GOLDILOCKS_P: u64 = 0xFFFF_FFFF_0000_0001;

// NOTE: We intentionally do not provide a "fast but potentially wrapping" mode here.
// The pm128 fused carry-chain style relations are not injective over the integers in F257
// (see tests in `we_gate_tiny/tests.rs` demonstrating the 257 = 16^2 + 1 bubble).

pub(crate) fn goldilocks_p_bal16_digits_le_const() -> [i8; 17] {
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
        let limb0: u64 = b
            .assignment[u_byte_vars[i]]
            .into_bigint()
            .as_ref()
            .get(0)
            .copied()
            .unwrap_or(0);
        u_buf[i] = (limb0 & 0xFF) as u8;
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
            let limb0: u64 = b.assignment[bor].into_bigint().as_ref().get(0).copied().unwrap_or(0);
            (limb0 & 0x1) as i16
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
            let limb0: u64 = b.assignment[borrow].into_bigint().as_ref().get(0).copied().unwrap_or(0);
            (limb0 & 0x1) as i16
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

fn alloc_u8_var<F: PrimeField>(b: &mut Dr1csBuilder<F>, v: u8) -> usize {
    let x = b.new_var(F::from(v as u64));
    let _ = decompose_existing_byte_var_to_bits::<F>(b, x);
    x
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

    // IR is the source of truth in the digit domain; tie `r_bytes` to the IR digits.
    let p_d_const = goldilocks_p_bal16_digits_le_const();
    let a_d = u64_bytes_to_bal16_digits_cached(b, *a_bytes);
    let c_d = u64_bytes_to_bal16_digits_cached(b, *b_bytes);
    let r_d_bytes = u64_bytes_to_bal16_digits_cached(b, r_bytes);
    debug_assert_eq!(a_d.len(), 17);
    debug_assert_eq!(c_d.len(), 17);
    debug_assert_eq!(r_d_bytes.len(), 17);

    let (ir, out_ir) = {
        let base_asg: &[F257] = &b.assignment;
        let mut ib = CmIrBuilder::new(base_asg);
        let a_ir: [IrVarRef; 17] = core::array::from_fn(|j| IrVarRef::Base(a_d[j]));
        let c_ir: [IrVarRef; 17] = core::array::from_fn(|j| IrVarRef::Base(c_d[j]));
        let out_ir = goldilocks_mul_mod_p_digits_ir(&mut ib, &a_ir, &c_ir, GOLDILOCKS_P, &p_d_const);
        (ib.ir, out_ir)
    };
    let lowered = lower_ir_into_builder(b, ir);
    let r_d_ir: [usize; 17] = core::array::from_fn(|j| lowered.map_var(out_ir[j]));

    for j in 0..17 {
        b.enforce_lc_times_one_eq_const(vec![(F257::ONE, r_d_bytes[j]), (-F257::ONE, r_d_ir[j])]);
    }

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
    let r_u: u64 = (prod % (GOLDILOCKS_P as u128)) as u64;

    // Allocate r as byte vars (output). The quotient `q` is internal; we allocate it directly as
    // balanced base-16 digits to avoid the expensive byte->digit conversion.
    let r_bytes_u8 = r_u.to_le_bytes();
    let mut r_bytes = [0usize; 8];
    for i in 0..8 {
        r_bytes[i] = alloc_u8_var::<F257>(b, r_bytes_u8[i]);
    }
    goldilocks_u64_enforce_lt_p_from_byte_vars::<F257>(b, &r_bytes);

    // IR is the source of truth for const-mul mod p in the digit domain.
    //
    // We still output canonical bytes, so we tie the IR-produced digits to the digit expansion
    // of `r_bytes`.
    let x_d = u64_bytes_to_bal16_digits_cached(b, *x_bytes);
    debug_assert_eq!(x_d.len(), 17);
    let r_d_bytes = u64_bytes_to_bal16_digits_cached(b, r_bytes);
    debug_assert_eq!(r_d_bytes.len(), 17);

    let p_d_const = goldilocks_p_bal16_digits_le_const();

    // Build IR fragment against a snapshot of the current base assignment, then lower.
    let (ir, out_ir) = {
        let base_asg: &[F257] = &b.assignment;
        let mut ib = CmIrBuilder::new(base_asg);
        let x_ir: [IrVarRef; 17] = core::array::from_fn(|j| IrVarRef::Base(x_d[j]));
        let out_ir = goldilocks_mul_const_mod_p_digits_ir(&mut ib, &x_ir, c, GOLDILOCKS_P, &p_d_const);
        (ib.ir, out_ir)
    };
    let lowered = lower_ir_into_builder(b, ir);
    let r_d_ir: [usize; 17] = core::array::from_fn(|j| lowered.map_var(out_ir[j]));

    // Constrain digit expansions equal: r_d_bytes == r_d_ir
    for j in 0..17 {
        b.enforce_lc_times_one_eq_const(vec![(F257::ONE, r_d_bytes[j]), (-F257::ONE, r_d_ir[j])]);
    }

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
    let r_u: u64 = (sum % (GOLDILOCKS_P as u128)) as u64;
    let r_bytes_u8 = r_u.to_le_bytes();
    let mut r_bytes = [0usize; 8];
    for i in 0..8 {
        r_bytes[i] = alloc_u8_var::<F257>(b, r_bytes_u8[i]);
    }
    goldilocks_u64_enforce_lt_p_from_byte_vars::<F257>(b, &r_bytes);

    // IR is the source of truth in the digit domain; tie `r_bytes` to the IR digits.
    let p_d_const = goldilocks_p_bal16_digits_le_const();
    let a_d = u64_bytes_to_bal16_digits_cached(b, *a_bytes);
    let c_d = u64_bytes_to_bal16_digits_cached(b, *c_bytes);
    let r_d_bytes = u64_bytes_to_bal16_digits_cached(b, r_bytes);
    debug_assert_eq!(a_d.len(), 17);
    debug_assert_eq!(c_d.len(), 17);
    debug_assert_eq!(r_d_bytes.len(), 17);

    let (ir, out_ir) = {
        let base_asg: &[F257] = &b.assignment;
        let mut ib = CmIrBuilder::new(base_asg);
        let a_ir: [IrVarRef; 17] = core::array::from_fn(|j| IrVarRef::Base(a_d[j]));
        let c_ir: [IrVarRef; 17] = core::array::from_fn(|j| IrVarRef::Base(c_d[j]));
        let out_ir = goldilocks_add_mod_p_digits_ir(&mut ib, &a_ir, &c_ir, GOLDILOCKS_P, &p_d_const);
        (ib.ir, out_ir)
    };
    let lowered = lower_ir_into_builder(b, ir);
    let r_d_ir: [usize; 17] = core::array::from_fn(|j| lowered.map_var(out_ir[j]));

    for j in 0..17 {
        b.enforce_lc_times_one_eq_const(vec![(F257::ONE, r_d_bytes[j]), (-F257::ONE, r_d_ir[j])]);
    }

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
    let r_u: u64 = if a_u >= c_u {
        a_u - c_u
    } else {
        (a_u as u128 + (GOLDILOCKS_P as u128) - (c_u as u128)) as u64
    };
    let r_bytes_u8 = r_u.to_le_bytes();
    let mut r_bytes = [0usize; 8];
    for i in 0..8 {
        r_bytes[i] = alloc_u8_var::<F257>(b, r_bytes_u8[i]);
    }
    goldilocks_u64_enforce_lt_p_from_byte_vars::<F257>(b, &r_bytes);

    // IR is the source of truth in the digit domain; tie `r_bytes` to the IR digits.
    let p_d_const = goldilocks_p_bal16_digits_le_const();
    let a_d = u64_bytes_to_bal16_digits_cached(b, *a_bytes);
    let c_d = u64_bytes_to_bal16_digits_cached(b, *c_bytes);
    let r_d_bytes = u64_bytes_to_bal16_digits_cached(b, r_bytes);
    debug_assert_eq!(a_d.len(), 17);
    debug_assert_eq!(c_d.len(), 17);
    debug_assert_eq!(r_d_bytes.len(), 17);

    let (ir, out_ir) = {
        let base_asg: &[F257] = &b.assignment;
        let mut ib = CmIrBuilder::new(base_asg);
        let a_ir: [IrVarRef; 17] = core::array::from_fn(|j| IrVarRef::Base(a_d[j]));
        let c_ir: [IrVarRef; 17] = core::array::from_fn(|j| IrVarRef::Base(c_d[j]));
        let out_ir = goldilocks_sub_mod_p_digits_ir(&mut ib, &a_ir, &c_ir, GOLDILOCKS_P, &p_d_const);
        (ib.ir, out_ir)
    };
    let lowered = lower_ir_into_builder(b, ir);
    let r_d_ir: [usize; 17] = core::array::from_fn(|j| lowered.map_var(out_ir[j]));

    for j in 0..17 {
        b.enforce_lc_times_one_eq_const(vec![(F257::ONE, r_d_bytes[j]), (-F257::ONE, r_d_ir[j])]);
    }

    let out = r_bytes;
    b.profile_exit(_prev);
    out
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

/// Convert 8 little-endian byte vars (0..255) into a canonical Goldilocks scalar (balanced base-16 digits, len 17).
#[inline]
pub(crate) fn goldilocks_scalar_from_u64_bytes_le_digits(b: &mut Dr1csBuilder<F257>, bytes_le: [usize; 8]) -> GoldilocksScalar {
    vec17_to_arr17(u64_bytes_to_bal16_digits_cached(b, bytes_le))
}
