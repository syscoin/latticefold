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

#[derive(Clone, Debug)]
struct ByteVar {
    /// Field var intended to be in {0..255}.
    byte: usize,
    /// Bit-decomposition (little-endian).
    bits: [usize; 8],
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

    let (mut inst, asg) = merge_sparse_dr1cs_share_one::<F257>(&[
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

    #[test]
    fn test_poseidon_f257_ops_arithmetization_satisfies() {
        // Record a tiny transcript trace in the **actual sponge field** (F257).
        //
        // IMPORTANT: we must not use `crate::recording_transcript::TracePoseidonTranscript`,
        // which lifts F257 digits into the outer base ring. For a tiny-field gate we want the
        // transcript ops directly over F257.
        let mut tr = symphony::transcript::TracePoseidonTranscript::<FrogRing>::empty::<()>();
        tr.absorb_field_element(&<FrogRing as stark_rings::PolyRing>::BaseRing::from(123u64));
        let _c = tr.get_challenge(); // SqueezeField(12) + Absorb(12)
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
}

