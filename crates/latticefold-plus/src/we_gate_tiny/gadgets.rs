use ark_ff::PrimeField;

use symphony::dpp_poseidon::{Constraint, SparseDr1csInstance};
use symphony::dpp_sumcheck::Dr1csBuilder;

use super::params::LIMB_BITS;

#[derive(Clone, Debug)]
pub(super) struct ByteVar {
    /// Field var intended to be in {0..255}.
    pub(super) byte: usize,
    /// Bit-decomposition (little-endian).
    pub(super) bits: [usize; 8],
}

pub(super) fn alloc_bool<F: PrimeField>(b: &mut Dr1csBuilder<F>, bit: bool) -> usize {
    let _prev = b.profile_enter("gadgets::alloc_bool");
    let v = b.new_var(if bit { F::ONE } else { F::ZERO });
    // v*(1-v)=0
    b.add_constraint(
        vec![(F::ONE, v)],
        vec![(F::ONE, b.one()), (-F::ONE, v)],
        vec![(F::ZERO, b.one())],
    );
    b.profile_exit(_prev);
    v
}

pub(super) fn alloc_byte<F: PrimeField>(b: &mut Dr1csBuilder<F>, v8: u8) -> ByteVar {
    let _prev = b.profile_enter("gadgets::alloc_byte");
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

    // Cache bit decomposition so downstream gadgets can reuse it.
    b.byte_bits_cache.insert(v, bits);

    b.profile_exit(_prev);
    ByteVar { byte: v, bits }
}

#[allow(dead_code)]
pub(super) fn alloc_u64_as_bytes_le<F: PrimeField>(b: &mut Dr1csBuilder<F>, x: u64) -> [ByteVar; 8] {
    let bytes = x.to_le_bytes();
    [
        alloc_byte::<F>(b, bytes[0]),
        alloc_byte::<F>(b, bytes[1]),
        alloc_byte::<F>(b, bytes[2]),
        alloc_byte::<F>(b, bytes[3]),
        alloc_byte::<F>(b, bytes[4]),
        alloc_byte::<F>(b, bytes[5]),
        alloc_byte::<F>(b, bytes[6]),
        alloc_byte::<F>(b, bytes[7]),
    ]
}

pub(super) fn decompose_existing_byte_var_to_bits<F: PrimeField>(
    b: &mut Dr1csBuilder<F>,
    byte_var: usize,
) -> [usize; 8] {
    let _prev = b.profile_enter("gadgets::decompose_byte_to_bits");
    if let Some(bits) = b.byte_bits_cache.get(&byte_var) {
        let bits = *bits;
        b.profile_exit(_prev);
        return bits;
    }
    // Constrain: byte_var = Σ 2^i * bit_i, with bit_i boolean.
    let mut bits = [0usize; 8];
    // Avoid `to_bytes_le()` allocations: read low limb directly.
    let limb0: u64 = b
        .assignment[byte_var]
        .into_bigint()
        .as_ref()
        .get(0)
        .copied()
        .unwrap_or(0);
    let v8: u16 = (limb0 & 0xFF) as u16;
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
    b.byte_bits_cache.insert(byte_var, bits);
    b.profile_exit(_prev);
    bits
}

#[allow(dead_code)]
pub(super) fn alloc_u7<F: PrimeField>(b: &mut Dr1csBuilder<F>, v7: u8) -> usize {
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

#[allow(dead_code)]
pub(super) fn alloc_u2_from_u8<F: PrimeField>(b: &mut Dr1csBuilder<F>, v2: u8) -> usize {
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
#[allow(dead_code)]
pub(super) fn const_zero<F: PrimeField>(b: &mut Dr1csBuilder<F>) -> usize {
    let v = b.new_var(F::ZERO);
    b.enforce_var_eq_const(v, F::ZERO);
    v
}

#[inline]
pub(super) fn bool_not<F: PrimeField>(b: &mut Dr1csBuilder<F>, x: usize) -> usize {
    let v = b.new_var(F::ONE - b.assignment[x]);
    b.add_constraint(
        vec![(F::ONE, b.one()), (-F::ONE, x)],
        vec![(F::ONE, b.one())],
        vec![(F::ONE, v)],
    );
    v
}

#[inline]
pub(super) fn bool_and<F: PrimeField>(b: &mut Dr1csBuilder<F>, x: usize, y: usize) -> usize {
    let v = b.new_var(b.assignment[x] * b.assignment[y]);
    b.enforce_mul(x, y, v);
    v
}

#[inline]
pub(super) fn bool_or<F: PrimeField>(b: &mut Dr1csBuilder<F>, x: usize, y: usize) -> usize {
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

#[allow(dead_code)]
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

#[allow(dead_code)]
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

#[allow(dead_code)]
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

#[allow(dead_code)]
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

#[inline]
pub(super) fn enforce_var_eq<F: PrimeField>(inst: &mut SparseDr1csInstance<F>, x: usize, y: usize) {
    // Enforce (x - y) * 1 = 0
    let a0 = inst.a_terms.len();
    inst.a_terms.extend_from_slice(&[(F::ONE, x), (-F::ONE, y)]);
    let a1 = inst.a_terms.len();
    let b0 = inst.b_terms.len();
    inst.b_terms.push((F::ONE, 0));
    let b1 = inst.b_terms.len();
    let c0 = inst.c_terms.len();
    inst.c_terms.push((F::ZERO, 0));
    let c1 = inst.c_terms.len();
    inst.constraints.push(Constraint { a: a0..a1, b: b0..b1, c: c0..c1 });
}

