use ark_ff::Field;
use latticefold::transcript::poseidon::F257;
use symphony::dpp_sumcheck::Dr1csBuilder;

use rayon::prelude::*;

use super::op_counts::tiny_cm_bump;
use super::goldilocks::{
    goldilocks_add_mod_p_from_byte_vars_assume_canonical, goldilocks_mul_mod_p_from_byte_vars_assume_canonical,
    goldilocks_mul_const_mod_p_from_byte_vars_assume_canonical,
    goldilocks_sub_mod_p_from_byte_vars_assume_canonical,
    goldilocks_scalar_from_u64_bytes_le_digits,
    goldilocks_p_bal16_digits_le_const, GoldilocksScalar,
};
use super::cm_ir::{
    bal4_to_bal16_digits_ir, goldilocks_add_mod_p_digits_ir, goldilocks_mul_const_mod_p_digits_bal4_ir,
    goldilocks_mul_mod_p_digits_ir, goldilocks_sub_mod_p_digits_ir, lower_ir_into_builder,
    ring_mul_negacyclic_ntt_goldilocks_d64_ir, IrBuilder, VarRef as IrVarRef,
};

/// A ring element whose coefficients are Goldilocks base-field scalars encoded as canonical 8-byte little-endian limbs.
///
/// - Length = ring dimension `d` (one coefficient per ring coefficient).
/// - Each coefficient is `[u8; 8]` represented as 8 F257 vars in `[0,255]`, and is constrained elsewhere to be `< p`.
pub(crate) type RingBytes = Vec<[usize; 8]>;

/// A ring element whose coefficients are Goldilocks base-field scalars encoded as balanced base-16 digits.
///
/// - Length = ring dimension `d` (one coefficient per ring coefficient).
/// - Each coefficient is a canonical Goldilocks scalar `[usize; 17]` (bal16 digits).
pub(crate) type RingDigits = Vec<GoldilocksScalar>;

#[inline]
fn alloc_const_byte(gb: &mut Dr1csBuilder<F257>, v: u8) -> usize {
    let x = gb.new_var(F257::from(v as u64));
    gb.enforce_var_eq_const(x, F257::from(v as u64));
    x
}

#[inline]
pub(crate) fn alloc_const_goldilocks_u64(gb: &mut Dr1csBuilder<F257>, v: u64) -> [usize; 8] {
    let _prev = gb.profile_enter("cm_math::alloc_const_goldilocks_u64");
    let bs = v.to_le_bytes();
    let mut out = [0usize; 8];
    for i in 0..8 {
        out[i] = alloc_const_byte(gb, bs[i]);
    }
    gb.profile_exit(_prev);
    out
}

#[inline]
pub(crate) fn ring_zero_bytes(gb: &mut Dr1csBuilder<F257>, d: usize) -> RingBytes {
    let _prev = gb.profile_enter("cm_math::ring_zero_bytes");
    let z = alloc_const_goldilocks_u64(gb, 0u64);
    let out = vec![z; d];
    gb.profile_exit(_prev);
    out
}

#[inline]
pub(crate) fn ring_eq_bytes(gb: &mut Dr1csBuilder<F257>, a: &RingBytes, b: &RingBytes) {
    let _prev = gb.profile_enter("cm_math::ring_eq_bytes");
    tiny_cm_bump(|c| c.ring_eq += 1);
    debug_assert_eq!(a.len(), b.len());
    for (ai, bi) in a.iter().zip(b.iter()) {
        for j in 0..8 {
            gb.enforce_lc_times_one_eq_const(vec![(F257::ONE, ai[j]), (-F257::ONE, bi[j])]);
        }
    }
    gb.profile_exit(_prev);
}

#[inline]
pub(crate) fn ring_add_bytes(gb: &mut Dr1csBuilder<F257>, a: &RingBytes, b: &RingBytes) -> RingBytes {
    let _prev = gb.profile_enter("cm_math::ring_add_bytes");
    tiny_cm_bump(|c| c.ring_add += 1);
    debug_assert_eq!(a.len(), b.len());
    let mut out = Vec::with_capacity(a.len());
    for i in 0..a.len() {
        out.push(goldilocks_add_mod_p_from_byte_vars_assume_canonical(gb, &a[i], &b[i]));
    }
    gb.profile_exit(_prev);
    out
}

#[inline]
pub(crate) fn ring_sub_bytes(gb: &mut Dr1csBuilder<F257>, a: &RingBytes, b: &RingBytes) -> RingBytes {
    let _prev = gb.profile_enter("cm_math::ring_sub_bytes");
    tiny_cm_bump(|c| c.ring_sub += 1);
    debug_assert_eq!(a.len(), b.len());
    let mut out = Vec::with_capacity(a.len());
    for i in 0..a.len() {
        out.push(goldilocks_sub_mod_p_from_byte_vars_assume_canonical(gb, &a[i], &b[i]));
    }
    gb.profile_exit(_prev);
    out
}

/// Scale a ring element by a base-field scalar (byte-encoded).
///
/// Since the scalar is in the base field, this is per-coefficient multiplication.
#[inline]
pub(crate) fn ring_scale_bytes(gb: &mut Dr1csBuilder<F257>, a: &RingBytes, s: &[usize; 8]) -> RingBytes {
    let _prev = gb.profile_enter("cm_math::ring_scale_bytes");
    tiny_cm_bump(|c| c.ring_scale += 1);
    let mut out = Vec::with_capacity(a.len());
    for i in 0..a.len() {
        out.push(goldilocks_mul_mod_p_from_byte_vars_assume_canonical(gb, &a[i], s));
    }
    gb.profile_exit(_prev);
    out
}

/// Compute Lagrange basis coefficients (L0,L1,L2) for interpolation of degree-2 sumcheck message
/// polynomials at points 0,1,2 evaluated at `r`.
///
/// This mirrors `we_gate_arith::lagrange_degree2` but over Goldilocks base field, with byte encodings.
pub(crate) fn lagrange_degree2_goldilocks(
    gb: &mut Dr1csBuilder<F257>,
    r: &[usize; 8],
    _inv2: &[usize; 8],
    one: &[usize; 8],
    two: &[usize; 8],
) -> ([usize; 8], [usize; 8], [usize; 8]) {
    let _prev = gb.profile_enter("cm_math::lagrange_degree2_goldilocks_bytes");
    // 2 subs, 4 muls, plus one "mul by const" semantic (inv2) in the big-gate accounting.
    // Here we count the higher-level helper call once.
    tiny_cm_bump(|c| c.scalar_sub_const += 0);
    // t1 = r - 1, t2 = r - 2
    let t1 = goldilocks_sub_mod_p_from_byte_vars_assume_canonical(gb, r, one);
    let t2 = goldilocks_sub_mod_p_from_byte_vars_assume_canonical(gb, r, two);

    // L0 = (r-1)(r-2)/2
    let p = goldilocks_mul_mod_p_from_byte_vars_assume_canonical(gb, &t1, &t2);
    let inv2_u64 = (crate::we_goldilocks_poseidon_f257::GOLDILOCKS_P + 1) / 2;
    let l0 = goldilocks_mul_const_mod_p_from_byte_vars_assume_canonical(gb, &p, inv2_u64);

    // L1 = -r(r-2) = 0 - r(r-2)
    let p = goldilocks_mul_mod_p_from_byte_vars_assume_canonical(gb, r, &t2);
    let zero = alloc_const_goldilocks_u64(gb, 0u64);
    let l1 = goldilocks_sub_mod_p_from_byte_vars_assume_canonical(gb, &zero, &p);

    // L2 = r(r-1)/2
    let p = goldilocks_mul_mod_p_from_byte_vars_assume_canonical(gb, r, &t1);
    let l2 = goldilocks_mul_const_mod_p_from_byte_vars_assume_canonical(gb, &p, inv2_u64);

    gb.profile_exit(_prev);
    (l0, l1, l2)
}

/// Verify a degree-2 sumcheck over ring elements (byte-encoded coefficients) and return the final claim.
///
/// - `claimed_sum`: initial claim (ring)
/// - `msgs`: per-round prover messages, each has 3 ring elements (g(0), g(1), g(2))
/// - `rs`: per-round verifier challenges (base-field scalars), byte-encoded
pub(crate) fn sumcheck_verify_degree2_ring_bytes(
    gb: &mut Dr1csBuilder<F257>,
    mut claimed_sum: RingBytes,
    msgs: &[[RingBytes; 3]],
    rs: &[[usize; 8]],
) -> Result<RingBytes, String> {
    if msgs.len() != rs.len() {
        return Err("sumcheck_verify_degree2_ring_bytes: msgs/rs length mismatch".to_string());
    }
    let _prev = gb.profile_enter("cm_math::sumcheck_verify_degree2_ring_bytes");
    let d = claimed_sum.len();

    // Constants in Goldilocks field.
    // inv2 is computed in host here as a fixed u64 constant; encoded as bytes.
    // (Caller can reuse these across calls later.)
    let one = alloc_const_goldilocks_u64(gb, 1u64);
    let two = alloc_const_goldilocks_u64(gb, 2u64);
    // inv2 mod p (p is odd): (p+1)/2
    let inv2_u64 = (crate::we_goldilocks_poseidon_f257::GOLDILOCKS_P + 1) / 2;
    let inv2 = alloc_const_goldilocks_u64(gb, inv2_u64);

    for (m, r) in msgs.iter().zip(rs.iter()) {
        if m[0].len() != d || m[1].len() != d || m[2].len() != d {
            gb.profile_exit(_prev);
            return Err("sumcheck_verify_degree2_ring_bytes: ring dimension mismatch".to_string());
        }

        // Check g(0) + g(1) == claimed_sum (coefficient-wise).
        let g01 = ring_add_bytes(gb, &m[0], &m[1]);
        ring_eq_bytes(gb, &g01, &claimed_sum);

        // Update claim = g(r) via Lagrange interpolation.
        let (l0, l1, l2) = lagrange_degree2_goldilocks(gb, r, &inv2, &one, &two);
        let t0 = ring_scale_bytes(gb, &m[0], &l0);
        let t1 = ring_scale_bytes(gb, &m[1], &l1);
        let t2 = ring_scale_bytes(gb, &m[2], &l2);
        let s01 = ring_add_bytes(gb, &t0, &t1);
        claimed_sum = ring_add_bytes(gb, &s01, &t2);
    }
    gb.profile_exit(_prev);
    Ok(claimed_sum)
}

// -----------------------------------------------------------------------------
// Digit-domain helpers (preferred for heavy CM math + ring multiplication)
// -----------------------------------------------------------------------------

#[inline]
pub(crate) fn goldilocks_bytes_to_digits(gb: &mut Dr1csBuilder<F257>, bytes_le: [usize; 8]) -> GoldilocksScalar {
    let _prev = gb.profile_enter("cm_math::goldilocks_bytes_to_digits");
    let out = goldilocks_scalar_from_u64_bytes_le_digits(gb, bytes_le);
    gb.profile_exit(_prev);
    out
}

#[inline]
pub(crate) fn ring_bytes_to_digits(gb: &mut Dr1csBuilder<F257>, a: &RingBytes) -> RingDigits {
    let _prev = gb.profile_enter("cm_math::ring_bytes_to_digits");
    tiny_cm_bump(|c| c.lc_to_var += 1);
    let out: RingDigits = a.iter().copied().map(|x| goldilocks_bytes_to_digits(gb, x)).collect();
    gb.profile_exit(_prev);
    out
}

#[inline]
pub(crate) fn ring_zero_digits(gb: &mut Dr1csBuilder<F257>, d: usize) -> RingDigits {
    let _prev = gb.profile_enter("cm_math::ring_zero_digits");
    let z_bytes = alloc_const_goldilocks_u64(gb, 0u64);
    let z = goldilocks_bytes_to_digits(gb, z_bytes);
    let out = vec![z; d];
    gb.profile_exit(_prev);
    out
}

#[inline]
pub(crate) fn ring_eq_digits(gb: &mut Dr1csBuilder<F257>, a: &RingDigits, b: &RingDigits) {
    let _prev = gb.profile_enter("cm_math::ring_eq_digits");
    tiny_cm_bump(|c| c.ring_eq += 1);
    debug_assert_eq!(a.len(), b.len());
    for (ai, bi) in a.iter().zip(b.iter()) {
        for j in 0..17 {
            gb.enforce_lc_times_one_eq_const(vec![(F257::ONE, ai[j]), (-F257::ONE, bi[j])]);
        }
    }
    gb.profile_exit(_prev);
}

#[inline]
pub(crate) fn ring_add_digits(gb: &mut Dr1csBuilder<F257>, a: &RingDigits, b: &RingDigits) -> RingDigits {
    let _prev = gb.profile_enter("cm_math::ring_add_digits");
    tiny_cm_bump(|c| c.ring_add += 1);
    debug_assert_eq!(a.len(), b.len());
    // Scalar ops come overwhelmingly from ring ops; build per-coefficient shards in parallel.
    tiny_cm_bump(|c| c.scalar_add += a.len() as u64);
    let base_asg: &[F257] = &gb.assignment;
    let frags: Vec<(_, [IrVarRef; 17])> = (0..a.len())
        .into_par_iter()
        .map(|i| {
            let mut ib = IrBuilder::new(base_asg);
            let ai: [IrVarRef; 17] = core::array::from_fn(|j| IrVarRef::Base(a[i][j]));
            let bi: [IrVarRef; 17] = core::array::from_fn(|j| IrVarRef::Base(b[i][j]));
            let out = goldilocks_add_mod_p_digits_ir(
                &mut ib,
                &ai,
                &bi,
                crate::we_goldilocks_poseidon_f257::GOLDILOCKS_P,
                &goldilocks_p_bal16_digits_le_const(),
            );
            Ok::<_, String>((ib.ir, out))
        })
        .collect::<Result<Vec<_>, _>>()
        .expect("ring_add_digits IR emit should be infallible");

    let mut out: RingDigits = Vec::with_capacity(a.len());
    for (ir, out_ir) in frags {
        let lowered = lower_ir_into_builder(gb, ir);
        let digits: GoldilocksScalar = core::array::from_fn(|j| lowered.map_var(out_ir[j]));
        out.push(digits);
    }
    gb.profile_exit(_prev);
    out
}

#[inline]
pub(crate) fn ring_sub_digits(gb: &mut Dr1csBuilder<F257>, a: &RingDigits, b: &RingDigits) -> RingDigits {
    let _prev = gb.profile_enter("cm_math::ring_sub_digits");
    tiny_cm_bump(|c| c.ring_sub += 1);
    debug_assert_eq!(a.len(), b.len());
    tiny_cm_bump(|c| c.scalar_sub += a.len() as u64);
    let base_asg: &[F257] = &gb.assignment;
    let frags: Vec<(_, [IrVarRef; 17])> = (0..a.len())
        .into_par_iter()
        .map(|i| {
            let mut ib = IrBuilder::new(base_asg);
            let ai: [IrVarRef; 17] = core::array::from_fn(|j| IrVarRef::Base(a[i][j]));
            let bi: [IrVarRef; 17] = core::array::from_fn(|j| IrVarRef::Base(b[i][j]));
            let out = goldilocks_sub_mod_p_digits_ir(
                &mut ib,
                &ai,
                &bi,
                crate::we_goldilocks_poseidon_f257::GOLDILOCKS_P,
                &goldilocks_p_bal16_digits_le_const(),
            );
            Ok::<_, String>((ib.ir, out))
        })
        .collect::<Result<Vec<_>, _>>()
        .expect("ring_sub_digits IR emit should be infallible");

    let mut out: RingDigits = Vec::with_capacity(a.len());
    for (ir, out_ir) in frags {
        let lowered = lower_ir_into_builder(gb, ir);
        let digits: GoldilocksScalar = core::array::from_fn(|j| lowered.map_var(out_ir[j]));
        out.push(digits);
    }
    gb.profile_exit(_prev);
    out
}

#[inline]
pub(crate) fn ring_scale_digits(gb: &mut Dr1csBuilder<F257>, a: &RingDigits, s: &GoldilocksScalar) -> RingDigits {
    let _prev = gb.profile_enter("cm_math::ring_scale_digits");
    tiny_cm_bump(|c| c.ring_scale += 1);
    tiny_cm_bump(|c| c.scalar_mul += a.len() as u64);
    let base_asg: &[F257] = &gb.assignment;
    let s_ir: [IrVarRef; 17] = core::array::from_fn(|j| IrVarRef::Base(s[j]));
    let frags: Vec<(_, [IrVarRef; 17])> = (0..a.len())
        .into_par_iter()
        .map(|i| {
            let mut ib = IrBuilder::new(base_asg);
            let ai: [IrVarRef; 17] = core::array::from_fn(|j| IrVarRef::Base(a[i][j]));
            let out = goldilocks_mul_mod_p_digits_ir(
                &mut ib,
                &ai,
                &s_ir,
                crate::we_goldilocks_poseidon_f257::GOLDILOCKS_P,
                &goldilocks_p_bal16_digits_le_const(),
            );
            Ok::<_, String>((ib.ir, out))
        })
        .collect::<Result<Vec<_>, _>>()
        .expect("ring_scale_digits IR emit should be infallible");

    let mut out: RingDigits = Vec::with_capacity(a.len());
    for (ir, out_ir) in frags {
        let lowered = lower_ir_into_builder(gb, ir);
        let digits: GoldilocksScalar = core::array::from_fn(|j| lowered.map_var(out_ir[j]));
        out.push(digits);
    }
    gb.profile_exit(_prev);
    out
}

/// Negacyclic ring multiplication for ring_dim=64 over Goldilocks (digit-domain).
pub(crate) fn ring_mul_negacyclic_digits_d64(
    gb: &mut Dr1csBuilder<F257>,
    a: &RingDigits,
    b: &RingDigits,
) -> Result<RingDigits, String> {
    let _prev = gb.profile_enter("cm_math::ring_mul_negacyclic_digits_d64");
    tiny_cm_bump(|c| c.ring_mul_negacyclic += 1);
    if a.len() != 64 || b.len() != 64 {
        return Err("ring_mul_negacyclic_digits_d64: expected ring_dim=64".to_string());
    }

    // Build an IR fragment for the ring-mul using only the current base assignment (read-only),
    // then lower into the mutable builder.
    //
    // This is the key decoupling needed to later build many ring-muls in parallel as IR shards.
    let a_ir: [[IrVarRef; 17]; 64] = core::array::from_fn(|i| core::array::from_fn(|j| IrVarRef::Base(a[i][j])));
    let b_ir: [[IrVarRef; 17]; 64] = core::array::from_fn(|i| core::array::from_fn(|j| IrVarRef::Base(b[i][j])));

    let mut irb = IrBuilder::new(&gb.assignment);
    let out_ir = ring_mul_negacyclic_ntt_goldilocks_d64_ir(&mut irb, &a_ir, &b_ir);
    let lowered = lower_ir_into_builder(gb, irb.ir);

    let out: [GoldilocksScalar; 64] = core::array::from_fn(|i| {
        core::array::from_fn(|j| lowered.map_var(out_ir[i][j]))
    });
    let out = Ok(out.into_iter().collect());
    gb.profile_exit(_prev);
    out
}

// -----------------------------------------------------------------------------
// Scalar digit-domain ops (IR is source of truth)
// -----------------------------------------------------------------------------

#[inline]
fn goldilocks_add_mod_p_digits(
    gb: &mut Dr1csBuilder<F257>,
    a: &GoldilocksScalar,
    c: &GoldilocksScalar,
) -> GoldilocksScalar {
    let _prev = gb.profile_enter("cm_math::scalar_add_mod_p_digits");
    tiny_cm_bump(|cc| cc.scalar_add += 1);
    let p_u64 = crate::we_goldilocks_poseidon_f257::GOLDILOCKS_P;
    let p_d = goldilocks_p_bal16_digits_le_const();
    let base_asg: &[F257] = &gb.assignment;
    let mut ib = IrBuilder::new(base_asg);
    let a_ir: [IrVarRef; 17] = core::array::from_fn(|j| IrVarRef::Base(a[j]));
    let c_ir: [IrVarRef; 17] = core::array::from_fn(|j| IrVarRef::Base(c[j]));
    let out_ir = goldilocks_add_mod_p_digits_ir(&mut ib, &a_ir, &c_ir, p_u64, &p_d);
    let lowered = lower_ir_into_builder(gb, ib.ir);
    let out = core::array::from_fn(|j| lowered.map_var(out_ir[j]));
    gb.profile_exit(_prev);
    out
}

#[inline]
fn goldilocks_sub_mod_p_digits(
    gb: &mut Dr1csBuilder<F257>,
    a: &GoldilocksScalar,
    c: &GoldilocksScalar,
) -> GoldilocksScalar {
    let _prev = gb.profile_enter("cm_math::scalar_sub_mod_p_digits");
    tiny_cm_bump(|cc| cc.scalar_sub += 1);
    let p_u64 = crate::we_goldilocks_poseidon_f257::GOLDILOCKS_P;
    let p_d = goldilocks_p_bal16_digits_le_const();
    let base_asg: &[F257] = &gb.assignment;
    let mut ib = IrBuilder::new(base_asg);
    let a_ir: [IrVarRef; 17] = core::array::from_fn(|j| IrVarRef::Base(a[j]));
    let c_ir: [IrVarRef; 17] = core::array::from_fn(|j| IrVarRef::Base(c[j]));
    let out_ir = goldilocks_sub_mod_p_digits_ir(&mut ib, &a_ir, &c_ir, p_u64, &p_d);
    let lowered = lower_ir_into_builder(gb, ib.ir);
    let out = core::array::from_fn(|j| lowered.map_var(out_ir[j]));
    gb.profile_exit(_prev);
    out
}

#[inline]
fn goldilocks_mul_mod_p_digits(
    gb: &mut Dr1csBuilder<F257>,
    a: &GoldilocksScalar,
    c: &GoldilocksScalar,
) -> GoldilocksScalar {
    let _prev = gb.profile_enter("cm_math::scalar_mul_mod_p_digits");
    tiny_cm_bump(|cc| cc.scalar_mul += 1);
    let p_u64 = crate::we_goldilocks_poseidon_f257::GOLDILOCKS_P;
    let p_d = goldilocks_p_bal16_digits_le_const();
    let base_asg: &[F257] = &gb.assignment;
    let mut ib = IrBuilder::new(base_asg);
    let a_ir: [IrVarRef; 17] = core::array::from_fn(|j| IrVarRef::Base(a[j]));
    let c_ir: [IrVarRef; 17] = core::array::from_fn(|j| IrVarRef::Base(c[j]));
    let out_ir = goldilocks_mul_mod_p_digits_ir(&mut ib, &a_ir, &c_ir, p_u64, &p_d);
    let lowered = lower_ir_into_builder(gb, ib.ir);
    let out = core::array::from_fn(|j| lowered.map_var(out_ir[j]));
    gb.profile_exit(_prev);
    out
}

#[inline]
fn goldilocks_mul_const_mod_p_digits(gb: &mut Dr1csBuilder<F257>, a: &GoldilocksScalar, k: u64) -> GoldilocksScalar {
    let _prev = gb.profile_enter("cm_math::scalar_mul_const_mod_p_digits");
    tiny_cm_bump(|cc| cc.scalar_mul_const += 1);
    let p_u64 = crate::we_goldilocks_poseidon_f257::GOLDILOCKS_P;
    debug_assert!(k < p_u64);
    let base_asg: &[F257] = &gb.assignment;
    let mut ib = IrBuilder::new(base_asg);
    let a_ir: [IrVarRef; 17] = core::array::from_fn(|j| IrVarRef::Base(a[j]));
    let a4 = ib.bal16_to_bal4_digits_cached(&a_ir);
    let r4 = goldilocks_mul_const_mod_p_digits_bal4_ir(&mut ib, &a4, k, p_u64);
    let out_ir = bal4_to_bal16_digits_ir(&mut ib, &r4);
    let lowered = lower_ir_into_builder(gb, ib.ir);
    let out = core::array::from_fn(|j| lowered.map_var(out_ir[j]));
    gb.profile_exit(_prev);
    out
}

/// Degree-2 Lagrange basis in digit encoding.
pub(crate) fn lagrange_degree2_goldilocks_digits(
    gb: &mut Dr1csBuilder<F257>,
    r: &GoldilocksScalar,
    inv2: &GoldilocksScalar,
    one: &GoldilocksScalar,
    two: &GoldilocksScalar,
) -> (GoldilocksScalar, GoldilocksScalar, GoldilocksScalar) {
    let inv2_u64 = (crate::we_goldilocks_poseidon_f257::GOLDILOCKS_P + 1) / 2;
    // t1 = r - 1, t2 = r - 2
    let t1 = goldilocks_sub_mod_p_digits(gb, r, one);
    let t2 = goldilocks_sub_mod_p_digits(gb, r, two);

    // L0 = (r-1)(r-2)/2
    let p = goldilocks_mul_mod_p_digits(gb, &t1, &t2);
    let _ = inv2;
    let l0 = goldilocks_mul_const_mod_p_digits(gb, &p, inv2_u64);

    // L1 = -r(r-2)
    let p = goldilocks_mul_mod_p_digits(gb, r, &t2);
    let zero_bytes = alloc_const_goldilocks_u64(gb, 0u64);
    let zero = goldilocks_bytes_to_digits(gb, zero_bytes);
    let l1 = goldilocks_sub_mod_p_digits(gb, &zero, &p);

    // L2 = r(r-1)/2
    let p = goldilocks_mul_mod_p_digits(gb, r, &t1);
    let l2 = goldilocks_mul_const_mod_p_digits(gb, &p, inv2_u64);

    (l0, l1, l2)
}

/// Verify a degree-2 sumcheck over ring elements (digit-encoded) and return the final claim.
pub(crate) fn sumcheck_verify_degree2_ring_digits(
    gb: &mut Dr1csBuilder<F257>,
    mut claimed_sum: RingDigits,
    msgs: &[[RingDigits; 3]],
    rs: &[GoldilocksScalar],
) -> Result<RingDigits, String> {
    if msgs.len() != rs.len() {
        return Err("sumcheck_verify_degree2_ring_digits: msgs/rs length mismatch".to_string());
    }
    let _prev = gb.profile_enter("cm_math::sumcheck_verify_degree2_ring_digits");
    let d = claimed_sum.len();

    let one_bytes = alloc_const_goldilocks_u64(gb, 1u64);
    let two_bytes = alloc_const_goldilocks_u64(gb, 2u64);
    let one = goldilocks_bytes_to_digits(gb, one_bytes);
    let two = goldilocks_bytes_to_digits(gb, two_bytes);
    let inv2_u64 = (crate::we_goldilocks_poseidon_f257::GOLDILOCKS_P + 1) / 2;
    let inv2_bytes = alloc_const_goldilocks_u64(gb, inv2_u64);
    let inv2 = goldilocks_bytes_to_digits(gb, inv2_bytes);

    for (m, r) in msgs.iter().zip(rs.iter()) {
        if m[0].len() != d || m[1].len() != d || m[2].len() != d {
            gb.profile_exit(_prev);
            return Err("sumcheck_verify_degree2_ring_digits: ring dimension mismatch".to_string());
        }

        // g(0) + g(1) == claim
        let g01 = ring_add_digits(gb, &m[0], &m[1]);
        ring_eq_digits(gb, &g01, &claimed_sum);

        // claim = g(r)
        let (l0, l1, l2) = lagrange_degree2_goldilocks_digits(gb, r, &inv2, &one, &two);
        let t0 = ring_scale_digits(gb, &m[0], &l0);
        let t1 = ring_scale_digits(gb, &m[1], &l1);
        let t2 = ring_scale_digits(gb, &m[2], &l2);
        let s01 = ring_add_digits(gb, &t0, &t1);
        claimed_sum = ring_add_digits(gb, &s01, &t2);
    }

    gb.profile_exit(_prev);
    Ok(claimed_sum)
}

// -----------------------------------------------------------------------------
// CM tensor evaluation (t(z)) in digit domain
// -----------------------------------------------------------------------------

#[inline]
fn goldilocks_const_u64_digits(gb: &mut Dr1csBuilder<F257>, v: u64) -> GoldilocksScalar {
    let bytes = alloc_const_goldilocks_u64(gb, v);
    goldilocks_bytes_to_digits(gb, bytes)
}

#[inline]
pub(crate) fn goldilocks_one_minus_digits(gb: &mut Dr1csBuilder<F257>, r: &GoldilocksScalar) -> GoldilocksScalar {
    let one = goldilocks_const_u64_digits(gb, 1u64);
    goldilocks_sub_mod_p_digits(gb, &one, r)
}

/// Tensor-expand a list of verifier challenges `c = [c0..c_{t-1}]` into length-2^t coefficients.
///
/// Convention matches LF+/CM: for each bit, expand `acc` into `[acc*(1-c_i), acc*c_i]`.
pub(crate) fn tensor_goldilocks_scalars_digits(gb: &mut Dr1csBuilder<F257>, c: &[GoldilocksScalar]) -> Vec<GoldilocksScalar> {
    let mut acc: Vec<GoldilocksScalar> = vec![goldilocks_const_u64_digits(gb, 1u64)];
    for ci in c {
        let one_minus = goldilocks_one_minus_digits(gb, ci);
        let mut next: Vec<GoldilocksScalar> = Vec::with_capacity(acc.len() * 2);
        for t in &acc {
            next.push(goldilocks_mul_mod_p_digits(gb, t, &one_minus));
            next.push(goldilocks_mul_mod_p_digits(gb, t, ci));
        }
        acc = next;
    }
    acc
}

/// Lift a tensor-expanded scalar table into constant-coeff ring elements.
pub(crate) fn tensor_goldilocks_ringconst_digits(
    gb: &mut Dr1csBuilder<F257>,
    tensor_scalars: &[GoldilocksScalar],
    ring_dim: usize,
) -> Vec<RingDigits> {
    tensor_scalars
        .iter()
        .map(|s| ring_const_coeff_digits(gb, s, ring_dim))
        .collect()
}

/// Evaluate the `x_powers` basis table (unit monomials) at point `r4` and return the resulting ring element.
///
/// If `x_powers[i] = x^i` (unit monomial), then the multilinear evaluation produces coefficients that are
/// exactly the tensor weights:
/// \[
///   w = \bigotimes_j [1-r_j, r_j]
/// \]
/// and the evaluated ring element is \(\sum_i w_i x^i\), i.e. coefficient `i` equals `w_i`.
pub(crate) fn eval_x_powers_basis_mle_ring_digits(
    gb: &mut Dr1csBuilder<F257>,
    r4: &[GoldilocksScalar],
    ring_dim: usize,
) -> Result<RingDigits, String> {
    if ring_dim == 0 || !ring_dim.is_power_of_two() {
        return Err("eval_x_powers_basis_mle_ring_digits: ring_dim must be power of two".to_string());
    }
    if (1usize << r4.len()) != ring_dim {
        return Err("eval_x_powers_basis_mle_ring_digits: r4 length mismatch".to_string());
    }
    let _prev = gb.profile_enter("cm_math::eval_x_powers_basis_mle_ring_digits");
    // weights length = ring_dim; each weight is a Goldilocks scalar in digit encoding.
    let weights = tensor_goldilocks_scalars_digits(gb, r4);
    debug_assert_eq!(weights.len(), ring_dim);
    // Interpret weights as the ring coefficients.
    gb.profile_exit(_prev);
    Ok(weights)
}

/// Compute powers \([1, x, x^2, ..., x^n]\) in Goldilocks scalar digit encoding.
pub(crate) fn goldilocks_pow_table_digits(
    gb: &mut Dr1csBuilder<F257>,
    x: &GoldilocksScalar,
    n: usize,
) -> Vec<GoldilocksScalar> {
    let _prev = gb.profile_enter("cm_math::goldilocks_pow_table_digits");
    tiny_cm_bump(|c| c.scalar_pow_table += 1);
    let mut out: Vec<GoldilocksScalar> = Vec::with_capacity(n + 1);
    let mut acc = goldilocks_const_u64_digits(gb, 1u64);
    out.push(acc);
    for _ in 0..n {
        acc = goldilocks_mul_mod_p_digits(gb, &acc, x);
        out.push(acc);
    }
    gb.profile_exit(_prev);
    out
}

/// Evaluate the multilinear equality polynomial `eq(c, r)` over Goldilocks scalars (digit encoding).
///
/// This matches `we_gate_arith::eq_eval_vars` but over the Goldilocks base field:
/// \(\prod_i (c_i r_i + (1-c_i)(1-r_i))\).
pub(crate) fn eq_eval_goldilocks_digits(
    gb: &mut Dr1csBuilder<F257>,
    c: &[GoldilocksScalar],
    r: &[GoldilocksScalar],
) -> Result<GoldilocksScalar, String> {
    if c.len() != r.len() {
        return Err("eq_eval_goldilocks_digits: length mismatch".to_string());
    }
    let _prev = gb.profile_enter("cm_math::eq_eval_goldilocks_digits");
    tiny_cm_bump(|cc| cc.eq_eval_vars += c.len() as u64);
    let mut acc = goldilocks_const_u64_digits(gb, 1u64);
    for (ci, ri) in c.iter().zip(r.iter()) {
        let one_minus_ci = goldilocks_one_minus_digits(gb, ci);
        let one_minus_ri = goldilocks_one_minus_digits(gb, ri);
        let ci_ri = goldilocks_mul_mod_p_digits(gb, ci, ri);
        let om = goldilocks_mul_mod_p_digits(gb, &one_minus_ci, &one_minus_ri);
        let t = goldilocks_add_mod_p_digits(gb, &ci_ri, &om);
        acc = goldilocks_mul_mod_p_digits(gb, &acc, &t);
    }
    gb.profile_exit(_prev);
    Ok(acc)
}

#[inline]
pub(crate) fn ring_const_coeff_digits(gb: &mut Dr1csBuilder<F257>, c0: &GoldilocksScalar, d: usize) -> RingDigits {
    let _prev = gb.profile_enter("cm_math::ring_const_coeff_digits");
    let z = goldilocks_const_u64_digits(gb, 0u64);
    let mut out = vec![z; d];
    if d > 0 {
        out[0] = *c0;
    }
    gb.profile_exit(_prev);
    out
}

#[inline]
pub(crate) fn ring_unit_monomial_digits(gb: &mut Dr1csBuilder<F257>, idx: usize, d: usize) -> RingDigits {
    let _prev = gb.profile_enter("cm_math::ring_unit_monomial_digits");
    let z = goldilocks_const_u64_digits(gb, 0u64);
    let one = goldilocks_const_u64_digits(gb, 1u64);
    let mut out = vec![z; d];
    if idx < d {
        out[idx] = one;
    }
    gb.profile_exit(_prev);
    out
}

/// Evaluate a small MLE table (ring-valued) at point `r` (LSB-first variable order).
///
/// `table.len()` must be a power of two, and `r.len() == log2(table.len())`.
pub(crate) fn eval_small_mle_ring_digits(
    gb: &mut Dr1csBuilder<F257>,
    table: &[RingDigits],
    r: &[GoldilocksScalar],
) -> RingDigits {
    let _prev = gb.profile_enter("cm_math::eval_small_mle_ring_digits");
    debug_assert!(!table.is_empty());
    debug_assert!(table.len().is_power_of_two());
    debug_assert_eq!(table.len(), 1usize << r.len());

    let mut cur: Vec<RingDigits> = table.to_vec();
    for ri in r.iter() {
        let mut next: Vec<RingDigits> = Vec::with_capacity(cur.len() / 2);
        for j in 0..(cur.len() / 2) {
            // Standard multilinear combine:
            // out = a*(1-ri) + b*ri
            //
            // Optimize to use ONE scalar multiplication instead of two:
            // out = a + (b - a) * ri
            let a = &cur[2 * j];
            let b = &cur[2 * j + 1];
            let diff = ring_sub_digits(gb, b, a);
            let t = ring_scale_digits(gb, &diff, ri);
            next.push(ring_add_digits(gb, a, &t));
        }
        cur = next;
    }
    debug_assert_eq!(cur.len(), 1);
    let out = cur.pop().unwrap();
    gb.profile_exit(_prev);
    out
}

/// Evaluate the CM tensor product t(z) at `r`, mirroring `tensor_eval::eval_t_z_optimized`.
///
/// Expected factor order / bit slicing (LSB-first):
/// - `x_powers` (size = d) uses the lowest `log2(d)` bits
/// - `dpp` (size = ell) uses the next `log2(ell)` bits
/// - `s_prime_flat` (size = k*d) uses the next `log2(k*d)` bits
/// - `tensor_c_ring` (size = kappa) uses the next `log2(kappa)` bits
///
/// Any remaining high bits are padded with the factor \(\prod (1 - r_i)\).
pub(crate) fn eval_t_z_optimized_ring_digits(
    gb: &mut Dr1csBuilder<F257>,
    tensor_c_ring: &[RingDigits],
    s_prime_flat: &[RingDigits],
    dpp: &[RingDigits],
    ring_dim: usize,
    r: &[GoldilocksScalar],
) -> Result<RingDigits, String> {
    let sizes = [ring_dim, dpp.len(), s_prime_flat.len(), tensor_c_ring.len()];
    if sizes.iter().any(|&s| s == 0 || !s.is_power_of_two()) {
        return Err("eval_t_z_optimized_ring_digits: expected power-of-two non-empty factor sizes".to_string());
    }
    let vars4 = sizes.map(|s| ark_std::log2(s) as usize);
    let tensor_vars = vars4.iter().sum::<usize>();
    if r.len() < tensor_vars {
        return Err("eval_t_z_optimized_ring_digits: r too short".to_string());
    }
    let _prev = gb.profile_enter("cm_math::eval_t_z_optimized_ring_digits");

    // Split r into chunks (innermost to outermost) as in tensor_eval::eval_t_z_optimized.
    let r4 = &r[0..vars4[0]]; // x_powers (lowest bits)
    let r3 = &r[vars4[0]..vars4[0] + vars4[1]];
    let r2 = &r[vars4[0] + vars4[1]..vars4[0] + vars4[1] + vars4[2]];
    let r1 = &r[vars4[0] + vars4[1] + vars4[2]..tensor_vars];

    let v1 = eval_small_mle_ring_digits(gb, tensor_c_ring, r1);
    let v2 = eval_small_mle_ring_digits(gb, s_prime_flat, r2);
    let v3 = eval_small_mle_ring_digits(gb, dpp, r3);
    // `x_powers` is the unit-monomial basis, evaluated via tensor weights.
    let v4 = match eval_x_powers_basis_mle_ring_digits(gb, r4, ring_dim) {
        Ok(v) => v,
        Err(e) => {
            gb.profile_exit(_prev);
            return Err(e);
        }
    };

    let mut res = match ring_mul_negacyclic_digits_d64(gb, &v1, &v2) {
        Ok(v) => v,
        Err(e) => {
            gb.profile_exit(_prev);
            return Err(e);
        }
    };
    res = match ring_mul_negacyclic_digits_d64(gb, &res, &v3) {
        Ok(v) => v,
        Err(e) => {
            gb.profile_exit(_prev);
            return Err(e);
        }
    };
    res = match ring_mul_negacyclic_digits_d64(gb, &res, &v4) {
        Ok(v) => v,
        Err(e) => {
            gb.profile_exit(_prev);
            return Err(e);
        }
    };

    // Padding factor: Π_{j=tensor_vars..} (1 - r[j]) as scalar.
    let mut pad = goldilocks_const_u64_digits(gb, 1u64);
    for rj in &r[tensor_vars..] {
        let om = goldilocks_one_minus_digits(gb, rj);
        pad = goldilocks_mul_mod_p_digits(gb, &pad, &om);
    }
    let out = Ok(ring_scale_digits(gb, &res, &pad));
    gb.profile_exit(_prev);
    out
}

/// Compute `t0(ro)` and `t1(ro)` together, sharing the expensive common subcomputations.
///
/// This is identical to calling `eval_t_z_optimized_ring_digits` twice, but avoids duplicating:
/// - evaluation of `s_prime_flat` MLE,
/// - evaluation of `dpp` MLE,
/// - basis `x_powers` tensor weights,
/// - pad computation.
pub(crate) fn eval_t_z_optimized_ring_digits_pair(
    gb: &mut Dr1csBuilder<F257>,
    tensor_c0_ring: &[RingDigits],
    tensor_c1_ring: &[RingDigits],
    s_prime_flat: &[RingDigits],
    dpp: &[RingDigits],
    ring_dim: usize,
    r: &[GoldilocksScalar],
) -> Result<(RingDigits, RingDigits), String> {
    let sizes = [ring_dim, dpp.len(), s_prime_flat.len(), tensor_c0_ring.len(), tensor_c1_ring.len()];
    if sizes.iter().any(|&s| s == 0 || !s.is_power_of_two()) {
        return Err("eval_t_z_optimized_ring_digits_pair: expected power-of-two non-empty factor sizes".to_string());
    }
    if tensor_c0_ring.len() != tensor_c1_ring.len() {
        return Err("eval_t_z_optimized_ring_digits_pair: tensor_c length mismatch".to_string());
    }
    let vars4 = [ring_dim, dpp.len(), s_prime_flat.len(), tensor_c0_ring.len()].map(|s| ark_std::log2(s) as usize);
    let tensor_vars = vars4.iter().sum::<usize>();
    if r.len() < tensor_vars {
        return Err("eval_t_z_optimized_ring_digits_pair: r too short".to_string());
    }
    let _prev = gb.profile_enter("cm_math::eval_t_z_optimized_ring_digits_pair");
    let r4 = &r[0..vars4[0]];
    let r3 = &r[vars4[0]..vars4[0] + vars4[1]];
    let r2 = &r[vars4[0] + vars4[1]..vars4[0] + vars4[1] + vars4[2]];
    let r1 = &r[vars4[0] + vars4[1] + vars4[2]..tensor_vars];

    let v2 = eval_small_mle_ring_digits(gb, s_prime_flat, r2);
    let v3 = eval_small_mle_ring_digits(gb, dpp, r3);
    let v4 = match eval_x_powers_basis_mle_ring_digits(gb, r4, ring_dim) {
        Ok(v) => v,
        Err(e) => {
            gb.profile_exit(_prev);
            return Err(e);
        }
    };

    // Common product u = v2*v3*v4.
    let mut u = match ring_mul_negacyclic_digits_d64(gb, &v2, &v3) {
        Ok(v) => v,
        Err(e) => {
            gb.profile_exit(_prev);
            return Err(e);
        }
    };
    u = match ring_mul_negacyclic_digits_d64(gb, &u, &v4) {
        Ok(v) => v,
        Err(e) => {
            gb.profile_exit(_prev);
            return Err(e);
        }
    };

    // v1 differs between t0/t1.
    let v10 = eval_small_mle_ring_digits(gb, tensor_c0_ring, r1);
    let v11 = eval_small_mle_ring_digits(gb, tensor_c1_ring, r1);

    let mut res0 = match ring_mul_negacyclic_digits_d64(gb, &v10, &u) {
        Ok(v) => v,
        Err(e) => {
            gb.profile_exit(_prev);
            return Err(e);
        }
    };
    let mut res1 = match ring_mul_negacyclic_digits_d64(gb, &v11, &u) {
        Ok(v) => v,
        Err(e) => {
            gb.profile_exit(_prev);
            return Err(e);
        }
    };

    // Shared padding factor.
    let mut pad = goldilocks_const_u64_digits(gb, 1u64);
    for rj in &r[tensor_vars..] {
        let om = goldilocks_one_minus_digits(gb, rj);
        pad = goldilocks_mul_mod_p_digits(gb, &pad, &om);
    }
    res0 = ring_scale_digits(gb, &res0, &pad);
    res1 = ring_scale_digits(gb, &res1, &pad);
    let out = Ok((res0, res1));
    gb.profile_exit(_prev);
    out
}

