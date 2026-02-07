use ark_ff::{Field, PrimeField};
use latticefold::transcript::poseidon::F257;
use symphony::dpp_sumcheck::Dr1csBuilder;

use rayon::prelude::*;

use super::op_counts::tiny_cm_bump;
use super::gadgets::alloc_byte;
use super::digits::f257_to_i32_bal;
use super::goldilocks::{
    goldilocks_scalar_from_u64_bytes_le_digits,
    goldilocks_p_bal16_digits_le_const, GoldilocksScalar,
    goldilocks_u64_enforce_lt_p_from_byte_vars,
};
use super::cm_ir::{
    bal4_to_bal16_digits_ir, goldilocks_add_mod_p_digits_ir, goldilocks_mul_const_mod_p_digits_bal4_ir,
    goldilocks_mul_mod_p_digits_bal4_ir, goldilocks_mul_mod_p_digits_ir, goldilocks_sub_mod_p_digits_ir, lower_ir_into_builder,
    ring_mul_negacyclic_ntt_goldilocks_d64_bal4_ir, IrBuilder, VarRef as IrVarRef,
};
use cyclotomic_rings::rings::GoldilocksRing64 as GR64;
use stark_rings::{psi, unit_monomial, CoeffRing};
use std::sync::OnceLock;

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
fn modinv_u64(p: u64, a: u64) -> u64 {
    // Extended Euclid over i128 (p fits in u64).
    fn egcd(a: i128, b: i128) -> (i128, i128, i128) {
        if a == 0 {
            return (b, 0, 1);
        }
        let (g, x, y) = egcd(b % a, a);
        (g, y - (b / a) * x, x)
    }
    let p_i = p as i128;
    let (g, x, _y) = egcd(a as i128, p_i);
    debug_assert!(g == 1 || g == -1);
    let x = if g == -1 { -x } else { x };
    x.rem_euclid(p_i) as u64
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

/// Boundary helper: convert a digit-encoded Goldilocks scalar to canonical 8-byte little-endian encoding.
///
/// This is used when we must export byte surfaces (e.g. `tcch0/tcch1`), but want all *math* to stay in digits.
///
/// Soundness:
/// - Allocates byte vars from the witness implied by `x_digits`.
/// - Enforces each byte is 8-bit, enforces `< p`, then re-parses the bytes back into digits and equates them to `x_digits`.
#[allow(dead_code)]
pub(crate) fn goldilocks_digits_to_bytes_canonical(
    gb: &mut Dr1csBuilder<F257>,
    x_digits: &GoldilocksScalar,
) -> [usize; 8] {
    let _prev = gb.profile_enter("cm_math::goldilocks_digits_to_bytes_canonical");
    let p_u64 = crate::we_goldilocks_poseidon_f257::GOLDILOCKS_P;

    // Witness compute: evaluate Σ d_i * 16^i in i128 then reduce mod p.
    let mut acc: i128 = 0;
    let mut pow: i128 = 1;
    for i in 0..17 {
        let di = f257_to_i32_bal(gb.assignment[x_digits[i]]) as i128;
        acc += di * pow;
        pow *= 16;
    }
    let v_u64 = acc.rem_euclid(p_u64 as i128) as u64;
    let bs = v_u64.to_le_bytes();

    // Allocate byte vars (with bit decompositions) for the canonical encoding.
    let mut out = [0usize; 8];
    for i in 0..8 {
        out[i] = alloc_byte::<F257>(gb, bs[i]).byte;
    }
    goldilocks_u64_enforce_lt_p_from_byte_vars::<F257>(gb, &out);

    // Re-parse bytes into digits and equate to the provided digits.
    let d2 = goldilocks_bytes_to_digits(gb, out);
    for i in 0..17 {
        gb.enforce_lc_times_one_eq_const(vec![(F257::ONE, d2[i]), (-F257::ONE, x_digits[i])]);
    }

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
#[track_caller]
pub(crate) fn ring_eq_digits(gb: &mut Dr1csBuilder<F257>, a: &RingDigits, b: &RingDigits) {
    // Optional debug: include callsite in the scope label to localize failing equalities.
    // Enable with `LFP_WE_GATE_OPMIX=1`.
    #[inline]
    fn dbg_ring_eq_caller_on() -> bool {
        static ON: OnceLock<bool> = OnceLock::new();
        *ON.get_or_init(|| match std::env::var("LFP_WE_GATE_OPMIX") {
            Ok(v) => v != "0",
            Err(_) => false,
        })
    }

    let dbg = dbg_ring_eq_caller_on();
    let prev_scope = if dbg {
        let loc = std::panic::Location::caller();
        let s = format!("cm_math::ring_eq_digits@{}:{}", loc.file(), loc.line());
        let leaked: &'static str = Box::leak(s.into_boxed_str());
        let prev = gb.profile_current;
        gb.profile_current = Some(leaked);
        prev
    } else {
        gb.profile_enter("cm_math::ring_eq_digits")
    };
    tiny_cm_bump(|c| c.ring_eq += 1);
    debug_assert_eq!(a.len(), b.len());
    for (ai, bi) in a.iter().zip(b.iter()) {
        for j in 0..17 {
            gb.enforce_lc_times_one_eq_const(vec![(F257::ONE, ai[j]), (-F257::ONE, bi[j])]);
        }
    }
    if dbg {
        gb.profile_current = prev_scope;
    } else {
        gb.profile_exit(prev_scope);
    }
}

#[inline]
pub(crate) fn ring_add_digits(gb: &mut Dr1csBuilder<F257>, a: &RingDigits, b: &RingDigits) -> RingDigits {
    let _prev = gb.profile_enter("cm_math::ring_add_digits");
    tiny_cm_bump(|c| c.ring_add += 1);
    debug_assert_eq!(a.len(), b.len());
    // Scalar ops come overwhelmingly from ring ops; build per-coefficient shards in parallel.
    tiny_cm_bump(|c| c.scalar_add += a.len() as u64);
    let base_asg: &[F257] = &gb.assignment;
    let count_only = gb.is_count_only();
    let frags: Vec<(_, [IrVarRef; 17])> = (0..a.len())
        .into_par_iter()
        .map(|i| {
            let mut ib = if count_only {
                IrBuilder::new_count_only(base_asg)
            } else {
                IrBuilder::new(base_asg)
            };
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
    let count_only = gb.is_count_only();
    let frags: Vec<(_, [IrVarRef; 17])> = (0..a.len())
        .into_par_iter()
        .map(|i| {
            let mut ib = if count_only {
                IrBuilder::new_count_only(base_asg)
            } else {
                IrBuilder::new(base_asg)
            };
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
    let count_only = gb.is_count_only();
    let s_ir: [IrVarRef; 17] = core::array::from_fn(|j| IrVarRef::Base(s[j]));
    // Build a single IR fragment for the whole ring-scaling, so the IR-side caches (notably
    // `bal16_to_bal4_digits_cached`) can reuse the scalar `s` decomposition across all coefficients.
    let p_u64 = crate::we_goldilocks_poseidon_f257::GOLDILOCKS_P;
    let p_d = goldilocks_p_bal16_digits_le_const();
    let mut ib = if count_only {
        IrBuilder::new_count_only(base_asg)
    } else {
        IrBuilder::new(base_asg)
    };
    let mut out_ir: Vec<[IrVarRef; 17]> = Vec::with_capacity(a.len());
    for i in 0..a.len() {
        let ai: [IrVarRef; 17] = core::array::from_fn(|j| IrVarRef::Base(a[i][j]));
        let oi = goldilocks_mul_mod_p_digits_ir(&mut ib, &ai, &s_ir, p_u64, &p_d);
        out_ir.push(oi);
    }

    let lowered = lower_ir_into_builder(gb, ib.ir);
    let mut out: RingDigits = Vec::with_capacity(a.len());
    for oi in out_ir {
        let digits: GoldilocksScalar = core::array::from_fn(|j| lowered.map_var(oi[j]));
        out.push(digits);
    }
    gb.profile_exit(_prev);
    out
}

// -----------------------------------------------------------------------------
// Scalar digit-domain ops (IR is source of truth)
// -----------------------------------------------------------------------------

#[inline]
pub(crate) fn goldilocks_add_mod_p_digits(
    gb: &mut Dr1csBuilder<F257>,
    a: &GoldilocksScalar,
    c: &GoldilocksScalar,
) -> GoldilocksScalar {
    let _prev = gb.profile_enter("cm_math::scalar_add_mod_p_digits");
    tiny_cm_bump(|cc| cc.scalar_add += 1);
    let p_u64 = crate::we_goldilocks_poseidon_f257::GOLDILOCKS_P;
    let p_d = goldilocks_p_bal16_digits_le_const();
    let base_asg: &[F257] = &gb.assignment;
    let mut ib = if gb.is_count_only() {
        IrBuilder::new_count_only(base_asg)
    } else {
        IrBuilder::new(base_asg)
    };
    let a_ir: [IrVarRef; 17] = core::array::from_fn(|j| IrVarRef::Base(a[j]));
    let c_ir: [IrVarRef; 17] = core::array::from_fn(|j| IrVarRef::Base(c[j]));
    let out_ir = goldilocks_add_mod_p_digits_ir(&mut ib, &a_ir, &c_ir, p_u64, &p_d);
    let lowered = lower_ir_into_builder(gb, ib.ir);
    let out = core::array::from_fn(|j| lowered.map_var(out_ir[j]));
    gb.profile_exit(_prev);
    out
}

#[inline]
pub(crate) fn goldilocks_sub_mod_p_digits(
    gb: &mut Dr1csBuilder<F257>,
    a: &GoldilocksScalar,
    c: &GoldilocksScalar,
) -> GoldilocksScalar {
    let _prev = gb.profile_enter("cm_math::scalar_sub_mod_p_digits");
    tiny_cm_bump(|cc| cc.scalar_sub += 1);
    let p_u64 = crate::we_goldilocks_poseidon_f257::GOLDILOCKS_P;
    let p_d = goldilocks_p_bal16_digits_le_const();
    let base_asg: &[F257] = &gb.assignment;
    let mut ib = if gb.is_count_only() {
        IrBuilder::new_count_only(base_asg)
    } else {
        IrBuilder::new(base_asg)
    };
    let a_ir: [IrVarRef; 17] = core::array::from_fn(|j| IrVarRef::Base(a[j]));
    let c_ir: [IrVarRef; 17] = core::array::from_fn(|j| IrVarRef::Base(c[j]));
    let out_ir = goldilocks_sub_mod_p_digits_ir(&mut ib, &a_ir, &c_ir, p_u64, &p_d);
    let lowered = lower_ir_into_builder(gb, ib.ir);
    let out = core::array::from_fn(|j| lowered.map_var(out_ir[j]));
    gb.profile_exit(_prev);
    out
}

#[inline]
pub(crate) fn goldilocks_mul_mod_p_digits(
    gb: &mut Dr1csBuilder<F257>,
    a: &GoldilocksScalar,
    c: &GoldilocksScalar,
) -> GoldilocksScalar {
    let _prev = gb.profile_enter("cm_math::scalar_mul_mod_p_digits");
    tiny_cm_bump(|cc| cc.scalar_mul += 1);
    let p_u64 = crate::we_goldilocks_poseidon_f257::GOLDILOCKS_P;
    let p_d = goldilocks_p_bal16_digits_le_const();
    let base_asg: &[F257] = &gb.assignment;
    let mut ib = if gb.is_count_only() {
        IrBuilder::new_count_only(base_asg)
    } else {
        IrBuilder::new(base_asg)
    };
    let a_ir: [IrVarRef; 17] = core::array::from_fn(|j| IrVarRef::Base(a[j]));
    let c_ir: [IrVarRef; 17] = core::array::from_fn(|j| IrVarRef::Base(c[j]));
    let out_ir = goldilocks_mul_mod_p_digits_ir(&mut ib, &a_ir, &c_ir, p_u64, &p_d);
    let lowered = lower_ir_into_builder(gb, ib.ir);
    let out = core::array::from_fn(|j| lowered.map_var(out_ir[j]));
    gb.profile_exit(_prev);
    out
}

#[inline]
pub(crate) fn goldilocks_mul_const_mod_p_digits(
    gb: &mut Dr1csBuilder<F257>,
    a: &GoldilocksScalar,
    k: u64,
) -> GoldilocksScalar {
    let _prev = gb.profile_enter("cm_math::scalar_mul_const_mod_p_digits");
    tiny_cm_bump(|cc| cc.scalar_mul_const += 1);
    let p_u64 = crate::we_goldilocks_poseidon_f257::GOLDILOCKS_P;
    debug_assert!(k < p_u64);
    let base_asg: &[F257] = &gb.assignment;
    let mut ib = if gb.is_count_only() {
        IrBuilder::new_count_only(base_asg)
    } else {
        IrBuilder::new(base_asg)
    };
    let a_ir: [IrVarRef; 17] = core::array::from_fn(|j| IrVarRef::Base(a[j]));
    let a4 = ib.bal16_to_bal4_digits_cached(&a_ir);
    let r4 = goldilocks_mul_const_mod_p_digits_bal4_ir(&mut ib, &a4, k, p_u64);
    let out_ir = bal4_to_bal16_digits_ir(&mut ib, &r4);
    let lowered = lower_ir_into_builder(gb, ib.ir);
    let out = core::array::from_fn(|j| lowered.map_var(out_ir[j]));
    gb.profile_exit(_prev);
    out
}

/// Evaluate a ring element (coefficient vector) at a base-field scalar point `x`.
///
/// This matches `setchk::ev` and `we_gate_arith::ring_eval_at_scalar`:
/// \(\sum_{t=0}^{d-1} coeff[t] \cdot x^t\).
pub(crate) fn ring_eval_at_scalar_digits(
    gb: &mut Dr1csBuilder<F257>,
    coeffs: &RingDigits,
    x: &GoldilocksScalar,
) -> Result<GoldilocksScalar, String> {
    if coeffs.is_empty() {
        return Err("ring_eval_at_scalar_digits: empty ring element".to_string());
    }
    let _prev = gb.profile_enter("cm_math::ring_eval_at_scalar_digits");
    let zero_bytes = alloc_const_goldilocks_u64(gb, 0u64);
    let one_bytes = alloc_const_goldilocks_u64(gb, 1u64);
    let zero = goldilocks_bytes_to_digits(gb, zero_bytes);
    let one = goldilocks_bytes_to_digits(gb, one_bytes);
    let mut acc = zero;
    let mut pow = one;
    for c in coeffs {
        let t = goldilocks_mul_mod_p_digits(gb, c, &pow);
        acc = goldilocks_add_mod_p_digits(gb, &acc, &t);
        pow = goldilocks_mul_mod_p_digits(gb, &pow, x);
    }
    gb.profile_exit(_prev);
    Ok(acc)
}

/// Compute `ct(psi * x)` as a Goldilocks scalar (digit encoding), for ring_dim=64.
///
/// This is the digit-domain analog of `we_gate_arith::ct_psi_mul_ring`, and is used by the rgchk checks.
pub(crate) fn ct_psi_mul_ring_digits_d64(
    gb: &mut Dr1csBuilder<F257>,
    x: &RingDigits,
) -> Result<GoldilocksScalar, String> {
    if x.len() != 64 {
        return Err("ct_psi_mul_ring_digits_d64: expected ring_dim=64".to_string());
    }
    let _prev = gb.profile_enter("cm_math::ct_psi_mul_ring_digits_d64");
    tiny_cm_bump(|cc| cc.ct_psi_mul_ring += 1);

    let psi_r = psi::<GR64>();
    let zero_bytes = alloc_const_goldilocks_u64(gb, 0u64);
    let zero = goldilocks_bytes_to_digits(gb, zero_bytes);
    let mut acc = zero;
    for j in 0..64 {
        let basis = unit_monomial::<GR64>(j);
        let w_br = (psi_r * basis).ct();
        let w_u64 = w_br.into_bigint().as_ref().get(0).copied().unwrap_or(0);
        if w_u64 == 0 {
            continue;
        }
        let t = goldilocks_mul_const_mod_p_digits(gb, &x[j], w_u64);
        acc = goldilocks_add_mod_p_digits(gb, &acc, &t);
    }
    gb.profile_exit(_prev);
    Ok(acc)
}

/// Degree-2 Lagrange basis in digit encoding.
pub(crate) fn lagrange_degree2_goldilocks_digits(
    gb: &mut Dr1csBuilder<F257>,
    r: &GoldilocksScalar,
    one: &GoldilocksScalar,
    two: &GoldilocksScalar,
) -> (GoldilocksScalar, GoldilocksScalar, GoldilocksScalar) {
    let inv2_u64 = (crate::we_goldilocks_poseidon_f257::GOLDILOCKS_P + 1) / 2;
    // t1 = r - 1, t2 = r - 2
    let t1 = goldilocks_sub_mod_p_digits(gb, r, one);
    let t2 = goldilocks_sub_mod_p_digits(gb, r, two);

    // L0 = (r-1)(r-2)/2
    let p = goldilocks_mul_mod_p_digits(gb, &t1, &t2);
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

/// Degree-3 Lagrange basis in digit encoding.
pub(crate) fn lagrange_degree3_goldilocks_digits(
    gb: &mut Dr1csBuilder<F257>,
    r: &GoldilocksScalar,
    one: &GoldilocksScalar,
    two: &GoldilocksScalar,
    three: &GoldilocksScalar,
) -> (GoldilocksScalar, GoldilocksScalar, GoldilocksScalar, GoldilocksScalar) {
    let p_u64 = crate::we_goldilocks_poseidon_f257::GOLDILOCKS_P;
    let inv2_u64 = (p_u64 + 1) / 2;
    let inv6_u64 = modinv_u64(p_u64, 6);

    // t1 = r-1, t2 = r-2, t3 = r-3
    let t1 = goldilocks_sub_mod_p_digits(gb, r, one);
    let t2 = goldilocks_sub_mod_p_digits(gb, r, two);
    let t3 = goldilocks_sub_mod_p_digits(gb, r, three);
    let zero = goldilocks_const_u64_digits(gb, 0u64);

    // L0 = -((r-1)(r-2)(r-3))/6
    let p01 = goldilocks_mul_mod_p_digits(gb, &t1, &t2);
    let p012 = goldilocks_mul_mod_p_digits(gb, &p01, &t3);
    let p012_div6 = goldilocks_mul_const_mod_p_digits(gb, &p012, inv6_u64);
    let l0 = goldilocks_sub_mod_p_digits(gb, &zero, &p012_div6);

    // L1 = r(r-2)(r-3)/2
    let p1 = goldilocks_mul_mod_p_digits(gb, r, &t2);
    let p1 = goldilocks_mul_mod_p_digits(gb, &p1, &t3);
    let l1 = goldilocks_mul_const_mod_p_digits(gb, &p1, inv2_u64);

    // L2 = -r(r-1)(r-3)/2
    let p2 = goldilocks_mul_mod_p_digits(gb, r, &t1);
    let p2 = goldilocks_mul_mod_p_digits(gb, &p2, &t3);
    let p2_div2 = goldilocks_mul_const_mod_p_digits(gb, &p2, inv2_u64);
    let l2 = goldilocks_sub_mod_p_digits(gb, &zero, &p2_div2);

    // L3 = r(r-1)(r-2)/6
    let p3 = goldilocks_mul_mod_p_digits(gb, r, &t1);
    let p3 = goldilocks_mul_mod_p_digits(gb, &p3, &t2);
    let l3 = goldilocks_mul_const_mod_p_digits(gb, &p3, inv6_u64);

    (l0, l1, l2, l3)
}

/// Verify a degree-3 sumcheck over ring elements (digit-encoded) and return the final claim.
pub(crate) fn sumcheck_verify_degree3_ring_digits(
    gb: &mut Dr1csBuilder<F257>,
    mut claimed_sum: RingDigits,
    msgs: &[[RingDigits; 4]],
    rs: &[GoldilocksScalar],
) -> Result<RingDigits, String> {
    if msgs.len() != rs.len() {
        return Err("sumcheck_verify_degree3_ring_digits: msgs/rs length mismatch".to_string());
    }
    let _prev = gb.profile_enter("cm_math::sumcheck_verify_degree3_ring_digits");
    let d = claimed_sum.len();

    let one = goldilocks_const_u64_digits(gb, 1u64);
    let two = goldilocks_const_u64_digits(gb, 2u64);
    let three = goldilocks_const_u64_digits(gb, 3u64);

    for (m, r) in msgs.iter().zip(rs.iter()) {
        if m[0].len() != d || m[1].len() != d || m[2].len() != d || m[3].len() != d {
            gb.profile_exit(_prev);
            return Err("sumcheck_verify_degree3_ring_digits: ring dimension mismatch".to_string());
        }

        // g(0) + g(1) == claim
        let g01 = ring_add_digits(gb, &m[0], &m[1]);
        ring_eq_digits(gb, &g01, &claimed_sum);

        // claim = g(r)
        let (l0, l1, l2, l3) = lagrange_degree3_goldilocks_digits(gb, r, &one, &two, &three);
        let t0 = ring_scale_digits(gb, &m[0], &l0);
        let t1 = ring_scale_digits(gb, &m[1], &l1);
        let t2 = ring_scale_digits(gb, &m[2], &l2);
        let t3 = ring_scale_digits(gb, &m[3], &l3);
        let s01 = ring_add_digits(gb, &t0, &t1);
        let s23 = ring_add_digits(gb, &t2, &t3);
        claimed_sum = ring_add_digits(gb, &s01, &s23);
    }

    gb.profile_exit(_prev);
    Ok(claimed_sum)
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

    for (m, r) in msgs.iter().zip(rs.iter()) {
        if m[0].len() != d || m[1].len() != d || m[2].len() != d {
            gb.profile_exit(_prev);
            return Err("sumcheck_verify_degree2_ring_digits: ring dimension mismatch".to_string());
        }

        // g(0) + g(1) == claim
        let g01 = ring_add_digits(gb, &m[0], &m[1]);
        ring_eq_digits(gb, &g01, &claimed_sum);

        // claim = g(r)
        let (l0, l1, l2) = lagrange_degree2_goldilocks_digits(gb, r, &one, &two);
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
    // To make the resulting coefficient ordering match the MLE evaluation used elsewhere in the
    // CM verifier (and in `we_gate_arith`), we must reverse `r4` here so that `r4[0]` becomes the
    // fastest-changing (LSB) variable.
    let r4_rev: Vec<GoldilocksScalar> = r4.iter().copied().rev().collect();
    // weights length = ring_dim; each weight is a Goldilocks scalar in digit encoding.
    let weights = tensor_goldilocks_scalars_digits(gb, &r4_rev);
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

    // Important performance/memory optimization:
    // avoid cloning the entire `table` up-front (each entry is a full ring element).
    //
    // Instead, do the first combine directly from references into `table`, producing `table.len()/2`
    // fresh ring elements, and then proceed level-by-level on the shrinking owned vector.
    if r.is_empty() {
        let out = table[0].clone();
        gb.profile_exit(_prev);
        return out;
    }

    let mut cur_refs: Vec<&RingDigits> = table.iter().collect();
    let mut cur_owned: Vec<RingDigits> = Vec::new();
    for ri in r.iter() {
        let mut next: Vec<RingDigits> = Vec::with_capacity(cur_refs.len() / 2);
        for j in 0..(cur_refs.len() / 2) {
            // Standard multilinear combine:
            // out = a*(1-ri) + b*ri
            //
            // Optimize to use ONE scalar multiplication instead of two:
            // out = a + (b - a) * ri
            let a = cur_refs[2 * j];
            let b = cur_refs[2 * j + 1];
            let diff = ring_sub_digits(gb, b, a);
            let t = ring_scale_digits(gb, &diff, ri);
            next.push(ring_add_digits(gb, a, &t));
        }
        cur_owned = next;
        cur_refs = cur_owned.iter().collect();
    }
    debug_assert_eq!(cur_owned.len(), 1);
    let out = cur_owned.pop().unwrap();
    gb.profile_exit(_prev);
    out
}

/// Compute `t0(ro)` and `t1(ro)` together, sharing the expensive common subcomputations.
///
/// This is logically identical to calling the single-evaluation routine twice, but avoids duplicating:
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
) -> Result<([[usize; 33]; 64], [[usize; 33]; 64]), String> {
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
    if ring_dim != 64 {
        return Err("eval_t_z_optimized_ring_digits_pair: expected ring_dim=64".to_string());
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

    // v1 differs between t0/t1.
    let v10 = eval_small_mle_ring_digits(gb, tensor_c0_ring, r1);
    let v11 = eval_small_mle_ring_digits(gb, tensor_c1_ring, r1);

    // Shared padding factor.
    let mut pad = goldilocks_const_u64_digits(gb, 1u64);
    for rj in &r[tensor_vars..] {
        let om = goldilocks_one_minus_digits(gb, rj);
        pad = goldilocks_mul_mod_p_digits(gb, &pad, &om);
    }

    // Big optimization: keep the heavy ring multiplications + pad scaling in the bal4 domain end-to-end.
    // This avoids materializing `t0(ro)`/`t1(ro)` as bal16 only to immediately convert them back to bal4
    // at the callsite (t(z) terms).
    //
    // This builds a single IR fragment:
    //   u  = v2*v3*v4
    //   t0 = v10*u*pad
    //   t1 = v11*u*pad
    // and returns the resulting ring elements as bal4 base vars.
    let p_u64 = crate::we_goldilocks_poseidon_f257::GOLDILOCKS_P;
    let (ir, out0_4, out1_4) = {
        #[inline]
        fn ringdigits64_to_ir(a: &RingDigits) -> Result<[[IrVarRef; 17]; 64], String> {
            if a.len() != 64 {
                return Err("eval_t_z_optimized_ring_digits_pair: expected ring_dim=64".to_string());
            }
            Ok(core::array::from_fn(|i| core::array::from_fn(|j| IrVarRef::Base(a[i][j]))))
        }

        let v2_16 = ringdigits64_to_ir(&v2)?;
        let v3_16 = ringdigits64_to_ir(&v3)?;
        let v4_16 = ringdigits64_to_ir(&v4)?;
        let v10_16 = ringdigits64_to_ir(&v10)?;
        let v11_16 = ringdigits64_to_ir(&v11)?;
        let pad16: [IrVarRef; 17] = core::array::from_fn(|k| IrVarRef::Base(pad[k]));

        let base_asg: &[F257] = gb.assignment.as_slice();
        let mut ib = if gb.is_count_only() {
            IrBuilder::new_count_only(base_asg)
        } else {
            IrBuilder::new(base_asg)
        };

        let v2_4: [[IrVarRef; 33]; 64] = core::array::from_fn(|i| ib.bal16_to_bal4_digits_cached(&v2_16[i]));
        let v3_4: [[IrVarRef; 33]; 64] = core::array::from_fn(|i| ib.bal16_to_bal4_digits_cached(&v3_16[i]));
        let v4_4: [[IrVarRef; 33]; 64] = core::array::from_fn(|i| ib.bal16_to_bal4_digits_cached(&v4_16[i]));
        let v10_4: [[IrVarRef; 33]; 64] = core::array::from_fn(|i| ib.bal16_to_bal4_digits_cached(&v10_16[i]));
        let v11_4: [[IrVarRef; 33]; 64] = core::array::from_fn(|i| ib.bal16_to_bal4_digits_cached(&v11_16[i]));
        let pad4: [IrVarRef; 33] = ib.bal16_to_bal4_digits_cached(&pad16);

        let u_4 = ring_mul_negacyclic_ntt_goldilocks_d64_bal4_ir(&mut ib, &v2_4, &v3_4);
        let u_4 = ring_mul_negacyclic_ntt_goldilocks_d64_bal4_ir(&mut ib, &u_4, &v4_4);

        let t0_4 = ring_mul_negacyclic_ntt_goldilocks_d64_bal4_ir(&mut ib, &v10_4, &u_4);
        let t1_4 = ring_mul_negacyclic_ntt_goldilocks_d64_bal4_ir(&mut ib, &v11_4, &u_4);

        let mut out0_4: [[IrVarRef; 33]; 64] = [[IrVarRef::Base(0); 33]; 64];
        let mut out1_4: [[IrVarRef; 33]; 64] = [[IrVarRef::Base(0); 33]; 64];
        for i in 0..64 {
            out0_4[i] = goldilocks_mul_mod_p_digits_bal4_ir(&mut ib, &t0_4[i], &pad4, p_u64);
            out1_4[i] = goldilocks_mul_mod_p_digits_bal4_ir(&mut ib, &t1_4[i], &pad4, p_u64);
        }

        // Keep op-mix accounting consistent with the old formulation (4 ring-muls + 2 ring-scales).
        tiny_cm_bump(|c| {
            c.ring_mul_negacyclic += 4;
            c.ring_scale += 2;
            c.scalar_mul += (2 * 64) as u64;
        });

        (ib.ir, out0_4, out1_4)
    };
    let lowered = lower_ir_into_builder(gb, ir);
    let out0_base: [[usize; 33]; 64] = core::array::from_fn(|i| core::array::from_fn(|k| lowered.map_var(out0_4[i][k])));
    let out1_base: [[usize; 33]; 64] = core::array::from_fn(|i| core::array::from_fn(|k| lowered.map_var(out1_4[i][k])));

    let out = Ok((out0_base, out1_base));
    gb.profile_exit(_prev);
    out
}

