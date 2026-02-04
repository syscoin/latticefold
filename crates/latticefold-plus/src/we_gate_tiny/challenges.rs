use ark_ff::{Field, PrimeField};

use latticefold::transcript::poseidon::F257;
use symphony::dpp_sumcheck::Dr1csBuilder;
use symphony::transcript::PoseidonTraceOp;

use super::op_counts::tiny_cm_bump;

use super::digits::{
    alloc_bal16_digit, f257_to_i32_bal, mul_u32ish9_to_fixed_bal16, u32_bytes_to_bal16_digits,
};
use super::gadgets::decompose_existing_byte_var_to_bits;
use super::params::{DIGITS_PER_TRY, LIMBS_U32, LIMBS_U64};
use super::cm_ir::{
    digit_to_byte_ir, lower_ir_into_builder, select_first_ok_u32_try_digits_ir, u32_bits_to_base128_limbs_ir,
    IrBuilder as CmIrBuilder, VarRef as CmVarRef,
};
use crate::transcript::DEFAULT_REJECTION_TRIES;

/// Explicit wiring of which Poseidon `SqueezeField` ops (by **squeeze-field op index**) are used
/// for each challenge type.
///
/// Indices refer to the order of `PoseidonTraceOp::SqueezeField(..)` occurrences in `ops`.
#[derive(Clone, Debug, Default)]
pub struct TinyCoinOpWiring {
    /// `SqueezeField(len=ring_dim)` op indices for `short_challenge(128)` blocks.
    pub short_squeeze_ops: Vec<usize>,
    /// `SqueezeField(len=8)` **start op indices** for bounded u32 scalar challenges.
    ///
    /// Each logical `get_challenge()` performs `DEFAULT_REJECTION_TRIES` consecutive `SqueezeField(len=8)`
    /// attempts (each followed by an absorb); this vector stores the **first** squeeze-op index of each group.
    pub u32_squeeze_ops: Vec<usize>,
}

/// Helper: number of `short_challenge(128)` blocks used by CM (the `s` and `s_prime` surface).
///
/// Matches `cm.rs`: `s` has 3 blocks, `s_prime` has `k * ring_dim` blocks.
fn cm_short_challenge_blocks(ring_dim: usize, k: usize) -> usize {
    3 + k * ring_dim
}

/// Helper: number of bounded scalar challenges (u32) consumed by CM after `absorb_comh`.
///
/// Matches the historical WE arithmetization shape:
/// - `c0`: `log_kappa` challenges
/// - `c1`: `log_kappa` challenges
/// - `rc0`: 1 challenge
/// - `sumcheck_r0`: `nvars_cm` challenges
/// - `rc1`: 1 challenge
/// - `sumcheck_r1`: `nvars_cm` challenges
fn cm_bounded_u32_challenges(log_kappa: usize, nvars_cm: usize) -> usize {
    2 * log_kappa + 2 + 2 * nvars_cm
}

/// Infer the CM coin op wiring from a Poseidon op schedule.
///
/// This follows the CM transcript schedule in `cm.rs`:
/// - first consume `short_need` occurrences of `SqueezeField(len=ring_dim)` for `short_challenge(128)`
/// - then consume `u32_need` occurrences of `SqueezeField(len=8)` for bounded scalar challenges
///
/// IMPORTANT:
/// - This is a **schedule** parser. It does not validate re-absorbs; Poseidon arithmetization handles that.
/// - If your full transcript has other `SqueezeField(len=8)` before CM scalar challenges, pass an
///   `squeeze_field_op_offset` to start counting from the CM segment.
pub fn infer_cm_coin_op_wiring_from_ops(
    ops: &[PoseidonTraceOp<F257>],
    ring_dim: usize,
    k: usize,
    log_kappa: usize,
    nvars_cm: usize,
    squeeze_field_op_offset: usize,
) -> Result<TinyCoinOpWiring, String> {
    let short_need = cm_short_challenge_blocks(ring_dim, k);
    let u32_need = cm_bounded_u32_challenges(log_kappa, nvars_cm);
    let tries = DEFAULT_REJECTION_TRIES;

    let mut out = TinyCoinOpWiring::default();
    let mut squeeze_field_op_idx = 0usize;
    let mut u32_try_blocks_seen = 0usize;

    // First pass: count squeeze-field ops, and select indices in the desired order.
    //
    // IMPORTANT: only treat `SqueezeField(len=8)` as a `get_challenge()` try if it is immediately
    // followed by `Absorb(len=8)`. This avoids accidentally consuming other squeeze-field ops
    // (e.g. `squeeze_bytes`) that happen to have len=8 but are *not* part of the fixed-tries
    // `get_challenge` schedule.
    for (op_i, op) in ops.iter().enumerate() {
        if let PoseidonTraceOp::SqueezeField(v) = op {
            let next_is_reabsorb_try = matches!(
                ops.get(op_i + 1),
                Some(PoseidonTraceOp::Absorb(a)) if a.len() == DIGITS_PER_TRY
            );
            if squeeze_field_op_idx >= squeeze_field_op_offset {
                if v.len() == ring_dim && out.short_squeeze_ops.len() < short_need {
                    out.short_squeeze_ops
                        .push(squeeze_field_op_idx - squeeze_field_op_offset);
                } else if out.short_squeeze_ops.len() == short_need
                    && v.len() == DIGITS_PER_TRY
                    && next_is_reabsorb_try
                    && u32_try_blocks_seen < u32_need * tries
                {
                    // One logical u32 challenge corresponds to `tries` consecutive squeeze blocks.
                    // Record only the first squeeze-op index of each group.
                    if (u32_try_blocks_seen % tries) == 0 {
                        out.u32_squeeze_ops
                            .push(squeeze_field_op_idx - squeeze_field_op_offset);
                    }
                    u32_try_blocks_seen += 1;
                }
            }
            squeeze_field_op_idx += 1;
        }
        if out.short_squeeze_ops.len() == short_need && out.u32_squeeze_ops.len() == u32_need {
            break;
        }
    }

    if out.short_squeeze_ops.len() != short_need {
        return Err(format!(
            "infer_cm_coin_op_wiring: need {} short squeezes (len={}), got {}",
            short_need,
            ring_dim,
            out.short_squeeze_ops.len()
        ));
    }
    if out.u32_squeeze_ops.len() != u32_need {
        return Err(format!(
            "infer_cm_coin_op_wiring: need {} u32 challenge starts (len=8), got {}",
            u32_need,
            out.u32_squeeze_ops.len()
        ));
    }
    Ok(out)
}

/// Select the first acceptable `get_challenge()` try (fixed schedule).
///
/// Input `digits_by_try` is a flat array of `tries * DIGITS_PER_TRY` digit vars,
/// ordered by try then digit index. Acceptance predicate matches `transcript.rs`:
/// accept iff the first 4 digits are all != 256.
///
/// Returns:
/// - selected digit vars `[d0..d7]` (each equals the corresponding digit of the chosen try)
/// - `found_bit` indicating that an acceptable try exists (enforced by caller if desired)
pub(super) fn select_first_ok_u32_try_digits(
    b: &mut Dr1csBuilder<F257>,
    digits_by_try: &[usize],
    tries: usize,
) -> ([usize; DIGITS_PER_TRY], usize) {
    assert_eq!(digits_by_try.len(), tries * DIGITS_PER_TRY);
    let (ir, out_ir, found_ir) = {
        let base_asg: &[F257] = &b.assignment;
        let mut ib = if b.is_count_only() {
            CmIrBuilder::new_count_only(base_asg)
        } else {
            CmIrBuilder::new(base_asg)
        };
        let digits_ir: Vec<CmVarRef> = digits_by_try.iter().copied().map(CmVarRef::Base).collect();
        let (out_ir, found_ir) = select_first_ok_u32_try_digits_ir(&mut ib, &digits_ir, tries);
        (ib.ir, out_ir, found_ir)
    };
    let lowered = lower_ir_into_builder(b, ir);
    let out: [usize; DIGITS_PER_TRY] = core::array::from_fn(|i| lowered.map_var(out_ir[i]));
    let found = lowered.map_var(found_ir);
    (out, found)
}

pub(super) fn squeeze_field_ranges_by_op_index(
    squeeze_field_ranges: &[(usize, usize)],
    op_indices: &[usize],
) -> Result<Vec<(usize, usize)>, String> {
    let mut out = Vec::with_capacity(op_indices.len());
    for &op_idx in op_indices {
        let r = squeeze_field_ranges
            .get(op_idx)
            .copied()
            .ok_or_else(|| format!("squeeze_field op idx {} out of range", op_idx))?;
        out.push(r);
    }
    Ok(out)
}

/// Wiring for a bounded scalar challenge (u32) derived from a `get_challenge()` digit block.
#[derive(Clone, Debug)]
pub struct BoundedU32ChallengeWiring {
    /// The 8 digit vars (F257) consumed for this challenge (schedule-only).
    pub digit_vars: Vec<usize>,
    /// The 4 byte-view vars (256 -> 0) packed into the u32.
    pub byte_vars: [usize; 4],
    /// Base-128 limbs (little-endian) representing the u32 value.
    pub limbs: [usize; LIMBS_U32],
    /// Balanced base-16 digits (little-endian), length 9, representing the same u32 value.
    pub bal16_digits: Vec<usize>,
    /// Balanced base-16 digits (little-endian), length 18, representing `u^2` as an integer.
    pub bal16_sq_digits: Vec<usize>,
}

/// Wiring for a Goldilocks-field challenge derived from a `get_challenge()` digit block.
///
/// We also expose the residue mod 257 for cheap tiny-field “stage 0” CM constraints.
#[derive(Clone, Debug)]
pub struct GoldilocksChallengeWiring {
    /// The 8 digit vars (F257) consumed for this challenge (schedule-only).
    pub digit_vars: Vec<usize>,
    /// The 8 byte-view vars (256 -> 0), little-endian.
    pub byte_vars: [usize; 8],
    /// Reduction bit `q ∈ {0,1}` such that u64 = z + q*p_Goldilocks.
    pub q_bit: usize,
    /// Reduced Goldilocks value `z` as base-128 limbs (little-endian).
    pub limbs: [usize; LIMBS_U64],
    /// The residue of the u64 byte-view modulo 257 (in F257), i.e. Σ (-1)^i * byte[i].
    pub res257: usize,
}

/// Wiring for short challenges `short_challenge(128)` over a ring of dimension `ring_dim`.
#[derive(Clone, Debug)]
pub struct ShortChallengeWiring {
    /// The digit vars (F257), length = `ring_dim`.
    pub digit_vars: Vec<usize>,
    /// The byte-view vars (256 -> 0), length = `ring_dim`.
    pub byte_vars: Vec<usize>,
    /// The coefficient vars (in F257), length = `ring_dim`.
    pub coeff_vars: Vec<usize>,
    /// Balanced base-16 digits (little-endian) for each coefficient (3 digits per coeff).
    pub coeff_bal16_digits: Vec<[usize; 3]>,
}

/// Map a base-257 digit `d ∈ {0..=256}` to a byte `b ∈ {0..=255}` via the transcript rule:
/// `256 -> 0`, else `b=d`.
///
/// In F257 arithmetic this is: `b = d + is_eq256(d)`, since `256 ≡ -1 (mod 257)`.
pub(super) fn digit_to_byte_var(b: &mut Dr1csBuilder<F257>, d: usize) -> usize {
    // IR is the source of truth for the "256 -> 0" digit→byte mapping.
    // Keep byte bit-decomposition/caching in the builder layer.
    let (ir, byte_ir) = {
        let base_asg: &[F257] = &b.assignment;
        let mut ib = if b.is_count_only() {
            CmIrBuilder::new_count_only(base_asg)
        } else {
            CmIrBuilder::new(base_asg)
        };
        let byte_ir = digit_to_byte_ir(&mut ib, CmVarRef::Base(d));
        (ib.ir, byte_ir)
    };
    let lowered = lower_ir_into_builder(b, ir);
    let byte = lowered.map_var(byte_ir);

    // Range-check byte to 8 bits.
    let _bits = decompose_existing_byte_var_to_bits::<F257>(b, byte);
    byte
}

/// Interpret the transcript's `get_challenge()` digit block as a **bounded u32** and return it as
/// base-128 limbs (little-endian).
///
/// Semantics must match `latticefold-plus/src/transcript.rs:get_challenge`:
/// - consume a fixed 8-digit block (schedule-only)
/// - map the first 4 digits through the byte view (256 -> 0)
/// - pack into a u32, but represent as base-128 limbs (so it embeds soundly in F257)
pub(super) fn bounded_u32_from_8_digits_base128(
    b: &mut Dr1csBuilder<F257>,
    digits: &[usize; DIGITS_PER_TRY],
) -> ([usize; LIMBS_U32], [usize; 4], Vec<usize>, Vec<usize>) {
    // First 4 digits -> bytes in 0..255.
    let mut bytes = [0usize; 4];
    for i in 0..4 {
        bytes[i] = digit_to_byte_var(b, digits[i]);
    }

    // Balanced base-16 digits (len 9) for the same u32 (used by CM mul gadgets).
    let bal16_digits = u32_bytes_to_bal16_digits(b, bytes);
    let bal16_sq_digits = mul_u32ish9_to_fixed_bal16(b, &bal16_digits, &bal16_digits, 18);

    // Bits 0..31 of the u32, little-endian.
    let mut bits32: [usize; 32] = [0usize; 32];
    for i in 0..4 {
        let bits = decompose_existing_byte_var_to_bits::<F257>(b, bytes[i]);
        for j in 0..8 {
            bits32[i * 8 + j] = bits[j];
        }
    }

    // Group bits into 7-bit base-128 limbs (IR is the source of truth).
    let limbs: [usize; LIMBS_U32] = {
        let (ir, limbs_ir) = {
            let base_asg: &[F257] = &b.assignment;
            let mut ib = if b.is_count_only() {
                CmIrBuilder::new_count_only(base_asg)
            } else {
                CmIrBuilder::new(base_asg)
            };
            let bits_ir: [CmVarRef; 32] = core::array::from_fn(|i| CmVarRef::Base(bits32[i]));
            let limbs_ir = u32_bits_to_base128_limbs_ir(&mut ib, &bits_ir);
            (ib.ir, limbs_ir)
        };
        let lowered = lower_ir_into_builder(b, ir);
        core::array::from_fn(|i| lowered.map_var(limbs_ir[i]))
    };

    (limbs, bytes, bal16_digits, bal16_sq_digits)
}

#[inline]
pub(super) fn res257_from_u64_bytes_le(gb: &mut Dr1csBuilder<F257>, bytes: &[usize; 8]) -> usize {
    // 256 ≡ -1 (mod 257), so Σ byte[i] * 256^i ≡ Σ (-1)^i * byte[i].
    let mut acc = F257::ZERO;
    let mut lc: Vec<(F257, usize)> = Vec::with_capacity(1 + 8);
    for i in 0..8 {
        let sign = if (i & 1) == 0 { F257::ONE } else { -F257::ONE };
        acc += gb.assignment[bytes[i]] * sign;
        lc.push((-sign, bytes[i]));
    }
    let v = gb.new_var(acc);
    lc.insert(0, (F257::ONE, v));
    gb.enforce_lc_times_one_eq_const(lc);
    v
}

pub(super) fn short_challenge_coeff_from_byte_var<F: PrimeField>(
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
    // Avoid `to_bytes_le()` allocations: read low limb directly.
    let limb0: u64 = b
        .assignment[byte_var]
        .into_bigint()
        .as_ref()
        .get(0)
        .copied()
        .unwrap_or(0);
    let low_u64 = (limb0 & 0xFF) & (u - 1);
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

pub(super) fn short_challenge_from_bytes<F: PrimeField>(
    b: &mut Dr1csBuilder<F>,
    bytes: &[usize],
    lambda: usize,
    ring_dim: usize,
) -> Vec<usize> {
    tiny_cm_bump(|c| c.short_challenge_from_bytes += 1);
    // Same parameterization as LF+/WE: u = 2^(lambda / d).
    debug_assert_eq!(bytes.len(), ring_dim);
    let exp = (lambda / ring_dim) as u32;
    let u = 1u64 << exp;
    bytes
        .iter()
        .map(|&by| short_challenge_coeff_from_byte_var::<F>(b, by, u))
        .collect()
}

pub(super) fn short_challenge_from_digits_128(
    b: &mut Dr1csBuilder<F257>,
    digit_vars: &[usize], // length = ring_dim
    ring_dim: usize,
) -> (Vec<usize> /* byte_vars */, Vec<usize> /* coeff_vars */, Vec<[usize; 3]> /* coeff digits */) {
    debug_assert_eq!(digit_vars.len(), ring_dim);
    let mut byte_vars = Vec::with_capacity(ring_dim);
    for &d in digit_vars {
        byte_vars.push(digit_to_byte_var(b, d));
    }
    let coeff_vars = short_challenge_from_bytes::<F257>(b, &byte_vars, 128, ring_dim);

    // Constrain balanced base-16 digits for each coeff (range [-128,127] always holds here).
    let mut coeff_bal16_digits: Vec<[usize; 3]> = Vec::with_capacity(ring_dim);
    let zero_digit = alloc_bal16_digit(b, 0);
    for &cv in &coeff_vars {
        let v = f257_to_i32_bal(b.assignment[cv]); // in [-128,127]
        debug_assert!((-128..=127).contains(&v));
        // Choose d0 in [-8,7] and d1 in [-8,7] such that v = d0 + 16*d1.
        let mut d0 = v % 16;
        if d0 > 7 {
            d0 -= 16;
        }
        if d0 < -8 {
            d0 += 16;
        }
        let d1 = (v - d0) / 16;
        debug_assert!((-8..=7).contains(&d0));
        debug_assert!((-8..=7).contains(&d1));

        let d0v = alloc_bal16_digit(b, d0 as i8);
        let d1v = alloc_bal16_digit(b, d1 as i8);
        // Enforce cv = d0 + 16*d1 (safe: RHS in [-136,119] < 257).
        b.enforce_lc_times_one_eq_const(vec![
            (F257::ONE, cv),
            (-F257::ONE, d0v),
            (-F257::from(16u64), d1v),
        ]);
        coeff_bal16_digits.push([d0v, d1v, zero_digit]);
    }

    (byte_vars, coeff_vars, coeff_bal16_digits)
}

