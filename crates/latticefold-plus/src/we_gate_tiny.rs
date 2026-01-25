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
    PoseidonByteWiring, SparseDr1csInstance,
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

fn frog_p_base128_digits_le() -> [u8; LIMBS_U64] {
    let mut out = [0u8; LIMBS_U64];
    let mut t = FROG_P;
    for i in 0..LIMBS_U64 {
        out[i] = (t & (LIMB_BASE_U64 - 1)) as u8;
        t >>= LIMB_BITS;
    }
    out
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
    let p_digits = frog_p_base128_digits_le();
    let mut borrow = b.new_var(F::ZERO);
    b.enforce_var_eq_const(borrow, F::ZERO);
    let mut borrow_final = borrow;
    for i in 0..LIMBS_U64 {
        // Witness borrow_{i+1} from integers.
        let ui = ((u >> (LIMB_BITS * i)) & (LIMB_BASE_U64 - 1)) as i16;
        let pi = p_digits[i] as i16;
        let bi = if i == 0 { 0i16 } else { borrow_final as i16 };
        let _ = (ui, pi, bi);

        let bnext = if i == LIMBS_U64 - 1 {
            // dummy; will be constrained by the LC.
            alloc_bool::<F>(b, u < FROG_P)
        } else {
            // witness below
            let t = (ui as i32) - (pi as i32) - (if i == 0 { 0 } else { 0 });
            alloc_bool::<F>(b, t < 0)
        };
        // We'll overwrite with a proper witness-based borrow bit next.
        drop(bnext);
    }

    // Proper comparator: run a base-128 borrow chain on (u - p) with witnessed borrows.
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

/// Build a small "glue demo" relation:
/// - Poseidon sponge arithmetization over F257 (WE mode, includes SqueezeBytes byte vars)
/// - Take the first 8 squeezed bytes and canonicalize into a Frog-field element via
///   `u -> (q,z)` with `z < p_frog`.
///
/// This is the core pattern we’ll reuse for transcript→challenge glue in the full gate.
pub fn build_poseidon_f257_with_frog_challenge_glue_from_ops(
    cfg: Option<&PoseidonConfig<F257>>,
    ops: &[PoseidonTraceOp<F257>],
) -> Result<(SparseDr1csInstance<F257>, Vec<F257>), String> {
    let (pose_inst, pose_asg, byte_wiring) = build_poseidon_f257_from_ops_with_bytes(cfg, ops)?;
    if byte_wiring.squeeze_byte_vars.len() < 8 {
        return Err("need at least 8 squeezed bytes for glue demo".to_string());
    }

    // Gadget part: allocate 8 input byte vars with the *same witness values* as Poseidon bytes.
    let mut gb = Dr1csBuilder::<F257>::new();
    gb.enforce_var_eq_const(gb.one(), F257::ONE);
    let mut in_bytes = [0usize; 8];
    for i in 0..8 {
        let v = pose_asg[byte_wiring.squeeze_byte_vars[i]];
        in_bytes[i] = gb.new_var(v);
    }
    let _ = reduce_u64_mod_frog_from_byte_vars::<F257>(&mut gb, &in_bytes);
    let (glue_inst, glue_asg) = gb.into_instance();
    let glue_nvars = glue_inst.nvars;

    // Merge parts (share var-0=ONE), then add explicit equality constraints to bind the
    // gadget input bytes to the Poseidon squeezed bytes.
    let (mut inst, asg) = merge_sparse_dr1cs_share_one::<F257>(&[
        (pose_inst, pose_asg),
        (glue_inst, glue_asg),
    ])
    .map_err(|e| format!("merge poseidon+glue failed: {e}"))?;

    let pose_nvars = inst.nvars - (glue_nvars - 1); // reconstruct nvars0
    let glue_offset = pose_nvars - 1;
    for i in 0..8 {
        let pose_global = byte_wiring.squeeze_byte_vars[i];
        let glue_local = in_bytes[i];
        let glue_global = if glue_local == 0 { 0 } else { glue_local + glue_offset };
        enforce_var_eq::<F257>(&mut inst, pose_global, glue_global);
    }

    debug_assert_eq!(inst.nvars, asg.len());
    Ok((inst, asg))
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
    fn test_poseidon_plus_glue_demo_satisfies() {
        // Record a small trace that actually includes squeezed bytes.
        let mut tr = symphony::transcript::TracePoseidonTranscript::<FrogRing>::empty::<()>();
        tr.absorb_field_element(&<FrogRing as stark_rings::PolyRing>::BaseRing::from(999u64));
        let _ = tr.squeeze_bytes(8);
        let ops: Vec<PoseidonTraceOp<F257>> = tr.trace().ops.clone();

        let (inst, asg) =
            build_poseidon_f257_with_frog_challenge_glue_from_ops(None, &ops)
                .expect("build_poseidon_f257_with_frog_challenge_glue_from_ops");
        inst.check(&asg).expect("poseidon+glue dR1CS satisfied");
    }
}

