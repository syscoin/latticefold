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
/// - `u = z + q * p_frog` as an **integer** (no wrap), enforced via base-256 carries
/// - the top carry is 0, preventing the `+2^64` wrap cheat.
///
/// This is the "single subtract" reduction justified by \(2^{64} < 2p\).
fn reduce_u64_mod_frog_from_bytes<F: PrimeField>(
    b: &mut Dr1csBuilder<F>,
    u_bytes: &[ByteVar; 8],
) -> (usize /* q bit */, [ByteVar; 8] /* z bytes */) {
    // Witness compute.
    let mut u_buf = [0u8; 8];
    for i in 0..8 {
        u_buf[i] = b.assignment[u_bytes[i].byte]
            .into_bigint()
            .to_bytes_le()
            .get(0)
            .copied()
            .unwrap_or(0);
    }
    let u = u64::from_le_bytes(u_buf);
    let q_u8: u8 = if u >= FROG_P { 1 } else { 0 };
    let q = alloc_bool::<F>(b, q_u8 == 1);
    let z_u64 = u.wrapping_sub((q_u8 as u64) * FROG_P);
    let z_bytes = alloc_u64_as_bytes_le::<F>(b, z_u64);

    // Enforce base-256 carry constraints:
    // z_i + q*p_i + carry_i = u_i + 256*carry_{i+1}, with carry_0=0, carry_8=0.
    let p_bytes = FROG_P.to_le_bytes();

    let carry0 = b.new_var(F::ZERO);
    b.enforce_var_eq_const(carry0, F::ZERO);
    let mut carry_i = carry0;
    let mut carry_val: u16 = 0;
    for i in 0..8 {
        let zi = z_u64.to_le_bytes()[i] as u16;
        let pi = p_bytes[i] as u16;
        let sum = zi + (q_u8 as u16) * pi + carry_val;
        let carry_next_val: u8 = (sum >> 8) as u8;
        debug_assert!(carry_next_val <= 1);
        carry_val = (carry_next_val as u16) & 1;

        let carry_next_bit = if i == 7 {
            // Enforce carry_8 == 0 (no wrap by +2^64).
            let v = b.new_var(F::ZERO);
            b.enforce_var_eq_const(v, F::ZERO);
            v
        } else {
            // carry is at most 1 for byte-wise addition here.
            alloc_bool::<F>(b, carry_next_val == 1)
        };

        // z_i + q*p_i + carry_i - u_i - 256*carry_{i+1} == 0
        let mut lc: Vec<(F, usize)> = Vec::new();
        lc.push((F::ONE, z_bytes[i].byte));
        if p_bytes[i] != 0 {
            lc.push((F::from(p_bytes[i] as u64), q));
        }
        lc.push((F::ONE, carry_i));
        lc.push((-F::ONE, u_bytes[i].byte));
        lc.push((-F::from(256u64), carry_next_bit));
        b.enforce_lc_times_one_eq_const(lc);

        carry_i = carry_next_bit;
    }

    // Canonicality: enforce z < p_frog so the mapping is unique (prevents choosing "wrong q").
    let z_lt_p = u64_lt_const_le_bytes::<F>(b, &z_bytes, FROG_P);
    b.enforce_var_eq_const(z_lt_p, F::ONE);

    (q, z_bytes)
}

/// Same as `reduce_u64_mod_frog_from_bytes`, but takes raw byte variables that are
/// already constrained to be 8-bit (e.g. Poseidon `SqueezeBytes` wiring).
#[allow(dead_code)]
fn reduce_u64_mod_frog_from_byte_vars<F: PrimeField>(
    b: &mut Dr1csBuilder<F>,
    u_byte_vars: &[usize; 8],
) -> (usize /* q bit */, [ByteVar; 8] /* z bytes */) {
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
    let z_u64 = u.wrapping_sub((q_u8 as u64) * FROG_P);
    let z_bytes = alloc_u64_as_bytes_le::<F>(b, z_u64);

    let p_bytes = FROG_P.to_le_bytes();
    let carry0 = b.new_var(F::ZERO);
    b.enforce_var_eq_const(carry0, F::ZERO);
    let mut carry_i = carry0;
    let mut carry_val: u16 = 0;
    for i in 0..8 {
        let zi = z_u64.to_le_bytes()[i] as u16;
        let pi = p_bytes[i] as u16;
        let sum = zi + (q_u8 as u16) * pi + carry_val;
        let carry_next_val: u8 = (sum >> 8) as u8;
        debug_assert!(carry_next_val <= 1);
        carry_val = (carry_next_val as u16) & 1;

        let carry_next_bit = if i == 7 {
            let v = b.new_var(F::ZERO);
            b.enforce_var_eq_const(v, F::ZERO);
            v
        } else {
            alloc_bool::<F>(b, carry_next_val == 1)
        };

        let mut lc: Vec<(F, usize)> = Vec::new();
        lc.push((F::ONE, z_bytes[i].byte));
        if p_bytes[i] != 0 {
            lc.push((F::from(p_bytes[i] as u64), q));
        }
        lc.push((F::ONE, carry_i));
        lc.push((-F::ONE, u_byte_vars[i]));
        lc.push((-F::from(256u64), carry_next_bit));
        b.enforce_lc_times_one_eq_const(lc);

        carry_i = carry_next_bit;
    }

    let z_lt_p = u64_lt_const_le_bytes::<F>(b, &z_bytes, FROG_P);
    b.enforce_var_eq_const(z_lt_p, F::ONE);

    (q, z_bytes)
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

    fn bytes_to_u64_le<F: PrimeField>(b: &Dr1csBuilder<F>, x: &[ByteVar; 8]) -> u64 {
        let mut buf = [0u8; 8];
        for i in 0..8 {
            buf[i] = var_to_u8::<F>(b, x[i].byte);
        }
        u64::from_le_bytes(buf)
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
    fn test_reduce_u64_mod_frog_from_bytes_no_wrap() {
        let mut b = Dr1csBuilder::<F257>::new();
        b.enforce_var_eq_const(b.one(), F257::ONE);

        let u = FROG_P + 12345;
        let u_bytes = alloc_u64_as_bytes_le::<F257>(&mut b, u);
        let (_q, _z) = reduce_u64_mod_frog_from_bytes::<F257>(&mut b, &u_bytes);

        let (inst, asg) = b.into_instance();
        inst.check(&asg).expect("reduce_u64_mod_frog_from_bytes dR1CS satisfied");
    }

    #[test]
    fn test_reduce_u64_mod_frog_branch_u_lt_p() {
        let mut b = Dr1csBuilder::<F257>::new();
        b.enforce_var_eq_const(b.one(), F257::ONE);

        let u = 42u64;
        let u_bytes = alloc_u64_as_bytes_le::<F257>(&mut b, u);
        let (q, z) = reduce_u64_mod_frog_from_bytes::<F257>(&mut b, &u_bytes);

        // q should be 0, z == u.
        assert_eq!(var_to_u8::<F257>(&b, q), 0);
        assert_eq!(bytes_to_u64_le::<F257>(&b, &z), u);

        let (inst, asg) = b.into_instance();
        inst.check(&asg).expect("branch u<p satisfied");
    }

    #[test]
    fn test_reduce_u64_mod_frog_branch_u_ge_p() {
        let mut b = Dr1csBuilder::<F257>::new();
        b.enforce_var_eq_const(b.one(), F257::ONE);

        let u = FROG_P + 424242u64;
        let u_bytes = alloc_u64_as_bytes_le::<F257>(&mut b, u);
        let (q, z) = reduce_u64_mod_frog_from_bytes::<F257>(&mut b, &u_bytes);

        // q should be 1, z == u - p.
        assert_eq!(var_to_u8::<F257>(&b, q), 1);
        assert_eq!(bytes_to_u64_le::<F257>(&b, &z), u - FROG_P);

        let (inst, asg) = b.into_instance();
        inst.check(&asg).expect("branch u>=p satisfied");
    }

    #[test]
    fn test_reduce_u64_mod_frog_rejects_wrong_q() {
        let mut b = Dr1csBuilder::<F257>::new();
        b.enforce_var_eq_const(b.one(), F257::ONE);

        let u = FROG_P + 7u64;
        let u_bytes = alloc_u64_as_bytes_le::<F257>(&mut b, u);
        let (q, _z) = reduce_u64_mod_frog_from_bytes::<F257>(&mut b, &u_bytes);

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

