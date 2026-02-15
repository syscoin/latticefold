//! WE/DPP gate arithmetization for LatticeFold+ (sparse dR1CS over a prime field).
//!
//! This module is a research/bench frontend: it arithmetizes the *verifier* computation,
//! keeping the relation log-scale in `n` and linear in the verifier-visible message sizes.

use ark_ff::{Field, PrimeField};
use latticefold::transcript::poseidon::f257_poseidon_config;
use latticefold::transcript::poseidon::F257;
use stark_rings::{CoeffRing, OverField, PolyRing, Zq};

use crate::recording_transcript::PoseidonTranscriptTrace;
use crate::we_gate_tiny as tiny;
use crate::we_statement::WeParams;
use crate::transcript::DEFAULT_REJECTION_TRIES;

/// Base prime field alias for ring types.
type BF<R> = <<R as PolyRing>::BaseRing as Field>::BasePrimeField;

/// Number of F257 digits per base-257 challenge (= 8 for Goldilocks).
const CHALLENGE_DIGITS: usize = 8;
const BIND_DIGEST_BYTES: usize = 32;
const CV_PREFIX_BYTES: usize = 64; // 8 Goldilocks elems × 8 bytes each

#[derive(Clone, Debug)]
pub struct WeStatementBindingWitness {
    pub vk_hash: [u8; 32],
    pub r1cs_digest: [u8; 32],
    pub gate_digest: [u8; 32],
    /// The **exact bytes absorbed** by the verifier transcript for the first 8 SP1 public words,
    /// i.e. the 8 fixed-width (8-byte) little-endian encodings of the centered-embedded Goldilocks
    /// base-field elements.
    ///
    /// This is the binding bridge: LF+/sumcheck verifies over transcript absorbs; we bind this
    /// absorbed prefix to the statement digest.
    pub committed_values_prefix_bytes: [u8; CV_PREFIX_BYTES],
}

#[cfg(feature = "we_gate")]
fn stmt_hash_ops_ids_only(
    vk_hash: [u8; 32],
    r1cs_digest: [u8; 32],
    gate_digest: [u8; 32],
) -> Vec<symphony::transcript::PoseidonTraceOp<F257>> {
    use symphony::transcript::PoseidonTraceOp;
    let mut ops: Vec<PoseidonTraceOp<F257>> = Vec::new();
    ops.push(PoseidonTraceOp::Absorb(vec![F257::from(1u64)])); // domain=1
    ops.push(PoseidonTraceOp::Absorb(
        vk_hash.iter().map(|&b| F257::from(b as u64)).collect(),
    ));
    ops.push(PoseidonTraceOp::Absorb(
        r1cs_digest.iter().map(|&b| F257::from(b as u64)).collect(),
    ));
    ops.push(PoseidonTraceOp::Absorb(
        gate_digest.iter().map(|&b| F257::from(b as u64)).collect(),
    ));
    ops.push(PoseidonTraceOp::SqueezeField(vec![F257::ZERO; BIND_DIGEST_BYTES]));
    ops
}

#[cfg(feature = "we_gate")]
fn stmt_hash_ops_stmt_only(
    params: &WeParams,
    committed_values_prefix_bytes: [u8; CV_PREFIX_BYTES],
) -> Vec<symphony::transcript::PoseidonTraceOp<F257>> {
    use symphony::transcript::PoseidonTraceOp;
    let mut ops: Vec<PoseidonTraceOp<F257>> = Vec::new();
    let params_f257 = params.to_field_vec::<F257>();
    ops.push(PoseidonTraceOp::Absorb(vec![F257::from(2u64)])); // domain=2
    // h_ids absorb is 32 elems (filled at witness time via extra_eqs to ids_hash squeeze outputs)
    ops.push(PoseidonTraceOp::Absorb(vec![F257::ZERO; BIND_DIGEST_BYTES]));
    ops.push(PoseidonTraceOp::Absorb(params_f257));
    ops.push(PoseidonTraceOp::Absorb(
        committed_values_prefix_bytes
            .iter()
            .map(|&b| F257::from(b as u64))
            .collect(),
    ));
    ops.push(PoseidonTraceOp::SqueezeField(vec![F257::ZERO; BIND_DIGEST_BYTES]));
    ops
}

use symphony::dpp_sumcheck::Dr1csBuilder;
use symphony::file_backed_dr1cs::{
    merge_file_backed_sparse_dr1cs_share_one, FileBackedSparseDr1csInstance,
};

#[cfg(feature = "we_gate")]
fn collect_nonreabsorb_absorb_ranges(
    ops: &[symphony::transcript::PoseidonTraceOp<F257>],
    pose_wiring: &symphony::dpp_poseidon::PoseidonDr1csWiring,
) -> Result<Vec<(usize, usize)>, String> {
    use symphony::transcript::PoseidonTraceOp;
    let mut out: Vec<(usize, usize)> = Vec::new();
    let mut absorb_idx = 0usize;
    let mut expect_reabsorb = false;
    for (op_i, op) in ops.iter().enumerate() {
        match op {
            PoseidonTraceOp::SqueezeField(v) => {
                expect_reabsorb = v.len() == CHALLENGE_DIGITS
                    && matches!(
                        ops.get(op_i + 1),
                        Some(PoseidonTraceOp::Absorb(a)) if a.len() == CHALLENGE_DIGITS
                    );
            }
            PoseidonTraceOp::Absorb(_v) => {
                let (ab_start, ab_len) = *pose_wiring
                    .absorb_ranges
                    .get(absorb_idx)
                    .ok_or("tiny gate: pose_wiring.absorb_ranges oob (collect_nonreabsorb_absorb_ranges)")?;
                absorb_idx += 1;
                let is_reabsorb = expect_reabsorb;
                expect_reabsorb = false;
                if is_reabsorb {
                    continue;
                }
                out.push((ab_start, ab_len));
            }
            PoseidonTraceOp::SqueezeBytes { .. } => {
                expect_reabsorb = false;
            }
        }
    }
    Ok(out)
}

#[cfg(feature = "we_gate")]
fn first_squeeze_field_op_index_of_len(
    ops: &[symphony::transcript::PoseidonTraceOp<F257>],
    len: usize,
) -> Result<usize, String> {
    let mut sf_idx = 0usize;
    for op in ops {
        if let symphony::transcript::PoseidonTraceOp::SqueezeField(v) = op {
            if v.len() == len {
                return Ok(sf_idx);
            }
            sf_idx += 1;
        }
    }
    Err(format!(
        "first_squeeze_field_op_index_of_len: no SqueezeField(len={len}) op found"
    ))
}

/// Collect `SqueezeField(len=CHALLENGE_DIGITS)` indices that correspond to `get_challenge()`.
///
/// In our transcript/trace, `get_challenge()` is recorded as:
/// - `SqueezeField(len=CHALLENGE_DIGITS)`
/// - `Absorb(len=CHALLENGE_DIGITS)` (Fiat–Shamir re-absorb)
///
/// This helper returns indices in the **SqueezeField-occurrence index space** (same convention as
/// `first_squeeze_field_op_index_of_len` and `TinyCoinOpWiring`).
#[cfg(feature = "we_gate")]
fn collect_get_challenge_squeeze_field_indices(
    ops: &[symphony::transcript::PoseidonTraceOp<F257>],
    sf_start: usize,
    sf_end: usize,
) -> Vec<usize> {
    let mut out = Vec::new();
    let mut sf_idx = 0usize;
    let mut try_seen = 0usize;
    for (i, op) in ops.iter().enumerate() {
        if let symphony::transcript::PoseidonTraceOp::SqueezeField(v) = op {
            let my_sf = sf_idx;
            sf_idx += 1;
            if my_sf < sf_start || my_sf >= sf_end {
                continue;
            }
            if v.len() != CHALLENGE_DIGITS {
                continue;
            }
            // get_challenge() immediately re-absorbs the squeezed elements.
            if let Some(symphony::transcript::PoseidonTraceOp::Absorb(a)) = ops.get(i + 1) {
                if a.len() == CHALLENGE_DIGITS {
                    // Fixed-tries rejection: return only the *start* squeeze-op index for each
                    // logical challenge (one per DEFAULT_REJECTION_TRIES consecutive tries).
                    if (try_seen % DEFAULT_REJECTION_TRIES) == 0 {
                        out.push(my_sf);
                    }
                    try_seen += 1;
                }
            }
        }
    }
    out
}

#[cfg(feature = "we_gate")]
fn build_we_dr1cs_for_plus_proof_shape_tiny<R>(
    trace: &PoseidonTranscriptTrace<BF<R>>,
    params: &WeParams,
    _public_inputs: &[BF<R>],
    n_lin_proofs: usize,
    mlen_mats: usize,
    pairs: &[(usize, usize)],
    out_dir: impl AsRef<std::path::Path>,
) -> Result<WeDr1csShape<F257>, String>
where
    R: OverField + CoeffRing + PolyRing,
    R::BaseRing: Zq + Field + PrimeField,
{
    // Statement public surface is exactly `stmt_digest` as 32 F257 elements.
    let public_inputs_len = BIND_DIGEST_BYTES;

    // Shape-only must still build the **full faithful** tiny-gate relation.
    //
    // We therefore supply a dummy proof-shaped `TinyExtraWitness` (all zeros) so the instance
    // includes *all* constraints (no "satisfiable placeholder" fallbacks).
    let proof = dummy_plus_proof_shape::<R>(params, mlen_mats, n_lin_proofs)?;
    let extra = tiny_extra_witness_from_plus_proof::<R>(params, &proof, mlen_mats)?;
    let ring_dim = R::dimension();

    // IMPORTANT: to guarantee shape == witness builder, use the *caller-provided* recorded trace
    // (rather than regenerating it here). This prevents any accidental schedule drift between the
    // arm-time and witness-time builders.
    //
    // The caller is responsible for ensuring `trace` corresponds to `public_inputs` and params.
    let ops_f257 = tiny::lift_recording_trace_ops_to_f257::<BF<R>>(&trace.ops)?;

    let k = params.k as usize;
    let log_kappa = ark_std::log2((params.kappa as usize).next_power_of_two()) as usize;
    let nvars_cm = params.nvars_cm as usize;

    // The CM segment begins at the first `SqueezeField(len=ring_dim)` (short challenges).
    let squeeze_field_op_offset = first_squeeze_field_op_index_of_len(&ops_f257, ring_dim)?;
    // Also collect the *prefix* get_challenge() squeezes so the tiny gate has access to all
    // transcript scalar coins before CM begins (needed for full verifier arithmetization).
    let prefix_u32_squeeze_ops =
        collect_get_challenge_squeeze_field_indices(&ops_f257, 0, squeeze_field_op_offset);
    let wiring_rel = tiny::infer_cm_coin_op_wiring_from_ops(
        &ops_f257,
        ring_dim,
        k,
        log_kappa,
        nvars_cm,
        squeeze_field_op_offset,
    )?;
    let mut wiring_abs = tiny::TinyCoinOpWiring::default();
    wiring_abs.short_squeeze_ops = wiring_rel
        .short_squeeze_ops
        .into_iter()
        .map(|i| i + squeeze_field_op_offset)
        .collect();
    wiring_abs.u32_squeeze_ops = wiring_rel
        .u32_squeeze_ops
        .into_iter()
        .map(|i| i + squeeze_field_op_offset)
        .collect();
    // Prepend prefix u32 challenges (keeps op order stable).
    wiring_abs
        .u32_squeeze_ops
        .splice(0..0, prefix_u32_squeeze_ops.into_iter());

    let (
        inst_pose,
        asg_pose_shape,
        _shorts,
        _u32s,
        _goldilocks,
        _surfaces_mul,
        _surfaces_sq,
        pose_wiring,
    ) =
        tiny::we_tiny_f257_build_cm_gate_from_trace_ops(
            None,
            &ops_f257,
            ring_dim,
            params,
            &wiring_abs,
            pairs,
            &extra,
            {
                let out_dir = out_dir.as_ref();
                out_dir.join("tiny_gate")
            },
        )?;

    // Public statement prefix (arm-time bound): [ONE] || [stmt_digest(32)].
    // WeParams are statement constants in-circuit, but not public inputs.
    // Build as a file-backed instance so the full tiny-shape pipeline is file-backed.
    let out_dir = out_dir.as_ref();
    let mut b_params = Dr1csBuilder::<F257>::new_file_backed(out_dir.join("params_prefix"))
        .map_err(|e| format!("tiny gate: params prefix new_file_backed failed: {e}"))?;
    // IMPORTANT (disk footprint / reuse-base fast path):
    // Keep this prefix module **constraint-free** so the outer file-backed merge can reuse the
    // dominant tiny-gate part directory as the final `merged/` output (zero-copy on unix).
    //
    // The constant-1 slot is already enforced by the main tiny-gate relation; and the merge code
    // requires `assignment[0] == 1` for every part.
    // Reserve public slots for stmt_digest field elements.
    for _ in 0..public_inputs_len {
        b_params.new_var(F257::from(0u64));
    }
    // Append WeParams as fixed witness-side constants.
    for &x in &params.to_field_vec::<F257>() {
        b_params.new_var(x);
    }
    let (params_inst, params_asg) = b_params
        .into_file_backed_instance()
        .map_err(|e| format!("tiny gate: params prefix into_file_backed_instance failed: {e}"))?;

    // Glue constraints between params prefix public digest slots and inner Poseidon vars.
    let mut extra_eqs: Vec<(usize, usize)> = Vec::new();
    let mut parts: Vec<(FileBackedSparseDr1csInstance<F257>, Vec<F257>)> =
        vec![(params_inst, params_asg), (inst_pose, asg_pose_shape)];
    {
        let _params_nvars = parts[0].1.len();
        let _tiny_nvars = parts[1].1.len();

        // Binding subrelation (must match `we_statement_hash_lf_plus` in `we_statement.rs`):
        //
        // IMPORTANT: statement hashing must be **proof-agnostic**. It must NOT be computed as a
        // continuation of the verifier transcript sponge (which depends on the proof).
        //
        // We therefore arithmetize independent Poseidon sponge gadgets (fresh sponges):
        // - ids_hash:  h_ids = Poseidon(ds=1 || vk_hash || r1cs_digest || gate_digest)
        // - stmt_hash: stmt  = Poseidon(ds=2 || h_ids || params(10) || committed_values_prefix_bytes(64))
        // and constrain `stmt == stmt_digest` (public statement).
        let cfg = f257_poseidon_config();
        let ops_ids = stmt_hash_ops_ids_only([0u8; 32], [0u8; 32], [0u8; 32]);
        let (inst_ids, asg_ids, wiring_ids) =
            symphony::dpp_poseidon::poseidon_sponge_dr1cs_from_ops_with_wiring_no_bytes_file_backed::<F257>(
                &cfg,
                &ops_ids,
                out_dir.join("ids_hash"),
            )
            .map_err(|e| format!("tiny gate: ids_hash poseidon build failed: {e:?}"))?;

        let ops_stmt = stmt_hash_ops_stmt_only(params, [0u8; CV_PREFIX_BYTES]);
        let (inst_stmt, asg_stmt, wiring_stmt) =
            symphony::dpp_poseidon::poseidon_sponge_dr1cs_from_ops_with_wiring_no_bytes_file_backed::<F257>(
                &cfg,
                &ops_stmt,
                out_dir.join("stmt_hash"),
            )
            .map_err(|e| format!("tiny gate: stmt_hash poseidon build failed: {e:?}"))?;

        // ---------------------------------------------------------------------
        // Bind SP1 committed-values digest lane to the verifier transcript prefix.
        //
        // The verifier transcript (and thus sumcheck/CM verification) depends on the absorbed
        // public-input prefix. We bind the first 8 absorbed public inputs (the committed-values
        // digest lane) to the exact 64 bytes absorbed by the transcript for those 8 elements.
        const CV_WORDS: usize = 8;
        let coeff_bytes = ((<R::BaseRing as PrimeField>::MODULUS_BIT_SIZE as usize) + 7) / 8;
        if coeff_bytes != 8 {
            return Err(format!(
                "tiny gate: expected 8-byte fixed-width encoding for base-field absorbs (got coeff_bytes={})",
                coeff_bytes
            ));
        }
        let canon_absorb_ranges = collect_nonreabsorb_absorb_ranges(&ops_f257, &pose_wiring)?;
        let absorb_len8_ranges: Vec<(usize, usize)> = canon_absorb_ranges
            .into_iter()
            .filter(|&(_st, ln)| ln == coeff_bytes)
            .collect();
        let cv_start_idx: usize = 0;
        if absorb_len8_ranges.len() < cv_start_idx.saturating_add(CV_WORDS) {
            return Err(format!(
                "tiny gate: need {} absorbed public-input elements of {} bytes to bind committed_values_prefix_bytes (found={} start_idx={})",
                CV_WORDS,
                coeff_bytes,
                absorb_len8_ranges.len(),
                cv_start_idx,
            ));
        }
        let cv_absorb_ranges: Vec<(usize, usize)> =
            absorb_len8_ranges[cv_start_idx..cv_start_idx + CV_WORDS].to_vec();

        // Glue module enforcing:
        // - absorbed bytes for the first 8 public-input elements equal `committed_values_prefix_bytes`
        let mut gb = Dr1csBuilder::<F257>::new_file_backed(out_dir.join("cv_prefix_glue"))
            .map_err(|e| format!("tiny gate: cv_prefix_glue new_file_backed failed: {e}"))?;
        gb.enforce_var_eq_const(gb.one(), F257::ONE);
        let mut digest_locals: Vec<usize> = Vec::with_capacity(CV_PREFIX_BYTES);
        for _ in 0..CV_PREFIX_BYTES {
            digest_locals.push(gb.new_var(F257::ZERO));
        }
        let mut absorb_locals: Vec<usize> = Vec::with_capacity(CV_WORDS * coeff_bytes);
        for _ in 0..(CV_WORDS * coeff_bytes) {
            absorb_locals.push(gb.new_var(F257::ZERO));
        }
        for i in 0..(CV_WORDS * coeff_bytes) {
            let v_ab = absorb_locals[i];
            let v_d = digest_locals[i];
            gb.enforce_lc_times_one_eq_const(vec![(F257::ONE, v_ab), (-F257::ONE, v_d)]);
        }
        let (glue_inst, glue_asg) = gb
            .into_file_backed_instance()
            .map_err(|e| format!("tiny gate: cv_prefix_glue into_file_backed_instance failed: {e}"))?;
        parts.push((glue_inst, glue_asg));
        parts.push((inst_ids, asg_ids));
        parts.push((inst_stmt, asg_stmt));

        // Compute per-part variable tail offsets for remapping local var indices into merged space.
        let mut var_tail_off: Vec<usize> = Vec::with_capacity(parts.len());
        let mut cur: usize = 0;
        for (inst, _asg) in &parts {
            var_tail_off.push(cur);
            cur = cur.saturating_add(inst.nvars.saturating_sub(1));
        }
        let remap = |part: usize, local: usize, var_tail_off: &[usize]| -> usize {
            if local == 0 { 0 } else { local + var_tail_off[part] }
        };

        // Glue digest locals to the prefix-byte vars inside the stmt_hash absorb (cv_prefix bytes).
        let (stmt_cv_ab_start, stmt_cv_ab_len) = *wiring_stmt
            .absorb_ranges
            .last()
            .ok_or("tiny gate: stmt_hash missing absorb_ranges")?;
        if stmt_cv_ab_len != CV_PREFIX_BYTES {
            return Err("tiny gate: stmt_hash cv_prefix absorb length mismatch".to_string());
        }
        for i in 0..CV_PREFIX_BYTES {
            let digest_byte_var = wiring_stmt.absorb_vars[stmt_cv_ab_start + i];
            extra_eqs.push((
                remap(4, digest_byte_var, &var_tail_off),
                remap(2, digest_locals[i], &var_tail_off),
            ));
        }
        // Glue absorbed public-input bytes (first 8 elements) to absorb_locals.
        for i in 0..CV_WORDS {
            let (ab_start, ab_len) = cv_absorb_ranges[i];
            debug_assert_eq!(ab_len, coeff_bytes);
            for j in 0..coeff_bytes {
                let pub_byte_var = pose_wiring.absorb_vars[ab_start + j];
                extra_eqs.push((
                    remap(1, pub_byte_var, &var_tail_off),
                    remap(2, absorb_locals[i * coeff_bytes + j], &var_tail_off),
                ));
            }
        }

        // Glue stmt_hash h_ids absorb vars to ids_hash squeeze outputs.
        let (ids_sf_start, ids_sf_len) = *wiring_ids
            .squeeze_field_ranges
            .last()
            .ok_or("tiny gate: ids_hash missing squeeze_field_ranges")?;
        if ids_sf_len != BIND_DIGEST_BYTES {
            return Err("tiny gate: ids_hash squeeze len mismatch".to_string());
        }
        let (stmt_ids_ab_start, stmt_ids_ab_len) = *wiring_stmt
            .absorb_ranges
            .get(1)
            .ok_or("tiny gate: stmt_hash missing h_ids absorb range")?;
        if stmt_ids_ab_len != BIND_DIGEST_BYTES {
            return Err("tiny gate: stmt_hash h_ids absorb len mismatch".to_string());
        }
        for j in 0..BIND_DIGEST_BYTES {
            let ids_sf_var = wiring_ids.squeeze_field_vars[ids_sf_start + j];
            let stmt_ids_var = wiring_stmt.absorb_vars[stmt_ids_ab_start + j];
            extra_eqs.push((
                remap(3, ids_sf_var, &var_tail_off),
                remap(4, stmt_ids_var, &var_tail_off),
            ));
        }

        // Enforce stmt_hash output equals stmt_digest public vars in params_prefix.
        let (stmt_sf_start, stmt_sf_len) = *wiring_stmt
            .squeeze_field_ranges
            .last()
            .ok_or("tiny gate: stmt_hash missing squeeze_field_ranges")?;
        if stmt_sf_len != BIND_DIGEST_BYTES {
            return Err("tiny gate: stmt_hash squeeze len mismatch".to_string());
        }
        for j in 0..BIND_DIGEST_BYTES {
            let stmt_sf_var = wiring_stmt.squeeze_field_vars[stmt_sf_start + j];
            let pub_var = 1usize + j;
            extra_eqs.push((
                remap(4, stmt_sf_var, &var_tail_off),
                remap(0, pub_var, &var_tail_off),
            ));
        }

        // Enforce stmt_hash params absorb equals params vars in params_prefix.
        let (stmt_params_ab_start, stmt_params_ab_len) = *wiring_stmt
            .absorb_ranges
            .get(2)
            .ok_or("tiny gate: stmt_hash missing params absorb range")?;
        if stmt_params_ab_len != 10 {
            return Err("tiny gate: stmt_hash params absorb len mismatch".to_string());
        }
        for j in 0..10usize {
            let stmt_params_var = wiring_stmt.absorb_vars[stmt_params_ab_start + j];
            let params_var = 1usize + 32usize + j;
            extra_eqs.push((
                remap(4, stmt_params_var, &var_tail_off),
                remap(0, params_var, &var_tail_off),
            ));
        }
    }

    let (inst, _asg) = merge_file_backed_sparse_dr1cs_share_one::<F257>(
        parts,
        out_dir.join("merged"),
        &extra_eqs,
    )?;
    Ok(WeDr1csShape { inst, public_len: 1 + public_inputs_len })
}


/// Shape-only WE gate output (arm-time artifact): fixed **file-backed** instance + public prefix length.
///
/// This is the canonical shape type for the tiny-field WE gate.
#[derive(Clone, Debug)]
pub struct WeDr1csShape<F: PrimeField> {
    pub inst: FileBackedSparseDr1csInstance<F>,
    pub public_len: usize,
}

#[cfg(feature = "we_gate")]
#[allow(dead_code)]
fn poseidon_trace_schedule_for_plus<R>(
    public_inputs_len: usize,
    params: &WeParams,
    n_lin_proofs: usize,
    mlen_mats: usize,
) -> Result<PoseidonTranscriptTrace<BF<R>>, String>
where
    R: OverField + CoeffRing + PolyRing,
    R::BaseRing: Zq + PrimeField,
{
    let public_inputs = vec![BF::<R>::ZERO; public_inputs_len];
    poseidon_trace_schedule_for_plus_with_public_inputs::<R>(&public_inputs, params, n_lin_proofs, mlen_mats)
}

/// Schedule generator with explicit public inputs.
///
/// This is useful for any caller that wants *statement binding* (public inputs absorbed into the
/// transcript prefix) without having to run the prover. The returned trace is self-consistent:
/// all squeeze outputs match the sponge state induced by the chosen absorbs.
#[cfg(feature = "we_gate")]
pub(crate) fn poseidon_trace_schedule_for_plus_with_public_inputs<R>(
    public_inputs: &[BF<R>],
    params: &WeParams,
    n_lin_proofs: usize,
    mlen_mats: usize,
) -> Result<PoseidonTranscriptTrace<BF<R>>, String>
where
    R: OverField + CoeffRing + PolyRing,
    R::BaseRing: Zq + PrimeField,
{
    // IMPORTANT: We must return a *self-consistent* transcript trace whose squeeze outputs match
    // the sponge state evolution induced by the absorbs. This is required now that the tiny gate
    // enforces Fiat–Shamir chaining (`get_challenge` re-absorbs its squeeze output).
    use crate::recording_transcript::TracePoseidonTranscript;
    use latticefold::transcript::Transcript;

    let d = R::dimension();
    let mut tr = TracePoseidonTranscript::<R>::empty::<()>();

    // Public inputs absorbed as base-field scalars.
    for pi in public_inputs {
        tr.absorb_field_element(pi);
    }

    // Π_lin proofs (ComR1CSProof::verify schedule).
    let nvars_lin = params.nvars_setchk as usize;
    for _ in 0..n_lin_proofs {
        // r = transcript.get_challenges(nvars)
        for _ in 0..nvars_lin {
            let _ = tr.get_challenge();
        }
        // absorb (nvars, degree=3) as scalars (matches MLSumcheck::verify_as_subprotocol header)
        tr.absorb_field_element(&BF::<R>::from(nvars_lin as u64));
        tr.absorb_field_element(&BF::<R>::from(3u64));
        // rounds: 4 ring evals + challenge + explicit absorb
        for _ in 0..nvars_lin {
            for _ in 0..4 {
                tr.absorb(&R::ZERO);
            }
            let ri = tr.get_challenge();
            // Verifier explicitly absorbs the sampled challenge as a scalar.
            tr.absorb_field_element(&ri);
        }
        // absorb (v,va,vb,vc) (ring, len=d each)
        for _ in 0..4 {
            tr.absorb(&R::ZERO);
        }
    }

    // --------------------------------------------------------------------
    // CmProof::verify transcript schedule (Dcom prefix + CM proper).
    // --------------------------------------------------------------------
    //
    // `l_instances` is the number of “instances” carried by the CM proof (Dcom commitments,
    // evaluation tables, etc). For the LF+ schedule, this matches the number of Π_lin proofs.
    if n_lin_proofs == 0 {
        return Err("poseidon_trace_schedule_for_plus: n_lin_proofs must be >= 1".to_string());
    }
    let l_instances = n_lin_proofs;
    let kappa = params.kappa as usize;
    let k_rg = params.k as usize;
    let out_nvars = nvars_lin;
    let out_e0_len = k_rg;
    let out_b_len = 1usize;
    let dcom_evals_len = 1usize;
    let dcom_eval_vec_len = 1 + mlen_mats;

    // Dcom::verify: absorb witness commitments (cm_f, C_Mf, cm_mtau), each ring elem.
    for _ in 0..l_instances {
        for _ in 0..kappa {
            tr.absorb(&R::ZERO);
        }
        for _ in 0..kappa {
            tr.absorb(&R::ZERO);
        }
        for _ in 0..kappa {
            tr.absorb(&R::ZERO);
        }
    }

    // Out::verify coins: per claim sample c (nvars), beta, alpha.
    let nclaims = out_e0_len + out_b_len;
    for _ in 0..nclaims {
        for _ in 0..out_nvars {
            let _ = tr.get_challenge();
        }
        let _ = tr.get_challenge(); // beta
        let _ = tr.get_challenge(); // alpha
    }
    if out_e0_len > 1 {
        let _ = tr.get_challenge(); // rc
    }

    // MLSumcheck::verify_as_subprotocol header (nvars, degree=3) as scalars.
    tr.absorb_field_element(&BF::<R>::from(out_nvars as u64));
    tr.absorb_field_element(&BF::<R>::from(3u64));
    for _ in 0..out_nvars {
        for _ in 0..4 {
            tr.absorb(&R::ZERO);
        }
        let ri = tr.get_challenge();
        tr.absorb_field_element(&ri);
    }

    // setchk::absorb_evaluations(out.e, out.b):
    // Absorb all `out.e` blocks (each is a Vec<Vec<R>> of length `k_rg`, and each `Vec<R>` has
    // length `d`), then absorb `out.b` (len=1 ring element).
    for _ in 0..dcom_eval_vec_len {
        for _ in 0..out_e0_len {
            for _ in 0..d {
                tr.absorb(&R::ZERO);
            }
        }
    }
    for _ in 0..out_b_len {
        tr.absorb(&R::ZERO);
    }

    // rgchk::absorb_evaluations(dcom.evals):
    // - absorb eval.a as base-ring scalars
    // - absorb eval.c as ring elements
    for _ in 0..dcom_evals_len {
        for _ in 0..dcom_eval_vec_len {
            tr.absorb_field_element(&BF::<R>::ZERO);
        }
        for _ in 0..dcom_eval_vec_len {
            tr.absorb(&R::ZERO);
        }
    }

    // CM short challenges: s(3) + s_prime(k*d) => need_short squeezes of n=d bytes.
    let need_short = 3 + (params.k as usize) * d;
    for _ in 0..need_short {
        let _ = tr.squeeze_bytes(d);
    }

    // absorb_comh: L × κ ring elements.
    for _ in 0..(l_instances * kappa) {
        tr.absorb(&R::ZERO);
    }

    // c0/c1 = get_challenges(log_kappa) twice.
    let log_kappa = ark_std::log2((params.kappa as usize).next_power_of_two()) as usize;
    for _ in 0..(2 * log_kappa) {
        let _ = tr.get_challenge();
    }

    // Two CM sumchecks (degree=2) + eval table absorbs.
    let nvars_cm = params.nvars_cm as usize;
    for _ in 0..2 {
        let _ = tr.get_challenge(); // rc
        tr.absorb_field_element(&BF::<R>::from(nvars_cm as u64)); // nvars
        tr.absorb_field_element(&BF::<R>::from(2u64)); // degree=2
        for _ in 0..nvars_cm {
            for _ in 0..3 {
                tr.absorb(&R::ZERO);
            }
            let ri = tr.get_challenge();
            tr.absorb_field_element(&ri);
        }
        // CM eval tables: L instances, with (1+mlen_mats) rows each, each row has 4 ring elems.
        for _ in 0..l_instances {
            for _ in 0..(1 + mlen_mats) {
                for _ in 0..4 {
                    tr.absorb(&R::ZERO);
                }
            }
        }
    }

    Ok(tr.trace().clone())
}

#[cfg(feature = "we_gate")]
pub(crate) fn dummy_plus_proof_shape<R>(
    params: &WeParams,
    mlen_mats: usize,
    n_lin_proofs: usize,
) -> Result<crate::plus::PlusProof<R, crate::r1cs::ComR1CSProof<R>>, String>
where
    R: OverField + CoeffRing + PolyRing,
    R::BaseRing: Zq + Field + PrimeField,
{
    use latticefold::utils::sumcheck::prover::ProverMsg;
    use latticefold::utils::sumcheck::Proof;

    // NOTE: Current WE-gate binding assumes L=1 (single folded instance) for the exposed-prefix bind.
    if n_lin_proofs != 1 {
        return Err("dummy_plus_proof_shape: currently requires n_lin_proofs == 1 (prefix binding assumes L=1)".to_string());
    }

    let nvars_lin = params.nvars_setchk as usize;
    let nvars_cm = params.nvars_cm as usize;
    let kappa = params.kappa as usize;
    let d = R::dimension();
    let k_rg = params.k as usize;

    // Helper: degree-3 sumcheck proof with nvars rounds, each msg has 4 evals.
    let sc_deg3 = || -> Proof<R> {
        let msgs = (0..nvars_lin)
            .map(|_| ProverMsg::new(vec![R::ZERO, R::ZERO, R::ZERO, R::ZERO]))
            .collect::<Vec<_>>();
        Proof::new(msgs)
    };
    // Helper: degree-2 sumcheck proof with nvars rounds, each msg has 3 evals.
    let sc_deg2 = || -> Proof<R> {
        let msgs = (0..nvars_cm)
            .map(|_| ProverMsg::new(vec![R::ZERO, R::ZERO, R::ZERO]))
            .collect::<Vec<_>>();
        Proof::new(msgs)
    };

    // ComR1CSProof (Π_lin) skeleton.
    let lproof = (0..n_lin_proofs)
        .map(|_| crate::r1cs::ComR1CSProof::<R> {
            sumcheck_proof: sc_deg3(),
            nvars: nvars_lin,
            r: vec![R::ZERO; nvars_lin],
            v: R::ZERO,
            va: R::ZERO,
            vb: R::ZERO,
            vc: R::ZERO,
        })
        .collect::<Vec<_>>();

    // SetChk out: shapes
    let out_e: Vec<Vec<Vec<R>>> = (0..(1 + mlen_mats))
        .map(|_| {
            // For each group, we have L*k ring vectors of length d.
            (0..k_rg)
                .map(|_| vec![R::ZERO; d])
                .collect::<Vec<_>>()
        })
        .collect();
    let out_b: Vec<R> = vec![R::ZERO; 1];

    let out_sc = crate::setchk::Out::<R> {
        nvars: nvars_lin,
        r: vec![R::BaseRing::ZERO; nvars_lin],
        sumcheck_proof: sc_deg3(),
        e: out_e,
        b: out_b,
    };

    // Dcom evals:
    // - `v`: evaluation over `M_f`, represented as a base-ring vector of ring dimension `d`.
    // - `a`: evaluation over `tau`, represented over `1+mlen_mats` chunks.
    // - `b/c`: evaluations over `m_tau / f`, as ring vectors of length `1+mlen_mats`.
    let eval_len = 1 + mlen_mats;
    let dcom_evals = crate::rgchk::DcomEvals::<R> {
        v: vec![R::BaseRing::ZERO; d],
        a: vec![R::BaseRing::ZERO; eval_len],
        b: vec![R::ZERO; eval_len],
        c: vec![R::ZERO; eval_len],
    };

    let fcoms = crate::rgchk::FComs::<R> {
        cm_f: vec![R::ZERO; kappa],
        C_Mf: vec![R::ZERO; kappa],
        cm_mtau: vec![R::ZERO; kappa],
    };

    let dcom = crate::rgchk::Dcom::<R> {
        evals: vec![dcom_evals],
        fcoms: vec![fcoms],
        out: out_sc,
        dparams: crate::rgchk::DecompParameters { b: params.decomp_b as u128, k: k_rg, l: params.l as usize },
    };

    // CmProof eval tables: per instance, rows length 1+Mlen.
    let rows = (0..(1 + mlen_mats))
        .map(|_| [R::ZERO, R::ZERO, R::ZERO, R::ZERO])
        .collect::<Vec<_>>();
    let ieval = crate::cm::InstanceEvals::new(rows);

    let cmproof = crate::cm::CmProof::<R> {
        dcom,
        comh: vec![vec![R::ZERO; kappa]],
        sumcheck_proofs: (sc_deg2(), sc_deg2()),
        evals: (vec![ieval.clone()], vec![ieval]),
    };

    // DecompProof skeleton.
    let vo_len = 1 + mlen_mats;
    let v_pairs = vec![(R::ZERO, R::ZERO); vo_len];
    let dproof = crate::decomp::DecompProof::<R> {
        C: (vec![R::ZERO; kappa], vec![R::ZERO; kappa]),
        v: (v_pairs.clone(), v_pairs.clone()),
    };

    let linb2x = crate::mlin::LinB2X::<R> {
        cm_g: vec![R::ZERO; kappa],
        ro: Vec::new(),
        vo: v_pairs,
    };

    Ok(crate::plus::PlusProof::<R, crate::r1cs::ComR1CSProof<R>> {
        linb2x,
        lproof,
        cmproof,
        dproof,
    })
}

#[cfg(feature = "we_gate")]
fn tiny_extra_witness_from_plus_proof<R>(
    params: &WeParams,
    proof: &crate::plus::PlusProof<R, crate::r1cs::ComR1CSProof<R>>,
    mlen_mats: usize,
) -> Result<tiny::TinyExtraWitness, String>
where
    R: OverField + CoeffRing + PolyRing,
    R::BaseRing: Zq + Field + PrimeField,
{
    let ring_dim = R::dimension();
    if ring_dim != 64 {
        return Err("tiny witness: only ring_dim=64 supported".to_string());
    }
    let kappa = params.kappa as usize;
    let eval_len = 1usize + mlen_mats;

    #[inline]
    fn base_to_u64<BR: PrimeField>(x: BR) -> u64 {
        x.into_bigint().as_ref().get(0).copied().unwrap_or(0)
    }
    #[inline]
    fn ring_to_u64_coeffs<Rr: PolyRing>(r: &Rr) -> Vec<u64>
    where
        Rr::BaseRing: PrimeField,
    {
        r.coeffs().iter().copied().map(base_to_u64::<Rr::BaseRing>).collect()
    }

    let mut dcom_eval_b: Vec<Vec<Vec<u64>>> = Vec::new();
    let mut dcom_eval_v: Vec<Vec<u64>> = Vec::new();
    for ev in &proof.cmproof.dcom.evals {
        if ev.b.len() != eval_len {
            return Err("tiny witness: dcom.evals[*].b length mismatch".to_string());
        }
        if ev.v.len() != ring_dim {
            return Err("tiny witness: dcom.evals[*].v length mismatch".to_string());
        }
        dcom_eval_b.push(ev.b.iter().map(|r| ring_to_u64_coeffs::<R>(r)).collect());
        dcom_eval_v.push(ev.v.iter().copied().map(base_to_u64::<R::BaseRing>).collect());
    }

    if proof.dproof.C.0.len() != kappa || proof.dproof.C.1.len() != kappa {
        return Err("tiny witness: dproof.C length mismatch".to_string());
    }
    let vlen = 1usize + mlen_mats;
    if proof.dproof.v.0.len() != vlen || proof.dproof.v.1.len() != vlen {
        return Err("tiny witness: dproof.v length mismatch".to_string());
    }

    Ok(tiny::TinyExtraWitness {
        dcom_eval_b,
        dcom_eval_v,
        decomp_c0: proof.dproof.C.0.iter().map(|r| ring_to_u64_coeffs::<R>(r)).collect(),
        decomp_c1: proof.dproof.C.1.iter().map(|r| ring_to_u64_coeffs::<R>(r)).collect(),
        decomp_v0a: proof.dproof.v.0.iter().map(|(a, _)| ring_to_u64_coeffs::<R>(a)).collect(),
        decomp_v0b: proof.dproof.v.0.iter().map(|(_, b)| ring_to_u64_coeffs::<R>(b)).collect(),
        decomp_v1a: proof.dproof.v.1.iter().map(|(a, _)| ring_to_u64_coeffs::<R>(a)).collect(),
        decomp_v1b: proof.dproof.v.1.iter().map(|(_, b)| ring_to_u64_coeffs::<R>(b)).collect(),
    })
}

// ---------------------------------------------------------------------------
// Shape caching: load / save helpers
// ---------------------------------------------------------------------------

/// Load a cached WE gate shape from disk.
///
/// Reads `shape_meta.txt` (sidecar with `public_len`) and opens the file-backed instance.
fn load_we_plus_tiny_shape(
    shape_dir: impl AsRef<std::path::Path>,
) -> Result<WeDr1csShape<F257>, String> {
    let dir = shape_dir.as_ref();
    let meta_path = dir.join("shape_meta.txt");
    let meta = std::fs::read_to_string(&meta_path)
        .map_err(|e| format!("load shape: read shape_meta.txt failed: {e}"))?;
    let mut public_len: Option<usize> = None;
    for line in meta.lines() {
        if let Some(rest) = line.strip_prefix("public_len=") {
            public_len = rest.trim().parse::<usize>().ok();
        }
    }
    let public_len = public_len.ok_or("load shape: shape_meta.txt missing public_len")?;

    let merged_dir = dir.join("merged");
    let layout = symphony::file_backed_dr1cs::FileBackedLayout {
        dir: merged_dir.clone(),
        coeff_size: 0, // filled by open()
        idx_size: 0,
        row_size: 0,
        nconstraints: 0,
        a_terms: 0,
        b_terms: 0,
        c_terms: 0,
    };
    let inst = FileBackedSparseDr1csInstance::<F257>::open(layout)
        .map_err(|e| format!("load shape: open merged instance failed: {e}"))?;
    Ok(WeDr1csShape { inst, public_len })
}

/// Write shape sidecar metadata so `load_we_plus_tiny_shape` can reload it.
fn save_shape_meta(
    shape_dir: impl AsRef<std::path::Path>,
    public_len: usize,
) -> Result<(), String> {
    let path = shape_dir.as_ref().join("shape_meta.txt");
    std::fs::write(&path, format!("public_len={public_len}\n"))
        .map_err(|e| format!("save shape_meta.txt failed: {e}"))
}

/// Check whether a valid cached shape exists in `shape_dir`.
fn shape_cache_exists(shape_dir: impl AsRef<std::path::Path>) -> bool {
    let dir = shape_dir.as_ref();
    dir.join("shape_meta.txt").is_file() && dir.join("merged").join("meta.txt").is_file()
}

// ---------------------------------------------------------------------------
// Canonical tiny-field WE gate function (shape caching + assignment-only)
// ---------------------------------------------------------------------------

/// Canonical WE gate builder with shape caching.
///
/// **Cache miss** (first call or shape dir deleted):
///   Builds the shape (full builder Pass0+Pass1 with dummy proof) and writes it to `shape_dir`.
///   Then computes the assignment via the lightweight count-only pass (Pass0 only).
///
/// **Cache hit** (shape files exist in `shape_dir`):
///   Loads the shape from disk.
///   Computes the assignment via the lightweight count-only pass (Pass0 only, no disk writes).
///
/// This is the single canonical entry point.  Tests that delete `shape_dir` will get a cache
/// miss (shape is rebuilt); production keeps the dir across invocations.
#[cfg(feature = "we_gate")]
pub fn build_or_load_we_plus_tiny_dr1cs<R>(
    trace: &PoseidonTranscriptTrace<BF<R>>,
    params: &WeParams,
    stmt_digest: &[F257; 32], // statement public digest in F257
    binding_witness: &WeStatementBindingWitness,
    proof: &crate::plus::PlusProof<R, crate::r1cs::ComR1CSProof<R>>,
    mlen_mats: usize,
    pairs: &[(usize, usize)],
    shape_dir: impl AsRef<std::path::Path>,
) -> Result<(WeDr1csShape<F257>, Vec<F257>), String>
where
    R: OverField + CoeffRing + PolyRing,
    R::BaseRing: Zq + Field + PrimeField,
{
    let ring_dim = R::dimension();
    if ring_dim != 64 {
        return Err("build_or_load_we_plus_tiny_dr1cs: only ring_dim=64 supported".to_string());
    }
    let shape_dir = shape_dir.as_ref();

    // ---- Shape: load from cache or build ----------------------------------
    let shape = if shape_cache_exists(shape_dir) {
        eprintln!("[we_gate] shape cache hit: {}", shape_dir.display());
        load_we_plus_tiny_shape(shape_dir)?
    } else {
        eprintln!("[we_gate] shape cache miss — building: {}", shape_dir.display());
        std::fs::create_dir_all(shape_dir)
            .map_err(|e| format!("create shape_dir failed: {e}"))?;
        let n_lin_proofs = proof.lproof.len();
        let shape = build_we_dr1cs_for_plus_proof_shape_tiny::<R>(
            trace,
            params,
            // Shape is value-agnostic for the 32-field stmt_digest public lane.
            &[],
            n_lin_proofs,
            mlen_mats,
            pairs,
            shape_dir,
        )?;
        save_shape_meta(shape_dir, shape.public_len)?;
        shape
    };

    // ---- Assignment: always compute via count-only pass -------------------
    let assignment = build_we_plus_tiny_assignment_only::<R>(
        trace,
        params,
        stmt_digest,
        binding_witness,
        proof,
        mlen_mats,
        pairs,
    )?;

    // Validate layout compatibility.
    if assignment.len() != shape.inst.nvars {
        return Err(format!(
            "build_or_load_we_plus_tiny_dr1cs: assignment/shape nvars mismatch (asg={} shape={}). \
             Stale cache? Delete {} and retry.",
            assignment.len(),
            shape.inst.nvars,
            shape_dir.display(),
        ));
    }

    Ok((shape, assignment))
}

/// Build or load **only the WE gate shape** (no assignment).
///
/// This is the correct entrypoint for **WE arming**: the armer needs the statement-bound
/// constraint system, but must not depend on any prover/witness material.
#[cfg(feature = "we_gate")]
pub fn build_or_load_we_plus_tiny_shape<R>(
    trace: &PoseidonTranscriptTrace<BF<R>>,
    params: &WeParams,
    public_inputs_basefield: &[BF<R>],
    n_lin_proofs: usize,
    mlen_mats: usize,
    pairs: &[(usize, usize)],
    shape_dir: impl AsRef<std::path::Path>,
) -> Result<WeDr1csShape<F257>, String>
where
    R: OverField + CoeffRing + PolyRing,
    R::BaseRing: Zq + Field + PrimeField,
{
    let ring_dim = R::dimension();
    if ring_dim != 64 {
        return Err("build_or_load_we_plus_tiny_shape: only ring_dim=64 supported".to_string());
    }
    let shape_dir = shape_dir.as_ref();

    if shape_cache_exists(shape_dir) {
        eprintln!("[we_gate] shape cache hit: {}", shape_dir.display());
        return load_we_plus_tiny_shape(shape_dir);
    }

    eprintln!("[we_gate] shape cache miss — building: {}", shape_dir.display());
    std::fs::create_dir_all(shape_dir).map_err(|e| format!("create shape_dir failed: {e}"))?;

    let shape = build_we_dr1cs_for_plus_proof_shape_tiny::<R>(
        trace,
        params,
        public_inputs_basefield,
        n_lin_proofs,
        mlen_mats,
        pairs,
        shape_dir,
    )?;
    save_shape_meta(shape_dir, shape.public_len)?;
    Ok(shape)
}

/// Compute only the assignment vector (no shape, no disk writes).
///
/// Runs the count-only builder pass (Pass 0) for the inner tiny gate, then assembles
/// the outer merge: `[ONE] ++ params_prefix[1..] ++ tiny_gate_asg[1..]`.
#[cfg(feature = "we_gate")]
fn build_we_plus_tiny_assignment_only<R>(
    trace: &PoseidonTranscriptTrace<BF<R>>,
    params: &WeParams,
    stmt_digest: &[F257; 32],
    binding_witness: &WeStatementBindingWitness,
    proof: &crate::plus::PlusProof<R, crate::r1cs::ComR1CSProof<R>>,
    mlen_mats: usize,
    pairs: &[(usize, usize)],
) -> Result<Vec<F257>, String>
where
    R: OverField + CoeffRing + PolyRing,
    R::BaseRing: Zq + Field + PrimeField,
{
    let ring_dim = R::dimension();
    let extra = tiny_extra_witness_from_plus_proof::<R>(params, proof, mlen_mats)?;

    // Lift recorded ops and infer wiring for the verifier transcript tiny gate.
    let ops_f257 = tiny::lift_recording_trace_ops_to_f257::<BF<R>>(&trace.ops)?;
    let k = params.k as usize;
    let log_kappa = ark_std::log2((params.kappa as usize).next_power_of_two()) as usize;
    let nvars_cm = params.nvars_cm as usize;
    let squeeze_field_op_offset = first_squeeze_field_op_index_of_len(&ops_f257, ring_dim)?;
    let prefix_u32_squeeze_ops =
        collect_get_challenge_squeeze_field_indices(&ops_f257, 0, squeeze_field_op_offset);
    let wiring_rel = tiny::infer_cm_coin_op_wiring_from_ops(
        &ops_f257,
        ring_dim,
        k,
        log_kappa,
        nvars_cm,
        squeeze_field_op_offset,
    )?;
    let mut wiring_abs = tiny::TinyCoinOpWiring::default();
    wiring_abs.short_squeeze_ops = wiring_rel
        .short_squeeze_ops
        .into_iter()
        .map(|i| i + squeeze_field_op_offset)
        .collect();
    wiring_abs.u32_squeeze_ops = wiring_rel
        .u32_squeeze_ops
        .into_iter()
        .map(|i| i + squeeze_field_op_offset)
        .collect();
    wiring_abs
        .u32_squeeze_ops
        .splice(0..0, prefix_u32_squeeze_ops.into_iter());

    // ---- Inner tiny gate assignment (count-only, no disk writes) ----------
    let (asg_pose, pose_wiring) = tiny::we_tiny_f257_build_assignment_only(
        None,
        &ops_f257,
        ring_dim,
        params,
        &wiring_abs,
        pairs,
        &extra,
    )?;

    // ---- Params prefix assignment -----------------------------------------
    // Layout: [ONE, stmt_digest(32), 10×WeParams]
    let mut params_asg: Vec<F257> = Vec::with_capacity(1 + stmt_digest.len() + 10);
    params_asg.push(F257::ONE);
    params_asg.extend_from_slice(stmt_digest);
    params_asg.extend(params.to_field_vec::<F257>());

    // ---- Committed-values digest prefix glue assignment --------------------
    //
    // Must match the shape builder's `cv_prefix_glue` part:
    // - 64 prefix bytes (from `binding_witness.committed_values_prefix_bytes`)
    // - 8×8 absorbed bytes for the first 8 non-reabsorb Absorb(len=8) ranges in the transcript.
    const CV_WORDS: usize = 8;
    let coeff_bytes = ((<R::BaseRing as PrimeField>::MODULUS_BIT_SIZE as usize) + 7) / 8;
    if coeff_bytes != 8 {
        return Err(format!(
            "assignment_only: expected coeff_bytes=8 for base-field absorbs (got={})",
            coeff_bytes
        ));
    }
    let canon_absorb_ranges = collect_nonreabsorb_absorb_ranges(&ops_f257, &pose_wiring)?;
    let absorb_len8_ranges: Vec<(usize, usize)> = canon_absorb_ranges
        .into_iter()
        .filter(|&(_st, ln)| ln == coeff_bytes)
        .collect();
    let cv_start_idx: usize = 0;
    if absorb_len8_ranges.len() < cv_start_idx.saturating_add(CV_WORDS) {
        return Err(format!(
            "assignment_only: need {} absorbed public-input elements of {} bytes to bind committed_values_prefix_bytes (found={} start_idx={})",
            CV_WORDS,
            coeff_bytes,
            absorb_len8_ranges.len(),
            cv_start_idx,
        ));
    }
    let cv_absorb_ranges: Vec<(usize, usize)> =
        absorb_len8_ranges[cv_start_idx..cv_start_idx + CV_WORDS].to_vec();

    let mut glue_asg: Vec<F257> = Vec::with_capacity(1 + CV_PREFIX_BYTES + CV_WORDS * coeff_bytes);
    glue_asg.push(F257::ONE);
    for &b in &binding_witness.committed_values_prefix_bytes {
        glue_asg.push(F257::from(b as u64));
    }
    for i in 0..CV_WORDS {
        let (ab_start, ab_len) = cv_absorb_ranges[i];
        debug_assert_eq!(ab_len, coeff_bytes);
        for j in 0..ab_len {
            let v = pose_wiring.absorb_vars[ab_start + j];
            let val = if v == 0 { F257::ONE } else { asg_pose[v] };
            glue_asg.push(val);
        }
    }

    // ---- Independent statement-hash gadget assignments ---------------------
    // Must match the shape builder merge order: ids_hash then stmt_hash.
    let cfg = f257_poseidon_config();

    // ids_hash assignment
    let ops_ids = stmt_hash_ops_ids_only(
        binding_witness.vk_hash,
        binding_witness.r1cs_digest,
        binding_witness.gate_digest,
    );
    let (_inst_ids, asg_ids, wiring_ids) =
        symphony::dpp_poseidon::poseidon_sponge_dr1cs_from_ops_with_wiring_no_bytes::<F257>(
            &cfg, &ops_ids,
        )
        .map_err(|e| format!("assignment_only: ids_hash poseidon build failed: {e:?}"))?;
    let (ids_sf_start, ids_sf_len) = *wiring_ids
        .squeeze_field_ranges
        .last()
        .ok_or("assignment_only: ids_hash missing squeeze_field_ranges")?;
    if ids_sf_len != BIND_DIGEST_BYTES {
        return Err("assignment_only: ids_hash squeeze len mismatch".to_string());
    }
    let mut h_ids = [F257::ZERO; BIND_DIGEST_BYTES];
    for j in 0..BIND_DIGEST_BYTES {
        let v = wiring_ids.squeeze_field_vars[ids_sf_start + j];
        h_ids[j] = asg_ids[v];
    }

    // stmt_hash assignment (absorb computed h_ids)
    use symphony::transcript::PoseidonTraceOp;
    let params_f257 = params.to_field_vec::<F257>();
    let cvprefix_f257: Vec<F257> = binding_witness
        .committed_values_prefix_bytes
        .iter()
        .map(|&b| F257::from(b as u64))
        .collect();
    let ops_stmt: Vec<PoseidonTraceOp<F257>> = vec![
        PoseidonTraceOp::Absorb(vec![F257::from(2u64)]),
        PoseidonTraceOp::Absorb(h_ids.to_vec()),
        PoseidonTraceOp::Absorb(params_f257),
        PoseidonTraceOp::Absorb(cvprefix_f257),
        PoseidonTraceOp::SqueezeField(vec![F257::ZERO; BIND_DIGEST_BYTES]),
    ];
    let (_inst_stmt, asg_stmt, _wiring_stmt) =
        symphony::dpp_poseidon::poseidon_sponge_dr1cs_from_ops_with_wiring_no_bytes::<F257>(
            &cfg, &ops_stmt,
        )
        .map_err(|e| format!("assignment_only: stmt_hash poseidon build failed: {e:?}"))?;

    // ---- Merged assignment ------------------------------------------------
    // Same merge order as merge_file_backed_sparse_dr1cs_share_one:
    // [ONE] ++ params_prefix[1..] ++ asg_pose[1..] ++ cv_prefix_glue[1..] ++ ids_hash[1..] ++ stmt_hash[1..]
    let mut merged: Vec<F257> = Vec::with_capacity(
        1 + (params_asg.len() - 1)
            + (asg_pose.len() - 1)
            + (glue_asg.len() - 1)
            + (asg_ids.len() - 1)
            + (asg_stmt.len() - 1),
    );
    merged.push(F257::ONE);
    merged.extend_from_slice(&params_asg[1..]);
    merged.extend_from_slice(&asg_pose[1..]);
    merged.extend_from_slice(&glue_asg[1..]);
    merged.extend_from_slice(&asg_ids[1..]);
    merged.extend_from_slice(&asg_stmt[1..]);

    Ok(merged)
}

#[cfg(all(test, feature = "we_gate"))]
#[allow(non_local_definitions)]
mod tests {
    #[cfg(feature = "parallel")]
    fn init_rayon_stack() {
        // Same mitigation as the SP1 oneproof harness:
        // large-stack computations can end up on Rayon worker threads (smaller default stacks),
        // causing intermittent stack overflows. Configure the *global* Rayon pool once.
        //
        // Override with `RAYON_STACK_SIZE_BYTES` (bytes). Default: 64 MiB.
        let stack_bytes: usize = std::env::var("RAYON_STACK_SIZE_BYTES")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(64 * 1024 * 1024);

        let mut builder = rayon::ThreadPoolBuilder::new().stack_size(stack_bytes);

        // Respect RAYON_NUM_THREADS if provided.
        if let Some(n) = std::env::var("RAYON_NUM_THREADS")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
        {
            builder = builder.num_threads(n);
        }

        // If Rayon was already initialized elsewhere, ignore and proceed.
        drop(builder.build_global());
    }

    #[cfg(not(feature = "parallel"))]
    fn init_rayon_stack() {}

    use super::*;
    use latticefold::arith::r1cs::R1CS;
    use latticefold::transcript::Transcript;
    use ark_ff::BigInteger;
    use cyclotomic_rings::rings::GoldilocksRing64 as R;
    use stark_rings_linalg::SparseMatrix;

    use crate::lin::LinearizedVerify;
    use crate::recording_transcript::TracePoseidonTranscript;
    use crate::rgchk::DecompParameters;

    // NOTE: We intentionally do not keep the old “shape builds and constraints check” tests here.
    // They were development scaffolding and are slow/ignored. The tiny gate is now exercised via
    // focused gadget-level tests in `we_gate_tiny/tests.rs`, and will be covered end-to-end by
    // real-trace tests in the main WE harness.

    #[test]
    #[ignore = "slow: builds Poseidon(F257) tiny gate + checks all constraints (GoldilocksRing64)"]
    fn test_tiny_gate_shape_builds_and_constraints_check_goldilocks() {
        use cyclotomic_rings::rings::GoldilocksRing64 as RR;

        init_rayon_stack();

        // Minimal-but-valid params to keep the schedule small but exercise the d64 path.
        //
        // IMPORTANT: CM verifier math (t(z) tensor evaluation) requires `nvars_cm` to be at least
        // the number of tensor variables: log2(d) + log2(ell) + log2(k*d) + log2(kappa).
        // For d=64, kappa=1, k=1, ell=1 this is 6 + 0 + 6 + 0 = 12.
        let ring_dim = <RR as PolyRing>::dimension() as u64;
        let nvars = 12u64;
        let params = WeParams {
            nvars_setchk: nvars,
            degree_setchk: 3,
            nvars_cm: nvars,
            degree_cm: 2,
            kappa: 1,
            ring_dim_d: ring_dim,
            decomp_b: 16,
            k: 1,
            l: 1,
            mlen: 0,
        };

        // Exercise one digit-mul surface.
        let pairs: Vec<(usize, usize)> = vec![(0, 0)];

        // Build a self-consistent trace and arithmetize the tiny gate.
        // Prefix binding requires at least `min(8,kappa)` public inputs absorbed before challenges.
        // Here kappa=1, so include exactly 1 dummy public input absorb in the trace.
        let trace = super::poseidon_trace_schedule_for_plus::<RR>(1, &params, 1, 0)
            .expect("poseidon_trace_schedule_for_plus");
        let ops_f257 = tiny::lift_recording_trace_ops_to_f257::<BF<RR>>(&trace.ops)
            .expect("lift_recording_trace_ops_to_f257");

        // The CM segment begins at the first `SqueezeField(len=ring_dim)` (short challenges).
        // To mirror the real verifier schedule, include prefix `get_challenge()` u32 squeezes.
        let squeeze_field_op_offset =
            super::first_squeeze_field_op_index_of_len(&ops_f257, <RR as PolyRing>::dimension())
                .expect("first short SqueezeField(len=ring_dim) exists");
        let prefix_u32_squeeze_ops =
            super::collect_get_challenge_squeeze_field_indices(&ops_f257, 0, squeeze_field_op_offset);

        let k = params.k as usize;
        let log_kappa = ark_std::log2((params.kappa as usize).next_power_of_two()) as usize;
        let nvars_cm = params.nvars_cm as usize;
        let wiring_rel = tiny::infer_cm_coin_op_wiring_from_ops(
            &ops_f257,
            <RR as PolyRing>::dimension(),
            k,
            log_kappa,
            nvars_cm,
            squeeze_field_op_offset,
        )
        .expect("infer_cm_coin_op_wiring_from_ops");

        let mut wiring_abs = tiny::TinyCoinOpWiring::default();
        wiring_abs.short_squeeze_ops = wiring_rel
            .short_squeeze_ops
            .into_iter()
            .map(|i| i + squeeze_field_op_offset)
            .collect();
        wiring_abs.u32_squeeze_ops = wiring_rel
            .u32_squeeze_ops
            .into_iter()
            .map(|i| i + squeeze_field_op_offset)
            .collect();
        wiring_abs
            .u32_squeeze_ops
            .splice(0..0, prefix_u32_squeeze_ops.into_iter());

        let out_dir = {
            let mut p = std::env::temp_dir();
            p.push("lfplus_test_tiny_gate_build_from_ops");
            let _ = std::fs::remove_dir_all(&p);
            std::fs::create_dir_all(&p).expect("create temp out_dir");
            p
        };
        let dummy_extra = tiny::TinyExtraWitness {
            dcom_eval_b: vec![vec![vec![0u64; <RR as PolyRing>::dimension()]]],
            dcom_eval_v: vec![vec![0u64; <RR as PolyRing>::dimension()]],
            decomp_c0: vec![vec![0u64; <RR as PolyRing>::dimension()]; params.kappa as usize],
            decomp_c1: vec![vec![0u64; <RR as PolyRing>::dimension()]; params.kappa as usize],
            decomp_v0a: vec![vec![0u64; <RR as PolyRing>::dimension()]; 1 + params.mlen as usize],
            decomp_v0b: vec![vec![0u64; <RR as PolyRing>::dimension()]; 1 + params.mlen as usize],
            decomp_v1a: vec![vec![0u64; <RR as PolyRing>::dimension()]; 1 + params.mlen as usize],
            decomp_v1b: vec![vec![0u64; <RR as PolyRing>::dimension()]; 1 + params.mlen as usize],
        };
        let (inst, asg, _shorts, _u32s, _goldilocks, _sm, _ssq, _w) =
            tiny::we_tiny_f257_build_cm_gate_from_trace_ops(
                None,
                &ops_f257,
                <RR as PolyRing>::dimension(),
                &params,
                &wiring_abs,
                &pairs,
                &dummy_extra,
                &out_dir,
            )
            .expect("we_tiny_f257_build_cm_gate_from_trace_ops");

        inst.check(&asg).expect("dr1cs check");
        let _ = std::fs::remove_dir_all(&out_dir);
    }


    #[test]
    #[ignore = "very slow in debug: runs full DPP prove+decap; run with `--release`"]
    fn test_tiny_gate_ringlwe_lock_roundtrip_small() {
        use crate::lockable_ringlwe::RingLweParams;
        use crate::we_statement::encode_public_x;
        use crate::we_tiny_lock::arm_lfplus_ringlwe_lock;
        use crate::utils::maybe_print_rss;
        use std::time::Instant;
        use rand::{rngs::StdRng, RngCore, SeedableRng};

        // Minimal-but-valid params to keep the schedule small.
        //
        // IMPORTANT: for ring_dim=64 the CM verifier math needs nvars_cm >= 12 (see also
        // `test_tiny_gate_shape_builds_and_constraints_check_goldilocks`).
        let ring_dim = <R as PolyRing>::dimension() as u64;
        let nvars_min = 12u64;
        let params = WeParams {
            nvars_setchk: nvars_min,
            degree_setchk: 3,
            nvars_cm: nvars_min,
            degree_cm: 2,
            kappa: 1,
            ring_dim_d: ring_dim,
            decomp_b: 16,
            k: 1,
            l: 1,
            mlen: 0,
        };
        let public_inputs_len = 8usize; // number of public **field elements**
        let n_lin_proofs = 1usize; // schedule builder currently assumes L=1
        let mlen_mats = 0usize;
        // Mirror the SP1 oneproof path: use the canonical tiny-gate builders.
        // Include an SP1-style exposed-prefix public input vector (len=8) so we exercise
        // prefix binding at arming/proving time.
        //
        // NOTE: `pairs` indices are interpreted over the full `u32_squeeze_ops` wiring, which
        // includes any prefix `get_challenge()` u32 squeezes.
        let pairs: Vec<(usize, usize)> = vec![(0, 0)];
        type BF0 = <<R as PolyRing>::BaseRing as ark_ff::Field>::BasePrimeField;
        // Public inputs as base-field elements (used for schedule trace generation).
        //
        // the tiny-gate public prefix is the transcript encoding of these field
        // elements: fixed-width little-endian bytes, lifted into F257 (0..=255). We must NOT
        // synthesize "bytes" from arbitrary F257 elements, since 256 is a valid F257 element but
        // is a special base-257 digit (sentinel) in the transcript's byte-view map.
        let mut public_inputs_bf: Vec<BF0> = (0..public_inputs_len)
            .map(|i| if (i % 3) == 0 { BF0::ONE } else { BF0::ZERO })
            .collect();
        // for kappa=1, the tiny gate's prefix binding constrains `cm_f[0]` to match the
        // first public input absorbed in the transcript prefix. Our dummy proof uses `cm_f[0]=0`,
        // so ensure the first public input is 0 to keep the roundtrip test focused on shape↔witness
        // consistency (not on statement/proof mismatch).
        if !public_inputs_bf.is_empty() {
            public_inputs_bf[0] = BF0::ZERO;
        }
        // Tiny-gate statement public prefix uses the byte encoding (8 bytes per base-field element).
        let _public_inputs_bytes_f257: Vec<F257> = public_inputs_bf
            .iter()
            .flat_map(|x| {
                let bytes = latticefold::transcript::bytes::prime_field_to_bytes_le_fixed::<BF0>(x);
                bytes.into_iter().map(|b| F257::from(b as u64))
            })
            .collect();

        // Build a self-consistent schedule trace whose initial absorbs match `public_inputs_f257`.
        let trace = super::poseidon_trace_schedule_for_plus_with_public_inputs::<R>(
            &public_inputs_bf,
            &params,
            n_lin_proofs,
            mlen_mats,
        )
        .expect("poseidon_trace_schedule_for_plus_with_public_inputs");

        // Canonical path: build/load shape + compute assignment in one call.
        let t_shape = Instant::now();
        let out_dir = {
            let mut p = std::env::temp_dir();
            p.push("lfplus_test_tiny_shape_roundtrip");
            let _ = std::fs::remove_dir_all(&p);
            std::fs::create_dir_all(&p).expect("create temp out_dir");
            p
        };
        let proof =
            dummy_plus_proof_shape::<R>(&params, mlen_mats, n_lin_proofs).expect("dummy_plus_proof_shape");
        let vk_hash = [1u8; 32];
        let r1cs_digest = [2u8; 32];
        let gate_digest = [3u8; 32];
        // In the SP1 path, the first 8 public inputs are the committed-values digest lane.
        // In this unit test, bind to the *actual* transcript-absorbed bytes for the first 8
        // public inputs so the gate glue is satisfiable.
        let committed_values_prefix_bytes: [u8; CV_PREFIX_BYTES] = {
            let mut out = [0u8; CV_PREFIX_BYTES];
            let pis = public_inputs_bf
                .get(0..8)
                .expect("test requires at least 8 public inputs for committed-values prefix");
            for i in 0..8usize {
                let bytes =
                    latticefold::transcript::bytes::prime_field_to_bytes_le_fixed::<BF0>(&pis[i]);
                out[8 * i..8 * i + 8].copy_from_slice(&bytes);
            }
            out
        };
        let committed_values_prefix_f257: [F257; CV_PREFIX_BYTES] =
            committed_values_prefix_bytes.map(|b| F257::from(b as u64));
        let stmt_digest_f257 = crate::we_statement::we_statement_hash_lf_plus::<R>(
            vk_hash,
            committed_values_prefix_f257,
            r1cs_digest,
            gate_digest,
            &params,
        );
        let binding = WeStatementBindingWitness {
            vk_hash,
            r1cs_digest,
            gate_digest,
            committed_values_prefix_bytes,
        };
        let (shape, asg) = build_or_load_we_plus_tiny_dr1cs::<R>(
            &trace,
            &params,
            &stmt_digest_f257,
            &binding,
            &proof,
            mlen_mats,
            &pairs,
            &out_dir,
        )
        .expect("build_or_load_we_plus_tiny_dr1cs");
        assert_eq!(shape.public_len, 1 + 32);
        assert_eq!(asg.len(), shape.inst.nvars);
        shape.inst.check(&asg).expect("shape should be satisfied by witness assignment");
        eprintln!(
            "[tiny_gate] built shape in {:?}: public_len={} nvars={} constraints={}",
            t_shape.elapsed(),
            shape.public_len,
            shape.inst.nvars,
            shape.inst.layout.nconstraints
        );

        // Arm and then prove+decap using the satisfying assignment split into (x || z_w).
        let armer_seed = [7u8; 32];
        let lock_j = 0u64;

        let ringlwe_params = RingLweParams::default();
        let dummy_payload: [u8; 0] = [];

        let mut rng = StdRng::seed_from_u64(42);
        let public_len = shape.public_len;
        let prover = crate::we_tiny_lock::we_ringlwe_prover_from_dr1cs::<F257>(
            shape.inst.clone(),
            shape.public_len,
        )
        .expect("we_ringlwe_prover_from_dr1cs");
        // Scaling harness:
        // - build shape once (done above)
        // - iterate hits_per_block values and measure arm/prove+decap
        let hits_list: Vec<u16> = std::env::var("LFP_TEST_HITS_LIST")
            .ok()
            .map(|s| {
                s.split(',')
                    .filter_map(|t| t.trim().parse::<u16>().ok())
                    .filter(|&v| v > 0)
                    .collect::<Vec<_>>()
            })
            .filter(|v| !v.is_empty())
            .unwrap_or_else(|| vec![1, 2, 4, 8, 16, 32]);
        let avail = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(0);
        let rayon_threads = rayon::current_num_threads();
        eprintln!(
            "[tiny_gate] scale: hits_list={:?} available_parallelism={} rayon_threads={}",
            hits_list, avail, rayon_threads
        );

        // Keep deterministic behavior by deriving per-lock RNG seeds sequentially first.
        let mut seed0 = [0u8; 32];
        let mut seed1 = [0u8; 32];
        rng.fill_bytes(&mut seed0);
        rng.fill_bytes(&mut seed1);
        let policy0 = crate::we_tiny_lock::WeRingLweLockArmingPolicy {
            base_rep_id: 0,
            max_rep_tries: 32,
            hint_budget_bytes: None,
        };
        let policy1 = crate::we_tiny_lock::WeRingLweLockArmingPolicy {
            base_rep_id: 1,
            max_rep_tries: 32,
            hint_budget_bytes: None,
        };

        for hits_per_block in hits_list {
        let t_arm = Instant::now();
            maybe_print_rss(&format!("tiny_gate:h{hits_per_block}:before_arm"));
            // Arm two lock contexts in parallel, matching oneproof's per-lock parallelism model.
            let (lock0, lock1) = rayon::join(
                || {
                    let mut rng0 = StdRng::from_seed(seed0);
                    arm_lfplus_ringlwe_lock::<R>(
                shape.clone(),
                &stmt_digest_f257,
                armer_seed,
                lock_j,
                0,
                        policy0,
                ringlwe_params.clone(),
                        hits_per_block,
                &dummy_payload,
                        &mut rng0,
                    )
                    .unwrap_or_else(|e| panic!("arm ctx0 failed: {e}"))
                    .lock
                },
                || {
                    let mut rng1 = StdRng::from_seed(seed1);
                    arm_lfplus_ringlwe_lock::<R>(
                        shape.clone(),
                &stmt_digest_f257,
                armer_seed,
                lock_j,
                0,
                        policy1,
                ringlwe_params.clone(),
                        hits_per_block,
                &dummy_payload,
                        &mut rng1,
                    )
                    .unwrap_or_else(|e| panic!("arm ctx1 failed: {e}"))
                    .lock
                },
            );
            maybe_print_rss(&format!("tiny_gate:h{hits_per_block}:after_arm"));
        eprintln!(
                "[tiny_gate] h={} armed in {:?}: sublocks(lock0)={} sublocks(lock1)={} proof_len={}",
                hits_per_block,
            t_arm.elapsed(),
                lock0.sublocks.len(),
                lock1.sublocks.len(),
            prover.proof_len()
        );

        let x = encode_public_x::<F257>(&stmt_digest_f257);
        assert_eq!(x.len(), public_len);
            assert_eq!(
                &asg[..public_len],
                x.as_slice(),
                "satisfying assignment public prefix mismatch"
            );
        let z_w = asg[public_len..].to_vec();

        let t_prove = Instant::now();
            maybe_print_rss(&format!("tiny_gate:h{hits_per_block}:before_prove_decap_stream"));
        // Flatten sublocks across both locks into one streaming pass.
            let mut coins_list: Vec<_> =
                Vec::with_capacity(lock0.sublocks.len() + lock1.sublocks.len());
            let mut meta: Vec<(usize, usize)> =
                Vec::with_capacity(lock0.sublocks.len() + lock1.sublocks.len()); // (lock_id, sublock_idx)
            for (si, sl) in lock0.sublocks.iter().enumerate() {
                coins_list.push(
                    prover
                        .derive_public_coins_from_stmt(
                            lock0.c_stmt.as_slice(),
                            sl.block_id as usize,
                            sl.rep_id,
                        )
                        .expect("derive_public_coins_from_stmt lock0"),
                );
            meta.push((0, si));
            }
            for (si, sl) in lock1.sublocks.iter().enumerate() {
                coins_list.push(
                    prover
                        .derive_public_coins_from_stmt(
                            lock1.c_stmt.as_slice(),
                            sl.block_id as usize,
                            sl.rep_id,
                        )
                        .expect("derive_public_coins_from_stmt lock1"),
                );
            meta.push((1, si));
            }
            // Canonical path: stream tails and fold them immediately (no tail vectors).
            let mut tail_dots_mod257: Vec<u16> = vec![0u16; coins_list.len()];
            let mut cur_ci: Option<usize> = None;
            let mut cur_blk: usize = 0;
            let mut buf_len: usize = 0;
            let mut buf64: [u16; 64] = [0u16; 64];
            let abg_list = prover
            .stream_pi0_and_collect_tails(
                &x,
                &z_w,
                &coins_list,
                    &mut |_chunk| {},
                    &mut |ci, _ti, t| {
                        if cur_ci != Some(ci) {
                            cur_ci = Some(ci);
                            cur_blk = 0;
                            buf_len = 0;
                        }
                        let td = crate::lockable_ringlwe::field_mod257_u16(t);
                        buf64[buf_len] = td;
                        buf_len += 1;
                        if buf_len == 64 {
                            let (lock_id, si) = meta[ci];
                            let sl = match lock_id {
                                0 => &lock0.sublocks[si],
                                1 => &lock1.sublocks[si],
                                _ => unreachable!(),
                            };
                            let blk = &sl.hints.tail_scales[cur_blk];
                            let add =
                                crate::lockable_ringlwe::dot_packed_block_mod257_u16(blk, &buf64);
                            let acc = &mut tail_dots_mod257[ci];
                            *acc = crate::lockable_ringlwe::add_mod257_u16(*acc, add);
                            cur_blk += 1;
                            buf_len = 0;
                    }
                },
            )
            .expect("stream_pi0_and_collect_tails");
            assert_eq!(abg_list.len(), meta.len());
            assert_eq!(buf_len, 0);
            use rayon::prelude::*;
        let mut cands0: Vec<Option<(u16, [u16; 2])>> = vec![None; lock0.sublocks.len()];
        let mut cands1: Vec<Option<(u16, [u16; 2])>> = vec![None; lock1.sublocks.len()];
            let results: Vec<(usize, usize, u16, [u16; 2])> = abg_list
                .par_iter()
                .enumerate()
                .map(|(gi, abgt)| {
            let (lock_id, si) = meta[gi];
                    let sl = match lock_id {
                        0 => &lock0.sublocks[si],
                        1 => &lock1.sublocks[si],
                        _ => unreachable!(),
                    };
                    let td = tail_dots_mod257[gi];
                    let cands =
                        crate::lockable_ringlwe::sublock_s_candidates_from_abg_tail(sl, abgt, td)
                            .expect("sublock candidates");
                    (lock_id, si, sl.channel_id, cands)
                })
                .collect();
            for (lock_id, si, ch, cands) in results {
            match lock_id {
                0 => cands0[si] = Some((ch, cands)),
                1 => cands1[si] = Some((ch, cands)),
                _ => unreachable!(),
            }
        }
            let sublock_cands0: Vec<(u16, [u16; 2])> =
                cands0.into_iter().map(|x| x.unwrap()).collect();
            let sublock_cands1: Vec<(u16, [u16; 2])> =
                cands1.into_iter().map(|x| x.unwrap()).collect();
            let pt0 =
                crate::lockable_ringlwe::decrypt_payload_from_sublock_s_candidates(&lock0, &sublock_cands0)
            .expect("decrypt_payload lock0");
            let pt1 =
                crate::lockable_ringlwe::decrypt_payload_from_sublock_s_candidates(&lock1, &sublock_cands1)
            .expect("decrypt_payload lock1");
        assert!(pt0.is_empty());
        assert!(pt1.is_empty());

        // Accepting set structure: fixed `{1,2}` for every sublock.
        for lock in [&lock0, &lock1] {
            for sl in &lock.sublocks {
                assert_eq!(sl.accepting_set[0], F257::ONE);
                assert_eq!(sl.accepting_set[1], F257::from(2u64));
            }
        }

            maybe_print_rss(&format!("tiny_gate:h{hits_per_block}:after_prove_decap_stream"));
            eprintln!(
                "[tiny_gate] h={} prove+decap(stream) in {:?}",
                hits_per_block,
                t_prove.elapsed()
            );
        }

        // Now it is safe to reclaim disk space used by the shape files.
        crate::fs_cleanup::fast_remove_dir_best_effort(&out_dir);
    }

    #[test]
    #[ignore = "very slow in debug: runs full DPP prove+decap; run with `--release`"]
    fn test_tiny_gate_ringlwe_payload_shamir_2of2_small() {
        use crate::lockable_ringlwe::RingLweParams;
        use crate::shamir_gf256::{reconstruct_secret_32, split_secret_32, ShamirConfig, ShamirShare};
        use crate::we_statement::encode_public_x;
        use crate::we_tiny_lock::arm_lfplus_ringlwe_lock;
        use crate::utils::maybe_print_rss;
        use rand::{rngs::StdRng, RngCore, SeedableRng};
        use std::time::Instant;

        // Same minimal params as the small roundtrip test.
        let ring_dim = <R as PolyRing>::dimension() as u64;
        let nvars_min = 12u64;
        let params = WeParams {
            nvars_setchk: nvars_min,
            degree_setchk: 3,
            nvars_cm: nvars_min,
            degree_cm: 2,
            kappa: 1,
            ring_dim_d: ring_dim,
            decomp_b: 16,
            k: 1,
            l: 1,
            mlen: 0,
        };
        let public_inputs_len = 8usize;
        let n_lin_proofs = 1usize;
        let mlen_mats = 0usize;
        let pairs: Vec<(usize, usize)> = vec![(0, 0)];
        type BF0 = <<R as PolyRing>::BaseRing as ark_ff::Field>::BasePrimeField;
        let mut public_inputs_bf: Vec<BF0> = (0..public_inputs_len)
            .map(|i| if (i % 3) == 0 { BF0::ONE } else { BF0::ZERO })
            .collect();
        if !public_inputs_bf.is_empty() {
            public_inputs_bf[0] = BF0::ZERO;
        }
        let _public_inputs_bytes_f257: Vec<F257> = public_inputs_bf
            .iter()
            .flat_map(|x| {
                let bytes = latticefold::transcript::bytes::prime_field_to_bytes_le_fixed::<BF0>(x);
                bytes.into_iter().map(|b| F257::from(b as u64))
            })
            .collect();

        let trace = super::poseidon_trace_schedule_for_plus_with_public_inputs::<R>(
            &public_inputs_bf,
            &params,
            n_lin_proofs,
            mlen_mats,
        )
        .expect("poseidon_trace_schedule_for_plus_with_public_inputs");

        // Shape + satisfying assignment (canonical cache-aware path).
        let keep_tiny_cache = std::env::var("LFP_KEEP_TINY_GATE_CACHE")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false);
        let out_dir = {
            let mut p = std::env::temp_dir();
            p.push("lfplus_test_tiny_payload_shamir_2of2");
            if !keep_tiny_cache {
                let _ = std::fs::remove_dir_all(&p);
            }
            std::fs::create_dir_all(&p).expect("create temp out_dir");
            p
        };
        let proof =
            dummy_plus_proof_shape::<R>(&params, mlen_mats, n_lin_proofs).expect("dummy_plus_proof_shape");
        let vk_hash = [1u8; 32];
        let r1cs_digest = [2u8; 32];
        let gate_digest = [3u8; 32];
        let committed_values_prefix_bytes: [u8; CV_PREFIX_BYTES] = {
            let mut out = [0u8; CV_PREFIX_BYTES];
            let pis = public_inputs_bf
                .get(0..8)
                .expect("test requires at least 8 public inputs for committed-values prefix");
            for i in 0..8usize {
                let bytes =
                    latticefold::transcript::bytes::prime_field_to_bytes_le_fixed::<BF0>(&pis[i]);
                out[8 * i..8 * i + 8].copy_from_slice(&bytes);
            }
            out
        };
        let committed_values_prefix_f257: [F257; CV_PREFIX_BYTES] =
            committed_values_prefix_bytes.map(|b| F257::from(b as u64));
        let stmt_digest_f257 = crate::we_statement::we_statement_hash_lf_plus::<R>(
            vk_hash,
            committed_values_prefix_f257,
            r1cs_digest,
            gate_digest,
            &params,
        );
        let binding = WeStatementBindingWitness {
            vk_hash,
            r1cs_digest,
            gate_digest,
            committed_values_prefix_bytes,
        };
        let (shape, asg) = build_or_load_we_plus_tiny_dr1cs::<R>(
            &trace,
            &params,
            &stmt_digest_f257,
            &binding,
            &proof,
            mlen_mats,
            &pairs,
            &out_dir,
        )
        .expect("build_or_load_we_plus_tiny_dr1cs");
        shape.inst.check(&asg).expect("shape should be satisfied by witness assignment");

        let public_len = shape.public_len;
        let x = encode_public_x::<F257>(&stmt_digest_f257);
        assert_eq!(x.len(), public_len);
        assert_eq!(&asg[..public_len], x.as_slice());
        let z_w = asg[public_len..].to_vec();

        // Shamir secret (T-of-R).
        //
        // This is the *intended* usage in the design docs: decapsulation under noise is
        // probabilistic (AEAD may fail). We publish R independent share-locks and only need
        // T successful decryptions (treat failures as erasures).
        // For the 1-hint RingLWE lock variant, we want to understand the *scaling* of
        // arming/decap and artifact size as we increase the number of locks per armer.
        //
        // Keep T small for test runtime, but set R=16 so we exercise the "16 locks per armer"
        // storage/perf profile end-to-end.
        let shamir = ShamirConfig {
            threshold: 3,
            shares: 16,
        };
        let mut rng = StdRng::seed_from_u64(20260205);
        let mut secret = [0u8; 32];
        rng.fill_bytes(&mut secret);
        let shares = split_secret_32(&mut rng, &shamir, secret).expect("split_secret_32");
        assert_eq!(shares.len(), shamir.shares);

        let prover = crate::we_tiny_lock::we_ringlwe_prover_from_dr1cs::<F257>(
            shape.inst.clone(),
            shape.public_len,
        )
        .expect("we_ringlwe_prover_from_dr1cs");

        // MLWE/RLWE hint locks (one per Shamir share).
        // Disable noise for deterministic AEAD key agreement in tests.
        let ringlwe_params = RingLweParams {
            ..RingLweParams::default()
        };
        let lock_j = 0u64;

        let t_arm = Instant::now();
        let mut locks = Vec::with_capacity(shamir.shares);
        for i in 0..shamir.shares {
            let mut rep_id = i as u64;
            let lock = loop {
                match arm_lfplus_ringlwe_lock::<R>(
                    shape.clone(),
                    &stmt_digest_f257,
                    [11u8.wrapping_add(i as u8); 32],
                    lock_j,
                    0,
                    crate::we_tiny_lock::WeRingLweLockArmingPolicy {
                        base_rep_id: rep_id,
                        max_rep_tries: 32,
                        hint_budget_bytes: None,
                    },
                    ringlwe_params.clone(),
                    1,
                    &shares[i].value,
                    &mut rng,
                ) {
                    Ok(lock_out) => break lock_out.lock,
                    Err(e) if e.contains("resample rep_id") => {
                        rep_id += 1;
                        continue;
                    }
                    Err(e) => panic!("arm lock[{i}] failed: {e}"),
                }
            };
            locks.push(lock);
        }
        eprintln!(
            "[tiny_payload_shamir] armed {} share locks (T={} of R={}) in {:?}: proof_len={}",
            shamir.shares,
            shamir.threshold,
            shamir.shares,
            t_arm.elapsed(),
            prover.proof_len()
        );

        // Prove once (π0 streamed), decap R times (tails differ).
        let t_prove = Instant::now();
        maybe_print_rss("tiny_payload_shamir:before_prove_decap_stream");
        // Flatten all sublocks across all locks into one streaming pass.
        let mut coins_list: Vec<_> = Vec::new();
        let mut meta: Vec<(usize, usize)> = Vec::new(); // (lock_i, sublock_i)
        for (li, l) in locks.iter().enumerate() {
            for (si, sl) in l.sublocks.iter().enumerate() {
                coins_list.push(
                    prover
                        .derive_public_coins_from_stmt(l.c_stmt.as_slice(), sl.block_id as usize, sl.rep_id)
                        .expect("derive_public_coins_from_stmt"),
                );
                meta.push((li, si));
            }
        }
        let mut tail_dots_mod257: Vec<u16> = vec![0u16; coins_list.len()];
        let mut cur_ci: Option<usize> = None;
        let mut cur_blk: usize = 0;
        let mut buf_len: usize = 0;
        let mut buf64: [u16; 64] = [0u16; 64];
        let abg_list = prover
            .stream_pi0_and_collect_tails(
                &x,
                &z_w,
                &coins_list,
                &mut |_chunk| {},
                &mut |ci, _ti, t| {
                    if cur_ci != Some(ci) {
                        cur_ci = Some(ci);
                        cur_blk = 0;
                        buf_len = 0;
                    }
                    let td = crate::lockable_ringlwe::field_mod257_u16(t);
                    buf64[buf_len] = td;
                    buf_len += 1;
                    if buf_len == 64 {
                        let (li, si) = meta[ci];
                        let sl = &locks[li].sublocks[si];
                        let blk = &sl.hints.tail_scales[cur_blk];
                        let add = crate::lockable_ringlwe::dot_packed_block_mod257_u16(blk, &buf64);
                        let acc = &mut tail_dots_mod257[ci];
                        *acc = crate::lockable_ringlwe::add_mod257_u16(*acc, add);
                        cur_blk += 1;
                        buf_len = 0;
                    }
                },
            )
            .expect("stream_pi0_and_collect_tails");
        assert_eq!(abg_list.len(), meta.len());
        assert_eq!(buf_len, 0);

        // Collect per-sublock candidates per lock, then decrypt deterministically (no branching).
        let mut per_lock_slots: Vec<Vec<Option<(u16, [u16; 2])>>> =
            locks.iter().map(|l| vec![None; l.sublocks.len()]).collect();
        for (gi, abgt) in abg_list.iter().enumerate() {
            let (li, si) = meta[gi];
            let sl = &locks[li].sublocks[si];
            let td = tail_dots_mod257[gi];
            let cands = crate::lockable_ringlwe::sublock_s_candidates_from_abg_tail(sl, abgt, td)
            .unwrap_or_else(|e| panic!("sublock cands[{gi}]: {e}"));
            let ch = sl.channel_id;
            per_lock_slots[li][si] = Some((ch, cands));
        }
        let mut selected: Vec<ShamirShare> = Vec::with_capacity(shamir.threshold);
        for li in 0..shamir.threshold {
            let sublock_cands: Vec<(u16, [u16; 2])> =
                per_lock_slots[li].iter().map(|x| x.unwrap()).collect();
            let pt = crate::lockable_ringlwe::decrypt_payload_from_sublock_s_candidates(
                &locks[li],
                &sublock_cands,
            )
            .expect("decrypt_payload");
            assert_eq!(pt.len(), 32, "share wrong length");
            let mut b = [0u8; 32];
            b.copy_from_slice(&pt);
            selected.push(ShamirShare {
                index: shares[li].index,
                value: b,
            });
        }
        let recovered = reconstruct_secret_32(&shamir, &selected).expect("reconstruct_secret_32");
        assert_eq!(recovered, secret, "failed to reconstruct correct secret");

        // Print global-check-only security accounting.
        let p = locks[0].p_channels;
        let l_required = shamir.threshold;
        let classical_bits =
            crate::lockable_ringlwe::classical_bits_global_check_only(p, l_required);
        let grover_bits = crate::lockable_ringlwe::grover_bits_global_check_only(p, l_required);
        let meets_pq128 =
            crate::lockable_ringlwe::meets_pq128_grover_global_check_only(p, l_required);
        eprintln!(
            "[amplification] model=global_check_only P={}, required_locks={}, cands=2^{:.0}, grover_bits={:.1}, pq128={}",
            p, l_required, classical_bits, grover_bits, meets_pq128
        );

        maybe_print_rss("tiny_payload_shamir:after_prove_decap_stream");
        eprintln!(
            "[tiny_payload_shamir] prove+decap(stream) in {:?}",
            t_prove.elapsed()
        );

        // Now safe to reclaim disk space used by the shape files (unless caching).
        if !keep_tiny_cache {
            crate::fs_cleanup::fast_remove_dir_best_effort(&out_dir);
        }
    }

    /// End-to-end PVUGC test: 3 armers × secp256k1 → P2WPKH Bitcoin address → WE lock → decap → recover.
    ///
    /// This is the "ultimate test": no per-lock tags, no per-armer hashes. The ONLY verification
    /// is that the recovered combined secret key produces the same P2WPKH address.
    #[test]
    #[ignore = "very slow in debug: runs full DPP prove+decap for 3 armers; run with `--release`"]
    fn test_pvugc_3armer_btc_address_we_lock_e2e() {
        use crate::lockable_ringlwe::RingLweParams;
        use crate::shamir_gf256::{reconstruct_secret_32, split_secret_32, ShamirConfig, ShamirShare};
        use crate::we_statement::encode_public_x;
        use crate::we_tiny_lock::arm_lfplus_ringlwe_lock;
        use crate::utils::maybe_print_rss;
        use k256::{ProjectivePoint, Scalar};
        use rand::{rngs::StdRng, RngCore, SeedableRng};
        use sha2::Digest;
        use std::time::Instant;

        const N_ARMERS: usize = 3;

        // --- Bitcoin P2WPKH address derivation ---
        fn pubkey_to_p2wpkh_hash(pubkey_compressed: &[u8; 33]) -> [u8; 20] {
            let sha = sha2::Sha256::digest(pubkey_compressed);
            let ripe = ripemd::Ripemd160::digest(&sha);
            let mut out = [0u8; 20];
            out.copy_from_slice(&ripe);
            out
        }

        fn scalar_from_bytes_mod_order(bytes: &[u8; 32]) -> Scalar {
            // Reduce mod secp256k1 order (k256 handles this).
            use k256::elliptic_curve::ops::Reduce;
            let uint = k256::U256::from_be_slice(bytes);
            Scalar::reduce(uint)
        }

        fn point_to_compressed(pt: &ProjectivePoint) -> [u8; 33] {
            use k256::elliptic_curve::group::GroupEncoding;
            let bytes = pt.to_bytes();
            let mut out = [0u8; 33];
            out.copy_from_slice(&bytes);
            out
        }

        // --- DPP / WE gate setup (reuse minimal params from the Shamir test) ---
        let ring_dim = <R as PolyRing>::dimension() as u64;
        let nvars_min = 12u64;
        let params = WeParams {
            nvars_setchk: nvars_min,
            degree_setchk: 3,
            nvars_cm: nvars_min,
            degree_cm: 2,
            kappa: 1,
            ring_dim_d: ring_dim,
            decomp_b: 16,
            k: 1,
            l: 1,
            mlen: 0,
        };
        let public_inputs_len = 8usize;
        let n_lin_proofs = 1usize;
        let mlen_mats = 0usize;
        let pairs: Vec<(usize, usize)> = vec![(0, 0)];
        type BF0 = <<R as PolyRing>::BaseRing as ark_ff::Field>::BasePrimeField;
        let mut public_inputs_bf: Vec<BF0> = (0..public_inputs_len)
            .map(|i| if (i % 3) == 0 { BF0::ONE } else { BF0::ZERO })
            .collect();
        if !public_inputs_bf.is_empty() {
            public_inputs_bf[0] = BF0::ZERO;
        }
        let _public_inputs_bytes_f257: Vec<F257> = public_inputs_bf
            .iter()
            .flat_map(|x| {
                let bytes = latticefold::transcript::bytes::prime_field_to_bytes_le_fixed::<BF0>(x);
                bytes.into_iter().map(|b| F257::from(b as u64))
            })
            .collect();

        let trace = super::poseidon_trace_schedule_for_plus_with_public_inputs::<R>(
            &public_inputs_bf,
            &params,
            n_lin_proofs,
            mlen_mats,
        )
        .expect("poseidon_trace_schedule_for_plus_with_public_inputs");

        let out_dir = {
            let mut p = std::env::temp_dir();
            p.push("lfplus_test_btc_3armer");
            let _ = std::fs::remove_dir_all(&p);
            std::fs::create_dir_all(&p).expect("create temp out_dir");
            p
        };
        let proof = dummy_plus_proof_shape::<R>(&params, mlen_mats, n_lin_proofs)
            .expect("dummy_plus_proof_shape");
        let vk_hash = [1u8; 32];
        let r1cs_digest = [2u8; 32];
        let gate_digest = [3u8; 32];
        let committed_values_prefix_bytes: [u8; CV_PREFIX_BYTES] = {
            let mut out = [0u8; CV_PREFIX_BYTES];
            let pis = public_inputs_bf
                .get(0..8)
                .expect("test requires at least 8 public inputs for committed-values prefix");
            for i in 0..8usize {
                let bytes =
                    latticefold::transcript::bytes::prime_field_to_bytes_le_fixed::<BF0>(&pis[i]);
                out[8 * i..8 * i + 8].copy_from_slice(&bytes);
            }
            out
        };
        let committed_values_prefix_f257: [F257; CV_PREFIX_BYTES] =
            committed_values_prefix_bytes.map(|b| F257::from(b as u64));
        let stmt_digest_f257 = crate::we_statement::we_statement_hash_lf_plus::<R>(
            vk_hash,
            committed_values_prefix_f257,
            r1cs_digest,
            gate_digest,
            &params,
        );
        let binding = WeStatementBindingWitness {
            vk_hash,
            r1cs_digest,
            gate_digest,
            committed_values_prefix_bytes,
        };
        let (shape, asg) = build_or_load_we_plus_tiny_dr1cs::<R>(
            &trace, &params, &stmt_digest_f257, &binding, &proof, mlen_mats, &pairs, &out_dir,
        )
        .expect("build_or_load_we_plus_tiny_dr1cs");
        shape.inst.check(&asg).expect("shape satisfied");

        let public_len = shape.public_len;
        let x = encode_public_x::<F257>(&stmt_digest_f257);
        let z_w = asg[public_len..].to_vec();

        let prover = crate::we_tiny_lock::we_ringlwe_prover_from_dr1cs::<F257>(
            shape.inst.clone(), shape.public_len,
        )
        .expect("we_ringlwe_prover_from_dr1cs");

        // =====================================================================
        // Phase 1: ARMING (3 armers, each with a secp256k1 secret)
        // =====================================================================
        let mut rng = StdRng::seed_from_u64(20260206);
        // Full K-of-K in the BTC e2e harness.
        let shamir = ShamirConfig {
            threshold: 16,
            shares: 16,
        };
        let ringlwe_params = RingLweParams::default();

        // Each armer samples a secret scalar. Individual P_j = s_j·G are NEVER published.
        // Only the combined P_combined is public (derived Bitcoin address).
        //
        // Production protocol (2 rounds, private channel between armers):
        //   Round 1: each armer j sends commitment H_j = SHA256(P_j || nonce_j) to all others.
        //   Round 2: each armer j reveals (P_j, nonce_j); all verify H_j and compute P_combined.
        //   Only P_combined is published on-chain. Individual P_j stay private to the armer group.
        //
        // This prevents:
        //   - Public per-armer oracle: nobody outside the group knows P_j
        //   - Equivocation: commitment H_j binds P_j before reveal
        //
        // Here we simulate this: each armer generates s_j, the test computes P_combined
        // (as the armers would privately), and only P_combined + the address are "public."

        let mut armer_secrets: Vec<[u8; 32]> = Vec::with_capacity(N_ARMERS);
        // Simulate private P_j accumulation (armers share P_j only with each other).
        let mut p_combined = ProjectivePoint::IDENTITY;
        for _j in 0..N_ARMERS {
            let mut sk = [0u8; 32];
            rng.fill_bytes(&mut sk);
            sk[31] |= 1; // ensure nonzero
            let scalar = scalar_from_bytes_mod_order(&sk);
            let scalar_bytes: [u8; 32] = scalar.to_bytes().into();
            armer_secrets.push(scalar_bytes);
            // Each armer adds their P_j to the running sum (private, between armers only).
            p_combined += ProjectivePoint::GENERATOR * scalar;
        }

        // ONLY P_combined is public. No individual P_j escapes the armer group.
        let p_combined_compressed = point_to_compressed(&p_combined);
        let address_hash = pubkey_to_p2wpkh_hash(&p_combined_compressed);
        eprintln!("[btc_3armer] P_combined (public) = {:?}", p_combined_compressed);
        eprintln!("[btc_3armer] P2WPKH address hash (public) = {:02x?}", address_hash);

        // Each armer: Shamir-split their secret, arm R locks.
        let t_arm = Instant::now();
        #[derive(Clone)]
        struct ArmerLockset<F: PrimeField> {
            locks: Vec<crate::lockable_ringlwe::RingLweLockArtifact<F>>,
            share_indices: Vec<u32>,
        }
        let mut armer_locksets: Vec<ArmerLockset<F257>> = Vec::with_capacity(N_ARMERS);
        for j in 0..N_ARMERS {
            let shares = split_secret_32(&mut rng, &shamir, armer_secrets[j])
                .expect("split_secret_32");
            let mut locks = Vec::with_capacity(shamir.shares);
            let mut indices = Vec::with_capacity(shamir.shares);
            for i in 0..shamir.shares {
                let mut rep_id = (j * 1000 + i) as u64;
                let lock = loop {
                    match arm_lfplus_ringlwe_lock::<R>(
                        shape.clone(),
                        &stmt_digest_f257,
                        // Unique armer seed per (armer, lock).
                        {
                            let mut seed = [0u8; 32];
                            seed[0] = j as u8;
                            seed[1] = i as u8;
                            seed[2..4].copy_from_slice(&(rep_id as u16).to_le_bytes());
                            seed
                        },
                        0,
                        0,
                        crate::we_tiny_lock::WeRingLweLockArmingPolicy {
                            base_rep_id: rep_id,
                            max_rep_tries: 32,
                            hint_budget_bytes: None,
                        },
                        ringlwe_params.clone(),
                        1,
                        &shares[i].value,
                        &mut rng,
                    ) {
                        Ok(lock_out) => break lock_out.lock,
                        Err(e) if e.contains("resample rep_id") => {
                            rep_id += 1;
                            continue;
                        }
                        Err(e) => panic!("arm armer[{j}] lock[{i}] failed: {e}"),
                    }
                };
                indices.push(shares[i].index);
                locks.push(lock);
            }
            armer_locksets.push(ArmerLockset { locks, share_indices: indices });
        }
        eprintln!(
            "[btc_3armer] armed {} armers × {} locks each in {:?}",
            N_ARMERS, shamir.shares, t_arm.elapsed()
        );

        // =====================================================================
        // Phase 2: DECAP — single π₀ pass feeds ALL armers' locks in parallel
        // =====================================================================
        let t_decap = Instant::now();
        maybe_print_rss("btc_3armer:before_decap");

        // Flatten all locks across all armers.
        let all_locks: Vec<&crate::lockable_ringlwe::RingLweLockArtifact<F257>> = armer_locksets
            .iter()
            .flat_map(|ls| ls.locks.iter())
            .collect();

        // Flatten all sublocks across all locks into one streaming pass.
        let mut coins_list: Vec<_> = Vec::new();
        let mut meta: Vec<(usize, usize)> = Vec::new(); // (lock_idx, sublock_idx)
        for (li, lock) in all_locks.iter().enumerate() {
            for (si, _sl) in lock.sublocks.iter().enumerate() {
                let sl = &lock.sublocks[si];
                coins_list.push(
                    prover
                        .derive_public_coins_from_stmt(lock.c_stmt.as_slice(), sl.block_id as usize, sl.rep_id)
                        .expect("derive_public_coins_from_stmt"),
                );
                meta.push((li, si));
            }
        }

        // Stream π₀ ONCE — all sublocks absorb the same chunks simultaneously.
        // Canonical path: stream tails and fold them immediately (no tail vectors).
        let mut tail_dots_mod257: Vec<u16> = vec![0u16; coins_list.len()];
        let mut cur_ci: Option<usize> = None;
        let mut cur_blk: usize = 0;
        let mut buf_len: usize = 0;
        let mut buf64: [u16; 64] = [0u16; 64];
        let abg_list = prover
            .stream_pi0_and_collect_tails(
                &x,
                &z_w,
                &coins_list,
                &mut |_chunk| {},
                &mut |ci, _ti, t| {
                    if cur_ci != Some(ci) {
                        cur_ci = Some(ci);
                        cur_blk = 0;
                        buf_len = 0;
                    }
                    let td = crate::lockable_ringlwe::field_mod257_u16(t);
                    buf64[buf_len] = td;
                    buf_len += 1;
                    if buf_len == 64 {
                        let (li, si) = meta[ci];
                        let lock = all_locks[li];
                        let sl = &lock.sublocks[si];
                        let blk = &sl.hints.tail_scales[cur_blk];
                        let add = crate::lockable_ringlwe::dot_packed_block_mod257_u16(blk, &buf64);
                        let acc = &mut tail_dots_mod257[ci];
                        *acc = crate::lockable_ringlwe::add_mod257_u16(*acc, add);
                        cur_blk += 1;
                        buf_len = 0;
                    }
                },
            )
            .expect("stream_pi0_and_collect_tails");
        assert_eq!(abg_list.len(), meta.len());
        assert_eq!(buf_len, 0);

        // Per lock: collect sublock candidates in canonical order, then decrypt deterministically.
        let mut per_lock_sublock_cands: Vec<Vec<Option<(u16, [u16; 2])>>> = all_locks
            .iter()
            .map(|l| vec![None; l.sublocks.len()])
            .collect();
        for (gi, abgt) in abg_list.iter().enumerate() {
            let (li, si) = meta[gi];
            let lock = all_locks[li];
            let sl = &lock.sublocks[si];
            let td = tail_dots_mod257[gi];
            let cands = crate::lockable_ringlwe::sublock_s_candidates_from_abg_tail(sl, abgt, td)
            .unwrap_or_else(|e| panic!("sublock candidates[{gi}]: {e}"));
            let ch = sl.channel_id;
            per_lock_sublock_cands[li][si] = Some((ch, cands));
        }

        let mut per_lock_share_bytes: Vec<[u8; 32]> = Vec::with_capacity(all_locks.len());
        for (li, lock) in all_locks.iter().enumerate() {
            let slots = &per_lock_sublock_cands[li];
            let sublock_cands: Vec<(u16, [u16; 2])> =
                slots.iter().map(|x| x.unwrap()).collect();
            let pt =
                crate::lockable_ringlwe::decrypt_payload_from_sublock_s_candidates(lock, &sublock_cands)
                    .unwrap_or_else(|e| panic!("lock[{li}] decrypt_payload: {e}"));
            assert_eq!(pt.len(), 32);
            let mut b = [0u8; 32];
            b.copy_from_slice(&pt);
            per_lock_share_bytes.push(b);
        }

        // Reconstruct each armer's secret deterministically (no branching).
        let mut armer_decapped_secrets: Vec<[u8; 32]> = Vec::with_capacity(N_ARMERS);
        for j in 0..N_ARMERS {
            let mut selected: Vec<ShamirShare> = Vec::with_capacity(shamir.threshold);
            for i in 0..shamir.threshold {
                let global_li = j * shamir.shares + i;
                selected.push(ShamirShare {
                    index: armer_locksets[j].share_indices[i],
                    value: per_lock_share_bytes[global_li],
                });
            }
            let secret = reconstruct_secret_32(&shamir, &selected).expect("reconstruct_secret_32");
            armer_decapped_secrets.push(secret);
        }

        // Combine armers and verify address binding.
        let mut sc_sum = Scalar::ZERO;
        for s in &armer_decapped_secrets {
            sc_sum = sc_sum + scalar_from_bytes_mod_order(s);
        }
        let pk_verify = ProjectivePoint::GENERATOR * sc_sum;
        assert_eq!(
            point_to_compressed(&pk_verify),
            p_combined_compressed,
            "recovered key does not match combined public key"
        );

        maybe_print_rss("btc_3armer:after_decap");
        eprintln!("[btc_3armer] decap + reconstruct + verify in {:?}", t_decap.elapsed());

        let p = all_locks[0].p_channels;
        let l_required_per_armer = shamir.threshold;
        let l_required_system = shamir.threshold * N_ARMERS;
        let classical_bits_per_armer =
            crate::lockable_ringlwe::classical_bits_global_check_only(p, l_required_per_armer);
        let grover_bits_per_armer =
            crate::lockable_ringlwe::grover_bits_global_check_only(p, l_required_per_armer);
        let classical_bits_system =
            crate::lockable_ringlwe::classical_bits_global_check_only(p, l_required_system);
        let grover_bits_system =
            crate::lockable_ringlwe::grover_bits_global_check_only(p, l_required_system);
        let meets_pq128_system =
            crate::lockable_ringlwe::meets_pq128_grover_global_check_only(p, l_required_system);
        eprintln!(
            "[amplification] model=global_check_only P={}, required_locks_per_armer={}, required_locks_system={}, cands_per_armer=2^{:.0}, grover_bits_per_armer={:.1}, cands_system=2^{:.0}, grover_bits_system={:.1}, pq128_system={}",
            p,
            l_required_per_armer,
            l_required_system,
            classical_bits_per_armer,
            grover_bits_per_armer,
            classical_bits_system,
            grover_bits_system,
            meets_pq128_system
        );

        // =====================================================================
        // Adversarial tests: verify the lock rejects corrupted inputs.
        // =====================================================================

        // --- ADV 1: Corrupted proof tail (one bit flip in first tail) ---
        // Restream with a corrupted tail. Should fail to recover the Bitcoin address.
        {
            // Flatten sublocks.
            let mut adv_coins: Vec<_> = Vec::new();
            let mut adv_meta: Vec<(usize, usize)> = Vec::new();
            for (li, lock) in all_locks.iter().enumerate() {
                for (si, _sl) in lock.sublocks.iter().enumerate() {
                    let sl = &lock.sublocks[si];
                    adv_coins.push(
                        prover
                            .derive_public_coins_from_stmt(lock.c_stmt.as_slice(), sl.block_id as usize, sl.rep_id)
                            .expect("derive_public_coins_from_stmt"),
                    );
                    adv_meta.push((li, si));
                }
            }

            // Canonical path: stream tails and fold them immediately (no tail vectors).
            // We simulate a "corrupted tail element" by perturbing the first visited tail element
            // for the first coin before folding.
            let mut tail_dots_mod257: Vec<u16> = vec![0u16; adv_coins.len()];
            let mut cur_ci: Option<usize> = None;
            let mut cur_blk: usize = 0;
            let mut buf_len: usize = 0;
            let mut buf64: [u16; 64] = [0u16; 64];
            let adv_abg_list = prover
                .stream_pi0_and_collect_tails(
                    &x,
                    &z_w,
                    &adv_coins,
                    &mut |_chunk| {},
                    &mut |ci, ti, t| {
                        if cur_ci != Some(ci) {
                            cur_ci = Some(ci);
                            cur_blk = 0;
                            buf_len = 0;
                        }
                        let mut td = crate::lockable_ringlwe::field_mod257_u16(t);
                        if ci == 0 && ti == 0 {
                            td = crate::lockable_ringlwe::add_mod257_u16(td, 42u16);
                        }
                        buf64[buf_len] = td;
                        buf_len += 1;
                        if buf_len == 64 {
                            let (li, si) = adv_meta[ci];
                            let lock = all_locks[li];
                            let sl = &lock.sublocks[si];
                            let blk = &sl.hints.tail_scales[cur_blk];
                            let add = crate::lockable_ringlwe::dot_packed_block_mod257_u16(blk, &buf64);
                            let acc = &mut tail_dots_mod257[ci];
                            *acc = crate::lockable_ringlwe::add_mod257_u16(*acc, add);
                            cur_blk += 1;
                            buf_len = 0;
                        }
                    },
                )
                .expect("adv stream");
            assert_eq!(adv_abg_list.len(), adv_meta.len());
            assert_eq!(buf_len, 0);

            let mut per_lock_slots: Vec<Vec<Option<(u16, [u16; 2])>>> =
                all_locks.iter().map(|l| vec![None; l.sublocks.len()]).collect();
            for (gi, abgt) in adv_abg_list.iter().enumerate() {
                let (li, si) = adv_meta[gi];
                let lock = all_locks[li];
                let sl = &lock.sublocks[si];
                let td = tail_dots_mod257[gi];
                let cands = crate::lockable_ringlwe::sublock_s_candidates_from_abg_tail(sl, abgt, td)
                    .unwrap();
                let ch = sl.channel_id;
                per_lock_slots[li][si] = Some((ch, cands));
            }

            let mut ok = true;
            let mut shares: Vec<[u8; 32]> = Vec::with_capacity(all_locks.len());
            for (li, lock) in all_locks.iter().enumerate() {
                let sublock_cands: Vec<(u16, [u16; 2])> =
                    per_lock_slots[li].iter().map(|x| x.unwrap()).collect();
                let pt = match crate::lockable_ringlwe::decrypt_payload_from_sublock_s_candidates(
                    lock,
                    &sublock_cands,
                ) {
                    Ok(v) => v,
                    Err(_) => {
                        ok = false;
                        break;
                    }
                };
                if pt.len() != 32 {
                    ok = false;
                    break;
                }
                let mut b = [0u8; 32];
                b.copy_from_slice(&pt);
                shares.push(b);
            }

            if ok {
                let mut sc_sum = Scalar::ZERO;
                for j in 0..N_ARMERS {
                    let mut selected: Vec<ShamirShare> = Vec::with_capacity(shamir.threshold);
                    for i in 0..shamir.threshold {
                        let global_li = j * shamir.shares + i;
                        selected.push(ShamirShare {
                            index: armer_locksets[j].share_indices[i],
                            value: shares[global_li],
                        });
                    }
                    let s = reconstruct_secret_32(&shamir, &selected).unwrap();
                    sc_sum = sc_sum + scalar_from_bytes_mod_order(&s);
                }
                let pk = ProjectivePoint::GENERATOR * sc_sum;
                assert_ne!(
                    pubkey_to_p2wpkh_hash(&point_to_compressed(&pk)),
                    address_hash,
                    "ADV1: corrupted tail should NOT recover the Bitcoin address"
                );
            }
            eprintln!("[btc_3armer] ADV1 (corrupted tail): correctly rejected");
        }

        // --- ADV 2: Tampered ciphertext (bit flip in armer 0's first lock) ---
        {
            let mut tampered_locksets = armer_locksets.clone();
            if let Some(first_byte) = tampered_locksets[0].locks[0].ct.ct.first_mut() {
                *first_byte ^= 0x80;
            }
            let tampered_all_locks: Vec<&crate::lockable_ringlwe::RingLweLockArtifact<F257>> =
                tampered_locksets.iter().flat_map(|ls| ls.locks.iter()).collect();

            // Flatten sublocks.
            let mut tam_coins: Vec<_> = Vec::new();
            let mut tam_meta: Vec<(usize, usize)> = Vec::new();
            for (li, lock) in tampered_all_locks.iter().enumerate() {
                for (si, _sl) in lock.sublocks.iter().enumerate() {
                    let sl = &lock.sublocks[si];
                    tam_coins.push(
                        prover
                            .derive_public_coins_from_stmt(lock.c_stmt.as_slice(), sl.block_id as usize, sl.rep_id)
                            .expect("derive_public_coins_from_stmt"),
                    );
                    tam_meta.push((li, si));
                }
            }

            let mut tail_dots_mod257: Vec<u16> = vec![0u16; tam_coins.len()];
            let mut cur_ci: Option<usize> = None;
            let mut cur_blk: usize = 0;
            let mut buf_len: usize = 0;
            let mut buf64: [u16; 64] = [0u16; 64];
            let tam_abg_list = prover
                .stream_pi0_and_collect_tails(
                    &x,
                    &z_w,
                    &tam_coins,
                    &mut |_chunk| {},
                    &mut |ci, _ti, t| {
                        if cur_ci != Some(ci) {
                            cur_ci = Some(ci);
                            cur_blk = 0;
                            buf_len = 0;
                        }
                        let td = crate::lockable_ringlwe::field_mod257_u16(t);
                        buf64[buf_len] = td;
                        buf_len += 1;
                        if buf_len == 64 {
                            let (li, si) = tam_meta[ci];
                            let lock = tampered_all_locks[li];
                            let sl = &lock.sublocks[si];
                            let blk = &sl.hints.tail_scales[cur_blk];
                            let add = crate::lockable_ringlwe::dot_packed_block_mod257_u16(blk, &buf64);
                            let acc = &mut tail_dots_mod257[ci];
                            *acc = crate::lockable_ringlwe::add_mod257_u16(*acc, add);
                            cur_blk += 1;
                            buf_len = 0;
                        }
                    },
                )
                .expect("tam stream");
            assert_eq!(tam_abg_list.len(), tam_meta.len());
            assert_eq!(buf_len, 0);

            let mut per_lock_slots: Vec<Vec<Option<(u16, [u16; 2])>>> = tampered_all_locks
                .iter()
                .map(|l| vec![None; l.sublocks.len()])
                .collect();
            for (gi, abgt) in tam_abg_list.iter().enumerate() {
                let (li, si) = tam_meta[gi];
                let lock = tampered_all_locks[li];
                let sl = &lock.sublocks[si];
                let td = tail_dots_mod257[gi];
                let cands = crate::lockable_ringlwe::sublock_s_candidates_from_abg_tail(sl, abgt, td)
                    .unwrap();
                let ch = sl.channel_id;
                per_lock_slots[li][si] = Some((ch, cands));
            }

            let mut ok = true;
            let mut shares: Vec<[u8; 32]> = Vec::with_capacity(tampered_all_locks.len());
            for (li, lock) in tampered_all_locks.iter().enumerate() {
                let sublock_cands: Vec<(u16, [u16; 2])> =
                    per_lock_slots[li].iter().map(|x| x.unwrap()).collect();
                let pt = match crate::lockable_ringlwe::decrypt_payload_from_sublock_s_candidates(
                    lock,
                    &sublock_cands,
                ) {
                    Ok(v) => v,
                    Err(_) => {
                        ok = false;
                        break;
                    }
                };
                if pt.len() != 32 {
                    ok = false;
                    break;
                }
                let mut b = [0u8; 32];
                b.copy_from_slice(&pt);
                shares.push(b);
            }

            if ok {
                let mut sc_sum = Scalar::ZERO;
                for j in 0..N_ARMERS {
                    let mut selected: Vec<ShamirShare> = Vec::with_capacity(shamir.threshold);
                    for i in 0..shamir.threshold {
                        let global_li = j * shamir.shares + i;
                        selected.push(ShamirShare {
                            index: tampered_locksets[j].share_indices[i],
                            value: shares[global_li],
                        });
                    }
                    let s = reconstruct_secret_32(&shamir, &selected).unwrap();
                    sc_sum = sc_sum + scalar_from_bytes_mod_order(&s);
                }
                let pk = ProjectivePoint::GENERATOR * sc_sum;
                assert_ne!(
                    pubkey_to_p2wpkh_hash(&point_to_compressed(&pk)),
                    address_hash,
                    "ADV2: tampered ciphertext should NOT recover the Bitcoin address"
                );
            }
            eprintln!("[btc_3armer] ADV2 (tampered ciphertext): correctly rejected");
        }

        // --- ADV 3: Wrong address (different combined key) ---
        // Even with correct decap, the recovered key shouldn't match a DIFFERENT address.
        {
            let wrong_scalar = scalar_from_bytes_mod_order(&[0xFFu8; 32]);
            let wrong_pk = ProjectivePoint::GENERATOR * wrong_scalar;
            let wrong_hash = pubkey_to_p2wpkh_hash(&point_to_compressed(&wrong_pk));
            assert_ne!(wrong_hash, address_hash, "sanity: wrong address should differ");
            // The correctly recovered s_combined should NOT match the wrong address.
            let wrong_check = pubkey_to_p2wpkh_hash(&point_to_compressed(
                &pk_verify
            ));
            assert_ne!(wrong_check, wrong_hash, "ADV3: correct key should not match wrong address");
            eprintln!("[btc_3armer] ADV3 (wrong address): correctly rejected");
        }

        eprintln!("[btc_3armer] all adversarial tests passed");

        crate::fs_cleanup::fast_remove_dir_best_effort(&out_dir);
    }

    #[test]
    #[ignore = "very slow in debug: production-like tiny-gate params; run with `--release`"]
    fn test_tiny_gate_ringlwe_lock_roundtrip_large_trace_params() {
        use crate::lockable_ringlwe::RingLweParams;
        use crate::we_statement::encode_public_x;
        use crate::we_tiny_lock::arm_lfplus_ringlwe_lock;
        use crate::utils::maybe_print_rss;
        use rand::{rngs::StdRng, SeedableRng};
        use std::time::Instant;

        // "Large trace params" defaults (more production-like), but allow overriding down for
        // scaling studies:
        //   LFP_TINY_GATE_NVARS=12 LFP_TINY_GATE_K=1 LFP_TINY_GATE_KAPPA=1
        //
        // Interpreting your shorthand:
        // - "npow20" ~ nvars=20 (sumcheck rounds / transcript schedule depth)
        // - "k8"     ~ k=8      (rgchk/setchk block count)
        let nvars_min: u64 = std::env::var("LFP_TINY_GATE_NVARS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(20)
            .max(12);
        let k_rg: u64 = std::env::var("LFP_TINY_GATE_K")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(8)
            .max(1);
        let kappa: u64 = std::env::var("LFP_TINY_GATE_KAPPA")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(8)
            .max(1);
        assert!(
            (kappa as usize).is_power_of_two(),
            "tiny gate requires kappa power-of-two; set LFP_TINY_GATE_KAPPA=1/2/4/8/..."
        );

        let ring_dim = <R as PolyRing>::dimension() as u64;
        let params = WeParams {
            nvars_setchk: nvars_min,
            degree_setchk: 3,
            nvars_cm: nvars_min,
            degree_cm: 2,
            kappa,
            ring_dim_d: ring_dim,
            decomp_b: 16,
            k: k_rg,
            l: 1,
            mlen: 0,
        };
        let public_inputs_len = 8usize; // number of public **field elements**
        let n_lin_proofs = 1usize; // schedule builder currently assumes L=1
        let mlen_mats = 0usize;
        let pairs: Vec<(usize, usize)> = vec![(0, 0)];
        type BF0 = <<R as PolyRing>::BaseRing as ark_ff::Field>::BasePrimeField;

        // Public inputs as base-field elements (used for schedule trace generation).
        let mut public_inputs_bf: Vec<BF0> = (0..public_inputs_len)
            .map(|i| if (i % 3) == 0 { BF0::ONE } else { BF0::ZERO })
            .collect();
        // Dummy proof uses cm_f[*] = 0, so ensure the exposed prefix absorbed into the transcript
        // matches that zero prefix (for as many coordinates as we expose).
        let kappa_exposed = (kappa as usize).min(public_inputs_len);
        for i in 0..kappa_exposed {
            public_inputs_bf[i] = BF0::ZERO;
        }

        let _public_inputs_bytes_f257: Vec<F257> = public_inputs_bf
            .iter()
            .flat_map(|x| {
                let bytes = latticefold::transcript::bytes::prime_field_to_bytes_le_fixed::<BF0>(x);
                bytes.into_iter().map(|b| F257::from(b as u64))
            })
            .collect();

        eprintln!(
            "[tiny_gate_large] params: nvars={} kappa={} k={} ring_dim={}",
            nvars_min, kappa, k_rg, ring_dim
        );

        let trace = super::poseidon_trace_schedule_for_plus_with_public_inputs::<R>(
            &public_inputs_bf,
            &params,
            n_lin_proofs,
            mlen_mats,
        )
        .expect("poseidon_trace_schedule_for_plus_with_public_inputs");

        // Optional: keep the shape cache directory between runs (to benchmark cache-hit paths).
        let keep_tiny_cache = std::env::var("LFP_KEEP_TINY_GATE_CACHE")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false);

        // Canonical path: build/load shape + compute assignment.
        let t_shape = Instant::now();
        let out_dir = {
            let mut p = std::env::temp_dir();
            p.push(format!(
                "lfplus_test_tiny_shape_roundtrip_nvars{nvars_min}_kappa{kappa}_k{k_rg}"
            ));
            if !keep_tiny_cache {
                let _ = std::fs::remove_dir_all(&p);
            }
            std::fs::create_dir_all(&p).expect("create temp out_dir");
            p
        };
        let proof =
            dummy_plus_proof_shape::<R>(&params, mlen_mats, n_lin_proofs).expect("dummy_plus_proof_shape");
        let vk_hash = [1u8; 32];
        let r1cs_digest = [2u8; 32];
        let gate_digest = [3u8; 32];
        let committed_values_prefix_bytes: [u8; CV_PREFIX_BYTES] = {
            let mut out = [0u8; CV_PREFIX_BYTES];
            let pis = public_inputs_bf
                .get(0..8)
                .expect("test requires at least 8 public inputs for committed-values prefix");
            for i in 0..8usize {
                let bytes =
                    latticefold::transcript::bytes::prime_field_to_bytes_le_fixed::<BF0>(&pis[i]);
                out[8 * i..8 * i + 8].copy_from_slice(&bytes);
            }
            out
        };
        let committed_values_prefix_f257: [F257; CV_PREFIX_BYTES] =
            committed_values_prefix_bytes.map(|b| F257::from(b as u64));
        let stmt_digest_f257 = crate::we_statement::we_statement_hash_lf_plus::<R>(
            vk_hash,
            committed_values_prefix_f257,
            r1cs_digest,
            gate_digest,
            &params,
        );
        let binding = WeStatementBindingWitness {
            vk_hash,
            r1cs_digest,
            gate_digest,
            committed_values_prefix_bytes,
        };
        let (shape, asg) = build_or_load_we_plus_tiny_dr1cs::<R>(
            &trace,
            &params,
            &stmt_digest_f257,
            &binding,
            &proof,
            mlen_mats,
            &pairs,
            &out_dir,
        )
        .expect("build_or_load_we_plus_tiny_dr1cs");
        assert_eq!(asg.len(), shape.inst.nvars);
        shape.inst.check(&asg).expect("shape should be satisfied by witness assignment");
        eprintln!(
            "[tiny_gate_large] built shape in {:?}: public_len={} nvars={} constraints={}",
            t_shape.elapsed(),
            shape.public_len,
            shape.inst.nvars,
            shape.inst.layout.nconstraints
        );

        // Arm and then prove+decap using the satisfying assignment split into (x || z_w).
        let armer_seed = [7u8; 32];
        let lock_j = 0u64;
        let ringlwe_params = RingLweParams::default();
        let dummy_payload: [u8; 0] = [];

        let mut rng = StdRng::seed_from_u64(42);
        let public_len = shape.public_len;
        let prover = crate::we_tiny_lock::we_ringlwe_prover_from_dr1cs::<F257>(
            shape.inst.clone(),
            shape.public_len,
        )
        .expect("we_ringlwe_prover_from_dr1cs");
        let shape1 = shape.clone();
        let t_arm = Instant::now();
        let mut rep0 = 0u64;
        let lock0 = loop {
            match arm_lfplus_ringlwe_lock::<R>(
                shape.clone(),
                &stmt_digest_f257,
                armer_seed,
                lock_j,
                0,
                crate::we_tiny_lock::WeRingLweLockArmingPolicy {
                    base_rep_id: rep0,
                    max_rep_tries: 32,
                    hint_budget_bytes: None,
                },
                ringlwe_params.clone(),
                1,
                &dummy_payload,
                &mut rng,
            ) {
                Ok(lock_out) => break lock_out.lock,
                Err(e) if e.contains("resample rep_id") => {
                    rep0 += 1;
                    continue;
                }
                Err(e) => panic!("arm ctx0 failed: {e}"),
            }
        };
        let mut rep1 = rep0 + 1;
        let lock1 = loop {
            match arm_lfplus_ringlwe_lock::<R>(
                shape1.clone(),
                &stmt_digest_f257,
                armer_seed,
                lock_j,
                0,
                crate::we_tiny_lock::WeRingLweLockArmingPolicy {
                    base_rep_id: rep1,
                    max_rep_tries: 32,
                    hint_budget_bytes: None,
                },
                ringlwe_params.clone(),
                1,
                &dummy_payload,
                &mut rng,
            ) {
                Ok(lock_out) => break lock_out.lock,
                Err(e) if e.contains("resample rep_id") => {
                    rep1 += 1;
                    continue;
                }
                Err(e) => panic!("arm ctx1 failed: {e}"),
            }
        };
        eprintln!(
            "[tiny_gate_large] armed in {:?}: proof_len={}",
            t_arm.elapsed(),
            prover.proof_len()
        );

        let x = encode_public_x::<F257>(&stmt_digest_f257);
        assert_eq!(x.len(), public_len);
        assert_eq!(&asg[..public_len], x.as_slice(), "satisfying assignment public prefix mismatch");
        let z_w = asg[public_len..].to_vec();

        let t_prove = Instant::now();
        maybe_print_rss("tiny_gate_large:before_prove_decap_stream");
        let meta = [(0usize, 0usize), (1usize, 0usize)];
        let mut tail_dots_mod257: [u16; 2] = [0u16; 2];
        let mut cur_ci: Option<usize> = None;
        let mut cur_blk: usize = 0;
        let mut buf_len: usize = 0;
        let mut buf64: [u16; 64] = [0u16; 64];
        let abg_list = prover
            .stream_pi0_and_collect_tails(
                &x,
                &z_w,
                &[
                    prover
                        .derive_public_coins_from_stmt(
                            lock0.c_stmt.as_slice(),
                            lock0.sublocks[0].block_id as usize,
                            lock0.sublocks[0].rep_id,
                        )
                        .expect("derive_public_coins_from_stmt lock0"),
                    prover
                        .derive_public_coins_from_stmt(
                            lock1.c_stmt.as_slice(),
                            lock1.sublocks[0].block_id as usize,
                            lock1.sublocks[0].rep_id,
                        )
                        .expect("derive_public_coins_from_stmt lock1"),
                ],
                &mut |_chunk| {},
                &mut |ci, _ti, t| {
                    if cur_ci != Some(ci) {
                        cur_ci = Some(ci);
                        cur_blk = 0;
                        buf_len = 0;
                    }
                    let td = crate::lockable_ringlwe::field_mod257_u16(t);
                    buf64[buf_len] = td;
                    buf_len += 1;
                    if buf_len == 64 {
                        let (lock_id, si) = meta[ci];
                        let sl = match lock_id {
                            0 => &lock0.sublocks[si],
                            1 => &lock1.sublocks[si],
                            _ => unreachable!(),
                        };
                        let blk = &sl.hints.tail_scales[cur_blk];
                        let add = crate::lockable_ringlwe::dot_packed_block_mod257_u16(blk, &buf64);
                        tail_dots_mod257[ci] = crate::lockable_ringlwe::add_mod257_u16(tail_dots_mod257[ci], add);
                        cur_blk += 1;
                        buf_len = 0;
                    }
                },
            )
            .expect("stream_pi0_and_collect_tails");
        assert_eq!(abg_list.len(), 2);
        assert_eq!(buf_len, 0);
        let _s0 = crate::lockable_ringlwe::sublock_s_candidates_from_abg_tail(
            &lock0.sublocks[0],
            &abg_list[0],
            tail_dots_mod257[0],
        )
        .expect("decap_finish0");
        let _s1 = crate::lockable_ringlwe::sublock_s_candidates_from_abg_tail(
            &lock1.sublocks[0],
            &abg_list[1],
            tail_dots_mod257[1],
        )
        .expect("decap_finish1");
        for lock in [&lock0, &lock1] {
            assert_eq!(lock.sublocks.len(), 1);
            assert_eq!(lock.sublocks[0].accepting_set[0], F257::ONE);
            assert_eq!(lock.sublocks[0].accepting_set[1], F257::from(2u64));
        }
        maybe_print_rss("tiny_gate_large:after_prove_decap_stream");
        eprintln!("[tiny_gate_large] prove+decap(stream) in {:?}", t_prove.elapsed());

        // Now it is safe to reclaim disk space used by the shape files (unless caching).
        if !keep_tiny_cache {
            crate::fs_cleanup::fast_remove_dir_best_effort(&out_dir);
        }
    }

    #[allow(dead_code)]
    fn identity_cs(n: usize) -> (R1CS<R>, Vec<R>) {
        let r1cs = R1CS::<R> {
            l: 1,
            A: SparseMatrix::identity(n),
            B: SparseMatrix::identity(n),
            C: SparseMatrix::identity(n),
        };
        let z = vec![<R as stark_rings::Ring>::ONE; n];
        (r1cs, z)
    }

    #[test]
    #[ignore = "tiny-field gate coverage; run with `--release -- --ignored`"]
    fn test_we_plus_tiny_param_binding_mlen_changes_shape() {
        use cyclotomic_rings::rings::GoldilocksRing64 as RR;
        use stark_rings::PolyRing;

        // Choose params so the schedule meaningfully depends on `mlen` (mlen_mats).
        let ring_dim = RR::dimension() as u64;
        let pairs: Vec<(usize, usize)> = vec![(0, 0)];
        // Prefix binding requires at least `min(8, kappa)` public inputs in the transcript prefix.
        // Here kappa=1 => need at least 1.
        let public_inputs_len = 1usize;
        let n_lin_proofs = 1usize;

        let base = WeParams {
            // For ring_dim=64, CM verifier math needs nvars_cm >= 12.
            nvars_setchk: 12,
            degree_setchk: 3,
            nvars_cm: 12,
            degree_cm: 2,
            kappa: 1,
            ring_dim_d: ring_dim,
            decomp_b: 16,
            k: 1,
            l: 1,
            mlen: 0,
        };
        let mut alt = base.clone();
        alt.mlen = 1;

        // Arm-time binding: changing `mlen` should change the arm-time circuit shape.
        let out_dir0 = {
            let mut p = std::env::temp_dir();
            p.push("lfplus_test_tiny_shape_mlen0");
            let _ = std::fs::remove_dir_all(&p);
            std::fs::create_dir_all(&p).expect("create temp out_dir0");
            p
        };
        let out_dir1 = {
            let mut p = std::env::temp_dir();
            p.push("lfplus_test_tiny_shape_mlen1");
            let _ = std::fs::remove_dir_all(&p);
            std::fs::create_dir_all(&p).expect("create temp out_dir1");
            p
        };
        let coeff_bytes = ((<RR as PolyRing>::BaseRing::MODULUS_BIT_SIZE as usize) + 7) / 8;
        let _public_inputs_bytes_f257: Vec<F257> =
            vec![F257::ZERO; public_inputs_len * coeff_bytes];

        let trace0 = poseidon_trace_schedule_for_plus_with_public_inputs::<RR>(
            &vec![<<RR as PolyRing>::BaseRing as ark_ff::Field>::BasePrimeField::ZERO; public_inputs_len],
            &base,
            n_lin_proofs,
            0,
        )
        .expect("poseidon_trace_schedule_for_plus_with_public_inputs(base)");
        let proof0 = dummy_plus_proof_shape::<RR>(&base, 0, n_lin_proofs).expect("dummy_plus_proof_shape(base)");
        let vk_hash = [1u8; 32];
        let r1cs_digest = [2u8; 32];
        let gate_digest = [3u8; 32];
        // This test builds the trace with all-zero public inputs, so the absorbed prefix bytes are zero.
        let committed_values_prefix_bytes = [0u8; CV_PREFIX_BYTES];
        let committed_values_prefix_f257: [F257; CV_PREFIX_BYTES] =
            committed_values_prefix_bytes.map(|b| F257::from(b as u64));
        let binding = WeStatementBindingWitness {
            vk_hash,
            r1cs_digest,
            gate_digest,
            committed_values_prefix_bytes,
        };
        let stmt_digest_f257 = crate::we_statement::we_statement_hash_lf_plus::<RR>(
            vk_hash,
            committed_values_prefix_f257,
            r1cs_digest,
            gate_digest,
            &base,
        );
        let (s0, _) = build_or_load_we_plus_tiny_dr1cs::<RR>(
            &trace0, &base, &stmt_digest_f257, &binding, &proof0, 0, &pairs, &out_dir0,
        )
        .expect("shape(base)");
        let trace1 = poseidon_trace_schedule_for_plus_with_public_inputs::<RR>(
            &vec![<<RR as PolyRing>::BaseRing as ark_ff::Field>::BasePrimeField::ZERO; public_inputs_len],
            &alt,
            n_lin_proofs,
            1,
        )
        .expect("poseidon_trace_schedule_for_plus_with_public_inputs(alt)");
        let proof1 = dummy_plus_proof_shape::<RR>(&alt, 1, n_lin_proofs).expect("dummy_plus_proof_shape(alt)");
        let stmt_digest_alt = crate::we_statement::we_statement_hash_lf_plus::<RR>(
            vk_hash,
            committed_values_prefix_f257,
            r1cs_digest,
            gate_digest,
            &alt,
        );
        let (s1, _) = build_or_load_we_plus_tiny_dr1cs::<RR>(
            &trace1, &alt, &stmt_digest_alt, &binding, &proof1, 1, &pairs, &out_dir1,
        )
        .expect("shape(alt)");
        assert_ne!(s0.public_len, 0);
        assert_ne!(s1.public_len, 0);
        // We don't require a specific delta, but `mlen` should affect at least one size metric.
        assert!(
            s0.inst.nvars != s1.inst.nvars
                || s0.inst.layout.nconstraints != s1.inst.layout.nconstraints
                || s0.inst.layout.a_terms != s1.inst.layout.a_terms
                || s0.inst.layout.b_terms != s1.inst.layout.b_terms
                || s0.inst.layout.c_terms != s1.inst.layout.c_terms,
            "expected mlen change to alter tiny gate shape metrics"
        );
        let _ = std::fs::remove_dir_all(&out_dir0);
        let _ = std::fs::remove_dir_all(&out_dir1);
    }

    #[test]
    #[ignore = "tiny-field gate coverage; run with `--release -- --ignored`"]
    fn test_we_plus_tiny_public_input_digest_unsat_on_flip() {
        use cyclotomic_rings::rings::GoldilocksRing64 as RR;
        use stark_rings::PolyRing;
        type BF0 = <<RR as PolyRing>::BaseRing as ark_ff::Field>::BasePrimeField;

        // Keep this small: we only need to cover "public input is bound into transcript absorbs".
        let ring_dim = RR::dimension() as u64;
        let public_inputs_len = 32usize; // number of public **field elements** (lighter than full SP1 digest)
        let n_lin_proofs = 1usize;
        let mlen_mats = 0usize;
        let pairs: Vec<(usize, usize)> = vec![(0, 0)];

        let params = WeParams {
            nvars_setchk: 12,
            degree_setchk: 3,
            nvars_cm: 12,
            degree_cm: 2,
            kappa: 1,
            ring_dim_d: ring_dim,
            decomp_b: 16,
            k: 1,
            l: 1,
            mlen: mlen_mats as u64,
        };

        // Choose a nontrivial public input pattern (digest-like bits).
        //
        // IMPORTANT: for kappa=1 the tiny gate's prefix binding enforces `cm_f[0]` matches
        // public_inputs[0] (constant-coefficient). Our dummy proof/trace uses `cm_f[0]=0`,
        // so keep public_inputs[0]=0 in this test; the flip below must still break satisfaction.
        let public_inputs_elems: Vec<F257> = (0..public_inputs_len)
            .map(|i| if (i % 3) == 1 { F257::ONE } else { F257::ZERO })
            .collect();
        // Tiny-gate statement public prefix uses byte encoding (8 bytes per base-field element).
        let _public_inputs_bytes: Vec<F257> = public_inputs_elems
            .iter()
            .flat_map(|x| {
                let u = x.into_bigint().as_ref().get(0).copied().unwrap_or(0) as u8;
                [u, 0, 0, 0, 0, 0, 0, 0].into_iter().map(|b| F257::from(b as u64))
            })
            .collect();
        let public_inputs_bf: Vec<BF0> = public_inputs_elems
            .iter()
            .map(|x| {
                let d = x.into_bigint().to_bytes_le().get(0).copied().unwrap_or(0) as u64;
                BF0::from(d)
            })
            .collect();

        // Build a self-consistent schedule trace whose initial absorbs match `public_inputs`.
        let trace = poseidon_trace_schedule_for_plus_with_public_inputs::<RR>(
            &public_inputs_bf,
            &params,
            n_lin_proofs,
            mlen_mats,
        )
        .expect("poseidon_trace_schedule_for_plus_with_public_inputs");

        let out_dir = {
            let mut p = std::env::temp_dir();
            p.push("lfplus_test_we_plus_tiny_public_inputs_bound");
            let _ = std::fs::remove_dir_all(&p);
            std::fs::create_dir_all(&p).expect("create temp out_dir");
            p
        };
        let proof = dummy_plus_proof_shape::<RR>(&params, mlen_mats, n_lin_proofs).expect("dummy_plus_proof_shape");
        let vk_hash = [1u8; 32];
        let r1cs_digest = [2u8; 32];
        let gate_digest = [3u8; 32];
        let committed_values_prefix_bytes: [u8; CV_PREFIX_BYTES] = {
            let mut out = [0u8; CV_PREFIX_BYTES];
            let pis = public_inputs_bf
                .get(0..8)
                .expect("test requires at least 8 public inputs for committed-values prefix");
            for i in 0..8usize {
                let bytes =
                    latticefold::transcript::bytes::prime_field_to_bytes_le_fixed::<BF0>(&pis[i]);
                out[8 * i..8 * i + 8].copy_from_slice(&bytes);
            }
            out
        };
        let committed_values_prefix_f257: [F257; CV_PREFIX_BYTES] =
            committed_values_prefix_bytes.map(|b| F257::from(b as u64));
        let stmt_digest_f257 = crate::we_statement::we_statement_hash_lf_plus::<RR>(
            vk_hash,
            committed_values_prefix_f257,
            r1cs_digest,
            gate_digest,
            &params,
        );
        let binding = WeStatementBindingWitness {
            vk_hash,
            r1cs_digest,
            gate_digest,
            committed_values_prefix_bytes,
        };
        let (shape, asg) = build_or_load_we_plus_tiny_dr1cs::<RR>(
            &trace,
            &params,
            &stmt_digest_f257,
            &binding,
            &proof,
            mlen_mats,
            &pairs,
            &out_dir,
        )
        .expect("build_or_load_we_plus_tiny_dr1cs");
        shape.inst.check(&asg).expect("baseline should satisfy");

        // Flip the first public input. Public prefix layout:
        // [ONE] || [stmt_digest(32)].
        let mut bad = asg.clone();
        let first_pi = 1usize;
        bad[first_pi] += F257::ONE;
        assert!(
            shape.inst.check(&bad).is_err(),
            "public input flip should break satisfaction (bound into transcript absorbs)"
        );
        let _ = std::fs::remove_dir_all(&out_dir);
    }

    #[test]
    #[ignore = "tiny-field gate coverage; run with `--release -- --ignored`"]
    fn test_we_plus_tiny_unsat_on_constraint_var_flip() {
        use cyclotomic_rings::rings::GoldilocksRing64 as RR;
        use stark_rings::PolyRing;

        let ring_dim = RR::dimension() as u64;
        // Prefix binding requires at least `min(8, kappa)` public inputs; here kappa=1.
        let public_inputs_len = 1usize;
        let n_lin_proofs = 1usize;
        let mlen_mats = 0usize;
        let pairs: Vec<(usize, usize)> = vec![(0, 0)];

        let params = WeParams {
            nvars_setchk: 12,
            degree_setchk: 3,
            nvars_cm: 12,
            degree_cm: 2,
            kappa: 1,
            ring_dim_d: ring_dim,
            decomp_b: 16,
            k: 1,
            l: 1,
            mlen: mlen_mats as u64,
        };

        let trace =
            poseidon_trace_schedule_for_plus::<RR>(public_inputs_len, &params, n_lin_proofs, mlen_mats)
                .expect("poseidon_trace_schedule_for_plus");
        let public_inputs_elems: Vec<F257> = vec![F257::ZERO; public_inputs_len];
        let _public_inputs_bytes: Vec<F257> = public_inputs_elems
            .iter()
            .flat_map(|_x| [0u8, 0, 0, 0, 0, 0, 0, 0].into_iter().map(|b| F257::from(b as u64)))
            .collect();

        let out_dir = {
            let mut p = std::env::temp_dir();
            p.push("lfplus_test_we_plus_tiny_var_flip");
            let _ = std::fs::remove_dir_all(&p);
            std::fs::create_dir_all(&p).expect("create temp out_dir");
            p
        };
        let proof = dummy_plus_proof_shape::<RR>(&params, mlen_mats, n_lin_proofs).expect("dummy_plus_proof_shape");
        let vk_hash = [1u8; 32];
        let r1cs_digest = [2u8; 32];
        let gate_digest = [3u8; 32];
        // This test uses an all-zero public input schedule, so the absorbed prefix bytes are zero.
        let committed_values_prefix_bytes = [0u8; CV_PREFIX_BYTES];
        let committed_values_prefix_f257: [F257; CV_PREFIX_BYTES] =
            committed_values_prefix_bytes.map(|b| F257::from(b as u64));
        let stmt_digest_f257 = crate::we_statement::we_statement_hash_lf_plus::<RR>(
            vk_hash,
            committed_values_prefix_f257,
            r1cs_digest,
            gate_digest,
            &params,
        );
        let binding = WeStatementBindingWitness {
            vk_hash,
            r1cs_digest,
            gate_digest,
            committed_values_prefix_bytes,
        };
        let (shape, asg) = build_or_load_we_plus_tiny_dr1cs::<RR>(
            &trace,
            &params,
            &stmt_digest_f257,
            &binding,
            &proof,
            mlen_mats,
            &pairs,
            &out_dir,
        )
        .expect("build_or_load_we_plus_tiny_dr1cs");
        shape.inst.check(&asg).expect("baseline should satisfy");

        // Perturb a non-public variable (the first witness slot after the public prefix).
        let pub_len = shape.public_len;
        let v = pub_len;
        assert!(v < asg.len(), "expected v within assignment");

        let mut bad = asg.clone();
        bad[v] += F257::ONE;
        assert!(
            shape.inst.check(&bad).is_err(),
            "flipping constrained non-public var should break satisfaction"
        );
        let _ = std::fs::remove_dir_all(&out_dir);
    }

    #[test]
    #[ignore = "slow: builds full Π_plus transcript schedule into Poseidon(F257) dR1CS"]
    fn test_plus_poseidon_schedule_lifts_to_f257_and_satisfies() {
        use cyclotomic_rings::rings::GoldilocksRing64 as RR;
        use stark_rings::PolyRing;
        use symphony::dpp_poseidon::poseidon_sponge_dr1cs_from_ops_with_wiring_no_bytes;

        // Small-ish params; we only care about schedule validity.
        let params = WeParams {
            nvars_setchk: 4,
            degree_setchk: 3,
            nvars_cm: 4,
            degree_cm: 2,
            kappa: 2,
            ring_dim_d: RR::dimension() as u64,
            decomp_b: 16,
            k: 1,
            l: 4,
            mlen: 1,
        };
        let public_inputs_len = 4usize;
        let n_lin_proofs = 1usize;
        let mlen_mats = 1usize;

        let trace =
            poseidon_trace_schedule_for_plus::<RR>(public_inputs_len, &params, n_lin_proofs, mlen_mats)
                .expect("poseidon_trace_schedule_for_plus");
        let ops_f257 = crate::we_gate_tiny::lift_recording_trace_ops_to_f257::<BF<RR>>(&trace.ops)
            .expect("lift trace ops to f257");

        let cfg = latticefold::transcript::poseidon::f257_poseidon_config();
        let (inst, asg, _wiring) =
            poseidon_sponge_dr1cs_from_ops_with_wiring_no_bytes::<latticefold::transcript::poseidon::F257>(
                &cfg,
                &ops_f257,
            )
            .expect("poseidon f257 no-bytes");
        inst.check(&asg).expect("poseidon f257 schedule satisfiable (no-bytes)");
    }

    // Large-scale (n=2^20) end-to-end WE->DPP sanity + digest "before/after" metric.
    // Run with: cargo test --release -p latticefold-plus test_large_trace --features we_gate -- --nocapture --ignored
    #[test]
    #[ignore]
    fn test_large_trace() {
        use cyclotomic_rings::rings::GoldilocksPoseidonConfig as PCF;
        use cyclotomic_rings::rings::GetPoseidonParams;
        use sha2::{Digest, Sha256};
        use cyclotomic_rings::rings::GoldilocksRing64 as RR;
        use stark_rings::PolyRing;

        use rand::RngCore;
        #[cfg(feature = "parallel")]
        use rayon::current_num_threads;



        // Default to the historical large-scale setting (n=2^20), but allow smaller for scaling studies.
        let n_pow: usize = std::env::var("LFP_TRACE_NPOW")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(20);
        let kappa = 2usize;
        let ell = 32usize;
        let d = RR::dimension();
        let b = (d / 2) as u128;
        // Match SP1 oneproof-style parameter choice:
        // - digit base is d' = d/2
        // - choose the *minimal* k to cover centered BabyBear values, then round up to pow2.
        //
        // Allow overriding p_bb and/or k for experiments.
        let p_bb: u64 = std::env::var("LFP_TRACE_PBB")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(2013265921); // BabyBear prime
        let d_prime_u64: u64 = (d / 2) as u64;
        fn next_power_of_two_u64(n: u64) -> u64 {
            if n <= 1 {
                return 1;
            }
            1u64 << (64 - (n - 1).leading_zeros())
        }
        fn min_k_for_bound(base: u64, bound: u64) -> u64 {
            debug_assert!(base >= 2 && base % 2 == 0);
            if bound == 0 {
                return 1;
            }
            let b = base as u128;
            let half = (base / 2) as u128;
            let target = bound as u128;
            let mut k: u64 = 1;
            let mut pow: u128 = b; // b^1
            loop {
                let max = half.saturating_mul(pow.saturating_sub(1) / (b - 1));
                if max >= target {
                    return k;
                }
                k += 1;
                pow = pow.saturating_mul(b);
            }
        }
        let k: usize = std::env::var("LFP_TRACE_K")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or_else(|| {
                let k_raw = min_k_for_bound(d_prime_u64, p_bb / 2);
                next_power_of_two_u64(k_raw) as usize
            });
        let dparams = DecompParameters { b, k, l: ell };

        // IMPORTANT: `RgInstance::from_f0_seeded` requires `n >= tau_unpadded_len` for `split`.
        // Keep the caller-provided `2^n_pow` as a *minimum*, but bump up if k/d/ell/kappa demand it.
        let n_min: usize = 1usize << n_pow;
        let tau_unpadded_len: usize = kappa * (k * d) * ell * d;
        let n: usize = n_min.max(tau_unpadded_len).next_power_of_two();
        let nvars = ark_std::log2(n) as usize;

        type BR = <RR as PolyRing>::BaseRing;
        type FSmall = <BR as ark_ff::Field>::BasePrimeField;
        let sp1_digest_bits: Vec<FSmall> = {
            let d: [u8; 32] = Sha256::digest(b"LFP_SP1_PUBLIC_INPUT_DIGEST_V1").into();
            crate::we_statement::digest32_to_bits_field::<FSmall>(d)
        };

        let run_one = |label: &str, sp1_digest_bits: &[FSmall]| {
            eprintln!("\n[test_large_trace] case={label} n=2^{n_pow} (sparse-base prover, GoldilocksRing64)");
            #[cfg(feature = "parallel")]
            eprintln!("[test_large_trace] rayon_threads={}", current_num_threads());
            #[cfg(not(feature = "parallel"))]
            eprintln!("[test_large_trace] rayon_threads=DISABLED(feature=parallel)");
            let mut rng = ark_std::test_rng();
            use crate::lin::LinParameters;
            use crate::plus::{PlusParameters, PlusProverSparseBase};
            use crate::r1cs::ComR1CSBase;
            use crate::utils::estimate_bound;
            use latticefold::arith::r1cs::R1CS;
            use latticefold::commitment::AjtaiCommitmentScheme;
            use std::sync::Arc;

            // A minimal Π_lin component so the transcript prefix is exercised.
            // Use a conservative bound for gadget decomposition; this is a *bench harness*, not a tight bound.
            let sop = RR::dimension() * 128;
            // IMPORTANT: balanced decomposition requires an even base.
            // `estimate_bound(..)+1` can be odd, so force it to be even.
            let mut b_bound = estimate_bound(sop, 1, d, k) + 1;
            if b_bound % 2 == 1 {
                b_bound += 1;
            }
            // Seeded Ajtai scheme (deterministic system parameter) with **prefix exposure**
            // for statement binding in the sparse/SP1 path.
            //
            // NOTE: This large-trace benchmark defaults to `kappa=2` for cost reasons; in that
            // configuration we can only expose/bind `min(kappa, 8)` coordinates.
            let expose_rows: usize = 8usize.min(kappa);
            let expose_col_offset: usize = 1; // witness[0] is the shared ONE=1
            const AJTAI_SEED: [u8; 32] = [7u8; 32];
            let scheme = AjtaiCommitmentScheme::<RR>::seeded_with_exposed_prefix(
                b"lf_plus_ajtai",
                AJTAI_SEED,
                kappa,
                n,
                expose_rows,
                expose_col_offset,
            );

            // Satisfiable const-coeff R1CS (base ring):
            // Use identity A=B=C so constraints are z_i^2 - z_i = 0, satisfied by boolean witness.
            let r1cs0 = R1CS::<BR> {
                l: 0,
                A: SparseMatrix::identity(n),
                B: SparseMatrix::identity(n),
                C: SparseMatrix::identity(n),
            };
            let bind_prefix = sp1_digest_bits
                .get(0..expose_rows)
                .expect("sp1_digest_bits shorter than expose_rows");
            let f0: Arc<Vec<BR>> = Arc::new(
                (0..n)
                    .map(|i| {
                        if i == 0 {
                            BR::ONE
                        } else if (expose_col_offset..expose_col_offset + expose_rows).contains(&i) {
                            // Make the exposed witness prefix equal the statement public inputs.
                            bind_prefix[i - expose_col_offset]
                        } else {
                            BR::from((rng.next_u64() & 1) as u64)
                        }
                    })
                    .collect(),
            );
            let cr1cs = ComR1CSBase::<RR>::from_f0_seeded_base(r1cs0, f0, 0, &scheme);
            let m0 = cr1cs.x.matrices_arc_base();

            let lin_params = LinParameters {
                kappa,
                decomp: dparams.clone(),
            };
            let pparams = PlusParameters { lin: lin_params, B: b_bound };

            let t0 = std::time::Instant::now();
            let transcript = crate::transcript::PoseidonTranscript::empty::<PCF>();
            let mut prover = PlusProverSparseBase::init_seeded_base(
                scheme.clone(),
                m0.clone(),
                1,
                pparams.clone(),
                transcript,
            );
            let proof = prover.prove_sparse_base(std::slice::from_ref(&cr1cs), &sp1_digest_bits);
            eprintln!("[test_large_trace] plus.prove: {:?}", t0.elapsed());

            let t1 = std::time::Instant::now();
            let mut rec = TracePoseidonTranscript::<RR>::empty::<PCF>();
            for b in sp1_digest_bits {
                rec.absorb_field_element(b);
            }
            // Mirror PlusVerifier::verify to record the full verifier trace.
            for lp in &proof.lproof {
                lp.verify(&mut rec);
            }
            proof
                .cmproof
                .verify_with_mlen(
                    m0.len(),
                    &mut rec,
                    bind_prefix,
                )
                .expect("cm verify");
            let trace = rec.into_trace();
            eprintln!("[test_large_trace] plus.verify(record): {:?}", t1.elapsed());

            let params = WeParams {
                nvars_setchk: nvars as u64,
                degree_setchk: 3,
                nvars_cm: nvars as u64,
                degree_cm: 2,
                kappa: kappa as u64,
                ring_dim_d: RR::dimension() as u64,
                decomp_b: b as u64,
                k: k as u64,
                l: ell as u64,
                mlen: m0.len() as u64,
            };
            let _poseidon_cfg = PCF::get_poseidon_config();

            // NOTE: this benchmark is now tiny-gate focused; we intentionally skip building the
            // large-field WE gate here.

            // -------------------------------------------------------------
            // Tiny gate (Poseidon(F257) + digit-domain verifier math)
            // -------------------------------------------------------------
            //
            // Build the full tiny gate from the **real proof trace** and check satisfaction.
            // This gives the closest estimate of the eventual F257 verifier-gate size.
            let _t_tiny = std::time::Instant::now();
            let ring_dim = RR::dimension();
            if ring_dim != 64 {
                panic!("test_large_trace: tiny gate only supports ring_dim=64");
            }
            let pairs: Vec<(usize, usize)> = vec![(0, 0)]; // minimal surface exercise

            // Fixed temp dir (no pid/seq). Best-effort cleanup before/after so users don't have to
            // manually delete large file-backed artifacts.
            use crate::fs_cleanup::fast_remove_dir_best_effort;

            // If set, keep `/tmp/lfplus_tiny_gate` between runs to enable cache hits (skips Pass1).
            // This is useful when iterating on unrelated code and you want stable runtime.
            let keep_tiny_cache = std::env::var("LFP_KEEP_TINY_GATE_CACHE")
                .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
                .unwrap_or(false);

            let out_dir = {
                let mut p = std::env::temp_dir();
                p.push("lfplus_tiny_gate");
                // If a previous run crashed or was interrupted, ensure a clean slate (unless caching).
                if !keep_tiny_cache {
                    fast_remove_dir_best_effort(&p);
                }
                std::fs::create_dir_all(&p).expect("create temp out_dir");
                p
            };
            let vk_hash = [1u8; 32];
            let r1cs_digest = [2u8; 32];
            let gate_digest = [3u8; 32];
            let committed_values_prefix_bytes: [u8; CV_PREFIX_BYTES] = {
                let mut out = [0u8; CV_PREFIX_BYTES];
                let pis = sp1_digest_bits
                    .get(0..8)
                    .expect("test requires at least 8 public inputs for committed-values prefix");
                for i in 0..8usize {
                    let bytes = latticefold::transcript::bytes::prime_field_to_bytes_le_fixed::<FSmall>(&pis[i]);
                    out[8 * i..8 * i + 8].copy_from_slice(&bytes);
                }
                out
            };
            let committed_values_prefix_f257: [F257; CV_PREFIX_BYTES] =
                committed_values_prefix_bytes.map(|b| F257::from(b as u64));
            let stmt_digest_f257 = crate::we_statement::we_statement_hash_lf_plus::<RR>(
                vk_hash,
                committed_values_prefix_f257,
                r1cs_digest,
                gate_digest,
                &params,
            );
            let binding = WeStatementBindingWitness {
                vk_hash,
                r1cs_digest,
                gate_digest,
                committed_values_prefix_bytes,
            };
            let (shape, tiny_asg) = build_or_load_we_plus_tiny_dr1cs::<RR>(
                &trace,
                &params,
                &stmt_digest_f257,
                &binding,
                &proof,
                m0.len(),
                &pairs,
                &out_dir,
            )
            .expect("build_or_load tiny gate (inst+asg) from proof");

            shape.inst.check(&tiny_asg).expect("tiny gate dr1cs check");
            if !keep_tiny_cache {
                fast_remove_dir_best_effort(&out_dir);
            }
        };

        run_one("sha256->bits", &sp1_digest_bits);
    }

}

