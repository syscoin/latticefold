//! WE/DPP gate arithmetization helpers (prime-field sparse dR1CS).
//!
//! This module is the *reusable wrapper* around the individual arithmetizers:
//! - Poseidon transcript trace -> sparse dR1CS (`dpp_poseidon`)
//! - AjtaiOpen(commitment, message) -> sparse dR1CS (`dpp_ajtai`)
//! - PCS evaluation verification -> sparse dR1CS (`dpp_pcs`)
//! - Merging multiple sparse dR1CS instances into one (`merge_sparse_dr1cs_share_one`)
//!
//! ## Architecture
//!
//! The WE gate (`R_WE`) is a single-phase design with the following components:
//!
//! - **Poseidon-FS transcript binding**: Ensures all FS challenges are bound to the protocol transcript.
//! - **Ajtai opening checks**: Verifies CP transcript message commitments (`cfs_*`).
//! - **Π_fold verifier math**: Sumcheck verification, monomial checks, and folding equations.
//! - **PCS evaluation proofs**: Binding aux values to the committed witness via lattice-based PCS.
//!
//! With PCS, there is no separate "π_lin" phase. The evaluation claims `aux = f(r)` are proven
//! directly by the PCS within the single WE gate verification.

use ark_crypto_primitives::sponge::poseidon::PoseidonConfig;
use ark_ff::Field;
use ark_std::log2;
use latticefold::commitment::AjtaiCommitmentScheme;
use latticefold::transcript::Transcript;
use latticefold::utils::sumcheck::MLSumcheck;
use rayon::prelude::*;
use stark_rings::{balanced_decomposition::Decompose, OverField, PolyRing, Ring, Zq};

use crate::dpp_ajtai::{ajtai_open_dr1cs_from_scheme, ajtai_open_dr1cs_from_scheme_full};
use crate::dpp_poseidon::{
    merge_sparse_dr1cs_share_one,
    merge_sparse_dr1cs_share_one_with_glue,
    poseidon_sponge_dr1cs_from_trace,
    poseidon_sponge_dr1cs_from_trace_with_wiring,
    SparseDr1csInstance,
};
use crate::dpp_pifold_math::pifold_verifier_math_dr1cs;
use crate::public_coin_transcript::FixedTranscript;
use crate::symphony_coins::{derive_beta_chi, derive_J};
use crate::symphony_pifold_batched::{PiFoldAuxWitness, PiFoldBatchedProof};
use crate::symphony_pifold_streaming::{CM_G_AGG_SEED, MON_B_AGG_SEED};
use crate::transcript::PoseidonTraceOp;

pub struct WeGateDr1csBuilder;

fn to_bf<R: OverField>(x: R::BaseRing) -> <R::BaseRing as Field>::BasePrimeField
where
    R::BaseRing: Field,
{
    x.to_base_prime_field_elements()
        .into_iter()
        .next()
        .expect("to_bf expects extension degree 1")
}

impl WeGateDr1csBuilder {
    /// Build the current arithmetized **R_cp** fragment:
    /// - Poseidon transcript trace constraints (FS-in-gate)
    /// - AjtaiOpen for CP transcript-message commitments (`cfs_had_u`, `cfs_mon_b`)
    ///
    /// This is the part that enforces transcript binding and that “aux matches its CP commitments”.
    pub fn r_cp_poseidon_and_cfs_openings<R>(
        poseidon_cfg: &PoseidonConfig<<<R as PolyRing>::BaseRing as Field>::BasePrimeField>,
        ops: &[PoseidonTraceOp<<<R as PolyRing>::BaseRing as Field>::BasePrimeField>],
        scheme_had: &AjtaiCommitmentScheme<R>,
        scheme_mon: &AjtaiCommitmentScheme<R>,
        aux: &PiFoldAuxWitness<R>,
        cfs_had_u: &[Vec<R>],
        cfs_mon_b: &[Vec<R>],
    ) -> Result<
        (
            SparseDr1csInstance<<<R as PolyRing>::BaseRing as Field>::BasePrimeField>,
            Vec<<<R as PolyRing>::BaseRing as Field>::BasePrimeField>,
        ),
        String,
    >
    where
        R: OverField + Ring + PolyRing,
        R::BaseRing: Zq + Field,
    {
        // Poseidon trace -> dR1CS.
        let (poseidon_inst, poseidon_asg, _replay2, _byte_wit) =
            poseidon_sponge_dr1cs_from_trace::<<<R as PolyRing>::BaseRing as Field>::BasePrimeField>(
                poseidon_cfg,
                ops,
            )
            .map_err(|e| format!("poseidon_sponge_dr1cs_from_trace: {e}"))?;

        if aux.had_u.len() != cfs_had_u.len() || aux.mon_b.len() != cfs_mon_b.len() {
            return Err("WeGateDr1csBuilder: aux/cfs length mismatch".to_string());
        }
        let ell = cfs_had_u.len();
        if ell == 0 {
            return Err("WeGateDr1csBuilder: empty batch".to_string());
        }

        // Ajtai-open checks for cfs_* - built in parallel per instance.
        let instance_parts: Vec<_> = (0..ell)
            .into_par_iter()
            .map(|i| {
                let mut had_msg: Vec<R> = Vec::with_capacity(3 * R::dimension());
                for blk in 0..3 {
                    for &x in &aux.had_u[i][blk] {
                        had_msg.push(R::from(x));
                    }
                }
                let (inst_had, asg_had) =
                    ajtai_open_dr1cs_from_scheme::<R>(scheme_had, &had_msg, &cfs_had_u[i])
                        .expect("ajtai_open_dr1cs_from_scheme failed");
                let (inst_mon, asg_mon) =
                    ajtai_open_dr1cs_from_scheme_full::<R>(scheme_mon, &aux.mon_b[i], &cfs_mon_b[i])
                        .expect("ajtai_open_dr1cs_from_scheme_full failed");
                vec![(inst_had, asg_had), (inst_mon, asg_mon)]
            })
            .collect();

        let mut parts: Vec<(
            SparseDr1csInstance<<<R as PolyRing>::BaseRing as Field>::BasePrimeField>,
            Vec<<<R as PolyRing>::BaseRing as Field>::BasePrimeField>,
        )> = Vec::with_capacity(1 + 2 * ell);
        parts.push((poseidon_inst, poseidon_asg));
        for inst_part in instance_parts {
            parts.extend(inst_part);
        }

        merge_sparse_dr1cs_share_one(&parts)
    }

    /// Build the arithmetized **R_cp** fragment including:
    /// - Poseidon transcript trace constraints (FS-in-gate)
    /// - Π_fold verifier arithmetic constraints (sumcheck verify + Eq(26) + monomial recomputation + Step-5)
    /// - AjtaiOpen for CP transcript-message commitments (`cfs_had_u`, `cfs_mon_b`)
    ///
    /// Additionally, this **glues** Π_fold's challenge variables to Poseidon's `SqueezeField` variables
    /// (so the Π_fold math is bound to the Poseidon transcript).
    ///
    /// Note: `derive_beta_chi` and `derive_J` both use `SqueezeBytes`; we currently do **not**
    /// arithmetize byte-decomposition, so β/J are treated as external witness values here.
    pub fn r_cp_poseidon_pifold_math_and_cfs_openings<R>(
        poseidon_cfg: &PoseidonConfig<<<R as PolyRing>::BaseRing as Field>::BasePrimeField>,
        ops: &[PoseidonTraceOp<<<R as PolyRing>::BaseRing as Field>::BasePrimeField>],
        cms: &[Vec<R>],
        proof: &PiFoldBatchedProof<R>,
        scheme_had: &AjtaiCommitmentScheme<R>,
        scheme_mon: &AjtaiCommitmentScheme<R>,
        aux: &PiFoldAuxWitness<R>,
        cfs_had_u: &[Vec<R>],
        cfs_mon_b: &[Vec<R>],
    ) -> Result<
        (
            SparseDr1csInstance<<<R as PolyRing>::BaseRing as Field>::BasePrimeField>,
            Vec<<<R as PolyRing>::BaseRing as Field>::BasePrimeField>,
        ),
        String,
    >
    where
        R: OverField + Ring + PolyRing,
        R::BaseRing: Zq + Field + Decompose,
    {
        if cms.len() != aux.had_u.len()
            || cms.len() != aux.mon_b.len()
            || cms.len() != cfs_had_u.len()
            || cms.len() != cfs_mon_b.len()
        {
            return Err("WeGateDr1csBuilder: cms/aux/cfs length mismatch".to_string());
        }
        let ell = cms.len();
        if ell == 0 {
            return Err("WeGateDr1csBuilder: empty batch".to_string());
        }

        // Poseidon trace -> dR1CS (+ wiring with squeeze-field var indices).
        let (poseidon_inst, poseidon_asg, _replay2, _byte_wit, wiring) =
            poseidon_sponge_dr1cs_from_trace_with_wiring::<<<R as PolyRing>::BaseRing as Field>::BasePrimeField>(
                poseidon_cfg,
                ops,
            )
            .map_err(|e| format!("poseidon_sponge_dr1cs_from_trace_with_wiring: {e}"))?;

        // Ajtai-open checks for cfs_* - built in parallel per instance.
        let cfs_parts_nested: Vec<Vec<_>> = (0..ell)
            .into_par_iter()
            .map(|i| {
                let mut had_msg: Vec<R> = Vec::with_capacity(3 * R::dimension());
                for blk in 0..3 {
                    for &x in &aux.had_u[i][blk] {
                        had_msg.push(R::from(x));
                    }
                }
                let (inst_had, asg_had) =
                    ajtai_open_dr1cs_from_scheme::<R>(scheme_had, &had_msg, &cfs_had_u[i])
                        .expect("ajtai_open_dr1cs_from_scheme failed");
                let (inst_mon, asg_mon) =
                    ajtai_open_dr1cs_from_scheme_full::<R>(scheme_mon, &aux.mon_b[i], &cfs_mon_b[i])
                        .expect("ajtai_open_dr1cs_from_scheme_full failed");
                vec![(inst_had, asg_had), (inst_mon, asg_mon)]
            })
            .collect();

        let mut cfs_parts: Vec<(
            SparseDr1csInstance<<<R as PolyRing>::BaseRing as Field>::BasePrimeField>,
            Vec<<<R as PolyRing>::BaseRing as Field>::BasePrimeField>,
        )> = Vec::with_capacity(2 * ell);
        for inst_part in cfs_parts_nested {
            cfs_parts.extend(inst_part);
        }
        let (cfs_inst, cfs_asg) = merge_sparse_dr1cs_share_one(&cfs_parts)?;

        let mut parts: Vec<(
            SparseDr1csInstance<<<R as PolyRing>::BaseRing as Field>::BasePrimeField>,
            Vec<<<R as PolyRing>::BaseRing as Field>::BasePrimeField>,
        )> = Vec::new();
        parts.push((poseidon_inst, poseidon_asg));
        parts.push((cfs_inst, cfs_asg));

        // ---------------------------------------------------------------------
        // Ajtai-open check for cm_g aggregate: enforce c_agg = A_agg · cm_g_flat.
        // This binds the aggregate (absorbed into transcript) to the individual cm_g values.
        // ---------------------------------------------------------------------
        let kappa = if !proof.cm_g.is_empty() && !proof.cm_g[0].is_empty() {
            proof.cm_g[0][0].len()
        } else {
            return Err("WeGateDr1csBuilder: empty cm_g for aggregate".to_string());
        };

        // Flatten cm_g: concat_{inst, dig, j} cm_g[inst][dig][j]
        let mut cm_g_flat: Vec<R> = Vec::new();
        for inst in proof.cm_g.iter() {
            for dig in inst.iter() {
                cm_g_flat.extend(dig.iter().copied());
            }
        }
        let n_agg = cm_g_flat.len();

        // Create the aggregate scheme (same as used in compute_cm_g_aggregate).
        let cm_g_agg_scheme =
            AjtaiCommitmentScheme::<R>::seeded(b"cm_g_agg", CM_G_AGG_SEED, kappa, n_agg);

        // Compute the expected aggregate (verifier-side recomputation).
        let cm_g_agg = cm_g_agg_scheme
            .commit(&cm_g_flat)
            .map_err(|e| format!("WeGateDr1csBuilder: cm_g_agg commit failed: {e:?}"))?
            .as_ref()
            .to_vec();

        // Enforce: A_agg · cm_g_flat = cm_g_agg (linear constraints).
        let (cm_g_agg_inst, cm_g_agg_asg) =
            ajtai_open_dr1cs_from_scheme_full::<R>(&cm_g_agg_scheme, &cm_g_flat, &cm_g_agg)?;
        parts.push((cm_g_agg_inst, cm_g_agg_asg));

        // Flatten mon_b: concat_{inst, dig} mon_b[inst][dig]
        let mut mon_b_flat: Vec<R> = Vec::new();
        for inst in aux.mon_b.iter() {
            mon_b_flat.extend(inst.iter().copied());
        }
        let n_mon_b_agg = mon_b_flat.len();

        // Create the aggregate scheme for mon_b (same as used in compute_mon_b_aggregate).
        let mon_b_agg_scheme =
            AjtaiCommitmentScheme::<R>::seeded(b"mon_b_agg", MON_B_AGG_SEED, kappa, n_mon_b_agg);

        // Compute the expected aggregate (verifier-side recomputation).
        let mon_b_agg = mon_b_agg_scheme
            .commit(&mon_b_flat)
            .map_err(|e| format!("WeGateDr1csBuilder: mon_b_agg commit failed: {e:?}"))?
            .as_ref()
            .to_vec();

        // Enforce: A_agg · mon_b_flat = mon_b_agg (linear constraints).
        let (mon_b_agg_inst, mon_b_agg_asg) =
            ajtai_open_dr1cs_from_scheme_full::<R>(&mon_b_agg_scheme, &mon_b_flat, &mon_b_agg)?;
        parts.push((mon_b_agg_inst, mon_b_agg_asg));

        // ---------------------------------------------------------------------
        // Extract the Π_fold verifier coin pieces from the proof's recorded coin stream
        // and compute rs_shared via sumcheck verification replay.
        // ---------------------------------------------------------------------
        let mut ts = FixedTranscript::<R>::new_with_coins_and_events(
            proof.coins.challenges.clone(),
            proof.coins.bytes.clone(),
            proof.coins.events.clone(),
        );

        // NOTE: FixedTranscript absorb is a no-op; this is purely to replay the same coin schedule.
        let beta_cts = derive_beta_chi::<R>(&mut ts, ell);
        let beta_cts_bf = beta_cts.iter().copied().map(to_bf::<R>).collect::<Vec<_>>();

        let m = proof.m;
        let log_m = log2(m) as usize;
        ts.absorb_field_element(&R::BaseRing::from(0x4c465053_50494841u128)); // "LFPS_PIHA"
        let s_base = ts.get_challenges(log_m);
        let s_base_bf = s_base.iter().copied().map(to_bf::<R>).collect::<Vec<_>>();
        let alpha_base_bf = to_bf::<R>(ts.get_challenge());

        let d = R::dimension();
        let g_len = m * d;
        let g_nvars = log2(g_len) as usize;
        let k_g = proof.rg_params.k_g;

        let mut cba_all_bf: Vec<Vec<(Vec<<R::BaseRing as Field>::BasePrimeField>, _, _)>> =
            Vec::with_capacity(ell);
        let mut rc_all_bf: Vec<Option<<R::BaseRing as Field>::BasePrimeField>> =
            Vec::with_capacity(ell);

        // Phase 1: absorb cm_f and derive J for each instance.
        for cm_f in cms {
            ts.absorb_slice(cm_f);
            let _j = derive_J::<R>(&mut ts, proof.rg_params.lambda_pj, proof.rg_params.l_h);
        }

        // Aggregate cm_g absorption (reuse cm_g_agg computed above for Ajtai check).
        ts.absorb_slice(&cm_g_agg);

        // Phase 2: derive coins for each instance.
        for _ in 0..ell {
            let mut cba = Vec::with_capacity(k_g);
            for _dig in 0..k_g {
                let c = ts
                    .get_challenges(g_nvars)
                    .into_iter()
                    .map(to_bf::<R>)
                    .collect::<Vec<_>>();
                let beta = to_bf::<R>(ts.get_challenge());
                let alpha = to_bf::<R>(ts.get_challenge());
                cba.push((c, beta, alpha));
            }
            let rc = (k_g > 1).then(|| to_bf::<R>(ts.get_challenge()));
            cba_all_bf.push(cba);
            rc_all_bf.push(rc);
        }
        let rhos_bf = ts
            .get_challenges(ell)
            .into_iter()
            .map(to_bf::<R>)
            .collect::<Vec<_>>();

        // Sumcheck randomness (shared): replay verifier schedule to get `mon_sc.point`.
        let hook_round = log2(proof.m_j.next_power_of_two()) as usize;
        let (_had_sc, mon_sc) = MLSumcheck::<R, _>::verify_two_as_subprotocol_shared_with_hook(
            &mut ts,
            log_m,
            3,
            R::ZERO,
            &proof.had_sumcheck,
            g_nvars,
            3,
            R::ZERO,
            &proof.mon_sumcheck,
            hook_round,
            |t, _sampled| {
                for v_i in &proof.v_digits_folded {
                    for x in v_i {
                        t.absorb_field_element(x);
                    }
                }
            },
        )
        .map_err(|e| format!("WeGateDr1csBuilder: sumcheck verify replay failed: {e}"))?;

        let rs_shared_bf = mon_sc
            .point
            .iter()
            .copied()
            .map(to_bf::<R>)
            .collect::<Vec<_>>();

        // Ensure we consumed the entire provided coin stream.
        if ts.remaining_challenges() != 0 || ts.remaining_bytes() != 0 || ts.remaining_events() != 0 {
            return Err(format!(
                "WeGateDr1csBuilder: leftover coins after replay: challenges={} bytes={} events={}",
                ts.remaining_challenges(),
                ts.remaining_bytes(),
                ts.remaining_events()
            ));
        }

        // Build Π_fold verifier math dR1CS.
        let (pifold_inst, pifold_asg, pifold_wiring) = pifold_verifier_math_dr1cs::<R>(
            proof,
            aux,
            &beta_cts_bf,
            &s_base_bf,
            alpha_base_bf,
            &cba_all_bf,
            &rc_all_bf,
            &rhos_bf,
            &rs_shared_bf,
        )?;
        parts.push((pifold_inst, pifold_asg));

        // ---------------------------------------------------------------------
        // Glue Π_fold challenge variables to Poseidon squeeze-field variables.
        // ---------------------------------------------------------------------
        // Expected number of `get_challenge` outputs for Π_fold verifier schedule:
        let per_inst = k_g * (g_nvars + 2) + if k_g > 1 { 1 } else { 0 };
        let total_challenges = log_m + 1 + ell * per_inst + ell + g_nvars;
        if wiring.squeeze_field_vars.len() != total_challenges {
            return Err(format!(
                "WeGateDr1csBuilder: poseidon wiring squeeze_field_vars len mismatch: got {} expected {}",
                wiring.squeeze_field_vars.len(),
                total_challenges
            ));
        }

        // Build glue list in local indices: (part_a, var_a, part_b, var_b).
        // part 0 = poseidon, part 1 = cfs-openings, part 2 = cm_g_agg, part 3 = mon_b_agg, part 4 = pifold-math
        const PIFOLD_PART: usize = 4;
        let mut glue: Vec<(usize, usize, usize, usize)> = Vec::new();

        let mut ch = 0usize;
        // s_base
        for (j, &v) in pifold_wiring.s_base.iter().enumerate() {
            let _ = j;
            glue.push((PIFOLD_PART, v, 0, wiring.squeeze_field_vars[ch]));
            ch += 1;
        }
        // alpha_base
        glue.push((PIFOLD_PART, pifold_wiring.alpha_base, 0, wiring.squeeze_field_vars[ch]));
        ch += 1;
        // per-instance per-digit: c vector, beta, alpha, optional rc
        for inst_idx in 0..ell {
            for dig in 0..k_g {
                let idx_flat = inst_idx * k_g + dig;
                for &cv in &pifold_wiring.c_all[idx_flat] {
                    glue.push((PIFOLD_PART, cv, 0, wiring.squeeze_field_vars[ch]));
                    ch += 1;
                }
                glue.push((PIFOLD_PART, pifold_wiring.beta_i_all[idx_flat], 0, wiring.squeeze_field_vars[ch]));
                ch += 1;
                glue.push((PIFOLD_PART, pifold_wiring.alpha_i_all[idx_flat], 0, wiring.squeeze_field_vars[ch]));
                ch += 1;
            }
            if let Some(rcv) = pifold_wiring.rc_all[inst_idx] {
                glue.push((PIFOLD_PART, rcv, 0, wiring.squeeze_field_vars[ch]));
                ch += 1;
            }
        }
        // rhos
        for &rv in &pifold_wiring.rhos {
            glue.push((PIFOLD_PART, rv, 0, wiring.squeeze_field_vars[ch]));
            ch += 1;
        }
        // rs_shared
        for &rv in &pifold_wiring.rs_shared {
            glue.push((PIFOLD_PART, rv, 0, wiring.squeeze_field_vars[ch]));
            ch += 1;
        }
        debug_assert_eq!(ch, total_challenges);

        merge_sparse_dr1cs_share_one_with_glue(&parts, &glue)
    }

    /// Build and merge:
    /// - Poseidon sponge dR1CS for `ops`
    /// - Ajtai-open dR1CS for all `cfs_had_u[i]` and `cfs_mon_b[i]` openings to `aux`
    ///
    /// Returns a single merged `(inst, assignment)` sharing a single constant-1 slot.
    pub fn poseidon_plus_cfs_openings<R>(
        poseidon_cfg: &PoseidonConfig<<<R as PolyRing>::BaseRing as Field>::BasePrimeField>,
        ops: &[PoseidonTraceOp<<<R as PolyRing>::BaseRing as Field>::BasePrimeField>],
        scheme_had: &AjtaiCommitmentScheme<R>,
        scheme_mon: &AjtaiCommitmentScheme<R>,
        aux: &PiFoldAuxWitness<R>,
        cfs_had_u: &[Vec<R>],
        cfs_mon_b: &[Vec<R>],
    ) -> Result<
        (
            SparseDr1csInstance<<<R as PolyRing>::BaseRing as Field>::BasePrimeField>,
            Vec<<<R as PolyRing>::BaseRing as Field>::BasePrimeField>,
        ),
        String,
    >
    where
        R: OverField + Ring,
        R: PolyRing,
        R::BaseRing: Zq + Field,
    {
        Self::r_cp_poseidon_and_cfs_openings::<R>(
            poseidon_cfg, ops, scheme_had, scheme_mon, aux, cfs_had_u, cfs_mon_b,
        )
    }

    /// Build a bring-up **R_o** fragment that binds witness openings `f` to `cm_f` via AjtaiOpen.
    ///
    /// This is *not* the production reduced relation; it is a correctness baseline.
    pub fn r_o_cm_f_openings<R>(
        scheme_f: &AjtaiCommitmentScheme<R>,
        cm_f: &[Vec<R>],
        f_openings: &[Vec<R>],
    ) -> Result<
        (
            SparseDr1csInstance<<<R as PolyRing>::BaseRing as Field>::BasePrimeField>,
            Vec<<<R as PolyRing>::BaseRing as Field>::BasePrimeField>,
        ),
        String,
    >
    where
        R: OverField + Ring + PolyRing,
        R::BaseRing: Zq + Field,
    {
        if cm_f.len() != f_openings.len() {
            return Err("WeGateDr1csBuilder: cm_f/f_openings length mismatch".to_string());
        }
        if cm_f.is_empty() {
            return Err("WeGateDr1csBuilder: empty batch".to_string());
        }

        let mut parts: Vec<(
            SparseDr1csInstance<<<R as PolyRing>::BaseRing as Field>::BasePrimeField>,
            Vec<<<R as PolyRing>::BaseRing as Field>::BasePrimeField>,
        )> = Vec::with_capacity(cm_f.len());

        for i in 0..cm_f.len() {
            let (inst_f, asg_f) =
                ajtai_open_dr1cs_from_scheme_full::<R>(scheme_f, &f_openings[i], &cm_f[i])?;
            parts.push((inst_f, asg_f));
        }
        merge_sparse_dr1cs_share_one(&parts)
    }

    /// Bring-up binding strategy: additionally enforce `AjtaiOpen(cm_f[i], f[i])` inside the merged system.
    ///
    /// WARNING: this is **not** the production Architecture‑T shape for SP1-sized witnesses.
    /// It is only intended for small tests / scaffolding, because arithmetizing AjtaiOpen over a large
    /// witness would require baking a huge dense linear system (one constraint per commitment row × ring lane,
    /// with width proportional to the witness length).
    pub fn poseidon_plus_cfs_plus_cm_f_openings<R>(
        poseidon_cfg: &PoseidonConfig<<<R as PolyRing>::BaseRing as Field>::BasePrimeField>,
        ops: &[PoseidonTraceOp<<<R as PolyRing>::BaseRing as Field>::BasePrimeField>],
        scheme_f: &AjtaiCommitmentScheme<R>,
        cm_f: &[Vec<R>],
        f_openings: &[Vec<R>],
        scheme_had: &AjtaiCommitmentScheme<R>,
        scheme_mon: &AjtaiCommitmentScheme<R>,
        aux: &PiFoldAuxWitness<R>,
        cfs_had_u: &[Vec<R>],
        cfs_mon_b: &[Vec<R>],
    ) -> Result<
        (
            SparseDr1csInstance<<<R as PolyRing>::BaseRing as Field>::BasePrimeField>,
            Vec<<<R as PolyRing>::BaseRing as Field>::BasePrimeField>,
        ),
        String,
    >
    where
        R: OverField + Ring + PolyRing,
        R::BaseRing: Zq + Field,
    {
        let (rcp_inst, rcp_asg) = Self::r_cp_poseidon_and_cfs_openings::<R>(
            poseidon_cfg, ops, scheme_had, scheme_mon, aux, cfs_had_u, cfs_mon_b,
        )?;
        let (ro_inst, ro_asg) = Self::r_o_cm_f_openings::<R>(scheme_f, cm_f, f_openings)?;
        merge_sparse_dr1cs_share_one(&[(rcp_inst, rcp_asg), (ro_inst, ro_asg)])
    }

    // PCS evaluation claims:
    // 1. Build PCS verification constraints using `dpp_pcs::pcs_verify_all_digits`
    // 2. Merge with the result of `r_cp_poseidon_pifold_math_and_cfs_openings`
}

