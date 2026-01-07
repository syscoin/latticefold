//! Symphony Π_fold (Figure 4) — **verifier** functions and data types.
//!
//! This module provides:
//! - Proof and output data types used by both prover and verifier
//! - Verifier functions for batched Π_fold proofs (shared-M and hetero-M variants)
//!
//! For the prover, use `prove_pi_fold_poseidon_fs` from `symphony_pifold_streaming`.
//!
//! Notes:
//! - We still recompute some per-instance values during verification from witness openings
//!   (correctness-first; not ZK yet).
//! - Full paper compression replaces these with commitment+opening checks (`VfyOpen`).

use ark_std::log2;
use cyclotomic_rings::rings::GetPoseidonParams;
use latticefold::{
    transcript::Transcript,
    utils::sumcheck::{
        utils::eq_eval,
        MLSumcheck, Proof,
    },
};
use stark_rings::{
    balanced_decomposition::{Decompose, DecomposeToVec},
    exp,
    psi,
    CoeffRing,
    OverField,
    Ring,
    Zq,
};
use stark_rings_linalg::SparseMatrix;
use stark_rings_poly::mle::DenseMultilinearExtension;

use crate::{
    symphony_coins::{derive_beta_chi, derive_J, ev, ts_weights},
    symphony_open::{NoOpen, VfyOpen},
    public_coin_transcript::FixedTranscript,
    rp_rgchk::{compose_v_digits, RPParams},
    symphony_cm::SymphonyCoins,
    symphony_fold::{SymphonyBatchLin, SymphonyInstance},
    transcript::PoseidonTranscriptMetrics,
};



#[derive(Clone, Debug)]
pub struct PiFoldBatchedProof<R: OverField>
where
    R::BaseRing: Zq,
{
    pub coins: SymphonyCoins<R>,
    pub rg_params: RPParams,

    /// m_J for the projection (n*lambda_pj/l_h).
    pub m_j: usize,
    /// m used by Π_mon (row-domain size).
    pub m: usize,

    /// Folded Π_rg hook message v_digits* (k_g × d), using scalar β (constants in the base field).
    pub v_digits_folded: Vec<Vec<R::BaseRing>>,

    /// Batched sumcheck proofs.
    pub had_sumcheck: Proof<R>,
    pub mon_sumcheck: Proof<R>,
}

/// Canonical prover output for Π_fold in this crate.
///
/// - `proof`: the public proof object (what a verifier receives)
/// - `aux`: the prover-side transcript witness messages (`m_i`) that the WE/DPP-facing relation
///   treats as witness and binds via CP commitments (`cfs_*`).
#[derive(Clone, Debug)]
pub struct PiFoldProverOutput<R: OverField>
where
    R::BaseRing: Zq,
{
    pub proof: PiFoldBatchedProof<R>,
    pub aux: PiFoldAuxWitness<R>,
    /// Optional CP transcript-message commitments (paper `c_fs,i = Commit(m_i)`), computed from `aux`.
    ///
    /// These are only populated when the prover is given commitment schemes for these messages.
    pub cfs_had_u: Vec<Vec<R>>,
    pub cfs_mon_b: Vec<Vec<R>>,
}

/// Auxiliary witness messages for WE/DPP-facing relation checks.
///
/// In the paper’s CP relation, these are part of the hidden folding transcript messages `m_i`
/// that are committed to; here we plumb them explicitly so the verifier (and thus a DPP target)
/// does **not** need to recompute them from the full witness openings.
#[derive(Clone, Debug)]
pub struct PiFoldAuxWitness<R: OverField>
where
    R::BaseRing: Zq,
{
    /// Per-instance Π_had “U” message at the had-sumcheck evaluation point:
    /// for each instance i: U[i] = [U1(d), U2(d), U3(d)] where each is length `d = dim(R)`.
    pub had_u: Vec<[Vec<R::BaseRing>; 3]>,
    /// Per-instance Π_rg monomial message `b` at the mon-sumcheck evaluation point:
    /// for each instance i: b[i][dig] is a ring element for dig ∈ [0, k_g).
    pub mon_b: Vec<Vec<R>>,
}

fn encode_had_u_instance<R: OverField>(u: &[Vec<R::BaseRing>; 3]) -> Vec<R>
where
    R::BaseRing: Zq,
{
    let d = R::dimension();
    debug_assert_eq!(u[0].len(), d);
    debug_assert_eq!(u[1].len(), d);
    debug_assert_eq!(u[2].len(), d);
    let mut out = Vec::with_capacity(3 * d);
    out.extend(u[0].iter().copied().map(R::from));
    out.extend(u[1].iter().copied().map(R::from));
    out.extend(u[2].iter().copied().map(R::from));
    out
}

fn absorb_public_inputs<R: OverField>(ts: &mut impl Transcript<R>, public_inputs: &[R::BaseRing])
where
    R::BaseRing: Zq,
{
    // Domain separate the *statement binding* portion from the rest of the folding transcript.
    ts.absorb_field_element(&R::BaseRing::from(0x4c465053_5055424cu128)); // "LFPS_PUBL"
    ts.absorb_field_element(&R::BaseRing::from(public_inputs.len() as u128));
    for x in public_inputs {
        ts.absorb_field_element(x);
    }
}

/// Verifier: verify the batched sumchecks and per-instance Step-5 checks, then fold outputs.


/// Verifier (FS replay): verify under the explicit coin stream in `proof.coins`.
pub fn verify_pi_fold_batched_and_fold_outputs_fs<R: CoeffRing>(
    M: [&SparseMatrix<R>; 3],
    cms: &[Vec<R>],
    proof: &PiFoldBatchedProof<R>,
    cms_openings: &[Vec<R>],
) -> Result<(SymphonyInstance<R>, SymphonyBatchLin<R>), String>
where
    R::BaseRing: Zq + Decompose,
{
    let mut ts = FixedTranscript::<R>::new_with_coins_and_events(
        proof.coins.challenges.clone(),
        proof.coins.bytes.clone(),
        proof.coins.events.clone(),
    );
    ts.absorb_field_element(&R::BaseRing::from(0x4c465053_50494250u128)); // "LFPS_PIBP"
    // NOTE: this helper replays only the folding/sumcheck transcript.
    // Opening verification is intentionally orthogonal (CP-style) and must be done by calling
    // `verify_pi_fold_batched_and_fold_outputs_with_openings` with a concrete `VfyOpen`.
    let out = verify_pi_fold_batched_and_fold_outputs_with_openings(&mut ts, M, cms, proof, &NoOpen, cms_openings)?;
    if ts.remaining_challenges() != 0 || ts.remaining_bytes() != 0 || ts.remaining_events() != 0 {
        return Err("PiFold: coin stream not fully consumed".to_string());
    }
    Ok(out)
}

/// Verifier (FS replay) for a *heterogeneous-matrix* batched Π_fold proof.
///
/// This replays the explicit coin stream in `proof.coins` and is useful as a diagnostic:
/// if Poseidon-FS verification fails but FS-replay succeeds, the transcript schedule is mismatched.
pub fn verify_pi_fold_batched_and_fold_outputs_fs_hetero_m<R: CoeffRing>(
    Ms: &[[&SparseMatrix<R>; 3]],
    cms: &[Vec<R>],
    proof: &PiFoldBatchedProof<R>,
    cms_openings: &[Vec<R>],
    aux: Option<&PiFoldAuxWitness<R>>,
    public_inputs: &[R::BaseRing],
) -> Result<(SymphonyInstance<R>, SymphonyBatchLin<R>), String>
where
    R::BaseRing: Zq + Decompose,
{
    let mut ts = FixedTranscript::<R>::new_with_coins_and_events(
        proof.coins.challenges.clone(),
        proof.coins.bytes.clone(),
        proof.coins.events.clone(),
    );
    ts.absorb_field_element(&R::BaseRing::from(0x4c465053_50494250u128)); // "LFPS_PIBP"
    absorb_public_inputs::<R>(&mut ts, public_inputs);

    let out = verify_pi_fold_batched_and_fold_outputs_with_openings_and_aux_hetero_m(
        &mut ts,
        Ms,
        cms,
        proof,
        &NoOpen,
        cms_openings,
        aux,
    )?;
    if ts.remaining_challenges() != 0 || ts.remaining_bytes() != 0 || ts.remaining_events() != 0 {
        return Err("PiFold: coin stream not fully consumed".to_string());
    }
    Ok(out)
}

/// Verifier (Poseidon-FS): recompute all challenges by hashing the transcript (Poseidon sponge).
pub fn verify_pi_fold_batched_and_fold_outputs_poseidon_fs<R: CoeffRing, PC>(
    M: [&SparseMatrix<R>; 3],
    cms: &[Vec<R>],
    proof: &PiFoldBatchedProof<R>,
    open: &impl VfyOpen<R>,
    cms_openings: &[Vec<R>],
    public_inputs: &[R::BaseRing],
) -> Result<(SymphonyInstance<R>, SymphonyBatchLin<R>), String>
where
    R::BaseRing: Zq + Decompose,
    PC: GetPoseidonParams<<<R>::BaseRing as ark_ff::Field>::BasePrimeField>,
{
    let mut ts = crate::transcript::PoseidonTranscript::<R>::empty::<PC>();
    ts.absorb_field_element(&R::BaseRing::from(0x4c465053_50494250u128)); // "LFPS_PIBP"
    absorb_public_inputs::<R>(&mut ts, public_inputs);
    verify_pi_fold_batched_and_fold_outputs_with_openings_and_aux(
        &mut ts,
        M,
        cms,
        proof,
        open,
        cms_openings,
        None,
    )
}

/// Verifier (Poseidon-FS) for a *heterogeneous-matrix* batched Π_fold proof.
///
/// This is the O(1)-verification path for SP1 chunking: each instance has its own `(M1,M2,M3)`.
pub fn verify_pi_fold_batched_and_fold_outputs_poseidon_fs_hetero_m<R: CoeffRing, PC>(
    Ms: &[[&SparseMatrix<R>; 3]],
    cms: &[Vec<R>],
    proof: &PiFoldBatchedProof<R>,
    open: &impl VfyOpen<R>,
    cms_openings: &[Vec<R>],
    aux: Option<&PiFoldAuxWitness<R>>,
    public_inputs: &[R::BaseRing],
) -> Result<(SymphonyInstance<R>, SymphonyBatchLin<R>), String>
where
    R::BaseRing: Zq + Decompose,
    PC: GetPoseidonParams<<<R>::BaseRing as ark_ff::Field>::BasePrimeField>,
{
    let mut ts = crate::transcript::PoseidonTranscript::<R>::empty::<PC>();
    ts.absorb_field_element(&R::BaseRing::from(0x4c465053_50494250u128)); // "LFPS_PIBP"
    absorb_public_inputs::<R>(&mut ts, public_inputs);
    verify_pi_fold_batched_and_fold_outputs_with_openings_and_aux_hetero_m(
        &mut ts,
        Ms,
        cms,
        proof,
        open,
        cms_openings,
        aux,
    )
}

/// Heterogeneous-matrix verifier with optional auxiliary witness messages (`had_u`, `mon_b`).
pub fn verify_pi_fold_batched_and_fold_outputs_with_openings_and_aux_hetero_m<R: CoeffRing>(
    transcript: &mut impl Transcript<R>,
    Ms: &[[&SparseMatrix<R>; 3]],
    cms: &[Vec<R>],
    proof: &PiFoldBatchedProof<R>,
    open: &impl VfyOpen<R>,
    cms_openings: &[Vec<R>],
    aux: Option<&PiFoldAuxWitness<R>>,
) -> Result<(SymphonyInstance<R>, SymphonyBatchLin<R>), String>
where
    R::BaseRing: Zq + Decompose,
{
    if cms.len() != Ms.len() {
        return Err("PiFold: Ms/cms length mismatch".to_string());
    }
    if !cms_openings.is_empty() {
        if cms.len() != cms_openings.len() {
            return Err("PiFold: cms/openings length mismatch".to_string());
        }
        for (cm, open_val) in cms.iter().zip(cms_openings.iter()) {
            open.verify_opening(transcript, "cm_witness", cm, &[], open_val, &[])?;
        }
    } else if aux.is_none() {
        return Err("PiFold: cms_openings required when aux is not provided".to_string());
    }

    let ell = cms.len();
    if ell == 0 {
        return Err("PiFold: length mismatch".to_string());
    }
    let rg_params = &proof.rg_params;

    let beta_cts = derive_beta_chi::<R>(transcript, ell);
    let beta_ring = beta_cts.iter().copied().map(R::from).collect::<Vec<R>>();

    let m = Ms[0][0].nrows;
    if !m.is_power_of_two() {
        return Err("PiFold: m must be power-of-two".to_string());
    }
    for inst_idx in 0..ell {
        for i in 0..3 {
            if Ms[inst_idx][i].nrows != m {
                return Err("PiFold: inconsistent m across instances".to_string());
            }
        }
    }

    let log_m = log2(m) as usize;
    let d = R::dimension();

    transcript.absorb_field_element(&R::BaseRing::from(0x4c465053_50494841u128)); // "LFPS_PIHA"
    let s_base = transcript.get_challenges(log_m);
    let alpha_base = transcript.get_challenge();

    let s_poly: Vec<R> = s_base.iter().copied().map(R::from).collect();
    let m_j = proof.m_j;
    if m != proof.m {
        return Err("PiFold: proof m mismatch".to_string());
    }
    if m < m_j || m % m_j != 0 {
        return Err("PiFold: require m_J<=m".to_string());
    }

    let g_len = m * d;
    if !g_len.is_power_of_two() {
        return Err("PiFold: require m*d power-of-two".to_string());
    }
    let g_nvars = log2(g_len) as usize;

    let mut cba_all: Vec<Vec<(Vec<R>, R::BaseRing, R::BaseRing)>> = Vec::with_capacity(ell);
    let mut rc_all: Vec<Option<R::BaseRing>> = Vec::with_capacity(ell);
    let mut Js: Vec<Vec<Vec<R::BaseRing>>> = Vec::with_capacity(ell);

    let (_n_f, blocks) = if aux.is_none() {
        let n_f = cms_openings[0].len();
        if n_f == 0 || n_f % rg_params.l_h != 0 {
            return Err("PiFold: invalid witness length".to_string());
        }
        for w in cms_openings.iter() {
            if w.len() != n_f {
                return Err("PiFold: inconsistent witness lengths".to_string());
            }
        }
        let blocks = n_f / rg_params.l_h;
        let m_j_expected = blocks * rg_params.lambda_pj;
        if m_j_expected != m_j {
            return Err("PiFold: m_J mismatch".to_string());
        }
        // Sanity-check witness width matches matrices.
        for inst_idx in 0..ell {
            for i in 0..3 {
                if Ms[inst_idx][i].ncols != n_f {
                    return Err("PiFold: matrix/witness width mismatch".to_string());
                }
            }
        }
        (n_f, blocks)
    } else {
        (0usize, 0usize)
    };

    for cm_f in cms.iter() {
        transcript.absorb_slice(cm_f);
        let J = derive_J::<R>(transcript, rg_params.lambda_pj, rg_params.l_h);
        Js.push(J);

        let mut cba: Vec<(Vec<R>, R::BaseRing, R::BaseRing)> = Vec::with_capacity(rg_params.k_g);
        for _ in 0..rg_params.k_g {
            let c: Vec<R> = transcript
                .get_challenges(g_nvars)
                .into_iter()
                .map(|x| x.into())
                .collect();
            let beta = transcript.get_challenge();
            let alpha = transcript.get_challenge();
            cba.push((c, beta, alpha));
        }
        let rc: Option<R::BaseRing> = (rg_params.k_g > 1).then(|| transcript.get_challenge());
        cba_all.push(cba);
        rc_all.push(rc);
    }

    let rhos = transcript
        .get_challenges(ell)
        .into_iter()
        .map(R::from)
        .collect::<Vec<R>>();

    let hook_round = log2(m_j.next_power_of_two()) as usize;
    let (had_sc, mon_sc) = MLSumcheck::<R, _>::verify_two_as_subprotocol_shared_with_hook(
        transcript,
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
    .map_err(|e| format!("PiFold: sumcheck verify failed: {e}"))?;

    // Recompute the expected hadamard sumcheck evaluation at the had point.
    let r_had = had_sc.point.clone();
    let eq_sr = eq_eval(&s_poly, &r_had.iter().copied().map(R::from).collect::<Vec<_>>())
        .map_err(|e| format!("PiFold: eq_eval failed: {e}"))?;

    let mut pow = R::BaseRing::ONE;
    let mut alpha_pows = Vec::with_capacity(d);
    for _ in 0..d {
        alpha_pows.push(R::from(pow));
        pow *= alpha_base;
    }

    let mut lhs = R::ZERO;
    for inst_idx in 0..ell {
        let U: [Vec<R::BaseRing>; 3] = if let Some(auxw) = aux {
            if auxw.had_u.len() != ell {
                return Err("PiFold: aux.had_u length mismatch".to_string());
            }
            auxw.had_u[inst_idx].clone()
        } else {
            let y = Ms[inst_idx]
                .iter()
                .map(|Mi| Mi.try_mul_vec(&cms_openings[inst_idx]).expect("mat-vec mul failed"))
                .collect::<Vec<Vec<R>>>();

            let mut U: [Vec<R::BaseRing>; 3] =
                [Vec::with_capacity(d), Vec::with_capacity(d), Vec::with_capacity(d)];
            for i in 0..3 {
                for j in 0..d {
                    let evals = (0..m)
                        .map(|row| R::from(y[i][row].coeffs()[j]))
                        .collect::<Vec<_>>();
                    let mle = DenseMultilinearExtension::from_evaluations_vec(log_m, evals);
                    let v = mle
                        .evaluate(&r_had.iter().copied().map(R::from).collect::<Vec<_>>())
                        .expect("MLE evaluate returned None");
                    U[i].push(v.ct());
                }
            }
            U
        };

        if U[0].len() != d || U[1].len() != d || U[2].len() != d {
            return Err("PiFold: aux had_U has wrong dimension".to_string());
        }

        let mut acc = R::ZERO;
        for j in 0..d {
            let u1 = R::from(U[0][j]);
            let u2 = R::from(U[1][j]);
            let u3 = R::from(U[2][j]);
            acc += alpha_pows[j] * (u1 * u2 - u3);
        }
        lhs += rhos[inst_idx] * (eq_sr * acc);

        for x in &U[0] {
            transcript.absorb_field_element(x);
        }
        for x in &U[1] {
            transcript.absorb_field_element(x);
        }
        for x in &U[2] {
            transcript.absorb_field_element(x);
        }
    }
    if lhs != had_sc.expected_evaluation {
        return Err("PiFold: hadamard recomputation mismatch".to_string());
    }

    // -----------------
    // Verify the batched Π_mon linkage and folded Π_rg Step-5 check (does not depend on M entries).
    // -----------------
    let r_mon_r: Vec<R> = mon_sc.point.iter().copied().map(R::from).collect();

    let mon_b: Vec<Vec<R>> = if let Some(auxw) = aux {
        if auxw.mon_b.len() != ell {
            return Err("PiFold: aux.mon_b length mismatch".to_string());
        }
        for b_inst in auxw.mon_b.iter() {
            if b_inst.len() != rg_params.k_g {
                return Err("PiFold: aux.mon_b wrong k_g".to_string());
            }
        }
        for b_inst in auxw.mon_b.iter() {
            transcript.absorb_slice(b_inst);
        }
        auxw.mon_b.clone()
    } else {
        let mut mon_b: Vec<Vec<R>> = Vec::with_capacity(ell);
        let expand_row = |row: usize| -> usize { row % m_j };
        for inst_idx in 0..ell {
            // Build H_digits for this instance (on m_J rows).
            let f = &cms_openings[inst_idx];
            let mut H = vec![vec![R::BaseRing::ZERO; d]; m_j];
            let Jref = &Js[inst_idx];
            for b in 0..blocks {
                for i in 0..rg_params.lambda_pj {
                    let out_row = b * rg_params.lambda_pj + i;
                    for t in 0..rg_params.l_h {
                        let in_row = b * rg_params.l_h + t;
                        let coef = Jref[i][t];
                        for col in 0..d {
                            H[out_row][col] += coef * f[in_row].coeffs()[col];
                        }
                    }
                }
            }
            let mut H_digits: Vec<Vec<Vec<R::BaseRing>>> =
                vec![vec![vec![R::BaseRing::ZERO; d]; m_j]; rg_params.k_g];
            for r in 0..m_j {
                let row_digits = H[r].decompose_to_vec(rg_params.d_prime, rg_params.k_g);
                for c in 0..d {
                    for i in 0..rg_params.k_g {
                        H_digits[i][r][c] = row_digits[c][i];
                    }
                }
            }

            let mut b_inst = Vec::with_capacity(rg_params.k_g);
            for dig in 0..rg_params.k_g {
                let mut gi = Vec::with_capacity(g_len);
                for c in 0..d {
                    for r in 0..m {
                        gi.push(exp::<R>(H_digits[dig][expand_row(r)][c]).expect("Exp failed"));
                    }
                }
                let mle = DenseMultilinearExtension::from_evaluations_vec(g_nvars, gi);
                b_inst.push(mle.evaluate(&r_mon_r).unwrap());
            }
            transcript.absorb_slice(&b_inst);
            mon_b.push(b_inst);
        }
        mon_b
    };

    let v_expected = mon_sc.expected_evaluation;

    let mut ver = R::ZERO;
    for inst_idx in 0..ell {
        let mut inst_acc = R::ZERO;
        // Precompute rc^dig iteratively to avoid repeated pow().
        let mut rc_pow = R::BaseRing::ONE;
        for dig in 0..rg_params.k_g {
            let (c, beta_i, alpha_i) = &cba_all[inst_idx][dig];
            let eq = eq_eval(c, &r_mon_r).unwrap();
            let b_i = mon_b[inst_idx][dig];
            let ev1 = R::from(ev(&b_i, *beta_i));
            let ev2 = R::from(ev(&b_i, *beta_i * *beta_i));
            let b_claim = ev1 * ev1 - ev2;
            let mut term = eq * R::from(*alpha_i) * b_claim;
            if let Some(rc) = &rc_all[inst_idx] {
                term *= R::from(rc_pow);
                rc_pow *= *rc;
            }
            inst_acc += term;
        }
        ver += rhos[inst_idx] * inst_acc;
    }

    if ver != v_expected {
        return Err("PiFold: batched monomial recomputation mismatch".to_string());
    }

    // Π_rg Step-5 check on the **folded** values.
    let log_m = log2(m.next_power_of_two()) as usize;
    let s_chals = mon_sc.point[log_m..].to_vec();
    let ts_s_full = ts_weights(&s_chals);
    let ts_s = &ts_s_full[..d];

    // Fold u*: for each digit, u*(dig) := Σ_i beta_i * u_i(dig)
    let mut u_folded = vec![R::ZERO; rg_params.k_g];
    for inst_idx in 0..ell {
        let b = beta_ring[inst_idx];
        for dig in 0..rg_params.k_g {
            u_folded[dig] += b * mon_b[inst_idx][dig];
        }
    }

    for dig in 0..rg_params.k_g {
        let lhs = (psi::<R>() * u_folded[dig]).ct();
        let rhs = proof.v_digits_folded[dig]
            .iter()
            .zip(ts_s.iter())
            .fold(R::BaseRing::ZERO, |acc, (&vij, &t)| acc + vij * t);
        if lhs != rhs {
            return Err(format!("PiFold: folded Step5 mismatch at dig={dig}"));
        }
    }

    // Construct folded outputs directly.
    let mut c_star = vec![R::ZERO; cms[0].len()];
    for (inst_idx, cm) in cms.iter().enumerate() {
        for (acc, x) in c_star.iter_mut().zip(cm.iter()) {
            *acc += beta_ring[inst_idx] * *x;
        }
    }

    // r and r' are shared across the batch.
    let log_mj = log2(m_j.next_power_of_two()) as usize;
    let r_prime = mon_sc.point.clone();
    let r_star = r_prime[..log_mj].to_vec();

    // u* is the folded u_folded computed above; v* is composed from folded v_digits.
    let v_star = compose_v_digits::<R>(&proof.v_digits_folded, rg_params.d_prime);

    let mut v_rq = R::ZERO;
    for (i, c) in v_star.iter().enumerate() {
        v_rq.coeffs_mut()[i] = *c;
    }

    let folded_inst = SymphonyInstance {
        c: c_star,
        r: r_star,
        v: v_rq,
    };
    let folded_bat = SymphonyBatchLin { r_prime, u: u_folded };

    Ok((folded_inst, folded_bat))
}

/// Derive the auxiliary transcript messages (`had_u`, `mon_b`) from witness openings under the
/// Poseidon-FS transcript schedule (shared-randomness / shared-rounds).
///
/// This is a helper for WE/DPP frontends: these messages are the “folding transcript witness”
/// that `R_cp` expects, but production systems would commit to them instead of recomputing them.
pub fn derive_pi_fold_aux_witness_poseidon_fs<R: CoeffRing, PC>(
    M: [&SparseMatrix<R>; 3],
    cms: &[Vec<R>],
    proof: &PiFoldBatchedProof<R>,
    cms_openings: &[Vec<R>],
    public_inputs: &[R::BaseRing],
) -> Result<PiFoldAuxWitness<R>, String>
where
    R::BaseRing: Zq + Decompose,
    PC: GetPoseidonParams<<<R>::BaseRing as ark_ff::Field>::BasePrimeField>,
{
    let mut ts = crate::transcript::PoseidonTranscript::<R>::empty::<PC>();
    ts.absorb_field_element(&R::BaseRing::from(0x4c465053_50494250u128)); // "LFPS_PIBP"
    absorb_public_inputs::<R>(&mut ts, public_inputs);

    // Run the verifier schedule up to and including the two sumcheck verifications, then
    // recompute the post-sumcheck transcript messages from the witness openings.
    //
    // NOTE: this helper intentionally skips commitment-opening checks (`VfyOpen`), since those are
    // orthogonal to the folding transcript schedule and should be enforced separately in `R_cp`.
    let ell = cms.len();
    if ell == 0 || cms_openings.len() != ell {
        return Err("PiFoldAux: length mismatch".to_string());
    }
    let rg_params = &proof.rg_params;
    let beta_cts = derive_beta_chi::<R>(&mut ts, ell);
    let beta_ring = beta_cts.iter().copied().map(R::from).collect::<Vec<R>>();
    let m = M[0].nrows;
    if !m.is_power_of_two() {
        return Err("PiFoldAux: m must be power-of-two".to_string());
    }
    let log_m = log2(m) as usize;
    let d = R::dimension();

    ts.absorb_field_element(&R::BaseRing::from(0x4c465053_50494841u128)); // "LFPS_PIHA"
    let s_base = ts.get_challenges(log_m);
    let alpha_base = ts.get_challenge();
    let s_poly: Vec<R> = s_base.iter().copied().map(R::from).collect();

    let m_j = proof.m_j;
    if m != proof.m {
        return Err("PiFoldAux: proof m mismatch".to_string());
    }
    if m < m_j || m % m_j != 0 {
        return Err("PiFoldAux: require m_J<=m".to_string());
    }
    let g_len = m * d;
    if !g_len.is_power_of_two() {
        return Err("PiFoldAux: require m*d power-of-two".to_string());
    }
    let g_nvars = log2(g_len) as usize;

    // Witness lengths must match and define m_J.
    let n_f = cms_openings[0].len();
    if n_f == 0 || n_f % rg_params.l_h != 0 {
        return Err("PiFoldAux: invalid witness length".to_string());
    }
    for w in cms_openings.iter() {
        if w.len() != n_f {
            return Err("PiFoldAux: inconsistent witness lengths".to_string());
        }
    }
    let blocks = n_f / rg_params.l_h;
    let m_j_expected = blocks * rg_params.lambda_pj;
    if m_j_expected != m_j {
        return Err("PiFoldAux: m_J mismatch".to_string());
    }

    let mut Js: Vec<Vec<Vec<R::BaseRing>>> = Vec::with_capacity(ell);
    let mut cba_all: Vec<Vec<(Vec<R>, R::BaseRing, R::BaseRing)>> = Vec::with_capacity(ell);
    let mut rc_all: Vec<Option<R::BaseRing>> = Vec::with_capacity(ell);

    for cm_f in cms.iter() {
        ts.absorb_slice(cm_f);
        let J = derive_J::<R>(&mut ts, rg_params.lambda_pj, rg_params.l_h);
        Js.push(J);

        let mut cba: Vec<(Vec<R>, R::BaseRing, R::BaseRing)> = Vec::with_capacity(rg_params.k_g);
        for _ in 0..rg_params.k_g {
            let c: Vec<R> = ts
                .get_challenges(g_nvars)
                .into_iter()
                .map(|x| x.into())
                .collect();
            let beta = ts.get_challenge();
            let alpha = ts.get_challenge();
            cba.push((c, beta, alpha));
        }
        let rc: Option<R::BaseRing> = (rg_params.k_g > 1).then(|| ts.get_challenge());
        cba_all.push(cba);
        rc_all.push(rc);
    }

    let rhos = ts
        .get_challenges(ell)
        .into_iter()
        .map(R::from)
        .collect::<Vec<R>>();

    // Verify the two batched sumchecks with shared challenges (to learn the evaluation points).
    let hook_round = log2(m_j.next_power_of_two()) as usize;
    let (had_sc, mon_sc) = MLSumcheck::<R, _>::verify_two_as_subprotocol_shared_with_hook(
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
    .map_err(|e| format!("PiFoldAux: sumcheck verify failed: {e}"))?;

    // Recompute had_u at the had evaluation point and absorb it (matching verifier schedule).
    let r_poly_had: Vec<R> = had_sc.point.iter().copied().map(R::from).collect();
    let eq_sr = eq_eval(&s_poly, &r_poly_had).map_err(|e| format!("PiFoldAux: eq_eval failed: {e}"))?;

    let mut pow = R::BaseRing::ONE;
    let mut alpha_pows = Vec::with_capacity(d);
    for _ in 0..d {
        alpha_pows.push(R::from(pow));
        pow *= alpha_base;
    }

    let mut had_u: Vec<[Vec<R::BaseRing>; 3]> = Vec::with_capacity(ell);
    let mut lhs = R::ZERO;
    for inst_idx in 0..ell {
        let y = M
            .iter()
            .map(|Mi| Mi.try_mul_vec(&cms_openings[inst_idx]).expect("mat-vec mul failed"))
            .collect::<Vec<Vec<R>>>();

        let mut U: [Vec<R::BaseRing>; 3] = [Vec::with_capacity(d), Vec::with_capacity(d), Vec::with_capacity(d)];
        for i in 0..3 {
            for j in 0..d {
                let evals = (0..m)
                    .map(|row| R::from(y[i][row].coeffs()[j]))
                    .collect::<Vec<_>>();
                let mle = DenseMultilinearExtension::from_evaluations_vec(log_m, evals);
                let v = mle.evaluate(&r_poly_had).expect("MLE evaluate returned None");
                U[i].push(v.ct());
            }
        }

        let mut acc = R::ZERO;
        for j in 0..d {
            let u1 = R::from(U[0][j]);
            let u2 = R::from(U[1][j]);
            let u3 = R::from(U[2][j]);
            acc += alpha_pows[j] * (u1 * u2 - u3);
        }
        lhs += rhos[inst_idx] * (eq_sr * acc);

        for x in &U[0] { ts.absorb_field_element(x); }
        for x in &U[1] { ts.absorb_field_element(x); }
        for x in &U[2] { ts.absorb_field_element(x); }
        had_u.push(U);
    }
    if lhs != had_sc.expected_evaluation {
        return Err("PiFoldAux: batched had Eq(26) mismatch".to_string());
    }

    // Recompute mon_b at the mon evaluation point and absorb it (matching verifier schedule).
    let r_mon_r: Vec<R> = mon_sc.point.iter().copied().map(R::from).collect();
    let expand_row = |row: usize| -> usize { row % m_j };

    let mut mon_b: Vec<Vec<R>> = Vec::with_capacity(ell);
    for inst_idx in 0..ell {
        // Build H_digits for this instance (on m_J rows).
        let f = &cms_openings[inst_idx];
        let mut H = vec![vec![R::BaseRing::ZERO; d]; m_j];
        let Jref = &Js[inst_idx];
        for b in 0..blocks {
            for i in 0..rg_params.lambda_pj {
                let out_row = b * rg_params.lambda_pj + i;
                for t in 0..rg_params.l_h {
                    let in_row = b * rg_params.l_h + t;
                    let coef = Jref[i][t];
                    for col in 0..d {
                        H[out_row][col] += coef * f[in_row].coeffs()[col];
                    }
                }
            }
        }
        let mut H_digits: Vec<Vec<Vec<R::BaseRing>>> =
            vec![vec![vec![R::BaseRing::ZERO; d]; m_j]; rg_params.k_g];
        for r in 0..m_j {
            let row_digits = H[r].decompose_to_vec(rg_params.d_prime, rg_params.k_g);
            for c in 0..d {
                for i in 0..rg_params.k_g {
                    H_digits[i][r][c] = row_digits[c][i];
                }
            }
        }

        let mut b_inst = Vec::with_capacity(rg_params.k_g);
        for dig in 0..rg_params.k_g {
            let mut gi = Vec::with_capacity(g_len);
            for c in 0..d {
                for r in 0..m {
                    gi.push(exp::<R>(H_digits[dig][expand_row(r)][c]).expect("Exp failed"));
                }
            }
            let mle = DenseMultilinearExtension::from_evaluations_vec(g_nvars, gi);
            b_inst.push(mle.evaluate(&r_mon_r).unwrap());
        }
        ts.absorb_slice(&b_inst);
        mon_b.push(b_inst);
    }

    // Optional consistency: batched mon recomputation should match sumcheck claim.
    let v_expected = mon_sc.expected_evaluation;
    let mut ver = R::ZERO;
    for inst_idx in 0..ell {
        let mut inst_acc = R::ZERO;
        // Precompute rc^dig iteratively to avoid repeated pow().
        let mut rc_pow = R::BaseRing::ONE;
        for dig in 0..rg_params.k_g {
            let (c, beta_i, alpha_i) = &cba_all[inst_idx][dig];
            let eq = eq_eval(c, &r_mon_r).unwrap();
            let b_i = mon_b[inst_idx][dig];
            let ev1 = R::from(ev(&b_i, *beta_i));
            let ev2 = R::from(ev(&b_i, *beta_i * *beta_i));
            let b_claim = ev1 * ev1 - ev2;
            let mut term = eq * R::from(*alpha_i) * b_claim;
            if let Some(rc) = &rc_all[inst_idx] {
                term *= R::from(rc_pow);
                rc_pow *= *rc;
            }
            inst_acc += term;
        }
        ver += rhos[inst_idx] * inst_acc;
    }
    if ver != v_expected {
        return Err("PiFoldAux: batched monomial recomputation mismatch".to_string());
    }

    // Optional consistency: folded Step-5 should pass under these recomputed b values.
    let log_m2 = log2(m.next_power_of_two()) as usize;
    let s_chals = mon_sc.point[log_m2..].to_vec();
    let ts_s_full = ts_weights(&s_chals);
    let ts_s = &ts_s_full[..d];
    let mut u_folded = vec![R::ZERO; rg_params.k_g];
    for inst_idx in 0..ell {
        let b = beta_ring[inst_idx];
        for dig in 0..rg_params.k_g {
            u_folded[dig] += b * mon_b[inst_idx][dig];
        }
    }
    for dig in 0..rg_params.k_g {
        let lhs = (psi::<R>() * u_folded[dig]).ct();
        let rhs = proof.v_digits_folded[dig]
            .iter()
            .zip(ts_s.iter())
            .fold(R::BaseRing::ZERO, |acc, (&vij, &t)| acc + vij * t);
        if lhs != rhs {
            return Err(format!("PiFoldAux: folded Step5 mismatch at dig={dig}"));
        }
    }

    Ok(PiFoldAuxWitness { had_u, mon_b })
}

/// WE/DPP-facing verifier: check Π_fold under Poseidon-FS using **CP transcript-message commitments**.
///
/// - `cm_f`: commitments that define the folded instance output (`c* = Σ β_i·cm_f[i]`)
/// - `cfs_*`: commitments to CP transcript messages (`m_i`), opened to `aux`
///
/// This closes the “two-witness gap” by binding the auxiliary transcript messages used in
/// verification to commitments that are part of the statement.
pub fn verify_pi_fold_batched_and_fold_outputs_poseidon_fs_cp<R: CoeffRing, PC>(
    M: [&SparseMatrix<R>; 3],
    cm_f: &[Vec<R>],
    proof: &PiFoldBatchedProof<R>,
    open: &impl VfyOpen<R>,
    cfs_had_u: &[Vec<R>],
    cfs_mon_b: &[Vec<R>],
    aux: &PiFoldAuxWitness<R>,
    public_inputs: &[R::BaseRing],
) -> Result<(SymphonyInstance<R>, SymphonyBatchLin<R>), String>
where
    R::BaseRing: Zq + Decompose,
    PC: GetPoseidonParams<<<R>::BaseRing as ark_ff::Field>::BasePrimeField>,
{
    verify_pi_fold_batched_and_fold_outputs_poseidon_fs_cp_with_metrics::<R, PC>(
        M,
        cm_f,
        proof,
        open,
        cfs_had_u,
        cfs_mon_b,
        aux,
        public_inputs,
    )
    .map(|(out, _metrics)| out)
}

/// WE/DPP-facing verifier, plus transcript metrics for empirical Poseidon cost estimation.
pub fn verify_pi_fold_batched_and_fold_outputs_poseidon_fs_cp_with_metrics<R: CoeffRing, PC>(
    M: [&SparseMatrix<R>; 3],
    cm_f: &[Vec<R>],
    proof: &PiFoldBatchedProof<R>,
    open: &impl VfyOpen<R>,
    cfs_had_u: &[Vec<R>],
    cfs_mon_b: &[Vec<R>],
    aux: &PiFoldAuxWitness<R>,
    public_inputs: &[R::BaseRing],
) -> Result<((SymphonyInstance<R>, SymphonyBatchLin<R>), PoseidonTranscriptMetrics), String>
where
    R::BaseRing: Zq + Decompose,
    PC: GetPoseidonParams<<<R>::BaseRing as ark_ff::Field>::BasePrimeField>,
{
    let mut ts = crate::transcript::PoseidonTranscript::<R>::empty::<PC>();
    ts.absorb_field_element(&R::BaseRing::from(0x4c465053_50494250u128)); // "LFPS_PIBP"
    absorb_public_inputs::<R>(&mut ts, public_inputs);

    if cfs_had_u.len() != cm_f.len() || cfs_mon_b.len() != cm_f.len() {
        return Err("PiFoldCP: cfs length mismatch".to_string());
    }

    // Now run the verifier logic using aux messages (opened to the CP commitments).
    let ell = cm_f.len();
    if ell == 0 {
        return Err("PiFoldCP: empty batch".to_string());
    }
    if aux.had_u.len() != ell || aux.mon_b.len() != ell {
        return Err("PiFoldCP: aux length mismatch".to_string());
    }

    // Reuse the core verifier but skip recomputation (aux supplies U/b).
    let out = verify_pi_fold_batched_and_fold_outputs_with_openings_and_aux(
        &mut ts,
        M,
        cm_f,
        proof,
        &NoOpen, // cm_f is not opened in CP relation
        &[],     // no witness openings needed; cm_f treated as part of the statement
        Some(aux),
    )?;

    // Verify CP commitment openings (domain-separated by `VfyOpen`).
    for i in 0..ell {
        let had_u_enc = encode_had_u_instance::<R>(&aux.had_u[i]);
        open.verify_opening(&mut ts, "cfs_had_u", &cfs_had_u[i], &[], &had_u_enc, &[])?;
        open.verify_opening(&mut ts, "cfs_mon_b", &cfs_mon_b[i], &[], &aux.mon_b[i], &[])?;
    }

    Ok((out, ts.metrics()))
}

/// WE/DPP-facing verifier for hetero-M batched Π_fold: check Poseidon-FS proof using CP transcript-message commitments.
///
/// This is the “one proof for all chunks” verification interface: `cm_f` and `cfs_*` are part of the
/// statement, while `aux` is the (opened) CP transcript witness.
pub fn verify_pi_fold_batched_and_fold_outputs_poseidon_fs_cp_hetero_m<R: CoeffRing, PC>(
    Ms: &[[&SparseMatrix<R>; 3]],
    cm_f: &[Vec<R>],
    proof: &PiFoldBatchedProof<R>,
    open: &impl VfyOpen<R>,
    cfs_had_u: &[Vec<R>],
    cfs_mon_b: &[Vec<R>],
    aux: &PiFoldAuxWitness<R>,
    public_inputs: &[R::BaseRing],
) -> Result<(SymphonyInstance<R>, SymphonyBatchLin<R>), String>
where
    R::BaseRing: Zq + Decompose,
    PC: GetPoseidonParams<<<R>::BaseRing as ark_ff::Field>::BasePrimeField>,
{
    verify_pi_fold_batched_and_fold_outputs_poseidon_fs_cp_hetero_m_with_metrics::<R, PC>(
        Ms,
        cm_f,
        proof,
        open,
        cfs_had_u,
        cfs_mon_b,
        aux,
        public_inputs,
    )
    .map(|(out, _metrics)| out)
}

/// WE/DPP-facing hetero-M verifier, plus transcript metrics for empirical Poseidon cost estimation.
pub fn verify_pi_fold_batched_and_fold_outputs_poseidon_fs_cp_hetero_m_with_metrics<
    R: CoeffRing,
    PC,
>(
    Ms: &[[&SparseMatrix<R>; 3]],
    cm_f: &[Vec<R>],
    proof: &PiFoldBatchedProof<R>,
    open: &impl VfyOpen<R>,
    cfs_had_u: &[Vec<R>],
    cfs_mon_b: &[Vec<R>],
    aux: &PiFoldAuxWitness<R>,
    public_inputs: &[R::BaseRing],
) -> Result<((SymphonyInstance<R>, SymphonyBatchLin<R>), PoseidonTranscriptMetrics), String>
where
    R::BaseRing: Zq + Decompose,
    PC: GetPoseidonParams<<<R>::BaseRing as ark_ff::Field>::BasePrimeField>,
{
    let mut ts = crate::transcript::PoseidonTranscript::<R>::empty::<PC>();
    ts.absorb_field_element(&R::BaseRing::from(0x4c465053_50494250u128)); // "LFPS_PIBP"
    absorb_public_inputs::<R>(&mut ts, public_inputs);

    if cfs_had_u.len() != cm_f.len() || cfs_mon_b.len() != cm_f.len() {
        return Err("PiFoldCP: cfs length mismatch".to_string());
    }
    let ell = cm_f.len();
    if ell == 0 {
        return Err("PiFoldCP: empty batch".to_string());
    }
    if aux.had_u.len() != ell || aux.mon_b.len() != ell {
        return Err("PiFoldCP: aux length mismatch".to_string());
    }

    // IMPORTANT: run the core Π_fold verifier using the *same* Poseidon transcript instance `ts`
    // so metrics include the full FS transcript work (coins + sumcheck transcript).
    let out = verify_pi_fold_batched_and_fold_outputs_with_openings_and_aux_hetero_m(
        &mut ts,
        Ms,
        cm_f,
        proof,
        &NoOpen, // cm_f is not opened in CP relation
        &[],
        Some(aux),
    )?;

    for i in 0..ell {
        let had_u_enc = encode_had_u_instance::<R>(&aux.had_u[i]);
        open.verify_opening(&mut ts, "cfs_had_u", &cfs_had_u[i], &[], &had_u_enc, &[])?;
        open.verify_opening(&mut ts, "cfs_mon_b", &cfs_mon_b[i], &[], &aux.mon_b[i], &[])?;
    }

    Ok((out, ts.metrics()))
}

/// Same as `verify_pi_fold_batched_and_fold_outputs_poseidon_fs_cp_hetero_m_with_metrics`, but returns
/// metrics even if verification fails (so callers can measure transcript work empirically on failing
/// instances / dummy witnesses).
pub fn verify_pi_fold_batched_and_fold_outputs_poseidon_fs_cp_hetero_m_with_metrics_result<
    R: CoeffRing,
    PC,
>(
    Ms: &[[&SparseMatrix<R>; 3]],
    cm_f: &[Vec<R>],
    proof: &PiFoldBatchedProof<R>,
    open: &impl VfyOpen<R>,
    cfs_had_u: &[Vec<R>],
    cfs_mon_b: &[Vec<R>],
    aux: &PiFoldAuxWitness<R>,
    public_inputs: &[R::BaseRing],
) -> (Result<(SymphonyInstance<R>, SymphonyBatchLin<R>), String>, PoseidonTranscriptMetrics)
where
    R::BaseRing: Zq + Decompose,
    PC: GetPoseidonParams<<<R>::BaseRing as ark_ff::Field>::BasePrimeField>,
{
    let mut ts = crate::transcript::PoseidonTranscript::<R>::empty::<PC>();
    ts.absorb_field_element(&R::BaseRing::from(0x4c465053_50494250u128)); // "LFPS_PIBP"
    absorb_public_inputs::<R>(&mut ts, public_inputs);

    let res: Result<(SymphonyInstance<R>, SymphonyBatchLin<R>), String> = (|| {
        if cfs_had_u.len() != cm_f.len() || cfs_mon_b.len() != cm_f.len() {
            return Err("PiFoldCP: cfs length mismatch".to_string());
        }
        let ell = cm_f.len();
        if ell == 0 {
            return Err("PiFoldCP: empty batch".to_string());
        }
        if aux.had_u.len() != ell || aux.mon_b.len() != ell {
            return Err("PiFoldCP: aux length mismatch".to_string());
        }

        // IMPORTANT: run the core Π_fold verifier using the *same* Poseidon transcript instance `ts`
        // so metrics include the full FS transcript work (coins + sumcheck transcript).
        let out = verify_pi_fold_batched_and_fold_outputs_with_openings_and_aux_hetero_m(
            &mut ts,
            Ms,
            cm_f,
            proof,
            &NoOpen, // cm_f is not opened in CP relation
            &[],
            Some(aux),
        )?;

        for i in 0..ell {
            let had_u_enc = encode_had_u_instance::<R>(&aux.had_u[i]);
            open.verify_opening(&mut ts, "cfs_had_u", &cfs_had_u[i], &[], &had_u_enc, &[])?;
            open.verify_opening(&mut ts, "cfs_mon_b", &cfs_mon_b[i], &[], &aux.mon_b[i], &[])?;
        }

        Ok(out)
    })();

    // Always return metrics, even on failure.
    let metrics = ts.metrics();
    (res, metrics)
}

/// Relation-check style wrapper for GM‑1 / DPP frontends:
/// returns `Ok(())` iff the folding verifier accepts under Poseidon-FS (Poseidon transcript).
pub fn check_verify_fold_poseidon_fs<R: CoeffRing, PC>(
    M: [&SparseMatrix<R>; 3],
    cms: &[Vec<R>],
    proof: &PiFoldBatchedProof<R>,
    open: &impl VfyOpen<R>,
    cms_openings: &[Vec<R>],
    public_inputs: &[R::BaseRing],
) -> Result<(), String>
where
    R::BaseRing: Zq + Decompose,
    PC: GetPoseidonParams<<<R>::BaseRing as ark_ff::Field>::BasePrimeField>,
{
    let _ = verify_pi_fold_batched_and_fold_outputs_poseidon_fs::<R, PC>(
        M,
        cms,
        proof,
        open,
        cms_openings,
        public_inputs,
    )?;
    Ok(())
}

pub fn verify_pi_fold_batched_and_fold_outputs_with_openings<R: CoeffRing>(
    transcript: &mut impl Transcript<R>,
    M: [&SparseMatrix<R>; 3],
    cms: &[Vec<R>],
    proof: &PiFoldBatchedProof<R>,
    open: &impl VfyOpen<R>,
    cms_openings: &[Vec<R>],
) -> Result<(SymphonyInstance<R>, SymphonyBatchLin<R>), String>
where
    R::BaseRing: Zq + Decompose,
{
    verify_pi_fold_batched_and_fold_outputs_with_openings_and_aux(
        transcript, M, cms, proof, open, cms_openings, None,
    )
}

/// Verifier with optional auxiliary witness messages (`had_u`, `mon_b`) to avoid
/// recomputing them from full witness openings.
pub fn verify_pi_fold_batched_and_fold_outputs_with_openings_and_aux<R: CoeffRing>(
    transcript: &mut impl Transcript<R>,
    M: [&SparseMatrix<R>; 3],
    cms: &[Vec<R>],
    proof: &PiFoldBatchedProof<R>,
    open: &impl VfyOpen<R>,
    cms_openings: &[Vec<R>],
    aux: Option<&PiFoldAuxWitness<R>>,
) -> Result<(SymphonyInstance<R>, SymphonyBatchLin<R>), String>
where
    R::BaseRing: Zq + Decompose,
{
    // Commitment-opening layer is verified **outside** the transcript schedule used for the
    // folding/sumcheck subprotocols.
    //
    // When `aux` is provided (CP/WE-facing path), the verifier does not need the full witness
    // openings; callers may pass `cms_openings = &[]` to skip opening checks for `cm_witness`.
    // (In that mode, `cm_f` is treated as part of the public statement.)
    if !cms_openings.is_empty() {
        if cms.len() != cms_openings.len() {
            return Err("PiFold: cms/openings length mismatch".to_string());
        }
        for (cm, open_val) in cms.iter().zip(cms_openings.iter()) {
            open.verify_opening(transcript, "cm_witness", cm, &[], open_val, &[])?;
        }
    } else if aux.is_none() {
        return Err("PiFold: cms_openings required when aux is not provided".to_string());
    }

    let ell = cms.len();
    if ell == 0 {
        return Err("PiFold: length mismatch".to_string());
    }
    let rg_params = &proof.rg_params;

    let beta_cts = derive_beta_chi::<R>(transcript, ell);
    let beta_ring = beta_cts.iter().copied().map(R::from).collect::<Vec<R>>();
    let m = M[0].nrows;
    if !m.is_power_of_two() {
        return Err("PiFold: m must be power-of-two".to_string());
    }
    let log_m = log2(m) as usize;
    let d = R::dimension();

    transcript.absorb_field_element(&R::BaseRing::from(0x4c465053_50494841u128)); // "LFPS_PIHA"
    let s_base = transcript.get_challenges(log_m);
    let alpha_base = transcript.get_challenge();

    let s_poly: Vec<R> = s_base.iter().copied().map(R::from).collect();
    let m_j = proof.m_j;
    if m != proof.m {
        return Err("PiFold: proof m mismatch".to_string());
    }
    if m < m_j || m % m_j != 0 {
        return Err("PiFold: require m_J<=m".to_string());
    }

    let g_len = m * d;
    if !g_len.is_power_of_two() {
        return Err("PiFold: require m*d power-of-two".to_string());
    }
    let g_nvars = log2(g_len) as usize;

    let mut cba_all: Vec<Vec<(Vec<R>, R::BaseRing, R::BaseRing)>> = Vec::with_capacity(ell);
    let mut rc_all: Vec<Option<R::BaseRing>> = Vec::with_capacity(ell);
    let mut Js: Vec<Vec<Vec<R::BaseRing>>> = Vec::with_capacity(ell);

    // If we don't have auxiliary transcript messages, we will recompute them from witness openings,
    // which requires a consistent witness length to define m_J.
    let (_n_f, blocks) = if aux.is_none() {
        let n_f = cms_openings[0].len();
        if n_f == 0 || n_f % rg_params.l_h != 0 {
            return Err("PiFold: invalid witness length".to_string());
        }
        for w in cms_openings.iter() {
            if w.len() != n_f {
                return Err("PiFold: inconsistent witness lengths".to_string());
            }
        }
        let blocks = n_f / rg_params.l_h;
        let m_j_expected = blocks * rg_params.lambda_pj;
        if m_j_expected != m_j {
            return Err("PiFold: m_J mismatch".to_string());
        }
        (n_f, blocks)
    } else {
        (0usize, 0usize)
    };

    for cm_f in cms.iter() {
        transcript.absorb_slice(cm_f);
        let J = derive_J::<R>(transcript, rg_params.lambda_pj, rg_params.l_h);
        Js.push(J);

        let mut cba: Vec<(Vec<R>, R::BaseRing, R::BaseRing)> = Vec::with_capacity(rg_params.k_g);
        for _ in 0..rg_params.k_g {
            let c: Vec<R> = transcript
                .get_challenges(g_nvars)
                .into_iter()
                .map(|x| x.into())
                .collect();
            let beta = transcript.get_challenge();
            let alpha = transcript.get_challenge();
            cba.push((c, beta, alpha));
        }
        let rc: Option<R::BaseRing> = (rg_params.k_g > 1).then(|| transcript.get_challenge());
        cba_all.push(cba);
        rc_all.push(rc);
    }

    let rhos = transcript
        .get_challenges(ell)
        .into_iter()
        .map(R::from)
        .collect::<Vec<R>>();

    // Verify the two batched sumchecks with shared challenges.
    let hook_round = log2(m_j.next_power_of_two()) as usize;
    let (had_sc, mon_sc) = MLSumcheck::<R, _>::verify_two_as_subprotocol_shared_with_hook(
        transcript,
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
    .map_err(|e| format!("PiFold: sumcheck verify failed: {e}"))?;

    // Batched Eq.(26) check for Π_had.
    // If `aux` is present, use it and avoid recomputing U from the full witness opening.
    let r_poly_had: Vec<R> = had_sc.point.iter().copied().map(R::from).collect();
    let eq_sr = eq_eval(&s_poly, &r_poly_had).map_err(|e| format!("PiFold: eq_eval failed: {e}"))?;

    let mut pow = R::BaseRing::ONE;
    let mut alpha_pows = Vec::with_capacity(d);
    for _ in 0..d {
        alpha_pows.push(R::from(pow));
        pow *= alpha_base;
    }

    let mut lhs = R::ZERO;
    for inst_idx in 0..ell {
        let U: [Vec<R::BaseRing>; 3] = if let Some(auxw) = aux {
            if auxw.had_u.len() != ell {
                return Err("PiFold: aux.had_u length mismatch".to_string());
            }
            auxw.had_u[inst_idx].clone()
        } else {
            // Recompute y = (M1 f, M2 f, M3 f).
            let y = M
                .iter()
                .map(|Mi| Mi.try_mul_vec(&cms_openings[inst_idx]).expect("mat-vec mul failed"))
                .collect::<Vec<Vec<R>>>();

            let mut U: [Vec<R::BaseRing>; 3] =
                [Vec::with_capacity(d), Vec::with_capacity(d), Vec::with_capacity(d)];
            for i in 0..3 {
                for j in 0..d {
                    let evals = (0..m)
                        .map(|row| R::from(y[i][row].coeffs()[j]))
                        .collect::<Vec<_>>();
                    let mle = DenseMultilinearExtension::from_evaluations_vec(log_m, evals);
                    let v = mle
                        .evaluate(&r_poly_had)
                        .expect("MLE evaluate returned None");
                    U[i].push(v.ct());
                }
            }
            U
        };

        if U[0].len() != d || U[1].len() != d || U[2].len() != d {
            return Err("PiFold: aux had_U has wrong dimension".to_string());
        }

        let mut acc = R::ZERO;
        for j in 0..d {
            let u1 = R::from(U[0][j]);
            let u2 = R::from(U[1][j]);
            let u3 = R::from(U[2][j]);
            acc += alpha_pows[j] * (u1 * u2 - u3);
        }
        lhs += rhos[inst_idx] * (eq_sr * acc);

        // absorb U (prover did this after sumcheck)
        for x in &U[0] { transcript.absorb_field_element(x); }
        for x in &U[1] { transcript.absorb_field_element(x); }
        for x in &U[2] { transcript.absorb_field_element(x); }
    }

    if lhs != had_sc.expected_evaluation {
        return Err("PiFold: batched had Eq(26) mismatch".to_string());
    }

    // Per-instance monomial `b` values at the mon point, and absorb them (prover does this after sumchecks).
    // If `aux` is present, use it and avoid recomputing from witness openings.
    let r_mon_r: Vec<R> = mon_sc.point.iter().copied().map(R::from).collect();
    let mon_b: Vec<Vec<R>> = if let Some(auxw) = aux {
        if auxw.mon_b.len() != ell {
            return Err("PiFold: aux.mon_b length mismatch".to_string());
        }
        for b_inst in auxw.mon_b.iter() {
            if b_inst.len() != rg_params.k_g {
                return Err("PiFold: aux.mon_b wrong k_g".to_string());
            }
        }
        for b_inst in auxw.mon_b.iter() {
            transcript.absorb_slice(b_inst);
        }
        auxw.mon_b.clone()
    } else {
        let mut mon_b: Vec<Vec<R>> = Vec::with_capacity(ell);
        let expand_row = |row: usize| -> usize { row % m_j };
        for inst_idx in 0..ell {
            // Build H_digits for this instance (on m_J rows).
            let f = &cms_openings[inst_idx];
            let mut H = vec![vec![R::BaseRing::ZERO; d]; m_j];
            let Jref = &Js[inst_idx];
            for b in 0..blocks {
                for i in 0..rg_params.lambda_pj {
                    let out_row = b * rg_params.lambda_pj + i;
                    for t in 0..rg_params.l_h {
                        let in_row = b * rg_params.l_h + t;
                        let coef = Jref[i][t];
                        for col in 0..d {
                            H[out_row][col] += coef * f[in_row].coeffs()[col];
                        }
                    }
                }
            }
            let mut H_digits: Vec<Vec<Vec<R::BaseRing>>> =
                vec![vec![vec![R::BaseRing::ZERO; d]; m_j]; rg_params.k_g];
            for r in 0..m_j {
                let row_digits = H[r].decompose_to_vec(rg_params.d_prime, rg_params.k_g);
                for c in 0..d {
                    for i in 0..rg_params.k_g {
                        H_digits[i][r][c] = row_digits[c][i];
                    }
                }
            }

            let mut b_inst = Vec::with_capacity(rg_params.k_g);
            for dig in 0..rg_params.k_g {
                let mut gi = Vec::with_capacity(g_len);
                for c in 0..d {
                    for r in 0..m {
                        gi.push(exp::<R>(H_digits[dig][expand_row(r)][c]).expect("Exp failed"));
                    }
                }
                let mle = DenseMultilinearExtension::from_evaluations_vec(g_nvars, gi);
                b_inst.push(mle.evaluate(&r_mon_r).unwrap());
            }
            transcript.absorb_slice(&b_inst);
            mon_b.push(b_inst);
        }
        mon_b
    };

    let v_expected = mon_sc.expected_evaluation;

    let mut ver = R::ZERO;
    for inst_idx in 0..ell {
        let mut inst_acc = R::ZERO;
        // Precompute rc^dig iteratively to avoid repeated pow().
        let mut rc_pow = R::BaseRing::ONE;
        for dig in 0..rg_params.k_g {
            let (c, beta_i, alpha_i) = &cba_all[inst_idx][dig];
            let eq = eq_eval(c, &r_mon_r).unwrap();
            let b_i = mon_b[inst_idx][dig];
            let ev1 = R::from(ev(&b_i, *beta_i));
            let ev2 = R::from(ev(&b_i, *beta_i * *beta_i));
            let b_claim = ev1 * ev1 - ev2;
            let mut term = eq * R::from(*alpha_i) * b_claim;
            if let Some(rc) = &rc_all[inst_idx] {
                term *= R::from(rc_pow);
                rc_pow *= *rc;
            }
            inst_acc += term;
        }
        ver += rhos[inst_idx] * inst_acc;
    }

    if ver != v_expected {
        return Err("PiFold: batched monomial recomputation mismatch".to_string());
    }

    // Π_rg Step-5 check on the **folded** values.
    //
    // With β restricted to base-field scalars (embedded as constants), this is sound and
    // matches the folding algebra used by Figure 4.
    let log_m = log2(m.next_power_of_two()) as usize;
    let s_chals = mon_sc.point[log_m..].to_vec();
    let ts_s_full = ts_weights(&s_chals);
    let ts_s = &ts_s_full[..d];

    // Fold u*: for each digit, u*(dig) := Σ_i beta_i * u_i(dig)
    let mut u_folded = vec![R::ZERO; rg_params.k_g];
    for inst_idx in 0..ell {
        let b = beta_ring[inst_idx];
        for dig in 0..rg_params.k_g {
            u_folded[dig] += b * mon_b[inst_idx][dig];
        }
    }

    for dig in 0..rg_params.k_g {
        let lhs = (psi::<R>() * u_folded[dig]).ct();
        let rhs = proof.v_digits_folded[dig]
                .iter()
                .zip(ts_s.iter())
                .fold(R::BaseRing::ZERO, |acc, (&vij, &t)| acc + vij * t);
            if lhs != rhs {
            return Err(format!("PiFold: folded Step5 mismatch at dig={dig}"));
        }
    }

    // Construct folded outputs directly.
    // c* = Σ beta_i * c_i  (beta_i are constants, but we still use ring arithmetic)
    let mut c_star = vec![R::ZERO; cms[0].len()];
    for (inst_idx, cm) in cms.iter().enumerate() {
        for (acc, x) in c_star.iter_mut().zip(cm.iter()) {
            *acc += beta_ring[inst_idx] * *x;
        }
    }

    // r and r' are shared across the batch.
    let log_mj = log2(m_j.next_power_of_two()) as usize;
    let r_prime = mon_sc.point.clone();
    let r_star = r_prime[..log_mj].to_vec();

    // u* is the folded u_folded computed above; v* is composed from folded v_digits.
    let v_star = compose_v_digits::<R>(&proof.v_digits_folded, rg_params.d_prime);

    let mut v_rq = R::ZERO;
    for (i, c) in v_star.iter().enumerate() {
        v_rq.coeffs_mut()[i] = *c;
    }

    let folded_inst = SymphonyInstance { c: c_star, r: r_star, v: v_rq };
    let folded_bat = SymphonyBatchLin { r_prime, u: u_folded };

    Ok((folded_inst, folded_bat))
}
