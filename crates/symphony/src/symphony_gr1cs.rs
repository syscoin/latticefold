//! Symphony Π_gr1cs (Figure 3) — interleaving Π_had and Π_rg.
//!
//! This module implements the **single-instance reduction** from Figure 3.
//!
//! Two entrypoints are provided:
//! - `prove_pi_gr1cs` / `verify_pi_gr1cs_and_output`: correctness-first composition
//!   (runs the two sumchecks sequentially).
//! - `prove_pi_gr1cs_shared_sumcheck` / `verify_pi_gr1cs_shared_sumcheck_and_output`:
//!   **paper-faithful Figure 3 Step-3** composition where Π_had's sumcheck and Π_mon's
//!   sumcheck are run in parallel with an identical verifier challenge stream prefix.

use ark_std::log2;
use crate::symphony_coins::{derive_J, ev, ts_weights};
use latticefold::{
    transcript::Transcript,
    utils::sumcheck::{
        utils::{build_eq_x_r, eq_eval},
        MLSumcheck, SumCheckError,
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
use thiserror::Error;

use crate::{
    rp_rgchk::{compose_v_digits, compute_auxj_lin_v_from_witness, verify_pi_rg_and_output, verify_pi_rg_output_relation_with_witness, RPConsistencyError, RPParams, RPRangeProof, RPRangeProver},
    setchk::Out as SetChkOut,
    symphony_had::{compute_pi_had_U_from_witness, prove_pi_had, verify_pi_had_and_output, verify_pi_had_output_relation_with_witness, PiHadProof, PiHadVerifiedOutput},
};

#[derive(Clone, Debug)]
pub struct PiGr1csProof<R: OverField>
where
    R::BaseRing: Zq,
{
    pub had: PiHadProof<R>,
    pub rg: RPRangeProof<R>,
}

#[derive(Clone, Debug)]
pub struct PiGr1csVerifiedOutput<R: OverField>
where
    R::BaseRing: Zq,
{
    pub had: PiHadVerifiedOutput<R>,
    pub rg_r: Vec<R::BaseRing>,
    pub rg_v: Vec<R::BaseRing>,
    pub rg_u: Vec<R>,
}

#[derive(Debug, Error)]
pub enum PiGr1csError<R: OverField>
where
    R::BaseRing: Zq,
{
    #[error("Π_had failed: {0}")]
    Had(#[from] SumCheckError<R>),
    #[error("Π_rg failed: {0}")]
    Rg(#[from] RPConsistencyError),
    #[error("Π_had output relation (Eq. (24)) failed against witness")]
    HadOutputRel,
    #[error("Π_rg output relation (Eq. (31)) failed against witness")]
    RgOutputRel,
    #[error("Π_mon (vector-set) recomputation mismatch under shared sumcheck")]
    SharedMonomialMismatch,
    #[error("Shared sumcheck point mismatch with proof contents")]
    SharedPointMismatch,
}

pub fn prove_pi_gr1cs<R: CoeffRing>(
    transcript: &mut impl Transcript<R>,
    M: [&SparseMatrix<R>; 3],
    cm_f: &[R],
    f: &[R],
    rg_params: RPParams,
) -> Result<PiGr1csProof<R>, PiGr1csError<R>>
where
    R::BaseRing: Zq + Decompose,
{
    // Figure 3 Step 1+4: run Π_had.
    let had = prove_pi_had(transcript, M, f)?;

    // Figure 3 Step 1+2+4: run Π_rg.
    let prover = RPRangeProver::<R>::new(f.to_vec(), rg_params);
    let rg = prover.prove_with_m(transcript, cm_f, Some(M[0].nrows));

    Ok(PiGr1csProof { had, rg })
}

pub fn verify_pi_gr1cs_and_output<R: CoeffRing>(
    transcript: &mut impl Transcript<R>,
    _M: [&SparseMatrix<R>; 3],
    cm_f: &[R],
    proof: &PiGr1csProof<R>,
) -> Result<PiGr1csVerifiedOutput<R>, PiGr1csError<R>>
where
    R::BaseRing: Zq,
{
    let had = verify_pi_had_and_output(transcript, &proof.had)?;

    let rg_out = verify_pi_rg_and_output(&proof.rg, cm_f, transcript)?;

    Ok(PiGr1csVerifiedOutput {
        had,
        rg_r: rg_out.r,
        rg_v: rg_out.v,
        rg_u: rg_out.u,
    })
}

/// Correctness-first bridge: verify Π_gr1cs and also enforce the two output relations
/// (Eq. (24) and Eq. (31)) against the explicit witness `f`.
pub fn verify_pi_gr1cs_output_relations_with_witness<R: CoeffRing>(
    transcript: &mut impl Transcript<R>,
    M: [&SparseMatrix<R>; 3],
    cm_f: &[R],
    proof: &PiGr1csProof<R>,
    f: &[R],
) -> Result<(), PiGr1csError<R>>
where
    R::BaseRing: Zq,
{
    // Verify Π_had and its output relation against f.
    verify_pi_had_output_relation_with_witness(transcript, M, &proof.had, f)
        .map_err(|_| PiGr1csError::HadOutputRel)?;

    // Verify Π_rg and its output relation against f.
    verify_pi_rg_output_relation_with_witness(&proof.rg, cm_f, f, transcript)
        .map_err(|_| PiGr1csError::RgOutputRel)?;

    Ok(())
}


/// Shared-sumcheck-aware bridge: verify the **shared-sumcheck** Π_gr1cs proof and also enforce the
/// two output relations (Eq. (24) and Eq. (31)) against the explicit witness `f`.
///
/// This is the analog of `verify_pi_gr1cs_output_relations_with_witness`, but it uses the
/// paper-faithful shared-sumcheck transcript schedule (Figure 3 Step-3).
pub fn verify_pi_gr1cs_shared_sumcheck_output_relations_with_witness<R: CoeffRing>(
    transcript: &mut impl Transcript<R>,
    M: [&SparseMatrix<R>; 3],
    cm_f: &[R],
    proof: &PiGr1csProof<R>,
    f: &[R],
) -> Result<(), PiGr1csError<R>>
where
    R::BaseRing: Zq,
{
    let out = verify_pi_gr1cs_shared_sumcheck_and_output(transcript, cm_f, proof)?;

    // Eq. (24) linkage for Π_had output.
    let expected_u = compute_pi_had_U_from_witness::<R>(M, f, &out.had.r);
    if expected_u != out.had.U_ct {
        return Err(PiGr1csError::HadOutputRel);
    }

    // Eq. (31) linkage for Π_rg output.
    let v_check = compute_auxj_lin_v_from_witness::<R>(f, &proof.rg.J, &out.rg_r, &proof.rg.params);
    if v_check != out.rg_v {
        return Err(PiGr1csError::RgOutputRel);
    }

    Ok(())
}

// -----------------------------------------------------------------------------
// Paper-faithful Figure 3 Step-3: shared sumcheck challenges (r̄, s̄, s)
// -----------------------------------------------------------------------------

/// Prove Π_gr1cs with **shared sumcheck challenges** between:
/// - Π_had's sumcheck (log m rounds)
/// - Π_mon's sumcheck inside Π_rg (log(m*d) rounds)
///
/// This matches Figure 3 Step-3: the two sumchecks share the verifier challenge stream prefix.
///
/// NOTE: implemented for the *vector-only* Π_mon used by Π_rg (our current prototype).
pub fn prove_pi_gr1cs_shared_sumcheck<R: CoeffRing>(
    transcript: &mut impl Transcript<R>,
    M: [&SparseMatrix<R>; 3],
    cm_f: &[R],
    f: &[R],
    rg_params: RPParams,
) -> Result<PiGr1csProof<R>, PiGr1csError<R>>
where
    R::BaseRing: Zq + Decompose,
{
    // -----------------
    // Π_had coins + MLEs (pre-sumcheck)
    // -----------------
    let m = M[0].nrows;
    let n = M[0].ncols;
    assert_eq!(f.len(), n);
    assert!(m.is_power_of_two());
    let log_m = log2(m) as usize;
    let d = R::dimension();

    transcript.absorb_field_element(&R::BaseRing::from(0x4c465053_50494841u128)); // "LFPS_PIHA"
    let s_base = transcript.get_challenges(log_m);
    let alpha_base = transcript.get_challenge();

    let s_r: Vec<R> = s_base.iter().copied().map(R::from).collect();
    let eq_mle = build_eq_x_r::<R>(&s_r).expect("build_eq_x_r failed");

    let y = M
        .iter()
        .map(|Mi| Mi.try_mul_vec(f).expect("mat-vec mul failed"))
        .collect::<Vec<Vec<R>>>();

    let mut mles_had: Vec<DenseMultilinearExtension<R>> = Vec::with_capacity(1 + 3 * d);
    mles_had.push(eq_mle);
    for i in 0..3 {
        for j in 0..d {
            let evals = (0..m)
                .map(|row| R::from(y[i][row].coeffs()[j]))
                .collect::<Vec<_>>();
            mles_had.push(DenseMultilinearExtension::from_evaluations_vec(log_m, evals));
        }
    }

    let mut alpha_pows = Vec::with_capacity(d);
    let mut pow = R::BaseRing::ONE;
    for _ in 0..d {
        alpha_pows.push(R::from(pow));
        pow *= alpha_base;
    }
    let comb_had = move |vals: &[R]| -> R {
        let eq = vals[0];
        let mut acc = R::ZERO;
        for j in 0..d {
            let g1 = vals[1 + j];
            let g2 = vals[1 + d + j];
            let g3 = vals[1 + 2 * d + j];
            acc += alpha_pows[j] * eq * (g1 * g2 - g3);
        }
        acc
    };

    // We'll need the same MLEs after sumcheck to compute U.
    let mles_had_for_u = mles_had.clone();

    // -----------------
    // Π_rg coins + Π_mon MLEs (pre-sumcheck)
    // -----------------
    let n_f = f.len();
    assert!(n_f % rg_params.l_h == 0, "l_h must divide n");

    transcript.absorb_slice(cm_f);
    let J = derive_J::<R>(transcript, rg_params.lambda_pj, rg_params.l_h);
    for row in &J {
        for x in row {
            transcript.absorb_field_element(x);
        }
    }

    let blocks = n_f / rg_params.l_h;
    let m_j = blocks * rg_params.lambda_pj;
    let m_rg = m; // Π_gr1cs uses m rows, with m_J <= m
    assert!(m_rg >= m_j);
    assert_eq!(m_rg % m_j, 0);

    // cf(f) as n×d base-ring matrix
    let mut cf = vec![vec![R::BaseRing::ZERO; d]; n_f];
    for (row, r) in cf.iter_mut().zip(f.iter()) {
        for (j, c) in r.coeffs().iter().enumerate() {
            row[j] = *c;
        }
    }

    // H on m_J rows
    let mut H = vec![vec![R::BaseRing::ZERO; d]; m_j];
    for b in 0..blocks {
        for i in 0..rg_params.lambda_pj {
            let out_row = b * rg_params.lambda_pj + i;
            for t in 0..rg_params.l_h {
                let in_row = b * rg_params.l_h + t;
                let coef = J[i][t];
                for col in 0..d {
                    H[out_row][col] += coef * cf[in_row][col];
                }
            }
        }
    }

    // Digit decomposition H = Σ (d')^i H^(i)
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

    // Lift to m by replication along the extra s̄ dimension.
    let expand_row = |row: usize| -> usize { row % m_j };

    // g^(i) vectors over m*d entries, flattened column-major (idx = col*m + row)
    let mut g: Vec<Vec<R>> = Vec::with_capacity(rg_params.k_g);
    for i in 0..rg_params.k_g {
        let mut gi = Vec::with_capacity(m_rg * d);
        for c in 0..d {
            for r in 0..m_rg {
                gi.push(exp::<R>(H_digits[i][expand_row(r)][c]).expect("Exp failed"));
            }
        }
        g.push(gi);
    }

    let g_len = m_rg * d;
    assert!(g_len.is_power_of_two(), "setchk vector-set length must be power-of-two");
    let g_nvars = log2(g_len) as usize;
    assert_eq!(g_nvars, log_m + log2(d.next_power_of_two()) as usize);

    // setchk vector-only pre-sumcheck coins
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

    // Π_mon sumcheck MLEs (vector-only): [m_j, m'_j, eq] per digit
    let tnvars = g_nvars;
    let mut mles_mon: Vec<DenseMultilinearExtension<R>> = Vec::with_capacity(rg_params.k_g * 3);
    let alphas = cba.iter().map(|(_, _, a)| *a).collect::<Vec<_>>();
    for (i, (_c, beta, _alpha)) in cba.iter().enumerate() {
        let m_j_evals = g[i].iter().map(|r| R::from(ev(r, *beta))).collect::<Vec<_>>();
        let m_prime = m_j_evals.iter().map(|z| *z * z).collect::<Vec<_>>();
        mles_mon.push(DenseMultilinearExtension::from_evaluations_vec(tnvars, m_j_evals));
        mles_mon.push(DenseMultilinearExtension::from_evaluations_vec(tnvars, m_prime));
        let eq = build_eq_x_r(&_c).unwrap();
        mles_mon.push(eq);
    }

    let comb_mon = move |vals: &[R]| -> R {
        let mut lc = R::ZERO;
        // Precompute rc^i iteratively to avoid repeated pow().
        let mut rc_pow = R::BaseRing::ONE;
        for i in 0..alphas.len() {
            let s = i * 3;
            let b_claim = vals[s] * vals[s] - vals[s + 1];
            let mut res = b_claim * R::from(alphas[i]);
            res *= vals[s + 2]; // eq
            lc += if let Some(rc) = &rc {
                let scaled = res * R::from(rc_pow);
                rc_pow *= *rc;
                scaled
            } else {
                res
            };
        }
        lc
    };

    // -----------------
    // Shared sumchecks (Figure 3 Step-3) with hook at |r̄|=log(m_J)
    // -----------------
    let hook_round = log2(m_j.next_power_of_two()) as usize;
    let mut v_digits: Option<Vec<Vec<R::BaseRing>>> = None;
    let ((had_sc, had_ps), (mon_sc, mon_ps)) =
        MLSumcheck::<R, _>::prove_two_as_subprotocol_shared_with_hook(
            transcript,
            // A: Π_had
            mles_had,
            log_m,
            3,
            comb_had,
            // B: Π_mon
            mles_mon,
            g_nvars,
            3,
            comb_mon,
            // hook
            hook_round,
            |t, sampled_r| {
                // sampled_r has length == log(m_J)
                let ts_r_full = ts_weights(sampled_r);
                let ts_r = &ts_r_full[..m_j];
                let mut vd = Vec::with_capacity(rg_params.k_g);
                for i in 0..rg_params.k_g {
                    let mut v_i = vec![R::BaseRing::ZERO; d];
                    for row in 0..m_j {
                        let w = ts_r[row];
                        for col in 0..d {
                            v_i[col] += H_digits[i][row][col] * w;
                        }
                    }
                    vd.push(v_i);
                }
                for v_i in &vd {
                    for x in v_i {
                        t.absorb_field_element(x);
                    }
                }
                v_digits = Some(vd);
            },
        );

    let v_digits = v_digits.expect("hook did not run");

    // -----------------
    // Finish Π_had: compute U = g(r) at the shared point and absorb it.
    // -----------------
    let r_had = had_ps.randomness.clone();
    let r_poly: Vec<R> = r_had.iter().copied().map(R::from).collect();
    let mut U: [Vec<R>; 3] = [Vec::with_capacity(d), Vec::with_capacity(d), Vec::with_capacity(d)];
    for i in 0..3 {
        for j in 0..d {
            let idx = 1 + i * d + j;
            let v = mles_had_for_u[idx]
                .evaluate(&r_poly)
                .expect("MLE evaluate returned None");
            U[i].push(v);
        }
    }
    transcript.absorb_slice(&U[0]);
    transcript.absorb_slice(&U[1]);
    transcript.absorb_slice(&U[2]);
    let had = PiHadProof {
        log_m,
        sumcheck: had_sc,
        U,
    };

    // -----------------
    // Finish Π_mon: compute b evaluations and absorb them.
    // -----------------
    let r_mon = mon_ps.randomness.clone();
    let r_poly_mon: Vec<R> = r_mon.iter().copied().map(R::from).collect();
    let b: Vec<R> = g
        .iter()
        .map(|mvec| {
            let mle = DenseMultilinearExtension::from_evaluations_slice(tnvars, mvec);
            mle.evaluate(&r_poly_mon).unwrap()
        })
        .collect();

    transcript.absorb_slice(&b);

    let mon = SetChkOut {
        nvars: g_nvars,
        r: r_mon,
        sumcheck_proof: mon_sc,
        // vector-only: no matrix evaluations
        e: vec![vec![]],
        b,
    };

    let rg = RPRangeProof {
        params: rg_params,
        m_j,
        m: m_rg,
        J,
        mon,
        v_digits,
    };

    Ok(PiGr1csProof { had, rg })
}

/// Verify the shared-sumcheck Π_gr1cs proof and return the reconstructed output `(r, U_ct, v)`
/// for Π_had and `(r, u, v)` for Π_rg.
pub fn verify_pi_gr1cs_shared_sumcheck_and_output<R: CoeffRing>(
    transcript: &mut impl Transcript<R>,
    cm_f: &[R],
    proof: &PiGr1csProof<R>,
) -> Result<PiGr1csVerifiedOutput<R>, PiGr1csError<R>>
where
    R::BaseRing: Zq,
{
    // -----------------
    // Replay pre-sumcheck transcript for Π_had and Π_rg.
    // -----------------
    let log_m = proof.had.log_m;

    transcript.absorb_field_element(&R::BaseRing::from(0x4c465053_50494841u128)); // "LFPS_PIHA"
    let s_base = transcript.get_challenges(log_m);
    let alpha_base = transcript.get_challenge();

    transcript.absorb_slice(cm_f);
    let J = derive_J::<R>(transcript, proof.rg.params.lambda_pj, proof.rg.params.l_h);
    if J != proof.rg.J {
        return Err(PiGr1csError::Rg(RPConsistencyError::JMismatch));
    }
    for row in &J {
        for x in row {
            transcript.absorb_field_element(x);
        }
    }

    // setchk vector-only pre-sumcheck coins
    let nvars_b = proof.rg.mon.nvars;
    let mut cba: Vec<(Vec<R>, R::BaseRing, R::BaseRing)> = Vec::with_capacity(proof.rg.params.k_g);
    for _ in 0..proof.rg.params.k_g {
        let c: Vec<R> = transcript
            .get_challenges(nvars_b)
            .into_iter()
            .map(|x| x.into())
            .collect();
        let beta = transcript.get_challenge();
        let alpha = transcript.get_challenge();
        cba.push((c, beta, alpha));
    }
    let rc: Option<R::BaseRing> = (proof.rg.params.k_g > 1).then(|| transcript.get_challenge());

    // -----------------
    // Shared sumcheck verification with hook absorbing v_digits.
    // -----------------
    let hook_round = log2(proof.rg.m_j.next_power_of_two()) as usize;
    let (had_sc, mon_sc) = MLSumcheck::<R, _>::verify_two_as_subprotocol_shared_with_hook(
        transcript,
        // A: Π_had
        log_m,
        3,
        R::ZERO,
        &proof.had.sumcheck,
        // B: Π_mon
        nvars_b,
        3,
        R::ZERO,
        &proof.rg.mon.sumcheck_proof,
        // hook
        hook_round,
        |t, _sampled| {
            for v_i in &proof.rg.v_digits {
                for x in v_i {
                    t.absorb_field_element(x);
                }
            }
        },
    )?;

    // Enforce that the stored points match the verified ones.
    if proof.rg.mon.r != mon_sc.point {
        return Err(PiGr1csError::SharedPointMismatch);
    }

    // -----------------
    // Π_had Eq. (26) check + output reconstruction.
    // -----------------
    let d = R::dimension();
    let s_poly: Vec<R> = s_base.iter().copied().map(R::from).collect();
    let r_poly: Vec<R> = had_sc.point.iter().copied().map(R::from).collect();
    let eq_sr = eq_eval(&s_poly, &r_poly).map_err(SumCheckError::EvaluationError)?;

    let mut pow = R::BaseRing::ONE;
    let mut acc = R::ZERO;
    for j in 0..d {
        let alpha_pow = R::from(pow);
        acc += alpha_pow * (proof.had.U[0][j] * proof.had.U[1][j] - proof.had.U[2][j]);
        pow *= alpha_base;
    }
    let lhs = eq_sr * acc;
    let rhs = had_sc.expected_evaluation;
    if lhs != rhs {
        return Err(SumCheckError::SumCheckFailed(lhs, rhs).into());
    }

    let had_out = {
        let r = had_sc.point;
        let U_ct: [Vec<R::BaseRing>; 3] = [
            proof.had.U[0].iter().map(|x| x.ct()).collect(),
            proof.had.U[1].iter().map(|x| x.ct()).collect(),
            proof.had.U[2].iter().map(|x| x.ct()).collect(),
        ];
        let mut v: [R; 3] = [R::ZERO, R::ZERO, R::ZERO];
        for i in 0..3 {
            for j in 0..d {
                v[i].coeffs_mut()[j] = U_ct[i][j];
            }
        }

        // absorb U after sumcheck, matching the prover
        transcript.absorb_slice(&proof.had.U[0]);
        transcript.absorb_slice(&proof.had.U[1]);
        transcript.absorb_slice(&proof.had.U[2]);

        PiHadVerifiedOutput { r, U_ct, v }
    };

    // -----------------
    // Π_mon recomputation check (vector-only), then absorb b (matching the prover).
    // -----------------
    transcript.absorb_slice(&proof.rg.mon.b);

    let r_mon_r: Vec<R> = mon_sc.point.iter().copied().map(R::from).collect();
    let v_expected = mon_sc.expected_evaluation;

    let mut ver = R::ZERO;
    // Precompute rc^i iteratively to avoid repeated pow().
    let mut rc_pow = R::BaseRing::ONE;
    for (i, b_i) in proof.rg.mon.b.iter().enumerate() {
        let (c, beta, alpha) = &cba[i];
        let eq = eq_eval(c, &r_mon_r).unwrap();

        let ev1 = R::from(ev(b_i, *beta));
        let ev2 = R::from(ev(b_i, *beta * *beta));
        let b_claim = ev1 * ev1 - ev2;

        if let Some(rc) = &rc {
            ver += eq * R::from(*alpha) * b_claim * R::from(rc_pow);
            rc_pow *= *rc;
        } else {
            ver += eq * R::from(*alpha) * b_claim;
        }
    }

    if ver != v_expected {
        return Err(PiGr1csError::SharedMonomialMismatch);
    }

    // -----------------
    // Π_rg Step-5 consistency + output reconstruction.
    // -----------------
    // Step-5 check uses u(i)=mon.b and v_digits, and depends on s challenges inside r'.
    // We reproduce it directly here.
    let log_m = log2(proof.rg.m.next_power_of_two()) as usize;
    let s_chals = proof.rg.mon.r[log_m..].to_vec();
    let ts_s_full = ts_weights(&s_chals);
    let ts_s = &ts_s_full[..d];

    for i in 0..proof.rg.params.k_g {
        let u_i = proof.rg.mon.b[i];
        let lhs = (psi::<R>() * u_i).ct();
        let rhs = proof.rg.v_digits[i]
            .iter()
            .zip(ts_s.iter())
            .fold(R::BaseRing::ZERO, |acc, (&vij, &t)| acc + vij * t);
        if lhs != rhs {
            return Err(PiGr1csError::Rg(RPConsistencyError::Step5Mismatch {
                idx: i,
                lhs: format!("{:?}", lhs),
                rhs: format!("{:?}", rhs),
            }));
        }
    }

    let log_mj = log2(proof.rg.m_j.next_power_of_two()) as usize;
    let r_prime = proof.rg.mon.r.clone();
    let rg_r = r_prime[..log_mj].to_vec();
    let rg_v = compose_v_digits::<R>(&proof.rg.v_digits, proof.rg.params.d_prime);
    let rg_u = proof.rg.mon.b.clone();

    Ok(PiGr1csVerifiedOutput {
        had: had_out,
        rg_r,
        rg_v,
        rg_u,
    })
}
