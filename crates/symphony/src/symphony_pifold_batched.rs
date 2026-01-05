//! Symphony Π_fold (Figure 4) — **batched sumcheck** prototype.
//!
//! This is a step beyond `symphony_pifold`:
//! - We compress a batch of ℓ Π_gr1cs instances into **one** Π_had sumcheck proof
//!   and **one** Π_mon sumcheck proof by batching with transcript-derived weights ρ.
//! - The sumchecks share verifier challenges exactly (Figure 3 Step-3 schedule).
//!
//! Notes:
//! - We still recompute some per-instance values during verification from witness openings
//!   (correctness-first; not ZK yet).
//! - Full paper compression replaces these with commitment+opening checks (`VfyOpen`).

use ark_std::log2;
use cyclotomic_rings::rings::GetPoseidonParams;
use latticefold::{
    commitment::AjtaiCommitmentScheme,
    transcript::Transcript,
    utils::sumcheck::{
        utils::{build_eq_x_r, eq_eval},
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
    recording_transcript::RecordingTranscriptRef,
    rp_rgchk::{compose_v_digits, RPParams},
    symphony_cm::SymphonyCoins,
    symphony_fold::{SymphonyBatchLin, SymphonyInstance},
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

/// Prover (paper-faithful FS): run once under Poseidon-FS and **record** the coin stream.
///
/// This avoids the 2× work of `prove_pi_fold_batched_sumcheck_fs` (record + fixed-coin replay),
/// while keeping a serializable `proof.coins` for WE/DPP frontends.
pub fn prove_pi_fold_batched_sumcheck_poseidon_fs<R: CoeffRing, PC>(
    M: [&SparseMatrix<R>; 3],
    cms: &[Vec<R>],
    witnesses: &[&[R]],
    public_inputs: &[R::BaseRing],
    cfs_had_u_scheme: Option<&AjtaiCommitmentScheme<R>>,
    cfs_mon_b_scheme: Option<&AjtaiCommitmentScheme<R>>,
    rg_params: RPParams,
) -> Result<PiFoldProverOutput<R>, String>
where
    R::BaseRing: Zq + Decompose,
    PC: GetPoseidonParams<<<R>::BaseRing as ark_ff::Field>::BasePrimeField>,
{
    if cms.is_empty() {
        return Err("PiFold: empty batch".to_string());
    }
    if cms.len() != witnesses.len() {
        return Err("PiFold: cms/witnesses length mismatch".to_string());
    }

    let mut ts = crate::transcript::PoseidonTranscript::<R>::empty::<PC>();
    let mut rts = RecordingTranscriptRef::<R, _>::new(&mut ts);
    rts.absorb_field_element(&R::BaseRing::from(0x4c465053_50494250u128)); // "LFPS_PIBP"
    absorb_public_inputs::<R>(&mut rts, public_inputs);

    let mut out = prove_pi_fold_batched_sumcheck(&mut rts, M, cms, witnesses, rg_params)
        .map_err(|e| format!("PiFold: poseidon-fs prove failed: {e}"))?;

    out.proof.coins = SymphonyCoins::<R> {
        challenges: rts.coins_challenges,
        bytes: rts.coins_bytes,
        events: rts.events,
    };

    // Optionally commit to CP transcript witness messages (WE/DPP-facing path).
    out.cfs_had_u.clear();
    out.cfs_mon_b.clear();
    match (cfs_had_u_scheme, cfs_mon_b_scheme) {
        (Some(had_s), Some(mon_s)) => {
            out.cfs_had_u = out
                .aux
                .had_u
                .iter()
                .map(|u| {
                    had_s
                        .commit(&encode_had_u_instance::<R>(u))
                        .map_err(|e| format!("PiFold: cfs_had_u commit failed: {e:?}"))
                        .map(|c| c.as_ref().to_vec())
                })
                .collect::<Result<Vec<_>, _>>()?;
            out.cfs_mon_b = out
                .aux
                .mon_b
                .iter()
                .map(|b| {
                    mon_s
                        .commit(b)
                        .map_err(|e| format!("PiFold: cfs_mon_b commit failed: {e:?}"))
                        .map(|c| c.as_ref().to_vec())
                })
                .collect::<Result<Vec<_>, _>>()?;
        }
        (None, None) => {}
        _ => {
            return Err(
                "PiFold: must provide both cfs_had_u_scheme and cfs_mon_b_scheme or neither".to_string(),
            );
        }
    }

    Ok(out)
}

/// Prover: derive a shared FS coin stream once, then produce a batched-sumcheck Π_fold proof.
pub fn prove_pi_fold_batched_sumcheck_fs<R: CoeffRing, PC>(
    M: [&SparseMatrix<R>; 3],
    cms: &[Vec<R>],
    witnesses: &[&[R]],
    public_inputs: &[R::BaseRing],
    cfs_had_u_scheme: Option<&AjtaiCommitmentScheme<R>>,
    cfs_mon_b_scheme: Option<&AjtaiCommitmentScheme<R>>,
    rg_params: RPParams,
) -> Result<PiFoldProverOutput<R>, String>
where
    R::BaseRing: Zq + Decompose,
    PC: GetPoseidonParams<<<R>::BaseRing as ark_ff::Field>::BasePrimeField>,
{
    // Historically this function did a record+replay (Poseidon transcript → FixedTranscript)
    // to enforce that the prover is a pure function of the coin stream. That costs ~2× time.
    // For production/streaming, use the one-pass Poseidon-FS prover.
    prove_pi_fold_batched_sumcheck_poseidon_fs::<R, PC>(
        M,
        cms,
        witnesses,
        public_inputs,
        cfs_had_u_scheme,
        cfs_mon_b_scheme,
        rg_params,
    )
}

/// Core prover under an arbitrary transcript (Poseidon or FixedTranscript).
pub fn prove_pi_fold_batched_sumcheck<R: CoeffRing>(
    transcript: &mut impl Transcript<R>,
    M: [&SparseMatrix<R>; 3],
    cms: &[Vec<R>],
    witnesses: &[&[R]],
    rg_params: RPParams,
) -> Result<PiFoldProverOutput<R>, String>
where
    R::BaseRing: Zq + Decompose,
{
    let ell = cms.len();
    let beta_cts = derive_beta_chi::<R>(transcript, ell);
    if ell == 0 || witnesses.len() != ell {
        return Err("PiFold: length mismatch".to_string());
    }

    // -----------------
    // Shared Π_had coins (Figure 1)
    // -----------------
    let m = M[0].nrows;
    if !m.is_power_of_two() {
        return Err("PiFold: m must be power-of-two".to_string());
    }
    let log_m = log2(m) as usize;
    let d = R::dimension();

    transcript.absorb_field_element(&R::BaseRing::from(0x4c465053_50494841u128)); // "LFPS_PIHA"
    let s_base = transcript.get_challenges(log_m);
    let alpha_base = transcript.get_challenge();

    let s_r: Vec<R> = s_base.iter().copied().map(R::from).collect();
    let eq_mle = build_eq_x_r::<R>(&s_r).expect("build_eq_x_r failed");

    let mut alpha_pows = Vec::with_capacity(d);
    let mut pow = R::BaseRing::ONE;
    for _ in 0..d {
        alpha_pows.push(R::from(pow));
        pow *= alpha_base;
    }

    // -----------------
    // Per-instance Π_rg coins and Π_mon setup
    // -----------------
    let mut H_digits_all: Vec<Vec<Vec<Vec<R::BaseRing>>>> = Vec::with_capacity(ell); // [ell][k_g][m_j][d]
    let mut cba_all: Vec<Vec<(Vec<R>, R::BaseRing, R::BaseRing)>> = Vec::with_capacity(ell);
    let mut rc_all: Vec<Option<R::BaseRing>> = Vec::with_capacity(ell);

    // This is used to build the monomial vector length.
    // Assume same n across instances.
    let n_f = witnesses[0].len();
    if n_f == 0 || n_f % rg_params.l_h != 0 {
        return Err("PiFold: invalid witness length".to_string());
    }
    let blocks = n_f / rg_params.l_h;
    let m_j = blocks * rg_params.lambda_pj;
    if m < m_j || m % m_j != 0 {
        return Err("PiFold: require m_J <= m and m multiple of m_J".to_string());
    }

    let g_len = m * d;
    if !g_len.is_power_of_two() {
        return Err("PiFold: require m*d power-of-two".to_string());
    }
    let g_nvars = log2(g_len) as usize;

    for (cm_f, f) in cms.iter().zip(witnesses.iter().copied()) {
        if f.len() != n_f {
            return Err("PiFold: inconsistent witness lengths".to_string());
        }

        // Bind statement commitment, derive and bind J.
        transcript.absorb_slice(cm_f);
        let J = derive_J::<R>(transcript, rg_params.lambda_pj, rg_params.l_h);
        for row in &J {
            for x in row {
                transcript.absorb_field_element(x);
            }
        }


        // Build H_digits for this instance (on m_J rows).
        let mut H = vec![vec![R::BaseRing::ZERO; d]; m_j];
        let Jref = &J;
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
        H_digits_all.push(H_digits);

        // setchk vector-only pre-sumcheck coins for this instance.
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

    // Batch weights ρ for instances (soundness combiner).
    let rhos = transcript
        .get_challenges(ell)
        .into_iter()
        .map(R::from)
        .collect::<Vec<R>>();

    // -----------------
    // Build batched Π_had sumcheck (one)
    // -----------------
    let had_mles_per = 1 + 3 * d;
    let mut mles_had_batched: Vec<DenseMultilinearExtension<R>> = Vec::with_capacity(ell * had_mles_per);

    // Precompute y vectors per instance.
    let ys = witnesses
        .iter()
        .copied()
        .map(|f| {
            M.iter()
                .map(|Mi| Mi.try_mul_vec(f).expect("mat-vec mul failed"))
                .collect::<Vec<Vec<R>>>()
        })
        .collect::<Vec<_>>();

    for inst_idx in 0..ell {
        mles_had_batched.push(eq_mle.clone());
        for i in 0..3 {
            for j in 0..d {
                let evals = (0..m)
                    .map(|row| R::from(ys[inst_idx][i][row].coeffs()[j]))
                    .collect::<Vec<_>>();
                mles_had_batched.push(DenseMultilinearExtension::from_evaluations_vec(log_m, evals));
            }
        }
    }

    let rhos_had = rhos.clone();
    let comb_had_batched = move |vals: &[R]| -> R {
        let mut acc_all = R::ZERO;
        for inst_idx in 0..ell {
            let base = inst_idx * had_mles_per;
            let eq = vals[base];
            let mut acc = R::ZERO;
            for j in 0..d {
                let g1 = vals[base + 1 + j];
                let g2 = vals[base + 1 + d + j];
                let g3 = vals[base + 1 + 2 * d + j];
                acc += alpha_pows[j] * eq * (g1 * g2 - g3);
            }
            acc_all += rhos_had[inst_idx] * acc;
        }
        acc_all
    };

    // -----------------
    // Build batched Π_mon sumcheck (one)
    // -----------------
    let mon_mles_per = 3 * rg_params.k_g;
    let mut mles_mon_batched: Vec<DenseMultilinearExtension<R>> = Vec::with_capacity(ell * mon_mles_per);

    // Also keep per-instance alphas in ring form.
    let alphas_ring = cba_all
        .iter()
        .map(|cba| cba.iter().map(|(_, _, a)| R::from(*a)).collect::<Vec<_>>())
        .collect::<Vec<_>>();

    // Build g vectors per instance and corresponding MLEs.
    let expand_row = |row: usize| -> usize { row % m_j };
    let mut g_all: Vec<Vec<Vec<R>>> = Vec::with_capacity(ell); // [ell][k_g][g_len]

    for inst_idx in 0..ell {
        let mut g_inst: Vec<Vec<R>> = Vec::with_capacity(rg_params.k_g);
        for dig in 0..rg_params.k_g {
            let mut gi = Vec::with_capacity(g_len);
            for c in 0..d {
                for r in 0..m {
                    gi.push(exp::<R>(H_digits_all[inst_idx][dig][expand_row(r)][c]).expect("Exp failed"));
                }
            }
            g_inst.push(gi);
        }
        g_all.push(g_inst);

        // For each digit, add (m_j, m'_j, eq(c)) MLEs.
        for dig in 0..rg_params.k_g {
            let (_c, beta_i, _a) = &cba_all[inst_idx][dig];
            let m_j_evals = g_all[inst_idx][dig]
                .iter()
                .map(|r| R::from(ev(r, *beta_i)))
                .collect::<Vec<_>>();
            let m_prime = m_j_evals.iter().map(|z| *z * z).collect::<Vec<_>>();
            mles_mon_batched.push(DenseMultilinearExtension::from_evaluations_vec(g_nvars, m_j_evals));
            mles_mon_batched.push(DenseMultilinearExtension::from_evaluations_vec(g_nvars, m_prime));
            let eq = build_eq_x_r(&cba_all[inst_idx][dig].0).unwrap();
            mles_mon_batched.push(eq);
        }
    }

    let comb_mon_batched = move |vals: &[R]| -> R {
        let mut acc_all = R::ZERO;
        for inst_idx in 0..ell {
            let mut lc = R::ZERO;
            // Precompute rc^dig iteratively to avoid repeated pow().
            let mut rc_pow = R::BaseRing::ONE;
            for dig in 0..rg_params.k_g {
                let base = inst_idx * mon_mles_per + dig * 3;
                let b_claim = vals[base] * vals[base] - vals[base + 1];
                let mut res = b_claim * alphas_ring[inst_idx][dig];
                res *= vals[base + 2];
                lc += if let Some(rc) = &rc_all[inst_idx] {
                    let scaled = res * R::from(rc_pow);
                    rc_pow *= *rc;
                    scaled
                } else {
                    res
                };
            }
            acc_all += rhos[inst_idx] * lc;
        }
        acc_all
    };

    // Hook: absorb all instances' v_digits after |r̄| = log(m_J).
    let hook_round = log2(m_j.next_power_of_two()) as usize;
    let mut v_digits_folded: Vec<Vec<R::BaseRing>> = vec![vec![R::BaseRing::ZERO; d]; rg_params.k_g];

    let ((had_sumcheck, had_ps), (mon_sumcheck, mon_ps)) = MLSumcheck::<R, _>::prove_two_as_subprotocol_shared_with_hook(
        transcript,
        mles_had_batched,
        log_m,
        3,
        comb_had_batched,
        mles_mon_batched,
        g_nvars,
        3,
        comb_mon_batched,
        hook_round,
        |t, sampled_r| {
            let ts_r_full = ts_weights(sampled_r);
            let ts_r = &ts_r_full[..m_j];
            // Compute folded v_digits* := Σ_i beta_i * v_digits_i  (beta_i are base-field scalars).
            for dig in 0..rg_params.k_g {
                for col in 0..d {
                    v_digits_folded[dig][col] = R::BaseRing::ZERO;
                }
            }
            for inst_idx in 0..ell {
                let b = beta_cts[inst_idx];
                for dig in 0..rg_params.k_g {
                    for row in 0..m_j {
                        let w = ts_r[row];
                        for col in 0..d {
                            v_digits_folded[dig][col] += b * H_digits_all[inst_idx][dig][row][col] * w;
                        }
                    }
                }
            }
            // Absorb only the folded message (protocol defines a single hook message).
            for v_i in &v_digits_folded {
                    for x in v_i {
                        t.absorb_field_element(x);
                    }
            }
        },
    );

    // Finish: compute per-instance U at had point, and per-instance b at mon point, then absorb them.
    // These are the CP transcript witness messages for the WE/DPP-facing path.
    let r_had = had_ps.randomness.clone();
    let ts_r_had = ts_weights(&r_had);

    let mut had_u: Vec<[Vec<R::BaseRing>; 3]> = Vec::with_capacity(ell);
    for inst_idx in 0..ell {
        let mut U: [Vec<R::BaseRing>; 3] = [Vec::with_capacity(d), Vec::with_capacity(d), Vec::with_capacity(d)];
        for i in 0..3 {
            for j in 0..d {
                let mut acc = R::BaseRing::ZERO;
                for row in 0..m {
                    acc += ts_r_had[row] * ys[inst_idx][i][row].coeffs()[j];
                }
                U[i].push(acc);
            }
        }
        for x in &U[0] { transcript.absorb_field_element(x); }
        for x in &U[1] { transcript.absorb_field_element(x); }
        for x in &U[2] { transcript.absorb_field_element(x); }
        had_u.push(U);
    }

    // ys is no longer needed past here; free it before monomial-side work.
    drop(ys);

    let r_mon = mon_ps.randomness.clone();
    let r_poly_mon: Vec<R> = r_mon.iter().copied().map(R::from).collect();

    let mut mon_b: Vec<Vec<R>> = Vec::with_capacity(ell);
    for inst_idx in 0..ell {
        let mut b_inst = Vec::with_capacity(rg_params.k_g);
        for dig in 0..rg_params.k_g {
            let mle = DenseMultilinearExtension::from_evaluations_slice(g_nvars, &g_all[inst_idx][dig]);
            b_inst.push(mle.evaluate(&r_poly_mon).unwrap());
        }
        transcript.absorb_slice(&b_inst);
        mon_b.push(b_inst);
    }

    Ok(PiFoldProverOutput {
        proof: PiFoldBatchedProof {
        coins: SymphonyCoins { challenges: vec![], bytes: vec![], events: vec![] },
        rg_params,
            m_j,
            m,
            v_digits_folded,
        had_sumcheck,
        mon_sumcheck,
        },
        aux: PiFoldAuxWitness { had_u, mon_b },
        cfs_had_u: vec![],
        cfs_mon_b: vec![],
    })
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

/// Verifier (paper-faithful FS): recompute all challenges by hashing the transcript (Poseidon sponge).
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
    verify_pi_fold_batched_and_fold_outputs_with_openings(&mut ts, M, cms, proof, open, cms_openings)
}

/// Derive the auxiliary transcript messages (`had_u`, `mon_b`) from witness openings under the
/// **paper-faithful** Poseidon-FS transcript schedule.
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
        for row in &J {
            for x in row {
                ts.absorb_field_element(x);
            }
        }
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
        &vec![Vec::<R>::new(); ell], // no witness openings needed; lengths ignored when aux present
        Some(aux),
    )?;

    // Verify CP commitment openings (domain-separated by `VfyOpen`).
    //
    // Note: we intentionally do NOT absorb `cfs_*` into the Poseidon-FS transcript used by Π_fold:
    // these messages are post-challenge in the schedule, and hashing them would only add cost
    // without strengthening soundness for the DPP target.
    for i in 0..ell {
        let had_u_enc = encode_had_u_instance::<R>(&aux.had_u[i]);
        open.verify_opening(&mut ts, "cfs_had_u", &cfs_had_u[i], &[], &had_u_enc, &[])?;
        open.verify_opening(&mut ts, "cfs_mon_b", &cfs_mon_b[i], &[], &aux.mon_b[i], &[])?;
    }

    Ok(out)
}

/// Relation-check style wrapper for GM‑1 / DPP frontends:
/// returns `Ok(())` iff the folding verifier accepts under paper-faithful FS (Poseidon transcript).
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
        transcript,
        M,
        cms,
        proof,
        open,
        cms_openings,
        None,
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
    if cms.len() != cms_openings.len() {
        return Err("PiFold: cms/openings length mismatch".to_string());
    }

    // Commitment-opening layer is verified **outside** the transcript schedule used for the
    // folding/sumcheck subprotocols. Our AjtaiOpenVerifier ignores the transcript argument,
    // matching the CP-SNARK intent.
    for (cm, open_val) in cms.iter().zip(cms_openings.iter()) {
        open.verify_opening(transcript, "cm_witness", cm, &[], open_val, &[])?;
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
        for row in &J {
            for x in row {
                transcript.absorb_field_element(x);
            }
        }
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
