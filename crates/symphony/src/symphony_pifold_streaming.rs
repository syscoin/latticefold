//! Streaming version of Symphony Π_fold prover.
//!
//! This module provides a memory-efficient version of the batched Π_fold prover
//! that uses streaming MLEs instead of materializing O(2^n) evaluation vectors.
//!
//! ## Key Optimizations
//!
//! 1. **Hadamard MLEs**: `SparseMatrixMle` computes y[row] = (M*w)[row] on demand
//! 2. **Eq MLEs**: `EqStreamingMle` computes eq(bits(idx), r) directly
//! 3. **Monomial MLEs**: Pre-computed per-digit (still dense, but smaller)
//!
//! ## Memory Comparison (1M constraints, d=16)
//!
//! | Component | Dense | Streaming |
//! |-----------|-------|-----------|
//! | y = M*w MLEs | 3 × 16MB | O(1) per eval |
//! | eq MLE | 16MB | O(1) per eval |
//! | m_j/m' MLEs | 2 × k_g × 256MB | Same (complex) |

use ark_std::log2;
use std::sync::Arc;
use stark_rings::{
    balanced_decomposition::{Decompose, DecomposeToVec},
    exp, CoeffRing, OverField, Ring, Zq,
};
use stark_rings_linalg::SparseMatrix;
use stark_rings_poly::mle::DenseMultilinearExtension;

use latticefold::{
    transcript::Transcript,
    utils::sumcheck::utils::build_eq_x_r,
};

use crate::mle_oracle::{EqStreamingMle, SparseMatrixMle};
use crate::rp_rgchk::RPParams;
use crate::streaming_sumcheck::{StreamingMle, StreamingProof, StreamingSumcheck};
use crate::symphony_cm::SymphonyCoins;
use crate::symphony_coins::{derive_beta_chi, derive_J, ev, ts_weights};
use crate::symphony_pifold_batched::{PiFoldAuxWitness, PiFoldBatchedProof, PiFoldProverOutput};

// Re-export streaming types
pub use crate::mle_oracle::ClosureMle;
pub use crate::streaming_sumcheck::{StreamingProverMsg, StreamingSumcheckState};

/// Streaming Π_fold prover.
///
/// This is a memory-efficient version of `prove_pi_fold_batched_sumcheck` that uses
/// streaming MLEs for the Hadamard side. The monomial side still uses dense MLEs
/// due to the complexity of the digit decomposition.
///
/// ## When to Use
///
/// Use this when:
/// - Memory is constrained (< 32GB for large instances)
/// - The sparse matrices are very sparse (most rows have few nonzeros)
///
/// Use the dense version when:
/// - Memory is plentiful
/// - Maximum throughput is needed (dense has better cache locality)
pub fn prove_pi_fold_streaming<R: CoeffRing>(
    transcript: &mut impl Transcript<R>,
    M: [Arc<SparseMatrix<R>>; 3],
    cms: &[Vec<R>],
    witnesses: &[Arc<Vec<R>>],
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
    // Shared Π_had coins
    // -----------------
    let m = M[0].nrows;
    if !m.is_power_of_two() {
        return Err("PiFold: m must be power-of-two".to_string());
    }
    let log_m = log2(m) as usize;
    let d = R::dimension();

    transcript.absorb_field_element(&R::BaseRing::from(0x4c465053_50494841u128));
    let s_base = transcript.get_challenges(log_m);
    let alpha_base = transcript.get_challenge();

    let s_r: Vec<R> = s_base.iter().copied().map(R::from).collect();

    let mut alpha_pows = Vec::with_capacity(d);
    let mut pow = R::BaseRing::ONE;
    for _ in 0..d {
        alpha_pows.push(R::from(pow));
        pow *= alpha_base;
    }

    // -----------------
    // Per-instance coins
    // -----------------
    let mut cba_all: Vec<Vec<(Vec<R>, R::BaseRing, R::BaseRing)>> = Vec::with_capacity(ell);
    let mut rc_all: Vec<Option<R::BaseRing>> = Vec::with_capacity(ell);
    let mut Js: Vec<Vec<Vec<R::BaseRing>>> = Vec::with_capacity(ell);

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

    for (cm_f, f) in cms.iter().zip(witnesses.iter()) {
        if f.len() != n_f {
            return Err("PiFold: inconsistent witness lengths".to_string());
        }

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

    // -----------------
    // STREAMING Hadamard MLEs
    // -----------------
    // Key optimization: use SparseMatrixMle instead of materializing y = M*w
    let mut mles_had: Vec<Box<dyn StreamingMle<R>>> = Vec::with_capacity(ell * 4);

    for inst_idx in 0..ell {
        mles_had.push(Box::new(EqStreamingMle::new(s_r.clone())));
        for i in 0..3 {
            mles_had.push(Box::new(SparseMatrixMle::new(
                M[i].clone(),
                witnesses[inst_idx].clone(),
            )));
        }
    }

    // Hadamard combiner (must match Π_had: coefficient-wise constraint with alpha powers)
    let rhos_had = rhos.clone();
    let comb_had = move |vals: &[R]| -> R {
        let mut acc_all = R::ZERO;
        for inst_idx in 0..ell {
            let base = inst_idx * 4;
            let eq = vals[base];
            let y1 = &vals[base + 1];
            let y2 = &vals[base + 2];
            let y3 = &vals[base + 3];
            let mut acc = R::ZERO;
            for j in 0..d {
                let term = y1.coeffs()[j] * y2.coeffs()[j] - y3.coeffs()[j];
                acc += alpha_pows[j] * eq * R::from(term);
            }
            acc_all += rhos_had[inst_idx] * acc;
        }
        acc_all
    };

    // -----------------
    // Monomial MLEs (still dense - complex digit structure)
    // -----------------
    let mon_mles_per = 3 * rg_params.k_g;
    let mut mles_mon_batched: Vec<DenseMultilinearExtension<R>> =
        Vec::with_capacity(ell * mon_mles_per);

    let alphas_ring = cba_all
        .iter()
        .map(|cba| cba.iter().map(|(_, _, a)| R::from(*a)).collect::<Vec<_>>())
        .collect::<Vec<_>>();

    // Helper: compute projected row digits
    let proj_row_digits = |f: &[R], J: &[Vec<R::BaseRing>], out_row: usize| -> Vec<Vec<R::BaseRing>> {
        let i = out_row % rg_params.lambda_pj;
        let b = out_row / rg_params.lambda_pj;
        let mut h_row = vec![R::BaseRing::ZERO; d];
        for t in 0..rg_params.l_h {
            let in_row = b * rg_params.l_h + t;
            if in_row >= f.len() {
                continue;
            }
            let coef = J[i][t];
            let coeffs = f[in_row].coeffs();
            for col in 0..d {
                h_row[col] += coef * coeffs[col];
            }
        }
        h_row.decompose_to_vec(rg_params.d_prime, rg_params.k_g)
    };

    let reps = m / m_j;

    for inst_idx in 0..ell {
        let f: &[R] = &witnesses[inst_idx];
        let J = &Js[inst_idx];

        for dig in 0..rg_params.k_g {
            let (_c, beta_i, _a) = &cba_all[inst_idx][dig];

            let mut m_j_evals: Vec<R> = vec![R::ZERO; g_len];
            let mut m_prime: Vec<R> = vec![R::ZERO; g_len];

            for out_row in 0..m_j {
                let digits = proj_row_digits(f, J, out_row);
                for col in 0..d {
                    let g = exp::<R>(digits[col][dig]).expect("Exp failed");
                    let mj_ring = R::from(ev(&g, *beta_i));
                    let mp_ring = mj_ring * mj_ring;
                    for rep in 0..reps {
                        let r = out_row + rep * m_j;
                        let idx = col * m + r;
                        m_j_evals[idx] = mj_ring;
                        m_prime[idx] = mp_ring;
                    }
                }
            }

            mles_mon_batched.push(DenseMultilinearExtension::from_evaluations_vec(
                g_nvars, m_j_evals,
            ));
            mles_mon_batched.push(DenseMultilinearExtension::from_evaluations_vec(
                g_nvars, m_prime,
            ));
            let eq = build_eq_x_r(&cba_all[inst_idx][dig].0).unwrap();
            mles_mon_batched.push(eq);
        }
    }

    let rc_all_clone = rc_all.clone();
    let comb_mon = move |vals: &[R]| -> R {
        let mut acc_all = R::ZERO;
        for inst_idx in 0..ell {
            let mut lc = R::ZERO;
            let mut rc_pow = R::BaseRing::ONE;
            for dig in 0..rg_params.k_g {
                let base = inst_idx * mon_mles_per + dig * 3;
                let b_claim = vals[base] * vals[base] - vals[base + 1];
                let mut res = b_claim * alphas_ring[inst_idx][dig];
                res *= vals[base + 2];
                lc += if let Some(rc) = &rc_all_clone[inst_idx] {
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

    // -----------------
    // Run sumchecks with SHARED challenges (must match verifier schedule)
    // -----------------
    use latticefold::utils::sumcheck::verifier::VerifierMsg;
    use latticefold::utils::sumcheck::IPForMLSumcheck;

    let hook_round = log2(m_j.next_power_of_two()) as usize;
    let mut v_digits_folded: Vec<Vec<R::BaseRing>> =
        vec![vec![R::BaseRing::ZERO; d]; rg_params.k_g];

    // Sumcheck transcript header (matches latticefold verify_two_as_subprotocol_shared_with_hook)
    transcript.absorb(&R::from(log_m as u128));
    transcript.absorb(&R::from(3u128));
    transcript.absorb(&R::from(g_nvars as u128));
    transcript.absorb(&R::from(3u128));

    let mut had_state = StreamingSumcheck::prover_init(mles_had, log_m, 3);
    let mut mon_state = IPForMLSumcheck::<R, crate::transcript::PoseidonTranscript<R>>::prover_init(
        mles_mon_batched,
        g_nvars,
        3,
    );

    let mut had_msgs = Vec::with_capacity(log_m);
    let mut mon_msgs = Vec::with_capacity(g_nvars);
    let mut sampled: Vec<R::BaseRing> = Vec::with_capacity(log_m.max(g_nvars));
    let mut v_msg_had: Option<R::BaseRing> = None;
    let mut v_msg_mon: Option<VerifierMsg<R>> = None;

    let rounds = log_m.max(g_nvars);
    for round_idx in 0..rounds {
        // Prover messages (in the same order as latticefold's shared schedule)
        if round_idx < log_m {
            let pm_had = StreamingSumcheck::prove_round(&mut had_state, v_msg_had, &comb_had);
            transcript.absorb_slice(&pm_had.evaluations);
            had_msgs.push(pm_had);
        }
        if round_idx < g_nvars {
            let pm_mon = IPForMLSumcheck::<R, crate::transcript::PoseidonTranscript<R>>::prove_round(
                &mut mon_state,
                &v_msg_mon,
                &comb_mon,
            );
            transcript.absorb_slice(&pm_mon.evaluations);
            mon_msgs.push(pm_mon);
        }

        // Shared verifier randomness
        let r = transcript.get_challenge();
        transcript.absorb(&R::from(r));
        sampled.push(r);

        if hook_round != 0 && sampled.len() == hook_round {
            // Bind folded v_digits* after |r̄| challenges, before continuing.
            let ts_r_full = ts_weights(&sampled);
            let ts_r = &ts_r_full[..m_j];
            for dig in 0..rg_params.k_g {
                for col in 0..d {
                    v_digits_folded[dig][col] = R::BaseRing::ZERO;
                }
            }
            for inst_idx in 0..ell {
                let b = beta_cts[inst_idx];
                let f: &[R] = &witnesses[inst_idx];
                let J = &Js[inst_idx];
                for row in 0..m_j {
                    let w = ts_r[row];
                    let digits = proj_row_digits(f, J, row);
                    for col in 0..d {
                        for dig in 0..rg_params.k_g {
                            v_digits_folded[dig][col] += b * digits[col][dig] * w;
                        }
                    }
                }
            }
            for v_i in &v_digits_folded {
                for x in v_i {
                    transcript.absorb_field_element(x);
                }
            }
        }

        v_msg_had = Some(r);
        v_msg_mon = Some(VerifierMsg { randomness: r });
    }

    // Final randomness vectors for each sumcheck (same convention as latticefold sumcheck state)
    let had_rand = sampled[..log_m].to_vec();
    let mon_rand = sampled[..g_nvars].to_vec();
    had_state.randomness = had_rand.clone();
    mon_state.randomness = mon_rand.clone();

    let had_sumcheck = convert_streaming_proof(&StreamingProof(had_msgs));
    let mon_sumcheck = latticefold::utils::sumcheck::Proof::new(mon_msgs);

    // -----------------
    // Compute aux witness
    // -----------------
    let ts_r_had = ts_weights(&had_rand);

    let mut had_u: Vec<[Vec<R::BaseRing>; 3]> = Vec::with_capacity(ell);
    for inst_idx in 0..ell {
        let mut U: [Vec<R::BaseRing>; 3] = [
            Vec::with_capacity(d),
            Vec::with_capacity(d),
            Vec::with_capacity(d),
        ];

        for i in 0..3 {
            for j in 0..d {
                let mut acc = R::BaseRing::ZERO;
                for row in 0..m {
                    let mut y_row = R::ZERO;
                    for (coeff, col_idx) in &M[i].coeffs[row] {
                        if *col_idx < witnesses[inst_idx].len() {
                            y_row += *coeff * witnesses[inst_idx][*col_idx];
                        }
                    }
                    acc += ts_r_had[row] * y_row.coeffs()[j];
                }
                U[i].push(acc);
            }
        }

        for x in &U[0] {
            transcript.absorb_field_element(x);
        }
        for x in &U[1] {
            transcript.absorb_field_element(x);
        }
        for x in &U[2] {
            transcript.absorb_field_element(x);
        }
        had_u.push(U);
    }

    let ts_r_mon = ts_weights(&mon_rand);

    let mut mon_b: Vec<Vec<R>> = Vec::with_capacity(ell);
    for inst_idx in 0..ell {
        let mut b_inst = Vec::with_capacity(rg_params.k_g);
        for dig in 0..rg_params.k_g {
            let f: &[R] = &witnesses[inst_idx];
            let J = &Js[inst_idx];

            let mut acc = R::ZERO;
            for out_row in 0..m_j {
                let digits = proj_row_digits(f, J, out_row);
                for col in 0..d {
                    let g = exp::<R>(digits[col][dig]).expect("Exp failed");
                    for rep in 0..reps {
                        let r = out_row + rep * m_j;
                        let idx = col * m + r;
                        acc += R::from(ts_r_mon[idx]) * g;
                    }
                }
            }
            b_inst.push(acc);
        }
        transcript.absorb_slice(&b_inst);
        mon_b.push(b_inst);
    }

    Ok(PiFoldProverOutput {
        proof: PiFoldBatchedProof {
            coins: SymphonyCoins {
                challenges: vec![],
                bytes: vec![],
                events: vec![],
            },
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

/// Convert streaming proof to latticefold Proof format.
pub fn convert_streaming_proof<R: OverField>(
    proof: &StreamingProof<R>,
) -> latticefold::utils::sumcheck::Proof<R> {
    use latticefold::utils::sumcheck::prover::ProverMsg;
    latticefold::utils::sumcheck::Proof::new(
        proof
            .0
            .iter()
            .map(|m| ProverMsg::new(m.evaluations.clone()))
            .collect(),
    )
}

/// Build streaming MLEs for Hadamard (helper for custom usage).
pub fn build_hadamard_streaming_mles<R: OverField>(
    matrices: [Arc<SparseMatrix<R>>; 3],
    witnesses: &[Arc<Vec<R>>],
    s_r: Vec<R>,
) -> Vec<Box<dyn StreamingMle<R>>> {
    let ell = witnesses.len();
    let mut mles: Vec<Box<dyn StreamingMle<R>>> = Vec::with_capacity(ell * 4);

    for inst_idx in 0..ell {
        mles.push(Box::new(EqStreamingMle::new(s_r.clone())));
        for i in 0..3 {
            mles.push(Box::new(SparseMatrixMle::new(
                matrices[i].clone(),
                witnesses[inst_idx].clone(),
            )));
        }
    }

    mles
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_streaming_prover_compiles() {
        // The prover compiles - actual testing needs ring instantiation
    }
}
