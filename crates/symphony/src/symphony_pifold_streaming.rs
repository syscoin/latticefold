//! Streaming version of Symphony Π_fold prover.
//!
//! This module provides a memory-efficient version of the batched Π_fold prover
//! that uses streaming MLEs instead of materializing O(2^n) evaluation vectors.
//!
//! ## Key Optimizations
//!
//! 1. **Hadamard MLEs**: sparse mat-vec is evaluated on-demand, then fixed rounds materialize dense
//! 2. **Eq MLEs**: evaluated in the base ring (constant-coefficient) and lifted to `R`
//! 3. **Monomial MLEs**: base-field scalars (m_j/m') + base-field eq(c), both streaming/compact
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
use std::time::Instant;
use stark_rings::{
    balanced_decomposition::{Decompose, DecomposeToVec},
    exp, CoeffRing, OverField, Ring, Zq,
};
use stark_rings_linalg::SparseMatrix;
#[cfg(feature = "parallel")]
use rayon::iter::IntoParallelIterator;
#[cfg(feature = "parallel")]
use rayon::iter::ParallelIterator;
use cyclotomic_rings::rings::GetPoseidonParams;
use latticefold::{
    commitment::AjtaiCommitmentScheme,
    transcript::Transcript,
};

use crate::recording_transcript::RecordingTranscriptRef;
use crate::rp_rgchk::RPParams;
use crate::streaming_sumcheck::{StreamingMleEnum, StreamingProof, StreamingSumcheck};
use crate::symphony_cm::SymphonyCoins;
use crate::symphony_coins::{derive_beta_chi, derive_J, ev, ts_weights};
use crate::symphony_pifold_batched::{PiFoldAuxWitness, PiFoldBatchedProof, PiFoldProverOutput};

// Re-export streaming types
pub use crate::streaming_sumcheck::{StreamingProverMsg, StreamingSumcheckState};

fn absorb_public_inputs<R: OverField>(ts: &mut impl Transcript<R>, public_inputs: &[R::BaseRing])
where
    R::BaseRing: Zq,
{
    // Keep identical statement-binding schedule to symphony_pifold_batched.
    ts.absorb_field_element(&R::BaseRing::from(0x4c465053_5055424cu128)); // "LFPS_PUBL"
    ts.absorb_field_element(&R::BaseRing::from(public_inputs.len() as u128));
    for x in public_inputs {
        ts.absorb_field_element(x);
    }
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

/// Poseidon-FS wrapper for the streaming Π_fold prover.
///
/// Produces the same `PiFoldProverOutput` shape as `prove_pi_fold_batched_sumcheck_fs`:
/// - records transcript coins into `proof.coins`
/// - optionally commits to CP transcript witness messages (`cfs_*`)
pub fn prove_pi_fold_streaming_sumcheck_fs<R: CoeffRing, PC>(
    M: [Arc<SparseMatrix<R>>; 3],
    cms: &[Vec<R>],
    witnesses: &[Arc<Vec<R>>],
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

    let mut out = prove_pi_fold_streaming(&mut rts, M, cms, witnesses, rg_params)
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
    let prof = std::env::var("SYMPHONY_PROFILE").ok().as_deref() == Some("1");
    let t_all = Instant::now();
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
    // Key optimization: use sparse mat-vec MLEs instead of materializing y = M*w
    let mut mles_had: Vec<StreamingMleEnum<R>> = Vec::with_capacity(ell * 4);

    for inst_idx in 0..ell {
        mles_had.push(StreamingMleEnum::eq_base(s_base.clone()));
        for i in 0..3 {
            mles_had.push(StreamingMleEnum::sparse_mat_vec(
                M[i].clone(),
                witnesses[inst_idx].clone(),
            ));
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
    // Monomial MLEs (streaming/compact)
    //
    // Critical: avoid materializing the huge dense eq(c) MLE tables (size = m*d ring elements).
    // We keep m_j/m' as base-ring scalars (compact), and make eq(c) fully streaming.
    // -----------------
    let mon_mles_per = 3 * rg_params.k_g;
    let mut mles_mon: Vec<StreamingMleEnum<R>> = Vec::with_capacity(ell * mon_mles_per);

    let alphas_ring = cba_all
        .iter()
        .map(|cba| cba.iter().map(|(_, _, a)| R::from(*a)).collect::<Vec<_>>())
        .collect::<Vec<_>>();

    let reps = m / m_j;

    // Precompute projected digits once per instance.
    // Layout: digits[(row * d + col) * k_g + dig] where
    // - row ∈ [0, m_j)
    // - col ∈ [0, d)
    // - dig ∈ [0, k_g)
    let t_digits = Instant::now();
    let mut proj_digits_by_inst: Vec<Arc<Vec<R::BaseRing>>> = Vec::with_capacity(ell);
    for inst_idx in 0..ell {
        let f: &[R] = &witnesses[inst_idx];
        let J = &Js[inst_idx];
        let k_g = rg_params.k_g;
        let lambda_pj = rg_params.lambda_pj;
        let l_h = rg_params.l_h;
        let d_prime = rg_params.d_prime;

        let mut digits_flat = vec![R::BaseRing::ZERO; m_j * d * k_g];

        #[cfg(feature = "parallel")]
        {
            use ark_std::cfg_into_iter;
            let out_ptr = digits_flat.as_mut_ptr() as usize;
            cfg_into_iter!(0..m_j).for_each(|out_row| {
                let i = out_row % lambda_pj;
                let b = out_row / lambda_pj;
                let mut h_row = vec![R::BaseRing::ZERO; d];
                for t in 0..l_h {
                    let in_row = b * l_h + t;
                    if in_row >= f.len() {
                        continue;
                    }
                    let coef = J[i][t];
                    let coeffs = f[in_row].coeffs();
                    for col in 0..d {
                        h_row[col] += coef * coeffs[col];
                    }
                }
                let digits = h_row.decompose_to_vec(d_prime, k_g);
                for col in 0..d {
                    for dig in 0..k_g {
                        let idx = (out_row * d + col) * k_g + dig;
                        unsafe {
                            *(out_ptr as *mut R::BaseRing).add(idx) = digits[col][dig];
                        }
                    }
                }
            });
        }
        #[cfg(not(feature = "parallel"))]
        {
            for out_row in 0..m_j {
                let i = out_row % lambda_pj;
                let b = out_row / lambda_pj;
                let mut h_row = vec![R::BaseRing::ZERO; d];
                for t in 0..l_h {
                    let in_row = b * l_h + t;
                    if in_row >= f.len() {
                        continue;
                    }
                    let coef = J[i][t];
                    let coeffs = f[in_row].coeffs();
                    for col in 0..d {
                        h_row[col] += coef * coeffs[col];
                    }
                }
                let digits = h_row.decompose_to_vec(d_prime, k_g);
                for col in 0..d {
                    for dig in 0..k_g {
                        digits_flat[(out_row * d + col) * k_g + dig] = digits[col][dig];
                    }
                }
            }
        }

        proj_digits_by_inst.push(Arc::new(digits_flat));
    }
    if prof {
        eprintln!(
            "[PiFold streaming] precomputed proj digits in {:?} (ell={}, m_j={}, d={}, k_g={})",
            t_digits.elapsed(),
            ell,
            m_j,
            d,
            rg_params.k_g
        );
    }

    let t_mon_build = Instant::now();
    for inst_idx in 0..ell {
        let digits_flat = proj_digits_by_inst[inst_idx].clone();

        for dig in 0..rg_params.k_g {
            let (c, beta_i, _a) = &cba_all[inst_idx][dig];

            // Compact m_j/m' as base-ring scalars.
            let mut m_j_evals: Vec<R::BaseRing> = vec![R::BaseRing::ZERO; g_len];
            let mut m_prime: Vec<R::BaseRing> = vec![R::BaseRing::ZERO; g_len];

            // Parallel fill (disjoint writes by out_row -> unique r -> unique idx).
            #[cfg(feature = "parallel")]
            {
                use ark_std::cfg_into_iter;
                // We pass raw pointers as usize to satisfy Send+Sync bounds; we only do disjoint writes.
                let mj_ptr = m_j_evals.as_mut_ptr() as usize;
                let mp_ptr = m_prime.as_mut_ptr() as usize;
                let m_local = m;
                let mj_local = m_j;
                let reps_local = reps;
                let beta = *beta_i;
                let dig_local = dig;
                let digits_flat = digits_flat.clone();
                cfg_into_iter!(0..m_j).for_each(|out_row| {
                    for col in 0..d {
                        let digit = digits_flat[(out_row * d + col) * rg_params.k_g + dig_local];
                        let g = exp::<R>(digit).expect("Exp failed");
                        let mjv = ev(&g, beta);
                        let mpv = mjv * mjv;
                        for rep in 0..reps_local {
                            let r = out_row + rep * mj_local;
                            let idx = col * m_local + r;
                            // SAFETY: for different out_row, r differs => idx differs for all col/rep.
                            unsafe {
                                *(mj_ptr as *mut R::BaseRing).add(idx) = mjv;
                                *(mp_ptr as *mut R::BaseRing).add(idx) = mpv;
                            }
                        }
                    }
                });
            }
            #[cfg(not(feature = "parallel"))]
            {
                for out_row in 0..m_j {
                    for col in 0..d {
                        let digit = digits_flat[(out_row * d + col) * rg_params.k_g + dig];
                        let g = exp::<R>(digit).expect("Exp failed");
                        let mj = ev(&g, *beta_i);
                        let mp = mj * mj;
                        for rep in 0..reps {
                            let r = out_row + rep * m_j;
                            let idx = col * m + r;
                            m_j_evals[idx] = mj;
                            m_prime[idx] = mp;
                        }
                    }
                }
            }

            mles_mon.push(StreamingMleEnum::base_scalar_vec(g_nvars, Arc::new(m_j_evals)));
            mles_mon.push(StreamingMleEnum::base_scalar_vec(g_nvars, Arc::new(m_prime)));
            // eq(c) as streaming MLE (no 2^n table).
            let c_base = c.iter().map(|x| x.coeffs()[0]).collect::<Vec<_>>();
            mles_mon.push(StreamingMleEnum::eq_base(c_base));
        }
    }
    if prof {
        eprintln!(
            "[PiFold streaming] built mon MLEs in {:?} (g_len={}, g_nvars={}, k_g={}, ell={})",
            t_mon_build.elapsed(),
            g_len,
            g_nvars,
            rg_params.k_g,
            ell
        );
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
    let hook_round = log2(m_j.next_power_of_two()) as usize;
    let mut v_digits_folded: Vec<Vec<R::BaseRing>> =
        vec![vec![R::BaseRing::ZERO; d]; rg_params.k_g];

    // Sumcheck transcript header (matches latticefold verify_two_as_subprotocol_shared_with_hook)
    transcript.absorb(&R::from(log_m as u128));
    transcript.absorb(&R::from(3u128));
    transcript.absorb(&R::from(g_nvars as u128));
    transcript.absorb(&R::from(3u128));

    let mut had_state = StreamingSumcheck::prover_init(mles_had, log_m, 3);
    let mut mon_state = StreamingSumcheck::prover_init(mles_mon, g_nvars, 3);

    let mut had_msgs = Vec::with_capacity(log_m);
    let mut mon_msgs = Vec::with_capacity(g_nvars);
    let mut sampled: Vec<R::BaseRing> = Vec::with_capacity(log_m.max(g_nvars));
    let mut v_msg_had: Option<R::BaseRing> = None;
    let mut v_msg_mon: Option<R::BaseRing> = None;

    let rounds = log_m.max(g_nvars);
    let t_sumcheck = Instant::now();
    for round_idx in 0..rounds {
        // Prover messages (in the same order as latticefold's shared schedule)
        if round_idx < log_m {
            let pm_had = StreamingSumcheck::prove_round(&mut had_state, v_msg_had, &comb_had);
            transcript.absorb_slice(&pm_had.evaluations);
            had_msgs.push(pm_had);
        }
        if round_idx < g_nvars {
            let pm_mon = StreamingSumcheck::prove_round(&mut mon_state, v_msg_mon, &comb_mon);
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
                let digits_flat = &proj_digits_by_inst[inst_idx];
                for row in 0..m_j {
                    let w = ts_r[row];
                    for col in 0..d {
                        for dig in 0..rg_params.k_g {
                            let digit = digits_flat[(row * d + col) * rg_params.k_g + dig];
                            v_digits_folded[dig][col] += b * digit * w;
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
        v_msg_mon = Some(r);
    }
    if prof {
        eprintln!(
            "[PiFold streaming] sumchecks done in {:?} (log_m={}, g_nvars={}, rounds={})",
            t_sumcheck.elapsed(),
            log_m,
            g_nvars,
            rounds
        );
    }

    // Final randomness vectors for each sumcheck (same convention as latticefold sumcheck state)
    let had_rand = sampled[..log_m].to_vec();
    let mon_rand = sampled[..g_nvars].to_vec();
    had_state.randomness = had_rand.clone();
    mon_state.randomness = mon_rand.clone();

    let had_sumcheck = convert_streaming_proof(&StreamingProof(had_msgs));
    let mon_sumcheck = convert_streaming_proof(&StreamingProof(mon_msgs));

    // -----------------
    // Compute aux witness
    // -----------------
    let ts_r_had = ts_weights(&had_rand);

    let t_aux_had = Instant::now();
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
    if prof {
        eprintln!("[PiFold streaming] aux had_u done in {:?}", t_aux_had.elapsed());
    }

    let ts_r_mon = ts_weights(&mon_rand);

    let t_aux_mon = Instant::now();
    let mut mon_b: Vec<Vec<R>> = Vec::with_capacity(ell);
    for inst_idx in 0..ell {
        let mut b_inst = Vec::with_capacity(rg_params.k_g);
        let digits_flat = proj_digits_by_inst[inst_idx].clone();
        for dig in 0..rg_params.k_g {
            #[cfg(feature = "parallel")]
            let acc = {
                use ark_std::cfg_into_iter;
                let dig_local = dig;
                let digits_flat = digits_flat.clone();
                cfg_into_iter!(0..m_j)
                    .map(|out_row| {
                        let mut local = R::ZERO;
                        for col in 0..d {
                            let digit = digits_flat[(out_row * d + col) * rg_params.k_g + dig_local];
                            let g = exp::<R>(digit).expect("Exp failed");
                            for rep in 0..reps {
                                let r = out_row + rep * m_j;
                                let idx = col * m + r;
                                local += R::from(ts_r_mon[idx]) * g;
                            }
                        }
                        local
                    })
                    .reduce(|| R::ZERO, |a, b| a + b)
            };
            #[cfg(not(feature = "parallel"))]
            let acc = {
                let mut acc = R::ZERO;
                for out_row in 0..m_j {
                    for col in 0..d {
                        let digit = digits_flat[(out_row * d + col) * rg_params.k_g + dig];
                        let g = exp::<R>(digit).expect("Exp failed");
                        for rep in 0..reps {
                            let r = out_row + rep * m_j;
                            let idx = col * m + r;
                            acc += R::from(ts_r_mon[idx]) * g;
                        }
                    }
                }
                acc
            };
            b_inst.push(acc);
        }
        transcript.absorb_slice(&b_inst);
        mon_b.push(b_inst);
    }
    if prof {
        eprintln!("[PiFold streaming] aux mon_b done in {:?}", t_aux_mon.elapsed());
        eprintln!("[PiFold streaming] total prove_pi_fold_streaming time {:?}", t_all.elapsed());
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
///
/// NOTE: `s_r` must be constant-coefficient ring elements (embedded base-ring scalars),
/// because we evaluate `eq(s, r)` in the base ring for performance.
pub fn build_hadamard_streaming_mles<R: OverField + stark_rings::PolyRing>(
    matrices: [Arc<SparseMatrix<R>>; 3],
    witnesses: &[Arc<Vec<R>>],
    s_r: Vec<R>,
) -> Vec<StreamingMleEnum<R>> {
    let ell = witnesses.len();
    let mut mles: Vec<StreamingMleEnum<R>> = Vec::with_capacity(ell * 4);

    for inst_idx in 0..ell {
        let s_base = s_r.iter().map(|x| x.coeffs()[0]).collect::<Vec<_>>();
        mles.push(StreamingMleEnum::eq_base(s_base));
        for i in 0..3 {
            mles.push(StreamingMleEnum::sparse_mat_vec(
                matrices[i].clone(),
                witnesses[inst_idx].clone(),
            ));
        }
    }

    mles
}
