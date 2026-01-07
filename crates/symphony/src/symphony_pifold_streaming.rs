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
use std::collections::HashMap;
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

/// Configuration for the streaming Π_fold prover.
///
/// Note: the symphony library does **not** read environment variables. If you want
/// runtime configuration, parse env vars in your binary/example and pass them through this struct.
#[derive(Clone, Debug)]
pub struct PiFoldStreamingConfig {
    /// Request the constant-coefficient fast path when inputs appear to be constant-coeff.
    ///
    /// Even when `true`, we still run a deterministic sampling check and fall back to the generic
    /// ring path if the check fails.
    pub request_const_coeff_fastpath: bool,
    /// Witness sample count used by the deterministic constant-coeff check.
    pub const_coeff_witness_samples: usize,
    /// Matrix nonzero-entry sample budget used by the deterministic constant-coeff check.
    pub const_coeff_matrix_entries: usize,
    /// Print internal phase timings (intended for benchmarks/profiling).
    pub profile: bool,
}

impl Default for PiFoldStreamingConfig {
    fn default() -> Self {
        Self {
            request_const_coeff_fastpath: true,
            const_coeff_witness_samples: 1024,
            const_coeff_matrix_entries: 4096,
            profile: false,
        }
    }
}

fn is_const_coeff_ring_elem<R: stark_rings::PolyRing>(x: &R) -> bool {
    x.coeffs()
        .iter()
        .skip(1)
        .all(|c| *c == <R as stark_rings::PolyRing>::BaseRing::ZERO)
}

fn seems_const_coeff_inputs<R: stark_rings::PolyRing>(
    witnesses: &[Arc<Vec<R>>],
    mats_to_check: &[Arc<stark_rings_linalg::SparseMatrix<R>>],
    w_samples: usize,
    m_entries: usize,
) -> bool {
    // Cheap, deterministic sampling to avoid unsound use of the fast path.
    // Defaults are tiny vs proving time; callers can adjust via `PiFoldStreamingConfig`.

    // Witness sample
    if w_samples > 0 {
        for w in witnesses.iter().take(1) {
            let step = (w.len() / w_samples.max(1)).max(1);
            let mut checked = 0usize;
            for i in (0..w.len()).step_by(step) {
                if !is_const_coeff_ring_elem(&w[i]) {
                    return false;
                }
                checked += 1;
                if checked >= w_samples {
                    break;
                }
            }
        }
    }

    // Matrix coefficient sample (scan nonzeros)
    if m_entries > 0 {
        let mut checked = 0usize;
        for m in mats_to_check.iter().take(3) {
            for row in &m.coeffs {
                for (coeff, _col) in row {
                    if !is_const_coeff_ring_elem(coeff) {
                        return false;
                    }
                    checked += 1;
                    if checked >= m_entries {
                        return true;
                    }
                }
            }
        }
    }

    true
}

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

/// Canonical Poseidon-FS prover for Π_fold (streaming sumcheck).
///
/// This is the canonical prover entrypoint. It batches
/// a list of instances where each instance can have its own matrices `(M1,M2,M3)` (hetero-M).
///
/// (If you have a shared matrix across all instances, pass the same `(M1,M2,M3)` for each
/// instance by cloning the `Arc`s.)
pub fn prove_pi_fold_poseidon_fs<R: CoeffRing, PC>(
    Ms: &[[Arc<SparseMatrix<R>>; 3]],
    cms: &[Vec<R>],
    witnesses: &[Arc<Vec<R>>],
    public_inputs: &[R::BaseRing],
    cfs_had_u_scheme: Option<&AjtaiCommitmentScheme<R>>,
    cfs_mon_b_scheme: Option<&AjtaiCommitmentScheme<R>>,
    rg_params: RPParams,
    config: &PiFoldStreamingConfig,
) -> Result<PiFoldProverOutput<R>, String>
where
    R::BaseRing: Zq + Decompose,
    PC: GetPoseidonParams<<<R>::BaseRing as ark_ff::Field>::BasePrimeField>,
{
    if cms.is_empty() {
        return Err("PiFold: empty batch".to_string());
    }
    if Ms.len() != cms.len() || cms.len() != witnesses.len() {
        return Err("PiFold: Ms/cms/witnesses length mismatch".to_string());
    }

    let mut ts = crate::transcript::PoseidonTranscript::<R>::empty::<PC>();
    let mut rts = RecordingTranscriptRef::<R, _>::new(&mut ts);
    rts.absorb_field_element(&R::BaseRing::from(0x4c465053_50494250u128)); // "LFPS_PIBP"
    absorb_public_inputs::<R>(&mut rts, public_inputs);

    let mut out = prove_pi_fold_streaming_hetero_m_with_config(&mut rts, Ms, cms, witnesses, rg_params, config)
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
                        .commit_const_coeff_fast(&encode_had_u_instance::<R>(u))
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
                        .commit_const_coeff_fast(b)
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
    let cfg = PiFoldStreamingConfig::default();
    prove_pi_fold_streaming_with_config(transcript, M, cms, witnesses, rg_params, &cfg)
}

pub fn prove_pi_fold_streaming_with_config<R: CoeffRing>(
    transcript: &mut impl Transcript<R>,
    M: [Arc<SparseMatrix<R>>; 3],
    cms: &[Vec<R>],
    witnesses: &[Arc<Vec<R>>],
    rg_params: RPParams,
    config: &PiFoldStreamingConfig,
) -> Result<PiFoldProverOutput<R>, String>
where
    R::BaseRing: Zq + Decompose,
{
    prove_pi_fold_streaming_impl(transcript, M, cms, witnesses, rg_params, config)
}

fn prove_pi_fold_streaming_impl<R: CoeffRing>(
    transcript: &mut impl Transcript<R>,
    M: [Arc<SparseMatrix<R>>; 3],
    cms: &[Vec<R>],
    witnesses: &[Arc<Vec<R>>],
    rg_params: RPParams,
    config: &PiFoldStreamingConfig,
) -> Result<PiFoldProverOutput<R>, String>
where
    R::BaseRing: Zq + Decompose,
{
    // Constant-coefficient fast path: enabled by default.
    //
    // For many real applications (including SP1-style R1CS), matrices and witnesses are base-field
    // scalars embedded into the ring as constant-coefficient polynomials. In that common case,
    // `y = M*w` stays constant-coefficient and Π_had can be evaluated mostly in the base ring.
    //
    // Safety: we auto-disable this fast path if a small deterministic sample detects any
    // non-constant coefficients.
    let const_coeff_fastpath = config.request_const_coeff_fastpath
        && seems_const_coeff_inputs::<R>(
            witnesses,
            &[M[0].clone(), M[1].clone(), M[2].clone()],
            config.const_coeff_witness_samples,
            config.const_coeff_matrix_entries,
        );
    // Matrix-only constant-coeff optimization: useful when the witness is *not* constant-coeff
    // (e.g. packing many base-field lanes across coefficients), but matrices are still scalar-embedded.
    let const_matrix_only = !const_coeff_fastpath
        && seems_const_coeff_inputs::<R>(&[], &[M[0].clone(), M[1].clone(), M[2].clone()], 0, config.const_coeff_matrix_entries);

    let prof = config.profile;
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
    let m_j = blocks
        .checked_mul(rg_params.lambda_pj)
        .ok_or_else(|| "PiFold: m_J overflow (blocks*lambda_pj)".to_string())?;
    if m < m_j || m % m_j != 0 {
        return Err("PiFold: require m_J <= m and m multiple of m_J".to_string());
    }

    let g_len = m
        .checked_mul(d)
        .ok_or_else(|| "PiFold: overflow in g_len=m*d".to_string())?;
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
    // Key optimization: use sparse mat-vec MLEs instead of materializing y = M*w.
    //
    // IMPORTANT: `eq(s, ·)` is shared across all instances (same `s`), so we include it ONCE
    // to avoid an unnecessary ~ℓ× blow-up in streaming evaluation work.
    let mut mles_had: Vec<StreamingMleEnum<R>> = Vec::with_capacity(1 + ell * 3);
    mles_had.push(StreamingMleEnum::eq_base(s_base.clone()));
    // Cache extracted base-field witnesses by `Arc` pointer so identical witnesses across instances
    // only incur the O(n_f) coeff[0] scan once (common in SP1 chunking).
    let mut w0_cache: HashMap<usize, Arc<Vec<R::BaseRing>>> = HashMap::new();
    for inst_idx in 0..ell {
        let w0 = if const_coeff_fastpath {
            let key = Arc::as_ptr(&witnesses[inst_idx]) as usize;
            Some(
                w0_cache
                    .entry(key)
                    .or_insert_with(|| {
                        Arc::new(
                            witnesses[inst_idx]
                                .iter()
                                .map(|x| x.coeffs()[0])
                                .collect::<Vec<_>>(),
                        )
                    })
                    .clone(),
            )
        } else {
            None
        };
        for i in 0..3 {
            if const_coeff_fastpath {
                mles_had.push(StreamingMleEnum::sparse_mat_vec_const_coeff(
                    M[i].clone(),
                    w0.as_ref().expect("w0 present").clone(),
                ));
            } else if const_matrix_only {
                mles_had.push(StreamingMleEnum::sparse_mat_vec_const_matrix(
                    M[i].clone(),
                    witnesses[inst_idx].clone(),
                ));
            } else {
                mles_had.push(StreamingMleEnum::sparse_mat_vec(
                M[i].clone(),
                witnesses[inst_idx].clone(),
                ));
            }
        }
    }

    // Hadamard combiner (must match Π_had: coefficient-wise constraint with alpha powers)
    //
    // When `CONST_COEFF_FASTPATH` is active we additionally provide a base-ring combiner to avoid
    // allocating/constructing full ring elements in the hot sumcheck loop.
    let rhos_had = rhos.clone();
    let comb_had0: Option<Box<dyn Fn(&[R::BaseRing]) -> R::BaseRing + Sync + Send>> =
        if const_coeff_fastpath {
            let rhos0 = rhos_had.iter().map(|x| x.coeffs()[0]).collect::<Vec<_>>();
            let alpha0 = alpha_pows[0].coeffs()[0];
            Some(Box::new(move |vals: &[R::BaseRing]| -> R::BaseRing {
                let eq0 = vals[0];
                let mut acc_all0 = R::BaseRing::ZERO;
        for inst_idx in 0..ell {
                    let base = 1 + inst_idx * 3;
                    let y10 = vals[base];
                    let y20 = vals[base + 1];
                    let y30 = vals[base + 2];
                    let term0 = y10 * y20 - y30;
                    acc_all0 += rhos0[inst_idx] * (alpha0 * eq0 * term0);
                }
                acc_all0
            }))
        } else {
            None
        };

    let comb_had: Box<dyn Fn(&[R]) -> R + Sync + Send> = Box::new(move |vals: &[R]| -> R {
        let eq = vals[0];
        let mut acc_all = R::ZERO;
        for inst_idx in 0..ell {
            let base = 1 + inst_idx * 3;
            let y1 = &vals[base];
            let y2 = &vals[base + 1];
            let y3 = &vals[base + 2];
            let mut acc = R::ZERO;
            for j in 0..d {
                let term = y1.coeffs()[j] * y2.coeffs()[j] - y3.coeffs()[j];
                acc += alpha_pows[j] * eq * R::from(term);
            }
            acc_all += rhos_had[inst_idx] * acc;
        }
        acc_all
    });

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
                if const_coeff_fastpath {
                    // Constant-coeff specialization: only coeff[0] can be nonzero.
                    let mut h0 = R::BaseRing::ZERO;
                    for t in 0..l_h {
                        let in_row = b * l_h + t;
                        if in_row >= f.len() {
                            continue;
                        }
                        let coef = J[i][t];
                        h0 += coef * f[in_row].coeffs()[0];
                    }
                    // Decompose exactly as the generic path does, but only
                    // write the nonzero column (col=0). Other columns remain zero in `digits_flat`.
        let mut h_row = vec![R::BaseRing::ZERO; d];
                    h_row[0] = h0;
                    let digits = h_row.decompose_to_vec(d_prime, k_g);
                    for dig in 0..k_g {
                        let idx = (out_row * d) * k_g + dig; // col=0
                        unsafe {
                            *(out_ptr as *mut R::BaseRing).add(idx) = digits[0][dig];
                        }
                    }
                } else {
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
                }
            });
        }
        #[cfg(not(feature = "parallel"))]
        {
            for out_row in 0..m_j {
                let i = out_row % lambda_pj;
                let b = out_row / lambda_pj;
                if const_coeff_fastpath {
                    let mut h0 = R::BaseRing::ZERO;
                    for t in 0..l_h {
                        let in_row = b * l_h + t;
                        if in_row >= f.len() {
                            continue;
                        }
                        let coef = J[i][t];
                        h0 += coef * f[in_row].coeffs()[0];
                    }
                    let mut h_row = vec![R::BaseRing::ZERO; d];
                    h_row[0] = h0;
                    let digits = h_row.decompose_to_vec(d_prime, k_g);
                    for dig in 0..k_g {
                        digits_flat[(out_row * d) * k_g + dig] = digits[0][dig];
                    }
                } else {
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

            // Compact m_j values as base-ring scalars over only m_j rows (periodic over m).
            // Store column-major: mj_compact[col * m_j + out_row].
            let mut mj_compact: Vec<R::BaseRing> = vec![R::BaseRing::ZERO; m_j * d];

            #[cfg(feature = "parallel")]
            {
                use ark_std::cfg_into_iter;
                let out_ptr = mj_compact.as_mut_ptr() as usize;
                let beta = *beta_i;
                let dig_local = dig;
                let digits_flat = digits_flat.clone();
                if const_coeff_fastpath && d > 1 {
                    // In constant-coeff mode, projected digits are zero in columns 1..d-1.
                    // So `ev(exp(digit), beta)` is constant for those columns (digit=0).
                    let g0 = exp::<R>(R::BaseRing::ZERO).expect("Exp failed");
                    let mjv_zero = ev(&g0, beta);
                    cfg_into_iter!(0..m_j).for_each(|out_row| {
                        let digit0 = digits_flat[(out_row * d) * rg_params.k_g + dig_local]; // col=0
                        let g = exp::<R>(digit0).expect("Exp failed");
                        let mjv0 = ev(&g, beta);
                        unsafe {
                            *(out_ptr as *mut R::BaseRing).add(out_row) = mjv0;
                        }
                    });
                    for col in 1..d {
                        mj_compact[col * m_j..(col + 1) * m_j].fill(mjv_zero);
                    }
                } else {
                    cfg_into_iter!(0..m_j).for_each(|out_row| {
                        for col in 0..d {
                            let digit = digits_flat[(out_row * d + col) * rg_params.k_g + dig_local];
                            let g = exp::<R>(digit).expect("Exp failed");
                            let mjv = ev(&g, beta);
                            let idx = col * m_j + out_row;
                            unsafe {
                                *(out_ptr as *mut R::BaseRing).add(idx) = mjv;
                            }
                        }
                    });
                }
            }
            #[cfg(not(feature = "parallel"))]
            {
                if const_coeff_fastpath && d > 1 {
                    let g0 = exp::<R>(R::BaseRing::ZERO).expect("Exp failed");
                    let mjv_zero = ev(&g0, *beta_i);
            for out_row in 0..m_j {
                        let digit0 = digits_flat[(out_row * d) * rg_params.k_g + dig]; // col=0
                        let g = exp::<R>(digit0).expect("Exp failed");
                        mj_compact[out_row] = ev(&g, *beta_i);
                    }
                    for col in 1..d {
                        mj_compact[col * m_j..(col + 1) * m_j].fill(mjv_zero);
                    }
                } else {
                    for out_row in 0..m_j {
                for col in 0..d {
                            let digit = digits_flat[(out_row * d + col) * rg_params.k_g + dig];
                            let g = exp::<R>(digit).expect("Exp failed");
                            let mjv = ev(&g, *beta_i);
                            mj_compact[col * m_j + out_row] = mjv;
                        }
                    }
                }
            }

            let mj_compact = Arc::new(mj_compact);
            mles_mon.push(StreamingMleEnum::periodic_base_scalar_vec(
                g_nvars,
                m,
                m_j,
                d,
                mj_compact.clone(),
                false,
            ));
            mles_mon.push(StreamingMleEnum::periodic_base_scalar_vec(
                g_nvars,
                m,
                m_j,
                d,
                mj_compact,
                true,
            ));
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
            let pm_had = if const_coeff_fastpath {
                StreamingSumcheck::prove_round_base(
                    &mut had_state,
                    v_msg_had,
                    comb_had0.as_ref().expect("comb_had0").as_ref(),
                )
            } else {
                StreamingSumcheck::prove_round(&mut had_state, v_msg_had, comb_had.as_ref())
            };
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

    // The streaming sumcheck round driver fixes variables using the *previous* round's challenge.
    // After the last round we have one remaining variable that must be fixed by the final sampled r.
    if log_m > 0 {
        had_state.fix_last_variable(*had_rand.last().expect("had_rand non-empty"));
    }
    if g_nvars > 0 {
        mon_state.fix_last_variable(*mon_rand.last().expect("mon_rand non-empty"));
    }

    let had_sumcheck = convert_streaming_proof(&StreamingProof(had_msgs));
    let mon_sumcheck = convert_streaming_proof(&StreamingProof(mon_msgs));

    // -----------------
    // Compute aux witness
    // -----------------
    let t_aux_had = Instant::now();
    let mut had_u: Vec<[Vec<R::BaseRing>; 3]> = Vec::with_capacity(ell);
    let had_evals = had_state.final_evals();
    for inst_idx in 0..ell {
        let base = 1 + inst_idx * 3;
        // MLE order: [eq(s,r)] then per instance: [y1, y2, y3]
        let y1 = had_evals[base];
        let y2 = had_evals[base + 1];
        let y3 = had_evals[base + 2];

        let mut U: [Vec<R::BaseRing>; 3] = [Vec::with_capacity(d), Vec::with_capacity(d), Vec::with_capacity(d)];
        U[0].extend_from_slice(y1.coeffs());
        U[1].extend_from_slice(y2.coeffs());
        U[2].extend_from_slice(y3.coeffs());

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

/// Streaming Π_fold prover where each instance has its own `(M1,M2,M3)`.
///
/// This is useful for SP1 chunking where chunks correspond to disjoint row-slices of the full R1CS.
pub fn prove_pi_fold_streaming_hetero_m<R: CoeffRing>(
    transcript: &mut impl Transcript<R>,
    Ms: &[[Arc<SparseMatrix<R>>; 3]],
    cms: &[Vec<R>],
    witnesses: &[Arc<Vec<R>>],
    rg_params: RPParams,
) -> Result<PiFoldProverOutput<R>, String>
where
    R::BaseRing: Zq + Decompose,
{
    let cfg = PiFoldStreamingConfig::default();
    prove_pi_fold_streaming_hetero_m_with_config(transcript, Ms, cms, witnesses, rg_params, &cfg)
}

pub fn prove_pi_fold_streaming_hetero_m_with_config<R: CoeffRing>(
    transcript: &mut impl Transcript<R>,
    Ms: &[[Arc<SparseMatrix<R>>; 3]],
    cms: &[Vec<R>],
    witnesses: &[Arc<Vec<R>>],
    rg_params: RPParams,
    config: &PiFoldStreamingConfig,
) -> Result<PiFoldProverOutput<R>, String>
where
    R::BaseRing: Zq + Decompose,
{
    // Cheap sample: first witness and first instance matrices.
    let const_coeff_fastpath = config.request_const_coeff_fastpath
        && seems_const_coeff_inputs::<R>(
            witnesses,
            &[Ms[0][0].clone(), Ms[0][1].clone(), Ms[0][2].clone()],
            config.const_coeff_witness_samples,
            config.const_coeff_matrix_entries,
        );
    let const_matrix_only = !const_coeff_fastpath
        && seems_const_coeff_inputs::<R>(
            &[],
            &[Ms[0][0].clone(), Ms[0][1].clone(), Ms[0][2].clone()],
            0,
            config.const_coeff_matrix_entries,
        );
    // Keep this function bit-for-bit aligned with `prove_pi_fold_streaming_impl` except that each
    // instance pulls y = M*w from its own matrices.
    let prof = config.profile;
    if prof {
        eprintln!(
            "[PiFold streaming hetero] pool: rayon_threads={}, const_coeff_fastpath={}",
            rayon::current_num_threads(),
            const_coeff_fastpath
        );
    }
    let t_all = Instant::now();
    let ell = cms.len();
    let beta_cts = derive_beta_chi::<R>(transcript, ell);
    if ell == 0 || witnesses.len() != ell || Ms.len() != ell {
        return Err("PiFold: length mismatch".to_string());
    }

    // -----------------
    // Shared Π_had coins
    // -----------------
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
    let t_coins = Instant::now();

    let n_f = witnesses[0].len();
    if n_f == 0 || n_f % rg_params.l_h != 0 {
        return Err("PiFold: invalid witness length".to_string());
    }
    for w in witnesses.iter() {
        if w.len() != n_f {
            return Err("PiFold: inconsistent witness lengths".to_string());
        }
    }
    for inst_idx in 0..ell {
        for i in 0..3 {
            if Ms[inst_idx][i].ncols != n_f {
                return Err("PiFold: matrix/witness width mismatch".to_string());
            }
        }
    }

    let blocks = n_f / rg_params.l_h;
    let m_j = blocks
        .checked_mul(rg_params.lambda_pj)
        .ok_or_else(|| "PiFold: m_J overflow".to_string())?;
    if m < m_j || m % m_j != 0 {
        return Err("PiFold: require m_J <= m and m multiple of m_J".to_string());
    }

    let g_len = m
        .checked_mul(d)
        .ok_or_else(|| "PiFold: g_len overflow".to_string())?;
    if !g_len.is_power_of_two() {
        return Err("PiFold: require m*d power-of-two".to_string());
    }
    let g_nvars = log2(g_len) as usize;

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
    if prof {
        eprintln!(
            "[PiFold streaming hetero] coins+J: {:?} (ell={}, k_g={}, g_nvars={})",
            t_coins.elapsed(),
            ell,
            rg_params.k_g,
            g_nvars
        );
    }

    let rhos = transcript
        .get_challenges(ell)
        .into_iter()
        .map(R::from)
        .collect::<Vec<R>>();

    // -----------------
    // STREAMING Hadamard MLEs
    // -----------------
    let t_had_build = Instant::now();
    let mut t_w0 = std::time::Duration::from_secs(0);
    // `eq(s, ·)` is shared across all instances; include it once.
    let mut mles_had: Vec<StreamingMleEnum<R>> = Vec::with_capacity(1 + ell * 3);
    mles_had.push(StreamingMleEnum::eq_base(s_base.clone()));
    let mut w0_cache: HashMap<usize, Arc<Vec<R::BaseRing>>> = HashMap::new();
    for inst_idx in 0..ell {
        let w0 = if const_coeff_fastpath {
            let key = Arc::as_ptr(&witnesses[inst_idx]) as usize;
            if let Some(v) = w0_cache.get(&key) {
                Some(v.clone())
            } else {
                let t0 = Instant::now();
                let v = Arc::new(
                    witnesses[inst_idx]
                        .iter()
                        .map(|x| x.coeffs()[0])
                        .collect::<Vec<_>>(),
                );
                t_w0 += t0.elapsed();
                w0_cache.insert(key, v.clone());
                Some(v)
            }
        } else {
            None
        };
        for i in 0..3 {
            if const_coeff_fastpath {
                mles_had.push(StreamingMleEnum::sparse_mat_vec_const_coeff(
                    Ms[inst_idx][i].clone(),
                    w0.as_ref().expect("w0 present").clone(),
                ));
            } else if const_matrix_only {
                mles_had.push(StreamingMleEnum::sparse_mat_vec_const_matrix(
                    Ms[inst_idx][i].clone(),
                    witnesses[inst_idx].clone(),
                ));
            } else {
                mles_had.push(StreamingMleEnum::sparse_mat_vec(
                    Ms[inst_idx][i].clone(),
                    witnesses[inst_idx].clone(),
                ));
            }
        }
    }
    if prof {
        eprintln!(
            "[PiFold streaming hetero] had MLE build: {:?} (w0={:?}, const_coeff_fastpath={})",
            t_had_build.elapsed(),
            t_w0,
            const_coeff_fastpath
        );
    }

    let rhos_had = rhos.clone();
    let comb_had0: Option<Box<dyn Fn(&[R::BaseRing]) -> R::BaseRing + Sync + Send>> =
        if const_coeff_fastpath {
            let rhos0 = rhos_had.iter().map(|x| x.coeffs()[0]).collect::<Vec<_>>();
            let alpha0 = alpha_pows[0].coeffs()[0];
            Some(Box::new(move |vals: &[R::BaseRing]| -> R::BaseRing {
                let eq0 = vals[0];
                let mut acc_all0 = R::BaseRing::ZERO;
                for inst_idx in 0..ell {
                    let base = 1 + inst_idx * 3;
                    let y10 = vals[base];
                    let y20 = vals[base + 1];
                    let y30 = vals[base + 2];
                    let term0 = y10 * y20 - y30;
                    acc_all0 += rhos0[inst_idx] * (alpha0 * eq0 * term0);
                }
                acc_all0
            }))
        } else {
            None
        };

    let comb_had: Box<dyn Fn(&[R]) -> R + Sync + Send> = Box::new(move |vals: &[R]| -> R {
        let eq = vals[0];
        let mut acc_all = R::ZERO;
        for inst_idx in 0..ell {
            let base = 1 + inst_idx * 3;
            let y1 = &vals[base];
            let y2 = &vals[base + 1];
            let y3 = &vals[base + 2];
            let mut acc = R::ZERO;
            for j in 0..d {
                let term = y1.coeffs()[j] * y2.coeffs()[j] - y3.coeffs()[j];
                acc += alpha_pows[j] * eq * R::from(term);
            }
            acc_all += rhos_had[inst_idx] * acc;
        }
        acc_all
    });

    // -----------------
    // Build monomial side MLEs (identical to shared-M path)
    // -----------------
    let mon_mles_per = 3 * rg_params.k_g;
    let mut mles_mon: Vec<StreamingMleEnum<R>> = Vec::with_capacity(ell * mon_mles_per);

    let alphas_ring = cba_all
        .iter()
        .map(|cba| cba.iter().map(|(_, _, a)| R::from(*a)).collect::<Vec<_>>())
        .collect::<Vec<_>>();

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
                if const_coeff_fastpath {
                    let mut h0 = R::BaseRing::ZERO;
                    for t in 0..l_h {
                        let in_row = b * l_h + t;
                        if in_row >= f.len() {
                            continue;
                        }
                        let coef = J[i][t];
                        h0 += coef * f[in_row].coeffs()[0];
                    }
                    let mut h_row = vec![R::BaseRing::ZERO; d];
                    h_row[0] = h0;
                    let digits = h_row.decompose_to_vec(d_prime, k_g);
                    for dig in 0..k_g {
                        let idx = (out_row * d) * k_g + dig; // col=0
                        unsafe {
                            let out = (out_ptr as *mut R::BaseRing).add(idx);
                            *out = digits[0][dig];
                        }
                    }
                } else {
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
                                let out = (out_ptr as *mut R::BaseRing).add(idx);
                                *out = digits[col][dig];
                            }
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
                if const_coeff_fastpath {
                    let mut h0 = R::BaseRing::ZERO;
                    for t in 0..l_h {
                        let in_row = b * l_h + t;
                        if in_row >= f.len() {
                            continue;
                        }
                        let coef = J[i][t];
                        h0 += coef * f[in_row].coeffs()[0];
                    }
                    let mut h_row = vec![R::BaseRing::ZERO; d];
                    h_row[0] = h0;
                    let digits = h_row.decompose_to_vec(d_prime, k_g);
                    for dig in 0..k_g {
                        digits_flat[(out_row * d) * k_g + dig] = digits[0][dig];
                    }
                } else {
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
        }
        proj_digits_by_inst.push(Arc::new(digits_flat));
    }
    if prof {
        eprintln!(
            "[PiFold streaming hetero] proj_digits: {:?} (rayon_threads={}, const_coeff_fastpath={})",
            t_digits.elapsed(),
            rayon::current_num_threads(),
            const_coeff_fastpath
        );
    }

    let t_mon_build = Instant::now();
    for inst_idx in 0..ell {
        let digits_flat = &proj_digits_by_inst[inst_idx];
        for dig in 0..rg_params.k_g {
            let (_c, beta_i, _alpha_i) = &cba_all[inst_idx][dig];

            // mj_compact[col*m_j + out_row] := ev(exp(digit), beta_i)
            let mut mj_compact = vec![R::BaseRing::ZERO; m_j * d];
            #[cfg(feature = "parallel")]
            {
                use ark_std::cfg_into_iter;
                let out_ptr = mj_compact.as_mut_ptr() as usize;
                let digits_flat = digits_flat.clone();
                let beta_i = *beta_i;
                let dig_local = dig;
                if const_coeff_fastpath && d > 1 {
                    let g0 = exp::<R>(R::BaseRing::ZERO).expect("Exp failed");
                    let mjv_zero = ev(&g0, beta_i);
                    cfg_into_iter!(0..m_j).for_each(|out_row| {
                        let digit0 = digits_flat[(out_row * d) * rg_params.k_g + dig_local]; // col=0
                        let g = exp::<R>(digit0).expect("Exp failed");
                        let mjv0 = ev(&g, beta_i);
                        unsafe {
                            let out = (out_ptr as *mut R::BaseRing).add(out_row);
                            *out = mjv0;
                        }
                    });
                    for col in 1..d {
                        mj_compact[col * m_j..(col + 1) * m_j].fill(mjv_zero);
                    }
                } else {
                    cfg_into_iter!(0..m_j).for_each(|out_row| {
                        for col in 0..d {
                            let digit = digits_flat[(out_row * d + col) * rg_params.k_g + dig_local];
                            let g = exp::<R>(digit).expect("Exp failed");
                            let mjv = ev(&g, beta_i);
                            let idx = col * m_j + out_row;
                            unsafe {
                                let out = (out_ptr as *mut R::BaseRing).add(idx);
                                *out = mjv;
                            }
                        }
                    });
                }
            }
            #[cfg(not(feature = "parallel"))]
            {
                if const_coeff_fastpath && d > 1 {
                    let g0 = exp::<R>(R::BaseRing::ZERO).expect("Exp failed");
                    let mjv_zero = ev(&g0, *beta_i);
                    for out_row in 0..m_j {
                        let digit0 = digits_flat[(out_row * d) * rg_params.k_g + dig]; // col=0
                        let g = exp::<R>(digit0).expect("Exp failed");
                        mj_compact[out_row] = ev(&g, *beta_i);
                    }
                    for col in 1..d {
                        mj_compact[col * m_j..(col + 1) * m_j].fill(mjv_zero);
                    }
                } else {
                    for out_row in 0..m_j {
                        for col in 0..d {
                            let digit = digits_flat[(out_row * d + col) * rg_params.k_g + dig];
                            let g = exp::<R>(digit).expect("Exp failed");
                            let mjv = ev(&g, *beta_i);
                            mj_compact[col * m_j + out_row] = mjv;
                        }
                    }
                }
            }
            let mj_compact = Arc::new(mj_compact);
            mles_mon.push(StreamingMleEnum::periodic_base_scalar_vec(
                g_nvars,
                m,
                m_j,
                d,
                mj_compact.clone(),
                false,
            ));
            mles_mon.push(StreamingMleEnum::periodic_base_scalar_vec(
                g_nvars,
                m,
                m_j,
                d,
                mj_compact,
                true,
            ));

            // eq(c) as streaming MLE.
            let (c, _beta_i2, _alpha_i2) = &cba_all[inst_idx][dig];
            let c_base = c.iter().map(|x| x.coeffs()[0]).collect::<Vec<_>>();
            mles_mon.push(StreamingMleEnum::eq_base(c_base));
        }
    }
    if prof {
        eprintln!(
            "[PiFold streaming hetero] built mon MLEs in {:?} (g_len={}, g_nvars={}, k_g={}, ell={})",
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
        if round_idx < log_m {
            let pm_had = if const_coeff_fastpath {
                StreamingSumcheck::prove_round_base(
                    &mut had_state,
                    v_msg_had,
                    comb_had0.as_ref().expect("comb_had0").as_ref(),
                )
            } else {
                StreamingSumcheck::prove_round(&mut had_state, v_msg_had, comb_had.as_ref())
            };
            transcript.absorb_slice(&pm_had.evaluations);
            had_msgs.push(pm_had);
        }
        if round_idx < g_nvars {
            let pm_mon = StreamingSumcheck::prove_round(&mut mon_state, v_msg_mon, &comb_mon);
            transcript.absorb_slice(&pm_mon.evaluations);
            mon_msgs.push(pm_mon);
        }

        let r = transcript.get_challenge();
        transcript.absorb(&R::from(r));
        sampled.push(r);

        if hook_round != 0 && sampled.len() == hook_round {
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
            "[PiFold streaming hetero] sumcheck: {:?} (rayon_threads={}, const_coeff_fastpath={})",
            t_sumcheck.elapsed(),
            rayon::current_num_threads(),
            const_coeff_fastpath
        );
        eprintln!("[PiFold streaming hetero] total: {:?}", t_all.elapsed());
    }

    let had_sumcheck = convert_streaming_proof(&StreamingProof(had_msgs));
    let mon_sumcheck = convert_streaming_proof(&StreamingProof(mon_msgs));

    // -----------------
    // Compute aux witness (match prove_pi_fold_streaming)
    // -----------------
    // Final randomness vectors for each sumcheck.
    let had_rand = sampled[..log_m].to_vec();
    let mon_rand = sampled[..g_nvars].to_vec();
    had_state.randomness = had_rand.clone();
    mon_state.randomness = mon_rand.clone();
    if log_m > 0 {
        had_state.fix_last_variable(*had_rand.last().expect("had_rand non-empty"));
    }
    if g_nvars > 0 {
        mon_state.fix_last_variable(*mon_rand.last().expect("mon_rand non-empty"));
    }

    let had_evals = had_state.final_evals();
    let mut had_u: Vec<[Vec<R::BaseRing>; 3]> = Vec::with_capacity(ell);
    for inst_idx in 0..ell {
        // MLE order: [eq(s,r)] then per instance: [y1, y2, y3]
        let base = 1 + inst_idx * 3;
        let y1 = had_evals[base];
        let y2 = had_evals[base + 1];
        let y3 = had_evals[base + 2];
        let mut U: [Vec<R::BaseRing>; 3] =
            [Vec::with_capacity(d), Vec::with_capacity(d), Vec::with_capacity(d)];
        U[0].extend_from_slice(y1.coeffs());
        U[1].extend_from_slice(y2.coeffs());
        U[2].extend_from_slice(y3.coeffs());
        had_u.push(U);
    }

    let reps = m / m_j;
    let ts_r_mon = ts_weights(&mon_rand);
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
