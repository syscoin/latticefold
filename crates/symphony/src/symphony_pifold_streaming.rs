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

use ark_ff::{BigInteger, Field, PrimeField};
use ark_std::log2;
use ark_std::UniformRand;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;
use sha2::{Digest, Sha256};
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

#[derive(Clone, Debug)]
enum ProjDigits<R: CoeffRing>
where
    R::BaseRing: Zq,
{
    /// Full digits table: length = m_j * d * k_g.
    Full(Arc<Vec<R::BaseRing>>),
    /// Const-coeff fast path: only col=0 digits, length = m_j * k_g.
    Col0(Arc<Vec<R::BaseRing>>),
}

impl<R: CoeffRing> ProjDigits<R>
where
    R::BaseRing: Zq,
{
    #[inline]
    fn digit(&self, _m_j: usize, d: usize, k_g: usize, out_row: usize, col: usize, dig: usize) -> R::BaseRing {
        match self {
            ProjDigits::Full(v) => v[(out_row * d + col) * k_g + dig],
            ProjDigits::Col0(v) => {
                if col == 0 {
                    v[out_row * k_g + dig]
                } else {
                    <R::BaseRing as Field>::ZERO
                }
            }
        }
    }
}

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
    /// EXPERIMENTAL (prover-only): derive per-instance Π_mon coins (and J) from an instance-local RNG
    /// seeded by `(cm_f[i], inst_idx)` instead of the sequential Poseidon transcript schedule.
    ///
    /// This enables parallelizing the expensive `cm_g` commitment generation across instances for
    /// performance experiments. Proofs produced under this flag will **not** verify against the
    /// canonical verifier.
    pub experimental_instance_local_coins: bool,
}

impl Default for PiFoldStreamingConfig {
    fn default() -> Self {
        Self {
            request_const_coeff_fastpath: true,
            const_coeff_witness_samples: 1024,
            const_coeff_matrix_entries: 4096,
            profile: false,
            experimental_instance_local_coins: false,
        }
    }
}

fn sha256_ring_vec<R: OverField>(tag: &[u8], v: &[R]) -> [u8; 32]
where
    R::BaseRing: ark_ff::Field,
{
    let mut h = Sha256::new();
    h.update(tag);
    h.update(&(v.len() as u64).to_le_bytes());
    for x in v {
        for c in x.coeffs() {
            for fp in c.to_base_prime_field_elements() {
                h.update(fp.into_bigint().to_bytes_le());
            }
        }
    }
    h.finalize().into()
}

fn sample_small_base<R: OverField, RNG: rand::Rng + ?Sized>(rng: &mut RNG) -> R::BaseRing
where
    R::BaseRing: Zq + Field,
{
    // Very small symmetric distribution in the base field: {0, +1, -1}.
    // This is closer to the paper’s “low-norm” sampling than uniform field elements.
    let r: u8 = rng.gen::<u8>() % 3;
    match r {
        0 => <R::BaseRing as Field>::ZERO,
        1 => <R::BaseRing as Field>::ONE,
        _ => -<R::BaseRing as Field>::ONE,
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
    cm_g_scheme: &AjtaiCommitmentScheme<R>,
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

    let mut out = prove_pi_fold_streaming_hetero_m_with_config(
        &mut rts,
        Ms,
        cms,
        witnesses,
        rg_params,
        config,
        cm_g_scheme,
    )
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


pub fn prove_pi_fold_streaming_hetero_m_with_config<R: CoeffRing>(
    transcript: &mut impl Transcript<R>,
    Ms: &[[Arc<SparseMatrix<R>>; 3]],
    cms: &[Vec<R>],
    witnesses: &[Arc<Vec<R>>],
    rg_params: RPParams,
    config: &PiFoldStreamingConfig,
    cm_g_scheme: &AjtaiCommitmentScheme<R>,
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
    let mut pow = <R::BaseRing as Field>::ONE;
    for _ in 0..d {
        alpha_pows.push(R::from(pow));
        pow *= alpha_base;
    }

    // -----------------
    // Per-instance prover messages + coins
    // -----------------
    //
    // Ordering (Figure 2 / Figure 4):
    // - absorb cm_f, derive J
    // - compute g^(i) and commit c(i)=Commit(g^(i)); absorb c(i)
    // - derive Π_mon coins (c,beta,alpha,...) that depend on the commitments
    let mut cba_all: Vec<Vec<(Vec<R>, R::BaseRing, R::BaseRing)>> = Vec::with_capacity(ell);
    let mut rc_all: Vec<Option<R::BaseRing>> = Vec::with_capacity(ell);
    let mut Js: Vec<Vec<Vec<R::BaseRing>>> = Vec::with_capacity(ell);
    let mut proj_digits_by_inst: Vec<ProjDigits<R>> = Vec::with_capacity(ell);
    let mut cm_g: Vec<Vec<Vec<R>>> = Vec::with_capacity(ell);
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

    if config.experimental_instance_local_coins {
        // Prover-only experiment: derive per-instance J and Π_mon coins from an instance-local RNG
        // seeded by (cm_f[i], inst_idx). This breaks compatibility with the verifier.
        use rand::rngs::StdRng;
        use rand::SeedableRng;

        #[cfg(feature = "parallel")]
        use rayon::prelude::*;

        let per_inst = (0..ell)
            .into_par_iter()
            .map(|inst_idx| {
                let cm_f = &cms[inst_idx];
                let f: &[R] = &witnesses[inst_idx];

                // Phase 1 seed: depends only on cm_f (commit-before-J).
                let mut h = Sha256::new();
                h.update(b"SYMPHONY_EXPERIMENTAL_PI_FOLD_COINS_V2");
                h.update(b"PHASE1_J_FROM_CMF");
                h.update(&(inst_idx as u64).to_le_bytes());
                let cm_f_digest = sha256_ring_vec::<R>(b"cm_f", cm_f);
                h.update(&cm_f_digest);
                let seed1: [u8; 32] = h.finalize().into();
                let mut rng1 = StdRng::from_seed(seed1);

                // J: lambda_pj x l_h
                let mut J: Vec<Vec<R::BaseRing>> = Vec::with_capacity(rg_params.lambda_pj);
                for _ in 0..rg_params.lambda_pj {
                    let mut row = Vec::with_capacity(rg_params.l_h);
                    for _ in 0..rg_params.l_h {
                        row.push(sample_small_base::<R, _>(&mut rng1));
                    }
                    J.push(row);
                }

                // We'll derive Π_mon coins in Phase 2 after cm_g is computed (commit-before-coins).

                // Projected digits (either full or col0-only)
                let k_g = rg_params.k_g;
                let lambda_pj = rg_params.lambda_pj;
                let l_h = rg_params.l_h;
                let d_prime = rg_params.d_prime;
                let proj: ProjDigits<R> = if const_coeff_fastpath {
                    let mut digits0 = vec![<R::BaseRing as Field>::ZERO; m_j * k_g];
                    #[cfg(feature = "parallel")]
                    {
                        use rayon::prelude::*;
                        digits0
                            .par_chunks_mut(k_g)
                            .enumerate()
                            .for_each(|(out_row, chunk)| {
                                let i = out_row % lambda_pj;
                                let b = out_row / lambda_pj;
                                let mut h0 = <R::BaseRing as Field>::ZERO;
                                for t in 0..l_h {
                                    let in_row = b * l_h + t;
                                    if in_row >= f.len() {
                                        continue;
                                    }
                                    let coef = J[i][t];
                                    h0 += coef * f[in_row].coeffs()[0];
                                }
                                let mut h_row = vec![<R::BaseRing as Field>::ZERO; d];
                                h_row[0] = h0;
                                let digits = h_row.decompose_to_vec(d_prime, k_g);
                                chunk.copy_from_slice(&digits[0][..k_g]);
                            });
                    }
                    #[cfg(not(feature = "parallel"))]
                    {
                        for out_row in 0..m_j {
                            let i = out_row % lambda_pj;
                            let b = out_row / lambda_pj;
                            let mut h0 = <R::BaseRing as Field>::ZERO;
                            for t in 0..l_h {
                                let in_row = b * l_h + t;
                                if in_row >= f.len() {
                                    continue;
                                }
                                let coef = J[i][t];
                                h0 += coef * f[in_row].coeffs()[0];
                            }
                            let mut h_row = vec![<R::BaseRing as Field>::ZERO; d];
                            h_row[0] = h0;
                            let digits = h_row.decompose_to_vec(d_prime, k_g);
                            digits0[out_row * k_g..(out_row + 1) * k_g].copy_from_slice(&digits[0][..k_g]);
                        }
                    }
                    ProjDigits::Col0(Arc::new(digits0))
                } else {
                    let mut digits_flat = vec![<R::BaseRing as Field>::ZERO; m_j * d * k_g];
                    #[cfg(feature = "parallel")]
                    {
                        use rayon::prelude::*;
                        digits_flat
                            .par_chunks_mut(d * k_g)
                            .enumerate()
                            .for_each(|(out_row, chunk)| {
                                let i = out_row % lambda_pj;
                                let b = out_row / lambda_pj;
                                let mut h_row = vec![<R::BaseRing as Field>::ZERO; d];
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
                                    let dst = &mut chunk[col * k_g..(col + 1) * k_g];
                                    dst.copy_from_slice(&digits[col][..k_g]);
                                }
                            });
                    }
                    #[cfg(not(feature = "parallel"))]
                    {
                        for out_row in 0..m_j {
                            let i = out_row % lambda_pj;
                            let b = out_row / lambda_pj;
                            let mut h_row = vec![<R::BaseRing as Field>::ZERO; d];
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
                    ProjDigits::Full(Arc::new(digits_flat))
                };

                // cm_g commitments (serial inner loop; outer parallel across instances)
                let n = m * d;
                let g0 = exp::<R>(<R::BaseRing as Field>::ZERO).expect("Exp failed");
                let commits = cm_g_scheme
                    .commit_many_with_serial(n, k_g, |j, out: &mut [R]| {
                        let col = j / m;
                        let row = j - col * m;
                        let out_row = row % m_j;
                        for dig in 0..k_g {
                            if const_coeff_fastpath && col != 0 {
                                out[dig] = g0;
                            } else {
                                let digit = proj.digit(m_j, d, k_g, out_row, col, dig);
                                out[dig] = exp::<R>(digit).expect("Exp failed");
                            }
                        }
                    })
                    .map_err(|e| format!("PiFold: cm_g commit failed: {e:?}"))?;
                let cm_g_inst = commits.into_iter().map(|c| c.as_ref().to_vec()).collect::<Vec<_>>();

                // Phase 2: derive Π_mon coins from (cm_f, cm_g, inst_idx).
                let mut h2 = Sha256::new();
                h2.update(b"SYMPHONY_EXPERIMENTAL_PI_FOLD_COINS_V2");
                h2.update(b"PHASE2_MON_COINS_FROM_CMF_CMG");
                h2.update(&(inst_idx as u64).to_le_bytes());
                h2.update(&cm_f_digest);
                let cm_g_digest: [u8; 32] = {
                    let mut hh = Sha256::new();
                    hh.update(b"cm_g");
                    hh.update(&(cm_g_inst.len() as u64).to_le_bytes());
                    for row in &cm_g_inst {
                        hh.update(&(row.len() as u64).to_le_bytes());
                        for x in row {
                            for c in x.coeffs() {
                                for fp in c.to_base_prime_field_elements() {
                                    hh.update(fp.into_bigint().to_bytes_le());
                                }
                            }
                        }
                    }
                    hh.finalize().into()
                };
                h2.update(&cm_g_digest);
                let seed2: [u8; 32] = h2.finalize().into();
                let mut rng2 = StdRng::from_seed(seed2);

                let mut cba: Vec<(Vec<R>, R::BaseRing, R::BaseRing)> = Vec::with_capacity(rg_params.k_g);
                for _ in 0..rg_params.k_g {
                    // c is a random point (embedded scalars), consistent with how eq(c) is used.
                    let mut c: Vec<R> = Vec::with_capacity(g_nvars);
                    for _ in 0..g_nvars {
                        let x = <R::BaseRing as Field>::BasePrimeField::rand(&mut rng2);
                        c.push(R::from(<R::BaseRing as Field>::from_base_prime_field(x)));
                    }
                    // beta/alpha/rc are sampled small to mimic low-norm scalars.
                    let beta = sample_small_base::<R, _>(&mut rng2);
                    let alpha = sample_small_base::<R, _>(&mut rng2);
                    cba.push((c, beta, alpha));
                }
                let rc: Option<R::BaseRing> = (rg_params.k_g > 1).then(|| sample_small_base::<R, _>(&mut rng2));

                Ok::<_, String>((inst_idx, J, cba, rc, proj, cm_g_inst))
            })
            .collect::<Result<Vec<_>, _>>()?;

        let mut per_inst = per_inst;
        per_inst.sort_by_key(|(i, _, _, _, _, _)| *i);
        for (_i, J, cba, rc, proj, cmg) in per_inst {
            Js.push(J);
            cba_all.push(cba);
            rc_all.push(rc);
            proj_digits_by_inst.push(proj);
            cm_g.push(cmg);
        }
    } else {
        for (inst_idx, cm_f) in cms.iter().enumerate() {
            let f: &[R] = &witnesses[inst_idx];

        transcript.absorb_slice(cm_f);
        let J = derive_J::<R>(transcript, rg_params.lambda_pj, rg_params.l_h);
            Js.push(J.clone());

        // Precompute projected digits once per instance.
        // Layout: digits[(row * d + col) * k_g + dig] where
        // - row ∈ [0, m_j)
        // - col ∈ [0, d)
        // - dig ∈ [0, k_g)
        let k_g = rg_params.k_g;
        let lambda_pj = rg_params.lambda_pj;
        let l_h = rg_params.l_h;
        let d_prime = rg_params.d_prime;

        let mut digits_flat = vec![<R::BaseRing as Field>::ZERO; m_j * d * k_g];

        #[cfg(feature = "parallel")]
        {
            use ark_std::cfg_into_iter;
            let out_ptr = digits_flat.as_mut_ptr() as usize;
            cfg_into_iter!(0..m_j).for_each(|out_row| {
                let i = out_row % lambda_pj;
                let b = out_row / lambda_pj;
                if const_coeff_fastpath {
                    // Constant-coeff specialization: only coeff[0] can be nonzero.
                    let mut h0 = <R::BaseRing as Field>::ZERO;
                    for t in 0..l_h {
                        let in_row = b * l_h + t;
                        if in_row >= f.len() {
                            continue;
                        }
                        let coef = J[i][t];
                        h0 += coef * f[in_row].coeffs()[0];
                    }
                    let mut h_row = vec![<R::BaseRing as Field>::ZERO; d];
                    h_row[0] = h0;
                    let digits = h_row.decompose_to_vec(d_prime, k_g);
                    for dig in 0..k_g {
                        let idx = (out_row * d) * k_g + dig; // col=0
                        unsafe {
                            *(out_ptr as *mut R::BaseRing).add(idx) = digits[0][dig];
                        }
                    }
                } else {
                    let mut h_row = vec![<R::BaseRing as Field>::ZERO; d];
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
                    let mut h0 = <R::BaseRing as Field>::ZERO;
                    for t in 0..l_h {
                        let in_row = b * l_h + t;
                        if in_row >= f.len() {
                            continue;
                        }
                        let coef = J[i][t];
                        h0 += coef * f[in_row].coeffs()[0];
                    }
                    let mut h_row = vec![<R::BaseRing as Field>::ZERO; d];
                    h_row[0] = h0;
                    let digits = h_row.decompose_to_vec(d_prime, k_g);
                    for dig in 0..k_g {
                        digits_flat[(out_row * d) * k_g + dig] = digits[0][dig];
                    }
                } else {
                    let mut h_row = vec![<R::BaseRing as Field>::ZERO; d];
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

        let digits_flat = Arc::new(digits_flat);
        proj_digits_by_inst.push(ProjDigits::Full(digits_flat.clone()));

        // Commit to g^(i) and bind commitments into the transcript before Π_mon challenges.
        //
        // Performance: do this as a *batched* Ajtai commit so we reuse the per-column RNG for all digits.
        let digits_for_commit: ProjDigits<R> = ProjDigits::Full(digits_flat.clone());
        let k_g = rg_params.k_g;
        let n = m * d;
        let g0 = exp::<R>(<R::BaseRing as Field>::ZERO).expect("Exp failed");
        let const_coeff = const_coeff_fastpath;
        let commits = cm_g_scheme
            .commit_many_with(n, k_g, move |j, out: &mut [R]| {
                // out[dig] = g^(dig)[j]
                let col = j / m;
                let row = j - col * m;
                let out_row = row % m_j;
                for dig in 0..k_g {
                    if const_coeff && col != 0 {
                        // In const-coeff fast path, projected digits are zero for columns 1..d-1,
                        // so g^(dig)[j] = exp(0) is constant for those columns.
                        out[dig] = g0;
                    } else {
                        let digit = digits_for_commit.digit(m_j, d, k_g, out_row, col, dig);
                        out[dig] = exp::<R>(digit).expect("Exp failed");
                    }
                }
            })
            .map_err(|e| format!("PiFold: cm_g commit failed: {e:?}"))?;

        let mut cm_g_inst: Vec<Vec<R>> = Vec::with_capacity(k_g);
        for c in commits {
            let c_vec = c.as_ref().to_vec();
            transcript.absorb_slice(&c_vec);
            cm_g_inst.push(c_vec);
        }
        cm_g.push(cm_g_inst);

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
                let mut acc_all0 = <R::BaseRing as Field>::ZERO;
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

    // NOTE: projected digits and cm_g commitments were computed above, and cm_g commitments were
    // absorbed into the transcript before Π_mon coins were derived.

    let t_mon_build = Instant::now();
    for inst_idx in 0..ell {
        let digits = &proj_digits_by_inst[inst_idx];
        for dig in 0..rg_params.k_g {
            let (_c, beta_i, _alpha_i) = &cba_all[inst_idx][dig];

            // mj_compact[col*m_j + out_row] := ev(exp(digit), beta_i)
            let mut mj_compact = vec![<R::BaseRing as Field>::ZERO; m_j * d];
            #[cfg(feature = "parallel")]
            {
                use ark_std::cfg_into_iter;
                let out_ptr = mj_compact.as_mut_ptr() as usize;
                let digits = digits.clone();
                let beta_i = *beta_i;
                let dig_local = dig;
                if const_coeff_fastpath && d > 1 {
                    let g0 = exp::<R>(<R::BaseRing as Field>::ZERO).expect("Exp failed");
                    let mjv_zero = ev(&g0, beta_i);
                    cfg_into_iter!(0..m_j).for_each(|out_row| {
                        let digit0 = digits.digit(m_j, d, rg_params.k_g, out_row, 0, dig_local);
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
                            let digit = digits.digit(m_j, d, rg_params.k_g, out_row, col, dig_local);
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
                    let g0 = exp::<R>(<R::BaseRing as Field>::ZERO).expect("Exp failed");
                    let mjv_zero = ev(&g0, *beta_i);
            for out_row in 0..m_j {
                        let digit0 = digits.digit(m_j, d, rg_params.k_g, out_row, 0, dig);
                        let g = exp::<R>(digit0).expect("Exp failed");
                        mj_compact[out_row] = ev(&g, *beta_i);
                    }
                    for col in 1..d {
                        mj_compact[col * m_j..(col + 1) * m_j].fill(mjv_zero);
                    }
                } else {
                    for out_row in 0..m_j {
                for col in 0..d {
                            let digit = digits.digit(m_j, d, rg_params.k_g, out_row, col, dig);
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
            let mut rc_pow = <R::BaseRing as Field>::ONE;
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
        vec![vec![<R::BaseRing as Field>::ZERO; d]; rg_params.k_g];

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
                    v_digits_folded[dig][col] = <R::BaseRing as Field>::ZERO;
                }
            }
            for inst_idx in 0..ell {
                let b = beta_cts[inst_idx];
                let digits = &proj_digits_by_inst[inst_idx];
                for row in 0..m_j {
                    let w = ts_r[row];
                    for col in 0..d {
                        for dig in 0..rg_params.k_g {
                            let digit = digits.digit(m_j, d, rg_params.k_g, row, col, dig);
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
        let digits = proj_digits_by_inst[inst_idx].clone();
        for dig in 0..rg_params.k_g {
            #[cfg(feature = "parallel")]
            let acc = {
                use ark_std::cfg_into_iter;
                let dig_local = dig;
                let digits = digits.clone();
                cfg_into_iter!(0..m_j)
                    .map(|out_row| {
                        let mut local = R::ZERO;
                        for col in 0..d {
                            let digit = digits.digit(m_j, d, rg_params.k_g, out_row, col, dig_local);
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
                        let digit = digits.digit(m_j, d, rg_params.k_g, out_row, col, dig);
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
            cm_g,
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
