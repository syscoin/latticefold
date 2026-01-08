//! π_lin: Phase 2 of the two-phase Symphony protocol.
//!
//! This module implements the reduced relation check for the batched-linear output
//! of Π_fold. Given the folded `(r', c_g, u)` from Phase 1, π_lin proves:
//!
//! For each digit `dig ∈ [0, k_g)`:
//! 1. `AjtaiCommit(g_*^(dig)) = c_g[dig]`  (commitment binding)
//! 2. `⟨eq(r', ·), g_*^(dig)⟩ = u[dig]`   (evaluation correctness)
//!
//! ## Architecture (Option B)
//!
//! To avoid O(n·κ) constraints in DPP, we use a split verification model:
//! - **DPP constrains**: Poseidon trace (binding ρ, γ, r_final), sumcheck verification,
//!   the equation `combined_coeff * g_eval == expected` (with combined_coeff as witness)
//! - **Decapsulator recomputes**: `a_rho(r_final)` and `combined_coeff` from public data
//! - **Key derivation**: `K = H(K0 || combined_coeff)` - enforces mandatory recomputation
//!
//! This makes the Ajtai binding check mandatory for key extraction without constraining
//! the expensive `a_rho(r_final)` computation in the circuit.

use ark_std::vec::Vec;
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use sha2::{Digest, Sha256};
use stark_rings::{OverField, PolyRing, Ring};
use std::sync::Arc;

use latticefold::commitment::AjtaiCommitmentScheme;
use latticefold::transcript::Transcript;
use latticefold::utils::sumcheck::{MLSumcheck, Proof as SumcheckProof};

use crate::streaming_sumcheck::{StreamingMleEnum, StreamingProof, StreamingSumcheck};
use crate::symphony_fold::SymphonyBatchLin;

// ============================================================================
// Domain Separators
// ============================================================================

/// Domain separator for π_lin transcript operations.
/// Uses the same u128 tag pattern as the rest of Symphony.
pub const PILIN_DOMAIN_SEP: u128 = 0x4c465053_50494c4e; // "LFPS_PILN"

/// Domain separator for key derivation.
pub const PILIN_KEY_DOMAIN: &[u8] = b"SYMPHONY_PILIN_KEY_V1";

// ============================================================================
// Proof and Statement Types
// ============================================================================

/// π_lin proof object.
#[derive(Clone, Debug)]
pub struct PiLinProof<R: OverField> {
    /// Sumcheck proofs for each digit.
    pub sumcheck_proofs: Vec<StreamingProof<R>>,
    
    /// Claimed evaluations `g^(dig)(r_final)` for each digit.
    pub g_evals: Vec<R>,
    
    /// The combined coefficients used in the proof (for DPP witness).
    /// These are `eq(r', r_final) + γ * a_rho(r_final)` for each digit.
    /// Decapsulator will recompute and verify these match.
    pub combined_coeffs: Vec<R>,
    // NOTE: `expected_evals` removed - it must come from DPP-bound sumcheck verification,
    // not prover-provided values. The DPP binds `subclaim.expected_evaluation` and decap
    // uses that to verify `combined_coeff * g_eval == subclaim_expected`.
}

/// Values extracted from π_lin for decapsulation.
/// These are derived from the DPP-verified Poseidon trace and sumcheck verification.
#[derive(Clone, Debug)]
pub struct PiLinDecapInputs<R: PolyRing> {
    /// Random linear combination challenges for Ajtai (κ elements).
    /// **Const-coeff optimization**: stored as base ring scalars.
    /// Source: DPP-bound Poseidon transcript coins.
    pub rho0: Vec<R::BaseRing>,
    /// Combining challenge.
    /// Source: DPP-bound Poseidon transcript coin.
    pub gamma: R,
    /// Sumcheck final points for each digit.
    /// Source: DPP-bound sumcheck coin sequence.
    pub r_finals: Vec<Vec<R::BaseRing>>,
    /// Evaluation point from Π_fold statement.
    /// Source: Public from Phase 1 folded output.
    pub r_prime: Vec<R::BaseRing>,
    /// Prover's claimed g(r_final) values.
    /// Source: Proof (prover-provided, but verified via subclaim).
    pub g_evals: Vec<R>,
    /// Prover's claimed combined_coeff values.
    /// Source: Proof (prover-provided, will be recomputed and verified).
    pub combined_coeffs: Vec<R>,
    /// Sumcheck subclaim expected evaluations.
    /// CRITICAL: Must come from DPP-VERIFIED sumcheck, NOT prover-provided.
    /// This is what the DPP arithmetization binds.
    pub subclaim_expected: Vec<R>,
}

/// Configuration for π_lin.
#[derive(Clone, Debug)]
pub struct PiLinConfig {
    /// Ajtai commitment domain.
    pub ajtai_domain: Vec<u8>,
    /// Ajtai commitment seed.
    pub ajtai_seed: [u8; 32],
    /// Number of commitment rows (κ).
    pub kappa: usize,
    /// Witness length per digit (m*d = 2^g_nvars).
    pub n: usize,
}

/// π_lin statement (public inputs).
#[derive(Clone, Debug)]
pub struct PiLinStatement<R: PolyRing> {
    /// The folded batch-linear output from Π_fold.
    pub batch_lin: SymphonyBatchLin<R>,
    /// Ajtai configuration.
    pub config: PiLinConfig,
}

/// π_lin witness (prover-side only).
/// Uses Arc to avoid cloning large witness tables.
#[derive(Clone, Debug)]
pub struct PiLinWitness<R: PolyRing> {
    /// Folded witness vectors: `g_*^(dig)[j]` for `dig ∈ [0, k_g)`, `j ∈ [0, m*d)`.
    /// Wrapped in Arc to avoid O(n) clones per digit.
    pub g_star: Vec<Arc<Vec<R>>>,
}

/// Subclaim output from π_lin verification.
#[derive(Clone, Debug)]
pub struct PiLinSubclaim<R: PolyRing> {
    /// The random evaluation point (output of sumcheck).
    pub r_final: Vec<R::BaseRing>,
    /// The claimed evaluation `g(r_final)`.
    pub g_eval: R,
    /// The combined coefficient at r_final (prover-claimed, verified at decap).
    pub combined_coeff: R,
    /// The expected evaluation from sumcheck subclaim.
    /// CRITICAL: This is DPP-bound and used to verify combined_coeff * g_eval at decap.
    pub expected_evaluation: R,
}

impl PiLinConfig {
    /// Number of variables in the sumcheck (log2 of witness length).
    pub fn g_nvars(&self) -> usize {
        debug_assert!(self.n.is_power_of_two());
        self.n.trailing_zeros() as usize
    }
}

// ============================================================================
// Core Computation Functions
// ============================================================================

/// Compute `a_ρ[j] = ⟨ρ, A[:,j]⟩` for a single column index.
/// 
/// **Const-coeff optimization**: takes precomputed `rho_ring` (lifted from base ring scalars)
/// to avoid repeated `R::from()` calls in hot loop.
#[inline]
pub fn compute_a_rho_at_index_with_rho_ring<R: OverField + PolyRing>(
    domain: &[u8],
    seed: &[u8; 32],
    kappa: usize,
    rho_ring: &[R],
    j: usize,
) -> R
where
    R::BaseRing: Ring,
{
    debug_assert_eq!(rho_ring.len(), kappa);
    let col_seed = AjtaiCommitmentScheme::<R>::derive_col_seed(domain, seed, j as u64);
    let mut rng = ChaCha20Rng::from_seed(col_seed);
    
    let mut sum = R::ZERO;
    for row in 0..kappa {
        let aij = R::rand(&mut rng);
        // rho_ring[row] is precomputed R::from(rho0[row]), so this is const-coeff × ring
        sum += rho_ring[row] * aij;
    }
    sum
}

/// Helper to lift base ring scalars to ring elements once.
#[inline]
pub fn lift_rho0_to_ring<R: OverField + PolyRing>(rho0: &[R::BaseRing]) -> Vec<R>
where
    R::BaseRing: Ring,
{
    rho0.iter().map(|&r| R::from(r)).collect()
}

/// Compute `a_rho(r_final)` - the MLE evaluation of A^T·ρ at point r_final.
///
/// This is the O(n·κ) computation that decapsulator must perform.
/// Parallelized: both eq_weights and a_rho are computed in parallel.
/// Precompute eq weights for all indices using tensor product.
/// This is O(n) instead of O(n·nvars) by exploiting the tensor structure:
/// eq(r, x) = Π_i (r_i·x_i + (1-r_i)·(1-x_i))
///          = Π_i ((1-r_i) + x_i·(2r_i - 1))  [for x_i ∈ {0,1}]
/// 
/// We compute this by iteratively expanding the product.
fn precompute_eq_weights0<R: OverField + PolyRing>(r: &[R::BaseRing], n: usize) -> Vec<R::BaseRing>
where
    R::BaseRing: Ring,
{
    let nvars = r.len();
    assert_eq!(n, 1usize << nvars);

    // Start with weights for 0 variables: [1]
    let mut weights: Vec<R::BaseRing> = Vec::with_capacity(n);
    weights.push(R::BaseRing::ONE);

    // Tensor-expand one variable at a time, preserving LSB-first indexing:
    // after processing k variables, `weights.len() == 2^k` and index bits [0..k).
    for (k, &rk) in r.iter().enumerate() {
        let prev = 1usize << k;
        debug_assert_eq!(weights.len(), prev);
        weights.resize(prev << 1, R::BaseRing::ZERO);

        let one_minus = R::BaseRing::ONE - rk;
        for i in 0..prev {
            let w = weights[i];
            weights[i] = w * one_minus;
            weights[i + prev] = w * rk;
        }
    }

    debug_assert_eq!(weights.len(), n);
    weights
}

#[cfg(feature = "parallel")]
pub fn compute_a_rho_mle_eval<R: OverField + PolyRing>(
    domain: &[u8],
    seed: &[u8; 32],
    kappa: usize,
    n: usize,
    rho0: &[R::BaseRing],
    r_final: &[R::BaseRing],
) -> R
where
    R::BaseRing: Ring + Send + Sync,
    R: Send + Sync,
{
    let nvars = r_final.len();
    assert_eq!(n, 1 << nvars);
    
    // Precompute eq weights once: O(n) instead of O(n·nvars) per-index
    let eq_weights0 = precompute_eq_weights0::<R>(r_final, n);
    
    // Precompute rho_ring once (avoid R::from() in hot loop)
    let rho_ring: Vec<R> = lift_rho0_to_ring(rho0);
    
    // Parallel computation: compute weighted sum Σ_j eq_weights[j] * a_rho[j]
    let domain_owned = domain.to_vec();
    let seed_owned = *seed;
    
    (0..n)
        .into_par_iter()
        .map(|j| {
            // Compute a_rho[j] with precomputed rho_ring
            let a_rho_j: R = compute_a_rho_at_index_with_rho_ring(&domain_owned, &seed_owned, kappa, &rho_ring, j);
            R::from(eq_weights0[j]) * a_rho_j
        })
        .reduce(|| R::ZERO, |a, b| a + b)
}

/// Compute `a_rho(r_final)` - the MLE evaluation of A^T·ρ at point r_final.
///
/// This is the O(n·κ) computation that decapsulator must perform.
/// Sequential version for non-parallel builds.
#[cfg(not(feature = "parallel"))]
pub fn compute_a_rho_mle_eval<R: OverField + PolyRing>(
    domain: &[u8],
    seed: &[u8; 32],
    kappa: usize,
    n: usize,
    rho0: &[R::BaseRing],
    r_final: &[R::BaseRing],
) -> R
where
    R::BaseRing: Ring,
{
    let nvars = r_final.len();
    assert_eq!(n, 1 << nvars);
    
    // Precompute eq weights once: O(n) instead of O(n·nvars) per-index
    let eq_weights0 = precompute_eq_weights0::<R>(r_final, n);
    
    // Precompute rho_ring once (avoid R::from() in hot loop)
    let rho_ring: Vec<R> = lift_rho0_to_ring(rho0);
    
    let mut sum = R::ZERO;
    for j in 0..n {
        // Compute a_rho[j] with precomputed rho_ring
        let a_rho_j: R = compute_a_rho_at_index_with_rho_ring(domain, seed, kappa, &rho_ring, j);
        sum += R::from(eq_weights0[j]) * a_rho_j;
    }
    
    sum
}

/// Evaluate the equality polynomial eq(r, x) at a point.
pub fn eval_eq_at_point<R: OverField + PolyRing>(r: &[R::BaseRing], x: &[R::BaseRing]) -> R
where
    R::BaseRing: Ring,
{
    assert_eq!(r.len(), x.len());
    let mut prod = R::ONE;
    for (ri, xi) in r.iter().zip(x.iter()) {
        let one = R::BaseRing::ONE;
        let term = *ri * *xi + (one - *ri) * (one - *xi);
        prod *= R::from(term);
    }
    prod
}

/// Evaluate a dense MLE at a point - auto-selects parallel version when available.
#[cfg(feature = "parallel")]
fn eval_mle_at_point_auto<R: OverField + PolyRing>(evals: &[R], point: &[R::BaseRing]) -> R
where
    R::BaseRing: Ring + Send + Sync,
    R: Send + Sync,
{
    let nvars = point.len();
    assert_eq!(evals.len(), 1 << nvars);
    
    if nvars == 0 {
        return evals[0];
    }
    
    let mut buf = evals.to_vec();
    let mut size = evals.len();
    
    // Parallel threshold: only parallelize if worth it
    const PAR_THRESHOLD: usize = 1024;
    
    for r in point.iter() {
        let half = size / 2;
        let r_ring = R::from(*r);
        let one_minus_r = R::from(R::BaseRing::ONE - *r);
        
        if half >= PAR_THRESHOLD {
            // Parallel: compute new values into a fresh buffer
            let new_buf: Vec<R> = (0..half)
                .into_par_iter()
                .map(|i| one_minus_r * buf[2 * i] + r_ring * buf[2 * i + 1])
                .collect();
            buf = new_buf;
        } else {
            // Sequential for small sizes (in-place)
            for i in 0..half {
                buf[i] = one_minus_r * buf[2 * i] + r_ring * buf[2 * i + 1];
            }
        }
        size = half;
    }
    
    buf[0]
}

/// Fallback for non-parallel builds.
#[cfg(not(feature = "parallel"))]
fn eval_mle_at_point_auto<R: OverField + PolyRing>(evals: &[R], point: &[R::BaseRing]) -> R
where
    R::BaseRing: Ring,
{
    eval_mle_at_point(evals, point)
}

// ============================================================================
// Prover
// ============================================================================

/// Prove π_lin: Phase 2 of Symphony.
///
/// # Performance Notes
///
/// **Memory**: Uses streaming `AjtaiRho` MLE - avoids O(n) upfront allocation.
/// Memory is O(n) only during sumcheck rounds (after first variable fix materializes).
/// `g_star` witness (O(n) per digit) is the main memory consumer.
/// 
/// **Compute**: O(n·κ) for a_rho(r_final), O(n·k_g) for sumcheck, O(n) for g_eval.
/// All parallelized when `parallel` feature is enabled.
pub fn prove_pilin<R, T>(
    transcript: &mut T,
    statement: &PiLinStatement<R>,
    witness: &PiLinWitness<R>,
) -> PiLinProof<R>
where
    R: OverField + PolyRing,
    R::BaseRing: Ring,
    T: Transcript<R>,
{
    let k_g = statement.batch_lin.u.len();
    let g_nvars = statement.config.g_nvars();
    let kappa = statement.config.kappa;
    
    // Validation
    assert_eq!(statement.batch_lin.c_g.len(), k_g);
    assert_eq!(witness.g_star.len(), k_g);
    for dig in 0..k_g {
        assert_eq!(statement.batch_lin.c_g[dig].len(), kappa);
        assert_eq!(witness.g_star[dig].len(), statement.config.n);
    }
    assert_eq!(statement.batch_lin.r_prime.len(), g_nvars);
    
    // Domain separator (proper u128 tag)
    transcript.absorb_field_element(&R::BaseRing::from(PILIN_DOMAIN_SEP));
    
    // Step 1: Sample ρ[κ] for Ajtai random linear combination
    // **Const-coeff optimization**: keep as base ring scalars to enable fast scalar×ring mul
    // Arc to avoid per-digit cloning
    let rho0: Arc<Vec<R::BaseRing>> = Arc::new((0..kappa)
        .map(|_| transcript.get_challenge())
        .collect());
    
    // Step 2: Sample γ for combining checks
    let gamma = R::from(transcript.get_challenge());
    
    // Step 3: Build streaming a_ρ MLE (computes on-demand, avoids O(n) memory)
    // After first variable fix, this will materialize to dense, but we avoid
    // the upfront allocation.
    
    // Step 4: Run k_g sumchecks
    let mut sumcheck_proofs = Vec::with_capacity(k_g);
    let mut g_evals = Vec::with_capacity(k_g);
    let mut combined_coeffs = Vec::with_capacity(k_g);
    
    for dig in 0..k_g {
        // Build MLEs - using streaming AjtaiRho to avoid O(n) upfront materialization
        let r_prime_base: Vec<R::BaseRing> = statement.batch_lin.r_prime.clone();
        let eq_mle = StreamingMleEnum::<R>::eq_base(r_prime_base.clone());
        let a_rho_mle = StreamingMleEnum::<R>::ajtai_rho(
            statement.config.ajtai_domain.clone(),
            statement.config.ajtai_seed,
            kappa,
            rho0.clone(),  // Base ring scalars for const-coeff fast path
            g_nvars,
        );
        // Use Arc - no clone needed since witness already uses Arc<Vec<R>>
        let g_mle = StreamingMleEnum::<R>::dense_arc(witness.g_star[dig].clone());
        
        let mles = vec![eq_mle, a_rho_mle, g_mle];
        
        // Combiner: (eq + γ * a_ρ) * g
        let gamma_local = gamma;
        let comb_fn = move |vals: &[R]| -> R {
            vals[0] * vals[2] + gamma_local * vals[1] * vals[2]
        };
        
        let (proof, randomness) = StreamingSumcheck::prove_as_subprotocol(
            transcript,
            mles,
            g_nvars,
            2,
            comb_fn,
        );
        
        // Compute values at r_final (uses parallel version when feature enabled)
        let g_eval = eval_mle_at_point_auto::<R>(&witness.g_star[dig], &randomness);
        let eq_at_rfinal = eval_eq_at_point::<R>(&r_prime_base, &randomness);
        // Compute a_rho(r_final) via streaming (O(n·κ) but avoids O(n) storage)
        let a_rho_at_rfinal: R = compute_a_rho_mle_eval(
            &statement.config.ajtai_domain,
            &statement.config.ajtai_seed,
            kappa,
            statement.config.n,
            &rho0,  // Base ring scalars for const-coeff optimization
            &randomness,
        );
        let combined_coeff = eq_at_rfinal + gamma * a_rho_at_rfinal;
        // Note: expected_eval = combined_coeff * g_eval is verified by sumcheck
        // and will be bound by DPP as subclaim.expected_evaluation
        
        sumcheck_proofs.push(proof);
        g_evals.push(g_eval);
        combined_coeffs.push(combined_coeff);
    }
    
    // Step 5: Absorb proof values into transcript
    transcript.absorb_slice(&g_evals);
    transcript.absorb_slice(&combined_coeffs);
    
    PiLinProof {
        sumcheck_proofs,
        g_evals,
        combined_coeffs,
    }
}

// ============================================================================
// Verifier (for interactive/DPP use)
// ============================================================================

/// Convert streaming proof to non-streaming format.
fn streaming_to_nonstreaming<R: OverField>(proof: &StreamingProof<R>) -> SumcheckProof<R> {
    use latticefold::utils::sumcheck::prover::ProverMsg;
    SumcheckProof::new(
        proof.0.iter().map(|m| ProverMsg::new(m.evaluations.clone())).collect(),
    )
}

/// Verify π_lin sumcheck structure (for DPP).
/// 
/// This verifies the sumcheck and that `combined_coeff * g_eval == expected`.
/// It does NOT verify that combined_coeff was computed correctly - that's for decapsulation.
pub fn verify_pilin_structure<R, T>(
    transcript: &mut T,
    statement: &PiLinStatement<R>,
    proof: &PiLinProof<R>,
) -> Result<Vec<PiLinSubclaim<R>>, String>
where
    R: OverField + PolyRing,
    R::BaseRing: Ring,
    T: Transcript<R>,
{
    let k_g = statement.batch_lin.u.len();
    let g_nvars = statement.config.g_nvars();
    let kappa = statement.config.kappa;
    
    // Validation
    if proof.sumcheck_proofs.len() != k_g {
        return Err(format!("PiLin: expected {} sumcheck proofs", k_g));
    }
    if proof.g_evals.len() != k_g || proof.combined_coeffs.len() != k_g {
        return Err("PiLin: proof length mismatch".into());
    }
    
    // Domain separator
    transcript.absorb_field_element(&R::BaseRing::from(PILIN_DOMAIN_SEP));
    
    // Sample ρ, γ (must match prover - use base ring scalars for const-coeff optimization)
    let rho0: Vec<R::BaseRing> = (0..kappa)
        .map(|_| transcript.get_challenge())
        .collect();
    let gamma = R::from(transcript.get_challenge());
    
    // Compute c_ρ for claimed sums (const-coeff: R::from(scalar) * c_g is O(d) not O(d²))
    let c_rho: Vec<R> = statement.batch_lin.c_g.iter()
        .map(|c_g_dig| {
            c_g_dig.iter().zip(rho0.iter())
                .map(|(c, r)| R::from(*r) * *c)
                .fold(R::ZERO, |a, b| a + b)
        })
        .collect();
    
    // Verify sumchecks
    let mut subclaims = Vec::with_capacity(k_g);
    
    for dig in 0..k_g {
        let claimed_sum = statement.batch_lin.u[dig] + gamma * c_rho[dig];
        let nonstreaming_proof = streaming_to_nonstreaming(&proof.sumcheck_proofs[dig]);
        
        let subclaim = MLSumcheck::<R, T>::verify_as_subprotocol(
            transcript,
            g_nvars,
            2,
            claimed_sum,
            &nonstreaming_proof,
        ).map_err(|e| format!("PiLin digit {}: sumcheck failed: {:?}", dig, e))?;
        
        // Check: combined_coeff * g_eval == expected (structure check)
        let expected = proof.combined_coeffs[dig] * proof.g_evals[dig];
        if expected != subclaim.expected_evaluation {
            return Err(format!(
                "PiLin digit {}: combined_coeff * g_eval != subclaim.expected",
                dig
            ));
        }
        
        subclaims.push(PiLinSubclaim {
            r_final: subclaim.point.clone(),
            g_eval: proof.g_evals[dig],
            combined_coeff: proof.combined_coeffs[dig],
            expected_evaluation: subclaim.expected_evaluation,
        });
    }
    
    // Absorb proof values (must match prover)
    transcript.absorb_slice(&proof.g_evals);
    transcript.absorb_slice(&proof.combined_coeffs);
    
    Ok(subclaims)
}

// ============================================================================
// Decapsulation
// ============================================================================

/// Decapsulation error types.
#[derive(Debug)]
pub enum DecapError {
    /// The recomputed combined_coeff doesn't match the proof.
    CombinedCoeffMismatch { digit: usize },
    /// The equation combined_coeff * g_eval != expected failed.
    EvalCheckFailed { digit: usize },
}

/// Verify π_lin binding and derive key component.
///
/// This is the mandatory O(n·κ) computation that enforces Ajtai binding.
/// Returns the key component that must be included in the final key derivation.
/// Parallelized: each digit's O(n·κ) computation runs in parallel.
#[cfg(feature = "parallel")]
pub fn decapsulate_pilin<R: OverField + PolyRing>(
    inputs: &PiLinDecapInputs<R>,
    config: &PiLinConfig,
) -> Result<Vec<u8>, DecapError>
where
    R::BaseRing: Ring + ark_ff::Field + Send + Sync,
    R: Send + Sync,
{
    use ark_ff::{BigInteger, Field, PrimeField};
    
    let k_g = inputs.g_evals.len();
    
    // Parallel recomputation and verification for each digit
    let results: Vec<Result<R, DecapError>> = (0..k_g)
        .into_par_iter()
        .map(|dig| {
            // O(n·κ) work: recompute a_rho(r_final)
            let a_rho_at_rfinal: R = compute_a_rho_mle_eval(
                &config.ajtai_domain,
                &config.ajtai_seed,
                config.kappa,
                config.n,
                &inputs.rho0,  // Base ring scalars for const-coeff optimization
                &inputs.r_finals[dig],
            );
            
            // Compute eq(r', r_final)
            let eq_at_rfinal = eval_eq_at_point::<R>(&inputs.r_prime, &inputs.r_finals[dig]);
            
            // Recompute combined_coeff
            let recomputed = eq_at_rfinal + inputs.gamma * a_rho_at_rfinal;
            
            // Verify it matches what prover claimed
            if recomputed != inputs.combined_coeffs[dig] {
                return Err(DecapError::CombinedCoeffMismatch { digit: dig });
            }
            
            // Verify the equation holds against DPP-bound subclaim
            // This is the critical check: combined_coeff * g_eval must equal
            // what the DPP-verified sumcheck produced
            let expected = recomputed * inputs.g_evals[dig];
            if expected != inputs.subclaim_expected[dig] {
                return Err(DecapError::EvalCheckFailed { digit: dig });
            }
            
            Ok(recomputed)
        })
        .collect();
    
    // Collect results, returning first error if any
    let mut recomputed_coeffs = Vec::with_capacity(k_g);
    for result in results {
        recomputed_coeffs.push(result?);
    }
    
    // Derive key component from verified combined_coeffs
    let mut hasher = Sha256::new();
    hasher.update(PILIN_KEY_DOMAIN);
    hasher.update(&(recomputed_coeffs.len() as u64).to_le_bytes());
    for coeff in &recomputed_coeffs {
        for c in coeff.coeffs() {
            for fp in c.to_base_prime_field_elements() {
                hasher.update(fp.into_bigint().to_bytes_le());
            }
        }
    }
    
    Ok(hasher.finalize().to_vec())
}

/// Verify π_lin binding and derive key component (sequential version).
#[cfg(not(feature = "parallel"))]
pub fn decapsulate_pilin<R: OverField + PolyRing>(
    inputs: &PiLinDecapInputs<R>,
    config: &PiLinConfig,
) -> Result<Vec<u8>, DecapError>
where
    R::BaseRing: Ring + ark_ff::Field,
{
    use ark_ff::{BigInteger, Field, PrimeField};
    
    let k_g = inputs.g_evals.len();
    let mut recomputed_coeffs = Vec::with_capacity(k_g);
    
    for dig in 0..k_g {
        // O(n·κ) work: recompute a_rho(r_final)
        let a_rho_at_rfinal: R = compute_a_rho_mle_eval(
            &config.ajtai_domain,
            &config.ajtai_seed,
            config.kappa,
            config.n,
            &inputs.rho0,  // Base ring scalars for const-coeff optimization
            &inputs.r_finals[dig],
        );
        
        // Compute eq(r', r_final)
        let eq_at_rfinal = eval_eq_at_point::<R>(&inputs.r_prime, &inputs.r_finals[dig]);
        
        // Recompute combined_coeff
        let recomputed = eq_at_rfinal + inputs.gamma * a_rho_at_rfinal;
        
        // Verify it matches what prover claimed
        if recomputed != inputs.combined_coeffs[dig] {
            return Err(DecapError::CombinedCoeffMismatch { digit: dig });
        }
        
        // Verify the equation holds against DPP-bound subclaim
        let expected = recomputed * inputs.g_evals[dig];
        if expected != inputs.subclaim_expected[dig] {
            return Err(DecapError::EvalCheckFailed { digit: dig });
        }
        
        recomputed_coeffs.push(recomputed);
    }
    
    // Derive key component from verified combined_coeffs
    let mut hasher = Sha256::new();
    hasher.update(PILIN_KEY_DOMAIN);
    hasher.update(&(recomputed_coeffs.len() as u64).to_le_bytes());
    for coeff in &recomputed_coeffs {
        for c in coeff.coeffs() {
            for fp in c.to_base_prime_field_elements() {
                hasher.update(fp.into_bigint().to_bytes_le());
            }
        }
    }
    
    Ok(hasher.finalize().to_vec())
}

/// Full key derivation including π_lin binding.
///
/// K = H(PILIN_KEY_DOMAIN || K0 || combined_coeffs)
pub fn derive_key_with_pilin<R: OverField + PolyRing>(
    k0: &[u8],
    inputs: &PiLinDecapInputs<R>,
    config: &PiLinConfig,
) -> Result<[u8; 32], DecapError>
where
    R::BaseRing: Ring + ark_ff::Field,
{
    // First verify the π_lin binding (mandatory O(n·κ) work)
    let pilin_component = decapsulate_pilin(inputs, config)?;
    
    // Derive final key
    let mut hasher = Sha256::new();
    hasher.update(PILIN_KEY_DOMAIN);
    hasher.update(k0);
    hasher.update(&pilin_component);
    
    let result = hasher.finalize();
    let mut key = [0u8; 32];
    key.copy_from_slice(&result);
    Ok(key)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use ark_std::UniformRand;
    use cyclotomic_rings::rings::FrogPoseidonConfig as PC;
    use stark_rings::cyclotomic_ring::models::frog_ring::RqPoly;
    use crate::transcript::PoseidonTranscript;
    
    type R = RqPoly;
    
    fn create_test_statement_and_witness(
        kappa: usize,
        g_nvars: usize,
        k_g: usize,
    ) -> (PiLinStatement<R>, PiLinWitness<R>) {
        let n = 1 << g_nvars;
        let ajtai_domain = b"TEST_PILIN".to_vec();
        let ajtai_seed = [0u8; 32];
        
        let ajtai = AjtaiCommitmentScheme::<R>::seeded(&ajtai_domain, ajtai_seed, kappa, n);
        
        let mut rng = rand::thread_rng();
        let mut g_star = Vec::with_capacity(k_g);
        let mut c_g = Vec::with_capacity(k_g);
        
        for _ in 0..k_g {
            let g: Vec<R> = (0..n).map(|_| R::rand(&mut rng)).collect();
            let c: Vec<R> = ajtai.commit(&g).expect("commit").as_ref().to_vec();
            g_star.push(Arc::new(g));
            c_g.push(c);
        }
        
        let r_prime: Vec<<R as PolyRing>::BaseRing> = (0..g_nvars)
            .map(|_| <R as PolyRing>::BaseRing::rand(&mut rng))
            .collect();
        
        let u: Vec<R> = g_star.iter()
            .map(|g| eval_mle_at_point_auto::<R>(g.as_ref(), &r_prime))  // Tests use small sizes
            .collect();
        
        let statement = PiLinStatement {
            batch_lin: SymphonyBatchLin { r_prime, c_g, u },
            config: PiLinConfig { ajtai_domain, ajtai_seed, kappa, n },
        };
        
        (statement, PiLinWitness { g_star })
    }
    
    #[test]
    fn test_pilin_prove_verify_structure() {
        let (statement, witness) = create_test_statement_and_witness(4, 3, 2);
        
        let mut prover_ts: PoseidonTranscript<R> = PoseidonTranscript::empty::<PC>();
        let proof = prove_pilin(&mut prover_ts, &statement, &witness);
        
        let mut verifier_ts: PoseidonTranscript<R> = PoseidonTranscript::empty::<PC>();
        let subclaims = verify_pilin_structure(&mut verifier_ts, &statement, &proof)
            .expect("structure verification failed");
        
        assert_eq!(subclaims.len(), 2);
    }
    
    #[test]
    fn test_pilin_decapsulation() {
        let (statement, witness) = create_test_statement_and_witness(4, 3, 2);
        let kappa = statement.config.kappa;
        let k_g = statement.batch_lin.u.len();
        
        let mut prover_ts: PoseidonTranscript<R> = PoseidonTranscript::empty::<PC>();
        let proof = prove_pilin(&mut prover_ts, &statement, &witness);
        
        // Simulate extracting inputs from verified DPP
        let mut replay_ts: PoseidonTranscript<R> = PoseidonTranscript::empty::<PC>();
        replay_ts.absorb_field_element(&<R as PolyRing>::BaseRing::from(PILIN_DOMAIN_SEP));
        
        // Sample rho0 as base ring scalars (const-coeff optimization)
        let rho0: Vec<<R as PolyRing>::BaseRing> = (0..kappa)
            .map(|_| replay_ts.get_challenge())
            .collect();
        let gamma = R::from(replay_ts.get_challenge());
        
        // Get r_finals from sumcheck verification
        let mut verify_ts: PoseidonTranscript<R> = PoseidonTranscript::empty::<PC>();
        let subclaims = verify_pilin_structure(&mut verify_ts, &statement, &proof).unwrap();
        
        let r_finals: Vec<Vec<<R as PolyRing>::BaseRing>> = subclaims.iter()
            .map(|sc| sc.r_final.clone())
            .collect();
        
        // Get subclaim_expected from the DPP-verified subclaims
        let subclaim_expected: Vec<R> = subclaims.iter()
            .map(|sc| sc.expected_evaluation)
            .collect();
        
        let inputs = PiLinDecapInputs {
            rho0,
            gamma,
            r_finals,
            r_prime: statement.batch_lin.r_prime.clone(),
            g_evals: proof.g_evals.clone(),
            combined_coeffs: proof.combined_coeffs.clone(),
            subclaim_expected,  // From DPP-verified sumcheck, not prover!
        };
        
        // Decapsulation should succeed
        let key_component = decapsulate_pilin(&inputs, &statement.config)
            .expect("decapsulation failed");
        
        assert!(!key_component.is_empty());
    }
    
    #[test]
    fn test_pilin_rejects_wrong_combined_coeff() {
        let (statement, witness) = create_test_statement_and_witness(4, 3, 2);
        let kappa = statement.config.kappa;
        
        let mut prover_ts: PoseidonTranscript<R> = PoseidonTranscript::empty::<PC>();
        let mut proof = prove_pilin(&mut prover_ts, &statement, &witness);
        
        // Tamper with combined_coeff
        proof.combined_coeffs[0] += R::ONE;
        
        // Structure verification should still pass (it uses prover's claimed values)
        let mut verify_ts: PoseidonTranscript<R> = PoseidonTranscript::empty::<PC>();
        // This will fail because combined_coeff * g_eval won't match expected
        let result = verify_pilin_structure(&mut verify_ts, &statement, &proof);
        assert!(result.is_err());
    }
    
    #[test]
    fn test_pilin_decap_rejects_tampered_proof() {
        // This test demonstrates that even if an attacker could bypass the DPP/structure
        // check (which they can't), the decapsulation recomputation would catch the lie.
        // In reality, the structure check also catches tampering (tested above), so this
        // is a defense-in-depth verification.
        
        let (statement, witness) = create_test_statement_and_witness(4, 3, 2);
        let kappa = statement.config.kappa;
        
        let mut prover_ts: PoseidonTranscript<R> = PoseidonTranscript::empty::<PC>();
        let proof = prove_pilin(&mut prover_ts, &statement, &witness);
        
        // Get valid r_finals and subclaim_expected from honest verification
        let mut verify_ts: PoseidonTranscript<R> = PoseidonTranscript::empty::<PC>();
        let subclaims = verify_pilin_structure(&mut verify_ts, &statement, &proof).unwrap();
        let r_finals: Vec<Vec<<R as PolyRing>::BaseRing>> = subclaims.iter()
            .map(|sc| sc.r_final.clone())
            .collect();
        let subclaim_expected: Vec<R> = subclaims.iter()
            .map(|sc| sc.expected_evaluation)
            .collect();
        
        // Replay transcript to get ρ, γ (rho0 as base ring scalars)
        let mut replay_ts: PoseidonTranscript<R> = PoseidonTranscript::empty::<PC>();
        replay_ts.absorb_field_element(&<R as PolyRing>::BaseRing::from(PILIN_DOMAIN_SEP));
        let rho0: Vec<<R as PolyRing>::BaseRing> = (0..kappa).map(|_| replay_ts.get_challenge()).collect();
        let gamma = R::from(replay_ts.get_challenge());
        
        // Create tampered inputs (simulate an adversary who claims wrong combined_coeff)
        // The subclaim_expected comes from the HONEST DPP verification, so it won't match
        // the tampered combined_coeff * g_eval
        let mut tampered_combined_coeffs = proof.combined_coeffs.clone();
        tampered_combined_coeffs[0] += R::ONE;
        
        let inputs = PiLinDecapInputs {
            rho0,
            gamma,
            r_finals,
            r_prime: statement.batch_lin.r_prime.clone(),
            g_evals: proof.g_evals.clone(),
            combined_coeffs: tampered_combined_coeffs,  // Tampered!
            subclaim_expected,  // From honest DPP - will NOT match tampered coeff
        };
        
        // Decapsulation catches the lie by recomputing combined_coeff
        let result = decapsulate_pilin(&inputs, &statement.config);
        assert!(matches!(result, Err(DecapError::CombinedCoeffMismatch { .. })));
    }
}
