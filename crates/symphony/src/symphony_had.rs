//! Symphony Π_had (Figure 1) — Hadamard-to-linear reduction.
//!
//! This is a **protocol-faithful** implementation of Figure 1 at the level of transcript
//! interactions + sumcheck structure:
//! - Verifier coins: `s ∈ K^{log m}` and `α ∈ K`
//! - One degree-3 sumcheck over `{0,1}^{log m}` for the claim:
//!   \[
//!     \sum_{b∈{0,1}^{log m}} \sum_{j=1}^d α^{j-1} · eq(s,b) · (g_{1,j}(b) g_{2,j}(b) - g_{3,j}(b)) = 0
//!   \]
//! - Prover sends evaluations `U_{i,j} := g_{i,j}(r)` at the sumcheck point `r`
//! - Verifier checks Eq. (26).
//!
//! Notes about representation:
//! - The paper runs sumcheck over the extension field `K` and treats `g_{i,j}` as `Z_q` values.
//! - In this codebase we run the ring-adapted sumcheck over `R` but embed `Z_q` scalars as
//!   *constant* ring elements via `R::from(base_ring_element)`. This matches the algebra used in
//!   verification (everything stays in the constant subring).

use latticefold::{
    transcript::Transcript,
    utils::sumcheck::{utils::build_eq_x_r, MLSumcheck, Proof, SumCheckError},
};
use stark_rings::{CoeffRing, OverField, Ring, Zq};
use stark_rings_linalg::SparseMatrix;
use stark_rings_poly::mle::DenseMultilinearExtension;
use crate::symphony_coins::ts_weights;

#[derive(Clone, Debug)]
pub struct PiHadProof<R: OverField> {
    /// log(m) (number of sumcheck rounds).
    pub log_m: usize,
    /// Sumcheck proof (degree-3, `log_m` rounds).
    pub sumcheck: Proof<R>,
    /// Prover message `U ∈ K^{3×d}` in Figure 1, stored as constant ring elements.
    ///
    /// Layout: `U[i][j]` corresponds to `U_{i+1, j}` in the paper (i ∈ {0,1,2}, j ∈ [0..d)).
    pub U: [Vec<R>; 3],
}

/// Output instance of Π_had (Figure 1), reconstructed from a verified `PiHadProof`.
///
/// This corresponds to Figure 1 step 4 output: `(c, r, v ∈ E^3)` where `r` is the sumcheck point
/// and `v_i := Σ_j X^{j-1}·U_{i,j}` (we represent each `v_i` as an `R_q` element by storing its
/// coefficient vector over the base ring).
#[derive(Clone, Debug)]
pub struct PiHadVerifiedOutput<R: OverField>
where
    R::BaseRing: Zq,
{
    pub r: Vec<R::BaseRing>,
    /// U as base-ring scalars (constant terms).
    pub U_ct: [Vec<R::BaseRing>; 3],
    /// v_i as `R_q` elements (coefficients are the U values).
    pub v: [R; 3],
}

/// Prove Π_had for the relation induced by matrices (M1,M2,M3) and witness `f`.
///
/// This proves that the Hadamard relation holds (in coefficient space) for `M1*f`, `M2*f`, `M3*f`.
pub fn prove_pi_had<R: CoeffRing>(
    transcript: &mut impl Transcript<R>,
    M: [&SparseMatrix<R>; 3],
    f: &[R],
) -> Result<PiHadProof<R>, SumCheckError<R>>
where
    R::BaseRing: Zq,
{
    let m = M[0].nrows;
    let n = M[0].ncols;
    assert_eq!(f.len(), n);
    assert!(m.is_power_of_two(), "Π_had assumes m is power of two");
    assert_eq!(M[1].nrows, m);
    assert_eq!(M[2].nrows, m);
    assert_eq!(M[1].ncols, n);
    assert_eq!(M[2].ncols, n);

    let log_m = ark_std::log2(m) as usize;
    let d = R::dimension();

    // Verifier coins (Figure 1 Step 1).
    transcript.absorb_field_element(&R::BaseRing::from(0x4c465053_50494841u128)); // "LFPS_PIHA"
    let s_base = transcript.get_challenges(log_m);
    let alpha_base = transcript.get_challenge();

    // Build eq(s, X) MLE (over R, but values are embedded constants).
    let s_r: Vec<R> = s_base.iter().copied().map(R::from).collect();
    let eq_mle = build_eq_x_r::<R>(&s_r).expect("build_eq_x_r failed");

    // Compute y_i := M_i * f  (vector of length m, ring elements).
    let y = M
        .iter()
        .map(|Mi| Mi.try_mul_vec(f).expect("mat-vec mul failed"))
        .collect::<Vec<Vec<R>>>();

    // Build MLEs g_{i,j} for i∈[3], j∈[d] by extracting coefficient columns and embedding them as constants.
    // MLE list layout: [eq, g1_0..g1_{d-1}, g2_0.., g3_0..]
    let mut mles: Vec<DenseMultilinearExtension<R>> = Vec::with_capacity(1 + 3 * d);
    mles.push(eq_mle);
    for i in 0..3 {
        for j in 0..d {
            let evals = (0..m)
                .map(|row| R::from(y[i][row].coeffs()[j]))
                .collect::<Vec<_>>();
            mles.push(DenseMultilinearExtension::from_evaluations_vec(log_m, evals));
        }
    }

    // Precompute α^{j} embedded as constants.
    let mut alpha_pows = Vec::with_capacity(d);
    let mut pow = R::BaseRing::ONE;
    for _ in 0..d {
        alpha_pows.push(R::from(pow));
        pow *= alpha_base;
    }

    // Sumcheck over comb_fn (Figure 1 Eq. (25) bundled).
    let comb_fn = move |vals: &[R]| -> R {
        // vals[0] = eq(s, b)
        let eq = vals[0];
        let mut acc = R::ZERO;
        // For each coefficient column j:
        // term = α^{j} * eq * (g1_j * g2_j - g3_j)
        for j in 0..d {
            let g1 = vals[1 + j];
            let g2 = vals[1 + d + j];
            let g3 = vals[1 + 2 * d + j];
            acc += alpha_pows[j] * eq * (g1 * g2 - g3);
        }
        acc
    };

    // We need `mles` afterwards to compute the prover message `U = g(r)`.
    // (Future: avoid the clone by returning the fixed/evaluated state from sumcheck.)
    let (sumcheck, prover_state) =
        MLSumcheck::<R, _>::prove_as_subprotocol(transcript, mles.clone(), log_m, 3, comb_fn);

    // The sumcheck point r is the verifier randomness from the prover state.
    // Evaluate g_{i,j}(r) and send U (Figure 1 Step 3).
    let r_poly: Vec<R> = prover_state.randomness.iter().copied().map(R::from).collect();
    let mut U: [Vec<R>; 3] = [Vec::with_capacity(d), Vec::with_capacity(d), Vec::with_capacity(d)];
    for i in 0..3 {
        for j in 0..d {
            // index into mles: 1 + i*d + j
            let idx = 1 + i * d + j;
            let v = mles[idx]
                .evaluate(&r_poly)
                .ok_or(SumCheckError::EvaluationError(
                    stark_rings_poly::polynomials::ArithErrors::InvalidParameters(
                        "MLE evaluate returned None".to_string(),
                    ),
                ))?;
            U[i].push(v);
        }
    }
    transcript.absorb_slice(&U[0]);
    transcript.absorb_slice(&U[1]);
    transcript.absorb_slice(&U[2]);

    Ok(PiHadProof { log_m, sumcheck, U })
}

/// Verify Π_had (Figure 1) and return the reconstructed output `(r, U_ct, v)`.
pub fn verify_pi_had_and_output<R: CoeffRing>(
    transcript: &mut impl Transcript<R>,
    proof: &PiHadProof<R>,
) -> Result<PiHadVerifiedOutput<R>, SumCheckError<R>>
where
    R::BaseRing: Zq,
{
    let d = R::dimension();
    assert_eq!(proof.U[0].len(), d);
    assert_eq!(proof.U[1].len(), d);
    assert_eq!(proof.U[2].len(), d);

    transcript.absorb_field_element(&R::BaseRing::from(0x4c465053_50494841u128)); // "LFPS_PIHA"

    // The verifier coins must be sampled in the same order as in proving.
    // (We don't need to expose them; they are bound via transcript.)
    let log_m = proof.log_m;
    let s_base = transcript.get_challenges(log_m);
    let alpha_base = transcript.get_challenge();

    // Verify sumcheck for claimed sum 0.
    let subclaim = MLSumcheck::<R, _>::verify_as_subprotocol(
        transcript,
        log_m,
        3,
        R::ZERO,
        &proof.sumcheck,
    )?;

    // Verify Eq. (26): eq(s,r) * Σ_j α^j (U1j*U2j - U3j) == e
    let r_poly: Vec<R> = subclaim.point.iter().copied().map(R::from).collect();
    let s_poly: Vec<R> = s_base.iter().copied().map(R::from).collect();
    let eq_sr = latticefold::utils::sumcheck::utils::eq_eval(&s_poly, &r_poly)?;

    let mut pow = R::BaseRing::ONE;
    let mut acc = R::ZERO;
    for j in 0..d {
        let alpha_pow = R::from(pow);
        acc += alpha_pow * (proof.U[0][j] * proof.U[1][j] - proof.U[2][j]);
        pow *= alpha_base;
    }
    let lhs = eq_sr * acc;
    let rhs = subclaim.expected_evaluation;
    if lhs != rhs {
        return Err(SumCheckError::SumCheckFailed(lhs, rhs));
    }

    // Reconstruct r (subclaim point) and output values.
    let r = subclaim.point;
    let U_ct: [Vec<R::BaseRing>; 3] = [
        proof.U[0].iter().map(|x| x.ct()).collect(),
        proof.U[1].iter().map(|x| x.ct()).collect(),
        proof.U[2].iter().map(|x| x.ct()).collect(),
    ];
    let mut v: [R; 3] = [R::ZERO, R::ZERO, R::ZERO];
    for i in 0..3 {
        for j in 0..d {
            v[i].coeffs_mut()[j] = U_ct[i][j];
        }
    }

    transcript.absorb_slice(&proof.U[0]);
    transcript.absorb_slice(&proof.U[1]);
    transcript.absorb_slice(&proof.U[2]);

    Ok(PiHadVerifiedOutput { r, U_ct, v })
}

/// Verify Π_had (Figure 1).
pub fn verify_pi_had<R: CoeffRing>(
    transcript: &mut impl Transcript<R>,
    proof: &PiHadProof<R>,
) -> Result<(), SumCheckError<R>>
where
    R::BaseRing: Zq,
{
    verify_pi_had_and_output(transcript, proof).map(|_| ())
}

/// Deterministically compute the expected U-table from an explicit witness `f`:
/// \(U_{i,j} := \langle g_{i,j}, ts(r)\rangle\) where `g_{i,j}` is the MLE of the `j`-th
/// coefficient column of `(M_i*f)`.
pub fn compute_pi_had_U_from_witness<R: CoeffRing>(
    M: [&SparseMatrix<R>; 3],
    f: &[R],
    r: &[R::BaseRing],
) -> [Vec<R::BaseRing>; 3]
where
    R::BaseRing: Zq,
{
    let m = M[0].nrows;
    let n = M[0].ncols;
    assert_eq!(f.len(), n);
    assert!(m.is_power_of_two());
    let log_m = ark_std::log2(m) as usize;
    assert_eq!(r.len(), log_m);
    let d = R::dimension();

    // y_i := M_i * f
    let y = M
        .iter()
        .map(|Mi| Mi.try_mul_vec(f).expect("mat-vec mul failed"))
        .collect::<Vec<Vec<R>>>();

    // ts(r) weights of length m
    let ts_r_full = ts_weights(r);
    let ts_r = &ts_r_full[..m];

    let mut out: [Vec<R::BaseRing>; 3] = [vec![R::BaseRing::ZERO; d], vec![R::BaseRing::ZERO; d], vec![R::BaseRing::ZERO; d]];
    for i in 0..3 {
        for j in 0..d {
            let mut acc = R::BaseRing::ZERO;
            for row in 0..m {
                acc += y[i][row].coeffs()[j] * ts_r[row];
            }
            out[i][j] = acc;
        }
    }
    out
}

/// **Paper-faithful output relation check (Eq. (24))**, given explicit witness `f`.
///
/// This enforces the missing linkage for Π_had’s output `R_aux_lin`:
/// \[
///   \forall i∈[3], \forall j∈[d]:\ \langle (M_i f)_{*,j}, ts(r)\rangle = U_{i,j}
/// \]
pub fn verify_pi_had_output_relation_with_witness<R: CoeffRing>(
    transcript: &mut impl Transcript<R>,
    M: [&SparseMatrix<R>; 3],
    proof: &PiHadProof<R>,
    f: &[R],
) -> Result<(), SumCheckError<R>>
where
    R::BaseRing: Zq,
{
    let out = verify_pi_had_and_output(transcript, proof)?;
    let expected = compute_pi_had_U_from_witness::<R>(M, f, &out.r);
    if expected != out.U_ct {
        return Err(SumCheckError::SumCheckFailed(R::ONE, R::ZERO));
    }
    Ok(())
}

