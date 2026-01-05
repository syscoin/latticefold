//! Streaming Sumcheck Prover for Symphony.
//!
//! This implements the sumcheck protocol without materializing O(2^n) evaluation tables.
//! Evaluations are computed on-demand via index-based lookup functions.
//!
//! Per HackMD (<https://hackmd.io/@zkpunk/SkleumdxZl>), the key insight is that
//! structured MLEs can be evaluated at hypercube indices without storing all values.

use ark_std::vec::Vec;
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use std::sync::Arc;
use stark_rings::{OverField, PolyRing, Ring};
use stark_rings_linalg::SparseMatrix;

use latticefold::transcript::Transcript;

/// Concrete streaming MLE variants (no dynamic dispatch in hot loops).
#[derive(Clone)]
pub enum StreamingMleEnum<R: OverField + PolyRing>
where
    R::BaseRing: Ring,
{
    Dense(DenseStreamingMle<R>),
    /// y[row] = (M * w)[row], computed from sparse rows (only used before the first fix).
    SparseMatVec {
        matrix: Arc<SparseMatrix<R>>,
        witness: Arc<Vec<R>>,
        num_vars: usize,
    },
    /// A base-ring scalar table (stored as BaseRing), lifted to R via R::from.
    BaseScalarVec {
        evals: Arc<Vec<R::BaseRing>>,
        num_vars: usize,
    },
    /// eq(bits(index), r) evaluated in the base ring, then lifted to R.
    EqBase {
        scale: R::BaseRing,
        r: Vec<R::BaseRing>,
        one_minus_r: Vec<R::BaseRing>,
    },
}

impl<R: OverField + PolyRing> StreamingMleEnum<R>
where
    R::BaseRing: Ring,
{
    pub fn dense(evals: Vec<R>) -> Self {
        Self::Dense(DenseStreamingMle::new(evals))
    }

    pub fn sparse_mat_vec(matrix: Arc<SparseMatrix<R>>, witness: Arc<Vec<R>>) -> Self {
        let nrows = matrix.nrows;
        assert!(nrows.is_power_of_two(), "nrows must be power-of-two");
        let num_vars = nrows.trailing_zeros() as usize;
        Self::SparseMatVec {
            matrix,
            witness,
            num_vars,
        }
    }

    pub fn base_scalar_vec(num_vars: usize, evals: Arc<Vec<R::BaseRing>>) -> Self {
        Self::BaseScalarVec { evals, num_vars }
    }

    pub fn eq_base(r: Vec<R::BaseRing>) -> Self {
        let one_minus_r = r.iter().copied().map(|x| R::BaseRing::ONE - x).collect();
        Self::EqBase {
            scale: R::BaseRing::ONE,
            r,
            one_minus_r,
        }
    }

    #[inline]
    pub fn num_vars(&self) -> usize {
        match self {
            StreamingMleEnum::Dense(m) => m.num_vars,
            StreamingMleEnum::SparseMatVec { num_vars, .. } => *num_vars,
            StreamingMleEnum::BaseScalarVec { num_vars, .. } => *num_vars,
            StreamingMleEnum::EqBase { r, .. } => r.len(),
        }
    }

    #[inline]
    pub fn eval_at_index(&self, index: usize) -> R {
        match self {
            StreamingMleEnum::Dense(m) => m.evals[index],
            StreamingMleEnum::SparseMatVec {
                matrix, witness, ..
            } => {
                let mut sum = R::ZERO;
                for (coeff, col_idx) in &matrix.coeffs[index] {
                    if *col_idx < witness.len() {
                        sum += *coeff * witness[*col_idx];
                    }
                }
                sum
            }
            StreamingMleEnum::BaseScalarVec { evals, .. } => R::from(evals[index]),
            StreamingMleEnum::EqBase {
                scale,
                r,
                one_minus_r,
            } => {
                let mut prod = R::BaseRing::ONE;
                for i in 0..r.len() {
                    let bit = ((index >> i) & 1) == 1;
                    prod *= if bit { r[i] } else { one_minus_r[i] };
                }
                R::from(*scale * prod)
            }
        }
    }

    /// Fix the next variable (LSB-first) and return a new MLE over one fewer variable.
    pub fn fix_variable(&self, r: R) -> StreamingMleEnum<R> {
        let nv = self.num_vars();
        assert!(nv > 0);
        let half = 1 << (nv - 1);
        match self {
            // Dense: materialize smaller dense.
            StreamingMleEnum::Dense(m) => {
                let new_evals: Vec<R> = (0..half)
                    .map(|i| (R::ONE - r) * m.evals[i << 1] + r * m.evals[(i << 1) | 1])
                    .collect();
                StreamingMleEnum::dense(new_evals)
            }
            // Sparse mat-vec: materialize to dense once fixed (avoids exponential recursion).
            StreamingMleEnum::SparseMatVec { .. } => {
                let new_evals: Vec<R> = (0..half)
                    .map(|i| {
                        let f0 = self.eval_at_index(i << 1);
                        let f1 = self.eval_at_index((i << 1) | 1);
                        (R::ONE - r) * f0 + r * f1
                    })
                    .collect();
                StreamingMleEnum::dense(new_evals)
            }
            // Base-scalar table: keep it base-scalar after fixing (critical for memory).
            //
            // Materializing this into `Vec<R>` blows up memory by ~dim(R)× because `R` stores `d`
            // base-ring coefficients. We only need constant-coefficient values here.
            StreamingMleEnum::BaseScalarVec { evals, num_vars } => {
                let r0 = r.coeffs()[0];
                let one_minus = R::BaseRing::ONE - r0;
                let new_evals: Vec<R::BaseRing> = (0..half)
                    .map(|i| one_minus * evals[i << 1] + r0 * evals[(i << 1) | 1])
                    .collect();
                StreamingMleEnum::BaseScalarVec {
                    evals: Arc::new(new_evals),
                    num_vars: num_vars - 1,
                }
            }
            // EqBase: keep structural (base-field) with updated scale and shortened r vector.
            StreamingMleEnum::EqBase {
                scale,
                r: rr,
                one_minus_r,
            } => {
                let r0 = r.coeffs()[0];
                let eq_factor = (R::BaseRing::ONE - r0) * one_minus_r[0] + r0 * rr[0];
                let new_scale = *scale * eq_factor;
                let new_r = rr[1..].to_vec();
                let new_om = one_minus_r[1..].to_vec();
                StreamingMleEnum::EqBase {
                    scale: new_scale,
                    r: new_r,
                    one_minus_r: new_om,
                }
            }
        }
    }
}

// ============================================================================
// Concrete Streaming MLE Implementations
// ============================================================================

/// Dense MLE (fallback, stores all evaluations).
#[derive(Clone)]
pub struct DenseStreamingMle<R: OverField> {
    evals: Vec<R>,
    num_vars: usize,
}

impl<R: OverField> DenseStreamingMle<R> {
    pub fn new(evals: Vec<R>) -> Self {
        let len = evals.len();
        assert!(len.is_power_of_two(), "Length must be power of 2");
        let num_vars = len.trailing_zeros() as usize;
        Self { evals, num_vars }
    }

    pub fn num_vars(&self) -> usize {
        self.num_vars
    }

    pub fn eval_at_index(&self, index: usize) -> R {
        self.evals[index]
    }
}

// ============================================================================
// Streaming Sumcheck Prover
// ============================================================================

/// Prover message for streaming sumcheck.
#[derive(Clone, Debug)]
pub struct StreamingProverMsg<R: OverField> {
    /// Evaluations at 0, 1, 2, ..., degree
    pub evaluations: Vec<R>,
}

/// Streaming sumcheck proof.
#[derive(Clone, Debug)]
pub struct StreamingProof<R: OverField>(pub Vec<StreamingProverMsg<R>>);

/// Streaming sumcheck prover state.
pub struct StreamingSumcheckState<R: OverField + PolyRing>
where
    R::BaseRing: Ring,
{
    mles: Vec<StreamingMleEnum<R>>,
    pub randomness: Vec<R::BaseRing>,
    num_vars: usize,
    max_degree: usize,
    round: usize,
}

/// Streaming sumcheck prover.
pub struct StreamingSumcheck;

impl StreamingSumcheck {
    /// Initialize prover with streaming MLEs.
    pub fn prover_init<R: OverField + PolyRing>(
        mles: Vec<StreamingMleEnum<R>>,
        nvars: usize,
        degree: usize,
    ) -> StreamingSumcheckState<R> {
        assert!(nvars > 0);
        assert!(!mles.is_empty());
        for m in &mles {
            assert_eq!(m.num_vars(), nvars);
        }

        StreamingSumcheckState {
            mles,
            randomness: Vec::with_capacity(nvars),
            num_vars: nvars,
            max_degree: degree,
            round: 0,
        }
    }

    /// Prove one round of sumcheck.
    pub fn prove_round<R: OverField + PolyRing>(
        state: &mut StreamingSumcheckState<R>,
        v_msg: Option<R::BaseRing>,
        comb_fn: &(dyn Fn(&[R]) -> R + Sync + Send),
    ) -> StreamingProverMsg<R> {
        // Handle previous round's randomness
        if let Some(r) = v_msg {
            assert!(state.round > 0);
            state.randomness.push(r);

            // Fix variable in all MLEs
            let r_ring = R::from(r);
            state.mles = state.mles.iter().map(|m| m.fix_variable(r_ring)).collect();
        } else {
            assert!(state.round == 0);
        }

        state.round += 1;
        assert!(state.round <= state.num_vars);

        let nv = state.mles[0].num_vars();
        let degree = state.max_degree;
        let domain_half = 1 << (nv - 1);
        let num_polys = state.mles.len();

        // Compute round polynomial evaluations at 0, 1, ..., degree
        // Using the Jolt-style optimization: for each b in {0,1}^{nv-1},
        // evaluate combiner at (0, b) and (1, b), then interpolate for higher degrees.

        struct Scratch<R> {
            evals: Vec<R>,
            steps: Vec<R>,
            vals0: Vec<R>,
            vals1: Vec<R>,
            vals: Vec<R>,
            levals: Vec<R>,
        }

        let scratch = || Scratch {
            evals: vec![R::ZERO; degree + 1],
            steps: vec![R::ZERO; num_polys],
            vals0: vec![R::ZERO; num_polys],
            vals1: vec![R::ZERO; num_polys],
            vals: vec![R::ZERO; num_polys],
            levals: vec![R::ZERO; degree + 1],
        };

        // Sequential or parallel iteration over domain
        #[cfg(not(feature = "parallel"))]
        let result = {
            let mut s = scratch();
            for b in 0..domain_half {
                // Evaluate all MLEs at indices 2*b (for x_i=0) and 2*b+1 (for x_i=1)
                for (i, mle) in state.mles.iter().enumerate() {
                    s.vals0[i] = mle.eval_at_index(b << 1);
                    s.vals1[i] = mle.eval_at_index((b << 1) | 1);
                }

                s.levals[0] = comb_fn(&s.vals0);
                s.levals[1] = comb_fn(&s.vals1);

                // Compute steps for interpolation: step[i] = vals1[i] - vals0[i]
                for i in 0..num_polys {
                    s.steps[i] = s.vals1[i] - s.vals0[i];
                    s.vals[i] = s.vals1[i];
                }

                // Extrapolate to higher degrees
                for d in 2..=degree {
                    for i in 0..num_polys {
                        s.vals[i] += s.steps[i];
                    }
                    s.levals[d] = comb_fn(&s.vals);
                }

                // Accumulate
                for (e, l) in s.evals.iter_mut().zip(s.levals.iter()) {
                    *e += *l;
                }
            }
            s.evals
        };

        #[cfg(feature = "parallel")]
        let result = {
            use ark_std::cfg_into_iter;
            let evaluations = cfg_into_iter!(0..domain_half)
                .fold(scratch, |mut s, b| {
                    for (i, mle) in state.mles.iter().enumerate() {
                        s.vals0[i] = mle.eval_at_index(b << 1);
                        s.vals1[i] = mle.eval_at_index((b << 1) | 1);
                    }

                    s.levals[0] = comb_fn(&s.vals0);
                    s.levals[1] = comb_fn(&s.vals1);

                    for i in 0..num_polys {
                        s.steps[i] = s.vals1[i] - s.vals0[i];
                        s.vals[i] = s.vals1[i];
                    }

                    for d in 2..=degree {
                        for i in 0..num_polys {
                            s.vals[i] += s.steps[i];
                        }
                        s.levals[d] = comb_fn(&s.vals);
                    }

                    for (e, l) in s.evals.iter_mut().zip(s.levals.iter()) {
                        *e += *l;
                    }
                    s
                })
                .map(|s| s.evals)
                .reduce(
                    || vec![R::ZERO; degree + 1],
                    |mut acc, evals| {
                        for (a, e) in acc.iter_mut().zip(evals) {
                            *a += e;
                        }
                        acc
                    },
                );
            evaluations
        };

        StreamingProverMsg { evaluations: result }
    }

    /// Run streaming sumcheck as subprotocol.
    pub fn prove_as_subprotocol<R: OverField + PolyRing, T: Transcript<R>>(
        transcript: &mut T,
        mles: Vec<StreamingMleEnum<R>>,
        nvars: usize,
        degree: usize,
        comb_fn: impl Fn(&[R]) -> R + Sync + Send,
    ) -> (StreamingProof<R>, Vec<R::BaseRing>) {
        transcript.absorb(&R::from(nvars as u128));
        transcript.absorb(&R::from(degree as u128));

        let mut state = Self::prover_init(mles, nvars, degree);
        let mut msgs = Vec::with_capacity(nvars);
        let mut v_msg: Option<R::BaseRing> = None;

        for _ in 0..nvars {
            let pm = Self::prove_round(&mut state, v_msg, &comb_fn);
            transcript.absorb_slice(&pm.evaluations);
            msgs.push(pm);

            let r = transcript.get_challenge();
            transcript.absorb(&R::from(r));
            v_msg = Some(r);
        }

        state.randomness.push(v_msg.unwrap());
        (StreamingProof(msgs), state.randomness)
    }

    /// Run streaming sumcheck with hook after `hook_round` challenges.
    pub fn prove_as_subprotocol_with_hook<R: OverField + PolyRing, T: Transcript<R>>(
        transcript: &mut T,
        mles: Vec<StreamingMleEnum<R>>,
        nvars: usize,
        degree: usize,
        comb_fn: impl Fn(&[R]) -> R + Sync + Send,
        hook_round: usize,
        mut hook: impl FnMut(&mut T, &[R::BaseRing]),
    ) -> (StreamingProof<R>, Vec<R::BaseRing>) {
        assert!(hook_round <= nvars);

        transcript.absorb(&R::from(nvars as u128));
        transcript.absorb(&R::from(degree as u128));

        let mut state = Self::prover_init(mles, nvars, degree);
        let mut msgs = Vec::with_capacity(nvars);
        let mut v_msg: Option<R::BaseRing> = None;
        let mut sampled: Vec<R::BaseRing> = Vec::with_capacity(nvars);

        for _ in 0..nvars {
            let pm = Self::prove_round(&mut state, v_msg, &comb_fn);
            transcript.absorb_slice(&pm.evaluations);
            msgs.push(pm);

            let r = transcript.get_challenge();
            transcript.absorb(&R::from(r));
            sampled.push(r);

            if hook_round != 0 && sampled.len() == hook_round {
                hook(transcript, &sampled);
            }

            v_msg = Some(r);
        }

        state.randomness.push(v_msg.unwrap());
        (StreamingProof(msgs), state.randomness)
    }

    /// Run TWO streaming sumchecks with shared randomness (Symphony schedule).
    pub fn prove_two_shared_with_hook<R: OverField + PolyRing, T: Transcript<R>>(
        transcript: &mut T,
        // A
        mles_a: Vec<StreamingMleEnum<R>>,
        nvars_a: usize,
        degree_a: usize,
        comb_fn_a: impl Fn(&[R]) -> R + Sync + Send,
        // B
        mles_b: Vec<StreamingMleEnum<R>>,
        nvars_b: usize,
        degree_b: usize,
        comb_fn_b: impl Fn(&[R]) -> R + Sync + Send,
        // hook
        hook_round: usize,
        mut hook: impl FnMut(&mut T, &[R::BaseRing]),
    ) -> (
        (StreamingProof<R>, Vec<R::BaseRing>),
        (StreamingProof<R>, Vec<R::BaseRing>),
    ) {
        assert!(nvars_a > 0 && nvars_b > 0);
        assert!(hook_round <= nvars_b);

        transcript.absorb(&R::from(nvars_a as u128));
        transcript.absorb(&R::from(degree_a as u128));
        transcript.absorb(&R::from(nvars_b as u128));
        transcript.absorb(&R::from(degree_b as u128));

        let mut state_a = Self::prover_init(mles_a, nvars_a, degree_a);
        let mut state_b = Self::prover_init(mles_b, nvars_b, degree_b);

        let mut msgs_a = Vec::with_capacity(nvars_a);
        let mut msgs_b = Vec::with_capacity(nvars_b);
        let mut sampled: Vec<R::BaseRing> = Vec::with_capacity(nvars_a.max(nvars_b));
        let mut v_msg: Option<R::BaseRing> = None;

        let rounds = nvars_a.max(nvars_b);
        for round_idx in 0..rounds {
            // Prover messages
            if round_idx < nvars_a {
                let pm_a = Self::prove_round(&mut state_a, v_msg, &comb_fn_a);
                transcript.absorb_slice(&pm_a.evaluations);
                msgs_a.push(pm_a);
            }
            if round_idx < nvars_b {
                let pm_b = Self::prove_round(&mut state_b, v_msg, &comb_fn_b);
                transcript.absorb_slice(&pm_b.evaluations);
                msgs_b.push(pm_b);
            }

            // Sample shared randomness
            let r = transcript.get_challenge();
            transcript.absorb(&R::from(r));
            sampled.push(r);

            if hook_round != 0 && sampled.len() == hook_round {
                hook(transcript, &sampled);
            }

            v_msg = Some(r);
        }

        // Push final randomness
        let last_a = sampled[nvars_a - 1];
        let last_b = sampled[nvars_b - 1];
        state_a.randomness.push(last_a);
        state_b.randomness.push(last_b);

        (
            (StreamingProof(msgs_a), state_a.randomness),
            (StreamingProof(msgs_b), state_b.randomness),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_streaming_sumcheck_matches_dense_on_linear_combiner() {
        use stark_rings::cyclotomic_ring::models::frog_ring::RqPoly as R;
        use stark_rings::PolyRing;
        use stark_rings::Ring;
        use latticefold::utils::sumcheck::prover::ProverMsg;
        use latticefold::utils::sumcheck::IPForMLSumcheck;

        // 3 variables => 8 evals
        let nvars = 3usize;
        let len = 1usize << nvars;
        let f1 = (0..len).map(|i| R::from((i + 1) as u128)).collect::<Vec<_>>();
        let f2 = (0..len)
            .map(|i| R::from(((i * 7 + 3) % 97) as u128))
            .collect::<Vec<_>>();

        let m1 = StreamingMleEnum::dense(f1.clone());
        let m2 = StreamingMleEnum::dense(f2.clone());

        let comb = |vals: &[R]| -> R { vals[0] * R::from(13u128) + vals[1] * R::from(29u128) };

        // Streaming: run the protocol directly with fixed challenges (BaseRing elements).
        type K = <R as PolyRing>::BaseRing;
        let challenges = vec![K::from(11u128), K::from(22u128), K::from(33u128)];

        // Asserted sum over {0,1}^n
        let mut st = StreamingSumcheck::prover_init(vec![m1, m2], nvars, 1);
        let mut v_msg = None;
        let mut msgs = Vec::new();
        for r in &challenges {
            let pm = StreamingSumcheck::prove_round(&mut st, v_msg, &comb);
            msgs.push(pm);
            v_msg = Some(*r);
        }

        // Verify the sumcheck identity on the produced univariates:
        // at each round, g_i(0)+g_i(1) must equal the previous claim.
        let mut claim = {
            let mut acc = R::ZERO;
            for idx in 0..len {
                acc += comb(&[f1[idx], f2[idx]]);
            }
            acc
        };
        for (round, pm) in msgs.iter().enumerate() {
            assert_eq!(
                pm.evaluations[0] + pm.evaluations[1],
                claim,
                "round {round} failed"
            );
            // next claim is g_i(r_i) (degree=1 so interpolate linearly)
            let r = R::from(challenges[round]);
            claim = (R::ONE - r) * pm.evaluations[0] + r * pm.evaluations[1];
        }

        // Stronger check: feed the same prover messages into latticefold's *real* verifier state
        // with the chosen randomness, and ensure the produced subclaim matches evaluation at that point.
        let mut vs = IPForMLSumcheck::<R, crate::transcript::PoseidonTranscript<R>>::verifier_init(nvars, 1);
        for (pm, r) in msgs.iter().zip(challenges.iter().copied()) {
            IPForMLSumcheck::<R, crate::transcript::PoseidonTranscript<R>>::verify_round_with_randomness(
                ProverMsg::new(pm.evaluations.clone()),
                &mut vs,
                r,
            );
        }
        let sub = IPForMLSumcheck::<R, crate::transcript::PoseidonTranscript<R>>::check_and_generate_subclaim(vs, {
            let mut acc = R::ZERO;
            for idx in 0..len {
                acc += comb(&[f1[idx], f2[idx]]);
            }
            acc
        })
        .expect("verifier should accept correct proof");

        // Compute expected evaluation by fixing the same point in the prover state.
        //
        // IMPORTANT: In the sumcheck schedule, the verifier samples `r_i` *after* the prover sends
        // the i-th univariate. The prover only applies `r_i` at the *start of the next round*.
        // So after `nvars` rounds, `st.mles` has only applied r_0..r_{nvars-2}; we must still
        // apply the last challenge r_{nvars-1} to reach the full evaluation point.
        let last_r = challenges[nvars - 1];
        let fixed_last = st
            .mles
            .iter()
            .map(|m| m.fix_variable(R::from(last_r)))
            .collect::<Vec<_>>();
        let vals_at_point: Vec<R> = fixed_last.iter().map(|m| m.eval_at_index(0)).collect();
        let expected = comb(&vals_at_point);
        assert_eq!(sub.expected_evaluation, expected, "subclaim mismatch");
    }

    #[test]
    fn test_dense_streaming_mle() {
        use stark_rings::cyclotomic_ring::models::frog_ring::RqPoly as R;
        let evals = vec![R::from(1u128), R::from(2u128), R::from(3u128), R::from(4u128)];
        let mle = DenseStreamingMle::new(evals.clone());
        assert_eq!(mle.num_vars(), 2);
        for i in 0..4 {
            assert_eq!(mle.eval_at_index(i), evals[i]);
        }
    }
}
