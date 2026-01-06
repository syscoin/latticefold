//! Symphony-style approximate range proof (random projection) — **prototype**
//!
//! This module is an **experimental scaffold** to help implement Symphony's
//! approximate range proof for ring vectors (ePrint 2025/1905, Section 3.4).
//!
//! Important:
//! - This is **NOT** a complete or secure drop-in replacement for LatticeFold+'s `rgchk`.
//! - The full Symphony construction proves monomiality *and* a linear consistency check
//!   that ties the projected digits back to the committed witness.
//! - Here we implement the mechanical steps: projection + balanced digit decomposition + `Exp`,
//!   and reuse the existing monomial set-check over the produced vectors.
//! - Do **not** interpret this as “hash-free verification”: Symphony still relies on RO/FS in ROM.

use ark_std::log2;
use latticefold::transcript::Transcript;
use ark_std::vec::Vec;
use stark_rings::{
    balanced_decomposition::{Decompose, DecomposeToVec},
    exp,
    psi,
    CoeffRing,
    OverField,
    PolyRing,
    Ring,
    Zq,
};
use crate::setchk::{In, MonomialSet, Out};
use crate::symphony_coins::{derive_J, ts_weights};

/// Parameters for Symphony-style random projection.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RPParams {
    /// Projection block size ℓ_h (must divide n).
    pub l_h: usize,
    /// Projection output length λ_pj (paper uses 256 as a typical choice).
    pub lambda_pj: usize,
    /// Number of digits (k_g in the paper).
    pub k_g: usize,
    /// Digit base d' (paper uses d' = d - 2).
    pub d_prime: u128,
}

impl Default for RPParams {
    fn default() -> Self {
        Self {
            l_h: 64,
            lambda_pj: 256,
            // With Symphony-style *small* projection coefficients, k_g can be modest.
            // (If you sample J uniformly in Zq, coefficients explode and decomposition fails.)
            k_g: 4,
            // Use the same base as the existing `rgchk` decomposition (b = d/2).
            // Symphony uses d' = d-2, but we keep b=d/2 here until we port the exact
            // Symphony digit decomposition/analysis.
            d_prime: 8,
        }
    }
}

#[derive(Clone, Debug)]
pub struct RPRangeProof<R: PolyRing> {
    /// Parameters used to derive J and decompose digits.
    pub params: RPParams,
    /// Projection output row count `m_J := n·λ_pj/ℓ_h` (Figure 3 / Eq. (44) uses M4 with `m_J` rows).
    pub m_j: usize,
    /// Row count used by the monomial check in Π_gr1cs: `m` where `m_J ≤ m` (Figure 3/4).
    ///
    /// We represent the monomial vectors over `m*d` entries, so the Π_mon sumcheck runs over
    /// `log(m*d)` rounds and its row-point is `(r̄, s̄) ∈ K^{log m}`.
    pub m: usize,
    /// Projection matrix J ∈ χ^{λ_pj×ℓ_h} (Figure 2 Step 1).
    ///
    /// In the paper, `J` is a verifier message. In the FS/ROM form used here, `J` is derived from
    /// the transcript. We store it so downstream checks (e.g. Eq. (31)) can be performed without
    /// assuming a concrete transcript config type.
    pub J: Vec<Vec<R::BaseRing>>,
    /// Monomial set-check proof over the g^(i) vectors.
    pub mon: Out<R>,
    /// Per-digit projected coefficient evaluations:
    ///
    /// For each i ∈ [k_g], v(i) := H(i)^T * ts(r) ∈ K^d, where r are the first log(m)
    /// challenges inside `mon.r` and ts(r) is the multilinear tensor vector.
    pub v_digits: Vec<Vec<R::BaseRing>>,
}

/// Public output of Π_rg (Figure 2), reconstructed from a verified `RPRangeProof`.
///
/// This mirrors the paper’s output instance:
/// - x*  = (c, r, v)
/// - xbat = (r'=(r||s), [(c(i), u(i))])
///
/// Notes:
/// - We currently expose `u(i)` as the ring element `u(i) ∈ E` (same ring type as `R`),
///   coming from the set-check output `mon.b[i]`.
/// - The commitment values `c(i)` are not produced by this module yet; they belong to the
///   *next* layer that proves/maintains openings (`VfyOpen`) for all commitments.
#[derive(Clone, Debug)]
pub struct PiRgVerifiedOutput<R: PolyRing> {
    /// r ∈ K^{log m} (the first log(m) challenges inside `r'`).
    pub r: Vec<R::BaseRing>,
    /// s ∈ K^{log d} (the last log(d) challenges inside `r'`).
    pub s: Vec<R::BaseRing>,
    /// r' = r||s (full challenge vector used for u(i)=<ts(r||s), g(i)>).
    pub r_prime: Vec<R::BaseRing>,
    /// u(i) ∈ E for i∈[k_g], as produced by the monomial set-check output.
    pub u: Vec<R>,
    /// v ∈ K^d (composed): v = Σ_i (d')^i * v_digits[i]
    pub v: Vec<R::BaseRing>,
}

/// Compose the per-digit projected values `v_digits` into a single `v`:
/// v = Σ_i (d')^i * v_digits[i]
pub fn compose_v_digits<R: CoeffRing>(v_digits: &[Vec<R::BaseRing>], d_prime: u128) -> Vec<R::BaseRing>
where
    R::BaseRing: Zq,
{
    let d = R::dimension();
    let mut acc = vec![R::BaseRing::ZERO; d];
    let mut pow = R::BaseRing::ONE;
    for v_i in v_digits {
        for (a, &x) in acc.iter_mut().zip(v_i.iter()) {
            *a += pow * x;
        }
        pow *= R::BaseRing::from(d_prime);
    }
    acc
}

/// Derive a χ-style small projection matrix J ∈ {−1,0,+1}^{λ_pj×ℓ_h} from the transcript.
///
/// This is a *Fiat–Shamir*-style derivation: both prover and verifier can recompute J
/// as long as they execute transcript operations in the same order.
/// Prover for random projection range check
#[derive(Clone, Debug)]
pub struct RPRangeProver<R: PolyRing> {
    /// Witness vector
    pub f: Vec<R>,
    pub params: RPParams,
}

impl<R: CoeffRing> RPRangeProver<R>
where
    R::BaseRing: Zq + Decompose,
{
    pub fn new(f: Vec<R>, params: RPParams) -> Self {
        Self { f, params }
    }

    /// Produce Π_rg with the default (Figure 2) choice `m = m_J`.
    ///
    /// Use `prove_with_m(...)` from Π_gr1cs / Π_fold where the paper requires `m_J ≤ m`.
    pub fn prove(&self, transcript: &mut impl Transcript<R>, cm_f: &[R]) -> RPRangeProof<R> {
        self.prove_with_m(transcript, cm_f, None)
    }

    /// Produce Π_rg, optionally padding the monomial-check row domain from `m_J` to `m` (Figure 3/4).
    ///
    /// When `m_target = Some(m)`, we require `m_J ≤ m`, `m` power-of-two, and `m` multiple of `m_J`.
    /// We lift `H` from `m_J` rows to `m` rows by **replication** along the extra `s̄` dimension.
    pub fn prove_with_m(
        &self,
        transcript: &mut impl Transcript<R>,
        cm_f: &[R],
        m_target: Option<usize>,
    ) -> RPRangeProof<R> {
        let n = self.f.len();
        assert!(n > 0);
        assert!(n % self.params.l_h == 0, "l_h must divide n");

        // Bind to the statement commitment (Ajtai): c := A*f.
        transcript.absorb_slice(cm_f);

        // --- Step 1 (paper): verifier sends projection matrix J ← χ^{λ_pj×ℓ_h}.
        // Here we derive J from the transcript (FS-style) and then absorb it back in to
        // bind it as part of the transcript state for subsequent challenges.
        let J = derive_J::<R>(transcript, self.params.lambda_pj, self.params.l_h);
        for row in &J {
            for x in row {
                transcript.absorb_field_element(x);
            }
        }

        // --- Step 2 (paper): compute H := (I_{n/ℓ_h} ⊗ J) * cf(f).
        let d = R::dimension();
        let mut cf = vec![vec![R::BaseRing::ZERO; d]; n];
        for (row, r) in cf.iter_mut().zip(self.f.iter()) {
            for (j, c) in r.coeffs().iter().enumerate() {
                row[j] = *c;
            }
        }

        let blocks = n / self.params.l_h;
        let m_j = blocks * self.params.lambda_pj;
        let m = if let Some(m_target) = m_target {
            assert!(m_target.is_power_of_two(), "m must be power-of-two");
            assert!(m_target >= m_j, "require m_J <= m");
            assert_eq!(m_target % m_j, 0, "require m multiple of m_J");
            m_target
        } else {
            m_j
        };
        let mut H = vec![vec![R::BaseRing::ZERO; d]; m_j];
        for b in 0..blocks {
            for i in 0..self.params.lambda_pj {
                let out_row = b * self.params.lambda_pj + i;
                for t in 0..self.params.l_h {
                    let in_row = b * self.params.l_h + t;
                    let coef = J[i][t];
                    for col in 0..d {
                        H[out_row][col] += coef * cf[in_row][col];
                    }
                }
            }
        }

        // --- Digit decomposition (paper Eq. (33)): H = Σ (d')^i * H^(i+1)
        let mut H_digits: Vec<Vec<Vec<R::BaseRing>>> =
            vec![vec![vec![R::BaseRing::ZERO; d]; m_j]; self.params.k_g];
        for r in 0..m_j {
            // Decompose the whole coefficient row (length d) at once.
            // This matches how `rgchk` uses DecomposeToVec on coefficient rows.
            let row_digits = H[r].decompose_to_vec(self.params.d_prime, self.params.k_g);
            for c in 0..d {
                for i in 0..self.params.k_g {
                    H_digits[i][r][c] = row_digits[c][i];
                }
            }
        }

        // For Π_gr1cs alignment (Figure 3), we lift H_digits from `m_J` rows to `m` rows by **replication**
        // along the extra `s̄` dimension. This makes the resulting monomial vectors constant along `s̄`,
        // so evaluations at `(r̄, s̄)` depend only on `r̄` as required when `v(i)` is sent after `r̄` only.
        let expand_row = |row: usize| -> usize { row % m_j };

        // --- Step 3: g^(i) := Exp(flt(H^(i))) is a monomial vector.
        let mut g: Vec<Vec<R>> = Vec::with_capacity(self.params.k_g);
        for i in 0..self.params.k_g {
            let mut gi = Vec::with_capacity(m * d);
            // IMPORTANT: choose a flattening order compatible with the underlying MLE variable order.
            //
            // `DenseMultilinearExtension::evaluate` in this codebase treats the first variable as the
            // least-significant bit of the evaluation index. To ensure the "row point" r is sampled
            // before the "column point" s (Figure 2 ordering), we flatten in **column-major**
            // order: idx = col * m + row.
            //
            // This makes the low bits correspond to the row index (size m), so the first log(m)
            // challenges are the row point r and the last log(d) challenges are the column point s.
            for c in 0..d {
                for r in 0..m {
                    gi.push(exp::<R>(H_digits[i][expand_row(r)][c]).expect("Exp failed"));
                }
            }
            g.push(gi);
        }

        // --- Step 4: Monomial set-check on the g^(i) vectors.
        // NOTE: This is only the monomiality part; the Symphony paper additionally checks
        // linear consistency tying g^(i) back to the committed witness via u(i)*t(X) vs <ts(s), v(i)>.
        let g_len = m * d;
        let g_nvars = log2(g_len.next_power_of_two()) as usize;
        let in_rel = In {
            nvars: g_nvars,
            sets: g.iter().cloned().map(MonomialSet::Vector).collect(),
        };
        // Figure 2: we must send/commit v^{(i)} after r is sampled but before s is sampled.
        let log_m = log2(m.next_power_of_two()) as usize;
        let log_d = log2(d.next_power_of_two()) as usize;
        assert_eq!(g_nvars, log_m + log_d);

        let mut v_digits: Option<Vec<Vec<R::BaseRing>>> = None;
        let mon = in_rel.set_check_with_hook(
            &[],
            transcript,
            // Hook after |r̄| = log(m_J) rounds (Figure 3 shared challenge (r̄, s̄, s)).
            log2(m_j.next_power_of_two()) as usize,
            |t, sampled_r| {
                // sampled_r length == log(m_J)
                let ts_r_full = ts_weights(sampled_r);
                let ts_r = &ts_r_full[..m_j];
                let mut vd = Vec::with_capacity(self.params.k_g);
                for i in 0..self.params.k_g {
                    let mut v_i = vec![R::BaseRing::ZERO; d];
                    for row in 0..m_j {
                        let w = ts_r[row];
                        for col in 0..d {
                            v_i[col] += H_digits[i][row][col] * w;
                        }
                    }
                    vd.push(v_i);
                }
                // Absorb v_digits into transcript so that the remaining `s` challenges depend on it.
                for v_i in &vd {
                    for x in v_i {
                        t.absorb_field_element(x);
                    }
                }
                v_digits = Some(vd);
            },
        );
        let v_digits = v_digits.expect("hook did not run; log_m likely incorrect");

        let proof = RPRangeProof {
            params: self.params.clone(),
            m_j,
            m,
            J,
            mon,
            v_digits,
        };

        proof
    }
}

/// Verifier-side helper for the current prototype.
///
/// Today this only verifies the monomial set-check proof (`proof.mon`). In the full Symphony
/// approximate range proof, the verifier additionally checks a linear relation tying the
/// projected digits back to the committed witness.
pub fn verify_monomial_only<R: OverField>(
    proof: &RPRangeProof<R>,
    cm_f: &[R],
    transcript: &mut impl Transcript<R>,
) -> bool {
    transcript.absorb_slice(cm_f);
    // Re-derive and bind J into the transcript.
    let J = derive_J::<R>(transcript, proof.params.lambda_pj, proof.params.l_h);
    if J != proof.J {
        return false;
    }
    for row in &J {
        for x in row {
            transcript.absorb_field_element(x);
        }
    }
    // Mirror Π_rg ordering: bind v_digits after r̄ is fixed but before sampling s̄ and s.
    let log_mj = log2(proof.m_j.next_power_of_two()) as usize;
    proof
        .mon
        .verify_with_hook(transcript, log_mj, |t, _r_prefix| {
            for v_i in &proof.v_digits {
                for x in v_i {
                    t.absorb_field_element(x);
                }
            }
        })
        .is_ok()
}

#[derive(Debug, thiserror::Error)]
pub enum RPConsistencyError {
    #[error("Monomial set-check failed")]
    Monomial,
    #[error("Digit evaluation length mismatch: expected {expected}, got {got}")]
    DigitEvalLen { expected: usize, got: usize },
    #[error("Digit v length mismatch: expected d={expected}, got {got}")]
    DigitVLen { expected: usize, got: usize },
    #[error("Step-5 check failed at digit {idx}: ct(psi*u)={lhs} != <ts(s), v(i)>={rhs}")]
    Step5Mismatch { idx: usize, lhs: String, rhs: String },
    #[error("Output relation (Eq. (31)) failed: H^T*ts(r) != v")]
    AuxJLinMismatch,
    #[error("Projection matrix J mismatch (proof is not bound to transcript)")]
    JMismatch,
}

/// Verify monomiality and the **Symphony Figure 2 Step-5 check**:
/// for each digit i, compare
///   lhs := ct( psi * u(i) )  where u(i)=<ts(r||s), g(i)> comes from the set-check output,
///   rhs := <ts(s), v(i)>    where v(i)=H(i)^T ts(r) is provided by the prover.
pub fn verify_monomial_plus_projection_consistency<R: CoeffRing>(
    proof: &RPRangeProof<R>,
    cm_f: &[R],
    transcript: &mut impl Transcript<R>,
) -> Result<(), RPConsistencyError>
where
    R::BaseRing: Zq,
{
    transcript.absorb_slice(cm_f);
    // Re-derive and bind J into the transcript.
    let J = derive_J::<R>(transcript, proof.params.lambda_pj, proof.params.l_h);
    if J != proof.J {
        return Err(RPConsistencyError::JMismatch);
    }
    for row in &J {
        for x in row {
            transcript.absorb_field_element(x);
        }
    }

    let d = R::dimension();
    let log_m = log2(proof.m.next_power_of_two()) as usize;
    let log_mj = log2(proof.m_j.next_power_of_two()) as usize;
    proof
        .mon
        .verify_with_hook(transcript, log_mj, |t, _r_prefix| {
            for v_i in &proof.v_digits {
                for x in v_i {
                    t.absorb_field_element(x);
                }
            }
        })
        .map_err(|_| RPConsistencyError::Monomial)?;

    let kg = proof.params.k_g;
    if proof.mon.b.len() != kg {
        return Err(RPConsistencyError::DigitEvalLen {
            expected: kg,
            got: proof.mon.b.len(),
        });
    }
    if proof.v_digits.len() != kg {
        return Err(RPConsistencyError::DigitEvalLen {
            expected: kg,
            got: proof.v_digits.len(),
        });
    }
    for (i, v_i) in proof.v_digits.iter().enumerate() {
        if v_i.len() != d {
            return Err(RPConsistencyError::DigitVLen {
                expected: d,
                got: v_i.len(),
            });
        }
        // also sanity: i used for error reporting only
        let _ = i;
    }

    // We standardize the convention: mon.r = r||s with |s|=log d.
    let s_chals = proof.mon.r[log_m..].to_vec();
    let ts_s_full = ts_weights(&s_chals);
    let ts_s = &ts_s_full[..d];

    for i in 0..kg {
        let u_i = proof.mon.b[i];
        let lhs = (psi::<R>() * u_i).ct();
        let rhs = proof.v_digits[i]
            .iter()
            .zip(ts_s.iter())
            .fold(R::BaseRing::ZERO, |acc, (&vij, &t)| acc + vij * t);
        if lhs != rhs {
            return Err(RPConsistencyError::Step5Mismatch {
                idx: i,
                lhs: format!("{:?}", lhs),
                rhs: format!("{:?}", rhs),
            });
        }
    }

    Ok(())
}

/// Verify Π_rg’s internal checks (Π_mon + Figure 2 Step-5) and return the reconstructed
/// public output (r, s, r', u, v).
///
/// This is a *reduction interface*: it does **not** (and cannot, on its own) enforce the
/// downstream output relations `R_auxJ_lin` or the `VfyOpen(...)` parts from the paper.
pub fn verify_pi_rg_and_output<R: CoeffRing>(
        proof: &RPRangeProof<R>,
    cm_f: &[R],
        transcript: &mut impl Transcript<R>,
) -> Result<PiRgVerifiedOutput<R>, RPConsistencyError>
where
    R::BaseRing: Zq,
{
    verify_monomial_plus_projection_consistency(proof, cm_f, transcript)?;

    let log_mj = log2(proof.m_j.next_power_of_two()) as usize;
    let log_m = log2(proof.m.next_power_of_two()) as usize;

    let r_prime = proof.mon.r.clone();
    // Output instance x* uses r̄ (length log m_J) and v ∈ K^d. The batchlin instance uses full r'=(r̄,s̄,s).
    let r = r_prime[..log_mj].to_vec();
    let s = r_prime[log_m..].to_vec();

    let u = proof.mon.b.clone();
    let v = compose_v_digits::<R>(&proof.v_digits, proof.params.d_prime);

    Ok(PiRgVerifiedOutput {
        r,
        s,
        r_prime,
        u,
        v,
    })
}

/// **Paper-faithful output relation check (Eq. (31))**, given the explicit witness `f`.
///
/// This enforces the missing linkage that Π_rg reduces to:
/// - `VfyOpen(ppcm, c, f) = 1` is *not* checked here (we only bind `c` into the transcript).
/// - We *do* check the linear relation: \( \langle (M_J f), ts(r) \rangle = v \),
///   where `J` is derived from the transcript (FS / public-coin replay) and `r` comes from Π_mon.
///
/// This is intentionally **not succinct** (it is O(n·d·λ_pj/ℓ_h) time); it is a correctness/soundness
/// bridge while we implement the full `Π_gr1cs` / `Π_fold` pipeline.
pub fn verify_pi_rg_output_relation_with_witness<R: CoeffRing>(
    proof: &RPRangeProof<R>,
    cm_f: &[R],
    f: &[R],
    transcript: &mut impl Transcript<R>,
) -> Result<(), RPConsistencyError>
where
    R::BaseRing: Zq,
{
    // First, verify the internal Π_rg checks and recover (r, v).
    let out = verify_pi_rg_and_output(proof, cm_f, transcript)?;

    let v_check = compute_auxj_lin_v_from_witness::<R>(f, &proof.J, &out.r, &proof.params);

    if v_check != out.v {
        return Err(RPConsistencyError::AuxJLinMismatch);
        }

        Ok(())
    }

/// Compute the Eq. (31) output value \(v\) directly from an explicit witness `f`, projection matrix `J`,
/// point `r`, and parameters.
///
/// This is the deterministic function underlying the `R_auxJ_lin` relation:
/// \[
///   v := \langle (M_J f), ts(r)\rangle
/// \]
/// where \(M_J := I_{n/\ell_h} \otimes J\).
pub fn compute_auxj_lin_v_from_witness<R: CoeffRing>(
    f: &[R],
    J: &[Vec<R::BaseRing>],
    r: &[R::BaseRing],
    params: &RPParams,
) -> Vec<R::BaseRing>
where
    R::BaseRing: Zq,
{
    let n = f.len();
    assert!(n % params.l_h == 0, "l_h must divide n");
    assert_eq!(J.len(), params.lambda_pj);
    for row in J {
        assert_eq!(row.len(), params.l_h);
    }

    let d = R::dimension();
    let blocks = n / params.l_h;
    let m = blocks * params.lambda_pj;

    // cf(f) as an n×d matrix over the base ring.
    let mut cf = vec![vec![R::BaseRing::ZERO; d]; n];
    for (row, r) in cf.iter_mut().zip(f.iter()) {
        for (j, c) in r.coeffs().iter().enumerate() {
            row[j] = *c;
        }
    }

    // H has m rows (λ_pj per block), each a length-d base-ring vector.
    let mut H = vec![vec![R::BaseRing::ZERO; d]; m];
    for b in 0..blocks {
        for i in 0..params.lambda_pj {
            let out_row = b * params.lambda_pj + i;
            for t in 0..params.l_h {
                let in_row = b * params.l_h + t;
                let coef = J[i][t];
                for col in 0..d {
                    H[out_row][col] += coef * cf[in_row][col];
                }
            }
        }
    }

    // v := H^T * ts(r) ∈ K^d
    let ts_r_full = ts_weights(r);
    let ts_r = &ts_r_full[..m];
    let mut v = vec![R::BaseRing::ZERO; d];
    for row in 0..m {
        let w = ts_r[row];
        for col in 0..d {
            v[col] += H[row][col] * w;
        }
    }
    v
}

/// Bind the canonical Pi_rg transcript (statement + prover messages) into `transcript`,
/// without performing any arithmetic checks.
///
/// Intended for CP-style Fiat-Shamir binding checks: coins should be derived from the public
/// transcript, but hashing/derivation remains outside the locked relation.
pub fn bind_pi_rg_transcript<R: CoeffRing>(
    proof: &RPRangeProof<R>,
    cm_f: &[R],
    transcript: &mut impl Transcript<R>,
) where
    R::BaseRing: Zq,
{
    transcript.absorb_slice(cm_f);
    // Re-derive and bind J into the transcript.
    let J = derive_J::<R>(transcript, proof.params.lambda_pj, proof.params.l_h);
    assert_eq!(J, proof.J, "bind_pi_rg_transcript: J mismatch");
    for row in &J {
        for x in row {
            transcript.absorb_field_element(x);
        }
    }

    // Mirror Pi_rg ordering: bind v_digits after the `r` prefix is fixed but before sampling `s`.
    let d = R::dimension();
    let _log_d = log2(d.next_power_of_two()) as usize;
    let log_mj = log2(proof.m_j.next_power_of_two()) as usize;
    proof.mon.bind_with_hook(transcript, log_mj, |t, _r_prefix| {
        for v_i in &proof.v_digits {
            for x in v_i {
                t.absorb_field_element(x);
            }
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_std::One;
    use ark_std::UniformRand;
    use cyclotomic_rings::rings::FrogPoseidonConfig as PC;
    use stark_rings::cyclotomic_ring::models::frog_ring::RqPoly as R;

    use crate::transcript::PoseidonTranscript;
    use stark_rings_poly::mle::DenseMultilinearExtension;

    #[test]
    fn test_symphony_projection_scaffold_verifies_monomial_setcheck() {
        let n = 1 << 10;
        let f = vec![R::one(); n];
        let A = stark_rings_linalg::Matrix::<R>::rand(&mut ark_std::test_rng(), 2, n);
        let cm_f = A.try_mul_vec(&f).unwrap();
        let params = RPParams {
            l_h: 64,
            lambda_pj: 32, // small for test runtime
            k_g: 4,
            d_prime: (R::dimension() as u128) / 2, // match existing rgchk base b=d/2 for stability
        };

        let prover = RPRangeProver::new(f, params);
        let mut ts_p = PoseidonTranscript::empty::<PC>();
        let proof = prover.prove(&mut ts_p, &cm_f);

        let mut ts_v = PoseidonTranscript::empty::<PC>();
        assert!(verify_monomial_only(&proof, &cm_f, &mut ts_v));

        let mut ts_v2 = PoseidonTranscript::empty::<PC>();
        verify_monomial_plus_projection_consistency(
            &proof,
            &cm_f,
            &mut ts_v2,
        )
        .unwrap();
    }

    #[test]
    fn test_pi_rg_verified_output_shapes() {
        // Pin the "Figure 2 output instance" shape we expose after verification:
        // - r' = r||s where |r|=log(m), |s|=log(d)
        // - u(i) are the k_g ring elements from the monomial subprotocol output
        // - v is composed from v_digits in base d'
        let n = 1 << 10;
        let f = vec![R::one(); n];
        let A = stark_rings_linalg::Matrix::<R>::rand(&mut ark_std::test_rng(), 2, n);
        let cm_f = A.try_mul_vec(&f).unwrap();
        let params = RPParams {
            l_h: 64,
            lambda_pj: 32,
            k_g: 4,
            d_prime: (R::dimension() as u128) / 2,
        };

        let prover = RPRangeProver::new(f, params.clone());
        let mut ts_p = PoseidonTranscript::empty::<PC>();
        let proof = prover.prove(&mut ts_p, &cm_f);

        let mut ts_v = PoseidonTranscript::empty::<PC>();
        let out = verify_pi_rg_and_output(&proof, &cm_f, &mut ts_v).unwrap();

        let d = R::dimension();
        let blocks = n / params.l_h;
        let m = blocks * params.lambda_pj;
        let log_m = log2(m.next_power_of_two()) as usize;
        let log_d = log2(d.next_power_of_two()) as usize;

        assert_eq!(out.r.len(), log_m);
        assert_eq!(out.s.len(), log_d);
        assert_eq!(out.r_prime.len(), log_m + log_d);
        assert_eq!(out.u.len(), params.k_g);
        assert_eq!(out.v.len(), d);

        // v is exactly the composition of the provided digits.
        assert_eq!(out.v, compose_v_digits::<R>(&proof.v_digits, params.d_prime));
        // u comes directly from the monomial subprotocol output.
        assert_eq!(out.u, proof.mon.b);
    }

    #[test]
    fn test_pi_rg_output_relation_with_witness_holds() {
        let n = 1 << 10;
        let f = vec![R::one(); n];
        let A = stark_rings_linalg::Matrix::<R>::rand(&mut ark_std::test_rng(), 2, n);
        let cm_f = A.try_mul_vec(&f).unwrap();
        let params = RPParams {
            l_h: 64,
            lambda_pj: 32,
            k_g: 4,
            d_prime: (R::dimension() as u128) / 2,
        };

        let prover = RPRangeProver::new(f.clone(), params);
        let mut ts_p = PoseidonTranscript::empty::<PC>();
        let proof = prover.prove(&mut ts_p, &cm_f);

        let mut ts_v = PoseidonTranscript::empty::<PC>();
        verify_pi_rg_output_relation_with_witness(&proof, &cm_f, &f, &mut ts_v).unwrap();
    }

    #[test]
    fn test_pi_rg_output_relation_with_witness_rejects_wrong_witness() {
        let n = 1 << 10;
        let f = vec![R::one(); n];
        let A = stark_rings_linalg::Matrix::<R>::rand(&mut ark_std::test_rng(), 2, n);
        let cm_f = A.try_mul_vec(&f).unwrap();
        let params = RPParams {
            l_h: 64,
            lambda_pj: 32,
            k_g: 4,
            d_prime: (R::dimension() as u128) / 2,
        };

        let prover = RPRangeProver::new(f.clone(), params);
        let mut ts_p = PoseidonTranscript::empty::<PC>();
        let proof = prover.prove(&mut ts_p, &cm_f);

        // Provide a different witness while keeping the same commitment bound into the transcript.
        let mut f_bad = f.clone();
        f_bad[0] += R::one();

        let mut ts_v = PoseidonTranscript::empty::<PC>();
        assert!(verify_pi_rg_output_relation_with_witness(&proof, &cm_f, &f_bad, &mut ts_v).is_err());
    }

    #[test]
    fn test_ts_weights_matches_mle_evaluate() {
        // Pin the canonical tensor ordering convention used by ts_weights().
        let nvars = 6;
        let n = 1 << nvars;
        let mut rng = ark_std::test_rng();
        let evals = (0..n)
            .map(|_| <R as PolyRing>::BaseRing::rand(&mut rng))
            .collect::<Vec<_>>();
        let point = (0..nvars)
            .map(|_| <R as PolyRing>::BaseRing::rand(&mut rng))
            .collect::<Vec<_>>();

        let mle = DenseMultilinearExtension::from_evaluations_vec(nvars, evals.clone());
        let v = mle.evaluate(&point).unwrap();
        let ts = ts_weights(&point);
        let dot = evals
            .iter()
            .zip(ts.iter())
            .fold(<R as PolyRing>::BaseRing::ZERO, |acc, (&f, &w)| acc + f * w);
        assert_eq!(v, dot);
    }

    #[test]
    fn test_symphony_projection_scaffold_wrong_hook_round_fails() {
        // Sanity: if we bind v_digits at the wrong boundary (off-by-one), the transcript differs
        // and verification should fail.
        let n = 1 << 10;
        let f = vec![R::one(); n];
        let A = stark_rings_linalg::Matrix::<R>::rand(&mut ark_std::test_rng(), 2, n);
        let cm_f = A.try_mul_vec(&f).unwrap();
        let params = RPParams {
            l_h: 64,
            lambda_pj: 32,
            k_g: 4,
            d_prime: (R::dimension() as u128) / 2,
        };

        // Prove with the normal (correct) ordering.
        let prover = RPRangeProver::new(f, params);
        let mut ts_p = PoseidonTranscript::empty::<PC>();
        let proof = prover.prove(&mut ts_p, &cm_f);

        // Verify with the normal verifier should succeed.
        let mut ts_v_ok = PoseidonTranscript::empty::<PC>();
        verify_monomial_plus_projection_consistency(&proof, &cm_f, &mut ts_v_ok).unwrap();

        // Now emulate a verifier that binds v_digits at the wrong round by using verify_with_hook
        // with an incorrect hook_round.
        let mut ts_v_bad = PoseidonTranscript::empty::<PC>();
        ts_v_bad.absorb_slice(&cm_f);
        let J = super::derive_J::<R>(&mut ts_v_bad, proof.params.lambda_pj, proof.params.l_h);
        for row in &J {
            for x in row {
                ts_v_bad.absorb_field_element(x);
            }
        }
        // wrong hook_round: shift by 1 (if possible)
        let d = R::dimension();
        let log_d = log2(d.next_power_of_two()) as usize;
        let log_m = proof.mon.nvars - log_d;
        let wrong = if log_m > 0 { log_m - 1 } else { 0 };
        let ok = proof
            .mon
            .verify_with_hook(&mut ts_v_bad, wrong, |t, _| {
                for v_i in &proof.v_digits {
                    for x in v_i {
                        t.absorb_field_element(x);
                    }
                }
            })
            .is_ok();
        assert!(!ok, "verification unexpectedly succeeded with wrong hook_round");
    }

    #[test]
    fn test_psi_ct_matches_manual_negacyclic_constant_term() {
        // This pins the "ut_1" meaning from the paper's notation into *our* concrete ring:
        // for negacyclic rings R_q = Z_q[X]/(X^d + 1), the constant term of (a*b) is:
        //   ct(a*b) = a0*b0 - Σ_{i=1..d-1} a_i * b_{d-i}.
        //
        // We compare this manual formula against `(psi::<R>() * u).ct()` for random u.
        let mut rng = ark_std::test_rng();
        let d = R::dimension();

        let psi_poly = psi::<R>();
        let psi_coeffs = psi_poly.coeffs();

        for _ in 0..32 {
            let u_coeffs = (0..d)
                .map(|_| <R as PolyRing>::BaseRing::rand(&mut rng))
                .collect::<Vec<_>>();
            let u: R = u_coeffs.clone().into();

            let lhs = (psi_poly * u).ct();

            let mut rhs = psi_coeffs[0] * u_coeffs[0];
            for i in 1..d {
                rhs -= psi_coeffs[i] * u_coeffs[d - i];
            }

            assert_eq!(lhs, rhs);
        }
    }
}
