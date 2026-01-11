use ark_std::log2;
use latticefold::{
    transcript::Transcript,
    utils::sumcheck::{
        utils::eq_eval,
        MLSumcheck, Proof, SumCheckError,
    },
};
use stark_rings::{OverField, PolyRing, Ring};
use stark_rings_linalg::{ops::Transpose, SparseMatrix};
use thiserror::Error;
use std::sync::Arc;
use std::time::Instant;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

// (legacy) build_eq_x_r is no longer used in the streaming prover path

// cM: double commitment, commitment to M
// M: witness matrix of monomials

#[derive(Clone, Debug)]
pub enum MonomialSet<R> {
    Matrix(SparseMatrix<R>),
    Vector(Vec<R>),
}

#[derive(Clone, Debug)]
pub struct In<R> {
    pub nvars: usize,
    pub sets: Vec<MonomialSet<R>>, // Ms and ms: n x m, or n
}

#[derive(Clone, Debug)]
pub struct Out<R: PolyRing> {
    pub nvars: usize,
    pub r: Vec<R::BaseRing>, // log n
    pub sumcheck_proof: Proof<R>,
    pub e: Vec<Vec<Vec<R>>>, // m, matrices outputs
    pub b: Vec<R>,           // vectors outputs
}

#[derive(Debug, Error)]
pub enum SetCheckError<R: Ring> {
    #[error("Sumcheck failed: {0}")]
    Sumcheck(#[from] SumCheckError<R>),
    #[error("Recomputed claim `v` mismatch: expected = {0}, received = {1}")]
    ExpectedEvaluation(R, R),
}

fn ev<R: PolyRing>(r: &R, x: R::BaseRing) -> R::BaseRing {
    r.coeffs()
        .iter()
        .fold(
            (R::BaseRing::ZERO, R::BaseRing::ONE),
            |(mut acc, exp), c| {
                acc += *c * exp;
                (acc, exp * x)
            },
        )
        .0
}

impl<R: OverField> In<R> {
    /// Monomial set check
    ///
    /// Proves sets rings are all unit monomials.
    /// Currently requires k >= 1 monomial matrices sets. TODO support other scenarios.
    /// If k > 1, sumcheck batching is employed.
    pub fn set_check(&self, M: &[SparseMatrix<R>], transcript: &mut impl Transcript<R>) -> Out<R> {
        let profile = std::env::var("LF_PLUS_PROFILE").ok().as_deref() == Some("1");
        let t_total = Instant::now();

        let Ms: Vec<&SparseMatrix<R>> = self
            .sets
            .iter()
            .filter_map(|set| match set {
                MonomialSet::Matrix(m) => Some(m),
                _ => None,
            })
            .collect();
        let ms: Vec<&Vec<R>> = self
            .sets
            .iter()
            .filter_map(|set| match set {
                MonomialSet::Vector(v) => Some(v),
                _ => None,
            })
            .collect();

        let ncols = Ms[0].ncols;
        let MTs = Ms.iter().map(|M| M.transpose()).collect::<Vec<_>>();
        let tnvars = log2(Ms[0].nrows.next_power_of_two()) as usize;

        // Streaming MLEs (avoid materializing DenseMultilinearExtension tables).
        use crate::streaming_sumcheck::{StreamingMleEnum, StreamingSumcheck};
        let mut mles: Vec<StreamingMleEnum<R>> =
            Vec::with_capacity((Ms.len() + ms.len()) * (ncols * 2 + 1));
        let mut alphas = Vec::with_capacity(Ms.len());

        // matrix sets
        for (mi, M) in Ms.iter().enumerate() {
            let t_mat = Instant::now();
            // Step 1
            let c0 = transcript.get_challenges(self.nvars);
            let one_minus_c0 = c0.iter().copied().map(|x| R::BaseRing::ONE - x).collect();
            let beta = transcript.get_challenge();

            // Step 2
            let MT = M.transpose();

            // Fast evaluation uses precomputed beta powers (degree = ring dimension).
            let beta_pows = beta_pows::<R>(beta);

            // Build per-column base-scalar tables in parallel (each is length n = M.nrows).
            #[cfg(feature = "parallel")]
            let col_tables: Vec<Arc<Vec<R::BaseRing>>> = (0..ncols)
                .into_par_iter()
                .map(|col| {
                    let row = &MT.coeffs[col];
                    let mut v = vec![R::BaseRing::ZERO; M.nrows];
                    for (r_ij, idx) in row.iter() {
                        v[*idx] = ev_fast::<R>(r_ij, &beta_pows);
                    }
                    Arc::new(v)
                })
                .collect();
            #[cfg(not(feature = "parallel"))]
            let col_tables: Vec<Arc<Vec<R::BaseRing>>> = (0..ncols)
                .map(|col| {
                    let row = &MT.coeffs[col];
                    let mut v = vec![R::BaseRing::ZERO; M.nrows];
                    for (r_ij, idx) in row.iter() {
                        v[*idx] = ev_fast::<R>(r_ij, &beta_pows);
                    }
                    Arc::new(v)
                })
                .collect();

            for col in 0..ncols {
                let tab = col_tables[col].clone();
                mles.push(StreamingMleEnum::BaseScalarArc { evals: tab.clone(), num_vars: tnvars, square: false });
                mles.push(StreamingMleEnum::BaseScalarArc { evals: tab, num_vars: tnvars, square: true });
            }

            // eq(x,c) as base-ring structured MLE (constant-coeff)
            mles.push(StreamingMleEnum::EqBase {
                scale: R::BaseRing::ONE,
                r: c0,
                one_minus_r: one_minus_c0,
            });

            let alpha = transcript.get_challenge();
            alphas.push(alpha);

            if profile {
                println!(
                    "[LF+ setchk] matrix_set[{mi}] build_tables: {:?} (nrows={}, ncols={})",
                    t_mat.elapsed(),
                    M.nrows,
                    ncols
                );
            }
        }

        // vector sets
        for (vi, m) in ms.iter().enumerate() {
            let t_vec = Instant::now();
            // Step 1
            let c0 = transcript.get_challenges(self.nvars);
            let one_minus_c0 = c0.iter().copied().map(|x| R::BaseRing::ONE - x).collect();
            let beta = transcript.get_challenge();

            let beta_pows = beta_pows::<R>(beta);
            let mut v0 = vec![R::BaseRing::ZERO; m.len()];
            for (i, r_i) in m.iter().enumerate() {
                v0[i] = ev_fast::<R>(r_i, &beta_pows);
            }
            let tab = Arc::new(v0);
            mles.push(StreamingMleEnum::BaseScalarArc { evals: tab.clone(), num_vars: tnvars, square: false });
            mles.push(StreamingMleEnum::BaseScalarArc { evals: tab, num_vars: tnvars, square: true });
            mles.push(StreamingMleEnum::EqBase {
                scale: R::BaseRing::ONE,
                r: c0,
                one_minus_r: one_minus_c0,
            });

            let alpha = transcript.get_challenge();
            alphas.push(alpha);

            if profile {
                println!(
                    "[LF+ setchk] vector_set[{vi}] build_table: {:?} (len={})",
                    t_vec.elapsed(),
                    m.len()
                );
            }
        }

        // random linear combinator, for batching
        let rc: Option<R::BaseRing> = (Ms.len() > 1).then(|| transcript.get_challenge());

        let comb_fn = |vals: &[R]| -> R {
            let mut lc = R::zero();
            for (i, alpha) in alphas.iter().enumerate().take(Ms.len()) {
                // 2 * ncols for (m_j, m_prime_j), +1 for eq
                let s = i * (2 * ncols + 1);
                let mut res = R::zero();
                for j in 0..ncols {
                    res += (vals[s + j * 2] * vals[s + j * 2] - vals[s + j * 2 + 1])
                        * alpha.pow([j as u64])
                }
                res *= vals[s + 2 * ncols]; // eq
                lc += if let Some(rc) = &rc {
                    res * rc.pow([i as u64])
                } else {
                    return res;
                };
            }
            for i in 0..ms.len() {
                let s_base = Ms.len() * (2 * ncols + 1);
                let s = s_base + i * 3;
                let mut res = R::zero();
                let alpha_idx = Ms.len() + i;
                res += (vals[s] * vals[s] - vals[s + 1]) * alphas[alpha_idx];
                res *= vals[s + 2]; // eq
                lc += if let Some(rc) = &rc {
                    res * rc.pow([alpha_idx as u64])
                } else {
                    return res;
                };
            }
            lc
        };

        let t_sc = Instant::now();
        let (sumcheck_proof, r, _final_vals) =
            StreamingSumcheck::prove_as_subprotocol(transcript, mles, self.nvars, 3, comb_fn);
        if profile {
            println!(
                "[LF+ setchk] sumcheck: {:?} (nvars={}, degree=3, ncols={}, Ms={}, ms={})",
                t_sc.elapsed(),
                self.nvars,
                ncols,
                Ms.len(),
                ms.len()
            );
        }

        // Step 3
        let t_step3 = Instant::now();
        let eq_r = build_eq_table_base::<R>(&r);
        let eq_r_ring: Vec<R> = eq_r.iter().copied().map(R::from).collect();

        // Precompute y_i = M_i^T * eq_r (so eval(M_i * row)(r) = <y_i, row>).
        let y_mats: Vec<Vec<R>> = M
            .iter()
            .map(|mi| {
                let mut y = vec![R::ZERO; mi.ncols];
                for (row_idx, row) in mi.coeffs.iter().enumerate() {
                    let w = eq_r_ring[row_idx];
                    for (coeff, col_idx) in row {
                        y[*col_idx] += *coeff * w;
                    }
                }
                y
            })
            .collect();

        let e: Vec<Vec<Vec<R>>> = {
            let mut e = Vec::with_capacity(1 + M.len());

            // e0: eval of each MT row at point r.
            #[cfg(feature = "parallel")]
            let e0 = MTs
                .par_iter()
                .map(|MT| {
                    MT.coeffs
                        .par_iter()
                        .map(|row| {
                            let mut acc = R::ZERO;
                            for &(rij, idx) in row {
                                acc += rij * eq_r_ring[idx];
                            }
                            acc
                        })
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<Vec<R>>>();
            #[cfg(not(feature = "parallel"))]
            let e0 = MTs
                .iter()
                .map(|MT| {
                    MT.coeffs
                        .iter()
                        .map(|row| {
                            let mut acc = R::ZERO;
                            for &(rij, idx) in row {
                                acc += rij * eq_r_ring[idx];
                            }
                            acc
                        })
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<Vec<R>>>();
            e.push(e0);

            // Mf
            for (mi, y) in y_mats.iter().enumerate() {
                #[cfg(feature = "parallel")]
                let ei = MTs
                    .par_iter()
                    .map(|MT| {
                        MT.coeffs
                            .par_iter()
                            .map(|row| {
                                let mut acc = R::ZERO;
                                for &(rij, idx) in row {
                                    acc += rij * y[idx];
                                }
                                acc
                            })
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<Vec<R>>>();
                #[cfg(not(feature = "parallel"))]
                let ei = MTs
                    .iter()
                    .map(|MT| {
                        MT.coeffs
                            .iter()
                            .map(|row| {
                                let mut acc = R::ZERO;
                                for &(rij, idx) in row {
                                    acc += rij * y[idx];
                                }
                                acc
                            })
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<Vec<R>>>();
                let _ = mi;
                e.push(ei);
            }
            e
        };

        #[cfg(feature = "parallel")]
        let b: Vec<R> = ms
            .par_iter()
            .map(|m| {
                let mut acc = R::ZERO;
                for (i, &mi) in m.iter().enumerate() {
                    acc += mi * eq_r_ring[i];
                }
                acc
            })
            .collect();
        #[cfg(not(feature = "parallel"))]
        let b: Vec<R> = ms
            .iter()
            .map(|m| {
                let mut acc = R::ZERO;
                for (i, &mi) in m.iter().enumerate() {
                    acc += mi * eq_r_ring[i];
                }
                acc
            })
            .collect();

        // Prover to Verifier messages
        absorb_evaluations(&e, &b, transcript);

        if profile {
            println!(
                "[LF+ setchk] step3(e,b)+absorb: {:?}  total: {:?}",
                t_step3.elapsed(),
                t_total.elapsed()
            );
        }

        Out {
            nvars: self.nvars,
            e,
            b,
            r,
            sumcheck_proof,
        }
    }
}

#[inline]
fn beta_pows<R: PolyRing>(beta: R::BaseRing) -> Vec<R::BaseRing>
where
    R::BaseRing: Ring,
{
    let d = R::dimension();
    let mut out = Vec::with_capacity(d);
    let mut acc = R::BaseRing::ONE;
    for _ in 0..d {
        out.push(acc);
        acc *= beta;
    }
    out
}

/// Fast `ev(r, beta)`:
/// - if `r` is monomial-like (<=1 nonzero coeff), do O(1) lookup via `beta_pows`
/// - otherwise fall back to full dot product against `beta_pows`
#[inline]
fn ev_fast<R: PolyRing>(r: &R, beta_pows: &[R::BaseRing]) -> R::BaseRing
where
    R::BaseRing: Ring,
{
    let coeffs = r.coeffs();
    debug_assert_eq!(coeffs.len(), beta_pows.len());

    let mut idx: Option<usize> = None;
    let mut c: R::BaseRing = R::BaseRing::ZERO;
    for (i, &ci) in coeffs.iter().enumerate() {
        if ci != R::BaseRing::ZERO {
            if idx.is_some() {
                // not monomial
                let mut acc = R::BaseRing::ZERO;
                for (cj, pj) in coeffs.iter().zip(beta_pows.iter()) {
                    if *cj != R::BaseRing::ZERO {
                        acc += *cj * *pj;
                    }
                }
                return acc;
            }
            idx = Some(i);
            c = ci;
        }
    }
    match idx {
        None => R::BaseRing::ZERO,
        Some(i) => c * beta_pows[i],
    }
}

/// Build eq(bits(idx), r) table (little-endian index order), matching latticefold build_eq_x_r_vec.
fn build_eq_table_base<R: PolyRing>(r: &[R::BaseRing]) -> Vec<R::BaseRing>
where
    R::BaseRing: Ring,
{
    let mut buf = vec![R::BaseRing::ONE];
    for &ri in r.iter().rev() {
        let mut res = vec![R::BaseRing::ZERO; buf.len() << 1];
        for (i, out) in res.iter_mut().enumerate() {
            let bi = buf[i >> 1];
            let tmp = ri * bi;
            *out = if (i & 1) == 0 { bi - tmp } else { tmp };
        }
        buf = res;
    }
    buf
}

impl<R: OverField> Out<R> {
    pub fn verify(&self, transcript: &mut impl Transcript<R>) -> Result<(), SetCheckError<R>> {
        let nclaims = self.e[0].len() + self.b.len();

        let cba: Vec<(Vec<R>, R::BaseRing, R::BaseRing)> = (0..nclaims)
            .map(|_| {
                let c: Vec<R> = transcript
                    .get_challenges(self.nvars)
                    .into_iter()
                    .map(|x| x.into())
                    .collect();
                let beta = transcript.get_challenge();
                let alpha = transcript.get_challenge();
                (c, beta, alpha)
            })
            .collect();

        let rc: Option<R::BaseRing> = (self.e[0].len() > 1).then(|| transcript.get_challenge());

        let subclaim = MLSumcheck::verify_as_subprotocol(
            transcript,
            self.nvars,
            3,
            R::zero(),
            &self.sumcheck_proof,
        )?;

        let r: Vec<R> = subclaim.point.into_iter().map(|x| x.into()).collect();

        let v = subclaim.expected_evaluation;

        // Prover to Verifier messages
        absorb_evaluations(&self.e, &self.b, transcript);

        use ark_std::One;
        let mut ver = R::zero();
        for (i, e) in self.e[0].iter().enumerate() {
            let c = &cba[i].0;
            let beta = &cba[i].1;
            let alpha = &cba[i].2;
            let eq = eq_eval(c, &r).unwrap();
            let e_sum = e
                .iter()
                .enumerate()
                .map(|(j, e_j)| {
                    let ev1 = R::from(ev(e_j, *beta));
                    let ev2 = R::from(ev(e_j, *beta * beta));
                    (ev1 * ev1 - ev2) * alpha.pow([j as u64])
                })
                .sum::<R>();
            ver += eq * e_sum * rc.as_ref().unwrap_or(&R::BaseRing::one()).pow([i as u64]);
        }
        for (i, b) in self.b.iter().enumerate() {
            let offset = self.e[0].len();
            let c = &cba[i + offset].0;
            let beta = &cba[i + offset].1;
            let alpha = &cba[i + offset].2;
            let eq = eq_eval(c, &r).unwrap();
            let b_claim = {
                let ev1 = R::from(ev(b, *beta));
                let ev2 = R::from(ev(b, *beta * *beta));
                ev1 * ev1 - ev2
            };
            ver += eq
                * *alpha
                * b_claim
                * rc.as_ref()
                    .unwrap_or(&R::BaseRing::one())
                    .pow([(i + offset) as u64]);
        }

        (ver == v)
            .then_some(())
            .ok_or(SetCheckError::ExpectedEvaluation(ver, v))?;

        Ok(())
    }
}

fn absorb_evaluations<R: OverField>(
    e: &[Vec<Vec<R>>],
    b: &[R],
    transcript: &mut impl Transcript<R>,
) {
    for ek in e {
        for ej in ek {
            transcript.absorb_slice(ej);
        }
    }
    transcript.absorb_slice(b);
}

#[cfg(test)]
mod tests {
    use ark_std::One;
    use cyclotomic_rings::rings::FrogPoseidonConfig as PC;
    use stark_rings::{cyclotomic_ring::models::frog_ring::RqPoly as R, unit_monomial};
    use stark_rings_linalg::SparseMatrix;

    use super::*;
    use crate::transcript::PoseidonTranscript;

    #[test]
    fn test_set_check() {
        let n = 4;
        let M = SparseMatrix::<R>::identity(n);

        let scin = In {
            sets: vec![MonomialSet::Matrix(M)],
            nvars: log2(n) as usize,
        };

        let mut ts = PoseidonTranscript::empty::<PC>();
        let out = scin.set_check(&[], &mut ts);

        let mut ts = PoseidonTranscript::empty::<PC>();
        out.verify(&mut ts).unwrap();
    }

    #[test]
    fn test_set_check_bad() {
        let n = 4;
        let mut M = SparseMatrix::<R>::identity(n);
        // 1 + X, not a monomial
        let mut onepx = R::one();
        onepx.coeffs_mut()[1] = 1u128.into();
        M.coeffs[0][0].0 = onepx;

        let scin = In {
            sets: vec![MonomialSet::Matrix(M)],
            nvars: log2(n) as usize,
        };

        let mut ts = PoseidonTranscript::empty::<PC>();
        let out = scin.set_check(&[], &mut ts);

        let mut ts = PoseidonTranscript::empty::<PC>();
        assert!(out.verify(&mut ts).is_err());
    }

    #[test]
    fn test_set_check_batched() {
        let n = 4;
        let M0 = SparseMatrix::<R>::identity(n);
        let M1 = SparseMatrix::<R>::identity(n);

        let scin = In {
            sets: vec![MonomialSet::Matrix(M0), MonomialSet::Matrix(M1)],
            nvars: log2(n) as usize,
        };

        let mut ts = PoseidonTranscript::empty::<PC>();
        let out = scin.set_check(&[], &mut ts);

        let mut ts = PoseidonTranscript::empty::<PC>();
        out.verify(&mut ts).unwrap();
    }

    #[test]
    fn test_set_check_batched_bad() {
        let n = 4;
        let M0 = SparseMatrix::<R>::identity(n);
        let mut M1 = SparseMatrix::<R>::identity(n);
        // 1 + X, not a monomial
        let mut onepx = R::one();
        onepx.coeffs_mut()[1] = 1u128.into();
        M1.coeffs[0][0].0 = onepx;

        let scin = In {
            sets: vec![MonomialSet::Matrix(M0), MonomialSet::Matrix(M1)],
            nvars: log2(n) as usize,
        };

        let mut ts = PoseidonTranscript::empty::<PC>();
        let out = scin.set_check(&[], &mut ts);

        let mut ts = PoseidonTranscript::empty::<PC>();
        assert!(out.verify(&mut ts).is_err());
    }

    #[test]
    fn test_set_check_mix() {
        let n = 4;
        let M0 = SparseMatrix::<R>::identity(n);
        let M1 = SparseMatrix::<R>::identity(n);
        let m0 = vec![R::one(); n];
        let m1 = vec![unit_monomial(2); n];

        let scin = In {
            sets: vec![
                MonomialSet::Matrix(M0),
                MonomialSet::Matrix(M1),
                MonomialSet::Vector(m0),
                MonomialSet::Vector(m1),
            ],
            nvars: log2(n) as usize,
        };

        let mut ts = PoseidonTranscript::empty::<PC>();
        let out = scin.set_check(&[], &mut ts);

        let mut ts = PoseidonTranscript::empty::<PC>();
        out.verify(&mut ts).unwrap();
    }

    #[test]
    fn test_set_check_mix_bad() {
        let n = 4;
        let M0 = SparseMatrix::<R>::identity(n);
        let M1 = SparseMatrix::<R>::identity(n);
        let mut m0 = vec![R::one(); n];
        let mut onepx = R::one();
        onepx.coeffs_mut()[1] = 1u128.into();
        m0[0] = onepx;

        let scin = In {
            sets: vec![
                MonomialSet::Matrix(M0),
                MonomialSet::Matrix(M1),
                MonomialSet::Vector(m0),
            ],
            nvars: log2(n) as usize,
        };

        let mut ts = PoseidonTranscript::empty::<PC>();
        let out = scin.set_check(&[], &mut ts);

        let mut ts = PoseidonTranscript::empty::<PC>();
        assert!(out.verify(&mut ts).is_err());
    }
}
