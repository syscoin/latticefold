use ark_std::iter::once;
use latticefold::transcript::Transcript;
use stark_rings::{
    balanced_decomposition::{Decompose, DecomposeToVec},
    exp, psi, CoeffRing, OverField, PolyRing, Ring, Zq,
};
use stark_rings_linalg::{Matrix, SparseMatrix};
use thiserror::Error;

use crate::{
    setchk::{In, MonomialSet, Out},
    utils::split,
};

#[cfg(feature = "parallel")]
use rayon::prelude::*;

// D_f: decomposed cf(f), Z n x dk
// M_f: EXP(D_f)

#[derive(Clone, Debug)]
pub struct DecompParameters {
    pub b: u128,
    pub k: usize,
    pub l: usize,
}

#[derive(Clone, Debug)]
pub struct FComs<R> {
    pub cm_f: Vec<R>,
    pub C_Mf: Vec<R>,
    pub cm_mtau: Vec<R>,
}

#[derive(Clone, Debug)]
pub struct Rg<R: PolyRing> {
    pub nvars: usize,
    pub instances: Vec<RgInstance<R>>, // L instances
    pub dparams: DecompParameters,
}

#[derive(Clone, Debug)]
pub struct RgInstance<R: PolyRing> {
    pub M_f: Vec<Matrix<R>>,   // n x d, k matrices, monomials
    pub tau: Vec<R::BaseRing>, // n
    pub m_tau: Vec<R>,         // n, monomials
    pub f: Vec<R>,             // n
    pub comM_f: Vec<Matrix<R>>,
    pub fcoms: FComs<R>,
}

#[derive(Clone, Debug)]
pub struct Dcom<R: PolyRing> {
    pub evals: Vec<DcomEvals<R>>, // L evals
    pub fcoms: Vec<FComs<R>>,     // L commitments
    pub out: Out<R>,              // set checks
    pub dparams: DecompParameters,
}

#[derive(Clone, Debug)]
pub struct DcomEvals<R: PolyRing> {
    pub v: Vec<R::BaseRing>, // eval over M_f
    pub a: Vec<R::BaseRing>, // eval over tau
    pub b: Vec<R>,           // eval over m_tau
    pub c: Vec<R>,           // eval over f
}

#[derive(Debug, Error)]
pub enum RangeCheckError<R: PolyRing> {
    #[error("Psi check failed: a = {0}, b = {1}")]
    PsiCheckAB(R::BaseRing, R),
    #[error("Psi check failed: v = {0}, u-comb = {1}")]
    PsiCheckVU(Vec<R::BaseRing>, Vec<R>),
}

impl<R: CoeffRing> Rg<R>
where
    R::BaseRing: Zq,
{
    /// Range checks
    ///
    /// Support for `L` [`RgInstance`]s mapped to the corresponding [`DcomEvals`].
    pub fn range_check(
        &self,
        M: &[SparseMatrix<R>],
        transcript: &mut impl Transcript<R>,
    ) -> Dcom<R> {
        let profile = std::env::var("LF_PLUS_PROFILE").ok().as_deref() == Some("1");
        let t_total = std::time::Instant::now();

        let mut sets = Vec::with_capacity(self.instances.len() * (self.instances[0].M_f.len() + 1));
        for inst in &self.instances {
            inst.M_f.iter().for_each(|m| {
                sets.push(MonomialSet::Matrix(SparseMatrix::<R>::from_dense(m)));
            });
        }
        for inst in &self.instances {
            sets.push(MonomialSet::Vector(inst.m_tau.clone()));
        }

        let in_rel = In {
            sets,
            nvars: self.nvars,
        };
        let out_rel = in_rel.set_check(M, transcript);

        // Precompute eq(bits(idx), r) table once (base ring).
        let t = std::time::Instant::now();
        let eq = build_eq_table_base::<R>(&out_rel.r);
        if profile {
            println!(
                "[LF+ Rg::range_check] set_check: {:?}, build_eq_table: {:?} (n={}, nvars={})",
                t_total.elapsed(),
                t.elapsed(),
                eq.len(),
                self.nvars
            );
        }

        let evals = self
            .instances
            .iter()
            .enumerate()
            .map(|(l, inst)| {
                let mut a = Vec::with_capacity(1 + M.len());
                let mut b = Vec::with_capacity(1 + M.len());
                // Let `c` be the evaluation of `f` over r
                let mut c = Vec::with_capacity(1 + M.len());

                // v: coefficient-wise evaluation of f at out_rel.r
                let v = eval_vec_coeffs_at_point::<R>(&inst.f, &eq);

                a.push(dot_base::<R>(&inst.tau, &eq));
                b.push(out_rel.b[l]);
                c.push(dot_ring::<R>(&inst.f, &eq));

                // Evaluate M * tau / m_tau / f at out_rel.r *without materializing length-n vectors*.
                //
                // For each matrix M:
                //   eval(M*w)(r) = Σ_row eq[row] * (Σ_{(coeff,col) in row} coeff * w[col])
                for m in M {
                    a.push(sparse_mat_vec_eval_ct::<R>(m, &inst.tau, &eq));
                    b.push(sparse_mat_vec_eval_ring::<R>(m, &inst.m_tau, &eq));
                    c.push(sparse_mat_vec_eval_ring::<R>(m, &inst.f, &eq));
                }
                DcomEvals { v, a, b, c }
            })
            .collect::<Vec<_>>();

        absorb_evaluations(&evals, transcript);

        if profile {
            println!("[LF+ Rg::range_check] evals+absorb: {:?}", t_total.elapsed());
        }

        Dcom {
            evals,
            fcoms: self
                .instances
                .iter()
                .map(|inst| inst.fcoms.clone())
                .collect(),
            out: out_rel,
            dparams: self.dparams.clone(),
        }
    }
}

impl<R: CoeffRing> Dcom<R>
where
    R::BaseRing: Zq,
{
    pub fn verify(&self, transcript: &mut impl Transcript<R>) -> Result<(), RangeCheckError<R>> {
        self.out.verify(transcript).unwrap(); //.map_err(|_| ())?;

        absorb_evaluations(&self.evals, transcript);

        for (l, eval) in self.evals.iter().enumerate() {
            // ct(psi b) =? a
            for (&a_i, b_i) in eval.a.iter().zip(eval.b.iter()) {
                ((psi::<R>() * b_i).ct() == a_i)
                    .then_some(())
                    .ok_or(RangeCheckError::PsiCheckAB(a_i, *b_i))?;
            }

            let d = R::dimension();
            let d_prime = d / 2;
            for (ni, _) in self.out.e.iter().enumerate() {
                let u_comb = self.out.e[ni]
                    .iter()
                    .skip(self.dparams.k * l)
                    .take(self.dparams.k)
                    .enumerate()
                    .fold(vec![R::zero(); d], |mut acc, (i, u_i)| {
                        let d_ppow = R::BaseRing::from(d_prime as u128).pow([i as u64]);
                        u_i.iter()
                            .zip(acc.iter_mut())
                            .for_each(|(u_ij, a_j)| *a_j += *u_ij * d_ppow);
                        acc
                    });

                // ct(psi (sum d^i u_i)) =? v
                let v_rec = u_comb
                    .iter()
                    .map(|&uc| (psi::<R>() * uc).ct())
                    .collect::<Vec<_>>();

                if ni == 0 {
                    (eval.v == v_rec)
                        .then_some(())
                        .ok_or(RangeCheckError::PsiCheckVU(v_rec, u_comb))?;
                } else {
                    (eval.c[ni].coeffs() == v_rec)
                        .then_some(())
                        .ok_or(RangeCheckError::PsiCheckVU(v_rec, u_comb))?;
                }
            }
        }

        Ok(())
    }
}

impl<R: PolyRing> RgInstance<R> {
    /// Construct monomial sets from `M_f` and `m_tau`
    pub fn sets(&self) -> Vec<MonomialSet<R>> {
        self.M_f
            .iter()
            .map(|m| MonomialSet::Matrix(SparseMatrix::<R>::from_dense(m)))
            .chain(once(MonomialSet::Vector(self.m_tau.clone())))
            .collect()
    }
}

impl<R: CoeffRing> RgInstance<R>
where
    R::BaseRing: Decompose + Zq,
    R: Decompose,
{
    pub fn from_f(f: Vec<R>, A: &Matrix<R>, decomp: &DecompParameters) -> Self {
        let n = f.len();

        let cfs: Matrix<_> = f
            .iter()
            .map(|r| r.coeffs().to_vec())
            .collect::<Vec<Vec<_>>>()
            .into();
        let dec = cfs
            .vals
            .iter()
            .map(|row| row.decompose_to_vec(decomp.b, decomp.k))
            .collect::<Vec<_>>();

        let mut D_f = vec![Matrix::zero(n, R::dimension()); decomp.k];

        // map dec: (Z n x d x k) to D_f: (Z n x d, k matrices)
        dec.iter().enumerate().for_each(|(n_i, drow)| {
            drow.iter().enumerate().for_each(|(d_i, coeffs)| {
                coeffs.iter().enumerate().for_each(|(k_i, coeff)| {
                    D_f[k_i].vals[n_i][d_i] = *coeff;
                });
            });
        });

        let M_f: Vec<Matrix<R>> = D_f
            .iter()
            .map(|m| {
                m.vals
                    .iter()
                    .map(|row| {
                        row.iter()
                            .map(|c| exp::<R>(*c).unwrap())
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>()
                    .into()
            })
            .collect::<Vec<_>>();

        let comM_f = M_f
            .iter()
            .map(|M| A.try_mul_mat(M).unwrap())
            .collect::<Vec<_>>();
        let com = Matrix::hconcat(&comM_f).unwrap();

        let tau = split(&com, n, (R::dimension() / 2) as u128, decomp.l);

        let m_tau = tau
            .iter()
            .map(|c| exp::<R>(*c).unwrap())
            .collect::<Vec<_>>();

        let cm_f = A.try_mul_vec(&f).unwrap();
        let C_Mf = A
            .try_mul_vec(&tau.iter().map(|z| R::from(*z)).collect::<Vec<R>>())
            .unwrap();
        let cm_mtau = A.try_mul_vec(&m_tau).unwrap();
        let fcoms = FComs {
            cm_f,
            C_Mf,
            cm_mtau,
        };

        Self {
            M_f,
            tau,
            m_tau,
            f,
            comM_f,
            fcoms,
        }
    }
}

fn absorb_evaluations<R: OverField>(evals: &[DcomEvals<R>], transcript: &mut impl Transcript<R>) {
    evals.iter().for_each(|eval| {
        transcript.absorb_slice(&eval.a.iter().map(|z| R::from(*z)).collect::<Vec<R>>());
        transcript.absorb_slice(&eval.c);
    });
}

/// Build the full eq table: eq(bits(idx), r) for all idx in {0,1}^nvars (LSB-first).
fn build_eq_table_base<R: PolyRing>(r: &[R::BaseRing]) -> Vec<R::BaseRing>
where
    R::BaseRing: Ring,
{
    let mut acc = vec![R::BaseRing::ONE];
    for &ri in r {
        let one_minus = R::BaseRing::ONE - ri;
        let mut next = Vec::with_capacity(acc.len() * 2);
        for &v in &acc {
            next.push(v * one_minus);
            next.push(v * ri);
        }
        acc = next;
    }
    acc
}

fn dot_base<R: PolyRing>(v: &[R::BaseRing], eq: &[R::BaseRing]) -> R::BaseRing
where
    R::BaseRing: Ring,
{
    debug_assert_eq!(v.len(), eq.len());
    #[cfg(feature = "parallel")]
    {
        v.par_iter()
            .zip(eq.par_iter())
            .map(|(&x, &w)| x * w)
            .reduce(|| R::BaseRing::ZERO, |a, b| a + b)
    }
    #[cfg(not(feature = "parallel"))]
    {
        v.iter().zip(eq.iter()).map(|(&x, &w)| x * w).sum()
    }
}

fn dot_ring<R>(v: &[R], eq: &[R::BaseRing]) -> R
where
    R: PolyRing + From<R::BaseRing>,
    R::BaseRing: Ring,
{
    debug_assert_eq!(v.len(), eq.len());
    #[cfg(feature = "parallel")]
    {
        v.par_iter()
            .zip(eq.par_iter())
            .map(|(&x, &w)| x * R::from(w))
            .reduce(|| R::ZERO, |a, b| a + b)
    }
    #[cfg(not(feature = "parallel"))]
    {
        v.iter().zip(eq.iter()).map(|(&x, &w)| x * R::from(w)).sum()
    }
}

fn eval_vec_coeffs_at_point<R: PolyRing>(v: &[R], eq: &[R::BaseRing]) -> Vec<R::BaseRing>
where
    R::BaseRing: Ring,
{
    debug_assert_eq!(v.len(), eq.len());
    let d = R::dimension();
    #[cfg(feature = "parallel")]
    {
        (0..d)
            .into_par_iter()
            .map(|j| {
                v.par_iter()
                    .zip(eq.par_iter())
                    .map(|(x, &w)| x.coeffs()[j] * w)
                    .reduce(|| R::BaseRing::ZERO, |a, b| a + b)
            })
            .collect()
    }
    #[cfg(not(feature = "parallel"))]
    {
        let mut out = vec![R::BaseRing::ZERO; d];
        for (x, &w) in v.iter().zip(eq.iter()) {
            for j in 0..d {
                out[j] += x.coeffs()[j] * w;
            }
        }
        out
    }
}

fn sparse_mat_vec_eval_ct<R: PolyRing>(
    m: &SparseMatrix<R>,
    witness0: &[R::BaseRing],
    eq: &[R::BaseRing],
) -> R::BaseRing
where
    R::BaseRing: Ring,
{
    debug_assert_eq!(m.nrows, eq.len());
    #[cfg(feature = "parallel")]
    {
        m.coeffs
            .par_iter()
            .zip(eq.par_iter())
            .map(|(row, &w_row)| {
                // row_dot is ring, but we only need constant term at the end.
                let mut sum0 = R::BaseRing::ZERO;
                for (coeff, col_idx) in row {
                    if *col_idx < witness0.len() {
                        sum0 += coeff.coeffs()[0] * witness0[*col_idx];
                    }
                }
                sum0 * w_row
            })
            .reduce(|| R::BaseRing::ZERO, |a, b| a + b)
    }
    #[cfg(not(feature = "parallel"))]
    {
        let mut acc = R::BaseRing::ZERO;
        for (row_idx, row) in m.coeffs.iter().enumerate() {
            let w_row = eq[row_idx];
            let mut sum0 = R::BaseRing::ZERO;
            for (coeff, col_idx) in row {
                if *col_idx < witness0.len() {
                    sum0 += coeff.coeffs()[0] * witness0[*col_idx];
                }
            }
            acc += sum0 * w_row;
        }
        acc
    }
}

fn sparse_mat_vec_eval_ring<R>(
    m: &SparseMatrix<R>,
    witness: &[R],
    eq: &[R::BaseRing],
) -> R
where
    R: PolyRing + From<R::BaseRing>,
    R::BaseRing: Ring,
{
    debug_assert_eq!(m.nrows, eq.len());
    #[cfg(feature = "parallel")]
    {
        m.coeffs
            .par_iter()
            .zip(eq.par_iter())
            .map(|(row, &w_row)| {
                let mut row_dot = R::ZERO;
                for (coeff, col_idx) in row {
                    if *col_idx < witness.len() {
                        row_dot += *coeff * witness[*col_idx];
                    }
                }
                row_dot * R::from(w_row)
            })
            .reduce(|| R::ZERO, |a, b| a + b)
    }
    #[cfg(not(feature = "parallel"))]
    {
        let mut acc = R::ZERO;
        for (row_idx, row) in m.coeffs.iter().enumerate() {
            let w_row = R::from(eq[row_idx]);
            let mut row_dot = R::ZERO;
            for (coeff, col_idx) in row {
                if *col_idx < witness.len() {
                    row_dot += *coeff * witness[*col_idx];
                }
            }
            acc += row_dot * w_row;
        }
        acc
    }
}

#[cfg(test)]
mod tests {
    use ark_ff::PrimeField;
    use ark_std::{log2, Zero};
    use cyclotomic_rings::rings::FrogPoseidonConfig as PC;
    use stark_rings::cyclotomic_ring::models::frog_ring::RqPoly as R;

    use super::*;
    use crate::transcript::PoseidonTranscript;

    #[test]
    fn test_range_check() {
        // f: [
        // 2 + 5X
        // 4 + X^2
        // ]
        let mut f = vec![R::zero(); 1 << 15];
        f[0].coeffs_mut()[0] = 2u128.into();
        f[0].coeffs_mut()[1] = 5u128.into();
        f[1].coeffs_mut()[0] = 4u128.into();
        f[1].coeffs_mut()[2] = 1u128.into();

        let n = f.len();
        let kappa = 1;
        let b = (R::dimension() / 2) as u128;
        let k = 2;
        // log_d' (q)
        let l = ((<<R as PolyRing>::BaseRing>::MODULUS.0[0] as f64).ln()
            / ((R::dimension() / 2) as f64).ln())
        .ceil() as usize;

        let A = Matrix::<R>::rand(&mut ark_std::test_rng(), kappa, n);

        let dparams = DecompParameters { b, k, l };
        let instance = RgInstance::from_f(f.clone(), &A, &dparams);

        let rg = Rg {
            nvars: log2(n) as usize,
            instances: vec![instance],
            dparams,
        };

        let mut ts = PoseidonTranscript::empty::<PC>();
        let dcom = rg.range_check(&[], &mut ts);

        let mut ts = PoseidonTranscript::empty::<PC>();
        dcom.verify(&mut ts).unwrap();
    }

    #[test]
    fn test_range_check_mm() {
        // f: [
        // 2 + 5X
        // 4 + X^2
        // ]
        let n = 1 << 15;
        let mut f = vec![R::zero(); n];
        f[0].coeffs_mut()[0] = 2u128.into();
        f[0].coeffs_mut()[1] = 5u128.into();
        f[1].coeffs_mut()[0] = 4u128.into();
        f[1].coeffs_mut()[2] = 1u128.into();

        let mut m = SparseMatrix::identity(n);
        m.coeffs[0][0].0 = 2u128.into();
        let M = vec![m];

        let kappa = 1;
        let b = (R::dimension() / 2) as u128;
        let k = 2;
        // log_d' (q)
        let l = ((<<R as PolyRing>::BaseRing>::MODULUS.0[0] as f64).ln()
            / ((R::dimension() / 2) as f64).ln())
        .ceil() as usize;

        let A = Matrix::<R>::rand(&mut ark_std::test_rng(), kappa, n);

        let dparams = DecompParameters { b, k, l };
        let instance = RgInstance::from_f(f.clone(), &A, &dparams);

        let rg = Rg {
            nvars: log2(n) as usize,
            instances: vec![instance],
            dparams,
        };

        let mut ts = PoseidonTranscript::empty::<PC>();
        let dcom = rg.range_check(&M, &mut ts);

        let mut ts = PoseidonTranscript::empty::<PC>();
        dcom.verify(&mut ts).unwrap();
    }
}
