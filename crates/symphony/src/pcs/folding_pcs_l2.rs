//! Folding PCS base protocol for ℓ=2 (paper 2024-281, Figure 5).
//!
//! We keep this as a standalone, **paper-faithful** algebraic verifier (and helper ops)
//! because it is the exact object we later arithmetize into dR1CS/DPP.
//!
//! Important modeling choices in Symphony:
//! - We represent vectors in a **row-block** layout compatible with `C^T ⊗ I_n` operations.
//! - Verifier challenges `C1,C2` are treated as Fiat–Shamir coins (Poseidon-derived in the WE gate),
//!   so the canonical proof object should NOT include them.

use ark_ff::{BigInteger, PrimeField};
use ark_std::vec::Vec;
use num_bigint::BigUint;
use num_traits::ToPrimitive;
use rayon::prelude::*;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BinMatrix<F: PrimeField> {
    pub rows: usize,
    pub cols: usize,
    /// Row-major entries, each must be 0/1 in `F`.
    pub data: Vec<F>,
}

impl<F: PrimeField> BinMatrix<F> {
    pub fn get(&self, row: usize, col: usize) -> F {
        self.data[row * self.cols + col]
    }
}

/// Dense matrix in row-major form.
#[derive(Clone, Debug)]
pub struct DenseMatrix<F: PrimeField> {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<F>,
}

impl<F: PrimeField> DenseMatrix<F> {
    pub fn new(rows: usize, cols: usize, data: Vec<F>) -> Self {
        assert_eq!(rows * cols, data.len());
        Self { rows, cols, data }
    }

    #[inline]
    pub fn get(&self, r: usize, c: usize) -> F {
        self.data[r * self.cols + c]
    }

    pub fn mul_vec_par(&self, x: &[F]) -> Vec<F> {
        assert_eq!(x.len(), self.cols);
        (0..self.rows)
            .into_par_iter()
            .map(|r| {
                let mut acc = F::ZERO;
                let base = r * self.cols;
                for c in 0..self.cols {
                    acc += self.data[base + c] * x[c];
                }
                acc
            })
            .collect()
    }
}

#[derive(Clone, Debug)]
pub struct FoldingPcsL2Params<F: PrimeField> {
    pub r: usize,
    pub kappa: usize,
    pub n: usize,
    /// Base-δ gadget: each packed element is represented by `alpha` digits.
    pub delta: u64,
    pub alpha: usize,
    /// Norm bounds for the three short vectors y0,y1,y2 (paper: β0,β1,β2).
    ///
    /// Interpreted as a **per-coordinate signed bound**: each coefficient must be in [-β, β].
    pub beta0: u64,
    pub beta1: u64,
    pub beta2: u64,
    /// Matrix A ∈ F^{n × (r*n*alpha)} (paper: A ∈ Z_q^{n×r n log q}).
    pub a: DenseMatrix<F>,
}

impl<F: PrimeField> FoldingPcsL2Params<F> {
    pub fn y0_len(&self) -> usize { self.kappa * self.r * self.n * self.alpha }
    pub fn y1_len(&self) -> usize { self.kappa * self.r * self.n * self.alpha }
    pub fn y2_len(&self) -> usize { self.kappa * self.r * self.n * self.alpha }

    pub fn f_len(&self) -> usize { self.r * self.r * self.r * self.kappa * self.n }
    pub fn t_len(&self) -> usize { self.kappa * self.n }

    pub fn v0_len(&self) -> usize { self.r * self.kappa * self.n }
    pub fn v1_len(&self) -> usize { self.r * self.kappa * self.n }
    pub fn v2_len(&self) -> usize { self.r * self.kappa * self.n }

    #[inline]
    fn delta_pows(&self) -> Vec<F> {
        let mut pows = Vec::with_capacity(self.alpha);
        let mut acc = F::ONE;
        let delta = F::from(self.delta);
        for _ in 0..self.alpha {
            pows.push(acc);
            acc *= delta;
        }
        pows
    }
}

/// Canonical proof object *without* verifier coins.
#[derive(Clone, Debug)]
pub struct FoldingPcsL2ProofCore<F: PrimeField> {
    pub y0: Vec<F>,
    pub v0: Vec<F>,
    pub y1: Vec<F>,
    pub v1: Vec<F>,
    pub y2: Vec<F>,
    pub v2: Vec<F>,
}

/// Multiply by (C^T ⊗ I_block).
pub fn kron_ct_in_mul<F: PrimeField>(c: &BinMatrix<F>, block_len: usize, v: &[F]) -> Vec<F> {
    assert_eq!(v.len(), c.rows * block_len);
    let mut out = vec![F::ZERO; c.cols * block_len];
    out.par_chunks_mut(block_len)
        .enumerate()
        .for_each(|(col, out_block)| {
            for row in 0..c.rows {
                let bit = c.get(row, col);
                if bit != F::ZERO {
                    let in_block = &v[row * block_len..(row + 1) * block_len];
                    for (o, x) in out_block.iter_mut().zip(in_block.iter()) {
                        *o += bit * *x;
                    }
                }
            }
        });
    out
}

/// Multiply by \((I_{\kappa n} \otimes x^T)\) on a vector in **row-block** layout.
pub fn kron_ikn_xt_mul<F: PrimeField>(x: &[F], kappa: usize, n: usize, v: &[F]) -> Vec<F> {
    let r = x.len();
    assert!(r > 0);
    assert_eq!(v.len(), r * kappa * n);
    (0..kappa)
        .into_par_iter()
        .flat_map_iter(|k| {
            (0..n).map(move |j| {
                let mut acc = F::ZERO;
                for a in 0..r {
                    acc += x[a] * v[((k * r + a) * n) + j];
                }
                acc
            })
        })
        .collect()
}

/// Apply gadget matrix G_{len} to digit vector y_digits, producing packed length `len`.
pub fn gadget_apply_digits<F: PrimeField>(delta_pows: &[F], len: usize, y_digits: &[F]) -> Vec<F> {
    let alpha = delta_pows.len();
    assert_eq!(y_digits.len(), len * alpha);
    (0..len)
        .into_par_iter()
        .map(|i| {
            let mut acc = F::ZERO;
            let base = i * alpha;
            for j in 0..alpha {
                acc += y_digits[base + j] * delta_pows[j];
            }
            acc
        })
        .collect()
}

/// Multiply by (I_κ ⊗ A) where A: n × (r*n*alpha), and input y is κ blocks of length r*n*alpha.
pub fn kron_i_a_mul<F: PrimeField>(a: &DenseMatrix<F>, kappa: usize, in_block_len: usize, y: &[F]) -> Vec<F> {
    assert_eq!(a.cols, in_block_len);
    assert_eq!(y.len(), kappa * in_block_len);
    (0..kappa)
        .into_par_iter()
        .flat_map_iter(|i| {
            let block = &y[i * in_block_len..(i + 1) * in_block_len];
            a.mul_vec_par(block)
        })
        .collect()
}

fn field_to_centered_mag_u64<Fp: PrimeField>(x: Fp) -> u64 {
    let v = BigUint::from_bytes_le(&x.into_bigint().to_bytes_le());
    let p = BigUint::from_bytes_le(&Fp::MODULUS.to_bytes_le());
    let half = &p >> 1;
    if v > half { (&p - v).to_u64().unwrap_or(u64::MAX) } else { v.to_u64().unwrap_or(u64::MAX) }
}

fn check_vec_bound<Fp: PrimeField>(v: &[Fp], beta: u64) -> Result<(), String> {
    for (i, &x) in v.iter().enumerate() {
        let mag = field_to_centered_mag_u64::<Fp>(x);
        if mag > beta {
            return Err(format!("verify: norm bound violated at idx {i}: |x|={mag} > {beta}"));
        }
    }
    Ok(())
}

/// Plain verifier for ℓ=2 folding PCS base protocol, with explicit verifier coins `C1,C2`.
pub fn verify_folding_pcs_l2_with_c_matrices<F: PrimeField>(
    p: &FoldingPcsL2Params<F>,
    t: &[F],
    x0: &[F],
    x1: &[F],
    x2: &[F],
    u: &[F],
    proof: &FoldingPcsL2ProofCore<F>,
    c1: &BinMatrix<F>,
    c2: &BinMatrix<F>,
) -> Result<(), String> {
    if t.len() != p.t_len() || u.len() != p.t_len() {
        return Err("dim: t/u".to_string());
    }
    if x0.len() != p.r || x1.len() != p.r || x2.len() != p.r {
        return Err("dim: x".to_string());
    }
    if proof.y0.len() != p.y0_len()
        || proof.y1.len() != p.y1_len()
        || proof.y2.len() != p.y2_len()
        || proof.v0.len() != p.v0_len()
        || proof.v1.len() != p.v1_len()
        || proof.v2.len() != p.v2_len()
    {
        return Err("dim: proof".to_string());
    }
    if c1.rows != p.r * p.kappa || c1.cols != p.kappa {
        return Err("dim: c1".to_string());
    }
    if c2.rows != p.r * p.kappa || c2.cols != p.kappa {
        return Err("dim: c2".to_string());
    }

    // Norm bounds for y0,y1,y2 (per-coordinate signed bounds).
    check_vec_bound::<F>(&proof.y0, p.beta0)?;
    check_vec_bound::<F>(&proof.y1, p.beta1)?;
    check_vec_bound::<F>(&proof.y2, p.beta2)?;

    let delta_pows = p.delta_pows();

    // (I_k ⊗ A) y0 = t
    let lhs0 = kron_i_a_mul(&p.a, p.kappa, p.r * p.n * p.alpha, &proof.y0);
    if lhs0 != t {
        return Err("verify: lhs0".to_string());
    }

    // (I_k ⊗ A) y1 = (C1^T ⊗ I_n) G y0
    let gy0 = gadget_apply_digits(&delta_pows, p.r * p.kappa * p.n, &proof.y0);
    let rhs1 = kron_ct_in_mul(c1, p.n, &gy0);
    let lhs1 = kron_i_a_mul(&p.a, p.kappa, p.r * p.n * p.alpha, &proof.y1);
    if lhs1 != rhs1 {
        return Err("verify: lhs1".to_string());
    }

    // (I_k ⊗ A) y2 = (C2^T ⊗ I_n) G y1
    let gy1 = gadget_apply_digits(&delta_pows, p.r * p.kappa * p.n, &proof.y1);
    let rhs2 = kron_ct_in_mul(c2, p.n, &gy1);
    let lhs2 = kron_i_a_mul(&p.a, p.kappa, p.r * p.n * p.alpha, &proof.y2);
    if lhs2 != rhs2 {
        return Err("verify: lhs2".to_string());
    }

    // G y2 = v2
    let gy2 = gadget_apply_digits(&delta_pows, p.r * p.kappa * p.n, &proof.y2);
    if gy2 != proof.v2 {
        return Err("verify: gy2".to_string());
    }

    // (I_{κn} ⊗ x2^T) v0 = u
    let u_re = kron_ikn_xt_mul(x2, p.kappa, p.n, &proof.v0);
    if u_re != u {
        return Err("verify: u".to_string());
    }

    // (I_{κn} ⊗ x1^T) v1 = (C1^T ⊗ I_n) v0
    let lhs_v1 = kron_ikn_xt_mul(x1, p.kappa, p.n, &proof.v1);
    let rhs_v1 = kron_ct_in_mul(c1, p.n, &proof.v0);
    if lhs_v1 != rhs_v1 {
        return Err("verify: v1".to_string());
    }

    // (I_{κn} ⊗ x0^T) v2 = (C2^T ⊗ I_n) v1
    let lhs_v2 = kron_ikn_xt_mul(x0, p.kappa, p.n, &proof.v2);
    let rhs_v2 = kron_ct_in_mul(c2, p.n, &proof.v1);
    if lhs_v2 != rhs_v2 {
        return Err("verify: v2".to_string());
    }

    Ok(())
}

// NOTE: We intentionally do NOT provide a wrapper that takes prover-supplied `C1/C2`.
// In this setting, `C1/C2` must be derived from the transcript and provided explicitly.

