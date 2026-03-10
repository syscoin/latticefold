//! Extension-field AADP backend for monolithic H12 seeds.
//!
//! This module lifts the current `F257` local-view relation into the characteristic-257 extension
//! field `F257Ext16`, so the outer capsule can encrypt one single `u128`-sized seed object rather
//! than 16 independent byte ciphertexts.
//!
//! This module only changes the **message carrier**. The relation witness remains typed as `F257`
//! in the API (`evaluate` / `decrypt_seed` take `&[F257]`), so there is no separate extension-field
//! witness input path that would need additional subfield-membership constraints here.

use ark_ff::PrimeField;
use latticefold::transcript::poseidon::F257;
use rand::RngCore;
use rayon::prelude::*;

use crate::{aadp_we::AadpConstraintSystem, f257_ext16::F257Ext16};

#[derive(Clone, Debug)]
pub struct AadpCiphertextExt16 {
    pub num_variables: usize,
    pub dim: usize,
    pub matrices: Vec<Vec<F257Ext16>>,
}

impl AadpCiphertextExt16 {
    pub fn evaluate(&self, witness: &[F257]) -> Result<Vec<F257Ext16>, String> {
        if witness.len() != self.num_variables {
            return Err(format!(
                "AADP-ext evaluate witness length mismatch: got={} expected={}",
                witness.len(),
                self.num_variables
            ));
        }
        if self.matrices.len() != self.num_variables + 1 {
            return Err("AADP-ext ciphertext matrix count mismatch".to_string());
        }
        let mut out = self
            .matrices
            .first()
            .cloned()
            .ok_or_else(|| "AADP-ext ciphertext has no constant matrix".to_string())?;
        let nn = self
            .dim
            .checked_mul(self.dim)
            .ok_or_else(|| "AADP-ext evaluate dim^2 overflow".to_string())?;
        if out.len() != nn {
            return Err("AADP-ext constant matrix size mismatch".to_string());
        }
        for (i, &x) in witness.iter().enumerate() {
            let m = self
                .matrices
                .get(i + 1)
                .ok_or_else(|| format!("AADP-ext missing witness matrix {i}"))?;
            let x_ext = F257Ext16::from_f257(f257_to_u16(x));
            for j in 0..nn {
                out[j] += m[j] * x_ext;
            }
        }
        Ok(out)
    }

    pub fn decrypt_seed(&self, witness: &[F257]) -> Result<[u8; 16], String> {
        let eval = self.evaluate(witness)?;
        let det_eval = determinant_ext(self.dim, eval.as_slice())?;
        let coeff = if self.dim == 1 {
            F257Ext16::one()
        } else {
            let mut minor = Vec::with_capacity((self.dim - 1) * (self.dim - 1));
            for r in 0..(self.dim - 1) {
                for c in 0..(self.dim - 1) {
                    minor.push(eval[r * self.dim + c]);
                }
            }
            determinant_ext(self.dim - 1, minor.as_slice())?
        };
        if coeff.is_zero() {
            return Err("AADP-ext decrypt failed: bottom-right cofactor is zero".to_string());
        }
        let msg = det_eval
            * coeff
                .inverse()
                .ok_or_else(|| "AADP-ext decrypt failed: cofactor inverse missing".to_string())?;
        let seed = msg.to_u128_seed()?;
        Ok(seed.to_le_bytes())
    }

    pub fn ciphertext_size_bytes_estimate(&self) -> u128 {
        let total_entries: u128 = self.matrices.iter().map(|m| m.len() as u128).sum();
        total_entries.saturating_mul(32)
    }
}

pub fn aadp_encrypt_ext16_seed<R: RngCore>(
    cs: &AadpConstraintSystem<F257>,
    seed: [u8; 16],
    rng: &mut R,
) -> Result<AadpCiphertextExt16, String> {
    let m = cs.constraints.len();
    if m == 0 {
        return Err("AADP-ext requires at least one constraint".to_string());
    }
    let dim = cs.matrix_dim();
    let nn = dim
        .checked_mul(dim)
        .ok_or_else(|| "AADP-ext dim^2 overflow".to_string())?;
    let mut matrices = vec![vec![F257Ext16::zero(); nn]; cs.num_variables + 1];

    for constraint in &cs.constraints {
        let l = random_matrix_ext16(dim, 4, rng);
        let r = random_matrix_ext16(4, dim, rng);
        let b_a = basis_matrix_contribution_ext16(dim, &l, &r, AadpBasisExt::A)?;
        let b_b = basis_matrix_contribution_ext16(dim, &l, &r, AadpBasisExt::B)?;
        let b_c = basis_matrix_contribution_ext16(dim, &l, &r, AadpBasisExt::C)?;
        let b_d = basis_matrix_contribution_ext16(dim, &l, &r, AadpBasisExt::D)?;
        let b_xi = basis_matrix_contribution_ext16(dim, &l, &r, AadpBasisExt::Xi)?;

        add_f257_form_to_ext_matrices(matrices.as_mut_slice(), &constraint.a, b_a.as_slice())?;
        add_f257_form_to_ext_matrices(matrices.as_mut_slice(), &constraint.b, b_b.as_slice())?;
        add_f257_form_to_ext_matrices(matrices.as_mut_slice(), &constraint.c, b_c.as_slice())?;
        add_f257_form_to_ext_matrices(matrices.as_mut_slice(), &constraint.d, b_d.as_slice())?;

        let xi_constant = F257Ext16::random(rng);
        for (dst, src) in matrices[0].iter_mut().zip(b_xi.iter()) {
            *dst += *src * xi_constant;
        }
        let xi_coeffs: Vec<F257Ext16> = (0..cs.num_variables)
            .map(|_| F257Ext16::random(rng))
            .collect();
        matrices[1..]
            .par_iter_mut()
            .zip(xi_coeffs.into_par_iter())
            .for_each(|(dst, coeff)| {
                for (d, s) in dst.iter_mut().zip(b_xi.iter()) {
                    *d += *s * coeff;
                }
            });
    }

    let msg = F257Ext16::from_u128_seed(u128::from_le_bytes(seed));
    let last = dim - 1;
    matrices[0][last * dim + last] += msg;

    Ok(AadpCiphertextExt16 {
        num_variables: cs.num_variables,
        dim,
        matrices,
    })
}

#[derive(Clone, Copy)]
enum AadpBasisExt {
    A,
    B,
    C,
    D,
    Xi,
}

fn add_f257_form_to_ext_matrices(
    matrices: &mut [Vec<F257Ext16>],
    form: &crate::aadp_we::AadpLinearForm<F257>,
    basis_matrix: &[F257Ext16],
) -> Result<(), String> {
    if matrices.is_empty() {
        return Err("AADP-ext matrices are empty".to_string());
    }
    if matrices[0].len() != basis_matrix.len() {
        return Err("AADP-ext basis matrix size mismatch".to_string());
    }
    let c0 = F257Ext16::from_f257(f257_to_u16(form.constant));
    for (dst, src) in matrices[0].iter_mut().zip(basis_matrix.iter()) {
        *dst += *src * c0;
    }
    for &(idx, coeff) in &form.terms {
        let dst = matrices
            .get_mut(idx + 1)
            .ok_or_else(|| format!("AADP-ext variable index out of range: idx={idx}"))?;
        let coeff = F257Ext16::from_f257(f257_to_u16(coeff));
        for (d, s) in dst.iter_mut().zip(basis_matrix.iter()) {
            *d += *s * coeff;
        }
    }
    Ok(())
}

fn basis_matrix_contribution_ext16(
    dim: usize,
    l: &[F257Ext16],
    r: &[F257Ext16],
    basis: AadpBasisExt,
) -> Result<Vec<F257Ext16>, String> {
    if l.len() != dim * 4 || r.len() != 4 * dim {
        return Err("AADP-ext L/R shape mismatch".to_string());
    }
    let mut out = vec![F257Ext16::zero(); dim * dim];
    for row in 0..dim {
        for col in 0..dim {
            let v = match basis {
                AadpBasisExt::A => l[row * 4] * r[col] + l[row * 4 + 3] * r[3 * dim + col],
                AadpBasisExt::B => {
                    l[row * 4 + 1] * r[dim + col] + l[row * 4 + 2] * r[2 * dim + col]
                }
                AadpBasisExt::C => l[row * 4] * r[dim + col] + l[row * 4 + 2] * r[3 * dim + col],
                AadpBasisExt::D => l[row * 4 + 1] * r[col] + l[row * 4 + 3] * r[2 * dim + col],
                AadpBasisExt::Xi => {
                    -l[row * 4] * r[2 * dim + col] + l[row * 4 + 1] * r[3 * dim + col]
                }
            };
            out[row * dim + col] = v;
        }
    }
    Ok(out)
}

fn random_matrix_ext16(rows: usize, cols: usize, rng: &mut impl RngCore) -> Vec<F257Ext16> {
    let mut out = Vec::with_capacity(rows * cols);
    for _ in 0..rows * cols {
        out.push(F257Ext16::random(rng));
    }
    out
}

fn determinant_ext(dim: usize, data: &[F257Ext16]) -> Result<F257Ext16, String> {
    if dim == 0 {
        return Ok(F257Ext16::one());
    }
    if data.len() != dim * dim {
        return Err(format!(
            "AADP-ext determinant size mismatch: len={} expected={}",
            data.len(),
            dim * dim
        ));
    }
    let mut a = data.to_vec();
    let mut det = F257Ext16::one();
    for i in 0..dim {
        let mut pivot = i;
        while pivot < dim && a[pivot * dim + i].is_zero() {
            pivot += 1;
        }
        if pivot == dim {
            return Ok(F257Ext16::zero());
        }
        if pivot != i {
            for c in 0..dim {
                a.swap(i * dim + c, pivot * dim + c);
            }
            det = -det;
        }
        let pivot_val = a[i * dim + i];
        det *= pivot_val;
        let inv = pivot_val
            .inverse()
            .ok_or_else(|| "AADP-ext pivot inverse unexpectedly missing".to_string())?;
        for row in (i + 1)..dim {
            let factor = a[row * dim + i] * inv;
            if factor.is_zero() {
                continue;
            }
            for col in i..dim {
                let idx = row * dim + col;
                let src = a[i * dim + col];
                a[idx] -= factor * src;
            }
        }
    }
    Ok(det)
}

#[inline]
fn f257_to_u16(f: F257) -> u16 {
    (f.into_bigint().as_ref()[0] % 257) as u16
}

#[cfg(test)]
mod tests {
    use ark_ff::Field;
    use rand::{rngs::StdRng, SeedableRng};

    use super::*;
    use crate::aadp_we::{AadpLinearForm, AadpMulConstraint};

    #[test]
    fn test_aadp_ext16_encrypt_decrypt_seed() {
        let cs = AadpConstraintSystem::<F257> {
            num_variables: 1,
            constraints: vec![AadpMulConstraint {
                a: AadpLinearForm {
                    constant: F257::ZERO,
                    terms: vec![(0, F257::ONE)],
                },
                b: AadpLinearForm {
                    constant: F257::ZERO,
                    terms: vec![(0, F257::ONE)],
                },
                c: AadpLinearForm {
                    constant: F257::ONE,
                    terms: Vec::new(),
                },
                d: AadpLinearForm {
                    constant: F257::ZERO,
                    terms: vec![(0, F257::ONE)],
                },
            }],
        };
        let witness = vec![F257::ONE];
        let seed = [42u8; 16];
        let mut rng = StdRng::seed_from_u64(7);
        let ct = aadp_encrypt_ext16_seed(&cs, seed, &mut rng).expect("encrypt ext16 seed");
        let got = ct
            .decrypt_seed(witness.as_slice())
            .expect("decrypt ext16 seed");
        assert_eq!(got, seed);
    }
}
