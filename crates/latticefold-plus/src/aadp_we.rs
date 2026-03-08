//! Arithmetic ADP witness-encryption backend.
//!
//! This is a direct implementation of the core AADP generator / encryption /
//! decryption pattern from `eprint 2026/175`, Sections 2.3 and 2.4.
//!
//! Scope of this module:
//! - represent arithmetic constraint systems of the form `a(x) * b(x) = c(x) * d(x)`
//! - compile them into AADP ciphertext coefficient matrices
//! - encrypt / decrypt a single field element message
//!
//! Important:
//! - this module does **not** decide whether a given constraint system is projectively safe
//! - callers are responsible for supplying a projectively safe constraint system when they need
//!   the security properties claimed in the paper

use ark_ff::PrimeField;
use rand::RngCore;

/// Sparse linear form over witness variables, with an explicit constant term.
///
/// The AADP formal variable `x0 = 1` is represented here by `constant`.
/// Witness variables are indexed `0..num_variables`.
#[derive(Clone, Debug, Default)]
pub struct AadpLinearForm<F: PrimeField> {
    pub constant: F,
    pub terms: Vec<(usize, F)>,
}

impl<F: PrimeField> AadpLinearForm<F> {
    pub fn eval(&self, witness: &[F]) -> Result<F, String> {
        let mut acc = self.constant;
        for &(idx, coeff) in &self.terms {
            let v = witness
                .get(idx)
                .ok_or_else(|| format!("linear form witness index out of range: idx={idx}"))?;
            acc += coeff * *v;
        }
        Ok(acc)
    }
}

/// One arithmetic constraint `a(x) * b(x) = c(x) * d(x)`.
#[derive(Clone, Debug)]
pub struct AadpMulConstraint<F: PrimeField> {
    pub a: AadpLinearForm<F>,
    pub b: AadpLinearForm<F>,
    pub c: AadpLinearForm<F>,
    pub d: AadpLinearForm<F>,
}

impl<F: PrimeField> AadpMulConstraint<F> {
    pub fn eval_holds(&self, witness: &[F]) -> Result<bool, String> {
        let a = self.a.eval(witness)?;
        let b = self.b.eval(witness)?;
        let c = self.c.eval(witness)?;
        let d = self.d.eval(witness)?;
        Ok(a * b == c * d)
    }
}

/// Arithmetic constraint system consumed by the AADP generator.
#[derive(Clone, Debug, Default)]
pub struct AadpConstraintSystem<F: PrimeField> {
    pub num_variables: usize,
    pub constraints: Vec<AadpMulConstraint<F>>,
}

impl<F: PrimeField> AadpConstraintSystem<F> {
    pub fn gate_count(&self) -> usize {
        self.constraints.len()
    }

    pub fn matrix_dim(&self) -> usize {
        self.constraints
            .len()
            .checked_mul(2)
            .and_then(|x| x.checked_add(1))
            .unwrap_or(1)
    }

    pub fn check_witness(&self, witness: &[F]) -> Result<(), String> {
        if witness.len() != self.num_variables {
            return Err(format!(
                "AADP witness length mismatch: got={} expected={}",
                witness.len(),
                self.num_variables
            ));
        }
        for (i, c) in self.constraints.iter().enumerate() {
            let a = c.a.eval(witness)?;
            let b = c.b.eval(witness)?;
            let cc = c.c.eval(witness)?;
            let d = c.d.eval(witness)?;
            let lhs = a * b;
            let rhs = cc * d;
            if lhs != rhs {
                return Err(format!(
                    "AADP witness violates constraint {i}: lhs={} rhs={} a={} b={} c={} d={} a_terms={} b_terms={} c_terms={} d_terms={}",
                    lhs.into_bigint().as_ref()[0],
                    rhs.into_bigint().as_ref()[0],
                    a.into_bigint().as_ref()[0],
                    b.into_bigint().as_ref()[0],
                    cc.into_bigint().as_ref()[0],
                    d.into_bigint().as_ref()[0],
                    c.a.terms.len(),
                    c.b.terms.len(),
                    c.c.terms.len(),
                    c.d.terms.len(),
                ));
            }
        }
        Ok(())
    }
}

/// One randomized AADP ciphertext.
///
/// `matrices[0]` is the constant matrix `M0` and `matrices[i + 1]` corresponds to witness
/// variable `x_i`.
#[derive(Clone, Debug)]
pub struct AadpCiphertext<F: PrimeField> {
    pub num_variables: usize,
    pub dim: usize,
    pub matrices: Vec<Vec<F>>,
}

/// Byte-wise AADP encryption of a short message.
///
/// This is useful for prototype backends over very small fields such as `F257`, where a single field
/// element cannot carry a 128-bit seed.
#[derive(Clone, Debug)]
pub struct AadpByteCiphertext<F: PrimeField> {
    pub parts: Vec<AadpCiphertext<F>>,
}

impl<F: PrimeField> AadpCiphertext<F> {
    pub fn evaluate(&self, witness: &[F]) -> Result<Vec<F>, String> {
        if witness.len() != self.num_variables {
            return Err(format!(
                "AADP evaluate witness length mismatch: got={} expected={}",
                witness.len(),
                self.num_variables
            ));
        }
        if self.matrices.len() != self.num_variables + 1 {
            return Err("AADP ciphertext matrix count mismatch".to_string());
        }
        let mut out = self
            .matrices
            .first()
            .cloned()
            .ok_or_else(|| "AADP ciphertext has no constant matrix".to_string())?;
        let nn = self
            .dim
            .checked_mul(self.dim)
            .ok_or_else(|| "AADP evaluate dim^2 overflow".to_string())?;
        if out.len() != nn {
            return Err("AADP constant matrix size mismatch".to_string());
        }
        for (i, &x) in witness.iter().enumerate() {
            let m = self
                .matrices
                .get(i + 1)
                .ok_or_else(|| format!("AADP missing witness matrix {i}"))?;
            if m.len() != nn {
                return Err(format!("AADP witness matrix size mismatch at index {i}"));
            }
            for j in 0..nn {
                out[j] += m[j] * x;
            }
        }
        Ok(out)
    }

    pub fn decrypt_scalar(&self, witness: &[F]) -> Result<F, String> {
        let eval = self.evaluate(witness)?;
        let det_eval = determinant(self.dim, eval.as_slice())?;
        let coeff = if self.dim == 1 {
            F::ONE
        } else {
            let mut minor = Vec::with_capacity((self.dim - 1) * (self.dim - 1));
            for r in 0..(self.dim - 1) {
                for c in 0..(self.dim - 1) {
                    minor.push(eval[r * self.dim + c]);
                }
            }
            determinant(self.dim - 1, minor.as_slice())?
        };
        if coeff.is_zero() {
            return Err("AADP decrypt failed: bottom-right cofactor is zero".to_string());
        }
        Ok(det_eval * coeff.inverse().unwrap())
    }

    #[cfg(test)]
    pub fn decrypt_u128(&self, witness: &[F]) -> Result<u128, String> {
        let msg = self.decrypt_scalar(witness)?;
        field_to_u128(msg)
    }

    pub fn ciphertext_size_bytes_estimate(&self, field_bytes: usize) -> u128 {
        let total_entries: u128 = self.matrices.iter().map(|m| m.len() as u128).sum();
        total_entries.saturating_mul(field_bytes as u128)
    }
}

/// Encrypt one `u128` seed under an arithmetic constraint system.
#[cfg(test)]
pub fn aadp_encrypt_u128<F: PrimeField, R: RngCore>(
    cs: &AadpConstraintSystem<F>,
    msg: u128,
    rng: &mut R,
) -> Result<AadpCiphertext<F>, String> {
    aadp_encrypt_scalar(cs, F::from(msg), rng)
}

pub fn aadp_encrypt_bytes<F: PrimeField, R: RngCore>(
    cs: &AadpConstraintSystem<F>,
    msg: &[u8],
    rng: &mut R,
) -> Result<AadpByteCiphertext<F>, String> {
    let mut parts = Vec::with_capacity(msg.len());
    for &b in msg {
        parts.push(aadp_encrypt_scalar(cs, F::from(b as u64), rng)?);
    }
    Ok(AadpByteCiphertext { parts })
}

/// Encrypt one field element under an arithmetic constraint system.
pub fn aadp_encrypt_scalar<F: PrimeField, R: RngCore>(
    cs: &AadpConstraintSystem<F>,
    msg: F,
    rng: &mut R,
) -> Result<AadpCiphertext<F>, String> {
    let m = cs.constraints.len();
    if m == 0 {
        return Err("AADP requires at least one constraint".to_string());
    }
    let dim = cs.matrix_dim();
    let nn = dim
        .checked_mul(dim)
        .ok_or_else(|| "AADP dim^2 overflow".to_string())?;
    let mut matrices = vec![vec![F::ZERO; nn]; cs.num_variables + 1];

    for constraint in &cs.constraints {
        let l = random_matrix::<F, R>(dim, 4, rng);
        let r = random_matrix::<F, R>(4, dim, rng);
        let b_a = basis_matrix_contribution::<F>(dim, &l, &r, AadpBasis::A)?;
        let b_b = basis_matrix_contribution::<F>(dim, &l, &r, AadpBasis::B)?;
        let b_c = basis_matrix_contribution::<F>(dim, &l, &r, AadpBasis::C)?;
        let b_d = basis_matrix_contribution::<F>(dim, &l, &r, AadpBasis::D)?;
        let b_xi = basis_matrix_contribution::<F>(dim, &l, &r, AadpBasis::Xi)?;

        // Add the user-supplied linear forms.
        add_linear_form_to_matrices(matrices.as_mut_slice(), &constraint.a, b_a.as_slice())?;
        add_linear_form_to_matrices(matrices.as_mut_slice(), &constraint.b, b_b.as_slice())?;
        add_linear_form_to_matrices(matrices.as_mut_slice(), &constraint.c, b_c.as_slice())?;
        add_linear_form_to_matrices(matrices.as_mut_slice(), &constraint.d, b_d.as_slice())?;

        // Add the fresh random xi(x) linear form.
        let mut xi = AadpLinearForm::<F> {
            constant: F::rand(rng),
            terms: Vec::with_capacity(cs.num_variables),
        };
        for idx in 0..cs.num_variables {
            xi.terms.push((idx, F::rand(rng)));
        }
        add_linear_form_to_matrices(matrices.as_mut_slice(), &xi, b_xi.as_slice())?;
    }

    // Algorithm 2: add the message to the bottom-right cell of M0.
    let last = dim - 1;
    matrices[0][last * dim + last] += msg;

    Ok(AadpCiphertext {
        num_variables: cs.num_variables,
        dim,
        matrices,
    })
}

#[derive(Clone, Copy)]
enum AadpBasis {
    A,
    B,
    C,
    D,
    Xi,
}

fn add_linear_form_to_matrices<F: PrimeField>(
    matrices: &mut [Vec<F>],
    form: &AadpLinearForm<F>,
    basis_matrix: &[F],
) -> Result<(), String> {
    if matrices.is_empty() {
        return Err("AADP matrices are empty".to_string());
    }
    if matrices[0].len() != basis_matrix.len() {
        return Err("AADP basis matrix size mismatch".to_string());
    }
    for (dst, src) in matrices[0].iter_mut().zip(basis_matrix.iter()) {
        *dst += *src * form.constant;
    }
    for &(idx, coeff) in &form.terms {
        let dst = matrices
            .get_mut(idx + 1)
            .ok_or_else(|| format!("AADP variable index out of range: idx={idx}"))?;
        if dst.len() != basis_matrix.len() {
            return Err(format!("AADP variable matrix size mismatch at idx={idx}"));
        }
        for (d, s) in dst.iter_mut().zip(basis_matrix.iter()) {
            *d += *s * coeff;
        }
    }
    Ok(())
}

fn basis_matrix_contribution<F: PrimeField>(
    dim: usize,
    l: &[F],
    r: &[F],
    basis: AadpBasis,
) -> Result<Vec<F>, String> {
    if l.len() != dim * 4 || r.len() != 4 * dim {
        return Err("AADP L/R shape mismatch".to_string());
    }
    let mut out = vec![F::ZERO; dim * dim];
    for row in 0..dim {
        for col in 0..dim {
            let v = match basis {
                AadpBasis::A => {
                    l[row * 4] * r[col] + l[row * 4 + 3] * r[3 * dim + col]
                }
                AadpBasis::B => {
                    l[row * 4 + 1] * r[dim + col] + l[row * 4 + 2] * r[2 * dim + col]
                }
                AadpBasis::C => {
                    l[row * 4] * r[dim + col] + l[row * 4 + 2] * r[3 * dim + col]
                }
                AadpBasis::D => {
                    l[row * 4 + 1] * r[col] + l[row * 4 + 3] * r[2 * dim + col]
                }
                AadpBasis::Xi => {
                    -l[row * 4] * r[2 * dim + col] + l[row * 4 + 1] * r[3 * dim + col]
                }
            };
            out[row * dim + col] = v;
        }
    }
    Ok(out)
}

fn random_matrix<F: PrimeField, R: RngCore>(rows: usize, cols: usize, rng: &mut R) -> Vec<F> {
    let mut out = Vec::with_capacity(rows * cols);
    for _ in 0..rows * cols {
        out.push(F::rand(rng));
    }
    out
}

fn determinant<F: PrimeField>(dim: usize, data: &[F]) -> Result<F, String> {
    if dim == 0 {
        return Ok(F::ONE);
    }
    if data.len() != dim * dim {
        return Err(format!(
            "AADP determinant size mismatch: len={} expected={}",
            data.len(),
            dim * dim
        ));
    }
    let mut a = data.to_vec();
    let mut det = F::ONE;
    for i in 0..dim {
        let mut pivot = i;
        while pivot < dim && a[pivot * dim + i].is_zero() {
            pivot += 1;
        }
        if pivot == dim {
            return Ok(F::ZERO);
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
            .ok_or_else(|| "AADP pivot inverse unexpectedly missing".to_string())?;
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

#[cfg(test)]
fn field_to_u128<F: PrimeField>(x: F) -> Result<u128, String> {
    let bigint = x.into_bigint();
    let limbs = bigint.as_ref();
    if limbs.len() > 2 && limbs[2..].iter().any(|&w| w != 0) {
        return Err("AADP decrypted field element does not fit in u128".to_string());
    }
    let lo = *limbs.get(0).unwrap_or(&0u64) as u128;
    let hi = *limbs.get(1).unwrap_or(&0u64) as u128;
    Ok(lo | (hi << 64))
}

impl<F: PrimeField> AadpByteCiphertext<F> {
    pub fn decrypt_bytes(&self, witness: &[F]) -> Result<Vec<u8>, String> {
        let mut out = Vec::with_capacity(self.parts.len());
        for (i, part) in self.parts.iter().enumerate() {
            let x = part.decrypt_scalar(witness)?;
            let bigint = x.into_bigint();
            let limbs = bigint.as_ref();
            if limbs.len() > 1 && limbs[1..].iter().any(|&w| w != 0) {
                return Err(format!(
                    "AADP decrypted byte part does not fit in u8 at index {i}"
                ));
            }
            let b = *limbs.get(0).unwrap_or(&0u64);
            if b > 255 {
                return Err(format!("AADP decrypted byte part out of range at index {i}: {b}"));
            }
            out.push(b as u8);
        }
        Ok(out)
    }

    pub fn ciphertext_size_bytes_estimate(&self, field_bytes: usize) -> u128 {
        self.parts
            .iter()
            .map(|ct| ct.ciphertext_size_bytes_estimate(field_bytes))
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_ff::Field;
    use latticefold::transcript::poseidon::F257;
    use rand::{rngs::StdRng, SeedableRng};

    #[test]
    fn test_aadp_encrypt_decrypt_scalar_simple_projective_safe_bitcheck() {
        // Single variable with certificate x^2 = x0 * x.
        //
        // This is the simplest projectively-safe constraint family from the paper.
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
        cs.check_witness(witness.as_slice()).expect("witness holds");

        let mut rng = StdRng::seed_from_u64(42);
        let msg = F257::from(77u64);
        let ct = aadp_encrypt_scalar(&cs, msg, &mut rng).expect("aadp encrypt");
        let got = ct.decrypt_scalar(witness.as_slice()).expect("aadp decrypt");
        assert_eq!(got, msg);
    }

    #[test]
    fn test_aadp_ciphertext_shape() {
        let cs = AadpConstraintSystem::<F257> {
            num_variables: 3,
            constraints: vec![AadpMulConstraint {
                a: AadpLinearForm {
                    constant: F257::ZERO,
                    terms: vec![(0, F257::ONE)],
                },
                b: AadpLinearForm {
                    constant: F257::ZERO,
                    terms: vec![(1, F257::ONE)],
                },
                c: AadpLinearForm {
                    constant: F257::ZERO,
                    terms: vec![(2, F257::ONE)],
                },
                d: AadpLinearForm {
                    constant: F257::ONE,
                    terms: Vec::new(),
                },
            }],
        };
        let mut rng = StdRng::seed_from_u64(7);
        let ct = aadp_encrypt_scalar(&cs, F257::from(5u64), &mut rng).expect("aadp encrypt");
        assert_eq!(ct.dim, 3);
        assert_eq!(ct.matrices.len(), 4);
        for m in &ct.matrices {
            assert_eq!(m.len(), 9);
        }
    }

    #[test]
    fn test_aadp_encrypt_decrypt_u128() {
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
        let mut rng = StdRng::seed_from_u64(99);
        let ct = aadp_encrypt_u128(&cs, 123u128, &mut rng).expect("aadp encrypt u128");
        let got = ct.decrypt_u128(witness.as_slice()).expect("aadp decrypt u128");
        assert_eq!(got, 123u128);
    }

    #[test]
    fn test_aadp_encrypt_decrypt_bytes() {
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
        let mut rng = StdRng::seed_from_u64(1234);
        let msg = [0u8, 1, 2, 3, 250, 251, 252, 253, 254, 255];
        let ct = aadp_encrypt_bytes(&cs, &msg, &mut rng).expect("aadp encrypt bytes");
        let got = ct.decrypt_bytes(witness.as_slice()).expect("aadp decrypt bytes");
        assert_eq!(got, msg);
    }
}

