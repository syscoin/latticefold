//! Experimental generic field-level GERM helpers.
//!
//! This module is intentionally narrower than the H12/AADP integration:
//! - it works directly over a prime field `F`
//! - it assumes the verifier residual families are already exported
//! - it does **not** include a commitment/opening layer yet
//!
//! Its purpose is to smoke-test the core GERM algebra on large-field DPP-style exports before
//! reconnecting a full frontend such as SP1.
//!
//! The default direct export path here intentionally models the *minimal* standalone GERM
//! architecture:
//! - multiplicative residual quads are part of the security story
//! - extra linear residual rows are included only when explicitly exported by the frontend

use ark_ff::{BigInteger, PrimeField};
use dpp::dr1cs_flpcp::{Dr1csBlockLinearConstraint, Dr1csInstanceSparse, Dr1csNpFlpcpSparseApi};
use sha2::Digest;

#[derive(Clone, Debug)]
pub struct GermFieldMulQuad<F: PrimeField> {
    pub a: F,
    pub b: F,
    pub c: F,
    pub d: F,
}

#[derive(Clone, Debug)]
pub struct GermFieldMulSumcheckRound<F: PrimeField> {
    pub evaluations: [F; 4],
}

#[derive(Clone, Debug)]
pub struct GermFieldMulSumcheckProof<F: PrimeField> {
    pub nvars: u16,
    pub rounds: Vec<GermFieldMulSumcheckRound<F>>,
    pub opening: GermFieldMulQuad<F>,
}

#[derive(Clone, Debug)]
pub struct GermFieldProof<F: PrimeField> {
    pub linear_fingerprints: Vec<F>,
    pub mul_sumcheck: GermFieldMulSumcheckProof<F>,
}

#[derive(Clone, Debug)]
pub struct GermFieldResidualInstance<F: PrimeField> {
    pub linear_residuals: Vec<F>,
    pub mul_quads: Vec<GermFieldMulQuad<F>>,
}

#[derive(Clone, Debug)]
pub struct GermFieldLinearRow<F: PrimeField> {
    pub constant: F,
    pub terms: Vec<(usize, F)>,
}

pub trait GermFieldResidualExporter<F: PrimeField>: Dr1csNpFlpcpSparseApi<F> {
    fn export_germ_residual_instance(
        &self,
        x: &[F],
        z_w: &[F],
        query_samples: &[(usize, F)],
    ) -> Result<GermFieldResidualInstance<F>, String>
    where
        Self: Sized,
    {
        export_residual_instance_from_flpcp(self, x, z_w, query_samples)
    }
}

impl<F: PrimeField, P: Dr1csNpFlpcpSparseApi<F>> GermFieldResidualExporter<F> for P {}

fn absorb_field<F: PrimeField>(h: &mut sha2::Sha256, x: &F) {
    let bytes = x.into_bigint().to_bytes_le();
    h.update((bytes.len() as u32).to_le_bytes());
    h.update(bytes);
}

fn field_from_hash<F: PrimeField>(h: &sha2::Sha256) -> F {
    let bytes: [u8; 32] = h.clone().finalize().into();
    F::from_le_bytes_mod_order(&bytes)
}

fn derive_linear_weight<F: PrimeField>(seed: &[u8; 32], fp_idx: usize, term_idx: usize) -> F {
    let mut h = sha2::Sha256::new();
    h.update(b"LFP_GERM_FIELD_LIN_WEIGHT_V1");
    h.update(seed);
    h.update((fp_idx as u32).to_le_bytes());
    h.update((term_idx as u32).to_le_bytes());
    field_from_hash::<F>(&h)
}

fn derive_mul_point<F: PrimeField>(seed: &[u8; 32], nvars: usize) -> Vec<F> {
    let mut out = Vec::with_capacity(nvars);
    for var_idx in 0..nvars {
        let mut h = sha2::Sha256::new();
        h.update(b"LFP_GERM_FIELD_MUL_POINT_V1");
        h.update(seed);
        h.update((var_idx as u32).to_le_bytes());
        out.push(field_from_hash::<F>(&h));
    }
    out
}

fn derive_sumcheck_round_challenge<F: PrimeField>(
    seed: &[u8; 32],
    rounds: &[GermFieldMulSumcheckRound<F>],
    round_idx: usize,
) -> F {
    let mut h = sha2::Sha256::new();
    h.update(b"LFP_GERM_FIELD_SUMCHECK_CHAL_V1");
    h.update(seed);
    h.update((round_idx as u32).to_le_bytes());
    for round in rounds {
        for eval in &round.evaluations {
            absorb_field(&mut h, eval);
        }
    }
    field_from_hash::<F>(&h)
}

fn interpolate_0123<F: PrimeField>(evals: &[F; 4], x: F) -> Result<F, String> {
    let xs = [
        F::from(0u64),
        F::from(1u64),
        F::from(2u64),
        F::from(3u64),
    ];
    let mut acc = F::ZERO;
    for i in 0..4 {
        let mut num = F::ONE;
        let mut den = F::ONE;
        for j in 0..4 {
            if i == j {
                continue;
            }
            num *= x - xs[j];
            den *= xs[i] - xs[j];
        }
        let den_inv = den
            .inverse()
            .ok_or_else(|| "germ_field: interpolation denominator was zero".to_string())?;
        acc += evals[i] * (num * den_inv);
    }
    Ok(acc)
}

fn eq_table<F: PrimeField>(point: &[F]) -> Vec<F> {
    if point.is_empty() {
        return vec![F::ONE];
    }
    let nvars = point.len();
    let size = 1usize << nvars;
    let mut out = vec![F::ZERO; size];
    for (idx, slot) in out.iter_mut().enumerate() {
        let mut acc = F::ONE;
        for (var_idx, r_i) in point.iter().enumerate() {
            let bit = (idx >> var_idx) & 1;
            acc *= if bit == 0 { F::ONE - *r_i } else { *r_i };
        }
        *slot = acc;
    }
    out
}

fn fold_mle_table<F: PrimeField>(table: &[F], r: F) -> Vec<F> {
    let mut out = Vec::with_capacity(table.len() / 2);
    for pair in table.chunks_exact(2) {
        out.push(pair[0] + (pair[1] - pair[0]) * r);
    }
    out
}

fn mul_sumcheck_nvars(total_checks: usize) -> usize {
    if total_checks <= 1 {
        0
    } else {
        total_checks.next_power_of_two().trailing_zeros() as usize
    }
}

fn build_mul_tables<F: PrimeField>(
    quads: &[GermFieldMulQuad<F>],
) -> (usize, Vec<F>, Vec<F>, Vec<F>, Vec<F>) {
    let total_checks = quads.len();
    let padded = total_checks.max(1).next_power_of_two();
    let mut a = vec![F::ZERO; padded];
    let mut b = vec![F::ZERO; padded];
    let mut c = vec![F::ZERO; padded];
    let mut d = vec![F::ZERO; padded];
    for (idx, quad) in quads.iter().enumerate() {
        a[idx] = quad.a;
        b[idx] = quad.b;
        c[idx] = quad.c;
        d[idx] = quad.d;
    }
    (total_checks, a, b, c, d)
}

fn prove_mul_sumcheck<F: PrimeField>(
    seed: &[u8; 32],
    quads: &[GermFieldMulQuad<F>],
) -> Result<GermFieldMulSumcheckProof<F>, String> {
    let (total_checks, mut a, mut b, mut c, mut d) = build_mul_tables(quads);
    let nvars = mul_sumcheck_nvars(total_checks);
    if nvars == 0 {
        return Ok(GermFieldMulSumcheckProof {
            nvars: 0,
            rounds: Vec::new(),
            opening: GermFieldMulQuad {
                a: a[0],
                b: b[0],
                c: c[0],
                d: d[0],
            },
        });
    }
    let point = derive_mul_point::<F>(seed, nvars);
    let mut eq = eq_table(point.as_slice());
    let mut rounds = Vec::with_capacity(nvars);
    for round_idx in 0..nvars {
        let mut evals = [F::ZERO; 4];
        for idx in 0..(a.len() / 2) {
            let i0 = 2 * idx;
            let i1 = i0 + 1;
            let a0 = a[i0];
            let a1 = a[i1];
            let b0 = b[i0];
            let b1 = b[i1];
            let c0 = c[i0];
            let c1 = c[i1];
            let d0 = d[i0];
            let d1 = d[i1];
            let e0 = eq[i0];
            let e1 = eq[i1];
            for (t_idx, t_u64) in [0u64, 1, 2, 3].iter().copied().enumerate() {
                let t = F::from(t_u64);
                let at = a0 + (a1 - a0) * t;
                let bt = b0 + (b1 - b0) * t;
                let ct = c0 + (c1 - c0) * t;
                let dt = d0 + (d1 - d0) * t;
                let et = e0 + (e1 - e0) * t;
                evals[t_idx] += et * ((ct * dt) - (at * bt));
            }
        }
        let round = GermFieldMulSumcheckRound { evaluations: evals };
        rounds.push(round);
        let r_sc = derive_sumcheck_round_challenge::<F>(seed, rounds.as_slice(), round_idx);
        a = fold_mle_table(a.as_slice(), r_sc);
        b = fold_mle_table(b.as_slice(), r_sc);
        c = fold_mle_table(c.as_slice(), r_sc);
        d = fold_mle_table(d.as_slice(), r_sc);
        eq = fold_mle_table(eq.as_slice(), r_sc);
    }
    Ok(GermFieldMulSumcheckProof {
        nvars: nvars as u16,
        rounds,
        opening: GermFieldMulQuad {
            a: a[0],
            b: b[0],
            c: c[0],
            d: d[0],
        },
    })
}

fn verify_mul_sumcheck<F: PrimeField>(
    seed: &[u8; 32],
    proof: &GermFieldMulSumcheckProof<F>,
) -> Result<(), String> {
    let nvars = proof.nvars as usize;
    if proof.rounds.len() != nvars {
        return Err(format!(
            "germ_field: sumcheck rounds mismatch: got={} expected={}",
            proof.rounds.len(),
            nvars
        ));
    }
    if nvars == 0 {
        let residual = proof.opening.c * proof.opening.d - proof.opening.a * proof.opening.b;
        if !residual.is_zero() {
            return Err("germ_field: final multiplicative residual nonzero".to_string());
        }
        return Ok(());
    }
    let point = derive_mul_point::<F>(seed, nvars);
    let mut claimed = F::ZERO;
    let mut sampled = Vec::with_capacity(nvars);
    for (round_idx, round) in proof.rounds.iter().enumerate() {
        let evals = round.evaluations;
        if evals[0] + evals[1] != claimed {
            return Err(format!(
                "germ_field: sumcheck round {} identity failed",
                round_idx
            ));
        }
        let r_sc = derive_sumcheck_round_challenge::<F>(seed, &proof.rounds[..=round_idx], round_idx);
        sampled.push(r_sc);
        claimed = interpolate_0123(&evals, r_sc)?;
    }
    let mut eq_eval = F::ONE;
    for (r_i, s_i) in point.iter().zip(sampled.iter()) {
        eq_eval *= (F::ONE - *r_i) * (F::ONE - *s_i) + (*r_i * *s_i);
    }
    let final_residual =
        claimed - (eq_eval * ((proof.opening.c * proof.opening.d) - (proof.opening.a * proof.opening.b)));
    if !final_residual.is_zero() {
        return Err("germ_field: final sumcheck opening residual nonzero".to_string());
    }
    Ok(())
}

fn eval_linear_constraints<F: PrimeField>(
    constraints: &[Dr1csBlockLinearConstraint<F>],
    w_eval: &[F],
) -> Result<Vec<F>, String> {
    let mut out = Vec::with_capacity(constraints.len());
    for (ci, lc) in constraints.iter().enumerate() {
        let mut acc = lc.constant;
        for &(pos, coeff) in &lc.terms {
            let v = w_eval.get(pos).ok_or_else(|| {
                format!("germ_field: linear constraint {} position {} out of range", ci, pos)
            })?;
            acc += coeff * *v;
        }
        out.push(acc);
    }
    Ok(out)
}

pub fn eval_linear_rows<F: PrimeField>(
    rows: &[GermFieldLinearRow<F>],
    witness: &[F],
) -> Result<Vec<F>, String> {
    let mut out = Vec::with_capacity(rows.len());
    for (ri, row) in rows.iter().enumerate() {
        let mut acc = row.constant;
        for &(idx, coeff) in &row.terms {
            let v = witness.get(idx).ok_or_else(|| {
                format!("germ_field: linear row {} witness index {} out of range", ri, idx)
            })?;
            acc += coeff * *v;
        }
        out.push(acc);
    }
    Ok(out)
}

pub fn export_residual_instance_from_sparse_dr1cs<F: PrimeField>(
    inst: &Dr1csInstanceSparse<F>,
    witness: &[F],
) -> Result<GermFieldResidualInstance<F>, String> {
    if witness.len() != inst.n {
        return Err(format!(
            "germ_field: sparse dR1CS witness length mismatch: got={} expected={}",
            witness.len(),
            inst.n
        ));
    }
    if inst.a.len() != inst.b.len() || inst.a.len() != inst.c.len() {
        return Err("germ_field: sparse dR1CS row family length mismatch".to_string());
    }
    let mut mul_quads = Vec::with_capacity(inst.k());
    for row_idx in 0..inst.k() {
        mul_quads.push(GermFieldMulQuad {
            a: inst.a[row_idx].dot(witness),
            b: inst.b[row_idx].dot(witness),
            c: inst.c[row_idx].dot(witness),
            d: F::ONE,
        });
    }
    Ok(GermFieldResidualInstance {
        linear_residuals: Vec::new(),
        mul_quads,
    })
}

pub fn export_residual_instance_from_flpcp<F: PrimeField, P: Dr1csNpFlpcpSparseApi<F>>(
    flpcp: &P,
    x: &[F],
    z_w: &[F],
    query_samples: &[(usize, F)],
) -> Result<GermFieldResidualInstance<F>, String> {
    let pi = flpcp.prove(x, z_w);
    let mut v0 = Vec::with_capacity(x.len() + pi.len());
    v0.extend_from_slice(x);
    v0.extend_from_slice(&pi);

    let mut mul_quads = Vec::with_capacity(query_samples.len());
    for &(idx, lambda) in query_samples {
        let (qs, _pred) = flpcp.queries_for_coins_sparse(idx, lambda, x)?;
        if qs.len() != 3 {
            return Err(format!(
                "germ_field: expected 3 sparse queries, got {}",
                qs.len()
            ));
        }
        let alpha = qs[0].dot(&v0);
        let beta = qs[1].dot(&v0);
        let gamma = qs[2].dot(&v0);
        mul_quads.push(GermFieldMulQuad {
            a: alpha,
            b: beta,
            c: gamma,
            d: F::ONE,
        });
    }

    Ok(GermFieldResidualInstance {
        linear_residuals: Vec::new(),
        mul_quads,
    })
}

fn linear_fingerprints<F: PrimeField>(seed: &[u8; 32], residuals: &[F], count: usize) -> Vec<F> {
    let mut out = Vec::with_capacity(count);
    for fp_idx in 0..count {
        let mut acc = F::ZERO;
        for (term_idx, residual) in residuals.iter().enumerate() {
            acc += derive_linear_weight::<F>(seed, fp_idx, term_idx) * *residual;
        }
        out.push(acc);
    }
    out
}

pub fn prove_field_germ<F: PrimeField>(
    seed: [u8; 32],
    linear_residuals: &[F],
    mul_quads: &[GermFieldMulQuad<F>],
) -> Result<GermFieldProof<F>, String> {
    let linear_fingerprints = linear_fingerprints::<F>(&seed, linear_residuals, 2);
    let mul_sumcheck = prove_mul_sumcheck::<F>(&seed, mul_quads)?;
    Ok(GermFieldProof {
        linear_fingerprints,
        mul_sumcheck,
    })
}

pub fn verify_field_germ<F: PrimeField>(
    seed: [u8; 32],
    linear_residuals: &[F],
    mul_quads: &[GermFieldMulQuad<F>],
    proof: &GermFieldProof<F>,
) -> Result<(), String> {
    let expected_linear = linear_fingerprints::<F>(&seed, linear_residuals, proof.linear_fingerprints.len());
    if expected_linear != proof.linear_fingerprints {
        return Err("germ_field: linear fingerprint mismatch".to_string());
    }
    for (idx, fp) in proof.linear_fingerprints.iter().enumerate() {
        if !fp.is_zero() {
            return Err(format!("germ_field: nonzero linear fingerprint at index {}", idx));
        }
    }
    // Structural check that the opening still matches the supplied quads family.
    let expected_sumcheck = prove_mul_sumcheck::<F>(&seed, mul_quads)?;
    if expected_sumcheck.nvars != proof.mul_sumcheck.nvars
        || expected_sumcheck.rounds.len() != proof.mul_sumcheck.rounds.len()
        || expected_sumcheck
            .rounds
            .iter()
            .zip(proof.mul_sumcheck.rounds.iter())
            .any(|(a, b)| a.evaluations != b.evaluations)
        || expected_sumcheck.opening.a != proof.mul_sumcheck.opening.a
        || expected_sumcheck.opening.b != proof.mul_sumcheck.opening.b
        || expected_sumcheck.opening.c != proof.mul_sumcheck.opening.c
        || expected_sumcheck.opening.d != proof.mul_sumcheck.opening.d
    {
        return Err("germ_field: multiplicative sumcheck transcript mismatch".to_string());
    }
    verify_mul_sumcheck::<F>(&seed, &proof.mul_sumcheck)
}

#[cfg(test)]
mod tests {
    use ark_ff::{Field, Fp64, MontBackend, MontConfig};
    use dpp::dr1cs_flpcp::{
        Dr1csInstanceSparse, MulCode, MulCodeDr1csNpFlpcpSparse, TensorRsMulCode,
    };
    use dpp::SparseVec;

    use super::*;

    #[derive(MontConfig)]
    #[modulus = "18446744069414584321"]
    #[generator = "7"]
    struct GoldilocksConfig;
    type Goldilocks = Fp64<MontBackend<GoldilocksConfig, 1>>;

    fn tiny_goldilocks_flpcp(
    ) -> MulCodeDr1csNpFlpcpSparse<Goldilocks, TensorRsMulCode<Goldilocks>> {
        let n_total = 3usize;
        let l_public = 1usize;
        let code = TensorRsMulCode::<Goldilocks>::new(1, 3).expect("tensor code");
        let k = code.dim_k();

        let mut a = vec![SparseVec::new(vec![(Goldilocks::ONE, 0)])];
        let mut b = vec![SparseVec::new(vec![(Goldilocks::ONE, 1)])];
        let mut c = vec![SparseVec::new(vec![(Goldilocks::ONE, 2)])];
        while a.len() < k {
            a.push(SparseVec::new(Vec::new()));
            b.push(SparseVec::new(Vec::new()));
            c.push(SparseVec::new(Vec::new()));
        }
        let inst = Dr1csInstanceSparse::<Goldilocks> { n: n_total, a, b, c };
        MulCodeDr1csNpFlpcpSparse::<Goldilocks, _>::new(inst, l_public, code)
            .expect("mulcode flpcp")
    }

    fn medium_goldilocks_flpcp(
    ) -> MulCodeDr1csNpFlpcpSparse<Goldilocks, TensorRsMulCode<Goldilocks>> {
        let n_total = 3usize;
        let l_public = 1usize;
        let code = TensorRsMulCode::<Goldilocks>::new(2, 3).expect("tensor code");
        let k = code.dim_k();

        let mut a = vec![SparseVec::new(vec![(Goldilocks::ONE, 0)])];
        let mut b = vec![SparseVec::new(vec![(Goldilocks::ONE, 1)])];
        let mut c = vec![SparseVec::new(vec![(Goldilocks::ONE, 2)])];
        while a.len() < k {
            a.push(SparseVec::new(Vec::new()));
            b.push(SparseVec::new(Vec::new()));
            c.push(SparseVec::new(Vec::new()));
        }
        let inst = Dr1csInstanceSparse::<Goldilocks> { n: n_total, a, b, c };
        MulCodeDr1csNpFlpcpSparse::<Goldilocks, _>::new(inst, l_public, code)
            .expect("mulcode flpcp")
    }

    #[test]
    fn test_goldilocks_sparse_dr1cs_germ_direct() {
        let inst = Dr1csInstanceSparse::<Goldilocks> {
            n: 5,
            a: vec![SparseVec::new(vec![(Goldilocks::ONE, 1)])],
            b: vec![SparseVec::new(vec![(Goldilocks::ONE, 2)])],
            c: vec![SparseVec::new(vec![(Goldilocks::ONE, 3)])],
        };
        // z = [1, 2, 3, 6, 5]
        let witness = vec![
            Goldilocks::ONE,
            Goldilocks::from(2u64),
            Goldilocks::from(3u64),
            Goldilocks::from(6u64),
            Goldilocks::from(5u64),
        ];
        let linear_rows = vec![GermFieldLinearRow {
            constant: Goldilocks::ZERO,
            terms: vec![
                (1, Goldilocks::ONE),
                (2, Goldilocks::ONE),
                (4, -Goldilocks::ONE),
            ],
        }];
        let linear_residuals =
            eval_linear_rows(linear_rows.as_slice(), witness.as_slice()).expect("eval_linear_rows");
        let residuals = export_residual_instance_from_sparse_dr1cs(&inst, &witness)
            .expect("export_residual_instance_from_sparse_dr1cs");
        let seed = [29u8; 32];
        let proof = prove_field_germ(
            seed,
            linear_residuals.as_slice(),
            residuals.mul_quads.as_slice(),
        )
        .expect("prove_field_germ");
        verify_field_germ(
            seed,
            linear_residuals.as_slice(),
            residuals.mul_quads.as_slice(),
            &proof,
        )
        .expect("verify_field_germ");

        let mut bad_linear = linear_residuals.clone();
        bad_linear[0] += Goldilocks::ONE;
        let bad = verify_field_germ(
            seed,
            bad_linear.as_slice(),
            residuals.mul_quads.as_slice(),
            &proof,
        );
        assert!(bad.is_err(), "tampered linear residuals must fail");
    }

    #[test]
    fn test_goldilocks_dpp_germ_smoke() {
        let flpcp = tiny_goldilocks_flpcp();
        let x = vec![Goldilocks::from(2u64)];
        let z_w = vec![Goldilocks::from(3u64), Goldilocks::from(6u64)];
        let query_samples = [
            (0usize, Goldilocks::from(7u64)),
            (1usize, Goldilocks::from(11u64)),
            (2usize, Goldilocks::from(13u64)),
            (3usize, Goldilocks::from(17u64)),
        ];
        let residuals = export_residual_instance_from_flpcp(
            &flpcp,
            &x,
            &z_w,
            &query_samples,
        )
        .expect("export_residual_instance_from_flpcp");
        assert!(
            residuals.linear_residuals.is_empty(),
            "direct DPP->GERM path intentionally excludes legacy side-cube linear constraints"
        );

        let seed = [9u8; 32];
        let proof = prove_field_germ(
            seed,
            residuals.linear_residuals.as_slice(),
            residuals.mul_quads.as_slice(),
        )
            .expect("prove_field_germ");
        verify_field_germ(
            seed,
            residuals.linear_residuals.as_slice(),
            residuals.mul_quads.as_slice(),
            &proof,
        )
            .expect("verify_field_germ");

        let mut bad_quads = residuals.mul_quads.clone();
        bad_quads[0].c += Goldilocks::ONE;
        let bad = verify_field_germ(
            seed,
            residuals.linear_residuals.as_slice(),
            bad_quads.as_slice(),
            &proof,
        );
        assert!(bad.is_err(), "tampered multiplicative quads must fail");
    }

    #[test]
    fn test_goldilocks_dpp_germ_smoke_medium_without_linear_side() {
        let flpcp = medium_goldilocks_flpcp();
        let x = vec![Goldilocks::from(2u64)];
        let z_w = vec![Goldilocks::from(3u64), Goldilocks::from(6u64)];
        let query_samples = [
            (0usize, Goldilocks::from(7u64)),
            (1usize, Goldilocks::from(11u64)),
            (2usize, Goldilocks::from(13u64)),
            (3usize, Goldilocks::from(17u64)),
            (4usize, Goldilocks::from(19u64)),
            (5usize, Goldilocks::from(23u64)),
            (6usize, Goldilocks::from(29u64)),
            (7usize, Goldilocks::from(31u64)),
        ];
        let residuals = export_residual_instance_from_flpcp(
            &flpcp,
            &x,
            &z_w,
            &query_samples,
        )
        .expect("export_residual_instance_from_flpcp");
        assert!(
            residuals.linear_residuals.is_empty(),
            "this medium smoke intentionally tests only the multiplicative GERM path"
        );
        let seed = [19u8; 32];
        let proof = prove_field_germ(
            seed,
            residuals.linear_residuals.as_slice(),
            residuals.mul_quads.as_slice(),
        )
        .expect("prove_field_germ");
        verify_field_germ(
            seed,
            residuals.linear_residuals.as_slice(),
            residuals.mul_quads.as_slice(),
            &proof,
        )
        .expect("verify_field_germ");
    }

}
