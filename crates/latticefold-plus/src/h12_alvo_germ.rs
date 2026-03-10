//! Global Extension-field Residual Mix (GERM) payload helpers for H12 ALVO.
//!
//! The canonical H12 proof is a compact verifier-facing GERM payload:
//! - local opened witness support needed by ALVO's anchor checks
//! - a small designated-challenge verifier payload for the global linear/multiplicative families
//!
//! Exact logical rows and per-repetition multiplicative quads are prover-time inputs used to derive the
//! compact payload; they are not stored in the canonical proof object.

use std::collections::{BTreeMap, BTreeSet};

use ark_ff::{Field, PrimeField};
use ark_std::Zero;
use cyclotomic_rings::rings::GoldilocksRing64 as AjtaiRing;
use dpp::theorem43::Theorem43AlvoLocalCheckSurface;
use latticefold::{commitment::AjtaiCommitmentScheme, transcript::poseidon::F257};
use rand::RngCore;
use sha2::Digest;
use stark_rings::PolyRing;

use crate::{
    aadp_we::{AadpConstraintSystem, AadpLinearForm, AadpMulConstraint},
    f257_ext16::{F257Ext16, F257_EXT16_DEGREE},
    h12_pi_commit::{block_len_for_index, f257_to_u16, H12PiBlockOpening},
};

pub const H12_GERM_COMMIT_ROWS: usize = 16;
pub const H12_GERM_BLIND_LEN: usize = 16;
pub const H12_GERM_EXT16_PROJECTION_CHECKS: usize = 4;
pub const H12_GERM_DESIGNATED_CHALLENGE_WORDS: usize = F257_EXT16_DEGREE;
pub const H12_GERM_LIN_FINGERPRINTS: usize = 2;
pub const H12_GERM_MUL_FINGERPRINTS: usize = 2;
const H12_GERM_AJTAI_DOMAIN: &[u8] = b"lfp_h12_germ_opened_view";
const GOLDILOCKS_P_U64: u64 = 0xFFFF_FFFF_0000_0001u64;

const H12_GERM_LIN_FINGERPRINT_COORDS: usize = H12_GERM_LIN_FINGERPRINTS * F257_EXT16_DEGREE;
const H12_GERM_EXT16_COORDS: usize = F257_EXT16_DEGREE;

#[derive(Clone, Debug)]
pub struct H12GermParams {
    pub pack_d: u16,
    pub commit_rows: u16,
    pub blind_len: u16,
}

#[derive(Clone, Debug)]
pub struct H12GermMulSumcheckRound {
    pub evaluations: [[u16; F257_EXT16_DEGREE]; 4],
}

#[derive(Clone, Debug)]
pub struct H12GermMulOpening {
    pub a_eval: [u16; F257_EXT16_DEGREE],
    pub b_eval: [u16; F257_EXT16_DEGREE],
    pub c_eval: [u16; F257_EXT16_DEGREE],
    pub d_eval: [u16; F257_EXT16_DEGREE],
}

#[derive(Clone, Debug)]
pub struct H12GermMulSumcheckProof {
    pub nvars: u16,
    pub rounds: Vec<H12GermMulSumcheckRound>,
    pub opening: H12GermMulOpening,
}

#[derive(Clone, Debug)]
pub struct H12GermVerifierPayload {
    pub global_err_packed_values: Vec<u16>,
    pub designated_challenge: Vec<u16>,
    pub opening_projection_residuals: Vec<u16>,
    pub h_projection_residuals: Vec<u16>,
    pub germ_linear_fingerprints: Vec<[u16; F257_EXT16_DEGREE]>,
    pub germ_mul_sumcheck: H12GermMulSumcheckProof,
}

#[derive(Clone, Debug)]
pub struct H12GermMulQuad {
    pub a: u16,
    pub b: u16,
    pub c: u16,
    pub d: u16,
}

#[derive(Clone, Debug)]
pub struct H12GermProof {
    pub params: H12GermParams,
    pub local_view_values: Vec<u16>,
    pub blind_values: Vec<u16>,
    pub ajtai_commitment_rows: Vec<Vec<u64>>,
    pub verifier_payload: H12GermVerifierPayload,
}

#[derive(Clone, Debug)]
pub struct H12GermCompiledConstraintSystem {
    pub z_w_positions: Vec<usize>,
    pub logical_block_indices: Vec<usize>,
    pub logical_block_positions: Vec<Vec<usize>>,
    pub logical_blocks_total: usize,
    pub z_w_len: usize,
    pub k_star: usize,
    pub low_cube_len: usize,
    pub local_view_len: usize,
    pub global_err_packed_len: usize,
    pub cs: AadpConstraintSystem<F257>,
    pub global_err_packed_offset: usize,
    pub blind_offset: usize,
    pub opening_projection_offset: usize,
    pub h_projection_offset: usize,
    pub germ_linear_fingerprint_offset: usize,
    pub germ_mul_sumcheck_round_offset: usize,
    pub germ_mul_sumcheck_final_offset: usize,
    pub germ_mul_sumcheck_rounds: usize,
    pub params: H12GermParams,
    pub matrix_seed: [u8; 32],
}

fn derive_germ_ajtai_seed(stage1_root: &[u8; 32]) -> [u8; 32] {
    let mut h = sha2::Sha256::new();
    h.update(b"LFP_H12_GERM_AJTAI_V1");
    h.update(stage1_root);
    h.finalize().into()
}

fn germ_commitment_root(rows: &[Vec<u64>]) -> [u8; 32] {
    let mut h = sha2::Sha256::new();
    h.update(b"LFP_H12_GERM_COMMIT_ROOT_V1");
    h.update((rows.len() as u32).to_le_bytes());
    for row in rows {
        h.update((row.len() as u32).to_le_bytes());
        for &v in row {
            h.update(v.to_le_bytes());
        }
    }
    h.finalize().into()
}

fn derive_germ_designated_challenge(
    matrix_seed: &[u8; 32],
    commitment_root: &[u8; 32],
) -> Vec<u16> {
    let mut out = Vec::with_capacity(H12_GERM_DESIGNATED_CHALLENGE_WORDS);
    for idx in 0..H12_GERM_DESIGNATED_CHALLENGE_WORDS {
        let mut h = sha2::Sha256::new();
        h.update(b"LFP_H12_GERM_DESIGNATED_CHALLENGE_EXT16_V1");
        h.update(matrix_seed);
        h.update(commitment_root);
        h.update((idx as u32).to_le_bytes());
        let bytes: [u8; 32] = h.finalize().into();
        out.push(u16::from_le_bytes([bytes[0], bytes[1]]) % 257);
    }
    out
}

fn ext16_projection_coeffs(
    designated_challenge: &[u16],
    domain: &[u8],
    proj_idx: usize,
    chunk_idx: usize,
) -> [u16; F257_EXT16_DEGREE] {
    let mut h = sha2::Sha256::new();
    h.update(domain);
    h.update((designated_challenge.len() as u32).to_le_bytes());
    for &w in designated_challenge {
        h.update(w.to_le_bytes());
    }
    h.update((proj_idx as u32).to_le_bytes());
    h.update((chunk_idx as u32).to_le_bytes());
    let bytes: [u8; 32] = h.finalize().into();
    let mut out = [0u16; F257_EXT16_DEGREE];
    let mut all_zero = true;
    for i in 0..F257_EXT16_DEGREE {
        let v = u16::from_le_bytes([bytes[2 * i], bytes[2 * i + 1]]) % 257;
        out[i] = v;
        if v != 0 {
            all_zero = false;
        }
    }
    if all_zero {
        out[0] = 1;
    }
    out
}

fn ext16_project_coords_with_count(
    designated_challenge: &[u16],
    domain: &[u8],
    coords: &[u16],
    projection_checks: usize,
) -> Vec<u16> {
    let mut out = Vec::with_capacity(projection_checks);
    for proj_idx in 0..projection_checks {
        let mut acc = F257::ZERO;
        for (chunk_idx, chunk) in coords.chunks(F257_EXT16_DEGREE).enumerate() {
            let coeffs = ext16_projection_coeffs(designated_challenge, domain, proj_idx, chunk_idx);
            for (i, &v) in chunk.iter().enumerate() {
                let vv = F257::from((v % 257) as u64);
                let ww = F257::from((coeffs[i] % 257) as u64);
                acc += vv * ww;
            }
        }
        out.push(f257_to_u16(acc));
    }
    out
}

fn ext16_fingerprint_weight(
    designated_challenge: &[u16],
    domain: &[u8],
    fp_idx: usize,
    term_idx: usize,
) -> F257Ext16 {
    let mut h = sha2::Sha256::new();
    h.update(domain);
    h.update((designated_challenge.len() as u32).to_le_bytes());
    for &w in designated_challenge {
        h.update(w.to_le_bytes());
    }
    h.update((fp_idx as u32).to_le_bytes());
    h.update((term_idx as u32).to_le_bytes());
    let bytes: [u8; 32] = h.finalize().into();
    let mut coeffs = [0u16; F257_EXT16_DEGREE];
    let mut all_zero = true;
    for i in 0..F257_EXT16_DEGREE {
        let v = u16::from_le_bytes([bytes[2 * i], bytes[2 * i + 1]]) % 257;
        coeffs[i] = v;
        if v != 0 {
            all_zero = false;
        }
    }
    if all_zero {
        coeffs[0] = 1;
    }
    F257Ext16 { coeffs }
}

fn ext16_mix_coords_to_fingerprints(
    designated_challenge: &[u16],
    domain: &[u8],
    coords: &[u16],
    fingerprints: usize,
) -> Vec<[u16; F257_EXT16_DEGREE]> {
    let mut out = Vec::with_capacity(fingerprints);
    for fp_idx in 0..fingerprints {
        let mut acc = F257Ext16::zero();
        for (term_idx, &coord) in coords.iter().enumerate() {
            let w = ext16_fingerprint_weight(designated_challenge, domain, fp_idx, term_idx);
            acc += w * F257Ext16::from_f257(coord % 257);
        }
        out.push(acc.coeffs);
    }
    out
}

fn flatten_ext16_fingerprints(fingerprints: &[[u16; F257_EXT16_DEGREE]]) -> Vec<u16> {
    let mut out = Vec::with_capacity(fingerprints.len() * F257_EXT16_DEGREE);
    for fp in fingerprints {
        out.extend(fp.iter().copied());
    }
    out
}

fn ext16_to_u16s(x: F257Ext16) -> [u16; F257_EXT16_DEGREE] {
    x.coeffs
}

fn ext16_from_u16s(coeffs: [u16; F257_EXT16_DEGREE]) -> F257Ext16 {
    F257Ext16 { coeffs }
}

fn mul_sumcheck_nvars(total_checks: usize) -> usize {
    if total_checks <= 1 {
        0
    } else {
        total_checks.next_power_of_two().trailing_zeros() as usize
    }
}

fn derive_germ_mul_point(
    designated_challenge: &[u16],
    nvars: usize,
) -> Result<Vec<F257Ext16>, String> {
    if designated_challenge.len() != H12_GERM_DESIGNATED_CHALLENGE_WORDS {
        return Err(format!(
            "H12 GERM mul point seed length mismatch: got={} expected={}",
            designated_challenge.len(),
            H12_GERM_DESIGNATED_CHALLENGE_WORDS
        ));
    }
    let mut out = Vec::with_capacity(nvars);
    for var_idx in 0..nvars {
        let mut h = sha2::Sha256::new();
        h.update(b"LFP_H12_GERM_MUL_POINT_V1");
        for &w in designated_challenge {
            h.update(w.to_le_bytes());
        }
        h.update((var_idx as u32).to_le_bytes());
        let bytes: [u8; 32] = h.finalize().into();
        let mut coeffs = [0u16; F257_EXT16_DEGREE];
        for i in 0..F257_EXT16_DEGREE {
            coeffs[i] = u16::from_le_bytes([bytes[2 * i], bytes[2 * i + 1]]) % 257;
        }
        out.push(F257Ext16 { coeffs });
    }
    Ok(out)
}

fn absorb_ext16_words(h: &mut sha2::Sha256, x: &F257Ext16) {
    for &c in &x.coeffs {
        h.update(c.to_le_bytes());
    }
}

fn derive_germ_mul_sumcheck_round_challenge(
    designated_challenge: &[u16],
    rounds: &[H12GermMulSumcheckRound],
    round_idx: usize,
) -> Result<F257Ext16, String> {
    if designated_challenge.len() != H12_GERM_DESIGNATED_CHALLENGE_WORDS {
        return Err(format!(
            "H12 GERM sumcheck seed length mismatch: got={} expected={}",
            designated_challenge.len(),
            H12_GERM_DESIGNATED_CHALLENGE_WORDS
        ));
    }
    let mut h = sha2::Sha256::new();
    h.update(b"LFP_H12_GERM_MUL_SC_CHAL_V1");
    for &w in designated_challenge {
        h.update(w.to_le_bytes());
    }
    h.update((round_idx as u32).to_le_bytes());
    for round in rounds {
        for eval in &round.evaluations {
            for &c in eval {
                h.update(c.to_le_bytes());
            }
        }
    }
    let bytes: [u8; 32] = h.finalize().into();
    let mut coeffs = [0u16; F257_EXT16_DEGREE];
    for i in 0..F257_EXT16_DEGREE {
        coeffs[i] = u16::from_le_bytes([bytes[2 * i], bytes[2 * i + 1]]) % 257;
    }
    Ok(F257Ext16 { coeffs })
}

fn ext16_interpolate_0123(evals: &[F257Ext16; 4], x: F257Ext16) -> Result<F257Ext16, String> {
    let xs = [
        F257Ext16::from_f257(0),
        F257Ext16::from_f257(1),
        F257Ext16::from_f257(2),
        F257Ext16::from_f257(3),
    ];
    let mut acc = F257Ext16::zero();
    for i in 0..4 {
        let mut num = F257Ext16::one();
        let mut den = F257Ext16::one();
        for j in 0..4 {
            if i == j {
                continue;
            }
            num *= x - xs[j];
            den *= xs[i] - xs[j];
        }
        let den_inv = den
            .inverse()
            .ok_or_else(|| "H12 GERM sumcheck interpolation denominator was zero".to_string())?;
        acc += evals[i] * (num * den_inv);
    }
    Ok(acc)
}

fn build_mul_quad_tables(
    per_rep_mul_quads: &[Vec<H12GermMulQuad>],
) -> Result<
    (
        usize,
        Vec<F257Ext16>,
        Vec<F257Ext16>,
        Vec<F257Ext16>,
        Vec<F257Ext16>,
    ),
    String,
> {
    if per_rep_mul_quads.is_empty() {
        return Ok((0, Vec::new(), Vec::new(), Vec::new(), Vec::new()));
    }
    let width = per_rep_mul_quads[0].len();
    for (rep_idx, quads) in per_rep_mul_quads.iter().enumerate() {
        if quads.len() != width {
            return Err(format!(
                "H12 GERM mul table width mismatch at rep {}: got={} expected={}",
                rep_idx,
                quads.len(),
                width
            ));
        }
    }
    let total_checks = per_rep_mul_quads.len().saturating_mul(width);
    let padded = total_checks.max(1).next_power_of_two();
    let mut a = vec![F257Ext16::zero(); padded];
    let mut b = vec![F257Ext16::zero(); padded];
    let mut c = vec![F257Ext16::zero(); padded];
    let mut d = vec![F257Ext16::zero(); padded];
    let mut idx = 0usize;
    for quads in per_rep_mul_quads {
        for quad in quads {
            a[idx] = F257Ext16::from_f257(quad.a % 257);
            b[idx] = F257Ext16::from_f257(quad.b % 257);
            c[idx] = F257Ext16::from_f257(quad.c % 257);
            d[idx] = F257Ext16::from_f257(quad.d % 257);
            idx += 1;
        }
    }
    Ok((total_checks, a, b, c, d))
}

fn build_eq_table(point: &[F257Ext16]) -> Vec<F257Ext16> {
    if point.is_empty() {
        return vec![F257Ext16::one()];
    }
    let nvars = point.len();
    let size = 1usize << nvars;
    let mut out = vec![F257Ext16::zero(); size];
    for (idx, slot) in out.iter_mut().enumerate() {
        let mut acc = F257Ext16::one();
        for (var_idx, r_i) in point.iter().enumerate() {
            let bit = (idx >> var_idx) & 1;
            acc *= if bit == 0 {
                F257Ext16::one() - *r_i
            } else {
                *r_i
            };
        }
        *slot = acc;
    }
    out
}

fn fold_mle_table(table: &[F257Ext16], r: F257Ext16) -> Vec<F257Ext16> {
    let mut out = Vec::with_capacity(table.len() / 2);
    for pair in table.chunks_exact(2) {
        let v0 = pair[0];
        let v1 = pair[1];
        out.push(v0 + (v1 - v0) * r);
    }
    out
}

fn prove_germ_mul_sumcheck_from_mul_quads(
    designated_challenge: &[u16],
    per_rep_mul_quads: &[Vec<H12GermMulQuad>],
) -> Result<H12GermMulSumcheckProof, String> {
    let (total_checks, mut a, mut b, mut c, mut d) = build_mul_quad_tables(per_rep_mul_quads)?;
    let nvars = mul_sumcheck_nvars(total_checks);
    if nvars == 0 {
        return Ok(H12GermMulSumcheckProof {
            nvars: 0,
            rounds: Vec::new(),
            opening: H12GermMulOpening {
                a_eval: ext16_to_u16s(a.first().copied().unwrap_or_else(F257Ext16::zero)),
                b_eval: ext16_to_u16s(b.first().copied().unwrap_or_else(F257Ext16::zero)),
                c_eval: ext16_to_u16s(c.first().copied().unwrap_or_else(F257Ext16::zero)),
                d_eval: ext16_to_u16s(d.first().copied().unwrap_or_else(F257Ext16::zero)),
            },
        });
    }
    let r_point = derive_germ_mul_point(designated_challenge, nvars)?;
    let mut eq = build_eq_table(r_point.as_slice());
    let mut rounds = Vec::with_capacity(nvars);
    for round_idx in 0..nvars {
        let mut evals = [F257Ext16::zero(); 4];
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
            for (t_idx, t_u16) in [0u16, 1, 2, 3].iter().copied().enumerate() {
                let t = F257Ext16::from_f257(t_u16);
                let at = a0 + (a1 - a0) * t;
                let bt = b0 + (b1 - b0) * t;
                let ct = c0 + (c1 - c0) * t;
                let dt = d0 + (d1 - d0) * t;
                let et = e0 + (e1 - e0) * t;
                evals[t_idx] += et * ((ct * dt) - (at * bt));
            }
        }
        let round = H12GermMulSumcheckRound {
            evaluations: [
                ext16_to_u16s(evals[0]),
                ext16_to_u16s(evals[1]),
                ext16_to_u16s(evals[2]),
                ext16_to_u16s(evals[3]),
            ],
        };
        rounds.push(round);
        let r_sc = derive_germ_mul_sumcheck_round_challenge(
            designated_challenge,
            rounds.as_slice(),
            round_idx,
        )?;
        a = fold_mle_table(a.as_slice(), r_sc);
        b = fold_mle_table(b.as_slice(), r_sc);
        c = fold_mle_table(c.as_slice(), r_sc);
        d = fold_mle_table(d.as_slice(), r_sc);
        eq = fold_mle_table(eq.as_slice(), r_sc);
    }
    Ok(H12GermMulSumcheckProof {
        nvars: nvars as u16,
        rounds,
        opening: H12GermMulOpening {
            a_eval: ext16_to_u16s(a[0]),
            b_eval: ext16_to_u16s(b[0]),
            c_eval: ext16_to_u16s(c[0]),
            d_eval: ext16_to_u16s(d[0]),
        },
    })
}

fn derive_mul_sumcheck_residuals_from_payload(
    designated_challenge: &[u16],
    proof: &H12GermMulSumcheckProof,
) -> Result<(Vec<u16>, Vec<u16>), String> {
    let nvars = proof.nvars as usize;
    if proof.rounds.len() != nvars {
        return Err(format!(
            "H12 GERM mul sumcheck round count mismatch: got={} expected={}",
            proof.rounds.len(),
            nvars
        ));
    }
    if nvars == 0 {
        return Ok((Vec::new(), Vec::new()));
    }
    let r_point = derive_germ_mul_point(designated_challenge, nvars)?;
    let mut claimed = F257Ext16::zero();
    let mut round_residuals = Vec::with_capacity(nvars * H12_GERM_EXT16_COORDS);
    let mut sampled = Vec::with_capacity(nvars);
    for (round_idx, round) in proof.rounds.iter().enumerate() {
        let evals = [
            ext16_from_u16s(round.evaluations[0]),
            ext16_from_u16s(round.evaluations[1]),
            ext16_from_u16s(round.evaluations[2]),
            ext16_from_u16s(round.evaluations[3]),
        ];
        let residual = evals[0] + evals[1] - claimed;
        round_residuals.extend(ext16_to_u16s(residual));
        let r_sc = derive_germ_mul_sumcheck_round_challenge(
            designated_challenge,
            &proof.rounds[..=round_idx],
            round_idx,
        )?;
        sampled.push(r_sc);
        claimed = ext16_interpolate_0123(&evals, r_sc)?;
    }
    let a_eval = ext16_from_u16s(proof.opening.a_eval);
    let b_eval = ext16_from_u16s(proof.opening.b_eval);
    let c_eval = ext16_from_u16s(proof.opening.c_eval);
    let d_eval = ext16_from_u16s(proof.opening.d_eval);
    let mut eq_eval = F257Ext16::one();
    for (r_i, s_i) in r_point.iter().zip(sampled.iter()) {
        let term = (F257Ext16::one() - *r_i) * (F257Ext16::one() - *s_i) + (*r_i * *s_i);
        eq_eval *= term;
    }
    let final_residual = claimed - (eq_eval * ((c_eval * d_eval) - (a_eval * b_eval)));
    Ok((round_residuals, ext16_to_u16s(final_residual).to_vec()))
}

#[inline]
fn mul_quad_residual(quad: &H12GermMulQuad) -> u16 {
    crate::lockable_ringlwe::sub_mod257_u16(
        crate::lockable_ringlwe::mul_mod257_u16(quad.c % 257, quad.d % 257),
        crate::lockable_ringlwe::mul_mod257_u16(quad.a % 257, quad.b % 257),
    )
}

fn mul_residual_coords_from_per_rep_mul_quads(
    per_rep_mul_quads: &[Vec<H12GermMulQuad>],
) -> Result<Vec<u16>, String> {
    if per_rep_mul_quads.is_empty() {
        return Ok(Vec::new());
    }
    let width = per_rep_mul_quads[0].len();
    for (rep_idx, quads) in per_rep_mul_quads.iter().enumerate() {
        if quads.len() != width {
            return Err(format!(
                "H12 GERM per-rep multiplicative width mismatch at rep {}: got={} expected={}",
                rep_idx,
                quads.len(),
                width
            ));
        }
    }
    let mut out = Vec::with_capacity(width.saturating_mul(per_rep_mul_quads.len()));
    for quads in per_rep_mul_quads {
        out.extend(quads.iter().map(mul_quad_residual));
    }
    Ok(out)
}

fn global_err_rep_mix_coeff(matrix_seed: &[u8; 32], rep_idx: usize) -> u16 {
    let mut h = sha2::Sha256::new();
    h.update(b"LFP_H12_GERM_GLOBAL_ERR_REP_COEFF_V1");
    h.update(matrix_seed);
    h.update((rep_idx as u32).to_le_bytes());
    let bytes: [u8; 32] = h.finalize().into();
    let mut coeff = u16::from_le_bytes([bytes[0], bytes[1]]) % 257;
    if coeff == 0 {
        coeff = 1;
    }
    coeff
}

fn global_err_chunk_coeff(matrix_seed: &[u8; 32], chunk_idx: usize, rel_pos: usize) -> u16 {
    let mut h = sha2::Sha256::new();
    h.update(b"LFP_H12_GERM_GLOBAL_ERR_CHUNK_COEFF_V1");
    h.update(matrix_seed);
    h.update((chunk_idx as u32).to_le_bytes());
    h.update((rel_pos as u32).to_le_bytes());
    let bytes: [u8; 32] = h.finalize().into();
    let mut coeff = u16::from_le_bytes([bytes[0], bytes[1]]) % 257;
    if coeff == 0 {
        coeff = 1;
    }
    coeff
}

fn global_err_packed_from_per_rep_mul_quads(
    matrix_seed: &[u8; 32],
    per_rep_mul_quads: &[Vec<H12GermMulQuad>],
    logical_blocks_total: usize,
    pack_d: usize,
    packed_len: usize,
) -> Result<Vec<u16>, String> {
    let pack_d = pack_d.max(1);
    if logical_blocks_total == 0 {
        if packed_len != 0 {
            return Err(format!(
                "H12 GERM global-err packed length mismatch: blocks=0 packed_len={}",
                packed_len
            ));
        }
        return Ok(Vec::new());
    }
    if per_rep_mul_quads.is_empty() {
        return Err("H12 GERM missing per-rep multiplicative rows for global residual layer".to_string());
    }
    let expected_packed = (logical_blocks_total + pack_d - 1) / pack_d;
    if packed_len != expected_packed {
        return Err(format!(
            "H12 GERM global-err packed length mismatch: got={} expected={}",
            packed_len, expected_packed
        ));
    }
    let mut mixed_by_block = vec![0u16; logical_blocks_total];
    for (rep_idx, quads) in per_rep_mul_quads.iter().enumerate() {
        if quads.len() != logical_blocks_total {
            return Err(format!(
                "H12 GERM per-rep global multiplicative length mismatch at rep {}: got={} expected={}",
                rep_idx,
                quads.len(),
                logical_blocks_total
            ));
        }
        let rep_coeff = global_err_rep_mix_coeff(matrix_seed, rep_idx);
        for (block_idx, quad) in quads.iter().enumerate() {
            let err = mul_quad_residual(quad);
            mixed_by_block[block_idx] = crate::lockable_ringlwe::add_mod257_u16(
                mixed_by_block[block_idx],
                crate::lockable_ringlwe::mul_mod257_u16(rep_coeff, err),
            );
        }
    }
    let mut out = Vec::with_capacity(packed_len);
    for chunk_idx in 0..packed_len {
        let start = chunk_idx * pack_d;
        let end = (start + pack_d).min(logical_blocks_total);
        let mut acc = 0u16;
        for block_idx in start..end {
            let coeff = global_err_chunk_coeff(matrix_seed, chunk_idx, block_idx - start);
            acc = crate::lockable_ringlwe::add_mod257_u16(
                acc,
                crate::lockable_ringlwe::mul_mod257_u16(coeff, mixed_by_block[block_idx]),
            );
        }
        out.push(acc);
    }
    Ok(out)
}

fn opening_residual_coords(
    expected_rows: &[Vec<u64>],
    claimed_rows: &[Vec<u64>],
) -> Result<Vec<u16>, String> {
    if expected_rows.len() != claimed_rows.len() {
        return Err(format!(
            "H12 GERM Ajtai row count mismatch: expected={} claimed={}",
            expected_rows.len(),
            claimed_rows.len()
        ));
    }
    let mut coords = Vec::new();
    for (row_idx, (expected_row, claimed_row)) in
        expected_rows.iter().zip(claimed_rows.iter()).enumerate()
    {
        if expected_row.len() != claimed_row.len() {
            return Err(format!(
                "H12 GERM Ajtai row lane mismatch at row {row_idx}: expected={} claimed={}",
                expected_row.len(),
                claimed_row.len()
            ));
        }
        for (&expected_lane, &claimed_lane) in expected_row.iter().zip(claimed_row.iter()) {
            let expected_mod = expected_lane % GOLDILOCKS_P_U64;
            let claimed_mod = claimed_lane % GOLDILOCKS_P_U64;
            let delta = if expected_mod >= claimed_mod {
                expected_mod - claimed_mod
            } else {
                GOLDILOCKS_P_U64 - (claimed_mod - expected_mod)
            };
            for b in delta.to_le_bytes() {
                coords.push((b as u16) % 257);
            }
        }
    }
    Ok(coords)
}

fn opening_projection_residuals(
    designated_challenge: &[u16],
    expected_rows: &[Vec<u64>],
    claimed_rows: &[Vec<u64>],
) -> Result<Vec<u16>, String> {
    if designated_challenge.len() != H12_GERM_DESIGNATED_CHALLENGE_WORDS {
        return Err(format!(
            "H12 GERM designated-challenge length mismatch: got={} expected={}",
            designated_challenge.len(),
            H12_GERM_DESIGNATED_CHALLENGE_WORDS
        ));
    }
    let coords = opening_residual_coords(expected_rows, claimed_rows)?;
    Ok(ext16_project_coords_with_count(
        designated_challenge,
        b"LFP_H12_GERM_OPENING_EXT16_PROJ_V1",
        coords.as_slice(),
        H12_GERM_EXT16_PROJECTION_CHECKS,
    ))
}

fn logical_rows_map_for_compiled<'a>(
    compiled: &H12GermCompiledConstraintSystem,
    logical_rows_in_compiled_order: &'a [Vec<u16>],
) -> Result<BTreeMap<usize, &'a [u16]>, String> {
    if logical_rows_in_compiled_order.len() != compiled.logical_block_indices.len() {
        return Err(format!(
            "H12 GERM logical-row payload count mismatch: got={} expected={}",
            logical_rows_in_compiled_order.len(),
            compiled.logical_block_indices.len()
        ));
    }
    let mut out = BTreeMap::<usize, &'a [u16]>::new();
    for (&logical_block, row) in compiled
        .logical_block_indices
        .iter()
        .zip(logical_rows_in_compiled_order.iter())
    {
        if row.len() < compiled.k_star {
            return Err(format!(
                "H12 GERM logical row too short: block={} got={} expected_at_least={}",
                logical_block,
                row.len(),
                compiled.k_star
            ));
        }
        out.insert(logical_block, row.as_slice());
    }
    Ok(out)
}

fn ensure_local_view_matches_logical_rows(
    compiled: &H12GermCompiledConstraintSystem,
    local_view: &[F257],
    logical_rows_by_block: &BTreeMap<usize, &[u16]>,
) -> Result<(), String> {
    if local_view.len() != compiled.local_view_len {
        return Err(format!(
            "H12 GERM local-view length mismatch: got={} expected={}",
            local_view.len(),
            compiled.local_view_len
        ));
    }
    let mut lv_offset = compiled.z_w_positions.len();
    for (&logical_block, positions) in compiled
        .logical_block_indices
        .iter()
        .zip(compiled.logical_block_positions.iter())
    {
        let row = logical_rows_by_block
            .get(&logical_block)
            .ok_or_else(|| format!("H12 GERM missing logical row for block {logical_block}"))?;
        for &pos in positions {
            let lv = *local_view.get(lv_offset).ok_or_else(|| {
                format!("H12 GERM local-view index out of range: idx={lv_offset}")
            })?;
            let got = f257_to_u16(lv);
            let expected = *row.get(pos).ok_or_else(|| {
                format!(
                    "H12 GERM logical row position out of range: block={} pos={} row_len={}",
                    logical_block,
                    pos,
                    row.len()
                )
            })? % 257;
            if got != expected {
                return Err(format!(
                    "H12 GERM logical-row payload mismatch at block={} pos={}: local_view={} payload={}",
                    logical_block, pos, got, expected
                ));
            }
            lv_offset += 1;
        }
    }
    if lv_offset != local_view.len() {
        return Err(format!(
            "H12 GERM local-view traversal mismatch: traversed={} len={}",
            lv_offset,
            local_view.len()
        ));
    }
    Ok(())
}

fn h_linear_residual_coords_from_surfaces_with_rows(
    surfaces: &[Theorem43AlvoLocalCheckSurface<F257>],
    logical_rows_by_block: &BTreeMap<usize, &[u16]>,
    global_err_packed_values: &[u16],
) -> Result<Vec<u16>, String> {
    let mut coords = Vec::new();
    for (surface_idx, surface) in surfaces.iter().enumerate() {
        let block_id = surface.local_check.block_id;
        let row = logical_rows_by_block.get(&block_id).ok_or_else(|| {
            format!(
                "H12 GERM missing logical row for H projection: surface_idx={} block_id={}",
                surface_idx, block_id
            )
        })?;
        for (h_idx, h_cons) in surface.h_w_eval_constraints.iter().enumerate() {
            let mut acc = h_cons.constant;
            for &(pos, coeff) in &h_cons.terms {
                let w_u16 = *row.get(pos).ok_or_else(|| {
                    format!(
                        "H12 GERM H_j row out of range: surface_idx={} constraint={} block_id={} pos={} row_len={}",
                        surface_idx,
                        h_idx,
                        block_id,
                        pos,
                        row.len()
                    )
                })?;
                acc += coeff * F257::from((w_u16 % 257) as u64);
            }
            coords.push(f257_to_u16(acc));
        }
    }
    // Fold global packed residual summary into the same 4-projection family
    // so we keep the original projection budget (4 opening + 4 H/global).
    coords.extend(global_err_packed_values.iter().map(|&x| x % 257));
    Ok(coords)
}

fn h_projection_residuals_from_surfaces_with_rows(
    designated_challenge: &[u16],
    surfaces: &[Theorem43AlvoLocalCheckSurface<F257>],
    logical_rows_by_block: &BTreeMap<usize, &[u16]>,
    global_err_packed_values: &[u16],
) -> Result<Vec<u16>, String> {
    if designated_challenge.len() != H12_GERM_DESIGNATED_CHALLENGE_WORDS {
        return Err(format!(
            "H12 GERM designated-challenge length mismatch: got={} expected={}",
            designated_challenge.len(),
            H12_GERM_DESIGNATED_CHALLENGE_WORDS
        ));
    }
    let coords = h_linear_residual_coords_from_surfaces_with_rows(
        surfaces,
        logical_rows_by_block,
        global_err_packed_values,
    )?;
    Ok(ext16_project_coords_with_count(
        designated_challenge,
        b"LFP_H12_GERM_H_PLUS_GLOBAL_EXT16_PROJ_V1",
        coords.as_slice(),
        H12_GERM_EXT16_PROJECTION_CHECKS,
    ))
}

fn germ_ajtai_scheme(matrix_seed: &[u8; 32], width: usize) -> AjtaiCommitmentScheme<AjtaiRing> {
    AjtaiCommitmentScheme::<AjtaiRing>::seeded(
        H12_GERM_AJTAI_DOMAIN,
        *matrix_seed,
        H12_GERM_COMMIT_ROWS,
        width,
    )
}

fn germ_zero_constraint(var_idx: usize) -> AadpMulConstraint<F257> {
    AadpMulConstraint {
        a: AadpLinearForm {
            constant: F257::ZERO,
            terms: vec![(var_idx, F257::ONE)],
        },
        b: AadpLinearForm {
            constant: F257::ONE,
            terms: Vec::new(),
        },
        c: AadpLinearForm {
            constant: F257::ZERO,
            terms: Vec::new(),
        },
        d: AadpLinearForm {
            constant: F257::ONE,
            terms: Vec::new(),
        },
    }
}

pub fn compile_germ_constraint_system(
    surfaces: &[Theorem43AlvoLocalCheckSurface<F257>],
    stage1_root: &[u8; 32],
    pack_d: usize,
) -> Result<H12GermCompiledConstraintSystem, String> {
    if surfaces.is_empty() {
        return Err("H12 GERM requires at least one surface".to_string());
    }
    let layout = &surfaces[0].proof_layout;
    let z_w_len = layout.z_w_len;
    let k_star = layout.k_star;
    let low_cube_len = layout.low_cube_len;
    if low_cube_len == 0 || low_cube_len > k_star {
        return Err("H12 GERM invalid low_cube_len".to_string());
    }
    let mut z_w_pos_set = BTreeSet::<usize>::new();
    let mut block_pos_sets = BTreeMap::<usize, BTreeSet<usize>>::new();
    for surface in surfaces {
        if surface.proof_layout.z_w_len != z_w_len
            || surface.proof_layout.k_star != k_star
            || surface.proof_layout.low_cube_len != low_cube_len
        {
            return Err("H12 GERM inconsistent proof layout".to_string());
        }
        let base = &surface.local_check;
        let mut mark_pi_idx = |pi_idx: usize| -> Result<(), String> {
            if pi_idx < z_w_len {
                z_w_pos_set.insert(pi_idx);
                return Ok(());
            }
            let off = pi_idx - z_w_len;
            let block_idx = off / k_star;
            let pos = off % k_star;
            block_pos_sets.entry(block_idx).or_default().insert(pos);
            Ok(())
        };
        for &(pi_idx, _) in &base.q1_pi_terms {
            mark_pi_idx(pi_idx)?;
        }
        for &(pi_idx, _) in &base.q2_pi_terms {
            mark_pi_idx(pi_idx)?;
        }
        for &(pi_idx, _) in &base.q3_pi_sparse_terms {
            mark_pi_idx(pi_idx)?;
        }
        for &(pos, _) in &base.q1_w_terms {
            if pos >= k_star {
                return Err(format!(
                    "H12 GERM q1 w_eval term position out of range: block={} pos={} k_star={}",
                    base.block_id, pos, k_star
                ));
            }
            mark_pi_idx(base.w_eval_block_pi_offset + pos)?;
        }
        for &(pos, _) in &base.q2_w_terms {
            if pos >= k_star {
                return Err(format!(
                    "H12 GERM q2 w_eval term position out of range: block={} pos={} k_star={}",
                    base.block_id, pos, k_star
                ));
            }
            mark_pi_idx(base.w_eval_block_pi_offset + pos)?;
        }
        for &(pos, _) in &base.q3_w_terms {
            if pos >= k_star {
                return Err(format!(
                    "H12 GERM q3 w_eval term position out of range: block={} pos={} k_star={}",
                    base.block_id, pos, k_star
                ));
            }
            mark_pi_idx(base.w_eval_block_pi_offset + pos)?;
        }
        // Do not pull H_j terms into local-view variable allocation.
        // H_j consistency is checked separately from opened logical rows and projection residuals.
        for (h_idx, h_cons) in surface.h_w_eval_constraints.iter().enumerate() {
            for &(pos, _) in &h_cons.terms {
                if pos >= k_star {
                    return Err(format!(
                        "H12 GERM H_j term position out of range: block={} constraint={} pos={} k_star={}",
                        base.block_id, h_idx, pos, k_star
                    ));
                }
            }
        }
    }
    let z_w_positions: Vec<usize> = z_w_pos_set.into_iter().collect();
    let logical_block_indices: Vec<usize> = block_pos_sets.keys().copied().collect();
    let z_w_var_map: BTreeMap<usize, usize> = z_w_positions
        .iter()
        .enumerate()
        .map(|(i, &pi_idx)| (pi_idx, i))
        .collect();
    let mut logical_block_positions = Vec::with_capacity(logical_block_indices.len());
    let mut block_pos_var_map = BTreeMap::<(usize, usize), usize>::new();
    let mut next_var = z_w_positions.len();
    for &block_idx in &logical_block_indices {
        let positions: Vec<usize> = block_pos_sets
            .get(&block_idx)
            .ok_or_else(|| {
                format!("H12 GERM missing gathered block position set for block {block_idx}")
            })?
            .iter()
            .copied()
            .collect();
        for &pos in &positions {
            block_pos_var_map.insert((block_idx, pos), next_var);
            next_var += 1;
        }
        logical_block_positions.push(positions);
    }
    let local_view_len = next_var;
    let logical_blocks_total = layout.blocks;
    let global_pack_d = pack_d.max(1);
    let global_err_packed_len = if logical_blocks_total == 0 {
        0
    } else {
        (logical_blocks_total + global_pack_d - 1) / global_pack_d
    };
    let matrix_seed = derive_germ_ajtai_seed(stage1_root);

    fn lower_pi_idx(
        pi_idx: usize,
        coeff: F257,
        z_w_len: usize,
        k_star: usize,
        z_w_var_map: &BTreeMap<usize, usize>,
        block_pos_var_map: &BTreeMap<(usize, usize), usize>,
    ) -> Result<Vec<(usize, F257)>, String> {
        if coeff.is_zero() {
            return Ok(Vec::new());
        }
        if pi_idx < z_w_len {
            let var_idx = *z_w_var_map
                .get(&pi_idx)
                .ok_or_else(|| format!("H12 GERM missing z_w witness position {pi_idx}"))?;
            return Ok(vec![(var_idx, coeff)]);
        }
        let off = pi_idx - z_w_len;
        let block_idx = off / k_star;
        let pos = off % k_star;
        let var_idx = *block_pos_var_map.get(&(block_idx, pos)).ok_or_else(|| {
            format!(
                "H12 GERM missing mapped w_eval position: block={} pos={}",
                block_idx, pos
            )
        })?;
        Ok(vec![(var_idx, coeff)])
    }

    let mut constraints = Vec::new();
    for surface in surfaces {
        let base = &surface.local_check;
        let mut alpha_terms = Vec::new();
        for &(pi_idx, coeff) in &base.q1_pi_terms {
            alpha_terms.extend(lower_pi_idx(
                pi_idx,
                coeff,
                z_w_len,
                k_star,
                &z_w_var_map,
                &block_pos_var_map,
            )?);
        }
        for &(pos, coeff) in &base.q1_w_terms {
            alpha_terms.extend(lower_pi_idx(
                base.w_eval_block_pi_offset + pos,
                coeff,
                z_w_len,
                k_star,
                &z_w_var_map,
                &block_pos_var_map,
            )?);
        }
        let mut beta_terms = Vec::new();
        for &(pi_idx, coeff) in &base.q2_pi_terms {
            beta_terms.extend(lower_pi_idx(
                pi_idx,
                coeff,
                z_w_len,
                k_star,
                &z_w_var_map,
                &block_pos_var_map,
            )?);
        }
        for &(pos, coeff) in &base.q2_w_terms {
            beta_terms.extend(lower_pi_idx(
                base.w_eval_block_pi_offset + pos,
                coeff,
                z_w_len,
                k_star,
                &z_w_var_map,
                &block_pos_var_map,
            )?);
        }
        let mut gamma_terms = Vec::new();
        for &(pi_idx, coeff) in &base.q3_pi_sparse_terms {
            gamma_terms.extend(lower_pi_idx(
                pi_idx,
                coeff,
                z_w_len,
                k_star,
                &z_w_var_map,
                &block_pos_var_map,
            )?);
        }
        for &(pos, coeff) in &base.q3_w_terms {
            gamma_terms.extend(lower_pi_idx(
                base.w_eval_block_pi_offset + pos,
                coeff,
                z_w_len,
                k_star,
                &z_w_var_map,
                &block_pos_var_map,
            )?);
        }
        constraints.push(AadpMulConstraint {
            a: AadpLinearForm {
                constant: base.q1_x_dot,
                terms: alpha_terms,
            },
            b: AadpLinearForm {
                constant: base.q2_x_dot,
                terms: beta_terms,
            },
            c: AadpLinearForm {
                constant: base.q3_x_dot_sparse,
                terms: gamma_terms,
            },
            d: AadpLinearForm {
                constant: F257::ONE,
                terms: Vec::new(),
            },
        });
    }
    let global_err_packed_offset = local_view_len;
    let blind_offset = global_err_packed_offset + global_err_packed_len;
    let opening_projection_offset = blind_offset + H12_GERM_BLIND_LEN;
    let h_projection_offset = opening_projection_offset + H12_GERM_EXT16_PROJECTION_CHECKS;
    let germ_linear_fingerprint_offset = h_projection_offset + H12_GERM_EXT16_PROJECTION_CHECKS;
    let total_mul_checks = surfaces.len().saturating_mul(logical_blocks_total);
    let germ_mul_sumcheck_rounds = mul_sumcheck_nvars(total_mul_checks);
    let germ_mul_sumcheck_round_offset =
        germ_linear_fingerprint_offset + H12_GERM_LIN_FINGERPRINT_COORDS;
    let germ_mul_sumcheck_final_offset =
        germ_mul_sumcheck_round_offset + germ_mul_sumcheck_rounds * H12_GERM_EXT16_COORDS;
    let num_variables = germ_mul_sumcheck_final_offset
        + if germ_mul_sumcheck_rounds == 0 {
            0
        } else {
            H12_GERM_EXT16_COORDS
        };
    for idx in 0..H12_GERM_EXT16_PROJECTION_CHECKS {
        constraints.push(germ_zero_constraint(opening_projection_offset + idx));
    }
    for idx in 0..H12_GERM_EXT16_PROJECTION_CHECKS {
        constraints.push(germ_zero_constraint(h_projection_offset + idx));
    }
    for idx in 0..H12_GERM_LIN_FINGERPRINT_COORDS {
        constraints.push(germ_zero_constraint(germ_linear_fingerprint_offset + idx));
    }
    for idx in 0..(germ_mul_sumcheck_rounds * H12_GERM_EXT16_COORDS) {
        constraints.push(germ_zero_constraint(germ_mul_sumcheck_round_offset + idx));
    }
    if germ_mul_sumcheck_rounds != 0 {
        for idx in 0..H12_GERM_EXT16_COORDS {
            constraints.push(germ_zero_constraint(germ_mul_sumcheck_final_offset + idx));
        }
    }
    Ok(H12GermCompiledConstraintSystem {
        z_w_positions,
        logical_block_indices,
        logical_block_positions,
        logical_blocks_total,
        z_w_len,
        k_star,
        low_cube_len,
        local_view_len,
        global_err_packed_len,
        cs: AadpConstraintSystem {
            num_variables,
            constraints,
        },
        global_err_packed_offset,
        blind_offset,
        opening_projection_offset,
        h_projection_offset,
        germ_linear_fingerprint_offset,
        germ_mul_sumcheck_round_offset,
        germ_mul_sumcheck_final_offset,
        germ_mul_sumcheck_rounds,
        params: H12GermParams {
            pack_d: pack_d as u16,
            commit_rows: H12_GERM_COMMIT_ROWS as u16,
            blind_len: H12_GERM_BLIND_LEN as u16,
        },
        matrix_seed,
    })
}

impl H12GermCompiledConstraintSystem {
    pub fn local_view_from_pi0(&self, pi0: &[F257]) -> Result<Vec<F257>, String> {
        let mut out = Vec::with_capacity(self.local_view_len);
        for &pi_idx in &self.z_w_positions {
            let v = *pi0
                .get(pi_idx)
                .ok_or_else(|| format!("H12 GERM z_w pi index out of range: {pi_idx}"))?;
            out.push(v);
        }
        for (block_idx, positions) in self
            .logical_block_indices
            .iter()
            .zip(self.logical_block_positions.iter())
        {
            let start = self
                .z_w_len
                .checked_add(
                    block_idx
                        .checked_mul(self.k_star)
                        .ok_or_else(|| "H12 GERM block start overflow".to_string())?,
                )
                .ok_or_else(|| "H12 GERM block start overflow".to_string())?;
            for &pos in positions {
                let pi_idx = start
                    .checked_add(pos)
                    .ok_or_else(|| "H12 GERM block position overflow".to_string())?;
                let v = *pi0
                    .get(pi_idx)
                    .ok_or_else(|| format!("H12 GERM pi index out of range: {pi_idx}"))?;
                out.push(v);
            }
        }
        Ok(out)
    }

    pub fn local_view_from_openings(
        &self,
        pi0_len: usize,
        openings: &[H12PiBlockOpening],
    ) -> Result<Vec<F257>, String> {
        let pack_d = self.params.pack_d as usize;
        let mut by_block = BTreeMap::<usize, Vec<F257>>::new();
        for opening in openings {
            let block_idx = opening.block_index as usize;
            if by_block.contains_key(&block_idx) {
                return Err(format!("H12 GERM duplicate opening for block {block_idx}"));
            }
            let expected_len = block_len_for_index(pi0_len, pack_d, block_idx);
            if opening.values.len() != expected_len {
                return Err(format!(
                    "H12 GERM opening length mismatch: block={} got={} expected={}",
                    block_idx,
                    opening.values.len(),
                    expected_len
                ));
            }
            by_block.insert(
                block_idx,
                opening
                    .values
                    .iter()
                    .copied()
                    .map(|x| F257::from((x % 257) as u64))
                    .collect(),
            );
        }
        let mut out = Vec::with_capacity(self.local_view_len);
        for &pi_idx in &self.z_w_positions {
            let block_idx = pi_idx / pack_d;
            let pos = pi_idx % pack_d;
            let vals = by_block
                .get(&block_idx)
                .ok_or_else(|| format!("H12 GERM missing opening for block {block_idx}"))?;
            let v = *vals
                .get(pos)
                .ok_or_else(|| format!("H12 GERM opening too short for pi_idx={pi_idx}"))?;
            out.push(v);
        }
        for (block_idx, positions) in self
            .logical_block_indices
            .iter()
            .zip(self.logical_block_positions.iter())
        {
            let start = self
                .z_w_len
                .checked_add(
                    block_idx
                        .checked_mul(self.k_star)
                        .ok_or_else(|| "H12 GERM block start overflow".to_string())?,
                )
                .ok_or_else(|| "H12 GERM block start overflow".to_string())?;
            for &rel in positions {
                let pi_idx = start
                    .checked_add(rel)
                    .ok_or_else(|| "H12 GERM block position overflow".to_string())?;
                let packed_block = pi_idx / pack_d;
                let pos = pi_idx % pack_d;
                let vals = by_block
                    .get(&packed_block)
                    .ok_or_else(|| format!("H12 GERM missing opening for block {packed_block}"))?;
                let v = *vals
                    .get(pos)
                    .ok_or_else(|| format!("H12 GERM opening too short for pi_idx={pi_idx}"))?;
                out.push(v);
            }
        }
        Ok(out)
    }
}

fn germ_fingerprints_from_surfaces_and_per_rep_mul_quads(
    designated_challenge: &[u16],
    surfaces: &[Theorem43AlvoLocalCheckSurface<F257>],
    logical_rows_by_block: &BTreeMap<usize, &[u16]>,
    global_err_packed_values: &[u16],
    per_rep_mul_quads: &[Vec<H12GermMulQuad>],
) -> Result<(Vec<[u16; F257_EXT16_DEGREE]>, Vec<[u16; F257_EXT16_DEGREE]>), String> {
    if designated_challenge.len() != H12_GERM_DESIGNATED_CHALLENGE_WORDS {
        return Err(format!(
            "H12 GERM designated-challenge length mismatch: got={} expected={}",
            designated_challenge.len(),
            H12_GERM_DESIGNATED_CHALLENGE_WORDS
        ));
    }
    let linear_coords = h_linear_residual_coords_from_surfaces_with_rows(
        surfaces,
        logical_rows_by_block,
        global_err_packed_values,
    )?;
    let mul_coords = mul_residual_coords_from_per_rep_mul_quads(per_rep_mul_quads)?;
    let linear_fp = ext16_mix_coords_to_fingerprints(
        designated_challenge,
        b"LFP_H12_GERM_LIN_FP_V1",
        linear_coords.as_slice(),
        H12_GERM_LIN_FINGERPRINTS,
    );
    let mul_fp = ext16_mix_coords_to_fingerprints(
        designated_challenge,
        b"LFP_H12_GERM_MUL_FP_V1",
        mul_coords.as_slice(),
        H12_GERM_MUL_FINGERPRINTS,
    );
    Ok((linear_fp, mul_fp))
}

fn ensure_base_residuals_zero(label: &str, coords: &[u16]) -> Result<(), String> {
    for (idx, &coord) in coords.iter().enumerate() {
        if coord % 257 != 0 {
            return Err(format!(
                "H12 GERM nonzero {label} residual at index {idx}: value={}",
                coord % 257
            ));
        }
    }
    Ok(())
}

fn ensure_ext16_fingerprints_zero(
    label: &str,
    fingerprints: &[[u16; F257_EXT16_DEGREE]],
) -> Result<(), String> {
    for (fp_idx, fp) in fingerprints.iter().enumerate() {
        for (coord_idx, &coord) in fp.iter().enumerate() {
            if coord % 257 != 0 {
                return Err(format!(
                    "H12 GERM nonzero {label} fingerprint at fp={} coord={}: value={}",
                    fp_idx,
                    coord_idx,
                    coord % 257
                ));
            }
        }
    }
    Ok(())
}

fn ensure_h_constraints_zero_from_surfaces(
    surfaces: &[Theorem43AlvoLocalCheckSurface<F257>],
    logical_rows_by_block: &BTreeMap<usize, &[u16]>,
) -> Result<(), String> {
    for (surface_idx, surface) in surfaces.iter().enumerate() {
        let block_id = surface.local_check.block_id;
        let row = logical_rows_by_block.get(&block_id).ok_or_else(|| {
            format!(
                "H12 GERM missing logical row for direct H_j check: surface_idx={} block_id={}",
                surface_idx, block_id
            )
        })?;
        for (h_idx, h_cons) in surface.h_w_eval_constraints.iter().enumerate() {
            let mut acc = h_cons.constant;
            for &(pos, coeff) in &h_cons.terms {
                let w_u16 = *row.get(pos).ok_or_else(|| {
                    format!(
                        "H12 GERM H_j row out of range in direct check: surface_idx={} constraint={} block_id={} pos={} row_len={}",
                        surface_idx,
                        h_idx,
                        block_id,
                        pos,
                        row.len()
                    )
                })?;
                acc += coeff * F257::from((w_u16 % 257) as u64);
            }
            if acc != F257::ZERO {
                return Err(format!(
                    "H12 GERM direct H_j constraint failed: surface_idx={} constraint={} block_id={} value={}",
                    surface_idx,
                    h_idx,
                    block_id,
                    f257_to_u16(acc)
                ));
            }
        }
    }
    Ok(())
}

fn ensure_per_rep_mul_quads_zero(
    per_rep_mul_quads: &[Vec<H12GermMulQuad>],
) -> Result<(), String> {
    for (rep_idx, quads) in per_rep_mul_quads.iter().enumerate() {
        for (quad_idx, quad) in quads.iter().enumerate() {
            let err = mul_quad_residual(quad);
            if err != 0 {
                return Err(format!(
                    "H12 GERM nonzero multiplicative residual at rep={} idx={}: value={}",
                    rep_idx,
                    quad_idx,
                    err
                ));
            }
        }
    }
    Ok(())
}

#[derive(Clone, Debug)]
struct DerivedGermWitnessData {
    global_err_packed_values: Vec<u16>,
    opening_projection_residuals: Vec<u16>,
    h_projection_residuals: Vec<u16>,
    germ_linear_fingerprints_flat: Vec<u16>,
    germ_mul_sumcheck_round_residuals: Vec<u16>,
    germ_mul_sumcheck_final_residual: Vec<u16>,
}

fn validate_germ_proof_shape(
    compiled: &H12GermCompiledConstraintSystem,
    proof: &H12GermProof,
) -> Result<(), String> {
    if proof.params.pack_d != compiled.params.pack_d
        || proof.params.commit_rows != compiled.params.commit_rows
        || proof.params.blind_len != compiled.params.blind_len
    {
        return Err("H12 GERM proof params mismatch".to_string());
    }
    if proof.local_view_values.len() != compiled.local_view_len {
        return Err(format!(
            "H12 GERM proof local-view length mismatch: got={} expected={}",
            proof.local_view_values.len(),
            compiled.local_view_len
        ));
    }
    if proof.blind_values.len() != compiled.params.blind_len as usize {
        return Err(format!(
            "H12 GERM proof blind length mismatch: got={} expected={}",
            proof.blind_values.len(),
            compiled.params.blind_len
        ));
    }
    if proof.verifier_payload.global_err_packed_values.len() != compiled.global_err_packed_len {
        return Err(format!(
            "H12 GERM proof global-err packed length mismatch: got={} expected={}",
            proof.verifier_payload.global_err_packed_values.len(),
            compiled.global_err_packed_len
        ));
    }
    if proof.verifier_payload.designated_challenge.len() != H12_GERM_DESIGNATED_CHALLENGE_WORDS {
        return Err(format!(
            "H12 GERM proof designated-challenge length mismatch: got={} expected={}",
            proof.verifier_payload.designated_challenge.len(),
            H12_GERM_DESIGNATED_CHALLENGE_WORDS
        ));
    }
    if proof.verifier_payload.opening_projection_residuals.len() != H12_GERM_EXT16_PROJECTION_CHECKS
    {
        return Err(format!(
            "H12 GERM proof opening projection length mismatch: got={} expected={}",
            proof.verifier_payload.opening_projection_residuals.len(),
            H12_GERM_EXT16_PROJECTION_CHECKS
        ));
    }
    if proof.verifier_payload.h_projection_residuals.len() != H12_GERM_EXT16_PROJECTION_CHECKS {
        return Err(format!(
            "H12 GERM proof H projection length mismatch: got={} expected={}",
            proof.verifier_payload.h_projection_residuals.len(),
            H12_GERM_EXT16_PROJECTION_CHECKS
        ));
    }
    if proof.verifier_payload.germ_linear_fingerprints.len() != H12_GERM_LIN_FINGERPRINTS {
        return Err(format!(
            "H12 GERM proof linear fingerprint count mismatch: got={} expected={}",
            proof.verifier_payload.germ_linear_fingerprints.len(),
            H12_GERM_LIN_FINGERPRINTS
        ));
    }
    if proof.verifier_payload.germ_mul_sumcheck.nvars as usize != compiled.germ_mul_sumcheck_rounds {
        return Err(format!(
            "H12 GERM proof mul sumcheck round-count mismatch: got={} expected={}",
            proof.verifier_payload.germ_mul_sumcheck.nvars,
            compiled.germ_mul_sumcheck_rounds
        ));
    }
    Ok(())
}

fn derive_germ_witness_data_from_payload(
    compiled: &H12GermCompiledConstraintSystem,
    proof: &H12GermProof,
) -> Result<DerivedGermWitnessData, String> {
    validate_germ_proof_shape(compiled, proof)?;
    verify_germ_ajtai_commitment(compiled, proof)?;
    let (round_residuals, final_residual) = derive_mul_sumcheck_residuals_from_payload(
        proof.verifier_payload.designated_challenge.as_slice(),
        &proof.verifier_payload.germ_mul_sumcheck,
    )?;
    Ok(DerivedGermWitnessData {
        global_err_packed_values: proof.verifier_payload.global_err_packed_values.clone(),
        opening_projection_residuals: proof.verifier_payload.opening_projection_residuals.clone(),
        h_projection_residuals: proof.verifier_payload.h_projection_residuals.clone(),
        germ_linear_fingerprints_flat: flatten_ext16_fingerprints(
            proof.verifier_payload.germ_linear_fingerprints.as_slice(),
        ),
        germ_mul_sumcheck_round_residuals: round_residuals,
        germ_mul_sumcheck_final_residual: final_residual,
    })
}

pub fn prove_germ_from_local_view_with_surfaces<R: RngCore>(
    compiled: &H12GermCompiledConstraintSystem,
    surfaces: &[Theorem43AlvoLocalCheckSurface<F257>],
    logical_rows_by_block: &BTreeMap<usize, Vec<u16>>,
    per_rep_mul_quads: &[Vec<H12GermMulQuad>],
    local_view: &[F257],
    rng: &mut R,
) -> Result<H12GermProof, String> {
    if local_view.len() != compiled.local_view_len {
        return Err(format!(
            "H12 GERM local-view length mismatch: got={} expected={}",
            local_view.len(),
            compiled.local_view_len
        ));
    }
    let global_err_packed_values = global_err_packed_from_per_rep_mul_quads(
        &compiled.matrix_seed,
        per_rep_mul_quads,
        compiled.logical_blocks_total,
        compiled.params.pack_d as usize,
        compiled.global_err_packed_len,
    )?;
    let blind_values: Vec<F257> = (0..compiled.params.blind_len as usize)
        .map(|_| random_f257(rng))
        .collect();
    let ajtai_commitment_rows = ajtai_commitment_rows(
        compiled,
        local_view,
        global_err_packed_values.as_slice(),
        blind_values.as_slice(),
    )?;
    let mut opened_logical_rows = Vec::with_capacity(compiled.logical_block_indices.len());
    for &logical_block in &compiled.logical_block_indices {
        let row = logical_rows_by_block.get(&logical_block).ok_or_else(|| {
            format!(
                "H12 GERM missing streamed logical row for block {}",
                logical_block
            )
        })?;
        opened_logical_rows.push(row.iter().map(|&v| v % 257).collect());
    }
    let logical_rows_map =
        logical_rows_map_for_compiled(compiled, opened_logical_rows.as_slice())?;
    ensure_local_view_matches_logical_rows(compiled, local_view, &logical_rows_map)?;
    ensure_h_constraints_zero_from_surfaces(surfaces, &logical_rows_map)?;
    ensure_per_rep_mul_quads_zero(per_rep_mul_quads)?;
    let commitment_root = germ_commitment_root(ajtai_commitment_rows.as_slice());
    let designated_challenge = derive_germ_designated_challenge(&compiled.matrix_seed, &commitment_root);
    let opening_projection_residuals = opening_projection_residuals(
        designated_challenge.as_slice(),
        ajtai_commitment_rows.as_slice(),
        ajtai_commitment_rows.as_slice(),
    )?;
    ensure_base_residuals_zero("opening projection", opening_projection_residuals.as_slice())?;
    let h_projection_residuals = h_projection_residuals_from_surfaces_with_rows(
        designated_challenge.as_slice(),
        surfaces,
        &logical_rows_map,
        global_err_packed_values.as_slice(),
    )?;
    ensure_base_residuals_zero("H projection", h_projection_residuals.as_slice())?;
    let germ_linear_fingerprints = ext16_mix_coords_to_fingerprints(
        designated_challenge.as_slice(),
        b"LFP_H12_GERM_LIN_FP_V1",
        h_linear_residual_coords_from_surfaces_with_rows(
            surfaces,
            &logical_rows_map,
            global_err_packed_values.as_slice(),
        )?
        .as_slice(),
        H12_GERM_LIN_FINGERPRINTS,
    );
    let germ_mul_sumcheck = prove_germ_mul_sumcheck_from_mul_quads(
        designated_challenge.as_slice(),
        per_rep_mul_quads,
    )?;
    let (_, final_mul_residual) = derive_mul_sumcheck_residuals_from_payload(
        designated_challenge.as_slice(),
        &germ_mul_sumcheck,
    )?;
    ensure_base_residuals_zero("mul sumcheck final", final_mul_residual.as_slice())?;
    ensure_ext16_fingerprints_zero("linear", germ_linear_fingerprints.as_slice())?;
    let proof = H12GermProof {
        params: compiled.params.clone(),
        local_view_values: local_view.iter().copied().map(f257_to_u16).collect(),
        blind_values: blind_values.into_iter().map(f257_to_u16).collect(),
        ajtai_commitment_rows,
        verifier_payload: H12GermVerifierPayload {
            global_err_packed_values,
            designated_challenge,
            opening_projection_residuals,
            h_projection_residuals,
            germ_linear_fingerprints,
            germ_mul_sumcheck,
        },
    };
    validate_germ_proof_shape(compiled, &proof)?;
    Ok(proof)
}

pub fn witness_from_germ_proof(
    compiled: &H12GermCompiledConstraintSystem,
    proof: &H12GermProof,
) -> Result<Vec<F257>, String> {
    let derived = derive_germ_witness_data_from_payload(compiled, proof)?;
    let mut out = Vec::with_capacity(compiled.cs.num_variables);
    out.extend(
        proof
            .local_view_values
            .iter()
            .copied()
            .map(|x| F257::from((x % 257) as u64)),
    );
    out.extend(
        derived
            .global_err_packed_values
            .iter()
            .copied()
            .map(|x| F257::from((x % 257) as u64)),
    );
    out.extend(
        proof
            .blind_values
            .iter()
            .copied()
            .map(|x| F257::from((x % 257) as u64)),
    );
    out.extend(
        derived
            .opening_projection_residuals
            .iter()
            .copied()
            .map(|x| F257::from((x % 257) as u64)),
    );
    out.extend(
        derived
            .h_projection_residuals
            .iter()
            .copied()
            .map(|x| F257::from((x % 257) as u64)),
    );
    out.extend(
        derived
            .germ_linear_fingerprints_flat
            .iter()
            .copied()
            .map(|x| F257::from((x % 257) as u64)),
    );
    out.extend(
        derived
            .germ_mul_sumcheck_round_residuals
            .iter()
            .copied()
            .map(|x| F257::from((x % 257) as u64)),
    );
    out.extend(
        derived
            .germ_mul_sumcheck_final_residual
            .iter()
            .copied()
            .map(|x| F257::from((x % 257) as u64)),
    );
    if out.len() != compiled.cs.num_variables {
        return Err(format!(
            "H12 GERM witness length mismatch after assembly: got={} expected={}",
            out.len(),
            compiled.cs.num_variables
        ));
    }
    Ok(out)
}

pub fn verify_germ_from_surfaces(
    compiled: &H12GermCompiledConstraintSystem,
    proof: &H12GermProof,
    surfaces: &[Theorem43AlvoLocalCheckSurface<F257>],
    logical_rows_by_block: &BTreeMap<usize, Vec<u16>>,
    per_rep_mul_quads: &[Vec<H12GermMulQuad>],
) -> Result<(), String> {
    validate_germ_proof_shape(compiled, proof)?;
    verify_germ_ajtai_commitment(compiled, proof)?;
    let local_view: Vec<F257> = proof
        .local_view_values
        .iter()
        .copied()
        .map(|x| F257::from((x % 257) as u64))
        .collect();
    let mut opened_logical_rows = Vec::with_capacity(compiled.logical_block_indices.len());
    for &logical_block in &compiled.logical_block_indices {
        let row = logical_rows_by_block.get(&logical_block).ok_or_else(|| {
            format!(
                "H12 GERM missing streamed logical row for block {}",
                logical_block
            )
        })?;
        opened_logical_rows.push(row.iter().map(|&v| v % 257).collect());
    }
    let logical_rows_map =
        logical_rows_map_for_compiled(compiled, opened_logical_rows.as_slice())?;
    ensure_local_view_matches_logical_rows(compiled, local_view.as_slice(), &logical_rows_map)?;
    ensure_h_constraints_zero_from_surfaces(surfaces, &logical_rows_map)?;
    ensure_per_rep_mul_quads_zero(per_rep_mul_quads)?;
    let expected_global_err_packed = global_err_packed_from_per_rep_mul_quads(
        &compiled.matrix_seed,
        per_rep_mul_quads,
        compiled.logical_blocks_total,
        compiled.params.pack_d as usize,
        compiled.global_err_packed_len,
    )?;
    if expected_global_err_packed != proof.verifier_payload.global_err_packed_values {
        return Err("H12 GERM global err packed semantics mismatch".to_string());
    }
    let expected_h_projection = h_projection_residuals_from_surfaces_with_rows(
        proof.verifier_payload.designated_challenge.as_slice(),
        surfaces,
        &logical_rows_map,
        proof.verifier_payload.global_err_packed_values.as_slice(),
    )?;
    if expected_h_projection != proof.verifier_payload.h_projection_residuals {
        return Err("H12 GERM H projection mismatch".to_string());
    }
    let expected_linear = ext16_mix_coords_to_fingerprints(
        proof.verifier_payload.designated_challenge.as_slice(),
        b"LFP_H12_GERM_LIN_FP_V1",
        h_linear_residual_coords_from_surfaces_with_rows(
            surfaces,
            &logical_rows_map,
            proof.verifier_payload.global_err_packed_values.as_slice(),
        )?
        .as_slice(),
        H12_GERM_LIN_FINGERPRINTS,
    );
    if expected_linear != proof.verifier_payload.germ_linear_fingerprints {
        return Err("H12 GERM linear fingerprint mismatch".to_string());
    }
    let expected_mul_sumcheck = prove_germ_mul_sumcheck_from_mul_quads(
        proof.verifier_payload.designated_challenge.as_slice(),
        per_rep_mul_quads,
    )?;
    if expected_mul_sumcheck.nvars != proof.verifier_payload.germ_mul_sumcheck.nvars
        || expected_mul_sumcheck.rounds.len() != proof.verifier_payload.germ_mul_sumcheck.rounds.len()
        || expected_mul_sumcheck
            .rounds
            .iter()
            .zip(proof.verifier_payload.germ_mul_sumcheck.rounds.iter())
            .any(|(a, b)| a.evaluations != b.evaluations)
        || expected_mul_sumcheck.opening.a_eval != proof.verifier_payload.germ_mul_sumcheck.opening.a_eval
        || expected_mul_sumcheck.opening.b_eval != proof.verifier_payload.germ_mul_sumcheck.opening.b_eval
        || expected_mul_sumcheck.opening.c_eval != proof.verifier_payload.germ_mul_sumcheck.opening.c_eval
        || expected_mul_sumcheck.opening.d_eval != proof.verifier_payload.germ_mul_sumcheck.opening.d_eval
    {
        return Err("H12 GERM multiplicative sumcheck mismatch".to_string());
    }
    let (round_residuals, final_residual) = derive_mul_sumcheck_residuals_from_payload(
        proof.verifier_payload.designated_challenge.as_slice(),
        &proof.verifier_payload.germ_mul_sumcheck,
    )?;
    ensure_base_residuals_zero("mul sumcheck round", round_residuals.as_slice())?;
    ensure_base_residuals_zero("mul sumcheck final", final_residual.as_slice())?;
    if expected_mul_sumcheck.nvars as usize != compiled.germ_mul_sumcheck_rounds {
        return Err("H12 GERM multiplicative sumcheck round count mismatch".to_string());
    }
    Ok(())
}

pub fn verify_germ_ajtai_commitment(
    compiled: &H12GermCompiledConstraintSystem,
    proof: &H12GermProof,
) -> Result<(), String> {
    let local_view: Vec<F257> = proof
        .local_view_values
        .iter()
        .copied()
        .map(|x| F257::from((x % 257) as u64))
        .collect();
    let blind_values: Vec<F257> = proof
        .blind_values
        .iter()
        .copied()
        .map(|x| F257::from((x % 257) as u64))
        .collect();
    validate_germ_proof_shape(compiled, proof)?;
    let global_err_packed_values = proof.verifier_payload.global_err_packed_values.clone();
    let expected = ajtai_commitment_rows(
        compiled,
        local_view.as_slice(),
        global_err_packed_values.as_slice(),
        blind_values.as_slice(),
    )?;
    if expected != proof.ajtai_commitment_rows {
        return Err("H12 GERM Ajtai commitment rows mismatch".to_string());
    }
    let commitment_root = germ_commitment_root(proof.ajtai_commitment_rows.as_slice());
    let expected_designated =
        derive_germ_designated_challenge(&compiled.matrix_seed, &commitment_root);
    if expected_designated != proof.verifier_payload.designated_challenge {
        return Err("H12 GERM designated challenge mismatch".to_string());
    }
    let expected_opening_projection = opening_projection_residuals(
        proof.verifier_payload.designated_challenge.as_slice(),
        expected.as_slice(),
        proof.ajtai_commitment_rows.as_slice(),
    )?;
    if expected_opening_projection != proof.verifier_payload.opening_projection_residuals {
        return Err("H12 GERM opening projection mismatch".to_string());
    }
    Ok(())
}

fn ajtai_commitment_rows(
    compiled: &H12GermCompiledConstraintSystem,
    local_view: &[F257],
    global_err_packed_values: &[u16],
    blind_values: &[F257],
) -> Result<Vec<Vec<u64>>, String> {
    if local_view.len() != compiled.local_view_len {
        return Err("H12 GERM local-view width mismatch".to_string());
    }
    if global_err_packed_values.len() != compiled.global_err_packed_len {
        return Err("H12 GERM global-err packed width mismatch".to_string());
    }
    if blind_values.len() != compiled.params.blind_len as usize {
        return Err("H12 GERM blind width mismatch".to_string());
    }
    let mut witness =
        Vec::with_capacity(blind_values.len() + local_view.len() + global_err_packed_values.len());
    witness.extend(
        local_view
            .iter()
            .map(|v| <AjtaiRing as PolyRing>::BaseRing::from(f257_to_u16(*v) as u128)),
    );
    witness.extend(
        global_err_packed_values
            .iter()
            .copied()
            .map(|v| <AjtaiRing as PolyRing>::BaseRing::from((v % 257) as u128)),
    );
    witness.extend(
        blind_values
            .iter()
            .map(|v| <AjtaiRing as PolyRing>::BaseRing::from(f257_to_u16(*v) as u128)),
    );
    let stage1_root = compiled.matrix_seed;
    let scheme = germ_ajtai_scheme(&stage1_root, witness.len());
    let commitment = scheme
        .commit_const_coeff_base_fast(witness.as_slice())
        .map_err(|e| format!("H12 GERM Ajtai commit failed: {e:?}"))?;
    Ok(commitment
        .as_ref()
        .iter()
        .map(|row| {
            row.coeffs()
                .iter()
                .map(|c| c.into_bigint().as_ref()[0])
                .collect()
        })
        .collect())
}

fn random_f257<R: RngCore>(rng: &mut R) -> F257 {
    let mut buf = [0u8; 2];
    rng.fill_bytes(&mut buf);
    F257::from((u16::from_le_bytes(buf) % 257) as u64)
}

#[cfg(test)]
mod tests {
    use dpp::{
        dr1cs_flpcp::Dr1csProofLayoutInfo,
        theorem43::{
            Theorem43AlvoLocalCheckSurface, Theorem43CapsuleLocalCheckSurface, Theorem43Coins,
        },
    };
    use rand::SeedableRng;
    use rand_chacha::ChaCha20Rng;

    use super::*;

    fn dummy_surface() -> Theorem43AlvoLocalCheckSurface<F257> {
        let local_check = Theorem43CapsuleLocalCheckSurface {
            block_id: 0,
            rep_id: 7,
            coins: Theorem43Coins {
                idx: 0,
                lambda: F257::ONE,
                rho: F257::ONE,
                sigma: F257::ONE,
                c_hit: F257::ONE,
            },
            q1_pi_terms: Vec::new(),
            q2_pi_terms: Vec::new(),
            q3_pi_sparse_terms: Vec::new(),
            q1_x_dot: F257::ZERO,
            q2_x_dot: F257::ZERO,
            q3_x_dot_sparse: F257::ZERO,
            q1_w_terms: Vec::new(),
            q2_w_terms: Vec::new(),
            q3_w_terms: Vec::new(),
            w_eval_block_pi_offset: 2,
            w_eval_block_len: 0,
            requires_dense_q3_w_eval: false,
        };
        Theorem43AlvoLocalCheckSurface {
            touched_pi_positions_exact: vec![0, 1],
            touched_pi_positions_conservative: vec![0, 1],
            local_check,
            proof_layout: Dr1csProofLayoutInfo {
                n: 2,
                n_total: 2,
                z_w_len: 2,
                pi0_len: 2,
                blocks: 0,
                ell_local: 2,
                k_star: 1,
                low_cube_len: 1,
                witness_positions_star: vec![0],
            },
            h_w_eval_constraints: Vec::new(),
        }
    }

    #[test]
    fn test_germ_roundtrip_witness_satisfies_compiled_relation() {
        let surfaces = vec![dummy_surface()];
        let compiled = compile_germ_constraint_system(&surfaces, &[9u8; 32], 64).expect("compile");
        let pi0 = vec![F257::from(2u64), F257::from(3u64)];
        let local_view = compiled
            .local_view_from_pi0(pi0.as_slice())
            .expect("local view");
        let logical_rows = std::collections::BTreeMap::<usize, Vec<u16>>::new();
        let mut rng = ChaCha20Rng::from_seed([7u8; 32]);
        let proof = prove_germ_from_local_view_with_surfaces(
            &compiled,
            surfaces.as_slice(),
            &logical_rows,
            &[],
            local_view.as_slice(),
            &mut rng,
        )
        .expect("prove");
        let witness = witness_from_germ_proof(&compiled, &proof).expect("witness");
        compiled
            .cs
            .check_witness(witness.as_slice())
            .expect("witness checks");
    }
}
