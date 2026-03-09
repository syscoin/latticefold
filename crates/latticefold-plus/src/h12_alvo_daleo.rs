//! Designated-challenge Ajtai Local-Opening proof for H12 ALVO.
//!
//! DALEO uses a **compressed local witness basis**:
//! - touched `z_w` coordinates
//! - low-cube prefix of each touched `w_eval` block
//! - affine Ajtai opening equations over that compressed witness
//!
//! Side-cube positions needed by the local ALVO checks are derived on the fly from the low cube.

use ark_ff::{Field, PrimeField};
use ark_std::Zero;
use cyclotomic_rings::rings::GoldilocksRing64 as AjtaiRing;
use dpp::theorem43::Theorem43AlvoLocalCheckSurface;
use latticefold::commitment::AjtaiCommitmentScheme;
use latticefold::transcript::poseidon::F257;
use rand::RngCore;
use sha2::Digest;
use stark_rings::PolyRing;
use std::collections::{BTreeMap, BTreeSet};

use crate::aadp_we::{AadpConstraintSystem, AadpLinearForm, AadpMulConstraint};
use crate::f257_ext16::F257_EXT16_DEGREE;
use crate::h12_pi_commit::{block_len_for_index, H12PiBlockOpening};
use crate::h12_pi_commit::f257_to_u16;

pub const H12_DALEO_COMMIT_ROWS: usize = 16;
pub const H12_DALEO_BLIND_LEN: usize = 16;
pub const H12_DALEO_EXT16_PROJECTION_CHECKS: usize = 4;
pub const H12_DALEO_DESIGNATED_CHALLENGE_WORDS: usize = F257_EXT16_DEGREE;
const H12_DALEO_AJTAI_DOMAIN: &[u8] = b"lfp_h12_daleo_local_view";
const GOLDILOCKS_P_U64: u64 = 0xFFFF_FFFF_0000_0001u64;

#[derive(Clone, Debug)]
pub struct H12DaleoParams {
    pub pack_d: u16,
    pub commit_rows: u16,
    pub blind_len: u16,
}

#[derive(Clone, Debug)]
pub struct H12DaleoProof {
    pub params: H12DaleoParams,
    pub local_view_values: Vec<u16>,
    pub blind_values: Vec<u16>,
    pub opening_projection_residuals: Vec<u16>,
    pub h_projection_residuals: Vec<u16>,
    // used to bind H projections to full exported H_j residual rows without inflating local_view.
    pub h_opened_logical_rows: Vec<Vec<u16>>,
    pub ajtai_commitment_rows: Vec<Vec<u64>>,
    pub designated_challenge: Vec<u16>,
}

#[derive(Clone, Debug)]
pub struct H12DaleoCompiledConstraintSystem {
    pub z_w_positions: Vec<usize>,
    pub logical_block_indices: Vec<usize>,
    pub logical_block_positions: Vec<Vec<usize>>,
    pub z_w_len: usize,
    pub k_star: usize,
    pub low_cube_len: usize,
    pub local_view_len: usize,
    pub cs: AadpConstraintSystem<F257>,
    pub blind_offset: usize,
    pub opening_projection_offset: usize,
    pub h_projection_offset: usize,
    pub params: H12DaleoParams,
    pub matrix_seed: [u8; 32],
}

fn derive_daleo_ajtai_seed(stage1_root: &[u8; 32]) -> [u8; 32] {
    let mut h = sha2::Sha256::new();
    h.update(b"LFP_H12_DALEO_AJTAI_V1");
    h.update(stage1_root);
    h.finalize().into()
}

fn daleo_commitment_root(rows: &[Vec<u64>]) -> [u8; 32] {
    let mut h = sha2::Sha256::new();
    h.update(b"LFP_H12_DALEO_COMMIT_ROOT_V1");
    h.update((rows.len() as u32).to_le_bytes());
    for row in rows {
        h.update((row.len() as u32).to_le_bytes());
        for &v in row {
            h.update(v.to_le_bytes());
        }
    }
    h.finalize().into()
}

fn derive_daleo_designated_challenge(
    matrix_seed: &[u8; 32],
    commitment_root: &[u8; 32],
) -> Vec<u16> {
    let mut out = Vec::with_capacity(H12_DALEO_DESIGNATED_CHALLENGE_WORDS);
    for idx in 0..H12_DALEO_DESIGNATED_CHALLENGE_WORDS {
        let mut h = sha2::Sha256::new();
        h.update(b"LFP_H12_DALEO_DESIGNATED_CHALLENGE_EXT16_V1");
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

fn ext16_project_coords(designated_challenge: &[u16], domain: &[u8], coords: &[u16]) -> Vec<u16> {
    let mut out = Vec::with_capacity(H12_DALEO_EXT16_PROJECTION_CHECKS);
    for proj_idx in 0..H12_DALEO_EXT16_PROJECTION_CHECKS {
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

fn opening_residual_coords(expected_rows: &[Vec<u64>], claimed_rows: &[Vec<u64>]) -> Result<Vec<u16>, String> {
    if expected_rows.len() != claimed_rows.len() {
        return Err(format!(
            "H12 DALEO Ajtai row count mismatch: expected={} claimed={}",
            expected_rows.len(),
            claimed_rows.len()
        ));
    }
    let mut coords = Vec::new();
    for (row_idx, (expected_row, claimed_row)) in expected_rows.iter().zip(claimed_rows.iter()).enumerate() {
        if expected_row.len() != claimed_row.len() {
            return Err(format!(
                "H12 DALEO Ajtai row lane mismatch at row {row_idx}: expected={} claimed={}",
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
    if designated_challenge.len() != H12_DALEO_DESIGNATED_CHALLENGE_WORDS {
        return Err(format!(
            "H12 DALEO designated-challenge length mismatch: got={} expected={}",
            designated_challenge.len(),
            H12_DALEO_DESIGNATED_CHALLENGE_WORDS
        ));
    }
    let coords = opening_residual_coords(expected_rows, claimed_rows)?;
    Ok(ext16_project_coords(
        designated_challenge,
        b"LFP_H12_DALEO_OPENING_EXT16_PROJ_V1",
        coords.as_slice(),
    ))
}

fn logical_rows_map_for_compiled<'a>(
    compiled: &H12DaleoCompiledConstraintSystem,
    logical_rows_in_compiled_order: &'a [Vec<u16>],
) -> Result<BTreeMap<usize, &'a [u16]>, String> {
    if logical_rows_in_compiled_order.len() != compiled.logical_block_indices.len() {
        return Err(format!(
            "H12 DALEO logical-row payload count mismatch: got={} expected={}",
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
                "H12 DALEO logical row too short: block={} got={} expected_at_least={}",
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
    compiled: &H12DaleoCompiledConstraintSystem,
    local_view: &[F257],
    logical_rows_by_block: &BTreeMap<usize, &[u16]>,
) -> Result<(), String> {
    if local_view.len() != compiled.local_view_len {
        return Err(format!(
            "H12 DALEO local-view length mismatch: got={} expected={}",
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
            .ok_or_else(|| format!("H12 DALEO missing logical row for block {logical_block}"))?;
        for &pos in positions {
            let lv = *local_view
                .get(lv_offset)
                .ok_or_else(|| format!("H12 DALEO local-view index out of range: idx={lv_offset}"))?;
            let got = f257_to_u16(lv);
            let expected = *row.get(pos).ok_or_else(|| {
                format!(
                    "H12 DALEO logical row position out of range: block={} pos={} row_len={}",
                    logical_block,
                    pos,
                    row.len()
                )
            })? % 257;
            if got != expected {
                return Err(format!(
                    "H12 DALEO logical-row payload mismatch at block={} pos={}: local_view={} payload={}",
                    logical_block, pos, got, expected
                ));
            }
            lv_offset += 1;
        }
    }
    if lv_offset != local_view.len() {
        return Err(format!(
            "H12 DALEO local-view traversal mismatch: traversed={} len={}",
            lv_offset,
            local_view.len()
        ));
    }
    Ok(())
}

fn h_projection_residuals_from_surfaces_with_rows(
    designated_challenge: &[u16],
    surfaces: &[Theorem43AlvoLocalCheckSurface<F257>],
    logical_rows_by_block: &BTreeMap<usize, &[u16]>,
) -> Result<Vec<u16>, String> {
    if designated_challenge.len() != H12_DALEO_DESIGNATED_CHALLENGE_WORDS {
        return Err(format!(
            "H12 DALEO designated-challenge length mismatch: got={} expected={}",
            designated_challenge.len(),
            H12_DALEO_DESIGNATED_CHALLENGE_WORDS
        ));
    }
    let mut coords = Vec::new();
    for (surface_idx, surface) in surfaces.iter().enumerate() {
        let block_id = surface.local_check.block_id;
        let row = logical_rows_by_block.get(&block_id).ok_or_else(|| {
            format!(
                "H12 DALEO missing logical row for H projection: surface_idx={} block_id={}",
                surface_idx, block_id
            )
        })?;
        for (h_idx, h_cons) in surface.h_w_eval_constraints.iter().enumerate() {
            let mut acc = h_cons.constant;
            for &(pos, coeff) in &h_cons.terms {
                let w_u16 = *row.get(pos).ok_or_else(|| {
                    format!(
                        "H12 DALEO H_j row out of range: surface_idx={} constraint={} block_id={} pos={} row_len={}",
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
    Ok(ext16_project_coords(
        designated_challenge,
        b"LFP_H12_DALEO_H_EXT16_PROJ_V1",
        coords.as_slice(),
    ))
}

fn daleo_ajtai_scheme(matrix_seed: &[u8; 32], width: usize) -> AjtaiCommitmentScheme<AjtaiRing> {
    AjtaiCommitmentScheme::<AjtaiRing>::seeded(
        H12_DALEO_AJTAI_DOMAIN,
        *matrix_seed,
        H12_DALEO_COMMIT_ROWS,
        width,
    )
}

fn daleo_zero_constraint(var_idx: usize) -> AadpMulConstraint<F257> {
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

pub fn compile_daleo_constraint_system(
    surfaces: &[Theorem43AlvoLocalCheckSurface<F257>],
    stage1_root: &[u8; 32],
    pack_d: usize,
) -> Result<H12DaleoCompiledConstraintSystem, String> {
    if surfaces.is_empty() {
        return Err("H12 DALEO requires at least one surface".to_string());
    }
    let layout = &surfaces[0].proof_layout;
    let z_w_len = layout.z_w_len;
    let k_star = layout.k_star;
    let low_cube_len = layout.low_cube_len;
    if low_cube_len == 0 || low_cube_len > k_star {
        return Err("H12 DALEO invalid low_cube_len".to_string());
    }
    let mut z_w_pos_set = BTreeSet::<usize>::new();
    let mut block_pos_sets = BTreeMap::<usize, BTreeSet<usize>>::new();
    for surface in surfaces {
        if surface.proof_layout.z_w_len != z_w_len
            || surface.proof_layout.k_star != k_star
            || surface.proof_layout.low_cube_len != low_cube_len
        {
            return Err("H12 DALEO inconsistent proof layout".to_string());
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
                    "H12 DALEO q1 w_eval term position out of range: block={} pos={} k_star={}",
                    base.block_id, pos, k_star
                ));
            }
            mark_pi_idx(base.w_eval_block_pi_offset + pos)?;
        }
        for &(pos, _) in &base.q2_w_terms {
            if pos >= k_star {
                return Err(format!(
                    "H12 DALEO q2 w_eval term position out of range: block={} pos={} k_star={}",
                    base.block_id, pos, k_star
                ));
            }
            mark_pi_idx(base.w_eval_block_pi_offset + pos)?;
        }
        for &(pos, _) in &base.q3_w_terms {
            if pos >= k_star {
                return Err(format!(
                    "H12 DALEO q3 w_eval term position out of range: block={} pos={} k_star={}",
                    base.block_id, pos, k_star
                ));
            }
            mark_pi_idx(base.w_eval_block_pi_offset + pos)?;
        }
        for (h_idx, h_cons) in surface.h_w_eval_constraints.iter().enumerate() {
            for &(pos, _) in &h_cons.terms {
                if pos >= k_star {
                    return Err(format!(
                        "H12 DALEO H_j term position out of range: block={} constraint={} pos={} k_star={}",
                        base.block_id, h_idx, pos, k_star
                    ));
                }
                mark_pi_idx(base.w_eval_block_pi_offset + pos)?;
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
            .ok_or_else(|| format!("H12 DALEO missing gathered block position set for block {block_idx}"))?
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
    let matrix_seed = derive_daleo_ajtai_seed(stage1_root);

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
                .ok_or_else(|| format!("H12 DALEO missing z_w witness position {pi_idx}"))?;
            return Ok(vec![(var_idx, coeff)]);
        }
        let off = pi_idx - z_w_len;
        let block_idx = off / k_star;
        let pos = off % k_star;
        let var_idx = *block_pos_var_map.get(&(block_idx, pos)).ok_or_else(|| {
            format!(
                "H12 DALEO missing mapped w_eval position: block={} pos={}",
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
    let blind_offset = local_view_len;
    let opening_projection_offset = blind_offset + H12_DALEO_BLIND_LEN;
    let h_projection_offset = opening_projection_offset + H12_DALEO_EXT16_PROJECTION_CHECKS;
    for idx in 0..H12_DALEO_EXT16_PROJECTION_CHECKS {
        constraints.push(daleo_zero_constraint(opening_projection_offset + idx));
    }
    for idx in 0..H12_DALEO_EXT16_PROJECTION_CHECKS {
        constraints.push(daleo_zero_constraint(h_projection_offset + idx));
    }
    Ok(H12DaleoCompiledConstraintSystem {
        z_w_positions,
        logical_block_indices,
        logical_block_positions,
        z_w_len,
        k_star,
        low_cube_len,
        local_view_len,
        cs: AadpConstraintSystem {
            num_variables,
            constraints,
        },
        blind_offset,
        opening_projection_offset,
        h_projection_offset,
        params: H12DaleoParams {
            pack_d: pack_d as u16,
            commit_rows: H12_DALEO_COMMIT_ROWS as u16,
            blind_len: H12_DALEO_BLIND_LEN as u16,
        },
        matrix_seed,
    })
}

impl H12DaleoCompiledConstraintSystem {
    pub fn local_view_from_pi0(&self, pi0: &[F257]) -> Result<Vec<F257>, String> {
        let mut out = Vec::with_capacity(self.local_view_len);
        for &pi_idx in &self.z_w_positions {
            let v = *pi0
                .get(pi_idx)
                .ok_or_else(|| format!("H12 DALEO z_w pi index out of range: {pi_idx}"))?;
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
                        .ok_or_else(|| "H12 DALEO block start overflow".to_string())?,
                )
                .ok_or_else(|| "H12 DALEO block start overflow".to_string())?;
            for &pos in positions {
                let pi_idx = start
                    .checked_add(pos)
                    .ok_or_else(|| "H12 DALEO block position overflow".to_string())?;
                let v = *pi0
                    .get(pi_idx)
                    .ok_or_else(|| format!("H12 DALEO pi index out of range: {pi_idx}"))?;
                out.push(v);
            }
        }
        Ok(out)
    }

    pub fn local_view_from_openings(&self, pi0_len: usize, openings: &[H12PiBlockOpening]) -> Result<Vec<F257>, String> {
        let pack_d = self.params.pack_d as usize;
        let mut by_block = BTreeMap::<usize, Vec<F257>>::new();
        for opening in openings {
            let block_idx = opening.block_index as usize;
            if by_block.contains_key(&block_idx) {
                return Err(format!("H12 DALEO duplicate opening for block {block_idx}"));
            }
            let expected_len = block_len_for_index(pi0_len, pack_d, block_idx);
            if opening.values.len() != expected_len {
                return Err(format!(
                    "H12 DALEO opening length mismatch: block={} got={} expected={}",
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
                .ok_or_else(|| format!("H12 DALEO missing opening for block {block_idx}"))?;
            let v = *vals
                .get(pos)
                .ok_or_else(|| format!("H12 DALEO opening too short for pi_idx={pi_idx}"))?;
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
                        .ok_or_else(|| "H12 DALEO block start overflow".to_string())?,
                )
                .ok_or_else(|| "H12 DALEO block start overflow".to_string())?;
            for &rel in positions {
                let pi_idx = start
                    .checked_add(rel)
                    .ok_or_else(|| "H12 DALEO block position overflow".to_string())?;
                let packed_block = pi_idx / pack_d;
                let pos = pi_idx % pack_d;
                let vals = by_block
                    .get(&packed_block)
                    .ok_or_else(|| format!("H12 DALEO missing opening for block {packed_block}"))?;
                let v = *vals
                    .get(pos)
                    .ok_or_else(|| format!("H12 DALEO opening too short for pi_idx={pi_idx}"))?;
                out.push(v);
            }
        }
        Ok(out)
    }
}

pub fn prove_daleo_from_local_view_with_surfaces<R: RngCore>(
    compiled: &H12DaleoCompiledConstraintSystem,
    surfaces: &[Theorem43AlvoLocalCheckSurface<F257>],
    logical_rows_by_block: &BTreeMap<usize, Vec<u16>>,
    local_view: &[F257],
    rng: &mut R,
) -> Result<H12DaleoProof, String> {
    if local_view.len() != compiled.local_view_len {
        return Err(format!(
            "H12 DALEO local-view length mismatch: got={} expected={}",
            local_view.len(),
            compiled.local_view_len
        ));
    }
    let blind_values: Vec<F257> = (0..compiled.params.blind_len as usize)
        .map(|_| random_f257(rng))
        .collect();
    let ajtai_commitment_rows =
        ajtai_commitment_rows(compiled, local_view, blind_values.as_slice())?;
    let commitment_root = daleo_commitment_root(ajtai_commitment_rows.as_slice());
    let designated_challenge_u16 =
        derive_daleo_designated_challenge(&compiled.matrix_seed, &commitment_root);
    let opening_projection_residuals = opening_projection_residuals(
        designated_challenge_u16.as_slice(),
        ajtai_commitment_rows.as_slice(),
        ajtai_commitment_rows.as_slice(),
    )?;

    // Capture full streamed rows for touched logical blocks, in compiled-order.
    let mut h_opened_logical_rows = Vec::with_capacity(compiled.logical_block_indices.len());
    for &logical_block in &compiled.logical_block_indices {
        let row = logical_rows_by_block.get(&logical_block).ok_or_else(|| {
            format!(
                "H12 DALEO missing streamed logical row for block {}",
                logical_block
            )
        })?;
        h_opened_logical_rows.push(row.iter().map(|&v| v % 257).collect());
    }
    let logical_rows_map =
        logical_rows_map_for_compiled(compiled, h_opened_logical_rows.as_slice())?;
    ensure_local_view_matches_logical_rows(compiled, local_view, &logical_rows_map)?;
    let h_projection_residuals = h_projection_residuals_from_surfaces_with_rows(
        designated_challenge_u16.as_slice(),
        surfaces,
        &logical_rows_map,
    )?;

    Ok(H12DaleoProof {
        params: compiled.params.clone(),
        local_view_values: local_view.iter().copied().map(f257_to_u16).collect(),
        blind_values: blind_values.into_iter().map(f257_to_u16).collect(),
        opening_projection_residuals,
        h_projection_residuals,
        h_opened_logical_rows,
        ajtai_commitment_rows,
        designated_challenge: designated_challenge_u16,
    })
}

pub fn witness_from_daleo_proof(
    compiled: &H12DaleoCompiledConstraintSystem,
    proof: &H12DaleoProof,
) -> Result<Vec<F257>, String> {
    if proof.params.pack_d != compiled.params.pack_d
        || proof.params.commit_rows != compiled.params.commit_rows
        || proof.params.blind_len != compiled.params.blind_len
    {
        return Err("H12 DALEO proof params mismatch".to_string());
    }
    if proof.local_view_values.len() != compiled.local_view_len {
        return Err(format!(
            "H12 DALEO proof local-view length mismatch: got={} expected={}",
            proof.local_view_values.len(),
            compiled.local_view_len
        ));
    }
    if proof.blind_values.len() != compiled.params.blind_len as usize {
        return Err(format!(
            "H12 DALEO proof blind length mismatch: got={} expected={}",
            proof.blind_values.len(),
            compiled.params.blind_len
        ));
    }
    if proof.opening_projection_residuals.len() != H12_DALEO_EXT16_PROJECTION_CHECKS {
        return Err(format!(
            "H12 DALEO proof opening projection length mismatch: got={} expected={}",
            proof.opening_projection_residuals.len(),
            H12_DALEO_EXT16_PROJECTION_CHECKS
        ));
    }
    if proof.h_projection_residuals.len() != H12_DALEO_EXT16_PROJECTION_CHECKS {
        return Err(format!(
            "H12 DALEO proof H projection length mismatch: got={} expected={}",
            proof.h_projection_residuals.len(),
            H12_DALEO_EXT16_PROJECTION_CHECKS
        ));
    }
    if proof.designated_challenge.len() != H12_DALEO_DESIGNATED_CHALLENGE_WORDS {
        return Err(format!(
            "H12 DALEO proof designated-challenge length mismatch: got={} expected={}",
            proof.designated_challenge.len(),
            H12_DALEO_DESIGNATED_CHALLENGE_WORDS
        ));
    }
    verify_daleo_ajtai_commitment(compiled, proof)?;
    let mut out = Vec::with_capacity(compiled.cs.num_variables);
    out.extend(
        proof.local_view_values
            .iter()
            .copied()
            .map(|x| F257::from((x % 257) as u64)),
    );
    out.extend(
        proof.blind_values
            .iter()
            .copied()
            .map(|x| F257::from((x % 257) as u64)),
    );
    out.extend(
        proof.opening_projection_residuals
            .iter()
            .copied()
            .map(|x| F257::from((x % 257) as u64)),
    );
    out.extend(
        proof.h_projection_residuals
            .iter()
            .copied()
            .map(|x| F257::from((x % 257) as u64)),
    );
    Ok(out)
}

pub fn verify_daleo_ajtai_commitment(
    compiled: &H12DaleoCompiledConstraintSystem,
    proof: &H12DaleoProof,
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
    let expected = ajtai_commitment_rows(compiled, local_view.as_slice(), blind_values.as_slice())?;
    let commitment_root = daleo_commitment_root(proof.ajtai_commitment_rows.as_slice());
    let expected_designated =
        derive_daleo_designated_challenge(&compiled.matrix_seed, &commitment_root);
    if expected_designated != proof.designated_challenge {
        return Err("H12 DALEO designated challenge mismatch".to_string());
    }
    let expected_opening_projection = opening_projection_residuals(
        proof.designated_challenge.as_slice(),
        expected.as_slice(),
        proof.ajtai_commitment_rows.as_slice(),
    )?;
    if expected_opening_projection != proof.opening_projection_residuals {
        return Err("H12 DALEO opening projection mismatch".to_string());
    }
    let logical_rows_map =
        logical_rows_map_for_compiled(compiled, proof.h_opened_logical_rows.as_slice())?;
    ensure_local_view_matches_logical_rows(compiled, local_view.as_slice(), &logical_rows_map)?;
    Ok(())
}

pub fn verify_daleo_h_projection_from_surfaces(
    compiled: &H12DaleoCompiledConstraintSystem,
    proof: &H12DaleoProof,
    surfaces: &[Theorem43AlvoLocalCheckSurface<F257>],
) -> Result<(), String> {
    let local_view: Vec<F257> = proof
        .local_view_values
        .iter()
        .copied()
        .map(|x| F257::from((x % 257) as u64))
        .collect();
    let logical_rows_map =
        logical_rows_map_for_compiled(compiled, proof.h_opened_logical_rows.as_slice())?;
    ensure_local_view_matches_logical_rows(compiled, local_view.as_slice(), &logical_rows_map)?;
    let expected_h_projection = h_projection_residuals_from_surfaces_with_rows(
        proof.designated_challenge.as_slice(),
        surfaces,
        &logical_rows_map,
    )?;
    if expected_h_projection != proof.h_projection_residuals {
        return Err("H12 DALEO strict H_j projection mismatch".to_string());
    }
    Ok(())
}

fn ajtai_commitment_rows(
    compiled: &H12DaleoCompiledConstraintSystem,
    local_view: &[F257],
    blind_values: &[F257],
) -> Result<Vec<Vec<u64>>, String> {
    if local_view.len() != compiled.local_view_len {
        return Err("H12 DALEO local-view width mismatch".to_string());
    }
    if blind_values.len() != compiled.params.blind_len as usize {
        return Err("H12 DALEO blind width mismatch".to_string());
    }
    let mut witness = Vec::with_capacity(blind_values.len() + local_view.len());
    witness.extend(
        local_view
            .iter()
            .map(|v| <AjtaiRing as PolyRing>::BaseRing::from(f257_to_u16(*v) as u128)),
    );
    witness.extend(
        blind_values
            .iter()
            .map(|v| <AjtaiRing as PolyRing>::BaseRing::from(f257_to_u16(*v) as u128)),
    );
    let stage1_root = compiled.matrix_seed;
    let scheme = daleo_ajtai_scheme(&stage1_root, witness.len());
    let commitment = scheme
        .commit_const_coeff_base_fast(witness.as_slice())
        .map_err(|e| format!("H12 DALEO Ajtai commit failed: {e:?}"))?;
    Ok(commitment
        .as_ref()
        .iter()
        .map(|row| row.coeffs().iter().map(|c| c.into_bigint().as_ref()[0]).collect())
        .collect())
}

fn random_f257<R: RngCore>(rng: &mut R) -> F257 {
    let mut buf = [0u8; 2];
    rng.fill_bytes(&mut buf);
    F257::from((u16::from_le_bytes(buf) % 257) as u64)
}

#[cfg(test)]
mod tests {
    use super::*;
    use dpp::dr1cs_flpcp::Dr1csProofLayoutInfo;
    use dpp::theorem43::{
        Theorem43AlvoLocalCheckSurface, Theorem43CapsuleLocalCheckSurface, Theorem43Coins,
    };
    use rand::SeedableRng;
    use rand_chacha::ChaCha20Rng;

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
    fn test_daleo_roundtrip_witness_satisfies_compiled_relation() {
        let surfaces = vec![dummy_surface()];
        let compiled = compile_daleo_constraint_system(&surfaces, &[9u8; 32], 64).expect("compile");
        let pi0 = vec![F257::from(2u64), F257::from(3u64)];
        let local_view = compiled
            .local_view_from_pi0(pi0.as_slice())
            .expect("local view");
        let logical_rows = std::collections::BTreeMap::<usize, Vec<u16>>::new();
        let mut rng = ChaCha20Rng::from_seed([7u8; 32]);
        let proof = prove_daleo_from_local_view_with_surfaces(
            &compiled,
            surfaces.as_slice(),
            &logical_rows,
            local_view.as_slice(),
            &mut rng,
        )
        .expect("prove");
        let witness = witness_from_daleo_proof(&compiled, &proof).expect("witness");
        compiled.cs.check_witness(witness.as_slice()).expect("witness checks");
    }
}
