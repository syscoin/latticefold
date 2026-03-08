//! H12 `R_cap` helpers.
//!
//! This module is intentionally lightweight:
//! - it does **not** implement AADP itself
//! - it estimates the outer capsule witness/gate budget from exported local-check surfaces
//! - it provides the canonical "current logical lock -> capsule checks" mapping

use std::collections::{BTreeMap, BTreeSet};
use std::io::{Read, Write};

use ark_ff::{Field, PrimeField};
use ark_std::Zero;
use dpp::theorem43::{Theorem43AlvoLocalCheckSurface, Theorem43CapsuleLocalCheckSurface};
use latticefold::transcript::poseidon::F257;
use rand::RngCore;
use sha2::Digest;

use crate::aadp_we::{
    aadp_encrypt_bytes, AadpByteCiphertext, AadpConstraintSystem, AadpLinearForm, AadpMulConstraint,
};
use crate::aadp_we_ext::{aadp_encrypt_ext16_seed, AadpCiphertextExt16};
use crate::f257_ext16::F257Ext16;
use crate::h12_alvo_daleo::{compile_daleo_constraint_system, H12DaleoCompiledConstraintSystem};
use crate::lockable_ringlwe::RingLweLockArtifact;

/// Current packed `pi0` block size used by `lockable_ringlwe`.
pub const H12_RCAP_PACK_D: usize = 64;
pub const H12_RCAP_SEED_BYTES: usize = 16;
const H12_F257_MOD: u16 = 257;

/// AADP cost model from `eprint 2026/175`, Section 4.10.
///
/// Rough model:
///
/// `size ~= v * (2g + 1)^2 * |F|`
///
/// where:
/// - `v` is the outer relation variable count
/// - `g` is the outer relation gate count
/// - `|F|` is the byte width of one field element in the target AADP field
pub fn aadp_ciphertext_size_bytes(v: usize, g: usize, field_bytes: usize) -> u128 {
    let v = v as u128;
    let g_term = (2usize.saturating_mul(g).saturating_add(1)) as u128;
    let field_bytes = field_bytes as u128;
    v.saturating_mul(g_term.saturating_mul(g_term))
        .saturating_mul(field_bytes)
}

/// Current "one logical lock -> capsule checks" mapping.
///
/// Each rep already exposes one `(anchor_block_id, rep_id)` pair, which is the natural
/// statement-fixed unit for a candidate H12 capsule local check.
pub fn capsule_checks_from_logical_lock<F: PrimeField>(
    reps: &[RingLweLockArtifact<F>],
) -> Vec<(usize, u64)> {
    reps.iter()
        .map(|rep| (rep.anchor_block_id as usize, rep.rep_id))
        .collect()
}

/// Decoupled outer-capsule mapping: select only the first `r_cap` current reps of a logical lock.
///
/// This is the recommended experimental branch for H12:
/// - inner H11 may use larger `R`
/// - outer capsule starts with `R_cap = 1`
///
/// For now we conservatively reuse existing rep schedules rather than synthesizing extra capsule-only
/// schedules. Therefore `r_cap` is capped at `reps.len()`.
pub fn capsule_checks_from_logical_lock_with_r_cap<F: PrimeField>(
    reps: &[RingLweLockArtifact<F>],
    r_cap: usize,
) -> Vec<(usize, u64)> {
    reps.iter()
        .take(r_cap.max(1).min(reps.len()))
        .map(|rep| (rep.anchor_block_id as usize, rep.rep_id))
        .collect()
}

#[derive(Clone, Debug, Default)]
pub struct H12RCapScheduleEstimate {
    pub check_count: usize,
    pub g_cap: usize,
    pub touched_pi_positions_exact: usize,
    pub touched_pi_positions_conservative: usize,
    pub touched_packed_pi_blocks_exact: usize,
    pub touched_packed_pi_blocks_conservative: usize,
    pub aadp_size_exact_bytes: u128,
    pub aadp_size_conservative_bytes: u128,
}

/// Estimate the H12 outer capsule size from exported local-check surfaces.
///
/// Two witness regimes are reported:
///
/// - exact:
///   uses the exported per-position `q1/q2/q3` witness footprint
/// - conservative:
///   includes the whole selected `w_eval` block whenever the surface indicates the current backend
///   should be treated conservatively
pub fn estimate_capsule_schedule<F: PrimeField>(
    surfaces: &[Theorem43CapsuleLocalCheckSurface<F>],
    pack_d: usize,
    field_bytes: usize,
) -> H12RCapScheduleEstimate {
    let mut exact_pos = BTreeSet::<usize>::new();
    let mut conservative_pos = BTreeSet::<usize>::new();
    let mut exact_blocks = BTreeSet::<usize>::new();
    let mut conservative_blocks = BTreeSet::<usize>::new();

    for s in surfaces {
        for pi_idx in s.touched_pi_positions(false) {
            exact_pos.insert(pi_idx);
            exact_blocks.insert(pi_idx / pack_d.max(1));
        }
        let conservative_full_block = s.requires_dense_q3_w_eval;
        for pi_idx in s.touched_pi_positions(conservative_full_block) {
            conservative_pos.insert(pi_idx);
            conservative_blocks.insert(pi_idx / pack_d.max(1));
        }
    }

    let g_cap = surfaces.len();
    H12RCapScheduleEstimate {
        check_count: surfaces.len(),
        g_cap,
        touched_pi_positions_exact: exact_pos.len(),
        touched_pi_positions_conservative: conservative_pos.len(),
        touched_packed_pi_blocks_exact: exact_blocks.len(),
        touched_packed_pi_blocks_conservative: conservative_blocks.len(),
        aadp_size_exact_bytes: aadp_ciphertext_size_bytes(exact_pos.len(), g_cap, field_bytes),
        aadp_size_conservative_bytes: aadp_ciphertext_size_bytes(
            conservative_pos.len(),
            g_cap,
            field_bytes,
        ),
    }
}

/// Estimate the H12 outer capsule size for the ALVO block-opening witness layout.
///
/// Unlike `estimate_capsule_schedule`, this counts the witness as the union of conservative opened
/// packed blocks, because the stage-2 ALVO witness is reconstructed from verified block openings.
pub fn estimate_alvo_schedule<F: PrimeField>(
    surfaces: &[Theorem43AlvoLocalCheckSurface<F>],
    pack_d: usize,
    field_bytes: usize,
) -> Result<H12RCapScheduleEstimate, String> {
    if surfaces.is_empty() {
        return Ok(H12RCapScheduleEstimate::default());
    }
    let pi0_len = surfaces[0].proof_layout.pi0_len;
    let mut exact_pos = BTreeSet::<usize>::new();
    let mut conservative_pos = BTreeSet::<usize>::new();
    let mut exact_blocks = BTreeSet::<usize>::new();
    let mut conservative_blocks = BTreeSet::<usize>::new();
    for s in surfaces {
        for &pi_idx in &s.touched_pi_positions_exact {
            exact_pos.insert(pi_idx);
        }
        for &pi_idx in &s.touched_pi_positions_conservative {
            conservative_pos.insert(pi_idx);
        }
        for block in s.packed_pi_blocks_exact(pack_d) {
            exact_blocks.insert(block);
        }
        for block in s.packed_pi_blocks_conservative(pack_d) {
            conservative_blocks.insert(block);
        }
    }
    let conservative_opened_values: usize = conservative_blocks
        .iter()
        .map(|&block_idx| {
            let start = block_idx.saturating_mul(pack_d);
            if start >= pi0_len {
                0
            } else {
                (pi0_len - start).min(pack_d)
            }
        })
        .sum();
    let exact_opened_values: usize = exact_blocks
        .iter()
        .map(|&block_idx| {
            let start = block_idx.saturating_mul(pack_d);
            if start >= pi0_len {
                0
            } else {
                (pi0_len - start).min(pack_d)
            }
        })
        .sum();
    let g_cap = surfaces.len();
    Ok(H12RCapScheduleEstimate {
        check_count: surfaces.len(),
        g_cap,
        touched_pi_positions_exact: exact_pos.len(),
        touched_pi_positions_conservative: conservative_pos.len(),
        touched_packed_pi_blocks_exact: exact_blocks.len(),
        touched_packed_pi_blocks_conservative: conservative_blocks.len(),
        aadp_size_exact_bytes: aadp_ciphertext_size_bytes(exact_opened_values, g_cap, field_bytes),
        aadp_size_conservative_bytes: aadp_ciphertext_size_bytes(
            conservative_opened_values,
            g_cap,
            field_bytes,
        ),
    })
}

#[derive(Clone, Debug)]
pub struct H12SeedEnvelopeF257 {
    pub r_cap_reps: u16,
    pub seed_capsule: AadpByteCiphertext<F257>,
    pub hidden_nonce: [u8; 12],
    pub hidden_ct: Vec<u8>,
}

#[derive(Clone, Debug)]
pub struct H12SeedEnvelopeExt16 {
    pub r_cap_reps: u16,
    pub seed_capsule: AadpCiphertextExt16,
    pub hidden_nonce: [u8; 12],
    pub hidden_ct: Vec<u8>,
}

#[derive(Clone, Debug)]
pub enum H12SeedEnvelope {
    ByteF257(H12SeedEnvelopeF257),
    Ext16(H12SeedEnvelopeExt16),
}

impl H12SeedEnvelope {
    pub fn r_cap_reps(&self) -> u16 {
        match self {
            H12SeedEnvelope::ByteF257(env) => env.r_cap_reps,
            H12SeedEnvelope::Ext16(env) => env.r_cap_reps,
        }
    }
}

#[derive(Clone, Debug)]
pub struct H12CompiledConstraintSystem {
    pub touched_pi_positions: Vec<usize>,
    pub cs: AadpConstraintSystem<F257>,
}

#[derive(Clone, Debug)]
pub struct H12AlvoSeedConstraintSystem {
    pub compiled: H12DaleoCompiledConstraintSystem,
}

/// Compile the current exact per-lock capsule relation over `F257`.
///
/// This is the current H12-lite experimental branch:
/// - exact exported local witness surface
/// - one `MulEq`-style constraint per selected capsule check
/// - no extra witness compression beyond the exported exact local support
pub fn compile_exact_constraint_system(
    surfaces: &[Theorem43CapsuleLocalCheckSurface<F257>],
) -> H12CompiledConstraintSystem {
    let mut touched = BTreeSet::<usize>::new();
    for s in surfaces {
        for pi_idx in s.touched_pi_positions(false) {
            touched.insert(pi_idx);
        }
    }
    let touched_pi_positions: Vec<usize> = touched.into_iter().collect();
    let pos_to_var: BTreeMap<usize, usize> = touched_pi_positions
        .iter()
        .enumerate()
        .map(|(i, &pi_idx)| (pi_idx, i))
        .collect();

    fn mk_form(
        constant: F257,
        pi_terms: &[(usize, F257)],
        w_terms: &[(usize, F257)],
        w_eval_block_pi_offset: usize,
        pos_to_var: &BTreeMap<usize, usize>,
    ) -> AadpLinearForm<F257> {
        let mut terms = Vec::with_capacity(pi_terms.len() + w_terms.len());
        for &(pi_idx, coeff) in pi_terms {
            if coeff.is_zero() {
                continue;
            }
            if let Some(&var_idx) = pos_to_var.get(&pi_idx) {
                terms.push((var_idx, coeff));
            }
        }
        for &(pos, coeff) in w_terms {
            if coeff.is_zero() {
                continue;
            }
            let pi_idx = w_eval_block_pi_offset + pos;
            if let Some(&var_idx) = pos_to_var.get(&pi_idx) {
                terms.push((var_idx, coeff));
            }
        }
        AadpLinearForm { constant, terms }
    }

    let mut constraints = Vec::with_capacity(surfaces.len());
    for s in surfaces {
        let alpha = mk_form(
            s.q1_x_dot,
            s.q1_pi_terms.as_slice(),
            s.q1_w_terms.as_slice(),
            s.w_eval_block_pi_offset,
            &pos_to_var,
        );
        let beta = mk_form(
            s.q2_x_dot,
            s.q2_pi_terms.as_slice(),
            s.q2_w_terms.as_slice(),
            s.w_eval_block_pi_offset,
            &pos_to_var,
        );
        let gamma = mk_form(
            s.q3_x_dot_sparse,
            s.q3_pi_sparse_terms.as_slice(),
            s.q3_w_terms.as_slice(),
            s.w_eval_block_pi_offset,
            &pos_to_var,
        );
        constraints.push(AadpMulConstraint {
            a: alpha,
            b: beta,
            c: gamma,
            d: AadpLinearForm {
                constant: F257::ONE,
                terms: Vec::new(),
            },
        });
    }

    let num_variables = touched_pi_positions.len();
    H12CompiledConstraintSystem {
        touched_pi_positions,
        cs: AadpConstraintSystem {
            num_variables,
            constraints,
        },
    }
}

pub fn compile_alvo_seed_constraint_system(
    surfaces: &[Theorem43AlvoLocalCheckSurface<F257>],
    stage1_root: &[u8; 32],
) -> Result<H12AlvoSeedConstraintSystem, String> {
    Ok(H12AlvoSeedConstraintSystem {
        compiled: compile_daleo_constraint_system(surfaces, stage1_root, H12_RCAP_PACK_D)?,
    })
}

fn derive_seed_key(seed_bytes: &[u8; H12_RCAP_SEED_BYTES]) -> [u8; 32] {
    let mut h = sha2::Sha256::new();
    h.update(b"LFP_H12_SEED_KEY_V1");
    h.update(seed_bytes);
    h.finalize().into()
}

fn derive_seed_key_ext16(seed: &F257Ext16) -> [u8; 32] {
    let mut h = sha2::Sha256::new();
    h.update(b"LFP_H12_EXT16_SEED_KEY_V1");
    h.update(seed.to_bytes_fixed());
    h.finalize().into()
}

fn sha256_32(chunks: &[&[u8]]) -> [u8; 32] {
    let mut h = sha2::Sha256::new();
    for c in chunks {
        h.update(c);
    }
    h.finalize().into()
}

fn xor_stream_encrypt(key: &[u8; 32], nonce: &[u8; 12], data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    let mut ctr = 0u32;
    let mut pos = 0usize;
    while pos < data.len() {
        let block = sha256_32(&[b"LFP_H12_XOR_STREAM_V1", key, nonce, &ctr.to_le_bytes()]);
        let take = (data.len() - pos).min(32);
        for j in 0..take {
            out.push(data[pos + j] ^ block[j]);
        }
        pos += take;
        ctr += 1;
    }
    out
}

fn xor_stream_decrypt(key: &[u8; 32], nonce: &[u8; 12], ct: &[u8]) -> Vec<u8> {
    xor_stream_encrypt(key, nonce, ct)
}

pub fn encrypt_seed_envelope<R: RngCore>(
    surfaces: &[Theorem43CapsuleLocalCheckSurface<F257>],
    r_cap_reps: u16,
    hidden_plain: &[u8],
    rng: &mut R,
) -> Result<H12SeedEnvelopeF257, String> {
    let compiled = compile_exact_constraint_system(surfaces);
    let mut seed = [0u8; H12_RCAP_SEED_BYTES];
    rng.fill_bytes(&mut seed);
    let seed_capsule = aadp_encrypt_bytes(&compiled.cs, &seed, rng)?;
    let mut hidden_nonce = [0u8; 12];
    rng.fill_bytes(&mut hidden_nonce);
    let hidden_ct = xor_stream_encrypt(&derive_seed_key(&seed), &hidden_nonce, hidden_plain);
    Ok(H12SeedEnvelopeF257 {
        r_cap_reps,
        seed_capsule,
        hidden_nonce,
        hidden_ct,
    })
}

pub fn decrypt_seed_envelope(
    envelope: &H12SeedEnvelopeF257,
    surfaces: &[Theorem43CapsuleLocalCheckSurface<F257>],
    witness_values: &[F257],
) -> Result<Vec<u8>, String> {
    let compiled = compile_exact_constraint_system(surfaces);
    if witness_values.len() != compiled.cs.num_variables {
        return Err(format!(
            "H12 witness length mismatch: got={} expected={}",
            witness_values.len(),
            compiled.cs.num_variables
        ));
    }
    let seed = envelope.seed_capsule.decrypt_bytes(witness_values)?;
    if seed.len() != H12_RCAP_SEED_BYTES {
        return Err("H12 decrypted seed length mismatch".to_string());
    }
    let mut seed_arr = [0u8; H12_RCAP_SEED_BYTES];
    seed_arr.copy_from_slice(seed.as_slice());
    Ok(xor_stream_decrypt(
        &derive_seed_key(&seed_arr),
        &envelope.hidden_nonce,
        envelope.hidden_ct.as_slice(),
    ))
}

pub fn encrypt_seed_envelope_ext16<R: RngCore>(
    surfaces: &[Theorem43CapsuleLocalCheckSurface<F257>],
    r_cap_reps: u16,
    hidden_plain: &[u8],
    rng: &mut R,
) -> Result<H12SeedEnvelopeExt16, String> {
    let compiled = compile_exact_constraint_system(surfaces);
    let mut seed = [0u8; H12_RCAP_SEED_BYTES];
    rng.fill_bytes(&mut seed);
    let seed_capsule = aadp_encrypt_ext16_seed(&compiled.cs, seed, rng)?;
    let seed_field = F257Ext16::from_u128_seed(u128::from_le_bytes(seed));
    let mut hidden_nonce = [0u8; 12];
    rng.fill_bytes(&mut hidden_nonce);
    let hidden_ct = xor_stream_encrypt(&derive_seed_key_ext16(&seed_field), &hidden_nonce, hidden_plain);
    Ok(H12SeedEnvelopeExt16 {
        r_cap_reps,
        seed_capsule,
        hidden_nonce,
        hidden_ct,
    })
}

pub fn decrypt_seed_envelope_ext16(
    envelope: &H12SeedEnvelopeExt16,
    surfaces: &[Theorem43CapsuleLocalCheckSurface<F257>],
    witness_values: &[F257],
) -> Result<Vec<u8>, String> {
    let compiled = compile_exact_constraint_system(surfaces);
    if witness_values.len() != compiled.cs.num_variables {
        return Err(format!(
            "H12-ext witness length mismatch: got={} expected={}",
            witness_values.len(),
            compiled.cs.num_variables
        ));
    }
    let seed_bytes = envelope.seed_capsule.decrypt_seed(witness_values)?;
    let seed_field = F257Ext16::from_u128_seed(u128::from_le_bytes(seed_bytes));
    Ok(xor_stream_decrypt(
        &derive_seed_key_ext16(&seed_field),
        &envelope.hidden_nonce,
        envelope.hidden_ct.as_slice(),
    ))
}

pub fn encrypt_seed_envelope_alvo<R: RngCore>(
    surfaces: &[Theorem43AlvoLocalCheckSurface<F257>],
    stage1_root: &[u8; 32],
    r_cap_reps: u16,
    hidden_plain: &[u8],
    rng: &mut R,
) -> Result<H12SeedEnvelopeF257, String> {
    let profile = std::env::var("LFP_PROFILE_H12_SEED")
        .ok()
        .is_some_and(|v| v != "0");
    let t_total = std::time::Instant::now();
    let t_compile = std::time::Instant::now();
    let compiled = compile_alvo_seed_constraint_system(surfaces, stage1_root)?;
    if profile {
        eprintln!(
            "[h12:profile] encrypt_seed_envelope_alvo compile elapsed={:?} vars={} gates={}",
            t_compile.elapsed(),
            compiled.compiled.cs.num_variables,
            compiled.compiled.cs.gate_count()
        );
    }
    let mut seed = [0u8; H12_RCAP_SEED_BYTES];
    rng.fill_bytes(&mut seed);
    let t_encrypt = std::time::Instant::now();
    let seed_capsule = aadp_encrypt_bytes(&compiled.compiled.cs, &seed, rng)?;
    if profile {
        eprintln!(
            "[h12:profile] encrypt_seed_envelope_alvo aadp_encrypt elapsed={:?}",
            t_encrypt.elapsed()
        );
    }
    let mut hidden_nonce = [0u8; 12];
    rng.fill_bytes(&mut hidden_nonce);
    let hidden_ct = xor_stream_encrypt(&derive_seed_key(&seed), &hidden_nonce, hidden_plain);
    if profile {
        eprintln!(
            "[h12:profile] encrypt_seed_envelope_alvo total elapsed={:?}",
            t_total.elapsed()
        );
    }
    Ok(H12SeedEnvelopeF257 {
        r_cap_reps,
        seed_capsule,
        hidden_nonce,
        hidden_ct,
    })
}

pub fn decrypt_seed_envelope_alvo(
    envelope: &H12SeedEnvelopeF257,
    surfaces: &[Theorem43AlvoLocalCheckSurface<F257>],
    stage1_root: &[u8; 32],
    witness_values: &[F257],
) -> Result<Vec<u8>, String> {
    let compiled = compile_alvo_seed_constraint_system(surfaces, stage1_root)?;
    compiled.compiled.cs.check_witness(witness_values)?;
    let seed = envelope.seed_capsule.decrypt_bytes(witness_values)?;
    if seed.len() != H12_RCAP_SEED_BYTES {
        return Err("H12-ALVO decrypted seed length mismatch".to_string());
    }
    let mut seed_arr = [0u8; H12_RCAP_SEED_BYTES];
    seed_arr.copy_from_slice(seed.as_slice());
    Ok(xor_stream_decrypt(
        &derive_seed_key(&seed_arr),
        &envelope.hidden_nonce,
        envelope.hidden_ct.as_slice(),
    ))
}

pub fn encrypt_seed_envelope_alvo_ext16<R: RngCore>(
    surfaces: &[Theorem43AlvoLocalCheckSurface<F257>],
    stage1_root: &[u8; 32],
    r_cap_reps: u16,
    hidden_plain: &[u8],
    rng: &mut R,
) -> Result<H12SeedEnvelopeExt16, String> {
    let profile = std::env::var("LFP_PROFILE_H12_SEED")
        .ok()
        .is_some_and(|v| v != "0");
    let t_total = std::time::Instant::now();
    let t_compile = std::time::Instant::now();
    let compiled = compile_alvo_seed_constraint_system(surfaces, stage1_root)?;
    if profile {
        eprintln!(
            "[h12:profile] encrypt_seed_envelope_alvo_ext16 compile elapsed={:?} vars={} gates={}",
            t_compile.elapsed(),
            compiled.compiled.cs.num_variables,
            compiled.compiled.cs.gate_count()
        );
    }
    let mut seed = [0u8; H12_RCAP_SEED_BYTES];
    rng.fill_bytes(&mut seed);
    let t_encrypt = std::time::Instant::now();
    let seed_capsule = aadp_encrypt_ext16_seed(&compiled.compiled.cs, seed, rng)?;
    if profile {
        eprintln!(
            "[h12:profile] encrypt_seed_envelope_alvo_ext16 aadp_encrypt elapsed={:?}",
            t_encrypt.elapsed()
        );
    }
    let seed_field = F257Ext16::from_u128_seed(u128::from_le_bytes(seed));
    let mut hidden_nonce = [0u8; 12];
    rng.fill_bytes(&mut hidden_nonce);
    let hidden_ct = xor_stream_encrypt(
        &derive_seed_key_ext16(&seed_field),
        &hidden_nonce,
        hidden_plain,
    );
    if profile {
        eprintln!(
            "[h12:profile] encrypt_seed_envelope_alvo_ext16 total elapsed={:?}",
            t_total.elapsed()
        );
    }
    Ok(H12SeedEnvelopeExt16 {
        r_cap_reps,
        seed_capsule,
        hidden_nonce,
        hidden_ct,
    })
}

pub fn decrypt_seed_envelope_alvo_ext16(
    envelope: &H12SeedEnvelopeExt16,
    surfaces: &[Theorem43AlvoLocalCheckSurface<F257>],
    stage1_root: &[u8; 32],
    witness_values: &[F257],
) -> Result<Vec<u8>, String> {
    let compiled = compile_alvo_seed_constraint_system(surfaces, stage1_root)?;
    compiled.compiled.cs.check_witness(witness_values)?;
    let seed_bytes = envelope.seed_capsule.decrypt_seed(witness_values)?;
    let seed_field = F257Ext16::from_u128_seed(u128::from_le_bytes(seed_bytes));
    Ok(xor_stream_decrypt(
        &derive_seed_key_ext16(&seed_field),
        &envelope.hidden_nonce,
        envelope.hidden_ct.as_slice(),
    ))
}

pub fn decrypt_seed_envelope_any(
    envelope: &H12SeedEnvelope,
    surfaces: &[Theorem43CapsuleLocalCheckSurface<F257>],
    witness_values: &[F257],
) -> Result<Vec<u8>, String> {
    match envelope {
        H12SeedEnvelope::ByteF257(env) => decrypt_seed_envelope(env, surfaces, witness_values),
        H12SeedEnvelope::Ext16(env) => decrypt_seed_envelope_ext16(env, surfaces, witness_values),
    }
}

pub fn hidden_plain_from_ct_ubits(reps_ct_ubits: &[[u8; 32]]) -> Vec<u8> {
    let mut out = Vec::with_capacity(reps_ct_ubits.len() * 32);
    for ct in reps_ct_ubits {
        out.extend_from_slice(ct);
    }
    out
}

pub fn hidden_plain_from_rep_hidden_state(reps: &[RingLweLockArtifact<F257>]) -> Vec<u8> {
    let mut out = Vec::with_capacity(reps.len() * (32 + 2 + 2 + 2));
    for rep in reps {
        out.extend_from_slice(&rep.ct_ubits);
        out.extend_from_slice(&f257_to_u16(&rep.accepting_set[0]).to_le_bytes());
        out.extend_from_slice(&f257_to_u16(&rep.accepting_set[1]).to_le_bytes());
        out.extend_from_slice(&f257_to_u16(&rep.offset).to_le_bytes());
    }
    out
}

pub fn rep_hidden_state_from_hidden_plain(
    hidden_plain: &[u8],
    reps: usize,
) -> Result<Vec<([u8; 32], [F257; 2], F257)>, String> {
    const PER_REP_BYTES: usize = 32 + 2 + 2 + 2;
    if hidden_plain.len() != reps * PER_REP_BYTES {
        return Err(format!(
            "H12 hidden rep-state length mismatch: got={} expected={}",
            hidden_plain.len(),
            reps * PER_REP_BYTES
        ));
    }
    let mut out = Vec::with_capacity(reps);
    for i in 0..reps {
        let base = i * PER_REP_BYTES;
        let mut ct = [0u8; 32];
        ct.copy_from_slice(&hidden_plain[base..base + 32]);
        let a0 = u16::from_le_bytes([hidden_plain[base + 32], hidden_plain[base + 33]]);
        let a1 = u16::from_le_bytes([hidden_plain[base + 34], hidden_plain[base + 35]]);
        let off = u16::from_le_bytes([hidden_plain[base + 36], hidden_plain[base + 37]]);
        out.push((ct, [u16_to_f257(a0), u16_to_f257(a1)], u16_to_f257(off)));
    }
    Ok(out)
}

pub struct H12Pi0SliceCollector {
    targets: Vec<usize>,
    values: Vec<F257>,
    cursor: usize,
    next_target: usize,
}

impl H12Pi0SliceCollector {
    pub fn from_surfaces(surfaces: &[Theorem43CapsuleLocalCheckSurface<F257>]) -> Self {
        let compiled = compile_exact_constraint_system(surfaces);
        let n = compiled.touched_pi_positions.len();
        Self {
            targets: compiled.touched_pi_positions,
            values: vec![F257::ZERO; n],
            cursor: 0,
            next_target: 0,
        }
    }

    pub fn absorb_chunk(&mut self, chunk: &[F257]) {
        let start = self.cursor;
        let end = start + chunk.len();
        while self.next_target < self.targets.len() {
            let target = self.targets[self.next_target];
            if target >= end {
                break;
            }
            if target >= start {
                self.values[self.next_target] = chunk[target - start];
            }
            self.next_target += 1;
        }
        self.cursor = end;
    }

    pub fn into_witness(self) -> Result<Vec<F257>, String> {
        if self.next_target != self.targets.len() {
            return Err(format!(
                "H12 collector incomplete: captured={} expected={}",
                self.next_target,
                self.targets.len()
            ));
        }
        Ok(self.values)
    }
}

#[inline]
fn f257_to_u16(f: &F257) -> u16 {
    (f.into_bigint().as_ref()[0] % 257) as u16
}

#[inline]
fn u16_to_f257(x: u16) -> F257 {
    F257::from((x % 257) as u64)
}

pub fn write_seed_envelope_f257(
    w: &mut impl Write,
    env: &H12SeedEnvelopeF257,
) -> std::io::Result<()> {
    fn write_u16(w: &mut impl Write, v: u16) -> std::io::Result<()> {
        w.write_all(&v.to_le_bytes())
    }
    fn write_u32(w: &mut impl Write, v: u32) -> std::io::Result<()> {
        w.write_all(&v.to_le_bytes())
    }
    write_u16(w, env.r_cap_reps)?;
    write_u32(w, env.seed_capsule.parts.len() as u32)?;
    for part in &env.seed_capsule.parts {
        write_u32(w, part.num_variables as u32)?;
        write_u32(w, part.dim as u32)?;
        write_u32(w, part.matrices.len() as u32)?;
        for m in &part.matrices {
            write_u32(w, m.len() as u32)?;
            for x in m {
                write_u16(w, f257_to_u16(x))?;
            }
        }
    }
    w.write_all(&env.hidden_nonce)?;
    write_u32(w, env.hidden_ct.len() as u32)?;
    w.write_all(&env.hidden_ct)?;
    Ok(())
}

pub fn write_seed_envelope(
    w: &mut impl Write,
    env: &H12SeedEnvelope,
) -> std::io::Result<()> {
    fn write_u16(w: &mut impl Write, v: u16) -> std::io::Result<()> {
        w.write_all(&v.to_le_bytes())
    }
    fn write_u32(w: &mut impl Write, v: u32) -> std::io::Result<()> {
        w.write_all(&v.to_le_bytes())
    }
    match env {
        H12SeedEnvelope::ByteF257(env) => {
            w.write_all(&[0u8])?;
            write_seed_envelope_f257(w, env)
        }
        H12SeedEnvelope::Ext16(env) => {
            w.write_all(&[1u8])?;
            write_u16(w, env.r_cap_reps)?;
            write_u32(w, env.seed_capsule.num_variables as u32)?;
            write_u32(w, env.seed_capsule.dim as u32)?;
            write_u32(w, env.seed_capsule.matrices.len() as u32)?;
            for m in &env.seed_capsule.matrices {
                write_u32(w, m.len() as u32)?;
                for x in m {
                    w.write_all(&x.to_bytes_fixed())?;
                }
            }
            w.write_all(&env.hidden_nonce)?;
            write_u32(w, env.hidden_ct.len() as u32)?;
            w.write_all(&env.hidden_ct)?;
            Ok(())
        }
    }
}

pub fn read_seed_envelope_f257(r: &mut impl Read) -> std::io::Result<H12SeedEnvelopeF257> {
    fn read_u16(r: &mut impl Read) -> std::io::Result<u16> {
        let mut b = [0u8; 2];
        r.read_exact(&mut b)?;
        Ok(u16::from_le_bytes(b))
    }
    fn read_u32(r: &mut impl Read) -> std::io::Result<u32> {
        let mut b = [0u8; 4];
        r.read_exact(&mut b)?;
        Ok(u32::from_le_bytes(b))
    }
    let r_cap_reps = read_u16(r)?;
    let part_n = read_u32(r)? as usize;
    let mut parts = Vec::with_capacity(part_n);
    for _ in 0..part_n {
        let num_variables = read_u32(r)? as usize;
        let dim = read_u32(r)? as usize;
        let m_n = read_u32(r)? as usize;
        let mut matrices = Vec::with_capacity(m_n);
        for _ in 0..m_n {
            let len = read_u32(r)? as usize;
            let mut m = Vec::with_capacity(len);
            for _ in 0..len {
                m.push(u16_to_f257(read_u16(r)?));
            }
            matrices.push(m);
        }
        parts.push(crate::aadp_we::AadpCiphertext {
            num_variables,
            dim,
            matrices,
        });
    }
    let mut hidden_nonce = [0u8; 12];
    r.read_exact(&mut hidden_nonce)?;
    let hidden_len = read_u32(r)? as usize;
    let mut hidden_ct = vec![0u8; hidden_len];
    r.read_exact(&mut hidden_ct)?;
    Ok(H12SeedEnvelopeF257 {
        r_cap_reps,
        seed_capsule: AadpByteCiphertext { parts },
        hidden_nonce,
        hidden_ct,
    })
}

pub fn read_seed_envelope(r: &mut impl Read) -> std::io::Result<H12SeedEnvelope> {
    fn read_u16(r: &mut impl Read) -> std::io::Result<u16> {
        let mut b = [0u8; 2];
        r.read_exact(&mut b)?;
        Ok(u16::from_le_bytes(b))
    }
    fn read_u32(r: &mut impl Read) -> std::io::Result<u32> {
        let mut b = [0u8; 4];
        r.read_exact(&mut b)?;
        Ok(u32::from_le_bytes(b))
    }
    let mut tag = [0u8; 1];
    r.read_exact(&mut tag)?;
    match tag[0] {
        0 => Ok(H12SeedEnvelope::ByteF257(read_seed_envelope_f257(r)?)),
        1 => {
            let r_cap_reps = read_u16(r)?;
            let num_variables = read_u32(r)? as usize;
            let dim = read_u32(r)? as usize;
            let m_n = read_u32(r)? as usize;
            let mut matrices = Vec::with_capacity(m_n);
            for _ in 0..m_n {
                let len = read_u32(r)? as usize;
                let mut m = Vec::with_capacity(len);
                for _ in 0..len {
                    let mut b = [0u8; 32];
                    r.read_exact(&mut b)?;
                    let mut coeffs = [0u16; crate::f257_ext16::F257_EXT16_DEGREE];
                    for i in 0..crate::f257_ext16::F257_EXT16_DEGREE {
                        coeffs[i] = u16::from_le_bytes([b[2 * i], b[2 * i + 1]]) % H12_F257_MOD;
                    }
                    m.push(F257Ext16 { coeffs });
                }
                matrices.push(m);
            }
            let mut hidden_nonce = [0u8; 12];
            r.read_exact(&mut hidden_nonce)?;
            let hidden_len = read_u32(r)? as usize;
            let mut hidden_ct = vec![0u8; hidden_len];
            r.read_exact(&mut hidden_ct)?;
            Ok(H12SeedEnvelope::Ext16(H12SeedEnvelopeExt16 {
                r_cap_reps,
                seed_capsule: AadpCiphertextExt16 {
                    num_variables,
                    dim,
                    matrices,
                },
                hidden_nonce,
                hidden_ct,
            }))
        }
        _ => Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "bad H12 seed envelope tag",
        )),
    }
}

