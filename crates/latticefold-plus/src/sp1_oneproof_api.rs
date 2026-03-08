//! API wrapper for the `examples/lf_plus_sp1_oneproof.rs` harness.
//!
//! This exists so downstream crates (like PVUGC) can call the SP1→LF+→WE-gate path **in-process**
//! without spawning binaries or scraping logs.
//!
//! This is research plumbing and intentionally returns only a small structured summary.

#![cfg(feature = "we_gate")]

use cyclotomic_rings::rings::{GetPoseidonParams, GoldilocksPoseidonConfig as PC};
use cyclotomic_rings::rings::GoldilocksRing64 as R;
use ark_ff::{Field, PrimeField};
use latticefold::commitment::AjtaiCommitmentScheme;
use latticefold::transcript::poseidon::F257;
use latticefold::transcript::Transcript;
use latticefold::transcript::bytes::prime_field_to_bytes_le_fixed;
use stark_rings::PolyRing;
use stark_rings_linalg::SparseMatrix;

use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;
use std::time::Instant;

use rand::{rngs::OsRng, rngs::StdRng, RngCore, SeedableRng};

use crate::h12_alvo::{
    derive_stage1_root, derive_stage2_root, digest_alvo_schedule, digest_stage1_lock_package_bytes,
    read_augmented_package, write_augmented_package, H12AlvoAnchorProjection, H12AlvoAugmentedPackage, H12AlvoClaim,
    H12AlvoLogicalLockAugmentation, H12AlvoRepSurfaceBundle,
};
use crate::h12_alvo_daleo::{prove_daleo_from_local_view, witness_from_daleo_proof};
use crate::lin::LinearizedVerify;
use crate::lockable_ringlwe::{
    RingLweLockArtifact, RingLweParams,
};
use crate::utils::maybe_print_rss;
use crate::we_statement::{
    encode_public_x, we_statement_hash_lf_plus, LFP_WE_GATE_DIGEST_V1,
};
use std::io::{Read, Write};
use rayon::prelude::*;
type BFSmall = <<R as PolyRing>::BaseRing as ark_ff::Field>::BasePrimeField;

/// Structured output for downstream wiring (no stdout parsing).
#[derive(Clone, Debug)]
pub struct Sp1OneProofWeGateOutput {
    pub stmt_digest: [F257; 32],
    pub lock_coin_seed: [u8; 32],
    /// Per-lock share candidate plaintexts.
    ///
    /// No-tag policy: decap returns both branch decryptions per sublock, and downstream code
    /// performs a global check to select a consistent tuple across armers/locks.
    pub share_candidates: Vec<(u32, Vec<[u8; 32]>)>,
    /// Optional *debug-only* local selector output.
    ///
    /// Production policy is **global-only** resolution: callers should apply an application-level
    /// check across *many* locks/armers (e.g., derived public key / address) to pick a consistent
    /// tuple. When rep-unlinkability is enabled, per-rep plaintexts are XOR-shares and there is
    /// no safe per-lock local oracle.
    ///
    /// Enable local selection only for debugging by setting `LFP_ONEPROOF_ENABLE_LOCAL_SELECTOR=1`.
    pub selected_shares: Vec<(u32, [u8; 32])>,
}


/// Manifest embedded in the public lock package file.
///
/// This is the canonical public metadata needed to sanity-check decapsulation is using the
/// correct package for a given statement/proof stream.
#[derive(Clone, Debug)]
pub struct Sp1OneProofWeGateLockPkgManifest {
    pub stmt_digest: [F257; 32],
    pub lock_coin_seed: [u8; 32],
}

#[derive(Clone, Debug)]
struct OneProofLogicalLock {
    share_index: u32,
    reps: Vec<RingLweLockArtifact<F257>>,
    h12_seed_envelope: Option<crate::h12_rcap::H12SeedEnvelope>,
    h12_alvo_schedule_digest: Option<[u8; 32]>,
    h12_alvo_rep_bundles_digest: Option<[u8; 32]>,
}

/// Structured output for the **arming** endpoint (write lock packages to disk).
#[derive(Clone, Debug)]
pub struct Sp1OneProofWeGateArmingOutput {
    pub manifest: Sp1OneProofWeGateLockPkgManifest,
    /// Number of logical share indices (P).
    pub k_locks: usize,
    /// Number of repetitions per logical share index (R).
    pub r_reps: usize,
    /// Total lock artifacts written (= P * R).
    pub total_locks: usize,
    /// Size (bytes) of the written lock package file.
    pub lock_pkg_bytes: u64,
}

/// Structured output for the stage-2 H12 ALVO augmentation step.
#[derive(Clone, Debug)]
pub struct Sp1OneProofH12AlvoAugmentOutput {
    pub lock_pkg_bytes: u64,
    pub logical_locks: usize,
    pub pi0_len: usize,
    pub pi0_commit_root: [u8; 32],
}

const TINY_EXTRA_SIDECAR_MAGIC: &[u8; 12] = b"LFP1TINYX1\0\0";

fn write_u64_matrix(w: &mut impl Write, rows: &[Vec<u64>]) -> std::io::Result<()> {
    write_u32(w, rows.len() as u32)?;
    for row in rows {
        write_u32(w, row.len() as u32)?;
        for &v in row {
            write_u64(w, v)?;
        }
    }
    Ok(())
}

fn read_u64_matrix(r: &mut impl Read) -> std::io::Result<Vec<Vec<u64>>> {
    let row_n = read_u32(r)? as usize;
    let mut rows = Vec::with_capacity(row_n);
    for _ in 0..row_n {
        let len = read_u32(r)? as usize;
        let mut row = Vec::with_capacity(len);
        for _ in 0..len {
            row.push(read_u64(r)?);
        }
        rows.push(row);
    }
    Ok(rows)
}

fn write_tiny_extra_sidecar(
    path: &str,
    r1lf_digest: &[u8; 32],
    stmt_digest: &[F257; 32],
    extra: &crate::we_gate_tiny::TinyExtraWitness,
) -> Result<(), String> {
    let mut w = std::io::BufWriter::new(
        std::fs::File::create(path)
            .map_err(|e| format!("oneproof: create tiny extra sidecar failed: {e}"))?,
    );
    w.write_all(TINY_EXTRA_SIDECAR_MAGIC)
        .map_err(|e| format!("oneproof: write tiny extra sidecar magic failed: {e}"))?;
    w.write_all(r1lf_digest)
        .map_err(|e| format!("oneproof: write tiny extra sidecar r1lf_digest failed: {e}"))?;
    for &f in stmt_digest {
        write_u16(&mut w, f257_to_u16(&f))
            .map_err(|e| format!("oneproof: write tiny extra sidecar stmt_digest failed: {e}"))?;
    }
    write_u64_matrix(&mut w, &extra.dcom_eval_v)
        .map_err(|e| format!("oneproof: write tiny extra sidecar dcom_eval_v failed: {e}"))?;
    write_u32(&mut w, extra.dcom_eval_b.len() as u32)
        .map_err(|e| format!("oneproof: write tiny extra sidecar dcom_eval_b outer len failed: {e}"))?;
    for rows in &extra.dcom_eval_b {
        write_u64_matrix(&mut w, rows)
            .map_err(|e| format!("oneproof: write tiny extra sidecar dcom_eval_b failed: {e}"))?;
    }
    write_u64_matrix(&mut w, &extra.decomp_c0)
        .map_err(|e| format!("oneproof: write tiny extra sidecar decomp_c0 failed: {e}"))?;
    write_u64_matrix(&mut w, &extra.decomp_c1)
        .map_err(|e| format!("oneproof: write tiny extra sidecar decomp_c1 failed: {e}"))?;
    write_u64_matrix(&mut w, &extra.decomp_v0a)
        .map_err(|e| format!("oneproof: write tiny extra sidecar decomp_v0a failed: {e}"))?;
    write_u64_matrix(&mut w, &extra.decomp_v0b)
        .map_err(|e| format!("oneproof: write tiny extra sidecar decomp_v0b failed: {e}"))?;
    write_u64_matrix(&mut w, &extra.decomp_v1a)
        .map_err(|e| format!("oneproof: write tiny extra sidecar decomp_v1a failed: {e}"))?;
    write_u64_matrix(&mut w, &extra.decomp_v1b)
        .map_err(|e| format!("oneproof: write tiny extra sidecar decomp_v1b failed: {e}"))?;
    w.flush()
        .map_err(|e| format!("oneproof: flush tiny extra sidecar failed: {e}"))?;
    Ok(())
}

fn read_tiny_extra_sidecar(
    path: &str,
) -> Result<([u8; 32], [F257; 32], crate::we_gate_tiny::TinyExtraWitness), String> {
    let mut r = std::io::BufReader::new(
        std::fs::File::open(path)
            .map_err(|e| format!("oneproof: open tiny extra sidecar failed: {e}"))?,
    );
    let mut magic = [0u8; 12];
    r.read_exact(&mut magic)
        .map_err(|e| format!("oneproof: read tiny extra sidecar magic failed: {e}"))?;
    if magic != *TINY_EXTRA_SIDECAR_MAGIC {
        return Err("oneproof: bad tiny extra sidecar magic".to_string());
    }
    let mut r1lf_digest = [0u8; 32];
    r.read_exact(&mut r1lf_digest)
        .map_err(|e| format!("oneproof: read tiny extra sidecar r1lf_digest failed: {e}"))?;
    let mut stmt_digest = [F257::ZERO; 32];
    for f in &mut stmt_digest {
        *f = u16_to_f257(read_u16(&mut r).map_err(|e| {
            format!("oneproof: read tiny extra sidecar stmt_digest failed: {e}")
        })?);
    }
    let dcom_eval_v = read_u64_matrix(&mut r)
        .map_err(|e| format!("oneproof: read tiny extra sidecar dcom_eval_v failed: {e}"))?;
    let outer_n = read_u32(&mut r)
        .map_err(|e| format!("oneproof: read tiny extra sidecar dcom_eval_b outer len failed: {e}"))?
        as usize;
    let mut dcom_eval_b = Vec::with_capacity(outer_n);
    for _ in 0..outer_n {
        dcom_eval_b.push(
            read_u64_matrix(&mut r)
                .map_err(|e| format!("oneproof: read tiny extra sidecar dcom_eval_b failed: {e}"))?,
        );
    }
    let extra = crate::we_gate_tiny::TinyExtraWitness {
        dcom_eval_b,
        dcom_eval_v,
        decomp_c0: read_u64_matrix(&mut r)
            .map_err(|e| format!("oneproof: read tiny extra sidecar decomp_c0 failed: {e}"))?,
        decomp_c1: read_u64_matrix(&mut r)
            .map_err(|e| format!("oneproof: read tiny extra sidecar decomp_c1 failed: {e}"))?,
        decomp_v0a: read_u64_matrix(&mut r)
            .map_err(|e| format!("oneproof: read tiny extra sidecar decomp_v0a failed: {e}"))?,
        decomp_v0b: read_u64_matrix(&mut r)
            .map_err(|e| format!("oneproof: read tiny extra sidecar decomp_v0b failed: {e}"))?,
        decomp_v1a: read_u64_matrix(&mut r)
            .map_err(|e| format!("oneproof: read tiny extra sidecar decomp_v1a failed: {e}"))?,
        decomp_v1b: read_u64_matrix(&mut r)
            .map_err(|e| format!("oneproof: read tiny extra sidecar decomp_v1b failed: {e}"))?,
    };
    Ok((r1lf_digest, stmt_digest, extra))
}

fn resolve_tiny_extra_sidecar_path(lock_pkg_in_path: &str) -> Option<String> {
    if let Ok(path) = std::env::var("LFP_ONEPROOF_TINY_EXTRA_IN") {
        return Some(path);
    }

    let lock_pkg = std::path::Path::new(lock_pkg_in_path);
    let mut candidates: Vec<std::path::PathBuf> = Vec::new();

    candidates.push(lock_pkg.with_extension("tinyextra"));
    candidates.push(std::path::PathBuf::from(format!("{lock_pkg_in_path}.tinyextra")));

    if let Some(stem) = lock_pkg.file_stem().and_then(|s| s.to_str()) {
        let mut sibling = lock_pkg.to_path_buf();
        sibling.set_file_name(format!("{stem}.tinyextra"));
        candidates.push(sibling);
    }

    candidates
        .into_iter()
        .find(|p| p.is_file())
        .map(|p| p.to_string_lossy().into_owned())
}

fn hex32(bytes: &[u8; 32]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(64);
    for &b in bytes {
        out.push(HEX[(b >> 4) as usize] as char);
        out.push(HEX[(b & 0x0f) as usize] as char);
    }
    out
}

fn parse_hex32(mut s: &str) -> Result<[u8; 32], String> {
    if let Some(rest) = s.strip_prefix("0x").or_else(|| s.strip_prefix("0X")) {
        s = rest;
    }
    let s = s.trim();
    if s.len() != 64 {
        return Err(format!("expected 64 hex chars (got {})", s.len()));
    }
    let mut out = [0u8; 32];
    for i in 0..32 {
        out[i] = u8::from_str_radix(&s[2 * i..2 * i + 2], 16)
            .map_err(|e| format!("hex decode failed at byte {i}: {e}"))?;
    }
    Ok(out)
}

fn parse_u64_csv(s: &str) -> Result<Vec<u64>, String> {
    let mut out = Vec::new();
    for (idx, raw) in s.split(',').enumerate() {
        let t = raw.trim();
        if t.is_empty() {
            continue;
        }
        let t = t.replace('_', "");
        let v = if let Some(rest) = t.strip_prefix("0x").or_else(|| t.strip_prefix("0X")) {
            u64::from_str_radix(rest, 16)
                .map_err(|e| format!("bad hex u64 at idx {idx}: {e}"))?
        } else {
            t.parse::<u64>()
                .map_err(|e| format!("bad decimal u64 at idx {idx}: {e}"))?
        };
        out.push(v);
    }
    Ok(out)
}

fn stmt_digest_to_bytes64(stmt_digest: &[F257; 32]) -> [u8; 64] {
    let mut out = [0u8; 64];
    for (i, e) in stmt_digest.iter().enumerate() {
        let v = (e.into_bigint().as_ref()[0] % 257) as u16;
        let b = v.to_le_bytes();
        out[2 * i] = b[0];
        out[2 * i + 1] = b[1];
    }
    out
}

fn public_prefix_bytes_from_public_words8(
    public_words8: [u64; 8],
    p_bb: u64,
) -> Result<[u8; 64], String> {
    // The shrink verifier's exported public inputs are BabyBear field elements (canonical reps),
    // and the verifier transcript absorbs them as Goldilocks base-field elements after the same
    // centered embedding used in `w_host` mapping.
    //
    // This helper reconstructs the exact 8×8 bytes (fixed-width LE) that the transcript absorbs
    // for those 8 public inputs.
    let mut out = [0u8; 64];
    for i in 0..8usize {
        let w = public_words8[i];
        let w_red = if p_bb == 0 { w } else { w % p_bb };
        let fe: BFSmall = babybear_u64_to_centered_host::<BFSmall>(w_red, p_bb);
        let bytes = prime_field_to_bytes_le_fixed::<BFSmall>(&fe);
        if bytes.len() != 8 {
            return Err(format!(
                "expected 8-byte fixed-width encoding for Goldilocks base field (got={})",
                bytes.len()
            ));
        }
        out[8 * i..8 * i + 8].copy_from_slice(&bytes);
    }
    Ok(out)
}

/// Arm the OneProof WE-gate lock(s) and write the lock package to disk.
///
/// This is the **arm-before-proof** endpoint: it produces the public lock artifacts
/// (hint blocks + ciphertexts) and writes them to `lock_pkg_out_path`.
///
/// Decapsulation can later be run (in a separate process) by loading the package with
/// `decap_sp1_oneproof_we_gate_from_files_with_lock_package`.
pub fn arm_sp1_oneproof_we_gate_write_lock_package(
    r1lf_path: &str,
    vk_hash: [u8; 32],
    public_values_digest_words8: [u64; 8],
    lock_pkg_out_path: &str,
    k_locks: usize,
) -> Result<Sp1OneProofWeGateArmingOutput, String> {
    // WE arming: statement-only.
    // - reads only public `.r1lf` header stats
    // - uses provided statement public inputs
    // - builds/loads WE-gate *shape only* (no assignment, no prover)
    // - arms locks and writes the public lock package to disk

    let r1lf_file_bytes = std::fs::metadata(r1lf_path)
        .map(|m| m.len())
        .unwrap_or(0);
    eprintln!("[oneproof:size] input_files: r1lf_bytes={}", r1lf_file_bytes);

    let pad_cols_to_multiple_of: usize = std::env::var("PAD_COLS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(256);

    let hdr = crate::sp1_r1lf::read_r1lf_stats(r1lf_path)
        .map_err(|e| format!("read_r1lf_stats: {e}"))?;
    if hdr.num_public == 0 {
        return Err("SP1 R1LF exports num_public=0; statement binding requires public inputs".to_string());
    }
    // For shape generation we only need the public-input *length*; values do not affect the
    // transcript schedule shape. Use zeros here to keep arming witness-free.
    let public_inputs: Vec<BFSmall> = vec![BFSmall::ZERO; hdr.num_public];

    // Keep this aligned with the decap path parameters.
    let kappa_expose: u64 = 8;
    let kappa_random: u64 = 8;
    let kappa: u64 = kappa_expose + kappa_random;
    // For this OneProof→LF+ pipeline we commit the base R1CS matrices (A,B,C).
    //
    // This must match the decap path's `m0.len()` (see `ComR1CSXBase::matrices_arc_base`).
    let mlen_mats: u64 = 3;

    let ncols = crate::sp1_r1lf::padded_ncols_from_header(&hdr, pad_cols_to_multiple_of)?;
    let we_params = crate::sp1_r1lf::sp1_default_we_params_for_r1lf_header_and_ncols::<R>(
        &hdr,
        ncols,
        kappa,
        mlen_mats,
    )
    .map_err(|e| format!("sp1_default_we_params_for_r1lf_header_and_ncols: {e}"))?;

    let cv_prefix_bytes = public_prefix_bytes_from_public_words8(public_values_digest_words8, hdr.p_bb)?;
    let cv_prefix_f257: [F257; 64] = cv_prefix_bytes.map(|b| F257::from(b as u64));
    let r1cs_digest = hdr.digest;
    let stmt_digest = we_statement_hash_lf_plus::<R>(
        vk_hash,
        cv_prefix_f257,
        r1cs_digest,
        LFP_WE_GATE_DIGEST_V1,
        &we_params,
    );

    // Derive one representative coin seed (j=0).
    //
    // IMPORTANT: `lock_coin_seed` is embedded in the public lock package manifest, so it MUST be
    // derived only from public data (no secret armer seeds).
    let lock_j: u64 = 0;
    let lock_coin_seed: [u8; 32] = {
        let mut h = Sha256::new();
        h.update(b"LFP_LOCK_COIN_V1");
        h.update(&stmt_digest_to_bytes64(&stmt_digest));
        h.update(&lock_j.to_le_bytes());
        h.finalize().into()
    };

    // WE arming must not run the prover or compute an assignment.
    //
    // Instead, we generate a self-consistent verifier transcript schedule and build/load the
    // statement-bound tiny-gate *shape only*.
    let n_lin_proofs = 1usize;
    let trace = crate::we_gate_arith::poseidon_trace_schedule_for_plus_with_public_inputs::<R>(
        &public_inputs,
        &we_params,
        n_lin_proofs,
        mlen_mats as usize,
    )?;

    // Shape cache directory.
    let out_dir = {
        let base = std::env::var("LFP_SHAPE_CACHE_DIR")
            .ok()
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|| {
                let mut p = std::env::temp_dir();
                p.push("lfplus_sp1_oneproof_tiny_gate");
                p
            });
        let out = base.join("cv32_public_v1");
        std::fs::create_dir_all(&out).map_err(|e| format!("create shape cache dir failed: {e}"))?;
        out
    };

    let pairs: Vec<(usize, usize)> = vec![(0, 0)];


    // WE arming must not depend on a satisfying assignment.
    let shape = crate::we_gate_arith::build_or_load_we_plus_tiny_shape::<R>(
        &trace,
        &we_params,
        &public_inputs,
        n_lin_proofs,
        mlen_mats as usize,
        &pairs,
        &out_dir,
    )
    .map_err(|e| format!("build_or_load_we_plus_tiny_shape: {e}"))?;

    // Arm the lock package(s).
    let block_id = 0usize;
    let mut ringlwe_params = RingLweParams::default();
    // Canonical hint encoding: dual-format sparse/dense-in-block.
    ringlwe_params._reserved0 = 2;
    // ---------------------------------------------------------------------
    // Secret seeds (naming is intentionally explicit).
    //
    // There are two *roles*:
    //
    // - MASTER secret seed: seeds the secret arming RNG (per-lock secret shares + rep-start
    //   jitter + per-lock coins). This MUST be secret and MUST NOT be derived from any public
    //   manifest field.
    //
    // - DPP secret seed: used for Theorem-4.3-related deterministic salts (e.g. rep_id sampling
    //   and other armer-only randomness). It MUST be secret; without it an attacker could
    //   reproduce armer-side randomness from the public artifact and reduce search/entropy.
    //
    // Default behavior: the master seed and DPP seed are INDEPENDENT.
    // - If you want determinism, set both env vars explicitly (you may set them equal or not).
    // - If you do not set an env var, that seed is sampled from OS randomness.
    //
    // Env vars (canonical, clear names):
    // - `LFP_ONEPROOF_MASTER_SEED_HEX` (32-byte hex): master secret seed for arming RNG
    // - `LFP_ONEPROOF_DPP_SEED_HEX` (32-byte hex): override secret DPP seed
    let master_seed32: [u8; 32] = if let Ok(seed_hex) = std::env::var("LFP_ONEPROOF_MASTER_SEED_HEX") {
        parse_hex32(&seed_hex).map_err(|e| format!("LFP_ONEPROOF_MASTER_SEED_HEX: {e}"))?
    } else {
        let mut s = [0u8; 32];
        OsRng.fill_bytes(&mut s);
        s
    };

    let secret_dpp_seed32: [u8; 32] =
        if let Ok(seed_hex) = std::env::var("LFP_ONEPROOF_DPP_SEED_HEX") {
            parse_hex32(&seed_hex).map_err(|e| format!("LFP_ONEPROOF_DPP_SEED_HEX: {e}"))?
        } else {
            let mut s = [0u8; 32];
            OsRng.fill_bytes(&mut s);
            s
        };

    // IMPORTANT SECURITY INVARIANT:
    // - `lock_coin_seed` is public (embedded in the lock package manifest).
    // - therefore, no *secret* material may be derived solely from `lock_coin_seed`.
    //
    // All secret arming randomness must come from a seed that is NOT recoverable from the lock
    // package.
    let mut secret_master_rng = StdRng::from_seed(master_seed32);

    // Combine policy: hash-combine only (no Shamir).
    //
    // `k_locks` is treated as the number of logical share indices (P). Each logical share can be
    // repeated R times (`LFP_ONEPROOF_REPS`) and resolved via canonical majority selection.
    //
    // We keep K-of-K across logical shares to avoid subset-selection ambiguity under the
    // no-per-lock-oracle policy.
    let p_locks = k_locks.max(1);
    let r_reps: usize = std::env::var("LFP_ONEPROOF_REPS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1)
        .max(1);
    let total_locks = p_locks
        .checked_mul(r_reps)
        .ok_or_else(|| format!("P*R overflow (P={}, R={})", p_locks, r_reps))?;

    let shares: Vec<(u32, [u8; 32])> = {
        let mut out: Vec<(u32, [u8; 32])> = Vec::with_capacity(p_locks);
        for j in 0..p_locks {
            let mut v = [0u8; 32];
            secret_master_rng.fill_bytes(&mut v);
            // Canonical 1-based indices (helps future-proof ordering).
            out.push(((j as u32) + 1, v));
        }
        out
    };
    let t_arm = Instant::now();
    let mut logical_locks: Vec<OneProofLogicalLock> = Vec::with_capacity(p_locks);
    let surface_exporter = if std::env::var("LFP_ONEPROOF_ENABLE_H12")
        .ok()
        .is_some_and(|v| v != "0")
    {
        Some(crate::we_tiny_lock::LfplusStmtSurfaceExporter::new(&shape, &stmt_digest)?)
    } else {
        None
    };
    // Residual-gate mix count (K). Legacy env name is accepted for backward compatibility.
    let _p_channels: u16 = 1;
    let gate_mix_k_raw: usize = std::env::var("LFP_ONEPROOF_GATE_MIX_K")
        .ok()
        .and_then(|s| s.parse().ok())
        .or_else(|| {
            std::env::var("LFP_ONEPROOF_HITS_PER_BLOCK")
                .ok()
                .and_then(|s| s.parse().ok())
        })
        .unwrap_or(4)
        .max(1);
    let gate_mix_k: u16 = gate_mix_k_raw.try_into().map_err(|_| {
        format!(
            "LFP_ONEPROOF_GATE_MIX_K out of range for u16 (value={})",
            gate_mix_k_raw
        )
    })?;
    // Sizing policy:
    // - Use a moderate hint budget with a small max retry count to reject only pathological
    //   outliers (low bias, fast arming).
    // - Sample `rep_id` pseudorandomly per try from a secret seed, rather than incrementing.
    //   This avoids selection structure and lets us resample on bad offsets/budgets.
    let hint_budget_bytes_opt: Option<usize> = std::env::var("LFP_ONEPROOF_HINT_BUDGET_BYTES")
        .ok()
        .and_then(|s| s.parse().ok());
    let max_rep_tries: usize = std::env::var("LFP_ONEPROOF_MAX_REP_TRIES")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(32)
        .max(1);

    // Pre-sample per-logical-lock RNG seeds *sequentially* from the master RNG so that arming remains
    // deterministic under a fixed `LFP_ONEPROOF_MASTER_SEED_HEX`, while allowing parallel lock
    // construction (the heavy part).
    let mut per_lock_rng_seed32: Vec<[u8; 32]> = Vec::with_capacity(p_locks);
    for _ in 0..p_locks {
        let mut s = [0u8; 32];
        secret_master_rng.fill_bytes(&mut s);
        per_lock_rng_seed32.push(s);
    }

    let mut results: Vec<(usize, OneProofLogicalLock)> = (0..p_locks)
        .into_par_iter()
        .map(|p| -> Result<(usize, OneProofLogicalLock), String> {
            let lock_j = p as u64;
            let mut rng = StdRng::from_seed(per_lock_rng_seed32[p]);
            let t_lock = Instant::now();
            let policy = crate::we_tiny_lock::WeRingLweLockArmingPolicy {
                base_rep_id: 0,
                max_rep_tries,
                hint_budget_bytes: hint_budget_bytes_opt,
            };
            let arm_out = crate::we_tiny_lock::arm_lfplus_ringlwe_logical_lock::<R>(
                shape.clone(),
                &stmt_digest,
                secret_dpp_seed32,
                lock_j,
                block_id,
                policy,
                ringlwe_params.clone(),
                gate_mix_k,
                r_reps,
                shares[p].1.as_slice(),
                &mut rng,
            )?;
            eprintln!(
                "[oneproof:profile] share_index={} arm_lfplus_ringlwe_logical_lock elapsed={:?} reps={}",
                shares[p].0,
                t_lock.elapsed(),
                arm_out.reps.len()
            );
            Ok((
                p,
                OneProofLogicalLock {
                    share_index: shares[p].0,
                    reps: arm_out.reps,
                    h12_seed_envelope: None,
                    h12_alvo_schedule_digest: None,
                    h12_alvo_rep_bundles_digest: None,
                },
            ))
        })
        .collect::<Result<Vec<_>, String>>()?;
    results.sort_by_key(|(j, _l)| *j);
    for (_p, ll) in results {
        logical_locks.push(ll);
    }

    let enable_h12 = std::env::var("LFP_ONEPROOF_ENABLE_H12")
        .ok()
        .is_some_and(|v| v != "0");
    if enable_h12 {
        let enable_h12_alvo = std::env::var("LFP_ONEPROOF_ENABLE_H12_ALVO")
            .ok()
            .is_some_and(|v| v != "0");
        let r_cap_reps: usize = std::env::var("LFP_H12_RCAP_REPS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(1)
            .max(1);
        let h12_seed_mode = std::env::var("LFP_H12_SEED_MODE")
            .unwrap_or_else(|_| "ext16".to_string());
        for ll in &mut logical_locks {
            let t_h12 = Instant::now();
            let checks = crate::h12_rcap::capsule_checks_from_logical_lock_with_r_cap(
                ll.reps.as_slice(),
                r_cap_reps,
            );
            let hidden_plain = crate::h12_rcap::hidden_plain_from_rep_hidden_state(ll.reps.as_slice());
            let mut h12_alvo_stage1_root: Option<[u8; 32]> = None;
            let mut cached_schedule_surfaces: Option<Vec<dpp::theorem43::Theorem43AlvoLocalCheckSurface<F257>>> = None;
            if enable_h12_alvo {
                let t_sched = Instant::now();
                let schedule_surfaces = surface_exporter
                    .as_ref()
                    .ok_or_else(|| "oneproof: missing shared ALVO exporter".to_string())?
                    .export_alvo_schedule(checks.as_slice())
                    .map_err(|e| format!("oneproof: export H12 ALVO schedule failed: {e}"))?;
                eprintln!(
                    "[oneproof:profile] share_index={} export_main_alvo_schedule elapsed={:?} surfaces={}",
                    ll.share_index,
                    t_sched.elapsed(),
                    schedule_surfaces.len()
                );
                cached_schedule_surfaces = Some(schedule_surfaces.clone());
                ll.h12_alvo_schedule_digest = Some(crate::h12_alvo::digest_alvo_schedule(
                    schedule_surfaces.as_slice(),
                    crate::h12_rcap::H12_RCAP_PACK_D,
                ));
                let schedule_digest = ll
                    .h12_alvo_schedule_digest
                    .expect("set above");
                h12_alvo_stage1_root = Some(derive_stage1_root(
                    &stmt_digest,
                    &lock_coin_seed,
                    ll.share_index,
                    r_cap_reps as u16,
                    &schedule_digest,
                ));
                let t_rep = Instant::now();
                let rep_bundle_hashes: Vec<[u8; 32]> = ll
                    .reps
                    .iter()
                    .map(digest_rep_bundle_metadata_from_lock)
                    .collect();
                eprintln!(
                    "[oneproof:profile] share_index={} rep_bundle_metadata_digest elapsed={:?} reps={}",
                    ll.share_index,
                    t_rep.elapsed(),
                    ll.reps.len()
                );
                ll.h12_alvo_rep_bundles_digest = Some(crate::h12_alvo::digest_rep_bundle_hashes(
                    rep_bundle_hashes.as_slice(),
                    crate::h12_rcap::H12_RCAP_PACK_D,
                ));
            }
            let t_env = Instant::now();
            let env = if enable_h12_alvo {
                let surfaces = cached_schedule_surfaces.take().ok_or_else(|| {
                    format!("oneproof: missing cached H12 ALVO schedule for share_index={}", ll.share_index)
                })?;
                let stage1_root = h12_alvo_stage1_root.ok_or_else(|| {
                    format!("oneproof: missing ALVO stage1 root for share_index={}", ll.share_index)
                })?;
                match h12_seed_mode.as_str() {
                    "f257-bytes" => crate::h12_rcap::H12SeedEnvelope::ByteF257(
                        crate::h12_rcap::encrypt_seed_envelope_alvo(
                            surfaces.as_slice(),
                            &stage1_root,
                            r_cap_reps as u16,
                            hidden_plain.as_slice(),
                            &mut secret_master_rng,
                        )
                        .map_err(|e| format!("oneproof: H12 ALVO seed envelope encrypt failed: {e}"))?,
                    ),
                    "ext16" => crate::h12_rcap::H12SeedEnvelope::Ext16(
                        crate::h12_rcap::encrypt_seed_envelope_alvo_ext16(
                            surfaces.as_slice(),
                            &stage1_root,
                            r_cap_reps as u16,
                            hidden_plain.as_slice(),
                            &mut secret_master_rng,
                        )
                        .map_err(|e| format!("oneproof: H12 ALVO ext16 seed envelope encrypt failed: {e}"))?,
                    ),
                    other => {
                        return Err(format!(
                            "oneproof: unsupported LFP_H12_SEED_MODE={other} (expected f257-bytes or ext16)"
                        ));
                    }
                }
                
            } else {
                let surfaces = surface_exporter
                    .as_ref()
                    .ok_or_else(|| "oneproof: missing shared capsule exporter".to_string())?
                    .export_capsule_schedule(checks.as_slice())
                    .map_err(|e| format!("oneproof: export H12 capsule schedule failed: {e}"))?;
                match h12_seed_mode.as_str() {
                    "f257-bytes" => crate::h12_rcap::H12SeedEnvelope::ByteF257(
                        crate::h12_rcap::encrypt_seed_envelope(
                            surfaces.as_slice(),
                            r_cap_reps as u16,
                            hidden_plain.as_slice(),
                            &mut secret_master_rng,
                        )
                        .map_err(|e| format!("oneproof: H12 seed envelope encrypt failed: {e}"))?,
                    ),
                    "ext16" => crate::h12_rcap::H12SeedEnvelope::Ext16(
                        crate::h12_rcap::encrypt_seed_envelope_ext16(
                            surfaces.as_slice(),
                            r_cap_reps as u16,
                            hidden_plain.as_slice(),
                            &mut secret_master_rng,
                        )
                        .map_err(|e| format!("oneproof: H12 ext16 seed envelope encrypt failed: {e}"))?,
                    ),
                    other => {
                        return Err(format!(
                            "oneproof: unsupported LFP_H12_SEED_MODE={other} (expected f257-bytes or ext16)"
                        ));
                    }
                }
            };
            if enable_h12_alvo {
                eprintln!(
                    "[oneproof:profile] share_index={} build_h12_seed_envelope elapsed={:?}",
                    ll.share_index,
                    t_env.elapsed()
                );
            }
            eprintln!(
                "[oneproof:profile] share_index={} build_h12_seed_envelope elapsed={:?}",
                ll.share_index,
                t_h12.elapsed()
            );
            for rep in &mut ll.reps {
                rep.ct_ubits = [0u8; 32];
                if enable_h12_alvo {
                    rep.accepting_set = [F257::ZERO, F257::ZERO];
                    rep.offset = F257::ZERO;
                }
            }
            ll.h12_seed_envelope = Some(env);
        }
    }

    if std::env::var("LFP_H12_RCAP_ESTIMATE").ok().is_some_and(|v| v != "0") {
        let enable_h12_alvo = std::env::var("LFP_ONEPROOF_ENABLE_H12_ALVO")
            .ok()
            .is_some_and(|v| v != "0");
        let r_cap_reps: usize = std::env::var("LFP_H12_RCAP_REPS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(1)
            .max(1);
        for ll in &logical_locks {
            let checks = crate::h12_rcap::capsule_checks_from_logical_lock_with_r_cap(
                ll.reps.as_slice(),
                r_cap_reps,
            );
            let est = if enable_h12_alvo {
                let surfaces = crate::we_tiny_lock::export_lfplus_alvo_schedule(
                    shape.clone(),
                    &stmt_digest,
                    checks.as_slice(),
                )
                .map_err(|e| format!("oneproof: export H12 ALVO schedule failed: {e}"))?;
                let est = crate::h12_rcap::estimate_alvo_schedule(
                    surfaces.as_slice(),
                    crate::h12_rcap::H12_RCAP_PACK_D,
                    32,
                )
                .map_err(|e| format!("oneproof: estimate H12 ALVO schedule failed: {e}"))?;
                est
            } else {
                let surfaces = crate::we_tiny_lock::export_lfplus_capsule_schedule(
                    shape.clone(),
                    &stmt_digest,
                    checks.as_slice(),
                )
                .map_err(|e| format!("oneproof: export H12 capsule schedule failed: {e}"))?;
                let est = crate::h12_rcap::estimate_capsule_schedule(
                    surfaces.as_slice(),
                    crate::h12_rcap::H12_RCAP_PACK_D,
                    32,
                );
                est
            };
            eprintln!(
                "[oneproof:h12] share_index={} h11_reps={} r_cap={} checks={} g_cap={} pi_pos_exact={} pi_pos_cons={} pi_blocks_exact={} pi_blocks_cons={} aadp_exact_bytes={} aadp_cons_bytes={}",
                ll.share_index,
                ll.reps.len(),
                r_cap_reps,
                est.check_count,
                est.g_cap,
                est.touched_pi_positions_exact,
                est.touched_pi_positions_conservative,
                est.touched_packed_pi_blocks_exact,
                est.touched_packed_pi_blocks_conservative,
                est.aadp_size_exact_bytes,
                est.aadp_size_conservative_bytes,
            );
        }
    }

    eprintln!(
        "[oneproof] armed P={} logical shares × R={} reps = {} artifacts in {:?}",
        p_locks,
        r_reps,
        total_locks,
        t_arm.elapsed()
    );

    let manifest = Sp1OneProofWeGateLockPkgManifest {
        stmt_digest,
        lock_coin_seed,
    };
    write_lock_package(lock_pkg_out_path, &manifest, &logical_locks)
        .map_err(|e| format!("oneproof: write lock package failed: {e}"))?;
    let file_bytes = std::fs::metadata(lock_pkg_out_path).map(|m| m.len()).unwrap_or(0);
    eprintln!(
        "[oneproof:size] wrote_lock_pkg: path={} bytes={}",
        lock_pkg_out_path, file_bytes
    );

    Ok(Sp1OneProofWeGateArmingOutput {
        manifest,
        k_locks: p_locks,
        r_reps,
        total_locks,
        lock_pkg_bytes: file_bytes,
    })
}

/// Decapsulate a OneProof WE-gate lock package from disk.
///
/// This is the **decap** endpoint: it reads the public lock artifacts from `lock_pkg_in_path`,
/// then runs the proof-induced streaming decapsulation and returns the recovered key.
pub fn decap_sp1_oneproof_we_gate_from_files_with_lock_package(
    r1lf_path: &str,
    witness_path: &str,
    lock_pkg_in_path: &str,
) -> Result<Sp1OneProofWeGateOutput, String> {
    decap_sp1_oneproof_we_gate_from_files_inner(r1lf_path, witness_path, lock_pkg_in_path)
}

fn decap_sp1_oneproof_we_gate_from_files_inner(
    r1lf_path: &str,
    witness_path: &str,
    lock_pkg_in_path: &str,
) -> Result<Sp1OneProofWeGateOutput, String> {
    // Print basic input sizes (useful for production sizing).
    let r1lf_file_bytes = std::fs::metadata(r1lf_path)
        .map(|m| m.len())
        .unwrap_or(0);
    let witness_file_bytes = std::fs::metadata(witness_path)
        .map(|m| m.len())
        .unwrap_or(0);
    eprintln!(
        "[oneproof:size] input_files: r1lf_bytes={} witness_bundle_bytes={}",
        r1lf_file_bytes, witness_file_bytes
    );

    // Keep defaults aligned with the example (but avoid printing / panics).
    let chunk_size: usize = std::env::var("CHUNK_SIZE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1 << 20);
    let pad_cols_to_multiple_of: usize = std::env::var("PAD_COLS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(256);

    let cache = crate::sp1_r1lf::open_sp1_r1lf_chunk_cache::<R>(
        r1lf_path,
        chunk_size,
        pad_cols_to_multiple_of,
    )
    .map_err(|e| format!("open_sp1_r1lf_chunk_cache: {e}"))?;

    // Materialize full (A,B,C) as SparseMatrix<F> (constant-coeff) by concatenating chunk rows.
    type F = <R as PolyRing>::BaseRing;
    let total_rows = cache.num_chunks * chunk_size;
    let mut a_rows: Vec<Vec<(F, usize)>> = Vec::with_capacity(total_rows);
    let mut b_rows: Vec<Vec<(F, usize)>> = Vec::with_capacity(total_rows);
    let mut c_rows: Vec<Vec<(F, usize)>> = Vec::with_capacity(total_rows);
    for chunk_idx in 0..cache.num_chunks {
        let [a, b, c] = cache.read_chunk(chunk_idx).map_err(|e| format!("read_chunk: {e}"))?;
        let conv = |m: stark_rings_linalg::SparseMatrix<F>| m.coeffs;
        let (ar, br, cr) = (conv(a), conv(b), conv(c));
        a_rows.extend(ar);
        b_rows.extend(br);
        c_rows.extend(cr);
    }
    let m_a = SparseMatrix::<F> { nrows: total_rows, ncols: cache.ncols, coeffs: a_rows };
    let m_b = SparseMatrix::<F> { nrows: total_rows, ncols: cache.ncols, coeffs: b_rows };
    let m_c = SparseMatrix::<F> { nrows: total_rows, ncols: cache.ncols, coeffs: c_rows };

    let bundle = crate::sp1_witness_io::load_sp1_witness_any(witness_path, cache.stats.num_vars)
        .map_err(|e| format!("load_sp1_witness_any: {e}"))?;
    let (w_u64, _base_len, _aux_len) = (bundle.witness, bundle.base_len, bundle.aux_len);

    // Map u64 witness -> Goldilocks base field scalars once (centered embedding mod p_bb).
    let p_bb = cache.stats.p_bb;
    let mut w_host: Vec<F> = Vec::with_capacity(w_u64.len());
    for &x in &w_u64 {
        w_host.push(babybear_u64_to_centered_host::<F>(x, p_bb));
    }

    // Public inputs from the SP1 export are witness[1..=l_pub].
    let l_pub = cache.stats.num_public;
    if l_pub == 0 {
        return Err("SP1 R1LF exports num_public=0; statement binding requires public inputs".to_string());
    }
    if w_host.len() < 1 + l_pub {
        return Err(format!(
            "witness too short for declared public inputs: w_len={} need_at_least={}",
            w_host.len(),
            1 + l_pub
        ));
    }
    let mut public_inputs: Vec<BFSmall> = w_host[1..1 + l_pub].to_vec();
    // Debug knob parity with the historical harness: mutate a public input and ensure things fail.
    if !public_inputs.is_empty()
        && std::env::var("FLIP_PUBLIC_INPUT0").ok().as_deref() == Some("1")
    {
        public_inputs[0] = <BFSmall as ark_ff::Field>::ONE - public_inputs[0];
    }

    // Mirror the example: deterministic Ajtai commitment scheme (system parameter) with prefix exposure.
    let kappa_expose: usize = 8;
    let kappa_random: usize = 8;
    let kappa: usize = kappa_expose + kappa_random;
    const AJTAI_SEED: [u8; 32] = *b"LFP_SP1_AJTAI_SEED_V1_0000000000";
    let ajtai = AjtaiCommitmentScheme::<R>::seeded_with_exposed_prefix(
        b"lf_plus_ajtai",
        AJTAI_SEED,
        kappa,
        cache.ncols,
        kappa_expose,
        1, // expose witness columns 1..=8 (skip shared ONE at column 0)
    );

    // Build ComR1CS from f0 (base witness) and keep matrices Arc for the prover.
    let mut f0 = w_host;
    f0.truncate(cache.stats.num_vars);
    let f0: Arc<Vec<F>> = Arc::new(f0);
    let r1cs = latticefold::arith::r1cs::R1CS::<F> { l: l_pub, A: m_a, B: m_b, C: m_c };
    let cr1cs =
        crate::r1cs::ComR1CSBase::<R>::from_f0_seeded_base(r1cs, f0, l_pub, &ajtai);
    let m0 = cr1cs.x.matrices_arc_base();

    // WE params derived from SP1 cache + chosen kappa.
    let we_params = crate::sp1_r1lf::sp1_default_we_params_for_r1lf_cache::<R>(
        &cache,
        kappa as u64,
        m0.len() as u64,
    )
    .map_err(|e| format!("sp1_default_we_params_for_r1lf_cache: {e}"))?;
    let dparams = crate::rgchk::DecompParameters {
        b: (we_params.decomp_b as u128),
        k: (we_params.k as usize),
        l: (we_params.l as usize),
    };
    let lin_params = crate::lin::LinParameters { kappa, decomp: dparams };

    // Choose decomposition base B > sqrt(q), rounded up to a power of 2 (same as example).
    fn isqrt_u128(x: u128) -> u128 {
        let mut lo: u128 = 0;
        let mut hi: u128 = 1u128 << 64;
        while lo + 1 < hi {
            let mid = (lo + hi) >> 1;
            if mid.saturating_mul(mid) <= x {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        lo
    }
    fn next_pow2_u128(x: u128) -> u128 {
        if x <= 1 {
            return 1;
        }
        let p = 128 - (x - 1).leading_zeros() as u32;
        1u128 << p
    }
    let q_u128: u128 = <F as PrimeField>::MODULUS.0[0] as u128;
    let b_min = isqrt_u128(q_u128) + 1;
    let b_decomp: u128 = next_pow2_u128(b_min);
    let pparams = crate::plus::PlusParameters { lin: lin_params, B: b_decomp };

    // Statement digest binds the shrink verifier's public-values digest lane (8 BabyBear words,
    // exported as the R1CS public inputs 1..=8) plus the core WE statement.
    if bundle.r1lf_digest != cache.stats.digest {
        return Err(format!(
            "witness bundle r1lf_digest does not match SP1_R1LF cache: bundle=0x{} cache=0x{}",
            hex32(&bundle.r1lf_digest),
            hex32(&cache.stats.digest)
        ));
    }
    let (vk_hash, _committed_values_digest_bytes) = bundle.public_inputs;
    let public_words8_witness: [u64; 8] = w_u64
        .get(1..9)
        .and_then(|s| s.try_into().ok())
        .ok_or_else(|| "oneproof: witness too short for public_words8 (need witness[1..=8])".to_string())?;
    let decap_override_csv = std::env::var("LFP_ONEPROOF_DECAP_PUBLIC_VALUES_DIGEST_U64_CSV").ok();
    let public_words8_stmt: [u64; 8] = match decap_override_csv.as_deref() {
        Some(csv) => {
            let xs = parse_u64_csv(&csv)
                .map_err(|e| format!("oneproof: bad LFP_ONEPROOF_DECAP_PUBLIC_VALUES_DIGEST_U64_CSV: {e}"))?;
            if xs.len() != 8 {
                return Err("oneproof: bad LFP_ONEPROOF_DECAP_PUBLIC_VALUES_DIGEST_U64_CSV (need 8 comma-separated u64 values)".to_string());
            }
            xs.try_into().unwrap()
        }
        None => public_words8_witness,
    };
    if std::env::var("LFP_ONEPROOF_DEBUG_STMT_WORDS8")
        .ok()
        .is_some_and(|v| v != "0")
    {
        eprintln!(
            "[oneproof] stmt_words8: override_used={} witness_words8={:?} stmt_words8={:?}",
            decap_override_csv.is_some(),
            public_words8_witness,
            public_words8_stmt
        );
    }
    let cv_prefix_bytes =
        public_prefix_bytes_from_public_words8(public_words8_stmt, cache.stats.p_bb)?;
    let cv_prefix_f257: [F257; 64] = cv_prefix_bytes.map(|b| F257::from(b as u64));
    let r1cs_digest = cache.stats.digest;
    let stmt_digest = we_statement_hash_lf_plus::<R>(
        vk_hash,
        cv_prefix_f257,
        r1cs_digest,
        LFP_WE_GATE_DIGEST_V1,
        &we_params,
    );

    // Derive one representative coin seed (j=0).
    //
    // IMPORTANT: `lock_coin_seed` is embedded in the public lock package manifest, so it MUST be
    // derived only from public data (no secret armer seeds).
    let lock_j: u64 = 0;
    let lock_coin_seed: [u8; 32] = {
        let mut h = Sha256::new();
        h.update(b"LFP_LOCK_COIN_V1");
        h.update(&stmt_digest_to_bytes64(&stmt_digest));
        h.update(&lock_j.to_le_bytes());
        h.finalize().into()
    };
    let binding_witness = crate::we_gate_arith::WeStatementBindingWitness {
        vk_hash,
        r1cs_digest,
        gate_digest: LFP_WE_GATE_DIGEST_V1,
        committed_values_prefix_bytes: cv_prefix_bytes,
    };

    // Plus prover.
    let mut prover = crate::plus::PlusProverSparseBase::init_seeded_base(
        ajtai.clone(),
        m0.clone(),
        1,
        pparams.clone(),
        crate::transcript::PoseidonTranscript::empty::<PC>(),
    );
    let proof = prover.prove_sparse_base(std::slice::from_ref(&cr1cs), &public_inputs);

    // Record verifier trace, then:
    // - arm the WE gate *shape* (instance depends only on params + sizes),
    // - compute a satisfying witness assignment for that armed instance,
    // - extract the witness tail for downstream lock scaffolding.
    let _poseidon_cfg = PC::get_poseidon_config();
    let mut rec = crate::recording_transcript::TracePoseidonTranscript::<R>::empty::<PC>();
    for b in &public_inputs {
        rec.absorb_field_element(b);
    }
    for lp in &proof.lproof {
        lp.verify(&mut rec);
    }
    // Skip native prefix binding check only if explicitly requested.
    let skip_prefix_check =
        std::env::var("LFP_SKIP_PREFIX_BINDING_CHECK").ok().as_deref() == Some("1");
    let bind_prefix: &[BFSmall] = if skip_prefix_check {
        &[]
    } else {
        public_inputs
            .get(0..8)
            .ok_or_else(|| "expected at least 8 public inputs for Ajtai prefix exposure".to_string())?
    };
    proof
        .cmproof
        .verify_with_mlen(m0.len(), &mut rec, bind_prefix)
        .map_err(|e| format!("cm proof verify: {e:?}"))?;
    let trace = rec.trace().clone();
    if let Ok(sidecar_out) = std::env::var("LFP_ONEPROOF_TINY_EXTRA_OUT") {
        let extra = crate::we_gate_arith::tiny_extra_witness_from_plus_proof::<R>(
            &we_params,
            &proof,
            m0.len(),
        )
        .map_err(|e| format!("oneproof: extract tiny extra witness failed: {e}"))?;
        write_tiny_extra_sidecar(&sidecar_out, &r1cs_digest, &stmt_digest, &extra)?;
        eprintln!("[oneproof] wrote_tiny_extra_sidecar path={sidecar_out}");
        if std::env::var("LFP_ONEPROOF_TINY_EXTRA_ONLY")
            .ok()
            .is_some_and(|v| v != "0")
        {
            eprintln!("[oneproof] tiny_extra_only=1 returning after sidecar write");
            return Ok(Sp1OneProofWeGateOutput {
                stmt_digest,
                lock_coin_seed,
                share_candidates: Vec::new(),
                selected_shares: Vec::new(),
            });
        }
    }

    // -------------------------------------------------------------------------
    // Tiny-field (F257) WE gate (Theorem 4.3 path)
    // -------------------------------------------------------------------------
    // Shape cache directory.
    //
    // The shape (constraint matrices) is compiled once and cached on disk.
    // Subsequent calls load the cached shape and only compute the per-proof assignment
    // (lightweight count-only pass, no disk writes).
    let out_dir = {
        let base = std::env::var("LFP_SHAPE_CACHE_DIR")
            .ok()
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|| {
                let mut p = std::env::temp_dir();
                p.push("lfplus_sp1_oneproof_tiny_gate");
                p
            });
        let out = base.join("cv32_public_v1");
        std::fs::create_dir_all(&out).map_err(|e| format!("create shape cache dir failed: {e}"))?;
        out
    };

    let pairs: Vec<(usize, usize)> = vec![(0, 0)];


    let (shape, assignment) = crate::we_gate_arith::build_or_load_we_plus_tiny_dr1cs::<R>(
        &trace,
        &we_params,
        &stmt_digest,
        &binding_witness,
        &proof,
        m0.len(),
        &pairs,
        &out_dir,
    )
    .map_err(|e| format!("build_or_load_we_plus_tiny_dr1cs: {e}"))?;

    // Optional sanity: the witness must satisfy the armed instance.
    // This can be expensive for large gates; enable only when debugging.
    if std::env::var("LFP_WE_GATE_CHECK_SAT").ok().as_deref() == Some("1") {
        let soft_sat = std::env::var("LFP_WE_GATE_CHECK_SAT_SOFT")
            .ok()
            .is_some_and(|v| v != "0");
        match shape.inst.check(&assignment) {
            Ok(()) => {}
            Err(e) => {
                if soft_sat {
                    eprintln!(
                        "[oneproof] we gate SAT check: UNSAT ({e}) -- continuing due to LFP_WE_GATE_CHECK_SAT_SOFT=1"
                    );
                } else {
                    return Err(format!("we gate armed instance not satisfied: {e}"));
                }
            }
        }
    }

    // -------------------------------------------------------------------------
    // Ring-LWE streaming decapsulation (Theorem 4.3 canonical path)
    // -------------------------------------------------------------------------
    let public_len = shape.public_len;
    let x = encode_public_x::<F257>(&stmt_digest);
    if x.len() != public_len {
        return Err(format!(
            "oneproof: bad public x length (x_len={} public_len={})",
            x.len(),
            public_len
        ));
    }
    if assignment.len() < public_len {
        return Err("oneproof: assignment shorter than public_len".to_string());
    }
    let asg_pub = assignment
        .get(0..public_len)
        .ok_or_else(|| "oneproof: assignment shorter than public_len".to_string())?;
    if asg_pub != x.as_slice() {
        return Err("oneproof: satisfying assignment public prefix mismatch".to_string());
    }
    let z_w = &assignment[public_len..];
    let (m, mut logical_locks) = read_lock_package(lock_pkg_in_path)
        .map_err(|e| format!("oneproof: read lock package failed: {e}"))?;
    // Canonical decap must enforce lock-package statement binding.
    // Keep a debug-only escape hatch for local troubleshooting.
    let skip_lockpkg_binding =
        std::env::var("LFP_SKIP_LOCKPKG_BINDING_CHECK").ok().as_deref() == Some("1");
    if !skip_lockpkg_binding {
        if m.stmt_digest != stmt_digest {
            return Err("oneproof: lock package stmt_digest mismatch".to_string());
        }
        if m.lock_coin_seed != lock_coin_seed {
            return Err("oneproof: lock package lock_coin_seed mismatch".to_string());
        }
    }
    if logical_locks.is_empty() {
        return Err("oneproof: lock package contained 0 locks".to_string());
    }
    // Combine policy is implicit: hash-combine of K-of-K per-lock 32-byte shares.
    let prover = crate::we_tiny_lock::we_ringlwe_prover_from_dr1cs::<F257>(
        shape.inst.clone(),
        shape.public_len,
    )?;
    let prof_h12 = std::env::var("LFP_H12_PROFILE").ok().is_some_and(|v| v != "0");

    let t_prove = Instant::now();
    maybe_print_rss("oneproof:before_prove_decap_stream");

    let _stmt_bytes64 = stmt_digest_to_bytes64(&stmt_digest);

    let rep_capacity: usize = logical_locks.iter().map(|ll| ll.reps.len()).sum();
    let mut flat_lock_ix: Vec<(usize, usize)> = Vec::with_capacity(rep_capacity);
    let mut flat_share_indices: Vec<u32> = Vec::with_capacity(rep_capacity);
    let mut flat_poison_blocks: Vec<usize> = Vec::with_capacity(rep_capacity);
    let mut flat_states = Vec::with_capacity(rep_capacity);
    let mut coin_ranges: Vec<(usize, usize)> = Vec::with_capacity(rep_capacity);
    let mut anchor_coin_idx: Vec<usize> = Vec::with_capacity(rep_capacity);
    let mut all_coins = Vec::new();
    let mut h12_surfaces: Vec<Option<Vec<dpp::theorem43::Theorem43CapsuleLocalCheckSurface<F257>>>> =
        Vec::with_capacity(logical_locks.len());
    let mut h12_collectors: Vec<Option<crate::h12_rcap::H12Pi0SliceCollector>> =
        Vec::with_capacity(logical_locks.len());
    let t_h12_setup = Instant::now();

    for ll in &logical_locks {
        if let Some(env) = &ll.h12_seed_envelope {
            let checks = crate::h12_rcap::capsule_checks_from_logical_lock_with_r_cap(
                ll.reps.as_slice(),
                env.r_cap_reps() as usize,
            );
            let surfaces = crate::we_tiny_lock::export_lfplus_capsule_schedule(
                shape.clone(),
                &stmt_digest,
                checks.as_slice(),
            )
            .map_err(|e| format!("oneproof: export H12 decap capsule schedule failed: {e}"))?;
            let collector = crate::h12_rcap::H12Pi0SliceCollector::from_surfaces(surfaces.as_slice());
            h12_surfaces.push(Some(surfaces));
            h12_collectors.push(Some(collector));
        } else {
            h12_surfaces.push(None);
            h12_collectors.push(None);
        }
    }
    if prof_h12 {
        eprintln!(
            "[oneproof:h12:profile] setup_surfaces elapsed={:.3}s logical_locks={}",
            t_h12_setup.elapsed().as_secs_f64(),
            logical_locks.len()
        );
    }

    for (ll_idx, ll) in logical_locks.iter().enumerate() {
        if ll.reps.is_empty() {
            return Err(format!(
                "oneproof: logical lock share_index={} has 0 reps",
                ll.share_index
            ));
        }
        for (rep_idx, l) in ll.reps.iter().enumerate() {
            let poison_blocks: usize = (l.poison_blocks as usize).max(1);
            let coins_start = all_coins.len();
            for b in 0..poison_blocks {
                all_coins.push(prover.derive_public_coins_from_stmt(
                    l.c_stmt.as_slice(),
                    b,
                    l.rep_id,
                )?);
            }
            let coins_end = all_coins.len();
            let anchor_idx = all_coins.len();
            all_coins.push(l.coins.clone());

            flat_poison_blocks.push(poison_blocks);
            flat_share_indices.push(ll.share_index);
            flat_states.push(l.decap_state(&x)?);
            coin_ranges.push((coins_start, coins_end));
            anchor_coin_idx.push(anchor_idx);
            flat_lock_ix.push((ll_idx, rep_idx));
        }
    }
    if all_coins.is_empty() {
        return Err("oneproof: no lock coins for decap".to_string());
    }

    // Single shared π0 stream for all lock reps.
    let mut on_pi0 = |chunk: &[F257]| {
        flat_states
            .par_iter_mut()
            .try_for_each(|st| st.absorb_chunk(chunk))
            .unwrap();
        for coll in &mut h12_collectors {
            if let Some(c) = coll.as_mut() {
                c.absorb_chunk(chunk);
            }
        }
    };
    let abg_all =
        prover.stream_pi0_and_collect_abg_full(&x, z_w, &all_coins, Some(&mut on_pi0))?;
    if abg_all.len() != all_coins.len() {
        return Err("oneproof: shared abg length mismatch".to_string());
    }
    if prof_h12 {
        eprintln!(
            "[oneproof:h12:profile] shared_pi0_stream elapsed={:.3}s reps={} coins={}",
            t_prove.elapsed().as_secs_f64(),
            rep_capacity,
            all_coins.len()
        );
    }
    let t_h12_finish_dots = Instant::now();
    let flat_stream_dots = flat_states
        .into_par_iter()
        .map(|st| st.finish_stream_dots())
        .collect::<Result<Vec<_>, String>>()?;
    if prof_h12 {
        eprintln!(
            "[oneproof:h12:profile] finish_stream_dots elapsed={:.3}s reps={}",
            t_h12_finish_dots.elapsed().as_secs_f64(),
            flat_stream_dots.len()
        );
    }

    let t_h12_unlock = Instant::now();
    for (ll_idx, ll) in logical_locks.iter_mut().enumerate() {
        let Some(env) = ll.h12_seed_envelope.as_ref() else {
            continue;
        };
        let surfaces = h12_surfaces[ll_idx]
            .as_ref()
            .ok_or_else(|| "oneproof: missing H12 surfaces during decap".to_string())?;
        let collector = h12_collectors[ll_idx]
            .take()
            .ok_or_else(|| "oneproof: missing H12 collector during decap".to_string())?;
        let witness_values = collector
            .into_witness()
            .map_err(|e| format!("oneproof: H12 witness collection failed: {e}"))?;
        let hidden_plain = crate::h12_rcap::decrypt_seed_envelope_any(
            env,
            surfaces.as_slice(),
            witness_values.as_slice(),
        )
        .map_err(|e| format!("oneproof: H12 seed capsule decrypt failed: {e}"))?;
        let rep_hidden = crate::h12_rcap::rep_hidden_state_from_hidden_plain(
            hidden_plain.as_slice(),
            ll.reps.len(),
        )
        .map_err(|e| format!("oneproof: H12 hidden state decode failed: {e}"))?;
        for (rep, (ct, accepting_set, offset)) in ll.reps.iter_mut().zip(rep_hidden.into_iter()) {
            rep.ct_ubits = ct;
            rep.accepting_set = accepting_set;
            rep.offset = offset;
        }
    }
    if prof_h12 {
        eprintln!(
            "[oneproof:h12:profile] unlock_hidden_state elapsed={:.3}s logical_locks={}",
            t_h12_unlock.elapsed().as_secs_f64(),
            logical_locks.len()
        );
    }
    let flat_locks: Vec<&RingLweLockArtifact<F257>> = flat_lock_ix
        .iter()
        .map(|&(ll_idx, rep_idx)| &logical_locks[ll_idx].reps[rep_idx])
        .collect();

    // Phase 1: compute per-rep candidate `s` sets (constant-shape 1..=256 scan).
    let t_phase1 = Instant::now();
    let mut indexed_s_hits = flat_stream_dots
        .into_par_iter()
        .enumerate()
        .map(|(i, dots)| -> Result<(usize, (u16, Vec<u16>)), String> {
            let l = flat_locks[i];
            let poison_blocks = flat_poison_blocks[i];
            let (coins_start, coins_end) = coin_ranges[i];
            if coins_end < coins_start || (coins_end - coins_start) != poison_blocks {
                return Err("oneproof: per-lock coin range mismatch".to_string());
            }
            let anchor_idx = anchor_coin_idx[i];

            let mut errs: Vec<u16> = Vec::with_capacity(poison_blocks);
            for b in 0..poison_blocks {
                let abg = &abg_all[coins_start + b];
                let a = crate::lockable_ringlwe::field_mod257_u16(&abg.alpha);
                let bb = crate::lockable_ringlwe::field_mod257_u16(&abg.beta);
                let gg = crate::lockable_ringlwe::field_mod257_u16(&abg.gamma);
                errs.push(crate::lockable_ringlwe::sub_mod257_u16(
                    gg,
                    crate::lockable_ringlwe::mul_mod257_u16(a, bb),
                ));
            }
            // Compute gate bit g(err).
            let g = crate::lockable_ringlwe::eval_err_gate_mod257_u16(
                &l.err_gate_hints,
                errs.as_slice(),
            )?;
            let anchor_abg = &abg_all[anchor_idx];
            // Use the full (x||π0) ABG components for the evaluation point.
            let alpha = crate::lockable_ringlwe::field_mod257_u16(&anchor_abg.alpha_pi_sparse);
            let beta = crate::lockable_ringlwe::field_mod257_u16(&anchor_abg.beta_pi_sparse);
            let gamma = crate::lockable_ringlwe::field_mod257_u16(&anchor_abg.gamma_pi_sparse);
            let rho = crate::lockable_ringlwe::field_mod257_u16(&l.coins.rho);
            let sigma = crate::lockable_ringlwe::field_mod257_u16(&l.coins.sigma);
            let u = crate::lockable_ringlwe::add_mod257_u16(
                crate::lockable_ringlwe::mul_mod257_u16(rho, alpha),
                crate::lockable_ringlwe::mul_mod257_u16(sigma, beta),
            );
            if std::env::var("LFP_DEBUG_IDENTITY").ok().as_deref() == Some("1") {
                let rep_filter = std::env::var("LFP_DEBUG_REP_ID")
                    .ok()
                    .and_then(|v| v.parse::<u64>().ok());
                if rep_filter.is_none() || rep_filter == Some(l.rep_id) {
                    eprintln!(
                        "[LF_ID_ABG] rep_id={} alpha={} beta={} gamma={} rho={} sigma={} u={}",
                        l.rep_id, alpha, beta, gamma, rho, sigma, u
                    );
                }
            }
            let s_sets = crate::lockable_ringlwe::finish_s_candidate_sets_from_stream_dots(
                l,
                dots,
                g,
                None,
                Some(u),
                Some(gamma),
                Some((alpha, beta)),
            )?;
            let hits = s_sets.into_iter().next().unwrap_or_else(|| (1u16..=256u16).collect());
            Ok((i, (g, hits)))
        })
        .collect::<Result<Vec<_>, String>>()?;
    if prof_h12 {
        eprintln!(
            "[oneproof:h12:profile] phase1_s_hits elapsed={:.3}s reps={}",
            t_phase1.elapsed().as_secs_f64(),
            indexed_s_hits.len()
        );
    }
    indexed_s_hits.sort_by_key(|(i, _)| *i);
    let per_rep_hits: Vec<(u16, Vec<u16>)> = indexed_s_hits.into_iter().map(|(_, v)| v).collect();

    // Phase 2: intersect `s` candidates across reps of the same logical lock, then decrypt only
    // those `s` values per rep.
    let per_rep_hits_only: Vec<Vec<u16>> = per_rep_hits.iter().map(|(_, v)| v.clone()).collect();
    let per_rep_s_intersect = crate::lockable_ringlwe::intersect_s_candidates_across_reps_by_share_index(
        flat_share_indices.as_slice(),
        per_rep_hits_only.as_slice(),
    )?;

    let t_phase2 = Instant::now();
    let mut candidates_per_lock: Vec<(u32, Vec<[u8; 32]>)> = Vec::with_capacity(flat_share_indices.len());
    for (ri, &share_idx) in flat_share_indices.iter().enumerate() {
        let lock = flat_locks[ri];
        let s_allow = per_rep_s_intersect
            .get(ri)
            .map(|v| v.as_slice())
            .unwrap_or(&[]);
        let mut seen = std::collections::BTreeSet::<[u8; 32]>::new();
        let mut outs: Vec<[u8; 32]> = Vec::new();
        for &s in s_allow {
            let p = lock.decrypt_payload_with_s_candidate(s);
            if p.len() != 32 {
                continue;
            }
            let mut arr = [0u8; 32];
            arr.copy_from_slice(&p);
            if seen.insert(arr) {
                outs.push(arr);
            }
        }
        candidates_per_lock.push((share_idx, outs));
    }
    if prof_h12 {
        eprintln!(
            "[oneproof:h12:profile] phase2_payload_decrypt elapsed={:.3}s reps={}",
            t_phase2.elapsed().as_secs_f64(),
            candidates_per_lock.len()
        );
    }

    maybe_print_rss("oneproof:after_prove_decap_stream");
    eprintln!("[oneproof] prove+decap(stream) in {:?}", t_prove.elapsed());

    let selected_shares = if std::env::var("LFP_ONEPROOF_ENABLE_LOCAL_SELECTOR")
        .ok()
        .as_deref()
        == Some("1")
    {
        crate::lockable_ringlwe::select_shares_by_majority(&candidates_per_lock)
    } else {
        Vec::new()
    };

    Ok(Sp1OneProofWeGateOutput {
        stmt_digest,
        lock_coin_seed,
        share_candidates: candidates_per_lock,
        selected_shares,
    })
}

fn needed_logical_blocks_for_opened_packed_blocks(
    z_w_len: usize,
    k_star: usize,
    pi0_len: usize,
    pack_d: usize,
    opened_block_indices: &[usize],
) -> Result<BTreeSet<usize>, String> {
    if k_star == 0 {
        return Err("oneproof: DALEO local-view build requires k_star > 0".to_string());
    }
    let mut needed_logical_blocks = BTreeSet::<usize>::new();
    for &packed_block in opened_block_indices {
        let block_start = packed_block
            .checked_mul(pack_d.max(1))
            .ok_or_else(|| format!("oneproof: packed block start overflow: block={packed_block}"))?;
        let block_len = crate::h12_pi_commit::block_len_for_index(pi0_len, pack_d.max(1), packed_block);
        for rel in 0..block_len {
            let pi_idx = block_start
                .checked_add(rel)
                .ok_or_else(|| "oneproof: packed block pi index overflow".to_string())?;
            if pi_idx < z_w_len {
                continue;
            }
            let off = pi_idx - z_w_len;
            needed_logical_blocks.insert(off / k_star);
        }
    }
    Ok(needed_logical_blocks)
}

fn collect_opened_rows_by_packed_blocks(
    z_w: &[F257],
    z_w_len: usize,
    k_star: usize,
    pi0_len: usize,
    opened_block_indices: &[usize],
    pack_d: usize,
    w_eval_by_logical: &BTreeMap<usize, Vec<u16>>,
) -> Result<BTreeMap<usize, Vec<u16>>, String> {
    if k_star == 0 {
        return Err("oneproof: DALEO local-view build requires k_star > 0".to_string());
    }
    let opened_set: BTreeSet<usize> = opened_block_indices.iter().copied().collect();
    if opened_set.is_empty() {
        return Err("oneproof: DALEO local-view build requires at least one opened block".to_string());
    }
    let needed_logical_blocks = needed_logical_blocks_for_opened_packed_blocks(
        z_w_len,
        k_star,
        pi0_len,
        pack_d.max(1),
        opened_block_indices,
    )?;
    for &logical_block in &needed_logical_blocks {
        if !w_eval_by_logical.contains_key(&logical_block) {
            return Err(format!(
                "oneproof: DALEO local-view missing streamed logical block {}",
                logical_block
            ));
        }
    }
    let mut opened_rows_by_block = BTreeMap::<usize, Vec<u16>>::new();
    for &packed_block in &opened_set {
        let block_start = packed_block
            .checked_mul(pack_d)
            .ok_or_else(|| format!("oneproof: packed block start overflow: block={packed_block}"))?;
        let block_len = crate::h12_pi_commit::block_len_for_index(pi0_len, pack_d, packed_block);
        let mut values = Vec::<u16>::with_capacity(block_len);
        for rel in 0..block_len {
            let pi_idx = block_start
                .checked_add(rel)
                .ok_or_else(|| "oneproof: packed block pi index overflow".to_string())?;
            if pi_idx < z_w_len {
                let v = z_w
                    .get(pi_idx)
                    .ok_or_else(|| format!("oneproof: sparse LV missing z_w position {}", pi_idx))?;
                values.push(crate::h12_pi_commit::f257_to_u16(*v));
                continue;
            }
            let off = pi_idx - z_w_len;
            let logical_block = off / k_star;
            let logical_pos = off % k_star;
            let logical_values = w_eval_by_logical.get(&logical_block).ok_or_else(|| {
                format!(
                    "oneproof: DALEO local-view missing streamed logical block {}",
                    logical_block
                )
            })?;
            let v = *logical_values.get(logical_pos).ok_or_else(|| {
                format!(
                    "oneproof: DALEO local-view logical position out of range: block={} pos={}",
                    logical_block, logical_pos
                )
            })?;
            values.push(v % 257);
        }
        opened_rows_by_block.insert(packed_block, values);
    }
    Ok(opened_rows_by_block)
}

fn opened_rows_u16_from_opened_rows(
    opened_rows: &BTreeMap<usize, Vec<u16>>,
) -> Result<BTreeMap<usize, [u16; crate::h12_rcap::H12_RCAP_PACK_D]>, String> {
    let mut out = BTreeMap::new();
    for (&block_idx, values) in opened_rows {
        if values.len() > crate::h12_rcap::H12_RCAP_PACK_D {
            return Err(format!(
                "oneproof: opened row too large for pack_d at block {}: got={} max={}",
                block_idx,
                values.len(),
                crate::h12_rcap::H12_RCAP_PACK_D
            ));
        }
        let mut row = [0u16; crate::h12_rcap::H12_RCAP_PACK_D];
        for (i, &v) in values.iter().enumerate() {
            row[i] = v % 257;
        }
        out.insert(block_idx, row);
    }
    Ok(out)
}

fn local_view_from_streamed_w_eval(
    compiled: &crate::h12_alvo_daleo::H12DaleoCompiledConstraintSystem,
    z_w: &[F257],
    w_eval_by_logical: &BTreeMap<usize, Vec<u16>>,
) -> Result<Vec<F257>, String> {
    let mut out = Vec::with_capacity(compiled.local_view_len);
    for &pi_idx in &compiled.z_w_positions {
        let v = *z_w
            .get(pi_idx)
            .ok_or_else(|| format!("oneproof: missing z_w position {} for DALEO local view", pi_idx))?;
        out.push(v);
    }
    for (logical_block, positions) in compiled
        .logical_block_indices
        .iter()
        .zip(compiled.logical_block_positions.iter())
    {
        let row = w_eval_by_logical
            .get(logical_block)
            .ok_or_else(|| format!("oneproof: missing logical block {} for DALEO local view", logical_block))?;
        for &pos in positions {
            let v = *row.get(pos).ok_or_else(|| {
                format!(
                    "oneproof: logical block {} missing DALEO position {}",
                    logical_block, pos
                )
            })?;
            out.push(F257::from((v % 257) as u64));
        }
    }
    Ok(out)
}

fn claim_from_capsule_surface_rows(
    surface: &dpp::theorem43::Theorem43CapsuleLocalCheckSurface<F257>,
    opened_rows: &BTreeMap<usize, [u16; crate::h12_rcap::H12_RCAP_PACK_D]>,
) -> Result<H12AlvoClaim, String> {
    fn eval_terms(
        terms: &[(usize, F257)],
        opened_rows: &BTreeMap<usize, [u16; crate::h12_rcap::H12_RCAP_PACK_D]>,
    ) -> Result<u16, String> {
        let mut acc = 0u16;
        for &(pi_idx, coeff) in terms {
            let block_idx = pi_idx / crate::h12_rcap::H12_RCAP_PACK_D;
            let pos = pi_idx % crate::h12_rcap::H12_RCAP_PACK_D;
            let row = opened_rows
                .get(&block_idx)
                .ok_or_else(|| format!("oneproof: missing opening block {block_idx} for pi_idx={pi_idx}"))?;
            let coeff_u16 = crate::lockable_ringlwe::field_mod257_u16(&coeff);
            acc = crate::lockable_ringlwe::add_mod257_u16(
                acc,
                crate::lockable_ringlwe::mul_mod257_u16(coeff_u16, row[pos]),
            );
        }
        Ok(acc)
    }
    fn eval_w_terms(
        terms: &[(usize, F257)],
        base_offset: usize,
        opened_rows: &BTreeMap<usize, [u16; crate::h12_rcap::H12_RCAP_PACK_D]>,
    ) -> Result<u16, String> {
        let mut acc = 0u16;
        for &(pos, coeff) in terms {
            let pi_idx = base_offset + pos;
            let block_idx = pi_idx / crate::h12_rcap::H12_RCAP_PACK_D;
            let lane = pi_idx % crate::h12_rcap::H12_RCAP_PACK_D;
            let row = opened_rows
                .get(&block_idx)
                .ok_or_else(|| format!("oneproof: missing opening block {block_idx} for pi_idx={pi_idx}"))?;
            let coeff_u16 = crate::lockable_ringlwe::field_mod257_u16(&coeff);
            acc = crate::lockable_ringlwe::add_mod257_u16(
                acc,
                crate::lockable_ringlwe::mul_mod257_u16(coeff_u16, row[lane]),
            );
        }
        Ok(acc)
    }

    let alpha = crate::lockable_ringlwe::add_mod257_u16(
        crate::lockable_ringlwe::field_mod257_u16(&surface.q1_x_dot),
        crate::lockable_ringlwe::add_mod257_u16(
            eval_terms(surface.q1_pi_terms.as_slice(), opened_rows)?,
            eval_w_terms(surface.q1_w_terms.as_slice(), surface.w_eval_block_pi_offset, opened_rows)?,
        ),
    );
    let beta = crate::lockable_ringlwe::add_mod257_u16(
        crate::lockable_ringlwe::field_mod257_u16(&surface.q2_x_dot),
        crate::lockable_ringlwe::add_mod257_u16(
            eval_terms(surface.q2_pi_terms.as_slice(), opened_rows)?,
            eval_w_terms(surface.q2_w_terms.as_slice(), surface.w_eval_block_pi_offset, opened_rows)?,
        ),
    );
    let gamma = crate::lockable_ringlwe::add_mod257_u16(
        crate::lockable_ringlwe::field_mod257_u16(&surface.q3_x_dot_sparse),
        crate::lockable_ringlwe::add_mod257_u16(
            eval_terms(surface.q3_pi_sparse_terms.as_slice(), opened_rows)?,
            eval_w_terms(surface.q3_w_terms.as_slice(), surface.w_eval_block_pi_offset, opened_rows)?,
        ),
    );
    let alpha_pi_sparse = eval_terms(surface.q1_pi_terms.as_slice(), opened_rows)?;
    let beta_pi_sparse = eval_terms(surface.q2_pi_terms.as_slice(), opened_rows)?;
    let gamma_pi_sparse = eval_terms(surface.q3_pi_sparse_terms.as_slice(), opened_rows)?;

    Ok(H12AlvoClaim {
        alpha,
        beta,
        gamma,
        alpha_pi_sparse,
        beta_pi_sparse,
        gamma_pi_sparse,
    })
}

fn compute_y_anchor_stream_from_openings(
    lock: &RingLweLockArtifact<F257>,
    openings_by_block: &BTreeMap<usize, [u16; crate::h12_rcap::H12_RCAP_PACK_D]>,
) -> Result<u16, String> {
    let mut acc = 0u16;
    for (block_idx, blk) in &lock.anchor_hints.hint_blocks_sparse {
        let row = openings_by_block
            .get(block_idx)
            .ok_or_else(|| format!("oneproof: missing opening block {} for anchor hint", block_idx))?;
        acc = crate::lockable_ringlwe::add_mod257_u16(
            acc,
            crate::lockable_ringlwe::dot_packed_block_mod257_u16(blk, row),
        );
    }
    Ok(acc)
}

fn digest_rep_bundle_metadata_from_lock(lock: &RingLweLockArtifact<F257>) -> [u8; 32] {
    let mut h = sha2::Sha256::new();
    h.update(b"LFP_H12_ALVO_REP_META_V1");
    h.update(lock.rep_id.to_le_bytes());
    h.update(lock.anchor_block_id.to_le_bytes());
    h.update(lock.poison_blocks.to_le_bytes());
    let mut block_indices: Vec<usize> = lock
        .anchor_hints
        .hint_blocks_sparse
        .iter()
        .map(|(block_idx, _blk)| *block_idx)
        .collect();
    block_indices.sort_unstable();
    h.update((block_indices.len() as u32).to_le_bytes());
    for block_idx in block_indices {
        h.update(block_idx.to_le_bytes());
    }
    h.finalize().into()
}

fn digest_rep_bundle_metadata_from_aug(rep_bundle: &H12AlvoRepSurfaceBundle) -> [u8; 32] {
    let mut h = sha2::Sha256::new();
    h.update(b"LFP_H12_ALVO_REP_META_V1");
    h.update(rep_bundle.rep_id.to_le_bytes());
    h.update((rep_bundle.anchor_block_id as u32).to_le_bytes());
    h.update((rep_bundle.poison_blocks as u32).to_le_bytes());
    let mut block_indices = rep_bundle.anchor_hint_blocks.clone();
    block_indices.sort_unstable();
    h.update((block_indices.len() as u32).to_le_bytes());
    for block_idx in block_indices {
        h.update(block_idx.to_le_bytes());
    }
    h.finalize().into()
}

fn opened_blocks_for_surface(
    surface: &dpp::theorem43::Theorem43AlvoLocalCheckSurface<F257>,
    blocks: &mut std::collections::BTreeSet<usize>,
) {
    // Use the exact touched view here: `q3_w_terms` already names the needed `w_eval` lanes,
    // and pulling the full conservative block set over-opens large slices of `pi0`.
    for block_idx in surface.packed_pi_blocks_exact(crate::h12_rcap::H12_RCAP_PACK_D) {
        blocks.insert(block_idx);
    }
}

fn compute_share_candidates_from_alvo_augmentations(
    logical_locks: &[OneProofLogicalLock],
    lock_augs: &[H12AlvoLogicalLockAugmentation],
) -> Result<Vec<(u32, Vec<[u8; 32]>)>, String> {
    if logical_locks.len() != lock_augs.len() {
        return Err(format!(
            "oneproof: ALVO logical-lock length mismatch: locks={} augmentations={}",
            logical_locks.len(),
            lock_augs.len()
        ));
    }
    let mut flat_locks: Vec<&RingLweLockArtifact<F257>> = Vec::new();
    let mut flat_share_indices: Vec<u32> = Vec::new();
    let mut per_rep_hits: Vec<(u16, Vec<u16>)> = Vec::new();
    for (ll, aug) in logical_locks.iter().zip(lock_augs.iter()) {
        if ll.reps.len() != aug.rep_bundles.len() {
            return Err(format!(
                "oneproof: ALVO rep-bundle count mismatch for share_index={}: reps={} bundles={}",
                ll.share_index,
                ll.reps.len(),
                aug.rep_bundles.len()
            ));
        }
        for (lock, rep_bundle) in ll.reps.iter().zip(aug.rep_bundles.iter()) {
            if lock.rep_id != rep_bundle.rep_id {
                return Err(format!(
                    "oneproof: ALVO rep_id mismatch for share_index={}: lock={} bundle={}",
                    ll.share_index,
                    lock.rep_id,
                    rep_bundle.rep_id
                ));
            }
            let expected_poison_blocks = (lock.poison_blocks as usize).max(1);
            if rep_bundle.poison_blocks != expected_poison_blocks {
                return Err(format!(
                    "oneproof: ALVO poison-block count mismatch for share_index={} rep_id={}: expected={} blocks={}",
                    ll.share_index,
                    rep_bundle.rep_id,
                    expected_poison_blocks,
                    rep_bundle.poison_blocks
                ));
            }
            let g = rep_bundle.poison_gate;
            let alpha = rep_bundle.anchor_projection.alpha_sparse;
            let beta = rep_bundle.anchor_projection.beta_sparse;
            let gamma = rep_bundle.anchor_projection.delta_gamma_pi;
            let rho = crate::lockable_ringlwe::field_mod257_u16(&lock.coins.rho);
            let sigma = crate::lockable_ringlwe::field_mod257_u16(&lock.coins.sigma);
            let u = crate::lockable_ringlwe::add_mod257_u16(
                crate::lockable_ringlwe::mul_mod257_u16(rho, alpha),
                crate::lockable_ringlwe::mul_mod257_u16(sigma, beta),
            );
            let y_anchor_stream = rep_bundle.anchor_projection.y_anchor_stream;
            let hits = crate::lockable_ringlwe::finish_s_candidate_sets_from_stream_dots(
                lock,
                crate::lockable_ringlwe::RingLweStreamDots {
                    y_anchor_stream,
                    alpha_pi_stream: 0,
                    beta_pi_stream: 0,
                    gamma_pi_stream: 0,
                },
                g,
                None,
                Some(u),
                Some(gamma),
                Some((alpha, beta)),
            )?
            .into_iter()
            .next()
            .unwrap_or_else(|| (1u16..=256u16).collect());
            flat_locks.push(lock);
            flat_share_indices.push(ll.share_index);
            per_rep_hits.push((g, hits));
        }
    }

    let per_rep_hits_only: Vec<Vec<u16>> = per_rep_hits.iter().map(|(_, v)| v.clone()).collect();
    let per_rep_s_intersect = crate::lockable_ringlwe::intersect_s_candidates_across_reps_by_share_index(
        flat_share_indices.as_slice(),
        per_rep_hits_only.as_slice(),
    )?;
    let mut candidates_per_lock: Vec<(u32, Vec<[u8; 32]>)> = Vec::with_capacity(flat_share_indices.len());
    for (ri, &share_idx) in flat_share_indices.iter().enumerate() {
        let lock = flat_locks[ri];
        let s_allow = per_rep_s_intersect
            .get(ri)
            .map(|v| v.as_slice())
            .unwrap_or(&[]);
        let mut seen = std::collections::BTreeSet::<[u8; 32]>::new();
        let mut outs: Vec<[u8; 32]> = Vec::new();
        for &s in s_allow {
            let p = lock.decrypt_payload_with_s_candidate(s);
            if p.len() != 32 {
                continue;
            }
            let mut arr = [0u8; 32];
            arr.copy_from_slice(&p);
            if seen.insert(arr) {
                outs.push(arr);
            }
        }
        candidates_per_lock.push((share_idx, outs));
    }
    Ok(candidates_per_lock)
}

fn build_h12_alvo_lock_augmentations(
    manifest: &Sp1OneProofWeGateLockPkgManifest,
    logical_locks: &[OneProofLogicalLock],
    surfaces_by_lock: &[Option<Vec<dpp::theorem43::Theorem43AlvoLocalCheckSurface<F257>>>],
    surface_exporter: &crate::we_tiny_lock::LfplusStmtSurfaceExporter,
    z_w: &[F257],
) -> Result<Vec<H12AlvoLogicalLockAugmentation>, String> {
    let profile_aug = std::env::var("LFP_PROFILE_H12_AUG")
        .ok()
        .is_some_and(|v| v != "0");
    if logical_locks.len() != surfaces_by_lock.len() {
        return Err("oneproof: H12 ALVO lock/surface/bundle length mismatch".to_string());
    }
    let mut out = Vec::new();
    for (ll, maybe_surfaces) in logical_locks.iter().zip(surfaces_by_lock.iter()) {
        let Some(env) = ll.h12_seed_envelope.as_ref() else {
            continue;
        };
        let surfaces = maybe_surfaces
            .as_ref()
            .ok_or_else(|| format!("oneproof: missing ALVO surfaces for share_index={}", ll.share_index))?;
        let expected_schedule_digest = ll.h12_alvo_schedule_digest.ok_or_else(|| {
            format!(
                "oneproof: missing ALVO schedule digest in stage1 lock for share_index={}",
                ll.share_index
            )
        })?;
        let expected_digest = ll.h12_alvo_rep_bundles_digest.ok_or_else(|| {
            format!(
                "oneproof: missing ALVO rep-bundle digest in stage1 lock for share_index={}",
                ll.share_index
            )
        })?;
        let rep_bundle_hashes: Vec<[u8; 32]> = ll
            .reps
            .iter()
            .map(digest_rep_bundle_metadata_from_lock)
            .collect();
        let got_digest = crate::h12_alvo::digest_rep_bundle_hashes(
            rep_bundle_hashes.as_slice(),
            crate::h12_rcap::H12_RCAP_PACK_D,
        );
        if got_digest != expected_digest {
            return Err(format!(
                "oneproof: ALVO rep-bundle digest mismatch for share_index={}",
                ll.share_index
            ));
        }
        let schedule_digest =
            digest_alvo_schedule(surfaces.as_slice(), crate::h12_rcap::H12_RCAP_PACK_D);
        if schedule_digest != expected_schedule_digest {
            return Err(format!(
                "oneproof: ALVO schedule digest mismatch for share_index={}",
                ll.share_index
            ));
        }
        let stage1_root = derive_stage1_root(
            &manifest.stmt_digest,
            &manifest.lock_coin_seed,
            ll.share_index,
            env.r_cap_reps(),
            &expected_schedule_digest,
        );
        let daleo_compiled =
            crate::h12_rcap::compile_alvo_seed_constraint_system(surfaces.as_slice(), &stage1_root)
                .map_err(|e| format!("oneproof: compile H12 DALEO seed relation failed: {e}"))?;
        if profile_aug {
            crate::utils::maybe_print_rss("oneproof:build_lock_aug:after_daleo_compile");
            eprintln!(
                "[oneproof:aug_profile] build_lock_aug daleo_view share_index={} z_w_positions={} logical_blocks={} local_view_len={}",
                ll.share_index,
                daleo_compiled.compiled.z_w_positions.len(),
                daleo_compiled.compiled.logical_block_indices.len(),
                daleo_compiled.compiled.local_view_len
            );
        }
        struct RepBundleMeta {
            rep_id: u64,
            anchor_block_id: usize,
            poison_blocks: usize,
            anchor_hint_blocks: Vec<usize>,
        }
        let rep_bundle_meta: Vec<RepBundleMeta> = ll
            .reps
            .iter()
            .map(|rep| {
                let anchor_hint_blocks = rep
                    .anchor_hints
                    .hint_blocks_sparse
                    .iter()
                    .map(|(block_idx, _)| *block_idx)
                    .collect();
                Ok(RepBundleMeta {
                    rep_id: rep.rep_id,
                    anchor_block_id: rep.anchor_block_id as usize,
                    poison_blocks: (rep.poison_blocks as usize).max(1),
                    anchor_hint_blocks,
                })
            })
            .collect::<Result<_, String>>()?;
        if profile_aug {
            crate::utils::maybe_print_rss("oneproof:build_lock_aug:after_rep_bundle_export");
            eprintln!(
                "[oneproof:aug_profile] build_lock_aug rep_meta share_index={} main_surfaces={} rep_bundles={} poison_surfaces={}",
                ll.share_index,
                surfaces.len(),
                rep_bundle_meta.len(),
                rep_bundle_meta.iter().map(|b| b.poison_blocks).sum::<usize>()
            );
        }
        let anchor_checks: Vec<(usize, u64)> = rep_bundle_meta
            .iter()
            .map(|rep| (rep.anchor_block_id, rep.rep_id))
            .collect();
        let anchor_surfaces = surface_exporter.export_capsule_schedule(anchor_checks.as_slice())?;
        if anchor_surfaces.len() != rep_bundle_meta.len() {
            return Err(format!(
                "oneproof: anchor surface count mismatch: reps={} surfaces={}",
                rep_bundle_meta.len(),
                anchor_surfaces.len()
            ));
        }
        let mut poison_checks: Vec<(usize, u64)> = Vec::new();
        let mut poison_owners: Vec<usize> = Vec::new();
        for (rep_idx, rep) in rep_bundle_meta.iter().enumerate() {
            for poison_block in 0..rep.poison_blocks {
                poison_checks.push((poison_block, rep.rep_id));
                poison_owners.push(rep_idx);
            }
        }

        let mut opened_blocks = std::collections::BTreeSet::new();
        for &pi_idx in &daleo_compiled.compiled.z_w_positions {
            opened_blocks.insert(pi_idx / crate::h12_rcap::H12_RCAP_PACK_D);
        }
        for (logical_block, positions) in daleo_compiled
            .compiled
            .logical_block_indices
            .iter()
            .zip(daleo_compiled.compiled.logical_block_positions.iter())
        {
            let base = daleo_compiled.compiled.z_w_len + logical_block * daleo_compiled.compiled.k_star;
            for &rel in positions {
                opened_blocks.insert((base + rel) / crate::h12_rcap::H12_RCAP_PACK_D);
            }
        }
        for surface in surfaces {
            opened_blocks_for_surface(surface, &mut opened_blocks);
        }
        for surface in &anchor_surfaces {
            for block_idx in surface.packed_pi_blocks(crate::h12_rcap::H12_RCAP_PACK_D, false) {
                opened_blocks.insert(block_idx);
            }
        }
        if profile_aug {
            crate::utils::maybe_print_rss("oneproof:build_lock_aug:before_daleo_local_view_capture");
            eprintln!(
                "[oneproof:aug_profile] build_lock_aug opened_blocks share_index={} opened_block_count={}",
                ll.share_index,
                opened_blocks.len()
            );
        }
        let opened_blocks_vec = opened_blocks.iter().copied().collect::<Vec<_>>();
        let pi0_len = surfaces
            .first()
            .map(|s| s.proof_layout.pi0_len)
            .ok_or_else(|| format!("oneproof: missing ALVO surfaces for share_index={}", ll.share_index))?;
        let pack_d = crate::h12_rcap::H12_RCAP_PACK_D.max(1);
        let mut needed_logical_blocks = needed_logical_blocks_for_opened_packed_blocks(
            daleo_compiled.compiled.z_w_len,
            daleo_compiled.compiled.k_star,
            pi0_len,
            pack_d,
            opened_blocks_vec.as_slice(),
        )?;
        for &logical_block in &daleo_compiled.compiled.logical_block_indices {
            needed_logical_blocks.insert(logical_block);
        }
        if profile_aug {
            eprintln!(
                "[oneproof:aug_profile] build_lock_aug daleo_local_view opened_packed_blocks={} needed_logical_blocks={}",
                opened_blocks_vec.len(),
                needed_logical_blocks.len()
            );
        }
        let mut streamed_w_eval_by_logical = BTreeMap::<usize, Vec<u16>>::new();
        if !needed_logical_blocks.is_empty() {
            use std::sync::Mutex;
            let selected = needed_logical_blocks.clone();
            let streamed = Mutex::new(BTreeMap::<usize, Vec<u16>>::new());
            let capture_hook = |block_id: usize, w_eval_u16: &[u16]| -> Result<(), String> {
                if !selected.contains(&block_id) {
                    return Ok(());
                }
                let mut guard = streamed
                    .lock()
                    .map_err(|_| "oneproof: DALEO local-view capture lock poisoned".to_string())?;
                if guard.contains_key(&block_id) {
                    return Err(format!(
                        "oneproof: DALEO local-view duplicate logical block {}",
                        block_id
                    ));
                }
                guard.insert(block_id, w_eval_u16.to_vec());
                Ok(())
            };
            surface_exporter.stream_w_eval_blocks_only_u16(z_w, &capture_hook)?;
            streamed_w_eval_by_logical = streamed
                .into_inner()
                .map_err(|_| "oneproof: DALEO local-view capture lock poisoned".to_string())?;
            for &logical_block in &needed_logical_blocks {
                if !streamed_w_eval_by_logical.contains_key(&logical_block) {
                    return Err(format!(
                        "oneproof: missing streamed logical block {} for DALEO local view",
                        logical_block
                    ));
                }
            }
        }
        let opened_rows_by_block = collect_opened_rows_by_packed_blocks(
            z_w,
            daleo_compiled.compiled.z_w_len,
            daleo_compiled.compiled.k_star,
            pi0_len,
            opened_blocks_vec.as_slice(),
            pack_d,
            &streamed_w_eval_by_logical,
        )?;
        let opened_by_block = opened_rows_u16_from_opened_rows(&opened_rows_by_block)?;
        if profile_aug {
            crate::utils::maybe_print_rss("oneproof:build_lock_aug:after_daleo_local_view_capture");
            eprintln!(
                "[oneproof:aug_profile] build_lock_aug opened_rows share_index={} opened_blocks={}",
                ll.share_index,
                opened_by_block.len()
            );
        }
        // Sanity-check the exported ALVO local equations directly from opened rows before any
        // DALEO projection/packing. If this fails, the witness/package pair is incompatible.
        for (surface_idx, surface) in surfaces.iter().enumerate() {
            let claim = claim_from_capsule_surface_rows(&surface.local_check, &opened_by_block)?;
            let lhs =
                crate::lockable_ringlwe::mul_mod257_u16(claim.alpha, claim.beta);
            if lhs != claim.gamma {
                return Err(format!(
                    "oneproof: direct ALVO local check mismatch for share_index={} surface_idx={} block_id={} rep_id={} alpha={} beta={} gamma={} lhs={}",
                    ll.share_index,
                    surface_idx,
                    surface.local_check.block_id,
                    surface.local_check.rep_id,
                    claim.alpha,
                    claim.beta,
                    claim.gamma,
                    lhs
                ));
            }
        }
        if std::env::var("LFP_H12_DALEO_VALIDATE_H")
            .ok()
            .is_some_and(|v| v != "0")
        {
            for (surface_idx, surface) in surfaces.iter().enumerate() {
                let row = streamed_w_eval_by_logical.get(&surface.local_check.block_id).ok_or_else(|| {
                    format!(
                        "oneproof: missing streamed row for H_j check: share_index={} block_id={}",
                        ll.share_index, surface.local_check.block_id
                    )
                })?;
                for (h_idx, h_cons) in surface.h_w_eval_constraints.iter().enumerate() {
                    let mut acc = crate::lockable_ringlwe::field_mod257_u16(&h_cons.constant);
                    for &(pos, coeff) in &h_cons.terms {
                        let v = *row.get(pos).ok_or_else(|| {
                            format!(
                                "oneproof: H_j row position out of range: share_index={} block_id={} constraint={} pos={} row_len={}",
                                ll.share_index,
                                surface.local_check.block_id,
                                h_idx,
                                pos,
                                row.len()
                            )
                        })?;
                        let coeff_u16 = crate::lockable_ringlwe::field_mod257_u16(&coeff);
                        acc = crate::lockable_ringlwe::add_mod257_u16(
                            acc,
                            crate::lockable_ringlwe::mul_mod257_u16(coeff_u16, v),
                        );
                    }
                    if acc != 0 {
                        return Err(format!(
                            "oneproof: direct H_j mismatch for share_index={} surface_idx={} block_id={} rep_id={} constraint={} value={}",
                            ll.share_index,
                            surface_idx,
                            surface.local_check.block_id,
                            surface.local_check.rep_id,
                            h_idx,
                            acc
                        ));
                    }
                }
            }
        }
        // Build + validate the DALEO witness immediately after local-view capture so ALVO
        // incompatibilities fail fast (before the expensive poison/rep stream path).
        let mut daleo_rng = OsRng;
        let local_view = local_view_from_streamed_w_eval(
            &daleo_compiled.compiled,
            z_w,
            &streamed_w_eval_by_logical,
        )?;
        let daleo_proof = prove_daleo_from_local_view(
            &daleo_compiled.compiled,
            local_view.as_slice(),
            &mut daleo_rng,
        )
        .map_err(|e| format!("oneproof: prove H12 DALEO failed: {e}"))?;
        let daleo_witness = witness_from_daleo_proof(&daleo_compiled.compiled, &daleo_proof)
            .map_err(|e| format!("oneproof: rebuild H12 DALEO witness failed: {e}"))?;
        daleo_compiled
            .compiled
            .cs
            .check_witness(daleo_witness.as_slice())
            .map_err(|e| {
                format!(
                    "oneproof: DALEO local-view witness failed for share_index={}: {}",
                    ll.share_index, e
                )
            })?;
        if profile_aug {
            crate::utils::maybe_print_rss("oneproof:build_lock_aug:after_daleo_proof");
            eprintln!(
                "[oneproof:aug_profile] build_lock_aug daleo_proof share_index={} local_view_values={} blind_values={} ajtai_rows={} lv_present={}",
                ll.share_index,
                daleo_proof.local_view_values.len(),
                daleo_proof.blind_values.len(),
                daleo_proof.ajtai_commitment_rows.len(),
                0
            );
        }
        let poison_abg_full = if poison_checks.is_empty() {
            Vec::new()
        } else {
            let poison_coins = surface_exporter.derive_public_coins_schedule(poison_checks.as_slice())?;
            if poison_coins.len() != poison_checks.len() || poison_owners.len() != poison_checks.len() {
                return Err("oneproof: poison coins/check ownership length mismatch".to_string());
            }
            surface_exporter.collect_abg_full_for_coins_with_w_eval_hook(
                z_w,
                poison_coins.as_slice(),
                None,
            )?
        };
        if poison_abg_full.len() != poison_checks.len() {
            return Err(format!(
                "oneproof: poison ABG length mismatch: expected={} got={}",
                poison_checks.len(),
                poison_abg_full.len()
            ));
        }
        let anchor_projections: Vec<H12AlvoAnchorProjection> = ll
            .reps
            .par_iter()
            .zip(anchor_surfaces.par_iter())
            .map(|(lock, surface)| {
                let claim = claim_from_capsule_surface_rows(surface, &opened_by_block)?;
                let y_anchor_stream = compute_y_anchor_stream_from_openings(lock, &opened_by_block)?;
                let alpha_sparse = crate::lockable_ringlwe::add_mod257_u16(
                    crate::lockable_ringlwe::field_mod257_u16(&surface.q1_x_dot),
                    claim.alpha_pi_sparse,
                );
                let beta_sparse = crate::lockable_ringlwe::add_mod257_u16(
                    crate::lockable_ringlwe::field_mod257_u16(&surface.q2_x_dot),
                    claim.beta_pi_sparse,
                );
                let gamma_pi = crate::lockable_ringlwe::sub_mod257_u16(
                    claim.gamma,
                    crate::lockable_ringlwe::field_mod257_u16(&surface.q3_x_dot_sparse),
                );
                Ok::<_, String>(H12AlvoAnchorProjection {
                    y_anchor_stream,
                    alpha_sparse,
                    beta_sparse,
                    delta_gamma_pi: crate::lockable_ringlwe::sub_mod257_u16(gamma_pi, claim.gamma_pi_sparse),
                })
            })
            .collect::<Result<_, String>>()?;
        let mut per_rep_errs: Vec<Vec<u16>> = rep_bundle_meta
            .iter()
            .map(|rep| Vec::with_capacity(rep.poison_blocks))
            .collect();
        for (abg, &rep_idx) in poison_abg_full.iter().zip(poison_owners.iter()) {
            let alpha = crate::lockable_ringlwe::field_mod257_u16(&abg.alpha);
            let beta = crate::lockable_ringlwe::field_mod257_u16(&abg.beta);
            let gamma = crate::lockable_ringlwe::field_mod257_u16(&abg.gamma);
            let err = crate::lockable_ringlwe::sub_mod257_u16(
                gamma,
                crate::lockable_ringlwe::mul_mod257_u16(alpha, beta),
            );
            per_rep_errs[rep_idx].push(err);
        }
        if profile_aug {
            crate::utils::maybe_print_rss("oneproof:build_lock_aug:after_rep_claim_stream");
            eprintln!(
                "[oneproof:aug_profile] build_lock_aug rep_claims share_index={} anchor_checks={} poison_checks={}",
                ll.share_index,
                anchor_projections.len(),
                poison_checks.len()
            );
        }
        let rep_bundles: Vec<H12AlvoRepSurfaceBundle> = (0..rep_bundle_meta.len())
            .into_par_iter()
            .map(|i| {
                let rep = &rep_bundle_meta[i];
                let anchor_projection = &anchor_projections[i];
                let errs = &per_rep_errs[i];
                let poison_gate = crate::lockable_ringlwe::eval_err_gate_mod257_u16(
                    &ll.reps
                        .iter()
                        .find(|lock_rep| lock_rep.rep_id == rep.rep_id)
                        .ok_or_else(|| format!("oneproof: missing lock rep for rep_id={}", rep.rep_id))?
                        .err_gate_hints,
                    errs.as_slice(),
                )?;
                Ok(H12AlvoRepSurfaceBundle {
                    rep_id: rep.rep_id,
                    anchor_block_id: rep.anchor_block_id,
                    anchor_hint_blocks: rep.anchor_hint_blocks.clone(),
                    anchor_projection: anchor_projection.clone(),
                    poison_blocks: rep.poison_blocks,
                    poison_gate,
                })
            })
            .collect::<Result<_, String>>()?;
        match env {
            crate::h12_rcap::H12SeedEnvelope::ByteF257(env) => {
                crate::h12_rcap::decrypt_seed_envelope_alvo(
                    env,
                    surfaces.as_slice(),
                    &stage1_root,
                    daleo_witness.as_slice(),
                )
                .map_err(|e| {
                    format!(
                        "oneproof: stage1 package is not ALVO-compatible for share_index={}: {}",
                        ll.share_index, e
                    )
                })?;
            }
            crate::h12_rcap::H12SeedEnvelope::Ext16(env) => {
                crate::h12_rcap::decrypt_seed_envelope_alvo_ext16(
                    env,
                    surfaces.as_slice(),
                    &stage1_root,
                    daleo_witness.as_slice(),
                )
                .map_err(|e| {
                    format!(
                        "oneproof: stage1 package is not ALVO-compatible for share_index={}: {}",
                        ll.share_index, e
                    )
                })?;
            }
        }
        let mut lock_aug = H12AlvoLogicalLockAugmentation {
            share_index: ll.share_index,
            r_cap_reps: env.r_cap_reps(),
            stage1_root,
            stage2_root: [0u8; 32],
            schedule_digest,
            surfaces: surfaces.clone(),
            daleo_proof: Some(daleo_proof),
            rep_bundles,
        };
        lock_aug.stage2_root = derive_stage2_root(&lock_aug);
        if profile_aug {
            crate::utils::maybe_print_rss("oneproof:build_lock_aug:after_lock_aug");
        }
        out.push(lock_aug);
    }
    Ok(out)
}

fn finish_h12_alvo_augmentation_from_assignment(
    profile_aug: bool,
    t_aug_total: Instant,
    stage1_pkg_hash: [u8; 32],
    raw_stage1_pkg: Vec<u8>,
    lock_pkg_in_path: &str,
    augmented_pkg_out_path: &str,
    shape: crate::we_gate_arith::WeDr1csShape<F257>,
    assignment: Vec<F257>,
    stmt_digest: [F257; 32],
    lock_coin_seed: [u8; 32],
) -> Result<Sp1OneProofH12AlvoAugmentOutput, String> {
    if profile_aug {
        maybe_print_rss("oneproof:aug_finish:start");
    }
    let x = encode_public_x::<F257>(&stmt_digest);
    if x.len() != shape.public_len {
        return Err(format!(
            "oneproof: bad public x length (x_len={} public_len={})",
            x.len(),
            shape.public_len
        ));
    }
    let z_w = &assignment[shape.public_len..];
    let (manifest, logical_locks) = read_lock_package(lock_pkg_in_path)
        .map_err(|e| format!("oneproof: read stage1 lock package failed: {e}"))?;
    if profile_aug {
        maybe_print_rss("oneproof:aug_finish:after_read_lock_package");
        eprintln!(
            "[oneproof:aug_profile] read_lock_package logical_locks={}",
            logical_locks.len()
        );
    }
    if manifest.stmt_digest != stmt_digest {
        return Err("oneproof: stage1 lock package stmt_digest mismatch".to_string());
    }
    if manifest.lock_coin_seed != lock_coin_seed {
        return Err("oneproof: stage1 lock package lock_coin_seed mismatch".to_string());
    }
    let surface_exporter =
        crate::we_tiny_lock::LfplusStmtSurfaceExporter::new(&shape, &stmt_digest)?;
    if profile_aug {
        maybe_print_rss("oneproof:aug_finish:after_surface_exporter_new");
    }
    let t_exports = Instant::now();
    let mut h12_surfaces: Vec<Option<Vec<dpp::theorem43::Theorem43AlvoLocalCheckSurface<F257>>>> =
        Vec::with_capacity(logical_locks.len());
    for ll in &logical_locks {
        if let Some(env) = &ll.h12_seed_envelope {
            let checks = crate::h12_rcap::capsule_checks_from_logical_lock_with_r_cap(
                ll.reps.as_slice(),
                env.r_cap_reps() as usize,
            );
            let surfaces = surface_exporter
                .export_alvo_schedule(checks.as_slice())
            .map_err(|e| format!("oneproof: export H12 ALVO stage2 schedule failed: {e}"))?;
            h12_surfaces.push(Some(surfaces));
            if profile_aug {
                maybe_print_rss("oneproof:aug_finish:after_main_schedule_export");
            }
        } else {
            h12_surfaces.push(None);
        }
    }
    if profile_aug {
        eprintln!(
            "[oneproof:aug_profile] export_stage2_surfaces elapsed={:?} logical_locks={}",
            t_exports.elapsed(),
            logical_locks.len()
        );
    }
    let t_lock_aug = Instant::now();
    let lock_augmentations = build_h12_alvo_lock_augmentations(
        &manifest,
        logical_locks.as_slice(),
        h12_surfaces.as_slice(),
        &surface_exporter,
        z_w,
    )?;
    if profile_aug {
        eprintln!(
            "[oneproof:aug_profile] build_lock_augmentations elapsed={:?} logical_locks={}",
            t_lock_aug.elapsed(),
            lock_augmentations.len()
        );
    }
    let augmented_pkg = H12AlvoAugmentedPackage {
        stage1_lock_package_hash: stage1_pkg_hash,
        stage1_lock_package: raw_stage1_pkg,
        logical_locks: lock_augmentations,
    };
    let t_write = Instant::now();
    let mut out =
        std::io::BufWriter::new(std::fs::File::create(augmented_pkg_out_path).map_err(|e| {
            format!("oneproof: create ALVO augmented package failed: {e}")
        })?);
    write_augmented_package(&mut out, &augmented_pkg)
        .map_err(|e| format!("oneproof: write ALVO augmented package failed: {e}"))?;
    out.flush()
        .map_err(|e| format!("oneproof: flush ALVO augmented package failed: {e}"))?;
    if profile_aug {
        eprintln!(
            "[oneproof:aug_profile] write_augmented_pkg elapsed={:?}",
            t_write.elapsed()
        );
        eprintln!(
            "[oneproof:aug_profile] total elapsed={:?}",
            t_aug_total.elapsed()
        );
    }
    let lock_pkg_bytes = std::fs::metadata(augmented_pkg_out_path)
        .map(|m| m.len())
        .unwrap_or(0);
    Ok(Sp1OneProofH12AlvoAugmentOutput {
        lock_pkg_bytes,
        logical_locks: augmented_pkg.logical_locks.len(),
        pi0_len: augmented_pkg
            .logical_locks
            .first()
            .and_then(|l| l.surfaces.first())
            .map(|s| s.proof_layout.pi0_len)
            .unwrap_or(0),
        pi0_commit_root: [0u8; 32],
    })
}

pub fn augment_sp1_oneproof_we_gate_h12_alvo_lock_package(
    r1lf_path: &str,
    witness_path: &str,
    lock_pkg_in_path: &str,
    augmented_pkg_out_path: &str,
) -> Result<Sp1OneProofH12AlvoAugmentOutput, String> {
    const MLEN_MATS_BASE: usize = 3;
    let profile_aug = std::env::var("LFP_PROFILE_H12_AUG")
        .ok()
        .is_some_and(|v| v != "0");
    let t_aug_total = Instant::now();
    let raw_stage1_pkg = std::fs::read(lock_pkg_in_path)
        .map_err(|e| format!("oneproof: read stage1 lock package bytes failed: {e}"))?;
    let stage1_pkg_hash = digest_stage1_lock_package_bytes(raw_stage1_pkg.as_slice());

    let chunk_size: usize = std::env::var("CHUNK_SIZE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1 << 20);
    let pad_cols_to_multiple_of: usize = std::env::var("PAD_COLS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(256);
    let cache = crate::sp1_r1lf::open_sp1_r1lf_chunk_cache::<R>(
        r1lf_path,
        chunk_size,
        pad_cols_to_multiple_of,
    )
    .map_err(|e| format!("open_sp1_r1lf_chunk_cache: {e}"))?;
    type F = <R as PolyRing>::BaseRing;
    let l_pub = cache.stats.num_public;
    if l_pub == 0 {
        return Err("SP1 R1LF exports num_public=0; statement binding requires public inputs".to_string());
    }
    if let Some(sidecar_in) = resolve_tiny_extra_sidecar_path(lock_pkg_in_path) {
        let t_witness = Instant::now();
        let prefix = crate::sp1_witness_io::load_sp1_witness_prefix(
            witness_path,
            cache.stats.num_vars,
            1 + l_pub,
        )
        .map_err(|e| format!("load_sp1_witness_prefix: {e}"))?;
        if profile_aug {
            eprintln!(
                "[oneproof:aug_profile] load_witness_prefix elapsed={:?} prefix_len={}",
                t_witness.elapsed(),
                prefix.witness_prefix.len()
            );
        }
        if prefix.r1lf_digest != cache.stats.digest {
            return Err(format!(
                "witness prefix r1lf_digest does not match SP1_R1LF cache: bundle=0x{} cache=0x{}",
                hex32(&prefix.r1lf_digest),
                hex32(&cache.stats.digest)
            ));
        }
        let p_bb = cache.stats.p_bb;
        let public_inputs: Vec<BFSmall> = prefix.witness_prefix[1..1 + l_pub]
            .iter()
            .copied()
            .map(|x| babybear_u64_to_centered_host::<BFSmall>(x, p_bb))
            .collect();
        let public_words8_witness: [u64; 8] = prefix
            .witness_prefix
            .get(1..9)
            .and_then(|s| s.try_into().ok())
            .ok_or_else(|| "oneproof: witness too short for public_words8 (need witness[1..=8])".to_string())?;
        let kappa_expose: usize = 8;
        let kappa_random: usize = 8;
        let kappa: usize = kappa_expose + kappa_random;
        let we_params = crate::sp1_r1lf::sp1_default_we_params_for_r1lf_cache::<R>(
            &cache,
            kappa as u64,
            MLEN_MATS_BASE as u64,
        )
        .map_err(|e| format!("sp1_default_we_params_for_r1lf_cache: {e}"))?;
        let (vk_hash, _committed_values_digest_bytes) = prefix.public_inputs;
        let cv_prefix_bytes =
            public_prefix_bytes_from_public_words8(public_words8_witness, cache.stats.p_bb)?;
        let cv_prefix_f257: [F257; 64] = cv_prefix_bytes.map(|b| F257::from(b as u64));
        let stmt_digest = we_statement_hash_lf_plus::<R>(
            vk_hash,
            cv_prefix_f257,
            cache.stats.digest,
            LFP_WE_GATE_DIGEST_V1,
            &we_params,
        );
        let lock_j: u64 = 0;
        let lock_coin_seed: [u8; 32] = {
            let mut h = Sha256::new();
            h.update(b"LFP_LOCK_COIN_V1");
            h.update(&stmt_digest_to_bytes64(&stmt_digest));
            h.update(&lock_j.to_le_bytes());
            h.finalize().into()
        };
        let binding_witness = crate::we_gate_arith::WeStatementBindingWitness {
            vk_hash,
            r1cs_digest: cache.stats.digest,
            gate_digest: LFP_WE_GATE_DIGEST_V1,
            committed_values_prefix_bytes: cv_prefix_bytes,
        };
        let t_sidecar = Instant::now();
        let (side_r1lf_digest, side_stmt_digest, extra) = read_tiny_extra_sidecar(&sidecar_in)?;
        if side_r1lf_digest != cache.stats.digest {
            return Err(format!(
                "oneproof: tiny extra sidecar r1lf_digest mismatch: sidecar=0x{} cache=0x{}",
                hex32(&side_r1lf_digest),
                hex32(&cache.stats.digest)
            ));
        }
        if side_stmt_digest != stmt_digest {
            return Err("oneproof: tiny extra sidecar stmt_digest mismatch".to_string());
        }
        let trace = crate::we_gate_arith::poseidon_trace_schedule_for_plus_with_public_inputs::<R>(
            public_inputs.as_slice(),
            &we_params,
            1,
            MLEN_MATS_BASE,
        )
        .map_err(|e| format!("oneproof: rebuild tiny trace schedule failed: {e}"))?;
        let out_dir = {
            let base = std::env::var("LFP_SHAPE_CACHE_DIR")
                .ok()
                .map(std::path::PathBuf::from)
                .unwrap_or_else(|| {
                    let mut p = std::env::temp_dir();
                    p.push("lfplus_sp1_oneproof_tiny_gate");
                    p
                });
            let out = base.join("cv32_public_v1");
            std::fs::create_dir_all(&out).map_err(|e| format!("create shape cache dir failed: {e}"))?;
            out
        };
        let pairs: Vec<(usize, usize)> = vec![(0, 0)];
        let shape = crate::we_gate_arith::build_or_load_we_plus_tiny_shape::<R>(
            &trace,
            &we_params,
            public_inputs.as_slice(),
            1,
            MLEN_MATS_BASE,
            &pairs,
            &out_dir,
        )
        .map_err(|e| format!("build_or_load_we_plus_tiny_shape: {e}"))?;
        let assignment = crate::we_gate_arith::build_we_plus_tiny_assignment_from_extra::<R>(
            &trace,
            &we_params,
            &stmt_digest,
            &binding_witness,
            &extra,
            MLEN_MATS_BASE,
            &pairs,
        )
        .map_err(|e| format!("build_we_plus_tiny_assignment_from_extra: {e}"))?;
        if profile_aug {
            eprintln!(
                "[oneproof:aug_profile] load_sidecar_and_rebuild_assignment elapsed={:?}",
                t_sidecar.elapsed()
            );
        }
        return finish_h12_alvo_augmentation_from_assignment(
            profile_aug,
            t_aug_total,
            stage1_pkg_hash,
            raw_stage1_pkg,
            lock_pkg_in_path,
            augmented_pkg_out_path,
            shape,
            assignment,
            stmt_digest,
            lock_coin_seed,
        );
    }
    if std::env::var("LFP_ONEPROOF_ALLOW_FALLBACK_AUG")
        .ok()
        .as_deref()
        != Some("1")
    {
        let inferred = [
            std::path::Path::new(lock_pkg_in_path)
                .with_extension("tinyextra")
                .to_string_lossy()
                .into_owned(),
            format!("{lock_pkg_in_path}.tinyextra"),
        ];
        return Err(format!(
            "oneproof: refusing fallback augmentation without tiny extra sidecar; set LFP_ONEPROOF_TINY_EXTRA_IN or place a sibling sidecar at one of: {}, {}",
            inferred[0], inferred[1]
        ));
    }
    let t_read_chunks = Instant::now();
    let total_rows = cache.num_chunks * chunk_size;
    let mut a_rows: Vec<Vec<(F, usize)>> = Vec::with_capacity(total_rows);
    let mut b_rows: Vec<Vec<(F, usize)>> = Vec::with_capacity(total_rows);
    let mut c_rows: Vec<Vec<(F, usize)>> = Vec::with_capacity(total_rows);
    for chunk_idx in 0..cache.num_chunks {
        let [a, b, c] = cache.read_chunk(chunk_idx).map_err(|e| format!("read_chunk: {e}"))?;
        let conv = |m: stark_rings_linalg::SparseMatrix<F>| m.coeffs;
        let (ar, br, cr) = (conv(a), conv(b), conv(c));
        a_rows.extend(ar);
        b_rows.extend(br);
        c_rows.extend(cr);
    }
    if profile_aug {
        eprintln!(
            "[oneproof:aug_profile] read_r1lf_chunks elapsed={:?} chunks={} total_rows={}",
            t_read_chunks.elapsed(),
            cache.num_chunks,
            total_rows
        );
    }
    let m_a = SparseMatrix::<F> { nrows: total_rows, ncols: cache.ncols, coeffs: a_rows };
    let m_b = SparseMatrix::<F> { nrows: total_rows, ncols: cache.ncols, coeffs: b_rows };
    let m_c = SparseMatrix::<F> { nrows: total_rows, ncols: cache.ncols, coeffs: c_rows };
    let kappa_expose: usize = 8;
    let kappa_random: usize = 8;
    let kappa: usize = kappa_expose + kappa_random;
    const AJTAI_SEED: [u8; 32] = *b"LFP_SP1_AJTAI_SEED_V1_0000000000";
    let ajtai = AjtaiCommitmentScheme::<R>::seeded_with_exposed_prefix(
        b"lf_plus_ajtai",
        AJTAI_SEED,
        kappa,
        cache.ncols,
        kappa_expose,
        1,
    );
    let t_witness = Instant::now();
    let bundle = crate::sp1_witness_io::load_sp1_witness_any(witness_path, cache.stats.num_vars)
        .map_err(|e| format!("load_sp1_witness_any: {e}"))?;
    let (w_u64, _base_len, _aux_len) = (bundle.witness, bundle.base_len, bundle.aux_len);
    let p_bb = cache.stats.p_bb;
    let mut w_host: Vec<F> = Vec::with_capacity(w_u64.len());
    for &x in &w_u64 {
        w_host.push(babybear_u64_to_centered_host::<F>(x, p_bb));
    }
    if profile_aug {
        eprintln!(
            "[oneproof:aug_profile] load_and_convert_witness elapsed={:?} witness_len={}",
            t_witness.elapsed(),
            w_u64.len()
        );
    }
    if w_host.len() < 1 + l_pub {
        return Err(format!(
            "witness too short for declared public inputs: w_len={} need_at_least={}",
            w_host.len(),
            1 + l_pub
        ));
    }
    let public_inputs: Vec<BFSmall> = w_host[1..1 + l_pub].to_vec();
    let mut f0 = w_host;
    f0.truncate(cache.stats.num_vars);
    let f0: Arc<Vec<F>> = Arc::new(f0);
    let r1cs = latticefold::arith::r1cs::R1CS::<F> { l: l_pub, A: m_a, B: m_b, C: m_c };
    let cr1cs =
        crate::r1cs::ComR1CSBase::<R>::from_f0_seeded_base(r1cs, f0, l_pub, &ajtai);
    let m0 = cr1cs.x.matrices_arc_base();
    let we_params = crate::sp1_r1lf::sp1_default_we_params_for_r1lf_cache::<R>(&cache, kappa as u64, m0.len() as u64)
    .map_err(|e| format!("sp1_default_we_params_for_r1lf_cache: {e}"))?;
    let dparams = crate::rgchk::DecompParameters {
        b: (we_params.decomp_b as u128),
        k: (we_params.k as usize),
        l: (we_params.l as usize),
    };
    let lin_params = crate::lin::LinParameters { kappa, decomp: dparams };
    fn isqrt_u128(x: u128) -> u128 {
        let mut lo: u128 = 0;
        let mut hi: u128 = 1u128 << 64;
        while lo + 1 < hi {
            let mid = (lo + hi) >> 1;
            if mid.saturating_mul(mid) <= x {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        lo
    }
    fn next_pow2_u128(x: u128) -> u128 {
        if x <= 1 {
            return 1;
        }
        let p = 128 - (x - 1).leading_zeros() as u32;
        1u128 << p
    }
    let q_u128: u128 = <F as PrimeField>::MODULUS.0[0] as u128;
    let b_min = isqrt_u128(q_u128) + 1;
    let b_decomp: u128 = next_pow2_u128(b_min);
    let pparams = crate::plus::PlusParameters { lin: lin_params, B: b_decomp };
    if bundle.r1lf_digest != cache.stats.digest {
        return Err(format!(
            "witness bundle r1lf_digest does not match SP1_R1LF cache: bundle=0x{} cache=0x{}",
            hex32(&bundle.r1lf_digest),
            hex32(&cache.stats.digest)
        ));
    }
    let (vk_hash, _committed_values_digest_bytes) = bundle.public_inputs;
    let public_words8_witness: [u64; 8] = w_u64
        .get(1..9)
        .and_then(|s| s.try_into().ok())
        .ok_or_else(|| "oneproof: witness too short for public_words8 (need witness[1..=8])".to_string())?;
    let cv_prefix_bytes =
        public_prefix_bytes_from_public_words8(public_words8_witness, cache.stats.p_bb)?;
    let cv_prefix_f257: [F257; 64] = cv_prefix_bytes.map(|b| F257::from(b as u64));
    let stmt_digest = we_statement_hash_lf_plus::<R>(
        vk_hash,
        cv_prefix_f257,
        cache.stats.digest,
        LFP_WE_GATE_DIGEST_V1,
        &we_params,
    );
    let lock_j: u64 = 0;
    let lock_coin_seed: [u8; 32] = {
        let mut h = Sha256::new();
        h.update(b"LFP_LOCK_COIN_V1");
        h.update(&stmt_digest_to_bytes64(&stmt_digest));
        h.update(&lock_j.to_le_bytes());
        h.finalize().into()
    };
    let binding_witness = crate::we_gate_arith::WeStatementBindingWitness {
        vk_hash,
        r1cs_digest: cache.stats.digest,
        gate_digest: LFP_WE_GATE_DIGEST_V1,
        committed_values_prefix_bytes: cv_prefix_bytes,
    };
    let out_dir = {
        let base = std::env::var("LFP_SHAPE_CACHE_DIR")
            .ok()
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|| {
                let mut p = std::env::temp_dir();
                p.push("lfplus_sp1_oneproof_tiny_gate");
                p
            });
        let out = base.join("cv32_public_v1");
        std::fs::create_dir_all(&out).map_err(|e| format!("create shape cache dir failed: {e}"))?;
        out
    };
    let pairs: Vec<(usize, usize)> = vec![(0, 0)];
    let t_shape = Instant::now();
    let t_prove = Instant::now();
    let mut prover = crate::plus::PlusProverSparseBase::init_seeded_base(
        ajtai.clone(),
        m0.clone(),
        1,
        pparams.clone(),
        crate::recording_transcript::TracePoseidonTranscript::<R>::empty::<PC>(),
    );
    let proof = prover.prove_sparse_base(std::slice::from_ref(&cr1cs), &public_inputs);
    if profile_aug {
        eprintln!(
            "[oneproof:aug_profile] prove_and_verify_sparse_base elapsed={:?}",
            t_prove.elapsed()
        );
    }
    let trace = prover.transcript.trace().clone();
    let (shape, assignment) = crate::we_gate_arith::build_or_load_we_plus_tiny_dr1cs::<R>(
        &trace,
        &we_params,
        &stmt_digest,
        &binding_witness,
        &proof,
        m0.len(),
        &pairs,
        &out_dir,
    )
    .map_err(|e| format!("build_or_load_we_plus_tiny_dr1cs: {e}"))?;
    if profile_aug {
        eprintln!(
            "[oneproof:aug_profile] build_or_load_shape elapsed={:?}",
            t_shape.elapsed()
        );
    }
    finish_h12_alvo_augmentation_from_assignment(
        profile_aug,
        t_aug_total,
        stage1_pkg_hash,
        raw_stage1_pkg,
        lock_pkg_in_path,
        augmented_pkg_out_path,
        shape,
        assignment,
        stmt_digest,
        lock_coin_seed,
    )
}

pub fn decap_sp1_oneproof_we_gate_from_augmented_h12_alvo_package(
    augmented_pkg_in_path: &str,
) -> Result<Sp1OneProofWeGateOutput, String> {
    let mut r = std::io::BufReader::new(
        std::fs::File::open(augmented_pkg_in_path)
            .map_err(|e| format!("oneproof: open ALVO augmented package failed: {e}"))?,
    );
    let augmented =
        read_augmented_package(&mut r).map_err(|e| format!("oneproof: read ALVO augmented package failed: {e}"))?;
    let expected_stage1_hash = digest_stage1_lock_package_bytes(augmented.stage1_lock_package.as_slice());
    if expected_stage1_hash != augmented.stage1_lock_package_hash {
        return Err("oneproof: ALVO augmented package stage1 byte hash mismatch".to_string());
    }
    let (manifest, mut logical_locks) = read_lock_package_from_bytes(augmented.stage1_lock_package.as_slice())
        .map_err(|e| format!("oneproof: decode embedded stage1 lock package failed: {e}"))?;
    if logical_locks.len() != augmented.logical_locks.len() {
        return Err(format!(
            "oneproof: ALVO augmented package logical-lock count mismatch: stage1={} stage2={}",
            logical_locks.len(),
            augmented.logical_locks.len()
        ));
    }
    for (ll, aug) in logical_locks.iter_mut().zip(augmented.logical_locks.iter()) {
        let env = ll
            .h12_seed_envelope
            .as_ref()
            .ok_or_else(|| format!("oneproof: missing H12 envelope for share_index={}", ll.share_index))?;
        let expected_schedule_digest = ll.h12_alvo_schedule_digest.ok_or_else(|| {
            format!(
                "oneproof: missing ALVO schedule digest in embedded stage1 lock for share_index={}",
                ll.share_index
            )
        })?;
        let expected_digest = ll.h12_alvo_rep_bundles_digest.ok_or_else(|| {
            format!(
                "oneproof: missing ALVO rep-bundle digest in embedded stage1 lock for share_index={}",
                ll.share_index
            )
        })?;
        let rep_bundle_hashes: Vec<[u8; 32]> = aug
            .rep_bundles
            .iter()
            .map(digest_rep_bundle_metadata_from_aug)
            .collect();
        let got_digest = crate::h12_alvo::digest_rep_bundle_hashes(
            rep_bundle_hashes.as_slice(),
            crate::h12_rcap::H12_RCAP_PACK_D,
        );
        if got_digest != expected_digest {
            return Err(format!(
                "oneproof: ALVO rep-bundle digest mismatch for share_index={}",
                ll.share_index
            ));
        }
        if ll.share_index != aug.share_index {
            return Err(format!(
                "oneproof: ALVO augmented share_index mismatch: stage1={} stage2={}",
                ll.share_index,
                aug.share_index
            ));
        }
        let schedule_digest =
            digest_alvo_schedule(aug.surfaces.as_slice(), crate::h12_rcap::H12_RCAP_PACK_D);
        if schedule_digest != expected_schedule_digest || schedule_digest != aug.schedule_digest {
            return Err(format!(
                "oneproof: ALVO schedule digest mismatch for share_index={}",
                ll.share_index
            ));
        }
        let expected_stage1_root = derive_stage1_root(
            &manifest.stmt_digest,
            &manifest.lock_coin_seed,
            ll.share_index,
            env.r_cap_reps(),
            &expected_schedule_digest,
        );
        if expected_stage1_root != aug.stage1_root {
            return Err(format!(
                "oneproof: ALVO stage1 root mismatch for share_index={}",
                ll.share_index
            ));
        }
        if aug.daleo_proof.is_none() {
            return Err(format!(
                "oneproof: missing H12 DALEO proof for share_index={}",
                ll.share_index
            ));
        }
        for (lock, rep_bundle) in ll.reps.iter().zip(aug.rep_bundles.iter()) {
            let expected_poison_blocks = (lock.poison_blocks as usize).max(1);
            if rep_bundle.poison_blocks != expected_poison_blocks {
                return Err(format!(
                    "oneproof: ALVO poison-block count mismatch for share_index={} rep_id={}: expected={} blocks={}",
                    ll.share_index,
                    rep_bundle.rep_id,
                    expected_poison_blocks,
                    rep_bundle.poison_blocks
                ));
            }
        }
        let expected_stage2_root = derive_stage2_root(aug);
        if expected_stage2_root != aug.stage2_root {
            return Err(format!(
                "oneproof: ALVO stage2 root mismatch for share_index={}",
                ll.share_index
            ));
        }
        let daleo_compiled =
            crate::h12_rcap::compile_alvo_seed_constraint_system(aug.surfaces.as_slice(), &aug.stage1_root)
                .map_err(|e| format!("oneproof: compile H12 DALEO seed relation failed: {e}"))?;
        let witness_values = witness_from_daleo_proof(
            &daleo_compiled.compiled,
            aug.daleo_proof.as_ref().expect("checked above"),
        )
        .map_err(|e| format!("oneproof: rebuild H12 DALEO witness failed: {e}"))?;
        for (lock, rep_bundle) in ll.reps.iter().zip(aug.rep_bundles.iter()) {
            let expected_poison_blocks = (lock.poison_blocks as usize).max(1);
            if rep_bundle.poison_blocks != expected_poison_blocks {
                return Err(format!(
                    "oneproof: ALVO poison-block count mismatch for share_index={} rep_id={}: expected={} blocks={}",
                    ll.share_index,
                    rep_bundle.rep_id,
                    expected_poison_blocks,
                    rep_bundle.poison_blocks
                ));
            }
        }
        let hidden_plain = match env {
            crate::h12_rcap::H12SeedEnvelope::ByteF257(env) => crate::h12_rcap::decrypt_seed_envelope_alvo(
                env,
                aug.surfaces.as_slice(),
                &aug.stage1_root,
                witness_values.as_slice(),
            ),
            crate::h12_rcap::H12SeedEnvelope::Ext16(env) => {
                crate::h12_rcap::decrypt_seed_envelope_alvo_ext16(
                    env,
                    aug.surfaces.as_slice(),
                    &aug.stage1_root,
                    witness_values.as_slice(),
                )
            }
        }
        .map_err(|e| format!("oneproof: H12 ALVO seed capsule decrypt failed: {e}"))?;
        let rep_hidden = crate::h12_rcap::rep_hidden_state_from_hidden_plain(
            hidden_plain.as_slice(),
            ll.reps.len(),
        )
        .map_err(|e| format!("oneproof: H12 ALVO hidden state decode failed: {e}"))?;
        for (rep, (ct, accepting_set, offset)) in ll.reps.iter_mut().zip(rep_hidden.into_iter()) {
            rep.ct_ubits = ct;
            rep.accepting_set = accepting_set;
            rep.offset = offset;
        }
    }
    let candidates_per_lock = compute_share_candidates_from_alvo_augmentations(
        logical_locks.as_slice(),
        augmented.logical_locks.as_slice(),
    )?;
    let selected_shares = if std::env::var("LFP_ONEPROOF_ENABLE_LOCAL_SELECTOR")
        .ok()
        .as_deref()
        == Some("1")
    {
        crate::lockable_ringlwe::select_shares_by_majority(&candidates_per_lock)
    } else {
        Vec::new()
    };
    Ok(Sp1OneProofWeGateOutput {
        stmt_digest: manifest.stmt_digest,
        lock_coin_seed: manifest.lock_coin_seed,
        share_candidates: candidates_per_lock,
        selected_shares,
    })
}

#[inline]
fn babybear_u64_to_centered_host<F: ark_ff::Field>(x: u64, p_bb: u64) -> F {
    debug_assert!(p_bb > 1);
    let half = p_bb / 2;
    if x > half {
        let neg = p_bb - x;
        -F::from(neg)
    } else {
        F::from(x)
    }
}

// ---------------------------------------------------------------------------
// OneProof lock package IO (research endpoint plumbing)
// ---------------------------------------------------------------------------

fn f257_to_u16(f: &F257) -> u16 {
    (f.into_bigint().as_ref()[0] % 257) as u16
}

fn u16_to_f257(x: u16) -> F257 {
    F257::from((x % 257) as u64)
}

fn write_u32(w: &mut impl Write, v: u32) -> std::io::Result<()> {
    w.write_all(&v.to_le_bytes())
}

fn write_u16(w: &mut impl Write, v: u16) -> std::io::Result<()> {
    w.write_all(&v.to_le_bytes())
}

fn write_u64(w: &mut impl Write, v: u64) -> std::io::Result<()> {
    w.write_all(&v.to_le_bytes())
}

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

fn read_u64(r: &mut impl Read) -> std::io::Result<u64> {
    let mut b = [0u8; 8];
    r.read_exact(&mut b)?;
    Ok(u64::from_le_bytes(b))
}

fn write_lock_package_to_writer(
    w: &mut impl Write,
    manifest: &Sp1OneProofWeGateLockPkgManifest,
    logical_locks: &[OneProofLogicalLock],
) -> std::io::Result<()> {
    fn write_packed_block(w: &mut impl Write, blk: &crate::lockable_ringlwe::PackedF257Block64) -> std::io::Result<()> {
        match blk {
            crate::lockable_ringlwe::PackedF257Block64::Sparse { entries } => {
                w.write_all(&[0u8])?;
                write_u32(w, entries.len() as u32)?;
                for (pf, c) in entries {
                    w.write_all(&[*pf, *c])?;
                }
                Ok(())
            }
            crate::lockable_ringlwe::PackedF257Block64::Dense { vals, is256_mask } => {
                w.write_all(&[1u8])?;
                w.write_all(vals)?;
                write_u64(w, *is256_mask)?;
                Ok(())
            }
        }
    }

    let has_h12 = logical_locks.iter().any(|ll| ll.h12_seed_envelope.is_some());
    if has_h12 && logical_locks.iter().any(|ll| ll.h12_seed_envelope.is_none()) {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "mixed H11/H12 logical locks not supported",
        ));
    }

    // Sparse-hints + residual-gate lock encoding.
    //
    // H12 extends H11 with one per-logical-lock seed capsule and hidden-state blob.
    w.write_all(if has_h12 { b"LFP1LOCKH12" } else { b"LFP1LOCKH11" })?;
    for f in &manifest.stmt_digest {
        w.write_all(&f257_to_u16(f).to_le_bytes())?;
    }
    w.write_all(&manifest.lock_coin_seed)?;

    write_u32(w, logical_locks.len() as u32)?;
    for ll in logical_locks {
        write_u32(w, ll.share_index)?;
        write_u32(w, ll.reps.len() as u32)?;
        for lock in &ll.reps {

            // c_stmt
            write_u32(w, lock.c_stmt.len() as u32)?;
            for f in &lock.c_stmt {
                w.write_all(&f257_to_u16(f).to_le_bytes())?;
            }

            // accepting_set + offset
            //
            // For H12 we move these degree-0 values behind the seed capsule and keep only zero
            // placeholders in the public package surface. They are restored after hidden-state
            // unlock during decapsulation.
            let a0 = if has_h12 { 0u16 } else { f257_to_u16(&lock.accepting_set[0]) };
            let a1 = if has_h12 { 0u16 } else { f257_to_u16(&lock.accepting_set[1]) };
            let off = if has_h12 { 0u16 } else { f257_to_u16(&lock.offset) };
            w.write_all(&a0.to_le_bytes())?;
            w.write_all(&a1.to_le_bytes())?;
            w.write_all(&off.to_le_bytes())?;

            // coin inputs / gating dims
            write_u32(w, lock.anchor_block_id)?;
            write_u64(w, lock.rep_id)?;
            write_u32(w, lock.poison_blocks)?;

            // sizes
            write_u64(w, lock.x_len as u64)?;
            write_u64(w, lock.pi_len as u64)?;
            write_u64(w, lock.len as u64)?;

            // params
            write_u32(w, lock.params._reserved0)?;
            w.write_all(&lock.params.domain_label)?;

            // coins
            write_u64(w, lock.coins.idx as u64)?;
            w.write_all(&f257_to_u16(&lock.coins.lambda).to_le_bytes())?;
            w.write_all(&f257_to_u16(&lock.coins.rho).to_le_bytes())?;
            w.write_all(&f257_to_u16(&lock.coins.sigma).to_le_bytes())?;
            w.write_all(&f257_to_u16(&lock.coins.c_hit).to_le_bytes())?;

            // basis x dots (ax,bx,gx)
            w.write_all(&f257_to_u16(&lock.basis_x_dots[0]).to_le_bytes())?;
            w.write_all(&f257_to_u16(&lock.basis_x_dots[1]).to_le_bytes())?;
            w.write_all(&f257_to_u16(&lock.basis_x_dots[2]).to_le_bytes())?;

            // basis hints
            for bh in [
                &lock.anchor_basis_hints.alpha,
                &lock.anchor_basis_hints.beta,
                &lock.anchor_basis_hints.gamma,
            ] {
                write_u32(w, bh.hint_blocks_sparse.len() as u32)?;
                for (block_idx, blk) in &bh.hint_blocks_sparse {
                    write_u32(w, *block_idx as u32)?;
                    write_packed_block(w, blk)?;
                }
            }

            // anchor hints
            write_u32(w, lock.anchor_hints.hint_blocks_sparse.len() as u32)?;
            for (block_idx, blk) in &lock.anchor_hints.hint_blocks_sparse {
                write_u32(w, *block_idx as u32)?;
                write_packed_block(w, blk)?;
            }

            // err gate hints
            w.write_all(&lock.err_gate_hints.k.to_le_bytes())?;
            write_u32(w, lock.err_gate_hints.blocks_per_mix)?;
            write_u32(w, lock.err_gate_hints.mixes.len() as u32)?;
            for blk in &lock.err_gate_hints.mixes {
                write_packed_block(w, blk)?;
            }

            // wrapped-key ciphertext entries (canonical count: 1)
            write_u32(w, lock.cts.len() as u32)?;
            for ct in &lock.cts {
                w.write_all(&ct.nonce)?;
                write_u32(w, ct.ct.len() as u32)?;
                w.write_all(&ct.ct)?;
            }
            // encrypted ubits (fixed 32 bytes)
            w.write_all(&lock.ct_ubits)?;
            // payload ciphertext (encrypted under wrapped random payload key)
            w.write_all(&lock.payload_ct.nonce)?;
            write_u32(w, lock.payload_ct.ct.len() as u32)?;
            w.write_all(&lock.payload_ct.ct)?;
        }
        if has_h12 {
            let env = ll.h12_seed_envelope.as_ref().ok_or_else(|| {
                std::io::Error::new(std::io::ErrorKind::InvalidInput, "missing H12 seed envelope")
            })?;
            crate::h12_rcap::write_seed_envelope(w, env)?;
            let mut flags = 0u8;
            if ll.h12_alvo_rep_bundles_digest.is_some() {
                flags |= 1u8;
            }
            if ll.h12_alvo_schedule_digest.is_some() {
                flags |= 1u8 << 1;
            }
            w.write_all(&[flags])?;
            if let Some(d) = ll.h12_alvo_rep_bundles_digest {
                w.write_all(&d)?;
            }
            if let Some(d) = ll.h12_alvo_schedule_digest {
                w.write_all(&d)?;
            }
        }

    }

    Ok(())
}

fn read_lock_package_from_reader(
    r: &mut impl Read,
) -> std::io::Result<(Sp1OneProofWeGateLockPkgManifest, Vec<OneProofLogicalLock>)> {
    const MAX_LOCKS_DEFAULT: usize = 4096;
    const MAX_C_STMT_LEN_DEFAULT: usize = 1 << 20;
    const MAX_CT_BYTES_DEFAULT: usize = 1 << 20;
    const MAX_HINT_BLOCKS_DEFAULT: usize = 1 << 20;
    const MAX_GATE_MIXES_DEFAULT: usize = 1 << 20;

    let mut magic = [0u8; 11];
    r.read_exact(&mut magic)?;
    let is_h08 = &magic == b"LFP1LOCKH08";
    let is_h09 = &magic == b"LFP1LOCKH09";
    let is_h10 = &magic == b"LFP1LOCKH10";
    let is_h11 = &magic == b"LFP1LOCKH11";
    let is_h12 = &magic == b"LFP1LOCKH12";
    if !is_h08 && !is_h09 && !is_h10 && !is_h11 && !is_h12 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "bad lock pkg magic",
        ));
    }

    // Embedded manifest (canonical public metadata).
    let mut stmt_digest = [F257::from(0u64); 32];
    let mut lock_coin_seed = [0u8; 32];
    for f in &mut stmt_digest {
        let mut b2 = [0u8; 2];
        r.read_exact(&mut b2)?;
        *f = u16_to_f257(u16::from_le_bytes(b2));
    }
    r.read_exact(&mut lock_coin_seed)?;
    let manifest = Sp1OneProofWeGateLockPkgManifest {
        stmt_digest,
        lock_coin_seed,
    };

    let max_locks = std::env::var("LFP_ONEPROOF_LOCKPKG_MAX_LOCKS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(MAX_LOCKS_DEFAULT);
    let max_c_stmt_len = std::env::var("LFP_ONEPROOF_LOCKPKG_MAX_C_STMT_LEN")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(MAX_C_STMT_LEN_DEFAULT);
    let max_ct_bytes = std::env::var("LFP_ONEPROOF_LOCKPKG_MAX_CT_BYTES")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(MAX_CT_BYTES_DEFAULT);

    let logical_n = read_u32(r)? as usize;
    if logical_n > max_locks {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("logical lock count exceeds limit (n={} max={})", logical_n, max_locks),
        ));
    }
    let mut out: Vec<OneProofLogicalLock> = Vec::with_capacity(logical_n);
    for _ in 0..logical_n {
        fn read_packed_block(r: &mut impl Read) -> std::io::Result<crate::lockable_ringlwe::PackedF257Block64> {
            let mut tag = [0u8; 1];
            r.read_exact(&mut tag)?;
            match tag[0] {
                0 => {
                    let m = read_u32(r)? as usize;
                    let mut entries: Vec<(u8, u8)> = Vec::with_capacity(m);
                    for _ in 0..m {
                        let mut b2 = [0u8; 2];
                        r.read_exact(&mut b2)?;
                        entries.push((b2[0], b2[1]));
                    }
                    Ok(crate::lockable_ringlwe::PackedF257Block64::Sparse { entries })
                }
                1 => {
                    let mut vals = [0u8; 64];
                    r.read_exact(&mut vals)?;
                    let mask = read_u64(r)?;
                    Ok(crate::lockable_ringlwe::PackedF257Block64::Dense { vals, is256_mask: mask })
                }
                _ => Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "bad packed block tag")),
            }
        }

        let share_index = read_u32(r)?;
        let reps_n = read_u32(r)? as usize;
        if reps_n == 0 {
            return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "logical lock has 0 reps"));
        }
        if reps_n > max_locks {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("rep count exceeds limit (n={} max={})", reps_n, max_locks),
            ));
        }
        let mut reps: Vec<RingLweLockArtifact<F257>> = Vec::with_capacity(reps_n);
        for _ in 0..reps_n {
            // c_stmt
            let c_len = read_u32(r)? as usize;
            if c_len > max_c_stmt_len {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("c_stmt length exceeds limit (len={} max={})", c_len, max_c_stmt_len),
                ));
            }
            let mut c_stmt = Vec::with_capacity(c_len);
            for _ in 0..c_len {
                let mut b = [0u8; 2];
                r.read_exact(&mut b)?;
                c_stmt.push(u16_to_f257(u16::from_le_bytes(b)));
            }

            // accepting_set + offset
            let mut a0b = [0u8; 2];
            let mut a1b = [0u8; 2];
            let mut offb = [0u8; 2];
            r.read_exact(&mut a0b)?;
            r.read_exact(&mut a1b)?;
            r.read_exact(&mut offb)?;
            let accepting_set = [u16_to_f257(u16::from_le_bytes(a0b)), u16_to_f257(u16::from_le_bytes(a1b))];
            let offset = u16_to_f257(u16::from_le_bytes(offb));

            // coin inputs / dims
            let anchor_block_id = read_u32(r)?;
            let rep_id = read_u64(r)?;
            let poison_blocks = read_u32(r)?;

            // sizes
            let x_len: usize = read_u64(r)?
                .try_into()
                .map_err(|_| std::io::Error::new(std::io::ErrorKind::InvalidData, "x_len overflow"))?;
            let pi_len: usize = read_u64(r)?
                .try_into()
                .map_err(|_| std::io::Error::new(std::io::ErrorKind::InvalidData, "pi_len overflow"))?;
            let len: usize = read_u64(r)?
                .try_into()
                .map_err(|_| std::io::Error::new(std::io::ErrorKind::InvalidData, "len overflow"))?;
            // params
            let reserved0 = read_u32(r)?;
            if reserved0 != 2 {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "non-canonical hint encoding (expected params._reserved0=2)",
                ));
            }
            let mut domain_label = [0u8; 32];
            r.read_exact(&mut domain_label)?;
            let params = RingLweParams {
                _reserved0: reserved0,
                domain_label,
            };

            // coins
            let idx = read_u64(r)? as usize;
            let mut lb = [0u8; 2];
            let mut rb = [0u8; 2];
            let mut sb = [0u8; 2];
            let mut chb = [0u8; 2];
            r.read_exact(&mut lb)?;
            r.read_exact(&mut rb)?;
            r.read_exact(&mut sb)?;
            r.read_exact(&mut chb)?;
            let coins = dpp::theorem43::Theorem43Coins {
                idx,
                lambda: u16_to_f257(u16::from_le_bytes(lb)),
                rho: u16_to_f257(u16::from_le_bytes(rb)),
                sigma: u16_to_f257(u16::from_le_bytes(sb)),
                c_hit: u16_to_f257(u16::from_le_bytes(chb)),
            };
            // basis x dots + basis hints (H11+); earlier versions set them to zero/empty.
            let mut basis_x_dots = [u16_to_f257(0u16); 3];
            let mut anchor_basis_hints = crate::lockable_ringlwe::AnchorBasisHints {
                alpha: crate::lockable_ringlwe::BranchHints { hint_blocks_sparse: vec![] },
                beta: crate::lockable_ringlwe::BranchHints { hint_blocks_sparse: vec![] },
                gamma: crate::lockable_ringlwe::BranchHints { hint_blocks_sparse: vec![] },
            };
            if is_h11 || is_h12 {
                for i in 0..3 {
                    let mut b = [0u8; 2];
                    r.read_exact(&mut b)?;
                    basis_x_dots[i] = u16_to_f257(u16::from_le_bytes(b));
                }
                fn read_branch_hints(
                    r: &mut impl Read,
                    max_blocks: usize,
                ) -> std::io::Result<crate::lockable_ringlwe::BranchHints> {
                    let nb = read_u32(r)? as usize;
                    if nb > max_blocks {
                        return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "hint block count exceeds cap"));
                    }
                    let mut v: Vec<(usize, crate::lockable_ringlwe::PackedF257Block64)> = Vec::with_capacity(nb);
                    for _ in 0..nb {
                        let block_idx = read_u32(r)? as usize;
                        let blk = read_packed_block(r)?;
                        v.push((block_idx, blk));
                    }
                    Ok(crate::lockable_ringlwe::BranchHints { hint_blocks_sparse: v })
                }
                let alpha = read_branch_hints(r, MAX_HINT_BLOCKS_DEFAULT)?;
                let beta = read_branch_hints(r, MAX_HINT_BLOCKS_DEFAULT)?;
                let gamma = read_branch_hints(r, MAX_HINT_BLOCKS_DEFAULT)?;
                anchor_basis_hints = crate::lockable_ringlwe::AnchorBasisHints { alpha, beta, gamma };
            }

            // anchor hints
            let nb = read_u32(r)? as usize;
            if nb > MAX_HINT_BLOCKS_DEFAULT {
                return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "hint block count exceeds cap"));
            }
            let mut v: Vec<(usize, crate::lockable_ringlwe::PackedF257Block64)> = Vec::with_capacity(nb);
            for _ in 0..nb {
                let block_idx = read_u32(r)? as usize;
                let blk = read_packed_block(r)?;
                v.push((block_idx, blk));
            }
            let anchor_hints = crate::lockable_ringlwe::BranchHints { hint_blocks_sparse: v };

            // err gate hints
            let mut kb = [0u8; 2];
            r.read_exact(&mut kb)?;
            let k = u16::from_le_bytes(kb);
            let blocks_per_mix = read_u32(r)?;
            let nm = read_u32(r)? as usize;
            if nm > MAX_GATE_MIXES_DEFAULT {
                return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "gate mixes exceeds cap"));
            }
            let mut mixes = Vec::with_capacity(nm);
            for _ in 0..nm {
                mixes.push(read_packed_block(r)?);
            }
            let err_gate_hints = crate::lockable_ringlwe::ErrGateHints { k, blocks_per_mix, mixes };

            // wrapped-key ciphertext entries (canonical count: 1)
            let n_ct = read_u32(r)? as usize;
            if n_ct != 1 {
                return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "wrapped key ciphertext count must be 1"));
            }
            let mut cts: Vec<crate::lockable_ringlwe::LockCiphertext> = Vec::with_capacity(n_ct);
            for _ in 0..n_ct {
                let mut nonce = [0u8; 12];
                r.read_exact(&mut nonce)?;
                let clen = read_u32(r)? as usize;
                if clen > max_ct_bytes {
                    return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "ct length exceeds cap"));
                }
                let mut ct = vec![0u8; clen];
                r.read_exact(&mut ct)?;
                cts.push(crate::lockable_ringlwe::LockCiphertext { nonce, ct });
            }
            // encrypted ubits (H09/H10; H08 sets it to zero for backward compatibility)
            let ct_ubits: [u8; 32] = if is_h09 || is_h10 || is_h11 || is_h12 {
                let mut b = [0u8; 32];
                r.read_exact(&mut b)?;
                b
            } else {
                [0u8; 32]
            };

            // payload ciphertext
            let mut payload_nonce = [0u8; 12];
            r.read_exact(&mut payload_nonce)?;
            let payload_clen = read_u32(r)? as usize;
            if payload_clen > max_ct_bytes {
                return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "payload ct length exceeds cap"));
            }
            let mut payload_ct_bytes = vec![0u8; payload_clen];
            r.read_exact(&mut payload_ct_bytes)?;
            let payload_ct = crate::lockable_ringlwe::LockCiphertext {
                nonce: payload_nonce,
                ct: payload_ct_bytes,
            };

            reps.push(RingLweLockArtifact {
                c_stmt,
                accepting_set,
                offset,
                basis_x_dots,
                anchor_block_id,
                rep_id,
                poison_blocks,
                x_len,
                pi_len,
                len,
                coins,
                params,
                anchor_hints,
                anchor_basis_hints,
                err_gate_hints,
                cts,
                ct_ubits,
                payload_ct,
            });
        }
        let (h12_seed_envelope, h12_alvo_schedule_digest, h12_alvo_rep_bundles_digest) = if is_h12 {
            let env = crate::h12_rcap::read_seed_envelope(r)?;
            let mut tag = [0u8; 1];
            r.read_exact(&mut tag)?;
            let rep_digest = if (tag[0] & 1u8) != 0 {
                let mut d = [0u8; 32];
                r.read_exact(&mut d)?;
                Some(d)
            } else {
                None
            };
            let schedule_digest = if (tag[0] & (1u8 << 1)) != 0 {
                let mut d = [0u8; 32];
                r.read_exact(&mut d)?;
                Some(d)
            } else {
                None
            };
            (Some(env), schedule_digest, rep_digest)
        } else {
            (None, None, None)
        };
        out.push(OneProofLogicalLock {
            share_index,
            reps,
            h12_seed_envelope,
            h12_alvo_schedule_digest,
            h12_alvo_rep_bundles_digest,
        });
    }
    Ok((manifest, out))
}

fn write_lock_package(
    path: &str,
    manifest: &Sp1OneProofWeGateLockPkgManifest,
    logical_locks: &[OneProofLogicalLock],
) -> std::io::Result<()> {
    fn zstd_enabled_for_path(path: &str) -> bool {
        let _ = path;
        // Canonical behavior: compress lock packages with zstd by default.
        //
        // The lock package is public and large; zstd reduces disk+network IO significantly.
        // Readers auto-detect compression via outer magic, so file extensions are not required.
        //
        // Set `LFP_ONEPROOF_LOCKPKG_ZSTD=0` to force raw (debug only).
        !std::env::var("LFP_ONEPROOF_LOCKPKG_ZSTD")
            .ok()
            .is_some_and(|v| v == "0")
    }
    fn zstd_level() -> i32 {
        std::env::var("LFP_ONEPROOF_LOCKPKG_ZSTD_LEVEL")
            .ok()
            .and_then(|s| s.parse::<i32>().ok())
            .unwrap_or(7)
    }

    // Optional wrapper compression:
    // - Outer magic: LFP1LOCKZ3
    // - Payload: zstd frame whose decompressed bytes begin with inner magic LFP1LOCKH08
    const MAGIC_ZSTD_V3: &[u8; 10] = b"LFP1LOCKZ3";

    let f = std::fs::File::create(path)?;
    let mut w = std::io::BufWriter::new(f);
    if zstd_enabled_for_path(path) {
        w.write_all(MAGIC_ZSTD_V3)?;
        let level = zstd_level();
        let mut enc = zstd::stream::write::Encoder::new(w, level)?;
        write_lock_package_to_writer(&mut enc, manifest, logical_locks)?;
        let mut w = enc.finish()?;
        w.flush()?;
        Ok(())
    } else {
        write_lock_package_to_writer(&mut w, manifest, logical_locks)?;
        w.flush()?;
        Ok(())
    }
}

fn read_lock_package(
    path: &str,
) -> std::io::Result<(Sp1OneProofWeGateLockPkgManifest, Vec<OneProofLogicalLock>)> {
    use std::io::{Seek, SeekFrom};

    // - Uncompressed: LFP1LOCKH08 / LFP1LOCKH09 / LFP1LOCKH10 / LFP1LOCKH11 / LFP1LOCKH12
    // - Compressed wrapper: LFP1LOCKZ3 || zstd(LFP1LOCKH0x || ...)
    const MAGIC_ZSTD_V3: &[u8; 10] = b"LFP1LOCKZ3";

    let mut f = std::fs::File::open(path)?;
    // Read enough bytes to disambiguate:
    // - raw magic is 11 bytes
    // - zstd wrapper magic is 10 bytes (prefix)
    let mut magic = [0u8; 11];
    f.read_exact(&mut magic)?;
    const MAGIC_RAW_H08: &[u8; 11] = b"LFP1LOCKH08";
    const MAGIC_RAW_H09: &[u8; 11] = b"LFP1LOCKH09";
    const MAGIC_RAW_H10: &[u8; 11] = b"LFP1LOCKH10";
    const MAGIC_RAW_H11: &[u8; 11] = b"LFP1LOCKH11";
    const MAGIC_RAW_H12: &[u8; 11] = b"LFP1LOCKH12";
    if &magic == MAGIC_RAW_H08
        || &magic == MAGIC_RAW_H09
        || &magic == MAGIC_RAW_H10
        || &magic == MAGIC_RAW_H11
        || &magic == MAGIC_RAW_H12
    {
        f.seek(SeekFrom::Start(0))?;
        let mut r = std::io::BufReader::new(f);
        return read_lock_package_from_reader(&mut r);
    }
    if &magic[0..10] == MAGIC_ZSTD_V3 {
        // Start decoding zstd payload immediately after wrapper magic.
        f.seek(SeekFrom::Start(10))?;
        let r = std::io::BufReader::new(f);
        let mut dec = zstd::stream::read::Decoder::new(r)?;
        return read_lock_package_from_reader(&mut dec);
    }
    Err(std::io::Error::new(
        std::io::ErrorKind::InvalidData,
        "bad lock pkg magic",
    ))
}

fn read_lock_package_from_bytes(
    bytes: &[u8],
) -> std::io::Result<(Sp1OneProofWeGateLockPkgManifest, Vec<OneProofLogicalLock>)> {
    use std::io::Cursor;

    const MAGIC_ZSTD_V3: &[u8; 10] = b"LFP1LOCKZ3";
    const MAGIC_RAW_H08: &[u8; 11] = b"LFP1LOCKH08";
    const MAGIC_RAW_H09: &[u8; 11] = b"LFP1LOCKH09";
    const MAGIC_RAW_H10: &[u8; 11] = b"LFP1LOCKH10";
    const MAGIC_RAW_H11: &[u8; 11] = b"LFP1LOCKH11";
    const MAGIC_RAW_H12: &[u8; 11] = b"LFP1LOCKH12";

    if bytes.len() >= 11 {
        let magic = &bytes[..11];
        if magic == MAGIC_RAW_H08
            || magic == MAGIC_RAW_H09
            || magic == MAGIC_RAW_H10
            || magic == MAGIC_RAW_H11
            || magic == MAGIC_RAW_H12
        {
            let mut cur = Cursor::new(bytes);
            return read_lock_package_from_reader(&mut cur);
        }
    }
    if bytes.len() >= 10 && &bytes[..10] == MAGIC_ZSTD_V3 {
        let cur = Cursor::new(&bytes[10..]);
        let mut dec = zstd::stream::read::Decoder::new(cur)?;
        return read_lock_package_from_reader(&mut dec);
    }
    Err(std::io::Error::new(
        std::io::ErrorKind::InvalidData,
        "bad lock pkg magic",
    ))
}
