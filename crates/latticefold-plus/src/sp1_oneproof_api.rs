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
use std::sync::Arc;
use std::time::Instant;

use rand::{rngs::OsRng, rngs::StdRng, RngCore, SeedableRng};

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
            Ok((
                p,
                OneProofLogicalLock {
                    share_index: shares[p].0,
                    reps: arm_out.reps,
                },
            ))
        })
        .collect::<Result<Vec<_>, String>>()?;
    results.sort_by_key(|(j, _l)| *j);
    for (_p, ll) in results {
        logical_locks.push(ll);
    }

    if std::env::var("LFP_H12_RCAP_ESTIMATE").ok().is_some_and(|v| v != "0") {
        let mut all_surfaces = Vec::new();
        for ll in &logical_locks {
            let checks = crate::h12_rcap::capsule_checks_from_logical_lock(ll.reps.as_slice());
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
            eprintln!(
                "[oneproof:h12] share_index={} checks={} g_cap={} pi_pos_exact={} pi_pos_cons={} pi_blocks_exact={} pi_blocks_cons={} aadp_exact_bytes={} aadp_cons_bytes={}",
                ll.share_index,
                est.check_count,
                est.g_cap,
                est.touched_pi_positions_exact,
                est.touched_pi_positions_conservative,
                est.touched_packed_pi_blocks_exact,
                est.touched_packed_pi_blocks_conservative,
                est.aadp_size_exact_bytes,
                est.aadp_size_conservative_bytes,
            );
            all_surfaces.extend(surfaces);
        }
        let pkg_est = crate::h12_rcap::estimate_capsule_schedule(
            all_surfaces.as_slice(),
            crate::h12_rcap::H12_RCAP_PACK_D,
            32,
        );
        eprintln!(
            "[oneproof:h12:package] checks={} g_cap={} pi_pos_exact={} pi_pos_cons={} pi_blocks_exact={} pi_blocks_cons={} aadp_exact_bytes={} aadp_cons_bytes={}",
            pkg_est.check_count,
            pkg_est.g_cap,
            pkg_est.touched_pi_positions_exact,
            pkg_est.touched_pi_positions_conservative,
            pkg_est.touched_packed_pi_blocks_exact,
            pkg_est.touched_packed_pi_blocks_conservative,
            pkg_est.aadp_size_exact_bytes,
            pkg_est.aadp_size_conservative_bytes,
        );
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
    let mut f0 = w_host.clone();
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
    let (m, logical_locks) = read_lock_package(lock_pkg_in_path)
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

    let t_prove = Instant::now();
    maybe_print_rss("oneproof:before_prove_decap_stream");

    let _stmt_bytes64 = stmt_digest_to_bytes64(&stmt_digest);

    let rep_capacity: usize = logical_locks.iter().map(|ll| ll.reps.len()).sum();
    let mut flat_locks: Vec<&RingLweLockArtifact<F257>> = Vec::with_capacity(rep_capacity);
    let mut flat_share_indices: Vec<u32> = Vec::with_capacity(rep_capacity);
    let mut flat_poison_blocks: Vec<usize> = Vec::with_capacity(rep_capacity);
    let mut flat_states = Vec::with_capacity(rep_capacity);
    let mut coin_ranges: Vec<(usize, usize)> = Vec::with_capacity(rep_capacity);
    let mut anchor_coin_idx: Vec<usize> = Vec::with_capacity(rep_capacity);
    let mut all_coins = Vec::new();

    for ll in &logical_locks {
        if ll.reps.is_empty() {
            return Err(format!(
                "oneproof: logical lock share_index={} has 0 reps",
                ll.share_index
            ));
        }
        for l in &ll.reps {
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
            flat_locks.push(l);
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
    };
    let abg_all =
        prover.stream_pi0_and_collect_abg_full(&x, z_w, &all_coins, Some(&mut on_pi0))?;
    if abg_all.len() != all_coins.len() {
        return Err("oneproof: shared abg length mismatch".to_string());
    }

    // Phase 1: compute per-rep candidate `s` sets (constant-shape 1..=256 scan).
    let mut indexed_s_hits = flat_states
        .into_par_iter()
        .enumerate()
        .map(|(i, st)| -> Result<(usize, (u16, Vec<u16>)), String> {
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
            let s_sets =
                st.finish_s_candidate_sets_with_gate(g, None, Some(u), Some(gamma), Some((alpha, beta)))?;
            let hits = s_sets.into_iter().next().unwrap_or_else(|| (1u16..=256u16).collect());
            Ok((i, (g, hits)))
        })
        .collect::<Result<Vec<_>, String>>()?;
    indexed_s_hits.sort_by_key(|(i, _)| *i);
    let per_rep_hits: Vec<(u16, Vec<u16>)> = indexed_s_hits.into_iter().map(|(_, v)| v).collect();

    // Phase 2: intersect `s` candidates across reps of the same logical lock, then decrypt only
    // those `s` values per rep.
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

fn write_u64(w: &mut impl Write, v: u64) -> std::io::Result<()> {
    w.write_all(&v.to_le_bytes())
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

    // Sparse-hints + residual-gate lock encoding.
    //
    // H10: H09 minus the high-term LUT ciphertext (we no longer ship any LUT carrier).
    w.write_all(b"LFP1LOCKH11")?;
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
            w.write_all(&f257_to_u16(&lock.accepting_set[0]).to_le_bytes())?;
            w.write_all(&f257_to_u16(&lock.accepting_set[1]).to_le_bytes())?;
            w.write_all(&f257_to_u16(&lock.offset).to_le_bytes())?;

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
    if !is_h08 && !is_h09 && !is_h10 && !is_h11 {
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
            if is_h11 {
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
            let ct_ubits: [u8; 32] = if is_h09 || is_h10 || is_h11 {
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
        out.push(OneProofLogicalLock { share_index, reps });
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

    // - Uncompressed: LFP1LOCKH08 / LFP1LOCKH09 / LFP1LOCKH10 / LFP1LOCKH11
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
    if &magic == MAGIC_RAW_H08 || &magic == MAGIC_RAW_H09 || &magic == MAGIC_RAW_H10 || &magic == MAGIC_RAW_H11 {
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
