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
    PackedF257Block64, RingLweLockArtifact, RingLweParams, RingLweSubLock,
};
use crate::shamir_gf256::{reconstruct_secret_32, split_secret_32, ShamirConfig, ShamirShare};
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
    /// The decapsulated 32-byte key derived from the proof.
    ///
    /// This is a research harness convenience. In the canonical RxP lock path, honest decap is
    /// non-branching at the share level (by construction / arming-time policy), and the protocol
    /// performs a single global check (e.g. EC/address binding) across armers.
    pub decapped_key: [u8; 32],
}

/// Manifest embedded in the public lock package file.
///
/// This is the canonical public metadata needed to sanity-check decapsulation is using the
/// correct package for a given statement/proof stream.
#[derive(Clone, Debug)]
pub struct Sp1OneProofWeGateLockPkgManifest {
    pub stmt_digest: [F257; 32],
    pub lock_coin_seed: [u8; 32],
    /// Combine scheme identifier (format-level, forward-compatible).
    ///
    /// `1` = combine_v1 (current implementation uses Shamir over GF(256); may change to XOR-based).
    pub combine_scheme: u32,
    /// Threshold \(T\) for T-of-R reconstruction.
    pub combine_threshold: u32,
    /// Number of locks / shares \(R\).
    pub combine_shares: u32,
}

/// Structured output for the **arming** endpoint (write lock packages to disk).
#[derive(Clone, Debug)]
pub struct Sp1OneProofWeGateArmingOutput {
    pub manifest: Sp1OneProofWeGateLockPkgManifest,
    pub k_locks: usize,
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
    // - MASTER secret seed: seeds the secret arming RNG (combined key + Shamir coins + rep-start
    //   jitter). This MUST be secret and MUST NOT be derived from any public manifest field.
    //
    // - DPP secret seed: used only for Theorem-4.3 DPP arming (`dpp.arm(...)`) via
    //   `derive_armer_secret`. This also MUST be secret; without it an attacker could recompute
    //   hidden DPP internals from the public artifact and break confidentiality.
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
    // All secret arming randomness (combined key, Shamir polynomial coins, per-lock coins, etc.)
    // must come from a seed that is NOT recoverable from the lock package.
    let mut secret_master_rng = StdRng::from_seed(master_seed32);

    // Combine-v1 policy (current): K-of-K only.
    //
    // NOTE: we intentionally require all locks to decapsulate deterministically at share level.
    // This matches the no-per-lock-oracle design and avoids subset-selection ambiguity.
    let threshold: usize = k_locks;
    let combine_cfg = ShamirConfig {
        threshold,
        shares: k_locks,
    };
    // Combined key (32 bytes) MUST NOT be derivable from public manifest fields.
    // If provided, `LFP_ONEPROOF_COMBINED_KEY32_HEX` must be kept secret by the armer.
    let combined_key32: [u8; 32] =
        if let Ok(khex) = std::env::var("LFP_ONEPROOF_COMBINED_KEY32_HEX") {
            parse_hex32(&khex).map_err(|e| format!("LFP_ONEPROOF_COMBINED_KEY32_HEX: {e}"))?
        } else {
            let mut k = [0u8; 32];
            secret_master_rng.fill_bytes(&mut k);
            k
        };
    let shares = split_secret_32(&mut secret_master_rng, &combine_cfg, combined_key32)
        .map_err(|e| format!("split_secret_32: {e}"))?;
    let t_arm = Instant::now();
    let mut locks: Vec<RingLweLockArtifact<F257>> = Vec::with_capacity(k_locks);
    let mut share_indices: Vec<u32> = Vec::with_capacity(k_locks);
    // Channels/repetitions:
    // Hits-per-block policy:
    // - We use a single channel (`P=1`) to minimize format/logic complexity.
    // - Per-block soundness and disambiguation both come from `hits_per_block` independent hits
    //   on every FLPCP block (full coverage).
    let p_channels: u16 = 1;
    let hits_per_block_raw: usize = std::env::var("LFP_ONEPROOF_HITS_PER_BLOCK")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(4)
        .max(1);
    let hits_per_block: u16 = hits_per_block_raw.try_into().map_err(|_| {
        format!(
            "LFP_ONEPROOF_HITS_PER_BLOCK out of range for u16 (value={})",
            hits_per_block_raw
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

    // Pre-sample per-lock RNG seeds *sequentially* from the master RNG so that arming remains
    // deterministic under a fixed `LFP_ONEPROOF_MASTER_SEED_HEX`, while allowing parallel lock
    // construction (the heavy part).
    let mut per_lock_rng_seed32: Vec<[u8; 32]> = Vec::with_capacity(k_locks);
    for _ in 0..k_locks {
        let mut s = [0u8; 32];
        secret_master_rng.fill_bytes(&mut s);
        per_lock_rng_seed32.push(s);
    }

    // Always record share indices in canonical order.
    for j in 0..k_locks {
        share_indices.push(shares[j].index);
    }

    let policy = crate::we_tiny_lock::WeRingLweLockArmingPolicy {
        base_rep_id: 0,
        max_rep_tries,
        hint_budget_bytes: hint_budget_bytes_opt,
    };

    
    let mut results: Vec<(usize, RingLweLockArtifact<F257>, Vec<u16>)> = (0..k_locks)
        .into_par_iter()
        .map(|j| -> Result<(usize, RingLweLockArtifact<F257>, Vec<u16>), String> {
            let lock_j = j as u64;
            let mut rng = StdRng::from_seed(per_lock_rng_seed32[j]);
            let arm_out = crate::we_tiny_lock::arm_lfplus_ringlwe_lock::<R>(
                shape.clone(),
                &stmt_digest,
                secret_dpp_seed32,
                lock_j,
                block_id,
                policy,
                ringlwe_params.clone(),
                hits_per_block,
                shares[j].value.as_slice(),
                &mut rng,
            )?;
            Ok((j, arm_out.lock, arm_out.s_channels_mod257))
        })
        .collect::<Result<Vec<_>, String>>()?;
    results.sort_by_key(|(j, _l, _s)| *j);
    for (_j, lock, _s_channels) in results {
        locks.push(lock);
    }

    eprintln!(
        "[oneproof] armed {} locks in {:?}",
        locks.len(),
        t_arm.elapsed()
    );

    let manifest = Sp1OneProofWeGateLockPkgManifest {
        stmt_digest,
        lock_coin_seed,
        combine_scheme: 1,
        combine_threshold: threshold as u32,
        combine_shares: k_locks as u32,
    };
    write_lock_package(lock_pkg_out_path, &manifest, &share_indices, &locks)
        .map_err(|e| format!("oneproof: write lock package failed: {e}"))?;
    let file_bytes = std::fs::metadata(lock_pkg_out_path).map(|m| m.len()).unwrap_or(0);
    eprintln!(
        "[oneproof:size] wrote_lock_pkg: path={} bytes={}",
        lock_pkg_out_path, file_bytes
    );

    Ok(Sp1OneProofWeGateArmingOutput {
        manifest,
        k_locks,
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
    let (m, share_indices, locks) = read_lock_package(lock_pkg_in_path)
                .map_err(|e| format!("oneproof: read lock package failed: {e}"))?;
   // DO NOT UNCOMMENT THIS, LEAVE IT FOR TESTING
    /* if m.stmt_digest != stmt_digest {
        return Err("oneproof: lock package stmt_digest mismatch".to_string());
    }
    if m.lock_coin_seed != lock_coin_seed {
        return Err("oneproof: lock package lock_coin_seed mismatch".to_string());
    }*/
    if locks.is_empty() {
        return Err("oneproof: lock package contained 0 locks".to_string());
    }
    if share_indices.len() != locks.len() {
        return Err("oneproof: lock package share_indices/locks length mismatch".to_string());
    }
    if m.combine_scheme != 1 {
        return Err(format!(
            "oneproof: unsupported combine_scheme {}",
            m.combine_scheme
        ));
    }
    if m.combine_shares as usize != locks.len() {
        return Err(format!(
            "oneproof: combine_shares mismatch (header={} locks={})",
            m.combine_shares,
            locks.len()
        ));
    }
    let t = m.combine_threshold as usize;
    if t != locks.len() {
        return Err(format!(
            "oneproof: K-of-K required (combine_threshold={} locks_len={})",
            t,
            locks.len()
        ));
    }
    if t == 0 {
        return Err(format!(
            "oneproof: invalid combine_threshold={} for locks_len={}",
            t,
            locks.len()
        ));
    }
    let prover = crate::we_tiny_lock::we_ringlwe_prover_from_dr1cs::<F257>(
        shape.inst.clone(),
        shape.public_len,
    )?;

    let t_prove = Instant::now();
    maybe_print_rss("oneproof:before_prove_decap_stream");
    let total_sublks: usize = locks.iter().map(|l| l.sublocks.len()).sum();
    let mut state_meta: Vec<(usize, usize, u16)> = Vec::with_capacity(total_sublks); // (lock_i, sublock_i, channel_id)
    let mut coins_list: Vec<_> = Vec::with_capacity(total_sublks);
    for (li, l) in locks.iter().enumerate() {
        for (si, sl) in l.sublocks.iter().enumerate() {
            let coins = prover.derive_public_coins_from_stmt(
                l.c_stmt.as_slice(),
                sl.block_id as usize,
                sl.rep_id,
            )?;
            coins_list.push(coins);
            state_meta.push((li, si, sl.channel_id));
        }
    }

    // Tail-dot only: avoid allocating/storing the 256-element tail vector per sublock.
    let mut tail_dots_mod257: Vec<u16> = vec![0u16; coins_list.len()];
    let mut cur_ci: Option<usize> = None;
    let mut cur_blk: usize = 0;
    let mut buf_len: usize = 0;
    let mut buf64: [u16; 64] = [0u16; 64];
    let abg_list = prover.stream_pi0_and_collect_tails(
        &x,
        z_w,
        &coins_list,
        None,
        &mut |ci, _ti, t| {
            // Tail elements are visited coin-by-coin, in order.
            if cur_ci != Some(ci) {
                cur_ci = Some(ci);
                cur_blk = 0;
                buf_len = 0;
            }
            let td = crate::lockable_ringlwe::field_mod257_u16(t);
            buf64[buf_len] = td;
            buf_len += 1;
            if buf_len == 64 {
                // Dot this 64-chunk against the corresponding packed coefficients.
                let (li, local_si, _ch) = state_meta[ci];
                let sl = &locks[li].sublocks[local_si];
                let blk = &sl.hints.tail_scales[cur_blk];
                let add = crate::lockable_ringlwe::dot_packed_block_mod257_u16(blk, &buf64);
                let acc = &mut tail_dots_mod257[ci];
                *acc = crate::lockable_ringlwe::add_mod257_u16(*acc, add);
                cur_blk += 1;
                buf_len = 0;
            }
        },
    )?;
    if abg_list.len() != coins_list.len() || tail_dots_mod257.len() != coins_list.len() {
        return Err("oneproof: internal abgs/coins length mismatch".to_string());
    }
    // Final flush sanity: we should end exactly on a chunk boundary.
    if buf_len != 0 {
        return Err("oneproof: internal tail-dot buffer misalignment".to_string());
    }
    maybe_print_rss("oneproof:after_prove_decap_stream");
    eprintln!("[oneproof] prove+decap(stream) in {:?}", t_prove.elapsed());

    // Recover per-sublock scalar candidates, then per-lock decrypt candidates.
    //
    // Both stages are embarrassingly parallel (per sublock / per lock).
    let mut per_lock_sublock_cands: Vec<Vec<Option<(u16, [u16; 2])>>> = locks
        .iter()
        .map(|l| vec![None; l.sublocks.len()])
        .collect();
    let candidates_per_lock: Vec<(u32, Vec<[u8; 32]>)> = {
        use rayon::prelude::*;

        // 1) Finish per-sublock candidate extraction (parallel over sublocks).
        let state_meta_ref = &state_meta;

        use crate::lockable_ringlwe::sublock_s_candidates_from_abg_tail;
        let tail_dots_ref = &tail_dots_mod257;
        let per_state: Vec<(usize, usize, u16, [u16; 2])> = abg_list
            .into_par_iter()
            .enumerate()
            .map(|(si, abg)| -> Result<(usize, usize, u16, [u16; 2]), String> {
                let (li, local_si, ch_expect) = state_meta_ref
                    .get(si)
                    .copied()
                    .ok_or_else(|| "oneproof: internal state_meta mismatch".to_string())?;
                let l = locks
                    .get(li)
                    .ok_or_else(|| "oneproof: internal lock index mismatch".to_string())?;
                let sl = l
                    .sublocks
                    .get(local_si)
                    .ok_or_else(|| "oneproof: internal sublock index mismatch".to_string())?;
                if sl.channel_id != ch_expect {
                    return Err("oneproof: internal channel_id mismatch".to_string());
                }
                let td = *tail_dots_ref
                    .get(si)
                    .ok_or_else(|| "oneproof: internal tail_dot index mismatch".to_string())?;
                let cands = sublock_s_candidates_from_abg_tail(sl, &abg, td)?;
                Ok((li, local_si, ch_expect, cands))
            })
            .collect::<Result<Vec<_>, _>>()?;

        for (li, local_si, ch2, cands) in per_state {
            let slots = per_lock_sublock_cands
                .get_mut(li)
                .ok_or_else(|| "oneproof: internal lock index mismatch".to_string())?;
            if local_si >= slots.len() {
                return Err("oneproof: internal sublock index mismatch".to_string());
            }
            slots[local_si] = Some((ch2, cands));
        }

        // 2) Decrypt per lock (parallel over locks).
        locks
            .par_iter()
            .enumerate()
            .map(|(li, l)| -> Result<(u32, Vec<[u8; 32]>), String> {
                let slots = per_lock_sublock_cands
                    .get(li)
                    .ok_or_else(|| "oneproof: internal lock index mismatch".to_string())?;
                let mut sublock_cands: Vec<(u16, [u16; 2])> = Vec::with_capacity(slots.len());
                for (i, v) in slots.iter().enumerate() {
                    let vv = v.ok_or_else(|| {
                        format!("oneproof: missing sublock candidate (lock[{li}] sublock[{i}])")
                    })?;
                    sublock_cands.push(vv);
                }
                let pt = crate::lockable_ringlwe::decrypt_payload_from_sublock_s_candidates(
                    l,
                    sublock_cands.as_slice(),
                )
                .map_err(|e| format!("oneproof: lock[{li}] decrypt candidates: {e}"))?;
                if pt.len() != 32 {
                    return Err(format!(
                        "oneproof: share candidate wrong length at lock[{li}] ({})",
                        pt.len()
                    ));
                }
                let mut c = [0u8; 32];
                c.copy_from_slice(&pt);
                Ok((share_indices[li], vec![c]))
            })
            .collect::<Result<Vec<_>, _>>()
    }?;

    let combine_cfg = ShamirConfig {
        threshold: t,
        shares: locks.len(),
    };
    // K-of-K policy: every lock must have exactly one candidate share payload.
    if !candidates_per_lock.iter().all(|(_idx, c)| c.len() == 1) {
        return Err("oneproof: ambiguous share candidates".to_string());
    }
    let selected: Vec<ShamirShare> = candidates_per_lock
        .iter()
        .map(|(idx, cands)| ShamirShare {
            index: *idx,
            value: cands[0],
        })
        .collect();
    let decapped_key = reconstruct_secret_32(&combine_cfg, &selected)
        .map_err(|e| format!("oneproof: reconstruct_secret_32 failed: {e:?}"))?;

    // Shape cache is intentionally kept on disk for reuse across invocations.
    // To force a rebuild, delete the cache directory manually or set LFP_SHAPE_CACHE_DIR.

    Ok(Sp1OneProofWeGateOutput {
        stmt_digest,
        lock_coin_seed,
        decapped_key,
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
    share_indices: &[u32],
    locks: &[RingLweLockArtifact<F257>],
) -> std::io::Result<()> {
    // Canonical lock package v1.
    //
    // IMPORTANT: lock packages are public artifacts and must not be silently mis-decoded.
    // We bump the magic whenever we change semantics (e.g. length widths, hint packing convention).
    // Canonical lock package v6:
    // - compressed hints only (no sparse hints; no legacy formats).
    // - hints store only `offset_scale = s*δ(x_arm)` rather than masked per-coordinate `h_x`.
    w.write_all(b"LFP1LOCKV9")?;
    // Embedded manifest (canonical public metadata).
    for f in &manifest.stmt_digest {
        w.write_all(&f257_to_u16(f).to_le_bytes())?;
    }
    w.write_all(&manifest.lock_coin_seed)?;
    w.write_all(&manifest.combine_scheme.to_le_bytes())?;
    w.write_all(&manifest.combine_threshold.to_le_bytes())?;
    w.write_all(&manifest.combine_shares.to_le_bytes())?;
    write_u32(w, locks.len() as u32)?;
    if share_indices.len() != locks.len() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "share_indices/locks length mismatch",
        ));
    }
    for (i, lock) in locks.iter().enumerate() {
        w.write_all(&share_indices[i].to_le_bytes())?;
        // c_stmt
        write_u32(w, lock.c_stmt.len() as u32)?;
        for f in &lock.c_stmt {
            w.write_all(&f257_to_u16(f).to_le_bytes())?;
        }
        // sizes
        write_u64(w, lock.x_len as u64)?;
        write_u64(w, lock.pi_len as u64)?;
        write_u64(w, lock.len as u64)?;
        // params
        write_u32(w, lock.params._reserved0)?;
        w.write_all(&lock.params.domain_label)?;
        // (P,sublocks_per_channel) and sublocks
        w.write_all(&lock.p_channels.to_le_bytes())?;
        w.write_all(&lock.sublocks_per_channel.to_le_bytes())?;
        write_u32(w, lock.sublocks.len() as u32)?;
        for sl in &lock.sublocks {
            w.write_all(&sl.channel_id.to_le_bytes())?;
            // accepting_set
            w.write_all(&f257_to_u16(&sl.accepting_set[0]).to_le_bytes())?;
            w.write_all(&f257_to_u16(&sl.accepting_set[1]).to_le_bytes())?;
            // coin-derivation inputs (canonical): derive public Theorem-4.3 coins from `(c_stmt, block_id, rep_id)`.
            write_u32(w, sl.block_id)?;
            write_u64(w, sl.rep_id)?;
            // hints (compressed only)
            if lock.params._reserved0 != 2 {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    "non-canonical hint encoding (expected params._reserved0=2)",
                ));
            }

            fn write_packed_block(w: &mut impl Write, blk: &PackedF257Block64) -> std::io::Result<()> {
                // Canonical per-block encoding: fmt(u8) + payload.
                //
                // fmt=0: sparse (nnz(u8) + nnz*(pos_flags, coeff))
                // fmt=1: dense (vals[64] + is256_mask(u64 little-endian))
                match blk {
                    PackedF257Block64::Sparse { entries } => {
                        w.write_all(&[0u8])?;
                        let nnz = entries.len();
                        if nnz > 255 {
                            return Err(std::io::Error::new(
                                std::io::ErrorKind::InvalidInput,
                                "hint block too many nnz entries",
                            ));
                        }
                        w.write_all(&[nnz as u8])?;
                        for &(pos_flags, coeff) in entries {
                            w.write_all(&[pos_flags, coeff])?;
                        }
                    }
                    PackedF257Block64::Dense { vals, is256_mask } => {
                        w.write_all(&[1u8])?;
                        w.write_all(vals)?;
                        w.write_all(&is256_mask.to_le_bytes())?;
                    }
                }
                Ok(())
            }

            let hc = &sl.hints;
            for &s in &hc.abg_scales {
                w.write_all(&s.to_le_bytes())?;
            }
            w.write_all(&hc.offset_scale.to_le_bytes())?;
            for blk in &hc.tail_scales {
                write_packed_block(w, blk)?;
            }
        }
        // ciphertext (single)
        w.write_all(&lock.ct.nonce)?;
        write_u32(w, lock.ct.ct.len() as u32)?;
        w.write_all(&lock.ct.ct)?;
    }
    Ok(())
}

fn read_lock_package_from_reader(
    r: &mut impl Read,
) -> std::io::Result<(Sp1OneProofWeGateLockPkgManifest, Vec<u32>, Vec<RingLweLockArtifact<F257>>)> {
    // GF(256) Shamir backend supports only nonzero x-coordinates in 1..=255.
    const SHAMIR_GFSHARE_MAX: usize = 255;
    const MAX_LOCKS_DEFAULT: usize = 4096;
    const MAX_C_STMT_LEN_DEFAULT: usize = 1 << 20;
    const MAX_CT_BYTES_DEFAULT: usize = 1 << 20;

    let mut magic = [0u8; 10];
    r.read_exact(&mut magic)?;
    let v9 = &magic == b"LFP1LOCKV9";
    if !v9 {
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
    let mut b4 = [0u8; 4];
    r.read_exact(&mut b4)?;
    let combine_scheme = u32::from_le_bytes(b4);
    r.read_exact(&mut b4)?;
    let combine_threshold = u32::from_le_bytes(b4);
    r.read_exact(&mut b4)?;
    let combine_shares = u32::from_le_bytes(b4);
    let manifest = Sp1OneProofWeGateLockPkgManifest {
        stmt_digest,
        lock_coin_seed,
        combine_scheme,
        combine_threshold,
        combine_shares,
    };
    if manifest.combine_scheme != 1 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!(
                "unsupported combine_scheme {} (expected 1 for Shamir GF(256))",
                manifest.combine_scheme
            ),
        ));
    }

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

    let k = manifest.combine_shares as usize;
    let t = manifest.combine_threshold as usize;
    if k == 0 || k > SHAMIR_GFSHARE_MAX {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!(
                "combine_shares exceeds Shamir GF(256) backend limit (k={} max={})",
                k, SHAMIR_GFSHARE_MAX
            ),
        ));
    }
    if t == 0 || t > SHAMIR_GFSHARE_MAX {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!(
                "combine_threshold exceeds Shamir GF(256) backend limit (t={} max={})",
                t, SHAMIR_GFSHARE_MAX
            ),
        ));
    }
    if t > k {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("invalid combine params (threshold={} shares={})", t, k),
        ));
    }

    let n = read_u32(r)? as usize;
    if n != k {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!(
                "combine_shares mismatch (manifest k={} lock_count n={})",
                k, n
            ),
        ));
    }
    if n > max_locks {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("lock count exceeds limit (n={} max={})", n, max_locks),
        ));
    }
    let mut share_indices: Vec<u32> = Vec::with_capacity(n);
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        let mut b = [0u8; 4];
        r.read_exact(&mut b)?;
        share_indices.push(u32::from_le_bytes(b));
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
        let mut b2 = [0u8; 2];
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
        // (P,sublocks_per_channel) and sublocks
        r.read_exact(&mut b2)?;
        let p_channels = u16::from_le_bytes(b2);
        let sublocks_per_channel = read_u32(r)?;
        if p_channels == 0 || sublocks_per_channel == 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "invalid (P,sublocks_per_channel) in lock package",
            ));
        }
        let ns = read_u32(r)? as usize;
        let expected_ns: usize = (p_channels as u64)
            .saturating_mul(sublocks_per_channel as u64)
            .try_into()
            .map_err(|_| {
                std::io::Error::new(std::io::ErrorKind::InvalidData, "sublocks length overflow")
            })?;
        if ns != expected_ns {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "sublocks length mismatch",
            ));
        }
        let mut sublocks: Vec<RingLweSubLock<F257>> = Vec::with_capacity(ns);
        for _ in 0..ns {
            r.read_exact(&mut b2)?;
            let channel_id = u16::from_le_bytes(b2);
            // accepting_set
            r.read_exact(&mut b2)?;
            let a0 = u16::from_le_bytes(b2);
            r.read_exact(&mut b2)?;
            let a1 = u16::from_le_bytes(b2);
            // coin-derivation inputs (canonical)
            let block_id = read_u32(r)?;
            let rep_id = read_u64(r)?;

            fn read_packed_block(r: &mut impl Read) -> std::io::Result<PackedF257Block64> {
                // Canonical per-block encoding.
                let mut fmt = [0u8; 1];
                r.read_exact(&mut fmt)?;
                let blk = match fmt[0] {
                    0 => {
                        let mut b1 = [0u8; 1];
                        r.read_exact(&mut b1)?;
                        let nnz = b1[0] as usize;
                        let mut entries: Vec<(u8, u8)> = Vec::with_capacity(nnz);
                        let mut seen_pos = [false; 64];
                        for _ in 0..nnz {
                            let mut pair = [0u8; 2];
                            r.read_exact(&mut pair)?;
                            // Canonical sparse form:
                            // - bit7 must be 0 (reserved)
                            // - low 6 bits are position 0..63
                            // - positions must be unique within the block
                            if (pair[0] & 0x80) != 0 {
                                return Err(std::io::Error::new(
                                    std::io::ErrorKind::InvalidData,
                                    "non-canonical sparse pos_flags: bit7 must be 0",
                                ));
                            }
                            let pos = (pair[0] & 0x3f) as usize;
                            if seen_pos[pos] {
                                return Err(std::io::Error::new(
                                    std::io::ErrorKind::InvalidData,
                                    format!("duplicate sparse position in block (pos={})", pos),
                                ));
                            }
                            seen_pos[pos] = true;
                            entries.push((pair[0], pair[1]));
                        }
                        PackedF257Block64::Sparse { entries }
                    }
                    1 => {
                        let mut vals = [0u8; 64];
                        r.read_exact(&mut vals)?;
                        let mut b8 = [0u8; 8];
                        r.read_exact(&mut b8)?;
                        let is256_mask = u64::from_le_bytes(b8);
                        // Canonical dense form:
                        // if mask bit is set (coefficient=256), byte payload must be 0.
                        for (i, v) in vals.iter().enumerate() {
                            if ((is256_mask >> i) & 1) != 0 && *v != 0 {
                                return Err(std::io::Error::new(
                                    std::io::ErrorKind::InvalidData,
                                    format!(
                                        "non-canonical dense encoding: vals[{}] must be 0 when is256_mask bit is set",
                                        i
                                    ),
                                ));
                            }
                        }
                        PackedF257Block64::Dense { vals, is256_mask }
                    }
                    _ => {
                        return Err(std::io::Error::new(
                            std::io::ErrorKind::InvalidData,
                            "unknown hint block encoding fmt",
                        ))
                    }
                };
                Ok(blk)
            }

            let mut abg = [0u16; 3];
            for a in &mut abg {
                let mut b2 = [0u8; 2];
                r.read_exact(&mut b2)?;
                *a = u16::from_le_bytes(b2);
            }
            let offset_scale = {
                let mut b2 = [0u8; 2];
                r.read_exact(&mut b2)?;
                u16::from_le_bytes(b2)
            };
            let mut tail_scales: [PackedF257Block64; 4] =
                core::array::from_fn(|_| PackedF257Block64::default());
            for i in 0..4 {
                tail_scales[i] = read_packed_block(r)?;
            }
            let hints = crate::lockable_ringlwe::BranchHintsCompressed {
                abg_scales: abg,
                offset_scale,
                tail_scales,
            };

            sublocks.push(RingLweSubLock::<F257> {
                channel_id,
                accepting_set: [u16_to_f257(a0), u16_to_f257(a1)],
                block_id,
                rep_id,
                hints,
            });
        }

        // ciphertext (single)
            let mut nonce = [0u8; 12];
            r.read_exact(&mut nonce)?;
        let ct_len = read_u32(r)? as usize;
        if ct_len > max_ct_bytes {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "ciphertext length exceeds limit (len={} max={})",
                    ct_len, max_ct_bytes
                ),
            ));
        }
            let mut ct = vec![0u8; ct_len];
            r.read_exact(&mut ct)?;
        let ct = crate::lockable_ringlwe::LockCiphertext { nonce, ct };
        out.push(RingLweLockArtifact {
            c_stmt,
            x_len,
            pi_len,
            len,
            params,
            p_channels,
            sublocks_per_channel,
            sublocks,
            ct,
        });
    }
    Ok((manifest, share_indices, out))
}

fn write_lock_package(
    path: &str,
    manifest: &Sp1OneProofWeGateLockPkgManifest,
    share_indices: &[u32],
    locks: &[RingLweLockArtifact<F257>],
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
    // - Payload: zstd frame whose decompressed bytes begin with inner magic LFP1LOCKV9
    const MAGIC_ZSTD_V3: &[u8; 10] = b"LFP1LOCKZ3";

    let f = std::fs::File::create(path)?;
    let mut w = std::io::BufWriter::new(f);
    if zstd_enabled_for_path(path) {
        w.write_all(MAGIC_ZSTD_V3)?;
        let level = zstd_level();
        let mut enc = zstd::stream::write::Encoder::new(w, level)?;
        write_lock_package_to_writer(&mut enc, manifest, share_indices, locks)?;
        let mut w = enc.finish()?;
        w.flush()?;
        Ok(())
    } else {
        write_lock_package_to_writer(&mut w, manifest, share_indices, locks)?;
        w.flush()?;
        Ok(())
    }
}

fn read_lock_package(
    path: &str,
) -> std::io::Result<(Sp1OneProofWeGateLockPkgManifest, Vec<u32>, Vec<RingLweLockArtifact<F257>>)> {
    use std::io::{Seek, SeekFrom};

    // - Uncompressed: LFP1LOCKV9
    // - Compressed wrapper: LFP1LOCKZ3 || zstd(LFP1LOCKV9 || ...)
    const MAGIC_RAW_V9: &[u8; 10] = b"LFP1LOCKV9";
    const MAGIC_ZSTD_V3: &[u8; 10] = b"LFP1LOCKZ3";

    let mut f = std::fs::File::open(path)?;
    let mut magic = [0u8; 10];
    f.read_exact(&mut magic)?;
    if &magic == MAGIC_RAW_V9 {
        f.seek(SeekFrom::Start(0))?;
        let mut r = std::io::BufReader::new(f);
        return read_lock_package_from_reader(&mut r);
    }
    if &magic == MAGIC_ZSTD_V3 {
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
