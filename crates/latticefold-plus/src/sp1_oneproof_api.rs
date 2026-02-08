//! API wrapper for the `examples/lf_plus_sp1_oneproof.rs` harness.
//!
//! This exists so downstream crates (like PVUGC) can call the SP1→LF+→WE-gate path **in-process**
//! without spawning binaries or scraping logs.
//!
//! This is research plumbing and intentionally returns only a small structured summary.

#![cfg(feature = "we_gate")]

use cyclotomic_rings::rings::{GetPoseidonParams, GoldilocksPoseidonConfig as PC};
use cyclotomic_rings::rings::GoldilocksRing64 as R;
use ark_ff::PrimeField;
use latticefold::commitment::AjtaiCommitmentScheme;
use latticefold::transcript::poseidon::F257;
use latticefold::transcript::Transcript;
use latticefold::transcript::bytes::field_to_bytes_le_fixed;
use stark_rings::PolyRing;
use stark_rings_linalg::SparseMatrix;

use sha2::{Digest, Sha256};
use std::sync::Arc;
use std::time::Instant;

use rand::{rngs::StdRng, SeedableRng};

use crate::lin::LinearizedVerify;
use crate::lockable_ringlwe::{PackedF257Block64, RingLweLockArtifact, RingLweParams};
use crate::shamir_gf256::{reconstruct_secret_32, split_secret_32, ShamirConfig, ShamirShare};
use crate::utils::maybe_print_rss;
use crate::we_statement::{encode_public_x, we_statement_hash_lf_plus, LFP_WE_GATE_DIGEST_V1};
use std::io::{Read, Write};

type BFSmall = <<R as PolyRing>::BaseRing as ark_ff::Field>::BasePrimeField;

/// Structured output for downstream wiring (no stdout parsing).
#[derive(Clone, Debug)]
pub struct Sp1OneProofWeGateOutput {
    pub stmt_digest: [u8; 32],
    pub lock_coin_seed: [u8; 32],
    /// The decapsulated 32-byte key derived from the proof.
    ///
    /// This is a research harness convenience: the underlying RingLWE lock has two branches, so
    /// we embed a short self-authenticating tag into the lock payload and pick the unique branch
    /// whose payload validates.
    pub decapped_key: [u8; 32],
}

/// Manifest embedded in the public lock package file.
///
/// This is the canonical public metadata needed to sanity-check decapsulation is using the
/// correct package for a given statement/proof stream.
#[derive(Clone, Debug)]
pub struct Sp1OneProofWeGateLockPkgManifest {
    pub stmt_digest: [u8; 32],
    pub lock_coin_seed: [u8; 32],
    /// Combine scheme identifier (format-level, forward-compatible).
    ///
    /// `1` = combine_v1 (current implementation uses Shamir over GF(256); may change to XOR-based).
    pub combine_scheme: u32,
    /// Threshold \(T\) for T-of-R reconstruction.
    pub combine_threshold: u32,
    /// Number of locks / shares \(R\).
    pub combine_shares: u32,
    /// Public tag binding the combined key to the statement.
    ///
    /// This lets decap select the unique valid reconstruction without any per-lock oracle.
    pub combined_key_tag: [u8; 32],
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
    public_inputs: &[BFSmall],
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
    if public_inputs.len() != hdr.num_public {
        return Err(format!(
            "public_inputs length mismatch: got={} expected_num_public={}",
            public_inputs.len(),
            hdr.num_public
        ));
    }

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

    let r1cs_digest = hdr.digest;
    let stmt_digest =
        we_statement_hash_lf_plus::<R>(vk_hash, r1cs_digest, LFP_WE_GATE_DIGEST_V1, &we_params, public_inputs);

    // Derive one representative coin seed (j=0) like the example.
    const ARMER_SEED: [u8; 32] = *b"LFP_ARMER_SEED_V1_00000000000000";
    let lock_j: u64 = 0;
    let lock_coin_seed: [u8; 32] = {
        let mut h = Sha256::new();
        h.update(b"LFP_LOCK_COIN_V1");
        h.update(&ARMER_SEED);
        h.update(&stmt_digest);
        h.update(&lock_j.to_le_bytes());
        h.finalize().into()
    };

    // WE arming must not run the prover or compute an assignment.
    //
    // Instead, we generate a self-consistent verifier transcript schedule and build/load the
    // statement-bound tiny-gate *shape only*.
    let n_lin_proofs = 1usize;
    let trace = crate::we_gate_arith::poseidon_trace_schedule_for_plus_with_public_inputs::<R>(
        public_inputs,
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
        std::fs::create_dir_all(&base).map_err(|e| format!("create shape cache dir failed: {e}"))?;
        base
    };

    let pairs: Vec<(usize, usize)> = vec![(0, 0)];

    // Map statement public inputs into tiny-gate public prefix bytes (F257).
    let public_inputs_f257: Vec<F257> = public_inputs
        .iter()
        .flat_map(|x| {
            field_to_bytes_le_fixed::<BFSmall>(x)
                .into_iter()
                .map(|b| F257::from(b as u64))
        })
        .collect();

    // WE arming must not depend on a satisfying assignment.
    let shape = crate::we_gate_arith::build_or_load_we_plus_tiny_shape::<R>(
        &trace,
        &we_params,
        public_inputs,
        n_lin_proofs,
        mlen_mats as usize,
        &pairs,
        &out_dir,
    )
    .map_err(|e| format!("build_or_load_we_plus_tiny_shape: {e}"))?;

    // Arm the lock package(s).
    let block_id = 0usize;
    let ringlwe_params = RingLweParams::default();
    let mut rng = StdRng::from_seed(lock_coin_seed);

    // Combine-v1: split a single 32-byte secret into R shares with threshold T.
    // This is statement-bound (seeded by lock_coin_seed + stmt_digest) and proof-agnostic.
    let threshold: usize = std::env::var("LFP_ONEPROOF_T")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(3)
        .min(k_locks)
        .max(2);
    let combine_cfg = ShamirConfig {
        threshold,
        shares: k_locks,
    };
    let combined_key32: [u8; 32] = {
        let mut h = Sha256::new();
        h.update(b"LFP_ONEPROOF_COMBINED_KEY_V1");
        h.update(&lock_coin_seed);
        h.update(&stmt_digest);
        h.finalize().into()
    };
    let shares = split_secret_32(&mut rng, &combine_cfg, combined_key32)
        .map_err(|e| format!("split_secret_32: {e}"))?;
    let combined_key_tag: [u8; 32] = {
        let mut h = Sha256::new();
        h.update(b"LFP_ONEPROOF_COMBINED_TAG_V1");
        h.update(&combined_key32);
        h.update(&stmt_digest);
        h.finalize().into()
    };

    let t_arm = Instant::now();
    let mut locks: Vec<RingLweLockArtifact<F257>> = Vec::with_capacity(k_locks);
    let mut share_indices: Vec<u32> = Vec::with_capacity(k_locks);
    for j in 0..k_locks {
        let lock_j = j as u64;
        let mut rep_id = j as u64;
        share_indices.push(shares[j].index);
        let lock = loop {
            match crate::we_tiny_lock::arm_lfplus_ringlwe_lock::<R>(
                shape.clone(),
                &we_params,
                &public_inputs_f257,
                stmt_digest,
                ARMER_SEED,
                lock_j,
                block_id,
                rep_id,
                ringlwe_params.clone(),
                shares[j].value.as_slice(),
                &mut rng,
            ) {
                Ok(lock) => break lock,
                Err(e) if e.contains("shifted accepting set contains 0") => {
                    rep_id += 1;
                    continue;
                }
                Err(e) => return Err(e),
            }
        };
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
        combined_key_tag,
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

    // Statement digest binds vk, r1cs digest, gate-id, params, and public inputs.
    if bundle.r1lf_digest != cache.stats.digest {
        return Err(format!(
            "witness bundle r1lf_digest does not match SP1_R1LF cache: bundle=0x{} cache=0x{}",
            hex32(&bundle.r1lf_digest),
            hex32(&cache.stats.digest)
        ));
    }
    let (vk_hash, _committed_values_digest) = bundle.public_inputs;
    let r1cs_digest = cache.stats.digest;
    let stmt_digest =
        we_statement_hash_lf_plus::<R>(vk_hash, r1cs_digest, LFP_WE_GATE_DIGEST_V1, &we_params, &public_inputs);

    // Derive one representative coin seed (j=0) like the example.
    const ARMER_SEED: [u8; 32] = *b"LFP_ARMER_SEED_V1_00000000000000";
    let lock_j: u64 = 0;
    let lock_coin_seed: [u8; 32] = {
        let mut h = Sha256::new();
        h.update(b"LFP_LOCK_COIN_V1");
        h.update(&ARMER_SEED);
        h.update(&stmt_digest);
        h.update(&lock_j.to_le_bytes());
        h.finalize().into()
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
        std::fs::create_dir_all(&base).map_err(|e| format!("create shape cache dir failed: {e}"))?;
        base
    };

    let pairs: Vec<(usize, usize)> = vec![(0, 0)];

    // Map statement public inputs into tiny-gate public prefix bytes (F257).
    //
    // IMPORTANT: Transcript public inputs are absorbed as fixed-width base-field byte strings
    // (`prime_field_to_bytes_le_fixed`), i.e. 8 bytes per Goldilocks base-field element.
    //
    // Therefore, the tiny gate's statement public prefix must be these bytes, not the field
    // elements themselves, and certainly not the centered BabyBear embedding.
    // IMPORTANT: these must be the *exact bytes* absorbed into the recorded transcript:
    // `PoseidonTranscript::absorb_field_element` encodes field elements using
    // `field_to_bytes_le_fixed` (fixed width, little-endian).
    //
    // Therefore we derive bytes from the **Goldilocks field elements** `public_inputs`,
    // not from the raw BabyBear u64 witness words.
    let public_inputs_f257: Vec<F257> = public_inputs
        .iter()
        .flat_map(|x| field_to_bytes_le_fixed::<BFSmall>(x).into_iter().map(|b| F257::from(b as u64)))
        .collect();

    let (shape, assignment) = crate::we_gate_arith::build_or_load_we_plus_tiny_dr1cs::<R>(
        &trace,
        &we_params,
        &public_inputs_f257,
        &proof,
        m0.len(),
        &pairs,
        &out_dir,
    )
    .map_err(|e| format!("build_or_load_we_plus_tiny_dr1cs: {e}"))?;

    // Optional sanity: the witness must satisfy the armed instance.
    // This can be expensive for large gates; enable only when debugging.
    if std::env::var("LFP_WE_GATE_CHECK_SAT").ok().as_deref() == Some("1") {
        shape
            .inst
            .check(&assignment)
            .map_err(|e| format!("we gate armed instance not satisfied: {e}"))?;
    }

    // -------------------------------------------------------------------------
    // Ring-LWE streaming decapsulation (Theorem 4.3 canonical path)
    // -------------------------------------------------------------------------
    let public_len = shape.public_len;
    let x = encode_public_x::<F257>(&we_params, &public_inputs_f257);
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
    if m.stmt_digest != stmt_digest {
        return Err("oneproof: lock package stmt_digest mismatch".to_string());
    }
    if m.lock_coin_seed != lock_coin_seed {
        return Err("oneproof: lock package lock_coin_seed mismatch".to_string());
    }
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
    if t < 2 || t > locks.len() {
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
    // Production sizing note:
    // `proof_len` here is the number of streamed π elements over F257. If transmitted as raw bytes,
    // the π stream is ~1 byte/element.
    let pi_len = prover.proof_len();
    let pi_stream_bytes = pi_len as u128;
    let pi_stream_gib = (pi_stream_bytes as f64) / (1024.0 * 1024.0 * 1024.0);
    let pi_blocks = (pi_len + (64 - 1)) / 64;

    // Lock artifact sizing (hints dominate).
    // Packed F257 per hinted block: block_idx(u32:4) + coeffs(64 bytes) + mask(u64:8) = 76 bytes.
    let bytes_per_hint_block = 4usize + 64usize + 8usize;
    let mut hinted_blocks_total = 0usize;
    let mut ct_bytes_total = 0usize;
    for l in &locks {
        hinted_blocks_total += l.branch_hints[0].hint_blocks_sparse.len();
        hinted_blocks_total += l.branch_hints[1].hint_blocks_sparse.len();
        ct_bytes_total += l.cts[0].nonce.len()
            + l.cts[0].ct.len()
            + l.cts[1].nonce.len()
            + l.cts[1].ct.len();
    }
    let hint_bytes_est = hinted_blocks_total.saturating_mul(bytes_per_hint_block);
    eprintln!(
        "[oneproof:size] pi_len={} (~{:.2} GiB if 1 byte/elem) pi_blocks={} locks={} hinted_blocks_total={} hint_bytes_est≈{} ct_bytes_total={}",
        pi_len,
        pi_stream_gib,
        pi_blocks,
        locks.len(),
        hinted_blocks_total,
        hint_bytes_est,
        ct_bytes_total
    );

    let t_prove = Instant::now();
    maybe_print_rss("oneproof:before_prove_decap_stream");
    let mut states: Vec<_> = Vec::with_capacity(locks.len());
    for (i, l) in locks.iter().enumerate() {
        let st = l
            .decap_state(&x)
            .map_err(|e| format!("oneproof: decap_state[{i}] failed: {e}"))?;
        states.push(st);
    }
    let coins_list = locks.iter().map(|l| l.coins.clone()).collect::<Vec<_>>();
    let mut err: Option<String> = None;
    let tails = prover.stream_pi0_and_collect_tails(&x, z_w, &coins_list, &mut |chunk| {
        if err.is_some() {
            return;
        }
        for st in &mut states {
            if let Err(e) = st.absorb_chunk(chunk) {
                err = Some(e);
                break;
            }
        }
    })?;
    if let Some(e) = err {
        return Err(format!("oneproof: stream decap absorb failed: {e}"));
    }
    if tails.len() != states.len() {
        return Err(format!(
            "oneproof: tails/locks mismatch (tails={} locks={})",
            tails.len(),
            states.len()
        ));
    }
    for (i, st) in states.iter_mut().enumerate() {
        st.absorb_chunk(&tails[i])?;
    }
    maybe_print_rss("oneproof:after_prove_decap_stream");
    eprintln!("[oneproof] prove+decap(stream) in {:?}", t_prove.elapsed());

    // Decrypt all locks: each produces 2 candidate 32-byte share payloads (unauthenticated).
    let mut candidates_per_lock: Vec<(u32, [[u8; 32]; 2])> = Vec::with_capacity(locks.len());
    for (i, st) in states.into_iter().enumerate() {
        let [pt0, pt1] = st
            .finish_decrypt_candidates()
            .map_err(|e| format!("oneproof: finish_decrypt_candidates[{i}]: {e}"))?;
        if pt0.len() != 32 || pt1.len() != 32 {
            return Err(format!(
                "oneproof: share candidate wrong length at lock[{i}] ({} / {})",
                pt0.len(),
                pt1.len()
            ));
        }
        let mut c0 = [0u8; 32];
        let mut c1 = [0u8; 32];
        c0.copy_from_slice(&pt0);
        c1.copy_from_slice(&pt1);
        candidates_per_lock.push((share_indices[i], [c0, c1]));
    }

    // Combine-v1: try 2^T branch assignments on the first T locks and pick the unique candidate
    // whose combined-key tag matches the package header.
    let subset = &candidates_per_lock[..t];
    let n_cands = subset.len();
    let combine_cfg = ShamirConfig {
        threshold: t,
        shares: locks.len(),
    };
    let mut picked: Option<[u8; 32]> = None;
    for mask in 0u64..(1u64 << n_cands) {
        let selected: Vec<ShamirShare> = subset
            .iter()
            .enumerate()
            .map(|(i, (idx, cands))| {
                let branch = ((mask >> i) & 1) as usize;
                ShamirShare {
                    index: *idx,
                    value: cands[branch],
                }
            })
            .collect();
        if let Ok(candidate) = reconstruct_secret_32(&combine_cfg, &selected) {
            let tag: [u8; 32] = {
                let mut h = Sha256::new();
                h.update(b"LFP_ONEPROOF_COMBINED_TAG_V1");
                h.update(&candidate);
                h.update(&stmt_digest);
                h.finalize().into()
            };
            if tag == m.combined_key_tag {
                if picked.is_some() {
                    return Err("oneproof: multiple combined-key candidates matched tag (unexpected)".to_string());
                }
                picked = Some(candidate);
            }
        }
    }
    let decapped_key = picked.ok_or_else(|| "oneproof: failed to reconstruct combined key".to_string())?;

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
    w.write_all(b"LFP1LOCK64")?;
    // Embedded manifest (canonical public metadata).
    w.write_all(&manifest.stmt_digest)?;
    w.write_all(&manifest.lock_coin_seed)?;
    w.write_all(&manifest.combine_scheme.to_le_bytes())?;
    w.write_all(&manifest.combine_threshold.to_le_bytes())?;
    w.write_all(&manifest.combine_shares.to_le_bytes())?;
    w.write_all(&manifest.combined_key_tag)?;
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
        // accepting_set + offset
        w.write_all(&f257_to_u16(&lock.accepting_set[0]).to_le_bytes())?;
        w.write_all(&f257_to_u16(&lock.accepting_set[1]).to_le_bytes())?;
        w.write_all(&f257_to_u16(&lock.offset).to_le_bytes())?;
        // sizes
        write_u64(w, lock.x_len as u64)?;
        write_u64(w, lock.pi_len as u64)?;
        write_u64(w, lock.len as u64)?;
        // coins
        write_u64(w, lock.coins.idx as u64)?;
        w.write_all(&f257_to_u16(&lock.coins.lambda).to_le_bytes())?;
        w.write_all(&f257_to_u16(&lock.coins.rho).to_le_bytes())?;
        w.write_all(&f257_to_u16(&lock.coins.sigma).to_le_bytes())?;
        // params
        write_u32(w, lock.params._reserved0)?;
        w.write_all(&lock.params.domain_label)?;
        // branch hints
        for b in 0..2 {
            write_u32(w, lock.branch_hints[b].hint_blocks_sparse.len() as u32)?;
            for (block_idx, blk) in &lock.branch_hints[b].hint_blocks_sparse {
                write_u32(w, *block_idx as u32)?;
                w.write_all(&blk.coeffs)?;
                write_u64(w, blk.mask_256)?;
            }
        }
        // ciphertexts
        for b in 0..2 {
            w.write_all(&lock.cts[b].nonce)?;
            write_u32(w, lock.cts[b].ct.len() as u32)?;
            w.write_all(&lock.cts[b].ct)?;
        }
    }
    Ok(())
}

fn read_lock_package_from_reader(
    r: &mut impl Read,
) -> std::io::Result<(Sp1OneProofWeGateLockPkgManifest, Vec<u32>, Vec<RingLweLockArtifact<F257>>)> {
    let mut magic = [0u8; 9];
    r.read_exact(&mut magic)?;
    if &magic != b"LFP1LOCK64" {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "bad lock pkg magic",
        ));
    }

    // Embedded manifest (canonical public metadata).
    let mut stmt_digest = [0u8; 32];
    let mut lock_coin_seed = [0u8; 32];
    r.read_exact(&mut stmt_digest)?;
    r.read_exact(&mut lock_coin_seed)?;
    let mut b4 = [0u8; 4];
    r.read_exact(&mut b4)?;
    let combine_scheme = u32::from_le_bytes(b4);
    r.read_exact(&mut b4)?;
    let combine_threshold = u32::from_le_bytes(b4);
    r.read_exact(&mut b4)?;
    let combine_shares = u32::from_le_bytes(b4);
    let mut combined_key_tag = [0u8; 32];
    r.read_exact(&mut combined_key_tag)?;
    let manifest = Sp1OneProofWeGateLockPkgManifest {
        stmt_digest,
        lock_coin_seed,
        combine_scheme,
        combine_threshold,
        combine_shares,
        combined_key_tag,
    };

    let n = read_u32(r)? as usize;
    let mut share_indices: Vec<u32> = Vec::with_capacity(n);
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        let mut b = [0u8; 4];
        r.read_exact(&mut b)?;
        share_indices.push(u32::from_le_bytes(b));
        // c_stmt
        let c_len = read_u32(r)? as usize;
        let mut c_stmt = Vec::with_capacity(c_len);
        for _ in 0..c_len {
            let mut b = [0u8; 2];
            r.read_exact(&mut b)?;
            c_stmt.push(u16_to_f257(u16::from_le_bytes(b)));
        }
        // accepting_set + offset
        let mut b2 = [0u8; 2];
        r.read_exact(&mut b2)?;
        let a0 = u16::from_le_bytes(b2);
        r.read_exact(&mut b2)?;
        let a1 = u16::from_le_bytes(b2);
        r.read_exact(&mut b2)?;
        let off = u16::from_le_bytes(b2);
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
        // coins
        let idx = read_u64(r)? as usize;
        r.read_exact(&mut b2)?;
        let lambda = u16::from_le_bytes(b2);
        r.read_exact(&mut b2)?;
        let rho = u16::from_le_bytes(b2);
        r.read_exact(&mut b2)?;
        let sigma = u16::from_le_bytes(b2);
        // params
        let reserved0 = read_u32(r)?;
        let mut domain_label = [0u8; 32];
        r.read_exact(&mut domain_label)?;
        let params = RingLweParams {
            _reserved0: reserved0,
            domain_label,
        };
        // hints
        let mut branch_hints = [
            crate::lockable_ringlwe::BranchHints {
                hint_blocks_sparse: Vec::new(),
            },
            crate::lockable_ringlwe::BranchHints {
                hint_blocks_sparse: Vec::new(),
            },
        ];
        for bi in 0..2 {
            let nh = read_u32(r)? as usize;
            let mut v = Vec::with_capacity(nh);
            for _ in 0..nh {
                let block_idx = read_u32(r)? as usize;
                let mut coeffs = [0u8; 64];
                r.read_exact(&mut coeffs)?;
                let mask_256 = read_u64(r)?;
                v.push((block_idx, PackedF257Block64 { coeffs, mask_256 }));
            }
            branch_hints[bi].hint_blocks_sparse = v;
        }
        // ciphertexts
        let mut cts = [
            crate::lockable_ringlwe::LockCiphertext {
                nonce: [0u8; 12],
                ct: Vec::new(),
            },
            crate::lockable_ringlwe::LockCiphertext {
                nonce: [0u8; 12],
                ct: Vec::new(),
            },
        ];
        for bi in 0..2 {
            let mut nonce = [0u8; 12];
            r.read_exact(&mut nonce)?;
            let ct_len = read_u32(r)? as usize;
            let mut ct = vec![0u8; ct_len];
            r.read_exact(&mut ct)?;
            cts[bi] = crate::lockable_ringlwe::LockCiphertext { nonce, ct };
        }
        out.push(RingLweLockArtifact {
            c_stmt,
            accepting_set: [u16_to_f257(a0), u16_to_f257(a1)],
            offset: u16_to_f257(off),
            x_len,
            pi_len,
            len,
            coins: dpp::theorem43::Theorem43Coins::<F257> {
                idx,
                lambda: u16_to_f257(lambda),
                rho: u16_to_f257(rho),
                sigma: u16_to_f257(sigma),
            },
            params,
            branch_hints,
            cts,
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
    let f = std::fs::File::create(path)?;
    let mut w = std::io::BufWriter::new(f);
    write_lock_package_to_writer(&mut w, manifest, share_indices, locks)?;
    w.flush()?;
    Ok(())
}

fn read_lock_package(
    path: &str,
) -> std::io::Result<(Sp1OneProofWeGateLockPkgManifest, Vec<u32>, Vec<RingLweLockArtifact<F257>>)> {
    let f = std::fs::File::open(path)?;
    let mut r = std::io::BufReader::new(f);
    read_lock_package_from_reader(&mut r)
}
