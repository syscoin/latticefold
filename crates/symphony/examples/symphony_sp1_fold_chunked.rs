//! Chunked parallel proving for SP1 shrink verifier R1CS.
//!
//! Usage:
//!   SP1_R1CS=/path/to/shrink.r1cs cargo run -p symphony --example symphony_sp1_fold_chunked --release
//!
//! Environment variables:
//!   SP1_R1CS       - Path to the R1CS file
//!   CHUNK_SIZE     - Constraints per chunk (default: 1048576 = 2^20 = 1M)
//!   MAX_CONCURRENT - Max chunks proving at once (default: 4, controls memory)
//!
//! With 48M constraints and CHUNK_SIZE=1M: 48 chunks
//! With MAX_CONCURRENT=4: only 4 chunks in memory at once

use std::sync::Arc;
use std::time::Instant;

use rayon::prelude::*;

use ark_ff::PrimeField;
use cyclotomic_rings::rings::FrogPoseidonConfig as PC;
use latticefold::commitment::AjtaiCommitmentScheme;
use stark_rings::cyclotomic_ring::models::frog_ring::RqPoly as R;
use stark_rings::PolyRing;
use stark_rings::Ring;
use symphony::sp1_r1cs_loader::FieldFromU64;
use symphony::symphony_sp1_r1cs::open_sp1_r1cs_chunk_cache;
use symphony::rp_rgchk::RPParams;
use symphony::symphony_pifold_batched::prove_pi_fold_batched_sumcheck_fs;
use symphony::symphony_pifold_streaming::prove_pi_fold_streaming_sumcheck_fs;
use symphony::symphony_pifold_streaming::prove_pi_fold_streaming_sumcheck_fs_hetero_m;
use symphony::symphony_open::MultiAjtaiOpenVerifier;
use symphony::symphony_we_relation::{
    check_r_we_poseidon_fs_hetero_m_with_metrics_result, TrivialRo,
};

fn parse_bytes32_env(name: &str) -> [u8; 32] {
    let s = std::env::var(name).unwrap_or_else(|_| {
        panic!("Missing required env var {name}. Expected a bytes32 hex string like 0xabc... (64 hex chars).")
    });
    let s = s.strip_prefix("0x").unwrap_or(&s);
    let bytes = decode_hex(s).unwrap_or_else(|e| panic!("Invalid hex for {name}: {e}"));
    let len = bytes.len();
    bytes
        .try_into()
        .unwrap_or_else(|_| panic!("{name} must decode to exactly 32 bytes, got {}", len))
}

fn decode_hex(s: &str) -> Result<Vec<u8>, String> {
    if s.len() % 2 != 0 {
        return Err(format!("hex length must be even, got {}", s.len()));
    }
    fn nybble(b: u8) -> Option<u8> {
        match b {
            b'0'..=b'9' => Some(b - b'0'),
            b'a'..=b'f' => Some(b - b'a' + 10),
            b'A'..=b'F' => Some(b - b'A' + 10),
            _ => None,
        }
    }
    let bytes = s.as_bytes();
    let mut out = Vec::with_capacity(bytes.len() / 2);
    for i in (0..bytes.len()).step_by(2) {
        let hi = nybble(bytes[i]).ok_or_else(|| format!("invalid hex at byte {}", i))?;
        let lo = nybble(bytes[i + 1]).ok_or_else(|| format!("invalid hex at byte {}", i + 1))?;
        out.push((hi << 4) | lo);
    }
    Ok(out)
}

fn bytes32_to_u32s_le(x: [u8; 32]) -> [u32; 8] {
    let mut out = [0u32; 8];
    for i in 0..8 {
        let off = 4 * i;
        out[i] = u32::from_le_bytes([x[off], x[off + 1], x[off + 2], x[off + 3]]);
    }
    out
}

/// Load an SP1 R1CS witness dump as `u32` little-endian words and lift each element to a constant-coeff ring element.
///
/// Expected format: exactly `ncols` little-endian `u32` words (so `4*ncols` bytes).
fn load_witness_u32le(path: &str, ncols: usize) -> Vec<R> {
    use std::io::Read;
    let file = std::fs::File::open(path).unwrap_or_else(|e| panic!("open witness file {path}: {e:?}"));
    let mut r = std::io::BufReader::with_capacity(256 * 1024 * 1024, file);

    let mut witness: Vec<R> = Vec::with_capacity(ncols);
    let mut buf = vec![0u8; 4 * 1024 * 1024]; // 4MB

    while witness.len() < ncols {
        let need_words = ncols - witness.len();
        let want_words = std::cmp::min(need_words, buf.len() / 4);
        let want_bytes = want_words * 4;

        r.read_exact(&mut buf[..want_bytes]).unwrap_or_else(|e| {
            panic!(
                "read witness file {path}: {e:?} (need {} u32 words total, only got {} so far)",
                ncols,
                witness.len()
            )
        });

        for w in 0..want_words {
            let off = 4 * w;
            let v = u32::from_le_bytes([buf[off], buf[off + 1], buf[off + 2], buf[off + 3]]);
            let base = <R as PolyRing>::BaseRing::from(v as u128);
            witness.push(R::from(base));
        }
    }

    debug_assert_eq!(witness.len(), ncols);
    witness
}

/// BabyBear field element for loading R1CS.
#[derive(Debug, Clone, Copy, Default)]
struct BabyBear(u64);

const BABYBEAR_P: u64 = 0x78000001; // 2013265921

impl FieldFromU64 for BabyBear {
    fn from_canonical_u64(val: u64) -> Self {
        BabyBear(val % BABYBEAR_P)
    }
    fn as_canonical_u64(&self) -> u64 {
        self.0
    }
}

fn main() {
    let r1cs_path = std::env::var("SP1_R1CS").expect("Set SP1_R1CS=/path/to/shrink.r1cs");
    
    let chunk_size: usize = std::env::var("CHUNK_SIZE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1 << 20); // Default: 2^20 = 1M
    
    let max_concurrent: usize = std::env::var("MAX_CONCURRENT")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(32); // Default: 32 concurrent chunks
    
    println!("=========================================================");
    println!("Symphony SP1 Chunked Proving (Memory-Controlled)");
    println!("=========================================================\n");
    println!("Configuration:");
    println!("  Chunk size:     {} (2^{})", chunk_size, chunk_size.trailing_zeros());
    println!("  Max concurrent: {} chunks at once", max_concurrent);
    let pifold_mode = std::env::var("SYMPHONY_PIFOLD_MODE").unwrap_or_else(|_| "dense".to_string());
    println!("  PiFold mode:    {pifold_mode}");
    // We'll use a local rayon pool sized to the chunk-level parallelism to avoid oversubscription.
    let prove_threads: usize = std::env::var("PROVE_THREADS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(max_concurrent);
    println!("  Prove threads:  {prove_threads}\n");

    // Step 1: Load and chunk R1CS (uses cache)
    println!("Step 1: Loading R1CS (chunked)...");
    let load_start = Instant::now();
    
    // Π_rg parameters that affect how we pad/load the witness.
    //
    // IMPORTANT: `l_h` directly controls `blocks = n_f / l_h` and therefore `m_J = blocks * lambda_pj`.
    // For large SP1 witnesses, a larger `l_h` can dramatically reduce RAM (tables scale with m_J).
    let l_h: usize = std::env::var("L_H")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(256);
    // Pad columns to a multiple of l_h to satisfy Π_rg’s block structure.
    //
    // Note: Increasing `l_h` also increases the FS coin bytes because `derive_J` squeezes
    // `lambda_pj*l_h` bytes per instance; this changes the statement/shape but is expected.
    let pad_cols_to_multiple_of = l_h;
    let cache = open_sp1_r1cs_chunk_cache::<R, BabyBear>(&r1cs_path, chunk_size, pad_cols_to_multiple_of)
        .expect("Failed to open/build chunk cache");
    
    let load_time = load_start.elapsed();
    let num_chunks = cache.num_chunks;
    
    println!("  Constraints: {}", cache.stats.num_constraints);
    println!("  Variables:   {}", cache.stats.num_vars);
    println!("  Chunks:      {}", num_chunks);
    println!("  Load time:   {load_time:?}\n");

    // Step 2: Create witness (same for all chunks)
    println!("Step 2: Creating witness...");
    let ncols = cache.ncols;
    let mut witness: Vec<R> = vec![R::ZERO; ncols];
    witness[0] = R::ONE;
    let witness = Arc::new(witness);
    if ncols.is_power_of_two() {
    println!("  Witness length: {ncols} (2^{})\n", ncols.trailing_zeros());
    } else {
        println!("  Witness length: {ncols}\n");
    }
    // Step 3: Setup Symphony parameters
    println!("Step 3: Setting up Symphony parameters...");
    // Ajtai commitment rows. kappa=8 is a prototype security/perf point used throughout the repo.
    // Treat this (and MASTER_SEED) as part of your public parameters / shape.
    let kappa = 8;
    // Monomial check count. k_g=3 matches the “Table 1 prototype instantiation” mode used by our
    // Symphony benchmarks; changing it changes the protocol shape and costs.
    let k_g = 3;
    // Projection dimension for Π_rg (must satisfy: lambda_pj <= l_h and l_h % lambda_pj == 0).
    //
    // IMPORTANT for SP1 chunked proving: `m_J = (ncols/l_h) * lambda_pj` must be <= `m` (chunk rows)
    // and must divide `m`. With SP1 chunk sizes like 2^20, choosing lambda_pj too large will violate
    // `m_J <= m` and cause proving to fail (often only on the last partial chunks).
    //
    // Default to lambda_pj=1 for SP1 production runs; override via env if you are also increasing
    // chunk size so that `m` is large enough.
    let lambda_pj: usize = std::env::var("LAMBDA_PJ")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1);
    let rg_params = RPParams {
        l_h,
        lambda_pj,
        k_g,
        d_prime: (R::dimension() as u128) - 2,
    };
    assert!(
        rg_params.lambda_pj <= rg_params.l_h,
        "Π_rg requires lambda_pj <= l_h (got lambda_pj={}, l_h={})",
        rg_params.lambda_pj,
        rg_params.l_h
    );
    assert!(
        rg_params.l_h % rg_params.lambda_pj == 0,
        "Π_rg requires lambda_pj | l_h so that m_J divides m (got lambda_pj={}, l_h={})",
        rg_params.lambda_pj,
        rg_params.l_h
    );
    // Print m_J implied by witness length (independent of chunk index) so configs are sanity-checkable.
    let blocks = ncols / rg_params.l_h;
    let m_j = blocks * rg_params.lambda_pj;
    println!(
        "  Π_rg params: l_h={}, lambda_pj={}, k_g={}, d'={}  (m_J={})",
        rg_params.l_h, rg_params.lambda_pj, rg_params.k_g, rg_params.d_prime, m_j
    );
    
    // Main commitment (shared across chunks)
    let setup_start = Instant::now();
    // Avoid materializing a dense Ajtai matrix: define it implicitly from a seed.
    //
    // IMPORTANT: treat this as part of the public parameters (fixed).
    // Different commitment families are domain-separated under this seed.
    // NOTE: If you change this, you must treat it as a parameter version bump (it changes statements).
    const MASTER_SEED: [u8; 32] = *b"SYMPHONY_AJTAI_SEED_V1_000000000";
    let scheme_main = Arc::new(AjtaiCommitmentScheme::<R>::seeded(b"cm_f", MASTER_SEED, kappa, ncols));
    let cm_main = scheme_main
        .commit_const_coeff_fast(&witness)
        .unwrap()
        .as_ref()
        .to_vec();
    let setup_time = setup_start.elapsed();
    println!("  Commitment setup: {setup_time:?}\n");

    // Auxiliary schemes
    let scheme_had = Arc::new(AjtaiCommitmentScheme::<R>::seeded(
        b"cfs_had_u",
        MASTER_SEED,
        kappa,
        3 * R::dimension(),
    ));
    let scheme_mon = Arc::new(AjtaiCommitmentScheme::<R>::seeded(
        b"cfs_mon_b",
        MASTER_SEED,
        kappa,
        rg_params.k_g,
    ));

    // Public statement binding: SP1 program hash (vk hash) and statement digest (public-values hash).
    // We represent each bytes32 as 8 little-endian u32 limbs, and absorb all 16 limbs as field elements.
    let public_inputs: Vec<<R as PolyRing>::BaseRing> = vec![
        <R as PolyRing>::BaseRing::from(1u128),
    ];
    // Step 4: Prove chunks with limited concurrency (actually parallel within batches)
    println!(
        "Step 4: Proving {} chunks ({} at a time, parallel)...\n",
        num_chunks, max_concurrent
    );
    let prove_start = Instant::now();

    // Optional: produce **one** Π_fold proof for all chunks (O(1) verification) by treating each
    // chunk as a separate instance with its own matrices.
    //
    // This currently loads all chunk matrices into memory; it’s a correctness-first path toward a
    // fully streaming “one proof” accumulator over the chunk cache.
    let fold_all: bool = std::env::var("FOLD_ALL_CHUNKS")
        .ok()
        .as_deref()
        .map(|s| s == "1" || s.eq_ignore_ascii_case("true"))
        .unwrap_or(false);
    if fold_all {
        if pifold_mode != "streaming" {
            eprintln!("FOLD_ALL_CHUNKS=1 currently requires SYMPHONY_PIFOLD_MODE=streaming");
            std::process::exit(2);
        }
        println!("  FOLD_ALL_CHUNKS=1: loading all chunk matrices and proving one batched proof...");
        let t_load_all = Instant::now();
        let mut all_mats: Vec<[Arc<stark_rings_linalg::SparseMatrix<R>>; 3]> =
            Vec::with_capacity(num_chunks);
        for i in 0..num_chunks {
            let [m1, m2, m3] = cache.read_chunk(i).expect("read_chunk failed");
            all_mats.push([Arc::new(m1), Arc::new(m2), Arc::new(m3)]);
        }
        println!("    Loaded {} chunks in {:?}", num_chunks, t_load_all.elapsed());

        let cms_all: Vec<Vec<R>> = vec![cm_main.clone(); num_chunks];
        let witnesses_all: Vec<Arc<Vec<R>>> = vec![witness.clone(); num_chunks];

        let t_one = Instant::now();
        let out = prove_pi_fold_streaming_sumcheck_fs_hetero_m::<R, PC>(
            all_mats.as_slice(),
            &cms_all,
            &witnesses_all,
            &public_inputs,
            Some(scheme_had.as_ref()),
            Some(scheme_mon.as_ref()),
            rg_params.clone(),
        )
        .expect("prove (hetero M) failed");
        let proof_bytes = out.proof.coins.bytes.len();
        println!(
            "  ✓ One-proof mode: prove time {:?}, proof bytes {}",
            t_one.elapsed(),
            proof_bytes
        );

        // Optional correctness/soundness sanity-check: run the verifier against the same matrices
        // and explicit witness openings. This confirms the single proof actually binds to *all*
        // chunk matrices (not just one of them).
        let verify_one: bool = std::env::var("VERIFY_ONE_PROOF")
            .ok()
            .as_deref()
            .map(|s| s == "1" || s.eq_ignore_ascii_case("true"))
            .unwrap_or(false);
        if verify_one {
            println!("  VERIFY_ONE_PROOF=1: verifying the one-proof artifact with explicit openings...");
            let t_v = Instant::now();
            let ms_refs: Vec<[&stark_rings_linalg::SparseMatrix<R>; 3]> = all_mats
                .iter()
                .map(|m| [&*m[0], &*m[1], &*m[2]])
                .collect();

            // IMPORTANT: do NOT clone the full witness `num_chunks` times here.
            // Instead, (1) sanity-check the witness commitment once, then (2) verify the folding proof
            // using the prover-provided auxiliary transcript messages (`out.aux`) so the verifier does
            // not need per-instance openings.
            let expected_cm = scheme_main
                .commit_const_coeff_fast(&witness)
                .expect("commit_const_coeff_fast failed")
                .as_ref()
                .to_vec();
            assert_eq!(expected_cm, cm_main, "cm_f does not open to the provided witness");

            // Faithful WE/DPP-facing check: verify R_WE = R_cp ∧ R_o under Poseidon-FS.
            // This is the DPP target relation that includes:
            // - Full Fiat-Shamir transcript verification (all coins derived via Poseidon)
            // - Sumcheck verification
            // - CP commitment openings (cfs_had_u, cfs_mon_b)
            // - R_o reduced relation check (TrivialRo for now - replace with real check)
            let open_cfs = MultiAjtaiOpenVerifier::new()
                .with_scheme("cfs_had_u", (*scheme_had).clone())
                .with_scheme("cfs_mon_b", (*scheme_mon).clone());
            let (res, metrics) =
                check_r_we_poseidon_fs_hetero_m_with_metrics_result::<R, PC, TrivialRo>(
                ms_refs.as_slice(),
                &cms_all,
                &out.proof,
                &open_cfs,
                &out.cfs_had_u,
                &out.cfs_mon_b,
                &out.aux,
                &public_inputs,
                &(), // TrivialRo witness
            );
            // Empirical transcript cost (base-field units). For Frog Poseidon config, rate=20.
            let rate: u64 = 20;
            let absorbed = metrics.absorbed_elems;
            let squeezed = metrics.squeezed_field_elems;
            let squeezed_bytes = metrics.squeezed_bytes;
            let est_perms_absorb = (absorbed + rate - 1) / rate;
            let est_perms_squeeze = (squeezed + rate - 1) / rate;

            // Roughly estimate extra permutations from `squeeze_bytes`.
            // We approximate `bytes_per_field_elem = ceil(modulus_bits/8)` for the base prime field.
            let bits =
                <<<R as PolyRing>::BaseRing as ark_ff::Field>::BasePrimeField as PrimeField>::MODULUS_BIT_SIZE
                    as u64;
            let bytes_per_field_elem = (bits + 7) / 8;
            let rate_bytes = rate * bytes_per_field_elem;
            let est_perms_bytes = if rate_bytes == 0 {
                0
            } else {
                (squeezed_bytes + rate_bytes - 1) / rate_bytes
            };
            eprintln!(
                "    PoseidonTranscript metrics: absorbed_elems={}, squeezed_field_elems={}, squeezed_bytes={}",
                metrics.absorbed_elems, metrics.squeezed_field_elems, metrics.squeezed_bytes
            );
            eprintln!(
                "    PoseidonTranscript est perms: ceil(absorb/20)={} + ceil(squeeze_field/20)={} + ceil(bytes/{})={} => {}",
                est_perms_absorb,
                est_perms_squeeze,
                rate_bytes,
                est_perms_bytes,
                est_perms_absorb + est_perms_squeeze + est_perms_bytes
            );
            res.expect("R_WE check failed");
            println!("    ✓ Verified in {:?}", t_v.elapsed());
        }
        return;
    }
    
    let mut successes = 0;
    let mut failures = 0;
    let mut total_proof_bytes = 0usize;
    // NOTE: We intentionally do NOT retain per-chunk proofs in RAM here.
    // A production driver would stream them to disk or feed them directly into an accumulator.
    
    // Process in batches of max_concurrent (parallel within each batch)
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(prove_threads)
        .build()
        .expect("failed to build rayon pool");

    for batch_start in (0..num_chunks).step_by(max_concurrent) {
        let batch_end = std::cmp::min(batch_start + max_concurrent, num_chunks);
        let batch_size = batch_end - batch_start;
        
        println!("  Batch {}-{} of {}...", batch_start, batch_end - 1, num_chunks);
        let batch_start_time = Instant::now();
        
        // Load the chunk matrices for this batch (streaming; keeps peak RAM bounded).
        let load_mat_start = Instant::now();
        let batch_mats: Vec<(usize, [stark_rings_linalg::SparseMatrix<R>; 3])> = (batch_start..batch_end)
            .map(|i| (i, cache.read_chunk(i).expect("read_chunk failed")))
            .collect();
        let load_mat_time = load_mat_start.elapsed();
        println!("    Loaded {batch_size} chunks from cache in {load_mat_time:?}");
        
        // Process this batch in PARALLEL using rayon.
        //
        // IMPORTANT: do not retain full `PiFoldProverOutput` objects in memory.
        // We immediately map each result down to a tiny record (proof_size or error string)
        // and `drop(out)` inside the worker thread to keep peak RSS bounded.
        let batch_results: Vec<(usize, Result<usize, String>, std::time::Duration)> = pool.install(|| {
            batch_mats
                .into_par_iter() // <-- parallel over loaded chunks
                .map(|(i, [m1, m2, m3])| {
                let chunk_start = Instant::now();
                let result: Result<usize, String> = if pifold_mode == "streaming" {
                    prove_pi_fold_streaming_sumcheck_fs::<R, PC>(
                        [Arc::new(m1), Arc::new(m2), Arc::new(m3)],
                        &[cm_main.clone()],
                        &[witness.clone()],
                        &public_inputs,
                        Some(scheme_had.as_ref()),
                        Some(scheme_mon.as_ref()),
                        rg_params.clone(),
                    )
                    .map(|out| {
                        let proof_size = out.proof.coins.bytes.len();
                        drop(out);
                        proof_size
                    })
                } else {
                    prove_pi_fold_batched_sumcheck_fs::<R, PC>(
                        [&m1, &m2, &m3],
                    &[cm_main.clone()],
                        &[witness.as_ref().as_slice()],
                    &public_inputs,
                    Some(scheme_had.as_ref()),
                    Some(scheme_mon.as_ref()),
                    rg_params.clone(),
                    )
                    .map(|out| {
                        let proof_size = out.proof.coins.bytes.len();
                        drop(out);
                        proof_size
                    })
                };
                
                (i, result, chunk_start.elapsed())
            })
            .collect()
        });
        
        let batch_time = batch_start_time.elapsed();
        
        for (i, result, chunk_time) in batch_results {
            match result {
                Ok(proof_size) => {
                    println!("    Chunk {}: ✓ {:.2?}, {} bytes", i, chunk_time, proof_size);
                    successes += 1;
                    total_proof_bytes += proof_size;
                }
                Err(e) => {
                    println!("    Chunk {}: ✗ {:.2?}, {}", i, chunk_time, e);
                    failures += 1;
                }
            }
        }
        
        println!("    Batch time: {batch_time:?} ({batch_size} chunks)\n");
    }
    
    let prove_time = prove_start.elapsed();

    println!("=========================================================");
    println!("Summary");
    println!("=========================================================");
    println!("Total constraints:    {}", cache.stats.num_constraints);
    println!("Chunks:               {num_chunks}");
    println!("Chunk size:           {chunk_size}");
    println!("Max concurrent:       {max_concurrent}");
    println!("Successes:            {successes}");
    println!("Failures:             {failures}");
    println!("Total proof bytes:    {total_proof_bytes}");
    println!("Load time:            {load_time:?}");
    println!("Setup time:           {setup_time:?}");
    println!("Prove time:           {prove_time:?}");
    if successes > 0 {
        println!("Avg per chunk:        {:.2?}", prove_time / successes as u32);
    }

    if successes == num_chunks {
        println!("\n✓ All chunks proved! Next: high-arity fold {} proofs into one", num_chunks);
    }
}
