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

use cyclotomic_rings::rings::FrogPoseidonConfig as PC;
use latticefold::commitment::AjtaiCommitmentScheme;
use stark_rings::cyclotomic_ring::models::frog_ring::RqPoly as R;
use stark_rings::{PolyRing, Ring};
use stark_rings_linalg::Matrix;
use symphony::sp1_r1cs_loader::FieldFromU64;
use symphony::symphony_sp1_r1cs::load_sp1_r1cs_chunked_cached;
use symphony::rp_rgchk::RPParams;
use symphony::symphony_pifold_batched::prove_pi_fold_batched_sumcheck_fs;

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
        .unwrap_or(4); // Default: 4 concurrent chunks
    
    println!("=========================================================");
    println!("Symphony SP1 Chunked Proving (Memory-Controlled)");
    println!("=========================================================\n");
    println!("Configuration:");
    println!("  Chunk size:     {} (2^{})", chunk_size, chunk_size.trailing_zeros());
    println!("  Max concurrent: {} chunks at once", max_concurrent);
    println!("  Total threads:  {}\n", rayon::current_num_threads());

    // Step 1: Load and chunk R1CS (uses cache)
    println!("Step 1: Loading R1CS (chunked)...");
    let load_start = Instant::now();
    
    let chunked = load_sp1_r1cs_chunked_cached::<R, BabyBear>(&r1cs_path, chunk_size)
        .expect("Failed to load R1CS");
    
    let load_time = load_start.elapsed();
    let num_chunks = chunked.chunks.len();
    
    println!("  Constraints: {}", chunked.stats.num_constraints);
    println!("  Variables:   {}", chunked.stats.num_vars);
    println!("  Chunks:      {}", num_chunks);
    println!("  Load time:   {load_time:?}\n");

    // Step 2: Create witness (same for all chunks)
    println!("Step 2: Creating witness...");
    let ncols = chunked.ncols;
    let mut witness: Vec<R> = vec![R::ZERO; ncols];
    witness[0] = R::ONE;
    let witness = Arc::new(witness);
    println!("  Witness length: {ncols} (2^{})\n", ncols.trailing_zeros());

    // Step 3: Setup Symphony parameters
    println!("Step 3: Setting up Symphony parameters...");
    let kappa = 8;
    let k_g = 3;
    let rg_params = RPParams {
        l_h: 64,
        lambda_pj: 1,
        k_g,
        d_prime: (R::dimension() as u128) - 2,
    };
    
    // Main commitment (shared across chunks)
    let setup_start = Instant::now();
    let a_main = Matrix::<R>::rand(&mut ark_std::test_rng(), kappa, ncols);
    let scheme_main = Arc::new(AjtaiCommitmentScheme::<R>::new(a_main));
    let cm_main = scheme_main.commit(&witness).unwrap().as_ref().to_vec();
    let setup_time = setup_start.elapsed();
    println!("  Commitment setup: {setup_time:?}\n");

    // Auxiliary schemes
    let a_had = Matrix::<R>::rand(&mut ark_std::test_rng(), kappa, 3 * R::dimension());
    let a_mon = Matrix::<R>::rand(&mut ark_std::test_rng(), kappa, rg_params.k_g);
    let scheme_had = Arc::new(AjtaiCommitmentScheme::<R>::new(a_had));
    let scheme_mon = Arc::new(AjtaiCommitmentScheme::<R>::new(a_mon));

    let public_inputs: Vec<<R as PolyRing>::BaseRing> = vec![
        <R as PolyRing>::BaseRing::from(1u128),
    ];

    // Step 4: Prove chunks with limited concurrency
    println!("Step 4: Proving {} chunks ({} at a time)...\n", num_chunks, max_concurrent);
    let prove_start = Instant::now();
    
    let mut successes = 0;
    let mut failures = 0;
    let mut total_proof_bytes = 0usize;
    
    // Process in batches of max_concurrent
    for batch_start in (0..num_chunks).step_by(max_concurrent) {
        let batch_end = std::cmp::min(batch_start + max_concurrent, num_chunks);
        let batch_size = batch_end - batch_start;
        
        println!("  Batch {}-{} of {}...", batch_start, batch_end - 1, num_chunks);
        let batch_start_time = Instant::now();
        
        // Process this batch in parallel using rayon's thread pool
        let batch_results: Vec<_> = (batch_start..batch_end)
            .into_iter()
            .map(|i| {
                let chunk_start = Instant::now();
                let [m1, m2, m3] = &chunked.chunks[i];
                
                let result = prove_pi_fold_batched_sumcheck_fs::<R, PC>(
                    [m1, m2, m3],
                    &[cm_main.clone()],
                    &[(*witness).clone()],
                    &public_inputs,
                    Some(scheme_had.as_ref()),
                    Some(scheme_mon.as_ref()),
                    rg_params.clone(),
                );
                
                (i, result, chunk_start.elapsed())
            })
            .collect();
        
        let batch_time = batch_start_time.elapsed();
        
        for (i, result, chunk_time) in batch_results {
            match result {
                Ok(out) => {
                    let proof_size = out.proof.coins.bytes.len();
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
    println!("Total constraints:    {}", chunked.stats.num_constraints);
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
