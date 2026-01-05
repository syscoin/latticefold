//! Chunked parallel proving for SP1 shrink verifier R1CS.
//!
//! Usage:
//!   SP1_R1CS=/path/to/shrink.r1cs CHUNK_SIZE=1048576 cargo run -p symphony --example symphony_sp1_fold_chunked --release
//!
//! Environment variables:
//!   SP1_R1CS    - Path to the R1CS file
//!   CHUNK_SIZE  - Constraints per chunk (default: 1048576 = 2^20)
//!
//! This splits the 67M constraint R1CS into ~64 chunks of 1M each,
//! proves each chunk in parallel across all cores, then folds.

use std::sync::Arc;
use std::time::Instant;

use rayon::prelude::*;

use cyclotomic_rings::rings::FrogPoseidonConfig as PC;
use latticefold::commitment::AjtaiCommitmentScheme;
use stark_rings::cyclotomic_ring::models::frog_ring::RqPoly as R;
use stark_rings::{PolyRing, Ring};
use stark_rings_linalg::{Matrix, SparseMatrix};
use symphony::sp1_r1cs_loader::FieldFromU64;
use symphony::symphony_sp1_r1cs::load_sp1_r1cs_as_symphony;
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

/// Next power of two
fn next_pow2(n: usize) -> usize {
    if n == 0 { return 1; }
    1 << (usize::BITS - (n - 1).leading_zeros())
}

/// Split a sparse matrix into row chunks, padding each to power of 2.
fn chunk_matrix<R: Ring + Clone + Send + Sync>(
    matrix: &SparseMatrix<R>,
    chunk_size: usize,
) -> Vec<SparseMatrix<R>> {
    let nrows = matrix.nrows;
    let ncols = matrix.ncols;
    let num_chunks = (nrows + chunk_size - 1) / chunk_size;
    
    (0..num_chunks).map(|i| {
        let start = i * chunk_size;
        let end = std::cmp::min(start + chunk_size, nrows);
        let actual_rows = end - start;
        let padded_rows = next_pow2(actual_rows);
        
        let mut coeffs: Vec<Vec<(R, usize)>> = matrix.coeffs[start..end].to_vec();
        
        // Pad with empty rows to power of 2
        coeffs.resize(padded_rows, Vec::new());
        
        SparseMatrix {
            nrows: padded_rows,
            ncols,
            coeffs,
        }
    }).collect()
}

fn main() {
    let r1cs_path = std::env::var("SP1_R1CS").expect(
        "Set SP1_R1CS=/path/to/shrink_verifier.r1cs"
    );
    
    let chunk_size: usize = std::env::var("CHUNK_SIZE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1 << 20); // Default: 2^20 = 1M constraints per chunk
    
    let num_threads = rayon::current_num_threads();
    
    println!("=========================================================");
    println!("Symphony SP1 Chunked Parallel Folding");
    println!("=========================================================\n");
    println!("Configuration:");
    println!("  Chunk size: {} constraints (2^{})", chunk_size, chunk_size.trailing_zeros());
    println!("  Threads: {num_threads}\n");

    // Step 1: Load R1CS (no padding - we'll chunk and pad per-chunk)
    println!("Step 1: Loading R1CS...");
    let load_start = Instant::now();
    
    // Load without padding (pass pad_to = Some(0) or we need a raw load)
    // Actually, let's load the raw R1CS and chunk it ourselves
    let ([m1_full, m2_full, m3_full], stats): ([SparseMatrix<R>; 3], _) = 
        load_sp1_r1cs_as_symphony::<R, BabyBear>(&r1cs_path, None)
            .expect("Failed to load R1CS");
    
    let load_time = load_start.elapsed();
    println!("  Loaded: {} constraints × {} vars", stats.num_constraints, stats.num_vars);
    println!("  Load time: {load_time:?}\n");

    // Step 2: Chunk matrices
    println!("Step 2: Chunking matrices...");
    let chunk_start = Instant::now();
    
    let chunks_a = chunk_matrix(&m1_full, chunk_size);
    let chunks_b = chunk_matrix(&m2_full, chunk_size);
    let chunks_c = chunk_matrix(&m3_full, chunk_size);
    
    let num_chunks = chunks_a.len();
    let chunk_time = chunk_start.elapsed();
    
    println!("  Created {} chunks", num_chunks);
    println!("  Chunk dimensions: {} rows × {} cols (padded per chunk)", 
        chunks_a[0].nrows, chunks_a[0].ncols);
    println!("  Chunking time: {chunk_time:?}\n");

    // Step 3: Create witness (same for all chunks since they share columns)
    println!("Step 3: Creating witness...");
    let n_cols = m1_full.ncols;
    let mut witness: Vec<R> = vec![R::ZERO; n_cols];
    witness[0] = R::ONE;
    let witness = Arc::new(witness);
    println!("  Witness length: {n_cols}\n");

    // Step 4: Setup Symphony parameters (shared across chunks)
    println!("Step 4: Setting up Symphony parameters...");
    let kappa = 8;
    let k_g = 3;
    let rg_params = RPParams {
        l_h: 64,
        lambda_pj: 1,
        k_g,
        d_prime: (R::dimension() as u128) - 2,
    };
    
    // Commitment for witness (shared)
    let setup_start = Instant::now();
    let a_main = Matrix::<R>::rand(&mut ark_std::test_rng(), kappa, n_cols);
    let scheme_main = Arc::new(AjtaiCommitmentScheme::<R>::new(a_main));
    let cm_main = scheme_main.commit(&witness).unwrap().as_ref().to_vec();
    let setup_time = setup_start.elapsed();
    println!("  Main commitment setup: {setup_time:?}\n");

    // Auxiliary schemes for each chunk
    let a_had = Matrix::<R>::rand(&mut ark_std::test_rng(), kappa, 3 * R::dimension());
    let a_mon = Matrix::<R>::rand(&mut ark_std::test_rng(), kappa, rg_params.k_g);
    let scheme_had = Arc::new(AjtaiCommitmentScheme::<R>::new(a_had));
    let scheme_mon = Arc::new(AjtaiCommitmentScheme::<R>::new(a_mon));

    let public_inputs: Vec<<R as PolyRing>::BaseRing> = vec![
        <R as PolyRing>::BaseRing::from(1u128),
    ];
    let public_inputs = Arc::new(public_inputs);

    // Step 5: Parallel prove all chunks
    println!("Step 5: Proving {} chunks in parallel...", num_chunks);
    let prove_start = Instant::now();
    
    let chunk_results: Vec<_> = (0..num_chunks)
        .into_par_iter()
        .map(|i| {
            let chunk_prove_start = Instant::now();
            
            let m1 = &chunks_a[i];
            let m2 = &chunks_b[i];
            let m3 = &chunks_c[i];
            
            // Note: witness is full-length, chunk matrices have full ncols
            let result = prove_pi_fold_batched_sumcheck_fs::<R, PC>(
                [m1, m2, m3],
                &[cm_main.clone()],
                &[(*witness).clone()],
                &public_inputs,
                Some(scheme_had.as_ref()),
                Some(scheme_mon.as_ref()),
                rg_params.clone(),
            );
            
            let chunk_time = chunk_prove_start.elapsed();
            (i, result, chunk_time)
        })
        .collect();
    
    let prove_time = prove_start.elapsed();
    
    // Report results
    let mut successes = 0;
    let mut failures = 0;
    
    println!("\nChunk results:");
    for (i, result, time) in &chunk_results {
        match result {
            Ok(out) => {
                println!("  Chunk {}: ✓ {:.2?}, proof {} bytes", 
                    i, time, out.proof.coins.bytes.len());
                successes += 1;
            }
            Err(e) => {
                println!("  Chunk {}: ✗ {:.2?}, error: {}", i, time, e);
                failures += 1;
            }
        }
    }
    
    println!("\n=========================================================");
    println!("Summary");
    println!("=========================================================");
    println!("Total constraints:    {}", stats.num_constraints);
    println!("Chunks:               {num_chunks}");
    println!("Chunk size:           {chunk_size}");
    println!("Successes:            {successes}");
    println!("Failures:             {failures}");
    println!("Load time:            {load_time:?}");
    println!("Chunk time:           {chunk_time:?}");
    println!("Total prove time:     {prove_time:?}");
    println!("Avg per chunk:        {:.2?}", prove_time / num_chunks as u32);
    println!("Threads used:         {num_threads}");

    if successes == num_chunks {
        println!("\n✓ All chunks proved successfully!");
        println!("Next step: High-arity fold {} chunk proofs into one", num_chunks);
    }
}
