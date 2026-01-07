//! Chunked streaming proving for SP1 shrink verifier R1CS.
//!
//! This example loads an SP1 R1CS and proves all chunks together using the
//! canonical streaming hetero-M Π_fold prover (`prove_pi_fold_poseidon_fs`).
//!
//! Usage:
//!   SP1_R1CS=/path/to/shrink.r1cs cargo run -p symphony --example symphony_sp1_fold_chunked --release
//!
//! Environment variables:
//!   SP1_R1CS       - Path to the R1CS file
//!   CHUNK_SIZE     - Constraints per chunk (default: 1048576 = 2^20 = 1M)
//!   VERIFY_PROOF   - Set to "1" to verify the proof after proving

use std::sync::Arc;
use std::time::Instant;

use ark_ff::PrimeField;
use cyclotomic_rings::rings::FrogPoseidonConfig as PC;
use latticefold::commitment::AjtaiCommitmentScheme;
use rayon::ThreadPoolBuilder;
use stark_rings::cyclotomic_ring::models::frog_ring::RqPoly as R;
use stark_rings::PolyRing;
use stark_rings::Ring;
use symphony::sp1_r1cs_loader::FieldFromU64;
use symphony::symphony_sp1_r1cs::open_sp1_r1cs_chunk_cache;
use symphony::rp_rgchk::RPParams;
use symphony::symphony_pifold_streaming::{prove_pi_fold_poseidon_fs, PiFoldStreamingConfig};
use symphony::symphony_open::MultiAjtaiOpenVerifier;
use symphony::symphony_we_relation::{
    check_r_we_poseidon_fs_hetero_m_with_metrics_result, TrivialRo,
};

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
    
    println!("=========================================================");
    println!("Symphony SP1 Streaming Proving");
    println!("=========================================================\n");
    println!("Configuration:");
    println!("  Chunk size: {} (2^{})", chunk_size, chunk_size.trailing_zeros());
    let prove_threads: usize = std::env::var("PROVE_THREADS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or_else(|| std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1));
    println!("  Prove threads: {prove_threads}");

    // Π_fold streaming configuration (library does not read env vars).
    let mut pifold_cfg = PiFoldStreamingConfig::default();
    pifold_cfg.profile = std::env::var("SYMPHONY_PROFILE").ok().as_deref() == Some("1");

    // Step 1: Load and chunk R1CS (uses cache)
    println!("\nStep 1: Loading R1CS (chunked)...");
    let load_start = Instant::now();
    
    // Π_rg parameters that affect how we pad/load the witness.
    let l_h: usize = std::env::var("L_H")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(256);
    let pad_cols_to_multiple_of = l_h;
    let cache = open_sp1_r1cs_chunk_cache::<R, BabyBear>(&r1cs_path, chunk_size, pad_cols_to_multiple_of)
        .expect("Failed to open/build chunk cache");
    
    let load_time = load_start.elapsed();
    let num_chunks = cache.num_chunks;
    
    println!("  Constraints: {}", cache.stats.num_constraints);
    println!("  Variables:   {}", cache.stats.num_vars);
    println!("  Chunks:      {}", num_chunks);
    println!("  Load time:   {load_time:?}");

    // Step 2: Create witness (same for all chunks)
    println!("\nStep 2: Creating witness...");
    let ncols = cache.ncols;
    let mut witness: Vec<R> = vec![R::ZERO; ncols];
    witness[0] = R::ONE;
    let witness = Arc::new(witness);
    if ncols.is_power_of_two() {
        println!("  Witness length: {ncols} (2^{})", ncols.trailing_zeros());
    } else {
        println!("  Witness length: {ncols}");
    }

    // Step 3: Setup Symphony parameters
    println!("\nStep 3: Setting up Symphony parameters...");
    let kappa = 8;
    let k_g = 3;
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
    let blocks = ncols / rg_params.l_h;
    let m_j = blocks * rg_params.lambda_pj;
    println!(
        "  Π_rg params: l_h={}, lambda_pj={}, k_g={}, d'={}  (m_J={})",
        rg_params.l_h, rg_params.lambda_pj, rg_params.k_g, rg_params.d_prime, m_j
    );
    
    // Main commitment (shared across chunks)
    let setup_start = Instant::now();
    const MASTER_SEED: [u8; 32] = *b"SYMPHONY_AJTAI_SEED_V1_000000000";
    let scheme_main = Arc::new(AjtaiCommitmentScheme::<R>::seeded(b"cm_f", MASTER_SEED, kappa, ncols));
    let cm_main = scheme_main
        .commit_const_coeff_fast(&witness)
        .unwrap()
        .as_ref()
        .to_vec();
    let setup_time = setup_start.elapsed();
    println!("  Commitment setup: {setup_time:?}");

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

    // Public statement binding
    let public_inputs: Vec<<R as PolyRing>::BaseRing> = vec![
        <R as PolyRing>::BaseRing::from(1u128),
    ];

    // Step 4: Load all chunk matrices
    println!("\nStep 4: Loading all chunk matrices...");
    let t_load_all = Instant::now();
    let mut all_mats: Vec<[Arc<stark_rings_linalg::SparseMatrix<R>>; 3]> =
        Vec::with_capacity(num_chunks);
    for i in 0..num_chunks {
        let [m1, m2, m3] = cache.read_chunk(i).expect("read_chunk failed");
        all_mats.push([Arc::new(m1), Arc::new(m2), Arc::new(m3)]);
    }
    println!("  Loaded {} chunks in {:?}", num_chunks, t_load_all.elapsed());

    // Step 5: Prove all chunks with streaming hetero-M prover
    println!("\nStep 5: Proving (streaming hetero-M)...");
    let cms_all: Vec<Vec<R>> = vec![cm_main.clone(); num_chunks];
    let witnesses_all: Vec<Arc<Vec<R>>> = vec![witness.clone(); num_chunks];

    let pool = ThreadPoolBuilder::new()
        .num_threads(prove_threads)
        .build()
        .expect("failed to build local rayon pool");
    let prove_start = Instant::now();
    let out = pool.install(|| {
        prove_pi_fold_poseidon_fs::<R, PC>(
            all_mats.as_slice(),
            &cms_all,
            &witnesses_all,
            &public_inputs,
            Some(scheme_had.as_ref()),
            Some(scheme_mon.as_ref()),
            rg_params.clone(),
            &pifold_cfg,
        )
    })
    .expect("prove failed");
    let prove_time = prove_start.elapsed();
    let proof_bytes = out.proof.coins.bytes.len();
    println!("  ✓ Prove time: {prove_time:?}");
    println!("  Proof bytes: {proof_bytes}");

    // Step 6: Optional verification
    let verify_proof: bool = std::env::var("VERIFY_PROOF")
        .ok()
        .as_deref()
        .map(|s| s == "1" || s.eq_ignore_ascii_case("true"))
        .unwrap_or(false);
    if verify_proof {
        println!("\nStep 6: Verifying proof...");
        let t_v = Instant::now();
        let ms_refs: Vec<[&stark_rings_linalg::SparseMatrix<R>; 3]> = all_mats
            .iter()
            .map(|m| [&*m[0], &*m[1], &*m[2]])
            .collect();

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

        // Print transcript metrics
        let rate: u64 = 20;
        let absorbed = metrics.absorbed_elems;
        let squeezed = metrics.squeezed_field_elems;
        let squeezed_bytes = metrics.squeezed_bytes;
        let est_perms_absorb = (absorbed + rate - 1) / rate;
        let est_perms_squeeze = (squeezed + rate - 1) / rate;
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
            "  Transcript metrics: absorbed={}, squeezed_field={}, squeezed_bytes={}",
            absorbed, squeezed, squeezed_bytes
        );
        eprintln!(
            "  Est perms: absorb={} + squeeze_field={} + bytes={} => {}",
            est_perms_absorb, est_perms_squeeze, est_perms_bytes,
            est_perms_absorb + est_perms_squeeze + est_perms_bytes
        );
        res.expect("R_WE check failed");
        println!("  ✓ Verified in {:?}", t_v.elapsed());
    }

    // Summary
    println!("\n=========================================================");
    println!("Summary");
    println!("=========================================================");
    println!("Constraints:      {}", cache.stats.num_constraints);
    println!("Variables:        {}", cache.stats.num_vars);
    println!("Chunks:           {num_chunks}");
    println!("Chunk size:       {chunk_size}");
    println!("Load time:        {load_time:?}");
    println!("Setup time:       {setup_time:?}");
    println!("Prove time:       {prove_time:?}");
    println!("Proof bytes:      {proof_bytes}");
    println!("\n✓ Successfully proved {} chunks as one folded proof", num_chunks);
}
