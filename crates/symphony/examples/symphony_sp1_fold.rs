//! Load SP1 shrink verifier R1CS and fold with Symphony.
//!
//! Usage:
//!   SP1_R1CS=/path/to/shrink_verifier.r1cs cargo run -p symphony --example symphony_sp1_fold --release
//!
//! This example:
//! 1. Loads the pre-compiled R1CS from SP1's shrink verifier
//! 2. Converts to Symphony's sparse matrix format  
//! 3. Creates a dummy witness for benchmarking (real witness would come from SP1 proof)
//! 4. Runs Symphony folding

use std::time::Instant;

use cyclotomic_rings::rings::FrogPoseidonConfig as PC;
use latticefold::commitment::AjtaiCommitmentScheme;
use stark_rings::cyclotomic_ring::models::frog_ring::RqPoly as R;
use stark_rings::{PolyRing, Ring};
use stark_rings_linalg::{Matrix, SparseMatrix};
use symphony::sp1_r1cs_loader::FieldFromU64;
use symphony::symphony_sp1_r1cs::load_sp1_r1cs_as_symphony;
use symphony::rp_rgchk::RPParams;
use symphony::symphony_pifold_batched::prove_pi_fold_batched_sumcheck_fs;
use symphony::symphony_we_relation::check_r_cp_poseidon_fs;
use symphony::symphony_open::MultiAjtaiOpenVerifier;

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
    let r1cs_path = std::env::var("SP1_R1CS").expect(
        "Set SP1_R1CS=/path/to/shrink_verifier.r1cs\n\
         Generate with: OUT_R1CS=shrink.r1cs cargo run -p sp1-prover --bin dump_shrink_verify_constraints --release"
    );

    println!("=========================================================");
    println!("Symphony SP1 Shrink Verifier Folding");
    println!("=========================================================\n");

    // Step 1: Load full R1CS and convert to Symphony format
    println!("Step 1: Loading R1CS and converting to Symphony matrices...");
    let load_start = Instant::now();
    
    let (sp1_r1cs, [m1, m2, m3], stats): (_, [SparseMatrix<R>; 3], _) = 
        load_sp1_r1cs_as_symphony::<R, BabyBear>(&r1cs_path, None)
            .expect("Failed to load R1CS");
    
    let load_time = load_start.elapsed();
    
    println!("  File: {r1cs_path}");
    println!("  Variables:     {:>12}", stats.num_vars);
    println!("  Constraints:   {:>12}", stats.num_constraints);
    println!("  Public inputs: {:>12}", stats.num_public);
    println!("  Non-zeros:     {:>12}", stats.total_nonzeros);
    println!("  Digest: {:02x}{:02x}{:02x}{:02x}...{:02x}{:02x}{:02x}{:02x}",
        stats.digest[0], stats.digest[1], stats.digest[2], stats.digest[3],
        stats.digest[28], stats.digest[29], stats.digest[30], stats.digest[31]);
    println!("  Load time: {load_time:?}");
    println!("  Matrix dimensions: {} rows × {} cols", m1.nrows, m1.ncols);
    
    let nnz_a: usize = m1.coeffs.iter().map(|r| r.len()).sum();
    let nnz_b: usize = m2.coeffs.iter().map(|r| r.len()).sum();
    let nnz_c: usize = m3.coeffs.iter().map(|r| r.len()).sum();
    println!("  Non-zeros: A={nnz_a}, B={nnz_b}, C={nnz_c}\n");

    // Step 2: Create dummy witness for benchmarking
    // Real witness would come from SP1 proof execution
    println!("Step 3: Creating dummy witness...");
    let n = sp1_r1cs.num_vars;
    
    // Dummy witness: w[0] = 1, rest = 0
    // This won't satisfy the R1CS but lets us benchmark the folding mechanics
    let mut witness: Vec<R> = vec![R::ZERO; n];
    witness[0] = R::ONE;
    println!("  Witness length: {n}");
    println!("  (Using dummy witness - real witness would come from SP1 proof)\n");

    // Step 3: Setup commitment scheme
    println!("Step 3: Setting up Ajtai commitment...");
    let kappa = 8; // Commitment security parameter
    let setup_start = Instant::now();
    let a = Matrix::<R>::rand(&mut ark_std::test_rng(), kappa, n);
    let scheme = AjtaiCommitmentScheme::<R>::new(a);
    let cm = scheme.commit(&witness).unwrap().as_ref().to_vec();
    let setup_time = setup_start.elapsed();
    println!("  Commitment setup time: {setup_time:?}");
    println!("  Commitment size: {} elements\n", cm.len());

    // Step 4: Configure Symphony parameters
    println!("Step 4: Configuring Symphony Π_rg parameters...");
    let k_g = 3; // Digit decomposition parameter
    let rg_params = RPParams {
        l_h: 64,
        lambda_pj: 1, // Small for testing
        k_g,
        d_prime: (R::dimension() as u128) - 2,
    };
    println!("  k_g (digits): {}", rg_params.k_g);
    println!("  l_h: {}", rg_params.l_h);
    println!("  d': {}", rg_params.d_prime);
    println!("  Ring dimension d: {}\n", R::dimension());

    // Step 5: Auxiliary commitment schemes
    println!("Step 5: Setting up auxiliary commitment schemes...");
    let a_had = Matrix::<R>::rand(&mut ark_std::test_rng(), kappa, 3 * R::dimension());
    let a_mon = Matrix::<R>::rand(&mut ark_std::test_rng(), kappa, rg_params.k_g);
    let scheme_had = AjtaiCommitmentScheme::<R>::new(a_had);
    let scheme_mon = AjtaiCommitmentScheme::<R>::new(a_mon);
    println!("  Hadamard scheme: {} × {}", kappa, 3 * R::dimension());
    println!("  Monomial scheme: {} × {}\n", kappa, rg_params.k_g);

    // Public inputs (statement binding)
    let public_inputs: Vec<<R as PolyRing>::BaseRing> = vec![
        <R as PolyRing>::BaseRing::from(1u128), // Placeholder
    ];

    // Step 6: Run Symphony Π_fold proving
    println!("Step 6: Running Symphony Π_fold proving...");
    println!("  (This proves the folded Hadamard relation with range proofs)\n");
    
    let prove_start = Instant::now();
    let result = prove_pi_fold_batched_sumcheck_fs::<R, PC>(
        [&m1, &m2, &m3],
        &[cm.clone()],
        &[witness.clone()],
        &public_inputs,
        Some(&scheme_had),
        Some(&scheme_mon),
        rg_params.clone(),
    );
    let prove_time = prove_start.elapsed();

    match result {
        Ok(out) => {
            println!("  ✓ Proving succeeded!");
            println!("  Prove time: {prove_time:?}");
            println!("  Proof size: {} bytes\n", out.proof.coins.bytes.len());

            // Step 7: Verify
            println!("Step 7: Running Symphony R_cp verification...");
            let open = MultiAjtaiOpenVerifier::<R>::new()
                .with_scheme("cfs_had_u", scheme_had)
                .with_scheme("cfs_mon_b", scheme_mon);

            let verify_start = Instant::now();
            let verify_result = check_r_cp_poseidon_fs::<R, PC>(
                [&m1, &m2, &m3],
                &[cm],
                &out.proof,
                &open,
                &out.cfs_had_u,
                &out.cfs_mon_b,
                &out.aux,
                &public_inputs,
            );
            let verify_time = verify_start.elapsed();

            match verify_result {
                Ok(_) => {
                    println!("  ✓ Verification succeeded!");
                    println!("  Verify time: {verify_time:?}");
                }
                Err(e) => {
                    println!("  ✗ Verification failed: {e}");
                    println!("  (Expected with dummy witness)");
                }
            }
        }
        Err(e) => {
            println!("  ✗ Proving failed: {e}");
            println!("  (May need real witness or parameter tuning)");
        }
    }

    println!("\n=========================================================");
    println!("Summary");
    println!("=========================================================");
    println!("R1CS constraints: {}", stats.num_constraints);
    println!("R1CS variables:   {}", stats.num_vars);
    println!("Load time:        {load_time:?}");
    println!("Setup time:       {setup_time:?}");
    println!("Prove time:       {prove_time:?}");
}
