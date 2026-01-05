//! Custom circuit example for PVUGC GM-1B measurement
//!
//! This example creates custom R1CS circuits of varying sizes,
//! converts them to CCS, folds them, and measures:
//! - Folding time
//! - Proof size
//! - Verification time
//!
//! Key observation: Verification time should be roughly constant
//! regardless of original circuit size (that's the point of folding).

use std::time::Instant;

use ark_serialize::{CanonicalSerialize, Compress};
use ark_std::{vec::Vec, UniformRand};
use cyclotomic_rings::{
    challenge_set::LatticefoldChallengeSet,
    rings::{GoldilocksChallengeSet, GoldilocksRingNTT},
};
use latticefold::{
    arith::{
        r1cs::{Constraint, ConstraintSystem, LinearCombination, R1CS},
        Arith, Witness, CCCS, CCS, LCCCS,
    },
    commitment::AjtaiCommitmentScheme,
    decomposition_parameters::DecompositionParams,
    nifs::{
        linearization::{LFLinearizationProver, LinearizationProver},
        NIFSProver, NIFSVerifier,
    },
    transcript::poseidon::PoseidonTranscript,
};

// Decomposition parameters
#[derive(Clone)]
pub struct CustomDP {}

impl DecompositionParams for CustomDP {
    const B: u128 = 1 << 15; // 32768
    const L: usize = 5;
    const B_SMALL: usize = 2;
    const K: usize = 15;
}

type RqNTT = GoldilocksRingNTT;
type CS = GoldilocksChallengeSet;
type T = PoseidonTranscript<RqNTT, CS>;
type DP = CustomDP;

/// Create a chain of squarings: x₀ → x₁ = x₀² → x₂ = x₁² → ... → xₙ
/// Returns (R1CS, z_vector, public_input_x, witness_w)
/// 
/// R1CS format: z = [x₀, 1, x₁, ..., xₙ] where index 1 is the constant "1"
/// CCS format: get_z_vector() adds "1" automatically, so w_ccs = [x₁, ..., xₙ]
fn create_squaring_chain(num_squarings: usize) -> (R1CS<RqNTT>, Vec<RqNTT>, Vec<RqNTT>, Vec<RqNTT>) {
    let mut cs = ConstraintSystem::<RqNTT>::new();
    
    // Variables layout in R1CS: [x₀] (public) + [1, x₁, x₂, ..., xₙ] (auxiliary)
    // Index 0: x₀ (public input)
    // Index 1: 1 (constant, auxiliary) - NOTE: CCS adds this automatically
    // Index 2..n+1: x₁, x₂, ..., xₙ (auxiliary)
    cs.ninputs = 1;               // Just x₀
    cs.nauxs = num_squarings + 1; // "1" + n witness values
    
    // Constraints: xᵢ * xᵢ = xᵢ₊₁
    for i in 0..num_squarings {
        let var_i = if i == 0 { 0 } else { i + 1 }; // x₀ at 0, xᵢ at i+1 for i>0
        let var_next = i + 2; // x₁ at 2, x₂ at 3, etc.
        
        let a = LinearCombination::single_term(1u64, var_i);
        let b = LinearCombination::single_term(1u64, var_i);
        let c = LinearCombination::single_term(1u64, var_next);
        cs.add_constraint(Constraint::new(a, b, c));
    }
    
    let r1cs = R1CS::from_constraint_system(cs);
    
    // Compute witness values
    let x0 = RqNTT::from(2u64);
    let mut squarings = Vec::with_capacity(num_squarings);
    let mut val = x0;
    for _ in 0..num_squarings {
        val = val * val;
        squarings.push(val);
    }
    
    // R1CS z vector: [x₀, 1, x₁, ..., xₙ]
    let mut z = vec![x0, RqNTT::from(1u64)];
    z.extend(&squarings);
    
    // For CCS:
    // - x_ccs = [x₀] (public input)
    // - w_ccs = [x₁, ..., xₙ] (witness WITHOUT "1" - get_z_vector adds it)
    let x_ccs = vec![x0];
    let w_ccs = squarings;
    
    (r1cs, z, x_ccs, w_ccs)
}

fn main() {
    println!("=== PVUGC GM-1B Measurement: Custom Circuit Folding ===\n");
    
    // Test various sizes - adjust as needed for GM-1B measurement
    for num_constraints in [1024, 4096, 16384, 65536] {
        println!("--- {} constraints ({:.1}K) ---", num_constraints, num_constraints as f64 / 1000.0);
        test_folding(num_constraints);
        println!();
    }
    
    println!("=== Key Observation ===");
    println!("Verification time should remain roughly constant as circuit size increases.");
    println!("This is because folding compresses the verification to constant size.");
}

fn test_folding(num_constraints: usize) {
    let mut rng = ark_std::test_rng();
    
    // Create circuit
    let (r1cs, z, x, w) = create_squaring_chain(num_constraints);
    
    // Verify R1CS
    if r1cs.check_relation(&z).is_err() {
        println!("  R1CS check FAILED");
        return;
    }
    
    // Parameters - n must equal decomposed witness length for commitment
    let wit_len = w.len();
    let decomposed_len = wit_len * DP::L;
    let n = decomposed_len; // n = wit_len * L
    let kappa = 4;
    
    println!("  Circuit: {} constraints, wit_len={}, decomposed={}, n={}", 
             num_constraints, wit_len, decomposed_len, n);
    
    // Convert to CCS (padded for folding)
    let ccs: CCS<RqNTT> = CCS::from_r1cs_padded(r1cs, n, DP::L);
    
    // Note: CCS check_relation needs properly padded z vector
    // The NIFS APIs handle this internally via get_z_vector
    println!("  CCS matrices: {} rows, {} cols", ccs.m, ccs.n);
    
    // Create commitment scheme
    let scheme: AjtaiCommitmentScheme<RqNTT> = AjtaiCommitmentScheme::rand(kappa, n, &mut rng);
    
    // Create witness
    let wit: Witness<RqNTT> = Witness::from_w_ccs::<DP>(w.clone());
    
    // Commit
    let cm = match wit.commit::<DP>(&scheme) {
        Ok(c) => c,
        Err(e) => {
            println!("  Commit failed: {:?}", e);
            return;
        }
    };
    
    // Create CCCS
    let cm_i = CCCS { cm: cm.clone(), x_ccs: x.clone() };
    
    // For NIFS we need an accumulator (LCCCS)
    // First, linearize to get an initial accumulator
    let mut transcript = PoseidonTranscript::<RqNTT, CS>::default();
    
    // Create a dummy accumulator witness
    let wit_acc: Witness<RqNTT> = Witness::from_w_ccs::<DP>(w.clone());
    
    let (acc, _) = match LFLinearizationProver::<_, T>::prove(&cm_i, &wit_acc, &mut transcript, &ccs) {
        Ok(result) => result,
        Err(e) => {
            println!("  Linearization failed: {:?}", e);
            return;
        }
    };
    
    // Now do the actual folding
    let mut prover_transcript = PoseidonTranscript::<RqNTT, CS>::default();
    let mut verifier_transcript = PoseidonTranscript::<RqNTT, CS>::default();
    
    // Time the proving
    let prove_start = Instant::now();
    let (new_acc, new_wit, proof) = match NIFSProver::<RqNTT, DP, T>::prove(
        &acc,
        &wit_acc,
        &cm_i,
        &wit,
        &mut prover_transcript,
        &ccs,
        &scheme,
    ) {
        Ok(result) => result,
        Err(e) => {
            println!("  NIFS prove failed: {:?}", e);
            return;
        }
    };
    let prove_time = prove_start.elapsed();
    
    // Serialize proof
    let mut proof_bytes = Vec::new();
    proof.serialize_with_mode(&mut proof_bytes, Compress::Yes).unwrap();
    let proof_size = proof_bytes.len();
    
    // Time the verification
    let verify_start = Instant::now();
    if let Err(e) = NIFSVerifier::<RqNTT, DP, T>::verify(
        &acc,
        &cm_i,
        &proof,
        &mut verifier_transcript,
        &ccs,
    ) {
        println!("  NIFS verify failed: {:?}", e);
        return;
    }
    let verify_time = verify_start.elapsed();
    
    println!("  Prove time: {:?}", prove_time);
    println!("  Verify time: {:?}", verify_time);
    println!("  Proof size: {} bytes", proof_size);
    
    // Key metric for GM-1B: verification operations
    // The verify_time gives us a proxy for verifier complexity
}
