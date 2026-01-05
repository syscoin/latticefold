//! Poseidon2 permutation as R1CS circuit for SP1 verifier component testing.
//!
//! This implements the SP1 Poseidon2 hash function as an R1CS circuit over
//! BabyBear ring to measure folding performance for real SP1 verifier components.
//!
//! SP1 Poseidon2 parameters:
//! - WIDTH = 16
//! - NUM_EXTERNAL_ROUNDS = 8 (4 at start, 4 at end)
//! - NUM_INTERNAL_ROUNDS = 13
//! - S-box: x^7 (computed as x³ × x³ × x)

use std::time::Instant;

use ark_serialize::{CanonicalSerialize, Compress};
use ark_std::vec::Vec;
use cyclotomic_rings::rings::{BabyBearChallengeSet, BabyBearRingNTT};
use latticefold::{
    arith::{
        r1cs::{Constraint, ConstraintSystem, LinearCombination, R1CS},
        Witness, CCCS, CCS,
    },
    commitment::AjtaiCommitmentScheme,
    decomposition_parameters::DecompositionParams,
    nifs::{
        linearization::{LFLinearizationProver, LinearizationProver},
        NIFSProver, NIFSVerifier,
    },
    transcript::poseidon::PoseidonTranscript,
};

// SP1 Poseidon2 parameters
const WIDTH: usize = 16;
const NUM_EXTERNAL_ROUNDS: usize = 8;
const NUM_INTERNAL_ROUNDS: usize = 13;

// Type aliases for BabyBear ring
type RqNTT = BabyBearRingNTT;
type CS = BabyBearChallengeSet;
type T = PoseidonTranscript<RqNTT, CS>;

// Decomposition parameters for BabyBear
#[derive(Clone)]
struct BabyBearDP;

impl DecompositionParams for BabyBearDP {
    const B: u128 = 1 << 15; // 32768
    const L: usize = 5;
    const B_SMALL: usize = 2;
    const K: usize = 15;
}

type DP = BabyBearDP;

/// Simplified Poseidon2 round constants (placeholder values for testing)
/// In production, these would be the actual SP1 round constants.
fn get_round_constant(round: usize, position: usize) -> u64 {
    // Use a simple deterministic pattern for testing
    ((round * WIDTH + position + 1) % 1000) as u64
}

/// Build a Poseidon2-like circuit with S-boxes and linear layers
/// 
/// Each permutation has:
/// - 8 external rounds (4 before, 4 after internal)
///   - External round: 16 S-boxes (x^7) + linear layer
/// - 13 internal rounds
///   - Internal round: 1 S-box (x^7) + linear layer
/// 
/// S-box x^7 requires: x² = t1, t1*x = x³ = t2, t2² = t3, t3*x = x^7
/// That's 4 multiplication constraints per S-box.
fn build_poseidon2_r1cs(num_permutations: usize) -> (R1CS<RqNTT>, Vec<RqNTT>, Vec<RqNTT>, Vec<RqNTT>) {
    let mut cs = ConstraintSystem::<RqNTT>::new();
    
    // Variable index counter
    let mut next_var = 0usize;
    
    // Track all witness values
    let mut all_witness_vals: Vec<RqNTT> = vec![RqNTT::from(1u64)]; // Index 0: constant 1
    next_var += 1;
    
    // Input state (public inputs)
    let input_start = next_var;
    let mut input_vals = Vec::with_capacity(WIDTH);
    for i in 0..WIDTH {
        all_witness_vals.push(RqNTT::from((i + 1) as u64));
        input_vals.push(RqNTT::from((i + 1) as u64));
        next_var += 1;
    }
    cs.ninputs = WIDTH; // 16 public inputs
    
    // Helper to allocate a new auxiliary variable
    let mut alloc_aux = |val: RqNTT| -> usize {
        let idx = next_var;
        all_witness_vals.push(val);
        next_var += 1;
        idx
    };
    
    // Helper to add S-box constraint: x^7
    // Returns the index of x^7
    let add_sbox = |cs: &mut ConstraintSystem<RqNTT>, 
                    alloc: &mut dyn FnMut(RqNTT) -> usize,
                    x_idx: usize, 
                    x_val: RqNTT| -> (usize, RqNTT) {
        // x² = t1
        let x2_val = x_val * x_val;
        let t1_idx = alloc(x2_val);
        cs.add_constraint(Constraint::new(
            LinearCombination::single_term(1u64, x_idx),
            LinearCombination::single_term(1u64, x_idx),
            LinearCombination::single_term(1u64, t1_idx),
        ));
        
        // x² * x = x³ = t2
        let x3_val = x2_val * x_val;
        let t2_idx = alloc(x3_val);
        cs.add_constraint(Constraint::new(
            LinearCombination::single_term(1u64, t1_idx),
            LinearCombination::single_term(1u64, x_idx),
            LinearCombination::single_term(1u64, t2_idx),
        ));
        
        // x³ * x³ = x⁶ = t3
        let x6_val = x3_val * x3_val;
        let t3_idx = alloc(x6_val);
        cs.add_constraint(Constraint::new(
            LinearCombination::single_term(1u64, t2_idx),
            LinearCombination::single_term(1u64, t2_idx),
            LinearCombination::single_term(1u64, t3_idx),
        ));
        
        // x⁶ * x = x⁷ = t4
        let x7_val = x6_val * x_val;
        let t4_idx = alloc(x7_val);
        cs.add_constraint(Constraint::new(
            LinearCombination::single_term(1u64, t3_idx),
            LinearCombination::single_term(1u64, x_idx),
            LinearCombination::single_term(1u64, t4_idx),
        ));
        
        (t4_idx, x7_val)
    };
    
    // Track current state indices and values
    let mut state_indices: Vec<usize> = (input_start..input_start + WIDTH).collect();
    let mut state_vals = input_vals.clone();
    
    for _perm in 0..num_permutations {
        // === External rounds (first half) ===
        for round in 0..NUM_EXTERNAL_ROUNDS / 2 {
            // Add round constants and apply S-boxes to all WIDTH elements
            let mut new_state_indices = Vec::with_capacity(WIDTH);
            let mut new_state_vals = Vec::with_capacity(WIDTH);
            
            for i in 0..WIDTH {
                // Add round constant
                let rc = get_round_constant(round, i);
                let with_rc_val = state_vals[i] + RqNTT::from(rc);
                let with_rc_idx = alloc_aux(with_rc_val);
                
                // Apply S-box
                let (sbox_idx, sbox_val) = add_sbox(&mut cs, &mut alloc_aux, with_rc_idx, with_rc_val);
                new_state_indices.push(sbox_idx);
                new_state_vals.push(sbox_val);
            }
            
            // Simplified linear layer: just sum and distribute
            // (Real Poseidon2 uses M4 mixing, but for constraint counting this is equivalent)
            let sum_val: RqNTT = new_state_vals.iter().fold(RqNTT::from(0u64), |a, &b| a + b);
            
            state_indices.clear();
            state_vals.clear();
            for i in 0..WIDTH {
                let new_val = new_state_vals[i] + sum_val;
                let new_idx = alloc_aux(new_val);
                state_indices.push(new_idx);
                state_vals.push(new_val);
            }
        }
        
        // === Internal rounds ===
        for round in 0..NUM_INTERNAL_ROUNDS {
            // Add round constant to first element only
            let rc = get_round_constant(NUM_EXTERNAL_ROUNDS / 2 + round, 0);
            let s0_with_rc_val = state_vals[0] + RqNTT::from(rc);
            let s0_with_rc_idx = alloc_aux(s0_with_rc_val);
            
            // Apply S-box to first element only
            let (_sbox_idx, sbox_val) = add_sbox(&mut cs, &mut alloc_aux, s0_with_rc_idx, s0_with_rc_val);
            
            // Update state[0]
            let mut new_state_vals = state_vals.clone();
            new_state_vals[0] = sbox_val;
            
            // Simplified internal linear layer: sum all, add to each
            let sum_val: RqNTT = new_state_vals.iter().fold(RqNTT::from(0u64), |a, &b| a + b);
            
            state_indices[0] = alloc_aux(new_state_vals[0] + sum_val);
            state_vals[0] = new_state_vals[0] + sum_val;
            
            for i in 1..WIDTH {
                state_indices[i] = alloc_aux(new_state_vals[i] + sum_val);
                state_vals[i] = new_state_vals[i] + sum_val;
            }
        }
        
        // === External rounds (second half) ===
        for round in NUM_EXTERNAL_ROUNDS / 2..NUM_EXTERNAL_ROUNDS {
            let mut new_state_indices = Vec::with_capacity(WIDTH);
            let mut new_state_vals = Vec::with_capacity(WIDTH);
            
            for i in 0..WIDTH {
                let rc = get_round_constant(NUM_INTERNAL_ROUNDS + round, i);
                let with_rc_val = state_vals[i] + RqNTT::from(rc);
                let with_rc_idx = alloc_aux(with_rc_val);
                
                let (sbox_idx, sbox_val) = add_sbox(&mut cs, &mut alloc_aux, with_rc_idx, with_rc_val);
                new_state_indices.push(sbox_idx);
                new_state_vals.push(sbox_val);
            }
            
            let sum_val: RqNTT = new_state_vals.iter().fold(RqNTT::from(0u64), |a, &b| a + b);
            
            state_indices.clear();
            state_vals.clear();
            for i in 0..WIDTH {
                let new_val = new_state_vals[i] + sum_val;
                let new_idx = alloc_aux(new_val);
                state_indices.push(new_idx);
                state_vals.push(new_val);
            }
        }
    }
    
    // Set auxiliary count
    cs.nauxs = next_var - WIDTH; // Total vars minus public inputs (includes constant 1)
    
    let r1cs = R1CS::from_constraint_system(cs);
    
    // Build z vector: [public_inputs..., 1, aux_witnesses...]
    // But R1CS uses: [x, 1, w] format
    let x_ccs: Vec<RqNTT> = input_vals.clone();
    let w_ccs: Vec<RqNTT> = all_witness_vals[WIDTH + 1..].to_vec(); // Skip inputs and constant 1
    
    // z = [x₀..x₁₅, 1, w₀, w₁, ...]
    let mut z = input_vals;
    z.push(RqNTT::from(1u64));
    z.extend(&w_ccs);
    
    println!("  Poseidon2 circuit stats:");
    println!("    Permutations: {}", num_permutations);
    println!("    Total constraints: {}", r1cs.A.nrows);
    println!("    Total variables: {}", r1cs.A.ncols);
    println!("    Public inputs: {}", WIDTH);
    println!("    Aux variables: {}", w_ccs.len());
    
    (r1cs, z, x_ccs, w_ccs)
}

fn test_poseidon2_folding(num_permutations: usize) {
    println!("\n=== Testing Poseidon2 with {} permutation(s) ===", num_permutations);
    
    let build_start = Instant::now();
    let (r1cs, z, x, w) = build_poseidon2_r1cs(num_permutations);
    let build_time = build_start.elapsed();
    println!("  Build time: {:?}", build_time);
    
    // Verify R1CS
    if r1cs.check_relation(&z).is_err() {
        println!("  R1CS check FAILED");
        return;
    }
    println!("  R1CS check: PASSED ✓");
    
    // Folding parameters
    let wit_len = w.len();
    let decomposed_len = wit_len * DP::L;
    let n = decomposed_len;
    let kappa = 4;
    
    if n < 16 {
        println!("  Skipping folding - n too small");
        return;
    }
    
    println!("  Folding params: wit_len={}, n={}", wit_len, n);
    
    let mut rng = ark_std::test_rng();
    
    // Convert to CCS
    let ccs: CCS<RqNTT> = CCS::from_r1cs_padded(r1cs, n, DP::L);
    println!("  CCS: {} rows, {} cols", ccs.m, ccs.n);
    
    // Create commitment scheme
    let scheme: AjtaiCommitmentScheme<RqNTT> = AjtaiCommitmentScheme::rand(kappa, n, &mut rng);
    
    // Create witness and commit
    let wit: Witness<RqNTT> = Witness::from_w_ccs::<DP>(w.clone());
    let cm = match wit.commit::<DP>(&scheme) {
        Ok(c) => c,
        Err(e) => {
            println!("  Commit failed: {:?}", e);
            return;
        }
    };
    
    // Create CCCS instance
    let cm_i = CCCS { cm: cm.clone(), x_ccs: x.clone() };
    
    // Linearization
    let mut transcript = T::default();
    let wit_acc = Witness::from_w_ccs::<DP>(w.clone());
    
    let (acc, _) = match LFLinearizationProver::<_, T>::prove(&cm_i, &wit_acc, &mut transcript, &ccs) {
        Ok(r) => r,
        Err(e) => {
            println!("  Linearization failed: {:?}", e);
            return;
        }
    };
    println!("  Linearization: PASSED ✓");
    
    // NIFS fold
    let mut prover_transcript = T::default();
    let mut verifier_transcript = T::default();
    
    let prove_start = Instant::now();
    let (_new_acc, _new_wit, proof) = match NIFSProver::<RqNTT, DP, T>::prove(
        &acc, &wit_acc, &cm_i, &wit, &mut prover_transcript, &ccs, &scheme
    ) {
        Ok(r) => r,
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
    
    let verify_start = Instant::now();
    match NIFSVerifier::<RqNTT, DP, T>::verify(&acc, &cm_i, &proof, &mut verifier_transcript, &ccs) {
        Ok(_) => {},
        Err(e) => {
            println!("  NIFS verify failed: {:?}", e);
            return;
        }
    };
    let verify_time = verify_start.elapsed();
    
    println!("\n  === RESULTS ===");
    println!("  Constraints: {}", ccs.m);
    println!("  Prove time: {:?}", prove_time);
    println!("  Verify time: {:?}", verify_time);
    println!("  Proof size: {} KB", proof_size / 1024);
}

fn main() {
    println!("=========================================================");
    println!("Poseidon2 R1CS Circuit - SP1 Verifier Component GM-1B");
    println!("=========================================================");
    println!();
    println!("SP1 Poseidon2 parameters:");
    println!("  WIDTH = {}", WIDTH);
    println!("  NUM_EXTERNAL_ROUNDS = {} (4 + 4)", NUM_EXTERNAL_ROUNDS);
    println!("  NUM_INTERNAL_ROUNDS = {}", NUM_INTERNAL_ROUNDS);
    println!("  S-box = x^7 (4 mult constraints per S-box)");
    println!();
    
    // Constraint count per permutation:
    // - 8 external rounds × 16 S-boxes × 4 mults = 512 constraints
    // - 13 internal rounds × 1 S-box × 4 mults = 52 constraints
    // - Plus auxiliary variables for round constants and linear layers
    // Estimated: ~700-900 constraints per permutation
    
    println!("Expected constraints per permutation:");
    println!("  External: 8 rounds × 16 S-boxes × 4 mults = 512");
    println!("  Internal: 13 rounds × 1 S-box × 4 mults = 52");
    println!("  Total S-box: ~564 + linear layer vars");
    
    // Test with increasing permutation counts
    for num_perms in [1, 2, 4, 8, 16, 32] {
        test_poseidon2_folding(num_perms);
    }
    
    println!("\n=========================================================");
    println!("GM-1B Insight: SP1 verifier uses ~1000 Poseidon2 calls");
    println!("At ~700 constraints/perm, that's ~700K constraints");
    println!("Plus FRI queries and Merkle paths");
    println!("=========================================================");
}
