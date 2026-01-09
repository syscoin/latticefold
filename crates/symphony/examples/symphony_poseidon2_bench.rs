//! Symphony Poseidon2 Benchmark — SP1 Verifier Component GM-1B
//!
//! This implements the SP1 Poseidon2 hash function as a Hadamard relation (M1·f ⊙ M2·f = M3·f)
//! and folds it using Symphony's Π_fold to measure performance.
//!
//! SP1 Poseidon2 parameters:
//! - WIDTH = 16
//! - NUM_EXTERNAL_ROUNDS = 8 (4 at start, 4 at end)
//! - NUM_INTERNAL_ROUNDS = 13
//! - S-box: x^7 (computed as x² → x³ → x⁶ → x⁷, 4 mult constraints per S-box)
//!
//! Run:
//! - `cargo run -p symphony --example symphony_poseidon2_bench --release

use std::sync::Arc;
use std::time::Instant;
use ark_ff::{Field, Fp256, MontBackend, MontConfig, PrimeField};

fn main() {
    use std::panic::{catch_unwind, AssertUnwindSafe};
    use stark_rings::PolyRing;
    use cyclotomic_rings::rings::FrogPoseidonConfig;
    use stark_rings::cyclotomic_ring::models::frog_ring;

    struct PanicHookGuard(Option<Box<dyn Fn(&std::panic::PanicHookInfo<'_>) + Sync + Send + 'static>>);
    impl PanicHookGuard {
        fn silence_expected_panics() -> Self {
            let prev = std::panic::take_hook();
            std::panic::set_hook(Box::new(|_| {
                // Intentionally silent: the benchmark uses `catch_unwind` for expected failures
                // (e.g. too-small k_g in balanced decomposition).
            }));
            Self(Some(prev))
        }
    }
    impl Drop for PanicHookGuard {
        fn drop(&mut self) {
            if let Some(prev) = self.0.take() {
                std::panic::set_hook(prev);
            }
        }
    }

    fn run_for_ring<
        R: stark_rings::CoeffRing,
        PC: cyclotomic_rings::rings::GetPoseidonParams<<<R>::BaseRing as ark_ff::Field>::BasePrimeField>,
    >(
        ring_name: &str,
    ) where
        R::BaseRing: stark_rings::Zq + stark_rings::balanced_decomposition::Decompose + ark_ff::PrimeField,
    {
        const WIDTH: usize = 16;
        const NUM_EXTERNAL_ROUNDS: usize = 8;
        const NUM_INTERNAL_ROUNDS: usize = 13;

        println!("=========================================================");
        println!("Symphony Poseidon2 Benchmark — {ring_name}");
        println!("=========================================================");
        println!();
        println!("SP1 Poseidon2 parameters:");
        println!("  WIDTH = {WIDTH}");
        println!("  NUM_EXTERNAL_ROUNDS = {NUM_EXTERNAL_ROUNDS} (4 + 4)");
        println!("  NUM_INTERNAL_ROUNDS = {NUM_INTERNAL_ROUNDS}");
        println!("  S-box = x^7 (4 mult constraints per S-box)");
        println!();

        println!("Expected constraints per permutation:");
        println!("  External: 8 rounds × 16 S-boxes × 4 mults = 512");
        println!("  Internal: 13 rounds × 1 S-box × 4 mults = 52");
        println!("  Total S-box: ~564 constraints");
        println!();

        // Digit-sizing hint based on modulus size.
        let d_prime: u32 = (R::dimension() as u32) / 2;
        let bits_per_digit = (d_prime as f64).log2();
        let mod_bits = <<R as PolyRing>::BaseRing as ark_ff::Field>::BasePrimeField::MODULUS_BIT_SIZE;
        let k_suggest = ((mod_bits as f64) / bits_per_digit).ceil() as usize;
        println!(
            "Digit-sizing hint: dim={} => d'={d_prime} (~{bits_per_digit:.2} bits/digit), modulus bits={mod_bits} => suggest k_g≈{k_suggest}",
            R::dimension()
        );

        // Sweep permutation counts.
        //
        // IMPORTANT:
        // This benchmark stresses witness magnitudes; for Frog (~64-bit modulus) and small ring dim,
        // a tiny k_g (like 3) will often overflow balanced decomposition. We therefore default to
        // the computed `k_suggest` unless overridden via `K_G`.
        let k_list = if let Ok(s) = std::env::var("K_G") {
            vec![s.parse::<usize>().expect("K_G must be a usize")]
        } else {
            vec![k_suggest.max(3)]
        };
        let perms_list: Vec<usize> = vec![1, 2, 4, 8];
        for &k_g in &k_list {
            for &num_perms in &perms_list {
                // Silence panic-hook output only for the `catch_unwind` region so expected failures
                // don't print scary backtraces.
                let _hook_guard = PanicHookGuard::silence_expected_panics();
                let r = catch_unwind(AssertUnwindSafe(|| {
                    test_symphony_poseidon2_folding::<R, PC>(num_perms, k_g);
                }));
                if let Err(_) = r {
                    println!("  !! PANIC: decomposition overflow for perms={num_perms}, k_g={k_g}");
                }
            }
        }
    }

    // NOTE: Frog is ~64-bit modulus and is very expensive once k_g is raised high enough.
    // For B1 we focus on BabyBear modulus first.
    // Default to Frog (matches the paper's instantiation modulus family and has d=16 in this repo).
    // Note: the built-in BabyBear ring in this repo has d=72, which is incompatible with our Π_fold
    // MLE domain requirement that m*d is a power-of-two.
    run_for_ring::<frog_ring::RqPoly, FrogPoseidonConfig>("Frog ring (64-bit prime, d=16)");

    // Constraint count per permutation:
    // - 8 external rounds × 16 S-boxes × 4 mults = 512 constraints
    // - 13 internal rounds × 1 S-box × 4 mults = 52 constraints
    // Total: ~564 mult constraints per permutation

    // All printing is done inside `run_for_ring`.
}

/// Helper alias: Poseidon's base prime field for a given ring `R`.
type BF<R> = <<R as stark_rings::PolyRing>::BaseRing as Field>::BasePrimeField;

// 256-bit prime field for the large-field embedding/packing path (Rev2).
// secp256k1 prime: 2^256 - 2^32 - 977
#[derive(MontConfig)]
#[modulus = "115792089237316195423570985008687907852837564279074904382605163141518161494337"]
#[generator = "7"]
pub struct Secp256k1Config;
type FLarge = Fp256<MontBackend<Secp256k1Config, 4>>;

/// Build Poseidon2 Hadamard relation matrices (M1, M2, M3) for Symphony.
///
/// For the Hadamard relation M1·f ⊙ M2·f = M3·f:
/// - Each S-box x^7 requires 4 mult constraints
/// - Witness f contains: [state_0..state_{WIDTH-1}, aux_vars...]
fn build_poseidon2_hadamard_matrices<R: stark_rings::PolyRing>(
    num_permutations: usize,
) -> (
    stark_rings_linalg::SparseMatrix<R>,
    stark_rings_linalg::SparseMatrix<R>,
    stark_rings_linalg::SparseMatrix<R>,
    Vec<R>,
    usize,
) {
    use stark_rings_linalg::SparseMatrix;

    const WIDTH: usize = 16;
    const NUM_EXTERNAL_ROUNDS: usize = 8;
    const NUM_INTERNAL_ROUNDS: usize = 13;

    // Count constraints and variables first
    let sboxes_per_perm = NUM_EXTERNAL_ROUNDS * WIDTH + NUM_INTERNAL_ROUNDS;
    let constraints_per_perm = sboxes_per_perm * 4; // 4 mult constraints per S-box
    let total_constraints = constraints_per_perm * num_permutations;

    // Variables: initial state + intermediate values
    // Each S-box creates 5 values: rc_add + x² + x³ + x⁶ + x⁷
    // External round: 16 sboxes × 5 + 16 state update = 96 per round
    // Internal round: 1 sbox × 5 + 16 state update = 21 per round
    let aux_per_round_external = WIDTH * 5 + WIDTH; // 5 per sbox + state update
    let aux_per_round_internal = 5 + WIDTH; // 1 sbox + state update
    let aux_per_perm = NUM_EXTERNAL_ROUNDS * aux_per_round_external / 2 * 2
        + NUM_INTERNAL_ROUNDS * aux_per_round_internal;

    let total_aux = aux_per_perm * num_permutations;
    let n = WIDTH + total_aux; // witness length

    // Round to power of 2 for folding (with extra headroom)
    let n_padded = (n * 2).next_power_of_two();
    
    // m must be >= m_J where m_J = (n / l_h) * lambda_pj
    // With l_h=64, lambda_pj=32: m_J = n_padded / 64 * 32 = n_padded / 2
    // So ensure m >= n_padded / 2 and m is a multiple of m_J
    let m_j_estimate = (n_padded / 64) * 32;
    let m_min = total_constraints.max(m_j_estimate);
    let m = m_min.next_power_of_two();

    // Build sparse matrices - format is (value, column_index) per row
    let mut m1_coeffs: Vec<Vec<(R, usize)>> = vec![Vec::new(); m];
    let mut m2_coeffs: Vec<Vec<(R, usize)>> = vec![Vec::new(); m];
    let mut m3_coeffs: Vec<Vec<(R, usize)>> = vec![Vec::new(); m];

    // Build witness
    let mut witness = vec![R::ZERO; n_padded];

    // Initialize state.
    //
    // IMPORTANT:
    // This benchmark only constrains the Poseidon2 S-box *multiplication* structure (x^7),
    // not the full set of linear/additive constraints of a real Poseidon2 permutation.
    //
    // We still use it to stress-test the *magnitude* of witness values for Π_rg decomposition
    // by actually computing x^7 chains (this is what broke and forced `ONE`-witnesses).
    let mut state_indices: Vec<usize> = (0..WIDTH).collect();
    let mut state_vals: Vec<R> = (0..WIDTH).map(|_| R::from(5u128)).collect(); // 5^7 = 78125 > 4096
    for i in 0..WIDTH {
        witness[i] = state_vals[i];
    }

    let mut next_aux = WIDTH;
    let mut constraint_idx = 0;
    let one = R::ONE;

    for _perm in 0..num_permutations {
        // === External rounds (first half) ===
        for _round in 0..NUM_EXTERNAL_ROUNDS / 2 {
            let mut new_state_indices = Vec::with_capacity(WIDTH);
            let mut new_state_vals = Vec::with_capacity(WIDTH);

            for i in 0..WIDTH {
                let with_rc_idx = next_aux;
                let with_rc_val = state_vals[i] + R::ONE;
                witness[with_rc_idx] = with_rc_val;
                next_aux += 1;

                let t1_idx = next_aux;
                let t1_val = with_rc_val * with_rc_val;
                witness[t1_idx] = t1_val;
                next_aux += 1;
                m1_coeffs[constraint_idx].push((one, with_rc_idx));
                m2_coeffs[constraint_idx].push((one, with_rc_idx));
                m3_coeffs[constraint_idx].push((one, t1_idx));
                constraint_idx += 1;

                let t2_idx = next_aux;
                let t2_val = t1_val * with_rc_val;
                witness[t2_idx] = t2_val;
                next_aux += 1;
                m1_coeffs[constraint_idx].push((one, t1_idx));
                m2_coeffs[constraint_idx].push((one, with_rc_idx));
                m3_coeffs[constraint_idx].push((one, t2_idx));
                constraint_idx += 1;

                let t3_idx = next_aux;
                let t3_val = t2_val * t2_val;
                witness[t3_idx] = t3_val;
                next_aux += 1;
                m1_coeffs[constraint_idx].push((one, t2_idx));
                m2_coeffs[constraint_idx].push((one, t2_idx));
                m3_coeffs[constraint_idx].push((one, t3_idx));
                constraint_idx += 1;

                let t4_idx = next_aux;
                let t4_val = t3_val * with_rc_val;
                witness[t4_idx] = t4_val;
                next_aux += 1;
                m1_coeffs[constraint_idx].push((one, t3_idx));
                m2_coeffs[constraint_idx].push((one, with_rc_idx));
                m3_coeffs[constraint_idx].push((one, t4_idx));
                constraint_idx += 1;

                new_state_indices.push(t4_idx);
                new_state_vals.push(t4_val);
            }

            // Linear layer placeholder: sum and add (like the R1CS toy circuit).
            // Not constrained here, but keeps subsequent S-box inputs growing realistically.
            let sum_val = new_state_vals.iter().copied().fold(R::ZERO, |a, b| a + b);
            state_indices.clear();
            state_vals.clear();
            for i in 0..WIDTH {
                let new_idx = next_aux;
                let new_val = new_state_vals[i] + sum_val;
                witness[new_idx] = new_val;
                next_aux += 1;
                state_indices.push(new_idx);
                state_vals.push(new_val);
            }
        }

        // === Internal rounds ===
        for _round in 0..NUM_INTERNAL_ROUNDS {
            // Round constant (+1)
            let s0_with_rc_idx = next_aux;
            let s0_with_rc_val = state_vals[0] + R::ONE;
            witness[s0_with_rc_idx] = s0_with_rc_val;
            next_aux += 1;

            // S-box for first element (4 constraints)
            let t1_idx = next_aux;
            let t1_val = s0_with_rc_val * s0_with_rc_val;
            witness[t1_idx] = t1_val;
            next_aux += 1;
            m1_coeffs[constraint_idx].push((one, s0_with_rc_idx));
            m2_coeffs[constraint_idx].push((one, s0_with_rc_idx));
            m3_coeffs[constraint_idx].push((one, t1_idx));
            constraint_idx += 1;

            let t2_idx = next_aux;
            let t2_val = t1_val * s0_with_rc_val;
            witness[t2_idx] = t2_val;
            next_aux += 1;
            m1_coeffs[constraint_idx].push((one, t1_idx));
            m2_coeffs[constraint_idx].push((one, s0_with_rc_idx));
            m3_coeffs[constraint_idx].push((one, t2_idx));
            constraint_idx += 1;

            let t3_idx = next_aux;
            let t3_val = t2_val * t2_val;
            witness[t3_idx] = t3_val;
            next_aux += 1;
            m1_coeffs[constraint_idx].push((one, t2_idx));
            m2_coeffs[constraint_idx].push((one, t2_idx));
            m3_coeffs[constraint_idx].push((one, t3_idx));
            constraint_idx += 1;

            let t4_idx = next_aux;
            let t4_val = t3_val * s0_with_rc_val;
            witness[t4_idx] = t4_val;
            next_aux += 1;
            m1_coeffs[constraint_idx].push((one, t3_idx));
            m2_coeffs[constraint_idx].push((one, s0_with_rc_idx));
            m3_coeffs[constraint_idx].push((one, t4_idx));
            constraint_idx += 1;

            // State update (all bounded)
            state_indices[0] = t4_idx;
            state_vals[0] = t4_val;
            let sum_val = state_vals.iter().copied().fold(R::ZERO, |a, b| a + b);
            for i in 0..WIDTH {
                let new_idx = next_aux;
                let new_val = state_vals[i] + sum_val;
                witness[new_idx] = new_val;
                next_aux += 1;
                state_indices[i] = new_idx;
                state_vals[i] = new_val;
            }
        }

        // === External rounds (second half) ===
        for _round in NUM_EXTERNAL_ROUNDS / 2..NUM_EXTERNAL_ROUNDS {
            let mut new_state_indices = Vec::with_capacity(WIDTH);
            let mut new_state_vals = Vec::with_capacity(WIDTH);

            for i in 0..WIDTH {
                let with_rc_idx = next_aux;
                let with_rc_val = state_vals[i] + R::ONE;
                witness[with_rc_idx] = with_rc_val;
                next_aux += 1;

                // S-box (4 constraints)
                let t1_idx = next_aux;
                let t1_val = with_rc_val * with_rc_val;
                witness[t1_idx] = t1_val;
                next_aux += 1;
                m1_coeffs[constraint_idx].push((one, with_rc_idx));
                m2_coeffs[constraint_idx].push((one, with_rc_idx));
                m3_coeffs[constraint_idx].push((one, t1_idx));
                constraint_idx += 1;

                let t2_idx = next_aux;
                let t2_val = t1_val * with_rc_val;
                witness[t2_idx] = t2_val;
                next_aux += 1;
                m1_coeffs[constraint_idx].push((one, t1_idx));
                m2_coeffs[constraint_idx].push((one, with_rc_idx));
                m3_coeffs[constraint_idx].push((one, t2_idx));
                constraint_idx += 1;

                let t3_idx = next_aux;
                let t3_val = t2_val * t2_val;
                witness[t3_idx] = t3_val;
                next_aux += 1;
                m1_coeffs[constraint_idx].push((one, t2_idx));
                m2_coeffs[constraint_idx].push((one, t2_idx));
                m3_coeffs[constraint_idx].push((one, t3_idx));
                constraint_idx += 1;

                let t4_idx = next_aux;
                let t4_val = t3_val * with_rc_val;
                witness[t4_idx] = t4_val;
                next_aux += 1;
                m1_coeffs[constraint_idx].push((one, t3_idx));
                m2_coeffs[constraint_idx].push((one, with_rc_idx));
                m3_coeffs[constraint_idx].push((one, t4_idx));
                constraint_idx += 1;

                new_state_indices.push(t4_idx);
                new_state_vals.push(t4_val);
            }

            // Linear layer
            let sum_val = new_state_vals.iter().copied().fold(R::ZERO, |a, b| a + b);
            state_indices.clear();
            state_vals.clear();
            for i in 0..WIDTH {
                let new_idx = next_aux;
                let new_val = new_state_vals[i] + sum_val;
                witness[new_idx] = new_val;
                next_aux += 1;
                state_indices.push(new_idx);
                state_vals.push(new_val);
            }
        }
    }

    // Construct SparseMatrix directly
    let m1 = SparseMatrix {
        nrows: m,
        ncols: n_padded,
        coeffs: m1_coeffs,
    };
    let m2 = SparseMatrix {
        nrows: m,
        ncols: n_padded,
        coeffs: m2_coeffs,
    };
    let m3 = SparseMatrix {
        nrows: m,
        ncols: n_padded,
        coeffs: m3_coeffs,
    };

    (m1, m2, m3, witness, constraint_idx)
}

fn r1cs_decompose_witness_and_expand_matrices<R: stark_rings::CoeffRing>(
    m1: &stark_rings_linalg::SparseMatrix<R>,
    m2: &stark_rings_linalg::SparseMatrix<R>,
    m3: &stark_rings_linalg::SparseMatrix<R>,
    witness: &[R],
    k_cs: usize,
    base_b: u128,
) -> (
    stark_rings_linalg::SparseMatrix<R>,
    stark_rings_linalg::SparseMatrix<R>,
    stark_rings_linalg::SparseMatrix<R>,
    Vec<R>,
)
where
    R::BaseRing: stark_rings::Zq + ark_ff::PrimeField,
{
    use ark_ff::Field as _;
    use ark_ff::PrimeField;
    use stark_rings_linalg::SparseMatrix;

    let d = R::dimension();
    let n = witness.len();

    // Read modulus into u128 (works for p <= 2^128).
    let modulus_big = <R::BaseRing as PrimeField>::MODULUS;
    let mut q: u128 = 0;
    for (i, limb) in modulus_big.as_ref().iter().enumerate() {
        q |= (*limb as u128) << (64 * i);
    }
    let q_half = q / 2;
    let b = base_b as i128;
    let b_half = (base_b / 2) as i128;

    // Decompose each ring element coefficient-wise into k_cs digits base b.
    // New witness layout: for original index j, digit t is at (j*k_cs + t).
    let mut out = vec![R::ZERO; n * k_cs];
    for (j, r) in witness.iter().enumerate() {
        let coeffs = r.coeffs();
        debug_assert_eq!(coeffs.len(), d);

        // digits[t][k] is the digit at position t of coefficient k.
        let mut digits: Vec<Vec<R::BaseRing>> = vec![vec![R::BaseRing::ZERO; d]; k_cs];
        for (k, c) in coeffs.iter().enumerate() {
            // Convert coefficient to centered i128 in (-q/2, q/2].
            let big = c.into_bigint();
            let mut u: u128 = 0;
            for (i, limb) in big.as_ref().iter().enumerate() {
                u |= (*limb as u128) << (64 * i);
            }
            let mut x: i128 = if u > q_half { u as i128 - q as i128 } else { u as i128 };

            for t in 0..k_cs {
                // Balanced remainder in [-b/2, b/2].
                let mut r0 = x.rem_euclid(b);
                if r0 > b_half {
                    r0 -= b;
                }
                x = (x - r0) / b;
                let fe = if r0 >= 0 {
                    R::BaseRing::from(r0 as u64)
                } else {
                    -R::BaseRing::from((-r0) as u64)
                };
                digits[t][k] = fe;
            }
        }
        for t in 0..k_cs {
            out[j * k_cs + t] = R::from(digits[t].clone());
        }
    }

    // Expand sparse matrices: each original column j becomes k_cs columns with weights b^t.
    fn expand<R: stark_rings::CoeffRing>(
        m: &SparseMatrix<R>,
        k_cs: usize,
        base_b: u128,
    ) -> SparseMatrix<R>
    where
        R::BaseRing: stark_rings::Zq,
    {
        let mut coeffs = vec![Vec::<(R, usize)>::new(); m.nrows];
        for (row, row_terms) in m.coeffs.iter().enumerate() {
            let mut new_row = Vec::with_capacity(row_terms.len() * k_cs);
            for (a, col) in row_terms.iter() {
                let mut bpow: u128 = 1;
                for t in 0..k_cs {
                    let w = R::from(bpow);
                    new_row.push((*a * w, col * k_cs + t));
                    bpow = bpow.saturating_mul(base_b);
                }
            }
            coeffs[row] = new_row;
        }
        SparseMatrix::<R> {
            nrows: m.nrows,
            ncols: m.ncols * k_cs,
            coeffs,
        }
    }

    (expand(m1, k_cs, base_b), expand(m2, k_cs, base_b), expand(m3, k_cs, base_b), out)
}

fn test_symphony_poseidon2_folding<
    R: stark_rings::CoeffRing,
    PC: cyclotomic_rings::rings::GetPoseidonParams<<<R>::BaseRing as ark_ff::Field>::BasePrimeField>,
>(
    num_permutations: usize,
    k_g: usize,
)
where
    R::BaseRing: stark_rings::Zq + stark_rings::balanced_decomposition::Decompose + ark_ff::PrimeField,
{
    use ark_ff::{BigInteger, Field, PrimeField};
    use latticefold::commitment::AjtaiCommitmentScheme;
    use symphony::{
        pcs::dpp_folding_pcs_l2::folding_pcs_l2_params,
        pcs::folding_pcs_l2::{
            kron_ct_in_mul, kron_i_a_mul, BinMatrix, DenseMatrix, FoldingPcsL2ProofCore,
            verify_folding_pcs_l2_with_c_matrices,
        },
        rp_rgchk::RPParams,
        symphony_open::MultiAjtaiOpenVerifier,
        symphony_pifold_batched::{verify_pi_fold_cp_poseidon_fs, PiFoldMatrices},
        symphony_pifold_streaming::{prove_pi_fold_poseidon_fs, PiFoldStreamingConfig},
        transcript::PoseidonTraceOp,
        we_gate_arith::WeGateDr1csBuilder,
    };
    use cyclotomic_rings::rings::GetPoseidonParams;

    println!("\n=== Testing Symphony Poseidon2 with {} permutation(s), k_g={} ===", num_permutations, k_g);

    let build_start = Instant::now();
    let (m1, m2, m3, witness, actual_constraints) =
        build_poseidon2_hadamard_matrices::<R>(num_permutations);
    let build_time = build_start.elapsed();

    // Paper-style R1CS witness decomposition (k_cs digits base 2^4), expanding the witness and the
    // linear maps so the Hadamard relation is unchanged. This is the Table 1 instantiation setup
    // and is enabled unconditionally in this benchmark.
    // R1CS witness decomposition always enabled in this benchmark
    println!("  R1CS witness decomposition: ENABLED (k_cs=16, b=2^4)");
    let (m1, m2, m3, witness) =
        r1cs_decompose_witness_and_expand_matrices::<R>(&m1, &m2, &m3, &witness, 16, 16);

    // After witness decomposition, the witness length grows by k_cs, which increases m_J.
    // Π_fold requires: m_J <= m and m multiple of m_J, and also requires m*d is a power-of-two.
    // This benchmark pads the matrices with empty rows to satisfy these shape constraints.
    let mut m1 = m1;
    let mut m2 = m2;
    let mut m3 = m3;

    let n = witness.len();
    let mut m = m1.nrows;

    println!("  Hadamard matrices built:");
    println!("    Permutations: {num_permutations}");
    println!("    Actual constraints: {actual_constraints}");
    println!("    Matrix rows (m): {m}");
    println!("    Witness length (n): {n}");
    println!("    Build time: {build_time:?}");

    // Build `rg_params` early so we can compute the required m_J for this run.
    let rg_params = if k_g == 3 {
        // Paper-style prototype mode:
        // - use d' := d-2 in Eq.(29) / (33)
        // - keep m_J small by using lambda_pj=1 (toy bench knob; paper uses 2^8)
        RPParams {
            l_h: 64,
            lambda_pj: 1,
            k_g,
            d_prime: (R::dimension() as u128) - 2,
        }
    } else {
        RPParams {
            l_h: 64,
            lambda_pj: 32,
            k_g,
            d_prime: (R::dimension() as u128) / 2,
        }
    };

    let blocks = n / rg_params.l_h;
    let m_j = blocks
        .checked_mul(rg_params.lambda_pj)
        .expect("m_J overflow");
    if m < m_j || m % m_j != 0 {
        let m_new = (m.max(m_j)).next_power_of_two();
        println!(
            "  !! Padding matrices: m={} -> {} to satisfy m_J={} (l_h={}, lambda_pj={})",
            m, m_new, m_j, rg_params.l_h, rg_params.lambda_pj
        );
        // Extend coefficient rows with empties (zero rows).
        m1.coeffs.resize(m_new, Vec::new());
        m2.coeffs.resize(m_new, Vec::new());
        m3.coeffs.resize(m_new, Vec::new());
        m1.nrows = m_new;
        m2.nrows = m_new;
        m3.nrows = m_new;
        m = m_new;
        println!("    Matrix rows (m): {m} (padded)");
    }

    // Witness magnitude proxy: maximum bit-length among base-prime-field limbs.
    //
    // This is not "signed magnitude" in Zq, but it is a useful indicator of whether values
    // are staying tiny vs spanning the full modulus.
    let mut max_bits = 0u32;
    for r in &witness {
        for c in r.coeffs() {
            for limb in c.to_base_prime_field_elements() {
                max_bits = max_bits.max(limb.into_bigint().num_bits());
            }
        }
    }
    println!("    Witness max limb bits: {max_bits}");

    // Verify Hadamard relation holds: M1·f ⊙ M2·f = M3·f
    let y1 = m1.try_mul_vec(&witness).expect("M1*f failed");
    let y2 = m2.try_mul_vec(&witness).expect("M2*f failed");
    let y3 = m3.try_mul_vec(&witness).expect("M3*f failed");
    let mut hadamard_ok = true;
    for row in 0..m {
        if y1[row] * y2[row] != y3[row] {
            hadamard_ok = false;
            println!("    Hadamard FAILED at row {row}");
            break;
        }
    }
    if hadamard_ok {
        println!("  Hadamard relation: PASSED ✓");
    } else {
        println!("  Hadamard relation: FAILED ✗");
        return;
    }

    // Use Arcs so we can run multiple provers without cloning huge structures.
    let m1 = Arc::new(m1);
    let m2 = Arc::new(m2);
    let m3 = Arc::new(m3);
    let witness = Arc::new(witness);

    // Commitment setup
    let kappa = 8; // Ajtai commitment rows
    // Seeded Ajtai is the intended "CRS-as-seed" instantiation (public, fixed seed).
    const MASTER_SEED: [u8; 32] = *b"SYMPHONY_AJTAI_SEED_V1_000000000";
    let scheme = AjtaiCommitmentScheme::<R>::seeded(b"cm_f", MASTER_SEED, kappa, n);
    let cm = scheme
        .commit_const_coeff_fast(witness.as_ref())
        .unwrap()
        .as_ref()
        .to_vec();

    // Symphony Π_rg parameters (defined above, after decomposition so it matches padded shapes).

    // CP commitment schemes for aux messages
    let scheme_had =
        AjtaiCommitmentScheme::<R>::seeded(b"cfs_had_u", MASTER_SEED, kappa, 3 * R::dimension());
    let scheme_mon = AjtaiCommitmentScheme::<R>::seeded(b"cfs_mon_b", MASTER_SEED, kappa, rg_params.k_g);
    let scheme_g = AjtaiCommitmentScheme::<R>::seeded(b"cm_g", MASTER_SEED, kappa, m * R::dimension());

    // Public inputs (statement binding)
    let public_inputs: Vec<R::BaseRing> = vec![
        R::BaseRing::from(0x5350315fu128), // "SP1_"
        R::BaseRing::from(num_permutations as u128),
        R::BaseRing::from(k_g as u128),
    ];

    // Open verifier for CP commitments (cloneable, so we can reuse across runs).
    let open = MultiAjtaiOpenVerifier::<R>::new()
        .with_scheme("cfs_had_u", scheme_had.clone())
        .with_scheme("cfs_mon_b", scheme_mon.clone());

    // Helper: verify R_cp for a produced output
    let verify = |out: &symphony::symphony_pifold_batched::PiFoldProverOutput<R>| -> Result<(), String> {
        let attempt = verify_pi_fold_cp_poseidon_fs::<R, PC>(
            PiFoldMatrices::Shared([m1.as_ref(), m2.as_ref(), m3.as_ref()]),
            &[cm.clone()],
            &out.proof,
            &open,
            &out.cfs_had_u,
            &out.cfs_mon_b,
            &out.aux,
            &public_inputs,
        );
        let _ = attempt.result?;
        Ok(())
    };

    // Streaming prover (canonical path)
    println!("  Π_fold prove (streaming): START...");

    // Build Ms: single instance
    let ms = vec![[m1.clone(), m2.clone(), m3.clone()]];

    let cfg = PiFoldStreamingConfig::default();
    let prove_start = Instant::now();
    let out = prove_pi_fold_poseidon_fs::<R, PC>(
        ms.as_slice(),
        &[cm.clone()],
        &[witness.clone()],
        &public_inputs,
        Some(&scheme_had),
        Some(&scheme_mon),
        &scheme_g,
        rg_params.clone(),
        &cfg,
    );
    let prove_time = prove_start.elapsed();
    let out = match out {
        Ok(o) => o,
        Err(e) => {
            println!("  Π_fold prove (streaming) FAILED: {e}");
            return;
        }
    };
    println!("  Π_fold prove (streaming): PASSED ✓ ({prove_time:?})");

    println!("  R_cp verify (streaming): START...");
    let verify_start = Instant::now();
    match verify(&out) {
        Ok(_) => {
            let verify_time = verify_start.elapsed();
            println!("  R_cp verify (streaming): PASSED ✓ ({verify_time:?})");
        }
        Err(e) => {
            println!("  R_cp verify (streaming) FAILED: {e}");
            return;
        }
    }

    // Gate dR1CS size report (R_cp vs full with PCS delta), using the *real verifier trace*.
    let poseidon_cfg = <PC as GetPoseidonParams<BF<R>>>::get_poseidon_config();
    let attempt = verify_pi_fold_cp_poseidon_fs::<R, PC>(
        PiFoldMatrices::Shared([m1.as_ref(), m2.as_ref(), m3.as_ref()]),
        &[cm.clone()],
        &out.proof,
        &open,
        &out.cfs_had_u,
        &out.cfs_mon_b,
        &out.aux,
        &public_inputs,
    );
    let _ = attempt
        .result
        .expect("unexpected verify failure when extracting trace for dr1cs sizing");
    let trace = attempt.trace;

    let (rcp, rcp_asg) = WeGateDr1csBuilder::r_cp_poseidon_pifold_math_and_cfs_openings::<R>(
        &poseidon_cfg,
        &trace.ops,
        &[cm.clone()],
        &out.proof,
        &scheme_had,
        &scheme_mon,
        &out.aux,
        &out.cfs_had_u,
        &out.cfs_mon_b,
    )
    .expect("build r_cp dr1cs failed");
    rcp.check(&rcp_asg).expect("r_cp dr1cs unsat");

    let squeeze_bytes: Vec<Vec<u8>> = trace
        .ops
        .iter()
        .filter_map(|op| match op {
            PoseidonTraceOp::SqueezeBytes { out, .. } => Some(out.clone()),
            _ => None,
        })
        .collect();
    if squeeze_bytes.is_empty() {
        println!(
            "    gate dr1cs: r_cp(nvars={}, constraints={}) (no SqueezeBytes => skip full pcs)",
            rcp.nvars,
            rcp.constraints.len()
        );
    } else {
        let pcs_coin_squeeze_idx = 0usize;
        let c_bytes = &squeeze_bytes[pcs_coin_squeeze_idx];
        let mut bits = Vec::with_capacity(c_bytes.len() * 8);
        for &b in c_bytes {
            for i in 0..8 {
                bits.push(((b >> i) & 1) == 1);
            }
        }

        // Tiny PCS instance (plumbing + delta measurement; not the full Poseidon2 witness PCS yet).
        let r = 1usize;
        let kappa_pcs = 2usize;
        let pcs_n = 4usize;
        let delta = 4u64;
        let alpha = 1usize;
        let beta0 = 1u64 << 10;
        let beta1 = 2 * beta0;
        let beta2 = 2 * beta1;
        let c1 = BinMatrix {
            rows: r * kappa_pcs,
            cols: kappa_pcs,
            data: (0..(r * kappa_pcs * kappa_pcs))
                .map(|i| if bits[i] { <BF<R> as Field>::ONE } else { <BF<R> as Field>::ZERO })
                .collect(),
        };
        let c2 = BinMatrix {
            rows: r * kappa_pcs,
            cols: kappa_pcs,
            data: (0..(r * kappa_pcs * kappa_pcs))
                .map(|i| {
                    if bits[(r * kappa_pcs * kappa_pcs) + i] {
                        <BF<R> as Field>::ONE
                    } else {
                        <BF<R> as Field>::ZERO
                    }
                })
                .collect(),
        };
        let mut a_data = vec![<BF<R> as Field>::ZERO; pcs_n * (r * pcs_n * alpha)];
        for i in 0..pcs_n {
            a_data[i * (r * pcs_n * alpha) + i] = <BF<R> as Field>::ONE;
        }
        let a = DenseMatrix::new(pcs_n, r * pcs_n * alpha, a_data);
        let pcs_params = folding_pcs_l2_params(r, kappa_pcs, pcs_n, delta, alpha, beta0, beta1, beta2, a);
        let x0 = vec![<BF<R> as Field>::ONE; r];
        let x1 = vec![<BF<R> as Field>::ONE; r];
        let x2 = vec![<BF<R> as Field>::ONE; r];
        let y0 = vec![<BF<R> as Field>::ONE; pcs_params.y0_len()];
        let y1 = kron_ct_in_mul(&c1, pcs_n, &y0);
        let y2 = kron_ct_in_mul(&c2, pcs_n, &y1);
        let t_pcs = kron_i_a_mul(&pcs_params.a, pcs_params.kappa, pcs_params.r * pcs_params.n * pcs_params.alpha, &y0);
        let v0 = y0.clone();
        let v1 = y1.clone();
        let v2 = y2.clone();
        let u_pcs = v0.clone();
        let pcs_core = FoldingPcsL2ProofCore { y0, v0, y1, v1, y2, v2 };
        verify_folding_pcs_l2_with_c_matrices(&pcs_params, &t_pcs, &x0, &x1, &x2, &u_pcs, &pcs_core, &c1, &c2)
            .expect("native folding pcs sanity failed");

        let (full, full_asg) = WeGateDr1csBuilder::poseidon_plus_pifold_plus_cfs_plus_pcs::<R>(
            &poseidon_cfg,
            &trace.ops,
            &[cm.clone()],
            &out.proof,
            &scheme_had,
            &scheme_mon,
            &out.aux,
            &out.cfs_had_u,
            &out.cfs_mon_b,
            &pcs_params,
            &t_pcs,
            &x0,
            &x1,
            &x2,
            &u_pcs,
            &pcs_core,
            pcs_coin_squeeze_idx,
        )
        .expect("build full gate dr1cs failed");
        full.check(&full_asg).expect("full gate dr1cs unsat");

        println!(
            "    gate dr1cs: r_cp(nvars={}, constraints={})  full(nvars={}, constraints={})  delta(nvars={}, constraints={})",
            rcp.nvars,
            rcp.constraints.len(),
            full.nvars,
            full.constraints.len(),
            full.nvars.saturating_sub(rcp.nvars),
            full.constraints.len().saturating_sub(rcp.constraints.len()),
        );

        // Optional: run the *actual DPP proving + verify-with-query* over the full gate dR1CS.
        //
        // Enable on a fast machine:
        //   DPP=1 cargo run -p symphony --example symphony_poseidon2_bench --release
        //
        // This is intentionally not CI-friendly (can be very slow for large gates).
        if std::env::var("DPP").ok().as_deref() == Some("1") {
            use dpp::{
                dr1cs_flpcp::Dr1csInstanceSparse as DppDr1csInstanceSparse,
                dr1cs_flpcp::RsDr1csNpFlpcpSparse,
                embedding::EmbeddingParams,
                pipeline::build_rev2_dpp_sparse_boolean_auto,
                sparse::SparseVec,
                BooleanProofFlpcpSparse,
            };
            use rand::{rngs::StdRng, SeedableRng};

            println!("    DPP: building sparse Dr1csInstance for full gate...");
            let t0 = Instant::now();
            let k = full.constraints.len();
            let nvars = full.nvars;
            let mut a_rows = Vec::with_capacity(k);
            let mut b_rows = Vec::with_capacity(k);
            let mut c_rows = Vec::with_capacity(k);
            for row in &full.constraints {
                a_rows.push(SparseVec::new(row.a.clone()));
                b_rows.push(SparseVec::new(row.b.clone()));
                c_rows.push(SparseVec::new(row.c.clone()));
            }
            let dr1cs = DppDr1csInstanceSparse::<BF<R>> { n: nvars, a: a_rows, b: b_rows, c: c_rows };
            println!("    DPP: dr1cs build: {:?} (nvars={}, k={})", t0.elapsed(), nvars, k);

            // NP-style RS FLPCP over BF: x=[] (public), witness z_w = full assignment.
            let ell = 2 * k;
            let flpcp = RsDr1csNpFlpcpSparse::<BF<R>>::new(dr1cs, 0, ell);
            let x_small: Vec<BF<R>> = vec![];

            println!("    DPP: RS-FLPCP prove: START...");
            let t1 = Instant::now();
            let pi_field = flpcp.prove(&x_small, &full_asg);
            let t1e = t1.elapsed();
            println!(
                "    DPP: RS-FLPCP prove: {:?} (pi_field_len={})",
                t1e,
                pi_field.len()
            );

            println!("    DPP: booleanize + embed/pack: START...");
            let t2 = Instant::now();
            let boolized = BooleanProofFlpcpSparse::<BF<R>, _>::new(flpcp.clone());
            let pi_bits = boolized.encode_proof_bits(&pi_field);
            let dpp = build_rev2_dpp_sparse_boolean_auto::<BF<R>, FLarge, _>(
                flpcp,
                EmbeddingParams { gamma: 2, assume_boolean_proof: true, k_prime: 0 },
            )
            .expect("build_rev2_dpp_sparse_boolean_auto");
            let t2e = t2.elapsed();
            println!(
                "    DPP: booleanize+build: {:?} (pi_bits_len={})",
                t2e,
                pi_bits.len()
            );

            // Lift x and π into FLarge.
            println!("    DPP: lift proof to large field: START...");
            let t3 = Instant::now();
            let x_large: Vec<FLarge> = vec![];
            let pi_large = pi_bits
                .iter()
                .map(|wi| FLarge::from_le_bytes_mod_order(&wi.into_bigint().to_bytes_le()))
                .collect::<Vec<_>>();
            let t3e = t3.elapsed();
            println!("    DPP: lift: {:?} (pi_large_len={})", t3e, pi_large.len());

            // Sample query and verify.
            println!("    DPP: verify_with_query: START...");
            let t4 = Instant::now();
            let mut rng = StdRng::seed_from_u64(12345);
            let q = dpp.sample_query(&mut rng, &x_large).expect("sample_query");
            let ok = dpp.verify_with_query(&x_large, &pi_large, &q).expect("verify_with_query");
            let t4e = t4.elapsed();
            println!("    DPP: verify_with_query: {:?} (ok={})", t4e, ok);
        }
    }

    println!("\n  === RESULTS ===");
    println!("  Permutations: {num_permutations}");
    println!("  Constraints: {actual_constraints} (padded to {m})");
    println!("  Witness size: {n}");
    println!("  Streaming: prove={prove_time:?}, coins_bytes={}", out.proof.coins.bytes.len());
}
