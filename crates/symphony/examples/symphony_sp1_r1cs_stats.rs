//! Load SP1 shrink verifier R1CS and report stats.
//!
//! Usage:
//!   SP1_R1CS=/path/to/shrink_verifier.r1cs \
//!     cargo run -p symphony --example symphony_sp1_r1cs_stats --release
//!
//! To generate the R1CS file, run in the SP1 fork:
//!   OUT_R1CS=shrink_verifier.r1cs cargo run -p sp1-prover \
//!     --bin dump_shrink_verify_constraints --release
#![allow(clippy::print_stdout)]

use symphony::symphony_sp1_r1cs::read_sp1_r1cs_stats;

fn main() {
    let r1cs_path = std::env::var("SP1_R1CS").expect("set SP1_R1CS=/path/to/shrink_verifier.r1cs");
    
    println!("Loading R1CS header from: {r1cs_path}");
    println!();
    
    match read_sp1_r1cs_stats(&r1cs_path) {
        Ok(stats) => {
            println!("=== SP1 Shrink Verifier R1CS Stats ===");
            println!("Variables:     {:>12}", stats.num_vars);
            println!("Constraints:   {:>12}", stats.num_constraints);
            println!("Public inputs: {:>12}", stats.num_public);
            println!("Non-zeros:     {:>12}", stats.total_nonzeros);
            println!("Digest:        {:02x}{:02x}{:02x}{:02x}...{:02x}{:02x}{:02x}{:02x}",
                stats.digest[0], stats.digest[1], stats.digest[2], stats.digest[3],
                stats.digest[28], stats.digest[29], stats.digest[30], stats.digest[31]);
            println!();
            
            // Estimate memory usage for loading
            let estimated_mem_mb = (stats.total_nonzeros * 12 + stats.num_constraints as u64 * 24) / 1_000_000;
            println!("Estimated memory to load: ~{estimated_mem_mb} MB");
            
            // Density metrics
            let total_possible = (stats.num_constraints as u64) * (stats.num_vars as u64);
            let density = stats.total_nonzeros as f64 / total_possible as f64;
            let avg_terms_per_row = stats.total_nonzeros as f64 / (stats.num_constraints as f64 * 3.0);
            println!("Matrix density: {:.6}%", density * 100.0);
            println!("Avg terms/row (per matrix): {:.1}", avg_terms_per_row);
        }
        Err(e) => {
            eprintln!("Error reading R1CS: {e}");
            std::process::exit(1);
        }
    }
}

