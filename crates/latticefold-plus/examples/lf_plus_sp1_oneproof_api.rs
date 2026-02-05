//! LF+ one-proof harness for SP1 shrink verifier R1LF (API-driven).
//!
//! This example is intentionally a **thin wrapper** over:
//! `latticefold_plus::sp1_oneproof_api::run_sp1_oneproof_we_gate_from_files`.
//!
//! Usage:
//!   SP1_R1LF=/path/to/shrink_verifier.r1lf \
//!   SP1_WITNESS=/path/to/shrink_verifier.witness.bundle \
//!     cargo run -p latticefold-plus --example lf_plus_sp1_oneproof --features we_gate --release

#![cfg(feature = "we_gate")]

fn hex32(bytes: [u8; 32]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(64);
    for &b in &bytes {
        out.push(HEX[(b >> 4) as usize] as char);
        out.push(HEX[(b & 0x0f) as usize] as char);
    }
    out
}

#[cfg(feature = "parallel")]
fn init_rayon_stack() {
    // Intermittent stack overflows can happen when a large-stack computation runs on a Rayon
    // worker thread (smaller stack) instead of the main thread (larger stack). This becomes more
    // likely with high parallelism (e.g. RAYON_NUM_THREADS=96).
    //
    // Mitigation: configure the global Rayon pool with an explicit (larger) stack size.
    // Set `RAYON_STACK_SIZE_BYTES` to override (bytes). Default: 32 MiB.
    let stack_bytes: usize = std::env::var("RAYON_STACK_SIZE_BYTES")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(32 * 1024 * 1024);

    let mut builder = rayon::ThreadPoolBuilder::new().stack_size(stack_bytes);

    // Respect RAYON_NUM_THREADS if provided.
    if let Some(n) = std::env::var("RAYON_NUM_THREADS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
    {
        builder = builder.num_threads(n);
    }

    // If Rayon was already initialized elsewhere, ignore and proceed.
    let _ = builder.build_global();
}

#[cfg(not(feature = "parallel"))]
fn init_rayon_stack() {}

fn main() {
    init_rayon_stack();
    let t_total = std::time::Instant::now();
    let r1lf_path = std::env::var("SP1_R1LF").expect("Set SP1_R1LF=/path/to/shrink.r1lf");
    let witness_path =
        std::env::var("SP1_WITNESS").expect("Set SP1_WITNESS=/path/to/shrink_verifier.witness.bundle");

    let out = latticefold_plus::sp1_oneproof_api::run_sp1_oneproof_we_gate_from_files(
        &r1lf_path,
        &witness_path,
    )
    .expect("run_sp1_oneproof_we_gate_from_files");

    println!("stmt_digest=0x{}", hex32(out.stmt_digest));
    println!("lock_coin_seed=0x{}", hex32(out.lock_coin_seed));
    eprintln!("[oneproof] total_elapsed={:?}", t_total.elapsed());
}

