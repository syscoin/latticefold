//! LF+ one-proof harness for SP1 shrink verifier R1LF (API-driven).
//!
//! This example is intentionally a **thin wrapper** over:
//! - `latticefold_plus::sp1_oneproof_api::arm_sp1_oneproof_we_gate_write_lock_package` (arming)
//! - `latticefold_plus::sp1_oneproof_api::decap_sp1_oneproof_we_gate_from_files_with_lock_package` (decap)
//!
//! Usage:
//!   SP1_R1LF=/path/to/shrink_verifier.r1lf \
//!   LFP_ONEPROOF_LOCK_PKG_OUT=/path/to/lock_pkg.bin \
//!   LFP_ONEPROOF_K=16 \
//!   SP1_VK_HASH_HEX=0x... \
//!   SP1_PUBLIC_INPUTS_U64=1,2,3,... \
//!     cargo run -p latticefold-plus --example lf_plus_sp1_oneproof_api --features we_gate --release
//!
//! Or decap (read pre-armed package):
//!   SP1_R1LF=/path/to/shrink_verifier.r1lf \
//!   SP1_WITNESS=/path/to/shrink_verifier.witness.bundle \
//!   LFP_ONEPROOF_LOCK_PKG_IN=/path/to/lock_pkg.bin \
//!     cargo run -p latticefold-plus --example lf_plus_sp1_oneproof_api --features we_gate --release
//!
//! One-time helper (extract statement inputs from witness bundle):
//!   SP1_R1LF=/path/to/shrink_verifier.r1lf \
//!   SP1_WITNESS=/path/to/shrink_verifier.witness.bundle \
//!   LFP_ONEPROOF_EXTRACT_STATEMENT=1 \
//!     cargo run -p latticefold-plus --example lf_plus_sp1_oneproof_api --features we_gate --release

#![cfg(feature = "we_gate")]

use ark_ff::Field;
use cyclotomic_rings::rings::GoldilocksRing64 as R;
use stark_rings::PolyRing;

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

#[inline]
fn babybear_u64_to_centered_host<F: Field>(x: u64, p_bb: u64) -> F {
    debug_assert!(p_bb > 1);
    let half = p_bb / 2;
    if x > half {
        let neg = p_bb - x;
        -F::from(neg)
    } else {
        F::from(x)
    }
}

fn parse_hex_32(mut s: &str) -> [u8; 32] {
    if let Some(rest) = s.strip_prefix("0x").or_else(|| s.strip_prefix("0X")) {
        s = rest;
    }
    let s = s.trim();
    assert!(s.len() == 64, "expected 64 hex chars for 32-byte value");
    let mut out = [0u8; 32];
    for i in 0..32 {
        let b = u8::from_str_radix(&s[2 * i..2 * i + 2], 16).expect("hex decode");
        out[i] = b;
    }
    out
}

fn parse_u64_csv(s: &str) -> Vec<u64> {
    let mut out = Vec::new();
    for raw in s.split(',') {
        let t = raw.trim();
        if t.is_empty() {
            continue;
        }
        let t = t.replace('_', "");
        let v = if let Some(rest) = t.strip_prefix("0x").or_else(|| t.strip_prefix("0X")) {
            u64::from_str_radix(rest, 16).expect("hex u64")
        } else {
            t.parse::<u64>().expect("decimal u64")
        };
        out.push(v);
    }
    out
}

fn main() {
    init_rayon_stack();
    let t_total = std::time::Instant::now();
    let r1lf_path = std::env::var("SP1_R1LF").expect("Set SP1_R1LF=/path/to/shrink.r1lf");
    let witness_path = std::env::var("SP1_WITNESS")
        .unwrap_or_else(|_| "/dev/null".to_string());

    // One-time helper: extract statement inputs from the witness bundle so arming can be run
    // witness-free later (WE headspace).
    if std::env::var("LFP_ONEPROOF_EXTRACT_STATEMENT")
        .ok()
        .as_deref()
        == Some("1")
    {
        let hdr = latticefold_plus::sp1_r1lf::read_r1lf_stats(&r1lf_path).expect("read_r1lf_stats");
        let bundle = latticefold_plus::sp1_witness_io::load_sp1_witness_any(&witness_path, hdr.num_vars)
            .expect("load_sp1_witness_any");
        if bundle.r1lf_digest != hdr.digest {
            panic!("witness bundle r1lf_digest mismatch vs r1lf header digest");
        }
        let (vk_hash, _committed_values_digest) = bundle.public_inputs;
        println!("SP1_VK_HASH_HEX=0x{}", hex32(vk_hash));
        let l_pub = hdr.num_public;
        let pub_words = &bundle.witness[1..1 + l_pub];
        let csv = pub_words
            .iter()
            .map(|x| x.to_string())
            .collect::<Vec<_>>()
            .join(",");
        println!("SP1_PUBLIC_INPUTS_U64={}", csv);
        eprintln!("[oneproof] extracted num_public={}", l_pub);
        return;
    }

    // If LOCK_PKG_OUT is set: run statement-only arming and write the public package.
    if let Ok(lock_pkg_out) = std::env::var("LFP_ONEPROOF_LOCK_PKG_OUT") {
        let k_locks: usize = std::env::var("LFP_ONEPROOF_K")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(16);

        let hdr = latticefold_plus::sp1_r1lf::read_r1lf_stats(&r1lf_path).expect("read_r1lf_stats");

        // Arming is statement-only: it must not depend on a proof-specific witness.
        // We require statement binding inputs directly.
        let vk_hash = parse_hex_32(
            &std::env::var("SP1_VK_HASH_HEX").expect("Set SP1_VK_HASH_HEX=0x... (32-byte hex)"),
        );
        let pub_u64 = parse_u64_csv(
            &std::env::var("SP1_PUBLIC_INPUTS_U64")
                .expect("Set SP1_PUBLIC_INPUTS_U64=comma-separated u64 witness words (len=num_public)"),
        );
        if pub_u64.len() != hdr.num_public {
            panic!(
                "SP1_PUBLIC_INPUTS_U64 length mismatch: got={} expected_num_public={}",
                pub_u64.len(),
                hdr.num_public
            );
        }
        type BFSmall = <<R as PolyRing>::BaseRing as Field>::BasePrimeField;
        let public_inputs: Vec<BFSmall> = pub_u64
            .iter()
            .map(|&x| babybear_u64_to_centered_host::<BFSmall>(x, hdr.p_bb))
            .collect();

        let out = latticefold_plus::sp1_oneproof_api::arm_sp1_oneproof_we_gate_write_lock_package(
            &r1lf_path,
            vk_hash,
            &public_inputs,
            &lock_pkg_out,
            k_locks,
        )
        .expect("arm_sp1_oneproof_we_gate_write_lock_package");

        println!("stmt_digest=0x{}", hex32(out.manifest.stmt_digest));
        println!("lock_coin_seed=0x{}", hex32(out.manifest.lock_coin_seed));
        println!("k_locks={}", out.k_locks);
        println!("lock_pkg_bytes={}", out.lock_pkg_bytes);
        eprintln!("[oneproof] total_elapsed={:?}", t_total.elapsed());
        return;
    }

    // Otherwise: decap using pre-armed public package.
    let lock_pkg_in = std::env::var("LFP_ONEPROOF_LOCK_PKG_IN")
        .expect("Set LFP_ONEPROOF_LOCK_PKG_IN=/path/to/lock_pkg.bin");
    let out =
        latticefold_plus::sp1_oneproof_api::decap_sp1_oneproof_we_gate_from_files_with_lock_package(
            &r1lf_path,
            &witness_path,
            &lock_pkg_in,
        )
        .expect("decap_sp1_oneproof_we_gate_from_files_with_lock_package");

    println!("stmt_digest=0x{}", hex32(out.stmt_digest));
    println!("lock_coin_seed=0x{}", hex32(out.lock_coin_seed));
    println!("decapped_key=0x{}", hex32(out.decapped_key));
    eprintln!("[oneproof] total_elapsed={:?}", t_total.elapsed());
}

