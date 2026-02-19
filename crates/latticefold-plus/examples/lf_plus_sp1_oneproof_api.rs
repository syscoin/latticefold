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
//!   SP1_PUBLIC_VALUES_DIGEST_U64_CSV=u0,u1,u2,u3,u4,u5,u6,u7 \
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

use ark_ff::PrimeField;
use latticefold::transcript::poseidon::F257;

fn hex32(bytes: [u8; 32]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(64);
    for &b in &bytes {
        out.push(HEX[(b >> 4) as usize] as char);
        out.push(HEX[(b & 0x0f) as usize] as char);
    }
    out
}

fn hex_stmt_digest(fields: [F257; 32]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(128);
    for f in &fields {
        let v = (f.into_bigint().as_ref()[0] % 257) as u16;
        let b = v.to_le_bytes();
        for &x in &b {
            out.push(HEX[(x >> 4) as usize] as char);
            out.push(HEX[(x & 0x0f) as usize] as char);
        }
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
    eprintln!(
        "[oneproof] rayon_threads={} RAYON_NUM_THREADS={:?}",
        rayon::current_num_threads(),
        std::env::var("RAYON_NUM_THREADS").ok()
    );
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
        if pub_words.len() < 8 {
            panic!("expected at least 8 public inputs (got={})", pub_words.len());
        }
        let digest_words8_csv = pub_words[0..8]
            .iter()
            .map(|x| x.to_string())
            .collect::<Vec<_>>()
            .join(",");
        println!("SP1_PUBLIC_VALUES_DIGEST_U64_CSV={}", digest_words8_csv);
        eprintln!(
            "[oneproof] extracted num_public={} (use first 8 public inputs as public_values.digest words)",
            l_pub
        );
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
        let public_values_digest_words8: [u64; 8] = {
            let csv = std::env::var("SP1_PUBLIC_VALUES_DIGEST_U64_CSV")
                .expect("Set SP1_PUBLIC_VALUES_DIGEST_U64_CSV=u0,u1,u2,u3,u4,u5,u6,u7");
            let xs = parse_u64_csv(&csv);
            assert!(xs.len() == 8, "SP1_PUBLIC_VALUES_DIGEST_U64_CSV must have 8 values");
            xs.try_into().unwrap()
        };
        if hdr.num_public < 8 {
            panic!(
                "SP1 R1LF num_public too small for public_values.digest lane: num_public={}",
                hdr.num_public
            );
        }

        let out = latticefold_plus::sp1_oneproof_api::arm_sp1_oneproof_we_gate_write_lock_package(
            &r1lf_path,
            vk_hash,
            public_values_digest_words8,
            &lock_pkg_out,
            k_locks,
        )
        .expect("arm_sp1_oneproof_we_gate_write_lock_package");

        println!("stmt_digest_f257_le16=0x{}", hex_stmt_digest(out.manifest.stmt_digest));
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

    println!("stmt_digest_f257_le16=0x{}", hex_stmt_digest(out.stmt_digest));
    println!("lock_coin_seed=0x{}", hex32(out.lock_coin_seed));
    println!("share_candidates_len={}", out.share_candidates.len());
    if let Some((share_idx, cands)) = out.share_candidates.first() {
        println!("first_share_index={}", share_idx);
        println!("first_share_candidates={}", cands.len());
        if let Some(c0) = cands.first() {
            println!("first_candidate=0x{}", hex32(*c0));
        }
    }
    eprintln!("[oneproof] total_elapsed={:?}", t_total.elapsed());
}

