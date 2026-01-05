//! Sanity-run: interpret SP1 shrink-verifier IR JSONL + witness blocks and check basic constraints.
//!
//! This is a wiring check, not a full verifier.
//!
//! Usage:
//!   SP1_IR_JSONL=/tmp/shrink_ir.jsonl SP1_WITNESS_JSON=/tmp/shrink_witness.json \
//!     cargo run -p latticefold-plus --example symphony_sp1_ir_sanity --release --features symphony
#![allow(clippy::print_stdout)]

use std::{fs::File, io::BufReader};

fn main() {
    let ir_path = std::env::var("SP1_IR_JSONL").expect("set SP1_IR_JSONL");
    let wit_path = std::env::var("SP1_WITNESS_JSON").expect("set SP1_WITNESS_JSON");

    let ir_file = File::open(&ir_path).expect("open SP1_IR_JSONL");
    let ops = latticefold_plus::symphony_sp1_ir::read_sp1_ir_jsonl(BufReader::new(ir_file))
        .expect("parse IR jsonl");

    let wit_bytes = std::fs::read(&wit_path).expect("read SP1_WITNESS_JSON");
    let witness =
        latticefold_plus::symphony_sp1_witness::read_sp1_witness_blocks_json(&wit_bytes).unwrap();

    let mut interp = latticefold_plus::symphony_sp1_interp::Sp1Interp::new();
    // Harness mode: don't fail fast on asserts yet (we haven't implemented ext arithmetic or all ops).
    // We still count mismatches and will tighten this as lowering coverage increases.
    interp.strict = false;
    interp.consume_witness_for_hints = true;
    let mut seen = std::collections::BTreeMap::<String, usize>::new();

    for op in &ops {
        *seen.entry(op.op.clone()).or_default() += 1;
        interp.step(op, &witness).expect("interp step");
    }

    println!("Sanity OK: processed {} ops", ops.len());
    println!("Assert failures (non-fatal): {}", interp.num_assert_failures);
    println!("Missing values (defaulted to 0): {}", interp.num_missing_values);
    println!("Top 10 opcode counts:");
    let mut v = seen.into_iter().collect::<Vec<_>>();
    v.sort_by(|(a_op, a_n), (b_op, b_n)| b_n.cmp(a_n).then_with(|| a_op.cmp(b_op)));
    for (op, n) in v.into_iter().take(10) {
        println!("  {op:40} {n}");
    }
}

