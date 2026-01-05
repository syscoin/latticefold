//! Range / canonicality audit for SP1 IR JSONL.
//!
//! Usage:
//!   SP1_IR_JSONL=/tmp/shrink_ir.jsonl \
//!     cargo run -p latticefold-plus --example symphony_sp1_range_audit --release --features symphony
#![allow(clippy::print_stdout)]

use std::{fs::File, io::BufReader};

fn main() {
    let ir_path = std::env::var("SP1_IR_JSONL").expect("set SP1_IR_JSONL");
    let ir_file = File::open(&ir_path).expect("open SP1_IR_JSONL");
    let ops = latticefold_plus::symphony_sp1_ir::read_sp1_ir_jsonl(BufReader::new(ir_file))
        .expect("parse IR jsonl");

    let report = latticefold_plus::symphony_sp1_range_audit::audit_range_sites(&ops);

    if report.items.is_empty() {
        println!("No range-relevant opcodes found in this IR stream.");
        return;
    }

    println!(
        "{:28} {:>8} {:>12} {:>12}  {}",
        "opcode", "count", "gnarkRange", "bbNativeNeeds", "notes"
    );
    for it in report.items {
        println!(
            "{:28} {:>8} {:>12} {:>12}  {}",
            it.opcode, it.count, it.gnark_range_checks, it.babybear_native_needs_range, it.notes
        );
    }
}

