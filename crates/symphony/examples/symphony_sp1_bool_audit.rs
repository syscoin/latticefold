//! Boolean-provenance audit for SP1 IR JSONL.
//!
//! Usage:
//!   SP1_IR_JSONL=/tmp/shrink_ir.jsonl \
//!     cargo run -p latticefold-plus --example symphony_sp1_bool_audit --release --features symphony
#![allow(clippy::print_stdout)]

use std::{fs::File, io::BufReader};

fn main() {
    let ir_path = std::env::var("SP1_IR_JSONL").expect("set SP1_IR_JSONL");
    let ir_file = File::open(&ir_path).expect("open SP1_IR_JSONL");
    let ops = latticefold_plus::symphony_sp1_ir::read_sp1_ir_jsonl(BufReader::new(ir_file))
        .expect("parse IR jsonl");

    // We enforce booleanity for Select conditions in our lowering, so treat missing provenance
    // for `Select` as a warning instead of a hard failure.
    let cfg = latticefold_plus::symphony_sp1_bool_audit::BoolAuditConfig {
        select_boolean_enforced_by_lowering: true,
    };
    let report = latticefold_plus::symphony_sp1_bool_audit::audit_boolean_provenance_with_config(&ops, &cfg);

    println!("Known-boolean vars: {}", report.known_boolean.len());
    for (op, n) in &report.requires_bool_counts {
        println!("Requires-boolean uses: {op} -> {n}");
    }
    if !report.warnings.is_empty() {
        println!(
            "Warnings: {} boolean requirements not proven by provenance but covered by lowering (showing first 10):",
            report.warnings.len()
        );
        for f in report.warnings.iter().take(10) {
            println!("  {:26} {}", f.op, f.detail);
        }
    }
    if report.failures.is_empty() {
        println!("OK: no boolean-provenance failures.");
    } else {
        println!("FAIL: {} boolean-provenance failures (showing first 20):", report.failures.len());
        for f in report.failures.iter().take(20) {
            println!("  {:26} {}", f.op, f.detail);
        }
        std::process::exit(1);
    }
}

