//! Print an opcode audit report: where constraints live + whether our lowering covers it.
//!
//! Usage:
//!   SP1_IR_JSONL=/tmp/shrink_ir.jsonl \
//!     cargo run -p latticefold-plus --example symphony_sp1_opcode_audit --release --features symphony
#![allow(clippy::print_stdout)]

use std::{collections::BTreeMap, fs::File, io::BufReader};

fn main() {
    let ir_path = std::env::var("SP1_IR_JSONL").expect("set SP1_IR_JSONL");
    let ir_file = File::open(&ir_path).expect("open SP1_IR_JSONL");
    let ops = latticefold_plus::symphony_sp1_ir::read_sp1_ir_jsonl(BufReader::new(ir_file))
        .expect("parse IR jsonl");

    let mut by_op = BTreeMap::<String, usize>::new();
    for op in &ops {
        *by_op.entry(op.op.clone()).or_default() += 1;
    }

    let mut v = by_op.into_iter().collect::<Vec<_>>();
    v.sort_by(|(a_op, a_n), (b_op, b_n)| b_n.cmp(a_n).then_with(|| a_op.cmp(b_op)));

    println!("Total ops: {}", ops.len());
    println!("Opcode audit (top 40):");
    println!(
        "{:40} {:>9} {:>16} {:>10} {:>10} {:>14}  {}",
        "opcode", "count", "source", "needsBool", "boolInSrc", "lowering", "notes"
    );
    for (op, n) in v.into_iter().take(40) {
        let a = latticefold_plus::symphony_sp1_opcode_audit::audit_opcode(&op);
        println!(
            "{:40} {:>9} {:>16?} {:>10} {:>10} {:>14?}  {}",
            op,
            n,
            a.source,
            a.needs_boolean,
            a.boolean_enforced_by_source,
            a.lowering,
            a.notes
        );
    }
}

