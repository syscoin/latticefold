//! Classify SP1 shrink-verifier IR opcodes into ALU vs precompiles/chips.
//!
//! Usage:
//!   SP1_IR_JSONL=/tmp/shrink_ir.jsonl \
//!     cargo run -p latticefold-plus --example symphony_sp1_ir_classify --release --features symphony
#![allow(clippy::print_stdout)]

use std::{collections::BTreeMap, fs::File, io::BufReader};

fn main() {
    let ir_path = std::env::var("SP1_IR_JSONL").expect("set SP1_IR_JSONL");
    let ir_file = File::open(&ir_path).expect("open SP1_IR_JSONL");
    let ops = latticefold_plus::symphony_sp1_ir::read_sp1_ir_jsonl(BufReader::new(ir_file))
        .expect("parse IR jsonl");

    let mut by_class = BTreeMap::<String, usize>::new();
    let mut by_op = BTreeMap::<String, usize>::new();

    for op in &ops {
        *by_op.entry(op.op.clone()).or_default() += 1;
        let cls = latticefold_plus::symphony_sp1_lower::classify_sp1_ir_op(op);
        *by_class.entry(format!("{cls:?}")).or_default() += 1;
    }

    println!("Total ops: {}", ops.len());
    println!();
    println!("By class:");
    for (k, v) in by_class {
        println!("  {k:14} {v}");
    }

    println!();
    println!("Top 15 opcodes:");
    let mut v = by_op.into_iter().collect::<Vec<_>>();
    v.sort_by(|(a_op, a_n), (b_op, b_n)| b_n.cmp(a_n).then_with(|| a_op.cmp(b_op)));
    for (op, n) in v.into_iter().take(15) {
        println!("  {op:40} {n}");
    }
}

