//! Quick sanity tool: read SP1 shrink-verifier IR JSONL and print an opcode histogram.
//!
//! Usage:
//!   SP1_IR_JSONL=/path/to/shrink_ir.jsonl \
//!     cargo run -p latticefold-plus --example symphony_sp1_ir_stats --release --features symphony
#![allow(clippy::print_stdout)]

use std::{
    collections::BTreeMap,
    fs::File,
    io::{BufReader, Write},
};

fn main() {
    let path = std::env::var("SP1_IR_JSONL").expect("set SP1_IR_JSONL=/path/to/shrink_ir.jsonl");
    let f = File::open(&path).expect("open SP1_IR_JSONL");
    let ops = latticefold_plus::symphony_sp1_ir::read_sp1_ir_jsonl(BufReader::new(f))
        .expect("parse jsonl");

    let mut counts: BTreeMap<String, usize> = BTreeMap::new();
    for op in ops {
        *counts.entry(op.op).or_default() += 1;
    }
    let mut counts_sorted = counts.into_iter().collect::<Vec<_>>();
    counts_sorted.sort_by(|(a_op, a_n), (b_op, b_n)| b_n.cmp(a_n).then_with(|| a_op.cmp(b_op)));

    let mut out = std::io::stdout().lock();
    let _ = writeln!(out, "SP1 IR opcode histogram for {path}:");
    for (op, n) in counts_sorted {
        let _ = writeln!(out, "  {op:40} {n}");
    }

    if let Ok(wit_path) = std::env::var("SP1_WITNESS_JSON") {
        let bytes = std::fs::read(&wit_path).expect("read SP1_WITNESS_JSON");
        let blocks = latticefold_plus::symphony_sp1_witness::read_sp1_witness_blocks_json(&bytes)
            .expect("parse witness json");
        let _ = writeln!(out, "\nSP1 witness blocks: {} ({wit_path})", blocks.len());
    }
}

