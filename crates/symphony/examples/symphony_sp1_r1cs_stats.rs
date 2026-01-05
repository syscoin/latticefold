//! Build a stub R1CS constraint set from the felt-level SP1 ALU subset and report counts.
//!
//! Usage:
//!   SP1_IR_JSONL=/tmp/shrink_ir.jsonl \
//!     cargo run -p latticefold-plus --example symphony_sp1_r1cs_stats --release --features symphony
#![allow(clippy::print_stdout)]

use std::{fs::File, io::BufReader};

fn main() {
    let ir_path = std::env::var("SP1_IR_JSONL").expect("set SP1_IR_JSONL");
    let ir_file = File::open(&ir_path).expect("open SP1_IR_JSONL");
    let ops = latticefold_plus::symphony_sp1_ir::read_sp1_ir_jsonl(BufReader::new(ir_file))
        .expect("parse IR jsonl");

    let r1cs = latticefold_plus::symphony_sp1_r1cs::lower_sp1_ir_to_r1cs_felt_subset(&ops);

    println!("Witness len (felt-only, w[0]=1): {}", r1cs.witness_len);
    println!("Constraints (felt-only subset): {}", r1cs.constraints.len());

    // Count tags.
    let mut by_tag = std::collections::BTreeMap::<&'static str, usize>::new();
    for c in &r1cs.constraints {
        *by_tag.entry(c.tag).or_default() += 1;
    }
    println!("Top 15 constraint tags:");
    let mut v = by_tag.into_iter().collect::<Vec<_>>();
    v.sort_by(|(a_t, a_n), (b_t, b_n)| b_n.cmp(a_n).then_with(|| a_t.cmp(b_t)));
    for (t, n) in v.into_iter().take(15) {
        println!("  {t:24} {n}");
    }
}

