//! WE-gate + DPP integration bench (research).
//!
//! Current scope:
//! - Build a WE sparse dR1CS for verifying one `CmProof` (commitment transform / Π_cm)
//! - Convert it into the prototype dpp::dr1cs_flpcp pipeline and run verification
//!
//! This is not yet the full LF+ WE gate (DecompProof still TODO).

#![allow(non_snake_case)]

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use cyclotomic_rings::rings::FrogPoseidonConfig as PC;
use cyclotomic_rings::rings::GetPoseidonParams;

use ark_ff::{BigInteger, Field, Fp384, MontBackend, MontConfig, PrimeField};
use rand::{rngs::StdRng, SeedableRng};

use latticefold_plus::cm::Cm;
use latticefold_plus::rgchk::{DecompParameters, Rg, RgInstance};
use stark_rings::cyclotomic_ring::models::frog_ring::RqPoly as R;
use stark_rings::PolyRing;
use stark_rings_linalg::{Matrix, SparseMatrix};

use latticefold_plus::recording_transcript::TracePoseidonTranscript;
use latticefold_plus::we_gate_arith::{build_we_dr1cs_for_cm_proof_debug, WeCmBuildDebug};
use latticefold_plus::we_statement::WeParams;

// -----------------------------------------------------------------------------
// Big field for Rev2 embedding (p' large enough for packing).
// -----------------------------------------------------------------------------

#[derive(MontConfig)]
// NIST P-384 prime (as used by Symphony’s Rev2 embedding bench).
#[modulus = "39402006196394479212279040100143613805079739270465446667948293404245721771496870329047266088258938001861606973112319"]
#[generator = "2"]
pub struct Secp384r1Config;
type FBig = Fp384<MontBackend<Secp384r1Config, 6>>;

fn lift_to_big<Fs: PrimeField>(x: Fs) -> FBig {
    FBig::from_le_bytes_mod_order(&x.into_bigint().to_bytes_le())
}

fn bench_we_dpp(c: &mut Criterion) {
    // Keep defaults small-ish so local runs work; override on server by editing this file for now.
    // Fast toy parameters: keeps tau_unpadded_len <= 1024 so this bench focuses on WE/DPP,
    // not prover-side RG setup.
    let k = 2usize;
    let kappa = 2usize;
    let ell = 1usize;
    let b = 2u128;
    let n = 1 << 10;
    let nvars = ark_std::log2(n) as usize;

    let dparams = DecompParameters { b, k, l: ell };
    let mut rng = ark_std::test_rng();

    // Single-instance Cm setup.
    let f = vec![R::from(<R as PolyRing>::BaseRing::ZERO); n];
    let A = Matrix::<R>::rand(&mut rng, kappa, n);
    let inst = RgInstance::from_f(f, &A, &dparams);
    let rg = Rg {
        nvars,
        instances: vec![inst],
        dparams: dparams.clone(),
    };
    let cm = Cm { rg };
    let M: Vec<SparseMatrix<R>> = vec![]; // keep Mlen=0 for now

    // Prover-side Cm proof.
    let mut ts = latticefold_plus::transcript::PoseidonTranscript::empty::<PC>();
    let (_com, proof) = cm.prove(&M, &mut ts);

    // Record verifier transcript ops.
    let mut rec = TracePoseidonTranscript::<R>::empty::<PC>();
    proof.verify(&M, &mut rec).expect("cm proof verify");
    let trace = rec.trace().clone();

    // Statement params prefix (placeholder values; we only bind layout in this bench).
    let params = WeParams {
        nvars_setchk: nvars as u64,
        degree_setchk: 3,
        nvars_cm: nvars as u64,
        degree_cm: 2,
        kappa: kappa as u64,
        ring_dim_d: R::dimension() as u64,
        k: k as u64,
        l: ell as u64,
        mlen: M.len() as u64,
    };

    let poseidon_cfg = PC::get_poseidon_config();

    let mut group = c.benchmark_group("we_dpp");
    group.sample_size(10);

    group.bench_function(BenchmarkId::new("build_we_dr1cs_cm_proof", n), |bch| {
        bch.iter(|| {
            let (out, dbg) =
                build_we_dr1cs_for_cm_proof_debug::<R>(&poseidon_cfg, &trace, &params, &proof, M.len())
                    .expect("build_we_dr1cs_for_cm_proof_debug");
            if let Err(e) = out.inst.check(&out.assignment) {
                let msg = explain_failed_constraint(&out, &dbg, &e);
                panic!("dr1cs satisfied: {e}\n{msg}");
            }
        })
    });

    group.bench_function(BenchmarkId::new("dpp_verify_cm_proof", n), |bch| {
        // Build once outside the timed loop.
        let (out, dbg) =
            build_we_dr1cs_for_cm_proof_debug::<R>(&poseidon_cfg, &trace, &params, &proof, M.len())
                .expect("build_we_dr1cs_for_cm_proof_debug");
        if let Err(e) = out.inst.check(&out.assignment) {
            let msg = explain_failed_constraint(&out, &dbg, &e);
            panic!("dr1cs satisfied: {e}\n{msg}");
        }

        type FSmall = <<R as PolyRing>::BaseRing as ark_ff::Field>::BasePrimeField;
        // Convert sparse dR1CS -> sparse dR1CS instance for the prototype RS FLPCP.
        let inst_sparse = dpp::dr1cs_flpcp::Dr1csInstanceSparse::<FSmall> {
            n: out.inst.nvars,
            a: out
                .inst
                .constraints
                .iter()
                .map(|row| dpp::SparseVec::new(row.a.clone()))
                .collect(),
            b: out
                .inst
                .constraints
                .iter()
                .map(|row| dpp::SparseVec::new(row.b.clone()))
                .collect(),
            c: out
                .inst
                .constraints
                .iter()
                .map(|row| dpp::SparseVec::new(row.c.clone()))
                .collect(),
        };
        let k_rows = inst_sparse.k();
        let ell = 2 * k_rows;
        // IMPORTANT (WE/DPP path):
        // Use the NP-style FLPCP so the *entire* dR1CS assignment lives in the proof (π),
        // and we can safely use the Rev2 "assume_boolean_proof=true" bound regime.
        //
        // This matches the Symphony bench structure (x is empty; witness is in π).
        let flpcp = dpp::dr1cs_flpcp::RsDr1csNpFlpcpSparse::<FSmall>::new(inst_sparse, 0, ell);

        let x_small: Vec<FSmall> = vec![];
        let z_w_small = out.assignment.clone();
        let pi_field_small = flpcp.prove(&x_small, &z_w_small);

        // Rev2 pipeline (Booleanize -> Embed -> Pack) into a large field.
        //
        // Use the same builder as Symphony to match bounds/packing behavior exactly.
        let boolized = dpp::BooleanProofFlpcpSparse::<FSmall, _>::new(flpcp.clone());
        let pi_bits_small = boolized.encode_proof_bits(&pi_field_small);

        let dppv = dpp::pipeline::build_rev2_dpp_sparse_boolean_auto::<FSmall, FBig, _>(
            flpcp,
            dpp::EmbeddingParams {
                gamma: 2,
                assume_boolean_proof: true,
                k_prime: 0,
            },
        )
        .expect("build_rev2_dpp_sparse_boolean_auto");

        let x_big: Vec<FBig> = vec![];
        let pi_big = pi_bits_small
            .iter()
            .copied()
            .map(lift_to_big::<FSmall>)
            .collect::<Vec<_>>();

        let mut rng = StdRng::seed_from_u64(123);
        let q = dppv.sample_query(&mut rng, &x_big).expect("dpp sample_query");
        bch.iter(|| {
            let ok = dppv
                .verify_with_query(&x_big, &pi_big, &q)
                .expect("dpp verify_with_query");
            assert!(ok);
        })
    });

    group.finish();
}

fn parse_failed_constraint_idx(msg: &str) -> Option<usize> {
    // expected "constraint {i} failed"
    let msg = msg.trim();
    let msg = msg.strip_prefix("constraint ")?;
    let msg = msg.strip_suffix(" failed")?;
    msg.parse::<usize>().ok()
}

fn explain_failed_constraint(
    out: &latticefold_plus::we_gate_arith::WeDr1csOutput<<<R as PolyRing>::BaseRing as Field>::BasePrimeField>,
    dbg: &WeCmBuildDebug,
    err: &str,
) -> String {
    let Some(i) = parse_failed_constraint_idx(err) else {
        return "[we_dpp] could not parse failed constraint index".to_string();
    };
    let mut acc = 0usize;
    let names = [
        "poseidon",
        "params",
        "setchk_verify",
        "dcom_absorb",
        "cm_short_bytes",
        "cm_field_chals",
        "cm_verify",
    ];
    for (part_idx, &cnt) in dbg.part_constraints.iter().enumerate() {
        if i < acc + cnt {
            let name = names.get(part_idx).copied().unwrap_or("unknown");
            let mut msg = format!(
                "[we_dpp] failed constraint {i} is in PART {part_idx} ({name}), start={acc}, len={cnt}"
            );
            if part_idx == 6 && !dbg.cm_phase_marks.is_empty() {
                let local = i - acc;
                let mut phase = "unknown";
                for (j, &m) in dbg.cm_phase_marks.iter().enumerate() {
                    if local < m {
                        phase = dbg.cm_phase_names.get(j).map(|s| s.as_str()).unwrap_or("unknown");
                        break;
                    }
                }
                if phase == "unknown" {
                    if let Some(last) = dbg.cm_phase_names.last() {
                        phase = last;
                    }
                }
                msg.push_str(&format!("\n[we_dpp] cm_verify local_idx={local}, phase≈{phase}"));
            }
            return msg;
        }
        acc += cnt;
    }
    // Glue constraints
    let glue_idx = i.saturating_sub(dbg.base_constraints);
    if glue_idx < dbg.glue.len() {
        let (pa, xa, pb, xb) = dbg.glue[glue_idx];
        // Compute merged-space indices to show witness mismatch.
        let mut offsets = Vec::with_capacity(dbg.part_nvars.len());
        let mut off = 0usize;
        for &nv in &dbg.part_nvars {
            offsets.push(off);
            off += nv - 1;
        }
        let ga = if xa == 0 { 0 } else { xa + offsets[pa] };
        let gb = if xb == 0 { 0 } else { xb + offsets[pb] };
        let va = out.assignment[ga];
        let vb = out.assignment[gb];
        return format!(
            "[we_dpp] failed constraint {i} is GLUE #{glue_idx}: (part {pa}, var {xa}) == (part {pb}, var {xb})\n\
             merged idxs: {ga} vs {gb}\n\
             values: {va:?} vs {vb:?}"
        );
    } else {
        format!("[we_dpp] failed constraint {i} is after all parts+glue??")
    }
}

criterion_group!(benches, bench_we_dpp);
criterion_main!(benches);

