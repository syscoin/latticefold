//! WE-gate + DPP integration bench (research).
//!
//! Current scope:
//! - Build a WE sparse dR1CS for verifying one `ComR1CSProof` (Π_lin)
//! - Convert it into the prototype dpp::dr1cs_flpcp pipeline and run verification
//!
//! This is *not* yet the full LF+ WE gate (CmProof / DecompProof still TODO).

#![allow(non_snake_case)]

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use cyclotomic_rings::rings::GoldilocksPoseidonConfig as PC;
use cyclotomic_rings::rings::GetPoseidonParams;

use ark_ff::{BigInteger, Fp384, MontBackend, MontConfig, PrimeField};
use rand::{rngs::StdRng, SeedableRng};

use latticefold::arith::r1cs::R1CS;
use stark_rings::balanced_decomposition::GadgetDecompose;
use stark_rings::cyclotomic_ring::models::goldilocks::RqPoly as R;
use stark_rings::PolyRing;
use stark_rings_linalg::{Matrix, SparseMatrix};

use latticefold_plus::lin::{Linearize, LinearizedVerify};
use latticefold_plus::recording_transcript::TracePoseidonTranscript;
use latticefold_plus::r1cs::ComR1CS;
use latticefold_plus::we_gate_arith::build_we_dr1cs_for_comr1cs_proof;
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

fn identity_cs(n: usize) -> (R1CS<R>, Vec<R>) {
    let r1cs = R1CS::<R> {
        l: 1,
        A: SparseMatrix::identity(n),
        B: SparseMatrix::identity(n),
        C: SparseMatrix::identity(n),
    };
    let z = vec![<R as stark_rings::Ring>::ONE; n];
    (r1cs, z)
}

fn bench_we_dpp(c: &mut Criterion) {
    // Keep defaults small-ish so local runs work; override on server by editing this file for now.
    let n = 1 << 10;
    let k = 4;
    let m = n / k;
    let b = 2;
    let kappa = 2;

    let (mut r1cs, z) = identity_cs(m);
    r1cs.A = r1cs.A.gadget_decompose(b, k);
    r1cs.B = r1cs.B.gadget_decompose(b, k);
    r1cs.C = r1cs.C.gadget_decompose(b, k);

    let A = Matrix::<R>::rand(&mut ark_std::test_rng(), kappa, n);
    let cr1cs = ComR1CS::new(r1cs, z, 1, b, k, &A);

    // Prover-side linearization proof (Π_lin).
    let mut ts = latticefold_plus::transcript::PoseidonTranscript::empty::<PC>();
    let (_linb, proof) = cr1cs.linearize(&mut ts);

    // Record verifier transcript ops.
    let mut rec = TracePoseidonTranscript::<R>::empty::<PC>();
    assert!(proof.verify(&mut rec));
    let trace = rec.trace().clone();

    // Statement params prefix (placeholder values; we only bind layout in this bench).
    let params = WeParams {
        nvars_setchk: 0,
        degree_setchk: 0,
        nvars_cm: 0,
        degree_cm: 0,
        kappa: kappa as u64,
        ring_dim_d: R::dimension() as u64,
        k: k as u64,
        l: 0,
        mlen: 0,
    };

    let poseidon_cfg = PC::get_poseidon_config();

    let mut group = c.benchmark_group("we_dpp");
    group.sample_size(10);

    group.bench_function(BenchmarkId::new("build_we_dr1cs_pi_lin", n), |bch| {
        bch.iter(|| {
            let out = build_we_dr1cs_for_comr1cs_proof::<R>(&poseidon_cfg, &trace, &params, &proof)
                .expect("build_we_dr1cs_for_comr1cs_proof");
            out.inst.check(&out.assignment).expect("dr1cs satisfied");
        })
    });

    group.bench_function(BenchmarkId::new("dpp_verify_pi_lin", n), |bch| {
        // Build once outside the timed loop.
        let out = build_we_dr1cs_for_comr1cs_proof::<R>(&poseidon_cfg, &trace, &params, &proof)
            .expect("build_we_dr1cs_for_comr1cs_proof");
        out.inst.check(&out.assignment).expect("dr1cs satisfied");

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

criterion_group!(benches, bench_we_dpp);
criterion_main!(benches);

