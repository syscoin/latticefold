#![cfg(feature = "symphony")]

use ark_std::One;
use cyclotomic_rings::rings::FrogPoseidonConfig as PC;
use latticefold_plus::{
    rp_rgchk::RPParams,
    symphony_gr1cs::{
        prove_pi_gr1cs,
        prove_pi_gr1cs_shared_sumcheck,
        verify_pi_gr1cs_and_output,
        verify_pi_gr1cs_output_relations_with_witness,
        verify_pi_gr1cs_shared_sumcheck_output_relations_with_witness,
        verify_pi_gr1cs_shared_sumcheck_and_output,
    },
    transcript::PoseidonTranscript,
};
use stark_rings::{cyclotomic_ring::models::frog_ring::RqPoly as R, PolyRing};
use stark_rings_linalg::{Matrix, SparseMatrix};

#[test]
fn test_pi_gr1cs_prove_verify_identity() {
    // Minimal Π_gr1cs instance:
    // - Hadamard part: M1=M2=M3=I so (M1f)◦(M2f)=(M3f) holds for f=1.
    // - Range part: we commit to f via Ajtai and run Π_rg on the same f.
    let n = 1 << 10;
    let m = 1 << 10;
    let f = vec![R::one(); n];

    let mut m1 = SparseMatrix::<R>::identity(m);
    let mut m2 = SparseMatrix::<R>::identity(m);
    let mut m3 = SparseMatrix::<R>::identity(m);
    m1.pad_cols(n);
    m2.pad_cols(n);
    m3.pad_cols(n);

    let a = Matrix::<R>::rand(&mut ark_std::test_rng(), 2, n);
    let cm_f = a.try_mul_vec(&f).unwrap();

    let rg_params = RPParams {
        l_h: 64,
        lambda_pj: 32,
        k_g: 4,
        d_prime: (R::dimension() as u128) / 2,
    };

    let mut ts_p = PoseidonTranscript::<R>::empty::<PC>();
    let proof = prove_pi_gr1cs(&mut ts_p, [&m1, &m2, &m3], &cm_f, &f, rg_params).unwrap();

    let mut ts_v = PoseidonTranscript::<R>::empty::<PC>();
    let out = verify_pi_gr1cs_and_output(&mut ts_v, [&m1, &m2, &m3], &cm_f, &proof).unwrap();
    assert!(!out.had.r.is_empty());
    assert!(!out.rg_r.is_empty());

    // Also enforce output relations vs explicit witness (correctness-first bridge).
    let mut ts_v2 = PoseidonTranscript::<R>::empty::<PC>();
    verify_pi_gr1cs_output_relations_with_witness(
        &mut ts_v2,
        [&m1, &m2, &m3],
        &cm_f,
        &proof,
        &f,
    )
    .unwrap();
}

#[test]
fn test_pi_gr1cs_shared_sumcheck_prove_verify_identity() {
    let n = 1 << 10;
    let m = 1 << 10;
    let f = vec![R::one(); n];

    let mut m1 = SparseMatrix::<R>::identity(m);
    let mut m2 = SparseMatrix::<R>::identity(m);
    let mut m3 = SparseMatrix::<R>::identity(m);
    m1.pad_cols(n);
    m2.pad_cols(n);
    m3.pad_cols(n);

    let a = Matrix::<R>::rand(&mut ark_std::test_rng(), 2, n);
    let cm_f = a.try_mul_vec(&f).unwrap();

    let rg_params = RPParams {
        l_h: 64,
        lambda_pj: 32,
        k_g: 4,
        d_prime: (R::dimension() as u128) / 2,
    };

    // Shared-sumcheck prover.
    let mut ts_p = PoseidonTranscript::<R>::empty::<PC>();
    let proof = prove_pi_gr1cs_shared_sumcheck(&mut ts_p, [&m1, &m2, &m3], &cm_f, &f, rg_params)
        .unwrap();

    // Shared-sumcheck verifier.
    let mut ts_v = PoseidonTranscript::<R>::empty::<PC>();
    let out = verify_pi_gr1cs_shared_sumcheck_and_output(&mut ts_v, &cm_f, &proof).unwrap();
    assert!(!out.had.r.is_empty());
    assert!(!out.rg_r.is_empty());

    // Also enforce output relations vs explicit witness (correctness-first bridge).
    let mut ts_v2 = PoseidonTranscript::<R>::empty::<PC>();
    verify_pi_gr1cs_shared_sumcheck_output_relations_with_witness(
        &mut ts_v2,
        [&m1, &m2, &m3],
        &cm_f,
        &proof,
        &f,
    )
    .unwrap();
}
