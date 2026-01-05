#![cfg(feature = "symphony")]

use ark_std::One;
use cyclotomic_rings::rings::FrogPoseidonConfig as PC;
use latticefold_plus::{symphony_had::*, transcript::PoseidonTranscript};
use stark_rings::cyclotomic_ring::models::frog_ring::RqPoly as R;
use stark_rings_linalg::SparseMatrix;

#[test]
fn test_pi_had_identity_holds() {
    // M1=M2=M3=I, f=1 => (1*1-1)=0 per row and per coefficient.
    let n = 1 << 8;
    let m = n;
    let f = vec![R::one(); n];

    let mut m1 = SparseMatrix::<R>::identity(m);
    let mut m2 = SparseMatrix::<R>::identity(m);
    let mut m3 = SparseMatrix::<R>::identity(m);
    m1.pad_cols(n);
    m2.pad_cols(n);
    m3.pad_cols(n);

    let mut ts_p = PoseidonTranscript::<R>::empty::<PC>();
    let proof = prove_pi_had(&mut ts_p, [&m1, &m2, &m3], &f).unwrap();

    let mut ts_v = PoseidonTranscript::<R>::empty::<PC>();
    verify_pi_had(&mut ts_v, &proof).unwrap();

    // Also enforce the paper output relation (Eq. (24)) with the explicit witness.
    let mut ts_v2 = PoseidonTranscript::<R>::empty::<PC>();
    verify_pi_had_output_relation_with_witness(&mut ts_v2, [&m1, &m2, &m3], &proof, &f).unwrap();
}

#[test]
fn test_pi_had_rejects_bad_relation() {
    // M1=M2=I, M3=0, f=1 => 1*1-0 != 0 -> should fail.
    let n = 1 << 8;
    let m = n;
    let f = vec![R::one(); n];

    let mut m1 = SparseMatrix::<R>::identity(m);
    let mut m2 = SparseMatrix::<R>::identity(m);
    let mut m3 = SparseMatrix::<R>::identity(m);
    m1.pad_cols(n);
    m2.pad_cols(n);
    m3.pad_cols(n);
    // make it the zero matrix
    m3.coeffs = vec![vec![]; m];

    let mut ts_p = PoseidonTranscript::<R>::empty::<PC>();
    let proof = prove_pi_had(&mut ts_p, [&m1, &m2, &m3], &f).unwrap();

    let mut ts_v = PoseidonTranscript::<R>::empty::<PC>();
    assert!(verify_pi_had(&mut ts_v, &proof).is_err());
}

#[test]
fn test_pi_had_output_relation_rejects_wrong_witness() {
    // Prove with f, then verify the output relation against a different f'.
    let n = 1 << 8;
    let m = n;
    let f = vec![R::one(); n];

    let mut m1 = SparseMatrix::<R>::identity(m);
    let mut m2 = SparseMatrix::<R>::identity(m);
    let mut m3 = SparseMatrix::<R>::identity(m);
    m1.pad_cols(n);
    m2.pad_cols(n);
    m3.pad_cols(n);

    let mut ts_p = PoseidonTranscript::<R>::empty::<PC>();
    let proof = prove_pi_had(&mut ts_p, [&m1, &m2, &m3], &f).unwrap();

    let mut f_bad = f.clone();
    f_bad[0] += R::one();

    let mut ts_v = PoseidonTranscript::<R>::empty::<PC>();
    assert!(verify_pi_had_output_relation_with_witness(&mut ts_v, [&m1, &m2, &m3], &proof, &f_bad).is_err());
}

