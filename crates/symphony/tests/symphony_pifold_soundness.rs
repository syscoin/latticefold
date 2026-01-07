//! Soundness tests for Π_fold verifier.
//!
//! These tests verify that the verifier correctly rejects:
//! - Tampered auxiliary witness data
//! - Mismatched public statements

use ark_std::One;
use cyclotomic_rings::rings::FrogPoseidonConfig as PC;
use std::sync::Arc;
use latticefold::transcript::Transcript;
use symphony::{
    rp_rgchk::RPParams,
    symphony_pifold_batched::{
        verify_pi_fold_batched_and_fold_outputs_poseidon_fs_hetero_m,
        verify_pi_fold_batched_and_fold_outputs_with_openings_and_aux_hetero_m,
    },
    symphony_pifold_streaming::{prove_pi_fold_poseidon_fs, PiFoldStreamingConfig},
    symphony_open::AjtaiOpenVerifier,
};
use latticefold::commitment::AjtaiCommitmentScheme;
use stark_rings::{cyclotomic_ring::models::frog_ring::RqPoly as R, PolyRing, Ring};
use stark_rings_linalg::SparseMatrix;

const MASTER_SEED: [u8; 32] = *b"SYMPHONY_AJTAI_SEED_V1_000000000";

fn setup_test_matrices(n: usize, m: usize) -> (Arc<SparseMatrix<R>>, Arc<SparseMatrix<R>>, Arc<SparseMatrix<R>>) {
    // Choose matrices so Π_had holds for any witness: M1=I, M2=0, M3=0.
    let mut m1 = SparseMatrix::<R>::identity(m);
    let mut m2 = SparseMatrix::<R>::identity(m);
    let mut m3 = SparseMatrix::<R>::identity(m);
    m1.pad_cols(n);
    m2.pad_cols(n);
    m3.pad_cols(n);
    for row in m2.coeffs.iter_mut() {
        row.clear();
    }
    for row in m3.coeffs.iter_mut() {
        row.clear();
    }
    (Arc::new(m1), Arc::new(m2), Arc::new(m3))
}

/// Test that aux witness path produces consistent output with standard verifier.
#[test]
fn test_aux_witness_path_matches() {
    let n = 1 << 10;
    let m = 1 << 10;

    let (m1, m2, m3) = setup_test_matrices(n, m);

    let scheme = AjtaiCommitmentScheme::<R>::seeded(b"cm_f", MASTER_SEED, 2, n);
    let open = AjtaiOpenVerifier { scheme: scheme.clone() };

    let f0 = Arc::new(vec![R::one(); n]);
    let f1 = Arc::new(vec![R::ZERO; n]);
    let cm0 = scheme.commit(f0.as_ref()).unwrap().as_ref().to_vec();
    let cm1 = scheme.commit(f1.as_ref()).unwrap().as_ref().to_vec();

    let rg_params = RPParams {
        l_h: 64,
        lambda_pj: 32,
        k_g: 4,
        d_prime: (R::dimension() as u128) / 2,
    };

    let ms = vec![
        [m1.clone(), m2.clone(), m3.clone()],
        [m1.clone(), m2.clone(), m3.clone()],
    ];

    let cfg = PiFoldStreamingConfig::default();
    let out = prove_pi_fold_poseidon_fs::<R, PC>(
        ms.as_slice(),
        &[cm0.clone(), cm1.clone()],
        &[f0.clone(), f1.clone()],
        &[],
        None,
        None,
        rg_params,
        &cfg,
    )
    .unwrap();
    let proof = out.proof;

    let ms_refs: Vec<[&SparseMatrix<R>; 3]> = vec![
        [&m1, &m2, &m3],
        [&m1, &m2, &m3],
    ];
    let out_ref = verify_pi_fold_batched_and_fold_outputs_poseidon_fs_hetero_m::<R, PC>(
        ms_refs.as_slice(),
        &[cm0.clone(), cm1.clone()],
        &proof,
        &open,
        &[f0.as_ref().clone(), f1.as_ref().clone()],
        None,
        &[],
    )
    .unwrap();

    let aux = out.aux;

    let mut ts = symphony::transcript::PoseidonTranscript::<R>::empty::<PC>();
    ts.absorb_field_element(&<R as PolyRing>::BaseRing::from(0x4c465053_50494250u128)); // "LFPS_PIBP"
    ts.absorb_field_element(&<R as PolyRing>::BaseRing::from(0x4c465053_5055424cu128)); // "LFPS_PUBL"
    ts.absorb_field_element(&<R as PolyRing>::BaseRing::from(0u128));
    let out_aux = verify_pi_fold_batched_and_fold_outputs_with_openings_and_aux_hetero_m(
        &mut ts,
        ms_refs.as_slice(),
        &[cm0, cm1],
        &proof,
        &open,
        &[f0.as_ref().clone(), f1.as_ref().clone()],
        Some(&aux),
    )
    .unwrap();

    assert_eq!(out_aux.0.c, out_ref.0.c);
    assert_eq!(out_aux.0.r, out_ref.0.r);
    assert_eq!(out_aux.0.v, out_ref.0.v);
    assert_eq!(out_aux.1.u, out_ref.1.u);
    assert_eq!(out_aux.1.r_prime, out_ref.1.r_prime);
}

/// Test that tampered aux witness is rejected by verifier.
#[test]
fn test_tampered_aux_witness_rejected() {
    let n = 1 << 10;
    let m = 1 << 10;

    let (m1, m2, m3) = setup_test_matrices(n, m);

    let scheme = AjtaiCommitmentScheme::<R>::seeded(b"cm_f", MASTER_SEED, 2, n);
    let open = AjtaiOpenVerifier { scheme: scheme.clone() };

    let f0 = Arc::new(vec![R::one(); n]);
    let f1 = Arc::new(vec![R::ZERO; n]);
    let cm0 = scheme.commit(f0.as_ref()).unwrap().as_ref().to_vec();
    let cm1 = scheme.commit(f1.as_ref()).unwrap().as_ref().to_vec();

    let rg_params = RPParams {
        l_h: 64,
        lambda_pj: 32,
        k_g: 4,
        d_prime: (R::dimension() as u128) / 2,
    };

    let ms = vec![
        [m1.clone(), m2.clone(), m3.clone()],
        [m1.clone(), m2.clone(), m3.clone()],
    ];

    let cfg = PiFoldStreamingConfig::default();
    let out = prove_pi_fold_poseidon_fs::<R, PC>(
        ms.as_slice(),
        &[cm0.clone(), cm1.clone()],
        &[f0.clone(), f1.clone()],
        &[],
        None,
        None,
        rg_params,
        &cfg,
    )
    .unwrap();
    let proof = out.proof;

    let mut aux = out.aux;

    // Tamper a single had_u coordinate that actually affects Eq.(26).
    // In this test config we have M2 = 0 and M3 = 0, so changing U1 alone would not change
    // u1*u2-u3. We therefore tamper U2.
    aux.had_u[0][1][0] += <R as PolyRing>::BaseRing::ONE;

    let ms_refs: Vec<[&SparseMatrix<R>; 3]> = vec![
        [&m1, &m2, &m3],
        [&m1, &m2, &m3],
    ];

    let mut ts = symphony::transcript::PoseidonTranscript::<R>::empty::<PC>();
    ts.absorb_field_element(&<R as PolyRing>::BaseRing::from(0x4c465053_50494250u128)); // "LFPS_PIBP"
    ts.absorb_field_element(&<R as PolyRing>::BaseRing::from(0x4c465053_5055424cu128)); // "LFPS_PUBL"
    ts.absorb_field_element(&<R as PolyRing>::BaseRing::from(0u128));

    let err = verify_pi_fold_batched_and_fold_outputs_with_openings_and_aux_hetero_m(
        &mut ts,
        ms_refs.as_slice(),
        &[cm0, cm1],
        &proof,
        &open,
        &[f0.as_ref().clone(), f1.as_ref().clone()],
        Some(&aux),
    )
    .unwrap_err();

    assert!(
        err.contains("Eq(26)") || err.contains("mismatch") || err.contains("Step5"),
        "unexpected error: {err}"
    );
}

/// Test that wrong public statement is rejected by verifier.
#[test]
fn test_wrong_public_statement_rejected() {
    let n = 1 << 10;
    let m = 1 << 10;

    let (m1, m2, m3) = setup_test_matrices(n, m);

    let scheme = AjtaiCommitmentScheme::<R>::seeded(b"cm_f", MASTER_SEED, 2, n);
    let open = AjtaiOpenVerifier { scheme: scheme.clone() };

    let f0 = Arc::new(vec![R::one(); n]);
    let f1 = Arc::new(vec![R::ZERO; n]);
    let cm0 = scheme.commit(f0.as_ref()).unwrap().as_ref().to_vec();
    let cm1 = scheme.commit(f1.as_ref()).unwrap().as_ref().to_vec();

    let rg_params = RPParams {
        l_h: 64,
        lambda_pj: 32,
        k_g: 4,
        d_prime: (R::dimension() as u128) / 2,
    };

    // "SP1 statement" placeholder: two base-field elements, like (vkey_hash, claim_digest).
    let stmt_ok = vec![<R as PolyRing>::BaseRing::from(123u128), <R as PolyRing>::BaseRing::from(456u128)];
    let stmt_bad = vec![<R as PolyRing>::BaseRing::from(123u128), <R as PolyRing>::BaseRing::from(999u128)];

    let ms = vec![
        [m1.clone(), m2.clone(), m3.clone()],
        [m1.clone(), m2.clone(), m3.clone()],
    ];

    let cfg = PiFoldStreamingConfig::default();
    let out = prove_pi_fold_poseidon_fs::<R, PC>(
        ms.as_slice(),
        &[cm0.clone(), cm1.clone()],
        &[f0.clone(), f1.clone()],
        &stmt_ok,
        None,
        None,
        rg_params,
        &cfg,
    )
    .unwrap();
    let proof = out.proof;

    let ms_refs: Vec<[&SparseMatrix<R>; 3]> = vec![
        [&m1, &m2, &m3],
        [&m1, &m2, &m3],
    ];

    // Verifying with the correct statement succeeds.
    let _ = verify_pi_fold_batched_and_fold_outputs_poseidon_fs_hetero_m::<R, PC>(
        ms_refs.as_slice(),
        &[cm0.clone(), cm1.clone()],
        &proof,
        &open,
        &[f0.as_ref().clone(), f1.as_ref().clone()],
        None,
        &stmt_ok,
    )
    .unwrap();

    // Verifying with a different statement must fail (different FS challenges).
    let err = verify_pi_fold_batched_and_fold_outputs_poseidon_fs_hetero_m::<R, PC>(
        ms_refs.as_slice(),
        &[cm0, cm1],
        &proof,
        &open,
        &[f0.as_ref().clone(), f1.as_ref().clone()],
        None,
        &stmt_bad,
    )
    .unwrap_err();
    assert!(err.contains("PiFold") || err.contains("sumcheck") || err.contains("mismatch"), "unexpected error: {err}");
}
