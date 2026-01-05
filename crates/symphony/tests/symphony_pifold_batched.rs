#![cfg(feature = "symphony")]


use ark_std::One;
use cyclotomic_rings::rings::FrogPoseidonConfig as PC;
use symphony::symphony_coins::derive_beta_chi;
use latticefold::transcript::Transcript;
use symphony::{
    rp_rgchk::RPParams,
    symphony_pifold_batched::{
        prove_pi_fold_batched_sumcheck_fs,
        verify_pi_fold_batched_and_fold_outputs_poseidon_fs,
        verify_pi_fold_batched_and_fold_outputs_with_openings_and_aux,
    },
    symphony_open::AjtaiOpenVerifier,
};
use latticefold::commitment::AjtaiCommitmentScheme;
use stark_rings::{cyclotomic_ring::models::frog_ring::RqPoly as R, PolyRing, Ring};
use stark_rings_linalg::{Matrix, SparseMatrix};

#[test]
fn test_pifold_batched_sumcheck_two_instances() {
    let n = 1 << 10;
    let m = 1 << 10;

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

    let a = Matrix::<R>::rand(&mut ark_std::test_rng(), 2, n);
    let scheme = AjtaiCommitmentScheme::<R>::new(a.clone());
    let open = AjtaiOpenVerifier { scheme: scheme.clone() };

    let f0 = vec![R::one(); n];
    let f1 = vec![R::ZERO; n];
    let cm0 = scheme.commit(&f0).unwrap().as_ref().to_vec();
    let cm1 = scheme.commit(&f1).unwrap().as_ref().to_vec();

    let rg_params = RPParams {
        l_h: 64,
        lambda_pj: 32,
        k_g: 4,
        d_prime: (R::dimension() as u128) / 2,
    };

    let out = prove_pi_fold_batched_sumcheck_fs::<R, PC>(
        [&m1, &m2, &m3],
        &[cm0.clone(), cm1.clone()],
        &[f0.clone(), f1.clone()],
        &[],
        None,
        None,
        rg_params.clone(),
    )
    .unwrap();
    let proof = out.proof;

    // Paper-faithful FS verification: challenges are derived by hashing the transcript (Poseidon).
    let (folded_inst, _folded_bat) =
        symphony::symphony_pifold_batched::verify_pi_fold_batched_and_fold_outputs_poseidon_fs::<R, PC>(
        [&m1, &m2, &m3],
        &[cm0.clone(), cm1.clone()],
        &proof,
            &open,
            &[f0.clone(), f1.clone()],
            &[],
    )
    .unwrap();

    // Recompute β from the FS coin stream.
    let mut ts = symphony::public_coin_transcript::FixedTranscript::<R>::new_with_coins_and_events(
        proof.coins.challenges.clone(),
        proof.coins.bytes.clone(),
        proof.coins.events.clone(),
    );
    ts.absorb_field_element(&<R as stark_rings::PolyRing>::BaseRing::from(0x4c465053_50494250u128)); // "LFPS_PIBP"
    let beta_cts = derive_beta_chi::<R>(&mut ts, 2);
    let beta0 = R::from(beta_cts[0]);
    let beta1 = R::from(beta_cts[1]);

    // Commitment fold must match manual.
    let mut c = vec![R::ZERO; cm0.len()];
    for i in 0..c.len() {
        c[i] = beta0 * cm0[i] + beta1 * cm1[i];
    }
    assert_eq!(folded_inst.c, c);
}

#[test]
fn test_pifold_batched_aux_witness_path_matches() {
    let n = 1 << 10;
    let m = 1 << 10;

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

    let a = Matrix::<R>::rand(&mut ark_std::test_rng(), 2, n);
    let scheme = AjtaiCommitmentScheme::<R>::new(a.clone());
    let open = AjtaiOpenVerifier { scheme: scheme.clone() };

    let f0 = vec![R::one(); n];
    let f1 = vec![R::ZERO; n];
    let cm0 = scheme.commit(&f0).unwrap().as_ref().to_vec();
    let cm1 = scheme.commit(&f1).unwrap().as_ref().to_vec();

    let rg_params = RPParams {
        l_h: 64,
        lambda_pj: 32,
        k_g: 4,
        d_prime: (R::dimension() as u128) / 2,
    };

    let out = prove_pi_fold_batched_sumcheck_fs::<R, PC>(
        [&m1, &m2, &m3],
        &[cm0.clone(), cm1.clone()],
        &[f0.clone(), f1.clone()],
        &[],
        None,
        None,
        rg_params,
    )
    .unwrap();
    let proof = out.proof;

    let out_ref = verify_pi_fold_batched_and_fold_outputs_poseidon_fs::<R, PC>(
        [&m1, &m2, &m3],
        &[cm0.clone(), cm1.clone()],
        &proof,
        &open,
        &[f0.clone(), f1.clone()],
        &[],
    )
    .unwrap();

    let aux = out.aux;

    let mut ts = symphony::transcript::PoseidonTranscript::<R>::empty::<PC>();
    ts.absorb_field_element(&<R as stark_rings::PolyRing>::BaseRing::from(0x4c465053_50494250u128)); // "LFPS_PIBP"
    // Match the library Poseidon-FS transcript prefix (public statement binding, empty here).
    ts.absorb_field_element(&<R as stark_rings::PolyRing>::BaseRing::from(0x4c465053_5055424cu128)); // "LFPS_PUBL"
    ts.absorb_field_element(&<R as stark_rings::PolyRing>::BaseRing::from(0u128));
    let out_aux = verify_pi_fold_batched_and_fold_outputs_with_openings_and_aux(
        &mut ts,
        [&m1, &m2, &m3],
        &[cm0, cm1],
        &proof,
        &open,
        &[f0, f1],
        Some(&aux),
    )
    .unwrap();

    assert_eq!(out_aux.0.c, out_ref.0.c);
    assert_eq!(out_aux.0.r, out_ref.0.r);
    assert_eq!(out_aux.0.v, out_ref.0.v);
    assert_eq!(out_aux.1.u, out_ref.1.u);
    assert_eq!(out_aux.1.r_prime, out_ref.1.r_prime);
}

#[test]
fn test_pifold_batched_aux_witness_rejects_tamper() {
    let n = 1 << 10;
    let m = 1 << 10;

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

    let a = Matrix::<R>::rand(&mut ark_std::test_rng(), 2, n);
    let scheme = AjtaiCommitmentScheme::<R>::new(a.clone());
    let open = AjtaiOpenVerifier { scheme: scheme.clone() };

    let f0 = vec![R::one(); n];
    let f1 = vec![R::ZERO; n];
    let cm0 = scheme.commit(&f0).unwrap().as_ref().to_vec();
    let cm1 = scheme.commit(&f1).unwrap().as_ref().to_vec();

    let rg_params = RPParams {
        l_h: 64,
        lambda_pj: 32,
        k_g: 4,
        d_prime: (R::dimension() as u128) / 2,
    };

    let out = prove_pi_fold_batched_sumcheck_fs::<R, PC>(
        [&m1, &m2, &m3],
        &[cm0.clone(), cm1.clone()],
        &[f0.clone(), f1.clone()],
        &[],
        None,
        None,
        rg_params,
    )
    .unwrap();
    let proof = out.proof;

    let mut aux = out.aux;

    // Tamper a single had_u coordinate that actually affects Eq.(26).
    // In this test config we have M2 = 0 and M3 = 0, so changing U1 alone would not change
    // u1*u2-u3. We therefore tamper U2.
    aux.had_u[0][1][0] += <R as stark_rings::PolyRing>::BaseRing::ONE;

    let mut ts = symphony::transcript::PoseidonTranscript::<R>::empty::<PC>();
    ts.absorb_field_element(&<R as stark_rings::PolyRing>::BaseRing::from(0x4c465053_50494250u128)); // "LFPS_PIBP"
    // Match the library Poseidon-FS transcript prefix (public statement binding, empty here).
    ts.absorb_field_element(&<R as stark_rings::PolyRing>::BaseRing::from(0x4c465053_5055424cu128)); // "LFPS_PUBL"
    ts.absorb_field_element(&<R as stark_rings::PolyRing>::BaseRing::from(0u128));

    let err = verify_pi_fold_batched_and_fold_outputs_with_openings_and_aux(
        &mut ts,
        [&m1, &m2, &m3],
        &[cm0, cm1],
        &proof,
        &open,
        &[f0, f1],
        Some(&aux),
    )
    .unwrap_err();

    assert!(
        err.contains("Eq(26)") || err.contains("mismatch") || err.contains("Step5"),
        "unexpected error: {err}"
    );
}

#[test]
fn test_pifold_batched_public_statement_binding_rejects_mismatch() {
    let n = 1 << 10;
    let m = 1 << 10;

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

    let a = Matrix::<R>::rand(&mut ark_std::test_rng(), 2, n);
    let scheme = AjtaiCommitmentScheme::<R>::new(a.clone());
    let open = AjtaiOpenVerifier { scheme: scheme.clone() };

    let f0 = vec![R::one(); n];
    let f1 = vec![R::ZERO; n];
    let cm0 = scheme.commit(&f0).unwrap().as_ref().to_vec();
    let cm1 = scheme.commit(&f1).unwrap().as_ref().to_vec();

    let rg_params = RPParams {
        l_h: 64,
        lambda_pj: 32,
        k_g: 4,
        d_prime: (R::dimension() as u128) / 2,
    };

    // "SP1 statement" placeholder: two base-field elements, like (vkey_hash, claim_digest).
    let stmt_ok = vec![<R as stark_rings::PolyRing>::BaseRing::from(123u128), <R as stark_rings::PolyRing>::BaseRing::from(456u128)];
    let stmt_bad = vec![<R as stark_rings::PolyRing>::BaseRing::from(123u128), <R as stark_rings::PolyRing>::BaseRing::from(999u128)];

    let out = prove_pi_fold_batched_sumcheck_fs::<R, PC>(
        [&m1, &m2, &m3],
        &[cm0.clone(), cm1.clone()],
        &[f0.clone(), f1.clone()],
        &stmt_ok,
        None,
        None,
        rg_params,
    )
    .unwrap();
    let proof = out.proof;

    // Verifying with the correct statement succeeds.
    let _ = verify_pi_fold_batched_and_fold_outputs_poseidon_fs::<R, PC>(
        [&m1, &m2, &m3],
        &[cm0.clone(), cm1.clone()],
        &proof,
        &open,
        &[f0.clone(), f1.clone()],
        &stmt_ok,
    )
    .unwrap();

    // Verifying with a different statement must fail (different FS challenges).
    let err = verify_pi_fold_batched_and_fold_outputs_poseidon_fs::<R, PC>(
        [&m1, &m2, &m3],
        &[cm0, cm1],
        &proof,
        &open,
        &[f0, f1],
        &stmt_bad,
    )
    .unwrap_err();
    assert!(err.contains("PiFold") || err.contains("sumcheck") || err.contains("mismatch"), "unexpected error: {err}");
}
