use cyclotomic_rings::rings::FrogPoseidonConfig as PC;
use latticefold::commitment::AjtaiCommitmentScheme;
use stark_rings::{cyclotomic_ring::models::frog_ring::RqPoly as R, PolyRing, Ring};
use stark_rings_linalg::SparseMatrix;
use symphony::rp_rgchk::RPParams;
use symphony::symphony_open::AjtaiOpenVerifier;
use symphony::symphony_open::MultiAjtaiOpenVerifier;
use symphony::symphony_pifold_batched::{
    verify_pi_fold_batched_and_fold_outputs_poseidon_fs_cp_hetero_m,
    verify_pi_fold_batched_and_fold_outputs_poseidon_fs_cp_hetero_m_with_metrics,
    verify_pi_fold_batched_and_fold_outputs_poseidon_fs_hetero_m,
};
use symphony::poseidon_trace::replay_poseidon_transcript_trace;
use symphony::symphony_pifold_streaming::{prove_pi_fold_poseidon_fs, PiFoldStreamingConfig};

const MASTER_SEED: [u8; 32] = *b"SYMPHONY_AJTAI_SEED_V1_000000000";

#[test]
fn test_pifold_streaming_hetero_m_roundtrip() {
    let n = 1 << 4; // 16 vars
    let m = 1 << 3; // 8 rows per chunk/instance

    // Two different A-matrices; B=C=0 so A*f ∘ B*f - C*f == 0 holds for any witness.
    let mut a0 = SparseMatrix::<R>::identity(m);
    let mut b0 = SparseMatrix::<R>::identity(m);
    let mut c0 = SparseMatrix::<R>::identity(m);
    a0.pad_cols(n);
    b0.pad_cols(n);
    c0.pad_cols(n);
    for row in b0.coeffs.iter_mut() {
        row.clear();
    }
    for row in c0.coeffs.iter_mut() {
        row.clear();
    }

    let mut a1 = SparseMatrix::<R>::identity(m);
    let mut b1 = SparseMatrix::<R>::identity(m);
    let mut c1 = SparseMatrix::<R>::identity(m);
    a1.pad_cols(n);
    b1.pad_cols(n);
    c1.pad_cols(n);
    // Make a1 different from a0 by swapping two rows (still sparse and well-formed).
    a1.coeffs.swap(0, 1);
    for row in b1.coeffs.iter_mut() {
        row.clear();
    }
    for row in c1.coeffs.iter_mut() {
        row.clear();
    }

    let ms: Vec<[std::sync::Arc<SparseMatrix<R>>; 3]> = vec![
        [
            std::sync::Arc::new(a0.clone()),
            std::sync::Arc::new(b0.clone()),
            std::sync::Arc::new(c0.clone()),
        ],
        [
            std::sync::Arc::new(a1.clone()),
            std::sync::Arc::new(b1.clone()),
            std::sync::Arc::new(c1.clone()),
        ],
    ];

    let rg_params = RPParams {
        l_h: 4,
        lambda_pj: 1,
        k_g: 2,
        d_prime: (R::dimension() as u128) - 2,
    };

    let scheme = AjtaiCommitmentScheme::<R>::seeded(b"cm_f", MASTER_SEED, 2, n);
    let open = AjtaiOpenVerifier { scheme: scheme.clone() };

    let f0 = vec![R::ONE; n];
    let f1 = (0..n).map(|i| if i % 2 == 0 { R::ONE } else { R::ZERO }).collect::<Vec<_>>();
    let cm0 = scheme.commit_const_coeff_fast(&f0).unwrap().as_ref().to_vec();
    let cm1 = scheme.commit_const_coeff_fast(&f1).unwrap().as_ref().to_vec();

    let cms = vec![cm0.clone(), cm1.clone()];
    let witnesses = vec![std::sync::Arc::new(f0.clone()), std::sync::Arc::new(f1.clone())];
    let public_inputs: Vec<<R as PolyRing>::BaseRing> = vec![];

    // CP transcript-message commitment schemes (WE/DPP-facing path).
    let scheme_had = AjtaiCommitmentScheme::<R>::seeded(
        b"cfs_had_u",
        MASTER_SEED,
        2,
        3 * R::dimension(),
    );
    let scheme_mon = AjtaiCommitmentScheme::<R>::seeded(b"cfs_mon_b", MASTER_SEED, 2, rg_params.k_g);

    let cfg = PiFoldStreamingConfig::default();
    let out = prove_pi_fold_poseidon_fs::<R, PC>(
        ms.as_slice(),
        &cms,
        &witnesses,
        &public_inputs,
        Some(&scheme_had),
        Some(&scheme_mon),
        rg_params.clone(),
        &cfg,
    )
    .expect("prove failed");

    // Verify using openings (correctness-first path).
    let ms_refs: Vec<[&SparseMatrix<R>; 3]> = vec![
        [&a0, &b0, &c0],
        [&a1, &b1, &c1],
    ];
    let (_folded_inst, _folded_bat) = verify_pi_fold_batched_and_fold_outputs_poseidon_fs_hetero_m::<R, PC>(
        ms_refs.as_slice(),
        &cms,
        &out.proof,
        &open,
        &[f0, f1],
        None,
        &public_inputs,
    )
    .expect("verify failed");

    // WE/CP-facing verification: verify using CP transcript-message commitments + openings to aux.
    let open_cfs = MultiAjtaiOpenVerifier::new()
        .with_scheme("cfs_had_u", scheme_had)
        .with_scheme("cfs_mon_b", scheme_mon);
    let _ = verify_pi_fold_batched_and_fold_outputs_poseidon_fs_cp_hetero_m::<R, PC>(
        ms_refs.as_slice(),
        &cms,
        &out.proof,
        &open_cfs,
        &out.cfs_had_u,
        &out.cfs_mon_b,
        &out.aux,
        &public_inputs,
    )
    .expect("cp verify failed");

    // Trace harness: ensure trace lengths match metrics counters.
    let (_out2, metrics, trace) = verify_pi_fold_batched_and_fold_outputs_poseidon_fs_cp_hetero_m_with_metrics::<R, PC>(
        ms_refs.as_slice(),
        &cms,
        &out.proof,
        &open_cfs,
        &out.cfs_had_u,
        &out.cfs_mon_b,
        &out.aux,
        &public_inputs,
    )
    .expect("cp verify (metrics) failed");

    assert_eq!(metrics.absorbed_elems as usize, trace.absorbed.len());
    assert_eq!(metrics.squeezed_field_elems as usize, trace.squeezed_field.len());
    assert_eq!(metrics.squeezed_bytes as usize, trace.squeezed_bytes.len());

    // Replay the trace to ensure it is a complete Poseidon transcript witness.
    let cfg = <PC as cyclotomic_rings::rings::GetPoseidonParams<<<R as PolyRing>::BaseRing as ark_ff::Field>::BasePrimeField>>::get_poseidon_config();
    let replay = replay_poseidon_transcript_trace(&cfg, &trace).expect("poseidon trace replay failed");
    // Sanity: permutation count should match simple upper-bound estimate order of magnitude.
    assert!(replay.permutes.len() > 10);
}

