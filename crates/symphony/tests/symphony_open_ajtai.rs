

use cyclotomic_rings::rings::FrogPoseidonConfig as PC;
use symphony::{
    rp_rgchk::RPParams,
    symphony_open::AjtaiOpenVerifier,
    symphony_pifold_batched::verify_pi_fold_batched_and_fold_outputs_with_openings,
    symphony_pifold_streaming::{prove_pi_fold_poseidon_fs, PiFoldStreamingConfig},
};
use latticefold::commitment::AjtaiCommitmentScheme;
use latticefold::transcript::Transcript;
use symphony::public_coin_transcript::FixedTranscript;
use stark_rings::{cyclotomic_ring::models::frog_ring::RqPoly as R, PolyRing, Ring};
use stark_rings_linalg::SparseMatrix;

#[test]
fn test_ajtai_openings_for_cms_are_verified() {
    let n = 1 << 10;
    let m = 1 << 10;

    // Matrices so Π_had holds for any witness: M1=I, M2=0, M3=0.
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

    // Ajtai matrix used for the witness commitments.
    const MASTER_SEED: [u8; 32] = *b"SYMPHONY_AJTAI_SEED_V1_000000000";
    let scheme = AjtaiCommitmentScheme::<R>::seeded(b"cm_f", MASTER_SEED, 2, n);
    let open = AjtaiOpenVerifier { scheme: scheme.clone() };

    // Two witnesses and their commitments.
    let f0 = vec![R::ONE; n];
    let f1 = vec![R::ZERO; n];
    let cm0 = scheme.commit(&f0).unwrap().as_ref().to_vec();
    let cm1 = scheme.commit(&f1).unwrap().as_ref().to_vec();

    let rg_params = RPParams {
        l_h: 64,
        lambda_pj: 32,
        k_g: 4,
        d_prime: (R::dimension() as u128) / 2,
    };

    // Prove (Poseidon-FS).
    let cfg = PiFoldStreamingConfig::default();
    let ms = vec![
        [
            std::sync::Arc::new(m1.clone()),
            std::sync::Arc::new(m2.clone()),
            std::sync::Arc::new(m3.clone()),
        ],
        [
            std::sync::Arc::new(m1.clone()),
            std::sync::Arc::new(m2.clone()),
            std::sync::Arc::new(m3.clone()),
        ],
    ];
    let out = prove_pi_fold_poseidon_fs::<R, PC>(
        ms.as_slice(),
        &[cm0.clone(), cm1.clone()],
        &[std::sync::Arc::new(f0.clone()), std::sync::Arc::new(f1.clone())],
        &[],
        None,
        None,
        rg_params,
        &cfg,
    )
    .unwrap();
    let proof = out.proof;

    // Verify using explicit coin replay transcript.
    let mut ts = FixedTranscript::<R>::new_with_coins_and_events(
        proof.coins.challenges.clone(),
        proof.coins.bytes.clone(),
        proof.coins.events.clone(),
    );
    ts.absorb_field_element(&<R as PolyRing>::BaseRing::from(0x4c465053_50494250u128)); // "LFPS_PIBP"

    let _ = verify_pi_fold_batched_and_fold_outputs_with_openings(
        &mut ts,
        [&m1, &m2, &m3],
        &[cm0, cm1],
        &proof,
        &open,
        &[f0, f1],
    )
    .unwrap();
}
