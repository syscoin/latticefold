#![cfg(feature = "symphony")]

use ark_std::One;
use cyclotomic_rings::rings::FrogPoseidonConfig as PC;
use symphony::{
    public_coin_transcript::FixedTranscript,
    recording_transcript::RecordingTranscriptRef,
    rp_rgchk::{verify_pi_rg_and_output, RPParams, RPRangeProver},
    symphony_fold::{fold_batchlin, fold_instances, pi_rg_to_fold_shapes, SymphonyBatchLin, SymphonyInstance},
    transcript::PoseidonTranscript,
};
use stark_rings::{cyclotomic_ring::models::frog_ring::RqPoly as R, PolyRing, Ring};
use stark_rings_linalg::Matrix;

#[test]
fn test_linear_folding_matches_manual_combination() {
    // Create two Π_rg outputs and fold them linearly with a chosen beta,
    // then compare to a direct manual combination.
    let n = 1 << 10;
    let params = RPParams {
        l_h: 64,
        lambda_pj: 32,
        k_g: 4,
        d_prime: (R::dimension() as u128) / 2,
    };

    let a = Matrix::<R>::rand(&mut ark_std::test_rng(), 2, n);

    // Derive a single shared (public-coin) schedule of challenges/bytes/events from a real transcript,
    // then replay that same coin stream for all instances. This matches Figure 4's "shared randomness".
    let (coins_chals, coins_bytes, coins_events) = {
        let f = vec![R::one(); n];
        let cm_f = a.try_mul_vec(&f).unwrap();
        let prover = RPRangeProver::<R>::new(f, params.clone());
        let mut ts = PoseidonTranscript::empty::<PC>();
        let mut rts = RecordingTranscriptRef::<R, _>::new(&mut ts);
        let _proof = prover.prove(&mut rts, &cm_f);
        (rts.coins_challenges, rts.coins_bytes, rts.events)
    };

    let mk = |seed: u128| -> (Vec<R>, SymphonyInstance<R>, SymphonyBatchLin<R>) {
        let f = vec![R::from(seed); n];
        let cm_f = a.try_mul_vec(&f).unwrap();

        let prover = RPRangeProver::<R>::new(f, params.clone());
        let mut ts_p = FixedTranscript::<R>::new_with_coins_and_events(
            coins_chals.clone(),
            coins_bytes.clone(),
            coins_events.clone(),
        );
        let proof = prover.prove(&mut ts_p, &cm_f);

        let mut ts_v = FixedTranscript::<R>::new_with_coins_and_events(
            coins_chals.clone(),
            coins_bytes.clone(),
            coins_events.clone(),
        );
        let out = verify_pi_rg_and_output(&proof, &cm_f, &mut ts_v).unwrap();
        let (inst, bat) = pi_rg_to_fold_shapes(cm_f.clone(), &out);
        (cm_f, inst, bat)
    };

    let (_cm0, inst0, bat0) = mk(7);
    let (_cm1, inst1, bat1) = mk(9);

    // In Symphony, β is sampled from a low-norm subset S ⊆ R_q.
    // For this unit test we just use small *constant* ring elements.
    let beta = vec![R::from(3u128), R::from(5u128)];

    let folded_inst = fold_instances(&beta, &[inst0.clone(), inst1.clone()]);
    let folded_bat = fold_batchlin(&beta, &[bat0.clone(), bat1.clone()]);

    // Manual: (c,r,v) fold.
    let b0_r = beta[0];
    let b1_r = beta[1];

    let mut c = vec![R::ZERO; inst0.c.len()];
    for i in 0..c.len() {
        c[i] = b0_r * inst0.c[i] + b1_r * inst1.c[i];
    }
    let r = inst0.r.clone(); // shared randomness (must match)
    let v = b0_r * inst0.v + b1_r * inst1.v;

    assert_eq!(folded_inst, SymphonyInstance { c, r, v });

    // Manual: batchlin fold.
    assert_eq!(folded_bat.r_prime, bat0.r_prime);
    let mut u = vec![R::ZERO; bat0.u.len()];
    for i in 0..u.len() {
        u[i] = b0_r * bat0.u[i] + b1_r * bat1.u[i];
    }
    assert_eq!(folded_bat, SymphonyBatchLin { r_prime: bat0.r_prime, u });
}

