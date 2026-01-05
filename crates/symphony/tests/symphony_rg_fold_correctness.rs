#![cfg(feature = "symphony")]

use ark_std::One;
use cyclotomic_rings::rings::FrogPoseidonConfig as PC;
use symphony::{
    public_coin_transcript::FixedTranscript,
    recording_transcript::RecordingTranscriptRef,
    rp_rgchk::{
        compute_auxj_lin_v_from_witness, verify_pi_rg_and_output,
        verify_pi_rg_output_relation_with_witness, RPParams, RPRangeProver,
    },
    symphony_fold::fold_instances,
    transcript::PoseidonTranscript,
};
use stark_rings::{cyclotomic_ring::models::frog_ring::RqPoly as R, PolyRing, Ring};
use stark_rings_linalg::Matrix;

#[test]
fn test_pi_rg_folding_preserves_auxj_lin_relation_with_explicit_witness() {
    // This is a **correctness-first** folding test:
    // - We run two Π_rg instances under shared public-coin randomness (same J, same r).
    // - We fold their public outputs with low-norm β ∈ R_q.
    // - We check that the folded output `v*` matches direct recomputation from the folded witness `f*`.
    //
    // This is *not* succinct and not ZK; it’s the bridge before implementing Π_fold’s batched sumchecks.

    let n = 1 << 10;
    let params = RPParams {
        l_h: 64,
        lambda_pj: 32,
        k_g: 4,
        d_prime: (R::dimension() as u128) / 2,
    };
    type K = <R as PolyRing>::BaseRing;

    // Shared Ajtai matrix for commitments.
    let a = Matrix::<R>::rand(&mut ark_std::test_rng(), 2, n);

    // Derive one shared public-coin schedule from a real transcript (for J + all subsequent challenges).
    let (coins_chals, coins_bytes, coins_events) = {
        let f = vec![R::one(); n];
        let cm_f = a.try_mul_vec(&f).unwrap();
        let prover = RPRangeProver::<R>::new(f, params.clone());
        let mut ts = PoseidonTranscript::empty::<PC>();
        let mut rts = RecordingTranscriptRef::<R, _>::new(&mut ts);
        let _proof = prover.prove(&mut rts, &cm_f);
        (rts.coins_challenges, rts.coins_bytes, rts.events)
    };

    let mk = |seed: u128| -> (Vec<R>, Vec<R>, R, Vec<K>, Vec<K>) {
        let f = vec![R::from(seed); n];
        let cm_f = a.try_mul_vec(&f).unwrap();

        let prover = RPRangeProver::<R>::new(f.clone(), params.clone());
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
        // Internal checks + reconstruct (r,v)
        let out = verify_pi_rg_and_output(&proof, &cm_f, &mut ts_v).unwrap();

        // Also check Eq.(31) against the explicit witness f.
        let mut ts_v2 = FixedTranscript::<R>::new_with_coins_and_events(
            coins_chals.clone(),
            coins_bytes.clone(),
            coins_events.clone(),
        );
        verify_pi_rg_output_relation_with_witness(&proof, &cm_f, &f, &mut ts_v2).unwrap();

        // Convert v (K^d) into an Rq element (module view) for folding.
        let mut v_rq = R::ZERO;
        for (i, c) in out.v.iter().enumerate() {
            v_rq.coeffs_mut()[i] = *c;
        }

        (cm_f, f, v_rq, out.r.clone(), out.v.clone())
    };

    let (cm0, f0, v0_rq, r0, _v0_kd) = mk(7);
    let (cm1, f1, v1_rq, r1, _v1_kd) = mk(9);
    assert_eq!(r0, r1, "shared randomness required: r must match");

    // Low-norm fold coefficients β ∈ Rq (small constants for the test).
    let beta = vec![R::from(3u128), R::from(5u128)];

    // Fold witness and commitment.
    let f_star = (0..n).map(|i| beta[0] * f0[i] + beta[1] * f1[i]).collect::<Vec<_>>();
    let cm_star_from_f = a.try_mul_vec(&f_star).unwrap();
    let cm_star = (0..cm0.len())
        .map(|i| beta[0] * cm0[i] + beta[1] * cm1[i])
        .collect::<Vec<_>>();
    assert_eq!(cm_star, cm_star_from_f, "Ajtai commitment must be linear");

    // Fold v in the module view (Rq element).
    let v_star_rq = beta[0] * v0_rq + beta[1] * v1_rq;

    // Recompute v* (K^d) from the folded witness directly (Eq. (31) deterministic function).
    // We can take J from either instance since it’s shared under the public-coin schedule:
    // recover it by recomputing once from a proof-like derivation (via the deterministic v function inputs).
    //
    // Here we just reconstruct `J` by running one more Π_rg and reading it from the proof:
    let j_shared = {
        let prover = RPRangeProver::<R>::new(vec![R::one(); n], params.clone());
        let cm = a.try_mul_vec(&vec![R::one(); n]).unwrap();
        let mut ts = FixedTranscript::<R>::new_with_coins_and_events(
            coins_chals.clone(),
            coins_bytes.clone(),
            coins_events.clone(),
        );
        prover.prove(&mut ts, &cm).J
    };
    let v_star_kd = compute_auxj_lin_v_from_witness::<R>(&f_star, &j_shared, &r0, &params);

    // Compare module view coefficients.
    for (i, c) in v_star_kd.iter().enumerate() {
        assert_eq!(v_star_rq.coeffs()[i], *c);
    }

    // And ensure our fold helper matches the same folded (c,r,v) shape.
    let inst0 = symphony::symphony_fold::SymphonyInstance {
        c: cm0,
        r: r0.clone(),
        v: v0_rq,
    };
    let inst1 = symphony::symphony_fold::SymphonyInstance {
        c: cm1,
        r: r1,
        v: v1_rq,
    };
    let inst_star = fold_instances(&beta, &[inst0, inst1]);
    assert_eq!(inst_star.c, cm_star);
    assert_eq!(inst_star.r, r0);
    assert_eq!(inst_star.v, v_star_rq);
}

