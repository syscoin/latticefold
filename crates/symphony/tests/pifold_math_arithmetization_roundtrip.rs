use cyclotomic_rings::rings::FrogPoseidonConfig as PC;
use ark_ff::Field;
use latticefold::utils::sumcheck::MLSumcheck;
use latticefold::utils::sumcheck::Proof;
use latticefold::transcript::Transcript;
use stark_rings::{cyclotomic_ring::models::frog_ring::RqPoly as R, PolyRing};
use stark_rings_linalg::SparseMatrix;

use symphony::dpp_pifold_math::pifold_verifier_math_dr1cs;
use symphony::public_coin_transcript::FixedTranscript;
use symphony::rp_rgchk::RPParams;
use symphony::symphony_coins::{derive_J, derive_beta_chi};
use symphony::symphony_open::MultiAjtaiOpenVerifier;
use symphony::symphony_pifold_batched::verify_pi_fold_batched_and_fold_outputs_poseidon_fs_cp_hetero_m_with_metrics;
use symphony::symphony_pifold_streaming::{prove_pi_fold_poseidon_fs, PiFoldStreamingConfig};
use latticefold::commitment::AjtaiCommitmentScheme;

const MASTER_SEED: [u8; 32] = *b"SYMPHONY_AJTAI_SEED_V1_000000000";

fn bf<Ring: PolyRing>(x: Ring::BaseRing) -> <Ring::BaseRing as ark_ff::Field>::BasePrimeField
where
    Ring::BaseRing: ark_ff::Field,
{
    x.to_base_prime_field_elements()
        .into_iter()
        .next()
        .expect("bf expects extension degree 1")
}

#[test]
fn test_pifold_math_dr1cs_roundtrip_satisfiable_and_tamper_fails() {
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
    let ms_refs: Vec<[&SparseMatrix<R>; 3]> = vec![[&a0, &b0, &c0], [&a1, &b1, &c1]];

    let rg_params = RPParams {
        l_h: 4,
        lambda_pj: 1,
        k_g: 2,
        d_prime: (R::dimension() as u128) - 2,
    };

    let scheme = AjtaiCommitmentScheme::<R>::seeded(b"cm_f", MASTER_SEED, 2, n);
    let f0 = vec![<R as stark_rings::Ring>::ONE; n];
    let f1 = (0..n)
        .map(|i| if i % 2 == 0 { <R as stark_rings::Ring>::ONE } else { <R as stark_rings::Ring>::ZERO })
        .collect::<Vec<_>>();
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

    // Sanity: CP verify should succeed and give us a Poseidon trace.
    let open_cfs = MultiAjtaiOpenVerifier::new()
        .with_scheme("cfs_had_u", scheme_had)
        .with_scheme("cfs_mon_b", scheme_mon);
    let (_folded_out, _metrics, _trace) =
        verify_pi_fold_batched_and_fold_outputs_poseidon_fs_cp_hetero_m_with_metrics::<R, PC>(
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

    // ---------------------------------------------------------------------
    // Re-extract the verifier coin pieces by replaying the exact verifier schedule
    // over a FixedTranscript seeded by the prover's recorded coins.
    // ---------------------------------------------------------------------
    let mut ts = FixedTranscript::<R>::new_with_coins_and_events(
        out.proof.coins.challenges.clone(),
        out.proof.coins.bytes.clone(),
        out.proof.coins.events.clone(),
    );

    type BF = <<R as PolyRing>::BaseRing as Field>::BasePrimeField;

    let ell = cms.len();
    let beta_cts = derive_beta_chi::<R>(&mut ts, ell);
    let beta_cts_bf = beta_cts.iter().copied().map(bf::<R>).collect::<Vec<_>>();

    let m = ms_refs[0][0].nrows;
    let log_m = (m as f64).log2() as usize;
    let d = R::dimension();
    let g_len = m * d;
    let g_nvars = (g_len as f64).log2() as usize;

    ts.absorb_field_element(&<R as PolyRing>::BaseRing::from(0x4c465053_50494841u128)); // "LFPS_PIHA"
    let s_base = ts.get_challenges(log_m);
    let s_base_bf = s_base.iter().copied().map(bf::<R>).collect::<Vec<_>>();
    let alpha_base = ts.get_challenge();
    let alpha_base_bf = bf::<R>(alpha_base);

    let mut cba_all_bf: Vec<Vec<(Vec<BF>, BF, BF)>> = Vec::with_capacity(ell);
    let mut rc_all_bf: Vec<Option<BF>> = Vec::with_capacity(ell);

    for cm_f in &cms {
        ts.absorb_slice(cm_f);
        let _j = derive_J::<R>(&mut ts, rg_params.lambda_pj, rg_params.l_h);
        let mut cba = Vec::with_capacity(rg_params.k_g);
        for _ in 0..rg_params.k_g {
            let c = ts.get_challenges(g_nvars);
            let c_bf = c.into_iter().map(bf::<R>).collect::<Vec<_>>();
            let beta = bf::<R>(ts.get_challenge());
            let alpha = bf::<R>(ts.get_challenge());
            cba.push((c_bf, beta, alpha));
        }
        let rc = (rg_params.k_g > 1).then(|| bf::<R>(ts.get_challenge()));
        cba_all_bf.push(cba);
        rc_all_bf.push(rc);
    }

    let rhos = ts.get_challenges(ell);
    let rhos_bf = rhos.into_iter().map(bf::<R>).collect::<Vec<_>>();

    let hook_round = (out.proof.m_j.next_power_of_two() as f64).log2() as usize;
    let (_had_sc, mon_sc) = MLSumcheck::<R, _>::verify_two_as_subprotocol_shared_with_hook(
        &mut ts,
        log_m,
        3,
        <R as stark_rings::Ring>::ZERO,
        &out.proof.had_sumcheck,
        g_nvars,
        3,
        <R as stark_rings::Ring>::ZERO,
        &out.proof.mon_sumcheck,
        hook_round,
        |t, _sampled| {
            for v_i in &out.proof.v_digits_folded {
                for x in v_i {
                    t.absorb_field_element(x);
                }
            }
        },
    )
    .expect("sumcheck verify replay failed");

    let rs_shared_bf = mon_sc.point.iter().copied().map(bf::<R>).collect::<Vec<_>>();

    // ---------------------------------------------------------------------
    // Build Π_fold arithmetic dR1CS and check satisfiable.
    // ---------------------------------------------------------------------
    let (inst, asg, _wiring) = pifold_verifier_math_dr1cs::<R>(
        &out.proof,
        &out.aux,
        &beta_cts_bf,
        &s_base_bf,
        alpha_base_bf,
        &cba_all_bf,
        &rc_all_bf,
        &rhos_bf,
        &rs_shared_bf,
    )
    .expect("build pifold math dr1cs");
    if let Err(e) = inst.check(&asg) {
        // Try to extract failing constraint index from error string like "constraint N failed".
        let msg = format!("{e:?}");
        let idx = msg
            .split_whitespace()
            .find_map(|tok| tok.parse::<usize>().ok());
        if let Some(i) = idx {
            let row = &inst.constraints[i];
            let eval_lc = |lc: &[(BF, usize)]| -> BF {
                lc.iter().fold(BF::ZERO, |acc, (c, v)| acc + (*c) * asg[*v])
            };
            let a = eval_lc(&row.a);
            let b2 = eval_lc(&row.b);
            let c = eval_lc(&row.c);
            let fmt_lc = |lc: &[(BF, usize)]| -> String {
                let mut s = String::new();
                for (k, (coef, var)) in lc.iter().take(12).enumerate() {
                    if k > 0 { s.push_str(" + "); }
                    s.push_str(&format!("{coef}*v{var}"));
                }
                if lc.len() > 12 { s.push_str(" + ..."); }
                s
            };
            panic!(
                "pifold math dr1cs unsat: {msg}\n  failing constraint {i}: (A·x)={a} (B·x)={b2} (C·x)={c} => LHS={}\n  A: {}\n  B: {}\n  C: {}",
                a * b2,
                fmt_lc(&row.a),
                fmt_lc(&row.b),
                fmt_lc(&row.c),
            );
        }
        panic!("pifold math dr1cs unsat: {msg}");
    }

    // Negative check: tamper one had sumcheck evaluation but keep aux fixed.
    let mut bad_proof = out.proof.clone();
    let mut bad_msgs = bad_proof.had_sumcheck.msgs().to_vec();
    bad_msgs[0].evaluations[0] = bad_msgs[0].evaluations[0] + <R as stark_rings::Ring>::ONE;
    bad_proof.had_sumcheck = Proof::new(bad_msgs);
    let (inst_bad, asg_bad, _wiring_bad) = pifold_verifier_math_dr1cs::<R>(
        &bad_proof,
        &out.aux,
        &beta_cts_bf,
        &s_base_bf,
        alpha_base_bf,
        &cba_all_bf,
        &rc_all_bf,
        &rhos_bf,
        &rs_shared_bf,
    )
    .expect("build pifold math dr1cs (bad)");
    assert!(inst_bad.check(&asg_bad).is_err(), "tampered proof unexpectedly satisfied Π_fold math dR1CS");
}

