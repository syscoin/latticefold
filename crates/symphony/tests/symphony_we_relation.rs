#![cfg(feature = "symphony")]

use ark_std::One;
use cyclotomic_rings::rings::FrogPoseidonConfig as PC;
use latticefold::commitment::AjtaiCommitmentScheme;
use latticefold_plus::{
    rp_rgchk::RPParams,
    symphony_open::MultiAjtaiOpenVerifier,
    symphony_pifold_batched::{prove_pi_fold_batched_sumcheck_fs},
    symphony_we_relation::{check_r_we_poseidon_fs, ReducedRelation},
};
use stark_rings::{cyclotomic_ring::models::frog_ring::RqPoly as R, PolyRing, Ring, Zq};
use stark_rings_linalg::{Matrix, SparseMatrix};

struct RoCheckCEquals;

impl ReducedRelation<R> for RoCheckCEquals
where
    <R as PolyRing>::BaseRing: Zq,
{
    type Witness = Vec<R>;

    fn check(
        public: &latticefold_plus::symphony_we_relation::FoldedOutput<R>,
        witness: &Self::Witness,
    ) -> Result<(), String> {
        if public.folded_inst.c != *witness {
            return Err("Ro: c mismatch".to_string());
        }
        Ok(())
    }
}

#[test]
fn test_r_we_conjunction_ok() {
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

    let a = Matrix::<R>::rand(&mut ark_std::test_rng(), 2, n);
    let scheme = AjtaiCommitmentScheme::<R>::new(a.clone());

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

    let a_had = Matrix::<R>::rand(&mut ark_std::test_rng(), 2, 3 * R::dimension());
    let a_mon = Matrix::<R>::rand(&mut ark_std::test_rng(), 2, rg_params.k_g);
    let scheme_had = AjtaiCommitmentScheme::<R>::new(a_had);
    let scheme_mon = AjtaiCommitmentScheme::<R>::new(a_mon);

    let out = prove_pi_fold_batched_sumcheck_fs::<R, PC>(
        [&m1, &m2, &m3],
        &[cm0.clone(), cm1.clone()],
        &[f0.clone(), f1.clone()],
        &[],
        Some(&scheme_had),
        Some(&scheme_mon),
        rg_params,
    )
    .unwrap();
    let proof = out.proof;

    // Prover-produced CP transcript witness (aux messages).
    let aux = out.aux;

    let cfs_had_u = out.cfs_had_u.clone();
    let cfs_mon_b = out.cfs_mon_b.clone();

    let open = MultiAjtaiOpenVerifier::<R>::new()
        .with_scheme("cfs_had_u", scheme_had)
        .with_scheme("cfs_mon_b", scheme_mon);

    // Compute expected folded output commitment via the reference verifier (no CP commitments).
    let (folded_inst, _folded_bat) =
        latticefold_plus::symphony_pifold_batched::verify_pi_fold_batched_and_fold_outputs_poseidon_fs::<R, PC>(
            [&m1, &m2, &m3],
            &[cm0.clone(), cm1.clone()],
            &proof,
            &latticefold_plus::symphony_open::NoOpen,
            &[f0.clone(), f1.clone()],
            &[],
        )
        .unwrap();

    check_r_we_poseidon_fs::<R, PC, RoCheckCEquals>(
        [&m1, &m2, &m3],
        &[cm0, cm1],
        &proof,
        &open,
        &cfs_had_u,
        &cfs_mon_b,
        &aux,
        &[],
        &folded_inst.c,
    )
    .unwrap();
}

#[test]
fn test_r_we_conjunction_rejects_bad_ro_witness() {
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

    let a = Matrix::<R>::rand(&mut ark_std::test_rng(), 2, n);
    let scheme = AjtaiCommitmentScheme::<R>::new(a.clone());

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

    let a_had = Matrix::<R>::rand(&mut ark_std::test_rng(), 2, 3 * R::dimension());
    let a_mon = Matrix::<R>::rand(&mut ark_std::test_rng(), 2, rg_params.k_g);
    let scheme_had = AjtaiCommitmentScheme::<R>::new(a_had);
    let scheme_mon = AjtaiCommitmentScheme::<R>::new(a_mon);

    let out = prove_pi_fold_batched_sumcheck_fs::<R, PC>(
        [&m1, &m2, &m3],
        &[cm0.clone(), cm1.clone()],
        &[f0.clone(), f1.clone()],
        &[],
        Some(&scheme_had),
        Some(&scheme_mon),
        rg_params,
    )
    .unwrap();
    let proof = out.proof;

    let aux = out.aux;

    let cfs_had_u = out.cfs_had_u.clone();
    let cfs_mon_b = out.cfs_mon_b.clone();

    let open = MultiAjtaiOpenVerifier::<R>::new()
        .with_scheme("cfs_had_u", scheme_had)
        .with_scheme("cfs_mon_b", scheme_mon);

    // Intentionally wrong R_o witness.
    let bad = vec![R::ONE; cm0.len()];

    let err = check_r_we_poseidon_fs::<R, PC, RoCheckCEquals>(
        [&m1, &m2, &m3],
        &[cm0, cm1],
        &proof,
        &open,
        &cfs_had_u,
        &cfs_mon_b,
        &aux,
        &[],
        &bad,
    )
    .unwrap_err();

    assert!(err.contains("Ro:"), "unexpected error: {err}");
}

#[test]
fn test_r_cp_poseidon_fs_rejects_tampered_commitment() {
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

    let a = Matrix::<R>::rand(&mut ark_std::test_rng(), 2, n);
    let scheme = AjtaiCommitmentScheme::<R>::new(a.clone());

    let f0 = vec![R::one(); n];
    let f1 = vec![R::ZERO; n];
    let mut cm0 = scheme.commit(&f0).unwrap().as_ref().to_vec();
    let cm1 = scheme.commit(&f1).unwrap().as_ref().to_vec();

    let rg_params = RPParams {
        l_h: 64,
        lambda_pj: 32,
        k_g: 4,
        d_prime: (R::dimension() as u128) / 2,
    };

    let a_had = Matrix::<R>::rand(&mut ark_std::test_rng(), 2, 3 * R::dimension());
    let a_mon = Matrix::<R>::rand(&mut ark_std::test_rng(), 2, rg_params.k_g);
    let scheme_had = AjtaiCommitmentScheme::<R>::new(a_had);
    let scheme_mon = AjtaiCommitmentScheme::<R>::new(a_mon);

    let out = prove_pi_fold_batched_sumcheck_fs::<R, PC>(
        [&m1, &m2, &m3],
        &[cm0.clone(), cm1.clone()],
        &[f0.clone(), f1.clone()],
        &[],
        Some(&scheme_had),
        Some(&scheme_mon),
        rg_params,
    )
    .unwrap();
    let proof = out.proof;

    let aux = out.aux;

    let cfs_had_u = out.cfs_had_u.clone();
    let cfs_mon_b = out.cfs_mon_b.clone();
    let open = MultiAjtaiOpenVerifier::<R>::new()
        .with_scheme("cfs_had_u", scheme_had)
        .with_scheme("cfs_mon_b", scheme_mon);

    // Tamper with commitment after proof was created. FS transcript binding should reject.
    cm0[0] += R::ONE;

    let err =
        latticefold_plus::symphony_we_relation::check_r_cp_poseidon_fs::<R, PC>(
            [&m1, &m2, &m3],
            &[cm0, cm1],
            &proof,
            &open,
            &cfs_had_u,
            &cfs_mon_b,
            &aux,
            &[],
        )
        .unwrap_err();

    // We don't rely on an exact string, but it should fail somewhere early in transcript-dependent checks.
    assert!(err.contains("m") || err.contains("PiFold") || err.contains("AjtaiOpen"), "unexpected error: {err}");
}

#[test]
fn test_r_cp_poseidon_fs_rejects_tampered_cfs_commitment() {
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

    // Underlying witness commitments (cm_f) used by Π_fold (not opened in CP relation).
    let a_f = Matrix::<R>::rand(&mut ark_std::test_rng(), 2, n);
    let scheme_f = AjtaiCommitmentScheme::<R>::new(a_f);
    let f0 = vec![R::one(); n];
    let f1 = vec![R::ZERO; n];
    let cm0 = scheme_f.commit(&f0).unwrap().as_ref().to_vec();
    let cm1 = scheme_f.commit(&f1).unwrap().as_ref().to_vec();

    let rg_params = RPParams {
        l_h: 64,
        lambda_pj: 32,
        k_g: 4,
        d_prime: (R::dimension() as u128) / 2,
    };

    let a_had = Matrix::<R>::rand(&mut ark_std::test_rng(), 2, 3 * R::dimension());
    let a_mon = Matrix::<R>::rand(&mut ark_std::test_rng(), 2, rg_params.k_g);
    let scheme_had = AjtaiCommitmentScheme::<R>::new(a_had);
    let scheme_mon = AjtaiCommitmentScheme::<R>::new(a_mon);

    let out = prove_pi_fold_batched_sumcheck_fs::<R, PC>(
        [&m1, &m2, &m3],
        &[cm0.clone(), cm1.clone()],
        &[f0.clone(), f1.clone()],
        &[],
        Some(&scheme_had),
        Some(&scheme_mon),
        rg_params,
    )
    .unwrap();
    let proof = out.proof;

    let aux = out.aux;
    let mut cfs_had_u = out.cfs_had_u;
    let cfs_mon_b = out.cfs_mon_b;

    // Tamper with a CP commitment after computing it; opening verification must fail.
    cfs_had_u[0][0] += R::ONE;

    let open = MultiAjtaiOpenVerifier::<R>::new()
        .with_scheme("cfs_had_u", scheme_had)
        .with_scheme("cfs_mon_b", scheme_mon);

    let err = latticefold_plus::symphony_we_relation::check_r_cp_poseidon_fs::<R, PC>(
        [&m1, &m2, &m3],
        &[cm0, cm1],
        &proof,
        &open,
        &cfs_had_u,
        &cfs_mon_b,
        &aux,
        &[],
    )
    .unwrap_err();

    assert!(
        err.contains("AjtaiOpen") || err.contains("commitment mismatch"),
        "unexpected error: {err}"
    );
}

#[test]
fn test_r_we_nonvacuous_had_identity_passes() {
    let n = 1 << 10;
    let m = 1 << 10;

    // Non-vacuous Π_had: M1=M2=M3=I, so y3 = y1 ⊙ y2 requires f ⊙ f = f entrywise.
    let mut m1 = SparseMatrix::<R>::identity(m);
    let mut m2 = SparseMatrix::<R>::identity(m);
    let mut m3 = SparseMatrix::<R>::identity(m);
    m1.pad_cols(n);
    m2.pad_cols(n);
    m3.pad_cols(n);

    // Use an idempotent witness: f[i] = 1, so 1*1 = 1.
    let f0 = vec![R::ONE; n];
    let f1 = vec![R::ZERO; n];

    // Commitments to f (cm_f).
    let a_f = Matrix::<R>::rand(&mut ark_std::test_rng(), 8, n);
    let scheme_f = AjtaiCommitmentScheme::<R>::new(a_f);
    let cm0 = scheme_f.commit(&f0).unwrap().as_ref().to_vec();
    let cm1 = scheme_f.commit(&f1).unwrap().as_ref().to_vec();

    let rg_params = RPParams {
        l_h: 64,
        lambda_pj: 32,
        k_g: 4,
        d_prime: (R::dimension() as u128) / 2,
    };

    // CP commitment schemes for aux messages (bigger row-count than other tests).
    let a_had = Matrix::<R>::rand(&mut ark_std::test_rng(), 32, 3 * R::dimension());
    let a_mon = Matrix::<R>::rand(&mut ark_std::test_rng(), 32, rg_params.k_g);
    let scheme_had = AjtaiCommitmentScheme::<R>::new(a_had);
    let scheme_mon = AjtaiCommitmentScheme::<R>::new(a_mon);

    let out = prove_pi_fold_batched_sumcheck_fs::<R, PC>(
        [&m1, &m2, &m3],
        &[cm0.clone(), cm1.clone()],
        &[f0.clone(), f1.clone()],
        &[],
        Some(&scheme_had),
        Some(&scheme_mon),
        rg_params,
    )
    .unwrap();

    let proof = out.proof;
    let aux = out.aux;
    let cfs_had_u = out.cfs_had_u;
    let cfs_mon_b = out.cfs_mon_b;

    let open = MultiAjtaiOpenVerifier::<R>::new()
        .with_scheme("cfs_had_u", scheme_had)
        .with_scheme("cfs_mon_b", scheme_mon);

    // Compute expected folded commitment via reference verifier (no CP commitments).
    let (folded_inst, _bat) =
        latticefold_plus::symphony_pifold_batched::verify_pi_fold_batched_and_fold_outputs_poseidon_fs::<R, PC>(
            [&m1, &m2, &m3],
            &[cm0.clone(), cm1.clone()],
            &proof,
            &latticefold_plus::symphony_open::NoOpen,
            &[f0, f1],
            &[],
        )
        .unwrap();

    check_r_we_poseidon_fs::<R, PC, RoCheckCEquals>(
        [&m1, &m2, &m3],
        &[scheme_f.commit(&vec![R::ONE; n]).unwrap().as_ref().to_vec(), scheme_f.commit(&vec![R::ZERO; n]).unwrap().as_ref().to_vec()],
        &proof,
        &open,
        &cfs_had_u,
        &cfs_mon_b,
        &aux,
        &[],
        &folded_inst.c,
    )
    .unwrap();
}

#[test]
fn test_r_we_nonvacuous_had_identity_rejects_non_idempotent_witness() {
    let n = 1 << 10;
    let m = 1 << 10;

    // M1=M2=M3=I as above, but use f=2 so f⊙f != f.
    let mut m1 = SparseMatrix::<R>::identity(m);
    let mut m2 = SparseMatrix::<R>::identity(m);
    let mut m3 = SparseMatrix::<R>::identity(m);
    m1.pad_cols(n);
    m2.pad_cols(n);
    m3.pad_cols(n);

    let two = R::from(2u128);
    let f0 = vec![two; n];
    let f1 = vec![R::ZERO; n];

    let a_f = Matrix::<R>::rand(&mut ark_std::test_rng(), 8, n);
    let scheme_f = AjtaiCommitmentScheme::<R>::new(a_f);
    let cm0 = scheme_f.commit(&f0).unwrap().as_ref().to_vec();
    let cm1 = scheme_f.commit(&f1).unwrap().as_ref().to_vec();

    let rg_params = RPParams {
        l_h: 64,
        lambda_pj: 32,
        k_g: 4,
        d_prime: (R::dimension() as u128) / 2,
    };

    let a_had = Matrix::<R>::rand(&mut ark_std::test_rng(), 16, 3 * R::dimension());
    let a_mon = Matrix::<R>::rand(&mut ark_std::test_rng(), 16, rg_params.k_g);
    let scheme_had = AjtaiCommitmentScheme::<R>::new(a_had);
    let scheme_mon = AjtaiCommitmentScheme::<R>::new(a_mon);

    let out = prove_pi_fold_batched_sumcheck_fs::<R, PC>(
        [&m1, &m2, &m3],
        &[cm0.clone(), cm1.clone()],
        &[f0, f1],
        &[],
        Some(&scheme_had),
        Some(&scheme_mon),
        rg_params,
    )
    .unwrap();

    let open = MultiAjtaiOpenVerifier::<R>::new()
        .with_scheme("cfs_had_u", scheme_had)
        .with_scheme("cfs_mon_b", scheme_mon);

    // Even with valid cfs commitments/openings, Π_had should reject because the underlying had relation is false.
    let err = latticefold_plus::symphony_we_relation::check_r_cp_poseidon_fs::<R, PC>(
        [&m1, &m2, &m3],
        &[cm0, cm1],
        &out.proof,
        &open,
        &out.cfs_had_u,
        &out.cfs_mon_b,
        &out.aux,
        &[],
    )
    .unwrap_err();

    assert!(
        err.contains("PiFold") || err.contains("Eq(26)") || err.contains("sumcheck"),
        "unexpected error: {err}"
    );
}
