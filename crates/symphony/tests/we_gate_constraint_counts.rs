use cyclotomic_rings::rings::FrogPoseidonConfig as PC;
use latticefold::commitment::AjtaiCommitmentScheme;
use stark_rings::{cyclotomic_ring::models::frog_ring::RqPoly as R, PolyRing};
use stark_rings_linalg::SparseMatrix;

use symphony::pcs::dpp_folding_pcs_l2::folding_pcs_l2_params;
use symphony::pcs::folding_pcs_l2::{
    gadget_apply_digits, kron_ct_in_mul, kron_i_a_mul, kron_ikn_xt_mul, BinMatrix, DenseMatrix,
    FoldingPcsL2ProofCore,
    verify_folding_pcs_l2_with_c_matrices,
};
use symphony::rp_rgchk::RPParams;
use symphony::symphony_open::MultiAjtaiOpenVerifier;
use symphony::symphony_pifold_batched::{verify_pi_fold_cp_poseidon_fs, PiFoldMatrices};
use symphony::symphony_pifold_streaming::{prove_pi_fold_poseidon_fs, PiFoldStreamingConfig};
use symphony::transcript::PoseidonTraceOp;
use symphony::we_gate_arith::WeGateDr1csBuilder;
use symphony::pcs::batchlin_pcs::{batchlin_scalar_pcs_params, BATCHLIN_PCS_DOMAIN_SEP};

use ark_ff::Field;

const MASTER_SEED: [u8; 32] = *b"SYMPHONY_AJTAI_SEED_V1_000000000";

fn bits_le_from_bytes(bytes: &[u8]) -> Vec<bool> {
    let mut out = Vec::with_capacity(bytes.len() * 8);
    for &b in bytes {
        for i in 0..8 {
            out.push(((b >> i) & 1) == 1);
        }
    }
    out
}

#[test]
fn we_gate_constraint_counts_real_pcs_in_gate() {
    // Keep this tiny: we just want a stable, deterministic constraint count smoke test.
    // NOTE: batchlin PCS (ℓ=2) currently requires log2(m*d) divisible by 3.
    // For Frog, d=16, so pick m=32 => m*d=512 => log_n=9.
    // Also keep n >= m so SparseMatrix::pad_cols(n) doesn't need to "shrink" identity matrices.
    let n = 1 << 5; // witness length
    let m = 1 << 5; // rows per instance

    // Two instances with sparse A, and B=C=0 so constraints hold for any witness.
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
        .map(|i| {
            if i % 2 == 0 {
                <R as stark_rings::Ring>::ONE
            } else {
                <R as stark_rings::Ring>::ZERO
            }
        })
        .collect::<Vec<_>>();
    let cm0 = scheme.commit_const_coeff_fast(&f0).unwrap().as_ref().to_vec();
    let cm1 = scheme.commit_const_coeff_fast(&f1).unwrap().as_ref().to_vec();
    let cms = vec![cm0, cm1];
    let witnesses = vec![std::sync::Arc::new(f0), std::sync::Arc::new(f1)];
    let public_inputs: Vec<<R as PolyRing>::BaseRing> = vec![];

    // CP transcript-message commitment schemes.
    // We build two copies so one can be moved into `open_cfs` and one can be used by the gate builder.
    let scheme_had_gate = AjtaiCommitmentScheme::<R>::seeded(
        b"cfs_had_u",
        MASTER_SEED,
        2,
        3 * R::dimension(),
    );
    let scheme_mon_gate =
        AjtaiCommitmentScheme::<R>::seeded(b"cfs_mon_b", MASTER_SEED, 2, rg_params.k_g);
    let scheme_had_verifier = AjtaiCommitmentScheme::<R>::seeded(
        b"cfs_had_u",
        MASTER_SEED,
        2,
        3 * R::dimension(),
    );
    let scheme_mon_verifier =
        AjtaiCommitmentScheme::<R>::seeded(b"cfs_mon_b", MASTER_SEED, 2, rg_params.k_g);
    let scheme_g = AjtaiCommitmentScheme::<R>::seeded(b"cm_g", MASTER_SEED, 2, m * R::dimension());

    let cfg = PiFoldStreamingConfig::default();
    let out = prove_pi_fold_poseidon_fs::<R, PC>(
        ms.as_slice(),
        &cms,
        &witnesses,
        &public_inputs,
        Some(&scheme_had_gate),
        Some(&scheme_mon_gate),
        &scheme_g,
        rg_params.clone(),
        &cfg,
    )
    .expect("prove failed");

    // Run the real verifier to obtain the Poseidon trace ops.
    let open_cfs = MultiAjtaiOpenVerifier::new()
        .with_scheme("cfs_had_u", scheme_had_verifier)
        .with_scheme("cfs_mon_b", scheme_mon_verifier);
    let attempt = verify_pi_fold_cp_poseidon_fs::<R, PC>(
        PiFoldMatrices::Hetero(ms_refs.as_slice()),
        &cms,
        &out.proof,
        &open_cfs,
        &out.cfs_had_u,
        &out.cfs_mon_b,
        &out.aux,
        &public_inputs,
    );
    attempt.result.expect("cp verify failed");
    let trace = attempt.trace;

    type BF = <<R as PolyRing>::BaseRing as Field>::BasePrimeField;
    let poseidon_cfg = <PC as cyclotomic_rings::rings::GetPoseidonParams<BF>>::get_poseidon_config();

    // ------------------------
    // R_cp (Poseidon + Π_fold-math + cfs-openings)
    // ------------------------
    let (rcp, rcp_asg) = WeGateDr1csBuilder::r_cp_poseidon_pifold_math_and_cfs_openings::<R>(
        &poseidon_cfg,
        &trace.ops,
        &cms,
        &out.proof,
        &scheme_had_gate,
        &scheme_mon_gate,
        &out.aux,
        &out.cfs_had_u,
        &out.cfs_mon_b,
    )
    .expect("build r_cp dr1cs failed");
    rcp.check(&rcp_asg).expect("r_cp unsat");

    // ------------------------
    // Full gate: R_cp × PCS (folding PCS ℓ=2, coins from Poseidon SqueezeBytes)
    // ------------------------
    let squeeze_bytes: Vec<Vec<u8>> = trace
        .ops
        .iter()
        .filter_map(|op| match op {
            PoseidonTraceOp::SqueezeBytes { out, .. } => Some(out.clone()),
            _ => None,
        })
        .collect();
    assert!(
        !squeeze_bytes.is_empty(),
        "expected verifier trace to contain at least one SqueezeBytes op"
    );
    let pcs_coin_squeeze_idx = 0usize;
    let c_bytes = &squeeze_bytes[pcs_coin_squeeze_idx];
    let bits = bits_le_from_bytes(c_bytes);

    // Tiny PCS instance (only used to measure gate arithmetization overhead).
    let r = 1usize;
    let kappa = 2usize;
    let pcs_n = 4usize;
    let delta = 4u64;
    let alpha = 1usize;
    let beta0 = 1u64 << 10;
    let beta1 = 2 * beta0;
    let beta2 = 2 * beta1;

    let c1 = BinMatrix {
        rows: r * kappa,
        cols: kappa,
        data: (0..(r * kappa * kappa))
            .map(|i| if bits[i] { BF::ONE } else { BF::ZERO })
            .collect(),
    };
    let c2 = BinMatrix {
        rows: r * kappa,
        cols: kappa,
        data: (0..(r * kappa * kappa))
            .map(|i| if bits[(r * kappa * kappa) + i] { BF::ONE } else { BF::ZERO })
            .collect(),
    };

    // A = I_{pcs_n}.
    let mut a_data = vec![BF::ZERO; pcs_n * (r * pcs_n * alpha)];
    for i in 0..pcs_n {
        a_data[i * (r * pcs_n * alpha) + i] = BF::ONE;
    }
    let a = DenseMatrix::new(pcs_n, r * pcs_n * alpha, a_data);
    let pcs_params = folding_pcs_l2_params(r, kappa, pcs_n, delta, alpha, beta0, beta1, beta2, a);

    let x0 = vec![BF::ONE; r];
    let x1 = vec![BF::ONE; r];
    let x2 = vec![BF::ONE; r];

    // Deterministic small y0 within bounds (no RNG needed for count test).
    let y0 = vec![BF::ONE; pcs_params.y0_len()];
    let y1 = kron_ct_in_mul(&c1, pcs_n, &y0);
    let y2 = kron_ct_in_mul(&c2, pcs_n, &y1);
    let t_pcs = kron_i_a_mul(&pcs_params.a, pcs_params.kappa, pcs_params.r * pcs_params.n * pcs_params.alpha, &y0);
    let mut delta_pows = Vec::with_capacity(alpha);
    let mut acc = BF::ONE;
    let delta_f = BF::from(delta);
    for _ in 0..alpha {
        delta_pows.push(acc);
        acc *= delta_f;
    }
    let v0 = gadget_apply_digits(&delta_pows, r * kappa * pcs_n, &y0);
    let v1 = gadget_apply_digits(&delta_pows, r * kappa * pcs_n, &y1);
    let v2 = gadget_apply_digits(&delta_pows, r * kappa * pcs_n, &y2);
    let u_pcs = kron_ikn_xt_mul(&x2, kappa, pcs_n, &v0);

    let pcs_core = FoldingPcsL2ProofCore { y0, v0, y1, v1, y2, v2 };
    verify_folding_pcs_l2_with_c_matrices(
        &pcs_params, &t_pcs, &x0, &x1, &x2, &u_pcs, &pcs_core, &c1, &c2,
    )
    .expect("native folding PCS sanity check failed");

    // Use the full trace ops here (production-shape).
    // PCS#2 (batchlin) must use the prover-produced commitment surface and transcript-bound coins.
    let log_n = ((out.proof.m * R::dimension()) as f64).log2() as usize;
    let pcs_params_g = batchlin_scalar_pcs_params::<BF>(log_n).expect("batchlin pcs params");
    let t_pcs_g = &out.proof.batchlin_pcs_t[0];
    let x0_g = &out.proof.batchlin_pcs_x0;
    let x1_g = &out.proof.batchlin_pcs_x1;
    let x2_g = &out.proof.batchlin_pcs_x2;
    let pcs_core_g = &out.proof.batchlin_pcs_core;

    // Find the SqueezeBytes index that follows `BATCHLIN_PCS_DOMAIN_SEP` absorption.
    let mut pcs_coin_squeeze_idx2: Option<usize> = None;
    {
        let marker = BF::from(BATCHLIN_PCS_DOMAIN_SEP);
        let mut squeeze_idx = 0usize;
        let mut saw_marker = false;
        for op in &trace.ops {
            match op {
                PoseidonTraceOp::Absorb(v) => {
                    if v.len() == 1 && v[0] == marker {
                        saw_marker = true;
                    }
                }
                PoseidonTraceOp::SqueezeBytes { .. } => {
                    if saw_marker {
                        pcs_coin_squeeze_idx2 = Some(squeeze_idx);
                        break;
                    }
                    squeeze_idx += 1;
                }
                _ => {}
            }
        }
    }
    let pcs_coin_squeeze_idx2 =
        pcs_coin_squeeze_idx2.expect("missing SqueezeBytes after BATCHLIN_PCS_DOMAIN_SEP");

    let (full, full_asg) = WeGateDr1csBuilder::poseidon_plus_pifold_plus_cfs_plus_pcs::<R>(
        &poseidon_cfg,
        &trace.ops,
        &cms,
        &out.proof,
        &scheme_had_gate,
        &scheme_mon_gate,
        &out.aux,
        &out.cfs_had_u,
        &out.cfs_mon_b,
        &pcs_params,
        &t_pcs,
        &x0,
        &x1,
        &x2,
        &pcs_core,
        pcs_coin_squeeze_idx,
        &pcs_params_g,
        t_pcs_g,
        x0_g,
        x1_g,
        x2_g,
        pcs_core_g,
        pcs_coin_squeeze_idx2,
    )
    .expect("build full gate dr1cs failed");
    full.check(&full_asg).expect("full gate unsat");

    // Print counts for humans (use `-- --nocapture`).
    eprintln!("==============================================================");
    eprintln!("WE gate constraint counts (real PCS-in-gate path)");
    eprintln!("  r_cp:   nvars={} constraints={}", rcp.nvars, rcp.constraints.len());
    eprintln!("  full:   nvars={} constraints={}", full.nvars, full.constraints.len());
    eprintln!("  delta:  nvars={} constraints={}", full.nvars.saturating_sub(rcp.nvars), full.constraints.len().saturating_sub(rcp.constraints.len()));
    eprintln!("==============================================================");

    // Keep a very weak invariant so the test fails if it accidentally becomes trivial.
    assert!(full.constraints.len() > rcp.constraints.len());
}

