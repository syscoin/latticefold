//! Micro-benchmark for Symphony Π_fold CP-style verification (Poseidon-FS).
//!
//! This is intentionally lightweight (no criterion) and meant for quick GM-style sanity checks:
//! measure how Π_fold proving/verifying scales with batch size ℓ under the current implementation,
//! including Ajtai opening checks for CP transcript-message commitments (`cfs_*`).

use std::sync::Arc;
use std::time::Instant;

fn main() {
    use cyclotomic_rings::rings::FrogPoseidonConfig as PC;
    use latticefold::commitment::AjtaiCommitmentScheme;
    use symphony::{
        pcs::dpp_folding_pcs_l2::folding_pcs_l2_params,
        pcs::folding_pcs_l2::{
            gadget_apply_digits, kron_ct_in_mul, kron_i_a_mul, kron_ikn_xt_mul, BinMatrix,
            DenseMatrix, FoldingPcsL2ProofCore,
            verify_folding_pcs_l2_with_c_matrices,
        },
        pcs::{cmf_pcs, folding_pcs_l2},
        rp_rgchk::RPParams,
        symphony_open::MultiAjtaiOpenVerifier,
        symphony_pifold_batched::{verify_pi_fold_cp_poseidon_fs, PiFoldMatrices},
        symphony_pifold_streaming::{prove_pi_fold_poseidon_fs, PiFoldStreamingConfig},
        symphony_we_relation::{FoldedOutput, TrivialRo},
        transcript::PoseidonTraceOp,
        poseidon_trace::find_squeeze_bytes_idx_after_absorb_marker,
        we_gate_arith::WeGateDr1csBuilder,
    };
    use cyclotomic_rings::rings::GetPoseidonParams;
    use ark_ff::Field;
    use stark_rings::{cyclotomic_ring::models::frog_ring::RqPoly as R, PolyRing, Ring};
    use stark_rings_linalg::SparseMatrix;

    // Keep these at the test-friendly "toy" scale for interactive runs. Increase if you want.
    let n = 1 << 10;
    let m = 1 << 10;

    // Non-vacuous Π_had: M1=M2=M3=I.
    let mut m1 = SparseMatrix::<R>::identity(m);
    let mut m2 = SparseMatrix::<R>::identity(m);
    let mut m3 = SparseMatrix::<R>::identity(m);
    m1.pad_cols(n);
    m2.pad_cols(n);
    m3.pad_cols(n);
    let m1 = Arc::new(m1);
    let m2 = Arc::new(m2);
    let m3 = Arc::new(m3);

    let rg_params = RPParams {
        l_h: 64,
        lambda_pj: 32,
        k_g: 4,
        d_prime: (R::dimension() as u128) / 2,
    };

    // "Statement binding" placeholder (e.g. vk hash + claim digest).
    let public_inputs: Vec<<R as PolyRing>::BaseRing> = vec![
        <R as PolyRing>::BaseRing::from(123u128),
        <R as PolyRing>::BaseRing::from(456u128),
    ];

    // cm_f commitment surface: PCS-backed (packed into ring elements for existing Π_fold APIs).
    const MASTER_SEED: [u8; 32] = *b"SYMPHONY_AJTAI_SEED_V1_000000000";
    let kappa_cm_f = 8usize;

    // Try a few batch sizes ℓ.
    for &ell in &[2usize, 4, 8] {
        // Build witnesses: alternate 1 and 0 (both idempotent).
        let mut witnesses: Vec<Arc<Vec<R>>> = Vec::with_capacity(ell);
        let mut cm_f: Vec<Vec<R>> = Vec::with_capacity(ell);
        for i in 0..ell {
            let f = if i % 2 == 0 {
                vec![R::ONE; n]
            } else {
                vec![R::ZERO; n]
            };
            type BF = <<R as PolyRing>::BaseRing as Field>::BasePrimeField;
            let flat_witness: Vec<BF> = f
                .iter()
                .flat_map(|re| {
                    re.coeffs()
                        .iter()
                        .map(|c| c.to_base_prime_field_elements().into_iter().next().expect("bf limb"))
                })
                .collect();
            let pcs_params_f = cmf_pcs::cmf_pcs_params_for_flat_len::<BF>(
                flat_witness.len(),
                kappa_cm_f,
            )
            .expect("cm_f pcs params");
            let f_pcs_f = cmf_pcs::pad_flat_message(&pcs_params_f, &flat_witness);
            let (t_pcs_f, _s_pcs_f) =
                folding_pcs_l2::commit(&pcs_params_f, &f_pcs_f).expect("cm_f pcs commit failed");
            let cm = cmf_pcs::pack_t_as_ring::<R>(&t_pcs_f);
            witnesses.push(Arc::new(f));
            cm_f.push(cm);
        }

        // Build Ms: shared matrices repeated for each instance.
        let ms: Vec<[Arc<SparseMatrix<R>>; 3]> = (0..ell)
            .map(|_| [m1.clone(), m2.clone(), m3.clone()])
            .collect();

        // CP commitments for aux messages (cfs_*). Use a larger row count so we can see the
        // Ajtai-open verification cost.
        let scheme_had =
            AjtaiCommitmentScheme::<R>::seeded(b"cfs_had_u", MASTER_SEED, 32, 3 * R::dimension());
        let scheme_mon =
            AjtaiCommitmentScheme::<R>::seeded(b"cfs_mon_b", MASTER_SEED, 32, rg_params.k_g);
        let scheme_g =
            AjtaiCommitmentScheme::<R>::seeded(b"cm_g", MASTER_SEED, 32, m * R::dimension());

        let cfg = PiFoldStreamingConfig::default();
        let prove_start = Instant::now();
        let out = prove_pi_fold_poseidon_fs::<R, PC>(
            ms.as_slice(),
            &cm_f,
            &witnesses,
            &public_inputs,
            Some(&scheme_had),
            Some(&scheme_mon),
            &scheme_g,
            rg_params.clone(),
            &cfg,
        )
        .unwrap();
        let prove_time = prove_start.elapsed();

        let open = MultiAjtaiOpenVerifier::<R>::new()
            .with_scheme("cfs_had_u", scheme_had)
            .with_scheme("cfs_mon_b", scheme_mon);

        let verify_start = Instant::now();
        let attempt = verify_pi_fold_cp_poseidon_fs::<R, PC>(
            PiFoldMatrices::Shared([&*m1, &*m2, &*m3]),
            &cm_f,
            &out.proof,
            &open,
            &out.cfs_had_u,
            &out.cfs_mon_b,
            &out.aux,
            &public_inputs,
        )
        ;
        let (folded_inst, folded_bat) = attempt.result.unwrap();
        let verify_time = verify_start.elapsed();
        let trace = attempt.trace;

        // Sanity: `TrivialRo` exists just to show how R_WE would be checked; we don't benchmark it.
        let folded_out = FoldedOutput {
            folded_inst,
            folded_bat,
        };
        let _: Result<(), String> =
            <TrivialRo as symphony::symphony_we_relation::ReducedRelation<R>>::check(&folded_out, &());

        println!(
            "ℓ={ell:>2} | Π_fold prove: {:>8.3}s | R_cp verify: {:>8.3}s | proof bytes: {}",
            prove_time.as_secs_f64(),
            verify_time.as_secs_f64(),
            out.proof.coins.bytes.len()
        );

        // Gate size report: build r_cp and full (with a tiny PCS instance, coins from trace SqueezeBytes).
        type BF = <<R as PolyRing>::BaseRing as Field>::BasePrimeField;
        let poseidon_cfg = <PC as GetPoseidonParams<BF>>::get_poseidon_config();

        let (rcp, rcp_asg) = WeGateDr1csBuilder::r_cp_poseidon_pifold_math_and_cfs_openings::<R>(
            &poseidon_cfg,
            &trace.ops,
            &cm_f,
            &out.proof,
            &open.schemes["cfs_had_u"],
            &open.schemes["cfs_mon_b"],
            &out.aux,
            &out.cfs_had_u,
            &out.cfs_mon_b,
        )
        .expect("build r_cp dr1cs failed");
        rcp.check(&rcp_asg).expect("r_cp unsat");

        let squeeze_bytes: Vec<Vec<u8>> = trace
            .ops
            .iter()
            .filter_map(|op| match op {
                PoseidonTraceOp::SqueezeBytes { out, .. } => Some(out.clone()),
                _ => None,
            })
            .collect();
        if squeeze_bytes.is_empty() {
            println!("    gate dr1cs: r_cp(nvars={}, constraints={}) (no SqueezeBytes => skip full pcs)",
                rcp.nvars, rcp.constraints.len()
            );
            continue;
        }

        let pcs_coin_squeeze_idx = find_squeeze_bytes_idx_after_absorb_marker(
            &trace.ops,
            BF::from(cmf_pcs::CMF_PCS_DOMAIN_SEP),
        )
        .expect("missing SqueezeBytes after CMF_PCS_DOMAIN_SEP");
        let c_bytes = &squeeze_bytes[pcs_coin_squeeze_idx];
        let mut bits = Vec::with_capacity(c_bytes.len() * 8);
        for &b in c_bytes {
            for i in 0..8 {
                bits.push(((b >> i) & 1) == 1);
            }
        }
        let r = 1usize;
        let kappa = 2usize;
        let pcs_n = 4usize;
        // Must satisfy delta^alpha >= modulus (enforced by folding_pcs_l2 exactness guard).
        let delta = 1u64 << 32;
        let alpha = 2usize;
        let beta0 = 1u64 << 10;
        let beta1 = 2 * beta0;
        let beta2 = 2 * beta1;
        let c1 = BinMatrix {
            rows: r * kappa,
            cols: kappa,
            data: (0..(r * kappa * kappa))
                .map(|i| if bits[i] { <BF as Field>::ONE } else { <BF as Field>::ZERO })
                .collect(),
        };
        let c2 = BinMatrix {
            rows: r * kappa,
            cols: kappa,
            data: (0..(r * kappa * kappa))
                .map(|i| {
                    if bits[(r * kappa * kappa) + i] {
                        <BF as Field>::ONE
                    } else {
                        <BF as Field>::ZERO
                    }
                })
                .collect(),
        };
        let mut a_data = vec![<BF as Field>::ZERO; pcs_n * (r * pcs_n * alpha)];
        for i in 0..pcs_n {
            a_data[i * (r * pcs_n * alpha) + i] = <BF as Field>::ONE;
        }
        let a = DenseMatrix::new(pcs_n, r * pcs_n * alpha, a_data);
        let pcs_params = folding_pcs_l2_params(r, kappa, pcs_n, delta, alpha, beta0, beta1, beta2, a);
        let x0 = vec![<BF as Field>::ONE; r];
        let x1 = vec![<BF as Field>::ONE; r];
        let x2 = vec![<BF as Field>::ONE; r];
        let y0 = vec![<BF as Field>::ONE; pcs_params.y0_len()];
        let y1 = kron_ct_in_mul(&c1, pcs_n, &y0);
        let y2 = kron_ct_in_mul(&c2, pcs_n, &y1);
        let t_pcs = kron_i_a_mul(&pcs_params.a, pcs_params.kappa, pcs_params.r * pcs_params.n * pcs_params.alpha, &y0);
        let mut delta_pows = Vec::with_capacity(alpha);
        let mut acc = <BF as Field>::ONE;
        let delta_f = <BF as From<u64>>::from(delta);
        for _ in 0..alpha {
            delta_pows.push(acc);
            acc *= delta_f;
        }
        let v0 = gadget_apply_digits(&delta_pows, r * kappa * pcs_n, &y0);
        let v1 = gadget_apply_digits(&delta_pows, r * kappa * pcs_n, &y1);
        let v2 = gadget_apply_digits(&delta_pows, r * kappa * pcs_n, &y2);
        let u_pcs = kron_ikn_xt_mul(&x2, kappa, pcs_n, &v0);
        let pcs_core = FoldingPcsL2ProofCore { y0, v0, y1, v1, y2, v2 };
        verify_folding_pcs_l2_with_c_matrices(&pcs_params, &t_pcs, &x0, &x1, &x2, &u_pcs, &pcs_core, &c1, &c2)
            .expect("native folding pcs sanity failed");

        let (full, full_asg) = WeGateDr1csBuilder::poseidon_plus_pifold_plus_cfs_plus_pcs::<R>(
            &poseidon_cfg,
            &trace.ops,
            &cm_f,
            &out.proof,
            &open.schemes["cfs_had_u"],
            &open.schemes["cfs_mon_b"],
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
            &pcs_params,
            &t_pcs,
            &x0,
            &x1,
            &x2,
            &pcs_core,
            pcs_coin_squeeze_idx.saturating_add(1),
        )
        .expect("build full gate dr1cs failed");
        full.check(&full_asg).expect("full gate unsat");

        println!(
            "    gate dr1cs: r_cp(nvars={}, constraints={})  full(nvars={}, constraints={})  delta(nvars={}, constraints={})",
            rcp.nvars,
            rcp.constraints.len(),
            full.nvars,
            full.constraints.len(),
            full.nvars.saturating_sub(rcp.nvars),
            full.constraints.len().saturating_sub(rcp.constraints.len()),
        );
    }
}
