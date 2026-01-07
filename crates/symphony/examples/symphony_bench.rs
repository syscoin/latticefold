//! Micro-benchmark for Symphony Π_fold / WE-facing `R_cp` checks.
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
        rp_rgchk::RPParams,
        symphony_open::MultiAjtaiOpenVerifier,
        symphony_pifold_streaming::{prove_pi_fold_poseidon_fs, PiFoldStreamingConfig},
        symphony_we_relation::{check_r_cp_poseidon_fs, TrivialRo},
    };
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

    // Ajtai commitment for cm_f.
    const MASTER_SEED: [u8; 32] = *b"SYMPHONY_AJTAI_SEED_V1_000000000";
    let scheme_f = AjtaiCommitmentScheme::<R>::seeded(b"cm_f", MASTER_SEED, 8, n);

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
            let cm = scheme_f.commit(&f).unwrap().as_ref().to_vec();
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

        let cfg = PiFoldStreamingConfig::default();
        let prove_start = Instant::now();
        let out = prove_pi_fold_poseidon_fs::<R, PC>(
            ms.as_slice(),
            &cm_f,
            &witnesses,
            &public_inputs,
            Some(&scheme_had),
            Some(&scheme_mon),
            rg_params.clone(),
            &cfg,
        )
        .unwrap();
        let prove_time = prove_start.elapsed();

        let open = MultiAjtaiOpenVerifier::<R>::new()
            .with_scheme("cfs_had_u", scheme_had)
            .with_scheme("cfs_mon_b", scheme_mon);

        let verify_start = Instant::now();
        let _out_folded = check_r_cp_poseidon_fs::<R, PC>(
            [&m1, &m2, &m3],
            &cm_f,
            &out.proof,
            &open,
            &out.cfs_had_u,
            &out.cfs_mon_b,
            &out.aux,
            &public_inputs,
        )
        .unwrap();
        let verify_time = verify_start.elapsed();

        // Sanity: `TrivialRo` exists just to show how R_WE would be checked; we don't benchmark it.
        let _: Result<(), String> = <TrivialRo as symphony::symphony_we_relation::ReducedRelation<R>>::check(
            &_out_folded,
            &(),
        );

        println!(
            "ℓ={ell:>2} | Π_fold prove: {:>8.3}s | R_cp verify: {:>8.3}s | proof bytes: {}",
            prove_time.as_secs_f64(),
            verify_time.as_secs_f64(),
            out.proof.coins.bytes.len()
        );
    }
}
