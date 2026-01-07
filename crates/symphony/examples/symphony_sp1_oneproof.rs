//! Chunked streaming proving for SP1 shrink verifier R1CS.
//!
//! This program does the following:
//! - open/build the SP1 chunk cache
//! - load all chunk matrices into memory
//! - run the canonical hetero-M streaming Π_fold prover (`prove_pi_fold_poseidon_fs`)
//!
//! Usage:
//!   SP1_R1CS=/path/to/shrink_verifier.r1cs \
//!     cargo run -p symphony --example symphony_sp1_oneproof --release
//!
//! To generate the R1CS file, run in the SP1 fork:
//!   OUT_R1CS=shrink_verifier.r1cs cargo run -p sp1-prover \
//!     --bin dump_shrink_verify_constraints --release

use std::sync::Arc;
use std::time::Instant;

use cyclotomic_rings::rings::FrogPoseidonConfig as PC;
use latticefold::commitment::AjtaiCommitmentScheme;
use stark_rings::cyclotomic_ring::models::frog_ring::RqPoly as R;
use stark_rings::PolyRing;
use stark_rings::Ring;
use symphony::rp_rgchk::RPParams;
use symphony::symphony_open::MultiAjtaiOpenVerifier;
use symphony::symphony_we_relation::check_r_cp_poseidon_fs_hetero_m_with_metrics;
use symphony::sp1_r1cs_loader::FieldFromU64;
use symphony::symphony_pifold_streaming::{
    prove_pi_fold_poseidon_fs, PiFoldStreamingConfig,
};
use symphony::symphony_sp1_r1cs::open_sp1_r1cs_chunk_cache;

/// BabyBear field element for loading R1CS.
#[derive(Debug, Clone, Copy, Default)]
struct BabyBear(u64);

const BABYBEAR_P: u64 = 0x78000001; // 2013265921

impl FieldFromU64 for BabyBear {
    fn from_canonical_u64(val: u64) -> Self {
        BabyBear(val % BABYBEAR_P)
    }
    fn as_canonical_u64(&self) -> u64 {
        self.0
    }
}

fn main() {
    let r1cs_path = std::env::var("SP1_R1CS").expect("Set SP1_R1CS=/path/to/shrink.r1cs");
    let chunk_size: usize = std::env::var("CHUNK_SIZE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1 << 20); // 1M

    let l_h: usize = std::env::var("L_H")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(256);
    let lambda_pj: usize = std::env::var("LAMBDA_PJ")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1);

    let mut cfg = PiFoldStreamingConfig::default();
    cfg.profile = std::env::var("SYMPHONY_PROFILE").ok().as_deref() == Some("1");

    println!("=========================================================");
    println!("Symphony SP1 One-Proof");
    println!("=========================================================");
    println!("  CHUNK_SIZE={chunk_size}  L_H={l_h}  LAMBDA_PJ={lambda_pj}");
    println!(
        "  parallel_feature={}  rayon_threads={}",
        cfg!(feature = "parallel"),
        rayon::current_num_threads()
    );

    let t_load = Instant::now();
    let pad_cols_to_multiple_of = l_h;
    let cache = open_sp1_r1cs_chunk_cache::<R, BabyBear>(&r1cs_path, chunk_size, pad_cols_to_multiple_of)
        .expect("Failed to open/build chunk cache");
    println!("  cache open: {:?}", t_load.elapsed());

    let num_chunks = cache.num_chunks;
    let ncols = cache.ncols;
    println!("  chunks={num_chunks} ncols={ncols}");

    // Witness (constant-coeff embedded)
    let mut witness: Vec<R> = vec![R::ZERO; ncols];
    witness[0] = R::ONE;
    let witness = Arc::new(witness);

    // Params
    let rg_params = RPParams {
        l_h,
        lambda_pj,
        k_g: 3,
        d_prime: (R::dimension() as u128) - 2,
    };

    // Commit
    const MASTER_SEED: [u8; 32] = *b"SYMPHONY_AJTAI_SEED_V1_000000000";
    let scheme_main = Arc::new(AjtaiCommitmentScheme::<R>::seeded(b"cm_f", MASTER_SEED, 8, ncols));
    let cm_main = scheme_main
        .commit_const_coeff_fast(&witness)
        .unwrap()
        .as_ref()
        .to_vec();
    let scheme_had = Arc::new(AjtaiCommitmentScheme::<R>::seeded(
        b"cfs_had_u",
        MASTER_SEED,
        8,
        3 * R::dimension(),
    ));
    let scheme_mon = Arc::new(AjtaiCommitmentScheme::<R>::seeded(
        b"cfs_mon_b",
        MASTER_SEED,
        8,
        rg_params.k_g,
    ));

    // Public statement binding (same placeholder as other bench/example)
    let public_inputs: Vec<<R as PolyRing>::BaseRing> = vec![
        <R as PolyRing>::BaseRing::from(1u128),
    ];

    // Load all mats
    let t_mats = Instant::now();
    let mut all_mats: Vec<[Arc<stark_rings_linalg::SparseMatrix<R>>; 3]> = Vec::with_capacity(num_chunks);
    for i in 0..num_chunks {
        let [m1, m2, m3] = cache.read_chunk(i).expect("read_chunk failed");
        all_mats.push([Arc::new(m1), Arc::new(m2), Arc::new(m3)]);
    }
    println!("  load all mats: {:?}", t_mats.elapsed());

    let cms_all: Vec<Vec<R>> = vec![cm_main; num_chunks];
    // Clone the Arc (refcount bump) so we can still use `witness` later for optional verification.
    let witnesses_all: Vec<Arc<Vec<R>>> = vec![witness.clone(); num_chunks];

    let t_prove = Instant::now();
    let out = prove_pi_fold_poseidon_fs::<R, PC>(
        all_mats.as_slice(),
        &cms_all,
        &witnesses_all,
        &public_inputs,
        Some(scheme_had.as_ref()),
        Some(scheme_mon.as_ref()),
        rg_params,
        &cfg,
    )
    .expect("prove failed");
    println!(
        "  prove total: {:?} (proof_bytes={})",
        t_prove.elapsed(),
        out.proof.coins.bytes.len()
    );

    // Optional: verify with transcript metrics (useful to estimate DPP-friendly cost).
    //
    // This runs the CP/aux verifier path:
    // - Poseidon-FS for challenges/bytes (records metrics + trace)
    // - verifies CP commitments `cfs_*` open to `aux`
    // - runs the core algebraic checks using `aux` (does NOT require full witness openings)
    //
    // Enable with:
    //   VERIFY=1 cargo run -p symphony --example symphony_sp1_oneproof --release
    if std::env::var("VERIFY").ok().as_deref() == Some("1") {
        let ms_ref: Vec<[&stark_rings_linalg::SparseMatrix<R>; 3]> = all_mats
            .iter()
            .map(|ms| [&*ms[0], &*ms[1], &*ms[2]])
            .collect();

        // Opening verifier for CP transcript-message commitments.
        let open_cfs = MultiAjtaiOpenVerifier::<R>::new()
            .with_scheme("cfs_had_u", (*scheme_had).clone())
            .with_scheme("cfs_mon_b", (*scheme_mon).clone());

        let t_vfy = Instant::now();
        let (_folded_out, metrics) = check_r_cp_poseidon_fs_hetero_m_with_metrics::<R, PC>(
            &ms_ref,
            &cms_all,
            &out.proof,
            &open_cfs,
            &out.cfs_had_u,
            &out.cfs_mon_b,
            &out.aux,
            &public_inputs,
        )
        .expect("VERIFY=1: CP/aux verify failed");
        println!("  verify (cp/aux): {:?}", t_vfy.elapsed());

        // Same estimator we use in other logs: rate=20 field elems; 160 bytes per perm block.
        let perms_absorb = (metrics.absorbed_elems + 19) / 20;
        let perms_squeeze_field = (metrics.squeezed_field_elems + 19) / 20;
        let perms_squeeze_bytes = (metrics.squeezed_bytes + 159) / 160;
        println!(
            "  transcript metrics: absorbed_elems={} squeezed_field_elems={} squeezed_bytes={}",
            metrics.absorbed_elems, metrics.squeezed_field_elems, metrics.squeezed_bytes
        );
        println!(
            "  est poseidon perms: ceil(absorb/20)={} + ceil(sq_field/20)={} + ceil(bytes/160)={} => {}",
            perms_absorb,
            perms_squeeze_field,
            perms_squeeze_bytes,
            perms_absorb + perms_squeeze_field + perms_squeeze_bytes
        );

        // Extra: time the Ajtai binding check for cm_f against the witness (linear work, no extra hashes).
        //
        // This is a proxy for the “bind cm_f to witness” part of the WE predicate; it’s O(ncols)
        // but avoids materializing witness openings per chunk (we reuse the same witness here).
        let t_cm = Instant::now();
        let cm_re = scheme_main
            .commit_const_coeff_fast(&witness)
            .expect("commit_const_coeff_fast")
            .as_ref()
            .to_vec();
        assert_eq!(cm_re, cms_all[0], "cm_f binding mismatch");
        println!("  ajtai cm_f recompute: {:?}", t_cm.elapsed());
    }
}

