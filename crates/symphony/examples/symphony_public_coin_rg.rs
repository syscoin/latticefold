//! CP-style Symphony Pi_rg demo (experimental).
//!
//! This demo shows the recommended CP-style split:
//! - Derive Fiat-Shamir coins externally (Poseidon transcript), recording the coin stream.
//! - Replay verification using explicit coins (FixedTranscript) to keep the predicate hash-free.

use ark_std::One;
use cyclotomic_rings::rings::FrogPoseidonConfig as PC;
use latticefold_plus::{
    symphony_cm::{prove_rg_fs, verify_rg, verify_rg_fs_bound},
    rp_rgchk::RPParams,
};
use stark_rings::cyclotomic_ring::models::frog_ring::RqPoly as R;
use stark_rings::PolyRing;
use stark_rings_linalg::Matrix;

fn main() {
    let n = 1 << 10;
    let f = vec![R::one(); n];
    let a = Matrix::<R>::rand(&mut ark_std::test_rng(), 2, n);

    let params = RPParams {
        l_h: 64,
        lambda_pj: 32,
        k_g: 4,
        d_prime: (R::dimension() as u128) / 2,
    };

    let (stmt, proof) = prove_rg_fs::<R, PC>(f.clone(), &a, params);
    let ok_replay = verify_rg(&a, &f, &stmt, &proof);
    let ok_bound = verify_rg_fs_bound::<R, PC>(&a, &f, &stmt, &proof);
    println!("verify_rg (replay): {}", ok_replay);
    println!("verify_rg_fs_bound (recompute FS coins): {}", ok_bound);
}

