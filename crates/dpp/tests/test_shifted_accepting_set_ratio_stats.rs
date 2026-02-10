use ark_ff::{BigInteger, Field, Fp64, MontBackend, MontConfig, PrimeField, Zero};

use dpp::dr1cs_flpcp::{Dr1csInstanceSparse, MulCode, MulCodeDr1csNpFlpcpSparse, TensorRsMulCode};
use dpp::{SparseVec, Theorem43Dpp};
use rayon::prelude::*;

fn env_u64(name: &str, default: u64) -> u64 {
    std::env::var(name).ok().and_then(|s| s.parse().ok()).unwrap_or(default)
}

#[derive(MontConfig)]
#[modulus = "257"]
#[generator = "3"]
pub struct F257Config;
type F257 = Fp64<MontBackend<F257Config, 1>>;

fn f_to_u16<F: PrimeField>(x: &F) -> u16 {
    // For F257 this fits in one byte, but keep it generic.
    let bytes = x.into_bigint().to_bytes_le();
    let b0 = bytes.get(0).copied().unwrap_or(0) as u16;
    let b1 = bytes.get(1).copied().unwrap_or(0) as u16;
    (b0 | (b1 << 8)) % 257
}

fn tiny_dpp() -> Theorem43Dpp<F257, MulCodeDr1csNpFlpcpSparse<F257, TensorRsMulCode<F257>>> {
    let n_total = 3usize;
    let l_public = 1usize;
    let code = TensorRsMulCode::<F257>::new(2, 1).expect("tensor code");
    let k = code.dim_k();

    // Tiny dR1CS over F257 with one constraint: z0 * z1 = z2.
    // Pad with empty constraints up to `k` so it can use the MulCode backend directly.
    let mut a = vec![SparseVec::new(vec![(F257::ONE, 0)])];
    let mut b = vec![SparseVec::new(vec![(F257::ONE, 1)])];
    let mut c = vec![SparseVec::new(vec![(F257::ONE, 2)])];
    while a.len() < k {
        a.push(SparseVec::new(Vec::new()));
        b.push(SparseVec::new(Vec::new()));
        c.push(SparseVec::new(Vec::new()));
    }
    let inst = Dr1csInstanceSparse::<F257> { n: n_total, a, b, c };
    let flpcp =
        MulCodeDr1csNpFlpcpSparse::<F257, _>::new(inst, l_public, code).expect("mulcode flpcp");
    Theorem43Dpp::<F257, _>::new(flpcp).expect("theorem43 new")
}

/// Compute the shifted accepting-set ratio class for a given `rep_id`:
///   A' = {1 - off, 2 - off},  r = (2-off)/(1-off) in F257*
/// We return a canonical representative up to inversion: min(r, r^{-1}) in integer form (1..256).
fn ratio_class_for_rep(
    dpp: &Theorem43Dpp<F257, MulCodeDr1csNpFlpcpSparse<F257, TensorRsMulCode<F257>>>,
    c_stmt: &[F257],
    x: &[F257],
    armer_secret: &[F257],
    rep_id: u64,
) -> Option<u16> {
    let art = dpp.arm(c_stmt, x, armer_secret, 0, rep_id).expect("arm");
    let mut scratch = dpp.query_scratch();
    let off = dpp
        .stream_query_terms_for_pi(x, &art.coins, &art.coeffs, &mut scratch, &mut |_idx, _c| {})
        .expect("offset");

    let a0 = F257::ONE - off;
    let a1 = F257::from(2u64) - off;
    // In the production lock path, we resample `rep_id` if shifted contains 0.
    // For stats, treat this as a "reject" event and skip.
    if a0.is_zero() || a1.is_zero() {
        return None;
    }

    let r = a1 * a0.inverse().expect("inv a0");
    assert!(!r.is_zero(), "ratio should be nonzero");
    let rinv = r.inverse().expect("inv r");

    let ru = f_to_u16(&r);
    let ri = f_to_u16(&rinv);
    // Both should be in 1..256.
    assert!(ru != 0 && ri != 0);
    Some(ru.min(ri))
}

#[test]
fn test_shifted_accepting_set_ratio_varies_with_rep_id() {
    let dpp = tiny_dpp();
    eprintln!(
        "[ratio-stats] rayon threads = {}",
        rayon::current_num_threads().max(1)
    );

    // Same tiny satisfying assignment as in the roundtrip test.
    let z0 = F257::from(2u64);
    let x = vec![z0];

    let c_stmt = vec![F257::from(99u64)];
    let armer_secret = vec![F257::from(7u64)];

    // Sample a moderate range of rep_id values and ensure the ratio classes aren't constant.
    let n = env_u64("DPP_RATIO_STATS_N", 2048);
    let results: Vec<Option<u16>> = (0..n)
        .into_par_iter()
        .map(|rep_id| ratio_class_for_rep(&dpp, &c_stmt, &x, &armer_secret, rep_id))
        .collect();
    let mut seen = std::collections::BTreeSet::<u16>::new();
    let mut rejects = 0u64;
    for r in results {
        if let Some(rc) = r {
            seen.insert(rc);
        } else {
            rejects += 1;
        }
    }

    eprintln!(
        "[ratio-stats] N={} rejects={} distinct_ratio_classes={}",
        n,
        rejects,
        seen.len()
    );

    // Extremely weak lower bound: if this fails, ratios are basically fixed/degenerate.
    assert!(
        seen.len() >= 128,
        "ratio class set too small ({} distinct over {} reps; rejects={})",
        seen.len(),
        n,
        rejects
    );
}

#[test]
fn test_ratio_class_collision_rate_for_r3_is_small() {
    let dpp = tiny_dpp();
    eprintln!(
        "[ratio-coll] rayon threads = {}",
        rayon::current_num_threads().max(1)
    );

    let z0 = F257::from(2u64);
    let x = vec![z0];
    let c_stmt = vec![F257::from(99u64)];
    let armer_secret = vec![F257::from(7u64)];

    // Group size R=3. We model the "bad ambiguity" condition as:
    // all three repetitions share the same ratio class up to inversion.
    //
    // If ratio classes are close to uniform/independent, this should happen with prob ~1/256^2.
    const R: u64 = 3;
    let groups: u64 = env_u64("DPP_RATIO_STATS_GROUPS", 20000);
    let bad = std::sync::atomic::AtomicU64::new(0);
    let skipped = std::sync::atomic::AtomicU64::new(0);
    (0..groups).into_par_iter().for_each(|g| {
        let base = g * R;
        let c0 = ratio_class_for_rep(&dpp, &c_stmt, &x, &armer_secret, base + 0);
        let c1 = ratio_class_for_rep(&dpp, &c_stmt, &x, &armer_secret, base + 1);
        let c2 = ratio_class_for_rep(&dpp, &c_stmt, &x, &armer_secret, base + 2);
        if let (Some(c0), Some(c1), Some(c2)) = (c0, c1, c2) {
            if c0 == c1 && c1 == c2 {
                bad.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
        } else {
            skipped.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
    });
    let bad = bad.load(std::sync::atomic::Ordering::Relaxed);
    let skipped = skipped.load(std::sync::atomic::Ordering::Relaxed);

    eprintln!(
        "[ratio-coll] R={} groups={} bad={} skipped={} bad_rate={:.6e}",
        R,
        groups,
        bad,
        skipped,
        (bad as f64) / (groups.max(1) as f64)
    );

    // Loose safety bound: if this is large, we don't get disambiguation from R=3.
    //
    // Expected under near-uniform ratio classes: ~groups/65536 ≈ 0.3 for groups=20000.
    // We allow up to 20 to avoid any flakiness due to deterministic structure.
    assert!(
        bad <= 20,
        "too many r-class collisions for R=3: bad={bad} / {groups} (skipped={skipped})"
    );
}

