use ark_ff::Field;
use latticefold::transcript::poseidon::F257;

use dpp::dr1cs_flpcp::{Dr1csInstanceSparse, MulCode, MulCodeDr1csNpFlpcpSparse, TensorRsMulCode};
use dpp::{SparseVec, Theorem43Dpp};

fn tiny_dpp() -> Theorem43Dpp<F257, MulCodeDr1csNpFlpcpSparse<F257, TensorRsMulCode<F257>>> {
    let n_total = 3usize;
    let l_public = 1usize;
    let code = TensorRsMulCode::<F257>::new(2, 3).expect("tensor code");
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

    let flpcp = MulCodeDr1csNpFlpcpSparse::<F257, _>::new(inst, l_public, code).expect("mulcode flpcp");
    Theorem43Dpp::<F257, _>::new(flpcp).expect("theorem43 new")
}

#[test]
fn test_tensor_rs_eval_all_matches_row_stream_small() {
    // Sanity-check the separable F257 eval-all fast path against per-index row streaming.
    // Keep this tiny to keep the test fast, but exercise rank=3.
    let base_k = 3usize;
    let rank = 3usize;
    let code = TensorRsMulCode::<F257>::new(base_k, rank).expect("tensor code");
    let k = code.dim_k();
    let k_star = code.dim_k_star();
    let side = 2 * base_k - 1;
    assert_eq!(k, base_k * base_k * base_k);
    assert_eq!(k_star, side * side * side);

    // Deterministic message in u16 (0..256) then interpreted in F257.
    let y_u16: Vec<u16> = (0..k).map(|i| ((7 * i + 11) % 257) as u16).collect();
    let y_f: Vec<F257> = y_u16.iter().map(|&v| F257::from(v as u64)).collect();

    let witness_pos = code.witness_positions_star().expect("witness positions");
    assert_eq!(witness_pos.len(), k_star);
    let eval = code
        .eval_e_at_positions(&witness_pos, &y_f)
        .expect("eval_e_at_positions");
    assert_eq!(eval.len(), k_star);

    // Compare a small sample of points, including a few low-cube and some outside.
    let sample_js = [
        0usize,
        1,
        base_k - 1,
        k - 1,
        k,               // first "high" index
        k + 1,
        k_star / 2,
        k_star - 1,
    ];
    for &j in sample_js.iter() {
        let idx = witness_pos[j];
        let mut slow = F257::ZERO;
        code.row_e_stream(idx, &mut |i, c| {
            slow += c * y_f[i];
        })
        .expect("row_e_stream");
        assert_eq!(slow, eval[j], "mismatch at j={j}, idx={idx}");
    }
}

#[test]
fn test_theorem43_f257_public_coin_derivation_is_deterministic() {
    // We no longer want unit tests to depend on `dpp.arm(...)` (it computes Sq/tail coeffs as
    // "toxic waste"). The canonical WE-gate path uses only public-coin derivation + streaming.
    let dpp = tiny_dpp();
    let x = vec![F257::from(2u64)];
    let c_stmt = vec![F257::from(99u64)];

    let c0 = dpp
        .derive_public_coins_from_stmt(&c_stmt, /*block_id=*/ 0, /*rep_id=*/ 7)
        .expect("derive_public_coins_from_stmt");
    let c1 = dpp
        .derive_public_coins_from_stmt(&c_stmt, /*block_id=*/ 0, /*rep_id=*/ 7)
        .expect("derive_public_coins_from_stmt");
    assert_eq!(c0.idx, c1.idx);
    assert_eq!(c0.lambda, c1.lambda);
    assert_eq!(c0.rho, c1.rho);
    assert_eq!(c0.sigma, c1.sigma);

    // Different rep_id should change coins (overwhelmingly likely, and required for security).
    let c2 = dpp
        .derive_public_coins_from_stmt(&c_stmt, /*block_id=*/ 0, /*rep_id=*/ 8)
        .expect("derive_public_coins_from_stmt");
    let same = c2.idx == c0.idx && c2.lambda == c0.lambda && c2.rho == c0.rho && c2.sigma == c0.sigma;
    assert!(!same, "coins unexpectedly identical under different rep_id");

    // Smoke-check: these coins are usable by streaming (π0 collection should not error).
    let z_w = vec![F257::from(5u64), x[0] * F257::from(5u64)];
    let _abg = dpp
        .stream_pi0_and_collect_abg_full(&x, &z_w, &[c0.clone()], None)
        .expect("stream_pi0_and_collect_abg_full");
}
