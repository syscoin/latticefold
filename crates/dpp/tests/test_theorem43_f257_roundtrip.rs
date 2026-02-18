use ark_ff::{BigInteger, Field, Fp64, MontBackend, MontConfig, PrimeField};

use dpp::dr1cs_flpcp::{Dr1csInstanceSparse, MulCode, MulCodeDr1csNpFlpcpSparse, TensorRsMulCode};
use dpp::{SparseVec, Theorem43Dpp};

#[derive(MontConfig)]
#[modulus = "257"]
#[generator = "3"]
pub struct F257Config;
type F257 = Fp64<MontBackend<F257Config, 1>>;

fn f_to_u64<F: PrimeField>(x: &F) -> u64 {
    x.into_bigint().to_bytes_le().get(0).copied().unwrap_or(0) as u64
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

    let flpcp = MulCodeDr1csNpFlpcpSparse::<F257, _>::new(inst, l_public, code).expect("mulcode flpcp");
    Theorem43Dpp::<F257, _>::new(flpcp).expect("theorem43 new")
}

fn collect_streamed_pi0_and_tail(
    dpp: &Theorem43Dpp<F257, MulCodeDr1csNpFlpcpSparse<F257, TensorRsMulCode<F257>>>,
    x: &[F257],
    z_w: &[F257],
    coins: &dpp::Theorem43Coins<F257>,
    pi0_len: usize,
) -> (Vec<F257>, Vec<F257>) {
    let mut pi0 = Vec::with_capacity(pi0_len);
    let mut tail: Vec<F257> = Vec::new();
    let tails = dpp
        .stream_pi0_and_collect_tails(
            x,
            z_w,
            &[coins.clone()],
            Some(&mut |chunk| {
                pi0.extend_from_slice(chunk);
            }),
            &mut |ci, _ti, t| {
                if ci == 0 {
                    tail.push(*t);
                }
            },
        )
        .expect("stream_pi0_and_collect_tails");
    assert_eq!(pi0.len(), pi0_len);
    assert_eq!(tails.len(), 1);
    (pi0, tail)
}

fn collect_streamed_tail_only(
    dpp: &Theorem43Dpp<F257, MulCodeDr1csNpFlpcpSparse<F257, TensorRsMulCode<F257>>>,
    x: &[F257],
    z_w: &[F257],
    coins: &dpp::Theorem43Coins<F257>,
    _pi0_len: usize,
) -> Vec<F257> {
    let mut tail: Vec<F257> = Vec::new();
    let tails = dpp
        .stream_pi0_and_collect_tails(
            x,
            z_w,
            &[coins.clone()],
            None,
            &mut |ci, _ti, t| {
                if ci == 0 {
                    tail.push(*t);
                }
            },
        )
        .expect("stream_pi0_and_collect_tails");
    assert_eq!(tails.len(), 1);
    tail
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
fn test_theorem43_f257_arm_prove_split_roundtrip() {
    // Tiny dR1CS over F257 with one constraint: z0 * z1 = z2.
    // Public: z0. Witness: (z1, z2).
    let dpp = tiny_dpp();

    // Satisfying assignment in F257.
    let z0 = F257::from(2u64);
    let z1 = F257::from(5u64);
    let z2 = z0 * z1;
    let x = vec![z0];
    let z_w = vec![z1, z2];

    // Statement binding (C_stmt) and armer secret are toy field elements here.
    let c_stmt = vec![F257::from(99u64)];
    let armer_secret = vec![F257::from(7u64)];

    // Arm (FS) without any proof.
    let art = dpp.arm(&c_stmt, &x, &armer_secret, 0, 0).expect("arm");
    assert_eq!(art.accepting_set[1], art.accepting_set[0] + F257::ONE);
    assert_eq!(art.len, x.len() + dpp.proof_len());

    // Prove later using public coins only (canonical split proof).
    let pi0_len = dpp.proof_len() - 2 - (257 - 3);
    let (pi0, tail) = collect_streamed_pi0_and_tail(&dpp, &x, &z_w, &art.coins, pi0_len);
    let a_full = dpp
        .answer_from_pi0_and_tail(&art, &x, &pi0, &tail)
        .expect("answer_from_pi0_and_tail");
    assert!(a_full == art.accepting_set[0] || a_full == art.accepting_set[1]);

    eprintln!(
        "theorem43/f257: proof_len={} (m={} + 2 + (p-3)=254), q_nnz={}, a(u8)={}",
        dpp.proof_len(),
        dpp.proof_len() - 2 - (257 - 3),
        art.stats.q_nnz,
        f_to_u64(&a_full)
    );
}

#[test]
fn test_theorem43_f257_reuse_single_pi0_many_coin_tails() {
    // Demonstrate the intended streaming optimization:
    // - stream/compute π0 = (z_w || all w_eval blocks) once
    // - for each additional (block_id, rep_id) coin instance, stream only the coin-dependent tail
    //   and re-use the same π0 to form a full proof π = (π0 || tail).
    let dpp = tiny_dpp();

    // Satisfying assignment in F257.
    let z0 = F257::from(2u64);
    let z1 = F257::from(5u64);
    let z2 = z0 * z1;
    let x = vec![z0];
    let z_w = vec![z1, z2];

    let c_stmt = vec![F257::from(99u64)];
    let armer_secret = vec![F257::from(7u64)];

    // NOTE: this tiny fixture has only one block, so we vary rep_id only.
    let art0 = dpp.arm(&c_stmt, &x, &armer_secret, 0, 0).expect("arm(0)");
    let art1 = dpp.arm(&c_stmt, &x, &armer_secret, 0, 1).expect("arm(1)");

    // For F257, proof_len = m + 2 + (p-3) and p=257.
    let pi0_len = dpp.proof_len() - 2 - (257 - 3);
    let tail_len = dpp.proof_len() - pi0_len;
    assert_eq!(tail_len, 256, "expected tail length for F257");

    // Stream π0 once (along with tail for the first coin instance).
    let (pi0, tail0) = collect_streamed_pi0_and_tail(&dpp, &x, &z_w, &art0.coins, pi0_len);
    assert_eq!(tail0.len(), tail_len);

    // For a second coin instance, stream only the tail and reuse π0.
    let tail1 = collect_streamed_tail_only(&dpp, &x, &z_w, &art1.coins, pi0_len);
    assert_eq!(tail1.len(), tail_len);

    // Verify both split proofs are accepted for their respective arming artifacts.
    let a0 = dpp
        .answer_from_pi0_and_tail(&art0, &x, &pi0, &tail0)
        .expect("answer_from_pi0_and_tail(0)");
    assert!(a0 == art0.accepting_set[0] || a0 == art0.accepting_set[1]);

    let a1 = dpp
        .answer_from_pi0_and_tail(&art1, &x, &pi0, &tail1)
        .expect("answer_from_pi0_and_tail(1)");
    assert!(a1 == art1.accepting_set[0] || a1 == art1.accepting_set[1]);
}
