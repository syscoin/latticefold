use ark_ff::{BigInteger, Field, Fp64, MontBackend, MontConfig, PrimeField};

use dpp::dr1cs_flpcp::{ChunkedMulCodeDr1csNpFlpcpSparse, Dr1csInstanceSparse, MulCode, TensorRsMulCode};
use dpp::{SparseVec, Theorem43Dpp};

#[derive(MontConfig)]
#[modulus = "257"]
#[generator = "3"]
pub struct F257Config;
type F257 = Fp64<MontBackend<F257Config, 1>>;

fn f_to_u64<F: PrimeField>(x: &F) -> u64 {
    x.into_bigint().to_bytes_le().get(0).copied().unwrap_or(0) as u64
}

fn chunked_tiny_dpp() -> Theorem43Dpp<F257, ChunkedMulCodeDr1csNpFlpcpSparse<F257, TensorRsMulCode<F257>>> {
    let n_total = 3usize;
    let a_row = SparseVec::new(vec![(F257::ONE, 0)]);
    let b_row = SparseVec::new(vec![(F257::ONE, 1)]);
    let c_row = SparseVec::new(vec![(F257::ONE, 2)]);
    let inst = Dr1csInstanceSparse::<F257> {
        n: n_total,
        a: vec![a_row],
        b: vec![b_row],
        c: vec![c_row],
    };
    let l_public = 1usize;
    let code = TensorRsMulCode::<F257>::new(2, 1).expect("tensor code");
    let k_block = code.dim_k();
    let blocks = {
        let mut out = Vec::new();
        let total = inst.k();
        let mut i = 0usize;
        while i < total {
            let end = usize::min(i + k_block, total);
            let mut a = inst.a[i..end].to_vec();
            let mut b = inst.b[i..end].to_vec();
            let mut c = inst.c[i..end].to_vec();
            while a.len() < k_block {
                a.push(SparseVec::new(Vec::new()));
                b.push(SparseVec::new(Vec::new()));
                c.push(SparseVec::new(Vec::new()));
            }
            out.push(Dr1csInstanceSparse { n: inst.n, a, b, c });
            i = end;
        }
        if out.is_empty() {
            out.push(inst);
        }
        out
    };
    let flpcp = ChunkedMulCodeDr1csNpFlpcpSparse::<F257, _>::new(blocks, l_public, code)
        .expect("chunked flpcp");
    Theorem43Dpp::<F257, _>::new(flpcp).expect("theorem43 new")
}

fn collect_streamed_pi(
    dpp: &Theorem43Dpp<F257, ChunkedMulCodeDr1csNpFlpcpSparse<F257, TensorRsMulCode<F257>>>,
    x: &[F257],
    z_w: &[F257],
    coins: &dpp::Theorem43Coins<F257>,
) -> Vec<F257> {
    let mut pi = Vec::new();
    dpp.prove_for_query_stream(x, z_w, coins, &mut |chunk| {
        pi.extend_from_slice(&chunk);
    })
    .expect("prove_for_query_stream");
    pi
}

#[test]
fn test_theorem43_f257_arm_prove_split_roundtrip() {
    // Tiny dR1CS over F257 with one constraint: z0 * z1 = z2.
    // Public: z0. Witness: (z1, z2).
    let dpp = chunked_tiny_dpp();

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
    assert_eq!(art.accepting_set, [F257::ONE, F257::from(2u64)]);
    assert_eq!(art.len, x.len() + dpp.proof_len());

    // Prove later using public coins only.
    let pi = collect_streamed_pi(&dpp, &x, &z_w, &art.coins);
    assert_eq!(pi.len(), dpp.proof_len());

    // Consistency check: streaming answer computation.
    let a_full = dpp.answer_for_stream(&art, &x, &pi).expect("answer_for_stream");

    eprintln!(
        "theorem43/f257: proof_len={} (m={} + 2 + (p-3)=254), q_nnz={}, a(u8)={}",
        dpp.proof_len(),
        dpp.proof_len() - 2 - (257 - 3),
        art.stats.q_nnz,
        f_to_u64(&a_full)
    );
}

