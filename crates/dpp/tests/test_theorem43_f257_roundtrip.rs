use ark_ff::{BigInteger, Field, Fp64, MontBackend, MontConfig, PrimeField};

use dpp::dr1cs_flpcp::{Dr1csInstanceSparse, RsDr1csNpFlpcpSparse};
use dpp::{BoundedFlpcpSparse, SparseVec, Theorem43Dpp};

#[derive(MontConfig)]
#[modulus = "257"]
#[generator = "3"]
pub struct F257Config;
type F257 = Fp64<MontBackend<F257Config, 1>>;

fn f_to_u64<F: PrimeField>(x: &F) -> u64 {
    x.into_bigint().to_bytes_le().get(0).copied().unwrap_or(0) as u64
}

#[test]
fn test_theorem43_f257_arm_fs_prove_split_roundtrip() {
    // Tiny dR1CS over F257 with one constraint: z0 * z1 = z2.
    // Public: z0. Witness: (z1, z2).
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
    let k_rows = inst.k();
    let ell = 2 * k_rows;
    assert!(ell <= 257, "F257 requires ell <= |F|");
    let flpcp = RsDr1csNpFlpcpSparse::<F257>::new(inst, l_public, ell);

    let dpp = Theorem43Dpp::<F257>::new(flpcp.clone()).expect("theorem43 new");

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
    let art = dpp.arm_fs(&c_stmt, &x, &armer_secret).expect("arm_fs");
    assert_eq!(art.accepting_set, [F257::ZERO, F257::ONE]);
    assert_eq!(art.len, x.len() + dpp.proof_len());

    // Prove later using public coins only.
    let pi = dpp.prove_for_query(&x, &z_w, &art.coins).expect("prove_for_query");
    assert_eq!(pi.len(), dpp.proof_len());

    // Consistency check: split_query identity.
    let (q_x, q_pi) = art.split_query(x.len(), pi.len()).expect("split_query");
    let a_full = art.answer_for(&x, &pi).expect("answer_for");
    let a_split = q_x.dot(&x) + q_pi.dot(&pi);
    assert_eq!(a_full, a_split);

    eprintln!(
        "theorem43/f257: proof_len={} (m={} + 2 + (p-3)=254), q_nnz={}, a(u8)={}",
        dpp.proof_len(),
        flpcp.m(),
        art.stats.q_nnz,
        f_to_u64(&a_full)
    );
}

