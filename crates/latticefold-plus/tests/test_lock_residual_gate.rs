#![cfg(feature = "we_gate")]

use ark_ff::Field;
use latticefold::transcript::poseidon::F257;
use latticefold_plus::lockable_ringlwe::{
    arm_ringlwe_lock, eval_err_gate_mod257_u16, AnchorBasisHints, BranchHints, ErrGateHints, PackedF257Block64,
    RingLweParams,
};
use rand::{RngCore, SeedableRng};
use rand_chacha::ChaCha20Rng;

fn dense_block_with_single(pos: usize, v: u16) -> PackedF257Block64 {
    assert!(pos < 64);
    assert!(v < 257);
    let mut vals = [0u8; 64];
    let mut mask: u64 = 0;
    if v == 256 {
        mask |= 1u64 << pos;
        vals[pos] = 0;
    } else {
        vals[pos] = (v & 0xFF) as u8;
    }
    PackedF257Block64::Dense {
        vals,
        is256_mask: mask,
    }
}

#[test]
fn test_anchor_sparse_decap_and_gate_kills() {
    let mut rng = ChaCha20Rng::from_seed([7u8; 32]);

    // Minimal lock: one π coordinate (pi_len=1), q touches only that coordinate.
    let c_stmt: Vec<F257> = vec![];
    let params = RingLweParams::default();
    let offset = F257::ZERO;
    let anchor_block_id: u32 = 0;
    let rep_id: u64 = 123;
    let poison_blocks: u32 = 1;
    let x_len = 0usize;
    let pi_len = 1usize;

    // Accepting set (shifted): choose a0!=a1 and avoid 0.
    let a0 = F257::from(5u64);
    let a1 = F257::from(9u64);
    let accepting_set_shifted = [a0, a1];

    // Public coins are only used for key derivation domain-binding in this lock layer.
    let coins = dpp::theorem43::Theorem43Coins::<F257> {
        idx: 0,
        lambda: F257::from(3u64),
        rho: F257::from(7u64),
        sigma: F257::from(11u64),
        c_hit: F257::from(1u64),
    };

    // Sparse query blocks: q[0]=1.
    let q_blocks = vec![(0usize, dense_block_with_single(0, 1u16))];
    let basis_x_dots = [F257::from(0u64); 3];
    let empty_basis = AnchorBasisHints {
        alpha: BranchHints { hint_blocks_sparse: vec![] },
        beta: BranchHints { hint_blocks_sparse: vec![] },
        gamma: BranchHints { hint_blocks_sparse: vec![] },
    };

    // Gate mixes: one mix with weight 1 on err[0]. The armer scales mixes by secret s, but the
    // gate remains correct because (s*z)^256 == z^256 for s!=0 in F257.
    let gate_hints = ErrGateHints {
        k: 1,
        blocks_per_mix: 1,
        mixes: vec![dense_block_with_single(0, 1u16)],
    };

    // Payload.
    let mut payload = [0u8; 32];
    rng.fill_bytes(&mut payload);

    let lock = arm_ringlwe_lock::<F257>(
        c_stmt,
        accepting_set_shifted,
        anchor_block_id,
        rep_id,
        None,
        poison_blocks,
        coins,
        offset,
        basis_x_dots,
        x_len,
        pi_len,
        [0u8; 32],
        q_blocks,
        empty_basis,
        gate_hints,
        params,
        payload.as_slice(),
        &mut rng,
    )
    .expect("arm_ringlwe_lock");

    // SAT-like residuals: err == 0 => gate opens.
    let errs_sat = vec![0u16];
    let g_sat = eval_err_gate_mod257_u16(&lock.err_gate_hints, errs_sat.as_slice()).unwrap();
    assert_eq!(g_sat, 1u16);

    // UNSAT-like residuals: err[0] == 1 => gate closes deterministically (due to unit weight).
    let errs_unsat = vec![1u16];
    let g_unsat = eval_err_gate_mod257_u16(&lock.err_gate_hints, errs_unsat.as_slice()).unwrap();
    assert_eq!(g_unsat, 0u16);

    // Construct a proof stream π with π[0]=a0 so one candidate decrypts correctly (when gate=1).
    let x: [F257; 0] = [];
    let mut st = lock.decap_state(&x).unwrap();
    st.absorb_chunk(&[a0]).unwrap();
    let plains = st.finish_decrypt_candidates_with_gate(g_sat, None, None).unwrap();
    assert_eq!(plains.len(), 256);
    let sat_matches = plains
        .iter()
        .filter(|p| p.as_slice() == payload.as_slice())
        .count();
    assert_eq!(sat_matches, 1, "SAT should yield exactly one payload match");

    // With g=0, we still return a fixed-shape candidate set (no local reject oracle).
    let mut st2 = lock.decap_state(&x).unwrap();
    st2.absorb_chunk(&[a0]).unwrap();
    let plains2 = st2.finish_decrypt_candidates_with_gate(g_unsat, None, None).unwrap();
    assert_eq!(plains2.len(), 256);
    let unsat_matches = plains2
        .iter()
        .filter(|p| p.as_slice() == payload.as_slice())
        .count();
    assert_eq!(unsat_matches, 1, "payload remains present but is not locally recognizable");
}

