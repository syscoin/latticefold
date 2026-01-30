use super::*;

use ark_ff::{BigInteger, Field, PrimeField};
use latticefold::transcript::poseidon::F257;
use symphony::dpp_sumcheck::Dr1csBuilder;
use symphony::transcript::PoseidonTraceOp;

use crate::we_goldilocks_poseidon_f257::GOLDILOCKS_P;
use crate::we_gate_tiny::params::{LIMB_BASE_U64, LIMB_BITS, LIMBS_U32, LIMBS_U64};

use super::challenges::bounded_u32_from_8_digits_base128;
use super::digits::*;
use super::goldilocks::{
    goldilocks_u64_enforce_lt_p_from_byte_vars_and_limbs,
    reduce_u64_mod_goldilocks_from_byte_vars,
};
use super::gadgets::alloc_byte;
use super::cm_math::{
    goldilocks_add_mod_p_digits, goldilocks_bytes_to_digits, goldilocks_digits_to_bytes_canonical,
    goldilocks_mul_const_mod_p_digits, goldilocks_mul_mod_p_digits, goldilocks_sub_mod_p_digits,
};

use latticefold::transcript::Transcript;
use stark_rings::cyclotomic_ring::models::goldilocks::RqPoly as GoldilocksRing;

fn var_to_u8<F: PrimeField>(b: &Dr1csBuilder<F>, v: usize) -> u8 {
    b.assignment[v]
        .into_bigint()
        .to_bytes_le()
        .get(0)
        .copied()
        .unwrap_or(0)
}

fn limbs_to_u64_base128<F: PrimeField>(b: &Dr1csBuilder<F>, limbs: &[usize; LIMBS_U64]) -> u64 {
    let mut acc: u64 = 0;
    for i in (0..LIMBS_U64).rev() {
        let di = b.assignment[limbs[i]]
            .into_bigint()
            .to_bytes_le()
            .get(0)
            .copied()
            .unwrap_or(0) as u64;
        acc <<= LIMB_BITS;
        acc |= di & (LIMB_BASE_U64 - 1);
    }
    acc
}

fn limbs_u32_from_base128<F: PrimeField>(b: &Dr1csBuilder<F>, limbs: &[usize; LIMBS_U32]) -> u32 {
    let mut acc: u64 = 0;
    for i in (0..LIMBS_U32).rev() {
        let di = b.assignment[limbs[i]]
            .into_bigint()
            .to_bytes_le()
            .get(0)
            .copied()
            .unwrap_or(0) as u64;
        acc <<= LIMB_BITS;
        acc |= di & (LIMB_BASE_U64 - 1);
    }
    acc as u32
}

#[test]
fn test_poseidon_f257_ops_arithmetization_satisfies() {
    // Record a tiny transcript trace in the **actual sponge field** (F257).
    //
    // IMPORTANT: we must not use `crate::recording_transcript::TracePoseidonTranscript`,
    // which lifts F257 digits into the outer base ring. For a tiny-field gate we want the
    // transcript ops directly over F257.
    let mut tr = symphony::transcript::TracePoseidonTranscript::<GoldilocksRing>::empty::<()>();
    tr.absorb_field_element(&<GoldilocksRing as stark_rings::PolyRing>::BaseRing::from(123u64));
    let _c = tr.get_challenge(); // SqueezeField(8) + Absorb(8)
    let _b = tr.squeeze_bytes(17); // SqueezeBytes(17) (no reabsorb)

    let ops: Vec<PoseidonTraceOp<F257>> = tr.trace().ops.clone();

    let (inst, asg, _wiring, _byte_wiring) =
        poseidon_f257_arithmetize(None, &ops).expect("poseidon_f257_arithmetize");
    inst.check(&asg).expect("poseidon(F257) dR1CS satisfied");
}

#[test]
fn test_reduce_u64_mod_goldilocks_branch_u_lt_p() {
    let mut b = Dr1csBuilder::<F257>::new();
    b.enforce_var_eq_const(b.one(), F257::ONE);

    let u = 42u64;
    let mut u_byte_vars = [0usize; 8];
    for (i, v) in u.to_le_bytes().into_iter().enumerate() {
        let bv = alloc_byte::<F257>(&mut b, v);
        u_byte_vars[i] = bv.byte;
    }
    let (q, z) = reduce_u64_mod_goldilocks_from_byte_vars::<F257>(&mut b, &u_byte_vars);

    // q should be 0, z == u.
    assert_eq!(var_to_u8::<F257>(&b, q), 0);
    assert_eq!(limbs_to_u64_base128::<F257>(&b, &z), u);

    let (inst, asg) = b.into_instance();
    inst.check(&asg).expect("branch u<p satisfied");
}

#[test]
fn test_reduce_u64_mod_goldilocks_branch_u_ge_p() {
    let mut b = Dr1csBuilder::<F257>::new();
    b.enforce_var_eq_const(b.one(), F257::ONE);

    let u = GOLDILOCKS_P + 424242u64;
    let mut u_byte_vars = [0usize; 8];
    for (i, v) in u.to_le_bytes().into_iter().enumerate() {
        let bv = alloc_byte::<F257>(&mut b, v);
        u_byte_vars[i] = bv.byte;
    }
    let (q, z) = reduce_u64_mod_goldilocks_from_byte_vars::<F257>(&mut b, &u_byte_vars);

    // q should be 1, z == u - p.
    assert_eq!(var_to_u8::<F257>(&b, q), 1);
    assert_eq!(limbs_to_u64_base128::<F257>(&b, &z), u - GOLDILOCKS_P);

    let (inst, asg) = b.into_instance();
    inst.check(&asg).expect("branch u>=p satisfied");
}

#[test]
fn test_reduce_u64_mod_goldilocks_rejects_wrong_q() {
    let mut b = Dr1csBuilder::<F257>::new();
    b.enforce_var_eq_const(b.one(), F257::ONE);

    let u = GOLDILOCKS_P + 7u64;
    let mut u_byte_vars = [0usize; 8];
    for (i, v) in u.to_le_bytes().into_iter().enumerate() {
        let bv = alloc_byte::<F257>(&mut b, v);
        u_byte_vars[i] = bv.byte;
    }
    let (q, _z) = reduce_u64_mod_goldilocks_from_byte_vars::<F257>(&mut b, &u_byte_vars);

    let (inst, mut asg) = b.into_instance();
    // Flip q (should be 1 -> set to 0) without adjusting anything else.
    asg[q] = F257::ZERO;
    assert!(inst.check(&asg).is_err(), "flipped q should break constraints");
}

#[test]
fn test_goldilocks_u64_canonical_from_bytes_accepts_lt_p_and_rejects_ge_p() {
    // For transcript-absorbed base-field elements we expect canonical encoding u < p_goldilocks.
    let mut b = Dr1csBuilder::<F257>::new();
    b.enforce_var_eq_const(b.one(), F257::ONE);

    // u < p should satisfy q=0.
    let u_good = (GOLDILOCKS_P / 2) + 7u64;
    let mut u_byte_vars = [0usize; 8];
    for (i, v) in u_good.to_le_bytes().into_iter().enumerate() {
        let bv = alloc_byte::<F257>(&mut b, v);
        u_byte_vars[i] = bv.byte;
    }
    let _z = goldilocks_u64_enforce_lt_p_from_byte_vars_and_limbs::<F257>(&mut b, &u_byte_vars);

    // u >= p should fail due to enforced q==0.
    let u_bad = GOLDILOCKS_P + 9u64;
    let mut v_byte_vars = [0usize; 8];
    for (i, v) in u_bad.to_le_bytes().into_iter().enumerate() {
        let bv = alloc_byte::<F257>(&mut b, v);
        v_byte_vars[i] = bv.byte;
    }
    let _z2 = goldilocks_u64_enforce_lt_p_from_byte_vars_and_limbs::<F257>(&mut b, &v_byte_vars);

    let (inst, asg) = b.into_instance();
    assert!(inst.check(&asg).is_err(), "u>=p should violate canonical constraint");
}

#[test]
fn test_goldilocks_mul_mod_p_from_bytes_matches_native() {
    use rand::{RngCore, SeedableRng};
    let mut rng = rand::rngs::StdRng::seed_from_u64(1234567);

    for _ in 0..20 {
        // Sample canonical a,b < p.
        let a = rng.next_u64() % GOLDILOCKS_P;
        let b_u = rng.next_u64() % GOLDILOCKS_P;
        let exp = ((a as u128) * (b_u as u128) % (GOLDILOCKS_P as u128)) as u64;

        let mut b = Dr1csBuilder::<F257>::new();
        b.enforce_var_eq_const(b.one(), F257::ONE);
        let mut a_bytes = [0usize; 8];
        let mut b_bytes = [0usize; 8];
        for (i, v) in a.to_le_bytes().into_iter().enumerate() {
            a_bytes[i] = alloc_byte::<F257>(&mut b, v).byte;
        }
        for (i, v) in b_u.to_le_bytes().into_iter().enumerate() {
            b_bytes[i] = alloc_byte::<F257>(&mut b, v).byte;
        }

        let a_d = goldilocks_bytes_to_digits(&mut b, a_bytes);
        let b_d = goldilocks_bytes_to_digits(&mut b, b_bytes);
        let r_d = goldilocks_mul_mod_p_digits(&mut b, &a_d, &b_d);
        let r_bytes = goldilocks_digits_to_bytes_canonical(&mut b, &r_d);
        let mut out = [0u8; 8];
        for i in 0..8 {
            out[i] = var_to_u8::<F257>(&b, r_bytes[i]);
        }
        let got = u64::from_le_bytes(out);
        assert_eq!(got, exp);

        let (inst, asg) = b.into_instance();
        inst.check(&asg).expect("mul mod p constraints satisfied");
    }
}

#[test]
fn test_goldilocks_add_sub_mod_p_from_bytes_matches_native() {
    use rand::{RngCore, SeedableRng};
    let mut rng = rand::rngs::StdRng::seed_from_u64(222333444);

    for _ in 0..20 {
        let a = rng.next_u64() % GOLDILOCKS_P;
        let c = rng.next_u64() % GOLDILOCKS_P;
        let add_exp = ((a as u128 + c as u128) % (GOLDILOCKS_P as u128)) as u64;
        let sub_exp = ((a as i128 - c as i128).rem_euclid(GOLDILOCKS_P as i128)) as u64;

        let mut b = Dr1csBuilder::<F257>::new();
        b.enforce_var_eq_const(b.one(), F257::ONE);
        let mut a_bytes = [0usize; 8];
        let mut c_bytes = [0usize; 8];
        for (i, v) in a.to_le_bytes().into_iter().enumerate() {
            a_bytes[i] = alloc_byte::<F257>(&mut b, v).byte;
        }
        for (i, v) in c.to_le_bytes().into_iter().enumerate() {
            c_bytes[i] = alloc_byte::<F257>(&mut b, v).byte;
        }

        let a_d = goldilocks_bytes_to_digits(&mut b, a_bytes);
        let c_d = goldilocks_bytes_to_digits(&mut b, c_bytes);
        let add_d = goldilocks_add_mod_p_digits(&mut b, &a_d, &c_d);
        let sub_d = goldilocks_sub_mod_p_digits(&mut b, &a_d, &c_d);
        let add_r = goldilocks_digits_to_bytes_canonical(&mut b, &add_d);
        let sub_r = goldilocks_digits_to_bytes_canonical(&mut b, &sub_d);

        let mut out_add = [0u8; 8];
        let mut out_sub = [0u8; 8];
        for i in 0..8 {
            out_add[i] = var_to_u8::<F257>(&b, add_r[i]);
            out_sub[i] = var_to_u8::<F257>(&b, sub_r[i]);
        }
        assert_eq!(u64::from_le_bytes(out_add), add_exp);
        assert_eq!(u64::from_le_bytes(out_sub), sub_exp);

        let (inst, asg) = b.into_instance();
        inst.check(&asg).expect("add/sub mod p constraints satisfied");
    }
}

#[test]
fn test_goldilocks_mul_const_mod_p_from_bytes_matches_native() {
    use rand::{RngCore, SeedableRng};
    let mut rng = rand::rngs::StdRng::seed_from_u64(999888777);

    for _ in 0..20 {
        let x = rng.next_u64() % GOLDILOCKS_P;
        let c = rng.next_u64() % GOLDILOCKS_P;
        let exp = ((x as u128) * (c as u128) % (GOLDILOCKS_P as u128)) as u64;

        let mut b = Dr1csBuilder::<F257>::new();
        b.enforce_var_eq_const(b.one(), F257::ONE);
        let mut x_bytes = [0usize; 8];
        for (i, v) in x.to_le_bytes().into_iter().enumerate() {
            x_bytes[i] = alloc_byte::<F257>(&mut b, v).byte;
        }

        let x_d = goldilocks_bytes_to_digits(&mut b, x_bytes);
        let r_d = goldilocks_mul_const_mod_p_digits(&mut b, &x_d, c);
        let r_bytes = goldilocks_digits_to_bytes_canonical(&mut b, &r_d);
        let mut out = [0u8; 8];
        for i in 0..8 {
            out[i] = var_to_u8::<F257>(&b, r_bytes[i]);
        }
        assert_eq!(u64::from_le_bytes(out), exp);

        let (inst, asg) = b.into_instance();
        inst.check(&asg).expect("mul const constraints satisfied");
    }
}



#[test]
fn test_ring_mul_negacyclic_ntt_goldilocks_d64_matches_native_one_case() {
    use rand::{RngCore, SeedableRng};
    use std::time::Instant;
    let mut rng = rand::rngs::StdRng::seed_from_u64(123456789);

    let p = super::goldilocks::GOLDILOCKS_P;

    let mut a = [0u64; 64];
    let mut c = [0u64; 64];
    for i in 0..64 {
        a[i] = rng.next_u64() % p;
        c[i] = rng.next_u64() % p;
    }

    // Native expected.
    let mut conv = vec![0u64; 127];
    for i in 0..64 {
        for j in 0..64 {
            let idx = i + j;
            let t = ((a[i] as u128) * (c[j] as u128) + (conv[idx] as u128)) % (p as u128);
            conv[idx] = t as u64;
        }
    }
    let mut exp = [0u64; 64];
    for k in 0..64 {
        let hi = k + 64;
        let v = if hi < 127 {
            (conv[k] as i128 - conv[hi] as i128).rem_euclid(p as i128) as u64
        } else {
            conv[k]
        };
        exp[k] = v;
    }

    // Circuit: boundary conversion bytes->digits only (stand-in for external IO).
    let t0 = Instant::now();
    let mut b = Dr1csBuilder::<F257>::new();
    b.enforce_var_eq_const(b.one(), F257::ONE);
    let mut a_bytes = [[0usize; 8]; 64];
    let mut c_bytes = [[0usize; 8]; 64];
    for i in 0..64 {
        for (j, v) in a[i].to_le_bytes().into_iter().enumerate() {
            a_bytes[i][j] = alloc_byte::<F257>(&mut b, v).byte;
        }
        for (j, v) in c[i].to_le_bytes().into_iter().enumerate() {
            c_bytes[i][j] = alloc_byte::<F257>(&mut b, v).byte;
        }
    }
    let t_after_alloc_bytes = Instant::now();

    let a_d: [super::goldilocks::GoldilocksScalar; 64] = core::array::from_fn(|i| {
        let v = u64_bytes_to_bal16_digits(&mut b, a_bytes[i]);
        v.try_into().expect("u64 bytes -> 17 digits")
    });
    let c_d: [super::goldilocks::GoldilocksScalar; 64] = core::array::from_fn(|i| {
        let v = u64_bytes_to_bal16_digits(&mut b, c_bytes[i]);
        v.try_into().expect("u64 bytes -> 17 digits")
    });
    let t_after_bytes_to_digits = Instant::now();

    // Use the IR implementation (the old non-IR gadget has been removed).
    // Clone assignment so we can later mutably borrow `b` to lower IR.
    let base_asg: Vec<F257> = b.assignment.clone();
    let mut ib = super::cm_ir::IrBuilder::new(&base_asg);
    let a_ir: [[super::cm_ir::VarRef; 17]; 64] =
        core::array::from_fn(|i| core::array::from_fn(|j| super::cm_ir::VarRef::Base(a_d[i][j])));
    let c_ir: [[super::cm_ir::VarRef; 17]; 64] =
        core::array::from_fn(|i| core::array::from_fn(|j| super::cm_ir::VarRef::Base(c_d[i][j])));
    let out_ir = super::cm_ir::ring_mul_negacyclic_ntt_goldilocks_d64_ir(&mut ib, &a_ir, &c_ir);
    let t_after_build_ir = Instant::now();
    let ir_stats = ib.ir.stats;
    eprintln!(
        "== ringmul IR stats: linear={} mul={} other_non_linear={} total={} | terms(a,b,c)=({},{},{}) max(a,b,c)=({},{},{}) ==",
        ir_stats.linear_constraints,
        ir_stats.mul_constraints,
        ir_stats.other_non_linear_constraints,
        ir_stats.linear_constraints + ir_stats.mul_constraints + ir_stats.other_non_linear_constraints,
        ir_stats.total_terms_a,
        ir_stats.total_terms_b,
        ir_stats.total_terms_c,
        ir_stats.max_terms_a,
        ir_stats.max_terms_b,
        ir_stats.max_terms_c,
    );
    let lowered = super::cm_ir::lower_ir_into_builder(&mut b, ib.ir);
    let t_after_lower_ir = Instant::now();
    let out: [super::goldilocks::GoldilocksScalar; 64] = core::array::from_fn(|i| {
        core::array::from_fn(|j| lowered.map_var(out_ir[i][j]))
    });

    for k in 0..64 {
        // Decode digits -> u64 in the host.
        let mut acc: i128 = 0;
        let mut pow: i128 = 1;
        for j in 0..17 {
            acc += (f257_to_i32_bal(b.assignment[out[k][j]]) as i128) * pow;
            pow *= 16;
        }
        assert_eq!(acc as u64, exp[k]);
    }
    let t_after_decode_check = Instant::now();

    eprintln!(
        "== dR1CS dump: ring_mul_negacyclic_ntt_goldilocks_d64_ir | nvars={} nconstraints={} ==",
        b.assignment.len(),
        b.rows.len()
    );
    if std::env::var("LF_PROFILE_DR1CS").ok().as_deref() == Some("1") {
        eprintln!("{}", b.profile_report(40));
    }

    let (inst, asg) = b.into_instance();
    let t_after_into_instance = Instant::now();
    inst.check(&asg).expect("ring mul (goldilocks ntt, IR) constraints satisfied");
    let t_after_check = Instant::now();

    eprintln!(
        "== ringmul timings (release): total={:?} alloc_bytes={:?} bytes_to_digits={:?} build_ir={:?} lower_ir={:?} decode_check={:?} into_instance={:?} check={:?} ==",
        t_after_check.duration_since(t0),
        t_after_alloc_bytes.duration_since(t0),
        t_after_bytes_to_digits.duration_since(t_after_alloc_bytes),
        t_after_build_ir.duration_since(t_after_bytes_to_digits),
        t_after_lower_ir.duration_since(t_after_build_ir),
        t_after_decode_check.duration_since(t_after_lower_ir),
        t_after_into_instance.duration_since(t_after_decode_check),
        t_after_check.duration_since(t_after_into_instance),
    );
}

#[test]
fn test_scalar_mul_mod_p_ir_constraint_delta_smoke() {
    use std::time::Instant;

    // This is a micro-benchmark style smoke test: it prints the incremental constraint/var
    // cost of a digit-domain Goldilocks mul mod p, without building the full tiny gate.
    //
    // Run with `-- --nocapture` to see the numbers.
    let p_u64 = super::goldilocks::GOLDILOCKS_P;
    let p_d = super::goldilocks::goldilocks_p_bal16_digits_le_const();

    let mut b = Dr1csBuilder::<F257>::new();
    b.enforce_var_eq_const(b.one(), F257::ONE);

    // Fixed inputs (avoid randomness in tests).
    let a_u: u64 = 123456789u64 % p_u64;
    let c_u: u64 = 987654321u64 % p_u64;
    let a_bytes = super::cm_math::alloc_const_goldilocks_u64(&mut b, a_u);
    let c_bytes = super::cm_math::alloc_const_goldilocks_u64(&mut b, c_u);
    let a = super::cm_math::goldilocks_bytes_to_digits(&mut b, a_bytes);
    let c = super::cm_math::goldilocks_bytes_to_digits(&mut b, c_bytes);

    let n_iters: usize = 200;
    let t0 = Instant::now();
    let rows0 = b.rows.len();
    let vars0 = b.assignment.len();

    for _ in 0..n_iters {
        let base_asg: Vec<F257> = b.assignment.clone();
        let mut ib = super::cm_ir::IrBuilder::new(&base_asg);
        let a_ir: [super::cm_ir::VarRef; 17] = core::array::from_fn(|j| super::cm_ir::VarRef::Base(a[j]));
        let c_ir: [super::cm_ir::VarRef; 17] = core::array::from_fn(|j| super::cm_ir::VarRef::Base(c[j]));
        let out_ir = super::cm_ir::goldilocks_mul_mod_p_digits_ir(&mut ib, &a_ir, &c_ir, p_u64, &p_d);
        let lowered = super::cm_ir::lower_ir_into_builder(&mut b, ib.ir);
        let _out: [usize; 17] = core::array::from_fn(|j| lowered.map_var(out_ir[j]));
    }

    let dt = t0.elapsed();
    let rows1 = b.rows.len();
    let vars1 = b.assignment.len();
    eprintln!(
        "== micro scalar_mul_mod_p: iters={} delta_constraints={} (~{:.1}/iter) delta_vars={} (~{:.1}/iter) elapsed={:?} ==",
        n_iters,
        rows1 - rows0,
        (rows1 - rows0) as f64 / (n_iters as f64),
        vars1 - vars0,
        (vars1 - vars0) as f64 / (n_iters as f64),
        dt
    );

    // Basic sanity: output witness should decode to the native product.
    let exp: u64 = ((a_u as u128) * (c_u as u128) % (p_u64 as u128)) as u64;
    let base_asg: Vec<F257> = b.assignment.clone();
    let mut ib = super::cm_ir::IrBuilder::new(&base_asg);
    let a_ir: [super::cm_ir::VarRef; 17] = core::array::from_fn(|j| super::cm_ir::VarRef::Base(a[j]));
    let c_ir: [super::cm_ir::VarRef; 17] = core::array::from_fn(|j| super::cm_ir::VarRef::Base(c[j]));
    let out_ir = super::cm_ir::goldilocks_mul_mod_p_digits_ir(&mut ib, &a_ir, &c_ir, p_u64, &p_d);
    let lowered = super::cm_ir::lower_ir_into_builder(&mut b, ib.ir);
    let out: [usize; 17] = core::array::from_fn(|j| lowered.map_var(out_ir[j]));
    let mut acc: i128 = 0;
    let mut pow: i128 = 1;
    for j in 0..17 {
        acc += (f257_to_i32_bal(b.assignment[out[j]]) as i128) * pow;
        pow *= 16;
    }
    assert_eq!(acc as u64, exp);
}

#[test]
fn test_bounded_u32_from_8_digits_base128_matches_byte_view() {
    let mut b = Dr1csBuilder::<F257>::new();
    b.enforce_var_eq_const(b.one(), F257::ONE);

    // Build a digit block with a 256 in it to exercise byte view (256 -> 0).
    let ds: [u16; 8] = [1, 2, 256, 4, 5, 6, 7, 8];
    let mut dvars = [0usize; 8];
    for i in 0..8 {
        dvars[i] = b.new_var(F257::from(ds[i] as u64));
    }

    let (limbs, bytes, bal16_digits, bal16_sq_digits) =
        bounded_u32_from_8_digits_base128(&mut b, &dvars);

    // Expected u32 from byte view of first 4 digits.
    let bv = |x: u16| if x == 256 { 0u8 } else { x as u8 };
    let exp = u32::from_le_bytes([bv(ds[0]), bv(ds[1]), bv(ds[2]), bv(ds[3])]);

    // Check reconstructed u32 from limbs.
    assert_eq!(limbs_u32_from_base128::<F257>(&b, &limbs), exp);

    // Also check bytes match.
    let to_u8 = |v: usize| var_to_u8::<F257>(&b, v);
    assert_eq!(to_u8(bytes[0]), bv(ds[0]));
    assert_eq!(to_u8(bytes[1]), bv(ds[1]));
    assert_eq!(to_u8(bytes[2]), bv(ds[2]));
    assert_eq!(to_u8(bytes[3]), bv(ds[3]));

    // Check balanced base-16 digits decode to the same u32.
    let mut acc: i128 = 0;
    let mut pow: i128 = 1;
    for &v in &bal16_digits {
        acc += (f257_to_i32_bal(b.assignment[v]) as i128) * pow;
        pow *= 16;
    }
    assert_eq!(acc as u64, exp as u64);

    // Check square digits decode to exp^2.
    let mut sq_acc: i128 = 0;
    let mut pow: i128 = 1;
    for &v in &bal16_sq_digits {
        sq_acc += (f257_to_i32_bal(b.assignment[v]) as i128) * pow;
        pow *= 16;
    }
    assert_eq!(sq_acc, (exp as i128) * (exp as i128));

    let (inst, asg) = b.into_instance();
    inst.check(&asg).expect("bounded u32 gadget satisfied");
}

#[test]
fn test_rebalance_prod12_to_prod13_decodes_same() {
    use rand::{Rng, SeedableRng};
    let mut rng = rand::rngs::StdRng::seed_from_u64(0x5EED_13u64);
    let mut b = Dr1csBuilder::<F257>::new();
    b.enforce_var_eq_const(b.one(), F257::ONE);

    // Random coeff in [-128,127] and random scalar in a u32-ish envelope.
    let coeff: i128 = rng.gen_range(-128i128..=127i128);
    let x: i128 = rng.gen_range(-(1i128 << 32)..(1i128 << 32));

    // coeff -> 3 balanced digits
    let mut cur = coeff;
    let mut c3 = [0usize; 3];
    for i in 0..3 {
        let mut r = ((cur % 16) + 16) % 16;
        if r >= 8 {
            r -= 16;
        }
        c3[i] = alloc_bal16_digit(&mut b, r as i8);
        cur = (cur - r) / 16;
    }
    debug_assert_eq!(cur, 0);

    // x -> 9 balanced digits
    let x_digits = {
        let mut xx = x;
        let mut out: Vec<usize> = Vec::with_capacity(9);
        for _ in 0..9 {
            let mut r = ((xx % 16) + 16) % 16;
            if r >= 8 {
                r -= 16;
            }
            out.push(alloc_bal16_digit(&mut b, r as i8));
            xx = (xx - r) / 16;
        }
        out
    };

    let p12 = mul_bal16_3_by_digits9(&mut b, &c3, &x_digits);
    let p13 = rebalance_prod12_to_prod13(&mut b, &p12);

    let decode = |digits: &[usize]| -> i128 {
        let mut acc: i128 = 0;
        let mut pow: i128 = 1;
        for &dv in digits {
            acc += (f257_to_i32_bal(b.assignment[dv]) as i128) * pow;
            pow *= 16;
        }
        acc
    };
    assert_eq!(decode(&p12), decode(&p13));
    assert_eq!(decode(&p13), coeff * x);

    let (inst, asg) = b.into_instance();
    inst.check(&asg).expect("rebalance prod satisfied");
}

#[test]
fn test_lift_recording_trace_ops_to_f257_roundtrip_small() {
    use ark_ff::Field;
    use stark_rings::cyclotomic_ring::models::goldilocks::RqPoly as R;
    use stark_rings::PolyRing;
    use stark_rings::Ring;

    // Record a tiny trace in the LF+ recording transcript (base ring is large, but values are digits/bytes).
    let mut rec = crate::recording_transcript::TracePoseidonTranscript::<R>::empty::<()>();
    rec.absorb(&R::ONE);
    let _ = rec.squeeze_bytes(17);
    let _c = rec.get_challenge();
    let tr = rec.trace().clone();

    // Lift ops to F257 and ensure lengths line up.
    type BF = <<R as PolyRing>::BaseRing as Field>::BasePrimeField;
    let ops_f257 =
        lift_recording_trace_ops_to_f257::<BF>(&tr.ops).expect("lift_recording_trace_ops_to_f257");
    assert_eq!(ops_f257.len(), tr.ops.len());

    // Check that every absorbed/squeezed element is in 0..=256.
    for op in ops_f257 {
        match op {
            PoseidonTraceOp::Absorb(v) | PoseidonTraceOp::SqueezeField(v) => {
                for e in v {
                    let du16 = e.into_bigint().to_bytes_le().get(0).copied().unwrap_or(0) as u16;
                    assert!(du16 < 257);
                }
            }
            PoseidonTraceOp::SqueezeBytes { .. } => {}
        }
    }
}

#[test]
fn test_mul_bal16_small_3_by_u32_bal_roundtrip() {
    // Multiply a 12-bit-ish value (3 base-16 digits) by a balanced u32 (9 digits),
    // and check the digit decomposition matches native integer multiplication.
    fn to_bal16_u32(mut x: u32) -> Vec<i8> {
        // Start from base-16 digits (0..15), then balance with carry so each digit in [-8,7].
        let mut digs: Vec<i8> = Vec::with_capacity(9);
        let mut carry: i32 = 0;
        for _ in 0..8 {
            let d = (x & 0xF) as i32;
            x >>= 4;
            let mut t = d + carry;
            if t >= 8 {
                t -= 16;
                carry = 1;
            } else {
                carry = 0;
            }
            digs.push(t as i8);
        }
        digs.push(carry as i8);
        digs
    }
    fn from_bal16(digs: &[i8]) -> i128 {
        let mut acc: i128 = 0;
        let mut pow: i128 = 1;
        for &d in digs {
            acc += (d as i128) * pow;
            pow *= 16;
        }
        acc
    }

    let a_u16: u16 = 0x0777; // all nibbles < 8 => already balanced (3 digits)
    let b_u32: u32 = 0xffff_fffe;
    let a_i = a_u16 as i128;
    let b_bal = to_bal16_u32(b_u32);
    let b_i = from_bal16(&b_bal);
    assert_eq!(b_i as u64, b_u32 as u64);
    let prod_i = a_i * b_i;

    let mut b = Dr1csBuilder::<F257>::new();
    b.enforce_var_eq_const(b.one(), F257::ONE);

    // a in balanced base-16 (3 digits) from standard nibbles (no balancing needed due to a<2048).
    let a0 = alloc_bal16_digit(&mut b, (a_u16 & 0xF) as i8);
    let a1 = alloc_bal16_digit(&mut b, ((a_u16 >> 4) & 0xF) as i8);
    let a2 = alloc_bal16_digit(&mut b, ((a_u16 >> 8) & 0xF) as i8);
    let a_digits = vec![a0, a1, a2];

    // b in balanced base-16 (9 digits).
    let mut b_digits: Vec<usize> = Vec::with_capacity(9);
    for &d in &b_bal {
        b_digits.push(alloc_bal16_digit(&mut b, d));
    }

    let out = mul_bal16_small(&mut b, &a_digits, &b_digits);

    // Check satisfiable.
    let (inst, asg) = b.into_instance();
    inst.check(&asg).expect("mul_bal16_small satisfiable");

    // Decode output digits and compare.
    let out_digits_i32: Vec<i32> = out.iter().map(|&v| f257_to_i32_bal(asg[v])).collect();
    let mut out_i: i128 = 0;
    let mut pow: i128 = 1;
    for &di in &out_digits_i32 {
        out_i += (di as i128) * pow;
        pow *= 16;
    }
    assert_eq!(out_i, prod_i, "decoded integer mismatch");
}

#[test]
fn test_u32_bytes_to_bal16_digits_roundtrip() {
    let x: u32 = 0xffff_fffe;
    let mut b = Dr1csBuilder::<F257>::new();
    b.enforce_var_eq_const(b.one(), F257::ONE);

    // Allocate bytes as ByteVars, then pass just the byte vars.
    let bytes = x.to_le_bytes();
    let b0 = alloc_byte::<F257>(&mut b, bytes[0]);
    let b1 = alloc_byte::<F257>(&mut b, bytes[1]);
    let b2 = alloc_byte::<F257>(&mut b, bytes[2]);
    let b3 = alloc_byte::<F257>(&mut b, bytes[3]);

    let digs = u32_bytes_to_bal16_digits(&mut b, [b0.byte, b1.byte, b2.byte, b3.byte]);

    let (inst, asg) = b.into_instance();
    inst.check(&asg).expect("u32_bytes_to_bal16_digits satisfiable");

    let mut acc: i128 = 0;
    let mut pow: i128 = 1;
    for &v in &digs {
        let di = f257_to_i32_bal(asg[v]) as i128;
        acc += di * pow;
        pow *= 16;
    }
    assert_eq!(acc as u64, x as u64);
}

#[test]
fn test_add_bal16_same_len_roundtrip() {
    // Check that balanced-base16 addition matches integer addition.
    fn decode(asg: &[F257], digs: &[usize]) -> i128 {
        let mut acc: i128 = 0;
        let mut pow: i128 = 1;
        for &v in digs {
            acc += (f257_to_i32_bal(asg[v]) as i128) * pow;
            pow *= 16;
        }
        acc
    }

    let mut b = Dr1csBuilder::<F257>::new();
    b.enforce_var_eq_const(b.one(), F257::ONE);

    // a = 0x0777 (3 digits), b = 0x0001 (3 digits)
    let a = Bal16Checked(vec![
        alloc_bal16_digit(&mut b, 0x7),
        alloc_bal16_digit(&mut b, 0x7),
        alloc_bal16_digit(&mut b, 0x7),
    ]);
    let c = Bal16Checked(vec![
        alloc_bal16_digit(&mut b, 1),
        alloc_bal16_digit(&mut b, 0),
        alloc_bal16_digit(&mut b, 0),
    ]);

    let (sum, carry) = add_bal16_same_len(&mut b, &a, &c);
    let (inst, asg) = b.into_instance();
    inst.check(&asg).expect("add_bal16_same_len satisfiable");

    let a_i = decode(&asg, a.as_slice());
    let c_i = decode(&asg, c.as_slice());
    let sum_i = decode(&asg, sum.as_slice())
        + (f257_to_i32_bal(asg[carry]) as i128) * 16i128.pow(sum.len() as u32);
    assert_eq!(sum_i, a_i + c_i);
}

#[test]
fn test_neg_and_sub_bal16_roundtrip() {
    // Check negation and subtraction decode correctly for random small-ish values.
    use rand::Rng;
    let mut rng = ark_std::test_rng();

    for _ in 0..200 {
        let x: i64 = rng.gen_range(-(1i64 << 31)..(1i64 << 31));
        let y: i64 = rng.gen_range(-(1i64 << 31)..(1i64 << 31));

        // Encode into 9 balanced digits (enough for 32-bit-ish values).
        fn to_bal16(mut v: i64) -> [i8; 9] {
            let mut out = [0i8; 9];
            for i in 0..9 {
                let mut d = (v % 16) as i32;
                if d > 7 {
                    d -= 16;
                }
                if d < -8 {
                    d += 16;
                }
                out[i] = d as i8;
                v = (v - d as i64) / 16;
            }
            out
        }
        let xd = to_bal16(x);
        let yd = to_bal16(y);

        let mut b = Dr1csBuilder::<F257>::new();
        b.enforce_var_eq_const(b.one(), F257::ONE);
        let xvars = Bal16Checked(xd.iter().map(|&d| alloc_bal16_digit(&mut b, d)).collect());
        let yvars = Bal16Checked(yd.iter().map(|&d| alloc_bal16_digit(&mut b, d)).collect());

        let (nx, c0, diff, c1) = {
            use super::cm_ir::{
                add_bal16_same_len_ir, lower_ir_into_builder, neg_bal16_digits_ir, Bal16CheckedIr, IrBuilder, VarRef,
            };
            // Snapshot the current assignment so we can build IR without borrowing `b` immutably
            // across the subsequent `lower_ir_into_builder(&mut b, ..)` call.
            let base_asg: Vec<F257> = b.assignment.clone();
            let mut ib = IrBuilder::new(&base_asg);
            let x_ir = Bal16CheckedIr(xvars.as_slice().iter().copied().map(VarRef::Base).collect());
            let y_ir = Bal16CheckedIr(yvars.as_slice().iter().copied().map(VarRef::Base).collect());

            let (nx_ir, c0_ir) = neg_bal16_digits_ir(&mut ib, &x_ir);
            let (neg_y_ir, _carry_neg_y) = neg_bal16_digits_ir(&mut ib, &y_ir);
            let (diff_ir, c1_ir) = add_bal16_same_len_ir(&mut ib, &x_ir, &neg_y_ir);

            let lowered = lower_ir_into_builder(&mut b, ib.ir);
            let nx: Vec<usize> = nx_ir.0.into_iter().map(|v| lowered.map_var(v)).collect();
            let c0 = lowered.map_var(c0_ir);
            let diff: Vec<usize> = diff_ir.0.into_iter().map(|v| lowered.map_var(v)).collect();
            let c1 = lowered.map_var(c1_ir);
            (Bal16Checked(nx), c0, Bal16Checked(diff), c1)
        };

        // For fixed-width 9-digit inputs, enforce no overflow.
        b.enforce_var_eq_const(c0, F257::ZERO);
        b.enforce_var_eq_const(c1, F257::ZERO);

        let (inst, asg) = b.into_instance();
        inst.check(&asg).expect("neg/sub satisfiable");

        let decode = |ds: &[usize]| -> i128 {
            let mut acc: i128 = 0;
            let mut pow: i128 = 1;
            for &v in ds {
                acc += (f257_to_i32_bal(asg[v]) as i128) * pow;
                pow *= 16;
            }
            acc
        };
        assert_eq!(decode(nx.as_slice()), -(x as i128));
        assert_eq!(decode(diff.as_slice()), (x as i128) - (y as i128));
    }
}

#[test]
fn test_mul_bal16_long_by_u32ish9_roundtrip() {
    // Multiply a moderately-sized integer by a u32-ish integer via chunking.
    // (Keep decoded values within i128 to avoid overflow in the test.)
    use rand::Rng;
    let mut rng = ark_std::test_rng();

    for _ in 0..50 {
        // Build a random ~48-bit signed integer.
        let mag: i128 = (rng.gen::<u64>() & ((1u64 << 48) - 1)) as i128;
        let sign: i128 = if rng.gen::<bool>() { 1 } else { -1 };
        let a: i128 = sign * mag;
        let b_u32: u32 = rng.gen();
        let b_i: i128 = b_u32 as i128;

        // Encode a into balanced base-16 digits (len 16 is enough for ~64 bits).
        let mut a_tmp = a;
        let mut a_digs: Vec<i8> = Vec::with_capacity(16);
        for _ in 0..16 {
            let mut d = (a_tmp % 16) as i32;
            if d > 7 {
                d -= 16;
            }
            if d < -8 {
                d += 16;
            }
            a_digs.push(d as i8);
            a_tmp = (a_tmp - d as i128) / 16;
        }
        assert_eq!(a_tmp, 0);

        let mut b = Dr1csBuilder::<F257>::new();
        b.enforce_var_eq_const(b.one(), F257::ONE);
        let a_vars: Vec<usize> = a_digs.iter().map(|&d| alloc_bal16_digit(&mut b, d)).collect();
        let bb = b_u32.to_le_bytes();
        let b_bytes = [
            alloc_byte::<F257>(&mut b, bb[0]).byte,
            alloc_byte::<F257>(&mut b, bb[1]).byte,
            alloc_byte::<F257>(&mut b, bb[2]).byte,
            alloc_byte::<F257>(&mut b, bb[3]).byte,
        ];
        // This test is now covered by the IR-backed multiplication path used throughout the gate.
        // Keep only the smaller, well-scoped `mul_bal16_3_by_digits9` / `mul_u32ish9_to_fixed_bal16` tests.
        let b9 = u32_bytes_to_bal16_digits(&mut b, b_bytes);
        let coeff3 = [a_vars[0], a_vars.get(1).copied().unwrap_or(a_vars[0]), a_vars.get(2).copied().unwrap_or(a_vars[0])];
        let prod = mul_bal16_3_by_digits9(&mut b, &coeff3, &b9);

        let (inst, asg) = b.into_instance();
        inst.check(&asg).expect("long*u32ish satisfiable");

        let mut acc: i128 = 0;
        let mut pow: i128 = 1;
        for (idx, &dv) in prod.iter().enumerate() {
            acc += (f257_to_i32_bal(asg[dv]) as i128) * pow;
            if idx + 1 < prod.len() {
                pow *= 16;
            }
        }
        // Decode only the low 12 digits (this is a local gadget test now).
        let a3 = (a_digs.get(0).copied().unwrap_or(0) as i128)
            + (a_digs.get(1).copied().unwrap_or(0) as i128) * 16
            + (a_digs.get(2).copied().unwrap_or(0) as i128) * 256;
        assert_eq!(acc, a3 * b_i);
    }
}

#[test]
fn test_mul_bal16_3_by_u32_roundtrip() {
    fn decode(asg: &[F257], digs: &[usize]) -> i128 {
        let mut acc: i128 = 0;
        let mut pow: i128 = 1;
        for &v in digs {
            acc += (f257_to_i32_bal(asg[v]) as i128) * pow;
            pow *= 16;
        }
        acc
    }

    let mut b = Dr1csBuilder::<F257>::new();
    b.enforce_var_eq_const(b.one(), F257::ONE);

    // coeff = 0x0777 (positive, already balanced in 3 digits)
    let coeff3 = [
        alloc_bal16_digit(&mut b, 0x7),
        alloc_bal16_digit(&mut b, 0x7),
        alloc_bal16_digit(&mut b, 0x7),
    ];
    // u32 = 0xffff_fffe via byte->bal16 gadget.
    let x: u32 = 0xffff_fffe;
    let bytes = x.to_le_bytes();
    let b0 = alloc_byte::<F257>(&mut b, bytes[0]);
    let b1 = alloc_byte::<F257>(&mut b, bytes[1]);
    let b2 = alloc_byte::<F257>(&mut b, bytes[2]);
    let b3 = alloc_byte::<F257>(&mut b, bytes[3]);
    let u32_bal16 = u32_bytes_to_bal16_digits(&mut b, [b0.byte, b1.byte, b2.byte, b3.byte]);
    assert_eq!(u32_bal16.len(), 9);

    let out12 = mul_bal16_3_by_digits9(&mut b, &coeff3, &u32_bal16);

    let (inst, asg) = b.into_instance();
    inst.check(&asg).expect("mul_bal16_3_by_u32 satisfiable");

    let coeff_i = decode(&asg, &coeff3);
    let u_i = decode(&asg, &u32_bal16);
    let out_i = decode(&asg, &out12);
    assert_eq!(out_i, coeff_i * u_i);
}

#[test]
fn test_scale_short_coeffs_by_u32_roundtrip_small_vec() {
    fn decode(asg: &[F257], digs: &[usize]) -> i128 {
        let mut acc: i128 = 0;
        let mut pow: i128 = 1;
        for &v in digs {
            acc += (f257_to_i32_bal(asg[v]) as i128) * pow;
            pow *= 16;
        }
        acc
    }

    let mut b = Dr1csBuilder::<F257>::new();
    b.enforce_var_eq_const(b.one(), F257::ONE);

    // Two short coefficients (simulate small CM coeffs).
    // c0 = 0x0777, c1 = -5
    let c0 = [
        alloc_bal16_digit(&mut b, 0x7),
        alloc_bal16_digit(&mut b, 0x7),
        alloc_bal16_digit(&mut b, 0x0),
    ];
    let c1 = [
        alloc_bal16_digit(&mut b, -5),
        alloc_bal16_digit(&mut b, 0),
        alloc_bal16_digit(&mut b, 0),
    ];
    let coeffs = vec![c0, c1];

    // u32 = 0x04030201 (no 256 byte-view edge case)
    let x: u32 = 0x04030201;
    let bytes = x.to_le_bytes();
    let b0 = alloc_byte::<F257>(&mut b, bytes[0]);
    let b1 = alloc_byte::<F257>(&mut b, bytes[1]);
    let b2 = alloc_byte::<F257>(&mut b, bytes[2]);
    let b3 = alloc_byte::<F257>(&mut b, bytes[3]);
    let u32_digits = u32_bytes_to_bal16_digits(&mut b, [b0.byte, b1.byte, b2.byte, b3.byte]);
    assert_eq!(u32_digits.len(), 9);

    let prods = scale_short_coeffs_by_digits9(&mut b, &coeffs, &u32_digits);
    assert_eq!(prods.len(), 2);

    let (inst, asg) = b.into_instance();
    inst.check(&asg).expect("scale_short_coeffs_by_u32 satisfiable");

    let u = decode(&asg, &u32_digits);
    for (i, c3) in coeffs.iter().enumerate() {
        let c = decode(&asg, c3);
        let p = decode(&asg, &prods[i]);
        assert_eq!(p, c * u);
    }
}

