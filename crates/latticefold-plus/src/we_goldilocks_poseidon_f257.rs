//! Experimental: emulate (parts of) goldilocks verifier plumbing inside a tiny field (F257) dR1CS.
//!
//! ## IMPORTANT: radix choice in F257
//! This file originally experimented with **base-256 byte limbs** and carry constraints.
//! That is **NOT sound over `F257`**, because `256 ≡ -1 (mod 257)`, so equations like
//! `out + 256*carry_next - ... = 0` do not model integer base-256 arithmetic.
//!
//! The sound approach in a tiny field is to use a radix `B < p_tiny` and keep each per-limb
//! carry/borrow equation within `(-p_tiny, p_tiny)` so that equality in the field coincides
//! with equality over the integers.
//!
//! We therefore implement **base-128 (7-bit) limbs** here (a good compromise between byte-cost
//! and soundness in `F257`).

use ark_ff::{BigInteger, PrimeField};
use symphony::dpp_sumcheck::Dr1csBuilder;

// TEMP (GL64 experiment): treat the "goldilocks modulus" as Goldilocks.
pub(crate) const GOLDILOCKS_P: u64 = 0xFFFF_FFFF_0000_0001u64;

const LIMB_BASE_U64: u64 = 128;
const LIMB_BITS: usize = 7;
const LIMBS_U64: usize = 10; // ceil(64/7) = 10, so u64 fits.

#[derive(Clone, Debug)]
struct Base128Num {
    /// Little-endian base-128 limbs, each constrained to be in 0..=127.
    limbs: [usize; LIMBS_U64],
}

fn const_var<F: PrimeField>(b: &mut Dr1csBuilder<F>, c: F) -> usize {
    let v = b.new_var(c);
    b.enforce_var_eq_const(v, c);
    v
}

fn alloc_bool<F: PrimeField>(b: &mut Dr1csBuilder<F>, bit: bool) -> usize {
    let v = b.new_var(if bit { F::ONE } else { F::ZERO });
    // v*(1-v)=0
    b.add_constraint(
        vec![(F::ONE, v)],
        vec![(F::ONE, b.one()), (-F::ONE, v)],
        vec![(F::ZERO, b.one())],
    );
    v
}

fn alloc_u7<F: PrimeField>(b: &mut Dr1csBuilder<F>, d: u8) -> usize {
    debug_assert!(d < 128);
    let mut bits = [0usize; LIMB_BITS];
    for i in 0..LIMB_BITS {
        bits[i] = alloc_bool::<F>(b, ((d >> i) & 1) == 1);
    }
    let v = b.new_var(F::from(d as u64));
    // v = sum 2^i * bits[i]
    let mut lc = vec![(F::ONE, v)];
    let mut pow = F::ONE;
    for i in 0..LIMB_BITS {
        lc.push((-pow, bits[i]));
        pow *= F::from(2u64);
    }
    b.enforce_lc_times_one_eq_const(lc);
    v
}

fn goldilocks_p_base128_digits_le() -> [u8; LIMBS_U64] {
    let mut out = [0u8; LIMBS_U64];
    let mut t = GOLDILOCKS_P;
    for i in 0..LIMBS_U64 {
        out[i] = (t & (LIMB_BASE_U64 - 1)) as u8;
        t >>= LIMB_BITS;
    }
    out
}

fn alloc_u64_base128<F: PrimeField>(b: &mut Dr1csBuilder<F>, x: u64) -> Base128Num {
    let mut limbs = [0usize; LIMBS_U64];
    for i in 0..LIMBS_U64 {
        let di = ((x >> (LIMB_BITS * i)) & (LIMB_BASE_U64 - 1)) as u8;
        limbs[i] = alloc_u7::<F>(b, di);
    }
    Base128Num { limbs }
}

fn base128_num_to_u64<F: PrimeField>(b: &Dr1csBuilder<F>, x: &Base128Num) -> u64 {
    let mut acc: u64 = 0;
    for i in (0..LIMBS_U64).rev() {
        let di = b.assignment[x.limbs[i]]
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

fn cmp_u_ge_goldilocks_p_base128<F: PrimeField>(b: &mut Dr1csBuilder<F>, u: &Base128Num) -> usize {
    // Compare u >= p by running a base-128 borrow chain on (u - p).
    let u_u64 = base128_num_to_u64::<F>(b, u);
    let p_digits = goldilocks_p_base128_digits_le();

    let mut borrow = const_var::<F>(b, F::ZERO);
    let mut borrow_final = const_var::<F>(b, F::ZERO);
    for i in 0..LIMBS_U64 {
        let ui = ((u_u64 >> (LIMB_BITS * i)) & (LIMB_BASE_U64 - 1)) as i16;
        let pi = p_digits[i] as i16;
        let bi = if i == 0 {
            0i16
        } else {
            b.assignment[borrow]
                .into_bigint()
                .to_bytes_le()
                .get(0)
                .copied()
                .unwrap_or(0) as i16
        };
        let mut t = ui - pi - bi;
        let borrow_next_u8 = if t < 0 { 1u8 } else { 0u8 };
        if t < 0 {
            t += LIMB_BASE_U64 as i16;
        }
        let diff_i = alloc_u7::<F>(b, t as u8);
        let borrow_next = alloc_bool::<F>(b, borrow_next_u8 == 1);

        // u_i - p_i - borrow_i + base*borrow_{i+1} - diff_i == 0
        b.enforce_lc_times_one_eq_const(vec![
            (F::ONE, u.limbs[i]),
            (-F::from(p_digits[i] as u64), b.one()),
            (-F::ONE, borrow),
            (F::from(LIMB_BASE_U64), borrow_next),
            (-F::ONE, diff_i),
        ]);

        borrow = borrow_next;
        borrow_final = borrow;
    }

    // is_ge = 1 - borrow_final  (since borrow_final=1 iff u < p)
    let is_ge = b.new_var(F::ONE - b.assignment[borrow_final]);
    b.enforce_lc_times_one_eq_const(vec![
        (F::ONE, is_ge),
        (F::ONE, borrow_final),
        (-F::ONE, b.one()),
    ]);
    is_ge
}

fn reduce_u64_mod_goldilocks_base128<F: PrimeField>(b: &mut Dr1csBuilder<F>, u: &Base128Num) -> (Base128Num, usize) {
    // No-loop reduction: since p > 2^63, for u in [0,2^64) we have floor(u/p) ∈ {0,1}.
    let u_u64 = base128_num_to_u64::<F>(b, u);
    let q_bit = if u_u64 >= GOLDILOCKS_P { 1u8 } else { 0u8 };
    let q = alloc_bool::<F>(b, q_bit == 1);

    // Also enforce q matches (u >= p) via a separate compare (prevents non-canonical choice).
    let is_ge = cmp_u_ge_goldilocks_p_base128::<F>(b, u);
    b.enforce_lc_times_one_eq_const(vec![(F::ONE, q), (-F::ONE, is_ge)]);

    // z = u - q*p (base-128 subtraction with borrows), and enforce final borrow=0.
    let z_u64 = if q_bit == 1 { u_u64 - GOLDILOCKS_P } else { u_u64 };
    let mut z_limbs = [0usize; LIMBS_U64];
    for i in 0..LIMBS_U64 {
        let di = ((z_u64 >> (LIMB_BITS * i)) & (LIMB_BASE_U64 - 1)) as u8;
        z_limbs[i] = alloc_u7::<F>(b, di);
    }
    let z = Base128Num { limbs: z_limbs };
    let p_digits = goldilocks_p_base128_digits_le();

    let mut borrow = const_var::<F>(b, F::ZERO);
    for i in 0..LIMBS_U64 {
        // Witness borrow_{i+1} from integers.
        let ui = ((u_u64 >> (LIMB_BITS * i)) & (LIMB_BASE_U64 - 1)) as i16;
        let zi = ((z_u64 >> (LIMB_BITS * i)) & (LIMB_BASE_U64 - 1)) as i16;
        let pi = p_digits[i] as i16;
        let bi = if i == 0 {
            0i16
        } else {
            b.assignment[borrow]
                .into_bigint()
                .to_bytes_le()
                .get(0)
                .copied()
                .unwrap_or(0) as i16
        };
        // ui - q*pi - bi - zi = -base*borrow_next
        let rhs = ui - (q_bit as i16) * pi - bi - zi;
        let borrow_next_u8 = if rhs < 0 { 1u8 } else { 0u8 };
        let borrow_next = alloc_bool::<F>(b, borrow_next_u8 == 1);

        // u_i - q*p_i - borrow_i + base*borrow_{i+1} - z_i == 0
        b.enforce_lc_times_one_eq_const(vec![
            (F::ONE, u.limbs[i]),
            (-F::from(p_digits[i] as u64), q),
            (-F::ONE, borrow),
            (F::from(LIMB_BASE_U64), borrow_next),
            (-F::ONE, z.limbs[i]),
        ]);

        borrow = borrow_next;
    }
    // No underflow allowed.
    b.enforce_var_eq_const(borrow, F::ZERO);

    (z, q)
}

pub fn count_one_boundary_canon_bytes<F: PrimeField>() -> (usize, usize) {
    let mut b = Dr1csBuilder::<F>::new();
    // Build u in [0,2^64), and reduce mod p with a single conditional subtract.
    let u = alloc_u64_base128::<F>(&mut b, (GOLDILOCKS_P / 2) + 12345);
    let _zq = reduce_u64_mod_goldilocks_base128::<F>(&mut b, &u);
    let (inst, _asg) = b.into_instance();
    (inst.nvars, inst.constraints.len())
}

// Historical note: a previous `mul_u64_bytes_mod_p` gadget lived here, but it used base-256
// carry arithmetic and is unsound in `F257`. If you need mul mod p inside a tiny field, use
// small-radix limbs (base <= 128) + appropriate boundedness/batching.

#[cfg(test)]
mod tests {
    use super::*;
    use ark_ff::{Fp64, MontBackend, MontConfig};

    #[derive(MontConfig)]
    #[modulus = "257"]
    #[generator = "3"]
    pub struct F257Config;
    type F257 = Fp64<MontBackend<F257Config, 1>>;

    #[test]
    fn print_stub_constraint_count() {
        let (nvars, ncons) = count_one_boundary_canon_bytes::<F257>();
        eprintln!(
            "[we_GOLDILOCKS_Poseidon_f257] reduce_u64_mod_goldilocks_base128 nvars={} constraints={}",
            nvars, ncons
        );
        assert!(nvars > 0 && ncons > 0);
    }

    #[test]
    fn sanity_reduce_matches_native() {
        let mut b = Dr1csBuilder::<F257>::new();
        let u_native: u64 = 18446744073709551615u64; // 2^64-1 edge case
        let u = alloc_u64_base128::<F257>(&mut b, u_native);
        let (z, q) = reduce_u64_mod_goldilocks_base128::<F257>(&mut b, &u);
        let z_u = base128_num_to_u64::<F257>(&b, &z);
        let q_u = b.assignment[q].into_bigint().to_bytes_le().get(0).copied().unwrap_or(0);
        assert_eq!(z_u, u_native % GOLDILOCKS_P);
        assert_eq!(q_u, if u_native >= GOLDILOCKS_P { 1 } else { 0 });
        let (inst, asg) = b.into_instance();
        inst.check(&asg).expect("constraints should be satisfied");
    }
}

