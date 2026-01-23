//! Experimental: emulate Frog Poseidon inside a tiny field (F257) dR1CS.
//!
//! Byte-limb (base-256) gadgets for boundary-only canonicalization.
//! This module intentionally avoids base-3 digit arithmetic.

use ark_ff::{BigInteger, PrimeField};
use symphony::dpp_sumcheck::Dr1csBuilder;

const FROG_P: u64 = 15912092521325583641u64;

#[derive(Clone, Debug)]
struct ByteVar {
    byte: usize,      // 0..255
    bits: [usize; 8], // boolean bits
}

#[derive(Clone, Debug)]
struct ByteNum {
    bytes: [ByteVar; 8], // little-endian bytes
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

fn alloc_byte<F: PrimeField>(b: &mut Dr1csBuilder<F>, d: u8) -> ByteVar {
    let mut bits = [0usize; 8];
    for i in 0..8 {
        bits[i] = alloc_bool::<F>(b, ((d >> i) & 1) == 1);
    }
    let v = b.new_var(F::from(d as u64));
    // v = sum 2^i * bits[i]
    let mut lc = vec![(F::ONE, v)];
    let mut pow = F::ONE;
    for i in 0..8 {
        lc.push((-pow, bits[i]));
        pow *= F::from(2u64);
    }
    b.enforce_lc_times_one_eq_const(lc);
    ByteVar { byte: v, bits }
}

fn alloc_u64_bytes<F: PrimeField>(b: &mut Dr1csBuilder<F>, x: u64) -> ByteNum {
    let mut bytes: [ByteVar; 8] = [
        alloc_byte::<F>(b, 0),
        alloc_byte::<F>(b, 0),
        alloc_byte::<F>(b, 0),
        alloc_byte::<F>(b, 0),
        alloc_byte::<F>(b, 0),
        alloc_byte::<F>(b, 0),
        alloc_byte::<F>(b, 0),
        alloc_byte::<F>(b, 0),
    ];
    let mut t = x;
    for i in 0..8 {
        bytes[i] = alloc_byte::<F>(b, (t & 0xff) as u8);
        t >>= 8;
    }
    ByteNum { bytes }
}

fn frog_p_bytes_le() -> [u8; 8] {
    (FROG_P as u64).to_le_bytes()
}

fn bytes_to_u64<F: PrimeField>(b: &Dr1csBuilder<F>, x: &ByteNum) -> u64 {
    let mut acc: u128 = 0;
    let mut pow: u128 = 1;
    for i in 0..8 {
        let bi = b.assignment[x.bytes[i].byte]
            .into_bigint()
            .to_bytes_le()
            .get(0)
            .copied()
            .unwrap_or(0) as u128;
        acc += bi * pow;
        pow <<= 8;
    }
    (acc as u64) % FROG_P
}

fn canonicalize_bytes_with_qbit<F: PrimeField>(
    b: &mut Dr1csBuilder<F>,
    lane: &ByteNum,
    q_bit: u8,
) -> ByteNum {
    // Enforce: lane = z + q*p  with q in {0,1}, z is canonical representative.
    let q = alloc_bool::<F>(b, q_bit == 1);
    let z_u = {
        let lane_u = bytes_to_u64::<F>(b, lane) as u128;
        if q_bit == 1 {
            lane_u.saturating_sub(FROG_P as u128) as u64
        } else {
            lane_u as u64
        }
    };
    let z = alloc_u64_bytes::<F>(b, z_u % FROG_P);
    let p_bytes = frog_p_bytes_le();
    let base_f = F::from(256u64);
    let mut carry = const_var::<F>(b, F::ZERO);
    for i in 0..8 {
        // lane_i + carry - z_i - q*p_i - 256*carry_next == 0
        let carry_next_u = {
            let lhs = b.assignment[lane.bytes[i].byte] + b.assignment[carry];
            let rhs0 = b.assignment[z.bytes[i].byte] + (F::from(p_bytes[i] as u64) * b.assignment[q]);
            let lhs_u = lhs.into_bigint().to_bytes_le().get(0).copied().unwrap_or(0) as i16;
            let rhs_u = rhs0.into_bigint().to_bytes_le().get(0).copied().unwrap_or(0) as i16;
            let mut diff = lhs_u - rhs_u;
            while diff < 0 {
                diff += 256;
            }
            while diff >= 512 {
                diff -= 256;
            }
            ((diff / 256) as u8).min(1)
        };
        let carry_next = alloc_bool::<F>(b, carry_next_u == 1);
        b.enforce_lc_times_one_eq_const(vec![
            (F::ONE, lane.bytes[i].byte),
            (F::ONE, carry),
            (-F::ONE, z.bytes[i].byte),
            (-F::from(p_bytes[i] as u64), q),
            (-base_f, carry_next),
        ]);
        carry = carry_next;
    }
    z
}

pub fn count_one_boundary_canon_bytes<F: PrimeField>() -> (usize, usize) {
    let mut b = Dr1csBuilder::<F>::new();
    // Build a lane in [0,2p) to allow q in {0,1}.
    let lane_u = (FROG_P / 2) + 12345;
    let lane = alloc_u64_bytes::<F>(&mut b, lane_u);
    let _z = canonicalize_bytes_with_qbit::<F>(&mut b, &lane, 0);
    let (inst, _asg) = b.into_instance();
    (inst.nvars, inst.constraints.len())
}

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
            "[we_frog_poseidon_f257] boundary_canon_bytes(qbit) nvars={} constraints={}",
            nvars, ncons
        );
        // For n=64 squeeze_bytes, usable_bytes=7 => 10 field elems per op.
        let canon_elems = 515u64 * 10u64;
        let canon_cost = canon_elems.saturating_mul(ncons as u64);
        eprintln!(
            "[we_frog_poseidon_f257] est_boundary_canon_bytes_constraints≈{} (elems={canon_elems})",
            canon_cost
        );
        assert!(nvars > 0 && ncons > 0);
    }
}
//!
//! We implement **non-native arithmetic** for Frog’s 64-bit prime field inside F257 by
//! representing integers in **base 3** digits (trits). This keeps all per-digit convolution
//! sums < 257, avoiding accidental mod-257 wrap in witnesses.
//!
//! WARNING: This is expected to be extremely expensive in constraints.

use ark_ff::{BigInteger, PrimeField};
use symphony::dpp_sumcheck::Dr1csBuilder;

const BASE: u64 = 3;
const NDIG64: usize = 41; // ceil(log_3(2^64)) = 41
const NDIG128: usize = 2 * NDIG64; // product length

const FROG_P: u64 = 15912092521325583641u64;

#[derive(Clone, Debug)]
struct DigNum {
    digs: [usize; NDIG64], // each in {0,1,2}
}

#[derive(Clone, Debug)]
struct DigNum128 {
    digs: [usize; NDIG128], // each in {0,1,2} (for raw products before reduction)
}

#[derive(Clone, Debug)]
struct ByteNum {
    bytes: [usize; 8], // each in {0..255}
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

/// Allocate a base-3 digit d ∈ {0,1,2}.
///
/// Encoding: d = b0 + 2*b1 with boolean b0,b1 and constraint b0*b1 = 0 (forbid 3).
fn alloc_trit<F: PrimeField>(b: &mut Dr1csBuilder<F>, d: u8) -> usize {
    debug_assert!(d < 3);
    let b0 = alloc_bool::<F>(b, (d & 1) == 1);
    let b1 = alloc_bool::<F>(b, d == 2);
    // b0*b1 == 0
    b.add_constraint(vec![(F::ONE, b0)], vec![(F::ONE, b1)], vec![(F::ZERO, b.one())]);
    let v = b.new_var(F::from(d as u64));
    // v - b0 - 2*b1 = 0
    b.enforce_lc_times_one_eq_const(vec![
        (F::ONE, v),
        (-F::ONE, b0),
        (-F::from(2u64), b1),
    ]);
    v
}

fn alloc_byte<F: PrimeField>(b: &mut Dr1csBuilder<F>, d: u8) -> usize {
    let mut bits = [0usize; 8];
    for i in 0..8 {
        bits[i] = alloc_bool::<F>(b, ((d >> i) & 1) == 1);
    }
    let v = b.new_var(F::from(d as u64));
    // v = sum 2^i * bits[i]
    let mut lc = vec![(F::ONE, v)];
    let mut pow = F::ONE;
    for i in 0..8 {
        lc.push((-pow, bits[i]));
        pow *= F::from(2u64);
    }
    b.enforce_lc_times_one_eq_const(lc);
    v
}

fn alloc_u64_base3<F: PrimeField>(b: &mut Dr1csBuilder<F>, x: u64) -> DigNum {
    let mut digs = [0usize; NDIG64];
    let mut t = x;
    for i in 0..NDIG64 {
        let di = (t % BASE) as u8;
        digs[i] = alloc_trit::<F>(b, di);
        t /= BASE;
    }
    DigNum { digs }
}

fn alloc_u64_bytes<F: PrimeField>(b: &mut Dr1csBuilder<F>, x: u64) -> ByteNum {
    let mut bytes = [0usize; 8];
    let mut t = x;
    for i in 0..8 {
        bytes[i] = alloc_byte::<F>(b, (t & 0xff) as u8);
        t >>= 8;
    }
    ByteNum { bytes }
}

/// Allocate an unconstrained small carry (0..max) by base-3 digits (length 5 covers 0..242).
fn alloc_carry<F: PrimeField>(b: &mut Dr1csBuilder<F>, val: u16) -> usize {
    debug_assert!(val < 243);
    let mut digs = [0usize; 5];
    let mut t = val as u64;
    for i in 0..5 {
        digs[i] = alloc_trit::<F>(b, (t % 3) as u8);
        t /= 3;
    }
    let v = b.new_var(F::from(val as u64));
    // v = Σ 3^i * digs[i]
    let mut lc = Vec::with_capacity(1 + 5);
    lc.push((F::ONE, v));
    let mut p = F::ONE;
    for &di in &digs {
        lc.push((-p, di));
        p *= F::from(3u64);
    }
    b.enforce_lc_times_one_eq_const(lc);
    v
}

fn mul_raw_base3<F: PrimeField>(b: &mut Dr1csBuilder<F>, a: &DigNum, c: &DigNum) -> DigNum128 {
    // product terms p[i][j] = a_i * b_j (each in {0,1,2} => product in {0,1,2,4})
    let mut prod_vars: Vec<usize> = Vec::with_capacity(NDIG64 * NDIG64);
    for i in 0..NDIG64 {
        for j in 0..NDIG64 {
            let v = b.new_var(b.assignment[a.digs[i]] * b.assignment[c.digs[j]]);
            b.enforce_mul(a.digs[i], c.digs[j], v);
            prod_vars.push(v);
        }
    }

    let base_f = F::from(3u64);
    let mut out = [0usize; NDIG128];
    let mut carry = const_var::<F>(b, F::ZERO);
    for k in 0..NDIG128 {
        // sum_k = carry + Σ_{i+j=k} prod[i,j]
        let mut terms: Vec<(F, usize)> = Vec::new();
        terms.push((F::ONE, carry));
        for i in 0..NDIG64 {
            let j = k.wrapping_sub(i);
            if j < NDIG64 {
                let pv = prod_vars[i * NDIG64 + j];
                terms.push((F::ONE, pv));
            }
        }
        let sum = b.new_var(terms.iter().fold(F::ZERO, |acc, (cc, idx)| acc + (*cc * b.assignment[*idx])));
        b.enforce_lc_times_one_eq_const({
            let mut lc = vec![(F::ONE, sum)];
            for (cc, idx) in terms {
                lc.push((-cc, idx));
            }
            lc
        });

        // Witness carry/out from the field element sum (which is < 257 as an integer by design).
        let sum_u = b.assignment[sum].into_bigint().to_bytes_le().get(0).copied().unwrap_or(0) as u16;
        let out_d = alloc_trit::<F>(b, (sum_u % 3) as u8);
        let carry_u = (sum_u / 3) as u16;
        let carry_d = alloc_carry::<F>(b, carry_u);

        // sum = out_d + 3*carry_d
        b.enforce_lc_times_one_eq_const(vec![
            (F::ONE, sum),
            (-F::ONE, out_d),
            (-base_f, carry_d),
        ]);
        out[k] = out_d;
        carry = carry_d;
    }
    DigNum128 { digs: out }
}

fn alloc_carry_trits<F: PrimeField>(b: &mut Dr1csBuilder<F>, val: u16, n_trits: usize) -> usize {
    // Allocate val in base-3 with n_trits digits.
    let mut digs: Vec<usize> = Vec::with_capacity(n_trits);
    let mut t = val as u64;
    for _ in 0..n_trits {
        digs.push(alloc_trit::<F>(b, (t % 3) as u8));
        t /= 3;
    }
    let v = b.new_var(F::from(val as u64));
    let mut lc = Vec::with_capacity(1 + n_trits);
    lc.push((F::ONE, v));
    let mut p = F::ONE;
    for &di in &digs {
        lc.push((-p, di));
        p *= F::from(3u64);
    }
    b.enforce_lc_times_one_eq_const(lc);
    v
}

fn frog_p_digits_base3() -> [u8; NDIG64] {
    let mut out = [0u8; NDIG64];
    let mut t = FROG_P as u128;
    for i in 0..NDIG64 {
        out[i] = (t % 3) as u8;
        t /= 3;
    }
    out
}

fn frog_p_bytes_le() -> [u8; 8] {
    (FROG_P as u64).to_le_bytes()
}

fn alloc_dignum_from_u64<F: PrimeField>(b: &mut Dr1csBuilder<F>, x: u64) -> DigNum {
    alloc_u64_base3::<F>(b, x)
}

fn extend_64_to_128<F: PrimeField>(b: &mut Dr1csBuilder<F>, x: &DigNum) -> DigNum128 {
    let mut digs = [0usize; NDIG128];
    for i in 0..NDIG64 {
        digs[i] = x.digs[i];
    }
    let z = const_var::<F>(b, F::ZERO);
    for i in NDIG64..NDIG128 {
        digs[i] = z;
    }
    DigNum128 { digs }
}

fn u64_digits_base3(mut x: u64) -> [u8; NDIG64] {
    let mut out = [0u8; NDIG64];
    for i in 0..NDIG64 {
        out[i] = (x % 3) as u8;
        x /= 3;
    }
    out
}

fn mul_raw_base3_const_digits<F: PrimeField>(
    b: &mut Dr1csBuilder<F>,
    a: &DigNum,
    c_digits: &[u8; NDIG64],
) -> DigNum128 {
    // Convolution with constant digits in {0,1,2}.
    let base_f = F::from(3u64);
    let mut out = [0usize; NDIG128];
    let mut carry = const_var::<F>(b, F::ZERO);
    for k in 0..NDIG128 {
        let mut sum_val = b.assignment[carry];
        let mut lc: Vec<(F, usize)> = vec![(F::ONE, carry)];
        for i in 0..NDIG64 {
            let j = k.wrapping_sub(i);
            if j < NDIG64 {
                let cd = c_digits[j];
                if cd == 0 {
                    continue;
                } else if cd == 1 {
                    lc.push((F::ONE, a.digs[i]));
                    sum_val += b.assignment[a.digs[i]];
                } else {
                    // 2 * a_i (linear)
                    lc.push((F::from(2u64), a.digs[i]));
                    sum_val += b.assignment[a.digs[i]] * F::from(2u64);
                }
            }
        }
        // sum = lc
        let sum = b.new_var(sum_val);
        lc.insert(0, (F::ONE, sum));
        for idx in 1..lc.len() {
            lc[idx].0 = -lc[idx].0;
        }
        b.enforce_lc_times_one_eq_const(lc);

        // sum_u is safe (<257) by construction (base3 digits).
        let sum_u = b.assignment[sum]
            .into_bigint()
            .to_bytes_le()
            .get(0)
            .copied()
            .unwrap_or(0) as u16;
        let out_d = alloc_trit::<F>(b, (sum_u % 3) as u8);
        let carry_u = (sum_u / 3) as u16;
        let carry_d = alloc_carry::<F>(b, carry_u);
        b.enforce_lc_times_one_eq_const(vec![
            (F::ONE, sum),
            (-F::ONE, out_d),
            (-base_f, carry_d),
        ]);
        out[k] = out_d;
        carry = carry_d;
    }
    DigNum128 { digs: out }
}

fn add_raw_base3_128<F: PrimeField>(b: &mut Dr1csBuilder<F>, a: &DigNum128, c: &DigNum128) -> DigNum128 {
    let base_f = F::from(3u64);
    let mut out = [0usize; NDIG128];
    let mut carry = const_var::<F>(b, F::ZERO);
    for k in 0..NDIG128 {
        let sum_val = b.assignment[a.digs[k]] + b.assignment[c.digs[k]] + b.assignment[carry];
        let sum = b.new_var(sum_val);
        b.enforce_lc_times_one_eq_const(vec![
            (F::ONE, sum),
            (-F::ONE, a.digs[k]),
            (-F::ONE, c.digs[k]),
            (-F::ONE, carry),
        ]);
        let sum_u = b.assignment[sum]
            .into_bigint()
            .to_bytes_le()
            .get(0)
            .copied()
            .unwrap_or(0) as u16;
        let out_d = alloc_trit::<F>(b, (sum_u % 3) as u8);
        let carry_u = (sum_u / 3) as u16;
        let carry_d = alloc_carry::<F>(b, carry_u);
        b.enforce_lc_times_one_eq_const(vec![
            (F::ONE, sum),
            (-F::ONE, out_d),
            (-base_f, carry_d),
        ]);
        out[k] = out_d;
        carry = carry_d;
    }
    DigNum128 { digs: out }
}

/// Compute x * k as base-3 digits with carry, where k is a 64-bit constant (Frog field element).
///
/// This uses **no multiplication constraints**: k's digits are in {0,1,2} and each term is either 0, x_i, or 2*x_i.
fn const_mul_raw_base3_u64<F: PrimeField>(b: &mut Dr1csBuilder<F>, x: &DigNum, k: u64) -> DigNum128 {
    let k_digits = u64_digits_base3(k);
    let base_f = F::from(3u64);
    let mut out = [0usize; NDIG128];
    let mut carry = const_var::<F>(b, F::ZERO);
    for j in 0..NDIG128 {
        let mut sum_val = b.assignment[carry];
        // Build linear combination for sum (carry + Σ x_i * k_{j-i}).
        let mut terms: Vec<(F, usize)> = vec![(F::ONE, carry)];
        for i in 0..NDIG64 {
            let kk = j.wrapping_sub(i);
            if kk < NDIG64 {
                match k_digits[kk] {
                    0 => {}
                    1 => {
                        terms.push((F::ONE, x.digs[i]));
                        sum_val += b.assignment[x.digs[i]];
                    }
                    2 => {
                        terms.push((F::from(2u64), x.digs[i]));
                        sum_val += b.assignment[x.digs[i]] * F::from(2u64);
                    }
                    _ => unreachable!(),
                }
            }
        }
        let sum = b.new_var(sum_val);
        // Enforce sum == Σ terms
        let mut lc = vec![(F::ONE, sum)];
        for (cc, idx) in terms {
            lc.push((-cc, idx));
        }
        b.enforce_lc_times_one_eq_const(lc);

        let sum_u = b.assignment[sum]
            .into_bigint()
            .to_bytes_le()
            .get(0)
            .copied()
            .unwrap_or(0) as u16;
        let out_d = alloc_trit::<F>(b, (sum_u % 3) as u8);
        let carry_u = (sum_u / 3) as u16;
        // carry bound for one digit: max about 82 (fits in 5 trits)
        let carry_d = alloc_carry_trits::<F>(b, carry_u, 5);
        b.enforce_lc_times_one_eq_const(vec![
            (F::ONE, sum),
            (-F::ONE, out_d),
            (-base_f, carry_d),
        ]);
        out[j] = out_d;
        carry = carry_d;
    }
    DigNum128 { digs: out }
}

/// Add 24 base-3 digit numbers (normalized digits 0..2) with carry.
///
/// This is intended for MDS accumulation, where we sum 24 constant-mul products per output word.
fn add_many_raw_base3_128<F: PrimeField>(b: &mut Dr1csBuilder<F>, xs: &[DigNum128]) -> DigNum128 {
    let base_f = F::from(3u64);
    let mut out = [0usize; NDIG128];
    let mut carry = const_var::<F>(b, F::ZERO);
    for j in 0..NDIG128 {
        let mut sum_val = b.assignment[carry];
        let mut terms: Vec<(F, usize)> = vec![(F::ONE, carry)];
        for x in xs {
            terms.push((F::ONE, x.digs[j]));
            sum_val += b.assignment[x.digs[j]];
        }
        let sum = b.new_var(sum_val);
        let mut lc = vec![(F::ONE, sum)];
        for (cc, idx) in terms {
            lc.push((-cc, idx));
        }
        b.enforce_lc_times_one_eq_const(lc);

        let sum_u = b.assignment[sum]
            .into_bigint()
            .to_bytes_le()
            .get(0)
            .copied()
            .unwrap_or(0) as u16;
        let out_d = alloc_trit::<F>(b, (sum_u % 3) as u8);
        let carry_u = (sum_u / 3) as u16;
        // For sum of 24 digits in {0,1,2}, carry fixed point <= 24; 4 trits (81) suffices.
        let carry_d = alloc_carry_trits::<F>(b, carry_u, 4);
        b.enforce_lc_times_one_eq_const(vec![
            (F::ONE, sum),
            (-F::ONE, out_d),
            (-base_f, carry_d),
        ]);
        out[j] = out_d;
        carry = carry_d;
    }
    DigNum128 { digs: out }
}

/// Constrain z ≡ a*b (mod FROG_P) using quotient q:
///   a*b = z + q*FROG_P  (as integers in base 3).
///
/// NOTE: This currently does not enforce `z < p` canonically (multiple reps possible).
fn mul_mod_frog<F: PrimeField>(b: &mut Dr1csBuilder<F>, a: &DigNum, c: &DigNum) -> DigNum {
    // Witness z and q from current assignment (best-effort): use u64 reps reduced mod p.
    // For constraint-counting we don't need perfect witnesses beyond satisfying this test harness.
    let a_u = dignum_to_u64::<F>(b, a);
    let c_u = dignum_to_u64::<F>(b, c);
    let prod = (a_u as u128) * (c_u as u128);
    let z_u = (prod % (FROG_P as u128)) as u64;
    let q_u = (prod / (FROG_P as u128)) as u64;

    let z = alloc_dignum_from_u64::<F>(b, z_u);
    let q = alloc_dignum_from_u64::<F>(b, q_u);

    let prod_d = mul_raw_base3::<F>(b, a, c);
    let p_digits = frog_p_digits_base3();
    let qp_d = mul_raw_base3_const_digits::<F>(b, &q, &p_digits);
    let z_ext = extend_64_to_128::<F>(b, &z);
    let sum = add_raw_base3_128::<F>(b, &z_ext, &qp_d);

    // Enforce digitwise equality: prod_d == sum
    for i in 0..NDIG128 {
        b.enforce_lc_times_one_eq_const(vec![
            (F::ONE, prod_d.digs[i]),
            (-F::ONE, sum.digs[i]),
        ]);
    }
    z
}

fn dignum_to_u64<F: PrimeField>(b: &Dr1csBuilder<F>, x: &DigNum) -> u64 {
    let mut acc: u128 = 0;
    let mut pow: u128 = 1;
    for i in 0..NDIG64 {
        let di = b.assignment[x.digs[i]]
            .into_bigint()
            .to_bytes_le()
            .get(0)
            .copied()
            .unwrap_or(0) as u128;
        acc += di * pow;
        pow *= 3;
    }
    (acc as u64) % FROG_P
}

/// Count constraints for one raw base-3 multiplication (no mod reduction).
pub fn count_one_raw_mul<F: PrimeField>() -> (usize, usize) {
    let mut b = Dr1csBuilder::<F>::new();
    let a = alloc_u64_base3::<F>(&mut b, 123456789);
    let c = alloc_u64_base3::<F>(&mut b, 987654321);
    let _p = mul_raw_base3::<F>(&mut b, &a, &c);
    let (inst, _asg) = b.into_instance();
    (inst.nvars, inst.constraints.len())
}

/// Count constraints for one modular multiplication (with q*p reduction).
pub fn count_one_mod_mul<F: PrimeField>() -> (usize, usize) {
    let mut b = Dr1csBuilder::<F>::new();
    let a = alloc_u64_base3::<F>(&mut b, 123456789);
    let c = alloc_u64_base3::<F>(&mut b, 987654321);
    let _z = mul_mod_frog::<F>(&mut b, &a, &c);
    let (inst, _asg) = b.into_instance();
    (inst.nvars, inst.constraints.len())
}

fn add_mod_frog<F: PrimeField>(b: &mut Dr1csBuilder<F>, a: &DigNum, c: &DigNum) -> DigNum {
    let a_u = dignum_to_u64::<F>(b, a);
    let c_u = dignum_to_u64::<F>(b, c);
    let s = (a_u as u128) + (c_u as u128);
    let z_u = (s % (FROG_P as u128)) as u64;
    let q_bit = if s >= (FROG_P as u128) { 1u8 } else { 0u8 };

    let z = alloc_dignum_from_u64::<F>(b, z_u);
    let q = alloc_bool::<F>(b, q_bit == 1);
    let p_digits = frog_p_digits_base3();

    let base_f = F::from(3u64);
    let mut carry = const_var::<F>(b, F::ZERO);
    let mut out = [0usize; NDIG64];
    for i in 0..NDIG64 {
        // LHS digit sum: a_i + b_i + carry
        // RHS: z_i + q * p_i + 3*carry_next
        let carry_next_u = {
            // witness from current assignment values (safe small)
            let lhs = b.assignment[a.digs[i]] + b.assignment[c.digs[i]] + b.assignment[carry];
            let rhs0 = b.assignment[z.digs[i]] + (F::from(p_digits[i] as u64) * b.assignment[q]);
            // Convert to integer in [0,256]
            let lhs_u = lhs.into_bigint().to_bytes_le().get(0).copied().unwrap_or(0) as i16;
            let rhs_u = rhs0.into_bigint().to_bytes_le().get(0).copied().unwrap_or(0) as i16;
            // carry_next = (lhs - rhs) / 3, adjusted in [0,2]
            let mut diff = lhs_u - rhs_u;
            // normalize into [0, 6] by adding/subtracting multiples of 3
            while diff < 0 {
                diff += 3;
            }
            while diff >= 9 {
                diff -= 3;
            }
            ((diff / 3) as u8).min(2)
        };
        let carry_next = alloc_trit::<F>(b, carry_next_u);

        let pd = F::from(p_digits[i] as u64);
        // Enforce: a_i + c_i + carry - z_i - pd*q - 3*carry_next == 0
        b.enforce_lc_times_one_eq_const(vec![
            (F::ONE, a.digs[i]),
            (F::ONE, c.digs[i]),
            (F::ONE, carry),
            (-F::ONE, z.digs[i]),
            (-pd, q),
            (-base_f, carry_next),
        ]);
        out[i] = z.digs[i];
        carry = carry_next;
    }
    DigNum { digs: out }
}

pub fn count_one_add_mod<F: PrimeField>() -> (usize, usize) {
    let mut b = Dr1csBuilder::<F>::new();
    let a = alloc_u64_base3::<F>(&mut b, 123456789);
    let c = alloc_u64_base3::<F>(&mut b, 987654321);
    let _z = add_mod_frog::<F>(&mut b, &a, &c);
    let (inst, _asg) = b.into_instance();
    (inst.nvars, inst.constraints.len())
}

pub fn count_one_const_mul<F: PrimeField>() -> (usize, usize) {
    let mut b = Dr1csBuilder::<F>::new();
    let a = alloc_u64_base3::<F>(&mut b, 123456789);
    let _p = const_mul_raw_base3_u64::<F>(&mut b, &a, 0xdeadbeefdeadbeefu64);
    let (inst, _asg) = b.into_instance();
    (inst.nvars, inst.constraints.len())
}

pub fn count_one_mds_row<F: PrimeField>() -> (usize, usize) {
    // Model one MDS output = Σ_j m_j * s_j for j=0..23 with constant m_j,
    // where each s_j is a Frog field element (u64).
    let mut b = Dr1csBuilder::<F>::new();
    let mut state: Vec<DigNum> = Vec::with_capacity(24);
    for j in 0..24 {
        state.push(alloc_u64_base3::<F>(&mut b, (j as u64) + 1));
    }
    let mut prods: Vec<DigNum128> = Vec::with_capacity(24);
    for j in 0..24 {
        // Use arbitrary-looking constants (not actual MDS yet; we're measuring cost shape).
        let k = 0x9e3779b97f4a7c15u64.wrapping_mul((j as u64) + 1);
        prods.push(const_mul_raw_base3_u64::<F>(&mut b, &state[j], k));
    }
    let _sum = add_many_raw_base3_128::<F>(&mut b, &prods);
    let (inst, _asg) = b.into_instance();
    (inst.nvars, inst.constraints.len())
}

/// Estimate constraints for one Frog Poseidon permutation using measured per-op costs.
pub fn estimate_one_frog_poseidon_perm_constraints(mod_mul_constraints: u64, add_constraints: u64) -> u64 {
    // From Frog Poseidon params: width=24, full_rounds=8, partial_rounds=22, alpha=7.
    // - MDS per round: 24 outputs, each = Σ_{j=0..23} mds_ij * state_j
    //   We model as 24 constant-muls + 23 adds per output.
    //   => muls_per_round = 24*24 = 576
    //      adds_per_round = 24*23 = 552
    // - Ark add per round: 24 adds
    // - S-box x^7: 4 muls.
    //   Full rounds: 8*24 S-boxes, partial rounds: 22*1.
    //   => sbox_muls = 4*(8*24 + 22) = 856
    let rounds = 30u64;
    let muls_mds = 576u64 * rounds;
    let adds_mds = 552u64 * rounds;
    let adds_ark = 24u64 * rounds;
    let muls_sbox = 856u64;
    let muls_total = muls_mds + muls_sbox;
    let adds_total = adds_mds + adds_ark;
    muls_total.saturating_mul(mod_mul_constraints) + adds_total.saturating_mul(add_constraints)
}

/// Estimate number of permutations for a Poseidon sponge absorb of `n_elems` field elements.
///
/// Frog config uses rate=20, capacity=4 (see `PoseidonConfig::new(..., 20, 4)`).
pub fn estimate_sponge_perm_count(n_elems: u64) -> u64 {
    let rate = 20u64;
    // Very rough: assume 1 permutation per full rate block, plus one finalization permute.
    // (Arkworks sponge behavior can differ slightly with padding/domain separation.)
    (n_elems + rate - 1) / rate + 1
}

fn canonicalize_with_qbit<F: PrimeField>(
    b: &mut Dr1csBuilder<F>,
    lane: &DigNum,
    q_bit: u8,
) -> DigNum {
    // Enforce: lane = z + q*p  with q in {0,1}, z is canonical representative.
    let q = alloc_bool::<F>(b, q_bit == 1);
    let z_u = {
        let lane_u = dignum_to_u64::<F>(b, lane) as u128;
        if q_bit == 1 {
            lane_u.saturating_sub(FROG_P as u128) as u64
        } else {
            lane_u as u64
        }
    };
    let z = alloc_dignum_from_u64::<F>(b, z_u % FROG_P);
    let p_digits = frog_p_digits_base3();
    let base_f = F::from(3u64);
    let mut carry = const_var::<F>(b, F::ZERO);
    for i in 0..NDIG64 {
        // lane_i + carry - z_i - q*p_i - 3*carry_next == 0
        let carry_next_u = {
            let lhs = b.assignment[lane.digs[i]] + b.assignment[carry];
            let rhs0 = b.assignment[z.digs[i]] + (F::from(p_digits[i] as u64) * b.assignment[q]);
            let lhs_u = lhs.into_bigint().to_bytes_le().get(0).copied().unwrap_or(0) as i16;
            let rhs_u = rhs0.into_bigint().to_bytes_le().get(0).copied().unwrap_or(0) as i16;
            let mut diff = lhs_u - rhs_u;
            while diff < 0 {
                diff += 3;
            }
            while diff >= 9 {
                diff -= 3;
            }
            ((diff / 3) as u8).min(2)
        };
        let carry_next = alloc_trit::<F>(b, carry_next_u);
        b.enforce_lc_times_one_eq_const(vec![
            (F::ONE, lane.digs[i]),
            (F::ONE, carry),
            (-F::ONE, z.digs[i]),
            (-F::from(p_digits[i] as u64), q),
            (-base_f, carry_next),
        ]);
        carry = carry_next;
    }
    z
}

pub fn count_one_boundary_canon<F: PrimeField>() -> (usize, usize) {
    let mut b = Dr1csBuilder::<F>::new();
    // Build a lane in [0,2p) to allow q in {0,1}.
    let lane_u = (FROG_P / 2) + 12345;
    let lane = alloc_u64_base3::<F>(&mut b, lane_u);
    let _z = canonicalize_with_qbit::<F>(&mut b, &lane, 0);
    let (inst, _asg) = b.into_instance();
    (inst.nvars, inst.constraints.len())
}

fn canonicalize_bytes_with_qbit<F: PrimeField>(
    b: &mut Dr1csBuilder<F>,
    lane: &ByteNum,
    q_bit: u8,
) -> ByteNum {
    // Enforce: lane = z + q*p  with q in {0,1}, z is canonical representative.
    let q = alloc_bool::<F>(b, q_bit == 1);
    let z_u = {
        let lane_u = bytes_to_u64::<F>(b, lane) as u128;
        if q_bit == 1 {
            lane_u.saturating_sub(FROG_P as u128) as u64
        } else {
            lane_u as u64
        }
    };
    let z = alloc_u64_bytes::<F>(b, z_u % FROG_P);
    let p_bytes = frog_p_bytes_le();
    let base_f = F::from(256u64);
    let mut carry = const_var::<F>(b, F::ZERO);
    for i in 0..8 {
        // lane_i + carry - z_i - q*p_i - 256*carry_next == 0
        let carry_next_u = {
            let lhs = b.assignment[lane.bytes[i]] + b.assignment[carry];
            let rhs0 = b.assignment[z.bytes[i]] + (F::from(p_bytes[i] as u64) * b.assignment[q]);
            let lhs_u = lhs.into_bigint().to_bytes_le().get(0).copied().unwrap_or(0) as i16;
            let rhs_u = rhs0.into_bigint().to_bytes_le().get(0).copied().unwrap_or(0) as i16;
            let mut diff = lhs_u - rhs_u;
            while diff < 0 {
                diff += 256;
            }
            while diff >= 512 {
                diff -= 256;
            }
            ((diff / 256) as u8).min(1)
        };
        let carry_next = alloc_bool::<F>(b, carry_next_u == 1);
        b.enforce_lc_times_one_eq_const(vec![
            (F::ONE, lane.bytes[i]),
            (F::ONE, carry),
            (-F::ONE, z.bytes[i]),
            (-F::from(p_bytes[i] as u64), q),
            (-base_f, carry_next),
        ]);
        carry = carry_next;
    }
    z
}

fn bytes_to_u64<F: PrimeField>(b: &Dr1csBuilder<F>, x: &ByteNum) -> u64 {
    let mut acc: u128 = 0;
    let mut pow: u128 = 1;
    for i in 0..8 {
        let bi = b.assignment[x.bytes[i]]
            .into_bigint()
            .to_bytes_le()
            .get(0)
            .copied()
            .unwrap_or(0) as u128;
        acc += bi * pow;
        pow <<= 8;
    }
    (acc as u64) % FROG_P
}

pub fn count_one_boundary_canon_bytes<F: PrimeField>() -> (usize, usize) {
    let mut b = Dr1csBuilder::<F>::new();
    // Build a lane in [0,2p) to allow q in {0,1}.
    let lane_u = (FROG_P / 2) + 12345;
    let lane = alloc_u64_bytes::<F>(&mut b, lane_u);
    let _z = canonicalize_bytes_with_qbit::<F>(&mut b, &lane, 0);
    let (inst, _asg) = b.into_instance();
    (inst.nvars, inst.constraints.len())
}

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
        let (nvars0, ncons0) = count_one_raw_mul::<F257>();
        let (nvars1, ncons1) = count_one_mod_mul::<F257>();
        let (nvars2, ncons2) = count_one_add_mod::<F257>();
        let (_nvars3, ncons3) = count_one_const_mul::<F257>();
        let (_nvars4, ncons4) = count_one_mds_row::<F257>();
        let (_nvars5, ncons5) = count_one_boundary_canon::<F257>();
        let (_nvars6, ncons6) = count_one_boundary_canon_bytes::<F257>();
        eprintln!("[we_frog_poseidon_f257] raw_mul_base3 nvars={nvars0} constraints={ncons0}");
        eprintln!("[we_frog_poseidon_f257] mod_mul_base3 nvars={nvars1} constraints={ncons1}");
        eprintln!("[we_frog_poseidon_f257] add_mod_base3 nvars={nvars2} constraints={ncons2}");
        eprintln!("[we_frog_poseidon_f257] const_mul_raw_base3 constraints={ncons3}");
        eprintln!("[we_frog_poseidon_f257] mds_row_raw_base3 constraints={ncons4}");
        eprintln!("[we_frog_poseidon_f257] boundary_canon(qbit) constraints={ncons5}");
        eprintln!("[we_frog_poseidon_f257] boundary_canon_bytes(qbit) constraints={ncons6}");

        // Old pessimistic estimate:
        let est_perm = estimate_one_frog_poseidon_perm_constraints(ncons1 as u64, ncons2 as u64);
        eprintln!("[we_frog_poseidon_f257] est_one_perm_constraints_pess≈{est_perm}");

        // Refined (still rough) estimate:
        // - MDS: 24 outputs/round => 24 * mds_row constraints/round
        // - Ark: 24 additions/round (use add_mod)
        // - S-box: 4 mod-muls per S-box (full rounds: 8*24, partial: 22*1)
        let rounds = 30u64;
        let mds = (24u64 * rounds).saturating_mul(ncons4 as u64);
        let ark = (24u64 * rounds).saturating_mul(ncons2 as u64);
        let sbox_muls = 4u64 * (8u64 * 24u64 + 22u64);
        let sbox = sbox_muls.saturating_mul(ncons1 as u64);
        let est_perm2 = mds + ark + sbox;
        eprintln!("[we_frog_poseidon_f257] est_one_perm_constraints_refined≈{est_perm2}");

        // Example: absorb c_stmt||x where c_stmt=256 elems, x=1 elem.
        let n_absorb = 257u64;
        let perms = estimate_sponge_perm_count(n_absorb);
        eprintln!("[we_frog_poseidon_f257] est_sponge_permutes(n_absorb={n_absorb})={perms}");
        eprintln!(
            "[we_frog_poseidon_f257] est_sponge_constraints≈{}",
            perms.saturating_mul(est_perm2)
        );
        // Boundary-only canonicalization (base-3) estimate:
        // For n=64 squeeze_bytes, usable_bytes=7 => 10 field elems per op.
        let canon_elems = 515u64 * 10u64;
        let canon_cost = canon_elems.saturating_mul(ncons5 as u64);
        eprintln!(
            "[we_frog_poseidon_f257] est_boundary_canon_constraints≈{} (elems={canon_elems})",
            canon_cost
        );
        // Boundary-only canonicalization (base-256 bytes) estimate:
        let canon_cost_bytes = canon_elems.saturating_mul(ncons6 as u64);
        eprintln!(
            "[we_frog_poseidon_f257] est_boundary_canon_bytes_constraints≈{} (elems={canon_elems})",
            canon_cost_bytes
        );
        assert!(nvars0 > 0 && ncons0 > 0);
        assert!(nvars1 > 0 && ncons1 > 0);
        assert!(nvars2 > 0 && ncons2 > 0);
    }
}

