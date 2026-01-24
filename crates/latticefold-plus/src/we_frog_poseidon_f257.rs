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

