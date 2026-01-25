//! Experimental: emulate Frog Poseidon inside a tiny field (F257) dR1CS.
//!
//! Byte-limb (base-256) gadgets for boundary-only canonicalization.
//! This module intentionally avoids base-3 digit arithmetic.

use ark_ff::{BigInteger, PrimeField};
use symphony::dpp_sumcheck::Dr1csBuilder;

const FROG_P: u64 = 15912092521325583641u64;

#[derive(Clone, Debug)]
struct ByteVar {
    byte: usize, // 0..255
}

#[derive(Clone, Debug)]
struct ByteNum {
    bytes: [ByteVar; 8], // little-endian bytes
}

fn alloc_small<F: PrimeField>(b: &mut Dr1csBuilder<F>, val: u16, bits: usize) -> usize {
    let mut lc = vec![(F::ONE, b.new_var(F::from(val as u64)))];
    let v = lc[0].1;
    let mut pow = F::ONE;
    for i in 0..bits {
        let bi = alloc_bool::<F>(b, ((val >> i) & 1) == 1);
        lc.push((-pow, bi));
        pow *= F::from(2u64);
    }
    b.enforce_lc_times_one_eq_const(lc);
    v
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
    ByteVar { byte: v }
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

fn alloc_u128_bytes<F: PrimeField>(b: &mut Dr1csBuilder<F>, x: u128) -> Vec<ByteVar> {
    let mut bytes: Vec<ByteVar> = Vec::with_capacity(16);
    let mut t = x;
    for _ in 0..16 {
        bytes.push(alloc_byte::<F>(b, (t & 0xff) as u8));
        t >>= 8;
    }
    bytes
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

fn mul_u64_bytes_mod_p<F: PrimeField>(
    b: &mut Dr1csBuilder<F>,
    x: &ByteNum,
    y: &ByteNum,
) -> (ByteNum, Vec<usize>) {
    let x_u = bytes_to_u64::<F>(b, x) as u128;
    let y_u = bytes_to_u64::<F>(b, y) as u128;
    let prod_u = x_u * y_u;

    let prod_bytes = alloc_u128_bytes::<F>(b, prod_u);

    // Enforce prod = x * y via schoolbook with carries.
    let x_b: [u16; 8] = {
        let mut out = [0u16; 8];
        for i in 0..8 {
            out[i] = b.assignment[x.bytes[i].byte]
                .into_bigint()
                .to_bytes_le()
                .get(0)
                .copied()
                .unwrap_or(0) as u16;
        }
        out
    };
    let y_b: [u16; 8] = {
        let mut out = [0u16; 8];
        for i in 0..8 {
            out[i] = b.assignment[y.bytes[i].byte]
                .into_bigint()
                .to_bytes_le()
                .get(0)
                .copied()
                .unwrap_or(0) as u16;
        }
        out
    };
    let mut carry_val: u32 = 0;
    let mut carry_vars: Vec<usize> = Vec::new();
    for k in 0..16 {
        let mut sum: u32 = carry_val;
        for i in 0..8 {
            for j in 0..8 {
                if i + j == k {
                    sum += (x_b[i] as u32) * (y_b[j] as u32);
                }
            }
        }
        let out_k = prod_bytes[k].byte;
        let out_val = (prod_u >> (8 * k)) as u8;
        let next = (sum - (out_val as u32)) / 256;
        let carry_next = if k == 15 { 0 } else { next as u16 };
        let carry_next_var = if k == 15 {
            const_var::<F>(b, F::ZERO)
        } else {
            alloc_small::<F>(b, carry_next, 12)
        };
        let mut lc: Vec<(F, usize)> = Vec::new();
        lc.push((F::ONE, out_k));
        lc.push((F::from(256u64), carry_next_var));
        lc.push((-F::ONE, const_var::<F>(b, F::from(out_val as u64))));
        if carry_val != 0 {
            let carry_var = if k == 0 {
                const_var::<F>(b, F::ZERO)
            } else {
                carry_vars[k - 1]
            };
            lc.push((-F::ONE, carry_var));
        }
        for i in 0..8 {
            for j in 0..8 {
                if i + j == k {
                    let m = b.new_var(
                        b.assignment[x.bytes[i].byte] * b.assignment[y.bytes[j].byte],
                    );
                    b.enforce_mul(x.bytes[i].byte, y.bytes[j].byte, m);
                    lc.push((-F::ONE, m));
                }
            }
        }
        b.enforce_lc_times_one_eq_const(lc);
        if k != 15 {
            carry_vars.push(carry_next_var);
        }
        carry_val = carry_next as u32;
    }

    // Reduction: prod = z + q*p
    let q_u = (prod_u / (FROG_P as u128)) as u64;
    let z_u = (prod_u % (FROG_P as u128)) as u64;
    let q = alloc_u64_bytes::<F>(b, q_u);
    let z = alloc_u64_bytes::<F>(b, z_u);

    let p_bytes = frog_p_bytes_le();
    let mut qp_bytes: Vec<ByteVar> = Vec::with_capacity(16);
    let mut carry_val2: u32 = 0;
    let mut carry_vars2: Vec<usize> = Vec::new();
    for k in 0..16 {
        let mut sum: u32 = carry_val2;
        for i in 0..8 {
            for j in 0..8 {
                if i + j == k {
                    sum += (b.assignment[q.bytes[i].byte]
                        .into_bigint()
                        .to_bytes_le()
                        .get(0)
                        .copied()
                        .unwrap_or(0) as u32)
                        * (p_bytes[j] as u32);
                }
            }
        }
        let out_val = (sum & 0xff) as u8;
        let out = alloc_byte::<F>(b, out_val);
        let next = (sum - (out_val as u32)) / 256;
        let carry_next = if k == 15 { 0 } else { next as u16 };
        let carry_next_var = if k == 15 {
            const_var::<F>(b, F::ZERO)
        } else {
            alloc_small::<F>(b, carry_next, 12)
        };
        let mut lc: Vec<(F, usize)> = Vec::new();
        lc.push((F::ONE, out.byte));
        lc.push((F::from(256u64), carry_next_var));
        lc.push((-F::ONE, const_var::<F>(b, F::from(out_val as u64))));
        if carry_val2 != 0 {
            let carry_var = if k == 0 {
                const_var::<F>(b, F::ZERO)
            } else {
                carry_vars2[k - 1]
            };
            lc.push((-F::ONE, carry_var));
        }
        for i in 0..8 {
            for j in 0..8 {
                if i + j == k {
                    let coeff = F::from(p_bytes[j] as u64);
                    if coeff != F::ZERO {
                        lc.push((-coeff, q.bytes[i].byte));
                    }
                }
            }
        }
        b.enforce_lc_times_one_eq_const(lc);
        if k != 15 {
            carry_vars2.push(carry_next_var);
        }
        carry_val2 = carry_next as u32;
        qp_bytes.push(out);
    }

    // Enforce prod = z + q*p (byte-wise with carry).
    let mut carry_val3: u32 = 0;
    let mut carry_vars3: Vec<usize> = Vec::new();
    for k in 0..16 {
        let lhs = prod_bytes[k].byte;
        let rhs_byte = if k < 8 { z.bytes[k].byte } else { const_var::<F>(b, F::ZERO) };
        let sum_val = (z_u as u128 + (q_u as u128) * (FROG_P as u128)) >> (8 * k);
        let out_val = (sum_val & 0xff) as u8;
        let next = ((sum_val >> 8) & 0xffff) as u16;
        let carry_next = if k == 15 { 0 } else { next };
        let carry_next_var = if k == 15 {
            const_var::<F>(b, F::ZERO)
        } else {
            alloc_small::<F>(b, carry_next, 12)
        };
        let mut lc: Vec<(F, usize)> = Vec::new();
        lc.push((F::ONE, lhs));
        lc.push((-F::ONE, rhs_byte));
        lc.push((-F::ONE, qp_bytes[k].byte));
        lc.push((F::from(256u64), carry_next_var));
        if carry_val3 != 0 {
            let carry_var = if k == 0 {
                const_var::<F>(b, F::ZERO)
            } else {
                carry_vars3[k - 1]
            };
            lc.push((-F::ONE, carry_var));
        }
        b.enforce_lc_times_one_eq_const(lc);
        if k != 15 {
            carry_vars3.push(carry_next_var);
        }
        carry_val3 = carry_next as u32;
        let _ = out_val;
    }

    (z, carry_vars)
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

pub fn count_one_mul_mod_frog<F: PrimeField>() -> (usize, usize) {
    let mut b = Dr1csBuilder::<F>::new();
    let x = alloc_u64_bytes::<F>(&mut b, 123456789u64);
    let y = alloc_u64_bytes::<F>(&mut b, 987654321u64);
    let _z = mul_u64_bytes_mod_p::<F>(&mut b, &x, &y);
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

    #[test]
    fn print_mul_mod_frog_constraint_count() {
        let (nvars, ncons) = count_one_mul_mod_frog::<F257>();
        eprintln!(
            "[we_frog_poseidon_f257] mul_mod_frog(u64) nvars={} constraints={}",
            nvars, ncons
        );
        assert!(nvars > 0 && ncons > 0);
    }
}

