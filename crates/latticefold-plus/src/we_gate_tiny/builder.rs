use std::collections::BTreeMap;
use std::sync::Arc;

use crate::transcript::DEFAULT_REJECTION_TRIES;

use ark_ff::Field;
use ark_crypto_primitives::sponge::poseidon::PoseidonConfig;
use latticefold::transcript::poseidon::F257;
use symphony::dpp_poseidon::{merge_sparse_dr1cs_share_one, Constraint, PoseidonDr1csWiring, SparseDr1csInstance};
use symphony::dpp_sumcheck::Dr1csBuilder;
use symphony::transcript::PoseidonTraceOp;

use crate::we_statement::WeParams;

use super::challenges::{
    bounded_u32_from_8_digits_base128, digit_to_byte_var, res257_from_u64_bytes_le,
    select_first_ok_u32_try_digits, short_challenge_from_digits_128, squeeze_field_ranges_by_op_index,
    BoundedU32ChallengeWiring, FrogChallengeWiring, ShortChallengeWiring, TinyCoinOpWiring,
};
use super::coins::{sample_frog_coin_unrolled_rejection_8_digits, FrogRejectionCoinWiring};
use super::digits::{
    rebalance_prod12_to_prod13, rebalance_prod21_to_prod22, scale_short_coeffs_by_digits18,
    scale_short_coeffs_by_digits9, sum_bal16_vectors_fixed_len, sum_product_digits_bal16,
    sum_product_digits_bal16_22, sum_products13_coeffwise_fixed_len, sum_products22_coeffwise_fixed_len,
};
use super::frog::{
    frog_u64_canonical_from_byte_vars, frog_u64_centered_le_bound_from_byte_vars,
    reduce_u64_mod_frog_from_byte_vars,
};
use super::gadgets::{decompose_existing_byte_var_to_bits, enforce_var_eq};
use super::params::DIGITS_PER_TRY;
use super::poseidon::poseidon_f257_arithmetize;
use super::surfaces::{CmDigitMulSqSurfaceWiring, CmDigitMulSurfaceWiring};

struct GlueCtx {
    gb: Dr1csBuilder<F257>,
    pose_asg: Vec<F257>,
    local_map: BTreeMap<usize, usize>,
}

impl GlueCtx {
    fn new(pose_asg: Vec<F257>) -> Self {
        let mut gb = Dr1csBuilder::<F257>::new();
        gb.enforce_var_eq_const(gb.one(), F257::ONE);
        Self { gb, pose_asg, local_map: BTreeMap::new() }
    }

    #[inline]
    fn copy_digit(&mut self, gv: usize) -> usize {
        if let Some(&lv) = self.local_map.get(&gv) {
            return lv;
        }
        let lv = self.gb.new_var(self.pose_asg[gv]);
        self.local_map.insert(gv, lv);
        lv
    }
}

fn validate_pairs(pairs: &[(usize, usize)], short_len: usize, u32_len: usize) -> Result<(), String> {
    for &(si, ui) in pairs {
        if si >= short_len {
            return Err(format!("short_block_idx {si} out of range"));
        }
        if ui >= u32_len {
            return Err(format!("u32_idx {ui} out of range"));
        }
    }
    Ok(())
}

fn build_short_blocks(
    glue: &mut GlueCtx,
    pose_wiring: &PoseidonDr1csWiring,
    ring_dim: usize,
    short_ranges: &[(usize, usize)],
) -> Result<Vec<ShortChallengeWiring>, String> {
    let mut out: Vec<ShortChallengeWiring> = Vec::with_capacity(short_ranges.len());
    for (si, (s_start, s_len)) in short_ranges.iter().copied().enumerate() {
        if s_len != ring_dim {
            return Err(format!(
                "short block {si} length mismatch (got {s_len}, expected {ring_dim})"
            ));
        }
        let short_digits_global = &pose_wiring.squeeze_field_vars[s_start..s_start + s_len];
        let mut short_digits_local: Vec<usize> = Vec::with_capacity(s_len);
        for &gv in short_digits_global {
            short_digits_local.push(glue.copy_digit(gv));
        }
        let (short_bvars, short_coeffs, short_coeff_digits) =
            short_challenge_from_digits_128(&mut glue.gb, &short_digits_local, ring_dim);
        out.push(ShortChallengeWiring {
            digit_vars: short_digits_local,
            byte_vars: short_bvars,
            coeff_vars: short_coeffs,
            coeff_bal16_digits: short_coeff_digits,
        });
    }
    Ok(out)
}

fn build_u32_and_frog_blocks(
    glue: &mut GlueCtx,
    pose_wiring: &PoseidonDr1csWiring,
    u32_starts: &[usize],
) -> Result<(Vec<BoundedU32ChallengeWiring>, Vec<FrogChallengeWiring>), String> {
    let tries: usize = DEFAULT_REJECTION_TRIES;
    let mut u32_out: Vec<BoundedU32ChallengeWiring> = Vec::with_capacity(u32_starts.len());
    let mut frog_out: Vec<FrogChallengeWiring> = Vec::with_capacity(u32_starts.len());
    for (ui, &u_start_op) in u32_starts.iter().enumerate() {
        // Copy all try digits locally (but keep wiring compact: store only the selected digits).
        let mut all_digits_local: Vec<usize> = Vec::with_capacity(tries * DIGITS_PER_TRY);
        for t in 0..tries {
            let op_idx = u_start_op + t;
            let (u_start, u_len) = pose_wiring
                .squeeze_field_ranges
                .get(op_idx)
                .copied()
                .ok_or_else(|| format!("u32 start op idx {u_start_op} + {t} out of range"))?;
            if u_len != DIGITS_PER_TRY {
                return Err(format!(
                    "u32 try block {ui}.{t} length mismatch (got {u_len}, expected {DIGITS_PER_TRY})"
                ));
            }
            for &gv in &pose_wiring.squeeze_field_vars[u_start..u_start + u_len] {
                all_digits_local.push(glue.copy_digit(gv));
            }
        }

        // Select the first acceptable try (digits[0..4] all != 256), matching `transcript.rs`.
        let (u_digits_local, found_bit) =
            select_first_ok_u32_try_digits(&mut glue.gb, &all_digits_local, tries);
        glue.gb.enforce_var_eq_const(found_bit, F257::ONE);

        let (u_limbs, u_bytes, u_bal16, u_bal16_sq) =
            bounded_u32_from_8_digits_base128(&mut glue.gb, &u_digits_local);

        let mut u64_bytes = [0usize; 8];
        u64_bytes[0..4].copy_from_slice(&u_bytes);
        for i in 4..8 {
            u64_bytes[i] = digit_to_byte_var::<F257>(&mut glue.gb, u_digits_local[i]);
        }
        let (q_bit, frog_limbs) = reduce_u64_mod_frog_from_byte_vars::<F257>(&mut glue.gb, &u64_bytes);
        let res257 = res257_from_u64_bytes_le(&mut glue.gb, &u64_bytes);

        u32_out.push(BoundedU32ChallengeWiring {
            digit_vars: u_digits_local.to_vec(),
            byte_vars: u_bytes,
            limbs: u_limbs,
            bal16_digits: u_bal16,
            bal16_sq_digits: u_bal16_sq,
        });
        frog_out.push(FrogChallengeWiring {
            digit_vars: u_digits_local.to_vec(),
            byte_vars: u64_bytes,
            q_bit,
            limbs: frog_limbs,
            res257,
        });
    }
    Ok((u32_out, frog_out))
}

fn build_frog_rejection_coins(
    glue: &mut GlueCtx,
    pose_wiring: &PoseidonDr1csWiring,
    frog_ranges: &[(usize, usize)],
) -> Result<Vec<FrogRejectionCoinWiring>, String> {
    let tries: usize = DEFAULT_REJECTION_TRIES;
    if !frog_ranges.is_empty() && (frog_ranges.len() % tries != 0) {
        return Err(format!(
            "frog_squeeze_ops length {} not divisible by tries={}",
            frog_ranges.len(),
            tries
        ));
    }
    let n_coins = if frog_ranges.is_empty() { 0 } else { frog_ranges.len() / tries };
    let mut out: Vec<FrogRejectionCoinWiring> = Vec::with_capacity(n_coins);
    for coin_idx in 0..n_coins {
        let mut digit_vars: Vec<usize> = Vec::with_capacity(tries * DIGITS_PER_TRY);
        for t in 0..tries {
            let (start, len) = frog_ranges[coin_idx * tries + t];
            if len != DIGITS_PER_TRY {
                return Err(format!(
                    "frog squeeze len mismatch (got {len}, expected {DIGITS_PER_TRY})"
                ));
            }
            for gv in &pose_wiring.squeeze_field_vars[start..start + len] {
                digit_vars.push(glue.copy_digit(*gv));
            }
        }
        let (coin_local, found_local) =
            sample_frog_coin_unrolled_rejection_8_digits::<F257>(&mut glue.gb, &digit_vars, tries);
        glue.gb.enforce_var_eq_const(found_local, F257::ONE);
        out.push(FrogRejectionCoinWiring {
            digit_vars,
            found_bit: found_local,
            coin_limbs: coin_local.to_vec(),
            tries,
        });
    }
    Ok(out)
}

fn compute_tcch(
    glue: &mut GlueCtx,
    ops: &[PoseidonTraceOp<F257>],
    pose_wiring: &PoseidonDr1csWiring,
    ring_dim: usize,
    params: &WeParams,
    wiring: &TinyCoinOpWiring,
    u32_ranges: &[(usize, usize)],
    frog_locals: &[FrogChallengeWiring],
) -> Result<(Vec<usize>, Vec<usize>), String> {
    // Preserves exact existing behavior; only moved out for readability/audit.
    let mut tcch0_local: Vec<usize> = Vec::new();
    let mut tcch1_local: Vec<usize> = Vec::new();

    if ring_dim > 0 && !wiring.short_squeeze_ops.is_empty() && !wiring.u32_squeeze_ops.is_empty() {
        let mut ring_elem_bytes: Option<usize> = None;
        for &(_st, ln) in &pose_wiring.absorb_ranges {
            if ln % ring_dim == 0 && ln > ring_dim {
                ring_elem_bytes = Some(match ring_elem_bytes {
                    None => ln,
                    Some(cur) => cur.min(ln),
                });
            }
        }
        if let Some(reb) = ring_elem_bytes {
            let coeff_bytes = reb / ring_dim;
            if coeff_bytes > 0 {
                let last_short_op = *wiring
                    .short_squeeze_ops
                    .iter()
                    .max()
                    .expect("non-empty short_squeeze_ops");
                let first_short_op = *wiring
                    .short_squeeze_ops
                    .iter()
                    .min()
                    .expect("non-empty short_squeeze_ops");

                let cm_u32_start = wiring
                    .u32_squeeze_ops
                    .iter()
                    .filter(|&&idx| idx < first_short_op)
                    .count();
                if cm_u32_start < wiring.u32_squeeze_ops.len() {
                    let first_cm_u32_op = wiring.u32_squeeze_ops[cm_u32_start];

                    let mut absorb_idx = 0usize;
                    let mut squeeze_field_op_idx = 0usize;
                    let mut after_short = false;
                    let mut comh_absorb_ranges: Vec<(usize, usize)> = Vec::new();
                    for op in ops {
                        match op {
                            PoseidonTraceOp::Absorb(_v) => {
                                let (ab_start, ab_len) = *pose_wiring
                                    .absorb_ranges
                                    .get(absorb_idx)
                                    .ok_or("tiny gate: pose_wiring.absorb_ranges oob")?;
                                absorb_idx += 1;
                                if after_short && squeeze_field_op_idx <= first_cm_u32_op {
                                    comh_absorb_ranges.push((ab_start, ab_len));
                                }
                            }
                            PoseidonTraceOp::SqueezeField(_v) => {
                                if squeeze_field_op_idx == last_short_op {
                                    after_short = true;
                                }
                                squeeze_field_op_idx += 1;
                            }
                            PoseidonTraceOp::SqueezeBytes { .. } => {}
                        }
                    }

                    let mut coh0_res: Vec<usize> = Vec::new();
                    for &(ab_start, ab_len) in &comh_absorb_ranges {
                        if ab_len < reb || (ab_len % reb) != 0 {
                            continue;
                        }
                        let n_blocks = ab_len / reb;
                        for blk in 0..n_blocks {
                            let blk_start = ab_start + blk * reb;
                            // The first coefficient of the ring element is encoded as `coeff_bytes` bytes.
                            // Enforce canonical Frog base-field encoding on that 8-byte chunk (no reduction).
                            if coeff_bytes == 8 {
                                let mut coeff0_bytes = [0usize; 8];
                                for i in 0..8 {
                                    let gv = pose_wiring.absorb_vars[blk_start + i];
                                    let lv = glue.copy_digit(gv);
                                    let _ = decompose_existing_byte_var_to_bits::<F257>(&mut glue.gb, lv);
                                    coeff0_bytes[i] = lv;
                                }
                                let _coeff0_limbs =
                                    frog_u64_canonical_from_byte_vars::<F257>(&mut glue.gb, &coeff0_bytes);
                            }
                            let mut acc = F257::ZERO;
                            let mut lc: Vec<(F257, usize)> = Vec::with_capacity(1 + coeff_bytes);
                            for i in 0..coeff_bytes {
                                let gv = pose_wiring.absorb_vars[blk_start + i];
                                let lv = glue.copy_digit(gv);
                                let _ = decompose_existing_byte_var_to_bits::<F257>(&mut glue.gb, lv);
                                let sign = if (i & 1) == 0 { F257::ONE } else { -F257::ONE };
                                acc += glue.gb.assignment[lv] * sign;
                                lc.push((-sign, lv));
                            }
                            let v = glue.gb.new_var(acc);
                            lc.insert(0, (F257::ONE, v));
                            glue.gb.enforce_lc_times_one_eq_const(lc);
                            coh0_res.push(v);
                        }
                    }

                    let n_comh_elems = coh0_res.len();
                    if n_comh_elems > 0 {
                        let kappa = params.kappa as usize;
                        if kappa == 0 || !kappa.is_power_of_two() {
                            return Err("tiny gate: params.kappa must be a power of two".to_string());
                        }
                        if (n_comh_elems % kappa) != 0 {
                            return Err("tiny gate: comh length not divisible by kappa".to_string());
                        }
                        let lg = usize::BITS as usize - 1 - kappa.leading_zeros() as usize;
                        let l_instances = n_comh_elems / kappa;

                        let c0_start = cm_u32_start;
                        let c1_start = cm_u32_start + lg;
                        if c1_start + lg <= u32_ranges.len() && c1_start + lg <= frog_locals.len() {
                            let mut c0_vars: Vec<usize> = Vec::with_capacity(lg);
                            let mut c1_vars: Vec<usize> = Vec::with_capacity(lg);
                            for i in 0..lg {
                                c0_vars.push(frog_locals[c0_start + i].res257);
                            }
                            for i in 0..lg {
                                c1_vars.push(frog_locals[c1_start + i].res257);
                            }

                                #[inline]
                                fn tensor_vars(gb: &mut Dr1csBuilder<F257>, c: &[usize]) -> Vec<usize> {
                                    let mut acc: Vec<usize> = vec![gb.new_var(F257::ONE)];
                                    gb.enforce_var_eq_const(acc[0], F257::ONE);
                                    for &ci in c {
                                        let one_minus = gb.new_var(F257::ONE - gb.assignment[ci]);
                                        gb.enforce_lc_times_one_eq_const(vec![
                                            (F257::ONE, one_minus),
                                            (F257::ONE, ci),
                                            (-F257::ONE, gb.one()),
                                        ]);
                                        let mut next = Vec::with_capacity(acc.len() * 2);
                                        for &t in &acc {
                                            let v0 = gb.new_var(gb.assignment[t] * gb.assignment[one_minus]);
                                            gb.enforce_mul(t, one_minus, v0);
                                            next.push(v0);
                                            let v1 = gb.new_var(gb.assignment[t] * gb.assignment[ci]);
                                            gb.enforce_mul(t, ci, v1);
                                            next.push(v1);
                                        }
                                        acc = next;
                                    }
                                    acc
                                }

                                let tensor_c0 = tensor_vars(&mut glue.gb, &c0_vars);
                                let tensor_c1 = tensor_vars(&mut glue.gb, &c1_vars);

                            tcch0_local.reserve(l_instances);
                            tcch1_local.reserve(l_instances);
                            for l in 0..l_instances {
                                let base = l * kappa;
                                let mut terms0: Vec<usize> = Vec::with_capacity(kappa);
                                let mut terms1: Vec<usize> = Vec::with_capacity(kappa);
                                for j in 0..kappa {
                                    let rj = coh0_res[base + j];
                                    let m0 = glue.gb.new_var(
                                        glue.gb.assignment[tensor_c0[j]] * glue.gb.assignment[rj],
                                    );
                                    glue.gb.enforce_mul(tensor_c0[j], rj, m0);
                                    let m1 = glue.gb.new_var(
                                        glue.gb.assignment[tensor_c1[j]] * glue.gb.assignment[rj],
                                    );
                                    glue.gb.enforce_mul(tensor_c1[j], rj, m1);
                                    terms0.push(m0);
                                    terms1.push(m1);
                                }
                                let sum0 = {
                                    let mut acc = F257::ZERO;
                                    for &t in &terms0 {
                                        acc += glue.gb.assignment[t];
                                    }
                                    let v = glue.gb.new_var(acc);
                                    let mut lc: Vec<(F257, usize)> =
                                        Vec::with_capacity(1 + terms0.len());
                                    lc.push((F257::ONE, v));
                                    for &t in &terms0 {
                                        lc.push((-F257::ONE, t));
                                    }
                                    glue.gb.enforce_lc_times_one_eq_const(lc);
                                    v
                                };
                                let sum1 = {
                                    let mut acc = F257::ZERO;
                                    for &t in &terms1 {
                                        acc += glue.gb.assignment[t];
                                    }
                                    let v = glue.gb.new_var(acc);
                                    let mut lc: Vec<(F257, usize)> =
                                        Vec::with_capacity(1 + terms1.len());
                                    lc.push((F257::ONE, v));
                                    for &t in &terms1 {
                                        lc.push((-F257::ONE, t));
                                    }
                                    glue.gb.enforce_lc_times_one_eq_const(lc);
                                    v
                                };
                                tcch0_local.push(sum0);
                                tcch1_local.push(sum1);
                            }
                        }
                    }
                }
            }
        }
    }

    Ok((tcch0_local, tcch1_local))
}

struct SurfaceLocal<const RAW: usize, const NORM: usize> {
    short_block_idx: usize,
    u32_idx: usize,
    products_raw: Vec<[usize; RAW]>,
    products_norm: Vec<[usize; NORM]>,
    sum_digits: Vec<usize>,
}

type UDigitsFn = for<'a> fn(&'a BoundedU32ChallengeWiring) -> &'a [usize];

#[allow(clippy::too_many_arguments)]
fn build_surfaces_generic<const RAW: usize, const NORM: usize, const SUM: usize>(
    glue: &mut GlueCtx,
    ring_dim: usize,
    pairs: &[(usize, usize)],
    short_locals: &[ShortChallengeWiring],
    u32_locals: &[BoundedU32ChallengeWiring],
    u_digits_of: UDigitsFn,
    scale_fn: fn(&mut Dr1csBuilder<F257>, &[[usize; 3]], &[usize]) -> Vec<[usize; RAW]>,
    rebalance_fn: fn(&mut Dr1csBuilder<F257>, &[usize; RAW]) -> [usize; NORM],
    sum_product_digits_fn: fn(&mut Dr1csBuilder<F257>, &[[usize; NORM]], usize) -> Vec<usize>,
    sum_products_coeffwise_fn: fn(&mut Dr1csBuilder<F257>, &[&[[usize; NORM]]], usize, usize) -> Vec<Vec<usize>>,
    u_digits_len: usize,
) -> Result<(Vec<SurfaceLocal<RAW, NORM>>, Vec<usize>, Vec<Vec<usize>>), String> {
    let mut surfaces: Vec<SurfaceLocal<RAW, NORM>> = Vec::with_capacity(pairs.len());
    let zero_f = glue.gb.new_var(F257::ZERO);
    glue.gb.enforce_var_eq_const(zero_f, F257::ZERO);
    let mut sum_prod_res_coeff_total: Vec<usize> = vec![zero_f; ring_dim];

    for &(si, ui) in pairs {
        let s = &short_locals[si];
        let u = &u32_locals[ui];

        let pow16: [F257; NORM] = {
            let mut p = [F257::ZERO; NORM];
            let mut cur = F257::ONE;
            let sixteen = F257::from(16u64);
            for i in 0..NORM {
                p[i] = cur;
                cur *= sixteen;
            }
            p
        };

        let u_digits = u_digits_of(u);
        if u_digits.len() != u_digits_len {
            return Err(format!(
                "u_digits length mismatch (got {}, expected {u_digits_len})",
                u_digits.len()
            ));
        }
        let u_res = {
            let mut acc = F257::ZERO;
            for i in 0..u_digits_len {
                acc += glue.gb.assignment[u_digits[i]] * pow16[i];
            }
            let v = glue.gb.new_var(acc);
            let mut lc: Vec<(F257, usize)> = Vec::with_capacity(1 + u_digits_len);
            lc.push((F257::ONE, v));
            for i in 0..u_digits_len {
                lc.push((-pow16[i], u_digits[i]));
            }
            glue.gb.enforce_lc_times_one_eq_const(lc);
            v
        };

        let products_raw = scale_fn(&mut glue.gb, &s.coeff_bal16_digits, u_digits);
        let products_norm = products_raw
            .iter()
            .map(|p| rebalance_fn(&mut glue.gb, p))
            .collect::<Vec<_>>();

        let mut sum_prod_res = glue.gb.new_var(F257::ZERO);
        glue.gb.enforce_var_eq_const(sum_prod_res, F257::ZERO);
        for j in 0..ring_dim {
            let c3 = &s.coeff_bal16_digits[j];
            let coeff_val = glue.gb.assignment[c3[0]]
                + glue.gb.assignment[c3[1]] * F257::from(16u64)
                + glue.gb.assignment[c3[2]] * F257::from(256u64);
            let coeff_res = glue.gb.new_var(coeff_val);
            glue.gb.enforce_lc_times_one_eq_const(vec![
                (F257::ONE, coeff_res),
                (-F257::ONE, c3[0]),
                (-F257::from(16u64), c3[1]),
                (-F257::from(256u64), c3[2]),
            ]);

            let p = &products_norm[j];
            let mut prod_val = F257::ZERO;
            for i in 0..NORM {
                prod_val += glue.gb.assignment[p[i]] * pow16[i];
            }
            let prod_res = glue.gb.new_var(prod_val);
            let mut lc: Vec<(F257, usize)> = Vec::with_capacity(1 + NORM);
            lc.push((F257::ONE, prod_res));
            for i in 0..NORM {
                lc.push((-pow16[i], p[i]));
            }
            glue.gb.enforce_lc_times_one_eq_const(lc);
            glue.gb.enforce_mul(coeff_res, u_res, prod_res);

            let next_sum = glue.gb.new_var(glue.gb.assignment[sum_prod_res] + glue.gb.assignment[prod_res]);
            glue.gb.enforce_lc_times_one_eq_const(vec![
                (F257::ONE, next_sum),
                (-F257::ONE, sum_prod_res),
                (-F257::ONE, prod_res),
            ]);
            sum_prod_res = next_sum;

            let prev = sum_prod_res_coeff_total[j];
            let next = glue.gb.new_var(glue.gb.assignment[prev] + glue.gb.assignment[prod_res]);
            glue.gb.enforce_lc_times_one_eq_const(vec![
                (F257::ONE, next),
                (-F257::ONE, prev),
                (-F257::ONE, prod_res),
            ]);
            sum_prod_res_coeff_total[j] = next;
        }

        let sum_digits = sum_product_digits_fn(&mut glue.gb, &products_norm, SUM);
        {
            debug_assert_eq!(sum_digits.len(), SUM);
            let mut pow16_sum = vec![F257::ZERO; SUM];
            let mut cur = F257::ONE;
            let sixteen = F257::from(16u64);
            for i in 0..SUM {
                pow16_sum[i] = cur;
                cur *= sixteen;
            }
            let mut acc = F257::ZERO;
            for i in 0..SUM {
                acc += glue.gb.assignment[sum_digits[i]] * pow16_sum[i];
            }
            let sum_digits_res = glue.gb.new_var(acc);
            let mut lc: Vec<(F257, usize)> = Vec::with_capacity(1 + SUM);
            lc.push((F257::ONE, sum_digits_res));
            for i in 0..SUM {
                lc.push((-pow16_sum[i], sum_digits[i]));
            }
            glue.gb.enforce_lc_times_one_eq_const(lc);
            glue.gb.enforce_lc_times_one_eq_const(vec![
                (F257::ONE, sum_digits_res),
                (-F257::ONE, sum_prod_res),
            ]);
        }

        surfaces.push(SurfaceLocal {
            short_block_idx: si,
            u32_idx: ui,
            products_raw,
            products_norm,
            sum_digits,
        });
    }

    let all_sum_digits = {
        let refs: Vec<&[usize]> = surfaces.iter().map(|s| s.sum_digits.as_slice()).collect();
        sum_bal16_vectors_fixed_len(&mut glue.gb, &refs, SUM)
    };
    let all_sum_coeffwise = {
        let refs: Vec<&[[usize; NORM]]> = surfaces.iter().map(|s| s.products_norm.as_slice()).collect();
        sum_products_coeffwise_fn(&mut glue.gb, &refs, ring_dim, SUM)
    };

    {
        debug_assert_eq!(all_sum_coeffwise.len(), ring_dim);
        let mut pow16_sum = vec![F257::ZERO; SUM];
        let mut cur = F257::ONE;
        let sixteen = F257::from(16u64);
        for i in 0..SUM {
            pow16_sum[i] = cur;
            cur *= sixteen;
        }
        for j in 0..ring_dim {
            let digs = &all_sum_coeffwise[j];
            debug_assert_eq!(digs.len(), SUM);
            let mut acc = F257::ZERO;
            for i in 0..SUM {
                acc += glue.gb.assignment[digs[i]] * pow16_sum[i];
            }
            let res = glue.gb.new_var(acc);
            let mut lc: Vec<(F257, usize)> = Vec::with_capacity(1 + SUM);
            lc.push((F257::ONE, res));
            for i in 0..SUM {
                lc.push((-pow16_sum[i], digs[i]));
            }
            glue.gb.enforce_lc_times_one_eq_const(lc);
            glue.gb.enforce_lc_times_one_eq_const(vec![
                (F257::ONE, res),
                (-F257::ONE, sum_prod_res_coeff_total[j]),
            ]);
        }
    }

    Ok((surfaces, all_sum_digits, all_sum_coeffwise))
}

fn build_mul_surfaces(
    glue: &mut GlueCtx,
    ring_dim: usize,
    pairs: &[(usize, usize)],
    short_locals: &[ShortChallengeWiring],
    u32_locals: &[BoundedU32ChallengeWiring],
) -> Result<(Vec<CmDigitMulSurfaceWiring>, Vec<usize>, Vec<Vec<usize>>), String> {
    fn u_digits(u: &BoundedU32ChallengeWiring) -> &[usize] {
        &u.bal16_digits
    }
    fn sum_digits(gb: &mut Dr1csBuilder<F257>, p: &[[usize; 13]], target_len: usize) -> Vec<usize> {
        sum_product_digits_bal16(gb, p, target_len)
    }

    let (locals, all_sum_digits, all_sum_coeffwise) = build_surfaces_generic::<12, 13, 16>(
        glue,
        ring_dim,
        pairs,
        short_locals,
        u32_locals,
        u_digits,
        scale_short_coeffs_by_digits9,
        rebalance_prod12_to_prod13,
        sum_digits,
        sum_products13_coeffwise_fixed_len,
        9,
    )?;
    let surfaces = locals
        .into_iter()
        .map(|s| CmDigitMulSurfaceWiring {
            short_block_idx: s.short_block_idx,
            u32_idx: s.u32_idx,
            products: s.products_raw,
            products13: s.products_norm,
            sum_digits: s.sum_digits,
            sum_all_pairs_digits: Arc::new(Vec::new()),
            sum_all_pairs_coeffwise: Arc::new(Vec::new()),
        })
        .collect();
    Ok((surfaces, all_sum_digits, all_sum_coeffwise))
}

fn build_sq_surfaces(
    glue: &mut GlueCtx,
    ring_dim: usize,
    pairs: &[(usize, usize)],
    short_locals: &[ShortChallengeWiring],
    u32_locals: &[BoundedU32ChallengeWiring],
) -> Result<(Vec<CmDigitMulSqSurfaceWiring>, Vec<usize>, Vec<Vec<usize>>), String> {
    fn u_digits(u: &BoundedU32ChallengeWiring) -> &[usize] {
        &u.bal16_sq_digits
    }
    fn sum_digits(gb: &mut Dr1csBuilder<F257>, p: &[[usize; 22]], target_len: usize) -> Vec<usize> {
        sum_product_digits_bal16_22(gb, p, target_len)
    }

    let (locals, all_sum_digits, all_sum_coeffwise) = build_surfaces_generic::<21, 22, 24>(
        glue,
        ring_dim,
        pairs,
        short_locals,
        u32_locals,
        u_digits,
        scale_short_coeffs_by_digits18,
        rebalance_prod21_to_prod22,
        sum_digits,
        sum_products22_coeffwise_fixed_len,
        18,
    )?;
    let surfaces = locals
        .into_iter()
        .map(|s| CmDigitMulSqSurfaceWiring {
            short_block_idx: s.short_block_idx,
            u32_idx: s.u32_idx,
            products21: s.products_raw,
            products22: s.products_norm,
            sum_digits: s.sum_digits,
            sum_all_pairs_digits: Arc::new(Vec::new()),
            sum_all_pairs_coeffwise: Arc::new(Vec::new()),
        })
        .collect();
    Ok((surfaces, all_sum_digits, all_sum_coeffwise))
}

fn enforce_fiat_shamir_reabsorb_semantics(
    inst: &mut SparseDr1csInstance<F257>,
    ops: &[PoseidonTraceOp<F257>],
    pose_wiring: &PoseidonDr1csWiring,
) -> Result<(), String> {
    let mut absorb_idx = 0usize;
    let mut squeeze_idx = 0usize;
    for (i, op) in ops.iter().enumerate() {
        match op {
            PoseidonTraceOp::Absorb(_) => {
                absorb_idx += 1;
            }
            PoseidonTraceOp::SqueezeField(out) => {
                let (sq_start, sq_len) = *pose_wiring
                    .squeeze_field_ranges
                    .get(squeeze_idx)
                    .ok_or("poseidon wiring squeeze_field_ranges oob (tiny)")?;
                squeeze_idx += 1;
                if sq_len != out.len() {
                    return Err("poseidon squeeze length mismatch (tiny)".to_string());
                }
                if out.len() != DIGITS_PER_TRY {
                    continue;
                }
                if let Some(PoseidonTraceOp::Absorb(next)) = ops.get(i + 1) {
                    if next.len() != DIGITS_PER_TRY {
                        return Err("poseidon reabsorb length mismatch (tiny)".to_string());
                    }
                    let (ab_start, ab_len) = *pose_wiring
                        .absorb_ranges
                        .get(absorb_idx)
                        .ok_or("poseidon wiring absorb_ranges oob after squeeze (tiny)")?;
                    if ab_len != sq_len {
                        return Err("poseidon reabsorb length mismatch (tiny)".to_string());
                    }
                    for j in 0..sq_len {
                        let v_sq = pose_wiring.squeeze_field_vars[sq_start + j];
                        let v_ab = pose_wiring.absorb_vars[ab_start + j];
                        inst.constraints.push(Constraint {
                            a: vec![(F257::ONE, v_ab), (-F257::ONE, v_sq)],
                            b: vec![(F257::ONE, 0)],
                            c: vec![(F257::ZERO, 0)],
                        });
                    }
                }
            }
            PoseidonTraceOp::SqueezeBytes { .. } => {}
        }
    }
    Ok(())
}

/// Enforce that every **non-reabsorb** Absorb chunk of length multiple of 8 consists of
/// canonical Frog base-field encodings (u64 < p_frog), interpreted as 8-byte little-endian scalars.
///
/// This is a byte/limb-only binding step: it does not perform any modular arithmetic beyond
/// the single-subtract canonicality check.
fn enforce_nonreabsorb_absorbs_are_canonical_frog(
    glue: &mut GlueCtx,
    ops: &[PoseidonTraceOp<F257>],
    pose_wiring: &PoseidonDr1csWiring,
) -> Result<(), String> {
    let mut absorb_idx = 0usize;
    let mut expect_reabsorb = false;
    for op in ops {
        match op {
            PoseidonTraceOp::SqueezeField(v) => {
                // Only get_challenge() does a fiat–shamir reabsorb, and it always squeezes 8 digits.
                expect_reabsorb = v.len() == DIGITS_PER_TRY;
            }
            PoseidonTraceOp::Absorb(_v) => {
                let (ab_start, ab_len) = *pose_wiring
                    .absorb_ranges
                    .get(absorb_idx)
                    .ok_or("pose_wiring.absorb_ranges oob (canonical frog)")?;
                absorb_idx += 1;
                let is_reabsorb = expect_reabsorb;
                expect_reabsorb = false;
                if is_reabsorb {
                    // Skip: these are F257 digits (can include 256), not base-field bytes.
                    continue;
                }
                if (ab_len % 8) != 0 {
                    continue;
                }
                let n_elems = ab_len / 8;
                for e in 0..n_elems {
                    let mut bytes = [0usize; 8];
                    for j in 0..8 {
                        let gv = pose_wiring.absorb_vars[ab_start + e * 8 + j];
                        let lv = glue.copy_digit(gv);
                        let _ = decompose_existing_byte_var_to_bits::<F257>(&mut glue.gb, lv);
                        bytes[j] = lv;
                    }
                    let _limbs = frog_u64_canonical_from_byte_vars::<F257>(&mut glue.gb, &bytes);
                }
            }
            PoseidonTraceOp::SqueezeBytes { .. } => {}
        }
    }
    Ok(())
}

fn sp1_centered_bound_u64(params: &WeParams, ring_dim: usize) -> Result<u64, String> {
    // Conservative digit bound from rgchk: |digit| <= D where D = d/2 - 1.
    if ring_dim < 4 || (ring_dim % 2) != 0 {
        return Err("tiny gate: ring_dim must be even and >= 4".to_string());
    }
    let d: u128 = ring_dim as u128;
    let d_half: u128 = d / 2;
    if d_half < 2 {
        return Err("tiny gate: ring_dim too small".to_string());
    }
    let D: u128 = d_half - 1;
    let b: u128 = params.decomp_b as u128;
    let k: u32 = params.k as u32;
    if b < 2 {
        return Err("tiny gate: decomp_b must be >= 2".to_string());
    }
    if k == 0 {
        return Err("tiny gate: k must be >= 1".to_string());
    }
    // bound = D * (b^k - 1)/(b - 1)
    let mut pow: u128 = 1;
    for _ in 0..k {
        pow = pow.saturating_mul(b);
    }
    let num = pow.saturating_sub(1);
    let denom = b - 1;
    let geom = num / denom;
    let bound_u128 = D.saturating_mul(geom);
    if bound_u128 > (u64::MAX as u128) {
        return Err("tiny gate: centered bound overflows u64".to_string());
    }
    Ok(bound_u128 as u64)
}

/// Enforce that every non-reabsorb absorbed 8-byte chunk is not only canonical (<p),
/// but also lies in the conservative centered range implied by the rgchk digit bound.
fn enforce_nonreabsorb_absorbs_are_centered_bounded_frog(
    glue: &mut GlueCtx,
    ops: &[PoseidonTraceOp<F257>],
    pose_wiring: &PoseidonDr1csWiring,
    params: &WeParams,
    ring_dim: usize,
) -> Result<(), String> {
    let bound = sp1_centered_bound_u64(params, ring_dim)?;
    let mut absorb_idx = 0usize;
    let mut expect_reabsorb = false;
    for op in ops {
        match op {
            PoseidonTraceOp::SqueezeField(v) => {
                expect_reabsorb = v.len() == DIGITS_PER_TRY;
            }
            PoseidonTraceOp::Absorb(_v) => {
                let (ab_start, ab_len) = *pose_wiring
                    .absorb_ranges
                    .get(absorb_idx)
                    .ok_or("pose_wiring.absorb_ranges oob (centered frog)")?;
                absorb_idx += 1;
                let is_reabsorb = expect_reabsorb;
                expect_reabsorb = false;
                if is_reabsorb || (ab_len % 8) != 0 {
                    continue;
                }
                let n_elems = ab_len / 8;
                for e in 0..n_elems {
                    let mut bytes = [0usize; 8];
                    for j in 0..8 {
                        let gv = pose_wiring.absorb_vars[ab_start + e * 8 + j];
                        let lv = glue.copy_digit(gv);
                        let _ = decompose_existing_byte_var_to_bits::<F257>(&mut glue.gb, lv);
                        bytes[j] = lv;
                    }
                    let _limbs =
                        frog_u64_centered_le_bound_from_byte_vars::<F257>(&mut glue.gb, &bytes, bound);
                }
            }
            PoseidonTraceOp::SqueezeBytes { .. } => {}
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn finalize(
    pose_inst: SparseDr1csInstance<F257>,
    pose_wiring: PoseidonDr1csWiring,
    ops: &[PoseidonTraceOp<F257>],
    glue: GlueCtx,
    short_locals: Vec<ShortChallengeWiring>,
    u32_locals: Vec<BoundedU32ChallengeWiring>,
    frog_locals: Vec<FrogChallengeWiring>,
    frog_rejection_locals: Vec<FrogRejectionCoinWiring>,
    tcch0_local: Vec<usize>,
    tcch1_local: Vec<usize>,
    surfaces_mul_local: Vec<CmDigitMulSurfaceWiring>,
    surfaces_sq_local: Vec<CmDigitMulSqSurfaceWiring>,
    all_sum_digits: Arc<Vec<usize>>,
    all_sum_coeffwise: Arc<Vec<Vec<usize>>>,
    all_sq_sum_digits: Arc<Vec<usize>>,
    all_sq_sum_coeffwise: Arc<Vec<Vec<usize>>>,
) -> Result<
    (
        SparseDr1csInstance<F257>,
        Vec<F257>,
        Vec<ShortChallengeWiring>,
        Vec<BoundedU32ChallengeWiring>,
        Vec<FrogChallengeWiring>,
        Vec<FrogRejectionCoinWiring>,
        Vec<usize>,
        Vec<usize>,
        Vec<CmDigitMulSurfaceWiring>,
        Vec<CmDigitMulSqSurfaceWiring>,
        PoseidonDr1csWiring,
    ),
    String,
> {
    let GlueCtx { gb, pose_asg, local_map } = glue;
    let (glue_inst, glue_asg) = gb.into_instance();
    let glue_nvars = glue_inst.nvars;

    let (mut inst, asg) = merge_sparse_dr1cs_share_one::<F257>(vec![
        (pose_inst, pose_asg),
        (glue_inst, glue_asg),
    ])
    .map_err(|e| format!("merge poseidon+cm+mul_glue failed: {e}"))?;

    enforce_fiat_shamir_reabsorb_semantics(&mut inst, ops, &pose_wiring)?;

    let pose_nvars = inst.nvars - (glue_nvars - 1);
    let glue_offset = pose_nvars - 1;
    for (&gv, &lv) in local_map.iter() {
        let gg = if lv == 0 { 0 } else { lv + glue_offset };
        enforce_var_eq::<F257>(&mut inst, gv, gg);
    }
    let to_glue_global = |glue_local: usize| -> usize {
        if glue_local == 0 { 0 } else { glue_local + glue_offset }
    };

    let shorts_out = short_locals
        .into_iter()
        .map(|w| ShortChallengeWiring {
            digit_vars: w.digit_vars.into_iter().map(to_glue_global).collect(),
            byte_vars: w.byte_vars.into_iter().map(to_glue_global).collect(),
            coeff_vars: w.coeff_vars.into_iter().map(to_glue_global).collect(),
            coeff_bal16_digits: w
                .coeff_bal16_digits
                .into_iter()
                .map(|a| a.map(to_glue_global))
                .collect(),
        })
        .collect::<Vec<_>>();
    let u32s_out = u32_locals
        .into_iter()
        .map(|w| BoundedU32ChallengeWiring {
            digit_vars: w.digit_vars.into_iter().map(to_glue_global).collect(),
            byte_vars: w.byte_vars.map(to_glue_global),
            limbs: w.limbs.map(to_glue_global),
            bal16_digits: w.bal16_digits.into_iter().map(to_glue_global).collect(),
            bal16_sq_digits: w.bal16_sq_digits.into_iter().map(to_glue_global).collect(),
        })
        .collect::<Vec<_>>();
    let frogs_out = frog_locals
        .into_iter()
        .map(|w| FrogChallengeWiring {
            digit_vars: w.digit_vars.into_iter().map(to_glue_global).collect(),
            byte_vars: w.byte_vars.map(to_glue_global),
            q_bit: to_glue_global(w.q_bit),
            limbs: w.limbs.map(to_glue_global),
            res257: to_glue_global(w.res257),
        })
        .collect::<Vec<_>>();
    let frog_rejection_out = frog_rejection_locals
        .into_iter()
        .map(|w| FrogRejectionCoinWiring {
            digit_vars: w.digit_vars.into_iter().map(to_glue_global).collect(),
            found_bit: to_glue_global(w.found_bit),
            coin_limbs: w.coin_limbs.into_iter().map(to_glue_global).collect(),
            tries: w.tries,
        })
        .collect::<Vec<_>>();

    let all_sum_digits_global: Arc<Vec<usize>> =
        Arc::new(all_sum_digits.iter().copied().map(to_glue_global).collect());
    let all_sum_coeffwise_global: Arc<Vec<Vec<usize>>> = Arc::new(
        all_sum_coeffwise
            .iter()
            .map(|v| v.iter().copied().map(to_glue_global).collect())
            .collect(),
    );
    let all_sq_sum_digits_global: Arc<Vec<usize>> =
        Arc::new(all_sq_sum_digits.iter().copied().map(to_glue_global).collect());
    let all_sq_sum_coeffwise_global: Arc<Vec<Vec<usize>>> = Arc::new(
        all_sq_sum_coeffwise
            .iter()
            .map(|v| v.iter().copied().map(to_glue_global).collect())
            .collect(),
    );

    let surfaces_out = surfaces_mul_local
        .into_iter()
        .map(|s| CmDigitMulSurfaceWiring {
            short_block_idx: s.short_block_idx,
            u32_idx: s.u32_idx,
            products: s.products.into_iter().map(|p| p.map(to_glue_global)).collect(),
            products13: s
                .products13
                .into_iter()
                .map(|p| p.map(to_glue_global))
                .collect(),
            sum_digits: s.sum_digits.into_iter().map(to_glue_global).collect(),
            sum_all_pairs_digits: all_sum_digits_global.clone(),
            sum_all_pairs_coeffwise: all_sum_coeffwise_global.clone(),
        })
        .collect::<Vec<_>>();

    let surfaces_sq_out = surfaces_sq_local
        .into_iter()
        .map(|s| CmDigitMulSqSurfaceWiring {
            short_block_idx: s.short_block_idx,
            u32_idx: s.u32_idx,
            products21: s.products21.into_iter().map(|arr| arr.map(to_glue_global)).collect(),
            products22: s.products22.into_iter().map(|arr| arr.map(to_glue_global)).collect(),
            sum_digits: s.sum_digits.into_iter().map(to_glue_global).collect(),
            sum_all_pairs_digits: all_sq_sum_digits_global.clone(),
            sum_all_pairs_coeffwise: all_sq_sum_coeffwise_global.clone(),
        })
        .collect::<Vec<_>>();

    Ok((
        inst,
        asg,
        shorts_out,
        u32s_out,
        frogs_out,
        frog_rejection_out,
        tcch0_local.into_iter().map(to_glue_global).collect(),
        tcch1_local.into_iter().map(to_glue_global).collect(),
        surfaces_out,
        surfaces_sq_out,
        pose_wiring,
    ))
}

pub(super) fn build(
    cfg: Option<&PoseidonConfig<F257>>,
    ops: &[PoseidonTraceOp<F257>],
    ring_dim: usize,
    params: &WeParams,
    wiring: &TinyCoinOpWiring,
    pairs: &[(usize, usize)],
) -> Result<
    (
        SparseDr1csInstance<F257>,
        Vec<F257>,
        Vec<ShortChallengeWiring>,
        Vec<BoundedU32ChallengeWiring>,
        Vec<FrogChallengeWiring>,
        Vec<FrogRejectionCoinWiring>,
        Vec<usize>,
        Vec<usize>,
        Vec<CmDigitMulSurfaceWiring>,
        Vec<CmDigitMulSqSurfaceWiring>,
        PoseidonDr1csWiring,
    ),
    String,
> {
    let (pose_inst, pose_asg, pose_wiring, _byte_wiring) = poseidon_f257_arithmetize(cfg, ops)?;

    let short_ranges = squeeze_field_ranges_by_op_index(&pose_wiring.squeeze_field_ranges, &wiring.short_squeeze_ops)?;
    let u32_ranges = squeeze_field_ranges_by_op_index(&pose_wiring.squeeze_field_ranges, &wiring.u32_squeeze_ops)?;
    let frog_ranges = squeeze_field_ranges_by_op_index(&pose_wiring.squeeze_field_ranges, &wiring.frog_squeeze_ops)?;
    validate_pairs(pairs, short_ranges.len(), u32_ranges.len())?;

    let mut glue = GlueCtx::new(pose_asg);

    // Bind all proof/statement payload absorbs that encode base-field elements as canonical 8-byte scalars.
    // (Skip fiat–shamir reabsorbs, which are F257 digits and may contain 256.)
    enforce_nonreabsorb_absorbs_are_canonical_frog(&mut glue, ops, &pose_wiring)?;

    // Deterministic schedule sanity (aligns with `CmProof::verify_with_mlen`):
    // After the first short squeeze, CM consumes:
    // - `log_kappa` challenges for c0
    // - `log_kappa` challenges for c1
    // - rc0, rc1 (2)
    // - sumcheck r0/r1 (2*nvars_cm)
    // Total = 2*log_kappa + 2 + 2*nvars_cm.
    if wiring.short_squeeze_ops.is_empty() {
        return Err("tiny gate: expected CM short squeezes (short_squeeze_ops empty)".to_string());
    }
    let first_short_op = *wiring
        .short_squeeze_ops
        .iter()
        .min()
        .expect("non-empty short_squeeze_ops");
    let cm_u32_start = wiring.u32_squeeze_ops.iter().filter(|&&idx| idx < first_short_op).count();
    let cm_u32_have = wiring.u32_squeeze_ops.len().saturating_sub(cm_u32_start);
    let kappa = params.kappa as usize;
    if kappa == 0 || !kappa.is_power_of_two() {
        return Err("tiny gate: params.kappa must be a power of two".to_string());
    }
    let log_kappa = usize::BITS as usize - 1 - kappa.leading_zeros() as usize;
    let nvars_cm = params.nvars_cm as usize;
    let cm_u32_need = 2 * log_kappa + 2 + 2 * nvars_cm;
    if cm_u32_have < cm_u32_need {
        return Err(format!(
            "tiny gate: not enough CM u32 challenges after absorb_comh: have={cm_u32_have} need={cm_u32_need}"
        ));
    }


    let short_locals = build_short_blocks(&mut glue, &pose_wiring, ring_dim, &short_ranges)?;
    let (u32_locals, frog_locals) =
        build_u32_and_frog_blocks(&mut glue, &pose_wiring, &wiring.u32_squeeze_ops)?;
    let frog_rejection_locals = build_frog_rejection_coins(&mut glue, &pose_wiring, &frog_ranges)?;
    let (tcch0_local, tcch1_local) =
        compute_tcch(&mut glue, ops, &pose_wiring, ring_dim, params, wiring, &u32_ranges, &frog_locals)?;

    let (mut surfaces_mul_local, all_sum_digits, all_sum_coeffwise) =
        build_mul_surfaces(&mut glue, ring_dim, pairs, &short_locals, &u32_locals)?;
    let (mut surfaces_sq_local, all_sq_sum_digits, all_sq_sum_coeffwise) =
        build_sq_surfaces(&mut glue, ring_dim, pairs, &short_locals, &u32_locals)?;

    let all_sum_digits = Arc::new(all_sum_digits);
    let all_sum_coeffwise = Arc::new(all_sum_coeffwise);
    for s in &mut surfaces_mul_local {
        s.sum_all_pairs_digits = all_sum_digits.clone();
        s.sum_all_pairs_coeffwise = all_sum_coeffwise.clone();
    }
    let all_sq_sum_digits = Arc::new(all_sq_sum_digits);
    let all_sq_sum_coeffwise = Arc::new(all_sq_sum_coeffwise);
    for s in &mut surfaces_sq_local {
        s.sum_all_pairs_digits = all_sq_sum_digits.clone();
        s.sum_all_pairs_coeffwise = all_sq_sum_coeffwise.clone();
    }

    finalize(
        pose_inst,
        pose_wiring,
        ops,
        glue,
        short_locals,
        u32_locals,
        frog_locals,
        frog_rejection_locals,
        tcch0_local,
        tcch1_local,
        surfaces_mul_local,
        surfaces_sq_local,
        all_sum_digits,
        all_sum_coeffwise,
        all_sq_sum_digits,
        all_sq_sum_coeffwise,
    )
}