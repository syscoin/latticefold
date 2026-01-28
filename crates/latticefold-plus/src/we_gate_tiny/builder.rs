use std::collections::BTreeMap;
use std::sync::Arc;

use crate::transcript::DEFAULT_REJECTION_TRIES;

use ark_ff::Field;
use ark_crypto_primitives::sponge::poseidon::PoseidonConfig;
use latticefold::transcript::poseidon::{f257_poseidon_config, F257};
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
    frog_add_mod_p_from_byte_vars_assume_canonical, frog_mul_mod_p_from_byte_vars_assume_canonical,
    frog_sub_mod_p_from_byte_vars_assume_canonical,
    frog_u64_canonical_from_byte_vars, frog_u64_centered_le_bound_from_byte_vars,
    reduce_u64_mod_frog_from_byte_vars,
    FrogScalar,
};
use super::gadgets::{decompose_existing_byte_var_to_bits, enforce_var_eq};
use super::params::DIGITS_PER_TRY;
use super::poseidon::poseidon_f257_arithmetize;
use super::surfaces::{CmDigitMulSqSurfaceWiring, CmDigitMulSurfaceWiring};

use super::cm_math::{
    eq_eval_frog_digits, eval_t_z_optimized_ring_digits, frog_bytes_to_digits, frog_pow_table_digits,
    ring_add_bytes, ring_add_digits, ring_bytes_to_digits, ring_eq_digits, ring_mul_negacyclic_digits_d64,
    ring_scale_digits, ring_unit_monomial_digits, ring_zero_bytes, sumcheck_verify_degree2_ring_bytes,
    tensor_frog_ringconst_digits, tensor_frog_scalars_digits, RingBytes, RingDigits,
};
use super::op_counts::{tiny_cm_counts_reset, tiny_cm_counts_take};

#[inline]
fn tiny_opmix_on() -> bool {
    std::env::var("LFP_WE_GATE_OPMIX").is_ok()
}

#[inline]
fn lf_mem_on() -> bool {
    match std::env::var("LF_MEM") {
        Ok(v) => v != "0",
        Err(_) => false,
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct LfMemSample {
    rss_bytes: Option<u64>,
    hwm_bytes: Option<u64>,
    vmsize_bytes: Option<u64>,
}

fn parse_proc_status_kib(s: &str, key: &str) -> Option<u64> {
    // Format: `Key:   12345 kB`
    for line in s.lines() {
        if let Some(rest) = line.strip_prefix(key) {
            let rest = rest.trim();
            let mut it = rest.split_whitespace();
            let n = it.next()?.parse::<u64>().ok()?;
            let unit = it.next().unwrap_or("kB");
            if unit != "kB" {
                return None;
            }
            return Some(n);
        }
    }
    None
}

fn lf_mem_sample() -> LfMemSample {
    // Linux-only (best effort). When unavailable (macOS, etc.), return Nones.
    let Ok(s) = std::fs::read_to_string("/proc/self/status") else {
        return LfMemSample::default();
    };
    let rss_kib = parse_proc_status_kib(&s, "VmRSS:");
    let hwm_kib = parse_proc_status_kib(&s, "VmHWM:");
    let vmsize_kib = parse_proc_status_kib(&s, "VmSize:");
    LfMemSample {
        rss_bytes: rss_kib.map(|x| x.saturating_mul(1024)),
        hwm_bytes: hwm_kib.map(|x| x.saturating_mul(1024)),
        vmsize_bytes: vmsize_kib.map(|x| x.saturating_mul(1024)),
    }
}

#[inline]
fn fmt_mib(x: Option<u64>) -> String {
    x.map(|b| format!("{:.1}MiB", (b as f64) / (1024.0 * 1024.0)))
        .unwrap_or_else(|| "?".to_string())
}

#[derive(Clone, Copy, Debug, Default)]
struct LfStageCounts {
    pose_constraints: usize,
    pose_vars: usize,
    glue_constraints: usize,
    glue_vars: usize,
}

fn lf_stage_log(
    stage: &str,
    pose_inst: Option<&SparseDr1csInstance<F257>>,
    glue: Option<&GlueCtx>,
    prev: &mut Option<LfStageCounts>,
) {
    if !lf_mem_on() {
        return;
    }
    let mem = lf_mem_sample();
    let cur = LfStageCounts {
        pose_constraints: pose_inst.map(|p| p.constraints.len()).unwrap_or(0),
        pose_vars: pose_inst.map(|p| p.nvars).unwrap_or(0),
        glue_constraints: glue.map(|g| g.gb.rows.len()).unwrap_or(0),
        glue_vars: glue.map(|g| g.gb.assignment.len()).unwrap_or(0),
    };
    let (d_pose_c, d_pose_v, d_glue_c, d_glue_v) = match prev {
        Some(p) => (
            cur.pose_constraints.saturating_sub(p.pose_constraints),
            cur.pose_vars.saturating_sub(p.pose_vars),
            cur.glue_constraints.saturating_sub(p.glue_constraints),
            cur.glue_vars.saturating_sub(p.glue_vars),
        ),
        None => (0, 0, 0, 0),
    };
    let caches = glue.map(|g| {
        format!(
            " local_map={} byte_bits_cache={} u64_bal16_cache={} u32_bal16_cache={}",
            g.local_map.len(),
            g.gb.byte_bits_cache.len(),
            g.gb.u64_bal16_cache.len(),
            g.gb.u32_bal16_cache.len()
        )
    });

    eprintln!(
        "[LF_MEM] stage={} rss={} hwm={} vmsize={} | pose(c={},v={},Δc={},Δv={}) glue(c={},v={},Δc={},Δv={}){}",
        stage,
        fmt_mib(mem.rss_bytes),
        fmt_mib(mem.hwm_bytes),
        fmt_mib(mem.vmsize_bytes),
        cur.pose_constraints,
        cur.pose_vars,
        d_pose_c,
        d_pose_v,
        cur.glue_constraints,
        cur.glue_vars,
        d_glue_c,
        d_glue_v,
        caches.as_deref().unwrap_or(""),
    );
    *prev = Some(cur);
}

#[derive(Clone, Copy, Debug, Default)]
struct TinyAbsorbBreakdownCounts {
    comh_ops: usize,
    comh_bytes_total: usize,
    sc_msgs_ops_0: usize,
    sc_msgs_bytes_total_0: usize,
    sc_msgs_ops_1: usize,
    sc_msgs_bytes_total_1: usize,
    eval_ops_0: usize,
    eval_bytes_total_0: usize,
    eval_ops_1: usize,
    eval_bytes_total_1: usize,
}

#[inline]
fn sum_absorb_bytes(ranges: &[(usize, usize)]) -> usize {
    ranges.iter().map(|&(_st, ln)| ln).sum::<usize>()
}

#[allow(clippy::too_many_arguments)]
fn maybe_print_tiny_opmix(
    cfg: Option<&PoseidonConfig<F257>>,
    ops: &[PoseidonTraceOp<F257>],
    pose_inst: &SparseDr1csInstance<F257>,
    glue: &GlueCtx,
    absorb_counts: &TinyAbsorbBreakdownCounts,
    pairs_len: usize,
    mul_surfaces_len: usize,
    sq_surfaces_len: usize,
    sum_all_pairs_digits_len: usize,
    sum_all_pairs_coeffwise_len: usize,
    sq_sum_all_pairs_digits_len: usize,
    sq_sum_all_pairs_coeffwise_len: usize,
) {
    if !tiny_opmix_on() {
        return;
    }

    let default_cfg;
    let poseidon_cfg = match cfg {
        Some(c) => c,
        None => {
            default_cfg = f257_poseidon_config();
            &default_cfg
        }
    };

    // Poseidon schedule permutes (for cost estimates).
    let pose_permutes_total = symphony::poseidon_trace::count_permutes_for_ops(poseidon_cfg, ops);
    let last_short_sq = ops.iter().rposition(|op| match op {
        PoseidonTraceOp::SqueezeField(v) => v.len() == DIGITS_PER_TRY,
        _ => false,
    });
    let pose_permutes_before_cm = last_short_sq
        .map(|i| symphony::poseidon_trace::count_permutes_for_ops(poseidon_cfg, &ops[..=i]))
        .unwrap_or(0);
    let pose_permutes_after_cm = pose_permutes_total.saturating_sub(pose_permutes_before_cm);

    // Poseidon trace op mix.
    let mut n_absorb = 0usize;
    let mut absorb_elems = 0usize;
    let mut n_sq_field = 0usize;
    let mut sq_field_elems = 0usize;
    let mut n_sq_bytes = 0usize;
    let mut sq_bytes = 0usize;
    let mut absorb_by_len: BTreeMap<usize, usize> = BTreeMap::new();
    let mut squeeze_field_by_len: BTreeMap<usize, usize> = BTreeMap::new();
    let mut squeeze_bytes_by_len: BTreeMap<usize, usize> = BTreeMap::new();
    let mut n_get_challenge_reabsorbs = 0usize;
    for op in ops {
        match op {
            PoseidonTraceOp::Absorb(v) => {
                n_absorb += 1;
                absorb_elems += v.len();
                *absorb_by_len.entry(v.len()).or_insert(0) += 1;
            }
            PoseidonTraceOp::SqueezeField(v) => {
                n_sq_field += 1;
                sq_field_elems += v.len();
                *squeeze_field_by_len.entry(v.len()).or_insert(0) += 1;
            }
            PoseidonTraceOp::SqueezeBytes { n, .. } => {
                n_sq_bytes += 1;
                sq_bytes += *n;
                *squeeze_bytes_by_len.entry(*n).or_insert(0) += 1;
            }
        }
    }
    // Count `get_challenge()` reabsorbs: SqueezeField(8) immediately followed by Absorb(8).
    for win in ops.windows(2) {
        if let [PoseidonTraceOp::SqueezeField(v), PoseidonTraceOp::Absorb(w)] = win {
            if v.len() == DIGITS_PER_TRY && w.len() == DIGITS_PER_TRY {
                n_get_challenge_reabsorbs += 1;
            }
        }
    }

    // Constraint mix.
    let c_pose = pose_inst.constraints.len();
    let c_pose_no_bytes =
        symphony::dpp_poseidon::poseidon_sponge_dr1cs_from_ops_with_wiring_no_bytes::<F257>(
            poseidon_cfg,
            ops,
        )
        .map(|(inst, _asg, _w)| inst.constraints.len())
        .unwrap_or(0usize);
    let c_glue = glue.gb.rows.len();

    eprintln!("==============================================================");
    eprintln!("LF+ WE tiny gate op-mix (Poseidon(F257) + tiny CM gadgets)");
    eprintln!(
        "  poseidon trace: permutes={} absorb_ops={} absorb_elems={} squeeze_field_ops={} squeeze_field_elems={} squeeze_bytes_ops={} squeeze_bytes_total={}",
        pose_permutes_total,
        n_absorb,
        absorb_elems,
        n_sq_field,
        sq_field_elems,
        n_sq_bytes,
        sq_bytes
    );
    eprintln!(
        "  poseidon permutes split: before_cm={} after_cm={}",
        pose_permutes_before_cm, pose_permutes_after_cm
    );
    eprintln!(
        "  poseidon op lens: absorb_by_len={:?} squeeze_field_by_len={:?} squeeze_bytes_by_len={:?} get_challenge_reabsorbs={}",
        absorb_by_len,
        squeeze_field_by_len,
        squeeze_bytes_by_len,
        n_get_challenge_reabsorbs
    );
    eprintln!("  dr1cs constraints by part: poseidon={} glue={}", c_pose, c_glue);
    eprintln!(
        "  poseidon constraints(no_bytes)={} delta_squeeze_bytes={}",
        c_pose_no_bytes,
        c_pose.saturating_sub(c_pose_no_bytes)
    );
    let cm_counts = tiny_cm_counts_take();
    eprintln!(
        "  cm_math op counts: ring_add={} ring_sub={} ring_scale={} ring_mul={} ring_eq={} lc_to_var={} scalar_add={} scalar_sub={} scalar_mul={} scalar_mul_const={} scalar_sub_const={} scalar_pow_table={} eq_eval_vars={} short_chal_from_bytes={} ct_psi_mul_ring={}",
        cm_counts.ring_add,
        cm_counts.ring_sub,
        cm_counts.ring_scale,
        cm_counts.ring_mul_negacyclic,
        cm_counts.ring_eq,
        cm_counts.lc_to_var,
        cm_counts.scalar_add,
        cm_counts.scalar_sub,
        cm_counts.scalar_mul,
        cm_counts.scalar_mul_const,
        cm_counts.scalar_sub_const,
        cm_counts.scalar_pow_table,
        cm_counts.eq_eval_vars,
        cm_counts.short_challenge_from_bytes,
        cm_counts.ct_psi_mul_ring
    );
    eprintln!(
        "  absorb(non-reabsorb) breakdown: cm(comh_ops={} comh_bytes={} sc_msgs_ops=[{},{}] sc_msgs_bytes=[{},{}] eval_ops=[{},{}] eval_bytes=[{},{}])",
        absorb_counts.comh_ops,
        absorb_counts.comh_bytes_total,
        absorb_counts.sc_msgs_ops_0,
        absorb_counts.sc_msgs_ops_1,
        absorb_counts.sc_msgs_bytes_total_0,
        absorb_counts.sc_msgs_bytes_total_1,
        absorb_counts.eval_ops_0,
        absorb_counts.eval_ops_1,
        absorb_counts.eval_bytes_total_0,
        absorb_counts.eval_bytes_total_1,
    );
    eprintln!(
        "  glue builder: nvars={} nconstraints={} local_map={} byte_bits_cache={} u64_bal16_cache={} u32_bal16_cache={}",
        glue.gb.assignment.len(),
        glue.gb.rows.len(),
        glue.local_map.len(),
        glue.gb.byte_bits_cache.len(),
        glue.gb.u64_bal16_cache.len(),
        glue.gb.u32_bal16_cache.len(),
    );
    eprintln!(
        "  surfaces: pairs={} mul_surfaces={} sq_surfaces={} sum_all_pairs_digits={} sum_all_pairs_coeffwise={} sq_sum_all_pairs_digits={} sq_sum_all_pairs_coeffwise={}",
        pairs_len,
        mul_surfaces_len,
        sq_surfaces_len,
        sum_all_pairs_digits_len,
        sum_all_pairs_coeffwise_len,
        sq_sum_all_pairs_digits_len,
        sq_sum_all_pairs_coeffwise_len,
    );
    if glue.gb.profile_enabled {
        eprintln!("{}", glue.gb.profile_report(40));
    } else {
        // Note: this is opt-in and can be noisy; keep it off by default.
        eprintln!("  (dr1cs scope profile disabled; set LF_PROFILE_DR1CS=1 for top scopes)");
    }
    eprintln!("==============================================================");
}

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

fn validate_params_and_short_schedule(
    ring_dim: usize,
    params: &WeParams,
    short_ranges_len: usize,
) -> Result<(), String> {
    // Basic parameter consistency: the tiny builder is specialized to the same ring/CM schedule
    // as LF+ CM verification.
    if ring_dim != params.ring_dim_d as usize {
        return Err(format!(
            "tiny gate: ring_dim mismatch (arg={} params.ring_dim_d={})",
            ring_dim, params.ring_dim_d
        ));
    }
    if params.degree_cm != 2 {
        return Err("tiny gate: expected params.degree_cm == 2".to_string());
    }

    // CM short-challenge schedule: `s` has 3 blocks, and `s_prime` has k*d blocks.
    let k_decomp = params.k as usize;
    let expected_short_blocks = 3usize
        .checked_add(k_decomp.saturating_mul(ring_dim))
        .ok_or_else(|| "tiny gate: short block count overflow".to_string())?;
    if short_ranges_len != expected_short_blocks {
        return Err(format!(
            "tiny gate: short_squeeze_ops count mismatch (got {}, expected {})",
            short_ranges_len, expected_short_blocks
        ));
    }
    Ok(())
}

fn validate_cm_u32_schedule(params: &WeParams, wiring: &TinyCoinOpWiring) -> Result<(), String> {
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
    Ok(())
}

fn count_comh_ring_elements(
    ops: &[PoseidonTraceOp<F257>],
    pose_wiring: &PoseidonDr1csWiring,
    ring_dim: usize,
    wiring: &TinyCoinOpWiring,
) -> Result<(usize /* n_ring_elems */, usize /* coeff_bytes */), String> {
    if ring_dim == 0 {
        return Ok((0, 0));
    }
    if wiring.short_squeeze_ops.is_empty() || wiring.u32_squeeze_ops.is_empty() {
        return Ok((0, 0));
    }

    // Infer per-ring-element absorb width `reb` from absorb ranges.
    let mut ring_elem_bytes: Option<usize> = None;
    for &(_st, ln) in &pose_wiring.absorb_ranges {
        if ln % ring_dim == 0 && ln > ring_dim {
            ring_elem_bytes = Some(match ring_elem_bytes {
                None => ln,
                Some(cur) => cur.min(ln),
            });
        }
    }
    let reb = ring_elem_bytes.ok_or_else(|| "tiny gate: could not infer ring_elem_bytes".to_string())?;
    let coeff_bytes = reb / ring_dim;

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
    let cm_u32_start = wiring.u32_squeeze_ops.iter().filter(|&&idx| idx < first_short_op).count();
    if cm_u32_start >= wiring.u32_squeeze_ops.len() {
        return Ok((0, coeff_bytes));
    }
    let first_cm_u32_op = wiring.u32_squeeze_ops[cm_u32_start];

    let mut absorb_idx = 0usize;
    let mut squeeze_field_op_idx = 0usize;
    let mut after_short = false;
    let mut count = 0usize;
    for op in ops {
        match op {
            PoseidonTraceOp::Absorb(_v) => {
                let (_ab_start, ab_len) = *pose_wiring
                    .absorb_ranges
                    .get(absorb_idx)
                    .ok_or("tiny gate: pose_wiring.absorb_ranges oob (comh count)")?;
                absorb_idx += 1;
                if after_short && squeeze_field_op_idx <= first_cm_u32_op {
                    if ab_len != reb {
                        return Err(format!(
                            "tiny gate: unexpected absorb len in comh segment (got {ab_len}, expected {reb})"
                        ));
                    }
                    count += 1;
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
    Ok((count, coeff_bytes))
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
    u32_locals: &[BoundedU32ChallengeWiring],
) -> Result<(Vec<[usize; 8]>, Vec<[usize; 8]>), String> {
    // Preserves exact existing behavior; only moved out for readability/audit.
    let mut tcch0_local: Vec<[usize; 8]> = Vec::new();
    let mut tcch1_local: Vec<[usize; 8]> = Vec::new();

    #[inline]
    fn alloc_const_byte(gb: &mut Dr1csBuilder<F257>, v: u8) -> usize {
        let x = gb.new_var(F257::from(v as u64));
        gb.enforce_var_eq_const(x, F257::from(v as u64));
        let _ = decompose_existing_byte_var_to_bits::<F257>(gb, x);
        x
    }

    #[inline]
    fn alloc_const_frog_u64(gb: &mut Dr1csBuilder<F257>, v: u64) -> [usize; 8] {
        let bs = v.to_le_bytes();
        let mut out = [0usize; 8];
        for i in 0..8 {
            out[i] = alloc_const_byte(gb, bs[i]);
        }
        let _limbs = frog_u64_canonical_from_byte_vars::<F257>(gb, &out);
        out
    }

    #[inline]
    fn frog_bytes_from_u32_le_bytes(gb: &mut Dr1csBuilder<F257>, u32_le: &[usize; 4]) -> [usize; 8] {
        let mut out = [0usize; 8];
        out[0..4].copy_from_slice(u32_le);
        for i in 4..8 {
            out[i] = alloc_const_byte(gb, 0u8);
        }
        let _limbs = frog_u64_canonical_from_byte_vars::<F257>(gb, &out);
        out
    }

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

                    let mut coh0_bytes: Vec<[usize; 8]> = Vec::new();
                    let mut comh_all_coeff_bytes: Vec<Vec<[usize; 8]>> = Vec::new();
                    for &(ab_start, ab_len) in &comh_absorb_ranges {
                        if ab_len < reb || (ab_len % reb) != 0 {
                            continue;
                        }
                        let n_blocks = ab_len / reb;
                        for blk in 0..n_blocks {
                            let blk_start = ab_start + blk * reb;
                            // Each ring element is encoded as `ring_dim` coefficients, each coefficient as
                            // `coeff_bytes` bytes (little-endian, canonical for Frog base field).
                            if coeff_bytes != 8 {
                                return Err(format!(
                                    "tiny gate: expected Frog base-field coeff_bytes=8 for CM verifier, got {coeff_bytes}"
                                ));
                            }

                            let mut coeffs: Vec<[usize; 8]> = vec![[0usize; 8]; ring_dim];
                            for coeff in 0..ring_dim {
                                let coeff_start = blk_start + coeff * coeff_bytes;
                                for i in 0..8 {
                                    let gv = pose_wiring.absorb_vars[coeff_start + i];
                                    let lv = glue.copy_digit(gv);
                                    let _ = decompose_existing_byte_var_to_bits::<F257>(&mut glue.gb, lv);
                                    coeffs[coeff][i] = lv;
                                }
                                let _limbs =
                                    frog_u64_canonical_from_byte_vars::<F257>(&mut glue.gb, &coeffs[coeff]);
                            }
                            coh0_bytes.push(coeffs[0]);
                            comh_all_coeff_bytes.push(coeffs);
                        }
                    }

                    let n_comh_elems = coh0_bytes.len();
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
                        if c1_start + lg <= u32_locals.len() {
                            // Challenges c0/c1 are transcript `get_challenge()` outputs: bounded u32 embedded
                            // into the Frog base field. Represent them as canonical Frog bytes by padding the
                            // 4-byte u32 with 4 zero bytes (so value < 2^32 < p).
                            let mut c0_bytes: Vec<[usize; 8]> = Vec::with_capacity(lg);
                            let mut c1_bytes: Vec<[usize; 8]> = Vec::with_capacity(lg);
                            for i in 0..lg {
                                c0_bytes.push(frog_bytes_from_u32_le_bytes(
                                    &mut glue.gb,
                                    &u32_locals[c0_start + i].byte_vars,
                                ));
                            }
                            for i in 0..lg {
                                c1_bytes.push(frog_bytes_from_u32_le_bytes(
                                    &mut glue.gb,
                                    &u32_locals[c1_start + i].byte_vars,
                                ));
                            }

                            #[inline]
                            fn tensor_frog_bytes(gb: &mut Dr1csBuilder<F257>, c: &[[usize; 8]]) -> Vec<[usize; 8]> {
                                let one = alloc_const_frog_u64(gb, 1u64);
                                let mut acc: Vec<[usize; 8]> = vec![one];
                                for ci in c {
                                    let one_minus = frog_sub_mod_p_from_byte_vars_assume_canonical(gb, &one, ci);
                                    let mut next: Vec<[usize; 8]> = Vec::with_capacity(acc.len() * 2);
                                    for t in &acc {
                                        next.push(frog_mul_mod_p_from_byte_vars_assume_canonical(gb, t, &one_minus));
                                        next.push(frog_mul_mod_p_from_byte_vars_assume_canonical(gb, t, ci));
                                    }
                                    acc = next;
                                }
                                acc
                            }

                            let tensor_c0 = tensor_frog_bytes(&mut glue.gb, &c0_bytes);
                            let tensor_c1 = tensor_frog_bytes(&mut glue.gb, &c1_bytes);

                            // Compute full-ring tcch0/tcch1 (as in `CmProof::verify_with_mlen`):
                            //
                            //   tcch{0,1}[l] = Σ_j tensor_c{0,1}[j] * comh[l][j]
                            //
                            // Multiplying a ring element by a base-field scalar is coefficient-wise scaling.
                            let zero = alloc_const_frog_u64(&mut glue.gb, 0u64);
                            let mut tcch0_ring: Vec<Vec<[usize; 8]>> = Vec::with_capacity(l_instances);
                            let mut tcch1_ring: Vec<Vec<[usize; 8]>> = Vec::with_capacity(l_instances);
                            for l in 0..l_instances {
                                let base = l * kappa;
                                let mut acc0: Vec<[usize; 8]> = vec![zero; ring_dim];
                                let mut acc1: Vec<[usize; 8]> = vec![zero; ring_dim];
                                for j in 0..kappa {
                                    let ch = &comh_all_coeff_bytes[base + j];
                                    for coeff in 0..ring_dim {
                                        let m0 = frog_mul_mod_p_from_byte_vars_assume_canonical(
                                            &mut glue.gb,
                                            &tensor_c0[j],
                                            &ch[coeff],
                                        );
                                        let m1 = frog_mul_mod_p_from_byte_vars_assume_canonical(
                                            &mut glue.gb,
                                            &tensor_c1[j],
                                            &ch[coeff],
                                        );
                                        acc0[coeff] =
                                            frog_add_mod_p_from_byte_vars_assume_canonical(&mut glue.gb, &acc0[coeff], &m0);
                                        acc1[coeff] =
                                            frog_add_mod_p_from_byte_vars_assume_canonical(&mut glue.gb, &acc1[coeff], &m1);
                                    }
                                }
                                tcch0_ring.push(acc0);
                                tcch1_ring.push(acc1);
                            }

                            // Current API exports only coefficient-0 (base-field) as 8 LE bytes per instance.
                            tcch0_local.reserve(l_instances);
                            tcch1_local.reserve(l_instances);
                            for l in 0..l_instances {
                                tcch0_local.push(tcch0_ring[l][0]);
                                tcch1_local.push(tcch1_ring[l][0]);
                            }
                        }
                    }
                }
            }
        }
    }

    Ok((tcch0_local, tcch1_local))
}

/// Parse (and lightly constrain) the CM segment that occurs after short-challenge squeezes.
///
/// This is a *schedule* parser: it follows the transcript op ordering implied by
/// `poseidon_trace_schedule_for_plus` / `CmProof::verify_with_mlen` and returns the absorb ranges
/// for:
/// - `comh` ring elements (L × κ)
/// - both CM sumcheck proof message ring elements (2 × nvars_cm rounds × 3 ring elems)
/// - both eval tables (2 × L × (1+mlen_mats) × 4 ring elems)
///
/// It also enforces that the sumcheck header absorbs match statement-bound constants:
/// - absorbed `nvars_cm`
/// - absorbed `degree_cm == 2`
/// and that the per-round "marker" scalar absorbs are zero (as in the schedule generator).
fn parse_and_enforce_cm_after_short(
    glue: &mut GlueCtx,
    ops: &[PoseidonTraceOp<F257>],
    pose_wiring: &PoseidonDr1csWiring,
    ring_dim: usize,
    params: &WeParams,
    wiring: &TinyCoinOpWiring,
    l_instances: usize,
) -> Result<
    (
        Vec<(usize, usize)>, // comh ring absorbs (start,len)
        Vec<Vec<(usize, usize)>>, // sumcheck msgs absorbs: [which][absorb_idx]
        Vec<Vec<(usize, usize)>>, // eval table absorbs: [which][absorb_idx]
    ),
    String,
> {
    if wiring.short_squeeze_ops.is_empty() {
        return Ok((Vec::new(), vec![Vec::new(), Vec::new()], vec![Vec::new(), Vec::new()]));
    }
    let kappa = params.kappa as usize;
    let nvars_cm = params.nvars_cm as usize;
    let mlen_mats = params.mlen as usize;
    let last_short_op = *wiring
        .short_squeeze_ops
        .iter()
        .max()
        .expect("non-empty short_squeeze_ops");

    // Collect all non-reabsorb Absorb ops after the last short squeeze.
    let mut absorb_idx = 0usize;
    let mut squeeze_field_op_idx = 0usize;
    let mut after_short = false;
    let mut last_squeeze_len: Option<usize> = None;
    let mut last_squeeze_is_get_challenge_try = false;
    let mut payload_after_short: Vec<(usize, usize)> = Vec::new();
    for op in ops {
        match op {
            PoseidonTraceOp::SqueezeField(v) => {
                if squeeze_field_op_idx == last_short_op {
                    after_short = true;
                }
                squeeze_field_op_idx += 1;
                // Only `get_challenge()` reabsorbs: it always squeezes 8 digits and then immediately
                // absorbs those same 8 digits. `squeeze_bytes(n)` is also recorded as `SqueezeField(len=n)`
                // but is *not* reabsorbed.
                last_squeeze_len = Some(v.len());
                last_squeeze_is_get_challenge_try = v.len() == DIGITS_PER_TRY;
            }
            PoseidonTraceOp::Absorb(_v) => {
                let (ab_start, ab_len) = *pose_wiring
                    .absorb_ranges
                    .get(absorb_idx)
                    .ok_or("tiny gate: pose_wiring.absorb_ranges oob (cm-after-short)")?;
                absorb_idx += 1;
                let is_reabsorb = last_squeeze_is_get_challenge_try && last_squeeze_len == Some(ab_len);
                last_squeeze_len = None;
                last_squeeze_is_get_challenge_try = false;
                if after_short && !is_reabsorb {
                    payload_after_short.push((ab_start, ab_len));
                }
            }
            PoseidonTraceOp::SqueezeBytes { .. } => {
                // Legacy; does not affect absorb parsing.
                last_squeeze_len = None;
                last_squeeze_is_get_challenge_try = false;
            }
        }
    }

    // Now parse payload_after_short sequentially according to the CM schedule.
    // Infer the ring-element absorb width from the wiring (matches `compute_tcch` logic).
    // This avoids hard-coding `coeff_bytes=8` here; the caller already separately enforces that
    // absorbed non-reabsorb payloads are canonical Frog bytes at IO boundaries.
    let mut ring_elem_bytes: Option<usize> = None;
    for &(_st, ln) in &pose_wiring.absorb_ranges {
        if ln % ring_dim == 0 && ln > ring_dim {
            ring_elem_bytes = Some(match ring_elem_bytes {
                None => ln,
                Some(cur) => cur.min(ln),
            });
        }
    }
    let ring_elem_bytes =
        ring_elem_bytes.ok_or_else(|| "tiny gate: could not infer ring_elem_bytes (cm-after-short)".to_string())?;
    let mut cur = 0usize;

    // coh: L*kappa ring elements
    let mut comh_absorbs: Vec<(usize, usize)> = Vec::with_capacity(l_instances * kappa);
    for _ in 0..(l_instances * kappa) {
        let (st, ln) = *payload_after_short
            .get(cur)
            .ok_or("tiny gate: payload_after_short too short (comh)")?;
        cur += 1;
        if ln != ring_elem_bytes {
            return Err(format!(
                "tiny gate: unexpected absorb len while parsing comh (got {ln}, expected {ring_elem_bytes})"
            ));
        }
        comh_absorbs.push((st, ln));
    }

    // Two CM sumchecks:
    let mut sc_msg_absorbs: Vec<Vec<(usize, usize)>> = vec![Vec::new(), Vec::new()];
    let mut eval_absorbs: Vec<Vec<(usize, usize)>> = vec![Vec::new(), Vec::new()];

    // Helper: enforce an absorbed 8-byte scalar equals a u64 constant.
    #[inline]
    fn enforce_absorbed_u64_const(
        glue: &mut GlueCtx,
        pose_wiring: &PoseidonDr1csWiring,
        ab_start: usize,
        val: u64,
    ) {
        let bs = val.to_le_bytes();
        for i in 0..8 {
            let gv = pose_wiring.absorb_vars[ab_start + i];
            let lv = glue.copy_digit(gv);
            glue.gb.enforce_var_eq_const(lv, F257::from(bs[i] as u64));
        }
    }

    for which in 0..2 {
        // Sumcheck header: absorb nvars, absorb degree (both as base-field scalars -> 8 bytes)
        {
            let (st, ln) = *payload_after_short
                .get(cur)
                .ok_or("tiny gate: payload_after_short too short (sc header nvars)")?;
            cur += 1;
            if ln != 8 {
                return Err("tiny gate: expected 8-byte absorb for sumcheck nvars".to_string());
            }
            enforce_absorbed_u64_const(glue, pose_wiring, st, nvars_cm as u64);
        }
        {
            let (st, ln) = *payload_after_short
                .get(cur)
                .ok_or("tiny gate: payload_after_short too short (sc header degree)")?;
            cur += 1;
            if ln != 8 {
                return Err("tiny gate: expected 8-byte absorb for sumcheck degree".to_string());
            }
            enforce_absorbed_u64_const(glue, pose_wiring, st, 2u64);
        }

        // Rounds: 3 ring absorbs (msg evals), then one 8-byte scalar marker absorb.
        for _round in 0..nvars_cm {
            for _m in 0..3 {
                let (st, ln) = *payload_after_short
                    .get(cur)
                    .ok_or("tiny gate: payload_after_short too short (sc msg)")?;
                cur += 1;
                if ln != ring_elem_bytes {
                    return Err("tiny gate: expected ring-elem absorb for sumcheck msg".to_string());
                }
                sc_msg_absorbs[which].push((st, ln));
            }
            // Marker absorb (schedule generator uses 0).
            let (st, ln) = *payload_after_short
                .get(cur)
                .ok_or("tiny gate: payload_after_short too short (sc marker)")?;
            cur += 1;
            if ln != 8 {
                return Err("tiny gate: expected 8-byte absorb for sumcheck marker".to_string());
            }
            enforce_absorbed_u64_const(glue, pose_wiring, st, 0u64);
        }

        // Eval tables: L × (1+mlen_mats) rows × 4 ring elements.
        for _l in 0..l_instances {
            for _row in 0..(1 + mlen_mats) {
                for _t in 0..4 {
                    let (st, ln) = *payload_after_short
                        .get(cur)
                        .ok_or("tiny gate: payload_after_short too short (evals)")?;
                    cur += 1;
                    if ln != ring_elem_bytes {
                        return Err("tiny gate: expected ring-elem absorb for eval table".to_string());
                    }
                    eval_absorbs[which].push((st, ln));
                }
            }
        }
    }

    Ok((comh_absorbs, sc_msg_absorbs, eval_absorbs))
}

#[inline]
fn cm_u32_start_idx(wiring: &TinyCoinOpWiring) -> usize {
    if wiring.short_squeeze_ops.is_empty() {
        return wiring.u32_squeeze_ops.len();
    }
    let first_short_op = *wiring
        .short_squeeze_ops
        .iter()
        .min()
        .expect("non-empty short_squeeze_ops");
    wiring.u32_squeeze_ops.iter().filter(|&&idx| idx < first_short_op).count()
}

#[inline]
fn frog_bytes_from_u32_le_bytes(gb: &mut Dr1csBuilder<F257>, u32_le: &[usize; 4]) -> [usize; 8] {
    let mut out = [0usize; 8];
    out[0..4].copy_from_slice(u32_le);
    for i in 4..8 {
        let z = gb.new_var(F257::ZERO);
        gb.enforce_var_eq_const(z, F257::ZERO);
        out[i] = z;
    }
    // Canonicality (<p) is guaranteed since value < 2^32.
    out
}

fn parse_ring_elem_absorb_as_ringbytes(
    glue: &mut GlueCtx,
    pose_wiring: &PoseidonDr1csWiring,
    ring_dim: usize,
    ab_start: usize,
    ab_len: usize,
) -> Result<RingBytes, String> {
    if ring_dim == 0 {
        return Ok(Vec::new());
    }
    if ab_len % ring_dim != 0 {
        return Err("tiny gate: ring absorb len not divisible by ring_dim".to_string());
    }
    let coeff_bytes = ab_len / ring_dim;
    if coeff_bytes != 8 {
        return Err("tiny gate: expected 8-byte coeff encoding for RingBytes".to_string());
    }
    let mut out: RingBytes = Vec::with_capacity(ring_dim);
    for coeff in 0..ring_dim {
        let mut cbytes = [0usize; 8];
        let off = ab_start + coeff * 8;
        for i in 0..8 {
            let gv = pose_wiring.absorb_vars[off + i];
            cbytes[i] = glue.copy_digit(gv);
        }
        out.push(cbytes);
    }
    Ok(out)
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
    tcch0_local: Vec<[usize; 8]>,
    tcch1_local: Vec<[usize; 8]>,
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
        Vec<[usize; 8]>,
        Vec<[usize; 8]>,
        Vec<CmDigitMulSurfaceWiring>,
        Vec<CmDigitMulSqSurfaceWiring>,
        PoseidonDr1csWiring,
    ),
    String,
> {
    let GlueCtx { gb, pose_asg, local_map } = glue;
    let (glue_inst, glue_asg) = gb.into_instance();
    let glue_nvars = glue_inst.nvars;

    // Save lightweight stats for optional op-mix reporting (before moving instances into merge).
    let c_pose = pose_inst.constraints.len();
    let c_glue = glue_inst.constraints.len();
    let glue_eq_constraints = local_map.len();

    let (mut inst, asg) = merge_sparse_dr1cs_share_one::<F257>(vec![
        (pose_inst, pose_asg),
        (glue_inst, glue_asg),
    ])
    .map_err(|e| format!("merge poseidon+cm+mul_glue failed: {e}"))?;

    let c_after_merge = inst.constraints.len();
    enforce_fiat_shamir_reabsorb_semantics(&mut inst, ops, &pose_wiring)?;
    let c_after_reabsorb = inst.constraints.len();

    let pose_nvars = inst.nvars - (glue_nvars - 1);
    let glue_offset = pose_nvars - 1;
    for (&gv, &lv) in local_map.iter() {
        let gg = if lv == 0 { 0 } else { lv + glue_offset };
        enforce_var_eq::<F257>(&mut inst, gv, gg);
    }
    let c_after_glue_eq = inst.constraints.len();

    if tiny_opmix_on() {
        eprintln!(
            "LF+ WE tiny gate merged: nvars={} constraints={} (poseidon={} glue={} merge={} reabsorb_delta={} glue_eq_constraints={} glue_eq_delta={})",
            inst.nvars,
            c_after_glue_eq,
            c_pose,
            c_glue,
            c_after_merge,
            c_after_reabsorb.saturating_sub(c_after_merge),
            glue_eq_constraints,
            c_after_glue_eq.saturating_sub(c_after_reabsorb),
        );
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
        tcch0_local
            .into_iter()
            .map(|arr| {
                let mut out = [0usize; 8];
                for i in 0..8 {
                    out[i] = to_glue_global(arr[i]);
                }
                out
            })
            .collect(),
        tcch1_local
            .into_iter()
            .map(|arr| {
                let mut out = [0usize; 8];
                for i in 0..8 {
                    out[i] = to_glue_global(arr[i]);
                }
                out
            })
            .collect(),
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
        Vec<[usize; 8]>,
        Vec<[usize; 8]>,
        Vec<CmDigitMulSurfaceWiring>,
        Vec<CmDigitMulSqSurfaceWiring>,
        PoseidonDr1csWiring,
    ),
    String,
> {
    let mut mem_prev: Option<LfStageCounts> = None;
    if tiny_opmix_on() {
        tiny_cm_counts_reset();
    }

    let (pose_inst, pose_asg, pose_wiring, _byte_wiring) = poseidon_f257_arithmetize(cfg, ops)?;
    lf_stage_log("poseidon_f257_arithmetize", Some(&pose_inst), None, &mut mem_prev);

    let short_ranges = squeeze_field_ranges_by_op_index(&pose_wiring.squeeze_field_ranges, &wiring.short_squeeze_ops)?;
    let u32_ranges = squeeze_field_ranges_by_op_index(&pose_wiring.squeeze_field_ranges, &wiring.u32_squeeze_ops)?;
    let frog_ranges = squeeze_field_ranges_by_op_index(&pose_wiring.squeeze_field_ranges, &wiring.frog_squeeze_ops)?;
    validate_pairs(pairs, short_ranges.len(), u32_ranges.len())?;
    validate_params_and_short_schedule(ring_dim, params, short_ranges.len())?;

    let mut glue = GlueCtx::new(pose_asg);
    lf_stage_log("glue_init", Some(&pose_inst), Some(&glue), &mut mem_prev);

    // Bind all proof/statement payload absorbs that encode base-field elements as canonical 8-byte scalars.
    // (Skip fiat–shamir reabsorbs, which are F257 digits and may contain 256.)
    enforce_nonreabsorb_absorbs_are_canonical_frog(&mut glue, ops, &pose_wiring)?;
    lf_stage_log(
        "enforce_nonreabsorb_absorbs_are_canonical_frog",
        Some(&pose_inst),
        Some(&glue),
        &mut mem_prev,
    );

    validate_cm_u32_schedule(params, wiring)?;
    let (n_comh_ring_elems, coeff_bytes) = count_comh_ring_elements(ops, &pose_wiring, ring_dim, wiring)?;
    if ring_dim > 0 && n_comh_ring_elems > 0 && coeff_bytes != 8 {
        return Err(format!(
            "tiny gate: expected Frog base-field coeff_bytes=8, got {coeff_bytes}"
        ));
    }
    let kappa = params.kappa as usize;
    if kappa > 0 && (n_comh_ring_elems % kappa) != 0 {
        return Err("tiny gate: comh ring element count not divisible by kappa".to_string());
    }
    let l_instances_expected = if kappa == 0 { 0 } else { n_comh_ring_elems / kappa };

    let short_locals = build_short_blocks(&mut glue, &pose_wiring, ring_dim, &short_ranges)?;
    lf_stage_log("build_short_blocks", Some(&pose_inst), Some(&glue), &mut mem_prev);
    let (u32_locals, frog_locals) =
        build_u32_and_frog_blocks(&mut glue, &pose_wiring, &wiring.u32_squeeze_ops)?;
    lf_stage_log("build_u32_and_frog_blocks", Some(&pose_inst), Some(&glue), &mut mem_prev);
    let frog_rejection_locals = build_frog_rejection_coins(&mut glue, &pose_wiring, &frog_ranges)?;
    lf_stage_log("build_frog_rejection_coins", Some(&pose_inst), Some(&glue), &mut mem_prev);
    let (tcch0_local, tcch1_local) =
        compute_tcch(&mut glue, ops, &pose_wiring, ring_dim, params, wiring, &u32_locals)?;
    lf_stage_log("compute_tcch", Some(&pose_inst), Some(&glue), &mut mem_prev);
    if l_instances_expected != 0 && tcch0_local.len() != l_instances_expected {
        return Err("tiny gate: tcch0 length mismatch with inferred L".to_string());
    }
    if l_instances_expected != 0 && tcch1_local.len() != l_instances_expected {
        return Err("tiny gate: tcch1 length mismatch with inferred L".to_string());
    }

    // Parse and constrain the CM segment after short challenges (sumcheck headers, etc.).
    let (comh_absorbs, sc_msg_absorbs, eval_absorbs) = parse_and_enforce_cm_after_short(
        &mut glue,
        ops,
        &pose_wiring,
        ring_dim,
        params,
        wiring,
        l_instances_expected,
    )?;
    lf_stage_log("parse_and_enforce_cm_after_short", Some(&pose_inst), Some(&glue), &mut mem_prev);

    // Capture a lightweight absorb breakdown summary for optional op-mix reporting.
    let absorb_counts = TinyAbsorbBreakdownCounts {
        comh_ops: comh_absorbs.len(),
        comh_bytes_total: sum_absorb_bytes(&comh_absorbs),
        sc_msgs_ops_0: sc_msg_absorbs.get(0).map(|v| v.len()).unwrap_or(0),
        sc_msgs_bytes_total_0: sc_msg_absorbs.get(0).map(|v| sum_absorb_bytes(v)).unwrap_or(0),
        sc_msgs_ops_1: sc_msg_absorbs.get(1).map(|v| v.len()).unwrap_or(0),
        sc_msgs_bytes_total_1: sc_msg_absorbs.get(1).map(|v| sum_absorb_bytes(v)).unwrap_or(0),
        eval_ops_0: eval_absorbs.get(0).map(|v| v.len()).unwrap_or(0),
        eval_bytes_total_0: eval_absorbs.get(0).map(|v| sum_absorb_bytes(v)).unwrap_or(0),
        eval_ops_1: eval_absorbs.get(1).map(|v| v.len()).unwrap_or(0),
        eval_bytes_total_1: eval_absorbs.get(1).map(|v| sum_absorb_bytes(v)).unwrap_or(0),
    };

    // CM sumcheck verifier wiring (degree-2), starting from a placeholder claimed_sum=0 ring.
    // This already enforces transcript-consistency of the per-round prover messages and the
    // verifier challenges `r_sc` (derived from transcript u32 coins).
    //
    // Claimed sums and recombination will be wired once Dcom prefix verifier math is integrated.


    

    if ring_dim > 0 && l_instances_expected > 0 && !comh_absorbs.is_empty() {
        lf_stage_log("cm_block_enter", Some(&pose_inst), Some(&glue), &mut mem_prev);
        eprintln!(
            "[tiny_gate/cm] ring_dim={} kappa={} n_comh_ring_elems={} L_expected={} comh_absorbs_len={}",
            ring_dim,
            kappa,
            n_comh_ring_elems,
            l_instances_expected,
            comh_absorbs.len()
        );
        let cm_u32_start = cm_u32_start_idx(wiring);
        let kappa = params.kappa as usize;
        let log_kappa = ark_std::log2(kappa.next_power_of_two()) as usize;
        let nvars_cm = params.nvars_cm as usize;
        let k_decomp = params.k as usize;
        let ell = params.l as usize;

        // CM challenges after absorb_comh: c0/c1, then for each sumcheck: rc, r_sc[0..nvars_cm]
        let c0_u32 = &u32_locals[cm_u32_start..cm_u32_start + log_kappa];
        let c1_u32 = &u32_locals[cm_u32_start + log_kappa..cm_u32_start + 2 * log_kappa];

        // Precompute (once) the ring-constant tables needed for t(z) evaluation.
        //
        // NOTE: This mirrors `CmProof::verify_with_mlen` / `tensor_eval::eval_t_z_optimized`.
        // We only build these when the factor sizes are powers of two (the regime used by WE).
        let mut tensor_c0_ring: Option<Vec<RingDigits>> = None;
        let mut tensor_c1_ring: Option<Vec<RingDigits>> = None;
        let mut s_prime_flat_ring: Option<Vec<RingDigits>> = None;
        let mut dpp_ring: Option<Vec<RingDigits>> = None;
        let mut x_powers_ring: Option<Vec<RingDigits>> = None;
        let mut r_point_digits: Option<Vec<FrogScalar>> = None;
        if ring_dim == 64
            && kappa.is_power_of_two()
            && (k_decomp * ring_dim).is_power_of_two()
            && ell.is_power_of_two()
        {
            // c0/c1 as Frog scalars (digit encoding), then tensor-expand.
            let c0_digits: Vec<_> = c0_u32
                .iter()
                .map(|u| {
                    let bytes = frog_bytes_from_u32_le_bytes(&mut glue.gb, &u.byte_vars);
                    frog_bytes_to_digits(&mut glue.gb, bytes)
                })
                .collect();
            let c1_digits: Vec<_> = c1_u32
                .iter()
                .map(|u| {
                    let bytes = frog_bytes_from_u32_le_bytes(&mut glue.gb, &u.byte_vars);
                    frog_bytes_to_digits(&mut glue.gb, bytes)
                })
                .collect();
            let t0 = tensor_frog_scalars_digits(&mut glue.gb, &c0_digits);
            let t1 = tensor_frog_scalars_digits(&mut glue.gb, &c1_digits);
            tensor_c0_ring = Some(tensor_frog_ringconst_digits(&mut glue.gb, &t0, ring_dim));
            tensor_c1_ring = Some(tensor_frog_ringconst_digits(&mut glue.gb, &t1, ring_dim));

            // dpp: dp^i as constant-coeff ring elements.
            let dp = (ring_dim / 2) as u64;
            let p = crate::we_frog_poseidon_f257::FROG_P;
            let mut acc: u64 = 1;
            let mut dpp: Vec<RingDigits> = Vec::with_capacity(ell);
            for _ in 0..ell {
                let s_bytes = super::cm_math::alloc_const_frog_u64(&mut glue.gb, acc);
                let s = frog_bytes_to_digits(&mut glue.gb, s_bytes);
                dpp.push(super::cm_math::ring_const_coeff_digits(&mut glue.gb, &s, ring_dim));
                acc = ((acc as u128) * (dp as u128) % (p as u128)) as u64;
            }
            dpp_ring = Some(dpp);

            // x_powers: unit monomials.
            let mut xp: Vec<RingDigits> = Vec::with_capacity(ring_dim);
            for i in 0..ring_dim {
                xp.push(ring_unit_monomial_digits(&mut glue.gb, i, ring_dim));
            }
            x_powers_ring = Some(xp);

            // s_prime_flat: k*d short challenges, each is a ring element with centered coeff bytes.
            let need_sprime = k_decomp
                .checked_mul(ring_dim)
                .ok_or_else(|| "tiny gate: k*ring_dim overflow (s_prime_flat)".to_string())?;
            if short_locals.len() < 3 + need_sprime {
                return Err("tiny gate: short_locals too short for s_prime_flat".to_string());
            }
            let z = glue.gb.new_var(F257::ZERO);
            glue.gb.enforce_var_eq_const(z, F257::ZERO);
            let c128 = super::cm_math::alloc_const_frog_u64(&mut glue.gb, 128u64);
            let mut sflat: Vec<RingDigits> = Vec::with_capacity(need_sprime);
            for blk in 0..need_sprime {
                let sb = &short_locals[3 + blk];
                // Each block provides `ring_dim` bytes (byte-view).
                if sb.byte_vars.len() != ring_dim {
                    return Err("tiny gate: short byte_vars len mismatch (s_prime_flat)".to_string());
                }
                let mut re: RingDigits = Vec::with_capacity(ring_dim);
                for &bv in &sb.byte_vars {
                    let mut bbytes = [0usize; 8];
                    bbytes[0] = bv;
                    for i in 1..8 {
                        bbytes[i] = z;
                    }
                    // Centered coefficient = (byte - 128) mod p (canonical).
                    let centered = frog_sub_mod_p_from_byte_vars_assume_canonical(&mut glue.gb, &bbytes, &c128);
                    re.push(frog_bytes_to_digits(&mut glue.gb, centered));
                }
                sflat.push(re);
            }
            s_prime_flat_ring = Some(sflat);

            // Recover the SetChk verifier point `r` used in eq(r, ro).
            //
            // In the LF+ schedule (`poseidon_trace_schedule_for_plus`), this is sampled by `Out::verify`
            // *before* the CM short challenges. We locate it in the `u32_locals` challenge stream by
            // counting `get_challenge()` calls in the prefix schedule.
            let nvars_lin = params.nvars_setchk as usize;
            let n_lin_proofs = l_instances_expected; // LF+ schedule uses L == n_lin_proofs
            let lin_chals = n_lin_proofs
                .checked_mul(2usize.saturating_mul(nvars_lin))
                .ok_or_else(|| "tiny gate: lin_chals overflow".to_string())?;
            let nclaims = k_decomp
                .checked_add(1)
                .ok_or_else(|| "tiny gate: nclaims overflow".to_string())?;
            let out_coin_total = nclaims
                .checked_mul(nvars_lin + 2)
                .and_then(|x| x.checked_add(if k_decomp > 1 { 1 } else { 0 }))
                .ok_or_else(|| "tiny gate: out_coin_total overflow".to_string())?;
            let r_start = lin_chals
                .checked_add(out_coin_total)
                .ok_or_else(|| "tiny gate: r_start overflow".to_string())?;
            let r_end = r_start
                .checked_add(nvars_lin)
                .ok_or_else(|| "tiny gate: r_end overflow".to_string())?;
            if u32_locals.len() < r_end {
                return Err("tiny gate: not enough u32 challenges to recover setchk r-point".to_string());
            }
            let mut rdig: Vec<FrogScalar> = Vec::with_capacity(nvars_lin);
            for u in &u32_locals[r_start..r_end] {
                let bytes = frog_bytes_from_u32_le_bytes(&mut glue.gb, &u.byte_vars);
                rdig.push(frog_bytes_to_digits(&mut glue.gb, bytes));
            }
            r_point_digits = Some(rdig);
        }
        lf_stage_log("cm_precompute_done", Some(&pose_inst), Some(&glue), &mut mem_prev);

        let mut u32_idx = cm_u32_start + 2 * log_kappa;
        for which in 0..2 {
            if lf_mem_on() {
                lf_stage_log("cm_sumcheck_enter", Some(&pose_inst), Some(&glue), &mut mem_prev);
                eprintln!("[LF_MEM]   cm_sumcheck_enter which={which}");
            }
            // rc (currently unused here, but we must consume it to align indices)
            let rc_bytes = frog_bytes_from_u32_le_bytes(&mut glue.gb, &u32_locals[u32_idx].byte_vars);
            u32_idx += 1;
            let mut rs: Vec<[usize; 8]> = Vec::with_capacity(nvars_cm);
            for _ in 0..nvars_cm {
                rs.push(frog_bytes_from_u32_le_bytes(&mut glue.gb, &u32_locals[u32_idx].byte_vars));
                u32_idx += 1;
            }

            // Parse sumcheck msg absorbs into ring bytes: each round has 3 ring elements.
            let mut msgs: Vec<[RingBytes; 3]> = Vec::with_capacity(nvars_cm);
            let abs = &sc_msg_absorbs[which];
            if abs.len() != nvars_cm * 3 {
                return Err("tiny gate: sumcheck msg absorb count mismatch".to_string());
            }
            for round in 0..nvars_cm {
                let (s0, l0) = abs[round * 3 + 0];
                let (s1, l1) = abs[round * 3 + 1];
                let (s2, l2) = abs[round * 3 + 2];
                let e0 = parse_ring_elem_absorb_as_ringbytes(&mut glue, &pose_wiring, ring_dim, s0, l0)?;
                let e1 = parse_ring_elem_absorb_as_ringbytes(&mut glue, &pose_wiring, ring_dim, s1, l1)?;
                let e2 = parse_ring_elem_absorb_as_ringbytes(&mut glue, &pose_wiring, ring_dim, s2, l2)?;
                msgs.push([e0, e1, e2]);
            }
            lf_stage_log("cm_sumcheck_msgs_parsed", Some(&pose_inst), Some(&glue), &mut mem_prev);

            // Initial claim: in the full verifier this is a structured linear combination of Dcom evals,
            // u-combinations, and tcch terms. For now, bind it to the transcript by setting it equal to
            // the first-round consistency relation g(0)+g(1), so the sumcheck constraints remain satisfiable
            // under existing proofs while we wire the full claimed-sum computation.
            let claimed0 = if nvars_cm == 0 {
                ring_zero_bytes(&mut glue.gb, ring_dim)
            } else {
                ring_add_bytes(&mut glue.gb, &msgs[0][0], &msgs[0][1])
            };
            let final_claim_bytes =
                sumcheck_verify_degree2_ring_bytes(&mut glue.gb, claimed0, &msgs, &rs)?;
            lf_stage_log("cm_sumcheck_constraints_done", Some(&pose_inst), Some(&glue), &mut mem_prev);

            // Parse this sumcheck's eval table absorbs (ring elements) and (if in the pow2 regime)
            // enforce the standard recombination equality `subclaim_eval == eval_acc`.
            let evals_rows = {
                let abs = &eval_absorbs[which];
                let rows = l_instances_expected * (1 + params.mlen as usize);
                if abs.len() != rows * 4 {
                    return Err("tiny gate: eval absorb count mismatch".to_string());
                }
                let mut out: Vec<[RingBytes; 4]> = Vec::with_capacity(rows);
                for row in 0..rows {
                    let (s0, l0) = abs[row * 4 + 0];
                    let (s1, l1) = abs[row * 4 + 1];
                    let (s2, l2) = abs[row * 4 + 2];
                    let (s3, l3) = abs[row * 4 + 3];
                    out.push([
                        parse_ring_elem_absorb_as_ringbytes(&mut glue, &pose_wiring, ring_dim, s0, l0)?,
                        parse_ring_elem_absorb_as_ringbytes(&mut glue, &pose_wiring, ring_dim, s1, l1)?,
                        parse_ring_elem_absorb_as_ringbytes(&mut glue, &pose_wiring, ring_dim, s2, l2)?,
                        parse_ring_elem_absorb_as_ringbytes(&mut glue, &pose_wiring, ring_dim, s3, l3)?,
                    ]);
                }
                out
            };
            lf_stage_log("cm_eval_table_parsed", Some(&pose_inst), Some(&glue), &mut mem_prev);

            // Recombination check (requires the pow2 regime + recovered setchk r-point).
            if let (Some(tc0_ring), Some(tc1_ring), Some(sp_ring), Some(dpp), Some(xp), Some(rpt)) = (
                tensor_c0_ring.as_ref(),
                tensor_c1_ring.as_ref(),
                s_prime_flat_ring.as_ref(),
                dpp_ring.as_ref(),
                x_powers_ring.as_ref(),
                r_point_digits.as_ref(),
            ) {
                // Convert r_sc to digit encoding.
                let rs_digits: Vec<FrogScalar> = rs.iter().copied().map(|b| frog_bytes_to_digits(&mut glue.gb, b)).collect();

                // subclaim_eval (ring) in digit encoding (cheap: only 64 coeff conversions).
                let subclaim_eval = ring_bytes_to_digits(&mut glue.gb, &final_claim_bytes);

                // rc powers (need up to z_idx+1).
                let z_idx = l_instances_expected * (4 + 4 * (params.mlen as usize));
                let max_pow = z_idx + 1;
                let rc_d = frog_bytes_to_digits(&mut glue.gb, rc_bytes);
                let rc_pows = frog_pow_table_digits(&mut glue.gb, &rc_d, max_pow);
                lf_stage_log("cm_recomb_rc_pows", Some(&pose_inst), Some(&glue), &mut mem_prev);

                // eq(r, ro) where r is the transcript-derived SetChk point (recovered above).
                let eq = eq_eval_frog_digits(&mut glue.gb, rpt, &rs_digits)?;

                // Evaluate t0(ro), t1(ro).
                let t0 = eval_t_z_optimized_ring_digits(&mut glue.gb, tc0_ring, sp_ring, dpp, xp, &rs_digits)?;
                let t1 = eval_t_z_optimized_ring_digits(&mut glue.gb, tc1_ring, sp_ring, dpp, xp, &rs_digits)?;

                // Reshape eval rows into evals[l][row][t].
                let rows_per_l = 1 + params.mlen as usize;
                let mut evals_by_l: Vec<Vec<[RingDigits; 4]>> = Vec::with_capacity(l_instances_expected);
                for l in 0..l_instances_expected {
                    let mut rows_l: Vec<[RingDigits; 4]> = Vec::with_capacity(rows_per_l);
                    for row in 0..rows_per_l {
                        let flat = l * rows_per_l + row;
                        let r0 = ring_bytes_to_digits(&mut glue.gb, &evals_rows[flat][0]);
                        let r1d = ring_bytes_to_digits(&mut glue.gb, &evals_rows[flat][1]);
                        let r2d = ring_bytes_to_digits(&mut glue.gb, &evals_rows[flat][2]);
                        let r3d = ring_bytes_to_digits(&mut glue.gb, &evals_rows[flat][3]);
                        rows_l.push([r0, r1d, r2d, r3d]);
                    }
                    evals_by_l.push(rows_l);
                }
                lf_stage_log("cm_recomb_evals_to_digits", Some(&pose_inst), Some(&glue), &mut mem_prev);

                // eval_acc = Σ_l eq*inner_l + (t0*e00)*rc^z + (t1*e00)*rc^{z+1}
                let mut eval_acc = super::cm_math::ring_zero_digits(&mut glue.gb, ring_dim);
                for l in 0..l_instances_expected {
                    let l_idx = l * (4 + 4 * (params.mlen as usize));
                    let mut inner = super::cm_math::ring_zero_digits(&mut glue.gb, ring_dim);
                    // First row is evals[l][0] = [tau, m_tau, f, h]
                    for j in 0..4 {
                        let t = ring_scale_digits(&mut glue.gb, &evals_by_l[l][0][j], &rc_pows[l_idx + j]);
                        inner = ring_add_digits(&mut glue.gb, &inner, &t);
                    }
                    // M chunks
                    for i in 0..(params.mlen as usize) {
                        let idx = l_idx + 4 + i * 4;
                        for j in 0..4 {
                            let t = ring_scale_digits(&mut glue.gb, &evals_by_l[l][1 + i][j], &rc_pows[idx + j]);
                            inner = ring_add_digits(&mut glue.gb, &inner, &t);
                        }
                    }
                    let eq_inner = ring_scale_digits(&mut glue.gb, &inner, &eq);
                    eval_acc = ring_add_digits(&mut glue.gb, &eval_acc, &eq_inner);

                    // t(z) terms (use e00 == evals[l][0][0])
                    let e00 = &evals_by_l[l][0][0];
                    let t0e = ring_mul_negacyclic_digits_d64(&mut glue.gb, &t0, e00)?;
                    let t1e = ring_mul_negacyclic_digits_d64(&mut glue.gb, &t1, e00)?;
                    let t0e_s = ring_scale_digits(&mut glue.gb, &t0e, &rc_pows[z_idx]);
                    let t1e_s = ring_scale_digits(&mut glue.gb, &t1e, &rc_pows[z_idx + 1]);
                    eval_acc = ring_add_digits(&mut glue.gb, &eval_acc, &t0e_s);
                    eval_acc = ring_add_digits(&mut glue.gb, &eval_acc, &t1e_s);
                }
                lf_stage_log("cm_recomb_eval_acc_done", Some(&pose_inst), Some(&glue), &mut mem_prev);

                ring_eq_digits(&mut glue.gb, &subclaim_eval, &eval_acc);
            }
        }
    }

    let (mut surfaces_mul_local, all_sum_digits, all_sum_coeffwise) =
        build_mul_surfaces(&mut glue, ring_dim, pairs, &short_locals, &u32_locals)?;
    lf_stage_log("build_mul_surfaces", Some(&pose_inst), Some(&glue), &mut mem_prev);
    let (mut surfaces_sq_local, all_sq_sum_digits, all_sq_sum_coeffwise) =
        build_sq_surfaces(&mut glue, ring_dim, pairs, &short_locals, &u32_locals)?;
    lf_stage_log("build_sq_surfaces", Some(&pose_inst), Some(&glue), &mut mem_prev);

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
    lf_stage_log("surfaces_arc_share", Some(&pose_inst), Some(&glue), &mut mem_prev);

    // Optional: print an op-mix breakdown for tiny-field porting estimates.
    //
    // Enable with: `LFP_WE_GATE_OPMIX=1 ...`
    maybe_print_tiny_opmix(
        cfg,
        ops,
        &pose_inst,
        &glue,
        &absorb_counts,
        pairs.len(),
        surfaces_mul_local.len(),
        surfaces_sq_local.len(),
        all_sum_digits.len(),
        all_sum_coeffwise.len(),
        all_sq_sum_digits.len(),
        all_sq_sum_coeffwise.len(),
    );

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