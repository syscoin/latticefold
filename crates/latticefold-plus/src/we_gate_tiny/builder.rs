use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::transcript::DEFAULT_REJECTION_TRIES;

use rayon::join;
use rayon::prelude::*;

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
    BoundedU32ChallengeWiring, GoldilocksChallengeWiring, ShortChallengeWiring, TinyCoinOpWiring,
};
use super::coins::{sample_goldilocks_coin_unrolled_rejection_8_digits, GoldilocksRejectionCoinWiring};
use super::digits::{
    rebalance_prod12_to_prod13, rebalance_prod21_to_prod22, scale_short_coeffs_by_digits18,
    scale_short_coeffs_by_digits9, sum_bal16_vectors_fixed_len, sum_product_digits_bal16,
    sum_product_digits_bal16_22, sum_products13_coeffwise_fixed_len, sum_products22_coeffwise_fixed_len,
    Bal16Checked,
};
use super::goldilocks::{
    goldilocks_u64_enforce_lt_p_from_byte_vars,
    reduce_u64_mod_goldilocks_from_byte_vars,
    GoldilocksScalar,
};
use super::gadgets::{alloc_byte, decompose_existing_byte_var_to_bits};
use super::params::DIGITS_PER_TRY;
use super::poseidon::poseidon_f257_arithmetize;
use super::surfaces::{CmDigitMulSqSurfaceWiring, CmDigitMulSurfaceWiring};

use super::cm_math::{
    alloc_const_goldilocks_u64,
    eq_eval_goldilocks_digits, eval_t_z_optimized_ring_digits_pair, goldilocks_bytes_to_digits, goldilocks_pow_table_digits,
    goldilocks_digits_to_bytes_canonical,
    goldilocks_add_mod_p_digits, goldilocks_mul_const_mod_p_digits, goldilocks_mul_mod_p_digits, goldilocks_sub_mod_p_digits,
    ct_psi_mul_ring_digits_d64,
    ring_eval_at_scalar_digits,
    ring_add_digits, ring_bytes_to_digits, ring_eq_digits,
    ring_const_coeff_digits,
    ring_scale_digits,
    tensor_goldilocks_ringconst_digits, tensor_goldilocks_scalars_digits, RingBytes, RingDigits,
};
use super::op_counts::{tiny_cm_counts_reset, tiny_cm_counts_take};

#[derive(Clone, Debug)]
struct DcomEvalDigits {
    a: Vec<GoldilocksScalar>,
    b: Vec<RingDigits>,
    c: Vec<RingDigits>,
    v: Vec<GoldilocksScalar>,
}

/// CM verifier shared precomputations built once in the **base** glue module.
///
/// The full WE gate computes these once and reuses them across the two CM sumchecks (`which=0,1`).
/// The tiny gate uses two separate CM glue modules for parallelism; to avoid duplicating heavy
/// negacyclic ring-muls, we compute shared tables + `u[l][ni]` once in the base module and let
/// each CM module import them.
#[derive(Clone, Debug)]
struct CmSharedPrecompBase {
    // tensor(c0) / tensor(c1) as ring-constant tables (length kappa)
    tensor_c0_ring: Vec<RingDigits>,
    tensor_c1_ring: Vec<RingDigits>,
    // flattened s_prime (length k*d), each entry is a ring element
    s_prime_flat_ring: Vec<RingDigits>,
    // dpp = dp^i as ring-constant elements (length ell)
    dpp_ring: Vec<RingDigits>,
    // recovered SetChk verifier point r (length nvars_setchk)
    r_point_digits: Vec<GoldilocksScalar>,
    // u[l][ni] (length L × (1+mlen))
    u: Vec<Vec<RingDigits>>,
}

/// Extra (non-transcript) witness values needed to make the tiny gate satisfiable for a **real** proof.
///
/// This covers objects that the real verifier checks algebraically but does not absorb (or does not fully absorb)
/// into the transcript in the tiny gate arithmetization, e.g.:
/// - `dcom.evals[*].b` and `dcom.evals[*].v`
/// - decomp proof pieces and LinB2X surfaces
///
/// Coefficients are Goldilocks field elements, stored as canonical `u64` reps mod `p`.
#[derive(Clone, Debug)]
pub(crate) struct TinyExtraWitness {
    pub(crate) dcom_eval_b: Vec<Vec<Vec<u64>>>, // [l][eval_len][ring_dim]
    pub(crate) dcom_eval_v: Vec<Vec<u64>>,      // [l][ring_dim]

    // DecompProof
    pub(crate) decomp_c0: Vec<Vec<u64>>, // [kappa][ring_dim]
    pub(crate) decomp_c1: Vec<Vec<u64>>, // [kappa][ring_dim]
    pub(crate) decomp_v0a: Vec<Vec<u64>>, // [vlen][ring_dim]
    pub(crate) decomp_v0b: Vec<Vec<u64>>, // [vlen][ring_dim]
    pub(crate) decomp_v1a: Vec<Vec<u64>>, // [vlen][ring_dim]
    pub(crate) decomp_v1b: Vec<Vec<u64>>, // [vlen][ring_dim]

    // LinB2X
    pub(crate) linb2x_cm_g: Vec<Vec<u64>>, // [kappa][ring_dim]
    pub(crate) linb2x_vo_a: Vec<Vec<u64>>, // [vlen][ring_dim]
    pub(crate) linb2x_vo_b: Vec<Vec<u64>>, // [vlen][ring_dim]
}

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
    pose_asg: Arc<Vec<F257>>,
    local_map: BTreeMap<usize, usize>,
    // Cache for imported base-glue vars so we don't allocate redundant copies.
    // Key: base-glue var index, Value: local var index in this module.
    base_map: BTreeMap<usize, usize>,
    // Extra "glue" equalities between this module's vars and the *base glue* module's vars.
    // Each entry is (base_var, local_var) in their respective local index spaces.
    base_eqs: Vec<(usize, usize)>,
}

impl GlueCtx {
    fn new(pose_asg: Arc<Vec<F257>>) -> Self {
        let mut gb = Dr1csBuilder::<F257>::new();
        gb.enforce_var_eq_const(gb.one(), F257::ONE);
        Self {
            gb,
            pose_asg,
            local_map: BTreeMap::new(),
            base_map: BTreeMap::new(),
            base_eqs: Vec::new(),
        }
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

    #[inline]
    fn import_base_var(&mut self, base_asg: &[F257], base_var: usize) -> usize {
        if base_var == 0 {
            return 0;
        }
        if let Some(&lv) = self.base_map.get(&base_var) {
            return lv;
        }
        let lv = self.gb.new_var(base_asg[base_var]);
        self.base_map.insert(base_var, lv);
        self.base_eqs.push((base_var, lv));
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

fn collect_nonreabsorb_absorbs_before_squeeze_field_op(
    ops: &[PoseidonTraceOp<F257>],
    pose_wiring: &PoseidonDr1csWiring,
    stop_before_squeeze_field_op_idx: usize,
) -> Result<Vec<(usize, usize)>, String> {
    let mut absorb_idx = 0usize;
    let mut squeeze_field_op_idx = 0usize;
    let mut payload: Vec<(usize, usize)> = Vec::new();
    let mut last_squeeze_len: Option<usize> = None;
    let mut last_squeeze_is_get_challenge_try = false;

    for op in ops {
        match op {
            PoseidonTraceOp::SqueezeField(v) => {
                if squeeze_field_op_idx >= stop_before_squeeze_field_op_idx {
                    break;
                }
                squeeze_field_op_idx += 1;
                last_squeeze_len = Some(v.len());
                last_squeeze_is_get_challenge_try = v.len() == DIGITS_PER_TRY;
            }
            PoseidonTraceOp::Absorb(_v) => {
                let (ab_start, ab_len) = *pose_wiring
                    .absorb_ranges
                    .get(absorb_idx)
                    .ok_or("tiny gate: pose_wiring.absorb_ranges oob (prefix absorbs)")?;
                absorb_idx += 1;
                let is_reabsorb = last_squeeze_is_get_challenge_try && last_squeeze_len == Some(ab_len);
                last_squeeze_len = None;
                last_squeeze_is_get_challenge_try = false;
                if !is_reabsorb {
                    payload.push((ab_start, ab_len));
                }
            }
            PoseidonTraceOp::SqueezeBytes { .. } => {
                last_squeeze_len = None;
                last_squeeze_is_get_challenge_try = false;
            }
        }
    }
    Ok(payload)
}

#[inline]
fn infer_ring_elem_bytes_from_wiring(ring_dim: usize, pose_wiring: &PoseidonDr1csWiring) -> Result<usize, String> {
    if ring_dim == 0 {
        return Ok(0);
    }
    let mut ring_elem_bytes: Option<usize> = None;
    for &(_st, ln) in &pose_wiring.absorb_ranges {
        if ln % ring_dim == 0 && ln > ring_dim {
            ring_elem_bytes = Some(match ring_elem_bytes {
                None => ln,
                Some(cur) => cur.min(ln),
            });
        }
    }
    ring_elem_bytes.ok_or_else(|| "tiny gate: could not infer ring_elem_bytes".to_string())
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

fn build_u32_and_goldilocks_blocks(
    glue: &mut GlueCtx,
    pose_wiring: &PoseidonDr1csWiring,
    u32_starts: &[usize],
) -> Result<(Vec<BoundedU32ChallengeWiring>, Vec<GoldilocksChallengeWiring>), String> {
    let tries: usize = DEFAULT_REJECTION_TRIES;
    let mut u32_out: Vec<BoundedU32ChallengeWiring> = Vec::with_capacity(u32_starts.len());
    let mut goldilocks_out: Vec<GoldilocksChallengeWiring> = Vec::with_capacity(u32_starts.len());
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
            u64_bytes[i] = digit_to_byte_var(&mut glue.gb, u_digits_local[i]);
        }
        let (q_bit, goldilocks_limbs) = reduce_u64_mod_goldilocks_from_byte_vars::<F257>(&mut glue.gb, &u64_bytes);
        let res257 = res257_from_u64_bytes_le(&mut glue.gb, &u64_bytes);

        u32_out.push(BoundedU32ChallengeWiring {
            digit_vars: u_digits_local.to_vec(),
            byte_vars: u_bytes,
            limbs: u_limbs,
            bal16_digits: u_bal16,
            bal16_sq_digits: u_bal16_sq,
        });
        goldilocks_out.push(GoldilocksChallengeWiring {
            digit_vars: u_digits_local.to_vec(),
            byte_vars: u64_bytes,
            q_bit,
            limbs: goldilocks_limbs,
            res257,
        });
    }
    Ok((u32_out, goldilocks_out))
}

fn build_goldilocks_rejection_coins(
    glue: &mut GlueCtx,
    pose_wiring: &PoseidonDr1csWiring,
    goldilocks_ranges: &[(usize, usize)],
) -> Result<Vec<GoldilocksRejectionCoinWiring>, String> {
    let tries: usize = DEFAULT_REJECTION_TRIES;
    if !goldilocks_ranges.is_empty() && (goldilocks_ranges.len() % tries != 0) {
        return Err(format!(
            "goldilocks_squeeze_ops length {} not divisible by tries={}",
            goldilocks_ranges.len(),
            tries
        ));
    }
    let n_coins = if goldilocks_ranges.is_empty() { 0 } else { goldilocks_ranges.len() / tries };
    let mut out: Vec<GoldilocksRejectionCoinWiring> = Vec::with_capacity(n_coins);
    for coin_idx in 0..n_coins {
        let mut digit_vars: Vec<usize> = Vec::with_capacity(tries * DIGITS_PER_TRY);
        for t in 0..tries {
            let (start, len) = goldilocks_ranges[coin_idx * tries + t];
            if len != DIGITS_PER_TRY {
                return Err(format!(
                    "goldilocks squeeze len mismatch (got {len}, expected {DIGITS_PER_TRY})"
                ));
            }
            for gv in &pose_wiring.squeeze_field_vars[start..start + len] {
                digit_vars.push(glue.copy_digit(*gv));
            }
        }
        let (coin_local, found_local) =
            sample_goldilocks_coin_unrolled_rejection_8_digits(&mut glue.gb, &digit_vars, tries);
        glue.gb.enforce_var_eq_const(found_local, F257::ONE);
        out.push(GoldilocksRejectionCoinWiring {
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
    fn goldilocks_bytes_from_u32_le_bytes(gb: &mut Dr1csBuilder<F257>, u32_le: &[usize; 4]) -> [usize; 8] {
        let mut out = [0usize; 8];
        out[0..4].copy_from_slice(u32_le);
        for i in 4..8 {
            out[i] = alloc_const_byte(gb, 0u8);
        }
        goldilocks_u64_enforce_lt_p_from_byte_vars::<F257>(gb, &out);
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
                            // `coeff_bytes` bytes (little-endian, canonical for Goldilocks base field).
                            if coeff_bytes != 8 {
                                return Err(format!(
                                    "tiny gate: expected Goldilocks base-field coeff_bytes=8 for CM verifier, got {coeff_bytes}"
                                ));
                            }

                            let mut coeffs: Vec<[usize; 8]> = vec![[0usize; 8]; ring_dim];
                            for coeff in 0..ring_dim {
                                let coeff_start = blk_start + coeff * coeff_bytes;
                                for i in 0..8 {
                                    let gv = pose_wiring.absorb_vars[coeff_start + i];
                                    let lv = glue.copy_digit(gv);
                                    coeffs[coeff][i] = lv;
                                }
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
                            // into the Goldilocks base field. Represent them as canonical Goldilocks bytes by padding the
                            // 4-byte u32 with 4 zero bytes (so value < 2^32 < p).
                            let mut c0_bytes: Vec<[usize; 8]> = Vec::with_capacity(lg);
                            let mut c1_bytes: Vec<[usize; 8]> = Vec::with_capacity(lg);
                            for i in 0..lg {
                                c0_bytes.push(goldilocks_bytes_from_u32_le_bytes(
                                    &mut glue.gb,
                                    &u32_locals[c0_start + i].byte_vars,
                                ));
                            }
                            for i in 0..lg {
                                c1_bytes.push(goldilocks_bytes_from_u32_le_bytes(
                                    &mut glue.gb,
                                    &u32_locals[c1_start + i].byte_vars,
                                ));
                            }

                            // Move to digits as early as possible:
                            // - c0/c1 from u32s → canonical bytes → digits
                            // - comh ring elements bytes → digits
                            // Then do all math in digit domain, and only convert back to bytes
                            // for the exported coefficient-0 surfaces.

                            let c0_digits: Vec<GoldilocksScalar> =
                                c0_bytes.iter().copied().map(|b| goldilocks_bytes_to_digits(&mut glue.gb, b)).collect();
                            let c1_digits: Vec<GoldilocksScalar> =
                                c1_bytes.iter().copied().map(|b| goldilocks_bytes_to_digits(&mut glue.gb, b)).collect();
                            let tensor_c0 = tensor_goldilocks_scalars_digits(&mut glue.gb, &c0_digits);
                            let tensor_c1 = tensor_goldilocks_scalars_digits(&mut glue.gb, &c1_digits);

                            // Convert all `comh[l][j]` to digit-encoded rings once.
                            let mut comh_all_coeff_digits: Vec<RingDigits> = Vec::with_capacity(comh_all_coeff_bytes.len());
                            for ch in &comh_all_coeff_bytes {
                                comh_all_coeff_digits.push(ring_bytes_to_digits(&mut glue.gb, ch));
                            }

                            let mut tcch0_ring: Vec<RingDigits> = Vec::with_capacity(l_instances);
                            let mut tcch1_ring: Vec<RingDigits> = Vec::with_capacity(l_instances);
                            for l in 0..l_instances {
                                let base = l * kappa;
                                let mut acc0 = super::cm_math::ring_zero_digits(&mut glue.gb, ring_dim);
                                let mut acc1 = super::cm_math::ring_zero_digits(&mut glue.gb, ring_dim);
                                for j in 0..kappa {
                                    let ch = &comh_all_coeff_digits[base + j];
                                    let m0 = ring_scale_digits(&mut glue.gb, ch, &tensor_c0[j]);
                                    let m1 = ring_scale_digits(&mut glue.gb, ch, &tensor_c1[j]);
                                    acc0 = ring_add_digits(&mut glue.gb, &acc0, &m0);
                                    acc1 = ring_add_digits(&mut glue.gb, &acc1, &m1);
                                }
                                tcch0_ring.push(acc0);
                                tcch1_ring.push(acc1);
                            }

                            // Export only coefficient-0 as canonical bytes.
                            tcch0_local.reserve(l_instances);
                            tcch1_local.reserve(l_instances);
                            for l in 0..l_instances {
                                tcch0_local.push(goldilocks_digits_to_bytes_canonical(&mut glue.gb, &tcch0_ring[l][0]));
                                tcch1_local.push(goldilocks_digits_to_bytes_canonical(&mut glue.gb, &tcch1_ring[l][0]));
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
    u32_locals: &[BoundedU32ChallengeWiring],
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
    // absorbed non-reabsorb payloads are canonical Goldilocks bytes at IO boundaries.
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

    // We also bind the per-round “explicit absorb of r_i” to the same u32 coins that were sampled
    // by `get_challenge()`. This mirrors the real verifier transcript schedule.
    let cm_u32_start = cm_u32_start_idx(wiring);
    let kappa = params.kappa as usize;
    let log_kappa = ark_std::log2(kappa.next_power_of_two()) as usize;
    let nvars_cm = params.nvars_cm as usize;

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

        // Rounds: 3 ring absorbs (msg evals), then one 8-byte scalar absorb of sampled r_i.
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
            // Explicit absorb of sampled r_i.
            let (st, ln) = *payload_after_short
                .get(cur)
                .ok_or("tiny gate: payload_after_short too short (sc marker)")?;
            cur += 1;
            if ln != 8 {
                return Err("tiny gate: expected 8-byte absorb for sumcheck marker".to_string());
            }
            // Bind absorbed bytes = u32 coin bytes (low 4) and zero padding (high 4).
            let u32_idx = cm_u32_start + 2 * log_kappa + which * (1 + nvars_cm) + 1 + _round;
            if u32_idx >= u32_locals.len() {
                return Err("tiny gate: u32_locals too short for CM sumcheck r_i binding".to_string());
            }
            let u = &u32_locals[u32_idx];
            let z = glue.gb.new_var(F257::ZERO);
            glue.gb.enforce_var_eq_const(z, F257::ZERO);
            for i in 0..4 {
                let gv = pose_wiring.absorb_vars[st + i];
                let lv = glue.copy_digit(gv);
                glue.gb.enforce_lc_times_one_eq_const(vec![(F257::ONE, lv), (-F257::ONE, u.byte_vars[i])]);
                if glue.gb.assignment[lv] != glue.gb.assignment[u.byte_vars[i]] {
                    return Err(format!(
                        "tiny gate: CM sumcheck r_i byte mismatch (which={which} round={_round} byte={i}): absorb={:?} u32_byte={:?}",
                        glue.gb.assignment[lv],
                        glue.gb.assignment[u.byte_vars[i]]
                    ));
                }
            }
            for i in 4..8 {
                let gv = pose_wiring.absorb_vars[st + i];
                let lv = glue.copy_digit(gv);
                glue.gb.enforce_lc_times_one_eq_const(vec![(F257::ONE, lv), (-F257::ONE, z)]);
                if glue.gb.assignment[lv] != F257::ZERO {
                    return Err(format!(
                        "tiny gate: CM sumcheck r_i high byte nonzero (which={which} round={_round} byte={i}): absorb={:?}",
                        glue.gb.assignment[lv]
                    ));
                }
            }
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
fn goldilocks_bytes_from_u32_le_bytes(gb: &mut Dr1csBuilder<F257>, u32_le: &[usize; 4]) -> [usize; 8] {
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

fn enforce_absorbed_u64_const(glue: &mut GlueCtx, pose_wiring: &PoseidonDr1csWiring, ab_start: usize, val: u64) {
    let bs = val.to_le_bytes();
    for i in 0..8 {
        let gv = pose_wiring.absorb_vars[ab_start + i];
        let lv = glue.copy_digit(gv);
        glue.gb.enforce_var_eq_const(lv, F257::from(bs[i] as u64));
    }
}

#[inline]
fn alloc_witness_goldilocks_u64_bytes(gb: &mut Dr1csBuilder<F257>, v: u64) -> [usize; 8] {
    let bs = v.to_le_bytes();
    let mut out = [0usize; 8];
    for i in 0..8 {
        out[i] = alloc_byte::<F257>(gb, bs[i]).byte;
    }
    goldilocks_u64_enforce_lt_p_from_byte_vars::<F257>(gb, &out);
    out
}

#[inline]
fn alloc_witness_goldilocks_u64_digits(gb: &mut Dr1csBuilder<F257>, v: u64) -> GoldilocksScalar {
    let bytes = alloc_witness_goldilocks_u64_bytes(gb, v);
    goldilocks_bytes_to_digits(gb, bytes)
}

fn count_nonreabsorb_absorbs_before_first_squeeze_field_op(
    ops: &[PoseidonTraceOp<F257>],
    pose_wiring: &PoseidonDr1csWiring,
) -> Result<usize, String> {
    let mut absorb_idx = 0usize;
    for op in ops {
        match op {
            PoseidonTraceOp::SqueezeField(_v) => break,
            PoseidonTraceOp::Absorb(_v) => {
                let (_st, ln) = *pose_wiring
                    .absorb_ranges
                    .get(absorb_idx)
                    .ok_or("tiny gate: pose_wiring.absorb_ranges oob (pre-squeeze)")?;
                absorb_idx += 1;
                if ln != 8 {
                    return Err("tiny gate: expected 8-byte scalar absorbs before first squeeze".to_string());
                }
            }
            PoseidonTraceOp::SqueezeBytes { .. } => {}
        }
    }
    Ok(absorb_idx)
}

struct SurfaceLocal<const RAW: usize, const NORM: usize> {
    short_block_idx: usize,
    u32_idx: usize,
    products_raw: Vec<[usize; RAW]>,
    products_norm: Vec<[usize; NORM]>,
    sum_digits: Bal16Checked,
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
    sum_product_digits_fn: fn(&mut Dr1csBuilder<F257>, &[[usize; NORM]], usize) -> Bal16Checked,
    sum_products_coeffwise_fn: fn(&mut Dr1csBuilder<F257>, &[&[[usize; NORM]]], usize, usize) -> Vec<Bal16Checked>,
    u_digits_len: usize,
) -> Result<(Vec<SurfaceLocal<RAW, NORM>>, Bal16Checked, Vec<Bal16Checked>), String> {
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
        let refs: Vec<&Bal16Checked> = surfaces.iter().map(|s| &s.sum_digits).collect();
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
    fn sum_digits(gb: &mut Dr1csBuilder<F257>, p: &[[usize; 13]], target_len: usize) -> Bal16Checked {
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
            sum_digits: s.sum_digits.into_vec(),
            sum_all_pairs_digits: Arc::new(Vec::new()),
            sum_all_pairs_coeffwise: Arc::new(Vec::new()),
        })
        .collect();
    let all_sum_digits = all_sum_digits.into_vec();
    let all_sum_coeffwise = all_sum_coeffwise.into_iter().map(|v| v.into_vec()).collect();
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
    fn sum_digits(gb: &mut Dr1csBuilder<F257>, p: &[[usize; 22]], target_len: usize) -> Bal16Checked {
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
            sum_digits: s.sum_digits.into_vec(),
            sum_all_pairs_digits: Arc::new(Vec::new()),
            sum_all_pairs_coeffwise: Arc::new(Vec::new()),
        })
        .collect();
    let all_sum_digits = all_sum_digits.into_vec();
    let all_sum_coeffwise = all_sum_coeffwise.into_iter().map(|v| v.into_vec()).collect();
    Ok((surfaces, all_sum_digits, all_sum_coeffwise))
}

fn enforce_fiat_shamir_reabsorb_semantics(
    inst: &mut SparseDr1csInstance<F257>,
    ops: &[PoseidonTraceOp<F257>],
    pose_wiring: &PoseidonDr1csWiring,
) -> Result<(), String> {
    // Collect all reabsorb equalities first, then bulk-append constraints/terms.
    // This keeps finalize-time overhead low for large instances.
    let mut eqs: Vec<(usize, usize)> = Vec::new(); // (v_ab, v_sq)

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
                        eqs.push((v_ab, v_sq));
                    }
                }
            }
            PoseidonTraceOp::SqueezeBytes { .. } => {}
        }
    }

    if eqs.is_empty() {
        return Ok(());
    }

    // Each equality adds:
    // - A: 2 terms (v_ab - v_sq)
    // - B: 1 term (ONE * var0)
    // - C: 1 term (ZERO * var0)
    // - 1 constraint referencing these ranges
    let n = eqs.len();
    inst.a_terms.reserve(n * 2);
    inst.b_terms.reserve(n);
    inst.c_terms.reserve(n);
    inst.constraints.reserve(n);

    let mut a0 = inst.a_terms.len();
    let mut b0 = inst.b_terms.len();
    let mut c0 = inst.c_terms.len();
    for (v_ab, v_sq) in eqs {
        inst.a_terms.push((F257::ONE, v_ab));
        inst.a_terms.push((-F257::ONE, v_sq));
        inst.b_terms.push((F257::ONE, 0));
        inst.c_terms.push((F257::ZERO, 0));
        inst.constraints.push(Constraint {
            a: a0..(a0 + 2),
            b: b0..(b0 + 1),
            c: c0..(c0 + 1),
        });
        a0 += 2;
        b0 += 1;
        c0 += 1;
    }
    Ok(())
}


fn collect_nonreabsorb_absorb_ranges(
    ops: &[PoseidonTraceOp<F257>],
    pose_wiring: &PoseidonDr1csWiring,
) -> Result<Vec<(usize, usize)>, String> {
    let mut out: Vec<(usize, usize)> = Vec::new();
    let mut absorb_idx = 0usize;
    let mut expect_reabsorb = false;
    for (op_i, op) in ops.iter().enumerate() {
        match op {
            PoseidonTraceOp::SqueezeField(v) => {
                expect_reabsorb = v.len() == DIGITS_PER_TRY
                    && matches!(ops.get(op_i + 1), Some(PoseidonTraceOp::Absorb(a)) if a.len() == DIGITS_PER_TRY);
            }
            PoseidonTraceOp::Absorb(_v) => {
                let (ab_start, ab_len) = *pose_wiring
                    .absorb_ranges
                    .get(absorb_idx)
                    .ok_or("pose_wiring.absorb_ranges oob (collect canonical goldilocks)")?;
                absorb_idx += 1;
                let is_reabsorb = expect_reabsorb;
                expect_reabsorb = false;
                if is_reabsorb {
                    continue;
                }
                if (ab_len % 8) != 0 {
                    continue;
                }
                out.push((ab_start, ab_len));
            }
            PoseidonTraceOp::SqueezeBytes { .. } => {
                expect_reabsorb = false;
            }
        }
    }
    Ok(out)
}

fn build_canonical_goldilocks_glue_for_ranges(
    pose_asg: Arc<Vec<F257>>,
    pose_wiring: &PoseidonDr1csWiring,
    ranges: &[(usize, usize)],
) -> Result<GlueCtx, String> {
    let mut glue = GlueCtx::new(pose_asg);
    for &(ab_start, ab_len) in ranges {
        let n_elems = ab_len / 8;
        for e in 0..n_elems {
            let mut bytes = [0usize; 8];
            for j in 0..8 {
                let gv = pose_wiring.absorb_vars[ab_start + e * 8 + j];
                let lv = glue.copy_digit(gv);
                let _ = decompose_existing_byte_var_to_bits::<F257>(&mut glue.gb, lv);
                bytes[j] = lv;
            }
            goldilocks_u64_enforce_lt_p_from_byte_vars::<F257>(&mut glue.gb, &bytes);
        }
    }
    Ok(glue)
}

#[allow(clippy::too_many_arguments)]
fn finalize(
    pose_inst: SparseDr1csInstance<F257>,
    pose_wiring: PoseidonDr1csWiring,
    ops: &[PoseidonTraceOp<F257>],
    glue: GlueCtx,
    extra_glues: Vec<GlueCtx>,
    short_locals: Vec<ShortChallengeWiring>,
    u32_locals: Vec<BoundedU32ChallengeWiring>,
    goldilocks_locals: Vec<GoldilocksChallengeWiring>,
    goldilocks_rejection_locals: Vec<GoldilocksRejectionCoinWiring>,
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
        Vec<GoldilocksChallengeWiring>,
        Vec<GoldilocksRejectionCoinWiring>,
        Vec<[usize; 8]>,
        Vec<[usize; 8]>,
        Vec<CmDigitMulSurfaceWiring>,
        Vec<CmDigitMulSqSurfaceWiring>,
        PoseidonDr1csWiring,
    ),
    String,
> {
    // Use the same single switch as op-mix printing.
    let timing_merge = tiny_opmix_on();

    // Convert glue builders to instances; keep per-part local maps so we can add explicit equality constraints.
    let GlueCtx { gb, pose_asg, local_map: base_local_map, base_eqs: base_base_eqs, .. } = glue;
    let (base_inst, base_asg) = gb.into_instance();
    debug_assert!(base_base_eqs.is_empty(), "base glue should not contain base_eqs");

    let mut extra_insts: Vec<(SparseDr1csInstance<F257>, Vec<F257>)> = Vec::with_capacity(extra_glues.len());
    let mut extra_maps: Vec<BTreeMap<usize, usize>> = Vec::with_capacity(extra_glues.len());
    let mut extra_base_eqs: Vec<Vec<(usize, usize)>> = Vec::with_capacity(extra_glues.len());
    for g in extra_glues {
        let GlueCtx { gb, pose_asg: _pa, local_map, base_eqs, .. } = g;
        let (inst, asg) = gb.into_instance();
        extra_insts.push((inst, asg));
        extra_maps.push(local_map);
        extra_base_eqs.push(base_eqs);
    }

    // Recover the owned pose assignment (avoid cloning).
    let pose_asg = Arc::try_unwrap(pose_asg)
        .map_err(|_| "tiny gate: internal error: pose assignment still shared at finalize")?;

    // Save lightweight stats for optional op-mix reporting (before moving instances into merge).
    let c_pose = pose_inst.constraints.len();
    let mut c_glue = base_inst.constraints.len();
    for (inst, _asg) in &extra_insts {
        c_glue = c_glue.saturating_add(inst.constraints.len());
    }
    let mut glue_eq_constraints = base_local_map.len();
    for m in &extra_maps {
        glue_eq_constraints = glue_eq_constraints.saturating_add(m.len());
    }
    for v in &extra_base_eqs {
        glue_eq_constraints = glue_eq_constraints.saturating_add(v.len());
    }

    // Compute part offsets in merged space (excluding var0).
    // Part 0: poseidon, part 1: base glue, parts 2..: extra glue modules.
    let mut offsets: Vec<usize> = Vec::with_capacity(2 + extra_insts.len());
    let mut cur = 0usize;
    offsets.push(cur);
    cur += pose_asg.len().saturating_sub(1);
    offsets.push(cur);
    cur += base_asg.len().saturating_sub(1);
    for (_inst, asg) in &extra_insts {
        offsets.push(cur);
        cur += asg.len().saturating_sub(1);
    }
    let remap = |part: usize, local: usize, offsets: &[usize]| -> usize {
        if local == 0 { 0 } else { local + offsets[part] }
    };

    let mut parts: Vec<(SparseDr1csInstance<F257>, Vec<F257>)> = Vec::with_capacity(2 + extra_insts.len());
    parts.push((pose_inst, pose_asg));
    parts.push((base_inst, base_asg));
    parts.extend(extra_insts);
    let t_merge = Instant::now();
    let (mut inst, asg) = merge_sparse_dr1cs_share_one::<F257>(parts)
        .map_err(|e| format!("merge poseidon+tiny-glue parts failed: {e}"))?;
    let dt_merge = t_merge.elapsed();

    let c_after_merge = inst.constraints.len();
    let t_reabsorb = Instant::now();
    enforce_fiat_shamir_reabsorb_semantics(&mut inst, ops, &pose_wiring)?;
    let dt_reabsorb = t_reabsorb.elapsed();
    let c_after_reabsorb = inst.constraints.len();

    // Add explicit equality constraints: pose var == each module's local copy.
    let t_eqs = Instant::now();
    let mut eq_pairs: Vec<(usize, usize)> = Vec::with_capacity(glue_eq_constraints);
    for (&gv, &lv) in base_local_map.iter() {
        eq_pairs.push((remap(0, gv, &offsets), remap(1, lv, &offsets)));
    }
    for (i, m) in extra_maps.iter().enumerate() {
        let part = 2 + i;
        for (&gv, &lv) in m.iter() {
            eq_pairs.push((remap(0, gv, &offsets), remap(part, lv, &offsets)));
        }
    }
    for (i, v) in extra_base_eqs.iter().enumerate() {
        let part = 2 + i;
        for &(base_var, local_var) in v {
            eq_pairs.push((remap(1, base_var, &offsets), remap(part, local_var, &offsets)));
        }
    }
    debug_assert_eq!(eq_pairs.len(), glue_eq_constraints);

    // Bulk-append equality constraints (faster than calling `enforce_var_eq` in a tight loop).
    // Each equality adds 1 constraint with:
    // - A: (x - y)
    // - B: (ONE * var0)
    // - C: (ZERO * var0)
    inst.a_terms.reserve(eq_pairs.len() * 2);
    inst.b_terms.reserve(eq_pairs.len());
    inst.c_terms.reserve(eq_pairs.len());
    inst.constraints.reserve(eq_pairs.len());
    let mut a0 = inst.a_terms.len();
    let mut b0 = inst.b_terms.len();
    let mut c0 = inst.c_terms.len();
    for (x, y) in eq_pairs {
        inst.a_terms.push((F257::ONE, x));
        inst.a_terms.push((-F257::ONE, y));
        inst.b_terms.push((F257::ONE, 0));
        inst.c_terms.push((F257::ZERO, 0));
        inst.constraints.push(Constraint { a: a0..(a0 + 2), b: b0..(b0 + 1), c: c0..(c0 + 1) });
        a0 += 2;
        b0 += 1;
        c0 += 1;
    }
    let c_after_glue_eq = inst.constraints.len();
    let dt_eqs = t_eqs.elapsed();

    if timing_merge {
        eprintln!(
            "tiny_gate: finalize timing: merge={:?} reabsorb={:?} glue_eqs={:?} parts={} constraints_after_merge={} constraints_after_reabsorb={} constraints_after_eqs={}",
            dt_merge,
            dt_reabsorb,
            dt_eqs,
            offsets.len(),
            c_after_merge,
            c_after_reabsorb,
            c_after_glue_eq,
        );
    }

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

    // All exported wiring is from the base glue module (part 1).
    let to_glue_global = |glue_local: usize| -> usize { remap(1, glue_local, &offsets) };

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
    let goldilocks_out = goldilocks_locals
        .into_iter()
        .map(|w| GoldilocksChallengeWiring {
            digit_vars: w.digit_vars.into_iter().map(to_glue_global).collect(),
            byte_vars: w.byte_vars.map(to_glue_global),
            q_bit: to_glue_global(w.q_bit),
            limbs: w.limbs.map(to_glue_global),
            res257: to_glue_global(w.res257),
        })
        .collect::<Vec<_>>();
    let goldilocks_rejection_out = goldilocks_rejection_locals
        .into_iter()
        .map(|w| GoldilocksRejectionCoinWiring {
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
        goldilocks_out,
        goldilocks_rejection_out,
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

fn build_cm_glue_for_which(
    _cfg: Option<&PoseidonConfig<F257>>,
    _ops: &[PoseidonTraceOp<F257>],
    pose_wiring: &PoseidonDr1csWiring,
    ring_dim: usize,
    params: &WeParams,
    wiring: &TinyCoinOpWiring,
    l_instances_expected: usize,
    comh_absorbs: &[(usize, usize)],
    sc_msg_absorbs: &[(usize, usize)],
    eval_absorbs: &[(usize, usize)],
    // Shared SetChk/Dcom values (allocated in the base glue module) that the CM verifier math needs.
    // These are passed as base-glue variable indices; this CM module will `import_base_var` them.
    setchk_out_e_vars_base: Option<Arc<Vec<Vec<Vec<RingDigits>>>>>,
    dcom_evals_base: Option<Arc<Vec<DcomEvalDigits>>>,
    // Shared CM precomputations built once in the base glue module (u, s_prime_flat, etc).
    cm_shared_base: Option<Arc<CmSharedPrecompBase>>,
    which: usize,
    pose_asg: Arc<Vec<F257>>,
    base_asg: &[F257],
    short_locals: &[ShortChallengeWiring],
    u32_locals: &[BoundedU32ChallengeWiring],
) -> Result<GlueCtx, String> {
    let mut glue = GlueCtx::new(pose_asg);
    if ring_dim == 0 || l_instances_expected == 0 {
        return Ok(glue);
    }

    // The CM segment begins at the last short squeeze, but the absorb ranges were already parsed
    // by the base glue module (and statement-bound there). Here we just consume the ranges.
    let cm_u32_start = cm_u32_start_idx(wiring);
    let kappa = params.kappa as usize;
    let log_kappa = ark_std::log2(kappa.next_power_of_two()) as usize;
    let nvars_cm = params.nvars_cm as usize;
    let k_decomp = params.k as usize;
    let ell = params.l as usize;

    // CM challenges after absorb_comh: c0/c1, then for each sumcheck: rc, r_sc[0..nvars_cm]
    let c0_u32 = &u32_locals[cm_u32_start..cm_u32_start + log_kappa];
    let c1_u32 = &u32_locals[cm_u32_start + log_kappa..cm_u32_start + 2 * log_kappa];

    // Precompute the ring-constant tables needed for t(z) evaluation.
    //
    // In the current tiny gate design, these are computed once in the base glue module and
    // imported by each per-`which` CM module. Keep a fallback path for legacy call sites.
    let mut tensor_c0_ring: Option<Vec<RingDigits>> = None;
    let mut tensor_c1_ring: Option<Vec<RingDigits>> = None;
    let mut s_prime_flat_ring: Option<Vec<RingDigits>> = None;
    let mut dpp_ring: Option<Vec<RingDigits>> = None;
    let mut r_point_digits: Option<Vec<GoldilocksScalar>> = None;
    if cm_shared_base.is_none()
        && ring_dim == 64
        && kappa.is_power_of_two()
        && (k_decomp * ring_dim).is_power_of_two()
        && ell.is_power_of_two()
    {
        // c0/c1 as Goldilocks scalars (digit encoding), then tensor-expand.
        let c0_digits: Vec<_> = c0_u32
            .iter()
            .map(|u| {
                let b0 = glue.import_base_var(base_asg, u.byte_vars[0]);
                let b1 = glue.import_base_var(base_asg, u.byte_vars[1]);
                let b2 = glue.import_base_var(base_asg, u.byte_vars[2]);
                let b3 = glue.import_base_var(base_asg, u.byte_vars[3]);
                let bytes = goldilocks_bytes_from_u32_le_bytes(&mut glue.gb, &[b0, b1, b2, b3]);
                goldilocks_bytes_to_digits(&mut glue.gb, bytes)
            })
            .collect();
        let c1_digits: Vec<_> = c1_u32
            .iter()
            .map(|u| {
                let b0 = glue.import_base_var(base_asg, u.byte_vars[0]);
                let b1 = glue.import_base_var(base_asg, u.byte_vars[1]);
                let b2 = glue.import_base_var(base_asg, u.byte_vars[2]);
                let b3 = glue.import_base_var(base_asg, u.byte_vars[3]);
                let bytes = goldilocks_bytes_from_u32_le_bytes(&mut glue.gb, &[b0, b1, b2, b3]);
                goldilocks_bytes_to_digits(&mut glue.gb, bytes)
            })
            .collect();
        let t0 = tensor_goldilocks_scalars_digits(&mut glue.gb, &c0_digits);
        let t1 = tensor_goldilocks_scalars_digits(&mut glue.gb, &c1_digits);
        tensor_c0_ring = Some(tensor_goldilocks_ringconst_digits(&mut glue.gb, &t0, ring_dim));
        tensor_c1_ring = Some(tensor_goldilocks_ringconst_digits(&mut glue.gb, &t1, ring_dim));

        // dpp: dp^i as constant-coeff ring elements.
        let dp = (ring_dim / 2) as u64;
        let p = crate::we_goldilocks_poseidon_f257::GOLDILOCKS_P;
        let mut acc: u64 = 1;
        let mut dpp: Vec<RingDigits> = Vec::with_capacity(ell);
        for _ in 0..ell {
            let s_bytes = super::cm_math::alloc_const_goldilocks_u64(&mut glue.gb, acc);
            let s = goldilocks_bytes_to_digits(&mut glue.gb, s_bytes);
            dpp.push(super::cm_math::ring_const_coeff_digits(&mut glue.gb, &s, ring_dim));
            acc = ((acc as u128) * (dp as u128) % (p as u128)) as u64;
        }
        dpp_ring = Some(dpp);

        // s_prime_flat: k*d short challenges, each is a ring element with centered coeff bytes.
        let need_sprime = k_decomp
            .checked_mul(ring_dim)
            .ok_or_else(|| "tiny gate: k*ring_dim overflow (s_prime_flat)".to_string())?;
        if short_locals.len() < 3 + need_sprime {
            return Err("tiny gate: short_locals too short for s_prime_flat".to_string());
        }
        let z = glue.gb.new_var(F257::ZERO);
        glue.gb.enforce_var_eq_const(z, F257::ZERO);
        let c128 = super::cm_math::alloc_const_goldilocks_u64(&mut glue.gb, 128u64);
        let c128_d = goldilocks_bytes_to_digits(&mut glue.gb, c128);
        let mut sflat: Vec<RingDigits> = Vec::with_capacity(need_sprime);
        for blk in 0..need_sprime {
            let sb = &short_locals[3 + blk];
            if sb.byte_vars.len() != ring_dim {
                return Err("tiny gate: short byte_vars len mismatch (s_prime_flat)".to_string());
            }
            let mut re: RingDigits = Vec::with_capacity(ring_dim);
            for &bv in &sb.byte_vars {
                let bv_local = glue.import_base_var(base_asg, bv);
                let mut bbytes = [0usize; 8];
                bbytes[0] = bv_local;
                for i in 1..8 {
                    bbytes[i] = z;
                }
                // Centered coefficient = (byte - 128) mod p, in digit domain.
                let bd = goldilocks_bytes_to_digits(&mut glue.gb, bbytes);
                let centered = super::cm_math::goldilocks_sub_mod_p_digits(&mut glue.gb, &bd, &c128_d);
                re.push(centered);
            }
            sflat.push(re);
        }
        s_prime_flat_ring = Some(sflat);

        // Recover the SetChk verifier point `r` used in eq(r, ro).
        let nvars_lin = params.nvars_setchk as usize;
        let n_lin_proofs = l_instances_expected;
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
        let mut rdig: Vec<GoldilocksScalar> = Vec::with_capacity(nvars_lin);
        for u in &u32_locals[r_start..r_end] {
            let b0 = glue.import_base_var(base_asg, u.byte_vars[0]);
            let b1 = glue.import_base_var(base_asg, u.byte_vars[1]);
            let b2 = glue.import_base_var(base_asg, u.byte_vars[2]);
            let b3 = glue.import_base_var(base_asg, u.byte_vars[3]);
            let bytes = goldilocks_bytes_from_u32_le_bytes(&mut glue.gb, &[b0, b1, b2, b3]);
            rdig.push(goldilocks_bytes_to_digits(&mut glue.gb, bytes));
        }
        r_point_digits = Some(rdig);
    }

    // If shared precomputations were provided by the base glue module, import them now.
    if let Some(shared) = cm_shared_base.as_ref() {
        #[inline]
        fn import_scalar(glue: &mut GlueCtx, base_asg: &[F257], s: &GoldilocksScalar) -> GoldilocksScalar {
            core::array::from_fn(|i| glue.import_base_var(base_asg, s[i]))
        }
        #[inline]
        fn import_ring(glue: &mut GlueCtx, base_asg: &[F257], r: &RingDigits) -> RingDigits {
            r.iter().map(|s| import_scalar(glue, base_asg, s)).collect()
        }

        tensor_c0_ring = Some(shared.tensor_c0_ring.iter().map(|r| import_ring(&mut glue, base_asg, r)).collect());
        tensor_c1_ring = Some(shared.tensor_c1_ring.iter().map(|r| import_ring(&mut glue, base_asg, r)).collect());
        s_prime_flat_ring = Some(shared.s_prime_flat_ring.iter().map(|r| import_ring(&mut glue, base_asg, r)).collect());
        dpp_ring = Some(shared.dpp_ring.iter().map(|r| import_ring(&mut glue, base_asg, r)).collect());
        r_point_digits = Some(shared.r_point_digits.iter().map(|s| import_scalar(&mut glue, base_asg, s)).collect());
    }

    // Select the u32 window for this sumcheck.
    let mut u32_idx = cm_u32_start + 2 * log_kappa + which * (1 + nvars_cm);
    let u = &u32_locals[u32_idx];
    let b0 = glue.import_base_var(base_asg, u.byte_vars[0]);
    let b1 = glue.import_base_var(base_asg, u.byte_vars[1]);
    let b2 = glue.import_base_var(base_asg, u.byte_vars[2]);
    let b3 = glue.import_base_var(base_asg, u.byte_vars[3]);
    let rc_bytes = goldilocks_bytes_from_u32_le_bytes(&mut glue.gb, &[b0, b1, b2, b3]);
    u32_idx += 1;
    let mut rs: Vec<[usize; 8]> = Vec::with_capacity(nvars_cm);
    for _ in 0..nvars_cm {
        let u = &u32_locals[u32_idx];
        let b0 = glue.import_base_var(base_asg, u.byte_vars[0]);
        let b1 = glue.import_base_var(base_asg, u.byte_vars[1]);
        let b2 = glue.import_base_var(base_asg, u.byte_vars[2]);
        let b3 = glue.import_base_var(base_asg, u.byte_vars[3]);
        rs.push(goldilocks_bytes_from_u32_le_bytes(&mut glue.gb, &[b0, b1, b2, b3]));
        u32_idx += 1;
    }

    // Parse sumcheck msg absorbs as ring-bytes (transcript IO), then immediately convert to digits.
    // This keeps the heavy sumcheck arithmetic in the digit domain.
    let mut msgs_digits: Vec<[RingDigits; 3]> = Vec::with_capacity(nvars_cm);
    if sc_msg_absorbs.len() != nvars_cm * 3 {
        return Err("tiny gate: sumcheck msg absorb count mismatch".to_string());
    }
    for round in 0..nvars_cm {
        let (s0, l0) = sc_msg_absorbs[round * 3 + 0];
        let (s1, l1) = sc_msg_absorbs[round * 3 + 1];
        let (s2, l2) = sc_msg_absorbs[round * 3 + 2];
        let e0b = parse_ring_elem_absorb_as_ringbytes(&mut glue, pose_wiring, ring_dim, s0, l0)?;
        let e1b = parse_ring_elem_absorb_as_ringbytes(&mut glue, pose_wiring, ring_dim, s1, l1)?;
        let e2b = parse_ring_elem_absorb_as_ringbytes(&mut glue, pose_wiring, ring_dim, s2, l2)?;
        let e0 = ring_bytes_to_digits(&mut glue.gb, &e0b);
        let e1 = ring_bytes_to_digits(&mut glue.gb, &e1b);
        let e2 = ring_bytes_to_digits(&mut glue.gb, &e2b);
        msgs_digits.push([e0, e1, e2]);
    }

    let rs_digits: Vec<GoldilocksScalar> =
        rs.iter().copied().map(|b| goldilocks_bytes_to_digits(&mut glue.gb, b)).collect();

    // --------------------------------------------------------------------
    // CM claimed_sum wiring (matches `we_gate_arith.rs::cm_verifier_math_dr1cs`)
    // --------------------------------------------------------------------
    //
    // If we have the shared SetChk/Dcom values, compute the true claimed_sum and feed it into the
    // degree-2 sumcheck verifier. Otherwise, fall back to a placeholder to keep "shape-only" runs
    // satisfiable (but underconstrained).
    let mut claimed_sum_opt: Option<RingDigits> = None;
    if let (Some(out_e_base), Some(dcom_evals_base)) = (setchk_out_e_vars_base.as_ref(), dcom_evals_base.as_ref()) {
        if let (Some(t0_ring), Some(t1_ring), Some(sp_ring), Some(_dpp), Some(_rpt)) = (
            tensor_c0_ring.as_ref(),
            tensor_c1_ring.as_ref(),
            s_prime_flat_ring.as_ref(),
            dpp_ring.as_ref(),
            r_point_digits.as_ref(),
        ) {
            // Only implement the pow2 fast path (same assumptions as `eval_t_z_optimized_ring_digits_pair`).
            let out_e_blocks = 1usize + (params.mlen as usize);
            let rows_per_l = 1usize + (params.mlen as usize);
            if dcom_evals_base.len() == l_instances_expected
                && out_e_base.len() == out_e_blocks
                && ring_dim == 64
            {
                #[inline]
                fn import_scalar(glue: &mut GlueCtx, base_asg: &[F257], s: &GoldilocksScalar) -> GoldilocksScalar {
                    core::array::from_fn(|i| glue.import_base_var(base_asg, s[i]))
                }
                #[inline]
                fn import_ring(glue: &mut GlueCtx, base_asg: &[F257], r: &RingDigits) -> RingDigits {
                    r.iter().map(|s| import_scalar(glue, base_asg, s)).collect()
                }

                // Parse `comh` absorbs and compute tcch0/tcch1:
                // tcch{0,1}[l] = Σ_j comh[l][j] * tensor_c{0,1}[j] (scalar scaling).
                let mut tcch0: Vec<RingDigits> = Vec::with_capacity(l_instances_expected);
                let mut tcch1: Vec<RingDigits> = Vec::with_capacity(l_instances_expected);
                if comh_absorbs.len() == l_instances_expected * kappa {
                    // tensor_c{0,1} are constant-coeff rings; coefficient-0 holds the scalar digits.
                    let tensor_c0_scalars: Vec<GoldilocksScalar> =
                        t0_ring.iter().map(|r| r[0]).collect();
                    let tensor_c1_scalars: Vec<GoldilocksScalar> =
                        t1_ring.iter().map(|r| r[0]).collect();

                    let mut idx = 0usize;
                    for _l in 0..l_instances_expected {
                        let mut acc0 = super::cm_math::ring_zero_digits(&mut glue.gb, ring_dim);
                        let mut acc1 = super::cm_math::ring_zero_digits(&mut glue.gb, ring_dim);
                        for j in 0..kappa {
                            let (st, ln) = comh_absorbs[idx];
                            idx += 1;
                            let rb = parse_ring_elem_absorb_as_ringbytes(&mut glue, pose_wiring, ring_dim, st, ln)?;
                            let comh_j = ring_bytes_to_digits(&mut glue.gb, &rb);
                            let s0 = tensor_c0_scalars[j];
                            let s1 = tensor_c1_scalars[j];
                            let t0j = ring_scale_digits(&mut glue.gb, &comh_j, &s0);
                            let t1j = ring_scale_digits(&mut glue.gb, &comh_j, &s1);
                            acc0 = ring_add_digits(&mut glue.gb, &acc0, &t0j);
                            acc1 = ring_add_digits(&mut glue.gb, &acc1, &t1j);
                        }
                        tcch0.push(acc0);
                        tcch1.push(acc1);
                    }
                }

                if tcch0.len() == l_instances_expected && tcch1.len() == l_instances_expected {
                    // rc powers for claimed_sum
                    let z_idx = l_instances_expected * (4 + 4 * (params.mlen as usize));
                    let max_pow = z_idx + 1;
                    let rc_d = goldilocks_bytes_to_digits(&mut glue.gb, rc_bytes);
                    let rc_pows = goldilocks_pow_table_digits(&mut glue.gb, &rc_d, max_pow);

                    // Compute u[l][ni] = Σ_{blk,col} out.e[ni][l*k+blk][col] * s_prime_flat[blk*d + col]
                    // (negacyclic ringmul), then build claimed_sum over all l.
                    let mut claimed_sum = super::cm_math::ring_zero_digits(&mut glue.gb, ring_dim);
                    for (l, ev) in dcom_evals_base.iter().enumerate() {
                        let l_idx = l * (4 + 4 * (params.mlen as usize));

                        // Import eval vectors from base glue vars.
                        let eval_a: Vec<GoldilocksScalar> = ev.a.iter().map(|s| import_scalar(&mut glue, base_asg, s)).collect();
                        let eval_b: Vec<RingDigits> = ev.b.iter().map(|r| import_ring(&mut glue, base_asg, r)).collect();
                        let eval_c: Vec<RingDigits> = ev.c.iter().map(|r| import_ring(&mut glue, base_asg, r)).collect();

                        // u_l for each ni (length rows_per_l).
                        //
                        // IMPORTANT: In the full WE gate, `u` is computed once and reused across the two CM
                        // sumchecks. The tiny gate has two per-`which` CM modules; to avoid duplicating the
                        // heavy negacyclic ring-muls, we import a base-glue precomputation when available.
                        let u_l: Vec<RingDigits> = if let Some(shared) = cm_shared_base.as_ref() {
                            let rows = shared
                                .u
                                .get(l)
                                .ok_or_else(|| "tiny gate: cm_shared_base.u missing l row".to_string())?;
                            if rows.len() != rows_per_l {
                                return Err("tiny gate: cm_shared_base.u rows_per_l mismatch".to_string());
                            }
                            rows.iter().map(|r| import_ring(&mut glue, base_asg, r)).collect()
                        } else {
                            let mut u_l: Vec<RingDigits> = Vec::with_capacity(rows_per_l);
                            for ni in 0..rows_per_l {
                                // NOTE: this is a huge independent workload (k*d negacyclic ring-muls per row).
                                // Build the ring-muls as IR shards in parallel, then lower sequentially in batches.
                                use super::cm_ir::{
                                    lower_ir_into_builder, ring_mul_negacyclic_ntt_goldilocks_d64_ir, IrBuilder,
                                    VarRef as IrVarRef,
                                };

                                #[inline]
                                fn ringdigits64_to_ir(a: &RingDigits) -> Result<[[IrVarRef; 17]; 64], String> {
                                    if a.len() != 64 {
                                        return Err("tiny gate: expected ring_dim=64 for IR ring-mul".to_string());
                                    }
                                    Ok(core::array::from_fn(|i| core::array::from_fn(|j| IrVarRef::Base(a[i][j]))))
                                }

                                #[inline]
                                fn map_ring_out(out_ir: &[[IrVarRef; 17]; 64], lowered: &super::cm_ir::LoweredIr) -> RingDigits {
                                    let out: [GoldilocksScalar; 64] = core::array::from_fn(|i| {
                                        core::array::from_fn(|j| lowered.map_var(out_ir[i][j]))
                                    });
                                    out.into_iter().collect()
                                }

                                let mut acc = super::cm_math::ring_zero_digits(&mut glue.gb, ring_dim);

                                // Collect/import the `uij` rings we need for this (l, ni) slice.
                                // We keep `sij` by index into `sp_ring` to avoid cloning it.
                                let mut terms: Vec<(RingDigits, usize)> = Vec::with_capacity(k_decomp * ring_dim);
                                for blk in 0..k_decomp {
                                    let idx = l
                                        .checked_mul(k_decomp)
                                        .and_then(|x| x.checked_add(blk))
                                        .ok_or_else(|| "tiny gate: u index overflow (cm)".to_string())?;
                                    if idx >= out_e_base[ni].len() {
                                        return Err("tiny gate: out.e too short for CM u computation".to_string());
                                    }
                                    for col in 0..ring_dim {
                                        let uij = &out_e_base[ni][idx][col];
                                        // Import uij ring from base glue.
                                        let uij_local = import_ring(&mut glue, base_asg, uij);
                                        let sp_idx = blk * ring_dim + col;
                                        terms.push((uij_local, sp_idx));
                                    }
                                }

                                // Batch size for IR shards (controls peak memory).
                                let batch_size: usize = 64;

                                for chunk in terms.chunks(batch_size) {
                                    // Build IR fragments in parallel (no access to glue.gb mutably).
                                    let base_asg_ir: &[F257] = &glue.gb.assignment;
                                    let frags: Vec<(_, [[IrVarRef; 17]; 64])> = chunk
                                        .par_iter()
                                        .map(|(uij_local, sp_idx)| -> Result<_, String> {
                                            let u_ir = ringdigits64_to_ir(uij_local)?;
                                            let s_ir = ringdigits64_to_ir(&sp_ring[*sp_idx])?;
                                            let mut ib = IrBuilder::new(base_asg_ir);
                                            let out_ir = ring_mul_negacyclic_ntt_goldilocks_d64_ir(&mut ib, &u_ir, &s_ir);
                                            // Keep op-mix accounting consistent even when ring-muls are built via IR shards.
                                            super::op_counts::tiny_cm_bump(|c| c.ring_mul_negacyclic += 1);
                                            Ok((ib.ir, out_ir))
                                        })
                                        .collect::<Result<Vec<_>, _>>()?;

                                    // Lower sequentially and accumulate.
                                    for (ir, out_ir) in frags {
                                        let lowered = lower_ir_into_builder(&mut glue.gb, ir);
                                        let prod = map_ring_out(&out_ir, &lowered);
                                        acc = ring_add_digits(&mut glue.gb, &acc, &prod);
                                    }
                                }

                                u_l.push(acc);
                            }
                            u_l
                        };

                        // a0 term (scalar -> const-coeff ring)
                        let a0pow = goldilocks_mul_mod_p_digits(&mut glue.gb, &eval_a[0], &rc_pows[l_idx]);
                        let a0r = ring_const_coeff_digits(&mut glue.gb, &a0pow, ring_dim);
                        claimed_sum = ring_add_digits(&mut glue.gb, &claimed_sum, &a0r);

                        // b0/c0/u0 terms (ring scaled)
                        let b0 = ring_scale_digits(&mut glue.gb, &eval_b[0], &rc_pows[l_idx + 1]);
                        let c0 = ring_scale_digits(&mut glue.gb, &eval_c[0], &rc_pows[l_idx + 2]);
                        let u0 = ring_scale_digits(&mut glue.gb, &u_l[0], &rc_pows[l_idx + 3]);
                        claimed_sum = ring_add_digits(&mut glue.gb, &claimed_sum, &b0);
                        claimed_sum = ring_add_digits(&mut glue.gb, &claimed_sum, &c0);
                        claimed_sum = ring_add_digits(&mut glue.gb, &claimed_sum, &u0);

                        for i in 0..(params.mlen as usize) {
                            let idx = l_idx + 4 + i * 4;
                            let aipow = goldilocks_mul_mod_p_digits(&mut glue.gb, &eval_a[1 + i], &rc_pows[idx]);
                            let air = ring_const_coeff_digits(&mut glue.gb, &aipow, ring_dim);
                            claimed_sum = ring_add_digits(&mut glue.gb, &claimed_sum, &air);

                            let bi = ring_scale_digits(&mut glue.gb, &eval_b[1 + i], &rc_pows[idx + 1]);
                            let ci = ring_scale_digits(&mut glue.gb, &eval_c[1 + i], &rc_pows[idx + 2]);
                            let ui = ring_scale_digits(&mut glue.gb, &u_l[1 + i], &rc_pows[idx + 3]);
                            claimed_sum = ring_add_digits(&mut glue.gb, &claimed_sum, &bi);
                            claimed_sum = ring_add_digits(&mut glue.gb, &claimed_sum, &ci);
                            claimed_sum = ring_add_digits(&mut glue.gb, &claimed_sum, &ui);
                        }

                        // tcch0/tcch1 terms
                        let tc0 = ring_scale_digits(&mut glue.gb, &tcch0[l], &rc_pows[z_idx]);
                        let tc1 = ring_scale_digits(&mut glue.gb, &tcch1[l], &rc_pows[z_idx + 1]);
                        claimed_sum = ring_add_digits(&mut glue.gb, &claimed_sum, &tc0);
                        claimed_sum = ring_add_digits(&mut glue.gb, &claimed_sum, &tc1);
                    }

                    claimed_sum_opt = Some(claimed_sum);
                }
            }
        }
    }

    let claimed_sum = claimed_sum_opt.unwrap_or_else(|| super::cm_math::ring_zero_digits(&mut glue.gb, ring_dim));
    let subclaim_eval = super::cm_math::sumcheck_verify_degree2_ring_digits(&mut glue.gb, claimed_sum, &msgs_digits, &rs_digits)?;

    // Eval table absorbs sanity.
    let rows_total = l_instances_expected * (1 + params.mlen as usize);
    if eval_absorbs.len() != rows_total * 4 {
        return Err("tiny gate: eval absorb count mismatch".to_string());
    }

    // Recombination check (requires the pow2 regime + recovered setchk r-point).
    if let (Some(tc0_ring), Some(tc1_ring), Some(sp_ring), Some(dpp), Some(rpt)) = (
        tensor_c0_ring.as_ref(),
        tensor_c1_ring.as_ref(),
        s_prime_flat_ring.as_ref(),
        dpp_ring.as_ref(),
        r_point_digits.as_ref(),
    ) {
        // rc powers (need up to z_idx+1).
        let z_idx = l_instances_expected * (4 + 4 * (params.mlen as usize));
        let max_pow = z_idx + 1;
        let rc_d = goldilocks_bytes_to_digits(&mut glue.gb, rc_bytes);
        let rc_pows = goldilocks_pow_table_digits(&mut glue.gb, &rc_d, max_pow);

        // eq(r, ro) where r is the transcript-derived SetChk point (recovered above).
        let eq = eq_eval_goldilocks_digits(&mut glue.gb, rpt, &rs_digits)?;

        // Evaluate t0(ro), t1(ro).
        let (t0, t1) = eval_t_z_optimized_ring_digits_pair(
            &mut glue.gb,
            tc0_ring,
            tc1_ring,
            sp_ring,
            dpp,
            ring_dim,
            &rs_digits,
        )?;

        // Stream recombination.
        let rows_per_l = 1 + params.mlen as usize;
        let mut eval_acc = super::cm_math::ring_zero_digits(&mut glue.gb, ring_dim);
        let mut e00s: Vec<RingDigits> = Vec::with_capacity(l_instances_expected);
        for l in 0..l_instances_expected {
            let l_idx = l * (4 + 4 * (params.mlen as usize));
            let mut inner = super::cm_math::ring_zero_digits(&mut glue.gb, ring_dim);
            let mut e00_opt: Option<RingDigits> = None;

            // First row (tau, m_tau, f, h)
            let flat0 = l * rows_per_l + 0;
            for j in 0..4 {
                let (st, ln) = eval_absorbs[flat0 * 4 + j];
                let rb = parse_ring_elem_absorb_as_ringbytes(&mut glue, pose_wiring, ring_dim, st, ln)?;
                let rd = ring_bytes_to_digits(&mut glue.gb, &rb);
                if j == 0 {
                    e00_opt = Some(rd.clone());
                }
                let t = ring_scale_digits(&mut glue.gb, &rd, &rc_pows[l_idx + j]);
                inner = ring_add_digits(&mut glue.gb, &inner, &t);
            }

            // M rows
            for i in 0..(params.mlen as usize) {
                let flat = l * rows_per_l + 1 + i;
                let idx = l_idx + 4 + i * 4;
                for j in 0..4 {
                    let (st, ln) = eval_absorbs[flat * 4 + j];
                    let rb = parse_ring_elem_absorb_as_ringbytes(&mut glue, pose_wiring, ring_dim, st, ln)?;
                    let rd = ring_bytes_to_digits(&mut glue.gb, &rb);
                    let t = ring_scale_digits(&mut glue.gb, &rd, &rc_pows[idx + j]);
                    inner = ring_add_digits(&mut glue.gb, &inner, &t);
                }
            }

            let eq_inner = ring_scale_digits(&mut glue.gb, &inner, &eq);
            eval_acc = ring_add_digits(&mut glue.gb, &eval_acc, &eq_inner);

            // Save e00 for the t(z) terms; we'll build all the expensive ring-muls in parallel as IR shards.
            let e00 = e00_opt.ok_or_else(|| "tiny gate: missing e00 in recombination".to_string())?;
            e00s.push(e00);
        }

        // t(z) terms: for each l, add t0(ro)*e00(l)*rc^z + t1(ro)*e00(l)*rc^{z+1}.
        //
        // These ring-muls are independent across l, so we build them as IR fragments in parallel,
        // then lower sequentially into this module's builder.
        {
            use super::cm_ir::{lower_ir_into_builder, ring_mul_negacyclic_ntt_goldilocks_d64_ir, IrBuilder, VarRef as IrVarRef};

            #[inline]
            fn ringdigits64_to_ir(a: &RingDigits) -> Result<[[IrVarRef; 17]; 64], String> {
                if a.len() != 64 {
                    return Err("tiny gate: expected ring_dim=64 for IR ring-mul".to_string());
                }
                Ok(core::array::from_fn(|i| core::array::from_fn(|j| IrVarRef::Base(a[i][j]))))
            }

            #[inline]
            fn map_ring_out(out_ir: &[[IrVarRef; 17]; 64], lowered: &super::cm_ir::LoweredIr) -> RingDigits {
                let out: [GoldilocksScalar; 64] = core::array::from_fn(|i| {
                    core::array::from_fn(|j| lowered.map_var(out_ir[i][j]))
                });
                out.into_iter().collect()
            }

            let t0_ir = ringdigits64_to_ir(&t0)?;
            let t1_ir = ringdigits64_to_ir(&t1)?;
            let base_asg: &[F257] = &glue.gb.assignment;

            let frags: Vec<(_, [[IrVarRef; 17]; 64], [[IrVarRef; 17]; 64])> = e00s
                .par_iter()
                .map(|e00| -> Result<_, String> {
                    let e00_ir = ringdigits64_to_ir(e00)?;
                    let mut ib = IrBuilder::new(base_asg);
                    let out0 = ring_mul_negacyclic_ntt_goldilocks_d64_ir(&mut ib, &t0_ir, &e00_ir);
                    let out1 = ring_mul_negacyclic_ntt_goldilocks_d64_ir(&mut ib, &t1_ir, &e00_ir);
                    // Keep op-mix accounting consistent even when ring-muls are built via IR shards.
                    super::op_counts::tiny_cm_bump(|c| c.ring_mul_negacyclic += 2);
                    Ok((ib.ir, out0, out1))
                })
                .collect::<Result<Vec<_>, _>>()?;

            for (ir, out0_ir, out1_ir) in frags {
                let lowered = lower_ir_into_builder(&mut glue.gb, ir);
                let t0e = map_ring_out(&out0_ir, &lowered);
                let t1e = map_ring_out(&out1_ir, &lowered);

                let t0e_s = ring_scale_digits(&mut glue.gb, &t0e, &rc_pows[z_idx]);
                let t1e_s = ring_scale_digits(&mut glue.gb, &t1e, &rc_pows[z_idx + 1]);
                eval_acc = ring_add_digits(&mut glue.gb, &eval_acc, &t0e_s);
                eval_acc = ring_add_digits(&mut glue.gb, &eval_acc, &t1e_s);
            }
        }

        ring_eq_digits(&mut glue.gb, &subclaim_eval, &eval_acc);
    }

    Ok(glue)
}

pub(super) fn build(
    cfg: Option<&PoseidonConfig<F257>>,
    ops: &[PoseidonTraceOp<F257>],
    ring_dim: usize,
    params: &WeParams,
    wiring: &TinyCoinOpWiring,
    pairs: &[(usize, usize)],
    extra_witness: Option<&TinyExtraWitness>,
) -> Result<
    (
        SparseDr1csInstance<F257>,
        Vec<F257>,
        Vec<ShortChallengeWiring>,
        Vec<BoundedU32ChallengeWiring>,
        Vec<GoldilocksChallengeWiring>,
        Vec<GoldilocksRejectionCoinWiring>,
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
    let goldilocks_ranges = squeeze_field_ranges_by_op_index(&pose_wiring.squeeze_field_ranges, &wiring.goldilocks_squeeze_ops)?;
    validate_pairs(pairs, short_ranges.len(), u32_ranges.len())?;
    validate_params_and_short_schedule(ring_dim, params, short_ranges.len())?;

    let pose_asg = Arc::new(pose_asg);
    let mut glue = GlueCtx::new(pose_asg.clone());
    lf_stage_log("glue_init", Some(&pose_inst), Some(&glue), &mut mem_prev);

    // Bind all proof/statement payload absorbs that encode base-field elements as canonical 8-byte scalars.
    // (Skip fiat–shamir reabsorbs, which are F257 digits and may contain 256.)
    //
    // This is a huge independent workload; build it as many glue modules in parallel and merge.
    let canonical_ranges = collect_nonreabsorb_absorb_ranges(ops, &pose_wiring)?;
    let n_threads = rayon::current_num_threads().max(1);
    // Over-decompose to improve load-balance; avoid too many tiny tasks / too many merge parts.
    // Override with `LFP_TINY_CANON_CHUNKS`.
    let n_chunks = std::env::var("LFP_TINY_CANON_CHUNKS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|&n| n > 0)
        .unwrap_or_else(|| {
            // Heuristic: pick a chunk count that keeps per-chunk work non-trivial to avoid
            // creating hundreds of tiny glue modules (which makes `merge_sparse_dr1cs_share_one`
            // dominate wall-time).
            //
            // For small traces, aim for ~256 absorb-ranges per chunk; for huge traces, cap to
            // ~2x threads (and 256 overall) for good parallelism without exploding merge parts.
            let target_chunk_ranges: usize = 256;
            let by_work = (canonical_ranges.len() + target_chunk_ranges - 1) / target_chunk_ranges;
            by_work.max(1).min((n_threads * 2).min(256).max(1))
        })
        .min(canonical_ranges.len().max(1));
    let chunk_size = (canonical_ranges.len() + n_chunks - 1) / n_chunks;
    if lf_mem_on() || tiny_opmix_on() {
        eprintln!(
            "[tiny_gate/par] canonical_goldilocks ranges={} threads={} chunks={} chunk_size={}",
            canonical_ranges.len(),
            n_threads,
            n_chunks,
            chunk_size.max(1)
        );
    }
    let canonical_glues: Vec<GlueCtx> = if canonical_ranges.is_empty() {
        Vec::new()
    } else {
        canonical_ranges
            .chunks(chunk_size.max(1))
            .collect::<Vec<_>>()
            .into_par_iter()
            .map(|chunk| build_canonical_goldilocks_glue_for_ranges(pose_asg.clone(), &pose_wiring, chunk))
            .collect::<Result<Vec<_>, _>>()?
    };
    // `lf_stage_log` expects a single glue ctx; log the base ctx here and rely on merge stats for totals.
    lf_stage_log(
        "enforce_nonreabsorb_absorbs_are_canonical_goldilocks(par)",
        Some(&pose_inst),
        Some(&glue),
        &mut mem_prev,
    );

    validate_cm_u32_schedule(params, wiring)?;
    let (n_comh_ring_elems, coeff_bytes) = count_comh_ring_elements(ops, &pose_wiring, ring_dim, wiring)?;
    if ring_dim > 0 && n_comh_ring_elems > 0 && coeff_bytes != 8 {
        return Err(format!(
            "tiny gate: expected Goldilocks base-field coeff_bytes=8, got {coeff_bytes}"
        ));
    }
    let kappa = params.kappa as usize;
    if kappa > 0 && (n_comh_ring_elems % kappa) != 0 {
        return Err("tiny gate: comh ring element count not divisible by kappa".to_string());
    }
    let l_instances_expected = if kappa == 0 { 0 } else { n_comh_ring_elems / kappa };

    let short_locals = build_short_blocks(&mut glue, &pose_wiring, ring_dim, &short_ranges)?;
    lf_stage_log("build_short_blocks", Some(&pose_inst), Some(&glue), &mut mem_prev);
    let (u32_locals, goldilocks_locals) =
        build_u32_and_goldilocks_blocks(&mut glue, &pose_wiring, &wiring.u32_squeeze_ops)?;
    lf_stage_log("build_u32_and_goldilocks_blocks", Some(&pose_inst), Some(&glue), &mut mem_prev);

    // Values parsed/allocated during SetChk/RgChk that the CM verifier math needs later.
    // Stored as base-glue vars and passed (by Arc) into CM submodules for import/glue.
    let mut setchk_out_e_vars_for_cm: Option<Vec<Vec<Vec<RingDigits>>>> = None;
    let mut dcom_evals_for_cm: Option<Vec<DcomEvalDigits>> = None;

    // ------------------------------------------------------------------------
    // Start arithmetizing Π_lin / SetChk verifier math: SetChk sumcheck (degree-3)
    // ------------------------------------------------------------------------
    //
    // This binds:
    // - the SetChk sumcheck header absorbs (nvars, degree=3)
    // - the per-round prover message ring elements (4 evals)
    // - the per-round verifier challenges r_i (both as absorbed bytes and as u32 coins)
    //
    // Full SetChk recombination + digest binding is wired later; this is the “first swap-in”
    // of the real verifier arithmetic.
    if ring_dim > 0 && !wiring.short_squeeze_ops.is_empty() {
        let first_short_op = *wiring
            .short_squeeze_ops
            .iter()
            .min()
            .expect("non-empty short_squeeze_ops");

        // Collect prefix (pre-CM-short) non-reabsorb absorbs and locate the SetChk sumcheck block.
        let prefix_payload =
            collect_nonreabsorb_absorbs_before_squeeze_field_op(ops, &pose_wiring, first_short_op)?;
        let ring_elem_bytes = infer_ring_elem_bytes_from_wiring(ring_dim, &pose_wiring)?;
        let nvars_setchk = params.nvars_setchk as usize;
        // Deterministic cursor (no searching):
        //
        // Match the LF+ verifier schedule (same assumption as `we_gate_arith.rs`):
        // prefix_absorbs =
        //   [ statement params (10 scalars, 8 bytes each) ]
        //   [ public inputs (public_inputs_len scalars, 8 bytes each) ]
        //   [ dcom commitments: cm_f, C_Mf, cm_mtau (ring elems) ]
        //   [ setchk sumcheck header + rounds ... ]
        //
        // We locate SetChk by a deterministic cursor that accounts for the full prefix schedule
        // (public inputs + Π_lin + Dcom commits) rather than assuming a params block here.
        let n_public_inputs = count_nonreabsorb_absorbs_before_first_squeeze_field_op(ops, &pose_wiring)?;
        if prefix_payload.len() < n_public_inputs {
            return Err("tiny gate: prefix_payload shorter than pre-squeeze public input absorbs".to_string());
        }
        for i in 0..n_public_inputs {
            if prefix_payload[i].1 != 8 {
                return Err("tiny gate: expected 8-byte public input absorbs in prefix".to_string());
            }
        }

        let n_lin_proofs = l_instances_expected;
        let mut cur = n_public_inputs;
        // Π_lin verifier math (degree-3 sumcheck + e*(va*vb-vc) == subclaim), deterministic cursor.
        //
        // NOTE: we only enforce the final ring-mul check when `ring_dim==64`, since the tiny backend
        // currently has an optimized NTT IR gadget only for that regime. For other ring dims we still
        // parse/bind the transcript schedule (sound transcript binding), but skip the ring-mul check.
        {
            use super::cm_ir::{lower_ir_into_builder, ring_mul_negacyclic_ntt_goldilocks_d64_ir, IrBuilder, VarRef as IrVarRef};

            #[inline]
            fn ringdigits64_to_ir(a: &RingDigits) -> Result<[[IrVarRef; 17]; 64], String> {
                if a.len() != 64 {
                    return Err("tiny gate: expected ring_dim=64 for Π_lin ring-mul".to_string());
                }
                Ok(core::array::from_fn(|i| core::array::from_fn(|j| IrVarRef::Base(a[i][j]))))
            }

            #[inline]
            fn map_ring_out(out_ir: &[[IrVarRef; 17]; 64], lowered: &super::cm_ir::LoweredIr) -> RingDigits {
                let out: [GoldilocksScalar; 64] =
                    core::array::from_fn(|i| core::array::from_fn(|j| lowered.map_var(out_ir[i][j])));
                out.into_iter().collect()
            }

            for lp in 0..n_lin_proofs {
                // Header absorbs must equal (nvars, degree=3).
                let (st_nv, ln_nv) = *prefix_payload.get(cur).ok_or("tiny gate: Π_lin header oob")?;
                let (st_deg, ln_deg) = *prefix_payload.get(cur + 1).ok_or("tiny gate: Π_lin header oob")?;
                if ln_nv != 8 || ln_deg != 8 {
                    return Err("tiny gate: Π_lin header absorb len mismatch".to_string());
                }
                enforce_absorbed_u64_const(&mut glue, &pose_wiring, st_nv, nvars_setchk as u64);
                enforce_absorbed_u64_const(&mut glue, &pose_wiring, st_deg, 3u64);
                cur += 2;

                // r_pre (nvars) then r_sc (nvars) from u32 challenge schedule.
                let base = lp
                    .checked_mul(2usize.saturating_mul(nvars_setchk))
                    .ok_or_else(|| "tiny gate: Π_lin u32 base overflow".to_string())?;
                let r_pre_u32 = u32_locals
                    .get(base..base + nvars_setchk)
                    .ok_or_else(|| "tiny gate: Π_lin r_pre u32 slice oob".to_string())?;
                let r_sc_u32 = u32_locals
                    .get(base + nvars_setchk..base + 2 * nvars_setchk)
                    .ok_or_else(|| "tiny gate: Π_lin r_sc u32 slice oob".to_string())?;

                let mut r_pre_digits: Vec<GoldilocksScalar> = Vec::with_capacity(nvars_setchk);
                for u in r_pre_u32 {
                    let b0 = u.byte_vars[0];
                    let b1 = u.byte_vars[1];
                    let b2 = u.byte_vars[2];
                    let b3 = u.byte_vars[3];
                    let bytes = goldilocks_bytes_from_u32_le_bytes(&mut glue.gb, &[b0, b1, b2, b3]);
                    r_pre_digits.push(goldilocks_bytes_to_digits(&mut glue.gb, bytes));
                }

                // Parse sumcheck msgs + bind absorbed r_i to r_sc u32 bytes.
                let mut msgs_digits: Vec<[RingDigits; 4]> = Vec::with_capacity(nvars_setchk);
                let mut rs_digits: Vec<GoldilocksScalar> = Vec::with_capacity(nvars_setchk);
                let z = glue.gb.new_var(F257::ZERO);
                glue.gb.enforce_var_eq_const(z, F257::ZERO);

                for round in 0..nvars_setchk {
                    let (s0, l0) = *prefix_payload.get(cur + 0).ok_or("tiny gate: Π_lin msg oob")?;
                    let (s1, l1) = *prefix_payload.get(cur + 1).ok_or("tiny gate: Π_lin msg oob")?;
                    let (s2, l2) = *prefix_payload.get(cur + 2).ok_or("tiny gate: Π_lin msg oob")?;
                    let (s3, l3) = *prefix_payload.get(cur + 3).ok_or("tiny gate: Π_lin msg oob")?;
                    cur += 4;
                    if l0 != ring_elem_bytes || l1 != ring_elem_bytes || l2 != ring_elem_bytes || l3 != ring_elem_bytes {
                        return Err("tiny gate: Π_lin msg ring absorb len mismatch".to_string());
                    }
                    let e0b = parse_ring_elem_absorb_as_ringbytes(&mut glue, &pose_wiring, ring_dim, s0, l0)?;
                    let e1b = parse_ring_elem_absorb_as_ringbytes(&mut glue, &pose_wiring, ring_dim, s1, l1)?;
                    let e2b = parse_ring_elem_absorb_as_ringbytes(&mut glue, &pose_wiring, ring_dim, s2, l2)?;
                    let e3b = parse_ring_elem_absorb_as_ringbytes(&mut glue, &pose_wiring, ring_dim, s3, l3)?;
                    let e0 = ring_bytes_to_digits(&mut glue.gb, &e0b);
                    let e1 = ring_bytes_to_digits(&mut glue.gb, &e1b);
                    let e2 = ring_bytes_to_digits(&mut glue.gb, &e2b);
                    let e3 = ring_bytes_to_digits(&mut glue.gb, &e3b);
                    msgs_digits.push([e0, e1, e2, e3]);

                    // Explicit absorb of r_i (8 bytes) must equal r_sc_u32 bytes || 0^4.
                    let (rst, rln) = *prefix_payload.get(cur).ok_or("tiny gate: Π_lin r_i absorb oob")?;
                    cur += 1;
                    if rln != 8 {
                        return Err("tiny gate: Π_lin r_i absorb len mismatch".to_string());
                    }
                    let u = &r_sc_u32[round];
                    for i in 0..4 {
                        let gv = pose_wiring.absorb_vars[rst + i];
                        let lv = glue.copy_digit(gv);
                        glue.gb.enforce_lc_times_one_eq_const(vec![(F257::ONE, lv), (-F257::ONE, u.byte_vars[i])]);
                        if glue.gb.assignment[lv] != glue.gb.assignment[u.byte_vars[i]] {
                            return Err(format!(
                                "tiny gate: Π_lin r_i byte mismatch (lp={lp} round={round} byte={i}): absorb={:?} u32_byte={:?}",
                                glue.gb.assignment[lv],
                                glue.gb.assignment[u.byte_vars[i]]
                            ));
                        }
                    }
                    for i in 4..8 {
                        let gv = pose_wiring.absorb_vars[rst + i];
                        let lv = glue.copy_digit(gv);
                        glue.gb.enforce_lc_times_one_eq_const(vec![(F257::ONE, lv), (-F257::ONE, z)]);
                        if glue.gb.assignment[lv] != F257::ZERO {
                            return Err(format!(
                                "tiny gate: Π_lin r_i high byte nonzero (lp={lp} round={round} byte={i}): absorb={:?}",
                                glue.gb.assignment[lv]
                            ));
                        }
                    }
                    let b0 = u.byte_vars[0];
                    let b1 = u.byte_vars[1];
                    let b2 = u.byte_vars[2];
                    let b3 = u.byte_vars[3];
                    let bytes = goldilocks_bytes_from_u32_le_bytes(&mut glue.gb, &[b0, b1, b2, b3]);
                    rs_digits.push(goldilocks_bytes_to_digits(&mut glue.gb, bytes));
                }

                let claimed0 = super::cm_math::ring_zero_digits(&mut glue.gb, ring_dim);
                let subclaim_eval =
                    super::cm_math::sumcheck_verify_degree3_ring_digits(&mut glue.gb, claimed0, &msgs_digits, &rs_digits)?;

                // Tail absorbs: (v,va,vb,vc) as ring elems.
                let mut tail: Vec<RingDigits> = Vec::with_capacity(4);
                for _ in 0..4 {
                    let (st, ln) = *prefix_payload.get(cur).ok_or("tiny gate: Π_lin tail oob")?;
                    cur += 1;
                    if ln != ring_elem_bytes {
                        return Err("tiny gate: Π_lin tail ring absorb len mismatch".to_string());
                    }
                    let rb = parse_ring_elem_absorb_as_ringbytes(&mut glue, &pose_wiring, ring_dim, st, ln)?;
                    tail.push(ring_bytes_to_digits(&mut glue.gb, &rb));
                }
                let va = &tail[1];
                let vb = &tail[2];
                let vc = &tail[3];

                if ring_dim == 64 {
                    // e = eq_eval(r_pre, r_sc)
                    let e = super::cm_math::eq_eval_goldilocks_digits(&mut glue.gb, &r_pre_digits, &rs_digits)?;
                    // lhs = e*(va*vb - vc)
                    // Snapshot assignment to avoid borrow conflicts with lowering.
                    let base_asg: Vec<F257> = glue.gb.assignment.clone();
                    let mut ib = IrBuilder::new(&base_asg);
                    let va_ir = ringdigits64_to_ir(va)?;
                    let vb_ir = ringdigits64_to_ir(vb)?;
                    let prod_ir = ring_mul_negacyclic_ntt_goldilocks_d64_ir(&mut ib, &va_ir, &vb_ir);
                    super::op_counts::tiny_cm_bump(|c| c.ring_mul_negacyclic += 1);
                    let lowered = lower_ir_into_builder(&mut glue.gb, ib.ir);
                    let vab = map_ring_out(&prod_ir, &lowered);
                    let diff = super::cm_math::ring_sub_digits(&mut glue.gb, &vab, vc);
                    let lhs = ring_scale_digits(&mut glue.gb, &diff, &e);
                    ring_eq_digits(&mut glue.gb, &lhs, &subclaim_eval);
                }
            }
        }

        // Skip Dcom commitment absorbs (cm_f, C_Mf, cm_mtau): 3 * L * kappa ring elems.
        let kappa = params.kappa as usize;
        let l_instances = l_instances_expected;
        let n_commit_ring_absorbs = 3usize
            .checked_mul(l_instances)
            .and_then(|x| x.checked_mul(kappa))
            .ok_or_else(|| "tiny gate: commitment absorb count overflow (setchk)".to_string())?;
        for _ in 0..n_commit_ring_absorbs {
            if prefix_payload.get(cur).map(|x| x.1) != Some(ring_elem_bytes) {
                return Err("tiny gate: dcom commitment ring absorb len mismatch".to_string());
            }
            cur += 1;
        }

        // Next in the prefix payload is the SetChk sumcheck header (nvars, degree=3).
        let start = cur;
        if start + 2 > prefix_payload.len() {
            return Err("tiny gate: prefix too short for SetChk header".to_string());
        }

        // Header absorbs: (nvars_setchk, degree=3).
        enforce_absorbed_u64_const(&mut glue, &pose_wiring, prefix_payload[start + 0].0, nvars_setchk as u64);
        enforce_absorbed_u64_const(&mut glue, &pose_wiring, prefix_payload[start + 1].0, 3u64);

        // Recover the SetChk verifier challenge point r from the u32 coin schedule.
        //
        // Mirrors the indexing used by the CM recombination gadget (eq(r, ro)).
        let n_lin_proofs = l_instances_expected;
        let lin_chals = n_lin_proofs
            .checked_mul(2usize.saturating_mul(nvars_setchk))
            .ok_or_else(|| "tiny gate: lin_chals overflow (setchk)".to_string())?;
        let k_decomp = params.k as usize;
        let nclaims = k_decomp
            .checked_add(1)
            .ok_or_else(|| "tiny gate: nclaims overflow (setchk)".to_string())?;
        let out_coin_total = nclaims
            .checked_mul(nvars_setchk + 2)
            .and_then(|x| x.checked_add(if k_decomp > 1 { 1 } else { 0 }))
            .ok_or_else(|| "tiny gate: out_coin_total overflow (setchk)".to_string())?;
        let r_start = lin_chals
            .checked_add(out_coin_total)
            .ok_or_else(|| "tiny gate: r_start overflow (setchk)".to_string())?;
        let r_end = r_start
            .checked_add(nvars_setchk)
            .ok_or_else(|| "tiny gate: r_end overflow (setchk)".to_string())?;
        if u32_locals.len() < r_end {
            return Err("tiny gate: not enough u32 challenges to recover setchk r-point".to_string());
        }

        // Parse the sumcheck prover messages as ring-bytes -> ring-digits.
        // Also bind each absorbed r_i scalar (8 bytes) to the corresponding u32 coin bytes.
        let mut msgs_digits: Vec<[RingDigits; 4]> = Vec::with_capacity(nvars_setchk);
        let mut rs_digits: Vec<GoldilocksScalar> = Vec::with_capacity(nvars_setchk);
        let z = glue.gb.new_var(F257::ZERO);
        glue.gb.enforce_var_eq_const(z, F257::ZERO);

        let mut cur = start + 2;
        for round in 0..nvars_setchk {
            // 4 ring element absorbs (degree-3 evals)
            let (s0, l0) = prefix_payload[cur + 0];
            let (s1, l1) = prefix_payload[cur + 1];
            let (s2, l2) = prefix_payload[cur + 2];
            let (s3, l3) = prefix_payload[cur + 3];
            cur += 4;
            let e0b = parse_ring_elem_absorb_as_ringbytes(&mut glue, &pose_wiring, ring_dim, s0, l0)?;
            let e1b = parse_ring_elem_absorb_as_ringbytes(&mut glue, &pose_wiring, ring_dim, s1, l1)?;
            let e2b = parse_ring_elem_absorb_as_ringbytes(&mut glue, &pose_wiring, ring_dim, s2, l2)?;
            let e3b = parse_ring_elem_absorb_as_ringbytes(&mut glue, &pose_wiring, ring_dim, s3, l3)?;
            let e0 = ring_bytes_to_digits(&mut glue.gb, &e0b);
            let e1 = ring_bytes_to_digits(&mut glue.gb, &e1b);
            let e2 = ring_bytes_to_digits(&mut glue.gb, &e2b);
            let e3 = ring_bytes_to_digits(&mut glue.gb, &e3b);
            msgs_digits.push([e0, e1, e2, e3]);

            // Absorbed r_i scalar (8 bytes) must match the u32 coin bytes (low 4) and zero padding (high 4).
            let (rst, rln) = prefix_payload[cur];
            cur += 1;
            if rln != 8 {
                return Err("tiny gate: expected 8-byte absorb for SetChk r_i".to_string());
            }
            let u = &u32_locals[r_start + round];
            for i in 0..4 {
                let gv = pose_wiring.absorb_vars[rst + i];
                let lv = glue.copy_digit(gv);
                glue.gb.enforce_lc_times_one_eq_const(vec![(F257::ONE, lv), (-F257::ONE, u.byte_vars[i])]);
                if glue.gb.assignment[lv] != glue.gb.assignment[u.byte_vars[i]] {
                    return Err(format!(
                        "tiny gate: SetChk r_i byte mismatch (round={round} byte={i}): absorb={:?} u32_byte={:?}",
                        glue.gb.assignment[lv],
                        glue.gb.assignment[u.byte_vars[i]]
                    ));
                }
            }
            for i in 4..8 {
                let gv = pose_wiring.absorb_vars[rst + i];
                let lv = glue.copy_digit(gv);
                glue.gb.enforce_lc_times_one_eq_const(vec![(F257::ONE, lv), (-F257::ONE, z)]);
                if glue.gb.assignment[lv] != F257::ZERO {
                    return Err(format!(
                        "tiny gate: SetChk r_i high byte nonzero (round={round} byte={i}): absorb={:?}",
                        glue.gb.assignment[lv]
                    ));
                }
            }
            let b0 = u.byte_vars[0];
            let b1 = u.byte_vars[1];
            let b2 = u.byte_vars[2];
            let b3 = u.byte_vars[3];
            let bytes = goldilocks_bytes_from_u32_le_bytes(&mut glue.gb, &[b0, b1, b2, b3]);
            rs_digits.push(goldilocks_bytes_to_digits(&mut glue.gb, bytes));
        }

        let claimed0 = super::cm_math::ring_zero_digits(&mut glue.gb, ring_dim);
        let v_sc = super::cm_math::sumcheck_verify_degree3_ring_digits(&mut glue.gb, claimed0, &msgs_digits, &rs_digits)?;

        // --------------------------------------------------------------------
        // absorb(out.e, out.b): full binding via transcript absorption
        // --------------------------------------------------------------------
        //
        // For the tiny-field gate we bind SetChk outputs by absorbing the full flattened vector
        // `flat(out.e,out.b)` into the transcript, instead of using Ajtai digest binding.
        //
        // This is chosen because Ajtai opening verification is cheap over the native base field,
        // but becomes extremely expensive when arithmetized inside F257 (digit-domain mod p mul-by-const).
        if ring_dim != 64 {
            return Err("tiny gate: setchk digest/recomb currently wired only for ring_dim=64".to_string());
        }

        // Current LF+ schedule generator uses:
        // - out_e0_len = k_rg = params.k
        // - out_b_len  = 1
        let k_rg = params.k as usize;
        let out_e0_len = k_rg;
        let out_b_len = 1usize;
        let nclaims_setchk = out_e0_len + out_b_len;
        let has_rc_setchk = out_e0_len > 1;

        // -----------------------
        // Out::verify coin wiring
        // -----------------------
        //
        // Coins are transcript-derived u32 challenges:
        // per claim: c (nvars), beta, alpha; optional rc; then the SetChk r-point coins (already bound above).
        let out_coin_start = lin_chals;
        let out_coin_total = nclaims_setchk
            .checked_mul(nvars_setchk + 2)
            .and_then(|x| x.checked_add(if has_rc_setchk { 1 } else { 0 }))
            .ok_or_else(|| "tiny gate: out_coin_total overflow (setchk recomb)".to_string())?;
        if u32_locals.len() < out_coin_start + out_coin_total {
            return Err("tiny gate: u32_locals too short for setchk Out::verify coins".to_string());
        }

        #[inline]
        fn u32_coin_to_goldilocks_digits(gb: &mut Dr1csBuilder<F257>, u: &BoundedU32ChallengeWiring) -> GoldilocksScalar {
            let bytes = goldilocks_bytes_from_u32_le_bytes(gb, &u.byte_vars);
            goldilocks_bytes_to_digits(gb, bytes)
        }

        let mut c_vars: Vec<Vec<GoldilocksScalar>> = Vec::with_capacity(nclaims_setchk);
        let mut beta_vars: Vec<GoldilocksScalar> = Vec::with_capacity(nclaims_setchk);
        let mut alpha_vars: Vec<GoldilocksScalar> = Vec::with_capacity(nclaims_setchk);
        for claim in 0..nclaims_setchk {
            let base = out_coin_start + claim * (nvars_setchk + 2);
            let mut c_point: Vec<GoldilocksScalar> = Vec::with_capacity(nvars_setchk);
            for j in 0..nvars_setchk {
                c_point.push(u32_coin_to_goldilocks_digits(&mut glue.gb, &u32_locals[base + j]));
            }
            let beta = u32_coin_to_goldilocks_digits(&mut glue.gb, &u32_locals[base + nvars_setchk]);
            let alpha = u32_coin_to_goldilocks_digits(&mut glue.gb, &u32_locals[base + nvars_setchk + 1]);
            c_vars.push(c_point);
            beta_vars.push(beta);
            alpha_vars.push(alpha);
        }
        let rc_opt: Option<GoldilocksScalar> = if has_rc_setchk {
            Some(u32_coin_to_goldilocks_digits(
                &mut glue.gb,
                &u32_locals[out_coin_start + nclaims_setchk * (nvars_setchk + 2)],
            ))
        } else {
            None
        };

        // ---------------------------------------
        // Parse absorbed out.e / out.b as ring-digits
        // ---------------------------------------
        let out_e_blocks = 1usize + (params.mlen as usize);
        let lane_len = ring_dim;
        let n_out_flat = out_e_blocks
            .checked_mul(out_e0_len)
            .and_then(|x| x.checked_mul(lane_len))
            .and_then(|x| x.checked_add(out_b_len))
            .ok_or_else(|| "tiny gate: n_out_flat overflow (setchk)".to_string())?;
        if cur + n_out_flat > prefix_payload.len() {
            return Err("tiny gate: prefix too short for absorbed out.e/out.b".to_string());
        }
        let mut out_e_vars: Vec<Vec<Vec<RingDigits>>> = vec![vec![Vec::new(); out_e0_len]; out_e_blocks];
        for i in 0..out_e0_len {
            for blk in 0..out_e_blocks {
                out_e_vars[blk][i] = Vec::with_capacity(lane_len);
                for _lane in 0..lane_len {
                    let (st, ln) = prefix_payload[cur];
                    cur += 1;
                    if ln != ring_elem_bytes {
                        return Err("tiny gate: out.e ring absorb len mismatch".to_string());
                    }
                    let rb = parse_ring_elem_absorb_as_ringbytes(&mut glue, &pose_wiring, ring_dim, st, ln)?;
                    out_e_vars[blk][i].push(ring_bytes_to_digits(&mut glue.gb, &rb));
                }
            }
        }
        let mut out_b_vars: Vec<RingDigits> = Vec::with_capacity(out_b_len);
        for _ in 0..out_b_len {
            let (st, ln) = prefix_payload[cur];
            cur += 1;
            if ln != ring_elem_bytes {
                return Err("tiny gate: out.b ring absorb len mismatch".to_string());
            }
            let rb = parse_ring_elem_absorb_as_ringbytes(&mut glue, &pose_wiring, ring_dim, st, ln)?;
            out_b_vars.push(ring_bytes_to_digits(&mut glue.gb, &rb));
        }

        // Constants used below.
        let zero_bytes = alloc_const_goldilocks_u64(&mut glue.gb, 0u64);
        let zero = goldilocks_bytes_to_digits(&mut glue.gb, zero_bytes);

        // ----------------------------
        // SetChk recombination check
        // ----------------------------
        //
        let one_bytes = alloc_const_goldilocks_u64(&mut glue.gb, 1u64);
        let one = goldilocks_bytes_to_digits(&mut glue.gb, one_bytes);
        let rc_base = rc_opt.unwrap_or(one);
        let rc_pows = goldilocks_pow_table_digits(&mut glue.gb, &rc_base, nclaims_setchk.saturating_sub(1));

        let mut ver = zero;
        for i in 0..out_e0_len {
            let eq = eq_eval_goldilocks_digits(&mut glue.gb, &c_vars[i], &rs_digits)?;
            let beta = beta_vars[i];
            let alpha = alpha_vars[i];
            let beta2 = goldilocks_mul_mod_p_digits(&mut glue.gb, &beta, &beta);
            let alpha_pows = goldilocks_pow_table_digits(&mut glue.gb, &alpha, lane_len.saturating_sub(1));

            let mut e_sum = zero;
            for blk in 0..out_e_blocks {
                for lane in 0..lane_len {
                    let ejv = &out_e_vars[blk][i][lane];
                    let ev1 = ring_eval_at_scalar_digits(&mut glue.gb, ejv, &beta)?;
                    let ev2 = ring_eval_at_scalar_digits(&mut glue.gb, ejv, &beta2)?;
                    if blk == 0 {
                        let ev1_sq = goldilocks_mul_mod_p_digits(&mut glue.gb, &ev1, &ev1);
                        let diff = goldilocks_sub_mod_p_digits(&mut glue.gb, &ev1_sq, &ev2);
                        let term = goldilocks_mul_mod_p_digits(&mut glue.gb, &diff, &alpha_pows[lane]);
                        e_sum = goldilocks_add_mod_p_digits(&mut glue.gb, &e_sum, &term);
                    }
                }
            }
            let t = goldilocks_mul_mod_p_digits(&mut glue.gb, &eq, &e_sum);
            let t = goldilocks_mul_mod_p_digits(&mut glue.gb, &t, &rc_pows[i]);
            ver = goldilocks_add_mod_p_digits(&mut glue.gb, &ver, &t);
        }
        for bi in 0..out_b_len {
            let offset = out_e0_len;
            let idx2 = bi + offset;
            let eq = eq_eval_goldilocks_digits(&mut glue.gb, &c_vars[idx2], &rs_digits)?;
            let alpha = alpha_vars[idx2];
            let beta = beta_vars[idx2];
            let beta2 = goldilocks_mul_mod_p_digits(&mut glue.gb, &beta, &beta);
            let b_ring = &out_b_vars[bi];
            let ev1 = ring_eval_at_scalar_digits(&mut glue.gb, b_ring, &beta)?;
            let ev2 = ring_eval_at_scalar_digits(&mut glue.gb, b_ring, &beta2)?;
            let ev1_sq = goldilocks_mul_mod_p_digits(&mut glue.gb, &ev1, &ev1);
            let b_claim = goldilocks_sub_mod_p_digits(&mut glue.gb, &ev1_sq, &ev2);
            let t = goldilocks_mul_mod_p_digits(&mut glue.gb, &eq, &alpha);
            let t = goldilocks_mul_mod_p_digits(&mut glue.gb, &t, &b_claim);
            let t = goldilocks_mul_mod_p_digits(&mut glue.gb, &t, &rc_pows[idx2]);
            ver = goldilocks_add_mod_p_digits(&mut glue.gb, &ver, &t);
        }

        let ver_ring = ring_const_coeff_digits(&mut glue.gb, &ver, ring_dim);
        ring_eq_digits(&mut glue.gb, &ver_ring, &v_sc);

        // ------------------------------------------------------------
        // rgchk::Dcom::verify checks + absorb(dcom.evals)
        // ------------------------------------------------------------
        //
        // This is the critical “binding” that ties the setchk outputs `out.e/out.b` to
        // the rest of the CM/Dcom verifier transcript. Mirrors `we_gate_arith.rs` (linear checks).
        //
        // Transcript schedule (see `poseidon_trace_schedule_for_plus_with_public_inputs`):
        // for each eval instance:
        // - absorb eval.a as base-ring scalars (8 bytes each), length = 1+mlen
        // - absorb eval.c as ring elements, length = 1+mlen
        //
        // We infer how many `eval` instances are present from the remaining prefix payload.
        let dcom_eval_vec_len = 1usize + (params.mlen as usize);
        let rem = prefix_payload.len().saturating_sub(cur);
        let per_eval = 2usize
            .checked_mul(dcom_eval_vec_len)
            .ok_or_else(|| "tiny gate: dcom evals per-eval overflow".to_string())?;
        if per_eval == 0 || (rem % per_eval) != 0 {
            return Err("tiny gate: prefix tail not aligned for dcom eval absorbs".to_string());
        }
        let l_evals = rem / per_eval;
        if l_evals == 0 {
            return Err("tiny gate: expected at least one dcom eval instance".to_string());
        }

        // dppow_const[i] = (decomp_b)^i in Goldilocks (digit encoding).
        let p_u64 = crate::we_goldilocks_poseidon_f257::GOLDILOCKS_P;
        let mut dppow: Vec<GoldilocksScalar> = Vec::with_capacity(k_rg);
        let mut pow_acc: u64 = 1;
        for _ in 0..k_rg {
            let bs = alloc_const_goldilocks_u64(&mut glue.gb, pow_acc);
            dppow.push(goldilocks_bytes_to_digits(&mut glue.gb, bs));
            pow_acc = ((pow_acc as u128) * (params.decomp_b as u128) % (p_u64 as u128)) as u64;
        }

        // Helper: parse an absorbed 8-byte scalar into a Goldilocks digit scalar.
        #[inline]
        fn parse_absorbed_scalar_as_goldilocks_digits(
            glue: &mut GlueCtx,
            pose_wiring: &PoseidonDr1csWiring,
            ab_start: usize,
        ) -> GoldilocksScalar {
            let mut bytes = [0usize; 8];
            for i in 0..8 {
                let gv = pose_wiring.absorb_vars[ab_start + i];
                bytes[i] = glue.copy_digit(gv);
            }
            goldilocks_bytes_to_digits(&mut glue.gb, bytes)
        }

        let mut dcom_evals_local: Vec<DcomEvalDigits> = Vec::with_capacity(l_evals);
        for l in 0..l_evals {
            // eval.a absorbs (scalars)
            let mut eval_a: Vec<GoldilocksScalar> = Vec::with_capacity(dcom_eval_vec_len);
            for _ in 0..dcom_eval_vec_len {
                let (st, ln) = *prefix_payload
                    .get(cur)
                    .ok_or("tiny gate: dcom eval.a absorb oob")?;
                cur += 1;
                if ln != 8 {
                    return Err("tiny gate: expected 8-byte absorb for dcom eval.a".to_string());
                }
                eval_a.push(parse_absorbed_scalar_as_goldilocks_digits(&mut glue, &pose_wiring, st));
            }
            // eval.c absorbs (ring elements)
            let mut eval_c: Vec<RingDigits> = Vec::with_capacity(dcom_eval_vec_len);
            for _ in 0..dcom_eval_vec_len {
                let (st, ln) = *prefix_payload
                    .get(cur)
                    .ok_or("tiny gate: dcom eval.c absorb oob")?;
                cur += 1;
                if ln != ring_elem_bytes {
                    return Err("tiny gate: expected ring-elem absorb for dcom eval.c".to_string());
                }
                let rb = parse_ring_elem_absorb_as_ringbytes(&mut glue, &pose_wiring, ring_dim, st, ln)?;
                eval_c.push(ring_bytes_to_digits(&mut glue.gb, &rb));
            }

            // Allocate witness eval.b ring elements (not absorbed) and eval.v scalars (not absorbed).
            let mut eval_b: Vec<RingDigits> = Vec::with_capacity(eval_a.len());
            for _ in 0..eval_a.len() {
                let mut r: RingDigits = Vec::with_capacity(ring_dim);
                for _ in 0..ring_dim {
                    r.push(alloc_witness_goldilocks_u64_digits(&mut glue.gb, 0u64));
                }
                eval_b.push(r);
            }
            let mut eval_v: Vec<GoldilocksScalar> = Vec::with_capacity(ring_dim);
            for _ in 0..ring_dim {
                eval_v.push(alloc_witness_goldilocks_u64_digits(&mut glue.gb, 0u64));
            }

            // If provided, overwrite the placeholder witness with real proof values.
            if let Some(w) = extra_witness {
                if let (Some(b_l), Some(v_l)) = (w.dcom_eval_b.get(l), w.dcom_eval_v.get(l)) {
                    if b_l.len() == eval_b.len() && v_l.len() == ring_dim {
                        for (i, b_i) in b_l.iter().enumerate() {
                            if b_i.len() == ring_dim {
                                for k in 0..ring_dim {
                                    eval_b[i][k] = alloc_witness_goldilocks_u64_digits(&mut glue.gb, b_i[k]);
                                }
                            }
                        }
                        for k in 0..ring_dim {
                            eval_v[k] = alloc_witness_goldilocks_u64_digits(&mut glue.gb, v_l[k]);
                        }
                    }
                }
            }

            // Check 1: ct(psi * eval.b[i]) == eval.a[i] for each i.
            for i in 0..eval_a.len() {
                let ct = ct_psi_mul_ring_digits_d64(&mut glue.gb, &eval_b[i])?;
                for di in 0..17 {
                    glue.gb.enforce_lc_times_one_eq_const(vec![
                        (F257::ONE, ct[di]),
                        (-F257::ONE, eval_a[i][di]),
                    ]);
                }
            }

            // Check 2: for each block ni and column col, ct(psi * Σ_i dppow[i] * out.e[ni][base+i][col]) == expected.
            let base = l
                .checked_mul(k_rg)
                .ok_or_else(|| "tiny gate: rgchk base overflow".to_string())?;
            for ni in 0..out_e_blocks {
                for col in 0..ring_dim {
                    let mut acc_ring = super::cm_math::ring_zero_digits(&mut glue.gb, ring_dim);
                    for i in 0..k_rg {
                        let idx = base + i;
                        if idx >= out_e_vars[ni].len() {
                            return Err("tiny gate: out.e length too short for rgchk".to_string());
                        }
                        let ui_col = &out_e_vars[ni][idx][col];
                        let t = ring_scale_digits(&mut glue.gb, ui_col, &dppow[i]);
                        acc_ring = ring_add_digits(&mut glue.gb, &acc_ring, &t);
                    }
                    let ct = ct_psi_mul_ring_digits_d64(&mut glue.gb, &acc_ring)?;
                    let expected = if ni == 0 {
                        eval_v[col]
                    } else {
                        // eval.c[ni] is a ring; take coefficient `col`.
                        *eval_c
                            .get(ni)
                            .ok_or("tiny gate: eval.c length mismatch (rgchk)")?
                            .get(col)
                            .ok_or("tiny gate: eval.c coeff index oob (rgchk)")?
                    };
                    for di in 0..17 {
                        glue.gb.enforce_lc_times_one_eq_const(vec![
                            (F257::ONE, ct[di]),
                            (-F257::ONE, expected[di]),
                        ]);
                    }
                }
            }

            // Save the Dcom eval vectors for later CM claimed-sum wiring.
            dcom_evals_local.push(DcomEvalDigits {
                a: eval_a,
                b: eval_b,
                c: eval_c,
                v: eval_v,
            });
        }

        // Export SetChk outputs and Dcom evals for the CM verifier math stage.
        setchk_out_e_vars_for_cm = Some(out_e_vars);
        dcom_evals_for_cm = Some(dcom_evals_local);
    }

    let goldilocks_rejection_locals = build_goldilocks_rejection_coins(&mut glue, &pose_wiring, &goldilocks_ranges)?;
    lf_stage_log("build_goldilocks_rejection_coins", Some(&pose_inst), Some(&glue), &mut mem_prev);
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
        &u32_locals,
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

    // Convert shared SetChk/Dcom values into Arc containers for parallel CM modules.
    let setchk_out_e_vars_for_cm = setchk_out_e_vars_for_cm.map(Arc::new);
    let dcom_evals_for_cm = dcom_evals_for_cm.map(Arc::new);

    // Compute CM shared precomputations once in the base glue module (matches full WE gate behavior):
    // - tensor(c0/c1), dpp, s_prime_flat, recovered setchk r-point
    // - u[l][ni] from out.e and s_prime_flat (heavy negacyclic ring-muls)
    //
    // Each per-`which` CM module will import these, avoiding duplicated negacyclic work.
    let cm_shared_base: Option<Arc<CmSharedPrecompBase>> = if ring_dim == 64
        && l_instances_expected > 0
        && params.kappa > 0
        && (params.kappa as usize).is_power_of_two()
        && ((params.k as usize) * ring_dim).is_power_of_two()
        && (params.l as usize).is_power_of_two()
        && setchk_out_e_vars_for_cm.is_some()
    {
        let kappa = params.kappa as usize;
        let log_kappa = ark_std::log2(kappa.next_power_of_two()) as usize;
        let k_decomp = params.k as usize;
        let ell = params.l as usize;
        let rows_per_l = 1usize + (params.mlen as usize);

        // Shared CM challenge seed blocks: c0/c1.
        let cm_u32_start = cm_u32_start_idx(wiring);
        if u32_locals.len() < cm_u32_start + 2 * log_kappa {
            None
        } else if short_locals.len() < 3 + k_decomp * ring_dim {
            None
        } else {
            // c0/c1 digits, then tensor-expand.
            let c0_u32 = &u32_locals[cm_u32_start..cm_u32_start + log_kappa];
            let c1_u32 = &u32_locals[cm_u32_start + log_kappa..cm_u32_start + 2 * log_kappa];
            let c0_digits: Vec<_> = c0_u32
                .iter()
                .map(|u| {
                    let bytes = goldilocks_bytes_from_u32_le_bytes(
                        &mut glue.gb,
                        &[u.byte_vars[0], u.byte_vars[1], u.byte_vars[2], u.byte_vars[3]],
                    );
                    goldilocks_bytes_to_digits(&mut glue.gb, bytes)
                })
                .collect();
            let c1_digits: Vec<_> = c1_u32
                .iter()
                .map(|u| {
                    let bytes = goldilocks_bytes_from_u32_le_bytes(
                        &mut glue.gb,
                        &[u.byte_vars[0], u.byte_vars[1], u.byte_vars[2], u.byte_vars[3]],
                    );
                    goldilocks_bytes_to_digits(&mut glue.gb, bytes)
                })
                .collect();
            let t0 = tensor_goldilocks_scalars_digits(&mut glue.gb, &c0_digits);
            let t1 = tensor_goldilocks_scalars_digits(&mut glue.gb, &c1_digits);
            let tensor_c0_ring = tensor_goldilocks_ringconst_digits(&mut glue.gb, &t0, ring_dim);
            let tensor_c1_ring = tensor_goldilocks_ringconst_digits(&mut glue.gb, &t1, ring_dim);

            // dpp: dp^i as constant-coeff ring elements.
            let dp = (ring_dim / 2) as u64;
            let p = crate::we_goldilocks_poseidon_f257::GOLDILOCKS_P;
            let mut acc: u64 = 1;
            let mut dpp: Vec<RingDigits> = Vec::with_capacity(ell);
            for _ in 0..ell {
                let s_bytes = super::cm_math::alloc_const_goldilocks_u64(&mut glue.gb, acc);
                let s = goldilocks_bytes_to_digits(&mut glue.gb, s_bytes);
                dpp.push(super::cm_math::ring_const_coeff_digits(&mut glue.gb, &s, ring_dim));
                acc = ((acc as u128) * (dp as u128) % (p as u128)) as u64;
            }

            // s_prime_flat: k*d short challenges, each is a ring element with centered coeff bytes.
            let need_sprime = k_decomp * ring_dim;
            let z = glue.gb.new_var(F257::ZERO);
            glue.gb.enforce_var_eq_const(z, F257::ZERO);
            let c128 = super::cm_math::alloc_const_goldilocks_u64(&mut glue.gb, 128u64);
            let c128_d = goldilocks_bytes_to_digits(&mut glue.gb, c128);
            let mut sflat: Vec<RingDigits> = Vec::with_capacity(need_sprime);
            for blk in 0..need_sprime {
                let sb = &short_locals[3 + blk];
                if sb.byte_vars.len() != ring_dim {
                    return Err("tiny gate: short byte_vars len mismatch (base s_prime_flat)".to_string());
                }
                let mut re: RingDigits = Vec::with_capacity(ring_dim);
                for &bv in &sb.byte_vars {
                    let mut bbytes = [0usize; 8];
                    bbytes[0] = bv;
                    for i in 1..8 {
                        bbytes[i] = z;
                    }
                    let bd = goldilocks_bytes_to_digits(&mut glue.gb, bbytes);
                    let centered = super::cm_math::goldilocks_sub_mod_p_digits(&mut glue.gb, &bd, &c128_d);
                    re.push(centered);
                }
                sflat.push(re);
            }

            // Recover the SetChk verifier point `r` used in eq(r, ro) (same deterministic cursor math as CM module).
            let nvars_lin = params.nvars_setchk as usize;
            let n_lin_proofs = l_instances_expected;
            let lin_chals = n_lin_proofs
                .checked_mul(2usize.saturating_mul(nvars_lin))
                .ok_or_else(|| "tiny gate: lin_chals overflow".to_string())?;
            let nclaims = k_decomp.checked_add(1).ok_or_else(|| "tiny gate: nclaims overflow".to_string())?;
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
                None
            } else {
                let mut rdig: Vec<GoldilocksScalar> = Vec::with_capacity(nvars_lin);
                for u in &u32_locals[r_start..r_end] {
                    let bytes = goldilocks_bytes_from_u32_le_bytes(
                        &mut glue.gb,
                        &[u.byte_vars[0], u.byte_vars[1], u.byte_vars[2], u.byte_vars[3]],
                    );
                    rdig.push(goldilocks_bytes_to_digits(&mut glue.gb, bytes));
                }

                // Compute u[l][ni] once (heavy): Σ out.e * s_prime_flat.
                let out_e_base = setchk_out_e_vars_for_cm.as_ref().unwrap().as_ref();
                if out_e_base.len() != rows_per_l {
                    None
                } else {
                    use super::cm_ir::{lower_ir_into_builder, ring_mul_negacyclic_ntt_goldilocks_d64_ir, IrBuilder, VarRef as IrVarRef};

                    #[inline]
                    fn ringdigits64_to_ir(a: &RingDigits) -> Result<[[IrVarRef; 17]; 64], String> {
                        if a.len() != 64 {
                            return Err("tiny gate: expected ring_dim=64 for IR ring-mul".to_string());
                        }
                        Ok(core::array::from_fn(|i| core::array::from_fn(|j| IrVarRef::Base(a[i][j]))))
                    }
                    #[inline]
                    fn map_ring_out(out_ir: &[[IrVarRef; 17]; 64], lowered: &super::cm_ir::LoweredIr) -> RingDigits {
                        let out: [GoldilocksScalar; 64] = core::array::from_fn(|i| {
                            core::array::from_fn(|j| lowered.map_var(out_ir[i][j]))
                        });
                        out.into_iter().collect()
                    }

                    let timing_u = tiny_opmix_on();
                    let mut u_ir_build_time = Duration::ZERO;
                    let mut u_lower_time = Duration::ZERO;
                    let mut u_frag_count: usize = 0;
                    let batch_size: usize = 64;

                    let mut u_all: Vec<Vec<RingDigits>> = Vec::with_capacity(l_instances_expected);
                    for l in 0..l_instances_expected {
                        let mut u_l: Vec<RingDigits> = Vec::with_capacity(rows_per_l);
                        for ni in 0..rows_per_l {
                            let mut acc_ring = super::cm_math::ring_zero_digits(&mut glue.gb, ring_dim);
                            // Collect term refs for this (l,ni).
                            let mut terms: Vec<(&RingDigits, usize)> = Vec::with_capacity(k_decomp * ring_dim);
                            for blk in 0..k_decomp {
                                let idx = l
                                    .checked_mul(k_decomp)
                                    .and_then(|x| x.checked_add(blk))
                                    .ok_or_else(|| "tiny gate: u index overflow (base cm)".to_string())?;
                                if idx >= out_e_base[ni].len() {
                                    return Err("tiny gate: out.e too short for base CM u computation".to_string());
                                }
                                for col in 0..ring_dim {
                                    let uij = &out_e_base[ni][idx][col];
                                    let sp_idx = blk * ring_dim + col;
                                    terms.push((uij, sp_idx));
                                }
                            }

                            for chunk in terms.chunks(batch_size) {
                                // IMPORTANT: take a fresh snapshot before lowering allocates new vars.
                                let base_asg_ir: &[F257] = glue.gb.assignment.as_slice();
                                let t_build = Instant::now();
                                let frags: Vec<(_, [[IrVarRef; 17]; 64])> = chunk
                                    .par_iter()
                                    .map(|(uij, sp_idx)| -> Result<_, String> {
                                        let u_ir = ringdigits64_to_ir(uij)?;
                                        let s_ir = ringdigits64_to_ir(&sflat[*sp_idx])?;
                                        let mut ib = IrBuilder::new(base_asg_ir);
                                        let out_ir = ring_mul_negacyclic_ntt_goldilocks_d64_ir(&mut ib, &u_ir, &s_ir);
                                        // Keep op-mix accounting consistent even when ring-muls are built via IR shards.
                                        super::op_counts::tiny_cm_bump(|c| c.ring_mul_negacyclic += 1);
                                        Ok((ib.ir, out_ir))
                                    })
                                    .collect::<Result<Vec<_>, _>>()?;
                                u_ir_build_time = u_ir_build_time.saturating_add(t_build.elapsed());
                                u_frag_count = u_frag_count.saturating_add(frags.len());

                                let t_lower = Instant::now();
                                for (ir, out_ir) in frags {
                                    let lowered = lower_ir_into_builder(&mut glue.gb, ir);
                                    let prod = map_ring_out(&out_ir, &lowered);
                                    acc_ring = ring_add_digits(&mut glue.gb, &acc_ring, &prod);
                                }
                                u_lower_time = u_lower_time.saturating_add(t_lower.elapsed());
                            }
                            u_l.push(acc_ring);
                        }
                        u_all.push(u_l);
                    }

                    if timing_u {
                        let expected_muls = l_instances_expected
                            .saturating_mul(rows_per_l)
                            .saturating_mul(k_decomp.saturating_mul(ring_dim));
                        eprintln!(
                            "tiny_gate: CM u_shared timing: ir_build={:?} lower+acc={:?} frags={} expected_muls={} batch={} threads={}",
                            u_ir_build_time,
                            u_lower_time,
                            u_frag_count,
                            expected_muls,
                            batch_size,
                            rayon::current_num_threads().max(1),
                        );
                    }

                    Some(Arc::new(CmSharedPrecompBase {
                        tensor_c0_ring,
                        tensor_c1_ring,
                        s_prime_flat_ring: sflat,
                        dpp_ring: dpp,
                        r_point_digits: rdig,
                        u: u_all,
                    }))
                }
            }
        }
    } else {
        None
    };

    // Build the CM verifier math in parallel modules (one per sumcheck), then merge by explicitly
    // gluing their local copies to Poseidon vars.
    let cm_extra_glues: Vec<GlueCtx> =
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
            let pose_asg = glue.pose_asg.clone();
            let base_asg = glue.gb.assignment.as_slice();
            let (g0, g1) = join(
                || {
                    build_cm_glue_for_which(
                        cfg,
                        ops,
                        &pose_wiring,
                        ring_dim,
                        params,
                        wiring,
                        l_instances_expected,
                        &comh_absorbs,
                        &sc_msg_absorbs[0],
                        &eval_absorbs[0],
                        setchk_out_e_vars_for_cm.clone(),
                        dcom_evals_for_cm.clone(),
                        cm_shared_base.clone(),
                        0,
                        pose_asg.clone(),
                        base_asg,
                        &short_locals,
                        &u32_locals,
                    )
                },
                || {
                    build_cm_glue_for_which(
                        cfg,
                        ops,
                        &pose_wiring,
                        ring_dim,
                        params,
                        wiring,
                        l_instances_expected,
                        &comh_absorbs,
                        &sc_msg_absorbs[1],
                        &eval_absorbs[1],
                        setchk_out_e_vars_for_cm.clone(),
                        dcom_evals_for_cm.clone(),
                        cm_shared_base.clone(),
                        1,
                        pose_asg.clone(),
                        base_asg,
                        &short_locals,
                        &u32_locals,
                    )
                },
            );
            vec![g0?, g1?]
        } else {
            Vec::new()
        };

    // ------------------------------------------------------------------------
    // Decomp verifier math (LinB2X + DecompProof)
    // ------------------------------------------------------------------------
    //
    // Mirrors `we_gate_arith.rs::decomp_verifier_math_dr1cs`:
    // - C0 + B*C1 == cm_g
    // - v0a + B*v1a == va, v0b + B*v1b == vb
    //
    // Note: this part does not interact with the transcript; these are pure algebraic checks over the ring.
    // In the tiny gate shape harness we allocate witness values as zeros (satisfiable); in the real
    // large-trace path the witness assignment must supply actual proof values.
    if ring_dim == 64 {
        let kappa = params.kappa as usize;
        let vlen = 1usize + (params.mlen as usize); // matches `we_gate_arith.rs` dummy proof shape
        let p_u64 = crate::we_goldilocks_poseidon_f257::GOLDILOCKS_P;
        let b_u64: u64 = ((params.decomp_b as u128) % (p_u64 as u128)) as u64;

        #[inline]
        fn alloc_witness_ring_digits(gb: &mut Dr1csBuilder<F257>, ring_dim: usize) -> RingDigits {
            let mut r: RingDigits = Vec::with_capacity(ring_dim);
            for _ in 0..ring_dim {
                r.push(alloc_witness_goldilocks_u64_digits(gb, 0u64));
            }
            r
        }
        #[inline]
        fn alloc_witness_ring_digits_from_u64s(
            gb: &mut Dr1csBuilder<F257>,
            ring_dim: usize,
            coeffs: Option<&[u64]>,
        ) -> RingDigits {
            if let Some(c) = coeffs {
                if c.len() == ring_dim {
                    let mut r: RingDigits = Vec::with_capacity(ring_dim);
                    for k in 0..ring_dim {
                        r.push(alloc_witness_goldilocks_u64_digits(gb, c[k]));
                    }
                    return r;
                }
            }
            alloc_witness_ring_digits(gb, ring_dim)
        }

        #[inline]
        fn ring_recompose_base_b(
            gb: &mut Dr1csBuilder<F257>,
            r0: &RingDigits,
            r1: &RingDigits,
            b_u64: u64,
        ) -> RingDigits {
            debug_assert_eq!(r0.len(), 64);
            debug_assert_eq!(r1.len(), 64);
            let mut out: RingDigits = Vec::with_capacity(64);
            for j in 0..64 {
                let t = goldilocks_mul_const_mod_p_digits(gb, &r1[j], b_u64);
                let s = goldilocks_add_mod_p_digits(gb, &r0[j], &t);
                out.push(s);
            }
            out
        }

        // Allocate DecompProof witnesses: C0,C1 (kappa each), v0/v1 (vlen pairs each).
        let mut dcomp_c0: Vec<RingDigits> = Vec::with_capacity(kappa);
        let mut dcomp_c1: Vec<RingDigits> = Vec::with_capacity(kappa);
        for _ in 0..kappa {
            dcomp_c0.push(alloc_witness_ring_digits(&mut glue.gb, ring_dim));
            dcomp_c1.push(alloc_witness_ring_digits(&mut glue.gb, ring_dim));
        }
        let mut v0a: Vec<RingDigits> = Vec::with_capacity(vlen);
        let mut v0b: Vec<RingDigits> = Vec::with_capacity(vlen);
        let mut v1a: Vec<RingDigits> = Vec::with_capacity(vlen);
        let mut v1b: Vec<RingDigits> = Vec::with_capacity(vlen);
        for _ in 0..vlen {
            v0a.push(alloc_witness_ring_digits(&mut glue.gb, ring_dim));
            v0b.push(alloc_witness_ring_digits(&mut glue.gb, ring_dim));
            v1a.push(alloc_witness_ring_digits(&mut glue.gb, ring_dim));
            v1b.push(alloc_witness_ring_digits(&mut glue.gb, ring_dim));
        }

        // Allocate LinB2X witnesses: cm_g (kappa), vo (vlen pairs).
        let mut cm_g: Vec<RingDigits> = Vec::with_capacity(kappa);
        for _ in 0..kappa {
            cm_g.push(alloc_witness_ring_digits(&mut glue.gb, ring_dim));
        }
        let mut va: Vec<RingDigits> = Vec::with_capacity(vlen);
        let mut vb: Vec<RingDigits> = Vec::with_capacity(vlen);
        for _ in 0..vlen {
            va.push(alloc_witness_ring_digits(&mut glue.gb, ring_dim));
            vb.push(alloc_witness_ring_digits(&mut glue.gb, ring_dim));
        }

        // If provided, allocate from the real proof coefficients.
        if let Some(w) = extra_witness {
            if w.decomp_c0.len() == kappa && w.decomp_c1.len() == kappa && w.linb2x_cm_g.len() == kappa {
                dcomp_c0 = (0..kappa)
                    .map(|i| alloc_witness_ring_digits_from_u64s(&mut glue.gb, ring_dim, Some(&w.decomp_c0[i])))
                    .collect();
                dcomp_c1 = (0..kappa)
                    .map(|i| alloc_witness_ring_digits_from_u64s(&mut glue.gb, ring_dim, Some(&w.decomp_c1[i])))
                    .collect();
                cm_g = (0..kappa)
                    .map(|i| alloc_witness_ring_digits_from_u64s(&mut glue.gb, ring_dim, Some(&w.linb2x_cm_g[i])))
                    .collect();
            }
            if w.decomp_v0a.len() == vlen
                && w.decomp_v0b.len() == vlen
                && w.decomp_v1a.len() == vlen
                && w.decomp_v1b.len() == vlen
                && w.linb2x_vo_a.len() == vlen
                && w.linb2x_vo_b.len() == vlen
            {
                v0a = (0..vlen)
                    .map(|i| alloc_witness_ring_digits_from_u64s(&mut glue.gb, ring_dim, Some(&w.decomp_v0a[i])))
                    .collect();
                v0b = (0..vlen)
                    .map(|i| alloc_witness_ring_digits_from_u64s(&mut glue.gb, ring_dim, Some(&w.decomp_v0b[i])))
                    .collect();
                v1a = (0..vlen)
                    .map(|i| alloc_witness_ring_digits_from_u64s(&mut glue.gb, ring_dim, Some(&w.decomp_v1a[i])))
                    .collect();
                v1b = (0..vlen)
                    .map(|i| alloc_witness_ring_digits_from_u64s(&mut glue.gb, ring_dim, Some(&w.decomp_v1b[i])))
                    .collect();
                va = (0..vlen)
                    .map(|i| alloc_witness_ring_digits_from_u64s(&mut glue.gb, ring_dim, Some(&w.linb2x_vo_a[i])))
                    .collect();
                vb = (0..vlen)
                    .map(|i| alloc_witness_ring_digits_from_u64s(&mut glue.gb, ring_dim, Some(&w.linb2x_vo_b[i])))
                    .collect();
            }
        }

        // Enforce recomposition equalities.
        for i in 0..kappa {
            let rec = ring_recompose_base_b(&mut glue.gb, &dcomp_c0[i], &dcomp_c1[i], b_u64);
            ring_eq_digits(&mut glue.gb, &rec, &cm_g[i]);
        }
        for i in 0..vlen {
            let rec_a = ring_recompose_base_b(&mut glue.gb, &v0a[i], &v1a[i], b_u64);
            let rec_b = ring_recompose_base_b(&mut glue.gb, &v0b[i], &v1b[i], b_u64);
            ring_eq_digits(&mut glue.gb, &rec_a, &va[i]);
            ring_eq_digits(&mut glue.gb, &rec_b, &vb[i]);
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

    // `finalize()` needs to recover the owned pose assignment without cloning.
    // Drop the extra Arc handle held by this stack frame.
    drop(pose_asg);

    finalize(
        pose_inst,
        pose_wiring,
        ops,
        glue,
        {
            let mut extra = canonical_glues;
            extra.extend(cm_extra_glues);
            extra
        },
        short_locals,
        u32_locals,
        goldilocks_locals,
        goldilocks_rejection_locals,
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