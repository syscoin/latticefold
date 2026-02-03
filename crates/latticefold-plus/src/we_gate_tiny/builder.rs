use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::transcript::DEFAULT_REJECTION_TRIES;

use rayon::join;
use rayon::prelude::*;

use ark_ff::{Field, PrimeField};
use ark_crypto_primitives::sponge::poseidon::PoseidonConfig;
use latticefold::transcript::poseidon::{f257_poseidon_config, F257};
use symphony::dpp_poseidon::{
    poseidon_sponge_dr1cs_from_ops_with_wiring_no_bytes_count_sharded, PoseidonDr1csWiring,
};
#[cfg(unix)]
use symphony::dpp_poseidon::poseidon_sponge_dr1cs_from_ops_with_wiring_no_bytes_range_sharded_into_files;
use symphony::dpp_sumcheck::Dr1csBuilder;
use symphony::file_backed_dr1cs::{
    FileBackedLayout, FileBackedSparseDr1csInstance,
};
#[cfg(unix)]
use symphony::file_backed_dr1cs::FileBackedRangeWriter;
use symphony::poseidon_trace::count_permutes_for_ops;
use symphony::transcript::PoseidonTraceOp;

use crate::we_statement::WeParams;

use super::challenges::{
    bounded_u32_from_8_digits_base128, res257_from_u64_bytes_le,
    select_first_ok_u32_try_digits, short_challenge_from_digits_128, squeeze_field_ranges_by_op_index,
    BoundedU32ChallengeWiring, GoldilocksChallengeWiring, ShortChallengeWiring, TinyCoinOpWiring,
};
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
use super::surfaces::{CmDigitMulSqSurfaceWiring, CmDigitMulSurfaceWiring};

use super::cm_math::{
    alloc_const_goldilocks_u64,
    eq_eval_goldilocks_digits, eval_t_z_optimized_ring_digits_pair, goldilocks_bytes_to_digits, goldilocks_pow_table_digits,
    goldilocks_add_mod_p_digits, goldilocks_mul_mod_p_digits, goldilocks_sub_mod_p_digits,
    ct_psi_mul_ring_digits_d64,
    ring_eval_at_scalar_digits,
    ring_add_digits, ring_bytes_to_digits, ring_eq_digits,
    ring_const_coeff_digits,
    tensor_goldilocks_ringconst_digits, tensor_goldilocks_scalars_digits, RingBytes, RingDigits,
};
use super::op_counts::tiny_cm_counts_take;

#[derive(Clone, Debug)]
struct DcomEvalDigits {
    a: Vec<GoldilocksScalar>,
    b: Vec<RingDigits>,
    c: Vec<RingDigits>,
}

#[derive(Clone, Debug)]
struct FComsDigits {
    cm_f: Vec<RingDigits>,
    c_mf: Vec<RingDigits>,
    cm_mtau: Vec<RingDigits>,
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
    // s[0..3] short-challenge ring elements (length 3), each entry is a ring element.
    //
    // These are used for:
    // - building CM reduced instance targets (cm_g, v_o) for Decomp checks
    // and should be computed once and reused.
    s_rings: [RingDigits; 3],
    // dpp = dp^i as ring-constant elements (length ell)
    dpp_ring: Vec<RingDigits>,
    // recovered SetChk verifier point r (length nvars_setchk)
    r_point_digits: Arc<Vec<GoldilocksScalar>>,
    // u[l][ni] (length L × (1+mlen))
    u: Vec<Vec<RingDigits>>,
}

// -----------------------------------------------------------------------------
// Small helpers (keep big functions readable)
// -----------------------------------------------------------------------------

#[inline]
fn import_scalar_from_base(glue: &mut GlueCtx, base_asg: &[F257], s: &GoldilocksScalar) -> GoldilocksScalar {
    core::array::from_fn(|i| glue.import_base_var(base_asg, s[i]))
}

#[inline]
fn import_ring_from_base(glue: &mut GlueCtx, base_asg: &[F257], r: &RingDigits) -> RingDigits {
    r.iter().map(|s| import_scalar_from_base(glue, base_asg, s)).collect()
}

/// Byte-bit layout for a short-challenge coefficient, expressed as 8 bytes (each 8 bits).
///
/// For our short-challenge regime, only byte 0 can be nonzero; bytes 1..7 are all zero.
type ShortCoeffBits = [[[usize; 8]; 8]; 64];

fn collect_short_coeff_bits(
    gb: &mut Dr1csBuilder<F257>,
    ring_dim: usize,
    exp: usize,
    zero_bits: [usize; 8],
    n_blocks: usize,
    mut get_byte_var: impl FnMut(usize, usize) -> usize,
) -> Result<Vec<ShortCoeffBits>, String> {
    if ring_dim != 64 {
        return Err("tiny gate: short coeff helper requires ring_dim=64".to_string());
    }
    let mut blocks_bits: Vec<ShortCoeffBits> = Vec::with_capacity(n_blocks);
    for blk in 0..n_blocks {
        let mut out_blk_bits: ShortCoeffBits = [[[0usize; 8]; 8]; 64];
        for col in 0..ring_dim {
            let bv = get_byte_var(blk, col);
            let bb = decompose_existing_byte_var_to_bits::<F257>(gb, bv);
            if bb.len() != 8 {
                return Err("tiny gate: internal error: byte bits len mismatch (short coeff)".to_string());
            }
            let byte0_bits: [usize; 8] = core::array::from_fn(|i| if i < exp { bb[i] } else { zero_bits[i] });
            out_blk_bits[col] = core::array::from_fn(|bi| if bi == 0 { byte0_bits } else { zero_bits });
        }
        blocks_bits.push(out_blk_bits);
    }
    Ok(blocks_bits)
}

fn build_short_coeff_digits_ir(
    gb: &mut Dr1csBuilder<F257>,
    blocks_bits: &[ShortCoeffBits],
    half_ir: &[super::cm_ir::VarRef; 17],
    p_u64: u64,
    p_d_const: &[i8; 17],
) -> (super::cm_ir::CmIr, Vec<[[super::cm_ir::VarRef; 17]; 64]>) {
    let base_asg: &[F257] = &gb.assignment;
    let mut ib = if gb.is_count_only() {
        super::cm_ir::IrBuilder::new_count_only(base_asg)
    } else {
        super::cm_ir::IrBuilder::new(base_asg)
    };
    let mut outs: Vec<[[super::cm_ir::VarRef; 17]; 64]> = Vec::with_capacity(blocks_bits.len());
    for out_blk_bits in blocks_bits {
        let mut out_blk: [[super::cm_ir::VarRef; 17]; 64] = [[super::cm_ir::VarRef::Base(gb.one()); 17]; 64];
        for col in 0..64 {
            let bytes_bits_ir: [[super::cm_ir::VarRef; 8]; 8] = core::array::from_fn(|bi| {
                core::array::from_fn(|j| super::cm_ir::VarRef::Base(out_blk_bits[col][bi][j]))
            });
            let low_digits_ir = super::cm_ir::u64_bytes_to_bal16_digits_from_bits_ir(&mut ib, &bytes_bits_ir);
            let coeff_ir =
                super::cm_ir::goldilocks_sub_mod_p_digits_ir(&mut ib, &low_digits_ir, half_ir, p_u64, p_d_const);
            out_blk[col] = coeff_ir;
        }
        outs.push(out_blk);
    }
    (ib.ir, outs)
}

fn map_short_coeff_digits_blocks_to_rings(
    lowered: &super::cm_ir::LoweredIr,
    outs: Vec<[[super::cm_ir::VarRef; 17]; 64]>,
    ring_dim: usize,
) -> Vec<RingDigits> {
    debug_assert_eq!(ring_dim, 64, "tiny gate: short coeff mapping assumes ring_dim=64");
    let mut out: Vec<RingDigits> = Vec::with_capacity(outs.len());
    for out_blk in outs {
        let mut ring: RingDigits = Vec::with_capacity(ring_dim);
        for col in 0..ring_dim {
            ring.push(core::array::from_fn(|j| lowered.map_var(out_blk[col][j])));
        }
        out.push(ring);
    }
    out
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
pub struct TinyExtraWitness {
    pub dcom_eval_b: Vec<Vec<Vec<u64>>>, // [l][eval_len][ring_dim]
    pub dcom_eval_v: Vec<Vec<u64>>,      // [l][ring_dim]

    // DecompProof
    pub decomp_c0: Vec<Vec<u64>>,  // [kappa][ring_dim]
    pub decomp_c1: Vec<Vec<u64>>,  // [kappa][ring_dim]
    pub decomp_v0a: Vec<Vec<u64>>, // [vlen][ring_dim]
    pub decomp_v0b: Vec<Vec<u64>>, // [vlen][ring_dim]
    pub decomp_v1a: Vec<Vec<u64>>, // [vlen][ring_dim]
    pub decomp_v1b: Vec<Vec<u64>>, // [vlen][ring_dim]
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

#[inline]
fn lf_profile_on() -> bool {
    match std::env::var("LF_PLUS_PROFILE") {
        Ok(v) => v != "0",
        Err(_) => false,
    }
}

#[inline]
fn lf_profile_log(msg: &str) {
    if lf_profile_on() {
        if lf_mem_on() {
            let m = lf_mem_sample();
            eprintln!(
                "[LF_PROFILE] tiny_gate {msg} mem(rss={} hwm={} vmsize={})",
                fmt_mib(m.rss_bytes),
                fmt_mib(m.hwm_bytes),
                fmt_mib(m.vmsize_bytes)
            );
        } else {
            eprintln!("[LF_PROFILE] tiny_gate {msg}");
        }
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
fn maybe_print_tiny_opmix_common(
    cfg: Option<&PoseidonConfig<F257>>,
    ops: &[PoseidonTraceOp<F257>],
    pose_constraints: usize,
    glue_constraints: usize,
    tag: &str,
    glue: &GlueCtx<'_>,
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
    let c_pose = pose_constraints;
    let c_pose_no_bytes = c_pose;
    let c_glue = glue_constraints;

    eprintln!("==============================================================");
    if tag.is_empty() {
        eprintln!("LF+ WE tiny gate op-mix (Poseidon(F257) + tiny CM gadgets)");
    } else {
        eprintln!("LF+ WE tiny gate op-mix (Poseidon(F257) + tiny CM gadgets) {tag}");
    }
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
        glue.gb.nconstraints(),
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
        eprintln!("  (dr1cs scope profile disabled; set LF_PROFILE_DR1CS=1 for top scopes)");
    }
    eprintln!("==============================================================");
}

struct GlueCtx<'a> {
    gb: Dr1csBuilder<F257>,
    pose_asg: &'a [F257],
    local_map: BTreeMap<usize, usize>,
    // Cache for imported base-glue vars so we don't allocate redundant copies.
    // Key: base-glue var index, Value: local var index in this module.
    base_map: BTreeMap<usize, usize>,
    // Extra "glue" equalities between this module's vars and the *base glue* module's vars.
    // Each entry is (base_var, local_var) in their respective local index spaces.
    base_eqs: Vec<(usize, usize)>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TinyGateBuildMode {
    Count,
    #[cfg(unix)]
    RangeBase,
}

#[cfg(unix)]
#[derive(Debug)]
struct TinyGateMergedFiles {
    fc_a: std::fs::File,
    fi_a: std::fs::File,
    fc_b: std::fs::File,
    fi_b: std::fs::File,
    fc_c: std::fs::File,
    fi_c: std::fs::File,
    f_rows: std::fs::File,
}

#[cfg(unix)]
impl TinyGateMergedFiles {
    fn try_clone_all(
        &self,
    ) -> Result<
        (
            std::fs::File,
            std::fs::File,
            std::fs::File,
            std::fs::File,
            std::fs::File,
            std::fs::File,
            std::fs::File,
        ),
        String,
    > {
        Ok((
            self.fc_a.try_clone().map_err(|e| format!("clone a_coeffs failed: {e}"))?,
            self.fi_a.try_clone().map_err(|e| format!("clone a_idx failed: {e}"))?,
            self.fc_b.try_clone().map_err(|e| format!("clone b_coeffs failed: {e}"))?,
            self.fi_b.try_clone().map_err(|e| format!("clone b_idx failed: {e}"))?,
            self.fc_c.try_clone().map_err(|e| format!("clone c_coeffs failed: {e}"))?,
            self.fi_c.try_clone().map_err(|e| format!("clone c_idx failed: {e}"))?,
            self.f_rows
                .try_clone()
                .map_err(|e| format!("clone constraints failed: {e}"))?,
        ))
    }
}

#[cfg(unix)]
#[derive(Clone, Debug)]
struct TinyGateRangeBase {
    files: Arc<TinyGateMergedFiles>,
    plan: Arc<TinyGatePlan>,
    part_base: usize,
}

// Stub type on non-unix so signatures can stay uniform.
#[cfg(not(unix))]
#[derive(Clone, Debug)]
struct TinyGateRangeBase;

#[derive(Clone, Debug)]
struct TinyGatePlan {
    // Part ordering matches the direct-to-merged plan:
    // part0=poseidon, part1=base_glue, parts2..=extra glues (canonical shards + CM shards).
    // These offsets are in the merged variable space and exclude var0.
    var_tail_off: Vec<usize>,
    // Offsets into merged term pools and constraint rows (in counts, not bytes), per part.
    row_off: Vec<u64>,
    a_off: Vec<u64>,
    b_off: Vec<u64>,
    c_off: Vec<u64>,
    // Equality constraints appended after all parts (in merged var space).
    eq_pairs: Vec<(usize, usize)>,
    // Totals for parts only (no eq tail).
    part_rows: u64,
    part_a_terms: u64,
    part_b_terms: u64,
    part_c_terms: u64,
    // Totals including eq tail.
    total_rows: u64,
    total_a_terms: u64,
    total_b_terms: u64,
    total_c_terms: u64,
}

fn tiny_gate_poseidon_shard_permutes(cfg: &PoseidonConfig<F257>, ops: &[PoseidonTraceOp<F257>]) -> usize {
    let n_threads = rayon::current_num_threads().max(1);
    let total_permutes = count_permutes_for_ops(cfg, ops);
    // Important for throughput: do NOT cap shard count at a small constant.
    // We want Poseidon (Pass0 count + Pass1 range-write) to scale with available cores.
    //
    // Shards are further implicitly capped by the `max(1024)` below, so tiny traces won't spawn
    // a silly number of shards.
    let target_shards = n_threads.min(256).max(2);
    let shard_permutes = (total_permutes + target_shards - 1) / target_shards;
    shard_permutes.max(1024)
}

fn build_count_plan(
    cfg: &PoseidonConfig<F257>,
    ops: &[PoseidonTraceOp<F257>],
    ring_dim: usize,
    params: &WeParams,
    wiring: &TinyCoinOpWiring,
    pairs: &[(usize, usize)],
    extra_witness: &TinyExtraWitness,
) -> Result<TinyGatePlan, String> {
    let t_all = Instant::now();
    // Poseidon (count-only sharded, no disk writes).
    let t = Instant::now();
    let shard_permutes = tiny_gate_poseidon_shard_permutes(cfg, ops);
    let (pose_asg, pose_wiring, pose_counts) =
        poseidon_sponge_dr1cs_from_ops_with_wiring_no_bytes_count_sharded::<F257>(cfg, ops, shard_permutes)
            .map_err(|e| format!("poseidon(F257) count-only sharded arith failed: {e}"))?;
    let pose_asg = pose_asg;
    lf_profile_log(&format!(
        "Pass0 poseidon_count elapsed={:?} shard_permutes={}",
        t.elapsed(),
        shard_permutes
    ));

    let t = Instant::now();
    let short_ranges =
        squeeze_field_ranges_by_op_index(&pose_wiring.squeeze_field_ranges, &wiring.short_squeeze_ops)?;
    let u32_ranges =
        squeeze_field_ranges_by_op_index(&pose_wiring.squeeze_field_ranges, &wiring.u32_squeeze_ops)?;
    validate_pairs(pairs, short_ranges.len(), u32_ranges.len())?;
    validate_params_and_short_schedule(ring_dim, params, short_ranges.len())?;
    lf_profile_log(&format!("Pass0 ranges+validate elapsed={:?}", t.elapsed()));

    // Base glue in count-only mode.
    let t = Instant::now();
    let mut glue = GlueCtx::new(&TinyGateBuildMode::Count, pose_asg.as_slice(), PathBuf::new(), 0, None)?;
    lf_profile_log(&format!("Pass0 base_glue_init elapsed={:?}", t.elapsed()));

    // Canonicality constraints: count-only shards.
    // NOTE: the function expects BuildDirs for naming only; in Count mode it does not write.
    let dirs_dummy = build_dirs("/dev/null");
    let t = Instant::now();
    let canonical_glues = build_canonicality_shards(
        &TinyGateBuildMode::Count,
        2,
        None,
        pose_asg.as_slice(),
        ops,
        &pose_wiring,
        &dirs_dummy,
    )?;
    lf_profile_log(&format!(
        "Pass0 canonicality_shards elapsed={:?} parts={}",
        t.elapsed(),
        canonical_glues.len()
    ));

    let t = Instant::now();
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
    lf_profile_log(&format!(
        "Pass0 cm_schedule+comh_count elapsed={:?} l_instances_expected={}",
        t.elapsed(),
        l_instances_expected
    ));

    let t = Instant::now();
    let short_locals = build_short_blocks(&mut glue, &pose_wiring, ring_dim, &short_ranges)?;
    let (u32_locals, goldilocks_locals) =
        build_u32_and_goldilocks_blocks(&mut glue, &pose_wiring, &wiring.u32_squeeze_ops)?;
    lf_profile_log(&format!(
        "Pass0 build_short/u32/goldilocks elapsed={:?} short_locals={} u32_locals={} goldilocks_locals={}",
        t.elapsed(),
        short_locals.len(),
        u32_locals.len(),
        goldilocks_locals.len()
    ));

    let mut setchk_out_e_vars_for_cm: Option<Vec<Vec<Vec<RingDigits>>>> = None;
    let mut dcom_evals_for_cm: Option<Vec<DcomEvalDigits>> = None;
    let mut fcoms_for_cm: Option<Vec<FComsDigits>> = None;
    let mut setchk_r_point_for_cm: Option<Arc<Vec<GoldilocksScalar>>> = None;

    let t = Instant::now();
    arithmetize_pi_lin_setchk_rgchk_prefix(
        &mut glue,
        ops,
        &pose_wiring,
        ring_dim,
        params,
        wiring,
        l_instances_expected,
        &u32_locals,
        extra_witness,
        &mut setchk_out_e_vars_for_cm,
        &mut dcom_evals_for_cm,
        &mut fcoms_for_cm,
        &mut setchk_r_point_for_cm,
    )?;
    lf_profile_log(&format!("Pass0 pi/lin/setchk/rgchk_prefix elapsed={:?}", t.elapsed()));

    let t = Instant::now();
    let (comh_absorbs, sc_msg_absorbs, eval_absorbs) = parse_and_enforce_cm_after_short(
        &mut glue,
        ops,
        &pose_wiring,
        ring_dim,
        params,
        wiring,
        l_instances_expected,
        &goldilocks_locals,
    )?;
    lf_profile_log(&format!("Pass0 parse+enforce_cm_after_short elapsed={:?}", t.elapsed()));

    let setchk_out_e_vars_for_cm = setchk_out_e_vars_for_cm.map(Arc::new);
    let dcom_evals_for_cm = dcom_evals_for_cm.map(Arc::new);
    let t = Instant::now();
    let cm_shared_base: Arc<CmSharedPrecompBase> = compute_cm_shared_precomp_base(
        &mut glue,
        ring_dim,
        params,
        wiring,
        l_instances_expected,
        &short_locals,
        &u32_locals,
        &setchk_out_e_vars_for_cm,
        &setchk_r_point_for_cm,
    )?;
    lf_profile_log(&format!("Pass0 cm_shared_precomp_base elapsed={:?}", t.elapsed()));

    // CM modules: count-only shards.
    let t = Instant::now();
    let cm_extra_glues = build_cm_shards(
        &TinyGateBuildMode::Count,
        2 + canonical_glues.len(),
        None,
        &pose_wiring,
        ring_dim,
        params,
        wiring,
        l_instances_expected,
        &comh_absorbs,
        &sc_msg_absorbs,
        &eval_absorbs,
        setchk_out_e_vars_for_cm.clone(),
        dcom_evals_for_cm.clone(),
        cm_shared_base.clone(),
        glue.pose_asg,
        glue.gb.assignment.as_slice(),
        &goldilocks_locals,
        &dirs_dummy,
    )?;
    lf_profile_log(&format!(
        "Pass0 cm_shards elapsed={:?} parts={}",
        t.elapsed(),
        cm_extra_glues.len()
    ));

    // Decomp verifier math (LinB2X + DecompProof) — algebraic-only checks.
    //
    // IMPORTANT: bind these checks to the CM-derived reduced instance x = (cm_g, r_o, v_o),
    // not to a prover-supplied copy.
    let (cm_g_target, vo_a_target, vo_b_target) = compute_cm_x_targets_for_decomp(
        &mut glue,
        ring_dim,
        params,
        l_instances_expected,
        &cm_shared_base,
        &pose_wiring,
        &comh_absorbs,
        &eval_absorbs,
        &fcoms_for_cm,
    )?;
    lf_profile_log(&format!("Pass0 cm_x_targets_for_decomp elapsed={:?}", t_all.elapsed()));
    add_decomp_linb2x_constraints(
        &mut glue,
        ring_dim,
        params,
        extra_witness,
        &cm_g_target,
        &vo_a_target,
        &vo_b_target,
    )?;

    // Surfaces (small vs Poseidon/CM, but must be counted for exact Pass0 sizing).
    let _ = build_surfaces_with_shared_arcs(&mut glue, ring_dim, pairs, &short_locals, &u32_locals)?;

    // Assemble the list of "extra glues" in canonical order: canonical shards + cm shards.
    let mut extra_glues: Vec<GlueCtx<'_>> = Vec::new();
    extra_glues.extend(canonical_glues);
    extra_glues.extend(cm_extra_glues);

    // Compute part offsets in merged space (excluding var0).
    // Part 0: poseidon, part 1: base glue, parts 2..: extra glue modules.
    let mut offsets: Vec<usize> = Vec::with_capacity(2 + extra_glues.len());
    let mut cur = 0usize;
    offsets.push(cur);
    cur += pose_asg.len().saturating_sub(1);
    offsets.push(cur);
    cur += glue.gb.assignment.len().saturating_sub(1);
    for g in &extra_glues {
        offsets.push(cur);
        cur += g.gb.assignment.len().saturating_sub(1);
    }
    let remap = |part: usize, local: usize, offsets: &[usize]| -> usize {
        if local == 0 { 0 } else { local + offsets[part] }
    };

    // Equality constraints: same construction as in Pass1 direct-to-merged.
    let mut eq_pairs: Vec<(usize, usize)> = Vec::new();
    {
        let reabsorb = collect_fiat_shamir_reabsorb_eqs(ops, &pose_wiring)?;
        eq_pairs.reserve(reabsorb.len());
        for (v_ab, v_sq) in reabsorb {
            eq_pairs.push((remap(0, v_ab, &offsets), remap(0, v_sq, &offsets)));
        }
    }
    // glue links between copied pose vars and module-local vars
    for (&gv, &lv) in glue.local_map.iter() {
        eq_pairs.push((remap(0, gv, &offsets), remap(1, lv, &offsets)));
    }
    for (i, m) in extra_glues.iter().enumerate() {
        let part = 2 + i;
        for (&gv, &lv) in m.local_map.iter() {
            eq_pairs.push((remap(0, gv, &offsets), remap(part, lv, &offsets)));
        }
    }
    for (i, g) in extra_glues.iter().enumerate() {
        let part = 2 + i;
        for &(base_var, local_var) in &g.base_eqs {
            eq_pairs.push((remap(1, base_var, &offsets), remap(part, local_var, &offsets)));
        }
    }

    // Gather per-part counts and compute pool offsets.
    let mut row_off: Vec<u64> = Vec::with_capacity(2 + extra_glues.len());
    let mut a_off: Vec<u64> = Vec::with_capacity(2 + extra_glues.len());
    let mut b_off: Vec<u64> = Vec::with_capacity(2 + extra_glues.len());
    let mut c_off: Vec<u64> = Vec::with_capacity(2 + extra_glues.len());
    let mut cur_rows: u64 = 0;
    let mut cur_a: u64 = 0;
    let mut cur_b: u64 = 0;
    let mut cur_c: u64 = 0;

    // Poseidon counts
    row_off.push(cur_rows);
    a_off.push(cur_a);
    b_off.push(cur_b);
    c_off.push(cur_c);
    cur_rows = cur_rows.saturating_add(pose_counts.rows);
    cur_a = cur_a.saturating_add(pose_counts.a_terms);
    cur_b = cur_b.saturating_add(pose_counts.b_terms);
    cur_c = cur_c.saturating_add(pose_counts.c_terms);

    // Base glue counts (Count mode: read counters without consuming the builder).
    let (base_rows, base_a_terms, base_b_terms, base_c_terms) = glue
        .gb
        .file_counts()
        .ok_or("tiny gate: expected file_counts in base glue count mode")?;
    row_off.push(cur_rows);
    a_off.push(cur_a);
    b_off.push(cur_b);
    c_off.push(cur_c);
    cur_rows = cur_rows.saturating_add(base_rows);
    cur_a = cur_a.saturating_add(base_a_terms);
    cur_b = cur_b.saturating_add(base_b_terms);
    cur_c = cur_c.saturating_add(base_c_terms);

    // Extra glues counts
    for g in &extra_glues {
        let (rows, a_terms, b_terms, c_terms) = g
            .gb
            .file_counts()
            .ok_or("tiny gate: expected file_counts in extra glue count mode")?;
        row_off.push(cur_rows);
        a_off.push(cur_a);
        b_off.push(cur_b);
        c_off.push(cur_c);
        cur_rows = cur_rows.saturating_add(rows);
        cur_a = cur_a.saturating_add(a_terms);
        cur_b = cur_b.saturating_add(b_terms);
        cur_c = cur_c.saturating_add(c_terms);
    }

    let part_rows = cur_rows;
    let part_a_terms = cur_a;
    let part_b_terms = cur_b;
    let part_c_terms = cur_c;
    let eqs = eq_pairs.len() as u64;
    let total_rows = part_rows.saturating_add(eqs);
    let total_a_terms = part_a_terms.saturating_add(2 * eqs);
    let total_b_terms = part_b_terms.saturating_add(eqs);
    let total_c_terms = part_c_terms.saturating_add(eqs);

    let plan = TinyGatePlan {
        var_tail_off: offsets,
        row_off,
        a_off,
        b_off,
        c_off,
        eq_pairs,
        part_rows,
        part_a_terms,
        part_b_terms,
        part_c_terms,
        total_rows,
        total_a_terms,
        total_b_terms,
        total_c_terms,
    };

    Ok(plan)
}

#[cfg(unix)]
fn prealloc_merged_files(
    merged_dir: &Path,
    total_a_terms: u64,
    total_b_terms: u64,
    total_c_terms: u64,
    total_rows: u64,
) -> Result<
    (
        std::fs::File,
        std::fs::File,
        std::fs::File,
        std::fs::File,
        std::fs::File,
        std::fs::File,
        std::fs::File,
    ),
    String,
> {
    use std::fs::OpenOptions;
    use std::io::{Seek, SeekFrom};

    // Clear any previous output quickly (rename+rm in background).
    crate::fs_cleanup::fast_remove_dir_best_effort_to_tmp(merged_dir);
    std::fs::create_dir_all(merged_dir).map_err(|e| format!("create merged_dir failed: {e}"))?;

    let open_rw = |p: &Path| -> Result<std::fs::File, String> {
        OpenOptions::new()
            .create(true)
            .read(true)
            .write(true)
            .truncate(true)
            .open(p)
            .map_err(|e| format!("open {p:?} failed: {e}"))
    };
    let mut fc_a = open_rw(&merged_dir.join("a_coeffs.bin"))?;
    let mut fi_a = open_rw(&merged_dir.join("a_idx.bin"))?;
    let mut fc_b = open_rw(&merged_dir.join("b_coeffs.bin"))?;
    let mut fi_b = open_rw(&merged_dir.join("b_idx.bin"))?;
    let mut fc_c = open_rw(&merged_dir.join("c_coeffs.bin"))?;
    let mut fi_c = open_rw(&merged_dir.join("c_idx.bin"))?;
    let mut f_rows = open_rw(&merged_dir.join("constraints.bin"))?;

    fc_a.set_len(total_a_terms.saturating_mul(2)).map_err(|e| e.to_string())?;
    fi_a.set_len(total_a_terms.saturating_mul(4)).map_err(|e| e.to_string())?;
    fc_b.set_len(total_b_terms.saturating_mul(2)).map_err(|e| e.to_string())?;
    fi_b.set_len(total_b_terms.saturating_mul(4)).map_err(|e| e.to_string())?;
    fc_c.set_len(total_c_terms.saturating_mul(2)).map_err(|e| e.to_string())?;
    fi_c.set_len(total_c_terms.saturating_mul(4)).map_err(|e| e.to_string())?;
    f_rows
        .set_len(total_rows.saturating_mul(12))
        .map_err(|e| e.to_string())?;

    let _ = fc_a.seek(SeekFrom::End(0));
    let _ = fi_a.seek(SeekFrom::End(0));
    let _ = fc_b.seek(SeekFrom::End(0));
    let _ = fi_b.seek(SeekFrom::End(0));
    let _ = fc_c.seek(SeekFrom::End(0));
    let _ = fi_c.seek(SeekFrom::End(0));
    let _ = f_rows.seek(SeekFrom::End(0));

    Ok((fc_a, fi_a, fc_b, fi_b, fc_c, fi_c, f_rows))
}

impl<'a> GlueCtx<'a> {
    fn new(
        mode: &TinyGateBuildMode,
        pose_asg: &'a [F257],
        _out_dir: impl AsRef<Path>,
        part_idx: usize,
        range: Option<&TinyGateRangeBase>,
    ) -> Result<Self, String> {
        let mut gb = match mode {
            TinyGateBuildMode::Count => Dr1csBuilder::<F257>::new_count_only(),
            #[cfg(unix)]
            TinyGateBuildMode::RangeBase => {
                let rb = range.ok_or("tiny gate: missing range base for RangeBase mode")?;
                let part = rb
                    .part_base
                    .checked_add(part_idx)
                    .ok_or("tiny gate: part idx overflow (range base)")?;
                if part >= rb.plan.row_off.len()
                    || part >= rb.plan.a_off.len()
                    || part >= rb.plan.b_off.len()
                    || part >= rb.plan.c_off.len()
                    || part >= rb.plan.var_tail_off.len()
                {
                    return Err("tiny gate: part idx out of range for plan (range base)".to_string());
                }
                let (fc_a, fi_a, fc_b, fi_b, fc_c, fi_c, f_rows) = rb.files.try_clone_all()?;
                let writer = FileBackedRangeWriter::new(
                    fc_a,
                    fi_a,
                    fc_b,
                    fi_b,
                    fc_c,
                    fi_c,
                    f_rows,
                    rb.plan.a_off[part],
                    rb.plan.b_off[part],
                    rb.plan.c_off[part],
                    rb.plan.row_off[part],
                );
                let var_tail_off: u32 = (rb.plan.var_tail_off[part] as u64)
                    .try_into()
                    .map_err(|_| "tiny gate: var_tail_off overflow u32".to_string())?;
                Dr1csBuilder::<F257>::new_file_backed_range(writer, var_tail_off)
            }
        };
        // var0 is the constant-1 slot
        gb.enforce_var_eq_const(gb.one(), F257::ONE);
        Ok(Self {
            gb,
            pose_asg,
            local_map: BTreeMap::new(),
            base_map: BTreeMap::new(),
            base_eqs: Vec::new(),
        })
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

#[derive(Clone, Debug)]
struct BuildDirs {
    root: PathBuf,
    base_glue_dir: PathBuf,
    merged_dir: PathBuf,
    cm0_dir: PathBuf,
    cm1_dir: PathBuf,
}

fn build_dirs(out_dir: impl AsRef<Path>) -> BuildDirs {
    let root: PathBuf = out_dir.as_ref().to_path_buf();
    BuildDirs {
        base_glue_dir: root.join("base_glue"),
        merged_dir: root.join("merged"),
        cm0_dir: root.join("cm0"),
        cm1_dir: root.join("cm1"),
        root,
    }
}

fn build_canonicality_shards<'p>(
    mode: &TinyGateBuildMode,
    part_base: usize,
    range: Option<&TinyGateRangeBase>,
    pose_asg: &'p [F257],
    ops: &[PoseidonTraceOp<F257>],
    pose_wiring: &PoseidonDr1csWiring,
    dirs: &BuildDirs,
) -> Result<Vec<GlueCtx<'p>>, String> {
    let canonical_ranges = collect_nonreabsorb_absorb_ranges(ops, pose_wiring)?;
    if canonical_ranges.is_empty() {
        return Ok(Vec::new());
    }
    let n_threads = rayon::current_num_threads().max(1);
    let n_chunks = std::env::var("LFP_TINY_CANON_CHUNKS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|&n| n > 0)
        .unwrap_or_else(|| {
            let target_chunk_ranges: usize = 256;
            let by_work = (canonical_ranges.len() + target_chunk_ranges - 1) / target_chunk_ranges;
            by_work.max(1).min((n_threads * 2).min(256).max(1))
        })
        .min(canonical_ranges.len().max(1));
    let chunk_size = (canonical_ranges.len() + n_chunks - 1) / n_chunks;
    canonical_ranges
        .chunks(chunk_size.max(1))
        .enumerate()
        .collect::<Vec<_>>()
        .into_par_iter()
        .map(|(idx, chunk)| -> Result<GlueCtx<'p>, String> {
            let dir = dirs.root.join(format!("canon_{idx}"));
            let mut g = GlueCtx::new(mode, pose_asg, dir, part_base + idx, range)?;
            enforce_canonical_goldilocks_for_ranges(&mut g, pose_wiring, chunk)?;
            Ok(g)
        })
        .collect::<Result<Vec<_>, _>>()
}

fn add_decomp_linb2x_constraints(
    glue: &mut GlueCtx,
    ring_dim: usize,
    params: &WeParams,
    extra_witness: &TinyExtraWitness,
    cm_g_target: &[RingDigits],
    vo_a_target: &[RingDigits],
    vo_b_target: &[RingDigits],
) -> Result<(), String> {
    // Mirrors `we_gate_arith.rs::decomp_verifier_math_dr1cs`:
    // - C0 + B*C1 == cm_g
    // - v0a + B*v1a == va, v0b + B*v1b == vb
    //
    // These do not interact with the transcript; they are purely ring algebraic checks.
    // The witness-time builder can populate these from a real proof via `TinyExtraWitness`.
    let kappa = params.kappa as usize;
    if kappa == 0 {
        return Ok(());
    }

    let p_u64 = crate::we_goldilocks_poseidon_f257::GOLDILOCKS_P;
    let b_u64: u64 = ((params.decomp_b as u128) % (p_u64 as u128)) as u64;
    let vlen = 1usize + (params.mlen as usize);
    if cm_g_target.len() != kappa {
        return Err("tiny gate: cm_g_target length mismatch (decomp)".to_string());
    }
    if vo_a_target.len() != vlen || vo_b_target.len() != vlen {
        return Err("tiny gate: vo_target length mismatch (decomp)".to_string());
    }

    #[inline]
    fn alloc_witness_ring_digits(
        gb: &mut Dr1csBuilder<F257>,
        ring_dim: usize,
        coeffs: Option<&[u64]>,
    ) -> RingDigits {
        let mut r: RingDigits = Vec::with_capacity(ring_dim);
        for j in 0..ring_dim {
            let u = coeffs.and_then(|c| c.get(j).copied()).unwrap_or(0u64);
            r.push(alloc_witness_goldilocks_u64_digits(gb, u));
        }
        r
    }

    #[inline]
    fn ring_recompose_base_b(gb: &mut Dr1csBuilder<F257>, r0: &RingDigits, r1: &RingDigits, b_u64: u64) -> RingDigits {
        debug_assert_eq!(r0.len(), r1.len());
        let mut out: RingDigits = Vec::with_capacity(r0.len());
        for j in 0..r0.len() {
            // Use a const-mul gadget here: b is a fixed parameter in WE.
            let t = super::cm_math::goldilocks_mul_const_mod_p_digits(gb, &r1[j], b_u64);
            let s = goldilocks_add_mod_p_digits(gb, &r0[j], &t);
            out.push(s);
        }
        out
    }

    // Allocate witnesses from the provided extra witness (dummy for shape, real for proof).
    let mut dcomp_c0: Vec<RingDigits> = Vec::with_capacity(kappa);
    let mut dcomp_c1: Vec<RingDigits> = Vec::with_capacity(kappa);
    for i in 0..kappa {
        let c0 = extra_witness.decomp_c0.get(i).map(|v| v.as_slice());
        let c1 = extra_witness.decomp_c1.get(i).map(|v| v.as_slice());
        dcomp_c0.push(alloc_witness_ring_digits(&mut glue.gb, ring_dim, c0));
        dcomp_c1.push(alloc_witness_ring_digits(&mut glue.gb, ring_dim, c1));
    }

    let mut v0a: Vec<RingDigits> = Vec::with_capacity(vlen);
    let mut v0b: Vec<RingDigits> = Vec::with_capacity(vlen);
    let mut v1a: Vec<RingDigits> = Vec::with_capacity(vlen);
    let mut v1b: Vec<RingDigits> = Vec::with_capacity(vlen);
    for i in 0..vlen {
        let v0a_i = extra_witness.decomp_v0a.get(i).map(|v| v.as_slice());
        let v0b_i = extra_witness.decomp_v0b.get(i).map(|v| v.as_slice());
        let v1a_i = extra_witness.decomp_v1a.get(i).map(|v| v.as_slice());
        let v1b_i = extra_witness.decomp_v1b.get(i).map(|v| v.as_slice());
        v0a.push(alloc_witness_ring_digits(&mut glue.gb, ring_dim, v0a_i));
        v0b.push(alloc_witness_ring_digits(&mut glue.gb, ring_dim, v0b_i));
        v1a.push(alloc_witness_ring_digits(&mut glue.gb, ring_dim, v1a_i));
        v1b.push(alloc_witness_ring_digits(&mut glue.gb, ring_dim, v1b_i));
    }

    // Enforce recomposition equalities.
    for i in 0..kappa {
        let rec = ring_recompose_base_b(&mut glue.gb, &dcomp_c0[i], &dcomp_c1[i], b_u64);
        ring_eq_digits(&mut glue.gb, &rec, &cm_g_target[i]);
    }
    for i in 0..vlen {
        let rec_a = ring_recompose_base_b(&mut glue.gb, &v0a[i], &v1a[i], b_u64);
        let rec_b = ring_recompose_base_b(&mut glue.gb, &v0b[i], &v1b[i], b_u64);
        ring_eq_digits(&mut glue.gb, &rec_a, &vo_a_target[i]);
        ring_eq_digits(&mut glue.gb, &rec_b, &vo_b_target[i]);
    }

    Ok(())
}

fn build_cm_shards<'p>(
    mode: &TinyGateBuildMode,
    part_base: usize,
    range: Option<&TinyGateRangeBase>,
    pose_wiring: &PoseidonDr1csWiring,
    ring_dim: usize,
    params: &WeParams,
    wiring: &TinyCoinOpWiring,
    l_instances_expected: usize,
    comh_absorbs: &[(usize, usize)],
    sc_msg_absorbs: &[Vec<(usize, usize)>],
    eval_absorbs: &[Vec<(usize, usize)>],
    setchk_out_e_vars_for_cm: Option<Arc<Vec<Vec<Vec<RingDigits>>>>>,
    dcom_evals_for_cm: Option<Arc<Vec<DcomEvalDigits>>>,
    cm_shared_base: Arc<CmSharedPrecompBase>,
    pose_asg: &'p [F257],
    base_asg: &[F257],
    goldilocks_locals: &[GoldilocksChallengeWiring],
    dirs: &BuildDirs,
) -> Result<Vec<GlueCtx<'p>>, String> {
    if !(ring_dim > 0 && l_instances_expected > 0 && !comh_absorbs.is_empty()) {
        return Ok(Vec::new());
    }
    let (g0, g1) = join(
        || {
            build_cm_glue_for_which(
                mode,
                part_base + 0,
                range,
                pose_wiring,
                ring_dim,
                params,
                wiring,
                l_instances_expected,
                comh_absorbs,
                &sc_msg_absorbs[0],
                &eval_absorbs[0],
                setchk_out_e_vars_for_cm.clone(),
                dcom_evals_for_cm.clone(),
                cm_shared_base.clone(),
                0,
                pose_asg,
                base_asg,
                goldilocks_locals,
                dirs.cm0_dir.as_path(),
            )
        },
        || {
            build_cm_glue_for_which(
                mode,
                part_base + 1,
                range,
                pose_wiring,
                ring_dim,
                params,
                wiring,
                l_instances_expected,
                comh_absorbs,
                &sc_msg_absorbs[1],
                &eval_absorbs[1],
                setchk_out_e_vars_for_cm.clone(),
                dcom_evals_for_cm.clone(),
                cm_shared_base.clone(),
                1,
                pose_asg,
                base_asg,
                goldilocks_locals,
                dirs.cm1_dir.as_path(),
            )
        },
    );
    let g0 = g0?;
    let g1 = g1?;

    // If dR1CS profiling is enabled, print per-shard scope summaries too (helps localize bugs
    // inside cm0/cm1 rather than only seeing the base glue profile).
    if g0.gb.profile_enabled {
        eprintln!(
            "== dR1CS profile: cm0 shard (vars={} constraints={}) ==",
            g0.gb.assignment.len().saturating_sub(1),
            g0.gb.nconstraints()
        );
        eprintln!("{}", g0.gb.profile_report(30));
    }
    if g1.gb.profile_enabled {
        eprintln!(
            "== dR1CS profile: cm1 shard (vars={} constraints={}) ==",
            g1.gb.assignment.len().saturating_sub(1),
            g1.gb.nconstraints()
        );
        eprintln!("{}", g1.gb.profile_report(30));
    }

    Ok(vec![g0, g1])
}

fn compute_cm_x_targets_for_decomp(
    glue: &mut GlueCtx,
    ring_dim: usize,
    params: &WeParams,
    l_instances_expected: usize,
    cm_shared_base: &Arc<CmSharedPrecompBase>,
    pose_wiring: &PoseidonDr1csWiring,
    comh_absorbs: &[(usize, usize)],
    eval_absorbs: &[Vec<(usize, usize)>],
    fcoms_for_cm: &Option<Vec<FComsDigits>>,
) -> Result<(Vec<RingDigits>, Vec<RingDigits>, Vec<RingDigits>), String> {
    let kappa = params.kappa as usize;
    if kappa == 0 {
        return Ok((Vec::new(), Vec::new(), Vec::new()));
    }
    let rows_per_l = 1usize + (params.mlen as usize);
    let rows_total = l_instances_expected
        .checked_mul(rows_per_l)
        .ok_or_else(|| "tiny gate: rows_total overflow (cm x targets)".to_string())?;
    if eval_absorbs.len() != 2 {
        return Err("tiny gate: expected eval_absorbs[2] (cm x targets)".to_string());
    }
    if eval_absorbs[0].len() != rows_total * 4 || eval_absorbs[1].len() != rows_total * 4 {
        return Err("tiny gate: eval_absorbs length mismatch (cm x targets)".to_string());
    }
    let fcoms = fcoms_for_cm
        .as_ref()
        .ok_or_else(|| "tiny gate: missing fcoms_for_cm (cm x targets)".to_string())?;
    if fcoms.len() != l_instances_expected {
        return Err("tiny gate: fcoms_for_cm length mismatch (cm x targets)".to_string());
    }
    if comh_absorbs.len() != l_instances_expected * kappa {
        return Err("tiny gate: comh_absorbs length mismatch (cm x targets)".to_string());
    }
    let s_rings: &[RingDigits; 3] = &cm_shared_base.s_rings;
    for i in 0..3 {
        if s_rings[i].len() != ring_dim {
            return Err("tiny gate: cm_shared_base.s_rings len mismatch".to_string());
        }
    }

    // Accumulate folded (over l) cm_g and vo.
    let mut cm_g_acc: Vec<RingDigits> = (0..kappa)
        .map(|_| super::cm_math::ring_zero_digits(&mut glue.gb, ring_dim))
        .collect();
    let mut vo_a_acc: Vec<RingDigits> = (0..rows_per_l)
        .map(|_| super::cm_math::ring_zero_digits(&mut glue.gb, ring_dim))
        .collect();
    let mut vo_b_acc: Vec<RingDigits> = (0..rows_per_l)
        .map(|_| super::cm_math::ring_zero_digits(&mut glue.gb, ring_dim))
        .collect();

    for l in 0..l_instances_expected {
        // Parse com(h)[l][*] from transcript absorbs.
        let mut comh_l: Vec<RingDigits> = Vec::with_capacity(kappa);
        for j in 0..kappa {
            let (st, ln) = comh_absorbs[l * kappa + j];
            let rb = parse_ring_elem_absorb_as_ringbytes(glue, pose_wiring, ring_dim, st, ln)?;
            comh_l.push(ring_bytes_to_digits(&mut glue.gb, &rb));
        }

        // cm_g[l][j] = s0*C_Mf + s1*cm_mtau + s2*cm_f + com(h).
        let fc = &fcoms[l];
        if fc.cm_f.len() != kappa || fc.c_mf.len() != kappa || fc.cm_mtau.len() != kappa {
            return Err("tiny gate: fcoms entry length mismatch (cm x targets)".to_string());
        }

        // vo[l][row] = (s0*e0 + s1*e1 + s2*e2 + e3) for each sumcheck (which=0,1).
        //
        // Performance note: The number of ring multiplications here is large (3*kappa + 6*rows_per_l per `l`).
        // To improve wall-clock time we build the ring-mul IR shards in parallel, then lower sequentially.
        #[derive(Clone, Copy, Debug)]
        enum MulKind {
            Cmg { j: usize, term: usize },     // term in 0..3
            Vo { which: usize, row: usize, term: usize }, // term in 0..3 (only 0..3 where 3 is unused)
        }

        // Parse evals for this `l` once (needs &mut glue).
        let mut eval0: Vec<[RingDigits; 4]> = Vec::with_capacity(rows_per_l);
        let mut eval1: Vec<[RingDigits; 4]> = Vec::with_capacity(rows_per_l);
        for row in 0..rows_per_l {
            let flat = l * rows_per_l + row;
            let idx0 = flat * 4;
            let idx1 = idx0 + 1;
            let idx2 = idx0 + 2;
            let idx3 = idx0 + 3;

            let mut parse_eval = |which: usize, idx: usize| -> Result<RingDigits, String> {
                let (st, ln) = eval_absorbs[which][idx];
                let rb = parse_ring_elem_absorb_as_ringbytes(glue, pose_wiring, ring_dim, st, ln)?;
                Ok(ring_bytes_to_digits(&mut glue.gb, &rb))
            };
            eval0.push([
                parse_eval(0, idx0)?,
                parse_eval(0, idx1)?,
                parse_eval(0, idx2)?,
                parse_eval(0, idx3)?,
            ]);
            eval1.push([
                parse_eval(1, idx0)?,
                parse_eval(1, idx1)?,
                parse_eval(1, idx2)?,
                parse_eval(1, idx3)?,
            ]);
        }

        // Build all ring-mul tasks for this `l`.
        let mut tasks: Vec<MulKind> = Vec::with_capacity(3 * kappa + 6 * rows_per_l);
        for j in 0..kappa {
            tasks.push(MulKind::Cmg { j, term: 0 }); // s0 * C_Mf
            tasks.push(MulKind::Cmg { j, term: 1 }); // s1 * cm_mtau
            tasks.push(MulKind::Cmg { j, term: 2 }); // s2 * cm_f
        }
        for row in 0..rows_per_l {
            for term in 0..3 {
                tasks.push(MulKind::Vo { which: 0, row, term });
                tasks.push(MulKind::Vo { which: 1, row, term });
            }
        }

        // Snapshot assignment for parallel IR building.
        //
        // IMPORTANT: keep this as a normal shared slice borrow so Rust prevents concurrent mutation
        // of `glue.gb.assignment` while Rayon threads are reading witness values.
        let base_asg: &[F257] = &glue.gb.assignment;

        use super::cm_ir::{lower_ir_into_builder, ring_mul_negacyclic_ntt_goldilocks_d64_ir, IrBuilder, VarRef as IrVarRef};
        #[inline]
        fn ringdigits64_to_ir(a: &RingDigits) -> Result<[[IrVarRef; 17]; 64], String> {
            if a.len() != 64 {
                return Err("tiny gate: expected ring element with 64 coeffs (cm x targets mul)".to_string());
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

        let frags: Vec<(MulKind, super::cm_ir::CmIr, [[IrVarRef; 17]; 64])> = tasks
            .into_par_iter()
            .map(|kind| -> Result<_, String> {
                let mut ib = if glue.gb.is_count_only() {
                    IrBuilder::new_count_only(base_asg)
                } else {
                    IrBuilder::new(base_asg)
                };

                let (lhs, rhs): (&RingDigits, &RingDigits) = match kind {
                    MulKind::Cmg { j, term: 0 } => (&s_rings[0], &fc.c_mf[j]),
                    MulKind::Cmg { j, term: 1 } => (&s_rings[1], &fc.cm_mtau[j]),
                    MulKind::Cmg { j, term: 2 } => (&s_rings[2], &fc.cm_f[j]),
                    MulKind::Cmg { .. } => return Err("tiny gate: invalid cmg mul term".to_string()),
                    MulKind::Vo { which: 0, row, term } => (&s_rings[term], &eval0[row][term]),
                    MulKind::Vo { which: 1, row, term } => (&s_rings[term], &eval1[row][term]),
                    MulKind::Vo { .. } => return Err("tiny gate: invalid vo mul kind".to_string()),
                };

                let lhs_ir = ringdigits64_to_ir(lhs)?;
                let rhs_ir = ringdigits64_to_ir(rhs)?;
                let out_ir = ring_mul_negacyclic_ntt_goldilocks_d64_ir(&mut ib, &lhs_ir, &rhs_ir);
                Ok((kind, ib.ir, out_ir))
            })
            .collect::<Result<Vec<_>, _>>()?;

        // Lower sequentially and assemble products.
        let mut cmg_mul: Vec<[Option<RingDigits>; 3]> = vec![[None, None, None]; kappa];
        let mut vo_mul0: Vec<[Option<RingDigits>; 3]> = vec![[None, None, None]; rows_per_l];
        let mut vo_mul1: Vec<[Option<RingDigits>; 3]> = vec![[None, None, None]; rows_per_l];
        super::op_counts::tiny_cm_bump(|c| c.ring_mul_negacyclic += frags.len() as u64);

        for (kind, ir, out_ir) in frags {
            let lowered = lower_ir_into_builder(&mut glue.gb, ir);
            let out = map_ring_out(&out_ir, &lowered);
            match kind {
                MulKind::Cmg { j, term } => {
                    cmg_mul[j][term] = Some(out);
                }
                MulKind::Vo { which: 0, row, term } => {
                    vo_mul0[row][term] = Some(out);
                }
                MulKind::Vo { which: 1, row, term } => {
                    vo_mul1[row][term] = Some(out);
                }
                _ => {}
            }
        }

        for j in 0..kappa {
            let t0 = cmg_mul[j][0].take().ok_or("tiny gate: missing cmg mul term0")?;
            let t1 = cmg_mul[j][1].take().ok_or("tiny gate: missing cmg mul term1")?;
            let t2 = cmg_mul[j][2].take().ok_or("tiny gate: missing cmg mul term2")?;
            let sum = ring_add_digits(&mut glue.gb, &t0, &t1);
            let sum = ring_add_digits(&mut glue.gb, &sum, &t2);
            let sum = ring_add_digits(&mut glue.gb, &sum, &comh_l[j]);
            cm_g_acc[j] = ring_add_digits(&mut glue.gb, &cm_g_acc[j], &sum);
        }

        for row in 0..rows_per_l {
            let a0 = vo_mul0[row][0].take().ok_or("tiny gate: missing vo0 mul term0")?;
            let a1 = vo_mul0[row][1].take().ok_or("tiny gate: missing vo0 mul term1")?;
            let a2 = vo_mul0[row][2].take().ok_or("tiny gate: missing vo0 mul term2")?;
            let a = ring_add_digits(&mut glue.gb, &a0, &a1);
            let a = ring_add_digits(&mut glue.gb, &a, &a2);
            let a = ring_add_digits(&mut glue.gb, &a, &eval0[row][3]);
            vo_a_acc[row] = ring_add_digits(&mut glue.gb, &vo_a_acc[row], &a);

            let b0 = vo_mul1[row][0].take().ok_or("tiny gate: missing vo1 mul term0")?;
            let b1 = vo_mul1[row][1].take().ok_or("tiny gate: missing vo1 mul term1")?;
            let b2 = vo_mul1[row][2].take().ok_or("tiny gate: missing vo1 mul term2")?;
            let b = ring_add_digits(&mut glue.gb, &b0, &b1);
            let b = ring_add_digits(&mut glue.gb, &b, &b2);
            let b = ring_add_digits(&mut glue.gb, &b, &eval1[row][3]);
            vo_b_acc[row] = ring_add_digits(&mut glue.gb, &vo_b_acc[row], &b);
        }
    }

    Ok((cm_g_acc, vo_a_acc, vo_b_acc))
}

fn build_surfaces_with_shared_arcs(
    glue: &mut GlueCtx,
    ring_dim: usize,
    pairs: &[(usize, usize)],
    short_locals: &[ShortChallengeWiring],
    u32_locals: &[BoundedU32ChallengeWiring],
) -> Result<
    (
        Vec<CmDigitMulSurfaceWiring>,
        Vec<CmDigitMulSqSurfaceWiring>,
        Arc<Vec<usize>>,
        Arc<Vec<Vec<usize>>>,
        Arc<Vec<usize>>,
        Arc<Vec<Vec<usize>>>,
    ),
    String,
> {
    let t_surfaces = Instant::now();
    eprintln!(
        "  cm surfaces(shared arcs): start (ring_dim={} pairs={} short_locals={} u32_locals={})",
        ring_dim,
        pairs.len(),
        short_locals.len(),
        u32_locals.len()
    );
    let (mut surfaces_mul_local, all_sum_digits, all_sum_coeffwise) =
        build_mul_surfaces(glue, ring_dim, pairs, short_locals, u32_locals)?;
    let (mut surfaces_sq_local, all_sq_sum_digits, all_sq_sum_coeffwise) =
        build_sq_surfaces(glue, ring_dim, pairs, short_locals, u32_locals)?;

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
    eprintln!(
        "  cm surfaces(shared arcs): done in {:?} (mul_surfaces={} sq_surfaces={})",
        t_surfaces.elapsed(),
        surfaces_mul_local.len(),
        surfaces_sq_local.len()
    );
    Ok((
        surfaces_mul_local,
        surfaces_sq_local,
        all_sum_digits,
        all_sum_coeffwise,
        all_sq_sum_digits,
        all_sq_sum_coeffwise,
    ))
}

fn arithmetize_pi_lin_setchk_rgchk_prefix(
    glue: &mut GlueCtx,
    ops: &[PoseidonTraceOp<F257>],
    pose_wiring: &PoseidonDr1csWiring,
    ring_dim: usize,
    params: &WeParams,
    wiring: &TinyCoinOpWiring,
    l_instances_expected: usize,
    u32_locals: &[BoundedU32ChallengeWiring],
    extra_witness: &TinyExtraWitness,
    setchk_out_e_vars_for_cm: &mut Option<Vec<Vec<Vec<RingDigits>>>>,
    dcom_evals_for_cm: &mut Option<Vec<DcomEvalDigits>>,
    fcoms_for_cm: &mut Option<Vec<FComsDigits>>,
    setchk_r_point_for_cm: &mut Option<Arc<Vec<GoldilocksScalar>>>,
) -> Result<(), String> {
    // Mirrors the corresponding block in `build()`; keep verifier-math constraints identical.
    if ring_dim == 0 || wiring.short_squeeze_ops.is_empty() {
        return Ok(());
    }
    let t_all = Instant::now();
    let first_short_op = *wiring
        .short_squeeze_ops
        .iter()
        .min()
        .expect("non-empty short_squeeze_ops");

    let prefix_payload =
        collect_nonreabsorb_absorbs_before_squeeze_field_op(ops, pose_wiring, first_short_op)?;
    let ring_elem_bytes = infer_ring_elem_bytes_from_wiring(ring_dim, pose_wiring)?;
    let nvars_setchk = params.nvars_setchk as usize;

    let n_public_inputs = count_nonreabsorb_absorbs_before_first_squeeze_field_op(ops, pose_wiring)?;
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
    {
        let t_pi = Instant::now();
        use super::cm_ir::{lower_ir_into_builder, ring_mul_negacyclic_ntt_goldilocks_d64_ir, IrBuilder, VarRef as IrVarRef};

        #[inline]
        fn ringdigits64_to_ir(a: &RingDigits) -> Result<[[IrVarRef; 17]; 64], String> {
            if a.len() != 64 {
                return Err("tiny gate: expected ring element with 64 coeffs (Π_lin ring-mul)".to_string());
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
            let (st_nv, ln_nv) = *prefix_payload.get(cur).ok_or("tiny gate: Π_lin header oob")?;
            let (st_deg, ln_deg) = *prefix_payload.get(cur + 1).ok_or("tiny gate: Π_lin header oob")?;
            if ln_nv != 8 || ln_deg != 8 {
                return Err("tiny gate: Π_lin header absorb len mismatch".to_string());
            }
            enforce_absorbed_u64_const(glue, pose_wiring, st_nv, nvars_setchk as u64);
            enforce_absorbed_u64_const(glue, pose_wiring, st_deg, 3u64);
            cur += 2;

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
                let e0b = parse_ring_elem_absorb_as_ringbytes(glue, pose_wiring, ring_dim, s0, l0)?;
                let e1b = parse_ring_elem_absorb_as_ringbytes(glue, pose_wiring, ring_dim, s1, l1)?;
                let e2b = parse_ring_elem_absorb_as_ringbytes(glue, pose_wiring, ring_dim, s2, l2)?;
                let e3b = parse_ring_elem_absorb_as_ringbytes(glue, pose_wiring, ring_dim, s3, l3)?;
                let e0 = ring_bytes_to_digits(&mut glue.gb, &e0b);
                let e1 = ring_bytes_to_digits(&mut glue.gb, &e1b);
                let e2 = ring_bytes_to_digits(&mut glue.gb, &e2b);
                let e3 = ring_bytes_to_digits(&mut glue.gb, &e3b);
                msgs_digits.push([e0, e1, e2, e3]);

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
                }
                for i in 4..8 {
                    let gv = pose_wiring.absorb_vars[rst + i];
                    let lv = glue.copy_digit(gv);
                    glue.gb.enforce_lc_times_one_eq_const(vec![(F257::ONE, lv), (-F257::ONE, z)]);
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

            let mut tail: Vec<RingDigits> = Vec::with_capacity(4);
            for _ in 0..4 {
                let (st, ln) = *prefix_payload.get(cur).ok_or("tiny gate: Π_lin tail oob")?;
                cur += 1;
                if ln != ring_elem_bytes {
                    return Err("tiny gate: Π_lin tail ring absorb len mismatch".to_string());
                }
                let rb = parse_ring_elem_absorb_as_ringbytes(glue, pose_wiring, ring_dim, st, ln)?;
                tail.push(ring_bytes_to_digits(&mut glue.gb, &rb));
            }
            let va = &tail[1];
            let vb = &tail[2];
            let vc = &tail[3];

            let e = super::cm_math::eq_eval_goldilocks_digits(&mut glue.gb, &r_pre_digits, &rs_digits)?;
            // Avoid cloning the full assignment: build IR using an immutable slice, then lower after borrow ends.
            let (ir, prod_ir) = {
                let base_asg: &[F257] = glue.gb.assignment.as_slice();
                let mut ib = if glue.gb.is_count_only() {
                    IrBuilder::new_count_only(base_asg)
                } else {
                    IrBuilder::new(base_asg)
                };
                let va_ir = ringdigits64_to_ir(va)?;
                let vb_ir = ringdigits64_to_ir(vb)?;
                let prod_ir = ring_mul_negacyclic_ntt_goldilocks_d64_ir(&mut ib, &va_ir, &vb_ir);
                (ib.ir, prod_ir)
            };
            super::op_counts::tiny_cm_bump(|c| c.ring_mul_negacyclic += 1);
            let lowered = lower_ir_into_builder(&mut glue.gb, ir);
            let vab = map_ring_out(&prod_ir, &lowered);
            let diff = super::cm_math::ring_sub_digits(&mut glue.gb, &vab, vc);
            // Enforce `diff * e == subclaim_eval` in the **bal4** domain to avoid allocating
            // the scaled ring as bal16 digits (which would require a bal4->bal16 carry-chain conversion
            // per coefficient).
            //
            // This is sound: we convert the existing checked bal16 digits into checked bal4 digits
            // via a carry chain, do a checked bal4 multiplication mod p, then equate bal4 digits.
            {
                use super::cm_ir::{goldilocks_mul_mod_p_digits_bal4_ir, IrBuilder, VarRef as IrVarRef};
                let p_u64 = crate::we_goldilocks_poseidon_f257::GOLDILOCKS_P;
                // Avoid cloning the full assignment: build IR using an immutable slice,
                // then lower after the borrow ends.
                let ir = {
                    let base_asg: &[F257] = glue.gb.assignment.as_slice();
                    let mut ib = if glue.gb.is_count_only() {
                        IrBuilder::new_count_only(base_asg)
                    } else {
                        IrBuilder::new(base_asg)
                    };
                    let e16: [IrVarRef; 17] = core::array::from_fn(|k| IrVarRef::Base(e[k]));
                    let e4 = ib.bal16_to_bal4_digits_cached(&e16);
                    let diff_ir = ringdigits64_to_ir(&diff)?;
                    let sub_ir = ringdigits64_to_ir(&subclaim_eval)?;
                    for i in 0..64 {
                        let d4 = ib.bal16_to_bal4_digits_cached(&diff_ir[i]);
                        let prod4 = goldilocks_mul_mod_p_digits_bal4_ir(&mut ib, &d4, &e4, p_u64);
                        let sub4 = ib.bal16_to_bal4_digits_cached(&sub_ir[i]);
                        for k in 0..33 {
                            ib.enforce_lc_eq_zero(vec![(F257::ONE, prod4[k]), (-F257::ONE, sub4[k])]);
                        }
                    }
                    ib.ir
                };
                let _lowered = super::cm_ir::lower_ir_into_builder(&mut glue.gb, ir);
            }
        }
        lf_profile_log(&format!(
            "prefix Π_lin elapsed={:?} nvars_setchk={} ring_elem_bytes={}",
            t_pi.elapsed(),
            nvars_setchk,
            ring_elem_bytes
        ));
    }

    // Dcom witness commitments: cm_f, C_Mf, cm_mtau (in that order, per instance).
    //
    // These are needed to derive the reduced statement component `cm_g` inside CM verification,
    // so we parse them into ring digits here.
    let kappa = params.kappa as usize;
    let l_instances = l_instances_expected;
    let mut fcoms: Vec<FComsDigits> = Vec::with_capacity(l_instances);
    let t_dcom = Instant::now();
    for l in 0..l_instances {
        let mut cm_f: Vec<RingDigits> = Vec::with_capacity(kappa);
        let mut c_mf: Vec<RingDigits> = Vec::with_capacity(kappa);
        let mut cm_mtau: Vec<RingDigits> = Vec::with_capacity(kappa);
        for _ in 0..kappa {
            let (st, ln) = *prefix_payload.get(cur).ok_or("tiny gate: dcom cm_f absorb oob")?;
            cur += 1;
            if ln != ring_elem_bytes {
                return Err("tiny gate: dcom cm_f ring absorb len mismatch".to_string());
            }
            let rb = parse_ring_elem_absorb_as_ringbytes(glue, pose_wiring, ring_dim, st, ln)?;
            cm_f.push(ring_bytes_to_digits(&mut glue.gb, &rb));
        }

        // Prefix binding: when L=1, bind the first `min(8, kappa)` witness-commitment rows of `cm_f`
        // to the first `min(8, kappa)` public inputs absorbed in the transcript prefix.
        //
        // This matches the LF+ verifier behavior (commit-before-challenge).
        const EXPOSE_MAX: usize = 8;
        let expose_rows = EXPOSE_MAX.min(kappa);
        if expose_rows > 0 {
            if l_instances != 1 {
                return Err(format!(
                    "tiny gate: dcom/rgchk prefix binding requires L=1 (got L={})",
                    l_instances
                ));
            }
            if n_public_inputs < expose_rows {
                return Err(format!(
                    "tiny gate: expected at least {expose_rows} public inputs for prefix binding (got {n_public_inputs})"
                ));
            }
            if l == 0 {
                for j in 0..expose_rows {
                    let (pst, pln) = prefix_payload[j];
                    if pln != 8 {
                        return Err("tiny gate: expected 8-byte public input absorbs in prefix (binding)".to_string());
                    }
                    let bytes_le: [usize; 8] = core::array::from_fn(|i| {
                        let gv = pose_wiring.absorb_vars[pst + i];
                        glue.copy_digit(gv)
                    });
                    let pv_digits = super::cm_math::goldilocks_bytes_to_digits(&mut glue.gb, bytes_le);
                    let pv_ring = super::cm_math::ring_const_coeff_digits(&mut glue.gb, &pv_digits, ring_dim);
                    super::cm_math::ring_eq_digits(&mut glue.gb, &cm_f[j], &pv_ring);
                }
            }
        }

        for _ in 0..kappa {
            let (st, ln) = *prefix_payload.get(cur).ok_or("tiny gate: dcom C_Mf absorb oob")?;
            cur += 1;
            if ln != ring_elem_bytes {
                return Err("tiny gate: dcom C_Mf ring absorb len mismatch".to_string());
            }
            let rb = parse_ring_elem_absorb_as_ringbytes(glue, pose_wiring, ring_dim, st, ln)?;
            c_mf.push(ring_bytes_to_digits(&mut glue.gb, &rb));
        }
        for _ in 0..kappa {
            let (st, ln) = *prefix_payload.get(cur).ok_or("tiny gate: dcom cm_mtau absorb oob")?;
            cur += 1;
            if ln != ring_elem_bytes {
                return Err("tiny gate: dcom cm_mtau ring absorb len mismatch".to_string());
            }
            let rb = parse_ring_elem_absorb_as_ringbytes(glue, pose_wiring, ring_dim, st, ln)?;
            cm_mtau.push(ring_bytes_to_digits(&mut glue.gb, &rb));
        }
        fcoms.push(FComsDigits { cm_f, c_mf, cm_mtau });
    }
    *fcoms_for_cm = Some(fcoms);
    lf_profile_log(&format!(
        "prefix dcom_fcoms elapsed={:?} L={} kappa={}",
        t_dcom.elapsed(),
        l_instances,
        kappa
    ));

    // Next in the prefix payload is the SetChk sumcheck header (nvars, degree=3).
    let t_setchk = Instant::now();
    let start = cur;
    if start + 2 > prefix_payload.len() {
        return Err("tiny gate: prefix too short for SetChk header".to_string());
    }
    enforce_absorbed_u64_const(glue, pose_wiring, prefix_payload[start + 0].0, nvars_setchk as u64);
    enforce_absorbed_u64_const(glue, pose_wiring, prefix_payload[start + 1].0, 3u64);

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

    // Parse the sumcheck prover messages and bind absorbed r_i to u32 coin bytes.
    let mut msgs_digits: Vec<[RingDigits; 4]> = Vec::with_capacity(nvars_setchk);
    let mut rs_digits: Vec<GoldilocksScalar> = Vec::with_capacity(nvars_setchk);
    let z = glue.gb.new_var(F257::ZERO);
    glue.gb.enforce_var_eq_const(z, F257::ZERO);
    let mut cur = start + 2;
    for round in 0..nvars_setchk {
        let (s0, l0) = prefix_payload[cur + 0];
        let (s1, l1) = prefix_payload[cur + 1];
        let (s2, l2) = prefix_payload[cur + 2];
        let (s3, l3) = prefix_payload[cur + 3];
        cur += 4;
        let e0b = parse_ring_elem_absorb_as_ringbytes(glue, pose_wiring, ring_dim, s0, l0)?;
        let e1b = parse_ring_elem_absorb_as_ringbytes(glue, pose_wiring, ring_dim, s1, l1)?;
        let e2b = parse_ring_elem_absorb_as_ringbytes(glue, pose_wiring, ring_dim, s2, l2)?;
        let e3b = parse_ring_elem_absorb_as_ringbytes(glue, pose_wiring, ring_dim, s3, l3)?;
        let e0 = ring_bytes_to_digits(&mut glue.gb, &e0b);
        let e1 = ring_bytes_to_digits(&mut glue.gb, &e1b);
        let e2 = ring_bytes_to_digits(&mut glue.gb, &e2b);
        let e3 = ring_bytes_to_digits(&mut glue.gb, &e3b);
        msgs_digits.push([e0, e1, e2, e3]);

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
        }
        for i in 4..8 {
            let gv = pose_wiring.absorb_vars[rst + i];
            let lv = glue.copy_digit(gv);
            glue.gb.enforce_lc_times_one_eq_const(vec![(F257::ONE, lv), (-F257::ONE, z)]);
        }
        let bytes = goldilocks_bytes_from_u32_le_bytes(
            &mut glue.gb,
            &[u.byte_vars[0], u.byte_vars[1], u.byte_vars[2], u.byte_vars[3]],
        );
        rs_digits.push(goldilocks_bytes_to_digits(&mut glue.gb, bytes));
    }
    // Plumb the setchk verifier point `r` forward for CM `eq(r, ro)`.
    // Avoid copying: move into an Arc and share.
    let rs_digits = Arc::new(rs_digits);
    *setchk_r_point_for_cm = Some(rs_digits.clone());
    let claimed0 = super::cm_math::ring_zero_digits(&mut glue.gb, ring_dim);
    let v_sc =
        super::cm_math::sumcheck_verify_degree3_ring_digits(&mut glue.gb, claimed0, &msgs_digits, rs_digits.as_ref())?;

    let k_rg = params.k as usize;
    let out_e0_len = k_rg;
    let out_b_len = 1usize;
    let nclaims_setchk = out_e0_len + out_b_len;
    let has_rc_setchk = out_e0_len > 1;

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

    // Parse absorbed out.e/out.b.
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
    // IMPORTANT: match LF+ transcript order in `setchk::absorb_evaluations`:
    // for ek in e { for ej in ek { transcript.absorb_slice(ej) } }.
    // That is: `blk` (ek) outermost, then `i` (ej), then `lane` (slice element).
    for blk in 0..out_e_blocks {
        for i in 0..out_e0_len {
            out_e_vars[blk][i] = Vec::with_capacity(lane_len);
            for _lane in 0..lane_len {
                let (st, ln) = prefix_payload[cur];
                cur += 1;
                if ln != ring_elem_bytes {
                    return Err("tiny gate: out.e ring absorb len mismatch".to_string());
                }
                let rb = parse_ring_elem_absorb_as_ringbytes(glue, pose_wiring, ring_dim, st, ln)?;
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
        let rb = parse_ring_elem_absorb_as_ringbytes(glue, pose_wiring, ring_dim, st, ln)?;
        out_b_vars.push(ring_bytes_to_digits(&mut glue.gb, &rb));
    }

    // SetChk recombination.
    let zero_bytes = alloc_const_goldilocks_u64(&mut glue.gb, 0u64);
    let zero = goldilocks_bytes_to_digits(&mut glue.gb, zero_bytes);
    let one_bytes = alloc_const_goldilocks_u64(&mut glue.gb, 1u64);
    let one = goldilocks_bytes_to_digits(&mut glue.gb, one_bytes);
    let rc_base = rc_opt.unwrap_or(one);
    let rc_pows = goldilocks_pow_table_digits(&mut glue.gb, &rc_base, nclaims_setchk.saturating_sub(1));

    let mut ver = zero;
    for i in 0..out_e0_len {
        let eq = eq_eval_goldilocks_digits(&mut glue.gb, &c_vars[i], rs_digits.as_ref())?;
        let beta = beta_vars[i];
        let alpha = alpha_vars[i];
        let beta2 = goldilocks_mul_mod_p_digits(&mut glue.gb, &beta, &beta);
        let alpha_pows = goldilocks_pow_table_digits(&mut glue.gb, &alpha, lane_len.saturating_sub(1));

        let mut e_sum = zero;
        {
            use super::cm_ir::{
                goldilocks_add_mod_p_digits_ir, goldilocks_mul_mod_p_digits_ir, goldilocks_sub_mod_p_digits_ir,
                lower_ir_into_builder, ring_eval_at_scalar_digits_d64_ir, IrBuilder, VarRef as IrVarRef,
                u64_to_bal16_digits_le_const,
            };
            let p_u64: u64 = crate::we_goldilocks_poseidon_f257::GOLDILOCKS_P;
            let p_d_const: [i8; 17] = u64_to_bal16_digits_le_const(p_u64);
            let beta_ir: [IrVarRef; 17] = core::array::from_fn(|j| IrVarRef::Base(beta[j]));
            let beta2_ir: [IrVarRef; 17] = core::array::from_fn(|j| IrVarRef::Base(beta2[j]));
            let base_asg_ir: &[F257] = &glue.gb.assignment;
            // Build IR fragments per lane-chunk in parallel, then lower fragments and accumulate.
            let lane_batch: usize = 8;
            let lane_chunks: Vec<std::ops::Range<usize>> = (0..lane_len)
                .step_by(lane_batch)
                .map(|s| s..(s + lane_batch).min(lane_len))
                .collect();

            let frags = lane_chunks
                .par_iter()
                .map(|r| -> Result<_, String> {
                    let mut ib = if glue.gb.is_count_only() {
                        IrBuilder::new_count_only(base_asg_ir)
                    } else {
                        IrBuilder::new(base_asg_ir)
                    };
                    let mut partial = {
                        let z = ib.new_var(F257::ZERO);
                        ib.ir.enforce_var_eq_const(z, F257::ZERO);
                        core::array::from_fn(|_| z)
                    };
                    for lane in r.clone() {
                        let ejv = &out_e_vars[0][i][lane];
                        if ejv.len() != 64 {
                            return Err("tiny gate: expected ring element with 64 coeffs (setchk ev IR)".to_string());
                        }
                        let coeffs_ir: [[IrVarRef; 17]; 64] =
                            core::array::from_fn(|t| core::array::from_fn(|j| IrVarRef::Base(ejv[t][j])));
                        let alpha_ir: [IrVarRef; 17] = core::array::from_fn(|j| IrVarRef::Base(alpha_pows[lane][j]));
                        let ev1 = ring_eval_at_scalar_digits_d64_ir(&mut ib, &coeffs_ir, &beta_ir, p_u64, &p_d_const);
                        let ev2 = ring_eval_at_scalar_digits_d64_ir(&mut ib, &coeffs_ir, &beta2_ir, p_u64, &p_d_const);
                        let ev1_sq = goldilocks_mul_mod_p_digits_ir(&mut ib, &ev1, &ev1, p_u64, &p_d_const);
                        let diff = goldilocks_sub_mod_p_digits_ir(&mut ib, &ev1_sq, &ev2, p_u64, &p_d_const);
                        let term = goldilocks_mul_mod_p_digits_ir(&mut ib, &diff, &alpha_ir, p_u64, &p_d_const);
                        partial = goldilocks_add_mod_p_digits_ir(&mut ib, &partial, &term, p_u64, &p_d_const);
                    }
                    Ok((ib.ir, partial))
                })
                .collect::<Result<Vec<_>, _>>()?;

            for (ir, partial_ir) in frags {
                let lowered = lower_ir_into_builder(&mut glue.gb, ir);
                let partial_d: GoldilocksScalar = core::array::from_fn(|j| lowered.map_var(partial_ir[j]));
                e_sum = goldilocks_add_mod_p_digits(&mut glue.gb, &e_sum, &partial_d);
            }
        }
        let t = goldilocks_mul_mod_p_digits(&mut glue.gb, &eq, &e_sum);
        let t = goldilocks_mul_mod_p_digits(&mut glue.gb, &t, &rc_pows[i]);
        ver = goldilocks_add_mod_p_digits(&mut glue.gb, &ver, &t);
    }
    for bi in 0..out_b_len {
        let offset = out_e0_len;
        let idx2 = bi + offset;
        let eq = eq_eval_goldilocks_digits(&mut glue.gb, &c_vars[idx2], rs_digits.as_ref())?;
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
    lf_profile_log(&format!(
        "prefix setchk_total elapsed={:?} nvars_setchk={} mlen={}",
        t_setchk.elapsed(),
        nvars_setchk,
        params.mlen
    ));

    // rgchk::Dcom::verify checks + absorb(dcom.evals).
    let t_rg = Instant::now();
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

    let p_u64 = crate::we_goldilocks_poseidon_f257::GOLDILOCKS_P;
    let mut dppow: Vec<GoldilocksScalar> = Vec::with_capacity(k_rg);
    let mut pow_acc: u64 = 1;
    for _ in 0..k_rg {
        let bs = alloc_const_goldilocks_u64(&mut glue.gb, pow_acc);
        dppow.push(goldilocks_bytes_to_digits(&mut glue.gb, bs));
        pow_acc = ((pow_acc as u128) * (params.decomp_b as u128) % (p_u64 as u128)) as u64;
    }

    // Precompute `dppow` in bal4 once (shared across all rgchk shards).
    let dppow4_base: Vec<[usize; 33]> = {
        use super::cm_ir::{lower_ir_into_builder, IrBuilder, VarRef as IrVarRef};
        // Avoid borrowing `glue.gb.assignment` across lowering: build IR under a scoped immutable borrow,
        // then lower after the borrow ends (no giant `assignment.clone()`).
        let (ir, out_ir): (super::cm_ir::CmIr, Vec<[IrVarRef; 33]>) = {
            let base_asg: &[F257] = glue.gb.assignment.as_slice();
            let mut ib = if glue.gb.is_count_only() {
                IrBuilder::new_count_only(base_asg)
            } else {
                IrBuilder::new(base_asg)
            };
            let mut out: Vec<[IrVarRef; 33]> = Vec::with_capacity(dppow.len());
            for s in &dppow {
                let s16: [IrVarRef; 17] = core::array::from_fn(|k| IrVarRef::Base(s[k]));
                out.push(ib.bal16_to_bal4_digits_cached(&s16));
            }
            (ib.ir, out)
        };
        let lowered = lower_ir_into_builder(&mut glue.gb, ir);
        out_ir
            .into_iter()
            .map(|d| core::array::from_fn(|k| lowered.map_var(d[k])))
            .collect()
    };

    // Precompute ct(psi * x) constant weights for ring_dim=64 once (host-side constants).
    let psi_ct_u64s: [u64; 64] = {
        use cyclotomic_rings::rings::GoldilocksRing64 as GR64;
        use stark_rings::{psi, unit_monomial, CoeffRing};
        let psi_r = psi::<GR64>();
        core::array::from_fn(|j| {
            let basis = unit_monomial::<GR64>(j);
            let w_br = (psi_r * basis).ct();
            w_br.into_bigint().as_ref().get(0).copied().unwrap_or(0)
        })
    };

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
    // Remaining rgchk/dcom checks are below.

    let mut dcom_evals_local: Vec<DcomEvalDigits> = Vec::with_capacity(l_evals);
    for l in 0..l_evals {
        let mut eval_a: Vec<GoldilocksScalar> = Vec::with_capacity(dcom_eval_vec_len);
        for _ in 0..dcom_eval_vec_len {
            let (st, ln) = *prefix_payload.get(cur).ok_or("tiny gate: dcom eval.a absorb oob")?;
            cur += 1;
            if ln != 8 {
                return Err("tiny gate: expected 8-byte absorb for dcom eval.a".to_string());
            }
            eval_a.push(parse_absorbed_scalar_as_goldilocks_digits(glue, pose_wiring, st));
        }
        let mut eval_c: Vec<RingDigits> = Vec::with_capacity(dcom_eval_vec_len);
        for _ in 0..dcom_eval_vec_len {
            let (st, ln) = *prefix_payload.get(cur).ok_or("tiny gate: dcom eval.c absorb oob")?;
            cur += 1;
            if ln != ring_elem_bytes {
                return Err("tiny gate: expected ring-elem absorb for dcom eval.c".to_string());
            }
            let rb = parse_ring_elem_absorb_as_ringbytes(glue, pose_wiring, ring_dim, st, ln)?;
            eval_c.push(ring_bytes_to_digits(&mut glue.gb, &rb));
        }

        // Extra witness provides `dcom.evals[*].b` and `dcom.evals[*].v`, which are not fully transcript-derived
        // but are required for the full faithful verifier arithmetic checks.
        let b_l = extra_witness
            .dcom_eval_b
            .get(l)
            .ok_or("tiny gate: missing dcom_eval_b for instance")?;
        let v_l = extra_witness
            .dcom_eval_v
            .get(l)
            .ok_or("tiny gate: missing dcom_eval_v for instance")?;
        if b_l.len() != eval_a.len() || v_l.len() != ring_dim {
            return Err("tiny gate: dcom extra witness length mismatch".to_string());
        }
        let mut eval_b: Vec<RingDigits> = Vec::with_capacity(eval_a.len());
        for b_i in b_l {
            if b_i.len() != ring_dim {
                return Err("tiny gate: dcom_eval_b row length mismatch".to_string());
            }
            let mut r: RingDigits = Vec::with_capacity(ring_dim);
            for k in 0..ring_dim {
                r.push(alloc_witness_goldilocks_u64_digits(&mut glue.gb, b_i[k]));
            }
            eval_b.push(r);
        }
        let mut eval_v: Vec<GoldilocksScalar> = Vec::with_capacity(ring_dim);
        for k in 0..ring_dim {
            eval_v.push(alloc_witness_goldilocks_u64_digits(&mut glue.gb, v_l[k]));
        }

        for i in 0..eval_a.len() {
            let ct = ct_psi_mul_ring_digits_d64(&mut glue.gb, &eval_b[i])?;
            for di in 0..17 {
                glue.gb.enforce_lc_times_one_eq_const(vec![
                    (F257::ONE, ct[di]),
                    (-F257::ONE, eval_a[i][di]),
                ]);
            }
        }

        let base = l
            .checked_mul(k_rg)
            .ok_or_else(|| "tiny gate: rgchk base overflow".to_string())?;
        for ni in 0..out_e_blocks {
            // Big optimization: compute the entire rgchk `ct(psi * Σ_i ui_col * dp^i)` per (ni,col)
            // as an IR shard in the bal4 domain, converting back to bal16 only once at the end.
            //
            // This removes the repeated bal16<->bal4 conversions inside `ring_scale_digits` and
            // inside `ct_psi_mul_ring_digits_d64` for each intermediate ring value.
            use super::cm_ir::{
                goldilocks_add_mod_p_digits_bal4_ir, goldilocks_mul_const_mod_p_digits_bal4_ir,
                goldilocks_mul_mod_p_digits_bal4_ir, goldilocks_sub_mod_p_digits_bal4_ir, lower_ir_into_builder,
                IrBuilder, VarRef as IrVarRef,
            };

            let cols = ring_dim;
            let col_batch: usize = (8 * rayon::current_num_threads().max(1)).max(1);
            for c0 in (0..cols).step_by(col_batch) {
                let c1 = (c0 + col_batch).min(cols);
                let batch_len = c1 - c0;
                let base_asg: &[F257] = &glue.gb.assignment;
                let count_only = glue.gb.is_count_only();

                // Key perf fix: build one IR per *group of columns* (still in parallel), then lower per group.
                // This reduces the number of `lower_ir_into_builder` calls from ~64 -> ~O(16).
                let n_threads = rayon::current_num_threads().max(1);
                let target_groups = n_threads.min(batch_len.max(1)).min(16).max(1);
                let col_group = (batch_len + target_groups - 1) / target_groups;
                let mut groups: Vec<(usize, usize)> = Vec::new();
                let mut g0 = 0usize;
                while g0 < batch_len {
                    let g1 = (g0 + col_group).min(batch_len);
                    groups.push((g0, g1));
                    g0 = g1;
                }

                let frags: Vec<(super::cm_ir::CmIr, Vec<([IrVarRef; 17], usize)>)> = groups
                    .into_par_iter()
                    .map(|(gs, ge)| -> Result<_, String> {
                        let mut ib = if count_only {
                            IrBuilder::new_count_only(base_asg)
                        } else {
                            IrBuilder::new(base_asg)
                        };
                        let z = ib.new_var(F257::ZERO);
                        ib.ir.enforce_var_eq_const(z, F257::ZERO);
                        let z4: [IrVarRef; 33] = [z; 33];

                        let mut outs: Vec<([IrVarRef; 17], usize)> = Vec::with_capacity(ge - gs);
                        for c_local in gs..ge {
                            let col = c0 + c_local;

                            // acc_ring in bal4 per coefficient.
                            let mut acc4: [[IrVarRef; 33]; 64] = [z4; 64];
                            for i in 0..k_rg {
                                let idx = base + i;
                                if idx >= out_e_vars[ni].len() {
                                    return Err("tiny gate: out.e length too short for rgchk".to_string());
                                }
                                let ui_col = &out_e_vars[ni][idx][col];
                                if ui_col.len() != 64 {
                                    return Err("tiny gate: expected ring element with 64 coeffs (rgchk)".to_string());
                                }
                                let s4: [IrVarRef; 33] =
                                    core::array::from_fn(|k| IrVarRef::Base(dppow4_base[i][k]));
                                for coeff in 0..64 {
                                    let ui16: [IrVarRef; 17] =
                                        core::array::from_fn(|k| IrVarRef::Base(ui_col[coeff][k]));
                                    let ui4 = ib.bal16_to_bal4_digits_cached(&ui16);
                                    let prod4 = goldilocks_mul_mod_p_digits_bal4_ir(&mut ib, &ui4, &s4, p_u64);
                                    acc4[coeff] =
                                        goldilocks_add_mod_p_digits_bal4_ir(&mut ib, &acc4[coeff], &prod4, p_u64);
                                }
                            }

                            // ct(psi * x) in bal4: Σ_j acc4[j] * psi_ct_u64s[j].
                            let mut ct4: [IrVarRef; 33] = z4;
                            for j in 0..64 {
                                let w_u64 = psi_ct_u64s[j];
                                if w_u64 == 0 {
                                    continue;
                                }
                                let t = if w_u64 == 1 {
                                    acc4[j]
                                } else if w_u64 == p_u64 - 1 {
                                    goldilocks_sub_mod_p_digits_bal4_ir(&mut ib, &z4, &acc4[j], p_u64)
                                } else {
                                    goldilocks_mul_const_mod_p_digits_bal4_ir(&mut ib, &acc4[j], w_u64, p_u64)
                                };
                                ct4 = goldilocks_add_mod_p_digits_bal4_ir(&mut ib, &ct4, &t, p_u64);
                            }

                            let ct16 = ib.bal4_to_bal16_digits_cached(&ct4);
                            outs.push((ct16, col));
                        }
                        Ok((ib.ir, outs))
                    })
                    .collect::<Result<Vec<_>, _>>()?;

                for (ir, outs) in frags {
                    let lowered = lower_ir_into_builder(&mut glue.gb, ir);
                    for (ct16_ir, col) in outs {
                        let ct: GoldilocksScalar = core::array::from_fn(|k| lowered.map_var(ct16_ir[k]));
                        let expected = if ni == 0 {
                            eval_v[col]
                        } else {
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
            }
        }

        dcom_evals_local.push(DcomEvalDigits {
            a: eval_a,
            b: eval_b,
            c: eval_c,
        });
    }

    lf_profile_log(&format!(
        "prefix rgchk+dcom_evals elapsed={:?} L_evals={}",
        t_rg.elapsed(),
        l_evals
    ));
    lf_profile_log(&format!("prefix total elapsed={:?}", t_all.elapsed()));

    *setchk_out_e_vars_for_cm = Some(out_e_vars);
    *dcom_evals_for_cm = Some(dcom_evals_local);
    Ok(())
}

fn compute_cm_shared_precomp_base(
    glue: &mut GlueCtx,
    ring_dim: usize,
    params: &WeParams,
    wiring: &TinyCoinOpWiring,
    l_instances_expected: usize,
    short_locals: &[ShortChallengeWiring],
    u32_locals: &[BoundedU32ChallengeWiring],
    setchk_out_e_vars_for_cm: &Option<Arc<Vec<Vec<Vec<RingDigits>>>>>,
    setchk_r_point_for_cm: &Option<Arc<Vec<GoldilocksScalar>>>,
) -> Result<Arc<CmSharedPrecompBase>, String> {
    // Exact logic previously inlined in `build()`. Keep it in one place so both in-memory and
    // file-backed builds share the *same* precompute logic and op-mix behavior.
    assert_eq!(ring_dim, 64);
    assert_eq!(l_instances_expected, 1, "unexpected number of CM instances");
    let can_precompute = l_instances_expected > 0
        && params.kappa > 0
        && (params.kappa as usize).is_power_of_two()
        && ((params.k as usize) * ring_dim).is_power_of_two()
        && (params.l as usize).is_power_of_two()
        && setchk_out_e_vars_for_cm.is_some()
        && setchk_r_point_for_cm.is_some();
    if !can_precompute {
        return Err("tiny gate: cm_shared_base required (expected pow2 regime + setchk outputs)".to_string());
    }
    {
        let kappa = params.kappa as usize;
        let log_kappa = ark_std::log2(kappa.next_power_of_two()) as usize;
        let k_decomp = params.k as usize;
        let ell = params.l as usize;
        let rows_per_l = 1usize + (params.mlen as usize);

        // Shared CM challenge seed blocks: c0/c1.
        let cm_u32_start = cm_u32_start_idx(wiring);
        if u32_locals.len() < cm_u32_start + 2 * log_kappa {
            return Err(format!(
                "tiny gate: cannot compute cm_shared_base (missing u32 challenges for c0/c1): have_u32={} need_u32={} (cm_u32_start={} log_kappa={})",
                u32_locals.len(),
                cm_u32_start + 2 * log_kappa,
                cm_u32_start,
                log_kappa
            ));
        }
        if short_locals.len() < 3 + k_decomp * ring_dim {
            return Err(format!(
                "tiny gate: cannot compute cm_shared_base (missing short challenges for s_prime_flat): have_short={} need_short={} (k={} ring_dim={})",
                short_locals.len(),
                3 + k_decomp * ring_dim,
                k_decomp,
                ring_dim
            ));
        }
        {
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

            // s_prime_flat: k*d short challenges; each entry is a ring element whose coefficients are
            // derived via the short-challenge rule (byte % u) - (u/2), NOT a centered byte.
            let need_sprime = k_decomp * ring_dim;
            // short_challenge(128): u = 2^(128/d), u <= 256 since d >= 1 and we only use this path for d=64.
            let exp = (128usize / ring_dim) as usize;
            let u = 1u64 << (exp as u32);
            debug_assert!(u.is_power_of_two() && u <= 256);
            let half = u / 2;
            let half_bytes = alloc_const_goldilocks_u64(&mut glue.gb, half);
            let half_digits = goldilocks_bytes_to_digits(&mut glue.gb, half_bytes);
            let zero_byte = glue.gb.new_var(F257::ZERO);
            glue.gb.enforce_var_eq_const(zero_byte, F257::ZERO);
            // build `s_prime_flat` in **batched IR** and lower in coarse chunks.
            //
            // Count-only friendliness: in Pass0 we build the IR in count-only mode and the lowerer
            // bumps counts via `stats` (no term streaming).
            let timing_u = tiny_opmix_on();
            let t_sflat = Instant::now();
            let p_u64 = crate::we_goldilocks_poseidon_f257::GOLDILOCKS_P;
            let p_d_const = super::goldilocks::goldilocks_p_bal16_digits_le_const();
            let zb = decompose_existing_byte_var_to_bits::<F257>(&mut glue.gb, zero_byte);
            if zb.len() != 8 {
                return Err("tiny gate: internal error: zero_byte bits len mismatch".to_string());
            }
            let zero_bits: [usize; 8] = core::array::from_fn(|i| zb[i]);
            let half_ir: [super::cm_ir::VarRef; 17] =
                core::array::from_fn(|j| super::cm_ir::VarRef::Base(half_digits[j]));

            // Also build s[0..3] once (only 3 rings) using a single batched IR + one lowering.
            //
            // This replaces the old per-coefficient `short_challenge_coeff_digits_from_byte_var_128` path.
            let s_rings: [RingDigits; 3] = {
                for i in 0..3 {
                    if short_locals[i].byte_vars.len() != ring_dim {
                        return Err("tiny gate: short byte_vars len mismatch (base s[0..3])".to_string());
                    }
                }
                let blocks_bits = collect_short_coeff_bits(
                    &mut glue.gb,
                    ring_dim,
                    exp,
                    zero_bits,
                    3,
                    |si, col| short_locals[si].byte_vars[col],
                )?;
                let (ir, outs) = build_short_coeff_digits_ir(&mut glue.gb, &blocks_bits, &half_ir, p_u64, &p_d_const);
                let lowered = super::cm_ir::lower_ir_into_builder(&mut glue.gb, ir);
                let mut it = map_short_coeff_digits_blocks_to_rings(&lowered, outs, ring_dim).into_iter();
                let r0 = it.next().expect("s_rings: missing ring 0");
                let r1 = it.next().expect("s_rings: missing ring 1");
                let r2 = it.next().expect("s_rings: missing ring 2");
                [r0, r1, r2]
            };

            const SFLAT_CHUNK_BLOCKS: usize = 16;
            let mut sflat: Vec<RingDigits> = Vec::with_capacity(need_sprime);
            let mut blk0 = 0usize;
            while blk0 < need_sprime {
                let blk1 = (blk0 + SFLAT_CHUNK_BLOCKS).min(need_sprime);

                // Collect bit references first (needs `&mut glue.gb`), then build IR without holding
                // an immutable borrow of `glue.gb.assignment`.
                for blk in blk0..blk1 {
                    let sb = &short_locals[3 + blk];
                    if sb.byte_vars.len() != ring_dim {
                        return Err("tiny gate: short byte_vars len mismatch (base s_prime_flat)".to_string());
                    }
                }
                let blocks_bits = collect_short_coeff_bits(
                    &mut glue.gb,
                    ring_dim,
                    exp,
                    zero_bits,
                    blk1 - blk0,
                    |i, col| short_locals[3 + blk0 + i].byte_vars[col],
                )?;
                let (ir, outs) = build_short_coeff_digits_ir(&mut glue.gb, &blocks_bits, &half_ir, p_u64, &p_d_const);
                let lowered = super::cm_ir::lower_ir_into_builder(&mut glue.gb, ir);
                sflat.extend(map_short_coeff_digits_blocks_to_rings(&lowered, outs, ring_dim));
                blk0 = blk1;
            }
            if timing_u {
                eprintln!("tiny_gate: CM s_prime_flat timing: elapsed={:?} blocks={} coeffs={}", t_sflat.elapsed(), need_sprime, need_sprime.saturating_mul(ring_dim));
            }

            // SetChk verifier point `r` used in eq(r, ro).
            //
            // This is computed (and transcript-bound) during the SetChk prefix arithmetization.
            // We still recompute the cursor/offset-derived `r` here and enforce equality, so we
            // never silently diverge from the transcript schedule.
            let rp = setchk_r_point_for_cm
                .as_ref()
                .expect("setchk_r_point_for_cm must be Some when setchk_out_e_vars_for_cm is Some");
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
            let mut rdig_cursor: Vec<GoldilocksScalar> = Vec::with_capacity(nvars_lin);
            for u in &u32_locals[r_start..r_end] {
                let bytes = goldilocks_bytes_from_u32_le_bytes(
                    &mut glue.gb,
                    &[u.byte_vars[0], u.byte_vars[1], u.byte_vars[2], u.byte_vars[3]],
                );
                rdig_cursor.push(goldilocks_bytes_to_digits(&mut glue.gb, bytes));
            }
            if rp.len() != rdig_cursor.len() {
                return Err("tiny gate: setchk r-point length mismatch".to_string());
            }
            for (a, b) in rp.iter().zip(rdig_cursor.iter()) {
                for i in 0..17 {
                    glue.gb.enforce_lc_times_one_eq_const(vec![(F257::ONE, a[i]), (-F257::ONE, b[i])]);
                }
            }
            let rdig: Arc<Vec<GoldilocksScalar>> = rp.clone();

            // Compute u[l][ni] once (heavy): Σ out.e * s_prime_flat.
            let out_e_base = setchk_out_e_vars_for_cm.as_ref().unwrap().as_ref();
            if out_e_base.len() != rows_per_l {
                return Err("tiny gate: setchk out.e rows_per_l mismatch".to_string());
            } else {
                    use super::cm_ir::{
                        bal4_to_bal16_digits_ir, goldilocks_add_mod_p_digits_bal4_ir, lower_ir_into_builder,
                        ring_mul_negacyclic_ntt_goldilocks_d64_bal4_ir, IrBuilder, VarRef as IrVarRef,
                    };

                    #[inline]
                    fn ringdigits64_to_ir(a: &RingDigits) -> Result<[[IrVarRef; 17]; 64], String> {
                        if a.len() != 64 {
                                    return Err("tiny gate: expected ring element with 64 coeffs (IR ring-mul)".to_string());
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
                    type Scalar4 = [usize; 33];
                    type Ring4 = [Scalar4; 64];
                    #[inline]
                    fn map_ring_out4(out_ir: &[[IrVarRef; 33]; 64], lowered: &super::cm_ir::LoweredIr) -> Ring4 {
                        core::array::from_fn(|i| core::array::from_fn(|j| lowered.map_var(out_ir[i][j])))
                    }

                    let mut u_ir_build_time = Duration::ZERO;
                    let mut u_lower_time = Duration::ZERO;
                    let mut u_frag_count: usize = 0;
                    let mut u_batch_used: usize = 0;
                    // Shard granularity for u_shared.
                    //
                    // If this is too large, small instances produce very few shards (e.g. frags=4),
                    // leaving most cores idle. If it's too small, we create too many fragments and
                    // pay overhead in IR merge + lowering.
                    //
                    // Default heuristic: target ~O(min(2*threads, 64)) shards per (l,ni),
                    // but keep shards coarse for large rows to avoid exploding “convert back”
                    // overhead (bal4->bal16 is done once per coefficient per shard).
                    let env_batch: Option<usize> = std::env::var("LFP_CM_U_SHARED_BATCH")
                        .ok()
                        .and_then(|s| s.parse::<usize>().ok())
                        .filter(|&v| v > 0);

                    let mut u_all: Vec<Vec<RingDigits>> = Vec::with_capacity(l_instances_expected);
                    for l in 0..l_instances_expected {
                        let mut u_l: Vec<RingDigits> = Vec::with_capacity(rows_per_l);
                        for ni in 0..rows_per_l {
                            // Number of terms in the u_shared sum for one (l, ni):
                            // Σ_{blk,col} out.e[ni][l*k + blk][col] * s_prime_flat[blk*ring_dim + col].
                            let terms_len: usize = k_decomp
                                .checked_mul(ring_dim)
                                .ok_or_else(|| "tiny gate: u term count overflow".to_string())?;

                            // Compute in shards: each shard sums `batch_size` products internally in **bal4**,
                            // converting back to bal16 only once per coefficient at the end.
                            //
                            // This avoids the very expensive builder-level `ring_add_digits` chain.
                            // IMPORTANT: use a normal shared slice borrow so Rust prevents concurrent
                            // mutation of `glue.gb.assignment` while Rayon threads are reading witnesses.
                            let base_asg: &[F257] = &glue.gb.assignment;

                            // Determine shard size for this row.
                            let batch_size: usize = if let Some(v) = env_batch {
                                v
                            } else {
                                let threads = rayon::current_num_threads().max(1);
                                // Now that we do the shard reduction in **bal4** (and convert to bal16 only once at the end),
                                // we can safely target more shards to saturate wide machines without multiplying conversion work.
                                let target_frags = (threads.saturating_mul(4)).min(256).max(1);
                                let raw = (terms_len + target_frags - 1) / target_frags;
                                // Keep shards reasonably coarse to avoid overwhelming IR/lowering overhead,
                                // but do not artificially cap parallelism on large instances.
                                let min_batch = if terms_len >= 512 { 16 } else { 8 };
                                raw.clamp(min_batch, 256)
                            };
                            u_batch_used = batch_size;

                            // Build shard IRs in parallel.
                            let t_build = Instant::now();
                            let shard_ranges: Vec<(usize, usize)> = (0..terms_len)
                                .step_by(batch_size)
                                .map(|s| (s, (s + batch_size).min(terms_len)))
                                .collect();
                            let frags: Vec<(_, [[IrVarRef; 33]; 64])> = shard_ranges
                                .into_par_iter()
                                .map(|(t0, t1)| -> Result<_, String> {
                                    let mut ib = if glue.gb.is_count_only() {
                                        IrBuilder::new_count_only(base_asg)
                                    } else {
                                        IrBuilder::new(base_asg)
                                    };

                                    // Local helper: allocate a reusable 0 constant inside this IR.
                                    // (We can't call cm_ir's `alloc_zero_const_ir` since it's private.)
                                    let zero_digit: IrVarRef = {
                                        let z = ib.new_var(F257::ZERO);
                                        ib.ir.enforce_var_eq_const(z, F257::ZERO);
                                        z
                                    };
                                    let zero4: [IrVarRef; 33] = [zero_digit; 33];
                                    let mut acc4: [[IrVarRef; 33]; 64] = [zero4; 64];

                                    // IMPORTANT: avoid building a per-(l,ni) `terms: Vec<(&RingDigits, usize)>`.
                                    // The pointer-chasing + allocation shows up as a big serial "setup" slice on
                                    // large instances, before the shard build/lower timings start.
                                    //
                                    // Map flat term index t -> (blk, col) -> (out.e cell, s_prime_flat idx).
                                    for t in t0..t1 {
                                        let blk = t / ring_dim;
                                        let col = t - blk * ring_dim;
                                        let idx = l
                                            .checked_mul(k_decomp)
                                            .and_then(|x| x.checked_add(blk))
                                            .ok_or_else(|| "tiny gate: u index overflow (base cm)".to_string())?;
                                        if idx >= out_e_base[ni].len() {
                                            return Err("tiny gate: out.e too short for base CM u computation".to_string());
                                        }
                                        let uij: &RingDigits = &out_e_base[ni][idx][col];
                                        let sp_idx: usize = t; // blk*ring_dim + col
                                        let u16 = ringdigits64_to_ir(uij)?;
                                        let s16 = ringdigits64_to_ir(&sflat[sp_idx])?;
                                        // Convert to bal4 once, then do NTT ringmul in bal4 domain.
                                        let mut u4: [[IrVarRef; 33]; 64] = [zero4; 64];
                                        let mut s4: [[IrVarRef; 33]; 64] = [zero4; 64];
                                        for i in 0..64 {
                                            u4[i] = ib.bal16_to_bal4_digits_cached(&u16[i]);
                                            s4[i] = ib.bal16_to_bal4_digits_cached(&s16[i]);
                                        }
                                        let prod4 = ring_mul_negacyclic_ntt_goldilocks_d64_bal4_ir(&mut ib, &u4, &s4);
                                        for i in 0..64 {
                                            acc4[i] = goldilocks_add_mod_p_digits_bal4_ir(
                                                &mut ib,
                                                &acc4[i],
                                                &prod4[i],
                                                crate::we_goldilocks_poseidon_f257::GOLDILOCKS_P,
                                            );
                                        }
                                        // Keep op-mix accounting consistent: one ring-mul per term.
                                        super::op_counts::tiny_cm_bump(|c| c.ring_mul_negacyclic += 1);
                                    }

                                    // Return the shard accumulator in **bal4**.
                                    Ok((ib.ir, acc4))
                                })
                                .collect::<Result<Vec<_>, _>>()?;

                            u_ir_build_time = u_ir_build_time.saturating_add(t_build.elapsed());
                            u_frag_count = u_frag_count.saturating_add(frags.len());

                            // Lower shards and reduce their outputs in **bal4**, then convert once to bal16.
                            let t_lower = Instant::now();
                            let mut cur4: Vec<Ring4> = Vec::with_capacity(frags.len());
                            for (ir, out4_ir) in frags {
                                let lowered = lower_ir_into_builder(&mut glue.gb, ir);
                                cur4.push(map_ring_out4(&out4_ir, &lowered));
                            }

                            // Reduce in bal4 using a tree of bal4 ring additions (pairwise).
                            // Build IRs in parallel, lower sequentially (single builder).
                            while cur4.len() > 1 {
                                let pairs_len: usize = cur4.len() / 2;
                                let carry: Option<Ring4> =
                                    if (cur4.len() & 1) == 1 { Some(*cur4.last().unwrap()) } else { None };

                                // Snapshot assignment for this reduction layer.
                                let base_asg2: &[F257] = &glue.gb.assignment;
                                let reduce_frags: Vec<(_, [[IrVarRef; 33]; 64])> = (0..pairs_len)
                                    .into_par_iter()
                                    .map(|pi| -> Result<_, String> {
                                        let a: &Ring4 = &cur4[2 * pi];
                                        let b: &Ring4 = &cur4[2 * pi + 1];
                                        let mut ib = if glue.gb.is_count_only() {
                                            IrBuilder::new_count_only(base_asg2)
                                        } else {
                                            IrBuilder::new(base_asg2)
                                        };
                                        let mut out: [[IrVarRef; 33]; 64] = [[IrVarRef::Base(0); 33]; 64];
                                        for i in 0..64 {
                                            let a4: [IrVarRef; 33] = core::array::from_fn(|k| IrVarRef::Base(a[i][k]));
                                            let b4: [IrVarRef; 33] = core::array::from_fn(|k| IrVarRef::Base(b[i][k]));
                                            out[i] = goldilocks_add_mod_p_digits_bal4_ir(
                                                &mut ib,
                                                &a4,
                                                &b4,
                                                crate::we_goldilocks_poseidon_f257::GOLDILOCKS_P,
                                            );
                                        }
                                        Ok((ib.ir, out))
                                    })
                                    .collect::<Result<Vec<_>, _>>()?;

                                let mut next4: Vec<Ring4> = Vec::with_capacity((cur4.len() + 1) / 2);
                                for (ir, out4_ir) in reduce_frags {
                                    let lowered = lower_ir_into_builder(&mut glue.gb, ir);
                                    next4.push(map_ring_out4(&out4_ir, &lowered));
                                }
                                if let Some(last) = carry {
                                    next4.push(last);
                                }
                                cur4 = next4;
                            }

                            let acc4: Ring4 = cur4
                                .pop()
                                .unwrap_or_else(|| core::array::from_fn(|_| core::array::from_fn(|_| glue.gb.one())));

                            // Convert the final accumulator to bal16 once per coefficient.
                            let base_asg_ir: &[F257] = &glue.gb.assignment;
                            let (ir_conv, out16_ir): (_, [[IrVarRef; 17]; 64]) = {
                                let mut ib = if glue.gb.is_count_only() {
                                    IrBuilder::new_count_only(base_asg_ir)
                                } else {
                                    IrBuilder::new(base_asg_ir)
                                };
                                let mut out: [[IrVarRef; 17]; 64] = [[IrVarRef::Base(0); 17]; 64];
                                for i in 0..64 {
                                    let a4: [IrVarRef; 33] = core::array::from_fn(|k| IrVarRef::Base(acc4[i][k]));
                                    out[i] = bal4_to_bal16_digits_ir(&mut ib, &a4);
                                }
                                (ib.ir, out)
                            };
                            let lowered = lower_ir_into_builder(&mut glue.gb, ir_conv);
                            let acc_ring: RingDigits = map_ring_out(&out16_ir, &lowered);
                            u_lower_time = u_lower_time.saturating_add(t_lower.elapsed());

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
                            if env_batch.is_some() { env_batch.unwrap_or(0) } else { u_batch_used },
                            rayon::current_num_threads().max(1),
                        );
                    }

                return Ok(Arc::new(CmSharedPrecompBase {
                    tensor_c0_ring,
                    tensor_c1_ring,
                    s_prime_flat_ring: sflat,
                    s_rings,
                    dpp_ring: dpp,
                    r_point_digits: rdig,
                    u: u_all,
                }));
            }
        }
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
    if ring_dim != 64 {
        return Err(format!("tiny gate: only supports ring_dim=64 (got {ring_dim})"));
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

    // `TracePoseidonTranscript::get_challenge()` returns a u32 embedded in the base ring.
    // Therefore, its canonical 8-byte little-endian encoding has the high 4 bytes equal to zero.
    let zero_byte = glue.gb.new_var(F257::ZERO);
    glue.gb.enforce_var_eq_const(zero_byte, F257::ZERO);

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
            u64_bytes[i] = zero_byte;
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
/// and that the per-round "marker" scalar absorbs match the sampled `r_i` bytes (as in the
/// transcript schedule: sample `r_i` via `get_challenge()`, then explicitly absorb `r_i`).
fn parse_and_enforce_cm_after_short(
    glue: &mut GlueCtx,
    ops: &[PoseidonTraceOp<F257>],
    pose_wiring: &PoseidonDr1csWiring,
    ring_dim: usize,
    params: &WeParams,
    wiring: &TinyCoinOpWiring,
    l_instances: usize,
    goldilocks_locals: &[GoldilocksChallengeWiring],
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

    // We also bind the per-round “explicit absorb of r_i” to the same coins that were sampled by
    // `get_challenge()`. This mirrors the real verifier transcript schedule. These are full
    // Goldilocks challenges (8 bytes), matching LF+.
    let cm_coin_start = cm_u32_start_idx(wiring);
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
            // Bind absorbed bytes = sampled challenge bytes (all 8).
            let chal_idx = cm_coin_start + 2 * log_kappa + which * (1 + nvars_cm) + 1 + _round;
            if chal_idx >= goldilocks_locals.len() {
                return Err("tiny gate: goldilocks_locals too short for CM sumcheck r_i binding".to_string());
            }
            let ch = &goldilocks_locals[chal_idx];
            for i in 0..8 {
                let gv = pose_wiring.absorb_vars[st + i];
                let lv = glue.copy_digit(gv);
                glue.gb.enforce_lc_times_one_eq_const(vec![
                    (F257::ONE, lv),
                    (-F257::ONE, ch.byte_vars[i]),
                ]);
                if glue.gb.assignment[lv] != glue.gb.assignment[ch.byte_vars[i]] {
                    return Err(format!(
                        "tiny gate: CM sumcheck r_i byte mismatch (which={which} round={_round} byte={i}): absorb={:?} chal_byte={:?}",
                        glue.gb.assignment[lv],
                        glue.gb.assignment[ch.byte_vars[i]]
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

fn collect_fiat_shamir_reabsorb_eqs(
    ops: &[PoseidonTraceOp<F257>],
    pose_wiring: &PoseidonDr1csWiring,
) -> Result<Vec<(usize, usize)>, String> {
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
    Ok(eqs)
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

fn enforce_canonical_goldilocks_for_ranges(
    glue: &mut GlueCtx,
    pose_wiring: &PoseidonDr1csWiring,
    ranges: &[(usize, usize)],
) -> Result<(), String> {
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
    Ok(())
}

fn build_cm_glue_for_which<'p>(
    mode: &TinyGateBuildMode,
    part_idx: usize,
    range: Option<&TinyGateRangeBase>,
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
    cm_shared_base: Arc<CmSharedPrecompBase>,
    which: usize,
    pose_asg: &'p [F257],
    base_asg: &[F257],
    goldilocks_locals: &[GoldilocksChallengeWiring],
    out_dir: &Path,
) -> Result<GlueCtx<'p>, String> {
    let mut glue = GlueCtx::new(mode, pose_asg, out_dir, part_idx, range)?;
    if ring_dim == 0 || l_instances_expected == 0 {
        return Ok(glue);
    }

    // Attach a top-level profiling scope for any constraints not already labeled by inner gadgets.
    // This helps turn the big "unlabeled" bucket into something actionable, per shard.
    //
    // IMPORTANT: this must *not* hold `&mut glue.gb` for the whole function, since we also need
    // `&mut glue` later (e.g. to parse transcript absorbs). Use a raw pointer for Drop-time exit.
    struct ProfileGuard<'a, F: ark_ff::PrimeField> {
        gb: *mut symphony::dpp_sumcheck::Dr1csBuilder<F>,
        prev: Option<&'static str>,
        _pd: core::marker::PhantomData<&'a mut symphony::dpp_sumcheck::Dr1csBuilder<F>>,
    }
    impl<'a, F: ark_ff::PrimeField> Drop for ProfileGuard<'a, F> {
        fn drop(&mut self) {
            // Safety: `gb` points to `glue.gb` which outlives this guard (same stack frame),
            // and we only use it to call `profile_exit` when the guard is being dropped.
            unsafe {
                (*self.gb).profile_exit(self.prev);
            }
        }
    }
    let _prof = if glue.gb.profile_enabled {
        let label: &'static str = if which == 0 { "cm0_total" } else { "cm1_total" };
        let prev = glue.gb.profile_enter(label);
        Some(ProfileGuard {
            gb: &mut glue.gb as *mut _,
            prev,
            _pd: core::marker::PhantomData,
        })
    } else {
        None
    };

    // The CM segment begins at the last short squeeze, but the absorb ranges were already parsed
    // by the base glue module (and statement-bound there). Here we just consume the ranges.
    // Index into the post-`absorb_comh` `get_challenge()` stream.
    // We interpret these as full Goldilocks challenges (8 bytes), matching LF+.
    let cm_coin_start = cm_u32_start_idx(wiring);
    let kappa = params.kappa as usize;
    let log_kappa = ark_std::log2(kappa.next_power_of_two()) as usize;
    let nvars_cm = params.nvars_cm as usize;

    // CM challenges after absorb_comh: c0/c1, then for each sumcheck: rc, r_sc[0..nvars_cm]
    // These are transcript `get_challenge()` values in the Goldilocks base field (8 bytes).
    let tensor_c0_ring: Vec<RingDigits> = cm_shared_base
        .tensor_c0_ring
        .iter()
        .map(|r| import_ring_from_base(&mut glue, base_asg, r))
        .collect();
    let tensor_c1_ring: Vec<RingDigits> = cm_shared_base
        .tensor_c1_ring
        .iter()
        .map(|r| import_ring_from_base(&mut glue, base_asg, r))
        .collect();
    let s_prime_flat_ring: Vec<RingDigits> = cm_shared_base
        .s_prime_flat_ring
        .iter()
        .map(|r| import_ring_from_base(&mut glue, base_asg, r))
        .collect();
    let dpp_ring: Vec<RingDigits> = cm_shared_base
        .dpp_ring
        .iter()
        .map(|r| import_ring_from_base(&mut glue, base_asg, r))
        .collect();
    let r_point_digits: Vec<GoldilocksScalar> = cm_shared_base
        .r_point_digits
        .iter()
        .map(|s| import_scalar_from_base(&mut glue, base_asg, s))
        .collect();

    // Select the Goldilocks `get_challenge()` window for this sumcheck:
    // rc, then nvars_cm per-round r's.
    let mut coin_idx = cm_coin_start + 2 * log_kappa + which * (1 + nvars_cm);
    let rc_ch = &goldilocks_locals[coin_idx];
    let rc_bytes: [usize; 8] = core::array::from_fn(|i| glue.import_base_var(base_asg, rc_ch.byte_vars[i]));
    coin_idx += 1;
    let mut rs: Vec<[usize; 8]> = Vec::with_capacity(nvars_cm);
    for _ in 0..nvars_cm {
        let ch = &goldilocks_locals[coin_idx];
        let bytes: [usize; 8] = core::array::from_fn(|i| glue.import_base_var(base_asg, ch.byte_vars[i]));
        rs.push(bytes);
        coin_idx += 1;
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
        // Only implement the pow2 fast path (same assumptions as `eval_t_z_optimized_ring_digits_pair`).
        let out_e_blocks = 1usize + (params.mlen as usize);
        let rows_per_l = 1usize + (params.mlen as usize);
        if dcom_evals_base.len() == l_instances_expected && out_e_base.len() == out_e_blocks {
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
                //
                // Big optimization: build one IR shard per `l` that does all scaling + accumulation in bal4,
                // then converts to bal16 once. This removes the repeated bal16<->bal4 conversions inside
                // `ring_scale_digits` and is embarrassingly parallel across `l`.
                let mut tcch0: Vec<RingDigits> = Vec::with_capacity(l_instances_expected);
                let mut tcch1: Vec<RingDigits> = Vec::with_capacity(l_instances_expected);
                if comh_absorbs.len() == l_instances_expected * kappa {
                    // tensor_c{0,1} are constant-coeff rings; coefficient-0 holds the scalar digits.
                    let tensor_c0_scalars: Vec<GoldilocksScalar> = tensor_c0_ring.iter().map(|r| r[0]).collect();
                    let tensor_c1_scalars: Vec<GoldilocksScalar> = tensor_c1_ring.iter().map(|r| r[0]).collect();

                    // Optional debug sanity: ensure tensor rings are truly constant-coeff (coeff>0 == 0),
                    // so extracting `r[0]` is semantically valid.
                    //
                    // Enable with `LFP_WE_GATE_OPMIX=1`.
                    if std::env::var("LFP_WE_GATE_OPMIX").ok().as_deref() == Some("1") {
                        for (ti, (r0, r1)) in tensor_c0_ring.iter().zip(tensor_c1_ring.iter()).enumerate() {
                            for coeff in 1..ring_dim.min(64) {
                                for d in 0..17 {
                                    let v0 = glue.gb.assignment[r0[coeff][d]];
                                    let v1 = glue.gb.assignment[r1[coeff][d]];
                                    assert!(
                                        v0 == F257::ZERO,
                                        "tensor_c0_ring[{ti}][{coeff}][{d}] nonzero: {:?}",
                                        v0
                                    );
                                    assert!(
                                        v1 == F257::ZERO,
                                        "tensor_c1_ring[{ti}][{coeff}][{d}] nonzero: {:?}",
                                        v1
                                    );
                                }
                            }
                        }
                    }

                    // Keep op-mix accounting consistent (replaces 2 * l_instances_expected * kappa calls to ring_scale_digits).
                    super::op_counts::tiny_cm_bump(|c| {
                        c.ring_scale += (2 * l_instances_expected * kappa) as u64;
                        c.scalar_mul += (2 * l_instances_expected * kappa * ring_dim) as u64;
                    });

                    // Precompute tensor scalars in bal4 once (shared across all shards).
                    let (s0_4_base, s1_4_base): (Vec<[usize; 33]>, Vec<[usize; 33]>) = {
                        use super::cm_ir::{lower_ir_into_builder, IrBuilder, VarRef as IrVarRef};
                        // Avoid cloning the full assignment: build IR under scoped immutable borrow,
                        // then lower after the borrow ends.
                        let (ir, s0_4_ir, s1_4_ir): (super::cm_ir::CmIr, Vec<[IrVarRef; 33]>, Vec<[IrVarRef; 33]>) = {
                            let base_asg: &[F257] = glue.gb.assignment.as_slice();
                            let mut ib = if glue.gb.is_count_only() {
                                IrBuilder::new_count_only(base_asg)
                            } else {
                                IrBuilder::new(base_asg)
                            };
                            let mut s0_4_ir: Vec<[IrVarRef; 33]> = Vec::with_capacity(kappa);
                            let mut s1_4_ir: Vec<[IrVarRef; 33]> = Vec::with_capacity(kappa);
                            for j in 0..kappa {
                                let s0_16: [IrVarRef; 17] =
                                    core::array::from_fn(|k| IrVarRef::Base(tensor_c0_scalars[j][k]));
                                let s1_16: [IrVarRef; 17] =
                                    core::array::from_fn(|k| IrVarRef::Base(tensor_c1_scalars[j][k]));
                                s0_4_ir.push(ib.bal16_to_bal4_digits_cached(&s0_16));
                                s1_4_ir.push(ib.bal16_to_bal4_digits_cached(&s1_16));
                            }
                            (ib.ir, s0_4_ir, s1_4_ir)
                        };
                        let lowered = lower_ir_into_builder(&mut glue.gb, ir);
                        let s0_4_base: Vec<[usize; 33]> =
                            s0_4_ir.iter().map(|d| core::array::from_fn(|k| lowered.map_var(d[k]))).collect();
                        let s1_4_base: Vec<[usize; 33]> =
                            s1_4_ir.iter().map(|d| core::array::from_fn(|k| lowered.map_var(d[k]))).collect();
                        (s0_4_base, s1_4_base)
                    };

                    // Build per-l IR shards in parallel, but stream inputs in **batches** to bound peak memory.
                    use super::cm_ir::{
                        goldilocks_add_mod_p_digits_bal4_ir, goldilocks_mul_mod_p_digits_bal4_ir,
                        lower_ir_into_builder, IrBuilder, VarRef as IrVarRef,
                    };
                    let p_u64 = crate::we_goldilocks_poseidon_f257::GOLDILOCKS_P;

                    #[inline]
                    fn ringdigits64_to_ir(a: &RingDigits) -> Result<[[IrVarRef; 17]; 64], String> {
                        if a.len() != 64 {
                            return Err("tiny gate: expected ring element with 64 coeffs (tcch bal4 shards)".to_string());
                        }
                        Ok(core::array::from_fn(|i| core::array::from_fn(|j| IrVarRef::Base(a[i][j]))))
                    }

                    let l_batch: usize = (8 * rayon::current_num_threads().max(1)).max(1);
                    for l0 in (0..l_instances_expected).step_by(l_batch) {
                        let l1 = (l0 + l_batch).min(l_instances_expected);
                        let batch_len = l1 - l0;

                        // Parse this batch's `comh` terms sequentially (needs &mut glue).
                        let mut comh_terms: Vec<RingDigits> = Vec::with_capacity(batch_len * kappa);
                        for l in l0..l1 {
                            for j in 0..kappa {
                                let idx = l * kappa + j;
                                let (st, ln) = comh_absorbs[idx];
                                let rb = parse_ring_elem_absorb_as_ringbytes(&mut glue, pose_wiring, ring_dim, st, ln)?;
                                let comh_j = ring_bytes_to_digits(&mut glue.gb, &rb);
                                comh_terms.push(comh_j);
                            }
                        }

                        // Snapshot assignment for parallel IR building.
                        //
                        // IMPORTANT: keep this as a normal shared slice borrow so Rust prevents concurrent mutation
                        // of `glue.gb.assignment` while Rayon threads are reading witness values.
                        let base_asg: &[F257] = &glue.gb.assignment;
                        let count_only = glue.gb.is_count_only();

                        // Build shards for this batch in parallel (indexed order preserved).
                        let frags: Vec<(_, [[IrVarRef; 17]; 64], [[IrVarRef; 17]; 64])> = (0..batch_len)
                            .into_par_iter()
                            .map(|l_local| -> Result<_, String> {
                                let mut ib = if count_only {
                                    IrBuilder::new_count_only(base_asg)
                                } else {
                                    IrBuilder::new(base_asg)
                                };
                                // zero digits in bal4: fixed const-0 var replicated.
                                let z = ib.new_var(F257::ZERO);
                                ib.ir.enforce_var_eq_const(z, F257::ZERO);
                                let mut acc0_4: [[IrVarRef; 33]; 64] = [[z; 33]; 64];
                                let mut acc1_4: [[IrVarRef; 33]; 64] = [[z; 33]; 64];

                                for j in 0..kappa {
                                    let term = &comh_terms[l_local * kappa + j];
                                    let term16 = ringdigits64_to_ir(term)?;
                                    let s0_4: [IrVarRef; 33] =
                                        core::array::from_fn(|k| IrVarRef::Base(s0_4_base[j][k]));
                                    let s1_4: [IrVarRef; 33] =
                                        core::array::from_fn(|k| IrVarRef::Base(s1_4_base[j][k]));
                                    for i in 0..64 {
                                        let r4 = ib.bal16_to_bal4_digits_cached(&term16[i]);
                                        let p0 = goldilocks_mul_mod_p_digits_bal4_ir(&mut ib, &r4, &s0_4, p_u64);
                                        let p1 = goldilocks_mul_mod_p_digits_bal4_ir(&mut ib, &r4, &s1_4, p_u64);
                                        acc0_4[i] = goldilocks_add_mod_p_digits_bal4_ir(&mut ib, &acc0_4[i], &p0, p_u64);
                                        acc1_4[i] = goldilocks_add_mod_p_digits_bal4_ir(&mut ib, &acc1_4[i], &p1, p_u64);
                                    }
                                }

                                let mut out0_16: [[IrVarRef; 17]; 64] = [[IrVarRef::Base(0); 17]; 64];
                                let mut out1_16: [[IrVarRef; 17]; 64] = [[IrVarRef::Base(0); 17]; 64];
                                for i in 0..64 {
                                    out0_16[i] = ib.bal4_to_bal16_digits_cached(&acc0_4[i]);
                                    out1_16[i] = ib.bal4_to_bal16_digits_cached(&acc1_4[i]);
                                }
                                Ok((ib.ir, out0_16, out1_16))
                            })
                            .collect::<Result<Vec<_>, _>>()?;

                        // Lower + append results in batch order.
                        for (ir, out0_ir, out1_ir) in frags {
                            let lowered = lower_ir_into_builder(&mut glue.gb, ir);
                            let out0: RingDigits = {
                                let out: [GoldilocksScalar; 64] =
                                    core::array::from_fn(|i| core::array::from_fn(|j| lowered.map_var(out0_ir[i][j])));
                                out.into_iter().collect()
                            };
                            let out1: RingDigits = {
                                let out: [GoldilocksScalar; 64] =
                                    core::array::from_fn(|i| core::array::from_fn(|j| lowered.map_var(out1_ir[i][j])));
                                out.into_iter().collect()
                            };
                            tcch0.push(out0);
                            tcch1.push(out1);
                        }
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
                    #[derive(Clone)]
                    struct ClaimedSumLData {
                        l_idx: usize,
                        eval_a: Vec<GoldilocksScalar>,
                        eval_b: Vec<RingDigits>,
                        eval_c: Vec<RingDigits>,
                        u_l: Vec<RingDigits>,
                    }

                    // Build claimed_sum in batches to bound peak memory.
                    let l_batch: usize = (4 * rayon::current_num_threads().max(1)).max(1);
                    let mut claimed_sum = super::cm_math::ring_zero_digits(&mut glue.gb, ring_dim);

                    for l0 in (0..l_instances_expected).step_by(l_batch) {
                        let l1 = (l0 + l_batch).min(l_instances_expected);
                        let mut per_l: Vec<(usize, ClaimedSumLData)> = Vec::with_capacity(l1 - l0);

                        // Collect this batch's per-l inputs (imports + u computation) sequentially.
                        for l in l0..l1 {
                            let ev = &dcom_evals_base[l];
                        let l_idx = l * (4 + 4 * (params.mlen as usize));

                        // Import eval vectors from base glue vars.
                        let eval_a: Vec<GoldilocksScalar> =
                            ev.a.iter().map(|s| import_scalar(&mut glue, base_asg, s)).collect();
                        let eval_b: Vec<RingDigits> = ev.b.iter().map(|r| import_ring(&mut glue, base_asg, r)).collect();
                        let eval_c: Vec<RingDigits> = ev.c.iter().map(|r| import_ring(&mut glue, base_asg, r)).collect();

                        // u_l for each ni (length rows_per_l).
                        //
                        // IMPORTANT: In the full WE gate, `u` is computed once and reused across the two CM
                        // sumchecks. The tiny gate has two per-`which` CM modules; to avoid duplicating the
                        // heavy negacyclic ring-muls, we import a base-glue precomputation when available.
                        let rows = cm_shared_base
                            .u
                            .get(l)
                            .ok_or_else(|| "tiny gate: cm_shared_base.u missing l row".to_string())?;
                        if rows.len() != rows_per_l {
                            return Err("tiny gate: cm_shared_base.u rows_per_l mismatch".to_string());
                        }
                        let u_l: Vec<RingDigits> = rows.iter().map(|r| import_ring(&mut glue, base_asg, r)).collect();

                        per_l.push((l, ClaimedSumLData {
                            l_idx,
                            eval_a,
                            eval_b,
                            eval_c,
                            u_l,
                        }));
                        }

                    // Build this batch's claimed_sum contributions as IR shards in parallel,
                    // then lower sequentially and accumulate in digit domain.
                    //
                    // This matches the LF+ verifier math term-for-term. We build the arithmetic in **bal4**
                    // (accumulate in bal4, convert to bal16 once per coefficient at the end) to avoid repeated
                    // bal16<->bal4 conversions inside tight loops and to match the other optimized CM paths.
                    {
                        use super::cm_ir::{
                            goldilocks_add_mod_p_digits_bal4_ir, goldilocks_mul_mod_p_digits_bal4_ir,
                            lower_ir_into_builder, IrBuilder, VarRef as IrVarRef,
                        };
                        let p_u64 = crate::we_goldilocks_poseidon_f257::GOLDILOCKS_P;

                        // Snapshot assignment for parallel IR building.
                        //
                        // IMPORTANT: keep this as a normal shared slice borrow so Rust prevents concurrent mutation
                        // of `glue.gb.assignment` while Rayon threads are reading witness values.
                        let base_asg: &[F257] = &glue.gb.assignment;
                        let count_only = glue.gb.is_count_only();

                        #[inline]
                        fn ringdigits64_to_ir(a: &RingDigits) -> Result<[[IrVarRef; 17]; 64], String> {
                            if a.len() != 64 {
                                return Err("tiny gate: expected ring element with 64 coeffs (claimed_sum bal4 shards)".to_string());
                            }
                            Ok(core::array::from_fn(|i| core::array::from_fn(|j| IrVarRef::Base(a[i][j]))))
                        }

                        let frags: Vec<(_, [[IrVarRef; 17]; 64])> = per_l
                            .par_iter()
                            .map(|(l, data)| -> Result<_, String> {
                                let l = *l;
                                debug_assert_eq!(data.l_idx, l * (4 + 4 * (params.mlen as usize)));
                                let mut ib = if count_only {
                                    IrBuilder::new_count_only(base_asg)
                                } else {
                                    IrBuilder::new(base_asg)
                                };

                                // Zero digits: fixed const-0 var replicated.
                                let z = ib.new_var(F257::ZERO);
                                ib.ir.enforce_var_eq_const(z, F257::ZERO);
                                let mut acc4: [[IrVarRef; 33]; 64] = [[z; 33]; 64];

                                #[inline]
                                fn scalar16_to_ir(x: &GoldilocksScalar) -> [IrVarRef; 17] {
                                    core::array::from_fn(|j| IrVarRef::Base(x[j]))
                                }

                                // Helper: add (ring16 * scalar16) into acc4 coefficientwise (bal4 mod-p).
                                #[inline]
                                fn add_scaled_ring_into_acc4(
                                    ib: &mut IrBuilder<'_>,
                                    acc4: &mut [[IrVarRef; 33]; 64],
                                    ring16: &RingDigits,
                                    scalar16: &[IrVarRef; 17],
                                    p_u64: u64,
                                ) -> Result<(), String> {
                                    let ring16_ir = ringdigits64_to_ir(ring16)?;
                                    let s4 = ib.bal16_to_bal4_digits_cached(scalar16);
                                    for i in 0..64 {
                                        let r4 = ib.bal16_to_bal4_digits_cached(&ring16_ir[i]);
                                        let prod4 = goldilocks_mul_mod_p_digits_bal4_ir(ib, &r4, &s4, p_u64);
                                        acc4[i] = goldilocks_add_mod_p_digits_bal4_ir(ib, &acc4[i], &prod4, p_u64);
                                    }
                                    Ok(())
                                }

                                // a0 term: (eval_a[0] * rc^l_idx) goes into coefficient 0 only.
                                let a0_16 = scalar16_to_ir(&data.eval_a[0]);
                                let rc0_16: [IrVarRef; 17] =
                                    core::array::from_fn(|k| IrVarRef::Base(rc_pows[data.l_idx][k]));
                                let a0_4 = ib.bal16_to_bal4_digits_cached(&a0_16);
                                let rc0_4 = ib.bal16_to_bal4_digits_cached(&rc0_16);
                                let a0pow4 = goldilocks_mul_mod_p_digits_bal4_ir(&mut ib, &a0_4, &rc0_4, p_u64);
                                acc4[0] = goldilocks_add_mod_p_digits_bal4_ir(&mut ib, &acc4[0], &a0pow4, p_u64);

                                // b0/c0/u0 terms.
                                let s_b0: [IrVarRef; 17] =
                                    core::array::from_fn(|k| IrVarRef::Base(rc_pows[data.l_idx + 1][k]));
                                let s_c0: [IrVarRef; 17] =
                                    core::array::from_fn(|k| IrVarRef::Base(rc_pows[data.l_idx + 2][k]));
                                let s_u0: [IrVarRef; 17] =
                                    core::array::from_fn(|k| IrVarRef::Base(rc_pows[data.l_idx + 3][k]));
                                add_scaled_ring_into_acc4(&mut ib, &mut acc4, &data.eval_b[0], &s_b0, p_u64)?;
                                add_scaled_ring_into_acc4(&mut ib, &mut acc4, &data.eval_c[0], &s_c0, p_u64)?;
                                add_scaled_ring_into_acc4(&mut ib, &mut acc4, &data.u_l[0], &s_u0, p_u64)?;

                                // M rows.
                                for i in 0..(params.mlen as usize) {
                                    let idx = data.l_idx + 4 + i * 4;

                                    // ai term: scalar into coeff 0 only.
                                    let ai_16 = scalar16_to_ir(&data.eval_a[1 + i]);
                                    let rci_16: [IrVarRef; 17] =
                                        core::array::from_fn(|k| IrVarRef::Base(rc_pows[idx][k]));
                                    let ai_4 = ib.bal16_to_bal4_digits_cached(&ai_16);
                                    let rci_4 = ib.bal16_to_bal4_digits_cached(&rci_16);
                                    let aipow4 = goldilocks_mul_mod_p_digits_bal4_ir(&mut ib, &ai_4, &rci_4, p_u64);
                                    acc4[0] = goldilocks_add_mod_p_digits_bal4_ir(&mut ib, &acc4[0], &aipow4, p_u64);

                                    let s_bi: [IrVarRef; 17] =
                                        core::array::from_fn(|k| IrVarRef::Base(rc_pows[idx + 1][k]));
                                    let s_ci: [IrVarRef; 17] =
                                        core::array::from_fn(|k| IrVarRef::Base(rc_pows[idx + 2][k]));
                                    let s_ui: [IrVarRef; 17] =
                                        core::array::from_fn(|k| IrVarRef::Base(rc_pows[idx + 3][k]));
                                    add_scaled_ring_into_acc4(
                                        &mut ib,
                                        &mut acc4,
                                        &data.eval_b[1 + i],
                                        &s_bi,
                                        p_u64,
                                    )?;
                                    add_scaled_ring_into_acc4(
                                        &mut ib,
                                        &mut acc4,
                                        &data.eval_c[1 + i],
                                        &s_ci,
                                        p_u64,
                                    )?;
                                    add_scaled_ring_into_acc4(
                                        &mut ib,
                                        &mut acc4,
                                        &data.u_l[1 + i],
                                        &s_ui,
                                        p_u64,
                                    )?;
                                }

                                // tcch0/tcch1 terms (same z_idx across all l).
                                let s_tc0: [IrVarRef; 17] =
                                    core::array::from_fn(|k| IrVarRef::Base(rc_pows[z_idx][k]));
                                let s_tc1: [IrVarRef; 17] =
                                    core::array::from_fn(|k| IrVarRef::Base(rc_pows[z_idx + 1][k]));
                                add_scaled_ring_into_acc4(&mut ib, &mut acc4, &tcch0[l], &s_tc0, p_u64)?;
                                add_scaled_ring_into_acc4(&mut ib, &mut acc4, &tcch1[l], &s_tc1, p_u64)?;

                                // Convert to bal16 once (required by downstream digit-domain gadgets).
                                let mut out16: [[IrVarRef; 17]; 64] = [[IrVarRef::Base(0); 17]; 64];
                                for coeff in 0..64 {
                                    out16[coeff] = ib.bal4_to_bal16_digits_cached(&acc4[coeff]);
                                }
                                Ok((ib.ir, out16))
                            })
                            .collect::<Result<Vec<_>, _>>()?;

                        for (ir, out_ir) in frags {
                            let lowered = lower_ir_into_builder(&mut glue.gb, ir);
                            let out: RingDigits = {
                                let out: [GoldilocksScalar; 64] = core::array::from_fn(|i| {
                                    core::array::from_fn(|j| lowered.map_var(out_ir[i][j]))
                                });
                                out.into_iter().collect()
                            };
                            claimed_sum = ring_add_digits(&mut glue.gb, &claimed_sum, &out);
                        }
                    }
                    } // end l-batch loop

                    claimed_sum_opt = Some(claimed_sum);
                }
        }
    }

    let claimed_sum = claimed_sum_opt.ok_or_else(|| {
        "tiny gate: CM claimed_sum missing (full faithful relation requires SetChk/Dcom wiring + pow2 preconditions)"
            .to_string()
    })?;
    let subclaim_eval = super::cm_math::sumcheck_verify_degree2_ring_digits(&mut glue.gb, claimed_sum, &msgs_digits, &rs_digits)?;

    // Eval table absorbs sanity.
    let rows_total = l_instances_expected * (1 + params.mlen as usize);
    if eval_absorbs.len() != rows_total * 4 {
        return Err("tiny gate: eval absorb count mismatch".to_string());
    }

    // Recombination check (requires the pow2 regime + recovered setchk r-point).
    {
        // rc powers (need up to z_idx+1).
        let z_idx = l_instances_expected * (4 + 4 * (params.mlen as usize));
        let max_pow = z_idx + 1;
        let rc_d = goldilocks_bytes_to_digits(&mut glue.gb, rc_bytes);
        let rc_pows = goldilocks_pow_table_digits(&mut glue.gb, &rc_d, max_pow);

        // eq(r, ro) where r is the transcript-derived SetChk point (recovered above).
        let eq = eq_eval_goldilocks_digits(&mut glue.gb, &r_point_digits, &rs_digits)?;

        // Evaluate t0(ro), t1(ro) directly as **bal4** digits (avoids bal16->bal4 conversion later).
        let (t0_4_base, t1_4_base) = eval_t_z_optimized_ring_digits_pair(
            &mut glue.gb,
            &tensor_c0_ring,
            &tensor_c1_ring,
            &s_prime_flat_ring,
            &dpp_ring,
            ring_dim,
            &rs_digits,
        )?;

        // Stream recombination.
        //
        // Big optimization: instead of `ring_scale_digits` (mul_mod_p + bal16<->bal4 conversions) per term,
        // build one IR shard per `l` that:
        // - converts each ring coefficient + scalar multiplier into bal4 once,
        // - does all muls + sums in bal4,
        // - converts to bal16 once at the end.
        //
        // This is also embarrassingly parallel across `l`.
        let rows_per_l = 1 + params.mlen as usize;

        // Parse + recombine in **batches** to bound peak memory.
        // We only need to keep `e00s` for the later t(z) terms.
        let mut e00s: Vec<RingDigits> = Vec::with_capacity(l_instances_expected);
        let mut eval_acc = super::cm_math::ring_zero_digits(&mut glue.gb, ring_dim);

        // We'll batch over `l` so we don't materialize all eval table ring elements at once.
        let l_batch: usize = (8 * rayon::current_num_threads().max(1)).max(1);

        // Precompute `eq` in bal4 once (shared across all per-l shards).
        let eq4_base: [usize; 33] = {
            use super::cm_ir::{IrBuilder, VarRef as IrVarRef};
            // Avoid borrowing `glue.gb.assignment` across lowering (which needs `&mut glue.gb`).
            let (ir, eq4_ir): (super::cm_ir::CmIr, [IrVarRef; 33]) = {
                let base_asg: &[F257] = glue.gb.assignment.as_slice();
                let mut ib = if glue.gb.is_count_only() {
                    IrBuilder::new_count_only(base_asg)
                } else {
                    IrBuilder::new(base_asg)
                };
                let eq16: [IrVarRef; 17] = core::array::from_fn(|j| IrVarRef::Base(eq[j]));
                let eq4 = ib.bal16_to_bal4_digits_cached(&eq16);
                (ib.ir, eq4)
            };
            let lowered = super::cm_ir::lower_ir_into_builder(&mut glue.gb, ir);
            core::array::from_fn(|k| lowered.map_var(eq4_ir[k]))
        };

            // Build recombination shards in parallel per batch, then lower + accumulate.
            for l0 in (0..l_instances_expected).step_by(l_batch) {
                let l1 = (l0 + l_batch).min(l_instances_expected);
                let batch_len = l1 - l0;

                // Parse this batch's eval table ring elements sequentially (needs `&mut glue`).
                let mut batch_terms: Vec<Vec<RingDigits>> = Vec::with_capacity(batch_len);
                for l in l0..l1 {
                    let mut terms: Vec<RingDigits> = Vec::with_capacity(rows_per_l * 4);
                    let mut e00_opt: Option<RingDigits> = None;
                    for row in 0..rows_per_l {
                        let flat = l * rows_per_l + row;
                        for j in 0..4 {
                            let (st, ln) = eval_absorbs[flat * 4 + j];
                            let rb = parse_ring_elem_absorb_as_ringbytes(&mut glue, pose_wiring, ring_dim, st, ln)?;
                            let rd = ring_bytes_to_digits(&mut glue.gb, &rb);
                            if row == 0 && j == 0 {
                                e00_opt = Some(rd.clone());
                            }
                            terms.push(rd);
                        }
                    }
                    let e00 = e00_opt.ok_or_else(|| "tiny gate: missing e00 in recombination".to_string())?;
                    e00s.push(e00);
                    batch_terms.push(terms);
                }

                let frags = {
                use super::cm_ir::{
                    goldilocks_add_mod_p_digits_bal4_ir, goldilocks_mul_mod_p_digits_bal4_ir,
                    IrBuilder, VarRef as IrVarRef,
                };

                #[inline]
                fn ringdigits64_to_ir(a: &RingDigits) -> Result<[[IrVarRef; 17]; 64], String> {
                    if a.len() != 64 {
                        return Err("tiny gate: expected ring element with 64 coeffs (bal4 recombination)".to_string());
                    }
                    Ok(core::array::from_fn(|i| core::array::from_fn(|j| IrVarRef::Base(a[i][j]))))
                }

                let p_u64 = crate::we_goldilocks_poseidon_f257::GOLDILOCKS_P;
                let base_asg: &[F257] = &glue.gb.assignment;
                let count_only = glue.gb.is_count_only();
                let eq4_ir: [IrVarRef; 33] = core::array::from_fn(|k| IrVarRef::Base(eq4_base[k]));

                let frags: Vec<(super::cm_ir::CmIr, [[IrVarRef; 17]; 64])> = (0..batch_len)
                    .into_par_iter()
                    .map(|l_local| -> Result<_, String> {
                        let l = l0 + l_local;
                        let l_idx = l * (4 + 4 * (params.mlen as usize));
                        let terms = &batch_terms[l_local];
                        debug_assert_eq!(terms.len(), rows_per_l * 4);

                        let mut ib = if count_only {
                            IrBuilder::new_count_only(base_asg)
                        } else {
                            IrBuilder::new(base_asg)
                        };
                        // zero digits in bal4: fixed const-0 var replicated.
                        let z = ib.new_var(F257::ZERO);
                        ib.ir.enforce_var_eq_const(z, F257::ZERO);
                        let mut inner4: [[IrVarRef; 33]; 64] = [[z; 33]; 64];

                        for row in 0..rows_per_l {
                            for j in 0..4 {
                                let term_idx = row * 4 + j;
                                let rd16 = ringdigits64_to_ir(&terms[term_idx])?;
                                let s16: [IrVarRef; 17] =
                                    core::array::from_fn(|k| IrVarRef::Base(rc_pows[l_idx + row * 4 + j][k]));
                                let s4 = ib.bal16_to_bal4_digits_cached(&s16);
                                for coeff in 0..64 {
                                    let rd4 = ib.bal16_to_bal4_digits_cached(&rd16[coeff]);
                                    let prod4 = goldilocks_mul_mod_p_digits_bal4_ir(&mut ib, &rd4, &s4, p_u64);
                                    inner4[coeff] =
                                        goldilocks_add_mod_p_digits_bal4_ir(&mut ib, &inner4[coeff], &prod4, p_u64);
                                }
                            }
                        }

                        // Multiply by eq and convert to bal16 once.
                        let mut out16: [[IrVarRef; 17]; 64] = [[IrVarRef::Base(0); 17]; 64];
                        for coeff in 0..64 {
                            let prod4 = goldilocks_mul_mod_p_digits_bal4_ir(&mut ib, &inner4[coeff], &eq4_ir, p_u64);
                            out16[coeff] = ib.bal4_to_bal16_digits_cached(&prod4);
                        }
                        Ok((ib.ir, out16))
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                frags
                };

                for (ir, out16_ir) in frags {
                    let lowered = super::cm_ir::lower_ir_into_builder(&mut glue.gb, ir);
                    let contrib: RingDigits = {
                        let out: [GoldilocksScalar; 64] = core::array::from_fn(|i| {
                            core::array::from_fn(|j| lowered.map_var(out16_ir[i][j]))
                        });
                        out.into_iter().collect()
                    };
                    eval_acc = ring_add_digits(&mut glue.gb, &eval_acc, &contrib);
                }
            }
        
        // t(z) terms: for each l, add t0(ro)*e00(l)*rc^z + t1(ro)*e00(l)*rc^{z+1}.
        //
        // These ring-muls are independent across l, so we build them as IR fragments in parallel,
        // then lower sequentially into this module's builder.
        {
            use super::cm_ir::{
                goldilocks_add_mod_p_digits_bal4_ir, goldilocks_mul_mod_p_digits_bal4_ir,
                lower_ir_into_builder, ring_mul_negacyclic_ntt_goldilocks_d64_bal4_ir, IrBuilder, VarRef as IrVarRef,
            };

            #[inline]
            fn ringdigits64_to_ir(a: &RingDigits) -> Result<[[IrVarRef; 17]; 64], String> {
                if a.len() != 64 {
                    return Err("tiny gate: expected ring element with 64 coeffs (IR ring-mul)".to_string());
                }
                Ok(core::array::from_fn(|i| core::array::from_fn(|j| IrVarRef::Base(a[i][j]))))
            }

            #[inline]
            fn map_ring_out(out_ir: &[[IrVarRef; 17]; 64], lowered: &super::cm_ir::LoweredIr) -> RingDigits {
                let out: [GoldilocksScalar; 64] =
                    core::array::from_fn(|i| core::array::from_fn(|j| lowered.map_var(out_ir[i][j])));
                out.into_iter().collect()
            }

            // Precompute rc^z scalars in bal4 once.
            // Avoid borrowing `glue.gb.assignment` across `lower_ir_into_builder(&mut glue.gb, ...)`.
            // IMPORTANT: after lowering this precompute IR, we must take a *fresh* snapshot for the
            // per-`l` shards, since they will read witness values of the newly allocated base vars.
            let (rcz4_base, rcz14_base): ([usize; 33], [usize; 33]) = {
                let rcz16: [IrVarRef; 17] = core::array::from_fn(|k| IrVarRef::Base(rc_pows[z_idx][k]));
                let rcz116: [IrVarRef; 17] = core::array::from_fn(|k| IrVarRef::Base(rc_pows[z_idx + 1][k]));
                let (ir, rcz4_ir, rcz14_ir) = {
                    let base_asg = glue.gb.assignment.as_slice();
                    let mut ib = if glue.gb.is_count_only() {
                        IrBuilder::new_count_only(base_asg)
                    } else {
                        IrBuilder::new(base_asg)
                    };
                    let rcz4 = ib.bal16_to_bal4_digits_cached(&rcz16);
                    let rcz14 = ib.bal16_to_bal4_digits_cached(&rcz116);
                    (ib.ir, rcz4, rcz14)
                };
                let lowered = lower_ir_into_builder(&mut glue.gb, ir);
                let rcz4_base: [usize; 33] = core::array::from_fn(|k| lowered.map_var(rcz4_ir[k]));
                let rcz14_base: [usize; 33] = core::array::from_fn(|k| lowered.map_var(rcz14_ir[k]));
                (rcz4_base, rcz14_base)
            };

            let p_u64 = crate::we_goldilocks_poseidon_f257::GOLDILOCKS_P;

            // Snapshot assignment for parallel IR building.
            //
            // IMPORTANT: keep this as a normal shared slice borrow so Rust prevents concurrent mutation
            // of `glue.gb.assignment` while Rayon threads are reading witness values.
            let base_asg: &[F257] = &glue.gb.assignment;
            let count_only = glue.gb.is_count_only();

            let frags: Vec<(_, [[IrVarRef; 17]; 64])> = e00s
                .par_iter()
                .map(|e00| -> Result<_, String> {
                    // Build one shard that does:
                    //  - ring-mul in bal4
                    //  - scale in bal4 by rc^z / rc^{z+1}
                    //  - add in bal4, then convert once to bal16 for accumulation
                    let e00_16 = ringdigits64_to_ir(e00)?;
                    let t0_4: [[IrVarRef; 33]; 64] =
                        core::array::from_fn(|i| core::array::from_fn(|k| IrVarRef::Base(t0_4_base[i][k])));
                    let t1_4: [[IrVarRef; 33]; 64] =
                        core::array::from_fn(|i| core::array::from_fn(|k| IrVarRef::Base(t1_4_base[i][k])));
                    let rcz4: [IrVarRef; 33] = core::array::from_fn(|k| IrVarRef::Base(rcz4_base[k]));
                    let rcz14: [IrVarRef; 33] = core::array::from_fn(|k| IrVarRef::Base(rcz14_base[k]));

                    let mut ib = if count_only {
                        IrBuilder::new_count_only(base_asg)
                    } else {
                        IrBuilder::new(base_asg)
                    };
                    let e00_4: [[IrVarRef; 33]; 64] = core::array::from_fn(|i| ib.bal16_to_bal4_digits_cached(&e00_16[i]));

                    let out0_4 = ring_mul_negacyclic_ntt_goldilocks_d64_bal4_ir(&mut ib, &t0_4, &e00_4);
                    let out1_4 = ring_mul_negacyclic_ntt_goldilocks_d64_bal4_ir(&mut ib, &t1_4, &e00_4);
                    super::op_counts::tiny_cm_bump(|c| c.ring_mul_negacyclic += 2);

                    let mut out16: [[IrVarRef; 17]; 64] = [[IrVarRef::Base(0); 17]; 64];
                    for i in 0..64 {
                        let s0 = goldilocks_mul_mod_p_digits_bal4_ir(&mut ib, &out0_4[i], &rcz4, p_u64);
                        let s1 = goldilocks_mul_mod_p_digits_bal4_ir(&mut ib, &out1_4[i], &rcz14, p_u64);
                        let sum4 = goldilocks_add_mod_p_digits_bal4_ir(&mut ib, &s0, &s1, p_u64);
                        out16[i] = ib.bal4_to_bal16_digits_cached(&sum4);
                    }
                    Ok((ib.ir, out16))
                })
                .collect::<Result<Vec<_>, _>>()?;

            for (ir, out_ir) in frags {
                let lowered = lower_ir_into_builder(&mut glue.gb, ir);
                let contrib = map_ring_out(&out_ir, &lowered);
                eval_acc = ring_add_digits(&mut glue.gb, &eval_acc, &contrib);
            }
        }

        ring_eq_digits(&mut glue.gb, &subclaim_eval, &eval_acc);
    }

    Ok(glue)
}

#[cfg(unix)]
#[allow(clippy::too_many_arguments)]
fn build_direct_to_merged_unix(
    cfg: Option<&PoseidonConfig<F257>>,
    ops: &[PoseidonTraceOp<F257>],
    ring_dim: usize,
    params: &WeParams,
    wiring: &TinyCoinOpWiring,
    pairs: &[(usize, usize)],
    extra_witness: &TinyExtraWitness,
    out_dir: impl AsRef<std::path::Path>,
) -> Result<
    (
        FileBackedSparseDr1csInstance<F257>,
        Vec<F257>,
        Vec<ShortChallengeWiring>,
        Vec<BoundedU32ChallengeWiring>,
        Vec<GoldilocksChallengeWiring>,
        Vec<CmDigitMulSurfaceWiring>,
        Vec<CmDigitMulSqSurfaceWiring>,
        PoseidonDr1csWiring,
    ),
    String,
> {
    use std::io::Write;

    let dirs = build_dirs(&out_dir);
    lf_profile_log(&format!(
        "start out_dir={} threads={}",
        dirs.root.display(),
        rayon::current_num_threads().max(1)
    ));
    let t_all = Instant::now();

    let default_cfg;
    let poseidon_cfg = match cfg {
        Some(c) => c,
        None => {
            default_cfg = f257_poseidon_config();
            &default_cfg
        }
    };

    // Pass 0: count plan for the entire tiny-gate (exact sizes + eq_pairs + var offsets).
    let t = Instant::now();
    let plan0 = build_count_plan(poseidon_cfg, ops, ring_dim, params, wiring, pairs, extra_witness)?;
    let plan = Arc::new(plan0);
    lf_profile_log(&format!("Pass0 total elapsed={:?}", t.elapsed()));

    // Op-mix counters should reflect the *final circuit* (Pass1), not Pass0 structural counting.
    if tiny_opmix_on() {
        super::op_counts::tiny_cm_counts_reset();
    }

    // Preallocate merged/* pools/rows.
    let (fc_a, fi_a, fc_b, fi_b, fc_c, fi_c, f_rows) = prealloc_merged_files(
        &dirs.merged_dir,
        plan.total_a_terms,
        plan.total_b_terms,
        plan.total_c_terms,
        plan.total_rows,
    )?;
    let files = Arc::new(TinyGateMergedFiles {
        fc_a,
        fi_a,
        fc_b,
        fi_b,
        fc_c,
        fi_c,
        f_rows,
    });
    let rb = TinyGateRangeBase {
        files: files.clone(),
        plan: plan.clone(),
        part_base: 0,
    };
    lf_profile_log(&format!("Pass1 prealloc merged elapsed={:?}", t_all.elapsed()));

    // Pass 1a: Poseidon sharded into merged ranges (part 0).
    let t = Instant::now();
    let shard_permutes = tiny_gate_poseidon_shard_permutes(poseidon_cfg, ops);
    let (pose_asg, pose_wiring, pose_range) =
        poseidon_sponge_dr1cs_from_ops_with_wiring_no_bytes_range_sharded_into_files::<F257>(
            poseidon_cfg,
            ops,
            shard_permutes,
            &files.fc_a,
            &files.fi_a,
            &files.fc_b,
            &files.fi_b,
            &files.fc_c,
            &files.fi_c,
            &files.f_rows,
            0,
            0,
            0,
            0,
            0,
        )
        .map_err(|e| format!("poseidon(F257) range-write sharded failed: {e}"))?;
    lf_profile_log(&format!(
        "Pass1 poseidon_range elapsed={:?} shard_permutes={}",
        t.elapsed(),
        shard_permutes
    ));
    // Validate poseidon counts match Pass0.
    let exp_pose_rows = plan.row_off.get(1).copied().unwrap_or(0);
    let exp_pose_a = plan.a_off.get(1).copied().unwrap_or(0);
    let exp_pose_b = plan.b_off.get(1).copied().unwrap_or(0);
    let exp_pose_c = plan.c_off.get(1).copied().unwrap_or(0);
    if pose_range.rows != exp_pose_rows
        || pose_range.a_terms != exp_pose_a
        || pose_range.b_terms != exp_pose_b
        || pose_range.c_terms != exp_pose_c
    {
        return Err(format!(
            "tiny gate: Pass0/Pass1 poseidon count mismatch: rows got {} exp {}, a_terms got {} exp {}, b_terms got {} exp {}, c_terms got {} exp {}",
            pose_range.rows, exp_pose_rows, pose_range.a_terms, exp_pose_a, pose_range.b_terms, exp_pose_b, pose_range.c_terms, exp_pose_c
        ));
    }

    // Base glue (part 1) and all submodules write directly into merged ranges.
    let mut glue = GlueCtx::new(
        &TinyGateBuildMode::RangeBase,
        pose_asg.as_slice(),
        &dirs.base_glue_dir,
        1,
        Some(&rb),
    )?;

    // Canonicality constraints: parallel shards (parts 2..).
    let t = Instant::now();
    let canonical_glues = build_canonicality_shards(
        &TinyGateBuildMode::RangeBase,
        2,
        Some(&rb),
        pose_asg.as_slice(),
        ops,
        &pose_wiring,
        &dirs,
    )?;
    lf_profile_log(&format!(
        "Pass1 canonicality_shards elapsed={:?} parts={}",
        t.elapsed(),
        canonical_glues.len()
    ));

    let t = Instant::now();
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
    lf_profile_log(&format!(
        "Pass1 cm_schedule+comh_count elapsed={:?} l_instances_expected={}",
        t.elapsed(),
        l_instances_expected
    ));

    let t = Instant::now();
    let short_ranges =
        squeeze_field_ranges_by_op_index(&pose_wiring.squeeze_field_ranges, &wiring.short_squeeze_ops)?;
    let u32_ranges =
        squeeze_field_ranges_by_op_index(&pose_wiring.squeeze_field_ranges, &wiring.u32_squeeze_ops)?;
    validate_pairs(pairs, short_ranges.len(), u32_ranges.len())?;
    validate_params_and_short_schedule(ring_dim, params, short_ranges.len())?;
    lf_profile_log(&format!("Pass1 ranges+validate elapsed={:?}", t.elapsed()));

    let t = Instant::now();
    let short_locals = build_short_blocks(&mut glue, &pose_wiring, ring_dim, &short_ranges)?;
    let (u32_locals, goldilocks_locals) =
        build_u32_and_goldilocks_blocks(&mut glue, &pose_wiring, &wiring.u32_squeeze_ops)?;
    lf_profile_log(&format!(
        "Pass1 build_short/u32/goldilocks elapsed={:?} short_locals={} u32_locals={} goldilocks_locals={}",
        t.elapsed(),
        short_locals.len(),
        u32_locals.len(),
        goldilocks_locals.len()
    ));

    // Values parsed/allocated during SetChk/RgChk that the CM verifier math needs later.
    let mut setchk_out_e_vars_for_cm: Option<Vec<Vec<Vec<RingDigits>>>> = None;
    let mut dcom_evals_for_cm: Option<Vec<DcomEvalDigits>> = None;
    let mut fcoms_for_cm: Option<Vec<FComsDigits>> = None;
    let mut setchk_r_point_for_cm: Option<Arc<Vec<GoldilocksScalar>>> = None;
    let t = Instant::now();
    arithmetize_pi_lin_setchk_rgchk_prefix(
        &mut glue,
        ops,
        &pose_wiring,
        ring_dim,
        params,
        wiring,
        l_instances_expected,
        &u32_locals,
        extra_witness,
        &mut setchk_out_e_vars_for_cm,
        &mut dcom_evals_for_cm,
        &mut fcoms_for_cm,
        &mut setchk_r_point_for_cm,
    )?;
    lf_profile_log(&format!("Pass1 pi/lin/setchk/rgchk_prefix elapsed={:?}", t.elapsed()));
    let t = Instant::now();
    let (comh_absorbs, sc_msg_absorbs, eval_absorbs) = parse_and_enforce_cm_after_short(
        &mut glue,
        ops,
        &pose_wiring,
        ring_dim,
        params,
        wiring,
        l_instances_expected,
        &goldilocks_locals,
    )?;
    lf_profile_log(&format!("Pass1 parse+enforce_cm_after_short elapsed={:?}", t.elapsed()));

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

    let setchk_out_e_vars_for_cm = setchk_out_e_vars_for_cm.map(Arc::new);
    let dcom_evals_for_cm = dcom_evals_for_cm.map(Arc::new);
    let t = Instant::now();
    let cm_shared_base: Arc<CmSharedPrecompBase> = compute_cm_shared_precomp_base(
        &mut glue,
        ring_dim,
        params,
        wiring,
        l_instances_expected,
        &short_locals,
        &u32_locals,
        &setchk_out_e_vars_for_cm,
        &setchk_r_point_for_cm,
    )?;
    lf_profile_log(&format!("Pass1 cm_shared_precomp_base elapsed={:?}", t.elapsed()));

    // CM modules: parts (2+canon_len) and (2+canon_len+1).
    let t = Instant::now();
    let cm_extra_glues = build_cm_shards(
        &TinyGateBuildMode::RangeBase,
        2 + canonical_glues.len(),
        Some(&rb),
        &pose_wiring,
        ring_dim,
        params,
        wiring,
        l_instances_expected,
        &comh_absorbs,
        &sc_msg_absorbs,
        &eval_absorbs,
        setchk_out_e_vars_for_cm.clone(),
        dcom_evals_for_cm.clone(),
        cm_shared_base.clone(),
        glue.pose_asg,
        glue.gb.assignment.as_slice(),
        &goldilocks_locals,
        &dirs,
    )?;
    lf_profile_log(&format!(
        "Pass1 cm_shards elapsed={:?} parts={}",
        t.elapsed(),
        cm_extra_glues.len()
    ));

    // Decomp verifier math.
    let (cm_g_target, vo_a_target, vo_b_target) = compute_cm_x_targets_for_decomp(
        &mut glue,
        ring_dim,
        params,
        l_instances_expected,
        &cm_shared_base,
        &pose_wiring,
        &comh_absorbs,
        &eval_absorbs,
        &fcoms_for_cm,
    )?;
    add_decomp_linb2x_constraints(
        &mut glue,
        ring_dim,
        params,
        extra_witness,
        &cm_g_target,
        &vo_a_target,
        &vo_b_target,
    )?;

    // Surfaces (comparatively small).
    let (
        surfaces_mul_local,
        surfaces_sq_local,
        all_sum_digits,
        all_sum_coeffwise,
        all_sq_sum_digits,
        all_sq_sum_coeffwise,
    ) = build_surfaces_with_shared_arcs(&mut glue, ring_dim, pairs, &short_locals, &u32_locals)?;

    // Optional: print an op-mix breakdown for tiny-field porting estimates (direct-to-merged build).
    //
    // Enable with: `LFP_WE_GATE_OPMIX=1 ...`
    maybe_print_tiny_opmix_common(
        cfg,
        ops,
        pose_range.rows as usize,
        glue.gb.nconstraints() as usize,
        "[direct-merged]",
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

    // ---------------------------------------------------------------------
    // Collect parts: assignments + ckpts + local maps for equality tail.
    // ---------------------------------------------------------------------
    #[cfg(unix)]
    fn expect_part_delta(off: &[u64], part: usize, total: u64) -> u64 {
        if part + 1 < off.len() {
            off[part + 1].saturating_sub(off[part])
        } else {
            total.saturating_sub(off[part])
        }
    }

    // Base glue range result (part 1).
    let GlueCtx {
        gb,
        pose_asg: _,
        local_map: base_local_map,
        base_eqs: base_base_eqs,
        ..
    } = glue;
    debug_assert!(base_base_eqs.is_empty(), "base glue should not contain base_eqs");
    let (base_asg, base_range) = gb.into_range_result()?;
    let exp_base_rows = expect_part_delta(&plan.row_off, 1, plan.part_rows);
    let exp_base_a = expect_part_delta(&plan.a_off, 1, plan.part_a_terms);
    let exp_base_b = expect_part_delta(&plan.b_off, 1, plan.part_b_terms);
    let exp_base_c = expect_part_delta(&plan.c_off, 1, plan.part_c_terms);
    if base_range.counts.rows != exp_base_rows
        || base_range.counts.a_terms != exp_base_a
        || base_range.counts.b_terms != exp_base_b
        || base_range.counts.c_terms != exp_base_c
    {
        return Err("tiny gate: base glue Pass0/Pass1 count mismatch".to_string());
    }

    // Extra glues: canonical shards then cm shards.
    #[cfg(unix)]
    struct ExtraOut {
        local_map: BTreeMap<usize, usize>,
        base_eqs: Vec<(usize, usize)>,
        asg: Vec<F257>,
        range: symphony::dpp_sumcheck::Dr1csRangeResult,
    }
    let mut extra_out: Vec<ExtraOut> = Vec::new();
    for (i, g) in canonical_glues
        .into_iter()
        .chain(cm_extra_glues.into_iter())
        .enumerate()
    {
        let GlueCtx { gb, pose_asg: _, local_map, base_eqs, .. } = g;
        let (asg, range) = gb.into_range_result()?;
        let part = 2 + i;
        let exp_rows = expect_part_delta(&plan.row_off, part, plan.part_rows);
        let exp_a = expect_part_delta(&plan.a_off, part, plan.part_a_terms);
        let exp_b = expect_part_delta(&plan.b_off, part, plan.part_b_terms);
        let exp_c = expect_part_delta(&plan.c_off, part, plan.part_c_terms);
        if range.counts.rows != exp_rows
            || range.counts.a_terms != exp_a
            || range.counts.b_terms != exp_b
            || range.counts.c_terms != exp_c
        {
            return Err(format!("tiny gate: extra glue Pass0/Pass1 count mismatch at part={part}"));
        }
        extra_out.push(ExtraOut {
            local_map,
            base_eqs,
            asg,
            range,
        });
    }


    // Exact offsets from Pass1 assignments (computed before we move/drop large assignment vectors).
    let pose_tail_len = pose_asg.len().saturating_sub(1);
    let base_tail_len = base_asg.len().saturating_sub(1);
    let extra_tail_lens: Vec<usize> = extra_out.iter().map(|eo| eo.asg.len().saturating_sub(1)).collect();
    let mut offsets: Vec<usize> = Vec::with_capacity(2 + extra_out.len());
    let mut cur = 0usize;
    offsets.push(cur);
    cur += pose_tail_len;
    offsets.push(cur);
    cur += base_tail_len;
    for &len in &extra_tail_lens {
        offsets.push(cur);
        cur += len;
    }
    if offsets != plan.var_tail_off {
        return Err("tiny gate: Pass1 var offsets diverged from Pass0 plan".to_string());
    }

    // Reconstruct merged assignment by **moving** tails (avoid duplicating large Vecs).
    let mut merged_asg: Vec<F257> = Vec::with_capacity(1usize.saturating_add(cur));
    merged_asg.push(F257::ONE);
    {
        let mut pose_asg = pose_asg;
        let mut tail = pose_asg.split_off(1);
        merged_asg.append(&mut tail);
        drop(pose_asg);
    }
    {
        let mut base_asg = base_asg;
        let mut tail = base_asg.split_off(1);
        merged_asg.append(&mut tail);
        drop(base_asg);
    }
    for eo in extra_out.iter_mut() {
        let mut asg = std::mem::take(&mut eo.asg);
        let mut tail = asg.split_off(1);
        merged_asg.append(&mut tail);
        // drop `asg` to free the backing allocation
    }
    let remap = |part: usize, local: usize, offsets: &[usize]| -> usize {
        if local == 0 { 0 } else { local + offsets[part] }
    };

    // Equality pairs as in the canonical direct-to-merged pipeline.
    let mut eq_pairs: Vec<(usize, usize)> = Vec::new();
    {
        let reabsorb = collect_fiat_shamir_reabsorb_eqs(ops, &pose_wiring)?;
        for (v_ab, v_sq) in reabsorb {
            eq_pairs.push((remap(0, v_ab, &offsets), remap(0, v_sq, &offsets)));
        }
    }
    for (&gv, &lv) in base_local_map.iter() {
        eq_pairs.push((remap(0, gv, &offsets), remap(1, lv, &offsets)));
    }
    for (i, eo) in extra_out.iter().enumerate() {
        let part = 2 + i;
        for (&gv, &lv) in eo.local_map.iter() {
            eq_pairs.push((remap(0, gv, &offsets), remap(part, lv, &offsets)));
        }
        for &(base_var, local_var) in &eo.base_eqs {
            eq_pairs.push((remap(1, base_var, &offsets), remap(part, local_var, &offsets)));
        }
    }
    if eq_pairs.len() != plan.eq_pairs.len() {
        return Err(format!(
            "tiny gate: Pass0/Pass1 eq_pairs length mismatch (got {}, expected {})",
            eq_pairs.len(),
            plan.eq_pairs.len()
        ));
    }

    // Append eq tail into reserved ranges.
    let mut eq_writer = FileBackedRangeWriter::new(
        files.fc_a.try_clone().map_err(|e| format!("clone a_coeffs failed: {e}"))?,
        files.fi_a.try_clone().map_err(|e| format!("clone a_idx failed: {e}"))?,
        files.fc_b.try_clone().map_err(|e| format!("clone b_coeffs failed: {e}"))?,
        files.fi_b.try_clone().map_err(|e| format!("clone b_idx failed: {e}"))?,
        files.fc_c.try_clone().map_err(|e| format!("clone c_coeffs failed: {e}"))?,
        files.fi_c.try_clone().map_err(|e| format!("clone c_idx failed: {e}"))?,
        files.f_rows.try_clone().map_err(|e| format!("clone constraints failed: {e}"))?,
        plan.part_a_terms,
        plan.part_b_terms,
        plan.part_c_terms,
        plan.part_rows,
    );
    const EQ_BATCH: usize = 4096;
    let one_bytes: [u8; 2] = [0x01, 0x00];
    let neg_one_bytes: [u8; 2] = [0x00, 0x01]; // 256 mod 257
    let zero_bytes: [u8; 2] = [0x00, 0x00];
    let mut i0 = 0usize;
    while i0 < eq_pairs.len() {
        let i1 = (i0 + EQ_BATCH).min(eq_pairs.len());
        let batch = &eq_pairs[i0..i1];
        let mut a_coeff: Vec<u8> = Vec::with_capacity(batch.len() * 2 * 2);
        let mut a_idx: Vec<u32> = Vec::with_capacity(batch.len() * 2);
        let mut b_coeff: Vec<u8> = Vec::with_capacity(batch.len() * 2);
        let mut b_idx: Vec<u32> = Vec::with_capacity(batch.len());
        let mut c_coeff: Vec<u8> = Vec::with_capacity(batch.len() * 2);
        let mut c_idx: Vec<u32> = Vec::with_capacity(batch.len());
        let mut lens: Vec<u32> = Vec::with_capacity(batch.len() * 3);
        for &(x, y) in batch {
            // A: [+1*x, -1*y]
            a_coeff.extend_from_slice(&one_bytes);
            a_coeff.extend_from_slice(&neg_one_bytes);
            a_idx.push((x as u64).try_into().map_err(|_| "tiny gate: eq var idx overflow u32")?);
            a_idx.push((y as u64).try_into().map_err(|_| "tiny gate: eq var idx overflow u32")?);
            // B: [+1*var0]
            b_coeff.extend_from_slice(&one_bytes);
            b_idx.push(0u32);
            // C: [0*var0]
            c_coeff.extend_from_slice(&zero_bytes);
            c_idx.push(0u32);
            lens.extend_from_slice(&[2u32, 1u32, 1u32]);
        }
        eq_writer.push_a_terms_raw_block(&a_coeff, &a_idx)?;
        eq_writer.push_b_terms_raw_block(&b_coeff, &b_idx)?;
        eq_writer.push_c_terms_raw_block(&c_coeff, &c_idx)?;
        eq_writer.push_constraint_lens_block(&lens)?;
        i0 = i1;
    }
    if (eq_pairs.len() as u64) != plan.total_rows.saturating_sub(plan.part_rows) {
        return Err("tiny gate: eq tail row count mismatch vs plan".to_string());
    }

    // Merge checkpoints.
    let mut ckpts_all: Vec<(u64, u64, u64, u64)> = Vec::new();
    ckpts_all.extend_from_slice(&pose_range.ckpts);
    ckpts_all.extend_from_slice(&base_range.ckpts);
    for eo in &extra_out {
        ckpts_all.extend_from_slice(&eo.range.ckpts);
    }
    ckpts_all.extend(eq_writer.take_ckpts().into_iter());
    ckpts_all.sort_by_key(|(row_idx, _, _, _)| *row_idx);
    ckpts_all.dedup_by_key(|(row_idx, _, _, _)| *row_idx);
    if ckpts_all.first().map(|x| x.0) != Some(0) {
        ckpts_all.insert(0, (0, 0, 0, 0));
    }
    {
        let mut f = std::io::BufWriter::new(
            std::fs::File::create(dirs.merged_dir.join("rows_ckpt.bin"))
                .map_err(|e| format!("create rows_ckpt failed: {e}"))?,
        );
        for (row_idx, a0, b0, c0) in &ckpts_all {
            f.write_all(&row_idx.to_le_bytes()).map_err(|e| e.to_string())?;
            f.write_all(&a0.to_le_bytes()).map_err(|e| e.to_string())?;
            f.write_all(&b0.to_le_bytes()).map_err(|e| e.to_string())?;
            f.write_all(&c0.to_le_bytes()).map_err(|e| e.to_string())?;
        }
    }

    // meta.txt for the merged instance.
    {
        let limbs = F257::MODULUS.as_ref();
        let modulus = limbs.get(0).copied().unwrap_or(0);
        let mut f = std::io::BufWriter::new(
            std::fs::File::create(dirs.merged_dir.join("meta.txt"))
                .map_err(|e| format!("create meta failed: {e}"))?,
        );
        writeln!(f, "nvars={}", merged_asg.len()).ok();
        writeln!(f, "constraints={}", plan.total_rows).ok();
        writeln!(f, "a_terms={}", plan.total_a_terms).ok();
        writeln!(f, "b_terms={}", plan.total_b_terms).ok();
        writeln!(f, "c_terms={}", plan.total_c_terms).ok();
        writeln!(f, "coeff_size=2").ok();
        writeln!(f, "idx_size=4").ok();
        writeln!(f, "row_size=12").ok();
        writeln!(f, "row_ckpt_stride={}", 1u64 << 20).ok();
        writeln!(f, "format=tiny_u16_u32_rows_len_u32_v1").ok();
        writeln!(f, "modulus={}", modulus).ok();
    }

    // Strict size invariants: fail fast if final files don't match the planned layout.
    {
        let check = |name: &str, want: u64| -> Result<(), String> {
            let p = dirs.merged_dir.join(name);
            let got = std::fs::metadata(&p)
                .map_err(|e| format!("stat {p:?} failed: {e}"))?
                .len();
            if got != want {
                return Err(format!(
                    "tiny gate: {name} size mismatch: got {got} bytes, expected {want}"
                ));
            }
            Ok(())
        };
        check("constraints.bin", plan.total_rows.saturating_mul(12))?;
        check("a_coeffs.bin", plan.total_a_terms.saturating_mul(2))?;
        check("a_idx.bin", plan.total_a_terms.saturating_mul(4))?;
        check("b_coeffs.bin", plan.total_b_terms.saturating_mul(2))?;
        check("b_idx.bin", plan.total_b_terms.saturating_mul(4))?;
        check("c_coeffs.bin", plan.total_c_terms.saturating_mul(2))?;
        check("c_idx.bin", plan.total_c_terms.saturating_mul(4))?;
        check("rows_ckpt.bin", (ckpts_all.len() as u64).saturating_mul(32))?;
    }

    let inst = FileBackedSparseDr1csInstance::<F257>::new(
        merged_asg.len(),
        FileBackedLayout {
            dir: dirs.merged_dir.clone(),
            coeff_size: 2,
            idx_size: 4,
            row_size: 12,
            nconstraints: plan.total_rows,
            a_terms: plan.total_a_terms,
            b_terms: plan.total_b_terms,
            c_terms: plan.total_c_terms,
        },
    );
    lf_profile_log(&format!(
        "finish nvars={} constraints={}",
        inst.nvars,
        inst.layout.nconstraints
    ));

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

    let all_sum_digits_global = Arc::new(all_sum_digits.iter().map(|&v| to_glue_global(v)).collect::<Vec<_>>());
    let all_sq_sum_digits_global = Arc::new(all_sq_sum_digits.iter().map(|&v| to_glue_global(v)).collect::<Vec<_>>());
    let all_sum_coeffwise_global = Arc::new(
        all_sum_coeffwise
            .iter()
            .map(|row| row.iter().map(|&v| to_glue_global(v)).collect::<Vec<_>>())
            .collect::<Vec<_>>(),
    );
    let all_sq_sum_coeffwise_global = Arc::new(
        all_sq_sum_coeffwise
            .iter()
            .map(|row| row.iter().map(|&v| to_glue_global(v)).collect::<Vec<_>>())
            .collect::<Vec<_>>(),
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
        merged_asg,
        shorts_out,
        u32s_out,
        goldilocks_out,
        surfaces_out,
        surfaces_sq_out,
        pose_wiring,
    ))
}

/// Build Poseidon(F257) + WE-tiny glue as a file-backed dR1CS instance.
pub(super) fn build(
    cfg: Option<&PoseidonConfig<F257>>,
    ops: &[PoseidonTraceOp<F257>],
    ring_dim: usize,
    params: &WeParams,
    wiring: &TinyCoinOpWiring,
    pairs: &[(usize, usize)],
    extra_witness: &TinyExtraWitness,
    out_dir: impl AsRef<std::path::Path>,
) -> Result<
    (
        FileBackedSparseDr1csInstance<F257>,
        Vec<F257>,
        Vec<ShortChallengeWiring>,
        Vec<BoundedU32ChallengeWiring>,
        Vec<GoldilocksChallengeWiring>,
        Vec<CmDigitMulSurfaceWiring>,
        Vec<CmDigitMulSqSurfaceWiring>,
        PoseidonDr1csWiring,
    ),
    String,
> {
    #[cfg(unix)]
    {
        return build_direct_to_merged_unix(cfg, ops, ring_dim, params, wiring, pairs, extra_witness, out_dir);
    }
    #[cfg(not(unix))]
    {
        let _ = (cfg, ops, ring_dim, params, wiring, pairs, extra_witness, out_dir);
        Err("tiny gate: direct-to-merged build requires unix".to_string())
    }
}