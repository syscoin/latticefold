//! H12 `R_cap` helpers.
//!
//! This module is intentionally lightweight:
//! - it does **not** implement AADP itself
//! - it estimates the outer capsule witness/gate budget from exported local-check surfaces
//! - it provides the canonical "current logical lock -> capsule checks" mapping

use std::collections::BTreeSet;

use ark_ff::PrimeField;
use dpp::theorem43::Theorem43CapsuleLocalCheckSurface;

use crate::lockable_ringlwe::RingLweLockArtifact;

/// Current packed `pi0` block size used by `lockable_ringlwe`.
pub const H12_RCAP_PACK_D: usize = 64;

/// AADP cost model from `eprint 2026/175`, Section 4.10.
///
/// Rough model:
///
/// `size ~= v * (2g + 1)^2 * |F|`
///
/// where:
/// - `v` is the outer relation variable count
/// - `g` is the outer relation gate count
/// - `|F|` is the byte width of one field element in the target AADP field
pub fn aadp_ciphertext_size_bytes(v: usize, g: usize, field_bytes: usize) -> u128 {
    let v = v as u128;
    let g_term = (2usize.saturating_mul(g).saturating_add(1)) as u128;
    let field_bytes = field_bytes as u128;
    v.saturating_mul(g_term.saturating_mul(g_term))
        .saturating_mul(field_bytes)
}

/// Current "one logical lock -> capsule checks" mapping.
///
/// Each rep already exposes one `(anchor_block_id, rep_id)` pair, which is the natural
/// statement-fixed unit for a candidate H12 capsule local check.
pub fn capsule_checks_from_logical_lock<F: PrimeField>(
    reps: &[RingLweLockArtifact<F>],
) -> Vec<(usize, u64)> {
    reps.iter()
        .map(|rep| (rep.anchor_block_id as usize, rep.rep_id))
        .collect()
}

#[derive(Clone, Debug, Default)]
pub struct H12RCapScheduleEstimate {
    pub check_count: usize,
    pub g_cap: usize,
    pub touched_pi_positions_exact: usize,
    pub touched_pi_positions_conservative: usize,
    pub touched_packed_pi_blocks_exact: usize,
    pub touched_packed_pi_blocks_conservative: usize,
    pub aadp_size_exact_bytes: u128,
    pub aadp_size_conservative_bytes: u128,
}

/// Estimate the H12 outer capsule size from exported local-check surfaces.
///
/// Two witness regimes are reported:
///
/// - exact:
///   uses the exported per-position `q1/q2/q3` witness footprint
/// - conservative:
///   includes the whole selected `w_eval` block whenever the surface indicates the current backend
///   should be treated conservatively
pub fn estimate_capsule_schedule<F: PrimeField>(
    surfaces: &[Theorem43CapsuleLocalCheckSurface<F>],
    pack_d: usize,
    field_bytes: usize,
) -> H12RCapScheduleEstimate {
    let mut exact_pos = BTreeSet::<usize>::new();
    let mut conservative_pos = BTreeSet::<usize>::new();
    let mut exact_blocks = BTreeSet::<usize>::new();
    let mut conservative_blocks = BTreeSet::<usize>::new();

    for s in surfaces {
        for pi_idx in s.touched_pi_positions(false) {
            exact_pos.insert(pi_idx);
            exact_blocks.insert(pi_idx / pack_d.max(1));
        }
        let conservative_full_block = s.requires_dense_q3_w_eval;
        for pi_idx in s.touched_pi_positions(conservative_full_block) {
            conservative_pos.insert(pi_idx);
            conservative_blocks.insert(pi_idx / pack_d.max(1));
        }
    }

    let g_cap = surfaces.len();
    H12RCapScheduleEstimate {
        check_count: surfaces.len(),
        g_cap,
        touched_pi_positions_exact: exact_pos.len(),
        touched_pi_positions_conservative: conservative_pos.len(),
        touched_packed_pi_blocks_exact: exact_blocks.len(),
        touched_packed_pi_blocks_conservative: conservative_blocks.len(),
        aadp_size_exact_bytes: aadp_ciphertext_size_bytes(exact_pos.len(), g_cap, field_bytes),
        aadp_size_conservative_bytes: aadp_ciphertext_size_bytes(
            conservative_pos.len(),
            g_cap,
            field_bytes,
        ),
    }
}

