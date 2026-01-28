use ark_crypto_primitives::sponge::poseidon::PoseidonConfig;

use latticefold::transcript::poseidon::F257;
use symphony::dpp_poseidon::{PoseidonDr1csWiring, SparseDr1csInstance};
use symphony::transcript::PoseidonTraceOp;

use crate::we_statement::WeParams;

use super::builder;

pub use super::coins::GoldilocksRejectionCoinWiring;
pub use super::challenges::{infer_cm_coin_op_wiring_from_ops, BoundedU32ChallengeWiring, GoldilocksChallengeWiring, ShortChallengeWiring, TinyCoinOpWiring};
pub use super::lift::lift_recording_trace_ops_to_f257;
pub use super::poseidon::poseidon_f257_arithmetize;
pub use super::surfaces::{CmDigitMulSqSurfaceWiring, CmDigitMulSurfaceWiring};

/// Build Poseidon(F257) + CM coin surface + digit-mul surfaces (and optional rejection-sampled Goldilocks coins).
///
/// Each pair `(short_block_idx, u32_idx)` requests multiplying all coeffs in that short block by
/// the bounded-u32 challenge at `u32_idx` **and** (in parallel) by its square `u32^2`.
pub fn we_tiny_f257_build_cm_gate_from_trace_ops(
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
        Vec<GoldilocksChallengeWiring>,
        Vec<GoldilocksRejectionCoinWiring>,
        Vec<[usize; 8]>, // tcch0 (Goldilocks base-field, 8 LE bytes) per instance
        Vec<[usize; 8]>, // tcch1 (Goldilocks base-field, 8 LE bytes) per instance
        Vec<CmDigitMulSurfaceWiring>,
        Vec<CmDigitMulSqSurfaceWiring>,
        PoseidonDr1csWiring,
    ),
    String,
> {
    builder::build(cfg, ops, ring_dim, params, wiring, pairs)
}

