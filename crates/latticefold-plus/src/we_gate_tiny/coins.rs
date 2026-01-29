use symphony::dpp_sumcheck::Dr1csBuilder;

use latticefold::transcript::poseidon::F257;

use super::params::{DIGITS_PER_TRY, LIMB_BASE_U64, LIMB_BITS, LIMBS_U64};
use crate::we_goldilocks_poseidon_f257::GOLDILOCKS_P;
use super::cm_ir::{
    lower_ir_into_builder, sample_goldilocks_coin_unrolled_rejection_8_digits_ir, IrBuilder as CmIrBuilder,
    VarRef as CmVarRef,
};

pub(super) fn goldilocks_p_base128_digits_le() -> [u8; LIMBS_U64] {
    let mut out = [0u8; LIMBS_U64];
    let mut t = GOLDILOCKS_P;
    for i in 0..LIMBS_U64 {
        out[i] = (t & (LIMB_BASE_U64 - 1)) as u8;
        t >>= LIMB_BITS;
    }
    out
}

pub(crate) fn sample_goldilocks_coin_unrolled_rejection_8_digits(
    b: &mut Dr1csBuilder<F257>,
    digit_vars: &[usize], // length = tries*8
    tries: usize,
) -> ([usize; LIMBS_U64], usize /* found */) {
    assert_eq!(digit_vars.len(), tries * DIGITS_PER_TRY);
    let p_digits = goldilocks_p_base128_digits_le();
    let (ir, out_ir, found_ir) = {
        let base_asg: &[F257] = &b.assignment;
        let mut ib = CmIrBuilder::new(base_asg);
        let digits_ir: Vec<CmVarRef> = digit_vars.iter().copied().map(CmVarRef::Base).collect();
        let (out_ir, found_ir) =
            sample_goldilocks_coin_unrolled_rejection_8_digits_ir(&mut ib, &digits_ir, tries, &p_digits);
        (ib.ir, out_ir, found_ir)
    };
    let lowered = lower_ir_into_builder(b, ir);
    let out: [usize; LIMBS_U64] = core::array::from_fn(|i| lowered.map_var(out_ir[i]));
    let found = lowered.map_var(found_ir);
    (out, found)
}

#[derive(Clone, Debug)]
pub struct GoldilocksRejectionCoinWiring {
    /// Global variable indices of all base-257 digit vars used (tries * 8).
    pub digit_vars: Vec<usize>,
    /// Global variable index of `found` (boolean), enforced to be 1 in the builder.
    pub found_bit: usize,
    /// Global variable indices of the selected coin value `u` as base-128 limbs (little-endian).
    pub coin_limbs: Vec<usize>,
    /// The fixed number of tries assumed by this wiring.
    pub tries: usize,
}

