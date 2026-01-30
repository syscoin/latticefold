use super::params::{DIGITS_PER_TRY, LIMB_BASE_U64, LIMB_BITS, LIMBS_U64};
use crate::we_goldilocks_poseidon_f257::GOLDILOCKS_P;

pub(super) fn goldilocks_p_base128_digits_le() -> [u8; LIMBS_U64] {
    let mut out = [0u8; LIMBS_U64];
    let mut t = GOLDILOCKS_P;
    for i in 0..LIMBS_U64 {
        out[i] = (t & (LIMB_BASE_U64 - 1)) as u8;
        t >>= LIMB_BITS;
    }
    out
}

