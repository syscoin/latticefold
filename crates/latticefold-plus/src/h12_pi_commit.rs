//! Packed `pi0` opening helpers used by H12 GERM.
//!
//! The live path uses Ajtai commitments, so this module only keeps:
//! - opened block payload shape (`H12PiBlockOpening`)
//! - packed block length helper (`block_len_for_index`)
//! - `F257 <-> u16` conversions

use ark_ff::PrimeField;
use latticefold::transcript::poseidon::F257;

#[derive(Clone, Debug)]
pub struct H12PiBlockOpening {
    pub block_index: u32,
    pub values: Vec<u16>,
}

pub fn block_len_for_index(pi0_len: usize, pack_d: usize, block_index: usize) -> usize {
    if pack_d == 0 {
        return 0;
    }
    let start = block_index.saturating_mul(pack_d);
    if start >= pi0_len {
        return 0;
    }
    (pi0_len - start).min(pack_d)
}

#[inline]
pub fn f257_to_u16(f: F257) -> u16 {
    (f.into_bigint().as_ref()[0] % 257) as u16
}

#[inline]
pub fn u16_to_f257(x: u16) -> F257 {
    F257::from((x % 257) as u64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_block_len_for_index() {
        assert_eq!(block_len_for_index(173, 64, 0), 64);
        assert_eq!(block_len_for_index(173, 64, 1), 64);
        assert_eq!(block_len_for_index(173, 64, 2), 45);
        assert_eq!(block_len_for_index(173, 64, 3), 0);
    }
}