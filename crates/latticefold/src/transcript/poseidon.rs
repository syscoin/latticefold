use ark_crypto_primitives::sponge::{
    poseidon::{PoseidonConfig, PoseidonSponge},
    CryptographicSponge,
};
use ark_ff::{BigInteger, Field, PrimeField};
use ark_std::marker::PhantomData;
use cyclotomic_rings::{challenge_set::LatticefoldChallengeSet, rings::SuitableRing};
use stark_rings::OverField;

use super::{Transcript, TranscriptWithShortChallenges};
use crate::ark_base::*;
use crate::transcript::bytes::{field_to_bytes_le_fixed, ring_to_bytes_le_fixed};

#[path = "f257_poseidon_t64_r32_c32.rs"]
mod f257_poseidon_t64_r32_c32;
pub use f257_poseidon_t64_r32_c32::F257;

pub fn f257_poseidon_config() -> PoseidonConfig<F257> {
    f257_poseidon_t64_r32_c32::poseidon_config_f257()
}

// Fixed-length challenge derivation from F257 digits to avoid rejection sampling.
const CHALLENGE_DIGITS: usize = 12;

/// PoseidonTranscript implements the Transcript trait using the Poseidon hash
#[derive(Clone)]
pub struct PoseidonTranscript<R: OverField, CS> {
    _marker: PhantomData<CS>,
    _marker_r: PhantomData<R>,
    sponge: PoseidonSponge<F257>,
}

impl<R: SuitableRing, CS: LatticefoldChallengeSet<R>> Default for PoseidonTranscript<R, CS> {
    fn default() -> Self {
        Self::new(&f257_poseidon_config())
    }
}

impl<R: OverField, CS> Transcript<R> for PoseidonTranscript<R, CS>
where
    R::BaseRing: Field,
{
    type TranscriptConfig = PoseidonConfig<F257>;

    fn new(config: &Self::TranscriptConfig) -> Self {
        let sponge = PoseidonSponge::<F257>::new(config);
        Self {
            sponge,
            _marker: PhantomData,
            _marker_r: PhantomData,
        }
    }

    fn absorb(&mut self, v: &R) {
        let bytes = ring_to_bytes_le_fixed::<R>(v);
        self.sponge
            .absorb(&bytes.iter().map(|b| F257::from(*b as u64)).collect::<Vec<_>>());
    }

    fn absorb_field_element(&mut self, v: &R::BaseRing) {
        let bytes = field_to_bytes_le_fixed::<R::BaseRing>(v);
        self.sponge
            .absorb(&bytes.iter().map(|b| F257::from(*b as u64)).collect::<Vec<_>>());
    }

    fn get_challenge(&mut self) -> R::BaseRing {
        // Derive a challenge for the *outer* base ring using a fixed number of
        // base-257 digits (no rejection). This keeps a fixed transcript schedule.
        let elems = self.sponge.squeeze_field_elements::<F257>(CHALLENGE_DIGITS);
        self.sponge.absorb(&elems);

        let mut acc = R::BaseRing::from(0u64);
        let mut pow = R::BaseRing::from(1u64);
        let base = R::BaseRing::from(257u64);
        for e in &elems {
            let d_u64: u64 = e
                .into_bigint()
                .to_bytes_le()
                .get(0)
                .copied()
                .unwrap_or(0) as u64;
            debug_assert!(d_u64 < 257u64);
            acc += R::BaseRing::from(d_u64) * pow;
            pow *= base;
        }
        acc
    }

    fn squeeze_bytes(&mut self, n: usize) -> Vec<u8> {
        self.sponge.squeeze_bytes(n)
    }
}

impl<R: SuitableRing, CS: LatticefoldChallengeSet<R>> TranscriptWithShortChallenges<R>
    for PoseidonTranscript<R, CS>
where
    R::BaseRing: Field,
{
    type ChallengeSet = CS;

    fn get_short_challenge(&mut self) -> R::CoefficientRepresentation {
        // Deterministic byte view of F257 digits: 256 maps to 0.
        let elems = self
            .sponge
            .squeeze_field_elements::<F257>(Self::ChallengeSet::BYTES_NEEDED);
        let random_bytes = elems
            .iter()
            .map(|e| {
                let d_u16: u16 = e
                    .into_bigint()
                    .to_bytes_le()
                    .get(0)
                    .copied()
                    .unwrap_or(0) as u16;
                debug_assert!(d_u16 < 257u16);
                if d_u16 == 256 { 0u8 } else { d_u16 as u8 }
            })
            .collect::<Vec<u8>>();

        Self::ChallengeSet::short_challenge_from_random_bytes(&random_bytes)
            .expect("not enough bytes to get a small challenge")
    }
}

#[cfg(test)]
mod tests {
    use cyclotomic_rings::rings::{GoldilocksChallengeSet, GoldilocksRingNTT};
    use stark_rings::PolyRing;
    use stark_rings::Ring;

    use super::*;

    #[test]
    fn test_transcript_determinism_big_challenge() {
        let mut t1 = PoseidonTranscript::<GoldilocksRingNTT, GoldilocksChallengeSet>::default();
        let mut t2 = PoseidonTranscript::<GoldilocksRingNTT, GoldilocksChallengeSet>::default();

        type BR = <GoldilocksRingNTT as PolyRing>::BaseRing;
        t1.absorb_field_element(&BR::from(0xFFu64));
        t2.absorb_field_element(&BR::from(0xFFu64));

        assert_eq!(t1.get_challenge(), t2.get_challenge());
    }

    #[test]
    fn test_transcript_determinism_short_challenge() {
        let mut t1 = PoseidonTranscript::<GoldilocksRingNTT, GoldilocksChallengeSet>::default();
        let mut t2 = PoseidonTranscript::<GoldilocksRingNTT, GoldilocksChallengeSet>::default();

        // Absorb any fixed ring element to ensure sponge state changes deterministically.
        t1.absorb(&GoldilocksRingNTT::ONE);
        t2.absorb(&GoldilocksRingNTT::ONE);

        assert_eq!(t1.get_short_challenge(), t2.get_short_challenge());
    }
}
