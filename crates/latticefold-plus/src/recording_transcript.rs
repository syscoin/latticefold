use ark_crypto_primitives::sponge::{
    poseidon::{PoseidonConfig, PoseidonSponge},
    CryptographicSponge,
};
use ark_ff::{BigInteger, PrimeField};
use ark_std::marker::PhantomData;
use latticefold::transcript::Transcript;
use latticefold::transcript::bytes::{prime_field_to_bytes_le_fixed, ring_to_bytes_le_fixed};
use latticefold::transcript::poseidon::{f257_poseidon_config, F257};
use stark_rings::OverField;

use crate::transcript::PoseidonTranscriptMetrics;

/// Poseidon sponge transcript operation trace (in the sponge's base prime field).
#[derive(Clone, Debug)]
pub enum PoseidonTraceOp<BF: PrimeField> {
    Absorb(Vec<BF>),
    SqueezeField(Vec<BF>),
    SqueezeBytes { n: usize, out: Vec<u8> },
}

/// Full transcript trace in terms of the Poseidon sponge's base prime field.
#[derive(Clone, Debug, Default)]
pub struct PoseidonTranscriptTrace<BF: PrimeField> {
    pub ops: Vec<PoseidonTraceOp<BF>>,
    pub absorbed: Vec<BF>,
    pub squeezed_field: Vec<BF>,
    pub squeezed_bytes: Vec<u8>,
}

impl<BF: PrimeField> PoseidonTranscriptTrace<BF> {
    /// Reconstruct `get_challenge()` scalars from the recorded trace.
    ///
    /// The trace stores Poseidon sponge squeeze outputs as **F257 digits** (each digit is a BF
    /// element in the range \(0..=256\)). Each `get_challenge()` consumes `digits_per_challenge`
    /// digits (currently 8 in LF/LF+), interprets them in **byte view** (256 -> 0), and returns
    /// the u32 formed by the first 4 bytes (little-endian).
    ///
    /// This helper extracts exactly the `SqueezeField` ops whose vector length equals
    /// `digits_per_challenge`, combines them, and returns the first `n` reconstructed challenges.
    pub fn challenge_scalars_base257(&self, digits_per_challenge: usize, n: usize) -> Vec<BF> {
        let mut out = Vec::with_capacity(n);
        for op in &self.ops {
            if let PoseidonTraceOp::SqueezeField(digits) = op {
                if digits.len() != digits_per_challenge {
                    continue;
                }
                // Pack the first 4 digits into a u32 in byte view (256 -> 0).
                let mut bs = [0u8; 4];
                for i in 0..4 {
                    let du16 = digits[i]
                        .into_bigint()
                        .to_bytes_le()
                        .get(0)
                        .copied()
                        .unwrap_or(0) as u16;
                    debug_assert!(du16 < 257u16);
                    bs[i] = if du16 == 256 { 0u8 } else { du16 as u8 };
                }
                let x = u32::from_le_bytes(bs);
                out.push(BF::from(x as u64));
                if out.len() == n {
                    break;
                }
            }
        }
        out
    }

    /// Convenience: reconstruct **all** `get_challenge()` scalars present in the trace.
    pub fn challenge_scalars_base257_all(&self, digits_per_challenge: usize) -> Vec<BF> {
        let n = self
            .ops
            .iter()
            .filter(|op| matches!(op, PoseidonTraceOp::SqueezeField(v) if v.len() == digits_per_challenge))
            .count();
        self.challenge_scalars_base257(digits_per_challenge, n)
    }
}

/// Poseidon transcript that records a full operation trace.
///
/// This is intended for WE/DPP arithmetization frontends: the prover can record a trace and later
/// provide it as part of a witness, with constraints enforcing that it matches the Poseidon
/// permutation schedule.
#[derive(Clone)]
pub struct TracePoseidonTranscript<R: OverField>
where
    R::BaseRing: PrimeField,
{
    sponge: PoseidonSponge<F257>,
    metrics: PoseidonTranscriptMetrics,
    scratch: Vec<F257>,
    trace: PoseidonTranscriptTrace<R::BaseRing>,
    _marker: PhantomData<R>,
}

impl<R: OverField> TracePoseidonTranscript<R>
where
    R::BaseRing: PrimeField,
{
    pub fn empty<P>() -> Self {
        Self::new(&f257_poseidon_config())
    }

    pub fn metrics(&self) -> PoseidonTranscriptMetrics {
        self.metrics
    }

    pub fn trace(&self) -> &PoseidonTranscriptTrace<R::BaseRing> {
        &self.trace
    }

    #[inline]
    fn lift_f257_to_base_ring(x: &F257) -> R::BaseRing {
        // F257 elements are always in 0..=256; lift that integer into the transcript's base ring.
        let d = x.into_bigint().to_bytes_le().get(0).copied().unwrap_or(0) as u64;
        debug_assert!(d < 257u64);
        R::BaseRing::from(d)
    }

    fn absorb_f257_elems_vec(
        &mut self,
        elems: Vec<F257>,
    ) {
        self.metrics.absorbed_elems += elems.len() as u64;
        self.sponge.absorb(&elems);
        let lifted = elems
            .iter()
            .map(|e| Self::lift_f257_to_base_ring(e))
            .collect::<Vec<_>>();
        self.trace.absorbed.extend_from_slice(&lifted);
        self.trace.ops.push(PoseidonTraceOp::Absorb(lifted));
    }
}

impl<R: OverField> Transcript<R> for TracePoseidonTranscript<R>
where
    R::BaseRing: PrimeField,
{
    type TranscriptConfig = PoseidonConfig<F257>;

    fn new(config: &Self::TranscriptConfig) -> Self {
        let sponge = PoseidonSponge::<F257>::new(config);
        Self {
            sponge,
            metrics: PoseidonTranscriptMetrics::default(),
            scratch: Vec::with_capacity(64),
            trace: PoseidonTranscriptTrace::default(),
            _marker: PhantomData,
        }
    }

    fn absorb(&mut self, v: &R) {
        self.scratch.clear();
        let bytes = ring_to_bytes_le_fixed::<R>(v);
        self.scratch.extend(bytes.iter().map(|b| F257::from(*b as u64)));
        let elems = self.scratch.clone();
        self.absorb_f257_elems_vec(elems);
    }

    fn absorb_field_element(&mut self, v: &R::BaseRing) {
        let bytes = prime_field_to_bytes_le_fixed::<R::BaseRing>(v);
        self.absorb_f257_elems_vec(bytes.iter().map(|b| F257::from(*b as u64)).collect());
    }

    fn get_challenge(&mut self) -> R::BaseRing {
        // Fixed-length digit schedule (no rejection) to keep a fixed schedule.
        // See `crate::transcript::PoseidonTranscript::get_challenge` for semantics:
        // pack the first 4 digits (byte view, 256 -> 0) into a u32.
        const CHALLENGE_DIGITS: usize = 8;
        let c = self.sponge.squeeze_field_elements::<F257>(CHALLENGE_DIGITS);
        self.metrics.squeezed_field_elems += c.len() as u64;
        let lifted = c
            .iter()
            .map(|e| Self::lift_f257_to_base_ring(e))
            .collect::<Vec<_>>();
        self.trace.squeezed_field.extend_from_slice(&lifted);
        self.trace.ops.push(PoseidonTraceOp::SqueezeField(lifted));

        // `get_challenge` re-absorbs the squeezed elements to evolve the sponge state.
        self.absorb_f257_elems_vec(c.clone());

        let mut bs = [0u8; 4];
        for i in 0..4 {
            let d = c[i]
                .into_bigint()
                .to_bytes_le()
                .get(0)
                .copied()
                .unwrap_or(0) as u16;
            debug_assert!(d < 257u16);
            bs[i] = if d == 256 { 0u8 } else { d as u8 };
        }
        let x = u32::from_le_bytes(bs);
        R::BaseRing::from(x as u64)
    }

    fn squeeze_bytes(&mut self, n: usize) -> Vec<u8> {
        self.metrics.squeezed_bytes += n as u64;
        let elems = self.sponge.squeeze_field_elements::<F257>(n);
        self.metrics.squeezed_field_elems += elems.len() as u64;
        let lifted = elems
            .iter()
            .map(|e| Self::lift_f257_to_base_ring(e))
            .collect::<Vec<_>>();
        self.trace.squeezed_field.extend_from_slice(&lifted);
        self.trace.ops.push(PoseidonTraceOp::SqueezeField(lifted));
        let out = elems
            .iter()
            .map(|e| {
                let d = e.into_bigint().to_bytes_le().get(0).copied().unwrap_or(0) as u16;
                debug_assert!(d < 257u16);
                if d == 256 { 0u8 } else { d as u8 }
            })
            .collect::<Vec<u8>>();
        self.trace.squeezed_bytes.extend_from_slice(&out);
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transcript::PoseidonTranscript as PlainPoseidonTranscript;
    use stark_rings::cyclotomic_ring::models::goldilocks::RqPoly as R;
    use stark_rings::PolyRing;
    use stark_rings::Ring;

    type BR = <R as PolyRing>::BaseRing;

    #[test]
    fn test_trace_transcript_matches_plain_transcript() {
        let mut plain = PlainPoseidonTranscript::<R>::empty::<()>();
        let mut trace = TracePoseidonTranscript::<R>::empty::<()>();

        // Mix ring and base-ring absorption.
        plain.absorb(&R::ONE);
        trace.absorb(&R::ONE);
        plain.absorb_field_element(&BR::from(0xBEEFu64));
        trace.absorb_field_element(&BR::from(0xBEEFu64));

        // Challenges must match exactly.
        for _ in 0..5 {
            assert_eq!(plain.get_challenge(), trace.get_challenge());
        }

        // Squeezed bytes must match exactly.
        let out_plain = plain.squeeze_bytes(64);
        let out_trace = trace.squeeze_bytes(64);
        assert_eq!(out_plain, out_trace);

        // Metrics should match, and trace buffers should be consistent with metrics.
        let m_plain = plain.metrics();
        let m = trace.metrics();
        assert_eq!(m_plain.absorbed_elems, m.absorbed_elems);
        assert_eq!(m_plain.squeezed_field_elems, m.squeezed_field_elems);
        assert_eq!(m_plain.squeezed_bytes, m.squeezed_bytes);

        let tr = trace.trace();
        assert_eq!(tr.absorbed.len() as u64, m.absorbed_elems);
        assert_eq!(tr.squeezed_field.len() as u64, m.squeezed_field_elems);
        assert_eq!(tr.squeezed_bytes.len() as u64, m.squeezed_bytes);
    }
}

