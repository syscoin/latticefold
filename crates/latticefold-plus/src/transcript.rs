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

/// Lightweight transcript metrics to estimate Poseidon sponge work.
///
/// Counts are in units of the Poseidon sponge's **F257 elements** (one per byte).
#[derive(Clone, Copy, Debug, Default)]
pub struct PoseidonTranscriptMetrics {
    /// Number of F257 elements absorbed into the sponge (== bytes absorbed).
    pub absorbed_elems: u64,
    /// Number of F257 elements squeezed during `get_challenge`.
    pub squeezed_field_elems: u64,
    /// Number of bytes squeezed via `squeeze_bytes`.
    pub squeezed_bytes: u64,
}

impl PoseidonTranscriptMetrics {
    /// Estimate number of Poseidon permutations based on sponge rate.
    /// 
    /// For a rate-r sponge:
    /// - Each permutation absorbs r field elements
    /// - Each permutation produces r field elements (squeeze)
    /// 
    /// This is a lower bound; actual count depends on interleaving.
    /// 
    /// `bytes_per_field_elem`: For Goldilocks (~64-bit), use 8. For larger fields, adjust.
    pub fn estimated_permutations(&self, rate: usize, bytes_per_field_elem: usize) -> u64 {
        let absorb_perms = (self.absorbed_elems + rate as u64 - 1) / rate as u64;
        let squeeze_perms = (self.squeezed_field_elems + rate as u64 - 1) / rate as u64;
        // Convert squeezed bytes to field elements
        let byte_field_elems = (self.squeezed_bytes + bytes_per_field_elem as u64 - 1) 
            / bytes_per_field_elem as u64;
        let byte_perms = (byte_field_elems + rate as u64 - 1) / rate as u64;
        absorb_perms + squeeze_perms + byte_perms
    }
}

/// Transcript using Poseidon sponge with metrics tracking.
///
/// Same implementation as LatticeFold's `PoseidonTranscript` though without the challenge set
/// generic / requirement on `SuitableRing`.
#[derive(Clone)]
pub struct PoseidonTranscript<R: OverField> {
    sponge: PoseidonSponge<F257>,
    metrics: PoseidonTranscriptMetrics,
    _marker: PhantomData<R>,
}

impl<R: OverField> PoseidonTranscript<R>
where
    R::BaseRing: PrimeField,
{
    pub fn empty<P>() -> Self {
        Self::new(&f257_poseidon_config())
    }

    /// Get current transcript metrics.
    pub fn metrics(&self) -> PoseidonTranscriptMetrics {
        self.metrics
    }

    /// Print a summary of transcript work.
    /// 
    /// Uses the F257 transcript sponge parameters (rate=8).
    pub fn print_metrics(&self) {
        let m = &self.metrics;
        println!("=== LF+ Transcript Metrics ===");
        println!("  Absorbed F257 elems:       {}", m.absorbed_elems);
        println!("  Squeezed field elems:      {}", m.squeezed_field_elems);
        println!("  Squeezed bytes:            {}", m.squeezed_bytes);
        // F257: one byte per field element.
        let perms = m.estimated_permutations(8, 1);
        println!("  Est. Poseidon permutations (rate=8): {}", perms);
        println!("==============================");
    }
}

impl<R: OverField> Transcript<R> for PoseidonTranscript<R>
where
    R::BaseRing: PrimeField,
{
    type TranscriptConfig = PoseidonConfig<F257>;

    fn new(config: &Self::TranscriptConfig) -> Self {
        let sponge = PoseidonSponge::<F257>::new(config);
        Self { 
            sponge,
            metrics: PoseidonTranscriptMetrics::default(),
            _marker: PhantomData,
        }
    }

    fn absorb(&mut self, v: &R) {
        let bytes = ring_to_bytes_le_fixed::<R>(v);
        self.metrics.absorbed_elems += bytes.len() as u64;
        self.sponge
            .absorb(&bytes.iter().map(|b| F257::from(*b as u64)).collect::<Vec<_>>());
    }

    fn absorb_field_element(&mut self, v: &R::BaseRing) {
        let bytes = prime_field_to_bytes_le_fixed::<R::BaseRing>(v);
        self.metrics.absorbed_elems += bytes.len() as u64;
        self.sponge
            .absorb(&bytes.iter().map(|b| F257::from(*b as u64)).collect::<Vec<_>>());
    }

    fn get_challenge(&mut self) -> R::BaseRing {
        // Fixed-length base-257 digits (no rejection) to keep a fixed schedule.
        const CHALLENGE_DIGITS: usize = 12;
        let elems = self.sponge.squeeze_field_elements::<F257>(CHALLENGE_DIGITS);
        self.metrics.squeezed_field_elems += elems.len() as u64;
        // Re-absorb squeezed elements (Fiat-Shamir)
        self.metrics.absorbed_elems += elems.len() as u64;
        self.sponge.absorb(&elems);

        let mut acc = R::BaseRing::from(0u64);
        let mut pow = R::BaseRing::from(1u64);
        let base = R::BaseRing::from(257u64);
        for e in &elems {
            let d = e.into_bigint().to_bytes_le().get(0).copied().unwrap_or(0) as u64;
            debug_assert!(d < 257u64);
            acc += R::BaseRing::from(d) * pow;
            pow *= base;
        }
        acc
    }

    fn squeeze_bytes(&mut self, n: usize) -> Vec<u8> {
        let elems = self.sponge.squeeze_field_elements::<F257>(n);
        self.metrics.squeezed_field_elems += elems.len() as u64;
        self.metrics.squeezed_bytes += n as u64;
        elems
            .iter()
            .map(|e| {
                let d = e.into_bigint().to_bytes_le().get(0).copied().unwrap_or(0) as u16;
                debug_assert!(d < 257u16);
                if d == 256 { 0u8 } else { d as u8 }
            })
            .collect()
    }
}

pub fn squeeze_challenges<R: OverField>(
    transcript: &mut impl Transcript<R>,
    name: &str,
    n: usize,
) -> Vec<R::BaseRing>
where
    R::BaseRing: PrimeField,
{
    let dom = <R::BaseRing as PrimeField>::from_be_bytes_mod_order(name.as_bytes());
    transcript.absorb_field_element(&dom);

    transcript.get_challenges(n)
}

pub fn squeeze_rchallenges<R: OverField>(
    transcript: &mut impl Transcript<R>,
    name: &str,
    n: usize,
) -> Vec<R>
where
    R::BaseRing: PrimeField,
{
    squeeze_challenges(transcript, name, n)
        .into_iter()
        .map(|z| R::from(z))
        .collect::<Vec<R>>()
}
