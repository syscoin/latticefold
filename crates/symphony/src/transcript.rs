use ark_crypto_primitives::sponge::{
    poseidon::{PoseidonConfig, PoseidonSponge},
    CryptographicSponge,
};
use ark_ff::{Field, PrimeField};
use cyclotomic_rings::rings::GetPoseidonParams;
use latticefold::transcript::Transcript;
use stark_rings::OverField;


/// Transcript using Poseidon sponge
///
/// Same implementation as LatticeFold's `PoseidonTranscript` though without the challenge set
/// generic / requirement on `SuitableRing`.
#[derive(Clone)]
pub struct PoseidonTranscript<R: OverField> {
    sponge: PoseidonSponge<<R::BaseRing as Field>::BasePrimeField>,
    metrics: PoseidonTranscriptMetrics,
    scratch: Vec<<R::BaseRing as Field>::BasePrimeField>,
}

/// Lightweight transcript metrics to estimate Poseidon sponge work in `R_cp` / `R_WE`.
///
/// Counts are in units of the Poseidon sponge's **base prime field** elements.
#[derive(Clone, Copy, Debug, Default)]
pub struct PoseidonTranscriptMetrics {
    /// Number of base-prime-field elements absorbed into the sponge.
    pub absorbed_elems: u64,
    /// Number of base-prime-field elements squeezed as challenges (`get_challenge`).
    pub squeezed_field_elems: u64,
    /// Number of bytes squeezed via `squeeze_bytes`.
    pub squeezed_bytes: u64,
}

impl<R: OverField> PoseidonTranscript<R> {
    pub fn empty<P: GetPoseidonParams<<<R>::BaseRing as Field>::BasePrimeField>>() -> Self {
        Self::new(&P::get_poseidon_config())
    }

    pub fn metrics(&self) -> PoseidonTranscriptMetrics {
        self.metrics
    }
}

impl<R: OverField> Transcript<R> for PoseidonTranscript<R> {
    type TranscriptConfig = PoseidonConfig<<R::BaseRing as Field>::BasePrimeField>;

    fn new(config: &Self::TranscriptConfig) -> Self {
        let sponge = PoseidonSponge::<<R::BaseRing as Field>::BasePrimeField>::new(config);
        Self {
            sponge,
            metrics: PoseidonTranscriptMetrics::default(),
            // Small default; will grow as needed (e.g. absorbing a full ring element).
            scratch: Vec::with_capacity(64),
        }
    }

    fn absorb(&mut self, v: &R) {
        self.scratch.clear();
        for c in v.coeffs() {
            self.scratch.extend(c.to_base_prime_field_elements());
        }
        self.metrics.absorbed_elems += self.scratch.len() as u64;
        self.sponge.absorb(&self.scratch);
    }

    /// Optimized scalar absorb: absorb just the base field element(s), NOT a full ring element.
    ///
    /// This reduces absorbed elements from d (ring dimension, e.g. 16 for Frog) to the base field
    /// extension degree (typically 1 for prime fields). Security is preserved because:
    /// 1. The transcript schedule is fixed and deterministic (not adversary-controlled)
    /// 2. Both prover and verifier follow the identical absorption sequence
    /// 3. Different values at any absorption point lead to different transcript states
    ///
    /// For SP1 one-proof mode (ℓ=47, l_h=512): reduces J absorption from ~385k elements to ~24k.
    fn absorb_field_element(&mut self, v: &R::BaseRing) {
        self.scratch.clear();
        self.scratch.extend(v.to_base_prime_field_elements());
        self.metrics.absorbed_elems += self.scratch.len() as u64;
        self.sponge.absorb(&self.scratch);
    }

    fn get_challenge(&mut self) -> R::BaseRing {
        let extension_degree = R::BaseRing::extension_degree();
        let c = self
            .sponge
            .squeeze_field_elements(extension_degree as usize);
        self.metrics.squeezed_field_elems += c.len() as u64;
        // `get_challenge` re-absorbs the squeezed elements to evolve the sponge state.
        self.metrics.absorbed_elems += c.len() as u64;
        self.sponge.absorb(&c);
        <R::BaseRing as Field>::from_base_prime_field_elems(&c)
            .expect("something went wrong: c does not contain extension_degree elements")
    }

    fn squeeze_bytes(&mut self, n: usize) -> Vec<u8> {
        self.metrics.squeezed_bytes += n as u64;
        self.sponge.squeeze_bytes(n)
    }
}

pub fn squeeze_challenges<R: OverField>(
    transcript: &mut impl Transcript<R>,
    name: &str,
    n: usize,
) -> Vec<R::BaseRing> {
    transcript.absorb_field_element(&<R::BaseRing as Field>::from_base_prime_field(
        <R::BaseRing as Field>::BasePrimeField::from_be_bytes_mod_order(name.as_bytes()),
    ));

    transcript.get_challenges(n)
}

pub fn squeeze_rchallenges<R: OverField>(
    transcript: &mut impl Transcript<R>,
    name: &str,
    n: usize,
) -> Vec<R> {
    squeeze_challenges(transcript, name, n)
        .into_iter()
        .map(|z| R::from(z))
        .collect::<Vec<R>>()
}
