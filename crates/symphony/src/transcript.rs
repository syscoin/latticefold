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


/// Transcript using Poseidon sponge
///
/// Same implementation as LatticeFold's `PoseidonTranscript` though without the challenge set
/// generic / requirement on `SuitableRing`.
#[derive(Clone)]
pub struct PoseidonTranscript<R: OverField> {
    sponge: PoseidonSponge<F257>,
    metrics: PoseidonTranscriptMetrics,
    scratch: Vec<F257>,
    _marker: PhantomData<R>,
}

/// Lightweight transcript metrics to estimate Poseidon sponge work in `R_cp` / `R_WE`.
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

impl<R: OverField> PoseidonTranscript<R>
where
    R::BaseRing: PrimeField,
{
    pub fn empty<P>() -> Self {
        Self::new(&f257_poseidon_config())
    }

    pub fn metrics(&self) -> PoseidonTranscriptMetrics {
        self.metrics
    }
}

// =============================================================================
// Trace transcript (for algebraic/DPP frontends)
// =============================================================================

#[derive(Clone, Debug)]
pub enum PoseidonTraceOp<BF: PrimeField> {
    Absorb(Vec<BF>),
    SqueezeField(Vec<BF>),
    SqueezeBytes { n: usize, out: Vec<u8> },
}

/// Full transcript trace in terms of the Poseidon sponge's base prime field.
///
/// This is intended for *arithmetization frontends*: the prover can record a trace and later
/// provide it as part of a witness, with constraints enforcing that it matches the Poseidon
/// permutation schedule.
#[derive(Clone, Debug, Default)]
pub struct PoseidonTranscriptTrace<BF: PrimeField> {
    pub ops: Vec<PoseidonTraceOp<BF>>,
    pub absorbed: Vec<BF>,
    pub squeezed_field: Vec<BF>,
    pub squeezed_bytes: Vec<u8>,
}

#[derive(Clone)]
pub struct TracePoseidonTranscript<R: OverField> {
    sponge: PoseidonSponge<F257>,
    metrics: PoseidonTranscriptMetrics,
    scratch: Vec<F257>,
    trace: PoseidonTranscriptTrace<F257>,
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

    pub fn trace(&self) -> &PoseidonTranscriptTrace<F257> {
        &self.trace
    }

    fn absorb_f257_elems_vec(&mut self, elems: Vec<F257>) {
        self.metrics.absorbed_elems += elems.len() as u64;
        self.sponge.absorb(&elems);
        self.trace.absorbed.extend_from_slice(&elems);
        self.trace.ops.push(PoseidonTraceOp::Absorb(elems));
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
        self.scratch.clear();
        let bytes = prime_field_to_bytes_le_fixed::<R::BaseRing>(v);
        self.scratch.extend(bytes.iter().map(|b| F257::from(*b as u64)));
        let elems = self.scratch.clone();
        self.absorb_f257_elems_vec(elems);
    }

    fn get_challenge(&mut self) -> R::BaseRing {
        let modulus_le = <R::BaseRing as PrimeField>::MODULUS.to_bytes_le();
        let mut m: u64 = 0;
        for (i, b) in modulus_le.iter().take(8).enumerate() {
            m |= (*b as u64) << (8 * i);
        }
        assert!(m != 0, "unexpected zero modulus");

        loop {
            let c = self.sponge.squeeze_field_elements::<F257>(8);
            self.metrics.squeezed_field_elems += c.len() as u64;
            self.trace.squeezed_field.extend_from_slice(&c);
            self.trace.ops.push(PoseidonTraceOp::SqueezeField(c.clone()));

            // `get_challenge` re-absorbs the squeezed elements to evolve the sponge state.
            self.absorb_f257_elems_vec(c.clone());

            let mut acc: u128 = 0;
            let mut pow: u128 = 1;
            for e in &c {
                let d = e.into_bigint().to_bytes_le().get(0).copied().unwrap_or(0) as u16;
                debug_assert!(d < 257u16);
                acc += (d as u128) * pow;
                pow *= 257u128;
            }
            let x = acc as u64;
            if x < m {
                return R::BaseRing::from(x);
            }
        }
    }

    fn squeeze_bytes(&mut self, n: usize) -> Vec<u8> {
        self.metrics.squeezed_bytes += n as u64;
        let out = self.sponge.squeeze_bytes(n);
        self.trace.squeezed_bytes.extend_from_slice(&out);
        self.trace.ops.push(PoseidonTraceOp::SqueezeBytes { n, out: out.clone() });
        out
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
            // Small default; will grow as needed (e.g. absorbing a full ring element).
            scratch: Vec::with_capacity(64),
            _marker: PhantomData,
        }
    }

    fn absorb(&mut self, v: &R) {
        self.scratch.clear();
        let bytes = ring_to_bytes_le_fixed::<R>(v);
        self.scratch.extend(bytes.iter().map(|b| F257::from(*b as u64)));
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
        let bytes = prime_field_to_bytes_le_fixed::<R::BaseRing>(v);
        self.scratch.extend(bytes.iter().map(|b| F257::from(*b as u64)));
        self.metrics.absorbed_elems += self.scratch.len() as u64;
        self.sponge.absorb(&self.scratch);
    }

    fn get_challenge(&mut self) -> R::BaseRing {
        let modulus_le = <R::BaseRing as PrimeField>::MODULUS.to_bytes_le();
        let mut m: u64 = 0;
        for (i, b) in modulus_le.iter().take(8).enumerate() {
            m |= (*b as u64) << (8 * i);
        }
        assert!(m != 0, "unexpected zero modulus");

        loop {
            let c = self.sponge.squeeze_field_elements::<F257>(8);
            self.metrics.squeezed_field_elems += c.len() as u64;
            // `get_challenge` re-absorbs the squeezed elements to evolve the sponge state.
            self.metrics.absorbed_elems += c.len() as u64;
            self.sponge.absorb(&c);

            let mut acc: u128 = 0;
            let mut pow: u128 = 1;
            for e in &c {
                let d = e.into_bigint().to_bytes_le().get(0).copied().unwrap_or(0) as u16;
                debug_assert!(d < 257u16);
                acc += (d as u128) * pow;
                pow *= 257u128;
            }
            let x = acc as u64;
            if x < m {
                return R::BaseRing::from(x);
            }
        }
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
