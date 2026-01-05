use latticefold::transcript::Transcript;
use stark_rings::OverField;

use crate::public_coin_transcript::CoinEvent;

/// Wraps a real transcript (e.g., PoseidonTranscript) and records the exact coin stream it outputs.
///
/// This supports the Symphony CP-style intent:
/// - Fiat–Shamir is computed **externally** (by the verifier / outside the locked relation),
/// - but the locked relation can take the resulting coins as explicit public inputs, without
///   re-implementing hashing inside the relation.
#[derive(Clone, Debug)]
pub struct RecordingTranscript<R: OverField, T: Transcript<R>> {
    inner: T,
    pub coins_challenges: Vec<R::BaseRing>,
    pub coins_bytes: Vec<u8>,
    pub events: Vec<CoinEvent>,
}

impl<R: OverField, T: Transcript<R>> RecordingTranscript<R, T> {
    pub fn new(inner: T) -> Self {
        Self {
            inner,
            coins_challenges: Vec::new(),
            coins_bytes: Vec::new(),
            events: Vec::new(),
        }
    }

    pub fn into_inner(self) -> T {
        self.inner
    }
}

impl<R: OverField, T: Transcript<R>> Transcript<R> for RecordingTranscript<R, T> {
    type TranscriptConfig = T::TranscriptConfig;

    fn new(config: &Self::TranscriptConfig) -> Self {
        Self::new(T::new(config))
    }

    fn absorb(&mut self, v: &R) {
        self.inner.absorb(v);
    }

    fn get_challenge(&mut self) -> R::BaseRing {
        let c = self.inner.get_challenge();
        self.coins_challenges.push(c);
        self.events.push(CoinEvent::Challenge);
        c
    }

    fn squeeze_bytes(&mut self, n: usize) -> Vec<u8> {
        let b = self.inner.squeeze_bytes(n);
        self.coins_bytes.extend_from_slice(&b);
        self.events.push(CoinEvent::Bytes(n));
        b
    }
}

/// Like `RecordingTranscript`, but wraps a mutable reference to an existing transcript.
///
/// This is useful when a higher-level protocol owns the transcript state and we want to
/// record the coin stream produced by a subprotocol without moving/cloning the transcript.
#[derive(Debug)]
pub struct RecordingTranscriptRef<'a, R: OverField, T: Transcript<R>> {
    inner: &'a mut T,
    pub coins_challenges: Vec<R::BaseRing>,
    pub coins_bytes: Vec<u8>,
    pub events: Vec<CoinEvent>,
}

impl<'a, R: OverField, T: Transcript<R>> RecordingTranscriptRef<'a, R, T> {
    pub fn new(inner: &'a mut T) -> Self {
        Self {
            inner,
            coins_challenges: Vec::new(),
            coins_bytes: Vec::new(),
            events: Vec::new(),
        }
    }
}

impl<'a, R: OverField, T: Transcript<R>> Transcript<R> for RecordingTranscriptRef<'a, R, T> {
    type TranscriptConfig = T::TranscriptConfig;

    fn new(_config: &Self::TranscriptConfig) -> Self {
        panic!("RecordingTranscriptRef::new must be constructed from an existing transcript ref");
    }

    fn absorb(&mut self, v: &R) {
        self.inner.absorb(v);
    }

    fn get_challenge(&mut self) -> R::BaseRing {
        let c = self.inner.get_challenge();
        self.coins_challenges.push(c);
        self.events.push(CoinEvent::Challenge);
        c
    }

    fn squeeze_bytes(&mut self, n: usize) -> Vec<u8> {
        let b = self.inner.squeeze_bytes(n);
        self.coins_bytes.extend_from_slice(&b);
        self.events.push(CoinEvent::Bytes(n));
        b
    }
}

