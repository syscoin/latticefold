use latticefold::transcript::Transcript;
use stark_rings::OverField;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CoinEvent {
    Challenge,
    Bytes(usize),
}

/// A "public-coin transcript": challenges/bytes are provided explicitly, rather than derived by hashing.
///
/// This is the mechanism needed for a CP-/public-coin style relation where verifier coins are part of
/// the public statement (instead of being computed via Fiat–Shamir inside the relation).
///
/// Security notes:
/// - `absorb()` is a no-op here, so **coins are not bound to prior prover messages**.
/// - This is useful as a *measurement / CP-replay harness* (hash-free predicate replay),
///   but it is **not** a drop-in replacement for Fiat–Shamir (ROM) soundness.
/// - Prefer deriving coins from a real transcript externally (e.g. Poseidon), then replaying
///   those coins with this transcript for the hash-free relation.
#[derive(Clone, Debug, Default)]
pub struct FixedTranscript<R: OverField> {
    challenges: Vec<R::BaseRing>,
    challenge_idx: usize,
    bytes: Vec<u8>,
    bytes_idx: usize,
    events: Option<Vec<CoinEvent>>,
    event_idx: usize,
}

impl<R: OverField> FixedTranscript<R> {
    pub fn new_with_coins(challenges: Vec<R::BaseRing>, bytes: Vec<u8>) -> Self {
        Self {
            challenges,
            challenge_idx: 0,
            bytes,
            bytes_idx: 0,
            events: None,
            event_idx: 0,
        }
    }

    pub fn new_with_coins_and_events(
        challenges: Vec<R::BaseRing>,
        bytes: Vec<u8>,
        events: Vec<CoinEvent>,
    ) -> Self {
        Self {
            challenges,
            challenge_idx: 0,
            bytes,
            bytes_idx: 0,
            events: Some(events),
            event_idx: 0,
        }
    }

    pub fn remaining_challenges(&self) -> usize {
        self.challenges.len().saturating_sub(self.challenge_idx)
    }

    pub fn remaining_bytes(&self) -> usize {
        self.bytes.len().saturating_sub(self.bytes_idx)
    }

    pub fn remaining_events(&self) -> usize {
        self.events
            .as_ref()
            .map(|e| e.len().saturating_sub(self.event_idx))
            .unwrap_or(0)
    }

    fn expect_event(&mut self, expected: CoinEvent) {
        let Some(events) = &self.events else { return };
        let got = events.get(self.event_idx).unwrap_or_else(|| {
            panic!(
                "FixedTranscript: out of events at idx={}, expected {:?}",
                self.event_idx, expected
            )
        });
        if *got != expected {
            panic!(
                "FixedTranscript: event mismatch at idx={}, got {:?}, expected {:?}",
                self.event_idx, got, expected
            );
        }
        self.event_idx += 1;
    }
}

impl<R: OverField> Transcript<R> for FixedTranscript<R> {
    type TranscriptConfig = ();

    fn new(_: &Self::TranscriptConfig) -> Self {
        Self::default()
    }

    fn absorb(&mut self, _v: &R) {
        // no-op (public-coin)
    }

    fn get_challenge(&mut self) -> R::BaseRing {
        self.expect_event(CoinEvent::Challenge);
        let c = *self
            .challenges
            .get(self.challenge_idx)
            .unwrap_or_else(|| panic!("FixedTranscript: out of challenges at idx={}", self.challenge_idx));
        self.challenge_idx += 1;
        c
    }

    fn squeeze_bytes(&mut self, n: usize) -> Vec<u8> {
        self.expect_event(CoinEvent::Bytes(n));
        let end = self
            .bytes_idx
            .checked_add(n)
            .unwrap_or_else(|| panic!("FixedTranscript: squeeze_bytes overflow"));
        if end > self.bytes.len() {
            panic!(
                "FixedTranscript: out of bytes: requested {}, remaining {}",
                n,
                self.bytes.len().saturating_sub(self.bytes_idx)
            );
        }
        let out = self.bytes[self.bytes_idx..end].to_vec();
        self.bytes_idx = end;
        out
    }
}

