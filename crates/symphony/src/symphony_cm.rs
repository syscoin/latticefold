//! Symphony CP-style (public-coin) wrapper - prototype.
//!
//! This module provides a minimal "commit-and-prove" style interface for the Pi_rg scaffold:
//! - The statement includes explicit verifier coins, replayed via `FixedTranscript`.
//! - The Ajtai opening check `A*f == cm_f` is performed outside the Pi_rg relation.
//!
//! Notes:
//! - `prove_rg` / `verify_rg` are a measurement harness and do NOT enforce Fiat-Shamir binding.
//! - `prove_rg_fs` records Fiat-Shamir coins from a real transcript (Poseidon) and then replays
//!   them through `FixedTranscript` to keep the predicate hash-free while preserving sequencing.
//! - `verify_rg_fs_bound` recomputes the Fiat-Shamir coin stream externally and checks it matches
//!   the statement coins, restoring ROM-style binding outside the relation.

use stark_rings::{balanced_decomposition::Decompose, CoeffRing, OverField, PolyRing, Zq};
use stark_rings_linalg::Matrix;

use crate::{
    public_coin_transcript::{CoinEvent, FixedTranscript},
    recording_transcript::RecordingTranscript,
    rp_rgchk::{
        bind_pi_rg_transcript, verify_monomial_plus_projection_consistency, RPParams, RPRangeProver,
        RPRangeProof,
    },
    transcript::PoseidonTranscript,
};

#[derive(Clone, Copy, Debug)]
pub struct CoinCounts {
    pub challenges: usize,
    pub bytes: usize,
}

/// Exact coin requirements for the Pi_rg scaffold in this codebase.
///
/// This lets us construct CP-style statements with explicit public coins of minimal size.
pub fn rg_coin_counts<R: PolyRing>(n: usize, params: &RPParams) -> CoinCounts {
    assert!(n > 0);
    assert!(n % params.l_h == 0, "l_h must divide n");
    let d = R::dimension();
    assert!(d.is_power_of_two(), "ring dimension must be power-of-two");

    let blocks = n / params.l_h;
    let m = blocks * params.lambda_pj;
    assert!(m.is_power_of_two(), "Pi_rg assumes m is power-of-two (got {})", m);

    let g_len = m * d;
    assert!(
        g_len.is_power_of_two(),
        "Pi_rg assumes m*d is power-of-two (got {})",
        g_len
    );
    let g_nvars = g_len.trailing_zeros() as usize; // log2(g_len)

    let nclaims = params.k_g;
    // setchk consumes:
    // - per claim: get_challenges(nvars) + beta + alpha  => nvars + 2
    // - optional rc if batching (nclaims>1)              => +1
    // - sumcheck rounds: one get_challenge per round     => +nvars
    let challenges = nclaims * (g_nvars + 2) + if nclaims > 1 { 1 } else { 0 } + g_nvars;

    // derive_J uses squeeze_bytes(lambda_pj*l_h)
    let bytes = params.lambda_pj * params.l_h;

    CoinCounts { challenges, bytes }
}

/// Public statement for the CP-style range check.
#[derive(Clone, Debug)]
pub struct SymphonyRgStatement<R: OverField> {
    /// Ajtai commitment to f: cm_f = A * f.
    pub cm_f: Vec<R>,
    /// Coins used by the verifier/transcript.
    pub coins: SymphonyCoins<R>,
}

/// Explicit coin stream + call schedule.
#[derive(Clone, Debug)]
pub struct SymphonyCoins<R: PolyRing> {
    /// Stream of ring base-field challenges returned by `Transcript::get_challenge()`.
    pub challenges: Vec<R::BaseRing>,
    /// Stream of bytes returned by `Transcript::squeeze_bytes()`.
    pub bytes: Vec<u8>,
    /// Call schedule for cross-type interleaving between challenges and bytes.
    pub events: Vec<CoinEvent>,
}

/// Proof object (currently just Pi_rg's output).
#[derive(Clone, Debug)]
pub struct SymphonyRgProof<R: PolyRing> {
    pub rp: RPRangeProof<R>,
}

/// Prover: given witness `f` and matrix `A`, produce the CP-style statement+proof.
///
/// WARNING: This does not enforce Fiat-Shamir binding (the prover chooses `coins`).
pub fn prove_rg<R: CoeffRing>(
    f: Vec<R>,
    A: &Matrix<R>,
    params: RPParams,
    coins: SymphonyCoins<R>,
) -> (SymphonyRgStatement<R>, SymphonyRgProof<R>)
where
    R::BaseRing: Zq + Decompose,
{
    let cm_f = A.try_mul_vec(&f).unwrap();
    let mut transcript = FixedTranscript::<R>::new_with_coins_and_events(
        coins.challenges.clone(),
        coins.bytes.clone(),
        coins.events.clone(),
    );

    let prover = RPRangeProver::<R>::new(f, params);
    let rp = prover.prove(&mut transcript, &cm_f);
    // Ensure coins were exactly sufficient (no over/under-provisioning).
    assert_eq!(transcript.remaining_challenges(), 0);
    assert_eq!(transcript.remaining_bytes(), 0);
    assert_eq!(transcript.remaining_events(), 0);

    (SymphonyRgStatement { cm_f, coins }, SymphonyRgProof { rp })
}

/// Verifier: check opening (external) and Pi_rg (replay).
///
/// WARNING: This does not enforce Fiat-Shamir binding; it only checks consistency w.r.t. provided coins.
pub fn verify_rg<R: CoeffRing>(
    A: &Matrix<R>,
    opening_f: &[R],
    stmt: &SymphonyRgStatement<R>,
    proof: &SymphonyRgProof<R>,
) -> bool
where
    R::BaseRing: Zq,
{
    // External Ajtai opening check (outside Pi_rg relation).
    if A.try_mul_vec(opening_f).ok().as_deref() != Some(&stmt.cm_f) {
        return false;
    }

    // Replay verification using explicit coins (measurement harness).
    let mut transcript = FixedTranscript::<R>::new_with_coins_and_events(
        stmt.coins.challenges.clone(),
        stmt.coins.bytes.clone(),
        stmt.coins.events.clone(),
    );
    let ok =
        verify_monomial_plus_projection_consistency(&proof.rp, &stmt.cm_f, &mut transcript).is_ok();
    let coins_ok = transcript.remaining_challenges() == 0
        && transcript.remaining_bytes() == 0
        && transcript.remaining_events() == 0;
    ok && coins_ok
}

/// Prove Pi_rg while recording the Fiat-Shamir coin stream produced by PoseidonTranscript.
///
/// This is the Symphony-faithful CP-style pattern:
/// - FS is computed externally (here by PoseidonTranscript),
/// - we record the coin stream and attach it to the statement,
/// - the hash-free predicate can then replay with FixedTranscript and the explicit coins.
pub fn prove_rg_fs<R: CoeffRing, PC>(
    f: Vec<R>,
    A: &Matrix<R>,
    params: RPParams,
) -> (SymphonyRgStatement<R>, SymphonyRgProof<R>)
where
    R::BaseRing: Zq + Decompose,
    PC: cyclotomic_rings::rings::GetPoseidonParams<<<R>::BaseRing as ark_ff::Field>::BasePrimeField>,
{
    let cm_f = A.try_mul_vec(&f).unwrap();

    let ts = PoseidonTranscript::<R>::empty::<PC>();
    let mut rts = RecordingTranscript::<R, _>::new(ts);
    let prover = RPRangeProver::<R>::new(f, params);
    let rp = prover.prove(&mut rts, &cm_f);

    let coins = derive_pi_rg_fs_coins::<R, PC>(&cm_f, &rp);

    // Replay with FixedTranscript and ensure it matches (sanity check).
    let mut fts = FixedTranscript::<R>::new_with_coins_and_events(
        coins.challenges.clone(),
        coins.bytes.clone(),
        coins.events.clone(),
    );
    assert!(verify_monomial_plus_projection_consistency(&rp, &cm_f, &mut fts).is_ok());
    assert_eq!(fts.remaining_challenges(), 0);
    assert_eq!(fts.remaining_bytes(), 0);
    assert_eq!(fts.remaining_events(), 0);

    (SymphonyRgStatement { cm_f, coins }, SymphonyRgProof { rp })
}

/// Verifier that enforces Fiat-Shamir binding (ROM) outside the relation.
///
/// It recomputes the FS coin stream from PoseidonTranscript while running Pi_rg verification,
/// and checks that it exactly matches `stmt.coins` (including the call schedule).
pub fn verify_rg_fs_bound<R: CoeffRing, PC>(
    A: &Matrix<R>,
    opening_f: &[R],
    stmt: &SymphonyRgStatement<R>,
    proof: &SymphonyRgProof<R>,
) -> bool
where
    R::BaseRing: Zq,
    PC: cyclotomic_rings::rings::GetPoseidonParams<<<R>::BaseRing as ark_ff::Field>::BasePrimeField>,
{
    // External Ajtai opening check (outside Pi_rg relation).
    if A.try_mul_vec(opening_f).ok().as_deref() != Some(&stmt.cm_f) {
        return false;
    }

    let derived = derive_pi_rg_fs_coins::<R, PC>(&stmt.cm_f, &proof.rp);
    derived.challenges == stmt.coins.challenges
        && derived.bytes == stmt.coins.bytes
        && derived.events == stmt.coins.events
}

/// Derive the Fiat-Shamir coin stream (including schedule) for Pi_rg from the canonical
/// public transcript binder.
pub fn derive_pi_rg_fs_coins<R: CoeffRing, PC>(
    cm_f: &[R],
    rp: &RPRangeProof<R>,
) -> SymphonyCoins<R>
where
    R::BaseRing: Zq,
    PC: cyclotomic_rings::rings::GetPoseidonParams<<<R>::BaseRing as ark_ff::Field>::BasePrimeField>,
{
    let ts = PoseidonTranscript::<R>::empty::<PC>();
    let mut rts = RecordingTranscript::<R, _>::new(ts);
    bind_pi_rg_transcript(rp, cm_f, &mut rts);
    SymphonyCoins::<R> {
        challenges: rts.coins_challenges,
        bytes: rts.coins_bytes,
        events: rts.events,
    }
}

