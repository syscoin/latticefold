//! Opening / commitment verification hooks (paper `VfyOpen` layer).
//!
//! The Symphony paper’s Π_fold / Π_gr1cs statements include a commitment-opening layer
//! (often written `VfyOpen(...)`) that lets the verifier check polynomial evaluations
//! without the prover sending full per-instance data.
//!
//! In this repo, we are not yet instantiating the full opening layer; however we
//! centralize the interface here so we can plug in a real commitment/PCS later.

use latticefold::transcript::Transcript;
use stark_rings::{CoeffRing, Zq};
use latticefold::commitment::AjtaiCommitmentScheme;
use std::collections::HashMap;

#[derive(Clone, Copy, Debug, Default)]
pub struct NoOpen;

/// Abstract opening verifier.
///
/// A concrete instantiation will typically:
/// - bind a commitment into the transcript,
/// - verify an opening proof at a point,
/// - and (optionally) bind the opened value.
///
/// For now, we provide `NoOpen` which performs no checks.
pub trait VfyOpen<R: CoeffRing>
where
    R::BaseRing: Zq,
{
    fn verify_opening<T: Transcript<R>>(
        &self,
        _transcript: &mut T,
        _domain: &'static str,
        _commitment: &[R],
        _point: &[R::BaseRing],
        _value: &[R],
        _proof_bytes: &[u8],
    ) -> Result<(), String>;
}

impl<R: CoeffRing> VfyOpen<R> for NoOpen
where
    R::BaseRing: Zq,
{
    fn verify_opening<T: Transcript<R>>(
        &self,
        _transcript: &mut T,
        _domain: &'static str,
        _commitment: &[R],
        _point: &[R::BaseRing],
        _value: &[R],
        _proof_bytes: &[u8],
    ) -> Result<(), String> {
        Ok(())
    }
}

/// Concrete opening verifier for Ajtai commitments: checks `A * m == c`.
#[derive(Clone, Debug)]
pub struct AjtaiOpenVerifier<R: CoeffRing>
where
    R::BaseRing: Zq,
{
    pub scheme: AjtaiCommitmentScheme<R>,
}

impl<R: CoeffRing> VfyOpen<R> for AjtaiOpenVerifier<R>
where
    R::BaseRing: Zq,
{
    fn verify_opening<T: Transcript<R>>(
        &self,
        _transcript: &mut T,
        _domain: &'static str,
        commitment: &[R],
        _point: &[R::BaseRing],
        value: &[R],
        _proof_bytes: &[u8],
    ) -> Result<(), String> {
        let expected = self
            .scheme
            .commit(value)
            .map_err(|e| format!("AjtaiOpen: commit error: {e:?}"))?;
        if expected.as_ref() != commitment {
            return Err("AjtaiOpen: commitment mismatch".to_string());
        }
        Ok(())
    }
}

/// A `VfyOpen` that supports multiple Ajtai commitment schemes keyed by `domain`.
///
/// This is useful for WE/DPP-facing relations where the statement includes several distinct
/// commitment families, e.g. `cm_f` (big witness commitment) vs `cfs_*` (CP transcript-message
/// commitments), each with different message lengths / matrices.
#[derive(Clone, Debug, Default)]
pub struct MultiAjtaiOpenVerifier<R: CoeffRing>
where
    R::BaseRing: Zq,
{
    pub schemes: HashMap<&'static str, AjtaiCommitmentScheme<R>>,
}

impl<R: CoeffRing> MultiAjtaiOpenVerifier<R>
where
    R::BaseRing: Zq,
{
    pub fn new() -> Self {
        Self { schemes: HashMap::new() }
    }

    pub fn with_scheme(mut self, domain: &'static str, scheme: AjtaiCommitmentScheme<R>) -> Self {
        self.schemes.insert(domain, scheme);
        self
    }
}

impl<R: CoeffRing> VfyOpen<R> for MultiAjtaiOpenVerifier<R>
where
    R::BaseRing: Zq,
{
    fn verify_opening<T: Transcript<R>>(
        &self,
        _transcript: &mut T,
        domain: &'static str,
        commitment: &[R],
        _point: &[R::BaseRing],
        value: &[R],
        _proof_bytes: &[u8],
    ) -> Result<(), String> {
        let scheme = self
            .schemes
            .get(domain)
            .ok_or_else(|| format!("AjtaiOpen: unknown domain `{domain}`"))?;
        let expected = scheme
            .commit(value)
            .map_err(|e| format!("AjtaiOpen: commit error: {e:?}"))?;
        if expected.as_ref() != commitment {
            return Err(format!("AjtaiOpen: commitment mismatch for domain `{domain}`"));
        }
        Ok(())
    }
}
