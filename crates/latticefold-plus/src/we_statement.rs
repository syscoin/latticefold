//! Helpers to bind WE locks to a statement (proof-agnostic arming).
//!
//! Following Symphony’s model, the armer derives per-lock coins from a **statement hash** that binds:
//! - a program/verifier id (`vk_hash`)
//! - an instance digest (`r1cs_digest`)
//! - a gate digest (`gate_digest`, i.e. exact WE gate relation version)
//! - and the **public inputs**
//!
//! Importantly, this intentionally does **not** include prover-chosen proof artifacts (e.g. Ajtai
//! commitments to witness), since arming must be possible without observing a specific proving run.

use ark_crypto_primitives::sponge::{poseidon::PoseidonSponge, CryptographicSponge};
use ark_ff::{Field, PrimeField};
use latticefold::transcript::poseidon::{f257_poseidon_config, F257};
use sha2::{Digest, Sha256};
use stark_rings::OverField;

/// Fixed parameters that must be statement-bound for WE/DPP.
///
/// These are implicit in native Rust (loop bounds, type params), but must be explicit in an
/// arithmetized WE gate to prevent “reinterpretation under different sizes”.
#[derive(Clone, Debug)]
pub struct WeParams {
    pub nvars_setchk: u64,
    pub degree_setchk: u64,
    pub nvars_cm: u64,
    pub degree_cm: u64,
    pub kappa: u64,
    pub ring_dim_d: u64,
    /// Balanced decomposition base `b` used in LF+ range-check (`RgChk`) digit decomposition.
    ///
    /// For SP1 BabyBear-in-Goldilocks boundedness, this is statement-bound (e.g. Goldilocks64 uses `16`).
    pub decomp_b: u64,
    pub k: u64,
    pub l: u64,
    pub mlen: u64,
}

impl WeParams {
    pub fn to_field_vec<BF: PrimeField>(&self) -> Vec<BF> {
        vec![
            BF::from(self.nvars_setchk),
            BF::from(self.degree_setchk),
            BF::from(self.nvars_cm),
            BF::from(self.degree_cm),
            BF::from(self.kappa),
            BF::from(self.ring_dim_d),
            BF::from(self.decomp_b),
            BF::from(self.k),
            BF::from(self.l),
            BF::from(self.mlen),
        ]
    }
}

/// Tiny-field Poseidon hash wrapper returning 32 native field elements.
///
/// - Absorbs input bytes as F257 elements in `[0,255]`.
/// - Squeezes 32 full F257 elements (`0..=256`) with no byte projection.
#[inline]
pub fn tiny_hash32_fields(parts: &[&[u8]]) -> [F257; 32] {
    let cfg = f257_poseidon_config();
    let mut sponge = PoseidonSponge::<F257>::new(&cfg);
    for p in parts {
        if p.is_empty() {
            continue;
        }
        let elems: Vec<F257> = p.iter().map(|&b| F257::from(b as u64)).collect();
        sponge.absorb(&elems);
    }
    let out_vec = sponge.squeeze_field_elements::<F257>(32);
    out_vec
        .try_into()
        .expect("squeeze_field_elements(32) must return exactly 32 elements")
}

/// Tiny-field Poseidon hash wrapper (native field outputs) with a compact one-byte domain separator.
#[inline]
pub fn tiny_hash32_fields_with_domain(domain: u8, parts: &[&[u8]]) -> [F257; 32] {
    let domain_buf = [domain];
    let mut with_domain: Vec<&[u8]> = Vec::with_capacity(parts.len() + 1);
    with_domain.push(&domain_buf);
    with_domain.extend_from_slice(parts);
    tiny_hash32_fields(&with_domain)
}

/// Tiny-field Poseidon hash wrapper returning 32 native field elements from field-element absorbs.
#[inline]
pub fn tiny_hash32_field_elems(parts: &[&[F257]]) -> [F257; 32] {
    let cfg = f257_poseidon_config();
    let mut sponge = PoseidonSponge::<F257>::new(&cfg);
    for p in parts {
        if p.is_empty() {
            continue;
        }
        sponge.absorb(&p.to_vec());
    }
    let out_vec = sponge.squeeze_field_elements::<F257>(32);
    out_vec
        .try_into()
        .expect("squeeze_field_elements(32) must return exactly 32 elements")
}

/// Tiny-field Poseidon hash wrapper (field absorbs + compact one-element domain separator).
#[inline]
pub fn tiny_hash32_field_elems_with_domain(domain: u8, parts: &[&[F257]]) -> [F257; 32] {
    let domain_elem = [F257::from(domain as u64)];
    let mut with_domain: Vec<&[F257]> = Vec::with_capacity(parts.len() + 1);
    with_domain.push(&domain_elem);
    with_domain.extend_from_slice(parts);
    tiny_hash32_field_elems(&with_domain)
}

/// Tiny-field Poseidon hash of the **public** statement for `R_WE` (LF+).
///
/// This intentionally excludes proof artifacts (e.g. witness commitments).
///
/// Structure:
/// - `h_ids = TinyHash(ds=1 || vk_hash || r1cs_digest || gate_digest)`
/// - `h_params = TinyHash(ds=2 || encode_le(params))`
/// - `we_core_digest = TinyHash(ds=3 || h_ids || h_params)`
/// - `stmt_digest = TinyHash(ds=4 || we_core_digest || committed_values_prefix_bytes)`
pub fn we_statement_hash_lf_plus<R: OverField>(
    vk_hash: [u8; 32],
    committed_values_prefix_bytes: [F257; 64],
    r1cs_digest: [u8; 32],
    gate_digest: [u8; 32],
    params: &WeParams,
) -> [F257; 32]
where
    R::BaseRing: Field,
{
    let h_ids = tiny_hash32_fields_with_domain(1, &[&vk_hash, &r1cs_digest, &gate_digest]);
    let h_params = params.to_field_vec::<F257>();
    tiny_hash32_field_elems_with_domain(2, &[&h_ids, &h_params, &committed_values_prefix_bytes])
}

/// Canonical encoding for a 32-byte digest as a field element.
///
/// Interpret the digest as a **little-endian** integer and reduce mod p.
pub fn digest32_to_field<BF: PrimeField>(digest: [u8; 32]) -> BF {
    BF::from_le_bytes_mod_order(&digest)
}

/// Collision-robust encoding of a 32-byte digest as **256 boolean field elements** (bits).
///
/// - Output length is exactly 256.
/// - Bit order is little-endian within each byte (LSB-first), matching typical `from_le_bytes` conventions.
///
/// This is intended for WE/DPP public inputs when we want:
/// - statement binding without relying on `mod p` reduction uniqueness, and
/// - DPP parameters that assume small / boolean public inputs.
pub fn digest32_to_bits_field<BF: PrimeField>(digest: [u8; 32]) -> Vec<BF> {
    let mut out = Vec::with_capacity(256);
    for &b in &digest {
        for i in 0..8 {
            let bit = (b >> i) & 1;
            out.push(BF::from(bit as u64));
        }
    }
    debug_assert_eq!(out.len(), 256);
    out
}


/// Gate digest for the LF+ WE gate relation.
///
/// In the SP1/WE arming model this should be a **precomputed constant** identifying the exact
/// WE gate version (code + constraint system generation), not something derived per-proof.
///
/// Temporary nonzero domain separator for this gate version.
///
/// Value is `SHA256("LFP_WE_GATE_DIGEST_V1")`.
///
/// NOTE: This is *not* a hash of the code/constraint system; it is just a stable, nonzero label.
pub const LFP_WE_GATE_DIGEST_V1: [u8; 32] = [
    0xc2, 0x50, 0x46, 0x54, 0x4b, 0x5b, 0xb8, 0xcc, 0x6f, 0xef, 0x2b, 0x55, 0x27, 0xb0,
    0x17, 0x77, 0xfb, 0x54, 0x0d, 0x5f, 0x1b, 0xc6, 0xda, 0x2b, 0x0d, 0x00, 0xc7, 0x4e,
    0xda, 0x7e, 0x02, 0x25,
];

/// Deterministic statement encoding for WE/DPP.
///
/// Current encoding:
/// - `x[0] = 1` (shared constant slot convention)
/// - fixed params (see `WeParams`)
/// - optional extra statement elements (e.g. commitment surface limbs, transcript-bound absorbs)
///
/// NOTE: The exact set of extra statement elements is decided by the WE arithmetizer; this module
/// just provides a stable prefix layout for params.
pub fn encode_public_x<BF: PrimeField>(stmt_digest: &[BF; 32]) -> Vec<BF> {
    let mut out = Vec::with_capacity(1 + stmt_digest.len());
    out.push(BF::ONE);
    out.extend_from_slice(stmt_digest);
    out
}

/// Deterministically derive the per-lock public coin seed.
pub fn derive_lock_coin_seed(
    armer_seed: [u8; 32],
    stmt_digest: [u8; 32],
    lock_j: u64,
) -> [u8; 32] {
    let mut h = Sha256::new();
    h.update(b"LFP_LOCK_COIN_V1");
    h.update(&armer_seed);
    h.update(&stmt_digest);
    h.update(&lock_j.to_le_bytes());
    h.finalize().into()
}

/// Derive an armer-private secret vector from `(armer_seed, stmt_digest, lock_j)`.
///
/// The output length controls the entropy injected into hidden-query derivation.
pub fn derive_armer_secret<BF: PrimeField>(
    armer_seed: [u8; 32],
    stmt_digest: [u8; 32],
    lock_j: u64,
    len: usize,
) -> Vec<BF> {
    let mut out = Vec::with_capacity(len);
    for i in 0..len {
        let mut h = Sha256::new();
        h.update(b"LFP_ARMER_SECRET_V1");
        h.update(&armer_seed);
        h.update(&stmt_digest);
        h.update(&lock_j.to_le_bytes());
        h.update(&(i as u64).to_le_bytes());
        let d: [u8; 32] = h.finalize().into();
        out.push(BF::from_le_bytes_mod_order(&d));
    }
    out
}

/// Arm a Theorem-4.3 tiny-field lock using statement-bound coins + armer-private secret.
///
/// - `stmt_digest` binds to the WE statement (public inputs + params + gate digest).
/// - `armer_seed` is private to each armer (N-of-N: each armer uses an independent seed).
/// - `lock_j` is the armer’s lock index (unique per armer/lock).
/// - `block_id` selects the chunk (derived from the chunked dR1CS instance).
/// - `rep_id` selects the public repetition index.
#[cfg(feature = "we_gate")]
pub fn arm_theorem43_from_statement<F: PrimeField>(
    dpp: &dpp::theorem43::Theorem43Dpp<F, impl dpp::dr1cs_flpcp::Dr1csNpFlpcpSparseApi<F>>,
    stmt_digest: [u8; 32],
    x: &[F],
    armer_seed: [u8; 32],
    lock_j: u64,
    block_id: usize,
    rep_id: u64,
) -> Result<dpp::theorem43::Theorem43LockArtifact<F>, String> {
    // Statement-binding commitment: 256 boolean field elements.
    let c_stmt = digest32_to_bits_field::<F>(stmt_digest);

    // Armer-private secret mixed into the hidden-query derivation.
    let armer_secret = derive_armer_secret::<F>(armer_seed, stmt_digest, lock_j, 4);

    dpp.arm(&c_stmt, x, &armer_secret, block_id, rep_id)
}
