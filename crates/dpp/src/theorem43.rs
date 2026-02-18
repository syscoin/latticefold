//! Tiny-field “lockable DPP” (A = {1,2}) for WE-style arming.
//!
//! This module provides a **non-enumerating** variant suitable for “arm-before-proof” demos:
//! - The armer samples a **single sparse query** `q`, but treats it as **toxic waste** (never published).
//! - The armer publishes only **public coins** plus an **accepting set** `A={1,2}` and uses `q` only
//!   internally to form a lock artifact (in production: LWE hints / AEAD KDF inputs).
//! - The prover/decapper later produces a proof **for the public coins**; decap uses the lock artifact
//!   without ever revealing `q`.
//!
//! This avoids any “enumerate all verifier randomness strings” blowup. It is intentionally
//! **not** a standard “proof-first” DPP in the strict sense; it matches a WE gate setting where
//! the query is hidden before the proof exists.
//!
//! Internally it uses the tiny-field gadgets from TR24-114 rev2 (Section 4.3.2–4.3.3):
//! - UV → powering via Vandermonde (gives Sq accepting set {0,1}), then we apply an affine shift
//!   to avoid the public target `0`.
//! - Multiplication relation checked via (ρ,σ) linearization and the Sq test
//! - Outer FLPCP for NP dR1CS provides (α,β,γ) = (E(Az)[i], E(Bz)[i], ...), with proof π₀=(z_w||w)

use ark_crypto_primitives::sponge::{poseidon::PoseidonSponge, CryptographicSponge};
use ark_ff::{BigInteger, PrimeField};
use rayon::prelude::*;
use std::marker::PhantomData;

use latticefold::transcript::bytes::field_to_bytes_le_fixed;
use latticefold::transcript::poseidon::{f257_poseidon_config, F257};

use crate::dr1cs_flpcp::{
    f_to_u16, is_f257_field, Dr1csNpFlpcpSparseApi, Dr1csQueryScratch, QuerySink,
};

#[cfg(test)]
mod tests {
    use super::*;
    use ark_ff::Field;

    // If we ever publish the full vector of scaled Sq coefficients `d_i = s * c_i` where
    // `c_i = -Σ_{λ∈U} λ^i` for i=1..256 over F257, then the polynomial
    //   T(u) = Σ_{i=1}^{256} d_i * u^i = s * S(u)
    // evaluates to either 0 or s on F257^*, so `s` becomes directly recoverable from public data.
    //
    // This test documents that algebraic fact.
    #[test]
    fn test_scaled_sq_power_sums_leak_scalar_via_indicator_eval() {
        // Pick a nonempty subset U ⊆ F257^*.
        let mut u_elems: Vec<F257> = Vec::new();
        for lam_u in 1u64..=256u64 {
            // Deterministic pseudo-random-ish subset: include ~half the elements.
            if (lam_u.wrapping_mul(17) ^ 0x5a) & 1 == 1 {
                u_elems.push(F257::from(lam_u));
            }
        }
        assert!(!u_elems.is_empty());

        // Compute c_i = -Σ_{λ∈U} λ^i for i=1..256.
        let mut c: Vec<F257> = vec![F257::from(0u64); 256];
        for i in 1u64..=256u64 {
            let mut acc = F257::from(0u64);
            for &lam in &u_elems {
                acc += lam.pow([i]);
            }
            c[(i - 1) as usize] = -acc;
        }

        // Choose s ∈ {1..256}.
        let s = F257::from(154u64);
        assert_ne!(s, F257::ZERO);

        // d_i = s * c_i
        let d: Vec<F257> = c.iter().map(|x| *x * s).collect();

        // Evaluate T(u) = Σ d_i u^i for all u in F257^*.
        let mut nonzero_vals: Vec<F257> = Vec::new();
        for u_u64 in 1u64..=256u64 {
            let u = F257::from(u_u64);
            let mut t = F257::from(0u64);
            let mut upow = u; // u^1
            for i in 0..256usize {
                t += d[i] * upow;
                upow *= u;
            }
            if t != F257::ZERO {
                nonzero_vals.push(t);
            }
        }
        // For this construction, T(u) should be either 0 or s, so all nonzero evals equal s.
        assert!(!nonzero_vals.is_empty());
        for v in nonzero_vals {
            assert_eq!(v, s);
        }
    }
}

/// Coins defining the single lockable query.
#[derive(Clone, Debug)]
pub struct Theorem43Coins<F: PrimeField> {
    pub idx: usize,
    pub lambda: F,
    pub rho: F,
    pub sigma: F,
    /// Public per-instance accepting-set base, so accepting set is `{c_hit, c_hit+1}`.
    ///
    /// This is derived deterministically from `(c_stmt, block_id, rep_id)` with domain separation.
    pub c_hit: F,
}

/// Public arming artifact for the hidden-query lockable DPP.
///
/// IMPORTANT:
/// - The query vector `q` is *not* part of the public API (it is stored privately here only as a
///   stand-in for “LWE hints” in this toy implementation).
/// - In the real PVUGC flow, this object would contain noisy LWE hints + AEAD material, not `q`.
#[derive(Clone, Debug)]
pub struct Theorem43LockArtifact<F: PrimeField> {
    /// Public statement-binding commitment (domain-separated).
    ///
    /// This is witness/proof-independent and can be agreed by all armers.
    pub c_stmt: Vec<F>,
    /// Accepting set (always `{1,2}`).
    pub accepting_set: [F; 2],
    /// Total length of `(x || π)` for sanity checks.
    pub len: usize,
    /// Public coins used by the prover to generate the matching proof.
    pub coins: Theorem43Coins<F>,
    /// Lightweight “full gate” stats (FS usage + query sparsity).
    pub stats: Theorem43ArmingStats,
    /// Hidden UV-derived Sq coefficients (TOXIC WASTE; do not publish).
    pub coeffs: Vec<F>,
}

#[derive(Clone, Debug, Default)]
pub struct Theorem43ArmingStats {
    /// Number of Poseidon sponge permutations used for FS derivation (best-effort).
    pub fs_permutes: u64,
    /// Number of nonzero terms in the hidden sparse query.
    pub q_nnz: usize,
}

impl<F: PrimeField> Theorem43LockArtifact<F> {}

/// Non-enumerating Theorem-4.3-style lockable query generator for the NP dR1CS FLPCP.
#[derive(Clone, Debug)]
pub struct Theorem43Dpp<F: PrimeField, P: Dr1csNpFlpcpSparseApi<F>> {
    flpcp: P,
    p: u64,
    q_minus_3: usize, // p-3
    _marker: PhantomData<F>,
}

/// Output of `stream_pi0_and_collect_tails`: per-coin `(alpha,beta,gamma)` (π-only).
///
/// The coin-dependent tail is streamed via the `on_tail_elem` callback (canonical path: no tail
/// vectors are allocated/stored by this function).
#[derive(Clone, Debug)]
pub struct Theorem43AbgTail<F: PrimeField> {
    pub alpha: F,
    pub beta: F,
    pub gamma: F,
}

/// Output of `stream_pi0_and_collect_abg_full`: per-coin `(alpha,beta,gamma)` over the full
/// vector `(x || π0)`.
#[derive(Clone, Debug)]
pub struct Theorem43AbgFull<F: PrimeField> {
    pub alpha: F,
    pub beta: F,
    pub gamma: F,
}

#[derive(Clone, Debug)]
struct Theorem43AbgAcc<F: PrimeField> {
    coins: Theorem43Coins<F>,
    alpha_pi: F,
    beta_pi: F,
    gamma_pi: F,
    alpha_full: F,
    beta_full: F,
    gamma_full: F,
}

/// Streaming query accumulator that tracks both:
/// - `acc_pi`: contribution from `π0` only (z_w + w_eval blocks)
/// - `acc_full`: contribution from `(x || π0)`
///
/// This is used to support statement-bound lock equations without double-counting `x`.
struct QueryStreamAcc<F: PrimeField> {
    acc_pi: F,
    acc_full: F,
    // Sparse terms over the selected block's `w_eval` slice in π0.
    //
    // In the canonical block-grouped schedule, each coin corresponds to exactly one block
    // (determined by `coins.idx / ell_local`), so we store only local `(coeff, pos)` terms for
    // that block to avoid per-block heap allocations.
    w_terms: Vec<(F, usize)>,
}

impl<F: PrimeField> QueryStreamAcc<F> {
    #[inline]
    fn add_w_eval(&mut self, w_eval: &[F]) -> Result<(), String> {
        for (c, pos) in self.w_terms.iter().copied() {
            if pos >= w_eval.len() {
                return Err("w_eval index out of range".to_string());
            }
            let t = c * w_eval[pos];
            self.acc_pi += t;
            self.acc_full += t;
        }
        Ok(())
    }
}

/// Helper to build `QueryStreamAcc` without materializing query vectors.
struct QueryStreamAccBuilder<'a, F: PrimeField> {
    x: &'a [F],
    z_w: &'a [F],
    z_w_len: usize,
    k_star: usize,
    blocks: usize,
    block_id: usize,
    n: usize,
    m: usize,
    acc_pi: F,
    acc_full: F,
    w_terms: Vec<(F, usize)>,
}

impl<'a, F: PrimeField> QueryStreamAccBuilder<'a, F> {
    fn new(
        x: &'a [F],
        z_w: &'a [F],
        z_w_len: usize,
        k_star: usize,
        blocks: usize,
        block_id: usize,
    ) -> Self {
        let n = x.len();
        let m = z_w_len + (k_star * blocks);
        Self {
            x,
            z_w,
            z_w_len,
            k_star,
            blocks,
            block_id,
            n,
            m,
            acc_pi: F::ZERO,
            acc_full: F::ZERO,
            w_terms: Vec::new(),
        }
    }

    #[inline]
    fn add_term(&mut self, c: F, idx: usize) -> Result<(), String> {
        if idx < self.n {
            self.acc_full += c * self.x[idx];
            return Ok(());
        }
        let pi_idx = idx - self.n;
        if pi_idx >= self.m {
            return Err("query index out of range".to_string());
        }
        if pi_idx < self.z_w_len {
            let v = self.z_w[pi_idx];
            self.acc_pi += c * v;
            self.acc_full += c * v;
            return Ok(());
        }
        let off = pi_idx - self.z_w_len;
        let block = off / self.k_star;
        let pos = off % self.k_star;
        if block >= self.blocks {
            return Err("query block out of range".to_string());
        }
        if block != self.block_id {
            return Err("query touched unexpected block".to_string());
        }
        self.w_terms.push((c, pos));
        Ok(())
    }

    fn finish(self) -> QueryStreamAcc<F> {
        QueryStreamAcc {
            acc_pi: self.acc_pi,
            acc_full: self.acc_full,
            w_terms: self.w_terms,
        }
    }
}

impl<F: PrimeField, P: Dr1csNpFlpcpSparseApi<F>> Theorem43Dpp<F, P> {
    pub fn new(flpcp: P) -> Result<Self, String> {
        let p = field_modulus_u64::<F>().ok_or("field modulus does not fit u64")?;
        if p < 3 || p % 2 == 0 {
            return Err("field must have odd characteristic >= 3".to_string());
        }
        if p > 257 {
            return Err("field too large for tiny-field theorem43 demo".to_string());
        }
        Ok(Self {
            flpcp,
            p,
            q_minus_3: (p as usize).saturating_sub(3),
            _marker: PhantomData,
        })
    }

    /// Length of the produced proof `π` (field elements).
    ///
    /// Layout:
    /// - π0 = FLPCP NP proof (z_w || w) of length `flpcp.m()`
    /// - μ, ν (2 elems)
    /// - u^3..u^{p-1} (p-3 elems)
    pub fn proof_len(&self) -> usize {
        self.flpcp.m() + 2 + self.q_minus_3
    }

    /// Verifier-side evaluation of the Theorem-4.3 check from a split proof `(π0, tail)`.
    ///
    /// Canonical layout:
    /// - `π0 = z_w || w_eval[0] || ... || w_eval[blocks-1]` of length `flpcp.m()`
    /// - `tail = (μ, ν, u^3..u^{p-1})` of length `p-1`
    ///
    /// Returns the field element that should lie in the accepting set (i.e. `1` or `2`)
    /// for a valid proof, as per the protocol’s reduction.
    pub fn answer_from_pi0_and_tail(
        &self,
        art: &Theorem43LockArtifact<F>,
        x: &[F],
        pi0: &[F],
        tail: &[F],
    ) -> Result<F, String> {
        let flpcp = &self.flpcp;
        if x.len() != flpcp.n() {
            return Err("bad public input length".to_string());
        }
        if art.coins.idx >= flpcp.ell() {
            return Err("bad idx coin".to_string());
        }
        let z_w_len = flpcp.z_w_len();
        let k_star = flpcp.k_star();
        let blocks = flpcp.blocks();
        let m0 = flpcp.m();
        if pi0.len() != m0 {
            return Err("bad pi0 length".to_string());
        }
        if pi0.len() < z_w_len {
            return Err("bad pi0 length".to_string());
        }
        if tail.len() != art.coeffs.len() {
            return Err("bad proof tail length".to_string());
        }
        if art.coeffs.len() != (self.p as usize) - 1 {
            return Err("bad Sq coeff length".to_string());
        }
        if tail.len() != 2 + self.q_minus_3 {
            return Err("bad tail length".to_string());
        }

        let z_w = &pi0[..z_w_len];
        let mut scratch = Dr1csQueryScratch::<F>::new(flpcp.n_total());
        scratch.clear_all();
        let ell_local = flpcp.ell_local();
        if ell_local == 0 {
            return Err("ell_local=0".to_string());
        }
        let block_id = art.coins.idx / ell_local;
        if block_id >= blocks {
            return Err("bad coin block_id".to_string());
        }
        let base = x.len() + z_w_len;
        let mut b0 = QueryStreamAccBuilder::<F>::new(x, z_w, z_w_len, k_star, blocks, block_id);
        let mut b1 = QueryStreamAccBuilder::<F>::new(x, z_w, z_w_len, k_star, blocks, block_id);
        let mut b2 = QueryStreamAccBuilder::<F>::new(x, z_w, z_w_len, k_star, blocks, block_id);
        let mut sink_err: Option<String> = None;
        struct Sink<'a, 'b, 'e, F: PrimeField> {
            b0: &'b mut QueryStreamAccBuilder<'a, F>,
            b1: &'b mut QueryStreamAccBuilder<'a, F>,
            b2: &'b mut QueryStreamAccBuilder<'a, F>,
            base: usize,
            err: &'e mut Option<String>,
        }
        impl<'a, 'b, 'e, F: PrimeField> QuerySink<F> for Sink<'a, 'b, 'e, F> {
            fn on_q1(&mut self, coeff: F, idx: usize) {
                if self.err.is_some() {
                    return;
                }
                if let Err(e) = self.b0.add_term(coeff, idx) {
                    *self.err = Some(e);
                }
            }
            fn on_q2(&mut self, coeff: F, idx: usize) {
                if self.err.is_some() {
                    return;
                }
                if let Err(e) = self.b1.add_term(coeff, idx) {
                    *self.err = Some(e);
                }
            }
            fn on_q3(&mut self, coeff: F, idx: usize) {
                if self.err.is_some() {
                    return;
                }
                // IMPORTANT: do NOT store dense `q3` witness (w_eval) terms.
                if idx >= self.base {
                    return;
                }
                if let Err(e) = self.b2.add_term(coeff, idx) {
                    *self.err = Some(e);
                }
            }
        }
        {
            let mut sink = Sink {
                b0: &mut b0,
                b1: &mut b1,
                b2: &mut b2,
                base,
                err: &mut sink_err,
            };
            flpcp.stream_queries_for_coins_sparse(
                art.coins.idx,
                art.coins.lambda,
                x,
                &mut scratch,
                &mut sink,
            )
            .map_err(|e| format!("outer coins->queries failed: {e}"))?;
        }
        if let Some(e) = sink_err {
            return Err(e);
        }
        let mut acc0 = b0.finish();
        let mut acc1 = b1.finish();
        let mut acc2 = b2.finish();

        // Stream w_eval blocks out of π0.
        let q3_block_id = block_id;
        let mut off = z_w_len;
        for b in 0..blocks {
            let end = off + k_star;
            if end > pi0.len() {
                return Err("bad w_eval slice".to_string());
            }
            let w_eval = &pi0[off..end];
            if b == q3_block_id {
                acc0.add_w_eval(w_eval)?;
                acc1.add_w_eval(w_eval)?;
                // q3 witness dot for the selected block (file-backed backends omit dense witness terms).
                let w_eval_u16: Vec<u16> = w_eval.iter().copied().map(f_to_u16).collect();
                let dot = flpcp.dot_q3_w_eval(art.coins.idx, art.coins.lambda, x, &w_eval_u16)?;
                acc2.acc_pi += dot;
                acc2.acc_full += dot;
            }
            off = end;
        }
        if off != pi0.len() {
            return Err("bad pi0 layout".to_string());
        }

        // Verifier check uses the full `(x||π0)` answers.
        let alpha = acc0.acc_full;
        let beta = acc1.acc_full;
        let gamma = acc2.acc_full;

        let coeffs = &art.coeffs;
        let c1 = coeffs[0];
        let c2 = coeffs[1];
        let coeff_alpha = c1 * art.coins.rho;
        let coeff_beta = c1 * art.coins.sigma;
        let coeff_gamma = {
            let two = F::from(2u64);
            c2 * (two * art.coins.rho * art.coins.sigma)
        };
        let coeff_mu = c2 * (art.coins.rho * art.coins.rho);
        let coeff_nu = c2 * (art.coins.sigma * art.coins.sigma);

        // `tail` layout: [μ, ν, u^3..u^{p-1}]
        let mut acc = coeff_alpha * alpha + coeff_beta * beta + coeff_gamma * gamma;
        acc += coeff_mu * tail[0];
        acc += coeff_nu * tail[1];
        for (i, c) in coeffs.iter().copied().enumerate().skip(2) {
            acc += c * tail[i];
        }
        // Public affine shift so accepting set is `{c_hit, c_hit+1}` instead of `{0,1}`.
        Ok(acc + art.coins.c_hit)
    }

    /// Deterministically derive the public coins for a given `(c_stmt, block_id, rep_id)`.
    ///
    /// IMPORTANT: coins are derived from the **package statement commitment** `c_stmt` (not from
    /// a decap-supplied `x`), to prevent retargeting the public coins under statement override.
    pub fn derive_public_coins_from_stmt(
        &self,
        c_stmt: &[F],
        block_id: usize,
        rep_id: u64,
    ) -> Result<Theorem43Coins<F>, String> {
        if block_id >= self.flpcp.blocks() {
            return Err("block_id out of range".to_string());
        }
        let ell_local = self.flpcp.ell_local();
        if ell_local == 0 {
            return Err("ell_local=0".to_string());
        }

        // 1) Derive PUBLIC coins.
        let cfg = f257_poseidon_config();
        let mut sp_coins = PoseidonSponge::<F257>::new(&cfg);
        let ds = vec![F257::from(43u64), F257::from(1u64)]; // theorem43, coins-v1
        sp_coins.absorb(&ds);
        absorb_field_bytes_as_f257::<F>(&mut sp_coins, c_stmt);
        absorb_usize_base257(&mut sp_coins, block_id);
        absorb_u64_base257(&mut sp_coins, rep_id);

        // Squeeze PUBLIC coins with rejection sampling (see comments in `arm`).
        let local_idx = loop {
            let cand = squeeze_usize_mod_base257(&mut sp_coins, ell_local)?;
            if is_f257_field::<F>() {
                const BASE_N: usize = 256;
                const BASE_K: usize = 48;
                const RANK: usize = 3;
                const SIDE: usize = 2 * BASE_K - 1; // 95
                let mut tmp = cand;
                let mut bad = false;
                for _ in 0..RANK {
                    let c = tmp % BASE_N;
                    if c >= BASE_K && c < SIDE {
                        bad = true;
                        break;
                    }
                    tmp /= BASE_N;
                }
                if bad {
                    continue;
                }
            }
            break cand;
        };
        let idx = block_id
            .checked_mul(ell_local)
            .and_then(|v| v.checked_add(local_idx))
            .ok_or("idx overflow")?;
        if idx >= self.flpcp.ell() {
            return Err("derived idx out of range".to_string());
        }

        fn squeeze_f257_as_f_reject_digits<F: PrimeField>(
            sp: &mut PoseidonSponge<F257>,
            reject: &[u16],
        ) -> Result<F, String> {
            loop {
                let x = sp.squeeze_field_elements::<F257>(1)[0];
                let d = f257_digit_u16(x)?;
                if !reject.contains(&d) {
                    return Ok(f257_to_f::<F>(x));
                }
            }
        }
        let lambda = squeeze_f257_as_f_reject_digits::<F>(&mut sp_coins, &[0, 1])?;
        let rho = squeeze_f257_as_f_reject_digits::<F>(&mut sp_coins, &[0])?;
        let sigma = squeeze_f257_as_f_reject_digits::<F>(&mut sp_coins, &[0])?;
        // Derive `c_hit ∈ {1..=127}` to avoid inverse-ratio collisions in `{c,c+1}` decoding.
        let c_hit: F = loop {
            let x = sp_coins.squeeze_field_elements::<F257>(1)[0];
            let d = f257_digit_u16(x)?;
            if (1..=127).contains(&d) {
                break F::from(d as u64);
            }
        };

        Ok(Theorem43Coins {
            idx,
            lambda,
            rho,
            sigma,
            c_hit,
        })
    }

    /// Arm using a fixed FS transcript for the **full-gate cost shape**, while keeping `q` hidden.
    ///
    /// - Public coins are derived from `(domain_sep, C_stmt, block_id, rep_id)`.
    /// - The hidden query randomness (UV bits) is derived from the same transcript **plus**
    ///   an armer-private secret salt `armer_secret` (and explicitly binds the public coins).
    ///
    /// This is the version you’d arithmetize inside the “full gate” to de-risk constraint cost.
    /// The permutation is a **toy cost proxy**, not a standardized hash.
    pub fn arm(
        &self,
        c_stmt: &[F],
        x: &[F],
        armer_secret: &[F],
        block_id: usize,
        rep_id: u64,
    ) -> Result<Theorem43LockArtifact<F>, String> {
        if x.len() != self.flpcp.n() {
            return Err("bad public input length".to_string());
        }
        if block_id >= self.flpcp.blocks() {
            return Err("block_id out of range".to_string());
        }

        // NOTE: this is host-side Fiat–Shamir (arm-before-proof), not the in-circuit DPP relation.
        //
        // Public coins are deterministic from (statement commitment, block_id, rep_id).
        // Hidden UV/Sq randomness is deterministic from (statement commitment, block_id, rep_id, coins, armer_secret).

        let cfg = f257_poseidon_config();
        let coins = self.derive_public_coins_from_stmt(c_stmt, block_id, rep_id)?;

        // 2) Derive HIDDEN UV bits / Sq coefficients, explicitly binding the public coins.
        let mut sp_hidden = PoseidonSponge::<F257>::new(&cfg);
        let ds = vec![F257::from(43u64), F257::from(2u64)]; // theorem43, coeffs-v2
        sp_hidden.absorb(&ds);
        absorb_field_bytes_as_f257::<F>(&mut sp_hidden, c_stmt);
        absorb_usize_base257(&mut sp_hidden, block_id);
        absorb_u64_base257(&mut sp_hidden, rep_id);
        absorb_usize_base257(&mut sp_hidden, coins.idx);
        absorb_field_bytes_as_f257::<F>(&mut sp_hidden, &[coins.lambda, coins.rho, coins.sigma]);
        absorb_field_bytes_as_f257::<F>(&mut sp_hidden, armer_secret);

        // Use base-257 digits as UV bits (0/1).
        //
        // We reject digit 256 so accepted digits are uniform in 0..255, making LSB extraction unbiased.
        let q_minus_1 = (self.p as usize) - 1;
        let q_bits = squeeze_unbiased_bits_from_f257_digits(&mut sp_hidden, q_minus_1)?;
        let coeffs = self.sq_coeffs_from_uv_bits(&q_bits)?;
        let len = x.len() + self.proof_len();

        Ok(Theorem43LockArtifact {
            c_stmt: c_stmt.to_vec(),
            accepting_set: [coins.c_hit, coins.c_hit + F::ONE],
            len,
            coins,
            stats: Theorem43ArmingStats {
                fs_permutes: 0,
                q_nnz: 0,
            },
            coeffs,
        })
    }

    #[inline]
    pub fn accept_answer(&self, a: &F) -> bool {
        *a == F::ONE || *a == F::from(2u64)
    }

    fn sq_coeffs_from_uv_bits(&self, q_bits: &[u8]) -> Result<Vec<F>, String> {
        let q_minus_1 = (self.p as usize) - 1;
        if q_bits.len() != q_minus_1 {
            return Err("bad uv bit length".to_string());
        }

        let mut coeffs = vec![F::ZERO; q_minus_1];
        for i in 1..=q_minus_1 {
            let mut acc = F::ZERO;
            for lam_u in 1..self.p {
                let bit = q_bits[(lam_u as usize) - 1];
                if bit == 0 {
                    continue;
                }
                let lam = F::from(lam_u);
                acc += lam.pow([i as u64]);
            }
            coeffs[i - 1] = -acc;
        }

        if coeffs.len() < 2 {
            return Err("field too small for Sq coefficients".to_string());
        }
        Ok(coeffs)
    }
}

impl<F: PrimeField, P: Dr1csNpFlpcpSparseApi<F> + Sync> Theorem43Dpp<F, P> {
    fn stream_pi0_and_collect_abg_accs(
        &self,
        x: &[F],
        z_w: &[F],
        coins_list: &[Theorem43Coins<F>],
        mut on_pi0_chunk: Option<&mut dyn FnMut(&[F])>,
    ) -> Result<Vec<Theorem43AbgAcc<F>>, String> {
        let flpcp = &self.flpcp;
        if x.len() != flpcp.n() {
            return Err("bad public input length".to_string());
        }
        let z_w_len = flpcp.z_w_len();
        if z_w.len() != z_w_len {
            return Err("bad witness length".to_string());
        }
        for coins in coins_list {
            if coins.idx >= flpcp.ell() {
                return Err("bad idx coin".to_string());
            }
        }

        let k_star = flpcp.k_star();
        let blocks = flpcp.blocks();
        let ell_local = flpcp.ell_local();
        if blocks == 0 || ell_local == 0 {
            return Err("invalid (blocks,ell_local)".to_string());
        }
        let base = x.len() + z_w_len;

        // Precompute per-coin query accumulators.
        struct AccSet<F: PrimeField> {
            coins: Theorem43Coins<F>,
            block_id: usize,
            acc0: QueryStreamAcc<F>,
            acc1: QueryStreamAcc<F>,
            acc2: QueryStreamAcc<F>,
        }
        let mut accs: Vec<AccSet<F>> = coins_list
            .par_iter()
            .map_init(
                || Dr1csQueryScratch::<F>::new(flpcp.n_total()),
                |scratch, coins| -> Result<AccSet<F>, String> {
                    scratch.clear_all();
                    let block_id = coins.idx / ell_local;
                    if block_id >= blocks {
                        return Err("bad coin block_id".to_string());
                    }
                    let mut b0 =
                        QueryStreamAccBuilder::<F>::new(x, z_w, z_w_len, k_star, blocks, block_id);
                    let mut b1 =
                        QueryStreamAccBuilder::<F>::new(x, z_w, z_w_len, k_star, blocks, block_id);
                    let mut b2 =
                        QueryStreamAccBuilder::<F>::new(x, z_w, z_w_len, k_star, blocks, block_id);
                    let mut sink_err: Option<String> = None;
                    struct Sink<'a, 'b, 'e, F: PrimeField> {
                        b0: &'b mut QueryStreamAccBuilder<'a, F>,
                        b1: &'b mut QueryStreamAccBuilder<'a, F>,
                        b2: &'b mut QueryStreamAccBuilder<'a, F>,
                        base: usize,
                        err: &'e mut Option<String>,
                    }
                    impl<'a, 'b, 'e, F: PrimeField> QuerySink<F> for Sink<'a, 'b, 'e, F> {
                        fn on_q1(&mut self, coeff: F, idx: usize) {
                            if self.err.is_some() {
                                return;
                            }
                            if let Err(e) = self.b0.add_term(coeff, idx) {
                                *self.err = Some(e);
                            }
                        }
                        fn on_q2(&mut self, coeff: F, idx: usize) {
                            if self.err.is_some() {
                                return;
                            }
                            if let Err(e) = self.b1.add_term(coeff, idx) {
                                *self.err = Some(e);
                            }
                        }
                        fn on_q3(&mut self, coeff: F, idx: usize) {
                            if self.err.is_some() {
                                return;
                            }
                            // IMPORTANT: do NOT store dense `q3` witness (w_eval) terms.
                            if idx >= self.base {
                                return;
                            }
                            if let Err(e) = self.b2.add_term(coeff, idx) {
                                *self.err = Some(e);
                            }
                        }
                    }
                    {
                        let mut sink = Sink {
                            b0: &mut b0,
                            b1: &mut b1,
                            b2: &mut b2,
                            base,
                            err: &mut sink_err,
                        };
                        flpcp
                            .stream_queries_for_coins_sparse(coins.idx, coins.lambda, x, scratch, &mut sink)
                            .map_err(|e| format!("outer coins->queries failed: {e}"))?;
                    }
                    if let Some(e) = sink_err {
                        return Err(e);
                    }
                    Ok(AccSet {
                        coins: coins.clone(),
                        block_id,
                        acc0: b0.finish(),
                        acc1: b1.finish(),
                        acc2: b2.finish(),
                    })
                },
            )
            .collect::<Result<Vec<_>, String>>()?;

        // Block-grouped schedule: each coin updates exactly its selected block.
        let avg_per_block = (accs.len().saturating_add(blocks).saturating_sub(1)) / blocks.max(1);
        let mut bucket: Vec<Vec<usize>> = (0..blocks)
            .map(|_| Vec::with_capacity(avg_per_block.max(1)))
            .collect();
        for (ci, a) in accs.iter().enumerate() {
            bucket[a.block_id].push(ci);
        }

        if let Some(cb) = on_pi0_chunk.as_deref_mut() {
            cb(z_w);
        }
        let emit_pi0 = on_pi0_chunk.is_some();

        let witness_pos = flpcp.witness_positions_star()?;
        if witness_pos.len() != k_star {
            return Err("witness positions length mismatch".to_string());
        }

        let err: Option<String> = None;

        struct BlockDots<F: PrimeField> {
            q1_w: Vec<F>,
            q2_w: Vec<F>,
            q3_w: Vec<F>,
        }
        let per_block: Vec<Option<std::sync::OnceLock<BlockDots<F>>>> = (0..blocks)
            .map(|b| {
                let n = bucket.get(b).map(|v| v.len()).unwrap_or(0);
                if n == 0 {
                    None
                } else {
                    Some(std::sync::OnceLock::new())
                }
            })
            .collect();

        let accs_ro: &[AccSet<F>] = accs.as_slice();
        let lut: Option<Vec<F>> = if is_f257_field::<F>() {
            Some((0u16..=256u16).map(|d| F::from(d as u64)).collect())
        } else {
            None
        };

        let on_block_hook = |b: usize, w_eval_u16: &[u16]| -> Result<(), String> {
            if b >= bucket.len() {
                return Err("stream_w_eval_blocks_with_hook: block id out of range".to_string());
            }
            let coins_b = &bucket[b];
            if coins_b.is_empty() {
                return Ok(());
            }
            if w_eval_u16.len() != k_star {
                return Err("stream_w_eval_blocks_with_hook: bad w_eval_u16 length".to_string());
            }
            let cell = per_block
                .get(b)
                .ok_or_else(|| "stream_w_eval_blocks_with_hook: per_block out of range".to_string())?
                .as_ref()
                .ok_or_else(|| "stream_w_eval_blocks_with_hook: missing per-block dots buffer".to_string())?;
            if cell.get().is_some() {
                return Err("stream_w_eval_blocks_with_hook: duplicate block hook invocation".to_string());
            }
            let n = coins_b.len();
            let mut dots = BlockDots::<F> {
                q1_w: vec![F::ZERO; n],
                q2_w: vec![F::ZERO; n],
                q3_w: vec![F::ZERO; n],
            };

            for (j, &ci) in coins_b.iter().enumerate() {
                let a = accs_ro
                    .get(ci)
                    .ok_or_else(|| "stream_w_eval_blocks_with_hook: coin index out of range".to_string())?;

                let mut s1 = F::ZERO;
                for (c, pos) in a.acc0.w_terms.iter().copied() {
                    if pos >= w_eval_u16.len() {
                        return Err("stream_w_eval_blocks_with_hook: q1 w_eval_u16 index out of range".to_string());
                    }
                    if let Some(lut) = lut.as_ref() {
                        s1 += c * lut[w_eval_u16[pos] as usize];
                    } else {
                        return Err("stream_w_eval_blocks_with_hook: q1 requires F257 u16 pipeline".to_string());
                    }
                }
                dots.q1_w[j] = s1;

                let mut s2 = F::ZERO;
                for (c, pos) in a.acc1.w_terms.iter().copied() {
                    if pos >= w_eval_u16.len() {
                        return Err("stream_w_eval_blocks_with_hook: q2 w_eval_u16 index out of range".to_string());
                    }
                    if let Some(lut) = lut.as_ref() {
                        s2 += c * lut[w_eval_u16[pos] as usize];
                    } else {
                        return Err("stream_w_eval_blocks_with_hook: q2 requires F257 u16 pipeline".to_string());
                    }
                }
                dots.q2_w[j] = s2;
            }

            // q3 witness dot: heavy dense part, batched.
            const MAX_BATCH: usize = 64;
            let mut idxs = [0usize; MAX_BATCH];
            let mut lambdas = [F::ZERO; MAX_BATCH];
            let mut out = [F::ZERO; MAX_BATCH];

            let mut off = 0usize;
            while off < coins_b.len() {
                let end = (off + MAX_BATCH).min(coins_b.len());
                let n = end - off;
                for j in 0..n {
                    let a = accs_ro
                        .get(coins_b[off + j])
                        .ok_or_else(|| "stream_w_eval_blocks_with_hook: coin index out of range (q3)".to_string())?;
                    idxs[j] = a.coins.idx;
                    lambdas[j] = a.coins.lambda;
                }
                // Backend expects `w_eval_u16` for this block; structured backends override this
                // to amortize dense q3 dot products.
                flpcp.dot_q3_w_eval_many(&idxs[..n], &lambdas[..n], x, w_eval_u16, &mut out[..n])?;
                for j in 0..n {
                    dots.q3_w[off + j] = out[j];
                }
                off = end;
            }

            cell.set(dots)
                .map_err(|_| "stream_w_eval_blocks_with_hook: once-lock already set".to_string())?;
            Ok(())
        };

        let mut on_block = |_: usize, w_eval: &[F]| {
            if let Some(cb) = on_pi0_chunk.as_deref_mut() {
                cb(w_eval);
            }
        };
        flpcp
            .stream_w_eval_blocks(
                &witness_pos,
                x,
                z_w,
                None,
                None,
                Some(&on_block_hook),
                if emit_pi0 { Some(&mut on_block) } else { None },
            )
            .map_err(|e| format!("stream_w_eval_blocks failed: {e}"))?;

        if let Some(e) = err {
            return Err(e);
        }

        // Apply per-block witness dots.
        for b in 0..blocks {
            let coins_b = &bucket[b];
            if coins_b.is_empty() {
                continue;
            }
            let cell = per_block
                .get(b)
                .and_then(|c| c.as_ref())
                .ok_or_else(|| "stream_w_eval_blocks: missing per-block dots buffer (apply)".to_string())?;
            let dots = cell
                .get()
                .ok_or_else(|| "stream_w_eval_blocks: per-block dots not set (apply)".to_string())?;
            if dots.q1_w.len() != coins_b.len()
                || dots.q2_w.len() != coins_b.len()
                || dots.q3_w.len() != coins_b.len()
            {
                return Err("stream_w_eval_blocks: per-block dots length mismatch (apply)".to_string());
            }
            for (j, &ci) in coins_b.iter().enumerate() {
                let a = accs
                    .get_mut(ci)
                    .ok_or_else(|| "stream_w_eval_blocks: bucket coin index out of range (apply)".to_string())?;
                let t1 = dots.q1_w[j];
                a.acc0.acc_pi += t1;
                a.acc0.acc_full += t1;
                let t2 = dots.q2_w[j];
                a.acc1.acc_pi += t2;
                a.acc1.acc_full += t2;
                let t3 = dots.q3_w[j];
                a.acc2.acc_pi += t3;
                a.acc2.acc_full += t3;
            }
        }

        let mut out = Vec::with_capacity(accs.len());
        for a in accs.into_iter() {
            out.push(Theorem43AbgAcc {
                coins: a.coins,
                alpha_pi: a.acc0.acc_pi,
                beta_pi: a.acc1.acc_pi,
                gamma_pi: a.acc2.acc_pi,
                alpha_full: a.acc0.acc_full,
                beta_full: a.acc1.acc_full,
                gamma_full: a.acc2.acc_full,
            });
        }
        Ok(out)
    }

    /// Stream `π0` once and return full `(alpha,beta,gamma)` for many coins.
    ///
    /// This is the same streaming schedule as `stream_pi0_and_collect_tails`, but it **does not**
    /// generate or stream tails. It exists for the WE "poison" backend, which needs the MulEq
    /// residual `err = gamma - alpha*beta` across blocks.
    pub fn stream_pi0_and_collect_abg_full(
        &self,
        x: &[F],
        z_w: &[F],
        coins_list: &[Theorem43Coins<F>],
        on_pi0_chunk: Option<&mut dyn FnMut(&[F])>,
    ) -> Result<Vec<Theorem43AbgFull<F>>, String> {
        let accs = self.stream_pi0_and_collect_abg_accs(x, z_w, coins_list, on_pi0_chunk)?;
        Ok(accs
            .into_iter()
            .map(|a| Theorem43AbgFull {
                alpha: a.alpha_full,
                beta: a.beta_full,
                gamma: a.gamma_full,
            })
            .collect())
    }

    /// Stream the coin-independent prefix `π0` once, and return coin-dependent tails for many coins.
    ///
    /// Proof layout is `π = (π0 || tail)` where:
    /// - `π0 = z_w || w_eval[0] || ... || w_eval[blocks-1]` depends only on `(x, z_w)`.
    /// - `tail = (μ, ν, u^3..u^{p-1})` depends on `coins` (via the hidden sparse query and (ρ,σ)).
    ///
    /// This is the asymptotically optimal way to decapsulate many coin instances against the
    /// same witness: stream `π0` once (to all decaps), then absorb each small tail.
    pub fn stream_pi0_and_collect_tails(
        &self,
        x: &[F],
        z_w: &[F],
        coins_list: &[Theorem43Coins<F>],
        on_pi0_chunk: Option<&mut dyn FnMut(&[F])>,
        on_tail_elem: &mut dyn FnMut(usize, usize, &F),
    ) -> Result<Vec<Theorem43AbgTail<F>>, String> {
        let accs = self.stream_pi0_and_collect_abg_accs(x, z_w, coins_list, on_pi0_chunk)?;
        let mut out = Vec::with_capacity(accs.len());
        for (ci, a) in accs.into_iter().enumerate() {
            let mu = a.alpha_full * a.alpha_full;
            let nu = a.beta_full * a.beta_full;
            let u = a.coins.rho * a.alpha_full + a.coins.sigma * a.beta_full;
            on_tail_elem(ci, 0, &mu);
            on_tail_elem(ci, 1, &nu);
            if self.q_minus_3 > 0 {
                let mut cur = (u * u) * u; // u^3
                for t in 0..self.q_minus_3 {
                    on_tail_elem(ci, 2 + t, &cur);
                    cur *= u;
                }
            }
            out.push(Theorem43AbgTail {
                alpha: a.alpha_pi,
                beta: a.beta_pi,
                gamma: a.gamma_pi,
            });
        }
        Ok(out)
    }

    pub fn query_scratch(&self) -> Dr1csQueryScratch<F> {
        Dr1csQueryScratch::new(self.flpcp.n_total())
    }

    pub fn stream_query_terms_for_pi(
        &self,
        x: &[F],
        coins: &Theorem43Coins<F>,
        coeffs: &[F],
        scratch: &mut Dr1csQueryScratch<F>,
        on_pi_term: &mut dyn FnMut(usize, F),
    ) -> Result<F, String> {
        let flpcp = &self.flpcp;
        if x.len() != flpcp.n() {
            return Err("bad public input length".to_string());
        }
        if coins.idx >= flpcp.ell() {
            return Err("bad idx coin".to_string());
        }
        if coeffs.len() != (self.p as usize) - 1 {
            return Err("bad Sq coeff length".to_string());
        }

        let c1 = coeffs[0];
        let c2 = coeffs[1];
        let coeff_alpha = c1 * coins.rho;
        let coeff_beta = c1 * coins.sigma;
        let coeff_gamma = {
            let two = F::from(2u64);
            c2 * (two * coins.rho * coins.sigma)
        };
        let coeff_mu = c2 * (coins.rho * coins.rho);
        let coeff_nu = c2 * (coins.sigma * coins.sigma);

        struct PiQuerySink<'a, F: PrimeField> {
            x: &'a [F],
            x_len: usize,
            offset: F,
            coeff_alpha: F,
            coeff_beta: F,
            coeff_gamma: F,
            on_pi_term: &'a mut dyn FnMut(usize, F),
        }
        impl<'a, F: PrimeField> QuerySink<F> for PiQuerySink<'a, F> {
            fn on_q1(&mut self, coeff: F, idx: usize) {
                let coeff = coeff * self.coeff_alpha;
                if coeff.is_zero() {
                    return;
                }
                if idx < self.x_len {
                    self.offset += coeff * self.x[idx];
                } else {
                    (self.on_pi_term)(idx - self.x_len, coeff);
                }
            }
            fn on_q2(&mut self, coeff: F, idx: usize) {
                let coeff = coeff * self.coeff_beta;
                if coeff.is_zero() {
                    return;
                }
                if idx < self.x_len {
                    self.offset += coeff * self.x[idx];
                } else {
                    (self.on_pi_term)(idx - self.x_len, coeff);
                }
            }
            fn on_q3(&mut self, coeff: F, idx: usize) {
                let coeff = coeff * self.coeff_gamma;
                if coeff.is_zero() {
                    return;
                }
                if idx < self.x_len {
                    self.offset += coeff * self.x[idx];
                } else {
                    (self.on_pi_term)(idx - self.x_len, coeff);
                }
            }
        }
        let m0 = flpcp.m();
        let x_len = x.len();
        let offset = {
            let mut sink = PiQuerySink {
                x,
                x_len,
                offset: coins.c_hit,
                coeff_alpha,
                coeff_beta,
                coeff_gamma,
                on_pi_term,
            };
            flpcp.stream_queries_for_coins_sparse(coins.idx, coins.lambda, x, scratch, &mut sink)?;
            sink.offset
        };

        on_pi_term(m0 + 0, coeff_mu);
        on_pi_term(m0 + 1, coeff_nu);
        for (t, c) in coeffs.iter().copied().skip(2).enumerate() {
            if c.is_zero() {
                continue;
            }
            on_pi_term(m0 + 2 + t, c);
        }

        Ok(offset)
    }

    /// Stream the combined x-query terms (the coefficients that multiply the public prefix `x`)
    /// for a given coin instance and Sq coefficients.
    ///
    /// This is used by the lock layer to publish masked x-coefficients `h_x = s*q_x`, so decap
    /// can compute the statement-dependent contribution inside the masked dot product.
    ///
    /// Notes:
    /// - This emits only the x-side terms (indices `< x_len`) after applying the linearization
    ///   weights `(coeff_alpha, coeff_beta, coeff_gamma)`.
    /// - It does **not** include the protocol constant `+1` (the caller can treat it as an
    ///   additional coefficient on `x[0]=1` if desired).
    pub fn stream_query_terms_for_x(
        &self,
        x: &[F],
        coins: &Theorem43Coins<F>,
        coeffs: &[F],
        scratch: &mut Dr1csQueryScratch<F>,
        on_x_term: &mut dyn FnMut(usize, F),
    ) -> Result<(), String> {
        let flpcp = &self.flpcp;
        if x.len() != flpcp.n() {
            return Err("bad public input length".to_string());
        }
        if coins.idx >= flpcp.ell() {
            return Err("bad idx coin".to_string());
        }
        if coeffs.len() != (self.p as usize) - 1 {
            return Err("bad Sq coeff length".to_string());
        }

        let c1 = coeffs[0];
        let c2 = coeffs[1];
        let coeff_alpha = c1 * coins.rho;
        let coeff_beta = c1 * coins.sigma;
        let coeff_gamma = {
            let two = F::from(2u64);
            c2 * (two * coins.rho * coins.sigma)
        };

        struct XQuerySink<'a, F: PrimeField> {
            x_len: usize,
            coeff_alpha: F,
            coeff_beta: F,
            coeff_gamma: F,
            on_x_term: &'a mut dyn FnMut(usize, F),
        }
        impl<'a, F: PrimeField> QuerySink<F> for XQuerySink<'a, F> {
            fn on_q1(&mut self, coeff: F, idx: usize) {
                let coeff = coeff * self.coeff_alpha;
                if coeff.is_zero() {
                    return;
                }
                if idx < self.x_len {
                    (self.on_x_term)(idx, coeff);
                }
            }
            fn on_q2(&mut self, coeff: F, idx: usize) {
                let coeff = coeff * self.coeff_beta;
                if coeff.is_zero() {
                    return;
                }
                if idx < self.x_len {
                    (self.on_x_term)(idx, coeff);
                }
            }
            fn on_q3(&mut self, coeff: F, idx: usize) {
                let coeff = coeff * self.coeff_gamma;
                if coeff.is_zero() {
                    return;
                }
                if idx < self.x_len {
                    (self.on_x_term)(idx, coeff);
                }
            }
        }

        let x_len = x.len();
        let mut sink = XQuerySink {
            x_len,
            coeff_alpha,
            coeff_beta,
            coeff_gamma,
            on_x_term,
        };
        flpcp.stream_queries_for_coins_sparse(coins.idx, coins.lambda, x, scratch, &mut sink)?;
        Ok(())
    }
}

fn field_modulus_u64<F: PrimeField>() -> Option<u64> {
    let bytes = F::MODULUS.to_bytes_le();
    if bytes.len() > 8 {
        return None;
    }
    let mut buf = [0u8; 8];
    buf[..bytes.len()].copy_from_slice(&bytes);
    Some(u64::from_le_bytes(buf))
}

fn f257_digit_u16(x: F257) -> Result<u16, String> {
    let bytes = x.into_bigint().to_bytes_le();
    let mut acc: u16 = 0;
    if !bytes.is_empty() {
        acc |= bytes[0] as u16;
    }
    if bytes.len() > 1 {
        acc |= (bytes[1] as u16) << 8;
    }
    if acc >= 257 {
        return Err("expected F257 digit in 0..=256".to_string());
    }
    Ok(acc)
}

fn squeeze_usize_mod_base257(sp: &mut PoseidonSponge<F257>, modulus: usize) -> Result<usize, String> {
    if modulus == 0 {
        return Err("squeeze_usize_mod_base257: modulus=0".to_string());
    }
    if modulus == 1 {
        return Ok(0);
    }
    let m = modulus as u128;

    // Choose d such that 257^d >= m, working in u128.
    let mut pow: u128 = 1;
    let mut d: usize = 0;
    while pow < m {
        if pow > (u128::MAX / 257) {
            return Err("squeeze_usize_mod_base257: modulus too large for u128 base-257 sampling".to_string());
        }
        pow *= 257;
        d += 1;
    }
    if d == 0 {
        d = 1;
    }
    let limit = (pow / m) * m; // largest multiple of m below pow

    loop {
        let digits = sp.squeeze_field_elements::<F257>(d);
        let mut r: u128 = 0;
        let mut mul: u128 = 1;
        for dig in digits {
            let du = f257_digit_u16(dig)? as u128;
            r = r
                .checked_add(du.checked_mul(mul).ok_or("overflow in base-257 accumulator")?)
                .ok_or("overflow in base-257 accumulator")?;
            mul = mul.checked_mul(257).ok_or("overflow in base-257 accumulator")?;
        }
        debug_assert!(r < pow);
        if r < limit {
            return Ok((r % m) as usize);
        }
    }
}

fn squeeze_unbiased_bits_from_f257_digits(
    sp: &mut PoseidonSponge<F257>,
    nbits: usize,
) -> Result<Vec<u8>, String> {
    let mut out = Vec::with_capacity(nbits);
    while out.len() < nbits {
        // Batch a little to amortize sponge calls.
        let need = (nbits - out.len()).min(64);
        let digits = sp.squeeze_field_elements::<F257>(need);
        for dig in digits {
            let du = f257_digit_u16(dig)?;
            if du == 256 {
                continue; // rejection to get uniform 0..255
            }
            out.push((du & 1) as u8);
            if out.len() == nbits {
                break;
            }
        }
    }
    Ok(out)
}

fn f257_to_f<F: PrimeField>(x: F257) -> F {
    let bytes = x.into_bigint().to_bytes_le();
    F::from_le_bytes_mod_order(&bytes)
}

fn absorb_field_bytes_as_f257<F: PrimeField>(sp: &mut PoseidonSponge<F257>, elems: &[F]) {
    let mut bytes = Vec::new();
    for e in elems {
        bytes.extend_from_slice(&field_to_bytes_le_fixed::<F>(e));
    }
    let digits = bytes
        .iter()
        .map(|b| F257::from(*b as u64))
        .collect::<Vec<_>>();
    sp.absorb(&digits);
}

fn absorb_usize_base257(sp: &mut PoseidonSponge<F257>, mut x: usize) {
    // Absorb little-endian base-257 digits. Ensure at least one digit is absorbed.
    if x == 0 {
        let digits = vec![F257::from(0u64)];
        sp.absorb(&digits);
        return;
    }
    let mut digits = Vec::new();
    while x > 0 {
        let d = (x % 257) as u64;
        digits.push(F257::from(d));
        x /= 257;
    }
    sp.absorb(&digits);
}

fn absorb_u64_base257(sp: &mut PoseidonSponge<F257>, mut x: u64) {
    if x == 0 {
        let digits = vec![F257::from(0u64)];
        sp.absorb(&digits);
        return;
    }
    let mut digits = Vec::new();
    while x > 0 {
        let d = (x % 257) as u64;
        digits.push(F257::from(d));
        x /= 257;
    }
    sp.absorb(&digits);
}

