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
use std::marker::PhantomData;

use latticefold::transcript::bytes::field_to_bytes_le_fixed;
use latticefold::transcript::poseidon::{f257_poseidon_config, F257};

use crate::dr1cs_flpcp::{
    f_to_u16, is_f257_field, Dr1csNpFlpcpSparseApi, Dr1csQueryScratch, QuerySink,
};

/// Coins defining the single lockable query.
#[derive(Clone, Debug)]
pub struct Theorem43Coins<F: PrimeField> {
    pub idx: usize,
    pub lambda: F,
    pub rho: F,
    pub sigma: F,
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

/// Streaming query accumulator that tracks both:
/// - `acc_pi`: contribution from `π0` only (z_w + w_eval blocks)
/// - `acc_full`: contribution from `(x || π0)`
///
/// This is used to support statement-bound lock equations without double-counting `x`.
struct QueryStreamAcc<F: PrimeField> {
    acc_pi: F,
    acc_full: F,
    // Sparse per-block terms over the `w_eval[block]` slices in π0.
    //
    // In production regimes, each query typically touches very few blocks (often 1), so keeping
    // this sparse avoids `O(blocks)` per-coin allocations and supports block-grouped streaming.
    block_terms: Vec<(usize, Vec<(F, usize)>)>,
}

impl<F: PrimeField> QueryStreamAcc<F> {
    fn add_block(&mut self, block_id: usize, w_eval: &[F]) -> Result<(), String> {
        // Sparse: if we don't touch this block, nothing to do.
        let Some((_b, terms)) = self.block_terms.iter().find(|(b, _)| *b == block_id) else {
            return Ok(());
        };
        for (c, pos) in terms {
            if *pos >= w_eval.len() {
                return Err("w_eval index out of range".to_string());
            }
            let t = *c * w_eval[*pos];
            self.acc_pi += t;
            self.acc_full += t;
        }
        Ok(())
    }

    #[inline]
    fn touched_blocks(&self) -> impl Iterator<Item = usize> + '_ {
        self.block_terms.iter().map(|(b, _)| *b)
    }
}

/// Helper to build `QueryStreamAcc` without materializing query vectors.
struct QueryStreamAccBuilder<'a, F: PrimeField> {
    x: &'a [F],
    z_w: &'a [F],
    z_w_len: usize,
    k_star: usize,
    blocks: usize,
    n: usize,
    m: usize,
    acc_pi: F,
    acc_full: F,
    block_terms: Vec<(usize, Vec<(F, usize)>)>,
}

impl<'a, F: PrimeField> QueryStreamAccBuilder<'a, F> {
    fn new(x: &'a [F], z_w: &'a [F], z_w_len: usize, k_star: usize, blocks: usize) -> Self {
        let n = x.len();
        let m = z_w_len + (k_star * blocks);
        Self {
            x,
            z_w,
            z_w_len,
            k_star,
            blocks,
            n,
            m,
            acc_pi: F::ZERO,
            acc_full: F::ZERO,
            block_terms: Vec::new(),
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
        if let Some((_b, terms)) = self.block_terms.iter_mut().find(|(b, _)| *b == block) {
            terms.push((c, pos));
        } else {
            self.block_terms.push((block, vec![(c, pos)]));
        }
        Ok(())
    }

    fn finish(self) -> QueryStreamAcc<F> {
        QueryStreamAcc {
            acc_pi: self.acc_pi,
            acc_full: self.acc_full,
            block_terms: self.block_terms,
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
        let mut b0 = QueryStreamAccBuilder::<F>::new(x, z_w, z_w_len, k_star, blocks);
        let mut b1 = QueryStreamAccBuilder::<F>::new(x, z_w, z_w_len, k_star, blocks);
        let mut b2 = QueryStreamAccBuilder::<F>::new(x, z_w, z_w_len, k_star, blocks);
        let mut sink_err: Option<String> = None;
        struct Sink<'a, 'b, 'e, F: PrimeField> {
            b0: &'b mut QueryStreamAccBuilder<'a, F>,
            b1: &'b mut QueryStreamAccBuilder<'a, F>,
            b2: &'b mut QueryStreamAccBuilder<'a, F>,
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
        let ell_local = flpcp.ell_local();
        if ell_local == 0 {
            return Err("ell_local=0".to_string());
        }
        let q3_block_id = art.coins.idx / ell_local;
        let mut off = z_w_len;
        for b in 0..blocks {
            let end = off + k_star;
            if end > pi0.len() {
                return Err("bad w_eval slice".to_string());
            }
            let w_eval = &pi0[off..end];
            acc0.add_block(b, w_eval)?;
            acc1.add_block(b, w_eval)?;
            // q3 witness dot for the selected block (file-backed backends omit dense witness terms).
            if b == q3_block_id {
                let dot = flpcp.dot_q3_w_eval(art.coins.idx, art.coins.lambda, x, &mut scratch, w_eval)?;
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
        Ok(acc + F::ONE)
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

        Ok(Theorem43Coins { idx, lambda, rho, sigma })
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
            accepting_set: [F::ONE, F::from(2u64)],
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
        on_pi0_chunk: &mut dyn FnMut(&[F]),
        on_tail_elem: &mut dyn FnMut(usize, usize, &F),
    ) -> Result<Vec<Theorem43AbgTail<F>>, String> {
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
        let mut accs: Vec<AccSet<F>> = Vec::with_capacity(coins_list.len());
        let mut scratch = Dr1csQueryScratch::<F>::new(flpcp.n_total());
        for coins in coins_list {
            scratch.clear_all();
            let block_id = coins.idx / ell_local;
            if block_id >= blocks {
                return Err("bad coin block_id".to_string());
            }
            let mut b0 = QueryStreamAccBuilder::<F>::new(x, z_w, z_w_len, k_star, blocks);
            let mut b1 = QueryStreamAccBuilder::<F>::new(x, z_w, z_w_len, k_star, blocks);
            let mut b2 = QueryStreamAccBuilder::<F>::new(x, z_w, z_w_len, k_star, blocks);
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
                    // We compute their dot product on-the-fly during `stream_w_eval_blocks` via
                    // `flpcp.dot_q3_w_eval`, to avoid massive per-coin allocations.
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
                    .stream_queries_for_coins_sparse(coins.idx, coins.lambda, x, &mut scratch, &mut sink)
                    .map_err(|e| format!("outer coins->queries failed: {e}"))?;
            }
            if let Some(e) = sink_err {
                return Err(e);
            }
            let acc0 = b0.finish();
            let acc1 = b1.finish();
            let acc2 = b2.finish();
            accs.push(AccSet {
                coins: coins.clone(),
                block_id,
                acc0,
                acc1,
                acc2,
            });
        }

        // Block-grouped schedule: each coin updates exactly its selected block.
        let mut bucket: Vec<Vec<usize>> = vec![Vec::new(); blocks];
        for (ci, a) in accs.iter().enumerate() {
            bucket[a.block_id].push(ci);
        }

        on_pi0_chunk(z_w);

        let witness_pos = flpcp.witness_positions_star()?;
        if witness_pos.len() != k_star {
            return Err("witness positions length mismatch".to_string());
        }

        let (x_u16, z_u16) = if is_f257_field::<F>() {
            (
                Some(x.iter().copied().map(f_to_u16).collect::<Vec<_>>()),
                Some(z_w.iter().copied().map(f_to_u16).collect::<Vec<_>>()),
            )
        } else {
            (None, None)
        };

        let mut err: Option<String> = None;
        flpcp.stream_w_eval_blocks(
            &witness_pos,
            x,
            z_w,
            x_u16.as_deref(),
            z_u16.as_deref(),
            &mut |b, w_eval| {
                if err.is_some() {
                    return;
                }
                if b >= bucket.len() {
                    err = Some("stream_w_eval_blocks: block id out of range".to_string());
                    return;
                }
                for &ci in bucket[b].iter() {
                    let a = match accs.get_mut(ci) {
                        Some(v) => v,
                        None => {
                            err = Some("stream_w_eval_blocks: bucket coin index out of range".to_string());
                            return;
                        }
                    };
                    if let Err(e) = a.acc0.add_block(b, w_eval) {
                        err = Some(e);
                        return;
                    }
                    if let Err(e) = a.acc1.add_block(b, w_eval) {
                        err = Some(e);
                        return;
                    }
                    // q3 witness dot: compute without materializing dense q3 witness terms.
                    match flpcp.dot_q3_w_eval(a.coins.idx, a.coins.lambda, x, &mut scratch, w_eval) {
                        Ok(dot) => {
                            a.acc2.acc_pi += dot;
                            a.acc2.acc_full += dot;
                        }
                        Err(e) => {
                            err = Some(e);
                            return;
                        }
                    }
                }
                on_pi0_chunk(w_eval);
            },
        )?;
        if let Some(e) = err {
            return Err(e);
        }

        let mut out = Vec::with_capacity(accs.len());
        for (ci, a) in accs.into_iter().enumerate() {
            // For lock equations we expose only the π-only contributions (exclude x terms),
            // to avoid double-counting when the lock separately accounts for `⟨h_x, x⟩`.
            let alpha_pi = a.acc0.acc_pi;
            let beta_pi = a.acc1.acc_pi;
            let gamma_pi = a.acc2.acc_pi;
            // Tail generation must follow the theorem43 transcript relation, which depends on
            // the full `(x||π0)` query answers.
            let alpha_full = a.acc0.acc_full;
            let beta_full = a.acc1.acc_full;

            let mu = alpha_full * alpha_full;
            let nu = beta_full * beta_full;
            let u = a.coins.rho * alpha_full + a.coins.sigma * beta_full;
            // Tail layout: [mu, nu, u^3..u^{p-1}]
            on_tail_elem(ci, 0, &mu);
            on_tail_elem(ci, 1, &nu);
            if self.q_minus_3 > 0 {
                let mut cur = (u * u) * u; // u^3
                for t in 0..self.q_minus_3 {
                    on_tail_elem(ci, 2 + t, &cur);
                    cur *= u;
                }
            }
            // NOTE: `alpha/beta/gamma` returned here are π-only.
            out.push(Theorem43AbgTail {
                alpha: alpha_pi,
                beta: beta_pi,
                gamma: gamma_pi,
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
                offset: F::ONE,
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

