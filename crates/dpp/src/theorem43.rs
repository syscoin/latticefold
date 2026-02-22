//! Tiny-field Theorem-4.3 utilities (F257) for the WE-gate pipeline.
//!
//! Canonical surface area here:
//! - deterministic public-coin derivation from a statement commitment
//! - streaming accumulation of `(alpha,beta,gamma)` over `(x || pi0)` for many coins, used by the
//!   tail-free "MulEq residual" backend (`err = gamma - alpha*beta`)
//!
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
use std::time::Instant;

use latticefold::transcript::bytes::field_to_bytes_le_fixed;
use latticefold::transcript::poseidon::{f257_poseidon_config, F257};

use crate::dr1cs_flpcp::{f_to_u16, is_f257_field, Dr1csNpFlpcpSparseApi, Dr1csQueryScratch, QuerySink};

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
    /// We keep this public so the lock layer can avoid the degenerate target `0`, while the
    /// hidden query randomness remains derived from `armer_secret`.
    pub c_hit: F,
}

/// Public arming artifact for the hidden-query lockable DPP (arm-before-proof).
///
/// This is intentionally minimal: the lock layer needs the accepting set and the hidden-query
/// coefficient material to build *masked* hint blocks; it must not publish any readable Sq/power
/// coefficient encodings.
#[derive(Clone, Debug)]
pub struct Theorem43LockArtifact<F: PrimeField> {
    pub c_stmt: Vec<F>,
    pub accepting_set: [F; 2],
    pub len: usize,
    pub coins: Theorem43Coins<F>,
    /// Hidden UV-derived coefficients (TOXIC WASTE; do not publish directly).
    pub coeffs: Vec<F>,
    /// Hidden UV-derived membership bits over `F_p^*` (TOXIC WASTE; do not publish directly).
    ///
    /// Layout: `uv_bits[lam-1] = 1_{lam ∈ U}` for `lam ∈ {1..=p-1}`.
    /// For the tiny-field WE gate, `p=257` so this has length 256.
    pub uv_bits: Vec<u8>,
}

#[derive(Clone, Debug, Default)]
pub struct Theorem43ArmingStats {
    pub fs_permutes: u64,
    pub q_nnz: usize,
}

/// Non-enumerating Theorem-4.3-style lockable query generator for the NP dR1CS FLPCP.
#[derive(Clone, Debug)]
pub struct Theorem43Dpp<F: PrimeField, P: Dr1csNpFlpcpSparseApi<F>> {
    flpcp: P,
    p: u64,
    _marker: PhantomData<F>,
}

/// Output of `stream_pi0_and_collect_abg_full`.
///
/// We expose both:
/// - `*_pi`: contribution from `π0` only
/// - `*_full`: contribution from `(x || π0)`
///
/// Additionally, for backends that do **not** emit dense `w_eval` query terms (e.g. tiny-lock),
/// we expose the **sparse** values that come only from streamed query terms (i.e. before adding
/// any per-block `w_eval` dot products).
#[derive(Clone, Debug)]
pub struct Theorem43AbgFull<F: PrimeField> {
    pub alpha_pi_sparse: F,
    pub beta_pi_sparse: F,
    pub gamma_pi_sparse: F,
    pub alpha_pi: F,
    pub beta_pi: F,
    pub gamma_pi: F,
    pub alpha_sparse: F,
    pub beta_sparse: F,
    pub gamma_sparse: F,
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
    // Snapshot of the streamed-only values (no per-block w_eval dots applied).
    acc_pi_sparse: F,
    acc_full_sparse: F,
    // Sparse terms over the selected block's `w_eval` slice in π0.
    //
    // In the canonical block-grouped schedule, each coin corresponds to exactly one block
    // (determined by `coins.idx / ell_local`), so we store only local `(coeff, pos)` terms for
    // that block to avoid per-block heap allocations.
    w_terms: Vec<(F, usize)>,
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
            acc_pi_sparse: self.acc_pi,
            acc_full_sparse: self.acc_full,
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
            _marker: PhantomData,
        })
    }

    /// Length of the lock-layer streamed proof payload `π` (field elements).
    ///
    /// Canonical WE lock path is tailless and streams only `π0 = (z_w || w)`.
    pub fn proof_len(&self) -> usize {
        self.flpcp.m()
    }

    /// Arm (host-side FS) and produce a public arming artifact.
    ///
    /// This is used by the lock layer (arm-before-proof). It derives:
    /// - public coins deterministically from `(c_stmt, block_id, rep_id)`
    /// - hidden UV-derived coefficients from the same transcript plus an armer-secret salt
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

        let cfg = f257_poseidon_config();
        let coins = self.derive_public_coins_from_stmt(c_stmt, block_id, rep_id)?;

        // Derive hidden UV bits / coefficients, explicitly binding the public coins.
        let mut sp_hidden = PoseidonSponge::<F257>::new(&cfg);
        let ds = vec![F257::from(43u64), F257::from(2u64)]; // theorem43, coeffs-v2
        sp_hidden.absorb(&ds);
        absorb_field_bytes_as_f257::<F>(&mut sp_hidden, c_stmt);
        absorb_usize_base257(&mut sp_hidden, block_id);
        absorb_u64_base257(&mut sp_hidden, rep_id);
        absorb_usize_base257(&mut sp_hidden, coins.idx);
        absorb_field_bytes_as_f257::<F>(&mut sp_hidden, &[coins.lambda, coins.rho, coins.sigma]);
        absorb_field_bytes_as_f257::<F>(&mut sp_hidden, armer_secret);

        let q_minus_1 = (self.p as usize) - 1;
        let q_bits = squeeze_unbiased_bits_from_f257_digits(&mut sp_hidden, q_minus_1)?;
        let coeffs = self.sq_coeffs_from_uv_bits(&q_bits)?;

        let len = x.len() + self.proof_len();
        Ok(Theorem43LockArtifact {
            c_stmt: c_stmt.to_vec(),
            accepting_set: [coins.c_hit, coins.c_hit + F::ONE],
            len,
            coins,
            coeffs,
            uv_bits: q_bits,
        })
    }

    /// Stream the lock-layer query terms that touch the proof portion `π` only.
    ///
    /// This emits the coefficients for `q_π` after applying the Theorem-4.3 linearization weights.
    /// It returns the *statement-dependent* offset contributed by the public prefix `x`.
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
            return Err("bad coeff length".to_string());
        }

        // Linearization weights (see TR24-114 rev2 / Theorem 4.3 reduction).
        let c1 = coeffs[0];
        let c2 = coeffs[1];
        let coeff_alpha = c1 * coins.rho;
        let coeff_beta = c1 * coins.sigma;
        let coeff_gamma = {
            let two = F::from(2u64);
            c2 * (two * coins.rho * coins.sigma)
        };
        let _coeff_mu = c2 * (coins.rho * coins.rho);
        let _coeff_nu = c2 * (coins.sigma * coins.sigma);

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

        let x_len = x.len();
        let offset = {
            let mut sink = PiQuerySink {
                x,
                x_len,
                // Tailless path: this offset tracks only the public-prefix contribution q_x · x.
                // The accepting-set base `{c_hit, c_hit+1}` is carried separately in `art.accepting_set`
                // and must not be folded into this offset.
                offset: F::ZERO,
                coeff_alpha,
                coeff_beta,
                coeff_gamma,
                on_pi_term,
            };
            flpcp.stream_queries_for_coins_sparse(coins.idx, coins.lambda, x, scratch, &mut sink)?;
            sink.offset
        };

        // Tailless lock path: no μ/ν/u^i coordinates are emitted.

        Ok(offset)
    }

    /// Stream the **UV-independent basis** query terms that touch the proof portion `π` only.
    ///
    /// This emits the raw FLPCP sparse query vectors `q1,q2,q3` (restricted to the proof suffix)
    /// without applying any Theorem-4.3 linearization weights.
    ///
    /// It returns the statement-dependent public-prefix dot products:
    /// - `ax = ⟨q1_x, x⟩`
    /// - `bx = ⟨q2_x, x⟩`
    /// - `gx = ⟨q3_x, x⟩`
    ///
    /// These depend only on `(x, coins, schedule)` and are safe to publish.
    pub fn stream_basis_query_terms_for_pi(
        &self,
        x: &[F],
        coins: &Theorem43Coins<F>,
        scratch: &mut Dr1csQueryScratch<F>,
        on_alpha_pi_term: &mut dyn FnMut(usize, F),
        on_beta_pi_term: &mut dyn FnMut(usize, F),
        on_gamma_pi_term: &mut dyn FnMut(usize, F),
    ) -> Result<(F, F, F), String> {
        let flpcp = &self.flpcp;
        if x.len() != flpcp.n() {
            return Err("bad public input length".to_string());
        }
        if coins.idx >= flpcp.ell() {
            return Err("bad idx coin".to_string());
        }

        struct BasisPiQuerySink<'a, F: PrimeField> {
            x: &'a [F],
            x_len: usize,
            ax: F,
            bx: F,
            gx: F,
            on_alpha_pi_term: &'a mut dyn FnMut(usize, F),
            on_beta_pi_term: &'a mut dyn FnMut(usize, F),
            on_gamma_pi_term: &'a mut dyn FnMut(usize, F),
        }
        impl<'a, F: PrimeField> QuerySink<F> for BasisPiQuerySink<'a, F> {
            fn on_q1(&mut self, coeff: F, idx: usize) {
                if coeff.is_zero() {
                    return;
                }
                if idx < self.x_len {
                    self.ax += coeff * self.x[idx];
                } else {
                    (self.on_alpha_pi_term)(idx - self.x_len, coeff);
                }
            }
            fn on_q2(&mut self, coeff: F, idx: usize) {
                if coeff.is_zero() {
                    return;
                }
                if idx < self.x_len {
                    self.bx += coeff * self.x[idx];
                } else {
                    (self.on_beta_pi_term)(idx - self.x_len, coeff);
                }
            }
            fn on_q3(&mut self, coeff: F, idx: usize) {
                if coeff.is_zero() {
                    return;
                }
                if idx < self.x_len {
                    self.gx += coeff * self.x[idx];
                } else {
                    (self.on_gamma_pi_term)(idx - self.x_len, coeff);
                }
            }
        }

        let x_len = x.len();
        let mut sink = BasisPiQuerySink {
            x,
            x_len,
            ax: F::ZERO,
            bx: F::ZERO,
            gx: F::ZERO,
            on_alpha_pi_term,
            on_beta_pi_term,
            on_gamma_pi_term,
        };
        flpcp.stream_queries_for_coins_sparse(coins.idx, coins.lambda, x, scratch, &mut sink)?;
        Ok((sink.ax, sink.bx, sink.gx))
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

        // Squeeze PUBLIC coins with rejection sampling.
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
        // Derive `c_hit ∈ {1..=127}` to avoid inverse-ratio collisions in `{c,c+1}` decoding and
        // to keep the accepting set away from 0.
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

    pub fn query_scratch(&self) -> Dr1csQueryScratch<F> {
        Dr1csQueryScratch::new(self.flpcp.n_total())
    }

}

impl<F: PrimeField, P: Dr1csNpFlpcpSparseApi<F> + Sync> Theorem43Dpp<F, P> {
    fn stream_pi0_and_collect_abg_accs(
        &self,
        x: &[F],
        z_w: &[F],
        coins_list: &[Theorem43Coins<F>],
        mut on_pi0_chunk: Option<&mut dyn FnMut(&[F])>,
    ) -> Result<Vec<Theorem43AbgFull<F>>, String> {
        let prof = std::env::var("LF_PROFILE").ok().as_deref() == Some("1");
        let t_total = Instant::now();
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
        let t_build_accs = Instant::now();
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
        if prof {
            eprintln!(
                "[LF_PROFILE] theorem43 build_accs elapsed={:.3}s coins={}",
                t_build_accs.elapsed().as_secs_f64(),
                coins_list.len()
            );
        }

        // Block-grouped schedule: each coin updates exactly its selected block.
        let t_bucket = Instant::now();
        let avg_per_block = (accs.len().saturating_add(blocks).saturating_sub(1)) / blocks.max(1);
        let mut bucket: Vec<Vec<usize>> = (0..blocks)
            .map(|_| Vec::with_capacity(avg_per_block.max(1)))
            .collect();
        for (ci, a) in accs.iter().enumerate() {
            bucket[a.block_id].push(ci);
        }
        if prof {
            eprintln!(
                "[LF_PROFILE] theorem43 bucket_schedule elapsed={:.3}s blocks={} coins={}",
                t_bucket.elapsed().as_secs_f64(),
                blocks,
                coins_list.len()
            );
        }

        if let Some(cb) = on_pi0_chunk.as_deref_mut() {
            cb(z_w);
        }
        let emit_pi0 = on_pi0_chunk.is_some();

        let t_setup = Instant::now();
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
        if prof {
            eprintln!(
                "[LF_PROFILE] theorem43 pre_stream_setup elapsed={:.3}s blocks={} coins={}",
                t_setup.elapsed().as_secs_f64(),
                blocks,
                coins_list.len()
            );
        }

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

            // q1/q2 witness dots in parallel over coins in this block.
            let q12 = coins_b
                .par_iter()
                .map(|&ci| -> Result<(F, F), String> {
                    let a = accs_ro.get(ci).ok_or_else(|| {
                        "stream_w_eval_blocks_with_hook: coin index out of range".to_string()
                    })?;
                    let lut_ref = lut.as_ref().ok_or_else(|| {
                        "stream_w_eval_blocks_with_hook: q1/q2 requires F257 u16 pipeline".to_string()
                    })?;

                    let mut s1 = F::ZERO;
                    for (c, pos) in a.acc0.w_terms.iter().copied() {
                        if pos >= w_eval_u16.len() {
                            return Err(
                                "stream_w_eval_blocks_with_hook: q1 w_eval_u16 index out of range"
                                    .to_string(),
                            );
                        }
                        s1 += c * lut_ref[w_eval_u16[pos] as usize];
                    }

                    let mut s2 = F::ZERO;
                    for (c, pos) in a.acc1.w_terms.iter().copied() {
                        if pos >= w_eval_u16.len() {
                            return Err(
                                "stream_w_eval_blocks_with_hook: q2 w_eval_u16 index out of range"
                                    .to_string(),
                            );
                        }
                        s2 += c * lut_ref[w_eval_u16[pos] as usize];
                    }
                    Ok((s1, s2))
                })
                .collect::<Result<Vec<_>, String>>()?;
            for (j, (s1, s2)) in q12.into_iter().enumerate() {
                dots.q1_w[j] = s1;
                dots.q2_w[j] = s2;
            }

            // q3 witness dot: heavy dense part, batched.
            const MAX_BATCH: usize = 64;
            let q3_chunks = coins_b
                .par_chunks(MAX_BATCH)
                .enumerate()
                .map(|(chunk_idx, chunk)| -> Result<(usize, Vec<F>), String> {
                    let mut idxs = Vec::with_capacity(chunk.len());
                    let mut lambdas = Vec::with_capacity(chunk.len());
                    for &ci in chunk {
                        let a = accs_ro.get(ci).ok_or_else(|| {
                            "stream_w_eval_blocks_with_hook: coin index out of range (q3)".to_string()
                        })?;
                        idxs.push(a.coins.idx);
                        lambdas.push(a.coins.lambda);
                    }
                    let mut out = vec![F::ZERO; chunk.len()];
                    // Backend expects `w_eval_u16` for this block; structured backends override this
                    // to amortize dense q3 dot products.
                    flpcp.dot_q3_w_eval_many(&idxs, &lambdas, x, w_eval_u16, &mut out)?;
                    Ok((chunk_idx * MAX_BATCH, out))
                })
                .collect::<Result<Vec<_>, String>>()?;
            for (off, vals) in q3_chunks {
                for (j, v) in vals.into_iter().enumerate() {
                    dots.q3_w[off + j] = v;
                }
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
        let x_u16_cache = if is_f257_field::<F>() {
            Some(x.iter().copied().map(f_to_u16).collect::<Vec<_>>())
        } else {
            None
        };
        let z_u16_cache = if is_f257_field::<F>() {
            Some(z_w.iter().copied().map(f_to_u16).collect::<Vec<_>>())
        } else {
            None
        };
        let t_stream = Instant::now();
        flpcp
            .stream_w_eval_blocks(
                &witness_pos,
                x,
                z_w,
                x_u16_cache.as_deref(),
                z_u16_cache.as_deref(),
                Some(&on_block_hook),
                if emit_pi0 { Some(&mut on_block) } else { None },
            )
            .map_err(|e| format!("stream_w_eval_blocks failed: {e}"))?;
        if std::env::var("LF_PROFILE").ok().as_deref() == Some("1") {
            eprintln!(
                "[LF_PROFILE] theorem43 stream_w_eval_blocks elapsed={:.3}s blocks={} coins={}",
                t_stream.elapsed().as_secs_f64(),
                blocks,
                coins_list.len()
            );
        }

        if let Some(e) = err {
            return Err(e);
        }

        // Apply per-block witness dots.
        let t_apply = Instant::now();
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
        if std::env::var("LF_PROFILE").ok().as_deref() == Some("1") {
            eprintln!(
                "[LF_PROFILE] theorem43 apply_per_block elapsed={:.3}s blocks={} coins={}",
                t_apply.elapsed().as_secs_f64(),
                blocks,
                coins_list.len()
            );
            eprintln!(
                "[LF_PROFILE] theorem43 total_stream_abg elapsed={:.3}s blocks={} coins={}",
                t_total.elapsed().as_secs_f64(),
                blocks,
                coins_list.len()
            );
        }

        let mut out = Vec::with_capacity(accs.len());
        for a in accs.into_iter() {
            out.push(Theorem43AbgFull {
                alpha_pi_sparse: a.acc0.acc_pi_sparse,
                beta_pi_sparse: a.acc1.acc_pi_sparse,
                gamma_pi_sparse: a.acc2.acc_pi_sparse,
                alpha_pi: a.acc0.acc_pi,
                beta_pi: a.acc1.acc_pi,
                gamma_pi: a.acc2.acc_pi,
                alpha_sparse: a.acc0.acc_full_sparse,
                beta_sparse: a.acc1.acc_full_sparse,
                gamma_sparse: a.acc2.acc_full_sparse,
                alpha: a.acc0.acc_full,
                beta: a.acc1.acc_full,
                gamma: a.acc2.acc_full,
            });
        }
        Ok(out)
    }

    /// Stream `π0` once and return full `(alpha,beta,gamma)` for many coins.
    ///
    /// This is the same streaming schedule as the legacy tail-streaming routine, but it **does not**
    /// generate or stream tails. It exists for the WE "poison" backend, which needs the MulEq
    /// residual `err = gamma - alpha*beta` across blocks.
    pub fn stream_pi0_and_collect_abg_full(
        &self,
        x: &[F],
        z_w: &[F],
        coins_list: &[Theorem43Coins<F>],
        on_pi0_chunk: Option<&mut dyn FnMut(&[F])>,
    ) -> Result<Vec<Theorem43AbgFull<F>>, String> {
        self.stream_pi0_and_collect_abg_accs(x, z_w, coins_list, on_pi0_chunk)
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

