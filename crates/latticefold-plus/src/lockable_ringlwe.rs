//! Lock layer for Theorem-4.3 WE arming (mod-257 linear hints + amplification).
//!
//! # Architecture
//!
//! This module provides the lock layer for the arm-before-proof WE scheme:
//! - arming publishes sparse hint blocks over F257 (one hint vector per accepting-set branch)
//!   and a **single** unauthenticated ciphertext per lock (XOR-stream under a derived key).
//! - decapsulation is streaming: absorb proof chunks, then `finish_decrypt_candidates()`.
//!
//! # Security model
//!
//! Per-lock security is **not** standalone 128-bit. In this deterministic mod-257 design, the lock
//! provides only a small per-lock ambiguity (on the order of ~8–9 bits), and security is achieved
//! through amplification (many locks per armer, no per-lock verification oracle).
//!
//! The lock encryption is **unauthenticated** (no per-lock MAC/tag). This prevents a per-lock
//! verification oracle, forcing the adversary to guess all T locks jointly.

use ark_ff::PrimeField;
use rand::RngCore;

/// Block packing size for sparse hints and streamed π.
const PACK_D: usize = 64;

/// Prime modulus for the tiny field.
const MOD_257: u16 = 257;

/// Goldilocks prime field modulus (for PVUGC-style noisy hints).
///
/// p = 2^64 - 2^32 + 1
pub(crate) const GOLDILOCKS_P: u64 = 18446744069414584321u64;

#[inline]
fn f_to_u64<F: PrimeField>(f: &F) -> u64 {
    f.into_bigint().as_ref()[0]
}

// NOTE: keep `sub_mod257` only if we reintroduce a subtraction-based packing.

#[inline]
pub(crate) fn field_mod257_u16<F: PrimeField>(f: &F) -> u16 {
    (f_to_u64(f) % 257) as u16
}

#[inline]
pub(crate) fn add_mod257_u16(a: u16, b: u16) -> u16 {
    let s = a + b;
    if s >= MOD_257 { s - MOD_257 } else { s }
}

#[inline]
pub(crate) fn sub_mod257_u16(a: u16, b: u16) -> u16 {
    // Returns (a - b) mod 257 with inputs assumed reduced.
    debug_assert!(a < MOD_257 && b < MOD_257);
    if a >= b { a - b } else { a + MOD_257 - b }
}

#[inline]
fn mul_mod257(a: u16, b: u16) -> u16 {
    // Fast reduction mod 257 using 256 ≡ -1 (mod 257).
    //
    // Precondition in the WE pipeline: values are always reduced to 0..=256.
    debug_assert!(a < MOD_257 && b < MOD_257);
    let prod = (a as u32) * (b as u32); // <= 65536
    let low = (prod & 0xFF) as i32; // 0..255
    let high = (prod >> 8) as i32; // 0..256
    let mut r = low - high; // -256..255
    if r < 0 {
        r += 257;
    }
    debug_assert!((0..257).contains(&r));
    r as u16
}

#[inline]
pub(crate) fn mul_mod257_u16(a: u16, b: u16) -> u16 {
    mul_mod257(a, b)
}

#[inline]
fn pow_mod257(mut a: u16, mut e: u16) -> u16 {
    // MOD_257 is prime (257), so inverse is a^(255) for a!=0.
    let mut acc = 1u16;
    while e > 0 {
        if (e & 1) != 0 {
            acc = mul_mod257(acc, a);
        }
        a = mul_mod257(a, a);
        e >>= 1;
    }
    acc
}

#[inline]
fn inv_mod257(a: u16) -> Result<u16, String> {
    if a % MOD_257 == 0 {
        return Err("inv_mod257: inverse of 0".to_string());
    }
    Ok(pow_mod257(a % MOD_257, 255))
}

// ---------------------------------------------------------------------------
// Goldilocks helpers (PVUGC-style hint arithmetic)
// ---------------------------------------------------------------------------

#[inline]
pub(crate) fn add_mod_goldilocks(a: u64, b: u64) -> u64 {
    let p = GOLDILOCKS_P as u128;
    let s = (a as u128) + (b as u128);
    (s % p) as u64
}

#[inline]
pub(crate) fn mul_mod_goldilocks(a: u64, b: u64) -> u64 {
    let p = GOLDILOCKS_P as u128;
    let prod = (a as u128) * (b as u128);
    (prod % p) as u64
}

#[inline]
fn goldilocks_from_i64(x: i64) -> u64 {
    if x >= 0 {
        (x as u64) % GOLDILOCKS_P
    } else {
        let t = ((-x) as u64) % GOLDILOCKS_P;
        if t == 0 { 0 } else { GOLDILOCKS_P - t }
    }
}

#[inline]
fn sample_nonzero_goldilocks(rng: &mut impl RngCore) -> u64 {
    loop {
        let v = rng.next_u64() % GOLDILOCKS_P;
        if v != 0 {
            return v;
        }
    }
}

#[inline]
fn sample_noise_uniform_i64(rng: &mut impl RngCore, bound: i64) -> i64 {
    if bound <= 0 {
        return 0;
    }
    // Uniform in [-bound, bound].
    let span: u64 = (2 * (bound as i128) + 1) as u64;
    let r = rng.next_u64() % span;
    (r as i64) - bound
}

/// PVUGC-style noisy hint vectors.
///
/// These are the intended replacement for publishing readable masked coefficient blocks
/// (e.g. `s * tail_coeffs (mod 257)`), which admits chosen-probe / scan-style attacks for tiny-image
/// carriers (like the current Sq/power-sum gadget).
#[derive(Clone, Debug)]
pub struct PvugcNoisyHints {
    pub modulus: u64,
    pub hint0: Vec<u64>,
    pub hint1: Vec<u64>,
}

#[derive(Clone, Copy, Debug)]
pub struct PvugcNoisyHintParams {
    /// Per-coordinate noise bound B: noise is sampled uniformly from [-B, B].
    ///
    /// This is a dependency-free scaffold; production should use a discrete Gaussian + reconciliation.
    pub noise_bound: i64,
    /// Reconciliation granularity: we round inner products to the nearest multiple of 2^t.
    ///
    /// Correctness sufficient condition (with bounded `s` and no modulus wrap):
    /// - sample secret scales as multiples of 2^t
    /// - ensure total noise on the inner product is < 2^(t-1)
    pub round_bits: u32,
    /// Sample the (pre-shift) secret scale `r` from `0..2^s_bits`, then set `s = r<<round_bits`.
    ///
    /// This keeps the true inner product far from the Goldilocks modulus so arithmetic does not wrap.
    pub s_bits: u32,
}

impl Default for PvugcNoisyHintParams {
    fn default() -> Self {
        Self {
            noise_bound: 8,
            round_bits: 20,
            s_bits: 32,
        }
    }
}

#[inline]
fn sample_bounded_scale_pow2(
    rng: &mut impl RngCore,
    round_bits: u32,
    s_bits: u32,
) -> Result<u64, String> {
    if round_bits >= 63 {
        return Err("pvugc: round_bits too large".to_string());
    }
    if s_bits == 0 || s_bits >= 63 {
        return Err("pvugc: s_bits out of range".to_string());
    }
    let mask = (1u64 << s_bits) - 1;
    loop {
        let r = rng.next_u64() & mask;
        if r == 0 {
            continue;
        }
        let s = r
            .checked_shl(round_bits)
            .ok_or_else(|| "pvugc: scale shift overflow".to_string())?;
        if s == 0 || s >= GOLDILOCKS_P {
            continue;
        }
        return Ok(s);
    }
}

/// Arm PVUGC-style noisy hints for a secret coefficient/query vector over mod-257 digits.
///
/// Input `q_mod257` is treated as *secret* (it replaces publishing readable `s*q (mod 257)`).
pub fn arm_pvugc_noisy_hints_goldilocks_from_secret_q_mod257(
    rng: &mut impl RngCore,
    params: PvugcNoisyHintParams,
    q_mod257: &[u16],
) -> (PvugcNoisyHints, (u64, u64)) {
    let s0 = sample_bounded_scale_pow2(rng, params.round_bits, params.s_bits)
        .expect("pvugc: bounded scale sampling failed");
    let s1 = sample_bounded_scale_pow2(rng, params.round_bits, params.s_bits)
        .expect("pvugc: bounded scale sampling failed");
    let mut hint0: Vec<u64> = Vec::with_capacity(q_mod257.len());
    let mut hint1: Vec<u64> = Vec::with_capacity(q_mod257.len());
    for &qi_u16 in q_mod257 {
        let qi = (qi_u16 as u64) % GOLDILOCKS_P;
        let e0 = goldilocks_from_i64(sample_noise_uniform_i64(rng, params.noise_bound));
        let e1 = goldilocks_from_i64(sample_noise_uniform_i64(rng, params.noise_bound));
        hint0.push(add_mod_goldilocks(mul_mod_goldilocks(s0, qi), e0));
        hint1.push(add_mod_goldilocks(mul_mod_goldilocks(s1, qi), e1));
    }
    (
        PvugcNoisyHints {
            modulus: GOLDILOCKS_P,
            hint0,
            hint1,
        },
        (s0, s1),
    )
}

#[inline]
pub(crate) fn round_to_pow2_multiple(x: u64, round_bits: u32) -> u64 {
    if round_bits == 0 {
        return x;
    }
    let step = 1u64 << round_bits;
    let half = step >> 1;
    (x.wrapping_add(half)) & !(step - 1)
}

// ---------------------------------------------------------------------------
// Noisy lock artifacts (no readable mod-257 scales)
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct FanoutCiphertext2 {
    pub ct0: LockCiphertext,
    pub ct1: LockCiphertext,
}

#[derive(Clone, Debug)]
pub struct NoisyBranchHints {
    pub params: PvugcNoisyHintParams,
    pub hints: PvugcNoisyHints,
}

#[derive(Clone, Debug)]
pub struct NoisySubLock<F: PrimeField> {
    pub accepting_set: [F; 2],
    pub anchor_block_id: u32,
    pub rep_id: u64,
    /// Number of blocks folded into the poison term (for this lock instance).
    ///
    /// The sublock's secret coefficient vector `q` includes `poison_blocks` additional coordinates,
    /// and the decap feature vector `v` appends the per-block MulEq residuals `err_b`.
    pub poison_blocks: u32,
    /// If nonzero, decap compresses the full `err_b` vector (length = `poison_blocks`) into
    /// `poison_proj_m` random linear projections in F257, and uses those as the feature vector
    /// coordinates (instead of appending all `err_b` directly).
    ///
    /// This shrinks PVUGC hint vectors from length `1+poison_blocks` down to `1+poison_proj_m`.
    pub poison_proj_m: u32,
    pub hints: NoisyBranchHints,
    /// Fanout ciphertexts for the two accepting-set branches (no tags).
    pub cts: FanoutCiphertext2,
}

#[derive(Clone, Debug)]
pub struct NoisyLockArtifact<F: PrimeField> {
    pub c_stmt: Vec<F>,
    pub x_len: usize,
    pub pi_len: usize,
    pub len: usize,
    pub params: RingLweParams,
    /// Sublocks (each yields 2 candidate plaintexts).
    pub sublocks: Vec<NoisySubLock<F>>,
}

/// Compute the two noisy inner products `(<hint0,v>, <hint1,v>)` where `v` is a mod-257 digit vector.
pub fn pvugc_noisy_inner_products_goldilocks(
    hints: &PvugcNoisyHints,
    v_mod257: &[u16],
) -> Result<(u64, u64), String> {
    if hints.modulus != GOLDILOCKS_P {
        return Err("pvugc_noisy_inner_products_goldilocks: unexpected modulus".to_string());
    }
    if hints.hint0.len() != v_mod257.len() || hints.hint1.len() != v_mod257.len() {
        return Err("pvugc_noisy_inner_products_goldilocks: length mismatch".to_string());
    }
    let mut y0: u64 = 0;
    let mut y1: u64 = 0;
    for i in 0..v_mod257.len() {
        let vi = (v_mod257[i] as u64) % GOLDILOCKS_P;
        y0 = add_mod_goldilocks(y0, mul_mod_goldilocks(hints.hint0[i], vi));
        y1 = add_mod_goldilocks(y1, mul_mod_goldilocks(hints.hint1[i], vi));
    }
    Ok((y0, y1))
}

/// Compress a per-block residual vector `errs` (mod 257) into `m` random linear projections.
///
/// This is a public deterministic map derived from `(stmt_digest_bytes64, rep_id, proj_idx)`.
/// It is used to reduce PVUGC hint dimension from `O(blocks)` down to `O(m)`.
pub(crate) fn project_errs_mod257_u16(
    stmt_digest_bytes64: &[u8; 64],
    rep_id: u64,
    errs: &[u16],
    m: usize,
) -> Vec<u16> {
    use rand::{RngCore, SeedableRng};
    use rand_chacha::ChaCha20Rng;
    use sha2::Digest;

    let mut out: Vec<u16> = Vec::with_capacity(m);
    for proj_idx in 0..m {
        let mut h = sha2::Sha256::new();
        h.update(b"LFP_POISON_PROJ_V1");
        h.update(stmt_digest_bytes64);
        h.update(&rep_id.to_le_bytes());
        h.update(&(proj_idx as u64).to_le_bytes());
        let seed: [u8; 32] = h.finalize().into();
        let mut prg = ChaCha20Rng::from_seed(seed);

        let mut acc: u16 = 0;
        for &e in errs {
            // Public projection coefficient in F257 (0..=256).
            let r = (prg.next_u32() % 257) as u16;
            let re = mul_mod257_u16(r, e);
            acc = add_mod257_u16(acc, re);
        }
        out.push(acc);
    }
    out
}


// ---------------------------------------------------------------------------
// Amplification parameters
// ---------------------------------------------------------------------------

/// Classical brute-force bits under the global-check-only model.
///
/// Security accounting is driven by the number of independent 8-bit channel secrets that
/// must be guessed jointly before one global check is available:
///   classical_bits = 8 * P * L
/// where P = channels per lock, L = required locks.
pub fn classical_bits_global_check_only(p_channels: u16, required_locks: usize) -> f64 {
    8.0 * (p_channels as f64) * (required_locks as f64)
}

/// Grover bits under the same model:
///   grover_bits = classical_bits / 2 = 4 * P * L
pub fn grover_bits_global_check_only(p_channels: u16, required_locks: usize) -> f64 {
    0.5 * classical_bits_global_check_only(p_channels, required_locks)
}

/// Heuristic PQ-128 condition under Grover-only accounting:
///   P * L >= 32  <=>  grover_bits >= 128
pub fn meets_pq128_grover_global_check_only(p_channels: u16, required_locks: usize) -> bool {
    (p_channels as usize) * required_locks >= 32
}

// ---------------------------------------------------------------------------
// Lock parameters and artifact types
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct RingLweParams {
    /// Reserved for future non-deterministic / non-mod257 designs.
    ///
    /// In the current deterministic mod-257 lock, these are not used for security.
    pub _reserved0: u32,
    /// Domain separation label.
    pub domain_label: [u8; 32],
}

impl Default for RingLweParams {
    fn default() -> Self {
        Self {
            // Single canonical encoding: dual-format sparse/dense-in-block hints.
            _reserved0: 2,
            domain_label: *b"LFP_RINGLWE_V2_00000000000000000",
        }
    }
}

/// Per-branch ciphertext: unauthenticated stream cipher (XOR) under a derived key.
///
/// Critical: there is **no per-lock authentication tag**, to avoid a per-lock verification oracle
/// that would collapse threshold amplification.
#[derive(Clone, Debug)]
pub struct LockCiphertext {
    pub nonce: [u8; 12],
    pub ct: Vec<u8>,
}

// ---------------------------------------------------------------------------
// Cryptographic helpers
// ---------------------------------------------------------------------------


fn sha256_32(chunks: &[&[u8]]) -> [u8; 32] {
    use sha2::Digest;
    let mut h = sha2::Sha256::new();
    for c in chunks {
        h.update(c);
    }
    h.finalize().into()
}

fn derive_noisy_fanout_key_bytes<F: PrimeField>(
    domain_label: &[u8; 32],
    c_stmt: &[F],
    anchor_block_id: u32,
    rep_id: u64,
    branch_id: u8,
    k0: u64,
    k1: u64,
) -> [u8; 32] {
    // Key binding:
    // - statement commitment (c_stmt)
    // - sublock coin-derivation inputs (anchor_block_id, rep_id)
    // - branch_id (0/1)
    // - reconciled inner products k0,k1 (large modulus)
    use sha2::Digest;
    let mut stmt_bytes = Vec::with_capacity(c_stmt.len() * 8);
    for f in c_stmt {
        stmt_bytes.extend_from_slice(&f_to_u64(f).to_le_bytes());
    }
    let mut h = sha2::Sha256::new();
    h.update(b"LFP_NOISY_FANOUT_KEY_V1");
    h.update(domain_label);
    h.update(&(c_stmt.len() as u64).to_le_bytes());
    h.update(stmt_bytes.as_slice());
    h.update(&anchor_block_id.to_le_bytes());
    h.update(&rep_id.to_le_bytes());
    h.update(&[branch_id]);
    h.update(&k0.to_le_bytes());
    h.update(&k1.to_le_bytes());
    h.finalize().into()
}

#[inline]
fn reconcile_k_pair(y0: u64, y1: u64, params: PvugcNoisyHintParams) -> (u64, u64) {
    // Scaffold reconciliation: round to nearest multiple of 2^t.
    let k0 = round_to_pow2_multiple(y0, params.round_bits);
    let k1 = round_to_pow2_multiple(y1, params.round_bits);
    (k0, k1)
}

pub fn arm_noisy_fanout_ciphertexts_for_accepting_set<F: PrimeField>(
    rng: &mut impl RngCore,
    domain_label: &[u8; 32],
    c_stmt: &[F],
    anchor_block_id: u32,
    rep_id: u64,
    accepting_set: &[F; 2],
    s0: u64,
    s1: u64,
    payload: &[u8],
) -> Result<FanoutCiphertext2, String> {
    // Note: `accepting_set` lives in the tiny field (mod 257 digits in 1..=256).
    let a0 = (f_to_u64(&accepting_set[0]) % 257) as u64;
    let a1 = (f_to_u64(&accepting_set[1]) % 257) as u64;
    if a0 == 0 || a1 == 0 {
        return Err("arm_noisy_fanout_ciphertexts: accepting set contains 0".to_string());
    }
    let k00 = mul_mod_goldilocks(s0, a0);
    let k10 = mul_mod_goldilocks(s1, a0);
    let k01 = mul_mod_goldilocks(s0, a1);
    let k11 = mul_mod_goldilocks(s1, a1);

    let key0 = derive_noisy_fanout_key_bytes(domain_label, c_stmt, anchor_block_id, rep_id, 0u8, k00, k10);
    let key1 = derive_noisy_fanout_key_bytes(domain_label, c_stmt, anchor_block_id, rep_id, 1u8, k01, k11);

    let mut nonce0 = [0u8; 12];
    let mut nonce1 = [0u8; 12];
    rng.fill_bytes(&mut nonce0);
    rng.fill_bytes(&mut nonce1);
    let ct0 = LockCiphertext {
        nonce: nonce0,
        ct: xor_stream_encrypt(&key0, &nonce0, payload),
    };
    let ct1 = LockCiphertext {
        nonce: nonce1,
        ct: xor_stream_encrypt(&key1, &nonce1, payload),
    };
    Ok(FanoutCiphertext2 { ct0, ct1 })
}

pub fn decap_noisy_fanout_candidates<F: PrimeField>(
    lock: &NoisyLockArtifact<F>,
    sl: &NoisySubLock<F>,
    v_mod257: &[u16],
) -> Result<[Vec<u8>; 2], String> {
    let (y0, y1) = pvugc_noisy_inner_products_goldilocks(&sl.hints.hints, v_mod257)?;
    let (k0, k1) = reconcile_k_pair(y0, y1, sl.hints.params);
    let key0 = derive_noisy_fanout_key_bytes(
        &lock.params.domain_label,
        &lock.c_stmt,
        sl.anchor_block_id,
        sl.rep_id,
        0u8,
        k0,
        k1,
    );
    let key1 = derive_noisy_fanout_key_bytes(
        &lock.params.domain_label,
        &lock.c_stmt,
        sl.anchor_block_id,
        sl.rep_id,
        1u8,
        k0,
        k1,
    );
    let p0 = xor_stream_decrypt(&key0, &sl.cts.ct0.nonce, &sl.cts.ct0.ct);
    let p1 = xor_stream_decrypt(&key1, &sl.cts.ct1.nonce, &sl.cts.ct1.ct);
    Ok([p0, p1])
}

/// Unauthenticated XOR stream cipher under a SHA256-derived keystream.
fn xor_stream_encrypt(key: &[u8; 32], nonce: &[u8; 12], data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    let mut ctr = 0u32;
    let mut pos = 0usize;
    while pos < data.len() {
        let block = sha256_32(&[b"LFP_XOR_STREAM_V1", key, nonce, &ctr.to_le_bytes()]);
        let take = (data.len() - pos).min(32);
        for j in 0..take {
            out.push(data[pos + j] ^ block[j]);
        }
        pos += take;
        ctr += 1;
    }
    out
}

#[inline]
fn xor_stream_decrypt(key: &[u8; 32], nonce: &[u8; 12], ct: &[u8]) -> Vec<u8> {
    xor_stream_encrypt(key, nonce, ct)
}


// ---------------------------------------------------------------------------
// Tests (attack/sanity harnesses)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{RngCore, SeedableRng};
    use rand_chacha::ChaCha20Rng;

    // This documents a critical weakness of the current "PVUGC noisy hints" scaffold:
    //
    // We publish per-coordinate values `hint[i] = s * q[i] + e[i] (mod p)` where:
    // - `s` is a single global scalar with a large known power-of-two factor (`s = r << t`)
    // - `q[i]` are tiny digits in 0..=256 (and include many random poison weights in 1..=256)
    // - `|e[i]|` is tiny, and `p` is huge so there is effectively no wrap
    //
    // Under those conditions, a passive attacker can recover `r` (hence `s`) by rounding
    // to remove the low-bit noise and taking a gcd across coordinates.
    #[test]
    fn test_noisy_hints_scaffold_is_gcd_breakable_passively() {
        fn center_lift_i128(x: u64) -> i128 {
            // Map `x ∈ [0,p)` to `(-p/2, p/2]` as a signed integer.
            let p = GOLDILOCKS_P as i128;
            let xi = x as i128;
            if xi > (p / 2) {
                xi - p
            } else {
                xi
            }
        }

        fn round_div_pow2_i128(x: i128, bits: u32) -> i128 {
            if bits == 0 {
                return x;
            }
            let half: i128 = 1i128 << (bits - 1);
            if x >= 0 {
                (x + half) >> bits
            } else {
                -(((-x) + half) >> bits)
            }
        }

        fn gcd_u128(mut a: u128, mut b: u128) -> u128 {
            while b != 0 {
                let r = a % b;
                a = b;
                b = r;
            }
            a
        }

        let mut rng = ChaCha20Rng::from_seed([9u8; 32]);
        let params = PvugcNoisyHintParams::default();

        // Use a moderately large vector and force a coordinate with q=1 so gcd(q)=1.
        let l = 4096usize;
        let mut q_mod257: Vec<u16> = Vec::with_capacity(l);
        q_mod257.push(1u16);
        for _ in 1..l {
            // 1..=256
            q_mod257.push(((rng.next_u32() as u16) & 255u16) + 1u16);
        }

        let (hints, (s0, s1)) =
            arm_pvugc_noisy_hints_goldilocks_from_secret_q_mod257(&mut rng, params, &q_mod257);
        assert_eq!(hints.modulus, GOLDILOCKS_P);
        assert_eq!(hints.hint0.len(), l);
        assert_eq!(hints.hint1.len(), l);

        // Recover s0 via rounding + gcd.
        let t = params.round_bits;
        let r0_true = (s0 >> t) as u128;
        let r1_true = (s1 >> t) as u128;
        assert!(r0_true != 0 && r1_true != 0);

        let mut g0: u128 = 0;
        let mut g1: u128 = 0;
        for i in 0..l {
            let x0 = center_lift_i128(hints.hint0[i]);
            let x1 = center_lift_i128(hints.hint1[i]);
            let a0 = round_div_pow2_i128(x0, t).unsigned_abs() as u128;
            let a1 = round_div_pow2_i128(x1, t).unsigned_abs() as u128;
            if a0 != 0 {
                g0 = if g0 == 0 { a0 } else { gcd_u128(g0, a0) };
            }
            if a1 != 0 {
                g1 = if g1 == 0 { a1 } else { gcd_u128(g1, a1) };
            }
        }

        assert_eq!(
            g0, r0_true,
            "passive gcd recovery failed for s0 (this test documents a known weakness)"
        );
        assert_eq!(
            g1, r1_true,
            "passive gcd recovery failed for s1 (this test documents a known weakness)"
        );

        // With s0 recovered, q digits are recoverable by rounding hint0[i]/s0.
        for i in 0..64usize {
            let x0 = center_lift_i128(hints.hint0[i]);
            let qi_est = {
                let num = x0;
                let den = s0 as i128;
                // nearest integer to num/den
                if num >= 0 {
                    ((num + den / 2) / den) as i128
                } else {
                    -(((-num + den / 2) / den) as i128)
                }
            };
            assert_eq!(qi_est as u16, q_mod257[i]);
        }
    }
}
