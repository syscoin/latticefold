//! Minimal Shamir secret sharing over GF(256) for 32-byte secrets.
//!
//! This is a small, dependency-light implementation intended for WE lockset scaffolding:
//! - split a 32-byte secret into N shares with threshold T
//! - reconstruct from any T valid shares
//!
//! Security note: this is information-theoretic given correct usage; it is *not* the cryptographic
//! hardness layer. The hardness is intended to come from the upstream DPP→(R)LWE “decap” mechanism.

use rand::RngCore;
use sha2::{Digest, Sha256};
use thiserror::Error;

#[derive(Clone, Copy, Debug)]
pub struct ShamirConfig {
    pub threshold: usize,
    pub shares: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ShamirShare {
    /// Share index in \([1..=N]\).
    pub index: u32,
    /// The 32-byte share payload (one byte per secret byte, independently shared).
    pub value: [u8; 32],
}

#[derive(Debug, Error)]
pub enum ShamirError {
    #[error("invalid parameters")]
    InvalidParams,
    #[error("insufficient shares")]
    InsufficientShares,
    #[error("duplicate share index")]
    DuplicateIndex,
    #[error("share index out of range")]
    IndexOutOfRange,
}

/// Split a 32-byte secret into `cfg.shares` shares with threshold `cfg.threshold`.
pub fn split_secret_32(
    rng: &mut dyn RngCore,
    cfg: &ShamirConfig,
    secret: [u8; 32],
) -> Result<Vec<ShamirShare>, ShamirError> {
    // This GF(256) implementation supports only 255 nonzero x-coordinates.
    if cfg.threshold < 2
        || cfg.shares < cfg.threshold
        || cfg.shares > 255
        || cfg.shares > u32::MAX as usize
    {
        return Err(ShamirError::InvalidParams);
    }

    // For each byte position, sample a random degree-(t-1) polynomial with constant term = secret[b].
    // Then evaluate it at x=1..=N in GF(256).
    let t = cfg.threshold;
    let n = cfg.shares;

    let mut coeffs: Vec<[u8; 32]> = Vec::with_capacity(t);
    coeffs.push(secret);
    for _ in 1..t {
        let mut c = [0u8; 32];
        rng.fill_bytes(&mut c);
        coeffs.push(c);
    }

    let mut out = Vec::with_capacity(n);
    for i in 1..=n {
        let x = i as u8; // require i<=255 for GF(256) x coords
        if x == 0 || i > 255 {
            return Err(ShamirError::InvalidParams);
        }

        let mut y = [0u8; 32];
        for b in 0..32 {
            // Horner: ((((a_{t-1} x + a_{t-2}) x + ...) x + a_0)
            let mut acc = coeffs[t - 1][b];
            for k in (0..(t - 1)).rev() {
                acc = gf256_add(gf256_mul(acc, x), coeffs[k][b]);
            }
            y[b] = acc;
        }
        out.push(ShamirShare {
            index: i as u32,
            value: y,
        });
    }
    Ok(out)
}

/// Reconstruct a 32-byte secret from any `cfg.threshold` shares using Lagrange interpolation at x=0.
pub fn reconstruct_secret_32(
    cfg: &ShamirConfig,
    shares: &[ShamirShare],
) -> Result<[u8; 32], ShamirError> {
    if cfg.threshold < 2 || cfg.shares < cfg.threshold || cfg.shares > 255 {
        return Err(ShamirError::InvalidParams);
    }
    if shares.len() < cfg.threshold {
        return Err(ShamirError::InsufficientShares);
    }

    // Take first t shares (caller typically filters/validates).
    let t = cfg.threshold;
    let xs: Vec<u8> = shares
        .iter()
        .take(t)
        .map(|s| {
            // Enforce spec: indices must be in [1..=N], and this GF(256) implementation only
            // supports x ∈ {1..=255}.
            if s.index == 0 || (s.index as usize) > cfg.shares || s.index > 255 {
                0
            } else {
                s.index as u8
            }
        })
        .collect();

    // Validate indices (nonzero, unique).
    for &x in &xs {
        if x == 0 {
            return Err(ShamirError::IndexOutOfRange);
        }
    }
    for i in 0..xs.len() {
        for j in (i + 1)..xs.len() {
            if xs[i] == xs[j] {
                return Err(ShamirError::DuplicateIndex);
            }
        }
    }

    // Lagrange basis at 0:
    // λ_i(0) = Π_{j≠i} x_j / (x_j - x_i)  in GF(256), where subtraction is XOR (same as add).
    //
    // Compute λ_i once (independent of the byte index), then apply to all 32 bytes.
    let mut lambdas: Vec<u8> = Vec::with_capacity(t);
    for i in 0..t {
        let xi = xs[i];
        let mut num = 1u8;
        let mut den = 1u8;
        for j in 0..t {
            if i == j {
                continue;
            }
            let xj = xs[j];
            num = gf256_mul(num, xj);
            den = gf256_mul(den, gf256_add(xj, xi)); // (xj - xi) == (xj + xi) in char-2
        }
        let li = gf256_mul(num, gf256_inv(den));
        lambdas.push(li);
    }

    let mut secret = [0u8; 32];
    for b in 0..32 {
        let mut acc = 0u8;
        for i in 0..t {
            let yi = shares[i].value[b];
            acc = gf256_add(acc, gf256_mul(lambdas[i], yi));
        }
        secret[b] = acc;
    }
    Ok(secret)
}

// ---------------------------------------------------------------------------
// WE lockset helper: reconstruction disambiguation via a single global tag
// ---------------------------------------------------------------------------

/// Canonical tag binding the combined key to a statement digest (combine-v1).
///
/// This is intentionally a **single global check**, not a per-lock oracle.
pub fn combined_key_tag_v1(combined_key32: &[u8; 32], stmt_digest: &[u8; 32]) -> [u8; 32] {
    let mut h = Sha256::new();
    h.update(b"LFP_ONEPROOF_COMBINED_TAG_V1");
    h.update(combined_key32);
    h.update(stmt_digest);
    h.finalize().into()
}

/// Recover the unique 32-byte combined key by enumerating share candidates and checking the global tag.
///
/// - `candidates_per_lock[i] = (share_index, candidates)` for lock `i`
/// - Only the **first `cfg.threshold` locks** are used (policy: avoid 2^K enumeration).
/// - `enum_cap` bounds the worst-case product of candidate set sizes.
pub fn reconstruct_combined_key_by_tag_v1_from_candidates(
    cfg: &ShamirConfig,
    stmt_digest: &[u8; 32],
    expected_tag: &[u8; 32],
    candidates_per_lock: &[(u32, Vec<[u8; 32]>)],
    enum_cap: u64,
) -> Result<[u8; 32], String> {
    let t = cfg.threshold;
    if t < 2 {
        return Err("combine: threshold < 2".to_string());
    }
    if candidates_per_lock.len() < t {
        return Err("combine: insufficient locks for threshold".to_string());
    }
    let subset = &candidates_per_lock[..t];

    // Fast path: all unambiguous → one reconstruction.
    let all_single = subset.iter().all(|(_idx, c)| c.len() == 1);
    if all_single {
        let selected: Vec<ShamirShare> = subset
            .iter()
            .map(|(idx, cands)| ShamirShare {
                index: *idx,
                value: cands[0],
            })
            .collect();
        let candidate = reconstruct_secret_32(cfg, &selected).map_err(|e| format!("{e}"))?;
        let tag = combined_key_tag_v1(&candidate, stmt_digest);
        if &tag == expected_tag {
            return Ok(candidate);
        }
        // Fall through to enumeration (should be rare but preserves behavior).
    }

    // Enumerate combinations; cap to avoid blowups.
    let mut total: u64 = 1;
    for (_idx, cands) in subset {
        total = total.saturating_mul(cands.len() as u64);
    }
    if total > enum_cap {
        return Err(format!(
            "combine: too many combinations in fallback enumeration (total={total} cap={enum_cap})"
        ));
    }

    let mut picked: Option<[u8; 32]> = None;
    for mut counter in 0u64..total {
        let mut selected: Vec<ShamirShare> = Vec::with_capacity(t);
        for (idx, cands) in subset {
            let n = cands.len() as u64;
            if n == 0 {
                return Err("combine: empty candidate set for some lock".to_string());
            }
            let choice = (counter % n) as usize;
            counter /= n;
            selected.push(ShamirShare {
                index: *idx,
                value: cands[choice],
            });
        }
        let candidate = match reconstruct_secret_32(cfg, &selected) {
            Ok(v) => v,
            Err(_) => continue,
        };
        let tag = combined_key_tag_v1(&candidate, stmt_digest);
        if &tag == expected_tag {
            if picked.is_some() {
                return Err("combine: multiple candidates matched tag (unexpected)".to_string());
            }
            picked = Some(candidate);
        }
    }

    picked.ok_or_else(|| "combine: failed to reconstruct combined key".to_string())
}

// ------------------------- GF(256) arithmetic -------------------------
// Field: GF(2^8) with irreducible polynomial x^8 + x^4 + x^3 + x + 1 (0x11b).

#[inline]
fn gf256_add(a: u8, b: u8) -> u8 {
    a ^ b
}

#[inline]
fn gf256_mul(mut a: u8, mut b: u8) -> u8 {
    let mut p: u8 = 0;
    for _ in 0..8 {
        if (b & 1) != 0 {
            p ^= a;
        }
        let hi = a & 0x80;
        a <<= 1;
        if hi != 0 {
            a ^= 0x1b; // reduction for 0x11b
        }
        b >>= 1;
    }
    p
}

#[inline]
fn gf256_pow(mut a: u8, mut e: u16) -> u8 {
    let mut r: u8 = 1;
    while e > 0 {
        if (e & 1) == 1 {
            r = gf256_mul(r, a);
        }
        a = gf256_mul(a, a);
        e >>= 1;
    }
    r
}

#[inline]
fn gf256_inv(a: u8) -> u8 {
    // a^(2^8-2) = a^254 for nonzero a
    // We rely on upstream validation to avoid inverting 0.
    gf256_pow(a, 254)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{rngs::StdRng, SeedableRng};

    #[test]
    fn test_split_reconstruct_32() {
        let mut rng = StdRng::seed_from_u64(123);
        let cfg = ShamirConfig {
            threshold: 5,
            shares: 20,
        };
        let secret = [7u8; 32];
        let shares = split_secret_32(&mut rng, &cfg, secret).unwrap();
        assert_eq!(shares.len(), cfg.shares);

        // Take any 5 shares.
        let rec = reconstruct_secret_32(&cfg, &shares[3..8]).unwrap();
        assert_eq!(rec, secret);
    }

    #[test]
    fn test_duplicate_index_reject() {
        let cfg = ShamirConfig {
            threshold: 2,
            shares: 3,
        };
        let s = ShamirShare {
            index: 1,
            value: [1u8; 32],
        };
        let bad = reconstruct_secret_32(&cfg, &[s.clone(), s]);
        assert!(matches!(bad, Err(ShamirError::DuplicateIndex)));
    }
}

