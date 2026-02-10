//! PxR lock sanity tests (no per-lock oracle model).
//!
//! These tests are **not** cryptographic proofs. They are regression/sanity checks for two
//! critical conditions used in the PxR security/liveness story:
//!
//! 1) **Ratio-class variability**: for `R=2`, we need to be able to find repetitions whose shifted
//!    accepting-set ratio classes differ (so per-channel intersections collapse deterministically).
//! 2) **Channel separation**: channel-specific domain separation must ensure channels do not
//!    accidentally share the same hidden query `q` (catastrophic: reveals `s1/s0` from hints).
//! 3) **Empirical non-leakage (weak)**: a simple likelihood-based heuristic should not predict the
//!    mod-257 scalar `s` from public hints `h = s*q` with meaningful advantage over 1/256.

#![cfg(feature = "we_gate")]

use ark_ff::{Field, PrimeField};
use dpp::dr1cs_flpcp::{Dr1csInstanceSparse, MulCode, MulCodeDr1csNpFlpcpSparse, TensorRsMulCode};
use dpp::SparseVec;
use dpp::theorem43::Theorem43Dpp;
use latticefold::transcript::poseidon::F257;
use rayon::prelude::*;

use latticefold_plus::lockable_ringlwe::PackedF257Block64;

const MOD_257: u16 = 257;
const PACK_D: usize = 64;

#[inline]
fn f_to_u16<F: PrimeField>(x: &F) -> u16 {
    // For F257 this fits in 16 bits; keep generic and reduce mod 257.
    (x.into_bigint().as_ref()[0] as u16) % MOD_257
}

#[inline]
fn mul_mod257(a: u16, b: u16) -> u16 {
    ((a as u32 * b as u32) % (MOD_257 as u32)) as u16
}

#[inline]
fn add_mod257(a: u16, b: u16) -> u16 {
    let s = a + b;
    if s >= MOD_257 { s - MOD_257 } else { s }
}

#[inline]
fn pow_mod257(mut a: u16, mut e: u16) -> u16 {
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
fn inv_mod257(a: u16) -> u16 {
    assert!(a % MOD_257 != 0);
    pow_mod257(a % MOD_257, 255)
}

/// Canonical ratio class: min(r, r^{-1}) where r = a1/a0 in F257*.
#[inline]
fn ratio_class(a0: F257, a1: F257) -> Option<u16> {
    let a0u = f_to_u16(&a0);
    let a1u = f_to_u16(&a1);
    if a0u == 0 || a1u == 0 {
        return None;
    }
    let r = mul_mod257(a1u, inv_mod257(a0u));
    let rinv = inv_mod257(r);
    Some(r.min(rinv))
}

fn tiny_dpp() -> Theorem43Dpp<F257, MulCodeDr1csNpFlpcpSparse<F257, TensorRsMulCode<F257>>> {
    // Tiny dR1CS over F257 with one constraint: z0 * z1 = z2.
    // Pad up to k so MulCode backend can run directly.
    let n_total = 3usize;
    let l_public = 1usize;
    let code = TensorRsMulCode::<F257>::new(2, 1).expect("tensor code");
    let k = code.dim_k();
    let mut a = vec![SparseVec::new(vec![(F257::ONE, 0)])];
    let mut b = vec![SparseVec::new(vec![(F257::ONE, 1)])];
    let mut c = vec![SparseVec::new(vec![(F257::ONE, 2)])];
    while a.len() < k {
        a.push(SparseVec::new(Vec::new()));
        b.push(SparseVec::new(Vec::new()));
        c.push(SparseVec::new(Vec::new()));
    }
    let inst = Dr1csInstanceSparse::<F257> { n: n_total, a, b, c };
    let flpcp = MulCodeDr1csNpFlpcpSparse::<F257, _>::new(inst, l_public, code)
        .expect("mulcode flpcp");
    Theorem43Dpp::<F257, _>::new(flpcp).expect("theorem43 new")
}

fn collect_q_blocks(
    dpp: &Theorem43Dpp<F257, MulCodeDr1csNpFlpcpSparse<F257, TensorRsMulCode<F257>>>,
    c_stmt: &[F257],
    x: &[F257],
    armer_secret: &[F257],
    block_id: usize,
    rep_id: u64,
) -> (Vec<(usize, PackedF257Block64)>, F257) {
    use std::collections::BTreeMap;

    let art = dpp
        .arm(c_stmt, x, armer_secret, block_id, rep_id)
        .expect("arm");
    let mut scratch = dpp.query_scratch();
    let pi_len = dpp.proof_len();
    let mut blocks: BTreeMap<usize, [u16; PACK_D]> = BTreeMap::new();
    let off = dpp
        .stream_query_terms_for_pi(
            x,
            &art.coins,
            &art.coeffs,
            &mut scratch,
            &mut |pi_idx, coeff| {
                assert!(pi_idx < pi_len, "q_pi idx out of range");
                let block = pi_idx / PACK_D;
                let pos = pi_idx % PACK_D;
                let row = blocks.entry(block).or_insert([0u16; PACK_D]);
                let v = f_to_u16(&coeff) % MOD_257;
                row[pos] = add_mod257(row[pos], v);
            },
        )
        .expect("stream_query_terms_for_pi");

    let q_blocks: Vec<(usize, PackedF257Block64)> = blocks
        .into_iter()
        .map(|(idx, row)| (idx, PackedF257Block64::from_dense_u16s(&row)))
        .collect();
    (q_blocks, off)
}

fn packed_block_to_pairs(blk: &PackedF257Block64) -> Vec<(usize, u16)> {
    match blk {
        PackedF257Block64::Sparse { entries } => entries
            .iter()
            .map(|(pos_flags, coeff)| {
                let pos = (pos_flags & 0x3f) as usize;
                let is_256 = ((pos_flags >> 6) & 1) != 0;
                let v = if is_256 { 256u16 } else { *coeff as u16 };
                (pos, v % MOD_257)
            })
            .filter(|(_p, v)| *v != 0)
            .collect(),
        PackedF257Block64::Dense { vals, is256_mask } => {
            let mut out = Vec::new();
            for i in 0..64 {
                let bit = (is256_mask >> i) & 1;
                let v = if bit != 0 { 256u16 } else { vals[i] as u16 };
                if v != 0 {
                    out.push((i, v % MOD_257));
                }
            }
            out
        }
    }
}

fn flatten_nonzero_coeffs(q_blocks: &[(usize, PackedF257Block64)]) -> Vec<u16> {
    let mut out = Vec::new();
    for (_bi, blk) in q_blocks {
        for (_pos, v) in packed_block_to_pairs(blk) {
            if v != 0 {
                out.push(v);
            }
        }
    }
    out
}

#[inline]
fn choose_tied_mode_det(freq: &[u32; 257], seed: u64) -> Option<u16> {
    let mut best_c = 0u32;
    for v in 1u16..=256u16 {
        let c = freq[v as usize];
        if c > best_c {
            best_c = c;
        }
    }
    if best_c == 0 {
        return None;
    }
    let mut ties: Vec<u16> = Vec::new();
    for v in 1u16..=256u16 {
        if freq[v as usize] == best_c {
            ties.push(v);
        }
    }
    let idx = (splitmix64(seed) as usize) % ties.len();
    Some(ties[idx])
}

#[inline]
fn hsig_seed(h: &HSig) -> u64 {
    // Deterministic seed from content (for sidecar diagnostics/tests).
    let mut acc: u64 = 0x9E37_79B9_7F4A_7C15;
    for &(b, p, v) in h {
        acc = splitmix64(acc ^ (b as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9));
        acc = splitmix64(acc ^ (p as u64).wrapping_mul(0x94D0_49BB_1331_11EB));
        acc = splitmix64(acc ^ ((v as u64) << 1));
    }
    acc
}

fn q_blocks_signature(q_blocks: &[(usize, PackedF257Block64)]) -> Vec<(usize, usize, u16)> {
    let mut out: Vec<(usize, usize, u16)> = Vec::new();
    for (block_idx, blk) in q_blocks {
        for (pos, v) in packed_block_to_pairs(blk) {
            if v != 0 {
                out.push((*block_idx, pos, v));
            }
        }
    }
    out.sort_unstable();
    out
}

#[inline]
fn wilson_upper_95(successes: u64, trials: u64) -> f64 {
    if trials == 0 {
        return 1.0;
    }
    let z = 1.959_963_984_540_054_f64; // 95% two-sided
    let n = trials as f64;
    let phat = (successes as f64) / n;
    let z2_over_n = (z * z) / n;
    let center = phat + (z * z) / (2.0 * n);
    let radius = z * ((phat * (1.0 - phat) / n) + (z * z) / (4.0 * n * n)).sqrt();
    (center + radius) / (1.0 + z2_over_n)
}

#[inline]
fn binomial_upper_tail_p_value(successes: u64, trials: u64, p0: f64) -> f64 {
    if trials == 0 {
        return 1.0;
    }
    if p0 <= 0.0 {
        return if successes == 0 { 1.0 } else { 0.0 };
    }
    if p0 >= 1.0 {
        return 1.0;
    }
    let n = trials as usize;
    let k = successes as usize;
    if k == 0 {
        return 1.0;
    }
    if k > n {
        return 0.0;
    }
    let q = 1.0 - p0;
    let mut pmf = q.powi(n as i32); // P[X=0]
    let mut cdf = pmf;
    for i in 0..(k - 1) {
        let num = (n - i) as f64;
        let den = (i + 1) as f64;
        pmf *= (num / den) * (p0 / q);
        cdf += pmf;
    }
    (1.0 - cdf).clamp(0.0, 1.0)
}

#[inline]
fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E3779B97F4A7C15);
    let mut z = x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

#[inline]
fn deterministic_s_from_rep_id(rep_id: u64) -> u16 {
    ((splitmix64(rep_id) & 0xff) as u16) + 1
}

fn train_logp_from_reps(
    dpp: &Theorem43Dpp<F257, MulCodeDr1csNpFlpcpSparse<F257, TensorRsMulCode<F257>>>,
    c_stmt: &[F257],
    x: &[F257],
    armer_secret: &[F257],
    block_id: usize,
    rep_start: u64,
    rep_count: u64,
) -> Result<Vec<f64>, String> {
    if rep_count == 0 {
        return Err("rep_count must be > 0".to_string());
    }
    let (counts, total) = (0..rep_count as usize)
        .into_par_iter()
        .map(|k| {
            let rep_id = rep_start + (k as u64);
            let (q_blocks, _off) = collect_q_blocks(dpp, c_stmt, x, armer_secret, block_id, rep_id);
            let mut local = [0u64; 257];
            let mut local_total = 0u64;
            for v in flatten_nonzero_coeffs(&q_blocks) {
                local[v as usize] += 1;
                local_total += 1;
            }
            (local, local_total)
        })
        .reduce(
            || ([0u64; 257], 0u64),
            |(mut a, ta), (b, tb)| {
                for i in 0..257 {
                    a[i] += b[i];
                }
                (a, ta + tb)
            },
        );

    if total == 0 {
        return Err("no training coefficients collected".to_string());
    }
    let alpha = 1.0f64;
    let denom = (total as f64) + alpha * 256.0;
    let mut logp = vec![0.0f64; 257];
    for v in 1..=256usize {
        let p = ((counts[v] as f64) + alpha) / denom;
        logp[v] = p.ln();
    }
    Ok(logp)
}

fn evaluate_guess_attack_over_reps(
    dpp: &Theorem43Dpp<F257, MulCodeDr1csNpFlpcpSparse<F257, TensorRsMulCode<F257>>>,
    c_stmt: &[F257],
    x: &[F257],
    armer_secret: &[F257],
    block_id: usize,
    rep_start: u64,
    rep_count: u64,
    logp: &[f64],
) -> (u64, u64) {
    (0..rep_count as usize)
        .into_par_iter()
        .map(|k| {
            let rep_id = rep_start + (k as u64);
            let (q_blocks, _off) = collect_q_blocks(dpp, c_stmt, x, armer_secret, block_id, rep_id);
            let q_vals = flatten_nonzero_coeffs(&q_blocks);
            if q_vals.len() < 64 {
                return (0u64, 0u64);
            }

            let s_true = deterministic_s_from_rep_id(rep_id);
            let mut h_vals: Vec<u16> = Vec::with_capacity(q_vals.len());
            for &q in &q_vals {
                h_vals.push(mul_mod257(s_true, q));
            }

            let mut best_s: u16 = 1;
            let mut best_score = f64::NEG_INFINITY;
            for s_guess in 1u16..=256u16 {
                let inv = inv_mod257(s_guess);
                let mut score = 0.0f64;
                for &h in &h_vals {
                    let q_prime = mul_mod257(h, inv);
                    if q_prime == 0 {
                        score += -1000.0;
                    } else {
                        score += logp[q_prime as usize];
                    }
                }
                if score > best_score {
                    best_score = score;
                    best_s = s_guess;
                }
            }
            ((best_s == s_true) as u64, 1u64)
        })
        .reduce(|| (0u64, 0u64), |(ca, ta), (cb, tb)| (ca + cb, ta + tb))
}

fn evaluate_mode_heuristic_over_reps(
    dpp: &Theorem43Dpp<F257, MulCodeDr1csNpFlpcpSparse<F257, TensorRsMulCode<F257>>>,
    c_stmt: &[F257],
    x: &[F257],
    armer_secret: &[F257],
    block_id: usize,
    rep_start: u64,
    rep_count: u64,
    mode_candidates: &[u16],
) -> (u64, u64) {
    (0..rep_count as usize)
        .into_par_iter()
        .map(|k| {
            let rep_id = rep_start + (k as u64);
            let (q_blocks, _off) = collect_q_blocks(dpp, c_stmt, x, armer_secret, block_id, rep_id);
            let q_vals = flatten_nonzero_coeffs(&q_blocks);
            if q_vals.len() < 64 {
                return (0u64, 0u64);
            }
            let s_true = deterministic_s_from_rep_id(rep_id);
            let mut h_vals: Vec<u16> = Vec::with_capacity(q_vals.len());
            for &q in &q_vals {
                h_vals.push(mul_mod257(s_true, q));
            }

            // Dumb heuristic #1: guess s from the most frequent nonzero hint coefficient.
            let mut freq = [0u32; 257];
            for &h in &h_vals {
                if h != 0 {
                    freq[h as usize] = freq[h as usize].saturating_add(1);
                }
            }
            let mode_h = match choose_tied_mode_det(&freq, rep_id ^ 0xA5A5_5A5A_D3C1_2B7Fu64) {
                Some(v) => v,
                None => return (0u64, 0u64),
            };

            // Optional small candidate set around mode_h, e.g. [1,2,255,256].
            let mut guessed = false;
            for &m in mode_candidates {
                let s_guess = mul_mod257(mode_h, m);
                if s_guess == s_true {
                    guessed = true;
                    break;
                }
            }
            (guessed as u64, 1u64)
        })
        .reduce(|| (0u64, 0u64), |(ca, ta), (cb, tb)| (ca + cb, ta + tb))
}

fn evaluate_small_coeff_prior_attack_over_reps(
    dpp: &Theorem43Dpp<F257, MulCodeDr1csNpFlpcpSparse<F257, TensorRsMulCode<F257>>>,
    c_stmt: &[F257],
    x: &[F257],
    armer_secret: &[F257],
    block_id: usize,
    rep_start: u64,
    rep_count: u64,
) -> (u64, u64) {
    (0..rep_count as usize)
        .into_par_iter()
        .map(|k| {
            let rep_id = rep_start + (k as u64);
            let (q_blocks, _off) = collect_q_blocks(dpp, c_stmt, x, armer_secret, block_id, rep_id);
            let q_vals = flatten_nonzero_coeffs(&q_blocks);
            if q_vals.len() < 64 {
                return (0u64, 0u64);
            }
            let s_true = deterministic_s_from_rep_id(rep_id);
            let mut h_vals: Vec<u16> = Vec::with_capacity(q_vals.len());
            for &q in &q_vals {
                h_vals.push(mul_mod257(s_true, q));
            }

            // Dumb heuristic #2: choose s maximizing a very crude prior that q=h/s
            // prefers coefficients close to {1,2,255,256}.
            let mut best_s = 1u16;
            let mut best_score = i64::MIN;
            for s_guess in 1u16..=256u16 {
                let inv = inv_mod257(s_guess);
                let mut score: i64 = 0;
                for &h in &h_vals {
                    let q_prime = mul_mod257(h, inv);
                    score += match q_prime {
                        1 | 256 => 3,
                        2 | 255 => 2,
                        3 | 254 => 1,
                        _ => 0,
                    };
                }
                if score > best_score {
                    best_score = score;
                    best_s = s_guess;
                }
            }
            ((best_s == s_true) as u64, 1u64)
        })
        .reduce(|| (0u64, 0u64), |(ca, ta), (cb, tb)| (ca + cb, ta + tb))
}

fn h_signature_for_rep_and_s(
    dpp: &Theorem43Dpp<F257, MulCodeDr1csNpFlpcpSparse<F257, TensorRsMulCode<F257>>>,
    c_stmt: &[F257],
    x: &[F257],
    armer_secret: &[F257],
    block_id: usize,
    rep_id: u64,
    s: u16,
) -> Vec<(usize, usize, u16)> {
    let (q_blocks, _off) = collect_q_blocks(dpp, c_stmt, x, armer_secret, block_id, rep_id);
    let mut out: Vec<(usize, usize, u16)> = Vec::new();
    for (b, blk) in &q_blocks {
        for (pos, qv) in packed_block_to_pairs(blk) {
            let hv = mul_mod257(s, qv);
            if hv != 0 {
                out.push((*b, pos, hv));
            }
        }
    }
    out.sort_unstable();
    out
}

fn constant_ratio_if_same_support(a: &[(usize, usize, u16)], b: &[(usize, usize, u16)]) -> Option<u16> {
    if a.len() != b.len() || a.is_empty() {
        return None;
    }
    let mut ratio: Option<u16> = None;
    for (ta, tb) in a.iter().zip(b.iter()) {
        if ta.0 != tb.0 || ta.1 != tb.1 {
            return None;
        }
        let av = ta.2;
        let bv = tb.2;
        if av == 0 || bv == 0 {
            return None;
        }
        let r = mul_mod257(bv, inv_mod257(av));
        match ratio {
            None => ratio = Some(r),
            Some(rr) if rr == r => {}
            Some(_) => return None,
        }
    }
    ratio
}

type HSig = Vec<(usize, usize, u16)>;

fn mode_from_hsig(h: &HSig) -> Option<u16> {
    if h.is_empty() {
        return None;
    }
    let mut freq = [0u32; 257];
    for &(_, _, v) in h {
        if v != 0 {
            freq[v as usize] = freq[v as usize].saturating_add(1);
        }
    }
    choose_tied_mode_det(&freq, hsig_seed(h))
}

fn small_coeff_prior_best_s_from_hsig(h: &HSig) -> Option<u16> {
    if h.is_empty() {
        return None;
    }
    let mut best_s = 1u16;
    let mut best_score = i64::MIN;
    for s_guess in 1u16..=256u16 {
        let inv = inv_mod257(s_guess);
        let mut score: i64 = 0;
        for &(_, _, hv) in h {
            let q_prime = mul_mod257(hv, inv);
            score += match q_prime {
                1 | 256 => 3,
                2 | 255 => 2,
                3 | 254 => 1,
                _ => 0,
            };
        }
        if score > best_score {
            best_score = score;
            best_s = s_guess;
        }
    }
    Some(best_s)
}

fn parse_sidecar_line(line: &str) -> Result<(u16, HSig), String> {
    // Format:
    //   s|block:pos:val,block:pos:val,...
    // Example:
    //   73|0:1:255,0:8:4,1:3:199
    let trimmed = line.trim();
    if trimmed.is_empty() {
        return Err("empty line".to_string());
    }
    let mut parts = trimmed.split('|');
    let s_part = parts
        .next()
        .ok_or_else(|| "missing s part".to_string())?;
    let tuples_part = parts
        .next()
        .ok_or_else(|| "missing tuple part".to_string())?;
    if parts.next().is_some() {
        return Err("too many '|' parts".to_string());
    }
    let s: u16 = s_part
        .parse::<u16>()
        .map_err(|e| format!("bad s parse: {e}"))?;
    if s == 0 || s > 256 {
        return Err("s must be in 1..=256".to_string());
    }
    let mut out: HSig = Vec::new();
    for tok in tuples_part.split(',') {
        let t = tok.trim();
        if t.is_empty() {
            continue;
        }
        let mut p = t.split(':');
        let b: usize = p
            .next()
            .ok_or_else(|| "missing block".to_string())?
            .parse::<usize>()
            .map_err(|e| format!("bad block parse: {e}"))?;
        let pos: usize = p
            .next()
            .ok_or_else(|| "missing pos".to_string())?
            .parse::<usize>()
            .map_err(|e| format!("bad pos parse: {e}"))?;
        let v: u16 = p
            .next()
            .ok_or_else(|| "missing val".to_string())?
            .parse::<u16>()
            .map_err(|e| format!("bad val parse: {e}"))?;
        if p.next().is_some() {
            return Err("tuple has too many ':' parts".to_string());
        }
        if v == 0 || v > 256 {
            return Err("val must be in 1..=256".to_string());
        }
        out.push((b, pos, v));
    }
    out.sort_unstable();
    Ok((s, out))
}

fn load_hsig_sidecar(path: &str) -> Result<Vec<(u16, HSig)>, String> {
    let data = std::fs::read_to_string(path).map_err(|e| format!("read sidecar failed: {e}"))?;
    let mut out: Vec<(u16, HSig)> = Vec::new();
    for (ln, line) in data.lines().enumerate() {
        let t = line.trim();
        if t.is_empty() || t.starts_with('#') {
            continue;
        }
        let parsed = parse_sidecar_line(t).map_err(|e| format!("line {}: {}", ln + 1, e))?;
        if parsed.1.len() >= 64 {
            out.push(parsed);
        }
    }
    if out.is_empty() {
        return Err("sidecar has no usable samples".to_string());
    }
    Ok(out)
}

#[test]
fn test_ratio_class_stats_are_healthy() {
    let dpp = tiny_dpp();
    // Same tiny satisfying assignment as in the old ratio-stats test.
    let x = vec![F257::from(2u64)];
    let c_stmt = vec![F257::from(99u64)];
    let armer_secret = vec![F257::from(7u64)];

    let n: u64 = std::env::var("DPP_RATIO_STATS_N")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(2048);

    let ratio_opts: Vec<Option<u16>> = (0..n as usize)
        .into_par_iter()
        .map(|k| {
            let rep_id = k as u64;
            let (_q, off) = collect_q_blocks(&dpp, &c_stmt, &x, &armer_secret, 0, rep_id);
            let a0 = F257::ONE - off;
            let a1 = F257::from(2u64) - off;
            ratio_class(a0, a1)
        })
        .collect();
    let rejects = ratio_opts.iter().filter(|v| v.is_none()).count() as u64;
    let ratio_classes: Vec<u16> = ratio_opts.into_iter().flatten().collect();
    let seen: std::collections::BTreeSet<u16> = ratio_classes.iter().copied().collect();

    // Coverage check: catch gross degeneracy/regressions in ratio classes.
    assert!(
        seen.len() >= 128,
        "ratio class set too small ({} distinct over {} reps; rejects={})",
        seen.len(),
        n,
        rejects
    );

    // among random triples of reps, count cases
    // where all three ratio classes are identical (indicates concentration in tiny subsets).
    let groups: usize = std::env::var("DPP_RATIO_GROUPS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(20_000);
    let m = ratio_classes.len();
    assert!(m >= 3, "not enough non-rejected ratios for triple sampling");
    let bad: usize = (0..groups as u64)
        .into_par_iter()
        .map(|g| {
            let i0 = (splitmix64(g ^ 0x1111_1111_1111_1111) as usize) % m;
            let i1 = (splitmix64(g ^ 0x2222_2222_2222_2222) as usize) % m;
            let i2 = (splitmix64(g ^ 0x3333_3333_3333_3333) as usize) % m;
            let r0 = ratio_classes[i0];
            let r1 = ratio_classes[i1];
            let r2 = ratio_classes[i2];
            (r0 == r1 && r1 == r2) as usize
        })
        .sum();
    // Very loose bound; this is a concentration regression guard, not a proof.
    // For near-uniform over ~128 classes, expected rate is around 1/128^2 ~= 6e-5.
    let bad_rate = (bad as f64) / (groups as f64);
    assert!(
        bad_rate <= 0.005,
        "ratio-class triple-collision too high: bad={} groups={} rate={:.6}",
        bad,
        groups,
        bad_rate
    );
}

#[test]
fn test_channel_separation_changes_q_blocks() {
    // This checks the *mechanism* used by OneProof: channel separation is achieved by deriving
    // channel-specific rep_id values. If this fails, two channels can accidentally share q.
    //
    // NOTE: This is a DPP-level sanity check; it doesn't rely on the full OneProof pipeline.
    let dpp = tiny_dpp();
    let x = vec![F257::from(2u64)];
    let c_stmt = vec![F257::from(99u64)];
    let armer_secret = vec![F257::from(7u64)];

    // Fixed "public" identifiers; channel separation is simulated by using different rep_id.
    let block_id = 0usize;
    let rep_id_ch0 = 1000u64;
    let rep_id_ch1 = 2000u64;

    let (q0, _off0) = collect_q_blocks(&dpp, &c_stmt, &x, &armer_secret, block_id, rep_id_ch0);
    let (q1, _off1) = collect_q_blocks(&dpp, &c_stmt, &x, &armer_secret, block_id, rep_id_ch1);

    // We only need "not identical" as a regression guard.
    // If they ever become identical, something is very wrong with transcript separation.
    assert_ne!(q0.len(), 0, "expected some q blocks");
    assert_ne!(q1.len(), 0, "expected some q blocks");
    let sig0 = q_blocks_signature(&q0);
    let sig1 = q_blocks_signature(&q1);
    assert_ne!(
        sig0, sig1,
        "q_blocks identical across channel-separated rep_id"
    );
}

#[test]
fn test_simple_guess_s_from_public_hints_has_no_obvious_advantage() {
    // Weak empirical test: use a crude likelihood model trained from q-coeff histograms.
    //
    // If this test ever starts failing, it indicates hint coefficients may be leaking s
    // (or the query distribution is extremely non-uniform in a way scaling preserves).
    let dpp = tiny_dpp();
    let x = vec![F257::from(2u64)];
    let c_stmt = vec![F257::from(99u64)];
    let armer_secret = vec![F257::from(7u64)];
    let block_id = 0usize;

    let train_n: u64 = std::env::var("DPP_S_LEAK_TRAIN_N")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(256);
    let test_n: u64 = std::env::var("DPP_S_LEAK_TEST_N")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(512);

    let logp = train_logp_from_reps(
        &dpp,
        &c_stmt,
        &x,
        &armer_secret,
        block_id,
        0,
        train_n,
    )
    .expect("train_logp_from_reps");

    // Test: attacker gets only h = s*q; tries all s' and picks max likelihood of q' = h/s'.
    let (correct, trials) = evaluate_guess_attack_over_reps(
        &dpp,
        &c_stmt,
        &x,
        &armer_secret,
        block_id,
        10_000,
        test_n,
        &logp,
    );
    assert!(trials > 0, "no leakage trials executed");

    // Baseline random guess success prob is 1/256 ≈ 0.00390625.
    // We set a very loose threshold to only catch glaring leakage.
    //
    // NOTE: If you tighten this later, make sure to increase test_n to reduce variance.
    let rate = (correct as f64) / (trials as f64);
    assert!(
        rate <= 0.02,
        "simple s-guess heuristic succeeded too often: rate={:.4} (correct={} / {}, requested_test_n={})",
        rate,
        correct,
        trials,
        test_n
    );
}

#[test]
#[ignore = "heavy empirical validation; run on large server"]
fn test_simple_guess_s_wilson_upper_bound_heavy() {
    let dpp = tiny_dpp();
    let x = vec![F257::from(2u64)];
    let c_stmt = vec![F257::from(99u64)];
    let armer_secret = vec![F257::from(7u64)];
    let block_id = 0usize;

    let train_n: u64 = std::env::var("DPP_S_LEAK_HEAVY_TRAIN_N")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(4096);
    let test_n: u64 = std::env::var("DPP_S_LEAK_HEAVY_TEST_N")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(8192);

    let logp = train_logp_from_reps(
        &dpp,
        &c_stmt,
        &x,
        &armer_secret,
        block_id,
        0,
        train_n,
    )
    .expect("train_logp_from_reps");

    let (correct, trials) = evaluate_guess_attack_over_reps(
        &dpp,
        &c_stmt,
        &x,
        &armer_secret,
        block_id,
        100_000,
        test_n,
        &logp,
    );
    assert!(trials > 0, "no heavy leakage trials executed");
    let p_hat = (correct as f64) / (trials as f64);
    let upper95 = wilson_upper_95(correct, trials);
    let baseline_single = 1.0f64 / 256.0;
    // tiny_dpp() can have mild synthetic-distribution skew; keep a conservative tolerance.
    let tolerated_baseline = baseline_single + 0.006;
    let alpha = 0.01f64;
    let p_upper = binomial_upper_tail_p_value(correct, trials, tolerated_baseline);
    assert!(
        p_upper >= alpha,
        "heavy leakage too strong: p_upper={:.6} alpha={:.6} p_hat={:.6} tolerated_baseline={:.6} raw_baseline={:.6} upper95={:.6} correct={} trials={}",
        p_upper,
        alpha,
        p_hat,
        tolerated_baseline,
        baseline_single,
        upper95,
        correct,
        trials
    );
}

#[test]
#[ignore = "heavy empirical validation; run on large server"]
fn test_dumb_mode_heuristics_do_not_beat_random_noticeably() {
    let dpp = tiny_dpp();
    let x = vec![F257::from(2u64)];
    let c_stmt = vec![F257::from(99u64)];
    let armer_secret = vec![F257::from(7u64)];
    let block_id = 0usize;

    let test_n: u64 = std::env::var("DPP_MODE_HEUR_TEST_N")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(65_536);

    // First run pure mode-h guess (candidate set = {1}).
    let (correct_mode, trials_mode) = evaluate_mode_heuristic_over_reps(
        &dpp,
        &c_stmt,
        &x,
        &armer_secret,
        block_id,
        200_000,
        test_n,
        &[1],
    );
    assert!(trials_mode > 0, "no trials for pure mode heuristic");
    let p_hat_mode = (correct_mode as f64) / (trials_mode as f64);
    let upper95_mode = wilson_upper_95(correct_mode, trials_mode);
    let baseline_single = 1.0f64 / 256.0;
    let tolerated_baseline_single = baseline_single + 0.004;
    let alpha = 0.01f64;
    let p_upper_mode =
        binomial_upper_tail_p_value(correct_mode, trials_mode, tolerated_baseline_single);
    assert!(
        p_upper_mode >= alpha,
        "pure mode heuristic too strong: p_upper={:.6} alpha={:.6} p_hat={:.6} tolerated_baseline={:.6} raw_baseline={:.6} upper95={:.6} correct={} trials={}",
        p_upper_mode,
        alpha,
        p_hat_mode,
        tolerated_baseline_single,
        baseline_single,
        upper95_mode,
        correct_mode,
        trials_mode
    );

    // Then run "mode times small set" heuristic from the suggestion.
    let (correct_smallset, trials_smallset) = evaluate_mode_heuristic_over_reps(
        &dpp,
        &c_stmt,
        &x,
        &armer_secret,
        block_id,
        300_000,
        test_n,
        &[1, 2, 255, 256],
    );
    assert!(trials_smallset > 0, "no trials for mode*smallset heuristic");
    let p_hat_smallset = (correct_smallset as f64) / (trials_smallset as f64);
    let upper95_smallset = wilson_upper_95(correct_smallset, trials_smallset);
    let baseline_four = 4.0f64 / 256.0;
    let p_upper_smallset =
        binomial_upper_tail_p_value(correct_smallset, trials_smallset, baseline_four);
    assert!(
        p_upper_smallset >= alpha,
        "mode*smallset heuristic too strong: p_upper={:.6} alpha={:.6} p_hat={:.6} baseline={:.6} upper95={:.6} correct={} trials={}",
        p_upper_smallset,
        alpha,
        p_hat_smallset,
        baseline_four,
        upper95_smallset,
        correct_smallset,
        trials_smallset
    );
}

#[test]
#[ignore = "heavy empirical validation; run on large server"]
fn test_small_coeff_prior_heuristic_does_not_beat_random_noticeably() {
    let dpp = tiny_dpp();
    let x = vec![F257::from(2u64)];
    let c_stmt = vec![F257::from(99u64)];
    let armer_secret = vec![F257::from(7u64)];
    let block_id = 0usize;

    let test_n: u64 = std::env::var("DPP_SMALLPRIOR_TEST_N")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(65_536);

    let (correct, trials) = evaluate_small_coeff_prior_attack_over_reps(
        &dpp,
        &c_stmt,
        &x,
        &armer_secret,
        block_id,
        400_000,
        test_n,
    );
    assert!(trials > 0, "no trials for small-coeff prior heuristic");
    let p_hat = (correct as f64) / (trials as f64);
    let upper95 = wilson_upper_95(correct, trials);
    let baseline_single = 1.0f64 / 256.0;
    let tolerated_baseline_single = baseline_single + 0.006;
    let alpha = 0.01f64;
    let p_upper = binomial_upper_tail_p_value(correct, trials, tolerated_baseline_single);
    assert!(
        p_upper >= alpha,
        "small-coeff prior heuristic too strong: p_upper={:.6} alpha={:.6} p_hat={:.6} tolerated_baseline={:.6} raw_baseline={:.6} upper95={:.6} correct={} trials={}",
        p_upper,
        alpha,
        p_hat,
        tolerated_baseline_single,
        baseline_single,
        upper95,
        correct,
        trials
    );
}

#[test]
#[ignore = "heavy empirical validation; run on large server"]
fn test_cross_channel_constant_ratio_redflag_is_rare() {
    let dpp = tiny_dpp();
    let x = vec![F257::from(2u64)];
    let c_stmt = vec![F257::from(99u64)];
    let armer_secret = vec![F257::from(7u64)];
    let block_id = 0usize;

    let n_pairs: u64 = std::env::var("DPP_CHANNEL_PAIR_N")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(65_536);

    let (equal_q, ratio_redflags, evaluated): (u64, u64, u64) = (0..n_pairs as usize)
        .into_par_iter()
        .map(|k| {
            let rep0 = 500_000u64 + (2 * k as u64);
            let rep1 = rep0 + 1;
            let s0 = deterministic_s_from_rep_id(rep0 ^ 0x9e37);
            let s1 = deterministic_s_from_rep_id(rep1 ^ 0x85eb);

            let q0 = q_blocks_signature(
                &collect_q_blocks(&dpp, &c_stmt, &x, &armer_secret, block_id, rep0).0,
            );
            let q1 = q_blocks_signature(
                &collect_q_blocks(&dpp, &c_stmt, &x, &armer_secret, block_id, rep1).0,
            );
            let eq_q = (q0 == q1) as u64;

            let h0 = h_signature_for_rep_and_s(&dpp, &c_stmt, &x, &armer_secret, block_id, rep0, s0);
            let h1 = h_signature_for_rep_and_s(&dpp, &c_stmt, &x, &armer_secret, block_id, rep1, s1);
            let red = constant_ratio_if_same_support(&h0, &h1).is_some() as u64;
            (eq_q, red, 1u64)
        })
        .reduce(
            || (0u64, 0u64, 0u64),
            |(a0, a1, a2), (b0, b1, b2)| (a0 + b0, a1 + b1, a2 + b2),
        );

    assert!(evaluated > 0, "no channel-pair trials executed");
    assert_eq!(
        equal_q, 0,
        "exact q equality across channel-style pairs is catastrophic (equal_q={} / {})",
        equal_q, evaluated
    );

    let redflag_rate = (ratio_redflags as f64) / (evaluated as f64);
    assert!(
        redflag_rate <= 0.001,
        "constant-ratio same-support redflag too frequent: redflags={} / {} (rate={:.6})",
        ratio_redflags,
        evaluated,
        redflag_rate
    );
}

#[test]
#[ignore = "requires DPP_PACKAGE_SIDECAR_PATH to evaluate production-like samples"]
fn test_package_sidecar_heuristics_if_provided() {
    let path = match std::env::var("DPP_PACKAGE_SIDECAR_PATH") {
        Ok(p) => p,
        Err(_) => {
            eprintln!(
                "[pxr] skipping package-sidecar heuristic test (set DPP_PACKAGE_SIDECAR_PATH)"
            );
            return;
        }
    };
    let samples = load_hsig_sidecar(&path).expect("load_hsig_sidecar");
    let n = samples.len() as u64;
    // For Wilson upper-bound assertions near 1/256-scale baselines, tiny sidecars are
    // statistically inconclusive. Return early instead of failing on noise-dominated bounds.
    let min_n: u64 = std::env::var("DPP_PACKAGE_SIDECAR_MIN_N")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(2048);
    if n < min_n {
        eprintln!(
            "[pxr] sidecar has {} usable samples (< {}), inconclusive for strict bounds; skipping",
            n, min_n
        );
        return;
    }

    // Heuristic A: pure mode guess.
    let mut correct_mode = 0u64;
    // Heuristic B: mode * small set.
    let mut correct_mode4 = 0u64;
    // Heuristic C: small-coeff prior exhaustive search over s.
    let mut correct_prior = 0u64;
    for (s_true, hsig) in &samples {
        if let Some(mode_h) = mode_from_hsig(hsig) {
            if mode_h == *s_true {
                correct_mode += 1;
            }
            let mut hit4 = false;
            for m in [1u16, 2u16, 255u16, 256u16] {
                if mul_mod257(mode_h, m) == *s_true {
                    hit4 = true;
                    break;
                }
            }
            if hit4 {
                correct_mode4 += 1;
            }
        }
        if let Some(best_s) = small_coeff_prior_best_s_from_hsig(hsig) {
            if best_s == *s_true {
                correct_prior += 1;
            }
        }
    }

    let up_mode = wilson_upper_95(correct_mode, n);
    let up_mode4 = wilson_upper_95(correct_mode4, n);
    let up_prior = wilson_upper_95(correct_prior, n);

    let baseline_single = 1.0f64 / 256.0;
    let baseline_four = 4.0f64 / 256.0;
    // One-sided null-hypothesis checks against random-guess baselines.
    // We only fail if the observed hit-count is unlikely under the baseline.
    let alpha = 0.01f64;
    let p_mode = binomial_upper_tail_p_value(correct_mode, n, baseline_single);
    let p_mode4 = binomial_upper_tail_p_value(correct_mode4, n, baseline_four);
    let p_prior = binomial_upper_tail_p_value(correct_prior, n, baseline_single);

    assert!(
        p_mode >= alpha,
        "sidecar pure-mode too strong: p_upper={:.6} alpha={:.6} observed_rate={:.6} baseline={:.6} upper95={:.6} correct={} n={}",
        p_mode,
        alpha,
        (correct_mode as f64) / (n as f64),
        baseline_single,
        up_mode,
        correct_mode,
        n
    );
    assert!(
        p_mode4 >= alpha,
        "sidecar mode*4 too strong: p_upper={:.6} alpha={:.6} observed_rate={:.6} baseline={:.6} upper95={:.6} correct={} n={}",
        p_mode4,
        alpha,
        (correct_mode4 as f64) / (n as f64),
        baseline_four,
        up_mode4,
        correct_mode4,
        n
    );
    assert!(
        p_prior >= alpha,
        "sidecar small-prior too strong: p_upper={:.6} alpha={:.6} observed_rate={:.6} baseline={:.6} upper95={:.6} correct={} n={}",
        p_prior,
        alpha,
        (correct_prior as f64) / (n as f64),
        baseline_single,
        up_prior,
        correct_prior,
        n
    );
}

