//! Systematic Reed–Solomon utilities (prototype).
//!
//! We implement enough RS machinery to support the dR1CS FLPCP construction in Section 4.1:
//! - fixed evaluation points α_0..α_{ℓ-1}
//! - barycentric weights for fast evaluation from the first `k` (or `2k`) “systematic” values
//! - Lagrange coefficients at a given evaluation point

use ark_ff::{BigInteger, Field, FftField, PrimeField};
use ark_poly::{EvaluationDomain, Radix2EvaluationDomain};

use rayon::prelude::*;

/// Compute barycentric weights for distinct points `xs`.
///
/// w_i = 1 / ∏_{j≠i} (x_i - x_j)
pub fn barycentric_weights<F: Field>(xs: &[F]) -> Vec<F> {
    let n = xs.len();
    let mut w = vec![F::ZERO; n];
    for i in 0..n {
        let mut denom = F::ONE;
        for j in 0..n {
            if i == j {
                continue;
            }
            denom *= xs[i] - xs[j];
        }
        w[i] = denom.inverse().expect("distinct points => invertible denom");
    }
    w
}

/// Batch-invert nonzero elements in-place using 1 field inversion.
fn batch_inverse_in_place<F: Field>(v: &mut [F]) {
    let n = v.len();
    if n == 0 {
        return;
    }
    // prefix[i] = ∏_{j< i} v[j]
    let mut prefix = Vec::with_capacity(n);
    let mut acc = F::ONE;
    for x in v.iter() {
        prefix.push(acc);
        acc *= *x;
    }
    let mut inv_acc = acc.inverse().expect("batch inversion: nonzero product");
    for i in (0..n).rev() {
        let xi = v[i];
        v[i] = inv_acc * prefix[i];
        inv_acc *= xi;
    }
}

/// Compute barycentric weights for consecutive integer points.
///
/// Points are `xs[i] = start + i` interpreted in the field.
///
/// For consecutive points, we have the closed form:
/// \[
///   \prod_{j\neq i} (x_i - x_j) = i!\,(-1)^{n-1-i}\,(n-1-i)!
/// \]
/// so
/// \[
///   w_i = (-1)^{n-1-i} / (i!\,(n-1-i)!)
/// \]
///
/// This computes all weights in **O(n)** time using factorial and inverse factorial tables.
pub fn barycentric_weights_consecutive<F: Field>(n: usize, start: u64) -> Vec<F> {
    if n == 0 {
        return Vec::new();
    }
    // factorials up to n-1 in the field
    let mut fact = vec![F::ONE; n];
    for i in 1..n {
        fact[i] = fact[i - 1] * F::from(i as u64);
    }
    // inv factorials
    let mut inv_fact = vec![F::ONE; n];
    inv_fact[n - 1] = fact[n - 1].inverse().expect("field nonzero factorial invertible");
    for i in (1..n).rev() {
        inv_fact[i - 1] = inv_fact[i] * F::from(i as u64);
    }
    // weights
    let mut w = vec![F::ZERO; n];
    for i in 0..n {
        let mut wi = inv_fact[i] * inv_fact[n - 1 - i];
        // (-1)^{n-1-i}
        if ((n - 1 - i) & 1) == 1 {
            wi = -wi;
        }
        w[i] = wi;
    }
    // Touch `start` so callers can’t accidentally ignore it when passing other sequences.
    // (The weights for consecutive points are translation-invariant.)
    let _ = start;
    w
}

/// Evaluate the unique degree < `xs.len()` polynomial interpolating `(xs[i], ys[i])` at `x`.
///
/// Uses barycentric interpolation.
pub fn barycentric_eval<F: Field>(xs: &[F], ws: &[F], ys: &[F], x: F) -> F {
    debug_assert_eq!(xs.len(), ws.len());
    debug_assert_eq!(xs.len(), ys.len());
    // If x is one of the xs, return the corresponding y.
    for (xi, yi) in xs.iter().zip(ys.iter()) {
        if *xi == x {
            return *yi;
        }
    }
    let mut num = F::ZERO;
    let mut den = F::ZERO;
    for i in 0..xs.len() {
        let inv = (x - xs[i]).inverse().expect("x != xs[i] so invertible");
        let t = ws[i] * inv;
        num += t * ys[i];
        den += t;
    }
    num * den.inverse().expect("den != 0 for distinct points")
}

/// Compute Lagrange coefficients `λ_i(x)` for the basis over points `xs` evaluated at `x`.
///
/// Returns vector `lambda` such that for any `ys`, `f(x) = Σ_i lambda[i] * ys[i]`.
pub fn lagrange_coeffs_at<F: Field>(xs: &[F], ws: &[F], x: F) -> Vec<F> {
    debug_assert_eq!(xs.len(), ws.len());
    // If x == xs[i], lambda is unit vector.
    for (i, xi) in xs.iter().enumerate() {
        if *xi == x {
            let mut out = vec![F::ZERO; xs.len()];
            out[i] = F::ONE;
            return out;
        }
    }
    // Compute inv(x - x_i) via batch inversion (1 inversion total).
    let mut diffs = xs.iter().map(|xi| x - *xi).collect::<Vec<_>>();
    batch_inverse_in_place(&mut diffs);

    let mut denom = F::ZERO;
    let mut ts = Vec::with_capacity(xs.len());
    for i in 0..xs.len() {
        let t = ws[i] * diffs[i];
        ts.push(t);
        denom += t;
    }
    let denom_inv = denom.inverse().expect("den != 0");
    ts.into_iter().map(|t| t * denom_inv).collect()
}

/// Convolution (polynomial multiplication).
///
/// Uses radix-2 FFT when the field supports a large enough 2-adic domain; otherwise falls back
/// to the naive O(n^2) algorithm (only intended for small test sizes).
fn convolution<F: FftField + PrimeField>(a: &[F], b: &[F]) -> Vec<F> {
    if a.is_empty() || b.is_empty() {
        return Vec::new();
    }
    let out_len = a.len() + b.len() - 1;
    let size = out_len.next_power_of_two();
    let domain = Radix2EvaluationDomain::<F>::new(size);

    if let Some(domain) = domain {
        let mut fa = vec![F::ZERO; size];
        fa[..a.len()].copy_from_slice(a);
        let mut fb = vec![F::ZERO; size];
        fb[..b.len()].copy_from_slice(b);

        domain.fft_in_place(&mut fa);
        domain.fft_in_place(&mut fb);
        for i in 0..size {
            fa[i] *= fb[i];
        }
        domain.ifft_in_place(&mut fa);
        fa.truncate(out_len);
        fa
    } else {
        // No large 2-adic domain in this field (e.g. Frog prime has v2(p-1)=3).
        //
        // For large instances, we use CRT+NTT convolution over a few NTT-friendly primes
        // and reconstruct the exact integer coefficient within a known bound, then reduce mod p.
        //
        // This keeps the RS extrapolation scalable even when `F` itself is not FFT-friendly.
        if out_len > (1 << 16) {
            convolution_crt_ntt::<F>(a, b, out_len, size)
        } else {
            // Small-size fallback.
            let mut out = vec![F::ZERO; out_len];
            for (i, ai) in a.iter().enumerate() {
                for (j, bj) in b.iter().enumerate() {
                    out[i + j] += *ai * *bj;
                }
            }
            out
        }
    }
}

// ---------------- CRT+NTT convolution for 64-bit prime fields ----------------

#[inline]
fn mul_mod_u32(a: u32, b: u32, m: u32) -> u32 {
    ((a as u64 * b as u64) % (m as u64)) as u32
}

#[inline]
fn add_mod_u32(a: u32, b: u32, m: u32) -> u32 {
    let s = a as u64 + b as u64;
    (if s >= m as u64 { s - m as u64 } else { s }) as u32
}

#[inline]
fn sub_mod_u32(a: u32, b: u32, m: u32) -> u32 {
    (if a >= b { a - b } else { a + m - b }) as u32
}

fn pow_mod_u32(mut a: u32, mut e: u32, m: u32) -> u32 {
    let mut r: u32 = 1;
    while e > 0 {
        if (e & 1) == 1 {
            r = mul_mod_u32(r, a, m);
        }
        a = mul_mod_u32(a, a, m);
        e >>= 1;
    }
    r
}

fn inv_mod_u32(a: u32, m: u32) -> u32 {
    // m is prime
    pow_mod_u32(a, m - 2, m)
}

fn bit_reverse_permute(a: &mut [u32]) {
    let n = a.len();
    let mut j = 0usize;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if i < j {
            a.swap(i, j);
        }
    }
}

fn ntt(a: &mut [u32], invert: bool, modulus: u32, primitive_root: u32) {
    let n = a.len();
    bit_reverse_permute(a);

    // Compute n-th primitive root: g^{(p-1)/n}
    let root = pow_mod_u32(primitive_root, (modulus - 1) / (n as u32), modulus);
    let root_inv = inv_mod_u32(root, modulus);

    let mut len = 2usize;
    while len <= n {
        let wlen = if invert {
            pow_mod_u32(root_inv, (n / len) as u32, modulus)
        } else {
            pow_mod_u32(root, (n / len) as u32, modulus)
        };
        for i in (0..n).step_by(len) {
            let mut w: u32 = 1;
            for j in 0..(len / 2) {
                let u = a[i + j];
                let v = mul_mod_u32(a[i + j + len / 2], w, modulus);
                a[i + j] = add_mod_u32(u, v, modulus);
                a[i + j + len / 2] = sub_mod_u32(u, v, modulus);
                w = mul_mod_u32(w, wlen, modulus);
            }
        }
        len <<= 1;
    }

    if invert {
        let n_inv = inv_mod_u32(n as u32, modulus);
        for x in a.iter_mut() {
            *x = mul_mod_u32(*x, n_inv, modulus);
        }
    }
}

fn convolution_ntt_mod(a: &[u32], b: &[u32], out_len: usize, size: usize, modulus: u32, primitive_root: u32) -> Vec<u32> {
    let mut fa = vec![0u32; size];
    let mut fb = vec![0u32; size];
    fa[..a.len()].copy_from_slice(a);
    fb[..b.len()].copy_from_slice(b);

    ntt(&mut fa, false, modulus, primitive_root);
    ntt(&mut fb, false, modulus, primitive_root);
    for i in 0..size {
        fa[i] = mul_mod_u32(fa[i], fb[i], modulus);
    }
    ntt(&mut fa, true, modulus, primitive_root);
    fa.truncate(out_len);
    fa
}

/// CRT+NTT convolution fallback.
///
/// Uses 5 NTT-friendly 32-bit primes whose product is ~155 bits, enough to reconstruct
/// coefficients bounded by `O(n * p^2)` for our RS extrapolation use-case at ~1e6 scale.
fn convolution_crt_ntt<F: PrimeField>(a: &[F], b: &[F], out_len: usize, size: usize) -> Vec<F> {
    // Number of CRT moduli we use for the NTT-based convolution fallback.
    //
    // Note: This must be large enough that the product of moduli exceeds the worst-case integer
    // coefficient growth in RS extrapolation (roughly O(n * p^2) where p is the prime modulus of F).
    //
    // For ~64-bit prime fields at ~2^22 scale, 6 moduli (~180 bits product) is ample.
    const K: usize = 6;
    // NTT-friendly primes with primitive root 3 for these common choices.
    // All have at least 2^22 | (p-1) (i.e. support NTT sizes up to 2^22 and beyond).
    // Each modulus must support size=next_power_of_two(out_len), i.e. have v2(mod-1) >= log2(size).
    // We use a few standard NTT primes (cp-algorithms style). Product is ~230 bits, giving ample
    // headroom for exact reconstruction of large integer convolution coefficients before reducing mod p.
    const MODS: [u32; K] = [
        167_772_161,   // v2=25, primitive root 3
        754_974_721,   // v2=24, primitive root 11
        998_244_353,   // v2=23, primitive root 3
        469_762_049,   // v2=26, primitive root 3
        935_329_793,   // v2=22, primitive root 3
        2_013_265_921, // v2=27, primitive root 31
    ];
    const ROOTS: [u32; K] = [3, 11, 3, 3, 3, 31];

    // Extract modulus of F as u64 (Frog prime fits in u64).
    let p_bytes = F::MODULUS.to_bytes_le();
    let mut p_u64: u64 = 0;
    for (i, byte) in p_bytes.iter().enumerate().take(8) {
        p_u64 |= (*byte as u64) << (8 * i);
    }
    assert!(p_u64 > 1, "expected small (<=64-bit) prime modulus");

    // Map field elements to u64 in [0,p).
    let to_u64 = |x: &F| -> u64 {
        let xb = x.into_bigint().to_bytes_le();
        let mut v: u64 = 0;
        for (i, byte) in xb.iter().enumerate().take(8) {
            v |= (*byte as u64) << (8 * i);
        }
        v % p_u64
    };

    let a_u = a.par_iter().map(to_u64).collect::<Vec<_>>();
    let b_u = b.par_iter().map(to_u64).collect::<Vec<_>>();

    // For each modulus, reduce inputs and convolve.
    let residues_vec: Vec<Vec<u32>> = (0..K)
        .into_par_iter()
        .map(|i| {
            let modulus = MODS[i];
            let root = ROOTS[i];
            assert!(((modulus - 1) as usize) % size == 0, "NTT modulus does not support size {size}");
            let aa = a_u.iter().map(|&v| (v % (modulus as u64)) as u32).collect::<Vec<_>>();
            let bb = b_u.iter().map(|&v| (v % (modulus as u64)) as u32).collect::<Vec<_>>();
            convolution_ntt_mod(&aa, &bb, out_len, size, modulus, root)
        })
        .collect();

    let residues: [Vec<u32>; K] = residues_vec
        .try_into()
        .expect("residues_vec length matches CRT modulus count");

    // Precompute prefix products modulo each modulus and modulo p.
    let mut prefix_mod = [[0u32; K]; K];
    let mut inv_prefix = [0u32; K];
    for i in 0..K {
        let mi = MODS[i];
        let mut prod: u64 = 1;
        for j in 0..i {
            prefix_mod[i][j] = (prod % (mi as u64)) as u32;
            prod = (prod * MODS[j] as u64) % (mi as u64);
        }
        let prod_i = (prod % (mi as u64)) as u32;
        inv_prefix[i] = inv_mod_u32(prod_i, mi);
    }
    let mut prefix_p = [0u64; K];
    prefix_p[0] = 1;
    for i in 1..K {
        prefix_p[i] = (((prefix_p[i - 1] as u128) * (MODS[i - 1] as u128)) % (p_u64 as u128)) as u64;
    }

    // Garner reconstruction per coefficient, then reduce mod p and map back to F.
    let out: Vec<F> = (0..out_len)
        .into_par_iter()
        .map(|t| {
            // mixed radix digits c[i] in [0, MODS[i])
            let mut c = [0u32; K];
            c[0] = residues[0][t];
            for i in 1..K {
                let mi = MODS[i];
                let mut acc = residues[i][t];
                for j in 0..i {
                    let term = mul_mod_u32(c[j], prefix_mod[i][j], mi);
                    acc = sub_mod_u32(acc, term, mi);
                }
                c[i] = mul_mod_u32(acc, inv_prefix[i], mi);
            }

            // x mod p
            let mut x_mod_p: u64 = 0;
            for i in 0..K {
                let add = (((c[i] as u128) * (prefix_p[i] as u128)) % (p_u64 as u128)) as u64;
                x_mod_p = (((x_mod_p as u128) + (add as u128)) % (p_u64 as u128)) as u64;
            }
            F::from_le_bytes_mod_order(&x_mod_p.to_le_bytes())
        })
        .collect();

    out
}

/// Extrapolate a degree < n polynomial given its values at 0..n-1 to its values at n..2n-1.
///
/// This is the “next block” extrapolation for consecutive integer points, computed in
/// **O(n log n)** via one convolution over the field (requires `F: FftField`).
///
/// Input: `y[i] = f(i)` for i=0..n-1.
/// Output: `out[s] = f(n+s)` for s=0..n-1.
pub fn extrapolate_consecutive_next_block<F: FftField + PrimeField>(y: &[F]) -> Vec<F> {
    let n = y.len();
    if n == 0 {
        return Vec::new();
    }

    // factorials and inv factorials up to 2n-1
    let mut fact = vec![F::ONE; 2 * n];
    for i in 1..2 * n {
        fact[i] = fact[i - 1] * F::from(i as u64);
    }
    let mut inv_fact = vec![F::ONE; 2 * n];
    inv_fact[2 * n - 1] = fact[2 * n - 1].inverse().expect("fact nonzero");
    for i in (1..2 * n).rev() {
        inv_fact[i - 1] = inv_fact[i] * F::from(i as u64);
    }

    // c_i = y_i * (-1)^{n-1-i} / (i! (n-1-i)!)
    let mut c = vec![F::ZERO; n];
    for i in 0..n {
        let mut ci = y[i] * inv_fact[i] * inv_fact[n - 1 - i];
        if ((n - 1 - i) & 1) == 1 {
            ci = -ci;
        }
        c[i] = ci;
    }

    // b_t = 1/(t+1) for t=0..(2n-2), i.e. inv(1..2n-1)
    let mut b = (1..=(2 * n - 1)).map(|t| F::from(t as u64)).collect::<Vec<_>>();
    batch_inverse_in_place(&mut b);

    // conv = c * b; then S_s = conv[n-1+s]
    let conv = convolution::<F>(&c, &b);

    let mut out = vec![F::ZERO; n];
    for s in 0..n {
        // P(n+s) = (n+s)! / s!
        out[s] = fact[n + s] * inv_fact[s] * conv[n - 1 + s];
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_ff::{Fp64, MontBackend, MontConfig};
    use rand::{RngCore, SeedableRng};
    use rand_chacha::ChaCha20Rng;

    // Frog prime: 15912092521325583641
    #[derive(MontConfig)]
    #[modulus = "15912092521325583641"]
    #[generator = "7"]
    pub struct FrogPrimeConfig;
    type FFrog = Fp64<MontBackend<FrogPrimeConfig, 1>>;

    #[test]
    fn test_convolution_crt_ntt_matches_naive_mod_p() {
        // Moderate sizes: big enough to exercise NTT path, small enough for naive O(n^2).
        let n = 2048usize;
        let m = 1536usize;
        let out_len = n + m - 1;
        let size = out_len.next_power_of_two();

        let mut rng = ChaCha20Rng::seed_from_u64(12345);
        let a = (0..n).map(|_| FFrog::from(rng.next_u64())).collect::<Vec<_>>();
        let b = (0..m).map(|_| FFrog::from(rng.next_u64())).collect::<Vec<_>>();

        let got = convolution_crt_ntt::<FFrog>(&a, &b, out_len, size);

        // Naive integer convolution mod p.
        let p = 15_912_092_521_325_583_641u64;
        let to_u64 = |x: &FFrog| -> u64 {
            let bytes = x.into_bigint().to_bytes_le();
            let mut v: u64 = 0;
            for (i, byte) in bytes.iter().enumerate().take(8) {
                v |= (*byte as u64) << (8 * i);
            }
            v % p
        };
        let a_u = a.iter().map(to_u64).collect::<Vec<_>>();
        let b_u = b.iter().map(to_u64).collect::<Vec<_>>();

        let mut exp_u = vec![0u64; out_len];
        for i in 0..n {
            let ai = a_u[i] as u128;
            for j in 0..m {
                exp_u[i + j] = ((exp_u[i + j] as u128 + ai * (b_u[j] as u128)) % (p as u128)) as u64;
            }
        }
        let exp = exp_u
            .iter()
            .map(|&v| FFrog::from_le_bytes_mod_order(&v.to_le_bytes()))
            .collect::<Vec<_>>();

        assert_eq!(got, exp);
    }
}


