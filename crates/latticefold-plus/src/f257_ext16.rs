//! Degree-16 extension field over `F257`.
//!
//! Representation:
//! - polynomial basis over `F257`
//! - modulus `X^16 - 3`
//!
//! Rationale:
//! - `3` is a primitive element of `F257^*` (order `256`)
//! - for `q = 257` and `n = 16`, the binomial `X^n - a` is irreducible when `a`
//!   has order `256`, so `X^16 - 3` gives a true field extension of size `257^16`
//! - this gives a single monolithic message object with about `128.09` bits

use core::{
    fmt,
    ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign},
};

use num_bigint::BigUint;
use num_traits::{One, Zero};
use rand::RngCore;

pub const F257_EXT16_DEGREE: usize = 16;
const MOD_257: u16 = 257;
const MODULUS_BINOMIAL_C: u16 = 3;

#[derive(Clone, Copy, PartialEq, Eq, Default)]
pub struct F257Ext16 {
    pub coeffs: [u16; F257_EXT16_DEGREE],
}

impl fmt::Debug for F257Ext16 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("F257Ext16")
            .field(&self.coeffs.as_slice())
            .finish()
    }
}

impl F257Ext16 {
    pub const fn zero() -> Self {
        Self {
            coeffs: [0u16; F257_EXT16_DEGREE],
        }
    }

    pub const fn one() -> Self {
        let mut coeffs = [0u16; F257_EXT16_DEGREE];
        coeffs[0] = 1;
        Self { coeffs }
    }

    pub fn is_zero(&self) -> bool {
        self.coeffs.iter().all(|&x| x == 0)
    }

    pub fn from_f257(x: u16) -> Self {
        let mut coeffs = [0u16; F257_EXT16_DEGREE];
        coeffs[0] = x % MOD_257;
        Self { coeffs }
    }

    pub fn random(rng: &mut impl RngCore) -> Self {
        let mut coeffs = [0u16; F257_EXT16_DEGREE];
        for c in &mut coeffs {
            *c = (rng.next_u32() % (MOD_257 as u32)) as u16;
        }
        Self { coeffs }
    }

    pub fn from_u128_seed(seed: u128) -> Self {
        let mut coeffs = [0u16; F257_EXT16_DEGREE];
        let mut x = seed;
        for c in &mut coeffs {
            *c = (x % (MOD_257 as u128)) as u16;
            x /= MOD_257 as u128;
        }
        Self { coeffs }
    }

    pub fn to_u128_seed(&self) -> Result<u128, String> {
        let mut acc = 0u128;
        for &c in self.coeffs.iter().rev() {
            acc = acc
                .checked_mul(MOD_257 as u128)
                .ok_or_else(|| "F257Ext16 seed value does not fit in u128".to_string())?;
            acc = acc
                .checked_add(c as u128)
                .ok_or_else(|| "F257Ext16 seed value does not fit in u128".to_string())?;
        }
        Ok(acc)
    }

    pub fn to_bytes_fixed(&self) -> [u8; 32] {
        let mut out = [0u8; 32];
        for (i, &c) in self.coeffs.iter().enumerate() {
            let b = c.to_le_bytes();
            out[2 * i] = b[0];
            out[2 * i + 1] = b[1];
        }
        out
    }

    pub fn inverse(self) -> Option<Self> {
        if self.is_zero() {
            return None;
        }
        let q = BigUint::from(MOD_257 as u32);
        let exp = q.pow(F257_EXT16_DEGREE as u32) - BigUint::from(2u32);
        Some(self.pow_biguint(&exp))
    }

    fn pow_biguint(self, exp: &BigUint) -> Self {
        let mut base = self;
        let mut e = exp.clone();
        let mut out = Self::one();
        while !e.is_zero() {
            if (&e & BigUint::one()) == BigUint::one() {
                out *= base;
            }
            e >>= 1usize;
            if !e.is_zero() {
                base *= base;
            }
        }
        out
    }
}

#[inline]
fn add_mod(a: u16, b: u16) -> u16 {
    let s = (a as u32) + (b as u32);
    (s % (MOD_257 as u32)) as u16
}

#[inline]
fn sub_mod(a: u16, b: u16) -> u16 {
    if a >= b {
        a - b
    } else {
        ((a as u32) + (MOD_257 as u32) - (b as u32)) as u16
    }
}

impl Add for F257Ext16 {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        let mut out = [0u16; F257_EXT16_DEGREE];
        for i in 0..F257_EXT16_DEGREE {
            out[i] = add_mod(self.coeffs[i], rhs.coeffs[i]);
        }
        Self { coeffs: out }
    }
}

impl AddAssign for F257Ext16 {
    fn add_assign(&mut self, rhs: Self) {
        for i in 0..F257_EXT16_DEGREE {
            self.coeffs[i] = add_mod(self.coeffs[i], rhs.coeffs[i]);
        }
    }
}

impl Sub for F257Ext16 {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self::Output {
        let mut out = [0u16; F257_EXT16_DEGREE];
        for i in 0..F257_EXT16_DEGREE {
            out[i] = sub_mod(self.coeffs[i], rhs.coeffs[i]);
        }
        Self { coeffs: out }
    }
}

impl SubAssign for F257Ext16 {
    fn sub_assign(&mut self, rhs: Self) {
        for i in 0..F257_EXT16_DEGREE {
            self.coeffs[i] = sub_mod(self.coeffs[i], rhs.coeffs[i]);
        }
    }
}

impl Neg for F257Ext16 {
    type Output = Self;
    fn neg(self) -> Self::Output {
        let mut out = [0u16; F257_EXT16_DEGREE];
        for i in 0..F257_EXT16_DEGREE {
            out[i] = if self.coeffs[i] == 0 {
                0
            } else {
                MOD_257 - self.coeffs[i]
            };
        }
        Self { coeffs: out }
    }
}

impl Mul for F257Ext16 {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self::Output {
        let mut tmp = [0u32; 2 * F257_EXT16_DEGREE - 1];
        for i in 0..F257_EXT16_DEGREE {
            for j in 0..F257_EXT16_DEGREE {
                tmp[i + j] += (self.coeffs[i] as u32) * (rhs.coeffs[j] as u32);
            }
        }
        for k in (F257_EXT16_DEGREE..tmp.len()).rev() {
            let carry = tmp[k];
            if carry == 0 {
                continue;
            }
            tmp[k - F257_EXT16_DEGREE] += carry * (MODULUS_BINOMIAL_C as u32);
        }
        let mut out = [0u16; F257_EXT16_DEGREE];
        for i in 0..F257_EXT16_DEGREE {
            out[i] = (tmp[i] % (MOD_257 as u32)) as u16;
        }
        Self { coeffs: out }
    }
}

impl MulAssign for F257Ext16 {
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

#[cfg(test)]
mod tests {
    use rand::{rngs::StdRng, SeedableRng};

    use super::*;

    #[test]
    fn test_roundtrip_seed_u128() {
        let xs = [0u128, 1, 255, 256, 257, u64::MAX as u128, u128::MAX >> 1];
        for &x in &xs {
            let f = F257Ext16::from_u128_seed(x);
            let got = f.to_u128_seed().expect("seed fits");
            assert_eq!(got, x);
        }
    }

    #[test]
    fn test_inverse_nonzero() {
        let mut rng = StdRng::seed_from_u64(42);
        for _ in 0..8 {
            let mut x = F257Ext16::random(&mut rng);
            while x.is_zero() {
                x = F257Ext16::random(&mut rng);
            }
            let inv = x.inverse().expect("inverse");
            assert_eq!(x * inv, F257Ext16::one());
        }
    }
}
