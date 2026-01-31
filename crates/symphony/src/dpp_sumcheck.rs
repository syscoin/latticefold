//! Sumcheck verifier arithmetization (sparse dR1CS over a prime field).
//!
//! This module is an intermediate building block toward arithmetizing the full Π_fold verifier.
//! It encodes **verification** of the multilinear sumcheck proof messages as constraints.
//!
//! Current scope:
//! - degree-3 prover messages (4 evaluations at points 0,1,2,3), matching the usage in Symphony.
//! - shared-randomness schedule for two sumchecks (same `r_i` per round).
//!
//! Note: this does **not** (yet) encode the post-sumcheck algebraic checks (Eq(26), monomial
//! recomputation, Step-5). Those are the next layer on top of the verified subclaims.

use ark_ff::PrimeField;
use ark_serialize::CanonicalSerialize;

use std::collections::BTreeMap;
use core::ops::Range;

use crate::dpp_poseidon::{Constraint, SparseDr1csInstance};
use crate::file_backed_dr1cs::{FileBackedSparseDr1csInstance, SparseDr1csFileWriter};

#[derive(Clone, Debug, Default)]
pub struct Dr1csProfileCounts {
    pub vars: u64,
    pub constraints: u64,
}

#[derive(Debug)]
pub struct Dr1csBuilder<F: PrimeField> {
    pub assignment: Vec<F>,
    pub rows: Vec<Constraint>,
    pub a_terms: Vec<(F, usize)>,
    pub b_terms: Vec<(F, usize)>,
    pub c_terms: Vec<(F, usize)>,
    /// Optional file-backed sink for constraints/term pools.
    ///
    /// When present, `rows/a_terms/b_terms/c_terms` are not populated (to avoid giant Vec pools).
    file_sink: Option<SparseDr1csFileWriter<F>>,
    file_rows: u64,
    file_a_terms: u64,
    file_b_terms: u64,
    file_c_terms: u64,
    /// Cache for reusing a byte var's bit-decomposition across gadgets.
    ///
    /// Key: byte variable index. Value: 8 boolean bit variable indices (little-endian).
    pub byte_bits_cache: BTreeMap<usize, [usize; 8]>,

    /// Cache for reusing `u64_bytes_to_bal16_digits` expansions across gadgets.
    ///
    /// Key: 8 byte variables (little-endian). Value: balanced-base16 digits (len 17).
    pub u64_bal16_cache: BTreeMap<[usize; 8], Vec<usize>>,
    /// Cache for reusing `u32_bytes_to_bal16_digits` expansions across gadgets.
    ///
    /// Key: 4 byte variables (little-endian). Value: balanced-base16 digits (len 9).
    pub u32_bal16_cache: BTreeMap<[usize; 4], Vec<usize>>,

    /// Cache for a single variable constrained to 0.
    ///
    /// Many gadgets need a reusable 0 variable; allocating it once avoids repeated boolean
    /// decompositions for constant-zero digits/limbs.
    pub zero_var_cache: Option<usize>,

    // -------------------------------------------------------------------------
    // Optional profiling (enabled by env var; low overhead when disabled).
    // -------------------------------------------------------------------------
    pub profile_enabled: bool,
    pub profile_current: Option<&'static str>,
    pub profile: BTreeMap<&'static str, Dr1csProfileCounts>,
}

impl<F: PrimeField + CanonicalSerialize> Dr1csBuilder<F> {
    pub fn new() -> Self {
        let profile_enabled = match std::env::var("LF_PROFILE_DR1CS") {
            Ok(v) => v != "0",
            Err(_) => false,
        };
        Self {
            assignment: vec![F::ONE],
            rows: Vec::new(),
            a_terms: Vec::new(),
            b_terms: Vec::new(),
            c_terms: Vec::new(),
            file_sink: None,
            file_rows: 0,
            file_a_terms: 0,
            file_b_terms: 0,
            file_c_terms: 0,
            byte_bits_cache: BTreeMap::new(),
            u64_bal16_cache: BTreeMap::new(),
            u32_bal16_cache: BTreeMap::new(),
            zero_var_cache: None,
            profile_enabled,
            profile_current: None,
            profile: BTreeMap::new(),
        }
    }
    /// Create a builder that streams constraints/terms to `dir` instead of storing giant Vec pools.
    pub fn new_file_backed(dir: impl AsRef<std::path::Path>) -> Result<Self, String>
    where
        F: CanonicalSerialize,
    {
        let mut b = Self::new();
        b.file_sink = Some(SparseDr1csFileWriter::<F>::create(dir)?);
        Ok(b)
    }

    #[inline]
    pub fn is_file_backed(&self) -> bool {
        self.file_sink.is_some()
    }

    #[inline]
    pub fn nconstraints(&self) -> u64 {
        if self.file_sink.is_some() {
            self.file_rows
        } else {
            self.rows.len() as u64
        }
    }
    pub fn one(&self) -> usize { 0 }
    pub fn new_var(&mut self, value: F) -> usize {
        let idx = self.assignment.len();
        self.assignment.push(value);
        if self.profile_enabled {
            let key = self.profile_current.unwrap_or("unlabeled");
            self.profile.entry(key).or_default().vars += 1;
        }
        idx
    }
    #[inline]
    fn push_terms(pool: &mut Vec<(F, usize)>, terms: &[(F, usize)]) -> Range<usize> {
        let start = pool.len();
        pool.extend_from_slice(terms);
        let end = pool.len();
        start..end
    }

    pub fn add_constraint(&mut self, a: Vec<(F, usize)>, b: Vec<(F, usize)>, c: Vec<(F, usize)>) {
        self.add_constraint_slices(&a, &b, &c);
        if self.profile_enabled {
            let key = self.profile_current.unwrap_or("unlabeled");
            self.profile.entry(key).or_default().constraints += 1;
        }
    }

    /// Add a constraint without materializing term vectors (stream-friendly).
    ///
    /// This is used by file-backed lowering paths to avoid allocating a `Vec` per constraint.
    #[inline]
    pub fn add_constraint_terms_iter<IA, IB, IC>(&mut self, a: IA, b: IB, c: IC)
    where
        IA: IntoIterator<Item = (F, usize)>,
        IB: IntoIterator<Item = (F, usize)>,
        IC: IntoIterator<Item = (F, usize)>,
    {
        if let Some(sink) = self.file_sink.as_mut() {
            let a0 = self.file_a_terms;
            let mut a_n: u64 = 0;
            for (coef, idx) in a.into_iter() {
                sink.push_a_term(&coef, idx as u64)
                    .expect("file-backed dr1cs write failed (a_term)");
                a_n += 1;
            }
            let a1 = a0 + a_n;
            self.file_a_terms = a1;

            let b0 = self.file_b_terms;
            let mut b_n: u64 = 0;
            for (coef, idx) in b.into_iter() {
                sink.push_b_term(&coef, idx as u64)
                    .expect("file-backed dr1cs write failed (b_term)");
                b_n += 1;
            }
            let b1 = b0 + b_n;
            self.file_b_terms = b1;

            let c0 = self.file_c_terms;
            let mut c_n: u64 = 0;
            for (coef, idx) in c.into_iter() {
                sink.push_c_term(&coef, idx as u64)
                    .expect("file-backed dr1cs write failed (c_term)");
                c_n += 1;
            }
            let c1 = c0 + c_n;
            self.file_c_terms = c1;

            sink.push_constraint_row(a0, a1, b0, b1, c0, c1)
                .expect("file-backed dr1cs write failed (constraint)");
            self.file_rows += 1;
        } else {
            let a0 = self.a_terms.len();
            self.a_terms.extend(a);
            let a1 = self.a_terms.len();
            let b0 = self.b_terms.len();
            self.b_terms.extend(b);
            let b1 = self.b_terms.len();
            let c0 = self.c_terms.len();
            self.c_terms.extend(c);
            let c1 = self.c_terms.len();
            self.rows.push(Constraint { a: a0..a1, b: b0..b1, c: c0..c1 });
        }
        if self.profile_enabled {
            let key = self.profile_current.unwrap_or("unlabeled");
            self.profile.entry(key).or_default().constraints += 1;
        }
    }

    #[inline]
    pub fn add_constraint_slices(&mut self, a: &[(F, usize)], b: &[(F, usize)], c: &[(F, usize)]) {
        if let Some(sink) = self.file_sink.as_mut() {
            // Stream to disk.
            let a0 = self.file_a_terms;
            for (coef, idx) in a.iter() {
                sink.push_a_term(coef, *idx as u64)
                    .expect("file-backed dr1cs write failed (a_term)");
            }
            let a1 = self.file_a_terms + (a.len() as u64);
            self.file_a_terms = a1;

            let b0 = self.file_b_terms;
            for (coef, idx) in b.iter() {
                sink.push_b_term(coef, *idx as u64)
                    .expect("file-backed dr1cs write failed (b_term)");
            }
            let b1 = self.file_b_terms + (b.len() as u64);
            self.file_b_terms = b1;

            let c0 = self.file_c_terms;
            for (coef, idx) in c.iter() {
                sink.push_c_term(coef, *idx as u64)
                    .expect("file-backed dr1cs write failed (c_term)");
            }
            let c1 = self.file_c_terms + (c.len() as u64);
            self.file_c_terms = c1;

            sink.push_constraint_row(a0, a1, b0, b1, c0, c1)
                .expect("file-backed dr1cs write failed (constraint)");
            self.file_rows += 1;
        } else {
            let ar = Self::push_terms(&mut self.a_terms, a);
            let br = Self::push_terms(&mut self.b_terms, b);
            let cr = Self::push_terms(&mut self.c_terms, c);
            self.rows.push(Constraint { a: ar, b: br, c: cr });
        }
    }
    pub fn enforce_lc_times_one_eq_const(&mut self, lc: Vec<(F, usize)>) {
        let one = self.one();
        self.add_constraint_slices(&lc, &[(F::ONE, one)], &[(F::ZERO, one)]);
        if self.profile_enabled {
            let key = self.profile_current.unwrap_or("unlabeled");
            self.profile.entry(key).or_default().constraints += 1;
        }
    }
    pub fn enforce_var_eq_const(&mut self, x: usize, c: F) {
        let one = self.one();
        self.add_constraint_slices(&[(F::ONE, x)], &[(F::ONE, one)], &[(c, one)]);
        if self.profile_enabled {
            let key = self.profile_current.unwrap_or("unlabeled");
            self.profile.entry(key).or_default().constraints += 1;
        }
    }
    pub fn enforce_mul(&mut self, x: usize, y: usize, out: usize) {
        self.add_constraint_slices(&[(F::ONE, x)], &[(F::ONE, y)], &[(F::ONE, out)]);
        if self.profile_enabled {
            let key = self.profile_current.unwrap_or("unlabeled");
            self.profile.entry(key).or_default().constraints += 1;
        }
    }

    /// Return a reusable variable constrained to 0.
    pub fn zero_var(&mut self) -> usize {
        if let Some(v) = self.zero_var_cache {
            return v;
        }
        let v = self.new_var(F::ZERO);
        self.enforce_var_eq_const(v, F::ZERO);
        self.zero_var_cache = Some(v);
        v
    }
    pub fn into_instance(self) -> (SparseDr1csInstance<F>, Vec<F>) {
        if self.file_sink.is_some() {
            panic!("into_instance called on file-backed Dr1csBuilder; use into_file_backed_instance");
        }
        let inst = SparseDr1csInstance {
            nvars: self.assignment.len(),
            constraints: self.rows,
            a_terms: self.a_terms,
            b_terms: self.b_terms,
            c_terms: self.c_terms,
        };
        (inst, self.assignment)
    }

    pub fn into_file_backed_instance(self) -> Result<(FileBackedSparseDr1csInstance<F>, Vec<F>), String>
    where
        F: CanonicalSerialize,
    {
        let mut me = self;
        let sink = me
            .file_sink
            .take()
            .ok_or_else(|| "into_file_backed_instance called on in-memory Dr1csBuilder".to_string())?;
        let inst = sink.finish(me.assignment.len())?;
        Ok((inst, me.assignment))
    }

    /// Enter a profiling scope; returns the previous scope label.
    ///
    /// Use `profile_exit(prev)` to restore.
    #[inline]
    pub fn profile_enter(&mut self, label: &'static str) -> Option<&'static str> {
        if !self.profile_enabled {
            return None;
        }
        let prev = self.profile_current;
        self.profile_current = Some(label);
        prev
    }

    #[inline]
    pub fn profile_exit(&mut self, prev: Option<&'static str>) {
        if !self.profile_enabled {
            return;
        }
        self.profile_current = prev;
    }

    pub fn profile_report(&self, top_n: usize) -> String {
        if !self.profile_enabled {
            return "LF_PROFILE_DR1CS disabled".to_string();
        }
        let mut total_vars: u64 = 0;
        let mut total_constraints: u64 = 0;
        for v in self.profile.values() {
            total_vars += v.vars;
            total_constraints += v.constraints;
        }
        let mut rows: Vec<(&'static str, Dr1csProfileCounts)> = self
            .profile
            .iter()
            .map(|(&k, v)| (k, v.clone()))
            .collect();
        rows.sort_by_key(|(_k, v)| std::cmp::Reverse(v.constraints));

        let mut out = String::new();
        out.push_str(&format!(
            "== dR1CS profile (LF_PROFILE_DR1CS=1): total vars={} total constraints={} ==\n",
            total_vars, total_constraints
        ));
        out.push_str("top scopes by constraints:\n");
        for (i, (k, v)) in rows.into_iter().take(top_n).enumerate() {
            let pct = if total_constraints == 0 {
                0.0
            } else {
                (v.constraints as f64) * 100.0 / (total_constraints as f64)
            };
            out.push_str(&format!(
                "  {:>2}. {:<40}  constraints={:<12} vars={:<12} ({:>5.1}%)\n",
                i + 1,
                k,
                v.constraints,
                v.vars,
                pct
            ));
        }
        out
    }
}

/// A "ring element" represented as `d` prime-field variables (coefficients).
#[derive(Clone, Debug)]
pub struct RingVars {
    pub coeffs: Vec<usize>,
}

impl RingVars {
    pub fn new(coeffs: Vec<usize>) -> Self { Self { coeffs } }
    pub fn d(&self) -> usize { self.coeffs.len() }
}

fn ring_add<F: PrimeField>(b: &mut Dr1csBuilder<F>, x: &RingVars, y: &RingVars) -> RingVars {
    assert_eq!(x.d(), y.d());
    let mut out = Vec::with_capacity(x.d());
    for i in 0..x.d() {
        let val = b.assignment[x.coeffs[i]] + b.assignment[y.coeffs[i]];
        let v = b.new_var(val);
        b.add_constraint(
            vec![(F::ONE, x.coeffs[i]), (F::ONE, y.coeffs[i])],
            vec![(F::ONE, b.one())],
            vec![(F::ONE, v)],
        );
        out.push(v);
    }
    RingVars::new(out)
}

fn ring_scale<F: PrimeField>(b: &mut Dr1csBuilder<F>, x: &RingVars, s: usize) -> RingVars {
    let mut out = Vec::with_capacity(x.d());
    for i in 0..x.d() {
        let val = b.assignment[x.coeffs[i]] * b.assignment[s];
        let v = b.new_var(val);
        b.enforce_mul(x.coeffs[i], s, v);
        out.push(v);
    }
    RingVars::new(out)
}

fn scalar_sub_const<F: PrimeField>(b: &mut Dr1csBuilder<F>, r: usize, c: F) -> usize {
    let val = b.assignment[r] - c;
    let v = b.new_var(val);
    b.add_constraint(vec![(F::ONE, r), (-c, b.one())], vec![(F::ONE, b.one())], vec![(F::ONE, v)]);
    v
}

fn scalar_mul<F: PrimeField>(b: &mut Dr1csBuilder<F>, x: usize, y: usize) -> usize {
    let val = b.assignment[x] * b.assignment[y];
    let v = b.new_var(val);
    b.enforce_mul(x, y, v);
    v
}

/// Degree-3 Lagrange interpolation at `r` for points 0,1,2,3.
///
/// Returns scalar variables `(L0(r), L1(r), L2(r), L3(r))`.
fn lagrange_degree3<F: PrimeField>(b: &mut Dr1csBuilder<F>, r: usize) -> (usize, usize, usize, usize) {
    let inv2 = F::from(2u64).inverse().unwrap();
    let inv6 = F::from(6u64).inverse().unwrap();

    // t1 = r-1, t2 = r-2, t3 = r-3
    let t1 = scalar_sub_const(b, r, F::ONE);
    let t2 = scalar_sub_const(b, r, F::from(2u64));
    let t3 = scalar_sub_const(b, r, F::from(3u64));

    // L0 = -(t1*t2*t3)/6
    let p12 = scalar_mul(b, t1, t2);
    let p123 = scalar_mul(b, p12, t3);
    let l0_val = -(b.assignment[p123] * inv6);
    let l0 = b.new_var(l0_val);
    // l0 = (-inv6)*p123
    b.add_constraint(vec![(-inv6, p123)], vec![(F::ONE, b.one())], vec![(F::ONE, l0)]);

    // L1 = r*(r-2)*(r-3)/2
    let p = scalar_mul(b, r, t2);
    let p = scalar_mul(b, p, t3);
    let l1_val = b.assignment[p] * inv2;
    let l1 = b.new_var(l1_val);
    b.add_constraint(vec![(inv2, p)], vec![(F::ONE, b.one())], vec![(F::ONE, l1)]);

    // L2 = -r*(r-1)*(r-3)/2
    let p = scalar_mul(b, r, t1);
    let p = scalar_mul(b, p, t3);
    let l2_val = -(b.assignment[p] * inv2);
    let l2 = b.new_var(l2_val);
    b.add_constraint(vec![(-inv2, p)], vec![(F::ONE, b.one())], vec![(F::ONE, l2)]);

    // L3 = r*(r-1)*(r-2)/6
    let p = scalar_mul(b, r, t1);
    let p = scalar_mul(b, p, t2);
    let l3_val = b.assignment[p] * inv6;
    let l3 = b.new_var(l3_val);
    b.add_constraint(vec![(inv6, p)], vec![(F::ONE, b.one())], vec![(F::ONE, l3)]);

    (l0, l1, l2, l3)
}

/// Verify one degree-3 sumcheck over "ring elements" represented coefficient-wise.
///
/// Inputs:
/// - `claimed_sum`: current claim (ring vars)
/// - `msgs[i][t]`: per-round, per-evaluation point ring vars (t in 0..4)
/// - `rs[i]`: per-round verifier challenge scalar vars
///
/// Returns the final subclaim value (ring vars) after all rounds.
pub fn sumcheck_verify_degree3<F: PrimeField>(
    b: &mut Dr1csBuilder<F>,
    mut claimed_sum: RingVars,
    msgs: &[ [RingVars; 4] ],
    rs: &[usize],
) -> Result<RingVars, String> {
    if msgs.len() != rs.len() {
        return Err("sumcheck_verify_degree3: msgs/rs length mismatch".to_string());
    }
    for (round, (m, &r)) in msgs.iter().zip(rs.iter()).enumerate() {
        // Check g(0)+g(1) == claimed_sum (coefficient-wise).
        let g01 = ring_add(b, &m[0], &m[1]);
        // enforce g01 == claimed_sum
        for i in 0..claimed_sum.d() {
            b.enforce_lc_times_one_eq_const(vec![
                (F::ONE, g01.coeffs[i]),
                (-F::ONE, claimed_sum.coeffs[i]),
            ]);
        }

        // Update claim = g(r) by Lagrange interpolation.
        let (l0, l1, l2, l3) = lagrange_degree3::<F>(b, r);
        let t0 = ring_scale(b, &m[0], l0);
        let t1 = ring_scale(b, &m[1], l1);
        let t2 = ring_scale(b, &m[2], l2);
        let t3 = ring_scale(b, &m[3], l3);
        let s01 = ring_add(b, &t0, &t1);
        let s23 = ring_add(b, &t2, &t3);
        let new_claim = ring_add(b, &s01, &s23);
        claimed_sum = new_claim;

        let _ = round;
    }
    Ok(claimed_sum)
}

