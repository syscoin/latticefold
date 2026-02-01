use ark_ff::{Field, PrimeField};
use latticefold::transcript::poseidon::F257;
use symphony::dpp_poseidon::{Constraint, SparseDr1csInstance};
use symphony::dpp_sumcheck::Dr1csBuilder;

use cyclotomic_rings::rings::goldilocks_ntt64 as gl_ntt64;

use std::collections::HashMap;
use core::ops::Range;
use rayon::prelude::*;

use super::digits::{f257_to_i32_bal, i32_to_f257};
use super::params::{DIGITS_PER_TRY, LIMB_BITS, LIMBS_U32};

// -----------------------------------------------------------------------------
// Fox #1 (maintainable): explicit checked vs loose digit types (IR-side)
// -----------------------------------------------------------------------------
//
// The goal is to make "checked vs loose" explicit in function signatures inside the IR, so we
// don't accidentally feed loose digits into gadgets that assume canonical balanced digits.
//
// - Bal16CheckedIr: each digit is intended to be in [-8,7] (range is proven when digits are
//   allocated via `alloc_bal16_digit_ir`, which uses 4 boolean bits).
// - Bal16LooseIr: digits are field elements with a static integer-lift bound |d_i| ≤ M < 128,
//   so linear constraints are injective over the integers (no mod-257 aliasing).

#[derive(Clone, Debug)]
pub(crate) struct Bal16CheckedIr(pub(crate) Vec<VarRef>);

impl Bal16CheckedIr {
    #[inline]
    pub(crate) fn len(&self) -> usize {
        self.0.len()
    }
}

impl core::ops::Deref for Bal16CheckedIr {
    type Target = [VarRef];
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

#[derive(Clone, Debug)]
pub(crate) struct Bal16LooseIr {
    pub(crate) digits: Vec<VarRef>,
    pub(crate) abs_bound: i32,
}

impl Bal16LooseIr {
    #[inline]
    pub(crate) fn len(&self) -> usize {
        self.digits.len()
    }
}

/// Variable reference for a CM IR fragment.
///
/// - `Base(i)` refers to an existing variable in the *base* glue module (same numbering as the base glue instance).
/// - `Local(i)` refers to a variable allocated within this IR fragment (1-indexed; `Local(0)` is unused).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) enum VarRef {
    Base(usize),
    Local(usize),
}

#[derive(Clone, Copy, Debug)]
struct Bal4ConstMulPrecomp {
    k4_const: [i8; 33],
    carry_bounds: [i32; 66],
}

#[derive(Clone, Debug)]
pub(crate) struct IrConstraint {
    pub(crate) a: Range<usize>,
    pub(crate) b: Range<usize>,
    pub(crate) c: Range<usize>,
}

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct CmIrStats {
    /// Constraints of the form (LC) * 1 = 0 (purely linear).
    pub(crate) linear_constraints: u64,
    /// Constraints of the canonical form `a*b=c` emitted via `enforce_mul`.
    pub(crate) mul_constraints: u64,
    /// Non-linear constraints emitted via `add_constraint` (bool/range gadgets etc),
    /// excluding `enforce_mul`.
    pub(crate) other_non_linear_constraints: u64,
    /// Total number of (coeff,var) terms across all A/B/C linear combinations.
    pub(crate) total_terms_a: u64,
    pub(crate) total_terms_b: u64,
    pub(crate) total_terms_c: u64,
    /// Max number of terms in any single A/B/C linear combination.
    pub(crate) max_terms_a: u32,
    pub(crate) max_terms_b: u32,
    pub(crate) max_terms_c: u32,
}

/// A parallel-build-friendly constraint fragment.
///
/// This deliberately avoids any dependence on `Dr1csBuilder` internals/caches, so it can be built
/// in parallel across many shards and then lowered once.
#[derive(Clone, Debug, Default)]
pub(crate) struct CmIr {
    /// Witness values for local vars. Index 0 is reserved as ONE.
    pub(crate) local_asg: Vec<F257>,
    pub(crate) constraints: Vec<IrConstraint>,
    /// Term pools for each side of R1CS constraints.
    ///
    /// This avoids per-constraint heap allocations during IR build, which is a major wall-time cost.
    pub(crate) a_terms: Vec<(F257, VarRef)>,
    pub(crate) b_terms: Vec<(F257, VarRef)>,
    pub(crate) c_terms: Vec<(F257, VarRef)>,
    pub(crate) stats: CmIrStats,
}

impl CmIr {
    #[inline]
    pub(crate) fn new() -> Self {
        // Reserve term 0 on B/C as the constant-one and constant-zero "atoms" so linear constraints
        // can reference them without allocating new vectors.
        Self {
            local_asg: vec![F257::ONE],
            constraints: Vec::new(),
            a_terms: Vec::new(),
            b_terms: vec![(F257::ONE, VarRef::Base(0))],
            c_terms: vec![(F257::ZERO, VarRef::Base(0))],
            stats: CmIrStats::default(),
        }
    }

    /// Allocate a new local var with the given witness value.
    #[inline]
    pub(crate) fn new_var(&mut self, v: F257) -> VarRef {
        let idx = self.local_asg.len();
        self.local_asg.push(v);
        VarRef::Local(idx)
    }

    /// Enforce `(Σ lc) * 1 = 0`.
    #[inline]
    pub(crate) fn enforce_lc_times_one_eq_zero(&mut self, lc: Vec<(F257, VarRef)>) {
        self.stats.linear_constraints += 1;
        self.stats.total_terms_a += lc.len() as u64;
        self.stats.total_terms_b += 1;
        self.stats.total_terms_c += 1;
        self.stats.max_terms_a = self.stats.max_terms_a.max(lc.len() as u32);
        self.stats.max_terms_b = self.stats.max_terms_b.max(1);
        self.stats.max_terms_c = self.stats.max_terms_c.max(1);
        let a0 = self.a_terms.len();
        self.a_terms.extend(lc);
        let a1 = self.a_terms.len();
        self.constraints.push(IrConstraint {
            a: a0..a1,
            b: 0..1, // (ONE, Base(0))
            c: 0..1, // (ZERO, Base(0))
        });
    }

    /// Enforce `x == const`.
    #[inline]
    pub(crate) fn enforce_var_eq_const(&mut self, x: VarRef, c: F257) {
        // (x - c) * 1 = 0
        self.enforce_lc_times_one_eq_zero(vec![(F257::ONE, x), (-c, VarRef::Base(0))]);
    }

    /// Enforce `a * b = c`.
    #[inline]
    pub(crate) fn enforce_mul(&mut self, a: VarRef, b: VarRef, c: VarRef) {
        self.stats.mul_constraints += 1;
        self.stats.total_terms_a += 1;
        self.stats.total_terms_b += 1;
        self.stats.total_terms_c += 1;
        self.stats.max_terms_a = self.stats.max_terms_a.max(1);
        self.stats.max_terms_b = self.stats.max_terms_b.max(1);
        self.stats.max_terms_c = self.stats.max_terms_c.max(1);
        let a0 = self.a_terms.len();
        self.a_terms.push((F257::ONE, a));
        let a1 = self.a_terms.len();
        let b0 = self.b_terms.len();
        self.b_terms.push((F257::ONE, b));
        let b1 = self.b_terms.len();
        let c0 = self.c_terms.len();
        self.c_terms.push((F257::ONE, c));
        let c1 = self.c_terms.len();
        self.constraints.push(IrConstraint { a: a0..a1, b: b0..b1, c: c0..c1 });
    }

    /// Add a raw R1CS constraint with linear combinations over `VarRef`.
    ///
    /// Used for boolean/range gadgets and other small non-linear constraints.
    #[inline]
    pub(crate) fn add_constraint(&mut self, a: Vec<(F257, VarRef)>, b: Vec<(F257, VarRef)>, c: Vec<(F257, VarRef)>) {
        // Classify purely linear constraints that multiply by ONE.
        let is_linear = b.len() == 1 && b[0].0 == F257::ONE && b[0].1 == VarRef::Base(0);
        if is_linear {
            self.stats.linear_constraints += 1;
        } else {
            self.stats.other_non_linear_constraints += 1;
        }
        self.stats.total_terms_a += a.len() as u64;
        self.stats.total_terms_b += b.len() as u64;
        self.stats.total_terms_c += c.len() as u64;
        self.stats.max_terms_a = self.stats.max_terms_a.max(a.len() as u32);
        self.stats.max_terms_b = self.stats.max_terms_b.max(b.len() as u32);
        self.stats.max_terms_c = self.stats.max_terms_c.max(c.len() as u32);
        let a0 = self.a_terms.len();
        self.a_terms.extend(a);
        let a1 = self.a_terms.len();

        let b_rng = if is_linear {
            0..1
        } else {
            let b0 = self.b_terms.len();
            self.b_terms.extend(b);
            let b1 = self.b_terms.len();
            b0..b1
        };

        // Reuse constant-zero on C when possible.
        let c_rng = if c.len() == 1 && c[0].0 == F257::ZERO && c[0].1 == VarRef::Base(0) {
            0..1
        } else {
            let c0 = self.c_terms.len();
            self.c_terms.extend(c);
            let c1 = self.c_terms.len();
            c0..c1
        };

        self.constraints.push(IrConstraint { a: a0..a1, b: b_rng, c: c_rng });
    }

    /// Lower this IR fragment into an existing sparse DR1CS instance/assignment in-place.
    ///
    /// Local variables are appended to `base_asg`, and constraints are appended to `base_inst`.
    pub(crate) fn lower_into(
        self,
        base_inst: &mut SparseDr1csInstance<F257>,
        base_asg: &mut Vec<F257>,
    ) -> Result<(), String> {
        let CmIr { local_asg, constraints, a_terms, b_terms, c_terms, stats } = self;
        let base_nvars = base_asg.len();
        // Big win for wall-time: avoid repeated reallocations during append.
        base_asg.reserve(local_asg.len().saturating_sub(1));
        base_inst.constraints.reserve(constraints.len());
        base_inst.a_terms.reserve(stats.total_terms_a as usize);
        base_inst.b_terms.reserve(stats.total_terms_b as usize);
        base_inst.c_terms.reserve(stats.total_terms_c as usize);
        // Append local vars (skip local_asg[0]=ONE).
        base_asg.extend_from_slice(&local_asg[1..]);
        let map = |v: VarRef, base_nvars: usize| -> usize {
            match v {
                VarRef::Base(i) => i,
                VarRef::Local(j) => {
                    // Local(1) becomes base_nvars, Local(2) base_nvars+1, ...
                    base_nvars + (j - 1)
                }
            }
        };
        for ic in constraints {
            let a0 = base_inst.a_terms.len();
            for &(coef, v) in &a_terms[ic.a.clone()] {
                base_inst.a_terms.push((coef, map(v, base_nvars)));
            }
            let a1 = base_inst.a_terms.len();
            let b0 = base_inst.b_terms.len();
            for &(coef, v) in &b_terms[ic.b.clone()] {
                base_inst.b_terms.push((coef, map(v, base_nvars)));
            }
            let b1 = base_inst.b_terms.len();
            let c0 = base_inst.c_terms.len();
            for &(coef, v) in &c_terms[ic.c.clone()] {
                base_inst.c_terms.push((coef, map(v, base_nvars)));
            }
            let c1 = base_inst.c_terms.len();
            base_inst.constraints.push(Constraint { a: a0..a1, b: b0..b1, c: c0..c1 });
        }
        base_inst.nvars = base_asg.len();
        Ok(())
    }
}

/// Lowering context returned by `lower_ir_into_builder`.
///
/// This captures how this IR fragment's locals were appended to the builder, so callsites
/// can map `VarRef` outputs into concrete `usize` variables.
#[derive(Clone, Debug)]
pub(crate) struct LoweredIr {
    base_nvars: usize,
    /// 1-indexed: `local_to_var[j]` is the concrete var index for `VarRef::Local(j)`.
    /// `local_to_var[0]` is unused.
    local_to_var: Vec<usize>,
}

impl LoweredIr {
    #[inline]
    pub(crate) fn map_var(&self, v: VarRef) -> usize {
        match v {
            VarRef::Base(i) => i,
            VarRef::Local(j) => {
                debug_assert!(j > 0);
                debug_assert_eq!(self.local_to_var[j], self.base_nvars + (j - 1));
                self.local_to_var[j]
            }
        }
    }
}

/// Lower a `CmIr` fragment into a mutable `Dr1csBuilder` by appending:
/// - all local witnesses as new vars
/// - all constraints as R1CS constraints over concrete var indices
pub(crate) fn lower_ir_into_builder(gb: &mut Dr1csBuilder<F257>, ir: CmIr) -> LoweredIr {
    let CmIr { local_asg, constraints, a_terms, b_terms, c_terms, stats } = ir;
    let base_nvars = gb.assignment.len();
    // File-backed builder path: stream constraints/terms instead of appending to giant Vec pools.
    //
    // This avoids the "term pool" RAM blow-ups at the cost of slower lowering (we can optimize
    // streaming writes later).
    if gb.is_file_backed() {
        // If constraint debugging is enabled, preserve the original per-constraint streaming path
        // so `LF_DEBUG_CONSTRAINT_AT` can target a specific next constraint deterministically.
        let debug_on = std::env::var("LF_DEBUG_CONSTRAINT_AT").ok().is_some();

        let mut local_to_var: Vec<usize> = Vec::with_capacity(local_asg.len());
        local_to_var.push(0); // Local(0) unused
        for &v in local_asg.iter().skip(1) {
            local_to_var.push(gb.new_var(v));
        }
        let lowered = LoweredIr { base_nvars, local_to_var };

        if debug_on || constraints.is_empty() {
            for ic in constraints {
                gb.add_constraint_terms_iter(
                    a_terms[ic.a.clone()]
                        .iter()
                        .map(|(coef, v)| (*coef, lowered.map_var(*v))),
                    b_terms[ic.b.clone()]
                        .iter()
                        .map(|(coef, v)| (*coef, lowered.map_var(*v))),
                    c_terms[ic.c.clone()]
                        .iter()
                        .map(|(coef, v)| (*coef, lowered.map_var(*v))),
                );
            }
            return lowered;
        }

        // Performance path (file-backed): do block-wise parallel remap+serialization into flat buffers,
        // then bulk append to the file-backed sink.
        //
        // This uses all cores on the CPU-heavy side (var remap + arkworks serialization) while keeping
        // IO writes large and sequential.
        let coeff_size = gb
            .file_coeff_size()
            .expect("file-backed lowering: missing coeff_size (file_sink must be Some)");

        // Tuneable: constraints per block (bounds peak RAM of flat buffers).
        let block_constraints: usize = std::env::var("LFP_TINY_FILE_LOWER_BLOCK_CONSTRAINTS")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .filter(|&n| n > 0)
            .unwrap_or(4096);

        // Local serialization helper: write canonical uncompressed bytes into `out` (fixed size).
        #[inline]
        fn serialize_fixed_noalloc<F: ark_serialize::CanonicalSerialize>(x: &F, out: &mut [u8]) {
            use ark_serialize::{Compress, CanonicalSerialize};
            use core::cmp::min;
            use std::io::Write as _;

            struct SliceWriter<'a> {
                buf: &'a mut [u8],
                pos: usize,
            }
            impl<'a> std::io::Write for SliceWriter<'a> {
                fn write(&mut self, data: &[u8]) -> std::io::Result<usize> {
                    let rem = self.buf.len().saturating_sub(self.pos);
                    let n = min(rem, data.len());
                    if n == 0 {
                        return Err(std::io::Error::new(std::io::ErrorKind::WriteZero, "SliceWriter overflow"));
                    }
                    self.buf[self.pos..self.pos + n].copy_from_slice(&data[..n]);
                    self.pos += n;
                    Ok(n)
                }
                fn flush(&mut self) -> std::io::Result<()> {
                    Ok(())
                }
            }

            let mut w = SliceWriter { buf: out, pos: 0 };
            x.serialize_with_mode(&mut w, Compress::No)
                .expect("serialize_fixed_noalloc failed");
            debug_assert_eq!(w.pos, out.len());
        }

        // Mirror `file_backed_dr1cs::hash_u64s` so we keep the same "fast hash + limb verify" strategy
        // as the sequential writer's `coeff_cache`.
        #[inline]
        fn hash_u64s(limbs: &[u64]) -> u64 {
            let mut h: u64 = 0x9E37_79B9_7F4A_7C15;
            for &x in limbs {
                let mut z = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
                z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
                z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
                z ^= z >> 31;
                h ^= z;
                h = h.rotate_left(13).wrapping_mul(0x9E37_79B9_7F4A_7C15);
            }
            h
        }

        #[derive(Clone, Debug)]
        struct CoeffCacheEntry {
            limbs: Vec<u64>,
            bytes: Vec<u8>,
        }

        struct CoeffCache {
            coeff_size: usize,
            zero_bytes: Vec<u8>,
            one_bytes: Vec<u8>,
            neg_one_bytes: Vec<u8>,
            buckets: HashMap<u64, Vec<CoeffCacheEntry>>,
        }
        impl CoeffCache {
            #[inline]
            fn new(coeff_size: usize) -> Self {
                let mut zero_bytes = vec![0u8; coeff_size];
                let mut one_bytes = vec![0u8; coeff_size];
                let mut neg_one_bytes = vec![0u8; coeff_size];
                serialize_fixed_noalloc(&F257::ZERO, &mut zero_bytes);
                serialize_fixed_noalloc(&F257::ONE, &mut one_bytes);
                serialize_fixed_noalloc(&(-F257::ONE), &mut neg_one_bytes);
                Self {
                    coeff_size,
                    zero_bytes,
                    one_bytes,
                    neg_one_bytes,
                    buckets: HashMap::new(),
                }
            }

            #[inline]
            fn write_coeff(&mut self, coef: &F257, out: &mut [u8]) {
                debug_assert_eq!(out.len(), self.coeff_size);
                // Hot fast-path for the most common coefficients in R1CS.
                if *coef == F257::ZERO {
                    out.copy_from_slice(&self.zero_bytes);
                    return;
                }
                if *coef == F257::ONE {
                    out.copy_from_slice(&self.one_bytes);
                    return;
                }
                if *coef == -F257::ONE {
                    out.copy_from_slice(&self.neg_one_bytes);
                    return;
                }

                let big = (*coef).into_bigint();
                let limbs = big.as_ref();
                let key = hash_u64s(limbs);
                if let Some(bucket) = self.buckets.get(&key) {
                    for ent in bucket {
                        if ent.limbs.as_slice() == limbs {
                            out.copy_from_slice(&ent.bytes);
                            return;
                        }
                    }
                }

                // Cache miss: serialize directly into `out` once, then stash a copy.
                serialize_fixed_noalloc(coef, out);
                let ent = CoeffCacheEntry {
                    limbs: limbs.to_vec(),
                    bytes: out.to_vec(),
                };
                self.buckets.entry(key).or_default().push(ent);
            }
        }

        // Capture mapping pieces for fast remap.
        let base_nvars_local = lowered.base_nvars;
        let local_to_var = &lowered.local_to_var;
        #[inline]
        fn map_var_fast(base_nvars: usize, local_to_var: &[usize], v: VarRef) -> u64 {
            match v {
                VarRef::Base(i) => i as u64,
                VarRef::Local(j) => {
                    debug_assert!(j > 0);
                    debug_assert_eq!(local_to_var[j], base_nvars + (j - 1));
                    local_to_var[j] as u64
                }
            }
        }

        let mut idx0: usize = 0;
        while idx0 < constraints.len() {
            let idx1 = (idx0 + block_constraints).min(constraints.len());
            let cons = &constraints[idx0..idx1];

            // Per-constraint term counts and prefix sums.
            let n = cons.len();
            let mut a_counts: Vec<usize> = Vec::with_capacity(n);
            let mut b_counts: Vec<usize> = Vec::with_capacity(n);
            let mut c_counts: Vec<usize> = Vec::with_capacity(n);
            for ic in cons {
                a_counts.push(ic.a.end - ic.a.start);
                b_counts.push(ic.b.end - ic.b.start);
                c_counts.push(ic.c.end - ic.c.start);
            }
            let mut a_offs: Vec<usize> = Vec::with_capacity(n + 1);
            let mut b_offs: Vec<usize> = Vec::with_capacity(n + 1);
            let mut c_offs: Vec<usize> = Vec::with_capacity(n + 1);
            a_offs.push(0);
            b_offs.push(0);
            c_offs.push(0);
            for i in 0..n {
                a_offs.push(a_offs[i] + a_counts[i]);
                b_offs.push(b_offs[i] + b_counts[i]);
                c_offs.push(c_offs[i] + c_counts[i]);
            }
            let total_a = *a_offs.last().unwrap_or(&0);
            let total_b = *b_offs.last().unwrap_or(&0);
            let total_c = *c_offs.last().unwrap_or(&0);

            // Flat buffers (coeff bytes + indices).
            let mut a_coeffs: Vec<u8> = vec![0u8; total_a.saturating_mul(coeff_size)];
            let mut b_coeffs: Vec<u8> = vec![0u8; total_b.saturating_mul(coeff_size)];
            let mut c_coeffs: Vec<u8> = vec![0u8; total_c.saturating_mul(coeff_size)];
            let mut a_idx: Vec<u64> = vec![0u64; total_a];
            let mut b_idx: Vec<u64> = vec![0u64; total_b];
            let mut c_idx: Vec<u64> = vec![0u64; total_c];

            // Parallel fill with disjoint writes.
            #[derive(Copy, Clone)]
            struct SyncPtr<T>(*mut T);
            unsafe impl<T> Send for SyncPtr<T> {}
            unsafe impl<T> Sync for SyncPtr<T> {}
            impl<T> SyncPtr<T> {
                #[inline]
                unsafe fn write(&self, off: usize, v: T) {
                    core::ptr::write(self.0.add(off), v);
                }
            }

            let out_a_idx: SyncPtr<u64> = SyncPtr(a_idx.as_mut_ptr());
            let out_b_idx: SyncPtr<u64> = SyncPtr(b_idx.as_mut_ptr());
            let out_c_idx: SyncPtr<u64> = SyncPtr(c_idx.as_mut_ptr());

            // Raw ptrs to coefficient byte buffers.
            let a_bytes_ptr: usize = a_coeffs.as_mut_ptr() as usize;
            let b_bytes_ptr: usize = b_coeffs.as_mut_ptr() as usize;
            let c_bytes_ptr: usize = c_coeffs.as_mut_ptr() as usize;

            use rayon::prelude::*;
            cons.par_iter()
                .enumerate()
                .for_each_init(|| CoeffCache::new(coeff_size), |cc, (i, ic)| unsafe {
                // A terms.
                let a_dst0 = a_offs[i];
                for (k, (coef, v)) in a_terms[ic.a.clone()].iter().enumerate() {
                    let pos = a_dst0 + k;
                    out_a_idx.write(pos, map_var_fast(base_nvars_local, local_to_var, *v));
                    let dst = (a_bytes_ptr as *mut u8).add(pos * coeff_size);
                    let dst_slice = core::slice::from_raw_parts_mut(dst, coeff_size);
                    cc.write_coeff(coef, dst_slice);
                }
                // B terms.
                let b_dst0 = b_offs[i];
                for (k, (coef, v)) in b_terms[ic.b.clone()].iter().enumerate() {
                    let pos = b_dst0 + k;
                    out_b_idx.write(pos, map_var_fast(base_nvars_local, local_to_var, *v));
                    let dst = (b_bytes_ptr as *mut u8).add(pos * coeff_size);
                    let dst_slice = core::slice::from_raw_parts_mut(dst, coeff_size);
                    cc.write_coeff(coef, dst_slice);
                }
                // C terms.
                let c_dst0 = c_offs[i];
                for (k, (coef, v)) in c_terms[ic.c.clone()].iter().enumerate() {
                    let pos = c_dst0 + k;
                    out_c_idx.write(pos, map_var_fast(base_nvars_local, local_to_var, *v));
                    let dst = (c_bytes_ptr as *mut u8).add(pos * coeff_size);
                    let dst_slice = core::slice::from_raw_parts_mut(dst, coeff_size);
                    cc.write_coeff(coef, dst_slice);
                }
            });

            // Bulk append: write all terms, then all rows.
            let (_rows0, a0, b0, c0) = gb
                .file_counts()
                .expect("file_counts must be Some in file-backed mode");

            // A pool.
            for t in 0..total_a {
                let off = t * coeff_size;
                gb.file_push_a_term_raw(&a_coeffs[off..off + coeff_size], a_idx[t])
                    .unwrap_or_else(|e| panic!("file-backed dr1cs write failed (a_term_raw): {e}"));
            }
            // B pool.
            for t in 0..total_b {
                let off = t * coeff_size;
                gb.file_push_b_term_raw(&b_coeffs[off..off + coeff_size], b_idx[t])
                    .unwrap_or_else(|e| panic!("file-backed dr1cs write failed (b_term_raw): {e}"));
            }
            // C pool.
            for t in 0..total_c {
                let off = t * coeff_size;
                gb.file_push_c_term_raw(&c_coeffs[off..off + coeff_size], c_idx[t])
                    .unwrap_or_else(|e| panic!("file-backed dr1cs write failed (c_term_raw): {e}"));
            }

            // Rows referencing the just-appended pools.
            for i in 0..n {
                let a_start = a0 + (a_offs[i] as u64);
                let a_end = a_start + (a_counts[i] as u64);
                let b_start = b0 + (b_offs[i] as u64);
                let b_end = b_start + (b_counts[i] as u64);
                let c_start = c0 + (c_offs[i] as u64);
                let c_end = c_start + (c_counts[i] as u64);
                gb.file_push_constraint_row(a_start, a_end, b_start, b_end, c_start, c_end)
                    .unwrap_or_else(|e| panic!("file-backed dr1cs write failed (constraint_row): {e}"));
            }

            idx0 = idx1;
        }

        return lowered;
    }
    // Big win for wall-time: avoid repeated reallocations during append.
    gb.assignment.reserve(local_asg.len().saturating_sub(1));
    gb.rows.reserve(constraints.len());
    gb.a_terms.reserve(stats.total_terms_a as usize);
    gb.b_terms.reserve(stats.total_terms_b as usize);
    gb.c_terms.reserve(stats.total_terms_c as usize);
    let mut local_to_var: Vec<usize> = Vec::with_capacity(local_asg.len());
    local_to_var.push(0); // Local(0) unused
    for &v in local_asg.iter().skip(1) {
        local_to_var.push(gb.new_var(v));
    }

    let lowered = LoweredIr { base_nvars, local_to_var };

    // Lower constraints/terms.
    //
    // Vars must be allocated sequentially (builder is append-only), but the remap+copy of the
    // IR's pooled term arrays into the builder can be parallelized.
    //
    // This is the dominant wall-time for large IR fragments (e.g. NTT ringmul shards).
    let do_profile = gb.profile_enabled;
    let key = gb.profile_current.unwrap_or("unlabeled");
    let n_new = constraints.len() as u64;

    // Threshold: below this, the sequential loop is faster than prefix-sums + Rayon overhead.
    let use_parallel = constraints.len() >= 50_000 && rayon::current_num_threads() > 1;
    if !use_parallel {
        for ic in constraints {
            let a0 = gb.a_terms.len();
            for &(coef, v) in &a_terms[ic.a.clone()] {
                gb.a_terms.push((coef, lowered.map_var(v)));
            }
            let a1 = gb.a_terms.len();
            let b0 = gb.b_terms.len();
            for &(coef, v) in &b_terms[ic.b.clone()] {
                gb.b_terms.push((coef, lowered.map_var(v)));
            }
            let b1 = gb.b_terms.len();
            let c0 = gb.c_terms.len();
            for &(coef, v) in &c_terms[ic.c.clone()] {
                gb.c_terms.push((coef, lowered.map_var(v)));
            }
            let c1 = gb.c_terms.len();
            gb.rows.push(Constraint { a: a0..a1, b: b0..b1, c: c0..c1 });
        }
    } else {
        let n = constraints.len();
        let mut a_counts: Vec<usize> = Vec::with_capacity(n);
        let mut b_counts: Vec<usize> = Vec::with_capacity(n);
        let mut c_counts: Vec<usize> = Vec::with_capacity(n);
        for ic in &constraints {
            a_counts.push(ic.a.end - ic.a.start);
            b_counts.push(ic.b.end - ic.b.start);
            c_counts.push(ic.c.end - ic.c.start);
        }
        let mut a_offs: Vec<usize> = Vec::with_capacity(n + 1);
        let mut b_offs: Vec<usize> = Vec::with_capacity(n + 1);
        let mut c_offs: Vec<usize> = Vec::with_capacity(n + 1);
        a_offs.push(0);
        b_offs.push(0);
        c_offs.push(0);
        for i in 0..n {
            a_offs.push(a_offs[i] + a_counts[i]);
            b_offs.push(b_offs[i] + b_counts[i]);
            c_offs.push(c_offs[i] + c_counts[i]);
        }
        let total_a = *a_offs.last().unwrap_or(&0);
        let total_b = *b_offs.last().unwrap_or(&0);
        let total_c = *c_offs.last().unwrap_or(&0);

        // Append space in destination vectors (we will fill in parallel with disjoint writes).
        let a_base = gb.a_terms.len();
        let b_base = gb.b_terms.len();
        let c_base = gb.c_terms.len();
        let row_base = gb.rows.len();
        gb.a_terms.reserve(total_a);
        gb.b_terms.reserve(total_b);
        gb.c_terms.reserve(total_c);
        gb.rows.reserve(n);
        unsafe {
            gb.a_terms.set_len(a_base + total_a);
            gb.b_terms.set_len(b_base + total_b);
            gb.c_terms.set_len(c_base + total_c);
            gb.rows.set_len(row_base + n);
        }

        // Raw pointers for parallel writes (each thread writes disjoint ranges by construction).
        #[derive(Copy, Clone)]
        struct SyncPtr<T>(*mut T);
        unsafe impl<T> Send for SyncPtr<T> {}
        unsafe impl<T> Sync for SyncPtr<T> {}
        impl<T> SyncPtr<T> {
            #[inline]
            unsafe fn write(&self, off: usize, v: T) {
                core::ptr::write(self.0.add(off), v);
            }
        }
        let out_a: SyncPtr<(F257, usize)> = SyncPtr(gb.a_terms.as_mut_ptr());
        let out_b: SyncPtr<(F257, usize)> = SyncPtr(gb.b_terms.as_mut_ptr());
        let out_c: SyncPtr<(F257, usize)> = SyncPtr(gb.c_terms.as_mut_ptr());
        let out_rows: SyncPtr<Constraint> = SyncPtr(gb.rows.as_mut_ptr());

        // Capture mapping pieces for cheap remap in the hot loop.
        let base_nvars_local = lowered.base_nvars;
        let local_to_var = &lowered.local_to_var;
        #[inline]
        fn map_var_fast(base_nvars: usize, local_to_var: &[usize], v: VarRef) -> usize {
            match v {
                VarRef::Base(i) => i,
                VarRef::Local(j) => {
                    debug_assert!(j > 0);
                    debug_assert_eq!(local_to_var[j], base_nvars + (j - 1));
                    local_to_var[j]
                }
            }
        }

        constraints
            .par_iter()
            .enumerate()
            .for_each(|(i, ic)| unsafe {
                // Fill A terms.
                let a_dst0 = a_base + a_offs[i];
                for (k, &(coef, v)) in a_terms[ic.a.clone()].iter().enumerate() {
                    out_a.write(a_dst0 + k, (coef, map_var_fast(base_nvars_local, local_to_var, v)));
                }
                // Fill B terms.
                let b_dst0 = b_base + b_offs[i];
                for (k, &(coef, v)) in b_terms[ic.b.clone()].iter().enumerate() {
                    out_b.write(b_dst0 + k, (coef, map_var_fast(base_nvars_local, local_to_var, v)));
                }
                // Fill C terms.
                let c_dst0 = c_base + c_offs[i];
                for (k, &(coef, v)) in c_terms[ic.c.clone()].iter().enumerate() {
                    out_c.write(c_dst0 + k, (coef, map_var_fast(base_nvars_local, local_to_var, v)));
                }
                // Row ranges.
                let a_rng = (a_dst0)..(a_dst0 + a_counts[i]);
                let b_rng = (b_dst0)..(b_dst0 + b_counts[i]);
                let c_rng = (c_dst0)..(c_dst0 + c_counts[i]);
                out_rows.write(row_base + i, Constraint { a: a_rng, b: b_rng, c: c_rng });
            });
    }
    if do_profile {
        gb.profile.entry(key).or_default().constraints += n_new;
    }

    lowered
}

/// Convenience IR builder that can compute witnesses from a base assignment.
///
/// This is the intended entry point for parallel CM gadget construction:
/// each thread builds a `CmIr` fragment using only local allocations + constraints
/// (referencing existing base vars via `VarRef::Base`), then the main thread lowers
/// fragments into the final DR1CS instance.
#[derive(Clone)]
pub(crate) struct IrBuilder<'a> {
    pub(crate) base_asg: &'a [F257],
    pub(crate) ir: CmIr,
    // Cache for expensive representation conversions used in hot paths.
    bal16_to_bal4_cache: HashMap<[VarRef; 17], [VarRef; 33]>,
    // Cache for constant-mul carry schedules (NTT twiddles repeat a lot).
    bal4_const_mul_cache: HashMap<u64, Bal4ConstMulPrecomp>,
}

impl<'a> IrBuilder<'a> {
    #[inline]
    pub(crate) fn new(base_asg: &'a [F257]) -> Self {
        Self {
            base_asg,
            ir: CmIr::new(),
            bal16_to_bal4_cache: HashMap::new(),
            bal4_const_mul_cache: HashMap::new(),
        }
    }

    /// Read the witness value for a var ref.
    #[inline]
    pub(crate) fn val(&self, v: VarRef) -> F257 {
        match v {
            VarRef::Base(i) => self.base_asg[i],
            VarRef::Local(j) => self.ir.local_asg[j],
        }
    }

    /// Allocate a new local var with witness.
    #[inline]
    pub(crate) fn new_var(&mut self, v: F257) -> VarRef {
        self.ir.new_var(v)
    }

    /// Return the global ONE (base var0).
    #[inline]
    pub(crate) fn one(&self) -> VarRef {
        VarRef::Base(0)
    }

    /// Create a linear constraint: Σ coeff*var == 0.
    #[inline]
    pub(crate) fn enforce_lc_eq_zero(&mut self, lc: Vec<(F257, VarRef)>) {
        self.ir.enforce_lc_times_one_eq_zero(lc)
    }

    /// Enforce `out = Σ coeff*var` (with constant term represented by `Base(0)`).
    #[inline]
    pub(crate) fn enforce_affine(&mut self, out: VarRef, terms: Vec<(F257, VarRef)>) {
        let mut lc: Vec<(F257, VarRef)> = Vec::with_capacity(1 + terms.len());
        lc.push((F257::ONE, out));
        for (c, v) in terms {
            lc.push((-c, v));
        }
        self.enforce_lc_eq_zero(lc);
    }

    /// Enforce multiplication `a * b = c`.
    #[inline]
    pub(crate) fn enforce_mul(&mut self, a: VarRef, b: VarRef, c: VarRef) {
        self.ir.enforce_mul(a, b, c)
    }

    /// Add a raw R1CS constraint with linear combinations over `VarRef`.
    #[inline]
    pub(crate) fn add_constraint(&mut self, a: Vec<(F257, VarRef)>, b: Vec<(F257, VarRef)>, c: Vec<(F257, VarRef)>) {
        self.ir.add_constraint(a, b, c);
    }

    /// Convert a bal16 scalar (17 digits, last is carry bit) to bal4 digits (33 digits),
    /// with caching keyed by the input var IDs.
    #[inline]
    pub(crate) fn bal16_to_bal4_digits_cached(&mut self, x16: &[VarRef; 17]) -> [VarRef; 33] {
        let key = *x16;
        if let Some(v) = self.bal16_to_bal4_cache.get(&key) {
            return *v;
        }
        let out = bal16_to_bal4_digits_ir(self, x16);
        self.bal16_to_bal4_cache.insert(key, out);
        out
    }

    #[inline]
    fn bal4_const_mul_precomp(&mut self, k: u64) -> Bal4ConstMulPrecomp {
        if let Some(p) = self.bal4_const_mul_cache.get(&k) {
            return *p;
        }
        let k4_const = u64_to_bal4_digits_le_const(k);
        let abs_k: [i32; 33] = core::array::from_fn(|i| (k4_const[i] as i32).abs());
        let mut pref: [i32; 34] = [0; 34];
        for i in 0..33 {
            pref[i + 1] = pref[i] + abs_k[i];
        }
        let mut carry_bounds: [i32; 66] = [31; 66];
        let mut carry_bound_raw: i32 = 0;
        for kk in 0..66 {
            let s = if kk <= 32 {
                pref[kk + 1]
            } else {
                let lo = kk - 32;
                pref[33] - pref[lo]
            };
            let m_k = 2 * s;
            let sum_bound = carry_bound_raw + m_k + 8;
            carry_bound_raw = (sum_bound + 3) / 4;
            carry_bounds[kk] = carry_bound_raw;
        }
        let p = Bal4ConstMulPrecomp { k4_const, carry_bounds };
        self.bal4_const_mul_cache.insert(k, p);
        p
    }
}

/// Allocate a boolean variable with witness `bit`.
///
/// Constraints:
/// - x ∈ {0,1} via x*(x-1)=0
pub(crate) fn alloc_bool_ir(b: &mut IrBuilder<'_>, bit: bool) -> VarRef {
    // Match `gadgets::alloc_bool` exactly, but over `VarRef`.
    let x = b.new_var(if bit { F257::ONE } else { F257::ZERO });
    // x*(1-x)=0
    b.add_constraint(
        vec![(F257::ONE, x)],
        vec![(F257::ONE, b.one()), (-F257::ONE, x)],
        vec![(F257::ZERO, b.one())],
    );
    x
}

/// Enforce an existing var is boolean: v*(1-v)=0.
#[inline]
pub(crate) fn enforce_bit_var_ir(b: &mut IrBuilder<'_>, v: VarRef) {
    b.add_constraint(
        vec![(F257::ONE, v)],
        vec![(F257::ONE, b.one()), (-F257::ONE, v)],
        vec![(F257::ZERO, b.one())],
    );
}

/// Boolean indicator for (d == 256), using the inverse trick.
///
/// Returns `is256 ∈ {0,1}` and enforces:
/// - diff = d - 256
/// - prod = diff * inv
/// - prod = 1 - is256
/// - diff * is256 = 0
///
/// This forces is256=1 iff d==256, else is256=0 (over F257).
pub(crate) fn digit_is_256_bit_ir(b: &mut IrBuilder<'_>, d: VarRef) -> VarRef {
    let dval = b.val(d);
    let is256 = alloc_bool_ir(b, dval == F257::from(256u64));

    let diff = b.new_var(dval - F257::from(256u64));
    // diff = d - 256
    b.enforce_lc_eq_zero(vec![
        (F257::ONE, diff),
        (-F257::ONE, d),
        (F257::from(256u64), b.one()),
    ]);

    let inv = b.new_var(if dval == F257::from(256u64) {
        F257::ZERO
    } else {
        (dval - F257::from(256u64)).inverse().unwrap()
    });
    let prod = b.new_var(b.val(diff) * b.val(inv));
    b.enforce_mul(diff, inv, prod);

    // prod = 1 - is256
    b.enforce_lc_eq_zero(vec![(F257::ONE, prod), (F257::ONE, is256), (-F257::ONE, b.one())]);

    // diff * is256 = 0
    let z = alloc_zero_const_ir(b);
    b.enforce_mul(diff, is256, z);

    is256
}

/// Map a base-257 digit `d ∈ {0..=256}` to a byte `b ∈ {0..=255}` via the transcript rule:
/// `256 -> 0`, else `b=d`.
///
/// In F257 arithmetic this is: `b = d + is_eq256(d)`, since `256 ≡ -1 (mod 257)`.
///
/// NOTE: this does **not** bit-decompose/range-check `byte`; callsites should do that in the
/// builder layer (so they can reuse caches).
pub(crate) fn digit_to_byte_ir(b: &mut IrBuilder<'_>, d: VarRef) -> VarRef {
    let is256 = digit_is_256_bit_ir(b, d);
    let byte = b.new_var(b.val(d) + b.val(is256));
    b.enforce_lc_eq_zero(vec![(F257::ONE, byte), (-F257::ONE, d), (-F257::ONE, is256)]);
    byte
}

/// Select the first acceptable `get_challenge()` try (fixed schedule) in IR form.
///
/// Input `digits_by_try` is length `tries * DIGITS_PER_TRY`, ordered by try then digit index.
/// Accept iff the first 4 digits are all != 256.
///
/// Returns:
/// - selected digits `[d0..d7]` (each equals the corresponding digit of the chosen try)
/// - `found_bit` indicating that an acceptable try exists (enforced by caller if desired)
pub(crate) fn select_first_ok_u32_try_digits_ir(
    b: &mut IrBuilder<'_>,
    digits_by_try: &[VarRef],
    tries: usize,
) -> ([VarRef; DIGITS_PER_TRY], VarRef) {
    assert_eq!(digits_by_try.len(), tries * DIGITS_PER_TRY);

    // found accumulates OR of ok bits; select_t picks first ok.
    let mut found = b.new_var(F257::ZERO);
    b.ir.enforce_var_eq_const(found, F257::ZERO);
    enforce_bit_var_ir(b, found);

    let mut selects: Vec<VarRef> = Vec::with_capacity(tries);

    for t in 0..tries {
        let base = t * DIGITS_PER_TRY;
        let e0 = digit_is_256_bit_ir(b, digits_by_try[base + 0]);
        let e1 = digit_is_256_bit_ir(b, digits_by_try[base + 1]);
        let e2 = digit_is_256_bit_ir(b, digits_by_try[base + 2]);
        let e3 = digit_is_256_bit_ir(b, digits_by_try[base + 3]);

        // o = 1 - e
        let one_minus = |b: &mut IrBuilder<'_>, e: VarRef| -> VarRef {
            let v = b.new_var(F257::ONE - b.val(e));
            b.enforce_lc_eq_zero(vec![(F257::ONE, v), (F257::ONE, e), (-F257::ONE, b.one())]);
            v
        };
        let o0 = one_minus(b, e0);
        let o1 = one_minus(b, e1);
        let o2 = one_minus(b, e2);
        let o3 = one_minus(b, e3);

        // ok = o0*o1*o2*o3 (boolean)
        let ok01 = b.new_var(b.val(o0) * b.val(o1));
        b.enforce_mul(o0, o1, ok01);
        let ok23 = b.new_var(b.val(o2) * b.val(o3));
        b.enforce_mul(o2, o3, ok23);
        let ok = b.new_var(b.val(ok01) * b.val(ok23));
        b.enforce_mul(ok01, ok23, ok);
        enforce_bit_var_ir(b, ok);

        let not_found = one_minus(b, found);
        let sel = b.new_var(b.val(ok) * b.val(not_found));
        b.enforce_mul(ok, not_found, sel);
        enforce_bit_var_ir(b, sel);
        selects.push(sel);

        // found' = found + sel
        let found_next = b.new_var(b.val(found) + b.val(sel));
        b.enforce_lc_eq_zero(vec![(F257::ONE, found_next), (-F257::ONE, found), (-F257::ONE, sel)]);
        enforce_bit_var_ir(b, found_next);
        found = found_next;
    }

    // selected digit i = Σ_t sel_t * digit_{t,i}
    let mut out: [VarRef; DIGITS_PER_TRY] = core::array::from_fn(|_| b.one());
    for i in 0..DIGITS_PER_TRY {
        let mut prods: Vec<VarRef> = Vec::with_capacity(tries);
        let mut acc = F257::ZERO;
        for t in 0..tries {
            let d = digits_by_try[t * DIGITS_PER_TRY + i];
            let p = b.new_var(b.val(selects[t]) * b.val(d));
            b.enforce_mul(selects[t], d, p);
            acc += b.val(p);
            prods.push(p);
        }
        let v = b.new_var(acc);
        let mut lc: Vec<(F257, VarRef)> = Vec::with_capacity(1 + prods.len());
        lc.push((F257::ONE, v));
        for p in prods {
            lc.push((-F257::ONE, p));
        }
        b.enforce_lc_eq_zero(lc);
        out[i] = v;
    }

    (out, found)
}

/// Pack a u32 from its little-endian bit variables into base-128 limbs (little-endian).
///
/// Assumes the first 32 bits of `bits32` are boolean.
pub(crate) fn u32_bits_to_base128_limbs_ir(b: &mut IrBuilder<'_>, bits32: &[VarRef; 32]) -> [VarRef; LIMBS_U32] {
    let mut limbs: [VarRef; LIMBS_U32] = core::array::from_fn(|_| b.one());
    for li in 0..LIMBS_U32 {
        let start = li * LIMB_BITS;
        let end = core::cmp::min(start + LIMB_BITS, 32);

        // Witness limb value from bits.
        let mut limb_u8: u8 = 0;
        for j in start..end {
            let bv = b.val(bits32[j]);
            debug_assert!(bv == F257::ZERO || bv == F257::ONE, "u32_bits_to_base128_limbs_ir: non-boolean bit witness");
            if bv == F257::ONE {
                limb_u8 |= 1u8 << (j - start);
            }
        }
        let limb = b.new_var(F257::from(limb_u8 as u64));

        // Enforce limb = Σ 2^k * bits32[start+k]
        let mut lc: Vec<(F257, VarRef)> = Vec::with_capacity(1 + (end - start));
        lc.push((F257::ONE, limb));
        for j in start..end {
            let p2 = F257::from(1u64 << (j - start));
            lc.push((-p2, bits32[j]));
        }
        b.enforce_lc_eq_zero(lc);
        limbs[li] = limb;
    }
    limbs
}

/// Allocate a signed carry `c ∈ {-1,0,1}` using the vanishing polynomial over F257:
/// \[
///   (c-1)\,c\,(c+1) = 0.
/// \]
///
/// Since 257 is prime > 3, this has exactly the intended roots in F257.
pub(crate) fn alloc_carry_pm1_ir(b: &mut IrBuilder<'_>, c: i32) -> VarRef {
    assert!((-1..=1).contains(&c));
    let c_var = b.new_var(i32_to_f257(c));

    // t = (c-1)*c
    let cv = b.val(c_var);
    let t = b.new_var((cv - F257::ONE) * cv);
    b.add_constraint(
        vec![(F257::ONE, c_var), (-F257::ONE, b.one())],
        vec![(F257::ONE, c_var)],
        vec![(F257::ONE, t)],
    );
    // t*(c+1) = 0
    b.add_constraint(
        vec![(F257::ONE, t)],
        vec![(F257::ONE, c_var), (F257::ONE, b.one())],
        vec![(F257::ZERO, b.one())],
    );
    c_var
}

// Small carry ranges (statement-only): allocate by offsetting into a small unsigned range and
// bit-decomposing that offset. This is used by normalization of loose digits.
#[inline]
pub(crate) fn alloc_carry_pm8_ir(b: &mut IrBuilder<'_>, c: i32) -> VarRef {
    debug_assert!((-8..=7).contains(&c));
    // Balanced signed encoding with 4 bits:
    //   c = b0 + 2*b1 + 4*b2 - 8*b3  ∈ [-8,7]
    //
    // This is strictly cheaper than the older "offset + recombine" encoding
    // (no extra `off_var`, one fewer linear constraint), while remaining a sound range check.
    let u: u8 = if c < 0 { (16 + c) as u8 } else { c as u8 }; // c mod 16
    let mut bits4 = [b.one(); 4];
    for i in 0..4 {
        bits4[i] = alloc_bool_ir(b, ((u >> i) & 1) == 1);
    }
    let c_var = b.new_var(i32_to_f257(c));
    b.enforce_lc_eq_zero(vec![
        (F257::ONE, c_var),
        (-F257::ONE, bits4[0]),
        (-F257::from(2u64), bits4[1]),
        (-F257::from(4u64), bits4[2]),
        (F257::from(8u64), bits4[3]),
    ]);
    c_var
}

#[inline]
pub(crate) fn alloc_carry_pm16_ir(b: &mut IrBuilder<'_>, c: i32) -> VarRef {
    debug_assert!((-16..=15).contains(&c));
    // Balanced signed encoding with 5 bits:
    //   c = b0 + 2*b1 + 4*b2 + 8*b3 - 16*b4  ∈ [-16,15]
    let u: u8 = if c < 0 { (32 + c) as u8 } else { c as u8 }; // c mod 32
    let mut bits5 = [b.one(); 5];
    for i in 0..5 {
        bits5[i] = alloc_bool_ir(b, ((u >> i) & 1) == 1);
    }
    let c_var = b.new_var(i32_to_f257(c));
    b.enforce_lc_eq_zero(vec![
        (F257::ONE, c_var),
        (-F257::ONE, bits5[0]),
        (-F257::from(2u64), bits5[1]),
        (-F257::from(4u64), bits5[2]),
        (-F257::from(8u64), bits5[3]),
        (F257::from(16u64), bits5[4]),
    ]);
    c_var
}

#[inline]
pub(crate) fn alloc_carry_pm32_ir(b: &mut IrBuilder<'_>, c: i32) -> VarRef {
    debug_assert!((-32..=31).contains(&c));
    // Balanced signed encoding with 6 bits:
    //   c = b0 + 2*b1 + 4*b2 + 8*b3 + 16*b4 - 32*b5  ∈ [-32,31]
    let u: u8 = if c < 0 { (64 + c) as u8 } else { c as u8 }; // c mod 64
    let mut bits6 = [b.one(); 6];
    for i in 0..6 {
        bits6[i] = alloc_bool_ir(b, ((u >> i) & 1) == 1);
    }
    let c_var = b.new_var(i32_to_f257(c));
    b.enforce_lc_eq_zero(vec![
        (F257::ONE, c_var),
        (-F257::ONE, bits6[0]),
        (-F257::from(2u64), bits6[1]),
        (-F257::from(4u64), bits6[2]),
        (-F257::from(8u64), bits6[3]),
        (-F257::from(16u64), bits6[4]),
        (F257::from(32u64), bits6[5]),
    ]);
    c_var
}

#[inline]
pub(crate) fn alloc_carry_pm64_ir(b: &mut IrBuilder<'_>, c: i32) -> VarRef {
    debug_assert!((-64..=63).contains(&c));
    // Balanced signed encoding with 7 bits:
    //   c = b0 + 2*b1 + 4*b2 + 8*b3 + 16*b4 + 32*b5 - 64*b6  ∈ [-64,63]
    let u: u8 = if c < 0 { (128 + c) as u8 } else { c as u8 }; // c mod 128
    let mut bits7 = [b.one(); 7];
    for i in 0..7 {
        bits7[i] = alloc_bool_ir(b, ((u >> i) & 1) == 1);
    }
    let c_var = b.new_var(i32_to_f257(c));
    b.enforce_lc_eq_zero(vec![
        (F257::ONE, c_var),
        (-F257::ONE, bits7[0]),
        (-F257::from(2u64), bits7[1]),
        (-F257::from(4u64), bits7[2]),
        (-F257::from(8u64), bits7[3]),
        (-F257::from(16u64), bits7[4]),
        (-F257::from(32u64), bits7[5]),
        (F257::from(64u64), bits7[6]),
    ]);
    c_var
}

#[inline]
fn alloc_carry_with_bound_ir(b: &mut IrBuilder<'_>, c: i32, bound: i32) -> VarRef {
    debug_assert!(bound >= 0);
    if bound <= 1 {
        alloc_carry_pm1_ir(b, c)
    } else if bound <= 2 {
        alloc_carry_pm2_ir(b, c)
    } else if bound <= 7 {
        alloc_carry_pm8_ir(b, c)
    } else if bound <= 15 {
        alloc_carry_pm16_ir(b, c)
    } else if bound <= 31 {
        alloc_carry_pm32_ir(b, c)
    } else if bound <= 63 {
        alloc_carry_pm64_ir(b, c)
    } else {
        // IMPORTANT (soundness): for bounds > 63 we do not have a cheap, sound range-check gadget.
        // If this triggers, it indicates a caller violated the preconditions / bound tracking.
        assert!(
            bound <= 63,
            "alloc_carry_with_bound_ir: unsupported bound={bound} (c={c}); caller violated preconditions"
        );
        // Unreachable, but keep a return for type-checking.
        alloc_carry_pm64_ir(b, c)
    }
}

/// Digitwise addition in the *loose* domain (no carries, no digit range checks).
pub(crate) fn add_bal16_loose_same_len_ir(b: &mut IrBuilder<'_>, a: &Bal16LooseIr, c: &Bal16LooseIr) -> Bal16LooseIr {
    assert_eq!(a.len(), c.len());
    let n = a.len();
    let mut out: Vec<VarRef> = Vec::with_capacity(n);
    for i in 0..n {
        let v = b.new_var(b.val(a.digits[i]) + b.val(c.digits[i]));
        // v = a_i + c_i
        b.enforce_lc_eq_zero(vec![(F257::ONE, v), (-F257::ONE, a.digits[i]), (-F257::ONE, c.digits[i])]);
        out.push(v);
    }
    Bal16LooseIr {
        digits: out,
        abs_bound: a.abs_bound.saturating_add(c.abs_bound),
    }
}

/// Normalize a *loose* base-16 digit vector into checked balanced digits ([-8,7]).
///
/// For each i: `loose_i + carry_i = checked_i + 16*carry_{i+1}`.
pub(crate) fn normalize_bal16_loose_same_len_ir(b: &mut IrBuilder<'_>, loose: &Bal16LooseIr) -> (Bal16CheckedIr, VarRef) {
    debug_assert!(loose.abs_bound >= 0);
    debug_assert!(loose.abs_bound < 128);
    let n = loose.len();
    let digit_abs_bound = loose.abs_bound;

    // Conservative carry bound schedule (same reasoning as builder-side).
    let mut carry_bound: i32 = 0;
    let mut carry_bounds: Vec<i32> = Vec::with_capacity(n);
    for _ in 0..n {
        let max_sum = digit_abs_bound + carry_bound;
        carry_bound = ((max_sum + 8) / 16) + 1;
        carry_bounds.push(carry_bound);
        debug_assert!(carry_bound < 128);
    }

    let mut out: Vec<VarRef> = Vec::with_capacity(n);
    let mut carry_i32: i32 = 0;
    // carry_0 = 0 (no var)
    let mut carry_var: Option<VarRef> = None;

    let div_floor = |x: i32, d: i32| -> i32 {
        debug_assert!(d > 0);
        if x >= 0 { x / d } else { -(((-x) + d - 1) / d) }
    };

    for i in 0..n {
        let dv = loose.digits[i];
        let di = f257_to_i32_bal(b.val(dv));
        debug_assert!(
            (-digit_abs_bound..=digit_abs_bound).contains(&di),
            "normalize_bal16_loose_same_len_ir: digit out of assumed bound"
        );
        let sum = di + carry_i32;

        let mut carry_next = div_floor(sum + 8, 16);
        let mut rem = sum - 16 * carry_next;
        while rem > 7 {
            carry_next += 1;
            rem -= 16;
        }
        while rem < -8 {
            carry_next -= 1;
            rem += 16;
        }
        debug_assert!((-8..=7).contains(&rem));
        debug_assert!(
            (-carry_bounds[i]..=carry_bounds[i]).contains(&carry_next),
            "normalize_bal16_loose_same_len_ir: carry out of bound"
        );

        let rem_digit = alloc_bal16_digit_ir(b, rem as i8);
        let carry_next_var = alloc_carry_with_bound_ir(b, carry_next, carry_bounds[i]);

        // loose_i + carry_i - rem_i - 16*carry_{i+1} = 0
        let mut lc = vec![
            (F257::ONE, dv),
            (-F257::ONE, rem_digit),
            (-F257::from(16u64), carry_next_var),
        ];
        if let Some(carryv) = carry_var {
            lc.insert(1, (F257::ONE, carryv));
        }
        b.enforce_lc_eq_zero(lc);

        out.push(rem_digit);
        carry_i32 = carry_next;
        carry_var = Some(carry_next_var);
    }

    (Bal16CheckedIr(out), carry_var.expect("normalize_bal16_loose_same_len_ir: non-empty"))
}

/// Add two balanced base-16 digit vectors of the same length.
///
/// Assumes each digit is in [-8,7]. Enforces output digits in [-8,7] and carry in {-1,0,1}.
pub(crate) fn add_bal16_same_len_ir(b: &mut IrBuilder<'_>, a: &Bal16CheckedIr, c: &Bal16CheckedIr) -> (Bal16CheckedIr, VarRef) {
    assert_eq!(a.len(), c.len());
    let n = a.len();
    let mut out: Vec<VarRef> = Vec::with_capacity(n);
    let mut carry_i32: i32 = 0;
    // carry starts at 0; do NOT allocate a var for it.
    let mut carry: Option<VarRef> = None;

    for i in 0..n {
        let ai = f257_to_i32_bal(b.val(a[i]));
        let ci = f257_to_i32_bal(b.val(c[i]));
        let sum = ai + ci + carry_i32;
        let mut carry_next = if sum >= 0 { (sum + 8) / 16 } else { -(((-sum) + 8) / 16) };
        let mut rem = sum - 16 * carry_next;
        while rem > 7 {
            carry_next += 1;
            rem -= 16;
        }
        while rem < -8 {
            carry_next -= 1;
            rem += 16;
        }
        debug_assert!((-1..=1).contains(&carry_next));
        debug_assert!((-8..=7).contains(&rem));

        let out_digit = alloc_bal16_digit_ir(b, rem as i8);
        let carry_next_var = alloc_carry_pm1_ir(b, carry_next);

        // a_i + c_i + carry - out_i - 16*carry_next = 0
        let mut lc = vec![
            (F257::ONE, a[i]),
            (F257::ONE, c[i]),
            (-F257::ONE, out_digit),
            (-F257::from(16u64), carry_next_var),
        ];
        if let Some(carry_var) = carry {
            lc.insert(2, (F257::ONE, carry_var));
        }
        b.enforce_lc_eq_zero(lc);

        out.push(out_digit);
        carry_i32 = carry_next;
        carry = Some(carry_next_var);
    }

    (Bal16CheckedIr(out), carry.expect("add_bal16_same_len_ir: non-empty input must produce carry var"))
}

/// Multiply two balanced base-16 digit vectors (little-endian), specialized for min(len)<=3.
///
/// Outputs little-endian digits with each digit in [-8,7], and a final carry digit.
pub(crate) fn mul_bal16_small_ir(b: &mut IrBuilder<'_>, a: &[VarRef], bb: &[VarRef]) -> Vec<VarRef> {
    let la = a.len();
    let lb = bb.len();
    assert!(la > 0 && lb > 0);
    assert!(la.min(lb) <= 3, "mul_bal16_small_ir requires min(len) <= 3");

    let mut out: Vec<VarRef> = Vec::with_capacity(la + lb);
    let mut carry_i32: i32 = 0;
    // carry starts at 0; do NOT allocate a var for it.
    let mut carry: Option<VarRef> = None;

    let div_floor = |x: i32, d: i32| -> i32 {
        debug_assert!(d > 0);
        if x >= 0 { x / d } else { -(((-x) + d - 1) / d) }
    };

    for k in 0..(la + lb - 1) {
        let mut sum: i32 = carry_i32;
        let mut prods: Vec<VarRef> = Vec::new();

        for i in 0..la {
            let j = k as i32 - i as i32;
            if j < 0 || j >= lb as i32 {
                continue;
            }
            let j = j as usize;
            let aval = f257_to_i32_bal(b.val(a[i]));
            let bval = f257_to_i32_bal(b.val(bb[j]));
            sum += aval * bval;

            let pv = b.new_var(b.val(a[i]) * b.val(bb[j]));
            b.enforce_mul(a[i], bb[j], pv);
            prods.push(pv);
        }

        let mut carry_next = div_floor(sum + 8, 16);
        let mut rem = sum - 16 * carry_next;
        while rem > 7 {
            carry_next += 1;
            rem -= 16;
        }
        while rem < -8 {
            carry_next -= 1;
            rem += 16;
        }
        debug_assert!((-8..=7).contains(&rem));
        debug_assert!((-14..=14).contains(&carry_next));

        let digit_var = alloc_bal16_digit_ir(b, rem as i8);
        // Use a simple statement-only carry allocator (superset of [-14,14]).
        let carry_next_var = alloc_carry_pm16_ir(b, carry_next);

        // carry + sum(prods) - digit - 16*carry_next = 0
        let mut lc: Vec<(F257, VarRef)> = Vec::with_capacity(2 + prods.len());
        if let Some(carryv) = carry {
            lc.push((F257::ONE, carryv));
        }
        for &p in &prods {
            lc.push((F257::ONE, p));
        }
        lc.push((-F257::ONE, digit_var));
        lc.push((-F257::from(16u64), carry_next_var));
        b.enforce_lc_eq_zero(lc);

        out.push(digit_var);
        carry_i32 = carry_next;
        carry = Some(carry_next_var);
    }

    out.push(carry.expect("mul_bal16_small_ir: non-empty must produce carry"));
    out
}

/// Rebalance the final digit of a `mul_bal16_small` product.
///
/// Input tail is assumed to be in [-11,11]; output replaces it with a balanced digit in [-8,7]
/// and appends a carry in [-2,2], enforcing: tail = rem + 16*carry2.
pub(crate) fn rebalance_tail_pm11_to_pm2_ir(b: &mut IrBuilder<'_>, digits: &[VarRef]) -> Vec<VarRef> {
    assert!(!digits.is_empty());
    let l = digits.len();
    let tail = f257_to_i32_bal(b.val(digits[l - 1]));
    debug_assert!((-11..=11).contains(&tail));

    let mut carry2 = if tail >= 0 { (tail + 8) / 16 } else { -(((-tail) + 8) / 16) };
    let mut rem = tail - 16 * carry2;
    while rem > 7 {
        carry2 += 1;
        rem -= 16;
    }
    while rem < -8 {
        carry2 -= 1;
        rem += 16;
    }
    debug_assert!((-8..=7).contains(&rem));
    debug_assert!((-2..=2).contains(&carry2));

    let rem_digit = alloc_bal16_digit_ir(b, rem as i8);
    let carry2_var = alloc_carry_pm2_ir(b, carry2);
    b.enforce_lc_eq_zero(vec![
        (F257::ONE, digits[l - 1]),
        (-F257::ONE, rem_digit),
        (-F257::from(16u64), carry2_var),
    ]);

    let mut out = Vec::with_capacity(l + 1);
    out.extend_from_slice(&digits[..l - 1]);
    out.push(rem_digit);
    out.push(carry2_var);
    out
}

/// Negate a balanced base-16 digit vector (little-endian), producing digits in [-8,7].
pub(crate) fn neg_bal16_digits_ir(b: &mut IrBuilder<'_>, x: &Bal16CheckedIr) -> (Bal16CheckedIr, VarRef) {
    let n = x.len();
    let mut out: Vec<VarRef> = Vec::with_capacity(n);
    let mut carry_i32: i32 = 0;
    // carry starts at 0; do NOT allocate a var for it.
    let mut carry: Option<VarRef> = None;

    for i in 0..n {
        let xi = f257_to_i32_bal(b.val(x[i]));
        let sum = (-xi) + carry_i32;
        let mut carry_next = if sum >= 0 { (sum + 8) / 16 } else { -(((-sum) + 8) / 16) };
        let mut rem = sum - 16 * carry_next;
        while rem > 7 {
            carry_next += 1;
            rem -= 16;
        }
        while rem < -8 {
            carry_next -= 1;
            rem += 16;
        }
        debug_assert!((-1..=1).contains(&carry_next));
        debug_assert!((-8..=7).contains(&rem));

        let out_digit = alloc_bal16_digit_ir(b, rem as i8);
        let carry_next_var = alloc_carry_pm1_ir(b, carry_next);

        // carry - x_i - out_i - 16*carry_next = 0
        let mut lc = vec![
            (-F257::ONE, x[i]),
            (-F257::ONE, out_digit),
            (-F257::from(16u64), carry_next_var),
        ];
        if let Some(carry_var) = carry {
            lc.insert(0, (F257::ONE, carry_var));
        }
        b.enforce_lc_eq_zero(lc);

        out.push(out_digit);
        carry_i32 = carry_next;
        carry = Some(carry_next_var);
    }

    (Bal16CheckedIr(out), carry.expect("neg_bal16_digits_ir: non-empty input must produce carry var"))
}

/// Compute the balanced-digit carry-out bit for a nibble.
///
/// Inputs are boolean vars:
/// - `carry_in` is the previous carry bit
/// - `b0,b1,b2,msb` are the 4 nibble bits (msb = sign bit)
///
/// Rule matches `digits::{u32,u64}_bytes_to_bal16_digits`:
///   carry_out = msb OR (carry_in AND (b0&b1&b2) AND !msb)
#[inline]
fn bal16_carry_out_ir(b: &mut IrBuilder<'_>, carry_in: VarRef, b0: VarRef, b1: VarRef, b2: VarRef, msb: VarRef) -> VarRef {
    // t01 = b0*b1
    let t01 = b.new_var(b.val(b0) * b.val(b1));
    b.enforce_mul(b0, b1, t01);
    // t012 = (b0*b1)*b2
    let t012 = b.new_var(b.val(t01) * b.val(b2));
    b.enforce_mul(t01, b2, t012);
    // carry_t = carry_in * (b0&b1&b2)
    let carry_t = b.new_var(b.val(carry_in) * b.val(t012));
    b.enforce_mul(carry_in, t012, carry_t);
    // carry_t_msb = carry_t * msb
    let carry_t_msb = b.new_var(b.val(carry_t) * b.val(msb));
    b.enforce_mul(carry_t, msb, carry_t_msb);
    // carry_out = msb + carry_t - carry_t_msb
    let c_out = b.new_var(b.val(msb) + b.val(carry_t) - b.val(carry_t_msb));
    b.enforce_affine(
        c_out,
        vec![(F257::ONE, msb), (F257::ONE, carry_t), (-F257::ONE, carry_t_msb)],
    );
    c_out
}

/// Convert 4 little-endian bytes (each already bit-decomposed) into balanced base-16 digits (len 9).
///
/// `bytes_bits[i][j]` is bit `j` of byte `i` (little-endian), where each bit var is boolean.
pub(crate) fn u32_bytes_to_bal16_digits_from_bits_ir(
    b: &mut IrBuilder<'_>,
    bytes_bits: &[[VarRef; 8]; 4],
) -> [VarRef; 9] {
    // carry starts at 0; do NOT allocate a var for it.
    let mut carry: Option<VarRef> = None;

    let mut out: [VarRef; 9] = core::array::from_fn(|_| b.one());
    let mut out_idx = 0usize;

    for byte in bytes_bits.iter() {
        // Low nibble then high nibble.
        for (b0, b1, b2, msb) in [
            (byte[0], byte[1], byte[2], byte[3]),
            (byte[4], byte[5], byte[6], byte[7]),
        ] {
            // When carry_in == 0, carry_out is exactly msb; avoid extra muls/vars.
            let c_out = if let Some(carry_in) = carry {
                bal16_carry_out_ir(b, carry_in, b0, b1, b2, msb)
            } else {
                msb
            };

            // These are all boolean witnesses; avoid bigint conversions in hot loops.
            let bit_i32 = |x: VarRef| -> i32 {
                let xv = b.val(x);
                debug_assert!(xv == F257::ZERO || xv == F257::ONE);
                if xv == F257::ONE { 1 } else { 0 }
            };
            let d_i: i32 = bit_i32(b0) + 2 * bit_i32(b1) + 4 * bit_i32(b2) + 8 * bit_i32(msb);
            let carry_i: i32 = carry.map(bit_i32).unwrap_or(0);
            let c_i: i32 = bit_i32(c_out);
            let v: i32 = d_i + carry_i - 16 * c_i;
            debug_assert!((-8..=7).contains(&v), "u32_bytes_to_bal16_digits_from_bits_ir: digit out of range");

            let out_digit = b.new_var(i32_to_f257(v));
            let mut terms = vec![
                (F257::ONE, b0),
                (F257::from(2u64), b1),
                (F257::from(4u64), b2),
                (F257::from(8u64), msb),
                (-F257::from(16u64), c_out),
            ];
            if let Some(carry_in) = carry {
                terms.insert(0, (F257::ONE, carry_in));
            }
            b.enforce_affine(out_digit, terms);

            out[out_idx] = out_digit;
            out_idx += 1;
            carry = Some(c_out);
        }
    }

    debug_assert_eq!(out_idx, 8);
    out[8] = carry.expect("u32_bytes_to_bal16_digits_from_bits_ir: expected final carry");
    out
}

/// Convert 8 little-endian bytes (each already bit-decomposed) into balanced base-16 digits (len 17).
///
/// `bytes_bits[i][j]` is bit `j` of byte `i` (little-endian), where each bit var is boolean.
pub(crate) fn u64_bytes_to_bal16_digits_from_bits_ir(
    b: &mut IrBuilder<'_>,
    bytes_bits: &[[VarRef; 8]; 8],
) -> [VarRef; 17] {
    // carry starts at 0; do NOT allocate a var for it.
    let mut carry: Option<VarRef> = None;

    let mut out: [VarRef; 17] = core::array::from_fn(|_| b.one());
    let mut out_idx = 0usize;

    for byte in bytes_bits.iter() {
        for (b0, b1, b2, msb) in [
            (byte[0], byte[1], byte[2], byte[3]),
            (byte[4], byte[5], byte[6], byte[7]),
        ] {
            let c_out = if let Some(carry_in) = carry {
                bal16_carry_out_ir(b, carry_in, b0, b1, b2, msb)
            } else {
                msb
            };

            // These are all boolean witnesses; avoid bigint conversions in hot loops.
            let bit_i32 = |x: VarRef| -> i32 {
                let xv = b.val(x);
                debug_assert!(xv == F257::ZERO || xv == F257::ONE);
                if xv == F257::ONE { 1 } else { 0 }
            };
            let d_i: i32 = bit_i32(b0) + 2 * bit_i32(b1) + 4 * bit_i32(b2) + 8 * bit_i32(msb);
            let carry_i: i32 = carry.map(bit_i32).unwrap_or(0);
            let c_i: i32 = bit_i32(c_out);
            let v: i32 = d_i + carry_i - 16 * c_i;
            debug_assert!((-8..=7).contains(&v), "u64_bytes_to_bal16_digits_from_bits_ir: digit out of range");

            let out_digit = b.new_var(i32_to_f257(v));
            let mut terms = vec![
                (F257::ONE, b0),
                (F257::from(2u64), b1),
                (F257::from(4u64), b2),
                (F257::from(8u64), msb),
                (-F257::from(16u64), c_out),
            ];
            if let Some(carry_in) = carry {
                terms.insert(0, (F257::ONE, carry_in));
            }
            b.enforce_affine(out_digit, terms);

            out[out_idx] = out_digit;
            out_idx += 1;
            carry = Some(c_out);
        }
    }

    debug_assert_eq!(out_idx, 16);
    out[16] = carry.expect("u64_bytes_to_bal16_digits_from_bits_ir: expected final carry");
    out
}

/// Allocate a balanced base-16 digit in [-8,7] (witnessed).
///
/// This mirrors `digits::alloc_bal16_digit` but emits into IR.
pub(crate) fn alloc_bal16_digit_ir(b: &mut IrBuilder<'_>, d: i8) -> VarRef {
    assert!((-8..=7).contains(&d));
    let nib = if d < 0 { (d as i16 + 16) as u8 } else { d as u8 };
    let mut bits4: [VarRef; 4] = core::array::from_fn(|_| VarRef::Base(0));
    for i in 0..4 {
        bits4[i] = alloc_bool_ir(b, ((nib >> i) & 1) == 1);
    }
    let out = b.new_var(i32_to_f257(d as i32));
    debug_assert_eq!(
        f257_to_i32_bal(b.val(out)),
        d as i32,
        "alloc_bal16_digit_ir: witness mismatch"
    );
    // out = b0 + 2*b1 + 4*b2 - 8*b3  (equivalently +8*b3 on LHS)
    b.enforce_lc_eq_zero(vec![
        (F257::ONE, out),
        (-F257::ONE, bits4[0]),
        (-F257::from(2u64), bits4[1]),
        (-F257::from(4u64), bits4[2]),
        (F257::from(8u64), bits4[3]),
    ]);
    out
}

/// Allocate a balanced base-4 digit `d ∈ {-2,-1,0,1}` with a 2-bit membership check.
///
/// Encoding:
///   d = b0 + 2*b1 - 2, where b0,b1 ∈ {0,1}.
///
/// This is the minimal cheap "digit is in the intended set" constraint. It is important for
/// soundness when digits are used as multiplicands: otherwise the carry-chain equalities can
/// admit alternative solutions with digits shifted by multiples of 4.
#[inline]
fn alloc_bal4_digit_ir(b: &mut IrBuilder<'_>, d: i32) -> VarRef {
    debug_assert!((-2..=1).contains(&d));
    let enc: u8 = (d + 2) as u8; // in 0..=3
    let b0 = alloc_bool_ir(b, (enc & 1) == 1);
    let b1 = alloc_bool_ir(b, (enc & 2) == 2);
    let dvar = b.new_var(i32_to_f257(d));
    // d - b0 - 2*b1 + 2 = 0  => d = b0 + 2*b1 - 2
    b.enforce_lc_eq_zero(vec![
        (F257::ONE, dvar),
        (-F257::ONE, b0),
        (-F257::from(2u64), b1),
        (F257::from(2u64), b.one()),
    ]);
    dvar
}

// -----------------------------------------------------------------------------
// Balanced base-4 digits (sound mul check: avoids 16^2+1 bubble)
// -----------------------------------------------------------------------------

#[inline]
fn u64_to_bal4_digits_le_const(x: u64) -> [i8; 33] {
    // Balanced base-4 digits in [-2,1] with a final carry bit in {0,1}.
    let mut out = [0i8; 33];
    let mut carry: i8 = 0;
    for i in 0..32 {
        let chunk = ((x >> (2 * i)) & 0x3) as i8; // 0..3
        let v = chunk + carry; // 0..4
        if v >= 2 {
            out[i] = v - 4; // -2,-1,0
            carry = 1;
        } else {
            out[i] = v; // 0,1
            carry = 0;
        }
    }
    out[32] = carry;
    out
}

/// Allocate u64 as balanced base-4 digits with explicit digit membership checks.
///
/// This is more expensive than `_raw_` but closes a soundness hole: without digit membership
/// constraints, a carry-chain equation can admit alternative solutions with digits shifted by
/// multiples of 4 while keeping bounded carries, and those digits may later be used as
/// multiplicands.
#[inline]
fn alloc_u64_as_bal4_digits_checked_ir(b: &mut IrBuilder<'_>, x: u64) -> [VarRef; 33] {
    let ds = u64_to_bal4_digits_le_const(x);
    debug_assert!(ds[32] == 0 || ds[32] == 1);
    let mut out: [VarRef; 33] = [VarRef::Base(0); 33];
    for i in 0..32 {
        out[i] = alloc_bal4_digit_ir(b, ds[i] as i32);
    }
    out[32] = alloc_bool_ir(b, ds[32] == 1);
    out
}

// NOTE (soundness): we intentionally do **not** bit-decompose/range-check every base-4 digit
// variable when it is already linked by a *bounded carry chain*.
//
// The classic underconstraint in F257 comes from the "carry bubble":
//   257 = 4*64 + 1  (analogously 257 = 16^2 + 1 in base-16)
//
// If carries were unconstrained, you could mutate an internal step by:
//   carry_next += 64
//   digit      += 1
// and keep `digit - 4*carry_next` unchanged mod 257 (since 4*64 = 256 ≡ -1).
//
// We prevent this by range-checking carries in a window of width < 64 (we use pm4/pm8/pm16/pm32,
// i.e. |carry| <= 31), so "carry += 64" is impossible. Therefore the carry chain is injective
// over the intended integer lift even though digits themselves are not bit-decomposed.

/// Convert checked bal4 digits (33) into checked bal16 digits (17).
///
/// This is used to return `[VarRef; 17]` to the rest of the bal16-based IR, while keeping the
/// multiplication soundness check in base-4. It is significantly cheaper than bal16->bal4.
pub(crate) fn bal4_to_bal16_digits_ir(b: &mut IrBuilder<'_>, x4: &[VarRef; 33]) -> [VarRef; 17] {
    let mut out: [VarRef; 17] = [VarRef::Base(0); 17];

    // carry_0 = 0 (no var)
    let mut carry_i32: i32 = 0;
    let mut carry_var: Option<VarRef> = None;

    for i in 0..16 {
        let a = x4[2 * i];
        let b4 = x4[2 * i + 1];
        let ai = f257_to_i32_bal(b.val(a));
        let bi = f257_to_i32_bal(b.val(b4));
        debug_assert!((-2..=1).contains(&ai));
        debug_assert!((-2..=1).contains(&bi));

        // sum in base-16 digit slot
        let sum = ai + 4 * bi + carry_i32;
        let mut carry_next = if sum >= 0 { (sum + 8) / 16 } else { -(((-sum) + 8) / 16) };
        let mut rem = sum - 16 * carry_next;
        while rem > 7 {
            carry_next += 1;
            rem -= 16;
        }
        while rem < -8 {
            carry_next -= 1;
            rem += 16;
        }
        debug_assert!((-1..=1).contains(&carry_next));
        debug_assert!((-8..=7).contains(&rem));

        let rem_var = alloc_bal16_digit_ir(b, rem as i8);
        let carry_next_var = alloc_carry_pm1_ir(b, carry_next);

        // a + 4*b + carry - rem - 16*carry_next = 0
        let mut lc = vec![
            (F257::ONE, a),
            (F257::from(4u64), b4),
            (-F257::ONE, rem_var),
            (-F257::from(16u64), carry_next_var),
        ];
        if let Some(cv) = carry_var {
            lc.insert(2, (F257::ONE, cv));
        } else {
            // carry_0 is 0 (no var)
        }
        b.enforce_lc_eq_zero(lc);

        out[i] = rem_var;
        carry_i32 = carry_next;
        carry_var = Some(carry_next_var);
    }

    // The top bit (2^64 == 16^16) is x4[32] plus any carry out from the low 64 bits.
    let carry16 = carry_var.expect("bal4_to_bal16_digits_ir: expected carry var");
    let top_w = b.val(x4[32]) + b.val(carry16);
    // top should be boolean (0 or 1)
    let top = b.new_var(top_w);
    enforce_bit_var_ir(b, top);
    // top = x4[32] + carry16
    b.enforce_lc_eq_zero(vec![
        (F257::ONE, top),
        (-F257::ONE, x4[32]),
        (-F257::ONE, carry16),
    ]);
    out[16] = top;
    out
}

/// Convert a checked bal16 scalar into checked bal4 digits via a base-4 carry chain.
///
/// This is the minimal "drop-in" soundness fix for multiplication gadgets:
/// it avoids the 16^2+1 degeneracy by doing the multiplication relation check in base-4,
/// where the problematic bubble corresponds to a ±64 carry jump.
fn bal16_to_bal4_digits_ir(b: &mut IrBuilder<'_>, x16: &[VarRef; 17]) -> [VarRef; 33] {
    let z = alloc_zero_const_ir(b);
    let mut out: [VarRef; 33] = [VarRef::Base(0); 33];
    // carry_0 = 0 (use a constant-zero var; no need to range-check)
    let mut carry_var = z;
    let mut carry_i32: i32 = 0;

    for k in 0..33 {
        let term_var: VarRef = if (k & 1) == 0 { x16[k / 2] } else { z };
        let term_i32: i32 = if (k & 1) == 0 { f257_to_i32_bal(b.val(x16[k / 2])) } else { 0 };

        let s = term_i32 + carry_i32;
        let rem = ((s % 4) + 4) % 4; // 0..3
        let digit = if rem >= 2 { rem - 4 } else { rem }; // -2,-1,0,1
        let carry_next = (s - digit) / 4;

        debug_assert!((-2..=1).contains(&digit));
        if (k & 1) == 0 {
            // Even step includes a base-16 digit term in [-8,7], so carry can reach ±2.
            debug_assert!((-2..=2).contains(&carry_next));
        } else {
            // Odd step term is 0, so carry shrinks to {-1,0,1}.
            debug_assert!((-1..=1).contains(&carry_next));
        }

        // IMPORTANT (soundness):
        // - For k < 32: digits are used as multiplicands later (NTT + pointwise mul), so enforce
        //   membership `digit ∈ {-2,-1,0,1}` via `alloc_bal4_digit_ir`.
        // - For k == 32: this digit is the 2^64 "top bit" in the base-4 view used by
        //   `digits4_to_u64_witness_ir`, so it MUST be boolean (0/1), not just in {-2,-1,0,1}.
        let dvar = if k == 32 {
            debug_assert!(digit == 0 || digit == 1, "bal16_to_bal4_digits_ir: top digit must be 0/1");
            alloc_bool_ir(b, digit == 1)
        } else {
            alloc_bal4_digit_ir(b, digit)
        };
        let cnext = if (k & 1) == 0 {
            alloc_carry_pm2_ir(b, carry_next)
        } else {
            alloc_carry_pm1_ir(b, carry_next)
        };

        // term + carry - digit - 4*carry_next = 0
        b.enforce_lc_eq_zero(vec![
            (F257::ONE, term_var),
            (F257::ONE, carry_var),
            (-F257::ONE, dvar),
            (-F257::from(4u64), cnext),
        ]);

        out[k] = dvar;
        carry_var = cnext;
        carry_i32 = carry_next;
    }

    b.ir.enforce_var_eq_const(carry_var, F257::ZERO);
    out
}

#[inline]
fn alloc_carry_pm16_bal4_ir(b: &mut IrBuilder<'_>, c: i32) -> VarRef {
    debug_assert!((-16..=15).contains(&c));
    let out = b.new_var(i32_to_f257(c));
    let u: u8 = if c < 0 { (32 + c) as u8 } else { c as u8 }; // c mod 32
    let mut bits5 = [b.one(); 5];
    for i in 0..5 {
        bits5[i] = alloc_bool_ir(b, ((u >> i) & 1) == 1);
    }
    // out = b0 + 2b1 + 4b2 + 8b3 - 16b4
    b.enforce_lc_eq_zero(vec![
        (F257::ONE, out),
        (-F257::ONE, bits5[0]),
        (-F257::from(2u64), bits5[1]),
        (-F257::from(4u64), bits5[2]),
        (-F257::from(8u64), bits5[3]),
        (F257::from(16u64), bits5[4]),
    ]);
    out
}

#[inline]
fn alloc_carry_pm8_bal4_ir(b: &mut IrBuilder<'_>, c: i32) -> VarRef {
    debug_assert!((-8..=7).contains(&c));
    let out = b.new_var(i32_to_f257(c));
    let u: u8 = if c < 0 { (16 + c) as u8 } else { c as u8 }; // c mod 16
    let mut bits4 = [b.one(); 4];
    for i in 0..4 {
        bits4[i] = alloc_bool_ir(b, ((u >> i) & 1) == 1);
    }
    // out = b0 + 2b1 + 4b2 - 8b3
    b.enforce_lc_eq_zero(vec![
        (F257::ONE, out),
        (-F257::ONE, bits4[0]),
        (-F257::from(2u64), bits4[1]),
        (-F257::from(4u64), bits4[2]),
        (F257::from(8u64), bits4[3]),
    ]);
    out
}

#[inline]
fn alloc_carry_pm4_bal4_ir(b: &mut IrBuilder<'_>, c: i32) -> VarRef {
    debug_assert!((-4..=3).contains(&c));
    let out = b.new_var(i32_to_f257(c));
    let u: u8 = if c < 0 { (8 + c) as u8 } else { c as u8 }; // c mod 8
    let mut bits3 = [b.one(); 3];
    for i in 0..3 {
        bits3[i] = alloc_bool_ir(b, ((u >> i) & 1) == 1);
    }
    // out = b0 + 2b1 - 4b2
    b.enforce_lc_eq_zero(vec![
        (F257::ONE, out),
        (-F257::ONE, bits3[0]),
        (-F257::from(2u64), bits3[1]),
        (F257::from(4u64), bits3[2]),
    ]);
    out
}

fn enforce_prod_const_eq_qp_plus_r_bal4_ir_with_carry_schedule(
    b: &mut IrBuilder<'_>,
    x4: &[VarRef; 33],
    c4_const: &[i8; 33],
    q4: &[VarRef; 33],
    r4: &[VarRef; 33],
    carry_bounds: &[i32; 66], // per-step |carry_next| bound
) {
    // carry_0 = 0
    let mut carry_var = if carry_bounds[0] <= 3 {
        alloc_carry_pm4_bal4_ir(b, 0)
    } else if carry_bounds[0] <= 7 {
        alloc_carry_pm8_bal4_ir(b, 0)
    } else if carry_bounds[0] <= 15 {
        alloc_carry_pm16_bal4_ir(b, 0)
    } else {
        alloc_carry_pm32_ir(b, 0)
    };
    let mut carry_i32: i32 = 0;

    for k in 0..66 {
        let mut sum: i32 = carry_i32;
        // Hot path: keep this tight to reduce allocator pressure.
        // Empirically this stays well below ~80 terms.
        let mut lc: Vec<(F257, VarRef)> = Vec::with_capacity(96);
        lc.push((F257::ONE, carry_var));

        // + Σ_i x_i * c_{k-i}
        for i in 0..33 {
            let j = k as i32 - i as i32;
            if !(0..33).contains(&j) {
                continue;
            }
            let dj = c4_const[j as usize] as i32;
            if dj == 0 {
                continue;
            }
            let xi = f257_to_i32_bal(b.val(x4[i]));
            sum += xi * dj;
            lc.push((i32_to_f257(dj), x4[i]));
        }

        // - r_k
        if k < 33 {
            let rk = f257_to_i32_bal(b.val(r4[k]));
            sum -= rk;
            lc.push((-F257::ONE, r4[k]));
        }

        // - q*p where p = 4^32 - 4^16 + 1  => q*p = q - q<<16 + q<<32
        if k < 33 {
            let qk = f257_to_i32_bal(b.val(q4[k]));
            sum -= qk;
            lc.push((-F257::ONE, q4[k]));
        }
        if k >= 16 && (k - 16) < 33 {
            let qk = f257_to_i32_bal(b.val(q4[k - 16]));
            sum += qk;
            lc.push((F257::ONE, q4[k - 16]));
        }
        if k >= 32 && (k - 32) < 33 {
            let qk = f257_to_i32_bal(b.val(q4[k - 32]));
            sum -= qk;
            lc.push((-F257::ONE, q4[k - 32]));
        }

        debug_assert!(sum % 4 == 0, "bal4 const-mul carry not divisible: sum={sum} k={k}");
        let carry_next: i32 = sum / 4;
        debug_assert!(
            (-carry_bounds[k]..=carry_bounds[k]).contains(&carry_next),
            "bal4 const-mul carry out of bound: {carry_next} at k={k} (sum={sum}, bound={})",
            carry_bounds[k]
        );
        let carry_next_var = if carry_bounds[k] <= 7 {
            if carry_bounds[k] <= 3 {
                alloc_carry_pm4_bal4_ir(b, carry_next)
            } else {
                alloc_carry_pm8_bal4_ir(b, carry_next)
            }
        } else if carry_bounds[k] <= 15 {
            alloc_carry_pm16_bal4_ir(b, carry_next)
        } else {
            alloc_carry_pm32_ir(b, carry_next)
        };
        lc.push((-F257::from(4u64), carry_next_var));

        b.enforce_lc_eq_zero(lc);
        carry_var = carry_next_var;
        carry_i32 = carry_next;
    }

    b.ir.enforce_var_eq_const(carry_var, F257::ZERO);
}

fn enforce_prod_var_eq_qp_plus_r_bal4_ir(
    b: &mut IrBuilder<'_>,
    a4: &[VarRef; 33],
    c4: &[VarRef; 33],
    q4: &[VarRef; 33],
    r4: &[VarRef; 33],
) {
    // Second Karatsuba cut: avoid 17×17 and 16×16 naive inner multiplies.
    //
    // We still do the *sound* base-4 carry-chain enforcement; we just reduce digit×digit muls.
    //
    // Top split at 17:
    //   a = a0 + 4^17 a1   (|a0|:17, |a1|:16)
    //   c = c0 + 4^17 c1
    //   a*c = z0 + 4^17 z1 + 4^34 z2
    //     z0 = a0*c0
    //     z2 = a1*c1
    //     z1 = (a0+a1)*(c0+c1) - z0 - z2
    //
    // For each inner product, use one more Karatsuba split:
    //   17 -> 9+8   (226 muls vs 289)
    //   16 -> 8+8   (192 muls vs 256)
    //
    // Total mul constraints per var-mul: 226 + 192 + 226 = 644 (vs 834).
    const M: usize = 17;
    const HI: usize = 16;
    const M0: usize = 9;
    const H0: usize = 8;

    #[derive(Clone)]
    struct Kara17 {
        ll: [[VarRef; M0]; M0],
        ll_v: [[i32; M0]; M0],
        hh: [[VarRef; H0]; H0],
        hh_v: [[i32; H0]; H0],
        ss: [[VarRef; M0]; M0],
        ss_v: [[i32; M0]; M0],
    }
    #[derive(Clone)]
    struct Kara16 {
        ll: [[VarRef; H0]; H0],
        ll_v: [[i32; H0]; H0],
        hh: [[VarRef; H0]; H0],
        hh_v: [[i32; H0]; H0],
        ss: [[VarRef; H0]; H0],
        ss_v: [[i32; H0]; H0],
    }

    let z = alloc_zero_const_ir(b);

    // Helper: push +/-1 * var into LC and update integer-lift sum using precomputed witness.
    #[inline]
    fn push_pm1(lc: &mut Vec<(F257, VarRef)>, sum: &mut i32, var: VarRef, var_v: i32, sign: i32) {
        debug_assert!(sign == 1 || sign == -1);
        *sum += sign * var_v;
        if sign == 1 {
            lc.push((F257::ONE, var));
        } else {
            lc.push((-F257::ONE, var));
        }
    }

    // Build a Karatsuba(9+8) multiplication "oracle" that can contribute coefficient `deg` of A*C
    // as a linear combination of bounded product vars.
    let kara17_build = |b: &mut IrBuilder<'_>, a: &[VarRef; M], c: &[VarRef; M]| -> Kara17 {
        // ll: a[0..9] * c[0..9]
        let mut ll: [[VarRef; M0]; M0] = [[VarRef::Base(0); M0]; M0];
        let mut ll_v: [[i32; M0]; M0] = [[0i32; M0]; M0];
        for i in 0..M0 {
            for j in 0..M0 {
                let pij = b.new_var(b.val(a[i]) * b.val(c[j]));
                b.enforce_mul(a[i], c[j], pij);
                ll[i][j] = pij;
                ll_v[i][j] = f257_to_i32_bal(b.val(pij));
            }
        }
        // hh: a[9..17] * c[9..17]
        let mut hh: [[VarRef; H0]; H0] = [[VarRef::Base(0); H0]; H0];
        let mut hh_v: [[i32; H0]; H0] = [[0i32; H0]; H0];
        for i in 0..H0 {
            for j in 0..H0 {
                let ai = a[M0 + i];
                let cj = c[M0 + j];
                let pij = b.new_var(b.val(ai) * b.val(cj));
                b.enforce_mul(ai, cj, pij);
                hh[i][j] = pij;
                hh_v[i][j] = f257_to_i32_bal(b.val(pij));
            }
        }
        // ss: (aL + aHpad) * (cL + cHpad), with pad at index 8.
        let mut sa: [VarRef; M0] = [VarRef::Base(0); M0];
        let mut sc: [VarRef; M0] = [VarRef::Base(0); M0];
        for i in 0..M0 {
            let ahi = if i < H0 { a[M0 + i] } else { z };
            let chi = if i < H0 { c[M0 + i] } else { z };
            let s0 = b.new_var(b.val(a[i]) + b.val(ahi));
            let s1 = b.new_var(b.val(c[i]) + b.val(chi));
            b.enforce_lc_eq_zero(vec![(F257::ONE, s0), (-F257::ONE, a[i]), (-F257::ONE, ahi)]);
            b.enforce_lc_eq_zero(vec![(F257::ONE, s1), (-F257::ONE, c[i]), (-F257::ONE, chi)]);
            sa[i] = s0;
            sc[i] = s1;
        }
        let mut ss: [[VarRef; M0]; M0] = [[VarRef::Base(0); M0]; M0];
        let mut ss_v: [[i32; M0]; M0] = [[0i32; M0]; M0];
        for i in 0..M0 {
            for j in 0..M0 {
                let pij = b.new_var(b.val(sa[i]) * b.val(sc[j]));
                b.enforce_mul(sa[i], sc[j], pij);
                ss[i][j] = pij;
                ss_v[i][j] = f257_to_i32_bal(b.val(pij));
            }
        }
        Kara17 { ll, ll_v, hh, hh_v, ss, ss_v }
    };

    let kara16_build = |b: &mut IrBuilder<'_>, a: &[VarRef; HI], c: &[VarRef; HI]| -> Kara16 {
        // ll: a[0..8] * c[0..8]
        let mut ll: [[VarRef; H0]; H0] = [[VarRef::Base(0); H0]; H0];
        let mut ll_v: [[i32; H0]; H0] = [[0i32; H0]; H0];
        for i in 0..H0 {
            for j in 0..H0 {
                let pij = b.new_var(b.val(a[i]) * b.val(c[j]));
                b.enforce_mul(a[i], c[j], pij);
                ll[i][j] = pij;
                ll_v[i][j] = f257_to_i32_bal(b.val(pij));
            }
        }
        // hh: a[8..16] * c[8..16]
        let mut hh: [[VarRef; H0]; H0] = [[VarRef::Base(0); H0]; H0];
        let mut hh_v: [[i32; H0]; H0] = [[0i32; H0]; H0];
        for i in 0..H0 {
            for j in 0..H0 {
                let ai = a[H0 + i];
                let cj = c[H0 + j];
                let pij = b.new_var(b.val(ai) * b.val(cj));
                b.enforce_mul(ai, cj, pij);
                hh[i][j] = pij;
                hh_v[i][j] = f257_to_i32_bal(b.val(pij));
            }
        }
        // ss: (aL+aH)*(cL+cH)
        let mut sa: [VarRef; H0] = [VarRef::Base(0); H0];
        let mut sc: [VarRef; H0] = [VarRef::Base(0); H0];
        for i in 0..H0 {
            let s0 = b.new_var(b.val(a[i]) + b.val(a[H0 + i]));
            let s1 = b.new_var(b.val(c[i]) + b.val(c[H0 + i]));
            b.enforce_lc_eq_zero(vec![(F257::ONE, s0), (-F257::ONE, a[i]), (-F257::ONE, a[H0 + i])]);
            b.enforce_lc_eq_zero(vec![(F257::ONE, s1), (-F257::ONE, c[i]), (-F257::ONE, c[H0 + i])]);
            sa[i] = s0;
            sc[i] = s1;
        }
        let mut ss: [[VarRef; H0]; H0] = [[VarRef::Base(0); H0]; H0];
        let mut ss_v: [[i32; H0]; H0] = [[0i32; H0]; H0];
        for i in 0..H0 {
            for j in 0..H0 {
                let pij = b.new_var(b.val(sa[i]) * b.val(sc[j]));
                b.enforce_mul(sa[i], sc[j], pij);
                ss[i][j] = pij;
                ss_v[i][j] = f257_to_i32_bal(b.val(pij));
            }
        }
        Kara16 { ll, ll_v, hh, hh_v, ss, ss_v }
    };

    // Big split: sum_a = a0 + a1pad (len 17), sum_c likewise.
    let mut sum_a: [VarRef; M] = [VarRef::Base(0); M];
    let mut sum_c: [VarRef; M] = [VarRef::Base(0); M];
    for i in 0..M {
        let ahi = if i < HI { a4[M + i] } else { z };
        let chi = if i < HI { c4[M + i] } else { z };
        let sa = b.new_var(b.val(a4[i]) + b.val(ahi));
        let sc = b.new_var(b.val(c4[i]) + b.val(chi));
        b.enforce_lc_eq_zero(vec![(F257::ONE, sa), (-F257::ONE, a4[i]), (-F257::ONE, ahi)]);
        b.enforce_lc_eq_zero(vec![(F257::ONE, sc), (-F257::ONE, c4[i]), (-F257::ONE, chi)]);
        sum_a[i] = sa;
        sum_c[i] = sc;
    }

    let a0: [VarRef; M] = core::array::from_fn(|i| a4[i]);
    let c0: [VarRef; M] = core::array::from_fn(|i| c4[i]);
    let a1: [VarRef; HI] = core::array::from_fn(|i| a4[M + i]);
    let c1: [VarRef; HI] = core::array::from_fn(|i| c4[M + i]);

    let kara_z0 = kara17_build(b, &a0, &c0);
    let kara_sprod = kara17_build(b, &sum_a, &sum_c);
    let kara_z2 = kara16_build(b, &a1, &c1);

    #[inline]
    fn kara17_add_coeff(
        k: &Kara17,
        deg: usize,
        lc: &mut Vec<(F257, VarRef)>,
        sum: &mut i32,
        sign: i32,
    ) {
        // deg in 0..=32
        // z0 part: ll
        if deg <= 2 * (M0 - 1) {
            let i_min = deg.saturating_sub(M0 - 1);
            let i_max = core::cmp::min(M0 - 1, deg);
            for i in i_min..=i_max {
                let j = deg - i;
                push_pm1(lc, sum, k.ll[i][j], k.ll_v[i][j], sign);
            }
        }
        // middle: ss - ll - hh, shifted by +9
        if (M0..=(M0 + 2 * (M0 - 1))).contains(&deg) {
            let t = deg - M0;
            let i_min = t.saturating_sub(M0 - 1);
            let i_max = core::cmp::min(M0 - 1, t);
            for i in i_min..=i_max {
                let j = t - i;
                push_pm1(lc, sum, k.ss[i][j], k.ss_v[i][j], sign);
                // - ll
                push_pm1(lc, sum, k.ll[i][j], k.ll_v[i][j], -sign);
                // - hh (only if within 8x8 and t<=14)
                if i < H0 && j < H0 && t <= 2 * (H0 - 1) {
                    push_pm1(lc, sum, k.hh[i][j], k.hh_v[i][j], -sign);
                }
            }
        }
        // high: hh shifted by +18
        if deg >= 2 * M0 && deg <= 2 * M0 + 2 * (H0 - 1) {
            let t = deg - 2 * M0;
            let i_min = t.saturating_sub(H0 - 1);
            let i_max = core::cmp::min(H0 - 1, t);
            for i in i_min..=i_max {
                let j = t - i;
                push_pm1(lc, sum, k.hh[i][j], k.hh_v[i][j], sign);
            }
        }
    }

    #[inline]
    fn kara16_add_coeff(
        k: &Kara16,
        deg: usize,
        lc: &mut Vec<(F257, VarRef)>,
        sum: &mut i32,
        sign: i32,
    ) {
        // deg in 0..=30
        if deg <= 2 * (H0 - 1) {
            let i_min = deg.saturating_sub(H0 - 1);
            let i_max = core::cmp::min(H0 - 1, deg);
            for i in i_min..=i_max {
                let j = deg - i;
                push_pm1(lc, sum, k.ll[i][j], k.ll_v[i][j], sign);
            }
        }
        if (H0..=(H0 + 2 * (H0 - 1))).contains(&deg) {
            let t = deg - H0;
            let i_min = t.saturating_sub(H0 - 1);
            let i_max = core::cmp::min(H0 - 1, t);
            for i in i_min..=i_max {
                let j = t - i;
                push_pm1(lc, sum, k.ss[i][j], k.ss_v[i][j], sign);
                push_pm1(lc, sum, k.ll[i][j], k.ll_v[i][j], -sign);
                push_pm1(lc, sum, k.hh[i][j], k.hh_v[i][j], -sign);
            }
        }
        if deg >= 2 * H0 && deg <= 2 * H0 + 2 * (H0 - 1) {
            let t = deg - 2 * H0;
            let i_min = t.saturating_sub(H0 - 1);
            let i_max = core::cmp::min(H0 - 1, t);
            for i in i_min..=i_max {
                let j = t - i;
                push_pm1(lc, sum, k.hh[i][j], k.hh_v[i][j], sign);
            }
        }
    }

    // Statement-derived per-step carry bounds:
    // - |a_i|,|c_j| <= 2 (balanced base-4 digits), so each product term <= 4.
    // - At step k, number of product terms is t_k = |{(i,j): i+j=k, 0<=i,j<=32}|.
    // - q*p contributes at most 3 terms of q (each <=2) => <=6, and r digit <=2.
    // So non-carry magnitude <= 4*t_k + 8. Propagate carry bound via:
    //   |carry_{k+1}| <= ceil((|carry_k| + bound_k)/4).
    let mut carry_bounds: [i32; 66] = [31; 66];
    let mut carry_bound_raw: i32 = 0;
    for k in 0..66 {
        let t_k: i32 = if k <= 32 {
            (k as i32) + 1
        } else if k <= 64 {
            (65 - k) as i32
        } else {
            0
        };
        let bound_k: i32 = 4 * t_k + 8;
        carry_bound_raw = (carry_bound_raw + bound_k + 3) / 4;
        carry_bounds[k] = carry_bound_raw;
    }

    let mut carry_var = alloc_carry_pm4_bal4_ir(b, 0);
    let mut carry_i32: i32 = 0;

    for k in 0..66 {
        let mut sum: i32 = carry_i32;
        // Hot path: keep this tight to reduce allocator pressure.
        // Empirically `max_terms_a` is < 80 (see ringmul IR stats), and total lc terms stays < 96.
        let mut lc: Vec<(F257, VarRef)> = Vec::with_capacity(96);
        lc.push((F257::ONE, carry_var));

        // - r_k
        if k < 33 {
            let rk = f257_to_i32_bal(b.val(r4[k]));
            sum -= rk;
            lc.push((-F257::ONE, r4[k]));
        }

        // + a*c via Karatsuba pieces (z0 + 4^17 z1 + 4^34 z2), with inner Karatsuba(9+8)/(8+8).
        //
        // z0 contribution: degree k in a0*c0 (0..32)
        if k <= 2 * (M - 1) {
            kara17_add_coeff(&kara_z0, k, &mut lc, &mut sum, 1);
        }
        // z1 contribution: degree t=k-17 in (a0+a1)*(c0+c1) - z0 - z2, shifted by +17.
        if (M..=(M + 2 * (M - 1))).contains(&k) {
            let t = k - M; // 0..32
            kara17_add_coeff(&kara_sprod, t, &mut lc, &mut sum, 1);
            kara17_add_coeff(&kara_z0, t, &mut lc, &mut sum, -1);
            if t <= 2 * (HI - 1) {
                kara16_add_coeff(&kara_z2, t, &mut lc, &mut sum, -1);
            }
        }
        // z2 contribution: degree u=k-34 in a1*c1, shifted by +34.
        if k >= 2 * M && k <= 2 * M + 2 * (HI - 1) {
            let u = k - 2 * M; // 0..30
            kara16_add_coeff(&kara_z2, u, &mut lc, &mut sum, 1);
        }

        // - q*p  (same sparse p)
        if k < 33 {
            let qk = f257_to_i32_bal(b.val(q4[k]));
            sum -= qk;
            lc.push((-F257::ONE, q4[k]));
        }
        if k >= 16 && (k - 16) < 33 {
            let qk = f257_to_i32_bal(b.val(q4[k - 16]));
            sum += qk;
            lc.push((F257::ONE, q4[k - 16]));
        }
        if k >= 32 && (k - 32) < 33 {
            let qk = f257_to_i32_bal(b.val(q4[k - 32]));
            sum -= qk;
            lc.push((-F257::ONE, q4[k - 32]));
        }

        debug_assert!(sum % 4 == 0, "bal4 var-mul carry not divisible: sum={sum} k={k}");
        let carry_next: i32 = sum / 4;
        debug_assert!(
            (-carry_bounds[k]..=carry_bounds[k]).contains(&carry_next),
            "bal4 var-mul carry out of bound: {carry_next} at k={k} (sum={sum}, bound={})",
            carry_bounds[k]
        );
        let carry_next_var = if carry_bounds[k] <= 3 {
            alloc_carry_pm4_bal4_ir(b, carry_next)
        } else if carry_bounds[k] <= 7 {
            alloc_carry_pm8_bal4_ir(b, carry_next)
        } else if carry_bounds[k] <= 15 {
            alloc_carry_pm16_bal4_ir(b, carry_next)
        } else {
            alloc_carry_pm32_ir(b, carry_next)
        };
        lc.push((-F257::from(4u64), carry_next_var));

        b.enforce_lc_eq_zero(lc);
        carry_var = carry_next_var;
        carry_i32 = carry_next;
    }

    b.ir.enforce_var_eq_const(carry_var, F257::ZERO);
}

/// Allocate a signed carry `c ∈ [-2,2]` as an F257 variable, with the vanishing polynomial gadget.
///
/// Mirrors `digits::alloc_carry_pm2`.
pub(crate) fn alloc_carry_pm2_ir(b: &mut IrBuilder<'_>, c: i32) -> VarRef {
    assert!((-2..=2).contains(&c));
    let c_var = b.new_var(i32_to_f257(c));

    let cv = b.val(c_var);
    // t1 = (c-2)(c-1)
    let t1_val = (cv - F257::from(2u64)) * (cv - F257::ONE);
    let t1 = b.new_var(t1_val);
    b.add_constraint(
        vec![(F257::ONE, c_var), (-F257::from(2u64), b.one())],
        vec![(F257::ONE, c_var), (-F257::ONE, b.one())],
        vec![(F257::ONE, t1)],
    );

    // t2 = t1 * c
    let t2_val = t1_val * cv;
    let t2 = b.new_var(t2_val);
    b.add_constraint(vec![(F257::ONE, t1)], vec![(F257::ONE, c_var)], vec![(F257::ONE, t2)]);

    // t3 = t2 * (c+1)
    let t3_val = t2_val * (cv + F257::ONE);
    let t3 = b.new_var(t3_val);
    b.add_constraint(
        vec![(F257::ONE, t2)],
        vec![(F257::ONE, c_var), (F257::ONE, b.one())],
        vec![(F257::ONE, t3)],
    );

    // t3 * (c+2) = 0
    b.add_constraint(
        vec![(F257::ONE, t3)],
        vec![(F257::ONE, c_var), (F257::from(2u64), b.one())],
        vec![(F257::ZERO, b.one())],
    );
    c_var
}

#[inline]
pub(crate) fn u64_to_bal16_digits_le_const(mut x: u64) -> [i8; 17] {
    // Match the balancing convention used elsewhere: digits in [-8,7] plus carry in {0,1}.
    let mut out = [0i8; 17];
    let mut carry: i16 = 0;
    for i in 0..16 {
        let nib = (x & 0xF) as i16;
        x >>= 4;
        let v = nib + carry;
        if v >= 8 {
            out[i] = (v - 16) as i8;
            carry = 1;
        } else {
            out[i] = v as i8;
            carry = 0;
        }
    }
    out[16] = carry as i8;
    out
}

#[inline]
pub(crate) fn alloc_u64_as_bal16_digits_witness_ir(b: &mut IrBuilder<'_>, x: u64) -> [VarRef; 17] {
    let ds = u64_to_bal16_digits_le_const(x);
    let mut out: [VarRef; 17] = [VarRef::Base(0); 17];
    for i in 0..16 {
        out[i] = alloc_bal16_digit_ir(b, ds[i]);
    }
    out[16] = alloc_bool_ir(b, ds[16] == 1);
    out
}

#[inline]
fn digits_to_u64_witness_ir(b: &IrBuilder<'_>, d: &[VarRef; 17]) -> u64 {
    let mut acc: i128 = 0;
    let mut pow: i128 = 1;
    for i in 0..17 {
        let di = f257_to_i32_bal(b.val(d[i])) as i128;
        acc += di * pow;
        pow *= 16;
    }
    debug_assert!(acc >= 0);
    acc as u64
}

#[inline]
fn pad17_to_18_with_zero(d: &[VarRef; 17], z: VarRef) -> [VarRef; 18] {
    let mut out = [z; 18];
    for i in 0..17 {
        out[i] = d[i];
    }
    out
}

#[inline]
fn alloc_zero_const_ir(b: &mut IrBuilder<'_>) -> VarRef {
    let z = b.new_var(F257::ZERO);
    b.ir.enforce_var_eq_const(z, F257::ZERO);
    z
}

/// Enforce `a + c = q*p + r` over balanced base-16 digits with carry bound [-2,2].
fn enforce_add_mod_p_relation_bal16_ir(
    b: &mut IrBuilder<'_>,
    a_d: &[VarRef; 17],
    c_d: &[VarRef; 17],
    r_d: &[VarRef; 17],
    q: VarRef,
    q_u8: u8,
    p_d_const: &[i8; 17],
) {
    debug_assert!(q_u8 == 0 || q_u8 == 1);
    // We need 18 digits for carry chain headroom.
    let z_digit = alloc_zero_const_ir(b);
    let a = pad17_to_18_with_zero(a_d, z_digit);
    let c = pad17_to_18_with_zero(c_d, z_digit);
    let r = pad17_to_18_with_zero(r_d, z_digit);

    // carry_0 = 0 (no var)
    let mut carry_var: Option<VarRef> = None;
    let mut carry_i32: i32 = 0;

    for k in 0..18 {
        let ak = f257_to_i32_bal(b.val(a[k]));
        let ck = f257_to_i32_bal(b.val(c[k]));
        let rk = f257_to_i32_bal(b.val(r[k]));
        let pk = if k < 17 { p_d_const[k] as i32 } else { 0 };

        let sum = carry_i32 + ak + ck - rk - (q_u8 as i32) * pk;
        debug_assert!(sum % 16 == 0, "add_mod_p carry not divisible: sum={sum} k={k}");
        let carry_next = sum / 16;
        debug_assert!(
            (-1..=1).contains(&carry_next),
            "add_mod_p carry out of pm1: {carry_next} (sum={sum}) at k={k}"
        );
        let carry_next_var = alloc_carry_pm1_ir(b, carry_next);

        // carry + a + c - r - q*pk - 16*carry_next = 0
        let mut lc: Vec<(F257, VarRef)> = Vec::with_capacity(6);
        if let Some(cv) = carry_var {
            lc.push((F257::ONE, cv));
        }
        lc.push((F257::ONE, a[k]));
        lc.push((F257::ONE, c[k]));
        lc.push((-F257::ONE, r[k]));
        if k < 17 && p_d_const[k] != 0 {
            lc.push((-i32_to_f257(p_d_const[k] as i32), q));
        }
        lc.push((-F257::from(16u64), carry_next_var));
        b.enforce_lc_eq_zero(lc);

        carry_var = Some(carry_next_var);
        carry_i32 = carry_next;
    }
    b.ir.enforce_var_eq_const(carry_var.expect("enforce_add_mod_p_relation_bal16_ir: empty"), F257::ZERO);
}

/// Enforce `a + q*p = c + r` over balanced base-16 digits with carry bound [-2,2].
fn enforce_sub_mod_p_relation_bal16_ir(
    b: &mut IrBuilder<'_>,
    a_d: &[VarRef; 17],
    c_d: &[VarRef; 17],
    r_d: &[VarRef; 17],
    q: VarRef,
    q_u8: u8,
    p_d_const: &[i8; 17],
) {
    debug_assert!(q_u8 == 0 || q_u8 == 1);
    let z_digit = alloc_zero_const_ir(b);
    let a = pad17_to_18_with_zero(a_d, z_digit);
    let c = pad17_to_18_with_zero(c_d, z_digit);
    let r = pad17_to_18_with_zero(r_d, z_digit);

    // carry_0 = 0 (no var)
    let mut carry_var: Option<VarRef> = None;
    let mut carry_i32: i32 = 0;

    for k in 0..18 {
        let ak = f257_to_i32_bal(b.val(a[k]));
        let ck = f257_to_i32_bal(b.val(c[k]));
        let rk = f257_to_i32_bal(b.val(r[k]));
        let pk = if k < 17 { p_d_const[k] as i32 } else { 0 };

        let sum = carry_i32 + ak + (q_u8 as i32) * pk - ck - rk;
        debug_assert!(sum % 16 == 0, "sub_mod_p carry not divisible: sum={sum} k={k}");
        let carry_next = sum / 16;
        debug_assert!(
            (-1..=1).contains(&carry_next),
            "sub_mod_p carry out of pm1: {carry_next} (sum={sum}) at k={k}"
        );
        let carry_next_var = alloc_carry_pm1_ir(b, carry_next);

        // carry + a - c - r + q*pk - 16*carry_next = 0
        let mut lc: Vec<(F257, VarRef)> = Vec::with_capacity(6);
        if let Some(cv) = carry_var {
            lc.push((F257::ONE, cv));
        }
        lc.push((F257::ONE, a[k]));
        lc.push((-F257::ONE, c[k]));
        lc.push((-F257::ONE, r[k]));
        if k < 17 && p_d_const[k] != 0 {
            lc.push((i32_to_f257(p_d_const[k] as i32), q));
        }
        lc.push((-F257::from(16u64), carry_next_var));
        b.enforce_lc_eq_zero(lc);

        carry_var = Some(carry_next_var);
        carry_i32 = carry_next;
    }
    b.ir.enforce_var_eq_const(carry_var.expect("enforce_sub_mod_p_relation_bal16_ir: empty"), F257::ZERO);
}

/// Digit-domain Goldilocks addition (mod p) in IR form.
pub(crate) fn goldilocks_add_mod_p_digits_ir(
    b: &mut IrBuilder<'_>,
    a: &[VarRef; 17],
    c: &[VarRef; 17],
    p_u64: u64,
    p_d_const: &[i8; 17],
) -> [VarRef; 17] {
    let a_u = digits_to_u64_witness_ir(b, a);
    let c_u = digits_to_u64_witness_ir(b, c);
    let sum = (a_u as u128) + (c_u as u128);
    let q_u8: u8 = if sum >= (p_u64 as u128) { 1 } else { 0 };
    let r_u: u64 = if q_u8 == 1 { (sum - (p_u64 as u128)) as u64 } else { sum as u64 };
    let q = if q_u8 == 1 {
        b.one()
    } else {
        let z = b.new_var(F257::ZERO);
        b.ir.enforce_var_eq_const(z, F257::ZERO);
        z
    };
    let r_d = alloc_u64_as_bal16_digits_witness_ir(b, r_u);
    enforce_add_mod_p_relation_bal16_ir(b, a, c, &r_d, q, q_u8, p_d_const);
    r_d
}

/// Digit-domain Goldilocks subtraction (mod p) in IR form.
pub(crate) fn goldilocks_sub_mod_p_digits_ir(
    b: &mut IrBuilder<'_>,
    a: &[VarRef; 17],
    c: &[VarRef; 17],
    p_u64: u64,
    p_d_const: &[i8; 17],
) -> [VarRef; 17] {
    let a_u = digits_to_u64_witness_ir(b, a);
    let c_u = digits_to_u64_witness_ir(b, c);
    let (q_u8, r_u) = if a_u >= c_u {
        (0u8, a_u - c_u)
    } else {
        (1u8, (a_u as u128 + (p_u64 as u128) - (c_u as u128)) as u64)
    };
    let q = if q_u8 == 1 {
        b.one()
    } else {
        let z = b.new_var(F257::ZERO);
        b.ir.enforce_var_eq_const(z, F257::ZERO);
        z
    };
    let r_d = alloc_u64_as_bal16_digits_witness_ir(b, r_u);
    enforce_sub_mod_p_relation_bal16_ir(b, a, c, &r_d, q, q_u8, p_d_const);
    r_d
}

// -----------------------------------------------------------------------------
// Goldilocks arithmetic in balanced base-4 (used to keep NTT in base4 end-to-end)
// -----------------------------------------------------------------------------

#[inline]
fn digits4_to_u64_witness_ir(b: &IrBuilder<'_>, d: &[VarRef; 33]) -> u64 {
    let mut acc: i128 = 0;
    let mut pow: i128 = 1;
    for i in 0..33 {
        let di = f257_to_i32_bal(b.val(d[i])) as i128;
        acc += di * pow;
        pow *= 4;
    }
    debug_assert!(acc >= 0);
    acc as u64
}

#[inline]
fn pad33_to_34_with_zero(d: &[VarRef; 33], z: VarRef) -> [VarRef; 34] {
    let mut out = [z; 34];
    for i in 0..33 {
        out[i] = d[i];
    }
    out
}

fn enforce_add_mod_p_relation_bal4_ir(
    b: &mut IrBuilder<'_>,
    a4: &[VarRef; 33],
    c4: &[VarRef; 33],
    r4: &[VarRef; 33],
    q: VarRef,
    q_u8: u8,
) {
    debug_assert!(q_u8 == 0 || q_u8 == 1);
    let z = alloc_zero_const_ir(b);
    let a = pad33_to_34_with_zero(a4, z);
    let c = pad33_to_34_with_zero(c4, z);
    let r = pad33_to_34_with_zero(r4, z);

    // carry_0 = 0 (no var). In balanced base-4 with digits in [-2,1], the carry in these
    // add/sub mod-p relations stays in {-1,0,1}, so we allocate pm1 carries for k>=0.
    let mut carry_var: Option<VarRef> = None;
    let mut carry_i32: i32 = 0;

    // p = 4^32 - 4^16 + 1  => digit at 0:+1, 16:-1, 32:+1
    for k in 0..34 {
        let ak = f257_to_i32_bal(b.val(a[k]));
        let ck = f257_to_i32_bal(b.val(c[k]));
        let rk = f257_to_i32_bal(b.val(r[k]));
        let pk: i32 = match k {
            0 => 1,
            16 => -1,
            32 => 1,
            _ => 0,
        };

        let sum = carry_i32 + ak + ck - rk - (q_u8 as i32) * pk;
        debug_assert!(sum % 4 == 0, "add_mod_p(bal4) carry not divisible: sum={sum} k={k}");
        let carry_next = sum / 4;
        debug_assert!(
            (-1..=1).contains(&carry_next),
            "add_mod_p(bal4) carry out of pm1: {carry_next} (sum={sum}) at k={k}"
        );
        let carry_next_var = alloc_carry_pm1_ir(b, carry_next);

        // carry + a + c - r - q*pk - 4*carry_next = 0
        let mut lc: Vec<(F257, VarRef)> = Vec::with_capacity(7);
        if let Some(cv) = carry_var {
            lc.push((F257::ONE, cv));
        }
        lc.push((F257::ONE, a[k]));
        lc.push((F257::ONE, c[k]));
        lc.push((-F257::ONE, r[k]));
        if pk != 0 && q_u8 == 1 {
            // q is boolean constant (0 or 1), so just include it when q==1.
            lc.push((-i32_to_f257(pk), q));
        }
        lc.push((-F257::from(4u64), carry_next_var));
        b.enforce_lc_eq_zero(lc);

        carry_var = Some(carry_next_var);
        carry_i32 = carry_next;
    }
    b.ir.enforce_var_eq_const(carry_var.expect("enforce_add_mod_p_relation_bal4_ir: empty"), F257::ZERO);
}

fn enforce_sub_mod_p_relation_bal4_ir(
    b: &mut IrBuilder<'_>,
    a4: &[VarRef; 33],
    c4: &[VarRef; 33],
    r4: &[VarRef; 33],
    q: VarRef,
    q_u8: u8,
) {
    debug_assert!(q_u8 == 0 || q_u8 == 1);
    let z = alloc_zero_const_ir(b);
    let a = pad33_to_34_with_zero(a4, z);
    let c = pad33_to_34_with_zero(c4, z);
    let r = pad33_to_34_with_zero(r4, z);

    // carry_0 = 0 (no var)
    let mut carry_var: Option<VarRef> = None;
    let mut carry_i32: i32 = 0;

    for k in 0..34 {
        let ak = f257_to_i32_bal(b.val(a[k]));
        let ck = f257_to_i32_bal(b.val(c[k]));
        let rk = f257_to_i32_bal(b.val(r[k]));
        let pk: i32 = match k {
            0 => 1,
            16 => -1,
            32 => 1,
            _ => 0,
        };

        let sum = carry_i32 + ak + (q_u8 as i32) * pk - ck - rk;
        debug_assert!(sum % 4 == 0, "sub_mod_p(bal4) carry not divisible: sum={sum} k={k}");
        let carry_next = sum / 4;
        debug_assert!(
            (-1..=1).contains(&carry_next),
            "sub_mod_p(bal4) carry out of pm1: {carry_next} (sum={sum}) at k={k}"
        );
        let carry_next_var = alloc_carry_pm1_ir(b, carry_next);

        // carry + a - c - r + q*pk - 4*carry_next = 0
        let mut lc: Vec<(F257, VarRef)> = Vec::with_capacity(7);
        if let Some(cv) = carry_var {
            lc.push((F257::ONE, cv));
        }
        lc.push((F257::ONE, a[k]));
        lc.push((-F257::ONE, c[k]));
        lc.push((-F257::ONE, r[k]));
        if pk != 0 && q_u8 == 1 {
            lc.push((i32_to_f257(pk), q));
        }
        lc.push((-F257::from(4u64), carry_next_var));
        b.enforce_lc_eq_zero(lc);

        carry_var = Some(carry_next_var);
        carry_i32 = carry_next;
    }
    b.ir.enforce_var_eq_const(carry_var.expect("enforce_sub_mod_p_relation_bal4_ir: empty"), F257::ZERO);
}

#[inline]
pub(crate) fn goldilocks_add_mod_p_digits_bal4_ir(b: &mut IrBuilder<'_>, a4: &[VarRef; 33], c4: &[VarRef; 33], p_u64: u64) -> [VarRef; 33] {
    let a_u = digits4_to_u64_witness_ir(b, a4);
    let c_u = digits4_to_u64_witness_ir(b, c4);
    let sum = (a_u as u128) + (c_u as u128);
    let q_u8: u8 = if sum >= (p_u64 as u128) { 1 } else { 0 };
    let r_u: u64 = if q_u8 == 1 { (sum - (p_u64 as u128)) as u64 } else { sum as u64 };
    let q = if q_u8 == 1 {
        b.one()
    } else {
        let z = b.new_var(F257::ZERO);
        b.ir.enforce_var_eq_const(z, F257::ZERO);
        z
    };
    // IMPORTANT (soundness): `r4[k]` are the "digit variables" of the base-4 carry chain.
    // Without explicit membership checks `r4[k] ∈ {-2,-1,0,1}`, the carry-chain equation admits
    // alternative integer solutions (e.g. shifting `r4[k]` by ±4 and compensating carry), and
    // those digits later flow into multiplication gadgets in the NTT pipeline.
    let r4 = alloc_u64_as_bal4_digits_checked_ir(b, r_u);
    enforce_add_mod_p_relation_bal4_ir(b, a4, c4, &r4, q, q_u8);
    r4
}

#[inline]
pub(crate) fn goldilocks_sub_mod_p_digits_bal4_ir(b: &mut IrBuilder<'_>, a4: &[VarRef; 33], c4: &[VarRef; 33], p_u64: u64) -> [VarRef; 33] {
    let a_u = digits4_to_u64_witness_ir(b, a4);
    let c_u = digits4_to_u64_witness_ir(b, c4);
    let (q_u8, r_u) = if a_u >= c_u {
        (0u8, a_u - c_u)
    } else {
        (1u8, (a_u as u128 + (p_u64 as u128) - (c_u as u128)) as u64)
    };
    let q = if q_u8 == 1 {
        b.one()
    } else {
        let z = b.new_var(F257::ZERO);
        b.ir.enforce_var_eq_const(z, F257::ZERO);
        z
    };
    let r4 = alloc_u64_as_bal4_digits_checked_ir(b, r_u);
    enforce_sub_mod_p_relation_bal4_ir(b, a4, c4, &r4, q, q_u8);
    r4
}

#[inline]
pub(crate) fn goldilocks_mul_const_mod_p_digits_bal4_ir(b: &mut IrBuilder<'_>, x4: &[VarRef; 33], k: u64, p_u64: u64) -> [VarRef; 33] {
    let x_u = digits4_to_u64_witness_ir(b, x4);
    let prod: u128 = (x_u as u128) * (k as u128);
    let q_u: u64 = (prod / (p_u64 as u128)) as u64;
    let r_u: u64 = (prod % (p_u64 as u128)) as u64;
    // Soundness: constrain quotient digits too (they participate in the carry chain).
    let q4 = alloc_u64_as_bal4_digits_checked_ir(b, q_u);
    // `r4` are digit variables; enforce membership to keep the carry chain injective.
    let r4 = alloc_u64_as_bal4_digits_checked_ir(b, r_u);
    let pre = b.bal4_const_mul_precomp(k);
    enforce_prod_const_eq_qp_plus_r_bal4_ir_with_carry_schedule(b, x4, &pre.k4_const, &q4, &r4, &pre.carry_bounds);
    r4
}

#[inline]
pub(crate) fn goldilocks_mul_mod_p_digits_bal4_ir(b: &mut IrBuilder<'_>, a4: &[VarRef; 33], c4: &[VarRef; 33], p_u64: u64) -> [VarRef; 33] {
    let a_u = digits4_to_u64_witness_ir(b, a4);
    let c_u = digits4_to_u64_witness_ir(b, c4);
    let prod_u: u128 = (a_u as u128) * (c_u as u128);
    let q_u: u64 = (prod_u / (p_u64 as u128)) as u64;
    let r_u: u64 = (prod_u % (p_u64 as u128)) as u64;
    let q4 = alloc_u64_as_bal4_digits_checked_ir(b, q_u);
    let r4 = alloc_u64_as_bal4_digits_checked_ir(b, r_u);
    enforce_prod_var_eq_qp_plus_r_bal4_ir(b, a4, c4, &q4, &r4);
    r4
}

/// Digit-domain Goldilocks multiplication (mod p) in IR form.
pub(crate) fn goldilocks_mul_mod_p_digits_ir(
    b: &mut IrBuilder<'_>,
    a: &[VarRef; 17],
    c: &[VarRef; 17],
    p_u64: u64,
    p_d_const: &[i8; 17],
) -> [VarRef; 17] {
    let _ = p_d_const;
    let a_u = digits_to_u64_witness_ir(b, a);
    let c_u = digits_to_u64_witness_ir(b, c);
    let prod_u: u128 = (a_u as u128) * (c_u as u128);
    let q_u: u64 = (prod_u / (p_u64 as u128)) as u64;
    let r_u: u64 = (prod_u % (p_u64 as u128)) as u64;

    // Enforce multiplication via a balanced base-4 carry chain (pm32 window),
    let a4 = b.bal16_to_bal4_digits_cached(a);
    let c4 = b.bal16_to_bal4_digits_cached(c);
    let q4 = alloc_u64_as_bal4_digits_checked_ir(b, q_u);
    let r4 = alloc_u64_as_bal4_digits_checked_ir(b, r_u);
    enforce_prod_var_eq_qp_plus_r_bal4_ir(b, &a4, &c4, &q4, &r4);
    bal4_to_bal16_digits_ir(b, &r4)
}

/// Evaluate a ring element (length 64) at a scalar point `x` (Goldilocks digits),
/// emitting IR for the full evaluation.
///
/// This matches `setchk::ev` / `cm_math::ring_eval_at_scalar_digits`:
/// \(\sum_{t=0}^{63} coeff[t] \cdot x^t\) mod p.
pub(crate) fn ring_eval_at_scalar_digits_d64_ir(
    b: &mut IrBuilder<'_>,
    coeffs: &[[VarRef; 17]; 64],
    x: &[VarRef; 17],
    p_u64: u64,
    p_d_const: &[i8; 17],
) -> [VarRef; 17] {
    let zero = alloc_u64_as_bal16_digits_witness_ir(b, 0u64);
    let one = alloc_u64_as_bal16_digits_witness_ir(b, 1u64);
    let mut acc = zero;
    let mut pow = one;
    for c in coeffs {
        let t = goldilocks_mul_mod_p_digits_ir(b, c, &pow, p_u64, p_d_const);
        acc = goldilocks_add_mod_p_digits_ir(b, &acc, &t, p_u64, p_d_const);
        pow = goldilocks_mul_mod_p_digits_ir(b, &pow, x, p_u64, p_d_const);
    }
    acc
}

/// Negacyclic ring multiplication for `d=64` over Goldilocks using an NTT-based method, emitting IR.
///
/// Mirrors `goldilocks::ring_mul_negacyclic_ntt_goldilocks_d64`, but is `Dr1csBuilder`-free:
/// it only allocates local vars + constraints in `CmIr`.
#[inline]
fn ring_mul_negacyclic_ntt_goldilocks_d64_bal4_core_ir(
    b: &mut IrBuilder<'_>,
    a4_in: &[[VarRef; 33]; 64],
    c4_in: &[[VarRef; 33]; 64],
) -> [[VarRef; 33]; 64] {
    // Shared schedule constants from `cyclotomic-rings` (keeps host + gate in sync).
    let p: u64 = gl_ntt64::GOLDILOCKS_P_U64;
    let omega: u64 = gl_ntt64::OMEGA_U64;
    let omega_inv: u64 = gl_ntt64::OMEGA_INV_U64;
    let inv_n: u64 = gl_ntt64::INV_N_U64;

    let zero_digit = alloc_zero_const_ir(b);
    let zero_scalar4: [VarRef; 33] = [zero_digit; 33];

    fn ntt_in_place_bal4(
        b: &mut IrBuilder<'_>,
        a: &mut [[VarRef; 33]; 64],
        omega: u64,
        p_u64: u64,
        zero: &[VarRef; 33],
    ) {
        // Bit-reversal permutation (purely structural).
        let mut tmp = *a;
        for i in 0..64 {
            tmp[gl_ntt64::BITREV_64[i]] = a[i];
        }
        *a = tmp;

        // Iterative Cooley–Tukey.
        let mut len = 2usize;
        while len <= 64 {
            let half = len / 2;
            for start in (0..64).step_by(len) {
                for j in 0..half {
                    let w: u64 = if omega == gl_ntt64::OMEGA_U64 {
                        match len {
                            2 => gl_ntt64::W_POWS_LEN_2[j],
                            4 => gl_ntt64::W_POWS_LEN_4[j],
                            8 => gl_ntt64::W_POWS_LEN_8[j],
                            16 => gl_ntt64::W_POWS_LEN_16[j],
                            32 => gl_ntt64::W_POWS_LEN_32[j],
                            64 => gl_ntt64::W_POWS_LEN_64[j],
                            _ => unreachable!(),
                        }
                    } else {
                        debug_assert_eq!(omega, gl_ntt64::OMEGA_INV_U64);
                        match len {
                            2 => gl_ntt64::IW_POWS_LEN_2[j],
                            4 => gl_ntt64::IW_POWS_LEN_4[j],
                            8 => gl_ntt64::IW_POWS_LEN_8[j],
                            16 => gl_ntt64::IW_POWS_LEN_16[j],
                            32 => gl_ntt64::IW_POWS_LEN_32[j],
                            64 => gl_ntt64::IW_POWS_LEN_64[j],
                            _ => unreachable!(),
                        }
                    };

                    let u = a[start + j];
                    let v = if w == 1 {
                        a[start + j + half]
                    } else if w == p_u64 - 1 {
                        // v = -x mod p = 0 - x
                        goldilocks_sub_mod_p_digits_bal4_ir(b, zero, &a[start + j + half], p_u64)
                    } else {
                        goldilocks_mul_const_mod_p_digits_bal4_ir(b, &a[start + j + half], w, p_u64)
                    };
                    a[start + j] = goldilocks_add_mod_p_digits_bal4_ir(b, &u, &v, p_u64);
                    a[start + j + half] = goldilocks_sub_mod_p_digits_bal4_ir(b, &u, &v, p_u64);
                }
            }
            len *= 2;
        }
    }

    fn intt_in_place_bal4(
        b: &mut IrBuilder<'_>,
        a: &mut [[VarRef; 33]; 64],
        omega_inv: u64,
        inv_n: u64,
        p_u64: u64,
        zero: &[VarRef; 33],
    ) {
        ntt_in_place_bal4(b, a, omega_inv, p_u64, zero);
        // scale by n^{-1}
        for i in 0..64 {
            a[i] = if inv_n == 1 {
                a[i]
            } else if inv_n == p_u64 - 1 {
                goldilocks_sub_mod_p_digits_bal4_ir(b, zero, &a[i], p_u64)
            } else {
                goldilocks_mul_const_mod_p_digits_bal4_ir(b, &a[i], inv_n, p_u64)
            };
        }
    }

    // Negacyclic via twist by ψ^i (ψ is primitive 128th root).
    let mut a_tw = [[zero_digit; 33]; 64];
    let mut c_tw = [[zero_digit; 33]; 64];
    for i in 0..64 {
        let psi_pow: u64 = gl_ntt64::PSI_POWS_64[i];
        if psi_pow == 1 {
            a_tw[i] = a4_in[i];
            c_tw[i] = c4_in[i];
        } else if psi_pow == p - 1 {
            a_tw[i] = goldilocks_sub_mod_p_digits_bal4_ir(b, &zero_scalar4, &a4_in[i], p);
            c_tw[i] = goldilocks_sub_mod_p_digits_bal4_ir(b, &zero_scalar4, &c4_in[i], p);
        } else {
            a_tw[i] = goldilocks_mul_const_mod_p_digits_bal4_ir(b, &a4_in[i], psi_pow, p);
            c_tw[i] = goldilocks_mul_const_mod_p_digits_bal4_ir(b, &c4_in[i], psi_pow, p);
        }
    }

    ntt_in_place_bal4(b, &mut a_tw, omega, p, &zero_scalar4);
    ntt_in_place_bal4(b, &mut c_tw, omega, p, &zero_scalar4);

    // Pointwise multiply.
    for i in 0..64 {
        a_tw[i] = goldilocks_mul_mod_p_digits_bal4_ir(b, &a_tw[i], &c_tw[i], p);
    }

    intt_in_place_bal4(b, &mut a_tw, omega_inv, inv_n, p, &zero_scalar4);

    // Untwist by ψ^{-i}.
    let mut out4 = [[zero_digit; 33]; 64];
    for i in 0..64 {
        let psi_inv_pow: u64 = gl_ntt64::PSI_INV_POWS_64[i];
        out4[i] = if psi_inv_pow == 1 {
            a_tw[i]
        } else if psi_inv_pow == p - 1 {
            goldilocks_sub_mod_p_digits_bal4_ir(b, &zero_scalar4, &a_tw[i], p)
        } else {
            goldilocks_mul_const_mod_p_digits_bal4_ir(b, &a_tw[i], psi_inv_pow, p)
        };
    }

    out4
}

pub(crate) fn ring_mul_negacyclic_ntt_goldilocks_d64_ir(
    b: &mut IrBuilder<'_>,
    a: &[[VarRef; 17]; 64],
    c: &[[VarRef; 17]; 64],
) -> [[VarRef; 17]; 64] {
    let zero_digit = alloc_zero_const_ir(b);

    // Convert inputs once: bal16 -> bal4.
    let mut a4 = [[zero_digit; 33]; 64];
    let mut c4 = [[zero_digit; 33]; 64];
    for i in 0..64 {
        a4[i] = b.bal16_to_bal4_digits_cached(&a[i]);
        c4[i] = b.bal16_to_bal4_digits_cached(&c[i]);
    }
    let out4 = ring_mul_negacyclic_ntt_goldilocks_d64_bal4_core_ir(b, &a4, &c4);

    // Convert outputs once: bal4 -> bal16.
    let mut out = [[zero_digit; 17]; 64];
    for i in 0..64 {
        out[i] = bal4_to_bal16_digits_ir(b, &out4[i]);
    }
    out
}

/// Negacyclic ring multiplication for `d=64` over Goldilocks using an NTT-based method, emitting IR,
/// with inputs/outputs in **checked bal4** digits.
///
/// This enables bal4 end-to-end pipelines that want to avoid repeated bal16<->bal4 conversions
/// around subsequent scaling/addition.
pub(crate) fn ring_mul_negacyclic_ntt_goldilocks_d64_bal4_ir(
    b: &mut IrBuilder<'_>,
    a4_in: &[[VarRef; 33]; 64],
    c4_in: &[[VarRef; 33]; 64],
) -> [[VarRef; 33]; 64] {
    ring_mul_negacyclic_ntt_goldilocks_d64_bal4_core_ir(b, a4_in, c4_in)
}

#[cfg(test)]
mod soundness_regression_tests {
    use super::*;
    use symphony::dpp_sumcheck::Dr1csBuilder;

    #[test]
    fn test_bal4_to_bal16_roundtrip_preserves_u64() {
        use rand::{RngCore, SeedableRng};
        let p: u64 = gl_ntt64::GOLDILOCKS_P_U64;

        let base_asg = [F257::ONE];
        let mut ib = IrBuilder::new(&base_asg);

        let mut rng = rand::rngs::StdRng::seed_from_u64(0x5a77_1eaf_1b01_91d3);
        for _ in 0..10_000 {
            let x: u64 = rng.next_u64() % p;
            let x16 = alloc_u64_as_bal16_digits_witness_ir(&mut ib, x);
            let x4 = ib.bal16_to_bal4_digits_cached(&x16);
            let y16 = bal4_to_bal16_digits_ir(&mut ib, &x4);

            let y: u64 = digits_to_u64_witness_ir(&ib, &y16);
            assert_eq!(y, x, "bal16->bal4->bal16 drift for x={x}");
        }
    }

    #[test]
    fn test_bal4_add_mul_mod_p_agree_with_bal16_and_u64() {
        use rand::{RngCore, SeedableRng};
        let p: u64 = gl_ntt64::GOLDILOCKS_P_U64;
        let p_d_const: [i8; 17] = u64_to_bal16_digits_le_const(p);

        let base_asg = [F257::ONE];
        let mut ib = IrBuilder::new(&base_asg);

        let mut rng = rand::rngs::StdRng::seed_from_u64(0x0d2e_9b7c_4c8a_aa51);
        for _ in 0..2_000 {
            let a: u64 = rng.next_u64() % p;
            let b: u64 = rng.next_u64() % p;

            let a16 = alloc_u64_as_bal16_digits_witness_ir(&mut ib, a);
            let b16 = alloc_u64_as_bal16_digits_witness_ir(&mut ib, b);

            // bal16 reference
            let sum16 = goldilocks_add_mod_p_digits_ir(&mut ib, &a16, &b16, p, &p_d_const);
            let prod16 = goldilocks_mul_mod_p_digits_ir(&mut ib, &a16, &b16, p, &p_d_const);
            let sum16_u = digits_to_u64_witness_ir(&ib, &sum16);
            let prod16_u = digits_to_u64_witness_ir(&ib, &prod16);

            // bal4 path + convert
            let a4 = ib.bal16_to_bal4_digits_cached(&a16);
            let b4 = ib.bal16_to_bal4_digits_cached(&b16);
            let sum4 = goldilocks_add_mod_p_digits_bal4_ir(&mut ib, &a4, &b4, p);
            let prod4 = goldilocks_mul_mod_p_digits_bal4_ir(&mut ib, &a4, &b4, p);
            let sum4_u = digits4_to_u64_witness_ir(&ib, &sum4);
            let prod4_u = digits4_to_u64_witness_ir(&ib, &prod4);
            let sum4_to_16 = bal4_to_bal16_digits_ir(&mut ib, &sum4);
            let prod4_to_16 = bal4_to_bal16_digits_ir(&mut ib, &prod4);
            let sum4_to_16_u = digits_to_u64_witness_ir(&ib, &sum4_to_16);
            let prod4_to_16_u = digits_to_u64_witness_ir(&ib, &prod4_to_16);

            // host expected
            let exp_sum = ((a as u128 + b as u128) % (p as u128)) as u64;
            let exp_prod = ((a as u128 * b as u128) % (p as u128)) as u64;

            assert_eq!(sum16_u, exp_sum, "bal16 add mismatch: a={a} b={b}");
            assert_eq!(prod16_u, exp_prod, "bal16 mul mismatch: a={a} b={b}");

            assert_eq!(sum4_u, exp_sum, "bal4 add mismatch: a={a} b={b}");
            assert_eq!(prod4_u, exp_prod, "bal4 mul mismatch: a={a} b={b}");

            assert_eq!(
                sum4_to_16_u, sum16_u,
                "bal4->bal16 add drift vs bal16: a={a} b={b}"
            );
            assert_eq!(
                prod4_to_16_u, prod16_u,
                "bal4->bal16 mul drift vs bal16: a={a} b={b}"
            );
        }
    }

    #[test]
    fn test_bal4_add_mul_mod_p_constraints_agree_with_bal16() {
        use rand::{RngCore, SeedableRng};
        let p: u64 = gl_ntt64::GOLDILOCKS_P_U64;
        let p_d_const: [i8; 17] = u64_to_bal16_digits_le_const(p);

        let mut rng = rand::rngs::StdRng::seed_from_u64(0x9f66_d6c1_0f26_4b2a);
        for _case in 0..200 {
            let a: u64 = rng.next_u64() % p;
            let b_u: u64 = rng.next_u64() % p;

            let mut b = Dr1csBuilder::<F257>::new();
            b.enforce_var_eq_const(b.one(), F257::ONE);
            let base_asg: Vec<F257> = b.assignment.clone();
            let mut ib = IrBuilder::new(&base_asg);

            let a16 = alloc_u64_as_bal16_digits_witness_ir(&mut ib, a);
            let b16 = alloc_u64_as_bal16_digits_witness_ir(&mut ib, b_u);

            // bal16 reference
            let sum16 = goldilocks_add_mod_p_digits_ir(&mut ib, &a16, &b16, p, &p_d_const);
            let prod16 = goldilocks_mul_mod_p_digits_ir(&mut ib, &a16, &b16, p, &p_d_const);

            // bal4 path + convert
            let a4 = ib.bal16_to_bal4_digits_cached(&a16);
            let b4 = ib.bal16_to_bal4_digits_cached(&b16);
            let sum4 = goldilocks_add_mod_p_digits_bal4_ir(&mut ib, &a4, &b4, p);
            let prod4 = goldilocks_mul_mod_p_digits_bal4_ir(&mut ib, &a4, &b4, p);
            let sum4_16 = bal4_to_bal16_digits_ir(&mut ib, &sum4);
            let prod4_16 = bal4_to_bal16_digits_ir(&mut ib, &prod4);

            // Enforce digitwise equality between the two constructions.
            for j in 0..17 {
                ib.enforce_lc_eq_zero(vec![
                    (F257::ONE, sum16[j]),
                    (-F257::ONE, sum4_16[j]),
                ]);
                ib.enforce_lc_eq_zero(vec![
                    (F257::ONE, prod16[j]),
                    (-F257::ONE, prod4_16[j]),
                ]);
            }

            let _lowered = lower_ir_into_builder(&mut b, ib.ir);
            let (inst, asg) = b.into_instance();
            inst.check(&asg)
                .unwrap_or_else(|e| panic!("constraints mismatch for a={a} b={b_u}: {e}"));
        }
    }

    #[test]
    fn test_good_bal4_carry_bounds_reject_bubble_mutation() {
        let p: u64 = gl_ntt64::GOLDILOCKS_P_U64;
        let k: u64 = gl_ntt64::W_POWS_LEN_64[3];
        let x: u64 = 123456789u64 % p;

        let mut b = Dr1csBuilder::<F257>::new();
        b.enforce_var_eq_const(b.one(), F257::ONE);

        let (carry_idx, r0_idx) = {
            let base_asg = [F257::ONE];
            let mut ib = IrBuilder::new(&base_asg);

            let x4 = alloc_u64_as_bal4_digits_checked_ir(&mut ib, x);
            let prod: u128 = (x as u128) * (k as u128);
            let q_u: u64 = (prod / (p as u128)) as u64;
            let r_u: u64 = (prod % (p as u128)) as u64;
            let q4 = alloc_u64_as_bal4_digits_checked_ir(&mut ib, q_u);
            let r4 = alloc_u64_as_bal4_digits_checked_ir(&mut ib, r_u);
            let k4 = u64_to_bal4_digits_le_const(k);

            // Use a simple bounded-carry chain (pm32 everywhere is enough to reject the bubble).
            let mut carry_var = alloc_carry_pm32_ir(&mut ib, 0);
            let mut carry_i32: i32 = 0;
            let mut captured_carry_next: Option<VarRef> = None;

            for kk in 0..66 {
                let mut sum: i32 = carry_i32;
                let mut lc: Vec<(F257, VarRef)> = Vec::with_capacity(256);
                lc.push((F257::ONE, carry_var));

                // + Σ_i x_i * k_{kk-i}
                for i in 0..33 {
                    let j = kk as i32 - i as i32;
                    if !(0..33).contains(&j) {
                        continue;
                    }
                    let dj = k4[j as usize] as i32;
                    if dj == 0 {
                        continue;
                    }
                    let xi = f257_to_i32_bal(ib.val(x4[i]));
                    sum += xi * dj;
                    lc.push((i32_to_f257(dj), x4[i]));
                }

                // - r_k (only for kk < 33)
                if kk < 33 {
                    let rk = f257_to_i32_bal(ib.val(r4[kk]));
                    sum -= rk;
                    lc.push((-F257::ONE, r4[kk]));
                }

                // - q*p where p = 4^32 - 4^16 + 1  => q*p = q - q<<16 + q<<32
                if kk < 33 {
                    let qk = f257_to_i32_bal(ib.val(q4[kk]));
                    sum -= qk;
                    lc.push((-F257::ONE, q4[kk]));
                }
                if kk >= 16 && (kk - 16) < 33 {
                    let qk = f257_to_i32_bal(ib.val(q4[kk - 16]));
                    sum += qk;
                    lc.push((F257::ONE, q4[kk - 16]));
                }
                if kk >= 32 && (kk - 32) < 33 {
                    let qk = f257_to_i32_bal(ib.val(q4[kk - 32]));
                    sum -= qk;
                    lc.push((-F257::ONE, q4[kk - 32]));
                }

                debug_assert!(sum % 4 == 0);
                let carry_next: i32 = sum / 4;
                debug_assert!((-32..=31).contains(&carry_next));
                let carry_next_var = alloc_carry_pm32_ir(&mut ib, carry_next);
                if kk == 0 {
                    captured_carry_next = Some(carry_next_var);
                }
                lc.push((-F257::from(4u64), carry_next_var));
                ib.enforce_lc_eq_zero(lc);
                carry_var = carry_next_var;
                carry_i32 = carry_next;
            }
            ib.ir.enforce_var_eq_const(carry_var, F257::ZERO);

            let lowered = lower_ir_into_builder(&mut b, ib.ir);
            let carry_idx = lowered.map_var(captured_carry_next.expect("carry_next@k=0"));
            let r0_idx = lowered.map_var(r4[0]);
            (carry_idx, r0_idx)
        };

        let (inst, mut asg) = b.into_instance();
        inst.check(&asg).expect("baseline satisfied");

        // Try the bubble mutation for base-4:
        // carry_next += 64, r0 += 1 keeps the LC unchanged mod 257 because (-4)*64 = -256 ≡ +1.
        // This must FAIL due to carry range check (pm32 forbids shifting by 64).
        asg[carry_idx] += F257::from(64u64);
        asg[r0_idx] += F257::ONE;

        assert!(
            inst.check(&asg).is_err(),
            "bounded base-4 carry gadget should reject bubble mutation"
        );
    }

    /// "Anti-cheat" fuzz test: build a tiny (size-16) Goldilocks NTT fragment in the bal4 domain,
    /// with fixed (constant) inputs and fixed expected outputs, then try to find alternate
    /// satisfying witnesses by perturbing internal variables.
    ///
    /// This is designed to catch "looks fine, passes tests, but underconstrained" failures.
    #[test]
    fn test_anticheat_fuzzer_tiny_ntt16_rejects_internal_mutations() {
        use rand::{RngCore, SeedableRng};
        use rand::rngs::OsRng;
        use std::collections::BTreeSet;

        let p: u64 = gl_ntt64::GOLDILOCKS_P_U64;

        // Randomize seeds on every run (as requested). We print them for reproducibility.
        let seed_inputs: u64 = OsRng.next_u64();
        let seed_fuzz: u64 = OsRng.next_u64();
        eprintln!(
            "NTT16 anticheat seeds: inputs=0x{seed_inputs:016x} fuzz=0x{seed_fuzz:016x}"
        );

        // Inputs.
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed_inputs);
        let mut a_u: [u64; 16] = [0u64; 16];
        for i in 0..16 {
            a_u[i] = rng.next_u64() % p;
        }

        // Host-side expected output for the 16-point NTT using the same stage twiddle tables.
        fn bitrev16(i: usize) -> usize {
            let mut x = i;
            let mut r = 0usize;
            for _ in 0..4 {
                r = (r << 1) | (x & 1);
                x >>= 1;
            }
            r
        }
        #[inline]
        fn add_mod_p(p: u64, a: u64, b: u64) -> u64 {
            (((a as u128) + (b as u128)) % (p as u128)) as u64
        }
        #[inline]
        fn sub_mod_p(p: u64, a: u64, b: u64) -> u64 {
            ((a as i128 - b as i128).rem_euclid(p as i128)) as u64
        }
        #[inline]
        fn mul_const_mod_p(p: u64, a: u64, k: u64) -> u64 {
            (((a as u128) * (k as u128)) % (p as u128)) as u64
        }
        fn ntt16_in_place_u64(a: &mut [u64; 16], p: u64) {
            // Bit-reversal permutation.
            let mut tmp = *a;
            for i in 0..16 {
                tmp[bitrev16(i)] = a[i];
            }
            *a = tmp;

            // Cooley–Tukey using the shared schedule tables.
            let mut len = 2usize;
            while len <= 16 {
                let half = len / 2;
                for start in (0..16).step_by(len) {
                    for j in 0..half {
                        let w: u64 = match len {
                            2 => gl_ntt64::W_POWS_LEN_2[j],
                            4 => gl_ntt64::W_POWS_LEN_4[j],
                            8 => gl_ntt64::W_POWS_LEN_8[j],
                            16 => gl_ntt64::W_POWS_LEN_16[j],
                            _ => unreachable!(),
                        };
                        let u = a[start + j];
                        let v0 = a[start + j + half];
                        let v = if w == 1 { v0 } else { mul_const_mod_p(p, v0, w) };
                        a[start + j] = add_mod_p(p, u, v);
                        a[start + j + half] = sub_mod_p(p, u, v);
                    }
                }
                len *= 2;
            }
        }

        let mut exp_u = a_u;
        ntt16_in_place_u64(&mut exp_u, p);

        // Circuit: build the same 16-point NTT in bal4 digits via IR, with IO fixed by constants.
        let mut b = Dr1csBuilder::<F257>::new();
        b.enforce_var_eq_const(b.one(), F257::ONE);
        let base_asg: Vec<F257> = b.assignment.clone();
        let mut ib = IrBuilder::new(&base_asg);

        let zero_digit = alloc_zero_const_ir(&mut ib);
        let zero_scalar4: [VarRef; 33] = [zero_digit; 33];

        // Allocate input scalars as checked bal4 digits, and FIX them to constants.
        let mut a4: [[VarRef; 33]; 16] = [[zero_digit; 33]; 16];
        let mut a4_in: [[VarRef; 33]; 16] = [[zero_digit; 33]; 16];
        for i in 0..16 {
            let d = u64_to_bal4_digits_le_const(a_u[i]);
            a4[i] = alloc_u64_as_bal4_digits_checked_ir(&mut ib, a_u[i]);
            for k in 0..33 {
                ib.ir.enforce_var_eq_const(a4[i][k], i32_to_f257(d[k] as i32));
            }
            a4_in[i] = a4[i];
        }

        // NTT in-place for size 16 in bal4 digits.
        fn ntt16_in_place_bal4_ir(
            b: &mut IrBuilder<'_>,
            a: &mut [[VarRef; 33]; 16],
            p_u64: u64,
            zero: &[VarRef; 33],
        ) {
            // Bit-reversal permutation (purely structural).
            let mut tmp = *a;
            for i in 0..16 {
                let mut x = i;
                let mut r = 0usize;
                for _ in 0..4 {
                    r = (r << 1) | (x & 1);
                    x >>= 1;
                }
                tmp[r] = a[i];
            }
            *a = tmp;

            let mut len = 2usize;
            while len <= 16 {
                let half = len / 2;
                for start in (0..16).step_by(len) {
                    for j in 0..half {
                        let w: u64 = match len {
                            2 => gl_ntt64::W_POWS_LEN_2[j],
                            4 => gl_ntt64::W_POWS_LEN_4[j],
                            8 => gl_ntt64::W_POWS_LEN_8[j],
                            16 => gl_ntt64::W_POWS_LEN_16[j],
                            _ => unreachable!(),
                        };
                        let u = a[start + j];
                        let v = if w == 1 {
                            a[start + j + half]
                        } else if w == p_u64 - 1 {
                            goldilocks_sub_mod_p_digits_bal4_ir(b, zero, &a[start + j + half], p_u64)
                        } else {
                            goldilocks_mul_const_mod_p_digits_bal4_ir(b, &a[start + j + half], w, p_u64)
                        };
                        a[start + j] = goldilocks_add_mod_p_digits_bal4_ir(b, &u, &v, p_u64);
                        a[start + j + half] = goldilocks_sub_mod_p_digits_bal4_ir(b, &u, &v, p_u64);
                    }
                }
                len *= 2;
            }
        }
        ntt16_in_place_bal4_ir(&mut ib, &mut a4, p, &zero_scalar4);

        // Allocate expected output digits and FIX them to constants.
        let mut exp4: [[VarRef; 33]; 16] = [[zero_digit; 33]; 16];
        for i in 0..16 {
            let d = u64_to_bal4_digits_le_const(exp_u[i]);
            exp4[i] = alloc_u64_as_bal4_digits_checked_ir(&mut ib, exp_u[i]);
            for k in 0..33 {
                ib.ir.enforce_var_eq_const(exp4[i][k], i32_to_f257(d[k] as i32));
            }
        }

        // Constrain computed output == expected output (digitwise).
        for i in 0..16 {
            for k in 0..33 {
                ib.enforce_lc_eq_zero(vec![(F257::ONE, a4[i][k]), (-F257::ONE, exp4[i][k])]);
            }
        }

        // Lower IR and check the original witness satisfies.
        let lowered = lower_ir_into_builder(&mut b, ib.ir);
        let (inst, asg) = b.into_instance();
        inst.check(&asg).expect("baseline NTT16 witness should satisfy");

        // Identify indices we consider "public IO": inputs + outputs + expected outputs.
        let mut io: BTreeSet<usize> = BTreeSet::new();
        io.insert(0); // builder's ONE
        for i in 0..16 {
            for k in 0..33 {
                io.insert(lowered.map_var(a4_in[i][k]));
                io.insert(lowered.map_var(a4[i][k]));
                io.insert(lowered.map_var(exp4[i][k]));
            }
        }

        // Heuristic classification: detect boolean vars by the exact constraint shape emitted by
        // `alloc_bool_ir`: x * (1 - x) = 0.
        let mut is_bool: Vec<bool> = vec![false; inst.nvars];
        for row in inst.constraints.iter() {
            let a = &inst.a_terms[row.a.clone()];
            let b = &inst.b_terms[row.b.clone()];
            let c = &inst.c_terms[row.c.clone()];
            if a.len() == 1
                && a[0].0 == F257::ONE
                && b.len() == 2
                && b[0].0 == F257::ONE
                && b[0].1 == 0
                && b[1].0 == -F257::ONE
                && b[1].1 == a[0].1
                && c.len() == 1
                && c[0].1 == 0
                && c[0].0 == F257::ZERO
            {
                let idx = a[0].1;
                if idx < is_bool.len() {
                    is_bool[idx] = true;
                }
            }
        }

        #[inline]
        fn can_mutate(io: &BTreeSet<usize>, is_bool: &[bool], idx: usize) -> bool {
            idx != 0 && !io.contains(&idx) && !is_bool.get(idx).copied().unwrap_or(false)
        }

        // "Linear repair" attempt:
        // After a random perturbation, try to satisfy any linear constraints that have exactly ONE
        // mutable variable by solving for that variable (Gauss-Seidel style).
        //
        // If the system is underconstrained, this kind of cheap repair can sometimes find an
        // alternate satisfying witness.
        fn linear_repair(
            inst: &symphony::dpp_poseidon::SparseDr1csInstance<F257>,
            io: &BTreeSet<usize>,
            is_bool: &[bool],
            pinned: &BTreeSet<usize>,
            asg: &mut [F257],
            max_passes: usize,
        ) {
            for _pass in 0..max_passes {
                let mut changed = false;
                for row in inst.constraints.iter() {
                    // Recognize linear constraints of the form: (A(z)) * 1 = c,
                    // i.e. B is exactly (1, one) and C is a constant term on (1, one).
                    let b = &inst.b_terms[row.b.clone()];
                    if b.len() != 1 || b[0].0 != F257::ONE || b[0].1 != 0 {
                        continue;
                    }
                    let c = &inst.c_terms[row.c.clone()];
                    if c.len() != 1 || c[0].1 != 0 {
                        continue;
                    }
                    let target = c[0].0;
                    let a = &inst.a_terms[row.a.clone()];

                    let mut unknown: Option<(usize, F257)> = None;
                    let mut sum_known: F257 = F257::ZERO;
                    for (coeff, idx) in a.iter().copied() {
                        if can_mutate(io, is_bool, idx) && !pinned.contains(&idx) {
                            if unknown.is_some() {
                                // More than one mutable var: skip (we're not doing full elimination).
                                unknown = None;
                                break;
                            }
                            unknown = Some((idx, coeff));
                        } else {
                            sum_known += coeff * asg[idx];
                        }
                    }
                    let Some((u_idx, u_coeff)) = unknown else { continue };
                    if u_coeff == F257::ZERO {
                        continue;
                    }
                    let rhs = target - sum_known;
                    let inv = u_coeff.inverse().unwrap();
                    let new_val = rhs * inv;
                    if asg[u_idx] != new_val {
                        asg[u_idx] = new_val;
                        changed = true;
                    }
                }
                if !changed {
                    break;
                }
            }
        }

        // Fuzz: perturb non-IO variables and assert constraints reject.
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed_fuzz);
        let deltas: [u64; 8] = [1, 2, 3, 4, 8, 16, 64, 128];
        let n_trials: usize = 2000;
        for _ in 0..n_trials {
            // 1-var perturbation.
            let mut asg2 = asg.clone();
            let mut idx = (rng.next_u64() as usize) % asg2.len();
            for _ in 0..64 {
                if can_mutate(&io, &is_bool, idx) {
                    break;
                }
                idx = (idx + 1) % asg2.len();
            }
            if !can_mutate(&io, &is_bool, idx) {
                continue;
            }
            let d = deltas[(rng.next_u64() as usize) % deltas.len()];
            asg2[idx] += F257::from(d);
            if inst.check(&asg2).is_ok() {
                panic!(
                    "found alternate satisfying witness in NTT16 fuzz: mutated idx={} by +{} (mod 257)",
                    idx, d
                );
            }

            // 2-var perturbation (helps catch low-dimensional DOF).
            let mut asg3 = asg.clone();
            let mut idx2 = (rng.next_u64() as usize) % asg3.len();
            for _ in 0..64 {
                if can_mutate(&io, &is_bool, idx2) && idx2 != idx {
                    break;
                }
                idx2 = (idx2 + 1) % asg3.len();
            }
            if !can_mutate(&io, &is_bool, idx2) || idx2 == idx {
                continue;
            }
            let d1 = deltas[(rng.next_u64() as usize) % deltas.len()];
            let d2 = deltas[(rng.next_u64() as usize) % deltas.len()];
            asg3[idx] += F257::from(d1);
            asg3[idx2] += F257::from(d2);
            if inst.check(&asg3).is_ok() {
                panic!(
                    "found alternate satisfying witness in NTT16 fuzz: mutated idx={} by +{}, idx2={} by +{} (mod 257)",
                    idx, d1, idx2, d2
                );
            }
        }

        // Extra: "repair-assisted" fuzzing. This is more solver-like: after perturbing a handful
        // of internal vars, try to repair single-unknown linear constraints.
        let n_trials_repair: usize = 250;
        for _ in 0..n_trials_repair {
            let mut asg4 = asg.clone();
            let mut pinned: BTreeSet<usize> = BTreeSet::new();
            // Apply a few random perturbations first.
            for _k in 0..6 {
                let mut idx = (rng.next_u64() as usize) % asg4.len();
                for _ in 0..64 {
                    if can_mutate(&io, &is_bool, idx) {
                        break;
                    }
                    idx = (idx + 1) % asg4.len();
                }
                if !can_mutate(&io, &is_bool, idx) {
                    continue;
                }
                let d = deltas[(rng.next_u64() as usize) % deltas.len()];
                asg4[idx] += F257::from(d);
                pinned.insert(idx);
            }

            // Attempt linear repair while keeping the perturbations pinned.
            // If it ever becomes satisfying, we've found a real alternate witness.
            linear_repair(&inst, &io, &is_bool, &pinned, &mut asg4, 4);
            if inst.check(&asg4).is_ok() {
                // Double-check it's not identical to baseline (should be impossible if pinned is non-empty,
                // but keep this robust).
                let mut diff_non_io: usize = 0;
                for i in 0..asg4.len() {
                    if io.contains(&i) {
                        continue;
                    }
                    if asg4[i] != asg[i] {
                        diff_non_io += 1;
                        if diff_non_io > 8 {
                            break;
                        }
                    }
                }
                panic!(
                    "found alternate satisfying witness in NTT16 fuzz via linear_repair(): pinned_mutations={} diff_non_io~={}",
                    pinned.len(),
                    diff_non_io
                );
            }
        }
    }
}

