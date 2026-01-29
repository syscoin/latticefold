use ark_ff::Field;
use latticefold::transcript::poseidon::F257;
use symphony::dpp_poseidon::{Constraint, SparseDr1csInstance};
use symphony::dpp_sumcheck::Dr1csBuilder;

use cyclotomic_rings::rings::goldilocks_ntt64 as gl_ntt64;

use super::digits::{f257_to_i32_bal, i32_to_f257};
use super::params::{DIGITS_PER_TRY, LIMB_BITS, LIMBS_U32};

/// Variable reference for a CM IR fragment.
///
/// - `Base(i)` refers to an existing variable in the *base* glue module (same numbering as the base glue instance).
/// - `Local(i)` refers to a variable allocated within this IR fragment (1-indexed; `Local(0)` is unused).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) enum VarRef {
    Base(usize),
    Local(usize),
}

#[derive(Clone, Debug)]
pub(crate) struct IrConstraint {
    pub(crate) a: Vec<(F257, VarRef)>,
    pub(crate) b: Vec<(F257, VarRef)>,
    pub(crate) c: Vec<(F257, VarRef)>,
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
}

impl CmIr {
    #[inline]
    pub(crate) fn new() -> Self {
        Self { local_asg: vec![F257::ONE], constraints: Vec::new() }
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
        self.constraints.push(IrConstraint {
            a: lc,
            b: vec![(F257::ONE, VarRef::Base(0))],
            c: vec![(F257::ZERO, VarRef::Base(0))],
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
        self.constraints.push(IrConstraint {
            a: vec![(F257::ONE, a)],
            b: vec![(F257::ONE, b)],
            c: vec![(F257::ONE, c)],
        });
    }

    /// Lower this IR fragment into an existing sparse DR1CS instance/assignment in-place.
    ///
    /// Local variables are appended to `base_asg`, and constraints are appended to `base_inst`.
    pub(crate) fn lower_into(
        self,
        base_inst: &mut SparseDr1csInstance<F257>,
        base_asg: &mut Vec<F257>,
    ) -> Result<(), String> {
        let base_nvars = base_asg.len();
        // Append local vars (skip local_asg[0]=ONE).
        base_asg.extend_from_slice(&self.local_asg[1..]);
        let map = |v: VarRef, base_nvars: usize| -> usize {
            match v {
                VarRef::Base(i) => i,
                VarRef::Local(j) => {
                    // Local(1) becomes base_nvars, Local(2) base_nvars+1, ...
                    base_nvars + (j - 1)
                }
            }
        };
        for ic in self.constraints {
            let a = ic.a.into_iter().map(|(c, v)| (c, map(v, base_nvars))).collect();
            let b = ic.b.into_iter().map(|(c, v)| (c, map(v, base_nvars))).collect();
            let c = ic.c.into_iter().map(|(c, v)| (c, map(v, base_nvars))).collect();
            base_inst.constraints.push(Constraint { a, b, c });
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
    let base_nvars = gb.assignment.len();
    let mut local_to_var: Vec<usize> = Vec::with_capacity(ir.local_asg.len());
    local_to_var.push(0); // Local(0) unused
    for &v in ir.local_asg.iter().skip(1) {
        local_to_var.push(gb.new_var(v));
    }

    let lowered = LoweredIr { base_nvars, local_to_var };

    for ic in ir.constraints {
        let a = ic.a.into_iter().map(|(c, v)| (c, lowered.map_var(v))).collect();
        let b = ic.b.into_iter().map(|(c, v)| (c, lowered.map_var(v))).collect();
        let c = ic.c.into_iter().map(|(c, v)| (c, lowered.map_var(v))).collect();
        gb.add_constraint(a, b, c);
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
}

impl<'a> IrBuilder<'a> {
    #[inline]
    pub(crate) fn new(base_asg: &'a [F257]) -> Self {
        Self { base_asg, ir: CmIr::new() }
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
        self.ir.constraints.push(IrConstraint { a, b, c });
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

/// Add two balanced base-16 digit vectors of the same length.
///
/// Assumes each digit is in [-8,7]. Enforces output digits in [-8,7] and carry in {-1,0,1}.
pub(crate) fn add_bal16_same_len_ir(b: &mut IrBuilder<'_>, a: &[VarRef], c: &[VarRef]) -> (Vec<VarRef>, VarRef) {
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

    (out, carry.expect("add_bal16_same_len_ir: non-empty input must produce carry var"))
}

/// Negate a balanced base-16 digit vector (little-endian), producing digits in [-8,7].
pub(crate) fn neg_bal16_digits_ir(b: &mut IrBuilder<'_>, x: &[VarRef]) -> (Vec<VarRef>, VarRef) {
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

    (out, carry.expect("neg_bal16_digits_ir: non-empty input must produce carry var"))
}

/// Subtract two balanced base-16 digit vectors of the same length: `a - c`.
pub(crate) fn sub_bal16_same_len_ir(b: &mut IrBuilder<'_>, a: &[VarRef], c: &[VarRef]) -> (Vec<VarRef>, VarRef) {
    assert_eq!(a.len(), c.len());
    let (neg_c, _carry_neg) = neg_bal16_digits_ir(b, c);
    add_bal16_same_len_ir(b, a, &neg_c)
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

/// Allocate a signed carry `c ∈ [-128,127]` as an F257 variable by forbidding +128.
///
/// enforce `(c - 128) * inv = 1` where `inv = (c-128)^{-1}` is witness-known.
pub(crate) fn alloc_carry_pm128_ir(b: &mut IrBuilder<'_>, c: i32) -> VarRef {
    assert!((-128..=127).contains(&c));
    let c_var = b.new_var(i32_to_f257(c));
    debug_assert_eq!(
        f257_to_i32_bal(b.val(c_var)),
        c,
        "alloc_carry_pm128_ir: witness mismatch"
    );
    let diff = b.val(c_var) - F257::from(128u64);
    debug_assert!(diff != F257::ZERO, "alloc_carry_pm128_ir: witness hit forbidden value 128");
    let inv = b.new_var(diff.inverse().unwrap());
    // (c - 128) * inv = 1
    b.add_constraint(
        vec![(F257::ONE, c_var), (-F257::from(128u64), b.one())],
        vec![(F257::ONE, inv)],
        vec![(F257::ONE, b.one())],
    );
    c_var
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

/// Enforce: x*k == q*p + r (mod p) using a base-16 carry chain, without materializing x*k digits.
///
/// This mirrors `goldilocks.rs::mul_const_mod_p`'s internal helper
/// `enforce_prod_const_eq_qp_plus_r_bal16`.
fn enforce_prod_const_eq_qp_plus_r_bal16_ir(
    b: &mut IrBuilder<'_>,
    x_d: &[VarRef; 17],
    k_d_const: &[i8; 17],
    q_d: &[VarRef; 17],
    p_d_const: &[i8; 17],
    r_d: &[VarRef; 17],
) {
    let max_len = 17usize
        .max(r_d.len())
        .max(q_d.len().saturating_add(p_d_const.len()).saturating_sub(1))
        + 1;

    let mut carry_i32: i32 = 0;
    let mut carry_var = alloc_carry_pm128_ir(b, 0);

    for k in 0..max_len {
        let mut sum: i32 = carry_i32;
        let mut lc: Vec<(F257, VarRef)> = Vec::with_capacity(64);
        lc.push((F257::ONE, carry_var));

        // -r_k
        if k < r_d.len() {
            let rk = f257_to_i32_bal(b.val(r_d[k]));
            sum -= rk;
            lc.push((-F257::ONE, r_d[k]));
        }

        // + Σ_i x_i * k_{k-i}
        for i in 0..17 {
            if i > k {
                break;
            }
            let j = k - i;
            if j >= 17 {
                continue;
            }
            let kd = k_d_const[j] as i32;
            if kd == 0 {
                continue;
            }
            let xi = f257_to_i32_bal(b.val(x_d[i]));
            sum += xi * kd;
            lc.push((i32_to_f257(kd), x_d[i]));
        }

        // - Σ_i q_i * p_{k-i}
        for i in 0..q_d.len() {
            if i > k {
                break;
            }
            let j = k - i;
            if j >= 17 {
                continue;
            }
            let pd = p_d_const[j] as i32;
            if pd == 0 {
                continue;
            }
            let qi = f257_to_i32_bal(b.val(q_d[i]));
            sum -= qi * pd;
            lc.push((-i32_to_f257(pd), q_d[i]));
        }

        debug_assert!(sum % 16 == 0, "const-mul carry check not divisible: sum={sum} at k={k}");
        let carry_next: i32 = sum / 16;
        debug_assert!(
            (-128..=127).contains(&carry_next),
            "const-mul carry out of pm128 bound: {carry_next} at k={k} (sum={sum})"
        );
        let carry_next_var = alloc_carry_pm128_ir(b, carry_next);
        lc.push((-F257::from(16u64), carry_next_var));

        // lc == 0
        b.enforce_lc_eq_zero(lc);
        carry_var = carry_next_var;
        carry_i32 = carry_next;
    }
    // Final carry must be 0.
    b.ir.enforce_var_eq_const(carry_var, F257::ZERO);
}

/// Digit-domain const multiplication mod Goldilocks p: r = x * k (mod p).
///
/// This returns `r` as 17 bal16 digits, and enforces the relation via the carry-chain gadget above.
pub(crate) fn goldilocks_mul_const_mod_p_digits_ir(
    b: &mut IrBuilder<'_>,
    x: &[VarRef; 17],
    k: u64,
    p_u64: u64,
    p_d_const: &[i8; 17],
) -> [VarRef; 17] {
    let x_u = digits_to_u64_witness_ir(b, x);
    let prod: u128 = (x_u as u128) * (k as u128);
    let q_u: u64 = (prod / (p_u64 as u128)) as u64;
    let r_u: u64 = (prod % (p_u64 as u128)) as u64;
    let q_d = alloc_u64_as_bal16_digits_witness_ir(b, q_u);
    let r_d = alloc_u64_as_bal16_digits_witness_ir(b, r_u);
    let k_d_const = u64_to_bal16_digits_le_const(k);
    enforce_prod_const_eq_qp_plus_r_bal16_ir(b, x, &k_d_const, &q_d, p_d_const, &r_d);
    r_d
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

    // carry_0 = 0
    let mut carry_var = alloc_carry_pm2_ir(b, 0);
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
            (-2..=2).contains(&carry_next),
            "add_mod_p carry out of pm2: {carry_next} (sum={sum}) at k={k}"
        );
        let carry_next_var = alloc_carry_pm2_ir(b, carry_next);

        // carry + a + c - r - q*pk - 16*carry_next = 0
        let mut lc: Vec<(F257, VarRef)> = Vec::with_capacity(6);
        lc.push((F257::ONE, carry_var));
        lc.push((F257::ONE, a[k]));
        lc.push((F257::ONE, c[k]));
        lc.push((-F257::ONE, r[k]));
        if k < 17 && p_d_const[k] != 0 {
            lc.push((-i32_to_f257(p_d_const[k] as i32), q));
        }
        lc.push((-F257::from(16u64), carry_next_var));
        b.enforce_lc_eq_zero(lc);

        carry_var = carry_next_var;
        carry_i32 = carry_next;
    }
    b.ir.enforce_var_eq_const(carry_var, F257::ZERO);
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

    let mut carry_var = alloc_carry_pm2_ir(b, 0);
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
            (-2..=2).contains(&carry_next),
            "sub_mod_p carry out of pm2: {carry_next} (sum={sum}) at k={k}"
        );
        let carry_next_var = alloc_carry_pm2_ir(b, carry_next);

        // carry + a - c - r + q*pk - 16*carry_next = 0
        let mut lc: Vec<(F257, VarRef)> = Vec::with_capacity(6);
        lc.push((F257::ONE, carry_var));
        lc.push((F257::ONE, a[k]));
        lc.push((-F257::ONE, c[k]));
        lc.push((-F257::ONE, r[k]));
        if k < 17 && p_d_const[k] != 0 {
            lc.push((i32_to_f257(p_d_const[k] as i32), q));
        }
        lc.push((-F257::from(16u64), carry_next_var));
        b.enforce_lc_eq_zero(lc);

        carry_var = carry_next_var;
        carry_i32 = carry_next;
    }
    b.ir.enforce_var_eq_const(carry_var, F257::ZERO);
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

/// Enforce: a*b == q*p + r in base-16 carry chain, without materializing a*b digits.
///
/// This is the var×var analogue of `enforce_prod_const_eq_qp_plus_r_bal16_ir`.
fn enforce_prod_var_eq_qp_plus_r_bal16_ir(
    b: &mut IrBuilder<'_>,
    a_d: &[VarRef; 17],
    b_d: &[VarRef; 17],
    q_d: &[VarRef; 17],
    p_d_const: &[i8; 17],
    r_d: &[VarRef; 17],
) {
    // Headroom digit: product and q*p both have length up to 33, so loop to 34.
    let max_len = 17usize
        .max(r_d.len())
        .max(q_d.len().saturating_add(p_d_const.len()).saturating_sub(1))
        .max(a_d.len().saturating_add(b_d.len()).saturating_sub(1))
        + 1;

    // Precompute digit products a_i * b_j (289 mul constraints, reused across all k).
    let mut prod: [[VarRef; 17]; 17] = [[VarRef::Base(0); 17]; 17];
    for i in 0..17 {
        for j in 0..17 {
            let pij = b.new_var(b.val(a_d[i]) * b.val(b_d[j]));
            b.enforce_mul(a_d[i], b_d[j], pij);
            prod[i][j] = pij;
        }
    }

    let mut carry_var = alloc_carry_pm128_ir(b, 0);
    let mut carry_i32: i32 = 0;

    for k in 0..max_len {
        let mut sum: i32 = carry_i32;
        let mut lc: Vec<(F257, VarRef)> = Vec::with_capacity(4 + 17 + 17);
        lc.push((F257::ONE, carry_var));

        // -r_k
        if k < r_d.len() {
            let rk = f257_to_i32_bal(b.val(r_d[k]));
            sum -= rk;
            lc.push((-F257::ONE, r_d[k]));
        }

        // + Σ_{i+j=k} a_i * b_j
        let i_min = k.saturating_sub(16);
        let i_max = core::cmp::min(16, k);
        for i in i_min..=i_max {
            let j = k - i;
            debug_assert!(j <= 16);
            let aval = f257_to_i32_bal(b.val(a_d[i]));
            let bval = f257_to_i32_bal(b.val(b_d[j]));
            sum += aval * bval;
            lc.push((F257::ONE, prod[i][j]));
        }

        // - Σ_i q_i * p_{k-i}
        for i in 0..q_d.len() {
            if i > k {
                break;
            }
            let j = k - i;
            if j >= 17 {
                continue;
            }
            let pd = p_d_const[j] as i32;
            if pd == 0 {
                continue;
            }
            let qi = f257_to_i32_bal(b.val(q_d[i]));
            sum -= qi * pd;
            lc.push((-i32_to_f257(pd), q_d[i]));
        }

        debug_assert!(sum % 16 == 0, "var-mul carry check not divisible: sum={sum} at k={k}");
        let carry_next: i32 = sum / 16;
        debug_assert!(
            (-128..=127).contains(&carry_next),
            "var-mul carry out of pm128 bound: {carry_next} at k={k} (sum={sum})"
        );
        let carry_next_var = alloc_carry_pm128_ir(b, carry_next);
        lc.push((-F257::from(16u64), carry_next_var));

        b.enforce_lc_eq_zero(lc);
        carry_var = carry_next_var;
        carry_i32 = carry_next;
    }
    b.ir.enforce_var_eq_const(carry_var, F257::ZERO);
}

/// Digit-domain Goldilocks multiplication (mod p) in IR form.
pub(crate) fn goldilocks_mul_mod_p_digits_ir(
    b: &mut IrBuilder<'_>,
    a: &[VarRef; 17],
    c: &[VarRef; 17],
    p_u64: u64,
    p_d_const: &[i8; 17],
) -> [VarRef; 17] {
    let a_u = digits_to_u64_witness_ir(b, a);
    let c_u = digits_to_u64_witness_ir(b, c);
    let prod_u: u128 = (a_u as u128) * (c_u as u128);
    let q_u: u64 = (prod_u / (p_u64 as u128)) as u64;
    let r_u: u64 = (prod_u % (p_u64 as u128)) as u64;

    let q_d = alloc_u64_as_bal16_digits_witness_ir(b, q_u);
    let r_d = alloc_u64_as_bal16_digits_witness_ir(b, r_u);

    enforce_prod_var_eq_qp_plus_r_bal16_ir(b, a, c, &q_d, p_d_const, &r_d);
    r_d
}

/// Negacyclic ring multiplication for `d=64` over Goldilocks using an NTT-based method, emitting IR.
///
/// Mirrors `goldilocks::ring_mul_negacyclic_ntt_goldilocks_d64`, but is `Dr1csBuilder`-free:
/// it only allocates local vars + constraints in `CmIr`.
pub(crate) fn ring_mul_negacyclic_ntt_goldilocks_d64_ir(
    b: &mut IrBuilder<'_>,
    a: &[[VarRef; 17]; 64],
    c: &[[VarRef; 17]; 64],
) -> [[VarRef; 17]; 64] {
    // Shared schedule constants from `cyclotomic-rings` (keeps host + gate in sync).
    let p: u64 = gl_ntt64::GOLDILOCKS_P_U64;
    let omega: u64 = gl_ntt64::OMEGA_U64;
    let omega_inv: u64 = gl_ntt64::OMEGA_INV_U64;
    let inv_n: u64 = gl_ntt64::INV_N_U64;

    let p_d_const = u64_to_bal16_digits_le_const(p);

    let zero_digit = alloc_zero_const_ir(b);
    let zero_scalar: [VarRef; 17] = [zero_digit; 17];

    fn ntt_in_place(
        b: &mut IrBuilder<'_>,
        a: &mut [[VarRef; 17]; 64],
        omega: u64,
        p_u64: u64,
        p_d: &[i8; 17],
        zero: &[VarRef; 17],
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
                        goldilocks_sub_mod_p_digits_ir(b, zero, &a[start + j + half], p_u64, p_d)
                    } else {
                        goldilocks_mul_const_mod_p_digits_ir(b, &a[start + j + half], w, p_u64, p_d)
                    };
                    a[start + j] = goldilocks_add_mod_p_digits_ir(b, &u, &v, p_u64, p_d);
                    a[start + j + half] = goldilocks_sub_mod_p_digits_ir(b, &u, &v, p_u64, p_d);
                }
            }
            len *= 2;
        }
    }

    fn intt_in_place(
        b: &mut IrBuilder<'_>,
        a: &mut [[VarRef; 17]; 64],
        omega_inv: u64,
        inv_n: u64,
        p_u64: u64,
        p_d: &[i8; 17],
        zero: &[VarRef; 17],
    ) {
        ntt_in_place(b, a, omega_inv, p_u64, p_d, zero);
        // scale by n^{-1}
        for i in 0..64 {
            a[i] = goldilocks_mul_const_mod_p_digits_ir(b, &a[i], inv_n, p_u64, p_d);
        }
    }

    // Negacyclic via twist by ψ^i (ψ is primitive 128th root).
    let mut a_tw = [[zero_digit; 17]; 64];
    let mut c_tw = [[zero_digit; 17]; 64];
    for i in 0..64 {
        let psi_pow: u64 = gl_ntt64::PSI_POWS_64[i];
        if psi_pow == 1 {
            a_tw[i] = a[i];
            c_tw[i] = c[i];
        } else {
            a_tw[i] = goldilocks_mul_const_mod_p_digits_ir(b, &a[i], psi_pow, p, &p_d_const);
            c_tw[i] = goldilocks_mul_const_mod_p_digits_ir(b, &c[i], psi_pow, p, &p_d_const);
        }
    }

    ntt_in_place(b, &mut a_tw, omega, p, &p_d_const, &zero_scalar);
    ntt_in_place(b, &mut c_tw, omega, p, &p_d_const, &zero_scalar);

    // Pointwise multiply.
    for i in 0..64 {
        a_tw[i] = goldilocks_mul_mod_p_digits_ir(b, &a_tw[i], &c_tw[i], p, &p_d_const);
    }

    intt_in_place(b, &mut a_tw, omega_inv, inv_n, p, &p_d_const, &zero_scalar);

    // Untwist by ψ^{-i}.
    let mut out = [[zero_digit; 17]; 64];
    for i in 0..64 {
        let psi_inv_pow: u64 = gl_ntt64::PSI_INV_POWS_64[i];
        out[i] = if psi_inv_pow == 1 {
            a_tw[i]
        } else {
            goldilocks_mul_const_mod_p_digits_ir(b, &a_tw[i], psi_inv_pow, p, &p_d_const)
        };
    }
    out
}

