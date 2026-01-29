use ark_ff::Field;
use latticefold::transcript::poseidon::F257;
use symphony::dpp_poseidon::{Constraint, SparseDr1csInstance};
use symphony::dpp_sumcheck::Dr1csBuilder;

use cyclotomic_rings::rings::goldilocks_ntt64 as gl_ntt64;

use super::digits::{f257_to_i32_bal, i32_to_f257};

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
    let x = b.new_var(if bit { F257::ONE } else { F257::ZERO });
    let xm1 = b.new_var(b.val(x) - F257::ONE);
    // xm1 = x - 1
    // xm1 = x + (-1)
    b.enforce_affine(xm1, vec![(F257::ONE, x), (-F257::ONE, b.one())]);
    // x*(x-1)=0
    let z = b.new_var(F257::ZERO);
    b.enforce_mul(x, xm1, z);
    // force z == 0
    b.ir.enforce_var_eq_const(z, F257::ZERO);
    x
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
/// This mirrors `digits::alloc_carry_pm128`:
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

/// Allocate a signed carry `c ∈ {-1,0,1}` using the vanishing polynomial (c-1)c(c+1)=0.
fn alloc_carry_pm1_ir(b: &mut IrBuilder<'_>, c: i32) -> VarRef {
    debug_assert!((-1..=1).contains(&c));
    let c_var = b.new_var(i32_to_f257(c));

    // cm1 = c - 1
    let cm1 = b.new_var(b.val(c_var) - F257::ONE);
    b.enforce_lc_eq_zero(vec![
        (F257::ONE, cm1),
        (-F257::ONE, c_var),
        (F257::ONE, b.one()),
    ]);

    // t = (c-1)*c
    let t = b.new_var(b.val(cm1) * b.val(c_var));
    b.enforce_mul(cm1, c_var, t);

    // cp1 = c + 1
    let cp1 = b.new_var(b.val(c_var) + F257::ONE);
    b.enforce_lc_eq_zero(vec![
        (F257::ONE, cp1),
        (-F257::ONE, c_var),
        (-F257::ONE, b.one()),
    ]);

    // t*(c+1) = 0
    let z = b.new_var(F257::ZERO);
    b.ir.enforce_var_eq_const(z, F257::ZERO);
    b.enforce_mul(t, cp1, z);

    c_var
}

/// Allocate a signed carry `c ∈ [-16,15]` by 5-bit decomposition of (c+16).
fn alloc_carry_pm16_ir(b: &mut IrBuilder<'_>, c: i32) -> VarRef {
    debug_assert!((-16..=15).contains(&c));
    let off_u8: u8 = (c + 16) as u8; // in [0,31]

    let mut bits = [VarRef::Base(0); 5];
    for i in 0..5 {
        bits[i] = alloc_bool_ir(b, ((off_u8 >> i) & 1) == 1);
    }

    let off_var = b.new_var(F257::from(off_u8 as u64));
    // off = Σ 2^i bits[i]
    b.enforce_lc_eq_zero(vec![
        (F257::ONE, off_var),
        (-F257::ONE, bits[0]),
        (-F257::from(2u64), bits[1]),
        (-F257::from(4u64), bits[2]),
        (-F257::from(8u64), bits[3]),
        (-F257::from(16u64), bits[4]),
    ]);

    // c = off - 16
    let c_var = b.new_var(i32_to_f257(c));
    b.enforce_lc_eq_zero(vec![
        (F257::ONE, c_var),
        (-F257::ONE, off_var),
        (F257::from(16u64), b.one()),
    ]);
    c_var
}

/// Allocate a signed carry `c ∈ [-8,7]` by 4-bit decomposition of (c+8).
fn alloc_carry_pm8_ir(b: &mut IrBuilder<'_>, c: i32) -> VarRef {
    debug_assert!((-8..=7).contains(&c));
    let off_u8: u8 = (c + 8) as u8; // in [0,15]

    let mut bits = [VarRef::Base(0); 4];
    for i in 0..4 {
        bits[i] = alloc_bool_ir(b, ((off_u8 >> i) & 1) == 1);
    }

    let off_var = b.new_var(F257::from(off_u8 as u64));
    // off = Σ 2^i bits[i]
    b.enforce_lc_eq_zero(vec![
        (F257::ONE, off_var),
        (-F257::ONE, bits[0]),
        (-F257::from(2u64), bits[1]),
        (-F257::from(4u64), bits[2]),
        (-F257::from(8u64), bits[3]),
    ]);

    // c = off - 8
    let c_var = b.new_var(i32_to_f257(c));
    b.enforce_lc_eq_zero(vec![
        (F257::ONE, c_var),
        (-F257::ONE, off_var),
        (F257::from(8u64), b.one()),
    ]);
    c_var
}

#[inline]
fn u64_to_bal16_digits_le_const(mut x: u64) -> [i8; 17] {
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
fn alloc_u64_as_bal16_digits_witness_ir(b: &mut IrBuilder<'_>, x: u64) -> [VarRef; 17] {
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
    // Sound no-wrap enforcement (IR analogue of `goldilocks.rs::enforce_prod_eq_qp_plus_r_bal16`):
    // materialize `prod = x*k` and `qp = q*p` via the sound const-RHS multiplier, add `r`, and
    // enforce equality to `prod` using the bal16 add gadget (pm1 carries).
    let zero = alloc_zero_const_ir(b);

    // prod = x*k  (const-RHS)
    let prod = mul_bal16_long_by_const_rhs_ir(b, x_d, k_d_const);

    // qp = q*p  (const-RHS)
    let qp = mul_bal16_long_by_const_rhs_ir(b, q_d, p_d_const);

    let max_len = prod
        .len()
        .max(r_d.len())
        .max(qp.len())
        .max(q_d.len().saturating_add(p_d_const.len()).saturating_sub(1))
        + 1;

    let prod_pad = pad_bal16_ir(b, prod, max_len, zero);
    let r_pad = pad_bal16_ir(b, r_d.to_vec(), max_len, zero);
    let qp_pad = pad_bal16_ir(b, qp, max_len, zero);

    let (sum, carry) = add_bal16_same_len_ir(b, &qp_pad, &r_pad);
    b.ir.enforce_var_eq_const(carry, F257::ZERO);

    enforce_bal16_vec_eq_ir(b, &prod_pad, &sum);
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

#[inline]
fn enforce_bal16_vec_eq_ir(b: &mut IrBuilder<'_>, a: &[VarRef], c: &[VarRef]) {
    debug_assert_eq!(a.len(), c.len());
    for (&ai, &ci) in a.iter().zip(c.iter()) {
        b.enforce_lc_eq_zero(vec![(F257::ONE, ai), (-F257::ONE, ci)]);
    }
}

#[inline]
fn pad_bal16_ir(b: &mut IrBuilder<'_>, mut v: Vec<VarRef>, target_len: usize, zero: VarRef) -> Vec<VarRef> {
    if v.len() > target_len {
        for &dv in &v[target_len..] {
            b.ir.enforce_var_eq_const(dv, F257::ZERO);
        }
        v.truncate(target_len);
        return v;
    }
    if v.len() < target_len {
        v.extend(std::iter::repeat(zero).take(target_len - v.len()));
    }
    v
}

/// Add two balanced base-16 digit vectors of the same length.
///
/// Assumes each digit is in [-8,7]. Enforces output digits in [-8,7] and carry in {-1,0,1}.
fn add_bal16_same_len_ir(b: &mut IrBuilder<'_>, a: &[VarRef], c: &[VarRef]) -> (Vec<VarRef>, VarRef) {
    debug_assert_eq!(a.len(), c.len());
    let n = a.len();
    let mut out: Vec<VarRef> = Vec::with_capacity(n);
    let mut carry_i32: i32 = 0;
    let mut carry = alloc_zero_const_ir(b);

    for i in 0..n {
        let ai = f257_to_i32_bal(b.val(a[i]));
        let ci = f257_to_i32_bal(b.val(c[i]));
        let sum = ai + ci + carry_i32;

        // Choose carry_out so remainder in [-8,7].
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
        b.enforce_lc_eq_zero(vec![
            (F257::ONE, a[i]),
            (F257::ONE, c[i]),
            (F257::ONE, carry),
            (-F257::ONE, out_digit),
            (-F257::from(16u64), carry_next_var),
        ]);

        out.push(out_digit);
        carry_i32 = carry_next;
        carry = carry_next_var;
    }

    (out, carry)
}

#[inline]
fn add_bal16_loose_in_place_ir(b: &mut IrBuilder<'_>, acc: &mut [VarRef], src: &[VarRef]) {
    debug_assert_eq!(acc.len(), src.len());
    for i in 0..acc.len() {
        let v = b.new_var(b.val(acc[i]) + b.val(src[i]));
        b.enforce_lc_eq_zero(vec![(F257::ONE, v), (-F257::ONE, acc[i]), (-F257::ONE, src[i])]);
        acc[i] = v;
    }
}

fn normalize_bal16_loose_same_len_with_bound_ir(
    b: &mut IrBuilder<'_>,
    loose: &[VarRef],
    digit_abs_bound: i32,
) -> (Vec<VarRef>, VarRef) {
    debug_assert!(digit_abs_bound >= 0);
    // Critical for no-wrap soundness when interpreting F257 as integers.
    debug_assert!(digit_abs_bound < 128);

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
        } else {
            // For our current uses (17×17 const-RHS with acc_bound<128), this should not happen.
            alloc_carry_pm128_ir(b, c)
        }
    }

    // Compute a conservative, statement-derived carry bound schedule from `digit_abs_bound`.
    let mut carry_bound: i32 = 0;
    let mut carry_bounds: Vec<i32> = Vec::with_capacity(loose.len());
    for _ in 0..loose.len() {
        let max_sum = digit_abs_bound + carry_bound;
        carry_bound = ((max_sum + 8) / 16) + 1;
        carry_bounds.push(carry_bound);
        debug_assert!(carry_bound < 128);
    }

    let mut out: Vec<VarRef> = Vec::with_capacity(loose.len());
    let mut carry_i32: i32 = 0;
    let mut carry_var = alloc_zero_const_ir(b);

    // div_floor(x/16) for possibly-negative x.
    let div_floor = |x: i32, d: i32| -> i32 {
        debug_assert!(d > 0);
        if x >= 0 { x / d } else { -(((-x) + d - 1) / d) }
    };

    for (i, &dv) in loose.iter().enumerate() {
        let di = f257_to_i32_bal(b.val(dv));
        debug_assert!((-digit_abs_bound..=digit_abs_bound).contains(&di));
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
        debug_assert!((-carry_bounds[i]..=carry_bounds[i]).contains(&carry_next));

        let rem_digit = alloc_bal16_digit_ir(b, rem as i8);
        let carry_next_var = alloc_carry_with_bound_ir(b, carry_next, carry_bounds[i]);

        // loose_i + carry_i - rem_i - 16*carry_{i+1} = 0
        b.enforce_lc_eq_zero(vec![
            (F257::ONE, dv),
            (F257::ONE, carry_var),
            (-F257::ONE, rem_digit),
            (-F257::from(16u64), carry_next_var),
        ]);

        out.push(rem_digit);
        carry_i32 = carry_next;
        carry_var = carry_next_var;
    }

    (out, carry_var)
}

fn rebalance_tail_pm16_to_pm1_ir(b: &mut IrBuilder<'_>, digits: &[VarRef]) -> Vec<VarRef> {
    debug_assert!(!digits.is_empty());
    let l = digits.len();
    let tail = f257_to_i32_bal(b.val(digits[l - 1]));
    debug_assert!((-16..=15).contains(&tail));

    let mut carry1 = if tail >= 0 { (tail + 8) / 16 } else { -(((-tail) + 8) / 16) };
    let mut rem = tail - 16 * carry1;
    while rem > 7 {
        carry1 += 1;
        rem -= 16;
    }
    while rem < -8 {
        carry1 -= 1;
        rem += 16;
    }
    debug_assert!((-8..=7).contains(&rem));
    debug_assert!((-1..=1).contains(&carry1));

    let rem_digit = alloc_bal16_digit_ir(b, rem as i8);
    let carry1_var = alloc_carry_pm1_ir(b, carry1);
    b.enforce_lc_eq_zero(vec![
        (F257::ONE, digits[l - 1]),
        (-F257::ONE, rem_digit),
        (-F257::from(16u64), carry1_var),
    ]);

    let mut out = Vec::with_capacity(l + 1);
    out.extend_from_slice(&digits[..l - 1]);
    out.push(rem_digit);
    out.push(carry1_var);
    out
}

fn shift_pad_bal16_ir(digits: &[VarRef], shift: usize, target_len: usize, zero_digit: VarRef) -> Vec<VarRef> {
    debug_assert!(shift <= target_len);
    debug_assert!(digits.len() + shift <= target_len);
    let mut out = Vec::with_capacity(target_len);
    out.extend(std::iter::repeat(zero_digit).take(shift));
    out.extend_from_slice(digits);
    out.extend(std::iter::repeat(zero_digit).take(target_len - shift - digits.len()));
    out
}

/// Multiply balanced base-16 digits by **constant** balanced digits (little-endian),
/// specialized for `a_len <= 4`.
fn mul_bal16_small_const_rhs4_ir(
    b: &mut IrBuilder<'_>,
    a: &[VarRef; 4],
    a_len: usize,
    bb_const: &[i8],
) -> Vec<VarRef> {
    debug_assert!(a_len >= 1 && a_len <= 4);
    let la = a_len;
    let lb = bb_const.len();

    #[inline]
    fn f257_from_i8(x: i8) -> F257 {
        if x >= 0 { F257::from(x as u64) } else { -F257::from((-x) as u64) }
    }

    let mut out: Vec<VarRef> = Vec::with_capacity(la + lb);
    let mut carry_i32: i32 = 0;
    let mut carry_var = alloc_zero_const_ir(b);

    let div_floor = |x: i32, d: i32| -> i32 {
        debug_assert!(d > 0);
        if x >= 0 { x / d } else { -(((-x) + d - 1) / d) }
    };

    for k in 0..(la + lb - 1) {
        let mut sum: i32 = carry_i32;
        let mut lc: Vec<(F257, VarRef)> = Vec::with_capacity(8);
        lc.push((F257::ONE, carry_var));

        for i in 0..la {
            let j = k as i32 - i as i32;
            if j < 0 || j >= lb as i32 {
                continue;
            }
            let j = j as usize;
            let aval = f257_to_i32_bal(b.val(a[i]));
            let bval = bb_const[j] as i32;
            sum += aval * bval;
            let cf = f257_from_i8(bb_const[j]);
            if cf != F257::ZERO {
                lc.push((cf, a[i]));
            }
        }

        // With <=4 terms of magnitude <=56 and carry in <=15, |sum| < 257.
        let mut carry = div_floor(sum + 8, 16);
        let mut rem = sum - 16 * carry;
        while rem > 7 {
            carry += 1;
            rem -= 16;
        }
        while rem < -8 {
            carry -= 1;
            rem += 16;
        }
        debug_assert!((-8..=7).contains(&rem));
        debug_assert!((-16..=15).contains(&carry));

        let digit_var = alloc_bal16_digit_ir(b, rem as i8);
        let carry_out_var = alloc_carry_pm16_ir(b, carry);
        lc.push((-F257::ONE, digit_var));
        lc.push((-F257::from(16u64), carry_out_var));
        b.enforce_lc_eq_zero(lc);

        out.push(digit_var);
        carry_i32 = carry;
        carry_var = carry_out_var;
    }

    out.push(carry_var);
    out
}

/// Multiply a long balanced digit vector by a **constant** long balanced digit vector.
///
/// IR analogue of `digits::mul_bal16_long_by_const_rhs`, implementing the provably-sound Fox #1
/// loose-accumulation path (sufficient for all current uses: 17×17 with acc_bound<128).
fn mul_bal16_long_by_const_rhs_ir(b: &mut IrBuilder<'_>, a: &[VarRef], bb_const: &[i8]) -> Vec<VarRef> {
    if a.is_empty() || bb_const.is_empty() {
        return vec![alloc_zero_const_ir(b)];
    }

    const BLK: usize = 4;
    let zero = alloc_zero_const_ir(b);
    let blocks = (a.len() + (BLK - 1)) / BLK;
    let per_block_len = bb_const.len() + BLK + 2;
    let target_len = per_block_len + BLK * (blocks - 1) + 3;

    let per_term_bound: i32 = 10;
    let acc_bound: i32 = (blocks as i32) * per_term_bound;

    debug_assert!(a.len() <= 19 && bb_const.len() <= 17 && acc_bound < 128);

    // Build all shifted block-products.
    let mut terms: Vec<Vec<VarRef>> = Vec::with_capacity(blocks);
    for blk in 0..blocks {
        let start = blk * BLK;
        let end = core::cmp::min(start + BLK, a.len());
        let mut coeff4 = [zero; 4];
        for j in 0..(end - start) {
            coeff4[j] = a[start + j];
        }
        let raw = mul_bal16_small_const_rhs4_ir(b, &coeff4, end - start, bb_const);
        let reb = rebalance_tail_pm16_to_pm1_ir(b, &raw);
        let shifted = shift_pad_bal16_ir(&reb, blk * BLK, target_len, zero);
        terms.push(shifted);
    }

    // Fox #1: accumulate as loose digits, normalize once.
    let mut acc = vec![zero; target_len];
    for t in &terms {
        add_bal16_loose_in_place_ir(b, &mut acc, t);
    }
    let (norm, carry) = normalize_bal16_loose_same_len_with_bound_ir(b, &acc, acc_bound);
    b.ir.enforce_var_eq_const(carry, F257::ZERO);
    norm
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

