//! WE/DPP gate arithmetization for LatticeFold+ (sparse dR1CS over a prime field).
//!
//! This module is a research/bench frontend: it arithmetizes the *verifier* computation,
//! keeping the relation log-scale in `n` and linear in the verifier-visible message sizes.

use ark_crypto_primitives::sponge::poseidon::PoseidonConfig;
use ark_ff::{BigInteger, Field, PrimeField};
use stark_rings::{CoeffRing, OverField, PolyRing, Zq};

use crate::recording_transcript::{PoseidonTraceOp as LfPoseidonTraceOp, PoseidonTranscriptTrace};
use crate::we_statement::WeParams;

// Reuse symphony’s sparse dR1CS primitives and Poseidon arithmetizer.
use symphony::dpp_poseidon::{
    merge_sparse_dr1cs_share_one_with_glue, poseidon_sponge_dr1cs_from_trace_with_wiring_and_bytes,
    Constraint, PoseidonDr1csWiring, SparseDr1csInstance,
};
use symphony::dpp_sumcheck::Dr1csBuilder;
use symphony::dpp_sumcheck::{sumcheck_verify_degree3, RingVars};

/// Output of WE-gate arithmetization (single merged sparse dR1CS instance).
#[derive(Clone, Debug)]
pub struct WeDr1csOutput<F: PrimeField> {
    pub inst: SparseDr1csInstance<F>,
    pub assignment: Vec<F>,
    /// Number of public variables `l` (prefix of the assignment vector) intended as `x`.
    pub public_len: usize,
}

fn lf_ops_to_symphony_ops<F: PrimeField>(ops: &[LfPoseidonTraceOp<F>]) -> Vec<symphony::transcript::PoseidonTraceOp<F>> {
    ops.iter()
        .map(|op| match op {
            LfPoseidonTraceOp::Absorb(v) => symphony::transcript::PoseidonTraceOp::Absorb(v.clone()),
            LfPoseidonTraceOp::SqueezeField(v) => symphony::transcript::PoseidonTraceOp::SqueezeField(v.clone()),
            LfPoseidonTraceOp::SqueezeBytes { n, out } => symphony::transcript::PoseidonTraceOp::SqueezeBytes { n: *n, out: out.clone() },
        })
        .collect()
}

/// Enforce that each `get_challenge` re-absorb equals the corresponding `SqueezeField` output.
///
/// Our trace transcript records `get_challenge` as:
/// - `SqueezeField(out)`
/// - `Absorb(out)`  (Fiat–Shamir re-absorb)
fn enforce_reabsorb_equals_squeeze<F: PrimeField>(
    inst: &mut SparseDr1csInstance<F>,
    wiring: &PoseidonDr1csWiring,
    ops: &[symphony::transcript::PoseidonTraceOp<F>],
) -> Result<(), String> {
    let mut absorb_idx = 0usize;
    let mut squeeze_idx = 0usize;
    for op in ops {
        match op {
            symphony::transcript::PoseidonTraceOp::Absorb(_) => {
                absorb_idx += 1;
            }
            symphony::transcript::PoseidonTraceOp::SqueezeField(out) => {
                // Next op must be Absorb(out)
                // We enforce equality elementwise: absorb_var == squeeze_var.
                let (sq_start, sq_len) = wiring
                    .squeeze_field_ranges
                    .get(squeeze_idx)
                    .copied()
                    .ok_or("poseidon wiring squeeze_field_ranges oob")?;
                squeeze_idx += 1;
                if sq_len != out.len() {
                    return Err("poseidon squeeze length mismatch".to_string());
                }
                // IMPORTANT: `absorb_idx` tracks how many Absorb ops we've *already processed*.
                // The re-absorb corresponding to this squeeze is the *next* Absorb op in the trace,
                // i.e. it has index `absorb_idx` in `absorb_ranges`. Do NOT increment `absorb_idx`
                // here; it will be incremented when the loop reaches that Absorb op.
                let (ab_start, ab_len) = wiring
                    .absorb_ranges
                    .get(absorb_idx)
                    .copied()
                    .ok_or("poseidon wiring absorb_ranges oob after squeeze")?;
                if ab_len != sq_len {
                    return Err("poseidon reabsorb length mismatch".to_string());
                }
                for j in 0..sq_len {
                    let v_sq = wiring.squeeze_field_vars[sq_start + j];
                    let v_ab = wiring.absorb_vars[ab_start + j];
                    // (v_ab - v_sq) * 1 = 0
                    inst.constraints.push(Constraint {
                        a: vec![(F::ONE, v_ab), (-F::ONE, v_sq)],
                        b: vec![(F::ONE, 0)],
                        c: vec![(F::ZERO, 0)],
                    });
                }
            }
            symphony::transcript::PoseidonTraceOp::SqueezeBytes { .. } => {}
        }
    }
    Ok(())
}

type BF<R> = <<R as PolyRing>::BaseRing as Field>::BasePrimeField;

fn ring_to_ringvars<R>(
    b: &mut Dr1csBuilder<BF<R>>,
    x: &R,
) -> RingVars
where
    R: PolyRing,
    R::BaseRing: Field,
{
    let mut coeffs = Vec::with_capacity(R::dimension());
    for c in x.coeffs() {
        let fp = c.to_base_prime_field_elements().into_iter().next().unwrap();
        let v = b.new_var(fp);
        coeffs.push(v);
    }
    RingVars::new(coeffs)
}

fn scalar_to_ringvars<R>(
    b: &mut Dr1csBuilder<BF<R>>,
    x: BF<R>,
) -> RingVars
where
    R: PolyRing,
    R::BaseRing: Field,
{
    let d = R::dimension();
    let mut coeffs = Vec::with_capacity(d);
    let v0 = b.new_var(x);
    coeffs.push(v0);
    for _ in 1..d {
        let vz = b.new_var(BF::<R>::ZERO);
        b.enforce_var_eq_const(vz, BF::<R>::ZERO);
        coeffs.push(vz);
    }
    RingVars::new(coeffs)
}

fn scalar_var_to_ringvars<R>(
    b: &mut Dr1csBuilder<BF<R>>,
    x0: usize,
) -> RingVars
where
    R: PolyRing,
    R::BaseRing: Field,
{
    let d = R::dimension();
    let mut coeffs = Vec::with_capacity(d);
    coeffs.push(x0);
    for _ in 1..d {
        let vz = b.new_var(BF::<R>::ZERO);
        b.enforce_var_eq_const(vz, BF::<R>::ZERO);
        coeffs.push(vz);
    }
    RingVars::new(coeffs)
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

fn ring_sub<F: PrimeField>(b: &mut Dr1csBuilder<F>, x: &RingVars, y: &RingVars) -> RingVars {
    assert_eq!(x.d(), y.d());
    let mut out = Vec::with_capacity(x.d());
    for i in 0..x.d() {
        let val = b.assignment[x.coeffs[i]] - b.assignment[y.coeffs[i]];
        let v = b.new_var(val);
        b.add_constraint(
            vec![(F::ONE, x.coeffs[i]), (-F::ONE, y.coeffs[i])],
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

fn ring_mul_negacyclic<F: PrimeField>(b: &mut Dr1csBuilder<F>, x: &RingVars, y: &RingVars) -> RingVars {
    // Negacyclic convolution mod (X^d + 1).
    let d = x.d();
    assert_eq!(d, y.d());
    let one = b.one();
    let mut out = Vec::with_capacity(d);
    for k in 0..d {
        // Build coefficient via sum of products; enforce via linear constraints over fresh vars.
        let mut acc_var = b.new_var(F::ZERO);
        b.enforce_var_eq_const(acc_var, F::ZERO);

        for i in 0..d {
            // j = k - i mod d
            let j = if i <= k { k - i } else { d + k - i };
            let sign = if i <= k { F::ONE } else { -F::ONE };
            let prod_val = b.assignment[x.coeffs[i]] * b.assignment[y.coeffs[j]];
            let prod = b.new_var(prod_val);
            b.enforce_mul(x.coeffs[i], y.coeffs[j], prod);
            // acc = acc + sign * prod
            let new_acc = b.new_var(b.assignment[acc_var] + sign * b.assignment[prod]);
            b.add_constraint(
                vec![(F::ONE, acc_var), (sign, prod)],
                vec![(F::ONE, one)],
                vec![(F::ONE, new_acc)],
            );
            acc_var = new_acc;
        }
        out.push(acc_var);
    }
    RingVars::new(out)
}

fn ring_eq<F: PrimeField>(b: &mut Dr1csBuilder<F>, x: &RingVars, y: &RingVars) {
    assert_eq!(x.d(), y.d());
    for i in 0..x.d() {
        b.enforce_lc_times_one_eq_const(vec![(F::ONE, x.coeffs[i]), (-F::ONE, y.coeffs[i])]);
    }
}

fn lc_to_var<F: PrimeField>(b: &mut Dr1csBuilder<F>, lc: Vec<(F, usize)>) -> usize {
    let val = lc
        .iter()
        .fold(F::ZERO, |acc, (c, idx)| acc + (*c * b.assignment[*idx]));
    let v = b.new_var(val);
    // lc * 1 = v
    b.add_constraint(lc, vec![(F::ONE, b.one())], vec![(F::ONE, v)]);
    v
}

fn enforce_lc_eq_var<F: PrimeField>(b: &mut Dr1csBuilder<F>, lc: Vec<(F, usize)>, v: usize) {
    b.add_constraint(lc, vec![(F::ONE, b.one())], vec![(F::ONE, v)]);
}

fn enforce_bool<F: PrimeField>(b: &mut Dr1csBuilder<F>, bit: usize) {
    // bit*(bit-1)=0
    b.add_constraint(
        vec![(F::ONE, bit)],
        vec![(F::ONE, bit), (-F::ONE, b.one())],
        vec![(F::ZERO, b.one())],
    );
}

fn const_var<F: PrimeField>(b: &mut Dr1csBuilder<F>, c: F) -> usize {
    let v = b.new_var(c);
    b.enforce_var_eq_const(v, c);
    v
}

/// Arithmetize one LF+ `short_challenge` coefficient:
///   coeff = (byte % u) - (u/2), where u is a power of two.
///
/// Returns a BF var holding `coeff`.
fn short_challenge_coeff_from_byte<F: PrimeField>(
    b: &mut Dr1csBuilder<F>,
    byte: usize,
    u: u64,
) -> usize {
    debug_assert!(u.is_power_of_two());
    debug_assert!(u <= 256);
    let half = (u / 2) as i64;
    let kbits = (u as f64).log2() as usize;
    debug_assert_eq!(1u64 << kbits, u);

    // Represent byte = r + u*q with:
    // - r = Σ_{i<k} r_i 2^i  (k bits)
    // - q = Σ_{j<8-k} q_j 2^j  ((8-k) bits)
    // Witness assignment for bits is derived from the current byte value.
    let byte_val_u64 = b.assignment[byte]
        .into_bigint()
        .to_bytes_le()
        .get(0)
        .copied()
        .unwrap_or(0) as u64;
    debug_assert!(byte_val_u64 < 256);
    let r_val = (byte_val_u64 % u) as u64;
    let q_val = (byte_val_u64 / u) as u64;

    let mut r_bits = Vec::with_capacity(kbits);
    let mut q_bits = Vec::with_capacity(8 - kbits);
    for i in 0..kbits {
        let bit = (r_val >> i) & 1;
        let v = b.new_var(F::from(bit));
        enforce_bool(b, v);
        r_bits.push(v);
    }
    for j in 0..(8 - kbits) {
        let bit = (q_val >> j) & 1;
        let v = b.new_var(F::from(bit));
        enforce_bool(b, v);
        q_bits.push(v);
    }

    let mut lc = vec![(-F::ONE, byte)];
    // r part
    for (i, &bi) in r_bits.iter().enumerate() {
        lc.push((F::from(1u64 << i), bi));
    }
    // q part
    for (j, &bj) in q_bits.iter().enumerate() {
        let coeff = u * (1u64 << j);
        lc.push((F::from(coeff), bj));
    }
    // Enforce: -byte + r + u*q == 0  => (-byte + ...)*1 = 0
    b.enforce_lc_times_one_eq_const(lc);

    // r = Σ r_i 2^i
    let mut r_lc = Vec::with_capacity(1 + r_bits.len());
    for (i, &bi) in r_bits.iter().enumerate() {
        r_lc.push((F::from(1u64 << i), bi));
    }
    let r = lc_to_var(b, r_lc);

    // coeff = r - half
    let _half = half; // keep for clarity: half = u/2, always positive here
    let coeff_val = b.assignment[r] - F::from((u / 2) as u64);
    let out = b.new_var(coeff_val);
    enforce_lc_eq_var(b, vec![(F::ONE, r), (-F::from((u / 2) as u64), b.one())], out);
    out
}

fn short_challenge_from_bytes<F: PrimeField>(
    b: &mut Dr1csBuilder<F>,
    bytes: &[usize],
    lambda: usize,
    ring_dim: usize,
) -> RingVars {
    debug_assert_eq!(bytes.len(), ring_dim);
    // Matches `utils::short_challenge`: u = 2^(lambda / d).
    let exp = (lambda / ring_dim) as u32;
    let u = 1u64 << exp;
    let mut coeffs = Vec::with_capacity(ring_dim);
    for &by in bytes {
        let c = short_challenge_coeff_from_byte::<F>(b, by, u);
        coeffs.push(c);
    }
    RingVars::new(coeffs)
}

#[derive(Clone, Debug)]
pub struct CmShortChallengeWiring {
    /// Byte variables (one per squeezed byte), in order.
    pub byte_vars: Vec<usize>,
    /// `s[0..3]` short challenges (ring elements as coefficient vars).
    pub s: Vec<RingVars>,
    /// Flattened `s_prime` of length `k*d` (ring elements as coefficient vars).
    pub s_prime_flat: Vec<RingVars>,
}

/// Build a dR1CS part that reconstructs all LF+ `short_challenge(128)` ring elements
/// from Poseidon `SqueezeBytes` outputs (bytes -> coefficients).
///
/// Assumption (holds for current LF+ verifier paths): the only `squeeze_bytes` calls in the
/// verifier transcript are from `short_challenge(128)` within `CmProof::verify`, and each call
/// squeezes exactly `R::dimension()` bytes.
fn cm_short_challenges_dr1cs<R>(
    trace: &PoseidonTranscriptTrace<BF<R>>,
    k: usize,
) -> Result<(SparseDr1csInstance<BF<R>>, Vec<BF<R>>, CmShortChallengeWiring), String>
where
    R: PolyRing,
    R::BaseRing: Field,
{
    let d = R::dimension();
    let lambda = 128usize;

    // Expected total number of short challenges consumed by CmProof::verify:
    // - s: 3
    // - s_prime: k*d
    let need = 3 + k * d;
    let need_bytes = need * d;
    if trace.squeezed_bytes.len() < need_bytes {
        return Err(format!(
            "cm_short_challenges_dr1cs: not enough squeezed bytes: need {}, got {}",
            need_bytes,
            trace.squeezed_bytes.len()
        ));
    }

    let mut b = Dr1csBuilder::<BF<R>>::new();
    b.enforce_var_eq_const(b.one(), BF::<R>::ONE);

    // Allocate byte vars (as field elements) for the needed bytes.
    let mut byte_vars = Vec::with_capacity(need_bytes);
    for &by in trace.squeezed_bytes.iter().take(need_bytes) {
        let v = const_var(&mut b, BF::<R>::from(by as u64));
        byte_vars.push(v);
    }

    // Reconstruct ring elements, chunking by d bytes.
    let mut rings: Vec<RingVars> = Vec::with_capacity(need);
    for i in 0..need {
        let start = i * d;
        let end = start + d;
        let rv = short_challenge_from_bytes::<BF<R>>(&mut b, &byte_vars[start..end], lambda, d);
        rings.push(rv);
    }

    let s = rings[0..3].to_vec();
    let s_prime_flat = rings[3..].to_vec();
    debug_assert_eq!(s_prime_flat.len(), k * d);

    let (inst, asg) = b.into_instance();
    Ok((
        inst,
        asg,
        CmShortChallengeWiring {
            byte_vars,
            s,
            s_prime_flat,
        },
    ))
}

/// Evaluate eq(c, r) where both are vectors of scalar (BF) variables.
fn eq_eval_vars<F: PrimeField>(b: &mut Dr1csBuilder<F>, c: &[usize], r: &[usize]) -> usize {
    assert_eq!(c.len(), r.len());
    let mut acc = b.new_var(F::ONE);
    b.enforce_var_eq_const(acc, F::ONE);
    for (&ci, &ri) in c.iter().zip(r.iter()) {
        let one = b.one();
        let one_minus_ci = b.new_var(F::ONE - b.assignment[ci]);
        b.add_constraint(vec![(F::ONE, one), (-F::ONE, ci)], vec![(F::ONE, one)], vec![(F::ONE, one_minus_ci)]);
        let one_minus_ri = b.new_var(F::ONE - b.assignment[ri]);
        b.add_constraint(vec![(F::ONE, one), (-F::ONE, ri)], vec![(F::ONE, one)], vec![(F::ONE, one_minus_ri)]);
        let ci_ri = b.new_var(b.assignment[ci] * b.assignment[ri]);
        b.enforce_mul(ci, ri, ci_ri);
        let om = b.new_var(b.assignment[one_minus_ci] * b.assignment[one_minus_ri]);
        b.enforce_mul(one_minus_ci, one_minus_ri, om);
        let t = b.new_var(b.assignment[ci_ri] + b.assignment[om]);
        b.add_constraint(vec![(F::ONE, ci_ri), (F::ONE, om)], vec![(F::ONE, one)], vec![(F::ONE, t)]);
        let new_acc = b.new_var(b.assignment[acc] * b.assignment[t]);
        b.enforce_mul(acc, t, new_acc);
        acc = new_acc;
    }
    acc
}

struct ChallengeCursor<'a, F: PrimeField> {
    vals: &'a [F],
    idx: usize,
    vars: Vec<usize>,
}

impl<'a, F: PrimeField> ChallengeCursor<'a, F> {
    fn new(vals: &'a [F]) -> Self {
        Self {
            vals,
            idx: 0,
            vars: Vec::new(),
        }
    }

    fn next(&mut self, b: &mut Dr1csBuilder<F>) -> usize {
        let v = self
            .vals
            .get(self.idx)
            .copied()
            .unwrap_or_else(|| panic!("challenge cursor oob at {}", self.idx));
        self.idx += 1;
        let var = b.new_var(v);
        self.vars.push(var);
        var
    }

    fn consumed(&self) -> usize {
        self.idx
    }

    fn all_vars(&self) -> &[usize] {
        &self.vars
    }
}

fn comr1cs_verifier_math_dr1cs<R>(
    proof: &crate::r1cs::ComR1CSProof<R>,
    ch: &mut ChallengeCursor<BF<R>>,
) -> Result<(SparseDr1csInstance<BF<R>>, Vec<BF<R>>, Vec<usize>), String>
where
    R: OverField + PolyRing,
    R::BaseRing: Field,
{
    use latticefold::utils::sumcheck::Proof as ScProof;

    let mut b = Dr1csBuilder::<BF<R>>::new();
    b.enforce_var_eq_const(b.one(), BF::<R>::ONE);

    let nvars = proof.nvars;

    // r = transcript.get_challenges(nvars)
    let mut r_pre = Vec::with_capacity(nvars);
    for _ in 0..nvars {
        r_pre.push(ch.next(&mut b));
    }

    // Sumcheck verifier challenges (one per round)
    let mut r_sc = Vec::with_capacity(nvars);
    for _ in 0..nvars {
        r_sc.push(ch.next(&mut b));
    }

    // Sumcheck prover messages: per-round 4 ring elements.
    let msgs: &ScProof<R> = &proof.sumcheck_proof;
    if msgs.msgs().len() != nvars {
        return Err("ComR1CSProof: sumcheck proof length mismatch".to_string());
    }
    let mut msg_vars: Vec<[RingVars; 4]> = Vec::with_capacity(nvars);
    for m in msgs.msgs() {
        if m.evaluations.len() != 4 {
            return Err("ComR1CSProof: expected degree-3 evals (len=4)".to_string());
        }
        let e0 = ring_to_ringvars::<R>(&mut b, &m.evaluations[0]);
        let e1 = ring_to_ringvars::<R>(&mut b, &m.evaluations[1]);
        let e2 = ring_to_ringvars::<R>(&mut b, &m.evaluations[2]);
        let e3 = ring_to_ringvars::<R>(&mut b, &m.evaluations[3]);
        msg_vars.push([e0, e1, e2, e3]);
    }

    // Verify sumcheck with claimed sum = 0.
    let claimed_sum = scalar_to_ringvars::<R>(&mut b, BF::<R>::ZERO);
    let subclaim_eval = sumcheck_verify_degree3::<BF<R>>(&mut b, claimed_sum, &msg_vars, &r_sc)?;

    // Allocate evals absorbed by transcript (we need them for arithmetic check).
    let va = ring_to_ringvars::<R>(&mut b, &proof.va);
    let vb = ring_to_ringvars::<R>(&mut b, &proof.vb);
    let vc = ring_to_ringvars::<R>(&mut b, &proof.vc);

    // e = eq_eval(r_pre, r_sc) (scalar), lifted to a constant-coeff ring element.
    let e = eq_eval_vars::<BF<R>>(&mut b, &r_pre, &r_sc);
    let e_ring = scalar_var_to_ringvars::<R>(&mut b, e);

    // Enforce: e * (va*vb - vc) == subclaim_eval
    let vab = ring_mul_negacyclic::<BF<R>>(&mut b, &va, &vb);
    let diff = ring_sub::<BF<R>>(&mut b, &vab, &vc);
    let lhs = ring_mul_negacyclic::<BF<R>>(&mut b, &e_ring, &diff);
    ring_eq::<BF<R>>(&mut b, &lhs, &subclaim_eval);

    let (inst, asg) = b.into_instance();
    Ok((inst, asg, ch.all_vars().to_vec()))
}

fn scalar_sub_const<F: PrimeField>(b: &mut Dr1csBuilder<F>, r: usize, c: F) -> usize {
    let val = b.assignment[r] - c;
    let v = b.new_var(val);
    // v = r - c
    b.add_constraint(
        vec![(F::ONE, r), (-c, b.one())],
        vec![(F::ONE, b.one())],
        vec![(F::ONE, v)],
    );
    v
}

fn scalar_mul<F: PrimeField>(b: &mut Dr1csBuilder<F>, x: usize, y: usize) -> usize {
    let val = b.assignment[x] * b.assignment[y];
    let v = b.new_var(val);
    b.enforce_mul(x, y, v);
    v
}

fn lagrange_degree2<F: PrimeField>(b: &mut Dr1csBuilder<F>, r: usize) -> (usize, usize, usize) {
    let inv2 = F::from(2u64).inverse().unwrap();

    // t1=r-1, t2=r-2
    let t1 = scalar_sub_const(b, r, F::ONE);
    let t2 = scalar_sub_const(b, r, F::from(2u64));

    // L0 = (r-1)(r-2)/2
    let p = scalar_mul(b, t1, t2);
    let l0 = b.new_var(b.assignment[p] * inv2);
    b.add_constraint(vec![(inv2, p)], vec![(F::ONE, b.one())], vec![(F::ONE, l0)]);

    // L1 = -r(r-2)
    let p = scalar_mul(b, r, t2);
    let l1 = b.new_var(-b.assignment[p]);
    b.add_constraint(vec![(-F::ONE, p)], vec![(F::ONE, b.one())], vec![(F::ONE, l1)]);

    // L2 = r(r-1)/2
    let p = scalar_mul(b, r, t1);
    let l2 = b.new_var(b.assignment[p] * inv2);
    b.add_constraint(vec![(inv2, p)], vec![(F::ONE, b.one())], vec![(F::ONE, l2)]);

    (l0, l1, l2)
}

pub fn sumcheck_verify_degree2<F: PrimeField>(
    b: &mut Dr1csBuilder<F>,
    mut claimed_sum: RingVars,
    msgs: &[[RingVars; 3]],
    rs: &[usize],
) -> Result<RingVars, String> {
    if msgs.len() != rs.len() {
        return Err("sumcheck_verify_degree2: msgs/rs length mismatch".to_string());
    }
    for (m, &r) in msgs.iter().zip(rs.iter()) {
        // Check g(0)+g(1) == claimed_sum (coefficient-wise).
        let g01 = ring_add(b, &m[0], &m[1]);
        ring_eq(b, &g01, &claimed_sum);

        // Update claim = g(r) by Lagrange interpolation on points 0,1,2.
        let (l0, l1, l2) = lagrange_degree2::<F>(b, r);
        let t0 = ring_scale(b, &m[0], l0);
        let t1 = ring_scale(b, &m[1], l1);
        let t2 = ring_scale(b, &m[2], l2);
        let s01 = ring_add(b, &t0, &t1);
        claimed_sum = ring_add(b, &s01, &t2);
    }
    Ok(claimed_sum)
}

/// WE-gate arithmetization for verifying one `ComR1CSProof` (the Π_lin proof).
///
/// This is a first “end-to-end inside WE” building block: it includes
/// - Poseidon transcript arithmetization
/// - FS re-absorb consistency constraints
/// - statement-bound params prefix (public inputs)
/// - Π_lin verifier arithmetic constraints
/// - glue constraints equating Π_lin challenge variables with Poseidon squeeze outputs
///
/// NOTE: This currently only covers the `ComR1CSProof` verifier path (see `r1cs.rs`).
#[cfg(feature = "we_gate")]
pub fn build_we_dr1cs_for_comr1cs_proof<R>(
    poseidon_cfg: &PoseidonConfig<BF<R>>,
    trace: &PoseidonTranscriptTrace<BF<R>>,
    params: &WeParams,
    proof: &crate::r1cs::ComR1CSProof<R>,
) -> Result<WeDr1csOutput<BF<R>>, String>
where
    R: OverField + CoeffRing + PolyRing,
    R::BaseRing: Zq + Field,
{
    // Poseidon trace -> dR1CS
    let ops = lf_ops_to_symphony_ops::<BF<R>>(&trace.ops);
    let (mut pose_inst, pose_asg, _replay, _byte_wit, wiring, _byte_wiring) =
        poseidon_sponge_dr1cs_from_trace_with_wiring_and_bytes::<BF<R>>(poseidon_cfg, &ops)
            .map_err(|e| format!("poseidon arith failed: {e}"))?;
    enforce_reabsorb_equals_squeeze::<BF<R>>(&mut pose_inst, &wiring, &ops)?;

    // Public statement params prefix (no constraints fixing their value).
    let mut b_params = Dr1csBuilder::<BF<R>>::new();
    b_params.enforce_var_eq_const(b_params.one(), BF::<R>::ONE);
    for &x in &params.to_field_vec::<BF<R>>() {
        let _ = b_params.new_var(x);
    }
    let (params_inst, params_asg) = b_params.into_instance();

    // Π_lin verifier arithmetic.
    let mut ch = ChallengeCursor::<BF<R>>::new(&trace.squeezed_field);
    let (lin_inst, lin_asg, lin_ch_vars) = comr1cs_verifier_math_dr1cs::<R>(proof, &mut ch)?;

    // Glue: each challenge var equals corresponding Poseidon squeeze-field var.
    if wiring.squeeze_field_vars.len() < lin_ch_vars.len() {
        return Err("poseidon wiring: not enough squeeze_field_vars for lin challenges".to_string());
    }
    let mut glue: Vec<(usize, usize, usize, usize)> = Vec::with_capacity(lin_ch_vars.len());
    for (i, &v_lin) in lin_ch_vars.iter().enumerate() {
        let v_pose = wiring.squeeze_field_vars[i];
        glue.push((0, v_pose, 2, v_lin));
    }

    let parts = vec![(pose_inst, pose_asg), (params_inst, params_asg), (lin_inst, lin_asg)];
    let (inst, assignment) =
        merge_sparse_dr1cs_share_one_with_glue(&parts, &glue).map_err(|e| e.to_string())?;

    // Public prefix: [1] + params (fixed 9 scalars)
    let public_len = 1 + 9;
    Ok(WeDr1csOutput {
        inst,
        assignment,
        public_len,
    })
}

/// WE-gate arithmetization for the **short-challenge derivation** portion of `CmProof::verify`.
///
/// Builds:
/// - Poseidon transcript arithmetization (+ byte wiring)
/// - constraints that reconstruct `short_challenge(128)` ring elements from those bytes
/// - glue constraints equating byte variables across the two parts
///
/// Returns both the merged dR1CS output and the wiring for `s` and `s_prime`.
#[cfg(feature = "we_gate")]
pub fn build_we_dr1cs_for_cm_short_challenges<R>(
    poseidon_cfg: &PoseidonConfig<BF<R>>,
    trace: &PoseidonTranscriptTrace<BF<R>>,
    params: &WeParams,
    k: usize,
) -> Result<(WeDr1csOutput<BF<R>>, CmShortChallengeWiring), String>
where
    R: OverField + CoeffRing + PolyRing,
    R::BaseRing: Zq + Field,
{
    // Poseidon trace -> dR1CS (+ wiring with squeeze-field + squeeze-byte var indices).
    let ops = lf_ops_to_symphony_ops::<BF<R>>(&trace.ops);
    let (mut pose_inst, pose_asg, _replay, _byte_wit, wiring, byte_wiring) =
        poseidon_sponge_dr1cs_from_trace_with_wiring_and_bytes::<BF<R>>(poseidon_cfg, &ops)
            .map_err(|e| format!("poseidon arith failed: {e}"))?;
    enforce_reabsorb_equals_squeeze::<BF<R>>(&mut pose_inst, &wiring, &ops)?;

    // Public statement params prefix (no constraints fixing their value).
    let mut b_params = Dr1csBuilder::<BF<R>>::new();
    b_params.enforce_var_eq_const(b_params.one(), BF::<R>::ONE);
    for &x in &params.to_field_vec::<BF<R>>() {
        let _ = b_params.new_var(x);
    }
    let (params_inst, params_asg) = b_params.into_instance();

    // Short-challenge reconstruction part (allocates its own byte vars; we glue them).
    let (coin_inst, coin_asg, coin_wiring) = cm_short_challenges_dr1cs::<R>(trace, k)?;

    // Glue all squeezed bytes in order.
    if byte_wiring.squeeze_byte_vars.len() < coin_wiring.byte_vars.len() {
        return Err("poseidon byte wiring: not enough squeeze_byte_vars".to_string());
    }
    let mut glue: Vec<(usize, usize, usize, usize)> = Vec::with_capacity(coin_wiring.byte_vars.len());
    for i in 0..coin_wiring.byte_vars.len() {
        glue.push((0, byte_wiring.squeeze_byte_vars[i], 2, coin_wiring.byte_vars[i]));
    }

    let parts = vec![(pose_inst, pose_asg), (params_inst, params_asg), (coin_inst, coin_asg)];
    let (inst, assignment) =
        merge_sparse_dr1cs_share_one_with_glue(&parts, &glue).map_err(|e| e.to_string())?;

    let public_len = 1 + 9;
    Ok((WeDr1csOutput { inst, assignment, public_len }, coin_wiring))
}

#[cfg(all(test, feature = "we_gate"))]
mod tests {
    use super::*;
    use cyclotomic_rings::rings::GoldilocksPoseidonConfig as PC;
    use latticefold::arith::r1cs::R1CS;
    use stark_rings::balanced_decomposition::GadgetDecompose;
    use stark_rings::cyclotomic_ring::models::goldilocks::RqPoly as R;
    use stark_rings::Ring;
    use stark_rings_linalg::{Matrix, SparseMatrix};

    use crate::lin::Linearize;
    use crate::lin::LinearizedVerify;
    use crate::r1cs::ComR1CS;
    use crate::recording_transcript::TracePoseidonTranscript;

    fn identity_cs(n: usize) -> (R1CS<R>, Vec<R>) {
        let r1cs = R1CS::<R> {
            l: 1,
            A: SparseMatrix::identity(n),
            B: SparseMatrix::identity(n),
            C: SparseMatrix::identity(n),
        };
        let z = vec![R::ONE; n];
        (r1cs, z)
    }

    #[test]
    fn test_we_arith_linearized_verify_constraints_satisfy() {
        let n = 1 << 7;
        let k = 4;
        let m = n / k;
        let b = 2;
        let kappa = 2;
        let (mut r1cs, z) = identity_cs(m);
        r1cs.A = r1cs.A.gadget_decompose(b, k);
        r1cs.B = r1cs.B.gadget_decompose(b, k);
        r1cs.C = r1cs.C.gadget_decompose(b, k);

        let A = Matrix::<R>::rand(&mut ark_std::test_rng(), kappa, n);
        let cr1cs = ComR1CS::new(r1cs, z, 1, b, k, &A);

        // Build proof.
        let mut ts = crate::transcript::PoseidonTranscript::empty::<PC>();
        let (_linb, lproof) = cr1cs.linearize(&mut ts);

        // Record the verifier transcript coin stream.
        let mut rec = TracePoseidonTranscript::<R>::empty::<PC>();
        assert!(lproof.verify(&mut rec));
        let trace = rec.trace().clone();

        // Build verifier-math dR1CS and check satisfaction.
        let mut ch = ChallengeCursor::<BF<R>>::new(&trace.squeezed_field);
        let (inst, asg, _ch_vars) = comr1cs_verifier_math_dr1cs::<R>(&lproof, &mut ch).unwrap();
        inst.check(&asg).unwrap();

        // Sanity: we consumed exactly 2*nvars squeeze-field scalars (r_pre and sumcheck rs).
        assert_eq!(ch.consumed(), 2 * lproof.nvars);
    }

    #[test]
    fn test_we_arith_sumcheck_degree2_constraints_satisfy() {
        use latticefold::utils::sumcheck::MLSumcheck;
        use stark_rings_poly::mle::DenseMultilinearExtension;
        use ark_std::UniformRand;

        // Small instance: sumcheck of product of two random MLEs (degree=2).
        let nvars = 6usize;
        let n = 1usize << nvars;
        let mut rng = ark_std::test_rng();
        let evals0 = (0..n).map(|_| R::rand(&mut rng)).collect::<Vec<_>>();
        let evals1 = (0..n).map(|_| R::rand(&mut rng)).collect::<Vec<_>>();
        let mle0 = DenseMultilinearExtension::from_evaluations_vec(nvars, evals0);
        let mle1 = DenseMultilinearExtension::from_evaluations_vec(nvars, evals1);

        let mut ts_p = crate::transcript::PoseidonTranscript::empty::<PC>();
        let (proof, _state) = MLSumcheck::<R, _>::prove_as_subprotocol(
            &mut ts_p,
            vec![mle0, mle1],
            nvars,
            2,
            |vals: &[R]| vals[0] * vals[1],
        );
        // Equivalent to MLSumcheck::extract_sum (avoid transcript type parameter inference).
        let claimed_sum = proof.msgs()[0].evaluations[0] + proof.msgs()[0].evaluations[1];

        // Run verifier to get the real transcript coin stream (r_i).
        let mut rec = crate::recording_transcript::TracePoseidonTranscript::<R>::empty::<PC>();
        let _sub = MLSumcheck::<R, _>::verify_as_subprotocol(
            &mut rec,
            nvars,
            2,
            claimed_sum,
            &proof,
        )
        .unwrap();
        let trace = rec.trace().clone();

        // Build dR1CS for sumcheck verify (standalone, with challenges from trace.squeezed_field).
        type F = BF<R>;
        let mut ch = ChallengeCursor::<F>::new(&trace.squeezed_field);
        let mut b = Dr1csBuilder::<F>::new();
        b.enforce_var_eq_const(b.one(), F::from(1u64));

        // Allocate sumcheck prover msgs (3 evals per round for degree 2).
        let msgs = proof.msgs().to_vec();
        assert_eq!(msgs.len(), nvars);
        let mut msg_vars: Vec<[RingVars; 3]> = Vec::with_capacity(nvars);
        for m in msgs {
            assert_eq!(m.evaluations.len(), 3);
            let e0 = ring_to_ringvars::<R>(&mut b, &m.evaluations[0]);
            let e1 = ring_to_ringvars::<R>(&mut b, &m.evaluations[1]);
            let e2 = ring_to_ringvars::<R>(&mut b, &m.evaluations[2]);
            msg_vars.push([e0, e1, e2]);
        }

        // Sample r_i from trace (matches verify_as_subprotocol schedule).
        let mut r_sc = Vec::with_capacity(nvars);
        for _ in 0..nvars {
            r_sc.push(ch.next(&mut b));
        }

        let claim0 = ring_to_ringvars::<R>(&mut b, &claimed_sum);
        let _final_claim = sumcheck_verify_degree2::<F>(&mut b, claim0, &msg_vars, &r_sc).unwrap();

        let (inst, asg) = b.into_instance();
        inst.check(&asg).unwrap();
    }

    #[test]
    fn test_short_challenge_coeff_from_byte_matches_rust() {
        type F = BF<R>;
        let u = 32u64; // typical for lambda=128,d=24 => floor(128/24)=5 => u=32

        for byte_u8 in [0u8, 1, 2, 15, 16, 31, 32, 33, 63, 64, 127, 128, 200, 255] {
            let mut b = Dr1csBuilder::<F>::new();
            b.enforce_var_eq_const(b.one(), F::from(1u64));
            let byte = const_var(&mut b, F::from(byte_u8 as u64));
            let coeff = short_challenge_coeff_from_byte::<F>(&mut b, byte, u);

            let expected_i64 = ((byte_u8 as u64) % u) as i64 - (u as i64 / 2);
            let expected = if expected_i64 >= 0 {
                F::from(expected_i64 as u64)
            } else {
                -F::from((-expected_i64) as u64)
            };

            assert_eq!(b.assignment[coeff], expected);
            let (inst, asg) = b.into_instance();
            inst.check(&asg).unwrap();
        }
    }

    #[test]
    fn test_short_challenge_from_bytes_matches_rust() {
        use crate::utils::short_challenge;

        type F = BF<R>;
        let d = R::dimension();

        // Get bytes from the real transcript by calling `short_challenge`.
        let mut rec = crate::recording_transcript::TracePoseidonTranscript::<R>::empty::<PC>();
        let r_sc = short_challenge::<R>(128, &mut rec);
        let bytes = match rec.trace().ops.last().unwrap() {
            crate::recording_transcript::PoseidonTraceOp::SqueezeBytes { out, .. } => out.clone(),
            _ => panic!("expected last op to be SqueezeBytes"),
        };
        assert_eq!(bytes.len(), d);

        // Arithmetize from byte vars.
        let mut b = Dr1csBuilder::<F>::new();
        b.enforce_var_eq_const(b.one(), F::from(1u64));
        let byte_vars = bytes
            .iter()
            .map(|&by| const_var(&mut b, F::from(by as u64)))
            .collect::<Vec<_>>();
        let ring = short_challenge_from_bytes::<F>(&mut b, &byte_vars, 128, d);

        // Compare coefficients.
        let expected_coeffs = r_sc.coeffs().to_vec();
        assert_eq!(expected_coeffs.len(), d);
        for i in 0..d {
            let exp_bf = expected_coeffs[i]
                .to_base_prime_field_elements()
                .into_iter()
                .next()
                .unwrap();
            assert_eq!(b.assignment[ring.coeffs[i]], exp_bf);
        }

        let (inst, asg) = b.into_instance();
        inst.check(&asg).unwrap();
    }
}

