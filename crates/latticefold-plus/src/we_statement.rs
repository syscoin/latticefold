use ark_ff::PrimeField;

/// Fixed parameters that must be statement-bound for WE/DPP.
///
/// These are implicit in native Rust (loop bounds, type params), but must be explicit in an
/// arithmetized WE gate to prevent “reinterpretation under different sizes”.
#[derive(Clone, Debug)]
pub struct WeParams {
    pub nvars_setchk: u64,
    pub degree_setchk: u64,
    pub nvars_cm: u64,
    pub degree_cm: u64,
    pub kappa: u64,
    pub ring_dim_d: u64,
    pub k: u64,
    pub l: u64,
    pub mlen: u64,
}

impl WeParams {
    pub fn to_field_vec<BF: PrimeField>(&self) -> Vec<BF> {
        vec![
            BF::from(self.nvars_setchk),
            BF::from(self.degree_setchk),
            BF::from(self.nvars_cm),
            BF::from(self.degree_cm),
            BF::from(self.kappa),
            BF::from(self.ring_dim_d),
            BF::from(self.k),
            BF::from(self.l),
            BF::from(self.mlen),
        ]
    }
}

/// Deterministic statement encoding for WE/DPP.
///
/// Current encoding:
/// - `x[0] = 1` (shared constant slot convention)
/// - fixed params (see `WeParams`)
/// - optional extra statement elements (e.g. commitment surface limbs, transcript-bound absorbs)
///
/// NOTE: The exact set of extra statement elements is decided by the WE arithmetizer; this module
/// just provides a stable prefix layout for params.
pub fn encode_public_x<BF: PrimeField>(params: &WeParams, extra: &[BF]) -> Vec<BF> {
    let mut out = Vec::with_capacity(1 + 9 + extra.len());
    out.push(BF::ONE);
    out.extend(params.to_field_vec::<BF>());
    out.extend_from_slice(extra);
    out
}

