use ark_ff::{BigInteger, PrimeField};

use latticefold::transcript::poseidon::F257;
use symphony::transcript::PoseidonTraceOp;

/// Convert an LF+ recorded transcript trace (stored over an arbitrary base prime field `BF`) into
/// an equivalent op list over `F257`, by interpreting each element as a small integer.
///
/// Requirements:
/// - each traced element must be an integer in `[0,256]` (as used by the F257 byte/digit surface)
pub fn lift_recording_trace_ops_to_f257<BF: PrimeField>(
    ops: &[crate::recording_transcript::PoseidonTraceOp<BF>],
) -> Result<Vec<PoseidonTraceOp<F257>>, String> {
    fn bf_to_u16<BF: PrimeField>(x: &BF) -> u16 {
        let bytes = x.into_bigint().to_bytes_le();
        (bytes.get(0).copied().unwrap_or(0) as u16) | ((bytes.get(1).copied().unwrap_or(0) as u16) << 8)
    }
    let mut out: Vec<PoseidonTraceOp<F257>> = Vec::with_capacity(ops.len());
    for op in ops {
        match op {
            crate::recording_transcript::PoseidonTraceOp::Absorb(v) => {
                let mut vv = Vec::with_capacity(v.len());
                for e in v {
                    let d = bf_to_u16::<BF>(e);
                    if d > 256 {
                        return Err(format!("trace element out of range: {d}"));
                    }
                    vv.push(F257::from(d as u64));
                }
                out.push(PoseidonTraceOp::Absorb(vv));
            }
            crate::recording_transcript::PoseidonTraceOp::SqueezeField(v) => {
                let mut vv = Vec::with_capacity(v.len());
                for e in v {
                    let d = bf_to_u16::<BF>(e);
                    if d > 256 {
                        return Err(format!("trace element out of range: {d}"));
                    }
                    vv.push(F257::from(d as u64));
                }
                out.push(PoseidonTraceOp::SqueezeField(vv));
            }
            crate::recording_transcript::PoseidonTraceOp::SqueezeBytes { n, out: bytes } => {
                // Legacy traces may include this op; map it directly.
                out.push(PoseidonTraceOp::SqueezeBytes { n: *n, out: bytes.clone() });
            }
        }
    }
    Ok(out)
}

