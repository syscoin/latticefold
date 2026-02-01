use ark_crypto_primitives::sponge::poseidon::PoseidonConfig;

use latticefold::transcript::poseidon::{f257_poseidon_config, F257};
use symphony::dpp_poseidon::{PoseidonByteWiring, PoseidonDr1csWiring};
use symphony::transcript::PoseidonTraceOp;

use symphony::dpp_poseidon::poseidon_sponge_dr1cs_from_ops_with_wiring_and_bytes_file_backed;
use symphony::file_backed_dr1cs::FileBackedSparseDr1csInstance;

/// Arithmetize a Poseidon(F257) transcript trace into a file-backed sparse dR1CS instance +
/// satisfying assignment, and return wiring needed to locate absorbed/squeezed variables (and
/// `SqueezeBytes` output).
pub fn poseidon_f257_arithmetize(
    cfg: Option<&PoseidonConfig<F257>>,
    ops: &[PoseidonTraceOp<F257>],
    out_dir: impl AsRef<std::path::Path>,
) -> Result<
    (
        FileBackedSparseDr1csInstance<F257>,
        Vec<F257>,
        PoseidonDr1csWiring,
        PoseidonByteWiring,
    ),
    String,
> {
    let default_cfg;
    let cfg = match cfg {
        Some(c) => c,
        None => {
            default_cfg = f257_poseidon_config();
            &default_cfg
        }
    };
    let (inst, asg, _replay, _byte_wit, wiring, byte_wiring) =
        poseidon_sponge_dr1cs_from_ops_with_wiring_and_bytes_file_backed::<F257>(cfg, ops, out_dir)
            .map_err(|e| format!("poseidon(F257) file-backed arith failed: {e}"))?;
    debug_assert_eq!(inst.nvars, asg.len());
    Ok((inst, asg, wiring, byte_wiring))
}

