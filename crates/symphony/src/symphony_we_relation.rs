//! WE/DPP-facing relation APIs for Symphony.
//!
//! Architecture-T direction: DPP should target the *relations directly*, not a verifier-of-a-proof.
//! Concretely, we expose:
//! - `R_cp`: folding transcript / tie relation (checked by running the folding verifier logic)
//! - `R_o`: reduced relation on the folded output instance
//! - `R_WE := R_cp ∧ R_o`: conjunction gate for WE/DPP.

use stark_rings::{balanced_decomposition::Decompose, CoeffRing, Zq};
use stark_rings_linalg::SparseMatrix;

use crate::{
    symphony_fold::{SymphonyBatchLin, SymphonyInstance},
    symphony_open::VfyOpen,
    symphony_pifold_batched::{
        verify_pi_fold_batched_and_fold_outputs_poseidon_fs_cp, PiFoldAuxWitness, PiFoldBatchedProof,
    },
};

/// Public output of the folding layer (the reduced instance).
#[derive(Clone, Debug)]
pub struct FoldedOutput<R: CoeffRing>
where
    R::BaseRing: Zq,
{
    pub folded_inst: SymphonyInstance<R>,
    pub folded_bat: SymphonyBatchLin<R>,
}

/// Reduced relation `R_o` checker (application-specific).
///
/// In the Symphony paper, `R_o` is the reduced relation after folding.
/// For WE/DPP, we avoid verifying an external SNARK proof and instead check the
/// underlying reduced relation directly (with an explicit reduced witness).
pub trait ReducedRelation<R: CoeffRing>
where
    R::BaseRing: Zq,
{
    type Witness;

    fn check(public: &FoldedOutput<R>, witness: &Self::Witness) -> Result<(), String>;
}

/// `R_cp` check: verify the folding proof under Poseidon-FS (Poseidon transcript),
/// including commitment-opening checks via `open`, and return the folded output.
pub fn check_r_cp_poseidon_fs<R: CoeffRing, PC>(
    M: [&SparseMatrix<R>; 3],
    cm_f: &[Vec<R>],
    proof: &PiFoldBatchedProof<R>,
    open: &impl VfyOpen<R>,
    cfs_had_u: &[Vec<R>],
    cfs_mon_b: &[Vec<R>],
    aux: &PiFoldAuxWitness<R>,
    public_inputs: &[R::BaseRing],
) -> Result<FoldedOutput<R>, String>
where
    R::BaseRing: Zq + Decompose,
    PC: cyclotomic_rings::rings::GetPoseidonParams<<<R>::BaseRing as ark_ff::Field>::BasePrimeField>,
{
    let (folded_inst, folded_bat) =
        verify_pi_fold_batched_and_fold_outputs_poseidon_fs_cp::<R, PC>(
            M,
            cm_f,
            proof,
            open,
            cfs_had_u,
            cfs_mon_b,
            aux,
            public_inputs,
        )?;
    Ok(FoldedOutput { folded_inst, folded_bat })
}

/// `R_WE := R_cp ∧ R_o` check: verify folding (R_cp) and then verify the reduced relation (R_o).
pub fn check_r_we_poseidon_fs<R: CoeffRing, PC, RO: ReducedRelation<R>>(
    M: [&SparseMatrix<R>; 3],
    cm_f: &[Vec<R>],
    proof: &PiFoldBatchedProof<R>,
    open: &impl VfyOpen<R>,
    cfs_had_u: &[Vec<R>],
    cfs_mon_b: &[Vec<R>],
    aux: &PiFoldAuxWitness<R>,
    public_inputs: &[R::BaseRing],
    ro_witness: &RO::Witness,
) -> Result<(), String>
where
    R::BaseRing: Zq + Decompose,
    PC: cyclotomic_rings::rings::GetPoseidonParams<<<R>::BaseRing as ark_ff::Field>::BasePrimeField>,
{
    let out = check_r_cp_poseidon_fs::<R, PC>(
        M,
        cm_f,
        proof,
        open,
        cfs_had_u,
        cfs_mon_b,
        aux,
        public_inputs,
    )?;
    RO::check(&out, ro_witness)?;
    Ok(())
}

/// Trivial reduced relation for testing / scaffolding.
pub struct TrivialRo;

impl<R: CoeffRing> ReducedRelation<R> for TrivialRo
where
    R::BaseRing: Zq,
{
    type Witness = ();

    fn check(_public: &FoldedOutput<R>, _witness: &Self::Witness) -> Result<(), String> {
        Ok(())
    }
}

