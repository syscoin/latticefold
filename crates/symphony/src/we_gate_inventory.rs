//! Inventory of constraints/messages for the WE/DPP gate (`R_WE`).
//!
//! This module is intentionally **descriptive**: it answers “what must the arithmetized DPP gate
//! enforce?” and “what is already covered by our current arithmetization scaffolding?”.
//!
//! It does **not** implement the full production gate yet; rather it provides a canonical checklist
//! so we stop losing track across refactors.

use stark_rings::{OverField, PolyRing, Zq};

use crate::symphony_pifold_batched::{PiFoldAuxWitness, PiFoldBatchedProof};

#[derive(Clone, Debug)]
pub struct WeGateInventory {
    /// Batch size ℓ (e.g. SP1 chunks).
    pub ell: usize,
    /// Ring dimension d.
    pub d: usize,
    /// Π_rg digit count k_g.
    pub k_g: usize,

    // -------------------------------------------------------------------------
    // Public statement surface (hashed/bound by `we_statement_hash_*` in Architecture‑T)
    // -------------------------------------------------------------------------
    /// Per-instance witness commitments `cm_f[i]` (length κ ring elements each).
    pub has_cm_f: bool,
    /// Per-instance CP commitments to transcript messages:
    /// - `cfs_had_u[i]` commits to `had_u[i]` (encoded as 3*d ring elems)
    /// - `cfs_mon_b[i]` commits to `mon_b[i]` (length k_g ring elems)
    pub has_cfs_had_u: bool,
    pub has_cfs_mon_b: bool,

    // -------------------------------------------------------------------------
    // What our current DPP arithmetization covers
    // -------------------------------------------------------------------------
    /// Poseidon transcript trace constraints (FS-in-gate).
    pub dpp_covers_poseidon_trace: bool,
    /// Ajtai-open constraints for `cfs_had_u` and `cfs_mon_b`.
    pub dpp_covers_cfs_openings: bool,

    // -------------------------------------------------------------------------
    // Remaining “next inventory item” for production binding
    // -------------------------------------------------------------------------
    /// Full Π_fold verifier arithmetic (sumcheck verification, Eq(26), monomial recomputation, Step‑5).
    pub missing_pifold_math_checks: bool,
    /// Production binding: constraints tying `aux` back to `cm_f` without opening full witness.
    ///
    /// This is the “auxcs_lin × batchlin” layer from the paper: linear/evaluation constraints that
    /// ensure `aux.had_u` and `aux.mon_b` are the correct derived/evaluated values from the committed
    /// witness.
    pub missing_aux_to_cm_f_binding: bool,
}

impl WeGateInventory {
    pub fn for_proof<R: PolyRing + OverField>(proof: &PiFoldBatchedProof<R>, ell: usize) -> Self
    where
        R::BaseRing: Zq,
    {
        Self {
            ell,
            d: R::dimension(),
            k_g: proof.rg_params.k_g,
            has_cm_f: true,
            has_cfs_had_u: true,
            has_cfs_mon_b: true,
            dpp_covers_poseidon_trace: true,
            dpp_covers_cfs_openings: true,
            missing_pifold_math_checks: true,
            missing_aux_to_cm_f_binding: true,
        }
    }

    /// The *size* of aux values the verifier consumes (per batch), if aux is used.
    ///
    /// This is useful for deciding which binding constraints we can afford to add.
    pub fn aux_sizes(&self) -> AuxSizes {
        AuxSizes {
            had_u_scalars: self.ell * 3 * self.d, // base scalars (ct) per instance
            mon_b_ring_elems: self.ell * self.k_g, // ring elems per instance
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct AuxSizes {
    pub had_u_scalars: usize,
    pub mon_b_ring_elems: usize,
}

/// Sanity helper: ensure aux shapes match the proof params.
pub fn check_aux_shapes<R: PolyRing + OverField>(
    aux: &PiFoldAuxWitness<R>,
    ell: usize,
    k_g: usize,
) -> Result<(), String>
where
    R::BaseRing: Zq,
{
    if aux.had_u.len() != ell || aux.mon_b.len() != ell {
        return Err("aux length mismatch".to_string());
    }
    for u in &aux.had_u {
        if u[0].len() != R::dimension() || u[1].len() != R::dimension() || u[2].len() != R::dimension() {
            return Err("aux.had_u wrong dimension".to_string());
        }
    }
    for b in &aux.mon_b {
        if b.len() != k_g {
            return Err("aux.mon_b wrong k_g".to_string());
        }
    }
    Ok(())
}

