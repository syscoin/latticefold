//! Shared ALVO-stage metadata for the research H12 redesign.

use dpp::theorem43::{Theorem43AlvoLocalCheckSurface, Theorem43Coins};
use latticefold::transcript::poseidon::F257;
use sha2::Digest;
use std::io::{Read, Write};

use crate::h12_alvo_daleo::{H12DaleoParams, H12DaleoProof};
use crate::h12_pi_commit::f257_to_u16;

#[derive(Clone, Debug)]
pub struct H12AlvoClaim {
    pub alpha: u16,
    pub beta: u16,
    pub gamma: u16,
    pub alpha_pi_sparse: u16,
    pub beta_pi_sparse: u16,
    pub gamma_pi_sparse: u16,
}

#[derive(Clone, Debug)]
pub struct H12AlvoAnchorProjection {
    pub y_anchor_stream: u16,
    pub alpha_sparse: u16,
    pub beta_sparse: u16,
    pub delta_gamma_pi: u16,
}

#[derive(Clone, Debug)]
pub struct H12AlvoRepSurfaceBundle {
    pub rep_id: u64,
    pub anchor_block_id: usize,
    pub anchor_hint_blocks: Vec<usize>,
    pub anchor_projection: H12AlvoAnchorProjection,
    pub poison_blocks: usize,
    pub poison_gate: u16,
}

#[derive(Clone, Debug)]
pub struct H12AlvoLogicalLockAugmentation {
    pub share_index: u32,
    pub r_cap_reps: u16,
    pub stage1_root: [u8; 32],
    pub stage2_root: [u8; 32],
    pub schedule_digest: [u8; 32],
    pub surfaces: Vec<Theorem43AlvoLocalCheckSurface<F257>>,
    pub daleo_proof: Option<H12DaleoProof>,
    pub rep_bundles: Vec<H12AlvoRepSurfaceBundle>,
}

#[derive(Clone, Debug)]
pub struct H12AlvoAugmentedPackage {
    pub stage1_lock_package_hash: [u8; 32],
    pub stage1_lock_package: Vec<u8>,
    pub logical_locks: Vec<H12AlvoLogicalLockAugmentation>,
}

pub fn digest_rep_bundle_hashes(rep_bundle_hashes: &[[u8; 32]], pack_d: usize) -> [u8; 32] {
    let mut h = sha2::Sha256::new();
    h.update(b"LFP_H12_ALVO_REP_BUNDLES_V2");
    h.update((pack_d as u32).to_le_bytes());
    h.update((rep_bundle_hashes.len() as u32).to_le_bytes());
    for hash in rep_bundle_hashes {
        h.update(hash);
    }
    h.finalize().into()
}

pub fn digest_alvo_schedule(
    surfaces: &[Theorem43AlvoLocalCheckSurface<F257>],
    pack_d: usize,
) -> [u8; 32] {
    let mut h = sha2::Sha256::new();
    h.update(b"LFP_H12_ALVO_SCHEDULE_V1");
    h.update((pack_d as u32).to_le_bytes());
    h.update((surfaces.len() as u32).to_le_bytes());
    for s in surfaces {
        digest_surface_into(&mut h, s, pack_d);
    }
    h.finalize().into()
}

pub fn derive_stage1_root(
    stmt_digest: &[F257; 32],
    lock_coin_seed: &[u8; 32],
    share_index: u32,
    r_cap_reps: u16,
    schedule_digest: &[u8; 32],
) -> [u8; 32] {
    let mut h = sha2::Sha256::new();
    h.update(b"LFP_H12_ALVO_STAGE1_ROOT_V1");
    for f in stmt_digest {
        h.update(f257_to_u16(*f).to_le_bytes());
    }
    h.update(lock_coin_seed);
    h.update(share_index.to_le_bytes());
    h.update(r_cap_reps.to_le_bytes());
    h.update(schedule_digest);
    h.finalize().into()
}

pub fn derive_stage2_root(lock: &H12AlvoLogicalLockAugmentation) -> [u8; 32] {
    let mut h = sha2::Sha256::new();
    h.update(b"LFP_H12_ALVO_STAGE2_ROOT_V5");
    h.update(lock.share_index.to_le_bytes());
    h.update(lock.r_cap_reps.to_le_bytes());
    h.update(lock.stage1_root);
    h.update(lock.schedule_digest);
    match &lock.daleo_proof {
        Some(proof) => {
            h.update([1u8]);
            digest_daleo_proof_into(&mut h, proof);
        }
        None => h.update([0u8]),
    }
    h.update((lock.rep_bundles.len() as u32).to_le_bytes());
    for rep in &lock.rep_bundles {
        h.update(rep.rep_id.to_le_bytes());
        h.update(rep.anchor_block_id.to_le_bytes());
        h.update((rep.anchor_hint_blocks.len() as u32).to_le_bytes());
        for &block_idx in &rep.anchor_hint_blocks {
            h.update(block_idx.to_le_bytes());
        }
        digest_anchor_projection_into(&mut h, &rep.anchor_projection);
        h.update(rep.poison_blocks.to_le_bytes());
        h.update(rep.poison_gate.to_le_bytes());
    }
    h.finalize().into()
}

pub fn digest_stage1_lock_package_bytes(bytes: &[u8]) -> [u8; 32] {
    let mut h = sha2::Sha256::new();
    h.update(b"LFP_H12_ALVO_STAGE1_PKG_V1");
    h.update((bytes.len() as u64).to_le_bytes());
    h.update(bytes);
    h.finalize().into()
}

fn digest_surface_into(
    h: &mut sha2::Sha256,
    surface: &Theorem43AlvoLocalCheckSurface<F257>,
    pack_d: usize,
) {
    let base = &surface.local_check;
    h.update(base.block_id.to_le_bytes());
    h.update(base.rep_id.to_le_bytes());
    digest_coins_into(h, &base.coins);
    digest_sparse_terms_into(h, base.q1_pi_terms.as_slice());
    digest_sparse_terms_into(h, base.q2_pi_terms.as_slice());
    digest_sparse_terms_into(h, base.q3_pi_sparse_terms.as_slice());
    h.update(f257_to_u16(base.q1_x_dot).to_le_bytes());
    h.update(f257_to_u16(base.q2_x_dot).to_le_bytes());
    h.update(f257_to_u16(base.q3_x_dot_sparse).to_le_bytes());
    digest_sparse_terms_into(h, base.q1_w_terms.as_slice());
    digest_sparse_terms_into(h, base.q2_w_terms.as_slice());
    digest_sparse_terms_into(h, base.q3_w_terms.as_slice());
    h.update(base.w_eval_block_pi_offset.to_le_bytes());
    h.update(base.w_eval_block_len.to_le_bytes());
    h.update([base.requires_dense_q3_w_eval as u8]);
    h.update(surface.proof_layout.n.to_le_bytes());
    h.update(surface.proof_layout.n_total.to_le_bytes());
    h.update(surface.proof_layout.z_w_len.to_le_bytes());
    h.update(surface.proof_layout.pi0_len.to_le_bytes());
    h.update(surface.proof_layout.blocks.to_le_bytes());
    h.update(surface.proof_layout.ell_local.to_le_bytes());
    h.update(surface.proof_layout.k_star.to_le_bytes());
    h.update(surface.proof_layout.low_cube_len.to_le_bytes());
    h.update((surface.proof_layout.witness_positions_star.len() as u32).to_le_bytes());
    for &pos in &surface.proof_layout.witness_positions_star {
        h.update(pos.to_le_bytes());
    }
    h.update((surface.h_w_eval_constraints.len() as u32).to_le_bytes());
    for lc in &surface.h_w_eval_constraints {
        h.update(f257_to_u16(lc.constant).to_le_bytes());
        h.update((lc.terms.len() as u32).to_le_bytes());
        for &(pos, coeff) in &lc.terms {
            h.update(pos.to_le_bytes());
            h.update(f257_to_u16(coeff).to_le_bytes());
        }
    }
    h.update((surface.touched_pi_positions_exact.len() as u32).to_le_bytes());
    for &pi_idx in &surface.touched_pi_positions_exact {
        h.update(pi_idx.to_le_bytes());
    }
    h.update((surface.touched_pi_positions_conservative.len() as u32).to_le_bytes());
    for &pi_idx in &surface.touched_pi_positions_conservative {
        h.update(pi_idx.to_le_bytes());
    }
    let exact_blocks = surface.packed_pi_blocks_exact(pack_d);
    h.update((exact_blocks.len() as u32).to_le_bytes());
    for block in exact_blocks {
        h.update(block.to_le_bytes());
    }
    let cons_blocks = surface.packed_pi_blocks_conservative(pack_d);
    h.update((cons_blocks.len() as u32).to_le_bytes());
    for block in cons_blocks {
        h.update(block.to_le_bytes());
    }
}

fn digest_coins_into(h: &mut sha2::Sha256, coins: &Theorem43Coins<F257>) {
    h.update(coins.idx.to_le_bytes());
    h.update(f257_to_u16(coins.lambda).to_le_bytes());
    h.update(f257_to_u16(coins.rho).to_le_bytes());
    h.update(f257_to_u16(coins.sigma).to_le_bytes());
    h.update(f257_to_u16(coins.c_hit).to_le_bytes());
}

fn digest_sparse_terms_into(h: &mut sha2::Sha256, terms: &[(usize, F257)]) {
    h.update((terms.len() as u32).to_le_bytes());
    for &(idx, coeff) in terms {
        h.update(idx.to_le_bytes());
        h.update(f257_to_u16(coeff).to_le_bytes());
    }
}

fn digest_anchor_projection_into(h: &mut sha2::Sha256, proj: &H12AlvoAnchorProjection) {
    h.update(proj.y_anchor_stream.to_le_bytes());
    h.update(proj.alpha_sparse.to_le_bytes());
    h.update(proj.beta_sparse.to_le_bytes());
    h.update(proj.delta_gamma_pi.to_le_bytes());
}

fn digest_daleo_proof_into(h: &mut sha2::Sha256, proof: &H12DaleoProof) {
    h.update(proof.params.pack_d.to_le_bytes());
    h.update(proof.params.commit_rows.to_le_bytes());
    h.update(proof.params.blind_len.to_le_bytes());
    h.update((proof.local_view_values.len() as u32).to_le_bytes());
    for &v in &proof.local_view_values {
        h.update(v.to_le_bytes());
    }
    h.update((proof.blind_values.len() as u32).to_le_bytes());
    for &v in &proof.blind_values {
        h.update(v.to_le_bytes());
    }
    h.update((proof.compressed_opening_bytes.len() as u32).to_le_bytes());
    for &v in &proof.compressed_opening_bytes {
        h.update(v.to_le_bytes());
    }
    h.update((proof.ajtai_commitment_rows.len() as u32).to_le_bytes());
    for row in &proof.ajtai_commitment_rows {
        h.update((row.len() as u32).to_le_bytes());
        for &v in row {
            h.update(v.to_le_bytes());
        }
    }
    h.update((proof.designated_challenge.len() as u32).to_le_bytes());
    for &v in &proof.designated_challenge {
        h.update(v.to_le_bytes());
    }
}

pub fn write_augmented_package(
    w: &mut impl Write,
    pkg: &H12AlvoAugmentedPackage,
) -> std::io::Result<()> {
    w.write_all(b"LFP1ALVOA5")?;
    w.write_all(&pkg.stage1_lock_package_hash)?;
    write_u32(w, pkg.stage1_lock_package.len() as u32)?;
    w.write_all(pkg.stage1_lock_package.as_slice())?;
    write_u32(w, pkg.logical_locks.len() as u32)?;
    for lock in &pkg.logical_locks {
        write_u32(w, lock.share_index)?;
        write_u16(w, lock.r_cap_reps)?;
        w.write_all(&lock.stage1_root)?;
        w.write_all(&lock.stage2_root)?;
        w.write_all(&lock.schedule_digest)?;
        write_u32(w, lock.surfaces.len() as u32)?;
        for surface in &lock.surfaces {
            write_surface(w, surface)?;
        }
        match &lock.daleo_proof {
            Some(proof) => {
                w.write_all(&[1u8])?;
                write_daleo_proof(w, proof)?;
            }
            None => w.write_all(&[0u8])?,
        }
        write_u32(w, lock.rep_bundles.len() as u32)?;
        for rep in &lock.rep_bundles {
            write_u64(w, rep.rep_id)?;
            write_u64(w, rep.anchor_block_id as u64)?;
            write_u32(w, rep.anchor_hint_blocks.len() as u32)?;
            for &block_idx in &rep.anchor_hint_blocks {
                write_u64(w, block_idx as u64)?;
            }
            write_anchor_projection(w, &rep.anchor_projection)?;
            write_u32(w, rep.poison_blocks as u32)?;
            write_u16(w, rep.poison_gate)?;
        }
    }
    Ok(())
}

pub fn read_augmented_package(r: &mut impl Read) -> std::io::Result<H12AlvoAugmentedPackage> {
    let mut magic = [0u8; 10];
    r.read_exact(&mut magic)?;
    if &magic != b"LFP1ALVOA5" {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "bad H12 ALVO augmented package magic",
        ));
    }
    let mut stage1_lock_package_hash = [0u8; 32];
    r.read_exact(&mut stage1_lock_package_hash)?;
    let stage1_len = read_u32(r)? as usize;
    let mut stage1_lock_package = vec![0u8; stage1_len];
    r.read_exact(stage1_lock_package.as_mut_slice())?;
    let lock_n = read_u32(r)? as usize;
    let mut logical_locks = Vec::with_capacity(lock_n);
    for _ in 0..lock_n {
        let share_index = read_u32(r)?;
        let r_cap_reps = read_u16(r)?;
        let mut stage1_root = [0u8; 32];
        r.read_exact(&mut stage1_root)?;
        let mut stage2_root = [0u8; 32];
        r.read_exact(&mut stage2_root)?;
        let mut schedule_digest = [0u8; 32];
        r.read_exact(&mut schedule_digest)?;
        let surface_n = read_u32(r)? as usize;
        let mut surfaces = Vec::with_capacity(surface_n);
        for _ in 0..surface_n {
            surfaces.push(read_surface(r)?);
        }
        let mut daleo_proof = None;
        let mut has_proof = [0u8; 1];
        r.read_exact(&mut has_proof)?;
        if has_proof[0] != 0 {
            daleo_proof = Some(read_daleo_proof(r)?);
        }
        let rep_n = read_u32(r)? as usize;
        let mut rep_bundles = Vec::with_capacity(rep_n);
        for _ in 0..rep_n {
            let rep_id = read_u64(r)?;
            let anchor_block_id = read_u64(r)? as usize;
            let hint_n = read_u32(r)? as usize;
            let mut anchor_hint_blocks = Vec::with_capacity(hint_n);
            for _ in 0..hint_n {
                anchor_hint_blocks.push(read_u64(r)? as usize);
            }
            let anchor_projection = read_anchor_projection(r)?;
            let poison_blocks = read_u32(r)? as usize;
            let poison_gate = read_u16(r)?;
            rep_bundles.push(H12AlvoRepSurfaceBundle {
                rep_id,
                anchor_block_id,
                anchor_hint_blocks,
                anchor_projection,
                poison_blocks,
                poison_gate,
            });
        }
        logical_locks.push(H12AlvoLogicalLockAugmentation {
            share_index,
            r_cap_reps,
            stage1_root,
            stage2_root,
            schedule_digest,
            surfaces,
            daleo_proof,
            rep_bundles,
        });
    }
    Ok(H12AlvoAugmentedPackage {
        stage1_lock_package_hash,
        stage1_lock_package,
        logical_locks,
    })
}

fn write_surface(
    w: &mut impl Write,
    surface: &Theorem43AlvoLocalCheckSurface<F257>,
) -> std::io::Result<()> {
    let base = &surface.local_check;
    write_u64(w, base.block_id as u64)?;
    write_u64(w, base.rep_id)?;
    write_coins(w, &base.coins)?;
    write_sparse_terms(w, base.q1_pi_terms.as_slice())?;
    write_sparse_terms(w, base.q2_pi_terms.as_slice())?;
    write_sparse_terms(w, base.q3_pi_sparse_terms.as_slice())?;
    write_u16(w, f257_to_u16(base.q1_x_dot))?;
    write_u16(w, f257_to_u16(base.q2_x_dot))?;
    write_u16(w, f257_to_u16(base.q3_x_dot_sparse))?;
    write_sparse_terms(w, base.q1_w_terms.as_slice())?;
    write_sparse_terms(w, base.q2_w_terms.as_slice())?;
    write_sparse_terms(w, base.q3_w_terms.as_slice())?;
    write_u64(w, base.w_eval_block_pi_offset as u64)?;
    write_u64(w, base.w_eval_block_len as u64)?;
    w.write_all(&[base.requires_dense_q3_w_eval as u8])?;
    let layout = &surface.proof_layout;
    write_u64(w, layout.n as u64)?;
    write_u64(w, layout.n_total as u64)?;
    write_u64(w, layout.z_w_len as u64)?;
    write_u64(w, layout.pi0_len as u64)?;
    write_u64(w, layout.blocks as u64)?;
    write_u64(w, layout.ell_local as u64)?;
    write_u64(w, layout.k_star as u64)?;
    write_u64(w, layout.low_cube_len as u64)?;
    write_u32(w, layout.witness_positions_star.len() as u32)?;
    for &pos in &layout.witness_positions_star {
        write_u64(w, pos as u64)?;
    }
    write_u32(w, surface.h_w_eval_constraints.len() as u32)?;
    for lc in &surface.h_w_eval_constraints {
        write_u16(w, f257_to_u16(lc.constant))?;
        write_u32(w, lc.terms.len() as u32)?;
        for &(pos, coeff) in &lc.terms {
            write_u64(w, pos as u64)?;
            write_u16(w, f257_to_u16(coeff))?;
        }
    }
    write_u32(w, surface.touched_pi_positions_exact.len() as u32)?;
    for &pi_idx in &surface.touched_pi_positions_exact {
        write_u64(w, pi_idx as u64)?;
    }
    write_u32(w, surface.touched_pi_positions_conservative.len() as u32)?;
    for &pi_idx in &surface.touched_pi_positions_conservative {
        write_u64(w, pi_idx as u64)?;
    }
    Ok(())
}

fn read_surface(r: &mut impl Read) -> std::io::Result<Theorem43AlvoLocalCheckSurface<F257>> {
    let block_id = read_u64(r)? as usize;
    let rep_id = read_u64(r)?;
    let coins = read_coins(r)?;
    let q1_pi_terms = read_sparse_terms(r)?;
    let q2_pi_terms = read_sparse_terms(r)?;
    let q3_pi_sparse_terms = read_sparse_terms(r)?;
    let q1_x_dot = u16_to_f257(read_u16(r)?);
    let q2_x_dot = u16_to_f257(read_u16(r)?);
    let q3_x_dot_sparse = u16_to_f257(read_u16(r)?);
    let q1_w_terms = read_sparse_terms(r)?;
    let q2_w_terms = read_sparse_terms(r)?;
    let q3_w_terms = read_sparse_terms(r)?;
    let w_eval_block_pi_offset = read_u64(r)? as usize;
    let w_eval_block_len = read_u64(r)? as usize;
    let mut dense = [0u8; 1];
    r.read_exact(&mut dense)?;
    let n = read_u64(r)? as usize;
    let n_total = read_u64(r)? as usize;
    let z_w_len = read_u64(r)? as usize;
    let pi0_len = read_u64(r)? as usize;
    let blocks = read_u64(r)? as usize;
    let ell_local = read_u64(r)? as usize;
    let k_star = read_u64(r)? as usize;
    let low_cube_len = read_u64(r)? as usize;
    let witness_pos_n = read_u32(r)? as usize;
    let mut witness_positions_star = Vec::with_capacity(witness_pos_n);
    for _ in 0..witness_pos_n {
        witness_positions_star.push(read_u64(r)? as usize);
    }
    let lc_n = read_u32(r)? as usize;
    let mut h_w_eval_constraints = Vec::with_capacity(lc_n);
    for _ in 0..lc_n {
        let constant = u16_to_f257(read_u16(r)?);
        let term_n = read_u32(r)? as usize;
        let mut terms = Vec::with_capacity(term_n);
        for _ in 0..term_n {
            terms.push((read_u64(r)? as usize, u16_to_f257(read_u16(r)?)));
        }
        h_w_eval_constraints.push(dpp::dr1cs_flpcp::Dr1csBlockLinearConstraint { constant, terms });
    }
    let exact_n = read_u32(r)? as usize;
    let mut touched_pi_positions_exact = Vec::with_capacity(exact_n);
    for _ in 0..exact_n {
        touched_pi_positions_exact.push(read_u64(r)? as usize);
    }
    let cons_n = read_u32(r)? as usize;
    let mut touched_pi_positions_conservative = Vec::with_capacity(cons_n);
    for _ in 0..cons_n {
        touched_pi_positions_conservative.push(read_u64(r)? as usize);
    }
    Ok(Theorem43AlvoLocalCheckSurface {
        local_check: dpp::theorem43::Theorem43CapsuleLocalCheckSurface {
            block_id,
            rep_id,
            coins,
            q1_pi_terms,
            q2_pi_terms,
            q3_pi_sparse_terms,
            q1_x_dot,
            q2_x_dot,
            q3_x_dot_sparse,
            q1_w_terms,
            q2_w_terms,
            q3_w_terms,
            w_eval_block_pi_offset,
            w_eval_block_len,
            requires_dense_q3_w_eval: dense[0] != 0,
        },
        proof_layout: dpp::dr1cs_flpcp::Dr1csProofLayoutInfo {
            n,
            n_total,
            z_w_len,
            pi0_len,
            blocks,
            ell_local,
            k_star,
            low_cube_len,
            witness_positions_star,
        },
        h_w_eval_constraints,
        touched_pi_positions_exact,
        touched_pi_positions_conservative,
    })
}

fn write_anchor_projection(w: &mut impl Write, proj: &H12AlvoAnchorProjection) -> std::io::Result<()> {
    write_u16(w, proj.y_anchor_stream)?;
    write_u16(w, proj.alpha_sparse)?;
    write_u16(w, proj.beta_sparse)?;
    write_u16(w, proj.delta_gamma_pi)?;
    Ok(())
}

fn read_anchor_projection(r: &mut impl Read) -> std::io::Result<H12AlvoAnchorProjection> {
    Ok(H12AlvoAnchorProjection {
        y_anchor_stream: read_u16(r)?,
        alpha_sparse: read_u16(r)?,
        beta_sparse: read_u16(r)?,
        delta_gamma_pi: read_u16(r)?,
    })
}

fn write_coins(w: &mut impl Write, coins: &Theorem43Coins<F257>) -> std::io::Result<()> {
    write_u64(w, coins.idx as u64)?;
    write_u16(w, f257_to_u16(coins.lambda))?;
    write_u16(w, f257_to_u16(coins.rho))?;
    write_u16(w, f257_to_u16(coins.sigma))?;
    write_u16(w, f257_to_u16(coins.c_hit))?;
    Ok(())
}

fn read_coins(r: &mut impl Read) -> std::io::Result<Theorem43Coins<F257>> {
    Ok(Theorem43Coins {
        idx: read_u64(r)? as usize,
        lambda: u16_to_f257(read_u16(r)?),
        rho: u16_to_f257(read_u16(r)?),
        sigma: u16_to_f257(read_u16(r)?),
        c_hit: u16_to_f257(read_u16(r)?),
    })
}

fn write_sparse_terms(w: &mut impl Write, terms: &[(usize, F257)]) -> std::io::Result<()> {
    write_u32(w, terms.len() as u32)?;
    for &(idx, coeff) in terms {
        write_u64(w, idx as u64)?;
        write_u16(w, f257_to_u16(coeff))?;
    }
    Ok(())
}

fn read_sparse_terms(r: &mut impl Read) -> std::io::Result<Vec<(usize, F257)>> {
    let n = read_u32(r)? as usize;
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        out.push((read_u64(r)? as usize, u16_to_f257(read_u16(r)?)));
    }
    Ok(out)
}

fn write_daleo_proof(w: &mut impl Write, proof: &H12DaleoProof) -> std::io::Result<()> {
    write_u16(w, proof.params.pack_d)?;
    write_u16(w, proof.params.commit_rows)?;
    write_u16(w, proof.params.blind_len)?;
    write_u32(w, proof.local_view_values.len() as u32)?;
    for &v in &proof.local_view_values {
        write_u16(w, v)?;
    }
    write_u32(w, proof.blind_values.len() as u32)?;
    for &v in &proof.blind_values {
        write_u16(w, v)?;
    }
    write_u32(w, proof.compressed_opening_bytes.len() as u32)?;
    for &v in &proof.compressed_opening_bytes {
        write_u16(w, v)?;
    }
    write_u32(w, proof.ajtai_commitment_rows.len() as u32)?;
    for row in &proof.ajtai_commitment_rows {
        write_u32(w, row.len() as u32)?;
        for &v in row {
            write_u64(w, v)?;
        }
    }
    write_u32(w, proof.designated_challenge.len() as u32)?;
    for &v in &proof.designated_challenge {
        write_u16(w, v)?;
    }
    Ok(())
}

fn read_daleo_proof(r: &mut impl Read) -> std::io::Result<H12DaleoProof> {
    let pack_d = read_u16(r)?;
    let commit_rows = read_u16(r)?;
    let blind_len = read_u16(r)?;
    let local_n = read_u32(r)? as usize;
    let mut local_view_values = Vec::with_capacity(local_n);
    for _ in 0..local_n {
        local_view_values.push(read_u16(r)?);
    }
    let blind_n = read_u32(r)? as usize;
    let mut blind_values = Vec::with_capacity(blind_n);
    for _ in 0..blind_n {
        blind_values.push(read_u16(r)?);
    }
    let compressed_n = read_u32(r)? as usize;
    let mut compressed_opening_bytes = Vec::with_capacity(compressed_n);
    for _ in 0..compressed_n {
        compressed_opening_bytes.push(read_u16(r)?);
    }
    let row_n = read_u32(r)? as usize;
    let mut ajtai_commitment_rows = Vec::with_capacity(row_n);
    for _ in 0..row_n {
        let len = read_u32(r)? as usize;
        let mut row = Vec::with_capacity(len);
        for _ in 0..len {
            row.push(read_u64(r)?);
        }
        ajtai_commitment_rows.push(row);
    }
    let challenge_n = read_u32(r)? as usize;
    let mut designated_challenge = Vec::with_capacity(challenge_n);
    for _ in 0..challenge_n {
        designated_challenge.push(read_u16(r)?);
    }
    Ok(H12DaleoProof {
        params: H12DaleoParams {
            pack_d,
            commit_rows,
            blind_len,
        },
        local_view_values,
        blind_values,
        compressed_opening_bytes,
        ajtai_commitment_rows,
        designated_challenge,
    })
}

fn write_u16(w: &mut impl Write, v: u16) -> std::io::Result<()> {
    w.write_all(&v.to_le_bytes())
}

fn write_u32(w: &mut impl Write, v: u32) -> std::io::Result<()> {
    w.write_all(&v.to_le_bytes())
}

fn write_u64(w: &mut impl Write, v: u64) -> std::io::Result<()> {
    w.write_all(&v.to_le_bytes())
}

fn read_u16(r: &mut impl Read) -> std::io::Result<u16> {
    let mut b = [0u8; 2];
    r.read_exact(&mut b)?;
    Ok(u16::from_le_bytes(b))
}

fn read_u32(r: &mut impl Read) -> std::io::Result<u32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(u32::from_le_bytes(b))
}

fn read_u64(r: &mut impl Read) -> std::io::Result<u64> {
    let mut b = [0u8; 8];
    r.read_exact(&mut b)?;
    Ok(u64::from_le_bytes(b))
}

fn u16_to_f257(x: u16) -> F257 {
    F257::from((x % 257) as u64)
}

#[cfg(test)]
mod tests {
    use ark_ff::Field;
    use super::*;
    use dpp::dr1cs_flpcp::Dr1csProofLayoutInfo;
    use dpp::theorem43::{Theorem43AlvoLocalCheckSurface, Theorem43CapsuleLocalCheckSurface};

    fn dummy_surface(rep_id: u64) -> Theorem43AlvoLocalCheckSurface<F257> {
        let local_check = Theorem43CapsuleLocalCheckSurface {
            block_id: 0,
            rep_id,
            coins: Theorem43Coins {
                idx: 1,
                lambda: F257::from(2u64),
                rho: F257::from(3u64),
                sigma: F257::from(4u64),
                c_hit: F257::from(5u64),
            },
            q1_pi_terms: vec![(0, F257::ONE)],
            q2_pi_terms: vec![(1, F257::ONE)],
            q3_pi_sparse_terms: vec![(2, F257::ONE)],
            q1_x_dot: F257::ZERO,
            q2_x_dot: F257::ZERO,
            q3_x_dot_sparse: F257::ZERO,
            q1_w_terms: vec![(0, F257::ONE)],
            q2_w_terms: vec![(1, F257::ONE)],
            q3_w_terms: vec![(2, F257::ONE)],
            w_eval_block_pi_offset: 8,
            w_eval_block_len: 4,
            requires_dense_q3_w_eval: true,
        };
        Theorem43AlvoLocalCheckSurface {
            touched_pi_positions_exact: local_check.touched_pi_positions(false),
            touched_pi_positions_conservative: local_check.touched_pi_positions(true),
            local_check,
            proof_layout: Dr1csProofLayoutInfo {
                n: 4,
                n_total: 8,
                z_w_len: 8,
                pi0_len: 16,
                blocks: 2,
                ell_local: 4,
                k_star: 4,
                low_cube_len: 2,
                witness_positions_star: vec![0, 1, 2, 3],
            },
            h_w_eval_constraints: Vec::new(),
        }
    }

    fn dummy_daleo_proof(tag: u8) -> H12DaleoProof {
        H12DaleoProof {
            params: H12DaleoParams {
                pack_d: 64,
                commit_rows: 16,
                blind_len: 16,
            },
            local_view_values: vec![tag as u16, (tag as u16) + 1],
            blind_values: vec![(tag as u16) + 2],
            compressed_opening_bytes: vec![(tag as u16) + 3],
            ajtai_commitment_rows: vec![vec![tag as u64, (tag as u64) + 1]],
            designated_challenge: vec![(tag as u16) + 4],
        }
    }

    #[test]
    fn test_stage1_root_changes_with_schedule() {
        let stmt_digest = [F257::from(7u64); 32];
        let lock_coin_seed = [9u8; 32];
        let a = vec![dummy_surface(1)];
        let b = vec![dummy_surface(2)];
        let a_digest = digest_alvo_schedule(a.as_slice(), 64);
        let b_digest = digest_alvo_schedule(b.as_slice(), 64);
        let ra = derive_stage1_root(&stmt_digest, &lock_coin_seed, 3, 1, &a_digest);
        let rb = derive_stage1_root(&stmt_digest, &lock_coin_seed, 3, 1, &b_digest);
        assert_ne!(ra, rb);
    }

    #[test]
    fn test_augmented_package_roundtrip() {
        let surface = dummy_surface(11);
        let rep_bundle = H12AlvoRepSurfaceBundle {
            rep_id: 11,
            anchor_block_id: 11,
            anchor_hint_blocks: vec![0, 2, 5],
            anchor_projection: H12AlvoAnchorProjection {
                y_anchor_stream: 20,
                alpha_sparse: 21,
                beta_sparse: 22,
                delta_gamma_pi: 23,
            },
            poison_blocks: 1,
            poison_gate: 30,
        };
        let mut lock = H12AlvoLogicalLockAugmentation {
            share_index: 7,
            r_cap_reps: 1,
            stage1_root: [3u8; 32],
            stage2_root: [0u8; 32],
            schedule_digest: digest_alvo_schedule(std::slice::from_ref(&surface), 64),
            surfaces: vec![surface],
            daleo_proof: Some(dummy_daleo_proof(55)),
            rep_bundles: vec![rep_bundle],
        };
        lock.stage2_root = derive_stage2_root(&lock);
        let pkg = H12AlvoAugmentedPackage {
            stage1_lock_package_hash: digest_stage1_lock_package_bytes(b"stage1"),
            stage1_lock_package: b"stage1".to_vec(),
            logical_locks: vec![lock],
        };
        let mut buf = Vec::new();
        write_augmented_package(&mut buf, &pkg).expect("write augmented package");
        let got = read_augmented_package(&mut std::io::Cursor::new(buf)).expect("read augmented package");
        assert_eq!(got.stage1_lock_package_hash, pkg.stage1_lock_package_hash);
        assert_eq!(got.stage1_lock_package, pkg.stage1_lock_package);
        assert_eq!(got.logical_locks.len(), 1);
        assert_eq!(got.logical_locks[0].share_index, 7);
        assert_eq!(got.logical_locks[0].rep_bundles[0].anchor_hint_blocks, vec![0, 2, 5]);
        assert_eq!(
            got.logical_locks[0]
                .daleo_proof
                .as_ref()
                .expect("proof present")
                .local_view_values,
            vec![55, 56]
        );
    }

    #[test]
    fn test_stage2_root_changes_with_daleo_proof_payload() {
        let surface = dummy_surface(5);
        let mut base = H12AlvoLogicalLockAugmentation {
            share_index: 1,
            r_cap_reps: 1,
            stage1_root: [1u8; 32],
            stage2_root: [0u8; 32],
            schedule_digest: digest_alvo_schedule(std::slice::from_ref(&surface), 64),
            surfaces: vec![surface],
            daleo_proof: Some(dummy_daleo_proof(9)),
            rep_bundles: Vec::new(),
        };
        let root_a = derive_stage2_root(&base);
        base.daleo_proof = Some(dummy_daleo_proof(10));
        let root_b = derive_stage2_root(&base);
        assert_ne!(root_a, root_b);
    }
}
