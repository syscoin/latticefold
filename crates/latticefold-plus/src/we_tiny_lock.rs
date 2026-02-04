//! Tiny-field (Theorem 4.3) lock helpers for WE arming.
//!
//! This module provides an **arm-before-proof** arming helper that binds to a statement digest
//! and an armer-private secret, using the Theorem-4.3 tiny-field DPP lockable query generator.
//!
//! Notes:
//! - This is intended for tiny fields (e.g., F257) where the accepting set is `{1,2}`.
//! - The lock artifact contains *public coins* and keeps the hidden query inside (as toxic waste),
//!   serving as a stand-in for LWE hints in the real lock layer.

use ark_ff::{FftField, PrimeField};
use rayon::prelude::*;
use std::sync::Arc;

use dpp::dr1cs_flpcp::{Dr1csNpFlpcpSparseApi, Dr1csQueryScratch, MulCode, QuerySink, TensorRsMulCode};
use dpp::packing::FlpcpPredicate;
use dpp::theorem43::{Theorem43Coins, Theorem43Dpp, Theorem43LockArtifact};
use dpp::SparseVec;
use symphony::file_backed_dr1cs::FileBackedSparseDr1csInstance;

#[inline]
fn is_f257_field<F: PrimeField>() -> bool {
    // `PrimeField::characteristic()` returns little-endian u64 limbs.
    // F257's characteristic is exactly 257.
    F::characteristic() == &[257u64]
}

use crate::lockable_ringlwe::{arm_ringlwe_lock, RingLweLockArtifact, RingLweParams};
use crate::lockable_ringlwe::QueryBlockAccumulator;

pub use crate::we_statement::arm_theorem43_from_statement;

#[cfg(feature = "we_gate")]
use crate::we_gate_arith;
#[cfg(feature = "we_gate")]
use crate::we_statement::WeParams;
#[cfg(feature = "we_gate")]
use latticefold::transcript::poseidon::F257;
#[cfg(feature = "we_gate")]
use stark_rings::{CoeffRing, OverField, PolyRing, Zq};

/// File-backed chunked multiplication-code FLPCP backend (NP dR1CS, sparse queries).
///
/// This is the critical "proper streaming" fix: it avoids materializing the full sparse dR1CS
/// (hundreds of millions of constraints) into RAM. Instead, it streams constraint rows and term
/// pools from the file-backed format in `/tmp`.
#[derive(Clone, Debug)]
struct FileBackedChunkedMulCodeDr1csNpFlpcpSparse<F: PrimeField, C: MulCode<F> + Sync> {
    fb: FileBackedSparseDr1csInstance<F>,
    l: usize,
    code: C,
    ckpts: Arc<Vec<(u64, u64, u64, u64)>>,
}

impl<F: PrimeField, C: MulCode<F> + Sync> FileBackedChunkedMulCodeDr1csNpFlpcpSparse<F, C> {
    fn new(fb: FileBackedSparseDr1csInstance<F>, l: usize, code: C) -> Result<Self, String> {
        if l > fb.nvars {
            return Err("file-backed flpcp: l > nvars".to_string());
        }
        let ckpts = Arc::new(Self::load_ckpts(&fb.layout.dir));
        Ok(Self { fb, l, code, ckpts })
    }

    #[inline]
    fn k(&self) -> usize {
        self.code.dim_k()
    }

    #[inline]
    fn nconstraints(&self) -> u64 {
        self.fb.layout.nconstraints
    }

    #[inline]
    fn blocks_u64(&self) -> u64 {
        let k = self.k().max(1) as u64;
        let n = self.nconstraints();
        (n + k - 1) / k
    }

    fn decode_block_idx(&self, idx: usize) -> Result<(usize, usize), String> {
        let ell = self.code.len_l();
        if ell == 0 {
            return Err("ell=0".to_string());
        }
        let block_id = idx / ell;
        let local_idx = idx % ell;
        if block_id >= self.blocks() {
            return Err("bad block id".to_string());
        }
        Ok((block_id, local_idx))
    }

    fn load_ckpts(dir: &std::path::Path) -> Vec<(u64, u64, u64, u64)> {
        use std::fs::File;
        use std::io::{BufReader, Read as IoRead};

        fn read_u64(r: &mut impl IoRead) -> Option<u64> {
            let mut buf = [0u8; 8];
            r.read_exact(&mut buf).ok()?;
            Some(u64::from_le_bytes(buf))
        }

        let p = dir.join("rows_ckpt.bin");
        let f = match File::open(&p) {
            Ok(f) => f,
            Err(_) => return Vec::new(),
        };
        let mut r = BufReader::with_capacity(8 * 1024 * 1024, f);
        let mut out = Vec::new();
        loop {
            let row_idx = match read_u64(&mut r) {
                Some(v) => v,
                None => break,
            };
            let a0 = read_u64(&mut r).unwrap_or(0);
            let b0 = read_u64(&mut r).unwrap_or(0);
            let c0 = read_u64(&mut r).unwrap_or(0);
            out.push((row_idx, a0, b0, c0));
        }
        out
    }

    fn ckpt_lookup(ckpts: &[(u64, u64, u64, u64)], row: u64) -> (u64, u64, u64, u64) {
        if ckpts.is_empty() {
            return (0, 0, 0, 0);
        }
        let mut lo = 0usize;
        let mut hi = ckpts.len();
        while lo + 1 < hi {
            let mid = (lo + hi) / 2;
            if ckpts[mid].0 <= row {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        if ckpts[lo].0 > row {
            (0, 0, 0, 0)
        } else {
            ckpts[lo]
        }
    }

    fn open_readers_at_row(
        &self,
        row_start: u64,
    ) -> Result<
        (
            std::io::BufReader<std::fs::File>,
            std::io::BufReader<std::fs::File>,
            std::io::BufReader<std::fs::File>,
            std::io::BufReader<std::fs::File>,
            std::io::BufReader<std::fs::File>,
            std::io::BufReader<std::fs::File>,
            std::io::BufReader<std::fs::File>,
        ),
        String,
    > {
        use std::fs::File;
        use std::io::{BufReader, Read as IoRead, Seek, SeekFrom};

        const ROW_LENS_SIZE: u64 = 12;
        fn read_u32(r: &mut impl IoRead) -> Result<u32, String> {
            let mut buf = [0u8; 4];
            r.read_exact(&mut buf).map_err(|e| e.to_string())?;
            Ok(u32::from_le_bytes(buf))
        }

        let dir = &self.fb.layout.dir;
        let (row0, a0, b0, c0) = Self::ckpt_lookup(self.ckpts.as_slice(), row_start);

        let mut fr = File::open(dir.join("constraints.bin")).map_err(|e| e.to_string())?;
        fr.seek(SeekFrom::Start(row0.saturating_mul(ROW_LENS_SIZE)))
            .map_err(|e| e.to_string())?;
        let mut rows = BufReader::with_capacity(8 * 1024 * 1024, fr);

        let mut fa_c = File::open(dir.join("a_coeffs.bin")).map_err(|e| e.to_string())?;
        let mut fa_i = File::open(dir.join("a_idx.bin")).map_err(|e| e.to_string())?;
        let mut fb_c = File::open(dir.join("b_coeffs.bin")).map_err(|e| e.to_string())?;
        let mut fb_i = File::open(dir.join("b_idx.bin")).map_err(|e| e.to_string())?;
        let mut fc_c = File::open(dir.join("c_coeffs.bin")).map_err(|e| e.to_string())?;
        let mut fc_i = File::open(dir.join("c_idx.bin")).map_err(|e| e.to_string())?;

        fa_c.seek(SeekFrom::Start(a0.saturating_mul(2))).map_err(|e| e.to_string())?;
        fa_i.seek(SeekFrom::Start(a0.saturating_mul(4))).map_err(|e| e.to_string())?;
        fb_c.seek(SeekFrom::Start(b0.saturating_mul(2))).map_err(|e| e.to_string())?;
        fb_i.seek(SeekFrom::Start(b0.saturating_mul(4))).map_err(|e| e.to_string())?;
        fc_c.seek(SeekFrom::Start(c0.saturating_mul(2))).map_err(|e| e.to_string())?;
        fc_i.seek(SeekFrom::Start(c0.saturating_mul(4))).map_err(|e| e.to_string())?;

        let mut a_coeffs = BufReader::with_capacity(8 * 1024 * 1024, fa_c);
        let mut a_idx = BufReader::with_capacity(8 * 1024 * 1024, fa_i);
        let mut b_coeffs = BufReader::with_capacity(8 * 1024 * 1024, fb_c);
        let mut b_idx = BufReader::with_capacity(8 * 1024 * 1024, fb_i);
        let mut c_coeffs = BufReader::with_capacity(8 * 1024 * 1024, fc_c);
        let mut c_idx = BufReader::with_capacity(8 * 1024 * 1024, fc_i);

        // Advance from row0 to row_start.
        //
        // Use seeks for term pools: we only need to advance file cursors, not decode values.
        for _ in row0..row_start {
            let a_len = read_u32(&mut rows)? as u64;
            let b_len = read_u32(&mut rows)? as u64;
            let c_len = read_u32(&mut rows)? as u64;
            if a_len != 0 {
                a_coeffs
                    .seek(SeekFrom::Current((a_len.saturating_mul(2)) as i64))
                    .map_err(|e| e.to_string())?;
                a_idx
                    .seek(SeekFrom::Current((a_len.saturating_mul(4)) as i64))
                    .map_err(|e| e.to_string())?;
            }
            if b_len != 0 {
                b_coeffs
                    .seek(SeekFrom::Current((b_len.saturating_mul(2)) as i64))
                    .map_err(|e| e.to_string())?;
                b_idx
                    .seek(SeekFrom::Current((b_len.saturating_mul(4)) as i64))
                    .map_err(|e| e.to_string())?;
            }
            if c_len != 0 {
                c_coeffs
                    .seek(SeekFrom::Current((c_len.saturating_mul(2)) as i64))
                    .map_err(|e| e.to_string())?;
                c_idx
                    .seek(SeekFrom::Current((c_len.saturating_mul(4)) as i64))
                    .map_err(|e| e.to_string())?;
            }
        }

        Ok((rows, a_coeffs, a_idx, b_coeffs, b_idx, c_coeffs, c_idx))
    }
}

impl<F: PrimeField, C: MulCode<F> + Sync> Dr1csNpFlpcpSparseApi<F>
    for FileBackedChunkedMulCodeDr1csNpFlpcpSparse<F, C>
{
    fn n(&self) -> usize {
        self.l
    }
    fn n_total(&self) -> usize {
        self.fb.nvars
    }
    fn z_w_len(&self) -> usize {
        self.fb.nvars - self.l
    }
    fn m(&self) -> usize {
        self.z_w_len() + (self.k_star() * self.blocks())
    }
    fn ell(&self) -> usize {
        self.ell_local() * self.blocks()
    }
    fn blocks(&self) -> usize {
        self.blocks_u64() as usize
    }
    fn ell_local(&self) -> usize {
        self.code.len_l()
    }
    fn k_star(&self) -> usize {
        self.code.dim_k_star()
    }
    fn witness_positions_star(&self) -> Result<Vec<usize>, String> {
        self.code.witness_positions_star()
    }

    fn stream_w_eval_blocks(
        &self,
        witness_pos: &[usize],
        x: &[F],
        z_w: &[F],
        x_u16: Option<&[u16]>,
        z_u16: Option<&[u16]>,
        on_block: &mut dyn FnMut(usize, &[F]),
    ) -> Result<(), String> {
        use std::io::{Read as IoRead, Seek, SeekFrom};

        fn read_u32(r: &mut impl IoRead) -> Result<u32, String> {
            let mut buf = [0u8; 4];
            r.read_exact(&mut buf).map_err(|e| e.to_string())?;
            Ok(u32::from_le_bytes(buf))
        }
        fn read_u16(r: &mut impl IoRead) -> Result<u16, String> {
            let mut buf = [0u8; 2];
            r.read_exact(&mut buf).map_err(|e| e.to_string())?;
            Ok(u16::from_le_bytes(buf))
        }

        if x.len() != self.l {
            return Err("stream_w_eval_blocks: bad x length".to_string());
        }
        if z_w.len() != self.z_w_len() {
            return Err("stream_w_eval_blocks: bad z_w length".to_string());
        }
        if witness_pos.len() != self.k_star() {
            return Err("stream_w_eval_blocks: witness positions length mismatch".to_string());
        }

        let k = self.k();
        let k_star = self.k_star();
        let blocks = self.blocks();
        let nconstraints = self.nconstraints();
        let f257_fast = x_u16.is_some() && z_u16.is_some() && is_f257_field::<F>();
        let (x_u16, z_u16) = if f257_fast {
            (x_u16.unwrap(), z_u16.unwrap())
        } else {
            (&[][..], &[][..])
        };

        // Coefficient lookup table for F257: avoid repeated `F::from(u16)`.
        let coeff_lut: Option<Vec<F>> = if is_f257_field::<F>() {
            Some((0u16..=256u16).map(|c| F::from(c as u64)).collect())
        } else {
            None
        };

        // Parallelize across blocks, but preserve in-order streaming output.
        //
        // We process bounded "windows" of blocks in parallel and then emit them sequentially
        // to the callback to preserve proof element order (required by streaming decap).
        let threads = rayon::current_num_threads().max(1);
        let window = (threads * 4).clamp(32, 512);

        let mut b0 = 0usize;
        while b0 < blocks {
            let b1 = (b0 + window).min(blocks);
            let out: Vec<Vec<F>> = (b0..b1)
                .into_par_iter()
                .map(|b| -> Result<Vec<F>, String> {
                    // If this block is past the end, it is all-zero (keeps proof length consistent).
                    let row_start = (b as u64).saturating_mul(k as u64);
                    if row_start >= nconstraints {
                        return Ok(vec![F::ZERO; k_star]);
                    }

                    let (mut rows, mut a_coeffs, mut a_idx, mut b_coeffs, mut b_idx, mut c_coeffs, mut c_idx) =
                        self.open_readers_at_row(row_start)?;

                    let mut y_a = vec![F::ZERO; k];
                    let mut y_b = vec![F::ZERO; k];
                    for i in 0..k {
                        let row = row_start.saturating_add(i as u64);
                        if row >= nconstraints {
                            break;
                        }
                        let a_len = read_u32(&mut rows)? as usize;
                        let b_len = read_u32(&mut rows)? as usize;
                        let c_len = read_u32(&mut rows)? as usize;

                        let (aval, bval) = if f257_fast {
                            // Work entirely mod 257 in integers.
                            const P: u64 = 257;
                            let mut aval_u: u64 = 0;
                            for _ in 0..a_len {
                                let cu16 = read_u16(&mut a_coeffs)? as u64;
                                let idx = read_u32(&mut a_idx)? as usize;
                                let v = if idx < self.l {
                                    x_u16[idx] as u64
                                } else {
                                    z_u16[idx - self.l] as u64
                                };
                                aval_u = aval_u.wrapping_add(cu16.wrapping_mul(v));
                            }
                            let mut bval_u: u64 = 0;
                            for _ in 0..b_len {
                                let cu16 = read_u16(&mut b_coeffs)? as u64;
                                let idx = read_u32(&mut b_idx)? as usize;
                                let v = if idx < self.l {
                                    x_u16[idx] as u64
                                } else {
                                    z_u16[idx - self.l] as u64
                                };
                                bval_u = bval_u.wrapping_add(cu16.wrapping_mul(v));
                            }
                            (F::from((aval_u % P) as u64), F::from((bval_u % P) as u64))
                        } else {
                            let lut = coeff_lut.as_deref();
                            let mut aval = F::ZERO;
                            for _ in 0..a_len {
                                let cu16 = read_u16(&mut a_coeffs)? as usize;
                                let idx = read_u32(&mut a_idx)? as usize;
                                let v = if idx < self.l { x[idx] } else { z_w[idx - self.l] };
                                let c = if let Some(lut) = lut {
                                    lut[cu16]
                                } else {
                                    F::from(cu16 as u64)
                                };
                                aval += c * v;
                            }
                            let mut bval = F::ZERO;
                            for _ in 0..b_len {
                                let cu16 = read_u16(&mut b_coeffs)? as usize;
                                let idx = read_u32(&mut b_idx)? as usize;
                                let v = if idx < self.l { x[idx] } else { z_w[idx - self.l] };
                                let c = if let Some(lut) = lut {
                                    lut[cu16]
                                } else {
                                    F::from(cu16 as u64)
                                };
                                bval += c * v;
                            }
                            (aval, bval)
                        };

                        // Skip C term pools (not used for w_eval) to keep readers aligned.
                        if c_len != 0 {
                            c_coeffs
                                .seek(SeekFrom::Current(((c_len as u64).saturating_mul(2)) as i64))
                                .map_err(|e| e.to_string())?;
                            c_idx
                                .seek(SeekFrom::Current(((c_len as u64).saturating_mul(4)) as i64))
                                .map_err(|e| e.to_string())?;
                        }
                        y_a[i] = aval;
                        y_b[i] = bval;
                    }

                    let ea = self.code.eval_e_at_positions(witness_pos, &y_a)?;
                    let eb = self.code.eval_e_at_positions(witness_pos, &y_b)?;
                    if ea.len() != k_star || eb.len() != k_star {
                        return Err("stream_w_eval_blocks: bad eval length".to_string());
                    }
                    let mut w_eval = vec![F::ZERO; k_star];
                    if k_star >= 256 {
                        w_eval
                            .par_iter_mut()
                            .enumerate()
                            .for_each(|(j, out)| *out = ea[j] * eb[j]);
                    } else {
                        for j in 0..k_star {
                            w_eval[j] = ea[j] * eb[j];
                        }
                    }
                    Ok(w_eval)
                })
                .collect::<Result<Vec<_>, _>>()?;

            for (i, w_eval) in out.iter().enumerate() {
                on_block(b0 + i, w_eval.as_slice());
            }
            b0 = b1;
        }

        Ok(())
    }

    fn stream_queries_for_coins_sparse(
        &self,
        idx: usize,
        lambda: F,
        x: &[F],
        scratch: &mut Dr1csQueryScratch<F>,
        sink: &mut dyn QuerySink<F>,
    ) -> Result<(), String> {
        use std::io::{Read as IoRead};

        fn read_u32(r: &mut impl IoRead) -> Result<u32, String> {
            let mut buf = [0u8; 4];
            r.read_exact(&mut buf).map_err(|e| e.to_string())?;
            Ok(u32::from_le_bytes(buf))
        }
        fn read_u16(r: &mut impl IoRead) -> Result<u16, String> {
            let mut buf = [0u8; 2];
            r.read_exact(&mut buf).map_err(|e| e.to_string())?;
            Ok(u16::from_le_bytes(buf))
        }

        if x.len() != self.l {
            return Err("stream_queries_for_coins_sparse: bad x length".to_string());
        }
        let (block_id, local_idx) = self.decode_block_idx(idx)?;
        let k = self.k();
        let k_star = self.k_star();
        let z_w_len = self.z_w_len();
        let base = self.l + z_w_len;
        let block_offset = base + (block_id * k_star);

        scratch.clear_all();

        // Coefficients for A/B rows (length k).
        let mut coeff_ab = vec![F::ZERO; k];
        self.code.row_e_stream(local_idx, &mut |i, c| {
            if i < k {
                coeff_ab[i] = c;
            }
        })?;

        // Coefficients for C low-cube rows (first k entries) + emit q3 on witness coords.
        let mut coeff_c = vec![F::ZERO; k];
        self.code.row_e_star_stream(local_idx, &mut |j, c| {
            if j < k {
                coeff_c[j] = c;
            }
            let coeff = if j < k { c - (lambda * c) } else { c };
            if !coeff.is_zero() {
                sink.on_q3(coeff, block_offset + j);
            }
        })?;

        let row_start = (block_id as u64).saturating_mul(k as u64);
        let nconstraints = self.nconstraints();
        let (mut rows, mut a_coeffs, mut a_idx, mut b_coeffs, mut b_idx, mut c_coeffs, mut c_idx) =
            self.open_readers_at_row(row_start)?;

        // Coefficient lookup table (tiny format): avoid repeated `F::from(u16)` in hot loops.
        let coeff_lut: Option<Vec<F>> = if is_f257_field::<F>() {
            Some((0u16..=256u16).map(|c| F::from(c as u64)).collect())
        } else {
            None
        };

        for i in 0..k {
            let row = row_start.saturating_add(i as u64);
            if row >= nconstraints {
                continue;
            }
            let a_len = read_u32(&mut rows)? as usize;
            let b_len = read_u32(&mut rows)? as usize;
            let c_len = read_u32(&mut rows)? as usize;

            let cab = coeff_ab[i];
            for _ in 0..a_len {
                let cu16 = read_u16(&mut a_coeffs)?;
                let vidx = read_u32(&mut a_idx)? as usize;
                if !cab.is_zero() {
                    let c = if let Some(lut) = coeff_lut.as_deref() {
                        lut[cu16 as usize]
                    } else {
                        F::from(cu16 as u64)
                    };
                    scratch.add_q1_term_on_z(vidx, c * cab);
                }
            }
            for _ in 0..b_len {
                let cu16 = read_u16(&mut b_coeffs)?;
                let vidx = read_u32(&mut b_idx)? as usize;
                if !cab.is_zero() {
                    let c = if let Some(lut) = coeff_lut.as_deref() {
                        lut[cu16 as usize]
                    } else {
                        F::from(cu16 as u64)
                    };
                    scratch.add_q2_term_on_z(vidx, c * cab);
                }
            }
            let cc = coeff_c[i];
            for _ in 0..c_len {
                let cu16 = read_u16(&mut c_coeffs)?;
                let vidx = read_u32(&mut c_idx)? as usize;
                if !cc.is_zero() {
                    let c = if let Some(lut) = coeff_lut.as_deref() {
                        lut[cu16 as usize]
                    } else {
                        F::from(cu16 as u64)
                    };
                    scratch.add_q3_cx2_term_on_z(vidx, c * cc);
                }
            }
        }

        for (vidx, c) in scratch.take_q1_terms_on_z().into_iter() {
            let (is_pub, j) = if vidx < self.l { (true, vidx) } else { (false, vidx - self.l) };
            let v_idx = if is_pub { j } else { self.l + j };
            sink.on_q1(c, v_idx);
        }
        for (vidx, c) in scratch.take_q2_terms_on_z().into_iter() {
            let (is_pub, j) = if vidx < self.l { (true, vidx) } else { (false, vidx - self.l) };
            let v_idx = if is_pub { j } else { self.l + j };
            sink.on_q2(c, v_idx);
        }
        for (vidx, c) in scratch.take_q3_cx2_terms_on_z().into_iter() {
            let cc = lambda * c;
            if cc.is_zero() {
                continue;
            }
            let (is_pub, j) = if vidx < self.l { (true, vidx) } else { (false, vidx - self.l) };
            let v_idx = if is_pub { j } else { self.l + j };
            sink.on_q3(cc, v_idx);
        }

        Ok(())
    }

    fn prove(&self, x: &[F], z_w: &[F]) -> Vec<F> {
        let mut pi = Vec::with_capacity(self.m());
        pi.extend_from_slice(z_w);
        let witness_pos = self.witness_positions_star().expect("witness_positions_star");
        self.stream_w_eval_blocks(&witness_pos, x, z_w, None, None, &mut |_, w_eval| {
            pi.extend_from_slice(w_eval);
        })
        .expect("stream_w_eval_blocks");
        pi
    }

    fn queries_for_coins_sparse(
        &self,
        idx: usize,
        lambda: F,
        x: &[F],
    ) -> Result<(Vec<SparseVec<F>>, FlpcpPredicate<F>), String> {
        struct VecSink<F: PrimeField> {
            q1: Vec<(F, usize)>,
            q2: Vec<(F, usize)>,
            q3: Vec<(F, usize)>,
        }
        impl<F: PrimeField> QuerySink<F> for VecSink<F> {
            fn on_q1(&mut self, coeff: F, idx: usize) {
                self.q1.push((coeff, idx));
            }
            fn on_q2(&mut self, coeff: F, idx: usize) {
                self.q2.push((coeff, idx));
            }
            fn on_q3(&mut self, coeff: F, idx: usize) {
                self.q3.push((coeff, idx));
            }
        }
        let mut sink = VecSink { q1: Vec::new(), q2: Vec::new(), q3: Vec::new() };
        let mut scratch = Dr1csQueryScratch::<F>::new(self.n_total());
        self.stream_queries_for_coins_sparse(idx, lambda, x, &mut scratch, &mut sink)?;
        Ok((
            vec![SparseVec::new(sink.q1), SparseVec::new(sink.q2), SparseVec::new(sink.q3)],
            FlpcpPredicate::MulEq,
        ))
    }
}

/// Extract the public coins from a lock artifact (convenience).
pub fn public_coins<F: PrimeField>(art: &Theorem43LockArtifact<F>) -> Theorem43Coins<F> {
    art.coins.clone()
}

// NOTE: test-only helpers live in the test module.

/// Arm a Theorem-4.3 tiny-field lock and wrap it in a Ring-LWE backend.
///
/// This produces a compact public lock artifact that does not reveal the hidden query.
pub(crate) fn arm_theorem43_ringlwe_from_statement<F: PrimeField>(
    dpp: &Theorem43Dpp<F, impl Dr1csNpFlpcpSparseApi<F> + Sync>,
    stmt_digest: [u8; 32],
    x: &[F],
    armer_seed: [u8; 32],
    lock_j: u64,
    block_id: usize,
    rep_id: u64,
    params: RingLweParams,
    rng: &mut impl rand::RngCore,
    scratch: &mut Dr1csQueryScratch<F>,
    acc: &mut QueryBlockAccumulator,
) -> Result<(RingLweLockArtifact<F>, Theorem43LockArtifact<F>), String> {
    let c_stmt = crate::we_statement::digest32_to_bits_field::<F>(stmt_digest);
    let armer_secret = crate::we_statement::derive_armer_secret::<F>(armer_seed, stmt_digest, lock_j, 4);
    let art = dpp.arm(&c_stmt, x, &armer_secret, block_id, rep_id)?;
    let pi_len = dpp.proof_len();
    let mut err: Option<String> = None;
    let offset_f = dpp
        .stream_query_terms_for_pi(x, &art.coins, &art.coeffs, scratch, &mut |pi_idx, coeff| {
            if err.is_some() {
                return;
            }
            if let Err(e) = acc.add_term(&coeff, pi_idx) {
                err = Some(e);
            }
        })?;
    if let Some(e) = err {
        return Err(e);
    }
    let q_blocks = acc.into_sparse_blocks();
    let lock = arm_ringlwe_lock(
        c_stmt,
        art.accepting_set,
        art.coins.clone(),
        offset_f,
        x.len(),
        pi_len,
        q_blocks,
        params,
        rng,
    )?;
    Ok((lock, art))
}

/// Streaming arming helper that keeps the chunked FLPCP backend available for proof streaming.
pub struct WeRingLweStreamingContext<F: PrimeField + FftField> {
    pub lock: RingLweLockArtifact<F>,
    theorem43_art: Theorem43LockArtifact<F>,
    dpp: Theorem43Dpp<F, FileBackedChunkedMulCodeDr1csNpFlpcpSparse<F, TensorRsMulCode<F>>>,
}

impl<F: PrimeField + FftField> WeRingLweStreamingContext<F> {
    pub fn stream_pi0_and_collect_tails(
        &self,
        x: &[F],
        z_w: &[F],
        coins_list: &[Theorem43Coins<F>],
        on_pi0_chunk: &mut dyn FnMut(&[F]),
    ) -> Result<Vec<Vec<F>>, String> {
        self.dpp
            .stream_pi0_and_collect_tails(x, z_w, coins_list, on_pi0_chunk)
    }

    pub fn proof_len(&self) -> usize {
        self.dpp.proof_len()
    }

    /// Compute the DPP answer from an already-streamed `(π0, tail)` split proof.
    ///
    /// This is the canonical verification path for Theorem 4.3, and is useful in tests
    /// to cross-check the Ring-LWE streaming decapsulation output without exposing any
    /// additional public API surface.
    pub fn answer_from_pi0_and_tail(&self, x: &[F], pi0: &[F], tail: &[F]) -> Result<F, String> {
        self.dpp.answer_from_pi0_and_tail(&self.theorem43_art, x, pi0, tail)
    }
}

/// Arm a Ring-LWE lock and return a streaming context for chunked proof generation.
pub(crate) fn arm_we_ringlwe_from_dr1cs_streaming<F: PrimeField + FftField>(
    dr1cs: FileBackedSparseDr1csInstance<F>,
    public_len: usize,
    stmt_digest: [u8; 32],
    x: &[F],
    armer_seed: [u8; 32],
    lock_j: u64,
    block_id: usize,
    rep_id: u64,
    params: RingLweParams,
    rng: &mut impl rand::RngCore,
) -> Result<WeRingLweStreamingContext<F>, String> {
    if x.len() != public_len {
        return Err("arm_we_ringlwe_from_dr1cs_streaming: x length != public_len".to_string());
    }
    let code = TensorRsMulCode::<F>::new(48, 3)?;
    let flpcp = FileBackedChunkedMulCodeDr1csNpFlpcpSparse::<F, _>::new(dr1cs, public_len, code)?;
    let dpp = Theorem43Dpp::<F, _>::new(flpcp)?;
    let mut scratch = dpp.query_scratch();
    let mut acc = QueryBlockAccumulator::new(dpp.proof_len())?;
    let (lock, theorem43_art) = arm_theorem43_ringlwe_from_statement(
        &dpp,
        stmt_digest,
        x,
        armer_seed,
        lock_j,
        block_id,
        rep_id,
        params,
        rng,
        &mut scratch,
        &mut acc,
    )?;
    Ok(WeRingLweStreamingContext {
        lock,
        theorem43_art,
        dpp,
    })
}

/// Arm the **LF+ tiny-field WE gate** (Poseidon(F257) + CM-coin surfaces) as a Theorem-4.3 lock.
///
/// This is the main wiring from:
/// - `we_gate_arith::build_we_dr1cs_for_plus_proof_shape_tiny` (arm-time instance construction)
/// into:
/// - `arm_we_ringlwe_from_dr1cs_streaming` (Theorem-4.3 + Ring-LWE wrapper).
///
/// Notes:
/// - The statement binding is carried by `stmt_digest` via `c_stmt` inside `arm_theorem43_from_statement`.
/// - `x` is the canonical public statement encoding:
///   `x = [ONE] || [10×WeParams] || [public_inputs...]`.
#[cfg(feature = "we_gate")]
pub(crate) fn arm_lfplus_we_gate_tiny_ringlwe_streaming<R>(
    shape: we_gate_arith::WeDr1csShape<F257>,
    params: &WeParams,
    public_inputs: &[F257],
    stmt_digest: [u8; 32],
    armer_seed: [u8; 32],
    lock_j: u64,
    block_id: usize,
    rep_id: u64,
    ringlwe_params: RingLweParams,
    rng: &mut impl rand::RngCore,
) -> Result<WeRingLweStreamingContext<F257>, String>
where
    R: OverField + CoeffRing + PolyRing,
    R::BaseRing: Zq + ark_ff::Field + ark_ff::PrimeField,
{
    let x = crate::we_statement::encode_public_x::<F257>(params, public_inputs);
    if x.len() != shape.public_len {
        return Err(format!(
            "arm_lfplus_we_gate_tiny_ringlwe_streaming: public_len mismatch (shape={} vs x={})",
            shape.public_len,
            x.len()
        ));
    }

    arm_we_ringlwe_from_dr1cs_streaming::<F257>(
        shape.inst,
        shape.public_len,
        stmt_digest,
        &x,
        armer_seed,
        lock_j,
        block_id,
        rep_id,
        ringlwe_params,
        rng,
    )
}
