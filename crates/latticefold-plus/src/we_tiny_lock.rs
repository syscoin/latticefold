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
use symphony::file_backed_dr1cs::{cfg_read_buf_bytes, FileBackedSparseDr1csInstance};

use crate::lockable_ringlwe::QueryBlockAccumulator;
use crate::lockable_ringlwe::{
    arm_ringlwe_lock, ratio_class_mod257_u16, sample_nonzero_f257_scalar, RingLweLockArtifact,
    RingLweParams, RingLweSubLock,
};

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
        let mut r = BufReader::with_capacity(cfg_read_buf_bytes(), f);
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
        let cap = cfg_read_buf_bytes();
        let mut rows = BufReader::with_capacity(cap, fr);

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

        let mut a_coeffs = BufReader::with_capacity(cap, fa_c);
        let mut a_idx = BufReader::with_capacity(cap, fa_i);
        let mut b_coeffs = BufReader::with_capacity(cap, fb_c);
        let mut b_idx = BufReader::with_capacity(cap, fb_i);
        let mut c_coeffs = BufReader::with_capacity(cap, fc_c);
        let mut c_idx = BufReader::with_capacity(cap, fc_i);

        // Advance from row0 to row_start.
        //
        // IMPORTANT: avoid per-row seeking (very expensive). We only need to advance file cursors,
        // so we sum the term counts and do a single seek per file.
        //
        // The constraints row-lengths file is fixed-width, so we advance it by decoding u32 triplets.
        // The term pools are contiguous, so total term-count deltas suffice.
        let mut a_skip: u64 = 0;
        let mut b_skip: u64 = 0;
        let mut c_skip: u64 = 0;
        for _ in row0..row_start {
            let a_len = read_u32(&mut rows)? as u64;
            let b_len = read_u32(&mut rows)? as u64;
            let c_len = read_u32(&mut rows)? as u64;
            a_skip = a_skip.saturating_add(a_len);
            b_skip = b_skip.saturating_add(b_len);
            c_skip = c_skip.saturating_add(c_len);
        }
        if a_skip != 0 {
            a_coeffs
                .seek(SeekFrom::Current((a_skip.saturating_mul(2)) as i64))
                .map_err(|e| e.to_string())?;
            a_idx
                .seek(SeekFrom::Current((a_skip.saturating_mul(4)) as i64))
                .map_err(|e| e.to_string())?;
        }
        if b_skip != 0 {
            b_coeffs
                .seek(SeekFrom::Current((b_skip.saturating_mul(2)) as i64))
                .map_err(|e| e.to_string())?;
            b_idx
                .seek(SeekFrom::Current((b_skip.saturating_mul(4)) as i64))
                .map_err(|e| e.to_string())?;
        }
        if c_skip != 0 {
            c_coeffs
                .seek(SeekFrom::Current((c_skip.saturating_mul(2)) as i64))
                .map_err(|e| e.to_string())?;
            c_idx
                .seek(SeekFrom::Current((c_skip.saturating_mul(4)) as i64))
                .map_err(|e| e.to_string())?;
        }

        Ok((rows, a_coeffs, a_idx, b_coeffs, b_idx, c_coeffs, c_idx))
    }

    fn open_readers_ab_at_row(
        &self,
        row_start: u64,
    ) -> Result<
        (
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
        let (row0, a0, b0, _c0) = Self::ckpt_lookup(self.ckpts.as_slice(), row_start);

        let mut fr = File::open(dir.join("constraints.bin")).map_err(|e| e.to_string())?;
        fr.seek(SeekFrom::Start(row0.saturating_mul(ROW_LENS_SIZE)))
            .map_err(|e| e.to_string())?;
        let cap = cfg_read_buf_bytes();
        let mut rows = BufReader::with_capacity(cap, fr);

        let mut fa_c = File::open(dir.join("a_coeffs.bin")).map_err(|e| e.to_string())?;
        let mut fa_i = File::open(dir.join("a_idx.bin")).map_err(|e| e.to_string())?;
        let mut fb_c = File::open(dir.join("b_coeffs.bin")).map_err(|e| e.to_string())?;
        let mut fb_i = File::open(dir.join("b_idx.bin")).map_err(|e| e.to_string())?;

        fa_c.seek(SeekFrom::Start(a0.saturating_mul(2))).map_err(|e| e.to_string())?;
        fa_i.seek(SeekFrom::Start(a0.saturating_mul(4))).map_err(|e| e.to_string())?;
        fb_c.seek(SeekFrom::Start(b0.saturating_mul(2))).map_err(|e| e.to_string())?;
        fb_i.seek(SeekFrom::Start(b0.saturating_mul(4))).map_err(|e| e.to_string())?;

        let mut a_coeffs = BufReader::with_capacity(cap, fa_c);
        let mut a_idx = BufReader::with_capacity(cap, fa_i);
        let mut b_coeffs = BufReader::with_capacity(cap, fb_c);
        let mut b_idx = BufReader::with_capacity(cap, fb_i);

        // Advance from row0 to row_start. Sum term counts and seek once per file.
        let mut a_skip: u64 = 0;
        let mut b_skip: u64 = 0;
        for _ in row0..row_start {
            let a_len = read_u32(&mut rows)? as u64;
            let b_len = read_u32(&mut rows)? as u64;
            let _c_len = read_u32(&mut rows)? as u64;
            a_skip = a_skip.saturating_add(a_len);
            b_skip = b_skip.saturating_add(b_len);
        }
        if a_skip != 0 {
            a_coeffs
                .seek(SeekFrom::Current((a_skip.saturating_mul(2)) as i64))
                .map_err(|e| e.to_string())?;
            a_idx
                .seek(SeekFrom::Current((a_skip.saturating_mul(4)) as i64))
                .map_err(|e| e.to_string())?;
        }
        if b_skip != 0 {
            b_coeffs
                .seek(SeekFrom::Current((b_skip.saturating_mul(2)) as i64))
                .map_err(|e| e.to_string())?;
            b_idx
                .seek(SeekFrom::Current((b_skip.saturating_mul(4)) as i64))
                .map_err(|e| e.to_string())?;
        }

        Ok((rows, a_coeffs, a_idx, b_coeffs, b_idx))
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
        use std::io::Read as IoRead;

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
        let f257_fast = x_u16.is_some() && z_u16.is_some();
        if !f257_fast {
            return Err("stream_w_eval_blocks: only F257 fast path is supported".to_string());
        }
        let (x_u16, z_u16) = (x_u16.unwrap(), z_u16.unwrap());

        let do_prof_w_eval = std::env::var("LF_PROFILE_DPP_W_EVAL").ok().as_deref() == Some("1");
        let open_ns = std::sync::atomic::AtomicU64::new(0);
        let eval_ns = std::sync::atomic::AtomicU64::new(0);
        let mul_ns = std::sync::atomic::AtomicU64::new(0);
        let blocks_done = std::sync::atomic::AtomicU64::new(0);

        // Parallelize across blocks, but preserve in-order streaming output.
        //
        // We process bounded "windows" of blocks in parallel and then emit them sequentially
        // to the callback to preserve proof element order (required by streaming decap).
        let threads = rayon::current_num_threads().max(1);
        let mut window = (threads * 4).clamp(8, 512);
        // Cap the in-flight window to avoid materializing too much `(window × k_star)` output.
        // This is crucial when k_star is large (e.g. Tensor-RS rank=3).
        let bytes_per_block = (k_star as u64).saturating_mul(std::mem::size_of::<F>() as u64);
        let max_window_bytes: u64 = std::env::var("LF_DPP_WINDOW_MAX_MB")
            .ok()
            .and_then(|s| s.parse::<u64>().ok())
            // Default: allow a few GiB of in-flight w_eval blocks. On realistic params
            // (k_star ~ 857k, F257 ~ 8 bytes) one block is ~6.8MiB, so 96 blocks is ~650MiB.
            .unwrap_or(2048)
            .saturating_mul(1024 * 1024);
        if bytes_per_block != 0 {
            let max_by_mem = (max_window_bytes / bytes_per_block).max(1) as usize;
            window = window.min(max_by_mem);
        }

        // Always process *contiguous* block ranges per rayon task, reusing a single set of open
        // readers + internal buffers. This significantly reduces `open+seek` overhead for
        // large instances and avoids nested-parallel overheads in the inner loops.
        let blocks_per_task: usize = std::env::var("LF_DPP_BLOCKS_PER_TASK")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(4)
            .max(1);

        let mut b0 = 0usize;
        while b0 < blocks {
            let b1 = (b0 + window).min(blocks);

            let mut ranges: Vec<(usize, usize)> = Vec::new();
            let mut s = b0;
            while s < b1 {
                let e = (s + blocks_per_task).min(b1);
                ranges.push((s, e));
                s = e;
            }

            let out_chunks: Vec<Vec<F>> = ranges
                .into_par_iter()
                .map_init(
                    || (vec![F::ZERO; k], vec![F::ZERO; k], vec![F::ZERO; k_star], vec![F::ZERO; k_star]),
                    |(y_a, y_b, ea_buf, eb_buf), (bs, be)| -> Result<Vec<F>, String> {
                        let n_blocks = be.saturating_sub(bs);
                        let mut out = vec![F::ZERO; n_blocks.saturating_mul(k_star)];
                        let row_start0 = (bs as u64).saturating_mul(k as u64);
                        if row_start0 >= nconstraints {
                            return Ok(out);
                        }
                        let t_open = std::time::Instant::now();
                        let (mut rows, mut a_coeffs, mut a_idx, mut b_coeffs, mut b_idx) =
                            self.open_readers_ab_at_row(row_start0)?;
                        if do_prof_w_eval {
                            let dt = t_open.elapsed();
                            open_ns.fetch_add(
                                dt.as_nanos().min(u64::MAX as u128) as u64,
                                std::sync::atomic::Ordering::Relaxed,
                            );
                        }

                        for (bi, b) in (bs..be).enumerate() {
                            let row_start = (b as u64).saturating_mul(k as u64);
                            if row_start >= nconstraints {
                                break;
                            }
                            y_a.fill(F::ZERO);
                            y_b.fill(F::ZERO);
                            for i in 0..k {
                                let row = row_start.saturating_add(i as u64);
                                if row >= nconstraints {
                                    break;
                                }
                                let a_len = read_u32(&mut rows)? as usize;
                                let b_len = read_u32(&mut rows)? as usize;
                                let _c_len = read_u32(&mut rows)? as usize;

                                let (aval, bval) = {
                                    const P: u64 = 257;
                                    let mut aval_u: u64 = 0;
                                    for _ in 0..a_len {
                                        let cu16 = read_u16(&mut a_coeffs)? as u64;
                                        let idx = read_u32(&mut a_idx)? as usize;
                                        let v = if idx < self.l { x_u16[idx] as u64 } else { z_u16[idx - self.l] as u64 };
                                        aval_u = aval_u.wrapping_add(cu16.wrapping_mul(v));
                                    }
                                    let mut bval_u: u64 = 0;
                                    for _ in 0..b_len {
                                        let cu16 = read_u16(&mut b_coeffs)? as u64;
                                        let idx = read_u32(&mut b_idx)? as usize;
                                        let v = if idx < self.l { x_u16[idx] as u64 } else { z_u16[idx - self.l] as u64 };
                                        bval_u = bval_u.wrapping_add(cu16.wrapping_mul(v));
                                    }
                                    (F::from((aval_u % P) as u64), F::from((bval_u % P) as u64))
                                };
                                y_a[i] = aval;
                                y_b[i] = bval;
                            }

                            let t_eval = std::time::Instant::now();
                            self.code
                                .eval_e_at_positions_into(witness_pos, y_a.as_slice(), ea_buf.as_mut_slice())?;
                            self.code
                                .eval_e_at_positions_into(witness_pos, y_b.as_slice(), eb_buf.as_mut_slice())?;
                            if do_prof_w_eval {
                                let dt = t_eval.elapsed();
                                eval_ns.fetch_add(
                                    dt.as_nanos().min(u64::MAX as u128) as u64,
                                    std::sync::atomic::Ordering::Relaxed,
                                );
                            }
                            let dst = &mut out[bi * k_star..(bi + 1) * k_star];
                            let t_mul = std::time::Instant::now();
                            // keep this multiply *sequential* to avoid nested Rayon overhead.
                            for j in 0..k_star {
                                dst[j] = ea_buf[j] * eb_buf[j];
                            }
                            if do_prof_w_eval {
                                let dt = t_mul.elapsed();
                                mul_ns.fetch_add(
                                    dt.as_nanos().min(u64::MAX as u128) as u64,
                                    std::sync::atomic::Ordering::Relaxed,
                                );
                                blocks_done.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                            }
                        }

                        Ok(out)
                    },
                )
                .collect::<Result<Vec<_>, _>>()?;

            // Emit in-order.
            let mut b_emit = b0;
            for chunk in out_chunks.iter() {
                for blk in 0..(chunk.len() / k_star) {
                    let s0 = blk * k_star;
                    let s1 = s0 + k_star;
                    on_block(b_emit, &chunk[s0..s1]);
                    b_emit += 1;
                }
            }
            b0 = b1;
        }

        if do_prof_w_eval {
            let b = blocks_done.load(std::sync::atomic::Ordering::Relaxed).max(1);
            let open = open_ns.load(std::sync::atomic::Ordering::Relaxed);
            let eval = eval_ns.load(std::sync::atomic::Ordering::Relaxed);
            let mul = mul_ns.load(std::sync::atomic::Ordering::Relaxed);
            eprintln!(
                "[LF_PROFILE] dpp w_eval totals: blocks={} open={:.3}s ({:.3}ms/block) eval={:.3}s ({:.3}ms/block) mul={:.3}s ({:.3}ms/block)",
                b,
                (open as f64) * 1e-9,
                (open as f64) * 1e-6 / (b as f64),
                (eval as f64) * 1e-9,
                (eval as f64) * 1e-6 / (b as f64),
                (mul as f64) * 1e-9,
                (mul as f64) * 1e-6 / (b as f64),
            );
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
        let coeff_lut: Vec<F> = (0u16..=256u16).map(|c| F::from(c as u64)).collect();

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
                    let c = coeff_lut[cu16 as usize];
                    scratch.add_q1_term_on_z(vidx, c * cab);
                }
            }
            for _ in 0..b_len {
                let cu16 = read_u16(&mut b_coeffs)?;
                let vidx = read_u32(&mut b_idx)? as usize;
                if !cab.is_zero() {
                    let c = coeff_lut[cu16 as usize];
                    scratch.add_q2_term_on_z(vidx, c * cab);
                }
            }
            let cc = coeff_c[i];
            for _ in 0..c_len {
                let cu16 = read_u16(&mut c_coeffs)?;
                let vidx = read_u32(&mut c_idx)? as usize;
                if !cc.is_zero() {
                    let c = coeff_lut[cu16 as usize];
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

/// Prover-side streaming context: a thin wrapper around the chunked, file-backed FLPCP backend.
///
/// This is intentionally **not** returned from arming: arming publishes a public lock artifact;
/// proving is a separate role that may happen much later by a different party.
pub struct WeRingLweProverContext<F: PrimeField + FftField> {
    dpp: Theorem43Dpp<F, FileBackedChunkedMulCodeDr1csNpFlpcpSparse<F, TensorRsMulCode<F>>>,
}

impl<F: PrimeField + FftField> WeRingLweProverContext<F> {
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
}

fn make_theorem43_dpp_from_dr1cs<F: PrimeField + FftField>(
    dr1cs: FileBackedSparseDr1csInstance<F>,
    public_len: usize,
) -> Result<Theorem43Dpp<F, FileBackedChunkedMulCodeDr1csNpFlpcpSparse<F, TensorRsMulCode<F>>>, String> {
    let code = TensorRsMulCode::<F>::new(48, 3)?;
    let flpcp = FileBackedChunkedMulCodeDr1csNpFlpcpSparse::<F, _>::new(dr1cs, public_len, code)?;
    Theorem43Dpp::<F, _>::new(flpcp)
}

/// Build a prover-side streaming context from a public DR1CS instance.
pub(crate) fn we_ringlwe_prover_from_dr1cs<F: PrimeField + FftField>(
    dr1cs: FileBackedSparseDr1csInstance<F>,
    public_len: usize,
) -> Result<WeRingLweProverContext<F>, String> {
    let dpp = make_theorem43_dpp_from_dr1cs::<F>(dr1cs, public_len)?;
    Ok(WeRingLweProverContext { dpp })
}

/// Arm-time output for a single Theorem-4.3 hidden-query sublock (no ciphertext).
pub(crate) struct WeRingLweSubLockArmOut<F: PrimeField> {
    pub c_stmt: Vec<F>,
    pub accepting_set_shifted: [F; 2],
    pub coins: dpp::theorem43::Theorem43Coins<F>,
    pub x_len: usize,
    pub pi_len: usize,
    pub q_blocks: Vec<(usize, crate::lockable_ringlwe::PackedF257Block64)>,
}

/// Arm (publish) the public data needed for a single sublock (one hidden query).
///
/// This returns the shifted accepting set, coins, and sparse query blocks `q` (so the caller can
/// scale by a chosen secret scalar and/or share a ciphertext across many sublocks).
pub(crate) fn arm_we_ringlwe_sublock_from_dr1cs<F: PrimeField + FftField>(
    dr1cs: FileBackedSparseDr1csInstance<F>,
    public_len: usize,
    stmt_digest: [u8; 32],
    x: &[F],
    armer_seed: [u8; 32],
    lock_j: u64,
    block_id: usize,
    rep_id: u64,
) -> Result<WeRingLweSubLockArmOut<F>, String> {
    if x.len() != public_len {
        return Err("arm_we_ringlwe_sublock_from_dr1cs: x length != public_len".to_string());
    }
    let dpp = make_theorem43_dpp_from_dr1cs::<F>(dr1cs, public_len)?;
    let mut scratch = dpp.query_scratch();
    let mut acc = QueryBlockAccumulator::<F>::new(dpp.proof_len())?;

    let c_stmt = crate::we_statement::digest32_to_bits_field::<F>(stmt_digest);
    let armer_secret = crate::we_statement::derive_armer_secret::<F>(armer_seed, stmt_digest, lock_j, 4);
    let art = dpp.arm(&c_stmt, x, &armer_secret, block_id, rep_id)?;
    let pi_len = dpp.proof_len();

    let mut err: Option<String> = None;
    let offset_f = dpp.stream_query_terms_for_pi(
        x,
        &art.coins,
        &art.coeffs,
        &mut scratch,
        &mut |pi_idx, coeff| {
            if err.is_some() {
                return;
            }
            if let Err(e) = acc.add_term(&coeff, pi_idx) {
                err = Some(e);
            }
        },
    )?;
    if let Some(e) = err {
        return Err(e);
    }
    let q_blocks = acc.into_sparse_blocks();

    // Shift accepting set by offset so decap only needs ⟨q_π, π⟩.
    let shifted = [art.accepting_set[0] - offset_f, art.accepting_set[1] - offset_f];
    if shifted[0].is_zero() || shifted[1].is_zero() {
        return Err(
            "arm_we_ringlwe_sublock_from_dr1cs: shifted accepting set contains 0; resample rep_id"
                .to_string(),
        );
    }

    Ok(WeRingLweSubLockArmOut {
        c_stmt,
        accepting_set_shifted: shifted,
        coins: art.coins.clone(),
        x_len: x.len(),
        pi_len,
        q_blocks,
    })
}

#[inline]
pub(crate) fn derive_rep_id_try(
    armer_seed32: &[u8; 32],
    stmt_digest: &[u8; 32],
    lock_j: u64,
    block_id: usize,
    base_rep_id: u64,
    channel_id: u16,
    rep: u16,
    try_idx: u64,
) -> u64 {
    use sha2::Digest;
    let mut h = sha2::Sha256::new();
    h.update(b"LFP_WE_RINGLWE_REP_ID_V1");
    h.update(armer_seed32);
    h.update(stmt_digest);
    h.update(&lock_j.to_le_bytes());
    h.update(&(block_id as u64).to_le_bytes());
    h.update(&base_rep_id.to_le_bytes());
    h.update(&channel_id.to_le_bytes());
    h.update(&rep.to_le_bytes());
    h.update(&try_idx.to_le_bytes());
    let out: [u8; 32] = h.finalize().into();
    u64::from_le_bytes(out[0..8].try_into().unwrap())
}

pub(crate) struct WeRingLweLockArmingPolicy {
    pub base_rep_id: u64,
    pub max_rep_tries: usize,
    pub hint_budget_bytes: Option<usize>,
}

pub(crate) struct WeRingLweLockArmOut<F: PrimeField> {
    pub lock: RingLweLockArtifact<F>,
    pub s_channels_mod257: Vec<u16>,
}

/// Arm (publish) a RingLWE lock artifact from a public DR1CS instance and statement `x`.
///
/// Canonical arming uses `(P channels) × (R reps)` sublocks and enforces per-channel ratio-class
/// distinctness (so honest decap is non-branching).
fn arm_we_ringlwe_lock_from_dr1cs<F: PrimeField + FftField>(
    dr1cs: FileBackedSparseDr1csInstance<F>,
    public_len: usize,
    stmt_digest: [u8; 32],
    x: &[F],
    armer_seed: [u8; 32],
    lock_j: u64,
    block_id: usize,
    policy: WeRingLweLockArmingPolicy,
    params: RingLweParams,
    p_channels: u16,
    r_reps: u16,
    payload: &[u8],
    rng: &mut impl rand::RngCore,
) -> Result<WeRingLweLockArmOut<F>, String> {
    if p_channels == 0 || r_reps == 0 {
        return Err("arm_we_ringlwe_lock_from_dr1cs: invalid (P,R)".to_string());
    }
    let max_rep_tries = policy.max_rep_tries.max(1);

    // Sample per-channel secrets.
    let mut s_channels: Vec<u16> = Vec::with_capacity(p_channels as usize);
    for _ in 0..p_channels {
        s_channels.push(sample_nonzero_f257_scalar(rng));
    }

    // Arm sublocks with ratio-class distinctness enforced within each channel.
    let mut sublocks: Vec<RingLweSubLock<F>> =
        Vec::with_capacity((p_channels as usize) * (r_reps as usize));
    let mut c_stmt: Option<Vec<F>> = None;
    let mut x_len: Option<usize> = None;
    let mut pi_len: Option<usize> = None;
    let mut global_try_idx: u64 = 0;

    for ch in 0..p_channels {
        let mut used_ratio_classes: Vec<u16> = Vec::with_capacity(r_reps as usize);
        for rep in 0..r_reps {
            let mut tries = 0usize;
            let mut best_sl: Option<RingLweSubLock<F>> = None;
            let mut best_hint_bytes: usize = usize::MAX;
            while tries < max_rep_tries {
                tries += 1;
                let rep_id = derive_rep_id_try(
                    &armer_seed,
                    &stmt_digest,
                    lock_j,
                    block_id,
                    policy.base_rep_id,
                    ch,
                    rep,
                    global_try_idx,
                );
                global_try_idx = global_try_idx.wrapping_add(1);
                let out = match arm_we_ringlwe_sublock_from_dr1cs::<F>(
                    dr1cs.clone(),
                    public_len,
                    stmt_digest,
                    x,
                    armer_seed,
                    lock_j,
                    block_id,
                    rep_id,
                ) {
                    Ok(v) => v,
                    Err(e) if e.contains("shifted accepting set contains 0") => {
                        continue;
                    }
                    Err(e) => return Err(e),
                };

                let ratio_class = ratio_class_mod257_u16(
                    &out.accepting_set_shifted[0],
                    &out.accepting_set_shifted[1],
                )?;
                if used_ratio_classes.contains(&ratio_class) {
                    continue;
                }

                if c_stmt.is_none() {
                    c_stmt = Some(out.c_stmt.clone());
                    x_len = Some(out.x_len);
                    pi_len = Some(out.pi_len);
                }

                let s = s_channels[ch as usize];
                let h_blocks: Vec<(usize, crate::lockable_ringlwe::PackedF257Block64)> = out
                    .q_blocks
                    .iter()
                    .map(|(block_idx, q)| (*block_idx, q.scale_mod257(s)))
                    .collect();
                let hint_bytes: usize = h_blocks
                    .iter()
                    .map(|(_bi, blk)| 4 + blk.on_disk_bytes())
                    .sum();
                let sl = RingLweSubLock::<F> {
                    channel_id: ch,
                    accepting_set: out.accepting_set_shifted,
                    coins: out.coins.clone(),
                    hints: crate::lockable_ringlwe::BranchHints {
                        hint_blocks_sparse: h_blocks,
                    },
                };
                if hint_bytes < best_hint_bytes {
                    best_hint_bytes = hint_bytes;
                    best_sl = Some(sl.clone());
                }
                if let Some(budget) = policy.hint_budget_bytes {
                    if hint_bytes > budget {
                        continue;
                    }
                }
                best_sl = Some(sl);
                break;
            }
            let sl = best_sl.ok_or_else(|| {
                "arm_we_ringlwe_lock_from_dr1cs: failed to arm sublock within retry budget"
                    .to_string()
            })?;
            let ratio_class =
                ratio_class_mod257_u16(&sl.accepting_set[0], &sl.accepting_set[1])?;
            if used_ratio_classes.contains(&ratio_class) {
                return Err(
                    "arm_we_ringlwe_lock_from_dr1cs: duplicate ratio class after retries".to_string(),
                );
            }
            used_ratio_classes.push(ratio_class);
            sublocks.push(sl);
        }
    }

    let lock = arm_ringlwe_lock(
        c_stmt.ok_or_else(|| "arm_we_ringlwe_lock_from_dr1cs: missing c_stmt".to_string())?,
        x_len.ok_or_else(|| "arm_we_ringlwe_lock_from_dr1cs: missing x_len".to_string())?,
        pi_len.ok_or_else(|| "arm_we_ringlwe_lock_from_dr1cs: missing pi_len".to_string())?,
        params,
        p_channels,
        r_reps,
        sublocks,
        payload,
        s_channels.as_slice(),
        rng,
    )?;
    Ok(WeRingLweLockArmOut {
        lock,
        s_channels_mod257: s_channels,
    })
}

/// Arm the **LF+ tiny-field WE gate** (Poseidon(F257) + CM-coin surfaces) as a Theorem-4.3 lock.
///
/// This is the main wiring from:
/// - `we_gate_arith::build_we_dr1cs_for_plus_proof_shape_tiny` (arm-time instance construction)
/// into:
/// - `arm_we_ringlwe_lock_from_dr1cs` (Theorem-4.3 + Ring-LWE wrapper).
///
/// Notes:
/// - The statement binding is carried by `stmt_digest` via `c_stmt` inside `arm_theorem43_from_statement`.
/// - `x` is the canonical public statement encoding:
///   `x = [ONE] || [10×WeParams] || [public_inputs...]`.
#[cfg(feature = "we_gate")]
pub(crate) fn arm_lfplus_ringlwe_lock<R>(
    shape: we_gate_arith::WeDr1csShape<F257>,
    params: &WeParams,
    public_inputs: &[F257],
    stmt_digest: [u8; 32],
    armer_seed: [u8; 32],
    lock_j: u64,
    block_id: usize,
    policy: WeRingLweLockArmingPolicy,
    ringlwe_params: RingLweParams,
    p_channels: u16,
    r_reps: u16,
    payload: &[u8],
    rng: &mut impl rand::RngCore,
) -> Result<WeRingLweLockArmOut<F257>, String>
where
    R: OverField + CoeffRing + PolyRing,
    R::BaseRing: Zq + ark_ff::Field + ark_ff::PrimeField,
{
    let x = crate::we_statement::encode_public_x::<F257>(params, public_inputs);
    if x.len() != shape.public_len {
        return Err(format!(
            "arm_lfplus_ringlwe_lock: public_len mismatch (shape={} vs x={})",
            shape.public_len,
            x.len()
        ));
    }

    arm_we_ringlwe_lock_from_dr1cs::<F257>(
        shape.inst,
        shape.public_len,
        stmt_digest,
        &x,
        armer_seed,
        lock_j,
        block_id,
        policy,
        ringlwe_params,
        p_channels,
        r_reps,
        payload,
        rng,
    )
}
