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
use sha2::Digest;
use std::sync::Arc;

use dpp::dr1cs_flpcp::{Dr1csNpFlpcpSparseApi, Dr1csQueryScratch, MulCode, QuerySink, TensorRsMulCode};
use dpp::packing::FlpcpPredicate;
use dpp::theorem43::{Theorem43Coins, Theorem43Dpp, Theorem43LockArtifact};
use dpp::SparseVec;
use symphony::file_backed_dr1cs::{cfg_read_buf_bytes, FileBackedSparseDr1csInstance};

use crate::lockable_ringlwe::{
    arm_ringlwe_lock, sample_nonzero_f257_scalar, RingLweLockArtifact,
    RingLweParams, RingLweSubLock,
};

pub use crate::we_statement::arm_theorem43_from_statement;

#[cfg(feature = "we_gate")]
use crate::we_gate_arith;
#[cfg(feature = "we_gate")]
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

    /// Precompute per-row dot products of the public prefix `x` against the sparse A/B/C rows
    /// for a given block.
    ///
    /// Returns `(ax, bx, cx)` where each is length `k` and:
    /// - `ax[i] = <A_row[i], x> mod 257` using only term indices `< l`
    /// - `bx[i] = <B_row[i], x> mod 257`
    /// - `cx[i] = <C_row[i], x> mod 257`
    ///
    /// This is the core "shared block precompute" needed for batched arming: for many hits on
    /// the same block, we scan the block's term pools **once**, then reuse these arrays to compute
    /// `δ(x) = 1 + <q_x, x>` cheaply per hit.
    fn precompute_public_x_row_dots_mod257_u16(
        &self,
        block_id: usize,
        x_u16: &[u16],
    ) -> Result<(Vec<u16>, Vec<u16>, Vec<u16>), String> {
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

        if x_u16.len() != self.l {
            return Err("precompute_public_x_row_dots_mod257_u16: bad x_u16 length".to_string());
        }
        let k = self.k();
        let row_start = (block_id as u64).saturating_mul(k as u64);
        let nconstraints = self.nconstraints();

        let (mut rows, mut a_coeffs, mut a_idx, mut b_coeffs, mut b_idx, mut c_coeffs, mut c_idx) =
            self.open_readers_at_row(row_start)?;

        let mut ax = vec![0u16; k];
        let mut bx = vec![0u16; k];
        let mut cx = vec![0u16; k];

        for i in 0..k {
            let row = row_start.saturating_add(i as u64);
            if row >= nconstraints {
                break;
            }
            let a_len = read_u32(&mut rows)? as usize;
            let b_len = read_u32(&mut rows)? as usize;
            let c_len = read_u32(&mut rows)? as usize;

            let mut acc_a: u16 = 0;
            for _ in 0..a_len {
                let cu16 = read_u16(&mut a_coeffs)?;
                let vidx = read_u32(&mut a_idx)? as usize;
                if vidx < self.l && cu16 != 0 {
                    let xv = x_u16[vidx];
                    if xv != 0 {
                        acc_a = crate::lockable_ringlwe::add_mod257_u16(
                            acc_a,
                            crate::lockable_ringlwe::mul_mod257_u16(cu16, xv),
                        );
                    }
                }
            }
            let mut acc_b: u16 = 0;
            for _ in 0..b_len {
                let cu16 = read_u16(&mut b_coeffs)?;
                let vidx = read_u32(&mut b_idx)? as usize;
                if vidx < self.l && cu16 != 0 {
                    let xv = x_u16[vidx];
                    if xv != 0 {
                        acc_b = crate::lockable_ringlwe::add_mod257_u16(
                            acc_b,
                            crate::lockable_ringlwe::mul_mod257_u16(cu16, xv),
                        );
                    }
                }
            }
            let mut acc_c: u16 = 0;
            for _ in 0..c_len {
                let cu16 = read_u16(&mut c_coeffs)?;
                let vidx = read_u32(&mut c_idx)? as usize;
                if vidx < self.l && cu16 != 0 {
                    let xv = x_u16[vidx];
                    if xv != 0 {
                        acc_c = crate::lockable_ringlwe::add_mod257_u16(
                            acc_c,
                            crate::lockable_ringlwe::mul_mod257_u16(cu16, xv),
                        );
                    }
                }
            }
            ax[i] = acc_a;
            bx[i] = acc_b;
            cx[i] = acc_c;
        }

        Ok((ax, bx, cx))
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
            .unwrap_or(16384)
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
            .unwrap_or(32)
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
        _scratch: &mut Dr1csQueryScratch<F>,
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
            // IMPORTANT: Do NOT stream dense q3 witness terms over the `w_eval` coordinates.
            //
            // In the large-block file-backed regime, these would be ~k_star terms per coin, which
            // is prohibitively expensive. Callers must account for the q3 witness contribution via
            // `dot_q3_w_eval(idx, lambda, w_eval)` during block streaming.
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
                    // NOTE: file-backed variable indices match `v=(x||z_w)` indexing.
                    sink.on_q1(c * cab, vidx);
                }
            }
            for _ in 0..b_len {
                let cu16 = read_u16(&mut b_coeffs)?;
                let vidx = read_u32(&mut b_idx)? as usize;
                if !cab.is_zero() {
                    let c = coeff_lut[cu16 as usize];
                    sink.on_q2(c * cab, vidx);
                }
            }
            let cc = coeff_c[i];
            for _ in 0..c_len {
                let cu16 = read_u16(&mut c_coeffs)?;
                let vidx = read_u32(&mut c_idx)? as usize;
                if !cc.is_zero() {
                    let c = coeff_lut[cu16 as usize];
                    // q3 z-side term is scaled by `lambda`.
                    let t = (c * cc) * lambda;
                    if !t.is_zero() {
                        sink.on_q3(t, vidx);
                    }
                }
            }
        }

        Ok(())
    }

    fn dot_q3_w_eval(
        &self,
        idx: usize,
        lambda: F,
        _x: &[F],
        _scratch: &mut Dr1csQueryScratch<F>,
        w_eval: &[F],
    ) -> Result<F, String> {
        let (_block_id, local_idx) = self.decode_block_idx(idx)?;
        let k = self.k();
        let k_star = self.k_star();
        if w_eval.len() != k_star {
            return Err("dot_q3_w_eval: bad w_eval length".to_string());
        }
        let mut s_star = F::ZERO;
        let mut s_low = F::ZERO;
        self.code.row_e_star_stream(local_idx, &mut |j, c| {
            let t = c * w_eval[j];
            s_star += t;
            if j < k {
                s_low += t;
            }
        })?;
        Ok(s_star - (lambda * s_low))
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
        on_tail_elem: &mut dyn FnMut(usize, usize, &F),
    ) -> Result<Vec<dpp::theorem43::Theorem43AbgTail<F>>, String> {
        self.dpp
            .stream_pi0_and_collect_tails(x, z_w, coins_list, on_pi0_chunk, on_tail_elem)
    }

    pub fn proof_len(&self) -> usize {
        self.dpp.proof_len()
    }

    pub fn derive_public_coins_from_stmt(
        &self,
        c_stmt: &[F],
        block_id: usize,
        rep_id: u64,
    ) -> Result<Theorem43Coins<F>, String> {
        self.dpp.derive_public_coins_from_stmt(c_stmt, block_id, rep_id)
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
    pub accepting_set: [F; 2],
    // Debug/analysis: Sq coefficients (mod 257) that determine which verifier terms are active.
    pub sq_c1_mod257: u16,
    pub sq_c2_mod257: u16,
    pub x_len: usize,
    pub pi_len: usize,
    /// Unscaled Theorem-4.3 combination coefficients (mod 257 digits), for the π0 answers:
    /// `coeff_alpha`, `coeff_beta`, `coeff_gamma`.
    pub abg_coeffs_mod257: [u16; 3],
    /// Unscaled statement-dependent offset digit `δ(x) = 1 + ⟨q_x, x⟩ (mod 257)`.
    pub delta_x_mod257: u16,
    /// Unscaled tail coefficients (mod 257 digits) in canonical order:
    /// `[coeff_mu, coeff_nu, c3, c4, ..., c_{p-1}]` of length 256 for F257.
    pub tail_coeffs_mod257: Vec<u16>,
}

/// Arm-time coefficients for one sublock, excluding `c_stmt` (which is constant across sublocks).
///
/// This is the fast path used by full-coverage arming to avoid per-hit `c_stmt` allocation/copy.
pub(crate) struct WeRingLweSubLockArmCoeffs<F: PrimeField> {
    pub accepting_set: [F; 2],
    pub abg_coeffs_mod257: [u16; 3],
    pub delta_x_mod257: u16,
    pub tail_coeffs_mod257: [u16; 256],
}

/// Arm (publish) the public data needed for a single sublock (one hidden query).
///
/// This returns the fixed accepting set `{1,2}`, public coins, and compressed hint coefficients.
pub(crate) fn arm_we_ringlwe_sublock_from_dr1cs<F: PrimeField + FftField>(
    dr1cs: FileBackedSparseDr1csInstance<F>,
    public_len: usize,
    stmt_digest: [F257; 32],
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

    let c_stmt: Vec<F> = stmt_digest
        .iter()
        .map(|e| F::from((e.into_bigint().as_ref()[0] % 257) as u64))
        .collect();
    let stmt_bytes64 = {
        let mut out = [0u8; 64];
        for (i, e) in stmt_digest.iter().enumerate() {
            let v = (e.into_bigint().as_ref()[0] % 257) as u16;
            let b = v.to_le_bytes();
            out[2 * i] = b[0];
            out[2 * i + 1] = b[1];
        }
        out
    };
    let armer_secret = {
        let mut out = Vec::with_capacity(4);
        for i in 0..4usize {
            let mut h = sha2::Sha256::new();
            h.update(b"LFP_ARMER_SECRET_V1");
            h.update(&armer_seed);
            h.update(&stmt_bytes64);
            h.update(&lock_j.to_le_bytes());
            h.update(&(i as u64).to_le_bytes());
            let d: [u8; 32] = h.finalize().into();
            out.push(F::from_le_bytes_mod_order(&d));
        }
        out
    };
    let art = dpp.arm(&c_stmt, x, &armer_secret, block_id, rep_id)?;
    // Reject degenerate Sq coefficients that can erase critical terms (notably the q3/gamma path)
    // and make UNSAT instances pass with non-negligible probability.
    //
    // In particular:
    // - `c1 = coeffs[0] == 0` removes the alpha/beta linear terms.
    // - `c2 = coeffs[1] == 0` removes the gamma/mu/nu terms, weakening the check drastically.
    //
    // These events occur with probability ~1/257 each over F257 and should be rejected.
    if art.coeffs.len() < 2 {
        return Err("arm_we_ringlwe_sublock_from_dr1cs: bad Sq coeff length; resample rep_id".to_string());
    }
    if art.coeffs[0].is_zero() || art.coeffs[1].is_zero() {
        return Err(
            "arm_we_ringlwe_sublock_from_dr1cs: degenerate Sq coeff (c1==0 or c2==0); resample rep_id"
                .to_string(),
        );
    }
    let sq_c1_mod257: u16 = (art.coeffs[0].into_bigint().as_ref()[0] % 257) as u16;
    let sq_c2_mod257: u16 = (art.coeffs[1].into_bigint().as_ref()[0] % 257) as u16;
    let pi_len = dpp.proof_len();

    // Instance-binding:
    // compute the *arming-statement* offset digit `δ(x_arm) = 1 + ⟨q_x, x_arm⟩ (mod 257)` and
    // later store only the masked scalar `offset_scale = s * δ(x_arm)` inside the package.
    //
    // We intentionally do NOT store masked per-coordinate `h_x` (x_scales), so the lock cannot
    // be re-targeted to an arbitrary decap-side statement.
    let mut delta_x_mod257: u16 = 1; // includes protocol constant +1
    dpp.stream_query_terms_for_x(
        x,
        &art.coins,
        &art.coeffs,
        &mut scratch,
        &mut |xi, coeff| {
            // `stream_query_terms_for_x` emits only indices `< x_len`.
            let c = (coeff.into_bigint().as_ref()[0] % 257) as u16;
            let xv = crate::lockable_ringlwe::field_mod257_u16(&x[xi]);
            delta_x_mod257 =
                crate::lockable_ringlwe::add_mod257_u16(delta_x_mod257, crate::lockable_ringlwe::mul_mod257_u16(c, xv));
        },
    )?;
    // If δ(x_arm)=0 then `offset_scale=0` and the lock degenerates back into a “language gate”
    // for SAT statements. Reject and resample.
    if delta_x_mod257 == 0 {
        return Err("arm_we_ringlwe_sublock_from_dr1cs: delta_x_mod257==0; resample rep_id".to_string());
    }

    // Unscaled combination coefficients (mod 257 digits).
    let c1 = art.coeffs[0];
    let c2 = art.coeffs[1];
    let coeff_alpha = c1 * art.coins.rho;
    let coeff_beta = c1 * art.coins.sigma;
    let coeff_gamma = {
        let two = F::from(2u64);
        c2 * (two * art.coins.rho * art.coins.sigma)
    };
    let abg_coeffs_mod257: [u16; 3] = [
        (coeff_alpha.into_bigint().as_ref()[0] % 257) as u16,
        (coeff_beta.into_bigint().as_ref()[0] % 257) as u16,
        (coeff_gamma.into_bigint().as_ref()[0] % 257) as u16,
    ];

    // Tail coefficients correspond to the tail values produced by
    // `Theorem43Dpp::stream_pi0_and_collect_tails`: `[mu, nu, u^3..u^{p-1}]`.
    let coeff_mu = c2 * (art.coins.rho * art.coins.rho);
    let coeff_nu = c2 * (art.coins.sigma * art.coins.sigma);
    let mut tail_coeffs_mod257: Vec<u16> = Vec::with_capacity(256);
    tail_coeffs_mod257.push((coeff_mu.into_bigint().as_ref()[0] % 257) as u16);
    tail_coeffs_mod257.push((coeff_nu.into_bigint().as_ref()[0] % 257) as u16);
    for c in art.coeffs.iter().copied().skip(2) {
        tail_coeffs_mod257.push((c.into_bigint().as_ref()[0] % 257) as u16);
    }
    if tail_coeffs_mod257.len() != 256 {
        return Err("arm_we_ringlwe_sublock_from_dr1cs: unexpected tail coeff length".to_string());
    }

    let accepting_set = [F::ONE, F::from(2u64)];

    Ok(WeRingLweSubLockArmOut {
        c_stmt,
        accepting_set,
        sq_c1_mod257,
        sq_c2_mod257,
        x_len: x.len(),
        pi_len,
        abg_coeffs_mod257,
        delta_x_mod257,
        tail_coeffs_mod257,
    })
}

fn arm_we_ringlwe_sublock_coeffs_from_dpp<F: PrimeField + FftField>(
    dpp: &Theorem43Dpp<F, FileBackedChunkedMulCodeDr1csNpFlpcpSparse<F, TensorRsMulCode<F>>>,
    scratch: &mut dpp::dr1cs_flpcp::Dr1csQueryScratch<F>,
    c_stmt: &[F],
    x: &[F],
    armer_secret: &[F],
    block_id: usize,
    rep_id: u64,
) -> Result<WeRingLweSubLockArmCoeffs<F>, String> {
    scratch.clear_all();
    let art = dpp.arm(c_stmt, x, armer_secret, block_id, rep_id)?;

    // Reject degenerate Sq coefficients.
    if art.coeffs.len() < 2 {
        return Err(
            "arm_we_ringlwe_sublock_coeffs_from_dpp: bad Sq coeff length; resample rep_id".to_string(),
        );
    }
    if art.coeffs[0].is_zero() || art.coeffs[1].is_zero() {
        return Err(
            "arm_we_ringlwe_sublock_coeffs_from_dpp: degenerate Sq coeff (c1==0 or c2==0); resample rep_id"
                .to_string(),
        );
    }

    // Instance-binding: compute `δ(x) = 1 + ⟨q_x, x⟩ (mod 257)` for the arming statement.
    let mut delta_x_mod257: u16 = 1;
    dpp.stream_query_terms_for_x(
        x,
        &art.coins,
        &art.coeffs,
        scratch,
        &mut |xi, coeff| {
            if xi >= x.len() {
                return;
            }
            let c = (coeff.into_bigint().as_ref()[0] % 257) as u16;
            let xj = (x[xi].into_bigint().as_ref()[0] % 257) as u16;
            delta_x_mod257 = crate::lockable_ringlwe::add_mod257_u16(
                delta_x_mod257,
                crate::lockable_ringlwe::mul_mod257_u16(c, xj),
            );
        },
    )?;
    if delta_x_mod257 == 0 {
        return Err(
            "arm_we_ringlwe_sublock_coeffs_from_dpp: delta_x_mod257==0; resample rep_id".to_string(),
        );
    }

    // Unscaled combination coefficients (mod 257 digits).
    let c1 = art.coeffs[0];
    let c2 = art.coeffs[1];
    let coeff_alpha = c1 * art.coins.rho;
    let coeff_beta = c1 * art.coins.sigma;
    let coeff_gamma = {
        let two = F::from(2u64);
        c2 * (two * art.coins.rho * art.coins.sigma)
    };
    let abg_coeffs_mod257: [u16; 3] = [
        (coeff_alpha.into_bigint().as_ref()[0] % 257) as u16,
        (coeff_beta.into_bigint().as_ref()[0] % 257) as u16,
        (coeff_gamma.into_bigint().as_ref()[0] % 257) as u16,
    ];

    // Tail coefficients correspond to `stream_pi0_and_collect_tails` tail layout:
    // `[mu, nu, u^3..u^{p-1}]` (len=256 for F257).
    let coeff_mu = c2 * (art.coins.rho * art.coins.rho);
    let coeff_nu = c2 * (art.coins.sigma * art.coins.sigma);
    let mut tail_coeffs_mod257: [u16; 256] = [0u16; 256];
    tail_coeffs_mod257[0] = (coeff_mu.into_bigint().as_ref()[0] % 257) as u16;
    tail_coeffs_mod257[1] = (coeff_nu.into_bigint().as_ref()[0] % 257) as u16;
    for (i, c) in art.coeffs.iter().copied().skip(2).enumerate() {
        tail_coeffs_mod257[2 + i] = (c.into_bigint().as_ref()[0] % 257) as u16;
    }

    Ok(WeRingLweSubLockArmCoeffs {
        accepting_set: [F::ONE, F::from(2u64)],
        abg_coeffs_mod257,
        delta_x_mod257,
        tail_coeffs_mod257,
    })
}

#[inline]
pub(crate) fn derive_rep_id_try(
    armer_seed32: &[u8; 32],
    stmt_digest: &[F257; 32],
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
    for e in stmt_digest {
        let v = (e.into_bigint().as_ref()[0] % 257) as u16;
        h.update(v.to_le_bytes());
    }
    h.update(&lock_j.to_le_bytes());
    h.update(&(block_id as u64).to_le_bytes());
    h.update(&base_rep_id.to_le_bytes());
    h.update(&channel_id.to_le_bytes());
    h.update(&rep.to_le_bytes());
    h.update(&try_idx.to_le_bytes());
    let out: [u8; 32] = h.finalize().into();
    u64::from_le_bytes(out[0..8].try_into().unwrap())
}

#[derive(Clone, Copy, Debug)]
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
/// Canonical arming uses `hits_per_block` independent sublocks per FLPCP block.
fn arm_we_ringlwe_lock_from_dr1cs<F: PrimeField + FftField>(
    dr1cs: FileBackedSparseDr1csInstance<F>,
    public_len: usize,
    stmt_digest: [F257; 32],
    x: &[F],
    armer_seed: [u8; 32],
    lock_j: u64,
    _block_id: usize,
    policy: WeRingLweLockArmingPolicy,
    params: RingLweParams,
    hits_per_block: u16,
    payload: &[u8],
    rng: &mut impl rand::RngCore,
) -> Result<WeRingLweLockArmOut<F>, String> {
    // Inline canonical construction so we can use the chunked backend for shared per-block precomputes.
    let code = TensorRsMulCode::<F>::new(48, 3)?;
    let flpcp = FileBackedChunkedMulCodeDr1csNpFlpcpSparse::<F, _>::new(dr1cs.clone(), public_len, code)?;
    let dpp = Theorem43Dpp::<F, _>::new(flpcp.clone())?;
    let mut scratch = dpp.query_scratch();
    if hits_per_block == 0 {
        return Err("arm_we_ringlwe_lock_from_dr1cs: hits_per_block=0".to_string());
    }
    // Canonical “hits-per-block” design uses a single channel secret.
    let p_channels: u16 = 1;
    let max_rep_tries = policy.max_rep_tries.max(1);

    // Sample per-channel secrets.
    let s_channels: Vec<u16> = vec![sample_nonzero_f257_scalar(rng)];

    // Statement commitment and armer secret are constant across all hits.
    let c_stmt: Vec<F> = stmt_digest
        .iter()
        .map(|e| F::from((e.into_bigint().as_ref()[0] % 257) as u64))
        .collect();
    let stmt_bytes64: [u8; 64] = {
        let mut out = [0u8; 64];
        for (i, e) in stmt_digest.iter().enumerate() {
            let v = (e.into_bigint().as_ref()[0] % 257) as u16;
            let b = v.to_le_bytes();
            out[2 * i] = b[0];
            out[2 * i + 1] = b[1];
        }
        out
    };
    let armer_secret: Vec<F> = {
        use sha2::Digest;
        let mut out = Vec::with_capacity(4);
        for i in 0..4usize {
            let mut h = sha2::Sha256::new();
            h.update(b"LFP_ARMER_SECRET_V1");
            h.update(&armer_seed);
            h.update(&stmt_bytes64);
            h.update(&lock_j.to_le_bytes());
            h.update(&(i as u64).to_le_bytes());
            let d: [u8; 32] = h.finalize().into();
            out.push(F::from_le_bytes_mod_order(&d));
        }
        out
    };
    let x_len: usize = x.len();
    let pi_len: usize = dpp.proof_len();

    // Block-selection policy.
    const MIN_NNZ_ROW_E: usize = 48usize.pow(3); // base_k^rank (keep consistent with constructor above)
    let blocks: usize = flpcp.blocks();
    let ell_local: usize = flpcp.ell_local().max(1);
    if blocks == 0 {
        return Err("arm_we_ringlwe_lock_from_dr1cs: blocks=0".to_string());
    }
    // Full-coverage schedule: every block gets `hits_per_block` independent hits.
    let sublocks_per_channel: u32 = (blocks as u64)
        .saturating_mul(hits_per_block as u64)
        .try_into()
        .map_err(|_| "arm_we_ringlwe_lock_from_dr1cs: sublocks_per_channel overflows u32")?;

    let ch: u16 = 0;
    let hits_usize: usize = hits_per_block as usize;

    // Precompute x mod 257 once (public prefix is tiny).
    let x_u16: Vec<u16> = x
        .iter()
        .map(|e| (e.into_bigint().as_ref()[0] % 257) as u16)
        .collect();

    // Shared-block optimization: scan each block once to get `<A_i,x>,<B_i,x>,<C_i,x>`,
    // then compute `δ(x)` per hit using tensor-code dot products (no per-hit constraint scanning).
    let per_block: Vec<Vec<(usize, RingLweSubLock<F>)>> = (0..blocks)
        .into_par_iter()
        .map_init(|| dpp.query_scratch(), |scratch, block_id_sl| -> Result<Vec<(usize, RingLweSubLock<F>)>, String> {
            let (ax, bx, cx) = flpcp.precompute_public_x_row_dots_mod257_u16(block_id_sl, &x_u16)?;
            let mut out_block: Vec<(usize, RingLweSubLock<F>)> = Vec::with_capacity(hits_usize);

            for rep_usize in 0..hits_usize {
                let rep: u16 = rep_usize
                    .try_into()
                    .map_err(|_| "arm_we_ringlwe_lock_from_dr1cs: rep overflow".to_string())?;

                let mut tries = 0usize;
                let mut best_sl: Option<RingLweSubLock<F>> = None;
                let mut best_hint_bytes: Option<usize> = None;
                while tries < max_rep_tries {
                    tries += 1;
                    let try_idx = (tries - 1) as u64;
                    let rep_id = derive_rep_id_try(
                        &armer_seed,
                        &stmt_digest,
                        lock_j,
                        block_id_sl,
                        policy.base_rep_id,
                        ch,
                        rep,
                        try_idx,
                    );

                    scratch.clear_all();
                    let art = dpp.arm(c_stmt.as_slice(), x, armer_secret.as_slice(), block_id_sl, rep_id)?;

                    // Reject degenerate Sq coefficients.
                    if art.coeffs.len() < 2 || art.coeffs[0].is_zero() || art.coeffs[1].is_zero() {
                        continue;
                    }

                    let local_idx = art.coins.idx % ell_local;
                    // Apply dense-support quality filtering (preserve existing behavior).
                    let nnz = flpcp.code.nnz_row_e_u16(local_idx)?;
                    if nnz < MIN_NNZ_ROW_E {
                        continue;
                    }

                    // Unscaled combination coefficients (mod 257 digits).
                    let c1 = art.coeffs[0];
                    let c2 = art.coeffs[1];
                    let coeff_alpha = c1 * art.coins.rho;
                    let coeff_beta = c1 * art.coins.sigma;
                    let coeff_gamma = {
                        let two = F::from(2u64);
                        c2 * (two * art.coins.rho * art.coins.sigma)
                    };
                    let abg_coeffs_mod257: [u16; 3] = [
                        (coeff_alpha.into_bigint().as_ref()[0] % 257) as u16,
                        (coeff_beta.into_bigint().as_ref()[0] % 257) as u16,
                        (coeff_gamma.into_bigint().as_ref()[0] % 257) as u16,
                    ];

                    // Compute δ(x) mod 257 using shared per-block row dots.
                    let sum_a = flpcp.code.dot_row_e_u16(local_idx, &ax)?;
                    let sum_b = flpcp.code.dot_row_e_u16(local_idx, &bx)?;
                    let sum_c = flpcp.code.dot_row_e_star_low_u16(local_idx, &cx)?;
                    let lambda_u16 = (art.coins.lambda.into_bigint().as_ref()[0] % 257) as u16;

                    let mut delta_x_mod257: u16 = 1;
                    delta_x_mod257 = crate::lockable_ringlwe::add_mod257_u16(
                        delta_x_mod257,
                        crate::lockable_ringlwe::mul_mod257_u16(abg_coeffs_mod257[0], sum_a),
                    );
                    delta_x_mod257 = crate::lockable_ringlwe::add_mod257_u16(
                        delta_x_mod257,
                        crate::lockable_ringlwe::mul_mod257_u16(abg_coeffs_mod257[1], sum_b),
                    );
                    let tmp_c = crate::lockable_ringlwe::mul_mod257_u16(lambda_u16, sum_c);
                    delta_x_mod257 = crate::lockable_ringlwe::add_mod257_u16(
                        delta_x_mod257,
                        crate::lockable_ringlwe::mul_mod257_u16(abg_coeffs_mod257[2], tmp_c),
                    );
                    if delta_x_mod257 == 0 {
                        continue;
                    }

                    // Tail coefficients for `stream_pi0_and_collect_tails`: `[mu, nu, u^3..u^{p-1}]`.
                    let coeff_mu = c2 * (art.coins.rho * art.coins.rho);
                    let coeff_nu = c2 * (art.coins.sigma * art.coins.sigma);
                    let mut tail_coeffs_mod257: [u16; 256] = [0u16; 256];
                    tail_coeffs_mod257[0] = (coeff_mu.into_bigint().as_ref()[0] % 257) as u16;
                    tail_coeffs_mod257[1] = (coeff_nu.into_bigint().as_ref()[0] % 257) as u16;
                    for (i, c) in art.coeffs.iter().copied().skip(2).enumerate() {
                        tail_coeffs_mod257[2 + i] = (c.into_bigint().as_ref()[0] % 257) as u16;
                    }

                    let s = s_channels[0];
                    let abg_scales: [u16; 3] = [
                        crate::lockable_ringlwe::mul_mod257_u16(s, abg_coeffs_mod257[0]),
                        crate::lockable_ringlwe::mul_mod257_u16(s, abg_coeffs_mod257[1]),
                        crate::lockable_ringlwe::mul_mod257_u16(s, abg_coeffs_mod257[2]),
                    ];
                    let offset_scale: u16 = crate::lockable_ringlwe::mul_mod257_u16(s, delta_x_mod257);
                    let tail_scales_blocks: [crate::lockable_ringlwe::PackedF257Block64; 4] =
                        core::array::from_fn(|bi| {
                            let mut tmp = [0u16; 64];
                            let start = bi * 64;
                            for j in 0..64 {
                                tmp[j] = crate::lockable_ringlwe::mul_mod257_u16(s, tail_coeffs_mod257[start + j]);
                            }
                            crate::lockable_ringlwe::PackedF257Block64::from_dense_u16s(&tmp)
                        });

                    let hint_bytes: usize = (3 * 2)
                        + 2
                        + tail_scales_blocks.iter().map(|blk| blk.on_disk_bytes()).sum::<usize>();

                    let sl = RingLweSubLock::<F> {
                        channel_id: ch,
                        accepting_set: [F::ONE, F::from(2u64)],
                        block_id: block_id_sl as u32,
                        rep_id,
                        hints: crate::lockable_ringlwe::BranchHintsCompressed {
                            abg_scales,
                            offset_scale,
                            tail_scales: tail_scales_blocks,
                        },
                    };

                    let within_budget = match policy.hint_budget_bytes {
                        Some(budget) => hint_bytes <= budget,
                        None => true,
                    };
                    if !within_budget {
                        continue;
                    }
                    match best_hint_bytes {
                        Some(cur) if hint_bytes >= cur => {}
                        _ => {
                            best_hint_bytes = Some(hint_bytes);
                            best_sl = Some(sl);
                        }
                    }
                }

                let sl = best_sl.ok_or_else(|| {
                    if let Some(budget) = policy.hint_budget_bytes {
                        format!(
                            "arm_we_ringlwe_lock_from_dr1cs: failed to find in-budget sublock within retry budget (hint_budget_bytes={})",
                            budget
                        )
                    } else {
                        "arm_we_ringlwe_lock_from_dr1cs: failed to arm sublock within retry budget".to_string()
                    }
                })?;

                let lin = block_id_sl.saturating_mul(hits_usize).saturating_add(rep_usize);
                out_block.push((lin, sl));
            }

            Ok(out_block)
        })
        .collect::<Result<Vec<_>, String>>()?;

    let mut pairs: Vec<(usize, RingLweSubLock<F>)> = per_block.into_iter().flatten().collect();
    pairs.sort_unstable_by_key(|(i, _)| *i);
    let sublocks: Vec<RingLweSubLock<F>> = pairs.into_iter().map(|(_, sl)| sl).collect();

    let lock = arm_ringlwe_lock(
        c_stmt,
        x_len,
        pi_len,
        params,
        p_channels,
        sublocks_per_channel,
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
///   `x = [ONE] || [stmt_digest(32)]`.
#[cfg(feature = "we_gate")]
pub(crate) fn arm_lfplus_ringlwe_lock<R>(
    shape: we_gate_arith::WeDr1csShape<F257>,
    stmt_digest: &[F257; 32],
    armer_seed: [u8; 32],
    lock_j: u64,
    block_id: usize,
    policy: WeRingLweLockArmingPolicy,
    ringlwe_params: RingLweParams,
    hits_per_block: u16,
    payload: &[u8],
    rng: &mut impl rand::RngCore,
) -> Result<WeRingLweLockArmOut<F257>, String>
where
    R: OverField + CoeffRing + PolyRing,
    R::BaseRing: Zq + ark_ff::Field + ark_ff::PrimeField,
{
    let x = crate::we_statement::encode_public_x::<F257>(stmt_digest);
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
        *stmt_digest,
        &x,
        armer_seed,
        lock_j,
        block_id,
        policy,
        ringlwe_params,
        hits_per_block,
        payload,
        rng,
    )
}
