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

use dpp::dr1cs_flpcp::{
    reduce_mod257_u64, Dr1csNpFlpcpSparseApi, Dr1csQueryScratch, MulCode, QuerySink, TensorRsMulCode,
};
use dpp::packing::FlpcpPredicate;
use dpp::theorem43::{Theorem43Coins, Theorem43Dpp};
use dpp::SparseVec;
use symphony::file_backed_dr1cs::{cfg_read_buf_bytes, FileBackedSparseDr1csInstance};

use crate::lockable_ringlwe::{arm_ringlwe_lock, ErrGateHints, PackedF257Block64, QueryBlockAccumulator, RingLweLockArtifact, RingLweParams};


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
        on_block_hook: Option<&(dyn Fn(usize, &[u16]) -> Result<(), String> + Sync)>,
        mut on_block: Option<&mut dyn FnMut(usize, &[F])>,
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
        if !dpp::dr1cs_flpcp::is_f257_field::<F>() {
            return Err("stream_w_eval_blocks: unsupported field (tiny-lock requires F257)".to_string());
        }
        let f257_fast = x_u16.is_some() && z_u16.is_some();
        if !f257_fast {
            return Err("stream_w_eval_blocks: only F257 fast path is supported".to_string());
        }
        let (x_u16, z_u16) = (x_u16.unwrap(), z_u16.unwrap());
        let want_f = on_block.is_some();
        let lut: Vec<F> = if want_f {
            (0u16..=256u16).map(|d| F::from(d as u64)).collect()
        } else {
            Vec::new()
        };

        #[inline(always)]
        fn mul_mod257_u16_fast(a: u16, b: u16) -> u16 {
            // 257 = 2^8 + 1, and 256 ≡ -1 (mod 257).
            let prod = (a as u32) * (b as u32); // <= 65536
            let low = (prod & 0xFF) as i32; // 0..255
            let high = (prod >> 8) as i32; // 0..256
            let mut r = low - high; // -256..255
            if r < 0 {
                r += 257;
            }
            r as u16
        }

        let do_prof_w_eval = std::env::var("LF_PROFILE_DPP_W_EVAL").ok().as_deref() == Some("1");
        let open_ns = std::sync::atomic::AtomicU64::new(0);
        let eval_ns = std::sync::atomic::AtomicU64::new(0);
        let mul_ns = std::sync::atomic::AtomicU64::new(0);
        let blocks_done = std::sync::atomic::AtomicU64::new(0);

        let threads = rayon::current_num_threads().max(1);
        let mut window = (threads * 4).clamp(8, 512);
        let bytes_per_block = (k_star as u64).saturating_mul(std::mem::size_of::<F>() as u64);
        let max_window_bytes: u64 = std::env::var("LF_DPP_WINDOW_MAX_MB")
            .ok()
            .and_then(|s| s.parse::<u64>().ok())
            .unwrap_or(16384)
            .saturating_mul(1024 * 1024);
        if bytes_per_block != 0 {
            let max_by_mem = (max_window_bytes / bytes_per_block).max(1) as usize;
            window = window.min(max_by_mem);
        }
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
            if want_f {
                let out_chunks: Vec<Vec<F>> = ranges
                    .into_par_iter()
                    .map_init(
                        || (vec![0u16; k], vec![0u16; k], vec![0u16; k_star], vec![0u16; k_star]),
                        |(y_a, y_b, ea_u16, eb_u16), (bs, be)| -> Result<Vec<F>, String> {
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
                                y_a.fill(0u16);
                                y_b.fill(0u16);
                                for i in 0..k {
                                    let row = row_start.saturating_add(i as u64);
                                    if row >= nconstraints {
                                        break;
                                    }
                                    let a_len = read_u32(&mut rows)? as usize;
                                    let b_len = read_u32(&mut rows)? as usize;
                                    let _c_len = read_u32(&mut rows)? as usize;

                                    let (aval, bval): (u16, u16) = {
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
                                        (reduce_mod257_u64(aval_u), reduce_mod257_u64(bval_u))
                                    };
                                    y_a[i] = aval;
                                    y_b[i] = bval;
                                }

                                let t_eval = std::time::Instant::now();
                                self.code.eval_e_at_positions_into_u16(
                                    witness_pos,
                                    y_a.as_slice(),
                                    ea_u16.as_mut_slice(),
                                )?;
                                self.code.eval_e_at_positions_into_u16(
                                    witness_pos,
                                    y_b.as_slice(),
                                    eb_u16.as_mut_slice(),
                                )?;
                                if do_prof_w_eval {
                                    let dt = t_eval.elapsed();
                                    eval_ns.fetch_add(
                                        dt.as_nanos().min(u64::MAX as u128) as u64,
                                        std::sync::atomic::Ordering::Relaxed,
                                    );
                                }
                                let t_mul = std::time::Instant::now();
                                for j in 0..k_star {
                                    let prod = mul_mod257_u16_fast(ea_u16[j], eb_u16[j]);
                                    // Overwrite `eb_u16` with `w_eval_u16` (Layout A) so downstream
                                    // hooks can consume the correct mod-257 witness values.
                                    eb_u16[j] = prod;
                                    out[bi * k_star + j] = lut[prod as usize];
                                }
                                if do_prof_w_eval {
                                    let dt = t_mul.elapsed();
                                    mul_ns.fetch_add(
                                        dt.as_nanos().min(u64::MAX as u128) as u64,
                                        std::sync::atomic::Ordering::Relaxed,
                                    );
                                    blocks_done.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                                }

                                if let Some(h) = on_block_hook {
                                    h(b, eb_u16.as_slice())?;
                                }
                            }

                            Ok(out)
                        },
                    )
                    .collect::<Result<Vec<_>, _>>()?;

                if let Some(cb) = on_block.as_deref_mut() {
                    let mut b_emit = b0;
                    for chunk in out_chunks.iter() {
                        for blk in 0..(chunk.len() / k_star) {
                            let s0 = blk * k_star;
                            let s1 = s0 + k_star;
                            cb(b_emit, &chunk[s0..s1]);
                            b_emit += 1;
                        }
                    }
                }
            } else {
                // No-π0 mode: run only the u16 pipeline + hook, avoiding any `F` materialization.
                ranges
                    .into_par_iter()
                    .try_for_each_init(
                        || (vec![0u16; k], vec![0u16; k], vec![0u16; k_star], vec![0u16; k_star]),
                        |(y_a, y_b, ea_u16, eb_u16), (bs, be)| -> Result<(), String> {
                            let row_start0 = (bs as u64).saturating_mul(k as u64);
                            if row_start0 >= nconstraints {
                                return Ok(());
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

                            for b in bs..be {
                                let row_start = (b as u64).saturating_mul(k as u64);
                                if row_start >= nconstraints {
                                    break;
                                }
                                y_a.fill(0u16);
                                y_b.fill(0u16);
                                for i in 0..k {
                                    let row = row_start.saturating_add(i as u64);
                                    if row >= nconstraints {
                                        break;
                                    }
                                    let a_len = read_u32(&mut rows)? as usize;
                                    let b_len = read_u32(&mut rows)? as usize;
                                    let _c_len = read_u32(&mut rows)? as usize;

                                    let (aval, bval): (u16, u16) = {
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
                                        (reduce_mod257_u64(aval_u), reduce_mod257_u64(bval_u))
                                    };
                                    y_a[i] = aval;
                                    y_b[i] = bval;
                                }

                                let t_eval = std::time::Instant::now();
                                self.code.eval_e_at_positions_into_u16(
                                    witness_pos,
                                    y_a.as_slice(),
                                    ea_u16.as_mut_slice(),
                                )?;
                                self.code.eval_e_at_positions_into_u16(
                                    witness_pos,
                                    y_b.as_slice(),
                                    eb_u16.as_mut_slice(),
                                )?;
                                if do_prof_w_eval {
                                    let dt = t_eval.elapsed();
                                    eval_ns.fetch_add(
                                        dt.as_nanos().min(u64::MAX as u128) as u64,
                                        std::sync::atomic::Ordering::Relaxed,
                                    );
                                }

                                let t_mul = std::time::Instant::now();
                                for j in 0..k_star {
                                    let prod = mul_mod257_u16_fast(ea_u16[j], eb_u16[j]);
                                    eb_u16[j] = prod;
                                }
                                if do_prof_w_eval {
                                    let dt = t_mul.elapsed();
                                    mul_ns.fetch_add(
                                        dt.as_nanos().min(u64::MAX as u128) as u64,
                                        std::sync::atomic::Ordering::Relaxed,
                                    );
                                    blocks_done.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                                }

                                if let Some(h) = on_block_hook {
                                    h(b, eb_u16.as_slice())?;
                                }
                            }
                            Ok(())
                        },
                    )?;
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
        if !dpp::dr1cs_flpcp::is_f257_field::<F>() {
            return Err("stream_queries_for_coins_sparse: unsupported field (tiny-lock requires F257)".to_string());
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
        w_eval_u16: &[u16],
    ) -> Result<F, String> {
        let k_star = self.k_star();
        if !dpp::dr1cs_flpcp::is_f257_field::<F>() {
            return Err("dot_q3_w_eval: unsupported field (tiny-lock requires F257)".to_string());
        }
        let (_block_id, local_idx) = self.decode_block_idx(idx)?;
        let mut star_u16 = [0u16; 1];
        let mut low_u16 = [0u16; 1];
        if w_eval_u16.len() != k_star {
            return Err("dot_q3_w_eval: bad w_eval_u16 length".to_string());
        }
        self.code.dot_row_e_star_many_mod257_u16(
            &[local_idx],
            w_eval_u16,
            &mut star_u16,
            &mut low_u16,
        )?;
        let lam_u16 = (lambda.into_bigint().as_ref()[0] % 257) as u16;
        let prod = crate::lockable_ringlwe::mul_mod257_u16(lam_u16, low_u16[0]);
        let dot_u16 = if star_u16[0] >= prod { star_u16[0] - prod } else { star_u16[0] + 257 - prod };
        Ok(F::from(dot_u16 as u64))
    }

    fn dot_q3_w_eval_many(
        &self,
        idxs: &[usize],
        lambdas: &[F],
        _x: &[F],
        w_eval_u16: &[u16],
        out: &mut [F],
    ) -> Result<(), String> {
        if idxs.len() != lambdas.len() || idxs.len() != out.len() {
            return Err("dot_q3_w_eval_many: length mismatch".to_string());
        }
        if idxs.is_empty() {
            return Ok(());
        }

        // Tiny-lock path is **F257-only**. Do not silently fall back to slow generic field logic.
        if !dpp::dr1cs_flpcp::is_f257_field::<F>() {
            return Err("dot_q3_w_eval_many: unsupported field (tiny-lock requires F257)".to_string());
        }

        // All coins must target the same streamed block.
        let (b0, _) = self.decode_block_idx(idxs[0])?;
        for &idx in idxs.iter() {
            let (b, _) = self.decode_block_idx(idx)?;
            if b != b0 {
                return Err("dot_q3_w_eval_many: mixed block ids".to_string());
            }
        }

        #[inline]
        fn sub_mod257_u16(a: u16, b: u16) -> u16 {
            if a >= b { a - b } else { a + 257 - b }
        }

        if w_eval_u16.len() != self.k_star() {
            return Err("dot_q3_w_eval_many: bad w_eval_u16 length".to_string());
        }

        // Batch in chunks to avoid large stack arrays and keep code paths predictable.
        const CHUNK: usize = 64;
        let mut locals: Vec<usize> = Vec::with_capacity(CHUNK);
        let mut star_u16: Vec<u16> = vec![0u16; CHUNK];
        let mut low_u16: Vec<u16> = vec![0u16; CHUNK];

        let mut off = 0usize;
        while off < idxs.len() {
            let end = (off + CHUNK).min(idxs.len());
            let n = end - off;
            locals.clear();
            for &idx in &idxs[off..end] {
                let (_, local) = self.decode_block_idx(idx)?;
                locals.push(local);
            }

            self.code.dot_row_e_star_many_mod257_u16(
                &locals,
                w_eval_u16,
                &mut star_u16[..n],
                &mut low_u16[..n],
            )?;

            for j in 0..n {
                let lam_u16 = (lambdas[off + j].into_bigint().as_ref()[0] % 257) as u16;
                let prod = crate::lockable_ringlwe::mul_mod257_u16(lam_u16, low_u16[j]);
                let dot_u16 = sub_mod257_u16(star_u16[j], prod);
                out[off + j] = F::from(dot_u16 as u64);
            }
            off = end;
        }
        Ok(())
    }

    fn prove(&self, x: &[F], z_w: &[F]) -> Vec<F> {
        let mut pi = Vec::with_capacity(self.m());
        pi.extend_from_slice(z_w);
        let witness_pos = self.witness_positions_star().expect("witness_positions_star");
        self.stream_w_eval_blocks(
            &witness_pos,
            x,
            z_w,
            None,
            None,
            None,
            Some(&mut |_, w_eval| {
                pi.extend_from_slice(w_eval);
            }),
        )
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

// NOTE: test-only helpers live in the test module.

/// Prover-side streaming context: a thin wrapper around the chunked, file-backed FLPCP backend.
///
/// This is intentionally **not** returned from arming: arming publishes a public lock artifact;
/// proving is a separate role that may happen much later by a different party.
pub struct WeRingLweProverContext<F: PrimeField + FftField> {
    dpp: Theorem43Dpp<F, FileBackedChunkedMulCodeDr1csNpFlpcpSparse<F, TensorRsMulCode<F>>>,
}

impl<F: PrimeField + FftField> WeRingLweProverContext<F> {
    pub fn stream_pi0_and_collect_abg_full(
        &self,
        x: &[F],
        z_w: &[F],
        coins_list: &[Theorem43Coins<F>],
        on_pi0_chunk: Option<&mut dyn FnMut(&[F])>,
    ) -> Result<Vec<dpp::theorem43::Theorem43AbgFull<F>>, String> {
        self.dpp
            .stream_pi0_and_collect_abg_full(x, z_w, coins_list, on_pi0_chunk)
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
}

pub(crate) struct WeRingLweLogicalLockArmOut<F: PrimeField> {
    pub reps: Vec<RingLweLockArtifact<F>>,
}

/// Arm (publish) a RingLWE lock artifact from a public DR1CS instance and statement `x`.
///
/// Canonical arming uses `gate_mix_k` independent residual-gate mixes per lock.
///
/// Security note:
/// - `gate_mix_k` controls the multiplicative residual gate strength (`Pr[false-open] ~ 257^-K`).
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
    gate_mix_k: u16,
    s_override: Option<u16>,
    payload: &[u8],
    rng: &mut impl rand::RngCore,
) -> Result<WeRingLweLockArmOut<F>, String> {
    // Canonical sparse-hint lock + residual-gate mixes (multiplicative kill-switch).
    if gate_mix_k == 0 {
        return Err("arm_we_ringlwe_lock_from_dr1cs: gate_mix_k=0".to_string());
    }

    // Inline construction so we can use the chunked backend.
    let code = TensorRsMulCode::<F>::new(48, 3)?;
    let flpcp =
        FileBackedChunkedMulCodeDr1csNpFlpcpSparse::<F, _>::new(dr1cs.clone(), public_len, code)?;
    let dpp = Theorem43Dpp::<F, _>::new(flpcp.clone())?;

    let blocks: usize = flpcp.blocks();
    if blocks == 0 {
        return Err("arm_we_ringlwe_lock_from_dr1cs: blocks=0".to_string());
    }

    // Statement commitment `c_stmt` is the mod-257 embedding of the public statement digest.
    let c_stmt: Vec<F> = stmt_digest
        .iter()
        .map(|e| F::from((e.into_bigint().as_ref()[0] % 257) as u64))
        .collect();

    // Hash a 64-byte digest down to 32 bytes for armer-secret derivation.
    let stmt_digest_bytes32: [u8; 32] = {
        use sha2::Digest;
        let mut h = sha2::Sha256::new();
        h.update(b"LFP_STMT_DIGEST32_V1");
        for e in &stmt_digest {
            let v = (e.into_bigint().as_ref()[0] % 257) as u16;
            h.update(v.to_le_bytes());
        }
        h.finalize().into()
    };
    let armer_secret: Vec<F> =
        crate::we_statement::derive_armer_secret::<F>(armer_seed, stmt_digest_bytes32, lock_j, 32);

    // Derive per-lock public coin inputs (anchor_block_id, rep_id) deterministically, with retry.
    //
    // We retry because `accepting_set_shifted` must avoid 0 (otherwise the corresponding branch key
    // is public), and because we may want future hint-budget / sparsity policies to reject.
    #[inline]
    fn derive_anchor_block_id(
        armer_seed32: &[u8; 32],
        stmt_digest: &[F257; 32],
        lock_j: u64,
        blocks: usize,
    ) -> u32 {
        use sha2::Digest;
        let mut h = sha2::Sha256::new();
        h.update(b"LFP_ANCHOR_BLOCK_V1");
        h.update(armer_seed32);
        for e in stmt_digest {
            let v = (e.into_bigint().as_ref()[0] % 257) as u16;
            h.update(v.to_le_bytes());
        }
        h.update(&lock_j.to_le_bytes());
        let out: [u8; 32] = h.finalize().into();
        let v = u64::from_le_bytes(out[0..8].try_into().unwrap_or([0u8; 8]));
        ((v as usize) % blocks.max(1)) as u32
    }

    let anchor_block_id: u32 = derive_anchor_block_id(&armer_seed, &stmt_digest, lock_j, blocks);
    let anchor_block_usize: usize = anchor_block_id as usize;

    let max_tries = policy.max_rep_tries.max(1);
    for try_i in 0..max_tries {
        let rep_id = derive_rep_id_try(
            &armer_seed,
            &stmt_digest,
            lock_j,
            anchor_block_usize,
            policy.base_rep_id,
            /*channel_id=*/ 0,
            /*rep=*/ 0,
            try_i as u64,
        );

        // Arm DPP lock artifact (public coins + toxic coeffs).
        let art = dpp.arm(c_stmt.as_slice(), x, armer_secret.as_slice(), anchor_block_usize, rep_id)?;
        if std::env::var("LFP_DEBUG_IDENTITY").ok().as_deref() == Some("1") {
            let rep_filter = std::env::var("LFP_DEBUG_REP_ID")
                .ok()
                .and_then(|s| s.parse::<u64>().ok());
            if rep_filter.is_none() || rep_filter == Some(rep_id) {
                let c1 = art.coeffs.get(0).copied().unwrap_or(F::ZERO);
                let c2 = art.coeffs.get(1).copied().unwrap_or(F::ZERO);
                eprintln!(
                    "[LF_ID_COEFF] rep_id={} c1={} c2={} coeff_len={}",
                    rep_id,
                    crate::lockable_ringlwe::field_mod257_u16(&c1),
                    crate::lockable_ringlwe::field_mod257_u16(&c2),
                    art.coeffs.len()
                );
            }
        }

        // Build sparse packed query blocks for the proof `π`.
        let pi_len = dpp.proof_len();
        let mut qacc = QueryBlockAccumulator::<F>::new(pi_len)?;
        let mut scratch = dpp.query_scratch();
        let offset = dpp.stream_query_terms_for_pi(
            x,
            &art.coins,
            art.coeffs.as_slice(),
            &mut scratch,
            &mut |idx, coeff| {
                // Best-effort: ignore out-of-range indices (should never happen).
                let _ = qacc.add_term(&coeff, idx);
            },
        )?;
        let q_blocks = qacc.into_sparse_blocks();

        let accepting_set_raw = art.accepting_set;
        let accepting_set = [accepting_set_raw[0] - offset, accepting_set_raw[1] - offset];
        if std::env::var("LFP_DEBUG_IDENTITY").ok().as_deref() == Some("1") {
            let rep_filter = std::env::var("LFP_DEBUG_REP_ID")
                .ok()
                .and_then(|s| s.parse::<u64>().ok());
            if rep_filter.is_none() || rep_filter == Some(rep_id) {
                eprintln!(
                    "[LF_ID_ARM_PRE] rep_id={} c_hit={} offset={} a_raw=({}, {}) a_shift=({}, {})",
                    rep_id,
                    crate::lockable_ringlwe::field_mod257_u16(&art.coins.c_hit),
                    crate::lockable_ringlwe::field_mod257_u16(&offset),
                    crate::lockable_ringlwe::field_mod257_u16(&accepting_set_raw[0]),
                    crate::lockable_ringlwe::field_mod257_u16(&accepting_set_raw[1]),
                    crate::lockable_ringlwe::field_mod257_u16(&accepting_set[0]),
                    crate::lockable_ringlwe::field_mod257_u16(&accepting_set[1]),
                );
            }
        }

        // Pack UV membership bits (subset indicator over {1..=256}) into 32 bytes.
        //
        // NOTE: these bits are toxic waste and must not be published directly. We encrypt them
        // under a pad derived from the hidden `s` inside `arm_ringlwe_lock`.
        if art.uv_bits.len() != 256 {
            return Err(format!(
                "arm_we_ringlwe_lock_from_dr1cs: uv_bits length mismatch (got {})",
                art.uv_bits.len()
            ));
        }
        let mut ubits_plain = [0u8; 32];
        for lam in 1usize..=256usize {
            let bit = art.uv_bits[lam - 1] & 1;
            if bit != 0 {
                let i = lam - 1;
                ubits_plain[i / 8] |= 1u8 << (i % 8);
            }
        }

        // Residual gate hint material: K mixes over the per-block residual vector err[b].
        let gate_k: usize = gate_mix_k as usize;
        let blocks_per_mix: usize = (blocks + 63) / 64;
        let mut mixes: Vec<PackedF257Block64> = Vec::with_capacity(gate_k * blocks_per_mix);
        {
            use rand::{RngCore, SeedableRng};
            use rand_chacha::ChaCha20Rng;
            use sha2::Digest;
            let mut hh = sha2::Sha256::new();
            hh.update(b"LFP_ERR_GATE_MIX_V1");
            hh.update(&armer_seed);
            hh.update(&stmt_digest_bytes32);
            hh.update(&lock_j.to_le_bytes());
            hh.update(&rep_id.to_le_bytes());
            let seed: [u8; 32] = hh.finalize().into();
            let mut prg = ChaCha20Rng::from_seed(seed);
            for _mix_i in 0..gate_k {
                for blk in 0..blocks_per_mix {
                    let mut vals = [0u8; 64];
                    let mut is256_mask: u64 = 0;
                    for j in 0..64 {
                        let idx = blk * 64 + j;
                        if idx >= blocks {
                            vals[j] = 0;
                            continue;
                        }
                        let r = (prg.next_u32() % 257) as u16;
                        vals[j] = (r & 0xFF) as u8;
                        if r == 256 {
                            is256_mask |= 1u64 << j;
                        }
                    }
                    mixes.push(PackedF257Block64::Dense { vals, is256_mask });
                }
            }
        }
        let gate_hints = ErrGateHints {
            k: gate_k as u16,
            blocks_per_mix: blocks_per_mix as u32,
            mixes,
        };

        match arm_ringlwe_lock::<F>(
            c_stmt.clone(),
            accepting_set,
            anchor_block_id,
            rep_id,
            s_override,
            blocks as u32,
            art.coins,
            offset,
            x.len(),
            pi_len,
            ubits_plain,
            q_blocks,
            gate_hints,
            params.clone(),
            payload,
            rng,
        ) {
            Ok(lock) => return Ok(WeRingLweLockArmOut { lock }),
            Err(e) => {
                if e.contains("shifted accepting set contains 0") && try_i + 1 < max_tries {
                    continue;
                }
                return Err(e);
            }
        }
    }
    Err("arm_we_ringlwe_lock_from_dr1cs: exhausted rep_id retry budget".to_string())
}

/// Arm the **LF+ tiny-field WE gate** (Poseidon(F257) + CM-coin surfaces) as a Theorem-4.3 lock.
///
/// This is the main wiring from:
/// - `we_gate_arith::build_we_dr1cs_for_plus_proof_shape_tiny` (arm-time instance construction)
/// into:
/// - `arm_we_ringlwe_lock_from_dr1cs` (Theorem-4.3 + Ring-LWE wrapper).
///
/// Notes:
/// - The statement binding is carried by `stmt_digest` via `c_stmt` inside `arm_we_ringlwe_lock_from_dr1cs`.
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
    gate_mix_k: u16,
    payload: &[u8],
    s_override: Option<u16>,
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
        gate_mix_k,
        s_override,
        payload,
        rng,
    )
}

#[cfg(feature = "we_gate")]
pub(crate) fn arm_lfplus_ringlwe_logical_lock<R>(
    shape: we_gate_arith::WeDr1csShape<F257>,
    stmt_digest: &[F257; 32],
    armer_seed: [u8; 32],
    lock_j: u64,
    block_id: usize,
    policy: WeRingLweLockArmingPolicy,
    ringlwe_params: RingLweParams,
    gate_mix_k: u16,
    reps: usize,
    payload: &[u8],
    rng: &mut impl rand::RngCore,
) -> Result<WeRingLweLogicalLockArmOut<F257>, String>
where
    R: OverField + CoeffRing + PolyRing,
    R::BaseRing: Zq + ark_ff::Field + ark_ff::PrimeField,
{
    use rand::{rngs::StdRng, SeedableRng};

    if reps == 0 {
        return Err("arm_lfplus_ringlwe_logical_lock: reps=0".to_string());
    }
    if payload.len() != 32 {
        return Err("arm_lfplus_ringlwe_logical_lock: payload must be exactly 32 bytes".to_string());
    }

    // Rep unlinkability: split the logical-lock master payload into `reps` XOR-shares.
    //
    // - Each rep carries a different 32-byte plaintext, so a package-only attacker cannot
    //   intersect candidate plaintext sets across reps to recover the logical payload.
    // - Honest recovery can reconstruct the logical payload only after selecting one candidate
    //   per rep (a global check stage), by XORing the rep plaintexts.
    let mut rep_payloads: Vec<[u8; 32]> = Vec::with_capacity(reps);
    if reps == 1 {
        let mut one = [0u8; 32];
        one.copy_from_slice(payload);
        rep_payloads.push(one);
    } else {
        let mut acc = [0u8; 32];
        for _ in 0..(reps - 1) {
            let mut m = [0u8; 32];
            rng.fill_bytes(&mut m);
            for i in 0..32 {
                acc[i] ^= m[i];
            }
            rep_payloads.push(m);
        }
        let mut last = [0u8; 32];
        last.copy_from_slice(payload);
        for i in 0..32 {
            last[i] ^= acc[i];
        }
        rep_payloads.push(last);
    }

    // Deterministic per-rep RNG fan-out from caller RNG.
    let mut rep_seeds: Vec<[u8; 32]> = Vec::with_capacity(reps);
    for _ in 0..reps {
        let mut s = [0u8; 32];
        rng.fill_bytes(&mut s);
        rep_seeds.push(s);
    }

    let mut out: Vec<RingLweLockArtifact<F257>> = Vec::with_capacity(reps);
    // Shared scalar `s` across reps enables rep-intersection at decap time (GPT-PRO strategy).
    let s_shared: u16 = crate::lockable_ringlwe::sample_nonzero_f257_scalar(rng);
    for r in 0..reps {
        let mut rrng = StdRng::from_seed(rep_seeds[r]);
        let rep_base = policy
            .base_rep_id
            .checked_add(r as u64)
            .ok_or_else(|| format!("rep base overflow: base={} r={}", policy.base_rep_id, r))?;
        let rep_policy = WeRingLweLockArmingPolicy {
            base_rep_id: rep_base,
            max_rep_tries: policy.max_rep_tries,
            hint_budget_bytes: policy.hint_budget_bytes,
        };
        let arm = arm_lfplus_ringlwe_lock::<R>(
            shape.clone(),
            stmt_digest,
            armer_seed,
            lock_j,
            block_id,
            rep_policy,
            ringlwe_params.clone(),
            gate_mix_k,
            rep_payloads[r].as_slice(),
            Some(s_shared),
            &mut rrng,
        )?;
        out.push(arm.lock);
    }

    Ok(WeRingLweLogicalLockArmOut { reps: out })
}
