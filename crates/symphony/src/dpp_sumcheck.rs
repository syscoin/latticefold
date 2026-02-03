//! Sumcheck verifier arithmetization (sparse dR1CS over a prime field).
//!
//! This module is an intermediate building block toward arithmetizing the full Π_fold verifier.
//! It encodes **verification** of the multilinear sumcheck proof messages as constraints.
//!
//! Current scope:
//! - degree-3 prover messages (4 evaluations at points 0,1,2,3), matching the usage in Symphony.
//! - shared-randomness schedule for two sumchecks (same `r_i` per round).
//!
//! Note: this does **not** (yet) encode the post-sumcheck algebraic checks (Eq(26), monomial
//! recomputation, Step-5). Those are the next layer on top of the verified subclaims.

use ark_ff::PrimeField;
use ark_serialize::CanonicalSerialize;

use std::collections::BTreeMap;
use core::ops::Range;
use std::sync::OnceLock;

use crate::dpp_poseidon::{Constraint, SparseDr1csInstance};
use crate::file_backed_dr1cs::{FileBackedSparseDr1csInstance, SparseDr1csFileWriter};
#[cfg(unix)]
use crate::file_backed_dr1cs::FileBackedRangeWriter;

#[derive(Clone, Debug, Default)]
pub struct Dr1csProfileCounts {
    pub vars: u64,
    pub constraints: u64,
}

#[derive(Debug)]
pub struct Dr1csBuilder<F: PrimeField> {
    pub assignment: Vec<F>,
    pub rows: Vec<Constraint>,
    pub a_terms: Vec<(F, usize)>,
    pub b_terms: Vec<(F, usize)>,
    pub c_terms: Vec<(F, usize)>,
    /// Optional file-backed sink for constraints/term pools.
    ///
    /// When present, `rows/a_terms/b_terms/c_terms` are not populated (to avoid giant Vec pools).
    file_sink: Option<Dr1csFileSink<F>>,
    file_rows: u64,
    file_a_terms: u64,
    file_b_terms: u64,
    file_c_terms: u64,
    /// Var index remap for direct-to-merged writing.
    ///
    /// Global var idx = 0 if local==0 else local + file_var_tail_off.
    file_var_tail_off: u32,
    // File-backed staging: accumulate large blocks and flush via raw-block APIs.
    fb_stage_bytes: usize,
    fb_stage_limit_bytes: usize,
    fb_a_coeffs: Vec<u8>,
    fb_a_idx: Vec<u32>,
    fb_b_coeffs: Vec<u8>,
    fb_b_idx: Vec<u32>,
    fb_c_coeffs: Vec<u8>,
    fb_c_idx: Vec<u32>,
    fb_row_lens: Vec<u32>, // 3*u32 per row
    fb_tmp_idx: Vec<u32>,  // scratch for remapping idx blocks
    /// Cache for reusing a byte var's bit-decomposition across gadgets.
    ///
    /// Key: byte variable index. Value: 8 boolean bit variable indices (little-endian).
    pub byte_bits_cache: BTreeMap<usize, [usize; 8]>,

    /// Cache for reusing `u64_bytes_to_bal16_digits` expansions across gadgets.
    ///
    /// Key: 8 byte variables (little-endian). Value: balanced-base16 digits (len 17).
    pub u64_bal16_cache: BTreeMap<[usize; 8], Vec<usize>>,
    /// Cache for reusing `u32_bytes_to_bal16_digits` expansions across gadgets.
    ///
    /// Key: 4 byte variables (little-endian). Value: balanced-base16 digits (len 9).
    pub u32_bal16_cache: BTreeMap<[usize; 4], Vec<usize>>,

    /// Cache for a single variable constrained to 0.
    ///
    /// Many gadgets need a reusable 0 variable; allocating it once avoids repeated boolean
    /// decompositions for constant-zero digits/limbs.
    pub zero_var_cache: Option<usize>,

    // -------------------------------------------------------------------------
    // Optional profiling (enabled by env var; low overhead when disabled).
    // -------------------------------------------------------------------------
    pub profile_enabled: bool,
    pub profile_current: Option<&'static str>,
    pub profile: BTreeMap<&'static str, Dr1csProfileCounts>,
    /// Optional debug tag to filter constraint-level debugging (e.g. "cm0", "cm1", "base_glue").
    ///
    /// In file-backed mode this is derived from the output directory name.
    debug_tag: Option<String>,
}

#[derive(Debug)]
enum Dr1csFileSink<F: PrimeField> {
    Append(SparseDr1csFileWriter<F>),
    #[cfg(unix)]
    Range(FileBackedRangeWriter),
    /// Count-only: maintain file_* counters but do not write any pools/rows.
    Count,
}

#[derive(Clone, Debug, Default)]
pub struct Dr1csFileCounts {
    pub rows: u64,
    pub a_terms: u64,
    pub b_terms: u64,
    pub c_terms: u64,
}

#[cfg(unix)]
#[derive(Clone, Debug, Default)]
pub struct Dr1csRangeResult {
    pub counts: Dr1csFileCounts,
    pub ckpts: Vec<(u64, u64, u64, u64)>,
}

#[cfg(unix)]
#[derive(Debug)]
pub struct Dr1csRangeSnapshot {
    pub out_fc_a: std::fs::File,
    pub out_fi_a: std::fs::File,
    pub out_fc_b: std::fs::File,
    pub out_fi_b: std::fs::File,
    pub out_fc_c: std::fs::File,
    pub out_fi_c: std::fs::File,
    pub out_rows: std::fs::File,
    // Base offsets (global terms/rows).
    pub base_a_terms: u64,
    pub base_b_terms: u64,
    pub base_c_terms: u64,
    pub base_rows: u64,
    // Already-written counts by this range writer (local to this shard).
    pub a_terms_written: u64,
    pub b_terms_written: u64,
    pub c_terms_written: u64,
    pub rows_written: u64,
    /// Variable tail offset applied when writing term indices into the merged instance.
    pub var_tail_off: u32,
}

impl<F: PrimeField + CanonicalSerialize> Dr1csBuilder<F> {
    #[inline]
    fn map_idx_u32(&self, idx: usize) -> u32 {
        if idx == 0 {
            return 0;
        }
        let add = self.file_var_tail_off as u64;
        let v = (idx as u64).saturating_add(add);
        v.try_into()
            .unwrap_or_else(|_| panic!("file-backed dr1cs: mapped var idx overflow u32 (idx={idx} add={add})"))
    }

    fn stage_limit_bytes() -> usize {
        // Keep the existing knob name (historically introduced for Poseidon).
        // Default: 512 MiB (favor throughput on large machines).
        let mb = std::env::var("LFP_POSEIDON_FILE_BACKED_STAGE_MB")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .filter(|&v| v > 0)
            .unwrap_or(512);
        mb.saturating_mul(1024 * 1024)
    }

    #[inline]
    fn fb_coeff_u16(&self, coef: F) -> u16 {
        let rep = coef.into_bigint();
        let limbs = rep.as_ref();
        debug_assert!(!limbs.is_empty());
        let v = limbs[0];
        v.try_into().unwrap_or_else(|_| panic!("file-backed dr1cs: coeff overflow u16 (v={v})"))
    }

    #[inline]
    fn fb_maybe_flush(&mut self) -> Result<(), String> {
        if self.fb_stage_bytes >= self.fb_stage_limit_bytes {
            self.fb_flush()?;
        }
        Ok(())
    }

    fn fb_flush(&mut self) -> Result<(), String> {
        let Some(sink) = self.file_sink.as_mut() else { return Ok(()); };
        if self.fb_row_lens.is_empty() {
            return Ok(());
        }
        // Write term pools first, then row lens block.
        match sink {
            Dr1csFileSink::Append(w) => {
                w.push_a_terms_raw_block(&self.fb_a_coeffs, &self.fb_a_idx)?;
                w.push_b_terms_raw_block(&self.fb_b_coeffs, &self.fb_b_idx)?;
                w.push_c_terms_raw_block(&self.fb_c_coeffs, &self.fb_c_idx)?;
                w.push_constraint_lens_block(&self.fb_row_lens)?;
            }
            #[cfg(unix)]
            Dr1csFileSink::Range(w) => {
                w.push_a_terms_raw_block(&self.fb_a_coeffs, &self.fb_a_idx)?;
                w.push_b_terms_raw_block(&self.fb_b_coeffs, &self.fb_b_idx)?;
                w.push_c_terms_raw_block(&self.fb_c_coeffs, &self.fb_c_idx)?;
                w.push_constraint_lens_block(&self.fb_row_lens)?;
            }
            Dr1csFileSink::Count => {
                // No-op: counts are derived from buffer sizes below.
            }
        }
        self.file_a_terms = self.file_a_terms.saturating_add(self.fb_a_idx.len() as u64);
        self.file_b_terms = self.file_b_terms.saturating_add(self.fb_b_idx.len() as u64);
        self.file_c_terms = self.file_c_terms.saturating_add(self.fb_c_idx.len() as u64);
        self.file_rows = self.file_rows.saturating_add((self.fb_row_lens.len() / 3) as u64);

        self.fb_a_coeffs.clear();
        self.fb_a_idx.clear();
        self.fb_b_coeffs.clear();
        self.fb_b_idx.clear();
        self.fb_c_coeffs.clear();
        self.fb_c_idx.clear();
        self.fb_row_lens.clear();
        self.fb_stage_bytes = 0;
        Ok(())
    }

    /// Flush any staged file-backed buffers immediately.
    ///
    /// This is primarily used by parallel lowering code that needs a stable notion of
    /// "current end offsets" before doing direct range `pwrite`s.
    pub fn file_backed_flush(&mut self) -> Result<(), String> {
        self.fb_flush()
    }
    pub fn new() -> Self {
        let profile_enabled = match std::env::var("LF_PROFILE_DR1CS") {
            Ok(v) => v != "0",
            Err(_) => false,
        };
        Self {
            assignment: vec![F::ONE],
            rows: Vec::new(),
            a_terms: Vec::new(),
            b_terms: Vec::new(),
            c_terms: Vec::new(),
            file_sink: None,
            file_rows: 0,
            file_a_terms: 0,
            file_b_terms: 0,
            file_c_terms: 0,
            file_var_tail_off: 0,
            fb_stage_bytes: 0,
            fb_stage_limit_bytes: Self::stage_limit_bytes(),
            fb_a_coeffs: Vec::new(),
            fb_a_idx: Vec::new(),
            fb_b_coeffs: Vec::new(),
            fb_b_idx: Vec::new(),
            fb_c_coeffs: Vec::new(),
            fb_c_idx: Vec::new(),
            fb_row_lens: Vec::new(),
            fb_tmp_idx: Vec::new(),
            byte_bits_cache: BTreeMap::new(),
            u64_bal16_cache: BTreeMap::new(),
            u32_bal16_cache: BTreeMap::new(),
            zero_var_cache: None,
            profile_enabled,
            profile_current: None,
            profile: BTreeMap::new(),
            debug_tag: None,
        }
    }
    /// Create a builder that streams constraints/terms to `dir` instead of storing giant Vec pools.
    pub fn new_file_backed(dir: impl AsRef<std::path::Path>) -> Result<Self, String>
    where
        F: CanonicalSerialize,
    {
        let mut b = Self::new();
        b.debug_tag = dir
            .as_ref()
            .file_name()
            .map(|s| s.to_string_lossy().to_string());
        b.file_sink = Some(Dr1csFileSink::Append(SparseDr1csFileWriter::<F>::create(dir)?));
        Ok(b)
    }

    #[cfg(unix)]
    pub fn new_file_backed_range(writer: FileBackedRangeWriter, var_tail_off: u32) -> Self {
        let mut b = Self::new();
        b.file_var_tail_off = var_tail_off;
        b.file_sink = Some(Dr1csFileSink::Range(writer));
        b
    }

    /// Count-only builder: computes assignment and exact row/term counts but does not write pools/rows.
    pub fn new_count_only() -> Self {
        let mut b = Self::new();
        b.file_sink = Some(Dr1csFileSink::Count);
        b
    }

    #[inline]
    pub fn is_file_backed(&self) -> bool {
        self.file_sink.is_some()
    }

    /// If file-backed, return the fixed serialized coefficient size (bytes).
    #[inline]
    pub fn file_coeff_size(&self) -> Option<usize> {
        match self.file_sink.as_ref() {
            Some(Dr1csFileSink::Append(s)) => Some(s.coeff_size()),
            Some(Dr1csFileSink::Count) => Some(2),
            #[cfg(unix)]
            Some(Dr1csFileSink::Range(_)) => Some(2),
            None => None,
        }
    }

    /// If file-backed, return the current on-disk counters (rows, a_terms, b_terms, c_terms).
    #[inline]
    pub fn file_counts(&self) -> Option<(u64, u64, u64, u64)> {
        if self.file_sink.is_some() {
            Some((self.file_rows, self.file_a_terms, self.file_b_terms, self.file_c_terms))
        } else {
            None
        }
    }

    /// Unix-only: if this is a range-backed builder, return a snapshot with cloned files and offsets.
    ///
    /// This is used by parallel lowering code which `pwrite`s into disjoint ranges directly.
    #[cfg(unix)]
    pub fn file_range_snapshot(&self) -> Result<Option<Dr1csRangeSnapshot>, String> {
        let Some(sink) = self.file_sink.as_ref() else {
            return Ok(None);
        };
        let Dr1csFileSink::Range(w) = sink else {
            return Ok(None);
        };
        let (out_fc_a, out_fi_a, out_fc_b, out_fi_b, out_fc_c, out_fi_c, out_rows) = w.try_clone_all_files()?;
        let (base_a_terms, base_b_terms, base_c_terms, base_rows) = w.base_offsets();
        let (a_terms_written, b_terms_written, c_terms_written, rows_written) = w.written_counts();
        Ok(Some(Dr1csRangeSnapshot {
            out_fc_a,
            out_fi_a,
            out_fc_b,
            out_fi_b,
            out_fc_c,
            out_fi_c,
            out_rows,
            base_a_terms,
            base_b_terms,
            base_c_terms,
            base_rows,
            a_terms_written,
            b_terms_written,
            c_terms_written,
            rows_written,
            var_tail_off: self.file_var_tail_off,
        }))
    }

    /// Unix-only: commit externally `pwrite`d range output into this builder's counters and ckpts.
    #[cfg(unix)]
    pub fn file_range_commit_parallel_write(
        &mut self,
        rows: u64,
        a_terms: u64,
        b_terms: u64,
        c_terms: u64,
        ckpts: Vec<(u64, u64, u64, u64)>,
    ) -> Result<(), String> {
        // Flush any staged sequential buffers so all offsets match the snapshot.
        self.fb_flush()?;
        let sink = self
            .file_sink
            .as_mut()
            .ok_or_else(|| "file_range_commit_parallel_write called on non-file-backed builder".to_string())?;
        let Dr1csFileSink::Range(w) = sink else {
            return Err("file_range_commit_parallel_write called on non-range builder".to_string());
        };
        w.bump_written_counts(rows, a_terms, b_terms, c_terms);
        w.extend_ckpts(ckpts);
        self.file_rows = self.file_rows.saturating_add(rows);
        self.file_a_terms = self.file_a_terms.saturating_add(a_terms);
        self.file_b_terms = self.file_b_terms.saturating_add(b_terms);
        self.file_c_terms = self.file_c_terms.saturating_add(c_terms);
        Ok(())
    }

    /// Return true if this builder is in **count-only** mode (no pools/rows written).
    #[inline]
    pub fn is_count_only(&self) -> bool {
        matches!(self.file_sink, Some(Dr1csFileSink::Count))
    }

    /// Count-only fast path: bump row/term counters without constructing any row/term buffers.
    ///
    /// This is intended for structural counting (Pass0) where the *shape* is known but we want
    /// to avoid building/streaming millions of constraints.
    pub fn count_only_bump_counts(
        &mut self,
        rows: u64,
        a_terms: u64,
        b_terms: u64,
        c_terms: u64,
    ) -> Result<(), String> {
        if !self.is_count_only() {
            return Err("count_only_bump_counts called on non-count-only builder".to_string());
        }
        self.file_rows = self.file_rows.saturating_add(rows);
        self.file_a_terms = self.file_a_terms.saturating_add(a_terms);
        self.file_b_terms = self.file_b_terms.saturating_add(b_terms);
        self.file_c_terms = self.file_c_terms.saturating_add(c_terms);
        Ok(())
    }

    /// File-backed: append a block of A-terms (tiny format, u32 indices).
    pub fn file_push_a_terms_raw_block(&mut self, coeff_bytes: &[u8], idx: &[u32]) -> Result<(), String> {
        let sink = self
            .file_sink
            .as_mut()
            .ok_or_else(|| "file_push_a_terms_raw_block called on non-file-backed builder".to_string())?;
        if matches!(sink, Dr1csFileSink::Count) {
            self.file_a_terms = self.file_a_terms.saturating_add(idx.len() as u64);
            return Ok(());
        }
        let idx_mapped: &[u32] = if self.file_var_tail_off == 0 {
            idx
        } else {
            self.fb_tmp_idx.clear();
            self.fb_tmp_idx.reserve(idx.len());
            let add = self.file_var_tail_off;
            for &v in idx {
                self.fb_tmp_idx.push(if v == 0 { 0 } else { v.saturating_add(add) });
            }
            &self.fb_tmp_idx
        };
        match sink {
            Dr1csFileSink::Append(w) => w.push_a_terms_raw_block(coeff_bytes, idx_mapped)?,
            #[cfg(unix)]
            Dr1csFileSink::Range(w) => w.push_a_terms_raw_block(coeff_bytes, idx_mapped)?,
            Dr1csFileSink::Count => {}
        }
        self.file_a_terms = self.file_a_terms.saturating_add(idx.len() as u64);
        Ok(())
    }

    /// File-backed: append a block of B-terms (tiny format, u32 indices).
    pub fn file_push_b_terms_raw_block(&mut self, coeff_bytes: &[u8], idx: &[u32]) -> Result<(), String> {
        let sink = self
            .file_sink
            .as_mut()
            .ok_or_else(|| "file_push_b_terms_raw_block called on non-file-backed builder".to_string())?;
        if matches!(sink, Dr1csFileSink::Count) {
            self.file_b_terms = self.file_b_terms.saturating_add(idx.len() as u64);
            return Ok(());
        }
        let idx_mapped: &[u32] = if self.file_var_tail_off == 0 {
            idx
        } else {
            self.fb_tmp_idx.clear();
            self.fb_tmp_idx.reserve(idx.len());
            let add = self.file_var_tail_off;
            for &v in idx {
                self.fb_tmp_idx.push(if v == 0 { 0 } else { v.saturating_add(add) });
            }
            &self.fb_tmp_idx
        };
        match sink {
            Dr1csFileSink::Append(w) => w.push_b_terms_raw_block(coeff_bytes, idx_mapped)?,
            #[cfg(unix)]
            Dr1csFileSink::Range(w) => w.push_b_terms_raw_block(coeff_bytes, idx_mapped)?,
            Dr1csFileSink::Count => {}
        }
        self.file_b_terms = self.file_b_terms.saturating_add(idx.len() as u64);
        Ok(())
    }

    /// File-backed: append a block of C-terms (tiny format, u32 indices).
    pub fn file_push_c_terms_raw_block(&mut self, coeff_bytes: &[u8], idx: &[u32]) -> Result<(), String> {
        let sink = self
            .file_sink
            .as_mut()
            .ok_or_else(|| "file_push_c_terms_raw_block called on non-file-backed builder".to_string())?;
        if matches!(sink, Dr1csFileSink::Count) {
            self.file_c_terms = self.file_c_terms.saturating_add(idx.len() as u64);
            return Ok(());
        }
        let idx_mapped: &[u32] = if self.file_var_tail_off == 0 {
            idx
        } else {
            self.fb_tmp_idx.clear();
            self.fb_tmp_idx.reserve(idx.len());
            let add = self.file_var_tail_off;
            for &v in idx {
                self.fb_tmp_idx.push(if v == 0 { 0 } else { v.saturating_add(add) });
            }
            &self.fb_tmp_idx
        };
        match sink {
            Dr1csFileSink::Append(w) => w.push_c_terms_raw_block(coeff_bytes, idx_mapped)?,
            #[cfg(unix)]
            Dr1csFileSink::Range(w) => w.push_c_terms_raw_block(coeff_bytes, idx_mapped)?,
            Dr1csFileSink::Count => {}
        }
        self.file_c_terms = self.file_c_terms.saturating_add(idx.len() as u64);
        Ok(())
    }

    /// File-backed: append a block of constraint row lengths (3 u32 words per row).
    pub fn file_push_constraint_lens_block(&mut self, lens: &[u32]) -> Result<(), String> {
        let sink = self
            .file_sink
            .as_mut()
            .ok_or_else(|| "file_push_constraint_lens_block called on non-file-backed builder".to_string())?;
        if matches!(sink, Dr1csFileSink::Count) {
            self.file_rows = self.file_rows.saturating_add((lens.len() / 3) as u64);
            return Ok(());
        }
        match sink {
            Dr1csFileSink::Append(w) => w.push_constraint_lens_block(lens)?,
            #[cfg(unix)]
            Dr1csFileSink::Range(w) => w.push_constraint_lens_block(lens)?,
            Dr1csFileSink::Count => {}
        }
        self.file_rows = self.file_rows.saturating_add((lens.len() / 3) as u64);
        Ok(())
    }

    #[inline]
    pub fn nconstraints(&self) -> u64 {
        if self.file_sink.is_some() {
            self.file_rows
        } else {
            self.rows.len() as u64
        }
    }
    pub fn one(&self) -> usize { 0 }
    pub fn new_var(&mut self, value: F) -> usize {
        let idx = self.assignment.len();
        self.assignment.push(value);
        if self.profile_enabled {
            let key = self.profile_current.unwrap_or("unlabeled");
            self.profile.entry(key).or_default().vars += 1;
        }
        idx
    }
    #[inline]
    fn push_terms(pool: &mut Vec<(F, usize)>, terms: &[(F, usize)]) -> Range<usize> {
        let start = pool.len();
        pool.extend_from_slice(terms);
        let end = pool.len();
        start..end
    }

    pub fn add_constraint(&mut self, a: Vec<(F, usize)>, b: Vec<(F, usize)>, c: Vec<(F, usize)>) {
        self.add_constraint_slices(&a, &b, &c);
        if self.profile_enabled {
            let key = self.profile_current.unwrap_or("unlabeled");
            self.profile.entry(key).or_default().constraints += 1;
        }
    }

    /// Add a constraint without materializing term vectors (stream-friendly).
    ///
    /// This is used by file-backed lowering paths to avoid allocating a `Vec` per constraint.
    #[inline]
    #[track_caller]
    pub fn add_constraint_terms_iter<IA, IB, IC>(&mut self, a: IA, b: IB, c: IC)
    where
        IA: IntoIterator<Item = (F, usize)>,
        IB: IntoIterator<Item = (F, usize)>,
        IC: IntoIterator<Item = (F, usize)>,
    {
        let dbg = self.debug_should_dump_next_constraint();
        if self.file_sink.is_some() {
            if dbg {
                let a_v: Vec<(F, usize)> = a.into_iter().collect();
                let b_v: Vec<(F, usize)> = b.into_iter().collect();
                let c_v: Vec<(F, usize)> = c.into_iter().collect();
                self.debug_dump_constraint_slices("add_constraint_terms_iter", &a_v, &b_v, &c_v);
                return self.add_constraint_slices(&a_v, &b_v, &c_v);
            }
            if matches!(self.file_sink, Some(Dr1csFileSink::Count)) {
                let mut a_len: u64 = 0;
                for _ in a.into_iter() {
                    a_len += 1;
                }
                let mut b_len: u64 = 0;
                for _ in b.into_iter() {
                    b_len += 1;
                }
                let mut c_len: u64 = 0;
                for _ in c.into_iter() {
                    c_len += 1;
                }
                self.file_a_terms = self.file_a_terms.saturating_add(a_len);
                self.file_b_terms = self.file_b_terms.saturating_add(b_len);
                self.file_c_terms = self.file_c_terms.saturating_add(c_len);
                self.file_rows = self.file_rows.saturating_add(1);
            } else {
            // Stage term blocks and row lens, then flush in large blocks.
            let a_len0 = self.fb_a_idx.len();
            for (coef, idx) in a.into_iter() {
                let vv = self.fb_coeff_u16(coef);
                self.fb_a_coeffs.extend_from_slice(&vv.to_le_bytes());
                self.fb_a_idx.push(self.map_idx_u32(idx));
            }
            let a_len = self.fb_a_idx.len().saturating_sub(a_len0);
            let b_len0 = self.fb_b_idx.len();
            for (coef, idx) in b.into_iter() {
                let vv = self.fb_coeff_u16(coef);
                self.fb_b_coeffs.extend_from_slice(&vv.to_le_bytes());
                self.fb_b_idx.push(self.map_idx_u32(idx));
            }
            let b_len = self.fb_b_idx.len().saturating_sub(b_len0);
            let c_len0 = self.fb_c_idx.len();
            for (coef, idx) in c.into_iter() {
                let vv = self.fb_coeff_u16(coef);
                self.fb_c_coeffs.extend_from_slice(&vv.to_le_bytes());
                self.fb_c_idx.push(self.map_idx_u32(idx));
            }
            let c_len = self.fb_c_idx.len().saturating_sub(c_len0);
            self.fb_row_lens
                .extend_from_slice(&[(a_len as u32), (b_len as u32), (c_len as u32)]);
            self.fb_stage_bytes = self
                .fb_stage_bytes
                .saturating_add(2 * (a_len + b_len + c_len))
                .saturating_add(12);
            self.fb_maybe_flush().expect("file-backed dr1cs flush failed");
            }
        } else {
            if dbg {
                let a_v: Vec<(F, usize)> = a.into_iter().collect();
                let b_v: Vec<(F, usize)> = b.into_iter().collect();
                let c_v: Vec<(F, usize)> = c.into_iter().collect();
                self.debug_dump_constraint_slices("add_constraint_terms_iter(in-mem)", &a_v, &b_v, &c_v);
                return self.add_constraint_slices(&a_v, &b_v, &c_v);
            }
            let a0 = self.a_terms.len();
            self.a_terms.extend(a);
            let a1 = self.a_terms.len();
            let b0 = self.b_terms.len();
            self.b_terms.extend(b);
            let b1 = self.b_terms.len();
            let c0 = self.c_terms.len();
            self.c_terms.extend(c);
            let c1 = self.c_terms.len();
            self.rows.push(Constraint { a: a0..a1, b: b0..b1, c: c0..c1 });
        }
        if self.profile_enabled {
            let key = self.profile_current.unwrap_or("unlabeled");
            self.profile.entry(key).or_default().constraints += 1;
        }
    }

    #[inline]
    #[track_caller]
    pub fn add_constraint_slices(&mut self, a: &[(F, usize)], b: &[(F, usize)], c: &[(F, usize)]) {
        if self.debug_should_dump_next_constraint() {
            self.debug_dump_constraint_slices("add_constraint_slices", a, b, c);
        }
        if matches!(self.file_sink, Some(Dr1csFileSink::Count)) {
            self.file_a_terms = self.file_a_terms.saturating_add(a.len() as u64);
            self.file_b_terms = self.file_b_terms.saturating_add(b.len() as u64);
            self.file_c_terms = self.file_c_terms.saturating_add(c.len() as u64);
            self.file_rows = self.file_rows.saturating_add(1);
        } else if self.file_sink.is_some() {
            for (coef, idx) in a.iter().copied() {
                let vv = self.fb_coeff_u16(coef);
                self.fb_a_coeffs.extend_from_slice(&vv.to_le_bytes());
                self.fb_a_idx.push(self.map_idx_u32(idx));
            }
            for (coef, idx) in b.iter().copied() {
                let vv = self.fb_coeff_u16(coef);
                self.fb_b_coeffs.extend_from_slice(&vv.to_le_bytes());
                self.fb_b_idx.push(self.map_idx_u32(idx));
            }
            for (coef, idx) in c.iter().copied() {
                let vv = self.fb_coeff_u16(coef);
                self.fb_c_coeffs.extend_from_slice(&vv.to_le_bytes());
                self.fb_c_idx.push(self.map_idx_u32(idx));
            }
            self.fb_row_lens
                .extend_from_slice(&[(a.len() as u32), (b.len() as u32), (c.len() as u32)]);
            self.fb_stage_bytes = self
                .fb_stage_bytes
                .saturating_add(2 * (a.len() + b.len() + c.len()))
                .saturating_add(12);
            self.fb_maybe_flush().expect("file-backed dr1cs flush failed");
        } else {
            let ar = Self::push_terms(&mut self.a_terms, a);
            let br = Self::push_terms(&mut self.b_terms, b);
            let cr = Self::push_terms(&mut self.c_terms, c);
            self.rows.push(Constraint { a: ar, b: br, c: cr });
        }
    }

    fn debug_cfg_target_idx() -> Option<u64> {
        static TARGET: OnceLock<Option<u64>> = OnceLock::new();
        *TARGET.get_or_init(|| {
            std::env::var("LF_DEBUG_CONSTRAINT_AT")
                .ok()
                .and_then(|s| s.parse::<u64>().ok())
        })
    }

    fn debug_cfg_tag() -> Option<&'static str> {
        static TAG: OnceLock<Option<&'static str>> = OnceLock::new();
        *TAG.get_or_init(|| {
            std::env::var("LF_DEBUG_CONSTRAINT_TAG").ok().map(|s| {
                let leaked: &'static mut str = Box::leak(s.into_boxed_str());
                &*leaked
            })
        })
    }

    #[inline]
    fn next_constraint_index(&self) -> u64 {
        if self.file_sink.is_some() {
            self.file_rows
        } else {
            self.rows.len() as u64
        }
    }

    #[inline]
    fn debug_should_dump_next_constraint(&self) -> bool {
        let Some(tgt) = Self::debug_cfg_target_idx() else {
            return false;
        };
        if self.next_constraint_index() != tgt {
            return false;
        }
        let Some(tag) = Self::debug_cfg_tag() else {
            return true;
        };
        let Some(my_tag) = self.debug_tag.as_deref() else {
            return false;
        };
        my_tag == tag
    }

    fn debug_dump_constraint_slices(&self, where_: &str, a: &[(F, usize)], b: &[(F, usize)], c: &[(F, usize)]) {
        let loc = std::panic::Location::caller();
        let scope = self.profile_current.unwrap_or("unlabeled");
        let idx = self.next_constraint_index();
        let tag = self.debug_tag.as_deref().unwrap_or("<none>");
        let show = |ts: &[(F, usize)]| -> Vec<(F, usize)> { ts.iter().take(8).cloned().collect() };
        eprintln!(
            "[LF_DEBUG_CONSTRAINT] tag={tag} idx={idx} scope={scope} where={where_} caller={}:{}:{} A(len={}, head={:?}) B(len={}, head={:?}) C(len={}, head={:?})",
            loc.file(),
            loc.line(),
            loc.column(),
            a.len(),
            show(a),
            b.len(),
            show(b),
            c.len(),
            show(c),
        );
    }
    pub fn enforce_lc_times_one_eq_const(&mut self, lc: Vec<(F, usize)>) {
        let one = self.one();
        self.add_constraint_slices(&lc, &[(F::ONE, one)], &[(F::ZERO, one)]);
        if self.profile_enabled {
            let key = self.profile_current.unwrap_or("unlabeled");
            self.profile.entry(key).or_default().constraints += 1;
        }
    }
    pub fn enforce_var_eq_const(&mut self, x: usize, c: F) {
        let one = self.one();
        self.add_constraint_slices(&[(F::ONE, x)], &[(F::ONE, one)], &[(c, one)]);
        if self.profile_enabled {
            let key = self.profile_current.unwrap_or("unlabeled");
            self.profile.entry(key).or_default().constraints += 1;
        }
    }
    pub fn enforce_mul(&mut self, x: usize, y: usize, out: usize) {
        self.add_constraint_slices(&[(F::ONE, x)], &[(F::ONE, y)], &[(F::ONE, out)]);
        if self.profile_enabled {
            let key = self.profile_current.unwrap_or("unlabeled");
            self.profile.entry(key).or_default().constraints += 1;
        }
    }

    /// Return a reusable variable constrained to 0.
    pub fn zero_var(&mut self) -> usize {
        if let Some(v) = self.zero_var_cache {
            return v;
        }
        let v = self.new_var(F::ZERO);
        self.enforce_var_eq_const(v, F::ZERO);
        self.zero_var_cache = Some(v);
        v
    }
    pub fn into_instance(self) -> (SparseDr1csInstance<F>, Vec<F>) {
        if self.file_sink.is_some() {
            panic!("into_instance called on file-backed Dr1csBuilder; use into_file_backed_instance");
        }
        let inst = SparseDr1csInstance {
            nvars: self.assignment.len(),
            constraints: self.rows,
            a_terms: self.a_terms,
            b_terms: self.b_terms,
            c_terms: self.c_terms,
        };
        (inst, self.assignment)
    }

    pub fn into_file_backed_instance(self) -> Result<(FileBackedSparseDr1csInstance<F>, Vec<F>), String>
    where
        F: CanonicalSerialize,
    {
        let mut me = self;
        me.fb_flush()?;
        let sink = me
            .file_sink
            .take()
            .ok_or_else(|| "into_file_backed_instance called on in-memory Dr1csBuilder".to_string())?;
        let Dr1csFileSink::Append(sink) = sink else {
            return Err("into_file_backed_instance called on non-append file-backed builder".to_string());
        };
        let inst = sink.finish(me.assignment.len())?;
        Ok((inst, me.assignment))
    }

    pub fn into_count_result(self) -> Result<(Vec<F>, Dr1csFileCounts), String> {
        let mut me = self;
        me.fb_flush()?;
        if !matches!(me.file_sink, Some(Dr1csFileSink::Count)) {
            return Err("into_count_result called on non-count builder".to_string());
        }
        Ok((
            me.assignment,
            Dr1csFileCounts {
                rows: me.file_rows,
                a_terms: me.file_a_terms,
                b_terms: me.file_b_terms,
                c_terms: me.file_c_terms,
            },
        ))
    }

    #[cfg(unix)]
    pub fn into_range_result(self) -> Result<(Vec<F>, Dr1csRangeResult), String> {
        let mut me = self;
        me.fb_flush()?;
        let sink = me
            .file_sink
            .take()
            .ok_or_else(|| "into_range_result called on in-memory Dr1csBuilder".to_string())?;
        let Dr1csFileSink::Range(mut w) = sink else {
            return Err("into_range_result called on non-range builder".to_string());
        };
        let ckpts = w.take_ckpts();
        Ok((
            me.assignment,
            Dr1csRangeResult {
                counts: Dr1csFileCounts {
                    rows: me.file_rows,
                    a_terms: me.file_a_terms,
                    b_terms: me.file_b_terms,
                    c_terms: me.file_c_terms,
                },
                ckpts,
            },
        ))
    }

    /// Enter a profiling scope; returns the previous scope label.
    ///
    /// Use `profile_exit(prev)` to restore.
    #[inline]
    pub fn profile_enter(&mut self, label: &'static str) -> Option<&'static str> {
        if !self.profile_enabled {
            return None;
        }
        let prev = self.profile_current;
        self.profile_current = Some(label);
        prev
    }

    #[inline]
    pub fn profile_exit(&mut self, prev: Option<&'static str>) {
        if !self.profile_enabled {
            return;
        }
        self.profile_current = prev;
    }

    pub fn profile_report(&self, top_n: usize) -> String {
        if !self.profile_enabled {
            return "LF_PROFILE_DR1CS disabled".to_string();
        }
        let mut total_vars: u64 = 0;
        let mut total_constraints: u64 = 0;
        for v in self.profile.values() {
            total_vars += v.vars;
            total_constraints += v.constraints;
        }
        let mut rows: Vec<(&'static str, Dr1csProfileCounts)> = self
            .profile
            .iter()
            .map(|(&k, v)| (k, v.clone()))
            .collect();
        rows.sort_by_key(|(_k, v)| std::cmp::Reverse(v.constraints));

        let mut out = String::new();
        out.push_str(&format!(
            "== dR1CS profile (LF_PROFILE_DR1CS=1): total vars={} total constraints={} ==\n",
            total_vars, total_constraints
        ));
        out.push_str("top scopes by constraints:\n");
        for (i, (k, v)) in rows.into_iter().take(top_n).enumerate() {
            let pct = if total_constraints == 0 {
                0.0
            } else {
                (v.constraints as f64) * 100.0 / (total_constraints as f64)
            };
            out.push_str(&format!(
                "  {:>2}. {:<40}  constraints={:<12} vars={:<12} ({:>5.1}%)\n",
                i + 1,
                k,
                v.constraints,
                v.vars,
                pct
            ));
        }
        out
    }
}

/// A "ring element" represented as `d` prime-field variables (coefficients).
#[derive(Clone, Debug)]
pub struct RingVars {
    pub coeffs: Vec<usize>,
}

impl RingVars {
    pub fn new(coeffs: Vec<usize>) -> Self { Self { coeffs } }
    pub fn d(&self) -> usize { self.coeffs.len() }
}

fn ring_add<F: PrimeField>(b: &mut Dr1csBuilder<F>, x: &RingVars, y: &RingVars) -> RingVars {
    assert_eq!(x.d(), y.d());
    let mut out = Vec::with_capacity(x.d());
    for i in 0..x.d() {
        let val = b.assignment[x.coeffs[i]] + b.assignment[y.coeffs[i]];
        let v = b.new_var(val);
        b.add_constraint(
            vec![(F::ONE, x.coeffs[i]), (F::ONE, y.coeffs[i])],
            vec![(F::ONE, b.one())],
            vec![(F::ONE, v)],
        );
        out.push(v);
    }
    RingVars::new(out)
}

fn ring_scale<F: PrimeField>(b: &mut Dr1csBuilder<F>, x: &RingVars, s: usize) -> RingVars {
    let mut out = Vec::with_capacity(x.d());
    for i in 0..x.d() {
        let val = b.assignment[x.coeffs[i]] * b.assignment[s];
        let v = b.new_var(val);
        b.enforce_mul(x.coeffs[i], s, v);
        out.push(v);
    }
    RingVars::new(out)
}

fn scalar_sub_const<F: PrimeField>(b: &mut Dr1csBuilder<F>, r: usize, c: F) -> usize {
    let val = b.assignment[r] - c;
    let v = b.new_var(val);
    b.add_constraint(vec![(F::ONE, r), (-c, b.one())], vec![(F::ONE, b.one())], vec![(F::ONE, v)]);
    v
}

fn scalar_mul<F: PrimeField>(b: &mut Dr1csBuilder<F>, x: usize, y: usize) -> usize {
    let val = b.assignment[x] * b.assignment[y];
    let v = b.new_var(val);
    b.enforce_mul(x, y, v);
    v
}

/// Degree-3 Lagrange interpolation at `r` for points 0,1,2,3.
///
/// Returns scalar variables `(L0(r), L1(r), L2(r), L3(r))`.
fn lagrange_degree3<F: PrimeField>(b: &mut Dr1csBuilder<F>, r: usize) -> (usize, usize, usize, usize) {
    let inv2 = F::from(2u64).inverse().unwrap();
    let inv6 = F::from(6u64).inverse().unwrap();

    // t1 = r-1, t2 = r-2, t3 = r-3
    let t1 = scalar_sub_const(b, r, F::ONE);
    let t2 = scalar_sub_const(b, r, F::from(2u64));
    let t3 = scalar_sub_const(b, r, F::from(3u64));

    // L0 = -(t1*t2*t3)/6
    let p12 = scalar_mul(b, t1, t2);
    let p123 = scalar_mul(b, p12, t3);
    let l0_val = -(b.assignment[p123] * inv6);
    let l0 = b.new_var(l0_val);
    // l0 = (-inv6)*p123
    b.add_constraint(vec![(-inv6, p123)], vec![(F::ONE, b.one())], vec![(F::ONE, l0)]);

    // L1 = r*(r-2)*(r-3)/2
    let p = scalar_mul(b, r, t2);
    let p = scalar_mul(b, p, t3);
    let l1_val = b.assignment[p] * inv2;
    let l1 = b.new_var(l1_val);
    b.add_constraint(vec![(inv2, p)], vec![(F::ONE, b.one())], vec![(F::ONE, l1)]);

    // L2 = -r*(r-1)*(r-3)/2
    let p = scalar_mul(b, r, t1);
    let p = scalar_mul(b, p, t3);
    let l2_val = -(b.assignment[p] * inv2);
    let l2 = b.new_var(l2_val);
    b.add_constraint(vec![(-inv2, p)], vec![(F::ONE, b.one())], vec![(F::ONE, l2)]);

    // L3 = r*(r-1)*(r-2)/6
    let p = scalar_mul(b, r, t1);
    let p = scalar_mul(b, p, t2);
    let l3_val = b.assignment[p] * inv6;
    let l3 = b.new_var(l3_val);
    b.add_constraint(vec![(inv6, p)], vec![(F::ONE, b.one())], vec![(F::ONE, l3)]);

    (l0, l1, l2, l3)
}

/// Verify one degree-3 sumcheck over "ring elements" represented coefficient-wise.
///
/// Inputs:
/// - `claimed_sum`: current claim (ring vars)
/// - `msgs[i][t]`: per-round, per-evaluation point ring vars (t in 0..4)
/// - `rs[i]`: per-round verifier challenge scalar vars
///
/// Returns the final subclaim value (ring vars) after all rounds.
pub fn sumcheck_verify_degree3<F: PrimeField>(
    b: &mut Dr1csBuilder<F>,
    mut claimed_sum: RingVars,
    msgs: &[ [RingVars; 4] ],
    rs: &[usize],
) -> Result<RingVars, String> {
    if msgs.len() != rs.len() {
        return Err("sumcheck_verify_degree3: msgs/rs length mismatch".to_string());
    }
    for (round, (m, &r)) in msgs.iter().zip(rs.iter()).enumerate() {
        // Check g(0)+g(1) == claimed_sum (coefficient-wise).
        let g01 = ring_add(b, &m[0], &m[1]);
        // enforce g01 == claimed_sum
        for i in 0..claimed_sum.d() {
            b.enforce_lc_times_one_eq_const(vec![
                (F::ONE, g01.coeffs[i]),
                (-F::ONE, claimed_sum.coeffs[i]),
            ]);
        }

        // Update claim = g(r) by Lagrange interpolation.
        let (l0, l1, l2, l3) = lagrange_degree3::<F>(b, r);
        let t0 = ring_scale(b, &m[0], l0);
        let t1 = ring_scale(b, &m[1], l1);
        let t2 = ring_scale(b, &m[2], l2);
        let t3 = ring_scale(b, &m[3], l3);
        let s01 = ring_add(b, &t0, &t1);
        let s23 = ring_add(b, &t2, &t3);
        let new_claim = ring_add(b, &s01, &s23);
        claimed_sum = new_claim;

        let _ = round;
    }
    Ok(claimed_sum)
}

