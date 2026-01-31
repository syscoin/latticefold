//! File-backed sparse dR1CS storage.
//!
//! This module is the first step toward scaling to very large WE instances (hundreds of millions
//! to billions of constraints) without keeping all constraint/term pools in RAM.
//!
//! Current scope:
//! - An **append-only writer** for constraints + term pools.
//! - A simple file-backed instance wrapper.
//! - A streaming merge (share var0) for multiple file-backed parts.
#![cfg(feature = "std")]

use std::fs::{create_dir_all, File};
use std::io::{BufReader, BufWriter, Read as IoRead, Write as IoWrite};
use std::path::{Path, PathBuf};
use std::io::Seek;
use std::collections::HashMap;

use ark_ff::PrimeField;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize, Compress, SerializationError};

use crate::dpp_poseidon::SparseDr1csInstance;

#[derive(Clone, Debug)]
pub struct FileBackedLayout {
    pub dir: PathBuf,
    pub coeff_size: usize,
    pub nconstraints: u64,
    pub a_terms: u64,
    pub b_terms: u64,
    pub c_terms: u64,
}

#[derive(Clone, Debug)]
pub struct FileBackedSparseDr1csInstance<F: PrimeField> {
    pub nvars: usize,
    pub layout: FileBackedLayout,
    _pd: core::marker::PhantomData<F>,
}

fn write_u64(w: &mut impl IoWrite, x: u64) -> std::io::Result<()> {
    w.write_all(&x.to_le_bytes())
}
fn read_u64(r: &mut impl IoRead) -> std::io::Result<u64> {
    let mut buf = [0u8; 8];
    r.read_exact(&mut buf)?;
    Ok(u64::from_le_bytes(buf))
}

fn term_paths(dir: &Path, which: &str) -> (PathBuf, PathBuf) {
    (dir.join(format!("{which}_coeffs.bin")), dir.join(format!("{which}_idx.bin")))
}

fn constraints_path(dir: &Path) -> PathBuf {
    dir.join("constraints.bin")
}
fn meta_path(dir: &Path) -> PathBuf {
    dir.join("meta.txt")
}

fn serialize_fixed<F: CanonicalSerialize>(x: &F, out: &mut [u8]) -> Result<(), SerializationError> {
    // CanonicalSerialize writes to a std::io::Write. Avoid allocating a Vec per coefficient:
    // write directly into the fixed-size output buffer.
    struct SliceWriter<'a> {
        buf: &'a mut [u8],
        pos: usize,
    }
    impl<'a> IoWrite for SliceWriter<'a> {
        fn write(&mut self, data: &[u8]) -> std::io::Result<usize> {
            let rem = self.buf.len().saturating_sub(self.pos);
            if data.len() > rem {
                return Err(std::io::Error::new(std::io::ErrorKind::WriteZero, "SliceWriter overflow"));
            }
            self.buf[self.pos..self.pos + data.len()].copy_from_slice(data);
            self.pos += data.len();
            Ok(data.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    let mut w = SliceWriter { buf: out, pos: 0 };
    // Use uncompressed encoding for speed (bigger files, fewer bit-level operations).
    x.serialize_with_mode(&mut w, Compress::No)?;
    if w.pos != out.len() {
        return Err(SerializationError::InvalidData);
    }
    Ok(())
}

fn deserialize_fixed<F: CanonicalDeserialize>(buf: &[u8]) -> Result<F, SerializationError> {
    let mut r = BufReader::new(buf);
    // Skip validation for speed (inputs are self-produced file-backed artifacts).
    F::deserialize_with_mode(&mut r, Compress::No, ark_serialize::Validate::No)
}

#[derive(Debug)]
pub struct SparseDr1csFileWriter<F> {
    dir: PathBuf,
    coeff_size: usize,
    fc_a: BufWriter<File>,
    fi_a: BufWriter<File>,
    fc_b: BufWriter<File>,
    fi_b: BufWriter<File>,
    fc_c: BufWriter<File>,
    fi_c: BufWriter<File>,
    f_rows: BufWriter<File>,
    // Running counts (u64 for huge instances).
    nconstraints: u64,
    a_terms: u64,
    b_terms: u64,
    c_terms: u64,
    // Reusable coefficient buffer.
    coeff_buf: Vec<u8>,
    // Cache of canonical serialized coefficients keyed by a fast hash of BigInt limbs.
    // This avoids repeatedly running arkworks serialization on hot constant coefficients.
    coeff_cache: HashMap<u64, Vec<CoeffCacheEntry>>,
    _pd: core::marker::PhantomData<F>,
}

#[derive(Clone, Debug)]
struct CoeffCacheEntry {
    limbs: Vec<u64>,
    bytes: Vec<u8>,
}

#[inline]
fn hash_u64s(limbs: &[u64]) -> u64 {
    // Simple mixing hash (deterministic, fast). Collisions are handled by verifying limbs.
    let mut h: u64 = 0x9E37_79B9_7F4A_7C15;
    for &x in limbs {
        // splitmix64-like mix
        let mut z = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        h ^= z;
        h = h.rotate_left(13).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    }
    h
}

impl<F: PrimeField + CanonicalSerialize> SparseDr1csFileWriter<F> {
    pub fn create(dir: impl AsRef<Path>) -> Result<Self, String> {
        let dir = dir.as_ref();
        create_dir_all(dir).map_err(|e| format!("create_dir_all failed: {e}"))?;

        // Determine fixed coeff size.
        let mut tmp = Vec::new();
        F::ONE
            .serialize_with_mode(&mut tmp, Compress::No)
            .map_err(|e| format!("serialize ONE failed: {e}"))?;
        let coeff_size = tmp.len();
        if coeff_size == 0 {
            return Err("invalid coeff_size=0".to_string());
        }

        // Buffered IO is critical: file-backed Poseidon emits huge volumes of small-ish records.
        // Use a large BufWriter capacity by default, configurable via env var.
        #[inline]
        fn buf_bytes() -> usize {
            let mb: usize = std::env::var("LFP_FILE_BACKED_BUF_MB")
                .ok()
                .and_then(|s| s.parse().ok())
                // Default: large buffers to reduce syscall overhead on huge traces.
                .unwrap_or(256);
            // Keep buffers meaningfully large even if configured smaller.
            mb.saturating_mul(1024 * 1024).max(32 * 1024 * 1024)
        }
        let cap = buf_bytes();

        let (pa_c, pa_i) = term_paths(dir, "a");
        let (pb_c, pb_i) = term_paths(dir, "b");
        let (pc_c, pc_i) = term_paths(dir, "c");
        let fc_a = BufWriter::with_capacity(
            cap,
            File::create(pa_c).map_err(|e| format!("create a_coeffs failed: {e}"))?,
        );
        let fi_a = BufWriter::with_capacity(
            cap,
            File::create(pa_i).map_err(|e| format!("create a_idx failed: {e}"))?,
        );
        let fc_b = BufWriter::with_capacity(
            cap,
            File::create(pb_c).map_err(|e| format!("create b_coeffs failed: {e}"))?,
        );
        let fi_b = BufWriter::with_capacity(
            cap,
            File::create(pb_i).map_err(|e| format!("create b_idx failed: {e}"))?,
        );
        let fc_c = BufWriter::with_capacity(
            cap,
            File::create(pc_c).map_err(|e| format!("create c_coeffs failed: {e}"))?,
        );
        let fi_c = BufWriter::with_capacity(
            cap,
            File::create(pc_i).map_err(|e| format!("create c_idx failed: {e}"))?,
        );
        let f_rows = BufWriter::with_capacity(
            cap,
            File::create(constraints_path(dir)).map_err(|e| format!("create constraints failed: {e}"))?,
        );
        Ok(Self {
            dir: dir.to_path_buf(),
            coeff_size,
            fc_a,
            fi_a,
            fc_b,
            fi_b,
            fc_c,
            fi_c,
            f_rows,
            nconstraints: 0,
            a_terms: 0,
            b_terms: 0,
            c_terms: 0,
            coeff_buf: vec![0u8; coeff_size],
            coeff_cache: HashMap::new(),
            _pd: core::marker::PhantomData,
        })
    }

    #[inline]
    pub fn coeff_size(&self) -> usize {
        self.coeff_size
    }

    #[inline]
    pub fn counts(&self) -> (u64, u64, u64, u64) {
        (self.nconstraints, self.a_terms, self.b_terms, self.c_terms)
    }

    #[inline]
    pub fn push_a_term(&mut self, coef: &F, idx: u64) -> Result<(), String> {
        self.write_coeff_cached(coef)?;
        self.fc_a.write_all(&self.coeff_buf).map_err(|e| e.to_string())?;
        write_u64(&mut self.fi_a, idx).map_err(|e| e.to_string())?;
        self.a_terms += 1;
        Ok(())
    }
    #[inline]
    pub fn push_a_term_raw(&mut self, coef_bytes: &[u8], idx: u64) -> Result<(), String> {
        debug_assert_eq!(coef_bytes.len(), self.coeff_size);
        self.fc_a.write_all(coef_bytes).map_err(|e| e.to_string())?;
        write_u64(&mut self.fi_a, idx).map_err(|e| e.to_string())?;
        self.a_terms += 1;
        Ok(())
    }
    #[inline]
    pub fn push_b_term(&mut self, coef: &F, idx: u64) -> Result<(), String> {
        self.write_coeff_cached(coef)?;
        self.fc_b.write_all(&self.coeff_buf).map_err(|e| e.to_string())?;
        write_u64(&mut self.fi_b, idx).map_err(|e| e.to_string())?;
        self.b_terms += 1;
        Ok(())
    }
    #[inline]
    pub fn push_b_term_raw(&mut self, coef_bytes: &[u8], idx: u64) -> Result<(), String> {
        debug_assert_eq!(coef_bytes.len(), self.coeff_size);
        self.fc_b.write_all(coef_bytes).map_err(|e| e.to_string())?;
        write_u64(&mut self.fi_b, idx).map_err(|e| e.to_string())?;
        self.b_terms += 1;
        Ok(())
    }
    #[inline]
    pub fn push_c_term(&mut self, coef: &F, idx: u64) -> Result<(), String> {
        self.write_coeff_cached(coef)?;
        self.fc_c.write_all(&self.coeff_buf).map_err(|e| e.to_string())?;
        write_u64(&mut self.fi_c, idx).map_err(|e| e.to_string())?;
        self.c_terms += 1;
        Ok(())
    }

    #[inline]
    fn write_coeff_cached(&mut self, coef: &F) -> Result<(), String> {
        let big = coef.into_bigint();
        let limbs = big.as_ref();
        let key = hash_u64s(limbs);
        if let Some(bucket) = self.coeff_cache.get(&key) {
            for ent in bucket {
                if ent.limbs.as_slice() == limbs {
                    self.coeff_buf.copy_from_slice(&ent.bytes);
                    return Ok(());
                }
            }
        }

        // Cache miss: serialize once.
        serialize_fixed(coef, &mut self.coeff_buf).map_err(|e| format!("serialize coeff failed: {e}"))?;
        let bytes = self.coeff_buf.clone();
        let ent = CoeffCacheEntry {
            limbs: limbs.to_vec(),
            bytes,
        };
        self.coeff_cache.entry(key).or_default().push(ent);
        Ok(())
    }
    #[inline]
    pub fn push_c_term_raw(&mut self, coef_bytes: &[u8], idx: u64) -> Result<(), String> {
        debug_assert_eq!(coef_bytes.len(), self.coeff_size);
        self.fc_c.write_all(coef_bytes).map_err(|e| e.to_string())?;
        write_u64(&mut self.fi_c, idx).map_err(|e| e.to_string())?;
        self.c_terms += 1;
        Ok(())
    }

    #[inline]
    pub fn push_constraint_row(
        &mut self,
        a0: u64,
        a1: u64,
        b0: u64,
        b1: u64,
        c0: u64,
        c1: u64,
    ) -> Result<(), String> {
        write_u64(&mut self.f_rows, a0).map_err(|e| e.to_string())?;
        write_u64(&mut self.f_rows, a1).map_err(|e| e.to_string())?;
        write_u64(&mut self.f_rows, b0).map_err(|e| e.to_string())?;
        write_u64(&mut self.f_rows, b1).map_err(|e| e.to_string())?;
        write_u64(&mut self.f_rows, c0).map_err(|e| e.to_string())?;
        write_u64(&mut self.f_rows, c1).map_err(|e| e.to_string())?;
        self.nconstraints += 1;
        Ok(())
    }

    pub fn finish(mut self, nvars: usize) -> Result<FileBackedSparseDr1csInstance<F>, String> {
        // Flush writers.
        self.fc_a.flush().map_err(|e| e.to_string())?;
        self.fi_a.flush().map_err(|e| e.to_string())?;
        self.fc_b.flush().map_err(|e| e.to_string())?;
        self.fi_b.flush().map_err(|e| e.to_string())?;
        self.fc_c.flush().map_err(|e| e.to_string())?;
        self.fi_c.flush().map_err(|e| e.to_string())?;
        self.f_rows.flush().map_err(|e| e.to_string())?;

        // Write meta (human-readable).
        {
            let mut f =
                BufWriter::new(File::create(meta_path(&self.dir)).map_err(|e| format!("create meta failed: {e}"))?);
            writeln!(f, "nvars={}", nvars).ok();
            writeln!(f, "constraints={}", self.nconstraints).ok();
            writeln!(f, "a_terms={}", self.a_terms).ok();
            writeln!(f, "b_terms={}", self.b_terms).ok();
            writeln!(f, "c_terms={}", self.c_terms).ok();
            writeln!(f, "coeff_size={}", self.coeff_size).ok();
        }

        let layout = FileBackedLayout {
            dir: self.dir.clone(),
            coeff_size: self.coeff_size,
            nconstraints: self.nconstraints,
            a_terms: self.a_terms,
            b_terms: self.b_terms,
            c_terms: self.c_terms,
        };
        Ok(FileBackedSparseDr1csInstance {
            nvars,
            layout,
            _pd: core::marker::PhantomData,
        })
    }
}

/// Dump an in-memory sparse instance to disk in a file-backed format.
///
/// This does **not** reduce peak RAM (the instance already exists), but is a useful stepping stone
/// for validating the file-backed format and checker before switching generators to stream directly.
pub fn dump_sparse_to_dir<F: PrimeField + CanonicalSerialize>(
    inst: &SparseDr1csInstance<F>,
    dir: impl AsRef<Path>,
) -> Result<FileBackedLayout, String> {
    let dir = dir.as_ref();
    create_dir_all(dir).map_err(|e| format!("create_dir_all failed: {e}"))?;

    // Determine fixed coeff size.
    let mut tmp = Vec::new();
    F::ONE
        .serialize_with_mode(&mut tmp, Compress::No)
        .map_err(|e| format!("serialize ONE failed: {e}"))?;
    let coeff_size = tmp.len();
    if coeff_size == 0 {
        return Err("invalid coeff_size=0".to_string());
    }

    // Constraints: for each row store (a_start,a_end,b_start,b_end,c_start,c_end) as u64.
    {
        let mut f = BufWriter::new(File::create(constraints_path(dir)).map_err(|e| format!("create constraints failed: {e}"))?);
        for row in &inst.constraints {
            write_u64(&mut f, row.a.start as u64).map_err(|e| e.to_string())?;
            write_u64(&mut f, row.a.end as u64).map_err(|e| e.to_string())?;
            write_u64(&mut f, row.b.start as u64).map_err(|e| e.to_string())?;
            write_u64(&mut f, row.b.end as u64).map_err(|e| e.to_string())?;
            write_u64(&mut f, row.c.start as u64).map_err(|e| e.to_string())?;
            write_u64(&mut f, row.c.end as u64).map_err(|e| e.to_string())?;
        }
    }

    // Term pools: write coeffs as fixed-size blobs, and indices as u64.
    for (which, terms) in [
        ("a", &inst.a_terms),
        ("b", &inst.b_terms),
        ("c", &inst.c_terms),
    ] {
        let (p_coeffs, p_idx) = term_paths(dir, which);
        let mut fc = BufWriter::new(File::create(p_coeffs).map_err(|e| format!("create {which}_coeffs failed: {e}"))?);
        let mut fi = BufWriter::new(File::create(p_idx).map_err(|e| format!("create {which}_idx failed: {e}"))?);
        let mut buf = vec![0u8; coeff_size];
        for (coef, idx) in terms.iter() {
            serialize_fixed(coef, &mut buf).map_err(|e| format!("serialize coeff failed: {e}"))?;
            fc.write_all(&buf).map_err(|e| e.to_string())?;
            write_u64(&mut fi, *idx as u64).map_err(|e| e.to_string())?;
        }
    }

    // Write meta (human-readable) last.
    {
        let mut f = BufWriter::new(File::create(meta_path(dir)).map_err(|e| format!("create meta failed: {e}"))?);
        writeln!(f, "nvars={}", inst.nvars).ok();
        writeln!(f, "constraints={}", inst.constraints.len()).ok();
        writeln!(f, "a_terms={}", inst.a_terms.len()).ok();
        writeln!(f, "b_terms={}", inst.b_terms.len()).ok();
        writeln!(f, "c_terms={}", inst.c_terms.len()).ok();
        writeln!(f, "coeff_size={}", coeff_size).ok();
    }

    Ok(FileBackedLayout {
        dir: dir.to_path_buf(),
        coeff_size,
        nconstraints: inst.constraints.len() as u64,
        a_terms: inst.a_terms.len() as u64,
        b_terms: inst.b_terms.len() as u64,
        c_terms: inst.c_terms.len() as u64,
    })
}

impl<F: PrimeField + CanonicalDeserialize + CanonicalSerialize> FileBackedSparseDr1csInstance<F> {
    pub fn open(layout: FileBackedLayout) -> Result<Self, String> {
        // Read nvars from meta (best-effort parsing).
        let meta = std::fs::read_to_string(meta_path(&layout.dir))
            .map_err(|e| format!("read meta failed: {e}"))?;
        let mut nvars: Option<usize> = None;
        let mut nconstraints: Option<u64> = None;
        let mut a_terms: Option<u64> = None;
        let mut b_terms: Option<u64> = None;
        let mut c_terms: Option<u64> = None;
        let mut coeff_size: Option<usize> = None;
        for line in meta.lines() {
            if let Some(rest) = line.strip_prefix("nvars=") {
                nvars = rest.trim().parse::<usize>().ok();
            } else if let Some(rest) = line.strip_prefix("constraints=") {
                nconstraints = rest.trim().parse::<u64>().ok();
            } else if let Some(rest) = line.strip_prefix("a_terms=") {
                a_terms = rest.trim().parse::<u64>().ok();
            } else if let Some(rest) = line.strip_prefix("b_terms=") {
                b_terms = rest.trim().parse::<u64>().ok();
            } else if let Some(rest) = line.strip_prefix("c_terms=") {
                c_terms = rest.trim().parse::<u64>().ok();
            } else if let Some(rest) = line.strip_prefix("coeff_size=") {
                coeff_size = rest.trim().parse::<usize>().ok();
            }
        }
        let nvars = nvars.ok_or("meta missing nvars")?;
        let layout = FileBackedLayout {
            dir: layout.dir,
            coeff_size: coeff_size.unwrap_or(layout.coeff_size),
            nconstraints: nconstraints.unwrap_or(layout.nconstraints),
            a_terms: a_terms.unwrap_or(layout.a_terms),
            b_terms: b_terms.unwrap_or(layout.b_terms),
            c_terms: c_terms.unwrap_or(layout.c_terms),
        };
        Ok(Self { nvars, layout, _pd: core::marker::PhantomData })
    }

    fn read_term(
        &self,
        which: &str,
        term_idx: u64,
    ) -> Result<(F, usize), String> {
        let (p_coeffs, p_idx) = term_paths(&self.layout.dir, which);
        let mut fc = File::open(p_coeffs).map_err(|e| format!("open {which}_coeffs failed: {e}"))?;
        let mut fi = File::open(p_idx).map_err(|e| format!("open {which}_idx failed: {e}"))?;

        let off_c = (term_idx as u64)
            .checked_mul(self.layout.coeff_size as u64)
            .ok_or("term coeff offset overflow")?;
        fc.seek(std::io::SeekFrom::Start(off_c))
            .map_err(|e| format!("seek coeff failed: {e}"))?;
        fi.seek(std::io::SeekFrom::Start(term_idx * 8))
            .map_err(|e| format!("seek idx failed: {e}"))?;

        let mut buf = vec![0u8; self.layout.coeff_size];
        fc.read_exact(&mut buf).map_err(|e| format!("read coeff failed: {e}"))?;
        let idx = read_u64(&mut fi).map_err(|e| format!("read idx failed: {e}"))? as usize;
        let coef = deserialize_fixed::<F>(&buf).map_err(|e| format!("deserialize coeff failed: {e}"))?;
        Ok((coef, idx))
    }

    /// Prototype checker: replays all constraints by reading term pools from disk.
    ///
    /// This is **slow** and intended only for correctness validation of the on-disk format.
    pub fn check(&self, assignment: &[F]) -> Result<(), String> {
        if assignment.len() != self.nvars {
            return Err(format!(
                "assignment length mismatch: expected {}, got {}",
                self.nvars,
                assignment.len()
            ));
        }
        // Iterate constraints in order.
        let mut f = BufReader::new(
            File::open(constraints_path(&self.layout.dir))
                .map_err(|e| format!("open constraints failed: {e}"))?,
        );
        let mut i: u64 = 0;
        loop {
            // Attempt to read next row; EOF => done.
            let a0 = match read_u64(&mut f) {
                Ok(v) => v,
                Err(_) => break,
            };
            let a1 = read_u64(&mut f).map_err(|e| e.to_string())?;
            let b0 = read_u64(&mut f).map_err(|e| e.to_string())?;
            let b1 = read_u64(&mut f).map_err(|e| e.to_string())?;
            let c0 = read_u64(&mut f).map_err(|e| e.to_string())?;
            let c1 = read_u64(&mut f).map_err(|e| e.to_string())?;

            let eval_lc = |which: &str, start: u64, end: u64| -> Result<F, String> {
                let mut acc = F::ZERO;
                for t in start..end {
                    let (coef, idx) = self.read_term(which, t)?;
                    acc += coef * assignment[idx];
                }
                Ok(acc)
            };
            let a = eval_lc("a", a0, a1)?;
            let b = eval_lc("b", b0, b1)?;
            let c = eval_lc("c", c0, c1)?;
            if a * b != c {
                return Err(format!("constraint {i} failed"));
            }
            i += 1;
        }
        Ok(())
    }
}

/// Merge multiple file-backed instances into one, sharing variable 0 as the constant-1 slot.
///
/// This is the file-backed analog of `merge_sparse_dr1cs_share_one`:
/// - var0 is shared across all parts
/// - all other variables are appended
/// - constraints/terms are concatenated with term-range and var-index offsets applied
pub fn merge_file_backed_sparse_dr1cs_share_one<F: PrimeField + CanonicalSerialize + CanonicalDeserialize + Copy>(
    parts: Vec<(FileBackedSparseDr1csInstance<F>, Vec<F>)>,
    out_dir: impl AsRef<Path>,
    extra_eqs: &[(usize, usize)],
) -> Result<(FileBackedSparseDr1csInstance<F>, Vec<F>), String> {
    if parts.is_empty() {
        return Err("merge_file_backed_sparse_dr1cs_share_one: empty parts".to_string());
    }
    // Validate and compute assignment.
    let mut new_assignment: Vec<F> = Vec::new();
    new_assignment.push(F::ONE);
    let mut tail_len: usize = 0;
    for (_inst, asg) in parts.iter() {
        if asg.is_empty() || asg[0] != F::ONE {
            return Err("merge_file_backed_sparse_dr1cs_share_one: each part must have assignment[0]=1".to_string());
        }
        new_assignment.extend_from_slice(&asg[1..]);
        tail_len += asg.len().saturating_sub(1);
    }
    let _ = tail_len;

    // IMPORTANT: merge is dominated by streaming IO + index remapping.
    // Large buffers significantly reduce syscall overhead during merges.
    const MERGE_BUF_BYTES: usize = 256 * 1024 * 1024;

    // If we have multiple Rayon threads available, do a parallel merge:
    // 1) For each part, in parallel, create remapped chunk files (idx/constraints remapped).
    // 2) Concatenate chunks in order into the final output files once.
    //
    // This avoids a single-threaded long-running remap loop on huge instances.
    let n_threads = rayon::current_num_threads().max(1);
    if n_threads > 1 && parts.len() > 1 {
        use rayon::prelude::*;

        let out_dir = out_dir.as_ref().to_path_buf();
        let _ = std::fs::remove_dir_all(&out_dir);
        create_dir_all(&out_dir).map_err(|e| format!("create_dir_all failed: {e}"))?;

        // Validate layouts are compatible and precompute offsets.
        let coeff_size = parts[0].0.layout.coeff_size;
        for (inst, _asg) in parts.iter() {
            if inst.layout.coeff_size != coeff_size {
                return Err("merge_file_backed_sparse_dr1cs_share_one: coeff_size mismatch across parts".to_string());
            }
        }

        // Prefix sums for output term offsets per part.
        let mut a_off: Vec<u64> = Vec::with_capacity(parts.len());
        let mut b_off: Vec<u64> = Vec::with_capacity(parts.len());
        let mut c_off: Vec<u64> = Vec::with_capacity(parts.len());
        let mut cur_a: u64 = 0;
        let mut cur_b: u64 = 0;
        let mut cur_c: u64 = 0;
        for (inst, _asg) in parts.iter() {
            a_off.push(cur_a);
            b_off.push(cur_b);
            c_off.push(cur_c);
            cur_a = cur_a.saturating_add(inst.layout.a_terms);
            cur_b = cur_b.saturating_add(inst.layout.b_terms);
            cur_c = cur_c.saturating_add(inst.layout.c_terms);
        }

        // Prefix sums for variable tail offsets per part (excluding shared var0).
        let mut var_tail_off: Vec<u64> = Vec::with_capacity(parts.len());
        let mut cur_v: u64 = 0;
        for (_inst, asg) in parts.iter() {
            var_tail_off.push(cur_v);
            cur_v = cur_v
                .checked_add(asg.len().saturating_sub(1) as u64)
                .ok_or("var_tail_off overflow")?;
        }

        // Helper: map local var index to global.
        #[inline]
        fn map_var(idx: u64, var_tail_off: u64) -> u64 {
            if idx == 0 { 0 } else { idx + var_tail_off }
        }

        // Temp dir for chunks.
        let tmp_dir = out_dir.join(".tmp_merge_parts");
        let _ = std::fs::remove_dir_all(&tmp_dir);
        create_dir_all(&tmp_dir).map_err(|e| format!("create tmp_merge_parts failed: {e}"))?;

        // Build remapped chunks in parallel.
        (0..parts.len())
            .into_par_iter()
            .try_for_each(|pi| -> Result<(), String> {
                let (inst, _asg) = &parts[pi];
                let v_off = var_tail_off[pi];

                // Remap idx files; coeff files can be copied verbatim.
                for which in ["a", "b", "c"] {
                    let (p_coeffs, p_idx) = term_paths(&inst.layout.dir, which);
                    let coeff_out = tmp_dir.join(format!("{pi:04}_{which}_coeffs.bin"));
                    let idx_out = tmp_dir.join(format!("{pi:04}_{which}_idx.bin"));

                    // Copy coeff bytes verbatim.
                    {
                        let mut r = BufReader::with_capacity(
                            MERGE_BUF_BYTES,
                            File::open(p_coeffs).map_err(|e| format!("open {which}_coeffs failed: {e}"))?,
                        );
                        let mut w = BufWriter::with_capacity(
                            MERGE_BUF_BYTES,
                            File::create(&coeff_out).map_err(|e| format!("create {which}_coeffs chunk failed: {e}"))?,
                        );
                        std::io::copy(&mut r, &mut w).map_err(|e| format!("copy {which}_coeffs failed: {e}"))?;
                        w.flush().ok();
                    }

                    // Remap indices.
                    {
                        let mut r = BufReader::with_capacity(
                            MERGE_BUF_BYTES,
                            File::open(p_idx).map_err(|e| format!("open {which}_idx failed: {e}"))?,
                        );
                        let mut w = BufWriter::with_capacity(
                            MERGE_BUF_BYTES,
                            File::create(&idx_out).map_err(|e| format!("create {which}_idx chunk failed: {e}"))?,
                        );
                        let n_terms = match which {
                            "a" => inst.layout.a_terms,
                            "b" => inst.layout.b_terms,
                            "c" => inst.layout.c_terms,
                            _ => unreachable!(),
                        };
                        for _ in 0..n_terms {
                            let idx = read_u64(&mut r).map_err(|e| format!("read {which}_idx failed: {e}"))?;
                            let mapped = map_var(idx, v_off);
                            write_u64(&mut w, mapped).map_err(|e| format!("write {which}_idx failed: {e}"))?;
                        }
                        w.flush().ok();
                    }
                }

                // Remap constraints (term ranges are offset by prefix sums, indices already global in output pools).
                {
                    let in_path = constraints_path(&inst.layout.dir);
                    let out_path = tmp_dir.join(format!("{pi:04}_constraints.bin"));
                    let mut r = BufReader::with_capacity(
                        MERGE_BUF_BYTES,
                        File::open(in_path).map_err(|e| format!("open constraints failed: {e}"))?,
                    );
                    let mut w = BufWriter::with_capacity(
                        MERGE_BUF_BYTES,
                        File::create(&out_path).map_err(|e| format!("create constraints chunk failed: {e}"))?,
                    );
                    let base_a = a_off[pi];
                    let base_b = b_off[pi];
                    let base_c = c_off[pi];
                    for _ in 0..inst.layout.nconstraints {
                        let a0 = read_u64(&mut r).map_err(|e| e.to_string())?;
                        let a1 = read_u64(&mut r).map_err(|e| e.to_string())?;
                        let b0 = read_u64(&mut r).map_err(|e| e.to_string())?;
                        let b1 = read_u64(&mut r).map_err(|e| e.to_string())?;
                        let c0 = read_u64(&mut r).map_err(|e| e.to_string())?;
                        let c1 = read_u64(&mut r).map_err(|e| e.to_string())?;
                        write_u64(&mut w, base_a + a0).map_err(|e| e.to_string())?;
                        write_u64(&mut w, base_a + a1).map_err(|e| e.to_string())?;
                        write_u64(&mut w, base_b + b0).map_err(|e| e.to_string())?;
                        write_u64(&mut w, base_b + b1).map_err(|e| e.to_string())?;
                        write_u64(&mut w, base_c + c0).map_err(|e| e.to_string())?;
                        write_u64(&mut w, base_c + c1).map_err(|e| e.to_string())?;
                    }
                    w.flush().ok();
                }

                Ok(())
            })?;

        // Concatenate chunks deterministically into final files.
        let mut out_fc_a = BufWriter::with_capacity(MERGE_BUF_BYTES, File::create(out_dir.join("a_coeffs.bin")).map_err(|e| e.to_string())?);
        let mut out_fi_a = BufWriter::with_capacity(MERGE_BUF_BYTES, File::create(out_dir.join("a_idx.bin")).map_err(|e| e.to_string())?);
        let mut out_fc_b = BufWriter::with_capacity(MERGE_BUF_BYTES, File::create(out_dir.join("b_coeffs.bin")).map_err(|e| e.to_string())?);
        let mut out_fi_b = BufWriter::with_capacity(MERGE_BUF_BYTES, File::create(out_dir.join("b_idx.bin")).map_err(|e| e.to_string())?);
        let mut out_fc_c = BufWriter::with_capacity(MERGE_BUF_BYTES, File::create(out_dir.join("c_coeffs.bin")).map_err(|e| e.to_string())?);
        let mut out_fi_c = BufWriter::with_capacity(MERGE_BUF_BYTES, File::create(out_dir.join("c_idx.bin")).map_err(|e| e.to_string())?);
        let mut out_rows = BufWriter::with_capacity(MERGE_BUF_BYTES, File::create(out_dir.join("constraints.bin")).map_err(|e| e.to_string())?);

        for pi in 0..parts.len() {
            for which in ["a", "b", "c"] {
                let coeff_in = tmp_dir.join(format!("{pi:04}_{which}_coeffs.bin"));
                let idx_in = tmp_dir.join(format!("{pi:04}_{which}_idx.bin"));
                let mut rc = BufReader::with_capacity(MERGE_BUF_BYTES, File::open(coeff_in).map_err(|e| e.to_string())?);
                let mut ri = BufReader::with_capacity(MERGE_BUF_BYTES, File::open(idx_in).map_err(|e| e.to_string())?);
                match which {
                    "a" => {
                        std::io::copy(&mut rc, &mut out_fc_a).map_err(|e| e.to_string())?;
                        std::io::copy(&mut ri, &mut out_fi_a).map_err(|e| e.to_string())?;
                    }
                    "b" => {
                        std::io::copy(&mut rc, &mut out_fc_b).map_err(|e| e.to_string())?;
                        std::io::copy(&mut ri, &mut out_fi_b).map_err(|e| e.to_string())?;
                    }
                    "c" => {
                        std::io::copy(&mut rc, &mut out_fc_c).map_err(|e| e.to_string())?;
                        std::io::copy(&mut ri, &mut out_fi_c).map_err(|e| e.to_string())?;
                    }
                    _ => unreachable!(),
                }
            }
            let cons_in = tmp_dir.join(format!("{pi:04}_constraints.bin"));
            let mut rr = BufReader::with_capacity(MERGE_BUF_BYTES, File::open(cons_in).map_err(|e| e.to_string())?);
            std::io::copy(&mut rr, &mut out_rows).map_err(|e| e.to_string())?;
        }

        // Append extra equality constraints by using the normal writer (small).
        // Re-open append handles and write directly in the same binary formats.
        if !extra_eqs.is_empty() {
            // Serialize constants once.
            let mut one_bytes = vec![0u8; coeff_size];
            let mut neg_one_bytes = vec![0u8; coeff_size];
            let mut zero_bytes = vec![0u8; coeff_size];
            serialize_fixed::<F>(&F::ONE, &mut one_bytes).map_err(|e| e.to_string())?;
            serialize_fixed::<F>(&(-F::ONE), &mut neg_one_bytes).map_err(|e| e.to_string())?;
            serialize_fixed::<F>(&F::ZERO, &mut zero_bytes).map_err(|e| e.to_string())?;

            // Current term counts (u64) after concatenation.
            let mut a_terms = cur_a;
            let mut b_terms = cur_b;
            let mut c_terms = cur_c;
            let mut nconstraints: u64 = parts.iter().map(|(i, _)| i.layout.nconstraints).sum::<u64>();

            for &(x, y) in extra_eqs {
                // A terms: +1*x, -1*y
                out_fc_a.write_all(&one_bytes).map_err(|e| e.to_string())?;
                write_u64(&mut out_fi_a, x as u64).map_err(|e| e.to_string())?;
                a_terms += 1;
                out_fc_a.write_all(&neg_one_bytes).map_err(|e| e.to_string())?;
                write_u64(&mut out_fi_a, y as u64).map_err(|e| e.to_string())?;
                a_terms += 1;

                // B terms: +1*var0
                out_fc_b.write_all(&one_bytes).map_err(|e| e.to_string())?;
                write_u64(&mut out_fi_b, 0).map_err(|e| e.to_string())?;
                b_terms += 1;

                // C terms: 0*var0
                out_fc_c.write_all(&zero_bytes).map_err(|e| e.to_string())?;
                write_u64(&mut out_fi_c, 0).map_err(|e| e.to_string())?;
                c_terms += 1;

                // Constraint row points to the new tail terms.
                let a0 = a_terms - 2;
                let a1 = a_terms;
                let b0 = b_terms - 1;
                let b1 = b_terms;
                let c0 = c_terms - 1;
                let c1 = c_terms;
                write_u64(&mut out_rows, a0).map_err(|e| e.to_string())?;
                write_u64(&mut out_rows, a1).map_err(|e| e.to_string())?;
                write_u64(&mut out_rows, b0).map_err(|e| e.to_string())?;
                write_u64(&mut out_rows, b1).map_err(|e| e.to_string())?;
                write_u64(&mut out_rows, c0).map_err(|e| e.to_string())?;
                write_u64(&mut out_rows, c1).map_err(|e| e.to_string())?;
                nconstraints += 1;
            }

            // Flush output writers before writing meta.
            out_fc_a.flush().ok();
            out_fi_a.flush().ok();
            out_fc_b.flush().ok();
            out_fi_b.flush().ok();
            out_fc_c.flush().ok();
            out_fi_c.flush().ok();
            out_rows.flush().ok();

            // Update totals.
            cur_a = a_terms;
            cur_b = b_terms;
            cur_c = c_terms;
            // nconstraints updated above; write meta below using updated totals.
            {
                let mut f = BufWriter::new(File::create(meta_path(&out_dir)).map_err(|e| format!("create meta failed: {e}"))?);
                writeln!(f, "nvars={}", new_assignment.len()).ok();
                writeln!(f, "constraints={}", nconstraints).ok();
                writeln!(f, "a_terms={}", cur_a).ok();
                writeln!(f, "b_terms={}", cur_b).ok();
                writeln!(f, "c_terms={}", cur_c).ok();
                writeln!(f, "coeff_size={}", coeff_size).ok();
            }

            let layout = FileBackedLayout {
                dir: out_dir.clone(),
                coeff_size,
                nconstraints,
                a_terms: cur_a,
                b_terms: cur_b,
                c_terms: cur_c,
            };
            let _ = std::fs::remove_dir_all(&tmp_dir);
            return Ok((FileBackedSparseDr1csInstance { nvars: new_assignment.len(), layout, _pd: core::marker::PhantomData }, new_assignment));
        }

        // Flush output writers before writing meta.
        out_fc_a.flush().ok();
        out_fi_a.flush().ok();
        out_fc_b.flush().ok();
        out_fi_b.flush().ok();
        out_fc_c.flush().ok();
        out_fi_c.flush().ok();
        out_rows.flush().ok();

        let nconstraints: u64 = parts.iter().map(|(i, _)| i.layout.nconstraints).sum::<u64>();
        {
            let mut f = BufWriter::new(File::create(meta_path(&out_dir)).map_err(|e| format!("create meta failed: {e}"))?);
            writeln!(f, "nvars={}", new_assignment.len()).ok();
            writeln!(f, "constraints={}", nconstraints).ok();
            writeln!(f, "a_terms={}", cur_a).ok();
            writeln!(f, "b_terms={}", cur_b).ok();
            writeln!(f, "c_terms={}", cur_c).ok();
            writeln!(f, "coeff_size={}", coeff_size).ok();
        }

        let layout = FileBackedLayout {
            dir: out_dir.clone(),
            coeff_size,
            nconstraints,
            a_terms: cur_a,
            b_terms: cur_b,
            c_terms: cur_c,
        };
        let _ = std::fs::remove_dir_all(&tmp_dir);
        return Ok((FileBackedSparseDr1csInstance { nvars: new_assignment.len(), layout, _pd: core::marker::PhantomData }, new_assignment));
    }

    let mut w = SparseDr1csFileWriter::<F>::create(out_dir)?;

    // Running offsets in the *output* term pools.
    let mut out_a: u64 = 0;
    let mut out_b: u64 = 0;
    let mut out_c: u64 = 0;

    // Running offsets in the *output* var assignment tail (excluding shared var0).
    let mut var_tail_off: u64 = 0;

    for (inst, asg) in parts.into_iter() {
        let local_nvars = asg.len() as u64;
        let local_tail = local_nvars.saturating_sub(1);

        // Helper: map local var index to global.
        #[inline]
        fn map_var(idx: u64, var_tail_off: u64) -> u64 {
            if idx == 0 { 0 } else { idx + var_tail_off }
        }

        // Copy term pools (a/b/c), remapping indices.
        for (which, n_terms, out_terms_off) in [
            ("a", inst.layout.a_terms, &mut out_a),
            ("b", inst.layout.b_terms, &mut out_b),
            ("c", inst.layout.c_terms, &mut out_c),
        ] {
            let (p_coeffs, p_idx) = term_paths(&inst.layout.dir, which);
            let mut fc = BufReader::with_capacity(
                MERGE_BUF_BYTES,
                File::open(p_coeffs).map_err(|e| format!("open {which}_coeffs failed: {e}"))?,
            );
            let mut fi = BufReader::with_capacity(
                MERGE_BUF_BYTES,
                File::open(p_idx).map_err(|e| format!("open {which}_idx failed: {e}"))?,
            );
            let mut buf = vec![0u8; inst.layout.coeff_size];
            for _ in 0..n_terms {
                fc.read_exact(&mut buf).map_err(|e| format!("read {which}_coeffs failed: {e}"))?;
                let idx = read_u64(&mut fi).map_err(|e| format!("read {which}_idx failed: {e}"))?;
                let mapped = map_var(idx, var_tail_off);
                match which {
                    "a" => w.push_a_term_raw(&buf, mapped)?,
                    "b" => w.push_b_term_raw(&buf, mapped)?,
                    "c" => w.push_c_term_raw(&buf, mapped)?,
                    _ => unreachable!(),
                }
            }
            *out_terms_off += n_terms;
        }

        // Copy constraints, offsetting term ranges by out_a/out_b/out_c *before* this part.
        let mut fr = BufReader::with_capacity(
            MERGE_BUF_BYTES,
            File::open(constraints_path(&inst.layout.dir))
                .map_err(|e| format!("open constraints failed: {e}"))?,
        );
        let base_a = out_a - inst.layout.a_terms;
        let base_b = out_b - inst.layout.b_terms;
        let base_c = out_c - inst.layout.c_terms;
        for _ in 0..inst.layout.nconstraints {
            let a0 = read_u64(&mut fr).map_err(|e| e.to_string())?;
            let a1 = read_u64(&mut fr).map_err(|e| e.to_string())?;
            let b0 = read_u64(&mut fr).map_err(|e| e.to_string())?;
            let b1 = read_u64(&mut fr).map_err(|e| e.to_string())?;
            let c0 = read_u64(&mut fr).map_err(|e| e.to_string())?;
            let c1 = read_u64(&mut fr).map_err(|e| e.to_string())?;
            w.push_constraint_row(
                base_a + a0,
                base_a + a1,
                base_b + b0,
                base_b + b1,
                base_c + c0,
                base_c + c1,
            )?;
        }

        // Advance var tail offset.
        var_tail_off = var_tail_off
            .checked_add(local_tail)
            .ok_or("var_tail_off overflow")?;
    }

    // Append extra equality constraints (x == y) in merged space, if any.
    if !extra_eqs.is_empty() {
        for &(x, y) in extra_eqs {
            // (x - y) * 1 = 0
            let a0 = w.a_terms;
            w.push_a_term(&F::ONE, x as u64)?;
            w.push_a_term(&(-F::ONE), y as u64)?;
            let a1 = w.a_terms;
            let b0 = w.b_terms;
            w.push_b_term(&F::ONE, 0)?;
            let b1 = w.b_terms;
            let c0 = w.c_terms;
            w.push_c_term(&F::ZERO, 0)?;
            let c1 = w.c_terms;
            w.push_constraint_row(a0, a1, b0, b1, c0, c1)?;
        }
    }

    // Finalize.
    let out_inst = w.finish(new_assignment.len())?;
    Ok((out_inst, new_assignment))
}

