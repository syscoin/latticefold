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
#[cfg(unix)]
use std::os::unix::fs::FileExt;

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

    /// Prototype checker: replays all constraints by reading term pools from disk.
    ///
    /// This is intended only for correctness validation of the on-disk format.
    ///
    /// NOTE: This can be extremely expensive for huge instances. It is implemented as a streaming
    /// scan over the constraint rows and term pools (and uses Rayon to parallelize across disjoint
    /// ranges of constraints when available), so it should at least saturate cores instead of
    /// doing pathological per-term open/seek IO.
    pub fn check(&self, assignment: &[F]) -> Result<(), String> {
        if assignment.len() != self.nvars {
            return Err(format!(
                "assignment length mismatch: expected {}, got {}",
                self.nvars,
                assignment.len()
            ));
        }

        // If the writer recorded the number of constraints, use it to avoid relying on EOF.
        let nconstraints = self.layout.nconstraints;
        if nconstraints == 0 {
            return Ok(());
        }

        // Per-thread streaming term reader (sequential reads, rare seeks).
        struct TermStream<F: PrimeField + CanonicalDeserialize> {
            coeff_size: usize,
            fc: BufReader<File>,
            fi: BufReader<File>,
            cur: u64,
            coeff_buf: Vec<u8>,
            _pd: core::marker::PhantomData<F>,
        }
        impl<F: PrimeField + CanonicalDeserialize> TermStream<F> {
            fn open(dir: &Path, which: &str, coeff_size: usize, start_term: u64) -> Result<Self, String> {
                let (p_coeffs, p_idx) = term_paths(dir, which);
                let mut fc = File::open(p_coeffs).map_err(|e| format!("open {which}_coeffs failed: {e}"))?;
                let mut fi = File::open(p_idx).map_err(|e| format!("open {which}_idx failed: {e}"))?;
                let off_c = (start_term as u128)
                    .saturating_mul(coeff_size as u128)
                    .min(u64::MAX as u128) as u64;
                fc.seek(std::io::SeekFrom::Start(off_c))
                    .map_err(|e| format!("seek {which}_coeffs failed: {e}"))?;
                fi.seek(std::io::SeekFrom::Start(start_term.saturating_mul(8)))
                    .map_err(|e| format!("seek {which}_idx failed: {e}"))?;
                Ok(Self {
                    coeff_size,
                    fc: BufReader::with_capacity(8 * 1024 * 1024, fc),
                    fi: BufReader::with_capacity(8 * 1024 * 1024, fi),
                    cur: start_term,
                    coeff_buf: vec![0u8; coeff_size],
                    _pd: core::marker::PhantomData,
                })
            }

            #[inline]
            fn seek_term(&mut self, which: &str, term: u64) -> Result<(), String> {
                if self.cur == term {
                    return Ok(());
                }
                let off_c = (term as u128)
                    .saturating_mul(self.coeff_size as u128)
                    .min(u64::MAX as u128) as u64;
                self.fc
                    .seek(std::io::SeekFrom::Start(off_c))
                    .map_err(|e| format!("seek {which}_coeffs failed: {e}"))?;
                self.fi
                    .seek(std::io::SeekFrom::Start(term.saturating_mul(8)))
                    .map_err(|e| format!("seek {which}_idx failed: {e}"))?;
                self.cur = term;
                Ok(())
            }

            #[inline]
            fn eval_range(&mut self, which: &str, start: u64, end: u64, assignment: &[F]) -> Result<F, String> {
                self.seek_term(which, start)?;
                let mut acc = F::ZERO;
                for _ in start..end {
                    self.fc
                        .read_exact(&mut self.coeff_buf)
                        .map_err(|e| format!("read {which}_coeffs failed: {e}"))?;
                    let idx = read_u64(&mut self.fi).map_err(|e| format!("read {which}_idx failed: {e}"))? as usize;
                    let coef =
                        deserialize_fixed::<F>(&self.coeff_buf).map_err(|e| format!("deserialize {which}_coeff failed: {e}"))?;
                    acc += coef * assignment[idx];
                    self.cur = self.cur.saturating_add(1);
                }
                Ok(acc)
            }
        }

        let coeff_size = self.layout.coeff_size;
        let dir = self.layout.dir.clone();

        // Split constraints into disjoint ranges and check them in parallel.
        let n_threads = rayon::current_num_threads().max(1) as u64;
        let chunk = (nconstraints / (n_threads.saturating_mul(4))).max(1_000_000);
        let mut ranges: Vec<(u64, u64)> = Vec::new();
        let mut s = 0u64;
        while s < nconstraints {
            let e = (s + chunk).min(nconstraints);
            ranges.push((s, e));
            s = e;
        }

        use rayon::prelude::*;
        ranges
            .into_par_iter()
            .try_for_each(|(c_start, c_end)| -> Result<(), String> {
                // Open constraints and seek to start row.
                let mut fr = File::open(constraints_path(&dir)).map_err(|e| format!("open constraints failed: {e}"))?;
                fr.seek(std::io::SeekFrom::Start(c_start.saturating_mul(48)))
                    .map_err(|e| format!("seek constraints failed: {e}"))?;
                let mut rows = BufReader::with_capacity(8 * 1024 * 1024, fr);

                // Read first row to get starting term pointers.
                let mut first = true;
                let mut ts_a: Option<TermStream<F>> = None;
                let mut ts_b: Option<TermStream<F>> = None;
                let mut ts_c: Option<TermStream<F>> = None;

                for ci in c_start..c_end {
                    let a0 = read_u64(&mut rows).map_err(|e| format!("read constraint row failed: {e}"))?;
                    let a1 = read_u64(&mut rows).map_err(|e| format!("read constraint row failed: {e}"))?;
                    let b0 = read_u64(&mut rows).map_err(|e| format!("read constraint row failed: {e}"))?;
                    let b1 = read_u64(&mut rows).map_err(|e| format!("read constraint row failed: {e}"))?;
                    let c0 = read_u64(&mut rows).map_err(|e| format!("read constraint row failed: {e}"))?;
                    let c1 = read_u64(&mut rows).map_err(|e| format!("read constraint row failed: {e}"))?;

                    if first {
                        ts_a = Some(TermStream::<F>::open(&dir, "a", coeff_size, a0)?);
                        ts_b = Some(TermStream::<F>::open(&dir, "b", coeff_size, b0)?);
                        ts_c = Some(TermStream::<F>::open(&dir, "c", coeff_size, c0)?);
                        first = false;
                    }

                    let a = ts_a.as_mut().unwrap().eval_range("a", a0, a1, assignment)?;
                    let b = ts_b.as_mut().unwrap().eval_range("b", b0, b1, assignment)?;
                    let c = ts_c.as_mut().unwrap().eval_range("c", c0, c1, assignment)?;
                    if a * b != c {
                        return Err(format!("constraint {ci} failed"));
                    }
                }
                Ok(())
            })
    }

    /// Debug helper: evaluate one constraint and return `(A·z, B·z, C·z)` along with ranges.
    ///
    /// Intended for postmortem debugging if `check()` reports a failing constraint index.
    pub fn debug_constraint(
        &self,
        assignment: &[F],
        constraint_idx: u64,
        show_terms: usize,
    ) -> Result<(), String> {
        if assignment.len() != self.nvars {
            return Err(format!(
                "assignment length mismatch: expected {}, got {}",
                self.nvars,
                assignment.len()
            ));
        }
        if constraint_idx >= self.layout.nconstraints {
            return Err(format!(
                "constraint_idx out of range: {} >= {}",
                constraint_idx, self.layout.nconstraints
            ));
        }

        // Read the constraint row.
        let mut fr = File::open(constraints_path(&self.layout.dir))
            .map_err(|e| format!("open constraints failed: {e}"))?;
        fr.seek(std::io::SeekFrom::Start(constraint_idx.saturating_mul(48)))
            .map_err(|e| format!("seek constraints failed: {e}"))?;
        let mut r = BufReader::new(fr);
        let a0 = read_u64(&mut r).map_err(|e| e.to_string())?;
        let a1 = read_u64(&mut r).map_err(|e| e.to_string())?;
        let b0 = read_u64(&mut r).map_err(|e| e.to_string())?;
        let b1 = read_u64(&mut r).map_err(|e| e.to_string())?;
        let c0 = read_u64(&mut r).map_err(|e| e.to_string())?;
        let c1 = read_u64(&mut r).map_err(|e| e.to_string())?;

        let eval_range = |which: &str, start: u64, end: u64| -> Result<(F, Vec<(F, usize)>), String> {
            let (p_coeffs, p_idx) = term_paths(&self.layout.dir, which);
            let mut fc = File::open(p_coeffs).map_err(|e| format!("open {which}_coeffs failed: {e}"))?;
            let mut fi = File::open(p_idx).map_err(|e| format!("open {which}_idx failed: {e}"))?;
            let off_c = (start as u128)
                .saturating_mul(self.layout.coeff_size as u128)
                .min(u64::MAX as u128) as u64;
            fc.seek(std::io::SeekFrom::Start(off_c))
                .map_err(|e| format!("seek {which}_coeffs failed: {e}"))?;
            fi.seek(std::io::SeekFrom::Start(start.saturating_mul(8)))
                .map_err(|e| format!("seek {which}_idx failed: {e}"))?;
            let mut buf = vec![0u8; self.layout.coeff_size];
            let mut acc = F::ZERO;
            let mut shown: Vec<(F, usize)> = Vec::new();
            let mut i = 0usize;
            for _ in start..end {
                fc.read_exact(&mut buf).map_err(|e| format!("read {which}_coeffs failed: {e}"))?;
                let idx = read_u64(&mut fi).map_err(|e| format!("read {which}_idx failed: {e}"))? as usize;
                let coef = deserialize_fixed::<F>(&buf).map_err(|e| format!("deserialize {which}_coeff failed: {e}"))?;
                acc += coef * assignment[idx];
                if i < show_terms {
                    shown.push((coef, idx));
                }
                i += 1;
            }
            Ok((acc, shown))
        };

        let (av, a_shown) = eval_range("a", a0, a1)?;
        let (bv, b_shown) = eval_range("b", b0, b1)?;
        let (cv, c_shown) = eval_range("c", c0, c1)?;
        let residual = av * bv - cv;

        eprintln!(
            "[file_backed_debug] constraint {} ranges: A=[{},{})(len={}) B=[{},{})(len={}) C=[{},{})(len={})",
            constraint_idx,
            a0, a1, a1.saturating_sub(a0),
            b0, b1, b1.saturating_sub(b0),
            c0, c1, c1.saturating_sub(c0),
        );
        eprintln!(
            "[file_backed_debug] values: (A·z)={av:?} (B·z)={bv:?} (C·z)={cv:?}  residual=A·B-C={residual:?}"
        );
        if show_terms > 0 {
            eprintln!("[file_backed_debug] A first {} terms: {:?}", a_shown.len(), a_shown);
            eprintln!("[file_backed_debug] B first {} terms: {:?}", b_shown.len(), b_shown);
            eprintln!("[file_backed_debug] C first {} terms: {:?}", c_shown.len(), c_shown);
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
    // Validate and compute merged assignment.
    // This can be a huge memory copy (hundreds of millions of field elements), so parallelize it
    // when Rayon has >1 thread available.
    let timing = match std::env::var("LF_PROFILE_DR1CS") {
        Ok(v) => v != "0",
        Err(_) => false,
    };
    let t_asg = std::time::Instant::now();
    let mut tail_lens: Vec<usize> = Vec::with_capacity(parts.len());
    let mut total_tail: usize = 0;
    for (_inst, asg) in parts.iter() {
        if asg.is_empty() || asg[0] != F::ONE {
            return Err("merge_file_backed_sparse_dr1cs_share_one: each part must have assignment[0]=1".to_string());
        }
        let tl = asg.len().saturating_sub(1);
        tail_lens.push(tl);
        total_tail = total_tail.saturating_add(tl);
    }
    let total_len = 1usize.saturating_add(total_tail);
    let mut new_assignment: Vec<F> = vec![F::ZERO; total_len];
    new_assignment[0] = F::ONE;
    // Prefix offsets into the tail region (starting at index 1).
    let mut tail_off: Vec<usize> = Vec::with_capacity(parts.len());
    {
        let mut cur = 0usize;
        for &tl in &tail_lens {
            tail_off.push(cur);
            cur = cur.saturating_add(tl);
        }
    }
    let n_threads = rayon::current_num_threads().max(1);
    if n_threads > 1 && parts.len() > 1 {
        use rayon::prelude::*;
        // SAFETY: each iteration writes to a disjoint slice of `new_assignment`.
        // Use a `usize` address so the captured value is `Sync + Send` for Rayon.
        let out_base = new_assignment.as_mut_ptr() as usize;
        (0..parts.len())
            .into_par_iter()
            .try_for_each(|pi| -> Result<(), String> {
                let tl = tail_lens[pi];
                if tl == 0 {
                    return Ok(());
                }
                let src = &parts[pi].1[1..];
                let dst_start = 1usize
                    .checked_add(tail_off[pi])
                    .ok_or_else(|| "merge_file_backed_sparse_dr1cs_share_one: assignment offset overflow".to_string())?;
                unsafe {
                    let dst_ptr = (out_base as *mut F).add(dst_start);
                    let dst = core::slice::from_raw_parts_mut(dst_ptr, tl);
                    dst.copy_from_slice(src);
                }
                Ok(())
            })?;
    } else {
        for (pi, (_inst, asg)) in parts.iter().enumerate() {
            let tl = tail_lens[pi];
            if tl == 0 {
                continue;
            }
            let start = 1 + tail_off[pi];
            new_assignment[start..start + tl].copy_from_slice(&asg[1..]);
        }
    }
    if timing {
        eprintln!(
            "file_backed_merge: assignment concat done in {:?} (nvars={})",
            t_asg.elapsed(),
            new_assignment.len()
        );
    }

    // IMPORTANT: merge is dominated by streaming IO + index remapping.
    // Large buffers significantly reduce syscall overhead during merges.
    // Use a moderately large chunk size per worker to avoid over-allocating when many Rayon workers run.
    const MERGE_BUF_BYTES: usize = 256 * 1024 * 1024;
    const STREAM_CHUNK_BYTES: usize = 32 * 1024 * 1024;
    let t_all = std::time::Instant::now();

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
        if timing {
            eprintln!(
                "file_backed_merge: parallel start parts={} threads={} out_dir={}",
                parts.len(),
                n_threads,
                out_dir.display()
            );
        }

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
        let mut row_off: Vec<u64> = Vec::with_capacity(parts.len());
        let mut cur_a: u64 = 0;
        let mut cur_b: u64 = 0;
        let mut cur_c: u64 = 0;
        let mut cur_rows: u64 = 0;
        for (inst, _asg) in parts.iter() {
            a_off.push(cur_a);
            b_off.push(cur_b);
            c_off.push(cur_c);
            row_off.push(cur_rows);
            cur_a = cur_a.saturating_add(inst.layout.a_terms);
            cur_b = cur_b.saturating_add(inst.layout.b_terms);
            cur_c = cur_c.saturating_add(inst.layout.c_terms);
            cur_rows = cur_rows.saturating_add(inst.layout.nconstraints);
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

        if timing {
            eprintln!("file_backed_merge: part ranges (vars/terms/rows):");
            for (pi, (inst, asg)) in parts.iter().enumerate() {
                let v_tail = var_tail_off[pi];
                let v_start = 1u64.saturating_add(v_tail);
                let v_end = v_start.saturating_add(asg.len().saturating_sub(1) as u64);
                let a0 = a_off[pi];
                let b0 = b_off[pi];
                let c0 = c_off[pi];
                let r0 = row_off[pi];
                eprintln!(
                    "  part[{pi}]: nvars={} (tail [{}..{}))  A[{}..{}) B[{}..{}) C[{}..{}) rows[{}..{}) dir={}",
                    asg.len(),
                    v_start,
                    v_end,
                    a0,
                    a0.saturating_add(inst.layout.a_terms),
                    b0,
                    b0.saturating_add(inst.layout.b_terms),
                    c0,
                    c0.saturating_add(inst.layout.c_terms),
                    r0,
                    r0.saturating_add(inst.layout.nconstraints),
                    inst.layout.dir.display()
                );
            }
            eprintln!(
                "file_backed_merge: totals pre-eqs: nvars={} A={} B={} C={} rows={}",
                new_assignment.len(),
                cur_a,
                cur_b,
                cur_c,
                cur_rows
            );
            eprintln!(
                "file_backed_merge: appended-eqs: eqs={} => final A={} B={} C={} rows={}",
                extra_eqs.len(),
                cur_a.saturating_add((extra_eqs.len() as u64).saturating_mul(2)),
                cur_b.saturating_add(extra_eqs.len() as u64),
                cur_c.saturating_add(extra_eqs.len() as u64),
                cur_rows.saturating_add(extra_eqs.len() as u64),
            );
        }

        // Helper: map local var index to global.
        #[inline]
        fn map_var(idx: u64, var_tail_off: u64) -> u64 {
            if idx == 0 { 0 } else { idx + var_tail_off }
        }

        #[cfg(unix)]
        {
            // Final sizes include appended equality constraints.
            let eqs = extra_eqs.len() as u64;
            let tot_a_terms = cur_a.saturating_add(eqs.saturating_mul(2));
            let tot_b_terms = cur_b.saturating_add(eqs);
            let tot_c_terms = cur_c.saturating_add(eqs);
            let tot_rows = cur_rows.saturating_add(eqs);

            let bytes_a_coeff = (tot_a_terms as u128)
                .saturating_mul(coeff_size as u128)
                .min(u64::MAX as u128) as u64;
            let bytes_b_coeff = (tot_b_terms as u128)
                .saturating_mul(coeff_size as u128)
                .min(u64::MAX as u128) as u64;
            let bytes_c_coeff = (tot_c_terms as u128)
                .saturating_mul(coeff_size as u128)
                .min(u64::MAX as u128) as u64;
            let bytes_a_idx = (tot_a_terms as u128).saturating_mul(8).min(u64::MAX as u128) as u64;
            let bytes_b_idx = (tot_b_terms as u128).saturating_mul(8).min(u64::MAX as u128) as u64;
            let bytes_c_idx = (tot_c_terms as u128).saturating_mul(8).min(u64::MAX as u128) as u64;
            let bytes_rows = (tot_rows as u128).saturating_mul(48).min(u64::MAX as u128) as u64;

            // Create output files and pre-size them to allow concurrent pwrite.
            let out_fc_a = File::create(out_dir.join("a_coeffs.bin")).map_err(|e| e.to_string())?;
            let out_fi_a = File::create(out_dir.join("a_idx.bin")).map_err(|e| e.to_string())?;
            let out_fc_b = File::create(out_dir.join("b_coeffs.bin")).map_err(|e| e.to_string())?;
            let out_fi_b = File::create(out_dir.join("b_idx.bin")).map_err(|e| e.to_string())?;
            let out_fc_c = File::create(out_dir.join("c_coeffs.bin")).map_err(|e| e.to_string())?;
            let out_fi_c = File::create(out_dir.join("c_idx.bin")).map_err(|e| e.to_string())?;
            let out_rows = File::create(out_dir.join("constraints.bin")).map_err(|e| e.to_string())?;

            out_fc_a.set_len(bytes_a_coeff).map_err(|e| e.to_string())?;
            out_fc_b.set_len(bytes_b_coeff).map_err(|e| e.to_string())?;
            out_fc_c.set_len(bytes_c_coeff).map_err(|e| e.to_string())?;
            out_fi_a.set_len(bytes_a_idx).map_err(|e| e.to_string())?;
            out_fi_b.set_len(bytes_b_idx).map_err(|e| e.to_string())?;
            out_fi_c.set_len(bytes_c_idx).map_err(|e| e.to_string())?;
            out_rows.set_len(bytes_rows).map_err(|e| e.to_string())?;

            // Helper to fully write with pwrite.
            #[inline]
            fn pwrite_all(f: &File, mut off: u64, mut buf: &[u8]) -> Result<(), String> {
                while !buf.is_empty() {
                    let n = f.write_at(buf, off).map_err(|e| e.to_string())?;
                    if n == 0 {
                        return Err("pwrite returned 0".to_string());
                    }
                    off = off.saturating_add(n as u64);
                    buf = &buf[n..];
                }
                Ok(())
            }

            // Create a job per (part, pool) to increase parallelism (3 term pools + rows per part).
            #[derive(Clone, Copy)]
            enum JobKind {
                CoeffA,
                IdxA,
                CoeffB,
                IdxB,
                CoeffC,
                IdxC,
                Rows,
            }
            #[derive(Clone, Copy)]
            struct Job {
                pi: usize,
                kind: JobKind,
            }
            let mut jobs: Vec<Job> = Vec::with_capacity(parts.len() * 7);
            for pi in 0..parts.len() {
                jobs.push(Job { pi, kind: JobKind::CoeffA });
                jobs.push(Job { pi, kind: JobKind::IdxA });
                jobs.push(Job { pi, kind: JobKind::CoeffB });
                jobs.push(Job { pi, kind: JobKind::IdxB });
                jobs.push(Job { pi, kind: JobKind::CoeffC });
                jobs.push(Job { pi, kind: JobKind::IdxC });
                jobs.push(Job { pi, kind: JobKind::Rows });
            }

            let t_direct = std::time::Instant::now();
            jobs.into_par_iter().try_for_each(|job| -> Result<(), String> {
                let (inst, _asg) = &parts[job.pi];
                let v_off = var_tail_off[job.pi];
                match job.kind {
                    JobKind::CoeffA | JobKind::CoeffB | JobKind::CoeffC => {
                        let (which, base_terms, out_file) = match job.kind {
                            JobKind::CoeffA => ("a", a_off[job.pi], &out_fc_a),
                            JobKind::CoeffB => ("b", b_off[job.pi], &out_fc_b),
                            JobKind::CoeffC => ("c", c_off[job.pi], &out_fc_c),
                            _ => unreachable!(),
                        };
                        let (p_coeffs, _p_idx) = term_paths(&inst.layout.dir, which);
                        let mut r = BufReader::with_capacity(
                            MERGE_BUF_BYTES.min(STREAM_CHUNK_BYTES),
                            File::open(p_coeffs).map_err(|e| format!("open {which}_coeffs failed: {e}"))?,
                        );
                        let mut buf = vec![0u8; STREAM_CHUNK_BYTES];
                        let mut wrote: u64 = 0;
                        let out_base = (base_terms as u128)
                            .saturating_mul(coeff_size as u128)
                            .min(u64::MAX as u128) as u64;
                        loop {
                            let n = r.read(&mut buf).map_err(|e| e.to_string())?;
                            if n == 0 {
                                break;
                            }
                            pwrite_all(out_file, out_base.saturating_add(wrote), &buf[..n])?;
                            wrote = wrote.saturating_add(n as u64);
                        }
                        Ok(())
                    }
                    JobKind::IdxA | JobKind::IdxB | JobKind::IdxC => {
                        let (which, base_terms, n_terms, out_file) = match job.kind {
                            JobKind::IdxA => ("a", a_off[job.pi], inst.layout.a_terms, &out_fi_a),
                            JobKind::IdxB => ("b", b_off[job.pi], inst.layout.b_terms, &out_fi_b),
                            JobKind::IdxC => ("c", c_off[job.pi], inst.layout.c_terms, &out_fi_c),
                            _ => unreachable!(),
                        };
                        let (_p_coeffs, p_idx) = term_paths(&inst.layout.dir, which);
                        let mut r = BufReader::with_capacity(
                            MERGE_BUF_BYTES.min(STREAM_CHUNK_BYTES),
                            File::open(p_idx).map_err(|e| format!("open {which}_idx failed: {e}"))?,
                        );

                        // Process u64 indices in big blocks.
                        let mut remaining = n_terms;
                        let mut out_u64_pos = base_terms;
                        let block_u64s = (STREAM_CHUNK_BYTES / 8).max(1) as u64;
                        let mut buf = vec![0u8; (block_u64s as usize) * 8];
                        while remaining > 0 {
                            let take = remaining.min(block_u64s);
                            let bytes = (take as usize) * 8;
                            r.read_exact(&mut buf[..bytes]).map_err(|e| e.to_string())?;
                            for i in 0..(take as usize) {
                                let j = i * 8;
                                let idx = u64::from_le_bytes(buf[j..j + 8].try_into().unwrap());
                                let mapped = map_var(idx, v_off);
                                buf[j..j + 8].copy_from_slice(&mapped.to_le_bytes());
                            }
                            let out_off = (out_u64_pos as u128).saturating_mul(8).min(u64::MAX as u128) as u64;
                            pwrite_all(out_file, out_off, &buf[..bytes])?;
                            out_u64_pos = out_u64_pos.saturating_add(take);
                            remaining -= take;
                        }
                        Ok(())
                    }
                    JobKind::Rows => {
                        let in_path = constraints_path(&inst.layout.dir);
                        let mut r = BufReader::with_capacity(
                            MERGE_BUF_BYTES.min(STREAM_CHUNK_BYTES),
                            File::open(in_path).map_err(|e| format!("open constraints failed: {e}"))?,
                        );
                        let base_a = a_off[job.pi];
                        let base_b = b_off[job.pi];
                        let base_c = c_off[job.pi];
                        let base_row = row_off[job.pi];

                        let mut remaining = inst.layout.nconstraints;
                        let block_rows = (STREAM_CHUNK_BYTES / 48).max(1) as u64;
                        let mut buf = vec![0u8; (block_rows as usize) * 48];
                        let mut out_row_pos = base_row;
                        while remaining > 0 {
                            let take = remaining.min(block_rows);
                            let bytes = (take as usize) * 48;
                            r.read_exact(&mut buf[..bytes]).map_err(|e| e.to_string())?;
                            for i in 0..(take as usize) {
                                let j = i * 48;
                                let read6 = |k: usize| -> u64 {
                                    u64::from_le_bytes(buf[j + 8 * k..j + 8 * (k + 1)].try_into().unwrap())
                                };
                                let a0 = read6(0);
                                let a1 = read6(1);
                                let b0 = read6(2);
                                let b1 = read6(3);
                                let c0 = read6(4);
                                let c1 = read6(5);
                                let mut w6 = |k: usize, x: u64| {
                                    buf[j + 8 * k..j + 8 * (k + 1)].copy_from_slice(&x.to_le_bytes());
                                };
                                w6(0, base_a + a0);
                                w6(1, base_a + a1);
                                w6(2, base_b + b0);
                                w6(3, base_b + b1);
                                w6(4, base_c + c0);
                                w6(5, base_c + c1);
                            }
                            let out_off = (out_row_pos as u128).saturating_mul(48).min(u64::MAX as u128) as u64;
                            pwrite_all(&out_rows, out_off, &buf[..bytes])?;
                            out_row_pos = out_row_pos.saturating_add(take);
                            remaining -= take;
                        }
                        Ok(())
                    }
                }
            })?;
            if timing {
                eprintln!("file_backed_merge: direct-write pools done in {:?}", t_direct.elapsed());
            }

            // Append equality constraints at the tail (write_at into pre-sized file ranges).
            if !extra_eqs.is_empty() {
                let t_eq = std::time::Instant::now();
                let mut one_bytes = vec![0u8; coeff_size];
                let mut neg_one_bytes = vec![0u8; coeff_size];
                let mut zero_bytes = vec![0u8; coeff_size];
                serialize_fixed::<F>(&F::ONE, &mut one_bytes).map_err(|e| e.to_string())?;
                serialize_fixed::<F>(&(-F::ONE), &mut neg_one_bytes).map_err(|e| e.to_string())?;
                serialize_fixed::<F>(&F::ZERO, &mut zero_bytes).map_err(|e| e.to_string())?;

                let base_a_terms = cur_a;
                let base_b_terms = cur_b;
                let base_c_terms = cur_c;
                let base_rows = cur_rows;

                const EQ_BATCH: usize = 4096;
                let mut i0 = 0usize;
                while i0 < extra_eqs.len() {
                    let i1 = (i0 + EQ_BATCH).min(extra_eqs.len());
                    let batch = &extra_eqs[i0..i1];

                    // Build batch buffers.
                    let mut a_coeff = vec![0u8; batch.len() * 2 * coeff_size];
                    let mut a_idx = vec![0u8; batch.len() * 2 * 8];
                    let mut b_coeff = vec![0u8; batch.len() * coeff_size];
                    let mut b_idx = vec![0u8; batch.len() * 8];
                    let mut c_coeff = vec![0u8; batch.len() * coeff_size];
                    let mut c_idx = vec![0u8; batch.len() * 8];
                    let mut rows = vec![0u8; batch.len() * 48];

                    for (j, &(x, y)) in batch.iter().enumerate() {
                        // A pool: [+1*x, -1*y]
                        a_coeff[(2 * j) * coeff_size..(2 * j + 1) * coeff_size].copy_from_slice(&one_bytes);
                        a_coeff[(2 * j + 1) * coeff_size..(2 * j + 2) * coeff_size].copy_from_slice(&neg_one_bytes);
                        a_idx[(2 * j) * 8..(2 * j + 1) * 8].copy_from_slice(&(x as u64).to_le_bytes());
                        a_idx[(2 * j + 1) * 8..(2 * j + 2) * 8].copy_from_slice(&(y as u64).to_le_bytes());

                        // B pool: [+1*var0]
                        b_coeff[j * coeff_size..(j + 1) * coeff_size].copy_from_slice(&one_bytes);
                        b_idx[j * 8..(j + 1) * 8].copy_from_slice(&0u64.to_le_bytes());

                        // C pool: [0*var0]
                        c_coeff[j * coeff_size..(j + 1) * coeff_size].copy_from_slice(&zero_bytes);
                        c_idx[j * 8..(j + 1) * 8].copy_from_slice(&0u64.to_le_bytes());

                        // Constraint row for this equality.
                        let eq_i = (i0 + j) as u64;
                        let a0 = base_a_terms + 2 * eq_i;
                        let a1 = a0 + 2;
                        let b0 = base_b_terms + eq_i;
                        let b1 = b0 + 1;
                        let c0 = base_c_terms + eq_i;
                        let c1 = c0 + 1;
                        let w6 = |buf: &mut [u8], k: usize, v: u64| {
                            buf[8 * k..8 * (k + 1)].copy_from_slice(&v.to_le_bytes());
                        };
                        let row = &mut rows[j * 48..(j + 1) * 48];
                        w6(row, 0, a0);
                        w6(row, 1, a1);
                        w6(row, 2, b0);
                        w6(row, 3, b1);
                        w6(row, 4, c0);
                        w6(row, 5, c1);
                    }

                    // Offsets (in bytes) to write this batch.
                    let eq0 = i0 as u64;
                    let a_coeff_off = ((base_a_terms + 2 * eq0) as u128)
                        .saturating_mul(coeff_size as u128)
                        .min(u64::MAX as u128) as u64;
                    let a_idx_off = ((base_a_terms + 2 * eq0) as u128).saturating_mul(8).min(u64::MAX as u128) as u64;
                    let b_coeff_off = ((base_b_terms + eq0) as u128)
                        .saturating_mul(coeff_size as u128)
                        .min(u64::MAX as u128) as u64;
                    let b_idx_off = ((base_b_terms + eq0) as u128).saturating_mul(8).min(u64::MAX as u128) as u64;
                    let c_coeff_off = ((base_c_terms + eq0) as u128)
                        .saturating_mul(coeff_size as u128)
                        .min(u64::MAX as u128) as u64;
                    let c_idx_off = ((base_c_terms + eq0) as u128).saturating_mul(8).min(u64::MAX as u128) as u64;
                    let rows_off = ((base_rows + eq0) as u128).saturating_mul(48).min(u64::MAX as u128) as u64;

                    pwrite_all(&out_fc_a, a_coeff_off, &a_coeff)?;
                    pwrite_all(&out_fi_a, a_idx_off, &a_idx)?;
                    pwrite_all(&out_fc_b, b_coeff_off, &b_coeff)?;
                    pwrite_all(&out_fi_b, b_idx_off, &b_idx)?;
                    pwrite_all(&out_fc_c, c_coeff_off, &c_coeff)?;
                    pwrite_all(&out_fi_c, c_idx_off, &c_idx)?;
                    pwrite_all(&out_rows, rows_off, &rows)?;

                    i0 = i1;
                }

                if timing {
                    eprintln!("file_backed_merge: appended eqs={} in {:?}", extra_eqs.len(), t_eq.elapsed());
                }
            }

            // Write meta (human-readable).
            {
                let mut f =
                    BufWriter::new(File::create(meta_path(&out_dir)).map_err(|e| format!("create meta failed: {e}"))?);
                writeln!(f, "nvars={}", new_assignment.len()).ok();
                writeln!(f, "constraints={}", tot_rows).ok();
                writeln!(f, "a_terms={}", tot_a_terms).ok();
                writeln!(f, "b_terms={}", tot_b_terms).ok();
                writeln!(f, "c_terms={}", tot_c_terms).ok();
                writeln!(f, "coeff_size={}", coeff_size).ok();
            }

            if timing {
                eprintln!("file_backed_merge: done total {:?}", t_all.elapsed());
            }

            let layout = FileBackedLayout {
                dir: out_dir.clone(),
                coeff_size,
                nconstraints: tot_rows,
                a_terms: tot_a_terms,
                b_terms: tot_b_terms,
                c_terms: tot_c_terms,
            };
            return Ok((
                FileBackedSparseDr1csInstance { nvars: new_assignment.len(), layout, _pd: core::marker::PhantomData },
                new_assignment,
            ));
        }
        #[cfg(not(unix))]
        {
            // Non-unix fallback: keep the sequential merge path below.
        }
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

