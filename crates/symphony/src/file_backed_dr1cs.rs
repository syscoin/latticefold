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
    // CanonicalSerialize writes to a std::io::Write; use a Vec and copy.
    let mut v = Vec::with_capacity(out.len());
    x.serialize_with_mode(&mut v, Compress::Yes)?;
    if v.len() != out.len() {
        return Err(SerializationError::InvalidData);
    }
    out.copy_from_slice(&v);
    Ok(())
}

fn deserialize_fixed<F: CanonicalDeserialize>(buf: &[u8]) -> Result<F, SerializationError> {
    let mut r = BufReader::new(buf);
    F::deserialize_with_mode(&mut r, Compress::Yes, ark_serialize::Validate::Yes)
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
    _pd: core::marker::PhantomData<F>,
}

impl<F: PrimeField + CanonicalSerialize> SparseDr1csFileWriter<F> {
    pub fn create(dir: impl AsRef<Path>) -> Result<Self, String> {
        let dir = dir.as_ref();
        create_dir_all(dir).map_err(|e| format!("create_dir_all failed: {e}"))?;

        // Determine fixed coeff size.
        let mut tmp = Vec::new();
        F::ONE
            .serialize_with_mode(&mut tmp, Compress::Yes)
            .map_err(|e| format!("serialize ONE failed: {e}"))?;
        let coeff_size = tmp.len();
        if coeff_size == 0 {
            return Err("invalid coeff_size=0".to_string());
        }

        let (pa_c, pa_i) = term_paths(dir, "a");
        let (pb_c, pb_i) = term_paths(dir, "b");
        let (pc_c, pc_i) = term_paths(dir, "c");
        let fc_a = BufWriter::new(File::create(pa_c).map_err(|e| format!("create a_coeffs failed: {e}"))?);
        let fi_a = BufWriter::new(File::create(pa_i).map_err(|e| format!("create a_idx failed: {e}"))?);
        let fc_b = BufWriter::new(File::create(pb_c).map_err(|e| format!("create b_coeffs failed: {e}"))?);
        let fi_b = BufWriter::new(File::create(pb_i).map_err(|e| format!("create b_idx failed: {e}"))?);
        let fc_c = BufWriter::new(File::create(pc_c).map_err(|e| format!("create c_coeffs failed: {e}"))?);
        let fi_c = BufWriter::new(File::create(pc_i).map_err(|e| format!("create c_idx failed: {e}"))?);
        let f_rows = BufWriter::new(
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
        serialize_fixed(coef, &mut self.coeff_buf).map_err(|e| format!("serialize coeff failed: {e}"))?;
        self.fc_a.write_all(&self.coeff_buf).map_err(|e| e.to_string())?;
        write_u64(&mut self.fi_a, idx).map_err(|e| e.to_string())?;
        self.a_terms += 1;
        Ok(())
    }
    #[inline]
    pub fn push_b_term(&mut self, coef: &F, idx: u64) -> Result<(), String> {
        serialize_fixed(coef, &mut self.coeff_buf).map_err(|e| format!("serialize coeff failed: {e}"))?;
        self.fc_b.write_all(&self.coeff_buf).map_err(|e| e.to_string())?;
        write_u64(&mut self.fi_b, idx).map_err(|e| e.to_string())?;
        self.b_terms += 1;
        Ok(())
    }
    #[inline]
    pub fn push_c_term(&mut self, coef: &F, idx: u64) -> Result<(), String> {
        serialize_fixed(coef, &mut self.coeff_buf).map_err(|e| format!("serialize coeff failed: {e}"))?;
        self.fc_c.write_all(&self.coeff_buf).map_err(|e| e.to_string())?;
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
        .serialize_with_mode(&mut tmp, Compress::Yes)
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
            let mut fc = BufReader::new(File::open(p_coeffs).map_err(|e| format!("open {which}_coeffs failed: {e}"))?);
            let mut fi = BufReader::new(File::open(p_idx).map_err(|e| format!("open {which}_idx failed: {e}"))?);
            let mut buf = vec![0u8; inst.layout.coeff_size];
            for _ in 0..n_terms {
                fc.read_exact(&mut buf).map_err(|e| format!("read {which}_coeffs failed: {e}"))?;
                let idx = read_u64(&mut fi).map_err(|e| format!("read {which}_idx failed: {e}"))?;
                let mapped = map_var(idx, var_tail_off);
                let coef = deserialize_fixed::<F>(&buf).map_err(|e| format!("deserialize coeff failed: {e}"))?;
                match which {
                    "a" => w.push_a_term(&coef, mapped)?,
                    "b" => w.push_b_term(&coef, mapped)?,
                    "c" => w.push_c_term(&coef, mapped)?,
                    _ => unreachable!(),
                }
            }
            *out_terms_off += n_terms;
        }

        // Copy constraints, offsetting term ranges by out_a/out_b/out_c *before* this part.
        let mut fr = BufReader::new(
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

