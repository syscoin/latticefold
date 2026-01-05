//! SP1 R1CS integration for Symphony.
//!
//! Loads pre-compiled R1CS from SP1's shrink verifier and converts to Symphony format.
//! Supports:
//! - Power-of-2 padding (required for MLE sumcheck)
//! - Symphony-native serialization (avoid re-conversion)
//! - Parallel conversion (always enabled via rayon)

use crate::sp1_r1cs_loader::{SP1R1CS, FieldFromU64, SparseRow as LoaderSparseRow};
use rayon::prelude::*;
use stark_rings::{OverField, Zq};
use stark_rings_linalg::SparseMatrix;
use std::io::{Read, Write};

/// Round up to next power of 2.
pub fn next_power_of_two(n: usize) -> usize {
    if n == 0 { return 1; }
    1usize << (usize::BITS - (n - 1).leading_zeros())
}

/// Convert SP1 R1CS to Symphony sparse matrices [A, B, C] with power-of-2 padding.
/// 
/// Symphony requires m (rows) to be a power of 2 for MLE sumcheck.
/// Extra rows are filled with 0·w[0] = 0·w[0] (trivially satisfied).
/// 
/// Returns (matrices, padded_rows, padded_cols).
/// 
/// Uses rayon for parallel conversion of rows and matrices.
pub fn sp1_r1cs_to_symphony_matrices_padded<R, F>(
    r1cs: &SP1R1CS<F>,
) -> ([SparseMatrix<R>; 3], usize, usize)
where
    R: OverField + Send + Sync,
    R::BaseRing: Zq + From<u64> + Send + Sync,
    F: FieldFromU64 + Clone + Send + Sync,
{
    let orig_rows = r1cs.num_constraints;
    let orig_cols = r1cs.num_vars;
    
    let padded_rows = next_power_of_two(orig_rows);
    let padded_cols = next_power_of_two(orig_cols);
    
    fn convert_matrix_parallel<R, F>(
        rows: &[LoaderSparseRow<F>], 
        padded_rows: usize, 
        padded_cols: usize
    ) -> SparseMatrix<R>
    where
        R: OverField + Send + Sync,
        R::BaseRing: Zq + From<u64> + Send + Sync,
        F: FieldFromU64 + Clone + Send + Sync,
    {
        // Convert rows in parallel
        let converted: Vec<Vec<(R, usize)>> = rows
            .par_iter()
            .map(|row| {
                row.terms
                    .iter()
                    .map(|(col_idx, coeff)| {
                        let val = R::from(R::BaseRing::from(coeff.as_canonical_u64()));
                        (val, *col_idx)
                    })
                    .collect()
            })
            .collect();
        
        // Extend with empty padding rows
        let mut coeffs = converted;
        coeffs.resize_with(padded_rows, Vec::new);
        
        SparseMatrix { nrows: padded_rows, ncols: padded_cols, coeffs }
    }
    
    // Convert all 3 matrices in parallel
    let (m1, (m2, m3)) = rayon::join(
        || convert_matrix_parallel(&r1cs.a, padded_rows, padded_cols),
        || rayon::join(
            || convert_matrix_parallel(&r1cs.b, padded_rows, padded_cols),
            || convert_matrix_parallel(&r1cs.c, padded_rows, padded_cols),
        ),
    );
    
    ([m1, m2, m3], padded_rows, padded_cols)
}

/// Load SP1 shrink verifier R1CS from file and convert to Symphony format.
/// Returns ([A, B, C] matrices with power-of-2 padding, stats).
pub fn load_sp1_r1cs_as_symphony<R, F>(
    path: &str,
    expected_digest: Option<&[u8; 32]>,
) -> std::io::Result<([SparseMatrix<R>; 3], SP1R1CSStats)>
where
    R: OverField + Send + Sync,
    R::BaseRing: Zq + From<u64> + Send + Sync,
    F: FieldFromU64 + Clone + Send + Sync,
{
    let r1cs: SP1R1CS<F> = if let Some(digest) = expected_digest {
        SP1R1CS::load_verified(path, digest)?
    } else {
        SP1R1CS::load(path)?
    };
    
    let digest = r1cs.compute_digest();
    let total_nonzeros = r1cs.a.iter().chain(r1cs.b.iter()).chain(r1cs.c.iter())
        .map(|row| row.terms.len() as u64)
        .sum();
    
    let stats = SP1R1CSStats {
        num_vars: r1cs.num_vars,
        num_constraints: r1cs.num_constraints,
        num_public: r1cs.num_public,
        total_nonzeros,
        digest,
    };
    
    // Use padded version for Symphony (requires power-of-2 rows)
    let (matrices, padded_rows, padded_cols) = sp1_r1cs_to_symphony_matrices_padded(&r1cs);
    
    eprintln!("  Padding: {} rows → {} (2^{}), {} cols → {} (2^{})",
        r1cs.num_constraints, padded_rows, padded_rows.trailing_zeros(),
        r1cs.num_vars, padded_cols, padded_cols.trailing_zeros());
    
    Ok((matrices, stats))
}

/// R1CS stats for quick inspection.
#[derive(Debug, Clone)]
pub struct SP1R1CSStats {
    pub num_vars: usize,
    pub num_constraints: usize,
    pub num_public: usize,
    pub total_nonzeros: u64,
    pub digest: [u8; 32],
}

/// Read just the header/stats from an R1CS file (fast, reads only 72 bytes).
pub fn read_sp1_r1cs_stats(path: &str) -> std::io::Result<SP1R1CSStats> {
    let mut file = std::fs::File::open(path)?;
    let mut header_bytes = [0u8; 72];
    file.read_exact(&mut header_bytes)?;
    
    // Use a dummy field type just for header parsing
    #[derive(Clone)]
    struct DummyField;
    impl FieldFromU64 for DummyField {
        fn from_canonical_u64(_: u64) -> Self { DummyField }
        fn as_canonical_u64(&self) -> u64 { 0 }
    }
    
    let header = SP1R1CS::<DummyField>::read_header(&header_bytes)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
    
    Ok(SP1R1CSStats {
        num_vars: header.num_vars,
        num_constraints: header.num_constraints,
        num_public: header.num_public,
        total_nonzeros: header.total_nonzeros,
        digest: header.digest,
    })
}

// ============================================================================
// Symphony-native serialization
// ============================================================================

/// Symphony matrix file header.
/// Format: SYMM (4) | version (4) | nrows (8) | ncols (8) | nnz (8) | digest (32)
const SYMPHONY_MAGIC: &[u8; 4] = b"SYMM";
const SYMPHONY_VERSION: u32 = 1;

/// Header for Symphony matrix file.
#[derive(Debug, Clone)]
pub struct SymphonyMatrixHeader {
    pub nrows: usize,
    pub ncols: usize,
    pub nnz: u64,
    pub digest: [u8; 32],
}

/// Buffer size for I/O (256 MB)
const IO_BUFFER_SIZE: usize = 256 * 1024 * 1024;

/// Save Symphony matrices to file (avoids re-conversion from SP1 R1CS).
/// 
/// File format:
/// - Header: SYMM | v1 | nrows | ncols | nnz_total | sp1_digest
/// - For each matrix (A, B, C):
///   - For each row: num_terms (u32), then (col_idx u32, coeff u64) pairs
/// 
/// Uses 256MB buffered I/O for performance on large files.
pub fn save_symphony_matrices<R>(
    path: &str,
    matrices: &[SparseMatrix<R>; 3],
    sp1_digest: &[u8; 32],
) -> std::io::Result<()>
where
    R: OverField,
    R::BaseRing: Zq + Into<u64>,
{
    use std::io::BufWriter;
    
    let file = std::fs::File::create(path)?;
    let mut writer = BufWriter::with_capacity(IO_BUFFER_SIZE, file);
    
    let nrows = matrices[0].nrows;
    let ncols = matrices[0].ncols;
    let nnz: u64 = matrices.iter()
        .flat_map(|m| m.coeffs.iter())
        .map(|row| row.len() as u64)
        .sum();
    
    // Write header
    writer.write_all(SYMPHONY_MAGIC)?;
    writer.write_all(&SYMPHONY_VERSION.to_le_bytes())?;
    writer.write_all(&(nrows as u64).to_le_bytes())?;
    writer.write_all(&(ncols as u64).to_le_bytes())?;
    writer.write_all(&nnz.to_le_bytes())?;
    writer.write_all(sp1_digest)?;
    
    // Write matrices
    for matrix in matrices {
        for row in &matrix.coeffs {
            writer.write_all(&(row.len() as u32).to_le_bytes())?;
            for (val, col_idx) in row {
                writer.write_all(&(*col_idx as u32).to_le_bytes())?;
                // Convert ring element to u64 via BaseRing
                let coeff_u64: u64 = val.coeffs()[0].into();
                writer.write_all(&coeff_u64.to_le_bytes())?;
            }
        }
    }
    
    writer.flush()?;
    Ok(())
}

/// Load Symphony matrices from file (fast, no conversion needed).
/// 
/// Uses 256MB buffered I/O for performance on large files.
pub fn load_symphony_matrices<R>(
    path: &str,
) -> std::io::Result<([SparseMatrix<R>; 3], SymphonyMatrixHeader)>
where
    R: OverField,
    R::BaseRing: Zq + From<u64>,
{
    use std::io::BufReader;
    
    let file = std::fs::File::open(path)?;
    let mut reader = BufReader::with_capacity(IO_BUFFER_SIZE, file);
    
    // Read header
    let mut magic = [0u8; 4];
    reader.read_exact(&mut magic)?;
    if &magic != SYMPHONY_MAGIC {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "Invalid Symphony matrix magic",
        ));
    }
    
    let mut buf4 = [0u8; 4];
    let mut buf8 = [0u8; 8];
    
    reader.read_exact(&mut buf4)?;
    let version = u32::from_le_bytes(buf4);
    if version != SYMPHONY_VERSION {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("Unsupported Symphony version: {version}"),
        ));
    }
    
    reader.read_exact(&mut buf8)?;
    let nrows = u64::from_le_bytes(buf8) as usize;
    reader.read_exact(&mut buf8)?;
    let ncols = u64::from_le_bytes(buf8) as usize;
    reader.read_exact(&mut buf8)?;
    let nnz = u64::from_le_bytes(buf8);
    
    let mut digest = [0u8; 32];
    reader.read_exact(&mut digest)?;
    
    let header = SymphonyMatrixHeader { nrows, ncols, nnz, digest };
    
    // Read matrices
    let mut matrices: [SparseMatrix<R>; 3] = std::array::from_fn(|_| SparseMatrix {
        nrows,
        ncols,
        coeffs: Vec::with_capacity(nrows),
    });
    
    for matrix in &mut matrices {
        for _ in 0..nrows {
            reader.read_exact(&mut buf4)?;
            let num_terms = u32::from_le_bytes(buf4) as usize;
            
            let mut row = Vec::with_capacity(num_terms);
            for _ in 0..num_terms {
                reader.read_exact(&mut buf4)?;
                let col_idx = u32::from_le_bytes(buf4) as usize;
                reader.read_exact(&mut buf8)?;
                let coeff = u64::from_le_bytes(buf8);
                let val = R::from(R::BaseRing::from(coeff));
                row.push((val, col_idx));
            }
            matrix.coeffs.push(row);
        }
    }
    
    Ok((matrices, header))
}

/// Load or convert SP1 R1CS to Symphony format.
/// 
/// If a `.symm` cache file exists and matches the SP1 digest, loads from cache.
/// Otherwise converts from SP1 R1CS and saves to cache.
pub fn load_sp1_r1cs_cached<R, F>(
    sp1_path: &str,
) -> std::io::Result<([SparseMatrix<R>; 3], SP1R1CSStats)>
where
    R: OverField + Send + Sync,
    R::BaseRing: Zq + From<u64> + Into<u64> + Send + Sync,
    F: FieldFromU64 + Clone + Send + Sync,
{
    let cache_path = format!("{}.symm", sp1_path);
    
    // Try loading from cache
    if let Ok((matrices, header)) = load_symphony_matrices::<R>(&cache_path) {
        // Verify SP1 digest matches
        let sp1_stats = read_sp1_r1cs_stats(sp1_path)?;
        if header.digest == sp1_stats.digest {
            eprintln!("  Loaded from cache: {cache_path}");
            return Ok((matrices, sp1_stats));
        }
        eprintln!("  Cache digest mismatch, re-converting...");
    }
    
    // Convert from SP1 R1CS
    let (matrices, stats) = load_sp1_r1cs_as_symphony::<R, F>(sp1_path, None)?;
    
    // Save to cache
    if let Err(e) = save_symphony_matrices::<R>(&cache_path, &matrices, &stats.digest) {
        eprintln!("  Warning: failed to save cache: {e}");
    } else {
        eprintln!("  Saved cache: {cache_path}");
    }
    
    Ok((matrices, stats))
}
