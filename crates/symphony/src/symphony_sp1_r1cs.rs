//! SP1 R1CS integration for Symphony.
//!
//! Loads pre-compiled R1CS from SP1's shrink verifier, chunks it, and converts to Symphony format.
//! Always uses chunking for memory efficiency.
//!
//! Main entry point: `load_sp1_r1cs_chunked_cached`

use crate::sp1_r1cs_loader::{SP1R1CS, FieldFromU64, SparseRow as LoaderSparseRow};
use ark_ff::PrimeField;
use rayon::prelude::*;
use stark_rings::{OverField, Zq};
use stark_rings_linalg::SparseMatrix;
use std::io::{Read, Write, BufReader, BufWriter};

/// Buffer size for I/O (256 MB)
const IO_BUFFER_SIZE: usize = 256 * 1024 * 1024;

/// Round up to next power of 2.
pub fn next_power_of_two(n: usize) -> usize {
    if n == 0 { return 1; }
    1usize << (usize::BITS - (n - 1).leading_zeros())
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

/// Chunked matrices - one chunk = (A_chunk, B_chunk, C_chunk)
pub struct ChunkedMatrices<R> {
    pub chunks: Vec<[SparseMatrix<R>; 3]>,
    pub stats: SP1R1CSStats,
    pub chunk_size: usize,
    pub ncols: usize, // Padded column count (same for all chunks)
}

// ============================================================================
// Symphony chunk cache format
// ============================================================================

const CHUNK_CACHE_MAGIC: &[u8; 4] = b"SYMC"; // Symphony Chunk
const CHUNK_CACHE_VERSION: u32 = 1;

/// Save chunked matrices to cache file.
fn save_chunked_cache<R>(
    path: &str,
    chunked: &ChunkedMatrices<R>,
) -> std::io::Result<()>
where
    R: OverField,
    R::BaseRing: Zq + PrimeField,
{
    let file = std::fs::File::create(path)?;
    let mut w = BufWriter::with_capacity(IO_BUFFER_SIZE, file);
    
    // Header
    w.write_all(CHUNK_CACHE_MAGIC)?;
    w.write_all(&CHUNK_CACHE_VERSION.to_le_bytes())?;
    w.write_all(&chunked.stats.digest)?;
    w.write_all(&(chunked.stats.num_vars as u64).to_le_bytes())?;
    w.write_all(&(chunked.stats.num_constraints as u64).to_le_bytes())?;
    w.write_all(&(chunked.stats.num_public as u64).to_le_bytes())?;
    w.write_all(&chunked.stats.total_nonzeros.to_le_bytes())?;
    w.write_all(&(chunked.chunk_size as u64).to_le_bytes())?;
    w.write_all(&(chunked.ncols as u64).to_le_bytes())?;
    w.write_all(&(chunked.chunks.len() as u64).to_le_bytes())?;
    
    // Each chunk: nrows, then 3 matrices
    for [ma, mb, mc] in &chunked.chunks {
        w.write_all(&(ma.nrows as u64).to_le_bytes())?;
        
        for matrix in [ma, mb, mc] {
            for row in &matrix.coeffs {
                w.write_all(&(row.len() as u32).to_le_bytes())?;
                for (val, col_idx) in row {
                    w.write_all(&(*col_idx as u32).to_le_bytes())?;
                    // Use PrimeField::into_bigint() to get underlying value as u64
                    let coeff_u64: u64 = val.coeffs()[0].into_bigint().as_ref()[0];
                    w.write_all(&coeff_u64.to_le_bytes())?;
                }
            }
        }
    }
    
    w.flush()?;
    Ok(())
}

/// Load chunked matrices from cache file.
fn load_chunked_cache<R>(
    path: &str,
    expected_digest: &[u8; 32],
) -> std::io::Result<ChunkedMatrices<R>>
where
    R: OverField,
    R::BaseRing: Zq + From<u64>,
{
    let file = std::fs::File::open(path)?;
    let mut r = BufReader::with_capacity(IO_BUFFER_SIZE, file);
    
    let mut magic = [0u8; 4];
    r.read_exact(&mut magic)?;
    if &magic != CHUNK_CACHE_MAGIC {
        return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "Invalid cache magic"));
    }
    
    let mut buf4 = [0u8; 4];
    let mut buf8 = [0u8; 8];
    
    r.read_exact(&mut buf4)?;
    let version = u32::from_le_bytes(buf4);
    if version != CHUNK_CACHE_VERSION {
        return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "Cache version mismatch"));
    }
    
    let mut digest = [0u8; 32];
    r.read_exact(&mut digest)?;
    if &digest != expected_digest {
        return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "Cache digest mismatch"));
    }
    
    r.read_exact(&mut buf8)?;
    let num_vars = u64::from_le_bytes(buf8) as usize;
    r.read_exact(&mut buf8)?;
    let num_constraints = u64::from_le_bytes(buf8) as usize;
    r.read_exact(&mut buf8)?;
    let num_public = u64::from_le_bytes(buf8) as usize;
    r.read_exact(&mut buf8)?;
    let total_nonzeros = u64::from_le_bytes(buf8);
    r.read_exact(&mut buf8)?;
    let chunk_size = u64::from_le_bytes(buf8) as usize;
    r.read_exact(&mut buf8)?;
    let ncols = u64::from_le_bytes(buf8) as usize;
    r.read_exact(&mut buf8)?;
    let num_chunks = u64::from_le_bytes(buf8) as usize;
    
    let stats = SP1R1CSStats { num_vars, num_constraints, num_public, total_nonzeros, digest };
    
    let mut chunks = Vec::with_capacity(num_chunks);
    
    for _ in 0..num_chunks {
        r.read_exact(&mut buf8)?;
        let nrows = u64::from_le_bytes(buf8) as usize;
        
        let mut chunk_matrices: [SparseMatrix<R>; 3] = std::array::from_fn(|_| SparseMatrix {
            nrows,
            ncols,
            coeffs: Vec::with_capacity(nrows),
        });
        
        for matrix in &mut chunk_matrices {
            for _ in 0..nrows {
                r.read_exact(&mut buf4)?;
                let num_terms = u32::from_le_bytes(buf4) as usize;
                
                let mut row = Vec::with_capacity(num_terms);
                for _ in 0..num_terms {
                    r.read_exact(&mut buf4)?;
                    let col_idx = u32::from_le_bytes(buf4) as usize;
                    r.read_exact(&mut buf8)?;
                    let coeff = u64::from_le_bytes(buf8);
                    let val = R::from(R::BaseRing::from(coeff));
                    row.push((val, col_idx));
                }
                matrix.coeffs.push(row);
            }
        }
        
        chunks.push(chunk_matrices);
    }
    
    Ok(ChunkedMatrices { chunks, stats, chunk_size, ncols })
}

// ============================================================================
// Main entry point
// ============================================================================

/// Load SP1 R1CS, chunk it, and convert to Symphony format.
/// 
/// Uses cache file (`{path}.chunks`) if available and digest matches.
/// 
/// # Arguments
/// * `path` - Path to SP1 R1CS file
/// * `chunk_size` - Constraints per chunk (e.g., 1<<20 = 1M)
/// 
/// # Returns
/// Chunked matrices ready for parallel proving with limited concurrency.
pub fn load_sp1_r1cs_chunked_cached<R, F>(
    path: &str,
    chunk_size: usize,
) -> std::io::Result<ChunkedMatrices<R>>
where
    R: OverField + Clone + Send + Sync,
    R::BaseRing: Zq + PrimeField + From<u64> + Send + Sync,
    F: FieldFromU64 + Clone + Send + Sync,
{
    // Read header to get digest
    let stats = read_sp1_r1cs_stats(path)?;
    let cache_path = format!("{}.chunks", path);
    
    // Try cache first
    if let Ok(cached) = load_chunked_cache::<R>(&cache_path, &stats.digest) {
        if cached.chunk_size == chunk_size {
            eprintln!("  ✓ Loaded from cache: {cache_path}");
            eprintln!("    {} chunks of {} constraints", cached.chunks.len(), chunk_size);
            return Ok(cached);
        }
        eprintln!("  Cache chunk_size mismatch ({} vs {}), re-converting...", 
            cached.chunk_size, chunk_size);
    }
    
    // Load and convert
    eprintln!("  Loading SP1 R1CS: {path}");
    let r1cs: SP1R1CS<F> = SP1R1CS::load(path)?;
    
    let ncols = next_power_of_two(r1cs.num_vars);
    let num_chunks = (r1cs.num_constraints + chunk_size - 1) / chunk_size;
    
    eprintln!("  Converting {} constraints → {} chunks of {} each", 
        r1cs.num_constraints, num_chunks, chunk_size);
    eprintln!("  Columns: {} → {} (padded to 2^{})", 
        r1cs.num_vars, ncols, ncols.trailing_zeros());
    
    // Convert rows to Symphony format (parallel)
    fn convert_rows<R, F>(rows: &[LoaderSparseRow<F>]) -> Vec<Vec<(R, usize)>>
    where
        R: OverField + Send + Sync,
        R::BaseRing: Zq + From<u64> + Send + Sync,
        F: FieldFromU64 + Clone + Send + Sync,
    {
        rows.par_iter()
            .map(|row| {
                row.terms.iter()
                    .map(|(col_idx, coeff)| {
                        let val = R::from(R::BaseRing::from(coeff.as_canonical_u64()));
                        (val, *col_idx)
                    })
                    .collect()
            })
            .collect()
    }
    
    let (rows_a, (rows_b, rows_c)) = rayon::join(
        || convert_rows::<R, F>(&r1cs.a),
        || rayon::join(
            || convert_rows::<R, F>(&r1cs.b),
            || convert_rows::<R, F>(&r1cs.c),
        ),
    );
    
    // Chunk the rows
    let mut chunks = Vec::with_capacity(num_chunks);
    for i in 0..num_chunks {
        let start = i * chunk_size;
        let end = std::cmp::min(start + chunk_size, r1cs.num_constraints);
        let actual_rows = end - start;
        let padded_rows = next_power_of_two(actual_rows);
        
        let make_chunk_matrix = |rows: &[Vec<(R, usize)>]| {
            let mut coeffs: Vec<Vec<(R, usize)>> = rows[start..end].to_vec();
            coeffs.resize_with(padded_rows, Vec::new);
            SparseMatrix { nrows: padded_rows, ncols, coeffs }
        };
        
        chunks.push([
            make_chunk_matrix(&rows_a),
            make_chunk_matrix(&rows_b),
            make_chunk_matrix(&rows_c),
        ]);
    }
    
    let result = ChunkedMatrices {
        chunks,
        stats: SP1R1CSStats {
            num_vars: r1cs.num_vars,
            num_constraints: r1cs.num_constraints,
            num_public: r1cs.num_public,
            total_nonzeros: (r1cs.a.iter().map(|r| r.terms.len()).sum::<usize>()
                + r1cs.b.iter().map(|r| r.terms.len()).sum::<usize>()
                + r1cs.c.iter().map(|r| r.terms.len()).sum::<usize>()) as u64,
            digest: stats.digest,
        },
        chunk_size,
        ncols,
    };
    
    // Save to cache
    eprintln!("  Saving cache: {cache_path}");
    if let Err(e) = save_chunked_cache(&cache_path, &result) {
        eprintln!("  Warning: failed to save cache: {e}");
    }
    
    Ok(result)
}

/// Read just the header/stats from an R1CS file (fast, reads only 72 bytes).
pub fn read_sp1_r1cs_stats(path: &str) -> std::io::Result<SP1R1CSStats> {
    let mut file = std::fs::File::open(path)?;
    let mut header_bytes = [0u8; 72];
    file.read_exact(&mut header_bytes)?;
    
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
