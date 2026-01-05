//! SP1 R1CS integration for Symphony.
//!
//! Loads pre-compiled R1CS from SP1's shrink verifier, chunks it, and converts to Symphony format.
//! Always uses chunking for memory efficiency.
//!
//! Main entry point: `load_sp1_r1cs_chunked_cached`

use crate::sp1_r1cs_loader::{SP1R1CS, FieldFromU64, SparseRow as LoaderSparseRow};
use ark_ff::PrimeField;
use stark_rings::{OverField, Zq};
use stark_rings_linalg::SparseMatrix;
use std::io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write};

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
const CHUNK_CACHE_VERSION: u32 = 2;

/// Random-access reader for a `.chunks` cache file (loads one chunk at a time).
pub struct ChunkCache<R> {
    pub stats: SP1R1CSStats,
    pub chunk_size: usize,
    pub ncols: usize,
    pub num_chunks: usize,
    cache_path: String,
    chunk_offsets: Vec<u64>, // absolute file offsets
    _phantom: std::marker::PhantomData<R>,
}

impl<R> ChunkCache<R>
where
    R: OverField,
    R::BaseRing: Zq + From<u64>,
{
    pub fn read_chunk(&self, chunk_idx: usize) -> std::io::Result<[SparseMatrix<R>; 3]> {
        if chunk_idx >= self.num_chunks {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "chunk_idx out of range",
            ));
        }
        let file = std::fs::File::open(&self.cache_path)?;
        let mut r = BufReader::with_capacity(IO_BUFFER_SIZE, file);
        r.seek(SeekFrom::Start(self.chunk_offsets[chunk_idx]))?;

        let mut buf4 = [0u8; 4];
        let mut buf8 = [0u8; 8];

        r.read_exact(&mut buf8)?;
        let nrows = u64::from_le_bytes(buf8) as usize;

        let mut chunk_matrices: [SparseMatrix<R>; 3] = std::array::from_fn(|_| SparseMatrix {
            nrows,
            ncols: self.ncols,
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
        Ok(chunk_matrices)
    }
}

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

    // Offset table (v2): filled after writing chunks.
    let num_chunks = chunked.chunks.len();
    let offsets_pos = w.stream_position()?;
    for _ in 0..num_chunks {
        w.write_all(&0u64.to_le_bytes())?;
    }
    w.flush()?;

    let mut offsets: Vec<u64> = vec![0u64; num_chunks];
    
    // Each chunk: nrows, then 3 matrices
    for (i, [ma, mb, mc]) in chunked.chunks.iter().enumerate() {
        offsets[i] = w.stream_position()?;
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

    // Backfill offsets.
    w.seek(SeekFrom::Start(offsets_pos))?;
    for off in offsets {
        w.write_all(&off.to_le_bytes())?;
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
    
    // Offset table (v2)
    let mut offsets = vec![0u64; num_chunks];
    for i in 0..num_chunks {
        r.read_exact(&mut buf8)?;
        offsets[i] = u64::from_le_bytes(buf8);
    }
    
    let mut chunks = Vec::with_capacity(num_chunks);
    
    for i in 0..num_chunks {
        r.seek(SeekFrom::Start(offsets[i]))?;
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

fn open_chunk_cache<R>(path: &str, expected_digest: &[u8; 32]) -> std::io::Result<ChunkCache<R>>
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
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "Cache version mismatch",
        ));
    }

    let mut digest = [0u8; 32];
    r.read_exact(&mut digest)?;
    if &digest != expected_digest {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "Cache digest mismatch",
        ));
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

    let mut offsets = vec![0u64; num_chunks];
    for i in 0..num_chunks {
        r.read_exact(&mut buf8)?;
        offsets[i] = u64::from_le_bytes(buf8);
    }

    Ok(ChunkCache {
        stats: SP1R1CSStats { num_vars, num_constraints, num_public, total_nonzeros, digest },
        chunk_size,
        ncols,
        num_chunks,
        cache_path: path.to_string(),
        chunk_offsets: offsets,
        _phantom: std::marker::PhantomData,
    })
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
    pad_cols_to_multiple_of: usize,
) -> std::io::Result<ChunkedMatrices<R>>
where
    R: OverField + Clone + Send + Sync,
    R::BaseRing: Zq + PrimeField + From<u64> + Send + Sync,
    F: FieldFromU64 + Clone + Send + Sync,
{
    // Read header to get digest
    let stats = read_sp1_r1cs_stats(path)?;
    let cache_path = format!("{}.chunks", path);
    
    if pad_cols_to_multiple_of == 0 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "pad_cols_to_multiple_of must be > 0",
        ));
    }

    // Try cache first.
    if let Ok(cached) = load_chunked_cache::<R>(&cache_path, &stats.digest) {
        if cached.chunk_size == chunk_size {
            eprintln!("  ✓ Loaded from cache: {cache_path}");
            eprintln!("    {} chunks of {} constraints", cached.chunks.len(), chunk_size);
            return Ok(cached);
        }
        eprintln!("  Cache chunk_size mismatch, re-converting...");
    }
    
    // Build cache with streaming conversion (avoid materializing converted rows for all constraints).
    eprintln!("  Loading SP1 R1CS: {path}");
    let r1cs: SP1R1CS<F> = SP1R1CS::load(path)?;
    
    // IMPORTANT: Π_rg requires `m % m_J == 0`, and with our chunking `m` is a power-of-two.
    // For `lambda_pj = 1`, this forces `blocks = ncols / l_h` to be a power-of-two divisor of `m`,
    // hence `ncols` must be `l_h * 2^k` (i.e. a power-of-two multiple of `l_h`).
    let blocks = (r1cs.num_vars + pad_cols_to_multiple_of - 1) / pad_cols_to_multiple_of;
    let blocks_pow2 = next_power_of_two(blocks);
    let ncols = blocks_pow2 * pad_cols_to_multiple_of;
    let num_chunks = (r1cs.num_constraints + chunk_size - 1) / chunk_size;
    
    eprintln!(
        "  Converting {} constraints → {} chunks of {} each",
        r1cs.num_constraints, num_chunks, chunk_size
    );
    eprintln!(
        "  Columns: {} → {} (padded to multiple of {})",
        r1cs.num_vars, ncols, pad_cols_to_multiple_of
    );

    fn convert_row<R, F>(row: &LoaderSparseRow<F>) -> Vec<(R, usize)>
    where
        R: OverField,
        R::BaseRing: Zq + From<u64>,
        F: FieldFromU64,
    {
        row.terms
            .iter()
                    .map(|(col_idx, coeff)| {
                        let val = R::from(R::BaseRing::from(coeff.as_canonical_u64()));
                        (val, *col_idx)
            })
            .collect()
    }
    
    let mut chunks: Vec<[SparseMatrix<R>; 3]> = Vec::with_capacity(num_chunks);
    for i in 0..num_chunks {
        let start = i * chunk_size;
        let end = std::cmp::min(start + chunk_size, r1cs.num_constraints);
        let actual_rows = end - start;
        let padded_rows = next_power_of_two(actual_rows);
        
        let make = |rows: &[LoaderSparseRow<F>]| -> SparseMatrix<R> {
            let mut coeffs: Vec<Vec<(R, usize)>> = (start..end)
                .map(|r| convert_row::<R, F>(&rows[r]))
                .collect();
            coeffs.resize_with(padded_rows, Vec::new);
            SparseMatrix { nrows: padded_rows, ncols, coeffs }
        };
        
        chunks.push([make(&r1cs.a), make(&r1cs.b), make(&r1cs.c)]);
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
    
    eprintln!("  Saving cache: {cache_path}");
    if let Err(e) = save_chunked_cache(&cache_path, &result) {
        eprintln!("  Warning: failed to save cache: {e}");
    }
    
    Ok(result)
}

/// Open (or build) the `.chunks` cache and return a random-access reader for chunk streaming.
pub fn open_sp1_r1cs_chunk_cache<R, F>(
    path: &str,
    chunk_size: usize,
    pad_cols_to_multiple_of: usize,
) -> std::io::Result<ChunkCache<R>>
where
    R: OverField + Clone + Send + Sync,
    R::BaseRing: Zq + PrimeField + From<u64> + Send + Sync,
    F: FieldFromU64 + Clone + Send + Sync,
{
    let stats = read_sp1_r1cs_stats(path)?;
    let cache_path = format!("{}.chunks", path);
    if pad_cols_to_multiple_of == 0 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "pad_cols_to_multiple_of must be > 0",
        ));
    }
    let blocks = (stats.num_vars + pad_cols_to_multiple_of - 1) / pad_cols_to_multiple_of;
    let blocks_pow2 = next_power_of_two(blocks);
    let expected_ncols = blocks_pow2 * pad_cols_to_multiple_of;

    if let Ok(cache) = open_chunk_cache::<R>(&cache_path, &stats.digest) {
        if cache.chunk_size == chunk_size && cache.ncols == expected_ncols {
            return Ok(cache);
        }
    }

    // Build cache with streaming conversion: convert per-chunk and write, avoiding giant
    // `rows_a/rows_b/rows_c` allocations.
    let r1cs: SP1R1CS<F> = SP1R1CS::load(path)?;
    let blocks = (r1cs.num_vars + pad_cols_to_multiple_of - 1) / pad_cols_to_multiple_of;
    let blocks_pow2 = next_power_of_two(blocks);
    let ncols = blocks_pow2 * pad_cols_to_multiple_of;
    let num_chunks = (r1cs.num_constraints + chunk_size - 1) / chunk_size;

    let total_nonzeros = (r1cs.a.iter().map(|r| r.terms.len()).sum::<usize>()
        + r1cs.b.iter().map(|r| r.terms.len()).sum::<usize>()
        + r1cs.c.iter().map(|r| r.terms.len()).sum::<usize>()) as u64;

    let chunked = ChunkedMatrices::<R> {
        chunks: (0..num_chunks)
            .map(|i| {
                let start = i * chunk_size;
                let end = std::cmp::min(start + chunk_size, r1cs.num_constraints);
                let actual_rows = end - start;
                let padded_rows = next_power_of_two(actual_rows);

                fn convert_row<R, F>(row: &LoaderSparseRow<F>) -> Vec<(R, usize)>
                where
                    R: OverField,
                    R::BaseRing: Zq + From<u64>,
                    F: FieldFromU64,
                {
                    row.terms
                        .iter()
                        .map(|(col_idx, coeff)| {
                            let val = R::from(R::BaseRing::from(coeff.as_canonical_u64()));
                            (val, *col_idx)
                        })
                        .collect()
                }

                let make = |rows: &[LoaderSparseRow<F>]| -> SparseMatrix<R> {
                    let mut coeffs: Vec<Vec<(R, usize)>> = (start..end)
                        .map(|r| convert_row::<R, F>(&rows[r]))
                        .collect();
                    coeffs.resize_with(padded_rows, Vec::new);
                    SparseMatrix { nrows: padded_rows, ncols, coeffs }
                };

                [make(&r1cs.a), make(&r1cs.b), make(&r1cs.c)]
            })
            .collect(),
        stats: SP1R1CSStats {
            num_vars: r1cs.num_vars,
            num_constraints: r1cs.num_constraints,
            num_public: r1cs.num_public,
            total_nonzeros,
            digest: stats.digest,
        },
        chunk_size,
        ncols,
    };

    let _ = save_chunked_cache(&cache_path, &chunked);
    open_chunk_cache::<R>(&cache_path, &stats.digest)
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
