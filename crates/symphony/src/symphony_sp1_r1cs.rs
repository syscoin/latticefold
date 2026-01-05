//! SP1 R1CS integration for Symphony.
//!
//! Loads pre-compiled R1CS from SP1's shrink verifier and converts to Symphony format.

use crate::sp1_r1cs_loader::{SP1R1CS, FieldFromU64, SparseRow as LoaderSparseRow};
use stark_rings::{OverField, Zq};
use stark_rings_linalg::SparseMatrix;

/// Convert SP1 R1CS to Symphony sparse matrices [A, B, C].
/// 
/// The R1CS constraint format is: (A·w) ⊙ (B·w) = (C·w)
/// where ⊙ is element-wise multiplication.
pub fn sp1_r1cs_to_symphony_matrices<R, F>(
    r1cs: &SP1R1CS<F>,
) -> [SparseMatrix<R>; 3]
where
    R: OverField,
    R::BaseRing: Zq + From<u64>,
    F: FieldFromU64 + Clone,
{
    let nrows = r1cs.num_constraints;
    let ncols = r1cs.num_vars;
    
    fn convert_matrix<R, F>(rows: &[LoaderSparseRow<F>], nrows: usize, ncols: usize) -> SparseMatrix<R>
    where
        R: OverField,
        R::BaseRing: Zq + From<u64>,
        F: FieldFromU64 + Clone,
    {
        // SparseMatrix stores coeffs as Vec<Vec<(value, col_idx)>> per row
        let mut coeffs: Vec<Vec<(R, usize)>> = vec![vec![]; nrows];
        
        for (row_idx, row) in rows.iter().enumerate() {
            for (col_idx, coeff) in &row.terms {
                let val = R::from(R::BaseRing::from(coeff.as_canonical_u64()));
                coeffs[row_idx].push((val, *col_idx));
            }
        }
        
        SparseMatrix { nrows, ncols, coeffs }
    }
    
    [
        convert_matrix(&r1cs.a, nrows, ncols),
        convert_matrix(&r1cs.b, nrows, ncols),
        convert_matrix(&r1cs.c, nrows, ncols),
    ]
}

/// Load SP1 shrink verifier R1CS from file and convert to Symphony format.
pub fn load_sp1_r1cs_as_symphony<R, F>(
    path: &str,
    expected_digest: Option<&[u8; 32]>,
) -> std::io::Result<(SP1R1CS<F>, [SparseMatrix<R>; 3])>
where
    R: OverField,
    R::BaseRing: Zq + From<u64>,
    F: FieldFromU64 + Clone,
{
    let r1cs = if let Some(digest) = expected_digest {
        SP1R1CS::load_verified(path, digest)?
    } else {
        SP1R1CS::load(path)?
    };
    
    let matrices = sp1_r1cs_to_symphony_matrices(&r1cs);
    Ok((r1cs, matrices))
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
    use std::io::Read;
    
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
