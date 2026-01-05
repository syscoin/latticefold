//! SP1 R1CS file loader for Symphony.
//! 
//! Loads pre-compiled R1CS from SP1's shrink verifier circuit.
//! No SP1 dependencies required - just sha2 for digest verification.
//!
//! File format (v2):
//! ```text
//! HEADER (72 bytes):
//!   Magic: "R1CS" (4 bytes)
//!   Version: 2 (4 bytes)
//!   Digest: SHA256 (32 bytes)
//!   num_vars (8 bytes)
//!   num_constraints (8 bytes)
//!   num_public (8 bytes)
//!   total_nonzeros (8 bytes)
//! BODY:
//!   A, B, C sparse matrices
//! ```

use sha2::{Sha256, Digest};
use std::io::Read;

/// A sparse row in an R1CS matrix.
#[derive(Debug, Clone, Default)]
pub struct SparseRow<F> {
    pub terms: Vec<(usize, F)>,
}

impl<F> SparseRow<F> {
    pub fn new() -> Self {
        Self { terms: Vec::new() }
    }
}

/// R1CS instance: (A·w) ⊙ (B·w) = (C·w)
#[derive(Debug, Clone)]
pub struct SP1R1CS<F> {
    pub num_vars: usize,
    pub num_constraints: usize,
    pub num_public: usize,
    pub a: Vec<SparseRow<F>>,
    pub b: Vec<SparseRow<F>>,
    pub c: Vec<SparseRow<F>>,
}

/// Trait for field elements loadable from u64.
pub trait FieldFromU64: Sized {
    fn from_canonical_u64(val: u64) -> Self;
    fn as_canonical_u64(&self) -> u64;
}

impl<F: FieldFromU64 + Clone> SP1R1CS<F> {
    /// Read header only (fast check without loading matrices).
    pub fn read_header(data: &[u8]) -> Result<SP1R1CSHeader, &'static str> {
        if data.len() < 72 {
            return Err("R1CS file too small");
        }
        if &data[0..4] != b"R1CS" {
            return Err("Invalid R1CS magic");
        }
        let version = u32::from_le_bytes(data[4..8].try_into().unwrap());
        if version != 2 {
            return Err("Unsupported R1CS version");
        }
        
        let mut digest = [0u8; 32];
        digest.copy_from_slice(&data[8..40]);
        
        Ok(SP1R1CSHeader {
            digest,
            num_vars: u64::from_le_bytes(data[40..48].try_into().unwrap()) as usize,
            num_constraints: u64::from_le_bytes(data[48..56].try_into().unwrap()) as usize,
            num_public: u64::from_le_bytes(data[56..64].try_into().unwrap()) as usize,
            total_nonzeros: u64::from_le_bytes(data[64..72].try_into().unwrap()),
        })
    }
    
    /// Compute digest of loaded R1CS (for verification).
    pub fn compute_digest(&self) -> [u8; 32] {
        let mut hasher = Sha256::new();
        
        hasher.update(b"R1CS_DIGEST_v1");
        hasher.update(&(self.num_vars as u64).to_le_bytes());
        hasher.update(&(self.num_constraints as u64).to_le_bytes());
        hasher.update(&(self.num_public as u64).to_le_bytes());
        
        for (tag, matrix) in [("A_MATRIX", &self.a), ("B_MATRIX", &self.b), ("C_MATRIX", &self.c)] {
            hasher.update(tag.as_bytes());
            for row in matrix.iter() {
                hasher.update(&(row.terms.len() as u64).to_le_bytes());
                for (idx, coeff) in &row.terms {
                    hasher.update(&(*idx as u64).to_le_bytes());
                    hasher.update(&coeff.as_canonical_u64().to_le_bytes());
                }
            }
        }
        
        hasher.finalize().into()
    }
    
    /// Load from bytes with integrity verification.
    pub fn from_bytes(data: &[u8]) -> Result<Self, &'static str> {
        let header = Self::read_header(data)?;
        let mut pos = 72;
        
        let mut a = Vec::with_capacity(header.num_constraints);
        let mut b = Vec::with_capacity(header.num_constraints);
        let mut c = Vec::with_capacity(header.num_constraints);
        
        for matrix in [&mut a, &mut b, &mut c] {
            for _ in 0..header.num_constraints {
                if pos + 4 > data.len() {
                    return Err("Unexpected end of data");
                }
                let num_terms = u32::from_le_bytes(data[pos..pos+4].try_into().unwrap()) as usize;
                pos += 4;
                
                let mut terms = Vec::with_capacity(num_terms);
                for _ in 0..num_terms {
                    if pos + 12 > data.len() {
                        return Err("Unexpected end of data");
                    }
                    let idx = u32::from_le_bytes(data[pos..pos+4].try_into().unwrap()) as usize;
                    pos += 4;
                    let coeff = F::from_canonical_u64(u64::from_le_bytes(data[pos..pos+8].try_into().unwrap()));
                    pos += 8;
                    terms.push((idx, coeff));
                }
                matrix.push(SparseRow { terms });
            }
        }
        
        let r1cs = Self { 
            num_vars: header.num_vars, 
            num_constraints: header.num_constraints, 
            num_public: header.num_public, 
            a, b, c 
        };
        
        // Verify integrity
        if r1cs.compute_digest() != header.digest {
            return Err("R1CS digest mismatch - file corrupted");
        }
        
        Ok(r1cs)
    }
    
    /// Load from file, verifying against expected digest.
    pub fn load_verified(path: &str, expected_digest: &[u8; 32]) -> std::io::Result<Self> {
        let mut file = std::fs::File::open(path)?;
        let mut header_buf = [0u8; 72];
        file.read_exact(&mut header_buf)?;
        
        let header = Self::read_header(&header_buf)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        
        if &header.digest != expected_digest {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("Digest mismatch: expected {:02x?}..., got {:02x?}...", 
                    &expected_digest[..8], &header.digest[..8])
            ));
        }
        
        // Full load
        let bytes = std::fs::read(path)?;
        Self::from_bytes(&bytes)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
    }
    
    /// Load from file (verifies internal consistency only).
    pub fn load(path: &str) -> std::io::Result<Self> {
        let bytes = std::fs::read(path)?;
        Self::from_bytes(&bytes)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
    }
}

/// R1CS file header (for quick inspection).
#[derive(Debug, Clone)]
pub struct SP1R1CSHeader {
    pub digest: [u8; 32],
    pub num_vars: usize,
    pub num_constraints: usize,
    pub num_public: usize,
    pub total_nonzeros: u64,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[derive(Debug, Clone, Copy, Default)]
    struct TestField(u64);
    
    impl FieldFromU64 for TestField {
        fn from_canonical_u64(val: u64) -> Self { TestField(val) }
        fn as_canonical_u64(&self) -> u64 { self.0 }
    }
    
    #[test]
    fn test_header_parsing() {
        let mut header = vec![0u8; 72];
        header[0..4].copy_from_slice(b"R1CS");
        header[4..8].copy_from_slice(&2u32.to_le_bytes());
        header[40..48].copy_from_slice(&100u64.to_le_bytes());
        header[48..56].copy_from_slice(&50u64.to_le_bytes());
        header[56..64].copy_from_slice(&5u64.to_le_bytes());
        header[64..72].copy_from_slice(&1000u64.to_le_bytes());
        
        let h = SP1R1CS::<TestField>::read_header(&header).unwrap();
        assert_eq!(h.num_vars, 100);
        assert_eq!(h.num_constraints, 50);
        assert_eq!(h.num_public, 5);
        assert_eq!(h.total_nonzeros, 1000);
    }
}
