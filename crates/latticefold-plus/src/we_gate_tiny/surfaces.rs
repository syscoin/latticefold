use std::sync::Arc;

/// Wiring for a simple “first CM digit-mul surface” stage:
/// multiply one short-challenge block's coefficients by one bounded-u32 challenge using the digit backend.
#[derive(Clone, Debug)]
pub struct CmDigitMulSurfaceWiring {
    pub short_block_idx: usize,
    pub u32_idx: usize,
    /// Per coefficient product digits (len 12 each), in ring coefficient order.
    pub products: Vec<[usize; 12]>,
    /// Same products as `products`, but with the tail carry normalized so all digits are in `[-8,7]`
    /// plus one final carry digit in `[-2,2]`.
    pub products13: Vec<[usize; 13]>,
    /// Sum of all coefficient products as balanced base-16 digits (little-endian), fixed length 16.
    pub sum_digits: Vec<usize>,
    /// Accumulated sum across **all requested digit-mul surfaces** in the batch builder.
    pub sum_all_pairs_digits: Arc<Vec<usize>>,
    /// Coefficient-wise sum across **all requested digit-mul surfaces**.
    ///
    /// Length = `ring_dim`; each entry is a balanced base-16 digit vector (little-endian) of length 16.
    pub sum_all_pairs_coeffwise: Arc<Vec<Vec<usize>>>,
}

/// Like `CmDigitMulSurfaceWiring`, but multiplies a short-challenge block by **u32^2** (18 digits).
#[derive(Clone, Debug)]
pub struct CmDigitMulSqSurfaceWiring {
    pub short_block_idx: usize,
    pub u32_idx: usize,
    /// Per coefficient product digits (len 21 each), in ring coefficient order.
    pub products21: Vec<[usize; 21]>,
    /// Same products as `products21`, but normalized to 22 digits (tail carry split to `[-8,7]` + `[-2,2]`).
    pub products22: Vec<[usize; 22]>,
    /// Sum of all coefficient products (balanced base-16 digits, little-endian), fixed length 24.
    pub sum_digits: Vec<usize>,
    /// Sum across all requested sq-surfaces in the batch, fixed length 24.
    pub sum_all_pairs_digits: Arc<Vec<usize>>,
    /// Coefficient-wise sum across all requested sq-surfaces, length ring_dim, each fixed length 24.
    pub sum_all_pairs_coeffwise: Arc<Vec<Vec<usize>>>,
}

