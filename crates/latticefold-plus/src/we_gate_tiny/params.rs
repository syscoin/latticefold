//! Parameters/constants shared across tiny-gate submodules.

/// Base for limb decomposition (7-bit limbs).
pub(super) const LIMB_BASE_U64: u64 = 128;
pub(super) const LIMB_BITS: usize = 7;

/// ceil(64/7) = 10.
pub(super) const LIMBS_U64: usize = 10;

/// Base-257 digits per challenge attempt (fixed by transcript design).
pub(super) const DIGITS_PER_TRY: usize = 8;

/// ceil(32/7) = 5.
pub(super) const LIMBS_U32: usize = 5;

