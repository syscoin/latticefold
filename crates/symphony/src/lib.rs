//! # LatticeFold+

#![allow(non_snake_case)]

// Core modules
pub mod setchk;
pub mod transcript;
pub mod utils;
pub mod public_coin_transcript;
pub mod recording_transcript;

// Symphony protocol modules
#[cfg(feature = "symphony")]
pub mod symphony_cm;
#[cfg(feature = "symphony")]
pub mod symphony_fold;
#[cfg(feature = "symphony")]
pub mod symphony_had;
#[cfg(feature = "symphony")]
pub mod symphony_gr1cs;
#[cfg(feature = "symphony")]
pub mod symphony_coins;
#[cfg(feature = "symphony")]
pub mod symphony_open;
#[cfg(feature = "symphony")]
pub mod symphony_pifold_batched;
#[cfg(feature = "symphony")]
pub mod symphony_we_relation;

// SP1 R1CS integration
#[cfg(feature = "symphony")]
pub mod sp1_r1cs_loader;
#[cfg(feature = "symphony")]
pub mod symphony_sp1_r1cs;

// Symphony Π_rg implementation
#[cfg(feature = "symphony")]
pub mod rp_rgchk;
