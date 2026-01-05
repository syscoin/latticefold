//! # Symphony - Lattice-based folding SNARK
//!
//! Implementation of the Symphony protocol for post-quantum witness encryption.

#![allow(non_snake_case)]

// Core modules
pub mod setchk;
pub mod transcript;
pub mod utils;
pub mod public_coin_transcript;
pub mod recording_transcript;

// Symphony protocol modules
pub mod symphony_cm;
pub mod symphony_fold;
pub mod symphony_had;
pub mod symphony_gr1cs;
pub mod symphony_coins;
pub mod symphony_open;
pub mod symphony_pifold_batched;
pub mod symphony_we_relation;

// SP1 R1CS integration
pub mod sp1_r1cs_loader;
pub mod symphony_sp1_r1cs;

// Symphony Π_rg implementation
pub mod rp_rgchk;

// Streaming sumcheck (memory-efficient MLE handling)
pub mod streaming_sumcheck;
pub mod mle_oracle;
pub mod symphony_pifold_streaming;
