//! SP1 witness file I/O helpers.
//!
//! Convention:
//! - `WITNESS_PATH` is a **bundle** that includes public inputs and the R1LF digest.

#![cfg(feature = "we_gate")]

use std::io::Read;

const SP1_WITNESS_BUNDLE_MAGIC: &[u8; 4] = b"SP1W";
const SP1_WITNESS_BUNDLE_VERSION: u32 = 1;
const SP1_WITNESS_BUNDLE_HEADER_LEN: usize = 112;

#[derive(Debug, Clone)]
pub struct Sp1WitnessBundle {
    pub witness: Vec<u64>,
    pub base_len: usize,
    pub aux_len: usize,
    pub public_inputs: ([u8; 32], [u8; 32]),
    pub r1lf_digest: [u8; 32],
}

/// Verify SP1's current shrink-verifier public-input layout in the exported witness:
/// - `witness[0] = 1`
/// - `num_public = 40`
/// - `witness[1..=8]` are the 8 BabyBear words of `sp1_vk_digest` (not checked here)
/// - `witness[1+8..1+8+32]` are the 32 bytes of `committed_values_digest` (checked here)
///
/// Rationale: this establishes that the SP1 R1LF export is "compliant to z" in the sense that
/// statement-defining public values occupy fixed witness indices (1..=num_public).
pub fn check_sp1_public_inputs_layout(
    bundle: &Sp1WitnessBundle,
    num_public: usize,
) -> Result<(), String> {
    if bundle.witness.is_empty() || bundle.witness[0] != 1 {
        return Err("SP1 witness must have w[0]=1".to_string());
    }
    if num_public != 40 {
        return Err(format!(
            "unexpected SP1 num_public={num_public} (expected 40 = 8 sp1_vk_digest words + 32 committed_value_digest bytes)"
        ));
    }
    if bundle.witness.len() < 1 + num_public {
        return Err(format!(
            "SP1 witness too short for public inputs: w_len={} need_at_least={}",
            bundle.witness.len(),
            1 + num_public
        ));
    }
    let (_vk_hash, committed_values_digest) = bundle.public_inputs;
    for i in 0..32 {
        let got = bundle.witness[1 + 8 + i];
        let exp = committed_values_digest[i] as u64;
        if got != exp {
            return Err(format!(
                "public input mismatch at idx={} (committed_value_digest[{}]): got={} expected={}",
                1 + 8 + i,
                i,
                got,
                exp
            ));
        }
    }
    Ok(())
}

/// Load a witness in either of these formats:
/// - **Single file only**: `witness_path` is the full witness of length `num_vars`.
///
/// Returns `(full_witness, base_len, aux_len)`.
pub fn load_sp1_witness_any(
    witness_path: &str,
    num_vars: usize,
) -> Result<Sp1WitnessBundle, String> {
    let mut bytes = Vec::new();
    std::fs::File::open(witness_path)
        .and_then(|mut f| f.read_to_end(&mut bytes))
        .map_err(|e| format!("read {witness_path}: {e}"))?;

    if bytes.len() >= SP1_WITNESS_BUNDLE_HEADER_LEN
        && bytes[0..4] == *SP1_WITNESS_BUNDLE_MAGIC
    {
        let version = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
        if version != SP1_WITNESS_BUNDLE_VERSION {
            return Err(format!(
                "unsupported witness bundle version {version} (expected {SP1_WITNESS_BUNDLE_VERSION})"
            ));
        }
        let mut r1lf_digest = [0u8; 32];
        r1lf_digest.copy_from_slice(&bytes[8..40]);
        let bundle_num_vars = u64::from_le_bytes(bytes[40..48].try_into().unwrap()) as usize;
        let mut vk_hash = [0u8; 32];
        vk_hash.copy_from_slice(&bytes[48..80]);
        let mut committed_values_digest = [0u8; 32];
        committed_values_digest.copy_from_slice(&bytes[80..112]);
        let witness_len = bundle_num_vars;
        if witness_len != num_vars {
            return Err(format!(
                "witness bundle num_vars mismatch: file has {}, expected {}",
                witness_len, num_vars
            ));
        }
        let expected_bytes = SP1_WITNESS_BUNDLE_HEADER_LEN + witness_len * 8;
        if bytes.len() != expected_bytes {
            return Err(format!(
                "witness bundle byte length mismatch: got {} expected {}",
                bytes.len(),
                expected_bytes
            ));
        }
        let mut witness = Vec::with_capacity(witness_len);
        for chunk in bytes[SP1_WITNESS_BUNDLE_HEADER_LEN..].chunks_exact(8) {
            witness.push(u64::from_le_bytes(chunk.try_into().unwrap()));
        }
        if witness.is_empty() || witness[0] != 1 {
            return Err("witness must have w[0]=1 (constant ONE slot)".to_string());
        }
        return Ok(Sp1WitnessBundle {
            witness,
            base_len: num_vars,
            aux_len: 0,
            public_inputs: (vk_hash, committed_values_digest),
            r1lf_digest,
        });
    }

    Err(format!(
        "unrecognized witness format: expected SP1 witness bundle. \
Set SP1_WITNESS when exporting from SP1."
    ))
}

