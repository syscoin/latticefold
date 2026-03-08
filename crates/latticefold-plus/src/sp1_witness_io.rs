//! SP1 witness file I/O helpers.
//!
//! Convention:
//! - `WITNESS_PATH` is a **bundle** that includes public inputs and the R1LF digest.

#![cfg(feature = "we_gate")]

use std::io::{Read, Seek, SeekFrom};

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

#[derive(Debug, Clone)]
pub struct Sp1WitnessPrefix {
    pub witness_prefix: Vec<u64>,
    pub public_inputs: ([u8; 32], [u8; 32]),
    pub r1lf_digest: [u8; 32],
}

pub fn load_sp1_witness_prefix(
    witness_path: &str,
    num_vars: usize,
    prefix_len: usize,
) -> Result<Sp1WitnessPrefix, String> {
    let mut f = std::fs::File::open(witness_path)
        .map_err(|e| format!("open {witness_path}: {e}"))?;
    let mut header = [0u8; SP1_WITNESS_BUNDLE_HEADER_LEN];
    f.read_exact(&mut header)
        .map_err(|e| format!("read witness header {witness_path}: {e}"))?;
    if header[0..4] != *SP1_WITNESS_BUNDLE_MAGIC {
        return Err("unrecognized witness format: expected SP1 witness bundle".to_string());
    }
    let version = u32::from_le_bytes(header[4..8].try_into().unwrap());
    if version != SP1_WITNESS_BUNDLE_VERSION {
        return Err(format!(
            "unsupported witness bundle version {version} (expected {SP1_WITNESS_BUNDLE_VERSION})"
        ));
    }
    let mut r1lf_digest = [0u8; 32];
    r1lf_digest.copy_from_slice(&header[8..40]);
    let bundle_num_vars = u64::from_le_bytes(header[40..48].try_into().unwrap()) as usize;
    if bundle_num_vars != num_vars {
        return Err(format!(
            "witness bundle num_vars mismatch: file has {}, expected {}",
            bundle_num_vars, num_vars
        ));
    }
    if prefix_len > bundle_num_vars {
        return Err(format!(
            "requested witness prefix length {} exceeds bundle length {}",
            prefix_len, bundle_num_vars
        ));
    }
    let mut vk_hash = [0u8; 32];
    vk_hash.copy_from_slice(&header[48..80]);
    let mut committed_values_digest = [0u8; 32];
    committed_values_digest.copy_from_slice(&header[80..112]);
    let expected_bytes = SP1_WITNESS_BUNDLE_HEADER_LEN + bundle_num_vars * 8;
    let file_len = f
        .seek(SeekFrom::End(0))
        .map_err(|e| format!("seek witness bundle {witness_path}: {e}"))? as usize;
    if file_len != expected_bytes {
        return Err(format!(
            "witness bundle byte length mismatch: got {} expected {}",
            file_len, expected_bytes
        ));
    }
    f.seek(SeekFrom::Start(SP1_WITNESS_BUNDLE_HEADER_LEN as u64))
        .map_err(|e| format!("seek witness body {witness_path}: {e}"))?;
    let mut bytes = vec![0u8; prefix_len * 8];
    f.read_exact(bytes.as_mut_slice())
        .map_err(|e| format!("read witness prefix {witness_path}: {e}"))?;
    let mut witness_prefix = Vec::with_capacity(prefix_len);
    for chunk in bytes.chunks_exact(8) {
        witness_prefix.push(u64::from_le_bytes(chunk.try_into().unwrap()));
    }
    if witness_prefix.is_empty() || witness_prefix[0] != 1 {
        return Err("witness must have w[0]=1 (constant ONE slot)".to_string());
    }
    Ok(Sp1WitnessPrefix {
        witness_prefix,
        public_inputs: (vk_hash, committed_values_digest),
        r1lf_digest,
    })
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

