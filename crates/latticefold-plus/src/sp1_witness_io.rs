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

#[derive(Debug, Clone)]
pub struct Sp1WitnessBundleHeader {
    pub r1lf_digest: [u8; 32],
    pub num_vars: usize,
    pub public_inputs: ([u8; 32], [u8; 32]),
}

/// Load only the witness bundle header plus a prefix of witness words.
///
/// This is intended for *arming*: the armer needs the statement-bound public inputs (and bundle
/// metadata) but should not need to materialize the full SP1 witness vector.
///
/// Returns `(header, witness_prefix_u64)`.
pub fn load_sp1_witness_bundle_header_and_prefix(
    witness_path: &str,
    expected_num_vars: usize,
    prefix_len: usize,
) -> Result<(Sp1WitnessBundleHeader, Vec<u64>), String> {
    use std::io::{Read, Seek, SeekFrom};

    let mut f = std::fs::File::open(witness_path).map_err(|e| format!("open {witness_path}: {e}"))?;

    let meta_len = std::fs::metadata(witness_path)
        .map_err(|e| format!("metadata {witness_path}: {e}"))?
        .len() as usize;

    let mut hdr = [0u8; SP1_WITNESS_BUNDLE_HEADER_LEN];
    f.read_exact(&mut hdr)
        .map_err(|e| format!("read witness bundle header {witness_path}: {e}"))?;

    if hdr[0..4] != *SP1_WITNESS_BUNDLE_MAGIC {
        return Err(
            "unrecognized witness format: expected SP1 witness bundle. Set SP1_WITNESS when exporting from SP1."
                .to_string(),
        );
    }

    let version = u32::from_le_bytes(hdr[4..8].try_into().unwrap());
    if version != SP1_WITNESS_BUNDLE_VERSION {
        return Err(format!(
            "unsupported witness bundle version {version} (expected {SP1_WITNESS_BUNDLE_VERSION})"
        ));
    }

    let mut r1lf_digest = [0u8; 32];
    r1lf_digest.copy_from_slice(&hdr[8..40]);
    let bundle_num_vars = u64::from_le_bytes(hdr[40..48].try_into().unwrap()) as usize;
    let mut vk_hash = [0u8; 32];
    vk_hash.copy_from_slice(&hdr[48..80]);
    let mut committed_values_digest = [0u8; 32];
    committed_values_digest.copy_from_slice(&hdr[80..112]);

    if bundle_num_vars != expected_num_vars {
        return Err(format!(
            "witness bundle num_vars mismatch: file has {}, expected {}",
            bundle_num_vars, expected_num_vars
        ));
    }

    let expected_bytes = SP1_WITNESS_BUNDLE_HEADER_LEN + bundle_num_vars * 8;
    if meta_len != expected_bytes {
        return Err(format!(
            "witness bundle byte length mismatch: got {} expected {}",
            meta_len, expected_bytes
        ));
    }

    // Read witness prefix.
    let prefix_len = prefix_len.min(bundle_num_vars);
    let mut prefix_bytes = vec![0u8; prefix_len * 8];
    if prefix_len > 0 {
        // Ensure we are positioned right after the header.
        f.seek(SeekFrom::Start(SP1_WITNESS_BUNDLE_HEADER_LEN as u64))
            .map_err(|e| format!("seek witness bundle {witness_path}: {e}"))?;
        f.read_exact(&mut prefix_bytes)
            .map_err(|e| format!("read witness prefix {witness_path}: {e}"))?;
    }

    let mut witness_prefix = Vec::with_capacity(prefix_len);
    for chunk in prefix_bytes.chunks_exact(8) {
        witness_prefix.push(u64::from_le_bytes(chunk.try_into().unwrap()));
    }

    if !witness_prefix.is_empty() && witness_prefix[0] != 1 {
        return Err("witness must have w[0]=1 (constant ONE slot)".to_string());
    }

    Ok((
        Sp1WitnessBundleHeader {
            r1lf_digest,
            num_vars: bundle_num_vars,
            public_inputs: (vk_hash, committed_values_digest),
        },
        witness_prefix,
    ))
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

