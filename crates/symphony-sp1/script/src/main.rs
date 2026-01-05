use sp1_sdk::{include_elf, utils, ProverClient, SP1Stdin};

/// Guest program ELF (built by `build.rs` via `sp1_build::build_program`).
const SYMPHONY_CP_ELF: &[u8] = include_elf!("symphony-sp1-program");

fn main() {
    utils::setup_logger();

    // Minimal end-to-end scaffolding: the guest reads a blob and commits its length.
    // Next step: feed the Symphony Π_fold transcript coin stream + public inputs to the guest,
    // and have it run the verifier logic (CP-style: coins are provided, no hashing in-guest).
    let payload = b"hello-symphony-cp";

    let mut stdin = SP1Stdin::new();
    stdin.write_slice(payload);

    let client = ProverClient::from_env();
    let (pk, vk) = client.setup(SYMPHONY_CP_ELF);

    client.execute(SYMPHONY_CP_ELF, &stdin).run().expect("execution failed");
    let proof = client.prove(&pk, &stdin).run().expect("proving failed");
    client.verify(&proof, &vk).expect("verification failed");

    // Public value is LE-encoded u32 length.
    let expected = (payload.len() as u32).to_le_bytes();
    assert_eq!(proof.public_values.as_ref(), expected);
}

