fn main() {
    // Build the zkVM guest program and make its ELF available via `include_elf!(...)`.
    //
    // NOTE: This is scaffolding. The guest currently only proves a trivial computation,
    // and will be wired to the Symphony Π_fold verifier next.
    sp1_build::build_program("../program");
}

