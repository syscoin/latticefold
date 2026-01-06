# Symphony

This crate implements the **Symphony** folding layer from
[*Symphony: Post-Quantum Folding from Lattices* (ePrint 2025/1905)](https://eprint.iacr.org/2025/1905)
and integrates it with the surrounding `latticefold` codebase.

At a high level, Symphony is a lattice-based folding SNARK design that is built around:
- A structured folding protocol `Π_fold` (with sub-protocols like `Π_rg` / range checks),
- **Shared-randomness** sumcheck schedules (to reduce verifier costs and avoid redundant transcript work),
- A transcript design intended to be **algebraic-friendly** (important for recursion / downstream compilers).

The focus of this repository’s Symphony implementation is **practical proving**:
streaming/compact MLE representations, avoiding large dense tables where possible, and
supporting high-concurrency proving for chunked constraint systems.

**DISCLAIMER:** This is a proof-of-concept prototype, and in particular has not received careful code review. This implementation is provided "as is" and NOT ready for production use. Use at your own risk.

## Relationship to LatticeFold / LatticeFold+

- **LatticeFold**: baseline lattice-based folding approach in this repo (see the `latticefold` crate).
- **LatticeFold+**: improves the folding pipeline and reduces overhead using an optimized algebraic
  range-proof structure (see
  [*LatticeFold+: Faster, Simpler, Shorter Lattice-Based Folding for Succinct Proof Systems* (ePrint 2025/247)](https://eprint.iacr.org/2025/247)).
- **Symphony**: a folding-layer design that emphasizes (i) a clean sumcheck schedule with shared
  verifier challenges and (ii) a transcript structure that is convenient for downstream “prove the
  verifier” settings without paying large hash-gadget costs.

In code terms, this crate provides:
- `Π_fold` provers/verifiers (batched and streaming variants),
- the `R_cp` (“tie / transcript consistency”) relation API for WE/DPP-facing uses,
- chunked SP1 R1CS loading helpers and example drivers.

## WE / SP1 notes (PVUGC PQ integration footnote)

The Symphony paper is not written around the SP1→WE deployment we target in PVUGC, so this repo
includes an explicit adaptation that is worth stating precisely.

- **PVUGC (PQ variant)**: the post‑quantum PVUGC integration work that targets SP1→WE; see the
  [PVUGC repository](https://github.com/sidhujag/PVUGC).
- **SP1**: the zkVM we currently target for the “chunked R1CS → folding” pipeline; see
  [Succinct’s SP1](https://github.com/succinctlabs/sp1).
- **WE (witness encryption)**: “decrypt iff a statement is true”. In our setting, the *decapper* is
  a public algorithm that takes a candidate statement and derives a key if (and only if) the gated
  relation holds.
- **DPP (Dot‑Product Proofs)**: a compiler that turns a final folded Symphony verifier relation into a linear predicate of
  the form \(\langle q, y\rangle \in A\) (see [*Dot‑Product Proofs* (ECCC TR24-114 / ePrint 2024/1138)](https://eprint.iacr.org/2024/1138)).

### Gate we target
We expose a WE/DPP‑facing gate as a conjunction of relations:

\[
R_{WE} \;:=\; R_{cp} \wedge R_o.
\]

- `R_cp` checks the folding transcript / coin schedule / commitment-opening consistency (the “tie” relation).
- `R_o` checks the reduced relation on the folded output (application-specific).

This avoids “verifier‑of‑a‑proof” inside the DPP target: DPP is meant to target the relations directly.

### Why we do **not** require a CP-SNARK in our current design
In the current PVUGC integration, the decapper/prover is allowed to treat auxiliary transcript messages
as **witness** and to run an algebraic verifier path. We are not currently aiming for witness *privacy*
at the folding layer (i.e., no claim of zero-knowledge for `Π_fold` outputs).

Under that model, using the conjunction gate above is safe in the following (minimal) sense:
- **Soundness**: an adversary who can make the public decapsulation succeed without a valid witness
  for the underlying statement would violate soundness of the composed gate (ultimately bounded by
  the folding layer + the DPP soundness used to linearize it, plus threshold amplification in the lock layer).
- **No additional leakage claim**: since the witness is not treated as secret at this layer, auxiliary
  messages (e.g., `had_u`, `mon_b`) being part of the witness does not create a new confidentiality target.

If/when we need witness-hiding or a public proof artifact with strong privacy guarantees, the interface
must be tightened accordingly (e.g., moving auxiliary messages behind commitments/openings checked in
the target relation, or wrapping in an external proof system).

Note: DPPs do not admit perfect completeness in general; our lock/threshold layer is designed to be
robust to non-perfect completeness (see Appendix A of TR24‑114 Revision 2).

### Parameter / “shape” deviations (intentional)
Some parameters in the SP1 chunked driver differ from the paper’s “default” regime because chunking
constrains the matrix row count `m` per chunk:
- We treat `(l_h, lambda_pj, k_g, d')` as part of the **statement shape**. Changing them changes the
  verifier predicate (and the coin schedule).
- For chunked SP1 proving with `m ≈ 2^20`, `lambda_pj` must be small enough that
  \(m_J = (n_f/l_h)\cdot \lambda_{pj} \le m\) and \(m_J \mid m\). In practice this means `lambda_pj=1`
  unless the chunk size is increased.
- Increasing `l_h` reduces memory (tables scale with `m_J`) but increases FS coin bytes because
  `derive_J` squeezes `lambda_pj*l_h` bytes. This is expected and does not change correctness.

## Benches
Run benches with,

```sh
cargo bench
```

## TODO
- Add / use 128-bit ring / or support 128-bit security through extensions. |S| = 128 bit, |C| = 64 bit;
- Parameter tuning (B, kappa, etc);
- Some sumchecks over Zq (avoid Rq promotions);
- Various paper-provided performance improvements;
- Explore NTT (e.g. `SuitableRing`);
- Support CCS;
- Enhance main (Prover/Verifier) and and more internal interfaces (sub-protocols): target real usability, reduce cloning;
- Add error types / error handling. Remove `unwrap`s;
- Docs;
- Add examples;
- Add more benches for each subprotocol. Add benches targeting LF comparison (with equivalent parameters).
