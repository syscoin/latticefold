# Minimal ZK well-formedness statement for PVUGC WE lock artifacts

This document specifies a **minimal** zero-knowledge statement that lets honest armers reject a malicious armer’s published lock artifacts that would **never decapsulate** (brick funds / sabotage liveness), even when a valid WE proof exists.

This is **not** a “standard RLWE reduction” document. The goal here is **liveness**:

- If the armer publishes artifacts that pass this ZK proof, then for any later valid proof stream \( \pi \) for the bound statement, decapsulation will produce a candidate Shamir share for that armer consistent with the armer’s published public key contribution.

The construction implemented in this repo is:

- **DPP**: Theorem 4.3 “hidden-query lockable DPP” over \( \mathbb{F}_{257} \) (arm-before-proof).
- **Lock**: Goldilocks-modulus ring arithmetic used only *outside* the tiny gate; no per-lock MAC/tag.
- **Payload encoding**: unauthenticated XOR-stream **DEM** under a derived key; decap derives the key and decrypts (no per-lock MAC/tag).
- **Amplification**: Batch Shamir reconstruction + a single global check (e.g. against the armer’s EC public key contribution).

Relevant code:

- DPP arming: `crates/dpp/src/theorem43.rs::Theorem43Dpp::arm`
- Query streaming: `crates/dpp/src/theorem43.rs::Theorem43Dpp::stream_query_terms_for_pi`
- Lock arming: `crates/latticefold-plus/src/we_tiny_lock.rs::arm_we_ringlwe_lock_from_dr1cs`
- Lock math: `crates/latticefold-plus/src/lockable_ringlwe.rs`

---

## Threat model (what we are preventing)

A malicious armer can try to publish a lock artifact that:

- looks syntactically correct, and binds to the same statement as honest armers, but
- is **not** consistent with any underlying secret/share/noise/query that would allow correct decoding, so that
- after funding, even an honest prover/decapper with a valid witness/proof cannot recover that armer’s share(s), bricking the combined key.

We assume:

- The prover/decapper is local (no external decryption oracle).
- We do **not** require ciphertext integrity for confidentiality (unauthenticated is OK).
- We do require the artifact to be **decapsulatable** under valid proof, i.e. noise must be bounded so rounding succeeds.

---

## Public inputs and statement binding

Fix the **WE statement** (the thing the eventual proof will prove), including:

- `stmt_digest` (or equivalently `c_stmt = digest32_to_bits_field(stmt_digest)`).
- the public statement vector `x` (in WE-gate mode: `x = [ONE] || [WeParams] || [public_inputs...]`).
- `block_id`, `rep_id` (selects the Thm-4.3 coin instance).
- the lock parameters `params = { secret_binomial_k, noise_log2_sigma, recon_bits, ... }` and the encoding conventions used by the lock.

**Binding rule (recommended minimal):**

- Treat `stmt_digest`, `block_id`, `rep_id`, and all lock parameters as **public inputs** to the well-formedness proof.
- Require the lock artifact’s published `c_stmt`, `coins`, `offset`, and `accepting_set_shifted` match the canonical derivation from these public inputs (details below).

Why: if parameters are not statement-bound, a malicious armer can pick pathological parameters (especially bounds) that make decap fail even for valid proofs.

---

## Minimal ZK well-formedness statement (per armer)

### Public inputs (what verifiers see)

For each armer \(j\):

- **EC public contribution**:
  - either `P_j` (armer’s public point), or the sequential delta `ΔP = P_out - P_in` if the protocol accumulates contributions.
- **WE/DPP statement binding**:
  - `stmt_digest` (or `c_stmt`), and the exact public statement vector `x` used by DPP arming,
  - `block_id`, `rep_id`,
  - DPP coins `coins = (idx, lambda, rho, sigma)` as published in the lock artifact,
  - lock parameters `params` (and any bounds derived from them).
- **Shape/length binding**:
  - `x_len` and `pi_len` (must match the DPP instance for this statement),
  - lockset layout: the list of lock instances \(\ell\) with their `(block_id, rep_id)` and their Shamir share index \(i_\ell\),
  - payload length (for secp256k1 scalar bytes: 32).
- **Published lock artifact** for that armer and its share-lockset:
  - `accepting_set_shifted` (2 elements, nonzero),
  - `offset`,
  - for each lock instance in the lockset:
    - `q_blocks` are *not* published directly in the current artifact; the public artifact includes:
    - hint blocks `branch_hints[b].hint0_blocks_sparse` and `branch_hints[b].hint1_blocks_sparse` (two sparse ring-hint vectors per branch),
    - ciphertexts `cts[b].nonce` and `cts[b].ct` (bytes).

### Witness (what the armer proves knowledge of)

The witness contains:

1. **The armer’s secret scalar** \(s_j\) (secp256k1 scalar, 32 bytes).
2. **Shamir polynomial coefficients** for splitting \(s_j\) into \(R\) shares with threshold \(T\):
   - polynomial \(f\) over GF(256), degree \(T-1\), with \(f(0)\) equal to the byte encoding of \(s_j\).
3. **Armer secret salt** (the secret input mixed into Thm‑4.3 hidden-query derivation):
   - either `armer_seed` (from which the armer secret vector is derived), or the derived `armer_secret` vector itself.
4. **Hidden-query randomness** for Theorem‑4.3:
   - the UV bits / Sq coefficient vector `coeffs` used by `Theorem43Dpp::stream_query_terms_for_pi`,
   - plus any intermediate values needed to show they were derived correctly from the Poseidon transcript.
5. **Lock secrets and noise** for each lock instance and branch:
   - per branch \(b\in\{0,1\}\): the scalar secret `s_scalar_b` used in hints/ciphertext,
   - per block: ring error terms `e_{b,block}` for hint correctness,
   - per payload byte: ciphertext noise `e'_{b,j}` for encoding correctness.

### Constraints (what must be proven)

We group constraints by *what they prevent*.

#### (A) EC scalar commitment (binds armer’s key contribution)

Prove:

- \( s_j \cdot G = P_j \) (or \(s_j\cdot G = \Delta P\)).

This prevents an armer from publishing shares for some \(s'_j\) while claiming a different public contribution.

#### (B) Shamir correctness (binds locked payloads to the same `s_j`)

Over GF(256) per byte, prove that the \(R\) share strings are polynomial evaluations of degree \(T-1\) with free term equal to the byte encoding of \(s_j\):

- For each share index \(i\in\{1,\dots,R\}\) and each byte position \(t\in\{0,\dots,31\}\):
  \[
  \text{share}_i[t] = f_t(i)
  \]
  where \(f_t\) is the degree-\(T-1\) polynomial for byte \(t\) with \(f_t(0) = s_j[t]\).

This prevents an armer from encrypting unrelated garbage shares that can never reconstruct \(s_j\).

#### (C) DPP arming transcript binding (binds query, coins, offset, accepting set)

Split into what is publicly checkable vs what needs ZK:

- **(C0) `c_stmt` correctness (public check, if both provided):**
  - If the artifact publishes `c_stmt` as field elements/bits, require it equals the canonical expansion of `stmt_digest`.
  - (In this repo, callers often pass `c_stmt` directly; PVUGC may prefer `stmt_digest` as the canonical public value.)

- **(C1) Public coins correctness (public check, no ZK needed):**
  - Recompute Theorem‑4.3 public coins from `(c_stmt, x, block_id, rep_id)` and check equality with published `coins`.

- **(C2) Hidden query correctness (ZK):**
  - Prove that the hidden-query randomness (`coeffs` / UV bits) is derived by the canonical Poseidon transcript from:
    \[
    (c\_\text{stmt}, x, block\_id, rep\_id, coins, armer\_secret)
    \]
    matching `Theorem43Dpp::arm`.

- **(C3) Query-to-(π) terms and offset correctness (ZK):**
  - Run the canonical `stream_query_terms_for_pi` logic (in-circuit) to derive:
    - the sparse π-query coefficients (fed into `QueryBlockAccumulator`), and
    - the `offset` returned by the streaming sink,
  - and prove these match the values implicitly used to form the published lock:
    - `offset` equals the artifact’s published offset,
    - `accepting_set_shifted = [1-offset, 2-offset]` equals the published shifted accepting set (and both nonzero).

This prevents an armer from using a different query/offset than the one the decapper will evaluate against a valid proof stream.

#### (C4) Multi-channel (“P channels”) domain separation (if used)

Some lock variants “pack” \(P\) independent *channels* inside a single lock instance, with the intent that the payload key is derived from **all** channels (e.g. KDF over \(2P\) small-field seeds).

If the lock uses packed channels, the well-formedness statement MUST treat the following as part of the public binding:

- the integer `P` (number of channels),
- a canonical channel index `part_id ∈ {0,1,...,P-1}` for each channel.

And the ZK proof MUST enforce **per-channel** transcript binding:

- **(C4.1) Per-channel public coins correctness:** for each `part_id`, the published coins for that channel equal the canonical Fiat–Shamir derivation from `(c_stmt, x, block_id, rep_id, part_id)` (domain-separated).
- **(C4.2) Per-channel hidden-query correctness:** for each `part_id`, the hidden-query randomness (`coeffs` / UV bits / Sq coefficients) is derived by the canonical transcript from
  \[
  (c\_\text{stmt}, x, block\_id, rep\_id, part\_id, coins, armer\_secret),
  \]
  matching the implementation.

**Security note:** the proof does not (and cannot) “prove independence.” Independence across channels is an *assumption* that the transcript hash / PRF behaves independently under distinct `part_id`s. The proof’s job is to prevent accidental or malicious *correlation-by-construction* (e.g., reusing the same query across channels) by enforcing correct domain separation and per-channel derivations.

#### (D) Lock artifact correctness (hints + ciphertext encodings + bounds)

This is the liveness-critical part: it prevents “publish artifacts that can’t decode.”

##### (D0) Structural correctness (shape/length)

Prove the artifact is structurally consistent with the DPP instance and lockset layout:

- the published `x_len` equals `len(x)` and matches the DPP instance for this statement,
- the published `pi_len` equals the expected streamed-π length for this statement,
- each lock instance \(\ell\) has ciphertext length exactly equal to the payload length (32 bytes for a secp256k1 scalar share),
- the lockset indices \(i_\ell\) are within \(1..=R\) and are all distinct (recommended; otherwise reconstruction guarantees degrade).

This blocks simple “truncate / wrong-shape / wrong-indexing” bricking attacks.

##### (D1) Packing correctness

Prove the canonical packing used by `lockable_ringlwe.rs`:

- embed each F257 coefficient into Goldilocks using the agreed embedding (currently **centered** in `lockable_ringlwe.rs`),
- pack each block of \(d=64\) coefficients into `GoldilocksRing64` via:
  - `q_ring[0] = q[0]`, `q_ring[d-i] = -q[i]` for \(i\ge 1\),
  - so that `coeff0_mul(q_ring, pi_ring) = ⟨q,π⟩` (as integers mod q).

If packing is wrong, decap computes the wrong inner product and cannot cancel.

##### (D2) Hint correctness

For each lock instance, branch \(b\), and each published hint block `(block_idx, h)` prove:

\[
h = q\_{ring,block} \cdot s\_{const,b} + e\_{b,block}
\]

where:

- `s_const,b` is the constant-polynomial embedding of the scalar secret `s_scalar_b`,
- `e_{b,block}` is the ring error.

##### (D3) Hint-noise bound (for seed-extraction stability)

Prove the magnitude bound needed for correctness (liveness).

In our current `lockable_ringlwe.rs` design, payload recovery is performed by:

- streaming a decapsulation signal \(y \in \mathbb{F}_q\) (Goldilocks),
- center-lifting \(y\) to an integer \(\tilde{y}\in(-q/2,q/2]\),
- reducing \(\tilde{y} \bmod 257\) to obtain small-field seed(s),
- deriving a DEM key via SHA-256, and
- decrypting an unauthenticated XOR-stream ciphertext.

Therefore, the hint/noise bounds are used to ensure **seed-extraction stability** (no wraparound in the center lift), rather than any Frodo-style per-byte rounding.

A simple sufficient condition is:

- each coefficient of each hint error block satisfies \(|e_{b,block}[t]| \le B_e\), for a public bound \(B_e\),
- and the public “no-wrap” bound in **(D5)** holds for all derived seed components.

This prevents a malicious armer from setting `e` so large that even a valid proof yields a wrapped decapsulation signal and bricks key derivation.

Importantly: this is an **upper bound** check. An armer using *less* noise (even zero) does not brick decap; it may impact confidentiality, but that is outside the liveness goal of this statement.

##### (D4) Ciphertext correctness (XOR-DEM under derived key)

(This matches `arm_ringlwe_lock` in `lockable_ringlwe.rs`.)

For each lock instance \(\ell\), define its associated Shamir share index \(i_\ell \in \{1,\dots,R\}\).
Let `payload_ℓ` be the 32-byte share string `share_{i_ℓ}`.

For each branch \(b\in\{0,1\}\), the armer publishes:

- a nonce `cts[b].nonce`,
- a ciphertext `cts[b].ct`,
- and hint material which determines (under a valid proof stream) seed components \((y_{b,i} \bmod 257)\) used for key derivation.

Let `K_b` be the derived 32-byte key computed from the published statement binding and the recovered seeds (exact KDF specified by the implementation; in current code it is SHA-256 over domain label, statement binding, coins, and the seed components).

The well-formedness proof MUST enforce that:

- `cts[b].ct == XOR_STREAM(K_b, cts[b].nonce, payload_ℓ)` for both branches \(b\),

where `XOR_STREAM` is the unauthenticated stream cipher used by the implementation.

This prevents an armer from publishing ciphertext bytes that are unrelated to the intended Shamir share payload under the key the decapper will derive.

##### (D5) Seed-extraction stability (no-wrap for mod‑257 key seeds)

Our current design reduces a center-lifted decapsulation signal modulo 257 to obtain key seeds.

In this design, correctness requires that the **center lift be stable**, i.e. that the “true” intended integer signal does **not** wrap modulo \(q\). Otherwise, \(\tilde{y}\) can differ by \(\pm q\), and since \(q \not\equiv 0 \pmod{257}\), the extracted seed \((\tilde{y} \bmod 257)\) can change, bricking decapsulation.

Therefore, the well-formedness proof MUST include a public “no-wrap” bound \(B_{\text{wrap}} < q/2\) and enforce:

- for each branch \(b\) and each seed component used by the DEM key derivation (e.g. \(y_{b,0}, y_{b,1}\) or \(2P\) components for \(P\) packed channels),
  \[
  |\tilde{y}_{b,i}| \le B_{\text{wrap}}.
  \]

**Operational note (arm-time retry):** armers MAY resample secrets/noise (and, if applicable, repetition coins) until the produced artifact satisfies the no-wrap bound. The ZK proof does not prove “a retry happened”; it proves the **final artifact** lies in the no-wrap region.

###### (D5.1) How to choose / compute a sound public no-wrap bound (arm-before-proof)

The key point is that we need a **worst-case** bound that does not depend on the future proof stream \(\pi\) itself.
This is possible in our current implementation because:

- the decapper embeds each streamed \(\pi\) coefficient from \( \mathbb{F}_{257} \) into Goldilocks using **centered representatives**,
  so each embedded coefficient satisfies
  \[
  |\pi_j^{(\text{emb})}| \le 128,
  \]
  independent of the witness/proof value (this is a *representation* bound, not a soundness claim).
- per lock, the decapsulation accumulator only processes **hinted blocks** (sparse), and each processed block contributes a signed dot-product
  `coeff0_mul_row(h_block, pi_row)` where the signs/permutation are fixed by the packing convention.

Let a processed hint block have Goldilocks coefficients \(h[0..d-1]\) and the corresponding embedded \(\pi\) row be \(r[0..d-1]\).
In the negacyclic coefficient-0 formula used by the implementation,
each term is of the form \(\pm h[i]\cdot r[j]\), so by the triangle inequality:

\[
\big|\langle h, r\rangle_{\pm}\big|
\le \sum_{j=0}^{d-1} |h[j]|\,|r[j]|
\le \|r\|_\infty \cdot \sum_{j=0}^{d-1} |h[j]|
\le 128 \cdot \|h\|_1.
\]

Now sum over **all processed (hinted) blocks** for that branch/component:

\[
|\tilde{y}_{b,i}| \le 128 \cdot \sum_{\text{hinted blocks}} \|h_{b,i,\text{block}}\|_1.
\]

This yields a sound, arm-time-computable public bound:

- define the centered-lift magnitude for a Goldilocks coefficient \(c\in\mathbb{F}_q\) as
  \[
  |c|_{\text{cent}} := \min(c_{\text{u64}},\, q - c_{\text{u64}}),
  \]
  where \(c_{\text{u64}}\in[0,q)\) is the canonical u64 representative;
- compute
  \[
  \|h\|_1 := \sum_{j=0}^{d-1} |h[j]|_{\text{cent}}
  \]
  for each published hint block;
- set
  \[
  B_{\text{wrap}} := 128 \cdot \sum_{\text{hinted blocks}} \|h_{b,i,\text{block}}\|_1.
  \]

Finally require

\[
B_{\text{wrap}} \;<\; \frac{q}{2}.
\]

**Notes:**

- If the lock uses **two hint vectors per branch** (e.g. `hint0` and `hint1` / two seed components), compute \(B_{\text{wrap}}\) **separately**
  for each component \(i\in\{0,1\}\).
- If the lock packs **P channels**, compute the bound for each channel/component, or conservatively sum the \(\ell_1\) norms across channels.
- This bound is conservative but **fully arm-before-proof**: it uses only public artifacts plus the representation bound \(\|\pi\|_\infty\le 128\).

###### (D5.2) Where this lives (ZK vs public check)

This “no-wrap” condition is a pure predicate over the **public hint coefficients** and public parameters, so it can be:

- enforced **inside** the ZK well-formedness circuit (as the doc’s minimal statement requires), or
- enforced as a **public pre-check** alongside verifying the ZK proof.

Either way, the security/liveness meaning is the same: the verifier rejects artifacts whose hints are so large that they can wrap the centered lift and brick mod‑257 seed extraction.

---

## Why these constraints cover all sabotage-to-brick attacks

Below is an exhaustive list of armer moves whose *only purpose* is to make decap fail even for an honest prover with a valid witness/proof stream, and which constraint blocks each move.

### 1) “Bind to the wrong statement / wrong coins / wrong rep_id”

- **Attack**: armer publishes artifacts under different `(stmt_digest, x, block_id, rep_id)` than the funded address expects, so a later proof for the intended statement produces a different \(\pi\), and decap fails.
- **Blocked by**:
  - **Public input binding** (the well-formedness proof’s statement includes these values),
  - **(C1)** coins recomputation,
  - **(C2–C3)** query/offset correctness tied to those coins and the same statement.

### 2) “Use a different hidden query than the one decap will evaluate”

- **Attack**: armer arms with a different hidden query (different `coeffs` / transcript) so the published hints correspond to a different `q_ring` than the future proof uses.
- **Blocked by**:
  - **(C2)** hidden-query derivation correctness from the canonical transcript,
  - **(C3)** stream-derived query terms and `offset` correctness,
  - **(D2)** hint correctness with that `q_ring`.

### 3) “Use the wrong offset / wrong shifted accepting set”

- **Attack**: publish a shifted accepting set not equal to `[1-offset, 2-offset]` (or allow a zero element), which breaks the decapper’s cancellation term.
- **Blocked by**:
  - **(C3)** `accepting_set_shifted = [1-offset, 2-offset]`,
  - the explicit **nonzero accepting-set** check (matches `arm_ringlwe_lock`).

### 4) “Break packing/embedding so the dot product is wrong”

- **Attack**: armer uses a different embedding of F257 into Goldilocks or a different ring packing, so `coeff0_mul` during decap is not aligned with the armer’s hint/ciphertext math.
- **Blocked by**:
  - **(D1)** packing correctness constraint fixing the embedding and packing convention.

### 5) “Publish random hints not tied to any secret/noise”

- **Attack**: publish arbitrary `h` blocks so that `y_b` at decap is unrelated to any branch signal.
- **Blocked by**:
  - **(D2)** hint equation \(h = q\cdot s + e\).

### 6) “Choose noise so large that rounding always fails”

- **Attack**: pick very large hint error (or other artifact parameters) so that even with the correct witness/proof, the decapsulation signal’s center-lift wraps modulo \(q\), changing \((\tilde{y}\bmod 257)\) and bricking key derivation / XOR-DEM decryption.
- **Blocked by**:
  - **(D3)** per-coefficient hint-noise bound,
  - **(D5)** the public no-wrap bound ensuring seed-extraction stability.

### 7) “Encrypt the wrong payload (not the Shamir share)”

- **Attack**: ciphertext encodes garbage bytes or bytes from a different share index, so reconstruction fails.
- **Blocked by**:
  - **(B)** Shamir share correctness from the same `s_j`,
  - **(D4)** ciphertext correctness for the lock’s designated share index \(i_\ell\).

### 8) “Make shares inconsistent with the claimed armer public key”

- **Attack**: armer publishes a `P_j` (or `ΔP`) that doesn’t match the `s_j` whose shares are being encrypted, so a global EC check can never succeed.
- **Blocked by**:
  - **(A)** EC scalar commitment,
  - **(B)** Shamir ties all shares to the same `s_j`.

### 9) “Permute or duplicate shares across locks”

- **Attack**: publish \(R\) locks but map them to the wrong Shamir x-coordinates (or duplicate a coordinate), reducing decap success or breaking reconstruction.
- **Blocked by**:
  - including the lock’s intended share index \(i_\ell\) as a **public input**,
  - **(B)** uses the correct evaluation point,
  - **(D4)** enforces that lock \(\ell\) encrypts `share_{i_ℓ}`.

### 10) “Break shapes (wrong `x_len`, `pi_len`, payload length)”

- **Attack**: publish artifacts whose internal lengths don’t match the statement/DPP instance (or truncate ciphertexts), guaranteeing decap failure or reconstruction failure.
- **Blocked by**:
  - **Shape/length binding** as public inputs,
  - **(D0)** structural correctness constraints.

---

## Minimality: what you can’t drop (if the goal is “no bricking”)

- **Must keep (A)** if the only global check is the armer’s EC public key. Otherwise, an armer can encrypt shares of some other scalar and still “well-form” the lock layer.
- **Must keep (B)** (or an equivalent binding of payload to `s_j`) or the armer can encrypt random bytes that never reconstruct the armer scalar.
- **Must keep (C2–C3)** (or publish the hidden-query seed, which is usually unacceptable) or the armer can arm to a different query than the later proof stream.
- **Must keep (D1–D5)** or the armer can mismatch embeddings, publish arbitrary hints, publish ciphertexts that do not decrypt to the intended share under the derived key, or set parameters so seed extraction wraps and decap fails.

You may be able to simplify:

- **Distributional proofs** (e.g. “these noises are Gaussian”) are *not* required for liveness; **bounds** are sufficient.
- **Nonzero secret constraints** (`s_b != 0`) are not required for liveness, but may be required for your confidentiality/security story.

---

## What this well-formedness proof does NOT cover

This proof is intentionally minimal for *liveness*, not for full cryptographic security:

- It does **not** prove hardness assumptions (RLWE, etc.).
- It does **not** prove the *eventual* prover has a witness.
- It does **not** prevent confidentiality attacks if you later add per-lock tags/MACs (that would reintroduce per-lock oracles).

