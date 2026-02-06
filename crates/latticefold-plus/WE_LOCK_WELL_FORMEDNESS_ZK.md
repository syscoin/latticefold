# Minimal ZK well-formedness statement for PVUGC WE lock artifacts

This document specifies a **minimal** zero-knowledge statement that lets honest armers reject a malicious armer’s published lock artifacts that would **never decapsulate** (brick funds / sabotage liveness), even when a valid WE proof exists.

This is **not** a “standard RLWE reduction” document. The goal here is **liveness**:

- If the armer publishes artifacts that pass this ZK proof, then for any later valid proof stream \( \pi \) for the bound statement, decapsulation will produce a candidate Shamir share for that armer consistent with the armer’s published public key contribution.

The construction implemented in this repo is:

- **DPP**: Theorem 4.3 “hidden-query lockable DPP” over \( \mathbb{F}_{257} \) (arm-before-proof).
- **Lock**: Goldilocks-modulus ring arithmetic used only *outside* the tiny gate; no per-lock MAC/tag.
- **Payload encoding**: Frodo/Regev-style **additive encoding** of share bytes; decap does rounding decode.
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
      - hint blocks `branch_hints[b].hint_blocks_sparse` (sparse ring elements per branch),
      - ciphertext encodings `cts[b].encoded` (Goldilocks field elements per payload byte).

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

##### (D3) Hint-noise bound (for rounding correctness)

Prove the magnitude bound needed for correctness (liveness). A simple sufficient condition is:

- each coefficient of `e_{b,block}` is within a public bound \(B_e\) (e.g. \(6\sigma\) if sampling Gaussian with tail cut, or a fixed box bound if using bounded noise-by-construction).

This prevents a malicious armer from setting `e` so large that even a valid proof yields `C - y` outside the rounding radius.

Importantly: this is an **upper bound** check. An armer using *less* noise (even zero) does not brick decap; it may impact confidentiality, but that is outside the liveness goal of this statement.

##### (D4) Ciphertext correctness (additive encoding)

(This matches `arm_ringlwe_lock` in `lockable_ringlwe.rs`.)

For each lock instance \(\ell\), define its associated Shamir share index \(i_\ell \in \{1,\dots,R\}\).
Let `payload_ℓ` be the 32-byte share string `share_{i_ℓ}`.

For each branch \(b\in\{0,1\}\), let:

- \(a_b\) be the **shifted accepting element** `accepting_set_shifted[b]` (an \( \mathbb{F}_{257} \) element),
- \(a_b^{(q)} = \texttt{embed\_f\_to\_fq}(a_b)\) be its Goldilocks embedding (currently centered \([-128,128]\)),
- \(s_b \in \mathbb{F}_q\) be the branch secret scalar `s_scalar_b` (sampled short; nonzero is recommended),
- `cts[b].encoded[j]` be the published Goldilocks element encoding byte `payload_ℓ[j]`.

Prove for every byte index \(j\):

\[
\texttt{cts}_b[j] = s_b \cdot a_b^{(q)} + \mathrm{Enc}(\mu_j) + e'_{b,j}
\]

where:

- \(\mu_j := \texttt{payload}_\ell[j] \in \{0,\dots,255\}\),
- \(\mathrm{Enc}(\mu) = \left\lfloor \dfrac{(2\mu+1)\,q}{512} \right\rfloor\) (midpoint bucket encoding),
- \(e'_{b,j}\) is the per-byte ciphertext noise (in code: `centered_binomial(k)` embedded to Goldilocks).

This prevents an armer from publishing ciphertexts that are not encryptions of the intended Shamir share bytes under the branch signal used by decap.

##### (D5) Ciphertext-noise bound (for rounding correctness)

Correctness of `frodo_decode_byte` holds if the total additive error is \(< q/512\) from the correct bucket midpoint.

In decap, for a valid proof stream, the decapper computes (per branch \(b\)):

- \(y_b := \texttt{coeff0\_mul}(\pi_{\text{ring}}, h_b)\),

and then decodes each byte as:

- \(\mu'_j := \mathrm{Dec}(\texttt{cts}_b[j] - y_b)\) with \(\mathrm{Dec}(z)=\left\lfloor \dfrac{z\cdot 256}{q}\right\rfloor\).

Under the hint equation \(h_b = q\_{\text{ring}}\cdot s_b^{\text{const}} + e_b\) and correctness of the DPP relation, we have:

\[
y_b = s_b \cdot a_b^{(q)} + \underbrace{\texttt{coeff0\_mul}(\pi_{\text{ring}}, e_b)}_{\text{hint-noise term}}
\]

Therefore the decoded value is correct if:

\[
\left|\; e'_{b,j} - \texttt{coeff0\_mul}(\pi_{\text{ring}}, e_b)\;\right| < q/512
\]

The armer cannot know \(\pi\) at arming time, so the well-formedness proof must enforce a **uniform bound** that guarantees correctness for all valid \(\pi\) (or for all \(\pi\) in the allowed value range).

A simple sufficient public bound is:

- fix \(B_\pi := 128\) (because `embed_f_to_fq` maps any F257 value to magnitude \(\le 128\)),
- require each coefficient of each hint error block satisfies \(|e_{b,block}[t]| \le B_e\),
- and each ciphertext noise satisfies \(|e'_{b,j}|\le B_{e'}\),
- then require the public inequality:
  \[
  B_{e'} + d \cdot B_\pi \cdot B_e < q/512
  \]
  where \(d=64\) is the ring dimension used by `GoldilocksRing64`.

This blocks the sabotage “choose noise so large that honest decap always mis-rounds.”

> Note: you can tighten this bound substantially using the exact `coeff0_mul` structure and the actual sparsity/blocking; but the proof obligation stays the same: publish a bound, prove the witness is within it, and ensure the bound implies \(<q/512\).

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

- **Attack**: pick very large hint error or ciphertext noise so that even with the correct witness/proof, `frodo_decode_byte` mis-rounds.
- **Blocked by**:
  - **(D3)** per-coefficient hint-noise bound,
  - **(D5)** ciphertext-noise bound + the public inequality ensuring total error \(<q/512\).

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
- **Must keep (D1–D5)** or the armer can mismatch embeddings, publish arbitrary hints, or set noise so rounding fails.

You may be able to simplify:

- **Distributional proofs** (e.g. “these noises are Gaussian”) are *not* required for liveness; **bounds** are sufficient.
- **Nonzero secret constraints** (`s_b != 0`) are not required for liveness, but may be required for your confidentiality/security story.

---

## What this well-formedness proof does NOT cover

This proof is intentionally minimal for *liveness*, not for full cryptographic security:

- It does **not** prove hardness assumptions (RLWE, etc.).
- It does **not** prove the *eventual* prover has a witness.
- It does **not** prevent confidentiality attacks if you later add per-lock tags/MACs (that would reintroduce per-lock oracles).

