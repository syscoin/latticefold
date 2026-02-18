# Minimal ZK well-formedness statement for LF+ WE lock artifacts (current design)

This document specifies a **minimal** zero-knowledge well-formedness statement that lets honest
participants reject a malicious armer’s published lock artifacts that would **never decapsulate**
(brick liveness), even when a valid WE proof stream \( \pi \) exists.

This is **not** a standard RLWE/KEM reduction. The goal here is **liveness**:

- If an armer publishes artifacts that verify under this ZK statement, then for any later **valid**
  proof stream \( \pi \) for the bound WE statement, the decapsulation algorithm (streaming in \( \pi \))
  will output **a candidate payload set that contains the correct Shamir share bytes** for that armer.

The construction implemented in this repo (as of the `wegate-fixlock` work) is:

- **DPP**: Theorem 4.3 “hidden-query lockable DPP” over \( \mathbb{F}_{257} \) (arm-before-proof).
- **Lock**: deterministic mod-257 linear hints (packed in 64-wide blocks, sparse-or-dense in-block).
- **Ciphertext**: a single unauthenticated XOR-stream DEM under a SHA-256-derived keystream.
- **Amplification**: Shamir share reconstruction + a single global check (e.g. EC/address binding).

Relevant code:

- DPP arming: `crates/dpp/src/theorem43.rs::Theorem43Dpp::arm`
- Query streaming: `crates/dpp/src/theorem43.rs::Theorem43Dpp::stream_query_terms_for_pi`
- Sublock arming (computes shifted accepting set, coins, and the hidden query blocks):  
  `crates/latticefold-plus/src/we_tiny_lock.rs::arm_we_ringlwe_sublock_from_dr1cs`
- Lock math + streaming decap: `crates/latticefold-plus/src/lockable_ringlwe.rs`

---

## Threat model (what we are preventing)

A malicious armer can try to publish a lock artifact that:

- binds to the right statement, but
- is **not** consistent with any underlying secret shares / hidden query / ciphertext binding, so that
- even an honest prover/decapper with a valid proof stream \( \pi \) cannot recover that armer’s share(s),
  bricking the combined key (in an \(N\)-of-\(N\) armer outer policy).

Assumptions / non-goals:

- The decapper is local (no external decryption oracle).
- We intentionally publish **no per-lock MAC/tag** (to avoid a per-lock verification oracle).
- We are not trying to prove “best possible” confidentiality here; only **non-bricking decappability**.
- At arming/ceremony time, the final WE witness is typically **not** available yet. This
  well-formedness proof is checked now so that later, **if** a valid WE proof stream \( \pi \) is
  provided for the bound statement, decapsulation succeeds.
- A malicious armer can always DoS by not publishing, or by publishing invalid proofs. This statement
  only says: **if** the well-formedness proof verifies, the artifact is decapsulatable.

---

## Artifact model (what exists today)

The public lock artifact is `RingLweLockArtifact<F257>`:

- `c_stmt`, `x_len`, `pi_len`, `len`
- `params: RingLweParams` (canonical today: `_reserved0=2`, dual-format in-block hints)
- `p_channels` (number of scalar channels \(P\))
- `r_reps` (repetitions per channel \(R\))
- `sublocks: Vec<RingLweSubLock<F257>>` (one per `(channel,rep)`), each with:
  - `channel_id`
  - `accepting_set` (the **shifted** 2-element accepting set, both nonzero)
  - `coins = (idx, lambda, rho, sigma)` (public Thm-4.3 coins)
  - `hints` (deterministic mod-257 hints packed in 64-wide blocks)
- `ct: LockCiphertext` (single nonce + ciphertext bytes)

Decap is streaming and per-sublock computes:
\[
y = \langle h, \pi \rangle \in \mathbb{F}_{257},
\]
then forms the 2-candidate set for the channel secret
\[
\{\, y/a_0,\; y/a_1 \,\}
\]
where \(\{a_0,a_1\}\) is that sublock’s shifted accepting set.

Higher-level logic can:

- **Intersect across repetitions** within a channel to recover a unique per-channel secret (policy), and/or
- **Enumerate** the small remaining candidate sets and defer disambiguation to the single global check.

This document’s ZK goal is only: the correct share is in the decapper’s candidate set(s).

---

## Public inputs and statement binding

Fix the WE statement (the thing the eventual proof proves), including:

- `stmt_digest` (or equivalently `c_stmt = digest32_to_bits_field(stmt_digest)`).
- the public statement vector `x` used by Theorem-4.3 query streaming.
- for each sublock: `(block_id, rep_id)` and the `channel_id` (domain separation inputs).
- lock parameters `RingLweParams` (including `_reserved0` and `domain_label`).
- the lock layout: `(P=p_channels, R=r_reps, share_index list, payload length)` as applicable.

---

## Minimal ZK well-formedness statement (per lock package)

### Public inputs

For each published lock package, the verifier sees:

- **Statement binding**: `stmt_digest` and the exact `x` used.
- **Lockset layout**: how many share-locks, each lock’s `(P,R)` and the per-lock list of sublocks
  (their `channel_id`, `block_id`, `rep_id`) and the per-lock Shamir share index used as payload.
- **Published artifacts**: the full `RingLweLockArtifact` objects (including `sublocks` + `ct`).

### Witness

This is the **well-formedness (arming-time) witness**, not the eventual WE witness used later to
produce \( \pi \).

It proves knowledge of *the secrets that make the artifact a real arming output*:

1. **Combined key** `combined_key32` (32 bytes).
2. **Combine scheme witness** (current combine-v1 is Shamir over GF(256)):
   - polynomial coefficients (per byte) showing the \(K\) payload shares are evaluations of
     degree-\(T-1\) polynomials with free term equal to the corresponding byte of `combined_key32`.
3. **Armer secret salt/seed** used by Thm‑4.3 hidden-query derivation (or the derived `armer_secret`)
   for each share-lock.
4. **Hidden-query witness** per sublock:
   - the UV bits / `coeffs` that `Theorem43Dpp::arm` derives from `(c_stmt, x, block_id, rep_id, coins, armer_secret)`.
5. **Per-channel lock secrets**:
   - the per-channel small-field secrets \(s^{(i)}\in\{1,\dots,256\}\subset \mathbb{F}_{257}^\*\) used to scale each sublock’s hidden query.

### Constraints (minimal for non-bricking decap)

#### (A) Bind payload shares to the same `combined_key32`

Prove Shamir correctness for the share bytes used as payloads:

- each published payload share for index \(i\) equals evaluation \(f(i)\) of a degree-\(T-1\)
  polynomial with \(f(0)=\texttt{combined\_key32}\) (bytewise in GF(256)).

This prevents publishing unrelated garbage shares that can never reconstruct `combined_key32`.

#### (C) Per-sublock Thm‑4.3 transcript binding (query/coins/accepting set)

For each sublock:

- **(C1) Public coins correctness (public check allowed):** recompute Thm‑4.3 public coins from
  `(c_stmt, x, block_id, rep_id)` (and any domain-separation inputs used in the implementation) and check equality with published `coins`.
- **(C2) Hidden query correctness (ZK):** prove the hidden-query witness (`coeffs` / UV bits) matches the canonical derivation from
  \[
  (c_\text{stmt}, x, block_id, rep_id, coins, armer_secret)
  \]
  matching `Theorem43Dpp::arm`.
- **(C3) Streamed query → accepting-set binding (ZK):** run `stream_query_terms_for_pi` logic and prove:
  - the sublock’s published `accepting_set` equals `[c_hit, c_hit+1]` for the derived `c_hit`,
  - both accepting elements are nonzero.

This prevents an armer from publishing mismatched `(accepting_set, coins)` that would make an honest decapper compute the wrong divisions.

#### (D) Hint correctness (deterministic mod‑257 scaling; no noise)

For each sublock and each published hinted block `(block_idx, h_block)`:

- derive the canonical hidden query block \(q_{block}\) from the streamed query terms, and prove:
  \[
  h_{block} = q_{block} \cdot s_{\text{channel}} \pmod{257}.
  \]

There is no “noise bound” in the current design: hints are deterministic mod‑257 objects.

#### (E) Ciphertext correctness (single ciphertext, key derived from all channels)

Let `payload_ℓ` be the 32-byte share string for this lock’s share index.

Let `K_lock` be the key derived by the implementation (`derive_payload_key_bytes_multi` in
`lockable_ringlwe.rs`), which binds:

- `domain_label`, `c_stmt`,
- `(P,R)` and all sublocks’ public `(channel_id, accepting_set, coins)`,
- the tuple of per-channel secrets \(s^{(0)},\dots,s^{(P-1)}\).

Then prove:
\[
ct = XOR\_STREAM(K_{lock}, nonce, payload_\ell).
\]

This prevents publishing a ciphertext unrelated to the intended payload under the key the decapper will derive.

---

## Required public policy check for **non-bricking** decap at \(R=2\)

With \(R=2\), the implementation expects repetitions to disambiguate each channel by intersecting
two 2-candidate sets.

To make this **deterministic** (and to prevent a malicious armer from making many locks ambiguous
and thereby exceeding the decap fallback cap), the following check should be enforced from **public**
data (either inside the ZK circuit or as a public pre-check tied to the ZK statement):

- for each sublock, compute the ratio class of its shifted accepting set:
  \[
  r = (a_1/a_0) \in \mathbb{F}_{257}^\*, \quad class(r)=\min(r,r^{-1})
  \]
If `class(r)` repeats within a channel, then the intersection can stay ambiguous, producing up to
\(2^P\) candidates per share-lock; across \(T=16\) share-locks this can exceed the implementation’s
fallback enumeration cap and brick liveness.

If you instead run with \(R\ge 3\), you can relax this to a probabilistic condition, but the current
canonical setting is \(R=2\) with this deterministic public ratio-class distinctness requirement.

---

## Why these constraints cover all sabotage-to-brick attacks

Below is an exhaustive list of armer moves whose *only purpose* is to make decap fail even for an honest prover with a valid witness/proof stream, and which constraint blocks each move.

### 1) “Bind to the wrong statement / wrong coins / wrong rep_id”

- **Attack**: armer publishes artifacts under different `(stmt_digest, x, block_id, rep_id)` than the funded address expects, so a later proof for the intended statement produces a different \(\pi\), and decap fails.
- **Blocked by**:
  - **Public input binding** (the well-formedness proof’s statement includes these values),
  - **(C1)** coins recomputation,
  - **(C2–C3)** query/accepting-set correctness tied to those coins and the same statement.

### 2) “Use a different hidden query than the one decap will evaluate”

- **Attack**: armer arms with a different hidden query (different `coeffs` / transcript) so the published hints correspond to a different `q_ring` than the future proof uses.
- **Blocked by**:
  - **(C2)** hidden-query derivation correctness from the canonical transcript,
  - **(C3)** stream-derived query terms and accepting-set correctness,
  - **(D2)** hint correctness with that `q_ring`.

### 3) “Use the wrong shifted accepting set”

- **Attack**: publish a shifted accepting set not equal to `[1-offset, 2-offset]` (or allow a zero element), which breaks the decapper’s division step.
- **Blocked by**:
  - **(C3)** `accepting_set = [offset, offset+1]`,
  - the explicit **nonzero accepting-set** check (matches `arm_ringlwe_lock`).

### 4) “Break packing/embedding so the dot product is wrong”

- **Attack**: armer uses a different embedding of F257 into Goldilocks or a different ring packing, so `coeff0_mul` during decap is not aligned with the armer’s hint/ciphertext math.
- **Blocked by**:
  - **(D1)** packing correctness constraint fixing the embedding and packing convention.

### 5) “Publish random hints not tied to any secret”

- **Attack**: publish arbitrary `h` blocks so that `y_b` at decap is unrelated to any branch signal.
- **Blocked by**:
  - **(D)** hint equation \(h = q\cdot s \pmod{257}\).

### 6) “Encrypt the wrong payload (not the Shamir share)”

- **Attack**: ciphertext encodes garbage bytes or bytes from a different share index, so reconstruction fails.
- **Blocked by**:
  - **(A)** Shamir share correctness from the same `combined_key32`,
  - **(E)** ciphertext correctness for the lock’s designated share index \(i_\ell\).

### 7) “Permute or duplicate shares across locks”

- **Attack**: publish \(R\) locks but map them to the wrong Shamir x-coordinates (or duplicate a coordinate), reducing decap success or breaking reconstruction.
- **Blocked by**:
  - including the lock’s intended share index \(i_\ell\) as a **public input**,
  - **(A)** uses the correct evaluation point,
  - **(E)** enforces that lock \(\ell\) encrypts `share_{i_ℓ}`.

### 8) “Break shapes (wrong `x_len`, `pi_len`, payload length)”

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

- **Nonzero secret constraints** (`s_channel != 0`) are not required for liveness, but are required
  for the division step in decap and for the intended confidentiality story.

---

## What this well-formedness proof does NOT cover

This proof is intentionally minimal for *liveness*, not for full cryptographic security:

- It does **not** prove hardness assumptions (RLWE, etc.).
- It does **not** prove the *eventual* prover has a witness.
- It does **not** prevent confidentiality attacks if you later add per-lock tags/MACs (that would reintroduce per-lock oracles).

