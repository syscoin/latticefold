# Scalable Thm‑4.3 outer FLPCP over `F257` (design note)

This note explains how to make `crates/dpp/src/theorem43.rs` scale to **millions of dR1CS constraints** over the tiny field `F257` **without changing the Thm‑4.3 wrapper**, i.e. while preserving:

- the exact **`(α,β,γ)` interface** with `FlpcpPredicate::MulEq`, and
- the tiny accepting set used by the lock (`A = {1,2}` in `theorem43.rs`).

It also documents the one subtle but critical implementation detail: **evaluation vs systematic** layouts for the square-code witness.

## Background: what `theorem43.rs` assumes from the outer layer

`theorem43.rs` is parameterized by an outer FLPCP backend implementing:

- `Dr1csNpFlpcpSparseApi<F>` (`crates/dpp/src/dr1cs_flpcp.rs`)

and it uses it as follows:

1. The prover creates an outer NP proof `π0 = (z_w || w)` via `flpcp.prove(x, z_w)`.
2. The verifier coins `(idx, lambda, rho, sigma)` determine 3 sparse query vectors
   `q1,q2,q3 = flpcp.queries_for_coins_sparse(idx, lambda, x)` with predicate `MulEq`.
3. The 3 answers on `v0=(x||π0)` are
   \[
   α = \langle q1, v0 \rangle,\quad
   β = \langle q2, v0 \rangle,\quad
   γ = \langle q3, v0 \rangle
   \]
   and the outer check is the standard folded multiplication constraint:
   \[
   α·β = γ.
   \]
4. The Thm‑4.3 “inner gadget” (UV → powering/Vandermonde → Sq) turns this `MulEq` into a
   lockable DPP with a constant accepting set (implemented as `A={1,2}` in our demo).

So the **only thing we must preserve** is: an outer backend that produces the above 3-query `MulEq`
interface over the same field `F257`, but with `ell()` (code length) supporting huge `k`.

## Why RS blocks scaling over `F257`

The current RS instantiation (`RsDr1csNpFlpcpSparse`) is the Reed–Solomon multiplication-code
backend (paper Cor. 4.8 / Thm 4.6 style). It needs a univariate evaluation domain of size `ell`
inside the field, hence `ell <= |F|`.

Over `F257`, this forces `ell <= 257`, and because the square-code side needs roughly `2k` points,
you end up with `k <= ~128` constraints per instance. This is the “domain too large” failure mode
covered by `tests/test_realistic_domain_too_large.rs`.

## The paper’s intended fix: keep the same FLPCP shape, swap the code family

The DPP paper’s small-field scaling is achieved by using a **multiplication code** family with long
block length `ell = Θ(k)` over constant-size fields (Cor. 4.9), instead of Reed–Solomon.

In our repo this is already reflected in the abstraction:

- `MulCode<F>` (`crates/dpp/src/dr1cs_flpcp.rs`)
- `MulCodeDr1csNpFlpcpSparse<F, C: MulCode<F>>`

This is the right “swap point”: we can keep `theorem43.rs` unchanged and only change the concrete
`MulCode<F257>` implementation used for large instances.

## A practical code family to implement now: tensor-power Reed–Solomon

True AG-code instantiations are mathematically clean but heavy to implement.
For engineering a scalable backend quickly, a reasonable “code-family swap” is:

- start with a small RS multiplication code over `F257` of length `n0 <= 257`,
- take a **tensor power** (a.k.a. product code) to obtain length
  \[
  \ell = n_0^t
  \]
  which can exceed millions for small constant `t`.

The tensor product of multiplication codes is again a multiplication code, and the tensor product of
their square codes serves as the square code:

- base: `E0: F^{k0} -> F^{n0}`
- square: `E0* : F^{k0*} -> F^{n0}` with `k0* = 2k0 - 1`
- tensor: `E = E0^{⊗ t}` and `E* = (E0*)^{⊗ t}`

This preserves the **exact same outer FLPCP structure** and still yields `MulEq` answers.

### Parameter sanity

For RS with dimension `k0` and length `n0`, the base relative distance is:

- \[
  δ_0 = 1 - \frac{k_0 - 1}{n_0}
  \]
- square-code relative distance is:
  \[
  δ_0^\star = 1 - \frac{(2k_0 - 1) - 1}{n_0}
           = 1 - \frac{2k_0 - 2}{n_0}
  \]

Tensoring gives:

- \[
  δ^\star = (δ_0^\star)^t
  \]

This decays with `t`, so in practice you choose `t` small and then amplify soundness using the
existing lock repetition / fan-out at the WE layer.

### Tensor-RS has a proof-size blowup (vs AG) — account for it

For the paper’s AG-style multiplication codes (Cor. 4.9), the square message length satisfies
`k* = Θ(k)` with a small constant factor.

For tensor-RS:

- `k = k0^t`
- `k* = (2k0 - 1)^t`
- so
  \[
  \frac{k^\star}{k} = \left(\frac{2k_0-1}{k_0}\right)^t \approx (2)^t.
  \]

This means **large `t` quickly makes the proof too big**. In practice, this strongly suggests
keeping `t ∈ {2,3}`.

### Suggested concrete parameters (good starting points)

Use `n0 = 256` (fits in `F257`, convenient power-of-two base), and pick `t = 3` as a first target:

- `ell = n0^t = 256^3 = 16,777,216`  (coin index space; `log2(ell) = 24` bits)

Then choose `k0` based on the trade-off between:

- block size `k = k0^3` (constraints per chunk),
- proof tail size `k* = (2k0-1)^3` (length of `w` chunk),
- square distance `δ* = (1 - (2k0-2)/n0)^3`.

Ballpark options:

- **`k0=32`**: `k=32,768`, `k*=250,047`, `δ*≈(1-62/256)^3≈0.435`
- **`k0=48`**: `k=110,592`, `k*=857,375`, `δ*≈(1-94/256)^3≈0.253`
- **`k0=64`**: `k=262,144`, `k*=2,048,383`, `δ*≈(1-126/256)^3≈0.131`

Interpretation:

- `k0=32` gives the best per-chunk soundness but more chunks.
- `k0=64` reduces chunk count, but weakens per-chunk soundness and increases the `w` size.
- Regardless, overall soundness is intended to come from **fan-out / repetition** at the WE layer.

## The critical implementation detail: systematic vs evaluation layout for `w`

Our current generic backend (`MulCodeDr1csNpFlpcpSparse`) assumes a **systematic** square code:

- `systematic_positions_star()` returns `k*` indices `idx[j]` such that
  \[
  E^\star(w)[idx[j]] = w[j]\quad\text{for all }w.
  \]

This assumption is convenient for proving: it lets the prover compute the message-vector `w` by
evaluating `E(Az)[idx]` and `E(Bz)[idx]` at systematic square-code positions and setting
`w[j] = E(Az)[idx[j]]·E(Bz)[idx[j]]`.

However, **plain evaluation RS is not systematic** without an extra linear transformation.
The current RS backend (`RsDr1csNpFlpcp*`) avoids this issue by choosing a different proof layout:

- it stores `w` as **claimed square-code evaluations** at designated positions (e.g. the prefix
  `[0..2k)` of the evaluation domain), not as “message coordinates”.

### Therefore: before implementing tensor-RS, we must pick one of two layouts

#### Layout A (recommended): “evaluation witness” layout (RS-style)

Store `w_eval` as a vector of square-code evaluations at a fixed index set `S ⊆ [ell]`:

- proof contains `w_eval[j] = E^\star(w_message)[S[j]]`
- `row_e_star(idx)` evaluates `E^\star` at arbitrary `idx` by a linear combination over the stored
  evaluations (requires an interpolation/extrapolation gadget compatible with the code family)

Pros:
- matches the existing RS approach and avoids implementing systematic encoding
- conceptually closer to “I can store a few evaluations and answer random-index queries”

Cons:
- requires a concrete, efficient “evaluate at random index from stored evaluations” mechanism for
  the chosen multiplication code family (for tensor codes, this becomes multi-dimensional and needs
  careful engineering).

#### Layout B: true systematic square code (generic `MulCode` assumption)

Implement a systematic transform for `E^\star` so that some subset of positions is identity on the
message vector.

Pros:
- fits the current `MulCodeDr1csNpFlpcpSparse` implementation as-is

Cons:
- for tensor-RS, this is nontrivial and likely more work than Layout A

**Design decision:** for a first scalable implementation, prefer **Layout A** unless we have strong
reason to invest in systematic encoding machinery.

## Repo API delta for Layout A (recommended)

Today’s `MulCode` trait is written for Layout B (systematic square-code witness):

- `row_e_star(idx)` is interpreted as coefficients over `F^{k*}` (message coordinates),
- `systematic_positions_star()` is assumed to exist and to witness systematicity.

But `RsDr1csNpFlpcpSparse` (the production-like RS NP backend) already demonstrates Layout A:
it stores `w` as **evaluations at a fixed index set** (the “first `2k` points”), and `lam_2k` provides
the linear functional to evaluate the square code at an arbitrary coin index.

To support Layout A generically (including tensor-RS), the cleanest minimal change is:

- Replace `systematic_positions_star()` with a “witness index set” method:
  - `witness_positions_star() -> Vec<usize>` of length `k_star`
  - semantics: the proof stores `w_eval[j] = E*(w_msg)[ witness_positions_star()[j] ]`
- Interpret `row_e_star(idx)` as coefficients **over the stored witness coordinates** `w_eval`,
  i.e. it returns the unique vector `λ_star(idx) ∈ F^{k_star}` such that:
  \[
  E^\star(w_\text{msg})[idx] \;=\; \langle λ^\star(idx),\; w_\text{eval} \rangle.
  \]

This matches RS exactly (where `witness_positions_star = [0..2k)` and `λ_star(idx)` is the usual
Lagrange coefficient vector), and it avoids implementing any “systematic encoding” transform.

In code terms, this would mean:

- `MulCodeDr1csNpFlpcpSparse::prove_checked()` iterates `idx in witness_positions_star()` and
  fills `w_eval[j] = E(Az)[idx] * E(Bz)[idx]` (using `row_e(idx)`).
- `MulCodeDr1csNpFlpcpSparse::queries_for_coins_sparse()` uses `row_e_star(idx)` to emit the
  `w`-part of `q3` over the stored `w_eval` coordinates, just like RS does today.

If we later implement a true AG multiplication code with systematic encoding, we can either:

- keep the “evaluation witness” interface (still works), or
- add an alternate systematic implementation behind the same interface by choosing
  `witness_positions_star()` to be the systematic subset.

## Tensor‑RS: unambiguous implementation details

This section makes the “tensor trick” precise enough that another agent can implement it without
guesswork. It assumes we follow **Layout A** (store square-code evaluations in the proof and define
`row_e_star(idx)` as coefficients over those stored evaluations).

### Notation and fixed parameters

Choose small constants:

- base field `F = F257`
- base length `n0 = 256` (points are `1..=256` in `F257`)
- base message side `k0` (e.g. 32, 48, 64)
- tensor rank `t ∈ {2,3}` (recommended `t=3`)

Derived sizes:

- constraint block size: `k = k0^t`
- square-message side: `k0* = 2*k0 - 1`
- square proof tail size: `k* = (k0*)^t`
- coin index space: `ell = n0^t`

Define the 1D evaluation points:

- `P[j] = (j+1) ∈ F` for `j=0..n0-1`
- message points: `P_msg = P[0..k0)`
- square-message points: `P_star = P[0..k0*)`

We treat the tensor code as a **product code of 1D evaluation-extension RS**:

- message is the table of values on the grid `P_msg^t` (size `k0^t`)
- codeword is the table of values on the grid `P^t` (size `n0^t`)
- square-message is values on `P_star^t` (size `(2k0-1)^t`)

This matches what `RsDr1csNpFlpcpSparse` already does in 1D:
it treats `y_a = A z` as `f(1..k)` and extends to more points.

### Canonical index decompositions

We will use mixed-radix decompositions to avoid ambiguity.

#### Codeword index (coin) `idx ∈ [0..ell)`

Interpret `idx` in base `n0`:

- `idx = Σ_{d=0..t-1} idx_d * n0^d`, where each `idx_d ∈ [0..n0)`.

Then the `t` evaluation points used by the tensor codeword coordinate are:

- `α_d = P[idx_d]`.

#### Constraint-row index `i ∈ [0..k)`

Interpret `i` in base `k0`:

- `i = Σ_{d=0..t-1} i_d * k0^d`, where each `i_d ∈ [0..k0)`.

This is the coordinate in the message grid `P_msg^t`.

#### Square-message coordinate enumeration for `w_eval`

We need a fixed ordering of the `k* = (k0*)^t` square-grid points, and (crucially) we want the
constraint subgrid `P_msg^t` to be a **prefix** of the `w` vector so the existing `j<k` logic matches
the paper formula “subtract λ·w_prefix”.

Define the stored `w_eval` order as:

1. **Prefix part**: enumerate the subgrid `U_in = [0..k0)^t` in base `k0` order
   (i.e., `u = (u_0,..,u_{t-1})` with `u_d = i_d` from the decomposition of `i` above).
   This produces exactly `k = k0^t` items.
2. **Suffix part**: enumerate the remaining square-grid points
   `U_out = [0..k0*)^t \ U_in` in base `k0*` lexicographic order (any fixed deterministic order is
   fine as long as it is documented and consistently used in `prove` and `row_e_star`).

So a square-grid tuple `u ∈ [0..k0*)^t` maps to a stored index `j(u) ∈ [0..k*)` by:

- if `u ∈ U_in`, then `j(u) = Σ u_d * k0^d` (base `k0`)
- else `j(u) = k + rank_out(u)` where `rank_out(u)` is the position of `u` in the chosen `U_out`
  enumeration.

### 1D Lagrange coefficient vectors (reuse existing code)

You already have the correct 1D primitive in `crates/dpp/src/rs.rs`:

- `ws_k0 = barycentric_weights_consecutive(k0, 1)`
- `ws_k0star = barycentric_weights_consecutive(k0*, 1)`
- `lagrange_coeffs_at(points, ws, alpha)` returns the coefficient vector `λ` such that for any
  values `v[j] = f(points[j])` (j in range), we have `f(alpha) = Σ λ[j]·v[j]`.

For tensor-RS we compute:

- for each dimension `d`, `lam_msg[d] = lagrange_coeffs_at(P_msg, ws_k0, α_d)` (length `k0`)
- for each dimension `d`, `lam_star[d] = lagrange_coeffs_at(P_star, ws_k0star, α_d)` (length `k0*`)

### Tensor coefficient formulas

Given a constraint index `i` with digits `i_d` (base `k0`), the coefficient of row `i` in the tensor
row-functional for `E(·)[idx]` is:

\[
\mathrm{coefE}(idx,i) \;=\; \prod_{d=0}^{t-1} \mathrm{lam\_msg}[d][i_d].
\]

Given a square-grid tuple `u` with digits `u_d` (base `k0*`), the coefficient of that square-grid
location in `E^\star(·)[idx]` is:

\[
\mathrm{coefE^\star}(idx,u) \;=\; \prod_{d=0}^{t-1} \mathrm{lam\_star}[d][u_d].
\]

When expressed over the stored `w_eval` vector, the coefficient for stored index `j` is just the
same value, but placed at the permuted position `j(u)` according to the ordering above.

### Prover algorithm for one constraint block (build `π0 = z_w || w_eval`)

Inputs:

- public `x` (length `l`)
- witness `z_w` (length `n_total - l`)
- sparse dR1CS instance with exactly `k = k0^t` constraints (this is why chunking is mandatory)

Steps:

1. Compute `y_a = A·z`, `y_b = B·z`, `y_c = C·z` as usual (length `k`), without materializing
   `z=(x||z_w)` (same pattern as `RsDr1csNpFlpcpSparse::prove`).
2. Reshape `y_a` and `y_b` into `t`-dimensional arrays of side `k0` using the base-`k0` digit
   decomposition (i.e., `y_a[i]` is at coordinate `(i_0,..,i_{t-1})`).
3. Extend each tensor from side `k0` to side `k0* = 2k0-1` **along each dimension**, using the
   existing 1D extrapolation primitive:

   - For any 1D line of length `k0` at consecutive points `1..k0`, call
     `extrapolate_consecutive_next_block(line)` to get values on `k0+1..2k0` (length `k0`),
     and then **truncate the last element** to keep only `k0-1` new points.
   - After processing all lines for dimension 0, the tensor side becomes `k0*` in that dimension.
   - Repeat for dimensions 1..t-1.

   After `t` passes you obtain `y_a_star` and `y_b_star` defined on the full square grid
   `[0..k0*)^t` (size `k*`).

4. Compute square evaluations pointwise:

   - for each square-grid tuple `u`, set `w_eval[j(u)] = y_a_star[u] * y_b_star[u]`.

5. Output proof:

   - `π0 = z_w || w_eval`.

Notes:

- This is the direct tensor generalization of RS’s “compute y_a on k points, extrapolate to ~2k,
  multiply pointwise to get w”.
- The cost is dominated by the tensor extension step; with `t=3` and `k0 ∈ {32,48,64}`, this is
  large but still a straightforward nested-loop + 1D extrapolation kernel.

### Verifier query construction for fixed coins `(idx, lambda)` (3 sparse queries)

The outer query construction follows the RS formula already implemented in
`RsDr1csNpFlpcpSparse::queries_for_coins_sparse`, but with tensor coefficients.

Given:

- coin `idx ∈ [0..ell)` and `λ ∈ F`
- `lam_msg[d]` and `lam_star[d]` computed as above

Define the 3 query vectors over `v = (x || z_w || w_eval)`:

1. `q1` encodes `E(Az)[idx]`:

   - For each constraint row `i ∈ [0..k)`, weight the sparse row `A_i` by `coefE(idx,i)` and sum:
     \[
     q_{A,z} = \sum_{i=0}^{k-1} \mathrm{coefE}(idx,i)\cdot A_i
     \]
   - Then map `z` indices into `v` indices exactly like the RS NP backend does.

2. `q2` encodes `E(Bz)[idx]` similarly.

3. `q3` encodes the folded square-code expression:

   \[
   γ = E^\star(w)[idx] + λ\cdot E^\star(Cz - w_{\mathrm{prefix}} \| 0)[idx]
   \]

   Implement it in two parts:

   - `x/z_w` part: uses the square-code coefficients restricted to the in-subgrid `U_in`:
     \[
     q_{C,z} = λ \cdot \sum_{i=0}^{k-1} \mathrm{coefE^\star}(idx, u(i)) \cdot C_i
     \]
     where `u(i)` is the `t`-tuple of base-`k0` digits of `i` (so `u(i) ∈ U_in`).

   - `w_eval` part: for each stored square coordinate `j(u)`:
     \[
     q_{w}[j(u)] =
     \begin{cases}
       \mathrm{coefE^\star}(idx,u)\cdot (1-λ) & \text{if } u \in U_{in} \\
       \mathrm{coefE^\star}(idx,u)           & \text{if } u \in U_{out}
     \end{cases}
     \]
     and then shift these indices into the `v` vector’s proof tail region (after `x||z_w`), exactly
     like RS does for its `base + j`.

Finally, return `(q1,q2,q3)` with `FlpcpPredicate::MulEq`.

This construction is “unambiguous” because every coefficient is either:

- a product of 1D Lagrange coefficients (`coefE` / `coefE*`), and
- placed into `w_eval` using the explicit prefix/suffix ordering above.

## Chunking requirement

All multiplication-code instantiations in this pipeline require:

- `code.dim_k() == inst.k()`

For tensor-RS, `dim_k()` is `k0^t`, so a large SP1-sized dR1CS with `K` constraints must be:

- chunked into blocks of size `k_block = k0^t` (with padding for the last chunk), and
- locked/armed per chunk, then combined via repetition/fan-out at the WE layer.

This is not optional: it is the mechanism that lets `ell` and the code family stay fixed while scaling
to arbitrary `K`.

## Chunking across armers (soundness vs reliability)

For very large `K` (hundreds of millions or more), chunking and repetition must be handled with
a clear separation between **soundness amplification** and **reliability** (decap failure).

Important correction:

- **Thm‑4.3 has perfect completeness** in our pipeline (we accept the full tiny set `{1,2}`),
  so there is no inherent “c≈1/2” completeness issue to fix at the DPP layer.
- Therefore **OR‑within‑chunk is not needed for completeness**.

Where OR‑within‑chunk *is* useful is **reliability**, not soundness:

- If the LWE/AEAD decapsulation has a small failure probability `p_fail`,
  then using `R` redundant attempts per chunk reduces chunk failure to
  `p_fail^R`, and overall failure to `B·p_fail^R` (union bound).
- This is a **reliability knob** only; it does **not** improve soundness.
- In fact, OR‑within‑chunk **hurts soundness** locally: per‑chunk cheating
  probability becomes `1 - (1-s)^R ≈ R·s` for small `s`.

Soundness amplification should still be done with independent checks (fan‑out),
combined with **AND** (or threshold) across those checks.

Implementation detail: if chunking is used, the outer backend should include a
**block selector** in the coins `(idx, lambda, block_id)` so that:

- the query always touches the same `z_w` region, and
- only the **selected** `w^(block_id)` region is touched.

This avoids duplicating `z_w` per chunk and preserves streaming / parallel decap.

### Soundness budgeting across chunks and armers (how many “locks” for 128-bit)

Let:

- `s` = per-check soundness error (cheating probability) for one Thm‑4.3 lock instance on a false statement.
  (This constant depends on the outer FLPCP’s square-code distance `δ*` *and* the inner gadget.)
- `B` = number of chunks (blocks) covering the full statement, so `K ≈ B · k_chunk`.
- `N` = number of independent armers (each contributes independent hidden-query checks).
- `R` = number of independent checks per `(armer, chunk)` used for *soundness amplification*.

If verification/decap requires **all chunks** to be consistent (AND across chunks), then a simple
union bound gives a conservative global soundness target:

\[
\Pr[\text{false accept}] \;\lesssim\; (N\cdot B)\cdot s^R.
\]

To target `2^-128` global soundness, it suffices to choose:

\[
R \;\ge\; \frac{128 + \log_2(NB)}{-\log_2(s)}.
\]

Interpretation:

- The “~310 locks” number comes from taking `s≈0.75` and `NB≈1`:
  `R ≈ 128 / 0.415 ≈ 308`.
- If you already have a built-in fan-out of `F` independent checks per armer (e.g. `F=40`), then
  “armings per chunk” is roughly `ceil(R / F)`.

Important: the above `R` is for **soundness**, not for **reliability**.
Reliability redundancy (OR-within-chunk) should be budgeted separately using `p_fail` (see above),
because OR-within-chunk can worsen soundness if used as a soundness amplifier.

## What does *not* change

- `crates/dpp/src/theorem43.rs` stays unchanged.
- The outer predicate stays `FlpcpPredicate::MulEq`.
- The “tiny accepting set” lock stays `A={1,2}`.
- The arm-before-proof flow remains the same (hidden sparse query stored as “toxic waste” in the
  artifact).

## Recommended next step (implementation work items)

1. Decide Layout A vs Layout B (above). For tensor-RS, start with Layout A.
2. Add a concrete `MulCode<F257>` implementation `TensorRsMulCode257`:
   - exposes `row_e(idx)` / `row_e_star(idx)` as row-functionals for the tensor code
   - supports very large `ell = n0^t` without requiring `ell <= |F|`
3. Add a block/chunk wrapper around the dR1CS instance builder so each `MulCodeDr1csNpFlpcpSparse`
   sees exactly `k = dim_k()` constraints.

