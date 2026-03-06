# H12 `R_cap` specification

This note freezes the intended outer seed-capsule relation for the GPT PRO
construction:

- outer capsule: `WE_AADP.Enc(I_cap, K)`
- hidden state: `sealed_hidden_state = AEAD_K(hidden_h11_state)`
- inner decap: current H11 path unchanged after `K` is released

The goal of this note is to pin the correct target **before** implementing the
AADP backend from `eprint 2026/175`.

## What H12 is

H12 is **not**:

- WE of the full current tiny-gate verifier relation
- WE of the full `25k`-poison-block residual gate
- WE of the current `g_err` computation "as is"

H12 **is**:

- a new, separate, arm-before-proof relation `R_cap`
- used only to release a seed `K`
- with the current anchored H11 filter left intact and run only after `K`
  opens the hidden H11 state

## Three different "block" notions

There are three different objects that must not be conflated.

1. FLPCP / poison blocks

- Count is currently on the order of `25k`
- Each corresponds to one outer local `MulEq` check
- These are the objects mixed by the current residual gate `g_err`

2. Packed proof blocks

- Implemented in `lockable_ringlwe.rs` with `PACK_D = 64`
- These are 64-wide storage / streaming chunks of the actual proof payload `pi0`
- Sparse hints are stored against these packed block indices

3. AADP variables / gates

- `v` counts witness variables in the outer AADP relation
- `g` counts nonlinear constraints in the outer AADP relation

The `25k -> 390` observation only says:

- a random projection vector over `25k` poison-block residuals can be stored in
  `ceil(25k / 64) ~= 390` packed blocks

It does **not** mean the current outer relation has `g = 390`.

## Why current `g_err` is not the outer capsule

Current `g_err` first forms all per-poison-block residuals

`err_b = gamma_b - alpha_b * beta_b`

for all `25k` poison blocks, and only afterwards mixes them.

So if wrapped directly in AADP, the nonlinear count is still approximately the
number of poison blocks:

- `g_current ~= 25k`

That is far outside the sub-GB regime.

## Correct H12 split

The correct split is:

1. `R_cap`

- proves knowledge of a future local proof view for a tiny, arm-before-proof
  schedule
- releases only `K`

2. hidden H11 state

- contains at minimum the current `ct_ubits`
- optionally also `accepting_set`, `offset`, or other degree-0 hidden state

3. current H11 decap

- recomputes the existing anchored H11 equations after `K` opens the hidden
  state
- keeps the current selectivity argument unchanged

## Recommended capsule unit: one logical lock

To keep `g_cap` small, the outer capsule should be **per logical lock**
(`share_index` / `lock_j`), not per whole package.

Reason:

- current package can have `P` logical locks and `R` reps per logical lock
- a package-wide capsule would push `g_cap` toward `P * R`
- a per-logical-lock capsule keeps `g_cap` near `R_cap`

Recommended structure:

- one `seed_capsule_j` per logical lock `j`
- one `sealed_hidden_state_j` per logical lock `j`
- derive all per-rep H11 unlock material for logical lock `j` from `K_j`

## Public instance for `R_cap`

For logical lock `j`, define:

`I_cap,j = (stmt_digest, manifest_hash, logical_lock_id, cap_schedule_j, x0=1)`

where:

- `stmt_digest` is the existing bound statement digest
- `manifest_hash` commits to public package metadata and domain separation
- `logical_lock_id` is the share index / lock id
- `cap_schedule_j` is a public, armer-fixed local schedule
- `x0 = 1` is the non-homogeneous constant wire required by AADP projective safety

`cap_schedule_j` must be derivable at arming time from public statement data and
armer randomness. It must not depend on the future proof values.

## Witness for `R_cap`

Let `T_cap,j` be the union of future proof coordinates read by the capsule
schedule for logical lock `j`.

The witness is:

`W_cap,j = pi0 | T_cap,j`

plus any local helper variables needed for linear recomputation inside the AADP
constraint system.

Important:

- `W_cap,j` is **not** the whole future proof
- `W_cap,j` is **not** the whole current tiny-gate witness
- `W_cap,j` is the local future proof slice for the capsule schedule only

## Correct nonlinear budget target

The outer relation must be designed so that:

- `g_cap ~= R_cap`

with target values:

- `R_cap = 16`, `24`, or `32`

This is the GPT PRO regime that keeps the AADP ciphertext near the `~1 GB`
target.

## Size formula

From `eprint 2026/175`, Section 4.10:

`size ~= v * (2g + 1)^2 * |F|`

Using `|F| = 32 bytes`:

- `g = 16` and `v ~= 25k` gives about `0.87 GB`
- `g = 24` and `v ~= 13k` gives about `1.0 GB`
- `g = 32` and `v ~= 7.4k` gives about `1.0 GB`

So H12 is only plausible if the outer relation has:

- small `g_cap` at the relation level, not only after postprocessing
- moderate `v_cap`

## Two rejected outer relations

### Rejected: full tiny-gate relation

This is much smaller than Pipe-v2, but it is still not the GPT PRO capsule.

### Rejected: full current poison-block residual relation

This relation still has roughly one nonlinear multiplication per poison block:

- `g ~= 25k`

Even with optimistic `v`, this lands in the hundreds of terabytes to petabyte
range, not the sub-GB regime.

## Candidate relation family for H12

The intended capsule relation is:

`R_cap,j(I_cap,j, W_cap,j) = 1`

where `cap_schedule_j` consists of `R_cap` selected local outer checks, and for
each selected check `r` the relation:

1. linearly recomputes the relevant local proof responses from `W_cap,j`
2. enforces one nonlinear local `MulEq`-style check
3. repeats this over `R_cap` selected checks

This is the only relation family that matches the GPT PRO message and keeps the
outer capsule small enough.

## Critical backend caveat

With the current backend, the natural source of local checks is the Theorem-4.3
outer FLPCP interface:

- public schedule via `derive_public_coins_from_stmt(c_stmt, block_id, rep_id)`
- local answers `(alpha, beta, gamma)`
- local check `alpha * beta = gamma`

However, the current H11-exported `anchor_basis_hints` are **not** themselves
the full capsule witness interface.

Reason:

- current H11 artifacts were designed for the inner anchored decap path, not
  for compiling a standalone outer AADP relation
- the capsule needs an explicit local-view export including:
  - sparse `q1` / `q2` proof terms
  - sparse `q3` proof terms
  - `q1` / `q2` witness-block coefficients
  - `q3` witness-block coefficients
  - the selected `w_eval` block span in `pi0`

Therefore:

- H12 cannot safely reuse the current H11 hints verbatim as `R_cap`
- H12 needs a **new capsule schedule builder** that exports the full local witness
  data required to verify each selected local `MulEq` check

This hook now exists in code:

- `dpp::theorem43::Theorem43CapsuleLocalCheckSurface`
- `Theorem43Dpp::export_capsule_local_check_surface_from_stmt(...)`
- `Theorem43Dpp::export_capsule_schedule_from_stmt(...)`
- `Dr1csNpFlpcpSparseApi::export_q3_w_eval_terms(...)`

and for the current file-backed backend:

- `FileBackedChunkedMulCodeDr1csNpFlpcpSparse::export_q3_w_eval_terms(...)`

What is still missing is the AADP relation/compiler itself and the H12 package
wiring in the OneProof path.

## Required new builder

Add a new builder at the DPP / tiny-lock boundary that, for each selected
capsule local check, exports:

- public part:
  - `block_id`
  - `rep_id`
  - `coins`
  - manifest / domain separators
  - any fixed coefficients needed by the local verifier

- witness layout description:
  - exact touched `pi0` coordinates for that selected local check
  - exact linear forms producing the local answers

- relation-level check:
  - one `MulEq`-style nonlinear gate per selected local check

The builder must be arm-before-proof:

- public schedule fixed from statement + armer randomness
- witness values supplied only later from the actual proof

## Current recommendation

Freeze the capsule design as:

- **outer unit:** per logical lock
- **outer repetition count:** `R_cap` independent from current H11 repetition
  count if necessary
- **outer relation:** selected local `MulEq` checks only
- **inner relation:** current H11 unchanged

Do **not** implement AADP over either of the two rejected relations above.

## Immediate implementation order

1. Add a stats path that reports:
   - selected local checks
   - touched packed `pi0` coordinates
   - estimated `v_cap`
   - exact `g_cap`
   - estimated AADP size
2. Implement AADP `WE.Enc / WE.Dec` for `R_cap`
3. Finally wire:
   - `seed_capsule_j`
   - `sealed_hidden_state_j`
   - H11 rehydration + current decap path

