# WE hint leakage: why “readable `s·coeffs` + tiny accepting set” is fatal

This note records a **structural** failure mode in the current “tiny-field WE gate” packaging
design, to prevent reintroducing it via future refactors.

## Setting

We consider any scheme that:

- Publishes a **readable** masked coefficient vector of the form
  \[
    d = s \cdot c \in \mathbb{F}^n
  \]
  where \(s\) is a per-channel secret scalar (e.g. `s_channels_mod257`) and \(c\) is some
  deterministic (statement-bound or coin-bound) coefficient vector; and
- Relies on a **tiny accepting set** for a derived value, typically of size 2 (e.g. `{1,2}` or
  `{c,c+1}`), so that decapsulation can “recover candidates” by dividing by each accepting element
  and intersecting across repetitions.

In the latticefold codebase, this is exactly the pattern of publishing `s * (coeff_mu, coeff_nu,
Sq_power_sum_coeffs...)` in compressed form.

## Why it breaks (high level)

If the public package contains *enough* of the masked vector `d = s·c` in a basis that admits a
public evaluation procedure \(E(\cdot)\) such that:

- For many public “probe points” \(u\), the evaluation \(E(c;u)\) lies in a **tiny known set**
  (e.g. \(\{0,1\}\)), and
- The map is linear in coefficients, so \(E(d;u) = s \cdot E(c;u)\),

then a passive attacker can recover \(s\) from the public package alone by probing \(E(d;u)\) at
many \(u\) and extracting the unique nonzero value (or the common ratio across nonzero probes).

This is not a “parameter tuning” issue: once `d` is readable and the evaluation map exists, the
attack is **information-theoretic** (no witness, no oracle, no brute force).

## Concrete instance: power-sum / indicator Sq coefficients over F257

The current `Sq` construction in `crates/dpp/src/theorem43.rs` uses power sums
\(c_i = -\sum_{\lambda\in U}\lambda^i\) so that the polynomial
\[
  S(u) = \sum_{i=1}^{p-1} c_i u^i
\]
acts like an indicator on \(U \subseteq \mathbb{F}_p^\*\), i.e. \(S(u)\in\{0,1\}\) for all
nonzero \(u\) (up to the scheme’s affine shift).

If we publish *all* masked coefficients \(d_i = s \cdot c_i\), then for every probe point \(u\):
\[
  T(u) := \sum_i d_i u^i = s \cdot S(u) \in \{0,s\}.
\]
Therefore **every nonzero evaluation equals exactly \(s\)**. The unit test
`test_scaled_sq_power_sums_leak_scalar_via_indicator_eval` demonstrates this directly.

The key point is that the attacker never needs the witness-dependent \(u\): they can choose probe
points \(u\) themselves because the coefficients are public.

## Why “simple blinding” doesn’t fix it

- **Changing `{1,2}` to `{c,c+1}`** addresses *bricking/disambiguation* under intersection-based
  decoding, but it does not address the above leakage: the leakage happens before any accepting-set
  disambiguation because it recovers `s` from the published coefficients alone.
- **Multiplying coefficients by a public factor** (e.g. publishing `(r*s)*c_i` and also publishing
  `r` or using an accepting set scaled by `r`) does not help: the same evaluation recovers `r*s`,
  and `r` can be divided out if it is public.
- **Sparsifying the published coefficients** does not help if the evaluation map still exists on
  enough probes to isolate the scale. (In this specific Sq/power-sum case, publishing the full
  vector is what makes the indicator evaluation exact; partial leakage can still be catastrophic
  depending on what is published.)

## Design rule (what we must not do)

Do **not** publish any public artifact from which an attacker can reconstruct a full masked
indicator/power-sum coefficient vector (or any equivalent representation that supports public
probing with tiny-range outputs).

Concretely, for the current Theorem-4.3/Sq gadget, this means:

- Do **not** publish `s * (c_1..c_{p-1})` in any reversible encoding (including packed blocks).
- Do **not** publish any alternative basis or linear sketch that is invertible (or sufficiently
  informative) to recover the masked monomial coefficients.

## Preferred direction

Use PVUGC-style **hiding hints + KDF-derived keys**, where the public hint material does not expose
a readable masked coefficient vector in a basis that supports public probing of a tiny-range
function. This is the direction tracked in the leakage-fix plan.

