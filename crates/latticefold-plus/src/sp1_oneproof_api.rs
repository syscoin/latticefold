//! API wrapper for the `examples/lf_plus_sp1_oneproof.rs` harness.
//!
//! This exists so downstream crates (like PVUGC) can call the SP1→LF+→WE-gate path **in-process**
//! without spawning binaries or scraping logs.
//!
//! This is research plumbing and intentionally returns only a small structured summary.

#![cfg(feature = "we_gate")]

use cyclotomic_rings::rings::{GetPoseidonParams, GoldilocksPoseidonConfig as PC};
use cyclotomic_rings::rings::GoldilocksRing64 as R;
use ark_ff::PrimeField;
use latticefold::commitment::AjtaiCommitmentScheme;
use latticefold::transcript::poseidon::F257;
use latticefold::transcript::Transcript;
use latticefold::transcript::bytes::field_to_bytes_le_fixed;
use stark_rings::PolyRing;
use stark_rings_linalg::SparseMatrix;

use sha2::{Digest, Sha256};
use std::sync::Arc;
use std::time::Instant;

use rand::{rngs::StdRng, SeedableRng};

use crate::lin::LinearizedVerify;
use crate::lockable_ringlwe::RingLweParams;
use crate::utils::maybe_print_rss;
use crate::we_statement::{encode_public_x, we_statement_hash_lf_plus, LFP_WE_GATE_DIGEST_V1};

/// Structured output for downstream wiring (no stdout parsing).
#[derive(Clone, Debug)]
pub struct Sp1OneProofWeGateOutput {
    pub stmt_digest: [u8; 32],
    pub lock_coin_seed: [u8; 32],
    /// The decapsulated 32-byte key derived from the proof.
    ///
    /// This is a research harness convenience: the underlying RingLWE lock has two branches, so
    /// we embed a short self-authenticating tag into the lock payload and pick the unique branch
    /// whose payload validates.
    pub decapped_key: [u8; 32],
}

fn hex32(bytes: &[u8; 32]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(64);
    for &b in bytes {
        out.push(HEX[(b >> 4) as usize] as char);
        out.push(HEX[(b & 0x0f) as usize] as char);
    }
    out
}

/// Run the SP1 oneproof + WE gate path from on-disk artifacts.
///
/// Inputs are the same formats the example consumes:
/// - `r1lf_path`: SP1 shrink verifier `.r1lf`
/// - `witness_path`: SP1 witness bundle `.bundle` ("SP1W")
pub fn run_sp1_oneproof_we_gate_from_files(
    r1lf_path: &str,
    witness_path: &str,
) -> Result<Sp1OneProofWeGateOutput, String> {
    // Keep defaults aligned with the example (but avoid printing / panics).
    let chunk_size: usize = std::env::var("CHUNK_SIZE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1 << 20);
    let pad_cols_to_multiple_of: usize = std::env::var("PAD_COLS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(256);

    let cache = crate::sp1_r1lf::open_sp1_r1lf_chunk_cache::<R>(
        r1lf_path,
        chunk_size,
        pad_cols_to_multiple_of,
    )
    .map_err(|e| format!("open_sp1_r1lf_chunk_cache: {e}"))?;

    // Materialize full (A,B,C) as SparseMatrix<F> (constant-coeff) by concatenating chunk rows.
    type F = <R as PolyRing>::BaseRing;
    let total_rows = cache.num_chunks * chunk_size;
    let mut a_rows: Vec<Vec<(F, usize)>> = Vec::with_capacity(total_rows);
    let mut b_rows: Vec<Vec<(F, usize)>> = Vec::with_capacity(total_rows);
    let mut c_rows: Vec<Vec<(F, usize)>> = Vec::with_capacity(total_rows);
    for chunk_idx in 0..cache.num_chunks {
        let [a, b, c] = cache.read_chunk(chunk_idx).map_err(|e| format!("read_chunk: {e}"))?;
        let conv = |m: stark_rings_linalg::SparseMatrix<F>| m.coeffs;
        let (ar, br, cr) = (conv(a), conv(b), conv(c));
        a_rows.extend(ar);
        b_rows.extend(br);
        c_rows.extend(cr);
    }
    let m_a = SparseMatrix::<F> { nrows: total_rows, ncols: cache.ncols, coeffs: a_rows };
    let m_b = SparseMatrix::<F> { nrows: total_rows, ncols: cache.ncols, coeffs: b_rows };
    let m_c = SparseMatrix::<F> { nrows: total_rows, ncols: cache.ncols, coeffs: c_rows };

    let bundle = crate::sp1_witness_io::load_sp1_witness_any(witness_path, cache.stats.num_vars)
        .map_err(|e| format!("load_sp1_witness_any: {e}"))?;
    let (w_u64, _base_len, _aux_len) = (bundle.witness, bundle.base_len, bundle.aux_len);

    // Map u64 witness -> Goldilocks base field scalars once (centered embedding mod p_bb).
    let p_bb = cache.stats.p_bb;
    let mut w_host: Vec<F> = Vec::with_capacity(w_u64.len());
    for &x in &w_u64 {
        w_host.push(babybear_u64_to_centered_host::<F>(x, p_bb));
    }

    // Public inputs from the SP1 export are witness[1..=l_pub].
    let l_pub = cache.stats.num_public;
    if l_pub == 0 {
        return Err("SP1 R1LF exports num_public=0; statement binding requires public inputs".to_string());
    }
    if w_host.len() < 1 + l_pub {
        return Err(format!(
            "witness too short for declared public inputs: w_len={} need_at_least={}",
            w_host.len(),
            1 + l_pub
        ));
    }
    type BFSmall = <<R as PolyRing>::BaseRing as ark_ff::Field>::BasePrimeField;
    let mut public_inputs: Vec<BFSmall> = w_host[1..1 + l_pub].to_vec();
    // Debug knob parity with the historical harness: mutate a public input and ensure things fail.
    if !public_inputs.is_empty()
        && std::env::var("FLIP_PUBLIC_INPUT0").ok().as_deref() == Some("1")
    {
        public_inputs[0] = <BFSmall as ark_ff::Field>::ONE - public_inputs[0];
    }

    // Mirror the example: deterministic Ajtai commitment scheme (system parameter) with prefix exposure.
    let kappa_expose: usize = 8;
    let kappa_random: usize = 8;
    let kappa: usize = kappa_expose + kappa_random;
    const AJTAI_SEED: [u8; 32] = *b"LFP_SP1_AJTAI_SEED_V1_0000000000";
    let ajtai = AjtaiCommitmentScheme::<R>::seeded_with_exposed_prefix(
        b"lf_plus_ajtai",
        AJTAI_SEED,
        kappa,
        cache.ncols,
        kappa_expose,
        1, // expose witness columns 1..=8 (skip shared ONE at column 0)
    );

    // Build ComR1CS from f0 (base witness) and keep matrices Arc for the prover.
    let mut f0 = w_host.clone();
    f0.truncate(cache.stats.num_vars);
    let f0: Arc<Vec<F>> = Arc::new(f0);
    let r1cs = latticefold::arith::r1cs::R1CS::<F> { l: l_pub, A: m_a, B: m_b, C: m_c };
    let cr1cs =
        crate::r1cs::ComR1CSBase::<R>::from_f0_seeded_base(r1cs, f0, l_pub, &ajtai);
    let m0 = cr1cs.x.matrices_arc_base();

    // WE params derived from SP1 cache + chosen kappa.
    let we_params = crate::sp1_r1lf::sp1_default_we_params_for_r1lf_cache::<R>(
        &cache,
        kappa as u64,
        m0.len() as u64,
    )
    .map_err(|e| format!("sp1_default_we_params_for_r1lf_cache: {e}"))?;
    let dparams = crate::rgchk::DecompParameters {
        b: (we_params.decomp_b as u128),
        k: (we_params.k as usize),
        l: (we_params.l as usize),
    };
    let lin_params = crate::lin::LinParameters { kappa, decomp: dparams };

    // Choose decomposition base B > sqrt(q), rounded up to a power of 2 (same as example).
    fn isqrt_u128(x: u128) -> u128 {
        let mut lo: u128 = 0;
        let mut hi: u128 = 1u128 << 64;
        while lo + 1 < hi {
            let mid = (lo + hi) >> 1;
            if mid.saturating_mul(mid) <= x {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        lo
    }
    fn next_pow2_u128(x: u128) -> u128 {
        if x <= 1 {
            return 1;
        }
        let p = 128 - (x - 1).leading_zeros() as u32;
        1u128 << p
    }
    let q_u128: u128 = <F as PrimeField>::MODULUS.0[0] as u128;
    let b_min = isqrt_u128(q_u128) + 1;
    let b_decomp: u128 = next_pow2_u128(b_min);
    let pparams = crate::plus::PlusParameters { lin: lin_params, B: b_decomp };

    // Statement digest binds vk, r1cs digest, gate-id, params, and public inputs.
    if bundle.r1lf_digest != cache.stats.digest {
        return Err(format!(
            "witness bundle r1lf_digest does not match SP1_R1LF cache: bundle=0x{} cache=0x{}",
            hex32(&bundle.r1lf_digest),
            hex32(&cache.stats.digest)
        ));
    }
    let (vk_hash, _committed_values_digest) = bundle.public_inputs;
    let r1cs_digest = cache.stats.digest;
    let stmt_digest =
        we_statement_hash_lf_plus::<R>(vk_hash, r1cs_digest, LFP_WE_GATE_DIGEST_V1, &we_params, &public_inputs);

    // Derive one representative coin seed (j=0) like the example.
    const ARMER_SEED: [u8; 32] = *b"LFP_ARMER_SEED_V1_00000000000000";
    let lock_j: u64 = 0;
    let lock_coin_seed: [u8; 32] = {
        let mut h = Sha256::new();
        h.update(b"LFP_LOCK_COIN_V1");
        h.update(&ARMER_SEED);
        h.update(&stmt_digest);
        h.update(&lock_j.to_le_bytes());
        h.finalize().into()
    };

    // Plus prover.
    let mut prover = crate::plus::PlusProverSparseBase::init_seeded_base(
        ajtai.clone(),
        m0.clone(),
        1,
        pparams.clone(),
        crate::transcript::PoseidonTranscript::empty::<PC>(),
    );
    let proof = prover.prove_sparse_base(std::slice::from_ref(&cr1cs), &public_inputs);

    // Record verifier trace, then:
    // - arm the WE gate *shape* (instance depends only on params + sizes),
    // - compute a satisfying witness assignment for that armed instance,
    // - extract the witness tail for downstream lock scaffolding.
    let _poseidon_cfg = PC::get_poseidon_config();
    let mut rec = crate::recording_transcript::TracePoseidonTranscript::<R>::empty::<PC>();
    for b in &public_inputs {
        rec.absorb_field_element(b);
    }
    for lp in &proof.lproof {
        lp.verify(&mut rec);
    }
    // Skip native prefix binding check only if explicitly requested.
    let skip_prefix_check =
        std::env::var("LFP_SKIP_PREFIX_BINDING_CHECK").ok().as_deref() == Some("1");
    let bind_prefix: &[BFSmall] = if skip_prefix_check {
        &[]
    } else {
        public_inputs
            .get(0..8)
            .ok_or_else(|| "expected at least 8 public inputs for Ajtai prefix exposure".to_string())?
    };
    proof
        .cmproof
        .verify_with_mlen(m0.len(), &mut rec, bind_prefix)
        .map_err(|e| format!("cm proof verify: {e:?}"))?;
    let trace = rec.trace().clone();

    // -------------------------------------------------------------------------
    // Tiny-field (F257) WE gate (Theorem 4.3 path)
    // -------------------------------------------------------------------------
    // Shape cache directory.
    //
    // The shape (constraint matrices) is compiled once and cached on disk.
    // Subsequent calls load the cached shape and only compute the per-proof assignment
    // (lightweight count-only pass, no disk writes).
    let out_dir = {
        let base = std::env::var("LFP_SHAPE_CACHE_DIR")
            .ok()
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|| {
                let mut p = std::env::temp_dir();
                p.push("lfplus_sp1_oneproof_tiny_gate");
                p
            });
        std::fs::create_dir_all(&base).map_err(|e| format!("create shape cache dir failed: {e}"))?;
        base
    };

    let pairs: Vec<(usize, usize)> = vec![(0, 0)];

    // Map statement public inputs into tiny-gate public prefix bytes (F257).
    //
    // IMPORTANT: Transcript public inputs are absorbed as fixed-width base-field byte strings
    // (`prime_field_to_bytes_le_fixed`), i.e. 8 bytes per Goldilocks base-field element.
    //
    // Therefore, the tiny gate's statement public prefix must be these bytes, not the field
    // elements themselves, and certainly not the centered BabyBear embedding.
    // IMPORTANT: these must be the *exact bytes* absorbed into the recorded transcript:
    // `PoseidonTranscript::absorb_field_element` encodes field elements using
    // `field_to_bytes_le_fixed` (fixed width, little-endian).
    //
    // Therefore we derive bytes from the **Goldilocks field elements** `public_inputs`,
    // not from the raw BabyBear u64 witness words.
    let public_inputs_f257: Vec<F257> = public_inputs
        .iter()
        .flat_map(|x| field_to_bytes_le_fixed::<BFSmall>(x).into_iter().map(|b| F257::from(b as u64)))
        .collect();

    let (shape, assignment) = crate::we_gate_arith::build_or_load_we_plus_tiny_dr1cs::<R>(
        &trace,
        &we_params,
        &public_inputs_f257,
        &proof,
        m0.len(),
        &pairs,
        &out_dir,
    )
    .map_err(|e| format!("build_or_load_we_plus_tiny_dr1cs: {e}"))?;

    // Optional sanity: the witness must satisfy the armed instance.
    // This can be expensive for large gates; enable only when debugging.
    if std::env::var("LFP_WE_GATE_CHECK_SAT").ok().as_deref() == Some("1") {
        shape
            .inst
            .check(&assignment)
            .map_err(|e| format!("we gate armed instance not satisfied: {e}"))?;
    }

    // -------------------------------------------------------------------------
    // Ring-LWE streaming decapsulation (Theorem 4.3 canonical path)
    // -------------------------------------------------------------------------
    let public_len = shape.public_len;
    let x = encode_public_x::<F257>(&we_params, &public_inputs_f257);
    if x.len() != public_len {
        return Err(format!(
            "oneproof: bad public x length (x_len={} public_len={})",
            x.len(),
            public_len
        ));
    }
    if assignment.len() < public_len {
        return Err("oneproof: assignment shorter than public_len".to_string());
    }
    let asg_pub = assignment
        .get(0..public_len)
        .ok_or_else(|| "oneproof: assignment shorter than public_len".to_string())?;
    if asg_pub != x.as_slice() {
        return Err("oneproof: satisfying assignment public prefix mismatch".to_string());
    }
    let z_w = &assignment[public_len..];

    let lock_j = 0u64;
    let block_id = 0usize;
    let mut rep_id = 0u64;
    // Deterministic by default; production should set σ>0 and enable reconciliation/rounding.
    let ringlwe_params = RingLweParams::default();
    // Deterministic RNG seed derived from the statement-binding lock coin seed.
    let mut rng = StdRng::from_seed(lock_coin_seed);

    let t_arm = Instant::now();
    // Payload (research harness):
    // We want this API to return a single decapped key (not 2 unauthenticated candidates).
    //
    // So we arm the lock with a payload containing:
    //   payload = tag16 || key32
    // where tag16 = H("LFP_ONEPROOF_TAG_V1" || key32 || stmt_digest)[0..16].
    //
    // After streaming decap, we try both candidates and pick the unique one whose tag validates.
    // This avoids depending on external branch-identification plumbing (e.g. Shamir) in PVUGC.
    // IMPORTANT: `rep_id` may be incremented on retry. `payload` (and therefore the returned
    // decapped key) must be derived *inside* the retry loop so it stays consistent with the
    // lock parameters that were actually armed.
    let lock = loop {
        let key32: [u8; 32] = {
            let mut h = Sha256::new();
            h.update(b"LFP_ONEPROOF_KEY_V1");
            h.update(&lock_coin_seed);
            h.update(&stmt_digest);
            h.update(&lock_j.to_le_bytes());
            h.update(&block_id.to_le_bytes());
            h.update(&rep_id.to_le_bytes());
            h.finalize().into()
        };
        let tag16: [u8; 16] = {
            let mut h = Sha256::new();
            h.update(b"LFP_ONEPROOF_TAG_V1");
            h.update(&key32);
            h.update(&stmt_digest);
            let full: [u8; 32] = h.finalize().into();
            full[0..16].try_into().expect("slice->array")
        };
        let payload: [u8; 48] = {
            let mut p = [0u8; 48];
            p[0..16].copy_from_slice(&tag16);
            p[16..48].copy_from_slice(&key32);
            p
        };
        match crate::we_tiny_lock::arm_lfplus_ringlwe_lock::<R>(
            shape.clone(),
            &we_params,
            &public_inputs_f257,
            stmt_digest,
            ARMER_SEED,
            lock_j,
            block_id,
            rep_id,
            ringlwe_params.clone(),
            &payload,
            &mut rng,
        ) {
            Ok(lock) => break lock,
            Err(e) if e.contains("shifted accepting set contains 0") => {
                rep_id += 1;
                continue;
            }
            Err(e) => return Err(e),
        }
    };
    let prover = crate::we_tiny_lock::we_ringlwe_prover_from_dr1cs::<F257>(
        shape.inst.clone(),
        shape.public_len,
    )?;
    eprintln!(
        "[oneproof] armed ringlwe in {:?}: proof_len={}",
        t_arm.elapsed(),
        prover.proof_len()
    );

    let t_prove = Instant::now();
    maybe_print_rss("oneproof:before_prove_decap_stream");
    let mut st = lock.decap_state(&x)?;
    let mut err: Option<String> = None;
    let tails = prover.stream_pi0_and_collect_tails(
        &x,
        z_w,
        std::slice::from_ref(&lock.coins),
        &mut |chunk| {
            if err.is_some() {
                return;
            }
            if let Err(e) = st.absorb_chunk(chunk) {
                err = Some(e);
            }
        },
    )?;
    if let Some(e) = err {
        return Err(format!("oneproof: stream decap absorb failed: {e}"));
    }
    if tails.len() != 1 {
        return Err(format!("oneproof: expected 1 tail, got {}", tails.len()));
    }
    st.absorb_chunk(&tails[0])?;
    let cands = st.finish_decrypt_candidates()?;
    maybe_print_rss("oneproof:after_prove_decap_stream");
    eprintln!("[oneproof] prove+decap(stream) in {:?}", t_prove.elapsed());
    // With unauthenticated encryption, branch identification is deferred to Shamir reconstruction.
    // The accepting set structure is verified from the lock artifact directly.
    let a0 = lock.accepting_set[0] + lock.offset;
    let a_u64 = a0.into_bigint().as_ref().get(0).copied().unwrap_or(0);
    eprintln!("[oneproof] accepting_set_answer={a_u64}");

    // Select the unique candidate whose tag validates.
    let mut picked: Option<[u8; 32]> = None;
    for cand in &cands {
        if cand.len() != 48 {
            continue;
        }
        let tag = &cand[0..16];
        let key = &cand[16..48];
        let mut h = Sha256::new();
        h.update(b"LFP_ONEPROOF_TAG_V1");
        h.update(key);
        h.update(&stmt_digest);
        let full: [u8; 32] = h.finalize().into();
        if tag == &full[0..16] {
            let arr: [u8; 32] = key.try_into().expect("len checked");
            if picked.is_some() {
                return Err("oneproof: multiple decap candidates validated tag (unexpected)".to_string());
            }
            picked = Some(arr);
        }
    }
    let decapped_key = picked.ok_or_else(|| "oneproof: no decap candidate validated tag".to_string())?;

    // Debug-only: keep deterministic footprints if you need to compare runs.
    let _ = hex32(&stmt_digest);

    // Shape cache is intentionally kept on disk for reuse across invocations.
    // To force a rebuild, delete the cache directory manually or set LFP_SHAPE_CACHE_DIR.

    Ok(Sp1OneProofWeGateOutput {
        stmt_digest,
        lock_coin_seed,
        decapped_key,
    })
}

#[inline]
fn babybear_u64_to_centered_host<F: ark_ff::Field>(x: u64, p_bb: u64) -> F {
    debug_assert!(p_bb > 1);
    let half = p_bb / 2;
    if x > half {
        let neg = p_bb - x;
        -F::from(neg)
    } else {
        F::from(x)
    }
}

