use std::sync::atomic::{AtomicU64, Ordering};

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct TinyCmOpCounts {
    pub ring_add: u64,
    pub ring_sub: u64,
    pub ring_scale: u64,
    pub ring_mul_negacyclic: u64,
    pub ring_eq: u64,
    pub lc_to_var: u64,
    pub scalar_add: u64,
    pub scalar_sub: u64,
    pub scalar_mul: u64,
    pub scalar_mul_const: u64,
    pub scalar_sub_const: u64,
    pub scalar_pow_table: u64,
    pub eq_eval_vars: u64,
    pub short_challenge_from_bytes: u64,
    pub ct_psi_mul_ring: u64,
}

#[derive(Debug, Default)]
struct TinyCmOpCountsAtomic {
    ring_add: AtomicU64,
    ring_sub: AtomicU64,
    ring_scale: AtomicU64,
    ring_mul_negacyclic: AtomicU64,
    ring_eq: AtomicU64,
    lc_to_var: AtomicU64,
    scalar_add: AtomicU64,
    scalar_sub: AtomicU64,
    scalar_mul: AtomicU64,
    scalar_mul_const: AtomicU64,
    scalar_sub_const: AtomicU64,
    scalar_pow_table: AtomicU64,
    eq_eval_vars: AtomicU64,
    short_challenge_from_bytes: AtomicU64,
    ct_psi_mul_ring: AtomicU64,
}

static TINY_COUNTS: TinyCmOpCountsAtomic = TinyCmOpCountsAtomic {
    ring_add: AtomicU64::new(0),
    ring_sub: AtomicU64::new(0),
    ring_scale: AtomicU64::new(0),
    ring_mul_negacyclic: AtomicU64::new(0),
    ring_eq: AtomicU64::new(0),
    lc_to_var: AtomicU64::new(0),
    scalar_add: AtomicU64::new(0),
    scalar_sub: AtomicU64::new(0),
    scalar_mul: AtomicU64::new(0),
    scalar_mul_const: AtomicU64::new(0),
    scalar_sub_const: AtomicU64::new(0),
    scalar_pow_table: AtomicU64::new(0),
    eq_eval_vars: AtomicU64::new(0),
    short_challenge_from_bytes: AtomicU64::new(0),
    ct_psi_mul_ring: AtomicU64::new(0),
};

impl TinyCmOpCountsAtomic {
    #[inline]
    fn add(&self, d: TinyCmOpCounts) {
        // Only enabled under LFP_WE_GATE_OPMIX, so this can be a bit verbose.
        if d.ring_add != 0 {
            self.ring_add.fetch_add(d.ring_add, Ordering::Relaxed);
        }
        if d.ring_sub != 0 {
            self.ring_sub.fetch_add(d.ring_sub, Ordering::Relaxed);
        }
        if d.ring_scale != 0 {
            self.ring_scale.fetch_add(d.ring_scale, Ordering::Relaxed);
        }
        if d.ring_mul_negacyclic != 0 {
            self.ring_mul_negacyclic.fetch_add(d.ring_mul_negacyclic, Ordering::Relaxed);
        }
        if d.ring_eq != 0 {
            self.ring_eq.fetch_add(d.ring_eq, Ordering::Relaxed);
        }
        if d.lc_to_var != 0 {
            self.lc_to_var.fetch_add(d.lc_to_var, Ordering::Relaxed);
        }
        if d.scalar_add != 0 {
            self.scalar_add.fetch_add(d.scalar_add, Ordering::Relaxed);
        }
        if d.scalar_sub != 0 {
            self.scalar_sub.fetch_add(d.scalar_sub, Ordering::Relaxed);
        }
        if d.scalar_mul != 0 {
            self.scalar_mul.fetch_add(d.scalar_mul, Ordering::Relaxed);
        }
        if d.scalar_mul_const != 0 {
            self.scalar_mul_const.fetch_add(d.scalar_mul_const, Ordering::Relaxed);
        }
        if d.scalar_sub_const != 0 {
            self.scalar_sub_const.fetch_add(d.scalar_sub_const, Ordering::Relaxed);
        }
        if d.scalar_pow_table != 0 {
            self.scalar_pow_table.fetch_add(d.scalar_pow_table, Ordering::Relaxed);
        }
        if d.eq_eval_vars != 0 {
            self.eq_eval_vars.fetch_add(d.eq_eval_vars, Ordering::Relaxed);
        }
        if d.short_challenge_from_bytes != 0 {
            self.short_challenge_from_bytes
                .fetch_add(d.short_challenge_from_bytes, Ordering::Relaxed);
        }
        if d.ct_psi_mul_ring != 0 {
            self.ct_psi_mul_ring.fetch_add(d.ct_psi_mul_ring, Ordering::Relaxed);
        }
    }

    #[inline]
    fn reset(&self) {
        self.ring_add.store(0, Ordering::Relaxed);
        self.ring_sub.store(0, Ordering::Relaxed);
        self.ring_scale.store(0, Ordering::Relaxed);
        self.ring_mul_negacyclic.store(0, Ordering::Relaxed);
        self.ring_eq.store(0, Ordering::Relaxed);
        self.lc_to_var.store(0, Ordering::Relaxed);
        self.scalar_add.store(0, Ordering::Relaxed);
        self.scalar_sub.store(0, Ordering::Relaxed);
        self.scalar_mul.store(0, Ordering::Relaxed);
        self.scalar_mul_const.store(0, Ordering::Relaxed);
        self.scalar_sub_const.store(0, Ordering::Relaxed);
        self.scalar_pow_table.store(0, Ordering::Relaxed);
        self.eq_eval_vars.store(0, Ordering::Relaxed);
        self.short_challenge_from_bytes.store(0, Ordering::Relaxed);
        self.ct_psi_mul_ring.store(0, Ordering::Relaxed);
    }

    #[inline]
    fn take(&self) -> TinyCmOpCounts {
        TinyCmOpCounts {
            ring_add: self.ring_add.swap(0, Ordering::Relaxed),
            ring_sub: self.ring_sub.swap(0, Ordering::Relaxed),
            ring_scale: self.ring_scale.swap(0, Ordering::Relaxed),
            ring_mul_negacyclic: self.ring_mul_negacyclic.swap(0, Ordering::Relaxed),
            ring_eq: self.ring_eq.swap(0, Ordering::Relaxed),
            lc_to_var: self.lc_to_var.swap(0, Ordering::Relaxed),
            scalar_add: self.scalar_add.swap(0, Ordering::Relaxed),
            scalar_sub: self.scalar_sub.swap(0, Ordering::Relaxed),
            scalar_mul: self.scalar_mul.swap(0, Ordering::Relaxed),
            scalar_mul_const: self.scalar_mul_const.swap(0, Ordering::Relaxed),
            scalar_sub_const: self.scalar_sub_const.swap(0, Ordering::Relaxed),
            scalar_pow_table: self.scalar_pow_table.swap(0, Ordering::Relaxed),
            eq_eval_vars: self.eq_eval_vars.swap(0, Ordering::Relaxed),
            short_challenge_from_bytes: self.short_challenge_from_bytes.swap(0, Ordering::Relaxed),
            ct_psi_mul_ring: self.ct_psi_mul_ring.swap(0, Ordering::Relaxed),
        }
    }
}

#[inline]
fn tiny_counts_on() -> bool {
    // Keep it aligned with the existing op-mix flag (same as big gate).
    std::env::var("LFP_WE_GATE_OPMIX").is_ok()
}

#[inline]
pub(crate) fn tiny_cm_bump<F: FnOnce(&mut TinyCmOpCounts)>(f: F) {
    if !tiny_counts_on() {
        return;
    }
    let mut d = TinyCmOpCounts::default();
    f(&mut d);
    TINY_COUNTS.add(d);
}

#[inline]
pub(crate) fn tiny_cm_counts_reset() {
    if !tiny_counts_on() {
        return;
    }
    TINY_COUNTS.reset();
}

#[inline]
pub(crate) fn tiny_cm_counts_take() -> TinyCmOpCounts {
    if !tiny_counts_on() {
        return TinyCmOpCounts::default();
    }
    TINY_COUNTS.take()
}

