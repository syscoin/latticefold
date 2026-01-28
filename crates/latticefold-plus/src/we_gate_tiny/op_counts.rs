use std::cell::RefCell;

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

thread_local! {
    static TINY_COUNTS: RefCell<TinyCmOpCounts> = RefCell::new(TinyCmOpCounts::default());
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
    TINY_COUNTS.with(|rc| f(&mut *rc.borrow_mut()));
}

#[inline]
pub(crate) fn tiny_cm_counts_reset() {
    if !tiny_counts_on() {
        return;
    }
    TINY_COUNTS.with(|rc| *rc.borrow_mut() = TinyCmOpCounts::default());
}

#[inline]
pub(crate) fn tiny_cm_counts_take() -> TinyCmOpCounts {
    if !tiny_counts_on() {
        return TinyCmOpCounts::default();
    }
    TINY_COUNTS.with(|rc| {
        let cur = *rc.borrow();
        *rc.borrow_mut() = TinyCmOpCounts::default();
        cur
    })
}

