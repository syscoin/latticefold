//! Microbenchmarks for ring-64 arithmetic (Goldilocks).
//!
//! This compares the coefficient-form `mul` implementations for:
//! - `cyclotomic_rings::rings::GoldilocksRing64`
//!
//! Run:
//! - `cargo bench -p latticefold-plus --bench ring64_mul`
//! - quick sanity: `cargo bench -p latticefold-plus --bench ring64_mul -- --test`

use ark_std::UniformRand;
use criterion::{black_box, criterion_group, criterion_main, BatchSize, Criterion, Throughput};
use cyclotomic_rings::rings::GoldilocksRing64;
use rand::{rngs::StdRng, SeedableRng};
use stark_rings::Ring;

fn bench_ring64_mul(c: &mut Criterion) {
    let mut group = c.benchmark_group("ring64/mul");
    group.sample_size(20);

    // Pre-generate inputs so we measure mul, not RNG.
    const N: usize = 1 << 12;
    group.throughput(Throughput::Elements(N as u64));

    group.bench_function("GoldilocksRing64", |b| {
        b.iter_batched(
            || {
                let mut rng = StdRng::seed_from_u64(0x604D_64);
                let mut pairs = Vec::with_capacity(N);
                for _ in 0..N {
                    pairs.push((
                        GoldilocksRing64::rand(&mut rng),
                        GoldilocksRing64::rand(&mut rng),
                    ));
                }
                pairs
            },
            |pairs| {
                let mut acc = GoldilocksRing64::ZERO;
                for (a, b) in pairs {
                    acc += black_box(a) * black_box(b);
                }
                black_box(acc)
            },
            BatchSize::SmallInput,
        );
    });

    group.finish();
}

criterion_group!(benches, bench_ring64_mul);
criterion_main!(benches);

