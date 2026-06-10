use criterion::{criterion_group, criterion_main, Criterion};
use q_cache::{hierarchical_cache::HierarchicalCache, CacheConfig};
use q_types::{Vertex, VertexId};
use tokio::runtime::Runtime;

fn bench_cache_performance(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let config = CacheConfig {
        target_tps: 100_000.0, // Phase 2 target
        ..CacheConfig::default()
    };

    c.bench_function("hierarchical_cache_performance", |b| {
        b.to_async(&rt).iter(|| async {
            let cache = HierarchicalCache::new(config.clone()).await.unwrap();
            let vertex_id = VertexId::new([1u8; 32]);
            let vertex = Vertex::default();

            // Benchmark cache operations
            let _ = cache.put_vertex(&vertex_id, vertex).await;
            let _ = cache.get_vertex(&vertex_id).await;
        })
    });
}

criterion_group!(benches, bench_cache_performance);
criterion_main!(benches);
