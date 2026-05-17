/// RocksDB Storage Performance Benchmark
/// Measures real disk I/O, transaction throughput, and consensus data storage performance

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};
use std::time::{Duration, Instant};
use std::sync::Arc;
use tempfile::TempDir;

// Simple mock types for benchmarking (avoiding dependency issues)
#[derive(Clone)]
pub struct MockVertex {
    pub id: [u8; 32],
    pub round: u64,
    pub author: [u8; 32],
    pub payload: Vec<u8>,
    pub timestamp: u64,
}

impl MockVertex {
    fn new(id: u8, round: u64, payload_size: usize) -> Self {
        let payload: Vec<u8> = (0..payload_size).map(|i| ((id as usize + i) % 256) as u8).collect();
        
        Self {
            id: [id; 32],
            round,
            author: [(id + 1) % 255; 32],
            payload,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_millis() as u64,
        }
    }
    
    fn size_bytes(&self) -> usize {
        32 + 8 + 32 + self.payload.len() + 8
    }
    
    fn serialize(&self) -> Vec<u8> {
        let mut data = Vec::with_capacity(self.size_bytes());
        data.extend_from_slice(&self.id);
        data.extend_from_slice(&self.round.to_be_bytes());
        data.extend_from_slice(&self.author);
        data.extend_from_slice(&self.payload.len().to_be_bytes());
        data.extend_from_slice(&self.payload);
        data.extend_from_slice(&self.timestamp.to_be_bytes());
        data
    }
}

/// High-performance RocksDB storage benchmark
pub struct StorageBenchmark {
    db: Arc<rocksdb::DB>,
    _temp_dir: TempDir,
}

impl StorageBenchmark {
    pub fn new() -> Self {
        let temp_dir = TempDir::new().unwrap();
        
        let mut opts = rocksdb::Options::default();
        opts.create_if_missing(true);
        opts.create_missing_column_families(true);
        
        // Performance optimizations for benchmarking
        opts.set_max_background_jobs(8);
        opts.set_write_buffer_size(64 * 1024 * 1024); // 64MB
        opts.set_max_write_buffer_number(4);
        opts.set_target_file_size_base(64 * 1024 * 1024);
        opts.set_compression_type(rocksdb::DBCompressionType::Lz4);
        
        // Create column families
        let cfs = vec![
            rocksdb::ColumnFamilyDescriptor::new("vertices", opts.clone()),
            rocksdb::ColumnFamilyDescriptor::new("payloads", opts.clone()),
            rocksdb::ColumnFamilyDescriptor::new("blocks", opts.clone()),
        ];
        
        let db = rocksdb::DB::open_cf_descriptors(&opts, temp_dir.path(), cfs).unwrap();
        
        Self {
            db: Arc::new(db),
            _temp_dir: temp_dir,
        }
    }
    
    /// Benchmark single vertex write performance
    pub fn bench_single_write(&self, vertex: &MockVertex) -> Duration {
        let start = Instant::now();
        
        let key = format!("vertex_{:08}_{}", vertex.round, hex::encode(&vertex.id[..8]));
        let data = vertex.serialize();
        
        if let Some(cf) = self.db.cf_handle("vertices") {
            self.db.put_cf(&cf, key.as_bytes(), &data).unwrap();
        }
        
        start.elapsed()
    }
    
    /// Benchmark batch write performance
    pub fn bench_batch_write(&self, vertices: &[MockVertex]) -> Duration {
        let start = Instant::now();
        
        let mut batch = rocksdb::WriteBatch::default();
        
        for vertex in vertices {
            let key = format!("vertex_{:08}_{}", vertex.round, hex::encode(&vertex.id[..8]));
            let data = vertex.serialize();
            
            if let Some(cf) = self.db.cf_handle("vertices") {
                batch.put_cf(&cf, key.as_bytes(), data);
            }
        }
        
        self.db.write(batch).unwrap();
        
        start.elapsed()
    }
    
    /// Benchmark read performance
    pub fn bench_read(&self, key: &str) -> Duration {
        let start = Instant::now();
        
        if let Some(cf) = self.db.cf_handle("vertices") {
            let _result = self.db.get_cf(&cf, key.as_bytes()).unwrap();
        }
        
        start.elapsed()
    }
    
    /// Calculate throughput metrics
    pub fn calculate_tps(&self, vertices: &[MockVertex], duration: Duration) -> f64 {
        vertices.len() as f64 / duration.as_secs_f64()
    }
    
    /// Calculate bandwidth metrics
    pub fn calculate_bps(&self, vertices: &[MockVertex], duration: Duration) -> f64 {
        let total_bytes: usize = vertices.iter().map(|v| v.size_bytes()).sum();
        total_bytes as f64 / duration.as_secs_f64()
    }
}

/// Benchmark suite for RocksDB performance
fn benchmark_storage_performance(c: &mut Criterion) {
    let storage = StorageBenchmark::new();
    
    // Single write performance
    let mut single_write_group = c.benchmark_group("single_write");
    
    for payload_size in [256, 512, 1024, 2048] {
        single_write_group.throughput(Throughput::Bytes(payload_size as u64));
        single_write_group.bench_with_input(
            BenchmarkId::new("vertex_write", payload_size),
            &payload_size,
            |b, &size| {
                b.iter(|| {
                    let vertex = MockVertex::new(42, 1, size);
                    black_box(storage.bench_single_write(&vertex))
                });
            }
        );
    }
    single_write_group.finish();
    
    // Batch write performance  
    let mut batch_write_group = c.benchmark_group("batch_write");
    
    for batch_size in [10, 50, 100, 500] {
        let vertices: Vec<MockVertex> = (0..batch_size)
            .map(|i| MockVertex::new(i as u8, 1, 512))
            .collect();
        
        let total_bytes: usize = vertices.iter().map(|v| v.size_bytes()).sum();
        batch_write_group.throughput(Throughput::Bytes(total_bytes as u64));
        
        batch_write_group.bench_with_input(
            BenchmarkId::new("vertex_batch", batch_size),
            &vertices,
            |b, vertices| {
                b.iter(|| {
                    black_box(storage.bench_batch_write(vertices))
                });
            }
        );
    }
    batch_write_group.finish();
}

/// High-throughput consensus simulation benchmark
fn benchmark_consensus_throughput(c: &mut Criterion) {
    let storage = StorageBenchmark::new();
    
    let mut consensus_group = c.benchmark_group("consensus_simulation");
    consensus_group.measurement_time(Duration::from_secs(10));
    
    // Simulate 5-node consensus with high transaction volume
    for tps_target in [1000, 2500, 5000, 10000] {
        let vertices_per_round = tps_target / 20; // 50ms rounds
        
        consensus_group.bench_with_input(
            BenchmarkId::new("consensus_tps", tps_target),
            &vertices_per_round,
            |b, &vertex_count| {
                b.iter_custom(|iters| {
                    let mut total_time = Duration::new(0, 0);
                    
                    for round in 0..iters {
                        // Generate vertices for this round (5 nodes)
                        let round_vertices: Vec<MockVertex> = (0..5)
                            .flat_map(|node_id| {
                                (0..vertex_count / 5).map(move |tx_id| {
                                    MockVertex::new(
                                        (node_id * 50 + tx_id) as u8,
                                        round,
                                        512 // 512 byte transactions
                                    )
                                })
                            })
                            .collect();
                        
                        // Measure consensus round processing time
                        let round_start = Instant::now();
                        let _duration = storage.bench_batch_write(&round_vertices);
                        let round_time = round_start.elapsed();
                        
                        total_time += round_time;
                        
                        // Calculate and report TPS for this round
                        let tps = storage.calculate_tps(&round_vertices, round_time);
                        let bps = storage.calculate_bps(&round_vertices, round_time);
                        
                        if round % 100 == 0 {
                            eprintln!("Round {}: {:.0} TPS, {:.2} MB/s", round, tps, bps / 1_000_000.0);
                        }
                    }
                    
                    total_time
                });
            }
        );
    }
    consensus_group.finish();
}

/// Measure real-world consensus scenario performance
fn benchmark_realistic_consensus(c: &mut Criterion) {
    let storage = StorageBenchmark::new();
    
    let mut realistic_group = c.benchmark_group("realistic_consensus");
    realistic_group.measurement_time(Duration::from_secs(15));
    
    // Simulate realistic Q-NarwhalKnight consensus workload
    realistic_group.bench_function("5_node_dag_knight", |b| {
        b.iter_custom(|_iters| {
            let test_duration = Duration::from_secs(5); // 5 second test
            let target_tps = 3000; // Realistic target
            let round_time_ms = 100; // 100ms consensus rounds
            
            let start_time = Instant::now();
            let end_time = start_time + test_duration;
            
            let mut round = 0u64;
            let mut total_vertices = 0;
            let mut total_bytes = 0;
            
            while Instant::now() < end_time {
                let round_start = Instant::now();
                
                // Generate vertices for 5 nodes
                let vertices_per_node = (target_tps * round_time_ms / 1000) / 5;
                let round_vertices: Vec<MockVertex> = (0..5)
                    .flat_map(|node_id| {
                        (0..vertices_per_node).map(move |tx_id| {
                            MockVertex::new(
                                (node_id * 50 + tx_id % 200) as u8,
                                round,
                                512 + (tx_id % 1024), // Variable transaction sizes
                            )
                        })
                    })
                    .collect();
                
                // Process consensus round
                let _write_time = storage.bench_batch_write(&round_vertices);
                
                // Track metrics
                total_vertices += round_vertices.len();
                total_bytes += round_vertices.iter().map(|v| v.size_bytes()).sum::<usize>();
                
                // Maintain target round timing
                let round_elapsed = round_start.elapsed();
                if round_elapsed < Duration::from_millis(round_time_ms) {
                    std::thread::sleep(Duration::from_millis(round_time_ms) - round_elapsed);
                }
                
                round += 1;
                
                // Progress reporting
                if round % 20 == 0 {
                    let elapsed = start_time.elapsed();
                    let current_tps = total_vertices as f64 / elapsed.as_secs_f64();
                    let current_bps = total_bytes as f64 / elapsed.as_secs_f64();
                    
                    eprintln!("🚀 DAG-Knight Round {}: {:.0} TPS, {:.2} MB/s", 
                             round, current_tps, current_bps / 1_000_000.0);
                }
            }
            
            let total_time = start_time.elapsed();
            
            // Final performance report
            let final_tps = total_vertices as f64 / total_time.as_secs_f64();
            let final_bps = total_bytes as f64 / total_time.as_secs_f64();
            
            println!("\n🏆 REALISTIC CONSENSUS PERFORMANCE RESULTS:");
            println!("📊 Duration: {:.2}s", total_time.as_secs_f64());
            println!("📊 Rounds: {}", round);
            println!("📊 Total Vertices: {}", total_vertices);
            println!("📊 Total Data: {:.2} MB", total_bytes as f64 / 1_000_000.0);
            println!("⚡ Achieved TPS: {:.0}", final_tps);
            println!("⚡ Achieved Bandwidth: {:.2} MB/s", final_bps / 1_000_000.0);
            println!("⚡ Average Round Time: {:.1}ms", total_time.as_millis() as f64 / round as f64);
            
            total_time
        });
    });
    
    realistic_group.finish();
}

criterion_group!(
    storage_benches,
    benchmark_storage_performance,
    benchmark_consensus_throughput, 
    benchmark_realistic_consensus
);

criterion_main!(storage_benches);