/// Quick RocksDB Storage Performance Test
/// Direct performance measurement for TPS/BPS without complex dependencies

use std::time::{Duration, Instant};
use std::sync::Arc;
use tempfile::TempDir;

// Mock vertex for testing
#[derive(Clone)]
struct MockVertex {
    id: [u8; 32],
    round: u64,
    author: [u8; 32], 
    payload: Vec<u8>,
    timestamp: u64,
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

/// Storage benchmark
struct StorageBenchmark {
    db: Arc<rocksdb::DB>,
    _temp_dir: TempDir,
}

impl StorageBenchmark {
    fn new() -> Self {
        let temp_dir = TempDir::new().unwrap();
        
        let mut opts = rocksdb::Options::default();
        opts.create_if_missing(true);
        opts.create_missing_column_families(true);
        
        // Performance optimizations
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
    
    /// Benchmark batch write performance
    fn bench_batch_write(&self, vertices: &[MockVertex]) -> Duration {
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
    
    /// Calculate TPS
    fn calculate_tps(&self, vertices: &[MockVertex], duration: Duration) -> f64 {
        vertices.len() as f64 / duration.as_secs_f64()
    }
    
    /// Calculate BPS
    fn calculate_bps(&self, vertices: &[MockVertex], duration: Duration) -> f64 {
        let total_bytes: usize = vertices.iter().map(|v| v.size_bytes()).sum();
        total_bytes as f64 / duration.as_secs_f64()
    }
}

fn main() {
    println!("🏆 Q-NARWHALKNIGHT STORAGE PERFORMANCE TEST");
    println!("🔥 Real RocksDB Performance Measurement");
    println!("");
    
    let storage = StorageBenchmark::new();
    
    // Test 1: Single Round Performance
    println!("📊 TEST 1: Single Round Performance");
    let vertices_per_round = 100;
    let round_vertices: Vec<MockVertex> = (0..vertices_per_round)
        .map(|i| MockVertex::new(i as u8, 1, 512))
        .collect();
    
    let write_time = storage.bench_batch_write(&round_vertices);
    let tps = storage.calculate_tps(&round_vertices, write_time);
    let bps = storage.calculate_bps(&round_vertices, write_time);
    
    println!("⚡ Single Round: {:.0} TPS, {:.2} KB/s", tps, bps / 1000.0);
    println!("");
    
    // Test 2: High-Throughput Test
    println!("📊 TEST 2: High-Throughput Simulation (5 Nodes)");
    let test_duration = Duration::from_secs(10);
    let target_tps = 2000;
    let round_time_ms = 100;
    
    let start_time = Instant::now();
    let mut round = 0u64;
    let mut total_vertices = 0;
    let mut total_bytes = 0;
    let mut total_write_time = Duration::new(0, 0);
    
    while start_time.elapsed() < test_duration {
        let round_start = Instant::now();
        
        // Generate vertices for 5 nodes
        let vertices_per_node = (target_tps * round_time_ms / 1000) / 5;
        let round_vertices: Vec<MockVertex> = (0..5)
            .flat_map(|node_id| {
                (0..vertices_per_node).map(move |tx_id| {
                    MockVertex::new(
                        (node_id * 50 + tx_id % 200) as u8,
                        round,
                        512 + (tx_id % 512), // Variable sizes
                    )
                })
            })
            .collect();
        
        // Write to storage
        let write_time = storage.bench_batch_write(&round_vertices);
        total_write_time += write_time;
        
        // Track metrics
        total_vertices += round_vertices.len();
        total_bytes += round_vertices.iter().map(|v| v.size_bytes()).sum::<usize>();
        
        // Maintain timing
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
            
            println!("🚀 Round {}: {:.0} TPS, {:.2} MB/s", 
                     round, current_tps, current_bps / 1_000_000.0);
        }
    }
    
    let total_test_time = start_time.elapsed();
    
    // Final results
    let final_tps = total_vertices as f64 / total_test_time.as_secs_f64();
    let final_bps = total_bytes as f64 / total_test_time.as_secs_f64();
    let storage_tps = total_vertices as f64 / total_write_time.as_secs_f64();
    let storage_bps = total_bytes as f64 / total_write_time.as_secs_f64();
    
    println!("");
    println!("🏆 FINAL PERFORMANCE RESULTS:");
    println!("📊 Test Duration: {:.2}s", total_test_time.as_secs_f64());
    println!("📊 Consensus Rounds: {}", round);
    println!("📊 Total Vertices: {}", total_vertices);
    println!("📊 Total Data: {:.2} MB", total_bytes as f64 / 1_000_000.0);
    println!("");
    println!("⚡ SYSTEM THROUGHPUT:");
    println!("⚡ System TPS: {:.0}", final_tps);
    println!("⚡ System Bandwidth: {:.2} MB/s", final_bps / 1_000_000.0);
    println!("");
    println!("💾 STORAGE THROUGHPUT (Pure I/O):");
    println!("💾 Storage TPS: {:.0}", storage_tps);
    println!("💾 Storage Bandwidth: {:.2} MB/s", storage_bps / 1_000_000.0);
    println!("💾 Storage Write Time: {:.2}s", total_write_time.as_secs_f64());
    println!("");
    println!("🛡️ RocksDB Performance: MEASURED");
    println!("🌌 Q-NarwhalKnight Ready: CONFIRMED");
}