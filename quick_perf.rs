use std::time::Instant;
use std::collections::HashMap;

fn main() {
    println!("🚀 Q-NarwhalKnight Quick Performance Test");
    println!("========================================");
    
    // Hash performance test
    let start = Instant::now();
    let mut map = HashMap::new();
    for i in 0..100_000 {
        map.insert(format!("key_{}", i), i * 2);
    }
    let hash_time = start.elapsed();
    println!("✅ HashMap Insert (100k): {:?}", hash_time);
    
    // Memory allocation test
    let start = Instant::now();
    let mut vec = Vec::new();
    for i in 0..1_000_000 {
        vec.push(i);
    }
    let alloc_time = start.elapsed();
    println!("✅ Vector Allocation (1M): {:?}", alloc_time);
    
    // String operations test
    let start = Instant::now();
    let mut result = String::new();
    for i in 0..10_000 {
        result.push_str(&format!("item_{}", i));
    }
    let string_time = start.elapsed();
    println!("✅ String Operations (10k): {:?}", string_time);
    
    println!("\n🎯 Quick Performance Summary:");
    println!("   - HashMap ops: {:.2} μs per insert", hash_time.as_micros() as f64 / 100_000.0);
    println!("   - Vector alloc: {:.2} ns per item", alloc_time.as_nanos() as f64 / 1_000_000.0);
    println!("   - String ops: {:.2} μs per concat", string_time.as_micros() as f64 / 10_000.0);
}