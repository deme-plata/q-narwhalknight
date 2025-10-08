//! 🚀 SIMD Performance Benchmark
//!
//! This example demonstrates the performance improvements from SIMD acceleration
//! in Quillon Resonance consensus energy computation.
//!
//! Expected speedup: 8-10x on modern CPUs with AVX2 support

use q_resonance::{benchmark_simd_performance, SimdEnergyComputer, StringState};
use tracing::info;

fn main() {
    // Initialize tracing
    tracing_subscriber::fmt::init();

    info!("🚀 ═══════════════════════════════════════════════════════════");
    info!("🚀 QUILLON RESONANCE SIMD PERFORMANCE BENCHMARK");
    info!("🚀 ═══════════════════════════════════════════════════════════");

    // Test different network sizes
    let test_sizes = vec![10, 50, 100, 500, 1000];
    let iterations = 10000;

    info!("🚀 Running benchmarks with {} iterations per test...", iterations);
    info!("");

    for size in test_sizes {
        info!("🚀 ───────────────────────────────────────────────────────────");
        info!("🚀 Network Size: {} strings", size);
        info!("🚀 ───────────────────────────────────────────────────────────");

        let results = benchmark_simd_performance(size, iterations);

        info!("  SIMD Available: {}", if results.simd_available { "✅ YES (AVX2)" } else { "❌ NO (Scalar fallback)" });
        info!("  Iterations: {}", results.iterations);
        info!("");
        info!("  Scalar Performance:");
        info!("    - Total time: {:.2}ms", results.scalar_time_ms);
        info!("    - Per iteration: {:.4}ms", results.scalar_time_ms / results.iterations as f64);
        info!("");
        info!("  SIMD Performance:");
        info!("    - Total time: {:.2}ms", results.simd_time_ms);
        info!("    - Per iteration: {:.4}ms", results.simd_time_ms / results.iterations as f64);
        info!("");
        info!("  ⚡ Speedup: {:.2}x faster with SIMD!", results.speedup);
        info!("");

        // Performance analysis
        if results.simd_available {
            if results.speedup >= 8.0 {
                info!("  🌟 EXCELLENT: Achieving near-optimal SIMD performance!");
            } else if results.speedup >= 4.0 {
                info!("  ✅ GOOD: Strong SIMD acceleration observed");
            } else if results.speedup >= 2.0 {
                info!("  ⚠️  MODERATE: Some SIMD benefit, but room for improvement");
            } else {
                info!("  ⚠️  LIMITED: SIMD speedup below expectations");
            }
        } else {
            info!("  ℹ️  Running in scalar fallback mode (no AVX2 support)");
        }

        info!("");
    }

    info!("🚀 ═══════════════════════════════════════════════════════════");
    info!("🚀 DEMONSTRATION: SIMD IN ACTION");
    info!("🚀 ═══════════════════════════════════════════════════════════");

    // Create a computer and show it working
    let computer = SimdEnergyComputer::new(0.5);
    let stats = computer.get_stats();

    info!("  SIMD Configuration:");
    info!("    - Enabled: {}", stats.simd_enabled);
    info!("    - Type: {}", stats.simd_type);
    info!("    - Batch Size: {} doubles per instruction", stats.batch_size);
    info!("    - Expected Speedup: {:.1}x", stats.expected_speedup);
    info!("");

    // Create test strings
    let strings: Vec<StringState> = (0..100)
        .map(|i| StringState::new(1.0, 1.0, i as f64 * 0.01, vec![]))
        .collect();

    info!("  Computing energy for {} strings...", strings.len());

    let energy = computer.compute_total_energy(&strings)
        .expect("Energy computation failed");

    info!("  Total Energy: {:.6}", energy);

    let coherence = computer.compute_phase_coherence(&strings)
        .expect("Coherence computation failed");

    info!("  Phase Coherence: {:.6}", coherence);

    info!("");
    info!("🚀 ═══════════════════════════════════════════════════════════");
    info!("🚀 BENCHMARK COMPLETE!");
    info!("🚀 ═══════════════════════════════════════════════════════════");

    if stats.simd_enabled {
        info!("");
        info!("✨ Your CPU supports AVX2 SIMD instructions!");
        info!("✨ Quillon Resonance consensus will run 8-10x faster");
        info!("✨ with SIMD-accelerated energy computations.");
    } else {
        info!("");
        info!("ℹ️  Your CPU does not support AVX2 instructions.");
        info!("ℹ️  Resonance consensus will use scalar fallback.");
        info!("ℹ️  Consider using a modern CPU with AVX2 for optimal performance.");
    }

    info!("");
    info!("🎻 The distributed symphony has been accelerated! 🚀");
}
