/// Performance Regression Detection Demo
///
/// Demonstrates how to use the Q-NarwhalKnight performance regression detection system
/// for continuous monitoring and automated alerting on performance degradations.
use anyhow::Result;
use chrono::Utc;
use q_benchmarks::performance_regression::{
    collect_system_metrics, BuildConfig, ConsensusPerformanceMetrics, GpuPerformanceMetrics,
    PerformanceMetrics, PerformanceRegressionDetector, RegressionConfig, SystemMetrics,
    ZkPerformanceMetrics,
};

#[tokio::main]
async fn main() -> Result<()> {
    println!("🚀 Q-NarwhalKnight Performance Regression Detection Demo");
    println!("======================================================");

    // Initialize regression detector with custom configuration
    let config = RegressionConfig {
        regression_threshold: 0.10,   // 10% threshold for demo
        improvement_threshold: -0.05, // 5% improvement detection
        baseline_window: 3,           // Use 3 samples for baseline (normally 10)
        history_limit: 100,
        alert_cooldown_hours: 1, // 1 hour cooldown for demo
    };

    let mut detector = PerformanceRegressionDetector::new(config);

    // Simulate baseline performance measurements (good performance)
    println!("\n📊 Recording baseline performance measurements...");
    for i in 1..=3 {
        let baseline_metrics = create_demo_metrics(
            &format!("commit_baseline_{}", i),
            2500.0,             // Good TPS
            85.0,               // Good Groth16 proving time
            8.5,                // Good verification time
            Some((45.0, 15.2)), // Good GPU performance (FFT time, speedup)
        )?;

        detector.record_metrics(baseline_metrics)?;
        println!("  ✅ Recorded baseline sample {}/3", i);
    }

    // Simulate performance regression (degraded performance)
    println!("\n⚠️  Simulating performance regression...");
    let regression_metrics = create_demo_metrics(
        "commit_regression_1",
        2100.0,             // 16% TPS drop - should trigger alert
        105.0,              // 23% proving time increase - should trigger alert
        9.2,                // 8% verification time increase - minor
        Some((58.0, 12.8)), // GPU regression - should trigger alert
    )?;

    detector.record_metrics(regression_metrics)?;

    // Perform regression analysis
    println!("\n🔍 Performing regression analysis...");
    let analysis = detector.analyze_regressions()?;

    // Display results
    println!("\n📋 REGRESSION ANALYSIS RESULTS");
    println!("================================");
    println!(
        "Commit Range: {} → {}",
        analysis.commit_range.0, analysis.commit_range.1
    );
    println!("Total Regressions: {}", analysis.detected_regressions.len());
    println!(
        "Performance Trend: {:.1}%",
        analysis.performance_summary.overall_performance_trend * 100.0
    );

    if !analysis.detected_regressions.is_empty() {
        println!("\n🚨 DETECTED REGRESSIONS:");
        for regression in &analysis.detected_regressions {
            let severity_emoji = match regression.severity {
                q_benchmarks::performance_regression::RegressionSeverity::Critical => "🚨",
                q_benchmarks::performance_regression::RegressionSeverity::Major => "⚠️ ",
                q_benchmarks::performance_regression::RegressionSeverity::Moderate => "📊",
                q_benchmarks::performance_regression::RegressionSeverity::Minor => "ℹ️ ",
            };

            println!(
                "  {} {} ({}): {:.1}% regression",
                severity_emoji,
                regression.metric_name,
                regression.metric_category,
                regression.regression_percent
            );
            println!(
                "     Baseline: {:.3} → Current: {:.3}",
                regression.baseline_value, regression.current_value
            );
        }
    } else {
        println!("\n✅ No regressions detected - system performance is stable!");
    }

    // Display recommendations
    println!("\n💡 RECOMMENDATIONS:");
    for (i, recommendation) in analysis.recommendations.iter().enumerate() {
        println!("  {}. {}", i + 1, recommendation);
    }

    // Demonstrate CSV export functionality
    println!("\n📁 Exporting performance data...");
    detector.export_metrics_csv("performance_regression_demo.csv")?;
    println!("  ✅ Performance data exported to: performance_regression_demo.csv");

    // Simulate recovery (improved performance)
    println!("\n🔧 Simulating performance recovery...");
    let recovery_metrics = create_demo_metrics(
        "commit_recovery_1",
        2650.0,             // Better than baseline TPS
        78.0,               // Better than baseline proving time
        8.1,                // Better than baseline verification
        Some((42.0, 16.5)), // Better than baseline GPU performance
    )?;

    detector.record_metrics(recovery_metrics)?;

    // Final analysis after recovery
    let final_analysis = detector.analyze_regressions()?;
    println!("\n🎉 RECOVERY ANALYSIS:");
    println!(
        "Total Active Regressions: {}",
        final_analysis.detected_regressions.len()
    );
    println!(
        "Performance Improvements: {}",
        final_analysis.performance_summary.improvements_detected
    );
    println!(
        "Overall Trend: {:.1}%",
        final_analysis.performance_summary.overall_performance_trend * 100.0
    );

    #[cfg(feature = "ci-integration")]
    {
        println!("\n🏗️  CI/CD INTEGRATION DEMO:");
        let ci_report = q_benchmarks::performance_regression::ci_integration::generate_ci_report(
            &final_analysis,
        );
        println!("{}", ci_report);

        let exit_code =
            q_benchmarks::performance_regression::ci_integration::get_ci_exit_code(&final_analysis);
        println!("CI Exit Code: {}", exit_code);
    }

    println!("\n✅ Performance regression detection demo completed!");
    println!("\n📚 Integration Tips:");
    println!("  • Run this in CI/CD pipelines after each build");
    println!("  • Set up automated alerts for critical regressions");
    println!("  • Use git bisect to identify problematic commits");
    println!("  • Monitor trends over time for proactive optimization");
    println!("  • Adjust thresholds based on your performance requirements");

    Ok(())
}

/// Create demo performance metrics with specified values
fn create_demo_metrics(
    commit: &str,
    tps: f64,
    groth16_proving_ms: f64,
    groth16_verification_ms: f64,
    gpu_data: Option<(f64, f64)>, // (fft_time_ms, speedup)
) -> Result<PerformanceMetrics> {
    let zk_metrics = ZkPerformanceMetrics {
        groth16_proving_time_ms: groth16_proving_ms,
        groth16_verification_time_ms: groth16_verification_ms,
        groth16_proof_size_bytes: 192, // Typical Groth16 proof size
        plonk_setup_time_ms: 450.0,
        plonk_proving_time_ms: groth16_proving_ms * 1.2, // PLONK typically slower
        plonk_verification_time_ms: groth16_verification_ms * 0.9,
        constraint_count: 1_000_000,
        witness_generation_time_ms: groth16_proving_ms * 0.3,
        circuit_compilation_time_ms: 1200.0,
        peak_memory_mb: 1500.0,
        average_memory_mb: 1200.0,
    };

    let consensus_metrics = ConsensusPerformanceMetrics {
        vertex_processing_time_ms: 12.5,
        consensus_round_time_ms: 2300.0,
        finality_time_ms: 4600.0,
        transactions_per_second: tps,
        vertices_per_second: 42.0,
        network_latency_ms: 145.0,
        dag_memory_usage_mb: 856.0,
        state_storage_size_mb: 2340.0,
    };

    let gpu_metrics = gpu_data.map(|(fft_time, speedup)| GpuPerformanceMetrics {
        gpu_fft_time_ms: fft_time,
        gpu_field_ops_time_ms: fft_time * 0.7,
        gpu_fri_commitment_time_ms: fft_time * 1.8,
        gpu_utilization_percent: 87.0,
        gpu_memory_usage_mb: 3200.0,
        gpu_power_usage_watts: 225.0,
        cpu_vs_gpu_speedup: speedup,
        parallel_efficiency: 0.82,
    });

    let system_metrics = collect_system_metrics()?;

    let build_config = BuildConfig {
        profile: "release".to_string(),
        features: vec!["gpu-acceleration".to_string(), "parallel".to_string()],
        rust_version: "1.70.0".to_string(),
        target: "x86_64-unknown-linux-gnu".to_string(),
    };

    Ok(PerformanceMetrics {
        timestamp: Utc::now(),
        git_commit: commit.to_string(),
        build_config,
        zk_metrics,
        consensus_metrics,
        gpu_metrics,
        system_metrics,
    })
}
