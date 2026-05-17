#!/usr/bin/env cargo run --bin performance_test --
/// Q-NarwhalKnight Performance Test Runner
/// Execute comprehensive 5-node performance testing with configurable parameters

use std::env;
use std::time::Duration;
use anyhow::Result;
use tokio::time::timeout;
use tracing::{info, error, Level};
use tracing_subscriber::{FmtSubscriber, filter::EnvFilter};

// Import our performance testing module
mod performance_test;
use performance_test::{PerformanceTest, PerformanceTestConfig};

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    let subscriber = FmtSubscriber::builder()
        .with_env_filter(EnvFilter::from_default_env().add_directive(Level::INFO.into()))
        .finish();
    tracing::subscriber::set_global_default(subscriber)?;

    info!("🚀 Q-NarwhalKnight Performance Test Runner Starting");

    // Parse command line arguments or use defaults
    let config = parse_test_config();
    
    info!("📊 Test Configuration:");
    info!("  • Nodes: {}", config.num_nodes);
    info!("  • Duration: {}s", config.test_duration_secs);
    info!("  • Target TPS: {:,}", config.target_tps);
    info!("  • Transaction Size: {} bytes", config.transaction_size_bytes);
    info!("  • Batch Size: {}", config.batch_size);

    // Run the performance test suite
    let test_results = run_comprehensive_tests(config.clone()).await?;
    
    // Generate and display results
    generate_final_report(&config, &test_results);
    
    info!("✅ Performance testing completed successfully");
    Ok(())
}

/// Parse command line arguments or use default configuration
fn parse_test_config() -> PerformanceTestConfig {
    let args: Vec<String> = env::args().collect();
    
    let mut config = PerformanceTestConfig::default();
    
    // Parse simple command line arguments
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--nodes" | "-n" => {
                if i + 1 < args.len() {
                    config.num_nodes = args[i + 1].parse().unwrap_or(5);
                    i += 1;
                }
            }
            "--duration" | "-d" => {
                if i + 1 < args.len() {
                    config.test_duration_secs = args[i + 1].parse().unwrap_or(60);
                    i += 1;
                }
            }
            "--tps" | "-t" => {
                if i + 1 < args.len() {
                    config.transactions_per_second = args[i + 1].parse().unwrap_or(1000);
                    i += 1;
                }
            }
            "--target" => {
                if i + 1 < args.len() {
                    config.target_tps = args[i + 1].parse().unwrap_or(48000);
                    i += 1;
                }
            }
            "--size" => {
                if i + 1 < args.len() {
                    config.transaction_size_bytes = args[i + 1].parse().unwrap_or(512);
                    i += 1;
                }
            }
            "--help" | "-h" => {
                print_help();
                std::process::exit(0);
            }
            _ => {}
        }
        i += 1;
    }
    
    config
}

/// Print help information
fn print_help() {
    println!(r#"
Q-NarwhalKnight Performance Test Runner

USAGE:
    cargo run --bin performance_test [OPTIONS]

OPTIONS:
    -n, --nodes <NUM>       Number of nodes (default: 5)
    -d, --duration <SECS>   Test duration in seconds (default: 60)
    -t, --tps <NUM>         Transactions per second to generate (default: 1000)
    --target <NUM>          Target TPS for comparison (default: 48000)
    --size <BYTES>          Transaction size in bytes (default: 512)
    -h, --help              Show this help message

EXAMPLES:
    cargo run --bin performance_test                    # Run with defaults
    cargo run --bin performance_test -n 5 -d 30        # 5 nodes, 30 second test
    cargo run --bin performance_test -t 5000           # High throughput test
    cargo run --bin performance_test --target 100000   # Ultra-high target TPS
"#);
}

/// Run comprehensive performance testing suite
async fn run_comprehensive_tests(config: PerformanceTestConfig) -> Result<Vec<TestScenarioResult>> {
    let mut results = Vec::new();
    
    // Scenario 1: Baseline performance test
    info!("🧪 Running Scenario 1: Baseline Performance");
    let baseline_result = run_single_test_scenario(
        "Baseline Performance", 
        config.clone()
    ).await?;
    results.push(baseline_result);
    
    // Scenario 2: High throughput test (2x normal load)
    info!("🧪 Running Scenario 2: High Throughput");
    let mut high_throughput_config = config.clone();
    high_throughput_config.transactions_per_second *= 2;
    high_throughput_config.batch_size = (high_throughput_config.batch_size * 2).min(500);
    
    let high_throughput_result = run_single_test_scenario(
        "High Throughput", 
        high_throughput_config
    ).await?;
    results.push(high_throughput_result);
    
    // Scenario 3: Large transaction test
    info!("🧪 Running Scenario 3: Large Transactions");
    let mut large_tx_config = config.clone();
    large_tx_config.transaction_size_bytes *= 4; // 2KB transactions
    large_tx_config.transactions_per_second /= 2; // Reduce TPS for larger transactions
    
    let large_tx_result = run_single_test_scenario(
        "Large Transactions", 
        large_tx_config
    ).await?;
    results.push(large_tx_result);
    
    // Scenario 4: Stress test (maximum load)
    info!("🧪 Running Scenario 4: Stress Test");
    let mut stress_config = config.clone();
    stress_config.transactions_per_second *= 5;
    stress_config.batch_size = 200;
    stress_config.test_duration_secs = 30; // Shorter for stress test
    
    let stress_result = run_single_test_scenario(
        "Stress Test", 
        stress_config
    ).await?;
    results.push(stress_result);
    
    Ok(results)
}

/// Run a single test scenario with timeout protection
async fn run_single_test_scenario(
    scenario_name: &str, 
    config: PerformanceTestConfig
) -> Result<TestScenarioResult> {
    info!("▶️  Starting scenario: {}", scenario_name);
    
    // Set timeout to prevent hanging tests
    let timeout_duration = Duration::from_secs(config.test_duration_secs + 120); // Extra buffer
    
    let scenario_result = timeout(timeout_duration, async {
        let mut test = PerformanceTest::new(config.clone()).await?;
        let results = test.run_performance_test().await?;
        
        Ok::<_, anyhow::Error>(TestScenarioResult {
            scenario_name: scenario_name.to_string(),
            config: config.clone(),
            results,
            success: true,
            error_message: None,
        })
    }).await;
    
    match scenario_result {
        Ok(Ok(result)) => {
            info!("✅ Scenario '{}' completed successfully", scenario_name);
            Ok(result)
        }
        Ok(Err(e)) => {
            error!("❌ Scenario '{}' failed: {}", scenario_name, e);
            Ok(TestScenarioResult {
                scenario_name: scenario_name.to_string(),
                config,
                results: Default::default(),
                success: false,
                error_message: Some(e.to_string()),
            })
        }
        Err(_) => {
            error!("⏰ Scenario '{}' timed out", scenario_name);
            Ok(TestScenarioResult {
                scenario_name: scenario_name.to_string(),
                config,
                results: Default::default(),
                success: false,
                error_message: Some("Test timed out".to_string()),
            })
        }
    }
}

/// Generate final comprehensive report
fn generate_final_report(config: &PerformanceTestConfig, results: &[TestScenarioResult]) {
    println!("\n{}", "=".repeat(80));
    println!("🏆 Q-NARWHALKNIGHT COMPREHENSIVE PERFORMANCE REPORT");
    println!("{}", "=".repeat(80));
    
    // Summary table
    println!("\n📊 SCENARIO SUMMARY:");
    println!("{:<20} {:<12} {:<12} {:<12} {:<12} {:<10}", 
             "Scenario", "Success", "TPS", "MB/s", "Finality", "Efficiency");
    println!("{}", "-".repeat(80));
    
    let mut total_successful = 0;
    let mut best_tps = 0.0;
    let mut best_bps = 0.0;
    let mut best_finality = f64::MAX;
    
    for result in results {
        let success_str = if result.success { "✅ PASS" } else { "❌ FAIL" };
        
        if result.success {
            total_successful += 1;
            best_tps = best_tps.max(result.results.average_tps);
            best_bps = best_bps.max(result.results.average_bps);
            best_finality = best_finality.min(result.results.average_finality_ms);
            
            println!("{:<20} {:<12} {:<12.0} {:<12.2} {:<12.1} {:<10.3}", 
                     result.scenario_name,
                     success_str,
                     result.results.average_tps,
                     result.results.average_bps / 1_000_000.0,
                     result.results.average_finality_ms,
                     result.results.consensus_efficiency);
        } else {
            println!("{:<20} {:<12} {:<12} {:<12} {:<12} {:<10}", 
                     result.scenario_name, success_str, "N/A", "N/A", "N/A", "N/A");
            if let Some(ref error) = result.error_message {
                println!("    Error: {}", error);
            }
        }
    }
    
    // Overall performance analysis
    println!("\n🎯 PERFORMANCE ANALYSIS:");
    println!("• Tests Passed: {}/{}", total_successful, results.len());
    println!("• Peak TPS Achieved: {:.2}", best_tps);
    println!("• Peak Bandwidth: {:.2} MB/s", best_bps / 1_000_000.0);
    println!("• Best Finality: {:.2} ms", if best_finality == f64::MAX { 0.0 } else { best_finality });
    
    // Target comparison
    println!("\n📈 TARGET COMPARISON:");
    let tps_achievement = (best_tps / config.target_tps as f64) * 100.0;
    let finality_achievement = if best_finality != f64::MAX {
        (config.target_finality_ms as f64 / best_finality) * 100.0
    } else {
        0.0
    };
    
    println!("• TPS Target: {:,} | Best: {:.0} ({:.1}%)", 
             config.target_tps, best_tps, tps_achievement);
    println!("• Finality Target: {} ms | Best: {:.1} ms ({:.1}%)", 
             config.target_finality_ms, 
             if best_finality == f64::MAX { 0.0 } else { best_finality }, 
             finality_achievement);
    
    // Recommendations
    println!("\n💡 RECOMMENDATIONS:");
    if tps_achievement >= 100.0 {
        println!("✅ TPS target achieved - excellent throughput performance");
    } else if tps_achievement >= 50.0 {
        println!("⚠️ TPS target partially achieved - consider optimization");
    } else {
        println!("❌ TPS target not achieved - requires significant optimization");
    }
    
    if finality_achievement >= 100.0 {
        println!("✅ Finality target achieved - excellent consensus speed");
    } else if finality_achievement >= 50.0 {
        println!("⚠️ Finality target partially achieved - consider consensus optimization");
    } else {
        println!("❌ Finality target not achieved - consensus speed needs improvement");
    }
    
    println!("\n{}", "=".repeat(80));
}

/// Result of a single test scenario
#[derive(Debug, Clone)]
struct TestScenarioResult {
    scenario_name: String,
    config: PerformanceTestConfig,
    results: performance_test::PerformanceTestResults,
    success: bool,
    error_message: Option<String>,
}

/// Default implementation for PerformanceTestResults to handle error cases
impl Default for performance_test::PerformanceTestResults {
    fn default() -> Self {
        Self {
            total_duration: Duration::from_secs(0),
            total_transactions: 0,
            total_bytes: 0,
            average_tps: 0.0,
            average_bps: 0.0,
            peak_tps: 0.0,
            peak_bps: 0.0,
            average_finality_ms: 0.0,
            consensus_efficiency: 0.0,
            node_metrics: vec![],
            network_partition_tolerance: false,
            quantum_entropy_quality: 0.0,
        }
    }
}