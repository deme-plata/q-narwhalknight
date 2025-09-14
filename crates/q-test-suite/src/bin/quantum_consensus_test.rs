/// Comprehensive quantum consensus testing binary
/// 
/// This binary runs the complete Q-NarwhalKnight Phase 2 test suite,
/// validating all quantum-enhanced consensus components.

use anyhow::Result;
use clap::{Parser, Subcommand};
use tracing::{info, warn, error};
use tracing_subscriber;

use q_test_suite::{TestSuite, TestSuiteConfig};

#[derive(Parser)]
#[command(name = "quantum-consensus-test")]
#[command(about = "Q-NarwhalKnight Phase 2 Quantum Consensus Test Suite")]
#[command(version)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
    
    /// Enable verbose logging
    #[arg(short, long)]
    verbose: bool,
    
    /// Test timeout in seconds
    #[arg(short, long, default_value = "300")]
    timeout: u64,
}

#[derive(Subcommand)]
enum Commands {
    /// Run all tests
    All {
        /// Number of iterations for statistical tests
        #[arg(short, long, default_value = "1000")]
        iterations: u32,
        
        /// Number of performance samples
        #[arg(short, long, default_value = "100")]
        samples: u32,
    },
    
    /// Run only integration tests
    Integration {
        #[arg(short, long, default_value = "100")]
        samples: u32,
    },
    
    /// Run only security tests
    Security {
        #[arg(short, long, default_value = "1000")]
        iterations: u32,
    },
    
    /// Run only performance tests
    Performance {
        #[arg(short, long, default_value = "100")]
        samples: u32,
    },
    
    /// Run only benchmarks
    Benchmarks,
    
    /// Generate test report
    Report {
        /// Output file path
        #[arg(short, long, default_value = "test-report.md")]
        output: String,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    
    // Initialize logging
    let log_level = if cli.verbose { "debug" } else { "info" };
    tracing_subscriber::fmt()
        .with_env_filter(format!("q_test_suite={},quantum_consensus_test={}", log_level, log_level))
        .init();
    
    info!("🚀 Q-NarwhalKnight Phase 2 Quantum Consensus Test Suite");
    info!("⚛️  Testing world's first quantum-enhanced DAG-BFT consensus");
    
    let config = TestSuiteConfig {
        test_timeout: std::time::Duration::from_secs(cli.timeout),
        benchmark_duration: std::time::Duration::from_secs(60),
        security_iterations: 1000,
        performance_samples: 100,
    };
    
    let suite = TestSuite::new(config);
    
    match cli.command {
        Commands::All { iterations, samples } => {
            info!("Running comprehensive test suite...");
            let mut config = suite.config.clone();
            config.security_iterations = iterations;
            config.performance_samples = samples;
            
            let suite = TestSuite::new(config);
            let results = suite.run_all_tests().await?;
            
            print_results(&results);
            
            if !results.all_passed() {
                error!("❌ Some tests failed!");
                std::process::exit(1);
            } else {
                info!("✅ All tests passed!");
            }
        }
        
        Commands::Integration { samples } => {
            info!("Running integration tests...");
            let results = q_test_suite::integration::run_integration_tests().await?;
            
            println!("\n🔗 Integration Test Results:");
            println!("{}", results.summary());
            
            if !results.passed {
                error!("❌ Integration tests failed!");
                std::process::exit(1);
            }
        }
        
        Commands::Security { iterations } => {
            info!("Running security validation...");
            let results = q_test_suite::security::run_security_tests(iterations).await?;
            
            println!("\n🔒 Security Test Results:");
            println!("{}", results.summary());
            
            if results.vulnerabilities_found > 0 {
                warn!("⚠️  Security vulnerabilities detected!");
                for detail in &results.details {
                    if !detail.passed {
                        warn!("  - {}: {}", detail.test_name, detail.description);
                    }
                }
                std::process::exit(1);
            }
        }
        
        Commands::Performance { samples } => {
            info!("Running performance tests...");
            let results = q_test_suite::performance::run_performance_tests(samples).await?;
            
            println!("\n⚡ Performance Test Results:");
            println!("{}", results.summary());
            
            if !results.passed {
                error!("❌ Performance targets not met!");
                std::process::exit(1);
            }
        }
        
        Commands::Benchmarks => {
            info!("Running detailed benchmarks...");
            let results = q_test_suite::benchmarks::run_benchmarks().await?;
            
            println!("\n📊 Benchmark Results:");
            println!("{}", results.summary());
            
            // Also run additional benchmark suites
            println!("\n🔄 Running parallel performance benchmarks...");
            q_test_suite::benchmarks::benchmark_parallel_performance().await?;
            
            println!("\n💾 Running memory usage benchmarks...");
            q_test_suite::benchmarks::benchmark_memory_usage().await?;
            
            println!("\n💪 Running stress tests...");
            q_test_suite::benchmarks::stress_test_components().await?;
            
            if !results.passed {
                error!("❌ Benchmark targets not met!");
                std::process::exit(1);
            }
        }
        
        Commands::Report { output } => {
            info!("Generating comprehensive test report...");
            
            // Run all tests to generate comprehensive report
            let results = suite.run_all_tests().await?;
            let report = results.generate_report();
            
            // Write report to file
            std::fs::write(&output, &report)?;
            
            info!("📄 Test report written to: {}", output);
            println!("\n{}", report);
        }
    }
    
    Ok(())
}

/// Print formatted test results
fn print_results(results: &q_test_suite::TestResults) {
    println!("\n" + "=".repeat(80).as_str());
    println!("🎯 Q-NARWHAL KNIGHT PHASE 2 TEST RESULTS");
    println!("=".repeat(80));
    
    println!("\n🔗 Integration Tests:");
    println!("  {}", results.integration.summary());
    
    println!("\n🔒 Security Validation:");
    println!("  {}", results.security.summary());
    
    println!("\n⚡ Performance Tests:");
    println!("  {}", results.performance.summary());
    
    println!("\n📊 Benchmarks:");
    println!("  {}", results.benchmarks.summary());
    
    println!("\n" + "=".repeat(80).as_str());
    
    if results.all_passed() {
        println!("🎉 QUANTUM CONSENSUS SYSTEM READY FOR PRODUCTION!");
        println!("⚛️  World's first quantum-enhanced DAG-BFT validated!");
    } else {
        println!("❌ Some tests failed - system not ready for production");
    }
    
    println!("=".repeat(80));
}