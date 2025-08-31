/// Comprehensive Phase 2 Testing Suite for Q-NarwhalKnight
/// 
/// This module provides integration tests and benchmarks for all quantum-enhanced
/// consensus components, validating the complete Phase 2 implementation.

pub mod integration;
pub mod benchmarks;
pub mod security;
pub mod performance;

use anyhow::Result;
use std::time::Duration;
use tokio::time::timeout;

/// Test suite configuration
pub struct TestSuiteConfig {
    pub test_timeout: Duration,
    pub benchmark_duration: Duration,
    pub security_iterations: u32,
    pub performance_samples: u32,
}

impl Default for TestSuiteConfig {
    fn default() -> Self {
        Self {
            test_timeout: Duration::from_secs(30),
            benchmark_duration: Duration::from_secs(10),
            security_iterations: 1000,
            performance_samples: 100,
        }
    }
}

/// Main test suite runner
pub struct TestSuite {
    config: TestSuiteConfig,
}

impl TestSuite {
    pub fn new(config: TestSuiteConfig) -> Self {
        Self { config }
    }

    /// Run all Phase 2 tests
    pub async fn run_all_tests(&self) -> Result<TestResults> {
        println!("🧪 Starting Q-NarwhalKnight Phase 2 Comprehensive Test Suite");
        
        let mut results = TestResults::default();
        
        // Integration tests
        println!("🔗 Running integration tests...");
        results.integration = timeout(
            self.config.test_timeout,
            integration::run_integration_tests()
        ).await??;
        
        // Security tests
        println!("🔒 Running security validation...");
        results.security = timeout(
            self.config.test_timeout,
            security::run_security_tests(self.config.security_iterations)
        ).await??;
        
        // Performance benchmarks
        println!("⚡ Running performance benchmarks...");
        results.performance = timeout(
            self.config.test_timeout,
            performance::run_performance_tests(self.config.performance_samples)
        ).await??;
        
        // Benchmarks
        println!("📊 Running detailed benchmarks...");
        results.benchmarks = timeout(
            self.config.benchmark_duration,
            benchmarks::run_benchmarks()
        ).await??;
        
        println!("✅ All Phase 2 tests completed successfully!");
        Ok(results)
    }
}

/// Aggregated test results
#[derive(Debug, Default)]
pub struct TestResults {
    pub integration: integration::IntegrationResults,
    pub security: security::SecurityResults,
    pub performance: performance::PerformanceResults,
    pub benchmarks: benchmarks::BenchmarkResults,
}

impl TestResults {
    /// Generate comprehensive test report
    pub fn generate_report(&self) -> String {
        format!(
            "# Q-NarwhalKnight Phase 2 Test Report\n\n\
            ## Integration Tests\n{}\n\n\
            ## Security Validation\n{}\n\n\
            ## Performance Tests\n{}\n\n\
            ## Benchmarks\n{}\n",
            self.integration.summary(),
            self.security.summary(),
            self.performance.summary(),
            self.benchmarks.summary()
        )
    }
    
    /// Check if all tests passed
    pub fn all_passed(&self) -> bool {
        self.integration.passed &&
        self.security.passed &&
        self.performance.passed &&
        self.benchmarks.passed
    }
}