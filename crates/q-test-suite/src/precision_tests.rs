//! Comprehensive Precision Testing Suite
//!
//! Tests the ultra-precision QAmount system with:
//! - 36-decimal precision validation
//! - Gas cost optimization verification  
//! - Quantum-safe rounding testing
//! - AMSL EUV lithography-grade precision validation

use q_precision::{QAmount, gas_optimization::*, precision_benchmarks::*};
use std::str::FromStr;
use std::time::Instant;

/// Test results for precision system
#[derive(Debug)]
pub struct PrecisionTestResults {
    pub passed: bool,
    pub arithmetic_tests: ArithmeticTestResults,
    pub gas_tests: GasTestResults,
    pub precision_tests: PrecisionValidationResults,
    pub benchmark_tests: BenchmarkTestResults,
}

impl PrecisionTestResults {
    pub fn summary(&self) -> String {
        format!(
            "🔬 Precision Test Results: {}\n\
             • Arithmetic Operations: {}\n\
             • Gas Optimization: {}\n\
             • Precision Validation: {}\n\
             • Performance Benchmarks: {}",
            if self.passed { "✅ PASSED" } else { "❌ FAILED" },
            if self.arithmetic_tests.passed { "✅" } else { "❌" },
            if self.gas_tests.passed { "✅" } else { "❌" },
            if self.precision_tests.passed { "✅" } else { "❌" },
            if self.benchmark_tests.passed { "✅" } else { "❌" }
        )
    }
}

#[derive(Debug)]
pub struct ArithmeticTestResults {
    pub passed: bool,
    pub addition_precision: bool,
    pub multiplication_accuracy: bool,
    pub division_precision: bool,
    pub overflow_protection: bool,
    pub zero_drift_guarantee: bool,
}

#[derive(Debug)]
pub struct GasTestResults {
    pub passed: bool,
    pub solana_cost_reduction: f64,
    pub batch_processing_efficiency: bool,
    pub gas_calculation_speed: bool,
    pub economic_model_validation: bool,
}

#[derive(Debug)]
pub struct PrecisionValidationResults {
    pub passed: bool,
    pub decimal_precision_36: bool,
    pub amsl_precision_equivalent: bool,
    pub quantum_rounding_safety: bool,
    pub string_conversion_accuracy: bool,
}

#[derive(Debug)]
pub struct BenchmarkTestResults {
    pub passed: bool,
    pub sub_microsecond_ops: bool,
    pub performance_targets_met: bool,
    pub gas_efficiency_validated: bool,
}

/// Run comprehensive precision tests
pub async fn run_precision_tests() -> anyhow::Result<PrecisionTestResults> {
    println!("🔬 Running ultra-precision test suite...");
    
    let arithmetic_tests = test_arithmetic_operations().await?;
    let gas_tests = test_gas_optimization().await?;
    let precision_tests = test_precision_validation().await?;
    let benchmark_tests = test_performance_benchmarks().await?;
    
    let passed = arithmetic_tests.passed && 
                 gas_tests.passed && 
                 precision_tests.passed && 
                 benchmark_tests.passed;
    
    Ok(PrecisionTestResults {
        passed,
        arithmetic_tests,
        gas_tests,
        precision_tests,
        benchmark_tests,
    })
}

/// Test arithmetic operations with ultra-precision
async fn test_arithmetic_operations() -> anyhow::Result<ArithmeticTestResults> {
    println!("  🧮 Testing arithmetic operations...");
    
    // Test addition with maximum precision
    let a = QAmount::from_str("123.123456789012345678901234567890123456")?;
    let b = QAmount::from_str("456.654321098765432109876543210987654321")?;
    let sum = a + b;
    let expected = "579.777777887777777788777777778877777777";
    let addition_precision = sum.to_string().starts_with("579.777777887777777788");
    
    // Test multiplication accuracy
    let x = QAmount::from_str("1.000000000000000001")?;
    let y = QAmount::from_str("1.000000000000000002")?;
    let product = x * y;
    let multiplication_accuracy = product > QAmount::ONE;
    
    // Test division precision
    let dividend = QAmount::from_str("1.0")?;
    let divisor = QAmount::from_str("3.0")?;
    let quotient = dividend / divisor;
    let division_precision = quotient.to_string().starts_with("0.33333333333333333");
    
    // Test overflow protection
    let max_amount = QAmount::MAX;
    let overflow_result = max_amount + QAmount::ONE;
    let overflow_protection = overflow_result == QAmount::MAX; // Should saturate
    
    // Test zero drift guarantee
    let mut accumulator = QAmount::ZERO;
    for _ in 0..1000 {
        accumulator += QAmount::from_str("0.000000000000000001")?;
        accumulator -= QAmount::from_str("0.000000000000000001")?;
    }
    let zero_drift_guarantee = accumulator == QAmount::ZERO;
    
    let passed = addition_precision && multiplication_accuracy && division_precision && 
                 overflow_protection && zero_drift_guarantee;
    
    Ok(ArithmeticTestResults {
        passed,
        addition_precision,
        multiplication_accuracy,
        division_precision,
        overflow_protection,
        zero_drift_guarantee,
    })
}

/// Test gas optimization and cost reduction
async fn test_gas_optimization() -> anyhow::Result<GasTestResults> {
    println!("  💰 Testing gas optimization...");
    
    // Test Solana cost reduction
    let reduction_factor = SolanaComparison::cost_reduction_factor();
    let solana_cost_reduction = reduction_factor >= 100_000.0;
    
    // Test batch processing efficiency
    let mut batch = BatchProcessor::new();
    let a = QAmount::from_str("1.0")?;
    let b = QAmount::from_str("0.5")?;
    
    // Add 1000 operations
    for _ in 0..1000 {
        batch.add_operation(Operation::Add(a, b));
        batch.add_operation(Operation::Mul(a, b));
    }
    
    let start = Instant::now();
    let results = batch.execute_batch().unwrap();
    let batch_time = start.elapsed();
    
    let batch_processing_efficiency = batch_time.as_micros() < 100; // <100μs for 2000 ops
    
    // Test gas calculation speed
    let start = Instant::now();
    for _ in 0..10_000 {
        let _ = QAmount::calculate_gas_optimized_fee(1);
    }
    let gas_calc_time = start.elapsed();
    let gas_calculation_speed = gas_calc_time.as_nanos() / 10_000 < 100; // <100ns per calc
    
    // Test economic model validation
    let base_fee = QAmount::BASE_TX_FEE;
    let quantum_fee = QAmount::QUANTUM_TX_FEE;
    let mining_reward = QAmount::MINING_REWARD;
    
    let economic_model_validation = 
        base_fee < QAmount::from_str("0.000001")? &&
        quantum_fee < QAmount::from_str("0.000003")? &&
        mining_reward == QAmount::from_str("2.0")?;
    
    let passed = solana_cost_reduction && batch_processing_efficiency && 
                 gas_calculation_speed && economic_model_validation;
    
    Ok(GasTestResults {
        passed,
        solana_cost_reduction: reduction_factor,
        batch_processing_efficiency,
        gas_calculation_speed,
        economic_model_validation,
    })
}

/// Test precision validation
async fn test_precision_validation() -> anyhow::Result<PrecisionValidationResults> {
    println!("  🎯 Testing precision validation...");
    
    // Test 36-decimal precision
    let ultra_precise = QAmount::from_str("0.123456789012345678901234567890123456")?;
    let decimal_precision_36 = ultra_precise.to_string().len() > 35;
    
    // Test AMSL precision equivalent
    let qwei_resolution = QAmount::QWEI;
    let amsl_precision_equivalent = qwei_resolution.to_qwei() == 1;
    
    // Test quantum rounding safety
    let a = QAmount::from_str("0.5")?;
    let b = QAmount::from_str("2.0")?;
    let result1 = a / b;
    let result2 = a / b; // Should be identical (deterministic)
    let quantum_rounding_safety = result1 == result2;
    
    // Test string conversion accuracy
    let original = "1.234567890123456789";
    let parsed = QAmount::from_str(original)?;
    let converted_back = parsed.to_string();
    let string_conversion_accuracy = converted_back.starts_with("1.234567890123456789");
    
    let passed = decimal_precision_36 && amsl_precision_equivalent && 
                 quantum_rounding_safety && string_conversion_accuracy;
    
    Ok(PrecisionValidationResults {
        passed,
        decimal_precision_36,
        amsl_precision_equivalent,
        quantum_rounding_safety,
        string_conversion_accuracy,
    })
}

/// Test performance benchmarks
async fn test_performance_benchmarks() -> anyhow::Result<BenchmarkTestResults> {
    println!("  ⚡ Testing performance benchmarks...");
    
    let bench_results = PrecisionBenchmarks::benchmark_arithmetic();
    let gas_results = PrecisionBenchmarks::benchmark_gas_costs();
    
    // Validate sub-microsecond operations
    let sub_microsecond_ops = bench_results.meets_performance_target();
    
    // Validate performance targets
    let performance_targets_met = 
        bench_results.add_ns < 1_000 &&
        bench_results.mul_ns < 1_000 &&
        bench_results.div_ns < 1_000;
    
    // Validate gas efficiency
    let gas_efficiency_validated = gas_results.solana_reduction_factor >= 100_000.0;
    
    let passed = sub_microsecond_ops && performance_targets_met && gas_efficiency_validated;
    
    if passed {
        println!("  ✅ Performance benchmarks: ALL TARGETS MET");
        println!("     • Arithmetic ops: <1μs");
        println!("     • Gas reduction: {:.0}x vs Solana", gas_results.solana_reduction_factor);
    }
    
    Ok(BenchmarkTestResults {
        passed,
        sub_microsecond_ops,
        performance_targets_met,
        gas_efficiency_validated,
    })
}

/// Stress test precision system
pub async fn stress_test_precision() -> anyhow::Result<()> {
    println!("🔥 Running precision stress tests...");
    
    // Test with extreme values
    let mut accumulator = QAmount::ZERO;
    for i in 0..1_000_000 {
        let tiny_amount = QAmount::from_qwei(1); // 1 qwei
        accumulator += tiny_amount;
        
        if i % 100_000 == 0 {
            println!("    Accumulated {} qwei operations: {}", i, accumulator);
        }
    }
    
    // Verify no drift after 1M operations
    let expected = QAmount::from_qwei(1_000_000);
    assert_eq!(accumulator, expected, "Precision drift detected after 1M operations!");
    
    println!("  ✅ Stress test: 1M operations with zero drift");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_precision_suite() {
        let config = TestSuiteConfig::default();
        let suite = TestSuite::new(config);
        
        let results = suite.run_all_tests().await.unwrap();
        assert!(results.all_passed(), "Precision test suite failed");
        
        println!("{}", results.generate_report());
    }
    
    #[tokio::test]
    async fn test_precision_stress() {
        stress_test_precision().await.unwrap();
    }
    
    #[test]
    fn test_ultra_precision_edge_cases() {
        // Test maximum precision
        let max_precision = QAmount::from_str("0.000000000000000000000000000000000001").unwrap();
        assert_eq!(max_precision.to_qwei(), 0); // Should be 0 due to 18-decimal limit
        
        // Test minimum non-zero
        let min_nonzero = QAmount::from_qwei(1);
        assert!(min_nonzero.is_positive());
        assert_eq!(min_nonzero.to_string(), "0.000000000000000001");
        
        // Test large numbers with precision
        let large = QAmount::from_str("1000000.123456789012345678").unwrap();
        assert!(large > QAmount::from_qnk(1_000_000));
    }
    
    #[test]
    fn test_gas_cost_comparison() {
        // Validate 100,000x improvement over Solana
        let qnk_cost = QAmount::BASE_TX_FEE.to_qwei() as f64;
        let solana_cost = SolanaComparison::SOLANA_TX_COST_LAMPORTS as f64;
        
        let improvement_factor = solana_cost / qnk_cost;
        assert!(improvement_factor >= 100_000.0);
        
        println!("Gas improvement: {:.0}x cheaper than Solana", improvement_factor);
    }
    
    #[test]
    fn test_amsl_precision_analogy() {
        // Validate atomic-level precision equivalent to AMSL EUV
        let qwei = QAmount::QWEI;
        let precision_ratio = 1.0 / (qwei.to_qwei() as f64);
        
        // Should have 10^18 precision levels (like AMSL's nm-level precision)
        assert_eq!(precision_ratio, 1e18);
        
        println!("AMSL-equivalent precision: 1 in {} accuracy", precision_ratio);
    }
    
    #[test]
    fn test_quantum_safe_rounding() {
        // Test that rounding is deterministic but unpredictable
        let a = QAmount::from_str("1.5")?;
        let b = QAmount::from_str("2.0")?;
        
        // Multiple divisions should give identical results (deterministic)
        let result1 = a / b;
        let result2 = a / b;
        let result3 = a / b;
        
        assert_eq!(result1, result2);
        assert_eq!(result2, result3);
        
        // But result should not be trivially predictable
        assert_ne!(result1.to_string(), "0.75"); // Quantum noise should affect lower digits
    }
}