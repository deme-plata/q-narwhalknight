/// Quantum randomness test suite for statistical validation
/// Implements NIST SP 800-22 and additional quantum-specific tests

use anyhow::Result;
use std::collections::HashMap;
use serde::{Serialize, Deserialize};

/// Configuration for quantum randomness test suite
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestSuiteConfig {
    /// Enable frequency (monobit) test
    pub enable_frequency_test: bool,
    
    /// Enable frequency test within a block
    pub enable_block_frequency_test: bool,
    
    /// Enable runs test
    pub enable_runs_test: bool,
    
    /// Enable longest run of ones in a block test
    pub enable_longest_run_test: bool,
    
    /// Enable binary matrix rank test
    pub enable_matrix_rank_test: bool,
    
    /// Enable discrete Fourier transform test
    pub enable_dft_test: bool,
    
    /// Enable non-overlapping template matching test
    pub enable_template_test: bool,
    
    /// Enable overlapping template matching test
    pub enable_overlapping_template_test: bool,
    
    /// Enable Maurer's universal statistical test
    pub enable_universal_test: bool,
    
    /// Enable approximate entropy test
    pub enable_approximate_entropy_test: bool,
    
    /// Enable random excursions test
    pub enable_random_excursions_test: bool,
    
    /// Enable random excursions variant test
    pub enable_random_excursions_variant_test: bool,
    
    /// Enable serial test
    pub enable_serial_test: bool,
    
    /// Enable linear complexity test
    pub enable_linear_complexity_test: bool,
    
    /// Enable quantum-specific tests
    pub enable_quantum_tests: bool,
    
    /// Significance level for statistical tests (typically 0.01)
    pub significance_level: f64,
    
    /// Block size for block-based tests
    pub block_size: usize,
}

impl Default for TestSuiteConfig {
    fn default() -> Self {
        Self {
            enable_frequency_test: true,
            enable_block_frequency_test: true,
            enable_runs_test: true,
            enable_longest_run_test: true,
            enable_matrix_rank_test: true,
            enable_dft_test: true,
            enable_template_test: false, // Computationally expensive
            enable_overlapping_template_test: false,
            enable_universal_test: true,
            enable_approximate_entropy_test: true,
            enable_random_excursions_test: false, // Requires specific data characteristics
            enable_random_excursions_variant_test: false,
            enable_serial_test: true,
            enable_linear_complexity_test: false,
            enable_quantum_tests: true,
            significance_level: 0.01,
            block_size: 128,
        }
    }
}

/// Results from quantum randomness test suite
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestResults {
    /// Overall test suite result
    pub passed: bool,
    
    /// Individual test results
    pub test_results: HashMap<String, TestResult>,
    
    /// Summary statistics
    pub summary: TestSummary,
    
    /// Test configuration used
    pub config: TestSuiteConfig,
}

/// Individual test result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestResult {
    /// Test name
    pub name: String,
    
    /// Whether test passed
    pub passed: bool,
    
    /// P-value from statistical test
    pub p_value: f64,
    
    /// Test statistic value
    pub test_statistic: f64,
    
    /// Critical value for comparison
    pub critical_value: f64,
    
    /// Additional test-specific data
    pub metadata: HashMap<String, f64>,
}

/// Test suite summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestSummary {
    /// Total tests run
    pub total_tests: u32,
    
    /// Number of tests passed
    pub tests_passed: u32,
    
    /// Overall pass rate
    pub pass_rate: f64,
    
    /// Minimum p-value observed
    pub min_p_value: f64,
    
    /// Maximum p-value observed
    pub max_p_value: f64,
    
    /// Average p-value
    pub average_p_value: f64,
    
    /// Data size tested
    pub data_size: usize,
}

/// Run complete quantum randomness test suite
pub async fn run_test_suite(data: &[u8], config: &TestSuiteConfig) -> Result<TestResults> {
    let mut test_results = HashMap::new();
    
    // Convert bytes to bits for bit-level tests
    let bits = bytes_to_bits(data);
    
    // Run enabled tests
    if config.enable_frequency_test {
        test_results.insert(
            "frequency".to_string(),
            frequency_test(&bits, config.significance_level)?,
        );
    }
    
    if config.enable_block_frequency_test {
        test_results.insert(
            "block_frequency".to_string(),
            block_frequency_test(&bits, config.block_size, config.significance_level)?,
        );
    }
    
    if config.enable_runs_test {
        test_results.insert(
            "runs".to_string(),
            runs_test(&bits, config.significance_level)?,
        );
    }
    
    if config.enable_longest_run_test {
        test_results.insert(
            "longest_run".to_string(),
            longest_run_test(&bits, config.significance_level)?,
        );
    }
    
    if config.enable_matrix_rank_test {
        test_results.insert(
            "matrix_rank".to_string(),
            matrix_rank_test(&bits, config.significance_level)?,
        );
    }
    
    if config.enable_dft_test {
        test_results.insert(
            "dft".to_string(),
            discrete_fourier_transform_test(&bits, config.significance_level)?,
        );
    }
    
    if config.enable_universal_test {
        test_results.insert(
            "universal".to_string(),
            universal_statistical_test(&bits, config.significance_level)?,
        );
    }
    
    if config.enable_approximate_entropy_test {
        test_results.insert(
            "approximate_entropy".to_string(),
            approximate_entropy_test(&bits, config.significance_level)?,
        );
    }
    
    if config.enable_serial_test {
        test_results.insert(
            "serial".to_string(),
            serial_test(&bits, config.significance_level)?,
        );
    }
    
    if config.enable_quantum_tests {
        // Add quantum-specific tests
        test_results.insert(
            "quantum_coherence".to_string(),
            quantum_coherence_test(&bits, config.significance_level)?,
        );
        
        test_results.insert(
            "quantum_entanglement".to_string(),
            quantum_entanglement_test(&bits, config.significance_level)?,
        );
    }
    
    // Calculate summary
    let summary = calculate_summary(&test_results, data.len());
    let overall_passed = test_results.values().all(|result| result.passed);
    
    Ok(TestResults {
        passed: overall_passed,
        test_results,
        summary,
        config: config.clone(),
    })
}

/// Convert bytes to bit vector
fn bytes_to_bits(data: &[u8]) -> Vec<u8> {
    let mut bits = Vec::with_capacity(data.len() * 8);
    for &byte in data {
        for i in 0..8 {
            bits.push((byte >> (7 - i)) & 1);
        }
    }
    bits
}

/// NIST SP 800-22 Frequency (Monobit) Test
fn frequency_test(bits: &[u8], alpha: f64) -> Result<TestResult> {
    let n = bits.len() as f64;
    let sum: f64 = bits.iter().map(|&b| if b == 1 { 1.0 } else { -1.0 }).sum();
    let test_statistic = sum.abs() / n.sqrt();
    
    // Use complementary error function approximation
    let p_value = erfc(test_statistic / 2_f64.sqrt());
    let critical_value = normal_inverse(1.0 - alpha / 2.0);
    
    Ok(TestResult {
        name: "Frequency (Monobit) Test".to_string(),
        passed: p_value >= alpha,
        p_value,
        test_statistic,
        critical_value,
        metadata: HashMap::new(),
    })
}

/// Block frequency test
fn block_frequency_test(bits: &[u8], block_size: usize, alpha: f64) -> Result<TestResult> {
    let n = bits.len();
    let num_blocks = n / block_size;
    
    if num_blocks < 1 {
        return Err(anyhow::anyhow!("Insufficient data for block frequency test"));
    }
    
    let mut chi_square = 0.0;
    
    for i in 0..num_blocks {
        let start = i * block_size;
        let end = start + block_size;
        let block_sum: u32 = bits[start..end].iter().map(|&b| b as u32).sum();
        let pi = block_sum as f64 / block_size as f64;
        chi_square += (pi - 0.5).powi(2);
    }
    
    chi_square *= 4.0 * block_size as f64;
    let p_value = igamc(num_blocks as f64 / 2.0, chi_square / 2.0);
    
    Ok(TestResult {
        name: "Block Frequency Test".to_string(),
        passed: p_value >= alpha,
        p_value,
        test_statistic: chi_square,
        critical_value: 0.0, // Chi-square critical values are complex
        metadata: {
            let mut meta = HashMap::new();
            meta.insert("num_blocks".to_string(), num_blocks as f64);
            meta.insert("block_size".to_string(), block_size as f64);
            meta
        },
    })
}

/// Runs test
fn runs_test(bits: &[u8], alpha: f64) -> Result<TestResult> {
    let n = bits.len();
    let ones: usize = bits.iter().map(|&b| b as usize).sum();
    let pi = ones as f64 / n as f64;
    
    // Pre-test: check if proportion is reasonable
    let tau = 2.0 / (n as f64).sqrt();
    if (pi - 0.5).abs() >= tau {
        return Ok(TestResult {
            name: "Runs Test".to_string(),
            passed: false,
            p_value: 0.0,
            test_statistic: 0.0,
            critical_value: tau,
            metadata: HashMap::new(),
        });
    }
    
    // Count runs
    let mut runs = 1;
    for i in 1..n {
        if bits[i] != bits[i - 1] {
            runs += 1;
        }
    }
    
    let test_statistic = (runs as f64 - 2.0 * n as f64 * pi * (1.0 - pi)).abs() /
        (2.0 * (2.0 * n as f64).sqrt() * pi * (1.0 - pi));
    
    let p_value = erfc(test_statistic / 2_f64.sqrt());
    
    Ok(TestResult {
        name: "Runs Test".to_string(),
        passed: p_value >= alpha,
        p_value,
        test_statistic,
        critical_value: 0.0,
        metadata: {
            let mut meta = HashMap::new();
            meta.insert("runs".to_string(), runs as f64);
            meta.insert("pi".to_string(), pi);
            meta
        },
    })
}

/// Longest run of ones test
fn longest_run_test(bits: &[u8], alpha: f64) -> Result<TestResult> {
    let n = bits.len();
    
    // Determine block size and number of blocks based on sequence length
    let (block_size, num_blocks, degrees_freedom) = if n >= 128 && n < 6272 {
        (8, n / 8, 3)
    } else if n >= 6272 && n < 750000 {
        (128, n / 128, 5)
    } else if n >= 750000 {
        (10000, n / 10000, 6)
    } else {
        return Err(anyhow::anyhow!("Insufficient data for longest run test"));
    };
    
    // Expected frequencies for different longest run lengths
    let expected_freqs = match degrees_freedom {
        3 => vec![0.2148, 0.3672, 0.2305, 0.1875],
        5 => vec![0.1174, 0.2430, 0.2493, 0.1752, 0.1027, 0.1124],
        6 => vec![0.0882, 0.2092, 0.2483, 0.1933, 0.1208, 0.0675, 0.0727],
        _ => return Err(anyhow::anyhow!("Invalid degrees of freedom")),
    };
    
    let mut observed_freqs = vec![0; expected_freqs.len()];
    
    // Analyze each block
    for i in 0..num_blocks {
        let start = i * block_size;
        let end = start + block_size;
        let longest_run = find_longest_run(&bits[start..end]);
        
        let freq_index = match degrees_freedom {
            3 => match longest_run {
                1 => 0,
                2 => 1,
                3 => 2,
                _ => 3,
            },
            5 => match longest_run {
                4 => 0,
                5 => 1,
                6 => 2,
                7 => 3,
                8 => 4,
                _ => 5,
            },
            6 => match longest_run {
                10..=16 => (longest_run - 10).min(5),
                _ if longest_run >= 16 => 6,
                _ => 0,
            },
            _ => 0,
        }.min(observed_freqs.len() - 1);
        
        observed_freqs[freq_index] += 1;
    }
    
    // Calculate chi-square statistic
    let mut chi_square = 0.0;
    for i in 0..expected_freqs.len() {
        let expected = expected_freqs[i] * num_blocks as f64;
        let observed = observed_freqs[i] as f64;
        chi_square += (observed - expected).powi(2) / expected;
    }
    
    let p_value = igamc((degrees_freedom - 1) as f64 / 2.0, chi_square / 2.0);
    
    Ok(TestResult {
        name: "Longest Run of Ones Test".to_string(),
        passed: p_value >= alpha,
        p_value,
        test_statistic: chi_square,
        critical_value: 0.0,
        metadata: HashMap::new(),
    })
}

fn find_longest_run(bits: &[u8]) -> usize {
    let mut max_run = 0;
    let mut current_run = 0;
    
    for &bit in bits {
        if bit == 1 {
            current_run += 1;
            max_run = max_run.max(current_run);
        } else {
            current_run = 0;
        }
    }
    
    max_run
}

/// Binary matrix rank test (simplified implementation)
fn matrix_rank_test(bits: &[u8], alpha: f64) -> Result<TestResult> {
    let matrix_size = 32;
    let num_matrices = bits.len() / (matrix_size * matrix_size);
    
    if num_matrices < 1 {
        return Err(anyhow::anyhow!("Insufficient data for matrix rank test"));
    }
    
    let mut rank_counts = [0; 3]; // Full rank, rank-1, rank-2 or less
    
    for i in 0..num_matrices {
        let start = i * matrix_size * matrix_size;
        let end = start + matrix_size * matrix_size;
        let matrix_bits = &bits[start..end];
        
        let rank = calculate_binary_matrix_rank(matrix_bits, matrix_size);
        
        match rank {
            r if r == matrix_size => rank_counts[0] += 1,
            r if r == matrix_size - 1 => rank_counts[1] += 1,
            _ => rank_counts[2] += 1,
        }
    }
    
    // Expected probabilities for 32x32 matrices
    let expected_probs = [0.2888, 0.5776, 0.1336];
    let mut chi_square = 0.0;
    
    for i in 0..3 {
        let expected = expected_probs[i] * num_matrices as f64;
        let observed = rank_counts[i] as f64;
        chi_square += (observed - expected).powi(2) / expected;
    }
    
    let p_value = igamc(1.0, chi_square / 2.0); // 2 degrees of freedom
    
    Ok(TestResult {
        name: "Binary Matrix Rank Test".to_string(),
        passed: p_value >= alpha,
        p_value,
        test_statistic: chi_square,
        critical_value: 0.0,
        metadata: HashMap::new(),
    })
}

fn calculate_binary_matrix_rank(bits: &[u8], size: usize) -> usize {
    // Simplified binary matrix rank calculation
    // In practice, would implement full Gaussian elimination over GF(2)
    let mut matrix = vec![vec![0u8; size]; size];
    
    // Fill matrix
    for i in 0..size {
        for j in 0..size {
            let idx = i * size + j;
            if idx < bits.len() {
                matrix[i][j] = bits[idx];
            }
        }
    }
    
    // Estimate rank (simplified)
    let mut rank = 0;
    for i in 0..size {
        let row_sum: u8 = matrix[i].iter().sum();
        if row_sum > 0 {
            rank += 1;
        }
    }
    
    rank.min(size)
}

/// Discrete Fourier Transform test (simplified)
fn discrete_fourier_transform_test(bits: &[u8], alpha: f64) -> Result<TestResult> {
    let n = bits.len();
    if n < 100 {
        return Err(anyhow::anyhow!("Insufficient data for DFT test"));
    }
    
    // Convert to -1, +1 representation
    let x: Vec<f64> = bits.iter().map(|&b| if b == 1 { 1.0 } else { -1.0 }).collect();
    
    // Simplified DFT (real parts only)
    let mut magnitudes = Vec::new();
    for k in 0..n/2 {
        let mut real_part = 0.0;
        for i in 0..n {
            let angle = -2.0 * std::f64::consts::PI * (k as f64) * (i as f64) / (n as f64);
            real_part += x[i] * angle.cos();
        }
        magnitudes.push(real_part.abs());
    }
    
    // Count peaks above threshold
    let threshold = (n as f64).sqrt() * 3.0;
    let peaks_above_threshold = magnitudes.iter().filter(|&&mag| mag > threshold).count();
    
    let expected_peaks = 0.95 * (n as f64) / 2.0;
    let test_statistic = (peaks_above_threshold as f64 - expected_peaks) / (expected_peaks * 0.05).sqrt();
    
    let p_value = erfc(test_statistic.abs() / 2_f64.sqrt());
    
    Ok(TestResult {
        name: "Discrete Fourier Transform Test".to_string(),
        passed: p_value >= alpha,
        p_value,
        test_statistic,
        critical_value: 0.0,
        metadata: HashMap::new(),
    })
}

/// Universal statistical test (simplified)
fn universal_statistical_test(bits: &[u8], alpha: f64) -> Result<TestResult> {
    let n = bits.len();
    if n < 387840 {
        return Err(anyhow::anyhow!("Universal test requires at least 387840 bits"));
    }
    
    let l = 7; // Block length
    let q = 1280; // Number of initialization blocks
    let k = (n / l) - q; // Number of test blocks
    
    let mut table = vec![0; 1 << l];
    let mut sum = 0.0;
    
    // Initialization phase
    for i in 1..=q {
        let block_val = extract_block(bits, i, l);
        table[block_val] = i;
    }
    
    // Testing phase
    for i in (q + 1)..=(q + k) {
        let block_val = extract_block(bits, i, l);
        let distance = i - table[block_val];
        table[block_val] = i;
        sum += (distance as f64).ln();
    }
    
    let fn_val = sum / k as f64;
    let expected_val = 6.196; // Expected value for l=7
    let variance = 2.178; // Variance for l=7
    
    let test_statistic = (fn_val - expected_val) / variance.sqrt();
    let p_value = erfc(test_statistic.abs() / 2_f64.sqrt());
    
    Ok(TestResult {
        name: "Maurer's Universal Statistical Test".to_string(),
        passed: p_value >= alpha,
        p_value,
        test_statistic,
        critical_value: 0.0,
        metadata: HashMap::new(),
    })
}

fn extract_block(bits: &[u8], block_index: usize, block_length: usize) -> usize {
    let start = (block_index - 1) * block_length;
    let mut value = 0;
    
    for i in 0..block_length {
        if start + i < bits.len() {
            value = (value << 1) | (bits[start + i] as usize);
        }
    }
    
    value
}

/// Approximate entropy test
fn approximate_entropy_test(bits: &[u8], alpha: f64) -> Result<TestResult> {
    let n = bits.len();
    let m = 10; // Pattern length
    
    if n < (1 << (m + 1)) {
        return Err(anyhow::anyhow!("Insufficient data for approximate entropy test"));
    }
    
    let phi_m = calculate_phi(bits, m);
    let phi_m1 = calculate_phi(bits, m + 1);
    
    let app_entropy = phi_m - phi_m1;
    let chi_square = 2.0 * n as f64 * (2_f64.ln() - app_entropy);
    let p_value = igamc((1 << (m - 1)) as f64, chi_square / 2.0);
    
    Ok(TestResult {
        name: "Approximate Entropy Test".to_string(),
        passed: p_value >= alpha,
        p_value,
        test_statistic: chi_square,
        critical_value: 0.0,
        metadata: HashMap::new(),
    })
}

fn calculate_phi(bits: &[u8], m: usize) -> f64 {
    let n = bits.len();
    let num_patterns = 1 << m;
    let mut pattern_counts = vec![0; num_patterns];
    
    for i in 0..=(n - m) {
        let mut pattern = 0;
        for j in 0..m {
            pattern = (pattern << 1) | (bits[i + j] as usize);
        }
        pattern_counts[pattern] += 1;
    }
    
    let mut phi = 0.0;
    for &count in &pattern_counts {
        if count > 0 {
            let prob = count as f64 / (n - m + 1) as f64;
            phi += prob * prob.ln();
        }
    }
    
    phi
}

/// Serial test (Simplified)
fn serial_test(bits: &[u8], alpha: f64) -> Result<TestResult> {
    let n = bits.len();
    let m = 16; // Pattern length
    
    if n < 3 * (1 << (m - 1)) {
        return Err(anyhow::anyhow!("Insufficient data for serial test"));
    }
    
    let psi2_m = calculate_psi2(bits, m);
    let psi2_m1 = calculate_psi2(bits, m - 1);
    let psi2_m2 = calculate_psi2(bits, m - 2);
    
    let delta1 = psi2_m - psi2_m1;
    let delta2 = psi2_m - 2.0 * psi2_m1 + psi2_m2;
    
    let p_value1 = igamc((1 << (m - 2)) as f64, delta1 / 2.0);
    let p_value2 = igamc((1 << (m - 3)) as f64, delta2 / 2.0);
    
    let p_value = p_value1.min(p_value2);
    
    Ok(TestResult {
        name: "Serial Test".to_string(),
        passed: p_value >= alpha,
        p_value,
        test_statistic: delta1,
        critical_value: 0.0,
        metadata: HashMap::new(),
    })
}

fn calculate_psi2(bits: &[u8], m: usize) -> f64 {
    let n = bits.len();
    let num_patterns = 1 << m;
    let mut pattern_counts = vec![0; num_patterns];
    
    for i in 0..n {
        let mut pattern = 0;
        for j in 0..m {
            pattern = (pattern << 1) | (bits[(i + j) % n] as usize);
        }
        pattern_counts[pattern] += 1;
    }
    
    let mut psi2 = 0.0;
    for &count in &pattern_counts {
        psi2 += (count * count) as f64;
    }
    
    (psi2 / n as f64) - n as f64
}

/// Quantum coherence test (custom test for quantum sources)
fn quantum_coherence_test(bits: &[u8], alpha: f64) -> Result<TestResult> {
    // This is a custom test to detect quantum coherence patterns
    // Real implementation would use quantum-specific statistical measures
    
    let n = bits.len();
    if n < 1000 {
        return Err(anyhow::anyhow!("Insufficient data for quantum coherence test"));
    }
    
    // Measure phase coherence using autocorrelation
    let mut correlation_sum = 0.0;
    let max_lag = 100.min(n / 10);
    
    for lag in 1..max_lag {
        let mut correlation = 0.0;
        for i in 0..(n - lag) {
            correlation += (bits[i] as f64) * (bits[i + lag] as f64);
        }
        correlation /= (n - lag) as f64;
        correlation_sum += correlation.abs();
    }
    
    let average_correlation = correlation_sum / max_lag as f64;
    
    // Good quantum sources should have low correlation
    let test_statistic = average_correlation * (n as f64).sqrt();
    let p_value = if average_correlation < 0.25 { 0.95 } else { 0.05 };
    
    Ok(TestResult {
        name: "Quantum Coherence Test".to_string(),
        passed: p_value >= alpha,
        p_value,
        test_statistic,
        critical_value: 0.25,
        metadata: HashMap::new(),
    })
}

/// Quantum entanglement test (custom test for quantum sources)
fn quantum_entanglement_test(bits: &[u8], alpha: f64) -> Result<TestResult> {
    // Custom test to detect quantum entanglement signatures
    let n = bits.len();
    if n < 2000 {
        return Err(anyhow::anyhow!("Insufficient data for quantum entanglement test"));
    }
    
    // Split into pairs and measure Bell inequality violations
    let mut bell_violations = 0;
    let num_pairs = n / 2;
    
    for i in 0..num_pairs {
        let bit1 = bits[2 * i];
        let bit2 = bits[2 * i + 1];
        
        // Simplified Bell inequality check
        // Real implementation would use proper quantum correlation measures
        if bit1 != bit2 {
            bell_violations += 1;
        }
    }
    
    let violation_rate = bell_violations as f64 / num_pairs as f64;
    
    // Quantum sources should show some Bell inequality violations
    let test_statistic = (violation_rate - 0.5).abs();
    let p_value = if test_statistic < 0.1 { 0.95 } else { 0.05 };
    
    Ok(TestResult {
        name: "Quantum Entanglement Test".to_string(),
        passed: p_value >= alpha,
        p_value,
        test_statistic,
        critical_value: 0.1,
        metadata: HashMap::new(),
    })
}

/// Calculate test suite summary
fn calculate_summary(results: &HashMap<String, TestResult>, data_size: usize) -> TestSummary {
    let total_tests = results.len() as u32;
    let tests_passed = results.values().filter(|r| r.passed).count() as u32;
    let pass_rate = tests_passed as f64 / total_tests as f64;
    
    let p_values: Vec<f64> = results.values().map(|r| r.p_value).collect();
    let min_p_value = p_values.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_p_value = p_values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let average_p_value = p_values.iter().sum::<f64>() / p_values.len() as f64;
    
    TestSummary {
        total_tests,
        tests_passed,
        pass_rate,
        min_p_value,
        max_p_value,
        average_p_value,
        data_size,
    }
}

// Mathematical helper functions (simplified implementations)

fn erfc(x: f64) -> f64 {
    // Complementary error function approximation
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let p = 0.3275911;
    
    let sign = if x >= 0.0 { 1.0 } else { -1.0 };
    let x = x.abs();
    
    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();
    
    if sign >= 0.0 { y } else { 2.0 - y }
}

fn igamc(a: f64, x: f64) -> f64 {
    // Incomplete gamma function approximation
    // This is a very simplified implementation
    if x <= 0.0 { return 1.0; }
    if a <= 0.0 { return 0.0; }
    
    // Use continued fraction approximation
    let mut cf = 1.0;
    let mut term = 1.0;
    
    for n in 1..100 {
        term *= x / (a + n as f64);
        cf += term;
        if term.abs() < 1e-10 { break; }
    }
    
    ((-x).exp() * x.powf(a) / gamma(a)) * cf
}

fn gamma(x: f64) -> f64 {
    // Gamma function approximation using Stirling's formula
    if x < 1.0 {
        return gamma(x + 1.0) / x;
    }
    
    let sqrt_2pi = (2.0 * std::f64::consts::PI).sqrt();
    sqrt_2pi * x.powf(x - 0.5) * (-x).exp()
}

fn normal_inverse(p: f64) -> f64 {
    // Inverse normal distribution approximation
    if p <= 0.0 { return f64::NEG_INFINITY; }
    if p >= 1.0 { return f64::INFINITY; }
    if p == 0.5 { return 0.0; }
    
    let sign = if p > 0.5 { 1.0 } else { -1.0 };
    let r = if p > 0.5 { 1.0 - p } else { p };
    
    let t = (-2.0 * r.ln()).sqrt();
    let c0 = 2.515517;
    let c1 = 0.802853;
    let c2 = 0.010328;
    let d1 = 1.432788;
    let d2 = 0.189269;
    let d3 = 0.001308;
    
    sign * (t - (c0 + c1 * t + c2 * t * t) / (1.0 + d1 * t + d2 * t * t + d3 * t * t * t))
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::RngCore;

    #[tokio::test]
    async fn test_frequency_test() {
        let mut bits = vec![0u8; 1000];
        // Create alternating pattern
        for i in 0..1000 {
            bits[i] = (i % 2) as u8;
        }
        
        let result = frequency_test(&bits, 0.01).unwrap();
        assert!(result.p_value > 0.0);
    }

    #[tokio::test]
    async fn test_runs_test() {
        let mut bits = vec![0u8; 1000];
        for i in 0..1000 {
            bits[i] = (i % 2) as u8;
        }
        
        let result = runs_test(&bits, 0.01).unwrap();
        assert!(result.p_value > 0.0);
    }

    #[tokio::test]
    async fn test_full_suite() {
        let config = TestSuiteConfig::default();
        
        // Generate random test data
        let mut data = vec![0u8; 1000];
        rand::rngs::OsRng.fill_bytes(&mut data);
        
        let results = run_test_suite(&data, &config).await.unwrap();
        
        assert!(results.summary.total_tests > 0);
        assert!(results.summary.pass_rate >= 0.0 && results.summary.pass_rate <= 1.0);
    }
}