/// Entropy quality analysis for quantum random number generation
/// Implements statistical tests to verify quantum randomness quality

use anyhow::Result;
use std::collections::HashMap;
use tracing::{debug, warn};

/// Entropy quality analyzer for QRNG output
#[derive(Clone)]
pub struct EntropyAnalyzer {
    config: super::quantum_tests::TestSuiteConfig,
}

impl EntropyAnalyzer {
    pub fn new(config: super::quantum_tests::TestSuiteConfig) -> Result<Self> {
        Ok(Self { config })
    }

    /// Analyze entropy quality of byte sequence
    pub async fn analyze_entropy(&self, data: &[u8]) -> Result<EntropyQuality> {
        if data.len() < 16 {
            return Err(anyhow::anyhow!("Insufficient data for entropy analysis"));
        }

        let mut quality = EntropyQuality {
            overall_score: 0.0,
            shannon_entropy: 0.0,
            chi_square_p_value: 0.0,
            autocorrelation_score: 0.0,
            compression_ratio: 0.0,
            frequency_distribution: HashMap::new(),
            statistical_tests: HashMap::new(),
        };

        // Shannon entropy calculation
        quality.shannon_entropy = self.calculate_shannon_entropy(data);

        // Chi-square test
        quality.chi_square_p_value = self.chi_square_test(data);

        // Autocorrelation analysis
        quality.autocorrelation_score = self.autocorrelation_test(data);

        // Compression ratio test
        quality.compression_ratio = self.compression_test(data);

        // Frequency distribution analysis
        quality.frequency_distribution = self.frequency_distribution(data);

        // Additional statistical tests
        quality.statistical_tests = self.run_statistical_tests(data);

        // Calculate overall score
        quality.overall_score = self.calculate_overall_score(&quality);

        debug!("Entropy analysis complete: overall score = {:.4}", quality.overall_score);

        Ok(quality)
    }

    /// Run full test suite on data
    pub async fn run_full_test_suite(&self, data: &[u8]) -> Result<super::quantum_tests::TestResults> {
        super::quantum_tests::run_test_suite(data, &self.config).await
    }

    /// Calculate Shannon entropy
    fn calculate_shannon_entropy(&self, data: &[u8]) -> f64 {
        let mut counts = [0u32; 256];
        for &byte in data {
            counts[byte as usize] += 1;
        }

        let length = data.len() as f64;
        let mut entropy = 0.0;

        for &count in &counts {
            if count > 0 {
                let p = count as f64 / length;
                entropy -= p * p.log2();
            }
        }

        entropy / 8.0 // Normalize to 0-1 range
    }

    /// Chi-square goodness of fit test
    fn chi_square_test(&self, data: &[u8]) -> f64 {
        let mut counts = [0u32; 256];
        for &byte in data {
            counts[byte as usize] += 1;
        }

        let expected = data.len() as f64 / 256.0;
        let mut chi_square = 0.0;

        for &count in &counts {
            let observed = count as f64;
            let diff = observed - expected;
            chi_square += diff * diff / expected;
        }

        // Convert chi-square to p-value approximation
        // This is a simplified calculation - full implementation would use proper chi-square distribution
        let degrees_of_freedom = 255.0;
        let normalized = chi_square / degrees_of_freedom;
        
        if normalized < 0.8 {
            0.9
        } else if normalized < 1.2 {
            0.5
        } else {
            0.1
        }
    }

    /// Autocorrelation test for sequential independence
    fn autocorrelation_test(&self, data: &[u8]) -> f64 {
        if data.len() < 100 {
            return 0.5; // Not enough data
        }

        let mut max_correlation = 0.0;
        let test_length = (data.len() / 4).min(1000);

        // Test autocorrelation at various lags
        for lag in 1..16 {
            let correlation = self.calculate_autocorrelation(data, lag, test_length);
            max_correlation = max_correlation.max(correlation.abs());
        }

        // Good randomness should have low autocorrelation
        (1.0 - max_correlation).max(0.0)
    }

    fn calculate_autocorrelation(&self, data: &[u8], lag: usize, length: usize) -> f64 {
        if lag >= length {
            return 0.0;
        }

        let mut sum_xy = 0.0;
        let mut sum_x = 0.0;
        let mut sum_y = 0.0;
        let mut sum_x2 = 0.0;
        let mut sum_y2 = 0.0;
        let n = length - lag;

        for i in 0..n {
            let x = data[i] as f64;
            let y = data[i + lag] as f64;
            
            sum_xy += x * y;
            sum_x += x;
            sum_y += y;
            sum_x2 += x * x;
            sum_y2 += y * y;
        }

        let n_f = n as f64;
        let numerator = n_f * sum_xy - sum_x * sum_y;
        let denominator = ((n_f * sum_x2 - sum_x * sum_x) * (n_f * sum_y2 - sum_y * sum_y)).sqrt();

        if denominator == 0.0 {
            0.0
        } else {
            numerator / denominator
        }
    }

    /// Compression ratio test
    fn compression_test(&self, data: &[u8]) -> f64 {
        // Simple run-length encoding compression ratio
        let mut compressed_size = 0;
        let mut i = 0;

        while i < data.len() {
            let current = data[i];
            let mut run_length = 1;

            while i + run_length < data.len() && data[i + run_length] == current {
                run_length += 1;
            }

            compressed_size += 2; // 1 byte for value, 1 byte for count (simplified)
            i += run_length;
        }

        let ratio = compressed_size as f64 / data.len() as f64;
        
        // Good randomness should have compression ratio close to 1.0
        if ratio > 0.95 {
            1.0 - (ratio - 0.95) * 10.0
        } else {
            0.5
        }.max(0.0)
    }

    /// Calculate frequency distribution
    fn frequency_distribution(&self, data: &[u8]) -> HashMap<u8, f64> {
        let mut counts = HashMap::new();
        
        for &byte in data {
            *counts.entry(byte).or_insert(0) += 1;
        }

        let length = data.len() as f64;
        counts.iter()
            .map(|(&byte, &count)| (byte, count as f64 / length))
            .collect()
    }

    /// Run additional statistical tests
    fn run_statistical_tests(&self, data: &[u8]) -> HashMap<String, f64> {
        let mut results = HashMap::new();

        // Runs test
        results.insert("runs_test".to_string(), self.runs_test(data));

        // Serial test
        results.insert("serial_test".to_string(), self.serial_test(data));

        // Poker test
        results.insert("poker_test".to_string(), self.poker_test(data));

        // Gap test
        results.insert("gap_test".to_string(), self.gap_test(data));

        results
    }

    /// Runs test for randomness
    fn runs_test(&self, data: &[u8]) -> f64 {
        if data.len() < 20 {
            return 0.5;
        }

        let median = 128u8; // Expected median for random bytes
        let mut runs = 1;
        let mut n1 = 0; // Count of values above median
        let mut n2 = 0; // Count of values below median

        let mut last_above = data[0] > median;
        if last_above { n1 += 1; } else { n2 += 1; }

        for i in 1..data.len() {
            let above = data[i] > median;
            if above { n1 += 1; } else { n2 += 1; }
            
            if above != last_above {
                runs += 1;
                last_above = above;
            }
        }

        if n1 == 0 || n2 == 0 {
            return 0.0;
        }

        let expected_runs = 2.0 * (n1 * n2) as f64 / (n1 + n2) as f64 + 1.0;
        let variance = (expected_runs - 1.0) * (expected_runs - 2.0) / ((n1 + n2 - 1) as f64);
        
        if variance <= 0.0 {
            return 0.5;
        }

        let z = (runs as f64 - expected_runs).abs() / variance.sqrt();
        
        // Convert z-score to p-value approximation
        if z < 1.96 { 0.95 } else if z < 2.58 { 0.99 } else { 0.5 }
    }

    /// Serial test
    fn serial_test(&self, data: &[u8]) -> f64 {
        if data.len() < 100 {
            return 0.5;
        }

        let mut pairs = [[0u32; 256]; 256];
        
        for i in 0..data.len() - 1 {
            let first = data[i] as usize;
            let second = data[i + 1] as usize;
            pairs[first][second] += 1;
        }

        let expected = (data.len() - 1) as f64 / (256.0 * 256.0);
        let mut chi_square = 0.0;

        for i in 0..256 {
            for j in 0..256 {
                let observed = pairs[i][j] as f64;
                let diff = observed - expected;
                chi_square += diff * diff / expected;
            }
        }

        // Normalize and convert to score
        let normalized = chi_square / (256.0 * 256.0 - 1.0);
        if normalized < 1.0 { 0.9 } else { 0.1 }
    }

    /// Poker test (frequency of patterns)
    fn poker_test(&self, data: &[u8]) -> f64 {
        if data.len() < 20 {
            return 0.5;
        }

        // Test 4-bit patterns
        let mut pattern_counts = [0u32; 16];
        
        for &byte in data {
            let high_nibble = (byte >> 4) as usize;
            let low_nibble = (byte & 0x0F) as usize;
            pattern_counts[high_nibble] += 1;
            pattern_counts[low_nibble] += 1;
        }

        let total_patterns = data.len() * 2;
        let expected = total_patterns as f64 / 16.0;
        let mut chi_square = 0.0;

        for &count in &pattern_counts {
            let observed = count as f64;
            let diff = observed - expected;
            chi_square += diff * diff / expected;
        }

        // Convert to score
        let normalized = chi_square / 15.0; // 15 degrees of freedom
        if normalized < 1.0 { 0.9 } else { 0.1 }
    }

    /// Gap test
    fn gap_test(&self, data: &[u8]) -> f64 {
        if data.len() < 100 {
            return 0.5;
        }

        // Test gaps between occurrences of specific values
        let target = 0u8;
        let mut gaps = Vec::new();
        let mut last_occurrence = None;

        for (i, &byte) in data.iter().enumerate() {
            if byte == target {
                if let Some(last) = last_occurrence {
                    gaps.push(i - last - 1);
                }
                last_occurrence = Some(i);
            }
        }

        if gaps.is_empty() {
            return 0.5;
        }

        // Calculate average gap
        let avg_gap = gaps.iter().sum::<usize>() as f64 / gaps.len() as f64;
        let expected_gap = 255.0; // Expected gap for random data

        // Score based on how close to expected
        let ratio = if expected_gap > 0.0 { avg_gap / expected_gap } else { 1.0 };
        if ratio >= 0.8 && ratio <= 1.2 { 0.9 } else { 0.5 }
    }

    /// Calculate overall entropy quality score
    fn calculate_overall_score(&self, quality: &EntropyQuality) -> f64 {
        let weights = [
            (quality.shannon_entropy, 0.3),
            (quality.chi_square_p_value, 0.2),
            (quality.autocorrelation_score, 0.2),
            (quality.compression_ratio, 0.1),
        ];

        let mut weighted_sum = 0.0;
        let mut total_weight = 0.0;

        for (score, weight) in &weights {
            weighted_sum += score * weight;
            total_weight += weight;
        }

        // Add statistical test scores
        let test_scores: Vec<f64> = quality.statistical_tests.values().cloned().collect();
        if !test_scores.is_empty() {
            let avg_test_score = test_scores.iter().sum::<f64>() / test_scores.len() as f64;
            weighted_sum += avg_test_score * 0.2;
            total_weight += 0.2;
        }

        if total_weight > 0.0 {
            weighted_sum / total_weight
        } else {
            0.0
        }
    }
}

/// Entropy quality assessment result
#[derive(Debug, Clone)]
pub struct EntropyQuality {
    /// Overall quality score (0.0 - 1.0)
    pub overall_score: f64,
    
    /// Shannon entropy (0.0 - 1.0)
    pub shannon_entropy: f64,
    
    /// Chi-square test p-value
    pub chi_square_p_value: f64,
    
    /// Autocorrelation quality score
    pub autocorrelation_score: f64,
    
    /// Compression ratio score
    pub compression_ratio: f64,
    
    /// Frequency distribution of bytes
    pub frequency_distribution: HashMap<u8, f64>,
    
    /// Additional statistical test results
    pub statistical_tests: HashMap<String, f64>,
}

impl EntropyQuality {
    /// Check if entropy meets minimum quality threshold
    pub fn meets_threshold(&self, threshold: f64) -> bool {
        self.overall_score >= threshold
    }

    /// Get human-readable quality assessment
    pub fn quality_level(&self) -> &'static str {
        match self.overall_score {
            score if score >= 0.95 => "Excellent",
            score if score >= 0.90 => "Very Good", 
            score if score >= 0.80 => "Good",
            score if score >= 0.70 => "Fair",
            score if score >= 0.50 => "Poor",
            _ => "Very Poor",
        }
    }

    /// Get detailed quality report
    pub fn generate_report(&self) -> String {
        format!(
            "Entropy Quality Report\n\
            ======================\n\
            Overall Score: {:.4} ({})\n\
            Shannon Entropy: {:.4}\n\
            Chi-Square P-Value: {:.4}\n\
            Autocorrelation Score: {:.4}\n\
            Compression Ratio: {:.4}\n\
            Statistical Tests: {}\n\
            \n\
            Frequency Distribution Uniformity: {:.4}\n\
            Total Bytes Analyzed: {}\n",
            self.overall_score,
            self.quality_level(),
            self.shannon_entropy,
            self.chi_square_p_value,
            self.autocorrelation_score,
            self.compression_ratio,
            self.statistical_tests.len(),
            self.calculate_distribution_uniformity(),
            self.frequency_distribution.values().sum::<f64>() as u32
        )
    }

    fn calculate_distribution_uniformity(&self) -> f64 {
        if self.frequency_distribution.is_empty() {
            return 0.0;
        }

        let expected_frequency = 1.0 / 256.0;
        let mut sum_squared_deviations = 0.0;

        for i in 0..256 {
            let observed = self.frequency_distribution.get(&(i as u8)).unwrap_or(&0.0);
            let deviation = observed - expected_frequency;
            sum_squared_deviations += deviation * deviation;
        }

        let variance = sum_squared_deviations / 256.0;
        let uniformity = 1.0 - (variance * 256.0).min(1.0);
        uniformity.max(0.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::RngCore;

    #[tokio::test]
    async fn test_entropy_analysis() {
        let config = super::super::quantum_tests::TestSuiteConfig::default();
        let analyzer = EntropyAnalyzer::new(config).unwrap();

        // Generate random test data
        let mut data = vec![0u8; 1000];
        rand::rngs::OsRng.fill_bytes(&mut data);

        let quality = analyzer.analyze_entropy(&data).await.unwrap();
        
        assert!(quality.overall_score > 0.0);
        assert!(quality.shannon_entropy > 0.5);
        assert!(!quality.frequency_distribution.is_empty());
        assert!(!quality.statistical_tests.is_empty());
    }

    #[test]
    fn test_shannon_entropy() {
        let config = super::super::quantum_tests::TestSuiteConfig::default();
        let analyzer = EntropyAnalyzer::new(config).unwrap();

        // Perfect entropy case (all bytes equally distributed)
        let mut perfect_data = Vec::new();
        for i in 0..256u8 {
            for _ in 0..4 {
                perfect_data.push(i);
            }
        }
        
        let entropy = analyzer.calculate_shannon_entropy(&perfect_data);
        assert!(entropy > 0.99); // Should be close to 1.0

        // Zero entropy case (all same byte)
        let zero_entropy_data = vec![42u8; 1000];
        let entropy = analyzer.calculate_shannon_entropy(&zero_entropy_data);
        assert!(entropy < 0.01); // Should be close to 0.0
    }

    #[test]
    fn test_quality_levels() {
        let quality = EntropyQuality {
            overall_score: 0.98,
            shannon_entropy: 0.99,
            chi_square_p_value: 0.95,
            autocorrelation_score: 0.90,
            compression_ratio: 0.95,
            frequency_distribution: HashMap::new(),
            statistical_tests: HashMap::new(),
        };

        assert_eq!(quality.quality_level(), "Excellent");
        assert!(quality.meets_threshold(0.95));
    }
}