/// Quantum Random Number Generator integration for Tor circuit seeding
/// Provides quantum-enhanced entropy for circuit creation and timing obfuscation
use anyhow::{Context, Result};
use q_quantum_rng::{QRNGConfig, QuantumRNG, QuantumRandomness};
use rand_chacha::{ChaChaRng, rand_core::{RngCore, SeedableRng}};  // Use compatible trait versions
use serde::{Deserialize, Serialize};
use std::{
    sync::Arc,
    time::{Duration, Instant, SystemTime},
};
use tokio::sync::{Mutex, RwLock};
use tracing::{debug, info, warn};

/// Type alias for quantum seeding manager
pub type QuantumSeedingManager = QuantumEntropyPool;

/// Quantum entropy pool for Tor operations
pub struct QuantumEntropyPool {
    /// Primary quantum RNG
    primary_qrng: Arc<QuantumRNG>,
    /// Backup quantum RNG (optional)
    backup_qrng: Option<Arc<QuantumRNG>>,
    /// ChaCha20 PRNG seeded with quantum entropy
    quantum_prng: Arc<Mutex<ChaChaRng>>,
    /// Entropy quality metrics
    entropy_quality: Arc<RwLock<EntropyQuality>>,
    /// Last quantum reseed time
    last_reseed: Arc<RwLock<Instant>>,
    /// Configuration
    config: QuantumSeedingConfig,
}

/// Configuration for quantum seeding
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumSeedingConfig {
    /// Minimum entropy quality threshold (0.0 - 1.0)
    pub min_entropy_quality: f64,
    /// Reseed interval for PRNG
    pub reseed_interval: Duration,
    /// Enable backup QRNG fallback
    pub enable_backup_qrng: bool,
    /// Quantum test interval
    pub quantum_test_interval: Duration,
    /// Fallback to classical randomness if quantum fails
    pub classical_fallback: bool,
    /// Circuit entropy buffer size
    pub entropy_buffer_size: usize,
}

impl Default for QuantumSeedingConfig {
    fn default() -> Self {
        Self {
            min_entropy_quality: 0.95,                 // 95% minimum quality
            reseed_interval: Duration::from_secs(300), // 5 minutes
            enable_backup_qrng: true,
            quantum_test_interval: Duration::from_secs(60), // 1 minute
            classical_fallback: true,
            entropy_buffer_size: 1024, // 1KB buffer
        }
    }
}

/// Entropy quality assessment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntropyQuality {
    /// Primary QRNG quality (0.0 - 1.0)
    pub primary_quality: f64,
    /// Backup QRNG quality (0.0 - 1.0)
    pub backup_quality: Option<f64>,
    /// Overall entropy score
    pub overall_score: f64,
    /// Last assessment time
    pub last_assessment: SystemTime,
    /// Number of quantum tests passed
    pub tests_passed: u64,
    /// Number of quantum tests failed
    pub tests_failed: u64,
}

impl QuantumEntropyPool {
    /// Create a new quantum entropy pool
    pub async fn new(config: QuantumSeedingConfig) -> Result<Self> {
        info!("🌊 Initializing quantum entropy pool for Tor circuits");

        // Initialize primary QRNG
        let qrng_config = QRNGConfig {
            min_entropy_quality: config.min_entropy_quality,
            pool_size: config.entropy_buffer_size,
            polling_interval_ms: config.quantum_test_interval.as_millis() as u64,
            enable_analysis: true,
            test_suite_config: Default::default(),
            fallback_enabled: config.classical_fallback,
        };

        let primary_qrng = Arc::new(
            QuantumRNG::new(q_types::Phase::Phase2, qrng_config.clone())
                .await
                .context("Failed to initialize primary QRNG")?,
        );

        // Initialize backup QRNG if enabled
        let backup_qrng = if config.enable_backup_qrng {
            let backup_config = qrng_config.clone();

            match QuantumRNG::new(q_types::Phase::Phase2, backup_config).await {
                Ok(backup) => Some(Arc::new(backup)),
                Err(e) => {
                    warn!("Failed to initialize backup QRNG: {}", e);
                    None
                }
            }
        } else {
            None
        };

        // Generate initial quantum seed
        let quantum_seed = Self::generate_quantum_seed(&primary_qrng).await?;
        let quantum_prng = Arc::new(Mutex::new(ChaChaRng::from_seed(quantum_seed)));

        let entropy_quality = Arc::new(RwLock::new(EntropyQuality {
            primary_quality: 1.0,
            backup_quality: backup_qrng.as_ref().map(|_| 1.0),
            overall_score: 1.0,
            last_assessment: SystemTime::now(),
            tests_passed: 0,
            tests_failed: 0,
        }));

        let pool = Self {
            primary_qrng,
            backup_qrng,
            quantum_prng,
            entropy_quality,
            last_reseed: Arc::new(RwLock::new(Instant::now())),
            config,
        };

        // Start background entropy monitoring
        pool.start_entropy_monitoring().await;

        info!("✅ Quantum entropy pool initialized successfully");
        Ok(pool)
    }

    /// Generate quantum seed for circuit creation
    pub async fn generate_circuit_seed(&self) -> Result<[u8; 32]> {
        // Check if reseed is needed
        if self.should_reseed().await {
            self.reseed_prng().await?;
        }

        // Generate seed using quantum PRNG
        let mut quantum_prng = self.quantum_prng.lock().await;
        let mut seed = [0u8; 32];
        quantum_prng.fill_bytes(&mut seed);

        debug!("Generated quantum circuit seed: {} bytes", seed.len());
        Ok(seed)
    }

    /// Generate quantum timing delays for anonymity
    pub async fn generate_quantum_delay(
        &self,
        min_delay: Duration,
        max_delay: Duration,
    ) -> Result<Duration> {
        let mut quantum_prng = self.quantum_prng.lock().await;

        let min_ms = min_delay.as_millis() as u64;
        let max_ms = max_delay.as_millis() as u64;

        if min_ms >= max_ms {
            return Ok(min_delay);
        }

        let range = max_ms - min_ms;
        let mut delay_bytes = [0u8; 8];
        quantum_prng.fill_bytes(&mut delay_bytes);

        let random_value = u64::from_be_bytes(delay_bytes);
        let delay_ms = min_ms + (random_value % range);

        Ok(Duration::from_millis(delay_ms))
    }

    /// Generate quantum nonce for circuit identification
    pub async fn generate_quantum_nonce(&self, length: usize) -> Result<Vec<u8>> {
        let mut quantum_prng = self.quantum_prng.lock().await;
        let mut nonce = vec![0u8; length];
        quantum_prng.fill_bytes(&mut nonce);

        debug!("Generated quantum nonce: {} bytes", length);
        Ok(nonce)
    }

    /// Get current entropy quality assessment
    pub async fn get_entropy_quality(&self) -> EntropyQuality {
        self.entropy_quality.read().await.clone()
    }

    /// Check if PRNG needs reseeding
    async fn should_reseed(&self) -> bool {
        let last_reseed = *self.last_reseed.read().await;
        last_reseed.elapsed() > self.config.reseed_interval
    }

    /// Reseed the PRNG with fresh quantum entropy
    async fn reseed_prng(&self) -> Result<()> {
        info!("🔄 Reseeding quantum PRNG with fresh entropy");

        let quantum_seed = match Self::generate_quantum_seed(&self.primary_qrng).await {
            Ok(seed) => seed,
            Err(_) => {
                if let Some(backup) = &self.backup_qrng {
                    warn!("Primary QRNG failed, using backup");
                    Self::generate_quantum_seed(backup).await?
                } else {
                    return Err(anyhow::anyhow!("No backup QRNG available"));
                }
            }
        };

        // Replace the PRNG with a newly seeded one
        let mut quantum_prng = self.quantum_prng.lock().await;
        *quantum_prng = ChaChaRng::from_seed(quantum_seed);

        // Update last reseed time
        let mut last_reseed = self.last_reseed.write().await;
        *last_reseed = Instant::now();

        info!("✅ PRNG reseeded successfully");
        Ok(())
    }

    /// Generate a 32-byte quantum seed
    async fn generate_quantum_seed(qrng: &QuantumRNG) -> Result<[u8; 32]> {
        let quantum_bytes = qrng
            .generate_bytes(32)
            .await
            .context("Failed to generate quantum seed")?;

        if quantum_bytes.len() != 32 {
            anyhow::bail!(
                "Invalid quantum seed length: expected 32, got {}",
                quantum_bytes.len()
            );
        }

        let mut seed = [0u8; 32];
        seed.copy_from_slice(&quantum_bytes);
        Ok(seed)
    }

    /// Start background entropy quality monitoring
    async fn start_entropy_monitoring(&self) {
        let primary_qrng = Arc::clone(&self.primary_qrng);
        let backup_qrng = self.backup_qrng.clone();
        let entropy_quality = Arc::clone(&self.entropy_quality);
        let test_interval = self.config.quantum_test_interval;

        tokio::spawn(async move {
            let mut interval = tokio::time::interval(test_interval);

            loop {
                interval.tick().await;

                // Test primary QRNG quality
                let primary_quality = match primary_qrng.get_entropy_quality().await {
                    Ok(quality) => quality,
                    Err(e) => {
                        warn!("Primary QRNG quality test failed: {}", e);
                        0.0
                    }
                };

                // Test backup QRNG quality if available
                let backup_quality = if let Some(backup) = &backup_qrng {
                    match backup.get_entropy_quality().await {
                        Ok(quality) => Some(quality),
                        Err(e) => {
                            warn!("Backup QRNG quality test failed: {}", e);
                            Some(0.0)
                        }
                    }
                } else {
                    None
                };

                // Calculate overall score
                let overall_score = if let Some(backup_qual) = backup_quality {
                    (primary_quality + backup_qual) / 2.0
                } else {
                    primary_quality
                };

                // Update entropy quality
                {
                    let mut quality = entropy_quality.write().await;
                    let tests_passed = if overall_score >= 0.9 {
                        quality.tests_passed + 1
                    } else {
                        quality.tests_passed
                    };
                    let tests_failed = if overall_score < 0.9 {
                        quality.tests_failed + 1
                    } else {
                        quality.tests_failed
                    };

                    *quality = EntropyQuality {
                        primary_quality,
                        backup_quality,
                        overall_score,
                        last_assessment: SystemTime::now(),
                        tests_passed,
                        tests_failed,
                    };
                }

                if overall_score < 0.9 {
                    warn!("⚠️ Low entropy quality detected: {:.2}", overall_score);
                } else {
                    debug!("✅ Entropy quality check passed: {:.2}", overall_score);
                }
            }
        });
    }

    /// Generate quantum-enhanced circuit parameters
    pub async fn generate_circuit_parameters(&self) -> Result<CircuitParameters> {
        let seed = self.generate_circuit_seed().await?;
        let nonce = self.generate_quantum_nonce(12).await?;
        let timing_offset = self
            .generate_quantum_delay(Duration::from_millis(10), Duration::from_millis(500))
            .await?;

        // Generate quantum-enhanced hop selection weights
        let mut quantum_prng = self.quantum_prng.lock().await;
        let mut hop_weights = [0u8; 16];
        quantum_prng.fill_bytes(&mut hop_weights);

        Ok(CircuitParameters {
            seed,
            nonce,
            timing_offset,
            hop_weights: hop_weights.to_vec(),
            created_at: SystemTime::now(),
        })
    }

    /// Test quantum randomness quality
    pub async fn test_randomness_quality(&self, sample_size: usize) -> Result<RandomnessTest> {
        info!(
            "🧪 Testing quantum randomness quality with {} samples",
            sample_size
        );

        let mut quantum_prng = self.quantum_prng.lock().await;
        let mut samples = vec![0u8; sample_size];
        quantum_prng.fill_bytes(&mut samples);
        drop(quantum_prng);

        // Perform basic statistical tests
        let entropy_score = Self::calculate_entropy(&samples);
        let chi_squared = Self::chi_squared_test(&samples);
        let runs_test = Self::runs_test(&samples);

        let passed_tests = [
            entropy_score > 7.0, // Good entropy
            chi_squared < 300.0, // Not too structured
            runs_test > 0.1,     // Good run distribution
        ]
        .iter()
        .filter(|&&x| x)
        .count();

        let quality_score = passed_tests as f64 / 3.0;

        Ok(RandomnessTest {
            sample_size,
            entropy_score,
            chi_squared,
            runs_test,
            quality_score,
            passed_tests,
            total_tests: 3,
        })
    }

    /// Calculate Shannon entropy of byte samples
    fn calculate_entropy(samples: &[u8]) -> f64 {
        let mut frequencies = [0usize; 256];
        for &byte in samples {
            frequencies[byte as usize] += 1;
        }

        let total = samples.len() as f64;
        let mut entropy = 0.0;

        for &freq in &frequencies {
            if freq > 0 {
                let probability = freq as f64 / total;
                entropy -= probability * probability.log2();
            }
        }

        entropy
    }

    /// Perform chi-squared test for uniformity
    fn chi_squared_test(samples: &[u8]) -> f64 {
        let mut frequencies = [0usize; 256];
        for &byte in samples {
            frequencies[byte as usize] += 1;
        }

        let expected = samples.len() as f64 / 256.0;
        let mut chi_squared = 0.0;

        for freq in &frequencies {
            let observed = *freq as f64;
            let diff = observed - expected;
            chi_squared += diff * diff / expected;
        }

        chi_squared
    }

    /// Perform runs test for independence
    fn runs_test(samples: &[u8]) -> f64 {
        if samples.len() < 2 {
            return 0.0;
        }

        let median = 128u8; // Assume median for bytes is 128
        let mut runs = 1;
        let mut above_median = 0;
        let mut below_median = 0;

        let mut last_above = samples[0] > median;
        if last_above {
            above_median += 1;
        } else {
            below_median += 1;
        }

        for &sample in &samples[1..] {
            let current_above = sample > median;
            if current_above != last_above {
                runs += 1;
            }
            if current_above {
                above_median += 1;
            } else {
                below_median += 1;
            }
            last_above = current_above;
        }

        let n = samples.len() as f64;
        let expected_runs = (2.0 * above_median as f64 * below_median as f64) / n + 1.0;
        let variance = (expected_runs - 1.0) * (expected_runs - 2.0) / (n - 1.0);
        let z_score = (runs as f64 - expected_runs) / variance.sqrt();

        // Return p-value approximation
        1.0 - z_score.abs() / 3.0 // Simplified p-value
    }
}

/// Quantum-enhanced circuit parameters
#[derive(Debug, Clone)]
pub struct CircuitParameters {
    /// Quantum seed for circuit
    pub seed: [u8; 32],
    /// Quantum nonce for identification
    pub nonce: Vec<u8>,
    /// Quantum timing offset
    pub timing_offset: Duration,
    /// Hop selection weights
    pub hop_weights: Vec<u8>,
    /// Creation timestamp
    pub created_at: SystemTime,
}

/// Randomness quality test results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RandomnessTest {
    pub sample_size: usize,
    pub entropy_score: f64,
    pub chi_squared: f64,
    pub runs_test: f64,
    pub quality_score: f64,
    pub passed_tests: usize,
    pub total_tests: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_entropy_calculation() {
        // Test with perfectly uniform distribution
        let uniform_samples: Vec<u8> = (0..=255).collect();
        let entropy = QuantumEntropyPool::calculate_entropy(&uniform_samples);
        assert!(entropy > 7.5); // Should be close to 8.0 for perfect uniformity

        // Test with biased distribution
        let biased_samples = vec![0u8; 256];
        let entropy = QuantumEntropyPool::calculate_entropy(&biased_samples);
        assert!(entropy < 1.0); // Should be 0 for completely uniform
    }

    #[test]
    fn test_chi_squared() {
        // Test with uniform distribution
        let uniform_samples: Vec<u8> = (0..=255).collect();
        let chi_squared = QuantumEntropyPool::chi_squared_test(&uniform_samples);
        assert!(chi_squared < 300.0); // Should be reasonably low

        // Test with biased distribution
        let biased_samples = vec![0u8; 256];
        let chi_squared = QuantumEntropyPool::chi_squared_test(&biased_samples);
        assert!(chi_squared > 1000.0); // Should be very high
    }

    #[test]
    fn test_runs_test() {
        // Test with alternating pattern
        let alternating: Vec<u8> = (0..100)
            .map(|i| if i % 2 == 0 { 50 } else { 200 })
            .collect();
        let runs_result = QuantumEntropyPool::runs_test(&alternating);
        assert!(runs_result > 0.0);

        // Test with constant values
        let constant = vec![100u8; 100];
        let runs_result = QuantumEntropyPool::runs_test(&constant);
        assert!(runs_result < 0.5); // Should indicate poor randomness
    }
}
