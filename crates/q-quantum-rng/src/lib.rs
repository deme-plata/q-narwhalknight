/// Quantum Random Number Generation for Q-NarwhalKnight Phase 2+
/// Provides hardware QRNG integration with entropy quality assessment

use q_types::*;
use anyhow::Result;
use async_trait::async_trait;
use rand::{RngCore, SeedableRng};
use sha3::{Digest, Sha3_256};
use std::collections::VecDeque;
use std::sync::Arc;
use tokio::sync::{RwLock, Mutex};
use tracing::{debug, info, warn, error};

pub mod hardware;
pub mod entropy_analysis;
pub mod pooling;
pub mod quantum_tests;

pub use hardware::{QRNGHardware, HardwareProvider};
pub use entropy_analysis::{EntropyAnalyzer, EntropyQuality};
pub use pooling::{EntropyPool, PoolingStrategy};

/// Main QRNG interface for quantum random number generation
pub struct QuantumRNG {
    /// Current operational phase
    phase: Phase,
    
    /// Hardware QRNG provider (Phase 2+)
    hardware: Option<Box<dyn QRNGHardware + Send + Sync>>,
    
    /// Entropy quality analyzer
    analyzer: EntropyAnalyzer,
    
    /// Entropy pool for buffering and mixing
    pool: Arc<Mutex<EntropyPool>>,
    
    /// Statistics tracking
    stats: RwLock<QRNGStats>,
    
    /// Configuration
    config: QRNGConfig,
}

/// QRNG configuration parameters
#[derive(Debug, Clone)]
pub struct QRNGConfig {
    /// Minimum entropy quality threshold (0.0 - 1.0)
    pub min_entropy_quality: f64,
    
    /// Buffer size for entropy pool
    pub pool_size: usize,
    
    /// Hardware polling interval (milliseconds)
    pub polling_interval_ms: u64,
    
    /// Enable continuous entropy analysis
    pub enable_analysis: bool,
    
    /// Quantum test suite configuration
    pub test_suite_config: quantum_tests::TestSuiteConfig,
    
    /// Fallback to classical RNG if hardware fails
    pub fallback_enabled: bool,
}

impl Default for QRNGConfig {
    fn default() -> Self {
        Self {
            min_entropy_quality: 0.95,
            pool_size: 8192,
            polling_interval_ms: 100,
            enable_analysis: true,
            test_suite_config: quantum_tests::TestSuiteConfig::default(),
            fallback_enabled: true,
        }
    }
}

/// QRNG performance and quality statistics
#[derive(Debug, Clone)]
pub struct QRNGStats {
    pub total_bytes_generated: u64,
    pub hardware_bytes_generated: u64,
    pub fallback_bytes_generated: u64,
    pub average_entropy_quality: f64,
    pub hardware_failures: u64,
    pub entropy_test_failures: u64,
    pub uptime_seconds: u64,
    pub last_hardware_poll: Option<chrono::DateTime<chrono::Utc>>,
    pub generation_rate_bytes_per_sec: f64,
}

impl QuantumRNG {
    /// Create new QRNG for specified phase
    pub async fn new(phase: Phase, config: QRNGConfig) -> Result<Self> {
        info!("Initializing Quantum RNG for {:?}", phase);

        let hardware = match phase {
            Phase::Phase0 | Phase::Phase1 => {
                info!("Phase 0/1: Using simulated QRNG");
                None
            }
            Phase::Phase2 => {
                info!("Phase 2: Initializing hardware QRNG");
                Self::init_hardware().await?
            }
            _ => {
                info!("Phase 3+: Initializing advanced quantum hardware");
                Self::init_advanced_hardware().await?
            }
        };

        let analyzer = EntropyAnalyzer::new(config.test_suite_config.clone())?;
        let pool = Arc::new(Mutex::new(EntropyPool::new(config.pool_size)));

        let qrng = Self {
            phase,
            hardware,
            analyzer,
            pool,
            stats: RwLock::new(QRNGStats {
                total_bytes_generated: 0,
                hardware_bytes_generated: 0,
                fallback_bytes_generated: 0,
                average_entropy_quality: 0.0,
                hardware_failures: 0,
                entropy_test_failures: 0,
                uptime_seconds: 0,
                last_hardware_poll: None,
                generation_rate_bytes_per_sec: 0.0,
            }),
            config,
        };

        // Start background entropy collection if hardware is available
        if qrng.hardware.is_some() {
            qrng.start_entropy_collection().await?;
        }

        info!("Quantum RNG initialized successfully for {:?}", phase);
        Ok(qrng)
    }

    /// Initialize hardware QRNG for Phase 2
    async fn init_hardware() -> Result<Option<Box<dyn QRNGHardware + Send + Sync>>> {
        #[cfg(feature = "hardware")]
        {
            // Try different hardware providers in order of preference
            let providers = vec![
                HardwareProvider::QuantumOptics,    // ID Quantique, PicoQuant, etc.
                HardwareProvider::ThermalNoise,     // Thermal noise sources
                HardwareProvider::RadioNoise,       // Radio frequency noise
                HardwareProvider::ChaosLaser,       // Chaos-based laser systems
            ];

            for provider in providers {
                match hardware::create_hardware_rng(provider).await {
                    Ok(hardware) => {
                        info!("Successfully initialized {:?} QRNG hardware", provider);
                        return Ok(Some(hardware));
                    }
                    Err(e) => {
                        warn!("Failed to initialize {:?}: {}", provider, e);
                        continue;
                    }
                }
            }

            warn!("No hardware QRNG available, will use fallback");
            Ok(None)
        }

        #[cfg(not(feature = "hardware"))]
        {
            info!("Hardware QRNG disabled at compile time");
            Ok(None)
        }
    }

    /// Initialize advanced quantum hardware for Phase 3+
    async fn init_advanced_hardware() -> Result<Option<Box<dyn QRNGHardware + Send + Sync>>> {
        // Phase 3+ would support more advanced quantum sources
        Self::init_hardware().await
    }

    /// Generate quantum random bytes
    pub async fn generate_bytes(&self, count: usize) -> Result<Vec<u8>> {
        let start_time = std::time::Instant::now();
        
        let bytes = match &self.hardware {
            Some(hardware) => {
                debug!("Generating {} bytes from quantum hardware", count);
                match self.generate_from_hardware(hardware.as_ref(), count).await {
                    Ok(bytes) => {
                        self.update_hardware_stats(bytes.len()).await;
                        bytes
                    }
                    Err(e) => {
                        error!("Hardware QRNG failed: {}, falling back", e);
                        self.increment_hardware_failures().await;
                        
                        if self.config.fallback_enabled {
                            self.generate_fallback(count).await?
                        } else {
                            return Err(e);
                        }
                    }
                }
            }
            None => {
                debug!("Generating {} bytes from fallback QRNG", count);
                self.generate_fallback(count).await?
            }
        };

        // Analyze entropy quality if enabled
        if self.config.enable_analysis {
            let quality = self.analyzer.analyze_entropy(&bytes).await?;
            
            if quality.overall_score < self.config.min_entropy_quality {
                warn!("Entropy quality below threshold: {:.3} < {:.3}", 
                      quality.overall_score, self.config.min_entropy_quality);
                
                if !self.config.fallback_enabled {
                    return Err(anyhow::anyhow!("Entropy quality insufficient"));
                }
                
                // Regenerate with fallback
                return Ok(self.generate_fallback(count).await?);
            }

            self.update_entropy_stats(quality.overall_score).await;
        }

        // Update generation rate statistics
        let generation_time = start_time.elapsed();
        self.update_generation_rate(bytes.len(), generation_time).await;

        // Update total bytes counter
        {
            let mut stats = self.stats.write().await;
            stats.total_bytes_generated += bytes.len() as u64;
        }

        debug!("Generated {} quantum random bytes in {:?}", bytes.len(), generation_time);
        Ok(bytes)
    }

    /// Generate random bytes from hardware
    async fn generate_from_hardware(
        &self, 
        hardware: &dyn QRNGHardware, 
        count: usize
    ) -> Result<Vec<u8>> {
        // Try to get bytes from entropy pool first
        {
            let mut pool = self.pool.lock().await;
            if let Some(bytes) = pool.extract_bytes(count) {
                return Ok(bytes);
            }
        }

        // Generate directly from hardware if pool is empty
        hardware.generate_random_bytes(count).await
    }

    /// Generate fallback randomness using classical methods
    async fn generate_fallback(&self, count: usize) -> Result<Vec<u8>> {
        // Use cryptographically secure fallback
        let mut bytes = vec![0u8; count];
        
        match self.phase {
            Phase::Phase0 | Phase::Phase1 => {
                // Classical CSPRNG
                let mut rng = rand::rngs::OsRng;
                rng.fill_bytes(&mut bytes);
            }
            _ => {
                // Enhanced entropy mixing for Phase 2+
                let mut rng = rand::rngs::OsRng;
                let mut base_entropy = vec![0u8; count];
                rng.fill_bytes(&mut base_entropy);
                
                // Mix with system entropy and timing
                let mut hasher = Sha3_256::new();
                hasher.update(&base_entropy);
                hasher.update(&std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_nanos()
                    .to_be_bytes());
                
                // Expand using XOF
                let seed = hasher.finalize();
                let mut csprng = rand::rngs::StdRng::from_seed(seed.into());
                csprng.fill_bytes(&mut bytes);
            }
        }

        // Update fallback stats
        {
            let mut stats = self.stats.write().await;
            stats.fallback_bytes_generated += count as u64;
        }

        Ok(bytes)
    }

    /// Start background entropy collection from hardware
    async fn start_entropy_collection(&self) -> Result<()> {
        if self.hardware.is_none() {
            return Ok(());
        }

        let pool = self.pool.clone();
        let hardware = self.hardware.as_ref().unwrap();
        let interval = tokio::time::Duration::from_millis(self.config.polling_interval_ms);

        // Clone what we need for the background task
        let hardware_clone = hardware.clone_box();
        let min_quality = self.config.min_entropy_quality;
        let analyzer = self.analyzer.clone();

        tokio::spawn(async move {
            let mut interval_timer = tokio::time::interval(interval);
            
            loop {
                interval_timer.tick().await;
                
                match hardware_clone.generate_random_bytes(256).await {
                    Ok(entropy) => {
                        // Analyze quality before adding to pool
                        match analyzer.analyze_entropy(&entropy).await {
                            Ok(quality) if quality.overall_score >= min_quality => {
                                let mut pool_guard = pool.lock().await;
                                pool_guard.add_entropy(entropy, quality.overall_score);
                            }
                            Ok(quality) => {
                                warn!("Low quality entropy rejected: {:.3}", quality.overall_score);
                            }
                            Err(e) => {
                                error!("Entropy analysis failed: {}", e);
                            }
                        }
                    }
                    Err(e) => {
                        error!("Background entropy collection failed: {}", e);
                        // Add exponential backoff here
                    }
                }
            }
        });

        info!("Started background entropy collection");
        Ok(())
    }

    /// Generate quantum random seed for cryptographic use
    pub async fn generate_seed<const N: usize>(&self) -> Result<[u8; N]> {
        let bytes = self.generate_bytes(N).await?;
        let mut seed = [0u8; N];
        seed.copy_from_slice(&bytes[..N]);
        Ok(seed)
    }

    /// Generate quantum random number in range
    pub async fn generate_range(&self, min: u64, max: u64) -> Result<u64> {
        if min >= max {
            return Err(anyhow::anyhow!("Invalid range: min >= max"));
        }

        let range = max - min;
        let bytes_needed = ((range as f64).log2().ceil() / 8.0).ceil() as usize;
        let bytes = self.generate_bytes(bytes_needed).await?;

        // Convert bytes to integer
        let mut value = 0u64;
        for (i, &byte) in bytes.iter().enumerate() {
            value |= (byte as u64) << (i * 8);
        }

        // Reduce modulo range and add offset
        Ok((value % range) + min)
    }

    /// Get QRNG statistics
    pub async fn get_stats(&self) -> QRNGStats {
        self.stats.read().await.clone()
    }

    /// Get current entropy pool status
    pub async fn get_pool_status(&self) -> pooling::PoolStatus {
        let pool = self.pool.lock().await;
        pool.get_status()
    }

    /// Force entropy pool refresh
    pub async fn refresh_pool(&self) -> Result<()> {
        if let Some(ref hardware) = self.hardware {
            let entropy = hardware.generate_random_bytes(1024).await?;
            let quality = self.analyzer.analyze_entropy(&entropy).await?;
            
            let mut pool = self.pool.lock().await;
            pool.add_entropy(entropy, quality.overall_score);
            
            info!("Entropy pool refreshed with {} bytes", 1024);
        }
        Ok(())
    }

    /// Perform quantum randomness test suite
    pub async fn run_test_suite(&self) -> Result<quantum_tests::TestResults> {
        let test_data = self.generate_bytes(10000).await?; // 10KB for testing
        self.analyzer.run_full_test_suite(&test_data).await
    }

    /// Update hardware generation statistics
    async fn update_hardware_stats(&self, bytes_generated: usize) {
        let mut stats = self.stats.write().await;
        stats.hardware_bytes_generated += bytes_generated as u64;
        stats.last_hardware_poll = Some(chrono::Utc::now());
    }

    /// Increment hardware failure counter
    async fn increment_hardware_failures(&self) {
        let mut stats = self.stats.write().await;
        stats.hardware_failures += 1;
    }

    /// Update entropy quality statistics
    async fn update_entropy_stats(&self, quality: f64) {
        let mut stats = self.stats.write().await;
        let total_samples = (stats.total_bytes_generated / 1024).max(1); // Approximate samples
        stats.average_entropy_quality = (stats.average_entropy_quality * (total_samples - 1) as f64 + quality) 
            / total_samples as f64;
    }

    /// Update generation rate statistics
    async fn update_generation_rate(&self, bytes: usize, duration: std::time::Duration) {
        let mut stats = self.stats.write().await;
        let rate = bytes as f64 / duration.as_secs_f64();
        
        // Exponential moving average
        stats.generation_rate_bytes_per_sec = if stats.generation_rate_bytes_per_sec == 0.0 {
            rate
        } else {
            stats.generation_rate_bytes_per_sec * 0.9 + rate * 0.1
        };
    }

    /// Get current phase
    pub fn get_phase(&self) -> Phase {
        self.phase
    }

    /// Check if hardware QRNG is available
    pub fn has_hardware(&self) -> bool {
        self.hardware.is_some()
    }
}

/// Trait for quantum random number generators
#[async_trait]
pub trait QuantumRandomness: Send + Sync {
    async fn generate_quantum_bytes(&self, count: usize) -> Result<Vec<u8>>;
    async fn generate_quantum_seed<const N: usize>(&self) -> Result<[u8; N]>;
    async fn get_entropy_quality(&self) -> Result<f64>;
}

#[async_trait]
impl QuantumRandomness for QuantumRNG {
    async fn generate_quantum_bytes(&self, count: usize) -> Result<Vec<u8>> {
        self.generate_bytes(count).await
    }

    async fn generate_quantum_seed<const N: usize>(&self) -> Result<[u8; N]> {
        self.generate_seed().await
    }

    async fn get_entropy_quality(&self) -> Result<f64> {
        let stats = self.get_stats().await;
        Ok(stats.average_entropy_quality)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_qrng_creation() {
        let config = QRNGConfig::default();
        let qrng = QuantumRNG::new(Phase::Phase0, config).await.unwrap();
        
        assert_eq!(qrng.get_phase(), Phase::Phase0);
        assert!(!qrng.has_hardware()); // Phase 0 should not have hardware
    }

    #[tokio::test]
    async fn test_fallback_generation() {
        let config = QRNGConfig::default();
        let qrng = QuantumRNG::new(Phase::Phase0, config).await.unwrap();
        
        let bytes = qrng.generate_bytes(32).await.unwrap();
        assert_eq!(bytes.len(), 32);
        
        // Basic randomness check - bytes should not be all zeros
        assert!(bytes.iter().any(|&b| b != 0));
    }

    #[tokio::test]
    async fn test_seed_generation() {
        let config = QRNGConfig::default();
        let qrng = QuantumRNG::new(Phase::Phase0, config).await.unwrap();
        
        let seed: [u8; 32] = qrng.generate_seed().await.unwrap();
        assert_eq!(seed.len(), 32);
    }

    #[tokio::test]
    async fn test_range_generation() {
        let config = QRNGConfig::default();
        let qrng = QuantumRNG::new(Phase::Phase0, config).await.unwrap();
        
        for _ in 0..100 {
            let value = qrng.generate_range(10, 20).await.unwrap();
            assert!(value >= 10 && value < 20);
        }
    }

    #[tokio::test]
    async fn test_statistics_tracking() {
        let config = QRNGConfig::default();
        let qrng = QuantumRNG::new(Phase::Phase0, config).await.unwrap();
        
        let _ = qrng.generate_bytes(1000).await.unwrap();
        
        let stats = qrng.get_stats().await;
        assert_eq!(stats.total_bytes_generated, 1000);
        assert_eq!(stats.fallback_bytes_generated, 1000);
        assert_eq!(stats.hardware_bytes_generated, 0);
    }
}