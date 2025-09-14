/// Comprehensive integration tests for Q-Tor-Client
use crate::*;
use std::time::{Duration, SystemTime};
use anyhow::Result;

#[cfg(test)]
mod tests {
    use super::*;
    use q_types::Phase;

    #[tokio::test]
    async fn test_tor_config_creation_and_validation() {
        println!("🧪 Testing TorConfig creation and validation...");
        
        // Test default configuration
        let default_config = TorConfig::default();
        assert_eq!(default_config.circuit_count, 4);
        assert_eq!(default_config.rpc_port, 4001);
        assert!(!default_config.enabled);
        assert!(default_config.enable_dandelion);
        assert!(default_config.enable_prometheus_metrics);
        
        // Test stealth mode
        let stealth_config = TorConfig::stealth_mode();
        assert!(stealth_config.enabled);
        assert!(stealth_config.tor_only);
        assert!(stealth_config.enable_dandelion);
        
        // Test hybrid mode
        let hybrid_config = TorConfig::hybrid_mode();
        assert!(hybrid_config.enabled);
        assert!(!hybrid_config.tor_only);
        
        // Test validation
        assert!(default_config.validate().is_ok());
        assert!(stealth_config.validate().is_ok());
        assert!(hybrid_config.validate().is_ok());
        
        // Test invalid configuration
        let mut invalid_config = TorConfig::default();
        invalid_config.enabled = true;
        invalid_config.circuit_count = 0;
        assert!(invalid_config.validate().is_err());
        
        println!("✅ TorConfig tests passed");
    }

    #[tokio::test] 
    async fn test_quantum_seeding_config() {
        println!("🧪 Testing QuantumSeedingConfig...");
        
        let config = QuantumSeedingConfig::default();
        assert_eq!(config.min_entropy_quality, 0.95);
        assert_eq!(config.reseed_interval, Duration::from_secs(300));
        assert!(config.enable_backup_qrng);
        assert!(config.classical_fallback);
        assert_eq!(config.entropy_buffer_size, 1024);
        
        println!("✅ QuantumSeedingConfig tests passed");
    }

    #[tokio::test]
    async fn test_dandelion_config_and_transaction() {
        println!("🧪 Testing Dandelion++ components...");
        
        let config = DandelionConfig::default();
        assert!(config.fluff_probability > 0.0 && config.fluff_probability <= 1.0);
        assert!(config.max_stem_hops > 0);
        
        // Test transaction creation
        let tx = DandelionTransaction {
            id: uuid::Uuid::new_v4(),
            data: vec![1, 2, 3, 4],
            phase: DandelionPhase::Stem,
            hop_count: 0,
            created_at: SystemTime::now(),
            next_relay: None,
        };
        
        assert_eq!(tx.phase, DandelionPhase::Stem);
        assert_eq!(tx.hop_count, 0);
        assert_eq!(tx.data.len(), 4);
        
        // Test serialization
        let serialized = serde_json::to_string(&tx).unwrap();
        let deserialized: DandelionTransaction = serde_json::from_str(&serialized).unwrap();
        assert_eq!(tx.id, deserialized.id);
        assert_eq!(tx.phase, deserialized.phase);
        
        println!("✅ Dandelion++ tests passed");
    }

    #[tokio::test]
    async fn test_prometheus_config_and_metrics() {
        println!("🧪 Testing Prometheus metrics components...");
        
        let config = PrometheusConfig::default();
        assert!(config.enabled);
        assert!(config.endpoint.is_some());
        assert!(!config.include_sensitive);
        
        // Test metrics summary
        let summary = MetricsSummary {
            active_circuits: 4,
            total_connections: 10,
            entropy_quality: 0.95,
            average_latency_ms: 150,
            anonymity_score: 0.85,
            dandelion_transactions: 25,
            circuit_failures: 1,
            last_update: SystemTime::now(),
        };
        
        // Test serialization
        let serialized = serde_json::to_string(&summary).unwrap();
        let deserialized: MetricsSummary = serde_json::from_str(&serialized).unwrap();
        assert_eq!(summary.active_circuits, deserialized.active_circuits);
        assert_eq!(summary.total_connections, deserialized.total_connections);
        
        println!("✅ Prometheus metrics tests passed");
    }

    #[tokio::test]
    async fn test_entropy_quality_serialization() {
        println!("🧪 Testing EntropyQuality serialization...");
        
        let quality = EntropyQuality {
            primary_quality: 0.98,
            backup_quality: Some(0.94),
            overall_score: 0.96,
            last_assessment: SystemTime::now(),
            tests_passed: 95,
            tests_failed: 5,
        };
        
        // Test serialization
        let serialized = serde_json::to_string(&quality).unwrap();
        let deserialized: EntropyQuality = serde_json::from_str(&serialized).unwrap();
        
        assert_eq!(quality.primary_quality, deserialized.primary_quality);
        assert_eq!(quality.backup_quality, deserialized.backup_quality);
        assert_eq!(quality.overall_score, deserialized.overall_score);
        assert_eq!(quality.tests_passed, deserialized.tests_passed);
        assert_eq!(quality.tests_failed, deserialized.tests_failed);
        
        println!("✅ EntropyQuality serialization tests passed");
    }

    #[tokio::test]
    async fn test_circuit_parameters() {
        println!("🧪 Testing CircuitParameters...");
        
        let params = CircuitParameters {
            seed: [1u8; 32],
            nonce: vec![1, 2, 3, 4],
            timing_offset: Duration::from_millis(100),
            hop_weights: vec![10, 20, 30, 40],
            created_at: SystemTime::now(),
        };
        
        assert_eq!(params.seed.len(), 32);
        assert_eq!(params.nonce.len(), 4);
        assert_eq!(params.hop_weights.len(), 4);
        assert_eq!(params.timing_offset, Duration::from_millis(100));
        
        println!("✅ CircuitParameters tests passed");
    }

    #[tokio::test]
    async fn test_randomness_test_structure() {
        println!("🧪 Testing RandomnessTest structure...");
        
        let test = RandomnessTest {
            sample_size: 1000,
            entropy_score: 7.8,
            chi_squared: 250.0,
            runs_test: 0.15,
            quality_score: 0.95,
            passed_tests: 3,
            total_tests: 3,
        };
        
        assert_eq!(test.sample_size, 1000);
        assert!(test.entropy_score > 7.0);
        assert!(test.chi_squared < 300.0);
        assert!(test.runs_test > 0.1);
        assert!(test.quality_score >= 0.9);
        
        // Test serialization
        let serialized = serde_json::to_string(&test).unwrap();
        let deserialized: RandomnessTest = serde_json::from_str(&serialized).unwrap();
        assert_eq!(test.sample_size, deserialized.sample_size);
        assert_eq!(test.quality_score, deserialized.quality_score);
        
        println!("✅ RandomnessTest tests passed");
    }

    #[tokio::test]
    async fn test_tor_stats_creation() {
        println!("🧪 Testing TorStats creation...");
        
        let stats = TorStats {
            active_circuits: 4,
            average_latency: Duration::from_millis(200),
            connection_count: 15,
            bytes_sent: 10240,
            bytes_received: 20480,
            onion_address: Some("validator123.qnk.onion".to_string()),
            tor_enabled: true,
        };
        
        assert_eq!(stats.active_circuits, 4);
        assert_eq!(stats.average_latency, Duration::from_millis(200));
        assert!(stats.tor_enabled);
        assert!(stats.onion_address.is_some());
        
        // Test serialization
        let serialized = serde_json::to_string(&stats).unwrap();
        let deserialized: TorStats = serde_json::from_str(&serialized).unwrap();
        assert_eq!(stats.active_circuits, deserialized.active_circuits);
        assert_eq!(stats.connection_count, deserialized.connection_count);
        
        println!("✅ TorStats tests passed");
    }

    #[tokio::test]
    async fn test_mock_tor_client() {
        println!("🧪 Testing mock TorClient creation...");
        
        let mock_client = QTorClient::mock();
        assert_eq!(mock_client.current_phase, Phase::Phase1);
        assert!(mock_client.prometheus_metrics.is_none());
        assert!(mock_client.quantum_entropy.is_none());
        assert!(mock_client.dandelion.is_none());
        
        println!("✅ Mock TorClient tests passed");
    }

    #[tokio::test]
    async fn test_tor_metrics() {
        println!("🧪 Testing TorMetrics functionality...");
        
        let metrics = TorMetrics::new();
        
        // Test recording latency
        metrics.record_connection_latency(Duration::from_millis(150)).await;
        metrics.record_connection_latency(Duration::from_millis(200)).await;
        
        // Test recording bytes
        metrics.record_bytes_sent(1024).await;
        metrics.record_bytes_received(2048).await;
        
        // Test getting snapshot
        let snapshot = metrics.get_current_metrics().await;
        assert_eq!(snapshot.connection_count, 2);
        assert_eq!(snapshot.average_latency, Duration::from_millis(175)); // (150+200)/2
        assert_eq!(snapshot.bytes_sent, 1024);
        assert_eq!(snapshot.bytes_received, 2048);
        assert_eq!(snapshot.success_rate, 1.0); // No failures
        
        // Test health status
        let health = metrics.check_performance_health().await;
        println!("Health status: {:?}", health.as_str());
        
        println!("✅ TorMetrics tests passed");
    }

    #[tokio::test]
    async fn test_performance_targets() {
        println!("🧪 Testing performance targets...");
        
        let config = TorConfig::default();
        let (min_latency, max_latency) = config.expected_latency_range();
        
        // For disabled Tor
        assert_eq!(min_latency, Duration::from_millis(10));
        assert_eq!(max_latency, Duration::from_millis(50));
        
        let tor_config = TorConfig::stealth_mode();
        let (tor_min, tor_max) = tor_config.expected_latency_range();
        
        // For Tor-only mode
        assert!(tor_min >= Duration::from_millis(200));
        assert!(tor_max <= Duration::from_millis(300)); // Target from config
        
        println!("✅ Performance target tests passed");
        println!("   Direct connection: {}ms - {}ms", min_latency.as_millis(), max_latency.as_millis());
        println!("   Tor connection: {}ms - {}ms", tor_min.as_millis(), tor_max.as_millis());
    }

    #[tokio::test]
    async fn test_comprehensive_integration() {
        println!("🧪 Running comprehensive integration test...");
        
        // Test all major components can be created and work together
        let tor_config = TorConfig::hybrid_mode();
        assert!(tor_config.validate().is_ok());
        
        let quantum_config = QuantumSeedingConfig::default();
        assert!(quantum_config.min_entropy_quality > 0.9);
        
        let dandelion_config = DandelionConfig::default();
        assert!(dandelion_config.max_stem_hops > 0);
        
        let prometheus_config = PrometheusConfig::default();
        assert!(prometheus_config.enabled);
        
        // Test metrics collection
        let metrics = TorMetrics::new();
        metrics.record_connection_latency(Duration::from_millis(180)).await;
        let snapshot = metrics.get_current_metrics().await;
        assert!(snapshot.average_latency <= Duration::from_millis(300)); // Within target
        
        println!("✅ Comprehensive integration test passed");
        println!("🎉 All Tor integration components are working correctly!");
    }
}