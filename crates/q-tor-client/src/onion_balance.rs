/// OnionBalance Integration for Q-NarwhalKnight
///
/// This module provides high-availability hidden services through OnionBalance,
/// allowing multiple backend instances to serve a single .onion address.
///
/// Features:
/// - Master descriptor management
/// - Backend instance registration and health monitoring
/// - Automatic failover and load distribution
/// - Descriptor refresh scheduling
/// - Frontend/backend key separation
///
/// This enables fault-tolerant validator operation with seamless failover.

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    sync::Arc,
    time::{Duration, Instant, SystemTime},
};
use tokio::sync::RwLock;
use tracing::{debug, error, info, warn};

/// OnionBalance operation mode
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OnionBalanceMode {
    /// Master mode - coordinates descriptor publishing
    Master,
    /// Backend mode - provides actual service
    Backend,
    /// Standalone - single instance, no load balancing
    Standalone,
}

impl OnionBalanceMode {
    pub fn name(&self) -> &'static str {
        match self {
            OnionBalanceMode::Master => "Master",
            OnionBalanceMode::Backend => "Backend",
            OnionBalanceMode::Standalone => "Standalone",
        }
    }
}

/// Health status of a backend instance
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BackendHealth {
    /// Backend is healthy and accepting requests
    Healthy,
    /// Backend is degraded but still operational
    Degraded,
    /// Backend is unhealthy and should not receive traffic
    Unhealthy,
    /// Backend health is unknown (not yet checked)
    Unknown,
}

impl BackendHealth {
    pub fn is_usable(&self) -> bool {
        matches!(self, BackendHealth::Healthy | BackendHealth::Degraded)
    }

    pub fn weight(&self) -> f64 {
        match self {
            BackendHealth::Healthy => 1.0,
            BackendHealth::Degraded => 0.5,
            BackendHealth::Unhealthy => 0.0,
            BackendHealth::Unknown => 0.0,
        }
    }
}

/// Backend instance information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackendInstance {
    /// Instance ID
    pub id: String,
    /// Instance onion address (56 chars base32)
    pub onion_address: String,
    /// Instance port
    pub port: u16,
    /// Current health status
    pub health: BackendHealth,
    /// Last health check time
    #[serde(skip)]
    pub last_check: Option<Instant>,
    /// Last successful response time
    #[serde(skip)]
    pub last_success: Option<Instant>,
    /// Consecutive failure count
    pub failure_count: u32,
    /// Average response latency (ms)
    pub avg_latency_ms: u64,
    /// Total requests handled
    pub requests_handled: u64,
    /// Is this instance active in rotation
    pub active: bool,
    /// Weight for load balancing (0.0 - 1.0)
    pub weight: f64,
    /// Geographic region (for geo-aware routing)
    pub region: Option<String>,
}

impl BackendInstance {
    pub fn new(id: String, onion_address: String, port: u16) -> Self {
        Self {
            id,
            onion_address,
            port,
            health: BackendHealth::Unknown,
            last_check: None,
            last_success: None,
            failure_count: 0,
            avg_latency_ms: 0,
            requests_handled: 0,
            active: true,
            weight: 1.0,
            region: None,
        }
    }

    pub fn with_region(mut self, region: String) -> Self {
        self.region = Some(region);
        self
    }

    pub fn with_weight(mut self, weight: f64) -> Self {
        self.weight = weight.clamp(0.0, 1.0);
        self
    }

    /// Get effective weight considering health
    pub fn effective_weight(&self) -> f64 {
        self.weight * self.health.weight()
    }
}

/// Configuration for OnionBalance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OnionBalanceConfig {
    /// Operation mode
    pub mode: OnionBalanceMode,
    /// Master onion address (frontend)
    pub master_address: Option<String>,
    /// Health check interval
    pub health_check_interval: Duration,
    /// Health check timeout
    pub health_check_timeout: Duration,
    /// Descriptor refresh interval
    pub descriptor_refresh_interval: Duration,
    /// Maximum backends to include in descriptor
    pub max_backends_in_descriptor: usize,
    /// Minimum healthy backends required
    pub min_healthy_backends: usize,
    /// Failure threshold before marking unhealthy
    pub failure_threshold: u32,
    /// Recovery threshold before marking healthy again
    pub recovery_threshold: u32,
    /// Enable geo-aware routing
    pub geo_aware_routing: bool,
    /// Preferred regions (in order of preference)
    pub preferred_regions: Vec<String>,
}

impl Default for OnionBalanceConfig {
    fn default() -> Self {
        Self {
            mode: OnionBalanceMode::Standalone,
            master_address: None,
            health_check_interval: Duration::from_secs(60),
            health_check_timeout: Duration::from_secs(10),
            descriptor_refresh_interval: Duration::from_secs(3600), // 1 hour
            max_backends_in_descriptor: 8,
            min_healthy_backends: 1,
            failure_threshold: 3,
            recovery_threshold: 2,
            geo_aware_routing: false,
            preferred_regions: Vec::new(),
        }
    }
}

/// Master descriptor for OnionBalance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MasterDescriptor {
    /// Master onion address
    pub address: String,
    /// Backend instances included
    pub backends: Vec<String>,
    /// Descriptor version
    pub version: u32,
    /// Creation time
    pub created_at: SystemTime,
    /// Valid until
    pub valid_until: SystemTime,
    /// Signature (placeholder for actual crypto)
    pub signature: Vec<u8>,
}

impl MasterDescriptor {
    pub fn new(address: String, backends: Vec<String>) -> Self {
        let now = SystemTime::now();
        Self {
            address,
            backends,
            version: 1,
            created_at: now,
            valid_until: now + Duration::from_secs(3600),
            signature: Vec::new(),
        }
    }

    pub fn is_valid(&self) -> bool {
        SystemTime::now() < self.valid_until
    }
}

/// Statistics for OnionBalance operation
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct OnionBalanceStats {
    /// Total backends registered
    pub total_backends: usize,
    /// Healthy backends
    pub healthy_backends: usize,
    /// Degraded backends
    pub degraded_backends: usize,
    /// Unhealthy backends
    pub unhealthy_backends: usize,
    /// Descriptors published
    pub descriptors_published: u64,
    /// Health checks performed
    pub health_checks_performed: u64,
    /// Failovers triggered
    pub failovers_triggered: u64,
    /// Current descriptor version
    pub current_descriptor_version: u32,
    /// Last descriptor publish time
    pub last_descriptor_publish: Option<SystemTime>,
}

/// OnionBalance manager
pub struct OnionBalanceManager {
    config: OnionBalanceConfig,
    backends: Arc<RwLock<HashMap<String, BackendInstance>>>,
    current_descriptor: Arc<RwLock<Option<MasterDescriptor>>>,
    stats: Arc<RwLock<OnionBalanceStats>>,
    is_running: Arc<std::sync::atomic::AtomicBool>,
}

impl OnionBalanceManager {
    /// Create a new OnionBalance manager
    pub fn new(config: OnionBalanceConfig) -> Self {
        info!("🧅 Creating OnionBalance Manager");
        info!("   Mode: {}", config.mode.name());
        if let Some(addr) = &config.master_address {
            info!("   Master address: {}", addr);
        }

        Self {
            config,
            backends: Arc::new(RwLock::new(HashMap::new())),
            current_descriptor: Arc::new(RwLock::new(None)),
            stats: Arc::new(RwLock::new(OnionBalanceStats::default())),
            is_running: Arc::new(std::sync::atomic::AtomicBool::new(false)),
        }
    }

    /// Register a backend instance
    pub async fn register_backend(&self, backend: BackendInstance) -> Result<()> {
        let id = backend.id.clone();
        let addr = backend.onion_address.clone();

        let mut backends = self.backends.write().await;
        backends.insert(id.clone(), backend);

        // Update stats
        let mut stats = self.stats.write().await;
        stats.total_backends = backends.len();

        info!("✅ Registered backend {}: {}", id, addr);
        Ok(())
    }

    /// Unregister a backend instance
    pub async fn unregister_backend(&self, id: &str) -> Result<()> {
        let mut backends = self.backends.write().await;

        if backends.remove(id).is_some() {
            let mut stats = self.stats.write().await;
            stats.total_backends = backends.len();
            info!("✅ Unregistered backend {}", id);
            Ok(())
        } else {
            Err(anyhow!("Backend not found: {}", id))
        }
    }

    /// Update backend health status
    pub async fn update_backend_health(&self, id: &str, health: BackendHealth, latency_ms: Option<u64>) {
        let mut backends = self.backends.write().await;

        if let Some(backend) = backends.get_mut(id) {
            let old_health = backend.health;
            backend.health = health;
            backend.last_check = Some(Instant::now());

            if health.is_usable() {
                backend.last_success = Some(Instant::now());
                backend.failure_count = 0;
                if let Some(latency) = latency_ms {
                    // Exponential moving average
                    if backend.avg_latency_ms == 0 {
                        backend.avg_latency_ms = latency;
                    } else {
                        backend.avg_latency_ms = (backend.avg_latency_ms * 7 + latency) / 8;
                    }
                }
            } else {
                backend.failure_count += 1;
            }

            // Log health transitions
            if old_health != health {
                match health {
                    BackendHealth::Healthy => info!("✅ Backend {} is now healthy", id),
                    BackendHealth::Degraded => warn!("⚠️ Backend {} is degraded", id),
                    BackendHealth::Unhealthy => error!("❌ Backend {} is unhealthy", id),
                    BackendHealth::Unknown => debug!("Backend {} health unknown", id),
                }
            }
        }
    }

    /// Get a healthy backend for routing
    pub async fn get_backend(&self, preferred_region: Option<&str>) -> Option<BackendInstance> {
        let backends = self.backends.read().await;

        let healthy: Vec<_> = backends.values()
            .filter(|b| b.active && b.health.is_usable())
            .collect();

        if healthy.is_empty() {
            return None;
        }

        // Prefer region if specified and geo-aware routing is enabled
        if self.config.geo_aware_routing {
            if let Some(region) = preferred_region {
                if let Some(backend) = healthy.iter()
                    .find(|b| b.region.as_deref() == Some(region))
                {
                    return Some((*backend).clone());
                }
            }

            // Try preferred regions
            for pref_region in &self.config.preferred_regions {
                if let Some(backend) = healthy.iter()
                    .find(|b| b.region.as_deref() == Some(pref_region.as_str()))
                {
                    return Some((*backend).clone());
                }
            }
        }

        // Weighted random selection based on effective weight
        let total_weight: f64 = healthy.iter().map(|b| b.effective_weight()).sum();
        if total_weight == 0.0 {
            return healthy.first().map(|b| (*b).clone());
        }

        let mut target = rand::random::<f64>() * total_weight;
        for backend in &healthy {
            target -= backend.effective_weight();
            if target <= 0.0 {
                return Some((*backend).clone());
            }
        }

        healthy.last().map(|b| (*b).clone())
    }

    /// Generate master descriptor
    pub async fn generate_descriptor(&self) -> Result<MasterDescriptor> {
        let backends = self.backends.read().await;

        let healthy_backends: Vec<String> = backends.values()
            .filter(|b| b.active && b.health.is_usable())
            .take(self.config.max_backends_in_descriptor)
            .map(|b| b.onion_address.clone())
            .collect();

        if healthy_backends.len() < self.config.min_healthy_backends {
            return Err(anyhow!(
                "Insufficient healthy backends: {} < {}",
                healthy_backends.len(),
                self.config.min_healthy_backends
            ));
        }

        let master_addr = self.config.master_address.clone()
            .ok_or_else(|| anyhow!("Master address not configured"))?;

        let descriptor = MasterDescriptor::new(master_addr, healthy_backends);

        info!(
            "Generated descriptor with {} backends (version {})",
            descriptor.backends.len(),
            descriptor.version
        );

        Ok(descriptor)
    }

    /// Publish master descriptor
    pub async fn publish_descriptor(&self) -> Result<()> {
        let descriptor = self.generate_descriptor().await?;

        // Store current descriptor
        {
            let mut current = self.current_descriptor.write().await;
            *current = Some(descriptor.clone());
        }

        // Update stats
        {
            let mut stats = self.stats.write().await;
            stats.descriptors_published += 1;
            stats.current_descriptor_version = descriptor.version;
            stats.last_descriptor_publish = Some(SystemTime::now());
        }

        // In production, this would publish to HSDir
        info!(
            "📢 Published descriptor for {} with {} backends",
            descriptor.address,
            descriptor.backends.len()
        );

        Ok(())
    }

    /// Perform health check on all backends
    pub async fn health_check_all(&self) {
        let backends = self.backends.read().await;
        let backend_ids: Vec<String> = backends.keys().cloned().collect();
        drop(backends);

        for id in backend_ids {
            self.health_check_backend(&id).await;
        }

        // Update stats
        let backends = self.backends.read().await;
        let mut stats = self.stats.write().await;
        stats.health_checks_performed += 1;
        stats.healthy_backends = backends.values().filter(|b| b.health == BackendHealth::Healthy).count();
        stats.degraded_backends = backends.values().filter(|b| b.health == BackendHealth::Degraded).count();
        stats.unhealthy_backends = backends.values().filter(|b| b.health == BackendHealth::Unhealthy).count();
    }

    /// Perform health check on a specific backend
    async fn health_check_backend(&self, id: &str) {
        let backend = {
            let backends = self.backends.read().await;
            backends.get(id).cloned()
        };

        let Some(backend) = backend else {
            return;
        };

        let start = Instant::now();

        // Simulate health check (in production, connect to backend)
        let health_result = self.perform_health_check(&backend).await;

        let latency = start.elapsed().as_millis() as u64;

        match health_result {
            Ok(()) => {
                self.update_backend_health(id, BackendHealth::Healthy, Some(latency)).await;
            }
            Err(e) => {
                warn!("Health check failed for {}: {}", id, e);

                let backends = self.backends.read().await;
                if let Some(b) = backends.get(id) {
                    let new_health = if b.failure_count + 1 >= self.config.failure_threshold {
                        BackendHealth::Unhealthy
                    } else {
                        BackendHealth::Degraded
                    };
                    drop(backends);
                    self.update_backend_health(id, new_health, None).await;
                }
            }
        }
    }

    /// Perform actual health check (placeholder)
    async fn perform_health_check(&self, backend: &BackendInstance) -> Result<()> {
        // In production, this would:
        // 1. Connect to backend onion address via Tor
        // 2. Send health check request
        // 3. Verify response

        let timeout = self.config.health_check_timeout;

        // Simulate check with timeout
        tokio::time::timeout(timeout, async {
            // Placeholder - would actually connect and check
            tokio::time::sleep(Duration::from_millis(50)).await;

            // Simulate occasional failures for testing
            if rand::random::<f64>() < 0.05 {
                return Err(anyhow!("Simulated health check failure"));
            }

            debug!("Health check OK for {} ({})", backend.id, backend.onion_address);
            Ok(())
        })
        .await
        .map_err(|_| anyhow!("Health check timed out"))?
    }

    /// Start background health monitoring
    pub async fn start_monitoring(&self) {
        use std::sync::atomic::Ordering;

        if self.is_running.swap(true, Ordering::SeqCst) {
            warn!("OnionBalance monitoring already running");
            return;
        }

        info!("🔍 Starting OnionBalance monitoring");

        let config = self.config.clone();
        let manager = OnionBalanceManager {
            config: self.config.clone(),
            backends: Arc::clone(&self.backends),
            current_descriptor: Arc::clone(&self.current_descriptor),
            stats: Arc::clone(&self.stats),
            is_running: Arc::clone(&self.is_running),
        };

        tokio::spawn(async move {
            let mut last_descriptor_publish = Instant::now();

            loop {
                if !manager.is_running.load(Ordering::SeqCst) {
                    break;
                }

                // Perform health checks
                manager.health_check_all().await;

                // Publish descriptor if due
                if config.mode == OnionBalanceMode::Master {
                    if last_descriptor_publish.elapsed() >= config.descriptor_refresh_interval {
                        if let Err(e) = manager.publish_descriptor().await {
                            error!("Failed to publish descriptor: {}", e);
                        }
                        last_descriptor_publish = Instant::now();
                    }
                }

                tokio::time::sleep(config.health_check_interval).await;
            }

            info!("🔍 OnionBalance monitoring stopped");
        });
    }

    /// Stop background monitoring
    pub fn stop_monitoring(&self) {
        use std::sync::atomic::Ordering;
        self.is_running.store(false, Ordering::SeqCst);
    }

    /// Get current statistics
    pub async fn get_stats(&self) -> OnionBalanceStats {
        self.stats.read().await.clone()
    }

    /// Get all backends
    pub async fn get_backends(&self) -> Vec<BackendInstance> {
        self.backends.read().await.values().cloned().collect()
    }

    /// Get current descriptor
    pub async fn get_descriptor(&self) -> Option<MasterDescriptor> {
        self.current_descriptor.read().await.clone()
    }

    /// Trigger manual failover
    pub async fn trigger_failover(&self, from_id: &str, to_id: &str) -> Result<()> {
        let mut backends = self.backends.write().await;

        // Deactivate source
        if let Some(from) = backends.get_mut(from_id) {
            from.active = false;
            info!("Deactivated backend {}", from_id);
        }

        // Ensure target is active
        if let Some(to) = backends.get_mut(to_id) {
            to.active = true;
            info!("Activated backend {}", to_id);
        } else {
            return Err(anyhow!("Target backend not found: {}", to_id));
        }

        // Update stats
        let mut stats = self.stats.write().await;
        stats.failovers_triggered += 1;

        info!("🔄 Failover triggered: {} -> {}", from_id, to_id);
        Ok(())
    }
}

/// High-availability hidden service wrapper
pub struct HAHiddenService {
    /// OnionBalance manager
    manager: Arc<OnionBalanceManager>,
    /// Master onion address
    master_address: String,
    /// Local backend instance (if running as backend)
    local_backend: Option<BackendInstance>,
}

impl HAHiddenService {
    /// Create a new high-availability hidden service
    pub async fn new(
        master_address: String,
        mode: OnionBalanceMode,
    ) -> Result<Self> {
        let config = OnionBalanceConfig {
            mode,
            master_address: Some(master_address.clone()),
            ..Default::default()
        };

        let manager = Arc::new(OnionBalanceManager::new(config));

        Ok(Self {
            manager,
            master_address,
            local_backend: None,
        })
    }

    /// Register this instance as a backend
    pub async fn register_as_backend(
        &mut self,
        instance_id: String,
        onion_address: String,
        port: u16,
    ) -> Result<()> {
        let backend = BackendInstance::new(instance_id.clone(), onion_address, port);
        self.manager.register_backend(backend.clone()).await?;
        self.local_backend = Some(backend);
        Ok(())
    }

    /// Add a remote backend
    pub async fn add_backend(&self, backend: BackendInstance) -> Result<()> {
        self.manager.register_backend(backend).await
    }

    /// Start the HA service
    pub async fn start(&self) -> Result<()> {
        self.manager.start_monitoring().await;

        // Initial descriptor publish for master mode
        if self.manager.config.mode == OnionBalanceMode::Master {
            self.manager.publish_descriptor().await?;
        }

        info!("✅ HA Hidden Service started: {}", self.master_address);
        Ok(())
    }

    /// Stop the HA service
    pub async fn stop(&self) {
        self.manager.stop_monitoring();
        info!("🛑 HA Hidden Service stopped");
    }

    /// Get the master onion address
    pub fn master_address(&self) -> &str {
        &self.master_address
    }

    /// Get the manager
    pub fn manager(&self) -> Arc<OnionBalanceManager> {
        Arc::clone(&self.manager)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_backend_registration() {
        let config = OnionBalanceConfig::default();
        let manager = OnionBalanceManager::new(config);

        let backend = BackendInstance::new(
            "test-1".to_string(),
            "testaddress123456789012345678901234567890123456".to_string(),
            8080,
        );

        manager.register_backend(backend).await.unwrap();

        let backends = manager.get_backends().await;
        assert_eq!(backends.len(), 1);
        assert_eq!(backends[0].id, "test-1");
    }

    #[tokio::test]
    async fn test_health_update() {
        let config = OnionBalanceConfig::default();
        let manager = OnionBalanceManager::new(config);

        let backend = BackendInstance::new(
            "test-1".to_string(),
            "testaddress123456789012345678901234567890123456".to_string(),
            8080,
        );

        manager.register_backend(backend).await.unwrap();
        manager.update_backend_health("test-1", BackendHealth::Healthy, Some(50)).await;

        let backends = manager.get_backends().await;
        assert_eq!(backends[0].health, BackendHealth::Healthy);
        assert_eq!(backends[0].avg_latency_ms, 50);
    }

    #[tokio::test]
    async fn test_backend_selection() {
        let config = OnionBalanceConfig::default();
        let manager = OnionBalanceManager::new(config);

        let backend1 = BackendInstance::new(
            "test-1".to_string(),
            "address1".to_string(),
            8080,
        );
        let mut backend2 = BackendInstance::new(
            "test-2".to_string(),
            "address2".to_string(),
            8080,
        );
        backend2.health = BackendHealth::Healthy;

        manager.register_backend(backend1).await.unwrap();
        manager.register_backend(backend2).await.unwrap();
        manager.update_backend_health("test-1", BackendHealth::Healthy, None).await;

        // Should get one of the healthy backends
        let selected = manager.get_backend(None).await;
        assert!(selected.is_some());
    }

    #[test]
    fn test_effective_weight() {
        let mut backend = BackendInstance::new(
            "test".to_string(),
            "addr".to_string(),
            8080,
        );

        backend.weight = 1.0;
        backend.health = BackendHealth::Healthy;
        assert_eq!(backend.effective_weight(), 1.0);

        backend.health = BackendHealth::Degraded;
        assert_eq!(backend.effective_weight(), 0.5);

        backend.health = BackendHealth::Unhealthy;
        assert_eq!(backend.effective_weight(), 0.0);
    }
}
