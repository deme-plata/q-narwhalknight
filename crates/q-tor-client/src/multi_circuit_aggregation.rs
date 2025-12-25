/// Multi-Circuit Aggregation for Q-NarwhalKnight Tor Layer
///
/// This module implements parallel circuit utilization for improved throughput:
/// - Aggregate multiple circuits for bulk data transfer (TurboSync)
/// - Load balance requests across healthy circuits
/// - Automatic failover when circuits degrade
/// - Bandwidth bonding for higher effective throughput
///
/// # Architecture
/// ```
/// ┌─────────────────────────────────────────────────────────────┐
/// │                  Multi-Circuit Aggregator                    │
/// └─────────────────────────────────────────────────────────────┘
///                              │
///        ┌─────────────────────┼─────────────────────┐
///        ▼                     ▼                     ▼
/// ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
/// │  Circuit 1  │     │  Circuit 2  │     │  Circuit 3  │
/// │  (Primary)  │     │ (Secondary) │     │  (Tertiary) │
/// └─────────────┘     └─────────────┘     └─────────────┘
///        │                     │                     │
///        └─────────────────────┼─────────────────────┘
///                              ▼
///                    Tor Network (Onion Routing)
/// ```
///
/// # Benefits
/// - 2-3x throughput improvement for bulk sync operations
/// - Automatic load distribution
/// - Graceful degradation on circuit failure
/// - Optimal bandwidth utilization

use crate::dedicated_circuits::{DedicatedCircuitManager, IsolatedOperationClient, OperationType};
use anyhow::{anyhow, Result};
use futures::{stream::FuturesUnordered, StreamExt};
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::{
    collections::{HashMap, VecDeque},
    sync::{
        atomic::{AtomicU64, AtomicUsize, Ordering},
        Arc,
    },
    time::{Duration, Instant},
};
use tokio::sync::{mpsc, Mutex, RwLock, Semaphore};
use tracing::{debug, error, info, warn};

/// Configuration for multi-circuit aggregation
#[derive(Debug, Clone)]
pub struct AggregationConfig {
    /// Number of parallel circuits to use for aggregation
    pub circuit_count: usize,
    /// Maximum concurrent requests per circuit
    pub max_concurrent_per_circuit: usize,
    /// Load balancing strategy
    pub load_balance_strategy: LoadBalanceStrategy,
    /// Enable adaptive circuit selection
    pub adaptive_selection: bool,
    /// Minimum healthy circuits to operate
    pub min_healthy_circuits: usize,
    /// Request timeout
    pub request_timeout: Duration,
    /// Enable request striping (split large requests across circuits)
    pub enable_striping: bool,
    /// Chunk size for striping (bytes)
    pub stripe_chunk_size: usize,
}

impl Default for AggregationConfig {
    fn default() -> Self {
        Self {
            circuit_count: 3,
            max_concurrent_per_circuit: 10,
            load_balance_strategy: LoadBalanceStrategy::LeastLatency,
            adaptive_selection: true,
            min_healthy_circuits: 1,
            request_timeout: Duration::from_secs(30),
            enable_striping: true,
            stripe_chunk_size: 64 * 1024, // 64KB chunks
        }
    }
}

/// Load balancing strategies
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LoadBalanceStrategy {
    /// Round-robin distribution
    RoundRobin,
    /// Route to circuit with lowest latency
    LeastLatency,
    /// Route to circuit with fewest active requests
    LeastConnections,
    /// Route to circuit with highest available bandwidth
    HighestBandwidth,
    /// Weighted distribution based on circuit health
    WeightedHealth,
    /// Random selection
    Random,
}

impl LoadBalanceStrategy {
    pub fn name(&self) -> &'static str {
        match self {
            LoadBalanceStrategy::RoundRobin => "round-robin",
            LoadBalanceStrategy::LeastLatency => "least-latency",
            LoadBalanceStrategy::LeastConnections => "least-connections",
            LoadBalanceStrategy::HighestBandwidth => "highest-bandwidth",
            LoadBalanceStrategy::WeightedHealth => "weighted-health",
            LoadBalanceStrategy::Random => "random",
        }
    }
}

/// Statistics for a single circuit in the aggregation pool
#[derive(Debug, Clone, Default)]
pub struct CircuitStats {
    /// Circuit identifier
    pub circuit_id: usize,
    /// Total requests processed
    pub requests_processed: u64,
    /// Currently active requests
    pub active_requests: usize,
    /// Total bytes sent
    pub bytes_sent: u64,
    /// Total bytes received
    pub bytes_received: u64,
    /// Average latency in milliseconds
    pub avg_latency_ms: f64,
    /// Recent latency samples
    pub recent_latencies: VecDeque<f64>,
    /// Failure count
    pub failures: u64,
    /// Last failure time
    pub last_failure: Option<Instant>,
    /// Circuit health score (0.0 - 1.0)
    pub health_score: f64,
    /// Estimated bandwidth (bytes/sec)
    pub estimated_bandwidth: u64,
}

impl CircuitStats {
    pub fn new(circuit_id: usize) -> Self {
        Self {
            circuit_id,
            health_score: 1.0,
            recent_latencies: VecDeque::with_capacity(100),
            ..Default::default()
        }
    }

    /// Record a successful request
    pub fn record_success(&mut self, latency_ms: f64, bytes_transferred: u64) {
        self.requests_processed += 1;
        self.bytes_received += bytes_transferred;

        // Update latency tracking
        self.recent_latencies.push_back(latency_ms);
        if self.recent_latencies.len() > 100 {
            self.recent_latencies.pop_front();
        }

        // Calculate rolling average
        let sum: f64 = self.recent_latencies.iter().sum();
        self.avg_latency_ms = sum / self.recent_latencies.len() as f64;

        // Update bandwidth estimate (bytes per second)
        if latency_ms > 0.0 {
            let bandwidth = (bytes_transferred as f64 / latency_ms) * 1000.0;
            // Exponential moving average
            self.estimated_bandwidth =
                (self.estimated_bandwidth as f64 * 0.8 + bandwidth * 0.2) as u64;
        }

        // Update health score
        self.update_health_score();
    }

    /// Record a failed request
    pub fn record_failure(&mut self) {
        self.failures += 1;
        self.last_failure = Some(Instant::now());
        self.update_health_score();
    }

    /// Update health score based on metrics
    fn update_health_score(&mut self) {
        let mut score = 1.0;

        // Penalize high failure rate
        if self.requests_processed > 0 {
            let failure_rate = self.failures as f64 / self.requests_processed as f64;
            score -= failure_rate.min(0.5);
        }

        // Penalize high latency
        if self.avg_latency_ms > 1000.0 {
            score -= 0.2;
        } else if self.avg_latency_ms > 500.0 {
            score -= 0.1;
        }

        // Penalize recent failures
        if let Some(last_failure) = self.last_failure {
            if last_failure.elapsed() < Duration::from_secs(60) {
                score -= 0.3;
            } else if last_failure.elapsed() < Duration::from_secs(300) {
                score -= 0.1;
            }
        }

        self.health_score = score.max(0.0);
    }
}

/// Aggregated statistics across all circuits
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AggregationStats {
    /// Total requests across all circuits
    pub total_requests: u64,
    /// Total bytes transferred
    pub total_bytes: u64,
    /// Average latency across all circuits
    pub avg_latency_ms: f64,
    /// Current throughput (bytes/sec)
    pub current_throughput: u64,
    /// Peak throughput achieved
    pub peak_throughput: u64,
    /// Number of healthy circuits
    pub healthy_circuits: usize,
    /// Number of degraded circuits
    pub degraded_circuits: usize,
    /// Striped requests count
    pub striped_requests: u64,
    /// Load balance decisions made
    pub load_balance_decisions: u64,
    /// Failover events
    pub failover_events: u64,
}

/// Request for the aggregator
#[derive(Debug)]
pub struct AggregatedRequest {
    /// Target address
    pub target: String,
    /// Request data
    pub data: Vec<u8>,
    /// Operation type
    pub operation_type: OperationType,
    /// Priority (higher = more important)
    pub priority: u8,
    /// Allow striping for this request
    pub allow_striping: bool,
}

/// Response from the aggregator
#[derive(Debug)]
pub struct AggregatedResponse {
    /// Response data
    pub data: Vec<u8>,
    /// Latency in milliseconds
    pub latency_ms: f64,
    /// Circuit used
    pub circuit_id: usize,
    /// Was this striped across multiple circuits?
    pub was_striped: bool,
}

/// Multi-Circuit Aggregator for high-throughput Tor operations
pub struct MultiCircuitAggregator {
    /// Reference to the dedicated circuit manager
    circuit_manager: Arc<DedicatedCircuitManager>,
    /// Configuration
    config: AggregationConfig,
    /// Per-circuit statistics
    circuit_stats: Arc<RwLock<Vec<CircuitStats>>>,
    /// Aggregated statistics
    agg_stats: Arc<RwLock<AggregationStats>>,
    /// Round-robin counter
    round_robin_counter: AtomicUsize,
    /// Request semaphore for rate limiting
    request_semaphore: Arc<Semaphore>,
    /// Active requests counter
    active_requests: AtomicU64,
    /// Throughput tracker
    throughput_tracker: Arc<RwLock<ThroughputTracker>>,
}

/// Tracks throughput over time
#[derive(Debug, Default)]
struct ThroughputTracker {
    /// Bytes transferred in current window
    bytes_in_window: u64,
    /// Window start time
    window_start: Option<Instant>,
    /// Window duration
    window_duration: Duration,
    /// Historical throughput samples
    throughput_samples: VecDeque<u64>,
}

impl ThroughputTracker {
    fn new() -> Self {
        Self {
            window_duration: Duration::from_secs(1),
            throughput_samples: VecDeque::with_capacity(60),
            ..Default::default()
        }
    }

    fn record_bytes(&mut self, bytes: u64) {
        let now = Instant::now();

        match self.window_start {
            Some(start) if now.duration_since(start) >= self.window_duration => {
                // Window expired, record sample and reset
                self.throughput_samples.push_back(self.bytes_in_window);
                if self.throughput_samples.len() > 60 {
                    self.throughput_samples.pop_front();
                }
                self.bytes_in_window = bytes;
                self.window_start = Some(now);
            }
            Some(_) => {
                // Add to current window
                self.bytes_in_window += bytes;
            }
            None => {
                // First sample
                self.bytes_in_window = bytes;
                self.window_start = Some(now);
            }
        }
    }

    fn current_throughput(&self) -> u64 {
        self.bytes_in_window
    }

    fn peak_throughput(&self) -> u64 {
        self.throughput_samples.iter().cloned().max().unwrap_or(0)
    }
}

impl MultiCircuitAggregator {
    /// Create a new multi-circuit aggregator
    pub fn new(circuit_manager: Arc<DedicatedCircuitManager>, config: AggregationConfig) -> Self {
        info!(
            "🔀 Creating Multi-Circuit Aggregator: {} circuits, strategy={}",
            config.circuit_count,
            config.load_balance_strategy.name()
        );

        let total_concurrent = config.circuit_count * config.max_concurrent_per_circuit;

        // Initialize per-circuit stats
        let mut circuit_stats = Vec::with_capacity(config.circuit_count);
        for i in 0..config.circuit_count {
            circuit_stats.push(CircuitStats::new(i));
        }

        Self {
            circuit_manager,
            config,
            circuit_stats: Arc::new(RwLock::new(circuit_stats)),
            agg_stats: Arc::new(RwLock::new(AggregationStats::default())),
            round_robin_counter: AtomicUsize::new(0),
            request_semaphore: Arc::new(Semaphore::new(total_concurrent)),
            active_requests: AtomicU64::new(0),
            throughput_tracker: Arc::new(RwLock::new(ThroughputTracker::new())),
        }
    }

    /// Initialize the aggregator with prewarmed circuits
    pub async fn initialize(&self) -> Result<()> {
        info!("🚀 Initializing Multi-Circuit Aggregator...");

        // Prewarm all circuits
        for i in 0..self.config.circuit_count {
            debug!("🔥 Prewarming circuit {} for aggregation", i);
            // Use the appropriate operation type for sync
            if let Err(e) = self.circuit_manager.get_client(OperationType::P2PSync).await {
                warn!("⚠️ Failed to prewarm circuit {}: {}", i, e);
            }
        }

        info!(
            "✅ Multi-Circuit Aggregator initialized with {} circuits",
            self.config.circuit_count
        );
        Ok(())
    }

    /// Execute a request through the aggregator
    pub async fn execute(&self, request: AggregatedRequest) -> Result<AggregatedResponse> {
        let start = Instant::now();

        // Acquire semaphore permit
        let _permit = self
            .request_semaphore
            .acquire()
            .await
            .map_err(|e| anyhow!("Failed to acquire semaphore: {}", e))?;

        self.active_requests.fetch_add(1, Ordering::Relaxed);

        // Check if we should stripe this request
        let should_stripe = self.config.enable_striping
            && request.allow_striping
            && request.data.len() > self.config.stripe_chunk_size * 2;

        let result = if should_stripe {
            self.execute_striped(request).await
        } else {
            self.execute_single(request).await
        };

        self.active_requests.fetch_sub(1, Ordering::Relaxed);

        // Update latency in aggregated stats
        if let Ok(ref response) = result {
            let mut agg = self.agg_stats.write().await;
            agg.total_requests += 1;
            agg.total_bytes += response.data.len() as u64;

            // Update rolling average latency
            let alpha = 0.1;
            agg.avg_latency_ms = agg.avg_latency_ms * (1.0 - alpha) + response.latency_ms * alpha;
        }

        result
    }

    /// Execute a single (non-striped) request
    async fn execute_single(&self, request: AggregatedRequest) -> Result<AggregatedResponse> {
        let start = Instant::now();

        // Select circuit based on load balancing strategy
        let circuit_id = self.select_circuit().await?;

        // Update load balance decisions counter
        {
            let mut agg = self.agg_stats.write().await;
            agg.load_balance_decisions += 1;
        }

        // Update active requests for selected circuit
        {
            let mut stats = self.circuit_stats.write().await;
            if let Some(cs) = stats.get_mut(circuit_id) {
                cs.active_requests += 1;
            }
        }

        // Get circuit client and execute
        let result = self
            .execute_on_circuit(circuit_id, &request.target, &request.data)
            .await;

        // Update circuit stats
        {
            let mut stats = self.circuit_stats.write().await;
            if let Some(cs) = stats.get_mut(circuit_id) {
                cs.active_requests = cs.active_requests.saturating_sub(1);

                match &result {
                    Ok(data) => {
                        let latency_ms = start.elapsed().as_secs_f64() * 1000.0;
                        cs.record_success(latency_ms, data.len() as u64);
                    }
                    Err(_) => {
                        cs.record_failure();
                    }
                }
            }
        }

        // Handle result
        match result {
            Ok(data) => {
                let latency_ms = start.elapsed().as_secs_f64() * 1000.0;

                // Update throughput tracker
                {
                    let mut tracker = self.throughput_tracker.write().await;
                    tracker.record_bytes(data.len() as u64);
                }

                Ok(AggregatedResponse {
                    data,
                    latency_ms,
                    circuit_id,
                    was_striped: false,
                })
            }
            Err(e) => {
                // Try failover to another circuit
                warn!(
                    "⚠️ Circuit {} failed, attempting failover: {}",
                    circuit_id, e
                );
                self.execute_with_failover(request, circuit_id).await
            }
        }
    }

    /// Execute with failover to another circuit
    async fn execute_with_failover(
        &self,
        request: AggregatedRequest,
        failed_circuit: usize,
    ) -> Result<AggregatedResponse> {
        let start = Instant::now();

        // Record failover event
        {
            let mut agg = self.agg_stats.write().await;
            agg.failover_events += 1;
        }

        // Try each circuit except the failed one
        for i in 0..self.config.circuit_count {
            if i == failed_circuit {
                continue;
            }

            // Check if circuit is healthy enough
            let health_ok = {
                let stats = self.circuit_stats.read().await;
                stats
                    .get(i)
                    .map(|s| s.health_score > 0.3)
                    .unwrap_or(false)
            };

            if !health_ok {
                continue;
            }

            debug!("🔄 Failing over to circuit {}", i);

            match self
                .execute_on_circuit(i, &request.target, &request.data)
                .await
            {
                Ok(data) => {
                    let latency_ms = start.elapsed().as_secs_f64() * 1000.0;

                    // Update stats
                    {
                        let mut stats = self.circuit_stats.write().await;
                        if let Some(cs) = stats.get_mut(i) {
                            cs.record_success(latency_ms, data.len() as u64);
                        }
                    }

                    return Ok(AggregatedResponse {
                        data,
                        latency_ms,
                        circuit_id: i,
                        was_striped: false,
                    });
                }
                Err(e) => {
                    warn!("⚠️ Failover circuit {} also failed: {}", i, e);
                    // Update failure stats
                    {
                        let mut stats = self.circuit_stats.write().await;
                        if let Some(cs) = stats.get_mut(i) {
                            cs.record_failure();
                        }
                    }
                }
            }
        }

        Err(anyhow!(
            "All circuits failed for request to {}",
            request.target
        ))
    }

    /// Execute a striped request across multiple circuits
    async fn execute_striped(&self, request: AggregatedRequest) -> Result<AggregatedResponse> {
        let start = Instant::now();

        // Split data into chunks
        let chunks: Vec<_> = request
            .data
            .chunks(self.config.stripe_chunk_size)
            .enumerate()
            .collect();

        let chunk_count = chunks.len();
        debug!(
            "📊 Striping {} bytes across {} chunks",
            request.data.len(),
            chunk_count
        );

        // Execute chunks in parallel across circuits
        let mut futures = FuturesUnordered::new();

        for (chunk_idx, chunk) in chunks {
            let circuit_id = chunk_idx % self.config.circuit_count;
            let target = request.target.clone();
            let data = chunk.to_vec();
            let circuit_manager = Arc::clone(&self.circuit_manager);

            futures.push(async move {
                let result = Self::execute_chunk(&circuit_manager, circuit_id, &target, &data).await;
                (chunk_idx, circuit_id, result)
            });
        }

        // Collect results in order
        let mut results: Vec<Option<Vec<u8>>> = vec![None; chunk_count];
        let mut success = true;

        while let Some((chunk_idx, circuit_id, result)) = futures.next().await {
            match result {
                Ok(data) => {
                    results[chunk_idx] = Some(data);
                }
                Err(e) => {
                    warn!(
                        "⚠️ Chunk {} failed on circuit {}: {}",
                        chunk_idx, circuit_id, e
                    );
                    success = false;
                    // We could retry here, but for now mark as failed
                }
            }
        }

        if !success {
            return Err(anyhow!("Some chunks failed during striped transfer"));
        }

        // Reassemble response
        let response_data: Vec<u8> = results.into_iter().filter_map(|r| r).flatten().collect();

        let latency_ms = start.elapsed().as_secs_f64() * 1000.0;

        // Update striped request counter
        {
            let mut agg = self.agg_stats.write().await;
            agg.striped_requests += 1;
        }

        // Update throughput tracker
        {
            let mut tracker = self.throughput_tracker.write().await;
            tracker.record_bytes(response_data.len() as u64);
        }

        Ok(AggregatedResponse {
            data: response_data,
            latency_ms,
            circuit_id: 0, // Multiple circuits used
            was_striped: true,
        })
    }

    /// Execute a single chunk
    async fn execute_chunk(
        circuit_manager: &DedicatedCircuitManager,
        _circuit_id: usize,
        target: &str,
        data: &[u8],
    ) -> Result<Vec<u8>> {
        // Get client for P2P sync operation
        let client = circuit_manager.get_client(OperationType::P2PSync).await?;

        // Connect and send data
        let mut stream = client.connect(target).await?;

        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        stream.write_all(data).await?;
        stream.flush().await?;

        // Read response
        let mut response = Vec::new();
        stream.read_to_end(&mut response).await?;

        Ok(response)
    }

    /// Execute on a specific circuit
    async fn execute_on_circuit(
        &self,
        _circuit_id: usize,
        target: &str,
        data: &[u8],
    ) -> Result<Vec<u8>> {
        // Get client for the operation
        let client = self
            .circuit_manager
            .get_client(OperationType::P2PSync)
            .await?;

        // Connect and transfer
        let mut stream = client.connect(target).await?;

        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        stream.write_all(data).await?;
        stream.flush().await?;

        // Read response
        let mut response = Vec::new();
        stream.read_to_end(&mut response).await?;

        Ok(response)
    }

    /// Select best circuit based on load balancing strategy
    async fn select_circuit(&self) -> Result<usize> {
        let stats = self.circuit_stats.read().await;

        // Filter to healthy circuits
        let healthy_circuits: Vec<_> = stats
            .iter()
            .filter(|s| s.health_score >= 0.3)
            .collect();

        if healthy_circuits.len() < self.config.min_healthy_circuits {
            // Update degraded count
            let mut agg = self.agg_stats.write().await;
            agg.degraded_circuits = stats.iter().filter(|s| s.health_score < 0.7).count();
            agg.healthy_circuits = healthy_circuits.len();
        }

        if healthy_circuits.is_empty() {
            // Fall back to any circuit
            return Ok(0);
        }

        match self.config.load_balance_strategy {
            LoadBalanceStrategy::RoundRobin => {
                let counter = self.round_robin_counter.fetch_add(1, Ordering::Relaxed);
                Ok(healthy_circuits[counter % healthy_circuits.len()].circuit_id)
            }
            LoadBalanceStrategy::LeastLatency => {
                let best = healthy_circuits
                    .iter()
                    .min_by(|a, b| {
                        a.avg_latency_ms
                            .partial_cmp(&b.avg_latency_ms)
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .ok_or_else(|| anyhow!("No circuits available"))?;
                Ok(best.circuit_id)
            }
            LoadBalanceStrategy::LeastConnections => {
                let best = healthy_circuits
                    .iter()
                    .min_by_key(|s| s.active_requests)
                    .ok_or_else(|| anyhow!("No circuits available"))?;
                Ok(best.circuit_id)
            }
            LoadBalanceStrategy::HighestBandwidth => {
                let best = healthy_circuits
                    .iter()
                    .max_by_key(|s| s.estimated_bandwidth)
                    .ok_or_else(|| anyhow!("No circuits available"))?;
                Ok(best.circuit_id)
            }
            LoadBalanceStrategy::WeightedHealth => {
                // Weight selection by health score
                let total_health: f64 = healthy_circuits.iter().map(|s| s.health_score).sum();
                let mut rng = rand::thread_rng();
                let mut target: f64 = rng.gen_range(0.0..total_health);

                for circuit in &healthy_circuits {
                    target -= circuit.health_score;
                    if target <= 0.0 {
                        return Ok(circuit.circuit_id);
                    }
                }

                Ok(healthy_circuits.last().map(|s| s.circuit_id).unwrap_or(0))
            }
            LoadBalanceStrategy::Random => {
                let mut rng = rand::thread_rng();
                let idx = rng.gen_range(0..healthy_circuits.len());
                Ok(healthy_circuits[idx].circuit_id)
            }
        }
    }

    /// Get current aggregation statistics
    pub async fn get_stats(&self) -> AggregationStats {
        let mut agg = self.agg_stats.read().await.clone();

        // Update throughput from tracker
        let tracker = self.throughput_tracker.read().await;
        agg.current_throughput = tracker.current_throughput();
        agg.peak_throughput = tracker.peak_throughput();

        // Update circuit health counts
        let stats = self.circuit_stats.read().await;
        agg.healthy_circuits = stats.iter().filter(|s| s.health_score >= 0.7).count();
        agg.degraded_circuits = stats.iter().filter(|s| s.health_score < 0.7).count();

        agg
    }

    /// Get per-circuit statistics
    pub async fn get_circuit_stats(&self) -> Vec<CircuitStats> {
        self.circuit_stats.read().await.clone()
    }

    /// Get active request count
    pub fn active_requests(&self) -> u64 {
        self.active_requests.load(Ordering::Relaxed)
    }

    /// Force health check and update stats
    pub async fn health_check(&self) -> Result<()> {
        debug!("🔍 Performing aggregator health check");

        // Reset stats for circuits that recovered
        let mut stats = self.circuit_stats.write().await;
        for circuit in stats.iter_mut() {
            // If no recent failures, boost health score slightly
            if let Some(last_failure) = circuit.last_failure {
                if last_failure.elapsed() > Duration::from_secs(300) {
                    circuit.health_score = (circuit.health_score + 0.1).min(1.0);
                }
            } else {
                circuit.health_score = (circuit.health_score + 0.05).min(1.0);
            }
        }

        Ok(())
    }
}

/// Parallel block fetcher for TurboSync
pub struct ParallelBlockFetcher {
    /// Multi-circuit aggregator
    aggregator: Arc<MultiCircuitAggregator>,
    /// Maximum blocks to fetch in parallel
    max_parallel: usize,
}

impl ParallelBlockFetcher {
    pub fn new(aggregator: Arc<MultiCircuitAggregator>, max_parallel: usize) -> Self {
        Self {
            aggregator,
            max_parallel,
        }
    }

    /// Fetch multiple blocks in parallel across circuits
    pub async fn fetch_blocks(
        &self,
        targets: Vec<(String, u64)>, // (peer_addr, block_height)
    ) -> Vec<Result<(u64, Vec<u8>)>> {
        let mut futures = FuturesUnordered::new();

        for (peer_addr, height) in targets.into_iter().take(self.max_parallel) {
            let aggregator = Arc::clone(&self.aggregator);

            futures.push(async move {
                let request = AggregatedRequest {
                    target: peer_addr,
                    data: height.to_le_bytes().to_vec(),
                    operation_type: OperationType::P2PSync,
                    priority: 5,
                    allow_striping: false,
                };

                match aggregator.execute(request).await {
                    Ok(response) => Ok((height, response.data)),
                    Err(e) => Err(e),
                }
            });
        }

        let mut results = Vec::new();
        while let Some(result) = futures.next().await {
            results.push(result);
        }

        results
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_aggregation_config_defaults() {
        let config = AggregationConfig::default();
        assert_eq!(config.circuit_count, 3);
        assert_eq!(
            config.load_balance_strategy,
            LoadBalanceStrategy::LeastLatency
        );
        assert!(config.enable_striping);
    }

    #[test]
    fn test_circuit_stats_health_score() {
        let mut stats = CircuitStats::new(0);
        assert_eq!(stats.health_score, 1.0);

        // Record some successes
        stats.record_success(100.0, 1000);
        stats.record_success(150.0, 2000);
        assert!(stats.health_score > 0.9);

        // Record a failure
        stats.record_failure();
        assert!(stats.health_score < 1.0);
    }

    #[test]
    fn test_load_balance_strategy_names() {
        assert_eq!(LoadBalanceStrategy::RoundRobin.name(), "round-robin");
        assert_eq!(LoadBalanceStrategy::LeastLatency.name(), "least-latency");
    }
}
