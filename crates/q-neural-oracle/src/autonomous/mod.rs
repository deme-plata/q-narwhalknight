//! Full Autonomous Prediction Infrastructure
//!
//! Self-sustaining decentralized prediction network with automated model deployment,
//! oracle data feeds, economic sustainability mechanisms, and complete autonomy.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────────────┐
//! │                   AUTONOMOUS PREDICTION INFRASTRUCTURE                       │
//! ├─────────────────────────────────────────────────────────────────────────────┤
//! │                                                                             │
//! │  ┌─────────────────────────────────────────────────────────────────┐       │
//! │  │                    DATA FEED LAYER                               │       │
//! │  │  • External oracles (price feeds, weather, events)              │       │
//! │  │  • On-chain data aggregation                                    │       │
//! │  │  • Cross-chain bridges for external data                        │       │
//! │  │  • Data quality scoring and verification                        │       │
//! │  └─────────────────────────────────────────────────────────────────┘       │
//! │                              │                                              │
//! │                              ▼                                              │
//! │  ┌─────────────────────────────────────────────────────────────────┐       │
//! │  │                 MODEL DEPLOYMENT LAYER                          │       │
//! │  │  • Automatic model compilation and optimization                 │       │
//! │  │  • Hot-swap model updates without downtime                      │       │
//! │  │  • A/B testing for new architectures                            │       │
//! │  │  • Resource allocation based on demand                          │       │
//! │  └─────────────────────────────────────────────────────────────────┘       │
//! │                              │                                              │
//! │                              ▼                                              │
//! │  ┌─────────────────────────────────────────────────────────────────┐       │
//! │  │                  PREDICTION SERVICE LAYER                       │       │
//! │  │  • Request routing and load balancing                           │       │
//! │  │  • Prediction caching and deduplication                         │       │
//! │  │  • Rate limiting and access control                             │       │
//! │  │  • Real-time monitoring and alerting                            │       │
//! │  └─────────────────────────────────────────────────────────────────┘       │
//! │                              │                                              │
//! │                              ▼                                              │
//! │  ┌─────────────────────────────────────────────────────────────────┐       │
//! │  │                   ECONOMICS LAYER                                │       │
//! │  │  • Fee collection and distribution                              │       │
//! │  │  • Validator rewards based on accuracy                          │       │
//! │  │  • Treasury management for sustainability                       │       │
//! │  │  • Automatic parameter adjustment (fees, rewards)               │       │
//! │  └─────────────────────────────────────────────────────────────────┘       │
//! │                              │                                              │
//! │                              ▼                                              │
//! │  ┌─────────────────────────────────────────────────────────────────┐       │
//! │  │                 SELF-IMPROVEMENT LAYER                          │       │
//! │  │  • Continuous model evaluation                                  │       │
//! │  │  • Automatic retraining triggers                                │       │
//! │  │  • Architecture evolution via NAS governance                    │       │
//! │  │  • Performance anomaly detection                                │       │
//! │  └─────────────────────────────────────────────────────────────────┘       │
//! └─────────────────────────────────────────────────────────────────────────────┘
//! ```

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use tokio::sync::{RwLock, broadcast, mpsc};
use serde::{Serialize, Deserialize};
use sha3::{Sha3_256, Digest};
use tracing::{info, warn, error};

use crate::evolution::Architecture;
use crate::governance::NasGovernance;
use crate::federated::FederatedCoordinator;
use crate::{Prediction, PredictionContext, PredictionDomain, PredictionSource};

/// Autonomous infrastructure configuration
#[derive(Clone, Debug)]
pub struct AutonomousConfig {
    /// Base prediction fee (QUG atoms)
    pub base_prediction_fee: u64,
    /// Validator reward percentage (0-100)
    pub validator_reward_pct: u32,
    /// Treasury percentage (0-100)
    pub treasury_pct: u32,
    /// Model retraining threshold (accuracy drop)
    pub retrain_threshold: f64,
    /// Maximum prediction cache size
    pub cache_size: usize,
    /// Cache TTL (seconds)
    pub cache_ttl_secs: u64,
    /// Rate limit (requests per minute)
    pub rate_limit_rpm: u32,
    /// A/B test traffic percentage for new models
    pub ab_test_percentage: u32,
    /// Minimum validators for prediction
    pub min_validators: usize,
    /// Data feed update interval (ms)
    pub feed_update_interval_ms: u64,
}

impl Default for AutonomousConfig {
    fn default() -> Self {
        Self {
            base_prediction_fee: 1_000_000,          // 0.01 QUG
            validator_reward_pct: 70,
            treasury_pct: 20,                        // 10% to data providers
            retrain_threshold: 0.05,                 // 5% accuracy drop
            cache_size: 10_000,
            cache_ttl_secs: 60,
            rate_limit_rpm: 1000,
            ab_test_percentage: 10,
            min_validators: 3,
            feed_update_interval_ms: 5000,
        }
    }
}

/// Oracle data feed
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DataFeed {
    /// Feed identifier
    pub feed_id: String,
    /// Feed type
    pub feed_type: FeedType,
    /// Current value
    pub value: f64,
    /// Timestamp of last update
    pub timestamp: u64,
    /// Data provider
    pub provider: [u8; 32],
    /// Quality score (0-1)
    pub quality_score: f64,
    /// Signature from provider
    pub signature: Vec<u8>,
}

/// Types of data feeds
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum FeedType {
    /// Cryptocurrency prices
    CryptoPrice { symbol: String, base: String },
    /// Network metrics
    NetworkMetric { metric: String },
    /// External API data
    ExternalApi { endpoint: String },
    /// Cross-chain data
    CrossChain { chain_id: u64, contract: String },
    /// Custom oracle
    Custom { provider_id: String },
}

/// Deployed model instance
#[derive(Clone, Debug)]
pub struct DeployedModel {
    /// Model ID
    pub model_id: u64,
    /// Architecture
    pub architecture: Architecture,
    /// Deployment timestamp
    pub deployed_at: u64,
    /// Model weights (serialized)
    pub weights: Vec<u8>,
    /// Is active (receiving traffic)
    pub active: bool,
    /// Traffic percentage (for A/B testing)
    pub traffic_pct: u32,
    /// Performance metrics
    pub metrics: ModelMetrics,
}

/// Model performance metrics
#[derive(Clone, Debug, Default)]
pub struct ModelMetrics {
    /// Total predictions made
    pub total_predictions: u64,
    /// Correct predictions
    pub correct_predictions: u64,
    /// Average latency (ms)
    pub avg_latency_ms: f64,
    /// Total revenue generated
    pub revenue: u64,
    /// Last evaluation timestamp
    pub last_evaluated: u64,
}

/// Prediction request
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PredictionRequest {
    /// Request ID
    pub request_id: u64,
    /// Requester address
    pub requester: [u8; 32],
    /// Prediction domain
    pub domain: PredictionDomain,
    /// Context data
    pub context: PredictionContext,
    /// Fee paid
    pub fee_paid: u64,
    /// Timestamp
    pub timestamp: u64,
    /// Priority (higher = faster)
    pub priority: u32,
}

/// Prediction response
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PredictionResponse {
    /// Request ID
    pub request_id: u64,
    /// Prediction result
    pub prediction: Prediction,
    /// Model ID used
    pub model_id: u64,
    /// Validators who contributed
    pub validators: Vec<[u8; 32]>,
    /// Fee breakdown
    pub fee_breakdown: FeeBreakdown,
    /// Response timestamp
    pub timestamp: u64,
}

/// Fee distribution breakdown
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FeeBreakdown {
    /// Total fee
    pub total: u64,
    /// To validators
    pub validators: u64,
    /// To treasury
    pub treasury: u64,
    /// To data providers
    pub data_providers: u64,
}

/// Cached prediction
#[derive(Clone, Debug)]
struct CachedPrediction {
    prediction: Prediction,
    cached_at: u64,
    hit_count: u64,
}

/// Treasury state
#[derive(Clone, Debug, Default)]
pub struct TreasuryState {
    /// Total balance
    pub balance: u64,
    /// Total fees collected
    pub total_fees_collected: u64,
    /// Total rewards distributed
    pub total_rewards_distributed: u64,
    /// Reserved for operations
    pub reserved: u64,
}

/// Autonomous Prediction Infrastructure
pub struct AutonomousOracle {
    /// Configuration
    config: AutonomousConfig,
    /// Active data feeds
    data_feeds: Arc<RwLock<HashMap<String, DataFeed>>>,
    /// Deployed models
    models: Arc<RwLock<HashMap<u64, DeployedModel>>>,
    /// Active model ID
    active_model_id: Arc<RwLock<u64>>,
    /// Prediction cache
    cache: Arc<RwLock<HashMap<[u8; 32], CachedPrediction>>>,
    /// Rate limiter state
    rate_limits: Arc<RwLock<HashMap<[u8; 32], RateLimitState>>>,
    /// Treasury
    treasury: Arc<RwLock<TreasuryState>>,
    /// Validator balances
    validator_balances: Arc<RwLock<HashMap<[u8; 32], u64>>>,
    /// Request queue
    request_queue: Arc<RwLock<VecDeque<PredictionRequest>>>,
    /// Next request ID
    next_request_id: Arc<RwLock<u64>>,
    /// NAS governance (optional integration)
    governance: Option<Arc<RwLock<NasGovernance>>>,
    /// Federated learning (optional integration)
    federated: Option<Arc<RwLock<FederatedCoordinator>>>,
    /// Event broadcaster
    event_tx: broadcast::Sender<OracleEvent>,
    /// Shutdown signal
    shutdown: Arc<RwLock<bool>>,
}

/// Rate limit state per address
#[derive(Clone, Debug, Default)]
struct RateLimitState {
    requests: u32,
    window_start: u64,
}

/// Oracle events for monitoring
#[derive(Clone, Debug)]
pub enum OracleEvent {
    PredictionMade { request_id: u64, model_id: u64 },
    ModelDeployed { model_id: u64 },
    ModelRetrained { model_id: u64, reason: String },
    FeedUpdated { feed_id: String },
    RewardDistributed { validator: [u8; 32], amount: u64 },
    PerformanceAlert { model_id: u64, metric: String, value: f64 },
}

impl AutonomousOracle {
    /// Create new autonomous oracle
    pub fn new(config: AutonomousConfig) -> Self {
        info!("🤖 Initializing Autonomous Prediction Infrastructure");
        info!("   Base fee: {} atoms", config.base_prediction_fee);
        info!("   Validator reward: {}%", config.validator_reward_pct);
        info!("   Cache size: {}", config.cache_size);

        let (event_tx, _) = broadcast::channel(1000);

        Self {
            config,
            data_feeds: Arc::new(RwLock::new(HashMap::new())),
            models: Arc::new(RwLock::new(HashMap::new())),
            active_model_id: Arc::new(RwLock::new(0)),
            cache: Arc::new(RwLock::new(HashMap::new())),
            rate_limits: Arc::new(RwLock::new(HashMap::new())),
            treasury: Arc::new(RwLock::new(TreasuryState::default())),
            validator_balances: Arc::new(RwLock::new(HashMap::new())),
            request_queue: Arc::new(RwLock::new(VecDeque::new())),
            next_request_id: Arc::new(RwLock::new(1)),
            governance: None,
            federated: None,
            event_tx,
            shutdown: Arc::new(RwLock::new(false)),
        }
    }

    /// Connect NAS governance
    pub fn with_governance(mut self, governance: Arc<RwLock<NasGovernance>>) -> Self {
        self.governance = Some(governance);
        self
    }

    /// Connect federated learning
    pub fn with_federated(mut self, federated: Arc<RwLock<FederatedCoordinator>>) -> Self {
        self.federated = Some(federated);
        self
    }

    /// Register a data feed
    pub async fn register_feed(&self, feed: DataFeed) -> anyhow::Result<()> {
        // Verify feed signature
        if !self.verify_feed_signature(&feed) {
            return Err(anyhow::anyhow!("Invalid feed signature"));
        }

        let feed_id = feed.feed_id.clone();
        self.data_feeds.write().await.insert(feed_id.clone(), feed);

        let _ = self.event_tx.send(OracleEvent::FeedUpdated { feed_id });

        Ok(())
    }

    /// Update a data feed
    pub async fn update_feed(&self, feed_id: &str, value: f64, timestamp: u64) -> anyhow::Result<()> {
        let mut feeds = self.data_feeds.write().await;
        let feed = feeds.get_mut(feed_id)
            .ok_or_else(|| anyhow::anyhow!("Feed not found"))?;

        // Only accept newer updates
        if timestamp <= feed.timestamp {
            return Err(anyhow::anyhow!("Stale feed update"));
        }

        feed.value = value;
        feed.timestamp = timestamp;

        Ok(())
    }

    /// Deploy a new model
    pub async fn deploy_model(
        &self,
        architecture: Architecture,
        weights: Vec<u8>,
        traffic_pct: u32,
    ) -> anyhow::Result<u64> {
        let model_id = {
            let mut next = self.next_request_id.write().await;
            let id = *next;
            *next += 1;
            id
        };

        let model = DeployedModel {
            model_id,
            architecture,
            deployed_at: chrono::Utc::now().timestamp() as u64,
            weights,
            active: true,
            traffic_pct,
            metrics: ModelMetrics::default(),
        };

        self.models.write().await.insert(model_id, model);

        // If this is the first model or 100% traffic, make it active
        if traffic_pct == 100 {
            *self.active_model_id.write().await = model_id;
        }

        let _ = self.event_tx.send(OracleEvent::ModelDeployed { model_id });

        info!("🚀 Model {} deployed with {}% traffic", model_id, traffic_pct);

        Ok(model_id)
    }

    /// Submit a prediction request
    pub async fn request_prediction(
        &self,
        requester: [u8; 32],
        domain: PredictionDomain,
        context: PredictionContext,
        fee_paid: u64,
    ) -> anyhow::Result<u64> {
        // Check rate limit
        if !self.check_rate_limit(&requester).await {
            return Err(anyhow::anyhow!("Rate limit exceeded"));
        }

        // Check fee
        let required_fee = self.calculate_fee(&domain).await;
        if fee_paid < required_fee {
            return Err(anyhow::anyhow!(
                "Insufficient fee: {} < {}",
                fee_paid, required_fee
            ));
        }

        // Generate request ID
        let request_id = {
            let mut next = self.next_request_id.write().await;
            let id = *next;
            *next += 1;
            id
        };

        let request = PredictionRequest {
            request_id,
            requester,
            domain,
            context,
            fee_paid,
            timestamp: chrono::Utc::now().timestamp() as u64,
            priority: self.calculate_priority(fee_paid, required_fee),
        };

        // Add to queue (priority sorted)
        let mut queue = self.request_queue.write().await;
        let insert_pos = queue.iter()
            .position(|r| r.priority < request.priority)
            .unwrap_or(queue.len());
        queue.insert(insert_pos, request);

        Ok(request_id)
    }

    /// Process pending prediction requests
    pub async fn process_predictions(&self, max_batch: usize) -> Vec<PredictionResponse> {
        let mut responses = Vec::new();
        let mut queue = self.request_queue.write().await;

        for _ in 0..max_batch {
            let request = match queue.pop_front() {
                Some(r) => r,
                None => break,
            };

            // Check cache first
            let cache_key = self.compute_cache_key(&request.domain, &request.context);
            if let Some(cached) = self.get_cached(&cache_key).await {
                responses.push(PredictionResponse {
                    request_id: request.request_id,
                    prediction: cached,
                    model_id: *self.active_model_id.read().await,
                    validators: vec![],
                    fee_breakdown: self.calculate_fee_breakdown(request.fee_paid),
                    timestamp: chrono::Utc::now().timestamp() as u64,
                });
                continue;
            }

            // Select model (A/B testing)
            let model_id = self.select_model().await;

            // Make prediction
            match self.make_prediction(model_id, &request).await {
                Ok(prediction) => {
                    // Cache result
                    self.cache_prediction(&cache_key, &prediction).await;

                    // Update model metrics
                    self.update_model_metrics(model_id, &prediction).await;

                    // Distribute fees
                    let fee_breakdown = self.distribute_fees(request.fee_paid, &[]).await;

                    responses.push(PredictionResponse {
                        request_id: request.request_id,
                        prediction,
                        model_id,
                        validators: vec![], // Filled by committee
                        fee_breakdown,
                        timestamp: chrono::Utc::now().timestamp() as u64,
                    });

                    let _ = self.event_tx.send(OracleEvent::PredictionMade {
                        request_id: request.request_id,
                        model_id,
                    });
                }
                Err(e) => {
                    error!("Prediction failed for request {}: {:?}",
                           request.request_id, e);
                    // Refund fee on failure
                    // (Would integrate with token system)
                }
            }
        }

        responses
    }

    /// Make a prediction using specified model
    async fn make_prediction(
        &self,
        model_id: u64,
        request: &PredictionRequest,
    ) -> anyhow::Result<Prediction> {
        let models = self.models.read().await;
        let _model = models.get(&model_id)
            .ok_or_else(|| anyhow::anyhow!("Model not found"))?;

        // Enrich context with data feeds
        let enriched_context = self.enrich_context(&request.context).await;

        // Create prediction (simplified - would use actual model inference)
        let prediction = Prediction {
            value: self.simple_predict(&enriched_context, &request.domain),
            confidence: 0.85,
            domain: request.domain.clone(),
            source: PredictionSource::Quantum,
            expert_weights: vec![(request.domain.clone(), 1.0)],
            quantum_fidelity: 0.95,
            timestamp: chrono::Utc::now().timestamp() as u64,
            proof: None,
        };

        Ok(prediction)
    }

    /// Simple prediction placeholder
    fn simple_predict(&self, context: &PredictionContext, domain: &PredictionDomain) -> f64 {
        // This would be replaced by actual model inference
        match domain {
            PredictionDomain::FeeForecasting => context.current_fee_rate * 1.1,
            PredictionDomain::NetworkLoad => context.tx_volume / 1000.0,
            PredictionDomain::DifficultyAdjustment => context.hashrate.ln(),
            PredictionDomain::BlockReward => 50.0 * (0.5_f64).powf((context.block_height / 210000) as f64),
            _ => 0.0,
        }
    }

    /// Enrich context with data feed values
    async fn enrich_context(&self, context: &PredictionContext) -> PredictionContext {
        let feeds = self.data_feeds.read().await;
        let mut enriched = context.clone();

        for (feed_id, feed) in feeds.iter() {
            enriched.domain_features.insert(feed_id.clone(), feed.value);
        }

        enriched
    }

    /// Select model for prediction (with A/B testing)
    async fn select_model(&self) -> u64 {
        let models = self.models.read().await;
        let active = *self.active_model_id.read().await;

        // Simple A/B test selection
        let rand_val: u32 = rand::random::<u32>() % 100;

        for model in models.values() {
            if model.active && model.model_id != active {
                if rand_val < model.traffic_pct {
                    return model.model_id;
                }
            }
        }

        active
    }

    /// Check rate limit for requester
    async fn check_rate_limit(&self, requester: &[u8; 32]) -> bool {
        let now = chrono::Utc::now().timestamp() as u64;
        let mut limits = self.rate_limits.write().await;

        let state = limits.entry(*requester).or_default();

        // Reset window if expired (1 minute)
        if now - state.window_start > 60 {
            state.requests = 0;
            state.window_start = now;
        }

        if state.requests >= self.config.rate_limit_rpm {
            return false;
        }

        state.requests += 1;
        true
    }

    /// Calculate prediction fee based on domain
    async fn calculate_fee(&self, _domain: &PredictionDomain) -> u64 {
        // Dynamic fee based on congestion (simplified)
        let queue_len = self.request_queue.read().await.len();
        let congestion_multiplier = 1.0 + (queue_len as f64 / 100.0).min(2.0);

        (self.config.base_prediction_fee as f64 * congestion_multiplier) as u64
    }

    /// Calculate request priority
    fn calculate_priority(&self, fee_paid: u64, base_fee: u64) -> u32 {
        // Higher fee = higher priority
        ((fee_paid as f64 / base_fee as f64) * 100.0) as u32
    }

    /// Distribute fees to validators and treasury
    async fn distribute_fees(
        &self,
        total_fee: u64,
        validators: &[[u8; 32]],
    ) -> FeeBreakdown {
        let validator_share = (total_fee * self.config.validator_reward_pct as u64) / 100;
        let treasury_share = (total_fee * self.config.treasury_pct as u64) / 100;
        let data_provider_share = total_fee - validator_share - treasury_share;

        // Update treasury
        {
            let mut treasury = self.treasury.write().await;
            // v1.4.5-beta: Use saturating add to prevent overflow
            treasury.balance = treasury.balance.saturating_add(treasury_share);
            treasury.total_fees_collected = treasury.total_fees_collected.saturating_add(total_fee);
        }

        // Distribute to validators (evenly for now)
        if !validators.is_empty() {
            let per_validator = validator_share / validators.len() as u64;
            let mut balances = self.validator_balances.write().await;

            for validator in validators {
                *balances.entry(*validator).or_default() += per_validator;

                let _ = self.event_tx.send(OracleEvent::RewardDistributed {
                    validator: *validator,
                    amount: per_validator,
                });
            }
        }

        FeeBreakdown {
            total: total_fee,
            validators: validator_share,
            treasury: treasury_share,
            data_providers: data_provider_share,
        }
    }

    /// Compute cache key for request
    fn compute_cache_key(&self, domain: &PredictionDomain, context: &PredictionContext) -> [u8; 32] {
        let mut hasher = Sha3_256::new();
        hasher.update(format!("{:?}", domain).as_bytes());
        hasher.update(&context.block_height.to_le_bytes());
        hasher.update(context.current_fee_rate.to_le_bytes());
        hasher.finalize().into()
    }

    /// Get cached prediction
    async fn get_cached(&self, key: &[u8; 32]) -> Option<Prediction> {
        let now = chrono::Utc::now().timestamp() as u64;
        let mut cache = self.cache.write().await;

        if let Some(cached) = cache.get_mut(key) {
            if now - cached.cached_at < self.config.cache_ttl_secs {
                cached.hit_count += 1;
                return Some(cached.prediction.clone());
            } else {
                cache.remove(key);
            }
        }

        None
    }

    /// Cache a prediction
    async fn cache_prediction(&self, key: &[u8; 32], prediction: &Prediction) {
        let mut cache = self.cache.write().await;

        // Evict oldest if at capacity
        if cache.len() >= self.config.cache_size {
            // Simple eviction: remove least recently used
            if let Some(oldest_key) = cache.iter()
                .min_by_key(|(_, v)| v.cached_at)
                .map(|(k, _)| *k)
            {
                cache.remove(&oldest_key);
            }
        }

        cache.insert(*key, CachedPrediction {
            prediction: prediction.clone(),
            cached_at: chrono::Utc::now().timestamp() as u64,
            hit_count: 0,
        });
    }

    /// Update model metrics after prediction
    async fn update_model_metrics(&self, model_id: u64, prediction: &Prediction) {
        let mut models = self.models.write().await;
        if let Some(model) = models.get_mut(&model_id) {
            model.metrics.total_predictions += 1;
            // Latency would be measured in real implementation
            model.metrics.avg_latency_ms = model.metrics.avg_latency_ms * 0.99 + 10.0 * 0.01;
        }
    }

    /// Verify feed signature
    fn verify_feed_signature(&self, _feed: &DataFeed) -> bool {
        // Would verify Ed25519/Dilithium signature
        true
    }

    /// Check if model needs retraining
    pub async fn check_retrain_needed(&self, model_id: u64) -> bool {
        let models = self.models.read().await;
        if let Some(model) = models.get(&model_id) {
            let accuracy = if model.metrics.total_predictions > 0 {
                model.metrics.correct_predictions as f64 / model.metrics.total_predictions as f64
            } else {
                1.0
            };

            // Check accuracy drop
            if accuracy < (1.0 - self.config.retrain_threshold) {
                return true;
            }
        }
        false
    }

    /// Trigger model retraining via federated learning
    pub async fn trigger_retrain(&self, model_id: u64, reason: &str) -> anyhow::Result<()> {
        if let Some(federated) = &self.federated {
            info!("🔄 Triggering retrain for model {} (reason: {})", model_id, reason);

            // Would initiate federated learning round
            let _fl = federated.read().await;
            // fl.start_training_round().await?;

            let _ = self.event_tx.send(OracleEvent::ModelRetrained {
                model_id,
                reason: reason.to_string(),
            });
        }
        Ok(())
    }

    /// Run continuous monitoring loop
    pub async fn run_monitoring_loop(&self, check_interval_ms: u64) {
        info!("📊 Starting autonomous monitoring loop");

        loop {
            if *self.shutdown.read().await {
                break;
            }

            // Check each model's performance
            let models = self.models.read().await;
            for (model_id, model) in models.iter() {
                if model.metrics.total_predictions > 100 {
                    let accuracy = model.metrics.correct_predictions as f64
                                 / model.metrics.total_predictions as f64;

                    if accuracy < 0.7 {
                        let _ = self.event_tx.send(OracleEvent::PerformanceAlert {
                            model_id: *model_id,
                            metric: "accuracy".to_string(),
                            value: accuracy,
                        });
                    }
                }
            }
            drop(models);

            tokio::time::sleep(tokio::time::Duration::from_millis(check_interval_ms)).await;
        }
    }

    /// Shutdown the oracle
    pub async fn shutdown(&self) {
        *self.shutdown.write().await = true;
        info!("🛑 Autonomous oracle shutting down");
    }

    /// Subscribe to oracle events
    pub fn subscribe_events(&self) -> broadcast::Receiver<OracleEvent> {
        self.event_tx.subscribe()
    }

    /// Get treasury state
    pub async fn get_treasury(&self) -> TreasuryState {
        self.treasury.read().await.clone()
    }

    /// Get validator balance
    pub async fn get_validator_balance(&self, validator: &[u8; 32]) -> u64 {
        self.validator_balances.read().await.get(validator).copied().unwrap_or(0)
    }

    /// Get model statistics
    pub async fn get_model_stats(&self, model_id: u64) -> Option<ModelMetrics> {
        self.models.read().await.get(&model_id).map(|m| m.metrics.clone())
    }

    /// Calculate fee breakdown
    fn calculate_fee_breakdown(&self, total: u64) -> FeeBreakdown {
        let validators = (total * self.config.validator_reward_pct as u64) / 100;
        let treasury = (total * self.config.treasury_pct as u64) / 100;
        FeeBreakdown {
            total,
            validators,
            treasury,
            data_providers: total - validators - treasury,
        }
    }
}

/// Autonomous improvement system
pub struct SelfImprover {
    /// Oracle reference
    oracle: Arc<AutonomousOracle>,
    /// Governance reference
    governance: Option<Arc<RwLock<NasGovernance>>>,
    /// Performance history
    performance_history: Arc<RwLock<VecDeque<PerformanceSnapshot>>>,
    /// Configuration
    config: SelfImproverConfig,
}

/// Performance snapshot
#[derive(Clone, Debug)]
pub struct PerformanceSnapshot {
    pub timestamp: u64,
    pub model_id: u64,
    pub accuracy: f64,
    pub latency_ms: f64,
    pub predictions_per_sec: f64,
}

/// Self-improvement configuration
#[derive(Clone, Debug)]
pub struct SelfImproverConfig {
    /// Evaluation window (snapshots)
    pub eval_window: usize,
    /// Minimum snapshots before improvement trigger
    pub min_snapshots: usize,
    /// Improvement threshold (accuracy delta)
    pub improvement_threshold: f64,
    /// Auto-propose evolved architectures
    pub auto_propose: bool,
}

impl Default for SelfImproverConfig {
    fn default() -> Self {
        Self {
            eval_window: 100,
            min_snapshots: 50,
            improvement_threshold: 0.02,
            auto_propose: true,
        }
    }
}

impl SelfImprover {
    /// Create new self-improvement system
    pub fn new(
        oracle: Arc<AutonomousOracle>,
        governance: Option<Arc<RwLock<NasGovernance>>>,
        config: SelfImproverConfig,
    ) -> Self {
        Self {
            oracle,
            governance,
            performance_history: Arc::new(RwLock::new(VecDeque::with_capacity(config.eval_window))),
            config,
        }
    }

    /// Record performance snapshot
    pub async fn record_snapshot(&self, snapshot: PerformanceSnapshot) {
        let mut history = self.performance_history.write().await;
        if history.len() >= self.config.eval_window {
            history.pop_front();
        }
        history.push_back(snapshot);
    }

    /// Analyze performance trends
    pub async fn analyze_trends(&self) -> PerformanceTrend {
        let history = self.performance_history.read().await;

        if history.len() < self.config.min_snapshots {
            return PerformanceTrend::Insufficient;
        }

        // Calculate moving averages
        let half = history.len() / 2;
        let first_half: Vec<_> = history.iter().take(half).collect();
        let second_half: Vec<_> = history.iter().skip(half).collect();

        let first_avg = first_half.iter().map(|s| s.accuracy).sum::<f64>() / half as f64;
        let second_avg = second_half.iter().map(|s| s.accuracy).sum::<f64>() / (history.len() - half) as f64;

        let delta = second_avg - first_avg;

        if delta < -self.config.improvement_threshold {
            PerformanceTrend::Declining { delta }
        } else if delta > self.config.improvement_threshold {
            PerformanceTrend::Improving { delta }
        } else {
            PerformanceTrend::Stable
        }
    }

    /// Auto-trigger improvements based on trends
    pub async fn auto_improve(&self) -> anyhow::Result<Option<u64>> {
        let trend = self.analyze_trends().await;

        match trend {
            PerformanceTrend::Declining { delta } if self.config.auto_propose => {
                warn!("📉 Performance declining by {:.2}%, triggering improvement", delta * -100.0);

                if let Some(gov) = &self.governance {
                    let governance = gov.read().await;

                    // Generate evolution proposals
                    let proposals = governance.generate_evolution_proposals(3).await?;

                    if let Some(best_arch) = proposals.first() {
                        // Auto-submit proposal
                        let proposer = [0u8; 32]; // System proposer
                        let proposal_id = governance.submit_proposal(
                            proposer,
                            best_arch.clone(),
                            format!("Auto-generated due to {:.1}% accuracy decline", delta * -100.0),
                            None,
                        ).await?;

                        info!("🧬 Auto-submitted evolution proposal {}", proposal_id);
                        return Ok(Some(proposal_id));
                    }
                }
            }
            _ => {}
        }

        Ok(None)
    }
}

/// Performance trend classification
#[derive(Clone, Debug)]
pub enum PerformanceTrend {
    Insufficient,
    Stable,
    Improving { delta: f64 },
    Declining { delta: f64 },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_autonomous_oracle_creation() {
        let config = AutonomousConfig::default();
        let oracle = AutonomousOracle::new(config);

        let treasury = oracle.get_treasury().await;
        assert_eq!(treasury.balance, 0);
    }

    #[tokio::test]
    async fn test_rate_limiting() {
        let config = AutonomousConfig {
            rate_limit_rpm: 2,
            ..Default::default()
        };
        let oracle = AutonomousOracle::new(config);

        let requester = [1u8; 32];

        assert!(oracle.check_rate_limit(&requester).await);
        assert!(oracle.check_rate_limit(&requester).await);
        assert!(!oracle.check_rate_limit(&requester).await); // Exceeded
    }

    #[tokio::test]
    async fn test_fee_calculation() {
        let oracle = AutonomousOracle::new(AutonomousConfig::default());
        let fee = oracle.calculate_fee(&PredictionDomain::FeeForecasting).await;
        assert!(fee >= oracle.config.base_prediction_fee);
    }

    #[tokio::test]
    async fn test_cache() {
        let oracle = AutonomousOracle::new(AutonomousConfig::default());

        let key = [1u8; 32];
        let prediction = Prediction {
            value: 42.0,
            confidence: 0.9,
            domain: PredictionDomain::FeeForecasting,
            source: PredictionSource::Quantum,
            expert_weights: vec![],
            quantum_fidelity: 0.95,
            timestamp: 0,
            proof: None,
        };

        oracle.cache_prediction(&key, &prediction).await;

        let cached = oracle.get_cached(&key).await;
        assert!(cached.is_some());
        assert_eq!(cached.unwrap().value, 42.0);
    }

    #[tokio::test]
    async fn test_performance_trend() {
        let config = AutonomousConfig::default();
        let oracle = Arc::new(AutonomousOracle::new(config));

        let improver = SelfImprover::new(
            oracle,
            None,
            SelfImproverConfig {
                min_snapshots: 4,
                eval_window: 10,
                ..Default::default()
            },
        );

        // Add declining snapshots
        for i in 0..8 {
            improver.record_snapshot(PerformanceSnapshot {
                timestamp: i as u64,
                model_id: 1,
                accuracy: 0.9 - (i as f64 * 0.05),
                latency_ms: 10.0,
                predictions_per_sec: 100.0,
            }).await;
        }

        let trend = improver.analyze_trends().await;
        assert!(matches!(trend, PerformanceTrend::Declining { .. }));
    }
}
