//! Inference Worker Pool — Runs AI inference on idle cores
//!
//! v9.6.0: Bridges the Compute Orchestrator with q-ai-inference engines.
//! When the orchestrator assigns cores to the AiInference layer, this pool
//! accepts inference tasks from gossipsub, local API, or tunnel mesh.
//!
//! v9.6.1: Revenue callback wiring (Issue #014).
//! - `ModelTier` enum for tiered pricing (Small/Medium/Large/XL)
//! - `record_job_completion()` calculates revenue from model tier + token count
//! - `RevenueCallback` type: external observers (e.g., orchestrator) get notified
//! - `get_revenue_summary()` returns aggregated revenue stats with per-hour rate
//!
//! ## Design
//!
//! - Shares an `Arc<dyn InferenceEngine>` with the chat API (one model loaded)
//! - Respects core budget — pauses if mining needs cores back
//! - Reports task completion + revenue to orchestrator via `record_task()`
//! - Supports graceful pause: finishes current token generation, won't start new tasks

use crate::ComputeLayer;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use parking_lot::RwLock;
use tracing::{info, warn, debug, error};

/// Per-token price in micro-QUG (default: 1 micro-QUG per token = 0.000001 QUG)
pub const DEFAULT_PRICE_PER_TOKEN_MICRO_QUG: u64 = 1;

/// v9.5.1: Read price per token from env var, fallback to default (#014)
pub fn configured_price_per_token() -> u64 {
    std::env::var("INFERENCE_PRICE_PER_TOKEN")
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(DEFAULT_PRICE_PER_TOKEN_MICRO_QUG)
}

// ═══════════════════════════════════════════════════════════════════
// Issue #014: Revenue callback types and tiered pricing
// ═══════════════════════════════════════════════════════════════════

/// Callback invoked when inference revenue is earned.
/// The argument is the revenue amount in micro-QUG.
/// Must be Send + Sync because the pool is accessed from multiple async tasks.
pub type RevenueCallback = Arc<dyn Fn(u64) + Send + Sync>;

/// Model tier determines per-1K-token pricing in micro-QUG.
///
/// Pricing is intentionally simple: a fixed rate per 1,000 tokens generated,
/// scaling with model size / compute cost.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModelTier {
    /// Small models (< 3B params): 10 micro-QUG per 1K tokens
    Small,
    /// Medium models (3B-13B params): 50 micro-QUG per 1K tokens
    Medium,
    /// Large models (13B-70B params): 200 micro-QUG per 1K tokens
    Large,
    /// XL models (70B+ params): 1000 micro-QUG per 1K tokens
    XL,
}

impl ModelTier {
    /// Price in micro-QUG per 1,000 tokens generated.
    pub fn price_per_1k_tokens(&self) -> u64 {
        match self {
            ModelTier::Small => 10,
            ModelTier::Medium => 50,
            ModelTier::Large => 200,
            ModelTier::XL => 1000,
        }
    }

    /// Calculate revenue for a given number of tokens.
    /// Uses integer arithmetic: `(tokens * price_per_1k) / 1000`, with a
    /// minimum of 1 micro-QUG for any non-zero token count to avoid
    /// rounding to zero on small completions.
    pub fn calculate_revenue(&self, tokens_generated: u64) -> u64 {
        if tokens_generated == 0 {
            return 0;
        }
        let revenue = tokens_generated
            .saturating_mul(self.price_per_1k_tokens())
            / 1000;
        // Guarantee at least 1 micro-QUG for any work done
        revenue.max(1)
    }
}

impl std::fmt::Display for ModelTier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ModelTier::Small => write!(f, "small"),
            ModelTier::Medium => write!(f, "medium"),
            ModelTier::Large => write!(f, "large"),
            ModelTier::XL => write!(f, "xl"),
        }
    }
}

impl std::str::FromStr for ModelTier {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "small" | "s" | "tiny" => Ok(ModelTier::Small),
            "medium" | "m" | "mid" => Ok(ModelTier::Medium),
            "large" | "l" | "big" => Ok(ModelTier::Large),
            "xl" | "xxl" | "extra-large" | "huge" => Ok(ModelTier::XL),
            _ => Err(format!("Unknown model tier: '{}'. Use: small, medium, large, xl", s)),
        }
    }
}

/// Aggregated revenue summary returned by `get_revenue_summary()`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RevenueSummary {
    /// Total revenue earned in micro-QUG since pool start
    pub total_revenue_micro_qug: u64,
    /// Total inference jobs completed
    pub total_jobs_completed: u64,
    /// Revenue per hour in micro-QUG (calculated from uptime)
    pub revenue_per_hour: f64,
    /// Total tokens generated across all jobs
    pub total_tokens_generated: u64,
    /// Pool uptime in seconds
    pub uptime_seconds: f64,
}

/// Inference task submitted to the pool
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceTask {
    /// Unique task ID
    pub id: String,
    /// Source: "api", "gossipsub", "tunnel"
    pub source: String,
    /// Requestor wallet (for billing)
    pub wallet: Option<String>,
    /// The prompt to process
    pub prompt: String,
    /// Max tokens to generate
    pub max_tokens: usize,
    /// Model to use (if specific model requested)
    pub model: Option<String>,
    /// Submitted timestamp (unix millis)
    pub submitted_ms: u64,
}

/// Result of a completed inference task
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceTaskResult {
    pub task_id: String,
    pub generated_text: String,
    pub tokens_generated: usize,
    pub tokens_per_second: f64,
    pub total_time_ms: f64,
    pub revenue_micro_qug: u64,
}

/// Aggregate statistics for the inference pool
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AIInferenceStats {
    pub total_requests_served: u64,
    pub total_tokens_generated: u64,
    pub revenue_earned_micro_qug: u64,
    pub avg_tokens_per_second: f32,
    pub model_loaded: String,
    pub active_since_ms: u64,
    pub tasks_in_queue: u32,
    pub active_tasks: u32,
    /// v9.5.1: Per-token price in micro-QUG (#014)
    pub price_per_token_micro_qug: u64,
    /// v9.5.1: Max concurrent tasks (synced from orchestrator core budget)
    pub max_concurrent: u64,
}

/// The Inference Worker Pool — runs inference on orchestrator-assigned cores
pub struct InferenceWorkerPool {
    /// Shared inference engine (same one used by chat API)
    engine: Arc<RwLock<Option<Arc<dyn q_ai_inference::InferenceEngine>>>>,
    /// Cores currently assigned by orchestrator
    assigned_cores: Arc<RwLock<Vec<usize>>>,
    /// Currently active concurrent tasks
    active_tasks: Arc<AtomicU64>,
    /// Total completed tasks
    completed_tasks: Arc<AtomicU64>,
    /// Total tokens generated
    total_tokens: Arc<AtomicU64>,
    /// Revenue earned in micro-QUG
    revenue_earned: Arc<AtomicU64>,
    /// v9.6.1 (#014): Jobs completed counter (distinct from completed_tasks which
    /// tracks the spawn() background loop; this tracks record_job_completion calls)
    jobs_completed: Arc<AtomicU64>,
    /// Cumulative tokens per second (for averaging)
    cumulative_tps: Arc<RwLock<f64>>,
    /// Task queue
    task_queue: Arc<RwLock<VecDeque<InferenceTask>>>,
    /// Whether the pool is accepting new tasks
    accepting: Arc<AtomicBool>,
    /// Whether the pool is running
    running: Arc<AtomicBool>,
    /// Name of loaded model
    model_name: Arc<RwLock<String>>,
    /// When pool was started (unix millis)
    started_ms: u64,
    /// Max concurrent tasks (derived from assigned cores)
    max_concurrent: Arc<AtomicU64>,
    /// Orchestrator callback for recording tasks
    orchestrator_record: Option<Arc<dyn Fn(ComputeLayer, u64) + Send + Sync>>,
    /// v9.6.1 (#014): External revenue callback — notified on every revenue event
    revenue_callback: Option<RevenueCallback>,
    /// #013: Round-robin counter for core affinity assignment
    next_core_index: Arc<AtomicU64>,
}

impl InferenceWorkerPool {
    /// Create a new inference worker pool
    pub fn new() -> Self {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        Self {
            engine: Arc::new(RwLock::new(None)),
            assigned_cores: Arc::new(RwLock::new(Vec::new())),
            active_tasks: Arc::new(AtomicU64::new(0)),
            completed_tasks: Arc::new(AtomicU64::new(0)),
            total_tokens: Arc::new(AtomicU64::new(0)),
            revenue_earned: Arc::new(AtomicU64::new(0)),
            jobs_completed: Arc::new(AtomicU64::new(0)),
            cumulative_tps: Arc::new(RwLock::new(0.0)),
            task_queue: Arc::new(RwLock::new(VecDeque::new())),
            accepting: Arc::new(AtomicBool::new(false)),
            running: Arc::new(AtomicBool::new(false)),
            model_name: Arc::new(RwLock::new("none".to_string())),
            started_ms: now,
            max_concurrent: Arc::new(AtomicU64::new(0)),
            orchestrator_record: None,
            revenue_callback: None,
            next_core_index: Arc::new(AtomicU64::new(0)),
        }
    }

    /// Set the inference engine (shared with chat API)
    pub fn set_engine(&self, engine: Arc<dyn q_ai_inference::InferenceEngine>) {
        let name = engine.engine_name().to_string();
        info!("🧠 [INFERENCE POOL] Engine loaded: {}", name);
        *self.model_name.write() = name;
        *self.engine.write() = Some(engine);
    }

    /// Set the orchestrator callback for recording task completions
    pub fn set_orchestrator_callback<F: Fn(ComputeLayer, u64) + Send + Sync + 'static>(&mut self, callback: F) {
        self.orchestrator_record = Some(Arc::new(callback));
    }

    /// v9.6.1 (#014): Set an external revenue callback.
    ///
    /// The callback is invoked with the revenue amount (in micro-QUG) every
    /// time an inference job completes and revenue is recorded. This allows
    /// external systems (dashboards, billing, orchestrator) to react to
    /// revenue events without polling.
    pub fn set_revenue_callback(&mut self, callback: RevenueCallback) {
        self.revenue_callback = Some(callback);
    }

    /// v9.6.1 (#014): Record completion of an inference job with tiered pricing.
    ///
    /// Calculates revenue based on model tier and tokens generated, then:
    /// 1. Increments the atomic `revenue_earned` counter
    /// 2. Increments the `jobs_completed` counter
    /// 3. Adds to the `total_tokens` counter
    /// 4. Invokes the revenue callback (if set)
    /// 5. Invokes the orchestrator callback (if set)
    /// 6. Logs the revenue event
    ///
    /// This method is safe to call from multiple async tasks concurrently.
    pub fn record_job_completion(
        &self,
        job_id: &str,
        tokens_generated: u64,
        model_tier: ModelTier,
    ) {
        let revenue = model_tier.calculate_revenue(tokens_generated);

        // 1. Increment atomic counters
        self.revenue_earned.fetch_add(revenue, Ordering::Relaxed);
        self.jobs_completed.fetch_add(1, Ordering::Relaxed);
        self.total_tokens.fetch_add(tokens_generated, Ordering::Relaxed);

        // 2. Invoke revenue callback (external observers)
        if let Some(ref cb) = self.revenue_callback {
            cb(revenue);
        }

        // 3. Invoke orchestrator callback (layer stats aggregation)
        if let Some(ref record_fn) = self.orchestrator_record {
            record_fn(ComputeLayer::AiInference, revenue);
        }

        info!(
            "🧠 [INFERENCE POOL] Job {} completed: {} tokens (tier={}), {} micro-QUG revenue",
            job_id, tokens_generated, model_tier, revenue
        );
    }

    /// v9.6.1 (#014): Get aggregated revenue summary.
    ///
    /// Returns total revenue, job count, tokens generated, and a calculated
    /// revenue-per-hour rate based on pool uptime.
    pub fn get_revenue_summary(&self) -> RevenueSummary {
        let total_revenue = self.revenue_earned.load(Ordering::Relaxed);
        let total_jobs = self.jobs_completed.load(Ordering::Relaxed);
        let total_tokens = self.total_tokens.load(Ordering::Relaxed);

        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        let uptime_ms = now_ms.saturating_sub(self.started_ms);
        let uptime_seconds = uptime_ms as f64 / 1000.0;

        // Calculate revenue per hour (avoid division by zero)
        let revenue_per_hour = if uptime_seconds > 0.0 {
            (total_revenue as f64 / uptime_seconds) * 3600.0
        } else {
            0.0
        };

        RevenueSummary {
            total_revenue_micro_qug: total_revenue,
            total_jobs_completed: total_jobs,
            revenue_per_hour,
            total_tokens_generated: total_tokens,
            uptime_seconds,
        }
    }

    /// Update assigned cores (called by orchestrator scheduler)
    pub fn update_cores(&self, cores: Vec<usize>) {
        let num_cores = cores.len();
        *self.assigned_cores.write() = cores;

        // Scale max concurrent: 1 task per 2 cores, 0 if no cores assigned
        let max_conc = if num_cores == 0 { 0 } else { (num_cores / 2).max(1) as u64 };
        self.max_concurrent.store(max_conc, Ordering::Relaxed);

        if num_cores > 0 && !self.accepting.load(Ordering::Relaxed) {
            self.accepting.store(true, Ordering::Relaxed);
            info!("🧠 [INFERENCE POOL] Activated with {} cores, max {} concurrent tasks", num_cores, max_conc);
        } else if num_cores == 0 && self.accepting.load(Ordering::Relaxed) {
            self.accepting.store(false, Ordering::Relaxed);
            info!("🧠 [INFERENCE POOL] Paused — all cores reclaimed by mining");
        }
    }

    /// Submit a task to the pool
    pub fn submit_task(&self, task: InferenceTask) -> bool {
        if !self.accepting.load(Ordering::Relaxed) {
            debug!("🧠 [INFERENCE POOL] Rejected task {} — pool paused", task.id);
            return false;
        }

        let queue_len = {
            let mut queue = self.task_queue.write();
            if queue.len() >= 100 {
                warn!("🧠 [INFERENCE POOL] Task queue full (100), rejecting {}", task.id);
                return false;
            }
            queue.push_back(task);
            queue.len()
        };

        debug!("🧠 [INFERENCE POOL] Task queued (queue_len={})", queue_len);
        true
    }

    /// Start the pool's background task processing loop
    pub fn spawn(&self) {
        if self.running.load(Ordering::Relaxed) {
            return;
        }
        self.running.store(true, Ordering::SeqCst);

        let engine = self.engine.clone();
        let task_queue = self.task_queue.clone();
        let active_tasks = self.active_tasks.clone();
        let completed_tasks = self.completed_tasks.clone();
        let total_tokens = self.total_tokens.clone();
        let revenue_earned = self.revenue_earned.clone();
        let cumulative_tps = self.cumulative_tps.clone();
        let accepting = self.accepting.clone();
        let running = self.running.clone();
        let max_concurrent = self.max_concurrent.clone();
        let orch_record = self.orchestrator_record.clone();
        let assigned_cores_ref = self.assigned_cores.clone();
        let next_core_idx = self.next_core_index.clone();

        let price_per_token = configured_price_per_token();

        tokio::spawn(async move {
            info!("🧠 [INFERENCE POOL] Background worker started (price={} µQUG/token)", price_per_token);
            let mut interval = tokio::time::interval(std::time::Duration::from_millis(100));

            loop {
                interval.tick().await;
                if !running.load(Ordering::Relaxed) {
                    break;
                }

                // Check if we can take a task
                if !accepting.load(Ordering::Relaxed) {
                    continue;
                }

                let current_active = active_tasks.load(Ordering::Relaxed);
                let max_conc = max_concurrent.load(Ordering::Relaxed);
                if current_active >= max_conc {
                    continue;
                }

                // Get engine
                let eng = {
                    let guard = engine.read();
                    match guard.as_ref() {
                        Some(e) => e.clone(),
                        None => continue,
                    }
                };

                // Dequeue a task
                let task = {
                    let mut queue = task_queue.write();
                    queue.pop_front()
                };

                let task = match task {
                    Some(t) => t,
                    None => continue,
                };

                // #013: Pick a core from the assigned range (round-robin)
                let pin_core_id = {
                    let cores = assigned_cores_ref.read();
                    if !cores.is_empty() {
                        let idx = next_core_idx.fetch_add(1, Ordering::Relaxed) as usize % cores.len();
                        Some(cores[idx])
                    } else {
                        None
                    }
                };

                // Spawn inference for this task
                let at = active_tasks.clone();
                let ct = completed_tasks.clone();
                let tt = total_tokens.clone();
                let re = revenue_earned.clone();
                let ctps = cumulative_tps.clone();
                let record = orch_record.clone();

                at.fetch_add(1, Ordering::Relaxed);

                tokio::spawn(async move {
                    let task_id = task.id.clone();

                    // #013: Pin this tokio worker thread to the assigned core
                    if let Some(core_id) = pin_core_id {
                        let success = core_affinity::set_for_current(core_affinity::CoreId { id: core_id });
                        if success {
                            debug!("🧠 [INFERENCE POOL] Task {} pinned to core {}", task_id, core_id);
                        } else {
                            debug!("🧠 [INFERENCE POOL] Task {} core affinity failed for core {} (advisory fallback)", task_id, core_id);
                        }
                    }

                    debug!("🧠 [INFERENCE POOL] Processing task {}", task_id);

                    match eng.generate(&task.prompt, task.max_tokens).await {
                        Ok(text) => {
                            let stats = eng.get_stats().await;
                            let tokens = stats.tokens_generated;
                            let revenue = tokens as u64 * price_per_token;

                            ct.fetch_add(1, Ordering::Relaxed);
                            tt.fetch_add(tokens as u64, Ordering::Relaxed);
                            re.fetch_add(revenue, Ordering::Relaxed);

                            {
                                let mut tps = ctps.write();
                                *tps += stats.tokens_per_second;
                            }

                            // Report to orchestrator
                            if let Some(ref record_fn) = record {
                                record_fn(ComputeLayer::AiInference, revenue);
                            }

                            debug!(
                                "🧠 [INFERENCE POOL] Task {} completed: {} tokens, {:.1} tok/s, {} µQUG revenue",
                                task_id, tokens, stats.tokens_per_second, revenue
                            );
                        }
                        Err(e) => {
                            error!("🧠 [INFERENCE POOL] Task {} failed: {}", task_id, e);
                        }
                    }

                    at.fetch_sub(1, Ordering::Relaxed);
                });
            }

            info!("🧠 [INFERENCE POOL] Background worker stopped");
        });
    }

    /// Stop the pool
    pub fn stop(&self) {
        self.running.store(false, Ordering::SeqCst);
        self.accepting.store(false, Ordering::SeqCst);
        info!("🧠 [INFERENCE POOL] Stopped");
    }

    /// Get current inference statistics
    pub fn stats(&self) -> AIInferenceStats {
        let completed = self.completed_tasks.load(Ordering::Relaxed);
        let avg_tps = if completed > 0 {
            let total_tps = *self.cumulative_tps.read();
            (total_tps / completed as f64) as f32
        } else {
            0.0
        };

        AIInferenceStats {
            total_requests_served: completed,
            total_tokens_generated: self.total_tokens.load(Ordering::Relaxed),
            revenue_earned_micro_qug: self.revenue_earned.load(Ordering::Relaxed),
            avg_tokens_per_second: avg_tps,
            model_loaded: self.model_name.read().clone(),
            active_since_ms: self.started_ms,
            tasks_in_queue: self.task_queue.read().len() as u32,
            active_tasks: self.active_tasks.load(Ordering::Relaxed) as u32,
            price_per_token_micro_qug: configured_price_per_token(),
            max_concurrent: self.max_concurrent.load(Ordering::Relaxed),
        }
    }

    /// Whether the pool is currently accepting tasks
    pub fn is_accepting(&self) -> bool {
        self.accepting.load(Ordering::Relaxed)
    }

    /// Whether the pool has an engine loaded
    pub fn has_engine(&self) -> bool {
        self.engine.read().is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pool_creation() {
        let pool = InferenceWorkerPool::new();
        assert!(!pool.is_accepting());
        assert!(!pool.has_engine());
        assert_eq!(pool.stats().total_requests_served, 0);
        // #035: starts at 0, not 2
        assert_eq!(pool.max_concurrent.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn test_core_update() {
        let pool = InferenceWorkerPool::new();

        // No cores = not accepting, max_concurrent = 0
        pool.update_cores(vec![]);
        assert!(!pool.is_accepting());
        assert_eq!(pool.max_concurrent.load(Ordering::Relaxed), 0);

        // Give it cores = accepting
        pool.update_cores(vec![4, 5, 6, 7]);
        assert!(pool.is_accepting());
        assert_eq!(pool.max_concurrent.load(Ordering::Relaxed), 2); // 4 cores / 2

        // Take cores back = paused, max_concurrent = 0
        pool.update_cores(vec![]);
        assert!(!pool.is_accepting());
        assert_eq!(pool.max_concurrent.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn test_configured_price_default() {
        // Without env var, should return default
        let price = configured_price_per_token();
        assert!(price >= 1, "Default price should be at least 1 µQUG/token");
    }

    #[test]
    fn test_stats_include_pricing() {
        let pool = InferenceWorkerPool::new();
        let stats = pool.stats();
        assert!(stats.price_per_token_micro_qug >= 1);
        assert_eq!(stats.max_concurrent, 0); // No cores assigned yet

        pool.update_cores(vec![0, 1, 2, 3]);
        let stats = pool.stats();
        assert_eq!(stats.max_concurrent, 2); // 4 cores / 2
    }

    #[test]
    fn test_task_reject_when_paused() {
        let pool = InferenceWorkerPool::new();
        let task = InferenceTask {
            id: "test-1".to_string(),
            source: "api".to_string(),
            wallet: None,
            prompt: "Hello".to_string(),
            max_tokens: 100,
            model: None,
            submitted_ms: 0,
        };

        // Pool not accepting, task rejected
        assert!(!pool.submit_task(task));
    }

    // ═══════════════════════════════════════════════════════════════════
    // Issue #014: Revenue callback and tiered pricing tests
    // ═══════════════════════════════════════════════════════════════════

    #[test]
    fn test_model_tier_pricing() {
        // Verify the per-1K-token prices match spec
        assert_eq!(ModelTier::Small.price_per_1k_tokens(), 10);
        assert_eq!(ModelTier::Medium.price_per_1k_tokens(), 50);
        assert_eq!(ModelTier::Large.price_per_1k_tokens(), 200);
        assert_eq!(ModelTier::XL.price_per_1k_tokens(), 1000);
    }

    #[test]
    fn test_model_tier_revenue_calculation() {
        // Small: 10 per 1K tokens
        // 1000 tokens * 10 / 1000 = 10 micro-QUG
        assert_eq!(ModelTier::Small.calculate_revenue(1000), 10);
        // 5000 tokens * 10 / 1000 = 50 micro-QUG
        assert_eq!(ModelTier::Small.calculate_revenue(5000), 50);

        // Medium: 50 per 1K tokens
        // 2000 tokens * 50 / 1000 = 100 micro-QUG
        assert_eq!(ModelTier::Medium.calculate_revenue(2000), 100);

        // Large: 200 per 1K tokens
        // 1000 tokens * 200 / 1000 = 200 micro-QUG
        assert_eq!(ModelTier::Large.calculate_revenue(1000), 200);

        // XL: 1000 per 1K tokens
        // 500 tokens * 1000 / 1000 = 500 micro-QUG
        assert_eq!(ModelTier::XL.calculate_revenue(500), 500);
        // 10000 tokens * 1000 / 1000 = 10000 micro-QUG
        assert_eq!(ModelTier::XL.calculate_revenue(10000), 10000);
    }

    #[test]
    fn test_model_tier_revenue_zero_tokens() {
        // Zero tokens = zero revenue, no minimum
        assert_eq!(ModelTier::Small.calculate_revenue(0), 0);
        assert_eq!(ModelTier::Medium.calculate_revenue(0), 0);
        assert_eq!(ModelTier::Large.calculate_revenue(0), 0);
        assert_eq!(ModelTier::XL.calculate_revenue(0), 0);
    }

    #[test]
    fn test_model_tier_revenue_small_token_count_minimum() {
        // Very small token count: 1 token at Small tier
        // 1 * 10 / 1000 = 0, but minimum is 1
        assert_eq!(ModelTier::Small.calculate_revenue(1), 1);
        // 5 tokens at Small: 5 * 10 / 1000 = 0, minimum 1
        assert_eq!(ModelTier::Small.calculate_revenue(5), 1);
        // 99 tokens at Small: 99 * 10 / 1000 = 0, minimum 1
        assert_eq!(ModelTier::Small.calculate_revenue(99), 1);
        // 100 tokens at Small: 100 * 10 / 1000 = 1 (exactly 1, no min needed)
        assert_eq!(ModelTier::Small.calculate_revenue(100), 1);
    }

    #[test]
    fn test_model_tier_revenue_no_overflow() {
        // Very large token count should not overflow (saturating_mul)
        let huge_tokens = u64::MAX / 2;
        // Should not panic — saturating arithmetic
        let revenue = ModelTier::XL.calculate_revenue(huge_tokens);
        assert!(revenue > 0);
    }

    #[test]
    fn test_model_tier_display() {
        assert_eq!(format!("{}", ModelTier::Small), "small");
        assert_eq!(format!("{}", ModelTier::Medium), "medium");
        assert_eq!(format!("{}", ModelTier::Large), "large");
        assert_eq!(format!("{}", ModelTier::XL), "xl");
    }

    #[test]
    fn test_model_tier_from_str() {
        assert_eq!("small".parse::<ModelTier>().unwrap(), ModelTier::Small);
        assert_eq!("s".parse::<ModelTier>().unwrap(), ModelTier::Small);
        assert_eq!("tiny".parse::<ModelTier>().unwrap(), ModelTier::Small);
        assert_eq!("medium".parse::<ModelTier>().unwrap(), ModelTier::Medium);
        assert_eq!("m".parse::<ModelTier>().unwrap(), ModelTier::Medium);
        assert_eq!("large".parse::<ModelTier>().unwrap(), ModelTier::Large);
        assert_eq!("l".parse::<ModelTier>().unwrap(), ModelTier::Large);
        assert_eq!("xl".parse::<ModelTier>().unwrap(), ModelTier::XL);
        assert_eq!("xxl".parse::<ModelTier>().unwrap(), ModelTier::XL);
        assert_eq!("huge".parse::<ModelTier>().unwrap(), ModelTier::XL);

        // Case insensitive
        assert_eq!("SMALL".parse::<ModelTier>().unwrap(), ModelTier::Small);
        assert_eq!("Medium".parse::<ModelTier>().unwrap(), ModelTier::Medium);

        // Invalid
        assert!("invalid".parse::<ModelTier>().is_err());
        assert!("".parse::<ModelTier>().is_err());
    }

    #[test]
    fn test_model_tier_serde_roundtrip() {
        for tier in &[ModelTier::Small, ModelTier::Medium, ModelTier::Large, ModelTier::XL] {
            let json = serde_json::to_string(tier).unwrap();
            let parsed: ModelTier = serde_json::from_str(&json).unwrap();
            assert_eq!(*tier, parsed);
        }
    }

    #[test]
    fn test_record_job_completion_increments_counters() {
        let pool = InferenceWorkerPool::new();

        // Before any completions, everything is zero
        assert_eq!(pool.revenue_earned.load(Ordering::Relaxed), 0);
        assert_eq!(pool.jobs_completed.load(Ordering::Relaxed), 0);

        // Record a medium-tier job: 2000 tokens * 50/1K = 100 micro-QUG
        pool.record_job_completion("job-1", 2000, ModelTier::Medium);

        assert_eq!(pool.revenue_earned.load(Ordering::Relaxed), 100);
        assert_eq!(pool.jobs_completed.load(Ordering::Relaxed), 1);
        assert_eq!(pool.total_tokens.load(Ordering::Relaxed), 2000);

        // Record a second job: large tier, 1000 tokens * 200/1K = 200 micro-QUG
        pool.record_job_completion("job-2", 1000, ModelTier::Large);

        assert_eq!(pool.revenue_earned.load(Ordering::Relaxed), 300); // 100 + 200
        assert_eq!(pool.jobs_completed.load(Ordering::Relaxed), 2);
        assert_eq!(pool.total_tokens.load(Ordering::Relaxed), 3000); // 2000 + 1000
    }

    #[test]
    fn test_record_job_completion_fires_revenue_callback() {
        let mut pool = InferenceWorkerPool::new();

        // Track callback invocations using a shared counter
        let callback_total = Arc::new(AtomicU64::new(0));
        let callback_total_clone = callback_total.clone();

        pool.set_revenue_callback(Arc::new(move |revenue| {
            callback_total_clone.fetch_add(revenue, Ordering::Relaxed);
        }));

        // Record a small-tier job: 1000 tokens * 10/1K = 10 micro-QUG
        pool.record_job_completion("cb-job-1", 1000, ModelTier::Small);
        assert_eq!(callback_total.load(Ordering::Relaxed), 10);

        // Record an XL-tier job: 5000 tokens * 1000/1K = 5000 micro-QUG
        pool.record_job_completion("cb-job-2", 5000, ModelTier::XL);
        assert_eq!(callback_total.load(Ordering::Relaxed), 5010); // 10 + 5000
    }

    #[test]
    fn test_record_job_completion_fires_orchestrator_callback() {
        let mut pool = InferenceWorkerPool::new();

        // Track orchestrator callback
        let orch_revenue = Arc::new(AtomicU64::new(0));
        let orch_tasks = Arc::new(AtomicU64::new(0));
        let orch_revenue_clone = orch_revenue.clone();
        let orch_tasks_clone = orch_tasks.clone();

        pool.set_orchestrator_callback(move |layer, revenue| {
            assert_eq!(layer, ComputeLayer::AiInference);
            orch_revenue_clone.fetch_add(revenue, Ordering::Relaxed);
            orch_tasks_clone.fetch_add(1, Ordering::Relaxed);
        });

        pool.record_job_completion("orch-job-1", 3000, ModelTier::Medium);
        // 3000 * 50 / 1000 = 150 micro-QUG
        assert_eq!(orch_revenue.load(Ordering::Relaxed), 150);
        assert_eq!(orch_tasks.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn test_record_job_completion_both_callbacks() {
        let mut pool = InferenceWorkerPool::new();

        let revenue_cb_total = Arc::new(AtomicU64::new(0));
        let orch_cb_total = Arc::new(AtomicU64::new(0));
        let rc = revenue_cb_total.clone();
        let oc = orch_cb_total.clone();

        pool.set_revenue_callback(Arc::new(move |revenue| {
            rc.fetch_add(revenue, Ordering::Relaxed);
        }));
        pool.set_orchestrator_callback(move |_layer, revenue| {
            oc.fetch_add(revenue, Ordering::Relaxed);
        });

        // Large tier: 4000 tokens * 200/1K = 800 micro-QUG
        pool.record_job_completion("both-job", 4000, ModelTier::Large);

        // Both callbacks should have received the same revenue amount
        assert_eq!(revenue_cb_total.load(Ordering::Relaxed), 800);
        assert_eq!(orch_cb_total.load(Ordering::Relaxed), 800);

        // Internal counter also matches
        assert_eq!(pool.revenue_earned.load(Ordering::Relaxed), 800);
    }

    #[test]
    fn test_record_job_completion_no_callbacks_set() {
        // Should not panic when no callbacks are set
        let pool = InferenceWorkerPool::new();
        pool.record_job_completion("no-cb-job", 500, ModelTier::Small);

        // Internal counters still increment
        assert_eq!(pool.jobs_completed.load(Ordering::Relaxed), 1);
        // 500 * 10 / 1000 = 5, but min is 1: actual = 5
        assert_eq!(pool.revenue_earned.load(Ordering::Relaxed), 5);
        assert_eq!(pool.total_tokens.load(Ordering::Relaxed), 500);
    }

    #[test]
    fn test_get_revenue_summary_empty() {
        let pool = InferenceWorkerPool::new();
        let summary = pool.get_revenue_summary();

        assert_eq!(summary.total_revenue_micro_qug, 0);
        assert_eq!(summary.total_jobs_completed, 0);
        assert_eq!(summary.total_tokens_generated, 0);
        assert!(summary.uptime_seconds >= 0.0);
        // revenue_per_hour should be 0 when no revenue
        assert_eq!(summary.revenue_per_hour, 0.0);
    }

    #[test]
    fn test_get_revenue_summary_after_jobs() {
        let pool = InferenceWorkerPool::new();

        // Record several jobs
        pool.record_job_completion("sum-1", 1000, ModelTier::Small);  // 10 micro-QUG
        pool.record_job_completion("sum-2", 2000, ModelTier::Medium); // 100 micro-QUG
        pool.record_job_completion("sum-3", 1000, ModelTier::Large);  // 200 micro-QUG

        let summary = pool.get_revenue_summary();

        assert_eq!(summary.total_revenue_micro_qug, 310); // 10 + 100 + 200
        assert_eq!(summary.total_jobs_completed, 3);
        assert_eq!(summary.total_tokens_generated, 4000); // 1000 + 2000 + 1000
        assert!(summary.uptime_seconds >= 0.0);
        // revenue_per_hour should be positive (uptime > 0)
        assert!(summary.revenue_per_hour >= 0.0);
    }

    #[test]
    fn test_revenue_summary_serde_roundtrip() {
        let summary = RevenueSummary {
            total_revenue_micro_qug: 42000,
            total_jobs_completed: 15,
            revenue_per_hour: 3600.0,
            total_tokens_generated: 100000,
            uptime_seconds: 42.0,
        };
        let json = serde_json::to_string(&summary).unwrap();
        let parsed: RevenueSummary = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.total_revenue_micro_qug, 42000);
        assert_eq!(parsed.total_jobs_completed, 15);
        assert_eq!(parsed.total_tokens_generated, 100000);
    }

    #[test]
    fn test_stats_reflects_record_job_completion() {
        // Verify that record_job_completion updates the stats() output too,
        // since both share the same Arc<AtomicU64> counters.
        let pool = InferenceWorkerPool::new();

        let stats_before = pool.stats();
        assert_eq!(stats_before.revenue_earned_micro_qug, 0);
        assert_eq!(stats_before.total_tokens_generated, 0);

        pool.record_job_completion("stats-job", 5000, ModelTier::XL);
        // 5000 * 1000 / 1000 = 5000 micro-QUG

        let stats_after = pool.stats();
        assert_eq!(stats_after.revenue_earned_micro_qug, 5000);
        assert_eq!(stats_after.total_tokens_generated, 5000);
    }
}
