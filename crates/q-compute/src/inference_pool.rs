//! Inference Worker Pool — Runs AI inference on idle cores
//!
//! v9.6.0: Bridges the Compute Orchestrator with q-ai-inference engines.
//! When the orchestrator assigns cores to the AiInference layer, this pool
//! accepts inference tasks from gossipsub, local API, or tunnel mesh.
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
            cumulative_tps: Arc::new(RwLock::new(0.0)),
            task_queue: Arc::new(RwLock::new(VecDeque::new())),
            accepting: Arc::new(AtomicBool::new(false)),
            running: Arc::new(AtomicBool::new(false)),
            model_name: Arc::new(RwLock::new("none".to_string())),
            started_ms: now,
            max_concurrent: Arc::new(AtomicU64::new(0)),
            orchestrator_record: None,
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
}
