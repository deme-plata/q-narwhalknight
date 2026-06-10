//! Pipeline trait definitions for the Agent Activity Panel.
//!
//! Steals the architectural pattern from xAI's Home Mixer (Apache-2.0,
//! `github.com/xai-org/x-algorithm`). Six trait-types compose into any
//! decision-streaming workload. Quillon Graph's "what to show in the panel"
//! is precisely this pattern.
//!
//! The same pattern fits other decision systems on chain (validator-anchor
//! election, mempool prioritisation, LP recommendation, DEX route-finding) —
//! treat the trait set as a general-purpose decision pipeline, not panel-
//! specific. Future modules MAY reuse it.

use async_trait::async_trait;
use parking_lot::RwLock;
use serde::Serialize;
use std::collections::HashMap;
use std::sync::Arc;

use crate::AppState;
use q_types::Address;

// ════════════════════════════════════════════════════════════════════════════
// Context passed through the pipeline
// ════════════════════════════════════════════════════════════════════════════

/// The wallet-scoped context every pipeline stage operates against.
///
/// Constructed once per request to `GET /api/v1/agent/panel/{addr}`,
/// passed by reference to each stage. Allows stages to read shared chain
/// state (mempool, recent blocks, DEX log) without re-fetching.
#[derive(Clone)]
pub struct PanelContext {
    /// The wallet this panel is rendering. All stages filter to tasks
    /// originated by or addressed to this wallet.
    pub wallet: Address,
    /// Whether the viewer is the wallet owner (full view) or an embed
    /// viewer (`AgentActivityPanel mode="embed"`) seeing read-only state.
    /// Filters / selectors may behave differently based on this flag.
    pub viewer_mode: ViewerMode,
    /// Shared application state for chain reads (mempool, RocksDB queries, etc.).
    pub state: Arc<AppState>,
    /// Server-side timestamp at which this pipeline run started. Used for
    /// "age" computations in TaskRow rendering and for cache-key construction.
    pub now: chrono::DateTime<chrono::Utc>,
    /// v10.10.10: query-level data populated by `QueryHydrator` stages BEFORE
    /// any Source runs. Subsequent stages read via `query_data(key)` instead
    /// of refetching per-candidate. Example keys:
    ///   - `"wallet_history"` — last N confirmed txs for this wallet
    ///   - `"wallet_balance"` — current balance snapshot
    ///   - `"wallet_block_list"` — set of wallet addresses this viewer has blocked
    ///   - `"recent_swaps"` — recent DEX swap rows touching this wallet
    /// Wrapped in `Arc<RwLock<...>>` so per-stage writes don't conflict; the
    /// pipeline serializes QueryHydrator writes so contention is rare in practice.
    pub query_data: Arc<RwLock<HashMap<&'static str, serde_json::Value>>>,
}

impl PanelContext {
    /// Construct a context with an empty query_data map. Use this in handlers
    /// + tests; the Pipeline populates `query_data` during `run`.
    pub fn new(
        wallet: Address,
        viewer_mode: ViewerMode,
        state: Arc<AppState>,
    ) -> Self {
        Self {
            wallet,
            viewer_mode,
            state,
            now: chrono::Utc::now(),
            query_data: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Read a value previously stored by a `QueryHydrator`. Returns a clone so
    /// downstream stages can keep the value while the lock is released.
    pub fn query_data(&self, key: &'static str) -> Option<serde_json::Value> {
        self.query_data.read().get(key).cloned()
    }

    /// Internal — used by Pipeline::run to populate the map after each
    /// QueryHydrator returns.
    pub(crate) fn set_query_data(&self, key: &'static str, value: serde_json::Value) {
        self.query_data.write().insert(key, value);
    }
}

/// Distinguishes "I'm the wallet owner viewing my own panel" from "I'm
/// embedding someone else's panel in my dashboard". Drives the trust-tier
/// indicator (`Observed` chip) per `docs/agent-activity-panel-spec.md` §2.4.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ViewerMode {
    /// Owner viewing their own panel. Full data, approval affordances shown.
    Owner,
    /// Read-only embed mode. Approval affordances hidden, trust-tier chip
    /// shows 🟠 Observed instead of the real tier.
    Embed,
}

// ════════════════════════════════════════════════════════════════════════════
// The six pipeline trait-types
// ════════════════════════════════════════════════════════════════════════════

/// **Source**: produces candidates from underlying state.
///
/// Examples (concrete impls land as Codex builds out L1-L4 of the spec):
/// - `MempoolTxSource` — reads `state.tx_pool` (DashMap) for the wallet
/// - `MinerSolutionSource` — reads mining-handler in-memory state
/// - `DexExecutionSource` — reads DEX swap log
/// - `TwitterDraftSource` — polls quillon-twitter-mcp SQLite drafts DB
/// - `PendingApprovalSource` — drafts DB filtered `status=PENDING`
/// - `BridgeIntentSource` — `CF_BTC_LP_INTENT` column family
/// - `ConfirmedTxSource` — recent_txs RocksDB query
/// - `TwitterPostSource` — `CF_TWITTER_ATTESTATIONS` column family
#[async_trait]
pub trait Source: Send + Sync {
    type Candidate: Send + Sync;
    async fn fetch(&self, ctx: &PanelContext) -> Vec<Self::Candidate>;
}

/// **Hydrator**: enriches each candidate with additional context.
///
/// Examples:
/// - `TokenMetadataHydrator` — adds token symbol/decimals to DEX-swap tasks
/// - `BlockReferenceHydrator` — adds block height + timestamp to confirmed-tx tasks
/// - `AttestationHydrator` — tags Twitter posts that have verified attestation
/// - `ApprovalUrlHydrator` — adds the `quillon.xyz/admin/twitter/q/<id>` URL to pending drafts
///
/// v10.10.10: per-stage `enable()` mirrors xAI's `candidate-pipeline/hydrator.rs:17-19`
/// — runtime gating for A/B testing without rebuilding. Default `true` keeps
/// existing impls compiling. `name()` defaults to type name for tracing spans.
#[async_trait]
pub trait Hydrator<C: Send + Sync>: Send + Sync {
    async fn enrich(&self, candidate: &mut C, ctx: &PanelContext);

    /// Whether this hydrator should run for the given context. Stages where
    /// `enable` returns false are skipped without invoking `enrich`.
    fn enable(&self, _ctx: &PanelContext) -> bool {
        true
    }

    /// Short stage name used in tracing spans + length-mismatch warnings.
    fn name(&self) -> &'static str {
        std::any::type_name::<Self>()
            .rsplit("::")
            .next()
            .unwrap_or("Hydrator")
    }
}

/// **QueryHydrator**: pre-fetches viewer-level context ONCE per pipeline run.
///
/// Mirrors `candidate-pipeline/query_hydrator.rs`. Runs BEFORE any Source — the
/// idea is "load the viewer's history/balance/AFL-1 attestations once, share
/// them across every candidate enrichment". Eliminates redundant per-candidate
/// fetches and turns N×M hydration into 1+N.
///
/// Returns a `serde_json::Value` that gets stored on the context's hot-loaded
/// map. Each QueryHydrator picks its own key. Downstream Hydrators/Scorers
/// read via `ctx.query_data(key)`.
#[async_trait]
pub trait QueryHydrator: Send + Sync {
    /// Unique key the hydrator stores its output under. Downstream stages
    /// look it up via `ctx.query_data(key)`. Convention: snake_case crate path.
    fn key(&self) -> &'static str;

    /// Compute the query-level data. Returns `None` to skip (treated as
    /// "data unavailable" by downstream — they fall back to per-candidate
    /// hydration if needed).
    async fn hydrate(&self, ctx: &PanelContext) -> Option<serde_json::Value>;

    fn enable(&self, _ctx: &PanelContext) -> bool {
        true
    }

    fn name(&self) -> &'static str {
        std::any::type_name::<Self>()
            .rsplit("::")
            .next()
            .unwrap_or("QueryHydrator")
    }
}

/// **Filter**: removes candidates that should not appear.
///
/// Examples:
/// - `AgeFilter` — drop tasks older than 5 minutes (NOW zone) or 24 h (DONE zone)
/// - `NotExpiredFilter` — drop QUEUED tasks whose approval window has expired
/// - `VisibilityFilter` — drop tasks marked private when viewer_mode = Embed
/// - `PreviouslySeenFilter` (v10.10.10) — drop tasks the viewer has already seen
pub trait Filter<C>: Send + Sync {
    fn keep(&self, candidate: &C, ctx: &PanelContext) -> bool;

    fn enable(&self, _ctx: &PanelContext) -> bool {
        true
    }

    fn name(&self) -> &'static str {
        std::any::type_name::<Self>()
            .rsplit("::")
            .next()
            .unwrap_or("Filter")
    }
}

/// **Scorer**: assigns a relevance score to each candidate. Higher = more relevant.
///
/// Examples:
/// - `RecencyScorer` — score by inverse age
/// - `TimestampScorer` (desc) — most-recent first for DONE zone
#[async_trait]
pub trait Scorer<C: Send + Sync>: Send + Sync {
    async fn score(&self, candidate: &C, ctx: &PanelContext) -> f64;

    fn enable(&self, _ctx: &PanelContext) -> bool {
        true
    }

    fn name(&self) -> &'static str {
        std::any::type_name::<Self>()
            .rsplit("::")
            .next()
            .unwrap_or("Scorer")
    }
}

/// **Selector**: picks the final ordered output from the scored candidate set.
///
/// Examples:
/// - `TopK { k: 10 }` — sort by score desc, take top K
/// - `FifoSelector` — preserve original order (for QUEUED zone where oldest-pending shows first)
/// - `ByStatusSelector` — group by task status, then by score within group
pub trait Selector<C>: Send + Sync {
    fn select(&self, candidates: Vec<(C, f64)>) -> Vec<C>;
}

/// **SideEffect**: fire-and-forget work triggered by a pipeline run.
///
/// Examples:
/// - `MetricEmitter` — emit panel-render metrics to Prometheus
/// - `AccessLogger` — log which wallet viewed which panel (privacy-respecting)
/// - `WarmCacheUpdater` — pre-warm caches for the wallet's likely next request
#[async_trait]
pub trait SideEffect: Send + Sync {
    async fn run(&self, ctx: &PanelContext);
}

// ════════════════════════════════════════════════════════════════════════════
// The Pipeline assembler — composes the six trait types into a runnable unit
// ════════════════════════════════════════════════════════════════════════════

/// Builds a single-candidate-type pipeline. Codex extends as needed (e.g.
/// `MultiSourcePipeline` for the panel where multiple Sources contribute to
/// one Candidate type, or a generic `Pipeline<C, D>` for type-converting stages).
///
/// v10.10.10: extended from 6 stages to 10 to match `xai-org/x-algorithm`'s
/// `candidate-pipeline/candidate_pipeline.rs:22-33`. The four added stages are:
///   - **`query_hydrators`** — pre-fetch viewer-level data ONCE before Sources run
///   - **`dependent_query_hydrators`** — second pass with first-pass results visible
///   - **`post_selection_hydrators`** — only enrich the FINAL top-K (saves work)
///   - **`post_selection_filters`** — last-mile drops after selection
pub struct Pipeline<C: Send + Sync> {
    query_hydrators: Vec<Box<dyn QueryHydrator>>,
    dependent_query_hydrators: Vec<Box<dyn QueryHydrator>>,
    sources: Vec<Box<dyn Source<Candidate = C>>>,
    hydrators: Vec<Box<dyn Hydrator<C>>>,
    filters: Vec<Box<dyn Filter<C>>>,
    scorers: Vec<Box<dyn Scorer<C>>>,
    selector: Option<Box<dyn Selector<C>>>,
    post_selection_hydrators: Vec<Box<dyn Hydrator<C>>>,
    post_selection_filters: Vec<Box<dyn Filter<C>>>,
    side_effects: Vec<Box<dyn SideEffect>>,
}

impl<C: Send + Sync + 'static> Pipeline<C> {
    pub fn new() -> Self {
        Self {
            query_hydrators: Vec::new(),
            dependent_query_hydrators: Vec::new(),
            sources: Vec::new(),
            hydrators: Vec::new(),
            filters: Vec::new(),
            scorers: Vec::new(),
            selector: None,
            post_selection_hydrators: Vec::new(),
            post_selection_filters: Vec::new(),
            side_effects: Vec::new(),
        }
    }

    pub fn query_hydrator(mut self, qh: impl QueryHydrator + 'static) -> Self {
        self.query_hydrators.push(Box::new(qh));
        self
    }

    pub fn dependent_query_hydrator(mut self, qh: impl QueryHydrator + 'static) -> Self {
        self.dependent_query_hydrators.push(Box::new(qh));
        self
    }

    pub fn source(mut self, s: impl Source<Candidate = C> + 'static) -> Self {
        self.sources.push(Box::new(s));
        self
    }

    pub fn hydrator(mut self, h: impl Hydrator<C> + 'static) -> Self {
        self.hydrators.push(Box::new(h));
        self
    }

    pub fn filter(mut self, f: impl Filter<C> + 'static) -> Self {
        self.filters.push(Box::new(f));
        self
    }

    pub fn scorer(mut self, s: impl Scorer<C> + 'static) -> Self {
        self.scorers.push(Box::new(s));
        self
    }

    pub fn selector(mut self, s: impl Selector<C> + 'static) -> Self {
        self.selector = Some(Box::new(s));
        self
    }

    pub fn post_selection_hydrator(mut self, h: impl Hydrator<C> + 'static) -> Self {
        self.post_selection_hydrators.push(Box::new(h));
        self
    }

    pub fn post_selection_filter(mut self, f: impl Filter<C> + 'static) -> Self {
        self.post_selection_filters.push(Box::new(f));
        self
    }

    pub fn side_effect(mut self, se: impl SideEffect + 'static) -> Self {
        self.side_effects.push(Box::new(se));
        self
    }

    /// Run the pipeline: ten stages, matching xAI's `candidate-pipeline`.
    ///
    /// 1. QueryHydrator           (parallel) — pre-fetch viewer-level data
    /// 2. DependentQueryHydrator  (parallel) — second pass with deps
    /// 3. Source                  (parallel) — fetch candidates
    /// 4. Hydrator                (per-candidate, sequential) — enrich all candidates
    /// 5. Filter                  (sync, fast) — drop irrelevant
    /// 6. Scorer                  (per-candidate, sequential) — compute scores
    /// 7. Selector                                           — pick top-K
    /// 8. PostSelectionHydrator   (per-candidate, sequential) — enrich ONLY survivors
    /// 9. PostSelectionFilter     (sync, fast) — last-mile drops
    /// 10. SideEffect             (fire-and-forget) — emit metrics/SSE/etc.
    ///
    /// Disabled stages (per-stage `enable(ctx) == false`) are skipped.
    /// `tracing::info!` spans bound each phase for observability.
    pub async fn run(self, ctx: &PanelContext) -> Vec<C> {
        let pipeline_start = std::time::Instant::now();

        // ── Phase 1: QueryHydrator (parallel) ────────────────────────────
        let qh_start = std::time::Instant::now();
        let enabled_qh: Vec<_> = self.query_hydrators.iter().filter(|h| h.enable(ctx)).collect();
        let qh_futures: Vec<_> = enabled_qh.iter().map(|h| async move {
            let key = h.key();
            let v = h.hydrate(ctx).await;
            (key, v)
        }).collect();
        for (key, maybe_value) in futures::future::join_all(qh_futures).await {
            if let Some(v) = maybe_value {
                ctx.set_query_data(key, v);
            }
        }
        tracing::debug!(
            stage = "query_hydrators",
            count = enabled_qh.len(),
            elapsed_ms = qh_start.elapsed().as_millis() as u64,
            "panel pipeline stage",
        );

        // ── Phase 2: DependentQueryHydrator (parallel, sees Phase 1 output) ──
        let dqh_start = std::time::Instant::now();
        let enabled_dqh: Vec<_> = self.dependent_query_hydrators.iter().filter(|h| h.enable(ctx)).collect();
        let dqh_futures: Vec<_> = enabled_dqh.iter().map(|h| async move {
            let key = h.key();
            let v = h.hydrate(ctx).await;
            (key, v)
        }).collect();
        for (key, maybe_value) in futures::future::join_all(dqh_futures).await {
            if let Some(v) = maybe_value {
                ctx.set_query_data(key, v);
            }
        }
        if !enabled_dqh.is_empty() {
            tracing::debug!(
                stage = "dependent_query_hydrators",
                count = enabled_dqh.len(),
                elapsed_ms = dqh_start.elapsed().as_millis() as u64,
                "panel pipeline stage",
            );
        }

        // ── Phase 3: Source fan-out (parallel) ───────────────────────────
        let source_start = std::time::Instant::now();
        let fetch_futures: Vec<_> = self.sources.iter().map(|s| s.fetch(ctx)).collect();
        let mut candidates: Vec<C> = futures::future::join_all(fetch_futures)
            .await
            .into_iter()
            .flatten()
            .collect();
        let source_count = candidates.len();
        tracing::debug!(
            stage = "source",
            count = self.sources.len(),
            candidates = source_count,
            elapsed_ms = source_start.elapsed().as_millis() as u64,
            "panel pipeline stage",
        );

        // ── Phase 4: Hydrator (per-candidate, sequential per hydrator) ──
        let hydration_start = std::time::Instant::now();
        for hydrator in &self.hydrators {
            if !hydrator.enable(ctx) {
                continue;
            }
            for c in candidates.iter_mut() {
                hydrator.enrich(c, ctx).await;
            }
        }
        tracing::debug!(
            stage = "hydrator",
            count = self.hydrators.iter().filter(|h| h.enable(ctx)).count(),
            elapsed_ms = hydration_start.elapsed().as_millis() as u64,
            "panel pipeline stage",
        );

        // ── Phase 5: Filter (sync, fast) ─────────────────────────────────
        let filter_start = std::time::Instant::now();
        let before_filter = candidates.len();
        let enabled_filters: Vec<&Box<dyn Filter<C>>> =
            self.filters.iter().filter(|f| f.enable(ctx)).collect();
        candidates.retain(|c| enabled_filters.iter().all(|f| f.keep(c, ctx)));
        tracing::debug!(
            stage = "filter",
            count = enabled_filters.len(),
            before = before_filter,
            after = candidates.len(),
            elapsed_ms = filter_start.elapsed().as_millis() as u64,
            "panel pipeline stage",
        );

        // ── Phase 6: Score (sum across all scorers) ──────────────────────
        let score_start = std::time::Instant::now();
        let scored: Vec<(C, f64)> = {
            let mut out = Vec::with_capacity(candidates.len());
            for c in candidates {
                let mut total = 0.0;
                for scorer in &self.scorers {
                    if !scorer.enable(ctx) {
                        continue;
                    }
                    total += scorer.score(&c, ctx).await;
                }
                out.push((c, total));
            }
            out
        };
        tracing::debug!(
            stage = "scorer",
            count = self.scorers.iter().filter(|s| s.enable(ctx)).count(),
            candidates = scored.len(),
            elapsed_ms = score_start.elapsed().as_millis() as u64,
            "panel pipeline stage",
        );

        // ── Phase 7: Select (pick final top-K) ───────────────────────────
        let select_start = std::time::Instant::now();
        let mut final_list = match &self.selector {
            Some(sel) => sel.select(scored),
            None => scored.into_iter().map(|(c, _)| c).collect(),
        };
        tracing::debug!(
            stage = "selector",
            output = final_list.len(),
            elapsed_ms = select_start.elapsed().as_millis() as u64,
            "panel pipeline stage",
        );

        // ── Phase 8: PostSelectionHydrator (only enrich survivors) ──────
        // This is the big perf win: hydrators like TokenMetadataHydrator are
        // expensive and we don't need them for the 4950 candidates that got
        // dropped during selection.
        let psh_start = std::time::Instant::now();
        for hydrator in &self.post_selection_hydrators {
            if !hydrator.enable(ctx) {
                continue;
            }
            for c in final_list.iter_mut() {
                hydrator.enrich(c, ctx).await;
            }
        }
        if !self.post_selection_hydrators.is_empty() {
            tracing::debug!(
                stage = "post_selection_hydrator",
                count = self.post_selection_hydrators.iter().filter(|h| h.enable(ctx)).count(),
                candidates = final_list.len(),
                elapsed_ms = psh_start.elapsed().as_millis() as u64,
                "panel pipeline stage",
            );
        }

        // ── Phase 9: PostSelectionFilter (last-mile drops) ──────────────
        let psf_start = std::time::Instant::now();
        let before_psf = final_list.len();
        let enabled_psf: Vec<&Box<dyn Filter<C>>> =
            self.post_selection_filters.iter().filter(|f| f.enable(ctx)).collect();
        final_list.retain(|c| enabled_psf.iter().all(|f| f.keep(c, ctx)));
        if !self.post_selection_filters.is_empty() {
            tracing::debug!(
                stage = "post_selection_filter",
                count = enabled_psf.len(),
                before = before_psf,
                after = final_list.len(),
                elapsed_ms = psf_start.elapsed().as_millis() as u64,
                "panel pipeline stage",
            );
        }

        // ── Phase 10: SideEffects (fire-and-forget) ─────────────────────
        for se in self.side_effects {
            let ctx_clone = ctx.clone();
            tokio::spawn(async move {
                se.run(&ctx_clone).await;
            });
        }

        tracing::info!(
            wallet_prefix = %hex::encode(&ctx.wallet[..4]),
            sources = source_count,
            final_count = final_list.len(),
            total_ms = pipeline_start.elapsed().as_millis() as u64,
            "panel pipeline run completed",
        );

        final_list
    }
}

impl<C: Send + Sync + 'static> Default for Pipeline<C> {
    fn default() -> Self {
        Self::new()
    }
}

// ════════════════════════════════════════════════════════════════════════════
// Reusable concrete impls — the simple ones that fit in this file. More
// complex ones (per-source impls hitting RocksDB, ML scorers, etc.) get
// their own files under `sources/`, `scorers/`, etc.
// ════════════════════════════════════════════════════════════════════════════

/// Top-K selector — sort by score descending, take first K.
pub struct TopK {
    pub k: usize,
}

impl<C: Send + Sync> Selector<C> for TopK {
    fn select(&self, mut scored: Vec<(C, f64)>) -> Vec<C> {
        // Stable sort by score desc; equal-score items keep relative order
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.into_iter().take(self.k).map(|(c, _)| c).collect()
    }
}

/// FIFO selector — preserve the order candidates were emitted in. Used for
/// the QUEUED zone where oldest-pending should show first (FIFO awareness).
pub struct FifoSelector;

impl<C: Send + Sync> Selector<C> for FifoSelector {
    fn select(&self, scored: Vec<(C, f64)>) -> Vec<C> {
        scored.into_iter().map(|(c, _)| c).collect()
    }
}

/// v10.10.10: Diversity-aware selector — picks the highest-scoring candidate
/// then applies diminishing returns to subsequent candidates that share a
/// "diversity key" (extracted via a closure). Prevents the panel from
/// degenerating into "10 txs to the same recipient" at high throughput.
///
/// Algorithm: greedy pick, with a per-key counter. After picking a candidate
/// with key K, every other candidate with key K has its score multiplied by
/// `lambda^count` (lambda < 1). Continue until K candidates picked.
///
/// Inspired by xAI's `author_diversity_scorer` (we couldn't read its source
/// directly because xai_decider/xai_feature_switches/xai_stats_receiver are
/// internal-only crates, but the upstream README + spec confirm the pattern).
pub struct DiversityTopK<C: Send + Sync, F: Fn(&C) -> String + Send + Sync> {
    pub k: usize,
    /// Penalty per repeat occurrence. 0.5 = each duplicate halves its score.
    /// Typical range 0.3-0.7. Lower = more diversity, higher = closer to TopK.
    pub lambda: f64,
    /// Closure extracting the diversity key from a candidate.
    pub key: F,
    /// Phantom marker — `C` only appears in the `F` Fn bound, which Rust
    /// considers an unused type parameter on the struct itself. Carrying a
    /// zero-sized `PhantomData<fn(&C)>` makes the parameter "used" while
    /// keeping the struct `Send + Sync` (we don't actually own a `C`).
    pub _phantom: std::marker::PhantomData<fn(&C)>,
}

impl<C: Send + Sync, F: Fn(&C) -> String + Send + Sync> Selector<C> for DiversityTopK<C, F> {
    fn select(&self, mut scored: Vec<(C, f64)>) -> Vec<C> {
        if self.k == 0 || scored.is_empty() {
            return Vec::new();
        }
        // Sort descending by score for the initial pass
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let mut picked: Vec<C> = Vec::with_capacity(self.k);
        let mut key_count: HashMap<String, u32> = HashMap::new();

        // Greedy with diminishing returns
        while picked.len() < self.k && !scored.is_empty() {
            // Find the highest currently-scored candidate after applying penalties
            let mut best_idx: Option<usize> = None;
            let mut best_adjusted: f64 = f64::NEG_INFINITY;
            for (i, (c, s)) in scored.iter().enumerate() {
                let k = (self.key)(c);
                let prior = *key_count.get(&k).unwrap_or(&0);
                let adjusted = s * self.lambda.powi(prior as i32);
                if adjusted > best_adjusted {
                    best_adjusted = adjusted;
                    best_idx = Some(i);
                }
            }
            if let Some(i) = best_idx {
                let (c, _) = scored.swap_remove(i);
                let k = (self.key)(&c);
                *key_count.entry(k).or_insert(0) += 1;
                picked.push(c);
            } else {
                break;
            }
        }

        picked
    }
}

// ════════════════════════════════════════════════════════════════════════════
// Public response types — what the REST endpoint serialises
// ════════════════════════════════════════════════════════════════════════════

/// The three-zone JSON snapshot served by `GET /api/v1/agent/panel/{addr}`.
///
/// Codex: concrete `Task` type lives in `sources` or a sibling module once
/// the per-source impls land. The shape here is intentionally minimal so
/// the frontend rendering contract is stable from day 1.
#[derive(Serialize, Debug, Clone)]
pub struct PanelSnapshot<T> {
    pub wallet: String,
    pub as_of: chrono::DateTime<chrono::Utc>,
    pub now: Vec<T>,
    pub queued: Vec<T>,
    pub done: Vec<T>,
    pub health: PanelHealth,
}

/// Chain-health summary at the top of the panel ("chain height N · last block X · M peers").
#[derive(Serialize, Debug, Clone)]
pub struct PanelHealth {
    pub current_height: u64,
    pub last_block_age_secs: u64,
    pub peer_count: u32,
    pub alive: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Simplest possible candidate type to verify the traits compose.
    #[derive(Debug, Clone, PartialEq)]
    struct DummyCandidate(u32);

    struct DummySource(Vec<u32>);

    #[async_trait]
    impl Source for DummySource {
        type Candidate = DummyCandidate;
        async fn fetch(&self, _ctx: &PanelContext) -> Vec<DummyCandidate> {
            self.0.iter().copied().map(DummyCandidate).collect()
        }
    }

    struct EvenOnlyFilter;
    impl Filter<DummyCandidate> for EvenOnlyFilter {
        fn keep(&self, c: &DummyCandidate, _ctx: &PanelContext) -> bool {
            c.0 % 2 == 0
        }
    }

    struct ValueScorer;
    #[async_trait]
    impl Scorer<DummyCandidate> for ValueScorer {
        async fn score(&self, c: &DummyCandidate, _ctx: &PanelContext) -> f64 {
            c.0 as f64
        }
    }

    // Note: a full integration test that actually runs the pipeline against
    // a constructed PanelContext requires a real AppState, which is heavyweight.
    // Codex: add such a test as part of the panel-handler PR once AppState
    // can be built more cheaply in tests, or use a `cfg(test)` mock.

    #[test]
    fn topk_selector_sorts_desc() {
        let sel = TopK { k: 3 };
        let input = vec![
            (DummyCandidate(1), 1.0),
            (DummyCandidate(3), 3.0),
            (DummyCandidate(2), 2.0),
            (DummyCandidate(4), 4.0),
        ];
        let output: Vec<DummyCandidate> = sel.select(input);
        assert_eq!(
            output,
            vec![DummyCandidate(4), DummyCandidate(3), DummyCandidate(2)]
        );
    }

    #[test]
    fn fifo_selector_preserves_order() {
        let sel = FifoSelector;
        let input = vec![
            (DummyCandidate(3), 99.0),
            (DummyCandidate(1), 1.0),
            (DummyCandidate(2), 50.0),
        ];
        let output: Vec<DummyCandidate> = sel.select(input);
        assert_eq!(
            output,
            vec![DummyCandidate(3), DummyCandidate(1), DummyCandidate(2)]
        );
    }
}
