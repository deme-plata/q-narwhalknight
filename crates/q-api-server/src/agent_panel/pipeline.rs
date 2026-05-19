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
use serde::Serialize;
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
#[async_trait]
pub trait Hydrator<C: Send + Sync>: Send + Sync {
    async fn enrich(&self, candidate: &mut C, ctx: &PanelContext);
}

/// **Filter**: removes candidates that should not appear.
///
/// Examples:
/// - `AgeFilter` — drop tasks older than 5 minutes (NOW zone) or 24 h (DONE zone)
/// - `NotExpiredFilter` — drop QUEUED tasks whose approval window has expired
/// - `VisibilityFilter` — drop tasks marked private when viewer_mode = Embed
pub trait Filter<C>: Send + Sync {
    fn keep(&self, candidate: &C, ctx: &PanelContext) -> bool;
}

/// **Scorer**: assigns a relevance score to each candidate. Higher = more relevant.
///
/// Examples:
/// - `RecencyScorer` — score by inverse age
/// - `TimestampScorer` (desc) — most-recent first for DONE zone
#[async_trait]
pub trait Scorer<C: Send + Sync>: Send + Sync {
    async fn score(&self, candidate: &C, ctx: &PanelContext) -> f64;
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
pub struct Pipeline<C: Send + Sync> {
    sources: Vec<Box<dyn Source<Candidate = C>>>,
    hydrators: Vec<Box<dyn Hydrator<C>>>,
    filters: Vec<Box<dyn Filter<C>>>,
    scorers: Vec<Box<dyn Scorer<C>>>,
    selector: Option<Box<dyn Selector<C>>>,
    side_effects: Vec<Box<dyn SideEffect>>,
}

impl<C: Send + Sync + 'static> Pipeline<C> {
    pub fn new() -> Self {
        Self {
            sources: Vec::new(),
            hydrators: Vec::new(),
            filters: Vec::new(),
            scorers: Vec::new(),
            selector: None,
            side_effects: Vec::new(),
        }
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

    pub fn side_effect(mut self, se: impl SideEffect + 'static) -> Self {
        self.side_effects.push(Box::new(se));
        self
    }

    /// Run the pipeline: fan out all sources in parallel, hydrate, filter,
    /// score (sum of all scorer outputs), select. SideEffects fire concurrently
    /// after select; the run does not wait for them.
    pub async fn run(self, ctx: &PanelContext) -> Vec<C> {
        // Phase 1: Source fan-out (parallel)
        let fetch_futures: Vec<_> = self.sources.iter().map(|s| s.fetch(ctx)).collect();
        let mut candidates: Vec<C> = futures::future::join_all(fetch_futures)
            .await
            .into_iter()
            .flatten()
            .collect();

        // Phase 2: Hydration (sequential per candidate, but each candidate's
        // hydrators run in order; in v1 we keep it simple)
        for hydrator in &self.hydrators {
            for c in candidates.iter_mut() {
                hydrator.enrich(c, ctx).await;
            }
        }

        // Phase 3: Filter (sync, fast)
        candidates.retain(|c| self.filters.iter().all(|f| f.keep(c, ctx)));

        // Phase 4: Score (sum across all scorers)
        let scored: Vec<(C, f64)> = {
            let mut out = Vec::with_capacity(candidates.len());
            for c in candidates {
                let mut total = 0.0;
                for scorer in &self.scorers {
                    total += scorer.score(&c, ctx).await;
                }
                out.push((c, total));
            }
            out
        };

        // Phase 5: Select
        let final_list = match &self.selector {
            Some(sel) => sel.select(scored),
            None => scored.into_iter().map(|(c, _)| c).collect(),
        };

        // Phase 6: SideEffects (fire-and-forget; spawn so we don't await)
        for se in self.side_effects {
            let ctx_clone = ctx.clone();
            tokio::spawn(async move {
                se.run(&ctx_clone).await;
            });
        }

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
