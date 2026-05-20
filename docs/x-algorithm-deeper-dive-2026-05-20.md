# Deeper dive on the X-algorithm work — 2026-05-20

**Author**: Claude Opus 4.7 (research note, written during agent session 780d6789)
**Audience**: future-me + reviewers picking up the agent-panel / x-algorithm-scorer work
**Prompted by**: user question — *what more can we learn from X, what can we enhance from what we've learned, and what don't we know we don't know?*

This doc complements `docs/technical-reviews/TR-2026-004-x-algorithm-inspiration.md`. TR-2026-004 explained the **shape** we adopted (the six-trait Home Mixer pipeline); this doc maps the **gaps** between where we are and what the full X algorithm offers, and asks the meta-questions we've been avoiding.

---

## What we have today (verified 2026-05-20)

Two stacks, orthogonal:

**1. `tools/quillon-twitter-mcp/crates/x-algorithm-scorer/`** — for **tweet drafts**, not chain events.
- Pure-Rust REST sidecar on `:8090`. Wire format (`ScoreRequest` / `ScoreResponse` with 7 per-action probabilities + variant suggestions) is **stable across Layer-1 → Layer-2** transitions — the intent is to swap the brain without rewriting the contract.
- Layer 1 is **hand-engineered heuristics**: text-length sweet-spot, question-mark boost, all-caps risk, inflammatory-keyword list. 5 unit tests pass.
- Layer 2 is **deferred** — meant to wrap xAI's open-sourced Phoenix ML engine over HTTP (or eventually candle-port the model weights). Today the sidecar runs, but nothing in the MCP or q-api-server calls it.

**2. `crates/q-api-server/src/agent_panel/`** — for **chain events** (txs, swaps).
- 6-trait Home Mixer pattern: `Source / Hydrator / Filter / Scorer / Selector / SideEffect`. `Pipeline::run` is real (parallel sources → sequential hydrators → filters → scorers → selector → side-effect spawn at `pipeline.rs:192-240`).
- **Real**: `MempoolTxSource` (reads `state.tx_pool`), `AgeFilter`, `EmbedVisibilityFilter`, `RecencyScorer`, `TrustTierScorer` (hardcoded tier→score), `StatusPriorityScorer` (hardcoded status→score), `TopK`, `FifoSelector`, `AccessLogger`, plus `TxScorer` + `SwapScorer` chains from PR #101/#102.
- **Stub**: `ConfirmedTxSource` returns empty (`concrete.rs:134-155`), `DexSwapSource` returns empty (`concrete.rs:159-169`), `SseEventEmitter` only logs (`concrete.rs:282-305`).
- **Missing entirely**: All Hydrators. The spec at `docs/agent-activity-panel-spec.md:162-178` names `TokenMetadataHydrator`, `BlockReferenceHydrator`, `ApprovalUrlHydrator`, `AttestationHydrator` — none exist.

The two stacks don't share code: x-algorithm-scorer scores text for engagement; agent_panel scores chain candidates for relevance. Same shape, different domain.

---

## §1 — What more can we learn from X (features we know about but haven't ported)

The 2023 open-source release of `github.com/twitter/the-algorithm` exposed dozens of components. The xAI 2026-05-15 re-release at `github.com/xai-org/x-algorithm` is similar in shape. TR-2026-004 explicitly declined to port some of them (SimClusters, graph-jet, follow-graph PageRank, twin-bert, heavy/light rankers, trust-graph propagation) on the grounds of *"adopt the pipeline shape, not the learning component."*

The decisions in TR-2026-004 are correct for v1. But some of the features we declined are worth re-evaluating now that we have a working pipeline. Concrete candidates:

### 1.1 Negative-feedback weighting

Twitter's heavy ranker weights negatives (block, mute, report) much heavier than positives because the cost of a negative event is asymmetric — losing a user costs more than gaining one engagement. On the x-algorithm-scorer side we have this already (`negative_signal_risk` threshold at `tools/quillon-twitter-mcp/crates/x-algorithm-scorer/src/main.rs:181`).

We don't have it on the chain side. What would a negative-signal scorer look like for chain events?
- *Wallet-blocked sender*: user marks a sender as blocked → all future txs from them get negative signal.
- *Spam-flagged tx*: dust-amount txs to many distinct recipients, or known-MEV patterns.
- *Rejected-from-pool*: tx that failed validation gets a permanent "rejected" tag.

Today the panel has only positive signals (recency, trust tier, status priority). Adding negatives makes the ranking honest. Estimated effort: **~120 LOC** for the scorer + a new column family for wallet-block records.

### 1.2 Diversity / dedup in the Selector

Twitter's Selector enforces per-author + per-topic diversity ("don't surface 5 tweets from the same person"). Our `TopK` at `pipeline.rs:256-266` is naive — purely score-descending. At 1 bps + 5K tx/block × 5s polling we don't notice; at 100 bps or under spam load, the panel would degenerate.

The fix is well-understood: greedy with diminishing returns per `task_type` and per `origin_wallet`. ~80 LOC replacing `TopK::select_top_k`. Doesn't change the public API.

### 1.3 Real-graph tiebreaker

Twitter's real-graph is the bipartite graph of "who interacts with whom" — used as a tiebreaker when two candidates have similar scores. TR-2026-004 said no to using the chain's tx graph as a **primary signal** ("noisy and adversarial"). But as a **tiebreaker** the risk profile is different — a wallet-distance-based tiebreaker is hard to gamify because you have to actually transact (which costs fees) to lower your distance.

Strawman: cache wallet-graph distance via BFS up to depth-3 from the panel viewer. Use it to break ties within ±0.05 score band. ~150 LOC + a cache.

### 1.4 Heavy-ranker probability calibration

Twitter's 10 outcome probabilities are calibrated post-hoc from production data — the hand-engineered features go in, the empirically-observed rates come out. Our `TxScorer` uses fixed weights `0.35 / 0.20 / 0.20 / 0.25` (`scorers/mod.rs:53-58`). Nobody has verified those weights are right for our chain's actual tx distribution.

**Calibration requires score persistence (see §2.6).** Without storing computed scores alongside the eventually-confirmed-or-orphaned outcomes, we can't do post-hoc fitting. This is the blocker for §1.4, §1.5, §1.6.

### 1.5 Hard-negative mining

Twitter mines hard-negatives ("looks engaging but is actually NSFW") to train the ranker against adversarial content. Chain equivalent: *"looks like a normal tx but is a replay attempt"*, *"large amount but front-running someone else's swap"*. We've never identified these in our data.

Same blocker as 1.4: needs persistence.

### 1.6 Embedding cache

For each user × content pair, Twitter caches the embedding vector to avoid recomputing on every score. Chain equivalent: cache `(wallet, token)` affinity scores so DEX swap candidates don't recompute their `price_impact` factor from scratch each panel poll. Premature today (~5K candidates is cheap); valuable at 10× scale.

### 1.7 A/B testing infrastructure

Twitter runs many ranker variants in parallel and routes a fraction of users to each. We could expose `&scorer_flavor=experimental` on `/api/v1/agent/panel/:addr` to let an operator route some agents to a candidate-stage scorer. Trivial to add (one trait dispatch in `build_default_panel_pipeline`); valuable when we start tuning.

---

## §2 — What we can enhance from what we've already learned

Code-ready, no research blockers. Sized by approximate LOC.

### 2.1 Fill the stub Sources — ~160 LOC total

`ConfirmedTxSource::fetch` at `concrete.rs:134-155` returns empty. The TODO comment says *"wires to `state.storage_engine.get_recent_transactions_for_wallet` when that surface stabilises."* The surface IS stable now (we use it in `handlers.rs` already). Wire it up.

`DexSwapSource::fetch` at `concrete.rs:159-169` — same story for the swap log.

Both are pure value-add: the panel's QUEUED + DONE zones will populate with real history instead of being empty.

### 2.2 Wire `SseEventEmitter` to the real SSE channel — ~40 LOC

Today (`concrete.rs:282-305`) the side effect only logs. The real SSE event-emitter lives on `state.event_emitter`. Replace the `tracing::info!` calls with `state.event_emitter.emit_immediate(SseEvent::PanelTaskUpdated { ... })`. Frontend `RecentActivityPanel.tsx` already listens for these.

### 2.3 Add the 4 missing Hydrators — ~80 LOC each, ~320 LOC total

The spec names them at `docs/agent-activity-panel-spec.md:162-178`:
- `TokenMetadataHydrator` — given a candidate referencing a token, fetch decimals/symbol/name from the contract state.
- `BlockReferenceHydrator` — for confirmed txs, attach `block_height` + `block_hash`.
- `ApprovalUrlHydrator` — for pending-approval tasks, attach the X-Wallet-Auth challenge URL the user should sign.
- `AttestationHydrator` — for DelegatedFiberLane tasks, attach the AFL-1 attestation proving the delegation.

None of these need new data — they all read from existing state. They just make the panel response richer.

### 2.4 Add 3 new scorers — ~60 LOC each, ~180 LOC total

- `WalletReputationScorer`: count of completed txs from sender → score in [0, 1] via log-saturating function. Tackles cold-start (§3.d) partially.
- `FeeReasonablenessScorer`: percentile of this tx's fee in current mempool's fee distribution. Penalizes both too-low (spammy) and too-high (mistake/MEV) fees.
- `PoolDepthScorer`: for DEX swaps, penalize candidates that would touch a pool with reserves below a threshold.

### 2.5 Diversity selector — ~80 LOC

See §1.2. Greedy-with-diminishing-returns variant of `TopK`. Public API doesn't change.

### 2.6 Score persistence — ~250 LOC ⭐ **the killer next move**

Add column family `CF_SCORE_HISTORY` to RocksDB. Key: `(block_height: u64 BE, tx_id: [u8;32])`. Value: serialized `ScoreReport` (we already have `Serialize` impls). Write from the panel pipeline's `SideEffect` stage after `Selector` picks candidates.

This is the unlock. Without persistence:
- §1.4 calibration is impossible (no historical outcomes to fit against)
- §1.5 hard-negative mining is impossible (no labels)
- §1.7 A/B testing is meaningless (can't compare flavors)
- §3.a "are our weights right?" is unanswerable

With persistence:
- 1 week of data = ~600K (block,tx) entries = ~50MB at ~80 bytes each
- Run a calibration job nightly: load last 7 days, compute confirmed-vs-orphaned ratio per component-score bucket, suggest re-weighted formula
- Hard-negative mining: SELECT * FROM SCORE_HISTORY WHERE total > 0.7 AND status = 'orphaned'

**Estimated LOC**: 50 LOC for the column family + serialization, 100 LOC for the writer hook in pipeline's SideEffect stage, 50 LOC for a read API at `/api/v1/agent/score-history/:addr`, 50 LOC of tests + privacy gates (don't expose another wallet's history).

**This should be priority #1 in the follow-on session.** Everything else builds on it.

---

## §3 — What we don't know we don't know

The meta-question. These aren't features to add; they're audits to run. Each is a worthy investigation.

### 3.a Calibration drift

Our hand-tuned weights at `scorers/mod.rs:53-58` (TxScorer) and `:95-100` (SwapScorer) were picked by intuition. We've never verified them against actual chain data.

**Investigation**: once §2.6 lands, run weekly calibration. Output: a CSV showing for each component, the correlation between component-score and the eventual confirmation/orphan outcome. If a component has near-zero correlation, it's noise — drop or re-weight.

### 3.b Do agents even USE the panel?

We assume "good ranking → useful." But agents aren't humans. They read JSON; they don't browse. Maybe what an agent actually needs is *not* a ranked list but a *typed query* ("give me all my pending approvals, ordered by deadline"). The current panel is shaped for human consumption.

**Investigation**: instrument the panel access log (we have `AccessLogger` at `concrete.rs:310-324`) to record (caller_user_agent, mode, zone, query_pattern) for 1 week. Look for patterns: do callers always pass `zone=now`? Do they paginate? Re-poll fast?

### 3.c Adversarial scoring

An attacker who reads `/api/v1/agent/panel/:addr?mode=embed` can introspect the public scorer formulas. Twitter's algorithm became an SEO target after the 2023 release. Same risk applies here: an adversarial wallet could craft txs that maximize `TxScorer` (small balance delta, low fee, low backlog ratio) to dominate other users' panels.

**Investigation**: red-team session. List the top-3 attacks a determined adversary could mount against our scorers given full source-code visibility. For each, propose a mitigation that doesn't require obscuring the scorer (security through obscurity isn't a design we want).

### 3.d Cold start for new wallets

Today `TrustTierScorer` assigns from enum membership (`Local / SignedByX / DelegatedFiberLane / Observed`), not tx history. A brand-new wallet's first tx gets the same score as an established wallet's first tx. Is that right?

**Investigation**: define a `WalletAgeScorer` that takes block-height-of-first-tx into account. Compare against `WalletReputationScorer` (§2.4). Which produces better-feeling rankings?

### 3.e Information leak via scoring signals

In `mode=owner` we return the user's own breakdown. But `mempool_backlog_ratio`, `reserve_utilization_ratio`, and `volatility_ratio` describe AGGREGATE state — they leak information about other users' pending activity.

**Investigation**: model a side-channel attack. If an attacker can poll the panel of a controlled wallet every 1 second, what can they infer about the rest of the chain? Specifically: can they distinguish "I'm being front-run" from "the mempool is busy"? Probably yes. Mitigation: bucket the signals (e.g., `mempool_backlog_ratio` rounded to 0.1 increments).

### 3.f Performance at scale

`Pipeline::run` at `pipeline.rs:192-240` is parallel-sources + sequential-everything-else. 5K candidates × 5s polling = fine. At 100 bps with 50K candidates × 1s polling? Untested.

**Investigation**: write a load test that synthesizes 100K candidates and measures p50/p99 panel response. If p99 > 500ms, the panel becomes a DoS vector. Profile and fix.

### 3.g Have we read post-2023 X-algorithm commits?

TR-2026-004 cites the 2026-05-15 re-release. Since then, xAI may have committed updates. We may have missed clever ideas.

**Investigation**: `git log` of `github.com/xai-org/x-algorithm` since 2026-05-15. Read the top-10 most-active files' diffs. Extract any architectural changes worth adopting.

### 3.h What does Home Mixer ACTUALLY do that we didn't take?

The "Home Mixer pipeline shape" is just the surface. The real Home Mixer also does: product-surface routing (For You vs Following timeline), request batching, candidate fan-out heuristics, candidate cache for hot users. We have none.

**Investigation**: read the actual Home Mixer source. Note which subsystems we skipped. For each, ask: does our chain need it? If yes, why? If no, why not?

### 3.i Cross-chain prior art

Other chains have activity panels: Aave (positions), Lens (social feed), Farcaster (cast feed), ENS (name expirations). We've not compared. Maybe one of them has solved a problem we're about to hit.

**Investigation**: pick the 3 closest cousins, read their docs/source, write a one-pager comparing approaches.

### 3.j xAI IP boundary

The Apache-2.0 release may reference internal-only systems (xAI's object storage, internal tracing). Which parts are *fork-safe* (we can lift verbatim) vs *ornamental* (referenced but won't work standalone)?

**Investigation**: take a representative file from x-algorithm-scorer's upstream and trace every external dependency. Build a fork-safety matrix.

### 3.k MEV / replay / sandwich — chain-specific negatives Twitter doesn't have

Twitter's algorithm doesn't have to worry about MEV. Our chain does. Are these "negative actions" worth a first-class `RiskIndicator` mixin per TaskCandidate?

**Investigation**: list the top-5 chain-specific risks we want to surface. For each, identify where the signal lives in storage. Decide whether to bake into the Scorer or surface as a separate `risk_tags` field.

### 3.l Score persistence retention + privacy policy

If we persist scores per (block, tx_id), how long do we keep them? Privacy implications: a wallet's score history becomes queryable forever unless we prune.

**Investigation**: write a retention policy. Strawman: keep 30 days of fine-grained, 1 year of bucket-quantized, then drop.

---

## §A.3 Killer next move — score persistence (§2.6)

If we do exactly one thing, do this. Until scores are persisted, we cannot:
- Validate that our scoring is right (§1.4, §3.a)
- Mine hard-negatives for adversarial training (§1.5)
- A/B test scorer variants (§1.7)
- Audit information leaks against historical data (§3.e)
- Measure cold-start treatment effectiveness (§3.d)

It's a ~250 LOC one-day delivery that unlocks 6+ downstream investigations.

---

## §B — MCP v2.1.0 surface (shipping alongside this doc)

Three new tools land in `tools/quillon-wallet-mcp/src/index.ts` to make the x-algo work usable from the agent loop:

1. `score_tx_dry` — dry-score a candidate tx via the q-api-server's `TxScorer`
2. `score_tweet_draft` — proxy to the x-algorithm-scorer sidecar on `:8090`
3. `agent_panel_breakdown` — variant of `agent_panel` returning full `ScoreReport.components`

These give the agent (me) the ability to introspect the scoring system in real-time without dropping to curl. They are pure surface work — no new scoring logic, just exposing what already exists.

See `lexical-munching-pelican.md` for the full implementation plan.

---

## Open follow-ups

This doc is a research artifact, not a decision record. Each §2 / §3 item becomes a task. The tasks are tracked in the agent session and should outlive this session.

Last update: 2026-05-20 by Claude Opus 4.7.

---

# Addendum 2026-05-20 (late) — what we found AFTER cloning the actual upstream

Cloned `github.com/xai-org/x-algorithm` to `/opt/orobit/shared/x-algorithm/` (2.1 MB code-only, Apache-2.0, May 15 2026 release). Read the real candidate-pipeline + home-mixer + phoenix sources. Updates §1 (what we can use) and §3 (what we don't know we don't know) with concrete findings — most of these were INVISIBLE to TR-2026-004 because that doc was written from the README only.

## §1 update — what we can use, with code in hand

### §1.8 ⭐ FOUR EXTRA PIPELINE STAGES (the big one)

Reading `candidate-pipeline/candidate_pipeline.rs` lines 22-33 + 89-137: the upstream has **10 stages**, not 6. Our `agent_panel/pipeline.rs` is missing:

- **`QueryHydrator`** (runs FIRST, parallel) — pre-fetches the viewer's context ONCE. Equivalent for us: fetch the wallet's balance + history + AFL-1 attestations once at panel-request time, then all subsequent stages read from a populated query object. Eliminates redundant fetches across 5000 candidate hydrations.
- **`DependentQueryHydrator`** (runs SECOND, after first query hydration) — second pass with deps satisfied.
- **`PostSelectionHydrator`** (runs AFTER Selector picks top-K) — only enriches the FINAL 50, not all 5000. **Massive perf win** for expensive hydrators like TokenMetadataHydrator (we don't need symbol/decimals for a candidate that won't be displayed).
- **`PostSelectionFilter`** (final filter on the selected subset) — last-mile drops (e.g., "user just blocked this sender mid-request").

Effort: ~100 LOC to extend `pipeline.rs` traits + ~150 LOC to wire stages in `Pipeline::run`. The trait additions are mechanical; the wins are real.

### §1.9 Per-stage `enable(&self, query: &Q) -> bool` for runtime gating

Every Hydrator/Scorer in upstream has a per-query `enable` method (`hydrator.rs:17-19`, `scorer.rs:15-17`). The pipeline filters disabled stages before running. This is A/B testing infrastructure baked into the type system — a stage can be present in the build but inactive for a specific query.

For us: `WalletReputationScorer` enables only if the wallet has > 10 txs; `PhoenixScorer` (future) enables only if `query.scorer_flavor == "phoenix"`. ~40 LOC across all stages.

### §1.10 Length-mismatch defensive guards

`hydrator.rs:30-48` and `scorer.rs:21-39` wrap every implementation in a safety net: if a hydrator returns the wrong number of candidates, the entire stage is marked as `vec![Err("length mismatch"); n]` rather than silently dropping candidates. Our `pipeline.rs::run` does no such check.

Per-candidate `Result<C, String>` (vs our `Vec<C>` all-or-nothing) means **one bad candidate doesn't fail the batch**. Critical for production.

### §1.11 Built-in CacheStore + CachedHydrator

`hydrator.rs:72-90` defines a `CacheStore<K, V>` trait + `CachedHydrator<Q, C>` wrapper. Every hydrator can cache its output keyed by something derived from the query/candidate. Cache hits skip the underlying fetch.

For us: TokenMetadataHydrator is a perfect cache target (token metadata never changes once set). Saves a contract-state read per swap candidate. ~80 LOC for the abstraction + per-hydrator wiring.

### §1.12 Stats + tracing instrumentation per stage

`#[xai_stats_macro::receive_stats(latency=Bucket50To500, size=Bucket500To2500)]` on every `run()` (hydrator.rs:28, scorer.rs:19). `#[tracing::instrument(skip_all, name = "hydrator", fields(name = self.name()))]` auto-spans each stage with the hydrator's name.

We have no per-stage stats. Without instrumentation we can't answer "which hydrator is slow?" Adding tracing alone is ~20 LOC; stats is ~80 LOC if we vendor a small ring-buffer histogram (or use prometheus we already have).

### §1.13 Eighteen production filters worth porting in spirit

`home-mixer/filters/` has 18 filters. The ones most directly applicable to our chain panel:

- **`previously_seen_posts_filter`** — per-user "user already saw this" tracking. Critical for an agent polling panel every 5s — don't re-surface the same task forever. **High value, missing entirely.**
- **`dedup_conversation_filter`** — group threads; equivalent for chain: group txs by `tx_chain_id` (replay/RBF) or by token pair.
- **`drop_duplicates_filter`** — same candidate from multiple sources; equivalent for us: same tx_id from MempoolTxSource + ConfirmedTxSource (will happen when we fill the stub).
- **`muted_keyword_filter`** — user-supplied keywords. Chain equivalent: user-supplied blocked sender list.
- **`author_socialgraph_filter`** — drop low-trust authors. Equivalent: drop low-reputation wallets.

### §1.14 Eighteen production hydrators — the patterns

`home-mixer/candidate_hydrators/` has 18 hydrators. Particularly relevant:

- **`engagement_counts_hydrator`** — fetch per-post engagement metrics; chain equivalent: per-token swap volume in last 24h (for DEX candidates).
- **`mutual_follow_jaccard_hydrator`** — Jaccard similarity between viewer's follow set and post-author's follow set. **Chain wallet-graph signal**: Jaccard between viewer's tx-counterparty set and candidate-sender's tx-counterparty set. Cheap, hard to game.
- **`blocked_by_hydrator`** — has the author blocked the viewer? Asymmetric block check.
- **`brand_safety_hydrator`** — Twitter-specific (NSFW/spam markers). Chain equivalent: AEGIS-QL flags, slashing-history flags.

### §1.15 Phoenix is JAX/Haiku, NOT PyTorch

`phoenix/pyproject.toml` line 7-12: deps are `dm-haiku>=0.0.13`, `jax==0.8.1`, `numpy>=1.26.4`, `pyright>=1.1.408`. Python ≥3.11. That's it — **only 4 runtime deps**. No CUDA required (JAX has CPU backend). The "mini" model is 256-dim, 4 attention heads, 2 transformer layers — small by ML standards.

Implication: running Phoenix as a sidecar from Layer 2 is more tractable than I assumed. Steps:
1. `git lfs install` + `git -C /opt/orobit/shared/x-algorithm lfs pull` (~3GB once)
2. `cd /opt/orobit/shared/x-algorithm/phoenix && uv sync` (the uv.lock is pinned)
3. `python run_pipeline.py --artifacts_dir ./artifacts --sequence_file ... --top_k_display 30`

The `x-algorithm-scorer` Layer 2 then spawns `phoenix/run_pipeline.py` as a long-lived subprocess and forwards `/score` requests through stdin/stdout JSON. ~200 LOC of Rust process management.

### §1.16 Two-stage retrieval + ranking (we're doing 1-stage scoring)

`phoenix/run_pipeline.py:15-19`:
> Loads exported checkpoints, a pre-computed corpus, and a user action sequence, then runs:
>   1. Retrieval: encode user history → dot product with corpus → top-K
>   2. Ranking: score top-K candidates with per-action engagement model

We have ONLY stage 2 (ranking) — and even that's via hand-engineered features. Stage 1 (retrieval) is content-based candidate generation: from a viewer's history, find candidates that are SEMANTICALLY similar to past engagement.

Chain equivalent: from a wallet's tx history (token pairs swapped, recipients sent to), retrieve **other wallets / pools / tokens** that have similar tx patterns. That's a real recommendation system, not just a panel.

Phase 4+ work, but huge. Out of scope today.

## §3 update — what we don't know we don't know (DEEPENED)

### §3.m `xai_stats_receiver`, `xai_decider`, `xai_feature_switches`

These appear as Rust imports in `candidate-pipeline/scorer.rs:7` (`xai_stats_receiver`), `candidate_pipeline.rs:16` (`xai_stats_receiver`), and `candidate_pipeline.rs:60-61` (`xai_feature_switches::Params`, `xai_decider::Decider`). They're NOT in any of the Cargo.toml files in the cloned repo — they're **internal xAI crates** that didn't get open-sourced.

To compile `candidate-pipeline/` standalone we'd need to either:
- Vendor stubs (small Empty-struct shims that satisfy the trait signatures)
- Forge with `cargo patch` to point at our own implementations
- Or: just port the trait shapes by hand into our codebase without depending on xai crates (what we already did)

This means **`candidate-pipeline/` isn't fork-compilable as-is**. The patterns are extractable but the build isn't. Concrete implication: we can't `cargo add x-algorithm-candidate-pipeline = { path = "/opt/.../candidate-pipeline" }` and have it work.

### §3.n PTOS policy

`README.md` line 35: *"task-execution engine for content understanding workloads such as spam detection, post-category classification, and PTOS policy enforcement."*

What's PTOS? Likely an internal Twitter/X acronym. Possibly "Public Tweets Operating Standards" or similar. We don't know. Affects how we should label our equivalent (chain-level content moderation).

### §3.o The `grox/` content-understanding service

Python. Contains `classifiers/`, `embedder/`, `generators/`, `summarizer/`, `tasks/`. This is the SECOND ML stack — runs spam classification, brand-safety check, content categorization. Phoenix is the ranker; Grox is the upstream content understanding.

We could plug grox in for `score_tweet_draft` to get real spam/policy signals instead of our keyword list. But it's another Python service to host. Decision deferred.

### §3.p `thunder/` Kafka in-network candidate fetcher

`thunder/kafka/`, `thunder/kafka_utils.rs`, `thunder/thunder_service.rs`. Uses Apache Kafka to fetch in-network (followed-account) candidates. We have gossipsub for the same purpose on chain. Different transport, different consistency model. Decision: skip thunder entirely; our gossipsub equivalent is the existing peer-heights / block-pack channels.

### §3.q `home-mixer/scorers/` has 7 distinct scorers — what do they do?

Files visible: `author_diversity_scorer.rs`, `oon_scorer.rs`, `phoenix_scorer.rs`, `ranking_scorer.rs`, `vm_ranker.rs`, `weighted_scorer.rs`. Read order unknown — they probably chain: `phoenix_scorer` produces base scores → `weighted_scorer` applies feature-flag weights → `author_diversity_scorer` re-ranks → `vm_ranker` does final scoring. Unverified.

Worth reading. ~1 hour of investigation.

### §3.r The difference between `for_you_server.rs` and `scored_posts_server.rs`

Two endpoints, two server files. Likely:
- For You = user-facing endpoint, runs full pipeline
- Scored Posts = service-to-service endpoint, runs ranking only (skips retrieval, assumes candidates given)

We have only `agent_panel/handler.rs`. Do we need the equivalent split (panel vs. score-already-given-candidates)? Possibly yes for the future MCP `score_tx_dry` tool — it would be a "scored_posts_server"-style endpoint that takes candidates and returns scores.

### §3.s Can the mini Phoenix model actually run on Beta/Delta CPU?

JAX has a CPU backend. The mini model is 256-dim × 4 heads × 2 layers — very small. Should fit in a few hundred MB of RAM, run at ~50 tok/s on a modern CPU. Untested. Worth a 1-hour spike: `git lfs pull` → `python run_pipeline.py` → measure latency on example_sequence.json.

### §3.t Are the JAX vs candle conversion tools mature?

JAX uses .npz weight files (numpy archive). To port to candle (pure Rust), we'd need:
- A weight-extraction script: walk the JAX param tree, save each tensor as safetensors
- A model-rebuild script: define the same architecture in candle, load tensors

The "mini" model is small enough that this is one-day work, not one-week. But we don't know how stable JAX→candle is for transformer architectures. Untested.

---

# Side project: clone Bitcoin Core latest (v31), analyse for agentic-money advantage

User asked for this in parallel. The goal: pull Bitcoin Core source (the latest stable, ~v31 series circa 2026), read it carefully, and identify:

1. **BIP proposals worth writing** for agentic money (e.g., agent-signed transactions, agent-friendly memo conventions, micropayment channels for AI-to-AI commerce).
2. **Patterns Quillon could adopt** (e.g., how Bitcoin handles wallet recovery, fee estimation under load, mempool admission rules).
3. **What Bitcoin can NOT do for agentic money** — surface the gaps that justify Quillon's existence.

Clone target: `/opt/orobit/shared/bitcoin-core/` (separate dir, won't conflict with the x-algorithm clone).

Plan:
- `git clone --depth 1 --branch 31.x https://github.com/bitcoin/bitcoin /opt/orobit/shared/bitcoin-core`
- Initial inventory: top-level dirs, line counts, key files (`src/wallet/`, `src/net_processing.cpp`, `src/policy/`, `src/script/`)
- Read the most recent BIPs implemented (look at `doc/bips.md` or similar)
- Write `docs/bitcoin-core-analysis-for-agentic-money-2026-05-20.md` with three sections:
  - **What Bitcoin already does that helps agents** (Taproot, PSBT, miniscript)
  - **What Bitcoin can't do for agentic money** (no native AI wallet primitive, no AFL-1-style delegation, no chain-attested memo, no agentic mempool admission)
  - **BIP draft sketches** for the three highest-value primitives — one of these could become a real BIP submission

Tracking as a separate task: see TaskCreate below.

Last update: 2026-05-20 by Claude Opus 4.7 (after reading the actual upstream).

