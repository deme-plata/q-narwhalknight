# Audit Posture — Quillon Graph, 2026-05-22

> **Response to:** the deep-research audit of *deme-plata/q-narwhalknight* completed 2026-05-22 + follow-up external review of this document (same date).
> **Project state at the time of writing:** v10.11.10 on branch `agent/cross-shard-simd-validation`, working tree on Beta.
> **Authors:** maintainer pair (human operator + Claude Opus 4.7).
> **Last honest-state correction:** 2026-05-22 (this revision incorporates the follow-up review's findings, including two STILL-APPLIES citation rows initially mis-classified as NOT-REPRODUCED).
> **Next scheduled honest-state review:** 2026-08-22.

> **Provenance note — read this first.** The follow-up review correctly observed that this project currently presents *multiple* identity surfaces depending on where you look. See [§0 — Provenance reconciliation](#0--provenance-reconciliation-tier-0-blocker-for-all-other-claims) below; that section is **Tier 0** and blocks every other claim in this document.

---

## §0 — Provenance reconciliation (Tier 0, blocker for all other claims)

The follow-up reviewer's most important finding is not a code-path issue. It is that **the project currently presents materially different identities, versions, and source-of-truth surfaces depending on where it is read**. Until that is reconciled, no other remediation can be independently audited — a reviewer cannot verify a row in §2 unless they know which tree they're reading.

Accessible evidence shows at least four distinct surfaces:

| Surface | What it says | Source |
|---|---|---|
| **This document** | v10.11.10 on `agent/cross-shard-simd-validation` | This file's header |
| **Connector-visible main / `Cargo.toml` workspace.package** | `version = "10.9.38"` *(per follow-up reviewer; this revision is being written from a different working tree)* | `Cargo.toml` `[workspace.package]` |
| **Connector-visible README** | "Quillon Graph", release line v10.9.x, repository tag v10.9.55 | `README.md` |
| **Public GitHub web surface (rendered to anonymous visitors)** | v0.0.1-alpha README, 0.1.0 minimal workspace with ~7 members, ~5 commits, GitLab clone instructions | github.com/deme-plata/q-narwhalknight as rendered to logged-out visitors |
| **Root `Cargo.toml` `repository` field** | `https://code.quillon.xyz` (claims a third remote as canonical) | `Cargo.toml` `[package.metadata]` or `[workspace.package] repository` |

A reader who comes from the public GitHub surface and a reader who comes from the local working tree are looking at different projects, in a meaningful sense. The differences are not "minor release notes"; they span the project name (Quillon Graph vs the older codename), the canonical remote (GitHub vs code.quillon.xyz vs GitLab), the version string (0.1.0 vs 10.9.38 vs 10.11.10), and the workspace size (~7 vs 76 members).

**Tier 0 commitment (lands within 7 days, by 2026-05-29):**

1. **Publish a single canonical-provenance block** at the top of `README.md`:
   - Canonical remote: which of GitHub `deme-plata/q-narwhalknight`, the local git daemon at `git://185.182.185.227/q-narwhalknight`, or `code.quillon.xyz` is authoritative; which are mirrors.
   - Canonical branch: `agent/cross-shard-simd-validation` if that remains the working branch, OR document the release branch lineage explicitly.
   - Current production binary SHA256: matched to the `Cargo.toml` `[workspace.package] version` field.
   - Public-surface drift notice: if the rendered GitHub front page lags behind the local tree, say so — link the local-tree state and the public-tree state side by side.
2. **Force-push the canonical surface forward** OR explicitly document the lag as intentional (e.g., "GitHub front page is updated only at release tags; live state lives at code.quillon.xyz"). Silent drift is the failure mode.
3. **Bump the public GitHub README to current state** (Quillon Graph branding, real architecture, real version) so the v0.0.1-alpha rendering stops misleading first-time visitors.
4. **Update `[workspace.package] repository`** to whichever URL is now canonical; remove the references that aren't.

Until Tier 0 lands, every other row in this document carries an implicit "as observed in the working tree on Beta at 2026-05-22" caveat that a reviewer cannot verify without operator credentials. That caveat is itself a finding.

**Why Tier 0 above Tier 1:** the original audit's documentation-drift findings (workspace size, Rust version, crypto naming) are real and important, but they are *symptoms* of the provenance problem, not the root cause. Fixing the symptoms while leaving the provenance ambiguous means the next audit will rediscover the same drift on whichever surface didn't get the fix.

---

## §1 — The audit

On 2026-05-22 a third-party deep-research analysis of the public GitHub repository *deme-plata/q-narwhalknight* (Quillon Graph) was delivered. The audit is unusually thorough: it inspected the workspace manifest, the consensus crates, the network layer, the benchmark harness, the AI inference manifest, the GUI packages, the operational runbook, and the external literature the project cites. It is also fair. Its central framing — *"q-narwhalknight is better treated as an ambitious, evolving systems monorepo than as a clean, publication-grade reference implementation"* — is correct.

The audit identifies five categories of problem: documentation drift (README/Cargo.toml/CLAUDE.md inconsistencies), citation rot (mis-attributed IACR ePrint references in code comments), consensus placeholders (single-candidate anchor election, empty-tx commit decisions, placeholder vertex in reliable broadcast, VDF fallback warns), version inconsistencies (libp2p, Rust toolchain, project version), and absent benchmark artifacts (harness without packaged results).

This document is the project's public response. Its purpose is **not** to dispute the audit — most of its findings are substantively correct. Its purpose is to:

1. Acknowledge each finding by name.
2. Map each finding to its **current verified state** at v10.11.10 (some are already addressed; others still hold; two of the specific citation-rot examples could not be reproduced in current source).
3. Commit to a tiered response — what lands within 7 days, what within 14 days, what is a named multi-week engineering work stream.
4. Establish a quarterly honest-state review cadence, so the project's posture remains self-auditing rather than drifting back into marketing-first claims.
5. Honor the design philosophy of the *Skin in the Cathedral* paper (papers/skin-in-the-cathedral-2026.tex, §6): operational dullness as competitive advantage. Answering an audit honestly is itself an act of operational dullness; it is what a maintenance-first project does.

The full audit text is preserved verbatim in [Appendix A](#appendix-a--audit-text-as-received) so external readers can read the original critique alongside this response, and judge for themselves whether the response is adequate.

---

## §2 — Verified state grid

This is the most important section of the document. Every row below is a specific audit finding mapped to its current state at HEAD. Each "STILL APPLIES" row cites a file path and line number that an external reviewer can `grep` directly. If a finding has been addressed, the row says so with commit SHA and date.

| # | Audit finding | Current state at v10.11.10 | Evidence | Response tier |
|---|---|---|---|---|
| 1 | `anchor_election.rs::elect_anchor()` builds `vec![one_candidate]` and selects a single candidate, not a multi-candidate set | **STILL APPLIES** | `crates/q-dag-knight/src/anchor_election.rs:268–270` | Tier 3 |
| 2 | `commit_logic.rs::evaluate_chain_commit()` returns `Ok(None)` with `// TODO: Implement causal dependency analysis` | **STILL APPLIES** | `crates/q-dag-knight/src/commit_logic.rs:330` | Tier 3 |
| 3 | `commit_logic.rs` commit decisions carry `let transactions = vec![]` (hardcoded empty payload) | **STILL APPLIES** | `crates/q-dag-knight/src/commit_logic.rs:348–350` | Tier 3 |
| 4 | `quantum_vdf.rs` emits `warn!("Quantum-resistant VDF not fully implemented, using enhanced post-quantum")` | **STILL APPLIES** | `crates/q-dag-knight/src/quantum_vdf.rs:384` | Tier 2 (relabel) + Tier 3 (real VDF) |
| 5 | `quantum_vdf.rs` emits `warn!("Quantum-native VDF not implemented, using quantum-resistant")` | **STILL APPLIES** | `crates/q-dag-knight/src/quantum_vdf.rs:396` | Tier 2 (relabel) + Tier 3 (real VDF) |
| 6 | `reliable_broadcast.rs` returns `create_placeholder_vertex(vertex_id)` instead of retrieving from storage | **STILL APPLIES** | `crates/q-narwhal-core/src/reliable_broadcast.rs:263–265` | Tier 2 |
| 7 | `reliable_broadcast.rs` hardcodes `f = 1` for a 4-validator assumption | **STILL APPLIES** | `crates/q-narwhal-core/src/reliable_broadcast.rs:67` | Tier 2 |
| 8 | README §Project Structure shows 7 crates; actual workspace has 76 active members | **STILL APPLIES** | `README.md` §Project Structure; `Cargo.toml` workspace.members. The follow-up review counted 82; the original audit said 82; this revision counted 76 via `awk` over the `members = [` block. Three different counts for the same artifact is itself evidence of drift; Tier 1 will publish a deterministic counting script alongside the README rewrite so future reviews agree. | Tier 1 |
| 9 | README says "Rust 1.70+" while `Cargo.toml` pins `rust-version = "1.86"` | **STILL APPLIES** | `README.md:69`; `Cargo.toml` [workspace.package] | Tier 1 |
| 10 | README still uses pre-standard names (Kyber, Dilithium) rather than ML-KEM / ML-DSA from FIPS 203 / FIPS 204 (Aug 13 2024) | **STILL APPLIES** | `README.md` lines 22, 56, 142, 145 | Tier 1 |
| 11 | `crates/q-api-server/Cargo.toml` declares `libp2p = "0.53"` while workspace pins `0.56` | **STILL APPLIES** | `crates/q-api-server/Cargo.toml:42` | Tier 1 |
| 12 | `CLAUDE.md` says "We do NOT use GitHub" while the project clearly does | **ALREADY FIXED** 2026-05-21 | `CLAUDE.md` lines 278–284 explicitly state "We use **both** GitHub and the local-git daemon" and acknowledge the previous claim was wrong | n/a |
| 13 | Code comment cites IACR ePrint 2025/1050 for "Genus-2 VDF" but the actual paper is about integral cryptanalysis of block ciphers | **STILL APPLIES** *(this row was initially mis-classified as NOT REPRODUCED in the first revision of this document — the follow-up review correctly identified the comment is present at the cited location; my Explore agent had grep'd for the URL pattern `iacr.org/2025/1050` while the actual citation reads `(IACR 2025/1050)`)* | `crates/q-dag-knight/src/lib.rs:28` — `// ✨ v1.0.58-beta: Genus-2 VDF for quantum-resistant anchor election (IACR 2025/1050)` (line 49 also references "Genus-2 VDF exports") | Tier 1 |
| 14 | Code comment cites IACR ePrint 2025/1056 for "lattice aggregate signatures" but the actual paper is about private signaling | **STILL APPLIES** *(mis-classified in revision 1 — same agent grep pattern issue as row 13)* | `crates/q-network/src/lib.rs:54` — `// ✨ v1.0.58-beta: Lattice Aggregate Signatures for 98% bandwidth reduction (IACR 2025/1056)` | Tier 1 |
| 15 | `q-tps-benchmark` harness exists with metric vocabulary (TPS, P50/P95/P99, success rate, transport ladder) but ships no packaged result files | **STILL APPLIES** | `crates/q-tps-benchmark/` contains source; no `bench-results/` directory present | Tier 1 |
| 16 | `qnk_libp2p_rx_bytes_total`, `qnk_libp2p_tx_bytes_total`, `qnk_block_pack_response_bytes` metrics declared but never incremented | **ALREADY FIXED** in v10.9.32 | Wired three increments in `crates/q-network/src/unified_network_manager.rs` (see Cargo.toml `[workspace.package] version` comment block for v10.9.32) | n/a |
| 17 | "DAG-Knight" branding does not map to a paper called DAG-Knight; underlying inspiration is DAG-Rider (Keidar et al., *All You Need is DAG*) | **STILL APPLIES** | README and crate name use "DAG-Knight" without naming the actual literature lineage | Tier 1 (documentation only) |

**How to re-verify this grid:** every "STILL APPLIES" row cites a specific file:line. From a checkout of `agent/cross-shard-simd-validation` at v10.11.10:

```bash
# Row 1
sed -n '268,270p' crates/q-dag-knight/src/anchor_election.rs
# Row 2
sed -n '328,332p' crates/q-dag-knight/src/commit_logic.rs
# Row 3
sed -n '346,352p' crates/q-dag-knight/src/commit_logic.rs
# Rows 4–5
grep -nE 'not fully implemented|not implemented, using' crates/q-dag-knight/src/quantum_vdf.rs
# Rows 6–7
sed -n '63,69p;261,267p' crates/q-narwhal-core/src/reliable_broadcast.rs
# Row 8
awk '/^\[workspace\]/,/^\[/' Cargo.toml | grep -cv '^#'
# Row 9
grep -n "Rust 1\." README.md
# Row 10
grep -n "Kyber\|Dilithium" README.md
# Row 11
grep -n "^libp2p" crates/q-api-server/Cargo.toml
# Row 12
sed -n '275,290p' CLAUDE.md
# Rows 13–14
grep -rn '2025/1050\|2025/1056' crates/
```

If any row's evidence command no longer matches the cited line numbers, the grid is stale; that itself is information about project drift and is itself useful evidence. The next quarterly review (§5) regenerates the grid.

---

## §3 — Response by tier

### Tier 1 — Documentation truthfulness (commits within 7 days, by 2026-05-29)

Tier 1 items are correction work, not refactor work. They land as a sequence of small PRs against `agent/cross-shard-simd-validation`. Each item has an explicit deliverable path.

**T1.1 — README architecture pass (row 8 + row 17).** Rewrite `README.md` §Project Structure to list the 72 actual workspace members grouped by domain — consensus, network, cryptography, application, AI/agents, tools, GUI/SDK — rather than a flat 7-crate diagram. Add a one-paragraph disclosure at the top of the section: *"This README documents the consensus and network slice. The full workspace contains 72 members spanning DeFi bridges, AI inference, mining pool, native and web wallets, and tooling — see `crates/` and `gui/` for the complete inventory."* Also add an explicit lineage sentence for the consensus design: *"The DAG-Knight consensus engine is inspired by DAG-Rider (Keidar et al., 'All You Need is DAG'), Narwhal/Tusk (Danezis et al.), and Bullshark (Spiegelman et al.). The name 'DAG-Knight' is a project label for this local adaptation; it does not refer to a separate published protocol."*

**T1.2 — Rust toolchain alignment (row 9).** Change `README.md:69` from "Rust 1.70+" to "Rust 1.86+" so it matches `Cargo.toml` `[workspace.package] rust-version = "1.86"`. Add a line: *"The post-quantum cryptography crates and AI inference dependencies require a recent Rust toolchain; the workspace pins 1.86 as the floor."*

**T1.3 — Cryptography nomenclature update (row 10).** In **operator-facing prose** (README, CLAUDE.md, docs/), rename Kyber → ML-KEM (FIPS 203, standardized 2024-08-13) and Dilithium → ML-DSA (FIPS 204, standardized 2024-08-13). The first mention of each in every document gets the form: *"ML-KEM-1024 (the NIST-standardized form of CRYSTALS-Kyber per FIPS 203, August 13, 2024)"*. Crate-level dependency names (`pqcrypto-kyber`, `pqcrypto-dilithium`) stay as-is because those are the published `crates.io` names; ecosystem compatibility wins over naming purity at the dependency level.

**T1.4 — libp2p version unification (row 11).** Change `crates/q-api-server/Cargo.toml:42` from `libp2p = "0.53"` to `libp2p.workspace = true` so it inherits 0.56 from the root workspace. Build verification: `cargo build --release --package q-api-server`. If the bump surfaces breaking API changes at call sites in `q-api-server`, fix them with targeted patches; if the cascade is large, hold the bump, write `docs/versioning.md` documenting the intentional divergence, and add a TODO for the next consensus-critical release window. The intent is that within 7 days the divergence is either closed or explicitly documented; ambiguity is not acceptable.

**T1.5 — Citations index (rows 13–14, even though not reproduced).** Create `docs/citations.md` cataloguing every IACR ePrint citation, every academic-paper reference, and every external standards-document citation present in code comments and documentation. For each entry: file path, line number, claim being supported, actual paper title, paper URL, and a one-line abstract. The audit could not be reproduced for the specific 2025/1050 and 2025/1056 examples in current source — but absence of evidence is not evidence of absence, and writing the catalogue is the disciplined response. Once it exists, regression is `grep`-detectable.

**T1.6 — First packaged benchmark result (row 15).** Run `q-tps-benchmark` `advanced_benchmark.rs` once on Epsilon at v10.11.10 with the JSON 1K-tx configuration (the simplest variant that exercises the harness end-to-end). Capture the output as `bench-results/v10.11.10-1k-json-epsilon-2026-05-22/result.json`. Add `bench-results/README.md` documenting:

- Exact commit SHA + binary SHA256 of `q-api-server` and `q-tps-benchmark`.
- Machine specs: Epsilon, 48 cores, 64 GiB RAM, Ubuntu 24.04, 10 Gbit/s uplink.
- Network conditions at run time: `mainnet-genesis`, peer count, current_height.
- Run command verbatim.
- Raw metric output (TPS, mean latency, P50/P95/P99 latency, success rate).
- Caveats: warm vs cold RocksDB cache state, `dex_ready` gate state, current `dev_fee_bps`.

This shifts the project's benchmark posture from *"harness + targets"* to *"harness + one reproducible result"*. Subsequent runs add new bundles under `bench-results/`; they never overwrite existing bundles.

### Tier 2 — Phase-honest relabeling (commits within 14 days, by 2026-06-05)

Tier 2 items are small-but-substantive code changes that move the project from misleading framing to honest framing. They are not full implementations of the missing functionality; they are accurate descriptions of what currently exists, plus tightening of the integration boundaries.

**T2.1 — `quantum_vdf.rs` honest relabeling (rows 4–5).** Today the code emits warns claiming the "quantum-resistant" and "quantum-native" VDF modes are "not fully implemented, using" the next-step-down construction. In reality the only implementation is iterated SHA-3 with periodic QRNG entropy injection. That is **not a VDF in the cryptographic sense** — no soundness proof, no succinct verification, no rigorous time-lower-bound guarantee. Tier 2 patch:

- Rename the public API surface from `QuantumVDF` to `QuantumBeacon` (or `IteratedHashBeacon` if we want maximal specificity).
- Replace the misleading warns with one explicit honest statement at construction time: *"QuantumBeacon: heuristic delay function based on iterated SHA-3 with QRNG entropy injection. NOT a VDF in the formal sense — no soundness proof, no succinct verification. See docs/roadmap-real-vdf.md for the literature-grade replacement work."*
- Add `docs/roadmap-real-vdf.md` queuing the Tier 3 work to ship Wesolowski-RSA or Pietrzak VDF.

This is a 1-day patch. It does not implement a real VDF — that is Tier 3. It eliminates the false framing.

**Strengthened per follow-up review:** for consensus- or leader-selection-adjacent code paths, unsupported VDF modes must **fail closed** rather than warn-and-continue. The current `warn!()` + fallback pattern is acceptable for prototyping, but it silently produces consensus artifacts that look as if they came from the requested mode when they didn't. The Tier 2 patch will replace the fallback with `return Err(BeaconError::ModeUnsupported { requested, available })` for the affected modes; callers that need a beacon must explicitly choose the supported one or handle the error. This eliminates a class of silent-misframing risk that the warning-based pattern can't.

**T2.2 — `reliable_broadcast.rs` validator-set-aware thresholds (row 7).** The hardcoded `f = 1` is correct for the current 4-validator deployment but breaks silently as the validator set grows. Tier 2 patch:

- Read validator-set cardinality from the on-chain registry at engine construction time (`BalanceFinalityEngine::update_validator_set` already does this for the Bracha-RB balance path; reuse the same accessor).
- Compute `f = (n - 1) / 3`, `echo_quorum = 2f + 1`, `ready_amplify = f + 1` dynamically.
- Recompute when the validator set changes (subscribe to a registry-update event).

Estimated effort: 3–4 days, plus regression test coverage for `n = 4, 7, 10, 13` validator counts.

**T2.3 — `reliable_broadcast.rs` real vertex retrieval (row 6).** Today after READY-quorum the code returns a placeholder vertex synthesized from just the vertex ID. The honest behavior is: look up the real vertex from `VertexStore`; if absent after READY-quorum (a *bona fide* invariant violation), return `Err(BroadcastError::VertexLost)` rather than synthesizing a fake one. Tier 2 patch:

- Replace `create_placeholder_vertex(vertex_id)` with `self.vertex_store.get(vertex_id).await?` (the `VertexStore` trait is already constructed; the lookup method may need adding).
- Define `BroadcastError::VertexLost` variant.
- The caller treats `VertexLost` as a delivery failure, not a delivery success — different from current behavior, which silently masked storage problems as successful delivery with garbage payload.

Estimated effort: 2–3 days, plus regression test that forces a storage miss and asserts `VertexLost` propagates.

### Tier 3 — Real consensus fills (named, scheduled, not promised)

Tier 3 items are the substantive engineering work. Each is a separate work stream with its own branch, regression suite, Docker-soak on Alpha (24h minimum, 48h preferred for chain-commit work), and PR with the verified-state grid row updated to **"FIXED at SHA xxxx, 20yy-mm-dd"**. Estimates are work-effort ranges, not delivery dates.

**T3.1 — Multi-candidate anchor election (row 1).** Walk the round's full candidate set from `vertex_store`, score each candidate (use the existing score function), select the highest-scoring candidate, deterministic tiebreak on `vertex_id` bytes for safety under adversarial scoring. Effort: 2–4 days, gated on regression tests that adversarially construct multi-candidate rounds and assert deterministic outcomes.

**T3.2 — Chain-commit + non-empty transaction payloads (rows 2–3).** Two coupled gaps: implement causal dependency analysis (line 330 TODO) so the commit rule actually walks the predecessor DAG; populate the commit decision's `transactions` vector by retrieving payloads from the committed vertex's predecessor set, deduping across the commit batch. Effort: 1 week of focused work, plus a 48h Docker-soak that produces verifiable commits with non-trivial payload counts.

**T3.3 — Literature-grade VDF (rows 4–5, second half).** Ship either Wesolowski-RSA or Pietrzak VDF as a replacement for the current iterated-SHA-3 heuristic. Verify against published vectors. Update `QuantumBeacon` (post-T2.1 name) to optionally use the real VDF for rounds where succinct verification matters, with a feature flag for backwards compat. Effort: 1–2 weeks depending on which scheme is chosen. The "quantum" framing is dropped — these are classical VDFs that happen to compose with our PQ signature stack.

**Why Tier 3 estimates are work-effort, not delivery dates.** The project is operated by a small maintainer team; scheduling these items competes with production operations, security cherry-picks, and user-facing feature work. Each Tier 3 PR will name its own delivery date when it opens, not when this document is written.

---

## §4 — Benchmark posture

The `q-tps-benchmark` harness at `crates/q-tps-benchmark/src/advanced_benchmark.rs` is a thoughtfully structured benchmark: it tracks TPS, P50/P95/P99 latency, success rate, and frames the optimization search as a transport ladder (JSON → MsgPack → batch → WebSocket streaming). The metric vocabulary is correct and the framing is right.

What the harness **does not** ship is a packaged result artifact bundle. Without one, the project's published throughput numbers ("1K → 10K → 100K → 1M+ TPS" appear in the harness's own variable names and adjacent documentation) are best read as *targets the harness is designed to measure*, not as *measured outcomes that have been independently reproduced*. The external literature is much cleaner on this dimension: Narwhal/Tusk (Danezis et al., EuroSys 2022) reports >130k tx/s for Narwhal-HotStuff and up to 600k tx/s with more workers; Tusk reports 160k tx/s WAN; Bullshark (Spiegelman et al.) reports 125k tx/s at 2-second latency for 50 parties. Those are the right baselines until this project publishes its own reproducible result bundle.

Tier 1 commits: by 2026-05-29, `bench-results/v10.11.10-1k-json-epsilon-2026-05-22/` will be in the tree. It will contain one configuration's output with full reproduction metadata. It will not claim 1M TPS. It will claim what it measures, with the caveats that apply.

This is a small artifact. Its value is not the number it reports; its value is that the project has now joined the convention of *publishing what was measured under what conditions*. Subsequent bundles compound; the harness becomes a real experimental tool rather than a marketing surface.

---

## §5 — Cadence: quarterly honest-state review

The verified-state grid in §2 is only useful if it stays current. The project commits to a quarterly cadence:

- **Next review:** 2026-08-22.
- **Process:** one session (Claude + maintainer, or solo human) re-runs every `grep` and `sed` command in §2 against current HEAD. Each row gets re-evaluated.
  - "STILL APPLIES" rows that still apply: remain.
  - "STILL APPLIES" rows that have been fixed: update to "FIXED at SHA xxxx, 20yy-mm-dd" with a one-sentence summary of what changed.
  - "ALREADY FIXED" rows: confirm the fix still holds (regression sweep).
  - New findings: add new rows.
- **Output:** a new file at `docs/audit-posture-2026-08-22.md` (or whatever the review date is). It supersedes this document for grid purposes but does not replace it as historical record.
- **Failure mode:** if the review does not happen by 2026-08-22 + 7 days, the project's own commit history shows that fact. External auditors should treat that as itself a finding about project state.

**Strengthened per follow-up review — machine-runnable evidence script (Tier 1 deliverable by 2026-05-29).** The §2 grid is currently `grep`/`sed`-reproducible by hand. That is the minimum bar; the higher bar is that the grid is **CI-enforced**. The Tier 1 patch will add `scripts/verify-audit-posture.sh` that:

1. Reads the date-stamped audit-posture markdown for the latest review.
2. For each "STILL APPLIES" row, runs the cited `grep`/`sed` command and confirms the expected pattern matches at the cited line range.
3. For each "ALREADY FIXED" row, confirms the cited commit SHA is in the current branch ancestry.
4. Exits non-zero on any mismatch.

CI runs this script on every push to `agent/cross-shard-simd-validation` and on every PR. Drift between the document and the code becomes a failed build, not a manual discovery. The follow-up reviewer's point — *"the strongest version of this would be a checked-in script that regenerates the grid from a fixed commit, emits the exact snippets, and fails CI if the document and code drift apart"* — is exactly what this delivers.

A **citation-lint** companion script (also Tier 1) reads `docs/citations.md`, fetches each IACR ePrint URL, compares the cited title against the actual paper title, and fails CI on mismatch. This eliminates the class of error that produced rows 13–14 (and made revision 1 of this document mis-classify them).

This cadence is the strongest mechanism the project has for not drifting back into marketing-first claims. The grid is reproducible by `grep`; the cadence is verifiable by `git log`. Neither relies on trust.

---

## §6 — Why this document exists

The *Skin in the Cathedral* paper (papers/skin-in-the-cathedral-2026.tex), §6, argues that operational dullness is competitive advantage. The wide-body jet (the 747) won not by being more glamorous than the Concorde but by being more *operationally honest* — its dispatch reliability and per-seat economics were trustable in a way the Concorde's were not. The analogy applies here. The audit's existence and this response are not bugs; they are the system working. External review, internal honesty, named gaps with timelines — these are the difference between a maintenance-first cathedral and a marketing-first feature factory.

Luca Pacioli, in the *Summa de Arithmetica* (1494), set out double-entry bookkeeping not as an aesthetic achievement but as a discipline: the bookkeeper's job is to make the books match the warehouse, not to make the books pretty. This document is the books matching the warehouse. The warehouse contains placeholder logic at four cited call sites; the books now say so, with line numbers, response tiers, and a date for the next inventory.

The project will be evaluated, eventually, by people who do not read marketing and do read code. This document is written for them.

---

## Appendix A — Audit text as received

The full audit text is preserved verbatim below. It was delivered 2026-05-22 by a third-party deep-research process; the author's name is not included here at their discretion. Section numbering matches the original.

> *(Audit text omitted from this Markdown render because of size; preserved at `docs/audit-2026-05-22-as-received.md` for traceability. Anyone responding to a row in §2 of this document should first re-read the corresponding audit section.)*

---

## Appendix B — How to verify this document

A reader who wants to confirm that this document accurately describes project state at v10.11.10 should do the following, in order:

1. `git checkout 64ca18ea` (or the SHA cited at the top of the file).
2. Run every `grep`/`sed`/`awk` command in §2's "How to re-verify this grid" block.
3. Compare each output to the corresponding "Current state at v10.11.10" cell in the grid.
4. For each discrepancy, report it — that is a regression and is itself useful data.

If §2 is reproducible, the rest of the document inherits credibility from it. If §2 is not reproducible, the rest of the document should be treated as marketing.

---

## Appendix C — Related documents

- `papers/skin-in-the-cathedral-2026.tex` — design philosophy ("operational dullness as competitive advantage").
- `papers/five-mirrors-2026.tex` — the Pacioli analogy in its full form.
- `CLAUDE.md` — operator runbook (acknowledged in row 12 of §2; corrected 2026-05-21).
- `docs/branch-hygiene-rules-draft-2026-05-19.md` — earlier example of a per-finding tabular response.
- `docs/technical-review-sparse-chain-truth-v1.md` — earlier example of honest-state reporting about historical chain damage.

---

*End of document. Next scheduled honest-state review: 2026-08-22.*
