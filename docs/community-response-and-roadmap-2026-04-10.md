# Community Response & Network Improvement Plan (Revised)
**Date:** 2026-04-10 (Revised after AI peer review)

---

## Part 1: Community Replies (Simplified)

### To 九九艳阳天 — "我3000个变成800个了"

> 理解您的担心。这是**钱包前端显示异常**，不是链上余额回档。您的真实余额没有被修改。我们已经在 v10.2.9 修复了这个显示问题，请先硬刷新（Ctrl+Shift+R）。如果刷新后仍不正确，请把钱包地址发给我们，我们立即做链上核对。

> We understand your concern. This was a **wallet display issue**, not a balance rollback on-chain. Your actual balance was never modified. We've fixed this in v10.2.9. Please hard refresh (Ctrl+Shift+R). If still incorrect, share your wallet address and we'll verify on-chain immediately.

### To jooosen — "钱包数据一直异常变化"

> 余额波动是前端显示bug，已修复。区块链上的余额从始至终是正确的。我们增加了多层保护，防止类似问题再次发生。

> The balance fluctuations were a frontend display bug, now fixed. Your on-chain balance was correct throughout. We've added multiple layers of protection to prevent this from recurring.

### To yyds — "用的小霸王服务器吧？"

> 主服务器是48核64GB内存10Gbit专线服务器。问题是软件bug不是硬件。过去48小时已部署15+个修复，网络在每次事件中都在加强。

> The primary server is a 48-core, 64GB RAM, 10Gbit dedicated machine. The issues were software bugs, not hardware. 15+ fixes deployed in the past 48 hours — the network is getting stronger with each incident.

### To MOJO NATIONS — Exchange listing timing

> You make an excellent point about network maturity. We agree completely. The stability fixes this week are exactly the hardening needed before listing. We are in active discussions with an exchange but listing will only proceed after reliability milestones are met. We will not rush a premature listing.

> 您说得完全正确，关于网络成熟度。本周的稳定性修复正是上市前需要的加固工作。我们正在与交易所积极讨论，但只有在可靠性里程碑达标后才会进行上市。我们不会仓促上市。

### To Polar Point — "可以连接metamask钱包吗"

> 目前不行。QUG是原生L1区块链，有自己的地址格式。MetaMask需要EVM兼容桥接，计划在交易所上市后开发。目前优先：网络稳定性 → 交易所上市 → 安卓钱包 → MetaMask。

> Not yet. QUG is a native L1 blockchain with its own address format. MetaMask requires an EVM-compatible bridge, planned after exchange listing. Current priority: network stability → exchange listing → Android wallet → MetaMask.

---

## Part 2: Gated Improvement Roadmap

### Gate A — Chain Reliability (CURRENT — target: 7 days zero stalls)

**What's done (v10.2.9):**
- [x] Height atomic stall fix (fetch_max)
- [x] Block-pack I/O early abort
- [x] Corrupt block log demotion
- [x] q-flux connection pileup fix
- [x] Producer task 30s timeout
- [x] Zombie flag detection (120s force-clear)
- [x] Crown-Ash block_in_place

**Still needed:**
- [ ] `spawn_blocking` for RocksDB calls inside produce_block() (prevents tokio starvation)
- [ ] Per-stage timing telemetry in produce_block() (queue wait → mempool → state read → assembly → VDF → DB commit → broadcast)
- [ ] Tokio scheduler health watchdog (detect worker thread starvation)
- [ ] "Last successful block" monotonic timestamp metric
- [ ] 7 consecutive days with zero unexplained stalls
- [ ] Controlled chaos test: disk slowdown + peer churn + compaction overlap

**Exit criteria:** 7 days continuous operation, no stalls >2 block intervals, no manual intervention needed.

### Gate B — Sync & Bootstrap Integrity

**Current state:** New nodes cannot sync from genesis (blocks 0-6M missing from all peers).

**Plan:**
- [ ] Fix contiguous-height pointer in transaction.rs (only advance when block N exists AND N-1 exists)
- [ ] Implement signed checkpoint snapshots (every 500K blocks)
- [ ] Checkpoint = {height, block_hash, state_root, signature}
- [ ] New node flow: fetch checkpoint → download snapshot → verify signature → sync forward from checkpoint
- [ ] Maintain explicit "validated contiguous tip" separate from "highest known height"
- [ ] A node must never report "synced" based only on a high-water mark — must prove contiguity

**Exit criteria:** A fresh node can sync to tip within 6 hours using checkpoint + P2P.

### Gate C — Exchange Integration (HiBT)

**Current state:** HiBT CEO Ryan says API integration is already done on their side. Need to verify what that means and complete our side.

**Verify with HiBT:**
- [ ] What exactly is "done"? Did they build a QUG node integration, or do they mean they're ready to start?
- [ ] Do they need us to run a dedicated node for them?
- [ ] What RPC/API format do they expect? (JSON-RPC 2.0, REST, custom?)
- [ ] Have they tested deposit detection and withdrawal signing?

**Our side (if not already done by HiBT):**
- [ ] Deposit address generation (deterministic HD wallet per user)
- [ ] Block scanning for deposit detection
- [ ] Withdrawal creation + HSM-backed signing
- [ ] Confirmation depth policy (start conservative: 30 confirmations at current block time)
- [ ] Reorg handling + reconciliation
- [ ] Idempotent withdrawal request IDs
- [ ] Health/version endpoint

**If Ryan's claim is accurate**, our remaining work is:
- [ ] Run a dedicated node for HiBT (stable, always-on)
- [ ] Provide API documentation
- [ ] Coordinate testnet dry-run
- [ ] Legal review sign-off
- [ ] Security deposit transfer

**Exit criteria:** Successful test deposit + test withdrawal on staging environment.

### Gate D — Listing Go-Live

- [ ] Gate A passed (7 days zero stalls)
- [ ] Gate B passed (sync works) OR HiBT runs our node with checkpoint
- [ ] Gate C passed (test trades successful)
- [ ] Legal review complete (jurisdiction, penalty clauses resolved)
- [ ] Incident response contacts established with HiBT
- [ ] Go-live checklist signed off by both parties
- [ ] Community announcement (24h advance notice)

**Timeline estimate:** 2-4 weeks from Gate A completion, assuming HiBT's API integration claim holds.

---

## Part 3: Questions for AI Reviewers

### For DeepSeek/ChatGPT Technical Review:

1. **spawn_blocking refactor**: We need to isolate RocksDB calls and VDF computation from the tokio async runtime inside `produce_block()`. What's the recommended pattern for splitting an async function into "async orchestration" + "spawn_blocking islands"? Our produce_block() does: queue drain → mempool snapshot → DAG-Knight consensus check → VDF verification → block assembly → RocksDB save → P2P broadcast.

2. **Checkpoint sync design**: We want to implement signed checkpoint snapshots for fast node bootstrap. What's the minimal trusted checkpoint format? Should we use RocksDB SST file export, or a custom binary snapshot? How do other L1 blockchains (Bitcoin, Kaspa, Monero) handle checkpoint-based bootstrapping?

3. **Exchange API for native L1**: HiBT says their API integration is "already done." For a native L1 blockchain (not ERC-20), what does "done" typically mean from the exchange side? What should we verify before trusting that claim?

4. **Wallet UX confidence**: Our users panic when balances flicker. What specific UI patterns should we implement? We're considering: confirmed/pending separation, block height display, sync status indicator, and "last updated N seconds ago" label.

5. **Network topology**: We have 4 nodes (Epsilon=10Gbit supernode, Beta/Gamma/Delta=1Gbit). Epsilon handles 80%+ of traffic. How should we redistribute to avoid single-point-of-failure before exchange listing?

---

## Part 4: HiBT Listing Status (Updated)

**Stage:** Active negotiation — CEO Ryan claims API integration done

**Key fact:** HiBT CEO Ryan joined the Telegram group, requested binary files, API docs, and whitepaper. He claims API integration is already complete on their side.

**What we need to verify:**
1. Does "API integration done" mean they built a QUG node wrapper, or just that their listing framework is ready?
2. Have they tested with our actual mainnet?
3. Do they need a dedicated node from us?
4. What confirmation depth are they using?

**Pricing:** 15,000 USDT (Standard Tier)

**Legal concerns (from TECHNICAL-LEGAL-REVIEW.md):**
- Jurisdiction triangle (Canada entity, UK governing law, SE Asian contact)
- 70% security deposit forfeiture clause — needs negotiation
- Exclusive trading obligation — needs removal or time-limiting
- IP licensing scope — too broad, needs narrowing
- Need independent legal counsel before signing

**Immediate next steps:**
1. Ask Ryan exactly what "API integration done" means
2. Provide him a dedicated stable node endpoint
3. Request a test deposit + test withdrawal on their staging
4. Finalize legal review (hire crypto lawyer if needed)
5. Sign agreement only after Gate A is passed
