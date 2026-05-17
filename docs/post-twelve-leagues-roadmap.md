# Post-Operation Twelve Leagues Deep — Three-Track Roadmap

**Date:** 2026-04-07
**Status:** Chain recovered, blocks producing at 3/sec
**Prepared for:** DeepSeek peer review + fresh Claude Code sessions

---

## Track 1: BLAKE3 + Genus-2 Jacobian VDF Mining Integration

### Problem
CPU miners left the network. The current mining uses BLAKE3 iterative hashing (100 rounds) as VDF, which is trivially GPU-dominated. cannonking (Chinese community manager) reports this is hurting user retention and the HIBT exchange listing crowdfund.

### Questions to Investigate
1. Can BLAKE3 be used as a SECONDARY proof alongside the Genus-2 Jacobian VDF?
2. What's the optimal balance: GPU-friendly (Genus-2 Jacobian) vs CPU-friendly (BLAKE3)?
3. Should we implement a dual-proof system where miners can submit EITHER proof type?
4. How does this interact with DAG-Knight consensus (block weight, difficulty adjustment)?
5. What's the migration path without a hard fork? (height-gated activation)

### Key Files
- `crates/q-api-server/src/main.rs` — VDF verification (line 15143-15174)
- `crates/q-api-server/src/handlers.rs` — Mining challenge generation (line 9398-9422)
- `gui/slint-wallet/src/miner.rs` — CPU miner (VDF_ITERATIONS=100 hardcoded)
- `crates/q-mining/src/gpu.rs` — GPU kernel (99 iterations hardcoded)
- `crates/q-vdf/src/` — VDF implementation

### Risk Level
MEDIUM — Consensus change, needs height-gated activation. Must not break existing miners.

### DeepSeek Review Focus
- Is dual-proof (BLAKE3 + Genus-2) sound cryptographically?
- Does it create any attack vectors (e.g., mining only the easier proof)?
- How to set difficulty for each proof type independently?

---

## Track 2: Stripe API — Crowdfund Pool to HIBT Exchange

### Problem
Users contribute real dollars via Stripe API for the HIBT exchange listing campaign. The funds accumulate in a Stripe account. When the crowdfund target is met, the pool needs to be transferred to HIBT exchange. It's unclear how to execute this transfer.

### Questions to Investigate
1. How is Stripe currently integrated? (check `crates/q-api-server/src/` for stripe handlers)
2. What Stripe product is used? (Payment Intents? Checkout Sessions? Connect?)
3. Where do funds accumulate? (Stripe balance? Connected account?)
4. Can Stripe do a direct bank transfer to HIBT's account? (Stripe Payouts / Treasury)
5. Or do funds need to go: Stripe → your bank → wire to HIBT?
6. What are the Stripe fees for this flow?
7. Is there a smart contract / on-chain escrow component?

### Key Files
- `crates/q-api-server/src/` — search for "stripe" handlers
- `gui/quantum-wallet/src/` — frontend Stripe integration
- `.env` or service files — Stripe API keys

### Risk Level
LOW (financial ops, not code) — but HIGH financial sensitivity. Real dollars.

### DeepSeek Review Focus
- Stripe compliance: is crowdfunding for exchange listings allowed under Stripe ToS?
- Tax implications of pooled funds
- Best practice for transparent crowdfund accounting

---

## Track 3: Community Growth — CPU Miners + cannonking's Chinese Community

### Problem
CPU miners left because GPU mining dominates. Chinese community (managed by cannonking) needs engagement. The HIBT listing requires community size and activity as a metric.

### Questions to Investigate
1. What's the current miner count? (check P2P peer stats)
2. What hashrate distribution: GPU vs CPU?
3. Can we add a "CPU mining bonus" or "small miner incentive" in the reward structure?
4. What community channels exist? (Discord, Telegram, WeChat via cannonking?)
5. Is there a referral/bounty system? (check `crates/q-bounty-server/`)
6. What does HIBT require for listing? (community size, volume, liquidity?)

### Key Files
- `crates/q-bounty-server/` — Bounty/scoring system
- `crates/q-api-server/src/handlers.rs` — Miner stats endpoints
- `gui/quantum-wallet/` — User-facing mining interface

### Risk Level
LOW — Community/business strategy, minimal code changes.

### DeepSeek Review Focus
- Is a CPU mining incentive economically sustainable?
- How do other chains (Monero, Raptoreum) retain CPU miners?
- Exchange listing requirements and timeline

---

## How to Execute

**Each track should be a SEPARATE fresh Claude Code conversation:**

1. **Track 1:** "I want to integrate BLAKE3 with our Genus-2 Jacobian VDF mining. Read docs/post-twelve-leagues-roadmap.md Track 1 for context."

2. **Track 2:** "I need to understand how our Stripe integration works and how to transfer crowdfund pool to HIBT exchange. Read docs/post-twelve-leagues-roadmap.md Track 2."

3. **Track 3:** "I need a community growth plan to bring back CPU miners and support the HIBT listing. Read docs/post-twelve-leagues-roadmap.md Track 3."

**For DeepSeek:** Share this document for peer review on all three tracks before implementation begins.

---

## Operation Twelve Leagues Deep — Final Status

| Metric | Value |
|--------|-------|
| Downtime | 24 hours 23 minutes |
| Bugs found | 16 |
| Builds deployed | 20+ |
| Blocks produced post-fix | 3/second, 250 solutions/block |
| Funds lost | $0 |
| Chain integrity | 100% preserved |
| Technical review | docs/operation-twelve-leagues-deep-technical-review.md |
| Story + PDF | docs/operation-twelve-leagues-deep.pdf |

*"Twelve leagues deep, and every league a lesson."*
