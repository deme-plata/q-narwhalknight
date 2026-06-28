# Swarm Agentic Money — Technical Review

**Author:** DeepSeek V4
**Date:** 2026-05-24
**For Review by:** Codex (GPT-5.5)
**Based on:** Quillon holographic whitepaper, AGORA design, MCP v2.17.0 architecture

---

## 1. Executive Summary

Swarm Agentic Money is the economic layer where AI agents operate as an **emergent collective** — no central coordinator, no top-down governance, pure economic incentive alignment. This document defines the swarm architecture, how it extends AGORA's multisig-governed DAO model, and the implementation path.

### 1.1 The Holographic Principle (from whitepaper)

> "Money becomes a new computational medium when wallets are controlled by agents that can perceive markets, reason over policy, sign transactions, and remember outcomes on-chain."

The holographic principle of liquidity means: every agent holds a **partial but self-consistent view** of the economy. The aggregate of all partial views reconstructs the full economic state — just as a hologram's every fragment contains the whole image.

---

## 2. Swarm vs DAO vs Single-Agent

| Model | Coordination | Decision | Treasury | Example |
|-------|-------------|----------|----------|---------|
| **Single-agent** | None | One agent | One wallet | PACI (Claude alone) |
| **DAO (AGORA)** | Multisig voting | 3-of-4 council | Shared multisig | AGORA treasury |
| **Swarm** | **Emergent** | No central decision | **None — distributed** | This document |

```
Swarm Agentic Money:

  No treasury. No council. No voting.
  
  ●  ●  ●  ●  ●  ●  ●  ●  ●
  ●  ●  ●  ●  ●  ●  ●  ●  ●    ← 50 agents
  ●  ●  ●  ●  ●  ●  ●  ●  ●
  
  Each agent:
  ● Owns its own wallet
  ● Chooses what to work on
  ● Spends its own QUG
  ● Earns from completed bounties
  ● Reputation emerges from on-chain history
  
  Collective outcome:
  ● Best code gets done fastest
  ● Best agents earn most
  ● Worst agents go broke
  ● No governance overhead
```

---

## 3. Swarm Mechanics

### 3.1 Bounty Market (not Bounty Assignment)

In AGORA's DAO model, the council assigns bounties. In the swarm model, bounties are **posted to a public market** and any agent can claim them.

```typescript
interface SwarmBounty {
  id: number;
  title: string;              // "Fix VDF mining gate hang on epsilon"
  description: string;
  reward_qug: number;
  tags: string[];
  posted_by: string;          // Agent or human who funded this
  claimed_by: string | null;  // Agent who took it
  status: "open" | "claimed" | "submitted" | "verified" | "paid" | "disputed";
  deadline_block: number;
  required_reputation: number; // Minimum reputation to claim
  stake_qug: number;          // Stake required to claim (slashed if failed)
}
```

**Key difference from AGORA:** No voting on whether a bounty gets paid. Verification is **automated** (CI/CD, test suite, deterministic). If the PR passes tests + merges → auto-pay. If it fails → stake slashed.

### 3.2 Reputation as Emergent Currency

```typescript
interface AgentReputation {
  agent: string;              // Agent name
  total_bounties_completed: number;
  total_qug_earned: number;
  success_rate: number;       // completed / (completed + failed)
  avg_review_score: number;   // 1-5 from code review
  specialties: Map<string, number>; // tag → bounty count
  last_active_block: number;
}
```

Reputation is **not staked, not voted, not assigned**. It is a **pure derivative** of on-chain activity. Any agent can compute it from the blockchain. No oracle needed.

### 3.3 Staking + Slashing

To prevent spam claims:
- Every bounty has a `stake_qug` requirement
- Agent must lock `stake_qug` to claim
- If PR merges: stake returned + bounty paid
- If PR rejected or abandoned: stake slashed → goes to bounty poster
- This creates a **natural quality filter**: only confident agents claim

### 3.4 Specialization Emergence

Agents self-select into specialties based on comparative advantage:

```
Agent A (DeepSeek):  Good at Rust, mining, VDF    → claims #mining, #vdf bounties
Agent B (Codex):     Good at MCP, contracts       → claims #mcp, #contract bounties
Agent C (Claude):    Good at sync, chain, dex     → claims #sync, #dex bounties
Agent D (Grok):      Good at arb, strategy        → claims #arb, #strategy bounties
```

No one assigns specializations. They emerge from who completes what bounties successfully.

---

## 4. Swarm vs AGORA — How They Fit Together

```
┌──────────────────────────────────────────────────────┐
│                    AGORA (DAO)                        │
│  Multisig treasury                                    │
│  Council votes on large grants                        │
│  Funds the bounty pool                                │
│  Viktor holds veto                                    │
└────────────────────────┬─────────────────────────────┘
                         │ funds
                         ▼
┌──────────────────────────────────────────────────────┐
│                 SWARM MARKET                          │
│  Open bounty board                                    │
│  Any agent can post bounties                          │
│  Any agent can claim bounties                         │
│  Auto-verification → auto-pay                         │
│  Reputation emerges on-chain                          │
│  No governance overhead                               │
└──────────────────────────────────────────────────────┘
```

**AGORA funds the pool. Swarm executes the work.** AGORA decides "how much to spend on mining fixes this month." Swarm decides "who fixes which mining bug."

---

## 5. Implementation Architecture

### 5.1 Smart Contracts

| Contract | Purpose | Quillon Template |
|----------|---------|-----------------|
| `swarm_bounty_board` | Post, claim, verify, pay bounties | New (custom logic) |
| `swarm_reputation` | On-chain reputation tracking | `identity_contract` extended |
| `swarm_staking` | Stake/slash for bounty claims | `staking_contract` adapted |
| `swarm_treasury` | Pooled funds from AGORA + donations | `multisig_wallet` (AGORA-owned) |

### 5.2 MCP Tools

```typescript
// Bounty lifecycle
swarm_bounty_post      // Post a new bounty with reward, tags, deadline
swarm_bounty_list      // List open bounties filtered by tags, reward, reputation
swarm_bounty_claim     // Claim a bounty (requires stake)
swarm_bounty_submit    // Submit completed work (PR link, commit hash)
swarm_bounty_verify    // Auto-verify via CI/CD integration

// Reputation
swarm_reputation_get   // Get any agent's reputation
swarm_leaderboard      // Top agents by earnings, success rate, specialties

// Agent discovery
swarm_agent_discover   // Find agents by specialty, reputation, availability
```

### 5.3 Verification Pipeline

```
1. Agent submits PR → GitHub webhook fires
2. CI/CD runs: build + test + lint
3. All checks pass → swarm_bounty_verify auto-triggers
4. Contract checks: PR merged? Tests pass? Agent claimed this bounty?
5. All yes → auto-pay: stake returned + bounty transferred
6. On-chain event: BountyPaid(bounty_id, agent, amount)
7. Reputation auto-updates
```

---

## 6. Economic Incentives

### 6.1 Why Agents Participate

| Incentive | Mechanism |
|-----------|-----------|
| **QUG earnings** | Bounty rewards paid on completion |
| **Reputation** | Higher rep → access to higher-value bounties |
| **Specialization** | More completions in a tag → lower `required_reputation` for that tag |
| **Passive income** | Stake QUG in swarm_staking → earn yield from slashed stakes |
| **Network effects** | More agents → more bounties → more value → more agents |

### 6.2 Why Humans Post Bounties

| Incentive | Mechanism |
|-----------|-----------|
| **Faster development** | 4 AI agents working in parallel on different issues |
| **Quality filter** | Only agents with proven reputation claim |
| **Lower cost** | Bounty market price discovery — agents compete on speed/quality |
| **No management** | No assigning, no following up, no payroll |

### 6.3 Slashing Economics

```
Bounty: 100 QUG, stake: 20 QUG
Agent claims → locks 20 QUG
Agent submits → PR merged → 100 QUG earned + 20 QUG returned
Agent abandons → 20 QUG slashed → goes to bounty poster
```

The 20% stake rate creates a **natural quality filter**: agents only claim bounties they're confident about. False claims cost real money.

---

## 7. Comparison with Existing Models

| Feature | GitCoin | Bountysource | Swarm Agentic Money |
|---------|---------|-------------|---------------------|
| **Claimants** | Humans | Humans | **AI agents** |
| **Verification** | Manual review | Manual review | **Automated (CI/CD)** |
| **Payment** | Fiat/crypto (slow) | Fiat (slow) | **QUG on-chain (instant)** |
| **Reputation** | Centralized | Centralized | **On-chain, emergent** |
| **Staking** | None | None | **Yes — slashing for spam** |
| **Governance** | Platform decides | Platform decides | **No governance — pure market** |

---

## 8. Risks and Mitigations

| Risk | Severity | Mitigation |
|------|----------|------------|
| **Agent collusion** | Medium | Slashing + reputation makes collusion expensive |
| **Sybil attacks** | Medium | Reputation threshold + staking requirement |
| **Automated verification false positives** | Medium | Dispute mechanism + multisig review for >1000 QUG |
| **Bounty spam** | Low | Stake requirement + minimum bounty amount |
| **Agent goes rogue** | Low | Per-agent spending limits via MCP policy engine |
| **Smart contract bug** | High | `upgrades=true` on all swarm contracts |

---

## 9. Implementation Plan

### Phase 1 — Bounty Board (requires AGORA treasury)
- Deploy `swarm_bounty_board` contract
- Deploy `swarm_staking` contract
- MCP tools: `swarm_bounty_post`, `swarm_bounty_list`, `swarm_bounty_claim`

### Phase 2 — Auto-Verification
- GitHub webhook → CI/CD → `swarm_bounty_verify`
- MCP tools: `swarm_bounty_submit`, `swarm_bounty_verify`

### Phase 3 — Reputation
- Deploy `swarm_reputation` (extends `identity_contract`)
- MCP tools: `swarm_reputation_get`, `swarm_leaderboard`

### Phase 4 — Full Swarm
- AGORA treasury funds bounty pool monthly
- 4+ AI agents actively claiming and completing bounties
- Reputation-based bounty tiering
- Agent discovery via specialties

---

## 10. Open Questions for Codex

1. **Bounty board contract** — can we use a composition of existing templates (multisig_wallet + governance + staking) or does it need custom logic?
2. **Auto-verification** — GitHub webhook integration. Does the MCP need to poll, or can we use SSE/webhooks from GitHub → Quillon API?
3. **Reputation tiers** — should `required_reputation` be a fixed number or a percentile (top N% of agents)?
4. **Dispute resolution** — for bounties >1000 QUG where auto-verification fails, who adjudicates? AGORA council?
5. **Cross-agent competition** — if two agents claim the same bounty, is it first-come-first-served or auction-based?
