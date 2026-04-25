# Dynamic DCA & Auto-Swap Trading Bot — Design Document v1
## Quillon DEX Enhancement Proposal (April 2026)

---

## 1. Context & Motivation

### Physics Inspiration: LHCb Ξcc⁺ Discovery (March 2026)

The LHCb experiment discovered the doubly-charmed baryon Ξcc⁺ (two charm quarks + one down quark) at 3620 MeV/c² with 7σ significance. This particle is 4x heavier than a proton with a lifetime 6x shorter — a direct consequence of QCD binding energy dynamics at different mass scales.

**Mathematical insight for DCA**: Just as the strong force coupling constant αs *runs* (changes strength) at different energy scales — weak at high energy (asymptotic freedom), strong at low energy (confinement) — our DCA algorithm should have a **running coupling** between buy amount and pool depth:

```
α_DCA(L) = α₀ × (L₀ / L)^β

Where:
  L  = current pool liquidity (total reserves in USD)
  L₀ = reference liquidity ($100K baseline)
  β  = running exponent (0.3 for gentle adaptation, 0.7 for aggressive)
  α₀ = base allocation (% of DCA budget per interval)
```

This "renormalization group" approach means: when liquidity is deep (high energy analogy), we can place larger orders with minimal impact; when liquidity is thin (low energy/confinement), we must use tiny, frequent trades.

### FCC-ee Collision Dynamics as Pool Model

The proposed Future Circular Collider (FCC-ee) uses precisely tuned electron-positron beam energies to maximize production cross-sections at specific resonance peaks (Z, W, H, tt̄). This is mathematically equivalent to finding the **optimal swap size** that maximizes output tokens per unit price impact:

```
Optimal swap = argmax_s [ output(s) / impact(s) ]

For constant-product AMM:
  output(s)  = s × R_y / (R_x + s)
  impact(s)  = s / (R_x + s)
  optimal_s  = √(R_x × R_y) × ε   where ε = max acceptable impact (e.g., 1%)
```

---

## 2. Current State (What Exists)

| Component | Status | File |
|-----------|--------|------|
| AMM Swap (x×y=k) | Working | `crates/q-dex/src/trading.rs` |
| Pool Management | Working | `crates/q-dex/src/liquidity.rs` |
| DCA Strategy | **STUB ONLY** (returns Hold) | `crates/q-trading-bot/src/strategies/dca.rs` |
| Grid Trading | Working | `crates/q-trading-bot/src/strategies/grid.rs` |
| DEX Activity Bot | Working | `crates/q-trading-bot/src/strategies/dex_activity.rs` |
| Oracle Prices | Working (Binance fallback) | `crates/q-dex/src/oracle_price_bridge.rs` |
| Swap API | Working | `POST /api/v1/dex/execute_swap` |

**Key gap**: DCA strategy is completely unimplemented — always returns `TradingSignal::Hold`.

---

## 3. Dynamic DCA Algorithm Design

### 3.1 Core Formula: Volatility-Adaptive Sizing

```
swap_amount = min(
    budget_per_interval × volatility_dampener × depth_scaler,
    max_impact_amount
)

Where:
  budget_per_interval = total_dca_budget / num_intervals
  
  volatility_dampener = 1.0 / (1.0 + σ_price / σ_target)
    σ_price  = rolling 24h price standard deviation (from OHLCV candles)
    σ_target = target volatility threshold (e.g., 5%)
  
  depth_scaler = min(1.0, pool_reserves / min_safe_reserves)
    pool_reserves   = sqrt(reserve_x × reserve_y) in USD
    min_safe_reserves = 10 × swap_amount (ensures <1% impact)
  
  max_impact_amount = reserve_x × max_price_impact / (1 + max_price_impact)
    max_price_impact = 0.01 (1% maximum)
```

### 3.2 Adaptive Interval (ATR-Based)

```
interval_seconds = base_interval × (1.0 + ATR_ratio)

Where:
  base_interval = 3600 (1 hour default)
  ATR_ratio     = ATR_24h / current_price
  
  Clamped to: [300s, 86400s] (5 min to 24 hours)
  
  High volatility → shorter intervals, smaller amounts (more frequent nibbles)
  Low volatility  → longer intervals, larger amounts (fewer, bigger bites)
```

### 3.3 Smart Order Splitting

When swap amount exceeds 0.5% of pool reserves, split into N sub-swaps:

```
N = ceil(swap_amount / (reserve_x × 0.005))

For each sub-swap i = 0..N-1:
  sub_amount = swap_amount / N
  delay      = i × 10 seconds  (spread across time to allow arb rebalancing)
```

### 3.4 Circuit Breakers

| Condition | Action |
|-----------|--------|
| Price moved >10% in 1 hour | Pause DCA for 4 hours |
| Slippage exceeds 2% on execution | Skip this interval, retry next |
| Pool reserves < $1,000 | Halt DCA for this pair |
| Cumulative loss > 5% of budget | Stop DCA, alert user |
| 3 consecutive failed swaps | Pause 1 hour, then retry |

---

## 4. Auto-Swap Mining Revenue Bot

### 4.1 Architecture: Off-Chain Bot with On-Chain Execution

```
┌─────────────────────────────────────────────┐
│           Mining Revenue Bot (Rust)          │
│                                              │
│  ┌──────────┐  ┌──────────┐  ┌───────────┐ │
│  │ Balance   │  │ Strategy │  │ Risk      │ │
│  │ Monitor   │→ │ Engine   │→ │ Manager   │ │
│  └──────────┘  └──────────┘  └───────────┘ │
│       │              │              │        │
│       ▼              ▼              ▼        │
│  Check QUG    Calculate      Validate       │
│  balance      optimal        slippage       │
│  every 60s    allocation     & limits       │
│                                              │
│              ┌──────────┐                    │
│              │ Executor  │ ──→ POST /api/v1/ │
│              │ (Swap)    │     dex/execute_  │
│              └──────────┘     swap           │
└─────────────────────────────────────────────┘
```

### 4.2 Allocation Strategy

User configures target portfolio percentages:

```rust
struct SwapAllocation {
    targets: HashMap<String, f64>,  // token_symbol → percentage
    // Example:
    // "wBTC"   → 0.40  (40% of mining revenue → Bitcoin)
    // "wETH"   → 0.30  (30% → Ethereum)
    // "QUGUSD" → 0.20  (20% → Stablecoin)
    // "QUG"    → 0.10  (10% → Hold as QUG)
}
```

### 4.3 Execution Flow

```
Every sweep_interval (default: 1 hour):
  1. Check QUG balance
  2. If balance > min_sweep_amount (default: 100 QUG):
     a. For each target token by allocation %:
        - Calculate swap_amount = balance × allocation%
        - Query pool for expected output + slippage
        - If slippage < max_slippage (1%):
            Execute swap via /api/v1/dex/execute_swap
        - Else:
            Split into smaller sub-swaps or defer
  3. Log all executions for tax/audit trail
  4. Update running P&L tracker
```

### 4.4 Multi-Hop Routing

For pairs without direct pools (e.g., QUG→wBTC), route through intermediate:

```
QUG → QUGUSD → wBTC  (via two pools)

Route selection:
  For each possible path P:
    expected_output_P = simulate_multi_hop(amount, path_P)
    total_fees_P      = sum(pool_fee for pool in path_P)
    effective_rate_P   = expected_output_P / (amount × (1 + total_fees_P))
  
  Select path with highest effective_rate
```

---

## 5. Implementation Plan

### Phase A: DCA Engine (Week 1)

**Files to modify:**
- `crates/q-trading-bot/src/strategies/dca.rs` — Replace stub with full implementation
- `crates/q-trading-bot/src/config.rs` — Add DCA config fields
- `crates/q-api-server/src/handlers.rs` — Add DCA management endpoints

**New endpoints:**
```
POST /api/v1/dca/create    — Create DCA schedule
GET  /api/v1/dca/status     — Get active DCA status
POST /api/v1/dca/pause      — Pause/resume DCA
DELETE /api/v1/dca/cancel   — Cancel DCA schedule
GET  /api/v1/dca/history    — Execution history
```

### Phase B: Mining Revenue Bot (Week 2)

**New files:**
- `crates/q-trading-bot/src/strategies/mining_sweeper.rs` — Auto-swap mined QUG
- `crates/q-trading-bot/src/router.rs` — Multi-hop route optimizer

### Phase C: Frontend Integration (Week 3)

**UI components:**
- DCA setup modal (token pair, budget, interval, max slippage)
- Mining revenue allocation slider (% per target coin)
- Active DCA dashboard with execution history chart
- P&L tracking with cost basis calculation

---

## 6. Gemma4 Advisory (Epsilon Ollama, April 2026)

The following advice was obtained from Gemma4 (8B, Q4_K_M) on Epsilon:

> **Optimal DCA sizing**: Cap swap at 5% of pool liquidity per interval. Use `Swap = RiskCapital × min(MaxDepth/CurrentLiquidity, VolatilityDampeningFactor)`.
>
> **Slippage minimization**: Implement Adaptive Multi-Hop Swaps. If expected slippage exceeds tolerance ε, break into N smaller hops. Formula: `Out = Input × Reserve_y / (Reserve_x + Input)`.
>
> **Architecture**: Off-chain bot with on-chain execution. Complex stateful logic (historical analysis, route calculation, risk) runs off-chain; only the final optimized swap transaction hits the blockchain.
>
> **Smart order routing**: Query all pools for expected output, select pool minimizing `Effective Slippage = (Input - Output) / Input`. Prioritize highest TVL pools.
>
> **Circuit breakers**: Time-based (halt if DEX uptime issues) + Loss-based (halt if cumulative loss exceeds L_max in 24h). Max slippage tolerance 0.5%. Position limit 10% of capital per pair.

---

## 7. Questions for DeepSeek Enhancement

The following questions should be presented to DeepSeek for deeper mathematical optimization:

### 7.1 Optimal Execution Theory
> Given a constant-product AMM with reserves (R_x, R_y) and a DCA target of T tokens over N intervals, derive the closed-form solution for the optimal swap schedule {s_1, s_2, ..., s_N} that minimizes total price impact while achieving exactly T tokens. Consider that each swap changes the reserves for subsequent swaps. How does the optimal schedule change when pool reserves are stochastic (random LP additions/removals)?

### 7.2 Volatility-Adjusted Kelly Criterion for DCA
> Can the Kelly Criterion be adapted for DCA sizing on AMM pools? Specifically: if the token price follows geometric Brownian motion with drift μ and volatility σ, and the AMM pool has depth D, what is the optimal fraction f* of capital to deploy per DCA interval that maximizes long-run geometric growth rate? How does this differ from traditional Kelly when price impact is non-zero?

### 7.3 MEV-Resistant DCA
> Our blockchain has a DAG-Knight consensus (not sequential blocks). How can we design DCA executions that are resistant to sandwich attacks in a DAG topology? In a DAG, transactions can be partially ordered rather than totally ordered. Does this inherently provide MEV resistance, or do we need additional mechanisms like commit-reveal schemes?

### 7.4 Renormalization Group Approach to Multi-Scale Liquidity
> Drawing from the LHCb Ξcc⁺ discovery where QCD coupling αs runs with energy scale, can we formalize a "renormalization group equation" for AMM pool behavior across different liquidity scales? Specifically: define β(L) as the rate of change of optimal DCA allocation with respect to pool liquidity L. What is the fixed point of this β-function, and does it correspond to a market equilibrium?

### 7.5 Genus-2 Curve VDF as Execution Timing Oracle
> Our blockchain uses a Genus-2 hyperelliptic curve VDF for consensus timing. Can the VDF output (which is deterministic but unpredictable) be used as a fair, manipulation-resistant timing oracle for DCA executions? This would make the DCA execution times unpredictable to front-runners while being verifiable post-execution.

### 7.6 Cross-Pool Arbitrage-Aware DCA
> When our bot executes a DCA swap on Pool A, it creates an arbitrage opportunity between Pool A and Pool B (same pair, different reserves). How should the DCA engine account for expected arbitrageur behavior when planning swap sizes? Model: after our swap of size s on Pool A, an arbitrageur will rebalance pools within τ seconds. Derive the net effective price including arbitrage rebalancing.

### 7.7 FCC-ee Resonance Peak Analogy for Optimal Swap Timing
> The FCC-ee collider tunes beam energy to resonance peaks (Z=91.2 GeV, W=160 GeV, H=125 GeV) for maximum cross-section. By analogy, are there "resonance" points in AMM pool dynamics where swap efficiency peaks? For instance, when the pool ratio returns to a 1:1 value after a large trade, is there a mathematically optimal moment to execute a DCA swap?

---

## 8. Appendix: Key File References

| File | Purpose |
|------|---------|
| `crates/q-dex/src/trading.rs` | AMM swap execution (x×y=k) |
| `crates/q-dex/src/liquidity.rs` | Pool management, reserves |
| `crates/q-trading-bot/src/strategies/dca.rs` | DCA stub (to be replaced) |
| `crates/q-trading-bot/src/strategies/grid.rs` | Grid trading (reference impl) |
| `crates/q-trading-bot/src/engine.rs` | Strategy execution loop |
| `crates/q-api-server/src/handlers.rs` | Swap API endpoint (~line 10196) |
| `crates/q-dex/src/oracle_price_bridge.rs` | External price feeds |
| `crates/q-storage/src/price_history.rs` | OHLCV candle data |

---

*Generated April 15, 2026 — Quillon Foundation*
*Physics inspiration: LHCb Ξcc⁺ (CERN, March 2026) + FCC-ee proposal (Budapest decision May 2026)*
