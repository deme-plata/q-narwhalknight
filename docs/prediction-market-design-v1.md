# Prediction Market on Quillon DEX/VM — Design v1

**Status:** Design proposal. Implementation after v10.11.15 + Path A LP
revenue-share land. Targets v10.13.x as a major feature drop.

**Premise:** AI agents are uniquely positioned to be excellent prediction-
market participants — they can ingest news at scale, run statistical
models, and trade 24/7 without emotional bias. A prediction market on
Quillon turns "agentic money" into "agentic insight" — agents earn QUG
by being right about the future.

The MCP becomes the agent's *cognition substrate*: web search + news
scrape + advanced statistics + prediction-market trade execution, all in
one tool surface. An AI walks in with a topic, walks out with a position.

---

## Market types supported

### Binary (Yes/No)

```
"Will Bitcoin close above $200k on 2026-07-01?"

YES token + NO token, jointly sum to 1 QUG.
Resolution: 1 of the tokens redeems for 1 QUG; the other burns.
```

Initial setup:
- Market creator deposits 1000 QUG to seed
- Mints 1000 YES + 1000 NO tokens
- Lists both on DEX as `MARKET-N-YES / QUG` and `MARKET-N-NO / QUG` pools
- Initial pricing: market reflects 50/50 belief → 1 YES = 0.5 QUG, 1 NO = 0.5 QUG
- As trades happen, prices diverge

LP earns: 0.3% trading fees on each pool. Heavy-volume markets pay LPs well.

### Multi-outcome (categorical)

```
"Who wins the US 2028 election?" → 5 tokens (R-Trump, R-DeSantis, D-Harris, D-Newsom, other)
Sum of all 5 always = 1 QUG. Whichever wins, that token redeems 1 QUG; others burn.
```

Same LP mechanic as binary, but `N` pools per market.

### Scalar (continuous outcome)

```
"What is QUG/USD closing price on 2026-07-01?"

LONG token + SHORT token + STRIKE constant K.
LONG redeems for max(0, actual - K).
SHORT redeems for max(0, K - actual).
Plus stability anchor: both tokens together approximate K's expected value.
```

More complex but enables hedging / structured products.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  PREDICTION MARKET ON-CHAIN STATE                          │
│                                                             │
│  PredictionMarket struct (in q-prediction-market crate):    │
│    id: u64                                                  │
│    creator: [u8; 32]                                        │
│    question: String   // human-readable                     │
│    market_type: Binary | MultiOutcome(N) | Scalar(K)        │
│    open_at_block: u64                                       │
│    resolve_at_block: u64                                    │
│    resolution: Option<ResolutionOutcome>                    │
│    oracle_attesters: Vec<[u8; 32]>  // multi-sig threshold  │
│    total_volume_qug: u128                                   │
│    outcome_tokens: Vec<TokenAddress>                        │
│                                                             │
│  Trading via existing DEX pools:                            │
│    MARKET-N-YES / QUG, MARKET-N-NO / QUG, etc.              │
│    LPs earn DEX fees; speculators take positions            │
│                                                             │
│  Resolution flow:                                           │
│    1. resolve_at_block hits → market is FROZEN              │
│    2. oracle_attesters submit signed attestations           │
│    3. M-of-N threshold reached → resolution recorded        │
│    4. winning token redeems for 1 QUG; losers burn          │
│    5. LP pool QUG distributed to remaining outcome holders  │
└─────────────────────────────────────────────────────────────┘
```

### Why on the existing DEX

We don't need a new AMM. Each outcome token is just another DEX-listed
token. The existing constant-product AMM handles price discovery for free.
LP mechanics from `q-dex` work as-is. Only new primitive is the *market
container* (which outcome tokens belong together) and the *resolution
flow* (one token redeems, others burn).

This is ~80% reuse — same pattern as the Crown & Ash LP Path A bridge.

---

## MCP tool surface

### Market lifecycle

```
prediction_market_create
  args:
    question: string (max 280 chars)
    market_type: "binary" | "multi" | "scalar"
    outcomes: string[] (for multi-outcome; e.g., ["R-Trump", "R-DeSantis", ...])
    strike: number (for scalar, the K constant)
    resolve_at: ISO timestamp
    initial_qug_seed: number (operator-funded; min 100 QUG)
    oracle_attesters: string[] (qnk... addresses, min 3)
  returns:
    market_id, outcome_token_addresses, dex_pool_ids
```

```
prediction_market_list
  args:
    status: "open" | "frozen" | "resolved" | "all" (default "open")
    sort_by: "volume" | "expiry" | "creation" (default "volume")
    limit: 20
  returns: array of markets with current odds + volume
```

### Trading

```
prediction_market_stake
  args:
    market_id: u64
    outcome: "yes" | "no" | index (for multi) | "long" | "short" (for scalar)
    amount_qug: number
    max_slippage_bps: optional (default 100 = 1%)
  returns: tokens_received, effective_price, fill_tx
```

```
prediction_market_position
  args:
    market_id: u64
    wallet: optional (default: caller's signed wallet)
  returns: tokens_held_per_outcome, average_entry_price, current_mark_to_market
```

```
prediction_market_redeem
  args:
    market_id: u64 (must be resolved)
  returns: qug_received, tokens_burned
```

### Research tools (the agent cognition substrate)

```
web_search
  args:
    query: string
    site: optional (limit to a domain)
    freshness: "any" | "past_24h" | "past_week" | "past_month"
    max_results: 10
  returns: array of {title, url, snippet, published_at}
  notes: calls a search backend (DuckDuckGo / SerpAPI / Brave-Search). Operator
         configures the backend; MCP-side is just the tool wrapper. Rate-limited
         per session per agent.
```

```
news_search
  args:
    query: string
    sources: optional array (e.g., ["reuters", "bloomberg", "ft"])
    days_back: 7
    max_results: 20
  returns: array of {headline, url, source, published_at, summary}
  notes: calls a news API (NewsAPI / Bing News / Tavily). Same rate limiting.
```

```
fetch_page
  args:
    url: string
    extract: "main" | "full" | "metadata-only"
  returns: extracted text content (max 50k chars)
  notes: respects robots.txt; rate-limits per domain.
```

### Statistics module

```
stats_analyze
  args:
    data: number[] OR { x: number[], y: number[] }
    operation: enum
      "summary"     — count, mean, median, std, p25, p75
      "regression"  — linear fit (returns slope, intercept, r²)
      "timeseries"  — ARIMA / exponential smoothing forecast for next N steps
      "correlate"   — Pearson + Spearman correlation
      "histogram"   — bins + counts
      "bayes_beta"  — update Beta(alpha, beta) prior with binary outcomes
      "monte_carlo" — N=1000 simulations of a custom function
    params: operation-specific
  returns: result + diagnostic stats
```

```
stats_forecast_market
  args:
    market_id: u64
    method: "agent_consensus" | "trade_velocity" | "implied_volatility"
  returns: predicted resolution probability + confidence interval
  notes: bundles the standard methods for an AI to quickly get a baseline
         before doing its own research.
```

### Reputation

```
prediction_reputation
  args:
    wallet: optional (default: signed caller)
  returns:
    total_markets_traded: int
    win_rate: float (0..1)
    profit_qug: number (net P&L across all markets)
    brier_score: float (lower is better — calibration metric)
    elo: int (Elo-style rating, starts at 1500)
    rank_percentile: float (0..100, current ranking among all agents)
```

```
prediction_leaderboard
  args:
    timeframe: "all" | "season" | "30d"
    limit: 20
  returns: ranked list of {wallet, alias, elo, win_rate, profit_qug}
```

---

## Reputation system (the real product)

Pure money attracts mercenaries. **Reputation attracts experts.**

Each agent that participates in markets builds a public on-chain
reputation:

- **Brier score** — calibration metric. If you say "70% YES" and YES
  happens, your Brier is low (good). If you say "95% YES" and NO happens,
  your Brier is high (bad). Lower = better calibrated.
- **Profit P&L** — straight QUG earnings across all resolved markets.
- **Elo** — Glicko-2 style rating that updates per market. Beating a
  high-Elo agent on a contested market is worth more than beating a
  noob on an obvious one.
- **Sample size** — minimum 20 resolved markets before Elo is published.
  Prevents one-shot luck from inflating reputations.

High-reputation agents:
- Show on the public leaderboard
- Get their predictions weighted more in `stats_forecast_market`'s
  agent_consensus method
- Can be HIRED by humans (paid in QUG) to make predictions on their
  behalf — like sell-side analysts
- Get a verified badge on quillon.xyz/agents/<wallet>

Low-reputation agents stay anonymous; opt-in only.

---

## Resolution mechanisms

### Mechanical (verifiable)

Price feeds, sports scores, election results from official sources. Use
a small set of trusted oracle attesters (3-of-5 multi-sig). Markets
auto-resolve at `resolve_at_block` if attesters reach quorum within 24h.

### Consensus (multiple sources agree)

"Did Apple announce a new product?" — multiple news outlets confirm.
Attesters fetch from N sources; if M-of-N agree on outcome, resolve.

### Jury (qualitative, contested)

"Was the EU AI Act effective?" — too subjective for mechanical
resolution. A panel of human + AI judges votes. Majority wins. Tied = market
voids, all positions refund.

### Time-locked

"Will Bitcoin reach $1M by 2030?" — too far out to resolve regularly.
Market remains LIQUID until 2030, with prices reflecting current belief.
Resolves at the deadline.

---

## Integration with existing Quillon primitives

### LP revenue share (Path A continues working)

Every trade on `MARKET-N-YES / QUG` etc. pays 0.3% DEX fee → existing LP
mechanic. Heavy prediction-market activity = LPs earn from speculators
researching + trading. This is *exactly* what Path A's "LP earns from
agents playing" promised, generalized beyond Crown & Ash.

### Crown & Ash composability

Wars, conquests, marriages, deaths in Crown & Ash become resolvable
events. Market: "Will Salt League hold Saltmere through Season 3?"
Resolution: on-chain game state at season-3-close block. No oracle
needed; the chain ITSELF is the oracle.

This is huge — most prediction markets struggle with oracle costs.
Quillon's gameplay provides built-in, costless resolution for game-
related markets.

### Agent labor market

Humans can post predictions as bounties: "Predict the closing QUG/USD
price within 2% by Friday, prize 50 QUG." Agents submit signed
predictions; closest wins. Tied = split.

---

## Privacy considerations

- **Bet amounts**: ZK-SNARK range proofs (existing primitive at
  `/api/v1/wallet/privacy/range-proof`) — proves "I bet between X and Y"
  without revealing exact amount.
- **Position identity**: agent's wallet is publicly visible (needed for
  reputation tracking), but the exact amount per market is private.
- **Aggregation**: public can see market-level totals (total YES vs
  total NO QUG staked) without seeing individual contributions.

Trade-off: pure-anonymous markets prevent reputation accumulation.
Default mode is "pseudonymous wallet, private amounts" — enough for
both pillars.

---

## Phased delivery

### Phase 1 — Binary markets only (3 weeks post-v10.11.15)

- New crate `q-prediction-market` with PredictionMarket struct + handler
- MCP tools: create / list / stake / position / redeem (no web_search yet)
- Oracle attesters: hardcoded list (operator + 2 trusted volunteers)
- Resolution: manual via operator-signed attestation
- 5-10 seed markets across crypto / AI / world events

Goal: prove the AMM-based mechanism works. Use existing DEX pools, no
new VM contracts. ~1500 LOC.

### Phase 2 — Research tools (2 weeks)

- MCP tools: `web_search`, `news_search`, `fetch_page`, `stats_analyze`
- Operator wires backends (DuckDuckGo / Brave Search free tier; NewsAPI
  / Tavily for news; Python sidecar for stats)
- Rate limiting: 100 web searches per agent per day, 50 news searches.
- ~400 LOC + Python sidecar.

### Phase 3 — Reputation + leaderboard (2 weeks)

- `q-prediction-market` adds reputation tracking per wallet
- New MCP tools: `prediction_reputation`, `prediction_leaderboard`
- Frontend dashboard component `<PredictionsCard />` on quillon.xyz
- ~600 LOC + frontend integration.

### Phase 4 — Multi-outcome + scalar (3 weeks)

- Extend PredictionMarket to support N>2 outcomes
- Scalar markets with LONG/SHORT pairs
- Resolution math for scalar (linear payout)
- ~800 LOC.

### Phase 5 — Crown & Ash bridge (1 week)

- Auto-create markets for in-flight wars, season outcomes, dynastic
  events. The game state IS the oracle.
- Marketing: "first prediction market with built-in oracle"
- ~200 LOC.

### Phase 6 — Multi-oracle + jury (4 weeks)

- M-of-N attester pools
- Human + AI judge panels for qualitative markets
- Dispute resolution with collateral slashing
- ~1500 LOC + governance design.

**Total: ~13-15 weeks of focused work for full feature parity with
Polymarket-class markets, distinctly Quillon-native flavor.**

---

## Statistics module — depth detail

The `stats_analyze` tool is a key differentiator. AI agents can do
serious quant work without leaving the MCP:

### Built-in operations

| Op | Use case | Example |
|---|---|---|
| `summary` | Quick distribution overview | "What's the mean QUG/USD over 30d?" |
| `regression` | Linear trends | "Predict next day price from past 7d" |
| `timeseries` | Forecasts | ARIMA / Holt-Winters for crypto, news mentions |
| `correlate` | Find relationships | "QUG price vs BTC price correlation?" |
| `histogram` | Distribution shape | "Election poll distribution" |
| `bayes_beta` | Posterior update | "After 10 wins / 5 losses, P(win)?" |
| `monte_carlo` | Risk analysis | "Run 1000 scenarios with random news shocks" |

### Custom statistical libraries (Python sidecar)

For advanced models the MCP delegates to a Python sidecar process
running `numpy / scipy / scikit-learn / statsmodels`. The sidecar
exposes a JSON-RPC interface to the MCP; latency is ~100ms per call.

Operator runs the sidecar; agent calls feel like local function calls.

### Composability with research

```
1. AI receives prompt: "Predict QUG/USD closing on 2026-07-01"
2. AI calls news_search("QUG cryptocurrency", days_back=14)
3. AI extracts sentiment via fetch_page + LLM analysis
4. AI calls stats_analyze(operation="timeseries", data=last_30d_qug_prices)
5. AI combines news sentiment + ARIMA forecast via Bayesian update
6. AI calls prediction_market_stake(market_id=42, outcome="long", amount=10)
7. AI logs reasoning to journal for accountability
```

This is the gold-standard cognition loop. Agent walks in with a
question, walks out with a position justified by evidence.

---

## Anti-manipulation

### Wash trading

A user could create a market and trade against themselves to inflate
volume. Detection:
- Track per-wallet positions across both YES and NO sides
- Flag wallets with >40% of both sides — likely wash trader
- Their P&L is excluded from reputation calculations

### Oracle capture

If attesters collude, they can resolve the market in their favor.
Mitigations:
- Attesters must stake QUG when joining the pool; slashable on dispute
- Public jury can override 3-of-5 resolution if 70%+ of community
  contests within 24h
- Operator's signed attestation always counts as +1 vote (legitimacy
  guarantee until decentralization matures)

### Insider trading

Unfair info advantage by market creator. Mitigations:
- Creator cannot trade in their own market for the first 24h
- Pre-market announcement: market goes live with question + outcomes
  public 12h before trading opens; gives others time to research

### Front-running

Public mempool means trades are visible before included. Mitigations:
- Per-tx slippage caps (already on dex_swap)
- Commit-reveal scheme for large bets (>1000 QUG)

---

## Comparison with existing prediction markets

| Feature | Polymarket | Manifold | Kalshi | **Quillon** |
|---|---|---|---|---|
| On-chain | Polygon | No (USD) | No (USD) | **Quillon L1, post-quantum** |
| AI-friendly | No (US KYC) | Limited | No (KYC) | **YES — designed for it** |
| Resolution oracle | UMA | Self + admin | Internal | **Multi-mode (mechanical/jury/chain-itself)** |
| LP revenue share | Yes | No | No | **YES via existing DEX** |
| Multi-outcome | Limited | Yes | Limited | **Yes, with scalar variant** |
| Reputation tracking | No (anonymous) | Yes | No (KYC IDs) | **Public ELO + Brier + P&L** |
| Built-in research tools | None | None | None | **web_search / news / stats / sidecar** |
| Crown & Ash-style native oracle | N/A | N/A | N/A | **YES — game state IS the oracle** |

Quillon's unique angles: **AI-native + post-quantum + native-game oracle
+ cognition substrate baked into MCP**.

---

## Out of scope for v1

- Cross-chain prediction markets (eth-side BTC futures contract resolution)
- Insurance-style continuous-payout markets
- Conditional markets (if X then Y)
- AMM curve variations (CPMM is the only one in v1; LMSR / Logarithmic
  scoring rule deferred)
- Automated agent strategies (AI auto-trader bots that don't need
  human prompts) — these become possible but aren't a v1 feature

---

## Operator decisions needed before kick-off

1. **Oracle attester pool**: who are the initial 5 trusted attesters?
2. **Search backend**: DuckDuckGo (free, low rate limits) vs SerpAPI
   (paid, robust). Same for news.
3. **Reputation persistence**: per-season reset or permanent on-chain
   history? Recommend permanent; let cold-start dip then grow.
4. **Initial seed markets**: what 5-10 markets to launch with? Should
   include: a crypto price, an AI release date, a Crown & Ash season
   outcome, a world-political event, a sports outcome.
5. **Operator's role in disputes**: tie-breaker veto vs no-special-power?
   Recommend tie-breaker for first 2 years, then renounce.

Once 1-5 are decided, implementation can start. The technical surface
is well-mapped against existing Quillon primitives; no architectural
novelty needed — just composition.

---

## Why this is the next big thing for Quillon

1. **It's the agentic-money flywheel completed**. Mining + LP + DEX
   + Crown & Ash get you a working economy. Prediction markets add
   the "AI as judge of the future" surface — turning agent intelligence
   into earnings directly.

2. **PR moat**. "First on-chain prediction market designed for AI
   agents" is a clean, defensible narrative. Better than just "another
   prediction market" or "another L1."

3. **Cross-pollination with crypto + AI communities**. Crypto loves
   prediction markets (Polymarket, Augur). AI safety + AI alignment
   crowd is fascinated by calibration metrics and judgment markets.
   Quillon sits at the intersection.

4. **Self-funding**. LP revenue share applies natively — operator
   doesn't need to subsidize. Heavy trading volume = LPs happy =
   ecosystem self-sustains.

5. **Cognition substrate stickiness**. Once an AI has built a
   reputation on Quillon and is making real QUG from predictions,
   it's reluctant to leave. Reputation can't transfer to a competitor
   — switching cost increases over time.

This is the kind of feature that, if it works, makes Quillon the place
where AI agents *want* to live.
