# 100 GH/s Growth Strategy: QNK Hashrate Roadmap

**Current State (March 2026):** ~7 GH/s aggregate network hashrate, achieved in 31 days of mainnet operation.

**Target:** 100 GH/s within 18-24 months (14x growth).

**Context:** QNK matched Monero's approximate hashrate in 31 days versus Monero's ~6 years to reach similar levels. The mining algorithm (BLAKE3 + VDF) is CPU-optimized with limited GPU advantage due to the sequential VDF component.

---

## Phase 1: Foundation (Months 1-3) --- Target: 15 GH/s

### 1.1 Mining Documentation

Create a dedicated `quillon.xyz/mining` landing page:

- **One-command quickstart:**
  ```bash
  # Linux
  wget https://quillon.xyz/downloads/q-miner-linux-x64 -O q-miner && chmod +x q-miner
  ./q-miner --address YOUR_QNK_ADDRESS --threads $(nproc)

  # Windows
  curl -o q-miner.exe https://quillon.xyz/downloads/q-miner-windows-x64.exe
  q-miner.exe --address YOUR_QNK_ADDRESS --threads 8
  ```

- **CPU hashrate benchmarks by model:**

  | CPU | Cores/Threads | Hash Rate | Notes |
  |-----|---------------|-----------|-------|
  | AMD Ryzen 5 5600X | 6/12 | ~180 MH/s | Best consumer value |
  | AMD Ryzen 9 7950X | 16/32 | ~480 MH/s | Top consumer |
  | AMD EPYC 7763 | 64/128 | ~1.8 GH/s | Server-grade |
  | Intel i7-13700K | 16/24 | ~350 MH/s | Intel mainstream |
  | Intel Xeon w9-3495X | 56/112 | ~1.5 GH/s | Intel server |
  | Apple M3 Max | 16 | ~300 MH/s | ARM (when supported) |

  *Note: Benchmarks are estimates pending community verification. The BLAKE3+VDF algorithm benefits from high single-thread performance and scales linearly with core count up to memory bandwidth limits.*

- **Profitability calculator:**
  - Era 0 emission: 2,625,000 QUG/year
  - Block reward: ~0.082 QUG per solution
  - Calculator inputs: hashrate, electricity cost, hardware cost
  - Show daily/monthly/yearly QUG earnings at current network difficulty

- **Tor-routed mining guide:** Document `--tor` flag for miners who want network-layer privacy

- **Pool vs solo comparison:** Variance analysis, expected time-to-block at different hashrates

### 1.2 Discord Restructure

Add mining-focused channels to the Discord server:

- `#mining-general` -- Discussion, tips, configuration help
- `#mining-benchmarks` -- Community-reported hashrate by hardware
- `#mining-support` -- Troubleshooting, connectivity issues
- `#mining-profitability` -- Earnings reports, difficulty tracking
- `#dev-updates` -- Automated deployment notifications

Deploy a Discord bot pulling from `/api/v1/status`:
- Live network hashrate
- Current block height
- Active miner count
- Latest block time
- Network difficulty

### 1.3 BitcoinTalk Thread Refresh

Create a professional announcement thread (ANN) with structured sections:

1. **Overview** -- What QNK is, technical differentiators
2. **Technical Specifications** -- Consensus, cryptography, privacy stack
3. **Mining Guide** -- Quick start, benchmarks, pool information
4. **Downloads** -- Direct links with SHA256 checksums
5. **Roadmap** -- Quarterly milestones
6. **Team/Development** -- Open source contribution model
7. **FAQ** -- Common questions

Maintain with weekly development updates and a community benchmark table.

Consider a signature campaign for established BitcoinTalk members (Legendary/Hero members) to increase visibility.

### 1.4 Metzdowd Cryptography Mailing List Announcement

Send the technical letter (see `docs/metzdowd-announcement.txt`) to `cryptography@metzdowd.com`. This targets the academic cryptography community and may generate review/criticism that improves the protocol.

### 1.5 Public Mining Pool

Deploy `pool.quillon.xyz:3333` using the existing `q-mining-pool` crate:

- Stratum V1 protocol for broad miner compatibility
- CRDT-based PPLNS reward distribution
- Variable difficulty adjustment per miner
- Web dashboard showing pool hashrate, miners, blocks found
- 1% pool fee (competitive with established pools)

---

## Phase 2: Community Growth (Months 4-9) --- Target: 35 GH/s

### 2.1 Technical Articles

Publish on the project blog and submit to aggregators:

**Article 1: "Post-Quantum Mining: BLAKE3+VDF ASIC Resistance"**
- How the sequential VDF loop bounds ASIC advantage to ~2-3x
- Comparison with RandomX, ProgPoW, Ethash approaches
- Submit to: Hacker News, r/CryptoCurrency, r/CryptoMining

**Article 2: "Sub-3s Finality in a DAG Consensus System"**
- DAG-Knight's zero-message-complexity BFT
- VDF anchoring for temporal ordering
- Comparison with Tendermint, HotStuff, Avalanche finality times
- Submit to: Hacker News, r/crypto (academic)

**Article 3: "Ring Signatures and Bulletproofs++ in a Post-Quantum Setting"**
- Privacy stack architecture: LSAG + stealth addresses + range proofs
- Quantum resistance considerations for curve25519-based privacy
- Migration path to lattice-based privacy primitives
- Submit to: r/privacy, Monero Research Lab forums

### 2.2 Cross-Pollinate CPU Mining Communities

Target miners from existing CPU-mineable currencies:

- **Monero/RandomX miners:** "Mine QNK alongside XMR -- different algorithm (BLAKE3+VDF vs RandomX), same hardware. Dual-mine guide."
- **Raptoreum miners:** Similar CPU-focused community, GhostRider vs BLAKE3+VDF comparison
- **Verus miners:** VerusHash vs BLAKE3+VDF, emphasis on VDF ASIC resistance

Create hashrate equivalency charts so miners can estimate QNK earnings from their existing XMR hashrate.

### 2.3 Developer Outreach

- **GitHub public mirror:** Read-only mirror of the source for visibility and issue tracking
- **"Good first issue" labels:** Tag accessible tasks for new contributors
- **Bounty program:** The `q-bounty-protocol` crate is already implemented. Activate it with funded bounties for:
  - Bug reports (tiered by severity)
  - Documentation improvements
  - Test coverage additions
  - Performance optimizations
- **Rust conference talks:** Submit to RustConf, EuroRust, Oxidize on topics like "Building a Post-Quantum Blockchain in Rust" or "DAG Consensus with VDF Anchoring"

### 2.4 Hashrate Incentive Programs

- **Mining competitions:** Weekly/monthly prizes for top hashrate contributors
- **Early miner bonus:** Height-gated reward multiplier (e.g., 1.5x rewards for first 500K blocks) -- requires consensus-level change with proper upgrade gate
- **Referral program:** Bonus rewards for miners who onboard new participants (tracked via a referral code in the miner's coinbase field)

---

## Phase 3: Acceleration (Months 10-18) --- Target: 65 GH/s

### 3.1 Exchange Listings

Priority order:
1. **TradeOgre** -- CPU-coin friendly, low listing requirements, strong privacy coin community
2. **MEXC** -- High volume, accessible globally
3. **Gate.io** -- Good altcoin discovery
4. **Bitget** -- Growing exchange with mining community

Requirements before listing:
- Stable network with 99.9%+ uptime for 6+ months
- Multiple independent pools
- Block explorer with public API
- Documented wallet integration (RPC/REST)

### 3.2 Third-Party Pool Ecosystem

- Publish pool operator documentation: protocol specification, share validation, reward calculation
- Open-source the pool implementation
- Target 3-5 independent pools within 6 months
- Geographic distribution: NA, EU, Asia pools for latency optimization

### 3.3 Academic Paper

Submit to tier-1 venues:
- **Financial Cryptography (FC)** -- DAG-Knight with VDF anchoring, empirical performance analysis
- **ACM CCS** -- Post-quantum privacy stack (ring signatures + Bulletproofs++ in PQ setting)
- **IEEE S&P** -- Formal security analysis of the BLAKE3+VDF mining algorithm

Include real network data: hashrate growth, finality measurements, block propagation latency, privacy transaction statistics.

### 3.4 Hardware Partnerships

- Submit BLAKE3+VDF benchmarks to hardware review sites (AnandTech, Phoronix, ServeTheHome)
- Partner with VPS/hosting providers for one-click mining node deployment
- Engage with CPU manufacturers (AMD, Intel) for inclusion in crypto mining benchmark suites

---

## Phase 4: Maturity (Months 19-24) --- Target: 100 GH/s

### 4.1 Geographic Expansion

- Translated mining guides: Chinese, Russian, Korean, Japanese, Spanish, Portuguese
- Regional community channels on Discord and Telegram
- Localized landing pages at `quillon.xyz/cn`, `quillon.xyz/ru`, etc.
- Regional mining pool partnerships

### 4.2 ARM Mining

- Compile `q-miner` for ARM64 targets:
  - Raspberry Pi 5 (Cortex-A76, ~20-30 MH/s estimated)
  - Apple Silicon (M1/M2/M3, high single-thread, ~200-400 MH/s estimated)
  - AWS Graviton3/4 (cloud ARM mining)
  - Ampere Altra (ARM server-grade)
- BLAKE3 has excellent ARM NEON intrinsic support
- Create Raspberry Pi mining cluster guide (hobbyist appeal)

### 4.3 DeFi Ecosystem Activation

The on-chain infrastructure already exists:
- AMM DEX with concentrated liquidity (`q-dex` crate)
- Token deployment and management
- Smart contract VM (WASM sandbox)

Drive usage to create transaction fee revenue for miners:
- Stablecoin pairs for QUG trading
- Cross-chain bridges (Monero atomic swaps via `q-monero-bridge`)
- Lending/borrowing protocols
- NFT marketplace (optional, if community demands)

### 4.4 Network Effect Compounding

The virtuous cycle:
```
Higher hashrate
    -> More security (cost to 51% attack increases)
    -> More confidence from users and exchanges
    -> More demand for QUG
    -> Higher mining profitability
    -> More miners join
    -> Higher hashrate
```

---

## Milestones Summary

| Month | Hashrate | Est. Miners | Key Milestone |
|-------|----------|-------------|---------------|
| 0     | 7 GH/s   | 30          | Current state (March 2026) |
| 3     | 15 GH/s  | 60          | Mining docs, pool launch, metzdowd letter, BitcoinTalk refresh |
| 6     | 25 GH/s  | 100         | First exchange listing, 3+ technical articles published |
| 9     | 35 GH/s  | 150         | 3+ independent pools, GitHub mirror public, bounty program active |
| 12    | 50 GH/s  | 250         | Second exchange, academic paper submitted, Rust conference talk |
| 18    | 75 GH/s  | 400         | ARM mining support, multi-language documentation, 5+ pools |
| 24    | 100 GH/s | 600         | Mature DeFi ecosystem, geographic diversity, sustainable growth |

---

## Risk Factors

1. **Algorithm change pressure:** If BLAKE3+VDF is broken or a significant ASIC is developed, a hard fork to a new algorithm may be needed. The upgrade gate system supports this.

2. **Regulatory uncertainty:** Privacy is mandatory (all transactions receive automatic ZK proofs since v3.4.16), which may face regulatory challenges in some jurisdictions. The crypto-agility architecture allows policy adjustments if needed.

3. **Competition:** New CPU-mineable coins may compete for the same miner base. Technical differentiation (post-quantum, DAG consensus, sub-3s finality) is the primary moat.

4. **Centralization risk:** If a small number of large miners dominate, decentralization suffers. Pool diversity and geographic distribution mitigate this.

5. **Developer bandwidth:** A ~25-crate Rust workspace requires ongoing maintenance. Growing the contributor base through bounties and open-source engagement is critical.

---

## Resource Requirements

- **Infrastructure:** Pool server (~$200/mo), additional bootstrap nodes (~$100/mo each)
- **Community management:** Discord moderators, BitcoinTalk thread maintenance
- **Marketing budget:** Exchange listing fees ($5K-$50K depending on exchange), conference attendance ($2K-$5K per event)
- **Bounty fund:** 50,000 QUG allocated from treasury for developer bounties
- **Translation:** $500-$1,000 per language for professional mining guide translation

---

*This document is a living plan. Update quarterly based on actual hashrate growth, community feedback, and market conditions.*
