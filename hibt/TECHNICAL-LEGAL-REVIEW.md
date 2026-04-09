# HiBT Exchange Listing Agreement — Technical & Legal Review

**Document under review:** `HiBT__QUG (2).docx` — Listing Agreement between KAIXUAN NETWORK CO., LTD (Party A / HiBT Exchange) and $QUG / Quillon Graph (Party B)
**Review date:** 2026-04-01
**Prepared by:** Quillon Graph core team
**Purpose:** Independent technical and legal review before signing. We request your detailed analysis, risk assessment, and recommended amendments.

---

## 1. PARTIES AND ENTITY INFORMATION

| Field | Party A (Exchange) | Party B (Project) |
|-------|-------------------|-------------------|
| **Name** | KAIXUAN NETWORK CO., LTD | $QUG (Quillon Graph) |
| **Contact** | SOCHEATA KHEANG | Not specified |
| **Registered Address** | 1 Warren Avenue, Oshawa, ON L1J 4E2, Canada | Not specified |
| **Business License** | 758797807RC0001 | Not specified |
| **Country** | Canada | Listed as "USA" |
| **Governing Law** | Article X states: United Kingdom | — |
| **Exchange Website** | www.hibt.com | https://quillon.xyz |
| **CMC Spot Ranking** | #38 (claimed) | — |

### ISSUE 1: Jurisdiction Triangle

The contract creates a three-jurisdiction problem:
- **Entity registered in:** Canada (Ontario)
- **Governing law:** United Kingdom (Article X)
- **Contact person name:** SOCHEATA KHEANG (suggests Southeast Asian connection)
- **Party B listed as:** USA

**Questions for review:**
1. Is it normal for a Canadian-registered entity to choose UK governing law?
2. In case of dispute, which court has jurisdiction — Canadian, UK, or another?
3. Should we insist on a specific arbitration venue (e.g., Hong Kong HKIAC, Singapore SIAC)?
4. Can we verify "KAIXUAN NETWORK CO., LTD" with Canadian business license 758797807RC0001 at the Canada Revenue Agency or Ontario business registry?
5. The "USA" designation for Party B — does this create any SEC/CFTC regulatory exposure for QUG even though it is not a US-based project?

---

## 2. ABOUT QUILLON GRAPH (QUG) — CRITICAL TECHNICAL CONTEXT

**This is essential context for understanding integration complexity and risk.**

QUG is **NOT** an ERC-20 token, BEP-20 token, or any token on an existing blockchain. It is a **native Layer 1 blockchain** with its own consensus mechanism, cryptography, and network. This is comparable to listing Bitcoin or Monero — not listing a Uniswap token.

### Technical Specifications

| Parameter | Value |
|-----------|-------|
| **Type** | Native L1 blockchain (own chain, own consensus) |
| **Consensus** | DAG-Knight (DAG-based BFT, zero-message complexity) |
| **Cryptography** | Post-quantum: Dilithium5 (NIST PQC Level 5) + Kyber1024 |
| **Classical crypto** | Ed25519 (crypto-agile, supports both) |
| **Max Supply** | 21,000,000 QUG |
| **Current Emission** | 2,625,000 QUG/year (Era 0), 4-year halving |
| **Block Time** | ~1 second |
| **Finality** | Sub-3 seconds |
| **Mining** | CPU + GPU (Proof of Work) |
| **Address Format** | `qnk-` prefixed addresses |
| **Network** | libp2p with Gossipsub + Kademlia DHT |
| **API** | REST API on port 8080, well-documented |
| **Smart Contracts** | WASM-based virtual machine |
| **Built-in DEX** | AMM decentralized exchange |
| **Privacy** | Tor integration, Dandelion++ gossip |
| **Nodes** | 4+ bootstrap nodes, 10Gbit supernode |
| **Open Source** | Yes, full source available |

### Integration Implications for HiBT

1. HiBT must **run a QUG full node** (not just call an ERC-20 contract)
2. HiBT must **process native blocks** (not rely on Ethereum/BSC block explorers)
3. HiBT must **handle qnk-prefixed addresses** (not 0x addresses)
4. HiBT must **support Ed25519 and optionally Dilithium5 signatures** (not secp256k1)
5. Deposit/withdrawal requires **native chain integration** (no simple smart contract call)
6. There is **no token contract address** to verify — the coin IS the chain

**This is significantly more complex than 99% of exchange listings.** It requires dedicated engineering effort from HiBT comparable to when exchanges first listed Bitcoin, Monero, or Kaspa.

---

## 3. FINANCIAL TERMS ANALYSIS

### 3.1 Fee Structure

| Item | Amount | Notes |
|------|--------|-------|
| **Listing Fee** | 15,000 USDT | Discounted from 20,000 (POW coin rate, referral discount) |
| **Deposit** | 5,000 USDT | Required in advance, deducted from listing fee |
| **Market Making (USDT)** | 3,000 USDT | Minimum top-up |
| **Market Making (QUG)** | 3,000 USDT equivalent in QUG tokens | At what price? Not specified |
| **Total Estimated** | ~21,000 USDT | Plus QUG token allocation for activities |

### ISSUE 2: QUG Token Valuation for Market Making

The contract states "$3000 USDT equivalent in $QUG tokens" but does not specify:
- **At what price is QUG valued?** There is no established market price since QUG is not yet listed on any CEX.
- **Who determines the price?** Party A, Party B, or market?
- **When is the price locked?** At signing, at deposit, or at listing?
- **What happens if QUG price changes significantly between deposit and listing?**

**Questions for review:**
1. Should we insist on a fixed QUG/USDT rate specified in the contract?
2. Should the QUG equivalent be calculated at listing price (first trade)?
3. Is 3,000 USDT in QUG reasonable for market making, or will it be insufficient to maintain an order book?

### ISSUE 3: Deposit Before Full Terms

Article II, 1.1 requires a 5,000 USDT deposit with blank dates. The contract itself has blank fields for the listing date (`[] [] [2026]`).

**Risk:** Paying a deposit before the listing date is contractually fixed.

### 3.2 Market Making Terms (Article II, 2.1-2.4)

**Positive:**
- 2.2: "not obligated to maintain a specific price level, trading volume or spread" — good, prevents manipulation liability
- 2.3: "All trading activities must be conducted organically" — good, CYA against wash trading accusations

**Concern:**
- 2.4: "If liquidity is exhausted, USDT or $QUG can be used to purchase $QUG from the open market" — this implies an obligation to maintain liquidity even if MM funds run out. Is this an ongoing financial obligation beyond the initial 3K+3K?

**Questions for review:**
1. Does clause 2.4 create an open-ended financial obligation?
2. Should we cap the total MM obligation (e.g., "not exceeding the initial deposit of $3,000 USDT + $3,000 equivalent QUG")?
3. Who controls the market making account? The contract says "project team will provide... and manage it at its sole discretion" — does this mean WE manage it, or THEY manage it?

---

## 4. RIGHTS AND OBLIGATIONS — DETAILED ANALYSIS

### 4.1 Party A (HiBT) Obligations

| Clause | Obligation | Assessment |
|--------|-----------|------------|
| 1.1 | Compliance review within 15 working days | Reasonable |
| 1.1(b) | Accepts open source code instead of smart contract audit | Excellent — correctly adapted for L1 |
| 1.2 | Suspension/delisting with 72h rectification + 30-day notice | Reasonable |
| 1.5 | Full deposit refund if not live within 90 days | Good protection |
| 1.6 | Deposit refund if L1 integration deemed impossible | Good protection |

### 4.2 Party B (QUG) Obligations

| Clause | Obligation | Assessment | Risk Level |
|--------|-----------|------------|------------|
| 2.1 | Pay activity fees | Standard | Low |
| 2.2 | Continuously develop and promote | Vague — what constitutes "continuous"? | Medium |
| 2.3 | Regularly inform Party A of progress | Reasonable | Low |
| 2.4 | Inform 7 days before key parameter changes | Reasonable for tokenomics changes | Low |
| 2.5 | Not imply HiBT endorsement | Standard | Low |
| 2.6 | No money laundering / illegal activity | Standard | Low |
| 2.8 | Disclose source code day after signing | Already open source — no issue | Low |
| 2.9 | Liable for losses caused by QUG on HiBT | **Very broad** — could include user trading losses | **HIGH** |
| 2.10 | No refund if we cancel after signing | Aggressive but standard | Medium |
| 2.11 | Any breach = no refund + termination | **Very one-sided** | **HIGH** |

### ISSUE 4: Liability Exposure (Clause 2.9)

This is the most concerning clause in the contract:

> "Party [A] has the right to be entitled to recourse against Party B for the losses caused by Party B's token exchange on Party A's platform (including but not limited to the compensation Party A shall pay the investors on behalf of Party B, Party A's punishment from authorised organs due to the illegality of Party B's project."

**This means:**
- If a user loses money trading QUG on HiBT, HiBT could seek compensation from us
- If a regulator fines HiBT for listing QUG, they pass the fine to us
- "Losses caused by Party B's token exchange" is extremely broad

**Questions for review:**
1. Should we add a liability cap (e.g., "not exceeding the listing fee paid")?
2. Should we narrow this to losses caused by "fraud, misrepresentation, or willful misconduct" rather than general trading losses?
3. Does this create unlimited financial exposure?
4. Is this standard in exchange listing agreements, or unusually aggressive?

### ISSUE 5: Breach Clause Asymmetry (Clause 2.11)

If Party B (us) breaches ANY obligation, Party A keeps all fees AND can terminate. But if Party A breaches (e.g., fails to provide promised marketing, delays integration), the contract does not specify equivalent remedies for Party B.

**Questions for review:**
1. Should we add a mutual breach clause with equivalent remedies?
2. Should we define what constitutes a "material breach" vs. a minor breach?
3. Should there be a cure period for breaches before termination?

---

## 5. DELISTING RULES — ARTICLE XI ANALYSIS

### 5.1 Security Triggers

| Trigger | Applicability to QUG | Risk |
|---------|---------------------|------|
| 1.1 "Smart contract overflow" | QUG has no smart contracts in the ERC-20 sense. WASM VM contracts are separate from the coin itself | Low — but wording is vague |
| 1.2 "Whitepaper doesn't match smart contract" | Not applicable — QUG is L1 | Low |
| 1.3 "Insufficient hash rate / 51% attack" | **Applicable** — QUG uses PoW mining. If hash rate drops, this could be triggered | **Medium** |
| 1.4 "Security threat during main chain swap" | Not applicable — QUG is already on its own chain | Low |

### ISSUE 6: Hash Rate / 51% Attack Clause (1.3)

This is the most relevant security trigger for QUG. As a relatively new L1 with CPU/GPU mining, the hash rate may fluctuate. HiBT could use low hash rate periods as justification for delisting.

**Questions for review:**
1. Should we define a minimum hash rate threshold in the contract?
2. Should we clarify that this refers to "demonstrated 51% attack" not "theoretical possibility"?
3. Is there a grace period or notification requirement before invoking this clause?

### 5.2 Project/Team Triggers

| Trigger | Risk Assessment |
|---------|----------------|
| 2.1 Major changes without 15-day notice | Medium — we need to remember to notify before any protocol upgrades |
| 2.2 Fake/misleading documents | Low — our tech is real and verifiable |
| 2.3 Key members in lawsuits | **Note: relevant to current legal situation** |
| 2.4 Progress behind schedule vs whitepaper | Medium — depends on what's promised |

### ISSUE 7: Key Members in Lawsuits (Clause 2.3)

> "The key members of the team are involved in lawsuit, arbitration, breaking negative news, or received major administrative punishment."

**Questions for review:**
1. Does this create a delisting risk given any ongoing legal matters of team members?
2. Should we define "key members" narrowly (e.g., CEO/CTO only)?
3. Does "breaking negative news" mean news ABOUT the project, or news in general? This is vaguely worded.

### 5.3 Token/Exchange Triggers

| Trigger | Risk Assessment |
|---------|----------------|
| 3.1 Token distribution doesn't match disclosure | Low — QUG emission is transparent on-chain |
| 3.2 **"Token keeps staying at low prices"** | **HIGH** — completely subjective |
| 3.3 New issuance/forking without notice | Medium — halving is pre-programmed, but protocol upgrades need notification |

### ISSUE 8: Subjective Price Trigger (Clause 3.2)

> "The token keeps staying at low prices cause by significant events that may affect the exchange ability or the value of the underlying cryptocurrency, due to Party B's concealing the project information."

**Questions for review:**
1. What constitutes "low prices"? There is no definition.
2. The clause requires concealment as a condition — does this mean price decline alone is NOT a trigger? Only price decline + concealment?
3. Should we clarify this clause to explicitly state: "Natural market price movements do not constitute grounds for delisting"?

### 5.4 Operations/Promotions Triggers

| Trigger | Risk Assessment |
|---------|----------------|
| 4.1 Website/social media inactive for 1 month | Medium — requires consistent social presence |
| 4.2 Insufficient PR response | Vague — what is "insufficient"? |
| 4.3 Key information discrepancies | Standard |
| 4.4 Unauthorized use of HiBT platform info | Standard |

### ISSUE 9: One-Month Inactivity Threshold (Clause 4.1)

One month of website or social media inactivity triggers delisting consideration. For a development-focused project, this is tight.

**Questions for review:**
1. Should we negotiate this to 3 months (industry standard)?
2. Does "inactive" mean zero posts, or reduced posting frequency?
3. Should we add an exception for planned maintenance or development sprints?

---

## 6. MISSING CONTRACT ELEMENTS

The following items are standard in exchange listing agreements but absent from this contract:

| Missing Element | Why It Matters |
|----------------|---------------|
| **Listing date** | Blank: `[] [] [2026]` — no committed go-live date |
| **Payment wallet address** | No specification of where to send USDT — risk of personal wallet scam |
| **Payment chain** | USDT exists on multiple chains (ERC-20, TRC-20, BEP-20) — which one? |
| **Token allocation for activities** | Activities (airdrops, trading competitions) require QUG but no amount is specified |
| **Unused token return** | If QUG tokens are allocated for marketing activities and not all are used, are they returned? |
| **IP/branding protection** | No clause protecting Quillon Graph's name, logo, or intellectual property |
| **Non-exclusivity** | No explicit statement that QUG can list on other exchanges simultaneously |
| **Party B termination rights** | Party A can terminate for breach; Party B has no equivalent right |
| **Insurance/custody** | How does HiBT secure QUG in their hot/cold wallets? What if they get hacked? |
| **Marketing deliverables** | The listing package promises banners, KOLs, AMAs etc. — none of this is in the contract |
| **SLA/uptime** | No guarantee that the QUG trading pair will be available X% of the time |
| **Audit rights** | No right for Party B to audit trading volume or verify organic vs wash trading |

### ISSUE 10: Marketing Promises vs Contract

ALanZ shared a detailed listing package (Standard: 15K USDT) promising:
- Listing announcements (9M+ exposure)
- Banner placement (3-5 days)
- Telegram promos across 11 communities
- Twitter posters (60K+)
- PR articles on 500+ media outlets
- Social media blast (200K+)

**None of these marketing deliverables appear in the contract.** The contract only covers the listing mechanics, fees, and legal obligations.

**Questions for review:**
1. Should marketing deliverables be annexed to the contract as Schedule/Exhibit A?
2. Without contractual commitment, can HiBT simply not deliver the marketing and claim it was "best effort"?
3. Is it standard practice for listing marketing to be separate from the listing agreement?

---

## 7. LEGAL STRUCTURE CONCERNS

### 7.1 Entity Verification

**KAIXUAN NETWORK CO., LTD** with Canadian business license `758797807RC0001`:
- The format `758797807RC0001` appears to be a Canadian Business Number (BN) with RC program identifier
- RC = Corporation Income Tax account
- This should be verifiable at the Canada Revenue Agency

**Questions for review:**
1. Can this entity be verified through public Canadian business registries?
2. Is "KAIXUAN NETWORK" a known entity in the crypto exchange space?
3. Is the Oshawa, Ontario address a physical office or a registered agent?

### 7.2 Governing Law vs Registration

- Entity: Canada
- Governing law: United Kingdom
- Contact: SOCHEATA KHEANG (name suggests Cambodian/Southeast Asian origin)
- Exchange: Claims global presence

This structure is not inherently problematic (many crypto companies use multi-jurisdiction structures), but it creates complexity in enforcement.

### 7.3 Regulatory Considerations

**Questions for review:**
1. Does listing QUG on a Canadian-registered exchange create Canadian securities law exposure?
2. UK governing law — does this bring FCA (Financial Conduct Authority) regulation into play?
3. The contract lists Party B's country as "USA" — does this trigger SEC jurisdiction?
4. Should we request correction to the actual country of Party B?

---

## 8. TECHNICAL INTEGRATION RISK

### 8.1 Can HiBT Actually Integrate a Native L1?

Most tier-2 exchanges primarily list ERC-20/BEP-20 tokens, which requires minimal engineering (just add a contract address). Integrating a native L1 requires:

1. Running and maintaining a full node
2. Building custom deposit/withdrawal infrastructure
3. Handling unique address formats (qnk- prefix)
4. Managing block confirmation requirements
5. Implementing hot/cold wallet security for a novel cryptographic scheme
6. Handling post-quantum signatures (Dilithium5) that most exchange infrastructure does not support

**ALanZ claimed 72-hour integration.** This seems optimistic for a novel L1 with post-quantum cryptography. Bitcoin integration typically takes exchanges weeks.

**Questions for review:**
1. Is 72-hour L1 integration realistic?
2. Should we require a technical proof-of-concept (test deposit/withdrawal) before the listing fee balance (10K USDT) is paid?
3. Should we add a technical milestone payment structure (e.g., 5K deposit → integration complete → 5K → go-live → 5K)?

### 8.2 Clause 1.1(b) — Smart Contract Audit Waiver

The contract correctly acknowledges QUG is L1 and accepts open source code instead of a smart contract audit. This is good but may need refinement:

**Question:** Should we provide a specific technical documentation package (API docs, node setup guide, integration guide) as a formal exhibit to the contract?

---

## 9. COMPARATIVE ANALYSIS — IS THIS A FAIR DEAL?

### 9.1 Pricing

| Exchange Tier | Typical Listing Fee | HiBT Offer |
|--------------|-------------------|------------|
| Tier 1 (Binance, Coinbase) | $1M - $10M+ | N/A |
| Tier 1.5 (KuCoin, OKX) | $100K - $500K | N/A |
| Tier 2 (MEXC, Gate.io, BitMart) | $20K - $100K | N/A |
| Tier 2-3 (HiBT, claimed #38) | $15K - $30K | **$15K + $6K MM = $21K** |

The pricing appears reasonable IF HiBT is genuinely ranked #38 on CMC. This needs independent verification.

### 9.2 Due Diligence Checklist (Outstanding)

- [ ] Verify HiBT CMC ranking at coinmarketcap.com
- [ ] Check HiBT on CoinGecko for volume verification
- [ ] Look for wash trading indicators
- [ ] Search for HiBT reviews on Reddit, BitcoinTalk, Trustpilot
- [ ] Verify ALanZ identity via hibt.com/support-center/channel-verification
- [ ] Verify KAIXUAN NETWORK CO., LTD in Canadian business registry
- [ ] Check if 1 Warren Avenue, Oshawa, ON L1J 4E2 is a real office
- [ ] Research any known scam reports associated with HiBT
- [ ] Check HiBT's listed coins — are they real projects or mostly dead tokens?

---

## 10. RECOMMENDED AMENDMENTS

Based on the above analysis, we recommend the following amendments before signing:

### Priority 1 — Must Have

| # | Amendment | Rationale |
|---|-----------|-----------|
| 1 | **Add liability cap** to clause 2.9: "Party B's total liability shall not exceed the total fees paid under this agreement" | Prevents unlimited financial exposure |
| 2 | **Fix jurisdiction**: Choose ONE governing law and specify arbitration venue (suggest HKIAC or SIAC) | Three-jurisdiction problem creates enforcement chaos |
| 3 | **Fill in listing date** with specific date or "within 90 days of deposit" | Blank dates are unacceptable in a signed contract |
| 4 | **Specify payment wallet** as corporate HiBT wallet with on-chain verification | Prevents personal wallet fraud |
| 5 | **Correct Party B country** to actual jurisdiction (not "USA" if incorrect) | Avoids inadvertent SEC jurisdiction |
| 6 | **Add mutual breach remedies**: Party B should have termination rights if Party A fails to deliver | Currently one-sided |
| 7 | **Add marketing deliverables** as Schedule A to the contract | Marketing promises are currently unenforceable |

### Priority 2 — Should Have

| # | Amendment | Rationale |
|---|-----------|-----------|
| 8 | **Define "low price"** in clause 3.2 or add: "Natural market movements do not constitute grounds for delisting" | Prevents subjective delisting |
| 9 | **Extend inactivity threshold** from 1 month to 3 months (clause 4.1) | Development sprints may reduce social presence |
| 10 | **Add token return clause**: Unused QUG allocated for activities must be returned within 30 days | Prevents QUG tokens from being kept without use |
| 11 | **Add milestone-based payment**: 5K deposit → integration verified → 5K → go-live → 5K | Reduces upfront risk |
| 12 | **Narrow "key members"** in clause 2.3 to named individuals | Prevents broad interpretation |
| 13 | **Define hash rate threshold** for clause 1.3 or require demonstrated attack, not theoretical risk | Prevents premature invocation |

### Priority 3 — Nice to Have

| # | Amendment | Rationale |
|---|-----------|-----------|
| 14 | **Add non-exclusivity clause** explicitly | Belt and suspenders |
| 15 | **Add IP protection clause** for Quillon Graph name and logo | Prevents unauthorized use |
| 16 | **Require test deposit/withdrawal** before go-live | Verifies integration works |
| 17 | **Add audit rights** for trading volume verification | Ensures reported volume is real |

---

## 11. QUESTIONS FOR YOUR ANALYSIS

Please provide your assessment on the following:

1. **Overall risk level**: Is this contract safe to sign with the recommended amendments? Or are there fundamental structural problems that amendments cannot fix?

2. **Entity legitimacy**: Based on the entity information (KAIXUAN NETWORK CO., LTD, Canadian BN 758797807RC0001, Oshawa Ontario address), does this appear to be a legitimate exchange operator?

3. **Jurisdiction strategy**: Given the Canada/UK/unclear structure, what is the optimal governing law and arbitration venue for Party B's protection?

4. **Liability exposure**: Is clause 2.9 standard in exchange listing agreements, or is it unusually aggressive? What is the realistic worst-case financial exposure?

5. **L1 integration risk**: Given that HiBT primarily lists ERC-20 tokens, what is the probability they can successfully integrate a novel post-quantum L1 blockchain? What should we require as proof of capability before paying the balance?

6. **Regulatory risk**: Does this contract structure (Canadian entity, UK law, "USA" party) create any regulatory tripwires we should be aware of?

7. **Scam indicators**: Based on the contract quality, entity structure, and terms — are there any red flags suggesting this may not be a legitimate exchange listing opportunity?

8. **Counter-offer strategy**: What is the most effective way to present our amendments without causing ALanZ to walk away? Should we frame it as "standard institutional requirements" or negotiate point by point?

9. **Payment protection**: What payment structure minimizes our risk? Escrow? Milestone payments? Letter of credit?

10. **Walk-away threshold**: At what point should we decline this deal and pursue alternative exchanges (MEXC, Gate.io, BitMart, KuCoin)?

---

## 12. REFERENCE DOCUMENTS

All documents are located at `/opt/orobit/shared/q-narwhalknight/hibt/`:

| File | Description |
|------|-------------|
| `HiBT__QUG (2).docx` | The listing agreement under review (this document analyzes it) |
| `HiBT__QUG.docx` | Previous version of listing agreement |
| `HIBT-Listing Package.pdf` | Marketing package and pricing overview |
| `Hibt Listing Solutions-2025-en.pdf` | Exchange capabilities and service tiers |
| `fwog.pdf` | Case study of a previous HiBT listing (FWOG token) |
| `HIBT-LISTING-BRIEF.md` | Our internal negotiation brief with strategy and red flags |

---

*This review is prepared for internal use and for consultation with external legal and technical advisors. It does not constitute legal advice. All contract amendments should be reviewed by a qualified attorney in the relevant jurisdiction(s) before signing.*
