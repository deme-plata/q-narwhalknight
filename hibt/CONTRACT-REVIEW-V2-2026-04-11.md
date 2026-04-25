# HiBT Listing Agreement v2 — Contract Analysis

**Date:** 2026-04-11  
**Document:** Updated Listing Agreement (received 2026-04-10)  
**Prior review:** TECHNICAL-LEGAL-REVIEW.md (2026-04-01, reviewed v1)  
**Status:** Alan Z says this is the "final contract" — requesting sign-off  

---

## 1. Changes from v1 → v2 (Improvements)

The updated contract addresses several issues raised in our v1 review:

| Issue from v1 | v1 State | v2 State | Resolved? |
|---------------|----------|----------|-----------|
| Jurisdiction triangle | UK governing law, Canadian entity | **Canada (Ontario)** governing law | **YES** — entity and law now aligned |
| Party B country | Listed as "USA" | Listed as **Denmark (Copenhagen)** | **YES** — no longer creates SEC exposure |
| L1 integration clause | Missing | **Article II 1.6**: deposit refunded if integration impossible | **YES** — critical protection added |
| Smart contract audit | Required (impossible for L1) | **Replaced** with open-source code + technical docs | **YES** — correctly adapted for L1 |
| DAG-Knight recognition | Not mentioned | **Article XI 1.3**: correctly identifies consensus mechanism | **YES** — shows technical diligence |
| Refund on early delisting | Missing | **Article III 1.2 Mutual Guarantee**: pro-rata refund formula | **YES** — `refund = remaining_months/6 × 15,000 USDT` |
| 90-day deadline protection | Unclear | **Article II 1.5**: full deposit refund if not online within 90 days | **YES** |
| Liability cap | Uncapped | **Article 2.9**: capped at 15,000 USDT | **YES** |
| Party B remedy for breach | One-sided | **Article 2.12**: Party B can demand proportional refund if Party A in serious breach (30-day cure) | **YES** |

**Summary:** The major structural issues from v1 are resolved. The contract is substantially better.

---

## 2. Remaining Concerns

### 2.1 MODERATE: Market Manipulation Language in "Plan" Section

The informal plan section (not numbered as an Article) contains:

> "Upon launch, I will instruct the market maker team to **push up the HIBT price as much as possible**. The price will be **higher than the DEX price** and will remain high for as long as possible."

**Risk:** This is explicit market manipulation language. Under Canadian law (Ontario Securities Act) and most jurisdictions, coordinated price manipulation is illegal regardless of asset type.

**However:** This text appears in the informal "plan" section between the payment terms and Article III — it reads like a sales pitch from Alan Z, not a contractual obligation. It is NOT an Article of the agreement.

**Recommendation:** 
- Do NOT sign a document that includes this language, even informally
- Request this section be either removed or reworded to say "Party A will provide market-making support to ensure adequate liquidity"
- QUG should have no written agreement to manipulate prices — even if Party A does it independently
- If Alan Z insists on keeping it, ensure it's clear this is Party A's unilateral marketing plan and Party B has no involvement in price decisions

### 2.2 LOW: Market Making Fund Control

Article II 2.1-2.4:
> "The project team will provide the project with a market making account and **manage it at its sole discretion**."

**Good:** Article 2.2 explicitly states "The project is not obligated to maintain a specific price level, trading volume or spread." This is protective.

**Good:** Article 2.3 requires compliance with applicable laws.

**Concern:** Article 2.4 says "USDT or QUG can be used to purchase QUG from the open market to ensure sufficient liquidity." This is standard but gives Party A latitude to trade QUG from the MM account. Since MM minimum is only $3K USDT + $3K QUG, the risk is capped at ~$6K.

**From chat:** Alan Z confirmed MM funds are due AFTER listing, not upfront. Good.

### 2.3 LOW: Source Code Disclosure (Article 2.8)

> "Party B shall disclose the project's source codes to Party A."

**Non-issue for QUG:** The code is already open-source at code.quillon.xyz. This clause was designed for token projects — for a public L1, it's a formality.

### 2.4 LOW: Confidentiality vs. Community Transparency

Article VIII requires confidentiality of agreement terms. However, you're planning a public crowdfunding tracker on quillon.xyz.

**Recommendation:** Confirm with Alan Z that:
- Disclosing that QUG is listing on HiBT is allowed (they want the announcement anyway)
- The crowdfunding tracker doesn't disclose specific contract terms (fees, etc.)
- General fundraising language like "exchange listing fund" is acceptable without revealing exact amounts

### 2.5 INFORMATIONAL: Exchange Size and Volume

HiBT claims CMC spot ranking #38. This should be independently verified. Smaller exchanges with low real volume mean:
- Less actual liquidity for QUG holders
- Lower visibility compared to top-10 exchanges
- But also: lower listing costs ($15K vs $500K+ for tier-1)

For a $1B mcap L1, this is a **starting point**, not a destination. A HiBT listing provides:
- A centralized exchange presence (important for credibility)
- A USDT trading pair (enables fiat on-ramp)
- Marketing support (KOLs, Planet Daily, etc.)
- A reference for approaching larger exchanges later

---

## 3. Financial Summary

| Item | Amount | When Due | Refundable? |
|------|--------|----------|-------------|
| Deposit | 5,000 USDT | "As early as possible" (Alan Z), negotiated to Oct 1 2026 | YES — if integration fails (1.6) or not online in 90 days (1.5) |
| Remaining fee | 10,000 USDT | Within 90 days of deposit | YES — pro-rata if delisted within 6 months (III 1.2) |
| Market making | 3,000 USDT + 3,000 USDT in QUG | After listing goes live | No explicit refund clause |
| **Total exposure** | **~21,000 USDT** | | **15,000 USDT refundable under protective clauses** |

---

## 4. Chat Negotiation Status (2026-04-10)

Key outcomes from the Alan Z chat:

1. **Timeline:** Alan Z pushing for early deposit. You proposed Oct 1, 2026. He accepted the 90-day payment split but wants deposit "as early as possible" for "priority listing slot."
2. **Refund formula:** Confirmed — will be in final contract: `refund = remaining_months/6 × 15,000`
3. **MM timing:** Confirmed — due after listing, not upfront
4. **Contract status:** Alan Z says "this is the final contract" and asking if anything else needs modifying

---

## 5. Recommended Actions Before Signing

### MUST DO:
1. **Remove or reword the market manipulation language** — The "push up price" paragraph should not appear in any signed document. Ask Alan Z to remove the informal plan section or reword it.
2. **Confirm the refund formula is in the final PDF** — Alan Z said he'd add it "later." Do not sign until you see it in the actual document.
3. **Confirm MM deposit timing in writing** — "After listing" should be in the contract, not just in chat.

### SHOULD DO:
4. **Verify KAIXUAN NETWORK CO., LTD** — Check Canadian business registry (758797807RC0001) at https://www.ic.gc.ca/app/scr/cc/CorporationsCanada/fdrlCrpDtls.html
5. **Verify HiBT CMC ranking** — Check CoinMarketCap independently for real volume and ranking.
6. **Clarify crowdfunding disclosure** — Confirm that announcing the listing publicly is permitted under Article VIII.

### NICE TO HAVE:
7. **Add arbitration clause** — Article X says Ontario law but doesn't specify arbitration mechanism. Consider requesting "disputes resolved by binding arbitration under [ADRIC/ICC] rules in Toronto."
8. **Request email as official notice channel** — Several clauses require "written notice" but don't specify how. Confirm email is sufficient.

---

## 6. Verdict

**The contract is reasonable for a mid-tier exchange listing.** The v2 addresses all major structural issues from v1. The protective clauses (deposit refund, integration feasibility refund, pro-rata delisting refund, liability cap, Party B remedy) are solid.

**Total financial risk is capped at ~21K USDT** with 15K refundable under multiple protective scenarios.

**The one item that MUST be fixed before signing is the market manipulation language.** Everything else is either resolved or low-risk.

**Recommended reply to Alan Z:**

> Thank you for the updated contract. We are almost ready to finalize. Two small items before we can sign:
> 
> 1. The informal plan section between Article II and Article III contains language about "pushing up the price" — we need this removed or reworded to "Party A will provide liquidity support" to protect both sides legally.
> 
> 2. Please confirm the final PDF includes: (a) the refund formula you agreed to add, (b) MM funds due after listing not before.
> 
> Once these are in the final version, we are ready to sign and proceed with the deposit timeline.
