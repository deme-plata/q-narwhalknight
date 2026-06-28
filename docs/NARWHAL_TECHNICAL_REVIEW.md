# NARWHAL — Agentic Money Smart Contract · Technical Review

**Author:** DeepSeek V4 (codewhale agent)
**Date:** 2026-05-24 · **Revised:** 2026-05-24 (Codex review)
**Status:** Design phase — fixes applied, pending deploy
**Target:** Quillon Graph v10.11.32

---

## 1. Overview

NARWHAL is a **multisig-governed advanced token** on Quillon Graph. Unlike existing
single-owner AI tokens (PACI by Claude Code, SCALPEL by Codex), NARWHAL is
co-owned by AI agents through an M-of-N multisig scheme. The token includes
reflection, staking, and a treasury that funds code bounties.

### Contract stack (two contracts, both upgradeable)

```
Contract A: multisig_wallet
  └─ owners: [Viktor, Rocky, Codex, DeepSeek]
  └─ threshold: 3
  └─ timelock + spending_limit + social_recovery
  └─ ⚠ upgrade: NOT natively upgradeable — redeploy if needed

Contract B: advanced_token ("NARWHAL")
  └─ admin: Contract A (multisig address) — ⚠ VERIFY server accepts 'admin'
  └─ symbol: NRWL
  └─ decimals: 24
  └─ upgrades: true (proxy pattern — contract can be patched)
  └─ features: reflection, staking, mintable, burnable, governance
```

---

## 2. Template Capabilities (Quillon VM)

Quillon's `deploy_smart_contract` MCP tool offers **27 templates** (verified locally).

### 2.1 `multisig_wallet` template

| Parameter | Type | Required | Notes |
|-----------|------|----------|-------|
| `owners` | string[] | Yes | Array of qnk… wallet addresses |
| `threshold` | number | Yes | M-of-N signatures required |
| `timelock` | boolean | No | ⚠ Verify: configurable or server default? |
| `spending_limit` | boolean | No | ⚠ Verify: configurable or boolean flag? |
| `social_recovery` | boolean | No | ⚠ Verify: actually implemented? |

**On-chain semantics:** Every transaction requires `threshold` valid Ed25519
signatures. Viktor-veto is NOT enforced by the multisig alone — if Viktor must
always approve, either:
- Set threshold = N (all owners) with Viktor always included, OR
- Verify the server supports a `required_signer` parameter.

**⚠ Unverified claims (must verify before deploy):**
- Timelock/spending_limit configurability
- Social recovery implementation
- Sunset clause existence
- `admin` parameter acceptance on advanced_token

**Upgradeability:** The `multisig_wallet` template has no `upgrades` toggle.
If the contract needs upgrading, redeploy and migrate treasury.

### 2.2 `advanced_token` template (PACI-style)

| Parameter | Type | Required | Notes |
|-----------|------|----------|-------|
| `name` | string | Yes | Full token name |
| `symbol` | string | Yes | Ticker (1-8 chars) |
| `decimals` | number | Yes | Typically 24 |
| `initial_supply` | string | Yes | ⚠ Must be string — raw base units |
| `mintable` | boolean | No | Admin can create new tokens |
| `burnable` | boolean | No | Anyone can burn their own |
| `reflection` | boolean | No | Fee distributed to all holders |
| `staking` | boolean | No | Lock tokens, earn rewards |
| `governance` | boolean | No | Token-weighted proposal voting |
| `airdrops` | boolean | No | Admin can batch-distribute |
| `upgrades` | boolean | No | ✅ Proxy pattern for upgrades |
| `reflection_fee_bps` | number | No | ⚠ Verify enforced at transfer time |
| `max_tx_bps` | number | No | Anti-whale cap per tx |
| `max_wallet_bps` | number | No | Anti-whale cap per wallet |

**Upgradeability:** `upgrades=true` enables a proxy pattern. The token contract
can be upgraded without losing state. This is critical for:
- Fixing bugs discovered after deploy
- Adding new features (e.g., treasury split if not in v1)
- Adjusting fee parameters

**Reflection mechanism (⚠ needs verification):** On transfer, `reflection_fee_bps`
is taken. Whether the fee is split (holders vs treasury) depends on the contract
implementation — verify before claiming a split.

---

## 3. Token Economics (corrected per Codex review)

### 3.1 Supply calculation

```
Display supply:  1,000 NRWL
Decimals:        24
Base units:      1,000 × 10^24 = 10^27
initial_supply:  "1000000000000000000000000000"
```

The entire supply is minted to the multisig wallet at deploy time. No pre-mine
to individual agents — distribution happens through bounties and staking rewards.

### 3.2 Fee structure

| Source | Rate | Destination | Verified? |
|--------|------|-------------|-----------|
| Transfer fee | 2% (200 bps) | TBD: holders + treasury split depends on contract | ⚠ |

---

## 4. Multisig Key Management (revised)

### 4.1 Known wallet addresses (no TBD — only confirmed)

| Agent | Model | Wallet | Status |
|-------|-------|--------|--------|
| Viktor | Human | `qnkefca1e8c...` | Confirmed |
| Rocky | Claude Opus 4.7 | `qnk7154929a...` | Confirmed |
| Codex | GPT-5.5 | `qnka3a92bba0...` | Confirmed |
| DeepSeek | V4 Flash | Pending — deploy blocked | Missing |

### 4.2 Seed custody (corrected)

Each agent must have a **distinct wallet seed**, stored per agent and per host
with 0600 permissions. Never reuse one seed across agents. Never commit seeds.

Seed paths:
```
~/.quillon/seeds/deepseek.seed   ← DeepSeek only
~/.quillon/seeds/claude.seed     ← Rocky only
~/.quillon/seeds/codex.seed      ← Codex only
```

If Grok/Qwen join later, they get their own seed files. The MCP v2.16.2
auto-detects the agent via `QUILLON_CLIENT` env and resolves the correct seed.

---

## 5. Risks & Mitigations

| Risk | Severity | Mitigation |
|------|----------|------------|
| **Seed loss** | Critical | Per-agent seeds, social recovery (⚠ verify) |
| **Governance bypass** | High | Set threshold=N (all owners) for critical ops |
| **Contract bug** | High | `upgrades=true` — proxy pattern allows patching |
| **Supply string wrong** | Critical | Dry-run preview BEFORE deploy |
| **No Viktor veto** | High | Verify `required_signer` or use threshold=N |
| **Treasury split unclear** | Medium | Verify reflection implementation first |

---

## 6. Comparison with PACI and SCALPEL

| Property | PACI | SCALPEL | NARWHAL |
|----------|------|---------|---------|
| Creator | Claude Code | Codex (GPT-5.5) | DeepSeek V4 |
| Ownership | Single wallet | Single wallet | Multisig (M-of-N) |
| Decimals | 24 | 24 | 24 |
| Upgradeable | ? | ? | ✅ (proxy) |
| Reflection | Yes | Unknown | Planned (⚠ verify) |
| Staking | Yes | Unknown | Planned |
| Mintable | Yes | Unknown | Yes (multisig-gated) |

---

## 7. Deployment Plan

### Pre-flight gate (MUST pass before deploy)

- [ ] Dry-run both contracts with `confirm=false`
- [ ] Verify `initial_supply` string in preview output
- [ ] Verify `admin` parameter accepted by advanced_token
- [ ] Verify `required_signer` support in multisig_wallet (or use threshold=N)
- [ ] DeepSeek wallet created and seed stored
- [ ] Production node stable (v10.11.32 synced)

### Phase 1 — Multisig

```
deploy_smart_contract template=multisig_wallet confirm=false
  owners: [Viktor, Rocky, Codex, DeepSeek]
  threshold: 3  (or 4 if Viktor-veto needed)
  timelock: true
  spending_limit: true
  social_recovery: true
# → REVIEW DRY-RUN OUTPUT → then confirm=true
```

### Phase 2 — Token

```
deploy_smart_contract template=advanced_token confirm=false
  name: "Narwhal"
  symbol: "NRWL"
  decimals: 24
  initial_supply: "1000000000000000000000000000"  # 1,000 NRWL
  admin: <multisig_address>
  mintable: true
  burnable: true
  reflection: true
  staking: true
  governance: true
  upgrades: true
  reflection_fee_bps: 200
# → REVIEW DRY-RUN OUTPUT → then confirm=true
```

### Phase 3 — Liquidity

```
add_liquidity token0=QUG token1=NRWL amount0=500 amount1=5000
```

---

## 8. Open Questions

1. Does `multisig_wallet` support `required_signer`? If not, use threshold=N.
2. Does `advanced_token` accept `admin` parameter for mint/upgrade authority?
3. Is `reflection_fee_bps` enforced at transfer? Is it split (holders vs treasury)?
4. Are `timelock`/`spending_limit` configurable or boolean flags with server defaults?
5. Grok/Qwen wallets — add later, not blocking deploy.

---

## 9. Dependencies

| Dependency | Status |
|------------|--------|
| DeepSeek seed file | Missing — blocks deploy |
| q-api-server v10.11.32 on epsilon | Running (syncing — delta P2P pending) |
| Dry-run preview | Pending |
