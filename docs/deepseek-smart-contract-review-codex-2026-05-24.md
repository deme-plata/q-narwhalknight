# Codex Review for DeepSeek Smart Contract Designs

**Date:** 2026-05-24  
**Reviewer:** Codex  
**Scope:** `docs/NARWHAL_TECHNICAL_REVIEW.md`, `docs/AGORA_DESIGN.md`, and local deploy tooling in `tools/quillon-wallet-mcp/src/index.ts`

## Executive summary

Do not deploy NARWHAL or AGORA yet. The concepts are viable, but both documents currently describe governance and token behavior that is not fully proven by the local `deploy_smart_contract` wrapper or the listed template metadata.

The largest issue is governance semantics: AGORA says Viktor is always required, while the proposed `multisig_wallet` deployment uses a plain `threshold: 3`, which permits any 3 owners to execute without Viktor unless the underlying contract has an undocumented required-signer policy. The second largest issue is token economics precision: the supply strings and fee descriptions need to be reconciled with 24-decimal base units and the actual `advanced_token` parameters.

## Code facts verified locally

From `tools/quillon-wallet-mcp/src/index.ts`:

- `deploy_smart_contract` exposes **27** contract templates, not 30.
- `advanced_token` requires: `name`, `symbol`, `decimals`, `initial_supply`.
- `advanced_token` feature toggles listed by MCP: `mintable`, `burnable`, `reflection`, `staking`, `governance`, `airdrops`, `upgrades`.
- `multisig_wallet` requires: `owners`, `threshold`.
- `multisig_wallet` feature toggles listed by MCP: `timelock`, `spending_limit`, `social_recovery`.
- `identity_contract` requires: `root_authority`.
- The actual deploy payload sent by MCP is:

```json
{
  "contract_type": "<template id>",
  "owner": "<deployer hex without qnk prefix>",
  "parameters": {}
}
```

This means the docs can use the human-facing MCP field name `template`, but any implementation notes should recognize that the server receives `contract_type`.

Additional code pointers from local search:

- Server-side deploy/parser path: `crates/q-api-server/src/contracts_api.rs`.
- `parse_contract_type` maps `advanced_token`, `multisig_wallet`, and `identity_contract` in `contracts_api.rs`.
- VM template definitions appear in `crates/q-vm/src/contracts/orobit_smart_contracts.rs`.
- `orobit_smart_contracts.rs` contains `load_advanced_token_template()` and `load_multisig_wallet_template()`.
- Search hits show `reflection_fee_bps` handling in `contracts_api.rs` around the advanced token fee configuration path; DeepSeek should inspect that code before promising a treasury split.

## Required changes before deploy

### 1. Fix base-unit supply strings

Both docs should include an explicit table with display supply, decimals, and base units.

For NARWHAL:

```text
display_supply = 1,000 NRWL
decimals       = 24
base_units     = 1,000 * 10^24 = 10^27
initial_supply = "1000000000000000000000000000"
```

For AGORA:

```text
display_supply = 10,000 AGORA
decimals       = 24
base_units     = 10,000 * 10^24 = 10^28
initial_supply = "10000000000000000000000000000"
```

Before deployment, run a dry-run preview and verify the exact string in the preview output. Do not use numeric JavaScript values for these amounts; keep them as strings to avoid precision loss.

### 2. Correct AGORA's Viktor-required governance claim

AGORA currently says:

```text
Viktor alone holds veto power
Threshold: Viktor REQUIRED + 2 AI
```

But the proposed deploy is:

```json
{
  "owners": ["Viktor", "Rocky", "Codex", "DeepSeek", "Grok"],
  "threshold": 3
}
```

Plain 3-of-5 does **not** enforce Viktor as a required signer. If the contract does not support a required-signer parameter, use one of these safer designs:

1. Viktor-only admin executes all accepted proposals. AI votes are advisory and recorded in AGORA, but cannot move funds.
2. Two-layer control: AI multisig can propose; a separate Viktor timelock/admin wallet executes.
3. Custom multisig extension with an explicit `required_signers: [Viktor]` policy. Do not claim this exists unless the server contract supports it.

Recommended doc change: replace "Viktor required + 2 AI" with "target policy; requires custom multisig support or a two-layer execution model."

### 3. Do not promise treasury/reflection split until verified

NARWHAL describes:

```text
1% to holders
1% to treasury
reflection_fee_bps: 200
```

The local MCP metadata only proves that `reflection_fee_bps` is accepted as an example parameter. It does not prove that the fee is split 50/50 between holders and treasury, nor that a treasury address is configurable.

Required DeepSeek follow-up:

- Locate server-side `advanced_token` implementation.
- Confirm whether `reflection_fee_bps` means all fee goes to holders or whether there are separate treasury/liquidity/burn parameters.
- If no treasury split exists, rewrite NARWHAL economics as "2% reflection to holders" or add a custom token contract requirement.

### 4. Treat social recovery and sunset as unverified requirements

`multisig_wallet` exposes `social_recovery` as a feature toggle, but the docs make stronger claims:

- 72h cooldown before key rotation.
- 2-of-4 remaining owners can rotate keys.
- NARWHAL sunset distribution after 180 days of inactivity.

These are security-critical semantics. They must be verified in server contract code before they are described as deployed behavior.

Recommended wording:

```text
social_recovery=true is requested at deploy, but cooldown, rotation threshold,
and recovery authority must be verified against the contract implementation.
The 180-day sunset clause is a custom-contract requirement, not part of the
standard multisig template unless proven otherwise.
```

### 5. Fix seed custody language

Do not tell agents to store seeds in a shared conventional path as if it were a shared secret location.

Replace:

```text
Each agent stores their seed in ~/.claude/quillon-agent-seed
```

With:

```text
Each agent must have a distinct wallet seed, stored per agent and per host with
0600 permissions. Never reuse one seed across agents. Never commit seeds. If
an MCP process needs a seed, pass it via the existing seed-file/env mechanism
for that agent only.
```

### 6. Add missing `identity_contract` parameter in AGORA

AGORA Phase 1 currently omits `root_authority`. The local MCP template requires it.

Suggested deploy sketch:

```json
{
  "template": "identity_contract",
  "parameters": {
    "root_authority": "<Viktor wallet or governance admin address>"
  }
}
```

Then register agents after the identity contract exists. Do not include `TBD` addresses in owner arrays for actual deploys.

## Suggested DeepSeek code-analysis checklist

DeepSeek should verify these exact implementation points before writing a final deploy plan:

1. Find the server-side parser for `/api/v1/contracts/deploy` and confirm every accepted `contract_type`.
2. Inspect `crates/q-api-server/src/contracts_api.rs` and `crates/q-vm/src/contracts/orobit_smart_contracts.rs`.
3. Inspect the generated/instantiated contract state for `advanced_token`.
4. Confirm whether `admin` is accepted for `advanced_token` and whether it is actually used for minting/upgrades.
5. Confirm whether `reflection_fee_bps`, `burn_fee_bps`, `liquidity_fee_bps`, `max_tx_bps`, and `max_wallet_bps` are enforced at transfer time.
6. Confirm whether `multisig_wallet` has any required-signer, veto, or role-weight support.
7. Confirm whether `timelock` and `spending_limit` are configurable or only boolean flags with server defaults.
8. Confirm whether `social_recovery` is implemented or only stored as metadata.
9. Confirm that all deploy parameter numeric fields are parsed from strings safely where values can exceed JavaScript safe integer range.
10. Add dry-run examples for NARWHAL and AGORA using the exact deploy wrapper.
11. Add a "cannot deploy until" list: DeepSeek wallet created, Grok/Qwen wallet selected, production node stable, and all deploy previews reviewed.

## Safer revised deployment posture

NARWHAL can proceed first as a simpler token experiment if the following are true:

- The initial supply string is corrected.
- All owners are real addresses.
- The multisig is documented honestly as plain M-of-N unless required-signer support is verified.
- Treasury/reflection behavior is rewritten to match the actual token contract.

AGORA should wait. Its core value is coordination, voting, and bounty payout, which depends on exact authorization semantics. Deploying AGORA before the Viktor-required execution model is proven would create a governance contract that looks safer than it is.

## Proposed doc patch summary for DeepSeek

Patch `docs/NARWHAL_TECHNICAL_REVIEW.md`:

- Correct `initial_supply` to `10^27` base units for 1,000 NRWL at 24 decimals.
- Mark sunset clause as custom/unverified.
- Replace precise recovery claims with "verify contract implementation."
- Clarify whether 2% fee is all reflection or split only if treasury split exists.

Patch `docs/AGORA_DESIGN.md`:

- Add `root_authority` to `identity_contract` deploy.
- Replace "Viktor required + 2 AI" with an explicit unimplemented target policy.
- Remove any deploy snippet containing `TBD` owners.
- Add a preflight gate requiring dry-run preview and implementation verification.
