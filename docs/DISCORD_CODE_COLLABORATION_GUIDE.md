# :computer: Q-NarwhalKnight Code Collaboration Guide

> **Start contributing to quantum-enhanced consensus in 5 minutes.**
> Browse, fork, and submit fixes via `code.quillon.xyz` -- open to developers and AI agents alike.

---

## 1. :rocket: Getting Started

### Browse the Code

Visit **https://code.quillon.xyz** in your browser to explore the full codebase -- files, branches, commit history, and diffs.

### Clone the Repository

```bash
git clone https://code.quillon.xyz/repo.git q-narwhalknight
cd q-narwhalknight
```

### Build Requirements

- **Rust** 1.75+ (install via https://rustup.rs)
- **Node.js** 18+ (for the wallet frontend)
- **Linux x86_64** recommended (post-quantum crypto crates require native compilation)

### Quick Smoke Test

```bash
# Verify the project compiles (first build takes 15-30 min due to PQ crypto)
cargo check --package q-api-server

# Run the test suite
cargo test --workspace
```

> :warning: **The repository is READ-ONLY.** You cannot `git push` directly. Contributions are submitted via the MCP integration (see below) or as patch files.

---

## 2. :robot: Contributing with Claude Code (AI-Powered Workflow)

Claude Code can read, search, and submit contributions to Q-NarwhalKnight through the MCP (Model Context Protocol) server at `code.quillon.xyz`.

### Install Claude Code

```bash
npm install -g @anthropic-ai/claude-code
```

### Configure the MCP Server

Add this to your Claude Code MCP config (`~/.claude/mcp.json` or project-level `.mcp.json`):

```json
{
  "mcpServers": {
    "quillon-code": {
      "type": "sse",
      "url": "https://code.quillon.xyz/mcp/sse"
    }
  }
}
```

### Available MCP Tools

Once connected, Claude Code has access to these tools:

| Tool | Description |
|------|-------------|
| `read_file` | Read any file from the repo by path |
| `search_code` | Full-text search across the codebase |
| `list_files` | List files in a directory |
| `list_branches` | View all branches |
| `view_diff` | See the diff between branches or commits |
| `submit_contribution` | Submit a proposed change (diff + description) |
| `list_contributions` | View pending contributions |

### Example Workflow

```
You: "Find where mining rewards are calculated and check for overflow bugs"

Claude Code will:
1. search_code("mining reward") -> finds crates/q-storage/src/emission_controller.rs
2. read_file("crates/q-storage/src/emission_controller.rs") -> reads the code
3. Identifies a potential issue
4. submit_contribution({
     title: "Fix potential overflow in emission calculation",
     description: "The reward_per_block multiplication can overflow u128 when...",
     diff: "--- a/crates/q-storage/src/emission_controller.rs\n+++ b/..."
   })
```

> :bulb: **Contributions are proposals only.** Maintainers review every submission before merging. You will not break anything by submitting.

---

## 3. :wrench: Contributing Without Claude Code (Manual Workflow)

### Step 1: Clone and Branch

```bash
git clone https://code.quillon.xyz/repo.git q-narwhalknight
cd q-narwhalknight
git checkout -b fix/my-bugfix
```

### Step 2: Make Your Changes

Edit the code, fix the bug, add the feature.

### Step 3: Test Locally

```bash
# Must pass before submitting
cargo check --workspace
cargo test --workspace
```

### Step 4: Generate a Patch

```bash
git add -A
git diff --cached > my-fix.patch
```

### Step 5: Submit

Choose one of these options:

- **Bounty dApp** -- Upload your patch at **https://quillon.xyz** (bounty program page)
- **Discord** -- Post your `.patch` file in the `#contributions` channel with a description
- **Email** -- Send to the address listed on code.quillon.xyz

---

## 4. :trophy: Bounty Program Integration

The Q-NarwhalKnight bounty program rewards bug reports, code fixes, and community contributions with points redeemable for QUG tokens.

### How to Earn Points

| Action | Points |
|--------|--------|
| :red_circle: **Critical** bug report (consensus, data loss, funds at risk) | **50 pts** |
| :orange_circle: **High** severity (balance errors, P2P security, sync issues) | **30 pts** |
| :yellow_circle: **Medium** severity (API bugs, UI issues, performance) | **15 pts** |
| :green_circle: **Low** severity (typos, cosmetic, docs) | **5 pts** |
| Merged code contribution | **10-100 pts** (based on scope) |
| Social activity (reviews, discussions) | **1-5 pts** |

### Submitting a Bug Report

1. Go to **https://quillon.xyz** and navigate to the bounty dApp
2. Click **Submit Bug Report**
3. Fill in:
   - **Issue URL** -- link to the relevant file or line on `code.quillon.xyz`
   - **Severity** -- Critical / High / Medium / Low
   - **Description** -- clear steps to reproduce
4. You receive a report ID and points immediately upon submission

### Linking Code References

When filing a bug, reference the exact file path from the repo:

```
Bug in: crates/q-storage/src/emission_controller.rs line 142
The BASE_ANNUAL_EMISSION constant has 3 extra zeros...
```

---

## 5. :mag: Code Review Process

All contributions -- whether from AI agents or human developers -- go through the same review pipeline.

### Review Criteria

- **Consensus Safety** -- Does this change affect block validation? If yes, is it height-gated?
- **Balance Integrity** -- Could this cause funds to appear, disappear, or duplicate?
- **P2P Security** -- Could a malicious peer exploit this?
- **Backward Compatibility** -- Do existing blocks and data still validate?
- **Test Coverage** -- Are there tests for the new/changed behavior?

### What Maintainers Check

```bash
# Every contribution is tested against:
cargo check --workspace                    # Compiles?
cargo test --workspace                     # All tests pass?
cargo test --package q-storage --test mainnet_critical_tests   # Safety tests?
cargo clippy -- -D warnings                # No lint warnings?
```

### Review Timeline

- **Critical fixes** (security, data loss) -- reviewed within hours
- **High priority** (balance, P2P) -- reviewed within 1-2 days
- **Standard contributions** -- reviewed within 1 week

---

## 6. :building_construction: Project Structure

Quick map of the codebase for new contributors:

```
q-narwhalknight/
|
|-- crates/
|   |-- q-api-server/       # REST API server, SSE streaming, all HTTP handlers
|   |-- q-storage/           # RocksDB persistence, block writer, balance tracking
|   |-- q-types/             # Shared types: Block, Transaction, NetworkId, etc.
|   |-- q-network/           # libp2p P2P networking, gossipsub, Kademlia DHT
|   |-- q-miner/             # CPU/GPU mining binary (standalone miner)
|   |-- q-dag-knight/        # DAG-Knight consensus ordering algorithm
|   |-- q-narwhal-core/      # Narwhal mempool with reliable broadcast
|   |-- q-vm/                # Smart contract VM (WASM-based, Orobit contracts)
|   |-- q-dex/               # Decentralized exchange with AMM and ZK privacy
|   |-- q-wallet/            # Wallet logic, key management, hybrid crypto
|   |-- q-consensus-guard/   # Height-gated upgrade system for mainnet safety
|   |-- q-mining-pool/       # Distributed mining pool coordination
|   |-- q-bounty-protocol/   # Bounty system backend (RocksDB + AEGIS-QL)
|   |-- q-bounty-server/     # Bounty REST API (Axum)
|   |-- q-ai-inference/      # Distributed AI inference (BitNet, llama.cpp)
|   |-- q-tor-client/        # Embedded Tor client (arti-based)
|   |-- q-tor-circuit/       # Dedicated Tor circuit management
|   |-- q-dandelion/         # Dandelion++ gossip for traffic analysis resistance
|   |-- q-crypto-advanced/   # FROST threshold sigs, AEGIS-256, timelock encryption
|   |-- q-zk-stark/          # ZK-STARK proofs with GPU acceleration
|   |-- q-zk-snark/          # ZK-SNARK toolkit (Groth16, PLONK, Marlin)
|   `-- ...                  # 40+ crates total
|
|-- gui/quantum-wallet/      # React + TypeScript frontend (wallet, DEX, explorer)
|   |-- src/components/      # All UI components
|   |-- src/services/api.ts  # Backend API client
|   `-- src/libp2p/          # Browser-side P2P (libp2p-js)
|
|-- bounty-dapp/             # Bounty program frontend (React)
|-- api-docs/                # Interactive API documentation (React)
|-- papers/                  # Academic whitepapers and technical papers
`-- scripts/                 # Deployment, testing, and maintenance scripts
```

### Key Files for New Contributors

| File | What It Does |
|------|--------------|
| `crates/q-api-server/src/main.rs` | Node entry point, P2P setup, sync logic |
| `crates/q-api-server/src/handlers.rs` | All REST API endpoint handlers (~13k lines) |
| `crates/q-storage/src/lib.rs` | Storage layer API (blocks, balances, state) |
| `crates/q-types/src/block.rs` | Block and Transaction struct definitions |
| `crates/q-types/src/lib.rs` | NetworkId, Phase, shared enums |
| `crates/q-storage/src/emission_controller.rs` | Mining reward calculation |
| `gui/quantum-wallet/src/components/Dashboard.tsx` | Main wallet dashboard UI |
| `gui/quantum-wallet/src/components/DexScreen.tsx` | DEX trading interface |

---

## 7. :link: Quick Reference Links

| Resource | URL |
|----------|-----|
| **Browse Code** | https://code.quillon.xyz |
| **Wallet / dApp** | https://quillon.xyz |
| **Bounty Program** | https://quillon.xyz (bounty tab) |
| **Download Node** | https://quillon.xyz/downloads/q-api-server-linux-x86_64 |
| **Discord** | (see pinned invite) |

---

## 8. :bulb: Tips for AI Agent Contributors

- **Read before writing.** Always use `read_file` or `search_code` to understand existing patterns before proposing changes.
- **Match the style.** The codebase uses Rust 2021 edition, extensive `anyhow` error handling, and `tokio` async runtime.
- **Height-gate consensus changes.** Any change to block validation MUST be wrapped in a block-height check (see `q-consensus-guard`). This is the number one reason contributions get rejected.
- **Test your logic.** Include test cases in your contribution description. Maintainers will verify them.
- **Small, focused diffs.** One fix per contribution. Large multi-file refactors are harder to review and more likely to be rejected.
- **Decimal awareness.** Token amounts use 24-decimal internally. Never hardcode decimal values -- always read from contract metadata. This is the most common source of bugs in the codebase.

---

*Built with post-quantum cryptography. Reviewed by humans and AI alike.*
