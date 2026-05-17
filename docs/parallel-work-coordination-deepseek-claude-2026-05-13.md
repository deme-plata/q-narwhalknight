# Parallel-Work Coordination: DeepSeek + Claude Code Agents

**Date:** 2026-05-13
**Project:** Quillon Graph (codename NarwhalKnight) — Live Mainnet, ~$2 B USD Market Cap
**Purpose:** Allow DeepSeek and Claude Code agents to make progress on the IVC SNARK track in parallel without stepping on each other.

---

# 🚨 READ FIRST — MAINNET PRODUCTION CONSTRAINTS

These rules are non-negotiable for both teams (DeepSeek + Claude Code agents). Apply identically.

## Hard rules

1. **No database schema migrations on existing column families.** The full list of touched CFs is at `crates/q-storage/src/lib.rs` lines 419–470. Do not modify them. The only new CF in this track is `cf_balance_smt` and it is **already shipped in shadow mode** at `crates/q-storage/src/balance_smt.rs`.
2. **Do not modify `balance_root_v1` computation** anywhere. It is the consensus-critical state-root that the live $2 B chain depends on. `balance_root_v2` (the SMT) runs in shadow mode until `BALANCE_ROOT_V2_HEIGHT` activation ~2 months from now.
3. **Do not modify `save_wallet_balance`, `save_wallet_balances`, `save_wallet_balances_batch`** without explicit Beta sign-off. These are balance-integrity hot paths. The shadow-mode wiring is described in this doc and is intentionally **defensive** — SMT errors must not propagate and break balance writes.
4. **No `unsafe`, no FFI, no `unwrap()` or `panic!()` in non-test code.** Use `Result` everywhere.
5. **Every public function must have ≥ 1 happy-path test + ≥ 3 adversarial tests + 1 cross-check against the production code path** (or `blake3::hash()`, `BalanceSmt::prove()`, etc., as appropriate).
6. **Soak before activation.** Even after code lands and tests pass, ≥ 1 week of mainnet soak before any consensus consumption. CLAUDE.md "Mandatory Testing & Deployment Safety Protocol" applies.

## Coordination rules

7. **Branch your work.** Use `code.quillon.xyz` (the project's local git server, **not GitHub**). Branch names:
   - DeepSeek track: `deepseek/<phase>-<short-description>` (e.g., `deepseek/blake3-multiblock`)
   - Claude Code agent track: `claude/<phase>-<short-description>` (e.g., `claude/merkle-gadget`)
   - Integration branch: `ivc/v1` — merge target.
8. **One file owner per file at a time.** See the file-ownership matrix below. Two teams may NOT modify the same file in the same PR cycle.
9. **Tests adjacent to the code they test.** No test module in a different crate than the function it tests.
10. **All commits sign-off, no GitHub.** `git commit -s -m "feat(<area>): <description>"` then `git update-server-info` on Beta after push so Epsilon can `git pull`.

---

# WHERE THE TRACK IS RIGHT NOW

## Shipped (in `main`, today)

| Component | Path | Status |
|---|---|---|
| `BalanceSmt` storage struct + 12 tests | `crates/q-storage/src/balance_smt.rs` | ✓ Production-grade, shadow mode |
| BLAKE3 single-block gadget | `crates/q-ivc/src/gadgets/blake3.rs` | ✓ 551 LOC, production-grade. SINGLE-BLOCK ONLY. |
| Poseidon gadget | `crates/q-ivc/src/gadgets/poseidon.rs` | ✓ 329 LOC, production-grade |
| NTT butterfly (Cooley-Tukey, negacyclic) | `crates/q-ivc/src/gadgets/ntt.rs` | ✓ 769 LOC. 3 dilithium test fixtures still need `Fr::one()` swap (task #22) |
| Dilithium5 primitives | `crates/q-ivc/src/gadgets/dilithium.rs` | ✓ 902 LOC, production-grade |
| EpochTransitionCircuit skeleton | `crates/q-ivc/src/circuits/epoch_transition.rs` | ◑ 208 LOC, drafted, composition incomplete |
| Archive status endpoint | `crates/q-api-server/src/handlers.rs::archive_status` | ✓ v10.9.18 |

## Blocked / not started

| Component | Path (planned) | Why blocked |
|---|---|---|
| Multi-block BLAKE3 gadget | `crates/q-ivc/src/gadgets/blake3.rs` (extend) | This is the **current gate**. Spec in `docs/deepseek-handoff-blake3-multiblock-2026-05-13.md`. Two DeepSeek iterations produced placeholders. |
| Merkle-path gadget | `crates/q-ivc/src/gadgets/merkle.rs` (new) | Blocked on multi-block BLAKE3 |
| δ-circuit composition | `crates/q-ivc/src/circuits/delta_block.rs` (new) | Blocked on Merkle gadget |
| Nova IVC wrapper | `crates/q-ivc/src/recursion/` (new module) | Blocked on δ-circuit |
| SMT wiring into `save_wallet_balances` | `crates/q-storage/src/lib.rs` | Ready-to-apply patch in §A below. Needs review. |
| TUI fast-readiness banner | `crates/q-tui/src/metrics.rs` + `ui/mod.rs` | Designed at `docs/blueprint-tui-fast-ready-visualization-2026-05-13.md`. Not yet coded. |
| `GET /api/v1/proof/tip` | `crates/q-api-server/src/handlers.rs` | Blocked on Nova wrapper |
| `--bootstrap-mode=proof` CLI | `crates/q-api-server/src/main.rs` | Blocked on `/api/v1/proof/tip` |

---

# FILE OWNERSHIP MATRIX

For each file in the SNARK track, exactly **one** of DeepSeek / Claude Code agent / Beta engineer owns active editing. Other teams may read but not modify until ownership transfers.

| File | Owner | Why |
|---|---|---|
| `crates/q-ivc/src/gadgets/blake3.rs` | **DeepSeek** | Path A multi-block BLAKE3 gadget. Spec in handoff doc. |
| `crates/q-ivc/src/gadgets/merkle.rs` (new) | **DeepSeek** | Blueprint 1B. Starts after multi-block BLAKE3 lands. |
| `crates/q-ivc/src/circuits/delta_block.rs` (new) | **DeepSeek** | Blueprint 2. Starts after Merkle gadget lands. |
| `crates/q-ivc/src/recursion/` (new module) | **Claude Code agent or external ZK engineer** | Nova/HyperNova IVC. Different domain expertise from BLAKE3/Merkle work. |
| `crates/q-ivc/src/gadgets/{dilithium,ntt,poseidon}.rs` | **Beta** | Frozen for the duration. 3 test fixture fixes (task #22) only. |
| `crates/q-ivc/src/circuits/epoch_transition.rs` | **Beta** | Frozen — existing skeleton. Do not replace. |
| `crates/q-storage/src/balance_smt.rs` | **Beta** | Already shipped. **No edits without Beta sign-off.** |
| `crates/q-storage/src/lib.rs` | **Beta** | Storage layer is balance-integrity critical. Shadow-mode SMT wiring patch in §A. |
| `crates/q-api-server/src/handlers.rs` | **Claude Code agent or Beta** | Wire-protocol endpoints. `/api/v1/proof/tip` etc. |
| `crates/q-api-server/src/main.rs` | **Beta** | Route wiring + CLI flags. |
| `crates/q-tui/src/metrics.rs` + `ui/*` | **Claude Code agent** | TUI Blueprint 7 readiness banner. |
| `crates/q-types/src/block.rs` | **Beta** | Block header schema. `state_root_v2: [u8;32]` field gets added here at activation. |

**Conflict resolution:** if two teams need the same file, the Beta engineer arbitrates. Default: whichever team's work is on the critical path (lower in the dependency stack) gets priority.

---

# DEPENDENCY GRAPH

```
┌─────────────────────────────────────────────────────────────────────────┐
│ Multi-block BLAKE3 (DeepSeek)                ── blocks ──>  Merkle      │
│   Path: gadgets/blake3.rs                                   (DeepSeek)  │
│   Status: spec written, 2 placeholder iterations            ──┐         │
│                                                                │         │
│ SMT shadow-mode wiring (Beta)                                 │         │
│   Path: storage/lib.rs + save_wallet_balances                 │         │
│   Status: patch ready in §A below                             │         │
│                                                                ▼         │
│ Archive endpoint (Beta — DONE v10.9.18)              δ-circuit          │
│   Path: handlers.rs::archive_status                  (DeepSeek)         │
│                                                       ──┐               │
│ TUI readiness banner (Claude agent)                     │               │
│   Path: q-tui/{metrics,ui}/                             ▼               │
│                                                      Nova IVC wrapper   │
│ Test debt cleanup (Beta — small)                     (Claude agent)     │
│   Path: gadgets/dilithium.rs tests                    ──┐               │
│                                                          ▼               │
│                                                  /api/v1/proof/tip      │
│                                                  (Claude agent)         │
│                                                       │                  │
│                                                       ▼                  │
│                                               --bootstrap-mode=proof    │
│                                               (Beta)                    │
└─────────────────────────────────────────────────────────────────────────┘
```

Items on the same horizontal line can proceed in parallel. Items below a `──>` depend on the work above completing first.

---

# § A. READY-TO-APPLY PATCH: SMT shadow-mode wiring

This is the Beta-side patch that turns `BalanceSmt` from "shipped but unused" into "shipped and producing real shadow-mode data on every block." It is intentionally **defensive** — every SMT call is wrapped so failures cannot propagate to break the balance write path.

## Files modified

- `crates/q-storage/src/lib.rs` — add CF descriptor, add field, construct instance
- `crates/q-storage/src/lib.rs::save_wallet_balances` — compose SMT update into WriteBatch

## Step 1 — add `cf_balance_smt` to the CF list

In `crates/q-storage/src/lib.rs`, around the CF constants block (line ~419), add a re-export:

```rust
pub use crate::balance_smt::CF_BALANCE_SMT;  // v10.9.18: SMT for balance_root_v2 (shadow mode)
```

In whatever function opens RocksDB with the descriptor list, add `CF_BALANCE_SMT` as a new `ColumnFamilyDescriptor`. Grep for `ColumnFamilyDescriptor::new(CF_` to find the existing list — add one more entry. Use default Options unless we have a specific tuning reason (we don't — SMT writes are small).

## Step 2 — add `balance_smt` field to the storage struct

Find the `pub struct StorageEngine` (or equivalent — whatever struct holds the `Arc<DB>`). Add:

```rust
/// v10.9.18: Sparse Merkle Tree for balance_root_v2 (shadow mode).
/// Maintained on every save_wallet_balances batch. NOT yet consumed by
/// consensus — `state_root_v2` block-header field activates at
/// `BALANCE_ROOT_V2_HEIGHT` (to be defined ~2 months out, post-soak).
pub balance_smt: Arc<crate::balance_smt::BalanceSmt>,
```

## Step 3 — construct it in the storage `new()` function

After `let db: Arc<DB> = ...` and after the DB is opened with `cf_balance_smt` in the descriptor list:

```rust
let balance_smt = match crate::balance_smt::BalanceSmt::open(db.clone()) {
    Ok(smt) => Arc::new(smt),
    Err(e) => {
        // Defensive: log loud but do NOT fail storage construction.
        // The shadow-mode SMT is non-critical at this stage.
        tracing::error!("⚠️  [BALANCE_SMT_V2] Failed to open SMT — shadow mode disabled this run: {}", e);
        // Build an empty placeholder pointing at a temporary in-memory DB?
        // No — just return the error and let the caller decide. If SMT open
        // fails the descriptor list is misconfigured, which should be loud.
        return Err(e.into());
    }
};
```

Then add `balance_smt` to the struct initializer.

## Step 4 — wire `save_wallet_balances`

The current function (around line 4432 in `crates/q-storage/src/lib.rs`) constructs a `WriteBatch` and writes to `cf_wallet_balances`. We compose the SMT update into the same batch.

**Critical defensive pattern:**

```rust
pub async fn save_wallet_balances(&self, balances: &HashMap<[u8; 32], u128>) -> Result<()> {
    // ... existing max-wins filtering, audit logging, etc. ...
    let filtered_updates: Vec<([u8; 32], u128)> =
        balances.iter().map(|(a, b)| (*a, *b)).collect();

    let mut batch = WriteBatch::default();

    // (a) Existing wallet-balance writes — UNCHANGED
    for (addr, balance) in &filtered_updates {
        batch.put_cf(&cf_wallet_balances, addr, balance.to_le_bytes());
    }

    // (b) v10.9.18: shadow-mode SMT update. Defensive — never fails the batch.
    let smt_new_root: Option<[u8; 32]> = match self
        .balance_smt
        .apply_to_batch(&mut batch, &filtered_updates)
    {
        Ok(root) => Some(root),
        Err(e) => {
            // Log loud but proceed without v2 update.
            // The v1 root in the existing storage path is still consensus-authoritative.
            tracing::error!(
                "⚠️  [BALANCE_SMT_V2] apply_to_batch failed (shadow mode, non-critical): {} — skipping v2 update for this block",
                e
            );
            None
        }
    };

    // (c) Commit
    self.db.write(batch).context("save_wallet_balances commit")?;

    // (d) Update SMT cache ONLY on success
    if let Some(new_root) = smt_new_root {
        self.balance_smt.commit_root(new_root);

        // Shadow-mode comparison logger — every 100 blocks log v1 vs v2 root
        if tracing::enabled!(tracing::Level::DEBUG) {
            // compute v1 root (existing path) for comparison
            // ... existing v1 computation ...
            tracing::debug!(
                "[BALANCE_ROOT_SHADOW] v1=<hex> v2={} (height=<n>)",
                hex::encode(&new_root[..8])
            );
        }
    }

    Ok(())
}
```

**Why defensive:** if `BalanceSmt::apply_to_batch` returns an error for any reason (RocksDB read failure, internal invariant violation), the SMT update is skipped but the wallet-balance write still commits. Production balance correctness is unaffected.

**Why the SMT update goes into the SAME `batch`:** atomicity. Either both v1 balances and v2 SMT nodes commit together (success path), or neither commits (RocksDB transaction fails). No partial state.

## Step 5 — shadow-mode comparison logging

Once shadow-mode wiring is live, add a periodic logger (every 100-1000 blocks) that re-computes the v1 root (via the existing `compute_state_root` path) and logs both side-by-side. After ~1 week of clean shadow-mode logs (zero divergences), `BALANCE_ROOT_V2_HEIGHT` can be defined and v2 consensus consumption introduced.

## Risk assessment

- **Storage construction:** if `cf_balance_smt` is missing from the descriptor list, `BalanceSmt::open` fails and the storage engine refuses to start. **Loud, not silent.** Operator sees the error at boot.
- **save_wallet_balances:** SMT errors are caught and logged; balance writes proceed. **Zero impact on consensus.**
- **Atomicity:** v1 + v2 commit together via shared WriteBatch. If RocksDB write fails, neither happens — same correctness as before.
- **Activation height:** consensus consumption gated. Until `block.height >= BALANCE_ROOT_V2_HEIGHT`, no node trusts v2. After activation, v1 deprecates over a 6-month grace window.

---

# § B. READY-TO-APPLY PATCH: TUI fast-readiness banner

This is the Claude-Code-agent-track patch. The full design is at `docs/blueprint-tui-fast-ready-visualization-2026-05-13.md`. Below is the minimal viable patch for v10.9.18 — the readiness banner only. Capability matrix + archive sparkline are follow-ups.

## Files modified

- `crates/q-tui/src/metrics.rs` — add `ReadinessMode` enum + fields
- `crates/q-tui/src/ui/mod.rs` (or `dashboard.rs`, wherever the top of the layout lives) — render the banner above existing content

## Step 1 — add enum + fields to `Metrics`

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReadinessMode {
    Bootstrapping,     // grey
    FastReady,         // green — proof-bootstrap verified
    CheckpointTrust,   // yellow — BAL-001 snapshot accepted
    GenesisSync,       // cyan — sync from height 1
    ArchiveComplete,   // bright green — full history available
}

// Add to Metrics struct (after the existing sync block):
pub readiness_mode: ReadinessMode,
#[serde(skip)]
pub readiness_changed_at: Option<std::time::Instant>,
pub archive_lowest_indexed_height: u64,
pub archive_tip_height: u64,
pub archive_complete: bool,
```

Implement `Default` so existing initialization compiles:

```rust
impl Default for ReadinessMode {
    fn default() -> Self { Self::Bootstrapping }
}
```

## Step 2 — wire the metrics from AppState

In `crates/q-api-server/src/main.rs`, the existing `update_tui_metrics` function populates the TUI's `Arc<RwLock<q_tui::Metrics>>`. Add:

```rust
// v10.9.18: readiness mode from AppState heights
let tip = app_state.current_height_atomic.load(Ordering::Relaxed);
let contiguous = app_state.contiguous_height_atomic.load(Ordering::Relaxed);
let archive_complete = contiguous >= tip && tip > 0;

let new_mode = if archive_complete {
    ReadinessMode::ArchiveComplete
} else if app_state.storage_engine.is_checkpoint_applied().await {
    ReadinessMode::CheckpointTrust
} else if tip > 0 && contiguous < 100 {
    ReadinessMode::GenesisSync
} else {
    ReadinessMode::Bootstrapping
};

let mut m = tui_metrics.write().unwrap();
if new_mode != m.readiness_mode {
    m.readiness_changed_at = Some(std::time::Instant::now());
}
m.readiness_mode = new_mode;
m.archive_tip_height = tip;
m.archive_lowest_indexed_height = contiguous;
m.archive_complete = archive_complete;
```

## Step 3 — render the banner

In the TUI layout (find where `Frame.size()` is split into chunks), add a 3-row block at the top:

```rust
let chunks = Layout::default()
    .direction(Direction::Vertical)
    .constraints([Constraint::Length(3), Constraint::Min(0)])
    .split(area);
draw_readiness_banner(frame, chunks[0], metrics);
draw_existing_content(frame, chunks[1], metrics);
```

Where `draw_readiness_banner`:

```rust
fn draw_readiness_banner(frame: &mut Frame, area: Rect, m: &Metrics) {
    let (color, icon, label, detail) = match m.readiness_mode {
        ReadinessMode::Bootstrapping     => (Color::Gray,       "⏳", "BOOTSTRAPPING",
            "verifying proof / dialing peers".to_string()),
        ReadinessMode::FastReady         => (Color::Green,      "⚡", "FAST-READY",
            "mine · transact · query state".to_string()),
        ReadinessMode::CheckpointTrust   => (Color::Yellow,     "📜", "CHECKPOINT-TRUST",
            "mine · transact · query state".to_string()),
        ReadinessMode::GenesisSync       => (Color::Cyan,       "🌅", "GENESIS-SYNC",
            format!("syncing from height 1 · at {}/{}",
                m.archive_lowest_indexed_height, m.archive_tip_height)),
        ReadinessMode::ArchiveComplete   => (Color::LightGreen, "⚓", "ARCHIVE-COMPLETE",
            "full history · all queries available".to_string()),
    };
    let flash = m.readiness_changed_at
        .map(|t| t.elapsed() < std::time::Duration::from_millis(300))
        .unwrap_or(false);
    let style = if flash {
        Style::default().fg(Color::Black).bg(color).add_modifier(Modifier::BOLD)
    } else {
        Style::default().fg(color).add_modifier(Modifier::BOLD)
    };
    let line = Line::from(vec![
        Span::styled(format!("  {}  ", icon), style),
        Span::styled(format!("{:<18}", label), style),
        Span::raw("  "),
        Span::styled(detail, Style::default().fg(Color::Gray)),
    ]);
    let block = Block::default().borders(Borders::ALL).border_style(style);
    frame.render_widget(Paragraph::new(line).block(block), area);
}
```

## What's deferred for v10.9.19+

- Capability matrix (4-line panel showing mine ✓ / transact ✓ / query state ✓ / query history ◐)
- Archive backfill sparkline (60-character rolling window of blocks/sec)
- The proof-verified transition animation (300 ms flash when proof verifies — needs Nova wrapper first)

---

# § C. PARALLEL WORK PLAN

## DeepSeek track — start here

**Priority 1 — multi-block BLAKE3 gadget** (handoff doc: `docs/deepseek-handoff-blake3-multiblock-2026-05-13.md`)

The current blocker. Build on top of existing `Blake3Gadget::compress`. Implement single-chunk multi-block + tree-mode chunking. Required tests: cross-check against native `blake3::hash()` byte-for-byte. **Without this, Blueprint 1B (Merkle gadget) cannot move.**

After this lands:

**Priority 2 — Merkle-path gadget** (Blueprint 1B in `docs/blueprints-ivc-snark-2026-05-13.md`)

Uses the multi-block BLAKE3 from Priority 1. Implements `MerklePathGadget::{leaf_hash, compute_root, enforce_membership}`. Tests use `BalanceSmt::prove()` from the shipped SMT to generate real fixtures (NOT mocked). Required dev-dep: `q-storage` and `tempfile`.

**Priority 3 — δ-circuit composition** (Blueprint 2)

Wires existing gadgets (BLAKE3, Dilithium5, NTT, Merkle) into one R1CS circuit that proves a block transition. Per Blueprint 2 in the blueprints doc.

## Claude Code agent track — runs in parallel

**Track A — Nova IVC wrapper** (Blueprint 3)

Wait for δ-circuit to land, then wrap it in Nova's `StepCircuit` trait. Choose between `nova-snark` (Microsoft) and `arkworks/nova` after spending 1 day prototyping a toy circuit on both. The folding loop should fold one block per step. Verifier should produce constant-cost (≤ 10 ms) proofs.

**Track B — TUI readiness banner** (Blueprint 7)

Spec in §B above. Self-contained, no SNARK dependency, lights up the `CHECKPOINT-TRUST` / `ARCHIVE-COMPLETE` states from existing AppState data. The `FAST-READY` state activates later when Nova lands.

**Track C — wire protocol** (Blueprint 5)

Wait for Nova wrapper, then add `GET /api/v1/proof/tip`, `POST /api/v1/proof/verify`, gossipsub topic `/qnk/mainnet-genesis/recursive-proof`, and the `--bootstrap-mode=proof` CLI flag.

## Beta track — coordination + integration + safety

- Apply §A SMT shadow-mode wiring after reviewing the patch
- Soak shadow-mode for ≥ 1 week, compare v1 vs v2 roots block-by-block
- Define `BALANCE_ROOT_V2_HEIGHT` after ≥ 1 week of zero-divergence shadow data
- Add `state_root_v2: [u8; 32]` field to block header (gated by activation height)
- Approve all merges from DeepSeek and Claude Code agent tracks
- Final mainnet activation after Phase 3 advisory mode succeeds

---

# § D. INTERFACE CONTRACTS (so the tracks don't drift)

These types and behaviors are stable. Both tracks build against them.

## `SmtProof` (already shipped, do not change)

```rust
// crates/q-storage/src/balance_smt.rs
pub struct SmtProof {
    pub addr: [u8; 32],
    pub balance: u128,
    pub siblings: [[u8; 32]; 256],
    pub empty_bitmap: [u8; 32],  // bit i = 1 iff sibling at depth i+1 is empty subtree
}
impl SmtProof {
    pub fn verify(&self, expected_root: &[u8; 32]) -> bool;
}
```

The Merkle gadget consumes this exact byte layout. Tests in the gadget call `BalanceSmt::prove()` → `SmtProof` → feed into the in-circuit gadget. Do not change `SmtProof` fields.

## `Blake3Gadget::hash_message` (new — DeepSeek to deliver)

```rust
// crates/q-ivc/src/gadgets/blake3.rs
impl Blake3Gadget {
    /// Universal hash entry point. Any byte-length input → 8-word BLAKE3 digest.
    /// MUST byte-for-byte match `blake3::hash(input).as_bytes()` for ALL inputs.
    pub fn hash_message<F: PrimeField>(
        cs: ConstraintSystemRef<F>,
        msg: &[UInt8<F>],
    ) -> Result<Vec<UInt32<F>>, SynthesisError>;
}
```

Merkle gadget calls this. δ-circuit calls this. Nova wrapper does NOT call this directly. The byte-order convention is BLAKE3-native (little-endian 32-bit words internally; final digest as flat byte stream in BLAKE3 spec order). Cross-check test against `blake3::hash()` is **mandatory**.

## `MerklePathGadget::enforce_membership` (new — DeepSeek to deliver after multi-block BLAKE3)

Signature per Blueprint 1B in `docs/blueprints-ivc-snark-2026-05-13.md`. Stable.

## `DeltaBlockCircuit` (new — DeepSeek to deliver after Merkle gadget)

Public inputs: `(state_root_prev, state_root_next, block_header_hash, block_height)`. Witness: block body + Dilithium signatures + 4 Merkle paths per tx + coinbase. Per Blueprint 2.

## `GET /api/v1/proof/tip` response (new — Claude agent track after Nova)

```json
{
  "tip_height": 11400000,
  "state_root": "0xabcd…",
  "block_header": { "height": 11400000, "state_root": "0xabcd…", ... },
  "proof_version": "nova-bn254-v1",
  "proof_size_bytes": 51232,
  "proof_b64": "<base64-encoded proof>"
}
```

Stable JSON schema. The `proof_version` string changes when the lattice migration happens (Phase 4): `nova-bn254-v1` → `latticefold-modulesis-v1` (or similar).

---

# § E. WHAT'S ACTUALLY READY FOR DEEPSEEK TO START ON RIGHT NOW

The list, in execution order:

1. **Read** `docs/deepseek-handoff-blake3-multiblock-2026-05-13.md` end-to-end. It has the spec for multi-block BLAKE3 in detail — chunking, flag handling, byte ordering, test plan, acceptance criteria.
2. **Implement** the multi-block BLAKE3 gadget in `crates/q-ivc/src/gadgets/blake3.rs`. **DO NOT MODIFY** the existing `compress`, `g_function`, `verify_hash`, or `alloc_*` functions. Add new public function `hash_message` and any private helpers needed. Existing single-block functions stay.
3. **Write 10 required tests** per the handoff doc. Cross-check every test against `blake3::hash()` byte-for-byte. Empty input, 64-byte input, 75-byte (Merkle node shape), 1024-byte boundary, 1025-byte (forces tree mode), 4097-byte multi-level tree, plus adversarial cases.
4. **Push to `code.quillon.xyz`** on branch `deepseek/blake3-multiblock`. Do NOT push to GitHub. Run `git update-server-info` after push.
5. **Notify Beta** in the project Discord `#dev-coordination` channel that the PR is ready for review.

**Acceptance criteria checklist** (copy this into the PR description):
- [ ] `cargo check --package q-ivc` clean
- [ ] `cargo test --package q-ivc` all pass (old + new)
- [ ] No file outside `crates/q-ivc/src/gadgets/blake3.rs` modified
- [ ] No `Cargo.toml` changes
- [ ] Existing public functions unchanged
- [ ] Empty input cross-check passes
- [ ] 64-byte input matches `verify_hash` output
- [ ] 75-byte Merkle-node-shape cross-check passes
- [ ] Multi-chunk (1025-byte) tree mode cross-check passes
- [ ] Constraint count for 75-byte input in 50K-150K range

**Total estimated effort: 1-3 working days** for a competent ZK engineer, longer if BLAKE3 spec subtleties bite (which is why the handoff doc lists the 10 most common ways this work goes wrong).

---

# § F. WHAT'S READY FOR CLAUDE CODE AGENTS TO START ON RIGHT NOW

In parallel with DeepSeek's Priority 1:

**Agent A — TUI readiness banner** (Blueprint 7, §B above)

Self-contained. Reads from `AppState.current_height_atomic` and `AppState.contiguous_height_atomic`. Renders the banner above existing TUI panes. No SNARK dependency. ~3 days.

**Agent B — q-flux logrotate hygiene**

The access log on Epsilon is 2.6 TB on the 80 TB partition (no fire, but no rotation). Add a `/etc/logrotate.d/q-flux` entry with daily rotation, 7-day retention, compress old. Small ops chore, ~1 hour.

**Agent C — q-ivc test fixture cleanup (task #22)**

Three dilithium test fixtures still use `Fr::neg_one()` for `roots[1]` in the `n=2` case. Should be `Fr::one()` per the NTT convention fix from 2026-05-12. Plus `test_use_hint_with_bias` has a math error to recompute via the FIPS 204 §6.5.2 formula. Small, mechanical, ~1.5 days.

**Agent D — instrumentation for Beta investigation** (Blueprint TBD)

Add periodic peer-state dumper task (every 10s logs all connected peers with multiaddrs + pending dial counts) and per-block-pack-request timing instrumentation. ~80 LOC, no production behavior change. Goes into `crates/q-network/src/unified_network_manager.rs`.

**Agent E — WASM browser verifier scaffolding** (full handoff: `docs/deepseek-handoff-wasm-browser-verifier-2026-05-13.md`)

Build a new crate `crates/q-ivc-verifier-wasm/` that compiles to wasm32 + a TypeScript wrapper at `gui/quantum-wallet/src/ivc/` that fetches `/api/v1/proof/tip`, verifies in-browser via the WASM module, and caches the verified state-root in IndexedDB. Phase 1 ships a placeholder verifier (returns true for non-empty proof) so the WASM build pipeline + JS API + caching are wired end-to-end. Phase 2 (months out) swaps in real Nova verification — a single-file change in `lib.rs`. Independent of DeepSeek's BLAKE3 track. ~2-3 days for Phase 1.

**This is the gap the whitepaper highlighted:** the whitepaper claims ≤250 ms WASM verification but we have zero WASM infrastructure today. Phase 1 of Agent E closes that gap on the *plumbing* side. The cryptographic security follows when Nova lands.

---

# § G. HOW TO COORDINATE — DAILY MECHANICS

1. **Morning standup** in `#dev-coordination` (project Discord): each team posts (a) what they shipped yesterday, (b) what they're working on today, (c) what they're blocked on.
2. **PR merges** via Beta. Push to `code.quillon.xyz`, post the branch name in `#dev-coordination`, wait for Beta to review + merge into `ivc/v1`.
3. **Integration testing** — every Friday, Beta merges `ivc/v1` into a test branch and runs the full test suite on Epsilon Docker (Debian 12). Anything that breaks the test suite gets reverted from `ivc/v1` until fixed.
4. **No direct merges to main** until ≥ 1 week of clean `ivc/v1` test suite.
5. **Mainnet deploy** only after `safe-deploy.sh` runs all 4000+ mainnet-safety tests (per CLAUDE.md "Mandatory Testing & Deployment Safety Protocol").

---

# § H. WHAT MUST NOT HAPPEN

These are kill-switch conditions:

1. **`balance_root_v1` computation modified** in any code path. If anyone touches `compute_state_root` in `crates/q-storage/`, abort the PR.
2. **`save_wallet_balance` or `save_wallet_balances` modified without the defensive pattern in §A.** SMT errors must never propagate to break wallet balance writes.
3. **`UInt8` → `UInt32` allocation via fresh-witness in the Merkle gadget.** Must use `UInt32::from_bits_le(byte.to_bits_le())` pattern. Allocating a fresh witness breaks soundness — circuit accepts wrong inputs silently.
4. **DeepSeek pushes placeholder code** (`// implement here ...` comments instead of actual implementation). Beta will reject the PR and re-prompt with a tighter spec.
5. **GitHub push.** Use `code.quillon.xyz`. GitHub is not authoritative for this project.

If any of (1)-(4) happens, the PR is rejected without negotiation. Continued violations get the contributor moved off the SNARK track.

---

# CHANGELOG

- **v10.9.18** (today): archive-status endpoint shipped, this coordination doc, TUI readiness banner specced (not yet coded), SMT shadow-mode wiring specced (not yet applied).
- v10.9.17: critical swarm-drive fix for warmup loop. Peer discovery now works.
- v10.9.16: extended bootstrap warmup + `--from-genesis` CLI flag.
- v10.9.15: Quillon Graph boot banner.
- v10.9.14: Phase 2 memory budget.

— Quillon Graph maintainers, 2026-05-13
