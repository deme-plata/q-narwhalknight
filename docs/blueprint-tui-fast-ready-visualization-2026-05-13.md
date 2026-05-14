# Blueprint 7: TUI "Fast-Ready" Readiness Visualization

**Date:** 2026-05-13
**Companion to:** `blueprints-ivc-snark-2026-05-13.md` (the proof-bootstrap blueprints)
**Audience:** External implementer (DeepSeek)
**Goal:** When a node boots via `--bootstrap-mode=proof` and the recursive SNARK verifies, the TUI must *visibly* tell the operator: **"You can mine. You can send. You can transact. Right now."** — even though archive backfill is still running.

---

## Why This Matters

The proof-bootstrap is invisible cryptography. The user clicks "join", sees a boot banner, and… something happens in 10 ms. Without UI feedback, they don't know what changed. They might still think "I need to wait for sync."

The TUI is the operator's window into node state. It has to make the **capability transition** legible:

```
   T-0     proof downloaded
   T+0.01  proof verified  ← THIS IS THE MAGIC MOMENT
   T+0.02  node accepts mining requests
   T+0.02  node accepts transactions
   T+0.02  node serves balance queries
   T+0.02  archive backfill begins in background (will run for hours)
```

The user should *see* T+0.01 happen.

---

## Three New Things to Show

### A. The Readiness Banner (top of screen, persistent)

A single line above the existing status bar that occupies one row and changes color + content based on the **operational mode**. Five visual states:

```
╭─────────────────────────────────────────────────────────────────────────╮
│ ⏳  BOOTSTRAPPING        verifying recursive proof…                     │  ← grey
╰─────────────────────────────────────────────────────────────────────────╯

╭─────────────────────────────────────────────────────────────────────────╮
│ ⚡  FAST-READY           mine · transact · query state │ archive 0.3%   │  ← green
╰─────────────────────────────────────────────────────────────────────────╯

╭─────────────────────────────────────────────────────────────────────────╮
│ 📜  CHECKPOINT-TRUST     mine · transact · query state │ no proof avail │  ← yellow
╰─────────────────────────────────────────────────────────────────────────╯

╭─────────────────────────────────────────────────────────────────────────╮
│ 🌅  GENESIS-SYNC         syncing from height 1 · ETA 5h 23m             │  ← cyan
╰─────────────────────────────────────────────────────────────────────────╯

╭─────────────────────────────────────────────────────────────────────────╮
│ ⚓  ARCHIVE-COMPLETE     full history · all queries available           │  ← bright green
╰─────────────────────────────────────────────────────────────────────────╯
```

### B. Capability Matrix (3 lines below the banner, only visible on dashboard pane)

Four operational capabilities, each one-line:

```
  ✓ MINE              ready · current difficulty: 0x000007ff…
  ✓ TRANSACT          ready · mempool: 234 txs · nonce next: 17
  ✓ QUERY STATE       state_root verified · 1.2M wallets indexed
  ◐ QUERY HISTORY     backfilling · 4.32M / 11.4M blocks (37.9%)
```

Each capability has an icon and a single-sentence status. The history line gets the `◐` half-icon when backfilling; the others get `✓` (ready) or `✗` (unavailable, with reason).

### C. Archive Progress Sparkline (1 line, only in dashboard)

A 60-character-wide horizontal sparkline showing backfill rate over the last 60 minutes:

```
  ▁▁▂▃▄▅▆▇█▇▆▅▄▅▆▇█▇▇▇▆▅▄▃▂▂▂▃▄▅▆▇▇▇▆▅▄▃▂▁▂▃▄▅▆▇█▇▆▅▄▃▂▁▁▁  175 b/s · ETA 18h
```

This gives the operator a sense of whether the backfill is healthy (steady) or degrading (declining). Uses the existing pattern from `crates/q-tui/src/ui/network.rs` for sparkline rendering.

---

## The Magic Moment (proof verification transition)

When the proof verifies (state changes BOOTSTRAPPING → FAST-READY), the banner animates briefly:

- Banner row flashes bright green for 300ms
- A 1-line transient toast appears below the banner for 3 seconds:
  > `⚡ proof verified in 8.4 ms · node is operational`
- Then settles into the steady-state green FAST-READY banner

This is the only animation. No bouncing icons, no progress spinners that don't reflect real work. The flash signifies that something genuinely happened — the node went from "trust me, I'm fine" to "I have cryptographic proof I'm fine."

Implementation: a `last_state_change_at: Option<Instant>` field on the Metrics struct; the banner renderer checks `state_change_at.elapsed() < Duration::from_millis(300)` and renders the flash variant if so.

---

## Data Model Changes — `crates/q-tui/src/metrics.rs`

Add to the `Metrics` struct:

```rust
// Readiness state
pub readiness_mode: ReadinessMode,
pub readiness_changed_at: Option<std::time::Instant>,

// Proof state (None until proof bootstrap completes)
pub proof_verified_height: Option<u64>,
pub proof_verified_at: Option<std::time::Instant>,
pub proof_verify_duration_ms: Option<f64>,
pub proof_version: Option<String>,     // "nova-bn254-v1" etc.

// Capability flags
pub cap_mine: CapabilityState,
pub cap_transact: CapabilityState,
pub cap_query_state: CapabilityState,
pub cap_query_history: CapabilityState,

// Archive backfill (separate from forward sync)
pub archive_lowest_indexed_height: u64,
pub archive_tip_height: u64,
pub archive_complete: bool,
pub archive_blocks_per_sec_recent: f32,
pub archive_eta_seconds: Option<u64>,
pub archive_history_sparkline: [f32; 60],   // last 60 minutes, blocks/sec per minute
```

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReadinessMode {
    Bootstrapping,     // grey  — initial state, verifying or fetching proof
    FastReady,         // green — proof verified, node operational, archive backfilling
    CheckpointTrust,   // yellow — checkpoint snapshot accepted, no proof verified
    GenesisSync,       // cyan — full sync from height 1
    ArchiveComplete,   // bright green — proof verified AND archive backfill complete
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CapabilityState {
    Ready,                              // ✓
    Partial { reason: &'static str },   // ◐ — e.g., "backfilling"
    Unavailable { reason: &'static str }, // ✗
}
```

---

## File Layout

```
crates/q-tui/src/metrics.rs            modify — extend Metrics + new enums
crates/q-tui/src/ui/dashboard.rs       modify — render banner + capability matrix + sparkline
crates/q-tui/src/ui/mod.rs             modify — render banner above all panes (persistent)
crates/q-tui/src/app.rs                modify — drive readiness transitions
crates/q-api-server/src/main.rs        modify — populate new Metrics fields from AppState
crates/q-api-server/src/handlers.rs    modify — new GET /api/v1/status/readiness endpoint
```

---

## Backend Wiring (`crates/q-api-server/src/main.rs`)

The TUI is updated via the `update_tui_metrics` function (already exists, see `crates/q-api-server/src/main.rs:960`). Extend it:

```rust
async fn update_tui_metrics(
    app_state: &Arc<AppState>,
    tui_metrics: &Arc<RwLock<q_tui::Metrics>>,
) {
    // ... existing fields ...

    let mut m = tui_metrics.write().unwrap();

    // Readiness mode
    let new_mode = compute_readiness_mode(app_state);
    if new_mode != m.readiness_mode {
        m.readiness_changed_at = Some(std::time::Instant::now());
    }
    m.readiness_mode = new_mode;

    // Proof state
    if let Some(pv) = app_state.proof_verifier.as_ref() {
        let snapshot = pv.snapshot().await;
        m.proof_verified_height = snapshot.verified_height;
        m.proof_verified_at = snapshot.verified_at;
        m.proof_verify_duration_ms = snapshot.verify_duration_ms;
        m.proof_version = snapshot.proof_version;
    }

    // Capability flags
    m.cap_mine          = if app_state.is_mining_ready() { CapabilityState::Ready } else { /* reason */ };
    m.cap_transact      = if app_state.is_tx_ready()     { CapabilityState::Ready } else { /* reason */ };
    m.cap_query_state   = if app_state.is_state_ready()  { CapabilityState::Ready } else { /* reason */ };
    m.cap_query_history = compute_history_capability(app_state);

    // Archive
    let archive = app_state.archive_status();
    m.archive_lowest_indexed_height = archive.lowest_indexed_height;
    m.archive_tip_height = archive.tip_height;
    m.archive_complete = archive.complete;
    m.archive_blocks_per_sec_recent = archive.blocks_per_sec_recent;
    m.archive_eta_seconds = archive.eta_seconds;
    // Shift sparkline window
    shift_sparkline_window(&mut m.archive_history_sparkline, archive.blocks_per_sec_recent);
}

fn compute_readiness_mode(app_state: &AppState) -> ReadinessMode {
    if app_state.is_archive_complete() && app_state.has_proof_verified() {
        ReadinessMode::ArchiveComplete
    } else if app_state.has_proof_verified() {
        ReadinessMode::FastReady
    } else if app_state.bootstrap_mode == BootstrapMode::Checkpoint {
        ReadinessMode::CheckpointTrust
    } else if app_state.is_syncing_from_genesis() {
        ReadinessMode::GenesisSync
    } else {
        ReadinessMode::Bootstrapping
    }
}
```

The capability predicates `is_mining_ready` etc. are simple checks against existing AppState — most already exist conceptually (the live system already knows when mining is ready). This is plumbing, not invention.

---

## Banner Renderer (`crates/q-tui/src/ui/mod.rs`)

The banner sits **above** all panes — it's the topmost row of the TUI, always visible regardless of which pane the user is on. ratatui pattern:

```rust
use ratatui::layout::{Constraint, Direction, Layout};

pub fn draw_layout(frame: &mut Frame, metrics: &Metrics) {
    let area = frame.size();
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),  // readiness banner (3 rows incl borders)
            Constraint::Min(0),     // existing content
        ])
        .split(area);

    draw_readiness_banner(frame, chunks[0], metrics);
    draw_existing_content(frame, chunks[1], metrics);
}

fn draw_readiness_banner(frame: &mut Frame, area: Rect, m: &Metrics) {
    let (color, icon, label, detail) = match m.readiness_mode {
        ReadinessMode::Bootstrapping     => (Color::Gray,         "⏳", "BOOTSTRAPPING",     bootstrap_detail(m)),
        ReadinessMode::FastReady         => (Color::Green,        "⚡", "FAST-READY",        fast_ready_detail(m)),
        ReadinessMode::CheckpointTrust   => (Color::Yellow,       "📜", "CHECKPOINT-TRUST",  checkpoint_detail(m)),
        ReadinessMode::GenesisSync       => (Color::Cyan,         "🌅", "GENESIS-SYNC",      genesis_detail(m)),
        ReadinessMode::ArchiveComplete   => (Color::LightGreen,   "⚓", "ARCHIVE-COMPLETE",  archive_complete_detail(m)),
    };

    // Flash brighter for 300ms after a state change
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
    let paragraph = Paragraph::new(line).block(block);
    frame.render_widget(paragraph, area);
}

fn fast_ready_detail(m: &Metrics) -> String {
    let pct = if m.archive_tip_height > 0 {
        (m.archive_lowest_indexed_height as f64 / m.archive_tip_height as f64) * 100.0
    } else { 0.0 };
    format!(
        "mine · transact · query state │ archive {:.1}% ({}/{})",
        100.0 - pct,
        format_height(m.archive_lowest_indexed_height),
        format_height(m.archive_tip_height),
    )
}
```

(The other `*_detail` functions are similar one-liners.)

---

## Capability Matrix Renderer (`crates/q-tui/src/ui/dashboard.rs`)

Add a new section to the dashboard pane, rendered between the existing height/peers panel and the log pane:

```rust
fn draw_capability_matrix(frame: &mut Frame, area: Rect, m: &Metrics) {
    let lines = vec![
        cap_line("MINE",          &m.cap_mine,          mine_status_detail(m)),
        cap_line("TRANSACT",      &m.cap_transact,      transact_status_detail(m)),
        cap_line("QUERY STATE",   &m.cap_query_state,   state_status_detail(m)),
        cap_line("QUERY HISTORY", &m.cap_query_history, history_status_detail(m)),
    ];
    let block = Block::default()
        .borders(Borders::ALL)
        .title(" ⚡ Capabilities ")
        .border_style(Style::default().fg(Color::Cyan));
    frame.render_widget(Paragraph::new(lines).block(block), area);
}

fn cap_line(label: &str, state: &CapabilityState, detail: String) -> Line<'static> {
    let (icon, color) = match state {
        CapabilityState::Ready                  => ("✓", Color::Green),
        CapabilityState::Partial { .. }         => ("◐", Color::Yellow),
        CapabilityState::Unavailable { .. }     => ("✗", Color::Red),
    };
    Line::from(vec![
        Span::raw("  "),
        Span::styled(icon, Style::default().fg(color).add_modifier(Modifier::BOLD)),
        Span::raw(" "),
        Span::styled(format!("{:<14}", label), Style::default().fg(color).add_modifier(Modifier::BOLD)),
        Span::raw(" "),
        Span::styled(detail, Style::default().fg(Color::Gray)),
    ])
}
```

The detail-formatting helpers are tiny:
```rust
fn mine_status_detail(m: &Metrics) -> String {
    if m.mining_enabled {
        format!("ready · hashrate: {:.2} MH/s · blocks mined: {}", m.hashrate / 1e6, m.blocks_mined)
    } else { "ready (idle — start a miner to begin)".to_string() }
}

fn transact_status_detail(m: &Metrics) -> String {
    format!("ready · mempool: {} txs · peers: {}", /* mempool_size from existing field */, m.peer_count)
}

fn state_status_detail(m: &Metrics) -> String {
    if let Some(h) = m.proof_verified_height {
        format!("proof-verified at height {} · {} ms", h, m.proof_verify_duration_ms.unwrap_or(0.0))
    } else {
        format!("checkpoint-trusted · {} wallets indexed", /* indexed_wallet_count */ 0)
    }
}

fn history_status_detail(m: &Metrics) -> String {
    if m.archive_complete {
        "all blocks available".to_string()
    } else {
        let pct = (m.archive_lowest_indexed_height as f64 / m.archive_tip_height.max(1) as f64) * 100.0;
        let eta = m.archive_eta_seconds.map(format_eta).unwrap_or_else(|| "—".to_string());
        format!(
            "backfilling · {}/{} ({:.1}%) · {:.0} b/s · ETA {}",
            format_height(m.archive_lowest_indexed_height),
            format_height(m.archive_tip_height),
            100.0 - pct,
            m.archive_blocks_per_sec_recent,
            eta,
        )
    }
}
```

---

## Archive Sparkline (`crates/q-tui/src/ui/dashboard.rs`)

Reuse the existing sparkline pattern from `network.rs` (which renders bandwidth as `▁▂▃▄▅▆▇█`). The render is one ratatui line — a `Vec<Span>`, each `Span` one block character colored by intensity:

```rust
fn draw_archive_sparkline(frame: &mut Frame, area: Rect, m: &Metrics) {
    const BARS: &[&str] = &[" ", "▁", "▂", "▃", "▄", "▅", "▆", "▇", "█"];
    let max = m.archive_history_sparkline.iter().cloned().fold(1f32, f32::max);

    let spans: Vec<Span> = m.archive_history_sparkline.iter().map(|&v| {
        let normalized = (v / max).clamp(0.0, 1.0);
        let idx = (normalized * (BARS.len() - 1) as f32).round() as usize;
        let color = if normalized > 0.7 { Color::Green }
                    else if normalized > 0.3 { Color::Yellow }
                    else { Color::Red };
        Span::styled(BARS[idx], Style::default().fg(color))
    }).collect();

    let summary = format!(
        "  {:.0} b/s · ETA {}",
        m.archive_blocks_per_sec_recent,
        m.archive_eta_seconds.map(format_eta).unwrap_or_else(|| "—".into()),
    );
    let mut all = spans;
    all.push(Span::raw(summary));

    let block = Block::default()
        .borders(Borders::ALL)
        .title(" 📦 Archive backfill (60 min) ")
        .border_style(Style::default().fg(Color::Cyan));
    frame.render_widget(Paragraph::new(Line::from(all)).block(block), area);
}
```

The 60-element rolling window is updated once per minute by a background tick.

---

## API Endpoint — `GET /api/v1/status/readiness`

The same data the TUI consumes, also exposed via JSON for the web wallet, MCP server, and external monitors:

```json
{
  "readiness_mode": "FastReady",
  "readiness_changed_at": "2026-05-13T10:14:23.789Z",
  "proof": {
    "verified": true,
    "version": "nova-bn254-v1",
    "height": 11400000,
    "verified_at": "2026-05-13T10:14:23.789Z",
    "verify_duration_ms": 8.4
  },
  "capabilities": {
    "mine":          { "state": "Ready" },
    "transact":      { "state": "Ready" },
    "query_state":   { "state": "Ready" },
    "query_history": { "state": "Partial", "reason": "backfilling" }
  },
  "archive": {
    "lowest_indexed_height": 4321001,
    "tip_height":            11400000,
    "complete":              false,
    "progress_pct":          37.9,
    "blocks_per_sec_recent": 175,
    "eta_seconds":           64800
  }
}
```

Used by:
- Web wallet's pre-tx readiness check ("can I send right now? yes → enable button")
- MCP `verify_node_consistency` to confirm both nodes are FastReady before comparing roots
- External dashboards (Grafana, Prometheus exporters)

---

## Test Plan

1. **Unit:** `ReadinessMode::transition` table — every (from, to) pair documented + unit-tested
2. **Snapshot:** render the TUI in each of the 5 readiness modes to an ANSI text buffer, snapshot-test
3. **Integration:** start a Docker container with `--bootstrap-mode=proof`, watch the TUI go BOOTSTRAPPING → FAST-READY in real time, capture the 300ms flash window
4. **Integration:** confirm the capability matrix transitions: `query_history: Partial → Ready` when archive completes
5. **API:** `GET /api/v1/status/readiness` returns the expected JSON shape

---

## Effort Estimate

| Component | LOC | Effort |
|-----------|-----|--------|
| Metrics struct + enum extensions | ~80 | 2 hrs |
| Readiness banner renderer | ~120 | 4 hrs |
| Capability matrix renderer | ~150 | 4 hrs |
| Archive sparkline renderer | ~80 | 3 hrs |
| Backend wiring in `update_tui_metrics` | ~200 | 1 day |
| `/api/v1/status/readiness` endpoint | ~80 | 2 hrs |
| Tests (unit + snapshot + integration) | ~250 | 1.5 days |
| **Total** | **~960 LOC** | **~3 working days** |

Independent of any SNARK work — this lights up immediately based on existing readiness signals (peer count, sync state, mining_ready, etc.) and gets richer as the proof-verify path lands.

---

## Backward Compatibility

A node running v10.9.15 (without proof bootstrap) shipped today will display:
- `readiness_mode: CheckpointTrust` (yellow banner) if running on the live mainnet from snapshot
- `readiness_mode: GenesisSync` (cyan banner) if syncing from height 1
- Capabilities reported the same way — no proof yet, but mine/transact/state/history all work as currently defined

So this can ship in v10.9.16 *without* the SNARK work, providing immediate UX value, then enriches itself the moment the proof-bootstrap path is wired in.

— Server Beta, 2026-05-13
