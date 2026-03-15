# Issue #030: TUI Quantum Q Animation

**State**: `closed`
**Priority**: MEDIUM
**Labels**: `starship-endgame`, `tui`, `visual`, `branding`
**Assigned**: Gamma
**Branch**: `feature/safe-batched-sync-v1.0.2`
**Created**: 2026-03-10

---

## Description

The node and miner TUI should periodically transform into a beautiful animated "Q" shape — a quantum-inspired animation that reinforces the Quillon brand. Like a screensaver that activates during idle moments or on special events (new block mined, milestone reached, startup splash).

## Current State

- Node TUI exists in `crates/q-api-server/src/tui.rs`
- Miner TUI exists in `crates/q-miner/src/tui.rs`
- Both use ratatui for terminal rendering
- No branding animation or splash screen

## Design

The Q animation should:
1. **Startup splash** (2-3 seconds): ASCII art Q morphs from particles into solid letter
2. **Block mined event**: Brief Q pulse animation (0.5s) overlaid on TUI
3. **Idle screensaver** (after 60s no input): Rotating/pulsing Q with quantum wave effects
4. **Milestone events**: Special animation for height milestones (every 100K blocks)

### ASCII Q Frames (Example)

```
Frame 1 (particles):        Frame 3 (forming):         Frame 5 (solid):
    .  . .  .                   ██████                     ██████████
  .    . .    .               ██      ██                 ██          ██
 .   .     .   .             ██        ██               ██            ██
 .  .       .  .             ██        ██               ██            ██
  .  .     .  .              ██      ██                 ██          ██
    .  . .  .                  ██████                     ██████████
       . .                        ████                        ██████
         .                           ██                          ████
```

### Color Scheme

- Cyan (#00FFFF) base — matches Quillon branding
- Purple (#A78BFA) quantum glow
- Amber (#F59E0B) energy particles
- Green (#10B981) for "block found" pulse

## Acceptance Criteria

- [x] `QAnimation` struct in shared crate (`crates/q-tui/src/ui/q_animation.rs`) — 647 lines
- [x] 8-frame ASCII Q animation with smooth interpolation — Converge(14f) + Glow(16f) + Dissolve(10f)
- [x] Startup splash (2-3s) on node and miner boot — `QAnimation::new()` auto-triggers Converge
- [x] Block-found pulse — `App::notify_block_found()` calls `q_animation.trigger()`
- [x] Idle screensaver mode (configurable timeout, default 3 min) — 720 ticks auto-trigger
- [x] Color gradients using ratatui `Color::Rgb` (cyan -> purple -> amber hue cycling)
- [x] Disable with `--no-animation` flag or `Q_NO_ANIMATION=1` env — checked in `new()`
- [x] Works in 80x24 minimum terminal size — guards `w < 30 || h < 15`
- [x] Does NOT interfere with log output or TUI data display — overlay applied AFTER render

## Depends On

None — purely visual, no consensus/safety impact.

## Files

- `crates/q-tui/src/q_animation.rs` — Animation engine
- `crates/q-api-server/src/tui.rs` — Node TUI integration
- `crates/q-miner/src/tui.rs` — Miner TUI integration
