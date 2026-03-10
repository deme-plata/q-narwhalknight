# Issue #004: Game Trainer Mode — Performance Cheat Engine

**State**: `closed`
**Priority**: HIGH
**Labels**: `starship-endgame`, `performance`, `trainer`
**Assigned**: Beta
**Branch**: `project/starship-endgame-revolution`
**Created**: 2026-03-08
**Updated**: 2026-03-10

---

## Description

Built-in performance optimizer that auto-applies every known trick to maximize node output — like a cheat engine for compute.

## Trainer Cheats

| Cheat | Key | Status | Real Implementation |
|-------|-----|--------|-------------------|
| INFINITE CORES | F1 | DONE | `core_affinity` pin + `SCHED_FIFO` real-time priority |
| GOD MODE MEMORY | F2 | DONE | `madvise(MADV_HUGEPAGE)` + `mlockall()` |
| SPEED HACK x100 | F3 | DONE | AVX-512/AVX2 SIMD detection + GPU flag |
| WALL HACK | F4 | DONE | Subscribe to compute-tunnel gossipsub |
| AIM BOT | F5 | DONE | Score-based task routing (capability * 1/latency * availability) |
| NO CLIP | F6 | DONE | `SCHED_FIFO` + `nice -20` + IRQ steering |
| INFINITE AMMO | F7 | DONE | Work queue prefetch + pipeline next before current completes |
| RAPID FIRE | F8 | DONE | Batch 8 mining solutions per gossipsub message |
| TELEPORT | F9 | DONE | `mmap()` zero-copy DB reads + `splice()` network paths |
| PRESTIGE MODE | F10 | DONE | CPU governor `performance` + disable C-states |
| NUKE | F11 | DONE | Enable all cheats simultaneously (`nuke_mode` flag + `activate_all()`) |
| TRAINER MENU | F12 | DONE | `get_trainer_menu()` formatted text UI with all 12 cheats + aggregate stats |

## Acceptance Criteria

- [x] `crates/q-compute/src/trainer.rs` — Trainer engine
- [x] Auto-detect hardware capabilities on startup
- [x] Apply safe defaults, allow user to enable aggressive mode
- [x] TUI overlay showing trainer status (`get_trainer_menu()` text-based menu)
- [x] Log performance gains vs baseline (`estimated_boost_pct()` only counts applied cheats, `calculate_display_boost()` shows theoretical)
- [x] `--trainer=full|safe|off` CLI flag

## Progress

- **2026-03-08**: F1, F2, F6, F10 implemented
- **2026-03-10**: F3, F4, F5, F7, F8, F9 implemented. 10/12 cheats working.
- **2026-03-10**: F11 NUKE implemented — `nuke_mode` flag, tracked in `active_cheats()`, `is_cheat_applied("F11")`. 11/12 cheats working.
- **2026-03-10**: F12 TRAINER MENU implemented — `get_trainer_menu()` returns formatted text UI. `toggle_cheat()` supports all F1-F12 by key or name. `status_summary()` for API consumers. 12/12 cheats done, 30 trainer tests passing. Issue CLOSED.

## Files

- `crates/q-compute/src/trainer.rs`
