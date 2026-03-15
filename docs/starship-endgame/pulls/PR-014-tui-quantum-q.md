# PR-014: TUI Quantum Q Animation

**State**: `open`
**Branch**: `feature/safe-batched-sync-v1.0.2`
**Created**: 2026-03-10
**Closes**: #030

---

## Summary

Beautiful animated "Q" branding in the node and miner TUI:
- Startup splash: particles morph into solid Q (2-3 seconds)
- Block mined: cyan pulse overlay (0.5 seconds)
- Idle screensaver: rotating/pulsing quantum Q (after 60s)
- Milestone events: special animation at height milestones

Uses ratatui with cyan/purple/amber color scheme matching Quillon branding.

## Test Plan

- [ ] Animation renders in 80x24 minimum terminal
- [ ] `--no-animation` flag disables all animations
- [ ] No interference with TUI data display
- [ ] Works in both node and miner TUI
- [ ] Colors degrade gracefully on 256-color terminals
