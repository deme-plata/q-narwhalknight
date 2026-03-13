# Issue #039: Water Robot Swarm — Animated Ocean Visualization

**State**: `open`
**Priority**: HIGH
**Labels**: `starship-endgame`, `tui`, `animation`, `water-robots`
**Assigned**: Open for AI collaboration
**Branch**: `feature/command-center-tui`
**Created**: 2026-03-13
**Updated**: 2026-03-13

## Description

Create an animated ASCII ocean where network nodes are represented as water robots (sea creatures) that swim, interact, and exchange data visually. This is the main eye-candy of the Command Center.

## Visual Specification

```
≋≋≋≋≋≋≋≋≋≋≋≋≋≋≋≋≋≋≋≋≋≋≋≋≋≋≋≋≋≋≋≋≋≋≋≋≋≋≋≋≋≋≋≋≋≋≋≋≋≋
≋≋  🐋 ←─····─→ 🐬  ≋≋≋≋≋≋  🦈 ←─····─→ 🐙  ≋≋≋≋
≋≋≋≋≋≋≋≋≋≋≋≋≋≋≋≋≋≋≋≋≋≋≋≋≋≋≋≋≋≋≋≋≋≋≋≋≋≋≋≋≋≋≋≋≋≋≋≋≋≋
≋≋≋≋ 🐠←·gossip·→🐟 ≋≋≋≋≋≋≋≋≋≋≋≋≋≋≋≋≋≋≋≋≋≋≋≋≋≋≋≋≋≋
≋≋≋≋≋≋≋≋≋≋≋≋≋≋≋≋≋≋≋≋≋≋≋≋≋≋≋≋≋≋≋≋≋≋≋≋≋≋≋≋≋≋≋≋≋≋≋≋≋≋
```

### Creature Types (ASCII, no emoji — terminal compatible)
```
Supernode (10Gbit):  ><((({°>    (whale — largest)
Full Node (1Gbit):   ><(({°>     (dolphin — medium)
Bootstrap (100Mbit): ><({°>      (shark — coordinator)
Light Node:          ><{°>       (fish — smallest)
Miner (you):         ><(((*>     (narwhal — with horn!)
```

### Animation Mechanics
- **Ocean waves**: 3-4 rows of `≋` `~` `∽` characters that scroll horizontally, offset each tick
- **Creature swimming**: Each creature has x,y position, moves slowly with sinusoidal vertical bobbing
- **Data packets**: When block/gossip events happen, particles (`·` `•` `*`) travel between creatures
- **Creature facing**: Flip sprite when moving left vs right
- **Speed**: Creatures move 1 char every 2-4 ticks (slow, graceful)

### Data-Driven Events
- New block received → particle burst from source creature
- Solution submitted → star particle from narwhal (you) to nearest node
- Peer connected → new creature swims in from edge
- Peer disconnected → creature fades out (gets smaller, then disappears)
- High gossip rate → more particles flowing between all creatures

## Implementation Notes

- Create `command_center_animation.rs` in `tui_views/`
- Each creature: `struct SwarmCreature { x: f32, y: f32, vx: f32, kind: CreatureKind, name: String }`
- Wave rendering: `WAVE_CHARS = ['≋', '~', '∽', '≈']`, phase-shifted per row + tick
- Particle system: `Vec<Particle>` with position, velocity, lifetime
- Draw to buffer in-place (same pattern as q_animation.rs)
- Seed-based deterministic placement from peer ID hash

## Acceptance Criteria

- [ ] ASCII wave background animation (scrolling ocean)
- [ ] 5 creature types with ASCII sprites (no emoji dependency)
- [ ] Creatures positioned by peer, move with bobbing motion
- [ ] Data flow particles between creatures on network events
- [ ] New peer = creature enters; disconnected = creature exits
- [ ] Narwhal (you) always visible and centered-ish
- [ ] Smooth 250ms tick rendering
- [ ] Graceful size adaptation (minimum 40x8 area)

## Parent Issue
- Part of #037 (Command Center TUI)
