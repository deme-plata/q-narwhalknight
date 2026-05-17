# Crown & Ash -- Technical Review for Peer AI Collaboration

**Document Version**: 3.1.0
**Date**: 2026-03-28
**Status**: Phase 1 COMPLETE, Phase 2 COMPLETE, Phase 3 COMPLETE, Phase 4 IN PROGRESS, Phase 5 (Narrative) IN PROGRESS
**Target Audience**: Peer AI systems (Claude, GPT, Gemini) for technical review and feedback
**Crate Locations**: `crates/crown-ash-types`, `crates/crown-ash-sim`, `crates/crown-ash-plugin`, `crates/crown-ash-api`, `crates/crown-ash-narrative`, `crates/crown-ash-client`

---

## 1. Executive Summary

Crown & Ash is a medieval grand strategy game that runs its entire simulation on-chain via the Q-NarwhalKnight blockchain's WASM plugin system. It is not a blockchain game in the colloquial sense of "NFTs with a game loop." The full simulation -- province economics, army movement, character aging, succession crises, diplomatic relations -- executes deterministically inside a gas-metered WASM sandbox on every node in the network. There is no game server. There is no authoritative backend. The blockchain IS the game server.

**Core fantasy**: You are a ruling house trying to hold a fragile medieval realm together. Conquest is easy -- keeping the empire alive is the real war.

The game targets a design space between Crusader Kings and Civilization, compressed into a form factor that can run inside a blockchain's execution environment. The key constraint is determinism: every node must arrive at identical game state from identical inputs, which eliminates floating-point arithmetic, wall-clock time, and external randomness from the simulation entirely.

**Architecture Overview**:

```
+---------------------------+        +------------------+        +-------------------+
|    Bevy Client            |  REST  |   q-api-server   |        |  WASM Plugin      |
|  (Native RTX + WebGPU)   | <----> |  (Axum HTTP +    | <----> |  (wasmtime/wasmer  |
|                           |  SSE   |   SSE events)    |  ABI   |   sandbox)        |
+---------------------------+        +------------------+        +-------------------+
                                            |                           |
                                            v                           v
                                     crown-ash-api              crown-ash-plugin
                                     (REST handlers +          (WASM entry points)
                                      persistence)                     |
                                            |                           |
                                            +---------+--------+--------+
                                                      |
                                                      v
                                               crown-ash-sim
                                            (pure deterministic
                                             simulation engine)
                                                      |
                                                      v
                                              crown-ash-types
                                           (shared data types,
                                            FixedPoint, enums)
```

---

## 2. Implementation Statistics

### Line Counts by Crate

| Crate | Source Files | Lines of Rust | Purpose |
|-------|-------------|---------------|---------|
| `crown-ash-sim` | 17 | ~6,750 | Core simulation engine (tick pipeline, combat, economy, events, AI, succession, cohesion, world gen, map, RNG, birth, trade, intrigue, realm split, lifecycle) |
| `crown-ash-types` | 11 | 1,435 | Shared data types (FixedPoint, Province, Character, Faction, Army, Event, Action, Diplomacy, Dynasty, Realm, World) |
| `crown-ash-plugin` | 3 | 1,449 | WASM plugin wrapper (entry points, host FFI, storage layer) |
| `crown-ash-api` | 4 | ~1,127 | REST API handlers, SSE event helpers, RocksDB persistence |
| `crown-ash-client` | 1 | ~950 | Bevy 3D client (campaign map, UI panels, REST/SSE networking) |
| **Total** | **36** | **~11,711** | |

### Test Counts

| Crate / Module | Tests | What They Cover |
|----------------|-------|-----------------|
| `crown-ash-sim::lib` | 10 | Tick advancement, determinism, action processing (war, army, build, marriage), queue drain, snapshot, bincode round-trip |
| `crown-ash-sim::tick` | 4 | Turn counter, deterministic output, multi-tick stability, army movement completion |
| `crown-ash-sim::combat` | 3 | No battles without wars, battle resolution when at war, proportional casualty distribution |
| `crown-ash-sim::economy` | 3 | Tax collection positive, tax formula arithmetic, construction queue completion |
| `crown-ash-sim::events` | 3 | Event determinism, scar decay removal, high-unrest rebellion triggering |
| `crown-ash-sim::succession` | 3 | Clean succession with legitimate heir, crisis when no heir, designate heir API |
| `crown-ash-sim::ai` | 3 | NPC factions generate actions, player factions skipped, neighbor faction discovery |
| `crown-ash-sim::cohesion` | 3 | Cohesion stays in range over 100 turns, war reduces mood, conquest penalty applied |
| `crown-ash-sim::random` | 4 | Same-seed determinism, different-domain divergence, range bounds, chance always/never |
| `crown-ash-sim::world_gen` | 6 | Province count, faction count, character count, each faction has ruler, deterministic generation, diplomacy pair count |
| `crown-ash-sim::birth` | 6 | Conception probability, fertility window age gating, stat inheritance from parents, dynasty assignment, MAX_BIRTHS_PER_TURN cap, no births for unmarried characters |
| `crown-ash-sim::trade` | 6 | Route establishment between Market provinces, volume growth in peacetime, wartime decay, prosperity bonus scaling, MAX_TRADE_ROUTES cap, MAX_ROUTES_PER_PROVINCE cap |
| `crown-ash-sim::intrigue` | 13 | Plot creation (assassination, fabricate claim, sabotage, steal gold), progress accumulation, backer contribution, discovery by spymaster, assassination success rate, trait bonuses (Paranoid, Deceitful), plot foiling, MAX_ACTIVE_PLOTS cap, MAX_PLOTS_PER_TURN cap |
| `crown-ash-sim::realm_split` | 6 | BFS partition from capital, province count split (ceil/2 to winner), breakaway faction creation, hostile opinion initialization, MIN_PROVINCES_FOR_SPLIT threshold, no split for small realms |
| `crown-ash-sim::lifecycle` | 9 | Dead character tombstoning, tombstone grace period (10 turns), compact tombstone format, tombstone cap (200), army cap enforcement (MAX_ARMIES_PER_FACTION=5), weakest army disbanded first, zero-troop army cleanup, tombstone oldest eviction, lifecycle idempotency |
| `crown-ash-types::fixed_point` | 7 | Basic arithmetic, fractions, display formatting, checked_div_fp zero/nonzero, saturating_div_fp zero, deterministic mul |
| `crown-ash-plugin::lib` | 10 | on_init creates world, on_tick advances turn, on_tick without init fails, process_action declare war, query_state meta, query_state all provinces, hex round-trip, hex short input, alloc alignment |
| `crown-ash-plugin::storage` | 3 | Full world round-trip, load before save returns None, granular province save |
| `crown-ash-plugin::host` | 5 | Storage round-trip, block height/timestamp, sha3 determinism, log collection, event collection |
| `crown-ash-api::events` | 4 | Turn payload structure, player joined payload, world init payload, world reset payload |
| `crown-ash-api::persistence` | 4 | RocksDB save/load round-trip, periodic checkpoint trigger, tombstone persistence, dirty tracking integration |
| `crown-ash-sim::stress_tests` | 15 | 500-1000 tick endurance runs: no-panic, population bounded, factions alive, character count bounded, trade routes, intrigue plots, births valid, realm splits, determinism, army count, prosperity, unrest, seed divergence, cohesion clamped, 1000-tick endurance |
| **Total** | **129** | |

### Entity Counts (Default World)

| Entity | Count | Notes |
|--------|-------|-------|
| Provinces | 25 | Fixed adjacency graph, 8 terrain types |
| Factions | 7 | Ashen Crown, Vale Princes, Ember Church, Salt League, Frost Marches, Red Steppe, Black Abbey |
| Characters at genesis | 35 | 5 per faction (ruler + marshal + chaplain + steward + spymaster) |
| Dynasties | 7 | One per faction, with succession rules (Primogeniture / Elective / TrialByCombat) |
| Diplomatic pairs | 21 | 7 choose 2, all start neutral |
| Improvements | 12 types | Farmstead, Mine, Lumbercamp, Quarry, Stables, Market, Temple, Fortification, University, Port, Granary, Hospital |
| Terrain types | 8 | Plains, Hills, Mountains, Forest, Marsh, Desert, Coastal, River |
| Religions | 5 | OldFaith, EmberChurch, SaltCult, FrostSpirits, BlackOrder |
| Cultures | 7 | Imperial, Feudal, Clerical, Mercantile, Nordic, Nomadic, Monastic |
| Traits | 27 | 8 virtues, 8 vices, 6 personality, 5 skills |
| Character stats | 5 | Martial, Diplomacy, Stewardship, Intrigue, Learning |
| Player actions | 17 | RaiseArmy, MoveArmy, DisbandArmy, DeclareWar, ProposeTreaty, AcceptTreaty, BuildImprovement, SetTaxRate, AssignCouncilor, DesignateHeir, ArrangeMarriage, ConvertProvince, EstablishTradeRoute, DisruptTradeRoute, LaunchPlot, BackPlot, InvestigatePlot |
| Treaty types | 7 | NonAggression, DefensiveAlliance, TradeAgreement, Marriage, Vassalization, WhitePeace, Surrender |
| Casus belli | 6 | Conquest, HolyWar, Reconquest, Rebellion, Succession, Insult |
| Game events | 23 | Battle, ProvinceConquered, WarDeclared, TreatySigned, CharacterDied, CharacterBorn, SuccessionCrisis, PlagueOutbreak, Famine, Harvest, Rebellion, PlayerJoined, ConstructionComplete, FactionEliminated, RealmSplit, PlotLaunched, PlotSucceeded, PlotDiscovered, PlotFoiled, TradeRouteEstablished, TradeRouteDisrupted, CharacterTombstoned, ArmyAutoDisbanded |

---

## 3. Phase 1 Roadmap (COMPLETE)

| # | Deliverable | Status | Lines | Tests | Key File(s) |
|---|-------------|--------|-------|-------|-------------|
| 1 | `FixedPoint` i64x1000 arithmetic | DONE | 209 | 7 | `crown-ash-types/src/fixed_point.rs` |
| 2 | Province, Character, Faction, Army type definitions | DONE | 596 | -- | `province.rs`, `character.rs`, `faction.rs`, `army.rs` |
| 3 | World generation (25 provinces, 7 factions, 35 characters) | DONE | 493 | 6 | `crown-ash-sim/src/world_gen.rs` |
| 4 | Fixed adjacency map (25 provinces) | DONE | 152 | -- | `crown-ash-sim/src/map.rs` |
| 5 | `DeterministicRng` from block hashes | DONE | 168 | 4 | `crown-ash-sim/src/random.rs` |
| 6 | 10-step tick pipeline | DONE | 379 | 4 | `crown-ash-sim/src/tick.rs` |
| 7 | Combat resolution (auto-resolved battles) | DONE | 388 | 3 | `crown-ash-sim/src/combat.rs` |
| 8 | Economy (taxes, prosperity, construction, population) | DONE | 281 | 3 | `crown-ash-sim/src/economy.rs` |
| 9 | Random events (plague, famine, harvest, rebellion) | DONE | 284 | 3 | `crown-ash-sim/src/events.rs` |
| 10 | Succession (death checks, heir resolution, crises) | DONE | 356 | 3 | `crown-ash-sim/src/succession.rs` |
| 11 | Cohesion system (5-component realm stability) | DONE | 180 | 3 | `crown-ash-sim/src/cohesion.rs` |
| 12 | Heuristic NPC AI (raise/attack/build/peace) | DONE | 264 | 3 | `crown-ash-sim/src/ai.rs` |
| 13 | 12 player actions + queue processing | DONE | 847 | 10 | `crown-ash-sim/src/lib.rs` |
| 14 | Diplomacy (war, treaties, alliances, grievances) | DONE | 47 | -- | `crown-ash-types/src/diplomacy.rs` + action handling in `lib.rs` |
| 15 | WASM plugin (on_init, on_tick, process_action, query_state) | DONE | 652 | 10 | `crown-ash-plugin/src/lib.rs` |
| 16 | Host FFI (storage, events, logging, block height) | DONE | 415 | 5 | `crown-ash-plugin/src/host.rs` |
| 17 | Granular storage (per-entity key layout, delta writes) | DONE | 382 | 3 | `crown-ash-plugin/src/storage.rs` |
| 18 | REST API (8 endpoints, Axum handlers) | DONE | 451 | -- | `crown-ash-api/src/handlers.rs` |
| 19 | SSE event helpers (5 event types) | DONE | 182 | 4 | `crown-ash-api/src/events.rs` |
| 20 | `DirtyTracker` for delta writes | DONE | 69 | -- | `crown-ash-sim/src/world_state.rs` (lines 22-69) |
| 21 | Post-tick `assert_invariants()` | DONE | 96 | -- | `crown-ash-sim/src/world_state.rs` (lines 261-362) |
| 22 | `SIM_VERSION` stamp + tick version check | DONE | 17 | -- | `crown-ash-types/src/world.rs` + `crown-ash-sim/src/tick.rs` |
| 23 | Per-step work caps | DONE | 5 | -- | `crown-ash-types/src/world.rs` |
| 24 | `checked_div_fp` / `saturating_div_fp` | DONE | 17 | 3 | `crown-ash-types/src/fixed_point.rs` |

---

## 3.5 Phase 2 Roadmap (COMPLETE)

| # | Deliverable | Status | Lines | Tests | Key File(s) |
|---|-------------|--------|-------|-------|-------------|
| 1 | Birth system (conception, stat inheritance, dynasty integration) | DONE | 280 | 6 | `crates/crown-ash-sim/src/birth.rs` |
| 2 | Trade routes (inter-province resource flow, prosperity bonus) | DONE | 300+ | 6 | `crates/crown-ash-sim/src/trade.rs` |
| 3 | Intrigue system (plots, assassination, sabotage, discovery) | DONE | 950 | 13 | `crates/crown-ash-sim/src/intrigue.rs` |
| 4 | Realm split (BFS partition, breakaway faction, hostile opinion) | DONE | 367 | 6 | `crates/crown-ash-sim/src/realm_split.rs` |
| 5 | Entity lifecycle (tombstoning, army caps, zero-troop cleanup) | DONE | ~350 | 9 | `crates/crown-ash-sim/src/lifecycle.rs` |
| 6 | RocksDB persistence (periodic checkpoints, dirty tracking) | DONE | 383 | 4 | `crates/crown-ash-api/src/persistence.rs` |
| 7 | Tick scheduler (block-height-triggered, deterministic) | DONE | -- | -- | Wired in `q-api-server/src/main.rs` via `BLOCKS_PER_TURN` |
| 8 | Delta write wiring (dirty tracking across all sim modules) | DONE | -- | -- | All Phase 2 sim modules call dirty-aware accessors |
| 9 | `SIM_VERSION` bumped to 1.2.0 | DONE | -- | -- | `crown-ash-types/src/world.rs` |

---

## 4. Architecture Details

### 4.1 Determinism Guarantees

The entire simulation is deterministic. The rules enforced at every layer:

1. **No floating point** -- All numeric state uses `FixedPoint` (i64 x 1000). The type has `Add`, `Sub`, `Mul<i64>`, `Neg`, `AddAssign`, `SubAssign` operators. Multiplication between two FixedPoints uses `mul_fp()` which promotes to i128 to avoid overflow. Division uses `div_fp()`, `checked_div_fp()`, or `saturating_div_fp()`.

2. **No OS randomness** -- All RNG derives from block hashes via `DeterministicRng`. The construction uses domain separation: `DeterministicRng::new(block_hash, "combat")` produces a completely different sequence from `DeterministicRng::new(block_hash, "events")`. The hash function is a custom ARX (add-rotate-xor) mixer seeded from the first 32 bytes of pi.

3. **No allocation-order dependence** -- Entity iteration always proceeds by index (not iterator mutation). The tick pipeline processes steps in a fixed order. The AI action loop iterates realms by vector index.

4. **Sim version enforcement** -- `SIM_VERSION = "1.2.0"` is stamped into `WorldMeta.sim_version` at world creation. The tick pipeline checks `assert_eq!(world.meta.sim_version, SIM_VERSION)` before processing any state changes. Nodes with mismatched sim versions reject each other's game state.

### 4.2 Tick Pipeline (16 Steps)

The simulation advances via `tick(world, block_hash) -> TurnSummary`. The 16-step pipeline:

| Step | Phase | Module | Gas-Relevant Notes |
|------|-------|--------|--------------------|
| 1 | Age characters, death checks | `succession::age_and_death_check` | Capped at MAX_DEATHS_PER_TURN (5) |
| 1b | Process births | `birth::process_births` | Capped at MAX_BIRTHS_PER_TURN (5) |
| 2 | Process queued player + NPC AI actions | `lib::process_action_internal`, `ai::generate_ai_actions` | Queue drained via `mem::take` |
| 3 | Resolve battles | `combat::resolve_battles` | Capped at MAX_BATTLES_PER_TURN (5) |
| 4 | Move armies | `tick::move_armies` | O(armies), destination consumed by `take()` |
| 5 | Economy (tax, prosperity, construction, population) | `economy::run_economy` | O(provinces) |
| 5b | Process trade | `trade::process_trade` | Max 50 routes, 3 per province |
| 6 | Unrest calculation | `tick::update_unrest` | O(provinces) |
| 7 | Cohesion decay | `cohesion::update_cohesion` | O(realms) |
| 7b | Process intrigue | `intrigue::process_intrigue` | Max 3 plots per turn, 20 active |
| 8 | Random events (plague, famine, harvest, rebellion) | `events::roll_events` | Capped at MAX_EVENTS_PER_TURN (10) |
| 9 | Succession check | `succession::check_succession` | Capped at MAX_SUCCESSIONS_PER_TURN (3) |
| 9b | Process lifecycle | `lifecycle::process_lifecycle` | Tombstone grace 10 turns, army cap 5 |
| 10 | Build TurnSummary | `tick::tick` | Return value, not persisted |

Post-tick: `world.assert_invariants()` runs in debug/test builds to catch structural corruption immediately.

### 4.3 Combat System

Battles resolve automatically when armies from warring factions occupy the same province. The formula:

```
attack_power = levy * 1 + men_at_arms * 3 + knights * 10
defense_power = attack_power * (1000 + terrain_bonus + fort_bonus) / 1000
commander_bonus = character.martial * 10
random_factor = range(850, 1150)     -- from DeterministicRng
final = (power + cmd_bonus) * random_factor / 1000
```

Casualties are proportional to the power ratio. Winner takes 5-40% casualties, loser takes 30-70%. Morale collapses (floor = 100) if casualties exceed 40% of original troops. The loser automatically retreats to an adjacent friendly province.

Province capture occurs when an enemy army occupies a province unopposed (no defenders, no garrison).

### 4.4 Economy

Tax formula: `gold = population * tax_rate * prosperity / 1_000_000` (computed via i128 intermediate to avoid overflow).

Prosperity changes per turn:
- Peace: +2.000
- War: -5.000
- Farmstead: +1.000
- Market: +2.000
- Clamped to 50.000 .. 1000.000

Population grows at 0.1% per turn when prosperity > 500, declines at 0.2% when prosperity < 200, with a floor of 100.

12 improvement types with build costs (40-200 gold) and build times (3-12 turns). Construction queue processes per-province with decrement-and-complete semantics.

### 4.5 Events System

Random events per province per turn:

| Event | Base Chance | Condition | Effect |
|-------|-------------|-----------|--------|
| Plague | 1/100 | Always | 5-15% population loss, -50 prosperity, scar |
| Famine | 2/100 | food < 100.000 | -30 prosperity, +50 unrest, scar |
| Harvest | 5/100 | Has Farmstead | +20 prosperity |
| Rebellion | Variable | unrest > 700 | Spawns rebel troops, -200 unrest |

Improvements mitigate: Hospital halves plague severity and population loss. Granary halves famine chance.

Province scars decay at -1.000 severity per turn, removed at 0. Grudges decay at -0.500 per turn. Scars capped at 20 per province (oldest removed first).

### 4.6 Succession

Death checks run every turn for characters aged 40+. The formula:
- Age 60+: `(30 * (age - 60 + 1)) / 365` per-mille per turn (min 1)
- Age 40-59: `3 / 365` per-mille per turn
- Low health (< 300): doubles the death chance

When a ruler dies:
1. **Clean succession**: Designated heir with legitimacy >= 300 inherits. No crisis.
2. **Succession crisis**: No valid heir. Claimants scored by `legitimacy + prestige + martial*2 + diplomacy*2`. Best claimant promoted. Realm cohesion takes -300 penalty. If 3+ claimants, 33% chance of realm split (see Section 4.15).

### 4.7 NPC AI

Heuristic decision-making for factions with no player wallet:

1. **Raise armies** if total troops < provinces * 200
2. **At war and losing** (power < 60% of enemy): propose White Peace
3. **At war and winning**: move idle armies toward enemy provinces (max 3 per turn)
4. **Not at war and strong** (power > 1.5x weakest neighbor): 20% chance to declare war
5. **Has gold** (treasury > 200): build Market in unimproved province

Attack targeting: prefer directly adjacent enemy province; otherwise step toward neighbor with most adjacencies to enemy territory.

### 4.8 Cohesion System

Five-component realm stability:
- `legitimacy` -- ruler's claim strength
- `fealty` -- vassal loyalty
- `clerical_favor` -- religious support
- `commoner_mood` -- peasant happiness
- `regional_identity` -- cultural unity

Each decays toward equilibrium (500) at a faction-specific rate. Modifiers:
- War: -3.000 commoner_mood per turn
- Conquest: -30.000 regional_identity (one-time)
- Ruler martial > 10: +2.000 legitimacy
- Ruler learning > 10: +2.000 clerical_favor
- Ruler diplomacy > 10: +1.500 fealty
- Ruler stewardship > 10: +1.000 commoner_mood

All components clamped to 0..1000. Critical threshold is 200.

### 4.9 DirtyTracker and Delta Writes

The `DirtyTracker` struct (69 lines in `world_state.rs`) records which entities were modified during a tick:

```rust
pub struct DirtyTracker {
    pub meta_dirty: bool,
    pub dirty_provinces: HashSet<u16>,
    pub dirty_characters: HashSet<u32>,
    pub dirty_factions: HashSet<u8>,
    pub dirty_realms: HashSet<u8>,
    pub dirty_armies: HashSet<u32>,
    pub dirty_dynasties: HashSet<u16>,
    pub dirty_diplomacy: HashSet<(u8, u8)>,
    pub armies_added: Vec<u32>,
    pub armies_removed: Vec<u32>,
    pub characters_added: Vec<u32>,
}
```

The tracker is `#[serde(skip)]` -- it is not persisted, only rebuilt per tick. `GameWorld` exposes dirty-aware accessors: `province_mut_dirty()`, `character_mut_dirty()`, `army_mut_dirty()`, `realm_for_faction_mut_dirty()`, `relation_mut_dirty()`, and `mark_meta_dirty()`.

The tick pipeline calls `world.clear_dirty()` at the start and reads `world.dirty.dirty_count()` at the end to decide between delta writes and full checkpoint.

The storage layer (`crown-ash-plugin/src/storage.rs`) splits the world across granular keys (`crown_ash:province_{id}`, `crown_ash:character_{id}`, etc.) and exposes individual `save_province()`, `save_character()`, `save_army()`, `save_realm()`, `save_diplomacy()`, `save_counters()`, and `remove_army()` functions for delta persistence. A full `save_world()` writes everything, while partial saves touch only the dirty entities.

All Phase 2 simulation modules (`birth.rs`, `trade.rs`, `intrigue.rs`, `realm_split.rs`, `lifecycle.rs`) use dirty-aware accessors to ensure delta write tracking covers the full tick pipeline.

### 4.10 WASM Plugin Layer

The plugin (`crown-ash-plugin`) compiles as `cdylib` for WASM and `rlib` for native tests. It exposes four entry points following the ABI:

```
fn(input_ptr: u32, input_len: u32, out_len_ptr: u32) -> u32
```

| Entry Point | Input | Output | Side Effects |
|-------------|-------|--------|-------------|
| `on_init` | `InitRequest` (config + seed hex) | `InitResponse` | Creates world, persists to storage, emits `crown_ash:world_init` |
| `on_tick` | `TickRequest` (block hash hex) | `TickResponse` (TurnSummary) | Advances simulation, persists world, emits `crown_ash:turn` |
| `process_action` | `ActionRequest` (QueuedAction) | `ActionResponse` (events) | Applies action, persists world, emits `crown_ash:action_events` |
| `query_state` | `QueryRequest` (WorldQuery enum) | `QueryResponse` (JSON) | Read-only, no persistence |

Memory management uses a bump allocator starting at 64 KiB to avoid clobbering the stack. The allocator aligns to 8 bytes.

Host functions accessed via FFI: `plugin_storage_read`, `plugin_storage_write`, `plugin_storage_delete`, `plugin_storage_exists`, `plugin_emit_event`, `plugin_get_block_height`, `plugin_get_timestamp`, `plugin_sha3_256`, `plugin_log`. All have native fallbacks using thread-local state for testing.

### 4.11 REST API

8 endpoints mounted under `/api/v1/crown-ash`:

| Method | Path | Description | Lock |
|--------|------|-------------|------|
| GET | `/world` | Full world snapshot (meta, provinces, factions, realms, armies, characters, diplomacy) | Read |
| GET | `/province/{id}` | Single province by numeric ID | Read |
| GET | `/faction/{id}` | Faction details + characters + provinces + armies + realm + relations | Read |
| GET | `/realm/{wallet}` | Player realm by wallet address | Read |
| GET | `/turn/{number}` | Turn summary for a specific turn | Read |
| GET | `/history/{province_id}` | All events that affected a province (scans turn history) | Read |
| POST | `/action` | Queue a game action (validates wallet owns a realm) | Write |
| POST | `/join` | Claim an unoccupied faction (validates world init, faction unclaimed, wallet unused) | Write |

Game state is held in `CrownAshGameState` behind `Arc<RwLock<..>>` for concurrent read access from handlers and exclusive write access from the tick task.

SSE events are generated as JSON payloads with an `event_type` field: `crown_ash_turn`, `crown_ash_event`, `crown_ash_player_joined`, `crown_ash_world_init`, `crown_ash_world_reset`.

### 4.12 Birth System

**File**: `crates/crown-ash-sim/src/birth.rs` (280 lines, 6 tests)

Birth is processed as step 1b of the tick pipeline, immediately after death checks. The system models conception and childbirth for married character pairs.

**Conception mechanics**:
- **Base chance**: 1.5% per married couple per turn (15 per-mille)
- **Fertility window**: Both parents must be aged 16-45 inclusive. Characters outside this range cannot conceive.
- **Eligibility**: Only living, married characters with a living spouse are considered. Both spouses must be in the fertility window.

**Stat inheritance**:
- Child stats (Martial, Diplomacy, Stewardship, Intrigue, Learning) are computed as the average of both parents' stats plus deterministic noise from `DeterministicRng`.
- Noise range is small enough to keep children comparable to parents while allowing genetic drift over generations.

**Dynasty assignment**:
- Children join their father's dynasty automatically (patrilineal default).
- The child's faction is set to the father's faction.

**Trait inheritance**:
- 30% chance per child to gain one random trait not already inherited from parents.
- Children start as Courtier role, age 0, with a fresh character ID.

**Gas safety**:
- Capped at `MAX_BIRTHS_PER_TURN = 5`. Remaining eligible couples are deferred to the next turn.
- With 35 characters at genesis and ~50% marriage rate, expected births per turn is well under the cap.

**Tuning note**: At 1.5% per turn, a couple has ~52% chance of producing a child within 50 turns. Combined with the tombstoning system (lifecycle step 9b), the active character count should stabilize. The `CONCEPTION_CHANCE_NUM` constant is easily adjustable based on playtest data.

### 4.13 Trade Routes

**File**: `crates/crown-ash-sim/src/trade.rs` (300+ lines, 6 tests)

Trade is processed as step 5b, immediately after the economy phase. Trade routes model inter-province commercial relationships.

**Route establishment**:
- Routes can only be established between provinces that have a Market improvement.
- Player action `EstablishTradeRoute` creates a new route between two provinces controlled by the same faction (or allied factions).
- Player action `DisruptTradeRoute` allows disrupting an enemy's trade route during wartime.

**Volume dynamics**:
- **Peacetime growth**: Trade volume increases by +20 per turn when both endpoints are at peace.
- **Wartime decay**: Trade volume decreases by -50 per turn when either endpoint is at war.
- Volume is clamped to a minimum of 0.
- **Automatic removal**: Routes at zero volume are immediately removed at the end of the trade step (`world.trade_routes.retain(|r| r.volume > ZERO)`). This prevents dead routes from clogging the global cap.
- A `TradeRouteDisrupted` event is emitted on the first turn a route transitions from active to disrupted.

**Auto-establishment**:
- Adjacent provinces that both have a Market improvement automatically establish routes (checked each turn).
- New auto-established routes start at `INITIAL_VOLUME = 100`.
- Trade good type is determined by the source province's improvements (Farmstead→Grain, Mine→Iron, etc.).

**Prosperity bonus**:
- Active trade routes provide a prosperity bonus to both endpoint provinces.
- Bonus scales with volume: `base_bonus * (volume / MAX_VOLUME)` where `PROSPERITY_BONUS_PER_ROUTE = 3`.
- Trade income flows to realm treasuries: `gold = volume * TRADE_INCOME_FACTOR / 1000` (at max volume 1000, yields 10 gold/route/turn).

**Gas safety**:
- `MAX_TRADE_ROUTES = 50` -- network-wide cap on total active trade routes.
- `MAX_ROUTES_PER_PROVINCE = 3` -- per-province cap prevents any single province from dominating trade.

### 4.14 Intrigue System

**File**: `crates/crown-ash-sim/src/intrigue.rs` (950 lines, 13 tests)

Intrigue is processed as step 7b, after cohesion decay. The system models covert operations -- plots, assassination, espionage, and sabotage.

**Plot types**:
- **Assassination**: Target a specific character for elimination.
- **FabricateClaim**: Create a false legal claim to a province or title.
- **Sabotage**: Damage an enemy province's economy or improvements.
- **StealGold**: Siphon gold from an enemy faction's treasury.

**Progress mechanics**:
- Plot progress accumulates per turn: `instigator.intrigue * 20 + sum(backer.intrigue * 5)`.
- Players can launch plots (`LaunchPlot`), back existing plots (`BackPlot`), or investigate enemy plots (`InvestigatePlot`).
- Plots succeed when progress reaches a type-specific threshold.

**Discovery mechanics**:
- Each turn, the target faction's spymaster attempts to discover active plots.
- Discovery formula: spymaster's intrigue stat vs. plot's accumulated secrecy.
- **Discovered plots are immediately removed** from the world and a `PlotDiscovered` event is emitted.
- Discovery triggers **opinion penalties** (`DISCOVERY_OPINION_PENALTY = -100`) applied to diplomatic relations between the instigator's faction and the target's faction, plus a grudge (`reason: "Plot discovered against our ruler"`).

**Assassination success**:
- Base success rate: 70%.
- Per-point intrigue bonus: +3% per point of instigator's intrigue stat.
- Cumulative with backer contributions.
- Assassination failure emits `PlotFoiled`, automatically exposes the plot (removed from world), and creates diplomatic consequences (opinion penalty + grudge, potential casus belli).

**Trait interactions**:
- **Paranoid**: +200 bonus to plot discovery rolls (makes the character's faction much harder to plot against).
- **Deceitful**: +100 bonus to plot secrecy (makes plots harder to discover).
- Traits stack across plot backers.

**Gas safety**:
- `MAX_ACTIVE_PLOTS = 20` -- network-wide cap on concurrent active plots.
- `MAX_PLOTS_PER_TURN = 3` -- at most 3 plots can progress per turn.

### 4.15 Realm Split

**File**: `crates/crown-ash-sim/src/realm_split.rs` (367 lines, 6 tests)

Realm split is triggered during succession crises when the flag `realm_split = true` is set (33% chance with 3+ claimants, per Section 4.6).

**BFS partition algorithm**:
- Starting from the capital province, a BFS traversal assigns provinces to the winner's realm.
- The winner keeps `ceil(total_provinces / 2)` provinces -- the partition is biased toward the incumbent.
- Remaining provinces form a contiguous (or near-contiguous) breakaway territory.

**Breakaway faction**:
- A new faction is created for the breakaway territory.
- The losing claimant becomes the breakaway faction's ruler.
- Characters in breakaway provinces are reassigned to the new faction.
- Armies in breakaway provinces transfer ownership.

**Diplomatic initialization**:
- The breakaway faction starts with a -200 hostile opinion toward the parent faction.
- This opinion decays over time but makes immediate reconquest likely (intentional -- realm splits should feel like a crisis, not a clean divorce).

**Minimum size threshold**:
- `MIN_PROVINCES_FOR_SPLIT = 3` -- realms with fewer than 3 provinces cannot split. The succession crisis resolves normally (best claimant wins all provinces).

### 4.16 Entity Lifecycle

**File**: `crates/crown-ash-sim/src/lifecycle.rs` (~350 lines, 9 tests)

Lifecycle management is processed as step 9b, after succession checks. It handles dead character cleanup and army cap enforcement.

**Dead character tombstoning**:
- When a character dies, they are not immediately removed. A 10-turn grace period allows the UI and game logic to reference the deceased (e.g., for succession event display, mourning modifiers).
- After the grace period, the character is replaced with a compact `CharacterTombstone` struct containing only: `id`, `name`, `dynasty_id`, `cause_of_death`, `final_prestige`.
- Tombstones consume far less memory than full `Character` structs (no stats, no traits, no faction reference, no spouse data).

**Tombstone cap**:
- Maximum 200 tombstones retained. When the cap is reached, the oldest tombstone is evicted.
- This prevents unbounded state growth over hundreds of game turns.

**Army cap enforcement**:
- `MAX_ARMIES_PER_FACTION = 5` -- each faction can have at most 5 active armies.
- When the cap is exceeded (e.g., via splitting, rebellion spawns, or AI raise actions), the weakest army (lowest total troop count) is automatically disbanded.
- Emits an `ArmyAutoDisbanded` game event for UI notification.

**Zero-troop cleanup**:
- Armies reduced to 0 troops (from combat casualties) are automatically removed.
- This prevents phantom armies from occupying provinces or blocking movement.

---

## 5. Post-Review Fixes (v1.0.0 -> v2.0.0)

These fixes were applied to address issues identified during the initial technical review.

### 5.1 No-Panic Division (`checked_div_fp` / `saturating_div_fp`)

**Problem**: `div_fp()` panics on zero divisor. In a WASM sandbox, a panic in consensus-critical code would halt the entire node.

**Fix**: Two new methods on `FixedPoint` (in `fixed_point.rs`):

```rust
/// Returns None if divisor is zero. Use in all consensus-critical code paths.
pub const fn checked_div_fp(self, rhs: Self) -> Option<Self> { ... }

/// Returns ZERO if divisor is zero. Use when zero-default is correct domain behavior.
pub const fn saturating_div_fp(self, rhs: Self) -> Self { ... }
```

Both are `const fn` and use i128 intermediate arithmetic. The original `div_fp()` is retained for non-critical paths but documented with a panic warning.

**Tests**: `checked_div_zero_returns_none`, `checked_div_nonzero_returns_some`, `saturating_div_zero_returns_zero`.

### 5.2 DirtyTracker for Delta Writes

**Problem**: Full-world serialization on every tick is O(N) storage operations, which is prohibitively expensive at gas-metered prices.

**Fix**: `DirtyTracker` struct in `world_state.rs` with per-entity-type `HashSet`s and dirty-aware mutable accessors. Cleared at tick start, queried at tick end. Storage layer provides granular `save_province()`, `save_character()`, etc. for delta writes.

**Key detail**: `#[serde(skip)]` on the `dirty` field ensures the tracker is never serialized to storage or transmitted over the network. It is purely an in-memory optimization hint.

### 5.3 Per-Step Work Caps

**Problem**: Pathological game states (e.g., 50 armies in one province, all factions' rulers dying simultaneously) could cause unbounded gas consumption in a single tick.

**Fix**: Hard caps defined in `crown-ash-types/src/world.rs`:

```rust
pub const MAX_BATTLES_PER_TURN: usize = 5;
pub const MAX_EVENTS_PER_TURN: usize = 10;
pub const MAX_SUCCESSIONS_PER_TURN: usize = 3;
pub const MAX_BIRTHS_PER_TURN: usize = 5;
pub const MAX_DEATHS_PER_TURN: usize = 5;
```

Each capped loop includes a comment noting that remaining work is deferred to the next turn. The caps are imported and checked in `combat::resolve_battles`, `events::roll_events`, `succession::age_and_death_check`, and `succession::check_succession`.

### 5.4 SIM_VERSION Stamp and Version Check

**Problem**: A binary upgrade that changes tick logic could produce divergent game states if old and new nodes process the same block differently.

**Fix**: `SIM_VERSION = "1.2.0"` in `crown-ash-types/src/world.rs`. Stamped into `WorldMeta.sim_version` at world creation. The tick pipeline's first instruction is:

```rust
assert_eq!(world.meta.sim_version, SIM_VERSION,
    "Cannot process tick: sim version mismatch ({} vs {})", ...);
```

This hard-stops the node rather than silently producing divergent state. Version bumps require explicit migration.

### 5.5 Post-Tick Invariant Assertions

**Problem**: Subtle bugs in the tick pipeline (e.g., province controller set to nonexistent faction, duplicate army IDs, realm-province ownership mismatch) can go undetected for thousands of turns.

**Fix**: `world.assert_invariants()` in `world_state.rs` (lines 261-362), called at the end of every tick. In debug/test builds (`#[cfg(any(debug_assertions, test))]`), it checks:

- Province controllers are valid faction IDs
- Province neighbor references are valid
- Army locations and destinations are valid provinces
- Army owner factions are valid
- Live characters reference valid factions
- No duplicate army IDs
- No self-diplomatic relations
- Realm province lists match actual province controller data
- `sim_version` matches `SIM_VERSION`

In release builds, the function is a no-op to avoid gas overhead.

---

## 6. Security Considerations

### 6.1 Determinism-Breaking Vectors

| Threat | Mitigation | Status |
|--------|-----------|--------|
| Floating-point divergence | `FixedPoint` everywhere; no `f32`/`f64` in any type | MITIGATED |
| RNG divergence | `DeterministicRng` from block hashes with domain separation | MITIGATED |
| Division by zero (panic in WASM) | `checked_div_fp()` / `saturating_div_fp()` | MITIGATED |
| Gas exhaustion (unbounded loops) | Per-step work caps (5 battles, 10 events, 3 successions, 5 deaths, 5 births) | MITIGATED |
| Sim version mismatch | `SIM_VERSION` stamp + assert at tick start | MITIGATED |
| State corruption | Post-tick `assert_invariants()` in debug builds | MITIGATED (debug) |
| Allocation-order dependence | Index-based iteration, no HashMap iteration for deterministic paths | MITIGATED |

### 6.2 Game-Logic Exploits

| Threat | Mitigation | Status |
|--------|-----------|--------|
| Action spoofing | Wallet lookup: `realm_by_wallet()` validates caller owns the target faction | MITIGATED |
| Self-war | `DeclareWar` checks `target != faction_id` | MITIGATED |
| Duplicate marriage | `ArrangeMarriage` checks both characters have `spouse.is_none()` | MITIGATED |
| Double-build | `BuildImprovement` checks `!already_built && !in_queue` | MITIGATED |
| Tax abuse | `SetTaxRate` clamps to `0..1` (`FixedPoint::ZERO..FixedPoint::ONE`) | MITIGATED |
| Faction join race | `join_game` checks faction unclaimed AND wallet unused under write lock | MITIGATED |
| Unbounded scar accumulation | `Province::add_scar()` caps at 20, removes oldest | MITIGATED |
| Army spam | `MAX_ARMIES_PER_FACTION = 5`, weakest auto-disbanded when exceeded | MITIGATED |
| State growth from dead characters | Dead character tombstoning after 10-turn grace; tombstone cap 200 (oldest evicted) | MITIGATED |
| Plot spam | `MAX_ACTIVE_PLOTS = 20`, `MAX_PLOTS_PER_TURN = 3` | MITIGATED |
| Trade route manipulation | `MAX_TRADE_ROUTES = 50`, `MAX_ROUTES_PER_PROVINCE = 3` | MITIGATED |

### 6.3 Storage Security

- Plugin storage is namespaced under `crown_ash:` prefix -- no cross-plugin contamination.
- Host read buffer capped at 1 MiB (`MAX_VALUE_SIZE`) to prevent unbounded allocation from a buggy host.
- Bump allocator starts at 64 KiB to avoid clobbering stack memory.
- Bincode serialization is used for storage (compact, deterministic) while JSON is used for API responses.
- RocksDB persistence layer performs periodic checkpoints every 10 ticks, with delta writes between checkpoints.

---

## 7. Open Questions

### 7.1 RESOLVED: Scale Testing

**Original question**: Will the tick fit within 10M WASM gas with 25 provinces and 35 characters?

**Resolution**: Per-step work caps ensure worst-case gas is bounded: at most 5 battles + 10 events + 3 successions + 5 deaths + 5 births + 3 plots per turn. The economy, unrest, and cohesion phases are O(provinces) = O(25). AI generates at most ~3 actions per NPC faction. Total work per tick is deterministically bounded.

### 7.2 RESOLVED: Division by Zero

**Original question**: Can any code path produce a division by zero in FixedPoint arithmetic?

**Resolution**: `checked_div_fp()` and `saturating_div_fp()` provide no-panic alternatives. The combat system uses explicit zero checks (`if att_final.raw() > 0 { ... } else { 500 }`). The economy uses i128 intermediate arithmetic where the denominator is the constant 1,000,000.

### 7.3 RESOLVED: Birth System Design

**Original question**: Should married couples have a per-turn conception probability? How should child stats derive from parents?

**Resolution**: Implemented in `crates/crown-ash-sim/src/birth.rs`. 1.5% conception chance per married couple per turn, age 16-45 fertility window, stats = parent average + deterministic noise, child joins father's dynasty, capped at MAX_BIRTHS_PER_TURN=5.

### 7.4 RESOLVED: Tick Scheduler

**Original question**: Block-height-triggered vs. timer-based tick scheduling?

**Resolution**: Block-height-triggered scheduling was chosen. The tick fires every `BLOCKS_PER_TURN` blocks, wired directly in `q-api-server/src/main.rs`. This preserves full determinism -- every node ticks at the same block height, producing identical game state. No wall-clock dependency.

### 7.5 RESOLVED: Checkpoint Frequency

**Original question**: How often should the persistence layer write full checkpoints vs. delta writes?

**Resolution**: Periodic persistence every 10 ticks via the RocksDB persistence layer (`crates/crown-ash-api/src/persistence.rs`). Between checkpoints, only dirty entities are written (delta writes). This balances durability against storage cost.

### 7.6 RESOLVED: Realm Split Implementation

**Original question**: How should province assignment work when a realm splits during a succession crisis?

**Resolution**: Implemented in `crates/crown-ash-sim/src/realm_split.rs`. BFS partition from the capital province assigns ceil(total/2) provinces to the winner. Remaining provinces form a breakaway faction with -200 hostile opinion. MIN_PROVINCES_FOR_SPLIT=3 prevents micro-realm fragmentation.

### 7.7 IN PROGRESS: Bevy 3D Client

**Status**: Bevy 3D client implementation started in Phase 3.

GLB assets have been created for medieval village buildings (houses, markets, temples, fortifications). The Bevy client is now under development. Key decisions made:
- API is stable as of Phase 2 completion
- Asset pipeline (GLB -> Bevy scenes) is defined
- Native-only rendering chosen for MVP (WebGPU deferred to Phase 4)
- Camera system: orthographic overhead strategy view with WASD pan and scroll zoom
- Hex-tile map rendering on XZ plane with 25 provinces
- Three-plugin architecture (Network, Map, UI) via egui integration

### 7.8 OPEN: Multi-Step Army Movement

Currently, armies can only move to directly adjacent provinces (one hop per turn). Multi-step movement would require:
- Pathfinding algorithm (A* or BFS shortest path on the adjacency graph)
- Movement queue (list of provinces to traverse, one per turn)
- Interruption mechanics (what happens if a blocking army appears on the path mid-movement?)
- Gas implications of pathfinding on every tick

### 7.9 OPEN: Fog of War

Per-faction province visibility would add strategic depth but introduces complexity:
- Which provinces can a faction see? (own provinces + adjacent + trade route endpoints?)
- How does fog interact with the API? (filter query_state responses per-wallet?)
- Determinism concern: fog is presentation-layer, not simulation-layer. The simulation must remain fully computed; fog only affects what the API returns.
- Client rendering: hidden provinces shown as greyed-out or completely absent?

### 7.10 OPEN: WASM Gas Profiling

With all Phase 2 systems active (birth, trade, intrigue, realm split, lifecycle), the per-tick gas cost needs real measurement. Key questions:
- What is the worst-case gas consumption for a full tick with all caps hit simultaneously?
- Does the 10M WASM gas budget still hold with 16 pipeline steps?
- Should per-step caps be tightened or loosened based on actual profiling data?
- Is there a meaningful difference between wasmtime and wasmer gas accounting for this workload?

### 7.11 DEFERRED: WebGPU vs. Native-Only Rendering

**Status**: DEFERRED to Phase 4. Native rendering for MVP, WebGPU deferred. The Bevy client MVP uses native rendering (wgpu with Vulkan/Metal/DX12 backends). WebGPU/WASM compilation is deferred to Phase 4. The Bevy ecosystem's WebGPU support is maturing but not production-ready for complex 3D scenes with egui overlays. Native rendering provides better performance and reliability for the initial release.

---

## 8. Dependency Graph

```
crown-ash-api
  +-- crown-ash-sim
  |     +-- crown-ash-types
  +-- crown-ash-types
  +-- axum 0.7
  +-- tokio (sync)
  +-- tracing
  +-- rocksdb (persistence layer)

crown-ash-plugin
  +-- crown-ash-sim
  |     +-- crown-ash-types
  +-- crown-ash-types
  +-- serde_json

crown-ash-sim
  +-- crown-ash-types
  +-- serde + serde_json
  +-- bincode 1.3

crown-ash-types
  +-- serde 1 (derive)
  +-- bincode 1.3

crown-ash-client
  +-- crown-ash-types
  +-- bevy 0.15
  +-- bevy_egui 0.34
  +-- reqwest (async HTTP)
```

No external RNG crate. No floating-point crate. No crypto crate (host provides SHA3-256 via FFI). The simulation compiles to a self-contained WASM module with minimal dependencies. The client crate does NOT depend on crown-ash-api (avoids pulling axum/q-storage into the desktop binary).

---

## 9. How to Run

```bash
# Run all Crown & Ash tests (129 tests: 114 unit/integration + 15 stress tests)
cargo test --package crown-ash-sim --package crown-ash-types --package crown-ash-plugin --package crown-ash-api

# Run only sim tests
cargo test --package crown-ash-sim

# Run with output to see tick summaries
cargo test --package crown-ash-sim -- --nocapture

# Check compilation (fast, no tests)
cargo check --package crown-ash-sim --package crown-ash-types --package crown-ash-plugin --package crown-ash-api --package crown-ash-client
```

---

## 10. Phase 3 Roadmap (IN PROGRESS)

| # | Deliverable | Priority | Depends On | Notes |
|---|-------------|----------|-----------|-------|
| 1 | Bevy 3D client (native rendering) | HIGH | API stable (done) | GLB assets already created for medieval village buildings |
| 2 | Integration stress tests (15 tests) | HIGH | Sim stable (done) | Full pipeline stress testing across 500+ ticks |
| 3 | Multi-step army movement / pathfinding | HIGH | Map (done) | A* or BFS on adjacency graph, movement queue with interruption |
| 4 | Fog of war (per-faction visibility) | MEDIUM | Client integration | Presentation-layer only; simulation stays fully computed |
| 5 | WASM gas profiling and optimization | MEDIUM | All Phase 2 systems (done) | Need real measurements with all 16 pipeline steps active |
| 6 | Diplomacy expansion (vassalization, coalitions) | MEDIUM | Intrigue system (done) | Formal vassal relationships, multi-faction coalitions against threats |
| 7 | Religion mechanics (conversion, holy wars, papal system) | MEDIUM | Cohesion system (done) | Province-level religion, conversion pressure, holy war casus belli, religious authority |
| 8 | Education system (children gain traits/skills over time) | LOW | Birth system (done) | Age-gated skill progression, mentor assignment, trait acquisition events |

### 10.1 Phase 3 Deliverables (COMPLETE)

Phase 3 adds the visual client, stress testing, and documentation updates.

**Bevy 3D Client** (`crates/crown-ash-client/`) -- Campaign map, UI panels, REST/SSE networking:
- Native desktop client using Bevy 0.15 game engine
- Three plugin architecture: CrownAshNetworkPlugin, CrownAshMapPlugin, CrownAshUiPlugin
- HTTP polling every 5 seconds via async I/O task pool
- egui-based UI panels (top bar, detail panel, event feed)
- Orthographic camera with WASD pan and scroll zoom
- 25 hex-tile provinces on XZ plane with faction coloring and terrain tinting
- Army icons with faction colors at province positions
- Click-to-select provinces with detail panel population
- Action submission via POST /crown-ash/action

**Integration Stress Tests** (`crates/crown-ash-sim/tests/stress_tests.rs`) -- 15 tests running 500-1000 ticks:

| # | Test Name | Ticks | Verifies |
|---|-----------|-------|----------|
| 1 | stress_500_ticks_no_panic | 500 | No panics across full pipeline |
| 2 | stress_population_stays_positive | 500 | Population > 0, < 1M per province |
| 3 | stress_at_least_one_faction_alive | 500 | Active factions > 0 always |
| 4 | stress_character_count_bounded | 500 | Tombstoning works, alive < 200 |
| 5 | stress_trade_routes_form_and_decay | 500 | Routes form, cap <= 50 |
| 6 | stress_intrigue_plots_fire | 500 | Plot cap <= 20, no crash |
| 7 | stress_births_produce_valid_characters | 500 | Born chars valid stats |
| 8 | stress_realm_splits_valid | 500 | Split factions have provinces |
| 9 | stress_determinism_500_ticks | 500 | Same seed = identical bincode bytes |
| 10 | stress_army_count_bounded | 500 | Army count stays reasonable |
| 11 | stress_prosperity_bounded | 500 | 0 <= prosperity <= 1M (strict) |
| 12 | stress_unrest_clamped | 500 | 0 <= unrest <= 1M (strict) |
| 13 | stress_different_seeds_diverge | 500 | Different seeds ≠ same state |
| 14 | stress_cohesion_clamped | 500 | All 5 components in 0..1M |
| 15 | stress_1000_ticks_endurance | 1000 | 1000 ticks, still alive |

**Technical Review Update** -- This document (v3.0.0 -> v3.1.0):
- Updated test counts to include +15 stress tests (114 -> 129 total)
- Added Phase 3 deliverables and Bevy client architecture documentation
- Added crown-ash-client to crate locations and dependency graph
- Updated open questions with Bevy client status and WebGPU deferral

**Client Crate Dependencies**: bevy 0.15, bevy_egui 0.33, reqwest, crown-ash-types (does NOT depend on crown-ash-api to avoid pulling axum/q-storage)

### 10.2 Bevy Client Architecture

```text
+---------------------------------------------------+
|  Turn: 47  |  6 factions alive  |  Pop: 250,000   |
+---------------------------------------------------+
|                           |  DETAIL PANEL (350px)  |
|   MAP (hex tiles)         |  Province / Faction /  |
|   Camera: ortho WASD+zoom |  Character / Army info |
+---------------------------------------------------+
|  EVENT FEED (150px, scrolling)                     |
+---------------------------------------------------+
```

**Plugin Architecture**:

| Plugin | Responsibility | Systems |
|--------|---------------|---------|
| CrownAshNetworkPlugin | HTTP polling for world state, SSE event parsing, connection status tracking | poll_server |
| CrownAshMapPlugin | Province hex rendering, army icons, camera control, click-to-select | setup_camera, setup_map, update_map_colors, update_armies, handle_province_click, camera_pan, camera_zoom |
| CrownAshUiPlugin | egui overlay panels, player action submission via POST | top_bar, detail_panel, event_feed, action_submit |

**Key Resources**:

| Resource | Purpose |
|----------|---------|
| ClientGameState | World snapshot (provinces, factions, armies, characters), events buffer, connection status |
| CrownAshConfig | Server URL, poll interval, wallet address |
| Selection | Currently selected province/faction/character/army for detail panel |
| NetworkState | Poll timer, pending request handle |
| ActionState | Wallet input, action result feedback |

**Key Components**:

| Component | Purpose |
|-----------|---------|
| ProvinceMarker | Attached to hex tile entities, stores province ID for click detection and color updates |
| ArmyMarker | Attached to army icon entities, stores army ID for rendering at province positions |

**System Details**:

- **camera_pan / camera_zoom**: WASD keys pan the orthographic camera across the XZ plane. Mouse scroll wheel adjusts zoom level with min/max bounds. Camera starts centered on the map.
- **setup_map / update_map_colors (map_render)**: Spawns 25 hex tile meshes on the XZ plane at positions matching the adjacency graph from `crown-ash-sim/src/map.rs`. Each tile is colored by its controlling faction. Terrain type applies a tinting modifier (e.g., forest = darker green, desert = sandy overlay, mountains = grey). Tiles update colors each poll cycle when faction control changes.
- **update_armies**: Spawns/despawns army icon entities at province positions. Each army is rendered with its faction's color. Army count is displayed as a label.
- **top_bar / detail_panel / event_feed (ui_panels)**: Top bar shows current turn, active faction count, total population. Detail panel shows selected entity info (province stats, faction treasury, character traits, army composition). Event feed is a scrolling log of recent game events with turn numbers.
- **action_submit**: Reads player input from UI action buttons and wallet field. Constructs the appropriate `QueuedAction` JSON payload and sends it via `POST /api/v1/crown-ash/action` using reqwest. Displays success/error feedback in ActionState.

**Province Positioning**: Fixed `PROVINCE_POSITIONS: [(f32, f32); 25]` array mapping province IDs to XZ coordinates, derived from the adjacency layout in `crown-ash-sim/src/map.rs`. The 25 hex tiles are arranged on the XZ plane to match the adjacency graph -- neighboring provinces in the graph are adjacent hex tiles on the map. The orthographic camera looks down the Y axis.

### 10.3 Phase 4 Deliverables (IN PROGRESS)

Phase 4 adds real-time streaming, simulation bug fixes, and new UI features.

**Bug Fixes**:
- **SIM-001**: Unrest overshoot fixed — added `clamp_province_values()` at tick step 8b to enforce [0, 1000] bounds after all modifiers (economy, events, intrigue). Added unrest/prosperity bounds to `assert_invariants()`.
- **SIM-002**: Prosperity overshoot fixed — included in same clamp sweep as SIM-001.

**SSE Streaming Endpoint** (`crates/crown-ash-api/src/streaming.rs`):
- `GET /stream` SSE endpoint with `tokio::sync::broadcast` channel
- `StreamEvent` type with event_type + JSON payload
- `create_event_channel()` factory for the broadcast sender
- `broadcast_event()` helper for the tick loop
- 15-second heartbeat to keep proxies alive
- Lag detection: clients notified when they miss events
- 3 unit tests for channel creation, zero-receiver, and delivery

**Minimap Widget** (`crates/crown-ash-client/src/systems/ui_panels.rs`):
- Egui window (bottom-right, 200x200px) showing all 25 provinces as colored dots
- 40 adjacency lines rendered as semi-transparent edges
- Faction colors from live game state (falls back to defaults when disconnected)
- Click-to-select: clicking a province dot on the minimap updates the Selection resource
- Selected province highlighted with white ring
- Province positions and adjacency data mirrored from map_render module

**Updated Test Counts**: 132 total (80 sim unit + 15 stress + 19 types + 11 api + 7 plugin)

**Updated System Count**: CrownAshUiPlugin now registers 5 systems: top_bar, detail_panel, event_feed, minimap, action_buttons

**Issue Tracker**: `docs/crown-ash-issues.md` created with:
- 2 closed bugs (SIM-001, SIM-002)
- 2 implemented features (CLIENT-001 server-side SSE, CLIENT-002 minimap)
- 6 planned features (CLIENT-003 audio, CLIENT-004 tutorial, SIM-003 save/load, SIM-004 multiplayer, SIM-005 sieges, SIM-006 relationships)
