# Crown & Ash — Issue Tracker

## Active Issues

(None currently)

---

## Planned Features (Phase 4 — UI/Streaming)

### CLIENT-001: SSE Streaming (Server + Client)
- **Status**: ✅ Implemented (both sides)
- **Priority**: High
- **Description**: Server-side `GET /stream` SSE endpoint with broadcast channel, heartbeat (15s), and lag detection. Client-side SSE consumer replaces HTTP polling: dedicated network thread with tokio runtime connects to SSE, parses events (crown_ash_turn triggers REST fetch for full WorldSnapshot), exponential backoff reconnection, 60s read timeout for dead connection detection. REST fallback on SSE disconnect. Includes SseParser with 8 unit tests.
- **Files**: `crates/crown-ash-api/src/streaming.rs`, `crates/crown-ash-client/src/plugins/network.rs`, `crates/crown-ash-client/src/resources/config.rs`

### CLIENT-002: Minimap Widget
- **Status**: ✅ Implemented
- **Priority**: Medium
- **Description**: Added egui minimap window (bottom-right) showing all 25 provinces as colored dots with adjacency lines. Click-to-select provinces on the minimap. Shows faction colors from live game state.
- **Files**: `crates/crown-ash-client/src/systems/ui_panels.rs`, `crates/crown-ash-client/src/plugins/ui.rs`

### CLIENT-005: Fix URL Path Mismatch (World Data Not Loading)
- **Status**: ✅ Closed
- **Priority**: Critical
- **Found**: v0.3.0 Windows testing — client connects but shows "Waiting for world..."
- **Root Cause**: Client used `/crown-ash/world` and `/crown-ash/stream` but server mounts at `/api/v1/crown-ash/*`. All 3 REST/SSE/Join URLs missing the `/api/v1` prefix.
- **Fix**: Updated `fetch_world_with_client()`, `consume_sse()`, and join POST URL to use `/api/v1/crown-ash/*`.
- **Files**: `crates/crown-ash-client/src/plugins/network.rs` (lines 235, 389), `crates/crown-ash-client/src/systems/ui_panels.rs` (join URL)

### CLIENT-006: OAuth2 Device-Login Integration
- **Status**: ✅ Closed
- **Priority**: High
- **Description**: Replaced raw wallet-address text input in Join dialog with OAuth2 device-login flow (same as miner uses). Player clicks "Login with Quillon Wallet" → browser opens `quillon.xyz/miner-login?code={device_code}` → player authenticates in browser → client auto-receives wallet address via polling. Includes manual wallet fallback mode, auto-browser-open via `open` crate, 10-minute polling timeout, and error/retry UI.
- **Files**: `crates/crown-ash-client/src/systems/ui_panels.rs` (JoinState, DeviceLoginPhase, join_dialog, request_device_login), `crates/crown-ash-client/Cargo.toml` (added `open = "5"`)

### CLIENT-007: Windows x64 Build Support
- **Status**: ✅ Closed
- **Priority**: High
- **Description**: Cross-compiled Bevy client to Windows x64 via MinGW (`x86_64-pc-windows-gnu`). Binary is 78MB stripped, statically linked (no DLL dependencies beyond Windows system DLLs). DirectX 12 via wgpu. Available at `quillon.xyz/downloads/crown-ash-client-v0.3.0-windows-x64.exe`.
- **Files**: `.cargo/config.toml` (already had MinGW linker config), `crates/crown-ash-client/Cargo.toml`

### CLIENT-003: Sound/Music Hooks
- **Status**: ⚪ Planned
- **Priority**: Low
- **Description**: Add Bevy audio resource loading for ambient medieval music and event sound effects (battle clash, plague toll, harvest cheer, rebellion drums). Use Bevy's `AudioPlugin` with volume controls in settings.
- **Files**: `crates/crown-ash-client/src/plugins/audio.rs` (new)

### CLIENT-004: Tutorial Overlay
- **Status**: ⚪ Planned
- **Priority**: Low
- **Description**: First-time player tutorial using egui overlay panels. Step through: select province → view details → raise army → move army → diplomacy. Track tutorial progress in a `TutorialState` resource.
- **Files**: `crates/crown-ash-client/src/systems/tutorial.rs` (new)

---

## Planned Features (Phase 5 — Narrative & AI Text Generation)

### NARR-001: Narrative Engine Crate (crown-ash-narrative)
- **Status**: ✅ Implemented
- **Priority**: High
- **Description**: New crate providing two-tier text generation for game events:
  - **Tier 1 (Template)**: Handwritten narrative templates with variable substitution. Every event gets rich prose instantly (no LLM needed). 50+ templates covering all 23 GameEvent variants with faction/terrain/character context. Zero latency, deterministic.
  - **Tier 2 (LLM)**: On-demand deep narrative for important moments (succession crises, epic battles, diplomatic betrayals). Uses `q-ai-inference` LlamaCppEngine with Mistral-7B or Nemotron. Streaming tokens to UI. Non-blocking (game continues while text generates).
- **Architecture**:
  ```
  GameEvent → NarrativeEngine → Tier 1 (instant template) → UI event feed
                              → Tier 2 (LLM, async)       → Chronicle panel
  ```
- **Files**: `crates/crown-ash-narrative/` (new crate)
- **Dependencies**: `crown-ash-types`, `q-ai-inference` (optional, feature-gated)

### NARR-002: Event Narrative Templates (Tier 1)
- **Status**: ✅ Implemented
- **Priority**: High
- **Description**: Rich handwritten templates for all 23 GameEvent types. Each event type has 3-5 template variants selected by context (faction culture, terrain, character traits). Templates use variable substitution: `{ruler_name}`, `{province_name}`, `{faction_name}`, `{casualties}`, etc.
- **Examples**:
  - Plague: *"A terrible pestilence sweeps through {province}. The streets of {province} are lined with the dead — {pop_lost} souls claimed by the black rot. The Temple of {religion} offers no comfort."*
  - Battle: *"Steel met steel on the plains of {province}. The {attacker} host clashed with {defender}'s garrison — {casualties} fell before the day was done. {victor} holds the field."*
  - Succession: *"The crown sits uneasy. With {dead_ruler}'s death, {faction} descends into chaos. {heir_count} claimants eye the throne."*
- **Coverage**: All 23 event types × 3-5 variants each = 80-120 templates
- **Files**: `crates/crown-ash-narrative/src/templates.rs`

### NARR-003: Character Chronicle System
- **Status**: ✅ Implemented
- **Priority**: High
- **Description**: Every character accumulates a narrative history — a "chronicle" of their life events. When you click a character in the detail panel, you see their story: born, married, fought battles, gained traits, ruled provinces, plotted assassinations, died.
  - Events are filtered per-character from the turn history
  - Template engine renders each event as a prose paragraph
  - LLM can optionally generate a "biography summary" for important characters
  - Chronicles persist across turns (stored in `CharacterChronicle` struct)
- **UI**: New "Chronicle" tab in the detail panel, scrollable prose text
- **Files**: `crates/crown-ash-narrative/src/chronicle.rs`, `crates/crown-ash-client/src/systems/ui_panels.rs`

### NARR-004: NPC Dialog & Personality System
- **Status**: ✅ Implemented (Core Personality Engine)
- **Priority**: Medium
- **Description**: Faction leaders and notable characters can "speak" via LLM-generated dialog. Each character has a personality profile derived from their traits (e.g., Ambitious + Cruel = threatening tone, Pious + Generous = benevolent sermons).
  - **Personality Prompt**: Constructed from character traits, faction culture, current situation (at war? losing? prosperous?)
  - **Dialog Triggers**: Diplomacy proposals, war declarations, succession speeches, plot discoveries
  - **Cascade Pattern**: Short dialog (1-2 sentences) generated on every trigger. Long monologue (paragraph) only for major events.
  - **Model**: Mistral-7B for fast dialog, Nemotron/larger model for important speeches
  - **Streaming**: Tokens stream into a speech bubble UI element in real-time
- **Files**: `crates/crown-ash-narrative/src/dialog.rs`, `crates/crown-ash-narrative/src/personality.rs`

### NARR-005: Province & Faction History Narratives
- **Status**: ✅ Implemented
- **Priority**: Medium
- **Description**: Rich history text for provinces and factions:
  - **Province History**: "Ashenmere has changed hands 3 times. Once a prosperous heartland of the Ashen Crown, it was conquered by the Frost Marches in Turn 45, only to fall to the Salt League in Turn 89..."
  - **Faction History**: "The Vale Princes rose from a minor house to control 8 provinces. Under King Aldric's rule, they waged 3 wars and signed 2 treaties..."
  - Generated from accumulated event history using templates + optional LLM summary
- **Files**: `crates/crown-ash-narrative/src/history.rs`

### NARR-006: LLM Integration via q-ai-inference
- **Status**: ✅ Implemented (Prompt Engine)
- **Priority**: Medium
- **Description**: Wire Crown & Ash narrative engine into the existing `q-ai-inference` crate:
  - Feature-gated: `crown-ash-narrative = { features = ["llm"] }` — without the feature, only templates work
  - Uses `LlamaCppEngine` for local inference (no API calls, no cloud)
  - Model: Mistral-7B-Instruct (4.37GB GGUF, already in model catalog)
  - Future: Nemotron-Mini for faster dialog, Nemotron-70B for epic narratives on GPU nodes
  - Deterministic mode available for blockchain-verifiable narrative (same event → same text on every node)
  - Token streaming via `StreamEvent::Token()` to client SSE
- **Server-Side**: Narrative generation runs on q-api-server, results broadcast via SSE
- **Client-Side**: Client receives pre-generated narrative text, no local LLM needed
- **Files**: `crates/crown-ash-narrative/src/llm.rs`, `crates/crown-ash-api/src/handlers.rs`

### NARR-007: Cascading Text Generation (Nemotron Cascade Pattern)
- **Status**: ✅ Implemented (Cascade Engine)
- **Priority**: Low
- **Description**: Multi-tier text generation cascade for optimal quality/speed tradeoff:
  ```
  Event occurs
    → Tier 0: Structured data (instant, always)     → API/SSE
    → Tier 1: Template narrative (instant, always)   → Event feed
    → Tier 2: Short LLM dialog (1-3s, if notable)   → Speech bubble
    → Tier 3: Deep LLM narrative (5-15s, if epic)    → Chronicle
  ```
  Each tier fires independently. Lower tiers never wait for higher tiers. The UI progressively enriches as higher-tier text arrives. This is the "cascade" pattern — fast first, rich later.
- **Cascade triggers**:
  - Every event → Tier 0 + 1
  - War declared, treaty signed, succession → Tier 0 + 1 + 2
  - Epic battle (>500 casualties), realm split, faction eliminated → All 4 tiers
- **Files**: `crates/crown-ash-narrative/src/cascade.rs`

### NARR-008: Client UI Narrative Integration
- **Status**: ✅ Implemented
- **Priority**: High
- **Description**: Wire the narrative engine into the Bevy client UI:
  - **Event feed**: Shows Tier 1 template prose instead of raw `format_event()`. Color-coded by importance: gold (Epic), blue (Notable), gray (Minor). Heading changed from "Event Log" to "Chronicle".
  - **Character Chronicle tab**: Scrollable life history in the detail panel when a character is selected. Shows personality archetype (Tyrant, Saint, Schemer, etc.) derived from traits.
  - **Province History**: "History" section at bottom of province detail panel showing accumulated event narrative (conquests, battles, plagues, etc.).
  - **Faction History**: "History" section at bottom of faction detail panel showing faction-level narrative (wars, treaties, conquests, succession crises).
  - **Narrative Update Systems**: Three Bevy systems (`update_event_narratives`, `update_chronicles`, `update_histories`) process new events incrementally each frame. History regenerates every 5 turns.
  - **NarrativeState Resource**: Central Bevy resource holding chronicles, event narratives, province/faction histories, and LLM results.
- **Files**: `crates/crown-ash-client/src/resources/narrative_state.rs`, `crates/crown-ash-client/src/systems/narrative_update.rs`, `crates/crown-ash-client/src/systems/ui_panels.rs`, `crates/crown-ash-client/src/plugins/ui.rs`

### NARR-009: Server-Side Cascade SSE Broadcasting
- **Status**: ✅ Implemented
- **Priority**: High
- **Description**: Server-side integration of the cascade engine into the SSE event system:
  - New SSE event types: `crown_ash_prose` (Tier 1), `crown_ash_dialog` (Tier 2), `crown_ash_epic` (Tier 3), `crown_ash_token` (streaming)
  - `broadcast_cascade_narratives()` processes all turn events through the cascade engine, broadcasts Tier 1 prose immediately, returns cascade results for async Tier 2/3 LLM processing
  - Each SSE event includes `tier` field (0-3) and `importance` field so clients know where to display and how to color-code
  - Payload builders for all 4 cascade SSE event types with tests
  - `crown-ash-narrative` crate added as dependency to `crown-ash-api`
- **Files**: `crates/crown-ash-api/src/events.rs`, `crates/crown-ash-api/Cargo.toml`

### NARR-010: Dialog Speech Bubble UI (Client)
- **Status**: ✅ Implemented
- **Priority**: High
- **Description**: Floating speech bubble overlay system for Tier 2 (dialog) and Tier 3 (epic) LLM-generated narrative text:
  - `DialogState` resource tracks active bubbles (max 4 visible, oldest evicted)
  - `DialogBubble` struct: speaker, text, tier, countdown timer, turn
  - Auto-dismiss: 8s for dialog, 12s for epic narratives, with 1.5s fade-out
  - Tier-based visual styling: dark blue-grey for dialog (speaker says: "text"), dark gold for epic (~Narrator~ italic text)
  - Thin progress bar shows remaining time per bubble
  - Stacked vertically from top-right corner, 120px apart
  - Network integration: `crown_ash_dialog` and `crown_ash_epic` SSE events parsed in network plugin, delivered via `NetMessage::NarrativeDialog` through the mailbox to `DialogState`
  - Zero-copy: bubbles rendered directly from `DialogState` each frame, timer ticked with Bevy `Time::delta_secs()`
- **Files**: `crates/crown-ash-client/src/resources/narrative_state.rs` (DialogBubble, DialogState), `crates/crown-ash-client/src/plugins/network.rs` (SSE handling + drain), `crates/crown-ash-client/src/systems/ui_panels.rs` (dialog_bubbles system), `crates/crown-ash-client/src/plugins/ui.rs` (registration)

---

## Planned Features (Phase 4 — Simulation Remaining)

### SIM-007: Multi-step Army Pathfinding
- **Status**: ✅ Closed (Phase 3)
- **Priority**: High
- **Description**: BFS pathfinding across the 25-province adjacency graph. Armies queue multi-hop routes via `movement_queue: Vec<ProvinceId>`. Each tick pops one hop, respecting ZOC (zone of control — enemy provinces block pathing). `plan_route()` returns shortest path avoiding enemy territory.
- **Files**: `crates/crown-ash-sim/src/combat.rs` (plan_route, advance_armies), `crates/crown-ash-types/src/army.rs` (movement_queue field)

### SIM-008: Religion Mechanics
- **Status**: ✅ Closed (Phase 3)
- **Priority**: High
- **Description**: Religious authority per realm (0-1000) based on province religion match, temples, ruler traits, and chaplain learning. Gradual province conversion (progress 0-1000 per turn). Heresy events when authority <300 (1/50 chance). Miracle events when authority >700 + temple (1/100 chance). Authority affects clerical_favor cohesion.
- **Files**: `crates/crown-ash-sim/src/religion.rs`, `crates/crown-ash-types/src/realm.rs` (religious_authority), `crates/crown-ash-types/src/province.rs` (conversion_progress), `crates/crown-ash-types/src/event.rs` (ReligiousConversion, Heresy, Miracle)

### SIM-009: Education System
- **Status**: ✅ Closed (Phase 3)
- **Priority**: High
- **Description**: Age-gated skill progression. Children 6-15 gain stats yearly: focus_stat += 2 + mentor_bonus (0-3), random off_stat += 1. Mentor is highest-learning adult in faction; mentor's best stat determines child's focus. At age 16, graduation grants a trait based on highest stat (Strategist/Brave for martial, Scholar/Theologian for learning, etc.).
- **Files**: `crates/crown-ash-sim/src/education.rs`

### SIM-010: Diplomacy Expansion (Vassals, Coalitions, Tribute)
- **Status**: ✅ Closed (Phase 3)
- **Priority**: High
- **Description**: Tribute collection (vassals pay 5 gold/province/turn to liege). Vassal revolts (1/20 chance when opinion < -300). Treaty expiration with ally list cleanup. Grievance decay with opinion restoration. Coalition formation when any faction controls >40% of provinces (defensive alliance, 50-turn expiry).
- **Files**: `crates/crown-ash-sim/src/diplomacy.rs`

### SIM-011: Fog of War
- **Status**: ✅ Closed (Phase 3)
- **Priority**: Medium
- **Description**: Presentation-layer filtering. `visible_provinces()` returns owned provinces + neighbors. `snapshot_world_for_faction()` produces a redacted WorldSnapshot: hidden provinces have zeroed population/garrison/resources, armies in non-visible provinces excluded, only own-faction characters + rulers + visible army commanders shown. Sim runs on full state; fog applied at read time.
- **Files**: `crates/crown-ash-sim/src/lib.rs` (visible_provinces, snapshot_world_for_faction)

---

## Planned Features (Phase 5 — Simulation)

### SIM-003: Save/Load Game State
- **Status**: ⚪ Planned
- **Priority**: High
- **Description**: Serialize `GameWorld` to/from bincode for save/load. Server endpoint `POST /crown-ash/save` and `POST /crown-ash/load`. Client button in settings panel. Saves stored on-chain as compressed blobs.
- **Files**: `crates/crown-ash-api/src/persistence.rs`, `crates/crown-ash-sim/src/world_state.rs`

### SIM-004: Multiplayer Lobby System
- **Status**: ⚪ Planned
- **Priority**: High
- **Description**: Pre-game lobby where players choose factions before world generation. Wallet-based authentication (reuse Q-NarwhalKnight wallet). Lobby state: waiting → ready → started. Max 5 human players (remaining factions are AI).
- **Files**: `crates/crown-ash-api/src/handlers.rs`, `crates/crown-ash-api/src/lib.rs`

### SIM-005: Siege Mechanics
- **Status**: ✅ Closed
- **Priority**: Medium
- **Description**: Fortified provinces (fortification > 0) require siege before capture. `SiegeProgress` struct on Army tracks target, defender, turns_besieged/required. Duration = (fortification + 1) × 3 turns. Province attrition during siege: -20 prosperity, +15 unrest, -0.5% population/turn. On completion: garrison destroyed, attacker takes 20% garrison casualties, province captured with WarDamage scar. Siege cancels if army moves away, dies, or defending army arrives. Besieging armies cannot move. Unfortified provinces still captured instantly. 6 tests (start, tick-to-completion, cancel-on-move, instant-capture, stays-put, army-cannot-move).
- **Files**: `crates/crown-ash-types/src/army.rs` (SiegeProgress), `crates/crown-ash-types/src/event.rs` (SiegeStarted/SiegeCompleted), `crates/crown-ash-sim/src/combat.rs` (process_sieges, apply_province_capture), `crates/crown-ash-sim/src/tick.rs` (step 3b)

### SIM-006: Character Relationships & Marriage Alliances
- **Status**: ✅ Closed
- **Priority**: Medium
- **Description**: Personal relationship system between characters. `PersonalRelation` struct with opinion (-1000 to +1000), named `RelationType` (Friend, Rival, Mentor, MarriageAlliance), and timed `OpinionModifier`s. Each tick: decay modifiers, same-faction proximity bonding (+2/turn), threshold checks (Friend at +50, Rival at -50), marriage alliance diplomatic effects (+3 faction opinion/turn, capped at 200), prune dead relations. `on_marriage()` creates MarriageAlliance relation with +30 initial opinion. Max 12 relations per character. 5 tests (proximity bonding, friendship threshold, rivalry threshold, marriage alliance faction boost, dead relation pruning).
- **Files**: `crates/crown-ash-types/src/character.rs` (PersonalRelation, RelationType, OpinionModifier), `crates/crown-ash-types/src/event.rs` (Friendship, Rivalry, MarriageAlliance), `crates/crown-ash-sim/src/relationships.rs` (new), `crates/crown-ash-sim/src/tick.rs` (step 7c)

---

## Closed Issues

### SIM-001: Unrest Can Exceed 1000 Cap After Random Events
- **Status**: ✅ Closed
- **Severity**: Medium
- **Found**: Phase 3 stress tests (tick 143, province 20, unrest=1008)
- **Root Cause**: `update_unrest()` (tick step 6) clamps to [0, 1000], but `roll_events()` (tick step 8) adds +50 unrest for famine events without re-clamping.
- **Fix**: Added `clamp_province_values()` at step 8b (after events, before succession). Added unrest/prosperity bounds to `assert_invariants()`. Tightened stress test to strict [0, 1000] bounds.
- **Files**: `crates/crown-ash-sim/src/tick.rs`, `crates/crown-ash-sim/src/world_state.rs`, `crates/crown-ash-sim/tests/stress_tests.rs`

### SIM-002: Prosperity Not Clamped After Famine/Plague Events
- **Status**: ✅ Closed
- **Severity**: Low
- **Found**: Code review during SIM-001 investigation
- **Root Cause**: Prosperity only upper-bounded during harvest events. Economy step could push above 1000.
- **Fix**: Included in same `clamp_province_values()` sweep as SIM-001.
- **Files**: `crates/crown-ash-sim/src/tick.rs`
