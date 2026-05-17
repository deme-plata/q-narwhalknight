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

### NARR-011: Turn Summary Narrative (Event Feed Headers)
- **Status**: ✅ Implemented
- **Priority**: Medium
- **Description**: Concise 1-3 sentence turn summaries displayed as gold italic headers in the event feed, grouping events by turn. `turn_summary()` aggregates events per turn — highlights battles, conquests, wars, deaths, plagues, and succession crises. Top 4 highlights rendered. Client-side: `NarrativeState.turn_summaries` stores (turn, prose) pairs; `update_turn_summaries` system generates summaries on turn change (catches up after reconnect); event feed renders turn header before each turn's events with visual separator. 7 tests covering empty turns, multi-event turns, war/death combinations.
- **Files**: `crates/crown-ash-narrative/src/history.rs` (TurnStats, turn_summary), `crates/crown-ash-client/src/systems/narrative_update.rs` (update_turn_summaries, event_turn), `crates/crown-ash-client/src/systems/ui_panels.rs` (event_feed turn headers), `crates/crown-ash-client/src/resources/narrative_state.rs` (turn_summaries field)

### CLIENT-008: Notification Toasts for Important Events
- **Status**: ✅ Implemented
- **Priority**: Medium
- **Description**: Auto-dismissing notification toasts that appear on the left side of the screen when Epic or Notable events occur. Distinct from dialog bubbles (LLM text, right side) and event feed (scrolling log, bottom). `ToastState` resource tracks up to 5 simultaneous toasts. Duration: 6s for Epic, 4s for Notable, 3s for Minor. 1s fade-out. Color-coded by importance: gold for Epic, blue for Notable. Triggered automatically from `update_event_narratives` when new events arrive — only Epic and Notable events push toasts using event summary text.
- **Files**: `crates/crown-ash-client/src/resources/narrative_state.rs` (NotificationToast, ToastState), `crates/crown-ash-client/src/systems/ui_panels.rs` (notification_toasts system), `crates/crown-ash-client/src/systems/narrative_update.rs` (toast push in update_event_narratives), `crates/crown-ash-client/src/plugins/ui.rs` (registration)

### NARR-012: War Detail Narrative (Faction Panel)
- **Status**: ✅ Implemented
- **Priority**: Medium
- **Description**: Rich war summary prose displayed in the faction detail panel when a faction is at war. `war_summary()` scans all events for war declarations, battles, province conquests, sieges, and peace treaties between two specific factions. Uses army_faction lookup to match battles to wars. Client-side: `NarrativeState.war_summaries` caches (faction_a, faction_b) → prose; `update_war_summaries` system refreshes every 3 turns; faction detail panel shows italic war prose (warm brown) below each enemy name. Order-independent key lookup (min, max).
- **Files**: `crates/crown-ash-narrative/src/history.rs` (WarStats, war_summary), `crates/crown-ash-client/src/systems/narrative_update.rs` (update_war_summaries), `crates/crown-ash-client/src/systems/ui_panels.rs` (faction detail war text), `crates/crown-ash-client/src/resources/narrative_state.rs` (war_summaries field)

### NARR-013: Character Relationship Narrative
- **Status**: ✅ Implemented
- **Priority**: Medium
- **Description**: Prose describing a character's personal relationships displayed in the character detail panel. `relationship_narrative()` converts `PersonalRelation` data + event history into natural language: friends ("close allies, bonded since turn 12"), rivals ("bitter rivals, their enmity born since turn 8"), mentors, marriage alliances (with faction names from MarriageAlliance events). Neutral untyped relationships skipped unless opinion is strong (>300 or <-300). Contextual search finds Friendship/Rivalry events for formation timing. 6 tests covering friend/rival/mentor/marriage/neutral/empty cases.
- **Files**: `crates/crown-ash-narrative/src/history.rs` (relationship_narrative, find_relationship_context, find_marriage_factions), `crates/crown-ash-client/src/systems/ui_panels.rs` (character detail panel wiring)

### NARR-014: Province Tooltip on Minimap Hover
- **Status**: ✅ Implemented
- **Priority**: Low
- **Description**: Brief narrative tooltip when hovering over province dots on the minimap. Shows province name, controller faction, and the last sentence of the province's cached history narrative. Tooltip appears when cursor is within 12px of a province dot. Uses existing `NarrativeState.province_histories` cache.
- **Files**: `crates/crown-ash-client/src/systems/ui_panels.rs` (minimap system hover detection + tooltip)

### NARR-016: Realm Prosperity Narrative
- **Status**: ✅ Implemented
- **Priority**: Medium
- **Description**: "State of the Realm" prose displayed at the top of the faction detail panel. `realm_prosperity()` aggregates harvests, famines, plagues, trade routes, construction, rebellions, and territory changes for a faction's controlled provinces. Generates mood-aware prose: "prospers" vs "suffers under hardship" based on good:bad event ratio. Includes specific stats: harvest/famine counts, active trade routes, improvements built, rebellions, border changes. Client-side: `NarrativeState.realm_prosperity` cached per faction, regenerated every 5 turns. 3 tests covering empty/thriving/suffering realms.
- **Files**: `crates/crown-ash-narrative/src/history.rs` (ProsperityStats, realm_prosperity, render_prosperity), `crates/crown-ash-client/src/systems/narrative_update.rs` (update_realm_prosperity), `crates/crown-ash-client/src/systems/ui_panels.rs` (faction detail panel), `crates/crown-ash-client/src/resources/narrative_state.rs` (realm_prosperity field)

### NARR-017: Battle Report Narrative
- **Status**: ✅ Implemented
- **Priority**: Medium
- **Description**: Detailed multi-paragraph battle reports. `battle_report()` generates prose from BattleResult data: location ("Battle of Frosthold"), combatants (from army→faction lookup), severity classification (brief/bloody/fierce/devastating based on total dead), casualty breakdown, and outcome narrative (crushing victory, routed, held firm, repelled). 3 tests covering basic/defender_wins/devastating_victory.
- **Files**: `crates/crown-ash-narrative/src/history.rs` (battle_report)

### NARR-018: Intrigue Plot Narrative
- **Status**: ✅ Implemented
- **Priority**: Medium
- **Description**: Rich prose for intrigue events: assassination ("orchestrated the assassination... swiftly, leaving no trace"), fabricated claims ("forged documents and bribed scribes"), seduction ("employed their charms"), sabotage, plot discovery, and foiled plots. `intrigue_narrative()` scans PlotSucceeded/PlotDiscovered/PlotFoiled events, producing (turn, prose) pairs. Client-side: `NarrativeState.intrigue_narratives` cached, refreshed every 3 turns. 3 tests covering assassination_success/discovered_and_foiled/empty.
- **Files**: `crates/crown-ash-narrative/src/history.rs` (intrigue_narrative), `crates/crown-ash-client/src/systems/narrative_update.rs` (update_intrigue_narratives), `crates/crown-ash-client/src/resources/narrative_state.rs` (intrigue_narratives field)

### NARR-019: Era Summary Narrative
- **Status**: ✅ Implemented
- **Priority**: Medium
- **Description**: World overview prose displayed when nothing is selected in the detail panel. `era_summary()` takes current turn, factions alive, active war pairs, faction province counts, and full event list. Aggregates stats (battles, casualties, wars, treaties, conquests, eliminations, plagues, rebellions, realm splits, trade routes, births, deaths), finds dominant faction, and renders a multi-paragraph overview covering power balance, active wars, bloodshed stats, political upheaval, and hardship. Client-side: `NarrativeState.era_summary_text` cached, refreshed every 10 turns via `update_era_summary` system. 3 tests covering peaceful/war-with-battles/dominant-empire scenarios.
- **Files**: `crates/crown-ash-narrative/src/history.rs` (era_summary, render_era_summary, EraStats), `crates/crown-ash-client/src/systems/narrative_update.rs` (update_era_summary), `crates/crown-ash-client/src/systems/ui_panels.rs` (detail_panel nothing-selected block)

### NARR-020: Province Religion Narrative
- **Status**: ✅ Implemented
- **Priority**: Medium
- **Description**: Rich prose for province religious history displayed in province detail "Faith" section. `religion_narrative()` scans ReligiousConversion, Heresy, and Miracle events for the province. Outputs "steadfast faith" for no activity, conversion history (single or multiple), heresy count, and miracle count. Client-side: `NarrativeState.province_religion` HashMap cached, refreshed every 5 turns via `update_province_religions`. 5 tests covering steadfast/single_conversion/multiple_conversions/heresies_and_miracles/ignores_other_provinces.
- **Files**: `crates/crown-ash-narrative/src/history.rs` (religion_narrative), `crates/crown-ash-client/src/systems/narrative_update.rs` (update_province_religions), `crates/crown-ash-client/src/systems/ui_panels.rs` (province detail Faith section)

### NARR-021: Diplomatic Relations Narrative
- **Status**: ✅ Implemented
- **Priority**: Medium
- **Description**: Bilateral diplomacy prose displayed in faction detail "Diplomatic Relations" section. `diplomacy_narrative()` takes two faction IDs, war status, events. Tracks wars declared, treaties signed, marriages, and province conquests between the pair. Renders current state (at war with Nth conflict / marriage alliance / treaty / uneasy peace / cautious neutrality) plus territory context (seized N provinces). Client-side: `NarrativeState.diplomacy_narratives` HashMap keyed by (faction_a, faction_b), refreshed every 3 turns via `update_diplomacy_narratives`. 5 tests covering neutral/at_war/repeated_wars/marriage/territory_conquest.
- **Files**: `crates/crown-ash-narrative/src/history.rs` (diplomacy_narrative), `crates/crown-ash-client/src/systems/narrative_update.rs` (update_diplomacy_narratives), `crates/crown-ash-client/src/systems/ui_panels.rs` (faction detail Diplomatic Relations section)

### NARR-025: Character Biography Narrative
- **Status**: ✅ Implemented
- **Priority**: Medium
- **Description**: Multi-paragraph character biography generated from identity, traits, battle history, marriages, plots survived, and death. `character_biography()` takes character metadata + events, produces prose like "King Aldric, aged 45, serves as Ruler of Ashen Crown. Known as Brave and Just. Participated in 3 battles on behalf of Ashen Crown. Wed Lady Isolde on turn 12." Wired into character detail "Biography" section before Chronicle. 3 tests covering living_ruler/dead_character/marriage.
- **Files**: `crates/crown-ash-narrative/src/history.rs` (character_biography), `crates/crown-ash-client/src/systems/ui_panels.rs` (character detail Biography section)

### NARR-026: Army Status Narrative
- **Status**: ✅ Implemented
- **Priority**: Medium
- **Description**: Army prose describing composition, commander, location, morale state, and battle record. `army_narrative()` takes army metadata + events, produces prose like "An army of Ashen Crown led by King Aldric, 620 strong, encamped at Frosthold. The host comprises 500 levy, 100 men-at-arms, and 20 knights. Morale is high." Wired into army detail panel as italic text. Battle record shows wins/losses. Morale text varies by level. 3 tests covering basic/battles/low_morale.
- **Files**: `crates/crown-ash-narrative/src/history.rs` (army_narrative), `crates/crown-ash-client/src/systems/ui_panels.rs` (army detail section)

### NARR-027: Construction/Improvement Narrative
- **Status**: ✅ Implemented
- **Priority**: Medium
- **Description**: Province improvement prose showing current buildings and construction history. `construction_narrative()` takes current improvements list + ConstructionComplete events, produces prose like "Frosthold is home to Market, Temple. Most recently, a University was completed on turn 25. 3 construction projects have been completed in total." Wired into province detail "Improvements" section. 4 tests covering empty/with_improvements/recent_build/multiple_builds.
- **Files**: `crates/crown-ash-narrative/src/history.rs` (construction_narrative), `crates/crown-ash-client/src/systems/ui_panels.rs` (province detail Improvements section)

### NARR-022: Siege Narrative
- **Status**: ✅ Implemented
- **Priority**: Medium
- **Description**: Province siege history prose. `siege_narrative()` scans SiegeStarted/SiegeCompleted events for a province. Shows total sieges endured, most recent siege details (attacker, duration, casualties), or ongoing siege info. Wired directly into province detail "Sieges" section (red-tinted italic). 4 tests covering empty/completed/ongoing/multiple.
- **Files**: `crates/crown-ash-narrative/src/history.rs` (siege_narrative), `crates/crown-ash-client/src/systems/ui_panels.rs` (province detail Sieges section)

### NARR-023: Trade Route Narrative
- **Status**: ✅ Implemented
- **Priority**: Medium
- **Description**: Province trade route prose. `trade_narrative()` scans TradeRouteEstablished/TradeRouteDisrupted events. Shows commerce status, most recent route (goods, partner, turn), and disruption details. Wired into province detail "Trade" section (green-tinted italic). 3 tests covering empty/established/disrupted.
- **Files**: `crates/crown-ash-narrative/src/history.rs` (trade_narrative), `crates/crown-ash-client/src/systems/ui_panels.rs` (province detail Trade section)

### NARR-024: Succession Crisis Narrative
- **Status**: ✅ Implemented
- **Priority**: Medium
- **Description**: Faction succession crisis prose. `succession_narrative()` scans SuccessionCrisis/RealmSplit events for a faction. Shows crisis count, claimant count, whether realm held or shattered. Details realm splits with rebel leader and provinces lost. Wired into faction detail "Succession" section (warm brown italic). 4 tests covering empty/single_crisis_held/realm_split/multiple_crises.
- **Files**: `crates/crown-ash-narrative/src/history.rs` (succession_narrative), `crates/crown-ash-client/src/systems/ui_panels.rs` (faction detail Succession section)

### NARR-015: Dynasty Lineage Narrative
- **Status**: ✅ Implemented
- **Priority**: Medium
- **Description**: Lineage prose displayed in the character detail panel. `dynasty_lineage()` scans CharacterBorn events for a dynasty to determine generation position (Founder, Second, Third...), parent identity and fate (death cause), and total dynasty deaths. Produces prose like "Hero — Third of their dynasty line. Child of Ancestor who fell in battle. 2 members of their bloodline have perished." 4 tests covering empty/parent_death/founder/death_counts.
- **Files**: `crates/crown-ash-narrative/src/history.rs` (dynasty_lineage), `crates/crown-ash-client/src/systems/ui_panels.rs` (character detail panel Lineage section)

### CLIENT-009: Fix Tokio Runtime Panic in Join/Login
- **Status**: ✅ Closed
- **Priority**: Critical
- **Description**: Join dialog and OAuth2 device-login used Bevy's `IoTaskPool::get().spawn()` for HTTP requests via reqwest. IoTaskPool is NOT a tokio runtime — reqwest panics with "no reactor running, must be called from the context of a Tokio 1.x runtime". Fix: replaced both IoTaskPool spawns with `std::thread::spawn` + a temporary `tokio::runtime::Builder::new_current_thread()`. Same pattern used by the SSE network thread.
- **Files**: `crates/crown-ash-client/src/systems/ui_panels.rs` (device-login spawn, join-game spawn)

### CLIENT-010: MCP Server for AI Gameplay
- **Status**: ✅ Closed
- **Priority**: Medium
- **Description**: TypeScript MCP server (`tools/crown-ash-mcp/`) enables Claude Code to play Crown & Ash as an AI agent. 24 tools across 4 categories: Observation (world overview, faction details, province details, diplomacy, all armies, my realm), Strategic Analysis (threats, economy, weak targets, characters, strategic briefing), Military Actions (raise/move/disband army, declare war), Diplomacy & Economy (propose/accept treaty, build improvement, set tax rate, trade routes), Characters & Intrigue (assign councilor, designate heir, arrange marriage, convert province, launch/back/investigate plots). Connects to `quillon.xyz` REST API. Configured in `.claude/projects/*/settings.json`.
- **Files**: `tools/crown-ash-mcp/src/index.ts`, `tools/crown-ash-mcp/package.json`

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

---

## Phase 5 — 3D Rendering & Assets

### RENDER-001: 3D Building Models for Province Improvements
- **Status**: ✅ Implemented
- **Priority**: High
- **Description**: GLB model asset pipeline mapping each `Improvement` enum variant to a 3D building model. 27 Sloyd AI models organized into `village/` (12), `town/` (12), `religious/` (3) directories. Buildings arranged in a clock-ring layout within each hex tile (12 outer slots at 30° intervals). Construction-in-progress buildings render at 60% scale. Asset catalog, BuildingAssets resource, BuildingState cache for change detection.
- **Files**: `crates/crown-ash-client/src/systems/building_render.rs`, `crates/crown-ash-client/src/components/building.rs`, `crates/crown-ash-client/assets/models/`

### RENDER-002: Terrain Elevation & Domed Hex Meshes
- **Status**: ✅ Implemented
- **Priority**: High
- **Description**: Per-terrain Y elevation (Mountains +0.45, Hills +0.20, Forest +0.08, Marsh -0.08, Coastal -0.05) gives the map 3D depth. Domed hex meshes with raised centre vertices for hills/mountains (dome heights: Mountains 0.35, Hills 0.15, Forest 0.05). Elevation applied to all entities: province labels, selection ring, army cubes, building models, movement animation.
- **Files**: `crates/crown-ash-client/src/systems/map_render.rs` (`terrain_elevation`, `province_elevation`, `terrain_dome_height`, `build_hex_mesh` now takes dome parameter)

### RENDER-003: Population-Based Decorative Buildings
- **Status**: ✅ Implemented
- **Priority**: Medium
- **Description**: Decorative village/town buildings spawn on inner ring (radius 0.40) based on province population thresholds (0-6 buildings). Village tier (<2000 pop): peasant houses, wells, chicken coops. Town tier (≥2000): merchant houses, taverns, storehouses. Deterministic model selection via province_id hash for visual variety.
- **Files**: `crates/crown-ash-client/src/systems/building_render.rs` (`DecoAssets`, `DecoState`, `update_decorations`), `crates/crown-ash-client/src/components/building.rs` (`DecoMarker`)

### RENDER-004: Tilted Camera & Sky Background
- **Status**: ✅ Implemented
- **Priority**: Medium
- **Description**: Changed camera from pure top-down to tilted isometric-style view (~59° from horizontal). Camera positioned at height 20 with Z-offset 12 behind focal point. Added CameraFocus resource tracking XZ focal point, panning moves focus smoothly. Dark blue sky background via ClearColor. Province labels rotated to face tilted camera angle.
- **Files**: `crates/crown-ash-client/src/systems/camera.rs` (complete rewrite with `CameraFocus` resource)

### RENDER-005: Procedural Terrain Features (Trees, Rocks, Water, Sand, Grass)
- **Status**: ✅ Implemented
- **Priority**: High
- **Description**: Full procedural terrain feature system generating 3D meshes per province based on terrain type. Forest: 5-8 cone+cylinder trees scattered via deterministic hash. Mountains: 3-5 displaced-octahedron rock clusters. Coastal/River: semi-transparent animated water hex overlay with gentle sine wave. Marsh: murky water + reed-like grass tufts. Desert: 2-4 elongated sand dune domes. Hills: grass tufts + occasional lone tree. Plains: sparse grass. All meshes generated at startup, instanced per province. Materials are lit (respond to sun) with appropriate roughness/metallic values.
- **Files**: `crates/crown-ash-client/src/systems/terrain_features.rs` (new, ~450 lines)

### RENDER-006: Lit 3D Rendering with Sun & Ambient Light
- **Status**: ✅ Implemented
- **Priority**: Medium
- **Description**: Upgraded DirectionalLight to 12000 illuminance sun with shadows enabled, angled to create depth on domed terrain. Added AmbientLight (brightness 400, cool-tinted) so shadows aren't pure black. Terrain feature materials switched from unlit to PBR with appropriate roughness (trees 0.95, rocks 0.85, water 0.1 + metallic 0.3, sand 0.95). Province hex tiles remain unlit for clear faction color readability.
- **Files**: `crates/crown-ash-client/src/systems/map_render.rs` (DirectionalLight + AmbientLight), `crates/crown-ash-client/src/systems/terrain_features.rs` (material properties)

### RENDER-007: Elevation-Aware Adjacency Lines & Movement Paths
- **Status**: ✅ Implemented
- **Priority**: Medium
- **Description**: Adjacency lines follow terrain elevation (endpoints at each province's Y + 0.02). Movement path dashes interpolate elevation between source/destination. Arrowheads at destination elevation.
- **Files**: `crates/crown-ash-client/src/systems/map_render.rs` (build_adjacency_lines_mesh, build_path_mesh)

### RENDER-008: Siege & Battle Visual Indicators
- **Status**: ✅ Implemented
- **Priority**: High
- **Description**: Besieged provinces pulse red (sinusoidal). High-unrest provinces desaturate toward grey. Floating crossed-sword markers above besieged provinces with bobbing animation and emissive red glow.
- **Files**: `crates/crown-ash-client/src/systems/map_render.rs` (SiegeMarker, update_siege_markers, animate_siege_markers)

### RENDER-009: Province Hover Tooltip & Highlight
- **Status**: ✅ Implemented
- **Priority**: High
- **Description**: Ray-cast hover detection writes hovered province to Selection. egui tooltip shows province name, terrain, controller, pop, fort, prosperity, unrest, improvements, garrison. Hovered hex gets subtle brightness boost. Tooltip hidden when province is already selected.
- **Files**: `crates/crown-ash-client/src/systems/map_render.rs` (update_hover), `crates/crown-ash-client/src/systems/ui_panels.rs` (hover_tooltip), `crates/crown-ash-client/src/resources/selection.rs`

### RENDER-010: Embedded Assets — Self-Contained Binary
- **Status**: 🔵 In Progress
- **Priority**: Critical
- **Found**: v0.3.4 — users download binary but see blank screen because 27 GLB model files (57MB) require a separate `assets/` folder
- **Root Cause**: Bevy's `AssetServer::load()` reads from filesystem `assets/` directory at runtime. Downloaded binary has no `assets/` folder → all model loads fail silently → blank screen with only procedural terrain meshes visible.
- **Fix**: Use Bevy 0.15's `embedded_asset!` macro to bundle all 27 GLB files into the binary. New `EmbeddedAssetsPlugin` in `src/plugins/embedded_assets.rs` registers every model. Building/deco render systems updated to use `embedded://crown_ash_client/assets/` prefix instead of filesystem paths.
- **Binary size impact**: +57MB (78MB → ~135MB). Acceptable tradeoff for single-file distribution.
- **Files**: `crates/crown-ash-client/src/plugins/embedded_assets.rs` (new), `crates/crown-ash-client/src/plugins/mod.rs`, `crates/crown-ash-client/src/main.rs`, `crates/crown-ash-client/src/systems/building_render.rs` (path prefix changes)

---

## Phase 6 — Performance Optimization

### PERF-001: io_uring Async Storage Engine for Game State
- **Status**: ⚪ Planned
- **Priority**: High
- **Description**: Replace synchronous file I/O in game state persistence (save/load, autosave, replay log) with Linux `io_uring` for zero-copy, kernel-bypassed async I/O. Use `tokio-uring` or `io-uring` crate for submission queue batching — submit multiple save operations (world state + event log + player state) as a single SQE batch instead of sequential `write()` calls. Benefits: eliminate syscall overhead for frequent autosaves (every turn), enable concurrent read/write for spectator replays, reduce fsync latency via `IORING_OP_FSYNC` batching.
  - **Target**: <1ms per autosave (currently ~5-15ms with blocking writes)
  - **Scope**: `crown-ash-api` persistence layer, `crown-ash-sim` replay recording
  - **Fallback**: Feature-gated (`io-uring` feature) — falls back to std::fs on non-Linux / older kernels
- **Files**: `crates/crown-ash-api/src/persistence.rs` (new), `crates/crown-ash-sim/src/replay.rs` (new)

### PERF-002: SIMD-Vectorized Combat & Economy Calculations
- **Status**: ⚪ Planned
- **Priority**: High
- **Description**: Vectorize hot-path simulation loops using portable SIMD (`std::simd` or `packed_simd2`):
  - **Combat resolution**: Batch-process all army damage calculations. 25 provinces × garrison + field armies = ~50-100 combat rolls/tick. Pack army stats (strength, morale, casualties) into `f32x8` SIMD lanes, compute damage/morale in parallel.
  - **Economy tick**: Province resource updates (population growth, tax revenue, prosperity decay, unrest) are 25 independent calculations — perfect for SIMD. Pack 8 provinces per SIMD register, compute `new_pop = pop + growth_rate * pop * prosperity_factor` in 4 SIMD ops instead of 25 scalar ops.
  - **Pathfinding**: BFS adjacency checks use bitwise province masks (`u32` bitmask for 25 provinces). SIMD bitwise ops can evaluate 8 path candidates simultaneously.
  - **Target**: 4-8× throughput for combat/economy ticks
  - **Scope**: `crown-ash-sim` crate, feature-gated (`simd` feature)
- **Files**: `crates/crown-ash-sim/src/combat.rs`, `crates/crown-ash-sim/src/economy.rs`, `crates/crown-ash-sim/src/pathfinding.rs`

### PERF-003: Async Event Pipeline with Lock-Free Channels
- **Status**: ⚪ Planned
- **Priority**: Medium
- **Description**: Replace current synchronous event processing with a lock-free MPSC pipeline using `crossbeam-channel` or `flume`. Architecture:
  ```
  SimTick → [lock-free channel] → NarrativeEngine (parallel)
                                → SSE Broadcaster (parallel)
                                → ReplayLogger (parallel)
                                → StatAggregator (parallel)
  ```
  Current flow processes narrative templates, SSE broadcasts, and logging sequentially after each tick. With lock-free fan-out, the sim can start the next tick immediately while consumers drain events in parallel. Benefits: decouple tick rate from narrative generation latency (especially when LLM Tier 2/3 generates text).
  - **Target**: Tick latency decoupled from broadcast latency
  - **Scope**: `crown-ash-api` event pipeline
- **Files**: `crates/crown-ash-api/src/events.rs`, `crates/crown-ash-api/src/handlers.rs`

### PERF-004: Memory-Mapped Game State with `mmap` + Copy-on-Write
- **Status**: ⚪ Planned
- **Priority**: Medium
- **Description**: Use `mmap` (via `memmap2` crate) for game state snapshots instead of serde serialization. Memory-map the `GameWorld` struct as a flat binary layout. Benefits: zero-copy snapshots for spectators (just `mmap` the file), instant save (kernel handles writeback), copy-on-write forking for "what-if" AI analysis branches. The sim can `fork()` the mmap'd state to evaluate hypothetical moves without cloning 25 province structs.
  - **Target**: Zero-copy state reads, <100μs snapshot creation
  - **Requirements**: Fixed-size structs (no `Vec`/`String` in mmap'd region — use arena allocators)
  - **Scope**: `crown-ash-sim` state management, `crown-ash-api` spectator endpoints
- **Files**: `crates/crown-ash-sim/src/world_state.rs`, `crates/crown-ash-api/src/spectator.rs` (new)

### PERF-005: Batched RocksDB Writes for Turn Persistence
- **Status**: ⚪ Planned
- **Priority**: Medium
- **Description**: When persisting turn results to the blockchain (on-chain game state), batch all province/army/character updates into a single RocksDB `WriteBatch` instead of individual `put()` calls. Each turn generates ~50-100 key updates (25 provinces + armies + characters + events). WriteBatch atomicity also provides crash-safe turn commits — either the full turn persists or nothing does.
  - **Target**: 1 fsync per turn instead of ~100
  - **Scope**: `crown-ash-api` blockchain persistence layer
- **Files**: `crates/crown-ash-api/src/persistence.rs`

### PERF-006: Bevy ECS Query Optimization — Archetypal Batching
- **Status**: ⚪ Planned
- **Priority**: Low
- **Description**: Profile and optimize Bevy ECS query patterns in render systems. Current `update_buildings()` iterates all `BuildingMarker` entities per province change — O(provinces × buildings). Restructure to use Bevy's `Changed<T>` filter and archetype-based batching:
  - Use `Changed<ClientGameState>` to skip frames with no world update
  - Split building entities into per-province archetypes for O(1) province lookup
  - Use `ParallelCommands` for bulk entity spawn/despawn
  - Batch material updates using `Assets<StandardMaterial>` direct mutation instead of per-entity material swaps
  - **Target**: <0.5ms per frame for building updates (currently ~2ms with full iteration)
- **Files**: `crates/crown-ash-client/src/systems/building_render.rs`, `crates/crown-ash-client/src/systems/map_render.rs`

### PERF-007: Kernel-Level Network I/O with io_uring for SSE
- **Status**: ⚪ Planned
- **Priority**: Medium
- **Description**: Replace tokio's epoll-based SSE stream handling with `io_uring`-backed socket I/O for the server-side SSE broadcaster. Benefits:
  - **Batched sends**: Submit all connected client SSE writes as a single SQE batch (one syscall for N clients instead of N syscalls)
  - **Zero-copy**: Use `IORING_OP_SEND_ZC` to send SSE event buffers directly from userspace without kernel copy
  - **Fixed buffers**: Pre-register SSE output buffers with `io_uring_register_buffers()` for kernel-pinned pages
  - **Target**: Handle 1000+ concurrent SSE clients with <1ms broadcast latency
  - **Requires**: Linux 5.19+ (for SEND_ZC), feature-gated
- **Files**: `crates/crown-ash-api/src/streaming.rs`

---

## Phase 7 — MCP Server Enhancement (Full AI Game Control)

### MCP-001: Enhanced World Observation Tools
- **Status**: ⚪ Planned
- **Priority**: High
- **Description**: Expand the existing 24-tool MCP server with deeper observation capabilities:
  - `get_province_history` — full narrative history for a province (sieges, conquests, trade, religion)
  - `get_character_biography` — complete character bio with relationships, chronicle, traits
  - `get_battle_reports` — detailed battle reports for recent/all conflicts
  - `get_turn_events` — raw event list for a specific turn range
  - `get_game_timeline` — era summaries + key events across all turns
  - `get_map_state` — full province ownership map with faction colors, terrain, improvements
- **Files**: `tools/crown-ash-mcp/src/index.ts`

### MCP-002: Advanced Military Strategy Tools
- **Status**: ⚪ Planned
- **Priority**: High
- **Description**: Add deep military control for AI agent gameplay:
  - `plan_campaign` — multi-turn invasion planner: target province → pathfind → estimate army needs → sequence of raise/move/attack orders
  - `evaluate_battle_odds` — pre-battle calculator: our army vs defender garrison + terrain + fortification → win probability + estimated casualties
  - `coordinate_armies` — multi-army pincer: send 2+ armies to converge on a target simultaneously
  - `set_army_stance` — defensive/aggressive/raiding stance affecting AI behavior
  - `request_reinforcements` — raise troops in rear provinces and queue movement to frontline
  - `retreat_army` — pull back to nearest friendly province
- **Files**: `tools/crown-ash-mcp/src/index.ts`

### MCP-003: Diplomacy & Espionage AI Tools
- **Status**: ⚪ Planned
- **Priority**: High
- **Description**: Full diplomatic AI capabilities:
  - `evaluate_alliance_value` — score potential allies by: shared enemies, border proximity, military strength, trustworthiness (treaty history)
  - `propose_coalition` — form defensive coalition against dominant faction
  - `launch_spy_network` — coordinate multiple intrigue plots: fabricate claim → assassinate heir → declare war in sequence
  - `evaluate_marriage_candidates` — score marriage targets by alliance value, trait inheritance, faction relations
  - `negotiate_peace` — auto-propose terms based on war exhaustion, casualties, territory changes
  - `break_alliance` — strategically betray an ally when the timing is right
- **Files**: `tools/crown-ash-mcp/src/index.ts`

### MCP-004: Economy & Development AI Tools
- **Status**: ⚪ Planned
- **Priority**: Medium
- **Description**: Economic management for AI agents:
  - `optimize_build_order` — recommend improvement build priority based on province terrain, resources, threats
  - `manage_tax_policy` — auto-adjust tax rates per province: high for stable, low for rebellious
  - `trade_network_analysis` — identify optimal trade route connections for maximum gold income
  - `economic_forecast` — predict income/expenses for next N turns based on current state
  - `emergency_fund_check` — warn if treasury is too low for planned military operations
- **Files**: `tools/crown-ash-mcp/src/index.ts`

### MCP-005: MCP Resources for Real-Time Game State
- **Status**: ⚪ Planned
- **Priority**: High
- **Description**: Add MCP Resources (read-only data streams) alongside tools:
  - `crown-ash://world` — full world snapshot (auto-updates each turn)
  - `crown-ash://my-faction` — detailed own-faction state
  - `crown-ash://events/latest` — latest turn events
  - `crown-ash://map` — ASCII or structured map representation
  - `crown-ash://diplomacy` — current diplomatic relations matrix
  - `crown-ash://leaderboard` — faction ranking by provinces, military, economy
  These resources allow Claude to maintain persistent game awareness without polling tools.
- **Files**: `tools/crown-ash-mcp/src/index.ts`

### MCP-006: Game Session Management
- **Status**: ⚪ Planned
- **Priority**: Medium
- **Description**: Tools for managing the game session itself:
  - `start_new_game` — create a new Crown & Ash game with parameters (map size, AI difficulty, starting era)
  - `save_game` / `load_game` — persistence
  - `set_game_speed` — adjust tick rate (pause, normal, fast, fastest)
  - `get_game_settings` — view current game configuration
  - `spectate_faction` — switch observation to a different faction (with fog of war)
  - `take_turn` — manually advance one turn (when game is paused)
- **Files**: `tools/crown-ash-mcp/src/index.ts`
