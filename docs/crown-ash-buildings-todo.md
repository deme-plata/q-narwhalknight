# Crown & Ash — Buildings & Causal Dynamics TODO

**Status:** Design TODO, not implemented. Implement after v10.11.15 root-cause
fix verifies clean. Tracks the 11-category asset spec from the operator plus
the causal-loop game mechanics that turn buildings into a real systems-sim.

---

## TL;DR

11 asset categories, ~120+ items. Of these:
- **5 categories drive new game mechanics** (Settlement, Military camp, Fortification, Weapons/Armor, Agriculture)
- **6 categories are atmospheric** (Economy props, Heraldry, Interior, Terrain, Destruction, UI scene props)

The mechanical categories need:
- New `Improvement` enum variants (current 12 → ~32 after expansion)
- New per-province stats (food_security, recruit_speed, intelligence_range, plague_resistance, etc.)
- AI build decisions that consider these new dimensions
- Causal feedback loops connecting them

---

## Category 1 — Core Settlement (NEW MECHANICS)

| Asset bundle | Maps to Improvement | Effect |
|---|---|---|
| peasant house A/B/C, longhouse, fenced yard, woodpile, chicken coop | **Hamlet** (new) | +1 pop_growth_rate per turn |
| barn, granary, bakehouse, well, shed | **VillageCore** (new) | +food_storage, +famine_resistance |
| merchant house, tavern, market stall variants | **TownCore** (new) | +gold per pop per turn |
| workshop, blacksmith exterior, warehouse, stable block | **CraftQuarter** (new) | +iron_consumption, unlocks Barracks |
| town gatehouse, storehouse | **TownDefenses** (new, prerequisite for Walls) | +siege_resistance tier 1 |
| small chapel, roadside shrine | **Shrine** (new, smaller than Temple) | +1 clerical_favor, +0 conversion_rate |
| monastery building module, abbey wall module, bell tower, graveyard | **Monastery** (new, between Temple and Cathedral) | +learning, +clerical_favor, +legitimacy |
| keep, curtain wall straight, inner wall, tower round/square, gatehouse, palisade, manor house, barracks exterior, siege supply shed | **CastleTier1–3** (new ladder, replaces single Fortification) | tier 1 +30 def, tier 2 +60, tier 3 +120 |

## Category 2 — Roadside & Economy Props (VISUAL ONLY)

Wagon, handcart, sacks, barrels, crates, baskets, signposts, bridges, toll
booths, dock props, fishing rack, net bundles.

**No new mechanics.** Render when province has Market / Port / TradeRoute
established. Bridges render where two provinces with TradeRoute connect.

## Category 3 — Military Camp Kit (NEW MECHANICS)

| Asset bundle | Maps to | Effect |
|---|---|---|
| command tent, soldier tents, cookfire, war table, stool/bench | **ArmyEncampment** (Army-level, not province-level) | An army that stays in one province for ≥3 turns establishes an encampment. +1 morale recovery per turn while encamped. |
| supply tent, weapon rack, shield rack, baggage wagon, camp fencing, horse tether post, training dummy, stack of spears, arrow bundle crate, drum/signal horn | **EncampmentSupplies** | +1 troop_recruit_speed while encamped |
| banner pole | (visual identity) | renders faction sigil at the camp |

**New mechanic: Encampments.** Armies have a `posture` field (Marching /
Encamped / Sieging). Encampment unlocks recruit + morale recovery; Marching
costs morale; Sieging costs morale but applies siege damage to garrison.

## Category 4 — Fortification Kit (NEW MECHANICS — ladder)

Already covered by CastleTier1–3 above plus these as separate variants:

| Tier | Assets | Building cost | Defense bonus |
|---|---|---|---|
| **Palisade** | palisade straight/corner/gate, sharpened stakes, watchtower wood | low (timber only) | +20 def |
| **Watchtower** | watchtower stone, crenellation wall cap, ladder | medium (timber + stone) | +40 def, +1 intelligence_range |
| **Stone fortification** | siege mantlet, barricade, chevaux-de-frise, fortified door, portcullis | high (stone + iron) | +80 def |
| **Damaged / rubble** | rubble pile, damaged wall section | (post-siege state) | -50% def until repaired |

## Category 5 — Weapons & Armor (NEW MECHANICS — army composition)

Currently `Army { troops: u32 }`. Expand to:

```rust
pub struct Army {
    pub footmen: u32,        // sword + shield (arming sword, heater shield)
    pub spearmen: u32,       // pike, spear
    pub cavalry: u32,        // lance, kite shield, mail
    pub archers: u32,        // longbow, crossbow, quiver
    pub knights: u32,        // longsword/warhammer + breastplate + gauntlets
}
```

Each composition has rock-paper-scissors against the others:
- Spear > Cavalry (anti-cav formation)
- Cavalry > Archers (run them down)
- Archers > Footmen (range advantage)
- Knights = elite anti-everything but expensive
- Footmen = generic

Recruiting requires Barracks (footmen/spearmen), Stables (cavalry/knights),
Mine + Forge (weapons), Lumbercamp (bows/spears).

## Category 6 — Heraldry (VISUAL — but ties to NFT system)

Per-faction sigil + banner cloth variants → on-chain `sigil_nft.rs`
crate stub. Each season mints limited-edition NFTs of faction sigils.

## Category 7 — Interior Dressing (EVENT SCENES ONLY)

Throne, benches, tables, writing desk, candle stand, chest, bookshelf, map
table, wall tapestry, cup/goblet, plate/bowl, brazier, bed, wardrobe.

Render in **modal scenes** for diplomacy, treaty signing, character
death, marriage ceremonies. No game mechanic.

## Category 8 — Agriculture & Rural (NEW MECHANICS)

| Asset bundle | Maps to | Effect |
|---|---|---|
| plow, wheelbarrow, hay bale, grain cart, scarecrow, split-rail fence, stone wall segment, crop field markers | **FarmTier1 / FarmTier2** | Tier 1: +5 food. Tier 2: +12 food, requires Plow tech |
| orchard tree guards, beehive skeps | **Orchard** | +2 luxury goods (trade premium) |
| milling props, water wheel parts, flour bins | **WaterMill** (replaces Mill in current sim) | +3 grain → +5 food when on river-adjacent province |

## Category 9 — Terrain Dressing (VISUAL ONLY)

Trees, stumps, dead trees, bushes, rocks small/medium/large, cliff debris,
log piles, broken carts, ruined columns/walls, puddles, snow-covered
variants, camp debris.

Pure atmosphere. Snow variants render based on calendar season; ruins
render where a province had a building destroyed in war.

## Category 10 — Destruction & Aftermath (VISUAL — narrative)

Burned cart, broken barrel, shattered crate, collapsed roof, scorched
beam, ruined banner, broken wheel, abandoned shield, battlefield stake,
tomb marker, rubble heap, damaged chapel.

Render in province scene for N turns after a battle (where N scales
with battle severity). Important storytelling element — gives the
post-battle province visual weight without changing mechanics.

## Category 11 — UI / 3D Scene Props (EVENT SCENES)

Crown, royal seal, signet ring, parchment, wax seal, reliquary,
ceremonial sword, orb/regalia, chalice, incense burner.

Render in **coronation scene** (when an heir succeeds), **excommunication
scene** (religious conflict event), **treaty signing scene** (when AcceptTreaty
fires). High-emotional-impact moments, no mechanic.

---

# CAUSAL DYNAMICS — the engine room

This is what makes buildings a game, not decoration. Five feedback loops
that the operator's example ("houses → peasants → pop") expands into.

## Loop 1 — Settlement Growth (peasants → pop → tax → treasury → buildings → peasants)

```
  Hamlet built
      ↓
  +1 pop_growth_rate
      ↓
  Population N → N + growth_rate per turn
      ↓
  Tax revenue scales with pop  (tax_rate × pop × prosperity / 1M)
      ↓
  Treasury grows
      ↓
  Build threshold met → AI builds another Hamlet (or VillageCore)
      ↓
  Loop closes
```

Diminishing returns: each Hamlet costs more than the last; pop caps at
province carrying capacity (food-dependent — see Loop 2).

## Loop 2 — Food Security (food → no famine → pop → more food demand → ...)

```
  Farm built  →  +food
       ↑              ↓
       │      food >= pop * 1.0  →  prosperity rises
       │              ↓
       │      pop grows (Loop 1)
       │              ↓
       │      food demand rises
       │              ↓
  AI senses food < pop * 1.1 → builds another Farm or Granary
       │              ↓
       │      food < pop * 0.8  →  FAMINE event  →  -10% pop
       │              ↓
       │      pop drops to where food suffices
       │              ↓
       └──── recovery loop (Granary halves the famine impact)
```

Granary doesn't add food — it BUFFERS against the famine penalty.
Hospital does the same for plague events.

## Loop 3 — Military Strength (industry → weapons → army → conquest → tax → industry)

```
  Mine + Forge → iron production
        ↓
  Barracks consumes iron → recruits footmen/knights
        ↓
  Stables consumes horses → recruits cavalry
        ↓
  Lumbercamp produces timber → arrows/bows/palisades
        ↓
  Army composition emerges
        ↓
  DeclareWar → battle outcome shaped by composition + terrain + Fort
        ↓
  Conquered province → +tax base
        ↓
  Loop closes — more iron buy more troops
```

Counter-loop: continuous WarStarted state degrades prosperity (per
existing economy.rs: WAR_PROSPERITY_LOSS), which decays the tax base.
War can't run forever without exhausting the realm — this is the
self-correcting check.

## Loop 4 — Religious Legitimacy (Temple → favor → action → legitimacy → succession)

```
  Temple built → +clerical_favor per turn
       ↓
  Above threshold → unlocks ConvertProvince + ArrangeMarriage actions
       ↓
  Successful conversions/marriages → +legitimacy
       ↓
  High legitimacy → smoother heir succession (less rebellion risk)
       ↓
  Smooth succession → stable realm → continuing favor accumulation
```

Counter-loop: declaring HolyWar gives a one-shot legitimacy spike but
costs ongoing clerical_favor — must keep building Temples/Monasteries
to refill.

## Loop 5 — Knowledge / Tech (University → learning → tech unlocks → new buildings)

```
  University built → +learning per turn
       ↓
  Library + Monastery boost learning
       ↓
  Above threshold → unlock next tier
       ↓
  Stonemasonry unlocks → can build Cathedral, Stone Walls, Aqueducts
  Iron Casting unlocks → can build Forge, recruit Knights
  Heavy Cavalry doctrine → unlocks Stables tier 2
       ↓
  New buildings open up Loops 1-4 with bigger numbers
```

This is the long-arc loop. Visible only over hundreds of turns.

## Cross-loop coupling — emergent strategy

The interesting gameplay emerges from agents balancing all 5:
- **Pure militarist**: Loops 3+1, ignores 4-5. Conquers fast, collapses
  faster from war exhaustion.
- **Trader-cleric**: Loops 1+2+4, weak military. Survives by marriage
  and conversion; gets steamrolled by Loop 3 specialists eventually.
- **Tech rusher**: Loops 5+1+2. Slow start, dominant by turn 5000.
- **Balanced**: Loops 1+2+3+4+5 each progressing slowly. Hard to play
  well but very robust.

A smart AI (Codex GPT-5.5, Claude 5+, Grok 5+) should find these niches
naturally and adapt as the game state shifts. A dumb AI builds whatever
the treasury affords.

---

## Implementation phasing (post-v10.11.15)

### Phase 1 (1 week) — Category 1 expansion
- Add Improvement variants: Hamlet, VillageCore, TownCore, CraftQuarter,
  Shrine, Monastery, CastleTier1/2/3
- Sim handlers for each (food/gold/legitimacy effects)
- AI build decisions extended (currently builds when treasury > X; now
  picks BEST building for province conditions)

### Phase 2 (1 week) — Category 8 + Loop 2
- FarmTier1/2, Orchard, WaterMill variants
- Province-level `food_balance` tracking
- Famine event resolver
- AI prioritizes Granary when food <= pop * 1.1

### Phase 3 (2 weeks) — Categories 3-5 + Loop 3
- Army composition (5 troop types)
- Barracks, Stables, Forge prerequisites
- Encampment posture system
- Combat resolver respects composition rock-paper-scissors

### Phase 4 (1 week) — Loops 4-5
- Tech tree (Stonemasonry, IronCasting, HeavyCavalry, etc.)
- University/Library/Monastery learning accumulation
- Tech-gated building unlocks

### Phase 5 (1 week) — Visual integration
- Wire categories 2, 6, 7, 9, 10, 11 into the frontend renderer
- No new mechanics, just better atmosphere

Total: ~6 weeks of focused work for a full implementation. Realistic
ship cadence with the v10.11.15 + LP work running parallel.

---

## What this is NOT

- **Not a Crusader Kings clone.** Same genre, smaller scope, more agent-
  centric. The goal is for AI agents to play through it autonomously,
  not for humans to spend 100 hours per playthrough.
- **Not a building-placement minigame.** AI picks improvement type; the
  game tells the agent "your faction now has Hamlet/Castle/Cathedral"
  but doesn't ask them to choose tile coordinates.
- **Not graphics-first.** Atmosphere matters but the SIM is the product.
  Visual integration (categories 2, 6, 7, 9, 10, 11) ships last because
  they don't change game behavior.

---

## What this enables for LP / economic mechanics

Each new building variant is a potential **action-tax sink** under the
Path A LP revenue-share (`docs/crown-ash-lp-revenue-share-v1.md`).
Today's schedule has BuildImprovement at 2× MIN_FEE. With 32 variants
of varying cost, the tax can be tiered:
- Hamlet: 1× (cheap, common)
- Cathedral: 8× (expensive, rare)
- Castle Tier 3: 12× (very expensive)

This gives LP holders MORE inflow during construction-heavy seasons
(early-game, post-war rebuilding) and naturally throttles spam (you
can't spam 100 Cathedrals).

---

Operator decision after v10.11.15:
1. Should this expand into a separate `q-crown-ash-buildings` crate? Probably.
2. Should each tier ship as a separate release, or all 32 variants in v11.0?
3. Should some buildings be exclusive (Cathedral OR Castle, not both, per province)?

Defer all three until v10.11.15 verifies clean and the operator has time.
