#!/usr/bin/env node
/**
 * Crown & Ash — MCP Server for AI Game Agent
 *
 * Provides tools for Claude Code to observe the game world, analyze strategic
 * situations, and submit actions to play Crown & Ash like a pro.
 *
 * API base: https://quillon.xyz/api/v1/crown-ash
 */

import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { z } from "zod";

// ─── Config ───────────────────────────────────────────────────────────────────

const BASE_URL = process.env.CROWN_ASH_URL || "https://quillon.xyz";
const API = `${BASE_URL}/api/v1/crown-ash`;
const WALLET = process.env.CROWN_ASH_WALLET || "";

// ─── HTTP helpers ─────────────────────────────────────────────────────────────

async function apiGet<T = any>(path: string): Promise<T> {
  const resp = await fetch(`${API}${path}`);
  if (!resp.ok) throw new Error(`GET ${path}: HTTP ${resp.status}`);
  const json = (await resp.json()) as any;
  if (json.success === false) throw new Error(json.error || "API error");
  return json.data ?? json;
}

async function apiPost<T = any>(path: string, body: unknown): Promise<T> {
  const resp = await fetch(`${API}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!resp.ok) {
    const text = await resp.text();
    throw new Error(`POST ${path}: HTTP ${resp.status} — ${text}`);
  }
  const json = (await resp.json()) as any;
  if (json.success === false) throw new Error(json.error || "API error");
  return json.data ?? json;
}

function requireWallet(): string {
  if (!WALLET) throw new Error("CROWN_ASH_WALLET not set — set it in env");
  return WALLET;
}

// ─── World state cache (refreshed per-call, avoids redundant fetches) ─────

let cachedWorld: any = null;
let cacheTime = 0;
const CACHE_TTL = 3000; // 3 seconds

async function getWorld(): Promise<any> {
  const now = Date.now();
  if (cachedWorld && now - cacheTime < CACHE_TTL) return cachedWorld;
  cachedWorld = await apiGet("/world");
  cacheTime = now;
  return cachedWorld;
}

function clearCache() {
  cachedWorld = null;
  cacheTime = 0;
}

// ─── Analysis helpers ─────────────────────────────────────────────────────────

function fp(val: any): number {
  if (typeof val === "number") return val / 1000;
  if (typeof val === "object" && val?.raw != null) return val.raw / 1000;
  return 0;
}

function armyPower(troops: any): number {
  return (troops.levy || 0) + (troops.men_at_arms || 0) * 3 + (troops.knights || 0) * 10;
}

function findMyFaction(world: any): any {
  const wallet = requireWallet();
  const faction = world.factions.find((f: any) => f.player_wallet === wallet);
  if (!faction) throw new Error(`No faction found for wallet ${wallet}`);
  return faction;
}

function myProvinces(world: any, factionId: number): any[] {
  return world.provinces.filter((p: any) => p.controller === factionId);
}

function myArmies(world: any, factionId: number): any[] {
  return (world.armies || []).filter((a: any) => a.owner_faction === factionId);
}

function myCharacters(world: any, factionId: number): any[] {
  return (world.characters || []).filter(
    (c: any) => c.faction === factionId && c.alive
  );
}

// ─── Action submission helper ─────────────────────────────────────────────────

async function submitAction(action: Record<string, any>): Promise<string> {
  const wallet = requireWallet();
  clearCache();
  const result = await apiPost("/action", { wallet, action });
  return JSON.stringify(result, null, 2);
}

// ─── MCP Server Setup ────────────────────────────────────────────────────────

const server = new McpServer({
  name: "crown-ash",
  version: "2.0.0",
});

// ═══════════════════════════════════════════════════════════════════════════════
// OBSERVATION TOOLS
// ═══════════════════════════════════════════════════════════════════════════════

server.tool(
  "get_world_overview",
  "Get a high-level overview of the entire game world: turn number, all factions (who is alive, who is player-controlled), province count, army count, and active wars. Start here to understand the current game state.",
  {},
  async () => {
    const world = await getWorld();
    const meta = world.meta || {};
    const factions = (world.factions || []).map((f: any) => ({
      id: f.id,
      name: f.name,
      alive: f.alive,
      player: f.player_wallet ? "human" : "AI",
      culture: f.culture,
      religion: f.religion,
      provinces: (world.provinces || []).filter((p: any) => p.controller === f.id).length,
      armies: (world.armies || []).filter((a: any) => a.owner_faction === f.id).length,
    }));
    const wars = (world.diplomacy || []).filter((d: any) => d.at_war);
    return {
      content: [
        {
          type: "text" as const,
          text: JSON.stringify(
            {
              turn: meta.turn,
              total_provinces: (world.provinces || []).length,
              total_armies: (world.armies || []).length,
              total_characters: (world.characters || []).filter((c: any) => c.alive).length,
              factions,
              active_wars: wars.map((w: any) => ({
                factions: [w.faction_a, w.faction_b],
                opinion: fp(w.opinion),
              })),
            },
            null,
            2
          ),
        },
      ],
    };
  }
);

server.tool(
  "get_my_realm",
  "Get detailed info about YOUR faction: provinces with resources/improvements, armies with troops/location, characters with stats/traits, treasury, and current wars. Requires CROWN_ASH_WALLET env var.",
  {},
  async () => {
    const world = await getWorld();
    const faction = findMyFaction(world);
    const provinces = myProvinces(world, faction.id);
    const armies = myArmies(world, faction.id);
    const chars = myCharacters(world, faction.id);

    const totalIncome = provinces.reduce(
      (sum: number, p: any) => sum + fp(p.resources?.gold || 0),
      0
    );

    const result = {
      faction: {
        id: faction.id,
        name: faction.name,
        culture: faction.culture,
        religion: faction.religion,
      },
      summary: {
        province_count: provinces.length,
        army_count: armies.length,
        total_soldiers: armies.reduce(
          (s: number, a: any) => s + (a.troops?.levy || 0) + (a.troops?.men_at_arms || 0) + (a.troops?.knights || 0),
          0
        ),
        total_gold_income: totalIncome.toFixed(1),
        character_count: chars.length,
      },
      provinces: provinces.map((p: any) => ({
        id: p.id,
        name: p.name,
        terrain: p.terrain,
        population: p.population,
        prosperity: fp(p.prosperity).toFixed(2),
        unrest: fp(p.unrest).toFixed(2),
        fortification: p.fortification,
        garrison: p.garrison,
        improvements: p.improvements,
        construction: p.construction_queue,
        tax_rate: fp(p.tax_rate).toFixed(2),
        resources: {
          gold: fp(p.resources?.gold),
          food: fp(p.resources?.food),
          iron: fp(p.resources?.iron),
          timber: fp(p.resources?.timber),
          stone: fp(p.resources?.stone),
          horses: fp(p.resources?.horses),
        },
        neighbors: p.neighbors,
        religion: p.religion,
        culture: p.culture,
      })),
      armies: armies.map((a: any) => ({
        id: a.id,
        location: a.location,
        destination: a.destination,
        movement_queue: a.movement_queue,
        troops: a.troops,
        total_power: armyPower(a.troops),
        morale: fp(a.morale).toFixed(2),
        supply: fp(a.supply).toFixed(2),
        siege: a.siege,
      })),
      characters: chars.map((c: any) => ({
        id: c.id,
        name: c.name,
        role: c.role,
        age: c.age,
        traits: c.traits,
        stats: {
          martial: fp(c.stats?.martial),
          diplomacy: fp(c.stats?.diplomacy),
          stewardship: fp(c.stats?.stewardship),
          intrigue: fp(c.stats?.intrigue),
          learning: fp(c.stats?.learning),
        },
        health: fp(c.health).toFixed(2),
        legitimacy: fp(c.legitimacy).toFixed(2),
        prestige: fp(c.prestige).toFixed(2),
        heir: c.heir,
        spouse: c.spouse,
      })),
    };

    return { content: [{ type: "text" as const, text: JSON.stringify(result, null, 2) }] };
  }
);

server.tool(
  "get_province_details",
  "Get full details of a specific province by ID: resources, garrison, improvements, terrain, neighbors, scars, grudges, conversion progress.",
  { province_id: z.number().int().describe("Province ID (u16)") },
  async ({ province_id }) => {
    const data = await apiGet(`/province/${province_id}`);
    return { content: [{ type: "text" as const, text: JSON.stringify(data, null, 2) }] };
  }
);

server.tool(
  "get_faction_details",
  "Get full details of a specific faction by ID: bonuses, culture, religion, player status.",
  { faction_id: z.number().int().describe("Faction ID (0-6)") },
  async ({ faction_id }) => {
    const data = await apiGet(`/faction/${faction_id}`);
    return { content: [{ type: "text" as const, text: JSON.stringify(data, null, 2) }] };
  }
);

server.tool(
  "get_diplomacy",
  "Get all diplomatic relations: who is at war, treaty statuses, opinion scores, and grievances between all faction pairs.",
  {},
  async () => {
    const world = await getWorld();
    const diplomacy = (world.diplomacy || []).map((d: any) => ({
      factions: [d.faction_a, d.faction_b],
      faction_names: [
        world.factions.find((f: any) => f.id === d.faction_a)?.name,
        world.factions.find((f: any) => f.id === d.faction_b)?.name,
      ],
      opinion: fp(d.opinion).toFixed(1),
      at_war: d.at_war,
      treaties: d.treaties,
      grievances: d.grievances,
    }));
    return { content: [{ type: "text" as const, text: JSON.stringify(diplomacy, null, 2) }] };
  }
);

server.tool(
  "get_all_armies",
  "Get a tactical overview of ALL armies on the map: location, troops, power, movement, siege status. Essential for planning attacks and defense.",
  {},
  async () => {
    const world = await getWorld();
    const armies = (world.armies || []).map((a: any) => {
      const prov = (world.provinces || []).find((p: any) => p.id === a.location);
      const faction = (world.factions || []).find((f: any) => f.id === a.owner_faction);
      return {
        id: a.id,
        owner: faction?.name || `faction-${a.owner_faction}`,
        owner_faction_id: a.owner_faction,
        location: a.location,
        location_name: prov?.name || "unknown",
        troops: a.troops,
        power: armyPower(a.troops),
        morale: fp(a.morale).toFixed(2),
        moving_to: a.destination,
        movement_queue: a.movement_queue,
        siege: a.siege,
      };
    });
    return { content: [{ type: "text" as const, text: JSON.stringify(armies, null, 2) }] };
  }
);

// ═══════════════════════════════════════════════════════════════════════════════
// STRATEGIC ANALYSIS TOOLS
// ═══════════════════════════════════════════════════════════════════════════════

server.tool(
  "analyze_threats",
  "Analyze military threats to your realm. Shows enemy armies near your borders, undefended provinces, and provinces under siege. Use this before deciding where to move your armies.",
  {},
  async () => {
    const world = await getWorld();
    const faction = findMyFaction(world);
    const provinces = myProvinces(world, faction.id);
    const myProvIds = new Set(provinces.map((p: any) => p.id));

    // Find border provinces (have neighbors controlled by other factions)
    const borderProvinces = provinces.filter((p: any) =>
      (p.neighbors || []).some(
        (n: number) =>
          !(world.provinces || []).find((pp: any) => pp.id === n && pp.controller === faction.id)
      )
    );

    // Find enemy armies near my borders
    const enemyArmies = (world.armies || [])
      .filter((a: any) => a.owner_faction !== faction.id)
      .filter((a: any) => {
        // Army is in my province, or in a neighbor of my province
        if (myProvIds.has(a.location)) return true;
        return provinces.some((p: any) => (p.neighbors || []).includes(a.location));
      })
      .map((a: any) => {
        const prov = (world.provinces || []).find((p: any) => p.id === a.location);
        const ownerFaction = (world.factions || []).find(
          (f: any) => f.id === a.owner_faction
        );
        return {
          army_id: a.id,
          owner: ownerFaction?.name,
          location: a.location,
          location_name: prov?.name,
          in_my_territory: myProvIds.has(a.location),
          troops: a.troops,
          power: armyPower(a.troops),
          moving_to: a.destination || a.movement_queue?.[0],
        };
      });

    // Undefended provinces (no garrison, no army present)
    const myArmyLocations = new Set(
      myArmies(world, faction.id).map((a: any) => a.location)
    );
    const undefended = provinces
      .filter(
        (p: any) =>
          (p.garrison?.levy || 0) + (p.garrison?.men_at_arms || 0) === 0 &&
          !myArmyLocations.has(p.id)
      )
      .map((p: any) => ({ id: p.id, name: p.name, fortification: p.fortification }));

    // Provinces under siege
    const sieges = (world.armies || [])
      .filter((a: any) => a.siege && a.owner_faction !== faction.id)
      .filter((a: any) => myProvIds.has(a.siege.target_province))
      .map((a: any) => ({
        province: a.siege.target_province,
        province_name: (world.provinces || []).find(
          (p: any) => p.id === a.siege.target_province
        )?.name,
        attacker_army: a.id,
        attacker_faction: (world.factions || []).find(
          (f: any) => f.id === a.owner_faction
        )?.name,
        turns_besieged: a.siege.turns_besieged,
        turns_required: a.siege.turns_required,
      }));

    return {
      content: [
        {
          type: "text" as const,
          text: JSON.stringify(
            {
              threat_level:
                enemyArmies.length > 2
                  ? "HIGH"
                  : enemyArmies.length > 0
                  ? "MODERATE"
                  : "LOW",
              enemy_armies_near_border: enemyArmies,
              undefended_provinces: undefended,
              provinces_under_siege: sieges,
              border_provinces: borderProvinces.map((p: any) => ({
                id: p.id,
                name: p.name,
                garrison: p.garrison,
                fortification: p.fortification,
              })),
            },
            null,
            2
          ),
        },
      ],
    };
  }
);

server.tool(
  "analyze_economy",
  "Analyze your economic situation: total income per resource, tax rates, improvements, construction in progress, and recommendations for economic growth.",
  {},
  async () => {
    const world = await getWorld();
    const faction = findMyFaction(world);
    const provinces = myProvinces(world, faction.id);

    const totals = {
      gold: 0, food: 0, iron: 0, timber: 0, stone: 0, horses: 0, trade_goods: 0,
    };
    for (const p of provinces) {
      const r = p.resources || {};
      totals.gold += fp(r.gold);
      totals.food += fp(r.food);
      totals.iron += fp(r.iron);
      totals.timber += fp(r.timber);
      totals.stone += fp(r.stone);
      totals.horses += fp(r.horses);
      totals.trade_goods += fp(r.trade_goods);
    }

    const avgProsperity =
      provinces.reduce((s: number, p: any) => s + fp(p.prosperity), 0) /
      (provinces.length || 1);

    const highUnrest = provinces
      .filter((p: any) => fp(p.unrest) > 0.3)
      .map((p: any) => ({
        id: p.id,
        name: p.name,
        unrest: fp(p.unrest).toFixed(2),
        tax_rate: fp(p.tax_rate).toFixed(2),
      }));

    const constructing = provinces
      .filter((p: any) => (p.construction_queue || []).length > 0)
      .map((p: any) => ({
        province: p.name,
        building: p.construction_queue.map((c: any) => ({
          improvement: c[0],
          turns_left: c[1],
        })),
      }));

    // Provinces without key improvements
    const needFarmstead = provinces
      .filter((p: any) => !(p.improvements || []).includes("Farmstead"))
      .map((p: any) => p.name);

    const needMarket = provinces
      .filter((p: any) => !(p.improvements || []).includes("Market"))
      .map((p: any) => p.name);

    return {
      content: [
        {
          type: "text" as const,
          text: JSON.stringify(
            {
              total_income: totals,
              average_prosperity: avgProsperity.toFixed(2),
              total_population: provinces.reduce(
                (s: number, p: any) => s + (p.population || 0),
                0
              ),
              high_unrest_provinces: highUnrest,
              under_construction: constructing,
              provinces_without_farmstead: needFarmstead.slice(0, 10),
              provinces_without_market: needMarket.slice(0, 10),
            },
            null,
            2
          ),
        },
      ],
    };
  }
);

server.tool(
  "find_weak_targets",
  "Find weakly defended enemy provinces adjacent to your territory — prime targets for conquest. Ranks by ease of capture (low garrison, low fortification, no defending army).",
  {},
  async () => {
    const world = await getWorld();
    const faction = findMyFaction(world);
    const provinces = myProvinces(world, faction.id);
    const myProvIds = new Set(provinces.map((p: any) => p.id));

    // Find enemy provinces adjacent to my territory
    const neighborIds = new Set<number>();
    for (const p of provinces) {
      for (const n of p.neighbors || []) {
        if (!myProvIds.has(n)) neighborIds.add(n);
      }
    }

    const enemyArmyLocations = new Map<number, number>();
    for (const a of world.armies || []) {
      if (a.owner_faction !== faction.id) {
        const existing = enemyArmyLocations.get(a.location) || 0;
        enemyArmyLocations.set(a.location, existing + armyPower(a.troops));
      }
    }

    const targets = [...neighborIds]
      .map((id) => {
        const prov = (world.provinces || []).find((p: any) => p.id === id);
        if (!prov) return null;
        const owner = (world.factions || []).find(
          (f: any) => f.id === prov.controller
        );
        const garrisonPower = armyPower(prov.garrison || { levy: 0, men_at_arms: 0, knights: 0 });
        const armyPowerPresent = enemyArmyLocations.get(id) || 0;
        const totalDefense = garrisonPower + armyPowerPresent + (prov.fortification || 0) * 5;

        return {
          province_id: id,
          name: prov.name,
          owner: owner?.name,
          owner_faction_id: prov.controller,
          terrain: prov.terrain,
          garrison: prov.garrison,
          garrison_power: garrisonPower,
          enemy_army_power: armyPowerPresent,
          fortification: prov.fortification,
          total_defense_score: totalDefense,
          population: prov.population,
          prosperity: fp(prov.prosperity).toFixed(2),
          improvements: prov.improvements,
        };
      })
      .filter(Boolean)
      .sort((a: any, b: any) => a.total_defense_score - b.total_defense_score);

    return {
      content: [
        {
          type: "text" as const,
          text: JSON.stringify(
            {
              weak_targets: targets.slice(0, 15),
              note: "Sorted by total_defense_score (lowest = easiest to capture). Consider declaring war first if not already at war.",
            },
            null,
            2
          ),
        },
      ],
    };
  }
);

server.tool(
  "analyze_characters",
  "Analyze your characters: who should be your heir, who should fill council roles, who has the best stats for each position, and which characters are potential plotters.",
  {},
  async () => {
    const world = await getWorld();
    const faction = findMyFaction(world);
    const chars = myCharacters(world, faction.id);

    const ruler = chars.find((c: any) => c.role === "Ruler");
    const council = chars.filter((c: any) =>
      ["Marshal", "Chaplain", "Steward", "Spymaster"].includes(c.role)
    );
    const unassigned = chars.filter(
      (c: any) => c.role === "Courtier" || c.role === "Duke"
    );

    // Best candidate for each role
    const bestFor = (stat: string) =>
      [...chars]
        .filter((c: any) => c.role !== "Ruler")
        .sort((a: any, b: any) => fp(b.stats?.[stat]) - fp(a.stats?.[stat]))
        .slice(0, 3)
        .map((c: any) => ({
          id: c.id,
          name: c.name,
          current_role: c.role,
          stat_value: fp(c.stats?.[stat]).toFixed(1),
          traits: c.traits,
        }));

    // Heir candidates (alive, adult or close to adult)
    const heirCandidates = chars
      .filter((c: any) => c.role !== "Ruler" && c.age >= 12)
      .sort(
        (a: any, b: any) =>
          fp(b.legitimacy) + fp(b.prestige) - (fp(a.legitimacy) + fp(a.prestige))
      )
      .slice(0, 5)
      .map((c: any) => ({
        id: c.id,
        name: c.name,
        age: c.age,
        role: c.role,
        legitimacy: fp(c.legitimacy).toFixed(1),
        prestige: fp(c.prestige).toFixed(1),
        traits: c.traits,
      }));

    return {
      content: [
        {
          type: "text" as const,
          text: JSON.stringify(
            {
              ruler: ruler
                ? {
                    id: ruler.id,
                    name: ruler.name,
                    age: ruler.age,
                    health: fp(ruler.health).toFixed(2),
                    legitimacy: fp(ruler.legitimacy).toFixed(2),
                    current_heir: ruler.heir,
                    traits: ruler.traits,
                  }
                : null,
              current_council: council.map((c: any) => ({
                role: c.role,
                id: c.id,
                name: c.name,
                key_stat: fp(
                  c.stats?.[
                    c.role === "Marshal"
                      ? "martial"
                      : c.role === "Steward"
                      ? "stewardship"
                      : c.role === "Spymaster"
                      ? "intrigue"
                      : "learning"
                  ]
                ).toFixed(1),
              })),
              best_marshal_candidates: bestFor("martial"),
              best_steward_candidates: bestFor("stewardship"),
              best_spymaster_candidates: bestFor("intrigue"),
              best_chaplain_candidates: bestFor("learning"),
              heir_candidates: heirCandidates,
              unassigned_courtiers: unassigned.map((c: any) => ({
                id: c.id,
                name: c.name,
                age: c.age,
                traits: c.traits,
              })),
            },
            null,
            2
          ),
        },
      ],
    };
  }
);

// ═══════════════════════════════════════════════════════════════════════════════
// ACTION TOOLS — MILITARY
// ═══════════════════════════════════════════════════════════════════════════════

server.tool(
  "raise_army",
  "Raise a new army from a province's population. The province must be under your control.",
  { province_id: z.number().int().describe("Province ID to raise army from") },
  async ({ province_id }) => {
    const result = await submitAction({ RaiseArmy: { province: province_id } });
    return { content: [{ type: "text" as const, text: `Army raised from province ${province_id}.\n${result}` }] };
  }
);

server.tool(
  "move_army",
  "Move an army to a target province using the shortest BFS path. The army moves 1 province per turn. Use this for multi-hop pathfinding.",
  {
    army_id: z.number().int().describe("Army ID to move"),
    target_province: z.number().int().describe("Target province ID"),
  },
  async ({ army_id, target_province }) => {
    const result = await submitAction({
      MoveArmyPath: { army: army_id, target: target_province },
    });
    return {
      content: [
        {
          type: "text" as const,
          text: `Army ${army_id} ordered to march to province ${target_province}.\n${result}`,
        },
      ],
    };
  }
);

server.tool(
  "disband_army",
  "Disband an army, returning troops to the province garrison.",
  { army_id: z.number().int().describe("Army ID to disband") },
  async ({ army_id }) => {
    const result = await submitAction({ DisbandArmy: { army: army_id } });
    return { content: [{ type: "text" as const, text: `Army ${army_id} disbanded.\n${result}` }] };
  }
);

// ═══════════════════════════════════════════════════════════════════════════════
// ACTION TOOLS — DIPLOMACY
// ═══════════════════════════════════════════════════════════════════════════════

server.tool(
  "declare_war",
  "Declare war on another faction. You need a casus belli (reason for war). Options: Conquest, HolyWar, Reconquest, Rebellion, Succession, Insult.",
  {
    target_faction: z.number().int().describe("Target faction ID (0-6)"),
    casus_belli: z
      .enum(["Conquest", "HolyWar", "Reconquest", "Rebellion", "Succession", "Insult"])
      .describe("Reason for war"),
  },
  async ({ target_faction, casus_belli }) => {
    const result = await submitAction({
      DeclareWar: { target: target_faction, casus_belli },
    });
    return {
      content: [
        {
          type: "text" as const,
          text: `War declared on faction ${target_faction} (${casus_belli})!\n${result}`,
        },
      ],
    };
  }
);

server.tool(
  "propose_treaty",
  "Propose a treaty to another faction. Types: NonAggression, DefensiveAlliance, TradeAgreement, Marriage, Vassalization, WhitePeace, Surrender.",
  {
    target_faction: z.number().int().describe("Target faction ID"),
    treaty_type: z
      .enum([
        "NonAggression",
        "DefensiveAlliance",
        "TradeAgreement",
        "Marriage",
        "Vassalization",
        "WhitePeace",
        "Surrender",
      ])
      .describe("Type of treaty"),
  },
  async ({ target_faction, treaty_type }) => {
    const result = await submitAction({
      ProposeTreaty: { target: target_faction, treaty: treaty_type },
    });
    return {
      content: [
        {
          type: "text" as const,
          text: `Treaty proposed (${treaty_type}) to faction ${target_faction}.\n${result}`,
        },
      ],
    };
  }
);

server.tool(
  "accept_treaty",
  "Accept a treaty proposed by another faction.",
  {
    from_faction: z.number().int().describe("Faction ID that proposed the treaty"),
    treaty_type: z
      .enum([
        "NonAggression",
        "DefensiveAlliance",
        "TradeAgreement",
        "Marriage",
        "Vassalization",
        "WhitePeace",
        "Surrender",
      ])
      .describe("Type of treaty to accept"),
  },
  async ({ from_faction, treaty_type }) => {
    const result = await submitAction({
      AcceptTreaty: { from: from_faction, treaty: treaty_type },
    });
    return {
      content: [
        {
          type: "text" as const,
          text: `Treaty accepted (${treaty_type}) from faction ${from_faction}.\n${result}`,
        },
      ],
    };
  }
);

// ═══════════════════════════════════════════════════════════════════════════════
// ACTION TOOLS — ECONOMY
// ═══════════════════════════════════════════════════════════════════════════════

server.tool(
  "build_improvement",
  "Build an improvement in a province. Options: Farmstead, Mine, Lumbercamp, Quarry, Stables, Market, Temple, Fortification, University, Port, Granary, Hospital.",
  {
    province_id: z.number().int().describe("Province ID"),
    improvement: z
      .enum([
        "Farmstead",
        "Mine",
        "Lumbercamp",
        "Quarry",
        "Stables",
        "Market",
        "Temple",
        "Fortification",
        "University",
        "Port",
        "Granary",
        "Hospital",
      ])
      .describe("Improvement to build"),
  },
  async ({ province_id, improvement }) => {
    const result = await submitAction({
      BuildImprovement: { province: province_id, improvement },
    });
    return {
      content: [
        {
          type: "text" as const,
          text: `Building ${improvement} in province ${province_id}.\n${result}`,
        },
      ],
    };
  }
);

server.tool(
  "set_tax_rate",
  "Set the tax rate for a province. Rate is 0.0 to 1.0 (0% to 100%). Higher taxes = more gold but more unrest. Recommended: 0.3-0.5 for stable provinces, lower for high-unrest ones.",
  {
    province_id: z.number().int().describe("Province ID"),
    rate: z.number().min(0).max(1).describe("Tax rate (0.0 to 1.0)"),
  },
  async ({ province_id, rate }) => {
    // Convert to FixedPoint (×1000)
    const fixedRate = Math.round(rate * 1000);
    const result = await submitAction({
      SetTaxRate: { province: province_id, rate: { raw: fixedRate } },
    });
    return {
      content: [
        {
          type: "text" as const,
          text: `Tax rate in province ${province_id} set to ${(rate * 100).toFixed(1)}%.\n${result}`,
        },
      ],
    };
  }
);

server.tool(
  "establish_trade_route",
  "Establish a trade route between two adjacent provinces you control. Both must be yours.",
  {
    from_province: z.number().int().describe("Source province ID"),
    to_province: z.number().int().describe("Destination province ID"),
  },
  async ({ from_province, to_province }) => {
    const result = await submitAction({
      EstablishTradeRoute: { from: from_province, to: to_province },
    });
    return {
      content: [
        {
          type: "text" as const,
          text: `Trade route established: province ${from_province} ↔ ${to_province}.\n${result}`,
        },
      ],
    };
  }
);

server.tool(
  "disrupt_trade_route",
  "Disrupt (raid/blockade) an enemy trade route by its route ID.",
  { route_id: z.number().int().describe("Trade route ID to disrupt") },
  async ({ route_id }) => {
    const result = await submitAction({ DisruptTradeRoute: { route_id } });
    return {
      content: [{ type: "text" as const, text: `Trade route ${route_id} disrupted.\n${result}` }],
    };
  }
);

// ═══════════════════════════════════════════════════════════════════════════════
// ACTION TOOLS — CHARACTERS & INTRIGUE
// ═══════════════════════════════════════════════════════════════════════════════

server.tool(
  "assign_councilor",
  "Assign a character to a council role: Marshal (military), Steward (economy), Chaplain (religion), Spymaster (intrigue). Use analyze_characters first to find the best candidate.",
  {
    character_id: z.number().int().describe("Character ID to assign"),
    role: z
      .enum(["Marshal", "Chaplain", "Steward", "Spymaster"])
      .describe("Council role"),
  },
  async ({ character_id, role }) => {
    const result = await submitAction({
      AssignCouncilor: { character: character_id, role },
    });
    return {
      content: [
        {
          type: "text" as const,
          text: `${role} position assigned to character ${character_id}.\n${result}`,
        },
      ],
    };
  }
);

server.tool(
  "designate_heir",
  "Designate a character as your heir for succession. Choose wisely — this affects legitimacy and stability.",
  { character_id: z.number().int().describe("Character ID to designate as heir") },
  async ({ character_id }) => {
    const result = await submitAction({ DesignateHeir: { character: character_id } });
    return {
      content: [
        {
          type: "text" as const,
          text: `Character ${character_id} designated as heir.\n${result}`,
        },
      ],
    };
  }
);

server.tool(
  "arrange_marriage",
  "Arrange a marriage between two characters. Great for diplomatic alliances and dynasty building.",
  {
    character_a: z.number().int().describe("First character ID"),
    character_b: z.number().int().describe("Second character ID"),
  },
  async ({ character_a, character_b }) => {
    const result = await submitAction({
      ArrangeMarriage: { a: character_a, b: character_b },
    });
    return {
      content: [
        {
          type: "text" as const,
          text: `Marriage arranged between characters ${character_a} and ${character_b}.\n${result}`,
        },
      ],
    };
  }
);

server.tool(
  "convert_province",
  "Begin religious conversion of a province. Religions: OldFaith, EmberChurch, SaltCult, FrostSpirits, BlackOrder. Conversion takes several turns.",
  {
    province_id: z.number().int().describe("Province ID to convert"),
    religion: z
      .enum(["OldFaith", "EmberChurch", "SaltCult", "FrostSpirits", "BlackOrder"])
      .describe("Target religion"),
  },
  async ({ province_id, religion }) => {
    const result = await submitAction({
      ConvertProvince: { province: province_id, religion },
    });
    return {
      content: [
        {
          type: "text" as const,
          text: `Converting province ${province_id} to ${religion}.\n${result}`,
        },
      ],
    };
  }
);

server.tool(
  "launch_plot",
  "Launch an intrigue plot against a target character. Types: Assassination (kill), Fabricate (claim province), Sabotage (reduce prosperity), Steal (take gold).",
  {
    target_character: z.number().int().describe("Character ID to target"),
    plot_type: z
      .enum(["Assassination", "Fabricate", "Sabotage", "Steal"])
      .describe("Type of plot"),
  },
  async ({ target_character, plot_type }) => {
    const result = await submitAction({
      LaunchPlot: { target: target_character, plot_type },
    });
    return {
      content: [
        {
          type: "text" as const,
          text: `Plot launched: ${plot_type} against character ${target_character}.\n${result}`,
        },
      ],
    };
  }
);

server.tool(
  "back_plot",
  "Back (support) an existing intrigue plot to increase its progress speed.",
  { plot_id: z.number().int().describe("Plot ID to back") },
  async ({ plot_id }) => {
    const result = await submitAction({ BackPlot: { plot_id } });
    return {
      content: [{ type: "text" as const, text: `Plot ${plot_id} backed.\n${result}` }],
    };
  }
);

server.tool(
  "investigate_plots",
  "Use a spymaster to investigate and detect enemy intrigue plots targeting your realm.",
  {
    spymaster_id: z
      .number()
      .int()
      .describe("Character ID of your spymaster to use"),
  },
  async ({ spymaster_id }) => {
    const result = await submitAction({ InvestigatePlot: { spymaster: spymaster_id } });
    return {
      content: [
        {
          type: "text" as const,
          text: `Spymaster ${spymaster_id} investigating plots...\n${result}`,
        },
      ],
    };
  }
);

// ═══════════════════════════════════════════════════════════════════════════════
// COMPOSITE STRATEGY TOOL
// ═══════════════════════════════════════════════════════════════════════════════

server.tool(
  "strategic_briefing",
  "Get a comprehensive strategic briefing: your realm summary, threat analysis, economic status, diplomatic situation, and recommended priorities. This is your go-to tool for deciding what to do each turn.",
  {},
  async () => {
    const world = await getWorld();
    const faction = findMyFaction(world);
    const provinces = myProvinces(world, faction.id);
    const armies = myArmies(world, faction.id);
    const chars = myCharacters(world, faction.id);
    const myProvIds = new Set(provinces.map((p: any) => p.id));

    // Military strength
    const myTotalPower = armies.reduce(
      (s: number, a: any) => s + armyPower(a.troops),
      0
    );

    // Per-faction power comparison
    const factionPowers = (world.factions || [])
      .filter((f: any) => f.alive && f.id !== faction.id)
      .map((f: any) => {
        const fArmies = (world.armies || []).filter(
          (a: any) => a.owner_faction === f.id
        );
        const fProvs = (world.provinces || []).filter(
          (p: any) => p.controller === f.id
        );
        return {
          name: f.name,
          id: f.id,
          provinces: fProvs.length,
          military_power: fArmies.reduce(
            (s: number, a: any) => s + armyPower(a.troops),
            0
          ),
          at_war_with_us: (world.diplomacy || []).some(
            (d: any) =>
              d.at_war &&
              ((d.faction_a === faction.id && d.faction_b === f.id) ||
                (d.faction_b === faction.id && d.faction_a === f.id))
          ),
        };
      })
      .sort((a: any, b: any) => b.military_power - a.military_power);

    // Threats
    const enemiesNearBorder = (world.armies || []).filter(
      (a: any) =>
        a.owner_faction !== faction.id &&
        provinces.some(
          (p: any) =>
            (p.neighbors || []).includes(a.location) || a.location === p.id
        )
    ).length;

    // Economy
    const totalGold = provinces.reduce(
      (s: number, p: any) => s + fp(p.resources?.gold || 0),
      0
    );

    // Recommendations
    const recommendations: string[] = [];

    if (armies.length === 0) {
      recommendations.push(
        "URGENT: You have no armies! Raise one immediately from your most populated province."
      );
    }

    const idleArmies = armies.filter((a: any) => !a.destination && !a.movement_queue?.length && !a.siege);
    if (idleArmies.length > 0) {
      recommendations.push(
        `${idleArmies.length} army(ies) idle — consider moving them to border provinces or attacking.`
      );
    }

    if (enemiesNearBorder > 0) {
      recommendations.push(
        `${enemiesNearBorder} enemy army(ies) near your borders — consider defensive positioning.`
      );
    }

    const noImprovements = provinces.filter(
      (p: any) => (p.improvements || []).length === 0
    );
    if (noImprovements.length > 0) {
      recommendations.push(
        `${noImprovements.length} province(s) have no improvements — build Farmsteads for food and Markets for gold.`
      );
    }

    const ruler = chars.find((c: any) => c.role === "Ruler");
    if (ruler && !ruler.heir) {
      recommendations.push(
        "Your ruler has no designated heir! Use designate_heir to prevent succession crisis."
      );
    }

    const atWar = factionPowers.filter((f: any) => f.at_war_with_us);
    if (atWar.length > 0) {
      recommendations.push(
        `Currently at war with: ${atWar.map((f: any) => f.name).join(", ")}. Focus military efforts!`
      );
    }

    return {
      content: [
        {
          type: "text" as const,
          text: JSON.stringify(
            {
              turn: world.meta?.turn,
              my_faction: faction.name,
              my_strength: {
                provinces: provinces.length,
                armies: armies.length,
                total_military_power: myTotalPower,
                total_gold_income: totalGold.toFixed(1),
                characters: chars.length,
              },
              rival_factions: factionPowers,
              threats: {
                enemy_armies_near_border: enemiesNearBorder,
                active_wars: atWar.length,
              },
              recommendations,
            },
            null,
            2
          ),
        },
      ],
    };
  }
);

// ═══════════════════════════════════════════════════════════════════════════════
// ENHANCED OBSERVATION TOOLS (v2.0)
// ═══════════════════════════════════════════════════════════════════════════════

server.tool(
  "get_province_history",
  "Get full narrative history for a province: sieges, conquests, trade, religion, construction, battles.",
  { province_id: z.number().describe("Province ID (0-24)") },
  async ({ province_id }) => {
    const world = await getWorld();
    const prov = world.provinces?.find((p: any) => p.id === province_id);
    if (!prov) throw new Error(`Province ${province_id} not found`);

    const events = (world.history || world.events || []).filter(
      (e: any) => e.province === province_id || e.province_id === province_id ||
        e.target_province === province_id || e.location === province_id
    );

    const sieges = events.filter((e: any) => e.type?.includes("Siege"));
    const battles = events.filter((e: any) => e.type?.includes("Battle"));
    const conquests = events.filter((e: any) => e.type?.includes("Conquest") || e.type?.includes("Capture"));
    const trades = events.filter((e: any) => e.type?.includes("Trade"));
    const constructions = events.filter((e: any) => e.type?.includes("Construction"));
    const religious = events.filter((e: any) => e.type?.includes("Relig") || e.type?.includes("Heresy") || e.type?.includes("Miracle"));

    return {
      content: [{
        type: "text" as const,
        text: JSON.stringify({
          province: { id: prov.id, name: prov.name, terrain: prov.terrain, controller: prov.controller },
          population: prov.population,
          prosperity: fp(prov.prosperity),
          unrest: fp(prov.unrest),
          improvements: prov.improvements,
          garrison: prov.garrison,
          history: {
            total_events: events.length,
            sieges: sieges.length,
            battles: battles.length,
            conquests: conquests.length,
            trades: trades.length,
            constructions: constructions.length,
            religious_events: religious.length,
            recent_events: events.slice(-10),
          },
        }, null, 2),
      }],
    };
  }
);

server.tool(
  "get_character_biography",
  "Get complete character biography: traits, relationships, battle history, marriages, chronicle.",
  { character_id: z.number().describe("Character ID") },
  async ({ character_id }) => {
    const world = await getWorld();
    const char = (world.characters || []).find((c: any) => c.id === character_id);
    if (!char) throw new Error(`Character ${character_id} not found`);

    const events = (world.history || world.events || []).filter(
      (e: any) => e.character === character_id || e.character_id === character_id ||
        e.attacker === character_id || e.defender === character_id ||
        e.plotter === character_id || e.target === character_id
    );

    const battles = events.filter((e: any) => e.type?.includes("Battle"));
    const marriages = events.filter((e: any) => e.type?.includes("Marriage"));
    const plots = events.filter((e: any) => e.type?.includes("Plot") || e.type?.includes("Assassination"));

    return {
      content: [{
        type: "text" as const,
        text: JSON.stringify({
          character: {
            id: char.id, name: char.name, age: char.age, alive: char.alive,
            faction: char.faction, role: char.role, dynasty: char.dynasty,
            traits: char.traits, stats: char.stats,
          },
          relationships: char.relationships || [],
          chronicle: {
            battles_fought: battles.length,
            marriages: marriages.length,
            plots_involved: plots.length,
            events: events.slice(-15),
          },
        }, null, 2),
      }],
    };
  }
);

server.tool(
  "get_battle_reports",
  "Get detailed battle reports for recent conflicts. Shows combatants, casualties, outcomes.",
  { last_n_turns: z.number().default(10).describe("How many recent turns to include") },
  async ({ last_n_turns }) => {
    const world = await getWorld();
    const currentTurn = world.meta?.turn || 0;
    const minTurn = Math.max(0, currentTurn - last_n_turns);

    const battles = (world.history || world.events || []).filter(
      (e: any) => e.type?.includes("Battle") && (e.turn ?? 0) >= minTurn
    );

    return {
      content: [{
        type: "text" as const,
        text: JSON.stringify({
          turn_range: `${minTurn}-${currentTurn}`,
          total_battles: battles.length,
          battles: battles.map((b: any) => ({
            turn: b.turn,
            location: b.province ?? b.location ?? b.province_id,
            attacker: b.attacker_faction ?? b.attacker,
            defender: b.defender_faction ?? b.defender,
            attacker_casualties: b.attacker_casualties ?? b.attacker_dead,
            defender_casualties: b.defender_casualties ?? b.defender_dead,
            victor: b.victor ?? b.winner,
          })),
        }, null, 2),
      }],
    };
  }
);

server.tool(
  "get_turn_events",
  "Get raw event list for a specific turn range.",
  {
    from_turn: z.number().describe("Starting turn"),
    to_turn: z.number().describe("Ending turn (inclusive)"),
  },
  async ({ from_turn, to_turn }) => {
    const world = await getWorld();
    const events = (world.history || world.events || []).filter(
      (e: any) => (e.turn ?? 0) >= from_turn && (e.turn ?? 0) <= to_turn
    );

    return {
      content: [{
        type: "text" as const,
        text: JSON.stringify({
          range: `${from_turn}-${to_turn}`,
          event_count: events.length,
          events,
        }, null, 2),
      }],
    };
  }
);

server.tool(
  "get_map_state",
  "Get full province ownership map with faction colors, terrain, improvements, and military presence.",
  {},
  async () => {
    const world = await getWorld();
    const provinces = (world.provinces || []).map((p: any) => {
      const faction = world.factions?.find((f: any) => f.id === p.controller);
      const armiesHere = (world.armies || []).filter((a: any) => a.location === p.id);
      return {
        id: p.id,
        name: p.name,
        terrain: p.terrain,
        controller: faction?.name ?? `faction_${p.controller}`,
        population: p.population,
        garrison: p.garrison,
        improvements: p.improvements,
        armies_present: armiesHere.map((a: any) => ({
          id: a.id,
          faction: a.owner_faction,
          troops: a.troops,
          power: armyPower(a.troops || {}),
        })),
        prosperity: fp(p.prosperity),
        unrest: fp(p.unrest),
        fortification: p.fortification ?? 0,
      };
    });

    return {
      content: [{
        type: "text" as const,
        text: JSON.stringify({ turn: world.meta?.turn, provinces }, null, 2),
      }],
    };
  }
);

// ═══════════════════════════════════════════════════════════════════════════════
// ADVANCED MILITARY STRATEGY TOOLS (v2.0)
// ═══════════════════════════════════════════════════════════════════════════════

server.tool(
  "evaluate_battle_odds",
  "Pre-battle calculator: estimate win probability and casualties for attacking a province.",
  {
    army_id: z.number().describe("Your army ID"),
    target_province: z.number().describe("Province ID to attack"),
  },
  async ({ army_id, target_province }) => {
    const world = await getWorld();
    const army = (world.armies || []).find((a: any) => a.id === army_id);
    if (!army) throw new Error(`Army ${army_id} not found`);

    const prov = world.provinces?.find((p: any) => p.id === target_province);
    if (!prov) throw new Error(`Province ${target_province} not found`);

    const myPower = armyPower(army.troops || {});
    const garrison = prov.garrison || 0;
    const fort = prov.fortification ?? 0;

    // Defender gets terrain + fortification bonus
    const terrainBonus: Record<string, number> = {
      Mountains: 0.5, Hills: 0.3, Forest: 0.2, Marsh: 0.15,
      Plains: 0, Desert: 0, Coastal: 0.1, River: 0.1,
    };
    const tBonus = terrainBonus[prov.terrain] ?? 0;
    const defPower = garrison * (1 + tBonus + fort * 0.2);

    // Defending armies at the province
    const defArmies = (world.armies || []).filter(
      (a: any) => a.location === target_province && a.owner_faction === prov.controller
    );
    const defArmyPower = defArmies.reduce((s: number, a: any) => s + armyPower(a.troops || {}), 0);
    const totalDefPower = defPower + defArmyPower;

    const ratio = totalDefPower > 0 ? myPower / totalDefPower : 999;
    const winProb = Math.min(0.95, Math.max(0.05, 0.5 + (ratio - 1) * 0.25));
    const estCasualties = Math.round(myPower * (1 - winProb) * 0.3);

    return {
      content: [{
        type: "text" as const,
        text: JSON.stringify({
          attacker: { army_id, power: myPower, troops: army.troops },
          defender: {
            province: prov.name,
            garrison,
            fortification: fort,
            terrain: prov.terrain,
            terrain_bonus: `+${(tBonus * 100).toFixed(0)}%`,
            garrison_effective_power: Math.round(defPower),
            defending_armies: defArmies.length,
            army_power: defArmyPower,
            total_defensive_power: Math.round(totalDefPower),
          },
          analysis: {
            power_ratio: ratio.toFixed(2),
            win_probability: `${(winProb * 100).toFixed(0)}%`,
            estimated_attacker_casualties: estCasualties,
            recommendation: winProb > 0.7 ? "ATTACK — favorable odds" :
              winProb > 0.5 ? "RISKY — consider reinforcements" :
              "AVOID — unfavorable, reinforce first",
          },
          siege_required: fort > 0,
          siege_duration_turns: fort > 0 ? (fort + 1) * 3 : 0,
        }, null, 2),
      }],
    };
  }
);

server.tool(
  "plan_campaign",
  "Multi-turn invasion planner: pathfind to target, estimate army needs, generate order sequence.",
  {
    target_province: z.number().describe("Province ID to conquer"),
    faction_id: z.number().optional().describe("Your faction ID (auto-detected if wallet set)"),
  },
  async ({ target_province, faction_id }) => {
    const world = await getWorld();
    let myFactionId = faction_id;
    if (!myFactionId && WALLET) {
      const f = findMyFaction(world);
      myFactionId = f.id;
    }
    if (myFactionId == null) throw new Error("Provide faction_id or set CROWN_ASH_WALLET");

    const myProvs = myProvinces(world, myFactionId);
    const myArmyList = myArmies(world, myFactionId);
    const target = world.provinces?.find((p: any) => p.id === target_province);
    if (!target) throw new Error(`Province ${target_province} not found`);

    // Simple BFS for shortest path from any owned province
    const adjacency: Record<number, number[]> = {};
    const adjPairs = [
      [0,1],[0,2],[0,18],[1,2],[1,7],[2,3],[2,8],[3,8],[3,21],
      [4,5],[4,9],[4,13],[5,6],[5,10],[5,14],[6,10],[6,15],
      [7,8],[7,9],[7,11],[8,9],[8,20],[8,21],[9,10],
      [10,5],[10,6],[10,12],[11,12],[11,19],[12,13],[12,10],
      [13,4],[13,17],[14,15],[14,16],[15,16],[16,17],[17,24],
      [18,19],[18,20],[19,11],[19,20],[20,8],[21,22],[22,23],[22,24],[23,24],
    ];
    for (const [a, b] of adjPairs) {
      if (!adjacency[a]) adjacency[a] = [];
      if (!adjacency[b]) adjacency[b] = [];
      adjacency[a].push(b);
      adjacency[b].push(a);
    }

    // BFS from target backwards to find closest owned province
    const visited = new Set<number>();
    const parent = new Map<number, number>();
    const queue = [target_province];
    visited.add(target_province);
    let staging: number | null = null;

    while (queue.length > 0) {
      const cur = queue.shift()!;
      if (myProvs.some((p: any) => p.id === cur) && cur !== target_province) {
        staging = cur;
        break;
      }
      for (const neighbor of (adjacency[cur] || [])) {
        if (!visited.has(neighbor)) {
          visited.add(neighbor);
          parent.set(neighbor, cur);
          queue.push(neighbor);
        }
      }
    }

    // Reconstruct path
    const path: number[] = [];
    if (staging != null) {
      let cur = staging;
      while (cur !== target_province) {
        path.push(cur);
        cur = parent.get(cur)!;
      }
      path.push(target_province);
    }

    const garrison = target.garrison || 0;
    const fort = target.fortification ?? 0;
    const defArmies = (world.armies || []).filter(
      (a: any) => a.location === target_province && a.owner_faction === target.controller
    );
    const totalDef = garrison + defArmies.reduce((s: number, a: any) => s + armyPower(a.troops || {}), 0);
    const recommendedPower = Math.round(totalDef * 2.5); // 2.5:1 advantage

    const orders: string[] = [];
    if (myArmyList.length === 0) {
      orders.push(`raise_army in province ${staging} with at least ${recommendedPower} power`);
    }
    for (let i = 0; i < path.length - 1; i++) {
      orders.push(`move_army to province ${path[i + 1]}`);
    }
    if (fort > 0) {
      orders.push(`siege will take ${(fort + 1) * 3} turns`);
    }
    orders.push(`attack province ${target_province}`);

    return {
      content: [{
        type: "text" as const,
        text: JSON.stringify({
          target: { id: target_province, name: target.name, controller: target.controller },
          path: path.map((id) => {
            const p = world.provinces?.find((pr: any) => pr.id === id);
            return { id, name: p?.name, controller: p?.controller };
          }),
          distance_turns: path.length - 1,
          staging_province: staging,
          enemy_strength: {
            garrison,
            fortification: fort,
            defending_armies: defArmies.length,
            total_defensive_power: totalDef,
          },
          recommended_attack_power: recommendedPower,
          available_armies: myArmyList.map((a: any) => ({
            id: a.id, location: a.location, power: armyPower(a.troops || {}),
          })),
          orders,
        }, null, 2),
      }],
    };
  }
);

server.tool(
  "coordinate_armies",
  "Multi-army pincer movement: send 2+ armies to converge on a target province simultaneously.",
  {
    army_ids: z.array(z.number()).describe("Army IDs to coordinate"),
    target_province: z.number().describe("Province ID to converge on"),
  },
  async ({ army_ids, target_province }) => {
    const world = await getWorld();
    const target = world.provinces?.find((p: any) => p.id === target_province);
    if (!target) throw new Error(`Province ${target_province} not found`);

    const moves = army_ids.map((aid: number) => {
      const army = (world.armies || []).find((a: any) => a.id === aid);
      if (!army) return { army_id: aid, error: "not found" };
      return {
        army_id: aid,
        current_location: army.location,
        power: armyPower(army.troops || {}),
        troops: army.troops,
      };
    });

    const totalPower = moves.reduce((s: number, m: any) => s + (m.power || 0), 0);

    return {
      content: [{
        type: "text" as const,
        text: JSON.stringify({
          target: { id: target_province, name: target.name },
          armies: moves,
          combined_power: totalPower,
          orders: army_ids.map((aid: number) => `move_army ${aid} to province ${target_province}`),
          note: "Submit each move_army action separately. Armies converge over subsequent turns.",
        }, null, 2),
      }],
    };
  }
);

// ═══════════════════════════════════════════════════════════════════════════════
// ADVANCED DIPLOMACY & ESPIONAGE TOOLS (v2.0)
// ═══════════════════════════════════════════════════════════════════════════════

server.tool(
  "evaluate_alliance_value",
  "Score a potential ally by: shared enemies, border proximity, military strength, treaty history.",
  { faction_id: z.number().describe("Faction ID to evaluate as potential ally") },
  async ({ faction_id }) => {
    const world = await getWorld();
    const myFaction = findMyFaction(world);
    const target = world.factions?.find((f: any) => f.id === faction_id);
    if (!target) throw new Error(`Faction ${faction_id} not found`);

    const diplomacy = world.diplomacy || [];
    const myWars = diplomacy.filter(
      (d: any) => d.at_war && (d.faction_a === myFaction.id || d.faction_b === myFaction.id)
    );
    const theirWars = diplomacy.filter(
      (d: any) => d.at_war && (d.faction_a === faction_id || d.faction_b === faction_id)
    );

    // Shared enemies
    const myEnemies = new Set(myWars.map((w: any) =>
      w.faction_a === myFaction.id ? w.faction_b : w.faction_a
    ));
    const theirEnemies = new Set(theirWars.map((w: any) =>
      w.faction_a === faction_id ? w.faction_b : w.faction_a
    ));
    const sharedEnemies = [...myEnemies].filter((e) => theirEnemies.has(e));

    // Military strength
    const theirArmies = myArmies(world, faction_id);
    const theirPower = theirArmies.reduce((s: number, a: any) => s + armyPower(a.troops || {}), 0);
    const theirProvCount = myProvinces(world, faction_id).length;

    // Border check
    const myProvIds = new Set(myProvinces(world, myFaction.id).map((p: any) => p.id));
    const theirProvIds = new Set(myProvinces(world, faction_id).map((p: any) => p.id));

    let score = 0;
    score += sharedEnemies.length * 30; // shared enemies are valuable
    score += theirPower > 0 ? Math.min(25, theirPower / 100) : 0; // military strength
    score += theirProvCount * 5; // territory = resources

    // Check if at war with us (terrible ally)
    const atWarWithUs = diplomacy.some(
      (d: any) => d.at_war &&
        ((d.faction_a === myFaction.id && d.faction_b === faction_id) ||
         (d.faction_b === myFaction.id && d.faction_a === faction_id))
    );
    if (atWarWithUs) score = -100;

    return {
      content: [{
        type: "text" as const,
        text: JSON.stringify({
          faction: { id: faction_id, name: target.name, alive: target.alive },
          alliance_score: score,
          shared_enemies: sharedEnemies,
          their_strength: { provinces: theirProvCount, military_power: theirPower, armies: theirArmies.length },
          at_war_with_us: atWarWithUs,
          recommendation: atWarWithUs ? "ENEMY — cannot ally" :
            score > 50 ? "EXCELLENT ally — pursue alliance" :
            score > 25 ? "GOOD ally — worth considering" :
            score > 0 ? "MARGINAL — ally only if needed" :
            "POOR — look elsewhere",
        }, null, 2),
      }],
    };
  }
);

server.tool(
  "negotiate_peace",
  "Analyze war exhaustion and propose peace terms based on current situation.",
  { enemy_faction_id: z.number().describe("Enemy faction ID to negotiate with") },
  async ({ enemy_faction_id }) => {
    const world = await getWorld();
    const myFaction = findMyFaction(world);
    const enemy = world.factions?.find((f: any) => f.id === enemy_faction_id);
    if (!enemy) throw new Error(`Faction ${enemy_faction_id} not found`);

    const myProvs = myProvinces(world, myFaction.id);
    const enemyProvs = myProvinces(world, enemy_faction_id);
    const myPower = myArmies(world, myFaction.id).reduce((s: number, a: any) => s + armyPower(a.troops || {}), 0);
    const enemyPower = myArmies(world, enemy_faction_id).reduce((s: number, a: any) => s + armyPower(a.troops || {}), 0);

    const events = world.history || world.events || [];
    const warBattles = events.filter(
      (e: any) => e.type?.includes("Battle") &&
        ((e.attacker_faction === myFaction.id && e.defender_faction === enemy_faction_id) ||
         (e.attacker_faction === enemy_faction_id && e.defender_faction === myFaction.id))
    );

    const winning = myPower > enemyPower * 1.3 && myProvs.length >= enemyProvs.length;

    return {
      content: [{
        type: "text" as const,
        text: JSON.stringify({
          war_status: {
            our_provinces: myProvs.length,
            their_provinces: enemyProvs.length,
            our_military_power: myPower,
            their_military_power: enemyPower,
            battles_fought: warBattles.length,
          },
          assessment: winning ? "We are WINNING — demand concessions" :
            myPower < enemyPower * 0.7 ? "We are LOSING — consider accepting terms" :
            "STALEMATE — white peace is reasonable",
          recommended_action: winning ? "propose_treaty with territory demands" :
            myPower < enemyPower * 0.7 ? "propose_treaty (accept white peace)" :
            "continue fighting or propose_treaty (white peace)",
        }, null, 2),
      }],
    };
  }
);

// ═══════════════════════════════════════════════════════════════════════════════
// ECONOMY & DEVELOPMENT TOOLS (v2.0)
// ═══════════════════════════════════════════════════════════════════════════════

server.tool(
  "optimize_build_order",
  "Recommend optimal improvement build priority for a province based on terrain, threats, and needs.",
  { province_id: z.number().describe("Province ID to analyze") },
  async ({ province_id }) => {
    const world = await getWorld();
    const prov = world.provinces?.find((p: any) => p.id === province_id);
    if (!prov) throw new Error(`Province ${province_id} not found`);

    const existing = new Set(prov.improvements || []);
    const queue = (prov.construction_queue || []).map((q: any) => q[0] || q);

    const recommendations: Array<{ improvement: string; priority: string; reason: string }> = [];

    // Context-aware recommendations
    const isBorder = (world.armies || []).some(
      (a: any) => a.location === province_id && a.owner_faction !== prov.controller
    );
    const lowProsperity = fp(prov.prosperity) < 0.4;
    const highUnrest = fp(prov.unrest) > 0.5;

    if (!existing.has("Farmstead") && prov.population < 3000) {
      recommendations.push({ improvement: "Farmstead", priority: "HIGH", reason: "Low population — food production increases growth" });
    }
    if (!existing.has("Market") && !lowProsperity) {
      recommendations.push({ improvement: "Market", priority: "HIGH", reason: "Gold income for military and further development" });
    }
    if (!existing.has("Fortification") && isBorder) {
      recommendations.push({ improvement: "Fortification", priority: "CRITICAL", reason: "Border province under threat — siege protection" });
    }
    if (!existing.has("Temple") && highUnrest) {
      recommendations.push({ improvement: "Temple", priority: "HIGH", reason: "High unrest — religious authority calms population" });
    }
    if (!existing.has("Granary") && prov.population > 2000) {
      recommendations.push({ improvement: "Granary", priority: "MEDIUM", reason: "Growing population needs food storage" });
    }
    if (prov.terrain === "Coastal" && !existing.has("Port")) {
      recommendations.push({ improvement: "Port", priority: "HIGH", reason: "Coastal province — port unlocks trade routes" });
    }
    if (!existing.has("University") && existing.size >= 4) {
      recommendations.push({ improvement: "University", priority: "LOW", reason: "Late-game tech boost" });
    }
    if (!existing.has("Hospital") && prov.population > 5000) {
      recommendations.push({ improvement: "Hospital", priority: "MEDIUM", reason: "Large population needs plague protection" });
    }

    return {
      content: [{
        type: "text" as const,
        text: JSON.stringify({
          province: { id: prov.id, name: prov.name, terrain: prov.terrain },
          existing_improvements: [...existing],
          under_construction: queue,
          recommendations: recommendations.sort((a, b) => {
            const pri: Record<string, number> = { CRITICAL: 0, HIGH: 1, MEDIUM: 2, LOW: 3 };
            return (pri[a.priority] ?? 9) - (pri[b.priority] ?? 9);
          }),
        }, null, 2),
      }],
    };
  }
);

server.tool(
  "economic_forecast",
  "Predict income/expenses for next N turns based on current realm state.",
  { turns_ahead: z.number().default(5).describe("Number of turns to forecast") },
  async ({ turns_ahead }) => {
    const world = await getWorld();
    const faction = findMyFaction(world);
    const provs = myProvinces(world, faction.id);
    const armies = myArmies(world, faction.id);

    // Income per turn
    const taxIncome = provs.reduce((s: number, p: any) => {
      const tax = (p.tax_rate ?? 0.3) * (p.population || 0) * fp(p.prosperity) * 0.01;
      return s + tax;
    }, 0);
    const tradeIncome = provs.reduce((s: number, p: any) => {
      const hasPort = (p.improvements || []).includes("Port");
      const hasMarket = (p.improvements || []).includes("Market");
      return s + (hasPort ? 5 : 0) + (hasMarket ? 3 : 0);
    }, 0);

    // Expenses per turn
    const armyUpkeep = armies.reduce((s: number, a: any) => {
      const t = a.troops || {};
      return s + (t.levy || 0) * 0.1 + (t.men_at_arms || 0) * 0.5 + (t.knights || 0) * 2;
    }, 0);

    const netPerTurn = taxIncome + tradeIncome - armyUpkeep;
    const treasury = faction.treasury ?? 0;

    const forecast = Array.from({ length: turns_ahead }, (_, i) => ({
      turn: (world.meta?.turn || 0) + i + 1,
      projected_treasury: Math.round(treasury + netPerTurn * (i + 1)),
    }));

    return {
      content: [{
        type: "text" as const,
        text: JSON.stringify({
          current_treasury: treasury,
          income_per_turn: {
            tax: Math.round(taxIncome),
            trade: Math.round(tradeIncome),
            total: Math.round(taxIncome + tradeIncome),
          },
          expenses_per_turn: {
            army_upkeep: Math.round(armyUpkeep),
            total: Math.round(armyUpkeep),
          },
          net_per_turn: Math.round(netPerTurn),
          forecast,
          warning: netPerTurn < 0 ? "DEFICIT — reduce army or increase income!" : null,
        }, null, 2),
      }],
    };
  }
);

// ═══════════════════════════════════════════════════════════════════════════════
// GAME SESSION TOOLS (v2.0)
// ═══════════════════════════════════════════════════════════════════════════════

server.tool(
  "take_turn",
  "Manually advance the game by one turn (when game is paused or waiting for input).",
  {},
  async () => {
    const wallet = requireWallet();
    clearCache();
    const result = await apiPost("/advance", { wallet });
    return {
      content: [{
        type: "text" as const,
        text: JSON.stringify({ status: "turn_advanced", result }, null, 2),
      }],
    };
  }
);

server.tool(
  "get_diplomatic_matrix",
  "Get the full diplomatic relations matrix: who's at war, allied, or neutral with whom.",
  {},
  async () => {
    const world = await getWorld();
    const factions = (world.factions || []).filter((f: any) => f.alive);
    const diplomacy = world.diplomacy || [];

    const matrix = factions.map((f: any) => {
      const relations = factions
        .filter((f2: any) => f2.id !== f.id)
        .map((f2: any) => {
          const rel = diplomacy.find(
            (d: any) =>
              (d.faction_a === f.id && d.faction_b === f2.id) ||
              (d.faction_b === f.id && d.faction_a === f2.id)
          );
          return {
            faction: f2.name,
            at_war: rel?.at_war ?? false,
            opinion: rel ? fp(rel.opinion) : 0,
            treaties: rel?.treaties ?? [],
          };
        });
      return { faction: f.name, id: f.id, relations };
    });

    return {
      content: [{
        type: "text" as const,
        text: JSON.stringify({ turn: world.meta?.turn, matrix }, null, 2),
      }],
    };
  }
);

// ═══════════════════════════════════════════════════════════════════════════════
// START SERVER
// ═══════════════════════════════════════════════════════════════════════════════

async function main() {
  const transport = new StdioServerTransport();
  await server.connect(transport);
}

main().catch((err) => {
  console.error("Crown & Ash MCP server error:", err);
  process.exit(1);
});
