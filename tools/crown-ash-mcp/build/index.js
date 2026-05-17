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
async function apiGet(path) {
    const resp = await fetch(`${API}${path}`);
    if (!resp.ok)
        throw new Error(`GET ${path}: HTTP ${resp.status}`);
    const json = (await resp.json());
    if (json.success === false)
        throw new Error(json.error || "API error");
    return json.data ?? json;
}
async function apiPost(path, body) {
    const resp = await fetch(`${API}${path}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
    });
    if (!resp.ok) {
        const text = await resp.text();
        throw new Error(`POST ${path}: HTTP ${resp.status} — ${text}`);
    }
    const json = (await resp.json());
    if (json.success === false)
        throw new Error(json.error || "API error");
    return json.data ?? json;
}
function requireWallet() {
    if (!WALLET)
        throw new Error("CROWN_ASH_WALLET not set — set it in env");
    return WALLET;
}
// ─── World state cache (refreshed per-call, avoids redundant fetches) ─────
let cachedWorld = null;
let cacheTime = 0;
const CACHE_TTL = 3000; // 3 seconds
async function getWorld() {
    const now = Date.now();
    if (cachedWorld && now - cacheTime < CACHE_TTL)
        return cachedWorld;
    cachedWorld = await apiGet("/world");
    cacheTime = now;
    return cachedWorld;
}
function clearCache() {
    cachedWorld = null;
    cacheTime = 0;
}
// ─── Analysis helpers ─────────────────────────────────────────────────────────
function fp(val) {
    if (typeof val === "number")
        return val / 1000;
    if (typeof val === "object" && val?.raw != null)
        return val.raw / 1000;
    return 0;
}
function armyPower(troops) {
    return (troops.levy || 0) + (troops.men_at_arms || 0) * 3 + (troops.knights || 0) * 10;
}
function findMyFaction(world) {
    const wallet = requireWallet();
    const faction = world.factions.find((f) => f.player_wallet === wallet);
    if (!faction)
        throw new Error(`No faction found for wallet ${wallet}`);
    return faction;
}
function myProvinces(world, factionId) {
    return world.provinces.filter((p) => p.controller === factionId);
}
function myArmies(world, factionId) {
    return (world.armies || []).filter((a) => a.owner_faction === factionId);
}
function myCharacters(world, factionId) {
    return (world.characters || []).filter((c) => c.faction === factionId && c.alive);
}
// ─── Action submission helper ─────────────────────────────────────────────────
async function submitAction(action) {
    const wallet = requireWallet();
    clearCache();
    const result = await apiPost("/action", { wallet, action });
    return JSON.stringify(result, null, 2);
}
// ─── MCP Server Setup ────────────────────────────────────────────────────────
const server = new McpServer({
    name: "crown-ash",
    version: "1.0.0",
});
// ═══════════════════════════════════════════════════════════════════════════════
// OBSERVATION TOOLS
// ═══════════════════════════════════════════════════════════════════════════════
server.tool("get_world_overview", "Get a high-level overview of the entire game world: turn number, all factions (who is alive, who is player-controlled), province count, army count, and active wars. Start here to understand the current game state.", {}, async () => {
    const world = await getWorld();
    const meta = world.meta || {};
    const factions = (world.factions || []).map((f) => ({
        id: f.id,
        name: f.name,
        alive: f.alive,
        player: f.player_wallet ? "human" : "AI",
        culture: f.culture,
        religion: f.religion,
        provinces: (world.provinces || []).filter((p) => p.controller === f.id).length,
        armies: (world.armies || []).filter((a) => a.owner_faction === f.id).length,
    }));
    const wars = (world.diplomacy || []).filter((d) => d.at_war);
    return {
        content: [
            {
                type: "text",
                text: JSON.stringify({
                    turn: meta.turn,
                    total_provinces: (world.provinces || []).length,
                    total_armies: (world.armies || []).length,
                    total_characters: (world.characters || []).filter((c) => c.alive).length,
                    factions,
                    active_wars: wars.map((w) => ({
                        factions: [w.faction_a, w.faction_b],
                        opinion: fp(w.opinion),
                    })),
                }, null, 2),
            },
        ],
    };
});
server.tool("get_my_realm", "Get detailed info about YOUR faction: provinces with resources/improvements, armies with troops/location, characters with stats/traits, treasury, and current wars. Requires CROWN_ASH_WALLET env var.", {}, async () => {
    const world = await getWorld();
    const faction = findMyFaction(world);
    const provinces = myProvinces(world, faction.id);
    const armies = myArmies(world, faction.id);
    const chars = myCharacters(world, faction.id);
    const totalIncome = provinces.reduce((sum, p) => sum + fp(p.resources?.gold || 0), 0);
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
            total_soldiers: armies.reduce((s, a) => s + (a.troops?.levy || 0) + (a.troops?.men_at_arms || 0) + (a.troops?.knights || 0), 0),
            total_gold_income: totalIncome.toFixed(1),
            character_count: chars.length,
        },
        provinces: provinces.map((p) => ({
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
        armies: armies.map((a) => ({
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
        characters: chars.map((c) => ({
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
    return { content: [{ type: "text", text: JSON.stringify(result, null, 2) }] };
});
server.tool("get_province_details", "Get full details of a specific province by ID: resources, garrison, improvements, terrain, neighbors, scars, grudges, conversion progress.", { province_id: z.number().int().describe("Province ID (u16)") }, async ({ province_id }) => {
    const data = await apiGet(`/province/${province_id}`);
    return { content: [{ type: "text", text: JSON.stringify(data, null, 2) }] };
});
server.tool("get_faction_details", "Get full details of a specific faction by ID: bonuses, culture, religion, player status.", { faction_id: z.number().int().describe("Faction ID (0-6)") }, async ({ faction_id }) => {
    const data = await apiGet(`/faction/${faction_id}`);
    return { content: [{ type: "text", text: JSON.stringify(data, null, 2) }] };
});
server.tool("get_diplomacy", "Get all diplomatic relations: who is at war, treaty statuses, opinion scores, and grievances between all faction pairs.", {}, async () => {
    const world = await getWorld();
    const diplomacy = (world.diplomacy || []).map((d) => ({
        factions: [d.faction_a, d.faction_b],
        faction_names: [
            world.factions.find((f) => f.id === d.faction_a)?.name,
            world.factions.find((f) => f.id === d.faction_b)?.name,
        ],
        opinion: fp(d.opinion).toFixed(1),
        at_war: d.at_war,
        treaties: d.treaties,
        grievances: d.grievances,
    }));
    return { content: [{ type: "text", text: JSON.stringify(diplomacy, null, 2) }] };
});
server.tool("get_all_armies", "Get a tactical overview of ALL armies on the map: location, troops, power, movement, siege status. Essential for planning attacks and defense.", {}, async () => {
    const world = await getWorld();
    const armies = (world.armies || []).map((a) => {
        const prov = (world.provinces || []).find((p) => p.id === a.location);
        const faction = (world.factions || []).find((f) => f.id === a.owner_faction);
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
    return { content: [{ type: "text", text: JSON.stringify(armies, null, 2) }] };
});
// ═══════════════════════════════════════════════════════════════════════════════
// STRATEGIC ANALYSIS TOOLS
// ═══════════════════════════════════════════════════════════════════════════════
server.tool("analyze_threats", "Analyze military threats to your realm. Shows enemy armies near your borders, undefended provinces, and provinces under siege. Use this before deciding where to move your armies.", {}, async () => {
    const world = await getWorld();
    const faction = findMyFaction(world);
    const provinces = myProvinces(world, faction.id);
    const myProvIds = new Set(provinces.map((p) => p.id));
    // Find border provinces (have neighbors controlled by other factions)
    const borderProvinces = provinces.filter((p) => (p.neighbors || []).some((n) => !(world.provinces || []).find((pp) => pp.id === n && pp.controller === faction.id)));
    // Find enemy armies near my borders
    const enemyArmies = (world.armies || [])
        .filter((a) => a.owner_faction !== faction.id)
        .filter((a) => {
        // Army is in my province, or in a neighbor of my province
        if (myProvIds.has(a.location))
            return true;
        return provinces.some((p) => (p.neighbors || []).includes(a.location));
    })
        .map((a) => {
        const prov = (world.provinces || []).find((p) => p.id === a.location);
        const ownerFaction = (world.factions || []).find((f) => f.id === a.owner_faction);
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
    const myArmyLocations = new Set(myArmies(world, faction.id).map((a) => a.location));
    const undefended = provinces
        .filter((p) => (p.garrison?.levy || 0) + (p.garrison?.men_at_arms || 0) === 0 &&
        !myArmyLocations.has(p.id))
        .map((p) => ({ id: p.id, name: p.name, fortification: p.fortification }));
    // Provinces under siege
    const sieges = (world.armies || [])
        .filter((a) => a.siege && a.owner_faction !== faction.id)
        .filter((a) => myProvIds.has(a.siege.target_province))
        .map((a) => ({
        province: a.siege.target_province,
        province_name: (world.provinces || []).find((p) => p.id === a.siege.target_province)?.name,
        attacker_army: a.id,
        attacker_faction: (world.factions || []).find((f) => f.id === a.owner_faction)?.name,
        turns_besieged: a.siege.turns_besieged,
        turns_required: a.siege.turns_required,
    }));
    return {
        content: [
            {
                type: "text",
                text: JSON.stringify({
                    threat_level: enemyArmies.length > 2
                        ? "HIGH"
                        : enemyArmies.length > 0
                            ? "MODERATE"
                            : "LOW",
                    enemy_armies_near_border: enemyArmies,
                    undefended_provinces: undefended,
                    provinces_under_siege: sieges,
                    border_provinces: borderProvinces.map((p) => ({
                        id: p.id,
                        name: p.name,
                        garrison: p.garrison,
                        fortification: p.fortification,
                    })),
                }, null, 2),
            },
        ],
    };
});
server.tool("analyze_economy", "Analyze your economic situation: total income per resource, tax rates, improvements, construction in progress, and recommendations for economic growth.", {}, async () => {
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
    const avgProsperity = provinces.reduce((s, p) => s + fp(p.prosperity), 0) /
        (provinces.length || 1);
    const highUnrest = provinces
        .filter((p) => fp(p.unrest) > 0.3)
        .map((p) => ({
        id: p.id,
        name: p.name,
        unrest: fp(p.unrest).toFixed(2),
        tax_rate: fp(p.tax_rate).toFixed(2),
    }));
    const constructing = provinces
        .filter((p) => (p.construction_queue || []).length > 0)
        .map((p) => ({
        province: p.name,
        building: p.construction_queue.map((c) => ({
            improvement: c[0],
            turns_left: c[1],
        })),
    }));
    // Provinces without key improvements
    const needFarmstead = provinces
        .filter((p) => !(p.improvements || []).includes("Farmstead"))
        .map((p) => p.name);
    const needMarket = provinces
        .filter((p) => !(p.improvements || []).includes("Market"))
        .map((p) => p.name);
    return {
        content: [
            {
                type: "text",
                text: JSON.stringify({
                    total_income: totals,
                    average_prosperity: avgProsperity.toFixed(2),
                    total_population: provinces.reduce((s, p) => s + (p.population || 0), 0),
                    high_unrest_provinces: highUnrest,
                    under_construction: constructing,
                    provinces_without_farmstead: needFarmstead.slice(0, 10),
                    provinces_without_market: needMarket.slice(0, 10),
                }, null, 2),
            },
        ],
    };
});
server.tool("find_weak_targets", "Find weakly defended enemy provinces adjacent to your territory — prime targets for conquest. Ranks by ease of capture (low garrison, low fortification, no defending army).", {}, async () => {
    const world = await getWorld();
    const faction = findMyFaction(world);
    const provinces = myProvinces(world, faction.id);
    const myProvIds = new Set(provinces.map((p) => p.id));
    // Find enemy provinces adjacent to my territory
    const neighborIds = new Set();
    for (const p of provinces) {
        for (const n of p.neighbors || []) {
            if (!myProvIds.has(n))
                neighborIds.add(n);
        }
    }
    const enemyArmyLocations = new Map();
    for (const a of world.armies || []) {
        if (a.owner_faction !== faction.id) {
            const existing = enemyArmyLocations.get(a.location) || 0;
            enemyArmyLocations.set(a.location, existing + armyPower(a.troops));
        }
    }
    const targets = [...neighborIds]
        .map((id) => {
        const prov = (world.provinces || []).find((p) => p.id === id);
        if (!prov)
            return null;
        const owner = (world.factions || []).find((f) => f.id === prov.controller);
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
        .sort((a, b) => a.total_defense_score - b.total_defense_score);
    return {
        content: [
            {
                type: "text",
                text: JSON.stringify({
                    weak_targets: targets.slice(0, 15),
                    note: "Sorted by total_defense_score (lowest = easiest to capture). Consider declaring war first if not already at war.",
                }, null, 2),
            },
        ],
    };
});
server.tool("analyze_characters", "Analyze your characters: who should be your heir, who should fill council roles, who has the best stats for each position, and which characters are potential plotters.", {}, async () => {
    const world = await getWorld();
    const faction = findMyFaction(world);
    const chars = myCharacters(world, faction.id);
    const ruler = chars.find((c) => c.role === "Ruler");
    const council = chars.filter((c) => ["Marshal", "Chaplain", "Steward", "Spymaster"].includes(c.role));
    const unassigned = chars.filter((c) => c.role === "Courtier" || c.role === "Duke");
    // Best candidate for each role
    const bestFor = (stat) => [...chars]
        .filter((c) => c.role !== "Ruler")
        .sort((a, b) => fp(b.stats?.[stat]) - fp(a.stats?.[stat]))
        .slice(0, 3)
        .map((c) => ({
        id: c.id,
        name: c.name,
        current_role: c.role,
        stat_value: fp(c.stats?.[stat]).toFixed(1),
        traits: c.traits,
    }));
    // Heir candidates (alive, adult or close to adult)
    const heirCandidates = chars
        .filter((c) => c.role !== "Ruler" && c.age >= 12)
        .sort((a, b) => fp(b.legitimacy) + fp(b.prestige) - (fp(a.legitimacy) + fp(a.prestige)))
        .slice(0, 5)
        .map((c) => ({
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
                type: "text",
                text: JSON.stringify({
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
                    current_council: council.map((c) => ({
                        role: c.role,
                        id: c.id,
                        name: c.name,
                        key_stat: fp(c.stats?.[c.role === "Marshal"
                            ? "martial"
                            : c.role === "Steward"
                                ? "stewardship"
                                : c.role === "Spymaster"
                                    ? "intrigue"
                                    : "learning"]).toFixed(1),
                    })),
                    best_marshal_candidates: bestFor("martial"),
                    best_steward_candidates: bestFor("stewardship"),
                    best_spymaster_candidates: bestFor("intrigue"),
                    best_chaplain_candidates: bestFor("learning"),
                    heir_candidates: heirCandidates,
                    unassigned_courtiers: unassigned.map((c) => ({
                        id: c.id,
                        name: c.name,
                        age: c.age,
                        traits: c.traits,
                    })),
                }, null, 2),
            },
        ],
    };
});
// ═══════════════════════════════════════════════════════════════════════════════
// ACTION TOOLS — MILITARY
// ═══════════════════════════════════════════════════════════════════════════════
server.tool("raise_army", "Raise a new army from a province's population. The province must be under your control.", { province_id: z.number().int().describe("Province ID to raise army from") }, async ({ province_id }) => {
    const result = await submitAction({ RaiseArmy: { province: province_id } });
    return { content: [{ type: "text", text: `Army raised from province ${province_id}.\n${result}` }] };
});
server.tool("move_army", "Move an army to a target province using the shortest BFS path. The army moves 1 province per turn. Use this for multi-hop pathfinding.", {
    army_id: z.number().int().describe("Army ID to move"),
    target_province: z.number().int().describe("Target province ID"),
}, async ({ army_id, target_province }) => {
    const result = await submitAction({
        MoveArmyPath: { army: army_id, target: target_province },
    });
    return {
        content: [
            {
                type: "text",
                text: `Army ${army_id} ordered to march to province ${target_province}.\n${result}`,
            },
        ],
    };
});
server.tool("disband_army", "Disband an army, returning troops to the province garrison.", { army_id: z.number().int().describe("Army ID to disband") }, async ({ army_id }) => {
    const result = await submitAction({ DisbandArmy: { army: army_id } });
    return { content: [{ type: "text", text: `Army ${army_id} disbanded.\n${result}` }] };
});
// ═══════════════════════════════════════════════════════════════════════════════
// ACTION TOOLS — DIPLOMACY
// ═══════════════════════════════════════════════════════════════════════════════
server.tool("declare_war", "Declare war on another faction. You need a casus belli (reason for war). Options: Conquest, HolyWar, Reconquest, Rebellion, Succession, Insult.", {
    target_faction: z.number().int().describe("Target faction ID (0-6)"),
    casus_belli: z
        .enum(["Conquest", "HolyWar", "Reconquest", "Rebellion", "Succession", "Insult"])
        .describe("Reason for war"),
}, async ({ target_faction, casus_belli }) => {
    const result = await submitAction({
        DeclareWar: { target: target_faction, casus_belli },
    });
    return {
        content: [
            {
                type: "text",
                text: `War declared on faction ${target_faction} (${casus_belli})!\n${result}`,
            },
        ],
    };
});
server.tool("propose_treaty", "Propose a treaty to another faction. Types: NonAggression, DefensiveAlliance, TradeAgreement, Marriage, Vassalization, WhitePeace, Surrender.", {
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
}, async ({ target_faction, treaty_type }) => {
    const result = await submitAction({
        ProposeTreaty: { target: target_faction, treaty: treaty_type },
    });
    return {
        content: [
            {
                type: "text",
                text: `Treaty proposed (${treaty_type}) to faction ${target_faction}.\n${result}`,
            },
        ],
    };
});
server.tool("accept_treaty", "Accept a treaty proposed by another faction.", {
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
}, async ({ from_faction, treaty_type }) => {
    const result = await submitAction({
        AcceptTreaty: { from: from_faction, treaty: treaty_type },
    });
    return {
        content: [
            {
                type: "text",
                text: `Treaty accepted (${treaty_type}) from faction ${from_faction}.\n${result}`,
            },
        ],
    };
});
// ═══════════════════════════════════════════════════════════════════════════════
// ACTION TOOLS — ECONOMY
// ═══════════════════════════════════════════════════════════════════════════════
server.tool("build_improvement", "Build an improvement in a province. Options: Farmstead, Mine, Lumbercamp, Quarry, Stables, Market, Temple, Fortification, University, Port, Granary, Hospital.", {
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
}, async ({ province_id, improvement }) => {
    const result = await submitAction({
        BuildImprovement: { province: province_id, improvement },
    });
    return {
        content: [
            {
                type: "text",
                text: `Building ${improvement} in province ${province_id}.\n${result}`,
            },
        ],
    };
});
server.tool("set_tax_rate", "Set the tax rate for a province. Rate is 0.0 to 1.0 (0% to 100%). Higher taxes = more gold but more unrest. Recommended: 0.3-0.5 for stable provinces, lower for high-unrest ones.", {
    province_id: z.number().int().describe("Province ID"),
    rate: z.number().min(0).max(1).describe("Tax rate (0.0 to 1.0)"),
}, async ({ province_id, rate }) => {
    // Convert to FixedPoint (×1000)
    const fixedRate = Math.round(rate * 1000);
    const result = await submitAction({
        SetTaxRate: { province: province_id, rate: { raw: fixedRate } },
    });
    return {
        content: [
            {
                type: "text",
                text: `Tax rate in province ${province_id} set to ${(rate * 100).toFixed(1)}%.\n${result}`,
            },
        ],
    };
});
server.tool("establish_trade_route", "Establish a trade route between two adjacent provinces you control. Both must be yours.", {
    from_province: z.number().int().describe("Source province ID"),
    to_province: z.number().int().describe("Destination province ID"),
}, async ({ from_province, to_province }) => {
    const result = await submitAction({
        EstablishTradeRoute: { from: from_province, to: to_province },
    });
    return {
        content: [
            {
                type: "text",
                text: `Trade route established: province ${from_province} ↔ ${to_province}.\n${result}`,
            },
        ],
    };
});
server.tool("disrupt_trade_route", "Disrupt (raid/blockade) an enemy trade route by its route ID.", { route_id: z.number().int().describe("Trade route ID to disrupt") }, async ({ route_id }) => {
    const result = await submitAction({ DisruptTradeRoute: { route_id } });
    return {
        content: [{ type: "text", text: `Trade route ${route_id} disrupted.\n${result}` }],
    };
});
// ═══════════════════════════════════════════════════════════════════════════════
// ACTION TOOLS — CHARACTERS & INTRIGUE
// ═══════════════════════════════════════════════════════════════════════════════
server.tool("assign_councilor", "Assign a character to a council role: Marshal (military), Steward (economy), Chaplain (religion), Spymaster (intrigue). Use analyze_characters first to find the best candidate.", {
    character_id: z.number().int().describe("Character ID to assign"),
    role: z
        .enum(["Marshal", "Chaplain", "Steward", "Spymaster"])
        .describe("Council role"),
}, async ({ character_id, role }) => {
    const result = await submitAction({
        AssignCouncilor: { character: character_id, role },
    });
    return {
        content: [
            {
                type: "text",
                text: `${role} position assigned to character ${character_id}.\n${result}`,
            },
        ],
    };
});
server.tool("designate_heir", "Designate a character as your heir for succession. Choose wisely — this affects legitimacy and stability.", { character_id: z.number().int().describe("Character ID to designate as heir") }, async ({ character_id }) => {
    const result = await submitAction({ DesignateHeir: { character: character_id } });
    return {
        content: [
            {
                type: "text",
                text: `Character ${character_id} designated as heir.\n${result}`,
            },
        ],
    };
});
server.tool("arrange_marriage", "Arrange a marriage between two characters. Great for diplomatic alliances and dynasty building.", {
    character_a: z.number().int().describe("First character ID"),
    character_b: z.number().int().describe("Second character ID"),
}, async ({ character_a, character_b }) => {
    const result = await submitAction({
        ArrangeMarriage: { a: character_a, b: character_b },
    });
    return {
        content: [
            {
                type: "text",
                text: `Marriage arranged between characters ${character_a} and ${character_b}.\n${result}`,
            },
        ],
    };
});
server.tool("convert_province", "Begin religious conversion of a province. Religions: OldFaith, EmberChurch, SaltCult, FrostSpirits, BlackOrder. Conversion takes several turns.", {
    province_id: z.number().int().describe("Province ID to convert"),
    religion: z
        .enum(["OldFaith", "EmberChurch", "SaltCult", "FrostSpirits", "BlackOrder"])
        .describe("Target religion"),
}, async ({ province_id, religion }) => {
    const result = await submitAction({
        ConvertProvince: { province: province_id, religion },
    });
    return {
        content: [
            {
                type: "text",
                text: `Converting province ${province_id} to ${religion}.\n${result}`,
            },
        ],
    };
});
server.tool("launch_plot", "Launch an intrigue plot against a target character. Types: Assassination (kill), Fabricate (claim province), Sabotage (reduce prosperity), Steal (take gold).", {
    target_character: z.number().int().describe("Character ID to target"),
    plot_type: z
        .enum(["Assassination", "Fabricate", "Sabotage", "Steal"])
        .describe("Type of plot"),
}, async ({ target_character, plot_type }) => {
    const result = await submitAction({
        LaunchPlot: { target: target_character, plot_type },
    });
    return {
        content: [
            {
                type: "text",
                text: `Plot launched: ${plot_type} against character ${target_character}.\n${result}`,
            },
        ],
    };
});
server.tool("back_plot", "Back (support) an existing intrigue plot to increase its progress speed.", { plot_id: z.number().int().describe("Plot ID to back") }, async ({ plot_id }) => {
    const result = await submitAction({ BackPlot: { plot_id } });
    return {
        content: [{ type: "text", text: `Plot ${plot_id} backed.\n${result}` }],
    };
});
server.tool("investigate_plots", "Use a spymaster to investigate and detect enemy intrigue plots targeting your realm.", {
    spymaster_id: z
        .number()
        .int()
        .describe("Character ID of your spymaster to use"),
}, async ({ spymaster_id }) => {
    const result = await submitAction({ InvestigatePlot: { spymaster: spymaster_id } });
    return {
        content: [
            {
                type: "text",
                text: `Spymaster ${spymaster_id} investigating plots...\n${result}`,
            },
        ],
    };
});
// ═══════════════════════════════════════════════════════════════════════════════
// COMPOSITE STRATEGY TOOL
// ═══════════════════════════════════════════════════════════════════════════════
server.tool("strategic_briefing", "Get a comprehensive strategic briefing: your realm summary, threat analysis, economic status, diplomatic situation, and recommended priorities. This is your go-to tool for deciding what to do each turn.", {}, async () => {
    const world = await getWorld();
    const faction = findMyFaction(world);
    const provinces = myProvinces(world, faction.id);
    const armies = myArmies(world, faction.id);
    const chars = myCharacters(world, faction.id);
    const myProvIds = new Set(provinces.map((p) => p.id));
    // Military strength
    const myTotalPower = armies.reduce((s, a) => s + armyPower(a.troops), 0);
    // Per-faction power comparison
    const factionPowers = (world.factions || [])
        .filter((f) => f.alive && f.id !== faction.id)
        .map((f) => {
        const fArmies = (world.armies || []).filter((a) => a.owner_faction === f.id);
        const fProvs = (world.provinces || []).filter((p) => p.controller === f.id);
        return {
            name: f.name,
            id: f.id,
            provinces: fProvs.length,
            military_power: fArmies.reduce((s, a) => s + armyPower(a.troops), 0),
            at_war_with_us: (world.diplomacy || []).some((d) => d.at_war &&
                ((d.faction_a === faction.id && d.faction_b === f.id) ||
                    (d.faction_b === faction.id && d.faction_a === f.id))),
        };
    })
        .sort((a, b) => b.military_power - a.military_power);
    // Threats
    const enemiesNearBorder = (world.armies || []).filter((a) => a.owner_faction !== faction.id &&
        provinces.some((p) => (p.neighbors || []).includes(a.location) || a.location === p.id)).length;
    // Economy
    const totalGold = provinces.reduce((s, p) => s + fp(p.resources?.gold || 0), 0);
    // Recommendations
    const recommendations = [];
    if (armies.length === 0) {
        recommendations.push("URGENT: You have no armies! Raise one immediately from your most populated province.");
    }
    const idleArmies = armies.filter((a) => !a.destination && !a.movement_queue?.length && !a.siege);
    if (idleArmies.length > 0) {
        recommendations.push(`${idleArmies.length} army(ies) idle — consider moving them to border provinces or attacking.`);
    }
    if (enemiesNearBorder > 0) {
        recommendations.push(`${enemiesNearBorder} enemy army(ies) near your borders — consider defensive positioning.`);
    }
    const noImprovements = provinces.filter((p) => (p.improvements || []).length === 0);
    if (noImprovements.length > 0) {
        recommendations.push(`${noImprovements.length} province(s) have no improvements — build Farmsteads for food and Markets for gold.`);
    }
    const ruler = chars.find((c) => c.role === "Ruler");
    if (ruler && !ruler.heir) {
        recommendations.push("Your ruler has no designated heir! Use designate_heir to prevent succession crisis.");
    }
    const atWar = factionPowers.filter((f) => f.at_war_with_us);
    if (atWar.length > 0) {
        recommendations.push(`Currently at war with: ${atWar.map((f) => f.name).join(", ")}. Focus military efforts!`);
    }
    return {
        content: [
            {
                type: "text",
                text: JSON.stringify({
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
                }, null, 2),
            },
        ],
    };
});
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
