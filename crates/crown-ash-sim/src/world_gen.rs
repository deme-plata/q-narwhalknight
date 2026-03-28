//! World generation — creates the initial game state.
//!
//! Generates 25 provinces with fixed adjacency, 7 factions, ~30 starting characters
//! (ruler + 4 councillors per faction, plus courtiers), and initial diplomatic relations.

use crown_ash_types::{
    Army, Character, CharacterRole, CharacterStats, DiplomaticRelation, Dynasty,
    Faction, FixedPoint, Province, Realm, RealmCohesion, Trait, WorldConfig, WorldMeta,
};
use crown_ash_types::faction::default_faction_templates;
use crown_ash_types::province::{Improvement, Resources, Troops};
use crown_ash_types::dynasty::SuccessionRule;

use crate::map::PROVINCE_DATA;
use crate::random::DeterministicRng;
use crate::world_state::GameWorld;

/// Character name pools per faction culture (10 names each to choose from).
const IMPERIAL_NAMES: &[&str] = &[
    "Aldric", "Cedric", "Elowen", "Theron", "Giselle",
    "Hadrian", "Isolde", "Roderic", "Seraphina", "Valentin",
];
const FEUDAL_NAMES: &[&str] = &[
    "Gareth", "Lysandra", "Baldwin", "Rowena", "Percival",
    "Elara", "Godfrey", "Matilda", "Tristan", "Beatrice",
];
const CLERICAL_NAMES: &[&str] = &[
    "Alaric", "Cassandra", "Dorian", "Evangeline", "Ignatius",
    "Mirabel", "Peregrine", "Theodora", "Ambrose", "Celestine",
];
const MERCANTILE_NAMES: &[&str] = &[
    "Lorenzo", "Vivienne", "Prospero", "Daria", "Silvius",
    "Ophelia", "Cassius", "Julianna", "Renato", "Isabetta",
];
const NORDIC_NAMES: &[&str] = &[
    "Bjorn", "Astrid", "Gunnar", "Sigrid", "Eirik",
    "Freya", "Ragnar", "Thyra", "Leif", "Ingrid",
];
const NOMADIC_NAMES: &[&str] = &[
    "Temur", "Altani", "Borte", "Kublai", "Yesugei",
    "Mandukhai", "Batu", "Sorghaghtani", "Ariq", "Toregene",
];
const MONASTIC_NAMES: &[&str] = &[
    "Corvus", "Nocturna", "Vesper", "Morrigan", "Cinis",
    "Tenebris", "Silentio", "Umbra", "Ashen", "Griselda",
];

/// Get name pool for a given faction ID.
fn name_pool(faction_id: u8) -> &'static [&'static str] {
    match faction_id {
        0 => IMPERIAL_NAMES,
        1 => FEUDAL_NAMES,
        2 => CLERICAL_NAMES,
        3 => MERCANTILE_NAMES,
        4 => NORDIC_NAMES,
        5 => NOMADIC_NAMES,
        6 => MONASTIC_NAMES,
        _ => IMPERIAL_NAMES,
    }
}

/// Dynasty name pools per faction.
const DYNASTY_NAMES: [&str; 7] = [
    "Ashborne",    // Ashen Crown
    "Thornfeld",   // Vale Princes
    "Pyreheart",   // Ember Church
    "Silvermark",  // Salt League
    "Winterfang",  // Frost Marches
    "Stormrider",  // Red Steppe
    "Veilshadow",  // Black Abbey
];

/// Initialise a new game world from a config and seed.
pub fn init_world(config: &WorldConfig, seed: [u8; 32]) -> GameWorld {
    let mut rng = DeterministicRng::new(seed, "worldgen");
    let templates = default_faction_templates();

    let mut next_character_id: u32 = 0;
    let mut characters: Vec<Character> = Vec::new();
    let mut dynasties: Vec<Dynasty> = Vec::new();
    let mut factions: Vec<Faction> = Vec::new();
    let mut realms: Vec<Realm> = Vec::new();
    let armies: Vec<Army> = Vec::new();

    // -----------------------------------------------------------------------
    // 1. Generate provinces from the fixed map.
    // -----------------------------------------------------------------------
    let mut provinces: Vec<Province> = Vec::with_capacity(config.province_count as usize);
    for data in PROVINCE_DATA.iter().take(config.province_count as usize) {
        let base_pop = config.starting_population_per_province;
        // Vary population slightly per province (+/- 20%).
        let pop_var = rng.range(-200, 200);
        let pop = ((base_pop as i64 * (1000 + pop_var)) / 1000).max(1000) as u32;

        let resources = starting_resources(data.terrain);
        provinces.push(Province {
            id: data.id,
            name: data.name.to_string(),
            terrain: data.terrain,
            controller: data.starting_faction,
            population: pop,
            prosperity: FixedPoint::from_int(500),
            unrest: FixedPoint::from_int(100),
            fortification: starting_fort(data.terrain),
            religion: templates[data.starting_faction as usize].religion,
            culture: templates[data.starting_faction as usize].culture,
            resources,
            garrison: Troops { levy: 200, men_at_arms: 50, knights: 5 },
            improvements: starting_improvements(data.terrain),
            construction_queue: Vec::new(),
            scars: Vec::new(),
            grudges: Vec::new(),
            last_famine_turn: None,
            last_siege_turn: None,
            tax_rate: FixedPoint::from_raw(200), // 20% default
            neighbors: data.neighbors.to_vec(),
            conversion_progress: None,
        });
    }

    // -----------------------------------------------------------------------
    // 2. Generate factions, dynasties, and characters.
    // -----------------------------------------------------------------------
    for (idx, tmpl) in templates.iter().enumerate().take(config.faction_count as usize) {
        let faction_id = idx as u8;

        // -- Dynasty --
        let dynasty_id = idx as u16;
        let founder_id = next_character_id;

        // -- Ruler --
        let ruler_age = rng.range(25, 55) as u8;
        let ruler_name = pick_name(&mut rng, faction_id, 0);
        let ruler = Character {
            id: founder_id,
            name: ruler_name,
            dynasty: dynasty_id,
            faction: faction_id,
            role: CharacterRole::Ruler,
            age: ruler_age,
            alive: true,
            traits: random_traits(&mut rng, 3),
            stats: random_stats(&mut rng),
            health: FixedPoint::from_int(800),
            legitimacy: tmpl.bonuses.legitimacy,
            prestige: FixedPoint::from_int(500),
            heir: None,
            spouse: None,
            children: Vec::new(),
            relations: Vec::new(),
            parent: None,
            death_turn: None,
            death_cause: None,
        };
        characters.push(ruler);
        next_character_id += 1;

        // -- 4 Councillors --
        let council_roles = [
            CharacterRole::Marshal,
            CharacterRole::Chaplain,
            CharacterRole::Steward,
            CharacterRole::Spymaster,
        ];
        let mut member_ids = vec![founder_id];
        for (ri, &role) in council_roles.iter().enumerate() {
            let cid = next_character_id;
            let age = rng.range(20, 50) as u8;
            let cname = pick_name(&mut rng, faction_id, (ri + 1) as u32);
            let c = Character {
                id: cid,
                name: cname,
                dynasty: dynasty_id,
                faction: faction_id,
                role,
                age,
                alive: true,
                traits: random_traits(&mut rng, 2),
                stats: random_stats(&mut rng),
                health: FixedPoint::from_int(700),
                legitimacy: FixedPoint::from_int(300),
                prestige: FixedPoint::from_int(200),
                heir: None,
                spouse: None,
                children: Vec::new(),
                relations: Vec::new(),
                parent: None,
                death_turn: None,
                death_cause: None,
            };
            characters.push(c);
            member_ids.push(cid);
            next_character_id += 1;
        }

        // Designate the first councillor child (marshal) as heir.
        if let Some(ruler_char) = characters.iter_mut().find(|c| c.id == founder_id) {
            ruler_char.heir = Some(founder_id + 1);
        }

        // -- Dynasty record --
        dynasties.push(Dynasty {
            id: dynasty_id,
            name: DYNASTY_NAMES[idx].to_string(),
            founder: founder_id,
            succession_rule: match faction_id {
                2 => SuccessionRule::Elective,        // Ember Church uses elections
                5 => SuccessionRule::TrialByCombat,   // Red Steppe combat
                _ => SuccessionRule::Primogeniture,
            },
            prestige: 0,
            members: member_ids,
            founded_turn: 0,
        });

        // -- Faction --
        factions.push(Faction {
            id: faction_id,
            name: tmpl.name.clone(),
            alive: true,
            culture: tmpl.culture,
            religion: tmpl.religion,
            bonuses: tmpl.bonuses.clone(),
            color_rgb: tmpl.color_rgb,
            player_wallet: None,
        });

        // -- Realm --
        let owned_provinces: Vec<u16> = provinces
            .iter()
            .filter(|p| p.controller == faction_id)
            .map(|p| p.id)
            .collect();

        realms.push(Realm {
            owner_wallet: String::new(),
            faction: faction_id,
            ruler: founder_id,
            provinces: owned_provinces,
            vassals: Vec::new(),
            treasury: FixedPoint::from_raw(config.starting_treasury),
            cohesion: RealmCohesion::default(),
            age: 0,
            at_war_with: Vec::new(),
            allies: Vec::new(),
            religious_authority: FixedPoint::from_int(500),
        });
    }

    // -----------------------------------------------------------------------
    // 3. Diplomatic relations (all pairs start neutral).
    // -----------------------------------------------------------------------
    let mut diplomacy = Vec::new();
    for a in 0..config.faction_count {
        for b in (a + 1)..config.faction_count {
            diplomacy.push(DiplomaticRelation::new(a, b));
        }
    }

    // -----------------------------------------------------------------------
    // 4. No starting armies (only garrisons).
    // -----------------------------------------------------------------------
    let _ = &armies; // armies start empty

    // -----------------------------------------------------------------------
    // 5. Assemble world.
    // -----------------------------------------------------------------------
    GameWorld {
        meta: WorldMeta {
            turn: 0,
            genesis_block: 0,
            player_count: 0,
            initialized: true,
            world_version: 1,
            sim_version: crown_ash_types::SIM_VERSION.to_string(),
        },
        provinces,
        characters,
        factions,
        realms,
        armies,
        dynasties,
        diplomacy,
        action_queue: Vec::new(),
        plots: Vec::new(),
        trade_routes: Vec::new(),
        tombstones: Vec::new(),
        next_character_id,
        next_army_id: 0,
        next_plot_id: 0,
        next_trade_route_id: 0,
        dirty: Default::default(),
    }
}

// ---------------------------------------------------------------------------
// Helper functions
// ---------------------------------------------------------------------------

/// Pick a deterministic name from the faction's pool.
fn pick_name(rng: &mut DeterministicRng, faction_id: u8, offset: u32) -> String {
    let pool = name_pool(faction_id);
    let idx = (rng.next_u32().wrapping_add(offset)) as usize % pool.len();
    pool[idx].to_string()
}

/// Random character stats (each 3..15 as FixedPoint from_int).
fn random_stats(rng: &mut DeterministicRng) -> CharacterStats {
    CharacterStats {
        martial: FixedPoint::from_int(rng.range(3, 15)),
        diplomacy: FixedPoint::from_int(rng.range(3, 15)),
        stewardship: FixedPoint::from_int(rng.range(3, 15)),
        intrigue: FixedPoint::from_int(rng.range(3, 15)),
        learning: FixedPoint::from_int(rng.range(3, 15)),
    }
}

/// Pick `count` random traits, avoiding duplicates.
fn random_traits(rng: &mut DeterministicRng, count: usize) -> Vec<Trait> {
    let all_traits = [
        Trait::Brave, Trait::Just, Trait::Pious, Trait::Temperate,
        Trait::Kind, Trait::Diligent, Trait::Patient, Trait::Honest,
        Trait::Craven, Trait::Cruel, Trait::Cynical, Trait::Gluttonous,
        Trait::Wrathful, Trait::Slothful, Trait::Impatient, Trait::Deceitful,
        Trait::Ambitious, Trait::Content, Trait::Gregarious, Trait::Shy,
        Trait::Paranoid, Trait::Trusting,
        Trait::Strategist, Trait::Administrator, Trait::Theologian,
        Trait::Schemer, Trait::Scholar,
    ];
    let mut picked = Vec::with_capacity(count);
    for _ in 0..count {
        let attempts = 20; // avoid infinite loop
        for _ in 0..attempts {
            let idx = rng.next_u32() as usize % all_traits.len();
            let t = all_traits[idx];
            if !picked.contains(&t) {
                picked.push(t);
                break;
            }
        }
    }
    picked
}

/// Starting resources based on terrain type.
fn starting_resources(terrain: crown_ash_types::Terrain) -> Resources {
    use crown_ash_types::Terrain::*;
    match terrain {
        Plains => Resources {
            food: FixedPoint::from_int(300),
            gold: FixedPoint::from_int(100),
            iron: FixedPoint::from_int(20),
            timber: FixedPoint::from_int(50),
            stone: FixedPoint::from_int(30),
            horses: FixedPoint::from_int(40),
            trade_goods: FixedPoint::from_int(30),
        },
        Hills => Resources {
            food: FixedPoint::from_int(150),
            gold: FixedPoint::from_int(80),
            iron: FixedPoint::from_int(100),
            timber: FixedPoint::from_int(40),
            stone: FixedPoint::from_int(120),
            horses: FixedPoint::from_int(20),
            trade_goods: FixedPoint::from_int(40),
        },
        Mountains => Resources {
            food: FixedPoint::from_int(50),
            gold: FixedPoint::from_int(200),
            iron: FixedPoint::from_int(200),
            timber: FixedPoint::from_int(20),
            stone: FixedPoint::from_int(200),
            horses: FixedPoint::from_int(5),
            trade_goods: FixedPoint::from_int(60),
        },
        Forest => Resources {
            food: FixedPoint::from_int(200),
            gold: FixedPoint::from_int(50),
            iron: FixedPoint::from_int(30),
            timber: FixedPoint::from_int(300),
            stone: FixedPoint::from_int(20),
            horses: FixedPoint::from_int(10),
            trade_goods: FixedPoint::from_int(40),
        },
        Marsh => Resources {
            food: FixedPoint::from_int(120),
            gold: FixedPoint::from_int(30),
            iron: FixedPoint::from_int(10),
            timber: FixedPoint::from_int(60),
            stone: FixedPoint::from_int(10),
            horses: FixedPoint::from_int(5),
            trade_goods: FixedPoint::from_int(20),
        },
        Desert => Resources {
            food: FixedPoint::from_int(60),
            gold: FixedPoint::from_int(150),
            iron: FixedPoint::from_int(40),
            timber: FixedPoint::from_int(5),
            stone: FixedPoint::from_int(80),
            horses: FixedPoint::from_int(60),
            trade_goods: FixedPoint::from_int(100),
        },
        Coastal => Resources {
            food: FixedPoint::from_int(250),
            gold: FixedPoint::from_int(120),
            iron: FixedPoint::from_int(20),
            timber: FixedPoint::from_int(40),
            stone: FixedPoint::from_int(30),
            horses: FixedPoint::from_int(10),
            trade_goods: FixedPoint::from_int(150),
        },
        River => Resources {
            food: FixedPoint::from_int(280),
            gold: FixedPoint::from_int(80),
            iron: FixedPoint::from_int(25),
            timber: FixedPoint::from_int(80),
            stone: FixedPoint::from_int(40),
            horses: FixedPoint::from_int(30),
            trade_goods: FixedPoint::from_int(60),
        },
    }
}

/// Starting fortification level based on terrain.
fn starting_fort(terrain: crown_ash_types::Terrain) -> u16 {
    use crown_ash_types::Terrain::*;
    match terrain {
        Mountains => 3,
        Hills => 2,
        Forest | Marsh => 1,
        _ => 0,
    }
}

/// Starting improvements based on terrain.
fn starting_improvements(terrain: crown_ash_types::Terrain) -> Vec<Improvement> {
    use crown_ash_types::Terrain::*;
    match terrain {
        Plains | River => vec![Improvement::Farmstead],
        Hills | Mountains => vec![Improvement::Mine],
        Forest => vec![Improvement::Lumbercamp],
        Coastal => vec![Improvement::Port],
        Desert => vec![Improvement::Market],
        Marsh => vec![],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generates_correct_province_count() {
        let config = WorldConfig::default();
        let world = init_world(&config, [0u8; 32]);
        assert_eq!(world.provinces.len(), 25);
    }

    #[test]
    fn generates_7_factions() {
        let config = WorldConfig::default();
        let world = init_world(&config, [0u8; 32]);
        assert_eq!(world.factions.len(), 7);
    }

    #[test]
    fn generates_characters() {
        let config = WorldConfig::default();
        let world = init_world(&config, [0u8; 32]);
        // 7 factions * 5 characters each = 35
        assert_eq!(world.characters.len(), 35);
    }

    #[test]
    fn each_faction_has_ruler() {
        let config = WorldConfig::default();
        let world = init_world(&config, [0u8; 32]);
        for f in 0..7u8 {
            let rulers: Vec<_> = world.characters.iter()
                .filter(|c| c.faction == f && c.role == CharacterRole::Ruler)
                .collect();
            assert_eq!(rulers.len(), 1, "Faction {} should have exactly 1 ruler", f);
        }
    }

    #[test]
    fn deterministic_generation() {
        let config = WorldConfig::default();
        let w1 = init_world(&config, [42u8; 32]);
        let w2 = init_world(&config, [42u8; 32]);
        // Same seed → same character names
        assert_eq!(w1.characters.len(), w2.characters.len());
        for (a, b) in w1.characters.iter().zip(w2.characters.iter()) {
            assert_eq!(a.name, b.name);
            assert_eq!(a.age, b.age);
        }
    }

    #[test]
    fn diplomacy_pairs() {
        let config = WorldConfig::default();
        let world = init_world(&config, [0u8; 32]);
        // 7 choose 2 = 21 pairs
        assert_eq!(world.diplomacy.len(), 21);
    }
}
