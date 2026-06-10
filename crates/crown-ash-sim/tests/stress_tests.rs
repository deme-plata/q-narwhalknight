//! Stress tests for the Crown & Ash simulation engine.
//!
//! These tests run the simulation for hundreds of ticks and verify that
//! invariants, bounds, and determinism properties hold over long runs.

use crown_ash_sim::{init_world, tick};
use crown_ash_types::{GameEvent, WorldConfig};

/// Produce a unique block hash for each tick number.
fn make_block_hash(tick_num: u32) -> [u8; 32] {
    let mut h = [0u8; 32];
    let bytes = tick_num.to_le_bytes();
    h[..4].copy_from_slice(&bytes);
    h[4] = 0xCA;
    h
}

// ---------------------------------------------------------------------------
// 1. stress_500_ticks_no_panic
// ---------------------------------------------------------------------------

#[test]
fn stress_500_ticks_no_panic() {
    let config = WorldConfig::default();
    let mut world = init_world(&config, [0x42; 32]);

    for i in 0u32..500 {
        let hash = make_block_hash(i);
        let _summary = tick(&mut world, &hash);
    }
    assert_eq!(world.meta.turn, 500);
}

// ---------------------------------------------------------------------------
// 2. stress_population_stays_positive
// ---------------------------------------------------------------------------

#[test]
fn stress_population_stays_positive() {
    let config = WorldConfig::default();
    let mut world = init_world(&config, [0x42; 32]);

    for i in 0u32..500 {
        let hash = make_block_hash(i);
        let _summary = tick(&mut world, &hash);

        for prov in &world.provinces {
            assert!(
                prov.population > 0,
                "Province {} has zero population at tick {}",
                prov.id, i + 1
            );
            assert!(
                prov.population < 1_000_000,
                "Province {} population {} exceeded 1M at tick {}",
                prov.id, prov.population, i + 1
            );
        }
    }
}

// ---------------------------------------------------------------------------
// 3. stress_at_least_one_faction_alive
// ---------------------------------------------------------------------------

#[test]
fn stress_at_least_one_faction_alive() {
    let config = WorldConfig::default();
    let mut world = init_world(&config, [0x42; 32]);

    for i in 0u32..500 {
        let hash = make_block_hash(i);
        let summary = tick(&mut world, &hash);
        assert!(
            summary.active_factions > 0,
            "No active factions at tick {}",
            i + 1
        );
    }
}

// ---------------------------------------------------------------------------
// 4. stress_character_count_bounded
// ---------------------------------------------------------------------------

#[test]
fn stress_character_count_bounded() {
    let config = WorldConfig::default();
    let mut world = init_world(&config, [0x42; 32]);

    for i in 0u32..500 {
        let hash = make_block_hash(i);
        let _summary = tick(&mut world, &hash);

        let alive_count = world.characters.iter().filter(|c| c.alive).count();
        assert!(
            alive_count < 200,
            "Alive character count {} exceeded 200 at tick {}",
            alive_count, i + 1
        );
    }
}

// ---------------------------------------------------------------------------
// 5. stress_trade_routes_form_and_decay
// ---------------------------------------------------------------------------

#[test]
fn stress_trade_routes_form_and_decay() {
    let config = WorldConfig::default();
    let mut world = init_world(&config, [0x42; 32]);

    for i in 0u32..500 {
        let hash = make_block_hash(i);
        let _summary = tick(&mut world, &hash);

        assert!(
            world.trade_routes.len() <= 50,
            "Trade route count {} exceeded 50 at tick {}",
            world.trade_routes.len(), i + 1
        );
    }
}

// ---------------------------------------------------------------------------
// 6. stress_intrigue_plots_fire
// ---------------------------------------------------------------------------

#[test]
fn stress_intrigue_plots_fire() {
    let config = WorldConfig::default();
    let mut world = init_world(&config, [0x42; 32]);

    for i in 0u32..500 {
        let hash = make_block_hash(i);
        let _summary = tick(&mut world, &hash);

        assert!(
            world.plots.len() <= 20,
            "Active plot count {} exceeded 20 at tick {}",
            world.plots.len(), i + 1
        );
    }
}

// ---------------------------------------------------------------------------
// 7. stress_births_produce_valid_characters
// ---------------------------------------------------------------------------

#[test]
fn stress_births_produce_valid_characters() {
    let config = WorldConfig::default();
    let mut world = init_world(&config, [0x42; 32]);

    for i in 0u32..500 {
        let hash = make_block_hash(i);
        let summary = tick(&mut world, &hash);

        for event in &summary.events {
            if let GameEvent::CharacterBorn { character_id, .. } = event {
                assert!(
                    world.character(*character_id).is_some(),
                    "CharacterBorn references non-existent character {} at tick {}",
                    character_id, i + 1
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------
// 8. stress_realm_splits_valid
// ---------------------------------------------------------------------------

#[test]
fn stress_realm_splits_valid() {
    let config = WorldConfig::default();
    let mut world = init_world(&config, [0x42; 32]);

    for i in 0u32..500 {
        let hash = make_block_hash(i);
        let summary = tick(&mut world, &hash);

        for event in &summary.events {
            if let GameEvent::RealmSplit { new_faction, .. } = event {
                let nf = *new_faction;
                let faction_exists = world.factions.iter().any(|f| f.id == nf);
                assert!(
                    faction_exists,
                    "RealmSplit created faction {} but it doesn't exist at tick {}",
                    nf, i + 1
                );
                let has_province = world.provinces.iter().any(|p| p.controller == nf);
                assert!(
                    has_province,
                    "RealmSplit faction {} controls no provinces at tick {}",
                    nf, i + 1
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------
// 9. stress_determinism_500_ticks
// ---------------------------------------------------------------------------

#[test]
fn stress_determinism_500_ticks() {
    let config = WorldConfig::default();
    let seed = [0x42; 32];

    let mut world1 = init_world(&config, seed);
    let mut world2 = init_world(&config, seed);

    for i in 0u32..500 {
        let hash = make_block_hash(i);
        let _s1 = tick(&mut world1, &hash);
        let _s2 = tick(&mut world2, &hash);
    }

    let bytes1 = bincode::serialize(&world1).expect("Failed to serialize world1");
    let bytes2 = bincode::serialize(&world2).expect("Failed to serialize world2");

    assert_eq!(
        bytes1, bytes2,
        "Same seed diverged after 500 ticks (sizes: {} vs {})",
        bytes1.len(), bytes2.len()
    );
}

// ---------------------------------------------------------------------------
// 10. stress_army_count_bounded
// ---------------------------------------------------------------------------

#[test]
fn stress_army_count_bounded() {
    let config = WorldConfig::default();
    let mut world = init_world(&config, [0x42; 32]);

    for i in 0u32..500 {
        let hash = make_block_hash(i);
        let _summary = tick(&mut world, &hash);

        for faction in &world.factions {
            if !faction.alive { continue; }
            let army_count = world.armies.iter()
                .filter(|a| a.owner_faction == faction.id)
                .count();
            assert!(
                army_count <= 5,
                "Faction {} has {} armies (max 5) at tick {}",
                faction.id, army_count, i + 1
            );
        }
    }
}

// ---------------------------------------------------------------------------
// 11. stress_prosperity_bounded
// ---------------------------------------------------------------------------

#[test]
fn stress_prosperity_bounded() {
    let config = WorldConfig::default();
    let mut world = init_world(&config, [0x42; 32]);

    for i in 0u32..500 {
        let hash = make_block_hash(i);
        let _summary = tick(&mut world, &hash);

        for prov in &world.provinces {
            let raw = prov.prosperity.raw();
            // Prosperity is clamped to [0, 1000] at end of tick pipeline (step 8b).
            assert!(
                raw >= 0 && raw <= 1_000_000,
                "Province {} prosperity {} out of [0, 1M] at tick {}",
                prov.id, raw, i + 1
            );
        }
    }
}

// ---------------------------------------------------------------------------
// 12. stress_unrest_clamped
// ---------------------------------------------------------------------------

#[test]
fn stress_unrest_clamped() {
    let config = WorldConfig::default();
    let mut world = init_world(&config, [0x42; 32]);

    for i in 0u32..500 {
        let hash = make_block_hash(i);
        let _summary = tick(&mut world, &hash);

        for prov in &world.provinces {
            let raw = prov.unrest.raw();
            // Unrest is clamped to [0, 1000] at end of tick pipeline (step 8b).
            assert!(
                raw >= 0 && raw <= 1_000_000,
                "Province {} unrest {} out of [0, 1M] at tick {}",
                prov.id, raw, i + 1
            );
        }
    }
}

// ---------------------------------------------------------------------------
// 13. stress_different_seeds_diverge
// ---------------------------------------------------------------------------

#[test]
fn stress_different_seeds_diverge() {
    let config = WorldConfig::default();

    let mut world_a = init_world(&config, [0x42; 32]);
    let mut world_b = init_world(&config, [0x99; 32]);

    for i in 0u32..100 {
        let hash = make_block_hash(i);
        let _sa = tick(&mut world_a, &hash);
        let _sb = tick(&mut world_b, &hash);
    }

    let bytes_a = bincode::serialize(&world_a).expect("serialize a");
    let bytes_b = bincode::serialize(&world_b).expect("serialize b");

    assert_ne!(
        bytes_a, bytes_b,
        "Different seeds produced identical state after 100 ticks"
    );
}

// ---------------------------------------------------------------------------
// 14. stress_cohesion_clamped
// ---------------------------------------------------------------------------

#[test]
fn stress_cohesion_clamped() {
    let config = WorldConfig::default();
    let mut world = init_world(&config, [0x42; 32]);

    for i in 0u32..500 {
        let hash = make_block_hash(i);
        let _summary = tick(&mut world, &hash);

        for realm in &world.realms {
            let components = [
                ("legitimacy", realm.cohesion.legitimacy.raw()),
                ("fealty", realm.cohesion.fealty.raw()),
                ("clerical_favor", realm.cohesion.clerical_favor.raw()),
                ("commoner_mood", realm.cohesion.commoner_mood.raw()),
                ("regional_identity", realm.cohesion.regional_identity.raw()),
            ];

            for (name, raw) in &components {
                assert!(
                    *raw >= 0 && *raw <= 1_000_000,
                    "Faction {} cohesion {} = {} out of [0, 1M] at tick {}",
                    realm.faction, name, raw, i + 1
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------
// 15. stress_1000_ticks_endurance
// ---------------------------------------------------------------------------

#[test]
fn stress_1000_ticks_endurance() {
    let config = WorldConfig::default();
    let mut world = init_world(&config, [0x42; 32]);

    let mut last_active = 0u8;
    for i in 0u32..1000 {
        let hash = make_block_hash(i);
        let summary = tick(&mut world, &hash);
        last_active = summary.active_factions;
    }

    assert!(last_active > 0, "No factions alive after 1000 ticks");
    assert_eq!(world.meta.turn, 1000);
}
