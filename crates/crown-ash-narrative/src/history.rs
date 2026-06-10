//! Province & Faction History — rich narrative summaries of accumulated events.
//!
//! When a player clicks a province or faction in the detail panel, they see
//! a prose history: who ruled it, what battles were fought, plagues endured,
//! improvements built, and hands it changed. Built from event history using
//! templates — no LLM required for basic history text.
//!
//! # Design
//!
//! History is aggregated from events, not stored incrementally. The caller
//! passes all events relevant to a province or faction, and this module
//! produces a readable narrative summary.

use crown_ash_types::GameEvent;

use crate::WorldContext;

// ─── Province History ───────────────────────────────────────────────────────

/// Aggregated province history statistics, used to generate prose.
#[derive(Debug, Clone, Default)]
struct ProvinceStats {
    times_conquered: u32,
    controllers: Vec<(u8, u32)>, // (faction_id, turn)
    battles_fought: u32,
    total_casualties: u32,
    plagues: u32,
    famines: u32,
    rebellions: u32,
    improvements_built: Vec<String>,
    sieges: u32,
    trade_routes: u32,
    miracles: u32,
    heresies: u32,
}

/// Generate a prose history for a province from its accumulated events.
///
/// Returns 2-5 paragraphs describing the province's story.
pub fn province_history(
    province_id: u16,
    province_name: &str,
    events: &[GameEvent],
    ctx: &WorldContext,
) -> String {
    let mut stats = ProvinceStats::default();

    for event in events {
        match event {
            GameEvent::ProvinceConquered { province, new_controller, turn, .. } => {
                if *province == province_id {
                    stats.times_conquered += 1;
                    stats.controllers.push((*new_controller, *turn));
                }
            }
            GameEvent::Battle(r) => {
                if r.province == province_id {
                    stats.battles_fought += 1;
                    stats.total_casualties += r.attacker_casualties + r.defender_casualties;
                }
            }
            GameEvent::PlagueOutbreak { province, .. } if *province == province_id => {
                stats.plagues += 1;
            }
            GameEvent::Famine { province, .. } if *province == province_id => {
                stats.famines += 1;
            }
            GameEvent::Rebellion { province, .. } if *province == province_id => {
                stats.rebellions += 1;
            }
            GameEvent::ConstructionComplete { province, improvement, .. } if *province == province_id => {
                stats.improvements_built.push(improvement.clone());
            }
            GameEvent::SiegeStarted { province, .. } if *province == province_id => {
                stats.sieges += 1;
            }
            GameEvent::TradeRouteEstablished { from, to, .. }
                if *from == province_id || *to == province_id =>
            {
                stats.trade_routes += 1;
            }
            GameEvent::Miracle { province, .. } if *province == province_id => {
                stats.miracles += 1;
            }
            GameEvent::Heresy { province, .. } if *province == province_id => {
                stats.heresies += 1;
            }
            _ => {}
        }
    }

    render_province_history(province_name, &stats, ctx)
}

fn render_province_history(name: &str, stats: &ProvinceStats, ctx: &WorldContext) -> String {
    let mut paragraphs: Vec<String> = Vec::new();

    // Opening
    if stats.times_conquered == 0 && stats.battles_fought == 0 {
        paragraphs.push(format!(
            "{} has known peace throughout recorded history. No invader's boot \
             has marred its soil, no siege engine has tested its walls.",
            name
        ));
    } else {
        paragraphs.push(format!(
            "The history of {} is written in blood and ash.", name
        ));
    }

    // Conquest history
    if stats.times_conquered > 0 {
        let controller_text: Vec<String> = stats.controllers.iter()
            .map(|(fid, turn)| format!("{} (turn {})", ctx.faction_name(*fid), turn))
            .collect();

        paragraphs.push(format!(
            "The province has changed hands {} time{}. It has been ruled by: {}.",
            stats.times_conquered,
            if stats.times_conquered == 1 { "" } else { "s" },
            controller_text.join(", then ")
        ));
    }

    // Military history
    if stats.battles_fought > 0 {
        paragraphs.push(format!(
            "{} battle{} {} been fought on its soil, claiming {} lives in total. \
             The earth itself seems stained with the memory of violence.",
            stats.battles_fought,
            if stats.battles_fought == 1 { "" } else { "s" },
            if stats.battles_fought == 1 { "has" } else { "have" },
            stats.total_casualties
        ));
    }

    if stats.sieges > 0 {
        paragraphs.push(format!(
            "Its walls have endured {} siege{}.",
            stats.sieges,
            if stats.sieges == 1 { "" } else { "s" }
        ));
    }

    // Hardship
    let mut hardships = Vec::new();
    if stats.plagues > 0 {
        hardships.push(format!("{} plague{}", stats.plagues, if stats.plagues == 1 { "" } else { "s" }));
    }
    if stats.famines > 0 {
        hardships.push(format!("{} famine{}", stats.famines, if stats.famines == 1 { "" } else { "s" }));
    }
    if stats.rebellions > 0 {
        hardships.push(format!("{} rebellion{}", stats.rebellions, if stats.rebellions == 1 { "" } else { "s" }));
    }
    if !hardships.is_empty() {
        paragraphs.push(format!(
            "The people have endured {}.",
            hardships.join(", ")
        ));
    }

    // Prosperity
    if !stats.improvements_built.is_empty() {
        paragraphs.push(format!(
            "Builders have raised {} here: {}.",
            stats.improvements_built.len(),
            stats.improvements_built.join(", ")
        ));
    }

    if stats.trade_routes > 0 {
        paragraphs.push(format!(
            "{} trade route{} pass{} through its markets.",
            stats.trade_routes,
            if stats.trade_routes == 1 { "" } else { "s" },
            if stats.trade_routes == 1 { "es" } else { "" }
        ));
    }

    // Religious
    if stats.miracles > 0 || stats.heresies > 0 {
        let mut religious = Vec::new();
        if stats.miracles > 0 {
            religious.push(format!("{} miracle{}", stats.miracles, if stats.miracles == 1 { "" } else { "s" }));
        }
        if stats.heresies > 0 {
            religious.push(format!("{} heretical outbreak{}", stats.heresies, if stats.heresies == 1 { "" } else { "s" }));
        }
        paragraphs.push(format!(
            "The province has witnessed {}.",
            religious.join(" and ")
        ));
    }

    paragraphs.join("\n\n")
}

// ─── Faction History ────────────────────────────────────────────────────────

/// Aggregated faction history statistics.
#[derive(Debug, Clone, Default)]
struct FactionStats {
    provinces_conquered: u32,
    provinces_lost: u32,
    wars_declared: u32,
    wars_received: u32,
    treaties_signed: u32,
    rulers_died: u32,
    succession_crises: u32,
    realm_splits: u32,
    marriages: u32,
    factions_eliminated: u32,
    was_eliminated: bool,
    eliminated_turn: u32,
}

/// Generate a prose history for a faction from accumulated events.
pub fn faction_history(
    faction_id: u8,
    faction_name: &str,
    events: &[GameEvent],
    ctx: &WorldContext,
) -> String {
    let mut stats = FactionStats::default();

    for event in events {
        match event {
            GameEvent::ProvinceConquered { old_controller, new_controller, .. } => {
                if *new_controller == faction_id { stats.provinces_conquered += 1; }
                if *old_controller == faction_id { stats.provinces_lost += 1; }
            }
            GameEvent::WarDeclared { attacker, defender, .. } => {
                if *attacker == faction_id { stats.wars_declared += 1; }
                if *defender == faction_id { stats.wars_received += 1; }
            }
            GameEvent::TreatySigned { faction_a, faction_b, .. } => {
                if *faction_a == faction_id || *faction_b == faction_id {
                    stats.treaties_signed += 1;
                }
            }
            GameEvent::CharacterDied {  .. } => {
                // We can't easily check faction from CharacterDied, count all ruler deaths
                // This is approximate — in practice filtered by faction before calling
                stats.rulers_died += 1;
            }
            GameEvent::SuccessionCrisis { faction,  .. } if *faction == faction_id => {
                stats.succession_crises += 1;
            }
            GameEvent::RealmSplit { original_faction, .. } if *original_faction == faction_id => {
                stats.realm_splits += 1;
            }
            GameEvent::FactionEliminated { faction, turn } if *faction == faction_id => {
                stats.was_eliminated = true;
                stats.eliminated_turn = *turn;
            }
            GameEvent::FactionEliminated { .. } => {
                stats.factions_eliminated += 1;
            }
            GameEvent::MarriageAlliance { faction_a, faction_b, .. }
                if *faction_a == faction_id || *faction_b == faction_id =>
            {
                stats.marriages += 1;
            }
            _ => {}
        }
    }

    render_faction_history(faction_name, &stats, ctx)
}

fn render_faction_history(name: &str, stats: &FactionStats, _ctx: &WorldContext) -> String {
    let mut paragraphs: Vec<String> = Vec::new();

    // Opening
    if stats.was_eliminated {
        paragraphs.push(format!(
            "{} is no more. Once a power in the realm, their dynasty ended \
             on turn {}. What follows is their story.",
            name, stats.eliminated_turn
        ));
    } else {
        paragraphs.push(format!(
            "The chronicle of {} — a house that endures.", name
        ));
    }

    // Territorial
    if stats.provinces_conquered > 0 || stats.provinces_lost > 0 {
        let net = stats.provinces_conquered as i32 - stats.provinces_lost as i32;
        let trend = if net > 0 { "expanding" } else if net < 0 { "contracting" } else { "holding steady" };
        paragraphs.push(format!(
            "In matters of territory, {} conquered {} province{} and lost {}. \
             The trend: {}.",
            name,
            stats.provinces_conquered,
            if stats.provinces_conquered == 1 { "" } else { "s" },
            stats.provinces_lost,
            trend
        ));
    }

    // Diplomacy & war
    let total_wars = stats.wars_declared + stats.wars_received;
    if total_wars > 0 || stats.treaties_signed > 0 {
        let mut diplo = Vec::new();
        if stats.wars_declared > 0 {
            diplo.push(format!("declared {} war{}", stats.wars_declared, if stats.wars_declared == 1 { "" } else { "s" }));
        }
        if stats.wars_received > 0 {
            diplo.push(format!("was attacked {} time{}", stats.wars_received, if stats.wars_received == 1 { "" } else { "s" }));
        }
        if stats.treaties_signed > 0 {
            diplo.push(format!("signed {} treat{}", stats.treaties_signed, if stats.treaties_signed == 1 { "y" } else { "ies" }));
        }
        paragraphs.push(format!(
            "{} {}.",
            name, diplo.join(", ")
        ));
    }

    // Internal stability
    if stats.succession_crises > 0 || stats.realm_splits > 0 {
        let mut internal = Vec::new();
        if stats.succession_crises > 0 {
            internal.push(format!("{} succession cris{}", stats.succession_crises, if stats.succession_crises == 1 { "is" } else { "es" }));
        }
        if stats.realm_splits > 0 {
            internal.push(format!("{} realm split{}", stats.realm_splits, if stats.realm_splits == 1 { "" } else { "s" }));
        }
        paragraphs.push(format!(
            "Internal turmoil has plagued the house: {}.",
            internal.join(" and ")
        ));
    }

    // Marriages
    if stats.marriages > 0 {
        paragraphs.push(format!(
            "{} marriage alliance{} {} forged ties with other houses.",
            stats.marriages,
            if stats.marriages == 1 { "" } else { "s" },
            if stats.marriages == 1 { "has" } else { "have" }
        ));
    }

    paragraphs.join("\n\n")
}

// ─── Turn Summary ─────────────────────────────────────────────────────────

/// Aggregated turn statistics for a single turn's events.
#[derive(Debug, Clone, Default)]
struct TurnStats {
    battles: u32,
    total_casualties: u32,
    provinces_conquered: Vec<(u16, u8, u8)>, // (province, old, new)
    wars_declared: Vec<(u8, u8)>,            // (attacker, defender)
    treaties_signed: Vec<(u8, u8)>,
    deaths: Vec<String>,                     // character names
    births: u32,
    plagues: u32,
    famines: u32,
    rebellions: u32,
    succession_crises: u32,
    factions_eliminated: Vec<u8>,
    realm_splits: u32,
    sieges_completed: u32,
    plots_succeeded: u32,
    miracles: u32,
}

/// Generate a prose summary for all events in a single turn.
///
/// Produces a concise 1-3 sentence overview highlighting the most important
/// events. Designed to be displayed as a turn header in the event feed.
pub fn turn_summary(turn: u32, events: &[GameEvent], ctx: &WorldContext) -> String {
    if events.is_empty() {
        return format!("Turn {} — The realm holds its breath. Nothing of note occurred.", turn);
    }

    let mut stats = TurnStats::default();

    for event in events {
        match event {
            GameEvent::Battle(r) => {
                stats.battles += 1;
                stats.total_casualties += r.attacker_casualties + r.defender_casualties;
            }
            GameEvent::ProvinceConquered { province, old_controller, new_controller, .. } => {
                stats.provinces_conquered.push((*province, *old_controller, *new_controller));
            }
            GameEvent::WarDeclared { attacker, defender, .. } => {
                stats.wars_declared.push((*attacker, *defender));
            }
            GameEvent::TreatySigned { faction_a, faction_b, .. } => {
                stats.treaties_signed.push((*faction_a, *faction_b));
            }
            GameEvent::CharacterDied { character_name, .. } => {
                stats.deaths.push(character_name.clone());
            }
            GameEvent::CharacterBorn { .. } => { stats.births += 1; }
            GameEvent::PlagueOutbreak { .. } => { stats.plagues += 1; }
            GameEvent::Famine { .. } => { stats.famines += 1; }
            GameEvent::Rebellion { .. } => { stats.rebellions += 1; }
            GameEvent::SuccessionCrisis { .. } => { stats.succession_crises += 1; }
            GameEvent::FactionEliminated { faction, .. } => {
                stats.factions_eliminated.push(*faction);
            }
            GameEvent::RealmSplit { .. } => { stats.realm_splits += 1; }
            GameEvent::SiegeCompleted { .. } => { stats.sieges_completed += 1; }
            GameEvent::PlotSucceeded { .. } => { stats.plots_succeeded += 1; }
            GameEvent::Miracle { .. } => { stats.miracles += 1; }
            _ => {}
        }
    }

    render_turn_summary(turn, &stats, ctx)
}

fn render_turn_summary(turn: u32, stats: &TurnStats, ctx: &WorldContext) -> String {
    let mut highlights: Vec<String> = Vec::new();

    // Factions eliminated (most dramatic — leads)
    for &fid in &stats.factions_eliminated {
        highlights.push(format!(
            "{} has been destroyed",
            ctx.faction_name(fid)
        ));
    }

    // Realm splits
    if stats.realm_splits > 0 {
        highlights.push(format!(
            "{} realm{} shattered",
            stats.realm_splits,
            if stats.realm_splits == 1 { "" } else { "s" }
        ));
    }

    // Wars declared
    for &(att, def) in &stats.wars_declared {
        highlights.push(format!(
            "{} declared war on {}",
            ctx.faction_name(att),
            ctx.faction_name(def)
        ));
    }

    // Battles and casualties
    if stats.battles > 0 {
        highlights.push(format!(
            "{} battle{}, {} dead",
            stats.battles,
            if stats.battles == 1 { "" } else { "s" },
            stats.total_casualties
        ));
    }

    // Province conquests
    if !stats.provinces_conquered.is_empty() {
        let count = stats.provinces_conquered.len();
        if count == 1 {
            let (pid, _old, new) = stats.provinces_conquered[0];
            highlights.push(format!(
                "{} seized {}",
                ctx.faction_name(new),
                ctx.province_name(pid)
            ));
        } else {
            highlights.push(format!(
                "{} province{} changed hands",
                count,
                if count == 1 { "" } else { "s" }
            ));
        }
    }

    // Sieges completed
    if stats.sieges_completed > 0 {
        highlights.push(format!(
            "{} siege{} concluded",
            stats.sieges_completed,
            if stats.sieges_completed == 1 { "" } else { "s" }
        ));
    }

    // Treaties
    for &(a, b) in &stats.treaties_signed {
        highlights.push(format!(
            "{} and {} signed peace",
            ctx.faction_name(a),
            ctx.faction_name(b)
        ));
    }

    // Deaths
    if !stats.deaths.is_empty() {
        if stats.deaths.len() == 1 {
            highlights.push(format!("{} perished", stats.deaths[0]));
        } else {
            highlights.push(format!("{} souls perished", stats.deaths.len()));
        }
    }

    // Succession crises
    if stats.succession_crises > 0 {
        highlights.push(format!(
            "{} succession cris{}",
            stats.succession_crises,
            if stats.succession_crises == 1 { "is" } else { "es" }
        ));
    }

    // Hardship (compact)
    let mut hardship = Vec::new();
    if stats.plagues > 0 { hardship.push(format!("{} plague{}", stats.plagues, if stats.plagues == 1 { "" } else { "s" })); }
    if stats.famines > 0 { hardship.push(format!("{} famine{}", stats.famines, if stats.famines == 1 { "" } else { "s" })); }
    if stats.rebellions > 0 { hardship.push(format!("{} rebellion{}", stats.rebellions, if stats.rebellions == 1 { "" } else { "s" })); }
    if !hardship.is_empty() {
        highlights.push(hardship.join(", "));
    }

    // Plots
    if stats.plots_succeeded > 0 {
        highlights.push(format!(
            "{} plot{} succeeded",
            stats.plots_succeeded,
            if stats.plots_succeeded == 1 { "" } else { "s" }
        ));
    }

    // Miracles
    if stats.miracles > 0 {
        highlights.push("a miracle was witnessed".to_string());
    }

    // Compose final text
    if highlights.is_empty() {
        return format!("Turn {} — A quiet turn. The realm stirs, but no great deeds mark the day.", turn);
    }

    // Take top 4 highlights to keep it concise
    let top: Vec<&str> = highlights.iter().map(|s| s.as_str()).take(4).collect();
    format!("Turn {} — {}", turn, capitalize_first(&top.join(". ")))
}

/// Capitalize the first character of a string.
fn capitalize_first(s: &str) -> String {
    let mut chars = s.chars();
    match chars.next() {
        None => String::new(),
        Some(c) => c.to_uppercase().to_string() + chars.as_str(),
    }
}

// ─── War Summary ──────────────────────────────────────────────────────────

/// Statistics about a specific war between two factions.
#[derive(Debug, Clone, Default)]
struct WarStats {
    started_turn: Option<u32>,
    casus_belli: Option<String>,
    battles: u32,
    total_casualties: u32,
    provinces_taken_by_a: u32,
    provinces_taken_by_b: u32,
    sieges: u32,
    peace_treaty: Option<(String, u32)>, // (type, turn)
}

/// Generate a prose summary of a war between two factions.
///
/// Scans events for all war-related activity between `faction_a` and `faction_b`:
/// war declarations, battles, province conquests, sieges, and treaties.
pub fn war_summary(
    faction_a: u8,
    faction_b: u8,
    events: &[GameEvent],
    ctx: &WorldContext,
) -> String {
    let mut stats = WarStats::default();

    for event in events {
        match event {
            GameEvent::WarDeclared { attacker, defender, casus_belli, turn } => {
                if (*attacker == faction_a && *defender == faction_b)
                    || (*attacker == faction_b && *defender == faction_a)
                {
                    stats.started_turn = Some(*turn);
                    stats.casus_belli = Some(casus_belli.clone());
                }
            }
            GameEvent::Battle(r) => {
                // Check if this battle involves both factions (via army owners).
                let att_faction = ctx.army_faction(r.attacker_army);
                let def_faction = r.defender_army.and_then(|d| ctx.army_faction(d));
                let involves_both = match (att_faction, def_faction) {
                    (Some(af), Some(df)) => {
                        (af == faction_a && df == faction_b)
                            || (af == faction_b && df == faction_a)
                    }
                    _ => false,
                };
                if involves_both {
                    stats.battles += 1;
                    stats.total_casualties += r.attacker_casualties + r.defender_casualties;
                }
            }
            GameEvent::ProvinceConquered { old_controller, new_controller, .. } => {
                if *old_controller == faction_b && *new_controller == faction_a {
                    stats.provinces_taken_by_a += 1;
                } else if *old_controller == faction_a && *new_controller == faction_b {
                    stats.provinces_taken_by_b += 1;
                }
            }
            GameEvent::SiegeCompleted { old_controller, new_controller, .. } => {
                if (*old_controller == faction_a && *new_controller == faction_b)
                    || (*old_controller == faction_b && *new_controller == faction_a)
                {
                    stats.sieges += 1;
                }
            }
            GameEvent::TreatySigned { faction_a: fa, faction_b: fb, treaty_type, turn } => {
                if (*fa == faction_a && *fb == faction_b)
                    || (*fa == faction_b && *fb == faction_a)
                {
                    stats.peace_treaty = Some((treaty_type.clone(), *turn));
                }
            }
            _ => {}
        }
    }

    render_war_summary(faction_a, faction_b, &stats, ctx)
}

fn render_war_summary(
    faction_a: u8,
    faction_b: u8,
    stats: &WarStats,
    ctx: &WorldContext,
) -> String {
    let name_a = ctx.faction_name(faction_a);
    let name_b = ctx.faction_name(faction_b);
    let mut paragraphs: Vec<String> = Vec::new();

    // Opening — when and why
    match (&stats.started_turn, &stats.casus_belli) {
        (Some(turn), Some(cb)) => {
            paragraphs.push(format!(
                "The war between {} and {} began on turn {}, \
                 sparked by a claim of \"{}\".",
                name_a, name_b, turn, cb
            ));
        }
        (Some(turn), None) => {
            paragraphs.push(format!(
                "War erupted between {} and {} on turn {}.",
                name_a, name_b, turn
            ));
        }
        _ => {
            paragraphs.push(format!(
                "{} and {} are locked in conflict.", name_a, name_b
            ));
        }
    }

    // Battles
    if stats.battles > 0 {
        paragraphs.push(format!(
            "{} battle{} {} been fought, claiming {} lives.",
            stats.battles,
            if stats.battles == 1 { "" } else { "s" },
            if stats.battles == 1 { "has" } else { "have" },
            stats.total_casualties
        ));
    }

    // Territory changes
    if stats.provinces_taken_by_a > 0 || stats.provinces_taken_by_b > 0 {
        let mut territory = Vec::new();
        if stats.provinces_taken_by_a > 0 {
            territory.push(format!(
                "{} seized {} province{}", name_a,
                stats.provinces_taken_by_a,
                if stats.provinces_taken_by_a == 1 { "" } else { "s" }
            ));
        }
        if stats.provinces_taken_by_b > 0 {
            territory.push(format!(
                "{} took {} province{}", name_b,
                stats.provinces_taken_by_b,
                if stats.provinces_taken_by_b == 1 { "" } else { "s" }
            ));
        }
        paragraphs.push(format!("{}.", territory.join(", while ")));
    }

    // Sieges
    if stats.sieges > 0 {
        paragraphs.push(format!(
            "{} siege{} {} concluded.",
            stats.sieges,
            if stats.sieges == 1 { "" } else { "s" },
            if stats.sieges == 1 { "has" } else { "have" }
        ));
    }

    // Peace treaty (if any)
    if let Some((treaty_type, turn)) = &stats.peace_treaty {
        paragraphs.push(format!(
            "A {} was signed on turn {}, ending the bloodshed.",
            treaty_type, turn
        ));
    } else if stats.battles > 0 {
        // Ongoing war assessment
        if stats.provinces_taken_by_a > stats.provinces_taken_by_b {
            paragraphs.push(format!(
                "The tide favors {} — but the war rages on.", name_a
            ));
        } else if stats.provinces_taken_by_b > stats.provinces_taken_by_a {
            paragraphs.push(format!(
                "{} holds the advantage, yet no peace is in sight.", name_b
            ));
        } else {
            paragraphs.push("Neither side has gained a clear advantage. The war grinds on.".to_string());
        }
    }

    paragraphs.join(" ")
}

// ─── Era Summary / State of the World ────────────────────────────────────────

/// World-level statistics for era summary generation.
#[derive(Default)]
struct EraStats {
    factions_alive: u32,
    factions_eliminated: u32,
    total_wars: u32,
    active_wars: u32,
    total_battles: u32,
    total_casualties: u32,
    provinces_conquered: u32,
    plagues: u32,
    rebellions: u32,
    treaties: u32,
    trade_routes: u32,
    characters_born: u32,
    characters_died: u32,
    realm_splits: u32,
    dominant_faction: Option<(u8, u32)>, // (faction_id, province_count)
}

/// Generate a "State of the World" era summary narrative.
///
/// Displayed in the detail panel when nothing is selected. Provides a
/// high-level overview of the game world: dominant power, ongoing conflicts,
/// population trends, and recent major events.
pub fn era_summary(
    current_turn: u32,
    factions_alive: u32,
    active_wars: &[(u8, u8)],     // current war pairs
    faction_provinces: &[(u8, u32)], // (faction_id, province_count)
    events: &[GameEvent],
    ctx: &WorldContext,
) -> String {
    let mut stats = EraStats::default();
    stats.factions_alive = factions_alive;
    stats.active_wars = active_wars.len() as u32;

    // Find dominant faction
    stats.dominant_faction = faction_provinces.iter()
        .max_by_key(|(_, count)| *count)
        .copied();

    for event in events {
        match event {
            GameEvent::Battle(r) => {
                stats.total_battles += 1;
                stats.total_casualties += r.attacker_casualties + r.defender_casualties;
            }
            GameEvent::WarDeclared { .. } => stats.total_wars += 1,
            GameEvent::TreatySigned { .. } => stats.treaties += 1,
            GameEvent::ProvinceConquered { .. } => stats.provinces_conquered += 1,
            GameEvent::FactionEliminated { .. } => stats.factions_eliminated += 1,
            GameEvent::PlagueOutbreak { .. } => stats.plagues += 1,
            GameEvent::Rebellion { .. } => stats.rebellions += 1,
            GameEvent::CharacterBorn { .. } => stats.characters_born += 1,
            GameEvent::CharacterDied { .. } => stats.characters_died += 1,
            GameEvent::RealmSplit { .. } => stats.realm_splits += 1,
            GameEvent::TradeRouteEstablished { .. } => stats.trade_routes += 1,
            _ => {}
        }
    }

    render_era_summary(current_turn, &stats, active_wars, ctx)
}

fn render_era_summary(
    current_turn: u32,
    stats: &EraStats,
    active_wars: &[(u8, u8)],
    ctx: &WorldContext,
) -> String {
    let mut parts: Vec<String> = Vec::new();

    // Opening — era and power balance
    if let Some((dom_id, dom_count)) = stats.dominant_faction {
        let dom_name = ctx.faction_name(dom_id);
        if dom_count > 8 {
            parts.push(format!(
                "Turn {} of the chronicle. The {} dominates the realm, \
                 holding {} provinces — a formidable empire among {} surviving factions.",
                current_turn, dom_name, dom_count, stats.factions_alive
            ));
        } else {
            parts.push(format!(
                "Turn {} of the chronicle. {} factions vie for supremacy \
                 across the fractured realm. {} holds the most territory with {} provinces.",
                current_turn, stats.factions_alive, dom_name, dom_count
            ));
        }
    } else {
        parts.push(format!(
            "Turn {} of the chronicle. {} factions contend for mastery of the realm.",
            current_turn, stats.factions_alive
        ));
    }

    // Active wars
    if stats.active_wars > 0 {
        if stats.active_wars == 1 {
            if let Some((a, b)) = active_wars.first() {
                parts.push(format!(
                    "War rages between {} and {}.",
                    ctx.faction_name(*a), ctx.faction_name(*b)
                ));
            }
        } else {
            parts.push(format!(
                "{} wars burn across the land, consuming lives and treasure.",
                stats.active_wars
            ));
        }
    } else if stats.total_wars > 0 {
        parts.push("An uneasy peace holds — for now.".to_string());
    } else {
        parts.push("The realm knows peace, though ambitions stir beneath the surface.".to_string());
    }

    // Bloodshed statistics
    if stats.total_battles > 0 {
        parts.push(format!(
            "{} battle{} {} been fought, claiming {} lives.",
            stats.total_battles,
            if stats.total_battles == 1 { "" } else { "s" },
            if stats.total_battles == 1 { "has" } else { "have" },
            stats.total_casualties
        ));
    }

    // Political upheaval
    if stats.factions_eliminated > 0 || stats.realm_splits > 0 {
        let mut upheaval = Vec::new();
        if stats.factions_eliminated > 0 {
            upheaval.push(format!(
                "{} faction{} erased from the map",
                stats.factions_eliminated,
                if stats.factions_eliminated == 1 { "" } else { "s" }
            ));
        }
        if stats.realm_splits > 0 {
            upheaval.push(format!(
                "{} realm{} shattered by succession",
                stats.realm_splits,
                if stats.realm_splits == 1 { "" } else { "s" }
            ));
        }
        parts.push(format!("{}.", upheaval.join(", and ")));
    }

    // Hardship
    if stats.plagues > 0 || stats.rebellions > 0 {
        let mut hardship = Vec::new();
        if stats.plagues > 0 {
            hardship.push(format!("{} plague{}", stats.plagues, if stats.plagues == 1 { "" } else { "s" }));
        }
        if stats.rebellions > 0 {
            hardship.push(format!("{} rebellion{}", stats.rebellions, if stats.rebellions == 1 { "" } else { "s" }));
        }
        parts.push(format!(
            "{} {} tested the realm's endurance.",
            capitalize_first(&hardship.join(" and ")),
            if stats.plagues + stats.rebellions == 1 { "has" } else { "have" }
        ));
    }

    parts.join(" ")
}

// ─── Character Relationship Narrative ────────────────────────────────────────

/// Generate prose describing a character's personal relationships.
///
/// Takes the character's relation list and scans events for context (when
/// the relationship formed, any shared battles, plots, or marriages).
pub fn relationship_narrative(
    character_id: u32,
    relations: &[(u32, Option<crown_ash_types::character::RelationType>, i64)], // (target_id, type, opinion)
    events: &[GameEvent],
    ctx: &WorldContext,
) -> String {
    if relations.is_empty() {
        return String::new();
    }

    let self_name = ctx.character_name(character_id);
    let mut lines: Vec<String> = Vec::new();

    for &(target_id, ref rel_type, opinion) in relations {
        let target_name = ctx.character_name(target_id);
        let context = find_relationship_context(character_id, target_id, events, ctx);

        let line = match rel_type {
            Some(crown_ash_types::character::RelationType::Friend) => {
                if let Some(ctx_text) = &context {
                    format!("{} and {} are close allies, bonded {}.", self_name, target_name, ctx_text)
                } else if opinion > 700 {
                    format!("{} and {} share a deep and abiding friendship.", self_name, target_name)
                } else {
                    format!("{} counts {} among their trusted companions.", self_name, target_name)
                }
            }
            Some(crown_ash_types::character::RelationType::Rival) => {
                if let Some(ctx_text) = &context {
                    format!("{} and {} are bitter rivals, their enmity born {}.", self_name, target_name, ctx_text)
                } else if opinion < -700 {
                    format!("A deep hatred festers between {} and {}.", self_name, target_name)
                } else {
                    format!("{} regards {} with open contempt.", self_name, target_name)
                }
            }
            Some(crown_ash_types::character::RelationType::Mentor) => {
                format!("{} serves as mentor to {}, guiding their education.", self_name, target_name)
            }
            Some(crown_ash_types::character::RelationType::MarriageAlliance) => {
                let faction_text = find_marriage_factions(character_id, target_id, events, ctx);
                if let Some(ft) = faction_text {
                    format!("{} wed {} in a diplomatic union binding {}.", self_name, target_name, ft)
                } else {
                    format!("{} and {} are joined in marriage.", self_name, target_name)
                }
            }
            None => {
                if opinion > 300 {
                    format!("{} holds {} in fair regard.", self_name, target_name)
                } else if opinion < -300 {
                    format!("{} harbors ill will toward {}.", self_name, target_name)
                } else {
                    continue; // Skip untyped neutral relationships
                }
            }
        };

        lines.push(line);
    }

    lines.join(" ")
}

/// Search events for context about when/how a relationship formed.
fn find_relationship_context(
    char_a: u32,
    char_b: u32,
    events: &[GameEvent],
    _ctx: &WorldContext,
) -> Option<String> {
    for event in events.iter().rev() {
        match event {
            GameEvent::Friendship { character_a, character_b, turn }
                if (*character_a == char_a && *character_b == char_b)
                    || (*character_a == char_b && *character_b == char_a) =>
            {
                return Some(format!("since turn {}", turn));
            }
            GameEvent::Rivalry { character_a, character_b, turn }
                if (*character_a == char_a && *character_b == char_b)
                    || (*character_a == char_b && *character_b == char_a) =>
            {
                return Some(format!("since turn {}", turn));
            }
            _ => {}
        }
    }
    None
}

/// Find faction names involved in a marriage alliance.
fn find_marriage_factions(
    char_a: u32,
    char_b: u32,
    events: &[GameEvent],
    ctx: &WorldContext,
) -> Option<String> {
    for event in events.iter().rev() {
        if let GameEvent::MarriageAlliance { character_a, character_b, faction_a, faction_b, .. } = event {
            if (*character_a == char_a && *character_b == char_b)
                || (*character_a == char_b && *character_b == char_a)
            {
                let fa = ctx.faction_name(*faction_a);
                let fb = ctx.faction_name(*faction_b);
                return Some(format!("{} and {}", fa, fb));
            }
        }
    }
    None
}

// ─── Dynasty Lineage Narrative ──────────────────────────────────────────────

/// Generate a lineage narrative for a character based on birth/death events
/// of their dynasty members.
pub fn dynasty_lineage(
    character_id: u32,
    dynasty_id: u16,
    events: &[GameEvent],
    ctx: &WorldContext,
) -> String {
    // Collect all characters born into this dynasty
    let mut dynasty_members: Vec<(u32, String, u32, u32)> = Vec::new(); // (id, name, parent, turn)
    for event in events {
        if let GameEvent::CharacterBorn { character_id: cid, character_name, parent, dynasty, turn } = event {
            if *dynasty == dynasty_id {
                dynasty_members.push((*cid, character_name.clone(), *parent, *turn));
            }
        }
    }

    if dynasty_members.is_empty() {
        return String::new();
    }

    let self_name = ctx.character_name(character_id);
    let mut lines: Vec<String> = Vec::new();

    // Count generation position
    let generation = dynasty_members.iter()
        .position(|(id, _, _, _)| *id == character_id)
        .map(|p| p + 1)
        .unwrap_or(1);

    let ordinal = match generation {
        1 => "Founder".to_string(),
        2 => "Second".to_string(),
        3 => "Third".to_string(),
        4 => "Fourth".to_string(),
        5 => "Fifth".to_string(),
        n => format!("{}th", n),
    };

    lines.push(format!("{} of their dynasty line.", ordinal));

    // Find parent and their fate
    if let Some((_, _, parent_id, _)) = dynasty_members.iter().find(|(id, _, _, _)| *id == character_id) {
        if *parent_id > 0 {
            let parent_name = ctx.character_name(*parent_id);
            let mut found_death = false;
            for event in events {
                if let GameEvent::CharacterDied { character_id: cid, cause, .. } = event {
                    if *cid == *parent_id {
                        let cause_text = match cause {
                            crown_ash_types::DeathCause::Battle => "who fell in battle",
                            crown_ash_types::DeathCause::OldAge => "who passed of old age",
                            crown_ash_types::DeathCause::Disease => "who succumbed to disease",
                            crown_ash_types::DeathCause::Assassination => "who was assassinated",
                            crown_ash_types::DeathCause::Execution => "who was executed",
                            crown_ash_types::DeathCause::Accident => "who perished by accident",
                        };
                        lines.push(format!("Child of {} {}.", parent_name, cause_text));
                        found_death = true;
                        break;
                    }
                }
            }
            if !found_death {
                lines.push(format!("Child of {}.", parent_name));
            }
        }
    }

    // Count dynasty deaths
    let dynasty_deaths = events.iter().filter(|e| {
        if let GameEvent::CharacterDied { character_id: cid, .. } = e {
            dynasty_members.iter().any(|(did, _, _, _)| did == cid)
        } else {
            false
        }
    }).count();

    if dynasty_deaths > 0 {
        lines.push(format!(
            "{} member{} of their bloodline {} perished.",
            dynasty_deaths,
            if dynasty_deaths == 1 { "" } else { "s" },
            if dynasty_deaths == 1 { "has" } else { "have" }
        ));
    }

    format!("{} — {}", self_name, lines.join(" "))
}

// ─── Realm Prosperity Narrative ──────────────────────────────────────────────

/// Aggregated prosperity statistics for a faction's realm.
#[derive(Default)]
struct ProsperityStats {
    harvests: u32,
    famines: u32,
    plagues: u32,
    trade_routes_established: u32,
    trade_routes_disrupted: u32,
    improvements_built: u32,
    rebellions: u32,
    provinces_gained: u32,
    provinces_lost: u32,
}

/// Generate a "State of the Realm" prose summary for a faction.
///
/// Describes economic trajectory: harvests, famines, trade, construction,
/// and overall prosperity trend.
pub fn realm_prosperity(
    faction_id: u8,
    controlled_provinces: &[u16],
    events: &[GameEvent],
    ctx: &WorldContext,
) -> String {
    let mut stats = ProsperityStats::default();

    for event in events {
        match event {
            GameEvent::Harvest { province, .. } => {
                if controlled_provinces.contains(province) {
                    stats.harvests += 1;
                }
            }
            GameEvent::Famine { province, .. } => {
                if controlled_provinces.contains(province) {
                    stats.famines += 1;
                }
            }
            GameEvent::PlagueOutbreak { province, .. } => {
                if controlled_provinces.contains(province) {
                    stats.plagues += 1;
                }
            }
            GameEvent::TradeRouteEstablished { from, to, .. } => {
                if controlled_provinces.contains(from) || controlled_provinces.contains(to) {
                    stats.trade_routes_established += 1;
                }
            }
            GameEvent::TradeRouteDisrupted { from, to, .. } => {
                if controlled_provinces.contains(from) || controlled_provinces.contains(to) {
                    stats.trade_routes_disrupted += 1;
                }
            }
            GameEvent::ConstructionComplete { province, .. } => {
                if controlled_provinces.contains(province) {
                    stats.improvements_built += 1;
                }
            }
            GameEvent::Rebellion { province, .. } => {
                if controlled_provinces.contains(province) {
                    stats.rebellions += 1;
                }
            }
            GameEvent::ProvinceConquered { new_controller, old_controller, .. } => {
                if *new_controller == faction_id {
                    stats.provinces_gained += 1;
                } else if *old_controller == faction_id {
                    stats.provinces_lost += 1;
                }
            }
            _ => {}
        }
    }

    render_prosperity(faction_id, &stats, ctx)
}

fn render_prosperity(
    faction_id: u8,
    stats: &ProsperityStats,
    ctx: &WorldContext,
) -> String {
    let name = ctx.faction_name(faction_id);
    let mut parts: Vec<String> = Vec::new();

    // Overall economic mood
    let good = stats.harvests + stats.trade_routes_established + stats.improvements_built;
    let bad = stats.famines + stats.plagues + stats.rebellions + stats.trade_routes_disrupted;

    if good == 0 && bad == 0 {
        return format!("The realm of {} endures in quiet stability.", name);
    }

    if good > bad * 2 {
        parts.push(format!("The realm of {} prospers.", name));
    } else if bad > good * 2 {
        parts.push(format!("The realm of {} suffers under hardship.", name));
    } else if good > bad {
        parts.push(format!("The realm of {} fares well, though not without troubles.", name));
    } else {
        parts.push(format!("The realm of {} weathers difficult times.", name));
    }

    // Harvests and famines
    if stats.harvests > 0 && stats.famines > 0 {
        parts.push(format!(
            "{} bountiful harvest{} and {} famine{} have marked its lands.",
            stats.harvests,
            if stats.harvests == 1 { "" } else { "s" },
            stats.famines,
            if stats.famines == 1 { "" } else { "s" }
        ));
    } else if stats.harvests > 0 {
        parts.push(format!(
            "{} bountiful harvest{} {} blessed its fields.",
            stats.harvests,
            if stats.harvests == 1 { "" } else { "s" },
            if stats.harvests == 1 { "has" } else { "have" }
        ));
    } else if stats.famines > 0 {
        parts.push(format!(
            "{} famine{} {} ravaged the countryside.",
            stats.famines,
            if stats.famines == 1 { "" } else { "s" },
            if stats.famines == 1 { "has" } else { "have" }
        ));
    }

    // Plague
    if stats.plagues > 0 {
        parts.push(format!(
            "Plague has struck {} time{}, thinning the populace.",
            stats.plagues,
            if stats.plagues == 1 { "" } else { "s" }
        ));
    }

    // Trade
    let active_routes = stats.trade_routes_established.saturating_sub(stats.trade_routes_disrupted);
    if active_routes > 0 {
        parts.push(format!(
            "{} trade route{} fuel{} the economy.",
            active_routes,
            if active_routes == 1 { "" } else { "s" },
            if active_routes == 1 { "s" } else { "" }
        ));
    }

    // Construction
    if stats.improvements_built > 0 {
        parts.push(format!(
            "{} improvement{} {} been constructed.",
            stats.improvements_built,
            if stats.improvements_built == 1 { "" } else { "s" },
            if stats.improvements_built == 1 { "has" } else { "have" }
        ));
    }

    // Unrest
    if stats.rebellions > 0 {
        parts.push(format!(
            "{} rebellion{} {} tested the crown's authority.",
            stats.rebellions,
            if stats.rebellions == 1 { "" } else { "s" },
            if stats.rebellions == 1 { "has" } else { "have" }
        ));
    }

    // Territory changes
    if stats.provinces_gained > 0 || stats.provinces_lost > 0 {
        if stats.provinces_gained > stats.provinces_lost {
            parts.push(format!("Its borders have expanded, claiming {} new province{}.",
                stats.provinces_gained,
                if stats.provinces_gained == 1 { "" } else { "s" }
            ));
        } else if stats.provinces_lost > stats.provinces_gained {
            parts.push(format!("{} province{} {} been lost to rival claimants.",
                stats.provinces_lost,
                if stats.provinces_lost == 1 { "" } else { "s" },
                if stats.provinces_lost == 1 { "has" } else { "have" }
            ));
        }
    }

    parts.join(" ")
}

// ─── Battle Report Narrative ────────────────────────────────────────────────

/// Generate a detailed multi-paragraph battle report from a BattleResult.
pub fn battle_report(
    result: &crown_ash_types::army::BattleResult,
    ctx: &WorldContext,
) -> String {
    let province_name = ctx.province_name(result.province);
    let att_faction = ctx.army_faction(result.attacker_army)
        .map(|f| ctx.faction_name(f).to_string())
        .unwrap_or_else(|| "an unknown force".to_string());

    let def_faction = result.defender_army
        .and_then(|d| ctx.army_faction(d))
        .map(|f| ctx.faction_name(f).to_string())
        .unwrap_or_else(|| "the garrison".to_string());

    let total_dead = result.attacker_casualties + result.defender_casualties;

    let mut paragraphs: Vec<String> = Vec::new();

    // Opening — location and combatants
    paragraphs.push(format!(
        "The Battle of {} saw the forces of {} clash with {} on turn {}.",
        province_name, att_faction, def_faction, result.turn
    ));

    // Casualties breakdown
    if total_dead > 0 {
        let severity = if total_dead > 500 {
            "devastating"
        } else if total_dead > 200 {
            "fierce"
        } else if total_dead > 50 {
            "bloody"
        } else {
            "brief"
        };

        paragraphs.push(format!(
            "The {} engagement claimed {} lives — {} among the attackers and {} among the defenders.",
            severity, total_dead, result.attacker_casualties, result.defender_casualties
        ));
    }

    // Outcome
    if result.attacker_won {
        if result.defender_casualties > result.attacker_casualties * 2 {
            paragraphs.push(format!(
                "A crushing victory for {} — the defenders were routed with terrible losses.",
                att_faction
            ));
        } else {
            paragraphs.push(format!(
                "The attackers carried the field, though not without cost.",
            ));
        }
    } else {
        if result.attacker_casualties > result.defender_casualties * 2 {
            paragraphs.push(format!(
                "{} held firm, inflicting devastating casualties on the invaders.",
                def_faction
            ));
        } else {
            paragraphs.push("The defenders held their ground, repelling the assault.".to_string());
        }
    }

    paragraphs.join(" ")
}

// ─── Intrigue Plot Narrative ────────────────────────────────────────────────

/// Generate narrative prose for intrigue events.
///
/// Covers plot launches, successes, discoveries, and foiled attempts.
pub fn intrigue_narrative(
    events: &[GameEvent],
    _ctx: &WorldContext,
) -> Vec<(u32, String)> {
    let mut results: Vec<(u32, String)> = Vec::new();

    for event in events {
        match event {
            GameEvent::PlotSucceeded { instigator_name, target_name, plot_type, turn } => {
                let prose = match plot_type.as_str() {
                    "Assassination" => format!(
                        "In the shadows of turn {}, {} orchestrated the assassination of {}. The deed was done swiftly, leaving no trace.",
                        turn, instigator_name, target_name
                    ),
                    "Fabricate Claim" => format!(
                        "Through forged documents and bribed scribes, {} successfully fabricated a claim against {} on turn {}.",
                        instigator_name, target_name, turn
                    ),
                    "Seduce" => format!(
                        "{} employed their charms to seduce {} on turn {}, creating a web of scandal and leverage.",
                        instigator_name, target_name, turn
                    ),
                    "Sabotage" => format!(
                        "On turn {}, {}'s agents sabotaged {}'s holdings, causing disruption and economic damage.",
                        turn, instigator_name, target_name
                    ),
                    _ => format!(
                        "On turn {}, {}'s {} plot against {} succeeded.",
                        turn, instigator_name, plot_type.to_lowercase(), target_name
                    ),
                };
                results.push((*turn, prose));
            }
            GameEvent::PlotDiscovered { instigator_name, target_name, discovered_by, turn } => {
                results.push((*turn, format!(
                    "Turn {}: {} uncovered a sinister plot by {} against {}. The schemer's reputation is tarnished.",
                    turn, discovered_by, instigator_name, target_name
                )));
            }
            GameEvent::PlotFoiled { instigator_name, target_name, turn } => {
                results.push((*turn, format!(
                    "Turn {}: {}'s plot against {} was foiled at the last moment. The would-be victim lives another day.",
                    turn, instigator_name, target_name
                )));
            }
            _ => {}
        }
    }

    results
}

// ─── Religion Narrative ─────────────────────────────────────────────────────

/// Generate prose about a province's religious history.
///
/// Covers conversions, heresy events, and miracles.
pub fn religion_narrative(
    province_id: u16,
    province_name: &str,
    current_religion: &str,
    events: &[GameEvent],
    _ctx: &WorldContext,
) -> String {
    let mut conversions: Vec<(String, String, u32)> = Vec::new(); // (old, new, turn)
    let mut heresies: u32 = 0;
    let mut miracles: u32 = 0;

    for event in events {
        match event {
            GameEvent::ReligiousConversion { province, old_religion, new_religion, turn }
                if *province == province_id =>
            {
                conversions.push((old_religion.clone(), new_religion.clone(), *turn));
            }
            GameEvent::Heresy { province, .. } if *province == province_id => {
                heresies += 1;
            }
            GameEvent::Miracle { province, .. } if *province == province_id => {
                miracles += 1;
            }
            _ => {}
        }
    }

    if conversions.is_empty() && heresies == 0 && miracles == 0 {
        return format!(
            "The people of {} hold steadfast to the {} faith.",
            province_name, current_religion
        );
    }

    let mut parts: Vec<String> = Vec::new();

    if let Some(last) = conversions.last() {
        if conversions.len() == 1 {
            parts.push(format!(
                "{} converted from {} to {} on turn {}.",
                province_name, last.0, last.1, last.2
            ));
        } else {
            parts.push(format!(
                "{} has changed faith {} times. Most recently, the province embraced {} on turn {}.",
                province_name, conversions.len(), last.1, last.2
            ));
        }
    }

    if heresies > 0 {
        parts.push(format!(
            "Heretical movements have plagued the province {} time{}, shaking religious authority.",
            heresies, if heresies == 1 { "" } else { "s" }
        ));
    }

    if miracles > 0 {
        parts.push(format!(
            "{} miracle{} {} been witnessed, strengthening the faith of the devout.",
            miracles,
            if miracles == 1 { "" } else { "s" },
            if miracles == 1 { "has" } else { "have" }
        ));
    }

    parts.join(" ")
}

// ─── Diplomacy Narrative ────────────────────────────────────────────────────

/// Generate narrative prose for the diplomatic relationship between two factions.
pub fn diplomacy_narrative(
    faction_a: u8,
    faction_b: u8,
    at_war: bool,
    events: &[GameEvent],
    ctx: &WorldContext,
) -> String {
    let name_a = ctx.faction_name(faction_a);
    let name_b = ctx.faction_name(faction_b);

    let mut wars_declared: u32 = 0;
    let mut treaties_signed: Vec<(String, u32)> = Vec::new();
    let mut marriages: u32 = 0;
    let mut provinces_taken_ab: u32 = 0;
    let mut provinces_taken_ba: u32 = 0;

    for event in events {
        match event {
            GameEvent::WarDeclared { attacker, defender, .. }
                if (*attacker == faction_a && *defender == faction_b)
                    || (*attacker == faction_b && *defender == faction_a) =>
            {
                wars_declared += 1;
            }
            GameEvent::TreatySigned { faction_a: fa, faction_b: fb, treaty_type, turn }
                if (*fa == faction_a && *fb == faction_b)
                    || (*fa == faction_b && *fb == faction_a) =>
            {
                treaties_signed.push((treaty_type.clone(), *turn));
            }
            GameEvent::MarriageAlliance { faction_a: fa, faction_b: fb, .. }
                if (*fa == faction_a && *fb == faction_b)
                    || (*fa == faction_b && *fb == faction_a) =>
            {
                marriages += 1;
            }
            GameEvent::ProvinceConquered { old_controller, new_controller, .. } => {
                if *old_controller == faction_b && *new_controller == faction_a {
                    provinces_taken_ab += 1;
                } else if *old_controller == faction_a && *new_controller == faction_b {
                    provinces_taken_ba += 1;
                }
            }
            _ => {}
        }
    }

    let mut parts: Vec<String> = Vec::new();

    // Current state
    if at_war {
        if wars_declared > 1 {
            let ordinal = match wars_declared { 2 => "2nd".to_string(), 3 => "3rd".to_string(), n => format!("{}th", n) };
            parts.push(format!(
                "{} and {} are locked in their {} conflict.",
                name_a, name_b, ordinal
            ));
        } else {
            parts.push(format!("{} and {} are at war.", name_a, name_b));
        }
    } else if marriages > 0 && wars_declared == 0 {
        parts.push(format!(
            "{} and {} are bound by marriage alliance — a bond of mutual interest, if not affection.",
            name_a, name_b
        ));
    } else if !treaties_signed.is_empty() {
        if let Some(last) = treaties_signed.last() {
            parts.push(format!(
                "A {} holds between {} and {} since turn {}.",
                last.0, name_a, name_b, last.1
            ));
        }
    } else if wars_declared > 0 {
        parts.push(format!(
            "An uneasy peace exists between {} and {}, scarred by {} previous war{}.",
            name_a, name_b, wars_declared,
            if wars_declared == 1 { "" } else { "s" }
        ));
    } else {
        parts.push(format!(
            "{} and {} regard each other with cautious neutrality.",
            name_a, name_b
        ));
    }

    // Territory context
    if provinces_taken_ab > 0 || provinces_taken_ba > 0 {
        if provinces_taken_ab > provinces_taken_ba {
            parts.push(format!(
                "{} has seized {} province{} from {}, fueling resentment.",
                name_a, provinces_taken_ab,
                if provinces_taken_ab == 1 { "" } else { "s" },
                name_b
            ));
        } else if provinces_taken_ba > provinces_taken_ab {
            parts.push(format!(
                "{} has lost {} province{} to {}'s ambitions.",
                name_a, provinces_taken_ba,
                if provinces_taken_ba == 1 { "" } else { "s" },
                name_b
            ));
        }
    }

    parts.join(" ")
}

// ─── Siege Narrative ────────────────────────────────────────────────────────

/// Generate prose about a province's siege history.
///
/// Scans for SiegeStarted and SiegeCompleted events targeting this province.
pub fn siege_narrative(
    province_id: u16,
    province_name: &str,
    events: &[GameEvent],
    ctx: &WorldContext,
) -> String {
    let mut sieges_started: Vec<(u8, u8, u32, u32)> = Vec::new(); // (attacker, defender, turns_req, turn)
    let mut sieges_completed: Vec<(u8, u8, u32, u32, u32)> = Vec::new(); // (old, new, lasted, casualties, turn)

    for event in events {
        match event {
            GameEvent::SiegeStarted { province, attacker_faction, defender_faction, turns_required, turn }
                if *province == province_id =>
            {
                sieges_started.push((*attacker_faction, *defender_faction, *turns_required, *turn));
            }
            GameEvent::SiegeCompleted { province, old_controller, new_controller, turns_lasted, attacker_casualties, turn }
                if *province == province_id =>
            {
                sieges_completed.push((*old_controller, *new_controller, *turns_lasted, *attacker_casualties, *turn));
            }
            _ => {}
        }
    }

    if sieges_started.is_empty() && sieges_completed.is_empty() {
        return String::new();
    }

    let mut parts: Vec<String> = Vec::new();

    let total = sieges_started.len().max(sieges_completed.len());
    if total == 1 {
        parts.push(format!(
            "The walls of {} have been tested once by siege.",
            province_name
        ));
    } else {
        parts.push(format!(
            "{} has endured {} sieges throughout its history.",
            province_name, total
        ));
    }

    // Detail the most recent completed siege
    if let Some(&(old, new, lasted, casualties, turn)) = sieges_completed.last() {
        let attacker_name = ctx.faction_name(new);
        let defender_name = ctx.faction_name(old);
        parts.push(format!(
            "Most recently, {} besieged the {}-held fortress for {} turn{}, \
             suffering {} casualties before taking the province on turn {}.",
            attacker_name, defender_name,
            lasted, if lasted == 1 { "" } else { "s" },
            casualties, turn
        ));
    } else if let Some(&(attacker, _defender, turns_req, turn)) = sieges_started.last() {
        // Siege started but not yet completed
        let attacker_name = ctx.faction_name(attacker);
        parts.push(format!(
            "{} currently besieges the province (begun turn {}, \
             estimated {} turns to breach the walls).",
            attacker_name, turn, turns_req
        ));
    }

    parts.join(" ")
}

// ─── Trade Narrative ────────────────────────────────────────────────────────

/// Generate prose about trade routes involving a province.
pub fn trade_narrative(
    province_id: u16,
    province_name: &str,
    events: &[GameEvent],
    ctx: &WorldContext,
) -> String {
    let mut established: Vec<(u16, String, u32)> = Vec::new(); // (partner, goods, turn)
    let mut disrupted: Vec<(u16, String, u32)> = Vec::new();   // (partner, reason, turn)

    for event in events {
        match event {
            GameEvent::TradeRouteEstablished { from, to, goods, turn } => {
                if *from == province_id {
                    established.push((*to, goods.clone(), *turn));
                } else if *to == province_id {
                    established.push((*from, goods.clone(), *turn));
                }
            }
            GameEvent::TradeRouteDisrupted { from, to, reason, turn } => {
                if *from == province_id {
                    disrupted.push((*to, reason.clone(), *turn));
                } else if *to == province_id {
                    disrupted.push((*from, reason.clone(), *turn));
                }
            }
            _ => {}
        }
    }

    if established.is_empty() && disrupted.is_empty() {
        return String::new();
    }

    let mut parts: Vec<String> = Vec::new();

    if established.len() > disrupted.len() {
        parts.push(format!(
            "{} is a hub of commerce, with {} trade route{} established.",
            province_name, established.len(),
            if established.len() == 1 { "" } else { "s" }
        ));
    } else if !established.is_empty() {
        parts.push(format!(
            "Trade flows through {}, though not without setbacks.",
            province_name
        ));
    }

    // Show most recent established route
    if let Some((partner, goods, turn)) = established.last() {
        let partner_name = ctx.province_name(*partner);
        parts.push(format!(
            "A route carrying {} was opened with {} on turn {}.",
            goods, partner_name, turn
        ));
    }

    // Show disruptions
    if !disrupted.is_empty() {
        if disrupted.len() == 1 {
            let (partner, reason, turn) = &disrupted[0];
            let partner_name = ctx.province_name(*partner);
            parts.push(format!(
                "Trade with {} was disrupted by {} on turn {}.",
                partner_name, reason, turn
            ));
        } else {
            parts.push(format!(
                "{} trade routes have been disrupted by conflict or misfortune.",
                disrupted.len()
            ));
        }
    }

    parts.join(" ")
}

// ─── Succession Narrative ───────────────────────────────────────────────────

/// Generate prose about succession crises and realm splits for a faction.
pub fn succession_narrative(
    faction_id: u8,
    events: &[GameEvent],
    ctx: &WorldContext,
) -> String {
    let mut crises: Vec<(u32, u32, usize, bool)> = Vec::new(); // (dead_ruler, turn, claimants, split)
    let mut splits: Vec<(u8, u32, u32, u32)> = Vec::new();     // (new_faction, rebel_leader, provinces_lost, turn)

    for event in events {
        match event {
            GameEvent::SuccessionCrisis { faction, dead_ruler, claimants, realm_split, turn }
                if *faction == faction_id =>
            {
                crises.push((*dead_ruler, *turn, claimants.len(), *realm_split));
            }
            GameEvent::RealmSplit { original_faction, new_faction, rebel_leader, provinces_lost, turn }
                if *original_faction == faction_id =>
            {
                splits.push((*new_faction, *rebel_leader, *provinces_lost, *turn));
            }
            _ => {}
        }
    }

    if crises.is_empty() && splits.is_empty() {
        return String::new();
    }

    let mut parts: Vec<String> = Vec::new();

    let faction_name = ctx.faction_name(faction_id);

    if crises.len() == 1 {
        let (ruler, turn, claimants, split) = crises[0];
        let ruler_name = ctx.character_name(ruler);
        if split {
            parts.push(format!(
                "The death of {} on turn {} plunged {} into succession crisis. \
                 {} claimants vied for the throne, and the realm was torn asunder.",
                ruler_name, turn, faction_name, claimants
            ));
        } else {
            parts.push(format!(
                "Upon the death of {} on turn {}, {} faced a succession crisis \
                 with {} claimants, but the realm held together.",
                ruler_name, turn, faction_name, claimants
            ));
        }
    } else if crises.len() > 1 {
        let splits_count = crises.iter().filter(|(_, _, _, s)| *s).count();
        parts.push(format!(
            "{} has weathered {} succession crises, {} of which fractured the realm.",
            faction_name, crises.len(), splits_count
        ));
    }

    // Detail realm splits
    for &(new_fid, rebel, provinces, turn) in &splits {
        let new_name = ctx.faction_name(new_fid);
        let rebel_name = ctx.character_name(rebel);
        parts.push(format!(
            "On turn {}, {} broke away under {}, taking {} province{} — \
             the birth of {}.",
            turn, rebel_name, rebel_name,
            provinces, if provinces == 1 { "" } else { "s" },
            new_name
        ));
    }

    parts.join(" ")
}

// ─── Character Biography ───────────────────────────────────────���────────────

/// Generate a multi-paragraph biography for a character.
///
/// Covers birth, role, age, traits, key life events (battles fought, plots,
/// marriages), and current status. Designed for the character detail panel.
pub fn character_biography(
    character_id: u32,
    character_name: &str,
    age: u8,
    role: &str,
    faction_id: u8,
    alive: bool,
    traits: &[&str],
    events: &[GameEvent],
    ctx: &WorldContext,
) -> String {
    let faction_name = ctx.faction_name(faction_id);
    let mut parts: Vec<String> = Vec::new();

    // Opening — identity and role
    if alive {
        parts.push(format!(
            "{}, aged {}, serves as {} of {}.",
            character_name, age, role, faction_name
        ));
    } else {
        parts.push(format!(
            "{} was a {} of {}, who lived to the age of {}.",
            character_name, role, faction_name, age
        ));
    }

    // Traits
    if !traits.is_empty() {
        let trait_text = if traits.len() == 1 {
            format!("Known as {}.", traits[0])
        } else if traits.len() == 2 {
            format!("Known as {} and {}.", traits[0], traits[1])
        } else {
            let (last, rest) = traits.split_last().unwrap();
            format!("Known as {}, and {}.",
                rest.iter().map(|s| *s).collect::<Vec<_>>().join(", "), last)
        };
        parts.push(trait_text);
    }

    // Count battles fought (approximate: faction's armies participated)
    let battles_fought = events.iter().filter(|e| match e {
        GameEvent::Battle(r) => {
            let att_faction = ctx.army_faction(r.attacker_army);
            let def_faction = r.defender_army.and_then(|d| ctx.army_faction(d));
            att_faction == Some(faction_id) || def_faction == Some(faction_id)
        }
        _ => false,
    }).count();

    if battles_fought > 0 {
        parts.push(format!(
            "Participated in {} battle{} on behalf of {}.",
            battles_fought,
            if battles_fought == 1 { "" } else { "s" },
            faction_name
        ));
    }

    // Marriages
    let marriages: Vec<_> = events.iter().filter_map(|e| match e {
        GameEvent::MarriageAlliance { character_a, character_b, turn, .. }
            if *character_a == character_id || *character_b == character_id =>
        {
            let spouse_id = if *character_a == character_id { *character_b } else { *character_a };
            Some((spouse_id, *turn))
        }
        _ => None,
    }).collect();

    for (spouse_id, turn) in &marriages {
        let spouse_name = ctx.character_name(*spouse_id);
        parts.push(format!("Wed {} on turn {}.", spouse_name, turn));
    }

    // Plots against this character
    let plots_against = events.iter().filter(|e| match e {
        GameEvent::PlotSucceeded { target_name, .. } => target_name == character_name,
        GameEvent::PlotDiscovered { target_name, .. } => target_name == character_name,
        GameEvent::PlotFoiled { target_name, .. } => target_name == character_name,
        _ => false,
    }).count();

    if plots_against > 0 {
        parts.push(format!(
            "Has been the target of {} intrigue plot{}.",
            plots_against,
            if plots_against == 1 { "" } else { "s" }
        ));
    }

    // Death
    for event in events {
        if let GameEvent::CharacterDied { character_id: cid, cause, turn, .. } = event {
            if *cid == character_id {
                let cause_text = match cause {
                    crown_ash_types::DeathCause::OldAge => "of old age",
                    crown_ash_types::DeathCause::Battle => "in battle",
                    crown_ash_types::DeathCause::Disease => "of disease",
                    crown_ash_types::DeathCause::Assassination => "by assassination",
                    crown_ash_types::DeathCause::Execution => "by execution",
                    crown_ash_types::DeathCause::Accident => "in an accident",
                };
                parts.push(format!("Died {} on turn {}.", cause_text, turn));
                break;
            }
        }
    }

    parts.join(" ")
}

// ─── Army Narrative ─────────────────────────────────────────────────────────

/// Generate prose about an army's status, composition, and history.
pub fn army_narrative(
    army_id: u32,
    owner_faction: u8,
    commander_id: Option<u32>,
    location: u16,
    levy: u32,
    men_at_arms: u32,
    knights: u16,
    morale: i64, // FixedPoint value (×1000)
    raised_turn: u32,
    events: &[GameEvent],
    ctx: &WorldContext,
) -> String {
    let faction_name = ctx.faction_name(owner_faction);
    let province_name = ctx.province_name(location);
    let total = levy + men_at_arms + knights as u32;

    let mut parts: Vec<String> = Vec::new();

    // Opening — identity and location
    let commander_text = match commander_id {
        Some(cid) => format!("led by {}", ctx.character_name(cid)),
        None => "without a named commander".to_string(),
    };
    parts.push(format!(
        "An army of {} {}, {} strong, encamped at {}.",
        faction_name, commander_text, total, province_name
    ));

    // Composition
    if knights > 0 && men_at_arms > 0 {
        parts.push(format!(
            "The host comprises {} levy, {} men-at-arms, and {} knight{}.",
            levy, men_at_arms, knights, if knights == 1 { "" } else { "s" }
        ));
    } else if men_at_arms > 0 {
        parts.push(format!(
            "The force includes {} levy and {} professional men-at-arms.",
            levy, men_at_arms
        ));
    }

    // Morale
    let morale_pct = morale / 10; // FixedPoint ÷ 10 → percentage
    if morale_pct >= 80 {
        parts.push("Morale is high — the soldiers are eager for battle.".to_string());
    } else if morale_pct >= 50 {
        parts.push("Morale holds steady, though the troops grow restless.".to_string());
    } else if morale_pct >= 20 {
        parts.push("Morale is wavering — discipline frays at the edges.".to_string());
    } else {
        parts.push("Morale has collapsed. Desertion looms.".to_string());
    }

    // Battle history
    let battles: Vec<_> = events.iter().filter_map(|e| match e {
        GameEvent::Battle(r) if r.attacker_army == army_id => {
            Some((true, r.attacker_won, r.attacker_casualties + r.defender_casualties, r.turn))
        }
        GameEvent::Battle(r) if r.defender_army == Some(army_id) => {
            Some((false, !r.attacker_won, r.attacker_casualties + r.defender_casualties, r.turn))
        }
        _ => None,
    }).collect();

    if !battles.is_empty() {
        let wins = battles.iter().filter(|(_, won, _, _)| *won).count();
        let losses = battles.len() - wins;
        parts.push(format!(
            "Battle record: {} victor{}, {} defeat{}.",
            wins, if wins == 1 { "y" } else { "ies" },
            losses, if losses == 1 { "" } else { "s" }
        ));
    }

    // Raised turn
    parts.push(format!("Raised on turn {}.", raised_turn));

    parts.join(" ")
}

// ─── Construction Narrative ─────────────────────────────────────────────────

/// Generate prose about construction activity in a province.
pub fn construction_narrative(
    province_id: u16,
    province_name: &str,
    current_improvements: &[&str],
    events: &[GameEvent],
    _ctx: &WorldContext,
) -> String {
    let mut completed: Vec<(String, u32)> = Vec::new(); // (improvement, turn)

    for event in events {
        if let GameEvent::ConstructionComplete { province, improvement, turn } = event {
            if *province == province_id {
                completed.push((improvement.clone(), *turn));
            }
        }
    }

    if current_improvements.is_empty() && completed.is_empty() {
        return String::new();
    }

    let mut parts: Vec<String> = Vec::new();

    // Current improvements
    if !current_improvements.is_empty() {
        if current_improvements.len() == 1 {
            parts.push(format!(
                "{} boasts a {}.",
                province_name, current_improvements[0]
            ));
        } else if current_improvements.len() <= 3 {
            parts.push(format!(
                "{} is home to {}.",
                province_name, current_improvements.join(", ")
            ));
        } else {
            parts.push(format!(
                "{} is a well-developed province with {} improvements: {}.",
                province_name, current_improvements.len(),
                current_improvements.join(", ")
            ));
        }
    }

    // Recent construction
    if let Some((last_imp, last_turn)) = completed.last() {
        parts.push(format!(
            "Most recently, a {} was completed on turn {}.",
            last_imp, last_turn
        ));
    }

    if completed.len() > 1 {
        parts.push(format!(
            "{} construction projects have been completed in total.",
            completed.len()
        ));
    }

    parts.join(" ")
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crown_ash_types::army::BattleResult;

    fn test_ctx() -> WorldContext {
        WorldContext {
            province_names: vec![
                (0, "Frosthold".into()),
                (7, "Ashenmere".into()),
            ],
            faction_names: vec![
                (0, "Ashen Crown".into()),
                (1, "Vale Princes".into()),
                (3, "Salt League".into()),
            ],
            character_names: vec![
                (1, "King Aldric".into()),
                (2, "Queen Isolde".into()),
            ],
            faction_cultures: vec![],
            army_factions: vec![(100, 0), (200, 1)],
            current_turn: 100,
        }
    }

    #[test]
    fn peaceful_province_history() {
        let ctx = test_ctx();
        let history = province_history(7, "Ashenmere", &[], &ctx);
        assert!(history.contains("peace"), "history={}", history);
    }

    #[test]
    fn province_with_conquests() {
        let ctx = test_ctx();
        let events = vec![
            GameEvent::ProvinceConquered {
                province: 7, old_controller: 0, new_controller: 1, turn: 20,
            },
            GameEvent::ProvinceConquered {
                province: 7, old_controller: 1, new_controller: 3, turn: 45,
            },
        ];
        let history = province_history(7, "Ashenmere", &events, &ctx);
        assert!(history.contains("changed hands 2 times"), "history={}", history);
        assert!(history.contains("Vale Princes"), "history={}", history);
        assert!(history.contains("Salt League"), "history={}", history);
    }

    #[test]
    fn province_with_battles_and_plagues() {
        let ctx = test_ctx();
        let events = vec![
            GameEvent::Battle(BattleResult {
                attacker_army: 100, defender_army: Some(200), province: 7,
                attacker_casualties: 50, defender_casualties: 30, attacker_won: true,
                random_factor: crown_ash_types::fixed_point::FixedPoint::from_int(1000), turn: 10,
            }),
            GameEvent::PlagueOutbreak {
                province: 7, severity: 500, population_lost: 200, turn: 30,
            },
            GameEvent::Famine { province: 7, severity: 300, turn: 40 },
        ];
        let history = province_history(7, "Ashenmere", &events, &ctx);
        assert!(history.contains("1 battle"), "history={}", history);
        assert!(history.contains("80 lives"), "history={}", history);
        assert!(history.contains("1 plague"), "history={}", history);
        assert!(history.contains("1 famine"), "history={}", history);
    }

    #[test]
    fn faction_conquest_history() {
        let ctx = test_ctx();
        let events = vec![
            GameEvent::ProvinceConquered {
                province: 7, old_controller: 1, new_controller: 0, turn: 10,
            },
            GameEvent::ProvinceConquered {
                province: 0, old_controller: 3, new_controller: 0, turn: 20,
            },
            GameEvent::WarDeclared {
                attacker: 0, defender: 1, casus_belli: "Conquest".into(), turn: 5,
            },
            GameEvent::TreatySigned {
                faction_a: 0, faction_b: 1, treaty_type: "White Peace".into(), turn: 25,
            },
        ];
        let history = faction_history(0, "Ashen Crown", &events, &ctx);
        assert!(history.contains("conquered 2 provinces"), "history={}", history);
        assert!(history.contains("declared 1 war"), "history={}", history);
        assert!(history.contains("signed 1 treaty"), "history={}", history);
        assert!(history.contains("expanding"), "history={}", history);
    }

    #[test]
    fn eliminated_faction_history() {
        let ctx = test_ctx();
        let events = vec![
            GameEvent::FactionEliminated { faction: 1, turn: 80 },
        ];
        let history = faction_history(1, "Vale Princes", &events, &ctx);
        assert!(history.contains("no more"), "history={}", history);
        assert!(history.contains("turn 80"), "history={}", history);
    }

    #[test]
    fn faction_with_internal_strife() {
        let ctx = test_ctx();
        let events = vec![
            GameEvent::SuccessionCrisis {
                faction: 0, dead_ruler: 1, claimants: vec![2, 3], realm_split: false, turn: 30,
            },
            GameEvent::RealmSplit {
                original_faction: 0, new_faction: 5, rebel_leader: 3, provinces_lost: 2, turn: 35,
            },
            GameEvent::MarriageAlliance {
                character_a: 1, character_b: 2, faction_a: 0, faction_b: 1, turn: 50,
            },
        ];
        let history = faction_history(0, "Ashen Crown", &events, &ctx);
        assert!(history.contains("succession"), "history={}", history);
        assert!(history.contains("realm split"), "history={}", history);
        assert!(history.contains("marriage"), "history={}", history);
    }

    // ── Turn Summary Tests ──

    #[test]
    fn empty_turn_summary() {
        let ctx = test_ctx();
        let summary = turn_summary(10, &[], &ctx);
        assert!(summary.contains("Turn 10"), "summary={}", summary);
        assert!(summary.contains("Nothing of note"), "summary={}", summary);
    }

    #[test]
    fn turn_with_battle_and_conquest() {
        let ctx = test_ctx();
        let events = vec![
            GameEvent::Battle(BattleResult {
                attacker_army: 100, defender_army: Some(200), province: 7,
                attacker_casualties: 50, defender_casualties: 30, attacker_won: true,
                random_factor: crown_ash_types::fixed_point::FixedPoint::from_int(1000), turn: 42,
            }),
            GameEvent::ProvinceConquered {
                province: 7, old_controller: 1, new_controller: 0, turn: 42,
            },
        ];
        let summary = turn_summary(42, &events, &ctx);
        assert!(summary.contains("Turn 42"), "summary={}", summary);
        assert!(summary.contains("battle"), "summary={}", summary);
        assert!(summary.contains("80 dead"), "summary={}", summary);
        assert!(summary.contains("Ashen Crown") || summary.contains("seized"), "summary={}", summary);
    }

    #[test]
    fn turn_with_faction_eliminated() {
        let ctx = test_ctx();
        let events = vec![
            GameEvent::FactionEliminated { faction: 1, turn: 50 },
        ];
        let summary = turn_summary(50, &events, &ctx);
        assert!(summary.contains("Vale Princes"), "summary={}", summary);
        assert!(summary.contains("destroyed"), "summary={}", summary);
    }

    #[test]
    fn turn_with_war_and_death() {
        let ctx = test_ctx();
        let events = vec![
            GameEvent::WarDeclared {
                attacker: 0, defender: 3, casus_belli: "Conquest".into(), turn: 30,
            },
            GameEvent::CharacterDied {
                character_id: 1, character_name: "King Aldric".into(),
                cause: crown_ash_types::DeathCause::Battle, turn: 30,
            },
        ];
        let summary = turn_summary(30, &events, &ctx);
        assert!(summary.contains("Ashen Crown"), "summary={}", summary);
        assert!(summary.contains("declared war"), "summary={}", summary);
        assert!(summary.contains("King Aldric"), "summary={}", summary);
    }

    // ── War Summary Tests ──

    #[test]
    fn war_summary_basic() {
        let ctx = test_ctx();
        let events = vec![
            GameEvent::WarDeclared {
                attacker: 0, defender: 1, casus_belli: "Conquest".into(), turn: 10,
            },
            GameEvent::Battle(BattleResult {
                attacker_army: 100, defender_army: Some(200), province: 7,
                attacker_casualties: 60, defender_casualties: 40, attacker_won: true,
                random_factor: crown_ash_types::fixed_point::FixedPoint::from_int(1000), turn: 12,
            }),
            GameEvent::ProvinceConquered {
                province: 7, old_controller: 1, new_controller: 0, turn: 12,
            },
        ];
        let summary = war_summary(0, 1, &events, &ctx);
        assert!(summary.contains("turn 10"), "summary={}", summary);
        assert!(summary.contains("Conquest"), "summary={}", summary);
        assert!(summary.contains("1 battle"), "summary={}", summary);
        assert!(summary.contains("100 lives"), "summary={}", summary);
        assert!(summary.contains("Ashen Crown seized 1 province"), "summary={}", summary);
    }

    #[test]
    fn war_summary_with_peace_treaty() {
        let ctx = test_ctx();
        let events = vec![
            GameEvent::WarDeclared {
                attacker: 1, defender: 0, casus_belli: "Border Dispute".into(), turn: 5,
            },
            GameEvent::TreatySigned {
                faction_a: 0, faction_b: 1, treaty_type: "White Peace".into(), turn: 20,
            },
        ];
        let summary = war_summary(0, 1, &events, &ctx);
        assert!(summary.contains("White Peace"), "summary={}", summary);
        assert!(summary.contains("turn 20"), "summary={}", summary);
        assert!(summary.contains("ending the bloodshed"), "summary={}", summary);
    }

    #[test]
    fn war_summary_no_war_declared() {
        let ctx = test_ctx();
        let summary = war_summary(0, 1, &[], &ctx);
        assert!(summary.contains("locked in conflict"), "summary={}", summary);
    }

    // ─── Relationship narrative tests ────────────────────────────────────

    #[test]
    fn relationship_empty_relations() {
        let ctx = test_ctx();
        let text = relationship_narrative(100, &[], &[], &ctx);
        assert!(text.is_empty());
    }

    #[test]
    fn relationship_friend_with_event_context() {
        let ctx = test_ctx();
        let relations = vec![
            (101, Some(crown_ash_types::character::RelationType::Friend), 600),
        ];
        let events = vec![
            GameEvent::Friendship { character_a: 100, character_b: 101, turn: 12 },
        ];
        let text = relationship_narrative(100, &relations, &events, &ctx);
        assert!(text.contains("allies"), "text={}", text);
        assert!(text.contains("since turn 12"), "text={}", text);
    }

    #[test]
    fn relationship_rival_deep_hatred() {
        let ctx = test_ctx();
        let relations = vec![
            (101, Some(crown_ash_types::character::RelationType::Rival), -800),
        ];
        let text = relationship_narrative(100, &relations, &[], &ctx);
        assert!(text.contains("hatred"), "text={}", text);
    }

    #[test]
    fn relationship_marriage_alliance_with_factions() {
        let ctx = test_ctx();
        let relations = vec![
            (101, Some(crown_ash_types::character::RelationType::MarriageAlliance), 500),
        ];
        let events = vec![
            GameEvent::MarriageAlliance {
                character_a: 100, character_b: 101,
                faction_a: 0, faction_b: 1, turn: 8,
            },
        ];
        let text = relationship_narrative(100, &relations, &events, &ctx);
        assert!(text.contains("wed"), "text={}", text);
        assert!(text.contains("Ashen Crown"), "text={}", text);
        assert!(text.contains("Vale Princes"), "text={}", text);
    }

    #[test]
    fn relationship_mentor() {
        let ctx = test_ctx();
        let relations = vec![
            (101, Some(crown_ash_types::character::RelationType::Mentor), 400),
        ];
        let text = relationship_narrative(100, &relations, &[], &ctx);
        assert!(text.contains("mentor"), "text={}", text);
    }

    #[test]
    fn relationship_neutral_skipped() {
        let ctx = test_ctx();
        let relations = vec![
            (101, None, 0), // Neutral, no type — should be skipped
        ];
        let text = relationship_narrative(100, &relations, &[], &ctx);
        assert!(text.is_empty(), "neutral should be skipped, got: {}", text);
    }

    // ─── Dynasty lineage tests ──────────────────────────────────────────

    #[test]
    fn dynasty_lineage_empty() {
        let ctx = test_ctx();
        let text = dynasty_lineage(100, 1, &[], &ctx);
        assert!(text.is_empty());
    }

    #[test]
    fn dynasty_lineage_with_parent_death() {
        let ctx = test_ctx();
        let events = vec![
            GameEvent::CharacterBorn {
                character_id: 99, character_name: "King Aldric".into(),
                parent: 0, dynasty: 1, turn: 1,
            },
            GameEvent::CharacterBorn {
                character_id: 100, character_name: "Prince Edric".into(),
                parent: 99, dynasty: 1, turn: 15,
            },
            GameEvent::CharacterDied {
                character_id: 99, character_name: "King Aldric".into(),
                cause: crown_ash_types::DeathCause::Battle, turn: 30,
            },
        ];
        let text = dynasty_lineage(100, 1, &events, &ctx);
        assert!(text.contains("Second"), "text={}", text);
        assert!(text.contains("fell in battle"), "text={}", text);
    }

    #[test]
    fn dynasty_lineage_founder() {
        let ctx = test_ctx();
        let events = vec![
            GameEvent::CharacterBorn {
                character_id: 100, character_name: "Lord Varen".into(),
                parent: 0, dynasty: 3, turn: 1,
            },
        ];
        let text = dynasty_lineage(100, 3, &events, &ctx);
        assert!(text.contains("Founder"), "text={}", text);
    }

    #[test]
    fn dynasty_lineage_counts_deaths() {
        let ctx = test_ctx();
        let events = vec![
            GameEvent::CharacterBorn {
                character_id: 50, character_name: "Ancestor".into(),
                parent: 0, dynasty: 2, turn: 1,
            },
            GameEvent::CharacterBorn {
                character_id: 51, character_name: "Sibling".into(),
                parent: 50, dynasty: 2, turn: 5,
            },
            GameEvent::CharacterBorn {
                character_id: 100, character_name: "Hero".into(),
                parent: 50, dynasty: 2, turn: 10,
            },
            GameEvent::CharacterDied {
                character_id: 50, character_name: "Ancestor".into(),
                cause: crown_ash_types::DeathCause::OldAge, turn: 20,
            },
            GameEvent::CharacterDied {
                character_id: 51, character_name: "Sibling".into(),
                cause: crown_ash_types::DeathCause::Disease, turn: 25,
            },
        ];
        let text = dynasty_lineage(100, 2, &events, &ctx);
        assert!(text.contains("Third"), "text={}", text);
        assert!(text.contains("2 members"), "text={}", text);
        assert!(text.contains("have perished"), "text={}", text);
    }

    // ─── Realm prosperity tests ─────────────────────────────────────────

    #[test]
    fn prosperity_empty_realm() {
        let ctx = test_ctx();
        let text = realm_prosperity(0, &[0, 7], &[], &ctx);
        assert!(text.contains("quiet stability"), "text={}", text);
    }

    #[test]
    fn prosperity_thriving_realm() {
        let ctx = test_ctx();
        let events = vec![
            GameEvent::Harvest { province: 0, prosperity_gain: 100, turn: 5 },
            GameEvent::Harvest { province: 7, prosperity_gain: 80, turn: 6 },
            GameEvent::TradeRouteEstablished { from: 0, to: 7, goods: "grain".into(), turn: 8 },
            GameEvent::ConstructionComplete { province: 0, improvement: "Market".into(), turn: 10 },
        ];
        let text = realm_prosperity(0, &[0, 7], &events, &ctx);
        assert!(text.contains("prospers"), "text={}", text);
        assert!(text.contains("2 bountiful harvests"), "text={}", text);
        assert!(text.contains("trade route"), "text={}", text);
        assert!(text.contains("1 improvement"), "text={}", text);
    }

    #[test]
    fn prosperity_suffering_realm() {
        let ctx = test_ctx();
        let events = vec![
            GameEvent::Famine { province: 0, severity: 500, turn: 3 },
            GameEvent::PlagueOutbreak { province: 7, severity: 700, population_lost: 200, turn: 5 },
            GameEvent::Rebellion { province: 0, rebels: 50, turn: 7 },
            GameEvent::Famine { province: 7, severity: 300, turn: 9 },
        ];
        let text = realm_prosperity(0, &[0, 7], &events, &ctx);
        assert!(text.contains("hardship") || text.contains("difficult"), "text={}", text);
        assert!(text.contains("famine"), "text={}", text);
        assert!(text.contains("Plague"), "text={}", text);
    }

    // ─── Battle report tests ────────────────────────────────────────────

    #[test]
    fn battle_report_basic() {
        let ctx = test_ctx();
        let result = BattleResult {
            province: 0,
            attacker_army: 100,
            defender_army: Some(200),
            attacker_casualties: 50,
            defender_casualties: 120,
            attacker_won: true,
            random_factor: crown_ash_types::FixedPoint(1000),
            turn: 15,
        };
        let text = battle_report(&result, &ctx);
        assert!(text.contains("Battle of Frosthold"), "text={}", text);
        assert!(text.contains("Ashen Crown"), "text={}", text);
        assert!(text.contains("Vale Princes"), "text={}", text);
        assert!(text.contains("170 lives"), "text={}", text);
        assert!(text.contains("turn 15"), "text={}", text);
    }

    #[test]
    fn battle_report_defender_wins() {
        let ctx = test_ctx();
        let result = BattleResult {
            province: 7,
            attacker_army: 100,
            defender_army: Some(200),
            attacker_casualties: 300,
            defender_casualties: 30,
            attacker_won: false,
            random_factor: crown_ash_types::FixedPoint(1000),
            turn: 22,
        };
        let text = battle_report(&result, &ctx);
        assert!(text.contains("held firm"), "text={}", text);
        assert!(text.contains("devastating casualties"), "text={}", text);
    }

    #[test]
    fn battle_report_devastating_victory() {
        let ctx = test_ctx();
        let result = BattleResult {
            province: 0,
            attacker_army: 100,
            defender_army: Some(200),
            attacker_casualties: 20,
            defender_casualties: 600,
            attacker_won: true,
            random_factor: crown_ash_types::FixedPoint(1000),
            turn: 30,
        };
        let text = battle_report(&result, &ctx);
        assert!(text.contains("crushing victory"), "text={}", text);
    }

    // ─── Intrigue narrative tests ───────────────────────────────────────

    #[test]
    fn intrigue_assassination_success() {
        let ctx = test_ctx();
        let events = vec![
            GameEvent::PlotSucceeded {
                instigator_name: "Lord Blackwood".into(),
                target_name: "King Aldric".into(),
                plot_type: "Assassination".into(),
                turn: 18,
            },
        ];
        let results = intrigue_narrative(&events, &ctx);
        assert_eq!(results.len(), 1);
        assert!(results[0].1.contains("assassination"), "text={}", results[0].1);
        assert!(results[0].1.contains("Lord Blackwood"), "text={}", results[0].1);
    }

    #[test]
    fn intrigue_discovered_and_foiled() {
        let ctx = test_ctx();
        let events = vec![
            GameEvent::PlotDiscovered {
                instigator_name: "Spy".into(),
                target_name: "Duke".into(),
                discovered_by: "Spymaster".into(),
                turn: 10,
            },
            GameEvent::PlotFoiled {
                instigator_name: "Assassin".into(),
                target_name: "Queen".into(),
                turn: 15,
            },
        ];
        let results = intrigue_narrative(&events, &ctx);
        assert_eq!(results.len(), 2);
        assert!(results[0].1.contains("uncovered"), "text={}", results[0].1);
        assert!(results[1].1.contains("foiled"), "text={}", results[1].1);
    }

    #[test]
    fn intrigue_no_events() {
        let ctx = test_ctx();
        let results = intrigue_narrative(&[], &ctx);
        assert!(results.is_empty());
    }

    // ─── Era summary tests ────────────────────────────────────────────

    #[test]
    fn era_summary_peaceful() {
        let ctx = test_ctx();
        let text = era_summary(50, 3, &[], &[(0, 5), (1, 3), (2, 2)], &[], &ctx);
        assert!(text.contains("Turn 50"), "text={}", text);
        assert!(text.contains("3 factions"), "text={}", text);
        assert!(text.contains("peace"), "text={}", text);
    }

    #[test]
    fn era_summary_with_war_and_battles() {
        let ctx = test_ctx();
        let events = vec![
            GameEvent::WarDeclared { attacker: 0, defender: 1, casus_belli: "Conquest".into(), turn: 10 },
            GameEvent::Battle(BattleResult {
                attacker_army: 100, defender_army: Some(200), province: 0,
                attacker_casualties: 80, defender_casualties: 120, attacker_won: true,
                random_factor: crown_ash_types::FixedPoint(1000), turn: 12,
            }),
            GameEvent::FactionEliminated { faction: 2, turn: 20 },
        ];
        let text = era_summary(25, 2, &[(0, 1)], &[(0, 7), (1, 3)], &events, &ctx);
        assert!(text.contains("War rages"), "text={}", text);
        assert!(text.contains("1 battle"), "text={}", text);
        assert!(text.contains("200 lives"), "text={}", text);
        assert!(text.contains("1 faction"), "text={}", text);
    }

    #[test]
    fn era_summary_dominant_empire() {
        let ctx = test_ctx();
        let text = era_summary(100, 4, &[], &[(0, 12), (1, 2), (2, 1)], &[], &ctx);
        assert!(text.contains("dominates"), "text={}", text);
        assert!(text.contains("Ashen Crown"), "text={}", text);
        assert!(text.contains("12 provinces"), "text={}", text);
    }

    // ─── Religion narrative tests ─────────────────────────────────────

    #[test]
    fn religion_steadfast() {
        let ctx = test_ctx();
        let text = religion_narrative(0, "Frosthold", "Order of the Flame", &[], &ctx);
        assert!(text.contains("steadfast"), "text={}", text);
        assert!(text.contains("Order of the Flame"), "text={}", text);
    }

    #[test]
    fn religion_single_conversion() {
        let ctx = test_ctx();
        let events = vec![
            GameEvent::ReligiousConversion {
                province: 0, old_religion: "Old Gods".into(),
                new_religion: "Order of the Flame".into(), turn: 15,
            },
        ];
        let text = religion_narrative(0, "Frosthold", "Order of the Flame", &events, &ctx);
        assert!(text.contains("converted from Old Gods"), "text={}", text);
        assert!(text.contains("turn 15"), "text={}", text);
    }

    #[test]
    fn religion_multiple_conversions() {
        let ctx = test_ctx();
        let events = vec![
            GameEvent::ReligiousConversion {
                province: 0, old_religion: "Old Gods".into(),
                new_religion: "Flame".into(), turn: 10,
            },
            GameEvent::ReligiousConversion {
                province: 0, old_religion: "Flame".into(),
                new_religion: "Shadow Cult".into(), turn: 30,
            },
        ];
        let text = religion_narrative(0, "Frosthold", "Shadow Cult", &events, &ctx);
        assert!(text.contains("changed faith 2 times"), "text={}", text);
        assert!(text.contains("Shadow Cult"), "text={}", text);
    }

    #[test]
    fn religion_heresies_and_miracles() {
        let ctx = test_ctx();
        let events = vec![
            GameEvent::Heresy { faction: 0, province: 0, severity: 100, turn: 5 },
            GameEvent::Miracle { province: 0, prosperity_gain: 50, turn: 10 },
            GameEvent::Miracle { province: 0, prosperity_gain: 30, turn: 20 },
        ];
        let text = religion_narrative(0, "Frosthold", "Order of the Flame", &events, &ctx);
        assert!(text.contains("Heretical"), "text={}", text);
        assert!(text.contains("2 miracles"), "text={}", text);
    }

    #[test]
    fn religion_ignores_other_provinces() {
        let ctx = test_ctx();
        let events = vec![
            GameEvent::ReligiousConversion {
                province: 7, old_religion: "Old Gods".into(),
                new_religion: "Flame".into(), turn: 10,
            },
        ];
        let text = religion_narrative(0, "Frosthold", "Order of the Flame", &events, &ctx);
        assert!(text.contains("steadfast"), "text={}", text); // no events for province 0
    }

    // ─── Diplomacy narrative tests ────────────────────────────────────

    #[test]
    fn diplomacy_neutral() {
        let ctx = test_ctx();
        let text = diplomacy_narrative(0, 1, false, &[], &ctx);
        assert!(text.contains("cautious neutrality"), "text={}", text);
        assert!(text.contains("Ashen Crown"), "text={}", text);
        assert!(text.contains("Vale Princes"), "text={}", text);
    }

    #[test]
    fn diplomacy_at_war_first_time() {
        let ctx = test_ctx();
        let events = vec![
            GameEvent::WarDeclared { attacker: 0, defender: 1, casus_belli: "Conquest".into(), turn: 5 },
        ];
        let text = diplomacy_narrative(0, 1, true, &events, &ctx);
        assert!(text.contains("at war"), "text={}", text);
    }

    #[test]
    fn diplomacy_repeated_wars() {
        let ctx = test_ctx();
        let events = vec![
            GameEvent::WarDeclared { attacker: 0, defender: 1, casus_belli: "X".into(), turn: 5 },
            GameEvent::TreatySigned { faction_a: 0, faction_b: 1, treaty_type: "White Peace".into(), turn: 10 },
            GameEvent::WarDeclared { attacker: 1, defender: 0, casus_belli: "Y".into(), turn: 20 },
        ];
        let text = diplomacy_narrative(0, 1, true, &events, &ctx);
        assert!(text.contains("2nd conflict"), "text={}", text);
    }

    #[test]
    fn diplomacy_marriage_alliance() {
        let ctx = test_ctx();
        let events = vec![
            GameEvent::MarriageAlliance {
                character_a: 100, character_b: 101,
                faction_a: 0, faction_b: 1, turn: 8,
            },
        ];
        let text = diplomacy_narrative(0, 1, false, &events, &ctx);
        assert!(text.contains("marriage alliance"), "text={}", text);
    }

    #[test]
    fn diplomacy_territory_conquest() {
        let ctx = test_ctx();
        let events = vec![
            GameEvent::WarDeclared { attacker: 0, defender: 1, casus_belli: "X".into(), turn: 5 },
            GameEvent::ProvinceConquered { province: 7, old_controller: 1, new_controller: 0, turn: 8 },
            GameEvent::ProvinceConquered { province: 3, old_controller: 1, new_controller: 0, turn: 9 },
            GameEvent::TreatySigned { faction_a: 0, faction_b: 1, treaty_type: "Peace".into(), turn: 15 },
        ];
        let text = diplomacy_narrative(0, 1, false, &events, &ctx);
        assert!(text.contains("seized 2 provinces"), "text={}", text);
    }

    // ─── Siege narrative tests ────────────────────────────────────────

    #[test]
    fn siege_empty() {
        let ctx = test_ctx();
        let text = siege_narrative(0, "Frosthold", &[], &ctx);
        assert!(text.is_empty());
    }

    #[test]
    fn siege_completed() {
        let ctx = test_ctx();
        let events = vec![
            GameEvent::SiegeStarted {
                province: 0, attacker_faction: 1, defender_faction: 0,
                turns_required: 6, turn: 10,
            },
            GameEvent::SiegeCompleted {
                province: 0, old_controller: 0, new_controller: 1,
                turns_lasted: 5, attacker_casualties: 30, turn: 15,
            },
        ];
        let text = siege_narrative(0, "Frosthold", &events, &ctx);
        assert!(text.contains("tested once"), "text={}", text);
        assert!(text.contains("Vale Princes"), "text={}", text);
        assert!(text.contains("5 turns"), "text={}", text);
        assert!(text.contains("30 casualties"), "text={}", text);
    }

    #[test]
    fn siege_ongoing() {
        let ctx = test_ctx();
        let events = vec![
            GameEvent::SiegeStarted {
                province: 0, attacker_faction: 1, defender_faction: 0,
                turns_required: 8, turn: 20,
            },
        ];
        let text = siege_narrative(0, "Frosthold", &events, &ctx);
        assert!(text.contains("currently besieges"), "text={}", text);
        assert!(text.contains("8 turns"), "text={}", text);
    }

    #[test]
    fn siege_multiple() {
        let ctx = test_ctx();
        let events = vec![
            GameEvent::SiegeStarted {
                province: 0, attacker_faction: 1, defender_faction: 0,
                turns_required: 4, turn: 5,
            },
            GameEvent::SiegeCompleted {
                province: 0, old_controller: 0, new_controller: 1,
                turns_lasted: 4, attacker_casualties: 20, turn: 9,
            },
            GameEvent::SiegeStarted {
                province: 0, attacker_faction: 0, defender_faction: 1,
                turns_required: 6, turn: 15,
            },
            GameEvent::SiegeCompleted {
                province: 0, old_controller: 1, new_controller: 0,
                turns_lasted: 5, attacker_casualties: 40, turn: 20,
            },
        ];
        let text = siege_narrative(0, "Frosthold", &events, &ctx);
        assert!(text.contains("2 sieges"), "text={}", text);
    }

    // ─── Trade narrative tests ────────────────────────────────────────

    #[test]
    fn trade_empty() {
        let ctx = test_ctx();
        let text = trade_narrative(0, "Frosthold", &[], &ctx);
        assert!(text.is_empty());
    }

    #[test]
    fn trade_established() {
        let ctx = test_ctx();
        let events = vec![
            GameEvent::TradeRouteEstablished {
                from: 0, to: 7, goods: "grain".into(), turn: 10,
            },
        ];
        let text = trade_narrative(0, "Frosthold", &events, &ctx);
        assert!(text.contains("commerce"), "text={}", text);
        assert!(text.contains("grain"), "text={}", text);
        assert!(text.contains("Ashenmere"), "text={}", text);
    }

    #[test]
    fn trade_disrupted() {
        let ctx = test_ctx();
        let events = vec![
            GameEvent::TradeRouteEstablished {
                from: 0, to: 7, goods: "iron".into(), turn: 5,
            },
            GameEvent::TradeRouteDisrupted {
                from: 0, to: 7, reason: "war".into(), turn: 12,
            },
        ];
        let text = trade_narrative(0, "Frosthold", &events, &ctx);
        assert!(text.contains("disrupted"), "text={}", text);
        assert!(text.contains("war"), "text={}", text);
    }

    // ─── Succession narrative tests ───────────────────────────────────

    #[test]
    fn succession_empty() {
        let ctx = test_ctx();
        let text = succession_narrative(0, &[], &ctx);
        assert!(text.is_empty());
    }

    #[test]
    fn succession_single_crisis_held() {
        let ctx = test_ctx();
        let events = vec![
            GameEvent::SuccessionCrisis {
                faction: 0, dead_ruler: 100, claimants: vec![101, 102],
                realm_split: false, turn: 15,
            },
        ];
        let text = succession_narrative(0, &events, &ctx);
        assert!(text.contains("death of"), "text={}", text);
        assert!(text.contains("held together"), "text={}", text);
        assert!(text.contains("2 claimants"), "text={}", text);
    }

    #[test]
    fn succession_realm_split() {
        let ctx = test_ctx();
        let events = vec![
            GameEvent::SuccessionCrisis {
                faction: 0, dead_ruler: 100, claimants: vec![101, 102, 103],
                realm_split: true, turn: 20,
            },
            GameEvent::RealmSplit {
                original_faction: 0, new_faction: 2, rebel_leader: 102,
                provinces_lost: 3, turn: 20,
            },
        ];
        let text = succession_narrative(0, &events, &ctx);
        assert!(text.contains("torn asunder"), "text={}", text);
        assert!(text.contains("3 provinces"), "text={}", text);
    }

    #[test]
    fn succession_multiple_crises() {
        let ctx = test_ctx();
        let events = vec![
            GameEvent::SuccessionCrisis {
                faction: 0, dead_ruler: 100, claimants: vec![101],
                realm_split: false, turn: 10,
            },
            GameEvent::SuccessionCrisis {
                faction: 0, dead_ruler: 101, claimants: vec![102, 103],
                realm_split: true, turn: 30,
            },
        ];
        let text = succession_narrative(0, &events, &ctx);
        assert!(text.contains("2 succession crises"), "text={}", text);
        assert!(text.contains("1 of which"), "text={}", text);
    }

    // ─── Character biography tests ────────────────────────────────────

    #[test]
    fn biography_living_ruler() {
        let ctx = test_ctx();
        let text = character_biography(
            100, "King Aldric", 45, "Ruler", 0, true,
            &["Brave", "Just"], &[], &ctx,
        );
        assert!(text.contains("King Aldric, aged 45"), "text={}", text);
        assert!(text.contains("Ruler of Ashen Crown"), "text={}", text);
        assert!(text.contains("Brave and Just"), "text={}", text);
    }

    #[test]
    fn biography_dead_character() {
        let ctx = test_ctx();
        let events = vec![
            GameEvent::CharacterDied {
                character_id: 100, character_name: "King Aldric".into(),
                cause: crown_ash_types::DeathCause::Assassination, turn: 30,
            },
        ];
        let text = character_biography(
            100, "King Aldric", 50, "Ruler", 0, false,
            &[], &events, &ctx,
        );
        assert!(text.contains("was a Ruler"), "text={}", text);
        assert!(text.contains("by assassination"), "text={}", text);
        assert!(text.contains("turn 30"), "text={}", text);
    }

    #[test]
    fn biography_with_marriage() {
        let ctx = test_ctx();
        let events = vec![
            GameEvent::MarriageAlliance {
                character_a: 100, character_b: 101,
                faction_a: 0, faction_b: 1, turn: 12,
            },
        ];
        let text = character_biography(
            100, "King Aldric", 35, "Ruler", 0, true,
            &[], &events, &ctx,
        );
        assert!(text.contains("Wed"), "text={}", text);
        assert!(text.contains("turn 12"), "text={}", text);
    }

    // ─── Army narrative tests ─────────────────────────────────────────

    #[test]
    fn army_basic() {
        let ctx = test_ctx();
        let text = army_narrative(
            100, 0, Some(100), 0,
            500, 100, 20, 850, 5, &[], &ctx,
        );
        assert!(text.contains("Ashen Crown"), "text={}", text);
        assert!(text.contains("620 strong"), "text={}", text);
        assert!(text.contains("Frosthold"), "text={}", text);
        assert!(text.contains("Morale is high"), "text={}", text);
        assert!(text.contains("Raised on turn 5"), "text={}", text);
    }

    #[test]
    fn army_with_battles() {
        let ctx = test_ctx();
        let events = vec![
            GameEvent::Battle(BattleResult {
                attacker_army: 100, defender_army: Some(200), province: 0,
                attacker_casualties: 50, defender_casualties: 80, attacker_won: true,
                random_factor: crown_ash_types::FixedPoint(1000), turn: 10,
            }),
            GameEvent::Battle(BattleResult {
                attacker_army: 200, defender_army: Some(100), province: 7,
                attacker_casualties: 60, defender_casualties: 40, attacker_won: true,
                random_factor: crown_ash_types::FixedPoint(1000), turn: 15,
            }),
        ];
        let text = army_narrative(
            100, 0, None, 0,
            300, 50, 0, 500, 3, &events, &ctx,
        );
        assert!(text.contains("1 victory"), "text={}", text);
        assert!(text.contains("1 defeat"), "text={}", text);
    }

    #[test]
    fn army_low_morale() {
        let ctx = test_ctx();
        let text = army_narrative(
            100, 0, None, 0,
            200, 0, 0, 150, 1, &[], &ctx,
        );
        assert!(text.contains("collapsed") || text.contains("Desertion"), "text={}", text);
    }

    // ─── Construction narrative tests ─────────────────────────────────

    #[test]
    fn construction_empty() {
        let ctx = test_ctx();
        let text = construction_narrative(0, "Frosthold", &[], &[], &ctx);
        assert!(text.is_empty());
    }

    #[test]
    fn construction_with_improvements() {
        let ctx = test_ctx();
        let text = construction_narrative(
            0, "Frosthold", &["Market", "Temple"], &[], &ctx,
        );
        assert!(text.contains("Market"), "text={}", text);
        assert!(text.contains("Temple"), "text={}", text);
    }

    #[test]
    fn construction_with_recent_build() {
        let ctx = test_ctx();
        let events = vec![
            GameEvent::ConstructionComplete {
                province: 0, improvement: "Fortification".into(), turn: 20,
            },
        ];
        let text = construction_narrative(
            0, "Frosthold", &["Fortification"], &events, &ctx,
        );
        assert!(text.contains("Fortification"), "text={}", text);
        assert!(text.contains("completed on turn 20"), "text={}", text);
    }

    #[test]
    fn construction_multiple_builds() {
        let ctx = test_ctx();
        let events = vec![
            GameEvent::ConstructionComplete {
                province: 0, improvement: "Market".into(), turn: 10,
            },
            GameEvent::ConstructionComplete {
                province: 0, improvement: "Temple".into(), turn: 15,
            },
            GameEvent::ConstructionComplete {
                province: 0, improvement: "University".into(), turn: 25,
            },
        ];
        let text = construction_narrative(
            0, "Frosthold", &["Market", "Temple", "University"], &events, &ctx,
        );
        assert!(text.contains("3 construction projects"), "text={}", text);
    }
}
