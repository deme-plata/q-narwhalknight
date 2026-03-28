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
}
