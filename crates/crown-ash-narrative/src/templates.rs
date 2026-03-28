//! Tier 1 — Template-based narrative generation.
//!
//! Handwritten prose templates for all 26 GameEvent variants. Each event type
//! has 2-4 template variants; the variant is selected deterministically by
//! hashing the turn number (so the same event always produces the same text
//! across all nodes — determinism preserved).
//!
//! Templates use simple string formatting — no regex, no parser, just
//! `format!()` with context lookups. This keeps Tier 1 at zero latency.

use crown_ash_types::GameEvent;
use crown_ash_types::event::DeathCause;
use crate::{Importance, WorldContext};

// ─── Template Selection ──────────────────────────────────────────────────────

/// Pick a template variant index deterministically from the turn number.
fn variant(turn: u32, num_variants: usize) -> usize {
    (turn as usize * 7919) % num_variants  // prime multiplier for spread
}

// ─── Full Prose Rendering ────────────────────────────────────────────────────

/// Render a GameEvent into rich narrative prose (Tier 1).
pub fn render(event: &GameEvent, ctx: &WorldContext) -> String {
    match event {
        // ── Battle ──────────────────────────────────────────────────────
        GameEvent::Battle(result) => {
            let prov = ctx.province_name(result.province);
            let attacker = ctx.army_faction_name(result.attacker_army);
            let defender = result.defender_army
                .map(|aid| ctx.army_faction_name(aid))
                .unwrap_or("the garrison");
            let total = result.attacker_casualties + result.defender_casualties;
            let victor = if result.attacker_won { attacker } else { defender };

            let templates = [
                format!(
                    "The fields of {} ran red with blood as {} clashed with {}. \
                     When the dust settled, {} held the field — but {} souls \
                     would never return to their families.",
                    prov, attacker, defender, victor, total
                ),
                format!(
                    "Steel rang against steel in {}. The host of {} met the \
                     defenders of {} in a brutal engagement. {} emerged victorious, \
                     though {} lay dead upon the churned earth.",
                    prov, attacker, defender, victor, total
                ),
                format!(
                    "A great battle was fought at {}. The banners of {} and {} \
                     clashed beneath a grey sky. {} claimed victory, but at \
                     a terrible cost — {} casualties between both sides.",
                    prov, attacker, defender, victor, total
                ),
            ];
            templates[variant(ctx.current_turn, templates.len())].clone()
        }

        // ── Province Conquered ──────────────────────────────────────────
        GameEvent::ProvinceConquered { province, old_controller, new_controller, turn } => {
            let prov = ctx.province_name(*province);
            let old = ctx.faction_name(*old_controller);
            let new = ctx.faction_name(*new_controller);

            let templates = [
                format!(
                    "The banners of {} have fallen over {}. Where once {} ruled, \
                     now {} plants its standard. The people watch in silence, \
                     uncertain what this change of masters will bring.",
                    old, prov, old, new
                ),
                format!(
                    "{} has been conquered! The forces of {} have swept aside \
                     {}'s garrison and claimed the province for their own. \
                     New laws, new taxes, new masters.",
                    prov, new, old
                ),
                format!(
                    "On turn {}, {} changed hands. {} wrested control from {}, \
                     adding another jewel to their growing domain. The old \
                     guard retreats to lick their wounds.",
                    turn, prov, new, old
                ),
            ];
            templates[variant(*turn, templates.len())].clone()
        }

        // ── War Declared ────────────────────────────────────────────────
        GameEvent::WarDeclared { attacker, defender, casus_belli, turn } => {
            let att = ctx.faction_name(*attacker);
            let def = ctx.faction_name(*defender);

            let templates = [
                format!(
                    "The drums of war thunder across the realm! {} has declared \
                     war upon {}, citing '{}' as justification. Armies muster, \
                     peasants flee, and the shadow of conflict falls over the land.",
                    att, def, casus_belli
                ),
                format!(
                    "\"There shall be no peace between us!\" With those words, \
                     {} committed to war against {}. The casus belli: {}. \
                     Blood will flow before this quarrel is settled.",
                    att, def, casus_belli
                ),
                format!(
                    "Turn {} marks a dark day. {} has raised its sword against {}, \
                     invoking the right of {}. Diplomats have failed. \
                     Now only steel will speak.",
                    turn, att, def, casus_belli
                ),
            ];
            templates[variant(*turn, templates.len())].clone()
        }

        // ── Treaty Signed ───────────────────────────────────────────────
        GameEvent::TreatySigned { faction_a, faction_b, treaty_type, turn } => {
            let a = ctx.faction_name(*faction_a);
            let b = ctx.faction_name(*faction_b);

            let templates = [
                format!(
                    "Peace at last! {} and {} have signed a {} treaty. \
                     The quills scratch parchment where swords once clashed. \
                     Whether this peace will hold remains to be seen.",
                    a, b, treaty_type
                ),
                format!(
                    "After much bloodshed, {} and {} have come to terms. \
                     A {} has been agreed upon. Soldiers lower their weapons, \
                     but few truly trust the silence.",
                    a, b, treaty_type
                ),
            ];
            templates[variant(*turn, templates.len())].clone()
        }

        // ── Character Died ──────────────────────────────────────────────
        GameEvent::CharacterDied { character_name, cause,  .. } => {
            let cause_text = match cause {
                DeathCause::OldAge => {
                    format!(
                        "{} has passed away in their bed, taken by the weight of years. \
                         A life lived fully, now ended. The realm mourns.",
                        character_name
                    )
                }
                DeathCause::Battle => {
                    format!(
                        "{} fell in battle, struck down amidst the chaos of war. \
                         Their body was carried from the field by loyal retainers. \
                         A warrior's death — but a loss all the same.",
                        character_name
                    )
                }
                DeathCause::Disease => {
                    format!(
                        "Disease has claimed {}. Despite the efforts of healers, \
                         the sickness proved too strong. Another light extinguished \
                         by the pestilence that stalks these lands.",
                        character_name
                    )
                }
                DeathCause::Assassination => {
                    format!(
                        "{} was found dead — a dagger in the dark has silenced them \
                         forever. Whispers of conspiracy fill the court. \
                         Who ordered the killing? The shadows hold their secrets.",
                        character_name
                    )
                }
                DeathCause::Execution => {
                    format!(
                        "{} was executed by order of their liege. The headsman's \
                         axe fell before a jeering crowd. Justice or tyranny? \
                         History will judge.",
                        character_name
                    )
                }
                DeathCause::Accident => {
                    format!(
                        "A tragic accident has taken {}. Some call it fate, \
                         others whisper of foul play. The truth may never \
                         be known.",
                        character_name
                    )
                }
            };
            cause_text
        }

        // ── Character Born ──────────────────────────────────────────────
        GameEvent::CharacterBorn { character_name, parent, turn, .. } => {
            let parent_name = ctx.character_name(*parent);

            let templates = [
                format!(
                    "A child is born! {} has entered the world, heir to {}. \
                     The dynasty grows stronger. What destiny awaits this \
                     new soul?",
                    character_name, parent_name
                ),
                format!(
                    "Rejoice! {} has been blessed with a child: {}. \
                     The nurses swaddle the babe while courtiers whisper \
                     about the future of the bloodline.",
                    parent_name, character_name
                ),
            ];
            templates[variant(*turn, templates.len())].clone()
        }

        // ── Succession Crisis ───────────────────────────────────────────
        GameEvent::SuccessionCrisis { faction, dead_ruler, claimants, realm_split, .. } => {
            let fac = ctx.faction_name(*faction);
            let ruler = ctx.character_name(*dead_ruler);
            let split_text = if *realm_split {
                " The realm has fractured — rival claimants carve out their own domains!"
            } else {
                " For now, the realm holds together, but tensions simmer."
            };

            format!(
                "Crisis! The death of {} has plunged {} into turmoil. \
                 {} claimants eye the throne, each believing the crown \
                 is rightfully theirs.{}",
                ruler, fac, claimants.len(), split_text
            )
        }

        // ── Plague Outbreak ─────────────────────────────────────────────
        GameEvent::PlagueOutbreak { province, severity, population_lost, turn } => {
            let prov = ctx.province_name(*province);
            let severity_word = if *severity > 600 {
                "devastating"
            } else if *severity > 300 {
                "severe"
            } else {
                "mild"
            };

            let templates = [
                format!(
                    "A {} plague sweeps through {}. The streets empty as \
                     the sick are carried to makeshift camps beyond the walls. \
                     {} souls are lost before the pestilence runs its course.",
                    severity_word, prov, population_lost
                ),
                format!(
                    "Death stalks {}. A {} outbreak of disease has claimed {} \
                     lives. The healers are overwhelmed, the gravediggers \
                     work through the night.",
                    prov, severity_word, population_lost
                ),
                format!(
                    "The black rot has come to {}. {} have perished in this \
                     {} epidemic. Funeral pyres light the horizon, and the \
                     surviving townsfolk pray for deliverance.",
                    prov, population_lost, severity_word
                ),
            ];
            templates[variant(*turn, templates.len())].clone()
        }

        // ── Famine ──────────────────────────────────────────────────────
        GameEvent::Famine { province, turn, .. } => {
            let prov = ctx.province_name(*province);

            let templates = [
                format!(
                    "Famine grips {}. The granaries are empty, the fields \
                     barren. Hunger drives the desperate to forage in the \
                     forests, and the weak begin to perish.",
                    prov
                ),
                format!(
                    "The harvest has failed in {}. Bread prices soar as \
                     merchants hoard what little remains. Unrest grows \
                     as bellies go empty.",
                    prov
                ),
            ];
            templates[variant(*turn, templates.len())].clone()
        }

        // ── Harvest ─────────────────────────────────────────────────────
        GameEvent::Harvest { province, turn, .. } => {
            let prov = ctx.province_name(*province);

            let templates = [
                format!(
                    "A bountiful harvest in {}! The fields overflow with grain, \
                     the orchards drip with fruit. Prosperity rises as the \
                     people celebrate their good fortune.",
                    prov
                ),
                format!(
                    "The gods smile upon {}. A magnificent harvest fills every \
                     barn and storehouse. Trade flourishes and the people \
                     give thanks.",
                    prov
                ),
            ];
            templates[variant(*turn, templates.len())].clone()
        }

        // ── Rebellion ───────────────────────────────────────────────────
        GameEvent::Rebellion { province, rebels, turn } => {
            let prov = ctx.province_name(*province);

            let templates = [
                format!(
                    "Rebellion in {}! {} angry peasants have taken up arms \
                     against their rulers. Pitchforks and torches fill the \
                     streets as the downtrodden demand justice.",
                    prov, rebels
                ),
                format!(
                    "The people of {} have had enough. {} rebels march on \
                     the keep, their fury born of years of taxation and \
                     neglect. The garrison scrambles to respond.",
                    prov, rebels
                ),
            ];
            templates[variant(*turn, templates.len())].clone()
        }

        // ── Player Joined ───────────────────────────────────────────────
        GameEvent::PlayerJoined { wallet, faction, .. } => {
            let fac = ctx.faction_name(*faction);
            let short_wallet = if wallet.len() > 10 {
                format!("{}...{}", &wallet[..6], &wallet[wallet.len()-4..])
            } else {
                wallet.clone()
            };

            format!(
                "A new ruler emerges! Wallet {} has claimed the throne of {}. \
                 The realm watches with anticipation as a new hand takes the reins of power.",
                short_wallet, fac
            )
        }

        // ── Construction Complete ────────────────────────────────────────
        GameEvent::ConstructionComplete { province, improvement, .. } => {
            let prov = ctx.province_name(*province);

            format!(
                "Construction complete! A new {} stands proudly in {}. \
                 The builders wipe their brows as the townsfolk gather \
                 to admire the new addition to their province.",
                improvement, prov
            )
        }

        // ── Faction Eliminated ──────────────────────────────────────────
        GameEvent::FactionEliminated { faction, turn } => {
            let fac = ctx.faction_name(*faction);

            let templates = [
                format!(
                    "The end has come for {}. Their last province has fallen, \
                     their armies scattered, their dynasty broken. Once a \
                     proud power, now nothing but a memory and a cautionary \
                     tale for the ages.",
                    fac
                ),
                format!(
                    "{} is no more. Stripped of every province, bereft of \
                     every ally, the once-mighty faction crumbles into dust. \
                     The vultures circle, and the survivors scatter to the winds.",
                    fac
                ),
            ];
            templates[variant(*turn, templates.len())].clone()
        }

        // ── Realm Split ─────────────────────────────────────────────────
        GameEvent::RealmSplit { original_faction, rebel_leader, provinces_lost, .. } => {
            let orig = ctx.faction_name(*original_faction);
            let rebel = ctx.character_name(*rebel_leader);

            format!(
                "The realm fractures! {} has led a breakaway from {}, \
                 carving out a new domain from {} provinces. A new faction \
                 rises from the ashes of the old — born of ambition, \
                 discontent, and the eternal hunger for power.",
                rebel, orig, provinces_lost
            )
        }

        // ── Plot Launched ───────────────────────────────────────────────
        GameEvent::PlotLaunched { instigator, target, plot_type, .. } => {
            let ins = ctx.character_name(*instigator);
            let tgt = ctx.character_name(*target);

            format!(
                "Shadows stir in the court. {} has begun plotting a {} \
                 against {}. Whispered meetings in dark corridors, \
                 coins changing hands, loyalties tested. The game of \
                 intrigue has begun.",
                ins, plot_type, tgt
            )
        }

        // ── Plot Succeeded ──────────────────────────────────────────────
        GameEvent::PlotSucceeded { instigator_name, target_name, plot_type, .. } => {
            format!(
                "The plot succeeds! {}'s {} against {} has been carried out. \
                 The schemer's patience and cunning have paid off. \
                 The court trembles at the implications.",
                instigator_name, plot_type, target_name
            )
        }

        // ── Plot Discovered ─────────────────────────────────────────────
        GameEvent::PlotDiscovered { instigator_name, target_name, discovered_by, .. } => {
            format!(
                "A conspiracy exposed! {} has uncovered {}'s plot against {}. \
                 The spymaster's network has proven its worth. \
                 Accusations fly and trust shatters like glass.",
                discovered_by, instigator_name, target_name
            )
        }

        // ── Plot Foiled ─────────────────────────────────────────────────
        GameEvent::PlotFoiled { instigator_name, target_name, .. } => {
            format!(
                "The plot has failed! {}'s scheme against {} was thwarted \
                 at the last moment. The would-be victim breathes a sigh \
                 of relief, while the plotter slinks back into the shadows.",
                instigator_name, target_name
            )
        }

        // ── Trade Route Established ─────────────────────────────────────
        GameEvent::TradeRouteEstablished { from, to, goods, .. } => {
            let f = ctx.province_name(*from);
            let t = ctx.province_name(*to);

            format!(
                "New trade route! Merchants now travel between {} and {}, \
                 carrying {} along dusty roads and river barges. \
                 Both provinces prosper from the exchange.",
                f, t, goods
            )
        }

        // ── Trade Route Disrupted ───────────────────────────────────────
        GameEvent::TradeRouteDisrupted { from, to, reason, .. } => {
            let f = ctx.province_name(*from);
            let t = ctx.province_name(*to);

            format!(
                "The trade route between {} and {} has been disrupted! \
                 {}: merchants turn back, goods rot in warehouses, \
                 and both provinces feel the economic sting.",
                f, t, reason
            )
        }

        // ── Character Tombstoned ────────────────────────────────────────
        GameEvent::CharacterTombstoned { character_name,  .. } => {
            format!(
                "{} has passed from living memory into legend. \
                 Their deeds — both noble and ignoble — are inscribed \
                 in the great chronicle for future generations to judge.",
                character_name
            )
        }

        // ── Army Auto-Disbanded ─────────────────────────────────────────
        GameEvent::ArmyAutoDisbanded { army_id, faction, troops_returned, province, .. } => {
            let fac = ctx.faction_name(*faction);
            let prov = ctx.province_name(*province);

            format!(
                "Army #{} of {} has been disbanded at {}. \
                 {} soldiers return to their homes, trading swords \
                 for plowshares — at least for now.",
                army_id, fac, prov, troops_returned
            )
        }

        // ── Religious Conversion ────────────────────────────────────────
        GameEvent::ReligiousConversion { province, old_religion, new_religion, .. } => {
            let prov = ctx.province_name(*province);

            format!(
                "The temples of {} in {} have been rededicated to {}. \
                 The old faith of {} fades as new prayers echo through \
                 the halls. Some accept the change; others mutter in defiance.",
                old_religion, prov, new_religion, old_religion
            )
        }

        // ── Heresy ──────────────────────────────────────────────────────
        GameEvent::Heresy { faction, province, .. } => {
            let fac = ctx.faction_name(*faction);
            let prov = ctx.province_name(*province);

            format!(
                "Heresy spreads through {}! In the shadow of {}'s cathedrals, \
                 forbidden doctrines take root. The clergy scramble to stamp \
                 out the dissent before it engulfs the faithful.",
                prov, fac
            )
        }

        // ── Miracle ─────────────────────────────────────────────────────
        GameEvent::Miracle { province, .. } => {
            let prov = ctx.province_name(*province);

            format!(
                "A miracle in {}! The faithful speak of divine intervention — \
                 a spring of clear water, a healing of the sick, a light \
                 in the cathedral at dawn. Prosperity blooms and pilgrims \
                 flock to witness the wonder.",
                prov
            )
        }

        // ── Siege Started ─────────────────────────────────────────────
        GameEvent::SiegeStarted { province, attacker_faction, defender_faction, turns_required, .. } => {
            let prov = ctx.province_name(*province);
            let att = ctx.faction_name(*attacker_faction);
            let def = ctx.faction_name(*defender_faction);

            let templates = [
                format!(
                    "The siege of {} has begun! The armies of {} encircle \
                     the fortifications held by {}. Sappers dig, trebuchets \
                     are assembled, and the garrison prepares for a long \
                     ordeal. The siege is expected to last {} turns.",
                    prov, att, def, turns_required
                ),
                format!(
                    "{}'s banners surround the walls of {}. The defenders of {} \
                     look out from the battlements as the besieging host \
                     settles in for a prolonged campaign — {} turns of \
                     hunger, fear, and the slow grind of attrition.",
                    att, prov, def, turns_required
                ),
            ];
            templates[variant(ctx.current_turn, templates.len())].clone()
        }

        // ── Siege Completed ───────────────────────────────────────────
        GameEvent::SiegeCompleted { province, old_controller, new_controller, turns_lasted, attacker_casualties, .. } => {
            let prov = ctx.province_name(*province);
            let old = ctx.faction_name(*old_controller);
            let new = ctx.faction_name(*new_controller);

            let templates = [
                format!(
                    "The siege of {} is over! After {} agonizing turns, the walls \
                     have fallen. {} wrests control from {}, though {} soldiers \
                     paid the price in the final assault. The gates are thrown \
                     open, and a new banner rises over the battered keep.",
                    prov, turns_lasted, new, old, attacker_casualties
                ),
                format!(
                    "At last, {} capitulates. {} turns of starvation and bombardment \
                     have broken the spirit of {}'s garrison. {} plants its standard \
                     on the shattered walls — but {} of their own lie dead in the \
                     ditches below.",
                    prov, turns_lasted, old, new, attacker_casualties
                ),
            ];
            templates[variant(ctx.current_turn, templates.len())].clone()
        }

        // ── Friendship ────────────────────────────────────────────────
        GameEvent::Friendship { character_a, character_b, .. } => {
            let a = ctx.character_name(*character_a);
            let b = ctx.character_name(*character_b);

            let templates = [
                format!(
                    "A bond of friendship has formed between {} and {}. \
                     Through shared counsel and mutual respect, these two \
                     have become trusted companions in a court full of vipers.",
                    a, b
                ),
                format!(
                    "{} and {} are now friends. In a realm where trust is rare \
                     and betrayal commonplace, their camaraderie is a beacon — \
                     or perhaps a target.",
                    a, b
                ),
            ];
            templates[variant(ctx.current_turn, templates.len())].clone()
        }

        // ── Rivalry ───────────────────────────────────────────────────
        GameEvent::Rivalry { character_a, character_b, .. } => {
            let a = ctx.character_name(*character_a);
            let b = ctx.character_name(*character_b);

            let templates = [
                format!(
                    "A bitter rivalry has ignited between {} and {}. \
                     Cold glances across the great hall, whispered insults, \
                     and barely concealed contempt — this feud will not \
                     end quietly.",
                    a, b
                ),
                format!(
                    "{} and {} have become rivals. What began as disagreement \
                     has curdled into genuine animosity. The court watches \
                     with equal parts dread and fascination.",
                    a, b
                ),
            ];
            templates[variant(ctx.current_turn, templates.len())].clone()
        }

        // ── Marriage Alliance ─────────────────────────────────────────
        GameEvent::MarriageAlliance { character_a, character_b, faction_a, faction_b, .. } => {
            let a = ctx.character_name(*character_a);
            let b = ctx.character_name(*character_b);
            let fa = ctx.faction_name(*faction_a);
            let fb = ctx.faction_name(*faction_b);

            let templates = [
                format!(
                    "A marriage alliance! {} of {} weds {} of {}. \
                     The union binds two houses together — a political \
                     calculus dressed in wedding finery. Whether love \
                     or ambition drives this match, only time will tell.",
                    a, fa, b, fb
                ),
                format!(
                    "The bells ring for {} and {}! This cross-faction marriage \
                     between {} and {} strengthens diplomatic ties. \
                     Dowries are exchanged, vows are spoken, and two \
                     realms breathe a little easier — for now.",
                    a, b, fa, fb
                ),
            ];
            templates[variant(ctx.current_turn, templates.len())].clone()
        }
    }
}

// ─── Short Summary Rendering ─────────────────────────────────────────────────

/// Render a one-line summary for the event feed (shorter than full prose).
pub fn render_summary(event: &GameEvent, ctx: &WorldContext) -> String {
    match event {
        GameEvent::Battle(r) => {
            let att = ctx.army_faction_name(r.attacker_army);
            let def = r.defender_army.map(|aid| ctx.army_faction_name(aid)).unwrap_or("the garrison");
            let vic = if r.attacker_won { att } else { def };
            format!("Battle at {} — {} victorious ({} casualties)", ctx.province_name(r.province), vic, r.attacker_casualties + r.defender_casualties)
        }
        GameEvent::ProvinceConquered { province, new_controller, .. } =>
            format!("{} conquered by {}", ctx.province_name(*province), ctx.faction_name(*new_controller)),
        GameEvent::WarDeclared { attacker, defender, .. } =>
            format!("{} declares war on {}", ctx.faction_name(*attacker), ctx.faction_name(*defender)),
        GameEvent::TreatySigned { faction_a, faction_b, treaty_type, .. } =>
            format!("{} signed between {} and {}", treaty_type, ctx.faction_name(*faction_a), ctx.faction_name(*faction_b)),
        GameEvent::CharacterDied { character_name, cause, .. } =>
            format!("{} died ({:?})", character_name, cause),
        GameEvent::CharacterBorn { character_name, .. } =>
            format!("{} was born", character_name),
        GameEvent::SuccessionCrisis { faction, realm_split, .. } => {
            let split = if *realm_split { " — realm split!" } else { "" };
            format!("Succession crisis in {}{}", ctx.faction_name(*faction), split)
        }
        GameEvent::PlagueOutbreak { province, population_lost, .. } =>
            format!("Plague in {} — {} dead", ctx.province_name(*province), population_lost),
        GameEvent::Famine { province, .. } =>
            format!("Famine strikes {}", ctx.province_name(*province)),
        GameEvent::Harvest { province, .. } =>
            format!("Bountiful harvest in {}", ctx.province_name(*province)),
        GameEvent::Rebellion { province, rebels, .. } =>
            format!("{} rebels rise in {}", rebels, ctx.province_name(*province)),
        GameEvent::PlayerJoined { faction, .. } =>
            format!("New player joins {}", ctx.faction_name(*faction)),
        GameEvent::ConstructionComplete { province, improvement, .. } =>
            format!("{} built in {}", improvement, ctx.province_name(*province)),
        GameEvent::FactionEliminated { faction, .. } =>
            format!("{} eliminated!", ctx.faction_name(*faction)),
        GameEvent::RealmSplit { original_faction, provinces_lost, .. } =>
            format!("{} loses {} provinces to rebellion", ctx.faction_name(*original_faction), provinces_lost),
        GameEvent::PlotLaunched { instigator, plot_type, .. } =>
            format!("{} plots {}", ctx.character_name(*instigator), plot_type),
        GameEvent::PlotSucceeded { instigator_name, plot_type, target_name, .. } =>
            format!("{}'s {} against {} succeeds", instigator_name, plot_type, target_name),
        GameEvent::PlotDiscovered { instigator_name, discovered_by, .. } =>
            format!("{}'s plot discovered by {}", instigator_name, discovered_by),
        GameEvent::PlotFoiled { instigator_name, target_name, .. } =>
            format!("{}'s plot against {} foiled", instigator_name, target_name),
        GameEvent::TradeRouteEstablished { from, to, goods, .. } =>
            format!("Trade ({}) between {} and {}", goods, ctx.province_name(*from), ctx.province_name(*to)),
        GameEvent::TradeRouteDisrupted { from, to, .. } =>
            format!("Trade disrupted: {} — {}", ctx.province_name(*from), ctx.province_name(*to)),
        GameEvent::CharacterTombstoned { character_name, .. } =>
            format!("{} passes into legend", character_name),
        GameEvent::ArmyAutoDisbanded { army_id, faction, .. } =>
            format!("Army #{} of {} disbanded", army_id, ctx.faction_name(*faction)),
        GameEvent::ReligiousConversion { province, new_religion, .. } =>
            format!("{} converts to {}", ctx.province_name(*province), new_religion),
        GameEvent::Heresy { province, .. } =>
            format!("Heresy spreads in {}", ctx.province_name(*province)),
        GameEvent::Miracle { province, .. } =>
            format!("Miracle in {}", ctx.province_name(*province)),
        GameEvent::SiegeStarted { province, attacker_faction, .. } =>
            format!("{} besieges {}", ctx.faction_name(*attacker_faction), ctx.province_name(*province)),
        GameEvent::SiegeCompleted { province, new_controller, turns_lasted, .. } =>
            format!("{} falls to {} after {} turns", ctx.province_name(*province), ctx.faction_name(*new_controller), turns_lasted),
        GameEvent::Friendship { character_a, character_b, .. } =>
            format!("{} and {} become friends", ctx.character_name(*character_a), ctx.character_name(*character_b)),
        GameEvent::Rivalry { character_a, character_b, .. } =>
            format!("{} and {} become rivals", ctx.character_name(*character_a), ctx.character_name(*character_b)),
        GameEvent::MarriageAlliance { character_a, character_b, faction_a, faction_b, .. } =>
            format!("Marriage: {} ({}) weds {} ({})", ctx.character_name(*character_a), ctx.faction_name(*faction_a), ctx.character_name(*character_b), ctx.faction_name(*faction_b)),
    }
}

// ─── LLM Prompt Builder ─────────────────────────────────────────────────────

/// Build an LLM prompt for Tier 2/3 narrative generation.
///
/// The prompt is structured as a medieval chronicler writing about the event.
/// It includes game context (faction names, character traits, situation) so
/// the LLM can produce contextually appropriate prose.
pub fn build_llm_prompt(event: &GameEvent, ctx: &WorldContext, importance: Importance) -> String {
    let length_instruction = match importance {
        Importance::Epic => "Write a dramatic 3-4 sentence paragraph.",
        Importance::Notable => "Write 1-2 vivid sentences.",
        Importance::Minor => "Write one brief sentence.",
    };

    let event_description = render_summary(event, ctx);

    format!(
        "You are a medieval chronicler recording the history of a feudal realm. \
         Write in the style of a medieval chronicle — formal, dramatic, evocative. \
         Use archaic-flavored English but keep it readable.\n\n\
         Event: {}\n\n\
         {}  Do not use modern language. Do not break character. \
         Do not add information not implied by the event.",
        event_description, length_instruction
    )
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crown_ash_types::army::BattleResult;

    fn ctx() -> WorldContext {
        WorldContext {
            province_names: vec![
                (0, "Frosthold".into()),
                (7, "Ashenmere".into()),
                (14, "Saltmere".into()),
            ],
            faction_names: vec![
                (0, "Ashen Crown".into()),
                (1, "Vale Princes".into()),
                (3, "Salt League".into()),
            ],
            character_names: vec![
                (1, "King Aldric".into()),
                (2, "Queen Isolde".into()),
                (5, "Duke Varen".into()),
            ],
            faction_cultures: vec![],
            army_factions: vec![
                (100, 0), // army 100 → Ashen Crown
                (200, 1), // army 200 → Vale Princes
            ],
            current_turn: 42,
        }
    }

    #[test]
    fn battle_prose_includes_province_and_factions() {
        let c = ctx();
        let event = GameEvent::Battle(BattleResult {
            attacker_army: 100,
            defender_army: Some(200),
            province: 7,
            attacker_casualties: 150,
            defender_casualties: 200,
            attacker_won: true,
            random_factor: crown_ash_types::fixed_point::FixedPoint::from_int(1000),
            turn: 42,
        });
        let prose = render(&event, &c);
        assert!(prose.contains("Ashenmere"), "prose={}", prose);
        assert!(prose.contains("Ashen Crown"), "prose={}", prose);
        assert!(prose.contains("Vale Princes"), "prose={}", prose);
    }

    #[test]
    fn plague_prose_severity_word() {
        let c = ctx();
        let event = GameEvent::PlagueOutbreak {
            province: 7,
            severity: 700,
            population_lost: 500,
            turn: 42,
        };
        let prose = render(&event, &c);
        assert!(prose.contains("devastating"), "prose={}", prose);
    }

    #[test]
    fn character_death_by_assassination() {
        let c = ctx();
        let event = GameEvent::CharacterDied {
            character_id: 1,
            character_name: "King Aldric".into(),
            cause: DeathCause::Assassination,
            turn: 42,
        };
        let prose = render(&event, &c);
        assert!(prose.contains("dagger"), "prose={}", prose);
        assert!(prose.contains("King Aldric"), "prose={}", prose);
    }

    #[test]
    fn faction_eliminated_prose() {
        let c = ctx();
        let event = GameEvent::FactionEliminated { faction: 1, turn: 42 };
        let prose = render(&event, &c);
        assert!(prose.contains("Vale Princes"), "prose={}", prose);
        assert!(prose.contains("no more") || prose.contains("end has come"), "prose={}", prose);
    }

    #[test]
    fn summary_is_shorter_than_prose() {
        let c = ctx();
        let event = GameEvent::WarDeclared {
            attacker: 0,
            defender: 1,
            casus_belli: "Conquest".into(),
            turn: 42,
        };
        let prose = render(&event, &c);
        let summary = render_summary(&event, &c);
        assert!(summary.len() < prose.len(), "summary should be shorter");
    }

    #[test]
    fn llm_prompt_contains_chronicler_instruction() {
        let c = ctx();
        let event = GameEvent::FactionEliminated { faction: 1, turn: 42 };
        let prompt = build_llm_prompt(&event, &c, Importance::Epic);
        assert!(prompt.contains("medieval chronicler"), "prompt={}", prompt);
        assert!(prompt.contains("3-4 sentence"), "prompt={}", prompt);
    }

    #[test]
    fn all_31_events_render_without_panic() {
        let c = ctx();
        let events: Vec<GameEvent> = vec![
            GameEvent::Battle(BattleResult {
                attacker_army: 100, defender_army: Some(200), province: 7,
                attacker_casualties: 100, defender_casualties: 80, attacker_won: true,
                random_factor: crown_ash_types::fixed_point::FixedPoint::from_int(1000), turn: 1,
            }),
            GameEvent::ProvinceConquered { province: 7, old_controller: 1, new_controller: 0, turn: 1 },
            GameEvent::WarDeclared { attacker: 0, defender: 1, casus_belli: "Conquest".into(), turn: 2 },
            GameEvent::TreatySigned { faction_a: 0, faction_b: 1, treaty_type: "White Peace".into(), turn: 3 },
            GameEvent::CharacterDied { character_id: 1, character_name: "King".into(), cause: DeathCause::OldAge, turn: 4 },
            GameEvent::CharacterBorn { character_id: 10, character_name: "Baby".into(), parent: 1, dynasty: 0, turn: 5 },
            GameEvent::SuccessionCrisis { faction: 0, dead_ruler: 1, claimants: vec![2,3], realm_split: false, turn: 6 },
            GameEvent::PlagueOutbreak { province: 0, severity: 500, population_lost: 200, turn: 7 },
            GameEvent::Famine { province: 0, severity: 300, turn: 8 },
            GameEvent::Harvest { province: 7, prosperity_gain: 20000, turn: 9 },
            GameEvent::Rebellion { province: 7, rebels: 150, turn: 10 },
            GameEvent::PlayerJoined { wallet: "0xABC123DEF456".into(), faction: 0, turn: 11 },
            GameEvent::ConstructionComplete { province: 7, improvement: "Market".into(), turn: 12 },
            GameEvent::FactionEliminated { faction: 1, turn: 13 },
            GameEvent::RealmSplit { original_faction: 0, new_faction: 7, rebel_leader: 5, provinces_lost: 3, turn: 14 },
            GameEvent::PlotLaunched { instigator: 1, target: 2, plot_type: "Assassination".into(), turn: 15 },
            GameEvent::PlotSucceeded { instigator_name: "Duke".into(), target_name: "King".into(), plot_type: "Assassination".into(), turn: 16 },
            GameEvent::PlotDiscovered { instigator_name: "Duke".into(), target_name: "King".into(), discovered_by: "Spy".into(), turn: 17 },
            GameEvent::PlotFoiled { instigator_name: "Duke".into(), target_name: "King".into(), turn: 18 },
            GameEvent::TradeRouteEstablished { from: 0, to: 7, goods: "Grain".into(), turn: 19 },
            GameEvent::TradeRouteDisrupted { from: 0, to: 7, reason: "War".into(), turn: 20 },
            GameEvent::CharacterTombstoned { character_id: 1, character_name: "Old King".into(), turn: 21 },
            GameEvent::ArmyAutoDisbanded { army_id: 1, faction: 0, troops_returned: 100, province: 7, turn: 22 },
            GameEvent::ReligiousConversion { province: 7, old_religion: "Old Gods".into(), new_religion: "Faith of Light".into(), turn: 23 },
            GameEvent::Heresy { faction: 0, province: 7, severity: 400, turn: 24 },
            GameEvent::Miracle { province: 7, prosperity_gain: 30000, turn: 25 },
            GameEvent::SiegeStarted { province: 7, attacker_faction: 0, defender_faction: 1, turns_required: 9, turn: 26 },
            GameEvent::SiegeCompleted { province: 7, old_controller: 1, new_controller: 0, turns_lasted: 9, attacker_casualties: 45, turn: 27 },
            GameEvent::Friendship { character_a: 1, character_b: 2, turn: 28 },
            GameEvent::Rivalry { character_a: 1, character_b: 5, turn: 29 },
            GameEvent::MarriageAlliance { character_a: 1, character_b: 2, faction_a: 0, faction_b: 1, turn: 30 },
        ];

        for event in &events {
            let prose = render(event, &c);
            assert!(!prose.is_empty(), "Empty prose for {:?}", event);
            let summary = render_summary(event, &c);
            assert!(!summary.is_empty(), "Empty summary for {:?}", event);
        }
    }
}
