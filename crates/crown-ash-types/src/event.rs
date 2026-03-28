//! GameEvent — narrative events emitted during simulation ticks.

use serde::{Deserialize, Serialize};
use crate::army::BattleResult;
use crate::province::ProvinceId;

/// Events generated during tick processing (emitted via SSE to clients).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GameEvent {
    /// Battle resolved between armies.
    Battle(BattleResult),
    /// Province changed controller.
    ProvinceConquered {
        province: ProvinceId,
        old_controller: u8,
        new_controller: u8,
        turn: u32,
    },
    /// War declared between factions.
    WarDeclared {
        attacker: u8,
        defender: u8,
        casus_belli: String,
        turn: u32,
    },
    /// Treaty signed.
    TreatySigned {
        faction_a: u8,
        faction_b: u8,
        treaty_type: String,
        turn: u32,
    },
    /// Character died.
    CharacterDied {
        character_id: u32,
        character_name: String,
        cause: DeathCause,
        turn: u32,
    },
    /// Character born.
    CharacterBorn {
        character_id: u32,
        character_name: String,
        parent: u32,
        dynasty: u16,
        turn: u32,
    },
    /// Succession crisis triggered.
    SuccessionCrisis {
        faction: u8,
        dead_ruler: u32,
        claimants: Vec<u32>,
        realm_split: bool,
        turn: u32,
    },
    /// Plague outbreak in province.
    PlagueOutbreak {
        province: ProvinceId,
        severity: i64,
        population_lost: u32,
        turn: u32,
    },
    /// Famine in province.
    Famine {
        province: ProvinceId,
        severity: i64,
        turn: u32,
    },
    /// Bountiful harvest.
    Harvest {
        province: ProvinceId,
        prosperity_gain: i64,
        turn: u32,
    },
    /// Peasant rebellion.
    Rebellion {
        province: ProvinceId,
        rebels: u32,
        turn: u32,
    },
    /// Player joined the game.
    PlayerJoined {
        wallet: String,
        faction: u8,
        turn: u32,
    },
    /// Construction completed in province.
    ConstructionComplete {
        province: ProvinceId,
        improvement: String,
        turn: u32,
    },
    /// Faction eliminated (all provinces lost).
    FactionEliminated {
        faction: u8,
        turn: u32,
    },
    /// Realm split during succession crisis — breakaway faction formed.
    RealmSplit {
        original_faction: u8,
        new_faction: u8,
        rebel_leader: u32,
        provinces_lost: u32,
        turn: u32,
    },
    /// An intrigue plot was launched.
    PlotLaunched {
        instigator: u32,
        target: u32,
        plot_type: String,
        turn: u32,
    },
    /// An intrigue plot succeeded and its effects were applied.
    PlotSucceeded {
        instigator_name: String,
        target_name: String,
        plot_type: String,
        turn: u32,
    },
    /// A plot was discovered by counter-intelligence.
    PlotDiscovered {
        instigator_name: String,
        target_name: String,
        discovered_by: String,
        turn: u32,
    },
    /// A plot was foiled (execution attempt failed).
    PlotFoiled {
        instigator_name: String,
        target_name: String,
        turn: u32,
    },
    /// A trade route was established between two provinces.
    TradeRouteEstablished {
        from: ProvinceId,
        to: ProvinceId,
        goods: String,
        turn: u32,
    },
    /// A trade route was disrupted (war, raid, or blockade).
    TradeRouteDisrupted {
        from: ProvinceId,
        to: ProvinceId,
        reason: String,
        turn: u32,
    },
    /// A dead character was converted to a tombstone and pruned from the world.
    CharacterTombstoned {
        character_id: u32,
        character_name: String,
        turn: u32,
    },
    /// An army was auto-disbanded because the faction exceeded its army cap.
    ArmyAutoDisbanded {
        army_id: u32,
        faction: u8,
        troops_returned: u32,
        province: ProvinceId,
        turn: u32,
    },
    /// Province completed religious conversion.
    ReligiousConversion {
        province: ProvinceId,
        old_religion: String,
        new_religion: String,
        turn: u32,
    },
    /// Heresy event — religious authority crumbles.
    Heresy {
        faction: u8,
        province: ProvinceId,
        severity: i64,
        turn: u32,
    },
    /// Religious miracle — boosts province prosperity and authority.
    Miracle {
        province: ProvinceId,
        prosperity_gain: i64,
        turn: u32,
    },
    /// An army begins besieging a fortified province.
    SiegeStarted {
        province: ProvinceId,
        attacker_faction: u8,
        defender_faction: u8,
        turns_required: u32,
        turn: u32,
    },
    /// A siege completes — province captured after prolonged assault.
    SiegeCompleted {
        province: ProvinceId,
        old_controller: u8,
        new_controller: u8,
        turns_lasted: u32,
        attacker_casualties: u32,
        turn: u32,
    },
    /// Two characters became friends.
    Friendship {
        character_a: u32,
        character_b: u32,
        turn: u32,
    },
    /// Two characters became rivals.
    Rivalry {
        character_a: u32,
        character_b: u32,
        turn: u32,
    },
    /// A cross-faction marriage alliance was formed.
    MarriageAlliance {
        character_a: u32,
        character_b: u32,
        faction_a: u8,
        faction_b: u8,
        turn: u32,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DeathCause {
    OldAge,
    Battle,
    Disease,
    Assassination,
    Execution,
    Accident,
}

/// Summary of a single game turn.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TurnSummary {
    pub turn: u32,
    pub block_height: u64,
    pub events: Vec<GameEvent>,
    pub active_factions: u8,
    pub total_armies: u32,
    pub total_population: u64,
}
