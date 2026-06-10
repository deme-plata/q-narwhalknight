//! Crown & Ash — Shared data types for medieval grand strategy on Q-NarwhalKnight.
//!
//! This crate contains all type definitions used by:
//! - `crown-ash-sim` (on-chain simulation engine)
//! - `crown-ash-plugin` (WASM plugin wrapper)
//! - `crown-ash-api` (REST API handlers)
//! - `crown-ash-client` (Bevy game client)
//!
//! # Design Principles
//! - **No floating point** — all numeric game state uses `FixedPoint` (i64 × 1000)
//! - **Deterministic** — identical state transitions on all nodes
//! - **Serializable** — all types derive `Serialize`/`Deserialize` for bincode + JSON

pub mod fixed_point;
pub mod province;
pub mod character;
pub mod faction;
pub mod realm;
pub mod army;
pub mod dynasty;
pub mod action;
pub mod event;
pub mod diplomacy;
pub mod economy;
pub mod world;

// Re-export key types at crate root for convenience.
pub use fixed_point::FixedPoint;
pub use province::{Province, ProvinceId, Terrain, Religion, Culture, Resources, Troops, Improvement};
pub use character::{Character, CharacterId, CharacterRole, CharacterStats, CharacterTombstone, Plot, PlotType, Trait, PersonalRelation, RelationType, OpinionModifier};
pub use faction::{Faction, FactionId, FactionTemplate, FactionBonuses};
pub use realm::{Realm, RealmCohesion};
pub use army::{Army, ArmyId, BattleResult, SiegeProgress};
pub use dynasty::{Dynasty, DynastyId, SuccessionRule};
pub use action::{GameAction, QueuedAction, CasusBelli, TreatyType, CouncilRole};
pub use event::{GameEvent, TurnSummary, DeathCause};
pub use diplomacy::DiplomaticRelation;
pub use economy::{TradeRoute, TradeGood};
pub use world::{
    WorldMeta, WorldConfig, BLOCKS_PER_TURN, MAX_FACTIONS, MAX_PROVINCES, SIM_VERSION,
    MAX_BATTLES_PER_TURN, MAX_EVENTS_PER_TURN, MAX_SUCCESSIONS_PER_TURN,
    MAX_BIRTHS_PER_TURN, MAX_DEATHS_PER_TURN,
};
