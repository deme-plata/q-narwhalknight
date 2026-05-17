//! Axum handler functions for Crown & Ash game endpoints.
//!
//! All handlers receive the shared game state via [`State<SharedGameState>`]
//! and return JSON responses.  Read operations acquire a read lock;
//! write operations (`submit_action`, `join_game`) acquire a write lock.

use axum::{
    extract::{Json, Path, State},
    http::StatusCode,
    response::IntoResponse,
};
use serde::{Deserialize, Serialize};
use tracing::{info, warn};

use crown_ash_types::{
    Army, Character, DiplomaticRelation, Faction, FactionId, GameAction, GameEvent,
    Province, ProvinceId, QueuedAction, Realm, WorldMeta,
    MAX_FACTIONS,
};

use crate::SharedGameState;

// ─── Response Envelope ─────────────────────────────────────────────────────────

/// Standard API response wrapper consistent with the rest of the Q-NarwhalKnight API.
#[derive(Debug, Serialize)]
pub struct ApiResponse<T: Serialize> {
    pub success: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<T>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

impl<T: Serialize> ApiResponse<T> {
    pub fn ok(data: T) -> Json<Self> {
        Json(Self {
            success: true,
            data: Some(data),
            error: None,
        })
    }

    pub fn err(msg: impl Into<String>) -> Json<Self> {
        Json(Self {
            success: false,
            data: None,
            error: Some(msg.into()),
        })
    }
}

// ─── Response DTOs ─────────────────────────────────────────────────────────────

/// Full world snapshot returned by `GET /world`.
#[derive(Debug, Serialize)]
pub struct WorldSnapshot {
    pub meta: WorldMeta,
    pub provinces: Vec<Province>,
    pub factions: Vec<Faction>,
    pub realms: Vec<Realm>,
    pub armies: Vec<Army>,
    pub characters: Vec<Character>,
    pub diplomacy: Vec<DiplomaticRelation>,
    pub action_queue_size: usize,
}

/// Faction details returned by `GET /faction/:id`.
#[derive(Debug, Serialize)]
pub struct FactionDetail {
    pub faction: Faction,
    pub characters: Vec<Character>,
    pub provinces: Vec<Province>,
    pub armies: Vec<Army>,
    pub realm: Option<Realm>,
    pub relations: Vec<DiplomaticRelation>,
}

/// Province history returned by `GET /history/:province_id`.
#[derive(Debug, Serialize)]
pub struct ProvinceHistory {
    pub province_id: ProvinceId,
    pub events: Vec<GameEvent>,
}

// ─── Request DTOs ──────────────────────────────────────────────────────────────

/// Body for `POST /action`.
#[derive(Debug, Deserialize)]
pub struct SubmitActionRequest {
    /// Wallet address of the player submitting the action.
    pub wallet: String,
    /// The game action to queue.
    pub action: GameAction,
}

/// Body for `POST /join`.
#[derive(Debug, Deserialize)]
pub struct JoinGameRequest {
    /// Wallet address of the joining player.
    pub wallet: String,
    /// Faction ID the player wants to control (0-6).
    pub faction: FactionId,
}

/// Confirmation returned after a successful join.
#[derive(Debug, Serialize)]
pub struct JoinGameResponse {
    pub wallet: String,
    pub faction: FactionId,
    pub faction_name: String,
    pub turn: u32,
}

/// Confirmation returned after a successful action submission.
#[derive(Debug, Serialize)]
pub struct ActionQueuedResponse {
    pub queued: bool,
    pub queue_position: usize,
    pub turn: u32,
}

// ─── Handlers ──────────────────────────────────────────────────────────────────

/// `GET /world` — Return the full world snapshot.
pub async fn get_world(
    State(state): State<SharedGameState>,
) -> Result<impl IntoResponse, StatusCode> {
    let gs = state.read().await;
    let world = &gs.world;

    let snapshot = WorldSnapshot {
        meta: world.meta.clone(),
        provinces: world.provinces.clone(),
        factions: world.factions.clone(),
        realms: world.realms.clone(),
        armies: world.armies.clone(),
        characters: world.characters.clone(),
        diplomacy: world.diplomacy.clone(),
        action_queue_size: gs.action_queue.len(),
    };

    Ok(ApiResponse::ok(snapshot))
}

/// `GET /province/:id` — Return a single province by numeric ID.
pub async fn get_province(
    State(state): State<SharedGameState>,
    Path(id): Path<u16>,
) -> Result<impl IntoResponse, StatusCode> {
    let gs = state.read().await;

    match gs.world.provinces.iter().find(|p| p.id == id) {
        Some(p) => Ok(ApiResponse::ok(p.clone())),
        None => {
            warn!(province_id = id, "Province not found");
            Err(StatusCode::NOT_FOUND)
        }
    }
}

/// `GET /faction/:id` — Return faction details including its characters,
/// provinces, armies, realm, and diplomatic relations.
pub async fn get_faction(
    State(state): State<SharedGameState>,
    Path(id): Path<u8>,
) -> Result<impl IntoResponse, StatusCode> {
    let gs = state.read().await;
    let world = &gs.world;

    let faction = match world.factions.iter().find(|f| f.id == id) {
        Some(f) => f.clone(),
        None => {
            warn!(faction_id = id, "Faction not found");
            return Err(StatusCode::NOT_FOUND);
        }
    };

    let characters: Vec<Character> = world
        .characters
        .iter()
        .filter(|c| c.faction == id && c.alive)
        .cloned()
        .collect();

    let provinces: Vec<Province> = world
        .provinces
        .iter()
        .filter(|p| p.controller == id)
        .cloned()
        .collect();

    let armies: Vec<Army> = world
        .armies
        .iter()
        .filter(|a| a.owner_faction == id)
        .cloned()
        .collect();

    let realm = world.realms.iter().find(|r| r.faction == id).cloned();

    let relations: Vec<DiplomaticRelation> = world
        .diplomacy
        .iter()
        .filter(|r| r.faction_a == id || r.faction_b == id)
        .cloned()
        .collect();

    let detail = FactionDetail {
        faction,
        characters,
        provinces,
        armies,
        realm,
        relations,
    };

    Ok(ApiResponse::ok(detail))
}

/// `GET /realm/:wallet` — Return the player realm associated with a wallet address.
pub async fn get_realm(
    State(state): State<SharedGameState>,
    Path(wallet): Path<String>,
) -> Result<impl IntoResponse, StatusCode> {
    let gs = state.read().await;

    match gs.world.realms.iter().find(|r| r.owner_wallet == wallet) {
        Some(r) => Ok(ApiResponse::ok(r.clone())),
        None => {
            warn!(wallet = %wallet, "Realm not found for wallet");
            Err(StatusCode::NOT_FOUND)
        }
    }
}

/// `GET /turn/:number` — Return the turn summary for a specific turn number.
pub async fn get_turn(
    State(state): State<SharedGameState>,
    Path(number): Path<u32>,
) -> Result<impl IntoResponse, StatusCode> {
    let gs = state.read().await;

    match gs.turn_history.iter().find(|t| t.turn == number) {
        Some(summary) => Ok(ApiResponse::ok(summary.clone())),
        None => {
            warn!(turn = number, "Turn summary not found");
            Err(StatusCode::NOT_FOUND)
        }
    }
}

/// `GET /history/:province_id` — Return all events that affected a given province.
///
/// Scans the complete turn history and collects events referencing the province.
pub async fn get_province_history(
    State(state): State<SharedGameState>,
    Path(province_id): Path<u16>,
) -> Result<impl IntoResponse, StatusCode> {
    let gs = state.read().await;

    // Verify the province exists.
    if !gs.world.provinces.iter().any(|p| p.id == province_id) {
        warn!(province_id, "Province not found for history lookup");
        return Err(StatusCode::NOT_FOUND);
    }

    let mut events: Vec<GameEvent> = Vec::new();

    for summary in &gs.turn_history {
        for event in &summary.events {
            if event_references_province(event, province_id) {
                events.push(event.clone());
            }
        }
    }

    let history = ProvinceHistory {
        province_id,
        events,
    };

    Ok(ApiResponse::ok(history))
}

/// `POST /action` — Queue a game action for the current turn.
///
/// The action is validated for basic correctness (wallet must own a realm)
/// and then appended to the action queue.  The simulation tick will process
/// it when the next turn resolves.
pub async fn submit_action(
    State(state): State<SharedGameState>,
    Json(body): Json<SubmitActionRequest>,
) -> Result<impl IntoResponse, StatusCode> {
    let mut gs = state.write().await;

    // Validate: the wallet must own a realm (check both realm.owner_wallet and faction.player_wallet).
    let has_realm = gs.world.realms.iter().any(|r| r.owner_wallet == body.wallet);
    let has_faction = gs.world.factions.iter().any(|f| f.player_wallet.as_deref() == Some(&body.wallet));

    // If wallet owns a faction but realm.owner_wallet is empty, fix it now.
    if !has_realm && has_faction {
        if let Some(faction) = gs.world.factions.iter().find(|f| f.player_wallet.as_deref() == Some(&body.wallet)) {
            let fid = faction.id;
            if let Some(realm) = gs.world.realms.iter_mut().find(|r| r.faction == fid) {
                info!(wallet = %body.wallet, faction = fid, "Auto-fixing realm owner_wallet");
                realm.owner_wallet = body.wallet.clone();
            }
        }
    } else if !has_realm && !has_faction {
        warn!(wallet = %body.wallet, "Action rejected: wallet has no realm");
        return Ok((
            StatusCode::BAD_REQUEST,
            ApiResponse::<ActionQueuedResponse>::err("Wallet does not own a realm"),
        ));
    }

    let current_turn = gs.world.meta.turn;

    let queued = QueuedAction {
        wallet: body.wallet.clone(),
        action: body.action,
        submitted_turn: current_turn,
    };

    gs.action_queue.push(queued);
    let queue_position = gs.action_queue.len();

    info!(
        wallet = %body.wallet,
        queue_position,
        turn = current_turn,
        "Game action queued"
    );

    Ok((
        StatusCode::OK,
        ApiResponse::ok(ActionQueuedResponse {
            queued: true,
            queue_position,
            turn: current_turn,
        }),
    ))
}

/// `POST /join` — Join the game by claiming an unoccupied faction.
///
/// Validates that:
/// - The world is initialized
/// - The faction ID is valid (0 .. MAX_FACTIONS)
/// - The faction is not already claimed by another player
/// - The wallet does not already own a different realm
pub async fn join_game(
    State(state): State<SharedGameState>,
    Json(body): Json<JoinGameRequest>,
) -> Result<impl IntoResponse, StatusCode> {
    let mut gs = state.write().await;

    // World must be initialized.
    if !gs.world.meta.initialized {
        warn!("Join rejected: world not yet initialized");
        return Ok((
            StatusCode::BAD_REQUEST,
            ApiResponse::<JoinGameResponse>::err("World has not been initialized yet"),
        ));
    }

    // Faction ID in bounds.
    if body.faction >= MAX_FACTIONS {
        warn!(faction = body.faction, "Join rejected: invalid faction ID");
        return Ok((
            StatusCode::BAD_REQUEST,
            ApiResponse::<JoinGameResponse>::err(format!(
                "Invalid faction ID {}. Must be 0-{}",
                body.faction,
                MAX_FACTIONS - 1
            )),
        ));
    }

    // Check the wallet doesn't already own a realm.
    let already_has_realm = gs.world.realms.iter().any(|r| r.owner_wallet == body.wallet);
    if already_has_realm {
        warn!(wallet = %body.wallet, "Join rejected: wallet already has a realm");
        return Ok((
            StatusCode::BAD_REQUEST,
            ApiResponse::<JoinGameResponse>::err("Wallet already controls a faction"),
        ));
    }

    // The faction must exist and not already have a player wallet.
    let faction = match gs.world.factions.iter_mut().find(|f| f.id == body.faction) {
        Some(f) => f,
        None => {
            return Ok((
                StatusCode::BAD_REQUEST,
                ApiResponse::<JoinGameResponse>::err("Faction does not exist in the world"),
            ));
        }
    };

    if faction.player_wallet.is_some() {
        warn!(
            faction = body.faction,
            "Join rejected: faction already claimed"
        );
        return Ok((
            StatusCode::BAD_REQUEST,
            ApiResponse::<JoinGameResponse>::err(format!(
                "Faction '{}' is already claimed by another player",
                faction.name
            )),
        ));
    }

    faction.player_wallet = Some(body.wallet.clone());
    let faction_name = faction.name.clone();
    let faction_id = body.faction;

    gs.world.meta.player_count += 1;
    let current_turn = gs.world.meta.turn;

    // Create a Realm entry so submit_action can find the wallet.
    // Only create if a realm doesn't already exist for this faction.
    if !gs.world.realms.iter().any(|r| r.faction == faction_id) {
        use crown_ash_types::{Realm, RealmCohesion, FixedPoint};

        let provinces: Vec<ProvinceId> = gs.world.provinces.iter()
            .filter(|p| p.controller == faction_id)
            .map(|p| p.id)
            .collect();

        let ruler = gs.world.characters.iter()
            .find(|c| c.faction == faction_id && c.alive && c.role == crown_ash_types::CharacterRole::Ruler)
            .map(|c| c.id)
            .unwrap_or(0);

        let treasury = provinces.iter()
            .filter_map(|pid| gs.world.provinces.iter().find(|p| p.id == *pid))
            .map(|p| p.resources.gold)
            .fold(FixedPoint::ZERO, |a, b| a + b);

        let at_war_with: Vec<u8> = gs.world.diplomacy.iter()
            .filter(|d| d.at_war && (d.faction_a == faction_id || d.faction_b == faction_id))
            .map(|d| if d.faction_a == faction_id { d.faction_b } else { d.faction_a })
            .collect();

        let allies: Vec<u8> = gs.world.diplomacy.iter()
            .filter(|d| !d.at_war && !d.treaties.is_empty()
                && (d.faction_a == faction_id || d.faction_b == faction_id))
            .map(|d| if d.faction_a == faction_id { d.faction_b } else { d.faction_a })
            .collect();

        gs.world.realms.push(Realm {
            owner_wallet: body.wallet.clone(),
            faction: faction_id,
            ruler,
            provinces,
            vassals: Vec::new(),
            treasury,
            cohesion: RealmCohesion::default(),
            age: 0,
            at_war_with,
            allies,
            religious_authority: FixedPoint::from_int(500),
        });

        info!(faction = faction_id, "Created Realm for player");
    } else {
        // Realm already exists (e.g., from simulation), update its owner_wallet.
        if let Some(realm) = gs.world.realms.iter_mut().find(|r| r.faction == faction_id) {
            realm.owner_wallet = body.wallet.clone();
        }
    }

    info!(
        wallet = %body.wallet,
        faction = body.faction,
        faction_name = %faction_name,
        turn = current_turn,
        "Player joined Crown & Ash"
    );

    Ok((
        StatusCode::OK,
        ApiResponse::ok(JoinGameResponse {
            wallet: body.wallet,
            faction: body.faction,
            faction_name,
            turn: current_turn,
        }),
    ))
}

// ─── Helpers ───────────────────────────────────────────────────────────────────

/// Returns `true` if a [`GameEvent`] references the given province.
fn event_references_province(event: &GameEvent, province_id: ProvinceId) -> bool {
    match event {
        GameEvent::Battle(result) => result.province == province_id,
        GameEvent::ProvinceConquered { province, .. } => *province == province_id,
        GameEvent::PlagueOutbreak { province, .. } => *province == province_id,
        GameEvent::Famine { province, .. } => *province == province_id,
        GameEvent::Harvest { province, .. } => *province == province_id,
        GameEvent::Rebellion { province, .. } => *province == province_id,
        GameEvent::ConstructionComplete { province, .. } => *province == province_id,
        // Trade route events reference two provinces.
        GameEvent::TradeRouteEstablished { from, to, .. }
        | GameEvent::TradeRouteDisrupted { from, to, .. } => {
            *from == province_id || *to == province_id
        }
        // Army disbanded in a specific province.
        GameEvent::ArmyAutoDisbanded { province, .. } => *province == province_id,
        // These events are not province-specific.
        GameEvent::WarDeclared { .. }
        | GameEvent::TreatySigned { .. }
        | GameEvent::CharacterDied { .. }
        | GameEvent::CharacterBorn { .. }
        | GameEvent::SuccessionCrisis { .. }
        | GameEvent::PlayerJoined { .. }
        | GameEvent::FactionEliminated { .. }
        | GameEvent::RealmSplit { .. }
        | GameEvent::PlotLaunched { .. }
        | GameEvent::PlotSucceeded { .. }
        | GameEvent::PlotDiscovered { .. }
        | GameEvent::PlotFoiled { .. }
        | GameEvent::CharacterTombstoned { .. }
        | GameEvent::Friendship { .. }
        | GameEvent::Rivalry { .. }
        | GameEvent::MarriageAlliance { .. } => false,
        // Province-specific religion and siege events.
        GameEvent::ReligiousConversion { province, .. }
        | GameEvent::Heresy { province, .. }
        | GameEvent::Miracle { province, .. }
        | GameEvent::SiegeStarted { province, .. }
        | GameEvent::SiegeCompleted { province, .. } => *province == province_id,
    }
}
