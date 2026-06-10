// ============================================================================
// game_items_api.rs — CS:GO2-Style Blockchain Collectibles (RWA Game Items)
// ============================================================================
// Provably fair case opening, marketplace, trade-up contracts, collections.
// Pre-seeded with 5 weapon cases and quantum/blockchain-themed items.
// ============================================================================

use axum::{
    extract::{Json, Path, Query, State},
    http::StatusCode,
    routing::{get, post},
    Router,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::sync::Arc;
use tracing::{error, info, warn};

use q_storage::BalanceStorage;

use crate::contracts_api::ApiResponse;
use crate::wallet_auth::AuthenticatedWallet;
use crate::AppState;

// ─── Storage Prefixes ────────────────────────────────────────────────────────

const GAME_ITEM_PREFIX: &str = "game_item_";
const GAME_LISTING_PREFIX: &str = "game_listing_";
const GAME_CASE_PREFIX: &str = "game_case_";
const GAME_COLLECTION_PREFIX: &str = "game_collection_";
const GAME_HISTORY_PREFIX: &str = "game_history_";
const GAME_NONCE_PREFIX: &str = "game_nonce_";

// ─── Enums ───────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ItemGrade {
    ConsumerGrade,
    IndustrialGrade,
    MilSpec,
    Restricted,
    Classified,
    Covert,
    Contraband,
}

impl ItemGrade {
    /// Drop weight (summed = 100.0). Used for weighted random selection.
    pub fn drop_weight(&self) -> f64 {
        match self {
            Self::ConsumerGrade => 79.92,
            Self::IndustrialGrade => 15.98,
            Self::MilSpec => 3.20,
            Self::Restricted => 0.64,
            Self::Classified => 0.128,
            Self::Covert => 0.026,
            Self::Contraband => 0.026,
        }
    }

    pub fn display_name(&self) -> &'static str {
        match self {
            Self::ConsumerGrade => "Consumer Grade",
            Self::IndustrialGrade => "Industrial Grade",
            Self::MilSpec => "Mil-Spec",
            Self::Restricted => "Restricted",
            Self::Classified => "Classified",
            Self::Covert => "Covert",
            Self::Contraband => "Contraband",
        }
    }

    pub fn color_hex(&self) -> &'static str {
        match self {
            Self::ConsumerGrade => "#b0c3d9",   // white/light gray
            Self::IndustrialGrade => "#5e98d9",  // light blue
            Self::MilSpec => "#4b69ff",          // blue
            Self::Restricted => "#8847ff",       // purple
            Self::Classified => "#d32ce6",       // pink
            Self::Covert => "#eb4b4b",           // red
            Self::Contraband => "#e4ae39",       // gold
        }
    }

    pub fn next_grade(&self) -> Option<Self> {
        match self {
            Self::ConsumerGrade => Some(Self::IndustrialGrade),
            Self::IndustrialGrade => Some(Self::MilSpec),
            Self::MilSpec => Some(Self::Restricted),
            Self::Restricted => Some(Self::Classified),
            Self::Classified => Some(Self::Covert),
            Self::Covert => None,
            Self::Contraband => None,
        }
    }

    fn all_grades() -> &'static [ItemGrade] {
        &[
            Self::ConsumerGrade,
            Self::IndustrialGrade,
            Self::MilSpec,
            Self::Restricted,
            Self::Classified,
            Self::Covert,
            Self::Contraband,
        ]
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WearCondition {
    FactoryNew,
    MinimalWear,
    FieldTested,
    WellWorn,
    BattleScared,
}

impl WearCondition {
    pub fn from_float(f: f64) -> Self {
        if f < 0.07 {
            Self::FactoryNew
        } else if f < 0.15 {
            Self::MinimalWear
        } else if f < 0.38 {
            Self::FieldTested
        } else if f < 0.45 {
            Self::WellWorn
        } else {
            Self::BattleScared
        }
    }

    pub fn display_name(&self) -> &'static str {
        match self {
            Self::FactoryNew => "Factory New",
            Self::MinimalWear => "Minimal Wear",
            Self::FieldTested => "Field-Tested",
            Self::WellWorn => "Well-Worn",
            Self::BattleScared => "Battle-Scarred",
        }
    }

    pub fn short_name(&self) -> &'static str {
        match self {
            Self::FactoryNew => "FN",
            Self::MinimalWear => "MW",
            Self::FieldTested => "FT",
            Self::WellWorn => "WW",
            Self::BattleScared => "BS",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ItemType {
    Skin,
    Case,
    Key,
    Sticker,
    Agent,
    Gloves,
    Knife,
    MusicKit,
    Graffiti,
    Patch,
    Pin,
    Charm,
}

impl ItemType {
    pub fn display_name(&self) -> &'static str {
        match self {
            Self::Skin => "Skin",
            Self::Case => "Case",
            Self::Key => "Key",
            Self::Sticker => "Sticker",
            Self::Agent => "Agent",
            Self::Gloves => "Gloves",
            Self::Knife => "Knife",
            Self::MusicKit => "Music Kit",
            Self::Graffiti => "Graffiti",
            Self::Patch => "Patch",
            Self::Pin => "Pin",
            Self::Charm => "Charm",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ListingStatus {
    Active,
    Sold,
    Cancelled,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HistoryEventType {
    CaseOpen,
    TradeUp,
    MarketBuy,
    MarketSell,
    MarketList,
    MarketCancel,
}

// ─── Core Types ──────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GameItem {
    pub id: String,
    pub name: String,
    pub weapon: String,
    pub collection: String,
    pub grade: ItemGrade,
    pub wear: WearCondition,
    pub float_value: f64,
    pub item_type: ItemType,
    pub owner_wallet: String,
    pub stattrak: bool,
    pub trade_count: u32,
    pub created_at: u64,
    pub listed_price: Option<f64>,
    pub case_open_seed: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CaseDropItem {
    pub name: String,
    pub weapon: String,
    pub item_type: ItemType,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CaseTemplate {
    pub id: String,
    pub name: String,
    pub description: String,
    pub image_theme: String,
    pub items_by_grade: HashMap<String, Vec<CaseDropItem>>,
    pub key_price_qug: f64,
    pub has_contraband_pool: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketplaceListing {
    pub listing_id: String,
    pub item_id: String,
    pub item_name: String,
    pub item_weapon: String,
    pub grade: ItemGrade,
    pub wear: WearCondition,
    pub float_value: f64,
    pub stattrak: bool,
    pub seller_wallet: String,
    pub price_qug: f64,
    pub listed_at: u64,
    pub status: ListingStatus,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GameCollection {
    pub id: String,
    pub name: String,
    pub description: String,
    pub required_items: Vec<String>,
    pub reward_description: String,
    pub reward_qug: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HistoryEntry {
    pub id: String,
    pub event_type: HistoryEventType,
    pub wallet: String,
    pub item_id: Option<String>,
    pub item_name: Option<String>,
    pub details: String,
    pub timestamp: u64,
}

// ─── Request / Response Types ────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
pub struct MarketplaceQuery {
    pub grade: Option<String>,
    pub item_type: Option<String>,
    pub wear: Option<String>,
    pub min_price: Option<f64>,
    pub max_price: Option<f64>,
    pub sort: Option<String>,
    pub search: Option<String>,
    pub limit: Option<usize>,
    pub offset: Option<usize>,
}

#[derive(Debug, Deserialize)]
pub struct ListItemRequest {
    pub item_id: String,
    pub price_qug: f64,
}

#[derive(Debug, Deserialize)]
pub struct BuyItemRequest {
    pub listing_id: String,
}

#[derive(Debug, Deserialize)]
pub struct OpenCaseRequest {
    pub case_id: String,
}

#[derive(Debug, Deserialize)]
pub struct TradeUpRequest {
    pub item_ids: Vec<String>,
}

#[derive(Debug, Deserialize)]
pub struct InventoryQuery {
    pub item_type: Option<String>,
    pub grade: Option<String>,
    pub sort: Option<String>,
}

// ─── Provably Fair Algorithm ─────────────────────────────────────────────────

fn sha256_hex(data: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(data);
    hex::encode(hasher.finalize())
}

fn hash_to_f64(hash: &str) -> f64 {
    // Take first 8 hex chars → u32 → f64 in [0, 1)
    let val = u32::from_str_radix(&hash[..8], 16).unwrap_or(0);
    val as f64 / u32::MAX as f64
}

/// Provably fair case opening. Anyone can verify by re-running with same inputs.
pub fn provably_fair_open(
    block_hash: &str,
    wallet_address: &str,
    nonce: u64,
    case_template: &CaseTemplate,
) -> (ItemGrade, usize, f64, bool, String) {
    let seed_input = format!("{}||{}||{}||QNK_CASE_OPEN", block_hash, wallet_address, nonce);
    let seed = sha256_hex(seed_input.as_bytes());

    // Grade selection
    let grade_hash = sha256_hex(format!("{}||GRADE", seed).as_bytes());
    let grade_roll = hash_to_f64(&grade_hash) * 100.0;

    let grade = select_grade(grade_roll, case_template.has_contraband_pool);

    // Item selection within grade
    let grade_key = serde_json::to_string(&grade).unwrap_or_default();
    let grade_key_clean = grade_key.trim_matches('"');
    let items_in_grade = case_template
        .items_by_grade
        .get(grade_key_clean)
        .cloned()
        .unwrap_or_default();

    let item_hash = sha256_hex(format!("{}||ITEM", seed).as_bytes());
    let item_idx = if items_in_grade.is_empty() {
        0
    } else {
        let val = u32::from_str_radix(&item_hash[..8], 16).unwrap_or(0);
        (val as usize) % items_in_grade.len()
    };

    // Float value [0.0, 1.0)
    let float_hash = sha256_hex(format!("{}||FLOAT", seed).as_bytes());
    let float_value = hash_to_f64(&float_hash);

    // StatTrak: 10% chance
    let stattrak_hash = sha256_hex(format!("{}||STATTRAK", seed).as_bytes());
    let stattrak = hash_to_f64(&stattrak_hash) < 0.10;

    (grade, item_idx, float_value, stattrak, seed)
}

fn select_grade(roll: f64, has_contraband: bool) -> ItemGrade {
    let mut cumulative = 0.0;
    for grade in ItemGrade::all_grades() {
        if *grade == ItemGrade::Contraband && !has_contraband {
            continue;
        }
        cumulative += grade.drop_weight();
        if roll < cumulative {
            return *grade;
        }
    }
    ItemGrade::ConsumerGrade
}

/// Deterministic trade-up: 10 items of same grade → 1 item of next grade
pub fn trade_up_result(item_ids: &[String]) -> String {
    let mut sorted = item_ids.to_vec();
    sorted.sort();
    let combined = sorted.join("||");
    sha256_hex(combined.as_bytes())
}

// ─── Persistence ─────────────────────────────────────────────────────────────

async fn persist_item(state: &AppState, item: &GameItem) {
    let key = format!("{}{}", GAME_ITEM_PREFIX, item.id);
    match serde_json::to_vec(item) {
        Ok(data) => {
            let kv = state.storage_engine.get_kv();
            if let Err(e) = kv.put_sync(q_storage::CF_MANIFEST, key.as_bytes(), &data).await {
                warn!("Failed to persist game item {}: {}", item.id, e);
            }
        }
        Err(e) => warn!("Failed to serialize game item: {}", e),
    }
}

async fn persist_listing(state: &AppState, listing: &MarketplaceListing) {
    let key = format!("{}{}", GAME_LISTING_PREFIX, listing.listing_id);
    match serde_json::to_vec(listing) {
        Ok(data) => {
            let kv = state.storage_engine.get_kv();
            if let Err(e) = kv.put_sync(q_storage::CF_MANIFEST, key.as_bytes(), &data).await {
                warn!("Failed to persist game listing {}: {}", listing.listing_id, e);
            }
        }
        Err(e) => warn!("Failed to serialize game listing: {}", e),
    }
}

async fn persist_case(state: &AppState, case: &CaseTemplate) {
    let key = format!("{}{}", GAME_CASE_PREFIX, case.id);
    match serde_json::to_vec(case) {
        Ok(data) => {
            let kv = state.storage_engine.get_kv();
            if let Err(e) = kv.put_sync(q_storage::CF_MANIFEST, key.as_bytes(), &data).await {
                warn!("Failed to persist game case {}: {}", case.id, e);
            }
        }
        Err(e) => warn!("Failed to serialize game case: {}", e),
    }
}

async fn persist_collection(state: &AppState, coll: &GameCollection) {
    let key = format!("{}{}", GAME_COLLECTION_PREFIX, coll.id);
    match serde_json::to_vec(coll) {
        Ok(data) => {
            let kv = state.storage_engine.get_kv();
            if let Err(e) = kv.put_sync(q_storage::CF_MANIFEST, key.as_bytes(), &data).await {
                warn!("Failed to persist game collection {}: {}", coll.id, e);
            }
        }
        Err(e) => warn!("Failed to serialize game collection: {}", e),
    }
}

async fn persist_history(state: &AppState, entry: &HistoryEntry) {
    let key = format!("{}{}", GAME_HISTORY_PREFIX, entry.id);
    match serde_json::to_vec(entry) {
        Ok(data) => {
            let kv = state.storage_engine.get_kv();
            if let Err(e) = kv.put_sync(q_storage::CF_MANIFEST, key.as_bytes(), &data).await {
                warn!("Failed to persist game history {}: {}", entry.id, e);
            }
        }
        Err(e) => warn!("Failed to serialize game history: {}", e),
    }
}

async fn get_wallet_nonce(state: &AppState, wallet: &str) -> u64 {
    let key = format!("{}{}", GAME_NONCE_PREFIX, wallet);
    let kv = state.storage_engine.get_kv();
    match kv.get(q_storage::CF_MANIFEST, key.as_bytes()).await {
        Ok(Some(data)) => {
            String::from_utf8_lossy(&data).parse::<u64>().unwrap_or(0)
        }
        _ => 0,
    }
}

async fn increment_wallet_nonce(state: &AppState, wallet: &str) -> u64 {
    let current = get_wallet_nonce(state, wallet).await;
    let next = current + 1;
    let key = format!("{}{}", GAME_NONCE_PREFIX, wallet);
    let kv = state.storage_engine.get_kv();
    let _ = kv
        .put_sync(q_storage::CF_MANIFEST, key.as_bytes(), next.to_string().as_bytes())
        .await;
    next
}

// ─── DB Loaders ──────────────────────────────────────────────────────────────

pub async fn load_items_from_db(state: &AppState) -> Vec<GameItem> {
    let mut items = Vec::new();
    let kv = state.storage_engine.get_kv();
    match kv.scan_prefix(q_storage::CF_MANIFEST, GAME_ITEM_PREFIX.as_bytes()).await {
        Ok(entries) => {
            for (_key, value) in entries {
                match serde_json::from_slice::<GameItem>(&value) {
                    Ok(item) => items.push(item),
                    Err(e) => warn!("Failed to deserialize game item: {}", e),
                }
            }
        }
        Err(e) => warn!("Failed to load game items from DB: {}", e),
    }
    items
}

pub async fn load_listings_from_db(state: &AppState) -> Vec<MarketplaceListing> {
    let mut listings = Vec::new();
    let kv = state.storage_engine.get_kv();
    match kv
        .scan_prefix(q_storage::CF_MANIFEST, GAME_LISTING_PREFIX.as_bytes())
        .await
    {
        Ok(entries) => {
            for (_key, value) in entries {
                match serde_json::from_slice::<MarketplaceListing>(&value) {
                    Ok(listing) => listings.push(listing),
                    Err(e) => warn!("Failed to deserialize game listing: {}", e),
                }
            }
        }
        Err(e) => warn!("Failed to load game listings from DB: {}", e),
    }
    listings
}

pub async fn load_cases_from_db(state: &AppState) -> Vec<CaseTemplate> {
    let mut cases = Vec::new();
    let kv = state.storage_engine.get_kv();
    match kv
        .scan_prefix(q_storage::CF_MANIFEST, GAME_CASE_PREFIX.as_bytes())
        .await
    {
        Ok(entries) => {
            for (_key, value) in entries {
                match serde_json::from_slice::<CaseTemplate>(&value) {
                    Ok(case) => cases.push(case),
                    Err(e) => warn!("Failed to deserialize game case: {}", e),
                }
            }
        }
        Err(e) => warn!("Failed to load game cases from DB: {}", e),
    }
    cases
}

pub async fn load_collections_from_db(state: &AppState) -> Vec<GameCollection> {
    let mut collections = Vec::new();
    let kv = state.storage_engine.get_kv();
    match kv
        .scan_prefix(q_storage::CF_MANIFEST, GAME_COLLECTION_PREFIX.as_bytes())
        .await
    {
        Ok(entries) => {
            for (_key, value) in entries {
                match serde_json::from_slice::<GameCollection>(&value) {
                    Ok(coll) => collections.push(coll),
                    Err(e) => warn!("Failed to deserialize game collection: {}", e),
                }
            }
        }
        Err(e) => warn!("Failed to load game collections from DB: {}", e),
    }
    collections
}

// ─── Pre-Seeded Content ──────────────────────────────────────────────────────

fn build_case_items(items: Vec<(&str, &str, ItemGrade, ItemType)>) -> HashMap<String, Vec<CaseDropItem>> {
    let mut map: HashMap<String, Vec<CaseDropItem>> = HashMap::new();
    for (name, weapon, grade, item_type) in items {
        let grade_key = serde_json::to_string(&grade).unwrap_or_default();
        let grade_key_clean = grade_key.trim_matches('"').to_string();
        map.entry(grade_key_clean).or_default().push(CaseDropItem {
            name: name.to_string(),
            weapon: weapon.to_string(),
            item_type,
        });
    }
    map
}

fn create_default_cases() -> Vec<CaseTemplate> {
    vec![
        CaseTemplate {
            id: "quantum_case".into(),
            name: "Quantum Case".into(),
            description: "Contains quantum-themed weapon skins from the Quillon universe.".into(),
            image_theme: "quantum".into(),
            items_by_grade: build_case_items(vec![
                ("Sand Dune", "P250", ItemGrade::ConsumerGrade, ItemType::Skin),
                ("Groundwater", "MAC-10", ItemGrade::ConsumerGrade, ItemType::Skin),
                ("Forest DDPAT", "Galil AR", ItemGrade::ConsumerGrade, ItemType::Skin),
                ("Urban Masked", "MP7", ItemGrade::ConsumerGrade, ItemType::Skin),
                ("Contractor", "P90", ItemGrade::ConsumerGrade, ItemType::Skin),
                ("Hash Storm", "M4A4", ItemGrade::IndustrialGrade, ItemType::Skin),
                ("Blockchain Blaze", "AK-47", ItemGrade::IndustrialGrade, ItemType::Skin),
                ("Node Runner", "UMP-45", ItemGrade::IndustrialGrade, ItemType::Skin),
                ("Quantum Dot", "USP-S", ItemGrade::MilSpec, ItemType::Skin),
                ("Superposition", "AWP", ItemGrade::MilSpec, ItemType::Skin),
                ("Entangled", "Glock-18", ItemGrade::Restricted, ItemType::Skin),
                ("Wave Function", "Desert Eagle", ItemGrade::Restricted, ItemType::Skin),
                ("Quantum Fade", "M4A1-S", ItemGrade::Classified, ItemType::Skin),
                ("Schrödinger", "AK-47", ItemGrade::Covert, ItemType::Skin),
                ("Narwhal Tusk", "Karambit", ItemGrade::Contraband, ItemType::Knife),
            ]),
            key_price_qug: 2.50,
            has_contraband_pool: true,
        },
        CaseTemplate {
            id: "genesis_case".into(),
            name: "Genesis Case".into(),
            description: "The first case ever minted on the Q-NarwhalKnight chain.".into(),
            image_theme: "genesis".into(),
            items_by_grade: build_case_items(vec![
                ("Rust Coat", "Sawed-Off", ItemGrade::ConsumerGrade, ItemType::Skin),
                ("Scorched", "Nova", ItemGrade::ConsumerGrade, ItemType::Skin),
                ("Safari Mesh", "XM1014", ItemGrade::ConsumerGrade, ItemType::Skin),
                ("Boreal Forest", "FAMAS", ItemGrade::ConsumerGrade, ItemType::Skin),
                ("Genesis Dawn", "SG 553", ItemGrade::IndustrialGrade, ItemType::Skin),
                ("Block Zero", "MP9", ItemGrade::IndustrialGrade, ItemType::Skin),
                ("Coinbase", "Five-SeveN", ItemGrade::IndustrialGrade, ItemType::Skin),
                ("Merkle Tree", "AUG", ItemGrade::MilSpec, ItemType::Skin),
                ("Hashrate", "M4A4", ItemGrade::MilSpec, ItemType::Skin),
                ("Proof of Work", "AWP", ItemGrade::Restricted, ItemType::Skin),
                ("Difficulty Spike", "AK-47", ItemGrade::Restricted, ItemType::Skin),
                ("Satoshi", "Desert Eagle", ItemGrade::Classified, ItemType::Skin),
                ("Genesis Block", "M4A1-S", ItemGrade::Covert, ItemType::Skin),
            ]),
            key_price_qug: 2.00,
            has_contraband_pool: false,
        },
        CaseTemplate {
            id: "narwhal_case".into(),
            name: "Narwhal Case".into(),
            description: "Deep-sea themed skins inspired by the Narwhal protocol.".into(),
            image_theme: "narwhal".into(),
            items_by_grade: build_case_items(vec![
                ("Algae", "PP-Bizon", ItemGrade::ConsumerGrade, ItemType::Skin),
                ("Seaweed", "Negev", ItemGrade::ConsumerGrade, ItemType::Skin),
                ("Coral Reef", "MAG-7", ItemGrade::ConsumerGrade, ItemType::Skin),
                ("Barnacle", "MP5-SD", ItemGrade::ConsumerGrade, ItemType::Skin),
                ("Deep Current", "P250", ItemGrade::ConsumerGrade, ItemType::Skin),
                ("Abyssal", "Tec-9", ItemGrade::IndustrialGrade, ItemType::Skin),
                ("Tidal Wave", "FAMAS", ItemGrade::IndustrialGrade, ItemType::Skin),
                ("Kraken Ink", "M4A4", ItemGrade::IndustrialGrade, ItemType::Skin),
                ("Arctic Horn", "USP-S", ItemGrade::MilSpec, ItemType::Skin),
                ("Bioluminescent", "AK-47", ItemGrade::MilSpec, ItemType::Skin),
                ("Leviathan", "AWP", ItemGrade::Restricted, ItemType::Skin),
                ("Tusk Ivory", "Desert Eagle", ItemGrade::Classified, ItemType::Skin),
                ("Narwhal King", "AK-47", ItemGrade::Covert, ItemType::Skin),
                ("Poseidon Trident", "Bayonet", ItemGrade::Contraband, ItemType::Knife),
            ]),
            key_price_qug: 3.00,
            has_contraband_pool: true,
        },
        CaseTemplate {
            id: "covert_case".into(),
            name: "Covert Operations Case".into(),
            description: "Military-grade skins for the most elite operators.".into(),
            image_theme: "covert".into(),
            items_by_grade: build_case_items(vec![
                ("Night Ops", "Galil AR", ItemGrade::ConsumerGrade, ItemType::Skin),
                ("Guerrilla", "MAC-10", ItemGrade::ConsumerGrade, ItemType::Skin),
                ("Sand Storm", "Dual Berettas", ItemGrade::ConsumerGrade, ItemType::Skin),
                ("Jungle Tiger", "CZ75-Auto", ItemGrade::ConsumerGrade, ItemType::Skin),
                ("Recon", "MP7", ItemGrade::IndustrialGrade, ItemType::Skin),
                ("Thermal", "SG 553", ItemGrade::IndustrialGrade, ItemType::Skin),
                ("Cipher", "Glock-18", ItemGrade::IndustrialGrade, ItemType::Skin),
                ("Black Site", "M4A1-S", ItemGrade::MilSpec, ItemType::Skin),
                ("Phantom", "USP-S", ItemGrade::MilSpec, ItemType::Skin),
                ("Stealth", "AWP", ItemGrade::Restricted, ItemType::Skin),
                ("Shadow Protocol", "AK-47", ItemGrade::Classified, ItemType::Skin),
                ("Dark Matter", "Desert Eagle", ItemGrade::Covert, ItemType::Skin),
            ]),
            key_price_qug: 2.50,
            has_contraband_pool: false,
        },
        CaseTemplate {
            id: "founders_case".into(),
            name: "Founders Case".into(),
            description: "Exclusive items celebrating the Q-NarwhalKnight founding team.".into(),
            image_theme: "founders".into(),
            items_by_grade: build_case_items(vec![
                ("Blueprint", "P90", ItemGrade::ConsumerGrade, ItemType::Skin),
                ("Prototype", "MP9", ItemGrade::ConsumerGrade, ItemType::Skin),
                ("Alpha Build", "UMP-45", ItemGrade::ConsumerGrade, ItemType::Skin),
                ("Whitepaper", "Five-SeveN", ItemGrade::ConsumerGrade, ItemType::Skin),
                ("Beta Access", "Tec-9", ItemGrade::ConsumerGrade, ItemType::Skin),
                ("ICO Gold", "AUG", ItemGrade::IndustrialGrade, ItemType::Skin),
                ("Token Sale", "SSG 08", ItemGrade::IndustrialGrade, ItemType::Skin),
                ("Roadmap", "Glock-18", ItemGrade::IndustrialGrade, ItemType::Skin),
                ("DAO Vote", "M4A4", ItemGrade::MilSpec, ItemType::Skin),
                ("Smart Contract", "AK-47", ItemGrade::MilSpec, ItemType::Skin),
                ("Validator", "AWP", ItemGrade::Restricted, ItemType::Skin),
                ("Consensus", "Desert Eagle", ItemGrade::Restricted, ItemType::Skin),
                ("Founder's Mark", "M4A1-S", ItemGrade::Classified, ItemType::Skin),
                ("Quantum Core", "AK-47", ItemGrade::Covert, ItemType::Skin),
                ("Quillon Blade", "M9 Bayonet", ItemGrade::Contraband, ItemType::Knife),
            ]),
            key_price_qug: 5.00,
            has_contraband_pool: true,
        },
    ]
}

fn create_default_collections() -> Vec<GameCollection> {
    vec![
        GameCollection {
            id: "quantum_set".into(),
            name: "Quantum Arsenal".into(),
            description: "Collect all quantum-themed skins to prove your mastery of superposition.".into(),
            required_items: vec![
                "Quantum Dot".into(),
                "Superposition".into(),
                "Entangled".into(),
                "Quantum Fade".into(),
                "Schrödinger".into(),
            ],
            reward_description: "Exclusive 'Quantum Observer' title + 50 QUG bonus".into(),
            reward_qug: 50.0,
        },
        GameCollection {
            id: "genesis_set".into(),
            name: "Genesis Collection".into(),
            description: "Own a piece of blockchain history with the complete Genesis set.".into(),
            required_items: vec![
                "Genesis Dawn".into(),
                "Block Zero".into(),
                "Coinbase".into(),
                "Merkle Tree".into(),
                "Genesis Block".into(),
            ],
            reward_description: "Exclusive 'Block 0 Miner' title + 100 QUG bonus".into(),
            reward_qug: 100.0,
        },
        GameCollection {
            id: "deep_sea_set".into(),
            name: "Deep Sea Collection".into(),
            description: "Dive into the abyss with the complete Narwhal deep-sea set.".into(),
            required_items: vec![
                "Abyssal".into(),
                "Kraken Ink".into(),
                "Leviathan".into(),
                "Tusk Ivory".into(),
                "Narwhal King".into(),
            ],
            reward_description: "Exclusive 'Abyss Walker' title + 75 QUG bonus".into(),
            reward_qug: 75.0,
        },
    ]
}

/// Seed default cases and collections on first startup
pub async fn seed_default_content(state: &AppState) {
    // Check if cases already exist
    let existing_cases = load_cases_from_db(state).await;
    if !existing_cases.is_empty() {
        info!(
            "🎮 Game items: {} cases already seeded, skipping",
            existing_cases.len()
        );
        return;
    }

    info!("🎮 Seeding default game item cases and collections...");

    let cases = create_default_cases();
    for case in &cases {
        persist_case(state, case).await;
    }
    info!("🎮 Seeded {} weapon cases", cases.len());

    let collections = create_default_collections();
    for coll in &collections {
        persist_collection(state, coll).await;
    }
    info!("🎮 Seeded {} collections", collections.len());
}

// ─── Utility ─────────────────────────────────────────────────────────────────

fn current_timestamp() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn generate_id() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let random_part: u32 = (ts as u32).wrapping_mul(2654435761); // Knuth's multiplicative hash
    format!("{:016x}{:08x}", ts, random_part)
}

// ─── API Handlers ────────────────────────────────────────────────────────────

/// GET /api/v1/game-items — Browse marketplace listings
pub async fn list_marketplace(
    State(state): State<Arc<AppState>>,
    Query(params): Query<MarketplaceQuery>,
) -> Result<Json<ApiResponse<serde_json::Value>>, StatusCode> {
    let listings = load_listings_from_db(&state).await;

    let mut active: Vec<&MarketplaceListing> = listings
        .iter()
        .filter(|l| matches!(l.status, ListingStatus::Active))
        .collect();

    // Filters
    if let Some(ref grade_filter) = params.grade {
        active.retain(|l| {
            serde_json::to_string(&l.grade)
                .unwrap_or_default()
                .trim_matches('"')
                == grade_filter
        });
    }
    if let Some(ref item_type_filter) = params.item_type {
        // We need to load the item to check type — filter by name match for now
        let _ = item_type_filter; // Items on listing have grade/wear but not type; skip this filter on listings
    }
    if let Some(ref wear_filter) = params.wear {
        active.retain(|l| {
            serde_json::to_string(&l.wear)
                .unwrap_or_default()
                .trim_matches('"')
                == wear_filter
        });
    }
    if let Some(min) = params.min_price {
        active.retain(|l| l.price_qug >= min);
    }
    if let Some(max) = params.max_price {
        active.retain(|l| l.price_qug <= max);
    }
    if let Some(ref search) = params.search {
        let s = search.to_lowercase();
        active.retain(|l| {
            l.item_name.to_lowercase().contains(&s) || l.item_weapon.to_lowercase().contains(&s)
        });
    }

    // Sort
    match params.sort.as_deref() {
        Some("price_low") => active.sort_by(|a, b| a.price_qug.partial_cmp(&b.price_qug).unwrap_or(std::cmp::Ordering::Equal)),
        Some("price_high") => active.sort_by(|a, b| b.price_qug.partial_cmp(&a.price_qug).unwrap_or(std::cmp::Ordering::Equal)),
        Some("grade") => active.sort_by_key(|l| std::cmp::Reverse(l.grade.drop_weight() as u64)),
        _ => active.sort_by(|a, b| b.listed_at.cmp(&a.listed_at)), // newest first
    }

    let total = active.len();
    let offset = params.offset.unwrap_or(0);
    let limit = params.limit.unwrap_or(50).min(200);
    let page: Vec<_> = active.into_iter().skip(offset).take(limit).collect();

    Ok(Json(ApiResponse::success(serde_json::json!({
        "listings": page,
        "total": total,
        "offset": offset,
        "limit": limit,
    }))))
}

/// GET /api/v1/game-items/stats — Marketplace statistics
pub async fn marketplace_stats(
    State(state): State<Arc<AppState>>,
) -> Result<Json<ApiResponse<serde_json::Value>>, StatusCode> {
    let items = load_items_from_db(&state).await;
    let listings = load_listings_from_db(&state).await;

    let total_items = items.len();
    let active_listings = listings.iter().filter(|l| matches!(l.status, ListingStatus::Active)).count();
    let total_sold = listings.iter().filter(|l| matches!(l.status, ListingStatus::Sold)).count();
    let total_volume: f64 = listings
        .iter()
        .filter(|l| matches!(l.status, ListingStatus::Sold))
        .map(|l| l.price_qug)
        .sum();

    // Floor prices by grade
    let mut floor_prices: HashMap<String, f64> = HashMap::new();
    for listing in listings.iter().filter(|l| matches!(l.status, ListingStatus::Active)) {
        let grade_key = listing.grade.display_name().to_string();
        let entry = floor_prices.entry(grade_key).or_insert(f64::MAX);
        if listing.price_qug < *entry {
            *entry = listing.price_qug;
        }
    }
    // Replace MAX with 0 for grades with no listings
    for val in floor_prices.values_mut() {
        if *val == f64::MAX {
            *val = 0.0;
        }
    }

    // Grade distribution
    let mut grade_distribution: HashMap<String, usize> = HashMap::new();
    for item in &items {
        *grade_distribution
            .entry(item.grade.display_name().to_string())
            .or_insert(0) += 1;
    }

    Ok(Json(ApiResponse::success(serde_json::json!({
        "total_items": total_items,
        "active_listings": active_listings,
        "total_sold": total_sold,
        "total_volume_qug": total_volume,
        "floor_prices": floor_prices,
        "grade_distribution": grade_distribution,
    }))))
}

/// GET /api/v1/game-items/cases — List available case templates
pub async fn list_cases(
    State(state): State<Arc<AppState>>,
) -> Result<Json<ApiResponse<serde_json::Value>>, StatusCode> {
    let cases = load_cases_from_db(&state).await;
    Ok(Json(ApiResponse::success(serde_json::json!({
        "cases": cases,
        "total": cases.len(),
    }))))
}

/// GET /api/v1/game-items/collections — Collection sets
pub async fn list_collections(
    State(state): State<Arc<AppState>>,
    Query(params): Query<HashMap<String, String>>,
) -> Result<Json<ApiResponse<serde_json::Value>>, StatusCode> {
    let collections = load_collections_from_db(&state).await;

    // If wallet provided, compute progress
    let mut collection_progress: Vec<serde_json::Value> = Vec::new();
    let wallet_filter = params.get("wallet").cloned().unwrap_or_default();

    let user_items = if !wallet_filter.is_empty() {
        let items = load_items_from_db(&state).await;
        items
            .into_iter()
            .filter(|i| i.owner_wallet == wallet_filter)
            .map(|i| i.name.clone())
            .collect::<std::collections::HashSet<String>>()
    } else {
        std::collections::HashSet::new()
    };

    for coll in &collections {
        let owned_count = coll
            .required_items
            .iter()
            .filter(|name| user_items.contains(*name))
            .count();
        collection_progress.push(serde_json::json!({
            "collection": coll,
            "owned": owned_count,
            "total": coll.required_items.len(),
            "complete": owned_count == coll.required_items.len(),
        }));
    }

    Ok(Json(ApiResponse::success(serde_json::json!({
        "collections": collection_progress,
    }))))
}

/// GET /api/v1/game-items/:id — Single item details
pub async fn get_item(
    Path(item_id): Path<String>,
    State(state): State<Arc<AppState>>,
) -> Result<Json<ApiResponse<serde_json::Value>>, StatusCode> {
    let key = format!("{}{}", GAME_ITEM_PREFIX, item_id);
    let kv = state.storage_engine.get_kv();
    match kv.get(q_storage::CF_MANIFEST, key.as_bytes()).await {
        Ok(Some(data)) => match serde_json::from_slice::<GameItem>(&data) {
            Ok(item) => Ok(Json(ApiResponse::success(serde_json::json!({
                "item": item,
                "grade_info": {
                    "name": item.grade.display_name(),
                    "color": item.grade.color_hex(),
                    "drop_rate": format!("{:.3}%", item.grade.drop_weight()),
                },
                "wear_info": {
                    "name": item.wear.display_name(),
                    "short": item.wear.short_name(),
                },
            })))),
            Err(e) => {
                error!("Failed to deserialize game item {}: {}", item_id, e);
                Ok(Json(ApiResponse::error("Item corrupted".into())))
            }
        },
        Ok(None) => Ok(Json(ApiResponse::error("Item not found".into()))),
        Err(e) => {
            error!("DB error loading game item {}: {}", item_id, e);
            Ok(Json(ApiResponse::error("Database error".into())))
        }
    }
}

/// GET /api/v1/game-items/inventory/:wallet — User's inventory
pub async fn get_inventory(
    Path(wallet): Path<String>,
    State(state): State<Arc<AppState>>,
    Query(params): Query<InventoryQuery>,
) -> Result<Json<ApiResponse<serde_json::Value>>, StatusCode> {
    let all_items = load_items_from_db(&state).await;
    let mut items: Vec<&GameItem> = all_items
        .iter()
        .filter(|i| i.owner_wallet == wallet)
        .collect();

    if let Some(ref grade_filter) = params.grade {
        items.retain(|i| {
            serde_json::to_string(&i.grade)
                .unwrap_or_default()
                .trim_matches('"')
                == grade_filter
        });
    }
    if let Some(ref type_filter) = params.item_type {
        items.retain(|i| {
            serde_json::to_string(&i.item_type)
                .unwrap_or_default()
                .trim_matches('"')
                == type_filter
        });
    }

    match params.sort.as_deref() {
        Some("grade") => items.sort_by(|a, b| {
            a.grade
                .drop_weight()
                .partial_cmp(&b.grade.drop_weight())
                .unwrap_or(std::cmp::Ordering::Equal)
        }),
        Some("wear") => items.sort_by(|a, b| {
            a.float_value
                .partial_cmp(&b.float_value)
                .unwrap_or(std::cmp::Ordering::Equal)
        }),
        _ => items.sort_by(|a, b| b.created_at.cmp(&a.created_at)),
    }

    Ok(Json(ApiResponse::success(serde_json::json!({
        "items": items,
        "total": items.len(),
        "wallet": wallet,
    }))))
}

/// POST /api/v1/game-items/list — List item for sale
pub async fn list_item_for_sale(
    auth: AuthenticatedWallet,
    State(state): State<Arc<AppState>>,
    Json(request): Json<ListItemRequest>,
) -> Result<Json<ApiResponse<serde_json::Value>>, StatusCode> {
    let wallet_hex = hex::encode(auth.address);

    if request.price_qug <= 0.0 {
        return Ok(Json(ApiResponse::error("Price must be positive".into())));
    }
    if request.price_qug > 1_000_000.0 {
        return Ok(Json(ApiResponse::error("Price exceeds maximum (1M QUG)".into())));
    }

    // Load item, verify ownership
    let item_key = format!("{}{}", GAME_ITEM_PREFIX, request.item_id);
    let kv = state.storage_engine.get_kv();
    let item_data = match kv.get(q_storage::CF_MANIFEST, item_key.as_bytes()).await {
        Ok(Some(d)) => d,
        Ok(None) => return Ok(Json(ApiResponse::error("Item not found".into()))),
        Err(e) => {
            error!("DB error: {}", e);
            return Ok(Json(ApiResponse::error("Database error".into())));
        }
    };

    let mut item: GameItem = match serde_json::from_slice(&item_data) {
        Ok(i) => i,
        Err(e) => {
            error!("Deserialize error: {}", e);
            return Ok(Json(ApiResponse::error("Item corrupted".into())));
        }
    };

    if item.owner_wallet != wallet_hex {
        return Ok(Json(ApiResponse::error("You don't own this item".into())));
    }

    if item.listed_price.is_some() {
        return Ok(Json(ApiResponse::error("Item already listed".into())));
    }

    // Check not already listed
    let existing_listings = load_listings_from_db(&state).await;
    if existing_listings.iter().any(|l| l.item_id == request.item_id && matches!(l.status, ListingStatus::Active)) {
        return Ok(Json(ApiResponse::error("Item already has active listing".into())));
    }

    let listing_id = generate_id();
    let now = current_timestamp();

    let listing = MarketplaceListing {
        listing_id: listing_id.clone(),
        item_id: item.id.clone(),
        item_name: item.name.clone(),
        item_weapon: item.weapon.clone(),
        grade: item.grade,
        wear: item.wear,
        float_value: item.float_value,
        stattrak: item.stattrak,
        seller_wallet: wallet_hex.clone(),
        price_qug: request.price_qug,
        listed_at: now,
        status: ListingStatus::Active,
    };

    item.listed_price = Some(request.price_qug);
    persist_item(&state, &item).await;
    persist_listing(&state, &listing).await;

    let history = HistoryEntry {
        id: generate_id(),
        event_type: HistoryEventType::MarketList,
        wallet: wallet_hex,
        item_id: Some(item.id.clone()),
        item_name: Some(item.name.clone()),
        details: format!("Listed {} for {:.2} QUG", item.name, request.price_qug),
        timestamp: now,
    };
    persist_history(&state, &history).await;

    info!("🎮 Item {} listed for {:.2} QUG by {}", item.name, request.price_qug, history.wallet);

    Ok(Json(ApiResponse::success(serde_json::json!({
        "listing_id": listing_id,
        "item_id": item.id,
        "price_qug": request.price_qug,
    }))))
}

/// POST /api/v1/game-items/buy — Buy a listed item
pub async fn buy_item(
    auth: AuthenticatedWallet,
    State(state): State<Arc<AppState>>,
    Json(request): Json<BuyItemRequest>,
) -> Result<Json<ApiResponse<serde_json::Value>>, StatusCode> {
    let buyer_wallet = hex::encode(auth.address);

    // Load listing
    let listing_key = format!("{}{}", GAME_LISTING_PREFIX, request.listing_id);
    let kv = state.storage_engine.get_kv();
    let listing_data = match kv.get(q_storage::CF_MANIFEST, listing_key.as_bytes()).await {
        Ok(Some(d)) => d,
        Ok(None) => return Ok(Json(ApiResponse::error("Listing not found".into()))),
        Err(e) => {
            error!("DB error: {}", e);
            return Ok(Json(ApiResponse::error("Database error".into())));
        }
    };

    let mut listing: MarketplaceListing = match serde_json::from_slice(&listing_data) {
        Ok(l) => l,
        Err(e) => {
            error!("Deserialize error: {}", e);
            return Ok(Json(ApiResponse::error("Listing corrupted".into())));
        }
    };

    if !matches!(listing.status, ListingStatus::Active) {
        return Ok(Json(ApiResponse::error("Listing is no longer active".into())));
    }

    if listing.seller_wallet == buyer_wallet {
        return Ok(Json(ApiResponse::error("Cannot buy your own listing".into())));
    }

    // Check buyer balance
    let buyer_balance = state
        .storage_engine
        .get_balance(&buyer_wallet)
        .await
        .unwrap_or(0);
    let price_atomic = (listing.price_qug * 1_000_000_000_000.0) as u128; // 12 decimals
    if buyer_balance < price_atomic {
        return Ok(Json(ApiResponse::error(format!(
            "Insufficient balance. Need {:.2} QUG, have {:.2} QUG",
            listing.price_qug,
            buyer_balance as f64 / 1_000_000_000_000.0
        ))));
    }

    // Transfer: deduct from buyer, credit to seller
    {
        // Deduct buyer
        let new_buyer_balance = buyer_balance.saturating_sub(price_atomic);
        if let Err(e) = state.storage_engine.set_balance(&buyer_wallet, new_buyer_balance).await {
            error!("Failed to deduct buyer balance: {}", e);
            return Ok(Json(ApiResponse::error("Balance update failed".into())));
        }

        // Credit seller
        let seller_balance = state.storage_engine.get_balance(&listing.seller_wallet).await.unwrap_or(0);
        let new_seller_balance = seller_balance.saturating_add(price_atomic);
        if let Err(e) = state.storage_engine.set_balance(&listing.seller_wallet, new_seller_balance).await {
            // Rollback buyer
            let _ = state.storage_engine.set_balance(&buyer_wallet, buyer_balance).await;
            error!("Failed to credit seller balance: {}", e);
            return Ok(Json(ApiResponse::error("Balance update failed".into())));
        }
    }

    // Transfer item ownership
    let item_key = format!("{}{}", GAME_ITEM_PREFIX, listing.item_id);
    if let Ok(Some(item_data)) = kv.get(q_storage::CF_MANIFEST, item_key.as_bytes()).await {
        if let Ok(mut item) = serde_json::from_slice::<GameItem>(&item_data) {
            item.owner_wallet = buyer_wallet.clone();
            item.listed_price = None;
            item.trade_count += 1;
            persist_item(&state, &item).await;
        }
    }

    // Update listing status
    listing.status = ListingStatus::Sold;
    persist_listing(&state, &listing).await;

    let now = current_timestamp();

    // History for buyer
    let buy_history = HistoryEntry {
        id: generate_id(),
        event_type: HistoryEventType::MarketBuy,
        wallet: buyer_wallet.clone(),
        item_id: Some(listing.item_id.clone()),
        item_name: Some(listing.item_name.clone()),
        details: format!("Bought {} for {:.2} QUG", listing.item_name, listing.price_qug),
        timestamp: now,
    };
    persist_history(&state, &buy_history).await;

    // History for seller
    let sell_history = HistoryEntry {
        id: generate_id(),
        event_type: HistoryEventType::MarketSell,
        wallet: listing.seller_wallet.clone(),
        item_id: Some(listing.item_id.clone()),
        item_name: Some(listing.item_name.clone()),
        details: format!("Sold {} for {:.2} QUG", listing.item_name, listing.price_qug),
        timestamp: now,
    };
    persist_history(&state, &sell_history).await;

    info!(
        "🎮 {} bought {} from {} for {:.2} QUG",
        buyer_wallet, listing.item_name, listing.seller_wallet, listing.price_qug
    );

    Ok(Json(ApiResponse::success(serde_json::json!({
        "item_id": listing.item_id,
        "item_name": listing.item_name,
        "price_qug": listing.price_qug,
        "seller": listing.seller_wallet,
        "buyer": buyer_wallet,
    }))))
}

/// POST /api/v1/game-items/open-case — Provably fair case opening
pub async fn open_case(
    auth: AuthenticatedWallet,
    State(state): State<Arc<AppState>>,
    Json(request): Json<OpenCaseRequest>,
) -> Result<Json<ApiResponse<serde_json::Value>>, StatusCode> {
    let wallet_hex = hex::encode(auth.address);

    // Load case template
    let case_key = format!("{}{}", GAME_CASE_PREFIX, request.case_id);
    let kv = state.storage_engine.get_kv();
    let case_data = match kv.get(q_storage::CF_MANIFEST, case_key.as_bytes()).await {
        Ok(Some(d)) => d,
        Ok(None) => return Ok(Json(ApiResponse::error("Case not found".into()))),
        Err(e) => {
            error!("DB error: {}", e);
            return Ok(Json(ApiResponse::error("Database error".into())));
        }
    };

    let case_template: CaseTemplate = match serde_json::from_slice(&case_data) {
        Ok(c) => c,
        Err(e) => {
            error!("Deserialize error: {}", e);
            return Ok(Json(ApiResponse::error("Case corrupted".into())));
        }
    };

    // Check balance for key price
    let balance = state
        .storage_engine
        .get_balance(&wallet_hex)
        .await
        .unwrap_or(0);
    let key_cost_atomic = (case_template.key_price_qug * 1_000_000_000_000.0) as u128;
    if balance < key_cost_atomic {
        return Ok(Json(ApiResponse::error(format!(
            "Need {:.2} QUG for a key, have {:.2} QUG",
            case_template.key_price_qug,
            balance as f64 / 1_000_000_000_000.0
        ))));
    }

    // Deduct key cost
    let new_balance = balance.saturating_sub(key_cost_atomic);
    if let Err(e) = state.storage_engine.set_balance(&wallet_hex, new_balance).await {
        error!("Failed to deduct key cost: {}", e);
        return Ok(Json(ApiResponse::error("Balance update failed".into())));
    }

    // Get nonce for provably fair
    let nonce = increment_wallet_nonce(&state, &wallet_hex).await;

    // Use latest block hash as randomness source
    let block_hash = {
        let height = state.current_height_atomic.load(std::sync::atomic::Ordering::Relaxed);
        if height > 0 {
            match state.storage_engine.get_qblock_by_height(height).await {
                Ok(Some(block)) => {
                    hex::encode(block.header.solutions_root)
                }
                _ => format!("height_{}", height),
            }
        } else {
            "genesis_0".to_string()
        }
    };

    // Provably fair roll
    let (grade, item_idx, float_value, stattrak, seed) =
        provably_fair_open(&block_hash, &wallet_hex, nonce, &case_template);

    // Get the item from the grade pool
    let grade_key = serde_json::to_string(&grade).unwrap_or_default();
    let grade_key_clean = grade_key.trim_matches('"');
    let items_pool = case_template
        .items_by_grade
        .get(grade_key_clean)
        .cloned()
        .unwrap_or_default();

    let (item_name, item_weapon, item_type) = if items_pool.is_empty() {
        // Fallback: shouldn't happen with seeded data
        ("Mystery Item".to_string(), "Unknown".to_string(), ItemType::Skin)
    } else {
        let idx = item_idx % items_pool.len();
        let drop_item = &items_pool[idx];
        (drop_item.name.clone(), drop_item.weapon.clone(), drop_item.item_type)
    };

    let wear = WearCondition::from_float(float_value);
    let now = current_timestamp();
    let item_id = generate_id();

    let new_item = GameItem {
        id: item_id.clone(),
        name: item_name.clone(),
        weapon: item_weapon.clone(),
        collection: case_template.name.clone(),
        grade,
        wear,
        float_value,
        item_type,
        owner_wallet: wallet_hex.clone(),
        stattrak,
        trade_count: 0,
        created_at: now,
        listed_price: None,
        case_open_seed: Some(seed.clone()),
    };

    persist_item(&state, &new_item).await;

    // History
    let history = HistoryEntry {
        id: generate_id(),
        event_type: HistoryEventType::CaseOpen,
        wallet: wallet_hex.clone(),
        item_id: Some(item_id.clone()),
        item_name: Some(item_name.clone()),
        details: format!(
            "Opened {} → {} | {} ({}) {:.4} float{}",
            case_template.name,
            item_name,
            grade.display_name(),
            wear.short_name(),
            float_value,
            if stattrak { " StatTrak™" } else { "" }
        ),
        timestamp: now,
    };
    persist_history(&state, &history).await;

    info!(
        "🎮 {} opened {} → {} ({}, {}, {:.4} float{})",
        wallet_hex,
        case_template.name,
        item_name,
        grade.display_name(),
        wear.short_name(),
        float_value,
        if stattrak { ", StatTrak™" } else { "" }
    );

    Ok(Json(ApiResponse::success(serde_json::json!({
        "item": new_item,
        "grade_info": {
            "name": grade.display_name(),
            "color": grade.color_hex(),
            "drop_rate": format!("{:.3}%", grade.drop_weight()),
        },
        "wear_info": {
            "name": wear.display_name(),
            "short": wear.short_name(),
        },
        "provably_fair": {
            "block_hash": block_hash,
            "wallet": wallet_hex,
            "nonce": nonce,
            "seed": seed,
            "algorithm": "SHA256(block_hash || wallet || nonce || 'QNK_CASE_OPEN')",
        },
        "key_cost_qug": case_template.key_price_qug,
    }))))
}

/// POST /api/v1/game-items/trade-up — 10-to-1 grade upgrade
pub async fn trade_up(
    auth: AuthenticatedWallet,
    State(state): State<Arc<AppState>>,
    Json(request): Json<TradeUpRequest>,
) -> Result<Json<ApiResponse<serde_json::Value>>, StatusCode> {
    let wallet_hex = hex::encode(auth.address);

    if request.item_ids.len() != 10 {
        return Ok(Json(ApiResponse::error(
            "Trade-up requires exactly 10 items".into(),
        )));
    }

    // Check for duplicates
    let unique: std::collections::HashSet<&String> = request.item_ids.iter().collect();
    if unique.len() != 10 {
        return Ok(Json(ApiResponse::error("Duplicate item IDs not allowed".into())));
    }

    // Load all items, verify ownership and same grade
    let mut items: Vec<GameItem> = Vec::new();
    let kv = state.storage_engine.get_kv();

    for item_id in &request.item_ids {
        let key = format!("{}{}", GAME_ITEM_PREFIX, item_id);
        match kv.get(q_storage::CF_MANIFEST, key.as_bytes()).await {
            Ok(Some(data)) => match serde_json::from_slice::<GameItem>(&data) {
                Ok(item) => {
                    if item.owner_wallet != wallet_hex {
                        return Ok(Json(ApiResponse::error(format!(
                            "You don't own item {}",
                            item_id
                        ))));
                    }
                    if item.listed_price.is_some() {
                        return Ok(Json(ApiResponse::error(format!(
                            "Item {} is currently listed for sale. Cancel listing first.",
                            item_id
                        ))));
                    }
                    items.push(item);
                }
                Err(_) => {
                    return Ok(Json(ApiResponse::error(format!(
                        "Item {} is corrupted",
                        item_id
                    ))));
                }
            },
            Ok(None) => {
                return Ok(Json(ApiResponse::error(format!(
                    "Item {} not found",
                    item_id
                ))));
            }
            Err(e) => {
                error!("DB error: {}", e);
                return Ok(Json(ApiResponse::error("Database error".into())));
            }
        }
    }

    // All must be same grade
    let base_grade = items[0].grade;
    if !items.iter().all(|i| i.grade == base_grade) {
        return Ok(Json(ApiResponse::error(
            "All 10 items must be the same grade".into(),
        )));
    }

    // Must have a next grade
    let next_grade = match base_grade.next_grade() {
        Some(g) => g,
        None => {
            return Ok(Json(ApiResponse::error(
                "Cannot trade up from Covert or Contraband grade".into(),
            )));
        }
    };

    // Deterministic result from item IDs
    let result_hash = trade_up_result(&request.item_ids);
    let float_value = hash_to_f64(&result_hash);
    let wear = WearCondition::from_float(float_value);
    let stattrak = {
        let st_hash = sha256_hex(format!("{}||STATTRAK", result_hash).as_bytes());
        hash_to_f64(&st_hash) < 0.10
    };

    // Pick a weapon/name — use hash to select from a pool of upgrade results
    let upgrade_names: Vec<(&str, &str)> = vec![
        ("Quantum Flux", "AK-47"),
        ("Dark Energy", "M4A1-S"),
        ("Supernova", "AWP"),
        ("Warp Drive", "Desert Eagle"),
        ("Photon Beam", "USP-S"),
        ("Nebula Storm", "Glock-18"),
        ("Event Horizon", "M4A4"),
        ("Singularity", "AK-47"),
    ];
    let name_idx = u32::from_str_radix(&result_hash[8..16], 16).unwrap_or(0) as usize % upgrade_names.len();
    let (result_name, result_weapon) = upgrade_names[name_idx];

    let now = current_timestamp();
    let new_item_id = generate_id();

    let result_item = GameItem {
        id: new_item_id.clone(),
        name: result_name.to_string(),
        weapon: result_weapon.to_string(),
        collection: "Trade-Up Contract".to_string(),
        grade: next_grade,
        wear,
        float_value,
        item_type: ItemType::Skin,
        owner_wallet: wallet_hex.clone(),
        stattrak,
        trade_count: 0,
        created_at: now,
        listed_price: None,
        case_open_seed: Some(result_hash.clone()),
    };

    // Delete consumed items
    for item in &items {
        let key = format!("{}{}", GAME_ITEM_PREFIX, item.id);
        // Remove by writing empty (or we could use delete if available)
        // For now, just overwrite ownership to "burned"
        let mut burned = item.clone();
        burned.owner_wallet = "BURNED_TRADE_UP".to_string();
        persist_item(&state, &burned).await;
    }

    // Persist new item
    persist_item(&state, &result_item).await;

    // History
    let history = HistoryEntry {
        id: generate_id(),
        event_type: HistoryEventType::TradeUp,
        wallet: wallet_hex.clone(),
        item_id: Some(new_item_id.clone()),
        item_name: Some(result_name.to_string()),
        details: format!(
            "Trade-Up: 10x {} → {} ({}, {}, {:.4} float{})",
            base_grade.display_name(),
            result_name,
            next_grade.display_name(),
            wear.short_name(),
            float_value,
            if stattrak { ", StatTrak™" } else { "" }
        ),
        timestamp: now,
    };
    persist_history(&state, &history).await;

    info!(
        "🎮 {} traded up 10x {} → {} ({})",
        wallet_hex,
        base_grade.display_name(),
        result_name,
        next_grade.display_name()
    );

    Ok(Json(ApiResponse::success(serde_json::json!({
        "result_item": result_item,
        "consumed_items": request.item_ids,
        "grade_upgrade": {
            "from": base_grade.display_name(),
            "to": next_grade.display_name(),
        },
        "provably_fair": {
            "input_hash": result_hash,
            "algorithm": "SHA256(sorted item IDs joined by '||')",
        },
    }))))
}

// ─── Router ──────────────────────────────────────────────────────────────────

pub fn create_game_items_router() -> Router<Arc<AppState>> {
    Router::new()
        .route("/", get(list_marketplace))
        .route("/stats", get(marketplace_stats))
        .route("/cases", get(list_cases))
        .route("/collections", get(list_collections))
        .route("/:id", get(get_item))
        .route("/inventory/:wallet", get(get_inventory))
        .route("/list", post(list_item_for_sale))
        .route("/buy", post(buy_item))
        .route("/open-case", post(open_case))
        .route("/trade-up", post(trade_up))
}
