// Storage schema types — pure serde structs, no backend dependency.
// Extracted from nova-chat src/storage/database.rs (types only, no InMemoryDatabase).
// libp2p::PeerId references replaced with String for serde compatibility.

use serde::{Deserialize, Serialize};
use super::message::{DeliveryStatus, MessageType};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessageRecord {
    pub id: String,
    pub sender_id: String,
    pub recipient_id: Option<String>,
    pub group_id: Option<String>,
    pub content: String,
    pub timestamp: u64,
    pub message_type: MessageType,
    pub delivery_status: DeliveryStatus,
    pub reply_to: Option<String>,
    pub attachments: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContactRecord {
    pub peer_id: String,
    pub display_name: Option<String>,
    pub last_seen: Option<u64>,
    pub is_blocked: bool,
    pub is_favorite: bool,
    pub added_at: u64,
    pub notes: Option<String>,
}

impl ContactRecord {
    pub fn new(peer_id: impl Into<String>, display_name: Option<String>) -> Self {
        Self {
            peer_id: peer_id.into(),
            display_name,
            last_seen: None,
            is_blocked: false,
            is_favorite: false,
            added_at: now_secs(),
            notes: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationRecord {
    pub conversation_id: String,
    pub conversation_type: ConversationType,
    /// Peer IDs as Strings (wallet addresses in the Quillon context)
    pub participants: Vec<String>,
    pub last_message_id: Option<String>,
    pub last_activity: u64,
    pub is_muted: bool,
    pub is_archived: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum ConversationType {
    DirectMessage,
    Group,
}

fn now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}
