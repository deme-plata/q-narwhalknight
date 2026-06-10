// Wire protocol types for P2P chat messages.
// Extracted from nova-chat src/messaging/protocol.rs + src/storage/database.rs.
// Uses only the richer protocol.rs version of NovaMessage (has message_type + delivery_status).

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum MessageType {
    DirectMessage,
    GroupMessage,
    SystemMessage,
    FileShare,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum DeliveryStatus {
    Pending,
    Sent,
    Delivered,
    Read,
    Failed,
}

/// Primary P2P message type — wire format shared between node and all frontends.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NovaMessage {
    pub id: String,
    pub sender: String,
    pub recipient: String,
    pub content: String,
    pub timestamp: u64,
    pub message_type: MessageType,
    pub delivery_status: DeliveryStatus,
    pub reply_to: Option<String>,
    pub group_id: Option<String>,
    pub attachments: Vec<String>,
}

impl NovaMessage {
    pub fn new_direct(sender: impl Into<String>, recipient: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            sender: sender.into(),
            recipient: recipient.into(),
            content: content.into(),
            timestamp: now_secs(),
            message_type: MessageType::DirectMessage,
            delivery_status: DeliveryStatus::Pending,
            reply_to: None,
            group_id: None,
            attachments: Vec::new(),
        }
    }

    pub fn new_group(sender: impl Into<String>, group_id: impl Into<String>, content: impl Into<String>) -> Self {
        let gid = group_id.into();
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            sender: sender.into(),
            recipient: gid.clone(),
            content: content.into(),
            timestamp: now_secs(),
            message_type: MessageType::GroupMessage,
            delivery_status: DeliveryStatus::Pending,
            reply_to: None,
            group_id: Some(gid),
            attachments: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessageDeliveryReceipt {
    pub message_id: String,
    pub sender_id: String,
    pub recipient_id: String,
    pub status: DeliveryStatus,
    pub timestamp: u64,
}

impl MessageDeliveryReceipt {
    pub fn new(message_id: impl Into<String>, sender_id: impl Into<String>, recipient_id: impl Into<String>, status: DeliveryStatus) -> Self {
        Self {
            message_id: message_id.into(),
            sender_id: sender_id.into(),
            recipient_id: recipient_id.into(),
            status,
            timestamp: now_secs(),
        }
    }
}

fn now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}
