// q-chat-types — canonical wire types for the Quillon P2P chat subsystem.
//
// Extracted from nova-chat and adapted for the node context:
// - No libp2p version dependency (PeerId → String throughout)
// - No sled/RocksDB dependency (pure serde structs)
// - No UI/platform dependencies
// - Compatible with nova-chat wire protocol over the signaling WebSocket

pub mod call;
pub mod codec;
pub mod message;
pub mod routing;
pub mod stats;
pub mod storage;

// Convenient re-exports
pub use call::{CallState, CallType, IceCandidate, PeerInfo, QualityPreset, Resolution};
pub use codec::{AudioCodec, CodecManager, VideoCodec};
pub use message::{DeliveryStatus, MessageDeliveryReceipt, MessageType, NovaMessage};
pub use routing::{RouteAction, RouteMessage, RouteMessageType, RouterConfig, RoutingStats};
pub use storage::{ContactRecord, ConversationRecord, ConversationType, MessageRecord};
