//! Q-NarwhalKnight Adaptive Node Configuration
//!
//! This crate provides intelligent auto-configuration for blockchain nodes
//! based on available system resources. The goal is "one binary that just works"
//! - the node automatically adapts its behavior to the hardware it runs on.

pub mod resource_profile;
pub mod adaptive_config;

pub use resource_profile::{SystemResourceProfile, DeviceClass};
pub use adaptive_config::{AdaptiveNodeConfig, PruningMode};
