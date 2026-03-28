//! Client configuration resource.
//!
//! Holds server URL and polling interval.  Defaults point at the production
//! bootstrap at `quillon.xyz`.

use bevy::prelude::*;

/// Runtime configuration for the Crown & Ash client.
///
/// Insert this resource *before* `CrownAshNetworkPlugin` to override defaults,
/// or let the plugin's `init_resource` create one with the default values.
#[derive(Resource, Clone)]
pub struct CrownAshConfig {
    /// Base URL of the q-api-server (no trailing slash).
    pub server_url: String,
    /// Seconds between world-snapshot polls.
    pub poll_interval_secs: f32,
}

impl Default for CrownAshConfig {
    fn default() -> Self {
        Self {
            server_url: "https://quillon.xyz".to_string(),
            poll_interval_secs: 5.0,
        }
    }
}
