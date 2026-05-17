//! 🧠 Thought-UI System: 12 Tabs Controlled by EEG
//! Visual interface projected through attosecond laser LED flashing

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// UI color codes for thought interface
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum UIColor {
    Green,  // 🟢 GO / Safe - α-wave burst ≥ 20 µV
    Red,    // 🔴 NO-GO / Attack - β-wave spike ≥ 30 µV
    Yellow, // 🟡 Caution / Pending - α → θ transition
    Pink,   // 🩷 Special Code / Admin - θ-wave burst
}

impl UIColor {
    /// Convert EEG amplitude to UI color
    pub fn from_eeg(eeg_amplitude: f64) -> Self {
        match eeg_amplitude {
            v if v >= 30.0 => UIColor::Red,    // β-wave spike
            v if v >= 20.0 => UIColor::Green,  // α-wave burst
            v if v >= 10.0 => UIColor::Yellow, // α → θ transition
            _ => UIColor::Pink,                // θ-wave burst
        }
    }

    /// Get LED flash frequency for thought projection
    pub fn flash_frequency(&self) -> u8 {
        match self {
            UIColor::Green => 10,  // 10 Hz (alpha wave)
            UIColor::Red => 15,    // 15 Hz (beta wave)
            UIColor::Yellow => 12, // 12 Hz (high alpha)
            UIColor::Pink => 8,    // 8 Hz (theta wave)
        }
    }

    /// Get emoji representation
    pub fn emoji(&self) -> &'static str {
        match self {
            UIColor::Green => "🟢",
            UIColor::Red => "🔴",
            UIColor::Yellow => "🟡",
            UIColor::Pink => "🩷",
        }
    }
}

/// The 12 tabs of the thought interface
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TabType {
    Home,       // 1. Dashboard + chain health
    Peers,      // 2. Tor peer list
    Send,       // 3. Sign & broadcast messages
    Receive,    // 4. Decode incoming packets
    Blocks,     // 5. Latest multiverse blocks
    Memos,      // 6. DNA-stored memos
    Topology,   // 7. Current brane position
    Health,     // 8. Coherence & entropy
    Logs,       // 9. Event log
    Config,     // 10. EEG thresholds
    Bridge,     // 11. Brane-slip controls
    Multiverse, // 12. Parallel-water stats
}

impl TabType {
    /// Get all tab types in order
    pub fn all() -> [TabType; 12] {
        [
            TabType::Home,
            TabType::Peers,
            TabType::Send,
            TabType::Receive,
            TabType::Blocks,
            TabType::Memos,
            TabType::Topology,
            TabType::Health,
            TabType::Logs,
            TabType::Config,
            TabType::Bridge,
            TabType::Multiverse,
        ]
    }

    /// Get tab number (1-12)
    pub fn number(&self) -> u8 {
        match self {
            TabType::Home => 1,
            TabType::Peers => 2,
            TabType::Send => 3,
            TabType::Receive => 4,
            TabType::Blocks => 5,
            TabType::Memos => 6,
            TabType::Topology => 7,
            TabType::Health => 8,
            TabType::Logs => 9,
            TabType::Config => 10,
            TabType::Bridge => 11,
            TabType::Multiverse => 12,
        }
    }

    /// Get mental label for thought navigation
    pub fn mental_label(&self) -> &'static str {
        match self {
            TabType::Home => "Root",
            TabType::Peers => "Others",
            TabType::Send => "Emit",
            TabType::Receive => "Catch",
            TabType::Blocks => "Chain",
            TabType::Memos => "Notes",
            TabType::Topology => "Shape",
            TabType::Health => "Pulse",
            TabType::Logs => "History",
            TabType::Config => "Tune",
            TabType::Bridge => "Hop",
            TabType::Multiverse => "Beyond",
        }
    }

    /// Get tab description
    pub fn description(&self) -> &'static str {
        match self {
            TabType::Home => "Dashboard + chain health",
            TabType::Peers => "Tor peer list",
            TabType::Send => "Sign & broadcast messages",
            TabType::Receive => "Decode incoming packets",
            TabType::Blocks => "Latest multiverse blocks",
            TabType::Memos => "DNA-stored memos",
            TabType::Topology => "Current brane position",
            TabType::Health => "Coherence & entropy",
            TabType::Logs => "Event log",
            TabType::Config => "EEG thresholds",
            TabType::Bridge => "Brane-slip controls",
            TabType::Multiverse => "Parallel-water stats",
        }
    }
}

/// Individual tab state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tab {
    pub tab_type: TabType,
    pub color: UIColor,
    pub active: bool,
    pub data: TabData,
    pub last_updated: u64, // attoseconds
}

/// Data content for each tab type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TabData {
    Dashboard {
        chain_height: u64,
        peer_count: u32,
        health_score: f64,
    },
    PeerList {
        peers: Vec<String>,
        connectivity: f64,
    },
    SendBuffer {
        pending_messages: u32,
        last_broadcast: String,
    },
    ReceiveBuffer {
        new_messages: u32,
        last_received: String,
    },
    BlockChain {
        latest_hash: String,
        block_count: u64,
    },
    MemoStore {
        dna_memos: Vec<String>,
        capacity_used: f64,
    },
    TopologyMap {
        current_brane: String,
        portal_count: u32,
    },
    HealthMonitor {
        coherence: f64,
        entropy: f64,
        temperature: f64,
    },
    EventLog {
        recent_events: Vec<String>,
        log_size: u64,
    },
    Configuration {
        eeg_threshold: f64,
        tor_enabled: bool,
    },
    BridgeControl {
        bridge_count: u32,
        success_rate: f64,
    },
    MultiverseStats {
        parallel_waters: u32,
        sync_status: String,
    },
}

/// Thought UI manager with 12 tabs
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TabManager {
    pub tabs: HashMap<TabType, Tab>,
    pub active_tab: TabType,
    pub eeg_amplitude: f64,
    pub thought_buffer: Vec<String>,
}

impl TabManager {
    /// Create new tab manager with all 12 tabs
    pub fn new() -> Self {
        let mut tabs = HashMap::new();

        for tab_type in TabType::all() {
            let tab = Tab {
                tab_type,
                color: UIColor::Green,
                active: tab_type == TabType::Home,
                data: Self::default_tab_data(&tab_type),
                last_updated: 0,
            };
            tabs.insert(tab_type, tab);
        }

        Self {
            tabs,
            active_tab: TabType::Home,
            eeg_amplitude: 0.0,
            thought_buffer: Vec::new(),
        }
    }

    /// Generate default data for each tab type
    fn default_tab_data(tab_type: &TabType) -> TabData {
        match tab_type {
            TabType::Home => TabData::Dashboard {
                chain_height: 0,
                peer_count: 0,
                health_score: 1.0,
            },
            TabType::Peers => TabData::PeerList {
                peers: Vec::new(),
                connectivity: 0.0,
            },
            TabType::Send => TabData::SendBuffer {
                pending_messages: 0,
                last_broadcast: "None".to_string(),
            },
            TabType::Receive => TabData::ReceiveBuffer {
                new_messages: 0,
                last_received: "None".to_string(),
            },
            TabType::Blocks => TabData::BlockChain {
                latest_hash: "genesis".to_string(),
                block_count: 0,
            },
            TabType::Memos => TabData::MemoStore {
                dna_memos: Vec::new(),
                capacity_used: 0.0,
            },
            TabType::Topology => TabData::TopologyMap {
                current_brane: "origin".to_string(),
                portal_count: 0,
            },
            TabType::Health => TabData::HealthMonitor {
                coherence: 0.92,
                entropy: 0.1,
                temperature: 295.0,
            },
            TabType::Logs => TabData::EventLog {
                recent_events: vec!["System initialized".to_string()],
                log_size: 1,
            },
            TabType::Config => TabData::Configuration {
                eeg_threshold: 20.0,
                tor_enabled: true,
            },
            TabType::Bridge => TabData::BridgeControl {
                bridge_count: 0,
                success_rate: 0.0,
            },
            TabType::Multiverse => TabData::MultiverseStats {
                parallel_waters: 1,
                sync_status: "Syncing".to_string(),
            },
        }
    }

    /// Update UI state based on EEG input
    pub fn update_from_eeg(&mut self, eeg_amplitude: f64) {
        self.eeg_amplitude = eeg_amplitude;
        let new_color = UIColor::from_eeg(eeg_amplitude);

        // Update active tab color
        if let Some(tab) = self.tabs.get_mut(&self.active_tab) {
            tab.color = new_color;
            tab.last_updated = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos() as u64
                / 1_000_000_000;
        }
    }

    /// Switch to tab based on thought pattern
    pub fn navigate_to_tab(&mut self, tab_number: u8) {
        if tab_number >= 1 && tab_number <= 12 {
            let target_tab = TabType::all()[(tab_number - 1) as usize];

            // Deactivate current tab
            if let Some(current) = self.tabs.get_mut(&self.active_tab) {
                current.active = false;
            }

            // Activate new tab
            if let Some(new_tab) = self.tabs.get_mut(&target_tab) {
                new_tab.active = true;
                self.active_tab = target_tab;
            }
        }
    }

    /// Process thought command
    pub fn process_thought(&mut self, thought: &str) -> Vec<String> {
        self.thought_buffer.push(thought.to_string());

        let mut responses = Vec::new();

        // Parse thought commands
        if thought.contains("tab") {
            if let Some(num_str) = thought.split_whitespace().find(|s| s.parse::<u8>().is_ok()) {
                if let Ok(tab_num) = num_str.parse::<u8>() {
                    self.navigate_to_tab(tab_num);
                    responses.push(format!(
                        "Navigated to tab {}: {}",
                        tab_num,
                        self.active_tab.mental_label()
                    ));
                }
            }
        }

        if thought.to_lowercase().contains("status") {
            responses.push(self.get_status_summary());
        }

        responses
    }

    /// Get current status summary
    pub fn get_status_summary(&self) -> String {
        let active_tab = &self.tabs[&self.active_tab];
        format!(
            "Active: Tab {} ({}) {} | EEG: {:.1}µV | Color: {}",
            active_tab.tab_type.number(),
            active_tab.tab_type.mental_label(),
            active_tab.color.emoji(),
            self.eeg_amplitude,
            format!("{:?}", active_tab.color)
        )
    }

    /// Render ASCII visualization of all tabs
    pub fn render_ascii(&self) -> String {
        let mut output = String::new();
        output.push_str("┌────────────────────────────────────────────────────────┐\n");
        output.push_str("│               🧠 AQUA-K-ATTO THOUGHT UI               │\n");
        output.push_str("├────────────────────────────────────────────────────────┤\n");

        for (i, tab_type) in TabType::all().iter().enumerate() {
            let tab = &self.tabs[tab_type];
            let active_marker = if tab.active { "►" } else { " " };
            let color_emoji = tab.color.emoji();

            output.push_str(&format!(
                "│ {}{}. {} {} {:12} │ {:<25} │\n",
                active_marker,
                i + 1,
                color_emoji,
                tab.tab_type.mental_label(),
                "",
                tab.tab_type.description()
            ));
        }

        output.push_str("├────────────────────────────────────────────────────────┤\n");
        output.push_str(&format!(
            "│ EEG: {:.1}µV | Active: {} | Thoughts: {}          │\n",
            self.eeg_amplitude,
            self.active_tab.mental_label(),
            self.thought_buffer.len()
        ));
        output.push_str("└────────────────────────────────────────────────────────┘");

        output
    }

    /// Generate LED flash pattern for current UI state
    pub fn generate_flash_pattern(&self) -> Vec<u8> {
        let active_tab = &self.tabs[&self.active_tab];
        let base_freq = active_tab.color.flash_frequency();

        // Create pattern: tab number encoded in pulse count + color frequency
        let mut pattern = Vec::new();

        // Tab identification pulses
        for _ in 0..active_tab.tab_type.number() {
            pattern.push(25); // High frequency identification pulse
            pattern.push(0); // Gap
        }

        // Color-coded sustained frequency
        for _ in 0..32 {
            pattern.push(base_freq);
        }

        pattern
    }

    /// Update tab data (called by various systems)
    pub fn update_tab_data(&mut self, tab_type: TabType, data: TabData) {
        if let Some(tab) = self.tabs.get_mut(&tab_type) {
            tab.data = data;
            tab.last_updated = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos() as u64
                / 1_000_000_000;
        }
    }

    /// Get current tab data
    pub fn get_tab_data(&self, tab_type: TabType) -> Option<&TabData> {
        self.tabs.get(&tab_type).map(|tab| &tab.data)
    }
}

/// Thought UI state for a single water robot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThoughtUI {
    pub tab: u8, // 1..12
    pub color: UIColor,
    pub label: String,
    pub flash_pattern: Vec<u8>,
    pub attosecond_timestamp: u64,
}

impl ThoughtUI {
    pub fn new() -> Self {
        Self {
            tab: 1,
            color: UIColor::Green,
            label: "Home".to_string(),
            flash_pattern: vec![10; 32],
            attosecond_timestamp: 0,
        }
    }

    /// Create from tab manager state
    pub fn from_tab_manager(manager: &TabManager) -> Self {
        let active_tab = &manager.tabs[&manager.active_tab];
        Self {
            tab: active_tab.tab_type.number(),
            color: active_tab.color,
            label: active_tab.tab_type.mental_label().to_string(),
            flash_pattern: manager.generate_flash_pattern(),
            attosecond_timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos() as u64
                / 1_000_000_000,
        }
    }

    /// EEG → color mapping (legacy compatibility)
    pub fn set_by_eeg(&mut self, eeg: f64) {
        self.color = UIColor::from_eeg(eeg);
        self.flash_pattern = vec![self.color.flash_frequency(); 32];
    }

    /// Encode into LED flash pattern (alpha-wave carrier)
    pub fn to_flash(&self) -> Vec<u8> {
        self.flash_pattern.clone()
    }

    /// Convert to thought-visible AR overlay data
    pub fn to_ar_overlay(&self) -> String {
        format!(
            "{{\"tab\":{},\"color\":\"{:?}\",\"label\":\"{}\",\"freq\":{},\"timestamp\":{}}}",
            self.tab,
            self.color,
            self.label,
            self.color.flash_frequency(),
            self.attosecond_timestamp
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ui_color_from_eeg() {
        assert_eq!(UIColor::from_eeg(35.0), UIColor::Red);
        assert_eq!(UIColor::from_eeg(25.0), UIColor::Green);
        assert_eq!(UIColor::from_eeg(15.0), UIColor::Yellow);
        assert_eq!(UIColor::from_eeg(5.0), UIColor::Pink);
    }

    #[test]
    fn test_tab_manager_creation() {
        let manager = TabManager::new();
        assert_eq!(manager.tabs.len(), 12);
        assert_eq!(manager.active_tab, TabType::Home);

        for tab_type in TabType::all() {
            assert!(manager.tabs.contains_key(&tab_type));
        }
    }

    #[test]
    fn test_tab_navigation() {
        let mut manager = TabManager::new();
        manager.navigate_to_tab(5); // Blocks tab
        assert_eq!(manager.active_tab, TabType::Blocks);

        let home_tab = &manager.tabs[&TabType::Home];
        let blocks_tab = &manager.tabs[&TabType::Blocks];
        assert!(!home_tab.active);
        assert!(blocks_tab.active);
    }

    #[test]
    fn test_thought_processing() {
        let mut manager = TabManager::new();
        let responses = manager.process_thought("go to tab 7");
        assert!(!responses.is_empty());
        assert_eq!(manager.active_tab, TabType::Topology);
    }

    #[test]
    fn test_flash_pattern_generation() {
        let manager = TabManager::new();
        let pattern = manager.generate_flash_pattern();
        assert!(!pattern.is_empty());
        assert!(pattern.len() > 32); // Should have identification + sustained pattern
    }
}
