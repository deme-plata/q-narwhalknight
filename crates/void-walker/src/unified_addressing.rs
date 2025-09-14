//! 🌌 Unified Multiverse Addressing System
//! Complete addressing scheme across all 5 multiverse theories
//! Enables water robots to navigate any theoretical multiverse framework

use serde::{Deserialize, Serialize};
use sha3::{Digest, Sha3_256};
use std::fmt;
use std::str::FromStr;

use crate::brane::{BraneCoord, TopoCharge};
use crate::eternal_inflation::{BubbleId, IsotopicSignature};
use crate::k_parameter::KParameterState;
use crate::many_worlds::{BranchId, PhaseFingerprint};

/// Universal multiverse address covering all 5 theories
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MultiverseAddress {
    /// Many-Worlds branch identifier
    pub branch_id: Option<BranchId>,
    /// Eternal Inflation bubble identifier  
    pub bubble_id: Option<BubbleId>,
    /// String Landscape brane coordinates
    pub brane_coord: Option<BraneCoord>,
    /// Tegmark Level IV mathematical structure
    pub k_parameter: Option<f64>,
    /// Brane Cosmology attosecond pulse timing
    pub pulse_timing: Option<u64>,
    /// Unified address hash for quick comparison
    pub unified_hash: [u8; 32],
    /// Address validity score (0..1)
    pub validity: f64,
}

impl MultiverseAddress {
    /// Create complete multiverse address from all components
    pub fn new(
        branch_id: Option<BranchId>,
        bubble_id: Option<BubbleId>,
        brane_coord: Option<BraneCoord>,
        k_parameter: Option<f64>,
        pulse_timing: Option<u64>,
    ) -> Self {
        let unified_hash = Self::compute_unified_hash(
            &branch_id,
            &bubble_id,
            &brane_coord,
            &k_parameter,
            &pulse_timing,
        );

        let validity = Self::calculate_validity(
            &branch_id,
            &bubble_id,
            &brane_coord,
            &k_parameter,
            &pulse_timing,
        );

        Self {
            branch_id,
            bubble_id,
            brane_coord,
            k_parameter,
            pulse_timing,
            unified_hash,
            validity,
        }
    }

    /// Create address from Many-Worlds branch only
    pub fn from_branch(branch_id: BranchId) -> Self {
        Self::new(Some(branch_id), None, None, None, None)
    }

    /// Create address from Eternal Inflation bubble only
    pub fn from_bubble(bubble_id: BubbleId) -> Self {
        Self::new(None, Some(bubble_id), None, None, None)
    }

    /// Create address from String Landscape brane only
    pub fn from_brane(brane_coord: BraneCoord) -> Self {
        Self::new(None, None, Some(brane_coord), None, None)
    }

    /// Create address from Tegmark Level IV K-parameter only
    pub fn from_k_parameter(k_param: f64) -> Self {
        Self::new(None, None, None, Some(k_param), None)
    }

    /// Create address from Brane Cosmology pulse timing only
    pub fn from_pulse(pulse_timing: u64) -> Self {
        Self::new(None, None, None, None, Some(pulse_timing))
    }

    /// Create complete address (all 5 theories)
    pub fn complete(
        branch_id: BranchId,
        bubble_id: BubbleId,
        brane_coord: BraneCoord,
        k_parameter: f64,
        pulse_timing: u64,
    ) -> Self {
        Self::new(
            Some(branch_id),
            Some(bubble_id),
            Some(brane_coord),
            Some(k_parameter),
            Some(pulse_timing),
        )
    }

    /// Compute unified hash from all components
    fn compute_unified_hash(
        branch_id: &Option<BranchId>,
        bubble_id: &Option<BubbleId>,
        brane_coord: &Option<BraneCoord>,
        k_parameter: &Option<f64>,
        pulse_timing: &Option<u64>,
    ) -> [u8; 32] {
        let mut hasher = Sha3_256::new();

        // Hash branch component
        if let Some(branch) = branch_id {
            hasher.update(b"BRANCH:");
            hasher.update(branch);
        }

        // Hash bubble component
        if let Some(bubble) = bubble_id {
            hasher.update(b"BUBBLE:");
            hasher.update(bubble);
        }

        // Hash brane component
        if let Some(brane) = brane_coord {
            hasher.update(b"BRANE:");
            for theta in &brane.theta {
                hasher.update(&theta.to_le_bytes());
            }
        }

        // Hash K-parameter component
        if let Some(k) = k_parameter {
            hasher.update(b"K_PARAM:");
            hasher.update(&k.to_le_bytes());
        }

        // Hash pulse timing component
        if let Some(pulse) = pulse_timing {
            hasher.update(b"PULSE:");
            hasher.update(&pulse.to_le_bytes());
        }

        hasher.update(b"UNIFIED_MULTIVERSE_ADDRESS");
        hasher.finalize().into()
    }

    /// Calculate address validity score
    fn calculate_validity(
        branch_id: &Option<BranchId>,
        bubble_id: &Option<BubbleId>,
        brane_coord: &Option<BraneCoord>,
        k_parameter: &Option<f64>,
        pulse_timing: &Option<u64>,
    ) -> f64 {
        let mut components = 0;
        let mut validity: f64 = 0.0;

        // Many-Worlds validity
        if branch_id.is_some() {
            components += 1;
            validity += 0.2; // 20% for each theory
        }

        // Eternal Inflation validity
        if bubble_id.is_some() {
            components += 1;
            validity += 0.2;
        }

        // String Landscape validity
        if let Some(brane) = brane_coord {
            components += 1;
            // Check if brane coordinates are in valid range [0, 2π)
            let valid_coords = brane
                .theta
                .iter()
                .all(|&theta| theta >= 0.0 && theta < 2.0 * std::f64::consts::PI);
            validity += if valid_coords { 0.2 } else { 0.1 };
        }

        // Tegmark Level IV validity
        if let Some(k) = k_parameter {
            components += 1;
            // Check if K-parameter is in reasonable mathematical range
            validity += if k.is_finite() && *k > 0.0 { 0.2 } else { 0.1 };
        }

        // Brane Cosmology validity
        if pulse_timing.is_some() {
            components += 1;
            validity += 0.2;
        }

        // Bonus for completeness
        if components == 5 {
            validity += 0.1; // 10% bonus for complete address
        }

        validity.min(1.0)
    }

    /// Check if address is complete (all 5 theories)
    pub fn is_complete(&self) -> bool {
        self.branch_id.is_some()
            && self.bubble_id.is_some()
            && self.brane_coord.is_some()
            && self.k_parameter.is_some()
            && self.pulse_timing.is_some()
    }

    /// Get theory coverage count (0-5)
    pub fn theory_coverage(&self) -> u8 {
        let mut count = 0;
        if self.branch_id.is_some() {
            count += 1;
        }
        if self.bubble_id.is_some() {
            count += 1;
        }
        if self.brane_coord.is_some() {
            count += 1;
        }
        if self.k_parameter.is_some() {
            count += 1;
        }
        if self.pulse_timing.is_some() {
            count += 1;
        }
        count
    }

    /// Calculate address distance in unified space
    pub fn distance(&self, other: &Self) -> f64 {
        let mut total_distance = 0.0;
        let mut components = 0;

        // Branch distance (Many-Worlds)
        match (&self.branch_id, &other.branch_id) {
            (Some(a), Some(b)) => {
                // Hamming distance for branch IDs
                let hamming = a
                    .iter()
                    .zip(b.iter())
                    .map(|(x, y)| (x ^ y).count_ones())
                    .sum::<u32>() as f64
                    / 256.0; // Normalize by bit count
                total_distance += hamming;
                components += 1;
            }
            _ => {} // Skip if either is missing
        }

        // Bubble distance (Eternal Inflation)
        match (&self.bubble_id, &other.bubble_id) {
            (Some(a), Some(b)) => {
                let hamming = a
                    .iter()
                    .zip(b.iter())
                    .map(|(x, y)| (x ^ y).count_ones())
                    .sum::<u32>() as f64
                    / 256.0;
                total_distance += hamming;
                components += 1;
            }
            _ => {}
        }

        // Brane distance (String Landscape)
        match (&self.brane_coord, &other.brane_coord) {
            (Some(a), Some(b)) => {
                total_distance += a.phase_distance(b) / std::f64::consts::PI; // Normalize
                components += 1;
            }
            _ => {}
        }

        // K-parameter distance (Tegmark Level IV)
        match (&self.k_parameter, &other.k_parameter) {
            (Some(a), Some(b)) => {
                total_distance += (a - b).abs() / (a.abs().max(b.abs()).max(1.0)); // Relative distance
                components += 1;
            }
            _ => {}
        }

        // Pulse timing distance (Brane Cosmology)
        match (&self.pulse_timing, &other.pulse_timing) {
            (Some(a), Some(b)) => {
                let max_pulse = a.max(b).max(&1);
                total_distance += (*a as i64 - *b as i64).abs() as f64 / *max_pulse as f64;
                components += 1;
            }
            _ => {}
        }

        if components > 0 {
            total_distance / components as f64
        } else {
            1.0 // Maximum distance if no common components
        }
    }

    /// Convert to canonical string representation
    pub fn to_canonical_string(&self) -> String {
        let mut parts = Vec::new();

        if let Some(branch) = &self.branch_id {
            parts.push(format!("Branch-{}", hex::encode(&branch[..8])));
        }

        if let Some(bubble) = &self.bubble_id {
            parts.push(format!("Bubble-{}", hex::encode(&bubble[..8])));
        }

        if let Some(brane) = &self.brane_coord {
            parts.push(brane.portal_address());
        }

        if let Some(k) = &self.k_parameter {
            parts.push(format!("K-{:.6}", k));
        }

        if let Some(pulse) = &self.pulse_timing {
            parts.push(format!("Pulse-{}", pulse));
        }

        if parts.is_empty() {
            "NULL-ADDRESS".to_string()
        } else {
            parts.join(":")
        }
    }

    /// Get short address hash (16 chars)
    pub fn short_hash(&self) -> String {
        hex::encode(&self.unified_hash[..8])
    }

    /// Merge with another address (combine non-None components)
    pub fn merge(&self, other: &Self) -> Self {
        Self::new(
            self.branch_id.or(other.branch_id),
            self.bubble_id.or(other.bubble_id),
            self.brane_coord.or(other.brane_coord),
            self.k_parameter.or(other.k_parameter),
            self.pulse_timing.or(other.pulse_timing),
        )
    }

    /// Check if this address is compatible with another (no conflicts)
    pub fn is_compatible(&self, other: &Self) -> bool {
        // Check branch compatibility
        if let (Some(a), Some(b)) = (&self.branch_id, &other.branch_id) {
            if a != b {
                return false;
            }
        }

        // Check bubble compatibility
        if let (Some(a), Some(b)) = (&self.bubble_id, &other.bubble_id) {
            if a != b {
                return false;
            }
        }

        // Check brane compatibility (within phase tolerance)
        if let (Some(a), Some(b)) = (&self.brane_coord, &other.brane_coord) {
            if a.phase_distance(b) > 0.1 {
                return false;
            }
        }

        // Check K-parameter compatibility (within relative tolerance)
        if let (Some(a), Some(b)) = (&self.k_parameter, &other.k_parameter) {
            let rel_diff = (a - b).abs() / a.abs().max(b.abs()).max(1.0);
            if rel_diff > 0.01 {
                return false;
            } // 1% tolerance
        }

        // Check pulse timing compatibility (within 1000 attoseconds)
        if let (Some(a), Some(b)) = (&self.pulse_timing, &other.pulse_timing) {
            if (*a as i64 - *b as i64).abs() > 1000 {
                return false;
            }
        }

        true
    }
}

impl fmt::Display for MultiverseAddress {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} [{}% valid, {}/5 theories]",
            self.to_canonical_string(),
            (self.validity * 100.0) as u32,
            self.theory_coverage()
        )
    }
}

impl FromStr for MultiverseAddress {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        if s == "NULL-ADDRESS" {
            return Ok(Self::new(None, None, None, None, None));
        }

        let parts: Vec<&str> = s.split(':').collect();
        let mut branch_id = None;
        let mut bubble_id = None;
        let mut brane_coord = None;
        let mut k_parameter = None;
        let mut pulse_timing = None;

        for part in parts {
            if part.starts_with("Branch-") {
                let hex_part = &part[7..];
                if hex_part.len() >= 16 {
                    let mut branch = [0u8; 32];
                    if let Ok(bytes) = hex::decode(&hex_part[..16]) {
                        branch[..8].copy_from_slice(&bytes);
                        branch_id = Some(branch);
                    }
                }
            } else if part.starts_with("Bubble-") {
                let hex_part = &part[7..];
                if hex_part.len() >= 16 {
                    let mut bubble = [0u8; 32];
                    if let Ok(bytes) = hex::decode(&hex_part[..16]) {
                        bubble[..8].copy_from_slice(&bytes);
                        bubble_id = Some(bubble);
                    }
                }
            } else if part.starts_with("mv-") {
                // Parse brane coordinates (simplified)
                let hex_part = &part[3..];
                if hex_part.len() >= 6 {
                    let mut theta = [0.0; 6];
                    for (i, chunk) in hex_part
                        .chars()
                        .collect::<Vec<_>>()
                        .chunks(2)
                        .enumerate()
                        .take(3)
                    {
                        if chunk.len() == 2 {
                            let hex_str: String = chunk.iter().collect();
                            if let Ok(val) = u8::from_str_radix(&hex_str, 16) {
                                theta[i] = (val as f64 / 255.0) * 2.0 * std::f64::consts::PI;
                            }
                        }
                    }
                    brane_coord = Some(BraneCoord { theta });
                }
            } else if part.starts_with("K-") {
                if let Ok(k) = part[2..].parse::<f64>() {
                    k_parameter = Some(k);
                }
            } else if part.starts_with("Pulse-") {
                if let Ok(pulse) = part[6..].parse::<u64>() {
                    pulse_timing = Some(pulse);
                }
            }
        }

        Ok(Self::new(
            branch_id,
            bubble_id,
            brane_coord,
            k_parameter,
            pulse_timing,
        ))
    }
}

/// Multiverse navigation router
#[derive(Clone, Debug, Default)]
pub struct MultiverseRouter {
    /// Address routing table
    pub routes: std::collections::HashMap<[u8; 32], Vec<MultiverseAddress>>,
    /// Navigation history
    pub navigation_history: Vec<(MultiverseAddress, MultiverseAddress)>,
}

impl MultiverseRouter {
    /// Create new multiverse router
    pub fn new() -> Self {
        Self {
            routes: std::collections::HashMap::new(),
            navigation_history: Vec::new(),
        }
    }

    /// Add route between two addresses
    pub fn add_route(&mut self, from: MultiverseAddress, to: MultiverseAddress) {
        let route_key = from.unified_hash;
        let routes = self.routes.entry(route_key).or_insert_with(Vec::new);
        routes.push(to);
    }

    /// Find route from source to destination
    pub fn find_route(
        &self,
        from: &MultiverseAddress,
        to: &MultiverseAddress,
    ) -> Option<Vec<MultiverseAddress>> {
        // Simplified pathfinding - direct route if exists
        if let Some(routes) = self.routes.get(&from.unified_hash) {
            for route in routes {
                if route.unified_hash == to.unified_hash {
                    return Some(vec![from.clone(), route.clone()]);
                }
            }
        }

        // Try to find compatible intermediate address
        for (intermediate_key, routes) in &self.routes {
            let intermediate = MultiverseAddress::new(None, None, None, None, None); // Placeholder
            if from.is_compatible(&intermediate)
                && routes.iter().any(|r| r.unified_hash == to.unified_hash)
            {
                return Some(vec![from.clone(), intermediate, to.clone()]);
            }
        }

        None
    }

    /// Record navigation between addresses
    pub fn record_navigation(&mut self, from: MultiverseAddress, to: MultiverseAddress) {
        self.navigation_history.push((from, to));

        // Limit history size
        if self.navigation_history.len() > 10000 {
            self.navigation_history.remove(0);
        }
    }

    /// Get navigation statistics
    pub fn get_navigation_stats(&self) -> NavigationStats {
        NavigationStats {
            total_routes: self.routes.len(),
            total_navigations: self.navigation_history.len(),
            unique_addresses: self
                .navigation_history
                .iter()
                .flat_map(|(from, to)| vec![from, to])
                .map(|addr| addr.unified_hash)
                .collect::<std::collections::HashSet<_>>()
                .len(),
        }
    }
}

/// Navigation statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NavigationStats {
    pub total_routes: usize,
    pub total_navigations: usize,
    pub unique_addresses: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::brane::BraneCoord;

    #[test]
    fn test_address_creation() {
        let branch_id = [1u8; 32];
        let bubble_id = [2u8; 32];
        let brane_coord = BraneCoord::origin();
        let k_parameter = 7.001234;
        let pulse_timing = 1000000;

        let addr = MultiverseAddress::complete(
            branch_id,
            bubble_id,
            brane_coord,
            k_parameter,
            pulse_timing,
        );

        assert!(addr.is_complete());
        assert_eq!(addr.theory_coverage(), 5);
        assert!(addr.validity > 0.9);
    }

    #[test]
    fn test_partial_addresses() {
        let branch_addr = MultiverseAddress::from_branch([1u8; 32]);
        assert_eq!(branch_addr.theory_coverage(), 1);
        assert!(!branch_addr.is_complete());

        let k_addr = MultiverseAddress::from_k_parameter(7.5);
        assert_eq!(k_addr.theory_coverage(), 1);
    }

    #[test]
    fn test_address_distance() {
        let addr1 = MultiverseAddress::from_k_parameter(7.0);
        let addr2 = MultiverseAddress::from_k_parameter(8.0);

        let distance = addr1.distance(&addr2);
        assert!(distance > 0.0);
        assert!(distance < 1.0);

        // Distance to self should be 0
        let self_distance = addr1.distance(&addr1);
        assert!(self_distance < 1e-10);
    }

    #[test]
    fn test_address_string_conversion() {
        let addr = MultiverseAddress::from_k_parameter(7.001234);
        let string_repr = addr.to_canonical_string();
        assert!(string_repr.contains("K-7.001234"));

        // Test parsing
        let parsed = MultiverseAddress::from_str(&string_repr).unwrap();
        assert_eq!(parsed.k_parameter, addr.k_parameter);
    }

    #[test]
    fn test_address_compatibility() {
        let addr1 = MultiverseAddress::from_k_parameter(7.0);
        let addr2 = MultiverseAddress::from_branch([1u8; 32]);

        // Should be compatible (no conflicts)
        assert!(addr1.is_compatible(&addr2));

        // Conflicting K-parameters should not be compatible
        let addr3 = MultiverseAddress::from_k_parameter(8.0);
        assert!(!addr1.is_compatible(&addr3));
    }

    #[test]
    fn test_address_merge() {
        let branch_addr = MultiverseAddress::from_branch([1u8; 32]);
        let k_addr = MultiverseAddress::from_k_parameter(7.0);

        let merged = branch_addr.merge(&k_addr);
        assert_eq!(merged.theory_coverage(), 2);
        assert!(merged.branch_id.is_some());
        assert!(merged.k_parameter.is_some());
    }

    #[test]
    fn test_multiverse_router() {
        let mut router = MultiverseRouter::new();

        let addr1 = MultiverseAddress::from_k_parameter(7.0);
        let addr2 = MultiverseAddress::from_branch([1u8; 32]);

        router.add_route(addr1.clone(), addr2.clone());
        router.record_navigation(addr1.clone(), addr2.clone());

        let stats = router.get_navigation_stats();
        assert_eq!(stats.total_routes, 1);
        assert_eq!(stats.total_navigations, 1);
    }
}
