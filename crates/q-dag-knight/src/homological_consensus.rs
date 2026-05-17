//! Homological Consensus for DAG-Knight Fork Detection
//!
//! v3.4.6-beta: Ported from QTFT blockchain for better partition handling.
//!
//! ## Algebraic Topology for Consensus
//!
//! Uses homology theory to detect network partitions and competing forks:
//!
//! - **H₀ (0th Betti number)**: Connected components
//!   - H₀ = 1: Healthy single connected chain
//!   - H₀ > 1: Network partition detected (multiple disconnected chains)
//!
//! - **H₁ (1st Betti number)**: Independent cycles/forks
//!   - H₁ = 0: No forks, single linear progression
//!   - H₁ > 0: Competing forks detected (independent chains from common ancestor)
//!
//! ## Formula
//!
//! For a DAG with V vertices, E edges, C connected components:
//! - H₀ = C (number of connected components)
//! - H₁ ≈ tips - 1 (for blockchain DAGs, where tips = competing heads)
//!
//! ## Performance
//!
//! - **Computation**: O(V + E) using union-find for H₀
//! - **Memory**: O(V) for tracking visited vertices
//! - **Frequency**: Should be called periodically (every ~10 blocks)

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, error, info, warn};

/// Homological state for DAG consensus
///
/// Captures topological properties of the blockchain DAG using
/// algebraic topology concepts (simplified for blockchain use).
#[derive(Debug, Clone)]
pub struct HomologicalState {
    /// H₀ = number of connected components (should be 1 for healthy chain)
    pub h0_dimension: usize,

    /// H₁ = number of independent cycles/forks (should be 0 for linear chain)
    /// In blockchain context: number of competing tips - 1
    pub h1_dimension: usize,

    /// Active chain tips (competing heads)
    pub chain_tips: HashSet<[u8; 32]>,

    /// Timestamp of last computation
    pub computed_at: std::time::Instant,

    /// Height range analyzed [min_height, max_height]
    pub height_range: (u64, u64),
}

impl Default for HomologicalState {
    fn default() -> Self {
        Self {
            h0_dimension: 0,
            h1_dimension: 0,
            chain_tips: HashSet::new(),
            computed_at: std::time::Instant::now(),
            height_range: (0, 0),
        }
    }
}

impl HomologicalState {
    /// Check if topology is valid (single connected chain with no forks)
    #[inline]
    pub fn is_valid(&self) -> bool {
        self.h0_dimension == 1 && self.h1_dimension == 0
    }

    /// Detect network partition (multiple disconnected components)
    #[inline]
    pub fn has_partition(&self) -> bool {
        self.h0_dimension > 1
    }

    /// Detect competing forks
    #[inline]
    pub fn has_forks(&self) -> bool {
        self.h1_dimension > 0
    }

    /// Get fork count (number of independent chains minus 1)
    #[inline]
    pub fn fork_count(&self) -> usize {
        self.h1_dimension
    }

    /// Get number of competing tips
    #[inline]
    pub fn tip_count(&self) -> usize {
        self.chain_tips.len()
    }
}

/// Result of homological fork detection
#[derive(Debug, Clone)]
pub enum HomologicalResult {
    /// Healthy topology: single connected chain with no forks
    Healthy {
        height: u64,
        tip: [u8; 32],
    },

    /// Network partition detected: multiple disconnected components
    NetworkPartition {
        components: usize,
        tips_per_component: Vec<usize>,
    },

    /// Competing forks detected: multiple chains from common ancestor
    CompetingForks {
        fork_count: usize,
        tips: HashSet<[u8; 32]>,
        common_ancestor_height: Option<u64>,
    },

    /// Insufficient data for analysis
    InsufficientData {
        reason: String,
    },
}

/// Simple block representation for homological analysis
#[derive(Debug, Clone)]
pub struct BlockNode {
    pub hash: [u8; 32],
    pub height: u64,
    pub parent_hash: [u8; 32],
    pub timestamp: u64,
}

/// Homological consensus analyzer
///
/// Tracks DAG topology and detects forks/partitions using
/// homology theory concepts.
pub struct HomologicalConsensus {
    /// Block nodes indexed by hash for O(1) lookup
    blocks: Arc<RwLock<HashMap<[u8; 32], BlockNode>>>,

    /// Children mapping: parent_hash -> [child_hashes]
    children: Arc<RwLock<HashMap<[u8; 32], Vec<[u8; 32]>>>>,

    /// Current chain tips (blocks with no children)
    tips: Arc<RwLock<HashSet<[u8; 32]>>>,

    /// Cached homological state
    cached_state: Arc<RwLock<Option<HomologicalState>>>,

    /// Configuration
    config: HomologicalConfig,
}

/// Configuration for homological analysis
#[derive(Debug, Clone)]
pub struct HomologicalConfig {
    /// Maximum blocks to analyze (memory bound)
    pub max_blocks: usize,

    /// How often to recompute homology (in blocks)
    pub recompute_interval: u64,

    /// Maximum allowed forks before alerting
    pub max_allowed_forks: usize,

    /// Maximum allowed partitions before alerting
    pub max_allowed_partitions: usize,
}

impl Default for HomologicalConfig {
    fn default() -> Self {
        Self {
            max_blocks: 100_000,
            recompute_interval: 10,
            max_allowed_forks: 3,
            max_allowed_partitions: 1,
        }
    }
}

impl HomologicalConsensus {
    /// Create new homological consensus analyzer
    pub fn new(config: HomologicalConfig) -> Self {
        info!("🔬 [HOMOLOGY] Initializing homological consensus analyzer");
        info!("   Max blocks: {}", config.max_blocks);
        info!("   Recompute interval: {} blocks", config.recompute_interval);
        info!("   Max allowed forks: {}", config.max_allowed_forks);

        Self {
            blocks: Arc::new(RwLock::new(HashMap::with_capacity(config.max_blocks))),
            children: Arc::new(RwLock::new(HashMap::new())),
            tips: Arc::new(RwLock::new(HashSet::new())),
            cached_state: Arc::new(RwLock::new(None)),
            config,
        }
    }

    /// Add a block to the homological analyzer
    pub async fn add_block(&self, block: BlockNode) {
        let hash = block.hash;
        let parent_hash = block.parent_hash;

        let mut blocks = self.blocks.write().await;
        let mut children = self.children.write().await;
        let mut tips = self.tips.write().await;

        // Skip if already added
        if blocks.contains_key(&hash) {
            return;
        }

        // Add block
        blocks.insert(hash, block);

        // Update children mapping
        children.entry(parent_hash).or_default().push(hash);

        // Update tips: new block is a tip, parent is no longer a tip
        tips.insert(hash);
        tips.remove(&parent_hash);

        // Memory bound: remove oldest blocks if exceeding limit
        if blocks.len() > self.config.max_blocks {
            self.prune_oldest_blocks(&mut blocks, &mut children, &mut tips).await;
        }

        // Invalidate cache
        let mut cached = self.cached_state.write().await;
        *cached = None;

        debug!("🔬 [HOMOLOGY] Added block {:?}, tips: {}", &hash[..4], tips.len());
    }

    /// Prune oldest blocks to stay within memory limit
    async fn prune_oldest_blocks(
        &self,
        blocks: &mut HashMap<[u8; 32], BlockNode>,
        children: &mut HashMap<[u8; 32], Vec<[u8; 32]>>,
        tips: &mut HashSet<[u8; 32]>,
    ) {
        // Find minimum height to keep (keep last 90% of max_blocks)
        let keep_count = self.config.max_blocks * 9 / 10;
        let remove_count = blocks.len() - keep_count;

        if remove_count == 0 {
            return;
        }

        // Find blocks with lowest heights to remove
        let mut heights: Vec<(u64, [u8; 32])> = blocks
            .iter()
            .map(|(hash, block)| (block.height, *hash))
            .collect();
        heights.sort_by_key(|(h, _)| *h);

        let to_remove: Vec<[u8; 32]> = heights
            .into_iter()
            .take(remove_count)
            .map(|(_, hash)| hash)
            .collect();

        for hash in to_remove {
            blocks.remove(&hash);
            children.remove(&hash);
            tips.remove(&hash);
        }

        debug!("🔬 [HOMOLOGY] Pruned {} old blocks", remove_count);
    }

    /// Compute homological state from current DAG
    pub async fn compute_homology(&self) -> HomologicalState {
        // Check cache first
        if let Some(cached) = &*self.cached_state.read().await {
            if cached.computed_at.elapsed().as_secs() < 5 {
                return cached.clone();
            }
        }

        let blocks = self.blocks.read().await;
        let tips = self.tips.read().await;

        if blocks.is_empty() {
            return HomologicalState::default();
        }

        // Compute H₀ using union-find for connected components
        let h0_dimension = self.compute_connected_components(&blocks).await;

        // Compute H₁ from tip count
        // In blockchain DAGs, forks = tips - 1 (one tip is the main chain)
        let h1_dimension = if tips.len() > 1 {
            tips.len() - 1
        } else {
            0
        };

        // Find height range
        let min_height = blocks.values().map(|b| b.height).min().unwrap_or(0);
        let max_height = blocks.values().map(|b| b.height).max().unwrap_or(0);

        let state = HomologicalState {
            h0_dimension,
            h1_dimension,
            chain_tips: tips.clone(),
            computed_at: std::time::Instant::now(),
            height_range: (min_height, max_height),
        };

        // Cache the result
        let mut cached = self.cached_state.write().await;
        *cached = Some(state.clone());

        info!(
            "🔬 [HOMOLOGY] Computed: H₀={}, H₁={}, tips={}, heights=[{}, {}]",
            state.h0_dimension,
            state.h1_dimension,
            state.chain_tips.len(),
            min_height,
            max_height
        );

        state
    }

    /// Compute connected components using union-find
    async fn compute_connected_components(&self, blocks: &HashMap<[u8; 32], BlockNode>) -> usize {
        if blocks.is_empty() {
            return 0;
        }

        // Simple BFS-based connected components (more readable than union-find)
        let mut visited: HashSet<[u8; 32]> = HashSet::new();
        let mut components = 0;

        let children = self.children.read().await;

        for hash in blocks.keys() {
            if visited.contains(hash) {
                continue;
            }

            // BFS from this block
            let mut queue: VecDeque<[u8; 32]> = VecDeque::new();
            queue.push_back(*hash);

            while let Some(current) = queue.pop_front() {
                if visited.contains(&current) {
                    continue;
                }
                visited.insert(current);

                // Add parent to queue
                if let Some(block) = blocks.get(&current) {
                    if blocks.contains_key(&block.parent_hash) && !visited.contains(&block.parent_hash) {
                        queue.push_back(block.parent_hash);
                    }
                }

                // Add children to queue
                if let Some(child_hashes) = children.get(&current) {
                    for child in child_hashes {
                        if !visited.contains(child) {
                            queue.push_back(*child);
                        }
                    }
                }
            }

            components += 1;
        }

        components
    }

    /// Detect fork situation and return appropriate result
    pub async fn detect_fork(&self) -> HomologicalResult {
        let state = self.compute_homology().await;

        if state.h0_dimension == 0 {
            return HomologicalResult::InsufficientData {
                reason: "No blocks available for analysis".to_string(),
            };
        }

        // Check for network partition first (most critical)
        if state.has_partition() {
            error!(
                "🚨 [HOMOLOGY] NETWORK PARTITION DETECTED! {} disconnected components",
                state.h0_dimension
            );
            return HomologicalResult::NetworkPartition {
                components: state.h0_dimension,
                tips_per_component: vec![state.chain_tips.len()], // Simplified
            };
        }

        // Check for competing forks
        if state.has_forks() {
            warn!(
                "⚠️ [HOMOLOGY] {} competing forks detected with {} tips",
                state.h1_dimension,
                state.chain_tips.len()
            );
            return HomologicalResult::CompetingForks {
                fork_count: state.h1_dimension,
                tips: state.chain_tips.clone(),
                common_ancestor_height: Some(state.height_range.0),
            };
        }

        // Healthy topology
        let tip = state.chain_tips.iter().next().copied().unwrap_or([0u8; 32]);
        HomologicalResult::Healthy {
            height: state.height_range.1,
            tip,
        }
    }

    /// Check if homology indicates healthy network state
    pub async fn is_healthy(&self) -> bool {
        let state = self.compute_homology().await;
        state.is_valid()
    }

    /// Get current tip count
    pub async fn tip_count(&self) -> usize {
        self.tips.read().await.len()
    }

    /// Get all current tips
    pub async fn get_tips(&self) -> HashSet<[u8; 32]> {
        self.tips.read().await.clone()
    }

    /// Get cached homological state (if available)
    pub async fn get_cached_state(&self) -> Option<HomologicalState> {
        self.cached_state.read().await.clone()
    }

    /// Get statistics for monitoring
    pub async fn get_stats(&self) -> HomologicalStats {
        let blocks = self.blocks.read().await;
        let tips = self.tips.read().await;
        let state = self.cached_state.read().await.clone().unwrap_or_default();

        HomologicalStats {
            total_blocks: blocks.len(),
            tip_count: tips.len(),
            h0_dimension: state.h0_dimension,
            h1_dimension: state.h1_dimension,
            is_healthy: state.is_valid(),
            height_range: state.height_range,
        }
    }
}

/// Statistics for monitoring
#[derive(Debug, Clone, Default)]
pub struct HomologicalStats {
    pub total_blocks: usize,
    pub tip_count: usize,
    pub h0_dimension: usize,
    pub h1_dimension: usize,
    pub is_healthy: bool,
    pub height_range: (u64, u64),
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_block(hash: u8, height: u64, parent: u8) -> BlockNode {
        let mut h = [0u8; 32];
        h[0] = hash;
        let mut p = [0u8; 32];
        p[0] = parent;
        BlockNode {
            hash: h,
            height,
            parent_hash: p,
            timestamp: height * 1000,
        }
    }

    #[tokio::test]
    async fn test_linear_chain_healthy() {
        let analyzer = HomologicalConsensus::new(HomologicalConfig::default());

        // Add linear chain: 0 <- 1 <- 2 <- 3
        analyzer.add_block(make_block(0, 0, 255)).await; // Genesis
        analyzer.add_block(make_block(1, 1, 0)).await;
        analyzer.add_block(make_block(2, 2, 1)).await;
        analyzer.add_block(make_block(3, 3, 2)).await;

        let state = analyzer.compute_homology().await;

        assert_eq!(state.h0_dimension, 1, "Should be single component");
        assert_eq!(state.h1_dimension, 0, "Should have no forks");
        assert!(state.is_valid());
        assert!(!state.has_forks());
        assert!(!state.has_partition());
    }

    #[tokio::test]
    async fn test_fork_detection() {
        let analyzer = HomologicalConsensus::new(HomologicalConfig::default());

        // Add chain with fork at block 1:
        // 0 <- 1 <- 2
        //      └── 3 (fork)
        analyzer.add_block(make_block(0, 0, 255)).await; // Genesis
        analyzer.add_block(make_block(1, 1, 0)).await;
        analyzer.add_block(make_block(2, 2, 1)).await;
        analyzer.add_block(make_block(3, 2, 1)).await; // Fork from block 1

        let state = analyzer.compute_homology().await;

        assert_eq!(state.h0_dimension, 1, "Should be single component");
        assert_eq!(state.h1_dimension, 1, "Should detect one fork");
        assert_eq!(state.tip_count(), 2, "Should have 2 tips");
        assert!(state.has_forks());
        assert!(!state.is_valid());
    }

    #[tokio::test]
    async fn test_detect_fork_result() {
        let analyzer = HomologicalConsensus::new(HomologicalConfig::default());

        // Create fork
        analyzer.add_block(make_block(0, 0, 255)).await;
        analyzer.add_block(make_block(1, 1, 0)).await;
        analyzer.add_block(make_block(2, 2, 1)).await;
        analyzer.add_block(make_block(3, 2, 1)).await; // Fork

        let result = analyzer.detect_fork().await;

        match result {
            HomologicalResult::CompetingForks { fork_count, tips, .. } => {
                assert_eq!(fork_count, 1);
                assert_eq!(tips.len(), 2);
            }
            _ => panic!("Expected CompetingForks, got {:?}", result),
        }
    }

    #[tokio::test]
    async fn test_healthy_chain_result() {
        let analyzer = HomologicalConsensus::new(HomologicalConfig::default());

        // Linear chain
        analyzer.add_block(make_block(0, 0, 255)).await;
        analyzer.add_block(make_block(1, 1, 0)).await;
        analyzer.add_block(make_block(2, 2, 1)).await;

        let result = analyzer.detect_fork().await;

        match result {
            HomologicalResult::Healthy { height, .. } => {
                assert_eq!(height, 2);
            }
            _ => panic!("Expected Healthy, got {:?}", result),
        }

        assert!(analyzer.is_healthy().await);
    }

    #[tokio::test]
    async fn test_multiple_forks() {
        let analyzer = HomologicalConsensus::new(HomologicalConfig::default());

        // Multiple forks from block 1:
        // 0 <- 1 <- 2
        //      ├── 3
        //      └── 4
        analyzer.add_block(make_block(0, 0, 255)).await;
        analyzer.add_block(make_block(1, 1, 0)).await;
        analyzer.add_block(make_block(2, 2, 1)).await;
        analyzer.add_block(make_block(3, 2, 1)).await;
        analyzer.add_block(make_block(4, 2, 1)).await;

        let state = analyzer.compute_homology().await;

        assert_eq!(state.h0_dimension, 1);
        assert_eq!(state.h1_dimension, 2, "Should detect 2 forks (3 tips - 1)");
        assert_eq!(state.tip_count(), 3);
    }

    #[tokio::test]
    async fn test_stats() {
        let analyzer = HomologicalConsensus::new(HomologicalConfig::default());

        analyzer.add_block(make_block(0, 0, 255)).await;
        analyzer.add_block(make_block(1, 1, 0)).await;
        analyzer.add_block(make_block(2, 2, 1)).await;

        // Force computation to populate cache
        let _ = analyzer.compute_homology().await;

        let stats = analyzer.get_stats().await;

        assert_eq!(stats.total_blocks, 3);
        assert_eq!(stats.tip_count, 1);
        assert!(stats.is_healthy);
        assert_eq!(stats.height_range, (0, 2));
    }
}
