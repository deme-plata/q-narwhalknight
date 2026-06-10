/// Causal Ordering Validator for Phase 2 DAG-Aware Sync (v1.0.4-beta)
///
/// Enforces causal ordering of blocks using dag_parents relationships.
/// Ensures that no block is processed before all its parents are committed.
///
/// Key Design Decisions (from expert feedback):
/// - Maintains processed_blocks set for O(1) parent lookups
/// - Validates entire DAG layers atomically (all-or-nothing)
/// - Bounded memory via periodic pruning of old processed blocks
/// - Detects cycles and invalid DAG structures
///
/// Security Properties:
/// - Rejects blocks with missing parents (prevents orphans)
/// - Detects circular dependencies (cycle detection)
/// - Enforces topological ordering (causality preservation)

use anyhow::{Context, Result};
use q_types::QBlock as Block;
use std::collections::{HashSet, VecDeque};
use tracing::{debug, info, warn};

/// Causal ordering validator for DAG-aware sync
pub struct CausalValidator {
    /// Set of blocks we've already processed (hash → marker)
    /// Bounded to prevent unbounded memory growth
    processed_blocks: HashSet<String>,

    /// Maximum size of processed_blocks before pruning
    max_processed_blocks: usize,

    /// FIFO queue for pruning old blocks (maintains insertion order)
    processing_order: VecDeque<String>,
}

/// Default maximum processed blocks (10M blocks = ~320MB at 32 bytes/hash)
const DEFAULT_MAX_PROCESSED: usize = 10_000_000;

impl CausalValidator {
    /// Create new causal validator
    pub fn new() -> Self {
        Self {
            processed_blocks: HashSet::new(),
            max_processed_blocks: DEFAULT_MAX_PROCESSED,
            processing_order: VecDeque::new(),
        }
    }

    /// Create validator with custom memory limit
    pub fn with_limit(max_processed_blocks: usize) -> Self {
        Self {
            processed_blocks: HashSet::new(),
            max_processed_blocks,
            processing_order: VecDeque::new(),
        }
    }

    /// Verify all dag_parents exist before processing block
    ///
    /// This is the core validation method. Call before inserting any block
    /// into the database to ensure causal ordering is preserved.
    ///
    /// # Returns
    /// * `Ok(())` - All parents are processed, safe to commit
    /// * `Err(_)` - At least one parent is missing, DO NOT commit
    pub fn validate_dependencies(&self, block: &Block) -> Result<()> {
        // Genesis block has no parents
        if block.header.height == 0 {
            return Ok(());
        }

        // Check all dag_parents are in processed set
        // Convert VertexId to string for comparison
        for parent_vertex in &block.dag_parents {
            let parent_hash = hex::encode(parent_vertex);
            if !self.processed_blocks.contains(&parent_hash) {
                let hash = hex::encode(block.calculate_hash());
                return Err(anyhow::anyhow!(
                    "Block {} (height {}) depends on unprocessed parent {}",
                    hash,
                    block.header.height,
                    parent_hash
                ));
            }
        }

        Ok(())
    }

    /// Mark block as processed
    ///
    /// Call this AFTER successfully committing a block to the database.
    /// This allows future children to pass validation.
    pub fn mark_processed(&mut self, block_hash: String) {
        // Add to set
        if self.processed_blocks.insert(block_hash.clone()) {
            // New entry - add to FIFO queue
            self.processing_order.push_back(block_hash.clone());

            // Prune if we exceed limit
            if self.processed_blocks.len() > self.max_processed_blocks {
                if let Some(oldest) = self.processing_order.pop_front() {
                    self.processed_blocks.remove(&oldest);
                    debug!("🗑️  Pruned oldest processed block: {}", oldest);
                }
            }
        }
    }

    /// Validate and mark an entire DAG layer
    ///
    /// This is the main entry point for Phase 2 sync.
    /// Validates all blocks in a layer (in parallel if needed) and marks
    /// them as processed atomically.
    ///
    /// # Returns
    /// * `Ok(Vec<String>)` - List of validated block hashes
    /// * `Err(_)` - At least one block failed validation
    pub fn validate_layer(&mut self, blocks: &[Block]) -> Result<Vec<String>> {
        let mut validated = Vec::new();

        // Phase 1: Validate all blocks in layer
        for block in blocks {
            let hash = hex::encode(block.calculate_hash());
            self.validate_dependencies(block).with_context(|| {
                format!(
                    "Causal validation failed for block {} (height {})",
                    hash,
                    block.header.height
                )
            })?;

            validated.push(hash);
        }

        // Phase 2: Mark all as processed (atomic commit)
        // Only do this if ALL blocks passed validation
        for block_hash in &validated {
            self.mark_processed(block_hash.clone());
        }

        debug!(
            "✅ Validated and marked {} blocks in layer",
            validated.len()
        );

        Ok(validated)
    }

    /// Check if a block has been processed
    pub fn is_processed(&self, block_hash: &str) -> bool {
        self.processed_blocks.contains(block_hash)
    }

    /// Get number of processed blocks
    pub fn processed_count(&self) -> usize {
        self.processed_blocks.len()
    }

    /// Detect cycles in DAG structure
    ///
    /// This is an advanced security feature to detect malicious peers
    /// sending blocks with circular dependencies.
    ///
    /// # Returns
    /// * `Ok(())` - No cycle detected
    /// * `Err(_)` - Cycle found, reject entire batch
    pub fn detect_cycle(&self, blocks: &[Block]) -> Result<()> {
        // Build adjacency list for blocks in this batch
        let mut graph: std::collections::HashMap<String, Vec<String>> =
            std::collections::HashMap::new();

        for block in blocks {
            let hash = hex::encode(block.calculate_hash());
            let parent_hashes: Vec<String> = block.dag_parents
                .iter()
                .map(|v| hex::encode(v))
                .collect();
            graph
                .entry(hash.clone())
                .or_default()
                .extend(parent_hashes);
        }

        // DFS-based cycle detection
        let mut visited = HashSet::new();
        let mut rec_stack = HashSet::new();

        for block in blocks {
            let hash = hex::encode(block.calculate_hash());
            if !visited.contains(&hash) {
                if self.has_cycle_dfs(&hash, &graph, &mut visited, &mut rec_stack) {
                    return Err(anyhow::anyhow!(
                        "Cycle detected in DAG structure starting at block {}",
                        hash
                    ));
                }
            }
        }

        Ok(())
    }

    /// Clear all state (for testing or reset)
    pub fn reset(&mut self) {
        self.processed_blocks.clear();
        self.processing_order.clear();
    }

    // ========== Internal Methods ==========

    /// DFS-based cycle detection helper
    fn has_cycle_dfs(
        &self,
        node: &str,
        graph: &std::collections::HashMap<String, Vec<String>>,
        visited: &mut HashSet<String>,
        rec_stack: &mut HashSet<String>,
    ) -> bool {
        visited.insert(node.to_string());
        rec_stack.insert(node.to_string());

        if let Some(neighbors) = graph.get(node) {
            for neighbor in neighbors {
                if !visited.contains(neighbor) {
                    if self.has_cycle_dfs(neighbor, graph, visited, rec_stack) {
                        return true;
                    }
                } else if rec_stack.contains(neighbor) {
                    // Back edge found - cycle detected
                    return true;
                }
            }
        }

        rec_stack.remove(node);
        false
    }
}

impl Default for CausalValidator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use q_types::{BlockHeader, QBlock};

    fn make_block(hash: &str, height: u64, parents: Vec<&str>) -> Block {
        let mut block = QBlock::default();
        block.header.height = height;
        block.header.dag_parents = parents.iter().map(|s| s.to_string()).collect();

        // Create a Block wrapper (assuming Block is an alias or wrapper for QBlock)
        // For testing, we'll use QBlock directly
        block
    }

    fn block_hash(block: &QBlock) -> String {
        // Simple hash for testing
        format!("hash_{}", block.header.height)
    }

    #[test]
    fn test_genesis_block() {
        let validator = CausalValidator::new();

        let genesis = make_block("genesis", 0, vec![]);
        assert!(validator.validate_dependencies(&genesis).is_ok());
    }

    #[test]
    fn test_valid_chain() {
        let mut validator = CausalValidator::new();

        // Process genesis
        let b0 = make_block("b0", 0, vec![]);
        validator.mark_processed("b0".to_string());

        // Process b1 (parent: b0)
        let mut b1 = make_block("b1", 1, vec!["b0"]);
        assert!(validator.validate_dependencies(&b1).is_ok());
        validator.mark_processed("b1".to_string());

        // Process b2 (parent: b1)
        let b2 = make_block("b2", 2, vec!["b1"]);
        assert!(validator.validate_dependencies(&b2).is_ok());
    }

    #[test]
    fn test_missing_parent() {
        let validator = CausalValidator::new();

        // Try to process block without processing its parent
        let b1 = make_block("b1", 1, vec!["b0"]);
        assert!(validator.validate_dependencies(&b1).is_err());
    }

    #[test]
    fn test_parallel_blocks_same_layer() {
        let mut validator = CausalValidator::new();

        // Process genesis
        validator.mark_processed("b0".to_string());

        // Process multiple blocks in parallel (all depend on b0)
        let b1a = make_block("b1a", 1, vec!["b0"]);
        let b1b = make_block("b1b", 1, vec!["b0"]);
        let b1c = make_block("b1c", 1, vec!["b0"]);

        assert!(validator.validate_dependencies(&b1a).is_ok());
        assert!(validator.validate_dependencies(&b1b).is_ok());
        assert!(validator.validate_dependencies(&b1c).is_ok());

        validator.mark_processed("b1a".to_string());
        validator.mark_processed("b1b".to_string());
        validator.mark_processed("b1c".to_string());

        // Now process block that depends on all of them
        let b2 = make_block("b2", 2, vec!["b1a", "b1b", "b1c"]);
        assert!(validator.validate_dependencies(&b2).is_ok());
    }

    #[test]
    fn test_validate_layer() {
        let mut validator = CausalValidator::new();

        // Setup: mark genesis as processed
        validator.mark_processed("b0".to_string());

        // Layer 1: Multiple blocks depending on b0
        let layer1 = vec![
            make_block("b1a", 1, vec!["b0"]),
            make_block("b1b", 1, vec!["b0"]),
            make_block("b1c", 1, vec!["b0"]),
        ];

        let result = validator.validate_layer(&layer1);
        assert!(result.is_ok());

        let validated = result.unwrap();
        assert_eq!(validated.len(), 3);

        // All blocks should now be marked as processed
        assert!(validator.is_processed("b1a"));
        assert!(validator.is_processed("b1b"));
        assert!(validator.is_processed("b1c"));
    }

    #[test]
    fn test_validate_layer_with_failure() {
        let mut validator = CausalValidator::new();

        // Setup: mark genesis as processed
        validator.mark_processed("b0".to_string());

        // Layer with one invalid block (missing parent)
        let layer = vec![
            make_block("b1a", 1, vec!["b0"]),
            make_block("b1b", 1, vec!["nonexistent"]), // Invalid!
        ];

        let result = validator.validate_layer(&layer);
        assert!(result.is_err());

        // No blocks should be marked as processed (atomic failure)
        assert!(!validator.is_processed("b1a"));
        assert!(!validator.is_processed("b1b"));
    }

    #[test]
    fn test_memory_pruning() {
        let mut validator = CausalValidator::with_limit(100);

        // Add 150 blocks
        for i in 0..150 {
            validator.mark_processed(format!("block_{}", i));
        }

        // Should be capped at 100
        assert_eq!(validator.processed_count(), 100);

        // Oldest blocks should be pruned
        assert!(!validator.is_processed("block_0"));
        assert!(!validator.is_processed("block_49"));

        // Newest blocks should be retained
        assert!(validator.is_processed("block_149"));
        assert!(validator.is_processed("block_100"));
    }

    #[test]
    fn test_cycle_detection() {
        let validator = CausalValidator::new();

        // Create a cycle: b1 → b2 → b3 → b1
        let mut b1 = make_block("b1", 1, vec!["b3"]);
        let mut b2 = make_block("b2", 2, vec!["b1"]);
        let mut b3 = make_block("b3", 3, vec!["b2"]);

        let blocks = vec![b1, b2, b3];

        let result = validator.detect_cycle(&blocks);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Cycle detected"));
    }

    #[test]
    fn test_no_cycle() {
        let validator = CausalValidator::new();

        // Valid DAG: b1 → b2, b1 → b3, b2 → b4, b3 → b4
        let blocks = vec![
            make_block("b1", 1, vec![]),
            make_block("b2", 2, vec!["b1"]),
            make_block("b3", 3, vec!["b1"]),
            make_block("b4", 4, vec!["b2", "b3"]),
        ];

        let result = validator.detect_cycle(&blocks);
        assert!(result.is_ok());
    }
}
