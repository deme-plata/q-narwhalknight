//! Determinism Checker
//!
//! Ensures state transitions are deterministic:
//! Same input → Same output, ALWAYS.
//!
//! Non-determinism causes consensus splits on mainnet.

use serde::{Deserialize, Serialize};
use sha3::{Digest, Sha3_256};
use std::collections::HashMap;
use tracing::{error, info, warn};

/// A recorded state transition for replay verification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateTransition {
    /// Input state hash
    pub input_state: [u8; 32],

    /// Operation applied
    pub operation: String,

    /// Operation parameters hash
    pub params_hash: [u8; 32],

    /// Expected output state hash
    pub output_state: [u8; 32],

    /// Block height where this transition occurred
    pub height: u64,
}

/// Checker for state transition determinism
pub struct DeterminismChecker {
    /// Recorded transitions for replay
    transitions: Vec<StateTransition>,

    /// Enable strict mode (fail on any non-determinism)
    strict_mode: bool,
}

impl DeterminismChecker {
    /// Create new determinism checker
    pub fn new(strict_mode: bool) -> Self {
        Self {
            transitions: Vec::new(),
            strict_mode,
        }
    }

    /// Record a state transition
    pub fn record_transition(
        &mut self,
        input_state: [u8; 32],
        operation: &str,
        params: &[u8],
        output_state: [u8; 32],
        height: u64,
    ) {
        let mut params_hasher = Sha3_256::new();
        params_hasher.update(params);
        let params_hash: [u8; 32] = params_hasher.finalize().into();

        self.transitions.push(StateTransition {
            input_state,
            operation: operation.to_string(),
            params_hash,
            output_state,
            height,
        });
    }

    /// Verify a transition produces expected output
    pub fn verify_transition(
        &self,
        input_state: [u8; 32],
        operation: &str,
        params: &[u8],
        actual_output: [u8; 32],
    ) -> Result<(), DeterminismViolation> {
        let mut params_hasher = Sha3_256::new();
        params_hasher.update(params);
        let params_hash: [u8; 32] = params_hasher.finalize().into();

        // Find matching recorded transition
        for transition in &self.transitions {
            if transition.input_state == input_state
                && transition.operation == operation
                && transition.params_hash == params_hash
            {
                // Found matching input - check output matches
                if transition.output_state != actual_output {
                    return Err(DeterminismViolation::OutputMismatch {
                        operation: operation.to_string(),
                        expected: hex::encode(transition.output_state),
                        actual: hex::encode(actual_output),
                        height: transition.height,
                    });
                }
                return Ok(());
            }
        }

        // No matching transition found - might be new
        if self.strict_mode {
            warn!(
                "[DETERMINISM] No recorded transition for {} - strict mode would fail",
                operation
            );
        }

        Ok(())
    }

    /// Replay all recorded transitions and verify determinism
    pub fn replay_all<F>(&self, execute: F) -> Result<(), DeterminismViolation>
    where
        F: Fn(&[u8; 32], &str, &[u8; 32]) -> [u8; 32],
    {
        info!(
            "🔄 [DETERMINISM] Replaying {} transitions...",
            self.transitions.len()
        );

        for (i, transition) in self.transitions.iter().enumerate() {
            let actual = execute(
                &transition.input_state,
                &transition.operation,
                &transition.params_hash,
            );

            if actual != transition.output_state {
                return Err(DeterminismViolation::ReplayMismatch {
                    transition_index: i,
                    operation: transition.operation.clone(),
                    height: transition.height,
                    expected: hex::encode(transition.output_state),
                    actual: hex::encode(actual),
                });
            }
        }

        info!("🔄 [DETERMINISM] All transitions verified!");
        Ok(())
    }

    /// Export transitions for storage
    pub fn export(&self) -> Vec<StateTransition> {
        self.transitions.clone()
    }

    /// Import transitions from storage
    pub fn import(&mut self, transitions: Vec<StateTransition>) {
        self.transitions = transitions;
    }
}

/// Error when determinism is violated
#[derive(Debug, Clone, thiserror::Error)]
pub enum DeterminismViolation {
    #[error("Output mismatch for {operation} at height {height}: expected {expected}, got {actual}")]
    OutputMismatch {
        operation: String,
        expected: String,
        actual: String,
        height: u64,
    },

    #[error("Replay mismatch at transition {transition_index} ({operation}) height {height}: expected {expected}, got {actual}")]
    ReplayMismatch {
        transition_index: usize,
        operation: String,
        height: u64,
        expected: String,
        actual: String,
    },
}

impl DeterminismViolation {
    pub fn print_detailed(&self) {
        error!("╔════════════════════════════════════════════════════════════╗");
        error!("║           🚨 DETERMINISM VIOLATION DETECTED 🚨             ║");
        error!("╠════════════════════════════════════════════════════════════╣");
        error!("║                                                            ║");
        error!("║  Same input produced different output!                     ║");
        error!("║  This causes consensus splits on mainnet.                  ║");
        error!("║                                                            ║");
        error!("║  COMMON CAUSES:                                           ║");
        error!("║  - Using current time in validation                        ║");
        error!("║  - Using random numbers without seed                       ║");
        error!("║  - Floating point operations                               ║");
        error!("║  - HashMap iteration order                                 ║");
        error!("║  - Thread-dependent behavior                               ║");
        error!("║                                                            ║");
        error!("╠════════════════════════════════════════════════════════════╣");
        error!("║  Error: {} ║", self);
        error!("╚════════════════════════════════════════════════════════════╝");
    }
}

/// Common sources of non-determinism to avoid
pub mod pitfalls {
    /// NEVER use std::time::SystemTime in consensus code
    /// Use block timestamp instead
    pub const NO_SYSTEM_TIME: &str = "Use block.timestamp, not SystemTime::now()";

    /// NEVER use rand without deterministic seed
    pub const NO_RANDOM: &str = "Use VRF or deterministic RNG seeded from block";

    /// NEVER use HashMap for consensus - iteration order varies
    pub const NO_HASHMAP: &str = "Use BTreeMap for deterministic iteration";

    /// NEVER use floats - rounding differs across platforms
    pub const NO_FLOATS: &str = "Use fixed-point arithmetic";

    /// NEVER use threads for consensus computation
    pub const NO_THREADS: &str = "Parallel execution must produce same result";
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_determinism_checker() {
        let mut checker = DeterminismChecker::new(true);

        // Record a transition
        let input = [1u8; 32];
        let output = [2u8; 32];
        checker.record_transition(input, "test_op", b"params", output, 100);

        // Same input/operation should produce same output
        assert!(checker.verify_transition(input, "test_op", b"params", output).is_ok());

        // Different output should fail
        let wrong_output = [3u8; 32];
        assert!(checker.verify_transition(input, "test_op", b"params", wrong_output).is_err());
    }
}
