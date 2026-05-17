//! 🌌 Many-Worlds Interpretation Implementation
//! Branch-ID addressing via quantum measurement phase fingerprinting
//! Enables water robots to track and navigate quantum measurement branches

use rand::Rng;
use serde::{Deserialize, Serialize};
use sha3::{Digest, Sha3_256};
use std::collections::HashMap;
use std::f64::consts::{PI, TAU};

/// Branch identifier for Many-Worlds quantum measurement branches
pub type BranchId = [u8; 32];

/// Quantum measurement branch tracking
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct QuantumBranch {
    /// Unique branch identifier from phase fingerprint
    pub branch_id: BranchId,
    /// Quantum measurement that created this branch
    pub measurement_vector: MeasurementVector,
    /// Parent branch (None for initial branch)
    pub parent_branch: Option<BranchId>,
    /// Child branches spawned from this one
    pub child_branches: Vec<BranchId>,
    /// Branch probability amplitude
    pub amplitude: f64,
    /// Phase fingerprint for addressing
    pub phase_fingerprint: PhaseFingerprint,
    /// Branch creation timestamp (attoseconds)
    pub created_at: u64,
    /// Branch coherence measure (0..1)
    pub coherence: f64,
}

/// Quantum measurement vector in Hilbert space
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MeasurementVector {
    /// Measurement basis states
    pub basis_states: Vec<ComplexAmplitude>,
    /// Observable being measured
    pub observable: String,
    /// Measurement result
    pub eigenvalue: f64,
    /// Measurement uncertainty
    pub uncertainty: f64,
}

/// Complex amplitude for quantum superposition
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ComplexAmplitude {
    pub real: f64,
    pub imag: f64,
}

impl ComplexAmplitude {
    pub fn new(real: f64, imag: f64) -> Self {
        Self { real, imag }
    }

    pub fn magnitude_squared(&self) -> f64 {
        self.real * self.real + self.imag * self.imag
    }

    pub fn phase(&self) -> f64 {
        self.imag.atan2(self.real)
    }
}

/// Phase fingerprint for Many-Worlds branch addressing
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PhaseFingerprint {
    /// Primary phase components (0..2π each)
    pub phases: [f64; 8],
    /// Phase correlation matrix
    pub correlations: [[f64; 8]; 8],
    /// Fingerprint hash for quick comparison
    pub fingerprint_hash: [u8; 32],
}

impl PhaseFingerprint {
    /// Generate phase fingerprint from measurement
    pub fn from_measurement(measurement: &MeasurementVector) -> Self {
        let mut phases = [0.0; 8];
        let mut correlations = [[0.0; 8]; 8];

        // Extract phases from measurement basis states
        for (i, state) in measurement.basis_states.iter().enumerate().take(8) {
            phases[i] = state.phase();
        }

        // Calculate phase correlations
        for i in 0..8 {
            for j in 0..8 {
                if i == j {
                    correlations[i][j] = 1.0;
                } else {
                    correlations[i][j] = (phases[i] - phases[j]).cos();
                }
            }
        }

        // Generate fingerprint hash
        let mut hasher = Sha3_256::new();
        for phase in &phases {
            hasher.update(&phase.to_le_bytes());
        }
        hasher.update(&measurement.observable.as_bytes());
        hasher.update(&measurement.eigenvalue.to_le_bytes());
        let fingerprint_hash = hasher.finalize().into();

        Self {
            phases,
            correlations,
            fingerprint_hash,
        }
    }

    /// Calculate phase distance between fingerprints
    pub fn phase_distance(&self, other: &Self) -> f64 {
        self.phases
            .iter()
            .zip(other.phases.iter())
            .map(|(a, b)| {
                let diff = (a - b).abs();
                diff.min(TAU - diff)
            })
            .sum::<f64>()
            / 8.0
    }

    /// Generate branch address string
    pub fn branch_address(&self) -> String {
        format!("Branch-{}", hex::encode(&self.fingerprint_hash[..8]))
    }
}

/// Many-Worlds navigation engine
#[derive(Clone, Debug, Default)]
pub struct ManyWorldsEngine {
    /// Current branch we're operating in
    pub current_branch: BranchId,
    /// Branch genealogy tree
    pub branch_tree: HashMap<BranchId, QuantumBranch>,
    /// Quantum measurement history
    pub measurement_history: Vec<MeasurementVector>,
    /// Branch coherence tracking
    pub coherence_tracker: CoherenceTracker,
}

/// Branch coherence tracking system
#[derive(Clone, Debug, Default)]
pub struct CoherenceTracker {
    /// Branch coherence measurements over time
    pub coherence_history: HashMap<BranchId, Vec<(u64, f64)>>,
    /// Decoherence rate constants
    pub decoherence_rates: HashMap<BranchId, f64>,
    /// Minimum coherence threshold for branch access
    pub min_coherence: f64,
}

impl ManyWorldsEngine {
    /// Create new Many-Worlds engine
    pub fn new() -> Self {
        let initial_branch = Self::genesis_branch();
        let current_branch = initial_branch.branch_id;
        let mut branch_tree = HashMap::new();
        branch_tree.insert(current_branch, initial_branch);

        Self {
            current_branch,
            branch_tree,
            measurement_history: Vec::new(),
            coherence_tracker: CoherenceTracker {
                coherence_history: HashMap::new(),
                decoherence_rates: HashMap::new(),
                min_coherence: 0.1,
            },
        }
    }

    /// Create the initial quantum branch (universe origin)
    fn genesis_branch() -> QuantumBranch {
        let measurement = MeasurementVector {
            basis_states: vec![
                ComplexAmplitude::new(1.0, 0.0), // |0⟩
                ComplexAmplitude::new(0.0, 0.0), // |1⟩
            ],
            observable: "initial_state".to_string(),
            eigenvalue: 0.0,
            uncertainty: 0.0,
        };

        let phase_fingerprint = PhaseFingerprint::from_measurement(&measurement);
        let branch_id = phase_fingerprint.fingerprint_hash;

        QuantumBranch {
            branch_id,
            measurement_vector: measurement,
            parent_branch: None,
            child_branches: Vec::new(),
            amplitude: 1.0,
            phase_fingerprint,
            created_at: Self::current_attoseconds(),
            coherence: 1.0,
        }
    }

    /// Perform quantum measurement and branch
    pub fn quantum_measurement(
        &mut self,
        observable: &str,
        basis_states: Vec<ComplexAmplitude>,
    ) -> Result<Vec<BranchId>, String> {
        let measurement = MeasurementVector {
            basis_states: basis_states.clone(),
            observable: observable.to_string(),
            eigenvalue: self.calculate_eigenvalue(&basis_states),
            uncertainty: self.calculate_uncertainty(&basis_states),
        };

        self.measurement_history.push(measurement.clone());

        // Create branches for each significant amplitude
        let mut new_branches = Vec::new();
        for (i, state) in basis_states.iter().enumerate() {
            let prob = state.magnitude_squared();
            if prob > 0.001 {
                // Only branch for significant probabilities
                let branch_id = self.create_branch(measurement.clone(), prob, i)?;
                new_branches.push(branch_id);
            }
        }

        // Update current branch with new children
        if let Some(current) = self.branch_tree.get_mut(&self.current_branch) {
            current.child_branches.extend(new_branches.iter());
        }

        Ok(new_branches)
    }

    /// Navigate to a specific quantum branch
    pub fn navigate_to_branch(&mut self, target_branch: BranchId) -> Result<(), String> {
        if !self.branch_tree.contains_key(&target_branch) {
            return Err(format!(
                "Branch {} not found in tree",
                hex::encode(&target_branch)
            ));
        }

        // Check branch coherence
        let branch = &self.branch_tree[&target_branch];
        if branch.coherence < self.coherence_tracker.min_coherence {
            return Err(format!(
                "Branch {} has insufficient coherence: {:.3}",
                hex::encode(&target_branch),
                branch.coherence
            ));
        }

        self.current_branch = target_branch;
        Ok(())
    }

    /// Get branch genealogy path from root to current
    pub fn get_branch_genealogy(&self) -> Vec<BranchId> {
        let mut path = Vec::new();
        let mut current_id = self.current_branch;

        while let Some(branch) = self.branch_tree.get(&current_id) {
            path.insert(0, current_id);
            match branch.parent_branch {
                Some(parent_id) => current_id = parent_id,
                None => break,
            }
        }

        path
    }

    /// Search for branches matching phase fingerprint pattern
    pub fn find_branches_by_phase(
        &self,
        target_fingerprint: &PhaseFingerprint,
        max_distance: f64,
    ) -> Vec<BranchId> {
        self.branch_tree
            .values()
            .filter(|branch| {
                branch.phase_fingerprint.phase_distance(target_fingerprint) <= max_distance
            })
            .map(|branch| branch.branch_id)
            .collect()
    }

    /// Create new quantum branch from measurement
    fn create_branch(
        &mut self,
        measurement: MeasurementVector,
        amplitude: f64,
        outcome_index: usize,
    ) -> Result<BranchId, String> {
        let phase_fingerprint = PhaseFingerprint::from_measurement(&measurement);
        let mut branch_id = phase_fingerprint.fingerprint_hash;

        // Make branch ID unique by incorporating outcome index
        branch_id[0] ^= outcome_index as u8;

        let new_branch = QuantumBranch {
            branch_id,
            measurement_vector: measurement,
            parent_branch: Some(self.current_branch),
            child_branches: Vec::new(),
            amplitude: amplitude.sqrt(), // Store amplitude, not probability
            phase_fingerprint,
            created_at: Self::current_attoseconds(),
            coherence: amplitude.sqrt(), // Initial coherence based on amplitude
        };

        self.branch_tree.insert(branch_id, new_branch);
        Ok(branch_id)
    }

    /// Calculate measurement eigenvalue
    fn calculate_eigenvalue(&self, basis_states: &[ComplexAmplitude]) -> f64 {
        basis_states
            .iter()
            .enumerate()
            .map(|(i, state)| (i as f64) * state.magnitude_squared())
            .sum()
    }

    /// Calculate measurement uncertainty
    fn calculate_uncertainty(&self, basis_states: &[ComplexAmplitude]) -> f64 {
        let expectation = self.calculate_eigenvalue(basis_states);
        let variance = basis_states
            .iter()
            .enumerate()
            .map(|(i, state)| {
                let diff = (i as f64) - expectation;
                diff * diff * state.magnitude_squared()
            })
            .sum::<f64>();

        variance.sqrt()
    }

    /// Update branch coherence (decoherence simulation)
    pub fn update_coherence(&mut self) {
        let current_time = Self::current_attoseconds();

        for (branch_id, branch) in self.branch_tree.iter_mut() {
            let age = current_time.saturating_sub(branch.created_at);
            let decoherence_rate = self
                .coherence_tracker
                .decoherence_rates
                .get(branch_id)
                .copied()
                .unwrap_or(1e-15); // Default decoherence rate

            // Exponential decoherence
            branch.coherence *= (-decoherence_rate * age as f64).exp();
            branch.coherence = branch.coherence.max(0.001); // Minimum coherence

            // Record coherence history
            let history = self
                .coherence_tracker
                .coherence_history
                .entry(*branch_id)
                .or_insert_with(Vec::new);
            history.push((current_time, branch.coherence));

            // Limit history size
            if history.len() > 1000 {
                history.remove(0);
            }
        }
    }

    /// Get current branch information
    pub fn current_branch_info(&self) -> Option<&QuantumBranch> {
        self.branch_tree.get(&self.current_branch)
    }

    /// Get branch statistics
    pub fn get_statistics(&self) -> ManyWorldsStats {
        ManyWorldsStats {
            total_branches: self.branch_tree.len(),
            active_branches: self
                .branch_tree
                .values()
                .filter(|b| b.coherence > self.coherence_tracker.min_coherence)
                .count(),
            current_branch: self.current_branch,
            measurement_count: self.measurement_history.len(),
            average_coherence: self.branch_tree.values().map(|b| b.coherence).sum::<f64>()
                / self.branch_tree.len() as f64,
            branch_depth: self.get_branch_genealogy().len(),
        }
    }

    /// Get current time in attoseconds
    fn current_attoseconds() -> u64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64
            / 1_000_000_000
    }
}

/// Statistics for Many-Worlds engine
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ManyWorldsStats {
    pub total_branches: usize,
    pub active_branches: usize,
    pub current_branch: BranchId,
    pub measurement_count: usize,
    pub average_coherence: f64,
    pub branch_depth: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_many_worlds_creation() {
        let engine = ManyWorldsEngine::new();
        assert_eq!(engine.branch_tree.len(), 1);
        assert!(engine.current_branch_info().is_some());
    }

    #[test]
    fn test_quantum_measurement_branching() {
        let mut engine = ManyWorldsEngine::new();

        // Simulate quantum measurement with superposition
        let basis_states = vec![
            ComplexAmplitude::new(0.707, 0.0), // |0⟩
            ComplexAmplitude::new(0.0, 0.707), // |1⟩
        ];

        let branches = engine.quantum_measurement("spin_z", basis_states).unwrap();
        assert_eq!(branches.len(), 2); // Should create 2 branches
        assert_eq!(engine.branch_tree.len(), 3); // Original + 2 new branches
    }

    #[test]
    fn test_branch_navigation() {
        let mut engine = ManyWorldsEngine::new();

        // Create branches
        let basis_states = vec![
            ComplexAmplitude::new(0.6, 0.0),
            ComplexAmplitude::new(0.8, 0.0),
        ];
        let branches = engine.quantum_measurement("test", basis_states).unwrap();

        // Navigate to first branch
        let result = engine.navigate_to_branch(branches[0]);
        assert!(result.is_ok());
        assert_eq!(engine.current_branch, branches[0]);
    }

    #[test]
    fn test_phase_fingerprint() {
        let measurement = MeasurementVector {
            basis_states: vec![
                ComplexAmplitude::new(0.707, 0.0),
                ComplexAmplitude::new(0.0, 0.707),
            ],
            observable: "test".to_string(),
            eigenvalue: 0.5,
            uncertainty: 0.5,
        };

        let fingerprint = PhaseFingerprint::from_measurement(&measurement);
        assert_eq!(fingerprint.fingerprint_hash.len(), 32);
        assert!(fingerprint.branch_address().starts_with("Branch-"));
    }

    #[test]
    fn test_coherence_tracking() {
        let mut engine = ManyWorldsEngine::new();

        // Initial coherence should be 1.0
        let initial_coherence = engine.current_branch_info().unwrap().coherence;
        assert_eq!(initial_coherence, 1.0);

        // Update coherence (simulate time passage)
        engine.update_coherence();

        // Coherence should decrease over time
        let updated_coherence = engine.current_branch_info().unwrap().coherence;
        assert!(updated_coherence <= initial_coherence);
    }
}
