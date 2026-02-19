//! Cryptographic phase definitions and height-gated algorithm selection.
//!
//! Q-NarwhalKnight transitions through four cryptographic eras, each activated
//! at a fixed block height.  This module defines the [`CryptoPhase`] enum and
//! the deterministic selection function that maps any block height to the
//! algorithm suite that was (or will be) active at that height.
//!
//! ## Phase Timeline
//!
//! | Phase | Heights | Algorithms |
//! |-------|---------|------------|
//! | Phase0_Genesis | 0 -- 999,999 | Ed25519 classical signatures |
//! | Phase1_Hybrid | 1,000,000 -- 2,499,999 | Ed25519 + SQIsign dual signatures |
//! | Phase2_PurePostQuantum | 2,500,000 -- 3,999,999 | SQIsign Level III only |
//! | Phase3_ThresholdGuardian | 4,000,000+ | FROST threshold + SQIsign |
//!
//! The activation heights are compile-time constants.  Any change to these
//! values constitutes a consensus-breaking upgrade and MUST be gated behind
//! the upgrade-gate mechanism in `q-consensus-guard`.

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Activation heights
// ---------------------------------------------------------------------------

/// Block height at which Phase 1 (Hybrid) activates.
pub const PHASE1_ACTIVATION_HEIGHT: u64 = 1_000_000;

/// Block height at which Phase 2 (Pure Post-Quantum) activates.
pub const PHASE2_ACTIVATION_HEIGHT: u64 = 2_500_000;

/// Block height at which Phase 3 (Threshold Guardian) activates.
pub const PHASE3_ACTIVATION_HEIGHT: u64 = 4_000_000;

// ---------------------------------------------------------------------------
// CryptoPhase enum
// ---------------------------------------------------------------------------

/// The cryptographic era of the network, determined by block height.
///
/// Each phase defines which signature algorithms are valid for block and
/// transaction signatures.  Verifiers MUST use this enum to select the
/// correct verification logic when replaying historical blocks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[allow(non_camel_case_types)]
pub enum CryptoPhase {
    /// Genesis era: classical Ed25519 signatures.
    ///
    /// Active from block 0 through `PHASE1_ACTIVATION_HEIGHT - 1`.
    /// All keys are standard 32-byte Ed25519 public keys.
    Phase0_Genesis,

    /// Hybrid era: Ed25519 + SQIsign dual signatures.
    ///
    /// Active from `PHASE1_ACTIVATION_HEIGHT` through `PHASE2_ACTIVATION_HEIGHT - 1`.
    /// Blocks carry both a classical and a post-quantum signature.  Verifiers
    /// must check BOTH signatures.  This phase allows the network to build
    /// confidence in the isogeny-based scheme while retaining a classical
    /// fallback.
    Phase1_Hybrid,

    /// Pure post-quantum era: SQIsign Level III only.
    ///
    /// Active from `PHASE2_ACTIVATION_HEIGHT` through `PHASE3_ACTIVATION_HEIGHT - 1`.
    /// Ed25519 signatures are no longer accepted for new blocks.  Historical
    /// blocks from earlier phases still verify under their original rules.
    Phase2_PurePostQuantum,

    /// Threshold guardian era: FROST threshold signatures + SQIsign.
    ///
    /// Active from `PHASE3_ACTIVATION_HEIGHT` onward.
    /// Block production requires a t-of-n threshold signature from the
    /// validator committee, with each share itself being a SQIsign signature.
    /// This provides both post-quantum security and distributed trust.
    Phase3_ThresholdGuardian,
}

impl CryptoPhase {
    /// Determine the active cryptographic phase for a given block height.
    ///
    /// This function is pure and deterministic: the same height always maps
    /// to the same phase, regardless of node state or configuration.
    ///
    /// # Examples
    ///
    /// ```
    /// use q_eternal_cypher::CryptoPhase;
    ///
    /// assert_eq!(CryptoPhase::select_algorithm(0), CryptoPhase::Phase0_Genesis);
    /// assert_eq!(CryptoPhase::select_algorithm(999_999), CryptoPhase::Phase0_Genesis);
    /// assert_eq!(CryptoPhase::select_algorithm(1_000_000), CryptoPhase::Phase1_Hybrid);
    /// assert_eq!(CryptoPhase::select_algorithm(2_500_000), CryptoPhase::Phase2_PurePostQuantum);
    /// assert_eq!(CryptoPhase::select_algorithm(4_000_000), CryptoPhase::Phase3_ThresholdGuardian);
    /// assert_eq!(CryptoPhase::select_algorithm(u64::MAX), CryptoPhase::Phase3_ThresholdGuardian);
    /// ```
    pub fn select_algorithm(height: u64) -> Self {
        if height >= PHASE3_ACTIVATION_HEIGHT {
            CryptoPhase::Phase3_ThresholdGuardian
        } else if height >= PHASE2_ACTIVATION_HEIGHT {
            CryptoPhase::Phase2_PurePostQuantum
        } else if height >= PHASE1_ACTIVATION_HEIGHT {
            CryptoPhase::Phase1_Hybrid
        } else {
            CryptoPhase::Phase0_Genesis
        }
    }

    /// Return a human-readable label for the phase.
    pub fn label(&self) -> &'static str {
        match self {
            CryptoPhase::Phase0_Genesis => "Phase 0: Genesis (Ed25519)",
            CryptoPhase::Phase1_Hybrid => "Phase 1: Hybrid (Ed25519 + SQIsign)",
            CryptoPhase::Phase2_PurePostQuantum => "Phase 2: Pure Post-Quantum (SQIsign III)",
            CryptoPhase::Phase3_ThresholdGuardian => "Phase 3: Threshold Guardian (FROST + SQIsign)",
        }
    }

    /// Return `true` if Ed25519 signatures are accepted in this phase.
    pub fn accepts_ed25519(&self) -> bool {
        matches!(
            self,
            CryptoPhase::Phase0_Genesis | CryptoPhase::Phase1_Hybrid
        )
    }

    /// Return `true` if SQIsign signatures are accepted in this phase.
    pub fn accepts_sqisign(&self) -> bool {
        matches!(
            self,
            CryptoPhase::Phase1_Hybrid
                | CryptoPhase::Phase2_PurePostQuantum
                | CryptoPhase::Phase3_ThresholdGuardian
        )
    }

    /// Return `true` if FROST threshold signatures are accepted in this phase.
    pub fn accepts_threshold(&self) -> bool {
        matches!(self, CryptoPhase::Phase3_ThresholdGuardian)
    }

    /// Return `true` if this phase requires dual (hybrid) signatures.
    pub fn requires_dual_signature(&self) -> bool {
        matches!(self, CryptoPhase::Phase1_Hybrid)
    }
}

impl Default for CryptoPhase {
    fn default() -> Self {
        CryptoPhase::Phase0_Genesis
    }
}

impl std::fmt::Display for CryptoPhase {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.label())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_phase_boundaries() {
        // Genesis phase
        assert_eq!(CryptoPhase::select_algorithm(0), CryptoPhase::Phase0_Genesis);
        assert_eq!(
            CryptoPhase::select_algorithm(PHASE1_ACTIVATION_HEIGHT - 1),
            CryptoPhase::Phase0_Genesis
        );

        // Hybrid phase
        assert_eq!(
            CryptoPhase::select_algorithm(PHASE1_ACTIVATION_HEIGHT),
            CryptoPhase::Phase1_Hybrid
        );
        assert_eq!(
            CryptoPhase::select_algorithm(PHASE2_ACTIVATION_HEIGHT - 1),
            CryptoPhase::Phase1_Hybrid
        );

        // Pure PQ phase
        assert_eq!(
            CryptoPhase::select_algorithm(PHASE2_ACTIVATION_HEIGHT),
            CryptoPhase::Phase2_PurePostQuantum
        );
        assert_eq!(
            CryptoPhase::select_algorithm(PHASE3_ACTIVATION_HEIGHT - 1),
            CryptoPhase::Phase2_PurePostQuantum
        );

        // Threshold guardian phase
        assert_eq!(
            CryptoPhase::select_algorithm(PHASE3_ACTIVATION_HEIGHT),
            CryptoPhase::Phase3_ThresholdGuardian
        );
        assert_eq!(
            CryptoPhase::select_algorithm(u64::MAX),
            CryptoPhase::Phase3_ThresholdGuardian
        );
    }

    #[test]
    fn test_algorithm_acceptance() {
        let p0 = CryptoPhase::Phase0_Genesis;
        assert!(p0.accepts_ed25519());
        assert!(!p0.accepts_sqisign());
        assert!(!p0.accepts_threshold());
        assert!(!p0.requires_dual_signature());

        let p1 = CryptoPhase::Phase1_Hybrid;
        assert!(p1.accepts_ed25519());
        assert!(p1.accepts_sqisign());
        assert!(!p1.accepts_threshold());
        assert!(p1.requires_dual_signature());

        let p2 = CryptoPhase::Phase2_PurePostQuantum;
        assert!(!p2.accepts_ed25519());
        assert!(p2.accepts_sqisign());
        assert!(!p2.accepts_threshold());
        assert!(!p2.requires_dual_signature());

        let p3 = CryptoPhase::Phase3_ThresholdGuardian;
        assert!(!p3.accepts_ed25519());
        assert!(p3.accepts_sqisign());
        assert!(p3.accepts_threshold());
        assert!(!p3.requires_dual_signature());
    }

    #[test]
    fn test_display_and_label() {
        for phase in [
            CryptoPhase::Phase0_Genesis,
            CryptoPhase::Phase1_Hybrid,
            CryptoPhase::Phase2_PurePostQuantum,
            CryptoPhase::Phase3_ThresholdGuardian,
        ] {
            let display = format!("{}", phase);
            assert!(!display.is_empty());
            assert_eq!(display, phase.label());
        }
    }

    #[test]
    fn test_default_is_genesis() {
        assert_eq!(CryptoPhase::default(), CryptoPhase::Phase0_Genesis);
    }

    #[test]
    fn test_serde_roundtrip() {
        for phase in [
            CryptoPhase::Phase0_Genesis,
            CryptoPhase::Phase1_Hybrid,
            CryptoPhase::Phase2_PurePostQuantum,
            CryptoPhase::Phase3_ThresholdGuardian,
        ] {
            let json = serde_json::to_string(&phase).unwrap();
            let recovered: CryptoPhase = serde_json::from_str(&json).unwrap();
            assert_eq!(phase, recovered);
        }
    }

    #[test]
    fn test_activation_height_ordering() {
        assert!(PHASE1_ACTIVATION_HEIGHT < PHASE2_ACTIVATION_HEIGHT);
        assert!(PHASE2_ACTIVATION_HEIGHT < PHASE3_ACTIVATION_HEIGHT);
    }
}
