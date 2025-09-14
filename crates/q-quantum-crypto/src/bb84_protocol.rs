//! 🔬 BB84 Protocol Implementation Details
//! Quantum key distribution protocol specifics

use serde::{Deserialize, Serialize};

/// BB84 protocol state
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum BB84State {
    /// Initializing protocol
    Initializing,
    /// Sending quantum states
    SendingStates,
    /// Receiving quantum states
    ReceivingStates,
    /// Basis reconciliation phase
    BasisReconciliation,
    /// Error estimation phase
    ErrorEstimation,
    /// Error correction phase
    ErrorCorrection,
    /// Privacy amplification phase
    PrivacyAmplification,
    /// Key established
    KeyEstablished,
    /// Protocol failed
    Failed,
}

/// Photon polarization for BB84
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum PhotonPolarization {
    /// Horizontal polarization (0°)
    Horizontal,
    /// Vertical polarization (90°)
    Vertical,
    /// Diagonal polarization (45°)
    Diagonal,
    /// Anti-diagonal polarization (135°)
    AntiDiagonal,
}

impl PhotonPolarization {
    /// Convert to bit value
    pub fn to_bit(self) -> bool {
        match self {
            PhotonPolarization::Horizontal | PhotonPolarization::Diagonal => false,
            PhotonPolarization::Vertical | PhotonPolarization::AntiDiagonal => true,
        }
    }

    /// Get measurement basis
    pub fn basis(self) -> MeasurementBasis {
        match self {
            PhotonPolarization::Horizontal | PhotonPolarization::Vertical => {
                MeasurementBasis::Rectilinear
            }
            PhotonPolarization::Diagonal | PhotonPolarization::AntiDiagonal => {
                MeasurementBasis::Diagonal
            }
        }
    }
}

/// Measurement basis for BB84
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum MeasurementBasis {
    /// Rectilinear basis (H/V)
    Rectilinear,
    /// Diagonal basis (+/-)
    Diagonal,
}

/// Quantum bit for BB84 protocol
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct QuantumBit {
    /// Bit value
    pub value: bool,
    /// Preparation/measurement basis
    pub basis: MeasurementBasis,
    /// Polarization used
    pub polarization: PhotonPolarization,
}

impl QuantumBit {
    /// Create new quantum bit
    pub fn new(value: bool, basis: MeasurementBasis) -> Self {
        let polarization = match (basis, value) {
            (MeasurementBasis::Rectilinear, false) => PhotonPolarization::Horizontal,
            (MeasurementBasis::Rectilinear, true) => PhotonPolarization::Vertical,
            (MeasurementBasis::Diagonal, false) => PhotonPolarization::Diagonal,
            (MeasurementBasis::Diagonal, true) => PhotonPolarization::AntiDiagonal,
        };

        Self {
            value,
            basis,
            polarization,
        }
    }

    /// Measure quantum bit with given basis
    pub fn measure(self, measurement_basis: MeasurementBasis) -> (bool, bool) {
        let correct_basis = self.basis == measurement_basis;

        if correct_basis {
            // Measurement in correct basis gives deterministic result
            (self.value, true)
        } else {
            // Measurement in wrong basis gives random result
            (rand::random::<bool>(), false)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_photon_polarization() {
        assert_eq!(PhotonPolarization::Horizontal.to_bit(), false);
        assert_eq!(PhotonPolarization::Vertical.to_bit(), true);
        assert_eq!(PhotonPolarization::Diagonal.to_bit(), false);
        assert_eq!(PhotonPolarization::AntiDiagonal.to_bit(), true);
    }

    #[test]
    fn test_quantum_bit_measurement() {
        let qubit = QuantumBit::new(true, MeasurementBasis::Rectilinear);

        // Measure in correct basis
        let (result, correct) = qubit.clone().measure(MeasurementBasis::Rectilinear);
        assert_eq!(result, true);
        assert!(correct);

        // Measure in wrong basis (result is random)
        let (_, correct) = qubit.measure(MeasurementBasis::Diagonal);
        assert!(!correct);
    }
}
