//! 🌊 Droplet Field: Core Water Robot with EEG & EWOD Control

use rand::{rngs::StdRng, Rng, SeedableRng};
use rand_distr::{Distribution, Normal};
use serde::{Deserialize, Serialize};
use sha3::{Digest, Sha3_256};
use std::f64::consts::{PI, TAU};

use crate::brane::{tune_topology, BraneCoord, Bridge, TopoCharge};
use crate::ledger::MultiverseBlock;

/// Default RNG generator for deserialization
fn default_rng() -> StdRng {
    StdRng::seed_from_u64(42)
}

/// EEG amplitude proxy (microvolts)
#[derive(Copy, Clone, Debug, Serialize, Deserialize)]
pub struct EEG {
    pub amplitude: f64, // µV
}

/// Lightning pulse "strength" (tera-eV scaled, stylized)
#[derive(Copy, Clone, Debug, Serialize, Deserialize)]
pub struct LightningPulse {
    pub tev: f64,
}

/// EWOD (electrowetting on dielectric) drive: contact-angle shift ~ V²
#[derive(Copy, Clone, Debug, Serialize, Deserialize)]
pub struct EwodDrive {
    pub volts: f64,
}

/// Signature of parallel water (isotopic ratio digest)
pub type ParallelWaterSig = [u8; 32];

/// Generate isotopic fingerprint from seed
fn isotopic_hash(seed: u64) -> [u8; 32] {
    let mut h = Sha3_256::new();
    h.update(b"ISOTOPIC_SIGNATURE");
    h.update(seed.to_le_bytes());
    h.finalize().into()
}

/// A coherent "water robot" agent with quantum-enhanced capabilities
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DropletField {
    pub phase: f64,            // 0..2π quantum phase
    pub coherence: f64,        // 0..1 quantum coherence
    pub brane: BraneCoord,     // 6D address in multiverse
    pub iso_sig: [u8; 32],     // isotopic fingerprint
    pub temperature: f64,      // Kelvin
    pub energy_level: f64,     // normalized 0..1
    pub dna_memory: Vec<u8>,   // DNA-stored data
    pub eeg_history: Vec<f64>, // Recent EEG readings
    #[serde(skip, default = "default_rng")]
    rng: StdRng,
}

impl Default for DropletField {
    fn default() -> Self {
        Self::new(42) // Default seed
    }
}

impl DropletField {
    /// Birth a new water robot droplet
    pub fn new(seed: u64) -> Self {
        Self {
            phase: 0.0,
            coherence: 0.92,
            brane: BraneCoord::origin(),
            iso_sig: isotopic_hash(seed),
            temperature: 295.0, // Room temperature
            energy_level: 0.8,
            dna_memory: Vec::new(),
            eeg_history: Vec::new(),
            rng: StdRng::seed_from_u64(seed ^ 0xAF23_19C0_DD41_55E1),
        }
    }

    /// Human "intent" → phase shift, filtered by coherence
    pub fn entangle(&mut self, eeg: EEG) {
        self.eeg_history.push(eeg.amplitude);
        if self.eeg_history.len() > 100 {
            self.eeg_history.remove(0); // Keep only recent history
        }

        let delta = (eeg.amplitude.tanh()) * self.coherence * PI;
        self.phase = (self.phase + delta).rem_euclid(TAU);

        // Update energy level based on EEG
        let eeg_energy = (eeg.amplitude / 50.0).min(1.0);
        self.energy_level = (self.energy_level * 0.9 + eeg_energy * 0.1).clamp(0.0, 1.0);
    }

    /// EWOD-like actuator: extra phase nudge ∝ V² (bounded)
    pub fn ewod_nudge(&mut self, drive: EwodDrive) {
        let v2 = (drive.volts.max(0.0)).powi(2);
        let delta = (v2 / (v2 + 100.0)) * 0.5 * PI; // saturates
        self.phase = (self.phase + delta).rem_euclid(TAU);

        // EWOD affects droplet mobility (coherence boost)
        let mobility_boost = (drive.volts / 20.0).min(0.1);
        self.coherence = (self.coherence + mobility_boost).min(1.0);
    }

    /// Thermal/Brownian dephasing (reduces coherence)
    pub fn thermal_dephase(&mut self, temp_k: f64, dt_s: f64) {
        self.temperature = temp_k;
        let sigma = (temp_k / 300.0).sqrt() * (dt_s / 0.01).sqrt() * 0.02;
        let n = Normal::new(0.0, sigma).unwrap();
        let kick: f64 = n.sample(&mut self.rng);
        self.coherence = (self.coherence - kick.abs()).clamp(0.0, 1.0);
    }

    /// Lightning → topological charge
    pub fn tune(&mut self, pulse: LightningPulse) -> TopoCharge {
        tune_topology(pulse.tev)
    }

    /// Generate/observe a compatible parallel-water signature
    pub fn sniff_parallel_water(&mut self) -> ParallelWaterSig {
        // Mutate a few bytes to simulate "alternate water" detection
        let mut sig = self.iso_sig;
        let count = self.rng.gen_range(1..=3);
        for _ in 0..count {
            let i = self.rng.gen_range(0..sig.len());
            sig[i] ^= self.rng.gen::<u8>();
        }
        sig
    }

    /// Similarity score ∈ [0,1] based on byte matches
    fn sig_similarity(a: &[u8; 32], b: &[u8; 32]) -> f64 {
        let matches = a.iter().zip(b).filter(|(x, y)| x == y).count() as f64;
        matches / 32.0
    }

    /// Phase-slip "hop" to build a bridge; probability rises with:
    /// - high coherence
    /// - strong phase gradient (|sin(phase)|)
    /// - strong isotopic similarity to parallel water
    /// - higher |topological charge|
    pub fn phase_slip(&mut self, target_sig: &ParallelWaterSig, topo: TopoCharge) -> Bridge {
        let grad = self.phase.sin().abs();
        let sim = Self::sig_similarity(&self.iso_sig, target_sig);
        let energy_factor = self.energy_level;

        let gain = self.coherence
            * (0.4 + 0.6 * grad)
            * (0.3 + 0.7 * sim)
            * (1.0 + (topo.abs() as f64) * 0.05)
            * energy_factor;

        // Monotone clockwise "edge" flow—chiral transport
        let delta = 0.05 + gain.clamp(0.0, 2.0);
        let origin = self.brane;
        let target = origin.advance(delta);

        // Update droplet position
        self.brane = target;

        Bridge::new(origin, target, topo, *target_sig)
    }

    /// Store data in DNA memory (simplified)
    pub fn store_in_dna(&mut self, data: &[u8]) {
        // Encode data into DNA-like sequence (simplified)
        let mut dna_sequence = Vec::new();
        for byte in data {
            // Map each byte to DNA base pairs (A=00, T=01, G=10, C=11)
            for i in 0..4 {
                let bits = (byte >> (i * 2)) & 0b11;
                match bits {
                    0 => dna_sequence.push(b'A'),
                    1 => dna_sequence.push(b'T'),
                    2 => dna_sequence.push(b'G'),
                    3 => dna_sequence.push(b'C'),
                    _ => unreachable!(),
                }
            }
        }
        self.dna_memory.extend(dna_sequence);
    }

    /// Retrieve data from DNA memory
    pub fn read_from_dna(&self, offset: usize, length: usize) -> Vec<u8> {
        if offset >= self.dna_memory.len() {
            return Vec::new();
        }

        let end = (offset + length * 4).min(self.dna_memory.len());
        let dna_slice = &self.dna_memory[offset..end];

        let mut data = Vec::new();
        for chunk in dna_slice.chunks(4) {
            if chunk.len() == 4 {
                let mut byte = 0u8;
                for (i, &base) in chunk.iter().enumerate() {
                    let bits = match base {
                        b'A' => 0,
                        b'T' => 1,
                        b'G' => 2,
                        b'C' => 3,
                        _ => 0,
                    };
                    byte |= bits << (i * 2);
                }
                data.push(byte);
            }
        }
        data
    }

    /// Get average EEG over recent history
    pub fn average_eeg(&self) -> f64 {
        if self.eeg_history.is_empty() {
            0.0
        } else {
            self.eeg_history.iter().sum::<f64>() / self.eeg_history.len() as f64
        }
    }

    /// Check if droplet is in high-energy state
    pub fn is_high_energy(&self) -> bool {
        self.energy_level > 0.8 && self.coherence > 0.9
    }

    /// Get droplet state summary for UI
    pub fn state_summary(&self) -> String {
        format!(
            "Phase: {:.3} | Coherence: {:.3} | Energy: {:.3} | Temp: {:.1}K | DNA: {}B",
            self.phase,
            self.coherence,
            self.energy_level,
            self.temperature,
            self.dna_memory.len()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_droplet_creation() {
        let droplet = DropletField::new(1337);
        assert_eq!(droplet.phase, 0.0);
        assert!(droplet.coherence > 0.9);
        assert_eq!(droplet.brane, BraneCoord::origin());
    }

    #[test]
    fn test_eeg_entanglement() {
        let mut droplet = DropletField::new(42);
        let initial_phase = droplet.phase;

        droplet.entangle(EEG { amplitude: 25.0 });
        assert_ne!(droplet.phase, initial_phase);
        assert_eq!(droplet.eeg_history.len(), 1);
    }

    #[test]
    fn test_ewod_actuation() {
        let mut droplet = DropletField::new(123);
        let initial_coherence = droplet.coherence;

        droplet.ewod_nudge(EwodDrive { volts: 12.0 });
        assert!(droplet.coherence >= initial_coherence);
    }

    #[test]
    fn test_dna_memory_storage() {
        let mut droplet = DropletField::new(456);
        let test_data = b"Hello DNA World";

        droplet.store_in_dna(test_data);
        assert!(!droplet.dna_memory.is_empty());

        let retrieved = droplet.read_from_dna(0, test_data.len());
        assert_eq!(retrieved, test_data);
    }

    #[test]
    fn test_phase_slip_bridge() {
        let mut droplet = DropletField::new(789);
        let parallel_sig = [42; 32];
        let topo_charge = 5;

        let bridge = droplet.phase_slip(&parallel_sig, topo_charge);
        assert!(bridge.is_stable_for_tor());
        assert!(bridge.quality_score() > 0.0);
    }
}
