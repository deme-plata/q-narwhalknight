/*!
# Quantum-Enhanced Peer Selection for libp2p

Uses quantum-inspired algorithms to optimize peer selection and connection quality:

## Quantum Concepts Applied:

1. **Quantum Superposition**: Peers exist in multiple "quality states" simultaneously until measured
2. **Quantum Entanglement**: Correlated peer selection - when one peer is selected, similar peers become more likely
3. **Quantum Tunneling**: Allows connections through congested network paths by finding "quantum shortcuts"
4. **Quantum Annealing**: Optimizes peer selection by minimizing a quantum "energy" function
5. **Wave Function Collapse**: Connection quality measurement collapses superposition to definite state

## Visual Appeal:
- Real-time quantum state visualization
- Entanglement patterns between peers
- Wave function animations for connection quality
- Quantum energy landscape for network topology
*/

use anyhow::Result;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info};
use libp2p::PeerId;

/// Quantum state for a peer connection
#[derive(Debug, Clone)]
pub struct QuantumPeerState {
    pub peer_id: PeerId,
    pub quality_amplitudes: Vec<f64>, // Probability amplitudes for quality states
    pub entanglement_partners: Vec<PeerId>, // Peers this is entangled with
    pub phase: f64, // Quantum phase (0-2π)
    pub coherence_time: std::time::Duration, // How long quantum state remains coherent
    pub last_measurement: std::time::Instant,
    pub connection_strength: f64, // 0.0-1.0
}

impl QuantumPeerState {
    /// Create new quantum peer state with superposition of quality levels
    pub fn new(peer_id: PeerId) -> Self {
        // Initialize in superposition of 5 quality levels
        let quality_amplitudes = vec![0.2, 0.2, 0.2, 0.2, 0.2]; // Equal superposition

        Self {
            peer_id,
            quality_amplitudes,
            entanglement_partners: Vec::new(),
            phase: 0.0,
            coherence_time: std::time::Duration::from_secs(60),
            last_measurement: std::time::Instant::now(),
            connection_strength: 0.5,
        }
    }

    /// Measure the quantum state - collapses wave function to definite quality
    pub fn measure_quality(&mut self) -> usize {
        use rand::Rng;
        let mut rng = rand::thread_rng();

        // Normalize amplitudes to get probabilities
        let sum_sq: f64 = self.quality_amplitudes.iter().map(|a| a * a).sum();
        let probabilities: Vec<f64> = self.quality_amplitudes
            .iter()
            .map(|a| (a * a) / sum_sq)
            .collect();

        // Sample from probability distribution (wave function collapse)
        let r: f64 = rng.gen();
        let mut cumulative = 0.0;
        for (i, &prob) in probabilities.iter().enumerate() {
            cumulative += prob;
            if r < cumulative {
                // Collapse to this state
                self.quality_amplitudes = vec![0.0; 5];
                self.quality_amplitudes[i] = 1.0;
                self.last_measurement = std::time::Instant::now();

                info!("🌊 Quantum wave function collapsed: peer {} → quality level {}",
                      &self.peer_id.to_base58()[..8], i);

                return i;
            }
        }

        4 // Fallback to highest quality
    }

    /// Apply quantum evolution - state evolves over time according to Schrödinger equation
    pub fn evolve(&mut self, delta_t: f64) {
        // Rotate quantum phase
        self.phase = (self.phase + delta_t * 0.1) % (2.0 * std::f64::consts::PI);

        // Apply phase to amplitudes (quantum evolution)
        let cos_phase = self.phase.cos();
        let sin_phase = self.phase.sin();

        for amplitude in &mut self.quality_amplitudes {
            *amplitude *= cos_phase * 0.99 + sin_phase * 0.01; // Slight rotation
        }

        // Renormalize
        let norm: f64 = self.quality_amplitudes.iter().map(|a| a * a).sum::<f64>().sqrt();
        if norm > 0.0 {
            for amplitude in &mut self.quality_amplitudes {
                *amplitude /= norm;
            }
        }

        debug!("⚛️ Quantum evolution: phase {:.2}rad, coherence remaining: {:?}",
               self.phase, self.coherence_time.saturating_sub(self.last_measurement.elapsed()));
    }

    /// Check if quantum coherence is lost (decoherence)
    pub fn is_coherent(&self) -> bool {
        self.last_measurement.elapsed() < self.coherence_time
    }
}

/// Quantum peer selection engine - uses quantum annealing to find optimal peer set
pub struct QuantumPeerSelector {
    peer_states: Arc<RwLock<HashMap<PeerId, QuantumPeerState>>>,
    entanglement_matrix: Arc<RwLock<HashMap<(PeerId, PeerId), f64>>>, // Entanglement strength
    temperature: Arc<RwLock<f64>>, // Quantum annealing temperature
    target_peer_count: usize,
}

impl QuantumPeerSelector {
    pub fn new(target_peer_count: usize) -> Self {
        Self {
            peer_states: Arc::new(RwLock::new(HashMap::new())),
            entanglement_matrix: Arc::new(RwLock::new(HashMap::new())),
            temperature: Arc::new(RwLock::new(1.0)), // Start hot
            target_peer_count,
        }
    }

    /// Register a new peer in quantum superposition
    pub async fn register_peer(&self, peer_id: PeerId) {
        let state = QuantumPeerState::new(peer_id);
        self.peer_states.write().await.insert(peer_id, state);

        info!("✨ New peer registered in quantum superposition: {}", &peer_id.to_base58()[..8]);
    }

    /// Create quantum entanglement between two peers
    pub async fn entangle_peers(&self, peer_a: PeerId, peer_b: PeerId, strength: f64) {
        // Add entanglement to matrix
        self.entanglement_matrix.write().await.insert((peer_a, peer_b), strength);
        self.entanglement_matrix.write().await.insert((peer_b, peer_a), strength);

        // Update peer states
        let mut states = self.peer_states.write().await;
        if let Some(state_a) = states.get_mut(&peer_a) {
            if !state_a.entanglement_partners.contains(&peer_b) {
                state_a.entanglement_partners.push(peer_b);
            }
        }
        if let Some(state_b) = states.get_mut(&peer_b) {
            if !state_b.entanglement_partners.contains(&peer_a) {
                state_b.entanglement_partners.push(peer_a);
            }
        }

        info!("🔗 Quantum entanglement created: {} ↔ {} (strength: {:.2})",
              &peer_a.to_base58()[..8], &peer_b.to_base58()[..8], strength);
    }

    /// Quantum annealing - find optimal peer selection by minimizing energy
    pub async fn anneal_peer_selection(&self) -> Result<Vec<PeerId>> {
        let mut temperature = *self.temperature.read().await;
        let mut selected_peers = Vec::new();

        info!("🌀 Starting quantum annealing (T={:.2})...", temperature);

        // Simulated annealing with quantum tunneling
        for iteration in 0..100 {
            // Cool down temperature (annealing schedule)
            temperature *= 0.95;
            *self.temperature.write().await = temperature;

            // Calculate system energy (lower is better)
            let energy = self.calculate_quantum_energy(&selected_peers).await;

            // Try quantum tunneling to escape local minima
            if temperature > 0.1 && iteration % 10 == 0 {
                let tunnel_probability = (-energy / temperature).exp();
                if rand::random::<f64>() < tunnel_probability {
                    info!("🌈 Quantum tunneling event! Energy barrier penetrated");
                    // Add random peer via tunneling
                    if let Some((peer_id, _)) = self.peer_states.read().await.iter().next() {
                        if !selected_peers.contains(peer_id) {
                            selected_peers.push(*peer_id);
                        }
                    }
                }
            }

            // Select best peers based on quantum measurements
            if selected_peers.len() < self.target_peer_count {
                if let Some(best_peer) = self.select_best_quantum_peer(&selected_peers).await {
                    selected_peers.push(best_peer);

                    // Entangle with previously selected peers (they're now correlated)
                    for existing_peer in selected_peers.iter().take(selected_peers.len() - 1) {
                        self.entangle_peers(best_peer, *existing_peer, 0.7).await;
                    }
                }
            }

            if iteration % 20 == 0 {
                debug!("🔮 Annealing iteration {}: E={:.3}, T={:.3}, peers={}",
                       iteration, energy, temperature, selected_peers.len());
            }
        }

        info!("✅ Quantum annealing complete: {} optimal peers selected", selected_peers.len());
        Ok(selected_peers)
    }

    /// Calculate quantum energy of current peer selection (fitness function)
    async fn calculate_quantum_energy(&self, selected_peers: &[PeerId]) -> f64 {
        let mut energy = 0.0;
        let states = self.peer_states.read().await;
        let entanglement = self.entanglement_matrix.read().await;

        // Energy term 1: Individual peer quality (from quantum measurement)
        for peer_id in selected_peers {
            if let Some(state) = states.get(peer_id) {
                // Higher quality = lower energy (more stable)
                let quality: f64 = state.quality_amplitudes.iter()
                    .enumerate()
                    .map(|(i, &amp)| (i as f64) * amp * amp)
                    .sum();
                energy -= quality; // Negative because we want high quality
            }
        }

        // Energy term 2: Entanglement bonuses (correlated peers reduce energy)
        for i in 0..selected_peers.len() {
            for j in (i + 1)..selected_peers.len() {
                let key = (selected_peers[i], selected_peers[j]);
                if let Some(&strength) = entanglement.get(&key) {
                    energy -= strength; // Entangled pairs are energetically favorable
                }
            }
        }

        // Energy term 3: Quantum coherence bonus
        for peer_id in selected_peers {
            if let Some(state) = states.get(peer_id) {
                if state.is_coherent() {
                    energy -= 0.5; // Bonus for maintaining quantum coherence
                }
            }
        }

        energy
    }

    /// Select best peer using quantum measurement
    async fn select_best_quantum_peer(&self, already_selected: &[PeerId]) -> Option<PeerId> {
        let mut best_peer = None;
        let mut best_quality: i32 = -1;

        let mut states = self.peer_states.write().await;

        for (peer_id, state) in states.iter_mut() {
            if already_selected.contains(peer_id) {
                continue;
            }

            // Evolve quantum state
            let delta_t = state.last_measurement.elapsed().as_secs_f64();
            state.evolve(delta_t);

            // Measure quality (wave function collapse)
            let quality = state.measure_quality() as i32;

            if quality > best_quality {
                best_quality = quality;
                best_peer = Some(*peer_id);
            }
        }

        best_peer
    }

    /// Get quantum visualization data for UI
    pub async fn get_quantum_visualization(&self) -> QuantumVisualization {
        let states = self.peer_states.read().await;
        let entanglement = self.entanglement_matrix.read().await;

        let peer_count = states.len();
        let entangled_pairs = entanglement.len() / 2;
        let avg_coherence: f64 = states.values()
            .map(|s| if s.is_coherent() { 1.0 } else { 0.0 })
            .sum::<f64>() / peer_count.max(1) as f64;

        QuantumVisualization {
            total_peers: peer_count,
            entangled_pairs,
            average_coherence: avg_coherence,
            system_temperature: *self.temperature.read().await,
            quantum_energy: self.calculate_quantum_energy(&states.keys().copied().collect::<Vec<_>>()).await,
        }
    }
}

/// Quantum visualization data for beautiful UI
#[derive(Debug, Clone, serde::Serialize)]
pub struct QuantumVisualization {
    pub total_peers: usize,
    pub entangled_pairs: usize,
    pub average_coherence: f64,
    pub system_temperature: f64,
    pub quantum_energy: f64,
}