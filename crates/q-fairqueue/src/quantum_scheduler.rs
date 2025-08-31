/// Quantum-enhanced scheduling algorithms for fair transaction ordering

use anyhow::{Result, anyhow};
use serde::{Serialize, Deserialize};
use q_quantum_rng::{QuantumRNG, QuantumRandomness};
use q_lattice_vrf::{LatticeVRF, VRFResult};
use std::collections::HashMap;
use tracing::{debug, trace};

use crate::{TransactionType, QueueConfig};

/// Scheduling policies for transaction ordering
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum SchedulingPolicy {
    /// First-In-First-Out (classical)
    FIFO,
    
    /// Weighted Round Robin
    WeightedRoundRobin,
    
    /// Quantum Fair Scheduling
    QuantumFair,
    
    /// Deficit Round Robin with quantum enhancement
    QuantumDeficitRR,
    
    /// Proportional fair with VRF
    VRFProportionalFair,
}

/// Quantum-enhanced scheduler
pub struct QuantumScheduler {
    /// Scheduling policy
    policy: SchedulingPolicy,
    
    /// Configuration
    config: QueueConfig,
    
    /// Quantum randomness source
    quantum_rng: Option<QuantumRNG>,
    
    /// Lattice VRF for deterministic randomness
    lattice_vrf: Option<LatticeVRF>,
    
    /// Round-robin state
    current_type_index: usize,
    
    /// Weighted scheduling counters
    type_counters: HashMap<TransactionType, u64>,
    
    /// Deficit counters for DRR
    deficits: HashMap<TransactionType, i64>,
    
    /// VRF round counter
    vrf_round: u64,
}

impl QuantumScheduler {
    /// Create new quantum scheduler
    pub async fn new(
        config: QueueConfig,
        quantum_rng: Option<&QuantumRNG>,
        lattice_vrf: Option<&LatticeVRF>,
    ) -> Result<Self> {
        let policy = if quantum_rng.is_some() && lattice_vrf.is_some() {
            SchedulingPolicy::VRFProportionalFair
        } else if quantum_rng.is_some() {
            SchedulingPolicy::QuantumFair
        } else {
            SchedulingPolicy::WeightedRoundRobin
        };

        let transaction_types = [
            TransactionType::Emergency,
            TransactionType::Consensus,
            TransactionType::QuantumBeacon,
            TransactionType::System,
            TransactionType::User,
        ];

        let mut type_counters = HashMap::new();
        let mut deficits = HashMap::new();

        for &tx_type in &transaction_types {
            type_counters.insert(tx_type, 0);
            deficits.insert(tx_type, 0);
        }

        Ok(Self {
            policy,
            config,
            quantum_rng: quantum_rng.cloned(),
            lattice_vrf: lattice_vrf.cloned(),
            current_type_index: 0,
            type_counters,
            deficits,
            vrf_round: 0,
        })
    }

    /// Select next transaction type to serve
    pub async fn select_next_type(&mut self) -> Result<Option<TransactionType>> {
        match self.policy {
            SchedulingPolicy::FIFO => self.select_fifo().await,
            SchedulingPolicy::WeightedRoundRobin => self.select_weighted_rr().await,
            SchedulingPolicy::QuantumFair => self.select_quantum_fair().await,
            SchedulingPolicy::QuantumDeficitRR => self.select_quantum_deficit_rr().await,
            SchedulingPolicy::VRFProportionalFair => self.select_vrf_proportional_fair().await,
        }
    }

    /// FIFO scheduling (simplest)
    async fn select_fifo(&mut self) -> Result<Option<TransactionType>> {
        // In FIFO, we check queues in priority order
        let priority_order = [
            TransactionType::Emergency,
            TransactionType::Consensus,
            TransactionType::QuantumBeacon,
            TransactionType::System,
            TransactionType::User,
        ];

        for &tx_type in &priority_order {
            // This would check if queue has items - simplified for now
            return Ok(Some(tx_type));
        }

        Ok(None)
    }

    /// Weighted Round Robin scheduling
    async fn select_weighted_rr(&mut self) -> Result<Option<TransactionType>> {
        let types = [
            TransactionType::Emergency,
            TransactionType::Consensus,
            TransactionType::QuantumBeacon,
            TransactionType::System,
            TransactionType::User,
        ];

        // Find type with lowest counter relative to weight
        let mut best_type = None;
        let mut best_ratio = f64::INFINITY;

        for &tx_type in &types {
            let counter = *self.type_counters.get(&tx_type).unwrap_or(&0);
            let weight = *self.config.type_weights.get(&tx_type).unwrap_or(&1.0);
            
            let ratio = counter as f64 / weight;
            if ratio < best_ratio {
                best_ratio = ratio;
                best_type = Some(tx_type);
            }
        }

        if let Some(selected) = best_type {
            *self.type_counters.entry(selected).or_insert(0) += 1;
            debug!("WRR selected {:?} (ratio: {:.3})", selected, best_ratio);
        }

        Ok(best_type)
    }

    /// Quantum Fair Scheduling using quantum randomness
    async fn select_quantum_fair(&mut self) -> Result<Option<TransactionType>> {
        if let Some(ref qrng) = self.quantum_rng {
            // Use quantum randomness for fair selection
            let random_bytes = qrng.generate_bytes(4).await?;
            let random_value = u32::from_be_bytes([
                random_bytes[0], random_bytes[1], random_bytes[2], random_bytes[3]
            ]);

            // Weight by type priorities but add quantum fairness
            let types = [
                (TransactionType::Emergency, 1000.0),
                (TransactionType::Consensus, 500.0),
                (TransactionType::QuantumBeacon, 400.0),
                (TransactionType::System, 200.0),
                (TransactionType::User, 100.0),
            ];

            let total_weight: f64 = types.iter().map(|(_, w)| w).sum();
            let selection_point = (random_value as f64 / u32::MAX as f64) * total_weight;

            let mut cumulative = 0.0;
            for (tx_type, weight) in types {
                cumulative += weight;
                if selection_point <= cumulative {
                    trace!("Quantum fair selected {:?} (point: {:.3}, cum: {:.3})", 
                           tx_type, selection_point, cumulative);
                    return Ok(Some(tx_type));
                }
            }

            // Fallback to highest priority
            Ok(Some(TransactionType::Emergency))
        } else {
            // Fall back to weighted round robin
            self.select_weighted_rr().await
        }
    }

    /// Quantum-enhanced Deficit Round Robin
    async fn select_quantum_deficit_rr(&mut self) -> Result<Option<TransactionType>> {
        let types = [
            TransactionType::Emergency,
            TransactionType::Consensus,
            TransactionType::QuantumBeacon,
            TransactionType::System,
            TransactionType::User,
        ];

        // Add quantum uncertainty to deficit calculations
        let quantum_adjustment = if let Some(ref qrng) = self.quantum_rng {
            let random_byte = qrng.generate_bytes(1).await?;
            (random_byte[0] as i64) - 128 // Range: -128 to +127
        } else {
            0
        };

        for &tx_type in &types {
            let weight = *self.config.type_weights.get(&tx_type).unwrap_or(&1.0) as i64;
            let current_deficit = *self.deficits.get(&tx_type).unwrap_or(&0);
            
            // Add quantum-adjusted quantum
            let quantum = weight + (quantum_adjustment / 10); // Small adjustment
            *self.deficits.entry(tx_type).or_insert(0) += quantum;
            
            // If deficit is positive, this type can be served
            if current_deficit + quantum > 0 {
                *self.deficits.entry(tx_type).or_insert(0) -= 100; // Cost per packet
                debug!("Quantum DRR selected {:?} (deficit: {}, quantum: {})", 
                       tx_type, current_deficit, quantum);
                return Ok(Some(tx_type));
            }
        }

        Ok(None)
    }

    /// VRF-based Proportional Fair scheduling
    async fn select_vrf_proportional_fair(&mut self) -> Result<Option<TransactionType>> {
        if let Some(ref vrf) = self.lattice_vrf {
            self.vrf_round += 1;
            
            // Create VRF input from round and scheduler state
            let mut vrf_input = Vec::new();
            vrf_input.extend_from_slice(&self.vrf_round.to_be_bytes());
            vrf_input.extend_from_slice(b"quantum-scheduler");
            
            match vrf.evaluate(&vrf_input, self.vrf_round).await {
                Ok(vrf_result) => {
                    // Use VRF output for deterministic but unpredictable selection
                    let output_bytes = vrf_result.output.as_bytes();
                    if output_bytes.len() >= 4 {
                        let selection_value = u32::from_be_bytes([
                            output_bytes[0], output_bytes[1], 
                            output_bytes[2], output_bytes[3]
                        ]);
                        
                        let selected_type = self.vrf_select_type(selection_value)?;
                        debug!("VRF proportional fair selected {:?} (value: {})", 
                               selected_type, selection_value);
                        return Ok(Some(selected_type));
                    }
                },
                Err(e) => {
                    debug!("VRF evaluation failed in scheduler: {}", e);
                }
            }
        }

        // Fallback to quantum fair or weighted RR
        if self.quantum_rng.is_some() {
            self.select_quantum_fair().await
        } else {
            self.select_weighted_rr().await
        }
    }

    /// Select transaction type based on VRF output
    fn vrf_select_type(&self, selection_value: u32) -> Result<TransactionType> {
        // Map VRF output to transaction types proportionally
        let weights = [
            (TransactionType::Emergency, 100),    // 10%
            (TransactionType::Consensus, 300),    // 30% 
            (TransactionType::QuantumBeacon, 100), // 10%
            (TransactionType::System, 200),       // 20%
            (TransactionType::User, 300),         // 30%
        ];

        let total_weight: u32 = weights.iter().map(|(_, w)| w).sum();
        let selection_point = selection_value % total_weight;

        let mut cumulative = 0u32;
        for (tx_type, weight) in weights {
            cumulative += weight;
            if selection_point < cumulative {
                return Ok(tx_type);
            }
        }

        // Fallback
        Ok(TransactionType::User)
    }

    /// Get scheduling statistics
    pub fn get_stats(&self) -> HashMap<String, u64> {
        let mut stats = HashMap::new();
        stats.insert("vrf_round".to_string(), self.vrf_round);
        
        for (tx_type, counter) in &self.type_counters {
            stats.insert(format!("{:?}_count", tx_type), *counter);
        }
        
        stats
    }

    /// Reset scheduler state
    pub fn reset(&mut self) {
        self.current_type_index = 0;
        self.vrf_round = 0;
        
        for counter in self.type_counters.values_mut() {
            *counter = 0;
        }
        
        for deficit in self.deficits.values_mut() {
            *deficit = 0;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_scheduler_creation() {
        let config = QueueConfig::default();
        let scheduler = QuantumScheduler::new(config, None, None).await.unwrap();
        
        assert_eq!(scheduler.policy, SchedulingPolicy::WeightedRoundRobin);
    }

    #[tokio::test]
    async fn test_weighted_round_robin() {
        let config = QueueConfig::default();
        let mut scheduler = QuantumScheduler::new(config, None, None).await.unwrap();
        
        // Should select based on weights
        let selected = scheduler.select_next_type().await.unwrap();
        assert!(selected.is_some());
    }
}