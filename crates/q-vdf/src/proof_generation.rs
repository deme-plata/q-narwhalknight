/// VDF proof generation strategies and utilities

use anyhow::Result;
use serde::{Serialize, Deserialize};
use crate::{VDFParameters, VDFProof, ProofType};
use q_lattice_vrf::VRFResult;
use num_bigint::BigUint;

/// Proof generation strategy
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum ProofStrategy {
    /// Fast proof (lower security)
    Fast,
    /// Standard proof (balanced)
    Standard, 
    /// Secure proof (higher security)
    Secure,
    /// Quantum-enhanced proof
    QuantumEnhanced,
}

/// VDF proof generator
pub struct ProofGenerator {
    parameters: VDFParameters,
    strategy: ProofStrategy,
}

impl ProofGenerator {
    pub fn new(parameters: VDFParameters) -> Result<Self> {
        Ok(Self {
            parameters,
            strategy: ProofStrategy::Standard,
        })
    }
    
    pub async fn generate_proof(
        &self,
        input: &[u8],
        output: &[u8],
        iterations: u64,
        vrf_result: Option<&VRFResult>,
    ) -> Result<VDFProof> {
        let start_time = std::time::Instant::now();
        
        let (proof_type, proof_data, aux_data) = match self.strategy {
            ProofStrategy::QuantumEnhanced => {
                self.generate_quantum_proof(input, output, iterations, vrf_result).await?
            },
            _ => {
                self.generate_classical_proof(input, output, iterations).await?
            }
        };
        
        let generation_time = start_time.elapsed();
        
        Ok(VDFProof {
            proof_type,
            proof_data,
            aux_data,
            generation_time_ns: generation_time.as_nanos() as u64,
            security_parameter: self.parameters.security_level.bits(),
        })
    }
    
    async fn generate_quantum_proof(
        &self,
        input: &[u8],
        output: &[u8], 
        iterations: u64,
        vrf_result: Option<&VRFResult>,
    ) -> Result<(ProofType, Vec<u8>, Vec<Vec<u8>>)> {
        let mut proof_data = Vec::new();
        let mut aux_data = Vec::new();
        
        // Add VRF proof if available
        if let Some(vrf) = vrf_result {
            aux_data.push(vrf.proof.data().to_vec());
        }
        
        // Classical proof component
        proof_data.extend_from_slice(output);
        
        Ok((ProofType::QuantumHybrid, proof_data, aux_data))
    }
    
    async fn generate_classical_proof(
        &self,
        _input: &[u8],
        output: &[u8],
        _iterations: u64,
    ) -> Result<(ProofType, Vec<u8>, Vec<Vec<u8>>)> {
        Ok((ProofType::Wesolowski, output.to_vec(), Vec::new()))
    }
}