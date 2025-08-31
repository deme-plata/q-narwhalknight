/// Pietrzak's VDF implementation
/// Based on "Simple Verifiable Delay Functions" by Pietrzak (2018)

use anyhow::{Result, anyhow};
use num_bigint::BigUint;
use num_traits::{Zero, One};
use sha3::{Digest, Sha3_256};
use tracing::{debug, trace};

use crate::VDFParameters;

/// Pietrzak VDF implementation
pub struct PietrzakVDF {
    parameters: VDFParameters,
    modulus: BigUint,
}

impl PietrzakVDF {
    pub fn new(parameters: VDFParameters) -> Result<Self> {
        parameters.validate()?;
        Ok(Self {
            modulus: parameters.modulus.clone(),
            parameters,
        })
    }
    
    /// Evaluate VDF using halving protocol
    pub fn evaluate(&self, g: &BigUint, iterations: u64) -> Result<BigUint> {
        debug!("Starting Pietrzak VDF evaluation with {} iterations", iterations);
        
        let mut y = g.clone();
        for _ in 0..iterations {
            y = y.modpow(&BigUint::from(2u32), &self.modulus);
        }
        
        Ok(y)
    }
    
    /// Generate proof using halving and recursion
    pub fn generate_proof(&self, g: &BigUint, y: &BigUint, iterations: u64) -> Result<Vec<BigUint>> {
        debug!("Generating Pietrzak proof for {} iterations", iterations);
        
        let mut proofs = Vec::new();
        self.generate_proof_recursive(g, y, iterations, &mut proofs)?;
        
        debug!("Generated {} proof elements", proofs.len());
        Ok(proofs)
    }
    
    fn generate_proof_recursive(
        &self,
        g: &BigUint,
        y: &BigUint,
        t: u64,
        proofs: &mut Vec<BigUint>
    ) -> Result<()> {
        if t <= 1 {
            return Ok(());
        }
        
        let t_half = t / 2;
        
        // Compute midpoint: μ = g^(2^(t/2)) mod N
        let mu = self.evaluate(g, t_half)?;
        proofs.push(mu.clone());
        
        // Generate challenge
        let r = self.generate_challenge(g, y, &mu, t)?;
        
        // Compute new base and output for recursion
        let g_new = g.modpow(&r, &self.modulus) * &mu;
        let g_new = g_new % &self.modulus;
        
        let y_new = mu.modpow(&r, &self.modulus) * y;
        let y_new = y_new % &self.modulus;
        
        // Recurse on halved problem
        self.generate_proof_recursive(&g_new, &y_new, t_half, proofs)
    }
    
    /// Verify proof using halving protocol
    pub fn verify_proof(
        &self,
        g: &BigUint,
        y: &BigUint,
        proofs: &[BigUint],
        iterations: u64
    ) -> Result<bool> {
        debug!("Verifying Pietrzak proof");
        
        self.verify_recursive(g, y, proofs, 0, iterations)
    }
    
    fn verify_recursive(
        &self,
        g: &BigUint,
        y: &BigUint,
        proofs: &[BigUint],
        proof_idx: usize,
        t: u64
    ) -> Result<bool> {
        if t <= 1 {
            return Ok(g.modpow(&BigUint::from(2u32), &self.modulus) == *y);
        }
        
        if proof_idx >= proofs.len() {
            return Ok(false);
        }
        
        let t_half = t / 2;
        let mu = &proofs[proof_idx];
        
        // Generate challenge
        let r = self.generate_challenge(g, y, mu, t)?;
        
        // Compute new values
        let g_new = g.modpow(&r, &self.modulus) * mu;
        let g_new = g_new % &self.modulus;
        
        let y_new = mu.modpow(&r, &self.modulus) * y;
        let y_new = y_new % &self.modulus;
        
        // Recurse
        self.verify_recursive(&g_new, &y_new, proofs, proof_idx + 1, t_half)
    }
    
    fn generate_challenge(
        &self,
        g: &BigUint,
        y: &BigUint,
        mu: &BigUint,
        t: u64
    ) -> Result<BigUint> {
        let mut hasher = Sha3_256::new();
        hasher.update(g.to_bytes_be());
        hasher.update(y.to_bytes_be());
        hasher.update(mu.to_bytes_be());
        hasher.update(t.to_be_bytes());
        hasher.update(b"pietrzak-challenge");
        
        let hash = hasher.finalize();
        Ok(BigUint::from_bytes_be(&hash) % &self.modulus)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_pietrzak_vdf() {
        let params = VDFParameters::with_time_parameter(8);
        let vdf = PietrzakVDF::new(params).unwrap();
        
        let g = BigUint::from(2u32);
        let y = vdf.evaluate(&g, 8).unwrap();
        let proofs = vdf.generate_proof(&g, &y, 8).unwrap();
        
        assert!(!proofs.is_empty());
        
        let valid = vdf.verify_proof(&g, &y, &proofs, 8).unwrap();
        assert!(valid);
    }
}