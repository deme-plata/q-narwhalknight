/// Post-Quantum Contract Deployment Signatures
/// v3.7.4: Dilithium5 signatures for smart contract deployments
///
/// Contract deployments are one-time operations, so the large signature
/// size (4,627 bytes) is acceptable. Using NIST Level 5 security ensures
/// quantum resistance for immutable deployed contracts.
///
/// Security: NIST Level 5 (AES-256 equivalent post-quantum security)
/// Use case: One signature per contract deployment (not stored on-chain per tx)

use pqcrypto_dilithium::dilithium5;
use pqcrypto_traits::sign::{PublicKey as PQPublicKey, SignedMessage};
use serde::{Deserialize, Serialize};
use sha3::{Digest, Sha3_256};
use std::collections::HashMap;

/// Helper module for serializing Option<[u8; 64]>
mod option_sig_64 {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    pub fn serialize<S>(value: &Option<[u8; 64]>, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match value {
            Some(arr) => {
                let vec: Vec<u8> = arr.to_vec();
                vec.serialize(serializer)
            }
            None => serializer.serialize_none(),
        }
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Option<[u8; 64]>, D::Error>
    where
        D: Deserializer<'de>,
    {
        let opt: Option<Vec<u8>> = Option::deserialize(deserializer)?;
        match opt {
            Some(vec) if vec.len() == 64 => {
                let mut arr = [0u8; 64];
                arr.copy_from_slice(&vec);
                Ok(Some(arr))
            }
            Some(_) => Err(serde::de::Error::custom("Expected 64-byte signature")),
            None => Ok(None),
        }
    }
}

/// Contract deployment with Dilithium5 signature
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PQContractDeployment {
    /// Contract bytecode hash (SHA3-256)
    pub code_hash: [u8; 32],

    /// Contract type identifier
    pub contract_type: String,

    /// Deployer's wallet address
    pub deployer: [u8; 32],

    /// Deployer's Dilithium5 public key (2,592 bytes)
    #[serde(with = "serde_bytes")]
    pub deployer_pubkey: Vec<u8>,

    /// Deployment parameters
    pub parameters: HashMap<String, serde_json::Value>,

    /// Deployment block height
    pub block_height: u64,

    /// Deployment timestamp (Unix seconds)
    pub timestamp: u64,

    /// Nonce (for replay protection)
    pub nonce: u64,

    /// Dilithium5 signature over deployment data (4,627 bytes)
    #[serde(with = "serde_bytes")]
    pub signature: Vec<u8>,

    /// Optional Ed25519 signature (for hybrid mode)
    #[serde(with = "option_sig_64")]
    pub ed25519_signature: Option<[u8; 64]>,
}

/// Deployment request before signing
#[derive(Debug, Clone)]
pub struct PQDeploymentRequest {
    pub code: Vec<u8>,
    pub contract_type: String,
    pub deployer: [u8; 32],
    pub parameters: HashMap<String, serde_json::Value>,
    pub block_height: u64,
    pub nonce: u64,
}

impl PQDeploymentRequest {
    /// Create a new deployment request
    pub fn new(
        code: Vec<u8>,
        contract_type: String,
        deployer: [u8; 32],
        parameters: HashMap<String, serde_json::Value>,
        block_height: u64,
        nonce: u64,
    ) -> Self {
        Self {
            code,
            contract_type,
            deployer,
            parameters,
            block_height,
            nonce,
        }
    }

    /// Compute code hash
    fn compute_code_hash(&self) -> [u8; 32] {
        let mut hasher = Sha3_256::new();
        hasher.update(&self.code);
        hasher.finalize().into()
    }

    /// Get deployment data for signing
    fn get_signing_data(&self) -> Vec<u8> {
        let mut data = Vec::new();

        // Include code hash (not full code - too large for signature)
        let code_hash = self.compute_code_hash();
        data.extend_from_slice(&code_hash);

        // Include contract type
        data.extend_from_slice(self.contract_type.as_bytes());
        data.push(0); // Null terminator

        // Include deployer
        data.extend_from_slice(&self.deployer);

        // Include parameters hash
        let params_json = serde_json::to_string(&self.parameters).unwrap_or_default();
        let mut params_hasher = Sha3_256::new();
        params_hasher.update(params_json.as_bytes());
        let params_hash: [u8; 32] = params_hasher.finalize().into();
        data.extend_from_slice(&params_hash);

        // Include block height
        data.extend_from_slice(&self.block_height.to_le_bytes());

        // Include nonce
        data.extend_from_slice(&self.nonce.to_le_bytes());

        data
    }

    /// Sign deployment with Dilithium5 and create deployment transaction
    pub fn sign(
        self,
        dilithium_secret: &dilithium5::SecretKey,
        dilithium_public: &dilithium5::PublicKey,
        ed25519_signature: Option<[u8; 64]>,
    ) -> PQContractDeployment {
        let signing_data = self.get_signing_data();
        let signature = dilithium5::sign(&signing_data, dilithium_secret)
            .as_bytes()
            .to_vec();

        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        tracing::info!(
            "🔐 [DILITHIUM5] Contract deployment signed: type={}, deployer={}, sig_size={}",
            self.contract_type,
            hex::encode(&self.deployer[..8]),
            signature.len()
        );

        PQContractDeployment {
            code_hash: self.compute_code_hash(),
            contract_type: self.contract_type,
            deployer: self.deployer,
            deployer_pubkey: dilithium_public.as_bytes().to_vec(),
            parameters: self.parameters,
            block_height: self.block_height,
            timestamp,
            nonce: self.nonce,
            signature,
            ed25519_signature,
        }
    }
}

impl PQContractDeployment {
    /// Verify the deployment signature
    pub fn verify(&self) -> Result<(), String> {
        // 1. Check public key size
        if self.deployer_pubkey.len() != 2592 {
            return Err(format!(
                "Invalid Dilithium5 public key size: expected 2592, got {}",
                self.deployer_pubkey.len()
            ));
        }

        // 2. Verify deployer address matches public key hash
        let expected_deployer = Self::derive_address_from_pubkey(&self.deployer_pubkey);
        if self.deployer != expected_deployer {
            return Err("Deployer address does not match public key".to_string());
        }

        // 3. Parse public key
        let pubkey = dilithium5::PublicKey::from_bytes(&self.deployer_pubkey)
            .map_err(|_| "Invalid Dilithium5 public key format")?;

        // 4. Parse signed message
        let signed_msg = SignedMessage::from_bytes(&self.signature)
            .map_err(|_| "Invalid Dilithium5 signature format")?;

        // 5. Verify signature and recover message
        let recovered_data = dilithium5::open(&signed_msg, &pubkey)
            .map_err(|_| "Dilithium5 signature verification failed")?;

        // 6. Reconstruct expected signing data
        let expected_data = self.reconstruct_signing_data();
        if recovered_data.as_slice() != expected_data.as_slice() {
            return Err("Deployment data mismatch".to_string());
        }

        tracing::info!(
            "✅ [DILITHIUM5] Contract deployment verified: type={}, deployer={}",
            self.contract_type,
            hex::encode(&self.deployer[..8])
        );

        Ok(())
    }

    /// Derive wallet address from Dilithium5 public key
    pub fn derive_address_from_pubkey(pubkey: &[u8]) -> [u8; 32] {
        let mut hasher = Sha3_256::new();
        hasher.update(b"qnk_deployer_address_v1");
        hasher.update(pubkey);
        hasher.finalize().into()
    }

    /// Reconstruct signing data for verification
    fn reconstruct_signing_data(&self) -> Vec<u8> {
        let mut data = Vec::new();

        // Include code hash
        data.extend_from_slice(&self.code_hash);

        // Include contract type
        data.extend_from_slice(self.contract_type.as_bytes());
        data.push(0);

        // Include deployer
        data.extend_from_slice(&self.deployer);

        // Include parameters hash
        let params_json = serde_json::to_string(&self.parameters).unwrap_or_default();
        let mut params_hasher = Sha3_256::new();
        params_hasher.update(params_json.as_bytes());
        let params_hash: [u8; 32] = params_hasher.finalize().into();
        data.extend_from_slice(&params_hash);

        // Include block height
        data.extend_from_slice(&self.block_height.to_le_bytes());

        // Include nonce
        data.extend_from_slice(&self.nonce.to_le_bytes());

        data
    }

    /// Get contract address (derived from code hash + deployer + nonce)
    pub fn get_contract_address(&self) -> [u8; 32] {
        let mut hasher = Sha3_256::new();
        hasher.update(b"qnk_contract_address_v1");
        hasher.update(&self.code_hash);
        hasher.update(&self.deployer);
        hasher.update(&self.nonce.to_le_bytes());
        hasher.finalize().into()
    }

    /// Get deployment ID for tracking
    pub fn deployment_id(&self) -> String {
        let addr = self.get_contract_address();
        format!("deploy-{}", hex::encode(&addr[..16]))
    }
}

/// Deployer keypair manager for contract deployments
pub struct ContractDeployerKeys {
    pub public_key: dilithium5::PublicKey,
    secret_key: dilithium5::SecretKey,
    pub address: [u8; 32],
}

impl ContractDeployerKeys {
    /// Generate new deployer keypair
    pub fn generate() -> Self {
        let (public_key, secret_key) = dilithium5::keypair();
        let address = PQContractDeployment::derive_address_from_pubkey(public_key.as_bytes());

        tracing::info!(
            "🔐 [DILITHIUM5] Generated contract deployer keys: address={}",
            hex::encode(&address[..8])
        );

        Self {
            public_key,
            secret_key,
            address,
        }
    }

    /// Sign a deployment request
    pub fn sign_deployment(
        &self,
        request: PQDeploymentRequest,
        ed25519_signature: Option<[u8; 64]>,
    ) -> PQContractDeployment {
        request.sign(&self.secret_key, &self.public_key, ed25519_signature)
    }

    /// Get public key bytes
    pub fn public_key_bytes(&self) -> Vec<u8> {
        self.public_key.as_bytes().to_vec()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_contract_deployment_signing() {
        // Generate deployer keys
        let deployer_keys = ContractDeployerKeys::generate();

        // Create deployment request
        let code = b"(module (func (export \"main\")))".to_vec();
        let mut params = HashMap::new();
        params.insert("name".to_string(), serde_json::json!("TestToken"));
        params.insert("symbol".to_string(), serde_json::json!("TEST"));

        let request = PQDeploymentRequest::new(
            code,
            "ERC20Token".to_string(),
            deployer_keys.address,
            params,
            1000,
            1,
        );

        // Sign deployment
        let deployment = deployer_keys.sign_deployment(request, None);

        // Verify deployment
        assert!(deployment.verify().is_ok());
    }

    #[test]
    fn test_deployment_address_derivation() {
        let deployer_keys = ContractDeployerKeys::generate();
        let code = b"test code".to_vec();

        let request = PQDeploymentRequest::new(
            code,
            "TestContract".to_string(),
            deployer_keys.address,
            HashMap::new(),
            100,
            1,
        );

        let deployment = deployer_keys.sign_deployment(request, None);

        // Address should be deterministic
        let addr1 = deployment.get_contract_address();
        let addr2 = deployment.get_contract_address();
        assert_eq!(addr1, addr2);
    }

    #[test]
    fn test_tampered_deployment_fails() {
        let deployer_keys = ContractDeployerKeys::generate();
        let code = b"test code".to_vec();

        let request = PQDeploymentRequest::new(
            code,
            "TestContract".to_string(),
            deployer_keys.address,
            HashMap::new(),
            100,
            1,
        );

        let mut deployment = deployer_keys.sign_deployment(request, None);

        // Tamper with the code hash
        deployment.code_hash[0] ^= 0xFF;

        // Verification should fail
        assert!(deployment.verify().is_err());
    }
}
