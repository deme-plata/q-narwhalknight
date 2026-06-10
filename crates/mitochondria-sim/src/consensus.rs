/// Biological consensus mechanism for water-robot blockchain
///
/// This module implements "Proof of Biosynthesis" - a novel consensus algorithm
/// where blockchain validation is based on DNA synthesis mass and biological activity.

use anyhow::Result;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tracing::{debug, info, warn};

use crate::{DropletNode, BiologicalConsensus, DNABlockchain, dna_storage};

/// Consensus configuration parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsensusConfig {
    pub minimum_validators: usize,      // Minimum droplets for consensus
    pub mass_threshold_pg: f64,         // Minimum DNA mass to validate
    pub consensus_timeout_ms: u64,      // Time limit for consensus round
    pub biosynthesis_reward: f64,       // Energy reward for validation
}

impl Default for ConsensusConfig {
    fn default() -> Self {
        Self {
            minimum_validators: 3,
            mass_threshold_pg: 5.0,
            consensus_timeout_ms: 5000,
            biosynthesis_reward: 0.2,
        }
    }
}

/// Consensus round state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsensusRound {
    pub round_id: u64,
    pub started_at: DateTime<Utc>,
    pub participating_droplets: Vec<String>,
    pub proposed_blocks: HashMap<String, ProposedBlock>,
    pub votes: HashMap<String, Vote>,
    pub leader_droplet: Option<String>,
    pub status: ConsensusStatus,
}

/// Block proposal for consensus
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProposedBlock {
    pub proposer: String,
    pub dna_sequence: String,           // Encoded blockchain data
    pub synthesis_proof: SynthesisProof, // Proof of biological work
    pub parent_hash: String,
    pub timestamp: DateTime<Utc>,
}

/// Proof that DNA synthesis actually occurred
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SynthesisProof {
    pub energy_expended: f64,
    pub synthesis_time_ms: u64,
    pub dna_mass_increase: f64,
    pub biological_signature: String,   // Hash of biological process
}

/// Vote in consensus round
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Vote {
    pub voter: String,
    pub block_hash: String,
    pub vote_weight: f64,               // Based on DNA mass
    pub biological_verification: bool,   // Did voter verify biosynthesis?
    pub timestamp: DateTime<Utc>,
}

/// Consensus round status
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ConsensusStatus {
    Proposing,
    Voting,
    Finalizing,
    Completed,
    Failed,
}

/// Select consensus leader based on DNA mass (heaviest droplet leads)
pub fn select_consensus_leader(
    droplets: &HashMap<String, DropletNode>,
    config: &ConsensusConfig
) -> Result<String> {
    let eligible_droplets: Vec<_> = droplets
        .iter()
        .filter(|(_, droplet)| {
            droplet.dna_data.total_mass_picograms >= config.mass_threshold_pg
                && droplet.energy_level > 0.3
        })
        .collect();

    if eligible_droplets.len() < config.minimum_validators {
        return Err(anyhow::anyhow!("Insufficient validators for consensus"));
    }

    // Select droplet with highest DNA mass as leader
    let leader = eligible_droplets
        .iter()
        .max_by(|(_, a), (_, b)| {
            a.dna_data.total_mass_picograms
                .partial_cmp(&b.dna_data.total_mass_picograms)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|(id, _)| (*id).clone())
        .ok_or_else(|| anyhow::anyhow!("No suitable leader found"))?;

    info!("👑 Selected consensus leader: {} (DNA mass: {:.2} pg)", 
          leader, droplets[&leader].dna_data.total_mass_picograms);
    
    Ok(leader)
}

/// Start new consensus round
pub async fn start_consensus_round(
    droplets: &HashMap<String, DropletNode>,
    config: &ConsensusConfig,
    round_id: u64
) -> Result<ConsensusRound> {
    let leader = select_consensus_leader(droplets, config)?;
    
    let participating_droplets: Vec<String> = droplets
        .iter()
        .filter(|(_, droplet)| {
            droplet.dna_data.total_mass_picograms >= config.mass_threshold_pg
        })
        .map(|(id, _)| id.clone())
        .collect();

    let round = ConsensusRound {
        round_id,
        started_at: Utc::now(),
        participating_droplets,
        proposed_blocks: HashMap::new(),
        votes: HashMap::new(),
        leader_droplet: Some(leader),
        status: ConsensusStatus::Proposing,
    };

    info!("🔄 Started consensus round {} with {} participants", 
          round_id, round.participating_droplets.len());
    
    Ok(round)
}

/// Create block proposal from droplet's DNA synthesis
pub fn create_block_proposal(
    droplet: &DropletNode,
    parent_hash: &str
) -> Result<ProposedBlock> {
    if droplet.dna_data.synthesis_history.is_empty() {
        return Err(anyhow::anyhow!("No DNA synthesis history to propose"));
    }

    let latest_synthesis = droplet.dna_data.synthesis_history.last().unwrap();
    
    let synthesis_proof = SynthesisProof {
        energy_expended: latest_synthesis.energy_cost,
        synthesis_time_ms: latest_synthesis.synthesis_time_ms,
        dna_mass_increase: latest_synthesis.energy_cost * 2.0,
        biological_signature: calculate_biological_signature(droplet),
    };

    let proposal = ProposedBlock {
        proposer: droplet.droplet_id.clone(),
        dna_sequence: latest_synthesis.sequence_added.clone(),
        synthesis_proof,
        parent_hash: parent_hash.to_string(),
        timestamp: Utc::now(),
    };

    debug!("📝 Created block proposal from droplet: {}", droplet.droplet_id);
    Ok(proposal)
}

/// Calculate unique biological signature for droplet
fn calculate_biological_signature(droplet: &DropletNode) -> String {
    // Simple signature based on droplet state
    let signature_data = format!(
        "{}:{}:{}:{:.2}:{:.2}",
        droplet.droplet_id,
        droplet.dna_data.latest_block_hash,
        droplet.dna_data.chain_length,
        droplet.energy_level,
        droplet.size_nanoliters
    );
    
    // Simple hash (in reality would use proper cryptographic hash)
    format!("{:08x}", signature_data.len() * 1337 + droplet.dna_data.chain_length * 42)
}

/// Verify biological synthesis proof
pub fn verify_synthesis_proof(
    proof: &SynthesisProof,
    droplet: &DropletNode
) -> bool {
    // Verify energy expenditure is realistic
    if proof.energy_expended < 0.1 || proof.energy_expended > 1.0 {
        warn!("❌ Invalid energy expenditure in synthesis proof: {}", proof.energy_expended);
        return false;
    }

    // Verify synthesis time is realistic
    if proof.synthesis_time_ms < 50 || proof.synthesis_time_ms > 10000 {
        warn!("❌ Invalid synthesis time in proof: {} ms", proof.synthesis_time_ms);
        return false;
    }

    // Verify biological signature
    let expected_signature = calculate_biological_signature(droplet);
    if proof.biological_signature != expected_signature {
        warn!("❌ Biological signature mismatch");
        return false;
    }

    true
}

/// Cast vote for block proposal
pub fn cast_vote(
    voter_droplet: &DropletNode,
    proposed_block: &ProposedBlock,
    config: &ConsensusConfig
) -> Result<Vote> {
    // Verify voter has sufficient DNA mass
    if voter_droplet.dna_data.total_mass_picograms < config.mass_threshold_pg {
        return Err(anyhow::anyhow!("Insufficient DNA mass to vote"));
    }

    // Vote weight based on DNA mass
    let vote_weight = voter_droplet.dna_data.total_mass_picograms / 100.0;
    
    // Basic biological verification (simplified)
    let biological_verification = verify_synthesis_proof(
        &proposed_block.synthesis_proof,
        voter_droplet
    );

    let vote = Vote {
        voter: voter_droplet.droplet_id.clone(),
        block_hash: format!("{:08x}", proposed_block.dna_sequence.len()),
        vote_weight,
        biological_verification,
        timestamp: Utc::now(),
    };

    debug!("🗳️ Cast vote from {} with weight {:.2}", 
           voter_droplet.droplet_id, vote_weight);
    
    Ok(vote)
}

/// Finalize consensus round and determine winner
pub fn finalize_consensus(
    round: &mut ConsensusRound,
    _droplets: &HashMap<String, DropletNode>
) -> Result<Option<ProposedBlock>> {
    if round.status != ConsensusStatus::Voting {
        return Err(anyhow::anyhow!("Round not in voting state"));
    }

    // Tally votes by block hash
    let mut vote_tallies: HashMap<String, f64> = HashMap::new();
    let mut verified_votes = 0;
    
    for vote in round.votes.values() {
        if vote.biological_verification {
            *vote_tallies.entry(vote.block_hash.clone()).or_insert(0.0) += vote.vote_weight;
            verified_votes += 1;
        }
    }

    // Need majority of verified votes
    if verified_votes < round.participating_droplets.len() / 2 + 1 {
        round.status = ConsensusStatus::Failed;
        warn!("❌ Consensus failed: insufficient verified votes");
        return Ok(None);
    }

    // Find winning block
    if let Some((winning_hash, _)) = vote_tallies
        .iter()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
    {
        // Find the winning block proposal
        let winning_block = round
            .proposed_blocks
            .values()
            .find(|block| {
                let block_hash = format!("{:08x}", block.dna_sequence.len());
                &block_hash == winning_hash
            })
            .cloned();

        round.status = ConsensusStatus::Completed;
        
        if let Some(ref block) = winning_block {
            info!("🏆 Consensus completed! Winning block from: {}", block.proposer);
        }
        
        Ok(winning_block)
    } else {
        round.status = ConsensusStatus::Failed;
        Ok(None)
    }
}

/// Update consensus state after successful round
pub async fn update_consensus_state(
    consensus: &mut BiologicalConsensus,
    winning_block: &ProposedBlock,
    droplets: &HashMap<String, DropletNode>
) -> Result<()> {
    consensus.total_network_dna_mass = dna_storage::calculate_total_dna_mass(droplets);
    consensus.heaviest_swarm_leader = dna_storage::find_heaviest_droplet(droplets);
    consensus.last_consensus_round = Utc::now();
    
    // Calculate confidence based on participation and verification
    consensus.consensus_confidence = 0.95; // Simplified confidence calculation
    
    info!("📊 Updated consensus state: leader={}, DNA mass={:.2} pg", 
          consensus.heaviest_swarm_leader, consensus.total_network_dna_mass);
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Position2D, DNASynthesisEvent};

    fn create_test_droplet(id: &str, dna_mass: f64) -> DropletNode {
        DropletNode {
            droplet_id: id.to_string(),
            position: Position2D { x: 0.0, y: 0.0, velocity_x: 0.0, velocity_y: 0.0 },
            dna_data: DNABlockchain {
                chain_length: 1,
                genesis_hash: "test".to_string(),
                latest_block_hash: "test".to_string(),
                total_mass_picograms: dna_mass,
                synthesis_history: vec![
                    DNASynthesisEvent {
                        block_height: 0,
                        sequence_added: "ATGC".to_string(),
                        synthesis_time_ms: 100,
                        energy_cost: 0.5,
                        synthesized_at: Utc::now(),
                    }
                ],
            },
            energy_level: 1.0,
            size_nanoliters: 10.0,
        }
    }

    #[test]
    fn test_leader_selection() {
        let mut droplets = HashMap::new();
        droplets.insert("droplet1".to_string(), create_test_droplet("droplet1", 10.0));
        droplets.insert("droplet2".to_string(), create_test_droplet("droplet2", 20.0)); // Heaviest
        droplets.insert("droplet3".to_string(), create_test_droplet("droplet3", 15.0));
        
        let config = ConsensusConfig::default();
        let leader = select_consensus_leader(&droplets, &config).unwrap();
        
        assert_eq!(leader, "droplet2"); // Should select heaviest
    }

    #[tokio::test]
    async fn test_consensus_round() {
        let mut droplets = HashMap::new();
        droplets.insert("droplet1".to_string(), create_test_droplet("droplet1", 10.0));
        droplets.insert("droplet2".to_string(), create_test_droplet("droplet2", 15.0));
        droplets.insert("droplet3".to_string(), create_test_droplet("droplet3", 12.0));
        
        let config = ConsensusConfig::default();
        let round = start_consensus_round(&droplets, &config, 1).await.unwrap();
        
        assert_eq!(round.round_id, 1);
        assert_eq!(round.status, ConsensusStatus::Proposing);
        assert!(round.leader_droplet.is_some());
        assert_eq!(round.participating_droplets.len(), 3);
    }

    #[test]
    fn test_biological_signature() {
        let droplet = create_test_droplet("test", 10.0);
        let sig1 = calculate_biological_signature(&droplet);
        let sig2 = calculate_biological_signature(&droplet);
        
        assert_eq!(sig1, sig2); // Should be deterministic
        assert!(!sig1.is_empty());
    }
}