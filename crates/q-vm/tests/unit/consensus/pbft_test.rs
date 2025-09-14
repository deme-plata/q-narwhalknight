use dagknight_vm::consensus::pbft::{PbftConsensus, Block, PrepareRequest, PrepareResponse, CommitRequest, CommitResponse};
use dagknight_vm::transaction::Transaction;
use tokio::sync::mpsc;
use std::time::{SystemTime, UNIX_EPOCH};

#[tokio::test]
async fn test_pbft_initialize() {
    // Create a simple consensus configuration
    let node_id = "node-0".to_string();
    let peers = vec!["node-1".to_string(), "node-2".to_string(), "node-3".to_string()];
    
    // Create a PBFT consensus instance
    let consensus = PbftConsensus::new(node_id, peers);
    
    // Check initial state
    assert_eq!(0, consensus.get_latest_finalized().await);
}

#[tokio::test]
async fn test_block_proposal_and_finalization() {
    // Create a mock network channel
    let (tx, mut rx) = mpsc::channel(100);
    
    // Create a simple consensus configuration
    let node_id = "node-0".to_string();
    let peers = vec!["node-1".to_string(), "node-2".to_string(), "node-3".to_string()];
    
    // Create a PBFT consensus instance with the mock network
    let mut consensus = PbftConsensus::new(node_id.clone(), peers.clone());
    
    // Create a test transaction
    let tx_hash = [1u8; 32];
    let tx = Transaction {
        hash: tx_hash,
        data: vec![1, 2, 3],
        sender: [0; 32],
        nonce: 0,
        signature: [0; 64],
        timestamp: SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs(),
    };
    
    // Create a block with the transaction
    let parent_hash = [0u8; 32]; // Genesis block hash
    let transactions = vec![tx];
    
    // Propose block (this should trigger PrepareRequest)
    let block_hash = consensus.propose_block(parent_hash, transactions.clone()).await.unwrap();
    
    // Simulate responses from other nodes
    
    // 1. PrepareResponse from node-1
    let prepare_response1 = PrepareResponse {
        view: 0,
        seq_num: 0,
        block_hash,
        node_id: "node-1".to_string(),
        timestamp: SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs(),
    };
    consensus.handle_prepare_response(prepare_response1).await;
    
    // 2. PrepareResponse from node-2
    let prepare_response2 = PrepareResponse {
        view: 0,
        seq_num: 0,
        block_hash,
        node_id: "node-2".to_string(),
        timestamp: SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs(),
    };
    consensus.handle_prepare_response(prepare_response2).await;
    
    // 3. CommitResponse from node-1
    let commit_response1 = CommitResponse {
        view: 0,
        seq_num: 0,
        block_hash,
        node_id: "node-1".to_string(),
        timestamp: SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs(),
    };
    consensus.handle_commit_response(commit_response1).await;
    
    // 4. CommitResponse from node-2
    let commit_response2 = CommitResponse {
        view: 0,
        seq_num: 0,
        block_hash,
        node_id: "node-2".to_string(),
        timestamp: SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs(),
    };
    consensus.handle_commit_response(commit_response2).await;
    
    // Check if block was finalized
    assert_eq!(0, consensus.get_latest_finalized().await);
    
    // Get the finalized block
    let block = consensus.get_finalized_block(0).await.unwrap();
    assert_eq!(block_hash, block.hash);
    assert_eq!(1, block.transactions.len());
    assert_eq!(tx_hash, block.transactions[0].hash);
}

#[tokio::test]
async fn test_view_change() {
    // Create a simple consensus configuration
    let node_id = "node-0".to_string();
    let peers = vec!["node-1".to_string(), "node-2".to_string(), "node-3".to_string()];
    
    // Create a PBFT consensus instance
    let mut consensus = PbftConsensus::new(node_id.clone(), peers.clone());
    
    // Trigger view change
    let view_change = consensus.create_view_change(1).await;
    
    // Simulate view changes from other nodes
    let view_change1 = consensus.create_view_change_for_test("node-1".to_string(), 1).await;
    let view_change2 = consensus.create_view_change_for_test("node-2".to_string(), 1).await;
    
    // Handle view changes
    consensus.handle_view_change(view_change1).await;
    consensus.handle_view_change(view_change2).await;
    
    // Check if view was updated
    assert_eq!(1, consensus.get_current_view().await);
}
