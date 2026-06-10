use dagknight_vm::mempool::Mempool;
use dagknight_vm::transaction::Transaction;
use std::time::{SystemTime, UNIX_EPOCH};

#[test]
fn test_add_and_get_transaction() {
    // Create a mempool
    let mempool = Mempool::new(100, 1);
    
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
    
    // Add transaction to mempool
    mempool.add_transaction(tx.clone(), 10).unwrap();
    
    // Get best transactions
    let txs = mempool.get_best_transactions(10);
    
    // Check if transaction was returned
    assert_eq!(1, txs.len());
    assert_eq!(tx_hash, txs[0].hash);
}

#[test]
fn test_remove_transaction() {
    // Create a mempool
    let mempool = Mempool::new(100, 1);
    
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
    
    // Add transaction to mempool
    mempool.add_transaction(tx.clone(), 10).unwrap();
    
    // Remove transaction
    let removed_tx = mempool.remove_transaction(&tx_hash).unwrap();
    
    // Check if transaction was removed
    assert_eq!(tx_hash, removed_tx.hash);
    assert_eq!(0, mempool.get_transaction_count());
}

#[test]
fn test_transaction_ordering_by_gas_price() {
    // Create a mempool
    let mempool = Mempool::new(100, 1);
    
    // Create test transactions with different gas prices
    let tx1_hash = [1u8; 32];
    let tx1 = Transaction {
        hash: tx1_hash,
        data: vec![1, 2, 3],
        sender: [0; 32],
        nonce: 0,
        signature: [0; 64],
        timestamp: SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs(),
    };
    
    let tx2_hash = [2u8; 32];
    let tx2 = Transaction {
        hash: tx2_hash,
        data: vec![4, 5, 6],
        sender: [0; 32],
        nonce: 1,
        signature: [0; 64],
        timestamp: SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs(),
    };
    
    // Add transactions with different gas prices
    mempool.add_transaction(tx1.clone(), 5).unwrap();
    mempool.add_transaction(tx2.clone(), 10).unwrap();
    
    // Get best transactions (should be ordered by gas price)
    let txs = mempool.get_best_transactions(10);
    
    // Check if transactions are ordered correctly
    assert_eq!(2, txs.len());
    assert_eq!(tx2_hash, txs[0].hash); // Higher gas price, should be first
    assert_eq!(tx1_hash, txs[1].hash);
}

#[test]
fn test_get_sender_transactions() {
    // Create a mempool
    let mempool = Mempool::new(100, 1);
    
    // Create sender addresses
    let sender1 = [1u8; 32];
    let sender2 = [2u8; 32];
    
    // Create test transactions for different senders
    let tx1_hash = [1u8; 32];
    let tx1 = Transaction {
        hash: tx1_hash,
        data: vec![1, 2, 3],
        sender: sender1,
        nonce: 0,
        signature: [0; 64],
        timestamp: SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs(),
    };
    
    let tx2_hash = [2u8; 32];
    let tx2 = Transaction {
        hash: tx2_hash,
        data: vec![4, 5, 6],
        sender: sender1,
        nonce: 1,
        signature: [0; 64],
        timestamp: SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs(),
    };
    
    let tx3_hash = [3u8; 32];
    let tx3 = Transaction {
        hash: tx3_hash,
        data: vec![7, 8, 9],
        sender: sender2,
        nonce: 0,
        signature: [0; 64],
        timestamp: SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs(),
    };
    
    // Add transactions to mempool
    mempool.add_transaction(tx1.clone(), 5).unwrap();
    mempool.add_transaction(tx2.clone(), 5).unwrap();
    mempool.add_transaction(tx3.clone(), 5).unwrap();
    
    // Get transactions for sender1
    let sender1_txs = mempool.get_sender_transactions(&sender1);
    
    // Check if we got the right transactions
    assert_eq!(2, sender1_txs.len());
    
    // Get transactions for sender2
    let sender2_txs = mempool.get_sender_transactions(&sender2);
    
    // Check if we got the right transactions
    assert_eq!(1, sender2_txs.len());
    assert_eq!(tx3_hash, sender2_txs[0].hash);
}
