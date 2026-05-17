use dagknight_vm::state::{StateDB, StateBatch, compute_state_root, StateTransition};
use std::sync::Arc;

#[test]
fn test_in_memory_state_db() {
    // Create in-memory state DB
    let state_db = StateDB::new_in_memory();
    
    // Store a key-value pair
    let key = b"test_key".to_vec();
    let value = b"test_value".to_vec();
    state_db.put(key.clone(), value.clone());
    
    // Retrieve the value
    let retrieved_value = state_db.get(&key).unwrap();
    assert_eq!(value, retrieved_value);
}

#[test]
fn test_state_batch() {
    // Create in-memory state DB
    let state_db = StateDB::new_in_memory();
    
    // Create a state batch
    let mut batch = StateBatch::new();
    
    // Add some operations to the batch
    let key1 = b"key1".to_vec();
    let value1 = b"value1".to_vec();
    batch.put(key1.clone(), value1.clone());
    
    let key2 = b"key2".to_vec();
    let value2 = b"value2".to_vec();
    batch.put(key2.clone(), value2.clone());
    
    // Commit the batch
    state_db.commit_batch(batch).unwrap();
    
    // Check if values were stored
    assert_eq!(value1, state_db.get(&key1).unwrap());
    assert_eq!(value2, state_db.get(&key2).unwrap());
}

#[test]
fn test_state_transition() {
    // Create in-memory state DB
    let state_db = Arc::new(StateDB::new_in_memory());
    
    // Initialize state with some values
    state_db.put(b"key1".to_vec(), b"value1".to_vec());
    state_db.put(b"key2".to_vec(), b"value2".to_vec());
    
    // Create a state transition
    let mut transition = StateTransition::new(state_db.clone());
    
    // Apply some changes
    transition.apply(b"key1".to_vec(), Some(b"new_value1".to_vec()));
    transition.apply(b"key3".to_vec(), Some(b"value3".to_vec()));
    transition.apply(b"key2".to_vec(), None); // Delete key2
    
    // Commit the transition
    let updated_state = transition.commit().unwrap();
    
    // Check if state was updated correctly
    assert_eq!(b"new_value1".to_vec(), updated_state.get(b"key1").unwrap());
    assert!(updated_state.get(b"key2").is_none());
    assert_eq!(b"value3".to_vec(), updated_state.get(b"key3").unwrap());
}

#[test]
fn test_state_root_calculation() {
    // Create in-memory state DB
    let state_db = StateDB::new_in_memory();
    
    // Initialize state with some values
    state_db.put(b"key1".to_vec(), b"value1".to_vec());
    state_db.put(b"key2".to_vec(), b"value2".to_vec());
    
    // Calculate state root
    let root1 = compute_state_root(&state_db);
    
    // Modify state
    state_db.put(b"key3".to_vec(), b"value3".to_vec());
    
    // Calculate state root again
    let root2 = compute_state_root(&state_db);
    
    // Roots should be different
    assert_ne!(root1, root2);
    
    // Create a new state with the same values
    let state_db2 = StateDB::new_in_memory();
    state_db2.put(b"key1".to_vec(), b"value1".to_vec());
    state_db2.put(b"key2".to_vec(), b"value2".to_vec());
    state_db2.put(b"key3".to_vec(), b"value3".to_vec());
    
    // Calculate state root
    let root3 = compute_state_root(&state_db2);
    
    // Should match the second root
    assert_eq!(root2, root3);
}
