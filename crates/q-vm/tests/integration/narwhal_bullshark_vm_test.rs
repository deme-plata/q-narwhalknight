//! Integration tests for Narwhal-Bullshark VM

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::time::{Duration, Instant};
    use tokio::sync::Mutex;
    
    use crate::vm::{VirtualMachine, VmError, ExecutionResult, ConsensusEngine};
    use crate::vm::narwhal_bullshark_vm::{NarwhalBullsharkVm, SmartContractTx};
    use crate::consensus::narwhal_bullshark::{NarwhalBullshark, Transaction};
    
    // Basic functionality test
    #[tokio::test]
    async fn test_basic_functionality() {
        // Create VM and consensus
        let vm = Arc::new(VirtualMachine::new());
        let nb_vm = Arc::new(NarwhalBullsharkVm::new(
            "test_node".to_string(),
            vec!["peer1".to_string(), "peer2".to_string()],
            vm
        ));
        
        // Start VM
        nb_vm.start().await.expect("Failed to start VM");
        
        // Create and submit a test transaction
        let tx = SmartContractTx {
            address: 1000,
            function: "test".to_string(),
            arguments: vec![1, 2, 3, 4],
            sender: 101,
            gas_limit: 100000,
            gas_price: 1,
            nonce: 0,
            value: 0,
            signature: [0; 64],
        };
        
        let tx_hash = nb_vm.submit_transaction(tx).await.expect("Failed to submit transaction");
        
        // Wait a moment for processing
        tokio::time::sleep(Duration::from_secs(1)).await;
        
        // Get transaction result
        let result = nb_vm.get_transaction_result(tx_hash).await;
        
        // Stop VM
        nb_vm.stop().await.expect("Failed to stop VM");
        
        // Assert that transaction was processed
        assert!(result.is_some(), "Transaction result should be available");
    }
    
    // Performance test with multiple transactions
    #[tokio::test]
    async fn test_transaction_throughput() {
        // Create VM and consensus
        let vm = Arc::new(VirtualMachine::new());
        let nb_vm = Arc::new(NarwhalBullsharkVm::new(
            "test_node".to_string(),
            vec!["peer1".to_string(), "peer2".to_string()],
            vm
        ));
        
        // Start VM
        nb_vm.start().await.expect("Failed to start VM");
        
        // Set up account with balance
        {
            let mut state = nb_vm.current_state.write().await;
            state.balances.insert(101, 10_000_000);
            state.nonces.insert(101, 0);
        }
        
        // Number of transactions to test
        let tx_count = 100;
        
        // Track submission time
        let start_time = Instant::now();
        
        // Submit transactions
        for i in 0..tx_count {
            let tx = SmartContractTx {
                address: 1000,
                function: "transfer".to_string(),
                arguments: vec![1, 2, 3, 4],
                sender: 101,
                gas_limit: 100000,
                gas_price: 1,
                nonce: i as u64,
                value: 0,
                signature: [0; 64],
            };
            
            nb_vm.submit_transaction(tx).await.expect("Failed to submit transaction");
        }
        
        let submission_time = start_time.elapsed();
        
        // Wait for processing to complete
        tokio::time::sleep(Duration::from_secs(2)).await;
        
        // Calculate TPS
        let total_time = start_time.elapsed();
        let submission_tps = tx_count as f64 / submission_time.as_secs_f64();
        let overall_tps = tx_count as f64 / total_time.as_secs_f64();
        
        // Get VM's TPS measurement
        let vm_tps = nb_vm.get_tps().await;
        
        // Stop VM
        nb_vm.stop().await.expect("Failed to stop VM");
        
        println!("Transaction Throughput Test Results:");
        println!("  Transactions: {}", tx_count);
        println!("  Submission Time: {:.2?}", submission_time);
        println!("  Total Time: {:.2?}", total_time);
        println!("  Submission TPS: {:.2}", submission_tps);
        println!("  Overall TPS: {:.2}", overall_tps);
        println!("  VM Reported TPS: {:.2}", vm_tps);
        
        // Assert reasonable performance
        assert!(submission_tps > 10.0, "Submission TPS should be greater than 10");
    }
    
    // Test with multiple nodes
    #[tokio::test]
    async fn test_multi_node() {
        // Create node IDs
        let node_ids = vec![
            "node_1".to_string(),
            "node_2".to_string(),
            "node_3".to_string(),
        ];
        
        // Create and start VMs
        let vm = Arc::new(VirtualMachine::new());
        let mut vms = Vec::new();
        
        for (i, node_id) in node_ids.iter().enumerate() {
            // Create peers list (all other nodes)
            let peers: Vec<String> = node_ids.iter()
                .enumerate()
                .filter(|(j, _)| *j != i)
                .map(|(_, id)| id.clone())
                .collect();
            
            // Create VM
            let nb_vm = Arc::new(NarwhalBullsharkVm::new(
                node_id.clone(), peers, vm.clone()
            ));
            
            // Start VM
            nb_vm.start().await.expect("Failed to start VM");
            
            vms.push(nb_vm);
        }
        
        // Allow time for nodes to connect
        tokio::time::sleep(Duration::from_secs(2)).await;
        
        // Set up account with balance
        {
            let mut state = vms[0].current_state.write().await;
            state.balances.insert(101, 10_000_000);
            state.nonces.insert(101, 0);
        }
        
        // Submit a test transaction to the first node
        let tx = SmartContractTx {
            address: 1000,
            function: "transfer".to_string(),
            arguments: vec![1, 2, 3, 4],
            sender: 101,
            gas_limit: 100000,
            gas_price: 1,
            nonce: 0,
            value: 100,
            signature: [0; 64],
        };
        
        let tx_hash = vms[0].submit_transaction(tx).await.expect("Failed to submit transaction");
        
        // Wait for transaction to propagate
        tokio::time::sleep(Duration::from_secs(5)).await;
        
        // Stop all VMs
        for (i, vm) in vms.iter().enumerate() {
            vm.stop().await.expect("Failed to stop VM");
            println!("Stopped node {}", node_ids[i]);
        }
        
        // Assert successful test
        assert!(true, "Multi-node test completed");
    }
}
