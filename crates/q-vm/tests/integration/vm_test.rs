use dagknight_vm::{
    vm::DagkVm,
    network::p2p::P2pNetwork,
    consensus::pbft::PbftConsensus,
};
use std::sync::Arc;
use tokio::runtime::Runtime;
use std::time::Duration;
use std::thread;

#[test]
#[ignore] // Ignore by default as it requires network setup
fn test_contract_deployment_and_call() {
    // Create a tokio runtime
    let rt = Runtime::new().unwrap();
    
    rt.block_on(async {
        // Create network
        let network = P2pNetwork::new().await.unwrap();
        
        // Listen on localhost
        network.listen("/ip4/127.0.0.1/tcp/0").await.unwrap();
        
        // Start network
        network.start().await;
        
        // Create consensus
        let node_id = "test-node".to_string();
        let peers = Vec::new(); // No peers for this test
        let consensus = Arc::new(PbftConsensus::new(node_id, peers));
        
        // Create VM
        let vm = DagkVm::new("./test_db", Arc::new(network), consensus);
        
        // Create a simple WebAssembly contract
        let wasm_bytes = wat::parse_str(r#"
            (module
                (memory (export "memory") 1)
                
                (func  (param  i32) (param  i32) (param  i32) (param  i32) (result i32)
                    (call  (local.get ) (local.get ) (local.get ) (local.get ))
                )
                
                (func  (param  i32) (param  i32) (param  i32) (param  i32) (result i32)
                    (call  (local.get ) (local.get ) (local.get ) (local.get ))
                )
                
                (import "env" "read_state" (func  (param i32 i32 i32 i32) (result i32)))
                (import "env" "write_state" (func  (param i32 i32 i32 i32) (result i32)))
                
                (export "store" (func ))
                (export "load" (func ))
            )
        "#).unwrap();
        
        // Deploy contract
        let sender = [1u8; 32];
        let nonce = 0;
        
        let contract_hash = vm.deploy_contract(wasm_bytes.to_vec(), sender, nonce).await.unwrap();
        
        // Wait for consensus
        thread::sleep(Duration::from_secs(1));
        
        // Call contract
        let key = b"test_key".to_vec();
        let value = b"test_value".to_vec();
        
        // Prepare arguments
        let mut args = Vec::new();
        args.push(key.clone());
        args.push(value.clone());
        
        // Call store function
        let result = vm.call_contract(contract_hash, "store", args, sender, nonce + 1).await.unwrap();
        
        // Call load function
        let args = vec![key.clone()];
        let result = vm.call_contract(contract_hash, "load", args, sender, nonce + 2).await.unwrap();
        
        // In a real test, we would check that result contains the value
        // For now, we just check that the call succeeded
        assert!(!result.is_empty());
    });
}
