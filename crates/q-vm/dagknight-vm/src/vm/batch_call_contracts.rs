    // Execute a batch of contract calls
    pub async fn batch_call_contracts(&self, calls: Vec<ContractCall>) -> Vec<ContractResult> {
        use std::collections::HashMap;
        
        // Prepare the calls with their contracts
        let mut call_with_contracts = Vec::new();
        
        for call in calls {
            if let Some(contract) = self.contract_registry.get(&call.contract_address) {
                call_with_contracts.push((call, Arc::new(contract.clone())));
            } else {
                // Contract not found, return an error for this call
                // For now, we'll just skip it
                continue;
            }
        }
        
        // Execute in parallel
        let batch_result = self.parallel_executor.execute_batch(call_with_contracts).await;
        let total_gas_used = batch_result.gas_used;
        
        // Clone the results to avoid using after move
        let results_clone = batch_result.results.clone();
        
        // Convert results to ContractResult objects
        let results: Vec<ContractResult> = results_clone.into_iter()
            .enumerate()
            .map(|(_idx, result)| {
                match result {
                    Ok(data) => ContractResult {
                        success: true,
                        return_data: data,
                        error: None,
                        gas_used: total_gas_used / batch_result.results.len() as u64, // Average
                        state_changes: HashMap::new(), // Simplified
                        logs: Vec::new(), // Simplified
                    },
                    Err(e) => ContractResult {
                        success: false,
                        return_data: Vec::new(),
                        error: Some(e.to_string()),
                        gas_used: total_gas_used / batch_result.results.len() as u64, // Average
                        state_changes: HashMap::new(),
                        logs: Vec::new(),
                    }
                }
            })
            .collect();
            
        results
    }
