#!/bin/bash
cd /opt/orobit/shared/q-narwhalknight

# Fix field names in quillon_bank_api.rs
sed -i 's/metrics\.total_qnkusd_supply/metrics.qnkusd_metrics.total_supply/g' crates/q-api-server/src/quillon_bank_api.rs
sed -i 's/metrics\.total_collateral_value/total_deposits_value/g' crates/q-api-server/src/quillon_bank_api.rs
sed -i 's/metrics\.collateralization_ratio/metrics.qnkusd_metrics.collateral_ratio/g' crates/q-api-server/src/quillon_bank_api.rs
sed -i 's/metrics\.total_loans_outstanding/total_loans_value/g' crates/q-api-server/src/quillon_bank_api.rs
sed -i 's/metrics\.reserve_ratio/metrics.qnkusd_metrics.collateral_ratio/g' crates/q-api-server/src/quillon_bank_api.rs
sed -i 's/metrics\.quantum_vaults_active/metrics.quantum_metrics.total_quantum_vaults/g' crates/q-api-server/src/quillon_bank_api.rs
sed -i 's/metrics\.pq_transactions_24h/metrics.quantum_metrics.post_quantum_transactions_24h/g' crates/q-api-server/src/quillon_bank_api.rs
sed -i 's/metrics\.quantum_privacy_adoption_rate/metrics.quantum_metrics.quantum_privacy_adoption/g' crates/q-api-server/src/quillon_bank_api.rs

echo "Fixed field names"