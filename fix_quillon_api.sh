#!/bin/bash
# Fix all method calls in quillon_bank_api.rs

cd /opt/orobit/shared/q-narwhalknight

# Replace get_metrics with get_bank_metrics
sed -i 's/bank_system\.get_metrics()/bank_system.get_bank_metrics()/g' crates/q-api-server/src/quillon_bank_api.rs

echo "Fixed method calls in quillon_bank_api.rs"