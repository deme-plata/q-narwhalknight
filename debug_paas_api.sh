#!/bin/bash

echo "=== PaaS API Key Generation Diagnostics ==="
echo

echo "1. API Server Status:"
ss -tlnp | grep :8080 || echo "  ❌ Not running on port 8080"
echo

echo "2. API Endpoint Test:"
curl -s -X POST http://localhost:8080/api/v1/privacy/paas/api-keys/generate \
  -H "Content-Type: application/json" \
  -d '{"wallet_address":"debug_test","tier":"free","expires_days":90}' | python3 -m json.tool 2>/dev/null || echo "Response not JSON"
echo

echo "3. Frontend Processes:"
ss -tlnp | grep -E ':(3000|3001|5173|8000)' || echo "  ℹ No dev server detected"
echo

echo "4. CORS Configuration:"
grep -n "CorsLayer" crates/q-api-server/src/main.rs | head -5
echo

echo "5. Route Registration:"
grep -n "/api-keys/generate" crates/q-api-server/src/paas_admin_api.rs
echo

echo "=== Diagnosis Complete ==="
