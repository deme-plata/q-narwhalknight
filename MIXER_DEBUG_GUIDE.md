# Mixer Debugging Guide

## Date: 2025-10-02

## Issue: "Insufficient balance for private transaction"

### ✅ FIXES APPLIED

All fixes are now in place in `crates/q-api-server/src/handlers.rs`:

1. **Line 3024**: Correct `total_cost` calculation
   ```rust
   let amount_u64 = (request.amount * 100_000_000.0) as u64;
   let mixer_fee = amount_u64 / 1000; // 0.1% mixing fee
   let total_cost = amount_u64 + mixer_fee;
   ```

2. **Line 3063**: Removed duplicate definition
   ```rust
   // Note: total_cost is already defined above at line 3024
   ```

3. **Lines 3146-3157**: Improved error message with exact amounts
   ```rust
   let balance_qnk = sender_balance as f64 / 100_000_000.0;
   let needed_qnk = total_cost as f64 / 100_000_000.0;
   let shortage_qnk = (total_cost - sender_balance) as f64 / 100_000_000.0;

   let error_msg = format!(
       "Insufficient balance for private transaction. Need {:.8} QNK (including 0.1% mixer fee), but only have {:.8} QNK. Short by {:.8} QNK.",
       needed_qnk, balance_qnk, shortage_qnk
   );
   ```

### 🚀 HOW TO TEST THE FIX

#### 1. Restart the API Server (REQUIRED)
```bash
# Stop any running server
killall q-api-server 2>/dev/null

# Build with 10-hour timeout (CRITICAL for quantum consensus)
timeout 36000 cargo build --release --package q-api-server

# Start server with debug logging to see mixer activity
RUST_LOG=debug ./target/x86_64-unknown-linux-gnu/release/q-api-server --port 8080
```

#### 2. Check Your Wallet Balance
```bash
# Get all wallet balances
curl -s http://localhost:8080/api/balance | jq

# Convert to QNK format (divide by 100,000,000)
curl -s http://localhost:8080/api/balance | jq -r '.data | to_entries[] | "\(.key): \(.value / 100000000) QNK"'
```

#### 3. Test Mixer with Different Scenarios

**Test Case A: Small Transaction (0.1 QNK)**
```bash
curl -X POST http://localhost:8080/api/privacy-mixer \
  -H "Content-Type: application/json" \
  -d '{
    "from": "YOUR_WALLET_ADDRESS",
    "to": "RECIPIENT_ADDRESS",
    "amount": 0.1,
    "privacy_level": "high"
  }'

# Required balance: 0.1001 QNK (0.1 + 0.0001 fee)
```

**Test Case B: Medium Transaction (10 QNK)**
```bash
curl -X POST http://localhost:8080/api/privacy-mixer \
  -H "Content-Type: application/json" \
  -d '{
    "from": "YOUR_WALLET_ADDRESS",
    "to": "RECIPIENT_ADDRESS",
    "amount": 10.0,
    "privacy_level": "high",
    "decoy_multiplier": 15.0
  }'

# Required balance: 10.01 QNK (10.0 + 0.01 fee)
```

**Test Case C: Large Transaction (100 QNK)**
```bash
curl -X POST http://localhost:8080/api/privacy-mixer \
  -H "Content-Type: application/json" \
  -d '{
    "from": "YOUR_WALLET_ADDRESS",
    "to": "RECIPIENT_ADDRESS",
    "amount": 100.0,
    "privacy_level": "maximum",
    "decoy_multiplier": 50.0,
    "enable_quantum_mixing": true
  }'

# Required balance: 100.1 QNK (100.0 + 0.1 fee)
```

#### 4. Debug Output to Look For

**Success Case:**
```
🔍 MIXER DEBUG: Sender balance check - address: 6a7c86d88326..., balance: 10010000000, total_cost: 10010000000
✅ MIXER DEBUG: Balance deducted successfully, new balance: 0
🎭 Privacy mixer processing transaction for 10.00 QNK
```

**Insufficient Balance Case (NEW ERROR MESSAGE):**
```
🔍 MIXER DEBUG: Sender balance check - address: 6a7c86d88326..., balance: 5000000000, total_cost: 10010000000
❌ MIXER DEBUG: Insufficient balance - needed: 10010000000, available: 5000000000
Error: "Insufficient balance for private transaction. Need 10.01000000 QNK (including 0.1% mixer fee), but only have 5.00000000 QNK. Short by 5.01000000 QNK."
```

### 📊 FEE CALCULATION REFERENCE

| Transaction Amount | Mixer Fee (0.1%) | Total Cost | Atomic Units |
|-------------------|------------------|------------|--------------|
| 0.1 QNK | 0.0001 QNK | 0.1001 QNK | 10,010,000 |
| 1 QNK | 0.001 QNK | 1.001 QNK | 100,100,000 |
| 10 QNK | 0.01 QNK | 10.01 QNK | 1,001,000,000 |
| 100 QNK | 0.1 QNK | 100.1 QNK | 10,010,000,000 |
| 1000 QNK | 1 QNK | 1001 QNK | 100,100,000,000 |

**Formula:**
- Fee = Amount / 1000 (0.1%)
- Total Cost = Amount + Fee
- Atomic Units = QNK × 100,000,000

### 🔍 TROUBLESHOOTING

#### Problem: Still getting "Insufficient balance" error

**Step 1: Verify server is running with the fix**
```bash
# Check if server process exists
ps aux | grep q-api-server

# Check server logs
tail -f /tmp/q-api-server.log  # or wherever your logs are
```

**Step 2: Verify your actual balance**
```bash
# Get balance for specific wallet
WALLET="YOUR_WALLET_ADDRESS"
curl -s http://localhost:8080/api/balance | jq -r ".data.\"$WALLET\" / 100000000"
```

**Step 3: Check the exact error message**
- Old error: "Insufficient balance for private transaction"
- New error: "Insufficient balance for private transaction. Need X.XXXXXXXX QNK (including 0.1% mixer fee), but only have Y.YYYYYYYY QNK. Short by Z.ZZZZZZZZ QNK."

If you see the old error message, the server hasn't been restarted with the fix.

**Step 4: Verify the fee calculation**
```bash
# For a 10 QNK transaction:
# Amount in atomic units: 10 * 100,000,000 = 1,000,000,000
# Fee: 1,000,000,000 / 1000 = 1,000,000 (0.01 QNK)
# Total: 1,000,000,000 + 1,000,000 = 1,001,000,000 (10.01 QNK)

python3 -c "amount = 10.0; fee = amount * 0.001; print(f'Amount: {amount} QNK, Fee: {fee} QNK, Total: {amount + fee} QNK')"
```

### 💡 COMMON SCENARIOS

#### Scenario 1: Wallet Has Exact Amount But No Fee
```
Balance: 10.00 QNK
Transaction: 10.00 QNK
Fee: 0.01 QNK
Total Needed: 10.01 QNK
Result: ❌ Insufficient balance (short by 0.01 QNK)
```

**Solution**: Either reduce transaction amount to 9.99 QNK or add more funds

#### Scenario 2: Using Fallback Address (No "from" field)
```
Request: { "to": "...", "amount": 10.0 }
Behavior: Server finds wallet with sufficient balance automatically
Warning: "No sender address provided in privacy mixer request - using fallback"
```

**Recommendation**: Always specify "from" field to avoid unexpected source wallet

#### Scenario 3: Multiple Wallets, Wrong One Used
```
Wallet A: 100 QNK ✅
Wallet B: 5 QNK ❌
Request uses Wallet B: ❌ Fails
```

**Solution**: Verify the "from" address in your request matches the wallet with funds

### 🎯 EXPECTED BEHAVIOR AFTER FIX

1. ✅ Error message shows exact QNK amounts (not just generic message)
2. ✅ Shows required amount including 0.1% fee
3. ✅ Shows actual balance available
4. ✅ Shows shortage amount
5. ✅ Debug logs show address, balance in atomic units, and total_cost

### 📝 RELATED FILES

- **handlers.rs**: Lines 3020-3160 (mixer implementation)
- **MIXER_BALANCE_FIX.md**: Original fix documentation
- **MIXER_DEBUG_GUIDE.md**: This debugging guide

### 🔐 PRIVACY LEVELS

**Standard** (5-10 decoys):
- Basic ring signatures
- Stealth address generation
- Fastest mixing time

**High** (15-20 decoys):
- Enhanced ring signatures (16 members)
- Quantum entropy for stealth keys
- Dandelion++ gossip
- Recommended for most users

**Maximum** (30-50 decoys):
- Maximum ring size
- Full quantum-enhanced privacy
- Extended mixing time
- ZK-STARK proofs
- Maximum anonymity

### ✅ SUCCESS CRITERIA

You'll know the fix is working when:

1. Error message shows exact QNK amounts (e.g., "Need 10.01000000 QNK")
2. Debug logs appear in server output with 🔍 and ✅/❌ emojis
3. Balance checks use the correct total_cost calculation
4. Transactions succeed when balance >= (amount + 0.1% fee)
5. Transactions fail with clear shortage information when insufficient

---

**Status**: ✅ All fixes applied, ready for testing after server restart

**Next Step**: Restart q-api-server with the rebuilt binary to apply the fix
