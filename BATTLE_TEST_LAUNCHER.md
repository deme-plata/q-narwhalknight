# 🚀 Q-NarwhalKnight Battle Test Launcher

## Coordinated Server Alpha ↔ Server Beta Testing

This document provides the exact commands for **Server Alpha** and **Server Beta** to launch the comprehensive battle test of FREE peer discovery methods.

### 🏁 **BATTLE TEST EXECUTION PLAN**

#### **Phase 1: Server Alpha Startup (3 minutes)**

**Server Alpha - Execute these commands:**

```bash
# 1. Navigate to Q-NarwhalKnight directory
cd /mnt/orobit-shared/q-narwhalknight

# 2. Pull latest code
git pull origin main

# 3. Launch Server Alpha battle test
./scripts/battle_test_alpha.sh
```

**Expected Alpha Output:**
```
🔥 Starting Q-NarwhalKnight Battle Test - Server Alpha
✅ Environment configured - Server Alpha (Bootstrap Node)
✅ Build successful
✅ Tor connectivity verified
🧅 Onion service established: validator123abc456def789ghi012jkl345mno678pqr901stu234vwx.onion
✅ Discovery monitoring started
═══════════════════════════════════════════
🏆 SERVER ALPHA BATTLE TEST READY
═══════════════════════════════════════════
🧅 Onion Address: validator123abc456def789ghi012jkl345mno678pqr901stu234vwx.onion
🚪 Port: 8333
📊 STATUS: Peers: 0 | Cost: $0.00 | Uptime: 00:03
```

#### **Phase 2: Server Beta Startup (5 minutes after Alpha)**

**Server Beta - Execute these commands:**

```bash
# 1. Wait for Server Alpha to be ready (check /mnt/shared/alpha_onion_info.env exists)
ls -la /mnt/shared/alpha_onion_info.env

# 2. Navigate to Q-NarwhalKnight Beta directory  
cd /mnt/orobit-shared/q-narwhalknight

# 3. Pull latest code
git pull origin main

# 4. Launch Server Beta battle test
./scripts/battle_test_beta.sh
```

**Expected Beta Output:**
```
🔥 Starting Q-NarwhalKnight Battle Test - Server Beta
✅ Server Alpha ready! Onion: validator123abc456def789ghi012jkl345mno678pqr901stu234vwx.onion
✅ Environment configured - Server Beta (Validator Node)
✅ Build successful
✅ Successfully connected to Server Alpha
🧅 Onion service established: validatorxyz789uvw012abc345def678ghi901jkl234mno567pqr890.onion
═══════════════════════════════════════════
🏆 SERVER BETA BATTLE TEST READY
═══════════════════════════════════════════
🧅 Onion Address: validatorxyz789uvw012abc345def678ghi901jkl234mno567pqr890.onion
🎯 Target: validator123abc456def789ghi012jkl345mno678pqr901stu234vwx.onion:8333
🔍 Discovery progress: 0 peers found, cost: $0.00 (30s elapsed)
🔍 Discovery progress: 1 peers found, cost: $0.00 (60s elapsed)
🎉 SUCCESS: Discovered Server Alpha in 67s via FREE methods!
📊 STATUS: Peers: 1 | Found Alpha: ✅ | Cost: $0.00 | Uptime: 00:02
```

### 🎯 **SUCCESS CRITERIA**

The battle test is **SUCCESSFUL** when you see:

#### **On Server Alpha:**
```
📊 STATUS: Peers: 1 | Cost: $0.00 | Uptime: 00:05
✅ Server Beta discovered and connected
🆓 All discovery methods active - FREE operation maintained
```

#### **On Server Beta:**  
```
🎉 SUCCESS: Discovered Server Alpha in <60s via FREE methods!
📊 STATUS: Peers: 1 | Found Alpha: ✅ | Cost: $0.00 | Uptime: 00:03
🏆 BATTLE TEST VICTORY: FREE discovery methods successfully connected!
```

### 🔍 **Real-Time Monitoring Commands**

**While tests are running, use these commands to monitor:**

#### **Server Alpha Monitoring:**
```bash
# Monitor Alpha logs
tail -f /mnt/orobit-shared/q-narwhalknight/logs/battle_test_alpha.log

# Check Alpha metrics
curl -s http://localhost:8333/peers/count
curl -s http://localhost:8333/discovery/cost

# Check Alpha onion address
cat /tmp/q_narwhal_alpha_onion.txt
```

#### **Server Beta Monitoring:**
```bash
# Monitor Beta logs
tail -f /mnt/orobit-shared/q-narwhalknight/logs/battle_test_beta.log

# Check Beta metrics  
curl -s http://localhost:8334/peers/count
curl -s http://localhost:8334/discovery/cost

# Check if Beta found Alpha
curl -s http://localhost:8334/peers/list | grep -o '[a-z2-7]\{56\}\.onion'
```

### 📊 **Generate Battle Test Report**

**After test completion, generate reports:**

#### **Server Alpha Report:**
```bash
cd /mnt/orobit-shared/q-narwhalknight
cargo run --example battle_test_report -- \
    --node alpha \
    --output battle_test_results/alpha_report.json

cat battle_test_results/alpha_report.json | jq '.battle_test_verdict'
```

#### **Server Beta Report:**
```bash
cd /mnt/orobit-shared/q-narwhalknight
cargo run --example battle_test_report -- \
    --node beta \
    --output battle_test_results/beta_report.json

cat battle_test_results/beta_report.json | jq '.cross_server_discovery_success'
```

#### **Combined Report:**
```bash
cd /mnt/orobit-shared/q-narwhalknight
cargo run --example battle_test_report -- \
    --combined \
    --alpha-results battle_test_results/alpha_monitor.json \
    --beta-results battle_test_results/beta_monitor.json \
    --output battle_test_results/combined_report.json

echo "=== BATTLE TEST RESULTS ==="
cat battle_test_results/combined_report.json | jq '{
    verdict: .battle_test_verdict,
    cross_discovery: .cross_server_discovery_success,
    discovery_time: .discovery_time_seconds,
    total_cost: .total_daily_cost,
    production_ready: .production_ready
}'
```

### 🛑 **Stop Battle Test**

**To cleanly stop the tests:**

#### **Server Alpha Stop:**
```bash
# Stop Alpha test (Ctrl+C or kill processes)
pkill -f "battle_test_alpha"
pkill -f "q-narwhal-validator.*alpha"

# Cleanup
rm -f /tmp/alpha_*_pid /tmp/q_narwhal_alpha_onion.txt
```

#### **Server Beta Stop:**  
```bash
# Stop Beta test (Ctrl+C or kill processes)
pkill -f "battle_test_beta" 
pkill -f "q-narwhal-validator.*beta"

# Cleanup
rm -f /tmp/beta_*_pid /tmp/q_narwhal_beta_onion.txt
```

### 🚨 **Troubleshooting**

#### **If Tor fails to start:**
```bash
# Install Tor
sudo apt update && sudo apt install tor

# Start Tor manually
sudo systemctl start tor
sudo systemctl enable tor

# Check Tor status
systemctl status tor
curl --socks5-hostname 127.0.0.1:9050 https://check.torproject.org/api/ip
```

#### **If build fails:**
```bash
# Update Rust
rustup update stable

# Clean build
cargo clean
cargo build --release

# Check dependencies
cargo check --workspace
```

#### **If discovery fails:**
```bash
# Check Tor connectivity
curl -s --socks5-hostname 127.0.0.1:9050 https://check.torproject.org/api/ip

# Verify onion address format (should be 56 chars + .onion)
cat /tmp/q_narwhal_alpha_onion.txt | wc -c  # Should be 62

# Check if ports are available
netstat -tlnp | grep :8333
netstat -tlnp | grep :8334

# Restart with debug logging
export RUST_LOG=debug
./scripts/battle_test_alpha.sh
```

### 🏆 **Victory Conditions**

The battle test **SUCCEEDS** when:

1. ✅ **Both servers generate real Tor onion addresses**
2. ✅ **Server Beta discovers Server Alpha within 5 minutes**  
3. ✅ **Cross-server connection established via FREE methods**
4. ✅ **Daily operating cost remains $0.00 on both servers**
5. ✅ **Multiple discovery methods (DHT, Bootstrap, Gossip) active**
6. ✅ **No network errors or connection failures**
7. ✅ **Battle test report shows "SUCCESS" verdict**

### 📋 **Expected Timeline**

```
T+0:00  - Server Alpha starts battle test
T+0:30  - Alpha onion service established  
T+0:45  - Alpha broadcasts onion address
T+1:00  - Server Beta starts battle test
T+1:30  - Beta onion service established
T+2:00  - Beta begins discovery of Alpha
T+2:30  - Beta discovers Alpha via FREE methods
T+3:00  - Cross-server connection confirmed
T+5:00  - Battle test SUCCESS declared
```

### 🎯 **Ready for Battle!**

**Server Alpha and Server Beta are now ready to prove that Q-NarwhalKnight's FREE discovery methods work in real production conditions with actual Tor networks and cross-server communication.**

**Execute the commands above to begin the battle test and validate that decentralized peer discovery can operate at $0.00 daily cost!** 🔥🏆