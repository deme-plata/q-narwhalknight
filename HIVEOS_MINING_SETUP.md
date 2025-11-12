# HiveOS Mining Setup for Q-NarwhalKnight (QNK)

## Overview

This guide provides complete HiveOS configuration for mining Q-NarwhalKnight cryptocurrency.

**Coin**: Q-NarwhalKnight (QNK)
**Algorithm**: DAG-Knight + Quantum VDF
**Pool**: Solo mining to network bootstrap nodes
**Miner**: q-miner (custom Rust-based miner)

---

## Quick Start

### Step 1: Download Miner Binary

```bash
# On your HiveOS rig
cd /tmp
wget https://quillon.xyz/downloads/q-miner-linux-x64
chmod +x q-miner-linux-x64
sudo mkdir -p /usr/local/bin
sudo mv q-miner-linux-x64 /usr/local/bin/q-miner
```

### Step 2: Create Flight Sheet

1. Navigate to **Flight Sheets** in HiveOS dashboard
2. Click **Create Flight Sheet**
3. Configure as follows:

**Flight Sheet Settings:**
- **Name**: `Q-NarwhalKnight Mining`
- **Coin**: Custom
- **Wallet**: Your QNK wallet address (qnk...)
- **Pool**: Configure custom
- **Miner**: Custom miner

---

## Custom Miner Configuration

### Create Custom Miner Package

HiveOS uses a specific directory structure for custom miners. Here's how to set it up:

#### 1. Create Miner Package Directory

```bash
# On HiveOS rig
sudo mkdir -p /hive/miners/custom/q-miner/1.0
cd /hive/miners/custom/q-miner/1.0
```

#### 2. Download Miner Binary

```bash
sudo wget -O q-miner https://quillon.xyz/downloads/q-miner-linux-x64
sudo chmod +x q-miner
```

#### 3. Create `h-manifest.conf`

```bash
sudo tee h-manifest.conf > /dev/null <<'EOF'
# HiveOS Miner Manifest for Q-NarwhalKnight
MINER_NAME=q-miner
MINER_FORK=qnk
MINER_VER=1.0

# Minimum HiveOS version required
HIVE_VER_MIN=0.6-00

# GPU types supported (CPU-only miner)
MINER_GPU=nvidia amd cpu

# Minimum GPU driver versions (not applicable for CPU mining)
NVIDIA_MIN_VER=0
AMD_MIN_VER=0

# Configuration file name
MINER_CONFIG_FILENAME=miner.conf

# Process name for monitoring
MINER_PROCESS_NAME=q-miner
EOF
```

#### 4. Create `h-config.sh`

```bash
sudo tee h-config.sh > /dev/null <<'EOF'
#!/usr/bin/env bash

# Q-NarwhalKnight Miner Configuration Script for HiveOS

# Get configuration from HiveOS
[[ -z $CUSTOM_TEMPLATE ]] && echo -e "${YELLOW}CUSTOM_TEMPLATE is empty${NOCOLOR}" && return 1
[[ -z $CUSTOM_URL ]] && echo -e "${YELLOW}CUSTOM_URL is empty${NOCOLOR}" && return 1

# Parse HiveOS configuration
conf="-t $CUSTOM_TEMPLATE"

# Extract wallet address from CUSTOM_TEMPLATE (should be qnk address)
WALLET=$CUSTOM_TEMPLATE

# Extract node URL from CUSTOM_URL (bootstrap node endpoint)
NODE_URL=$CUSTOM_URL

# Number of threads (default to CPU count)
if [[ ! -z $CUSTOM_USER_CONFIG ]]; then
    THREADS=$(echo $CUSTOM_USER_CONFIG | jq -r '.threads // empty')
fi

# Default to all CPU cores if not specified
if [[ -z $THREADS ]]; then
    THREADS=$(nproc)
fi

# Generate miner configuration
cat > $MINER_DIR/$MINER_VER/miner.conf <<MINER_CONF
{
  "wallet_address": "$WALLET",
  "node_url": "$NODE_URL",
  "threads": $THREADS,
  "log_level": "info"
}
MINER_CONF

echo "Q-NarwhalKnight miner configured:"
echo "  Wallet: $WALLET"
echo "  Node: $NODE_URL"
echo "  Threads: $THREADS"
EOF

sudo chmod +x h-config.sh
```

#### 5. Create `h-run.sh`

```bash
sudo tee h-run.sh > /dev/null <<'EOF'
#!/usr/bin/env bash

# Q-NarwhalKnight Miner Run Script for HiveOS

cd $(dirname $0)

# Load configuration
[[ ! -f miner.conf ]] && echo "No miner.conf found" && exit 1

# Parse configuration
WALLET=$(jq -r '.wallet_address' miner.conf)
NODE_URL=$(jq -r '.node_url' miner.conf)
THREADS=$(jq -r '.threads' miner.conf)

# Launch miner
./q-miner \
  --wallet "$WALLET" \
  --node "$NODE_URL" \
  --threads "$THREADS" \
  2>&1 | tee --append $MINER_LOG_BASENAME.log
EOF

sudo chmod +x h-run.sh
```

#### 6. Create `h-stats.sh`

```bash
sudo tee h-stats.sh > /dev/null <<'EOF'
#!/usr/bin/env bash

# Q-NarwhalKnight Miner Stats Script for HiveOS

# Get miner stats from log file or API
stats_raw=$(tail -100 $MINER_LOG_BASENAME.log 2>/dev/null | grep -oP '(?<=Hashrate: )[0-9.]+(?= KH/s)')

# Calculate total hashrate
khs=0
for rate in $stats_raw; do
    khs=$(echo "$khs + $rate" | bc)
done

# Get accepted shares
accepted=$(tail -100 $MINER_LOG_BASENAME.log 2>/dev/null | grep -c "Solution accepted")

# Get rejected shares
rejected=$(tail -100 $MINER_LOG_BASENAME.log 2>/dev/null | grep -c "Solution rejected")

# Get uptime
uptime=$(ps -p $(pgrep -f q-miner) -o etime= 2>/dev/null | tr -d ' ')

# Output stats in HiveOS format (JSON)
stats=$(jq -nc \
    --arg hs "$khs" \
    --arg ac "$accepted" \
    --arg rj "$rejected" \
    --arg uptime "$uptime" \
    '{
        hs: [$hs],
        hs_units: "khs",
        temp: [],
        fan: [],
        uptime: $uptime,
        ar: [$ac, $rj],
        algo: "dag-knight"
    }')

echo "$stats"
EOF

sudo chmod +x h-stats.sh
```

---

## Flight Sheet Configuration

### Web UI Configuration

**Flight Sheet Name**: Q-NarwhalKnight Solo Mining

**Coin Settings:**
- Coin: `Custom`
- Coin symbol: `QNK`
- Algorithm: `DAG-Knight`

**Wallet:**
- Create a wallet named `QNK Wallet`
- Address: Your QNK wallet address (starts with `qnk`)

**Pool Configuration:**
- Pool Server: `https://quillon.xyz` (or your bootstrap node)
- Pool Port: `8080`
- Pool Type: Solo

**Miner:**
- Miner: `Custom`
- Installation URL: `/hive/miners/custom/q-miner/1.0`
- Miner configuration template: `%WAL%`
- Pool URL: `https://quillon.xyz:8080`

**Extra Config Arguments (JSON):**
```json
{
  "threads": 8,
  "log_level": "info"
}
```

---

## Configuration File Alternative

If you prefer to configure via HiveOS config files:

### `/hive-config/rig.conf`

Add this section:

```bash
# Q-NarwhalKnight Mining
CUSTOM_MINER="q-miner"
CUSTOM_NAME="Q-NarwhalKnight"
CUSTOM_TEMPLATE="qnk1234567890abcdef1234567890abcdef1234567890abcdef1234567890ab"
CUSTOM_URL="https://quillon.xyz:8080"
CUSTOM_PASS="x"
CUSTOM_USER_CONFIG='{"threads": 8, "log_level": "info"}'
```

---

## Bootstrap Node Endpoints

### Primary Bootstrap Node
- **URL**: `https://quillon.xyz`
- **API Port**: `8080`
- **P2P Port**: `9001`

### Configuration Examples

**Mining to Primary Bootstrap:**
```bash
./q-miner \
  --wallet qnk1234567890abcdef1234567890abcdef1234567890abcdef1234567890ab \
  --node https://quillon.xyz:8080 \
  --threads 8
```

**Mining with Multiple Nodes (Failover):**
```bash
./q-miner \
  --wallet qnk1234567890abcdef1234567890abcdef1234567890abcdef1234567890ab \
  --node https://quillon.xyz:8080 \
  --fallback-node https://backup-node.quillon.xyz:8080 \
  --threads 8
```

---

## Mining Performance Tuning

### CPU Optimization

**Recommended Thread Configuration:**
- **AMD Ryzen 9**: 16 threads (all cores)
- **AMD Ryzen 7**: 8-12 threads
- **Intel i9**: 16 threads
- **Intel i7**: 8 threads
- **Server CPUs (EPYC, Xeon)**: 32-64 threads

**Performance Tips:**
- Use all physical cores for best performance
- Enable SMT/Hyper-Threading
- Set CPU governor to `performance` mode

```bash
# Set performance mode (on HiveOS rig)
sudo cpupower frequency-set -g performance
```

### Expected Hashrates

| CPU Model | Threads | Hashrate (KH/s) |
|-----------|---------|-----------------|
| AMD Ryzen 9 5950X | 32 | ~45 KH/s |
| AMD Ryzen 9 3950X | 32 | ~38 KH/s |
| Intel i9-12900K | 24 | ~42 KH/s |
| AMD EPYC 7742 | 128 | ~180 KH/s |
| Intel Xeon Gold 6248R | 96 | ~140 KH/s |

---

## Monitoring and Stats

### HiveOS Dashboard

Once configured, your miner will appear in the HiveOS dashboard with:
- **Hashrate**: Real-time KH/s
- **Accepted Shares**: Valid solutions submitted
- **Rejected Shares**: Invalid solutions
- **Uptime**: Mining session duration
- **Temperature**: CPU temperature (if sensors available)

### API Monitoring

The miner exposes stats via the node API:

```bash
# Check miner stats
curl https://quillon.xyz:8080/api/v1/mining/stats

# Check wallet balance
curl https://quillon.xyz:8080/api/v1/wallets/qnk.../balance
```

---

## Troubleshooting

### Miner Not Starting

**Check logs:**
```bash
tail -100 /var/log/miner/q-miner/q-miner.log
```

**Common issues:**
- Invalid wallet address format (must start with `qnk`)
- Node URL unreachable
- Port 8080 blocked by firewall

### Low Hashrate

**Solutions:**
- Increase thread count: `"threads": 16`
- Enable CPU performance mode
- Check CPU temperature (thermal throttling)
- Ensure no other CPU-intensive tasks running

### Connection Issues

**Check node connectivity:**
```bash
curl -I https://quillon.xyz:8080/api/v1/health
```

**Expected response:**
```
HTTP/2 200
content-type: application/json
```

### Shares Rejected

**Possible causes:**
- Network latency too high (>500ms)
- Miner binary outdated
- Node out of sync

**Solution:**
```bash
# Update miner binary
cd /hive/miners/custom/q-miner/1.0
sudo wget -O q-miner https://quillon.xyz/downloads/q-miner-linux-x64
sudo chmod +x q-miner
miner restart
```

---

## Advanced Configuration

### Multiple Rigs

For mining farms with multiple HiveOS rigs:

1. Create a single flight sheet template
2. Apply to all rigs
3. Each rig uses same wallet address
4. Hashrates aggregate on the network

### Load Balancing

Distribute mining across multiple bootstrap nodes:

**rig1.conf:**
```bash
CUSTOM_URL="https://quillon.xyz:8080"
```

**rig2.conf:**
```bash
CUSTOM_URL="https://backup-node.quillon.xyz:8080"
```

### Auto-Restart on Failure

HiveOS automatically restarts miners on failure, but you can customize:

**Watchdog Configuration:**
```bash
# In HiveOS -> Workers -> [Your Rig] -> Tuning
# Set "Miner restart on no hashes" to 5 minutes
# Set "Reboot on no hashes" to 10 minutes
```

---

## Security Considerations

### Wallet Security
- ✅ **Never share private keys** - Only provide public address (qnk...)
- ✅ **Use unique wallet per rig** for better tracking
- ✅ **Backup wallet seed phrase** securely offline

### Network Security
- ✅ **Use HTTPS connections** to bootstrap nodes
- ✅ **Verify SSL certificates** for node endpoints
- ✅ **Enable HiveOS firewall** to restrict access

### Mining Pool Security
- ✅ **Verify bootstrap node authenticity** (check quillon.xyz SSL)
- ✅ **Monitor for abnormal rejections** (>5% is suspicious)
- ✅ **Check balance regularly** via API or GUI

---

## Economic Considerations

### Block Rewards

- **Current Block Reward**: 0.000099 QNK per solution
- **Network Difficulty**: Dynamic (adjusts every 100 blocks)
- **Block Time**: ~2 seconds (DAG allows parallel blocks)

### Profitability Calculator

```
Daily Mining Revenue = (Your Hashrate / Network Hashrate) × Daily Block Reward × QNK Price

Example:
- Your Hashrate: 45 KH/s
- Network Hashrate: 500 KH/s
- Daily Blocks: ~43,200 (2s block time)
- Block Reward: 0.000099 QNK
- Daily Earnings: (45/500) × 43,200 × 0.000099 ≈ 0.385 QNK/day
```

### Power Efficiency

**AMD Ryzen 9 5950X Example:**
- Hashrate: 45 KH/s
- Power Draw: 140W mining
- Efficiency: 0.32 KH/s per Watt

**Cost Analysis:**
```
Power Cost = (140W × 24h × $0.12/kWh) / 1000 = $0.40/day
Daily Revenue = 0.385 QNK × QNK_price
Break-even if: QNK_price > $1.04
```

---

## Update Procedure

When new miner versions are released:

```bash
# Stop miner
miner stop

# Backup old version
cd /hive/miners/custom/q-miner
sudo cp -r 1.0 1.0.backup

# Download new version
cd 1.0
sudo wget -O q-miner https://quillon.xyz/downloads/q-miner-linux-x64
sudo chmod +x q-miner

# Restart miner
miner start
```

---

## Support

### Community Resources
- **Website**: https://quillon.xyz
- **GitHub**: https://github.com/q-narwhalknight
- **Discord**: [Join our Discord server]
- **Telegram**: [Join Telegram group]

### Reporting Issues

When reporting mining issues, provide:
1. HiveOS version: `hive-replace --list`
2. Miner version: `./q-miner --version`
3. CPU model: `lscpu | grep "Model name"`
4. Wallet address (public address only)
5. Recent logs: `tail -100 /var/log/miner/q-miner/q-miner.log`

---

## Changelog

### v1.0 (Current)
- Initial HiveOS support
- CPU mining only
- Solo mining to bootstrap nodes
- Basic stats reporting

### Future Roadmap
- [ ] GPU mining support (CUDA/OpenCL)
- [ ] Pool mining protocol
- [ ] Advanced overclocking profiles
- [ ] Profit switching algorithms
- [ ] Dual mining support

---

**Happy Mining! ⛏️**

Generated with Claude Code
Co-Authored-By: Claude <noreply@anthropic.com>
