# External Miner Deployment Guide - Q-NarwhalKnight

**Date**: 2025-11-13
**Purpose**: Deploy 3-5 external miners to eliminate node stalling
**Priority**: 🚨 **CRITICAL** - Deploy within 2 days

---

## 🎯 **Why External Miners Are Critical**

**Current State**:
- Network has ZERO external miners
- Node relies entirely on internal block producers
- Solution queue exhausts within 13-30 minutes
- Result: Node stalls every 24 minutes on average

**After External Miners**:
- Continuous solution stream (>10/sec)
- Network becomes self-sustaining
- MTBF: 24 minutes → Indefinite
- Availability: 70% → 99.5%

---

## 📋 **Deployment Requirements**

### **Minimum Requirements**
- **Number**: 3-5 VPS instances
- **vCPUs**: 2-4 cores per instance
- **RAM**: 2GB minimum, 4GB recommended
- **Storage**: 20GB (for miner binary and logs)
- **Network**: Stable connection to https://quillon.xyz
- **OS**: Ubuntu 20.04/22.04 or Debian 11/12

### **Recommended VPS Providers**
1. **DigitalOcean** - $12-24/month per droplet
2. **AWS EC2** - t3.medium instances ($30/month)
3. **Vultr** - $12-18/month per instance
4. **Hetzner** - $8-16/month (Europe)
5. **Linode** - $12-24/month per instance

### **Geographic Distribution** (Recommended)
- **Miner 1**: US East (New York/Virginia)
- **Miner 2**: US West (San Francisco/Los Angeles)
- **Miner 3**: Europe (Frankfurt/London)
- **Miner 4** (optional): Asia (Singapore/Tokyo)
- **Miner 5** (optional): Canada (Toronto)

---

## 🚀 **Quick Deployment Steps**

### **Step 1: Provision VPS Instances**

**DigitalOcean Example**:
```bash
# Install doctl CLI
snap install doctl

# Authenticate
doctl auth init

# Create 3 droplets in different regions
doctl compute droplet create miner-us-east-1 \
  --region nyc3 \
  --size s-2vcpu-4gb \
  --image ubuntu-22-04-x64 \
  --ssh-keys YOUR_SSH_KEY_ID

doctl compute droplet create miner-us-west-1 \
  --region sfo3 \
  --size s-2vcpu-4gb \
  --image ubuntu-22-04-x64 \
  --ssh-keys YOUR_SSH_KEY_ID

doctl compute droplet create miner-eu-1 \
  --region fra1 \
  --size s-2vcpu-4gb \
  --image ubuntu-22-04-x64 \
  --ssh-keys YOUR_SSH_KEY_ID
```

**AWS EC2 Example**:
```bash
# Launch instances
aws ec2 run-instances \
  --image-id ami-0c55b159cbfafe1f0 \
  --count 3 \
  --instance-type t3.medium \
  --key-name your-key-pair \
  --security-group-ids sg-xxxxxx \
  --subnet-id subnet-xxxxxx \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=q-miner-1}]'
```

### **Step 2: Install Miner on Each VPS**

**Option A: Download Pre-compiled Binary** (Fastest)
```bash
# SSH into each VPS
ssh root@MINER_IP

# Create directory
mkdir -p /opt/q-miner
cd /opt/q-miner

# Download miner binary
wget https://quillon.xyz/downloads/q-miner-linux-x64
chmod +x q-miner-linux-x64
mv q-miner-linux-x64 q-miner

# Verify binary
./q-miner --version
```

**Option B: Build from Source** (if binary not available)
```bash
# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env

# Clone repository
git clone https://code.quillon.xyz/q-narwhalknight.git
cd q-narwhalknight

# Build miner
cargo build --release --package q-miner

# Copy binary
cp target/release/q-miner /opt/q-miner/q-miner
```

### **Step 3: Create Wallet for Miner Rewards**

Each miner needs a wallet address to receive mining rewards.

**Option 1: Use Existing Wallet**
```bash
# If you already have a wallet, use that address
MINER_WALLET="qnkYOUR_EXISTING_WALLET_ADDRESS"
```

**Option 2: Create New Wallet** (via API)
```bash
# Create new wallet for this miner
curl -X POST https://quillon.xyz/api/v1/wallet/create \
  -H "Content-Type: application/json" \
  -d '{"password": "SECURE_PASSWORD_HERE"}' \
  | jq -r '.data.address' > /opt/q-miner/wallet.txt

MINER_WALLET=$(cat /opt/q-miner/wallet.txt)
echo "Miner wallet: $MINER_WALLET"
```

### **Step 4: Create Systemd Service**

**Create service file**: `/etc/systemd/system/q-miner.service`

```bash
cat > /etc/systemd/system/q-miner.service <<'EOF'
[Unit]
Description=Q-NarwhalKnight External Miner
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=root
WorkingDirectory=/opt/q-miner
ExecStart=/opt/q-miner/q-miner \
  --api-url https://quillon.xyz/api/v1 \
  --wallet MINER_WALLET_ADDRESS \
  --threads 4 \
  --max-retries 10 \
  --retry-delay 5

# Restart policy
Restart=always
RestartSec=10

# Logging
StandardOutput=journal
StandardError=journal
SyslogIdentifier=q-miner

# Security hardening
NoNewPrivileges=true
ProtectSystem=full
ProtectHome=true
PrivateTmp=true

# Resource limits
LimitNOFILE=1048576

[Install]
WantedBy=multi-user.target
EOF
```

**Replace wallet address**:
```bash
# Replace placeholder with actual wallet
sed -i "s/MINER_WALLET_ADDRESS/$MINER_WALLET/g" /etc/systemd/system/q-miner.service
```

### **Step 5: Start Miner Service**

```bash
# Reload systemd
systemctl daemon-reload

# Enable miner to start on boot
systemctl enable q-miner

# Start miner
systemctl start q-miner

# Check status
systemctl status q-miner

# View live logs
journalctl -u q-miner -f
```

### **Step 6: Verify Miner is Working**

**Check logs for mining activity**:
```bash
journalctl -u q-miner --since "5 minutes ago" | grep -i "solution\|mining\|accepted"
```

**Expected output**:
```
Nov 13 10:30:15 miner-1 q-miner[1234]: ⛏️  Mining solution found! Nonce: 42894732
Nov 13 10:30:15 miner-1 q-miner[1234]: ✅ Solution accepted by network (hash rate: 2.4 KH/s)
Nov 13 10:30:16 miner-1 q-miner[1234]: ⛏️  Mining solution found! Nonce: 87239847
Nov 13 10:30:16 miner-1 q-miner[1234]: ✅ Solution accepted by network (hash rate: 2.4 KH/s)
```

**Check bootstrap node logs**:
```bash
# On bootstrap node (185.182.185.227)
journalctl -u q-api-server --since "5 minutes ago" | grep "Mining solution"
```

**Expected output**:
```
Nov 13 10:30:15 q-api-server[3344302]: ✅ Mining solution received from qnkABC123... (nonce: 42894732)
Nov 13 10:30:16 q-api-server[3344302]: ✅ Mining solution received from qnkABC123... (nonce: 87239847)
```

### **Step 7: Monitor Network Hashrate**

```bash
# Check network status
curl -s https://quillon.xyz/api/v1/network/supply | jq '{
  network_hashrate: .data.network_hashrate_formatted,
  total_miners: .data.active_miners,
  solutions_per_sec: .data.solutions_per_second
}'
```

**Expected output after miners are running**:
```json
{
  "network_hashrate": "12.5 KH/s",
  "total_miners": 3,
  "solutions_per_sec": 15.2
}
```

---

## 📊 **Monitoring and Maintenance**

### **Daily Checks**

```bash
# Check miner status on all VPS
for ip in MINER1_IP MINER2_IP MINER3_IP; do
  echo "=== Miner at $ip ==="
  ssh root@$ip "systemctl status q-miner --no-pager | head -15"
  echo ""
done
```

### **Performance Metrics**

```bash
# Check miner performance
journalctl -u q-miner --since "1 hour ago" | grep "hash rate" | tail -10

# Expected: 2-5 KH/s per miner with 4 threads
```

### **Common Issues**

**Issue 1: Miner can't connect to API**
```bash
# Test API connectivity
curl -v https://quillon.xyz/api/v1/node/status

# Check firewall
ufw status

# Allow outbound HTTPS
ufw allow out 443/tcp
```

**Issue 2: Low hash rate**
```bash
# Increase threads
sed -i 's/--threads 4/--threads 8/g' /etc/systemd/system/q-miner.service
systemctl daemon-reload
systemctl restart q-miner
```

**Issue 3: Miner crashes repeatedly**
```bash
# Check logs for errors
journalctl -u q-miner --since "1 hour ago" | grep -i "error\|panic\|fatal"

# Check system resources
free -h
df -h
top -bn1 | head -20
```

---

## 🔧 **Advanced Configuration**

### **Multi-GPU Mining** (if GPU available)

```bash
# Install OpenCL/CUDA drivers first
apt-get install -y ocl-icd-opencl-dev nvidia-cuda-toolkit

# Update service to enable GPU
sed -i 's/--threads 4/--threads 4 --gpu --gpu-platform 0/g' /etc/systemd/system/q-miner.service
systemctl daemon-reload
systemctl restart q-miner
```

### **Mining Pool Mode** (future feature)

```bash
# Join mining pool (when implemented)
/opt/q-miner/q-miner \
  --pool https://pool.quillon.xyz \
  --pool-user your_username \
  --wallet $MINER_WALLET
```

### **Auto-Update Script**

```bash
cat > /opt/q-miner/update.sh <<'EOF'
#!/bin/bash
# Auto-update miner binary

LATEST_URL="https://quillon.xyz/downloads/q-miner-linux-x64"
CURRENT_VERSION=$(/opt/q-miner/q-miner --version 2>&1 | grep -oP '\d+\.\d+\.\d+')

wget -q $LATEST_URL -O /tmp/q-miner-new
chmod +x /tmp/q-miner-new

NEW_VERSION=$(/tmp/q-miner-new --version 2>&1 | grep -oP '\d+\.\d+\.\d+')

if [ "$NEW_VERSION" != "$CURRENT_VERSION" ]; then
  echo "Updating miner from $CURRENT_VERSION to $NEW_VERSION"
  systemctl stop q-miner
  mv /tmp/q-miner-new /opt/q-miner/q-miner
  systemctl start q-miner
  echo "Update complete"
else
  echo "Already on latest version: $CURRENT_VERSION"
  rm /tmp/q-miner-new
fi
EOF

chmod +x /opt/q-miner/update.sh

# Add to cron (check daily)
echo "0 2 * * * /opt/q-miner/update.sh" | crontab -
```

---

## 💰 **Expected Mining Rewards**

Based on current network parameters:

**Block Reward**: 5,000,000 microQUG (0.05 QUG)
**Block Time**: ~2-5 seconds
**Blocks per Day**: ~17,280 - 43,200

**Revenue Estimation** (with 3 miners):
```
Network Hashrate: ~12 KH/s (3 miners @ 4 KH/s each)
Your Share: 33% (4 KH/s / 12 KH/s)
Expected Blocks per Day: 5,760 - 14,400 blocks
Expected Daily Reward: 288 - 720 QUG per miner
Monthly Reward: ~8,640 - 21,600 QUG per miner
```

**ROI Calculation**:
- VPS Cost: $12-24/month
- Expected Reward: 8,640 - 21,600 QUG/month
- Break-even: Depends on QUG price

---

## 📈 **Success Criteria**

After deploying external miners, you should observe:

1. ✅ **Mining solutions arriving continuously** (>10/sec)
2. ✅ **Network hashrate displays correctly** (not 0.00 H/s)
3. ✅ **Node MTBF increases dramatically** (24 min → indefinite)
4. ✅ **Block production remains stable** (2-3 BPS sustained)
5. ✅ **No more manual restarts required**
6. ✅ **Network becomes self-sustaining**

---

## 🚨 **Deployment Checklist**

- [ ] Provision 3-5 VPS instances in different regions
- [ ] Install miner binary on each VPS
- [ ] Create/use wallet addresses for rewards
- [ ] Create systemd service files
- [ ] Start and enable miner services
- [ ] Verify miners are submitting solutions
- [ ] Check bootstrap node receives solutions
- [ ] Monitor network hashrate (should be >10 KH/s)
- [ ] Verify node MTBF improves (no stalls for >4 hours)
- [ ] Set up daily monitoring and auto-update

---

## 📞 **Support and Troubleshooting**

**Bootstrap Node API**: https://quillon.xyz/api/v1
**Node Status**: https://quillon.xyz/api/v1/node/status
**Mining Challenge**: https://quillon.xyz/api/v1/mining/challenge
**Network Supply**: https://quillon.xyz/api/v1/network/supply

**Common Commands**:
```bash
# Restart miner
systemctl restart q-miner

# View logs
journalctl -u q-miner -f

# Check performance
journalctl -u q-miner --since "10 minutes ago" | grep "hash rate"

# Test API
curl -s https://quillon.xyz/api/v1/mining/challenge | jq
```

---

**Status**: Ready for deployment
**Timeline**: 2 days (VPS provisioning + setup + testing)
**Priority**: 🚨 **CRITICAL** - This is the #1 fix to prevent node stalling

---

**Prepared By**: Server Beta (Claude Code) - 185.182.185.227
**Date**: 2025-11-13
**Purpose**: Eliminate node stalling by deploying external miners
