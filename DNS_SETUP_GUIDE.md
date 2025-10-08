# DNS TXT Records Setup for Q-NarwhalKnight Discovery

## Namecheap DNS Configuration for `quantum.bitcoin.oro.xyz`

This guide shows how to set up DNS TXT records on Namecheap for Q-NarwhalKnight peer discovery using your domain `quantum.bitcoin.oro.xyz`.

### 🌐 **Namecheap Setup Instructions**

#### Step 1: Access Namecheap DNS Management

1. **Log into Namecheap**
   - Go to https://namecheap.com
   - Sign in to your account
   - Navigate to **Domain List**

2. **Access DNS Management**
   - Find your domain `oro.xyz` in the list
   - Click **Manage** next to the domain
   - Go to **Advanced DNS** tab

#### Step 2: Create Q-NarwhalKnight TXT Records

Add the following TXT records for Q-NarwhalKnight peer discovery:

| Type | Host | Value | TTL |
|------|------|-------|-----|
| TXT | `_qnk._tcp.quantum.bitcoin` | `"node=validator123abc456def.onion:8333"` | 300 |
| TXT | `_qnk._tcp.quantum.bitcoin` | `"protocol=q-narwhalknight/0.1.0"` | 300 |
| TXT | `_qnk._tcp.quantum.bitcoin` | `"capabilities=consensus,mempool,quantum"` | 300 |
| TXT | `_qnk-bootstrap._tcp.quantum.bitcoin` | `"bootstrap=bootstrap1.qnk.onion:8333"` | 300 |
| TXT | `_qnk-bootstrap._tcp.quantum.bitcoin` | `"bootstrap=bootstrap2.qnk.onion:8333"` | 300 |

#### Step 3: Detailed Namecheap Interface Steps

**Adding Each TXT Record:**

1. **Click "Add New Record"**
2. **Select Record Type:** TXT
3. **Enter Host:** `_qnk._tcp.quantum.bitcoin`
4. **Enter Value:** `"node=your-validator-onion-address.onion:8333"`
5. **Set TTL:** 300 (5 minutes)
6. **Click "Save Changes"**

**Example for your first record:**
```
Type: TXT
Host: _qnk._tcp.quantum.bitcoin
Value: "node=validator123abc456def789ghi012jkl345mno678pqr901stu234vwx.onion:8333"
TTL: 300
```

### 🔧 **Q-NarwhalKnight DNS Discovery Format**

#### Standard TXT Record Formats:

```dns
; Node advertisement
_qnk._tcp.quantum.bitcoin.oro.xyz. IN TXT "node=validator123...xyz.onion:8333"
_qnk._tcp.quantum.bitcoin.oro.xyz. IN TXT "protocol=q-narwhalknight/0.1.0"
_qnk._tcp.quantum.bitcoin.oro.xyz. IN TXT "capabilities=consensus,mempool,quantum"

; Bootstrap nodes
_qnk-bootstrap._tcp.quantum.bitcoin.oro.xyz. IN TXT "bootstrap=bootstrap1.qnk.onion:8333"
_qnk-bootstrap._tcp.quantum.bitcoin.oro.xyz. IN TXT "bootstrap=bootstrap2.qnk.onion:8333"

; Network information
_qnk-network._tcp.quantum.bitcoin.oro.xyz. IN TXT "network=mainnet"
_qnk-network._tcp.quantum.bitcoin.oro.xyz. IN TXT "phase=phase1"
_qnk-network._tcp.quantum.bitcoin.oro.xyz. IN TXT "version=0.1.0"
```

### 🆓 **Cost Analysis for DNS Discovery**

| Method | Setup Cost | Annual Cost | Daily Cost |
|--------|------------|-------------|------------|
| **DNS TXT Records** | $0 | ~$12/year domain | $0.03/day |
| **Bitcoin OP_RETURN** | $0 | $52,560,000/year | $144,000/day |
| **FREE Methods Only** | $0 | $0/year | $0.00/day |

**DNS is 4,800,000x cheaper than Bitcoin OP_RETURN!**

### 📋 **Complete Namecheap Configuration**

Here's your complete DNS setup for `quantum.bitcoin.oro.xyz`:

```
# Q-NarwhalKnight Node Records
Type: TXT
Host: _qnk._tcp.quantum.bitcoin
Value: "node=validatorabc123def456ghi789jkl012mno345pqr678stu901vwx.onion:8333"
TTL: 300

Type: TXT  
Host: _qnk._tcp.quantum.bitcoin
Value: "protocol=q-narwhalknight/0.1.0"
TTL: 300

Type: TXT
Host: _qnk._tcp.quantum.bitcoin  
Value: "capabilities=consensus,mempool,quantum,dag-knight"
TTL: 300

Type: TXT
Host: _qnk._tcp.quantum.bitcoin
Value: "phase=phase1"
TTL: 300

# Bootstrap Node Records
Type: TXT
Host: _qnk-bootstrap._tcp.quantum.bitcoin
Value: "bootstrap=bootstrap1.qnk.onion:8333"
TTL: 300

Type: TXT
Host: _qnk-bootstrap._tcp.quantum.bitcoin
Value: "bootstrap=bootstrap2.qnk.onion:8333"  
TTL: 300

Type: TXT
Host: _qnk-bootstrap._tcp.quantum.bitcoin
Value: "bootstrap=bootstrap3.qnk.onion:8333"
TTL: 300

# Network Configuration
Type: TXT
Host: _qnk-network._tcp.quantum.bitcoin
Value: "network=mainnet"
TTL: 300

Type: TXT
Host: _qnk-network._tcp.quantum.bitcoin
Value: "genesis=0x1234567890abcdef..."
TTL: 300
```

### 🔍 **Verification Commands**

After setting up the DNS records, verify them with these commands:

```bash
# Check Q-NarwhalKnight node records
dig TXT _qnk._tcp.quantum.bitcoin.oro.xyz

# Check bootstrap node records  
dig TXT _qnk-bootstrap._tcp.quantum.bitcoin.oro.xyz

# Check network configuration
dig TXT _qnk-network._tcp.quantum.bitcoin.oro.xyz

# Alternative verification
nslookup -type=TXT _qnk._tcp.quantum.bitcoin.oro.xyz
```

**Expected Output:**
```
_qnk._tcp.quantum.bitcoin.oro.xyz. 300 IN TXT "node=validator123...xyz.onion:8333"
_qnk._tcp.quantum.bitcoin.oro.xyz. 300 IN TXT "protocol=q-narwhalknight/0.1.0"
_qnk._tcp.quantum.bitcoin.oro.xyz. 300 IN TXT "capabilities=consensus,mempool,quantum"
```

### 💻 **Q-NarwhalKnight DNS Discovery Code**

Here's how Q-NarwhalKnight will discover peers via your DNS records:

```rust
use trust_dns_resolver::{Resolver, config::*};

pub async fn discover_peers_via_dns(domain: &str) -> Result<Vec<String>> {
    let resolver = Resolver::new(ResolverConfig::default(), ResolverOpts::default())?;
    
    // Query Q-NarwhalKnight TXT records
    let query = format!("_qnk._tcp.{}", domain);
    let response = resolver.txt_lookup(&query).await?;
    
    let mut peers = Vec::new();
    
    for record in response.iter() {
        let txt_data = record.to_string();
        
        // Parse node records
        if txt_data.starts_with("node=") {
            let node_addr = txt_data.strip_prefix("node=").unwrap();
            peers.push(node_addr.to_string());
            info!("🌐 DNS discovered peer: {} (cost: $0.03/day)", node_addr);
        }
    }
    
    Ok(peers)
}
```

### 🎯 **Integration with FREE Discovery**

DNS discovery integrates with the FREE discovery system:

```toml
# free-discovery-config.toml
[dns_discovery]
enabled = true                    # Enable DNS discovery
domains = [
    "quantum.bitcoin.oro.xyz"     # Your domain
]
ttl_seconds = 300                # 5 minute cache
cost_per_query = 0.0001          # Negligible cost

[cost_tracking]
dns_annual_cost = 12.00          # $12/year domain cost
dns_daily_cost = 0.03            # $0.03/day amortized
alert_threshold = 1.00           # Alert if cost > $1/day
```

### 🚀 **Production Deployment Steps**

1. **Set up DNS records** (as shown above)
2. **Wait for DNS propagation** (5-60 minutes)
3. **Update Q-NarwhalKnight config** to include your domain
4. **Start node with DNS discovery enabled**
5. **Verify discovery in logs**

### 📊 **Discovery Method Comparison**

| Method | Cost/Day | Setup | Latency | Reliability |
|--------|----------|--------|---------|------------|
| **Tor DHT** | $0.00 | Easy | 1-30s | High |
| **Bootstrap** | $0.00 | Easy | 1-10s | High |
| **Gossip** | $0.00 | Auto | <1s | High |
| **DNS TXT** | $0.03 | Manual | 1-5s | Very High |
| **Bitcoin OP_RETURN** | $144,000 | Hard | 10-60m | High |

**Recommendation:** Use FREE methods (Tor DHT + Bootstrap + Gossip) as primary, DNS as optional backup.

### 🔐 **Security Considerations**

**DNS Advantages:**
- ✅ Very low cost ($0.03/day vs $144,000/day)
- ✅ Standard internet infrastructure  
- ✅ Fast resolution (1-5 seconds)
- ✅ Easy to update

**DNS Considerations:**
- ⚠️ DNS queries not anonymous (use Tor for DNS queries)
- ⚠️ Domain registrar dependency
- ⚠️ DNS cache poisoning risk (mitigated by multiple sources)
- ⚠️ Annual domain renewal required

### 🛡️ **Secure DNS Usage**

For maximum security, configure Tor to proxy DNS queries:

```bash
# Configure Tor for DNS queries
echo "DNSPort 9053" >> /etc/tor/torrc  
echo "AutomapHostsOnResolve 1" >> /etc/tor/torrc
sudo systemctl restart tor

# Use Tor for DNS resolution
dig @127.0.0.1 -p 9053 TXT _qnk._tcp.quantum.bitcoin.oro.xyz
```

### 🏆 **Final Result**

Your domain `quantum.bitcoin.oro.xyz` will now serve as a discovery point for Q-NarwhalKnight nodes at a cost of **$0.03/day** - **4,800,000x cheaper** than Bitcoin OP_RETURN while providing **faster discovery** and **easier management**.

The Q-NarwhalKnight network can now use:
1. 🆓 **Tor methods** ($0.00/day) - Primary discovery  
2. 🌐 **DNS records** ($0.03/day) - Fast backup discovery
3. ⚠️ **Bitcoin OP_RETURN** ($144,000/day) - Only if absolutely necessary

**Best of all worlds: Fast, cheap, secure, and decentralized!**