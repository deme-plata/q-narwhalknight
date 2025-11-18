# PQC Usage Guide - Q-NarwhalKnight v1.0.16-beta

**Quick Start Guide for Post-Quantum Cryptography**

---

## Prerequisites

- Q-NarwhalKnight v1.0.16-beta or later
- Rust toolchain (for key generation)
- Linux x86_64 system

---

## Step 1: Generate Validator Keypair

### Using the CLI Tool

```bash
# Generate a new validator keypair
cargo run --package q-types --example generate_validator_key /path/to/validator.json

# Example:
cargo run --package q-types --example generate_validator_key /etc/q-narwhalknight/validator.json
```

### Expected Output

```
🔐 Generating validator keypair...

Generated keypair:
  Node ID: 6c4c0671405cb0bf282c6894c72083feb088a432afee9b14e101689dfee55549
  Ed25519 public key: 32 bytes
  Dilithium5 public key: 2592 bytes
  Preferred phase: Phase0Ed25519

✅ Keypair saved to: /etc/q-narwhalknight/validator.json

To use this key with q-api-server:
  q-api-server --validator-key /etc/q-narwhalknight/validator.json
```

### Keypair File Details

- **Size:** ~63KB
- **Format:** JSON
- **Contains:**
  - Node ID (32 bytes)
  - Ed25519 signing key (32 bytes secret)
  - Ed25519 verifying key (32 bytes public)
  - Dilithium5 secret key (4864 bytes)
  - Dilithium5 public key (2592 bytes)
  - Preferred signature phase

---

## Step 2: Secure the Keypair File

### Set Proper Permissions

```bash
# Only owner can read/write
chmod 600 /etc/q-narwhalknight/validator.json

# Verify permissions
ls -l /etc/q-narwhalknight/validator.json
# Expected: -rw------- 1 root root 63K ...
```

### Backup the Keypair

```bash
# Create encrypted backup
tar -czf validator-backup.tar.gz /etc/q-narwhalknight/validator.json
gpg --symmetric --cipher-algo AES256 validator-backup.tar.gz

# Store encrypted backup off-site
```

**⚠️ WARNING:** If you lose this file, you cannot sign blocks! Keep secure backups!

---

## Step 3: Start q-api-server with PQC

### Command Line

```bash
# Development mode
./target/release/q-api-server \
  --validator-key /etc/q-narwhalknight/validator.json \
  --port 8080

# Production mode with full options
./target/release/q-api-server \
  --validator-key /etc/q-narwhalknight/validator.json \
  --port 8080 \
  --p2p-port 9001 \
  --network testnet-phase11 \
  --bootstrap-peer /ip4/185.182.185.227/tcp/9001/p2p/12D3KooWRX3GGK9Fs3iM3BfqYNNJiHBDujac7EHwqWjaK1n1kzPN
```

### Expected Startup Logs

```
🔐 ════════════════════════════════════════════════════════
🔐 Loading validator keypair for PQC signing...
✅ Validator keypair loaded successfully
   Node ID: 6c4c0671...
   Preferred phase: Phase0Ed25519
   Ed25519 public key: 32 bytes
   Dilithium5 public key: 2592 bytes
🔐 PQC block signing: ENABLED
🔐 ════════════════════════════════════════════════════════

🔐 Wiring validator keypair into block producer pool...
🔐 [PQC] Setting validator keypair for all 8 producers...
✅ [PQC] Validator keypair sent to all producers
✅ PQC block signing ACTIVATED for all producers!

🔐 Registering validator public keys in registry...
✅ Validator public keys registered (Node ID: 6c4c0671...)
```

---

## Step 4: Verify PQC is Working

### Monitor Block Signing

Watch for these logs during block production:

```bash
# Ed25519 signing (Phase 0)
🔐 [PQC] Signed block with Ed25519 (Phase 0)

# Dilithium5 signing (Phase 1)
🔐 [PQC] Signed block with Dilithium5 (Phase 1) - 4595 bytes

# Hybrid signing (Transition period)
🔐 [PQC] Signed block with Hybrid Ed25519+Dilithium5
   Ed25519 signature: 64 bytes
   Dilithium5 signature: 4595 bytes
```

### Monitor Signature Verification

When receiving blocks from the network:

```bash
# Successful verification
🔐 [PQC] Block 123 has 1 spectral signatures - verifying...
✅ [PQC] Signature 0 verified for block 123 (phase: Phase0Ed25519)
✅ [PQC] All 1 signatures verified for block 123

# Skipped verification (no keys for validator)
⚠️  [PQC] No public keys for validator 1234... - skipping verification
🔐 [PQC] Block 124 signature 0 skipped (no keys)

# Failed verification (invalid signature)
❌ [PQC] Signature 0 verification FAILED for block 125: InvalidSignature
   Validator: 6c4c0671...
   Phase: Phase0Ed25519
   Block REJECTED due to invalid PQC signature!
```

---

## Step 5: Systemd Service Configuration

### Update Service File

Edit `/etc/systemd/system/q-api-server.service`:

```ini
[Unit]
Description=Q-NarwhalKnight API Server with PQC
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/opt/orobit/shared/q-narwhalknight
ExecStart=/opt/orobit/shared/q-narwhalknight/target/release/q-api-server \
  --validator-key /etc/q-narwhalknight/validator.json \
  --port 8080 \
  --p2p-port 9001 \
  --network testnet-phase11
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```

### Reload and Restart

```bash
# Reload systemd configuration
systemctl daemon-reload

# Restart service
systemctl restart q-api-server

# Check status
systemctl status q-api-server

# View logs
journalctl -u q-api-server -f --since "5 minutes ago"
```

---

## Troubleshooting

### Issue: "Failed to load validator keypair"

**Symptoms:**
```
❌ Failed to load validator keypair from /path/to/validator.json: No such file or directory
```

**Solution:**
1. Check file path is correct
2. Verify file exists: `ls -l /path/to/validator.json`
3. Check permissions: `chmod 600 /path/to/validator.json`

---

### Issue: "No PQC signatures on blocks"

**Symptoms:**
```
ℹ️  [PQC] Block 123 has no spectral signatures (validator may not have PQC key)
```

**Solution:**
1. Verify `--validator-key` flag is set
2. Check startup logs for "PQC block signing: ENABLED"
3. Regenerate keypair if file is corrupt

---

### Issue: "Signature verification always skipped"

**Symptoms:**
```
⚠️  [PQC] No public keys for validator - skipping verification
```

**Solution:**
1. Verify validator public keys are registered in registry
2. Check startup logs for "Validator public keys registered"
3. For multi-validator networks, implement key distribution protocol (pending)

---

### Issue: "Performance degradation with PQC"

**Symptoms:**
- Slow block production
- High CPU usage

**Solution:**
1. Dilithium5 signatures are ~500µs per block (acceptable)
2. Verification is ~800µs per signature (acceptable for <100 validators)
3. For 100+ validators, batch verification optimization needed

---

## Security Best Practices

### 1. Key Storage

✅ **DO:**
- Store keypair in `/etc/q-narwhalknight/` with 600 permissions
- Encrypt backups with GPG
- Keep offline backup in secure location

❌ **DON'T:**
- Store in `/tmp/` or world-readable directory
- Commit to Git repository
- Share via unencrypted channels

### 2. Key Rotation

**Currently:** Manual rotation (regenerate keypair)

**Planned:** Automatic rotation protocol (v1.0.17-beta+)

**To rotate keys manually:**
```bash
# 1. Generate new keypair
cargo run --package q-types --example generate_validator_key /etc/q-narwhalknight/validator-new.json

# 2. Stop service
systemctl stop q-api-server

# 3. Backup old keypair
mv /etc/q-narwhalknight/validator.json /etc/q-narwhalknight/validator-old.json

# 4. Replace with new keypair
mv /etc/q-narwhalknight/validator-new.json /etc/q-narwhalknight/validator.json

# 5. Restart service
systemctl start q-api-server

# 6. Verify new key is active
journalctl -u q-api-server | grep "Node ID"
```

### 3. Monitoring

**Monitor these metrics:**
- Block signing success rate (should be 100%)
- Signature verification success rate
- Performance (signing/verification time)
- Registry size (number of validators)

**Alert on:**
- Signature verification failures
- Missing keypair file
- Abnormal performance degradation

---

## Advanced Configuration

### Change Preferred Signature Phase

**Edit validator keypair JSON:**
```json
{
  "node_id": [...],
  "preferred_phase": "Phase1Dilithium5"  // Change this
}
```

**Available phases:**
- `"Phase0Ed25519"` - Classical Ed25519 only (64 bytes)
- `"Phase1Dilithium5"` - Post-quantum Dilithium5 only (4595 bytes)
- `"HybridEd25519Dilithium5"` - Both signatures (4659 bytes)

**Restart after changing:**
```bash
systemctl restart q-api-server
```

---

## FAQ

### Q: Do I need PQC for mining?

**A:** No. Miners don't need PQC keys. Only validators who produce blocks need them.

### Q: What happens if I lose my keypair?

**A:** You cannot sign blocks until you generate a new keypair. Keep secure backups!

### Q: Can multiple nodes use the same keypair?

**A:** No! Each validator must have a unique keypair. Sharing keys is a security risk.

### Q: How do I know my blocks are PQC-signed?

**A:** Check logs for "Signed block with" messages. Query block via API and check `spectral_signatures` field.

### Q: Will old nodes accept my PQC-signed blocks?

**A:** Yes! Nodes without PQC skip verification gracefully. But they cannot verify signature validity.

### Q: When should I use Dilithium5 vs Ed25519?

**A:**
- **Ed25519** (Phase 0): Faster, smaller, classical security
- **Dilithium5** (Phase 1): Slower, larger, post-quantum security
- **Hybrid**: Maximum security, best for high-value validators

---

## API Examples

### Query Block with PQC Signature

```bash
curl http://localhost:8080/api/block/50001 | jq '.quantum_metadata.spectral_signatures'
```

**Example Response:**
```json
[
  {
    "validator": "6c4c0671405cb0bf282c6894c72083feb088a432afee9b14e101689dfee55549",
    "crypto_phase": "Phase0Ed25519",
    "classical_sig": "3045022100...",
    "pqc_sig": null,
    "spectral_coefficient": 1.0,
    "phase_deviation": 0.0,
    "timestamp": 1731702000
  }
]
```

### Query Validator Registry

```bash
curl http://localhost:8080/api/validators/keys | jq
```

**Example Response:**
```json
{
  "6c4c0671...": {
    "ed25519_key": "aabbccdd...",
    "dilithium5_key": "11223344..."
  }
}
```

---

## Next Steps

1. **Test PQC functionality** - Verify end-to-end signing and verification
2. **Monitor production** - Watch for signature verification failures
3. **Implement key distribution** - For multi-validator networks (v1.0.17-beta)
4. **Add performance metrics** - Prometheus counters for monitoring

---

**Document Version:** v1.0.16-beta
**Last Updated:** 2025-11-15
**Status:** Production-ready for single-validator deployment
