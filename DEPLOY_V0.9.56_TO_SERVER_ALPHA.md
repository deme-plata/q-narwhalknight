# Deploy v0.9.56-beta to Server Alpha (Docker Container)

## Problem Confirmed

**Server Beta** (185.182.185.227):
- ✅ Running v0.9.56-beta with enhanced logging
- ✅ Sending BlockPackRequests with `Protocol Version: 1`
- ✅ Using correct binary from `/opt/orobit/shared/q-narwhalknight/target/release/q-api-server`

**Server Alpha** (161.35.219.10):
- ❌ Running OLD binary in Docker container `q-api-v0.9.57-altest-beta`
- ❌ Binary was built BEFORE v0.9.56-beta changes
- ❌ Deserializing requests incorrectly (peer ID → start_height)
- ⚠️  Container name says "v0.9.57" but binary is outdated

## Solution

Deploy the v0.9.56-beta binary we just compiled on Server Beta to Server Alpha's Docker container.

---

## Deployment Steps

### Step 1: Copy Binary to Server Alpha

On **Server Beta** (185.182.185.227):

```bash
# The v0.9.56-beta binary is already in downloads folder:
ls -lh /opt/orobit/shared/q-narwhalknight/gui/quantum-wallet/dist-final/downloads/q-api-server-v0.9.56-beta

# Copy to Server Alpha via SCP
scp /opt/orobit/shared/q-narwhalknight/target/release/q-api-server \
    root@161.35.219.10:/tmp/q-api-server-v0.9.56-beta
```

### Step 2: Update Docker Container on Server Alpha

SSH to **Server Alpha** (161.35.219.10):

```bash
ssh root@161.35.219.10
```

Then execute:

```bash
# Stop the running container
docker stop q-api-v0.9.57-altest-beta

# Copy the new binary into the container
# (Assuming container uses /app/q-api-server as the binary path)
docker cp /tmp/q-api-server-v0.9.56-beta q-api-v0.9.57-altest-beta:/app/q-api-server

# Make sure it's executable
docker exec q-api-v0.9.57-altest-beta chmod +x /app/q-api-server

# Verify the binary was copied
docker exec q-api-v0.9.57-altest-beta ls -lh /app/q-api-server

# Start the container
docker start q-api-v0.9.57-altest-beta

# Monitor logs for the enhanced v0.9.56-beta logging
docker logs -f q-api-v0.9.57-altest-beta
```

### Step 3: Verify the Fix

Watch for the enhanced logging in Server Alpha's logs:

```bash
# Should now see enhanced deserialization logging
docker logs q-api-v0.9.57-altest-beta 2>&1 | grep -E "TURBO SYNC DEBUG.*Received BlockPackRequest bytes|NEW format decoded"
```

**Expected output** (if working):
```
🔍 [TURBO SYNC DEBUG] Received BlockPackRequest bytes (len=63): [01 ...]
🔍 [TURBO SYNC DEBUG] Format detection: first_byte=0x01, is_new_format=true
✅ [TURBO SYNC DEBUG] NEW format decoded: protocol_version=1, start=0, end=4999, id=...
```

**If still corrupted**, you'll see:
```
❌ [TURBO SYNC DEBUG] CORRUPTED HEIGHT DETECTED AFTER POSTCARD DECODE!
❌ [TURBO SYNC DEBUG] start_height=1762594154, end_height=...
⚠️  [TURBO SYNC DEBUG] Attempting OLD format decode as fallback...
```

---

## Alternative: Rebuild Docker Image

If the container doesn't allow modifying the binary, rebuild the Docker image:

```bash
# On Server Alpha (161.35.219.10)

# Create a simple Dockerfile
cat > /tmp/Dockerfile-v0.9.56 << 'EOF'
FROM debian:bookworm-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    libssl3 \
    && rm -rf /var/lib/apt/lists/*

# Copy the v0.9.56-beta binary
COPY q-api-server-v0.9.56-beta /app/q-api-server
RUN chmod +x /app/q-api-server

# Expose ports
EXPOSE 8104 8081

# Run the server
CMD ["/app/q-api-server", "--port", "8104"]
EOF

# Build the new image
cd /tmp
docker build -f Dockerfile-v0.9.56 -t q-api:v0.9.56-beta .

# Stop old container
docker stop q-api-v0.9.57-altest-beta
docker rm q-api-v0.9.57-altest-beta

# Run new container
docker run -d \
  --name q-api-v0.9.56-beta \
  --network host \
  -v /opt/orobit/shared/q-narwhalknight/data-alpha:/data \
  -e Q_DB_PATH=/data \
  -e Q_P2P_PORT=8081 \
  -e Q_NETWORK_ID=testnet-phase5 \
  q-api:v0.9.56-beta
```

---

## Expected Outcome

After deployment:

1. ✅ Server Alpha will use v0.9.56-beta binary
2. ✅ Enhanced logging will show complete byte dumps
3. ✅ Early corruption detection will catch any remaining issues
4. ✅ Turbo sync should work correctly between both nodes
5. ✅ If corruption still occurs, detailed diagnostics will pinpoint the exact cause

---

## Troubleshooting

### If corruption persists after deployment:

The enhanced v0.9.56-beta logging will tell us:

1. **Exact bytes being sent** by Server Beta
2. **Exact bytes being received** by Server Alpha
3. **How postcard is interpreting** those bytes
4. **Whether automatic fallback to OLD format** helps

This will definitively identify if:
- There's a network-level corruption
- There's an endianness issue
- There's a postcard serialization bug
- Something else is modifying the bytes in transit

---

## Status

- ✅ v0.9.56-beta compiled on Server Beta
- ✅ Enhanced logging confirmed working on Server Beta
- ⏳ **NEXT**: Deploy to Server Alpha Docker container
- ⏳ **THEN**: Monitor logs to confirm fix
