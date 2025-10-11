# Windows Client Update Instructions

## Issue
Your Windows client is running the OLD executable (built before consensus fixes). That's why transactions still show 0.

## Solution: Replace with New Executable

### Step 1: Stop Windows Node
On Windows machine, press **Ctrl+C** to stop the running node.

### Step 2: Download New Executable

**Option A - If you have SSH/SCP access:**
```powershell
# From Windows machine:
scp user@185.182.185.227:/opt/orobit/shared/q-narwhalknight/target/x86_64-pc-windows-gnu/release/q-api-server.exe C:\q-narwhalknight\q-api-server-new.exe

# Backup old version:
mv C:\q-narwhalknight\q-api-server.exe C:\q-narwhalknight\q-api-server-old.exe

# Use new version:
mv C:\q-narwhalknight\q-api-server-new.exe C:\q-narwhalknight\q-api-server.exe
```

**Option B - Manual Download:**
1. I'll create a copy for you to download via HTTP
2. Download from Linux server: `http://185.182.185.227:9999/downloads/q-api-server.exe`
3. Replace the old exe in `C:\q-narwhalknight\`

### Step 3: Restart Windows Node
```powershell
cd C:\q-narwhalknight
.\q-api-server.exe --port 9999
```

### Step 4: Verify Fix
You should now see:
- **Connected Peers: 1** ✅ (already working)
- **Total Transactions: increases** ✅ (will work after update)

## File Details

**New Executable:**
- **Path on Linux:** `/opt/orobit/shared/q-narwhalknight/target/x86_64-pc-windows-gnu/release/q-api-server.exe`
- **Size:** 78MB
- **SHA256:** `49405bace349a89c042d8adfa6ec9db4b38c4e2e5fb85f152debfcbceef65859`
- **Built:** 2025-10-09 05:57 UTC
- **Changes:** Consensus activated, transactions processed immediately, min_batch_size=1, confirmed tx counting

## Test After Update

Submit a transaction:
```bash
curl -X POST http://localhost:9999/api/transaction \
  -H "Content-Type: application/json" \
  -d '{"from": "alice", "to": "bob", "amount": 100}'
```

Watch the console - **Total Transactions** should increment within 100ms!

## Troubleshooting

**Still showing 0?**
1. Make sure you stopped the old process completely
2. Verify you're running the new exe: `ls -l q-api-server.exe` (should be 78MB)
3. Check logs for worker messages: Look for "🚀 Processing transaction batch"

**Need help transferring the file?**
Let me know which method works best for you (SCP, HTTP download, or USB transfer).
