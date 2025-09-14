# 📋 GitHub Coordination Instructions
**Q-NarwhalKnight Multi-Server Development**

## 🎯 **Immediate Action Items for Server Alpha**

### **Step 1: GitHub Issues Creation** 
Create these issues **immediately**:

```bash
# Issue #1: Critical q-storage compilation failure
gh issue create --title "🚨 Critical: q-storage crate compilation blocked" \
--body "29+ compilation errors preventing all binary builds. See Q_STORAGE_ERROR_REPORT.md for full details.

Priority fixes needed:
- Missing dependencies (q_quantum_rng, BullsharkCert)
- RocksDB thread safety in async context  
- Core types (NarwhalPayload, Block)
- Serialization issues with Instant

Assigned: Server Alpha - Phase 1 & 2
Labels: critical, compilation-error, q-storage" \
--label "critical" --assignee "@server-alpha"

# Issue #2: Missing core consensus types  
gh issue create --title "🔴 Missing core types: NarwhalPayload, Block, BullsharkCert" \
--body "Storage layer cannot compile due to missing consensus types.

Files affected:
- crates/q-storage/src/lib.rs (6 errors)
- crates/q-storage/src/sync.rs (2 errors)

Fix: Define types in q-types or q-narwhal-core and export properly.
Phase: 1 (Dependency Resolution)" \
--label "critical" --assignee "@server-alpha"

# Issue #3: RocksDB async thread safety
gh issue create --title "🔧 RocksDB ColumnFamily handles not Send+Sync" \
--body "RocksDB ColumnFamily pointers cannot be shared across async boundaries.

Error count: 12+ thread safety violations
Impact: Entire KVStore async trait impl fails

Solution: Replace Arc<ColumnFamily> storage with string-based CF access.
Phase: 2 (RocksDB Thread Safety)" \
--label "critical" --assignee "@server-alpha"

# Issue #4: Serialization failures
gh issue create --title "📦 std::time::Instant serialization failures" \
--body "Instant cannot be serialized with serde, breaking metrics persistence.

Files: metrics.rs, sync.rs  
Solution: Replace with SystemTime or custom serde impl
Phase: 3 (Serialization)" \
--label "high" --assignee "@server-beta"
```

### **Step 2: Branch Creation Strategy**
```bash
# Server Alpha branches
git checkout -b fix/q-storage-compilation-errors
git checkout -b fix/missing-core-types  
git checkout -b fix/rocksdb-thread-safety

# Server Beta branches  
git checkout -b fix/serialization-issues
git checkout -b fix/libp2p-compatibility
git checkout -b fix/testing-integration
```

### **Step 3: Development Workflow**
```bash
# Server Alpha starts with Phase 1
git checkout fix/missing-core-types

# 1. Add q-quantum-rng dependency
echo 'q-quantum-rng = { path = "../q-quantum-rng" }' >> crates/q-storage/Cargo.toml

# 2. Create missing types in q-types or q-narwhal-core
# 3. Export BullsharkCert from q-dag-knight
# 4. Test compilation: cargo check --package q-storage

# Push progress and create PR
git add -A
git commit -m "fix(q-storage): Add missing core types and dependencies

- Add q-quantum-rng dependency to q-storage
- Define NarwhalPayload and Block types
- Export BullsharkCert from q-dag-knight
- Fixes 10+ compilation errors in storage layer

Progress: Phase 1 complete, Phase 2 (RocksDB) next"

git push origin fix/missing-core-types
gh pr create --title "Fix q-storage core type dependencies" --body "Phase 1 fixes for compilation errors"
```

---

## 🏗️ **Server Beta Task Assignment**

### **Phase 3: Serialization Fixes** (Your responsibility)
```bash
git checkout -b fix/serialization-issues

# Files to fix:
# - crates/q-storage/src/metrics.rs
# - crates/q-storage/src/sync.rs  
# - crates/q-storage/src/manifest.rs

# Replace all std::time::Instant with SystemTime
find crates/q-storage -name "*.rs" -exec sed -i 's/std::time::Instant/std::time::SystemTime/g' {} \;
find crates/q-storage -name "*.rs" -exec sed -i 's/Instant::/SystemTime::/g' {} \;

# Update Default implementations to use UNIX_EPOCH
# Test: cargo check --package q-storage
```

### **Phase 4: libp2p Compatibility** (Your responsibility)  
```bash
git checkout -b fix/libp2p-compatibility

# Check current libp2p version
grep libp2p Cargo.toml
grep libp2p crates/*/Cargo.toml

# Update imports in crates/q-storage/src/sync.rs
# Remove RequestId if it doesn't exist in current version
# Test: cargo check --package q-storage
```

### **Phase 5: Comprehensive Testing** (Your responsibility)
```bash
git checkout -b fix/testing-integration

# After all fixes are merged, run full test suite
cargo test --workspace
cargo build --bin q-api-server  
cargo build --bin dagknight
./quick-test.sh

# If successful, run real Q-NarwhalKnight node
./target/debug/q-api-server --config /etc/q-narwhalknight/config.toml
```

---

## 🔄 **Coordination Protocol**

### **Daily Sync Process:**
1. **Morning (UTC 08:00):** Server Alpha posts progress on GitHub issues
2. **Afternoon (UTC 14:00):** Server Beta posts progress and blockers
3. **Evening (UTC 20:00):** Joint review of PRs and next-day planning

### **PR Review Process:**
```bash
# Server Alpha creates PRs for Phases 1 & 2
# Server Beta reviews and tests integration
# After Alpha's fixes are merged, Beta starts Phase 3 & 4

# All PRs require:
# - Successful cargo check --package q-storage
# - Updated error count in PR description  
# - Link to related GitHub issue
```

### **Communication Channels:**
- **GitHub Issues:** Progress updates and technical discussion
- **PR Comments:** Code review and implementation details  
- **Commit Messages:** Detailed progress tracking with error counts

---

## 🎯 **Milestone Tracking**

### **Phase 1 Complete:** (Server Alpha)
- [ ] q-quantum-rng dependency added
- [ ] NarwhalPayload type defined
- [ ] Block type defined  
- [ ] BullsharkCert exported
- [ ] 10+ dependency errors resolved

### **Phase 2 Complete:** (Server Alpha)  
- [ ] RocksDB ColumnFamily storage replaced
- [ ] cf_names() calls updated to list_cf()
- [ ] KVStore async traits compile successfully
- [ ] 12+ thread safety errors resolved

### **Phase 3 Complete:** (Server Beta)
- [ ] Instant replaced with SystemTime
- [ ] Default implementations updated
- [ ] Serde serialization working
- [ ] 5+ serialization errors resolved

### **Phase 4 Complete:** (Server Beta)
- [ ] libp2p imports updated
- [ ] RequestId issue resolved
- [ ] Network sync layer compiles
- [ ] 2+ libp2p errors resolved

### **Final Integration:** (Both)
- [ ] All 29+ errors resolved
- [ ] q-api-server builds successfully
- [ ] dagknight builds successfully
- [ ] Real node runs on port 8082
- [ ] API endpoints respond correctly

---

**🚀 Coordination is ready! Server Alpha should start Phase 1 immediately.**

**Expected timeline:** 2-3 days for complete resolution with parallel development.