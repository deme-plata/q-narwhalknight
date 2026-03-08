# PR: OCSP Auto-Fetch + Config Validation + Duration Parse Fix

**Branch**: `feature/safe-batched-sync-v1.0.2`
**Closes**: #019, #020

## Summary

- **OCSP auto-fetch**: New `ocsp_fetch` module automatically fetches OCSP staples from the CA's responder every 12 hours, saving 50-100ms per new TLS connection. Parses the cert's AIA extension for the responder URL, builds the OCSP request DER, and triggers TLS hot-reload on success.
- **Duration parse fix**: `parse_duration("100ms")` was broken — `strip_suffix('s')` matched before `strip_suffix("ms")`. Reordered to check "ms" first.
- **Config validation tests**: 12 new tests for FluxConfig::validate() covering empty backends, invalid addresses, timeout relationships, and limit constraints.
- **OCSP DER encoding tests**: 4 unit tests for ASN.1 DER encoding helpers.

## Test Plan

- [x] `cargo check --package q-flux` — 0 errors, 1 preexisting warning
- [x] `cargo test --package q-flux --lib` — 124 tests pass (up from 108)
- [x] Duration parse test covers all 4 formats: `30s`, `100ms`, `5m`, `60`
- [x] Config validation rejects: empty backends, bad addresses, bad timeouts
- [x] DER encoding: length, sequence, octet string, integer with leading zero

## Files Changed

| File | Change |
|------|--------|
| `Cargo.toml` | Add `x509-parser = "0.16"` workspace dep |
| `crates/q-flux/src/ocsp_fetch.rs` | **New**: 413 lines — OCSP auto-fetch module |
| `crates/q-flux/src/config.rs` | Fix duration parse order, add tests |
| `crates/q-flux/src/main.rs` | Spawn OCSP refresh background thread |
| `crates/q-flux/src/admin.rs` | Add OCSP status to `/status` JSON |
| `crates/q-flux/src/lib.rs` | Register `ocsp_fetch` module |
| `crates/q-flux/Cargo.toml` | Add x509-parser, sha2, reqwest deps |
