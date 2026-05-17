# Issue #019: OCSP Auto-Fetch and Periodic Refresh

**Priority**: High
**Status**: Done
**Labels**: tls, security, performance

## Problem

OCSP stapling eliminates 50-100ms per new TLS connection (client doesn't need to make a separate OCSP request to the CA). q-flux already supports stapling a DER file, but:

1. The OCSP response file had to be generated manually (e.g., via `openssl ocsp`)
2. OCSP responses expire (typically 3-7 days) — no automatic refresh
3. No visibility into OCSP status from the admin panel

## Solution

New `ocsp_fetch` module with automatic OCSP response fetching:

1. **Cert parsing**: Uses `x509-parser` to extract the OCSP responder URL from the certificate's Authority Information Access (AIA) extension
2. **OCSP request building**: Constructs DER-encoded OCSP request with SHA-256 hashes of issuer name/key + certificate serial number
3. **HTTP fetch**: POSTs the OCSP request to the CA's responder via `reqwest`
4. **Atomic write**: Writes response to temp file, renames to target path
5. **TLS hot-reload**: Triggers `SharedTlsConfig::reload()` to pick up the new staple
6. **Background task**: Refreshes every 12 hours on a dedicated thread
7. **Admin visibility**: OCSP status (responder URL, last fetch, bytes, errors) in `/status` JSON

## Files Changed

- `crates/q-flux/src/ocsp_fetch.rs` — New module (413 lines)
- `crates/q-flux/src/main.rs` — Spawn OCSP refresh thread
- `crates/q-flux/src/admin.rs` — Add OCSP status to `/status` endpoint
- `crates/q-flux/src/lib.rs` — Register module
- `crates/q-flux/Cargo.toml` — Add x509-parser, sha2, reqwest deps
- `Cargo.toml` — Add x509-parser to workspace

## Testing

- 4 unit tests for DER encoding (length, sequence, octet string, integer)
- Integration tested via `cargo check --package q-flux` (compiles clean)
- 124 total tests passing

## Configuration

No config changes needed — the OCSP path is auto-generated from the cert path if not explicitly set:

```toml
[tls]
cert = "/etc/letsencrypt/live/example.com/fullchain.pem"
key = "/etc/letsencrypt/live/example.com/privkey.pem"
# Optional: explicit OCSP staple path (auto-generated if omitted)
# ocsp_staple = "/etc/letsencrypt/live/example.com/ocsp.der"
```
