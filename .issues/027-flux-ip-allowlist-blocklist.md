# Issue #027: IP Allowlist / Blocklist with CIDR Support

**Status**: Planned
**Priority**: Medium
**Component**: q-flux
**Labels**: security, config

## Description

Add configurable IP allowlist/blocklist with CIDR range support for the proxy and admin server. This enables blocking known bad actors (DDoS sources, scanner bots) and restricting admin access to trusted networks.

## Approach

1. Parse CIDR ranges in config: `blocklist = ["1.2.3.0/24", "5.6.7.8"]`
2. Build a radix trie for O(1) IP lookups (ipnetwork crate)
3. Check incoming connections in `worker_loop` before TLS handshake (saves CPU)
4. Admin server: default to `127.0.0.1/8` allowlist (localhost only)
5. Hot-reload blocklist via admin API (POST /blocklist)
6. Log blocked connections with reason

## Config Example

```toml
[security]
admin_allowlist = ["127.0.0.1/8", "10.0.0.0/8"]
blocklist = ["192.168.1.0/24"]
blocklist_action = "drop"  # "drop" (silent) or "reject" (send 403)
```

## Files to Change

- `crates/q-flux/src/config.rs` — security config section
- `crates/q-flux/src/worker.rs` — IP check before TLS
- `crates/q-flux/src/admin.rs` — admin allowlist + blocklist API
- `crates/q-flux/Cargo.toml` — add `ipnetwork` dependency
