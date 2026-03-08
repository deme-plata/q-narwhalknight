# Issue #021: ACME Certificate Automation (Let's Encrypt)

**Priority**: Medium
**Status**: Planned
**Labels**: tls, security, automation

## Problem

TLS certificates must be manually renewed and deployed. Let's Encrypt certs expire every 90 days. If forgotten, the proxy goes down.

## Proposed Solution

Integrate ACME (RFC 8555) certificate management directly into q-flux:

1. **ACME client**: Use `instant-acme` or `acme2` crate for ACME v2 protocol
2. **HTTP-01 challenge**: Serve `/.well-known/acme-challenge/` responses from the proxy itself (no separate web server needed)
3. **Auto-renewal**: Background task checks cert expiry, renews 30 days before expiration
4. **TLS hot-reload**: After renewal, call `SharedTlsConfig::reload()` with new cert
5. **OCSP integration**: Immediately fetch OCSP staple for the new cert via `ocsp_fetch`

## Config

```toml
[tls.acme]
enabled = true
email = "admin@quillon.xyz"
domains = ["quillon.xyz", "www.quillon.xyz"]
# Directory URL (default: Let's Encrypt production)
# directory = "https://acme-v02.api.letsencrypt.org/directory"
# staging = false  # Use staging for testing
```

## Dependencies

- ACME client crate (TBD)
- Already have: x509-parser, reqwest, OCSP auto-fetch

## Complexity

High — ACME protocol has several steps (account creation, order, authorization, challenge, finalization). Recommend starting with HTTP-01 challenge only.
