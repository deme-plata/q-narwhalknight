# Security Audit: VM + DEX Pathing in Quillon Graph

Date: 2026-05-21
Scope: `q-flux` runtime/service configs + proxy pathing relevant to Quillon graph/DEX traffic.

## Executive Summary

- **Overall posture: Moderate-Strong**, with good baseline controls already in place:
  - systemd service hardening (`NoNewPrivileges`, `ProtectSystem=strict`).
  - admin bind to loopback (`127.0.0.1:9090`).
  - path traversal defenses in static serving.
  - upstream health checks + cluster failover.
  - default security response headers.
- **Highest-priority risks** are mostly **configuration hardening gaps**, not obvious code-RCE bugs.

## Findings

### 1) VM service hardening is good, but not maximal (Medium)

Current units set:
- `NoNewPrivileges=true`
- `ProtectSystem=strict`
- constrained write paths

Hardening controls not present that are usually recommended for internet-facing reverse proxies:
- `PrivateTmp=true`
- `ProtectHome=true` (or `read-only` depending artifact paths)
- `ProtectKernelTunables=true`
- `ProtectKernelModules=true`
- `ProtectControlGroups=true`
- `LockPersonality=true`
- `RestrictSUIDSGID=true`
- `RestrictAddressFamilies=` (at least `AF_INET AF_INET6 AF_UNIX`)
- `SystemCallFilter=@system-service`

Impact: extra kernel/userland attack surface if proxy process is compromised.

### 2) Epsilon config lacks explicit per-IP/global rate limiting (Medium)

Delta config includes:
- `rate_limit_per_ip`
- `rate_limit_burst`
- `rate_limit_global_rps`

Epsilon config does not set these values explicitly. If defaults are permissive, this increases DEX/API abuse risk on the primary node.

Impact: higher susceptibility to L7 flood/abuse impacting DEX paths and graph reads.

### 3) Access control defaults to disabled unless configured (Medium)

Code default mode is `disabled` for IP access control. This is fine for public endpoints, but risky for admin/internal-only paths if config drift occurs.

Impact: exposure risk if operators assume deny-by-default behavior.

### 4) Security response headers present but CSP/HSTS are not enforced here (Low-Medium)

Proxy writes:
- `x-content-type-options`
- `x-frame-options`
- `referrer-policy`

Missing in this layer:
- strict `content-security-policy`
- `strict-transport-security` policy

Impact: browser-side risk reduction is incomplete unless handled upstream/app-side.

### 5) Observability graph appears operationally safe (Low)

Grafana dashboard file is a panel definition and does not itself expose credentials/secrets. Primary risk is deployment-side auth/config, not dashboard JSON content.

## Positive Controls Confirmed

- Static file path traversal checks are implemented.
- Admin endpoint loopback-only in sample node configs.
- systemd memory ceilings reduce blast radius under load.
- TLS certs loaded from `/etc/letsencrypt` with readonly systemd path.

## Recommendations (Priority Order)

1. **Harden systemd units further** (add controls in Finding #1).
2. **Align Epsilon with Delta rate-limit controls** and validate effective defaults.
3. **Explicitly configure access control mode** in all production TOMLs (no implicit default reliance).
4. **Add HSTS + CSP strategy** at proxy or app edge for DEX/web graph domains.
5. **Add a security regression checklist** to deployment docs (unit hardening + rate-limit parity + admin bind checks).

## Suggested Quick Patch Set

- `q-flux.service` and `q-flux-delta.service`:
  - add hardening knobs listed in Finding #1.
- `q-flux-epsilon.toml`:
  - set `rate_limit_per_ip`, `rate_limit_burst`, `rate_limit_global_rps` to production baseline.
- all env TOMLs:
  - set `[access_control] mode = "blocklist"` (or `allowlist` where appropriate) explicitly.

## Notes

This audit is static/repo-based and does not include live VM kernel/package validation, runtime socket scan, or external penetration testing.
