# Issue #023: HTTP/2 Server Push for Static Assets

**Status**: Planned
**Priority**: Low
**Component**: q-flux
**Labels**: performance, h2

## Description

When q-flux serves an HTML page via static_serve, it could proactively push associated CSS/JS assets to the client via HTTP/2 server push. This eliminates one round-trip for critical rendering resources.

## Approach

1. On `text/html` responses, scan for `<link rel="stylesheet">` and `<script src>` tags
2. Issue PUSH_PROMISE frames for matched assets before sending the HTML body
3. Only push assets that exist in the static root and are hashed (immutable)
4. Respect `Cache-Digest` header to avoid pushing resources the client already has
5. Feature-gate behind config: `[server] h2_server_push = true`

## Benefits

- Eliminates 1 RTT for above-the-fold CSS/JS on first load
- Particularly effective for the quantum-wallet SPA (single HTML + hashed bundles)
- No client-side changes needed

## Files to Change

- `crates/q-flux/src/h2_proxy.rs` — add push logic for static HTML responses
- `crates/q-flux/src/static_serve.rs` — extract asset references from HTML
- `crates/q-flux/src/config.rs` — `h2_server_push` config option
