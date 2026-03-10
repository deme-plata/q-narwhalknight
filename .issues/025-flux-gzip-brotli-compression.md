# Issue #025: Response Compression (gzip + Brotli)

**Status**: Done
**Priority**: High
**Component**: q-flux
**Labels**: performance, proxy

## Description

q-flux currently proxies responses verbatim without compression. Adding transparent gzip/Brotli compression for text-based responses (HTML, CSS, JS, JSON) can reduce bandwidth by 60-80%, significantly improving page load times for the wallet frontend and reducing bandwidth costs.

## Approach

1. Check `Accept-Encoding` header for `br` (Brotli) or `gzip` support
2. For compressible MIME types (text/*, application/json, application/javascript), compress on-the-fly
3. Use `flate2` for gzip and `brotli` crate for Brotli
4. Pre-compress static assets at build time for zero-cost serving
5. Skip compression for already-compressed types (images, wasm, video)
6. Skip compression for small responses (< 1KB — compression overhead exceeds savings)
7. Add `Vary: Accept-Encoding` header for correct caching
8. Feature-gate: `[server] compression = true`

## Benefits

- 60-80% bandwidth reduction on text responses
- Faster wallet frontend loads (especially on mobile/slow connections)
- Pre-compressed static files served at wire speed (no CPU overhead)
- Brotli offers 15-20% better compression than gzip for web assets

## Files to Change

- `crates/q-flux/src/proxy.rs` — on-the-fly compression in `write_response`
- `crates/q-flux/src/static_serve.rs` — serve pre-compressed `.br` / `.gz` variants
- `crates/q-flux/src/config.rs` — compression config
- `crates/q-flux/Cargo.toml` — add `flate2`, `brotli` dependencies
