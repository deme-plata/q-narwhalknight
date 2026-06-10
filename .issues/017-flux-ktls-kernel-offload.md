# Issue #017: kTLS Kernel TLS Offload

**Status**: Partial (config + detection done, key extraction pending rustls API)
**Priority**: Medium
**Component**: q-flux
**Labels**: performance, tls, kernel

## Description

Linux 4.13+ supports kTLS (kernel TLS) which offloads TLS record encryption/decryption to the kernel. With `sendfile()` support, encrypted static file serving can go directly from page cache to NIC without userspace copies. For dynamic content, the kernel handles record framing, reducing context switches.

## Benefits

- **Static files**: Zero-copy TLS via `sendfile()` — data never enters userspace
- **Dynamic content**: Kernel handles record framing, fewer context switches
- **Measured gains**: 2-5% CPU reduction on TLS-heavy workloads (Nginx reports similar)
- **Hardware acceleration**: AES-NI instructions used by kernel automatically

## Approach

1. After TLS handshake completes (rustls), extract the symmetric keys
2. Call `setsockopt(SOL_TLS, TLS_TX, ...)` to hand keys to kernel
3. For TX: kernel encrypts records transparently on `write()`/`sendfile()`
4. For RX (optional): kernel decrypts records transparently on `read()`
5. Feature-gate behind `#[cfg(target_os = "linux")]` + runtime detection

## Key Challenges

- rustls doesn't expose raw symmetric keys easily — may need unsafe extraction
- kTLS only supports specific cipher suites (AES-128-GCM, AES-256-GCM, CHACHA20-POLY1305)
- Fallback to userspace TLS if kTLS setup fails (graceful degradation)
- Testing: Need to verify correctness with both paths

## Files to Change

- `crates/q-flux/src/acceptor.rs` — kTLS setup after TLS handshake
- `crates/q-flux/src/static_serve.rs` — use `sendfile()` when kTLS active
- `crates/q-flux/src/config.rs` — `enable_ktls: bool` config option
- `crates/q-flux/Cargo.toml` — feature gate `ktls`

## References

- Linux kTLS: https://docs.kernel.org/networking/tls.html
- Cloudflare kTLS blog: https://blog.cloudflare.com/ktls-and-ssl_sendfile/
