# Q-NarwhalKnight Bootstrap Node - Dockerfile
# v2.9.0-beta: Multi-bootstrap decentralization
#
# This Dockerfile builds a lean bootstrap node image optimized for:
# - Fast blockchain sync
# - P2P peer serving
# - Low resource usage (no mining, no AI inference)

# Build stage
FROM rust:1.75-bookworm AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    pkg-config \
    libssl-dev \
    libclang-dev \
    clang \
    protobuf-compiler \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /build

# Copy workspace files
COPY Cargo.toml Cargo.lock ./
COPY crates ./crates

# Build release binary (only q-api-server for bootstrap node)
RUN cargo build --release --package q-api-server --bin q-api-server

# Runtime stage - minimal image
FROM debian:bookworm-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    libssl3 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 qnk

# Create data directories
RUN mkdir -p /data /identity && chown -R qnk:qnk /data /identity

# Copy binary from builder
COPY --from=builder /build/target/release/q-api-server /usr/local/bin/

# Set ownership
RUN chown qnk:qnk /usr/local/bin/q-api-server

# Switch to non-root user
USER qnk

# Set working directory
WORKDIR /data

# Expose ports
EXPOSE 8080 9001

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8080/api/v1/status || exit 1

# Default environment
ENV RUST_LOG=info \
    Q_NETWORK_ID=testnet-phase16 \
    Q_DATA_DIR=/data \
    Q_API_PORT=8080 \
    Q_P2P_PORT=9001 \
    Q_DISABLE_MINING=true

# Run the bootstrap node
ENTRYPOINT ["/usr/local/bin/q-api-server"]
CMD ["--data-dir", "/data", "--network", "testnet-phase16"]
