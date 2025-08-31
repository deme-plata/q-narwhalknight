# Multi-stage build for Q-NarwhalKnight
FROM rust:1.75-slim as builder

# Install system dependencies
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    protobuf-compiler \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy Cargo files
COPY Cargo.toml Cargo.lock ./
COPY crates ./crates

# Build the application
RUN cargo build --release --bin q-api-server

# Runtime stage
FROM debian:bookworm-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    libssl3 \
    && rm -rf /var/lib/apt/lists/*

# Create app user
RUN useradd -r -s /bin/false qnarwhal

# Create app directory
WORKDIR /app

# Copy binary from builder stage
COPY --from=builder /app/target/release/q-api-server /app/q-api-server

# Create data directory
RUN mkdir -p /app/data && chown qnarwhal:qnarwhal /app/data

# Switch to app user
USER qnarwhal

# Expose ports
EXPOSE 8080 8081

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
  CMD curl -f http://localhost:${Q_API_PORT:-8080}/health || exit 1

# Start the application
CMD ["./q-api-server"]