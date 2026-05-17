# Q-NarwhalKnight Build Automation

# Install development dependencies
dev-setup:
    #!/usr/bin/env bash
    set -euxo pipefail
    # Install Rust nightly
    rustup install nightly
    rustup component add clippy rustfmt
    # Install additional tools
    cargo install cargo-watch cargo-audit
    # Install Docker and Docker Compose
    which docker || echo "Please install Docker manually"
    # Install protobuf compiler
    which protoc || echo "Please install protoc manually"

# Format, lint, and test
check:
    cargo fmt --all -- --check
    cargo clippy --all-targets --all-features -- -D warnings
    cargo test --all-features

# Run all tests
test:
    cargo test --all-features --workspace

# Build all crates
build:
    cargo build --all-features --workspace

# Build in release mode
build-release:
    cargo build --release --all-features --workspace

# Run the API server locally
run-api:
    cargo run --bin q-api-server

# Run a single validator node
run-validator:
    cargo run --bin q-validator

# Spin up local testnet with 4 validators
testnet-up:
    docker compose -f docker/compose-phase0.yml up -d

# Tear down local testnet
testnet-down:
    docker compose -f docker/compose-phase0.yml down

# Run benchmarks
bench:
    cargo bench --workspace

# Run load tester against local testnet
bench-local:
    just testnet-up
    sleep 10  # Wait for validators to start
    cargo run --bin q-load-tester -- --target http://localhost:8080
    just testnet-down

# Clean build artifacts
clean:
    cargo clean

# Update dependencies
update:
    cargo update

# Security audit
audit:
    cargo audit

# Generate documentation
docs:
    cargo doc --all-features --workspace --no-deps

# Watch for changes and run tests
watch:
    cargo watch -x "test --all-features"

# Initialize git repository with proper hooks
git-setup:
    #!/usr/bin/env bash
    git init
    echo "#!/bin/sh\njust check" > .git/hooks/pre-commit
    chmod +x .git/hooks/pre-commit

# Generate new validator keys
keygen:
    cargo run --bin q-keygen

# Default recipe
default: check build