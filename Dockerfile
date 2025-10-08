# Q-NarwhalKnight Quantum Consensus Node
# Historic first Docker deployment of cross-server AI developed system

FROM ubuntu:22.04

# Fix package issues and install system dependencies including Tor
RUN apt-get update --fix-missing || apt-get update -o Acquire::AllowInsecureRepositories=true && \
    apt-get install -y --allow-unauthenticated \
    curl \
    ca-certificates \
    tor \
    supervisor \
    && rm -rf /var/lib/apt/lists/*

# Create application directory
WORKDIR /app

# Copy the compiled binary
RUN mkdir -p target/release
COPY target/release/q-api-server /app/q-api-server

# Create basic configuration
RUN mkdir -p /app/configs

# Create data directories
RUN mkdir -p /app/data/tor /app/data/node /app/logs

# Configure Tor
COPY torrc /etc/tor/torrc

# Configure supervisor to manage both Tor and Q-NarwhalKnight
COPY supervisord.conf /etc/supervisor/conf.d/qnk.conf

# Make binary executable
RUN chmod +x /app/q-api-server

# Expose ports
EXPOSE 8080 8333 8334 9050 9051

# Start supervisor to manage Tor and Q-NarwhalKnight
CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/supervisord.conf"]

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Environment variables
ENV RUST_LOG=info
ENV Q_NODE_DATA_DIR=/app/data/node