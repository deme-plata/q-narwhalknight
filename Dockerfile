# Q-NarwhalKnight Node — Auto-updating mainnet node image
# The binary is NOT baked in. It is downloaded on first run from quillon.xyz
# and auto-updated in-place via the built-in P2P upgrade mechanism.
#
# Usage:
#   docker run -d -p 8080:8080 -p 9001:9001 \
#     -v qnk-data:/data quillon/q-node:latest

FROM debian:12-slim

RUN apt-get update -qq && \
    apt-get install -y --no-install-recommends \
        curl ca-certificates && \
    rm -rf /var/lib/apt/lists/*

COPY docker-entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# /data is the persistent volume — stores binary, chain data, identity keys
VOLUME ["/data"]

EXPOSE 8080 9001 9002

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=5 \
    CMD curl -sf http://localhost:8080/api/v1/status > /dev/null || exit 1

ENTRYPOINT ["/entrypoint.sh"]
