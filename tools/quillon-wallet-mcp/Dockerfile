# Quillon Wallet MCP — stdio server image for smithery.ai
#
# This image is what smithery.ai builds when it picks up a push to
# github.com/deme-plata/q-narwhalknight (per smithery.yaml).
# It does NOT include the chain node — it talks to the production
# chain at https://quillon.xyz over HTTPS, signing X-Wallet-Auth
# headers with an Ed25519 keypair derived from the user's seed.
#
# Build:
#   docker build -f tools/quillon-wallet-mcp/Dockerfile -t quillon-wallet-mcp .
#
# Run (locally — smithery wraps this in its own runtime):
#   docker run --rm -i -e QNK_SEED="your-seed-string" quillon-wallet-mcp
#
# Smithery's commandFunction (in smithery.yaml) wraps the user-supplied
# walletSeed into the QNK_SEED env var at runtime, so the container's
# entrypoint just runs `node build/index.js` over stdio.

FROM node:20-alpine AS build

WORKDIR /app

# Install build deps first so changes to source don't bust the layer cache.
COPY package.json package-lock.json ./
RUN npm ci --no-audit --no-fund

# Copy source + TypeScript config, compile to build/.
COPY tsconfig.json ./
COPY src ./src
RUN npm run build

# ── Runtime image — smaller, no toolchain ───────────────────────────
FROM node:20-alpine

WORKDIR /app

# Bring forward only what the runtime needs.
COPY --from=build /app/build ./build
COPY --from=build /app/node_modules ./node_modules
COPY package.json ./

# Smithery passes config via env vars (see smithery.yaml commandFunction).
# QNK_SEED is required; QUILLON_API_BASE defaults to quillon.xyz.
ENV NODE_ENV=production

# MCP servers speak stdio. No port to EXPOSE; smithery wraps the stdio.
CMD ["node", "build/index.js"]
