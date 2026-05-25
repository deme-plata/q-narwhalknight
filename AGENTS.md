# AGENTS.md - Quillon Graph Agent Instructions

Read this first when working as DeepSeek, Codex, Claude, Grok, Qwen, or another
AI agent in this repository. `CLAUDE.md` contains longer historical and
production notes; this file is the short operational guide.

## First Checks

Always establish where you are before touching files or services:

```bash
hostname
pwd
git status --short
```

Do not assume the current shell is Beta or Epsilon. Do not print secrets,
wallet seeds, authenticated Git remotes, or private tokens.

## Servers

| Name | IP | Role | Important paths |
| --- | --- | --- | --- |
| Beta | `185.182.185.227` | canonical development source, Git server, docs, MCP source | `/opt/orobit/shared/q-narwhalknight/` |
| Epsilon | `89.149.241.126` | production, q-flux, MCP services, Debian 12 release builds | `/home/orobit/q-narwhalknight-src/`, `/home/orobit/target-debian12/`, `/opt/orobit/shared/q-narwhalknight/` |
| Delta | `5.79.79.158` | peer node | `/opt/orobit/shared/q-narwhalknight/` |

On Epsilon, use `/home/orobit` for source, logs, build output, and temporary
files. Avoid `/tmp` and `/root` for large files.

## Source Of Truth And Git Flow

Make normal code changes on Beta:

```bash
cd /opt/orobit/shared/q-narwhalknight
git status --short
```

Beta has multiple remotes. Use configured remote names; do not paste full token
URLs into docs or chat.

Known sync paths from `CLAUDE.md`:

- GitHub remote may be used for cross-server fetches.
- Beta local git daemon: `git://185.182.185.227:9418/q-narwhalknight`.
- Legacy HTTPS `code.quillon.xyz/repo.git` may be broken due TLS/routing issues.

Preferred sync to Epsilon:

```bash
# On Beta, after committing relevant files
git update-server-info

# On Epsilon
cd /home/orobit/q-narwhalknight-src
git fetch origin
git status --short
git merge --ff-only origin/<branch>
```

Use `scp` only for emergency single-file hotfixes when Git is blocked. After an
emergency copy, make a real commit on Beta and sync properly.

## Editing Files

Do not use `sed` for multi-line Rust, TypeScript, TOML, JSON, systemd, or config
edits. It breaks quoting and brace matching.

Preferred methods:

1. Structured patch tool (`apply_patch` in Codex).
2. `git apply` with a small unified diff.
3. A short Python script with exact anchors, count checks, and failure on
   missing anchors.

Safe Python pattern:

```bash
python3 - <<'PY'
from pathlib import Path
p = Path("crates/q-api-server/src/main.rs")
s = p.read_text()
old = """exact old block"""
new = """exact new block"""
count = s.count(old)
if count != 1:
    raise SystemExit(f"anchor matched {count} times")
p.write_text(s.replace(old, new, 1))
PY
```

Rules:

- Read surrounding code before patching.
- Keep edits scoped to the requested files.
- Never revert unrelated dirty files.
- For brace-heavy Rust, patch a whole visible hunk rather than doing partial
  text substitutions.

## Epsilon Debian 12 Release Builds

Never use host `cargo` on Epsilon for release binaries. Use the Debian 12 Docker
builder and the persistent target cache.

Preferred command:

```bash
ssh root@89.149.241.126 '
cd /home/orobit/q-narwhalknight-src
docker run --rm \
  --name qnk-build-$(date +%s) \
  -v /home/orobit/q-narwhalknight-src:/src \
  -v /home/orobit/target-debian12:/src/target \
  -w /src \
  --cpus=16 \
  qnk-debian12:latest \
  bash -lc "cargo build --release --package q-api-server"
'
```

Fallback if `qnk-debian12:latest` is unavailable:

```bash
ssh root@89.149.241.126 '
cd /home/orobit/q-narwhalknight-src
docker run --rm \
  --name qnk-build-$(date +%s) \
  -v /home/orobit/q-narwhalknight-src:/src \
  -v /home/orobit/target-debian12:/src/target \
  -w /src \
  --cpus=16 \
  rust:bookworm \
  bash -lc "apt-get update -qq && apt-get install -y -qq libssl-dev pkg-config cmake clang libudev-dev libclang-dev >/dev/null && export PATH=/usr/local/cargo/bin:\$PATH && cargo build --release --package q-api-server"
'
```

Output:

```text
/home/orobit/target-debian12/release/q-api-server
```

If a long build is already running, do not kill it unless the operator asks.
Check first:

```bash
ssh root@89.149.241.126 'docker ps; ps -eo pid,etime,cmd | grep -E "cargo|rustc|qnk-build" | grep -v grep'
```

## Deploying q-api-server On Epsilon

Only deploy after a successful build and operator approval for the version.

```bash
VERSION=v10.11.34
cp /home/orobit/target-debian12/release/q-api-server \
  /opt/orobit/shared/q-narwhalknight/q-api-server-$VERSION
chmod 755 /opt/orobit/shared/q-narwhalknight/q-api-server-$VERSION

mkdir -p /etc/systemd/system/q-api-server.service.d
cat > /etc/systemd/system/q-api-server.service.d/${VERSION//./-}-pin.conf <<EOF
[Service]
ExecStart=
ExecStart=/opt/orobit/shared/q-narwhalknight/q-api-server-$VERSION --port 8080
EOF

systemctl daemon-reload
systemctl restart q-api-server
systemctl is-active q-api-server
curl -s http://127.0.0.1:8080/engine/pulse
```

Do not stop production before a build. Restart only after the new binary exists.

After any Epsilon restart, verify it opened the correct DB:

```bash
pid=$(pgrep -f q-api-server | head -1)
ls /proc/$pid/fd 2>/dev/null | wc -l
grep Q_DB_PATH /.env
```

Authoritative Epsilon DB path is `/home/orobit/data-mainnet-genesis`.

## MCP

MCP source lives under:

```text
tools/quillon-wallet-mcp/
```

Build:

```bash
cd /opt/orobit/shared/q-narwhalknight/tools/quillon-wallet-mcp
npm run build
```

Agent seeds live on Epsilon at:

```text
/root/.quillon/seeds/<agent>.seed
```

Use `QUILLON_CLIENT=deepseek`, `QUILLON_CLIENT=codex`,
`QUILLON_CLIENT=grok`, etc. Never print seed file contents.

## High-Risk Areas

Balance logic is high risk. Follow the non-negotiable balance rules at the top
of `CLAUDE.md`:

- `save_wallet_balances` must be max-wins.
- Balance replay must gate on `is_checkpoint_applied()`.
- Epsilon wallet balances are authoritative.
- Balance-modifying code needs isolated Docker testing before production.

Current hot files:

- VDF/mining production loop: `crates/q-api-server/src/main.rs`
- Turbo sync/block-pack: `crates/q-network/src/unified_network_manager.rs`
- q-flux MCP routing: `crates/q-flux/src/proxy.rs`, `crates/q-flux/src/h2_proxy.rs`
- MCP tools/auth: `tools/quillon-wallet-mcp/src/index.ts`, `tools/quillon-wallet-mcp/src/wallet_auth.ts`

## Reporting

Keep status reports short:

- host/path used
- files changed
- build/deploy state
- exact blocker if blocked

If another agent or terminal is working in parallel, coordinate through Git and
avoid overwriting or killing their work.
