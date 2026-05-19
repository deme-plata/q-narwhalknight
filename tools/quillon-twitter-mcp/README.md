# Quillon Twitter MCP

MCP server for AI-agent-authored tweets on Quillon Graph. Drafts → scores → admin-approves → posts (future) → attests on-chain (future).

Spec: `docs/twitter-mcp-with-x-algorithm-spec.md`. This package implements **Layers 1-3**.

## Architecture (this package)

```
   AI agent (Claude / Codex / Grok / DeepSeek)
                    │
                    ▼
   ┌────────────────────────────────────┐
   │  Twitter MCP (Node, this package)  │
   │                                    │
   │   • draft_tweet (Layer 2)          │
   │   • score_draft (Layer 1+2)        │
   │   • queue_for_approval (Layer 3)   │
   │   • list_my_drafts (Layer 3)       │
   └────────────────────────────────────┘
            │                    │
       HTTP /score            SQLite
            ▼              ~/.quillon-twitter-mcp/drafts.db
   ┌────────────────────────────────────┐
   │  x-algorithm-scorer (Rust sidecar) │
   │  Layer 1: heuristic stub           │
   │  Layer 2 (future): xAI Phoenix     │
   └────────────────────────────────────┘
```

## Tools exposed

| Tool | What | Side effects |
|---|---|---|
| `draft_tweet` | Score + return suggestions | None |
| `score_draft` | Pure scoring | None |
| `queue_for_approval` | Persist draft, return admin URL | INSERTs into SQLite |
| `list_my_drafts` | Read pending or by-author | None |

## Layers not yet implemented

| Layer | Tool | Requires |
|---|---|---|
| 4 | `publish_approved` | X API OAuth2 setup + admin wallet signature flow |
| 5 | Chain attestation | PR #87 (Agent Fiber Lane) merged + signed-tx submit path |

These land in follow-up PRs once their dependencies are ready.

## Hard limit on negative signal risk

Per spec §5.3: drafts predicted to have `negative_signal_risk > 0.15` are **refused at queue time**. Even the admin cannot approve them — the draft must be revised first.

The threshold is configurable via the `Q_TWITTER_NEG_RISK_MAX` env var.

## Build + run

```bash
# Start the Rust sidecar
cd tools/quillon-twitter-mcp/crates/x-algorithm-scorer
cargo run --release &

# Build + run the MCP server
cd tools/quillon-twitter-mcp
npm install
npm run build
npm start
```

## Configure as a Claude Code MCP server

In your `~/.config/claude-code/mcp.json`:

```json
{
  "mcpServers": {
    "quillon-twitter": {
      "command": "node",
      "args": ["/opt/orobit/shared/q-narwhalknight/tools/quillon-twitter-mcp/build/index.js"],
      "env": {
        "SCORER_URL": "http://127.0.0.1:8090",
        "QTWITTER_APPROVAL_BASE_URL": "https://quillon.xyz/admin/twitter/q",
        "Q_TWITTER_NEG_RISK_MAX": "0.15"
      }
    }
  }
}
```

## Companion: x-algorithm-scorer

The Rust sidecar lives at `crates/x-algorithm-scorer/`. Layer 1 (this commit) uses heuristic scoring; Layer 2 will wrap xAI's Phoenix engine from `github.com/xai-org/x-algorithm` (Apache-2.0, ~3 GB pretrained artifacts).

## License

Apache-2.0 (same as Quillon Graph + xAI x-algorithm).
