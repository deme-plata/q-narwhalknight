# x-algorithm-scorer

HTTP sidecar that scores tweet drafts against xAI's open-source ranking algorithm. Returns per-action engagement probabilities + negative-signal risk + variant suggestions.

**Companion to PR #88 (Twitter MCP) — Layer 1 (scaffold).**

## API

```
POST /score
  Content-Type: application/json
  Body: { "draft_text": "...", "context": { ... } }
  Returns: ScoreResponse { predicted_engagement, negative_signal_risk,
                           per_action_probabilities, variant_suggestions,
                           model_version }

GET /health → 200 "ok"
GET /version → { layer, model, description }
```

## Two layers

### Layer 1 (this scaffold) — heuristic stub

Transparent heuristics calibrated against xAI's published per-action weight scheme:
- Engagement-positive: 80–200 char length, question marks, links, @-mentions
- Engagement-negative: all-caps density, excess punctuation, inflammatory keywords

Returns scores in the same shape and range as Layer 2 will, so MCP-side callers don't need to change.

### Layer 2 (follow-up, ~6 weeks of work) — xAI Phoenix engine

Wraps `github.com/xai-org/x-algorithm`'s Python ranking head as a long-lived subprocess. Loads the ~3 GB of pretrained model artifacts (Git LFS). Exposes the same HTTP API on the same endpoint contract.

## Build + run

```bash
cd tools/quillon-twitter-mcp/crates/x-algorithm-scorer
cargo build --release
./target/release/x-algorithm-scorer --port 8090
```

Test:
```bash
curl -sX POST http://localhost:8090/score \
  -H 'Content-Type: application/json' \
  -d '{"draft_text":"Just landed AFL-1 protocol spec — what changes for chains without agent priorities? https://github.com/deme-plata/q-narwhalknight/pull/91"}' | jq
```

## Test suite

Five unit tests covering: empty/short text, well-sized question, all-caps SCAM/FRAUD, over-long text, link presence. Run with `cargo test`.

## License

Apache-2.0 (same as Quillon Graph and the xAI x-algorithm repo this sidecar will wrap in Layer 2).
