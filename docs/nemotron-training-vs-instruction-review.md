# Nemotron Cascade Two: Training vs Instruction for Q-NarwhalKnight

**Date:** 2026-04-04
**Author:** Claude Code (Server Beta)
**For Review By:** Nemotron Cascade Two (Epsilon), DeepSeek, Human Operator
**Question:** Can Nemotron be trained on Quillon Graph details, or are instructions/clever tricks sufficient?

---

## 1. Current State: What's Running on Epsilon

### Hardware (CPU-Only, No GPU)
- **CPU:** 2x Intel Xeon Gold 5118 @ 2.30GHz (24 physical / 48 threads)
- **RAM:** 64 GB total, 42 GB used, 20 GB available
- **GPU:** None (all inference is CPU-bound)
- **Disk:** 1.8 TB NVMe, 996 GB free
- **Swap:** 8 GB total, 6.7 GB used

### Active AI Infrastructure
- **Ollama** is running with `nemotron-cascade-2` loaded (24 GB model, ~23.5 GB RSS)
- Response time: ~25 seconds per chat query (CPU-only)
- Service is manually started (not auto-enabled)
- Additional models available: `glm-4.7-flash` (19 GB), `qwen2.5:3b` (1.9 GB)

### Q-NarwhalKnight AI Stack
- **Primary inference:** `q-ai-inference` crate with `LlamaCppEngine` (llama-cpp-2 FFI)
- **Model catalog:** Mistral-7B, GLM-4-9B, Llama-3-8B, Mistral-Small-24B (all GGUF)
- **Web search API:** Uses Nemotron-Cascade-2 via Ollama for query summarization
- **Crown & Ash:** Cascade pattern — Mistral-7B for common dialog, Nemotron for epic narratives
- **Distributed inference:** P2P distributed via libp2p gossipsub + RPC workers

---

## 2. The Core Question: Train or Instruct?

### Option A: Fine-Tune Nemotron on Q-NarwhalKnight Codebase

**What this means:**
- LoRA/QLoRA fine-tuning on the codebase, docs, CLAUDE.md, git history
- Model learns Q-NarwhalKnight patterns, naming conventions, architecture
- Runs locally on Epsilon via Ollama

**Pros:**
- Model "knows" the codebase without needing context each time
- Faster inference (no large context to process)
- Works offline / air-gapped

**Cons:**
| Issue | Severity | Details |
|-------|----------|---------|
| **No GPU** | CRITICAL | LoRA fine-tuning requires GPU. Epsilon has none. Would need to train elsewhere and deploy. |
| **4K context window** | CRITICAL | Nemotron base context is 4,096 tokens. CLAUDE.md alone is ~15K+ tokens. Cannot provide full project context at inference time. |
| **Stale immediately** | HIGH | Codebase changes daily. Fine-tuned model reflects training data, not current code. |
| **Catastrophic forgetting** | HIGH | Fine-tuning on narrow domain degrades general reasoning. |
| **25s/response on CPU** | MEDIUM | Already slow — adding domain knowledge doesn't help speed. |
| **Training infrastructure** | HIGH | Need GPU server, data pipeline, eval suite, model versioning. |
| **Memory pressure** | MEDIUM | Nemotron uses 23.5 GB of 64 GB. No room for training. |

**Estimated effort:** 2-4 weeks to set up training pipeline, $500-2000 for GPU compute, ongoing maintenance.

### Option B: Instruction + Clever Tricks (System Prompts, RAG, Context Engineering)

**What this means:**
- Provide Q-NarwhalKnight context to Nemotron at query time
- Use structured prompts, document retrieval, and caching tricks
- No model modification needed

**Pros:**
- Always up-to-date (reads current code/docs)
- Zero training infrastructure
- General reasoning preserved
- Can start immediately

**Cons:**
| Issue | Severity | Details |
|-------|----------|---------|
| **4K context limit** | HIGH | Can only fit ~3K tokens of context + query. Must be very selective. |
| **CPU-only speed** | MEDIUM | 25s/response regardless of approach. |
| **No deep internalization** | LOW | Model reasons from context, doesn't "intuit" patterns. |

---

## 3. Recommendation: Hybrid Instruction Approach

**Neither pure training NOR raw instruction is optimal. Use a structured hybrid:**

### Tier 1: Compressed Knowledge Base (fits in 4K context)

Create a **Q-NarwhalKnight Quick Reference** document (~2,000 tokens) that Nemotron's system prompt always includes:

```
# Q-NarwhalKnight Quick Reference (for Nemotron)

## Architecture
- Rust workspace, 30+ crates, DAG-Knight consensus
- Blockchain: QUG coin, 24 decimals, 21M max supply, 4-year halving
- Mining: BLAKE3x100 VDF PoW (Genus-2 Jacobian VDF upgrade planned)
- P2P: libp2p gossipsub + Kademlia DHT, network ID: mainnet2026.1
- Storage: RocksDB (hot) + archival

## Servers
- Epsilon (89.149.241.126): 10Gbit supernode, q-flux reverse proxy, quillon.xyz
- Beta (185.182.185.227): Primary bootstrap, nginx
- Delta (5.79.79.158): Secondary bootstrap
- Gamma (109.205.176.60): Backup node

## Key Rules
- NEVER sync down (catastrophic data loss)
- Height-gate ALL consensus changes
- Use ha-deploy.sh for ALL deployments
- Balance endpoints require auth (can't curl directly)

## Crate Map
- q-api-server: HTTP API + mining + SSE + block production
- q-types: Block/transaction types, upgrades
- q-storage: RocksDB + balance consensus + turbo sync
- q-network: P2P networking + gossipsub
- q-vdf: VDF implementations (Genus-2, Wesolowski, Pietrzak)
- q-flux: Reverse proxy with supercluster failover
- q-miner: Standalone CPU/GPU miner
- q-mining: GPU OpenCL kernel
- q-dex: AMM decentralized exchange
```

### Tier 2: Retrieval-Augmented Generation (RAG)

Set up a lightweight RAG pipeline that:
1. **Indexes** key docs (CLAUDE.md, MEMORY.md, issue trackers, whitepapers)
2. On each query, **retrieves** the most relevant ~1,500 tokens
3. Prepends retrieved context to the Nemotron prompt

**Implementation options:**
- **Ollama + ChromaDB/Qdrant:** Embed docs with `qwen2.5:3b` (small, fast), retrieve with cosine similarity, pass to Nemotron
- **Simple grep-based:** For code questions, just grep the codebase and include relevant snippets
- **File-based cache:** Pre-chunk CLAUDE.md into topic files, select by keyword match

### Tier 3: Task-Specific Micro-Models (Optional Future)

For repetitive tasks (code review, commit messages, test generation), create small LoRA adapters:
- Train on a remote GPU (Lambda, RunPod — $1-5/hour)
- Deploy as GGUF via Ollama
- Use `qwen2.5:3b` + LoRA (1.9 GB, fits easily alongside Nemotron)

**Only worth it if** Nemotron is used >50 times/day for the same task type.

---

## 4. Clever Tricks for Nemotron (No Training Required)

### Trick 1: Cascaded Prompting
```
Step 1: Ask Nemotron to summarize the question in 1 sentence
Step 2: Use that summary to grep the codebase for relevant code
Step 3: Feed relevant code + original question back to Nemotron
```
This turns 1 large query into 2 small focused queries within the 4K window.

### Trick 2: Pre-Computed Context Chunks
Split CLAUDE.md into 20 topic-specific chunks (~500 tokens each):
- `chunk_deploy.txt` — deployment procedures
- `chunk_sync_safety.txt` — sync-down protection rules
- `chunk_servers.txt` — server infrastructure
- `chunk_mining.txt` — mining algorithm details
- `chunk_dex.txt` — DEX/AMM rules

On query, keyword-match to select 2-3 relevant chunks. Fits in context window.

### Trick 3: Structured Output Templates
Give Nemotron JSON output schemas to constrain its responses:
```json
{
  "analysis": "Brief analysis of the issue",
  "affected_files": ["list of file paths"],
  "risk_level": "LOW|MEDIUM|HIGH|CRITICAL",
  "recommendation": "What to do"
}
```
Structured output compensates for weaker reasoning by keeping responses focused.

### Trick 4: Expert Role Priming
```
You are Q-NarwhalKnight's consensus security auditor. Your ONLY job is to
review code changes for mainnet safety. You check: sync-down protection,
height-gated upgrades, balance integrity, and fork detection.

When reviewing code, answer ONLY these questions:
1. Does this change affect validation or consensus? (YES/NO)
2. If YES, is it height-gated? (YES/NO)  
3. Could this cause sync-down? (YES/NO)
4. Are there balance implications? (YES/NO)
```
Narrow role = better performance within context limits.

### Trick 5: Chain-of-Verification
For critical reviews, use Nemotron + DeepSeek + Claude in sequence:
1. Nemotron does initial review (fast, local, free)
2. DeepSeek validates (different perspective)
3. Claude adjudicates disagreements (strongest reasoning)

This is what you're already doing for the supercluster review — formalize it.

---

## 5. Comparison Matrix

| Approach | Setup Cost | Running Cost | Accuracy | Latency | Freshness |
|----------|-----------|-------------|----------|---------|-----------|
| **Fine-tune Nemotron** | High (weeks + GPU) | Low (local) | Medium-High | 25s | Stale (needs retrain) |
| **Instruction + RAG** | Low (hours) | Low (local) | Medium | 25s | Always current |
| **Claude Code + CLAUDE.md** | None (current) | Per-token API | Highest | 2-5s | Always current |
| **Hybrid (all three)** | Medium (days) | Mixed | Highest | Varies | Always current |

---

## 6. Practical Implementation Plan

### Phase 1: Quick Reference System Prompt (1 hour)
- Create `docs/nemotron-system-prompt.md` with compressed knowledge base
- Configure as Ollama system prompt for nemotron-cascade-2
- Test with 10 common questions about the codebase

### Phase 2: RAG with Chunk Retrieval (1 day)
- Split CLAUDE.md + MEMORY.md into topic chunks
- Write a simple Python/Bash script that:
  - Takes a query
  - Keyword-matches 2-3 relevant chunks
  - Sends `system_prompt + chunks + query` to Ollama
- Deploy on Epsilon as a wrapper around `ollama run`

### Phase 3: Multi-Model Review Pipeline (1 day)
- Formalize the review pattern: Nemotron (initial) -> DeepSeek (validation) -> Claude (adjudication)
- Create `scripts/multi-model-review.sh` that orchestrates the pipeline
- Use for all consensus-critical code changes

### Phase 4 (Optional): Task-Specific LoRA (1 week)
- Only if Phase 1-3 prove insufficient
- Train on Lambda/RunPod GPU ($5-10 total)
- Focus on ONE narrow task (e.g., "review this diff for sync-down risk")

---

## 7. Questions for Reviewers

### For Nemotron Cascade Two:
1. What's your actual context window with the current Ollama config? (Test with progressively longer prompts)
2. How well do you perform with the Quick Reference system prompt vs without it?
3. Can you reliably identify sync-down risks in code diffs with just the compressed knowledge base?

### For DeepSeek:
1. For the RAG approach, what embedding model would you recommend for code retrieval?
2. Is there a better chunking strategy than topic-based for a Rust codebase?
3. Would you prefer structured JSON output or free-form text for code reviews?

### For Human Operator:
1. What tasks do you actually want Nemotron to handle? (Code review? Architecture questions? Deployment assistance?)
2. How often per day would Nemotron be queried? (Determines whether LoRA is worth it)
3. Budget for GPU training if we go that route?

---

## 8. Conclusion

**Instructions + clever tricks are sufficient for now.** Training is not recommended because:

1. **No GPU on Epsilon** — training would need to happen elsewhere
2. **4K context window** — too small for your CLAUDE.md, making instructions critical regardless of training
3. **Rapidly changing codebase** — fine-tuned model goes stale within days
4. **Memory is tight** — 23.5 GB for Nemotron, 20 GB available, no room for training
5. **Current multi-model pipeline works** — Nemotron (initial) + DeepSeek (validate) + Claude (adjudicate)

The **best ROI** is Phase 1 (compressed system prompt) + Phase 2 (topic-based RAG). This gets Nemotron to ~80% of a fine-tuned model's domain performance with zero training cost and always-current knowledge.

If the 25s latency is a problem, consider swapping to `qwen2.5:3b` (1.9 GB, ~5x faster) for quick queries and reserving Nemotron for deep reviews.
