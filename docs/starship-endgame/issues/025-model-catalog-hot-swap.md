# Issue #025: Model Catalog & Hot-Swap — Dynamic Model Management

**State**: `in_progress`
**Priority**: MEDIUM
**Labels**: `starship-endgame`, `ai-inference`, `ops`
**Assigned**: Beta
**Branch**: `feature/safe-batched-sync-v1.0.2`
**Created**: 2026-03-10

---

## Description

`model_catalog.rs` defines a static registry of 4 GGUF models (`Mistral7B`, `GLM4Flash`, `Llama3_8B`, `MistralSmall24B`) with auto-download support. But there's no runtime management — models can't be loaded/unloaded without restarting the node. Node operators need to:

1. See which models are available and their disk/VRAM requirements
2. Download new models on-the-fly
3. Load/unload models without restarting (hot-swap)
4. Set per-model pricing in the rate card

## Current State

- `ModelCatalog` has static `CatalogEntry` list with URLs, sizes, SHA256 hashes
- `ModelManager` in `q-ai-inference` manages loaded models but requires restart to change
- `ModelTier` enum (`Small/Medium/Large/XL`) maps to pricing but not to actual model selection
- No API endpoint to list/manage models
- No auto-download triggered by incoming inference request for an unloaded model

## Hot-Swap Flow

```
Operator: POST /api/v1/ai/models/load { model: "mistral-small-24b" }
  → Check disk: model exists? If not → auto-download from catalog
  → Check VRAM: enough free? If not → evict LRU model
  → Load model into inference engine
  → Update ModelTier pricing
  → Model available for requests within ~30s

Operator: POST /api/v1/ai/models/unload { model: "llama3-8b" }
  → Wait for in-flight requests to complete (30s grace period)
  → Unload model, free VRAM
  → Update available model list

Auto-load: Incoming request for unloaded model
  → If model in catalog and VRAM available → auto-load + queue request
  → If VRAM insufficient → reject with 503 + suggest smaller model
```

## Acceptance Criteria

- [ ] `GET /api/v1/ai/models` — List all catalog models with status (available/loaded/downloading)
- [ ] `POST /api/v1/ai/models/load` — Download + load a model (async, returns task ID)
- [ ] `POST /api/v1/ai/models/unload` — Graceful unload with in-flight drain
- [ ] `GET /api/v1/ai/models/:id/status` — Download progress, VRAM usage, request count
- [ ] LRU eviction when VRAM is full and new model requested
- [ ] Auto-download from catalog URL with SHA256 verification
- [ ] Per-model pricing override in rate card config

## Depends On

- #014 (Inference revenue wiring — ModelTier pricing)
- #023 (Multi-GPU scheduling — model-to-GPU assignment)

## Progress

**Current**: model_catalog.rs (761 lines) — ModelCatalog with 4 default GGUF models and extensibility, auto-download from catalog URLs with SHA256 verification. VramBudgetManager with LRU eviction. Hot-swap load/unload with graceful in-flight request draining.

## Files

- `crates/q-ai-inference/src/model_catalog.rs` — ModelCatalog, VramBudgetManager, hot-swap logic
- `crates/q-ai-inference/src/model_manager.rs` — Load/unload implementation
- `crates/q-api-server/src/ai_api.rs` — Model management endpoints
- `crates/q-compute/src/inference_pool.rs` — Model request routing
