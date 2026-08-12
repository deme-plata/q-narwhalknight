# Untracked-but-referenced build paths

A plain `git clone` or `git worktree add` of this repo is missing some files/directories that
tracked `Cargo.toml`s reference. Two different reasons — don't conflate them.

## `crates/q-quillon-bank/` and `crates/q-quillon-bank-cli/` — INTENTIONALLY excluded, proprietary

`.gitignore` excludes these **by name**, labeled `# Proprietary Quillon Bank CLI`. This is a
deliberate confidentiality boundary, not an oversight — a public/GitHub checkout of this repo
is **not expected to build the full binary** as a result. Do not add these paths to git, do not
"fix" this by committing them, and do not push them to any public remote. If a build needs them,
get the crate contents through a channel that isn't a public git history.

(A 2026-08-12 session note briefly suggested committing these as a build-hygiene fix — that was
wrong; it didn't check why the exclusion existed before recommending undoing it. Corrected here.)

## `mistral.rs/` — vendored LLM inference dependency (used by `q-ai-inference`)

Also deliberately excluded (`.gitignore` line 2) — it's a full upstream project with its own git
history, not source this repo owns. To restore it for a build:

```bash
git clone https://github.com/EricLBuehler/mistral.rs.git
cd mistral.rs
git checkout eb90e9020208be0a44f51f7b2c227030660e62adf1
git apply ../mistral.rs.local-patches.patch   # 4-file Mistral3 GGUF arch-detection patch
```

The patch (committed alongside this file) teaches mistral.rs's GGUF device-map loader to
recognize `GGUFArchitecture::Mistral3` (it uses the same tensor structure as Llama, so four
spots in `gguf_metadata.rs`/`gguf/mod.rs`/`quantized_llama.rs`/`pipeline/gguf.rs` needed the
extra match arm). Small (25 insertions / 5 deletions), not yet upstreamed.

Do NOT commit `mistral.rs/target/` — that's 3.5 of the 3.7 GB on disk (build artifacts, not
source).

## `crates/q-ai-inference/src/simple_kv_cache.rs` — genuine gap, now fixed

Single file; its `pub mod simple_kv_cache;` declaration was committed but the file itself
wasn't (an incidental collision with the `.gitignore` `simple_*` scratch-file pattern, not
intentional). Force-added 2026-08-12.

## Verifying you have everything before a fresh build

```bash
git status --porcelain --ignored crates/ | grep -v '\.bak\|\.pre-\|q-quillon-bank\|mistral.rs'
```

Anything left after filtering out the known-intentional exclusions above and the old
`.bak*`/`.pre-*` editor backups is a real gap — investigate before assuming a build failure is
a code bug.
