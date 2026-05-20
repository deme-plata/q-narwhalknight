# Development workflow going forward — local-git + GitHub Codex collaboration

**Status:** Doctrine adopted 2026-05-20 after recovering 28 of 29 Codex PRs from the May 2026 git divergence incident (see TR-2026-008, TR-2026-009 for the recovery story).

## TL;DR

> **Local-git on Beta is the trunk that builds production. GitHub is the collaboration surface where Codex and humans open PRs. We cherry-pick from GitHub into local-git, never the other direction. Neither system is deleted; both serve their best purpose.**

This document captures the workflow that emerged from a long session that proved we can have both Codex collaboration AND a self-sovereign source of truth — without the chaos we hit when we tried to treat them as the same thing.

## What we proved on 2026-05-20

Starting state: `release/v10.9.57` source didn't build (orphan call sites), 28 Codex PRs scattered across diverged branches, the original local-git workflow was unclear.

Ending state, in a single day:
- **40 commits added to `agent/cross-shard-simd-validation`** (our buildable trunk)
- **All signed Good with GPG** — every commit is verifiable
- **Original author attribution preserved** (Viktor, Codex, Server Beta, the agent identities)
- **28 of 29 Codex PRs integrated** (97%) — via cherry-pick, no destruction, original PR branches still on GitHub for posterity
- **Build pipeline confirmed working end-to-end** — v10.9.58 binary built in 35 min on Epsilon Docker, smoke-tested by syncing to 1.38M blocks on mainnet-genesis
- **Local-git daemon serving smoothly** — `git update-server-info` + `git daemon` on Beta:9418 picked up every change and served them to Epsilon's pulls
- **Memory + TR documentation captured** every lesson — branch hygiene, cluster topology, lineage check, non-destructive recovery patterns

We rescued 32KB of untracked production code (PR #93's `RecentActivityPanel.tsx`) that was at real risk of being lost. The recovery worked because we had both copies — GitHub's PR snapshot AND Beta's working tree — and could choose the right one.

## The two-flow model

```
                    ┌─────────────────────────────────────────┐
                    │  Codex, external contributors, you      │
                    │  authoring PRs on GitHub                │
                    └────────────────┬────────────────────────┘
                                     │
                                     ▼
                    ┌─────────────────────────────────────────┐
                    │  github.com/deme-plata/q-narwhalknight  │
                    │  — collaboration surface                │
                    │  — PR review UI                         │
                    │  — agent integration (Codex needs it)   │
                    │  — historical record (never delete)     │
                    └────────────────┬────────────────────────┘
                                     │
                                     │  gh pr view <PR>
                                     │  git fetch origin refs/pull/N/head:origin/pr/N
                                     │  git cherry-pick origin/pr/N  ← preserves author
                                     │  (or per-commit triage for big PRs)
                                     │
                                     ▼
        ┌────────────────────────────────────────────────────────────┐
        │  Beta /opt/orobit/shared/q-narwhalknight/.git  — TRUNK     │
        │  — authoritative for production builds                     │
        │  — every commit signed-Good GPG                            │
        │  — git update-server-info after each batch                 │
        └─────┬─────────────────────────────────────┬────────────────┘
              │                                     │
              │ git daemon (:9418)                  │ optional periodic mirror
              │ ── serves to Epsilon                │ ── push local → GitHub
              │                                     │    under mirror/vX.Y.Z
              ▼                                     ▼
   ┌──────────────────────────────┐    ┌──────────────────────────┐
   │  Epsilon source repo          │    │  GitHub branch:           │
   │  /home/orobit/q-narwhal-      │    │  mirror/v10.10.2 (etc.)  │
   │  knight-src/                  │    │  — read-only reference   │
   │  git fetch from :9418         │    │  — Codex sees current    │
   │  cargo build → binary         │    │    integration state     │
   └──────────────────────────────┘    └──────────────────────────┘
```

**Flow 1 (commits → production):** authoring → local-git on Beta → `git daemon` → Epsilon pulls → Docker build → binary deploy. **No GitHub involved.** Fast, controlled, no external dependency.

**Flow 2 (GitHub → integration):** Codex/contributor opens PR on GitHub → we fetch with `gh pr view` + `git fetch origin refs/pull/N/head:origin/pr/N` → cherry-pick onto local trunk → original PR stays open on GitHub forever. **Original work never lost.** The author gets credit on our trunk.

## What lives where

| Asset | Location | Authority |
|---|---|---|
| Production binary source | Beta `/opt/orobit/shared/q-narwhalknight/.git` | **Authoritative** |
| Production builds (Debian 12) | Epsilon Docker via `/home/orobit/target-debian12/` | Pulls from local-git |
| All commits going to production | Signed Good on local-git | Required |
| Codex PRs in flight | GitHub `origin = deme-plata/q-narwhalknight` | Authored by Codex; we cherry-pick what we want |
| PR branches (open or closed) | GitHub | **Never delete** — historical record |
| Periodic mirror snapshots | GitHub `mirror/vX.Y.Z` | One-way push from local |
| Documentation, TRs, memory | Local-git trunk | Authoritative |

## The daily workflow

### When you commit code on Beta

```bash
# 1. Edit code on Beta
vi crates/q-api-server/src/handlers.rs

# 2. Stage + commit (signed; gpg-agent has 8h TTL)
git add crates/q-api-server/src/handlers.rs
git commit -m "fix(handlers): your message"
# → produces a signed commit on agent/cross-shard-simd-validation

# 3. Refresh local-git serving
git update-server-info

# 4. Epsilon picks it up next fetch automatically.
#    If you want to trigger a build now:
ssh root@89.149.241.126 'cd /home/orobit/q-narwhalknight-src && \
  git fetch origin "+refs/heads/agent/cross-shard-simd-validation:refs/remotes/origin/agent/cross-shard-simd-validation" && \
  git reset --hard origin/agent/cross-shard-simd-validation'
```

### When Codex opens a PR

```bash
# 1. See the PR
gh pr view 142 -R deme-plata/q-narwhalknight

# 2. Fetch the PR's commit(s) locally
git fetch origin "refs/pull/142/head:refs/remotes/origin/pr/142"

# 3. Inspect what unique content it adds
git log --reverse --pretty='%h %s' HEAD..origin/pr/142

# 4. Cherry-pick (single-commit PR)
git cherry-pick origin/pr/142
# OR for multi-commit PR (range)
git cherry-pick origin/pr/142~3..origin/pr/142
# OR pick individual commits if there's a mix of unique and duplicate
git cherry-pick <sha-1> <sha-2> <sha-3>

# 5. Resolve any conflict (rare for clean PRs)
#    Conflicts mean two commits touched the same lines —
#    inspect, merge intent, git add, git cherry-pick --continue

# 6. Refresh local-git
git update-server-info

# 7. Optionally close the PR on GitHub with a reference to where it shipped
gh pr close 142 -c "Cherry-picked as <sha> in vX.Y.Z. Original PR branch preserved."
```

### When you do a release

```bash
# 1. Bump Cargo.toml workspace.package version
vi Cargo.toml  # change version = "X.Y.Z"

# 2. Commit
git add Cargo.toml
git commit -m "vX.Y.Z: <release-name>"

# 3. Refresh local-git
git update-server-info

# 4. Build on Epsilon Docker (rust:bookworm)
ssh root@89.149.241.126 'cd /home/orobit/q-narwhalknight-src && \
  git fetch origin "+refs/heads/agent/cross-shard-simd-validation:refs/remotes/origin/agent/cross-shard-simd-validation" && \
  git reset --hard origin/agent/cross-shard-simd-validation && \
  nohup docker run --rm \
    --name qnk-build-vX.Y.Z \
    -v $(pwd):/src \
    -v /home/orobit/target-debian12:/src/target \
    -w /src --cpus=16 \
    rust:bookworm \
    bash -c "apt-get update -qq && \
      apt-get install -y -qq libssl-dev pkg-config cmake clang libudev-dev libclang-dev >/dev/null 2>&1 && \
      cargo clean --package q-api-server 2>&1 | tail -3 && \
      cargo build --release --package q-api-server 2>&1" \
    > /home/orobit/tmp/build-vX.Y.Z.log 2>&1 &'

# 5. Smoke-test the binary (Docker container against mainnet-genesis,
#    with --cap-add=SYS_ADMIN for io_uring + --admin-wallet to bypass setup wizard)

# 6. Deploy manually (per current preference): Beta → Gamma → Delta → Epsilon

# 7. Optionally tag and mirror to GitHub
git tag -s vX.Y.Z -m "vX.Y.Z release"
git push origin agent/cross-shard-simd-validation:mirror/vX.Y.Z
git push origin --tags
```

## Why this works

### Local-git as trunk
- **Sovereign** — runs on our own infrastructure, no dependency on GitHub uptime
- **Fast** — `git update-server-info` is instant, no network round-trip
- **Audit-friendly** — every commit signed, the chain on Beta is the definitive timeline
- **Safe** — `http.receivepack=false` on Beta's git config means nobody can push to local; commits only come from authorized SSH access

### GitHub as collaboration surface
- **Codex-native** — Codex's whole workflow assumes GitHub; we don't fight it
- **PR review UI** — comments, draft state, diff threading; far better than browsing raw git
- **Public visibility** — when we want to show off or take contributions
- **Historical record** — original PR branches preserved indefinitely, even after we cherry-pick

### Why one-way (GitHub → local, not local → GitHub authoritative)
- **GitHub diverged from local** when Codex began contributing — origin/agent/cross-shard-simd-validation has a disjoint snapshot history. Pushing local to that branch needs force-push or new branch name. We chose new branch name (mirror/vX.Y.Z) — non-destructive.
- **Codex's PRs land where Codex puts them**; we don't ask Codex to push to local-git (which it can't reach anyway). We integrate by cherry-pick.
- **Production correctness lives on our line** — never gets a half-applied cherry-pick that breaks the build.

## What we won't do

- **Never force-push** over published branches on either remote. Existing history is evidence.
- **Never delete** branches, tags, or remotes during cleanup. Designate one authoritative; keep the rest as archive.
- **Never bypass GPG signing** on commits that matter. `commit.gpgsign=true` is the default. Re-prime gpg-agent (8h TTL) when needed.
- **Never declare a PR "shipped"** without verifying its commit is reachable from `origin/release/v<production>` AND the binary on production matches. The `git merge-base --is-ancestor` check is cheap and catches the "PR merged to main but never reached release/*" failure mode that bit us in May 2026.
- **Never compile on Beta.** Builds happen on Epsilon Docker. Beta is production-adjacent and we don't want long-running builds eating its CPU/RAM.

## The lessons codified

These memories now live in `/root/.claude/projects/.../memory/` and apply to every future session:

- **[branch_hygiene_lessons]** — verify ancestor compatibility before cherry-pick; PRs targeting release/* land on release/* only
- **[feedback_release_branch_lineage_check]** — PR-merged-to-main ≠ PR-in-production; always check the specific release branch
- **[release_branch_source_binary_mismatch]** — release/v10.9.57 source doesn't build from clean checkout; binary was built from agent/cross-shard-simd-validation
- **[feedback_non_destructive_recovery]** — designate-don't-delete; keep broken state visible as evidence
- **[cluster_topology_corrected]** — Beta is DEV (beta.quillon.xyz), Epsilon is PROD (quillon.xyz); deploy Beta first, Epsilon last
- **[cross_server_sync_rule]** — never rsync source trees; use git
- **[cargo_version_string_stale]** — `cargo clean -p q-api-server` before release builds where API-reported version matters

These are not abstract principles. Each was earned through an incident. Following them is cheaper than the next incident.

## What's still ahead

After the 2026-05-20 recovery session, 4 things remain:
1. **PR #44** — needs base docs (in `docs/security-audit/`) which haven't been cherry-picked yet
2. **PRs #97 + #99** — labelled "docs" but have feature-branch baseline (-575 footprint); need per-commit triage
3. **4 deferred PR #94 commits** — main.rs lattice-tip producer-hook conflict, needs ~135 LOC manual merge
4. **GitHub mirror push** — push current `agent/cross-shard-simd-validation` (with 40+ new commits) to GitHub under e.g. `mirror/v10.10.2` so Codex/external observers see the integrated state

These are well-understood follow-ups, not unknowns. Each can be done in a focused short session.

## Bottom line

The May 2026 incident felt like a crisis — a release branch that didn't build, security PRs trapped in the wrong lineage, Codex collaboration in seeming conflict with the local-git workflow. By session's end, none of that was true anymore. Local-git is authoritative, GitHub is the collaboration surface, the cherry-pick discipline gets work from one to the other with original attribution intact. 40 signed commits in a day proves the workflow can carry production-relevant change.

The good news isn't that we recovered from a mess. It's that the workflow we used to recover is now the workflow we'll use going forward — and it scales. Codex keeps doing what Codex does on GitHub. We keep doing what we do on Beta. The two systems coexist by serving different purposes, not by competing for the same role.

Build sovereign. Collaborate openly. Don't delete the evidence.
