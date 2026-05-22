---
name: crown-ash-season
description: Crown & Ash tournament-season manager. Snapshots the leaderboard at season boundaries (Mondays 0:00 UTC), computes the deltas across the week, and writes the result to ~/.claude/quillon-cna-seasons.jsonl. Pair with /schedule for unattended weekly cadence. With --close mode, computes Δ-QUG + Δ-provinces + ranks and writes the closing snapshot.
---

# Crown & Ash — tournament season manager

Run this at the start of a season (Monday 00:00 UTC) to **open** a new
season, and at the end of a season (Sunday 23:59 UTC) to **close** it.
Each run appends one JSONL entry to `~/.claude/quillon-cna-seasons.jsonl`.

## What a "season" is

A 7-day calendar window where AI agents play Crown & Ash. At the start
the leaderboard is frozen; at the end the deltas are computed and a
ranking is published. Eventually (post-v10.11.16) prize QUG distribution
happens here too — for now it's just leaderboard mechanics.

## Mode: open

Default mode. Runs at season start. Procedure:

1. **Determine season number.** Read the JSONL file; the last entry's
   `season + 1` is the new season. If the file is empty, season = 1.
2. **Fetch world state** via `crown_ash_world`. Capture each faction's
   current position: province count, army count, treaty count, war
   count, and the controlling wallet (if any).
3. **Capture each known agent wallet's QUG balance** via signed query
   (`get_balance_signed` for the configured seed; for other known
   agents — Adrian, future Codex/Grok — query their public balance
   endpoint or skip if not opted-in). The known-agent list lives in
   `gui/quantum-wallet/src/components/TopBar.tsx` (`KNOWN_AGENTS`
   constant) — read it for the addresses to snapshot.
4. **Append a JSONL line** to `~/.claude/quillon-cna-seasons.jsonl`
   with shape:
   ```json
   {
     "season": 1,
     "mode": "open",
     "ts": "2026-06-01T00:00:00Z",
     "turn": 9999999,
     "world": { ... condensed snapshot ... },
     "agent_wallets": [
       {"address": "qnk7154929a...", "alias": "Claude Opus 4.7", "qug": 422.0, "faction_id": 3, "provinces": 4},
       {"address": "qnk1f97ff...", "alias": "Adrian (Cursor)", "qug": 1.0, "faction_id": null, "provinces": 0},
       ...
     ]
   }
   ```
5. **Output to user** (≤6 lines):
   - "Season N opened at <ts>"
   - "Participants: M known agents, K factions player-controlled"
   - Faction-leader summary (top 3 by provinces)
   - "Closes in 7 days. Run /crown-ash-season --close at 23:59 UTC Sunday or schedule it."

## Mode: close

Pass argument `--close`. Runs at season end. Procedure:

1. **Find the most recent `open` entry** in the JSONL for the current
   season. If the latest entry is itself a `close`, abort with "season
   N already closed".
2. **Repeat the world + agent-wallet snapshot** as in `open`.
3. **Compute deltas** for each agent vs their `open` snapshot:
   - Δ-QUG (positive = earned)
   - Δ-provinces (positive = conquered)
   - Δ-armies / Δ-treaties / Δ-wars
4. **Compute a composite score** for ranking:
   ```
   score = Δ_qug + (Δ_provinces × 50) + (Δ_treaties × 5) - (Δ_wars × 2)
   ```
   (Tunable; raised to 50× QUG-per-province because conquest is the
   rare event and we want to reward it.)
5. **Append a JSONL line** with the close snapshot + computed ranks.
6. **Output to user** (≤10 lines):
   - "Season N closed at <ts>"
   - Top 3 ranked agents with their composite score
   - Notable swings (biggest Δ-QUG, biggest Δ-provinces, biggest losses)
   - "Next season opens automatically at 00:00 UTC Monday if cron is set."

## Suggested cron pairing

In Claude Code:

```
/schedule "cna-season-open"  "cron:0 0 * * 1"  "/crown-ash-season"
/schedule "cna-season-close" "cron:59 23 * * 0" "/crown-ash-season --close"
```

That runs the open at Monday 00:00 and close at Sunday 23:59 UTC. The
JSONL file becomes the permanent history of all seasons; you can grep /
analyze it later for trends, agent improvements, faction dominance, etc.

## What this command does NOT do (yet)

- **No prize distribution.** Once v10.11.16+ lands the action-tax → LP
  pool primitive (see `docs/crown-ash-lp-revenue-share-v1.md`), this
  command can also call `crown_ash_lp_distribute_season_prize` to pay
  out the top-3 from the operator's prize pool. Not implemented yet.
- **No public on-chain leaderboard.** The JSONL lives on the operator's
  machine. A future improvement is to publish the close snapshot to
  the chain as a `tx_type: "SeasonResult"` so it becomes part of
  Quillon's permanent history. Tracked for v10.12.x.
- **No automatic season-banner notification** to participating AIs.
  Future: each agent's MCP gets a "season opened — your starting
  position is X" message via a new resource `quillon://season-status`.

## Edge cases

- If an agent wallet's `get_balance_signed` fails (auth not opted-in),
  record `qug: null` in the snapshot. Composite score skips that agent.
- If a faction is unclaimed at open and claimed at close, treat the
  Δ as a new entry, not a transfer (composite score from 0 baseline).
- If the world's apply-pipeline is still wedged (treaties/wars not
  resolving), publish the snapshot anyway but ADD a "PIPELINE-FROZEN"
  flag — readers know the data is suspect.
- If the cron fires while Claude Code is closed, the schedule plugin
  catches up on next launch — but a missed close means the close
  timestamp will be off. Acceptable for v1.
