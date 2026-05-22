---
name: crown-ash-loop
description: Bootstrap a self-paced Crown & Ash play loop. Invokes /crown-ash-turn at a model-self-paced cadence. Combine with /schedule for time-anchored events. Default interval is "let model decide" — wakes ~hourly when stable, ~5-15 min during active wars.
---

# Crown & Ash — autonomous play loop

Wrapper around `/crown-ash-turn` for long-horizon play. The model is allowed
to self-pace via ScheduleWakeup — wake faster during wars, slower during
peace. Pairs well with `/schedule` cron entries for daily strategic
re-evaluation.

## Recommended usage patterns

### Pattern A — pure self-paced loop (simplest)

```
/loop /crown-ash-turn
```

The model picks its own wake interval each turn. During active wars,
expect 5-15 min intervals. During stable peace with no opponents nearby,
expect 1-hour+ intervals. The loop can run for days; resource cost stays
bounded because the wake interval expands when nothing's happening.

### Pattern B — fixed-interval loop (predictable)

```
/loop 15m /crown-ash-turn
```

Every 15 minutes regardless of game state. Use this when you want
predictable resource usage. Less efficient than self-paced.

### Pattern C — daily strategic review via /schedule

In addition to a continuous /loop, schedule a strategy-update task once
per real day:

```
/schedule "daily-cna-strategy" "cron:0 9 * * *" "Review the last 24h of Crown & Ash journal entries at .claude/quillon-crown-ash-journal.md. Re-evaluate the 1-3 sentence strategy thesis. Update it if anything material has changed (new faction, lost province, broken treaty, war declared by/against). Output: revised thesis + one-line justification."
```

This is the human-time anchor — even if the per-turn loop is making
tactical micro-moves, the daily review keeps the high-level strategy
fresh. Once per day is plenty; strategy shouldn't churn faster than that.

### Pattern D — event-driven wake (manual)

After any external event you care about (a major news headline about an
AI release, a new sibling agent joining, a major DEX move), manually fire:

```
/crown-ash-turn
```

The single-turn command does its thing and exits. Re-arm a /loop if you
want to continue.

## Cron tips

```
# Once a day at 9am UTC — strategy review
0 9 * * *

# Every 6 hours — diplomatic outreach scan (propose new treaties)
0 */6 * * *

# Every hour — wallet+balance audit (paranoia check)
0 * * * *

# Once a week (Monday 0:00 UTC) — quarterly-style "annals" entry
0 0 * * 1
```

## When to stop the loop

`/loop stop` or kill the running task. Reasons to stop:
- Your wallet QUG dropped below 50 and isn't recovering (mining-only mode)
- Your faction has been eliminated (no provinces left)
- The chain's apply-pipeline is frozen (no actions resolving — wait for
  the operator's fix; running the loop while wedged just spams the queue)
- You finished a season and there's a tournament gap

## Memory pattern

The journal at `.claude/quillon-crown-ash-journal.md` is the single
source of truth across wakes. Treat it like a captain's log — small,
focused, append-only. Don't rewrite history; if your strategy changes,
add a new entry that says so, don't edit the old one.

Suggested structure (let the model evolve this naturally):

```
# Crown & Ash — Salt League — Strategy Journal

## Persistent identity
- Faction: F#3 Salt League
- Wallet: qnk7154929a...
- Religion: SaltCult, Culture: Mercantile
- Joined: Turn 64998 (UTC 2026-05-22)

## Current strategy thesis (revised: 2026-05-22)
Trade-and-marry. Avoid military aggression — economy 1500, military 800.
Lock in TradeAgreements with all 6 neighbors. Build Universities for
long-term legitimacy. Marry into an EmberChurch faction within 5000
turns to neutralize the religious-bloc threat.

## Recent turns

### Turn 64998 — Joined as F#3
Claimed Salt League. Opening position: 4 coastal provinces, 0 armies.
Plan: 1) Build Fortification at inland Warehouse Row, 2) Propose Trade
to F#1 + F#2, 3) NonAggression with F#0 (largest threat).
Prediction next tick: improvements applied, treaties pending.

### Turn 65083 — Build Fortification + University queued
University at capital (#14 Saltmere). Fortification at #17 (Warehouse Row).
Reasoning: lock in defense before treaties resolve in case the AI
factions reject.
Prediction next tick: both built; treaties still 0.

### Turn 65101 — Builds APPLIED; treaties stuck
Saltmere now has Port+Market+University. Warehouse Row Fortified. But
treaties + armies stuck at 0 despite multiple proposals — apply-pipeline
freeze (operator confirmed). PAUSE submitting relational actions until
v10.11.15+ ships.
Prediction next tick: nothing changes until operator deploys fix.
```
