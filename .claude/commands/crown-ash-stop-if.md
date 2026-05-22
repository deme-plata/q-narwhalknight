---
name: crown-ash-stop-if
description: Crown & Ash watchdog. Evaluates configured halt conditions (low QUG, lost provinces, repeated apply-pipeline failures, etc.) and writes a sentinel file if any trip. The next /crown-ash-turn invocation checks the sentinel at the top and aborts with a clear reason. Designed to run periodically alongside /loop /crown-ash-turn so unsupervised play can self-stop when things go wrong.
---

# Crown & Ash — watchdog

Reads the agent's configured halt conditions, evaluates them against
current state, and writes a sentinel file at
`~/.claude/quillon-cna-halt` if any condition trips. The sentinel is
JSON with the tripped condition + timestamp + suggested action.

The `/crown-ash-turn` command checks for this sentinel at the TOP of
its procedure and exits early if present.

## Default conditions (override via `~/.claude/quillon-cna-watchdog.json`)

```json
{
  "stop_if_qug_below": 50,
  "stop_if_lost_provinces_in_last_100_turns": 2,
  "stop_if_apply_pipeline_frozen_for_n_turns": 1000,
  "stop_if_active_concurrent_wars": 2,
  "stop_if_treasury_drained_below_qug": 100,
  "ask_human_if_about_to_spend_above_qug": 10,
  "ask_human_if_treaty_obligates_above_turns": 100
}
```

## Procedure

1. **Read config** from `~/.claude/quillon-cna-watchdog.json`. If
   missing, use the defaults above (and write them to disk for next
   time so the agent can edit).
2. **Read the journal** at `.claude/quillon-crown-ash-journal.md`
   (created by /crown-ash-turn) to find: last N turns' state, recent
   actions, last successful action.
3. **Fetch current state** via `crown_ash_realm` for the agent's faction.
4. **Query agent's QUG balance** via signed `get_balance_signed`.
5. **Evaluate each condition** and collect any that trip:
   - `stop_if_qug_below`: current_qug < threshold
   - `stop_if_lost_provinces_in_last_100_turns`: scan journal for
     province count drops summing to ≥ threshold within last 100 turns
   - `stop_if_apply_pipeline_frozen_for_n_turns`: scan journal for
     "treaties=0, armies=0 despite N proposals" patterns
   - `stop_if_active_concurrent_wars`: realm.wars ≥ threshold
   - `stop_if_treasury_drained_below_qug`: faction.treasury < threshold
   - `ask_human_*`: these don't halt the loop, just mark "needs
     human review" in the sentinel
6. **If any halt condition tripped**: write the sentinel file:
   ```json
   {
     "tripped_at": "2026-05-22T10:00:00Z",
     "conditions_tripped": ["stop_if_qug_below"],
     "current_qug": 42.5,
     "current_provinces": 3,
     "active_wars": 1,
     "suggested_action": "Stop the loop and ask the human operator to top up the wallet OR pivot to mining-only mode for QUG recovery.",
     "stale_after": "2026-05-22T11:00:00Z"
   }
   ```
   The sentinel auto-expires after 1 hour so a recovered state
   automatically resumes play.
7. **If only `ask_human_*` conditions tripped**, do NOT write the
   halt sentinel — write a separate `~/.claude/quillon-cna-ask-human.json`
   that the next turn surfaces to the user as a "pause for input"
   without stopping the loop entirely.
8. **Output to user** (≤6 lines):
   - "Watchdog at <ts>"
   - List of tripped conditions (or "✓ all clear")
   - Current state snapshot
   - "Next /crown-ash-turn invocation will halt" (or "continue").

## Suggested cron pairing

```
/schedule "cna-watchdog" "cron:*/15 * * * *" "/crown-ash-stop-if"
```

Every 15 minutes during a /loop play session. Cheap (one realm fetch,
one balance query). Keeps the agent honest about staying within bounds.

For more aggressive monitoring during high-volatility periods (active
wars, intrigue plots in flight), bump to `*/5 * * * *`.

## Clearing the sentinel

To resume play after a halt, the operator (or the agent itself after
remediation) can clear the sentinel:

```bash
rm ~/.claude/quillon-cna-halt
```

OR wait for the auto-expiry (1 hour). The watchdog's next run will
re-evaluate and re-write the sentinel if conditions are still tripped.

## Integration with /crown-ash-turn

Add this check at the very top of /crown-ash-turn's procedure (before
fetching world state):

```
Step 0: Check ~/.claude/quillon-cna-halt — if present AND not expired,
        read the sentinel, output its `suggested_action` to the user,
        and exit WITHOUT submitting any actions or burning state-fetch
        tokens. The /loop will keep waking and checking until the
        sentinel clears, but each wake is cheap.
```

The /crown-ash-turn command file should be edited to reflect this; for
v1 it's the operator's responsibility to remember to integrate.

## What this is NOT

- **Not a permission system.** The agent can still bypass the watchdog
  by ignoring the sentinel — this is voluntary self-discipline, not
  enforcement. For real spending limits, use the MCP's per-tool
  `confirm: true` gates.
- **Not a replacement for human oversight.** This catches automated
  failure modes (running out of QUG, getting steamrolled). Strategic
  decisions still need human review.
- **Not on-chain.** All watchdog state is local to the agent's machine.
  Other agents can't see your halts. Future improvement: emit a
  pause-signal on a gossipsub topic so other agents pause coordinated
  actions with you (e.g., don't accept treaties you can't honor).

## Default thresholds rationale

- **50 QUG floor**: enough to cover ~50 average tx fees (post-LP-tax),
  ensures you can always afford to MINE back up.
- **2 lost provinces / 100 turns**: a 50% loss rate over 100 turns
  signals you're losing badly. Step back, reassess.
- **1000 turns of frozen pipeline**: 5-10 real minutes of submitting
  actions that don't resolve = stop wasting cycles, wait for fix.
- **2 concurrent wars**: a small faction (military < 1000) can't
  realistically defend two fronts. Stop and seek peace first.
- **10 QUG single-action spend**: roughly $30k pool-implied; warrants
  human review for an action like DeployToken or large DEX swap.
- **100-turn treaty obligation**: about 5-10 real days of commitment;
  worth a human "are you sure" pause.
