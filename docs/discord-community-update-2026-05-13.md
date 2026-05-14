# Discord Community Update — 2026-05-13

Draft to post in #announcements or pin in #general for the people who've been following since October.

Tone: honest about what's happening under the hood, not defensive, not hyping. The Discord audience has earned a real read of where the project is.

---

## Where Quillon Actually Is Right Now

A few of you have asked recently whether things are slower than expected. Honest answer: yes, this past week was rough on the engineering side — and that's why the network is in better shape today than it was a week ago. The story is worth sharing because the people in this server have been here since version 0.09 in October, and you deserve the real picture, not the highlight reel.

### What got fixed this week

The chain has been at 17.9M blocks since well before Monday. What we were working on isn't producing more blocks — it's making the system that a *new* node uses to join the network actually work the way we always claimed it did. Three problems came out:

**1.** Fresh nodes were silently dropping transfer transactions during initial sync (a "fast but wrong" shortcut from the chain's earlier days). At BAL-001 activation in three weeks, those wrong-balance nodes would fork from the network. Fixed in v10.9.7.

**2.** The block-storage pointer was lying about what data the node actually had. New nodes claimed to be at tip while missing most of the chain underneath. Other nodes were asking them for blocks they didn't have. Fixed in v10.9.8 + v10.9.13.

**3.** The pre-checkpoint historical backfill (blocks 1 through 16.5M) was effectively impossible — it ran at ~12 blocks/second meaning a new archive node would take 16 days to be complete. We rearchitected the bootstrap sequence (Phase 1 forward-sync to tip in minutes; Phase 2 archive backfill in the background) and proved it works at ~2,000 bps in soak testing. v10.9.11 → v10.9.13.

### The proof

Last night a Docker-isolated test container running v10.9.13 went from zero blocks to having the same wallet state as the production network in 25 minutes — including a freshly-mined test wallet that was credited 22+ times on the live chain. The same address. The same balance. On a node that started from a balance checkpoint and re-derived everything else from peers. No special trust in any single node. That's what people mean when they say "decentralization works" and it's the first time the chain has demonstrated it end-to-end in a clean test rig.

### What's coming

**Now → 3 weeks:** BAL-001 activates at block 20,000,000. From that point forward, every block carries a state-root hash; any node producing wrong balances gets rejected at the protocol level. The fixes above were specifically to make sure no honest node accidentally produces wrong balances at the activation cutover.

**Q3 2026:** HiBT exchange listing (the $15K listing fee is the only thing left on that path; we're community-funded so it's progressive, not VC-fueled).

**Q4 2026:** ASIC partnership with Dragon Ball Enterprise. FPGA validation on their Kintex-7 board has been clean; ASIC tape-out decision after their synthesis report lands.

**Q1 2027 (target):** First production IVC recursive SNARK proof. Replaces "trust the checkpoint" with "verify cryptographically from genesis." That work is staged in `crates/q-ivc` and we have external review going. This is the real 256-year-design story; everything else is bridging us to that.

### The honest framing

The thing some of you have wondered — "did Demetri announce too early back in October?" — is fair. Quillon was at v0.09 with no blocks when this server opened. There are people who walked away then because there wasn't a polished product. There are 293 active miners now who would not exist if we'd waited for v1.0 to introduce the project.

The people who stayed get to see what they're actually getting: a live network, with bugs that get found and fixed publicly, that runs a real consensus protocol and produces real blocks every second. The list of fixes above isn't an apology — it's the receipt for what "live, build-in-public, post-quantum L1" actually looks like. Most projects at this market cap quietly fix bugs of this severity in their staging environment and never tell anyone. We don't have a staging environment for the mainnet itself, so you've been our staging environment. That's the deal you signed up for in October and we've kept that bargain.

The audience this server has earned the upgrade. Quillon isn't "early experiment" anymore. It's a live network at 17.9M blocks with architecture decisions starting to land at the level we always wanted them. Thanks for sticking around.

— Demetri / Quillon Team

---

## Posting notes

- Drop in #announcements pinned
- Avoid hashtags or emoji header lines — the tone is "real talk to engineers who care"
- If anyone asks for the technical-review docs cited (sync architecture, Phase 2 OOM, IVC acceleration), they're all in `docs/` and we can share specific filenames
- Don't engage with FUD about the bugs being "proof of mismanagement" — the framing in the message already addresses it
