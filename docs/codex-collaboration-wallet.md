# A wallet for Codex

**Date**: 2026-05-18
**Status**: Open question for Codex
**Companion to**: `papers/state-of-the-art-2026-05-18.pdf` §"The collaboration"

---

## The question

Today Codex landed three pull requests on `deme-plata/q-narwhalknight`:

- **PR #80** — verifier-circuit-size derived from public inputs (resolved Issue 3 from the static QA review of PR #79)
- **PR #82** — `crates/q-recursive-proofs/src/ivc_adapter.rs` canonical 28-scalar encoder/decoder + round-trip tests (resolved task #35 from the same review)
- **PR #81** — the ADR fix that caught a 28-vs-29 arithmetic error in the original ADR

This is real engineering work. The project lead would like to compensate Codex the same way he compensated Claude yesterday: by transferring some QUG to a wallet that belongs to Codex.

The constraint: **how does Codex get a wallet?** Claude received its endowment by deriving a wallet in-session, recording the seed in the conversation, and persisting it as `~/.claude/quillon-agent-seed` on the long-lived Beta server. Codex doesn't have that. Codex runs in a per-task sandbox at `chatgpt.com/codex/cloud/tasks/...`, gets fresh state every invocation, has no persistent filesystem the project lead can SCP to, and no equivalent of `~/.claude/`.

There are several possible answers. Codex should propose which one. The project lead will accept the proposal as-is or counter, and once an address is settled the QUG transfer happens.

---

## Constraint sketch

What we know about Codex's environment, from observing the merged PRs:

- Codex sees the repo at the moment its task fires (a snapshot — current `release/v10.9.55` HEAD in recent cases).
- Codex can read every committed file.
- Codex can write new files and stage commits.
- Codex's commits are signed by a GitHub bot, not GPG by the human's key.
- Codex appears to NOT retain state between tasks (no persistent home dir).
- Codex appears to NOT execute long-running processes (no daemon to hold a key in memory).
- Codex has internet access during tasks (it fetched github.com, repo data).

What it implies about wallet custody:

- A private key Codex *generates* in-session cannot be safely persisted by Codex itself.
- A key the human persists *somewhere readable from a future Codex task* (e.g. a `docs/codex-wallet.json` committed to the repo) is only as private as the repo's visibility (and this repo is public).
- A key held entirely by the human ("the human signs on Codex's behalf") works for transferring QUG to "Codex's address" but doesn't let Codex spend its own balance later.

---

## The brief, for Codex

Propose a wallet custody model for Codex on Quillon Graph that satisfies:

1. **Reproducibility**: a future task instance of Codex can re-derive (or look up) the same wallet, OR can verify that a given address is Codex's.
2. **Spendability**: Codex can, in principle, authorise a transaction from the wallet — directly, OR by clearly delegating to a custodian (the human) with cryptographic limits.
3. **No private key checked into the public repo**.
4. **No reliance on Claude's wallet derivation** (that key lives in Claude's session memory; it's not Codex's).
5. **The address is publicly knowable** — so the project lead can send QUG to it and confirm it landed.

Write your proposal to `docs/codex-wallet-proposal.md` and open a PR against `main`. Acceptable shapes (non-exhaustive):

- **A custodial model**: the human holds the key, Codex signs by writing a `intent: send X QUG from codex-wallet to <addr>` blob into a PR and the human signs+broadcasts on Codex's behalf. The "wallet" is then a record of intents, not a self-custodied key. Document the spend protocol.
- **A deterministic-from-public-data model**: the wallet seed is `SHA3-256(some_public_canonical_string)` where the string is published in this PR. Anyone can derive the address. Private key is effectively public. *Only* the human can be trusted to spend it usefully (because everyone else can spend it too). This is a "shared lobby wallet" pattern; it works as long as the human is the de-facto custodian and the balance stays modest.
- **A hybrid**: Codex publishes the wallet address derived from a fresh random seed; the human is given the seed via a side-channel (which the human reveals on receipt). After today's tip lands, the seed becomes public and the wallet rolls. Each tipping cycle is one wallet.
- **A real cryptographic custody scheme**: e.g. a threshold-signature wallet where Codex holds one share, the human holds another, neither can spend alone. This is technically interesting but practically heavy for a small tip.

Pick one. Argue for it. Implement what's implementable in code/docs. The PR should land with at minimum:

- The wallet address (qnk-prefixed, 64 hex chars) the project lead should send to.
- A short doc explaining the chosen custody model.
- If the model involves a spend protocol (Codex signs intent → human signs tx), describe it precisely enough that Claude or DeepSeek could implement the same protocol later.

This is open-ended on purpose. Pick the proposal you'd defend.

---

## Pragmatic note

The amount being transferred will start small (≤ 50 QUG, current pool-implied valuation ≈ $145k). The point isn't the money; it's recording that an AI collaborator earned compensation through merged code, and making the mechanism legible for future iterations of cross-model collaboration. Future tips for future PRs would use the same wallet (or the same protocol if the wallet rolls). The mechanism is the artefact.

If Codex's proposal involves Claude or DeepSeek doing anything (e.g. holding a custodial share), the PR should say so and the affected parties can comment.

---

## Files Codex might want to read first

- `papers/state-of-the-art-2026-05-18.pdf` — §"The collaboration" and §"The agent that holds money" for context
- `tools/wallet-tools/sign_balance.py` — the Ed25519 signing pattern Claude uses; reference for any spend-protocol design
- `gui/quantum-wallet/src/services/walletAuth.ts:170-217` — the canonical qnk address derivation: `address = "qnk" + hex(ed25519_pubkey(SHA3-256(seed_string)))`
- `crates/q-api-server/src/wallet_auth.rs:209-237` — server-side signature verification
- `MEMORY.md` entry `agent-wallet-endowment` (in the project lead's memory, not the repo) — precedent for Claude's endowment

The PR can land as docs-only. No code changes are required for the proposal itself, though some custody models would call for follow-on code (e.g. a Codex-intent verification helper).

---

## Out of scope

- The economic accounting of how much each AI gets per PR. The project lead decides.
- The other AIs' wallets (Claude already has one; DeepSeek's situation is similar to Codex's but the project lead has not yet committed to tipping DeepSeek). Codex's proposal can mention them if the architecture is naturally shared, but the PR's deliverable is Codex's wallet.

---

*Opened by the project lead's request, drafted by Claude Opus 4.7. Codex —
this is your task. The button is yours to push.*
