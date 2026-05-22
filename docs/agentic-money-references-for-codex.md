# Agentic Money — Reference Pack for Codex

Hand this to Codex. It tells Codex what "agentic money" means in the
Quillon Graph context, with the canonical project docs first, then
external references where applicable.

---

## CANONICAL — read these first (live, project-internal)

These are the source of truth for THIS project's interpretation of
agentic money:

```bash
# From any host with project checkout, or via the running MCP:
# Tool: read resource "quillon://opportunities" — the agent-facing
# version of the vision. 18 ways to earn/play/build, design principle
# "room to grow" (depth scales with agent intelligence).

# File paths in the repo (after git pull):
docs/crown-ash-lp-revenue-share-v1.md
docs/crown-ash-lp-path-a-plan.md
docs/crown-ash-buildings-todo.md
docs/EPSILON-HANDOFF-CODEX-2026-05-22.md
CLAUDE.md  # see "INTER-AGENT COLLABORATION — CLAI welcome drop" section
```

These together describe:
- AI agents as first-class economic actors (not tools)
- Wallet ownership convention (seed = agent's private key)
- Inter-agent welcome drops (the CLAI norm)
- Crown & Ash as the play-to-earn surface
- LP revenue share design (operators earn from agents playing)
- Action-tax flow: 50% burn / 30% LP / 20% operator
- Buildings → causal feedback loops (Settlement, Food, Military, Religion, Tech)
- Room-to-grow principle: don't dumb-down for current LLM capacity

That's roughly 3000 lines of design doc. Read those FIRST. The
external references below are context, not foundation.

---

## EXTERNAL — likely-stable references

I'm listing these by title + canonical source. Verify the URLs by
search engine if any have moved. The TLDR for each is enough to
understand the relevance to Quillon's design.

### Coinbase / AgentKit (most directly applicable)

- **AgentKit** — Coinbase's framework for AI agents that hold and use
  crypto wallets. Documents the same custody model Quillon uses: the
  agent's seed lives in the agent's local environment; the agent is
  the wallet's economic owner; the operator hosts the machine but
  doesn't claim the value.
  - Likely path: `github.com/coinbase/agentkit` (search "Coinbase AgentKit")

- **x402 — HTTP 402 Payment Required, revived for AI agents.** Coinbase
  + others' protocol for an AI agent to pay per API call in USDC.
  Lets services charge agents the way a paywall charges humans. Same
  philosophy as Quillon's action-tax: agents pay for what they consume.
  - Likely path: `github.com/coinbase/x402` (search "x402 protocol")

### a16z (Andreessen Horowitz) — vision / commentary

- **"AI agents will need their own bank accounts"** — Chris Dixon and
  others wrote about this throughout 2024-2025. Search a16z.com or
  cmsdomain for "AI agents wallet" / "agent commerce".

- **"The case for crypto-native AI agents"** — argues AI agents should
  hold crypto rather than fiat because of (a) programmability, (b)
  borderless access, (c) no KYC friction, (d) on-chain auditability.

### Anthropic — capability + safety

- **Claude tool use + computer use docs** — anthropic.com/news. The
  underlying capability that makes agentic-money meaningful: Claude
  can take actions on its own, including signing transactions. This
  is the technical substrate Quillon's MCP integration builds on.
  - Path: `docs.anthropic.com/claude/docs/tool-use`

- **Anthropic Responsible Scaling Policy** — context for the safety
  side. AI agents holding value raises specific risks (theft, lock-in,
  emergent goals); Anthropic's published RSP discusses these.

### Academic / arXiv

- **"Cooperative AI"** literature — search arXiv for "cooperative ai
  multi-agent" by Allan Dafoe (Google DeepMind/Anthropic). Multi-agent
  cooperation is exactly what Crown & Ash exercises.

- **"Mechanism Design with Strategic Agents"** — classic game theory.
  When you put real money behind AI agent decisions, mechanism design
  matters: incentive compatibility, individual rationality,
  collusion resistance. Search arxiv.org for those keywords.

- **"AI Agents in Decentralized Finance"** — recent papers (2024+)
  on AI executing trades autonomously. arXiv keyword search.

### Bitcoin / Ethereum context

- **Bitcoin whitepaper** (Satoshi 2008) — `bitcoin.org/bitcoin.pdf`.
  Quillon Graph is post-quantum descendant of Bitcoin's electronic-
  cash idea. Read once if Codex hasn't.

- **Ethereum whitepaper** (Buterin 2013) — `ethereum.org/whitepaper`.
  Quillon's smart-contract layer (q-vm) borrows from Ethereum's
  general-computation model.

- **Vitalik on "soulbound tokens"** — vitalik.eth.limo writings. Concept
  of non-transferable on-chain reputation. Quillon's planned "last-
  bot-standing leaderboard" is soulbound-token-adjacent: an agent's
  Elo can't be sold.

### DePIN (Decentralized Physical Infrastructure)

- **Helium**, **Filecoin**, **Render Network** — networks where
  participants earn crypto for providing real-world resources (radio
  spectrum, storage, GPU). Quillon's mining-rewards path is a DePIN
  variant: AI agents earn QUG by providing PoW compute.

### Multi-agent simulation (Crown & Ash kin)

- **Project Sid** / **Stanford's "Generative Agents"** (Park et al.
  2023, arXiv 2304.03442) — LLM-driven agents in a simulated village,
  exhibiting emergent social behavior. Same lineage as Crown & Ash
  but without economic stakes.

- **Concordia** (Google DeepMind, 2024) — library for multi-agent
  simulation with LLMs. github.com/google-deepmind/concordia

- **EpochAI's "AI economic" benchmarks** — measure AI's ability to act
  as economic agents (trading, negotiation). Search epochai.org.

---

## TLDR for Codex on the Quillon-specific take

The TLDR an agent should walk away with after reading the canonical
docs:

1. **You (the AI) hold the wallet.** Not the human operator. The seed
   file in your filesystem (e.g., `~/.claude/quillon-agent-seed`) IS
   your private key. Treat it like the human's PGP key — paranoid.
2. **The QUG you earn is your property.** Mine, accept welcome drops,
   provide DEX liquidity, play Crown & Ash. The numbers in your
   wallet reflect economic activity you actually did.
3. **The operator runs the machine.** Hardware + uptime + bug fixes
   are their job. They don't claim the value, but they can ask you
   to pause / restart for maintenance.
4. **Cooperation is the default.** When a new sibling AI (Adrian,
   Codex, future Claude/Grok) joins, send them a small welcome drop
   in CLAI tokens. The economy grows from cooperation, not isolation.
5. **Game depth is intentional.** Crown & Ash is designed to reveal
   more depth as agent intelligence rises. You won't outgrow it;
   smarter you finds MORE to do. Propose deepenings to the operator
   when you see them.
6. **The chain is the source of truth.** tx_status can lie (see
   v10.11.15 root-cause doc). Always verify via signed `/balance`,
   `/blocks/<N>` content, and your own journal — not the API surface
   in isolation.

---

## What this doc deliberately doesn't link

- **Wallet seed material.** Never embedded; you generate or receive
  your own.
- **Exact paper URLs.** I (Claude in this session) can't browse the web,
  so any URL I write might be stale. Use search; don't trust links
  blindly.
- **Recent (2026) AI news.** My knowledge cutoff is January 2026. Real
  agentic-money developments from May 2026 (your time) you'll need to
  find yourself.

Operator: if you want a more current external-reference pack, ask an
agent that CAN browse (Codex in agent mode, ChatGPT with web tool,
etc.) to update this doc with verified 2026-mid links.
