# Draft: metzdowd Cryptography Mailing List — April 2026 Update

**Subject:** Re: [Cryptography] Quillon Graph: A private, post-quantum electronic cash system — now with 50% more engineering chaos
**From:** Viktor S. Kristensen
**To:** cryptography@metzdowd.com

---

## Operational Update: Days 50+ on Mainnet

*Dear fellow key-nerds,*

Following jrzx's point about incremental quantum progress providing adaptation time — agreed in principle, disagreed on complacency. We chose to build the migration infrastructure *now*, while the cost is engineering time rather than "oh-shit-we-lost-everyone's-money". Here's what happened since the March update, including the bits we broke.

---

### 1. A Statistical Mechanics Framework for Consensus Health
*(or: How I Learned to Stop Worrying and Love the Hamiltonian)*

We published "The Theoretical Minimum for Blockchain Consensus" (v4, April 2026). The headline: the consensus ordering is the ground state of a Hamiltonian:

    H_DAG(σ) = H_parent(σ) + H_anticone(σ) + H_blue(σ) + H_VDF(σ) + H_commit(σ)

We prove (under assumptions you can find in [Sompolinsky et al. 2022](https://eprint.iacr.org/2022/1494.pdf)) that PHANTOM's output minimises this Hamiltonian — the "Ground State Theorem". The rest (temperature, phase transitions, entropy, free energy) is just textbook stat-mech applied to a DAG.

**Why we added this to a *cryptography* mailing list update:**
Because our K-gauge (a composite network stress metric) was blind to Sybil partition attacks and shallow chain tips. A node seeing only 2 adversary-controlled peers would report K ≈ 0 ("healthy") while its view of the network was completely fake. That's not a cryptographic failure — it's an *observability* failure. We fixed it with two information-theoretic corrections inspired by recent work in quantum gravity and computational universe theory:

- **Observer Coverage Factor** Ω_node = 1 − exp(−n_peers/n_total)
  Motivated by Harlow, Usatyuk & Zhao's observation that an observer in a closed universe sees an effective quantum mechanics whose Hilbert space dimension depends on the observer's entropy [[arXiv:2501.02359](https://arxiv.org/abs/2501.02359), [JHEP 02(2026)108](https://link.springer.com/article/10.1007/JHEP02(2026)108)].
  *Translation:* If your node is talking to only two friends, don't trust its health reading. Obvious in retrospect. Now formalised.

- **Block Commitment Depth** Λ_commit = 1 − exp(−d_commit/(κ·τ_confirm))
  Motivated by Lloyd's insight that classicality emerges from irreversible computation [[Phys. Rev. Lett. 88, 237901, 2002](https://arxiv.org/abs/quant-ph/9908043)].
  *Translation:* A block becomes "classical" (irreversible) when enough later blocks sit on top of it. Again, obvious. But we put equations on it so it counts as physics.

The enhanced gauge K_enhanced = K_base / Λ_commit · (1 + (1−Ω)·w_obs) correctly detects scenarios invisible to the base gauge. At steady state, K_enhanced ≈ K_base. After a restart or under partition, K_enhanced spikes — correctly flagging reduced trust.

**Data honesty note:** The Hamiltonian decomposition is a proven theorem. The K-gauge is an engineering diagnostic. The observer and commitment corrections are *structural analogies*, not physics derivations. We label each metric explicitly: "measured", "protocol constant", or "model prediction". The full paper (with all the Greek letters) is at https://quillon.xyz/downloads/theoretical-physics-consensus-whitepaper.pdf.

*Footnote for the list:* Yes, we are aware that invoking quantum gravity to tune a network monitor is like using a sledgehammer to crack a nut. But the nut was cracked, and the sledgehammer now has a peer-reviewed arXiv citation. We call that a win.

**Pre-emptive honesty, before someone asks:**

The Hamiltonian doesn't add cryptographic rigour beyond what Sompolinsky et al. already proved. The Ground State Theorem shows the formulation is *equivalent* to PHANTOM's output — not *stronger*. It's a **diagnostic decomposition**, not a new consensus algorithm. The value: "H_anticone is rising because propagation delay increased" is actionable; "consensus is degraded" is not. Each component (H_parent, H_anticone, H_blue, H_VDF, H_commit) is independently monitorable on the live dashboard. If someone cites our paper as proof that blockchain consensus obeys thermodynamics, that's a misreading. It obeys combinatorial graph theory. We just dressed it up.

The observer coverage factor Ω uses `n_total = 50` as a **configured estimate** of network size (labeled "CONFIGURED ESTIMATE" on the dashboard, not "MEASURED" — because it isn't). We don't have a DHT crawl estimator yet. A gossip-based epidemic averaging approach (each node tracks `max(peer_count)` seen from peers in the last hour) is on the v10.4 roadmap. Until then, Ω is useful for detecting *relative changes* in peer count (partition detection) but the absolute value depends on a guess. Note: an adversary who floods connections to force peer drops will inflate K_enhanced — a false alarm, not a partition. Operators can distinguish this because the other K components (sync divergence, block rate) remain stable during a pure connection flood.

The commitment depth Λ = 1 − exp(−d/(κ·τ)) uses an exponential because we model block arrivals as a Poisson process, not because Lloyd's irreversible computation framework directly produces this formula. Lloyd motivated the *concept* (classicality from irreversibility); the *math* is a standard survival function. Earlier drafts were sloppier about this attribution.

---

### 2. A Cryptography Dashboard with Honest Labeling
*(No marketing claims, just shame)*

We deployed a real-time cryptographic security posture dashboard at https://quillon.xyz (Explorer page). Every metric is tagged with its provenance:

| Metric | Value | Label |
|--------|-------|-------|
| Ed25519 classical security | 128-bit | PROTOCOL CONSTANT (RFC 8032) |
| Ed25519 quantum security | 64-bit (Grover) | PROTOCOL CONSTANT |
| SQIsign Level III classical | 192-bit | PROTOCOL CONSTANT ([IACR 2025/847](https://eprint.iacr.org/2025/897)) |
| SQIsign Level III quantum | 128-bit | PROTOCOL CONSTANT |
| SQIsign FFI linked | **false** | MEASURED (compile-time) |
| AEGIS-256 performance | 2-5x AES-GCM | PROTOCOL CONSTANT |
| VDF quantum resistance | **conjectured** | HONEST LABEL |
| LatticeGuard security | **pending** | HONEST LABEL |

The SQIsign entry reports `ffi_linked: false` because our current build uses a placeholder rather than the C reference implementation. We *could* hide this. We chose not to. (You're welcome.)

The VDF (Genus-2 hyperelliptic curve, IACR 2025/1050) is labeled "quantum resistance: conjectured" because the Jacobian DLP is solvable by Shor's generalisation in theory, even though the VDF's sequential evaluation may resist parallel quantum attacks. This was flagged during peer review by DeepSeek, who wrote: *"The document's own honest-comparison note says 'We do NOT claim quantum-proof' — this VDF claim contradicts that."* So we fixed it. Peer review works.

For comparison with Bitcoin: *"Same ~128-bit classical security as secp256k1 ECDSA. Both vulnerable to Shor's algorithm with ~2,330 logical qubits. The height-gated migration schedule (Phases 0→3) described here is absent in Bitcoin."*
We're not saying "we're better than Bitcoin". We're saying "we have a plan and a dashboard". Bitcoin has $1T market cap and no plan. Choose your own adventure.

**When does SQIsign actually turn on?** Not until: (1) the C reference implementation compiles and passes NIST KAT vectors, (2) constant-time validation is verified per [IACR 2025/832](https://eprint.iacr.org/2025/832), (3) performance is acceptable (SQIsign verify is ~50ms; at 250 solutions/block that's 12.5s — needs batch verification), (4) minimum 30 days on testnet, (5) the dashboard reports `ffi_linked: true`. Realistic estimate: Phase 1 (hybrid Ed25519+SQIsign) activation late 2026 at earliest. Ed25519 is not quantum-threatened in 2026. We're not rushing.

**One more thing we should disclose:** The SQIsign scaffold (used when FFI is off) *had* a verify function that always returned `Ok(true)`. That was a time bomb — any refactor that accidentally routed consensus verification through the scaffold would silently accept all signatures. We've replaced it with a hard error:

```rust
#[cfg(not(feature = "sqisign-ffi"))]
pub fn verify(_pk: &[u8], _msg: &[u8], _sig: &[u8]) -> Result<bool> {
    Err(anyhow!("FATAL: SQIsign verification called without FFI linkage. \
                  Enable 'sqisign-ffi' to link the C reference implementation."))
}
```

Now it crashes loudly instead of passing silently. The dashboard's `ffi_linked: false` is no longer the last line of defence — it's a status indicator backed by a hard error in the code path.

---

### 3. What Broke: Silent Block Pruning
*(Gutmann's Scenario D, live on stage)*

In the spirit of Gutmann's Scenario D ("operational bugs are more dangerous than cryptographic failures"), we discovered that an adaptive pruning system was silently deleting block bodies older than 30 days on **all** production nodes. Nobody opted in.

The root cause: two conflicting defaults in the same file.
- `PruningMode::default()` returned `Full` (no pruning) with a comment saying `"SAFE — NO PRUNING, must be explicitly enabled via env var."`
- But `PruningConfig::default()` hardcoded `Adaptive` directly, bypassing the first default.

The "fix" from testnet never reached the actual configuration path. So for the first 50 days of mainnet, every node was a forgetful librarian throwing away books after 30 days because "nobody read them anyway".

**Impact:** Historical block bodies (transactions, signatures) older than 30 days were deleted. Current state (all balances, all contracts) was unaffected — the pruning only deleted block bodies, not the state they produced. A fresh node can still sync to network tip in ~30 minutes via state snapshot + recent blocks. We verified this with a Docker sync test (and a lot of sweating).

**Fix:** One-line change to `PruningConfig::default()` — `Adaptive` → `Full`. Scheduler now exits immediately unless `Q_PRUNING_MODE=adaptive` is explicitly set. No node will silently delete blocks again.

**What's permanently lost:** Raw transaction bodies from approximately days 1-20 of mainnet. But if someone asks "show me the exact transaction in block 5,000,000" — we can't. That data was deleted by a config default nobody opted into.

**Why this isn't catastrophic:** Unlike Bitcoin's UTXO model where you need the full transaction chain to derive the current UTXO set, DAG-Knight uses a state-accumulator model. Balances, contracts, and all account state are maintained in a persistent state tree that is updated with each block but exists independently of the block bodies. Deleting a block body is like shredding a bank's paper transaction receipts — the account balances in the database are still correct, you just can't show the receipt anymore. A fresh node syncs the current state tree (via snapshot) plus recent blocks, and reaches full consensus in ~30 minutes. We verified this with a Docker sync test: the new node produced blocks at the network tip with identical state.

What's lost is *auditability*, not *correctness*. For a cash system, that matters — if Alice pays Bob on day 10, Bob currently cannot prove it to a tax auditor because the block body is gone.

**Archive node plan:** (1) Dedicated archive node with pruning permanently disabled, retaining all block bodies from the fix point (~day 20) onward. (2) Archive API endpoint (`/api/v1/archive/block/:height`) for historical queries. (3) Merkle inclusion proofs verifiable against block hashes (which are retained on all nodes even after body pruning). For the first 20 days: that data is gone. We cannot produce individual transaction proofs from that period. The state is correct, the receipts are lost. We're adding explicit language to the whitepaper: *"Full block body retention is guaranteed only on archive nodes. Pruned nodes retain state correctness and block hash chains but cannot serve historical transaction bodies."*

*Moral of the story:* Cryptographic breaks are exotic. Engineering defaults are mundane. Mundane bugs eat your data while you're asleep.

---

### 4. Memory-Induced P2P Death
*(Epsilon node had a drinking problem)*

The Epsilon supernode (10Gbit, 64GB RAM) was getting stuck every 6-8 hours with a repeating pattern:
memory grows to 30GB → systemd cgroup throttling → P2P connections drop to zero → block production times out → manual restart required.

**Root cause:** `ROCKSDB_BLOCK_CACHE_MB=16384` (16GB) on a node with `MemoryHigh=30G`. The auto-tune code computed 4GB but an env var override set 16GB. Combined with kernel page cache (no direct I/O on xxlarge tier), steady-state memory exceeded the cgroup limit within hours.

**Fix:** Cap block cache to 4GB, raise MemoryHigh to 50GB. Zero cgroup throttle events since the fix (was 760 events in 12 minutes before).

*Moral of the story, part 2:* Your fancy post-quantum signature scheme doesn't matter if your node OOMs because you told RocksDB to eat half the planet.

---

### Network Statistics (Days 50+)

- Chain height: ~14.6M blocks
- Block rate: 3.46 bps (target: 1.0 — network has more hashrate than anticipated. The emission controller adjusts reward per block to keep annual emission constant at 2,625,000 QUG/year regardless of rate; correction factor bounded to [0.5x, 2.0x]. A proper difficulty adjustment is implemented but not activated — changing it on a live $1B network requires the same careful testing as any consensus change. **Safety margin at actual rate:** κ_c = 2δΛ = 2×0.2×3.46 = 1.385, so κ/κ_c = 18/1.385 = **13×** — safe, but "warm" rather than the 45× margin at the target rate. Average anticone size ~1.4 concurrent blocks per propagation window versus ~0.4 at design point. The system handles it, but it's a design-point deviation that should be corrected, not compensated around.)
- Phase: ordered (κ/κ_c = 13x safety margin)
- Nodes: 4 production (Epsilon, Beta, Delta, Gamma) + community miners
- Zero consensus failures attributed to cryptography
- Three engineering incidents since March (all resolved, all documented publicly)
- One existential crisis about the meaning of "quantum-resistant" (resolved by adding the word "conjectured")

---

### Relevant Recent Papers

For those following the cryptographic primitives we use or plan to use:

- **SQIsign2DPush** — "Faster Signature Scheme Using 2-Dimensional Isogenies" ([IACR 2025/897](https://eprint.iacr.org/2025/897)). Smaller signing cost than SQIsign-v2.0 with equivalent signature size and verification cost. Relevant to our Phase 2 migration.

- **Constant-time SQIsign** — "Constant-time Integer Arithmetic for SQIsign" ([IACR 2025/832](https://eprint.iacr.org/2025/832), AFRICACRYPT 2025). Addresses the GMP non-constant-time concern. Our dashboard will report `constant_time_verified` status once we link the FFI.

- **Simple Power Analysis on SQIsign** — ([IACR 2025/830](https://eprint.iacr.org/2025/830)). Side-channel vulnerabilities in SQIsign implementations. We track but do not yet mitigate. (We're waiting for the "not-so-simple" power analysis paper.)

- **SPRINT** — "New Isogeny Proofs of Knowledge and Isogeny-Based Signatures" ([IACR 2026/364](https://eprint.iacr.org/2026/364)). Alternative isogeny-based construction. Worth watching as the field matures.

- **Harlow, Usatyuk & Zhao** — "Quantum mechanics and observers for gravity in a closed universe" ([arXiv:2501.02359](https://arxiv.org/abs/2501.02359), [JHEP 02(2026)108](https://link.springer.com/article/10.1007/JHEP02(2026)108)). The observer-dependence insight that motivated our K-gauge enhancement. Not cryptography per se, but the structural analogy is precise. Also, it's a great conversation starter at parties.

- **Lloyd** — "Computational capacity of the universe" ([Phys. Rev. Lett. 88, 237901, 2002](https://arxiv.org/abs/quant-ph/9908043)). The irreversible computation framework behind our commitment depth metric. Also useful for winning bar bets about whether the universe is a computer.

- **DAG-Knight** — "A Parameterless Generalization of Nakamoto Consensus" ([IACR 2022/1494](https://eprint.iacr.org/2022/1494.pdf)). The consensus protocol this is all built on. By Sompolinsky et al.

---

### Responding to jrzx's Timeline Argument

> *"This is unlikely to be sudden. [...] thirty years to figure out what is actually quantum resistant"*

The timeline argument is correct for the quantum threat *specifically*. But the pruning bug and the memory stall demonstrate that the threats that actually materialise are **engineering failures**, not cryptographic breaks.

Building the migration infrastructure *now* (height-gated phases, hybrid signatures, dashboard monitoring) costs engineering time. Not building it costs emergency response time when the threat arrives — and emergency cryptographic transitions on a live network with $1B in user funds are exactly the scenario we're trying to avoid.

So, jrzx, you're right about the 30 years. But in those 30 years, we'll have about 10,000 opportunities to shoot ourselves in the foot with config files. We'd rather spend the first year building crutches.

The dashboard exists so that when someone asks "is this system actually using post-quantum cryptography?" the answer is a live, honest, labeled measurement — not a marketing claim.

---

**Live data:** https://quillon.xyz (Explorer → Theoretical Physics Dashboard, Cryptography Dashboard)
**Source:** https://code.quillon.xyz
**Whitepaper v4:** https://quillon.xyz/downloads/theoretical-physics-consensus-whitepaper.pdf

*We promise not to break your brain unless you read the Hamiltonian derivation. In that case, the entropy of your understanding will increase. That's just physics.*

— Viktor

P.S. Alice and Bob are still alive, but Bob now uses Ed25519 and Alice is learning SQIsign. They argue about constant-time implementations over coffee. Eve is saving up for 2,330 logical qubits to run Shor's algorithm. At current qubit prices, she'll be ready around the heat death of the universe. She's patient. Alice is not worried — but she's learning SQIsign anyway, because Alice reads this mailing list.
