# Draft Reply: Re: Quillon Graph — Response to A. Cryptographer

**Subject:** Re: [Cryptography] Quillon Graph: A post-quantum cash system with engineering chaos
**From:** Viktor S. Kristensen
**To:** cryptography@metzdowd.com

---

A. Cryptographer,

Every one of these is a fair hit. Let me take them in order, without hiding behind Greek letters.

### 1. "The Hamiltonian is just a restatement of the rules"

You're right. The Hamiltonian doesn't add cryptographic rigour to the DAG-Knight proof. Sompolinsky et al.'s combinatorial argument is the actual security theorem. What we proved (the "Ground State Theorem") is that the Hamiltonian formulation is *equivalent* to PHANTOM's output — not that it's *stronger*.

The value is operational, not cryptographic. Statistical mechanics gives us a decomposition language: "the network has high effective temperature" means "block propagation delay is large relative to block rate, causing ordering ambiguity." That's the same fact expressed differently — but the decomposition into H_parent, H_anticone, H_blue, H_VDF gives operators a dashboard where each component is independently actionable. You can't act on "consensus is degraded." You can act on "H_anticone is rising because propagation delay increased."

You're correct that this risks physics-washing. We tried to prevent that with the data honesty labels and the explicit statement in the paper: "This is an engineering contribution, not a physics contribution." But the risk is real, and your point is noted. If someone cites our paper as "proof that blockchain consensus obeys the laws of thermodynamics," that's a misreading we should actively correct.

To your specific challenge — does it lead to a new *algorithm*? Not yet. But it does lead to a new *diagnostic* (the K-gauge), which detected a class of attack (Sybil partition + shallow tip) that our previous monitoring missed entirely. Whether that justifies the formalism or whether a simpler heuristic would have worked equally well is a fair question. Probably the latter. But the formalism forced us to think carefully about what "network health" means, and that thinking produced a better monitor. I'll take the pragmatic win.

### 2. "How do you measure n_total?"

You caught the weakest point. We don't measure it. `n_total` is currently hardcoded to 50 (estimated network size). The paper's Section 24 (L11) explicitly states: *"n_total is hardcoded — we do not have a reliable network size estimator. A DHT crawl estimator or peer-exchange protocol could provide this but is not implemented."*

So yes, Ω_node is computed from local peer count divided by a configured constant:

    Ω = 1 − exp(−n_peers / 50)

This makes it a *post hoc* diagnostic with a tunable parameter, not a self-contained local measurement. A node with 12 peers gets Ω = 0.21. A node with 2 peers gets Ω = 0.04. The absolute value of Ω depends on the (guessed) denominator; the *relative* signal — "my peer count dropped significantly" — is what the K-gauge actually responds to.

An adversary who controls Ω by inflating `n_total` would need to also control the peer discovery mechanism (Kademlia DHT + hardcoded bootstrap). Deflating it is easier: partition the node and it sees fewer peers, Ω drops, K_enhanced rises — which is the correct response (flag the partition).

Could we replace this with a simpler "if peer_count < threshold, warn"? Absolutely. The exponential form gives diminishing returns (going from 0 to 5 peers matters more than going from 50 to 55), which matches operational reality. But I concede the Harlow analogy is doing more rhetorical work than mathematical work here.

**Action item:** We should implement a DHT crawl estimator for `n_total` and label Ω as "approximate until measured" in the dashboard. Adding this to the roadmap.

### 3. "Has there been cryptanalysis of the genus-2 VDF against quantum time-memory tradeoffs?"

Honest answer: no, we haven't evaluated against Eichlseder et al. (Asiacrypt 2024) specifically. Thank you for the reference — I wasn't aware of that result. If parallel quantum collision search reduces the effective security margin of genus-2 VDFs, that's directly relevant to us and we need to assess the impact.

Our current position: the VDF is labeled "conjectured" precisely because we lack confidence in its quantum resistance. The SHA3-based sequential hashing fallback (which is what actually runs in production — `advanced_crypto_enabled: false` in the dashboard) doesn't have this concern, since SHA3 preimage resistance under Grover is well-understood (halved security, still 128-bit for SHA3-256).

**Action item:** Evaluate Eichlseder et al. against our Genus-2 VDF parameters. If the result applies, downgrade the label from "conjectured" to "unlikely" and document the specific attack vector. Alternatively, commit to the SHA3 fallback as the production VDF and treat Genus-2 as research-only.

### 4. "You've effectively lost non-state data forever"

Correct. Blocks from approximately days 1-20 of mainnet are gone from all production nodes. We cannot serve historical transaction proofs for that period.

What remains:
- All balance state (who has what coins) — cryptographically committed in the state tree
- All block hashes and height pointers — the chain structure is intact
- All blocks from approximately day 20 onward — pruning stopped when we deployed the fix

What's lost:
- Raw transaction bodies (sender, receiver, amount, signature) for blocks before ~day 20
- The ability to independently re-verify those transactions from raw data

We do not currently have community backups containing the pruned blocks. The fix prevents future loss, but the historical loss is permanent unless someone archived a pre-pruning database snapshot.

You're right that this should be stated clearly. We'll add a "Data Retention" section to the whitepaper acknowledging:
1. Blocks from the first ~20 days of mainnet were irrecoverably pruned due to a configuration bug
2. The system now defaults to full retention (no pruning unless explicitly enabled)
3. The network relies on state snapshots, not full block replay, for new node synchronisation
4. An archive node policy should be established for long-term auditability

This is an honest loss. Not catastrophic (no funds were affected), but real, and it should be documented rather than glossed over.

### 5. "Cryptographic agility is itself an engineering nightmare"

Agreed. The height-gated phase transition mechanism (phases 0→3) is designed for exactly the scenario you describe:

- **Phase 0 (current):** Ed25519 only. All blocks use Ed25519. Simple.
- **Phase 1 (height 1,000,000):** HybridEd25519SQIsign. Both signatures required on every block. Old Ed25519-only blocks remain valid below the activation height. New blocks carry both. This is the "belt and suspenders" phase.
- **Phase 2 (height 2,500,000):** SQIsign Level III only. Ed25519 signatures are no longer required (but still accepted for historical block verification). The chain doesn't forget how to verify old signatures — it just stops requiring them on new blocks.
- **Phase 3 (height 4,000,000):** FROST threshold + SQIsign. Requires a validator committee.

The mixed-epoch problem you raise is handled by the `verify_spectral_signature_extended` function, which dispatches on the block's `SignaturePhase` field. A block at height 999,999 is verified with Phase 0 rules (Ed25519). A block at height 1,000,001 is verified with Phase 1 rules (both must pass). Old signatures never "expire" — they remain valid under their era's rules. This is the same pattern as Bitcoin's soft fork activation (BIP9/BIP8), adapted for signature scheme transitions.

**When do we expect to turn SQIsign on?** Not until:
1. The C reference implementation compiles and passes NIST KAT vectors on our target platforms
2. Constant-time validation is verified (we're tracking [IACR 2025/832](https://eprint.iacr.org/2025/832))
3. Performance is acceptable (SQIsign verify is ~50ms; at 250 solutions per block, that's 12.5s — dangerously close to block time, needs batch verification or reduced solution count)
4. Minimum 30 days on testnet
5. The dashboard reports `ffi_linked: true` and `constant_time_verified: true`

Realistic timeline: Phase 1 activation in late 2026 at the earliest. We're not rushing this. Ed25519 is not quantum-threatened in 2026.

### 6. "Where does the exponential in Λ_commit come from?"

Fair catch. It's not from Lloyd directly. It's a Poisson process assumption on block arrivals.

If blocks arrive at rate λ (blocks/second) and are independently produced, the probability that a block at depth d has *not* been built upon by k or more descendants follows an exponential decay. The commitment factor Λ = 1 − exp(−d/(κ·τ_confirm)) is the CDF of this process, parameterised by the DAG-Knight tolerance κ and a confirmation depth constant τ_confirm = 100 blocks.

Lloyd's contribution is the *motivation*: the idea that irreversible computation creates classicality. The *formula* is a standard survival function applied to block confirmations. We should have said "motivated by Lloyd's framework, derived from a Poisson arrival model" rather than implying Lloyd's work directly produced the exponential. Fixing this in the next paper revision.

### 7. "Block rate 3.46 bps vs target 1.0 bps — explain"

This is not a consensus bug, though I understand why it looks like one.

The block rate is higher than the target because the network has more hashrate than anticipated at launch. The difficulty adjustment targets a specific block interval, but the current implementation allows the emission controller to compensate by adjusting the *reward per block* rather than the *block rate*. This keeps total annual emission constant (2,625,000 QUG/year) regardless of how fast blocks are produced.

The emission controller tracks: `actual_emission / target_emission = correction_factor`. At 3.46 bps, the correction factor is ~0.96 (reducing reward per block by ~4% to compensate for the higher production rate). Over any 24-hour window, total emission matches the target within 5%.

You're right that this creates a potential attack surface. If a miner can manipulate the block rate (e.g., by selectively withholding blocks, then releasing them in bursts), they could exploit the emission correction. The defence is that the correction factor is bounded: it cannot exceed 2.0x or drop below 0.5x, preventing extreme manipulation. But a formal analysis of timing attacks against this mechanism is on our TODO list.

The proper fix is a difficulty adjustment algorithm that targets 1.0 bps. We have one in the codebase but it's not activated because changing difficulty adjustment on a live $1B network requires extensive testing. It's Phase B in our mining fairness roadmap.

### Bottom line

You asked whether the stat-mech framework leads to a new algorithm or just restates existing rules. Right now, it's the latter with a diagnostic bonus. That's worth being honest about.

The real contributions of this update are:
1. A dashboard that shows `ffi_linked: false` on the front page
2. Three engineering post-mortems published openly
3. A migration plan with compile-time activation heights
4. An invitation to scrutinise all of the above

When we link the SQIsign FFI and the dashboard reports `constant_time_verified: true`, I'll send another update. Until then, we'll keep publishing our config-file disasters. At least those are entertaining.

— Viktor

P.S. If you do run a node, the binary is 85MB and syncs to tip in 30 minutes. The config file has 4 environment variables. We've triple-checked that none of them secretly delete your blocks. (Twice.)

P.P.S. Eve has upgraded to a quantum laptop but still only gets 64 bits of security against Ed25519. She's considering a career change.
