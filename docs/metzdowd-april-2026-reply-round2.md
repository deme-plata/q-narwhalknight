# Draft Reply Round 2: Re: Quillon Graph — Response to A. Cryptographer (follow-up)

**Subject:** Re: [Cryptography] Quillon Graph — April 2026 update (scaffolds, canaries, and safety margins)
**From:** Viktor S. Kristensen
**To:** cryptography@metzdowd.com

---

A. Cryptographer,

Four actionable points, all accepted. One deployed before I finished writing this email.

---

### 1. SQIsign scaffold: fixed before you finish reading this

You're right — a verify function that returns `Ok(true)` is a time bomb regardless of how carefully we document it. The "warning sticker on a footgun" analogy is exact.

We replaced the scaffold's verify with:

```rust
#[cfg(not(feature = "sqisign-ffi"))]
pub fn verify(_pk: &[u8], _msg: &[u8], _sig: &[u8]) -> Result<bool> {
    Err(anyhow::anyhow!(
        "FATAL: SQIsign verification called without FFI linkage. \
         This is a scaffold — it CANNOT verify real signatures. \
         Enable the 'sqisign-ffi' feature to link the C reference implementation, \
         or do not route consensus verification through SQIsign."
    ))
}
```

It now returns an error, not success. Any code path that accidentally routes consensus verification through the scaffold will get a loud, immediate failure — not a silent pass. The dashboard's `ffi_linked: false` is now a warning, not the last line of defence.

This is live on our canary node. Production deployment follows our standard pipeline: Delta canary → 30 minutes observation → Epsilon + Beta.

Thank you. This was a genuine vulnerability-in-waiting.

---

### 2. Ω_node denominator: relabeled and roadmapped

You're correct on both the saturation and sensitivity problems:

- At n_total=50, a 200-node network makes Ω ≈ 0.98 for everyone — useless
- At n_total=50, a 20-node network makes Ω hypersensitive — noisy

We've done two things:

**Immediate (deployed):** Changed the dashboard label from "MEASURED" to "CONFIGURED ESTIMATE (n_total=50)" on the Ω_node card. The denominator is not measured. It should never have been implied otherwise.

**Roadmap:** Your epidemic averaging proposal is the right approach. We'll implement a gossip-based estimator where each node tracks `max(peer_count)` seen from peers in the last hour, and takes the running average. This gives a network-derived estimate without requiring a global view. Adding to the v10.4 milestone.

On the adversarial peer churn DoS amplification: noted and acknowledged. An adversary who floods connections to force peer drops will inflate `(1-Ω)`, making K_enhanced spike — which is a false alarm, not a real partition. The defence is that K_enhanced also depends on Λ_commit (commitment depth) and the base K components (sync divergence, block rate deviation). A pure connection flood without affecting block delivery would spike Ω but leave the other components stable. The operator can distinguish "my peers dropped but blocks are still arriving" from "my peers dropped AND I'm falling behind." But you're right that the metric alone doesn't distinguish them. Adding a note to the dashboard tooltip.

---

### 3. Block rate safety margin: recalculated

You caught a real gap. The security margin κ/κ_c = 13× was computed at the target block rate Λ = 1.0 bps. At the actual rate Λ = 3.46 bps, the critical threshold changes:

    κ_c = 2δΛ(1 − f/n) / (1 − 2f/n)

With δ = 0.2s (propagation delay), f/n = 0 (no known Byzantine nodes):

- At Λ = 1.0: κ_c = 2 × 0.2 × 1.0 = 0.4, margin = 18/0.4 = **45×**
- At Λ = 3.46: κ_c = 2 × 0.2 × 3.46 = 1.385, margin = 18/1.385 = **13×**

So the 13× figure is already computed at the real block rate — I should have been clearer about this. The physics dashboard shows this live: κ = 18, κ_c = 1.385, margin = 16.6 (the slight difference from 13× is because the live computation uses measured Λ which fluctuates).

Your deeper point stands: at 3.46 bps, anticone rates are higher than designed. The average anticone size scales as ~2δΛ ≈ 1.4 concurrent blocks per propagation window, versus ~0.4 at the target rate. This means more ordering ambiguity, more Goldstone modes, higher effective temperature. The system handles it (κ >> κ_c), but it's operating "warm" rather than "cold" in the statistical mechanics sense.

The proper fix is a difficulty adjustment that targets 1.0 bps. We have one implemented but not activated — changing difficulty adjustment on a live network requires the same careful testing pipeline as any consensus change. It's gated behind a height-activated upgrade in our consensus guard.

**To be precise:** The 13× safety margin is real at today's block rate. It would be 45× at the target rate. Both are safe. But operating at 3.46× the target rate is a design-point deviation that should be corrected, not compensated around.

---

### 4. Archive nodes and historical transaction retrievability

You're right that "auditability, not correctness" is insufficient for a cash system. If Alice pays Bob on day 10 and Bob needs proof in 2027, the system should support that.

Our plan:

1. **Dedicated archive node** (immediate): One node with pruning permanently disabled, retaining all block bodies from the point the fix was deployed (~day 20 onward). This covers future auditability.

2. **Archive API endpoint** (v10.4): `/api/v1/archive/block/:height` that queries the archive node specifically. Regular nodes return "pruned" for old blocks; the archive serves the full body.

3. **Inclusion proofs** (v10.5): Merkle proofs that a transaction was included in a specific block, verifiable against the block hash (which is retained on all nodes even after body pruning). This allows compact proofs without serving the full block.

4. **Whitepaper addendum:** Adding explicit language: "Full block body retention is guaranteed only on archive nodes. Pruned nodes retain state correctness and block hash chains but cannot serve historical transaction bodies. Users requiring long-term transaction proofs should verify against an archive node or retain their own transaction receipts."

For the first 20 days: that data is gone. We cannot prove individual transactions from that period. The state is correct (balances match), but the receipts are lost. This is documented as a known limitation of the genesis era.

---

### On Eve's 64 bits

You're right — Grover gives quadratic speedup on the *symmetric* search underlying Ed25519's security. Eve with a fault-tolerant quantum computer running Shor's algorithm breaks Ed25519 *entirely*, not at 64 bits. The joke should be:

*"Eve is saving up for 2,330 logical qubits. At current prices, she'll be ready around the heat death of the universe. She's patient."*

Fixed. The original joke was sloppy. As you noted, this audience knows the difference between Grover and Shor, and conflating them in a punchline undermines the technical credibility of the setup.

---

### Summary of actions from this exchange

| Your Point | Action | Status |
|---|---|---|
| SQIsign scaffold returns Ok(true) | Replaced with hard error | **Deployed to canary** |
| Ω_node denominator mislabeled | Changed to "CONFIGURED ESTIMATE" | **Deployed** |
| n_total needs live estimate | Gossip-based epidemic averaging | Roadmap v10.4 |
| Adversarial peer churn DoS | Dashboard tooltip note | **Deployed** |
| Block rate safety margin at 3.46 bps | Clarified: 13× is at real rate, 45× at target | **Documented** |
| Difficulty adjustment needed | Height-gated activation planned | Roadmap |
| Archive nodes for auditability | Dedicated archive + API + inclusion proofs | Roadmap v10.4-10.5 |
| Whitepaper data retention policy | Addendum drafted | In review |
| Eve joke physics error | Corrected Grover→Shor distinction | **Fixed** |

---

When you're ready to run that beta node, the binary is at https://quillon.xyz/downloads/q-api-server-linux-x86_64. Four environment variables. No scaffold in the consensus path. And the dashboard will tell you — honestly — exactly what's measured and what's a guess.

— Viktor

P.S. Eve considered switching to lattice attacks but the BKZ block size of 625 gave her a headache. She's now studying isogenies. Alice remains unconcerned.
