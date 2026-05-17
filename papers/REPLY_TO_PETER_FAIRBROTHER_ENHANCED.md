# Enhanced Reply to Peter Fairbrother

**Re: Temporal Dimensions of Cryptographic Security in Post-Quantum Infrastructure**

---

Peter,

Your point about classical breaks is well-taken and actually strengthens the case for defense-in-depth. As I argue in **Section 6 (The Snowden Baseline: Adversarial Capability is Real)** of the whitepaper, HNDL need not be quantum-exclusive. A classical algorithmic improvement (another Pollard-rho, improved index calculus, unexpected structure in specific curves) could collapse security margins retroactively. The hybrid approach addresses both vectors: if lattice assumptions fall classically, the classical layer remains; if classical falls to Shor or classical improvement, the PQ layer remains.

This is formalized in **Section 12.1 (Hybrid Classical-Postquantum Construction)**, which specifies:

> The architecture combines multiple cryptographic layers under distinct hardness assumptions, ensuring that compromise of any single primitive leaves other defensive layers intact.

## On Signatures and MITM Prevention

Agreed. For ephemeral key agreement where signatures merely authenticate the handshake, future forgery is irrelevant - the session keys are gone. The threat model differs.

But blockchain signatures create a subtler problem that I address in **Section 11 (Implications for Distributed Ledger Systems)**. Consider:

1. Alice signs transaction T at time t₀, moving funds from address A to address B
2. Transaction T is recorded on-chain with Alice's public key exposed
3. At time t₁ > t₀, an adversary recovers Alice's private key (quantum or classical break)
4. Address A still has a residual balance (Alice didn't move everything)
5. Adversary forges a new transaction T' moving residual funds to their address

The original signature on T remains valid - as you say, "pre-forgability signatures are secure." But the *key* is now compromised, and any funds remaining at addresses with exposed public keys become stealable. This isn't HNDL (decrypt stored data) or HNFL (forge historical signatures) - it's what I term "Harvest-Now-Steal-Later" (HNSL). The signature on T is fine; the problem is the key exposure that T created.

Bitcoin addresses that have *never* signed a transaction (pure receive addresses) are protected by hash preimage resistance. Addresses that *have* signed are protected only by ECDLP hardness. Satoshi's coins are safe until they move; everyone who's spent from an address is exposed.

## On 1536-bit RSA/DH

For ephemeral use, probably fine for decades. For immutable records, you're betting that no classical improvement appears in the lifetime of the chain. As noted in **Section 4 (Shannon's Limit Revisited: The Immutable Ceiling)**, GNFS improvements have been steady - 512-bit fell in 1999, 768-bit in 2009, 829-bit in 2020. The trajectory continues. 1536-bit has comfortable margin, but "comfortable margin on an immutable record" is a different risk calculation than "comfortable margin on tomorrow's TLS session."

The paper's **Section 10.1 (The Asymmetric Payoff Matrix)** formalizes this:

| Scenario | Prepare Now | Wait and See |
|----------|-------------|--------------|
| Threat Materializes Early | Protected | Catastrophic Loss |
| Threat Materializes Late | Small Overhead | Protected |
| Threat Never Materializes | Wasted Effort | Optimal |

The asymmetry is clear: early preparation has bounded downside (computational overhead), while waiting has potentially unbounded downside for permanent records.

## On "Quantum? Shmauntum. When/if I see it coming..."

This is where we part ways, and respectfully, it's where the immutability problem bites hardest.

For updatable systems, your attitude is entirely sensible. When you see it coming, you update. The cost of being wrong is a scramble to deploy patches. Uncomfortable, but survivable.

For immutable ledgers, "when I see it coming" is too late. As I argue in **Section 1 (Introduction: The Infrastructure Moment)**:

> Cryptographic choices made in 2025 will determine security postures for systems that must remain secure through 2050, 2075, and potentially beyond. Yet paradoxically, much of the current discourse treats quantum computing's cryptographic threat as a distant concern requiring no immediate action.

The data recorded in 2025 with 2025 cryptography cannot be updated in 2045 when the threat materializes. You can protect *new* data; you cannot protect *old* data. The window for defensive action is *before* the data is recorded, not when the threat arrives.

**Section 9 (The Hellman Imperative: Ethics of Action Under Uncertainty)** addresses this directly, citing Martin Hellman's 1976 warning about 56-bit DES:

> One must ask whether national security is best served by deployment of ciphers which have unknown, but potentially serious, weaknesses.

The same principle applies: are permanent records best served by cryptography with known quantum vulnerabilities, even if the timing is uncertain?

## On "Forever Impractical Due to Noise, Errors, and Coherence"

I'd have agreed with this framing until December 2024. The Willow below-threshold result changes the picture materially.

Prior to Willow, it was theoretically possible that error correction overhead would grow faster than physical qubit counts - that adding more qubits would make things worse, not better. This would have been a fundamental barrier, not an engineering challenge.

**Section 3 (The Google Willow Inflection)** analyzes this shift in detail:

> Willow demonstrated the opposite: adding qubits to their error-correcting code *reduced* the logical error rate. The exponential suppression of errors with code distance has now been empirically verified. The path to fault tolerance is no longer blocked by a physical barrier; it's gated by scale and engineering.

I'm not claiming CRQC next year, or even next decade. I'm claiming the "forever impractical" escape hatch has narrowed considerably. For systems designed to persist indefinitely, that narrowing matters.

**Section 2.4 (Synthesis: The Optimist Consensus)** compiles expert timeline estimates:

| Expert/Organization | Timeline |
|--------------------|----------|
| IBM | 2029 (100K logical qubits) |
| Google DeepMind | 2028-2035 |
| Michio Kaku | "Decades away" |
| NIST | "Within 10-20 years" |

The variance itself is informative: even the most conservative mainstream estimates put cryptographically-relevant quantum computing within the operational lifetime of infrastructure being deployed today.

## On Years of Analysis

You're correct that RSA and DH have survived decades of scrutiny, while NIST PQ algorithms have had only years. This is a genuine concern and why **Section 7 (The Gutmann Objection Revisited)** addresses Peter Gutmann's critique directly:

> Gutmann's objection deserves serious engagement precisely because it comes from genuine security engineering perspective rather than marketing or academic positioning.

But the comparison cuts both ways: RSA and DH also have proven quantum attacks (Shor), while lattice problems have no known quantum speedup beyond Grover's square root. The analysis gap is real; so is the algorithmic threat gap.

**Section 12.2 (Acknowledging Residual Risk)** states:

> The honest assessment is that we are choosing between known quantum vulnerabilities and unknown classical ones. Neither choice is risk-free. The hybrid approach minimizes total expected risk by requiring successful attacks on both layers.

The question isn't whether PQ crypto is battle-tested like RSA - it isn't. The question is whether "battle-tested and quantum-vulnerable" is preferable to "less-tested and quantum-resistant" for data that will exist when the quantum question is settled. For ephemeral data, classical wins on maturity. For permanent records, the risk calculus inverts.

## The AI Acceleration Factor

**Section 5 (AI as Landscape Navigator: The Variance Compression Model)** introduces a consideration often absent from quantum timeline discussions: artificial intelligence as a variance compressor in the quantum development landscape.

The paper models AI's impact formally in **Section 5.4**:

> AI acts as a meta-optimizer that compresses the variance in research trajectories. It doesn't guarantee faster arrival at the goal, but it reduces the probability of taking inefficient paths and increases the probability of finding efficient ones.

With NVIDIA Rubin bringing 1.4 exaFLOPs AI compute per node (**Section 5.3**), the simulation and optimization capacity available for quantum error correction research exceeds anything previously possible. This doesn't guarantee faster CRQC - but it changes the probability distribution of timelines in ways that should concern long-term infrastructure planners.

## Conclusion

Appreciate the continued engagement. Your engineering skepticism is valuable - it keeps the discussion grounded in what's actually achievable rather than what's theoretically elegant.

The core thesis of the paper (**Section 13: Conclusion**) is not that quantum computing will definitely arrive soon, but that:

> For systems designed to persist through the quantum transition - whenever it occurs - the time for cryptographic preparation is now. The inflection point is not when quantum computers become powerful enough to break existing cryptography; it is when permanent records begin using vulnerable cryptography.

For ephemeral systems, your approach is entirely defensible. For permanent records, I maintain the risk calculus points elsewhere.

Best regards,

**Viktor**

---

## Reference

Viktor. *Temporal Dimensions of Cryptographic Security: Decision Theory for Post-Quantum Infrastructure*. January 2026.

Available at: [https://quillon.xyz/downloads/temporal-cryptographic-security-v2.pdf](https://quillon.xyz/downloads/temporal-cryptographic-security-v2.pdf)

**Paper Sections Cited:**

1. Section 1: Introduction - The Infrastructure Moment
2. Section 2.4: Synthesis - The Optimist Consensus
3. Section 3: The Google Willow Inflection
4. Section 4: Shannon's Limit Revisited - The Immutable Ceiling
5. Section 5: AI as Landscape Navigator - The Variance Compression Model
6. Section 5.3: NVIDIA Rubin and Exascale AI Infrastructure
7. Section 5.4: Formal Model - AI as Variance Compressor
8. Section 6: The Snowden Baseline - Adversarial Capability is Real
9. Section 7: The Gutmann Objection Revisited
10. Section 9: The Hellman Imperative - Ethics of Action Under Uncertainty
11. Section 10.1: The Asymmetric Payoff Matrix
12. Section 11: Implications for Distributed Ledger Systems
13. Section 12.1: Hybrid Classical-Postquantum Construction
14. Section 12.2: Acknowledging Residual Risk
15. Section 13: Conclusion - The Inflection Point is Now
