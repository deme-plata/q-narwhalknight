# Draft Response to Peter Fairbrother (Jan 8, 2026)

---

Peter,

Your point about classical breaks is well-taken and actually strengthens the case for defense-in-depth. HNDL need not be quantum-exclusive — a classical algorithmic improvement (another Pollard-rho, improved index calculus, unexpected structure in specific curves) could collapse security margins retroactively. The hybrid approach addresses both vectors: if lattice assumptions fall classically, the classical layer remains; if classical falls to Shor or classical improvement, the PQ layer remains.

On signatures and MITM prevention: agreed. For ephemeral key agreement where signatures merely authenticate the handshake, future forgery is irrelevant — the session keys are gone. The threat model differs.

But blockchain signatures create a subtler problem. Consider:

1. Alice signs transaction T at time t₀, moving funds from address A to address B
2. Transaction T is recorded on-chain with Alice's public key exposed
3. At time t₁ > t₀, an adversary recovers Alice's private key (quantum or classical break)
4. Address A still has a residual balance (Alice didn't move everything)
5. Adversary forges a new transaction T' moving residual funds to their address

The original signature on T remains valid — as you say, "pre-forgability signatures are secure." But the *key* is now compromised, and any funds remaining at addresses with exposed public keys become stealable. This isn't HNDL (decrypt stored data) or HNFL (forge historical signatures) — it's "Harvest-Now-Steal-Later" (HNSL?). The signature on T is fine; the problem is the key exposure that T created.

Bitcoin addresses that have *never* signed a transaction (pure receive addresses) are protected by hash preimage resistance. Addresses that *have* signed are protected only by ECDLP hardness. Satoshi's coins are safe until they move; everyone who's spent from an address is exposed.

On 1536-bit RSA/DH: for ephemeral use, probably fine for decades. For immutable records, you're betting that no classical improvement appears in the lifetime of the chain. GNFS improvements have been steady — 512-bit fell in 1999, 768-bit in 2009, 829-bit in 2020. The trajectory continues. 1536-bit has comfortable margin, but "comfortable margin on an immutable record" is a different risk calculation than "comfortable margin on tomorrow's TLS session."

On "Quantum? Shmauntum. When/if I see it coming...":

This is where we part ways, and respectfully, it's where the immutability problem bites hardest.

For updatable systems, your attitude is entirely sensible. When you see it coming, you update. The cost of being wrong is a scramble to deploy patches. Uncomfortable, but survivable.

For immutable ledgers, "when I see it coming" is too late. The data recorded in 2025 with 2025 cryptography cannot be updated in 2045 when the threat materializes. You can protect *new* data; you cannot protect *old* data. The window for defensive action is *before* the data is recorded, not when the threat arrives.

This is the core asymmetry the paper tries to formalize: updateable systems can wait for evidence; permanent records cannot.

On "forever impractical due to noise, errors, and coherence":

I'd have agreed with this framing until December 2024. The Willow below-threshold result changes the picture materially.

Prior to Willow, it was theoretically possible that error correction overhead would grow faster than physical qubit counts — that adding more qubits would make things worse, not better. This would have been a fundamental barrier, not an engineering challenge.

Willow demonstrated the opposite: adding qubits to their error-correcting code *reduced* the logical error rate. The exponential suppression of errors with code distance has now been empirically verified. The path to fault tolerance is no longer blocked by a physical barrier; it's gated by scale and engineering.

I'm not claiming CRQC next year, or even next decade. I'm claiming the "forever impractical" escape hatch has narrowed considerably. For systems designed to persist indefinitely, that narrowing matters.

On the years of analysis point: You're correct that RSA and DH have survived decades of scrutiny, while NIST PQ algorithms have had only years. This is a genuine concern and why the hybrid approach exists. But the comparison cuts both ways: RSA and DH also have proven quantum attacks (Shor), while lattice problems have no known quantum speedup beyond Grover's square root. The analysis gap is real; so is the algorithmic threat gap.

The question isn't whether PQ crypto is battle-tested like RSA — it isn't. The question is whether "battle-tested and quantum-vulnerable" is preferable to "less-tested and quantum-resistant" for data that will exist when the quantum question is settled. For ephemeral data, classical wins on maturity. For permanent records, the risk calculus inverts.

Appreciate the continued engagement. Your engineering skepticism is valuable — it keeps the discussion grounded in what's actually achievable rather than what's theoretically elegant.

Best regards,
Viktor

---

**Key points to emphasize:**

1. Classical breaks are a valid HNDL vector — hybrid addresses both
2. Signature MITM use case is different from blockchain key exposure
3. "See it coming" works for updateable systems, fails for immutable ones
4. Willow below-threshold result closes the "forever impractical" argument
5. Hybrid acknowledges the PQ maturity gap while hedging against quantum threat

**Tone:** Respectful, technically engaged, acknowledges valid points while maintaining position on immutability asymmetry.
