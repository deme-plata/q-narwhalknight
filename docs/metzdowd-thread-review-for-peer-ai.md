# Quillon Graph — Metzdowd Thread Review Document

**Purpose:** Comprehensive summary of all criticisms, arguments, and replies in the [Cryptography] Quillon Graph thread on the metzdowd mailing list (January--March 2026), annotated with commentary for peer AI review.

**Thread URL:** https://www.metzdowd.com/pipermail/cryptography/2026-January/039255.html (January), https://www.metzdowd.com/pipermail/cryptography/2026-March/039383.html (March)

**Participants:** Viktor S. Kristensen (OP/developer), Peter Gutmann (Auckland), Peter Fairbrother, John Gilmore, Ray Dillinger (Bear), Jerry Leichter, iang, Howard Chu, zeb, jrzx

---

## 1. Thread Summary

Viktor announced Quillon Graph (Q-NarwhalKnight) in December 2025 as a "private, post-quantum electronic cash system." The thread ran 30 messages across January and March 2026. The criticisms fall into five categories:

1. **Quantum skepticism** — Is PQC necessary, or premature/harmful?
2. **Premature standardization** — NIST PQC as a DES-like mistake
3. **NSA trust deficit** — Why trust government crypto recommendations?
4. **Privacy architecture** — Can a DAG-BFT be private?
5. **Scalability** — Can total ordering work at global scale?

---

## 2. Critic-by-Critic Summary with Key Quotes

### 2.1 Peter Gutmann (University of Auckland)

**Position:** Quantum threat is unproven; PQC migration is a distraction from real vulnerabilities.

**Key messages:** 039252, 039253, 039261, 039267, 039283, 039297

**Core arguments:**

(a) "Scenario D — Desinformatsiya": While the world focuses on an unproven quantum threat, real vulnerabilities go unaddressed. Introducing new PQC code creates fresh attack surface.

> "While the world is fixated on dealing with a threat that no-one has been able to prove exists, we're not addressing actual vulnerabilities that are being exploited by attackers. Also, we're introducing a host of additional vulnerabilities through a pile of new, unproven crypto code that will no doubt give attackers footholds for years if not decades to come. If I wanted to perform a long-term DoS on effective use of crypto, this is exactly how I'd do it." — Gutmann, Jan 11

(b) No demonstrated quantum result against (EC)DH, only factoring claims (which are "easier to cheat with"):

> "Nothing is subject to HNDL. Have you noticed how all the claimed quantum factorisation records are for just that, factorisation? No-one has ever claimed to have achieved any quantum result against (EC)DH, which would be needed for HNDL against TLS, SSH, IPsec..." — Gutmann, Jan 7

(c) Classical crypto "just works" — complexity is the enemy:

> "As long as you use FIPS 186-style parameter gen and verification... you're sorted, you don't have to worry about a neverending parade of subtle problems and corner cases. It just works." — Gutmann, Jan 8

(d) Defended DES standardization against Bear's critique — the benefit of a standard outweighs prematurity:

> "Instead of a million homebrew pieces of snake oil there was now a well-defined standard that you could look at and say 'this won't be trivially broken in half an hour by an undergraduate university student'... As the NSA said at the time, it was the biggest mistake they ever made." — Gutmann, Jan 12

**Commentary:** Gutmann is the most prolific and most skeptical critic. His "Scenario D" framing is the strongest argument against PQC urgency — that it diverts engineering attention from real, exploitable bugs. Viktor conceded this point with operational evidence (sync-down bug, OOM crashes, port collisions all more dangerous than any quantum attack in the first month). However, Gutmann's position conflates "dominant threat today" with "only threat worth addressing." The asymmetry between recoverable engineering bugs and irrecoverable cryptographic breaks remains unaddressed in his framing.

**Open question for review:** Does Gutmann's position account for the HNSL (Harvest Now, Steal Later) threat specific to immutable ledgers, where public keys are permanently exposed?

---

### 2.2 John Gilmore

**Position:** NSA's advocacy for PQC is inherently suspicious given 70 years of documented adversarial behavior toward civilian cryptography.

**Key messages:** 039276, 039310

**Core arguments:**

(a) Historical pattern of NSA undermining civilian crypto: Crypto AG, DES, Dual_EC_DRBG.

> "Why is NSA pushing belt-without-suspenders quantum-resistant crypto? Shrug. What *is* known is they've publicly advocated broken or jiggered easily-breakable crypto for at least seven decades. On this list nobody needs to list all of them, but let's start with Crypto AG, DES, and Dual EC DRBG. You could bet that they've changed their spots, but it's a sucker bet." — Gilmore, Jan 10

(b) The DES parallel applies directly — NSA told people "the security is fine" while exploiting the weakness.

(c) Later recommended reading Gosling's "Phase Relationships in the Standardization Process" (1990) regarding when standards are mature enough to be net positive.

**Commentary:** Gilmore's critique is institutional rather than technical. He does not argue that lattice problems are weak; he argues that trusting NIST recommendations requires trusting an organization with a documented adversarial track record. Viktor responded with three mitigations: (1) no NIST-blessed RNG, (2) hybrid classical+PQ where breaking one is insufficient, (3) height-gated algorithm replacement if lattice assumptions fall. Gilmore did not respond to these mitigations.

**Open question for review:** Is the hybrid approach sufficient to address the "NSA might have broken lattice assumptions" scenario? If both layers must verify, what is the attack surface of the combining mechanism itself?

---

### 2.3 Ray Dillinger (Bear)

**Position:** PQC standards are "ludicrously premature" and will cause DES-like damage by inhibiting diverse research.

**Key messages:** 039290 (Jan 12), 039384 (Mar 2)

**Core arguments:**

(a) Premature standards are actively counterproductive because they lock in a single approach before the problem is understood:

> "Any standard for quantum-resistant cryptography is a ludicrously premature standard made without any input from real-world systems or threats. As such it can only be harmful." — Bear, Jan 12

(b) Diversity is the answer — maintain "a few hundred different attempted solutions" before standardizing:

> "When the iron does start to get hot, it would be nice to look over a stable of a few hundred different attempted solutions, consider whatever the best ten or twelve turn out to be against whatever reality happens to turn out to be, and THEN try to discern the criteria necessary for a standard to address." — Bear, Jan 12

(c) The NIST recommendation for lattice-only (not hybrid) is "particularly suspicious":

> "The odd fixation on using the 'QC-resistant' lattice algorithm *by itself* is particularly suspicious. That algorithm hasn't yet had sufficient attention, expertise, and time devoted to its analysis. As such it should ONLY be deployed in multi-layered implementations, and any recommendation to the contrary seems like a red flag that some kind of shenanigans are going on." — Bear, Jan 12

(d) March 2 response to mainnet launch: sardonic sympathy.

> "My condolences. Try to remain optimistic through the hardships you must endure, value your relationships with family and those close to you, and have courage." — Bear, Mar 2

**Commentary:** Bear's diversity argument is one of the strongest in the thread. Viktor directly incorporated it: the crypto-agility architecture (height-gated activation) allows algorithm replacement without chain reset, treating PQC as one of many candidate layers rather than a permanent commitment. Bear's concern about lattice-only deployment is addressed by the hybrid design — Quillon uses classical + PQ layers, requiring both to verify.

Bear's March condolences turned out prescient: Viktor's March 26 status report documented 28 critical bugs in the first 10 days of mainnet.

**Open question for review:** Is height-gated crypto-agility actually sufficient to implement Bear's "hundreds of solutions" principle? In practice, how many algorithm replacements can the protocol undergo before the validation rules become unmanageably complex?

---

### 2.4 Peter Fairbrother

**Position:** DAG-BFTs aren't private; HNDL has limited blockchain targets; classical crypto is sufficient.

**Key messages:** 039240, 039241, 039255

**Core arguments:**

(a) DAG-based BFTs produce public blockchains, not private ones:

> "DAG-based BFTs like Bullshark or Mysticeti produce a consensus blockchain quickly (though not in 50ms), but the blockchain isn't particularly private, especially if one node/validator makes the blockchain available." — Fairbrother, Jan 7

(b) Signatures aren't subject to HNDL — only encrypted data is:

> "Signatures aren't really subject to HNDL — at later, either they are still secure and can't be forged, or it is known that they were both secure and published... or they can be forged later — in which case they can't be relied on later." — Fairbrother, Jan 7

(c) Quantum computing may be "forever impractical due to noise, errors."

(d) Prefers 1536-bit RSA/DH as "practical protection."

**Commentary:** Fairbrother's privacy critique was the "most technically important" according to Viktor. The response clarified that privacy comes from the transaction layer (LSAG ring signatures, stealth addresses, Bulletproofs++) rather than the DAG consensus layer. The DAG orders "opaque transactions" — validators verify validity proofs without learning transaction contents.

Fairbrother's HNDL/signature distinction is correct for classical breaks but misses the HNSL blockchain-specific threat: a quantum adversary does not forge historical signatures, they recover the private key from published public keys and forge *new* transactions to steal residual funds.

**Open question for review:** Is the separation between "opaque transaction layer" and "public DAG consensus layer" architecturally clean, or are there side channels through which DAG structure leaks transaction information?

---

### 2.5 iang

**Position:** Complexity is the enemy; TLAs love complexity because developers make mistakes.

**Key message:** 039258

> "Worse than that — it costs 3 times in complexity and therefore attack surface — you've introduced 2 algorithms to attack and the gap between them is a complexity that introduces weaknesses." — iang, Jan 8

> "Unless you've got something that the spooks really value and you're likely a kinetic target already, you're better off going best of simple class and forgetting about the tail risk." — iang, Jan 8

**Commentary:** iang's complexity argument echoes Gutmann's Scenario D from a different angle: hybrid crypto triples the attack surface. This is a legitimate concern. Viktor's 767,000-line codebase is evidence of this complexity cost. However, iang's "swap over with an emergency update" advice assumes the migration window exceeds the exploitation window — exactly the assumption that HNSL invalidates for immutable ledgers.

---

### 2.6 Jerry Leichter

**Position:** NSA reserves complexity for external use; internal designs are simpler.

**Key message:** 039264

> "We've gotten to see a few of their designs — Skipjack, for example. For internal or approved-for-US-secrets use, they seem to go for reasonably simple designs... The complexity seems to be reserved for stuff they expect others to implement." — Leichter, Jan 8

**Commentary:** Supports Gilmore's institutional distrust thesis from a technical implementation angle. If NSA's own crypto is simpler than what they recommend for others, the motive is suspect.

---

### 2.7 Howard Chu

**Position:** ARM compatibility concerns are overblown.

**Key message:** 039383 (Mar 1)

Chu dismissed concerns about AES-NI lacking on Raspberry Pi, noting most ARM SoCs now have AES instruction support. Recommended deprioritizing optimization for Broadcom-based Pis.

**Commentary:** Narrow technical point, not a criticism of the project's fundamentals. Viktor confirmed only 7 of 182 miners were ARM-based.

---

### 2.8 zeb (zeb@qtt.se)

**Position:** 680,000 lines is extraordinarily large.

**Key message:** 039339 (Jan 25)

> "680,000+ lines of Rust feels extraordinarily large, even for a reference implementation and even including tests, examples, tooling, and vendored deps. Have you considered a minimal implementation (like a few hundred lines)?" — zeb, Jan 25

**Commentary:** Valid concern about auditability. Viktor clarified the count and acknowledged the codebase grew to ~944,000 lines by March. The size is a consequence of implementing privacy proofs, PQ crypto, DAG consensus, Tor integration, and a full API server in a single workspace. A "few hundred lines" minimal implementation would cover only the consensus core, not the full system.

---

### 2.9 jrzx

**Position:** Scalability and ordering cannot work at global scale; quantum threat is premature.

**Key message:** 039442 (Mar 27)

*Part 1 — Scalability (posted to list):*

> "The capability to order and prove the validity of a very large number of transactions per second implies that, at the intended scale (replacing the fiat money in the world as a whole) there will be an enormous amount of data." — jrzx, Mar 27

> "I don't see that total ordering can scale to the required or claimed size? Is it perhaps a partial order — that all transactions in one group are later than the previous group and earlier than the next group?" — jrzx, Mar 27

*Part 2 — Quantum skepticism (attributed to jrzx, possibly separate communication):*

> "Protected by blockchain immutability. Spend authorizations of unspent transaction outputs will not suddenly become forgeable."

> "Now is far too early to start generating quantum resistant unspent transaction outputs, for quantum computing remains purely theoretical, and quantum resistance is not merely theoretical but hypothetical, indistinguishable in practice from snake oil, grantsmanship, and cryptocurrency scamming."

> "When we have quantum computers that can maintain quantum coherence while factoring arbitrary sixty four bit numbers, then it might become useful to think about what a quantum resistant algorithm might look like."

**Commentary:** jrzx's scalability question correctly identifies that the system uses partial ordering, not total ordering. This is by design (DAG-BFT), not a limitation. Viktor's draft reply confirms this and provides honest scaling numbers.

The quantum skepticism mirrors Gutmann/Bear's position but goes further: characterizing PQC as "indistinguishable from snake oil." This ignores: (a) NIST's 8-year standardization process, (b) NSA CNSA 2.0 mandates, (c) published error correction milestones (Google Willow, Microsoft Majorana 1), (d) the HNSL threat specific to immutable ledgers where public keys cannot be un-published. The "wait until 64-bit factoring" benchmark is a red herring — ECC is 2.6x easier to break quantumly than RSA, and error correction scaling (not integer factoring) is the relevant hardware milestone.

---

## 3. Cross-Cutting Themes

### 3.1 The Core Tension: Engineering Bugs vs. Cryptographic Breaks

All critics agree that real-world engineering failures dominate today. Viktor conceded this with operational evidence. The disagreement is about whether this justifies ignoring the asymmetry between:
- **Recoverable** failures (bugs, OOM, sync errors — fixable in hours)
- **Irrecoverable** failures (ECDLP break — all published public keys permanently compromised)

### 3.2 Historical Precedent for "We'll Migrate When Threatened"

The thread lacks discussion of cases where theoretical cryptographic vulnerabilities became practical exploits before the ecosystem migrated. Real examples that support Viktor's position:

- **ECDSA nonce reuse (2013):** Android PRNG bug → private key recovery → funds stolen from Bitcoin wallets
- **Polynonce attack (2023):** Kudelski Security recovered hundreds of Bitcoin/Ethereum private keys from on-chain signatures. ~$40K stolen from a single wallet.
- **Biased nonce lattice attacks (2019):** 144 BTC ($9.4M) stolen via lattice-based ECDSA key recovery from blockchain signatures
- **Eclipse via Tor (2015):** Biryukov & Pustogarov demonstrated that 40 Tor exit nodes could create a "virtual Bitcoin reality" for all Tor-connected users, enabling double spends
- **51% attacks on ETC/BTG (2018-2020):** $18M stolen from BTG; ETC suffered three 51% attacks in one month with 7,000+ block reorgs

In every case, the vulnerability was known theoretically before exploitation. In every case, the ecosystem did not migrate in time.

### 3.3 The Premature Standardization Debate

Bear and Gutmann disagree about DES:
- **Bear:** DES standardization was harmful — locked the world into a weak algorithm for 30 years
- **Gutmann:** DES standardization was beneficial — replaced "a million homebrew snake oil" with something auditable

Both are correct for their respective time horizons. The resolution may be crypto-agility: standardize the *replacement mechanism*, not the algorithm.

### 3.4 NSA Trust

Gilmore, Bear, Leichter, and iang all express distrust of NSA's motives for pushing PQC. No participant defends NSA's integrity. Viktor's response (hybrid design, no NIST RNG, algorithm replaceability) attempts to be trust-minimizing but does not fully resolve the concern: if the lattice assumptions underlying ML-DSA/ML-KEM were secretly broken, the hybrid approach still provides classical security, but the PQC layer becomes dead weight that adds complexity without protection.

---

## 4. Viktor's Acknowledged Weaknesses (from March 26 status report)

Viktor's own March 26 post acknowledged these limitations:

1. Privacy layer uses quantum-vulnerable ring signatures (curve25519-based LSAG)
2. No independent audit of any cryptographic implementation
3. SQIsign FFI only covers Level I; Levels III/V still use hash scaffolds
4. VDF security parameters not formally analyzed
5. DAG-Knight consensus not formally verified (4,000+ tests but no proof)
6. Tor integration not evaluated against global passive adversary
7. 28 critical bugs in first 10 days of mainnet
8. Emission constant error (3 extra zeros in a u128 integer)
9. TCP self-connection bug from ephemeral port range overlap
10. 75,000 stale TCP connections from HTTP keepalive misconfiguration

**Commentary:** This level of honesty is unusual for a cryptocurrency project announcement and likely contributed to the thread's relatively civil tone. The acknowledged weaknesses align with Gutmann's Scenario D: engineering reliability is the immediate threat, not quantum computing.

---

## 5. Questions for Peer AI Review

1. **Is the jrzx reply technically accurate?** Specifically: Is DAG-Knight actually a partial order? Does the "blocks at the same DAG depth are concurrent" framing correctly describe the protocol?

2. **Are the scaling numbers honest?** 31-138 GB for 11.8M blocks across 34 days. Does this extrapolate reasonably to global scale?

3. **Is the HNSL (Harvest Now, Steal Later) argument sound?** Does blockchain immutability truly fail to protect future spendability? Is the "lost wallets can't migrate" point valid?

4. **Is the "factoring 64-bit numbers" rebuttal correct?** Is ECC genuinely 2.6x easier to break quantumly than RSA? Is error correction scaling the right metric?

5. **Are the historical exploit examples accurate and relevant?** ECDSA nonce reuse (2013), Polynonce (2023), biased nonce lattice (2019), Biryukov Tor eclipse (2015), ETC/BTG 51% attacks (2018-2020).

6. **Does the crypto-agility architecture actually address Bear's diversity argument?** Can height-gated algorithm replacement practically work through multiple algorithm generations?

7. **Does Gutmann's "Scenario D" have a valid rebuttal?** Can PQC migration and engineering reliability be pursued simultaneously without one cannibalizing the other?

8. **Tone:** Is the reply appropriately respectful for the metzdowd list audience (senior cryptographers, cypherpunks)?

---

## 6. Document References

- Viktor's March 26 status report: https://www.metzdowd.com/pipermail/cryptography/2026-March/039440.html
- jrzx's scalability questions: https://www.metzdowd.com/pipermail/cryptography/2026-March/039442.html
- Draft reply to jrzx: `docs/metzdowd-reply-jrzx-scalability.md`
- Gutmann's "bollocks" slides: https://www.cs.auckland.ac.nz/~pgut001/pubs/bollocks.pdf
- Gosling's standardization phases (recommended by Gilmore): https://nighthacks.com/jag/StandardsPhases/StandardsPhases.html
- Biryukov & Pustogarov, "Bitcoin over Tor isn't a good idea": https://arxiv.org/abs/1410.6079
- Kudelski Polynonce attack: https://research.kudelskisecurity.com/2023/03/06/polynonce-a-tale-of-a-novel-ecdsa-attack-and-bitcoin-tears/
- Breitner & Nadia, biased nonce lattice attacks: https://eprint.iacr.org/2019/023.pdf
