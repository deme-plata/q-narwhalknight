---
title: "Re: Quillon Graph — A private, post-quantum electronic cash system"
subtitle: "Follow-up to the January 2026 thread on the Metzdowd Cryptography Mailing List"
author: "Viktor S. Kristensen"
date: "March 2026"
geometry: "margin=1in"
fontsize: 11pt
documentclass: article
header-includes:
  - \usepackage{booktabs}
  - \usepackage{longtable}
  - \usepackage{hyperref}
  - \hypersetup{colorlinks=true, linkcolor=blue, urlcolor=blue, citecolor=blue}
  - \usepackage{fancyhdr}
  - \pagestyle{fancy}
  - \fancyhead[L]{\small Quillon Graph — Metzdowd Follow-up}
  - \fancyhead[R]{\small March 2026}
  - \fancyfoot[C]{\thepage}
  - \setlength{\parskip}{0.5em}
  - \setlength{\parindent}{0em}
---

\vspace{-1em}

**To:** cryptography@metzdowd.com \
**In-Reply-To:** January 2026 thread \
**Source:** <https://quillon.xyz> \
**Binary:** <https://quillon.xyz/downloads/q-api-server-linux-x86_64>

---

List,

This is a follow-up to the Quillon Graph thread from January. The network launched on February 22, 2026 and has been running continuously for 33 days. I owe this list an honest status report, corrections to claims I made in January, and responses to the criticisms raised --- particularly by Bear, Peter Gutmann, John Gilmore, and Peter Fairbrother.

# 1. Corrections to January Claims

Several things I stated or implied in January were wrong or misleading. I want to correct them before anything else.

## (a) The mining algorithm is BLAKE3 + VDF, not SHA-3.

The January posts referenced SHA-3 in several places. The actual proof-of-work implementation uses iterative BLAKE3 hashing:

```
h = BLAKE3(header || nonce)
for i in 0..T:  h = BLAKE3(h)
check: h < target
```

SHA-3-256 appears in the protocol layer as an algorithm identifier, in challenge-response hashing, and in a cfg-gated CPU fallback path within the hybrid mining library (compiled when `gpu-mining` feature is disabled). However, the primary PoW implementation that all miners and nodes actually execute --- `DagKnightVDF` in `q-miner` and `q-api-server` --- uses BLAKE3. I should have been precise about this in January.

## (b) The 50ms finality figure needs clarification.

Peter Fairbrother questioned this. There are two metrics:

- **User-visible confirmation: <50ms.** The SSE streaming system pushes transaction events to connected wallets within 50ms of node acceptance. Users see their balance update almost instantly. This is the number users experience.

- **DAG-Knight consensus finality: 1.4 seconds.** This is when the transaction is ordered in the DAG with sufficient probability of irreversibility ($\delta=1$ confirmation depth).

In January I conflated these. The 50ms figure is real and measurable --- but it is node-level acceptance and event delivery, not consensus finality. For a merchant accepting payment, 1.4 seconds is the honest finality number. For a user watching their wallet, sub-50ms is the honest UX number. Both are legitimate metrics; I should have distinguished them.

## (c) The TPS figures need context.

The January thread referenced 48,000 TPS. Independent benchmarking confirms the throughput is real --- the HTTP binary batch protocol peaks at 12,414 TPS (1000-tx batches, P99 latency 125ms, 100% success rate) and sustains 8,600--9,700 TPS across batch sizes from 100 to 10,000. Single-transaction PaaS API calls measure 1,273 TPS at 73ms P99.

However, throughput without finality context is misleading. The full picture across layers:

| Layer | TPS | Latency |
|:------|----:|--------:|
| HTTP binary batch (peak) | 12,414 | 125ms P99 |
| Optimistic finality (SSE) | --- | <100ms |
| AEGIS-256 affirmation | --- | 53--160ms |
| DAG-Knight 1-conf finality | --- | ~1.4s avg |
| Deep confirmation (3-block) | --- | ~4.3s avg |

The <100ms optimistic finality is real and measurable: when a sender signs a transaction, the receiver's wallet balance updates within 100ms via SSE push, backed by an AEGIS-256 authenticated affirmation certificate. This is not a UI trick --- the balance update is cryptographically attested --- but it precedes full DAG consensus ordering by ~1.3 seconds.

I should have presented all layers in January rather than citing only the peak throughput.

## (d) The codebase size requires clarification.

I cited 680,000+ lines in January, measured across an 83-crate workspace. The current figure for non-test, non-backup Rust application code is approximately 767,000 lines (555,000 excluding blanks and comments). Including the TypeScript frontend and test suites, the full project is approximately 944,000 lines. The January number was roughly accurate for its time; the codebase has grown since then.

## (e) SQIsign: real isogeny arithmetic now available via FFI.

Since the January thread, we completed the FFI integration path recommended in our own technical review. The official NIST Round 2 SQIsign C reference implementation (`SQISign/the-sqisign`, Apache-2.0) is now vendored in the codebase and callable from Rust via two new crates:

- **`q-sqisign-sys`** --- Raw FFI bindings to the C reference (75,545 lines of production C, 203 files). Compiles via CMake, links GMP. Exposes keygen, sign, verify, open.

- **`q-sqisign`** --- Safe Rust wrapper with `KeyPair::generate()`, `sign()`, `verify()`. Secret keys zeroized on drop. 11 tests including roundtrip, tampering, wrong-key rejection.

**What is now real (Level I, via C FFI):**

- Key generation produces actual isogeny-derived keypairs (pk=65 bytes j-invariant, sk=353 bytes isogeny coefficients)
- `sign()` computes the real dimension-4 isogeny push-through via KLPT + Deuring correspondence (~30ms on modern Intel)
- `verify()` checks that the isogeny diagram commutes (~1.5ms)
- All $\mathbb{F}_{p^2}$ arithmetic, Vélu formulas, quaternion algebra, and KLPT lattice reduction are implemented in the vendored C
- NIST KAT vectors included (900 test cases per security level)
- Detached signature size: 148 bytes (smaller than the 204-byte scaffold estimate)

**What still uses the hash-based scaffold:**

- Levels III and V (only Level I C reference is vendored)
- Builds without the `sqisign-ffi` feature flag (fallback path)

The integration is feature-gated: `cargo build --features sqisign-ffi` enables the real C implementation for Level I. The height-gated activation point (`PHASE2_SQISIGN_MANDATORY`, block 2,000,000) has already been passed --- the chain is currently above 11 million blocks.

Mainnet blocks are signed with SQIsign (Phase 2) by default. The chain transitioned through Ed25519 (Phase 0) $\to$ Dilithium5 (Phase 1, now deprecated) $\to$ SQIsign (Phase 2, 204-byte compact signatures, 95.6% smaller than Dilithium5's 4,627 bytes).

# 2. Addressing the Criticisms

## Bear's premature standardization argument (Jan 12)

> *"Any standard for quantum-resistant cryptography is a ludicrously premature standard made without any input from real-world systems or threats."*

Bear, you were right about the prescription problem. In January I argued that hybrid classical+PQ was the answer. Having now run the system on mainnet for a month, I can report what hybrid actually costs in practice:

- Dilithium5 signatures are 4,627 bytes vs Ed25519's 64 bytes. On a DAG with thousands of blocks per minute, this is significant storage overhead.
- Kyber1024 KEM adds ~3,200 bytes to each P2P handshake. With 80+ connected peers, this is measurable.
- The real cost is not space --- it is *attention*. Engineering time spent on PQ integration is time not spent on consensus correctness, storage reliability, and operational safety.

In our first month of mainnet, we had:

- A height regression bug that lost 3,000 blocks on restart
- A RocksDB memory leak that OOM-killed the bootstrap node
- A gossipsub flood that crashed the 10Gbit supernode 6x/hour
- An ephemeral port collision that caused `bind()` failures

None of these were cryptographic failures. All of them were engineering failures that threatened the network more immediately than any quantum computer. This is exactly Gutmann's Scenario D point: *"while the world is fixated on dealing with a threat that no-one has been able to prove exists, we're not addressing actual vulnerabilities."*

I still believe hybrid is correct for an immutable ledger. But Bear's point about premature standards diverting engineering resources from real threats is empirically validated by our first month of operation.

Your suggestion to maintain "a few hundred different attempted solutions" before standardizing has merit. The crypto-agility architecture (height-gated validation rules, algorithm field in block headers) was designed for exactly this: the PQ layer can be replaced without a chain reset if lattice assumptions fall or better constructions emerge.

## Peter Gutmann's Scenario D --- Desinformatsiya (Jan 11)

> *"While the world is fixated on dealing with a threat that no-one has been able to prove exists, we're not addressing actual vulnerabilities."*

Peter, this hit hard. Our operational experience confirms it.

The three most dangerous bugs we encountered in mainnet's first month were:

1. **Sync-down:** a peer announcing a lower height could cause the node to delete blocks down to the peer's height, destroying the chain. This is a trivial implementation bug with catastrophic consequences --- orders of magnitude more dangerous than any quantum attack.

2. **Unbounded block-pack allocation:** syncing peers requesting 500+ blocks spawned unbounded tokio tasks, each allocating 50--150MB. Four concurrent sync requests = 10GB allocation = OOM. We gated this with a semaphore (max 4 concurrent, max 200 blocks per response, ~200MB worst case).

3. **RocksDB auto-configuring block cache** to 1/3 of RAM (16GB on our main node), leaving insufficient memory for the application. Fixed by explicit cap: `ROCKSDB_BLOCK_CACHE_MB=4096`.

These are pedestrian engineering problems. They are also the ones that actually threatened user funds. Your "bollocks" framing has a real engineering corollary: the threat model that dominates *in practice* is implementation bugs, not algorithmic breaks.

That said, I maintain the immutable ledger asymmetry argument. These engineering bugs are fixable (and were fixed, within hours). A classical cryptographic break of the signature scheme, if it occurs after keys are exposed on-chain, is *not fixable* --- those funds are permanently stealable. The distinction between "recoverable engineering failure" and "irrecoverable cryptographic failure" still favors defense-in-depth.

## John Gilmore's trust deficit (Jan 10)

> *"Why is NSA pushing belt-without-suspenders quantum-resistant crypto? [...] You could bet that they've changed their spots, but it's a sucker bet."*

John, I cannot distinguish your Scenarios A/B/C from the outside. The implementation responds as follows:

- **No NIST-blessed RNG.** We use `getrandom` (OS entropy) and optionally hardware TRNG, not NIST SP 800-90A.
- **Parameter transparency.** Dilithium5 and Kyber1024 use the published NIST parameters, but the protocol is designed for algorithm replacement. If someone produces a convincing cryptanalytic result against ML-DSA/ML-KEM, we can activate a replacement at a future block height without chain reset.
- **Belt AND suspenders.** Ed25519 remains as the Phase 0 layer. Dilithium5 is Phase 1. Both signatures must verify for blocks in the hybrid phase. Breaking one is insufficient.

Whether NSA has broken lattice assumptions is unknowable. Whether hybrid classical+lattice is strictly harder to break than either alone is a theorem, not a bet.

## Peter Fairbrother's privacy question (Jan 7)

> *"DAG-based BFTs like Bullshark or Mysticeti produce a consensus blockchain quickly, but the blockchain isn't particularly private."*

This was the most technically important critique. The privacy does not come from the DAG consensus layer --- it comes from the transaction layer above it:

- **LSAG ring signatures** [1] over Ristretto points hide the sender within a ring of decoy inputs. Key images prevent double-spends without de-anonymization.
- **Stealth addresses** ensure each payment goes to a unique one-time address, preventing address clustering.
- **Bulletproofs++** [2] range proofs hide transaction amounts.
- **Dandelion++** [3] with embedded Tor (4 circuits per validator, independently rotated) prevents IP-to-transaction correlation.

The DAG-BFT layer orders opaque transactions. Validators verify validity proofs without learning transaction contents.

Since v3.4.16-beta, privacy proofs are **mandatory --- not opt-in**. Every transaction submitted through the node automatically receives maximum privacy (Bulletproofs + STARK + LatticeGuard) before broadcast. Users do not choose a privacy level; the highest available level is applied by default.

The anonymity set is therefore the entire transaction set, not a subset of privacy-conscious users. This addresses the fundamental weakness of opt-in privacy systems (Zcash's shielded pool problem) where the anonymity set is limited to the small fraction of users who choose to use privacy features.

# 3. What Actually Happened on Mainnet

The network launched February 22, 2026 at 12:00 UTC. Empirical observations from 33 days of operation:

| Metric | Value |
|:-------|:------|
| Aggregate hashrate | ~7 GH/s (BLAKE3+VDF, CPU-only miners) |
| Block production | Continuous since genesis, no halts |
| Bootstrap nodes | 4 (geographically distributed) |
| Connected miners | 316+ (322 at last measurement) |
| Chain height | ~11.4M blocks |
| Finality | ~1.4s (1-confirmation) |
| Reorgs | 0 (DAG structure absorbs concurrent blocks) |
| Consensus failures | 0 |
| Unplanned outages | 3 (all engineering bugs, all recovered) |

The 7 GH/s figure is notable only because it represents organic adoption from 316+ CPU miners without exchange listings, mining pools, or marketing. For context, established CPU-mineable currencies took years to reach comparable hashrates (Monero launched April 2014, reached ~7 GH/s around 2019--2020 with RandomX).

The comparison is imperfect --- 2026 has mature mining infrastructure that 2014 did not. But the speed of adoption suggests the BLAKE3+VDF algorithm and economic parameters (2,625,000 QUG/year, 21M max supply, 4-year halving) are attractive to existing CPU miners.

# 4. BLAKE3+VDF Mining Details

Since this was mis-stated in January, the precise algorithm:

```
Input = prev_block_hash || miner_pubkey || nonce || timestamp
h = BLAKE3(Input)
for i in 0..T:
    h = BLAKE3(h)          // Sequential -- cannot parallelize
valid = (h < difficulty_target)
```

BLAKE3 was chosen over SHA-3 for:

- 3--5x faster on x86-64 with AVX2/AVX-512 SIMD
- Excellent ARM NEON performance (future target)
- Cache-friendly: the iterated VDF loop fits in L1

The VDF loop ($T$ sequential iterations) provides ASIC resistance. Custom hardware can optimize the BLAKE3 compression function by perhaps 2--3x over a modern CPU, but cannot skip iterations. This bounds the ASIC advantage to a small constant factor, unlike SHA-256d where ASICs achieve >10,000x over CPUs.

GPU mining is supported (OpenCL kernel for BLAKE3), but the sequential VDF loop limits effective GPU parallelism. Each GPU thread runs an independent VDF chain --- more threads help, but cannot accelerate any single chain.

# 5. Architectural Overview

The system is a Rust workspace of ~80 crates. The consensus-critical components (listing only the relevant subset):

| Crate | Purpose |
|:------|:--------|
| `q-dag-knight` | DAG-BFT ordering (Sompolinsky et al. [4]) |
| `q-vdf` | Wesolowski [5] + Pietrzak [6] VDFs |
| `q-types` | Block/tx types, signature verification |
| `q-storage` | RocksDB blockchain storage |
| `q-network` | libp2p gossipsub + Kademlia DHT |

Post-quantum and privacy:

| Crate | Purpose |
|:------|:--------|
| `q-quantum-crypto` | Dilithium5 + Kyber1024 |
| `q-quantum-mixing` | LSAG ring sigs, stealth addrs, Bulletproofs++ |
| `q-zk-snark` | Groth16/PLONK/Marlin (arkworks [7]) |
| `q-zk-stark` | Custom AIR/FRI STARK prover |
| `q-dandelion` | Dandelion++ relay with Tor bridge |
| `q-tor-client` | Embedded arti Tor client |
| `q-tor-circuit` | 4 dedicated circuits per validator |

The signature scheme progression:

| Phase | Scheme | Signature Size | Status |
|:------|:-------|---------------:|:-------|
| Phase 0 | Ed25519 | 64 bytes | Genesis through early chain |
| Phase 1 | Dilithium5 | 4,627 bytes | Deprecated |
| Phase 2 | SQIsign | 204 bytes | Current default (via C FFI) |

Phase transitions are height-gated: blocks below the activation height validate under old rules; blocks above under new rules. Old blocks never need re-validation. This is how we maintain crypto-agility without chain resets --- the answer to Bear's "what happens when the standard is wrong."

# 6. Known Limitations and Open Problems

Honest assessment of where the system falls short:

**(a)** The privacy layer uses curve25519-based ring signatures, which are quantum-vulnerable. Migrating ring signatures to lattice-based constructions is an open research problem. Current candidates (e.g., Esgin et al. [8]) produce signatures that are orders of magnitude larger. This is the single largest technical debt in the system.

**(b)** No independent audit of any cryptographic implementation. The Dilithium5 integration uses `pqcrypto-dilithium` (which has had some review), but our ring signature, Bulletproofs++, and STARK implementations are custom and unaudited. This is a serious limitation.

**(c)** The compact signature scheme (Phase 2) wraps the official NIST Round 2 SQIsign C reference via FFI for Level I. Levels III and V still use a hash-based scaffold. No pure-Rust SQIsign library exists as of March 2026. Side-channel hardening of the C signing path (KLPT, quaternion lattice reduction) remains an open problem tracked upstream.

**(d)** VDF security parameters have not been formally analyzed. The RSA group size and iteration count were empirically tuned. A formal analysis relating these to concrete security levels is needed.

**(e)** The DAG-Knight consensus has not been formally verified. We test extensively (4,000+ tests including adversarial scenarios), but testing is not proof.

**(f)** The Tor integration has not been evaluated against a global passive adversary. The 4-circuit architecture provides defense-in-depth but has not been analyzed for traffic correlation resistance under realistic threat models.

# 7. What We Learned from This List

The January thread was the most useful technical review the project has received. Specific changes made in response:

- Clarified finality claims (50ms UX vs 1.4s consensus)
- Separated HNDL and HNFL threat models in documentation
- Completed SQIsign FFI integration (NIST Round 2 C reference, Level I real isogeny arithmetic, 148-byte signatures)
- Prioritized engineering reliability over feature additions (Gutmann's Scenario D was persuasive)
- Documented the hybrid architecture as "belt AND suspenders" not "belt instead of suspenders" (Gilmore's framing)
- Added algorithm replacement infrastructure (Bear's diversity argument) via height-gated crypto-agility

The network is stronger for this discussion. I welcome continued criticism.

# 8. The Empirical Case for the Quantum Threat

In Section 2 I conceded --- honestly --- that Gutmann's Scenario D has engineering merit. Our operational failures were all classical bugs, not cryptographic breaks. But conceding that engineering bugs dominate *today* does not concede that cryptographic breaks are fiction *tomorrow*. The distinction matters for immutable systems, and the evidence has moved significantly since Gutmann's "Bollocks" presentation in August 2024.

I want to lay out the empirical case. Not speculation. Not vendor roadmaps. Published results, government mandates, and formal risk frameworks.

## (a) Shor's algorithm is a theorem, not a conjecture.

The mathematical result is settled: a sufficiently large, fault-tolerant quantum computer running Shor's algorithm solves the elliptic curve discrete logarithm problem in polynomial time. This breaks Ed25519, ECDSA (secp256k1, P-256), and every deployed elliptic curve signature scheme.

The debate is exclusively about hardware timelines. Not "if" --- "when."

## (b) The hardware trajectory is no longer speculative.

Gutmann's central empirical claim is that quantum computers "haven't managed to factor any number greater than 21 without cheating." This was a fair observation in 2024. But factoring small integers is not the relevant metric for assessing the trajectory toward cryptographically relevant quantum computers (CRQCs). The relevant metric is error correction scaling. Recent milestones:

**December 2024 --- Google Willow.** 105 superconducting qubits. First demonstration of exponential error suppression with increasing surface code size ($3\times3 \to 5\times5 \to 7\times7$, each step reducing encoded error rate by 2.14x). Logical memory exceeded physical qubit lifetime by 2.4x --- the "beyond breakeven" threshold. Published in *Nature* [9].

**February 2025 --- Microsoft Majorana 1.** First quantum processor built on topological qubits (topoconductor material). Designed to scale to one million qubits on a single chip. If hardware-protected qubits work as designed, the physical-to-logical qubit ratio drops from ~1000:1 toward ~10:1 [10].

**March 2025 --- USTC Zuchongzhi 3.0.** 105-qubit superconducting processor demonstrating quantum computational advantage. Shanghai University factored a 90-bit RSA integer via quantum annealing --- the largest quantum-assisted factorization to date [11].

**2025 --- Oxford Ionics.** Achieved 99.99% two-qubit gate fidelity in trapped-ion hardware --- the quality threshold required for fault-tolerant operation [12].

**2025 --- IBM roadmap.** Kookaburra processor (logical qubits) targeted for 2026. Starling processor (200 logical qubits from ~10,000 physical) targeted for 2028. IBM expects verified quantum advantage confirmed by end of 2026 [13].

Google crossed the error correction scaling threshold. Microsoft introduced a fundamentally new qubit architecture. China is investing \$10--15 billion (vs US \$1B, EU \$1B). These are not press releases about future intent --- they are published results about hardware that exists today.

## (c) ECC is the easiest quantum target.

Breaking P-256 requires 2.6x fewer logical qubits and 148x fewer quantum gates than breaking RSA-3072 at equivalent classical security. Estimates for breaking a 256-bit elliptic curve:

| Metric | Range |
|:-------|:------|
| Logical qubits | 523 -- 2,619 |
| Physical qubits | 10,000 -- 1,000,000 |

Kim et al. (2026) demonstrated a 40% improvement in qubit-count-depth product over prior constructions, with P-224 breakable using 6.9--19.1 million physical qubits in 34--96 minutes [14].

In August 2025, researchers published "Brace for Impact" --- a graded ECDLP challenge ladder from 6-bit to full 256-bit secp256k1, specifically designed to benchmark Shor's algorithm progress against Bitcoin's curve [15]. The existence of a measurement framework for this threat is itself evidence of the research community's assessment of its proximity.

The plausible break-in window for 256-bit ECC is now estimated at 2027--2033 under optimistic assumptions [16].

## (d) The governments agree --- and they have better intelligence.

Gutmann argues that the quantum threat is unproven. The agencies with the most complete intelligence on quantum computing capabilities --- including classified programs --- have concluded otherwise:

| Agency | Action |
|:-------|:-------|
| NSA (CNSA 2.0) | PQC deployment mandatory for new classified systems by 2027. Full transition by 2035. |
| NIST | Finalized first three PQC standards August 2024. Published CSWP 39 (December 2025) on crypto-agility. |
| CISA | Joint advisory with NSA/NIST: adversaries are "already harvesting encrypted data" for future decryption. |
| Federal Reserve | Published formal economic analysis of HNDL risks across financial sectors (2025). |

The NSA mandating PQC migration by 2027 for new classified systems is the strongest available signal. This is not an organization that wastes resources on imaginary threats, nor one that publicly mandates expensive infrastructure changes without classified evidence supporting the timeline [17][18][19].

## (e) Mosca's theorem and the blockchain special case.

Michele Mosca's risk framework [20] formalizes the migration urgency as an inequality:

$$X + Y > Q \implies \text{migration is already overdue}$$

Where:

- $X$ = time the data must remain secure
- $Y$ = time required to migrate cryptographic systems
- $Q$ = time until quantum computers break current crypto

For a TLS session: $X$ = seconds to hours. For a classified document: $X$ = 25--75 years. For an immutable blockchain:

$$X = \infty$$

When $X = \infty$, the inequality $X + Y > Q$ is satisfied for *any* finite $Q$. The migration is overdue by definition, regardless of whether $Q$ is 5 years or 50.

This is not a rhetorical trick. It is the mathematical consequence of publishing cryptographic commitments to a permanent, append-only ledger. TLS sessions expire. Classified documents get declassified. Blockchain transactions are forever.

Mosca himself estimates a 1-in-7 chance of fundamental public-key crypto being broken by quantum by 2026, and 1-in-2 by 2031 [20].

## (f) HNDL, HNFL, and the blockchain-specific HNSL.

The standard "Harvest Now, Decrypt Later" framework --- recently formalized as a temporal cybersecurity risk model by Pavlidis et al. [21] --- addresses encrypted communications. But blockchains face a more specific threat that I described in the January thread as **HNSL: Harvest Now, Steal Later**.

The attack is straightforward:

1. Alice sends a transaction at time $t_0$. Her Ed25519 public key is permanently published on-chain.
2. At time $t_1$, a CRQC solves the ECDLP for Alice's public key, recovering her private key.
3. The adversary forges new transactions from Alice's address, stealing any residual funds.

This is distinct from HNDL (which targets ciphertext) and HNFL (which targets signature trust). **HNSL targets funds.** The adversary does not need stored ciphertext or forged documents --- they need only the public keys that are already published, today, on every blockchain that uses elliptic curve signatures.

For Bitcoin alone, approximately 5--10 million addresses have exposed public keys (via P2PKH spend scripts). For Ethereum, all addresses have exposed public keys (every transaction reveals it). These keys cannot be un-published. They are permanent targets.

## (g) The asymmetry argument --- why both threats demand attention.

Gutmann is correct: engineering bugs are the dominant threat today. Our operational data proves it. But the failure modes are fundamentally different:

| Failure Class | Recoverability | Example |
|:--------------|:---------------|:--------|
| Engineering bug | **Recoverable.** We fixed sync-down in hours. We fixed OOM in hours. The network continued. | Height regression, memory leak, port collision |
| Cryptographic break | **Irrecoverable.** Every public key ever published on every blockchain becomes a permanent target. No patch. No rollback. No fork can un-expose keys. | Shor's algorithm solving ECDLP |

This asymmetry --- recoverable vs irrecoverable failure --- is why defense-in-depth is not "disinformation" or wasted effort. It is the rational response to a low-probability, catastrophic, irreversible event. The same engineering discipline that buys insurance, builds redundant systems, and wears seatbelts.

We should fix the engineering bugs AND prepare for the cryptographic break. These are not competing priorities --- they are complementary layers of the same risk management framework.

## (h) What the temporal security framework says.

The formal treatment is in our paper [22], but the core argument is compact:

For **updatable systems** (TLS, Signal, SSH), the window for defensive action extends indefinitely. When a new threat emerges, you rotate keys, update protocols, and move on. The threat and the response exist in the same time frame.

For **immutable systems** (blockchains, signed legal documents, timestamped archives), the window for defensive action is *before the data is recorded*. Data recorded in 2026 with 2026 cryptography cannot be updated in 2040 when the threat materializes. The cryptographic choice is permanent.

This temporal asymmetry is not a theoretical concern. It is the defining property of an append-only ledger. And it means that for blockchain systems specifically, the question is not "is the quantum threat real today?" but "will it be real during the lifetime of data we are recording today?" For an immutable ledger, that lifetime is forever.

NIST's CSWP 39 [18] validates the crypto-agility approach --- the ability to replace algorithms without replacing the system. Our height-gated activation mechanism is a direct implementation of this principle: Phase 0 (Ed25519) $\to$ Phase 1 (Dilithium5) $\to$ Phase 2 (SQIsign), each activated at a block height, each preserving validation of all historical blocks.

The answer to "what happens when the standard is wrong" (Bear's concern) is: activate a new algorithm at a future height. No chain reset. No lost history. The old algorithm validates old blocks. The new algorithm validates new blocks. This is crypto-agility in practice, not theory.

## (i) Quantum physics as a constructive security tool --- not just a threat.

Gutmann frames quantum computing exclusively as a speculative threat that diverts resources from real problems. But quantum physics is not only a future adversary --- it is a present-day tool for improving security. Our Tor integration demonstrates this concretely.

The anonymity layer uses **quantum random number generation (QRNG)** from hardware entropy sources to seed every privacy-critical operation. This is not PQC (defending against quantum attacks). This is using quantum physics offensively to improve classical security today:

**Circuit entropy.** Each of the 4 dedicated Tor circuits per validator is seeded with 32 bytes of entropy from hardware QRNG (photonic shot noise or thermal quantum noise sources). The seed initializes a ChaCha20 PRNG that generates all circuit parameters --- circuit IDs, nonces, hop selection weights. True quantum randomness eliminates the predictability that algorithmic PRNGs can never fully escape. The circuits rotate every 4--6 minutes with fresh quantum entropy, with the shortest rotation interval (240 seconds) assigned to the quantum beacon circuit that distributes entropy to peers.

**Timing obfuscation in Dandelion++.** The Dandelion++ stem phase [3] adds random delays before relaying transactions to prevent timing correlation. These delays (100ms--1500ms) are drawn from a quantum-seeded PRNG. An adversary performing traffic analysis must contend with delays whose source is genuinely non-deterministic --- not pseudo-random with a discoverable seed, but random in the physical sense. The stem$\to$fluff transition probability (0.15) is also quantum-derived.

**Relay selection.** Which peer receives a relayed transaction during the stem phase is selected using quantum randomness. This prevents an adversary who has compromised the PRNG state from predicting relay paths.

**Entropy quality monitoring.** The system continuously validates QRNG output using three statistical tests: Shannon entropy (randomness), chi-squared (uniformity), and runs tests (independence). If entropy quality drops below 95%, the system falls back to a secondary QRNG source. If both fail, it degrades to OS entropy (`getrandom`) with a logged warning. The quality threshold is enforced every 60 seconds.

**Multi-source entropy mixing.** The quantum entropy pool aggregates from multiple physical sources --- quantum hardware (0.95 quality score), OS entropy (0.90), atmospheric noise (0.70), and CPU timing jitter (0.60) --- mixing them via XOR into the ChaCha20 state. This defense-in-depth means that compromising any single entropy source is insufficient.

This architecture is informed by recent research on post-quantum onion routing. Ghosh and Kate [23] demonstrated that hybrid post-quantum key exchange for Tor circuits is both feasible and more efficient than the current `ntor` protocol (33% computational improvement). Pavlidis et al. [24] proposed integrating quantum key distribution directly into onion routing relay protocols. Our implementation takes a pragmatic middle path: we use QRNG for entropy (available today with commodity hardware) and lattice-based KEMs (Kyber-768) for circuit handshakes (available today in software), rather than waiting for QKD infrastructure that does not yet exist at scale.

The point for this list: quantum physics is not a binary --- either a threat to defend against or a distraction to ignore. It is simultaneously a near-term tool for improving privacy (QRNG), a medium-term infrastructure upgrade (PQC for circuit handshakes), and a long-term threat to classical cryptography (Shor's algorithm). Treating it as only one of these mischaracterizes the engineering landscape.

---

# References

\small

[1] J. Liu, V. Wei, D. Wong, "Linkable Spontaneous Anonymous Group Signature for Ad Hoc Groups," ACISP 2004, LNCS 3108.

[2] L. Eagen, D. Fiore, A. Gabizon, "Bulletproofs++: Next Generation Confidential Transactions via Reciprocal Set Membership Arguments," ePrint 2022/510.

[3] G. Fanti et al., "Dandelion++: Lightweight Cryptocurrency Networking with Formal Anonymity Guarantees," ACM SIGMETRICS 2018.

[4] Y. Sompolinsky, S. Wyborski, A. Zohar, "DAG-Knight: An Asynchronous Byzantine Fault Tolerant Consensus Protocol," ePrint 2022/1494.

[5] B. Wesolowski, "Efficient Verifiable Delay Functions," EUROCRYPT 2019, LNCS 11478.

[6] K. Pietrzak, "Simple Verifiable Delay Functions," ITCS 2019, LIPIcs 124.

[7] arkworks contributors, "arkworks: An Ecosystem for zkSNARKs," <https://arkworks.rs>

[8] M.F. Esgin, R. Steinfeld, D. Sakzad, J.K. Liu, D. Liu, "Short Lattice-based One-out-of-Many Proofs and Applications to Ring Signatures," ACNS 2019, LNCS 11464.

[9] Google Quantum AI, "Quantum error correction below the surface code threshold," *Nature* 638, 920--926 (2025). <https://www.nature.com/articles/s41586-024-08449-y>

[10] Microsoft Azure Quantum, "Majorana 1: The world's first quantum processor powered by topological qubits," Feb 2025. <https://azure.microsoft.com/en-us/blog/quantum/2025/02/19/microsoft-unveils-majorana-1>

[11] USTC, Zuchongzhi 3.0 (105-qubit superconducting processor). Shanghai University RSA-90 quantum annealing factorization.

[12] Oxford Ionics, 99.99% two-qubit gate fidelity (2025).

[13] IBM, "New Quantum Processors, Software, and Algorithm Breakthroughs," Nov 2025. <https://newsroom.ibm.com/2025-11-12>

[14] Kim et al., "Quantum Resource Requirements for Breaking Elliptic Curve Cryptography," Preprints.org 2025. <https://www.preprints.org/manuscript/202509.2429>

[15] M. Allende Lopez et al., "Brace for Impact: ECDLP Challenges for Quantum Cryptanalysis," arXiv:2508.14011.

[16] Cambridge Judge Business School, "Why quantum matters now for blockchain," 2025. <https://www.jbs.cam.ac.uk/2025/why-quantum-matters-now-for-blockchain/>

[17] NSA, "Commercial National Security Algorithm Suite 2.0" (CNSA 2.0). CISA/NSA/NIST joint advisory on HNDL threats.

[18] NIST CSWP 39, "Considerations for Achieving Cryptographic Agility: Strategies and Practices," Dec 2025. <https://nvlpubs.nist.gov/nistpubs/CSWP/NIST.CSWP.39.pdf>

[19] Federal Reserve, "Harvest Now Decrypt Later: Examining Post-Quantum Cryptographic Risks," FEDS 2025-093. <https://www.federalreserve.gov/econres/feds/files/2025093pap.pdf>

[20] M. Mosca, "Cybersecurity in an Era with Quantum Computers: Will We Be Ready?" IEEE Security & Privacy 16(5), 2018.

[21] S. Pavlidis et al., "Harvest-Now, Decrypt-Later: A Temporal Cybersecurity Risk in the Quantum Transition," MDPI Computers 2024, 6(4), 100. <https://www.mdpi.com/2673-4001/6/4/100>

[22] V. Kristensen, "Temporal Dimensions of Cryptographic Security: Decision Theory for Post-Quantum Infrastructure." <https://quillon.xyz/downloads/temporal-cryptographic-security-v2.pdf>

[23] R. Ghosh, A. Kate, "Post-Quantum Forward-Secure Onion Routing (Future Anonymity in Today's Budget)," ACNS 2016, LNCS 9696. <https://eprint.iacr.org/2015/008.pdf>

[24] M. Pavlidis et al., "Onion Routing Key Distribution for Quantum Key Distribution Networks," arXiv:2502.06657 (2025). <https://arxiv.org/pdf/2502.06657>

[25] S. Baumann et al., "Post Quantum Migration of Tor," ePrint 2025/479. <https://eprint.iacr.org/2025/479.pdf>
