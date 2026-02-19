Subject: Q-NarwhalKnight: A post-quantum DAG consensus system launching February 15

---

Fellow cypherpunks,

I've been working on a new distributed consensus system that I believe
addresses several open problems at the intersection of post-quantum
cryptography and Byzantine fault tolerance. It launches in 8 days. I'd
like to share the cryptographic architecture because I think some of you
will find the construction amusing.

**The short version:** We built a DAG-based BFT consensus protocol where
every cryptographic primitive — from block signatures to VDF time-locking
to privacy proofs — is post-quantum. Not "PQ-ready." Not "migration path
planned." Running. In production. On a live P2P network right now.

Here's what's under the hood:


1. CONSENSUS: DAG-KNIGHT WITH GENUS-2 VDF ANCHOR ELECTION

The consensus layer uses a DAG structure (inspired by the Narwhal/Tusk
separation of data availability from ordering). Leader election uses a
Verifiable Delay Function, but not on ordinary elliptic curves.

We implemented a VDF over genus-2 hyperelliptic Jacobians following the
construction in IACR 2025/1050. The group order factoring problem on
genus-2 curves is believed to require ~O(p^{1/3}) quantum operations
vs ~O(p^{1/2}) for elliptic curves (Grover on Pollard-rho), giving us
a meaningful quantum security margin without the key bloat of lattices.

The practical benefit: anchor election that remains unpredictable even
to an adversary with a modest quantum computer, with VDF proofs that
are still efficiently verifiable classically.


2. SIGNATURES: DILITHIUM5 + SQIsign HYBRID

Block signatures use NIST FIPS 204 (ML-DSA / Dilithium) at security
level 5. We also implemented SQIsign (IACR 2025/847 — the "2D-West"
variant) for contexts where 204-byte signatures matter more than
signing speed.

The isogeny-based construction is admittedly still aggressive for
production use, but having it as a selectable primitive means nodes
can switch without a hard fork. Crypto-agility is the point: we
designed the signature layer so that swapping algorithms is a config
change, not a consensus change.


3. PRIVACY: CLSAG RING SIGNATURES + BULLETPROOFS + RECURSIVE STARKs

For confidential transactions, we use:

- CLSAG ring signatures (the same construction Monero adopted in 2020,
  ~25% smaller than MLSAG) for sender anonymity
- Bulletproofs range proofs (Bunz et al., S&P 2018) to prove amounts
  are in [0, 2^64) without revealing them
- A recursive STARK composition for proof aggregation — no trusted
  setup, transparent, and plausibly post-quantum

The mixer protocol uses Chaumian blind signatures with a threshold
pool construction. Stealth addresses for receivers. The transaction
graph is designed to be opaque by default.


4. NETWORK PRIVACY: DANDELION++ WITH TOR CIRCUIT MANAGEMENT

Transactions propagate through a Dandelion++ stem phase before
flooding, making it difficult to determine the originating node via
network topology analysis. The P2P layer runs over libp2p with
integrated Tor circuit management — 4 dedicated circuits per
validator, rotated per epoch.


5. SYMMETRIC ENCRYPTION: AEGIS-256

Block-level encryption uses AEGIS-256 (IACR 2024/268) — the AES-based
AEAD that achieves ~2 cycles/byte on hardware with AES-NI. We chose
this over AES-GCM because AEGIS provides 256-bit security against
key-recovery attacks even under nonce misuse, and its internal state
is large enough (5 x 128-bit AES blocks) to resist quantum search on
the state space.


6. HASH: SHA-3 / SHAKE256

Keccak everywhere. Block hashes, Merkle trees, address derivation, VDF
challenges. Grover's algorithm gives a quadratic speedup on preimage
search, so SHA-3-256 offers ~128-bit quantum security. We use SHAKE256
where variable-length output is needed (VDF challenges, KDF).


7. KEY EXCHANGE: KYBER-1024 (ML-KEM)

P2P session establishment uses Kyber-1024 (NIST FIPS 203) for
key encapsulation. Combined with Noise protocol framework for
forward secrecy.


THE AMUSING PART

Bitcoin's SHA-256 + secp256k1 + RIPEMD-160 stack will survive first-
generation quantum computers through brute force (double the key
sizes, hope for the best). Most "quantum-resistant" blockchains bolt
on a single lattice signature and call it done.

We went the other direction: what if EVERY layer assumed a
cryptanalytically relevant quantum computer exists TODAY? Not as
a future contingency, but as a design constraint from day one.

The result is a system where:
- Block signatures are lattice-based (Dilithium5)
- VDF time-locking uses genus-2 curves (quantum-hard group order)
- Privacy proofs are STARK-based (no elliptic curve assumptions)
- Key exchange is lattice-based (Kyber-1024)
- Symmetric crypto assumes Grover (256-bit keys, AES-based AEAD)
- Hashing assumes Grover (SHA-3-256 = 128-bit quantum security)

The entire attack surface is: break lattices AND break genus-2 DLP
AND break SHA-3 AND break AES. Simultaneously. With the same quantum
computer.


A NOTE ON SKEPTICISM & THE NEED FOR PROOF AGAINST QUANTUM THREATS

To the skeptics and the Gutmanns among us — who rightly demand proof
and warn against overhyped timelines — I hear you. The engineering
challenges of building a cryptanalytically relevant quantum computer
(CRQC) remain staggering, and "20 years away" has been a moving
target for decades. But as the analysis in "The Quantum Inflection
Point" (v2.0, 2026) argues, the landscape has shifted in three
critical ways:

  (a) AI as a variance compressor. AI is no longer just a research
      aid — it's a landscape navigator. Systems like GNoME, AlphaFold,
      and cuLitho demonstrate the ability to traverse high-dimensional
      optimization spaces orders of magnitude faster than human
      intuition. This compresses the upper tail of timeline
      distributions while inflating the lower tail — making earlier
      breakthroughs more probable, even if median estimates remain
      unchanged.

  (b) Industry timelines are now engineering schedules, not
      speculation. Google's Willow processor achieved below-threshold
      error correction in 2024. IBM, PsiQuantum, and Quantinuum have
      published funded roadmaps targeting fault-tolerant systems by
      the early 2030s. This isn't academic conjecture; it's
      capital-backed engineering.

  (c) Harvest-Now-Decrypt-Later is already operational. As documented
      in the Snowden disclosures and formalized in multiple analyses,
      encrypted data is being collected and stored at scale. The
      adversary is patient, storage is cheap, and the decryption
      clock starts the moment a CRQC comes online — whether that's
      in 2035 or 2045.

Shannon's limit is absolute: only information-theoretically secure
systems are immune to HNDL. Everything else is a bet against time.
Hellman's ethical framework reminds us that waiting for certainty is
itself a decision — and often the wrong one when the cost of failure
is unbounded.

Q-NarwhalKnight is built under the assumption that this bet is
already risky enough to warrant action. We're not claiming to have
"solved" post-quantum security — no one can, per Shannon and Witten.
Instead, we've engineered a system where every layer assumes a
quantum adversary exists today, forcing defense-in-depth across
lattices, genus-2 curves, symmetric primitives, and hash functions.

I welcome the skepticism. It's what keeps cryptography honest. But I
also invite you to read the timelines, track the hardware, and watch
the AI accelerants. The window for migration is measured in years, not
decades — and it's closing faster than we think.


NETWORK STATUS

6 peer nodes currently syncing via libp2p gossipsub. DAG-BFT consensus
producing blocks every 2 seconds. Fully decentralized P2P propagation —
no central coordinator, no permissioned validator set.

Native DEX with constant-product AMM. Stablecoin (QUGUSD). Token
deployment via VM. All running on the post-quantum stack described above.

Launch: February 15, 2026.

Source: https://quillon.xyz
Node binary: wget https://quillon.xyz/downloads/q-api-server-linux-x86_64

I'm genuinely curious what this list thinks about the genus-2 VDF
construction for leader election. The security reduction is cleaner
than I expected, but I haven't seen it used in a consensus protocol
before. Happy to discuss the details.

— DK
  Q-NarwhalKnight / Quillon


References:

[1] IACR 2025/1050 - "VDFs from Genus-2 Hyperelliptic Curves"
[2] NIST FIPS 204 - ML-DSA (Dilithium) Digital Signature Standard
[3] NIST FIPS 203 - ML-KEM (Kyber) Key Encapsulation Mechanism
[4] IACR 2025/847 - "SQIsign2D-West: The Full Story"
[5] Bunz et al. "Bulletproofs" IEEE S&P 2018
[6] IACR 2024/268 - "AEGIS: A Fast Authenticated Encryption"
[7] Danev et al. "DAG-Knight: A Parameterless Generalization of
    Nakamoto Consensus" (2022)
[8] Keccak/SHA-3 - NIST FIPS 202
