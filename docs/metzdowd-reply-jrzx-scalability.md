From: Viktor S. Kristensen <viktor@quillon.org>
Subject: Re: [Cryptography] Quillon Graph: A private, post-quantum electronic cash system
To: cryptography@metzdowd.com

jrzx,

Your questions are good ones. I will address the scalability concerns
first, then the quantum argument.

## Ordering and Scale

> I don't see that total ordering can scale to the required or claimed
> size? Is it perhaps a partial order -- that all transactions in one
> group are later than the previous group and earlier than the next
> group?

Yes. You have identified the design correctly.

The system does not produce a total order over all transactions. It
produces a partial order over blocks (vertices in a directed acyclic
graph), with deterministic tie-breaking for concurrent vertices. Let me
describe how this works concretely, because the mechanism is not the
same as a linear blockchain.

**The DAG structure.** Each block references one or more parent blocks,
forming a DAG rather than a chain. Block A is ordered before block B if
A is an ancestor of B in the DAG --- that is, if there exists a directed
path from A to B through parent references. If neither A nor B is an
ancestor of the other, they are *concurrent*: the DAG's causal structure
does not impose an order between them. The set of blocks concurrent with
a given block is called its *anticone*.

**Ordering within a round.** Blocks are grouped into rounds. Each round
advances when the network has received certified vertices from a
quorum (2f+1 of 3f+1 validators). Within a round, concurrent blocks are
ordered deterministically by vertex ID hash --- a canonical tie-breaking
rule that all honest nodes compute identically. This gives a total order
within each round, and a partial order between rounds (round r blocks
precede round r+1 blocks, but blocks in the same round may have been
produced independently).

**Anchor election.** In even-numbered rounds, the protocol elects an
*anchor* --- a single vertex selected via a VDF (Verifiable Delay
Function) seeded with quantum-derived entropy. The anchor's causal
history determines which preceding vertices are committed. Specifically:
all vertices in the anchor's past cone that have not already been
committed are finalized in topological order when the anchor is elected.
This is the commit rule --- it produces a linearization of the partial
order at each anchor point.

The VDF serves two purposes: (1) it provides a verifiable proof of
sequential computation that cannot be parallelized, bounding the ASIC
advantage to a small constant (2-3x over a modern CPU); (2) it seeds
the anchor election with entropy from a quantum random number generator,
making the election unpredictable to any adversary who does not control
the QRNG source. The VDF uses iterative BLAKE3 hashing:

    h = BLAKE3(vertex_id || quantum_beacon || round)
    for i in 0..T:  h = BLAKE3(h)

The candidate with the minimal VDF output wins the anchor election ---
a deterministic, verifiable selection that all nodes can independently
verify.

**Reliable broadcast.** Before a block enters the DAG, it must survive
Bracha's reliable broadcast protocol [1]: the proposer sends the vertex
to all validators (SEND), who echo it (ECHO at 2f+1 threshold), then
amplify (READY at f+1 threshold), and finally deliver (at 2f+1 READY).
The Byzantine guarantee: if any honest node delivers a vertex, all
honest nodes deliver it. This eliminates equivocation --- a Byzantine
proposer cannot show different blocks to different subsets of the
network.

**Finality.** With delta=1 (aggressive configuration), the commit
decision for round r vertices occurs at round r+1. The time to
finality is approximately 1.4 seconds for consensus ordering. The
sub-50ms figure I cited in my January announcement refers to the SSE
event delivery latency --- the time between node acceptance and wallet
notification --- not consensus finality. I conflated these two metrics
in January, and corrected the error in my March 26 status update.

**Fork detection.** The system monitors DAG topology using an algebraic
approach: connected components (H_0) and independent cycles (H_1). A
healthy DAG has H_0 = 1 (one connected component) and H_1 = 0 (no
forks). If H_0 > 1, the network is partitioned. If H_1 > 0, a fork
exists. This is computed continuously via union-find over the block
graph.

So to answer your question directly: the ordering is a partial order
with deterministic linearization at anchor points. All transactions in
one group (the anchor's committed past cone) are ordered before all
transactions in the next group (the next anchor's committed past cone).
Within a group, the order is deterministic. Between groups, the order is
total. This is exactly the model you describe.

This is not a deficiency --- it is the design. Total ordering of every
transaction individually across a global network is fundamentally
incompatible with both high throughput and Byzantine fault tolerance.
DAG-BFT protocols accept the partial order as the correct trade-off:
you get consistency (all honest nodes agree on the committed order) and
availability (the system continues producing blocks under asynchrony),
at the cost of not totally ordering concurrent events within a single
round.

For double-spend prevention, this is sufficient: a transaction is valid
if no conflicting transaction appears in any committed block. The key
image mechanism (derived from LSAG ring signatures [2]) makes conflicts
detectable without knowing transaction contents.

## Bandwidth, Storage, and Processing

> How does the system's local peer storage and bandwidth scale under
> enormous numbers of transactions. How much network bandwidth,
> storage, and processing power does a full peer need when there are
> enormous numbers of transactions?

The current empirical numbers after 34 days of mainnet operation:

| Resource          | Current (11.8M blocks, 316 miners)      |
|:------------------|:----------------------------------------|
| Blockchain DB     | 31-138 GB (RocksDB, encrypted at rest)  |
| Block rate        | ~4 blocks/second                        |
| Bandwidth (P2P)   | ~2-4 MB/s steady state                  |
| RAM (full node)   | 6-13 GB (configurable RocksDB cache)    |
| CPU (full node)   | 2-4 cores sustained                     |

The DB size range reflects node configuration: 31 GB with aggressive
compaction on a secondary node, 138 GB on the primary bootstrap node
which retains full indexes, balance state history, and uncompacted
RocksDB SSTables.

At the current scale, a commodity VPS ($20-40/month) runs a full node.
This is comparable to running a Bitcoin full node or a pruned Ethereum
node.

> Since the system relies on concise proofs, in theory not everyone
> should need to keep everything around, not everyone needs to prove
> the validity of every transaction, and to do so would violate the
> privacy promises.

Correct. The privacy architecture enforces this structurally:

1. **Full nodes** store all blocks but cannot read transaction contents
   (Bulletproofs++ range proofs [3], LSAG ring signatures, stealth
   addresses). They verify validity proofs --- that inputs exist, that
   amounts balance, that no double-spend occurred --- without learning
   who sent what to whom.

2. **Light clients** (miners, wallets) do not store the full chain.
   They receive block headers, verify proof-of-work, and trust the
   full node network for transaction validity. SPV-equivalent security.

3. **Pruning.** Full nodes can prune spent transaction outputs. The
   validity proofs are self-contained --- once verified at block
   acceptance time, the proof data is not needed for future validation.
   Only unspent outputs and block headers are strictly required for
   ongoing consensus.

The honest answer to "how does this scale to replacing fiat globally" is:
it does not, today, and neither does any other system. At Visa's 65,000
TPS sustained, you would need datacenter-class bandwidth and storage ---
within reach of institutional infrastructure but not commodity hardware.

The scaling path is the same one every blockchain project faces:
sharding, state channels, rollups, or some combination. We have not
implemented any of these. The current system is a single-shard DAG-BFT
that handles thousands of TPS with commodity hardware. Scaling beyond
that requires architectural work that is not yet done.

I would rather state this honestly than claim the system already scales
to global fiat replacement. It does not. The consensus layer is designed
to be partitionable (the DAG structure naturally supports sharding by
subgraph), but this remains unimplemented and unproven.

## On the Quantum Argument

> Protected by blockchain immutability.

No. Blockchain immutability protects *historical records* --- it does
not protect *future spendability*. The distinction is critical.

When Alice sends a Bitcoin transaction, her public key is published
on-chain permanently. The historical record of that transaction is
indeed immutable. But Alice's unspent outputs are protected by the
*hardness of the ECDLP for that public key*. If a quantum computer
recovers Alice's private key from her published public key, the
adversary can forge new transactions spending Alice's remaining funds.
The historical record is untouched; Alice's money is stolen.

This is the HNSL (Harvest Now, Steal Later) attack I described in my
March 26 post. It does not require breaking immutability. It requires
only solving the discrete logarithm problem for public keys that are
already published and cannot be un-published.

> Spend authorizations of unspent transaction outputs will not suddenly
> become forgeable.

They will, for every address whose public key has been revealed. In
Bitcoin, this is every P2PKH address that has ever sent a transaction
(~5-10 million addresses). In Ethereum, this is every address that has
ever sent any transaction (all of them). The public keys are on-chain,
today, permanently.

> When there is a real threat that they might become forgeable in the
> next decade or so, *then* people will spend them to generate quantum
> resistant unspent transaction outputs.

This assumes:

(a) Users will be aware of the threat in time.
(b) Users will act before adversaries do.
(c) The migration can happen faster than the attack.
(d) Lost/abandoned wallets (estimated 3-4M BTC) can be migrated.

Assumption (d) alone invalidates the strategy. Coins in lost wallets
cannot be migrated. They become permanently stealable. The total value
at risk is in the hundreds of billions of dollars.

Mosca's inequality [4] applies: if the time to migrate (Y) plus the
time the data must remain secure (X) exceeds the time until quantum
computers break the crypto (Q), then migration is already overdue. For
an immutable ledger, X = infinity, so the inequality is satisfied for
any finite Q.

> Now is far too early to start generating quantum resistant unspent
> transaction outputs, for quantum computing remains purely theoretical,
> and quantum resistance is not merely theoretical but hypothetical,
> indistinguishable in practice from snake oil, grantsmanship, and
> cryptocurrency scamming.

The organizations with the most complete intelligence on quantum
computing capabilities --- including classified programs --- disagree
with this assessment. And they are not merely expressing concern. They
are mandating expensive migration on aggressive timelines.

**NSA CNSA 2.0 (September 2022, updated December 2024).** The NSA's
Commercial National Security Algorithm Suite 2.0 mandates post-quantum
cryptography for all National Security Systems on a staggered timeline:

| System category              | Must prefer PQC by | Exclusively PQC by |
|:-----------------------------|:-------------------|:-------------------|
| Software/firmware signing    | **2025**           | 2030               |
| Web browsers, servers, cloud | **2025**           | 2033               |
| Traditional networking (VPN) | **2026**           | 2030               |
| Operating systems            | **2027**           | 2033               |
| New NSS equipment purchases  | **2027 (Jan 1)**   | ---                |
| All National Security Systems| ---                | **2035**           |

Read those dates carefully. Software and firmware signing systems were
required to prefer PQC algorithms *last year*. New National Security
System equipment must be CNSA 2.0 compliant by January 1, 2027 ---
nine months from now. This is not a distant planning horizon. This is
operational reality for every defense contractor and classified system
vendor in the United States.

The mandated algorithms are ML-KEM-1024 (FIPS 203) for key
encapsulation and ML-DSA-87 (FIPS 204) for digital signatures --- the
same lattice-based constructions that have been studied for 30+ years,
with no known quantum speedup beyond Grover's square root.

**CISA/NSA/NIST Joint Advisory (August 2023).** The three agencies
jointly stated that "cyber threat actors could be targeting data today
that would still require protection in the future" and that adversaries
are assessed to be *already collecting* encrypted data for future
quantum decryption. Their recommendation: organizations must begin
"creating quantum-readiness roadmaps, conducting inventories, applying
risk assessments and analysis, and engaging vendors" --- immediately,
not when quantum computers arrive.

**White House NSM-10 (May 2022).** National Security Memorandum on
Promoting United States Leadership in Quantum Computing While Mitigating
Risks to Vulnerable Cryptographic Systems. Goal: mitigate as much
quantum risk as feasible by 2035. The ONCD projects $7.1 billion in
total government-wide migration cost between 2025 and 2035.

**UK NCSC (March 2025).** Three-phase roadmap: identify all
cryptographic services needing upgrades by 2028, execute high-priority
upgrades by 2031, complete full migration by 2035.

**European Union (June 2025).** Coordinated implementation roadmap:
all Member States should start transitioning by end of 2026, critical
infrastructure transitioned no later than end of 2030, full transition
by 2035.

**Federal Reserve (2025).** FEDS 2025-093, "Harvest Now Decrypt Later:
Examining Post-Quantum Cryptography and the Data Privacy Risks for
Distributed Ledger Networks." The U.S. central bank published a formal
economics discussion paper specifically warning that blockchain networks
face HNDL risk, and that "the immutability of distributed ledgers, a
feature celebrated for enhancing trust, is also their *greatest
weakness* against quantum threats. Because blockchains are designed to
preserve every transaction permanently, they inadvertently preserve
every vulnerability as well."

The logic chain is straightforward:

1. The NSA has better intelligence on the state of quantum computing
   --- including classified programs in the US, China, and elsewhere ---
   than any academic, any startup, or any mailing list participant.

2. The NSA explicitly acknowledges uncertainty: "NSA does not know when
   or even if a quantum computer of sufficient size and power to exploit
   public key cryptography will be developed."

3. Despite this uncertainty, the NSA is mandating a transition that will
   cost the U.S. government $7.1 billion, disrupt every National
   Security System, and impose compliance deadlines starting in 2025.

4. Every Five Eyes nation has aligned on the same 2035 deadline. The EU
   has published a converging roadmap. These are the same governments
   whose intelligence agencies are most likely to know the actual state
   of quantum computing programs worldwide.

5. None of these agencies are waiting for quantum computers to factor
   64-bit numbers. They are acting now because cryptographic transitions
   take 20+ years to complete, and by the time a cryptographically
   relevant quantum computer is *confirmed*, it will be too late.

If you believe the NSA is wasting $7.1 billion of taxpayer money on
snake oil, that is a coherent position --- but it requires explaining
why the world's most capable signals intelligence agency, with access
to classified intelligence on quantum programs that the rest of us do
not have, has reached a different conclusion.

> When we have quantum computers that can maintain quantum coherence
> while factoring arbitrary sixty four bit numbers, then it might become
> useful to think about what a quantum resistant algorithm might look
> like.

The factoring benchmark is a red herring for two reasons:

1. **ECC is easier to break than RSA/factoring.** Breaking a 256-bit
   elliptic curve requires 2.6x fewer logical qubits and 148x fewer
   quantum gates than breaking RSA-3072 at equivalent classical
   security [5]. The relevant milestone is not "factor 64-bit numbers"
   but "solve a 256-bit ECDLP." These are different problems with
   different quantum resource requirements.

2. **Error correction scaling matters more than integer size.** Google's
   Willow (December 2024, published in Nature [6]) demonstrated
   exponential error suppression with increasing surface code size ---
   each step from 3x3 to 5x5 to 7x7 reduced the logical error rate by
   2.14x. This is the metric that determines the timeline to a
   cryptographically relevant quantum computer, not which small integers
   have been factored as demonstrations. Extrapolating this scaling
   curve to the ~1,000-4,000 logical qubits needed for ECDLP gives a
   timeline in years, not decades. This is why governments are acting
   now.

The argument "we will migrate when threatened" also has a poor empirical
track record. Classical cryptographic weaknesses in deployed blockchain
systems have already caused real losses --- and the victims did not
migrate in time:

- **ECDSA nonce reuse (2013).** A bug in Android's PRNG caused Bitcoin
  wallets to reuse ECDSA nonces across multi-input transactions.
  Attackers recovered private keys algebraically and stole funds. The
  vulnerability was trivial to exploit once identified --- the math is
  undergraduate algebra, not quantum computing.

- **Polynonce attack (2023).** Kudelski Security [7] demonstrated that
  polynomial relations between ECDSA nonces allow private key recovery
  from as few as 4 signatures. Applied to the Bitcoin and Ethereum
  blockchains, they recovered hundreds of private keys. Approximately
  $40,000 was stolen from a single wallet. The attack uses polynomial
  root-finding --- a technique not anticipated when ECDSA was
  standardized.

- **Biased nonce lattice attacks (2019).** Breitner and Nadia [8]
  demonstrated lattice-based recovery of ECDSA private keys from
  signatures with biased nonces collected from the Bitcoin, Ethereum,
  and Ripple blockchains. Total documented theft: 144 BTC ($9.4 million
  at the time).

- **Eclipse/MitM via Tor (2015).** Biryukov and Pustogarov [9] showed
  that a low-resource attacker could isolate all Bitcoin-over-Tor users
  into a virtual Bitcoin reality, controlling which blocks and
  transactions they see. Combined with a mining collaborator, this
  enables double spends. The attack required only 40 Tor exit nodes.

- **51% attacks on ETC and BTG (2018-2020).** Ethereum Classic suffered
  three 51% attacks in a single month (August 2020), with
  reorganizations of 3,693, 4,000+, and 7,000+ blocks. Bitcoin Gold
  lost $18M in 2018 to the same class of attack. Both chains used
  algorithms where hash power was rentable on commodity markets.

In every case, the vulnerability was known in theory before it was
exploited in practice. In every case, users did not migrate in time.
The pattern is consistent: the window between "theoretical
vulnerability" and "funds stolen" is shorter than the window between
"theoretical vulnerability" and "ecosystem migration."

Shor's algorithm is a proven theoretical vulnerability against every
deployed ECC system. The question is not whether the window will close,
but when --- and whether the ecosystem will have migrated by then. The
historical record suggests it will not, unless the infrastructure for
migration is built before the threat materializes.

That said --- I want to be honest about uncertainty. I conceded in my
March 26 post that our first month of mainnet operation validates
Gutmann's Scenario D observation: all three critical failures we
experienced were engineering bugs, not cryptographic breaks. The
pedestrian threats dominate today. But engineering bugs are recoverable.
A cryptographic break of deployed ECC is not. The asymmetry between
recoverable and irrecoverable failure modes justifies defense-in-depth.

Our implementation uses height-gated crypto-agility: Phase 0 (Ed25519)
-> Phase 1 (Dilithium5) -> Phase 2 (SQIsign, 148-byte isogeny-based
signatures via NIST Round 2 C reference [10]). If lattice assumptions
fall, we activate a new algorithm at a future block height. No chain
reset. No lost history. The old algorithm validates old blocks; the new
algorithm validates new blocks.

This is not betting on any single PQC construction being correct. It is
building the infrastructure to replace constructions when they fail ---
which, as Bear argued in January, is the responsible engineering
approach when standards are premature.

Viktor S. Kristensen
Quillon Graph

## References

[1] G. Bracha, "Asynchronous Byzantine Agreement Protocols," Information
    and Computation 75(2), 1987.

[2] J. Liu, V. Wei, D. Wong, "Linkable Spontaneous Anonymous Group
    Signature for Ad Hoc Groups," ACISP 2004, LNCS 3108.

[3] L. Eagen, D. Fiore, A. Gabizon, "Bulletproofs++: Next Generation
    Confidential Transactions via Reciprocal Set Membership Arguments,"
    ePrint 2022/510.

[4] M. Mosca, "Cybersecurity in an Era with Quantum Computers: Will We
    Be Ready?" IEEE Security & Privacy 16(5), 2018.

[5] Kim et al., "Quantum Resource Requirements for Breaking Elliptic
    Curve Cryptography," Preprints.org 2025 (preprint, not yet
    peer-reviewed).

[6] Google Quantum AI, "Quantum error correction below the surface code
    threshold," Nature 638, 920-926 (2025).

[7] N. Amiet, M. Macchetti, "Polynonce: A Tale of a Novel ECDSA Attack
    and Bitcoin Tears," Kudelski Security / DEF CON 31, 2023.

[8] J. Breitner, N. Heninger, "Biased Nonce Sense: Lattice Attacks
    against Weak ECDSA Signatures in Cryptocurrencies," ePrint 2019/023.

[9] A. Biryukov, I. Pustogarov, "Bitcoin over Tor isn't a Good Idea,"
    IEEE S&P 2015.

[10] SQIsign: NIST PQC Round 2 submission, https://sqisign.org
