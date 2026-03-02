**To:** cryptography@metzdowd.com
**From:** Viktor S. Kristensen <viktor@quillon.xyz>
**Subject:** Re: Q-NarwhalKnight — Mainnet Genesis live, 3.6M blocks, some field notes on surviving a launch

---

Fellow cypherpunks,

A brief update to this thread. The system I described in December
and January is no longer theoretical. Q-NarwhalKnight mainnet
launched on February 22, 2026 at 12:00 UTC. As of this writing:

  - 3,673,000+ blocks produced
  - 73 connected peers via libp2p gossipsub
  - 182 unique miners
  - ~1 GH/s aggregate network hashrate
  - Every block signed with Dilithium5 since genesis

The Dilithium5/Kyber1024/SHA3-256/BLAKE3 stack described in my
earlier post has been running in continuous production for four
days. No signature verification failures. No key exchange
renegotiation bugs. No hash collisions (one would hope). The
genus-2 VDF anchor election has survived 3.6 million rounds
without a single equivocation event.

I promised this list I'd report back with field data rather than
benchmarks. Here it is — along with some candid observations
about what actually goes wrong when you ship a post-quantum
consensus system to production.


FIELD DATA: WHAT 3.6 MILLION BLOCKS TAUGHT US

1. Post-quantum signatures in the wild

Dilithium5's 4,595-byte signatures are not a theoretical burden —
they're a practical one. At ~1 block/second with 73 peers, the
gossipsub layer is moving roughly 335 KB/s of signature data
alone. On a 1 Gbit link, this is rounding error. On a 100 Mbit
link (more on that shortly), it becomes 2.7% of total capacity
just for signatures.

For comparison, Ed25519 at 64 bytes would consume ~4.7 KB/s.
The 72x overhead is real but manageable — Dilithium5 signatures
compress well under LZ4 (typically 38-42% reduction), and our
gossipsub batching amortizes the per-message overhead across
8-16 blocks per transmission.

Verification performance: ~2,400 Dilithium5 verifications/second
on a single core (AMD EPYC). Not a bottleneck at 1 bps, but it
will matter at 100 bps if we ever get there. We've pre-allocated
a dedicated verification thread pool for this reason.

2. VDF timing under adversarial network conditions

The genus-2 VDF proved more robust than expected to network jitter.
Our Kalman-filtered latency estimator (Project APOLLO — see [1])
predicts VDF evaluation windows with sufficient margin that even
200ms network spikes don't cause missed slots. We observed exactly
zero VDF timing failures across 3.6M blocks.

However, we did observe an interesting effect: when the network
hashrate spiked from 200 MH/s to 1 GH/s over 48 hours, the
adaptive difficulty adjustment oscillated for approximately 1,200
blocks (~20 minutes) before stabilizing. The PID rate controller
damped the oscillation, but the initial Kp gain was too aggressive
for a 5x hashrate jump. We've since added anti-windup clamping.

3. SHA3-256 vs BLAKE3: the dual-hash tradeoff

We use SHA3-256 for consensus-critical hashes (block hashing,
Merkle roots, address derivation) and BLAKE3 for mining
proof-of-work. This was initially a purity-vs-performance
compromise, but in production it's proven to be a useful
separation of concerns: SHA3-256's sponge construction gives
us domain separation for free (different capacity/rate ratios
per use case), while BLAKE3's SIMD acceleration lets miners
extract 35-85% more hash throughput on modern CPUs.

The miners are pushing BLAKE3 hard enough that we've added
NUMA-aware thread pinning and zero-allocation hot loops. One
contributor built an "extreme" binary with AVX-512 intrinsics
that achieves approximately 2.1x throughput over the portable
build — then discovered the hard way that deploying an AVX-512
binary to an AVX2-only machine produces a SIGILL, not a friendly
error message. (Compiler flags: the original "works on my
machine" problem.)


THE ENGINEERING COMEDY OF ERRORS

In the spirit of full transparency — and because this list has
always valued honest post-mortems over press releases — here is
a partial accounting of what went wrong between "testnet works"
and "mainnet works":

28 critical bugs discovered across 24 network phase transitions.

The greatest hits:

(a) THE FOUR-BUG CASCADE (Phase 8)

Four bugs, each hidden behind the previous fix. The environment
variable parser couldn't read the new network ID. But fixing that
revealed the CLI arguments took priority over env vars, nullifying
the fix. But fixing THAT revealed the NetworkConfig constructor
had a hardcoded phase. But fixing THAT revealed the block producer
embedded the wrong network ID in every block header.

Net effect: the node subscribed to the correct gossipsub topic,
then published to a different one. Complete silence. No error.
No warning. Just a node that looked healthy in every metric
except the one that mattered: it was talking to itself.

(If Dolev and Yao had written a Byzantine fault scenario about
misconfigured gossipsub topics, it would read exactly like our
production logs.)

(b) THE COMPRESSION ASYMMETRY (Rehearsal 1)

LZ4 compression on Server Beta, LZ4 decompression on Server
Gamma, configured with incompatible frame parameters. Blocks
serialized on one node couldn't deserialize on another. The
sync pipeline silently dropped every block. No error. No crash.
The blocks just... ceased to exist mid-transit.

Claude Shannon would have appreciated the irony: a communication
channel with nonzero capacity and zero mutual information.

(c) THE GENESIS FORK (Rehearsal 3)

Staggered server starts during the mainnet rehearsal. Beta starts
at T-10 minutes. Gamma starts at T-5 minutes. In those five
minutes, Gamma can't reach Beta, so it helpfully creates its own
genesis block. Result: two independent blockchains in the first
300 seconds of a network designed to be Byzantine fault tolerant.

The fix is embarrassingly obvious in hindsight: stop ALL nodes
before starting ANY of them. Coordinate genesis, don't race it.

(Lamport would note that this is precisely the problem his
logical clocks were designed to prevent. We had physical clocks.
They weren't enough.)

(d) THE BANDWIDTH DEGRADATION (Launch Week)

Four days before mainnet genesis, our hosting provider
(Contabo — may their SLA rest in peace) degraded the primary
bootstrap node from 1 Gbit/s to 100 Mbit/s. No notification.
No explanation. No ticket. Just a 90% bandwidth reduction on
the server that every miner in the world would connect to in
96 hours.

TCP window tuning. Kernel Receive Packet Steering. Nginx buffer
optimization. Gossipsub message batching. We squeezed every
useful bit through that pipe.

The system survived. The Kalman network predictor in our sync
engine (which estimates bandwidth, latency, and optimal chunk
sizes in real time) turned out to be worth every line of its
implementation. It adapted to the constrained link within
approximately 30 seconds of the degradation.

(e) THE ROGUE PROCESS MASSACRE (Post-Launch)

Server Delta — our third bootstrap node — crashed 20 times in
one week. Root cause: four processes nobody invited. A Zcash
node (3.5 GB). A Bitcoin node (2.3 GB). Two Iron Fish instances
(1.4 GB each). Total: 7.2 GB of RAM consumed by blockchain
software that wasn't ours, on an 8 GB server, with 8 GB swap
fully exhausted. The kernel's OOM killer executed our consensus
node with SIGKILL and absolutely no remorse.

The fix involved `kill -9`, `systemctl mask`, and a stern
conversation with whoever provisioned that VPS.


TOKENOMICS (For the Economists on the List)

21,000,000 QUG maximum supply. 24 decimal places. 4-year
halving across 64 eras (256 years to full emission). Era 0
emits 2,625,000 QUG/year. Adaptive block rewards adjust to
actual block rate via a PID controller with anti-windup.
No pre-mine. No ICO. No allocation. Everyone started at
genesis block zero.

The emission schedule is a geometric series:

  E(k) = E_0 / 2^k,  k = 0, 1, ..., 63

  where E_0 = 10,500,000 QUG (Era 0 total, over 4 years)

  Sum(k=0..63) E(k) = 2*E_0 * (1 - 2^(-64)) ~ 21,000,000

Bitcoin's schedule, essentially, but with ~1 second blocks
instead of ~600, adaptive difficulty, and DAG-based consensus
instead of longest-chain.


HASHRATE NOTE

The network crossed 1 GH/s this week. For context: there are
approximately 7.5 x 10^18 grains of sand on Earth. At 1 GH/s,
the network computes one SHA3-256 + BLAKE3 VDF proof for every
7.5 billion grains of sand, every second. To hash every grain
individually would take approximately 237 years at current rate.

This is, of course, entirely useless information from a
cryptographic perspective. But it makes for a good pub
conversation, and I suspect some of you frequent pubs.


WHAT'S NEXT

- Exchange listings: Community-funded campaigns for BitMart,
  LBank, and MEXC are active. ($12,252 raised so far — crypto
  is apparently easier to build than to list.)
- Cross-chain bridge: HTLC atomic swaps with 7-of-11 multi-sig
  committee for wBTC, wETH, wZEC, wIRON. Whitepaper published [2].
- QUG-V1: A 16-core heterogeneous RISC-V cluster chip (TSMC 5nm)
  optimized for BLAKE3 + VDF mining. The whitepaper [3] describes
  the architecture. Whether we'll actually tape out is a question
  of funding, physics, and how many more 3 AM debugging sessions
  my circadian rhythm can absorb.

The full technical whitepaper (v2.0, Mainnet Edition) is at [4].
33 research papers published to date, covering everything from
consensus security analysis to K-parameter trust frameworks to
a RISC-V chip that may or may not exist someday.


OPEN QUESTIONS FOR THIS LIST

1. Our genus-2 VDF has survived 3.6M evaluations without issue,
   but the security reduction assumes hardness of the discrete
   log problem on genus-2 Jacobians. Has anyone on this list
   seen recent work on index calculus improvements for genus-2
   beyond Gaudry's 2009 bounds? We're currently parameterized
   for ~128-bit classical / ~85-bit quantum security, and I'd
   like to sanity-check that margin.

2. The hybrid signature scheme (Ed25519 AND Dilithium5, both
   must verify) adds approximately 4.7 KB per block header. At
   scale (>100 bps), this becomes non-trivial for light clients.
   Has anyone explored efficient hybrid signature aggregation
   schemes that preserve the "either-or" security guarantee while
   reducing wire format? BLS aggregation doesn't extend cleanly
   to lattices.

3. Our AEGIS-256 block encryption relies on AES-NI hardware
   support. On ARM (Raspberry Pi miners — yes, we have those),
   we fall back to software AES, which is roughly 8x slower.
   Is there a compelling post-quantum AEAD construction that
   performs well on both x86 (with AES-NI) and ARM (without)?
   Ascon is interesting but we haven't benchmarked it in our
   pipeline yet.

I continue to find this list's signal-to-noise ratio refreshing
in a field where most "announcements" are tokenomics decks with
gradient backgrounds. Thank you for the rigorous feedback on
the K-parameter framework — the Monte Carlo validation has
held up well under the parameter perturbations several of you
suggested.

The source is at https://quillon.xyz. The node binary is a
single wget away:

  wget https://quillon.xyz/downloads/q-api-server-linux-x86_64

The chain verifies itself. Don't trust — hash.


— Viktor S. Kristensen (DK)
  Q-NarwhalKnight / Quillon
  https://quillon.xyz


References:

[1] "Project APOLLO: Adaptive Pipeline-Parallel Optimized Locking
    & Orchestration for Blockchain Synchronization," v2.1.0,
    February 2026.
    https://quillon.xyz/downloads/apollo-sync-optimization-whitepaper.pdf

[2] "Cross-Chain Bridge Protocol: Trustless Atomic Swaps with
    Multi-Sig Committee Validation," v2.0, February 2026.
    https://quillon.xyz/downloads/cross-chain-bridge-whitepaper.pdf

[3] "The QUG-V1: A Pocket Supercomputer for Decentralized Mining
    and Node Operation," v1.0.0, February 2026.
    https://quillon.xyz/downloads/qug-v1-pocket-supercomputer.pdf

[4] "Q-NarwhalKnight: A Quantum-Resistant Peer-to-Peer Network
    Architecture," Technical Whitepaper v2.0, Mainnet Edition.
    https://quillon.xyz/downloads/quillon-libp2p-network-whitepaper.pdf

[5] NIST FIPS 204 - ML-DSA (Dilithium) Digital Signature Standard
[6] NIST FIPS 203 - ML-KEM (Kyber) Key Encapsulation Mechanism
[7] Gaudry, "Index calculus for abelian varieties of small
    dimension and the elliptic curve discrete logarithm problem,"
    J. Symb. Comp., 2009.
