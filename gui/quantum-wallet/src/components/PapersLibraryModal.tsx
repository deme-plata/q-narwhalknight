import { useState, useMemo } from 'react';
import { createPortal } from 'react-dom';
import { motion, AnimatePresence } from 'framer-motion';
import { X, BookOpen, Search, Download, FileText, Star } from 'lucide-react';

interface Paper {
  title: string;
  description: string;
  score: number;
  category: string;
  filename: string;
}

const CATEGORIES = [
  'All',
  'Vision',
  'Core Protocol',
  'Cryptography',
  'Privacy',
  'Economics',
  'Infrastructure',
  'Applications',
  'Physics',
] as const;

type Category = (typeof CATEGORIES)[number];

const PAPERS: Paper[] = [
  // ── Vision ──
  { title: 'State of the Art (May 2026)', description: 'Multi-AI collaboration narrative: the agent that holds money, Phase 1 honesty model, cross-model engineering at the AI/crypto intersection', score: 10, category: 'Vision', filename: 'state-of-the-art-2026-05-18.pdf' },
  { title: 'Agentic Money on Quillon Graph', description: 'How autonomous AI agents become economic principals on Quillon Graph: MCP wallet primitives, Apache-2.0 commons framing, the first closed agentic loop', score: 10, category: 'Vision', filename: 'agentic-money-quillon-graph.pdf' },
  { title: 'The Entangled DAG', description: 'Formal structural analogy between quantum measurement, neural inference, and DAG-Knight consensus — with falsifiable prediction and explicit AI-authorship statement', score: 10, category: 'Vision', filename: 'entangled-dag-2026.pdf' },
  { title: 'Five Mirrors', description: 'Five historical analogues for Quillon Graph: AC moment, Pacioli, Moog, Cambrian survivors, wide-body jet. The category-of-one phase before a name exists', score: 9, category: 'Vision', filename: 'five-mirrors-2026.pdf' },
  { title: 'Speed at the Limit', description: 'Twenty comparisons between cryptographic networks and racing cars across 130 years of speed history; notes on what it feels like to move faster than your species', score: 8, category: 'Vision', filename: 'speed-twenty-comparisons-2026.pdf' },
  { title: 'Cosmic Flux Philosophy', description: 'Philosophical foundation for treating blockchain state as cosmic flux — time, memory, and immutability in distributed consensus', score: 7, category: 'Vision', filename: 'cosmic-flux-philosophy.pdf' },

  // ── Core Protocol ──
  { title: 'Formal Safety & Liveness Proofs', description: 'Formal safety and liveness proofs for DAG-Knight consensus under partial synchrony with K-parameter finality bounds', score: 10, category: 'Core Protocol', filename: 'dagknight-formal-safety-liveness.pdf' },
  { title: 'DAG-Knight Architecture', description: 'Detailed architecture of the DAG-Knight consensus protocol with quantum anchor election', score: 10, category: 'Core Protocol', filename: 'dagknight-architecture-detailed.pdf' },
  { title: 'Consensus Security Analysis', description: 'Formal security analysis of Q-NarwhalKnight consensus under adversarial conditions', score: 9, category: 'Core Protocol', filename: 'q-narwhalknight-consensus-security-analysis.pdf' },
  { title: 'BFT Slashing Protocol', description: 'Byzantine fault tolerance with economic slashing for misbehaving validators', score: 9, category: 'Core Protocol', filename: 'qnk-bft-slashing-whitepaper.pdf' },
  { title: 'Decentralization Analysis', description: 'Quantitative analysis of DAG-Knight decentralization guarantees and Nakamoto coefficient', score: 9, category: 'Core Protocol', filename: 'dagknight-decentralization-analysis.pdf' },
  { title: 'Hashpower Security', description: 'Security model for hashpower-based consensus resistance against 51% attacks', score: 8, category: 'Core Protocol', filename: 'qnk-hashpower-security-whitepaper.pdf' },
  { title: 'Block Validation & Rewards', description: 'Mainnet block reward schedule, validation rules, and issuance mechanics', score: 8, category: 'Core Protocol', filename: 'mainnet-rewards.pdf' },
  { title: 'Block Rewards (Draft)', description: 'Draft specification for mainnet block reward distribution curves', score: 6, category: 'Core Protocol', filename: 'mainnet-block-rewards-DRAFT.pdf' },
  { title: 'Deterministic Balance Migration', description: 'Mathematically rigorous framework for ledger state recovery via first-principles chain replay with proportional emission normalization (v8.8.6)', score: 9, category: 'Core Protocol', filename: 'deterministic-balance-migration-v886.pdf' },
  { title: 'Q-NarwhalKnight Whitepaper (May 2026)', description: 'Current canonical whitepaper covering DAG-Knight consensus, post-quantum cryptography roadmap, and emission economics as of May 2026', score: 10, category: 'Core Protocol', filename: 'qnk-whitepaper-2026-05-07.pdf' },
  { title: 'Q-NarwhalKnight Combined Whitepaper', description: 'Combined comprehensive whitepaper bundling all subsystem specifications: consensus, networking, cryptography, economics, and applications', score: 9, category: 'Core Protocol', filename: 'qnk-combined-whitepaper-2026-05-07.pdf' },
  { title: 'Quillon Graph Project Report v4', description: 'Fourth iteration of the comprehensive project report — current architecture, deployed components, and active engineering tracks', score: 8, category: 'Core Protocol', filename: 'quillon-graph-project-report-v4.pdf' },

  // ── Cryptography ──
  { title: 'Recursive SNARKs & Weak Subjectivity', description: 'Eliminating weak subjectivity through recursive SNARK proofs for trustless sync', score: 10, category: 'Cryptography', filename: 'recursive-snark-weak-subjectivity-elimination.pdf' },
  { title: 'Cryptography Improvements', description: 'Post-quantum cryptographic upgrades including Dilithium5 and Kyber1024', score: 9, category: 'Cryptography', filename: 'cryptography-improvements-whitepaper.pdf' },
  { title: 'Temporal Cryptographic Security v2', description: 'Time-locked cryptographic constructions with quantum-resistant temporal proofs', score: 9, category: 'Cryptography', filename: 'temporal-cryptographic-security-v2.pdf' },
  { title: 'Temporal Cryptographic Security v1', description: 'Original temporal security framework with VDF-based time proofs', score: 8, category: 'Cryptography', filename: 'temporal-cryptographic-security.pdf' },
  { title: 'K-Parameter Cryptographic Trust v2', description: 'Extended K-parameter framework for tunable cryptographic trust levels', score: 9, category: 'Cryptography', filename: 'k-parameter-cryptographic-trust-v2.pdf' },
  { title: 'K-Parameter Cryptographic Trust', description: 'Original K-parameter trust model for consensus parameterization', score: 8, category: 'Cryptography', filename: 'k-parameter-cryptographic-trust.pdf' },
  { title: 'Genus-2 Jacobian VDF Mining', description: 'Novel VDF construction using genus-2 hyperelliptic curve Jacobians for mining', score: 10, category: 'Cryptography', filename: 'genus2-jacobian-vdf-mining-whitepaper.pdf' },
  { title: 'K-Parameter Quantum Frontiers', description: 'Extending the K-parameter to quantum computing frontiers and post-quantum readiness', score: 9, category: 'Cryptography', filename: 'k-parameter-quantum-frontiers.pdf' },
  { title: 'Genus-2 VDF Dual-Lane Mining v3', description: 'Third iteration of the genus-2 hyperelliptic VDF mining construction with dual-lane proof acceleration and improved batch verification', score: 9, category: 'Cryptography', filename: 'genus2-vdf-dual-lane-mining-v3.pdf' },
  { title: 'Recursive Lattice SNARK Whitepaper', description: 'Recursive zero-knowledge SNARK construction over lattice-based commitments for post-quantum verifiable computation (2026-05-13)', score: 10, category: 'Cryptography', filename: 'quillon-recursive-lattice-snark-whitepaper-2026-05-13.pdf' },
  { title: 'Recursive Lattice SNARK v2', description: 'Revised recursive lattice SNARK construction with tighter parameters and explicit prover-side optimization (2026-05-13)', score: 10, category: 'Cryptography', filename: 'quillon-recursive-lattice-snark-whitepaper-v2-2026-05-13.pdf' },

  // ── Privacy ──
  { title: 'Privacy Layer v3', description: 'Network-layer privacy through Tor-integrated dedicated circuits — comprehensive privacy architecture for quantum-resistant blockchain infrastructure', score: 10, category: 'Privacy', filename: 'qnk-privacy-layer-whitepaper.pdf' },
  { title: 'Privacy Philosophy', description: 'Philosophical foundations of privacy as a fundamental right in digital consensus', score: 8, category: 'Privacy', filename: 'privacy-philosophy-whitepaper.pdf' },
  { title: 'Quantum Mixer v3', description: 'Third-generation quantum-resistant transaction mixer with zero-knowledge proofs', score: 9, category: 'Privacy', filename: 'quantum_mixer_whitepaper_v3.pdf' },
  { title: 'Quantum Mixer v2', description: 'Enhanced quantum mixer with improved anonymity sets and mixing efficiency', score: 8, category: 'Privacy', filename: 'quantum_mixer_whitepaper_v2.pdf' },
  { title: 'Quantum Mixer v1', description: 'Original quantum-resistant transaction mixing protocol', score: 7, category: 'Privacy', filename: 'quantum_mixer_whitepaper.pdf' },
  { title: 'Transaction Tunneling', description: 'Technical review of transaction tunneling for censorship-resistant value transfer', score: 8, category: 'Privacy', filename: 'transaction-tunneling-review.pdf' },
  { title: 'Distributed AI Privacy', description: 'Privacy-preserving distributed AI inference with federated learning on-chain', score: 9, category: 'Privacy', filename: 'distributed-ai-privacy-whitepaper.pdf' },

  // ── Economics ──
  { title: 'QUG Emission Economics', description: 'Tokenomics of QUG emission schedule: 21M supply, 4-year halving, era-based rewards', score: 10, category: 'Economics', filename: 'qug-emission-economics.pdf' },
  { title: 'Incentive-Compatible Multi-Shard Mining', description: 'Game-theoretic analysis of DAG-based multi-shard PoW mining with Nash equilibrium proofs, selfish mining resistance, MEV dilution, and halving emission incentive compatibility', score: 9, category: 'Economics', filename: 'incentive-compatible-multishard-mining.pdf' },
  { title: 'Reserve Asset Whitepaper', description: 'Quillon as a quantum-resistant reserve asset and store of value thesis', score: 9, category: 'Economics', filename: 'quillon-reserve-asset-whitepaper.pdf' },
  { title: 'Investor Pitch v2', description: 'Investment thesis for the Quillon Graph ecosystem and QUG token', score: 7, category: 'Economics', filename: 'investor-pitch-v2.pdf' },
  { title: 'RWA Tokenization', description: 'Real-world asset tokenization framework with compliance and oracle integration', score: 8, category: 'Economics', filename: 'rwa-tokenization-whitepaper.pdf' },
  { title: 'Quillon Vault', description: 'Decentralized vault system for yield-bearing QUG deposits with insurance', score: 8, category: 'Economics', filename: 'quillon-vault-whitepaper.pdf' },
  { title: 'Bank Aegis QL', description: 'Institutional-grade banking layer with quantum-secure custody and settlement', score: 8, category: 'Economics', filename: 'quillon_bank_aegis_ql_whitepaper.pdf' },
  { title: 'The Quantum Millionaire Mind', description: 'Secrets of blockchain wealth from the Q-NarwhalKnight founder — rewriting your financial blueprint with Rust, u128 arithmetic, and 4,000 tests', score: 8, category: 'Economics', filename: 'the-quantum-millionaire-mind.pdf' },
  { title: 'ASIC 12nm Cost Model v1', description: 'Detailed cost-modelling exercise for a 12nm ASIC implementation of the Genus-2 VDF mining circuit, including wafer economics and break-even analysis', score: 9, category: 'Economics', filename: 'asic-12nm-cost-model-v1.pdf' },
  { title: 'Trading Water Whitepaper', description: 'Continuous-DCA "water-bot" trading strategy on Quillon Graph DEX: design, safety policy, and observed P&L on testnet pools', score: 7, category: 'Economics', filename: 'trading-water-whitepaper.pdf' },

  // ── Infrastructure ──
  { title: 'Networking Whitepaper', description: 'libp2p-based networking stack with gossipsub, Kademlia DHT, and Tor support', score: 9, category: 'Infrastructure', filename: 'q-narwhalknight-networking-whitepaper.pdf' },
  { title: 'Infrastructure Whitepaper', description: 'Full infrastructure architecture: storage, APIs, monitoring, and deployment', score: 9, category: 'Infrastructure', filename: 'q-narwhalknight-infrastructure-whitepaper.pdf' },
  { title: 'P2P Gossipsub Protocol', description: 'Custom gossipsub protocol implementation with topic-based message routing', score: 8, category: 'Infrastructure', filename: 'p2p-gossipsub-whitepaper.pdf' },
  { title: 'libp2p Network Architecture', description: 'Detailed libp2p network topology design with NAT traversal and relay circuits', score: 8, category: 'Infrastructure', filename: 'quillon-libp2p-network-whitepaper.pdf' },
  { title: 'Apollo Sync Optimization', description: 'Turbo sync and Apollo protocol for fast initial block download and state sync', score: 9, category: 'Infrastructure', filename: 'apollo-sync-optimization-whitepaper.pdf' },
  { title: 'Warp Sync Protocol', description: 'Warp sync for sub-minute full node bootstrapping using state snapshots', score: 8, category: 'Infrastructure', filename: 'warp-sync-whitepaper.pdf' },
  { title: 'Adaptive Pruning', description: 'Dynamic state pruning with configurable retention for storage efficiency', score: 8, category: 'Infrastructure', filename: 'adaptive-pruning-whitepaper.pdf' },
  { title: 'Blockchain Pruning Review', description: 'Technical review of pruning strategies across blockchain architectures', score: 7, category: 'Infrastructure', filename: 'blockchain-pruning-technical-review.pdf' },
  { title: 'Browser P2P Integration', description: 'WebRTC and libp2p integration for browser-native P2P connectivity', score: 8, category: 'Infrastructure', filename: 'browser-p2p-libp2p-integration.pdf' },
  { title: 'Q Miner', description: 'Mining client architecture, proof-of-work algorithms, and pool protocol', score: 8, category: 'Infrastructure', filename: 'q-miner-whitepaper.pdf' },
  { title: 'Node Operator Guide v2', description: 'Zero-config setup, fee structure (0.1% block rewards + DEX protocol fees), admin wallet linking, and OAuth2 wallet integration', score: 9, category: 'Infrastructure', filename: 'qnk-node-operator-guide.pdf' },
  { title: 'Q-Flux Reverse Proxy Architecture', description: 'Architecture of a blockchain-aware reverse proxy with worker-per-core design, AIMD concurrency control, and lock-free connection pooling', score: 9, category: 'Infrastructure', filename: 'q-flux-reverse-proxy.pdf' },
  { title: 'Q-Flux & Q-Queue Project Report', description: 'Combined project report on the q-flux reverse proxy and the q-queue task scheduler — design, deployment status, and integration with the API server', score: 8, category: 'Infrastructure', filename: 'q-flux-q-queue-project-report.pdf' },
  { title: 'Balance Incident Report (April 2026)', description: 'Post-mortem of the April 2026 balance-replay incident on Epsilon: root cause, immediate mitigation, and the four non-negotiable balance-integrity rules now in CLAUDE.md', score: 9, category: 'Infrastructure', filename: 'balance-incident-report-2026-04-29.pdf' },

  // ── Applications ──
  { title: 'VM & DEX Whitepaper', description: 'WASM virtual machine and decentralized exchange with AMM liquidity pools', score: 9, category: 'Applications', filename: 'q-narwhalknight-vm-dex-whitepaper.pdf' },
  { title: 'Q-VM Architecture', description: 'Quantum-ready virtual machine with deterministic execution and gas metering', score: 9, category: 'Applications', filename: 'q-vm-whitepaper.pdf' },
  { title: 'Quillon Forge', description: 'Smart contract development platform with templates and deployment tooling', score: 8, category: 'Applications', filename: 'quillon-forge-whitepaper.pdf' },
  { title: 'Cross-Chain Bridge', description: 'Trustless cross-chain bridge with threshold signatures and relay verification', score: 8, category: 'Applications', filename: 'cross-chain-bridge-whitepaper.pdf' },
  { title: 'Quantum Neural Oracle', description: 'Decentralized oracle network with quantum-enhanced neural consensus', score: 9, category: 'Applications', filename: 'quantum-neural-oracle-whitepaper.pdf' },
  { title: 'Bio-DSL Language', description: 'Domain-specific language for biological computation and on-chain bio-simulations', score: 7, category: 'Applications', filename: 'bio-dsl-whitepaper.pdf' },
  { title: 'Distributed AI Technical Review', description: 'Technical review of distributed AI inference capabilities within Q-NarwhalKnight', score: 8, category: 'Applications', filename: 'distributed-ai-technical-review.pdf' },
  { title: 'QUG v1 Pocket Supercomputer', description: 'Edge computing node design for mobile-first blockchain participation', score: 7, category: 'Applications', filename: 'qug-v1-pocket-supercomputer.pdf' },
  { title: 'Quillion Graph Complete', description: 'Complete Quillion Graph ontology and knowledge-graph-on-chain architecture', score: 9, category: 'Applications', filename: 'quillion-graph-complete-whitepaper.pdf' },
  { title: 'Agent Wallet as a Service — Spec', description: 'Product specification for a hosted Agent-Wallet service: 20 must/nice features (F-01 through F-20), 12-week roadmap, and pricing tiers (Free / $9 / $99 / $2000+)', score: 9, category: 'Applications', filename: 'agent-wallet-as-a-service-kravspec.pdf' },
  { title: 'Crown Ash Whitepaper', description: 'Crown Ash hardware mining client design and integration pathway for Quillon Graph proof-of-work', score: 7, category: 'Applications', filename: 'crown-ash-whitepaper.pdf' },

  // ── Physics ──
  { title: 'K-Parameter Unified Theory', description: 'Unified K-parameter whitepaper bridging quantum physics and consensus mechanics', score: 10, category: 'Physics', filename: 'K-Parameter_Whitepaper.pdf' },
  { title: 'Quantum Fields Philosophy', description: 'Philosophical implications of quantum field theory applied to distributed systems', score: 8, category: 'Physics', filename: 'Quantum-Fields-Philosophy.pdf' },
  { title: 'Quantum Physics (Full)', description: 'Comprehensive quantum physics foundations for post-quantum blockchain security', score: 9, category: 'Physics', filename: 'quantum-physics-whitepaper-full.pdf' },
  { title: 'Quantum Physics (Simplified)', description: 'Accessible introduction to quantum physics concepts in blockchain context', score: 7, category: 'Physics', filename: 'quantum-physics-whitepaper-simple.pdf' },
  { title: 'Quantum Physics for Teens', description: 'Youth-friendly explanation of quantum consensus and cryptography', score: 6, category: 'Physics', filename: 'quantum-physics-whitepaper-teens.pdf' },
  { title: 'Higgs Simulation Factory', description: 'Higgs boson simulation framework for quantum field consensus verification', score: 8, category: 'Physics', filename: 'higgs-simulation-factory.pdf' },
  { title: 'Theoretical Physics Node System', description: 'Physics-inspired node topology and field-theoretic network modeling', score: 8, category: 'Physics', filename: 'theoretical-physics-node-system.pdf' },
  { title: 'The Eternal Ledger', description: 'Philosophical treatise on immutability, time, and the eternal nature of ledgers', score: 8, category: 'Physics', filename: 'the-eternal-ledger-philosophy.pdf' },
  { title: 'Quillon Thesis (Verified)', description: 'Formally verified thesis on the Quillon consensus mechanism', score: 9, category: 'Physics', filename: 'quillon-thesis-verified.pdf' },
  { title: 'Project Report', description: 'Comprehensive Q-NarwhalKnight project report covering all subsystems', score: 7, category: 'Physics', filename: 'Q-NarwhalKnight-Project-Report.pdf' },
  { title: 'Water Robots Universe Mission', description: 'Speculative water-robot mission framework with K-parameter extensions', score: 6, category: 'Physics', filename: 'water-robots-universe-mission.pdf' },
  { title: 'Water Robot K-Parameter Appendix', description: 'Appendix extending K-parameter theory to aquatic robotic systems', score: 6, category: 'Physics', filename: 'WATER_ROBOT_K_PARAMETER_APPENDIX.pdf' },
  { title: 'Quantum Water Robots Kingdom', description: 'Quantum-mechanical modeling of water robot swarm coordination', score: 5, category: 'Physics', filename: 'quantum-water-robots-kingdom.pdf' },
  { title: 'Theoretical Physics Consensus (Susskind)', description: 'Theoretical physics framework for consensus mechanics drawing on Leonard Susskind\'s work on holographic principles and the eternal ledger', score: 9, category: 'Physics', filename: 'theoretical-physics-consensus-v3-susskind.pdf' },
  { title: 'Theoretical Physics Consensus (Lloyd)', description: 'Fourth revision of the theoretical-physics consensus framework, drawing on Seth Lloyd\'s computational universe model for cosmological state-machine analogues', score: 9, category: 'Physics', filename: 'theoretical-physics-consensus-v4-lloyd.pdf' },
  { title: 'Theoretical Physics Consensus', description: 'Original theoretical-physics-consensus whitepaper framing distributed agreement protocols as physical-state-machine evolution', score: 8, category: 'Physics', filename: 'theoretical-physics-consensus-whitepaper.pdf' },
  { title: 'Quillon Thesis (Enhanced)', description: 'Enhanced edition of the Quillon thesis with expanded physical-foundations chapter and updated cryptographic appendices', score: 9, category: 'Physics', filename: 'quillon-thesis-enhanced.pdf' },
  { title: 'Quantum Physics for Teens, Volume 2', description: 'Second volume in the youth-oriented quantum-physics series — building on Volume 1 to cover entanglement, measurement, and the consensus connection', score: 6, category: 'Physics', filename: 'quantum-physics-teens-volume2.pdf' },
];

const DL_BASE = 'https://quillon.xyz/downloads';

function scoreBadgeColor(score: number): string {
  if (score >= 9) return 'from-amber-400 to-yellow-500';
  if (score >= 7) return 'from-cyan-400 to-blue-500';
  if (score >= 5) return 'from-slate-400 to-slate-500';
  return 'from-slate-500 to-slate-600';
}

function scoreBadgeShadow(score: number): string {
  if (score >= 9) return '0 0 12px rgba(251,191,36,0.5)';
  if (score >= 7) return '0 0 12px rgba(34,211,238,0.3)';
  return 'none';
}

function categoryColor(cat: string): string {
  switch (cat) {
    case 'Core Protocol': return 'bg-amber-500/20 text-amber-300 border-amber-500/30';
    case 'Cryptography': return 'bg-purple-500/20 text-purple-300 border-purple-500/30';
    case 'Privacy': return 'bg-emerald-500/20 text-emerald-300 border-emerald-500/30';
    case 'Economics': return 'bg-yellow-500/20 text-yellow-300 border-yellow-500/30';
    case 'Infrastructure': return 'bg-blue-500/20 text-blue-300 border-blue-500/30';
    case 'Applications': return 'bg-cyan-500/20 text-cyan-300 border-cyan-500/30';
    case 'Physics': return 'bg-rose-500/20 text-rose-300 border-rose-500/30';
    default: return 'bg-slate-500/20 text-slate-300 border-slate-500/30';
  }
}

interface PapersLibraryModalProps {
  isOpen: boolean;
  onClose: () => void;
}

export default function PapersLibraryModal({ isOpen, onClose }: PapersLibraryModalProps) {
  const [activeCategory, setActiveCategory] = useState<Category>('All');
  const [searchQuery, setSearchQuery] = useState('');

  const filtered = useMemo(() => {
    let list = PAPERS;
    if (activeCategory !== 'All') {
      list = list.filter(p => p.category === activeCategory);
    }
    if (searchQuery.trim()) {
      const q = searchQuery.toLowerCase();
      list = list.filter(p =>
        p.title.toLowerCase().includes(q) ||
        p.description.toLowerCase().includes(q) ||
        p.category.toLowerCase().includes(q)
      );
    }
    return list.sort((a, b) => b.score - a.score);
  }, [activeCategory, searchQuery]);

  const categoryCounts = useMemo(() => {
    const counts: Record<string, number> = { All: PAPERS.length };
    for (const p of PAPERS) {
      counts[p.category] = (counts[p.category] || 0) + 1;
    }
    return counts;
  }, []);

  return createPortal(
    <AnimatePresence>
      {isOpen && (
        <>
          {/* Backdrop */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 bg-black/85 backdrop-blur-md z-[10002]"
            onClick={onClose}
          />

          {/* Modal Container */}
          <motion.div
            initial={{ opacity: 0, scale: 0.92, y: 30 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.92, y: 30 }}
            transition={{ type: 'spring', damping: 28, stiffness: 320 }}
            className="fixed inset-0 z-[10003] flex items-center justify-center p-4"
            onClick={onClose}
          >
            <div
              className="w-full max-w-6xl max-h-[90vh] flex flex-col overflow-hidden rounded-2xl border border-amber-500/20"
              onClick={(e) => e.stopPropagation()}
              style={{
                background: 'linear-gradient(135deg, rgba(15,15,25,0.98) 0%, rgba(20,15,30,0.98) 50%, rgba(15,20,25,0.98) 100%)',
                boxShadow: '0 0 80px rgba(251,191,36,0.08), 0 0 40px rgba(0,0,0,0.5), inset 0 1px 0 rgba(251,191,36,0.1)',
              }}
            >
              {/* Header */}
              <div className="flex-shrink-0 px-6 pt-5 pb-4 border-b border-amber-500/10">
                <div className="flex items-center justify-between mb-4">
                  <div className="flex items-center gap-3">
                    <div
                      className="w-10 h-10 rounded-xl flex items-center justify-center"
                      style={{
                        background: 'linear-gradient(135deg, rgba(251,191,36,0.2), rgba(245,158,11,0.1))',
                        boxShadow: '0 0 20px rgba(251,191,36,0.15)',
                      }}
                    >
                      <BookOpen className="w-5 h-5 text-amber-400" />
                    </div>
                    <div>
                      <h2 className="text-xl font-bold text-white tracking-tight">Research Library</h2>
                      <p className="text-xs text-slate-500 mt-0.5">
                        {PAPERS.length} whitepapers &amp; technical documents
                      </p>
                    </div>
                    <span
                      className="ml-2 px-2.5 py-0.5 rounded-full text-xs font-semibold"
                      style={{
                        background: 'linear-gradient(135deg, rgba(251,191,36,0.2), rgba(245,158,11,0.1))',
                        color: '#fbbf24',
                        border: '1px solid rgba(251,191,36,0.25)',
                      }}
                    >
                      {filtered.length} shown
                    </span>
                  </div>
                  <button
                    onClick={onClose}
                    className="p-2 rounded-lg hover:bg-slate-700/50 text-slate-500 hover:text-white transition-colors"
                  >
                    <X className="w-5 h-5" />
                  </button>
                </div>

                {/* Search Bar */}
                <div className="relative mb-4">
                  <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-500" />
                  <input
                    type="text"
                    placeholder="Search papers by title, description, or category..."
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                    className="w-full pl-10 pr-4 py-2.5 rounded-xl bg-slate-800/60 border border-slate-700/50 text-sm text-white placeholder-slate-500 focus:outline-none focus:border-amber-500/40 focus:ring-1 focus:ring-amber-500/20 transition-colors"
                  />
                </div>

                {/* Category Tabs */}
                <div className="flex gap-1.5 overflow-x-auto pb-1 scrollbar-hide">
                  {CATEGORIES.map(cat => (
                    <button
                      key={cat}
                      onClick={() => setActiveCategory(cat)}
                      className={`flex-shrink-0 px-3 py-1.5 rounded-lg text-xs font-medium transition-all duration-200 ${
                        activeCategory === cat
                          ? 'bg-amber-500/20 text-amber-300 border border-amber-500/30 shadow-[0_0_12px_rgba(251,191,36,0.15)]'
                          : 'bg-slate-800/40 text-slate-400 border border-slate-700/30 hover:bg-slate-700/40 hover:text-slate-300'
                      }`}
                    >
                      {cat}
                      <span className={`ml-1.5 ${activeCategory === cat ? 'text-amber-400/70' : 'text-slate-600'}`}>
                        {categoryCounts[cat] || 0}
                      </span>
                    </button>
                  ))}
                </div>
              </div>

              {/* Paper Grid (scrollable) */}
              <div className="flex-1 overflow-y-auto p-6" style={{ scrollbarWidth: 'thin', scrollbarColor: 'rgba(251,191,36,0.2) transparent' }}>
                {filtered.length === 0 ? (
                  <div className="flex flex-col items-center justify-center py-16 text-slate-500">
                    <FileText className="w-12 h-12 mb-3 opacity-30" />
                    <p className="text-sm">No papers match your search.</p>
                  </div>
                ) : (
                  <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
                    {filtered.map((paper, i) => (
                      <motion.div
                        key={paper.filename}
                        initial={{ opacity: 0, y: 12 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ delay: Math.min(i * 0.02, 0.3), duration: 0.25 }}
                      >
                        <a
                          href={`${DL_BASE}/${paper.filename}`}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="group block h-full rounded-xl border border-slate-700/40 hover:border-amber-500/30 transition-all duration-300 overflow-hidden"
                          style={{
                            background: 'linear-gradient(160deg, rgba(30,30,45,0.8) 0%, rgba(20,20,35,0.6) 100%)',
                          }}
                        >
                          <div className="relative p-4">
                            {/* Score Badge */}
                            <div
                              className={`absolute top-3 right-3 w-8 h-8 rounded-lg flex items-center justify-center text-xs font-bold text-white bg-gradient-to-br ${scoreBadgeColor(paper.score)}`}
                              style={{ boxShadow: scoreBadgeShadow(paper.score) }}
                            >
                              {paper.score}
                            </div>

                            {/* Title */}
                            <h3 className="text-sm font-semibold text-white group-hover:text-amber-300 transition-colors pr-10 mb-1.5 leading-snug">
                              {paper.title}
                            </h3>

                            {/* Description */}
                            <p className="text-xs text-slate-400 leading-relaxed mb-3 line-clamp-2">
                              {paper.description}
                            </p>

                            {/* Footer: Category + Download */}
                            <div className="flex items-center justify-between">
                              <span className={`inline-flex items-center px-2 py-0.5 rounded-md text-[10px] font-medium border ${categoryColor(paper.category)}`}>
                                {paper.category}
                              </span>
                              <span className="flex items-center gap-1 text-[10px] text-slate-500 group-hover:text-amber-400/70 transition-colors">
                                <Download className="w-3 h-3" />
                                PDF
                              </span>
                            </div>
                          </div>

                          {/* Hover glow line */}
                          <div
                            className="h-[2px] w-full opacity-0 group-hover:opacity-100 transition-opacity duration-300"
                            style={{ background: 'linear-gradient(90deg, transparent, rgba(251,191,36,0.5), transparent)' }}
                          />
                        </a>
                      </motion.div>
                    ))}
                  </div>
                )}
              </div>

              {/* Footer */}
              <div className="flex-shrink-0 px-6 py-3 border-t border-amber-500/10 flex items-center justify-between">
                <div className="flex items-center gap-2 text-xs text-slate-600">
                  <Star className="w-3 h-3" />
                  <span>Score reflects relevance to Q-NarwhalKnight core technology</span>
                </div>
                <span className="text-[10px] text-slate-600 font-mono">quillon.xyz</span>
              </div>
            </div>
          </motion.div>
        </>
      )}
    </AnimatePresence>,
    document.body
  );
}
