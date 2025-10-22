import { motion } from 'framer-motion';
import { Shield, Lock, Eye, Globe, Zap, CheckCircle2, ArrowRight, Rocket, Clock } from 'lucide-react';

export default function PaaSComingSoon() {
  const upcomingEndpoints = [
    {
      method: 'POST',
      path: '/api/v1/privacy/tor/relay',
      description: 'Route transactions through Tor network with quantum-seeded circuits',
      status: 'Q1 2026'
    },
    {
      method: 'POST',
      path: '/api/v1/privacy/mix/submit',
      description: 'Submit transactions for mixing with ring signatures and stealth addresses',
      status: 'Q1 2026'
    },
    {
      method: 'POST',
      path: '/api/v1/privacy/ring-signature/generate',
      description: 'Generate lattice-based quantum-resistant ring signatures',
      status: 'Q2 2026 (Experimental)'
    },
    {
      method: 'POST',
      path: '/api/v1/privacy/stealth-address/generate',
      description: 'Create one-time stealth addresses for enhanced privacy',
      status: 'Q1 2026'
    },
    {
      method: 'POST',
      path: '/api/v1/privacy/zk-stark/prove',
      description: 'Generate zero-knowledge STARK proofs for private transactions',
      status: 'Q1 2026'
    },
    {
      method: 'GET',
      path: '/api/v1/privacy/paas/statistics',
      description: 'Retrieve PaaS usage statistics and revenue metrics',
      status: 'Q1 2026'
    }
  ];

  const features = [
    {
      icon: Shield,
      title: 'Quantum-Resistant Privacy',
      description: 'Post-quantum cryptography (Dilithium5, Kyber1024) protecting against future threats'
    },
    {
      icon: Globe,
      title: 'Cross-Chain Support',
      description: 'Bitcoin, Ethereum, Solana, and more via universal RESTful API'
    },
    {
      icon: Lock,
      title: 'Triple-Layer Security',
      description: 'Network (Tor) + Transport (Noise) + Application (Mixing) protection'
    },
    {
      icon: Eye,
      title: 'Compliance-Ready',
      description: 'Opt-in KYT/KYO, IVMS-101 Travel Rule, audit trails with ZK proofs'
    },
    {
      icon: Zap,
      title: 'Enterprise SLA',
      description: '99.9% uptime guarantee, dedicated infrastructure, 24/7 support'
    },
    {
      icon: Rocket,
      title: 'MEV Protection',
      description: 'Flashbots/MEV-Share integration with randomized timing'
    }
  ];

  const pricingTiers = [
    {
      name: 'Pay-Per-Use',
      price: 'From 0.001 QNK',
      description: 'Individual users and developers',
      features: [
        'Tor relay: 0.001 QNK/MB',
        'Transaction mixing: 0.1% fee',
        'Ring signatures: 0.001 QNK',
        'ZK-STARK proofs: 0.01 QNK',
        '95% uptime SLA'
      ]
    },
    {
      name: 'Professional',
      price: '$499/month',
      description: 'Wallets, DApps, trading bots',
      features: [
        '100,000 API calls/month',
        'Priority Tor circuits',
        'Dedicated support',
        '99% uptime SLA',
        'Webhook notifications'
      ],
      highlighted: true
    },
    {
      name: 'Enterprise',
      price: '$1,999/month',
      description: 'Exchanges, DeFi protocols, institutions',
      features: [
        'Unlimited API calls',
        'Controlled egress relays',
        'Priority mixing pools',
        '99.9% uptime SLA',
        'Custom compliance configs'
      ]
    },
    {
      name: 'White-Label',
      price: '$9,999/month',
      description: 'Blockchain networks, large enterprises',
      features: [
        'Your branding',
        'On-premise deployment',
        'Source code access',
        'Dedicated infrastructure',
        'Regulatory consulting'
      ]
    }
  ];

  return (
    <div className="space-y-8">
      {/* Hero Section */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="relative overflow-hidden rounded-2xl bg-gradient-to-br from-quantum-purple/20 via-quantum-indigo/20 to-quantum-cyan/20 border border-quantum-purple/30 p-8"
      >
        <div className="relative z-10">
          <div className="flex items-center gap-3 mb-4">
            <div className="p-3 bg-gradient-to-br from-quantum-purple to-quantum-cyan rounded-xl">
              <Shield className="w-8 h-8 text-white" />
            </div>
            <div>
              <h1 className="text-3xl font-bold bg-gradient-to-r from-quantum-cyan via-quantum-purple to-quantum-pink bg-clip-text text-transparent">
                Privacy-as-a-Service (PaaS)
              </h1>
              <p className="text-quantum-cyan text-sm">Coming Q1 2026</p>
            </div>
          </div>
          <p className="text-lg text-gray-300 mb-6 max-w-3xl">
            Enterprise-grade quantum-resistant privacy infrastructure for any blockchain.
            Universal privacy layer with cross-chain support, compliance tools, and 99.9% SLA.
          </p>
          <div className="flex flex-wrap gap-4">
            <a
              href="/PRIVACY_AS_A_SERVICE_WHITEPAPER.pdf"
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-2 px-6 py-3 bg-gradient-to-r from-quantum-purple to-quantum-cyan rounded-lg text-white font-medium hover:shadow-lg hover:shadow-quantum-purple/50 transition-all"
            >
              <Rocket className="w-5 h-5" />
              Download Whitepaper
            </a>
            <button className="inline-flex items-center gap-2 px-6 py-3 bg-quantum-dark/50 border border-quantum-purple/30 rounded-lg text-white font-medium hover:bg-quantum-purple/10 transition-all">
              <Clock className="w-5 h-5" />
              Join Waitlist (Coming Soon)
            </button>
          </div>
        </div>
        <div className="absolute top-0 right-0 w-64 h-64 bg-gradient-to-br from-quantum-cyan/20 to-transparent rounded-full blur-3xl" />
        <div className="absolute bottom-0 left-0 w-64 h-64 bg-gradient-to-tr from-quantum-purple/20 to-transparent rounded-full blur-3xl" />
      </motion.div>

      {/* Key Features */}
      <div>
        <h2 className="text-2xl font-bold text-white mb-6">Key Features</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {features.map((feature, index) => {
            const Icon = feature.icon;
            return (
              <motion.div
                key={feature.title}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: index * 0.1 }}
                className="p-6 bg-quantum-dark/50 border border-quantum-purple/30 rounded-xl hover:border-quantum-cyan/50 transition-all"
              >
                <div className="flex items-start gap-4">
                  <div className="p-3 bg-gradient-to-br from-quantum-purple/20 to-quantum-cyan/20 rounded-lg">
                    <Icon className="w-6 h-6 text-quantum-cyan" />
                  </div>
                  <div className="flex-1">
                    <h3 className="font-semibold text-white mb-2">{feature.title}</h3>
                    <p className="text-sm text-gray-400">{feature.description}</p>
                  </div>
                </div>
              </motion.div>
            );
          })}
        </div>
      </div>

      {/* Upcoming API Endpoints */}
      <div>
        <h2 className="text-2xl font-bold text-white mb-6">Upcoming API Endpoints</h2>
        <div className="space-y-3">
          {upcomingEndpoints.map((endpoint, index) => (
            <motion.div
              key={endpoint.path}
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: index * 0.1 }}
              className="p-4 bg-quantum-dark/50 border border-quantum-purple/30 rounded-xl hover:border-quantum-cyan/50 transition-all"
            >
              <div className="flex items-start justify-between gap-4">
                <div className="flex-1">
                  <div className="flex items-center gap-3 mb-2">
                    <span className={`px-3 py-1 rounded-lg text-xs font-mono font-semibold ${
                      endpoint.method === 'POST'
                        ? 'bg-quantum-cyan/20 text-quantum-cyan border border-quantum-cyan/30'
                        : 'bg-quantum-purple/20 text-quantum-purple border border-quantum-purple/30'
                    }`}>
                      {endpoint.method}
                    </span>
                    <code className="text-sm text-white font-mono">{endpoint.path}</code>
                  </div>
                  <p className="text-sm text-gray-400">{endpoint.description}</p>
                </div>
                <div className="flex items-center gap-2 px-3 py-1 bg-quantum-purple/10 border border-quantum-purple/30 rounded-lg">
                  <Clock className="w-4 h-4 text-quantum-purple" />
                  <span className="text-xs text-quantum-purple font-medium">{endpoint.status}</span>
                </div>
              </div>
            </motion.div>
          ))}
        </div>
      </div>

      {/* Pricing Tiers */}
      <div>
        <h2 className="text-2xl font-bold text-white mb-6">Pricing Tiers</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          {pricingTiers.map((tier, index) => (
            <motion.div
              key={tier.name}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: index * 0.1 }}
              className={`p-6 rounded-xl transition-all ${
                tier.highlighted
                  ? 'bg-gradient-to-br from-quantum-purple/30 to-quantum-cyan/30 border-2 border-quantum-cyan shadow-lg shadow-quantum-cyan/20'
                  : 'bg-quantum-dark/50 border border-quantum-purple/30 hover:border-quantum-cyan/50'
              }`}
            >
              {tier.highlighted && (
                <div className="mb-4">
                  <span className="px-3 py-1 bg-gradient-to-r from-quantum-cyan to-quantum-purple rounded-full text-xs font-semibold text-white">
                    POPULAR
                  </span>
                </div>
              )}
              <h3 className="text-xl font-bold text-white mb-2">{tier.name}</h3>
              <div className="text-2xl font-bold bg-gradient-to-r from-quantum-cyan to-quantum-purple bg-clip-text text-transparent mb-2">
                {tier.price}
              </div>
              <p className="text-sm text-gray-400 mb-6">{tier.description}</p>
              <ul className="space-y-2">
                {tier.features.map((feature, idx) => (
                  <li key={idx} className="flex items-start gap-2 text-sm">
                    <CheckCircle2 className="w-4 h-4 text-quantum-cyan flex-shrink-0 mt-0.5" />
                    <span className="text-gray-300">{feature}</span>
                  </li>
                ))}
              </ul>
            </motion.div>
          ))}
        </div>
      </div>

      {/* Technical Highlights */}
      <div className="p-6 bg-quantum-dark/50 border border-quantum-purple/30 rounded-xl">
        <h2 className="text-2xl font-bold text-white mb-4">Technical Highlights</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div>
            <h3 className="text-lg font-semibold text-quantum-cyan mb-3">Cryptography</h3>
            <ul className="space-y-2 text-sm text-gray-300">
              <li className="flex items-start gap-2">
                <ArrowRight className="w-4 h-4 text-quantum-cyan flex-shrink-0 mt-0.5" />
                <span>Hybrid request signatures (ECDSA + Dilithium5)</span>
              </li>
              <li className="flex items-start gap-2">
                <ArrowRight className="w-4 h-4 text-quantum-cyan flex-shrink-0 mt-0.5" />
                <span>Lattice-based ring signatures (experimental, Q2 2026)</span>
              </li>
              <li className="flex items-start gap-2">
                <ArrowRight className="w-4 h-4 text-quantum-cyan flex-shrink-0 mt-0.5" />
                <span>ZK-STARK proofs (no trusted setup, quantum-resistant)</span>
              </li>
              <li className="flex items-start gap-2">
                <ArrowRight className="w-4 h-4 text-quantum-cyan flex-shrink-0 mt-0.5" />
                <span>Epsilon-differential privacy (ε ≈ 0.7 for maximum privacy)</span>
              </li>
            </ul>
          </div>
          <div>
            <h3 className="text-lg font-semibold text-quantum-purple mb-3">Compliance</h3>
            <ul className="space-y-2 text-sm text-gray-300">
              <li className="flex items-start gap-2">
                <ArrowRight className="w-4 h-4 text-quantum-purple flex-shrink-0 mt-0.5" />
                <span>KYT/KYO sanctions screening (OFAC, UN, EU)</span>
              </li>
              <li className="flex items-start gap-2">
                <ArrowRight className="w-4 h-4 text-quantum-purple flex-shrink-0 mt-0.5" />
                <span>FATF Travel Rule (IVMS-101, TRP, OpenVASP)</span>
              </li>
              <li className="flex items-start gap-2">
                <ArrowRight className="w-4 h-4 text-quantum-purple flex-shrink-0 mt-0.5" />
                <span>ZK-attested audit trails (prove compliance without revealing data)</span>
              </li>
              <li className="flex items-start gap-2">
                <ArrowRight className="w-4 h-4 text-quantum-purple flex-shrink-0 mt-0.5" />
                <span>Threshold governance for lawful disclosure (M-of-N)</span>
              </li>
            </ul>
          </div>
        </div>
      </div>

      {/* CTA */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.5 }}
        className="text-center p-8 bg-gradient-to-r from-quantum-purple/10 via-quantum-indigo/10 to-quantum-cyan/10 border border-quantum-purple/30 rounded-xl"
      >
        <h3 className="text-2xl font-bold text-white mb-4">
          Ready to integrate privacy into your blockchain?
        </h3>
        <p className="text-gray-300 mb-6 max-w-2xl mx-auto">
          PaaS launches Q1 2026. Download the whitepaper for technical details, pricing, and integration guides.
        </p>
        <div className="flex flex-wrap justify-center gap-4">
          <a
            href="/PRIVACY_AS_A_SERVICE_WHITEPAPER.pdf"
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center gap-2 px-8 py-4 bg-gradient-to-r from-quantum-purple to-quantum-cyan rounded-lg text-white font-semibold hover:shadow-xl hover:shadow-quantum-purple/50 transition-all"
          >
            Download Full Whitepaper (v1.1)
            <ArrowRight className="w-5 h-5" />
          </a>
        </div>
      </motion.div>
    </div>
  );
}
