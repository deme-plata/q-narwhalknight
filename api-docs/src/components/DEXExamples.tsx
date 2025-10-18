import { motion } from 'framer-motion';
import { Network, TrendingUp } from 'lucide-react';

export default function DEXExamples() {
  return (
    <div className="space-y-8">
      <h1 className="text-4xl font-bold bg-gradient-to-r from-quantum-cyan via-quantum-purple to-quantum-pink bg-clip-text text-transparent mb-4">DEX Building Guide</h1>
      <p className="text-xl text-gray-300">Create a decentralized exchange with instant finality and high throughput</p>
      <motion.div className="p-8 bg-quantum-purple/20 backdrop-blur-xl border border-quantum-purple/30 rounded-2xl" initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
        <div className="flex items-center gap-3 mb-4"><Network className="w-7 h-7 text-quantum-purple" /><h2 className="text-2xl font-bold text-white">Trading Pair Implementation</h2></div>
        <pre className="p-4 bg-quantum-dark/50 rounded-lg text-sm text-quantum-cyan border border-quantum-purple/20 overflow-x-auto">
{`class DEX {
  async createOrder(from, to, amount, price) {
    // Create liquidity pool transaction
    const tx = await fetch('http://localhost:8080/api/v1/transactions/send', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ from, to, amount, private_key: '...' })
    });
    return tx.json();
  }

  async getPrice(pair) {
    // Query on-chain oracle
    const balance = await fetch(\`http://localhost:8080/api/v1/wallets/\${pair}/balance\`);
    return balance.json();
  }
}`}
        </pre>
      </motion.div>
      <motion.div className="p-8 bg-quantum-cyan/10 backdrop-blur-xl border border-quantum-cyan/30 rounded-2xl" initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
        <div className="flex items-center gap-3 mb-4"><TrendingUp className="w-7 h-7 text-quantum-cyan" /><h2 className="text-2xl font-bold text-white">Key Benefits</h2></div>
        <ul className="space-y-2 text-gray-300">
          <li>• Sub-50ms finality - No waiting for confirmations</li>
          <li>• 1M+ TPS - Handle high-frequency trading</li>
          <li>• Post-quantum secure - Future-proof your DEX</li>
        </ul>
      </motion.div>
    </div>
  );
}
