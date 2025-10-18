import { motion } from 'framer-motion';
import { Wallet } from 'lucide-react';

export default function WalletExamples() {
  return (
    <div className="space-y-8">
      <h1 className="text-4xl font-bold bg-gradient-to-r from-quantum-cyan via-quantum-purple to-quantum-pink bg-clip-text text-transparent mb-4">Wallet Integration</h1>
      <p className="text-xl text-gray-300">Build a complete wallet application in minutes with our simple API</p>
      <motion.div className="p-8 bg-quantum-indigo/20 backdrop-blur-xl border border-quantum-cyan/30 rounded-2xl" initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
        <div className="flex items-center gap-3 mb-4"><Wallet className="w-7 h-7 text-quantum-cyan" /><h2 className="text-2xl font-bold text-white">Complete Wallet Example</h2></div>
        <pre className="p-4 bg-quantum-dark/50 rounded-lg text-sm text-quantum-cyan border border-quantum-purple/20 overflow-x-auto">
{`// 1. Create wallet
const wallet = await fetch('http://localhost:8080/api/v1/wallets/create', { method: 'POST' }).then(r => r.json());

// 2. Get faucet tokens
await fetch('http://localhost:8080/api/v1/faucet', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ wallet_address: wallet.data.address })
});

// 3. Check balance
const balance = await fetch(\`http://localhost:8080/api/v1/wallets/\${wallet.data.address}/balance\`).then(r => r.json());

// 4. Send transaction
await fetch('http://localhost:8080/api/v1/transactions/send', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    from: wallet.data.address,
    to: 'qnk...',
    amount: 50,
    private_key: wallet.data.private_key
  })
});`}
        </pre>
      </motion.div>
    </div>
  );
}
