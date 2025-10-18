import { motion } from 'framer-motion';
import { Zap, Radio } from 'lucide-react';

export default function WebSocketGuide() {
  return (
    <div className="space-y-8">
      <h1 className="text-4xl font-bold bg-gradient-to-r from-quantum-cyan via-quantum-purple to-quantum-pink bg-clip-text text-transparent mb-4">WebSocket Streaming</h1>
      <p className="text-xl text-gray-300">Real-time updates for balances, transactions, and network events</p>
      <motion.div className="p-8 bg-quantum-indigo/20 backdrop-blur-xl border border-quantum-cyan/30 rounded-2xl" initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
        <div className="flex items-center gap-3 mb-4"><Zap className="w-7 h-7 text-quantum-cyan" /><h2 className="text-2xl font-bold text-white">WebSocket Connection</h2></div>
        <pre className="p-4 bg-quantum-dark/50 rounded-lg text-sm text-quantum-cyan border border-quantum-purple/20 overflow-x-auto">
{`const ws = new WebSocket('ws://localhost:8080/ws');

// Subscribe to balance updates
ws.send(JSON.stringify({
  type: 'subscribe',
  channel: 'balance',
  address: 'qnk...'
}));

// Listen for updates
ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log('New balance:', data.balance);
};`}
        </pre>
      </motion.div>
      <motion.div className="p-8 bg-quantum-purple/10 backdrop-blur-xl border border-quantum-purple/30 rounded-2xl" initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
        <div className="flex items-center gap-3 mb-4"><Radio className="w-7 h-7 text-quantum-purple" /><h2 className="text-2xl font-bold text-white">Available Channels</h2></div>
        <ul className="space-y-2 text-gray-300">
          <li>• <code className="text-quantum-cyan">balance</code> - Real-time balance updates</li>
          <li>• <code className="text-quantum-cyan">transactions</code> - New transaction notifications</li>
          <li>• <code className="text-quantum-cyan">network</code> - Network status and peer count</li>
        </ul>
      </motion.div>
    </div>
  );
}
