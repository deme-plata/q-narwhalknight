import { useState } from 'react';
import { motion } from 'framer-motion';
import { Code2, Copy, CheckCircle2, ChevronDown, ChevronRight, Lock, Shield } from 'lucide-react';

interface EndpointProps {
  method: string;
  path: string;
  description: string;
  request?: string;
  response: string;
  example: string;
  requiresAuth?: boolean;
  authType?: string;
}

function Endpoint({ method, path, description, request, response, example, requiresAuth, authType }: EndpointProps) {
  const [expanded, setExpanded] = useState(false);
  const [copied, setCopied] = useState(false);

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const getMethodColor = (method: string) => {
    const colors: Record<string, string> = {
      GET: 'from-quantum-cyan to-quantum-cyan/70',
      POST: 'from-quantum-green to-quantum-green/70',
      PUT: 'from-quantum-purple to-quantum-purple/70',
      DELETE: 'from-quantum-pink to-quantum-pink/70',
    };
    return colors[method] || 'from-gray-500 to-gray-600';
  };

  return (
    <motion.div className="border border-quantum-purple/30 rounded-xl overflow-hidden bg-quantum-indigo/10 backdrop-blur-xl" initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}>
      <button onClick={() => setExpanded(!expanded)} className="w-full px-6 py-4 flex items-center justify-between hover:bg-quantum-purple/5 transition-all">
        <div className="flex items-center gap-4 flex-wrap">
          <div className={`px-3 py-1 bg-gradient-to-r ${getMethodColor(method)} rounded-lg text-white font-bold text-sm`}>{method}</div>
          <code className="text-quantum-cyan font-mono text-sm md:text-base">{path}</code>
          {requiresAuth && (
            <div className="flex items-center gap-1 px-2 py-1 bg-quantum-purple/20 rounded-md border border-quantum-purple/30">
              <Lock className="w-3 h-3 text-quantum-purple" />
              <span className="text-xs text-quantum-purple font-medium">{authType || 'Auth Required'}</span>
            </div>
          )}
        </div>
        {expanded ? <ChevronDown className="w-5 h-5 text-gray-400" /> : <ChevronRight className="w-5 h-5 text-gray-400" />}
      </button>
      {expanded && (
        <motion.div initial={{ opacity: 0, height: 0 }} animate={{ opacity: 1, height: 'auto' }} className="px-6 pb-6 space-y-4">
          <p className="text-gray-300">{description}</p>
          {requiresAuth && (
            <div className="p-4 bg-quantum-purple/10 rounded-lg border border-quantum-purple/30">
              <div className="flex items-center gap-2 mb-2">
                <Shield className="w-4 h-4 text-quantum-purple" />
                <h4 className="text-white font-bold text-sm">Authentication Required</h4>
              </div>
              <p className="text-gray-400 text-sm">{authType || 'This endpoint requires wallet signature authentication via the X-Wallet-Auth header.'}</p>
            </div>
          )}
          {request && (<div><h4 className="text-white font-bold mb-2">Request</h4><pre className="p-4 bg-quantum-dark/50 rounded-lg text-sm text-quantum-green border border-quantum-purple/20 overflow-x-auto">{request}</pre></div>)}
          <div><h4 className="text-white font-bold mb-2">Response</h4><pre className="p-4 bg-quantum-dark/50 rounded-lg text-sm text-quantum-cyan border border-quantum-purple/20 overflow-x-auto">{response}</pre></div>
          <div><button onClick={() => copyToClipboard(example)} className="flex items-center gap-2 px-3 py-1 text-sm text-gray-400 hover:text-white">{copied ? <CheckCircle2 className="w-4 h-4" /> : <Copy className="w-4 h-4" />}{copied ? 'Copied' : 'Copy'}</button><pre className="p-4 bg-quantum-dark/50 rounded-lg text-sm text-quantum-pink border border-quantum-purple/20 overflow-x-auto">{example}</pre></div>
        </motion.div>
      )}
    </motion.div>
  );
}

export default function APIEndpoints() {
  const endpoints: EndpointProps[] = [
    // Status & Health
    {
      method: 'GET',
      path: '/api/v1/status',
      description: 'Get node status, TPS metrics, and network health',
      response: '{ "success": true, "data": { "status": "running", "tps": 48000, "peers": 12 } }',
      example: 'curl https://quillon.xyz/api/v1/status'
    },

    // Wallet Management
    {
      method: 'POST',
      path: '/api/v1/wallets/create',
      description: 'Create new quantum-resistant wallet with Dilithium5 keys',
      response: '{ "success": true, "data": { "address": "qnk470294cd0...", "private_key": "...", "mnemonic": "word1 word2..." } }',
      example: 'curl -X POST https://quillon.xyz/api/v1/wallets/create'
    },
    {
      method: 'GET',
      path: '/api/v1/wallets/{address}/balance',
      description: 'Get wallet QUG balance (now requires authentication)',
      response: '{ "success": true, "data": { "balance": 1000000000, "balance_qug": "10.00000000" } }',
      example: 'curl https://quillon.xyz/api/v1/wallets/qnk470294cd0.../balance \\\n  -H "X-Wallet-Auth: wallet=qnk...; signature=0x...; message=timestamp:1234567890"',
      requiresAuth: true,
      authType: 'Wallet Signature'
    },
    {
      method: 'POST',
      path: '/api/v1/wallets/{address}/token-balances',
      description: 'Get all token balances for a wallet (QUG, QUGUSD, custom tokens)',
      response: '{ "success": true, "data": { "QUG": 1000, "QUGUSD": 500, "MyToken": 250 } }',
      example: 'curl -X POST https://quillon.xyz/api/v1/wallets/qnk470294cd0.../token-balances \\\n  -H "X-Wallet-Auth: wallet=qnk...; signature=0x...; message=timestamp:1234567890"',
      requiresAuth: true,
      authType: 'Wallet Signature'
    },

    // Faucet
    {
      method: 'POST',
      path: '/api/v1/faucet',
      description: 'Request free QUG tokens (dev/testnet)',
      request: '{ "wallet_address": "qnk470294cd0..." }',
      response: '{ "success": true, "data": { "amount": 10000000000, "tx_hash": "0x..." } }',
      example: 'curl -X POST https://quillon.xyz/api/v1/faucet \\\n  -H "Content-Type: application/json" \\\n  -d \'{"wallet_address":"qnk470294cd0..."}\''
    },

    // Transactions
    {
      method: 'POST',
      path: '/api/v1/transactions/submit',
      description: 'Submit signed transaction to the network',
      request: '{ "from": "qnk...", "to": "qnk...", "amount": 1000000000, "signature": "0x..." }',
      response: '{ "success": true, "data": { "tx_hash": "0x...", "status": "pending" } }',
      example: 'curl -X POST https://quillon.xyz/api/v1/transactions/submit \\\n  -H "Content-Type: application/json" \\\n  -d \'{"from":"qnk...","to":"qnk...","amount":1000000000,"signature":"0x..."}\''
    },
    {
      method: 'GET',
      path: '/api/v1/transactions/{hash}',
      description: 'Get transaction details by hash',
      response: '{ "success": true, "data": { "hash": "0x...", "from": "qnk...", "to": "qnk...", "status": "confirmed" } }',
      example: 'curl https://quillon.xyz/api/v1/transactions/0x...'
    },
    {
      method: 'POST',
      path: '/api/v1/wallets/{address}/transactions/recent',
      description: 'Get recent transactions for a wallet (requires auth)',
      response: '{ "success": true, "data": [{ "hash": "0x...", "type": "send", "amount": 1000 }] }',
      example: 'curl -X POST https://quillon.xyz/api/v1/wallets/qnk.../transactions/recent \\\n  -H "X-Wallet-Auth: wallet=qnk...; signature=0x...; message=timestamp:1234567890"',
      requiresAuth: true,
      authType: 'Wallet Signature'
    },

    // DEX & Swaps
    {
      method: 'POST',
      path: '/api/v1/dex/swap',
      description: 'Execute token swap through liquidity pools',
      request: '{ "from_token": "QUG", "to_token": "QUGUSD", "amount_in": 100000000, "min_amount_out": 99000000, "wallet_address": "qnk..." }',
      response: '{ "success": true, "data": { "amount_out": 99500000, "price_impact": 0.5, "tx_hash": "0x..." } }',
      example: 'curl -X POST https://quillon.xyz/api/v1/dex/swap \\\n  -H "Content-Type: application/json" \\\n  -H "X-Wallet-Auth: wallet=qnk...; signature=0x...; message=timestamp:1234567890" \\\n  -d \'{"from_token":"QUG","to_token":"QUGUSD","amount_in":100000000,"min_amount_out":99000000,"wallet_address":"qnk..."}\'',
      requiresAuth: true,
      authType: 'Wallet Signature'
    },
    {
      method: 'GET',
      path: '/api/v1/dex/pools',
      description: 'Get all liquidity pools with TVL and APR',
      response: '{ "success": true, "data": [{ "pair": "QUG/QUGUSD", "tvl": 1000000, "apr": 15.5 }] }',
      example: 'curl https://quillon.xyz/api/v1/dex/pools'
    },

    // Stripe Payment Integration (NEW)
    {
      method: 'POST',
      path: '/api/v1/payment/create-intent',
      description: 'Create Stripe payment intent to top up USD wallet balance',
      request: '{ "wallet_address": "qnk...", "amount": "10.00" }',
      response: '{ "success": true, "data": { "payment_intent_id": "pi_...", "client_secret": "...", "amount": 1000 } }',
      example: 'curl -X POST https://quillon.xyz/api/v1/payment/create-intent \\\n  -H "Content-Type: application/json" \\\n  -d \'{"wallet_address":"qnk...","amount":"10.00"}\''
    },
    {
      method: 'POST',
      path: '/api/v1/payment/balance',
      description: 'Get USD balance from Stripe wallet',
      request: '{ "wallet_address": "qnk..." }',
      response: '{ "success": true, "data": { "wallet_address": "qnk...", "balance_usd": "10.00", "balance_cents": 1000 } }',
      example: 'curl -X POST https://quillon.xyz/api/v1/payment/balance \\\n  -H "Content-Type: application/json" \\\n  -d \'{"wallet_address":"qnk..."}\''
    },
    {
      method: 'POST',
      path: '/api/v1/payment/convert-to-qugusd',
      description: 'Convert USD to QUGUSD stablecoin (1:1 with 0.1% fee)',
      request: '{ "wallet_address": "qnk...", "usd_amount": "10.00" }',
      response: '{ "success": true, "data": { "usd_deducted": "10.00", "qugusd_minted": "9.99", "conversion_fee": "0.01" } }',
      example: 'curl -X POST https://quillon.xyz/api/v1/payment/convert-to-qugusd \\\n  -H "Content-Type: application/json" \\\n  -d \'{"wallet_address":"qnk...","usd_amount":"10.00"}\''
    },
    {
      method: 'POST',
      path: '/api/v1/payment/transfer',
      description: 'Transfer USD from one wallet to another (wallet-to-wallet)',
      request: '{ "from_wallet": "qnk...", "to_wallet": "qnk...", "amount_usd": "5.00" }',
      response: '{ "success": true, "data": { "amount_transferred": "5.00", "transaction_id": "...", "from_new_balance": "5.00" } }',
      example: 'curl -X POST https://quillon.xyz/api/v1/payment/transfer \\\n  -H "Content-Type: application/json" \\\n  -d \'{"from_wallet":"qnk...","to_wallet":"qnk...","amount_usd":"5.00"}\''
    },

    // Smart Contracts
    {
      method: 'POST',
      path: '/api/v1/contracts/deploy',
      description: 'Deploy WASM smart contract to the network',
      request: '{ "wasm_code": "0x...", "constructor_args": [], "owner": "qnk..." }',
      response: '{ "success": true, "data": { "contract_address": "qnk...", "tx_hash": "0x..." } }',
      example: 'curl -X POST https://quillon.xyz/api/v1/contracts/deploy \\\n  -H "Content-Type: application/json" \\\n  -d \'{"wasm_code":"0x...","owner":"qnk..."}\'',
      requiresAuth: true,
      authType: 'Wallet Signature'
    },
  ];

  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-4xl font-bold bg-gradient-to-r from-quantum-cyan via-quantum-purple to-quantum-pink bg-clip-text text-transparent mb-4">REST API Endpoints</h1>
        <p className="text-gray-400">
          Complete API reference with authentication requirements and new USD payment features.
        </p>
      </div>

      <div className="p-6 bg-quantum-indigo/20 backdrop-blur-xl border border-quantum-cyan/30 rounded-2xl">
        <div className="flex items-center gap-3 mb-2">
          <Code2 className="w-6 h-6 text-quantum-cyan" />
          <h3 className="text-lg font-bold text-white">Base URL</h3>
        </div>
        <code className="block p-3 bg-quantum-dark/50 rounded-lg text-quantum-cyan border border-quantum-purple/20">
          https://quillon.xyz
        </code>
        <p className="text-gray-400 text-sm mt-2">
          Production: <code className="text-quantum-cyan">https://quillon.xyz</code>
        </p>
      </div>

      <div className="p-6 bg-quantum-purple/10 backdrop-blur-xl border border-quantum-purple/30 rounded-2xl">
        <div className="flex items-center gap-3 mb-3">
          <Shield className="w-6 h-6 text-quantum-purple" />
          <h3 className="text-lg font-bold text-white">Authentication</h3>
        </div>
        <p className="text-gray-400 mb-3">
          Endpoints marked with <Lock className="w-3 h-3 inline text-quantum-purple" /> require wallet signature authentication.
        </p>
        <div className="space-y-2 text-sm">
          <div className="p-3 bg-quantum-dark/30 rounded-lg">
            <p className="text-white font-medium mb-1">Header Format:</p>
            <code className="text-quantum-cyan text-xs">
              X-Wallet-Auth: wallet=qnk...; signature=0x...; message=timestamp:1234567890
            </code>
          </div>
          <div className="p-3 bg-quantum-dark/30 rounded-lg">
            <p className="text-white font-medium mb-1">Signature:</p>
            <p className="text-gray-400 text-xs">
              Sign the message "timestamp:&lt;unix_timestamp&gt;" with your wallet's private key (Dilithium5 or Ed25519)
            </p>
          </div>
        </div>
      </div>

      <div className="space-y-4">
        <h2 className="text-2xl font-bold text-white">Available Endpoints</h2>
        {endpoints.map((endpoint, i) => (<Endpoint key={i} {...endpoint} />))}
      </div>

      <div className="p-6 bg-gradient-to-br from-quantum-green/10 to-quantum-cyan/10 backdrop-blur-xl border border-quantum-green/30 rounded-2xl">
        <h3 className="text-lg font-bold text-white mb-2">✨ New USD Payment Features</h3>
        <ul className="space-y-2 text-gray-300 text-sm">
          <li className="flex items-start gap-2">
            <CheckCircle2 className="w-4 h-4 text-quantum-green mt-0.5 flex-shrink-0" />
            <span><strong>USD → QUGUSD Conversion:</strong> Convert Stripe USD balance to on-chain QUGUSD (1:1 minus 0.1% fee)</span>
          </li>
          <li className="flex items-start gap-2">
            <CheckCircle2 className="w-4 h-4 text-quantum-green mt-0.5 flex-shrink-0" />
            <span><strong>USD Transfers:</strong> Send USD between wallets instantly (off-chain)</span>
          </li>
          <li className="flex items-start gap-2">
            <CheckCircle2 className="w-4 h-4 text-quantum-green mt-0.5 flex-shrink-0" />
            <span><strong>USD Swaps:</strong> Use USD for DEX swaps (auto-converts to QUGUSD first)</span>
          </li>
        </ul>
      </div>
    </div>
  );
}
