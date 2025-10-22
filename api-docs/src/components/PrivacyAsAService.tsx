import { motion } from 'framer-motion';
import { Shield, Lock, Eye, Globe, Zap, CheckCircle2, ArrowRight, Rocket, Code } from 'lucide-react';
import { useState } from 'react';

export default function PrivacyAsAService() {
  const [selectedChain, setSelectedChain] = useState<'bitcoin' | 'ethereum' | 'solana' | 'sui' | 'litecoin' | 'dogecoin' | 'polygon' | 'avalanche'>('bitcoin');

  const endpoints = [
    {
      method: 'POST',
      path: '/api/v1/privacy/mix/submit',
      description: 'Submit signed transactions for mixing with ring signatures and stealth addresses',
      status: 'Production'
    },
    {
      method: 'POST',
      path: '/api/v1/privacy/tor/relay',
      description: 'Route transactions through Tor network with quantum-seeded circuits',
      status: 'Production'
    },
    {
      method: 'POST',
      path: '/api/v1/privacy/ethereum/mev-protect',
      description: 'Protect Ethereum transactions from MEV exploitation via private mempool',
      status: 'Production'
    },
    {
      method: 'POST',
      path: '/api/v1/privacy/stealth-address/generate',
      description: 'Create one-time stealth addresses for enhanced privacy',
      status: 'Production'
    },
    {
      method: 'POST',
      path: '/api/v1/privacy/zk-stark/verify',
      description: 'Verify zero-knowledge STARK proofs for private transactions',
      status: 'Production'
    },
    {
      method: 'GET',
      path: '/api/v1/privacy/paas/statistics',
      description: 'Retrieve PaaS usage statistics and revenue metrics',
      status: 'Production'
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
      title: 'Client-Side Security',
      description: 'You sign transactions locally - private keys NEVER leave your machine'
    },
    {
      icon: Eye,
      title: 'Compliance-Ready',
      description: 'KYT/AML screening, IVMS-101 Travel Rule, audit trails with ZK proofs'
    },
    {
      icon: Zap,
      title: 'Enterprise SLA',
      description: '99.95% uptime guarantee, <500ms latency, 24/7 support'
    },
    {
      icon: Rocket,
      title: 'MEV Protection',
      description: 'Flashbots integration with randomized timing saves $500+ per trade'
    }
  ];

  const bitcoinExample = `import requests
import bitcoin  # pip install python-bitcoinlib

API_KEY = "your_api_key"
BASE_URL = "https://api.quillon.xyz"

def mix_bitcoin_transaction(
    from_address: str,
    to_address: str,
    amount_satoshis: int,
    private_key: str  # ⚠️ STAYS ON YOUR MACHINE
) -> dict:
    """
    Mix a Bitcoin transaction for maximum privacy.

    SECURITY: You sign transactions CLIENT-SIDE.
              Private keys NEVER leave your machine.
    """

    # Step 1: Create and SIGN transaction LOCALLY
    tx = bitcoin.Transaction()
    # ... (add inputs, outputs, etc.)

    # CRITICAL: Sign with YOUR private key on YOUR machine
    tx.sign(private_key)  # CLIENT-SIDE only
    signed_tx_hex = tx.serialize().hex()

    # Step 2: Submit SIGNED transaction to mixing service
    # We coordinate mixing but cannot steal funds
    response = requests.post(
        f"{BASE_URL}/api/v1/privacy/mix/submit",
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
            "Idempotency-Key": str(uuid.uuid4())
        },
        json={
            "chain": "bitcoin",
            "signed_transaction_hex": signed_tx_hex,
            "privacy_level": "maximum",  # epsilon < 0.7
            "options": {
                "stealth_address": True,  # One-time address
                "tor_relay": True,         # Route via Tor
                "timing_jitter": 120       # Random delay 0-120s
            }
        }
    )

    result = response.json()

    return {
        "mixed_tx_id": result["data"]["transaction_id"],
        "privacy_epsilon": result["data"]["privacy_epsilon"],
        "anonymity_set_size": result["data"]["anonymity_set"],
        "estimated_delivery": result["data"]["estimated_delivery_time"]
    }

# Example usage
result = mix_bitcoin_transaction(
    from_address="bc1qar0srrr7xfkvy5l643lydnw9re59gtzzwf5mdq",
    to_address="bc1qxy2kgdygjrsqtzq2n0yrf2493p83kkfjhx0wlh",
    amount_satoshis=10_000_000,  # 0.1 BTC
    private_key="L5oLkpV3aqBjhki6LmvChTCV6odsp4SXM6FfU2gpqgGx8aYLYUY1"
)

print(f"Transaction mixed successfully!")
print(f"Privacy level: epsilon = {result['privacy_epsilon']}")
print(f"Anonymity set: {result['anonymity_set_size']} participants")`;

  const ethereumExample = `const { ethers } = require('ethers');
const axios = require('axios');

const API_KEY = "your_api_key";
const BASE_URL = "https://api.quillon.xyz";

async function privateUniswapSwap(
  tokenIn,      // e.g., WETH address
  tokenOut,     // e.g., USDC address
  amountIn,     // Amount in Wei
  minAmountOut, // Minimum output (slippage)
  wallet        // ethers.Wallet instance
) {
  // Step 1: Build Uniswap swap transaction
  const uniswapRouter = "0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D";
  const router = new ethers.Contract(uniswapRouter, abi, wallet);

  const tx = await router.populateTransaction.swapExactTokensForTokens(
    amountIn,
    minAmountOut,
    [tokenIn, tokenOut],
    wallet.address,
    Math.floor(Date.now() / 1000) + 60 * 20 // 20 min deadline
  );

  // Step 2: Sign transaction LOCALLY (private key stays on YOUR machine)
  const signedTx = await wallet.signTransaction(tx);

  // Step 3: Submit to Q-NarwhalKnight private relay
  const response = await axios.post(
    \`\${BASE_URL}/api/v1/privacy/ethereum/mev-protect\`,
    {
      signed_transaction: signedTx,  // Already signed by YOU
      max_block_number: null,
      options: {
        tor_relay: true,          // Hide IP
        flashbots_relay: true,    // Private mempool
        simulate: true,           // Pre-execution simulation
        require_success: true     // Revert if unprofitable
      }
    },
    {
      headers: {
        'Authorization': \`Bearer \${API_KEY}\`,
        'Content-Type': 'application/json',
        'Idempotency-Key': ethers.utils.id(signedTx)
      }
    }
  );

  const result = response.data;

  return {
    transactionHash: result.data.transaction_hash,
    mevProtected: true,
    estimatedSavings: result.data.estimated_mev_savings_usd,
    blockNumber: result.data.included_in_block
  };
}

// Example usage
const provider = new ethers.providers.JsonRpcProvider(
  "https://mainnet.infura.io/v3/YOUR_KEY"
);
const wallet = new ethers.Wallet("YOUR_PRIVATE_KEY", provider);

const WETH = "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2";
const USDC = "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48";

privateUniswapSwap(
  WETH,
  USDC,
  ethers.utils.parseEther("1.0"),
  ethers.utils.parseUnits("1800", 6),
  wallet
).then(result => {
  console.log(\`Trade executed with MEV protection!\`);
  console.log(\`Estimated savings: $\${result.estimatedSavings}\`);
});`;

  const solanaExample = `const { Connection, PublicKey, Transaction, SystemProgram } = require('@solana/web3.js');
const axios = require('axios');

const API_KEY = "your_api_key";
const BASE_URL = "https://api.quillon.xyz";

async function mixSolanaTransaction(
  fromPubkey,
  toPubkey,
  amountLamports,
  wallet  // Keypair with private key - STAYS LOCAL
) {
  // Step 1: Build Solana transfer instruction
  const connection = new Connection("https://api.mainnet-beta.solana.com");

  const transaction = new Transaction().add(
    SystemProgram.transfer({
      fromPubkey: fromPubkey,
      toPubkey: toPubkey,
      lamports: amountLamports
    })
  );

  // Step 2: Get recent blockhash
  const { blockhash } = await connection.getRecentBlockhash();
  transaction.recentBlockhash = blockhash;
  transaction.feePayer = fromPubkey;

  // Step 3: Sign transaction LOCALLY (private key never sent!)
  transaction.sign(wallet);

  // Step 4: Serialize and submit to mixing API
  const serializedTx = transaction.serialize({
    requireAllSignatures: false
  }).toString('base64');

  const response = await axios.post(
    \`\${BASE_URL}/api/v1/privacy/mix/submit\`,
    {
      chain: "solana",
      transaction_base64: serializedTx,  // Already signed
      privacy_level: "standard",  // Solana: epsilon ~2.3
      recipient_address: toPubkey.toBase58(),
      amount_lamports: amountLamports,
      options: {
        temporary_accounts: 3,  // 3 intermediate accounts
        tor_relay: true,
        timing_jitter: 60
      }
    },
    {
      headers: {
        'Authorization': \`Bearer \${API_KEY}\`,
        'Content-Type': 'application/json',
        'Idempotency-Key': transaction.signature.toString('hex')
      }
    }
  );

  return response.data;
}

// Example usage
const { Keypair } = require('@solana/web3.js');
const wallet = Keypair.fromSecretKey(
  Uint8Array.from([/* your secret key */])
);

const recipient = new PublicKey(
  "9B5XszUGdMaxCZ7uSQhPzdks5ZQSmWxrmzCSvtJ6Ns6g"
);

mixSolanaTransaction(
  wallet.publicKey,
  recipient,
  1_000_000_000,  // 1 SOL
  wallet
).then(result => {
  console.log(\`Solana mixing complete!\`);
  console.log(\`Transaction signature: \${result.data.signature}\`);
  console.log(\`Privacy epsilon: \${result.data.privacy_epsilon}\`);
});`;

  const suiExample = `const { JsonRpcProvider, RawSigner, TransactionBlock } = require('@mysten/sui.js');
const axios = require('axios');

const API_KEY = "your_api_key";
const BASE_URL = "https://api.quillon.xyz";

async function mixSuiTransaction(
  senderAddress,
  recipientAddress,
  amountMist,  // SUI amount in MIST (1 SUI = 1B MIST)
  signer  // RawSigner with private key - STAYS LOCAL
) {
  // Step 1: Build SUI transfer transaction
  const provider = new JsonRpcProvider('https://fullnode.mainnet.sui.io');

  const tx = new TransactionBlock();
  const [coin] = tx.splitCoins(tx.gas, [tx.pure(amountMist)]);
  tx.transferObjects([coin], tx.pure(recipientAddress));

  // Step 2: Sign transaction LOCALLY (private key never sent!)
  const signedTx = await signer.signTransactionBlock({
    transactionBlock: tx
  });

  // Step 3: Submit signed transaction to mixing API
  const response = await axios.post(
    \`\${BASE_URL}/api/v1/privacy/mix/submit\`,
    {
      chain: "sui",
      transaction_bytes: signedTx.bytes,  // Already signed
      signature: signedTx.signature,
      privacy_level: "standard",
      recipient_address: recipientAddress,
      amount_mist: amountMist,
      options: {
        tor_relay: true,
        timing_jitter: 90,
        stealth_address: true
      }
    },
    {
      headers: {
        'Authorization': \`Bearer \${API_KEY}\`,
        'Content-Type': 'application/json',
        'Idempotency-Key': signedTx.digest
      }
    }
  );

  return response.data;
}

// Example usage
const provider = new JsonRpcProvider('https://fullnode.mainnet.sui.io');
const signer = new RawSigner(keypair, provider);

mixSuiTransaction(
  "0x1234...5678",  // Your SUI address
  "0x9876...5432",  // Recipient address
  1_000_000_000,    // 1 SUI
  signer
).then(result => {
  console.log(\`SUI mixing complete!\`);
  console.log(\`Transaction digest: \${result.data.digest}\`);
  console.log(\`Privacy epsilon: \${result.data.privacy_epsilon}\`);
});`;

  const litecoinExample = `const bitcoin = require('bitcoinjs-lib');  // Works with Litecoin too
const axios = require('axios');

const API_KEY = "your_api_key";
const BASE_URL = "https://api.quillon.xyz";

async function mixLitecoinTransaction(
  fromAddress,
  toAddress,
  amountLitoshis,  // 1 LTC = 100,000,000 litoshis
  privateKey  // WIF format - STAYS LOCAL
) {
  // Step 1: Create Litecoin transaction (similar to Bitcoin)
  const network = bitcoin.networks.litecoin;  // Litecoin network params
  const keyPair = bitcoin.ECPair.fromWIF(privateKey, network);

  const psbt = new bitcoin.Psbt({ network });
  // ... add inputs and outputs

  // Step 2: Sign transaction LOCALLY
  psbt.signAllInputs(keyPair);
  psbt.finalizeAllInputs();
  const signedTx = psbt.extractTransaction();
  const signedTxHex = signedTx.toHex();

  // Step 3: Submit to mixing service
  const response = await axios.post(
    \`\${BASE_URL}/api/v1/privacy/mix/submit\`,
    {
      chain: "litecoin",
      signed_transaction_hex: signedTxHex,  // Already signed
      privacy_level: "maximum",
      options: {
        stealth_address: true,
        tor_relay: true,
        timing_jitter: 150,
        mweb_integration: true  // Litecoin MWEB privacy
      }
    },
    {
      headers: {
        'Authorization': \`Bearer \${API_KEY}\`,
        'Content-Type': 'application/json'
      }
    }
  );

  return response.data;
}

// Example: Mix 5 LTC privately
mixLitecoinTransaction(
  "ltc1q...",  // Your Litecoin address
  "ltc1q...",  // Recipient address
  500_000_000,  // 5 LTC
  "T3...WIF"    // Your private key (local only)
).then(result => {
  console.log(\`Litecoin mixed successfully!\`);
  console.log(\`TX ID: \${result.data.transaction_id}\`);
  console.log(\`MWEB privacy: \${result.data.mweb_enabled}\`);
});`;

  const dogecoinExample = `const bitcoin = require('bitcoinjs-lib');
const axios = require('axios');

const API_KEY = "your_api_key";
const BASE_URL = "https://api.quillon.xyz";

async function mixDogecoinTransaction(
  fromAddress,
  toAddress,
  amountKoinu,  // 1 DOGE = 100,000,000 koinu (Dogecoin satoshis)
  privateKey  // WIF format - STAYS LOCAL
) {
  // Step 1: Create Dogecoin transaction
  const network = {
    messagePrefix: '\\x19Dogecoin Signed Message:\\n',
    bech32: 'doge',
    bip32: { public: 0x02facafd, private: 0x02fac398 },
    pubKeyHash: 0x1e,
    scriptHash: 0x16,
    wif: 0x9e
  };

  const keyPair = bitcoin.ECPair.fromWIF(privateKey, network);
  const psbt = new bitcoin.Psbt({ network });
  // ... add inputs and outputs

  // Step 2: Sign transaction LOCALLY (Much secure! Very privacy!)
  psbt.signAllInputs(keyPair);
  psbt.finalizeAllInputs();
  const signedTx = psbt.extractTransaction();
  const signedTxHex = signedTx.toHex();

  // Step 3: Submit to mixing service (Wow!)
  const response = await axios.post(
    \`\${BASE_URL}/api/v1/privacy/mix/submit\`,
    {
      chain: "dogecoin",
      signed_transaction_hex: signedTxHex,
      privacy_level: "standard",  // Such privacy!
      options: {
        stealth_address: true,  // Much anonymous!
        tor_relay: true,         // Very hidden!
        timing_jitter: 180       // So random!
      }
    },
    {
      headers: {
        'Authorization': \`Bearer \${API_KEY}\`,
        'Content-Type': 'application/json'
      }
    }
  );

  return response.data;
}

// Example: Mix 1000 DOGE (To the moon! 🚀)
mixDogecoinTransaction(
  "D...",  // Your DOGE address
  "D...",  // Recipient address
  100_000_000_000,  // 1000 DOGE (much amount!)
  "QW...WIF"        // Your private key (local only)
).then(result => {
  console.log(\`Such privacy! Much success!\`);
  console.log(\`Transaction ID: \${result.data.transaction_id}\`);
  console.log(\`Anonymity set: \${result.data.anonymity_set} shibes\`);
});`;

  const polygonExample = `const { ethers } = require('ethers');
const axios = require('axios');

const API_KEY = "your_api_key";
const BASE_URL = "https://api.quillon.xyz";

async function mixPolygonTransaction(
  fromAddress,
  toAddress,
  amountWei,
  wallet  // ethers.Wallet - STAYS LOCAL
) {
  // Step 1: Build Polygon (EVM-compatible) transaction
  const provider = new ethers.providers.JsonRpcProvider(
    "https://polygon-rpc.com"
  );

  const tx = {
    to: toAddress,
    value: amountWei,
    gasLimit: 21000,
    maxFeePerGas: ethers.utils.parseUnits('50', 'gwei'),
    maxPriorityFeePerGas: ethers.utils.parseUnits('30', 'gwei'),
    nonce: await provider.getTransactionCount(fromAddress),
    type: 2,  // EIP-1559
    chainId: 137  // Polygon Mainnet
  };

  // Step 2: Sign transaction LOCALLY
  const signedTx = await wallet.signTransaction(tx);

  // Step 3: Submit to privacy service
  const response = await axios.post(
    \`\${BASE_URL}/api/v1/privacy/mix/submit\`,
    {
      chain: "polygon",
      signed_transaction: signedTx,
      privacy_level: "maximum",
      options: {
        tor_relay: true,
        flashbots_relay: false,  // Not available on Polygon
        timing_jitter: 60
      }
    },
    {
      headers: {
        'Authorization': \`Bearer \${API_KEY}\`,
        'Content-Type': 'application/json'
      }
    }
  );

  return response.data;
}

// Example: Mix 100 MATIC
const provider = new ethers.providers.JsonRpcProvider(
  "https://polygon-rpc.com"
);
const wallet = new ethers.Wallet("YOUR_PRIVATE_KEY", provider);

mixPolygonTransaction(
  wallet.address,
  "0x...",  // Recipient
  ethers.utils.parseEther("100"),  // 100 MATIC
  wallet
).then(result => {
  console.log(\`Polygon mixing complete!\`);
  console.log(\`TX Hash: \${result.data.transaction_hash}\`);
  console.log(\`Gas saved vs Ethereum: ~95%\`);
});`;

  const avalancheExample = `const { ethers } = require('ethers');
const axios = require('axios');

const API_KEY = "your_api_key";
const BASE_URL = "https://api.quillon.xyz";

async function mixAvalancheTransaction(
  fromAddress,
  toAddress,
  amountWei,
  wallet  // ethers.Wallet - STAYS LOCAL
) {
  // Step 1: Build Avalanche C-Chain transaction
  const provider = new ethers.providers.JsonRpcProvider(
    "https://api.avax.network/ext/bc/C/rpc"
  );

  const tx = {
    to: toAddress,
    value: amountWei,
    gasLimit: 21000,
    gasPrice: await provider.getGasPrice(),
    nonce: await provider.getTransactionCount(fromAddress),
    chainId: 43114  // Avalanche C-Chain
  };

  // Step 2: Sign transaction LOCALLY (sub-second finality!)
  const signedTx = await wallet.signTransaction(tx);

  // Step 3: Submit to privacy mixer
  const response = await axios.post(
    \`\${BASE_URL}/api/v1/privacy/mix/submit\`,
    {
      chain: "avalanche",
      signed_transaction: signedTx,
      privacy_level: "standard",
      options: {
        tor_relay: true,
        subnet_routing: "c-chain",  // C-Chain, X-Chain, or P-Chain
        timing_jitter: 30  // Fast finality = shorter jitter
      }
    },
    {
      headers: {
        'Authorization': \`Bearer \${API_KEY}\`,
        'Content-Type': 'application/json'
      }
    }
  );

  return response.data;
}

// Example: Mix 50 AVAX
const provider = new ethers.providers.JsonRpcProvider(
  "https://api.avax.network/ext/bc/C/rpc"
);
const wallet = new ethers.Wallet("YOUR_PRIVATE_KEY", provider);

mixAvalancheTransaction(
  wallet.address,
  "0x...",  // Recipient
  ethers.utils.parseEther("50"),  // 50 AVAX
  wallet
).then(result => {
  console.log(\`Avalanche mixing complete!\`);
  console.log(\`TX Hash: \${result.data.transaction_hash}\`);
  console.log(\`Finality: <2 seconds ⚡\`);
});`;

  const chainExamples = {
    bitcoin: bitcoinExample,
    ethereum: ethereumExample,
    solana: solanaExample,
    sui: suiExample,
    litecoin: litecoinExample,
    dogecoin: dogecoinExample,
    polygon: polygonExample,
    avalanche: avalancheExample
  };

  const pricingTiers = [
    {
      name: 'Free',
      price: '$0/month',
      description: 'Individual developers',
      features: [
        '10,000 API calls/day',
        'Standard mixing (epsilon ~2.3)',
        'Community support',
        '95% uptime',
        'Rate limit: 100 req/min'
      ]
    },
    {
      name: 'Professional',
      price: '$499/month',
      description: 'Wallets, DApps, trading bots',
      features: [
        '500,000 API calls/month',
        'Maximum privacy (epsilon <0.7)',
        'Email support (24h response)',
        '99.5% uptime SLA',
        'Rate limit: 5,000 req/min'
      ],
      highlighted: true
    },
    {
      name: 'Enterprise',
      price: '$1,999/month',
      description: 'Exchanges, DeFi protocols',
      features: [
        'Unlimited API calls',
        'Priority mixing pools',
        'Phone support (1h response)',
        '99.95% uptime SLA',
        'Custom compliance configs'
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
              <p className="text-quantum-cyan text-sm font-semibold">✅ Production Ready - Live Now</p>
            </div>
          </div>
          <p className="text-lg text-gray-300 mb-6 max-w-3xl">
            Enterprise-grade quantum-resistant privacy infrastructure for any blockchain.
            Universal privacy layer with cross-chain support, compliance tools, and 99.95% SLA.
          </p>
          <div className="flex flex-wrap gap-4">
            <a
              href="/PAAS_DEVELOPER_INTEGRATION_GUIDE.pdf"
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-2 px-6 py-3 bg-gradient-to-r from-quantum-purple to-quantum-cyan rounded-lg text-white font-medium hover:shadow-lg hover:shadow-quantum-purple/50 transition-all"
            >
              <Rocket className="w-5 h-5" />
              Download Developer Guide
            </a>
            <a
              href="/PRIVACY_AS_A_SERVICE_WHITEPAPER_ENHANCED.pdf"
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-2 px-6 py-3 bg-gradient-to-r from-blue-600 to-purple-600 rounded-lg text-white font-medium hover:shadow-lg hover:shadow-blue-500/50 transition-all"
            >
              <Shield className="w-5 h-5" />
              Read Whitepaper
            </a>
            <a
              href="https://quillon.xyz/console"
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-2 px-6 py-3 bg-quantum-dark/50 border border-quantum-purple/30 rounded-lg text-white font-medium hover:bg-quantum-purple/10 transition-all"
            >
              <Code className="w-5 h-5" />
              Get API Key
            </a>
          </div>
        </div>
        <div className="absolute top-0 right-0 w-64 h-64 bg-gradient-to-br from-quantum-cyan/20 to-transparent rounded-full blur-3xl" />
        <div className="absolute bottom-0 left-0 w-64 h-64 bg-gradient-to-tr from-quantum-purple/20 to-transparent rounded-full blur-3xl" />
      </motion.div>

      {/* Security Notice */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        className="p-4 bg-quantum-cyan/10 border border-quantum-cyan/30 rounded-xl flex items-start gap-3"
      >
        <Shield className="w-5 h-5 text-quantum-cyan flex-shrink-0 mt-0.5" />
        <div>
          <h3 className="text-sm font-semibold text-quantum-cyan mb-1">Client-Side Security Model</h3>
          <p className="text-sm text-gray-300">
            <strong>Your private keys NEVER leave your machine.</strong> You sign transactions client-side,
            then send signed transactions to our API. We coordinate mixing/privacy but cannot steal funds.
          </p>
        </div>
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

      {/* API Endpoints */}
      <div>
        <h2 className="text-2xl font-bold text-white mb-6">Production API Endpoints</h2>
        <div className="space-y-3">
          {endpoints.map((endpoint, index) => (
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
                <div className="flex items-center gap-2 px-3 py-1 bg-quantum-cyan/10 border border-quantum-cyan/30 rounded-lg">
                  <CheckCircle2 className="w-4 h-4 text-quantum-cyan" />
                  <span className="text-xs text-quantum-cyan font-medium">{endpoint.status}</span>
                </div>
              </div>
            </motion.div>
          ))}
        </div>
      </div>

      {/* Code Examples */}
      <div>
        <h2 className="text-2xl font-bold text-white mb-6">Integration Examples</h2>

        {/* Chain Selector */}
        <div className="flex flex-wrap gap-2 mb-4">
          {(['bitcoin', 'ethereum', 'solana', 'sui', 'litecoin', 'dogecoin', 'polygon', 'avalanche'] as const).map((chain) => (
            <button
              key={chain}
              onClick={() => setSelectedChain(chain)}
              className={`px-4 py-2 rounded-lg text-sm font-medium transition-all ${
                selectedChain === chain
                  ? 'bg-gradient-to-r from-quantum-purple to-quantum-cyan text-white'
                  : 'bg-quantum-dark/50 border border-quantum-purple/30 text-gray-400 hover:text-white'
              }`}
            >
              {chain.charAt(0).toUpperCase() + chain.slice(1)}
            </button>
          ))}
        </div>

        {/* Code Display */}
        <div className="relative">
          <div className="absolute top-4 right-4 px-3 py-1 bg-quantum-purple/20 border border-quantum-purple/30 rounded-lg text-xs text-quantum-purple font-mono">
            {(selectedChain === 'bitcoin' || selectedChain === 'litecoin') && 'Python'}
            {(selectedChain === 'ethereum' || selectedChain === 'solana' || selectedChain === 'sui' ||
              selectedChain === 'polygon' || selectedChain === 'avalanche') && 'JavaScript'}
            {selectedChain === 'dogecoin' && 'JavaScript (Much Wow!)'}
          </div>
          <pre className="bg-quantum-dark/80 border border-quantum-purple/30 rounded-xl p-6 overflow-x-auto text-sm">
            <code className="text-gray-300 font-mono">{chainExamples[selectedChain]}</code>
          </pre>
        </div>
      </div>

      {/* Pricing */}
      <div>
        <h2 className="text-2xl font-bold text-white mb-6">Pricing Tiers</h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
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

      {/* CTA */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.5 }}
        className="text-center p-8 bg-gradient-to-r from-quantum-purple/10 via-quantum-indigo/10 to-quantum-cyan/10 border border-quantum-purple/30 rounded-xl"
      >
        <h3 className="text-2xl font-bold text-white mb-4">
          Ready to integrate privacy into your blockchain application?
        </h3>
        <p className="text-gray-300 mb-6 max-w-2xl mx-auto">
          Get started today with our comprehensive developer guide and production-ready API endpoints.
        </p>
        <div className="flex flex-wrap justify-center gap-4">
          <a
            href="/PAAS_DEVELOPER_INTEGRATION_GUIDE.pdf"
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center gap-2 px-8 py-4 bg-gradient-to-r from-quantum-purple to-quantum-cyan rounded-lg text-white font-semibold hover:shadow-xl hover:shadow-quantum-purple/50 transition-all"
          >
            Download Full Developer Guide
            <ArrowRight className="w-5 h-5" />
          </a>
        </div>
      </motion.div>
    </div>
  );
}
