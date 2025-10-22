/**
 * Q-NarwhalKnight Privacy-as-a-Service - Production Solana SDK
 *
 * PRODUCTION-READY Solana integration with proper account management,
 * recent blockhash handling, priority fees, and comprehensive error handling.
 *
 * SECURITY NOTE: This handles real Solana transactions. Always test on devnet first!
 *
 * Installation:
 *   npm install @solana/web3.js axios uuid bs58
 *
 * Usage:
 *   const { QNarwhalKnightPaaSClient, SolanaWallet } = require('./q_paas_solana_production');
 *
 *   const client = new QNarwhalKnightPaaSClient('your_api_key');
 *   const wallet = new SolanaWallet(secretKey, rpcUrl);
 *
 *   const result = await client.mixSolanaTransaction(wallet, recipient, amount);
 */

const {
    Connection,
    PublicKey,
    Transaction,
    SystemProgram,
    LAMPORTS_PER_SOL,
    sendAndConfirmTransaction,
    Keypair,
    ComputeBudgetProgram
} = require('@solana/web3.js');
const axios = require('axios');
const { v4: uuidv4 } = require('uuid');
const bs58 = require('bs58');

/**
 * Privacy levels for Solana transactions
 */
const PrivacyLevel = {
    STANDARD: 'standard',      // ε ~2.3
    HIGH: 'high',              // ε ~1.5
    MAXIMUM: 'maximum',        // ε < 0.7
    PARANOID: 'paranoid'       // ε < 0.3 (very slow)
};

/**
 * Production Solana wallet with proper account management
 */
class SolanaWallet {
    constructor(secretKey, rpcUrl = 'https://api.mainnet-beta.solana.com') {
        // secretKey can be Uint8Array, number array, or base58 string
        if (typeof secretKey === 'string') {
            this.keypair = Keypair.fromSecretKey(bs58.decode(secretKey));
        } else if (Array.isArray(secretKey)) {
            this.keypair = Keypair.fromSecretKey(Uint8Array.from(secretKey));
        } else {
            this.keypair = Keypair.fromSecretKey(secretKey);
        }

        this.publicKey = this.keypair.publicKey;
        this.connection = new Connection(rpcUrl, 'confirmed');
    }

    /**
     * Get wallet balance in lamports
     */
    async getBalance() {
        return await this.connection.getBalance(this.publicKey);
    }

    /**
     * Get wallet balance in SOL
     */
    async getBalanceSOL() {
        const lamports = await this.getBalance();
        return lamports / LAMPORTS_PER_SOL;
    }

    /**
     * Get recent blockhash with retry logic
     */
    async getRecentBlockhash() {
        let retries = 3;
        while (retries > 0) {
            try {
                const { blockhash, lastValidBlockHeight } = await this.connection.getLatestBlockhash('confirmed');
                return { blockhash, lastValidBlockHeight };
            } catch (error) {
                retries--;
                if (retries === 0) throw error;
                await new Promise(resolve => setTimeout(resolve, 1000));
            }
        }
    }

    /**
     * Estimate transaction fees with priority fee
     */
    async estimateFees(transaction, priorityLevel = 'medium') {
        try {
            // Get recent prioritization fees
            const recentFees = await this.connection.getRecentPrioritizationFees();

            let priorityFee = 0;
            if (recentFees.length > 0) {
                const fees = recentFees.map(f => f.prioritizationFee).sort((a, b) => a - b);
                const medianIndex = Math.floor(fees.length / 2);

                switch (priorityLevel) {
                    case 'low':
                        priorityFee = fees[Math.floor(fees.length * 0.25)];
                        break;
                    case 'medium':
                        priorityFee = fees[medianIndex];
                        break;
                    case 'high':
                        priorityFee = fees[Math.floor(fees.length * 0.75)];
                        break;
                    case 'veryHigh':
                        priorityFee = fees[fees.length - 1];
                        break;
                    default:
                        priorityFee = fees[medianIndex];
                }
            }

            // Base fee is 5000 lamports per signature
            const baseFee = 5000 * transaction.signatures.length;

            return {
                baseFee,
                priorityFee: priorityFee || 5000, // Default 5000 if no recent fees
                total: baseFee + priorityFee
            };

        } catch (error) {
            console.error('Fee estimation failed:', error.message);
            // Fallback to conservative estimate
            return { baseFee: 5000, priorityFee: 10000, total: 15000 };
        }
    }

    /**
     * Build and sign transaction with proper recent blockhash and fees
     */
    async buildAndSignTransaction(instructions, options = {}) {
        const {
            priorityLevel = 'medium',
            computeUnitLimit = 200000,
            signers = []
        } = options;

        try {
            // Get recent blockhash
            const { blockhash, lastValidBlockHeight } = await this.getRecentBlockhash();

            // Add compute budget instructions for priority fees
            const computeBudgetIx = ComputeBudgetProgram.setComputeUnitLimit({
                units: computeUnitLimit
            });

            // Get priority fee
            const recentFees = await this.connection.getRecentPrioritizationFees();
            let priorityFee = 5000; // Default
            if (recentFees.length > 0) {
                const fees = recentFees.map(f => f.prioritizationFee).sort((a, b) => a - b);
                priorityFee = fees[Math.floor(fees.length * 0.5)] || 5000;
            }

            const computePriceIx = ComputeBudgetProgram.setComputeUnitPrice({
                microLamports: Math.floor(priorityFee)
            });

            // Build transaction
            const transaction = new Transaction({
                recentBlockhash: blockhash,
                lastValidBlockHeight,
                feePayer: this.publicKey
            });

            transaction.add(computeBudgetIx);
            transaction.add(computePriceIx);
            transaction.add(...instructions);

            // Sign transaction
            transaction.sign(this.keypair, ...signers);

            return {
                transaction,
                serialized: transaction.serialize(),
                blockhash,
                lastValidBlockHeight
            };

        } catch (error) {
            throw new Error(`Failed to build transaction: ${error.message}`);
        }
    }

    /**
     * Send and confirm transaction with retry logic
     */
    async sendAndConfirmTransaction(transaction, options = {}) {
        const {
            maxRetries = 3,
            skipPreflight = false,
            preflightCommitment = 'confirmed',
            commitment = 'confirmed'
        } = options;

        let retries = maxRetries;
        let lastError;

        while (retries > 0) {
            try {
                const signature = await this.connection.sendRawTransaction(
                    transaction.serialize(),
                    {
                        skipPreflight,
                        preflightCommitment,
                        maxRetries: 0 // We handle retries ourselves
                    }
                );

                // Wait for confirmation
                const confirmation = await this.connection.confirmTransaction(
                    {
                        signature,
                        blockhash: transaction.recentBlockhash,
                        lastValidBlockHeight: transaction.lastValidBlockHeight
                    },
                    commitment
                );

                if (confirmation.value.err) {
                    throw new Error(`Transaction failed: ${JSON.stringify(confirmation.value.err)}`);
                }

                return signature;

            } catch (error) {
                lastError = error;
                retries--;

                if (retries > 0) {
                    console.log(`Transaction failed, retrying... (${retries} retries left)`);
                    await new Promise(resolve => setTimeout(resolve, 2000));
                }
            }
        }

        throw new Error(`Transaction failed after ${maxRetries} retries: ${lastError.message}`);
    }

    /**
     * Get SPL token balance
     */
    async getTokenBalance(tokenMintAddress) {
        try {
            const tokenAccounts = await this.connection.getParsedTokenAccountsByOwner(
                this.publicKey,
                { mint: new PublicKey(tokenMintAddress) }
            );

            if (tokenAccounts.value.length === 0) {
                return 0;
            }

            return tokenAccounts.value[0].account.data.parsed.info.tokenAmount.uiAmount;
        } catch (error) {
            throw new Error(`Failed to get token balance: ${error.message}`);
        }
    }
}

/**
 * Production-ready Q-NarwhalKnight PaaS client for Solana
 */
class QNarwhalKnightPaaSClient {
    constructor(apiKey, baseUrl = 'http://localhost:8080', options = {}) {
        this.apiKey = apiKey;
        this.baseUrl = baseUrl.replace(/\/$/, '');
        this.timeout = options.timeout || 30000;
        this.maxRetries = options.maxRetries || 3;

        // Create axios instance with retry logic
        this.client = axios.create({
            baseURL: this.baseUrl,
            timeout: this.timeout,
            headers: {
                'Authorization': `Bearer ${this.apiKey}`,
                'Content-Type': 'application/json'
            }
        });

        // Add retry interceptor
        this._setupRetryInterceptor();
    }

    _setupRetryInterceptor() {
        this.client.interceptors.response.use(
            response => response,
            async error => {
                const config = error.config;

                if (!config || !config.retry) {
                    config.retry = 0;
                }

                // Retry on network errors or 5xx errors
                const shouldRetry =
                    !error.response ||
                    error.response.status >= 500 ||
                    error.response.status === 429;

                if (shouldRetry && config.retry < this.maxRetries) {
                    config.retry += 1;

                    // Exponential backoff: 2^n seconds
                    const delay = Math.pow(2, config.retry) * 1000;
                    await new Promise(resolve => setTimeout(resolve, delay));

                    return this.client(config);
                }

                return Promise.reject(error);
            }
        );
    }

    /**
     * Mix Solana transaction for privacy
     */
    async mixSolanaTransaction(
        wallet,
        recipientAddress,
        amountLamports,
        options = {}
    ) {
        const {
            privacyLevel = PrivacyLevel.STANDARD,
            torRelay = true,
            timingJitterSeconds = 120,
            useTemporaryAccounts = true,
            priorityLevel = 'medium'
        } = options;

        try {
            // Step 1: Check balance
            const balance = await wallet.getBalance();
            const estimatedFees = 15000; // Conservative estimate

            if (balance < amountLamports + estimatedFees) {
                throw new Error(`Insufficient balance: have ${balance} lamports, need ${amountLamports + estimatedFees}`);
            }

            // Step 2: Build transfer instruction
            const transferIx = SystemProgram.transfer({
                fromPubkey: wallet.publicKey,
                toPubkey: new PublicKey(recipientAddress),
                lamports: amountLamports
            });

            // Step 3: Build and sign transaction
            const { transaction, serialized } = await wallet.buildAndSignTransaction(
                [transferIx],
                { priorityLevel }
            );

            // Step 4: Submit to Q-NarwhalKnight mixing service
            const idempotencyKey = uuidv4();

            const response = await this.client.post(
                '/api/v1/privacy/mix/submit',
                {
                    chain: 'solana',
                    signed_transaction_base64: serialized.toString('base64'),
                    privacy_level: privacyLevel,
                    options: {
                        tor_relay: torRelay,
                        timing_jitter_seconds: timingJitterSeconds,
                        use_temporary_accounts: useTemporaryAccounts
                    }
                },
                {
                    headers: {
                        'Idempotency-Key': idempotencyKey
                    }
                }
            );

            const result = response.data;

            if (!result.success) {
                throw new Error(`Transaction mixing failed: ${result.error || 'Unknown error'}`);
            }

            return {
                mixingId: result.data.mixing_id,
                transactionHash: result.data.transaction_hash,
                estimatedCompletionTime: result.data.estimated_completion_time,
                privacyEpsilon: result.data.privacy_epsilon,
                anonymitySet: result.data.anonymity_set,
                idempotencyKey
            };

        } catch (error) {
            if (error.response) {
                throw new Error(`API Error (${error.response.status}): ${JSON.stringify(error.response.data)}`);
            } else if (error.request) {
                throw new Error('Network error: No response from server');
            } else {
                throw error;
            }
        }
    }

    /**
     * Mix SPL token transfer
     */
    async mixSPLTokenTransfer(
        wallet,
        tokenMintAddress,
        recipientAddress,
        amount,
        options = {}
    ) {
        const {
            privacyLevel = PrivacyLevel.MAXIMUM,
            torRelay = true,
            timingJitterSeconds = 180
        } = options;

        try {
            // Step 1: Check token balance
            const balance = await wallet.getTokenBalance(tokenMintAddress);
            if (balance < amount) {
                throw new Error(`Insufficient token balance: have ${balance}, need ${amount}`);
            }

            // Step 2: Get or create associated token accounts
            // (In production, use @solana/spl-token package for this)
            // For now, submit to API which handles account creation

            const idempotencyKey = uuidv4();

            const response = await this.client.post(
                '/api/v1/privacy/solana/mix-spl-token',
                {
                    token_mint: tokenMintAddress,
                    recipient: recipientAddress,
                    amount: amount.toString(),
                    privacy_level: privacyLevel,
                    options: {
                        tor_relay: torRelay,
                        timing_jitter_seconds: timingJitterSeconds
                    }
                },
                {
                    headers: {
                        'Idempotency-Key': idempotencyKey
                    }
                }
            );

            const result = response.data;

            if (!result.success) {
                throw new Error(`SPL token mixing failed: ${result.error || 'Unknown error'}`);
            }

            return {
                mixingId: result.data.mixing_id,
                estimatedCompletionTime: result.data.estimated_completion_time,
                privacyEpsilon: result.data.privacy_epsilon,
                idempotencyKey
            };

        } catch (error) {
            if (error.response) {
                throw new Error(`API Error (${error.response.status}): ${JSON.stringify(error.response.data)}`);
            } else if (error.request) {
                throw new Error('Network error: No response from server');
            } else {
                throw error;
            }
        }
    }

    /**
     * Check mixing status
     */
    async checkMixingStatus(mixingId) {
        try {
            const response = await this.client.get(`/api/v1/privacy/mix/status/${mixingId}`);
            return response.data;
        } catch (error) {
            throw new Error(`Failed to check mixing status: ${error.message}`);
        }
    }
}

// Export for use
module.exports = {
    QNarwhalKnightPaaSClient,
    SolanaWallet,
    PrivacyLevel
};

/**
 * PRODUCTION EXAMPLE USAGE
 */
if (require.main === module) {
    async function main() {
        console.log('=== Q-NarwhalKnight PaaS - Production Solana Example ===\n');

        // Initialize client and wallet
        const apiKey = process.env.QNKPAAS_API_KEY || 'your_api_key';
        const secretKey = process.env.SOLANA_SECRET_KEY || 'your_base58_secret_key';
        const rpcUrl = process.env.SOLANA_RPC_URL || 'https://api.mainnet-beta.solana.com';

        const client = new QNarwhalKnightPaaSClient(apiKey);
        const wallet = new SolanaWallet(secretKey, rpcUrl);

        console.log(`Wallet Address: ${wallet.publicKey.toBase58()}`);

        const balance = await wallet.getBalanceSOL();
        console.log(`Balance: ${balance} SOL\n`);

        // Example 1: Mix SOL transfer
        const recipientAddress = 'RECIPIENT_PUBLIC_KEY_HERE';
        const amountSOL = 0.1; // 0.1 SOL

        try {
            console.log(`Mixing ${amountSOL} SOL transfer...`);
            const result = await client.mixSolanaTransaction(
                wallet,
                recipientAddress,
                amountSOL * LAMPORTS_PER_SOL,
                {
                    privacyLevel: PrivacyLevel.MAXIMUM,
                    torRelay: true,
                    timingJitterSeconds: 180
                }
            );

            console.log(`✓ Transaction mixed!`);
            console.log(`  Mixing ID: ${result.mixingId}`);
            console.log(`  TX Hash: ${result.transactionHash}`);
            console.log(`  Privacy: ε = ${result.privacyEpsilon}`);
            console.log(`  Anonymity set: ${result.anonymitySet} participants\n`);

            // Poll for completion
            console.log('Waiting for mixing to complete...');
            let status;
            do {
                await new Promise(resolve => setTimeout(resolve, 10000)); // Wait 10s
                status = await client.checkMixingStatus(result.mixingId);
                console.log(`  Status: ${status.data.status}`);
            } while (status.data.status === 'pending');

            console.log(`✓ Mixing completed!\n`);

        } catch (error) {
            console.error(`✗ Mixing failed: ${error.message}\n`);
        }

        // Example 2: Mix SPL token (e.g., USDC)
        const USDC_MINT = 'EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v';
        const tokenAmount = 100; // 100 USDC

        try {
            console.log(`Mixing ${tokenAmount} USDC...`);
            const tokenResult = await client.mixSPLTokenTransfer(
                wallet,
                USDC_MINT,
                recipientAddress,
                tokenAmount,
                {
                    privacyLevel: PrivacyLevel.MAXIMUM,
                    torRelay: true
                }
            );

            console.log(`✓ Token mixing initiated!`);
            console.log(`  Mixing ID: ${tokenResult.mixingId}`);
            console.log(`  Privacy: ε = ${tokenResult.privacyEpsilon}\n`);

        } catch (error) {
            console.error(`✗ Token mixing failed: ${error.message}\n`);
        }
    }

    main().catch(console.error);
}
