/// Dune Analytics table schema definitions.
/// Each table maps to a custom Dune table created via the API.
///
/// v8.6.5: Enhanced with derived analytics columns, tooltips for dashboards,
/// and new tables for economics/distribution analysis.

#[derive(Debug, Clone)]
pub struct ColumnDef {
    pub name: &'static str,
    pub dune_type: &'static str,
    /// University-level tooltip (precise, technical)
    pub tooltip_academic: &'static str,
    /// High-school-level tooltip (simple, intuitive)
    pub tooltip_simple: &'static str,
}

#[derive(Debug, Clone)]
pub struct TableDef {
    pub name: &'static str,
    pub description: &'static str,
    pub columns: &'static [ColumnDef],
}

// ─────────────────────────────────────────────────────────────
// TABLE 1: BLOCKS
// ─────────────────────────────────────────────────────────────
pub const TABLE_BLOCKS: TableDef = TableDef {
    name: "qnk_blocks",
    description: "Q-NarwhalKnight block headers with derived analytics (block time, throughput, cumulative emission)",
    columns: &[
        ColumnDef {
            name: "height",
            dune_type: "integer",
            tooltip_academic: "The ordinal position of this block in the canonical chain, monotonically increasing from genesis (height 0).",
            tooltip_simple: "The block number — how many blocks have been created before this one, counting from the very first block.",
        },
        ColumnDef {
            name: "timestamp",
            dune_type: "timestamp",
            tooltip_academic: "The Unix timestamp (UTC) at which this block was finalized by the DAG-Knight consensus protocol.",
            tooltip_simple: "The exact date and time this block was added to the blockchain.",
        },
        ColumnDef {
            name: "hash",
            dune_type: "varchar",
            tooltip_academic: "The BLAKE3 cryptographic digest of the block header, serving as the unique block identifier (32 bytes, hex-encoded).",
            tooltip_simple: "A unique fingerprint for this block — like a serial number that can never be duplicated.",
        },
        ColumnDef {
            name: "proposer",
            dune_type: "varchar",
            tooltip_academic: "The wallet address of the miner whose proof-of-work solution was accepted for this block's coinbase reward.",
            tooltip_simple: "The wallet address of the miner who created this block and earned the mining reward.",
        },
        ColumnDef {
            name: "tx_count",
            dune_type: "integer",
            tooltip_academic: "The number of transactions included in this block, encompassing transfers, coinbase rewards, DEX swaps, and contract interactions.",
            tooltip_simple: "How many transactions (payments, swaps, etc.) are packed inside this block.",
        },
        ColumnDef {
            name: "block_reward_qug",
            dune_type: "double",
            tooltip_academic: "The coinbase reward issued to the proposer, denominated in QUG. Subject to the 4-year halving emission schedule (Era 0: ~2,625,000 QUG/year).",
            tooltip_simple: "How many QUG coins the miner earned for creating this block. This amount halves every 4 years, like Bitcoin.",
        },
        ColumnDef {
            name: "size_bytes",
            dune_type: "integer",
            tooltip_academic: "The serialized size of the complete block (header + transactions) in bytes. Indicator of block utilization and network throughput.",
            tooltip_simple: "How much data (in bytes) this block takes up — bigger blocks mean more transactions were processed.",
        },
        ColumnDef {
            name: "dag_round",
            dune_type: "integer",
            tooltip_academic: "The DAG-Knight consensus round in which this block achieved finality. Lower rounds relative to height indicate faster consensus convergence.",
            tooltip_simple: "Which round of the consensus voting process finalized this block. Lower is faster.",
        },
        ColumnDef {
            name: "block_time_sec",
            dune_type: "double",
            tooltip_academic: "The elapsed time in seconds between this block's timestamp and its predecessor's. A measure of instantaneous throughput (target: ~1 block/sec).",
            tooltip_simple: "How many seconds passed between this block and the previous one. Ideally around 1 second.",
        },
        ColumnDef {
            name: "cumulative_emission_qug",
            dune_type: "double",
            tooltip_academic: "The running total of all QUG ever minted from genesis through this block height. Approaches the asymptotic maximum of 21,000,000 QUG over 256 years.",
            tooltip_simple: "The total amount of QUG coins that have been created from the beginning up to this block. The maximum is 21 million.",
        },
        ColumnDef {
            name: "founder_fee_qug",
            dune_type: "double",
            tooltip_academic: "The 1.9% protocol development fee extracted from this block's coinbase reward, directed to the founder wallet (efca1e8c...).",
            tooltip_simple: "1.9% of the block reward goes to the project's founder wallet to fund development. This is transparent and in every block.",
        },
        ColumnDef {
            name: "miner_net_reward_qug",
            dune_type: "double",
            tooltip_academic: "The net reward received by the miner after deducting the 1.9% founder fee and 0.1% node operator fee: block_reward * 0.98.",
            tooltip_simple: "What the miner actually keeps after the small protocol fees are subtracted — about 98% of the block reward.",
        },
    ],
};

// ─────────────────────────────────────────────────────────────
// TABLE 2: TRANSACTIONS
// ─────────────────────────────────────────────────────────────
pub const TABLE_TRANSACTIONS: TableDef = TableDef {
    name: "qnk_transactions",
    description: "All Q-NarwhalKnight transactions with type classification and value metrics",
    columns: &[
        ColumnDef {
            name: "tx_hash",
            dune_type: "varchar",
            tooltip_academic: "The cryptographic hash uniquely identifying this transaction. Computed over the signed transaction payload using BLAKE3.",
            tooltip_simple: "A unique ID for this transaction — like a receipt number that proves this specific payment happened.",
        },
        ColumnDef {
            name: "block_height",
            dune_type: "integer",
            tooltip_academic: "The block height at which this transaction was included and finalized in the canonical chain.",
            tooltip_simple: "Which block number this transaction was included in.",
        },
        ColumnDef {
            name: "block_timestamp",
            dune_type: "timestamp",
            tooltip_academic: "The timestamp of the block containing this transaction, representing the approximate time of finality.",
            tooltip_simple: "When this transaction was confirmed on the blockchain.",
        },
        ColumnDef {
            name: "tx_type",
            dune_type: "varchar",
            tooltip_academic: "Transaction classification: 'transfer' (value movement), 'coinbase' (mining reward), 'swap' (DEX trade), 'deploy' (contract deployment), 'call' (contract invocation).",
            tooltip_simple: "What kind of transaction this is: a regular payment, a mining reward, a token swap on the DEX, or a smart contract action.",
        },
        ColumnDef {
            name: "from_address",
            dune_type: "varchar",
            tooltip_academic: "The sender's wallet address (hex-encoded Ed25519 public key hash). '0x00..00' for coinbase transactions which create new coins.",
            tooltip_simple: "Who sent this transaction. For mining rewards, this is blank because new coins are created from nothing.",
        },
        ColumnDef {
            name: "to_address",
            dune_type: "varchar",
            tooltip_academic: "The recipient's wallet address. For DEX swaps, this is the AMM pool address. For contract calls, the contract address.",
            tooltip_simple: "Who received the coins or tokens in this transaction.",
        },
        ColumnDef {
            name: "amount_qug",
            dune_type: "double",
            tooltip_academic: "The transaction value denominated in QUG (converted from u128 base units with 24 decimal precision). For swaps, represents the input token amount.",
            tooltip_simple: "How many QUG coins were sent in this transaction.",
        },
        ColumnDef {
            name: "fee_qug",
            dune_type: "double",
            tooltip_academic: "The transaction fee paid to validators, denominated in QUG. Currently zero for most transactions during the early adoption phase.",
            tooltip_simple: "The fee paid for this transaction. Currently very low or zero during the early network phase.",
        },
        ColumnDef {
            name: "is_coinbase",
            dune_type: "boolean",
            tooltip_academic: "Boolean flag indicating whether this transaction is a coinbase (mining reward). Coinbase transactions have no sender and create new QUG supply.",
            tooltip_simple: "True if this transaction is a mining reward (new coins being created), false for regular transactions.",
        },
    ],
};

// ─────────────────────────────────────────────────────────────
// TABLE 3: DAILY METRICS
// ─────────────────────────────────────────────────────────────
pub const TABLE_DAILY_METRICS: TableDef = TableDef {
    name: "qnk_daily_metrics",
    description: "Daily aggregated blockchain activity — blocks, volume, fees, addresses, miners, velocity",
    columns: &[
        ColumnDef {
            name: "date",
            dune_type: "varchar",
            tooltip_academic: "The calendar date (ISO 8601: YYYY-MM-DD, UTC) for which these metrics are aggregated.",
            tooltip_simple: "The date these daily stats are for.",
        },
        ColumnDef {
            name: "block_count",
            dune_type: "integer",
            tooltip_academic: "Total number of blocks finalized during this UTC day. At target throughput (~1 bps), approximately 86,400 blocks/day.",
            tooltip_simple: "How many blocks were created this day. At full speed, about 86,400 blocks per day (one per second).",
        },
        ColumnDef {
            name: "tx_count",
            dune_type: "integer",
            tooltip_academic: "Aggregate transaction count for the day across all transaction types (transfers, coinbase, swaps, contract calls).",
            tooltip_simple: "Total number of transactions that happened this day.",
        },
        ColumnDef {
            name: "total_volume_qug",
            dune_type: "double",
            tooltip_academic: "Sum of all transaction values for the day in QUG, excluding coinbase. A proxy for economic activity and network utilization.",
            tooltip_simple: "The total value of all QUG moved around in transactions this day — shows how active the network is.",
        },
        ColumnDef {
            name: "total_fees_qug",
            dune_type: "double",
            tooltip_academic: "Aggregate transaction fees collected by validators for the day, denominated in QUG.",
            tooltip_simple: "Total fees paid by users for their transactions this day.",
        },
        ColumnDef {
            name: "active_addresses",
            dune_type: "integer",
            tooltip_academic: "Count of distinct wallet addresses that appeared as sender or recipient in at least one transaction during this day.",
            tooltip_simple: "How many different wallets were active (sent or received something) this day.",
        },
        ColumnDef {
            name: "total_emission_qug",
            dune_type: "double",
            tooltip_academic: "Total new QUG minted via coinbase rewards during this day. Should approximate daily_target = annual_emission / 365.25.",
            tooltip_simple: "How many new QUG coins were created by miners this day. This follows a predictable schedule.",
        },
        ColumnDef {
            name: "unique_miners",
            dune_type: "integer",
            tooltip_academic: "Count of distinct miner addresses that produced at least one block during this day. A measure of mining decentralization.",
            tooltip_simple: "How many different miners earned rewards this day — more miners means the network is more decentralized.",
        },
        ColumnDef {
            name: "swap_count",
            dune_type: "integer",
            tooltip_academic: "Number of DEX swap transactions executed via the on-chain AMM (constant-product formula) during this day.",
            tooltip_simple: "How many token swaps happened on the built-in decentralized exchange this day.",
        },
        ColumnDef {
            name: "avg_block_time_sec",
            dune_type: "double",
            tooltip_academic: "Mean inter-block interval for the day in seconds. Computed as 86400 / block_count. Deviations from 1.0s indicate network stress or low miner participation.",
            tooltip_simple: "The average time between blocks this day. Should be close to 1 second when the network is healthy.",
        },
        ColumnDef {
            name: "avg_block_reward_qug",
            dune_type: "double",
            tooltip_academic: "Mean coinbase reward per block for the day: total_emission / block_count. Varies with actual block production rate vs emission target.",
            tooltip_simple: "The average mining reward per block this day.",
        },
        ColumnDef {
            name: "velocity",
            dune_type: "double",
            tooltip_academic: "Token velocity: total_volume / circulating_supply. Measures how frequently each coin changes hands. Higher velocity implies greater economic activity relative to supply.",
            tooltip_simple: "How fast coins are moving around. A velocity of 1.0 means every coin changed hands once today on average.",
        },
        ColumnDef {
            name: "nvt_ratio",
            dune_type: "double",
            tooltip_academic: "Network Value to Transactions ratio (NVT): analogous to P/E ratio for blockchains. High NVT suggests overvaluation relative to on-chain utility, low NVT suggests undervaluation.",
            tooltip_simple: "A valuation metric like the price-to-earnings ratio for stocks. Lower means the network is being used more relative to its value.",
        },
    ],
};

// ─────────────────────────────────────────────────────────────
// TABLE 4: MINING REWARDS
// ─────────────────────────────────────────────────────────────
pub const TABLE_MINING_REWARDS: TableDef = TableDef {
    name: "qnk_mining_rewards",
    description: "Per-miner block rewards with era classification and fee breakdown",
    columns: &[
        ColumnDef {
            name: "block_height",
            dune_type: "integer",
            tooltip_academic: "The block height at which this mining reward was issued.",
            tooltip_simple: "Which block number generated this mining reward.",
        },
        ColumnDef {
            name: "timestamp",
            dune_type: "timestamp",
            tooltip_academic: "The timestamp when this block was finalized and the reward was issued to the miner.",
            tooltip_simple: "When this mining reward was earned.",
        },
        ColumnDef {
            name: "miner_address",
            dune_type: "varchar",
            tooltip_academic: "The wallet address that received this block's coinbase reward. Identified by the mining solution's wallet_address field.",
            tooltip_simple: "The wallet that earned this mining reward.",
        },
        ColumnDef {
            name: "reward_qug",
            dune_type: "double",
            tooltip_academic: "The gross coinbase reward before fee deductions, denominated in QUG. Determined by the emission controller's adaptive rate algorithm.",
            tooltip_simple: "How many QUG coins the miner earned for this block (before any fees).",
        },
        ColumnDef {
            name: "era",
            dune_type: "integer",
            tooltip_academic: "The emission era (0-63). Each era spans 4 years, with the annual emission halving at each transition. Era 0: 2,625,000 QUG/year.",
            tooltip_simple: "Which 4-year era this reward belongs to. Like Bitcoin halvings — the reward gets cut in half every era.",
        },
        ColumnDef {
            name: "founder_fee_qug",
            dune_type: "double",
            tooltip_academic: "The 1.9% development fund allocation from this block's reward, transparently directed to the hardcoded founder wallet.",
            tooltip_simple: "The 1.9% of this block's reward that goes to fund project development. Fully transparent and verifiable.",
        },
        ColumnDef {
            name: "miner_net_qug",
            dune_type: "double",
            tooltip_academic: "The miner's net reward after all protocol fee deductions (1.9% founder + 0.1% operator = 2.0%). Equals reward_qug * 0.98.",
            tooltip_simple: "What the miner actually keeps — 98% of the block reward after the small protocol fees.",
        },
    ],
};

// ─────────────────────────────────────────────────────────────
// TABLE 5: TOKEN SUPPLY
// ─────────────────────────────────────────────────────────────
pub const TABLE_TOKEN_SUPPLY: TableDef = TableDef {
    name: "qnk_token_supply",
    description: "Hourly token supply snapshots with inflation rate, stock-to-flow, and multi-token tracking",
    columns: &[
        ColumnDef {
            name: "timestamp",
            dune_type: "timestamp",
            tooltip_academic: "The UTC timestamp of this supply snapshot, taken at approximately 1-hour intervals.",
            tooltip_simple: "When this supply snapshot was recorded.",
        },
        ColumnDef {
            name: "total_supply_qug",
            dune_type: "double",
            tooltip_academic: "The total QUG in circulation at this moment — the sum of all coinbase rewards ever issued, representing provably minted supply.",
            tooltip_simple: "Total QUG coins that exist right now. Every coin was created through mining.",
        },
        ColumnDef {
            name: "max_supply_qug",
            dune_type: "double",
            tooltip_academic: "The asymptotic maximum supply: 21,000,000 QUG. Enforced by the emission controller with u128 arithmetic precision (10^24 base units).",
            tooltip_simple: "The maximum number of QUG that will ever exist: 21 million. This limit is hardcoded and can never be changed.",
        },
        ColumnDef {
            name: "pct_mined",
            dune_type: "double",
            tooltip_academic: "The percentage of max supply already minted: (total_supply / 21_000_000) * 100. Approaches 100% asymptotically over 256 years.",
            tooltip_simple: "What percentage of all possible QUG has been mined so far. It takes 256 years to reach 100%.",
        },
        ColumnDef {
            name: "era",
            dune_type: "integer",
            tooltip_academic: "The current emission era (0-63). Determines the annual emission rate via the halving formula: base_emission / 2^era.",
            tooltip_simple: "The current 4-year period in the coin's lifetime. Each era, mining rewards are cut in half.",
        },
        ColumnDef {
            name: "annual_target_qug",
            dune_type: "double",
            tooltip_academic: "The target annual emission for the current era. Era 0: 2,625,000 QUG/year. Halves to 1,312,500 in Era 1, etc.",
            tooltip_simple: "How many QUG are supposed to be created this year by mining. This number halves every 4 years.",
        },
        ColumnDef {
            name: "inflation_rate_pct",
            dune_type: "double",
            tooltip_academic: "Annualized monetary inflation: (annual_target / total_supply) * 100. Decreases monotonically as supply grows and emission halves.",
            tooltip_simple: "The yearly inflation rate — how fast new coins are being created compared to existing coins. Goes down over time.",
        },
        ColumnDef {
            name: "stock_to_flow",
            dune_type: "double",
            tooltip_academic: "Stock-to-Flow ratio: total_supply / annual_emission. A scarcity metric popularized in commodity analysis. Higher S2F implies greater scarcity.",
            tooltip_simple: "A measure of how scarce QUG is. Higher means more scarce. Gold has a stock-to-flow of about 62. Bitcoin's is around 50.",
        },
        ColumnDef {
            name: "qugusd_supply",
            dune_type: "double",
            tooltip_academic: "Circulating supply of the QUGUSD stablecoin, algorithmically pegged to 1 USD via the collateralized debt position (CDP) mechanism.",
            tooltip_simple: "How many QUGUSD stablecoins (pegged to the US Dollar) are currently in circulation.",
        },
        ColumnDef {
            name: "qcredit_supply",
            dune_type: "double",
            tooltip_academic: "Circulating supply of QCREDIT, the governance and credit token used for borrowing against QUG collateral in the Quillon Bank system.",
            tooltip_simple: "How many QCREDIT tokens exist — used for borrowing and governance in the Quillon Bank.",
        },
        ColumnDef {
            name: "qusd_supply",
            dune_type: "double",
            tooltip_academic: "Circulating supply of QUSD, the secondary stablecoin used in the Quillon Bank lending markets.",
            tooltip_simple: "How many QUSD stablecoins are currently in circulation.",
        },
    ],
};

// ─────────────────────────────────────────────────────────────
// TABLE 6: DEX SWAPS
// ─────────────────────────────────────────────────────────────
pub const TABLE_DEX_SWAPS: TableDef = TableDef {
    name: "qnk_dex_swaps",
    description: "DEX swap activity with price impact and volume metrics",
    columns: &[
        ColumnDef {
            name: "tx_hash",
            dune_type: "varchar",
            tooltip_academic: "The transaction hash of the swap execution. Links to qnk_transactions for full transaction metadata.",
            tooltip_simple: "The unique ID of this swap transaction.",
        },
        ColumnDef {
            name: "block_height",
            dune_type: "integer",
            tooltip_academic: "The block in which this swap was finalized.",
            tooltip_simple: "Which block this swap was included in.",
        },
        ColumnDef {
            name: "timestamp",
            dune_type: "timestamp",
            tooltip_academic: "The block timestamp when this swap achieved finality.",
            tooltip_simple: "When this swap happened.",
        },
        ColumnDef {
            name: "wallet",
            dune_type: "varchar",
            tooltip_academic: "The wallet address of the trader who initiated this swap against the AMM liquidity pool.",
            tooltip_simple: "The wallet that made this swap.",
        },
        ColumnDef {
            name: "token_in",
            dune_type: "varchar",
            tooltip_academic: "The input token symbol or contract address. The token the trader sold into the AMM pool.",
            tooltip_simple: "Which token the trader gave up (sold).",
        },
        ColumnDef {
            name: "token_out",
            dune_type: "varchar",
            tooltip_academic: "The output token symbol or contract address. The token the trader received from the AMM pool.",
            tooltip_simple: "Which token the trader received (bought).",
        },
        ColumnDef {
            name: "amount_in_display",
            dune_type: "double",
            tooltip_academic: "The input token amount in human-readable display units (adjusted for token decimals).",
            tooltip_simple: "How much of the input token was traded.",
        },
        ColumnDef {
            name: "amount_out_display",
            dune_type: "double",
            tooltip_academic: "The output token amount in human-readable display units. Determined by the constant-product AMM formula: x*y=k.",
            tooltip_simple: "How much of the output token was received. The DEX uses a math formula (x*y=k) to calculate fair prices automatically.",
        },
        ColumnDef {
            name: "effective_price",
            dune_type: "double",
            tooltip_academic: "The realized exchange rate: amount_out / amount_in. Includes price impact (slippage) from the constant-product invariant.",
            tooltip_simple: "The actual price the trader got. May differ slightly from the displayed price due to slippage.",
        },
    ],
};

// ─────────────────────────────────────────────────────────────
// TABLE 7: TOP HOLDERS
// ─────────────────────────────────────────────────────────────
pub const TABLE_TOP_HOLDERS: TableDef = TableDef {
    name: "qnk_top_holders",
    description: "Daily top 100 token holders with wealth distribution metrics",
    columns: &[
        ColumnDef {
            name: "snapshot_date",
            dune_type: "varchar",
            tooltip_academic: "The UTC date (YYYY-MM-DD) of this wealth distribution snapshot.",
            tooltip_simple: "The date this wealth snapshot was taken.",
        },
        ColumnDef {
            name: "rank",
            dune_type: "integer",
            tooltip_academic: "The ordinal rank of this address by QUG balance, where 1 = largest holder.",
            tooltip_simple: "This wallet's rank among all holders. #1 has the most coins.",
        },
        ColumnDef {
            name: "address",
            dune_type: "varchar",
            tooltip_academic: "The wallet address (hex-encoded). Known addresses include: founder wallet (efca1e8c...), bootstrap nodes, early miners.",
            tooltip_simple: "The wallet address of this holder.",
        },
        ColumnDef {
            name: "balance_qug",
            dune_type: "double",
            tooltip_academic: "The QUG balance of this address at the time of the snapshot. Derived from the UTXO-like balance consensus engine.",
            tooltip_simple: "How many QUG coins this wallet holds.",
        },
        ColumnDef {
            name: "pct_of_supply",
            dune_type: "double",
            tooltip_academic: "This address's balance as a percentage of total circulating supply: (balance / total_supply) * 100. Key metric for Gini coefficient estimation.",
            tooltip_simple: "What percentage of all existing QUG this wallet owns.",
        },
        ColumnDef {
            name: "is_founder",
            dune_type: "boolean",
            tooltip_academic: "Whether this address is the hardcoded protocol founder wallet (efca1e8c...) receiving the 1.9% coinbase fee.",
            tooltip_simple: "Whether this is the project's founder wallet (receives 1.9% of all mining rewards for development funding).",
        },
        ColumnDef {
            name: "is_contract",
            dune_type: "boolean",
            tooltip_academic: "Whether this address is a smart contract (DEX pool, vault, etc.) rather than a user-controlled wallet.",
            tooltip_simple: "Whether this is a smart contract (like a DEX pool) rather than a person's wallet.",
        },
    ],
};

// ─────────────────────────────────────────────────────────────
// TABLE 8: NETWORK STATS
// ─────────────────────────────────────────────────────────────
pub const TABLE_NETWORK_STATS: TableDef = TableDef {
    name: "qnk_network_stats",
    description: "5-minute network health snapshots — peers, hashrate, difficulty, security metrics",
    columns: &[
        ColumnDef {
            name: "timestamp",
            dune_type: "timestamp",
            tooltip_academic: "The UTC timestamp of this network health snapshot, captured every 300 seconds.",
            tooltip_simple: "When this network snapshot was taken (every 5 minutes).",
        },
        ColumnDef {
            name: "block_height",
            dune_type: "integer",
            tooltip_academic: "The canonical chain height at the time of this snapshot.",
            tooltip_simple: "The current block number when this snapshot was taken.",
        },
        ColumnDef {
            name: "peer_count",
            dune_type: "integer",
            tooltip_academic: "Number of libp2p gossipsub peers connected to the bootstrap node. Reflects P2P network topology density.",
            tooltip_simple: "How many computers (nodes) are connected to the network right now.",
        },
        ColumnDef {
            name: "active_miners",
            dune_type: "integer",
            tooltip_academic: "Number of distinct mining wallets that have submitted valid proof-of-work solutions within the last 5-minute window.",
            tooltip_simple: "How many miners are actively working to create new blocks right now.",
        },
        ColumnDef {
            name: "total_hashrate_khs",
            dune_type: "double",
            tooltip_academic: "Aggregate network hash rate in kilohashes per second (kH/s). Estimated from mining solution submission rates and difficulty.",
            tooltip_simple: "Total computing power of all miners combined, measured in kilohashes per second. More hashrate = more secure network.",
        },
        ColumnDef {
            name: "difficulty",
            dune_type: "double",
            tooltip_academic: "The current mining difficulty target. Adjusts dynamically to maintain ~1 block/second throughput regardless of total hashrate.",
            tooltip_simple: "How hard it is to mine a block. Automatically adjusts so blocks come about once per second.",
        },
        ColumnDef {
            name: "blocks_per_minute",
            dune_type: "double",
            tooltip_academic: "Rolling 5-minute average of block production rate, normalized to blocks/minute. Target: ~60 blocks/min at 1 bps.",
            tooltip_simple: "How many blocks are being created per minute. Should be around 60 when the network is running at full speed.",
        },
        ColumnDef {
            name: "nakamoto_coefficient",
            dune_type: "integer",
            tooltip_academic: "The minimum number of miners needed to control >50% of recent block production. Higher = more decentralized. Named after Satoshi Nakamoto.",
            tooltip_simple: "How many miners you'd need to team up to control the network. Higher is better — means the network is more decentralized and secure.",
        },
    ],
};

// ─────────────────────────────────────────────────────────────
// TABLE 9: EMISSION SCHEDULE
// ─────────────────────────────────────────────────────────────
pub const TABLE_EMISSION_SCHEDULE: TableDef = TableDef {
    name: "qnk_emission_schedule",
    description: "64-era halving emission schedule — the complete 256-year monetary policy of QUG",
    columns: &[
        ColumnDef {
            name: "era",
            dune_type: "integer",
            tooltip_academic: "The emission era index (0 through 63). Each era spans exactly 4 calendar years, yielding a 256-year total emission timeline.",
            tooltip_simple: "The era number (0-63). Each era is 4 years long. We're currently in Era 0 (the first era).",
        },
        ColumnDef {
            name: "start_year",
            dune_type: "integer",
            tooltip_academic: "The calendar year in which this era begins. Era 0 starts at genesis (2026), Era 1 at 2030, etc.",
            tooltip_simple: "What year this era starts. Era 0 starts in 2026, Era 1 in 2030, and so on.",
        },
        ColumnDef {
            name: "annual_emission_qug",
            dune_type: "double",
            tooltip_academic: "The target annual emission for this era: base_emission / 2^era. Era 0 = 2,625,000 QUG/year, Era 1 = 1,312,500, Era 2 = 656,250, ...",
            tooltip_simple: "How many QUG will be created per year during this era. It halves each era: 2.6M, then 1.3M, then 656K, and so on.",
        },
        ColumnDef {
            name: "cumulative_supply_qug",
            dune_type: "double",
            tooltip_academic: "The total supply at the end of this era: sum of (emission_i * 4) for i=0..era. Converges to 21,000,000 QUG.",
            tooltip_simple: "Total QUG that will exist by the end of this era. Gets closer and closer to 21 million but never quite reaches it.",
        },
        ColumnDef {
            name: "halving_factor",
            dune_type: "double",
            tooltip_academic: "The cumulative halving multiplier: 1 / 2^era. Represents the fraction of Era 0's emission rate still active in this era.",
            tooltip_simple: "How much the reward has shrunk compared to the start. After 1 halving it's 0.5 (half), after 2 it's 0.25 (quarter), etc.",
        },
        ColumnDef {
            name: "is_current",
            dune_type: "boolean",
            tooltip_academic: "Boolean indicating whether this is the currently active emission era, determined by comparing chain age to era boundaries.",
            tooltip_simple: "Whether the blockchain is currently in this era right now.",
        },
        ColumnDef {
            name: "pct_of_max_supply",
            dune_type: "double",
            tooltip_academic: "The percentage of max supply (21M) emitted during this era alone: (era_emission * 4 / 21_000_000) * 100.",
            tooltip_simple: "What fraction of the total 21 million QUG will be created during this specific 4-year era.",
        },
        ColumnDef {
            name: "btc_halving_equivalent",
            dune_type: "integer",
            tooltip_academic: "The approximate Bitcoin halving number that corresponds to this era's emission rate. Era 0 ≈ BTC Halving 0 (50 BTC/block era).",
            tooltip_simple: "Which Bitcoin halving this era is comparable to. Helps compare QUG's emission schedule to Bitcoin's.",
        },
    ],
};

// ─────────────────────────────────────────────────────────────
// TABLE 10: MINER ECONOMICS (NEW)
// ─────────────────────────────────────────────────────────────
pub const TABLE_MINER_ECONOMICS: TableDef = TableDef {
    name: "qnk_miner_economics",
    description: "Daily per-miner profitability and dominance metrics",
    columns: &[
        ColumnDef {
            name: "date",
            dune_type: "varchar",
            tooltip_academic: "The UTC date for this miner economics snapshot.",
            tooltip_simple: "The date these mining stats are for.",
        },
        ColumnDef {
            name: "miner_address",
            dune_type: "varchar",
            tooltip_academic: "The wallet address of this miner.",
            tooltip_simple: "The miner's wallet address.",
        },
        ColumnDef {
            name: "blocks_mined",
            dune_type: "integer",
            tooltip_academic: "Number of blocks produced by this miner during the day. A proxy for the miner's share of total network hashrate.",
            tooltip_simple: "How many blocks this miner created today.",
        },
        ColumnDef {
            name: "total_reward_qug",
            dune_type: "double",
            tooltip_academic: "Sum of all coinbase rewards earned by this miner during the day (before fee deductions).",
            tooltip_simple: "Total QUG earned by this miner today.",
        },
        ColumnDef {
            name: "pct_of_blocks",
            dune_type: "double",
            tooltip_academic: "This miner's share of total block production for the day: (miner_blocks / total_blocks) * 100. Measures hashrate dominance.",
            tooltip_simple: "What percentage of today's blocks were mined by this miner. Higher = more powerful miner.",
        },
        ColumnDef {
            name: "cumulative_reward_qug",
            dune_type: "double",
            tooltip_academic: "Running lifetime total of all QUG earned by this miner address since genesis.",
            tooltip_simple: "Total QUG this miner has earned since the network started.",
        },
        ColumnDef {
            name: "avg_block_interval_sec",
            dune_type: "double",
            tooltip_academic: "Mean time between consecutive blocks mined by this address. Shorter intervals indicate higher hashrate or more consistent mining.",
            tooltip_simple: "Average time between blocks for this miner. Shorter = faster/more powerful miner.",
        },
    ],
};

// ─────────────────────────────────────────────────────────────
// TABLE 11: WEALTH DISTRIBUTION (NEW)
// ─────────────────────────────────────────────────────────────
pub const TABLE_WEALTH_DISTRIBUTION: TableDef = TableDef {
    name: "qnk_wealth_distribution",
    description: "Daily wealth concentration metrics — Gini coefficient, Herfindahl index, supply distribution by tier",
    columns: &[
        ColumnDef {
            name: "date",
            dune_type: "varchar",
            tooltip_academic: "The UTC date for this wealth distribution analysis.",
            tooltip_simple: "The date of this wealth distribution snapshot.",
        },
        ColumnDef {
            name: "total_holders",
            dune_type: "integer",
            tooltip_academic: "Count of unique addresses with non-zero QUG balance.",
            tooltip_simple: "How many wallets currently hold any QUG.",
        },
        ColumnDef {
            name: "gini_coefficient",
            dune_type: "double",
            tooltip_academic: "The Gini coefficient (0-1) measuring wealth inequality across all holders. 0 = perfect equality, 1 = one address holds everything. Computed via trapezoidal Lorenz curve approximation.",
            tooltip_simple: "A number from 0 to 1 showing how evenly coins are distributed. 0 means everyone has the same amount, 1 means one person has everything.",
        },
        ColumnDef {
            name: "top10_pct",
            dune_type: "double",
            tooltip_academic: "Percentage of total circulating supply held by the top 10 addresses. A key centralization risk indicator.",
            tooltip_simple: "What percentage of all QUG is owned by the top 10 richest wallets.",
        },
        ColumnDef {
            name: "top50_pct",
            dune_type: "double",
            tooltip_academic: "Percentage of circulating supply held by the top 50 addresses.",
            tooltip_simple: "What percentage of all QUG is owned by the top 50 richest wallets.",
        },
        ColumnDef {
            name: "whales",
            dune_type: "integer",
            tooltip_academic: "Number of addresses holding > 1% of circulating supply. Whale movements can cause significant price volatility.",
            tooltip_simple: "How many wallets own more than 1% of all QUG. These are the 'whales' — big players.",
        },
        ColumnDef {
            name: "dolphins",
            dune_type: "integer",
            tooltip_academic: "Number of addresses holding 0.1-1% of circulating supply.",
            tooltip_simple: "How many wallets own between 0.1% and 1% of all QUG. Medium-sized holders.",
        },
        ColumnDef {
            name: "fish",
            dune_type: "integer",
            tooltip_academic: "Number of addresses holding 0.01-0.1% of circulating supply.",
            tooltip_simple: "How many wallets own between 0.01% and 0.1% of all QUG. Small but meaningful holders.",
        },
        ColumnDef {
            name: "shrimp",
            dune_type: "integer",
            tooltip_academic: "Number of addresses holding < 0.01% of circulating supply. The long tail of retail participants.",
            tooltip_simple: "How many wallets own less than 0.01% of all QUG. Regular everyday users.",
        },
        ColumnDef {
            name: "herfindahl_index",
            dune_type: "double",
            tooltip_academic: "The Herfindahl-Hirschman Index (HHI) for supply concentration: sum of squared market shares. Range 0-10,000. <1,500 = competitive, >2,500 = concentrated.",
            tooltip_simple: "A standard measure of concentration used in economics. Below 1,500 means well-distributed, above 2,500 means too concentrated.",
        },
    ],
};

// ─────────────────────────────────────────────────────────────
// TABLE 12: BLOCK TIME ANALYSIS (NEW)
// ─────────────────────────────────────────────────────────────
pub const TABLE_BLOCK_TIME_ANALYSIS: TableDef = TableDef {
    name: "qnk_block_time_analysis",
    description: "Hourly block production statistics — throughput, orphans, consensus health",
    columns: &[
        ColumnDef {
            name: "hour",
            dune_type: "timestamp",
            tooltip_academic: "The start of the 1-hour window for this analysis (floored to hour boundary, UTC).",
            tooltip_simple: "The hour this analysis covers.",
        },
        ColumnDef {
            name: "blocks_produced",
            dune_type: "integer",
            tooltip_academic: "Number of blocks finalized during this hour. Target: ~3,600 at 1 block/second.",
            tooltip_simple: "How many blocks were created this hour. Should be about 3,600 (one per second).",
        },
        ColumnDef {
            name: "avg_block_time_sec",
            dune_type: "double",
            tooltip_academic: "Mean inter-block interval during this hour. Computed as 3600 / blocks_produced.",
            tooltip_simple: "Average seconds between blocks this hour. Should be close to 1.0.",
        },
        ColumnDef {
            name: "min_block_time_sec",
            dune_type: "double",
            tooltip_academic: "The shortest inter-block interval observed during this hour. Very short intervals may indicate network bursts.",
            tooltip_simple: "The fastest block time this hour — the shortest gap between two consecutive blocks.",
        },
        ColumnDef {
            name: "max_block_time_sec",
            dune_type: "double",
            tooltip_academic: "The longest inter-block interval during this hour. Extended intervals may indicate miner dropout or network partitions.",
            tooltip_simple: "The slowest block time this hour. Long gaps might mean miners went offline temporarily.",
        },
        ColumnDef {
            name: "stddev_block_time",
            dune_type: "double",
            tooltip_academic: "Standard deviation of inter-block intervals. Lower values indicate more consistent block production (healthy network). Target: <0.5s.",
            tooltip_simple: "How consistent block times were. Lower is better — means blocks are coming at a steady, predictable pace.",
        },
        ColumnDef {
            name: "total_emission_qug",
            dune_type: "double",
            tooltip_academic: "Total QUG minted via coinbase during this hour.",
            tooltip_simple: "How many new QUG were mined this hour.",
        },
        ColumnDef {
            name: "unique_miners",
            dune_type: "integer",
            tooltip_academic: "Distinct miner addresses that produced blocks during this hour.",
            tooltip_simple: "How many different miners earned rewards this hour.",
        },
        ColumnDef {
            name: "total_tx_count",
            dune_type: "integer",
            tooltip_academic: "Aggregate transaction count across all blocks in this hour.",
            tooltip_simple: "Total transactions processed this hour.",
        },
        ColumnDef {
            name: "avg_txs_per_block",
            dune_type: "double",
            tooltip_academic: "Mean transactions per block during this hour: total_tx_count / blocks_produced.",
            tooltip_simple: "Average number of transactions in each block this hour.",
        },
    ],
};

// ─────────────────────────────────────────────────────────────
// MASTER TABLE LIST
// ─────────────────────────────────────────────────────────────
pub const ALL_TABLES: &[&TableDef] = &[
    &TABLE_BLOCKS,
    &TABLE_TRANSACTIONS,
    &TABLE_DAILY_METRICS,
    &TABLE_MINING_REWARDS,
    &TABLE_TOKEN_SUPPLY,
    &TABLE_DEX_SWAPS,
    &TABLE_TOP_HOLDERS,
    &TABLE_NETWORK_STATS,
    &TABLE_EMISSION_SCHEDULE,
    &TABLE_MINER_ECONOMICS,
    &TABLE_WEALTH_DISTRIBUTION,
    &TABLE_BLOCK_TIME_ANALYSIS,
];
