/// Type-safe CSV builders for each Dune table.
/// All monetary amounts are converted from u128 base units (24 decimals) to f64 QUG.
///
/// v8.6.5: Enhanced with derived analytics columns (block time, fees, economics).

const QUG_DIVISOR: f64 = 1_000_000_000_000_000_000_000_000.0; // 10^24

/// Convert u128 base units to QUG display value.
#[inline]
pub fn to_qug(base_units: u128) -> f64 {
    base_units as f64 / QUG_DIVISOR
}

/// Format a Unix timestamp as ISO 8601 for Dune.
#[inline]
pub fn unix_to_iso(ts: u64) -> String {
    chrono::DateTime::from_timestamp(ts as i64, 0)
        .map(|dt| dt.format("%Y-%m-%d %H:%M:%S").to_string())
        .unwrap_or_else(|| "1970-01-01 00:00:00".to_string())
}

/// Escape a CSV field (double-quote if it contains comma, quote, or newline).
#[inline]
fn csv_escape(s: &str) -> String {
    if s.contains(',') || s.contains('"') || s.contains('\n') {
        format!("\"{}\"", s.replace('"', "\"\""))
    } else {
        s.to_string()
    }
}

// ─────────────────────────────────────────────────────────
// TABLE 1: BLOCKS (enhanced with block_time, cumulative emission, fee breakdown)
// ─────────────────────────────────────────────────────────
pub struct BlocksCsv {
    buf: String,
    prev_timestamp: Option<u64>,
    cumulative_emission: f64,
}

impl BlocksCsv {
    pub fn new() -> Self {
        Self {
            buf: "height,timestamp,hash,proposer,tx_count,block_reward_qug,size_bytes,dag_round,block_time_sec,cumulative_emission_qug,founder_fee_qug,miner_net_reward_qug\n".to_string(),
            prev_timestamp: None,
            cumulative_emission: 0.0,
        }
    }

    /// Set the cumulative emission up to this point (for resume after backfill).
    pub fn set_cumulative_emission(&mut self, val: f64) {
        self.cumulative_emission = val;
    }

    /// Set the timestamp of the block before the first block in this batch.
    pub fn set_prev_timestamp(&mut self, ts: u64) {
        self.prev_timestamp = Some(ts);
    }

    pub fn add_row(
        &mut self,
        height: u64,
        timestamp: u64,
        hash: &str,
        proposer: &str,
        tx_count: usize,
        block_reward: u128,
        size_bytes: usize,
        dag_round: u64,
    ) {
        use std::fmt::Write;

        let reward_qug = to_qug(block_reward);
        let block_time = self.prev_timestamp
            .map(|prev| if timestamp > prev { (timestamp - prev) as f64 } else { 0.0 })
            .unwrap_or(0.0);
        self.prev_timestamp = Some(timestamp);
        self.cumulative_emission += reward_qug;

        let founder_fee = reward_qug * 0.019;
        let miner_net = reward_qug * 0.98;

        let _ = writeln!(
            self.buf,
            "{},{},{},{},{},{:.12},{},{},{:.3},{:.12},{:.12},{:.12}",
            height,
            unix_to_iso(timestamp),
            csv_escape(hash),
            csv_escape(proposer),
            tx_count,
            reward_qug,
            size_bytes,
            dag_round,
            block_time,
            self.cumulative_emission,
            founder_fee,
            miner_net,
        );
    }

    pub fn finish(self) -> String {
        self.buf
    }

    pub fn row_count(&self) -> usize {
        self.buf.lines().count().saturating_sub(1)
    }

    pub fn cumulative_emission(&self) -> f64 {
        self.cumulative_emission
    }
}

// ─────────────────────────────────────────────────────────
// TABLE 2: TRANSACTIONS
// ─────────────────────────────────────────────────────────
pub struct TransactionsCsv {
    buf: String,
}

impl TransactionsCsv {
    pub fn new() -> Self {
        Self {
            buf: "tx_hash,block_height,block_timestamp,tx_type,from_address,to_address,amount_qug,fee_qug,is_coinbase\n".to_string(),
        }
    }

    pub fn add_row(
        &mut self,
        tx_hash: &str,
        block_height: u64,
        block_timestamp: u64,
        tx_type: &str,
        from_address: &str,
        to_address: &str,
        amount: u128,
        fee: u128,
        is_coinbase: bool,
    ) {
        use std::fmt::Write;
        let _ = writeln!(
            self.buf,
            "{},{},{},{},{},{},{:.12},{:.12},{}",
            csv_escape(tx_hash),
            block_height,
            unix_to_iso(block_timestamp),
            csv_escape(tx_type),
            csv_escape(from_address),
            csv_escape(to_address),
            to_qug(amount),
            to_qug(fee),
            is_coinbase,
        );
    }

    pub fn finish(self) -> String {
        self.buf
    }

    pub fn row_count(&self) -> usize {
        self.buf.lines().count().saturating_sub(1)
    }
}

// ─────────────────────────────────────────────────────────
// TABLE 3: MINING REWARDS (enhanced with fee breakdown)
// ─────────────────────────────────────────────────────────
pub struct MiningRewardsCsv {
    buf: String,
}

impl MiningRewardsCsv {
    pub fn new() -> Self {
        Self {
            buf: "block_height,timestamp,miner_address,reward_qug,era,founder_fee_qug,miner_net_qug\n".to_string(),
        }
    }

    pub fn add_row(
        &mut self,
        block_height: u64,
        timestamp: u64,
        miner_address: &str,
        reward: u128,
        era: u64,
    ) {
        use std::fmt::Write;
        let reward_qug = to_qug(reward);
        let founder_fee = reward_qug * 0.019;
        let miner_net = reward_qug * 0.98;
        let _ = writeln!(
            self.buf,
            "{},{},{},{:.12},{},{:.12},{:.12}",
            block_height,
            unix_to_iso(timestamp),
            csv_escape(miner_address),
            reward_qug,
            era,
            founder_fee,
            miner_net,
        );
    }

    pub fn finish(self) -> String {
        self.buf
    }

    pub fn row_count(&self) -> usize {
        self.buf.lines().count().saturating_sub(1)
    }
}

// ─────────────────────────────────────────────────────────
// TABLE 4: DAILY METRICS (enhanced with velocity, NVT, avg block time)
// ─────────────────────────────────────────────────────────
pub struct DailyMetricsCsv {
    buf: String,
}

impl DailyMetricsCsv {
    pub fn new() -> Self {
        Self {
            buf: "date,block_count,tx_count,total_volume_qug,total_fees_qug,active_addresses,total_emission_qug,unique_miners,swap_count,avg_block_time_sec,avg_block_reward_qug,velocity,nvt_ratio\n".to_string(),
        }
    }

    pub fn add_row(
        &mut self,
        date: &str,
        block_count: u64,
        tx_count: u64,
        total_volume: u128,
        total_fees: u128,
        active_addresses: u64,
        total_emission: u128,
        unique_miners: u64,
        swap_count: u64,
        circulating_supply: u128,
    ) {
        use std::fmt::Write;
        let vol = to_qug(total_volume);
        let emission = to_qug(total_emission);
        let supply = to_qug(circulating_supply);

        let avg_block_time = if block_count > 0 { 86400.0 / block_count as f64 } else { 0.0 };
        let avg_block_reward = if block_count > 0 { emission / block_count as f64 } else { 0.0 };
        let velocity = if supply > 0.0 { vol / supply } else { 0.0 };
        // NVT: network_value / daily_tx_volume (use supply as proxy for network value)
        let nvt = if vol > 0.0 { supply / vol } else { 0.0 };

        let _ = writeln!(
            self.buf,
            "{},{},{},{:.12},{:.12},{},{:.12},{},{},{:.4},{:.12},{:.6},{:.4}",
            csv_escape(date),
            block_count,
            tx_count,
            vol,
            to_qug(total_fees),
            active_addresses,
            emission,
            unique_miners,
            swap_count,
            avg_block_time,
            avg_block_reward,
            velocity,
            nvt,
        );
    }

    pub fn finish(self) -> String {
        self.buf
    }
}

// ─────────────────────────────────────────────────────────
// TABLE 5: TOKEN SUPPLY (enhanced with inflation rate, stock-to-flow)
// ─────────────────────────────────────────────────────────
pub struct TokenSupplyCsv {
    buf: String,
}

impl TokenSupplyCsv {
    pub fn new() -> Self {
        Self {
            buf: "timestamp,total_supply_qug,max_supply_qug,pct_mined,era,annual_target_qug,inflation_rate_pct,stock_to_flow,qugusd_supply,qcredit_supply,qusd_supply\n".to_string(),
        }
    }

    pub fn add_row(
        &mut self,
        timestamp: u64,
        total_supply: u128,
        max_supply: u128,
        pct_mined: f64,
        era: u64,
        annual_target: u128,
        qugusd_supply: f64,
        qcredit_supply: f64,
        qusd_supply: f64,
    ) {
        use std::fmt::Write;
        let supply = to_qug(total_supply);
        let target = to_qug(annual_target);

        let inflation_rate = if supply > 0.0 { (target / supply) * 100.0 } else { 0.0 };
        let stock_to_flow = if target > 0.0 { supply / target } else { 0.0 };

        let _ = writeln!(
            self.buf,
            "{},{:.12},{:.12},{:.6},{},{:.12},{:.4},{:.4},{:.6},{:.6},{:.6}",
            unix_to_iso(timestamp),
            supply,
            to_qug(max_supply),
            pct_mined,
            era,
            target,
            inflation_rate,
            stock_to_flow,
            qugusd_supply,
            qcredit_supply,
            qusd_supply,
        );
    }

    pub fn finish(self) -> String {
        self.buf
    }
}

// ─────────────────────────────────────────────────────────
// TABLE 6: DEX SWAPS (enhanced with effective price)
// ─────────────────────────────────────────────────────────
pub struct DexSwapsCsv {
    buf: String,
}

impl DexSwapsCsv {
    pub fn new() -> Self {
        Self {
            buf: "tx_hash,block_height,timestamp,wallet,token_in,token_out,amount_in_display,amount_out_display,effective_price\n".to_string(),
        }
    }

    pub fn add_row(
        &mut self,
        tx_hash: &str,
        block_height: u64,
        timestamp: u64,
        wallet: &str,
        token_in: &str,
        token_out: &str,
        amount_in: u128,
        amount_out: u128,
    ) {
        use std::fmt::Write;
        let in_display = to_qug(amount_in);
        let out_display = to_qug(amount_out);
        let price = if in_display > 0.0 { out_display / in_display } else { 0.0 };

        let _ = writeln!(
            self.buf,
            "{},{},{},{},{},{},{:.12},{:.12},{:.8}",
            csv_escape(tx_hash),
            block_height,
            unix_to_iso(timestamp),
            csv_escape(wallet),
            csv_escape(token_in),
            csv_escape(token_out),
            in_display,
            out_display,
            price,
        );
    }

    pub fn finish(self) -> String {
        self.buf
    }

    pub fn row_count(&self) -> usize {
        self.buf.lines().count().saturating_sub(1)
    }
}

// ─────────────────────────────────────────────────────────
// TABLE 7: TOP HOLDERS (enhanced with is_founder, is_contract)
// ─────────────────────────────────────────────────────────
pub struct TopHoldersCsv {
    buf: String,
}

impl TopHoldersCsv {
    pub fn new() -> Self {
        Self {
            buf: "snapshot_date,rank,address,balance_qug,pct_of_supply,is_founder,is_contract\n".to_string(),
        }
    }

    pub fn add_row(
        &mut self,
        snapshot_date: &str,
        rank: u32,
        address: &str,
        balance: u128,
        pct_of_supply: f64,
    ) {
        use std::fmt::Write;
        // Founder wallet starts with "efca1e8c"
        let is_founder = address.starts_with("efca1e8c");
        // Contract addresses start with "c0" prefix (convention)
        let is_contract = address.starts_with("c0") || address.starts_with("C0");

        let _ = writeln!(
            self.buf,
            "{},{},{},{:.12},{:.6},{},{}",
            csv_escape(snapshot_date),
            rank,
            csv_escape(address),
            to_qug(balance),
            pct_of_supply,
            is_founder,
            is_contract,
        );
    }

    pub fn finish(self) -> String {
        self.buf
    }
}

// ─────────────────────────────────────────────────────────
// TABLE 8: NETWORK STATS (enhanced with blocks_per_minute, nakamoto)
// ─────────────────────────────────────────────────────────
pub struct NetworkStatsCsv {
    buf: String,
}

impl NetworkStatsCsv {
    pub fn new() -> Self {
        Self {
            buf: "timestamp,block_height,peer_count,active_miners,total_hashrate_khs,difficulty,blocks_per_minute,nakamoto_coefficient\n".to_string(),
        }
    }

    pub fn add_row(
        &mut self,
        timestamp: u64,
        block_height: u64,
        peer_count: u32,
        active_miners: u32,
        total_hashrate_khs: f64,
        difficulty: f64,
        blocks_per_minute: f64,
        nakamoto_coefficient: u32,
    ) {
        use std::fmt::Write;
        let _ = writeln!(
            self.buf,
            "{},{},{},{},{:.4},{:.4},{:.2},{}",
            unix_to_iso(timestamp),
            block_height,
            peer_count,
            active_miners,
            total_hashrate_khs,
            difficulty,
            blocks_per_minute,
            nakamoto_coefficient,
        );
    }

    pub fn finish(self) -> String {
        self.buf
    }
}

// ─────────────────────────────────────────────────────────
// TABLE 9: EMISSION SCHEDULE (enhanced with pct_of_max, btc equivalent)
// ─────────────────────────────────────────────────────────
pub struct EmissionScheduleCsv {
    buf: String,
}

impl EmissionScheduleCsv {
    pub fn new() -> Self {
        Self {
            buf: "era,start_year,annual_emission_qug,cumulative_supply_qug,halving_factor,is_current,pct_of_max_supply,btc_halving_equivalent\n".to_string(),
        }
    }

    pub fn add_row(
        &mut self,
        era: u64,
        start_year: u64,
        annual_emission: u128,
        cumulative_supply: u128,
        halving_factor: f64,
        is_current: bool,
    ) {
        use std::fmt::Write;
        let emission_qug = to_qug(annual_emission);
        let era_total = emission_qug * 4.0; // 4 years per era
        let pct_of_max = (era_total / 21_000_000.0) * 100.0;

        let _ = writeln!(
            self.buf,
            "{},{},{:.12},{:.12},{:.18},{},{:.4},{}",
            era,
            start_year,
            emission_qug,
            to_qug(cumulative_supply),
            halving_factor,
            is_current,
            pct_of_max,
            era, // QUG era N ≈ BTC halving N
        );
    }

    pub fn finish(self) -> String {
        self.buf
    }
}

// ─────────────────────────────────────────────────────────
// TABLE 10: MINER ECONOMICS (NEW)
// ─────────────────────────────────────────────────────────
pub struct MinerEconomicsCsv {
    buf: String,
}

impl MinerEconomicsCsv {
    pub fn new() -> Self {
        Self {
            buf: "date,miner_address,blocks_mined,total_reward_qug,pct_of_blocks,cumulative_reward_qug,avg_block_interval_sec\n".to_string(),
        }
    }

    pub fn add_row(
        &mut self,
        date: &str,
        miner_address: &str,
        blocks_mined: u64,
        total_reward: u128,
        pct_of_blocks: f64,
        cumulative_reward: u128,
        avg_block_interval: f64,
    ) {
        use std::fmt::Write;
        let _ = writeln!(
            self.buf,
            "{},{},{},{:.12},{:.4},{:.12},{:.2}",
            csv_escape(date),
            csv_escape(miner_address),
            blocks_mined,
            to_qug(total_reward),
            pct_of_blocks,
            to_qug(cumulative_reward),
            avg_block_interval,
        );
    }

    pub fn finish(self) -> String {
        self.buf
    }
}

// ─────────────────────────────────────────────────────────
// TABLE 11: WEALTH DISTRIBUTION (NEW)
// ─────────────────────────────────────────────────────────
pub struct WealthDistributionCsv {
    buf: String,
}

impl WealthDistributionCsv {
    pub fn new() -> Self {
        Self {
            buf: "date,total_holders,gini_coefficient,top10_pct,top50_pct,whales,dolphins,fish,shrimp,herfindahl_index\n".to_string(),
        }
    }

    pub fn add_row(
        &mut self,
        date: &str,
        total_holders: u64,
        gini: f64,
        top10_pct: f64,
        top50_pct: f64,
        whales: u64,
        dolphins: u64,
        fish: u64,
        shrimp: u64,
        hhi: f64,
    ) {
        use std::fmt::Write;
        let _ = writeln!(
            self.buf,
            "{},{},{:.6},{:.4},{:.4},{},{},{},{},{:.4}",
            csv_escape(date),
            total_holders,
            gini,
            top10_pct,
            top50_pct,
            whales,
            dolphins,
            fish,
            shrimp,
            hhi,
        );
    }

    pub fn finish(self) -> String {
        self.buf
    }
}

// ─────────────────────────────────────────────────────────
// TABLE 12: BLOCK TIME ANALYSIS (NEW)
// ─────────────────────────────────────────────────────────
pub struct BlockTimeAnalysisCsv {
    buf: String,
}

impl BlockTimeAnalysisCsv {
    pub fn new() -> Self {
        Self {
            buf: "hour,blocks_produced,avg_block_time_sec,min_block_time_sec,max_block_time_sec,stddev_block_time,total_emission_qug,unique_miners,total_tx_count,avg_txs_per_block\n".to_string(),
        }
    }

    pub fn add_row(
        &mut self,
        hour_ts: u64,
        blocks_produced: u64,
        avg_bt: f64,
        min_bt: f64,
        max_bt: f64,
        stddev_bt: f64,
        total_emission: u128,
        unique_miners: u64,
        total_tx_count: u64,
        avg_txs_per_block: f64,
    ) {
        use std::fmt::Write;
        let _ = writeln!(
            self.buf,
            "{},{},{:.4},{:.4},{:.4},{:.4},{:.12},{},{},{:.2}",
            unix_to_iso(hour_ts),
            blocks_produced,
            avg_bt,
            min_bt,
            max_bt,
            stddev_bt,
            to_qug(total_emission),
            unique_miners,
            total_tx_count,
            avg_txs_per_block,
        );
    }

    pub fn finish(self) -> String {
        self.buf
    }
}
