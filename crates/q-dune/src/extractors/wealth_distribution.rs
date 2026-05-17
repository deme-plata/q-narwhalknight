use crate::csv_formatter::{to_qug, WealthDistributionCsv};
use q_storage::StorageEngine;
use std::sync::Arc;
use tracing::debug;

/// Compute wealth distribution metrics: Gini, HHI, concentration tiers.
/// Snapshots the current balance state and produces one row per date.
pub async fn extract_wealth_distribution(
    storage: &Arc<StorageEngine>,
    date: &str,
) -> anyhow::Result<String> {
    let mut csv = WealthDistributionCsv::new();

    let balances = storage.load_wallet_balances().await?;

    // Filter out zero balances
    let mut amounts: Vec<u128> = balances.values().copied().filter(|b| *b > 0).collect();
    amounts.sort_unstable();

    let total_holders = amounts.len() as u64;
    if total_holders == 0 {
        return Ok(csv.finish());
    }

    let total_supply: f64 = amounts.iter().map(|b| to_qug(*b)).sum();

    // Gini coefficient: measures inequality (0 = perfect equality, 1 = perfect inequality)
    let gini = compute_gini(&amounts);

    // Top-10 and Top-50 concentration
    let (top10_pct, top50_pct) = {
        let n = amounts.len();
        let top10_count = (n / 10).max(1);
        let top50_count = (n / 2).max(1);

        let top10_sum: f64 = amounts[n.saturating_sub(top10_count)..].iter()
            .map(|b| to_qug(*b)).sum();
        let top50_sum: f64 = amounts[n.saturating_sub(top50_count)..].iter()
            .map(|b| to_qug(*b)).sum();

        let t10 = if total_supply > 0.0 { (top10_sum / total_supply) * 100.0 } else { 0.0 };
        let t50 = if total_supply > 0.0 { (top50_sum / total_supply) * 100.0 } else { 0.0 };
        (t10, t50)
    };

    // Tier classification (by QUG balance)
    // Whale: >= 10,000 QUG, Dolphin: >= 1,000, Fish: >= 100, Shrimp: < 100
    let whale_threshold = 10_000.0;
    let dolphin_threshold = 1_000.0;
    let fish_threshold = 100.0;

    let mut whales: u64 = 0;
    let mut dolphins: u64 = 0;
    let mut fish: u64 = 0;
    let mut shrimp: u64 = 0;

    for &balance in &amounts {
        let qug = to_qug(balance);
        if qug >= whale_threshold {
            whales += 1;
        } else if qug >= dolphin_threshold {
            dolphins += 1;
        } else if qug >= fish_threshold {
            fish += 1;
        } else {
            shrimp += 1;
        }
    }

    // Herfindahl-Hirschman Index: sum of squared market shares
    // HHI = 10,000 means perfect monopoly; HHI < 1,500 is competitive
    let hhi = if total_supply > 0.0 {
        amounts.iter()
            .map(|b| {
                let share = to_qug(*b) / total_supply;
                share * share
            })
            .sum::<f64>() * 10_000.0
    } else {
        0.0
    };

    csv.add_row(
        date,
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

    debug!(
        "[Dune] Wealth distribution for {}: {} holders, gini={:.4}, HHI={:.1}, whales={}, dolphins={}, fish={}, shrimp={}",
        date, total_holders, gini, hhi, whales, dolphins, fish, shrimp
    );
    Ok(csv.finish())
}

/// Compute the Gini coefficient from a sorted (ascending) array of values.
/// Returns 0.0 for empty or single-element arrays.
fn compute_gini(sorted_values: &[u128]) -> f64 {
    let n = sorted_values.len();
    if n <= 1 {
        return 0.0;
    }

    // Gini = (2 * Σ(i * x_i)) / (n * Σ(x_i)) - (n + 1) / n
    let total: f64 = sorted_values.iter().map(|v| *v as f64).sum();
    if total == 0.0 {
        return 0.0;
    }

    let weighted_sum: f64 = sorted_values.iter()
        .enumerate()
        .map(|(i, v)| (i + 1) as f64 * *v as f64)
        .sum();

    let gini = (2.0 * weighted_sum) / (n as f64 * total) - (n as f64 + 1.0) / n as f64;
    gini.max(0.0).min(1.0) // clamp to [0, 1]
}
