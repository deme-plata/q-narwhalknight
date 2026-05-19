//! Deterministic, O(1) tx + swap scorers. Surfaces a normalised 0..10 score
//! plus per-component breakdown on tx-receipts and DEX swap responses. UI
//! renders the breakdown in TransactionDetailsModal + SwapSuccessModal.
//!
//! Single source of truth for ScoreReport — handlers.rs and dex_integration_api.rs
//! re-export from here. Don't define duplicate ScoreReport types elsewhere.

use q_types::Transaction;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoreReport {
    /// Weighted total, normalised to 0..10 (UI renders with green ≥7, amber 4-6, red <4).
    pub total: f64,
    pub components: Vec<ScoreComponent>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoreComponent {
    pub name: String,
    /// Component value 0..10.
    pub value: f64,
    /// Weight 0..1 — contribution to total.
    pub weight: f64,
    pub explanation: String,
}

#[derive(Debug, Clone, Copy)]
pub struct TxContext {
    pub sender_balance_before: f64,
    pub sender_balance_after: f64,
    pub fee_paid: f64,
    pub amount: f64,
    pub mempool_backlog_ratio: f64,
    pub reserve_utilization_ratio: f64,
}

#[derive(Debug, Clone, Copy)]
pub struct SwapRecord {
    pub amount_in: f64,
    pub amount_out: f64,
    pub expected_out: f64,
    pub fee_paid: f64,
}

#[derive(Debug, Clone, Copy)]
pub struct SwapContext {
    pub reserve_in: f64,
    pub reserve_out: f64,
    pub pre_swap_price: f64,
    pub post_swap_price: f64,
    pub volatility_ratio: f64,
}

pub struct TxScorer;
pub struct SwapScorer;

impl TxScorer {
    pub fn score_tx(_tx: &Transaction, ctx: &TxContext) -> ScoreReport {
        const COMPONENTS: [(&str, f64); 4] = [
            ("balance_delta_health", 0.35),
            ("fee_burden", 0.20),
            ("mempool_pressure", 0.20),
            ("reserve_utilization", 0.25),
        ];

        let balance_delta = (ctx.sender_balance_before - ctx.sender_balance_after).max(0.0);
        let balance_delta_health = unit_interval(1.0 - (balance_delta / (ctx.sender_balance_before + 1.0)));
        let fee_burden = unit_interval(1.0 - (ctx.fee_paid / (ctx.amount.abs() + 1.0)));
        let mempool_pressure = unit_interval(1.0 - ctx.mempool_backlog_ratio);
        let reserve_utilization = unit_interval(1.0 - ctx.reserve_utilization_ratio);

        let values = [
            balance_delta_health,
            fee_burden,
            mempool_pressure,
            reserve_utilization,
        ];

        let mut components = Vec::with_capacity(COMPONENTS.len());
        let mut total_unit = 0.0;

        for ((name, weight), value) in COMPONENTS.into_iter().zip(values) {
            total_unit += value * weight;
            components.push(ScoreComponent {
                name: name.to_string(),
                value: value * 10.0,
                weight,
                explanation: tx_explanation(name, value),
            });
        }

        ScoreReport {
            total: unit_interval(total_unit) * 10.0,
            components,
        }
    }
}

impl SwapScorer {
    pub fn score_swap(swap: &SwapRecord, ctx: &SwapContext) -> ScoreReport {
        const COMPONENTS: [(&str, f64); 4] = [
            ("execution_quality", 0.35),
            ("fee_efficiency", 0.20),
            ("price_impact", 0.25),
            ("reserve_depth", 0.20),
        ];

        let execution_quality = unit_interval(swap.amount_out / (swap.expected_out + 1.0));
        let fee_efficiency = unit_interval(1.0 - (swap.fee_paid / (swap.amount_in.abs() + 1.0)));
        let price_impact = unit_interval(1.0 - ((ctx.post_swap_price - ctx.pre_swap_price).abs() / (ctx.pre_swap_price.abs() + 1.0)));
        let reserve_depth = unit_interval((ctx.reserve_in + ctx.reserve_out) / ((ctx.reserve_in + ctx.reserve_out) + swap.amount_in.abs() + 1.0));

        let volatility_penalty = unit_interval(1.0 - ctx.volatility_ratio);
        let values = [
            execution_quality * volatility_penalty,
            fee_efficiency,
            price_impact * volatility_penalty,
            reserve_depth,
        ];

        let mut components = Vec::with_capacity(COMPONENTS.len());
        let mut total_unit = 0.0;

        for ((name, weight), value) in COMPONENTS.into_iter().zip(values) {
            total_unit += value * weight;
            components.push(ScoreComponent {
                name: name.to_string(),
                value: value * 10.0,
                weight,
                explanation: swap_explanation(name, value),
            });
        }

        ScoreReport {
            total: unit_interval(total_unit) * 10.0,
            components,
        }
    }
}

#[inline]
fn unit_interval(v: f64) -> f64 {
    v.clamp(0.0, 1.0)
}

/// Human-readable explanation per tx component. Deterministic — chosen by
/// thresholding `value` (0..1 input).
fn tx_explanation(name: &str, value: f64) -> String {
    match name {
        "balance_delta_health" if value >= 0.8 => "sender retains most of balance after tx".into(),
        "balance_delta_health" if value >= 0.4 => "sender's balance drops noticeably from tx".into(),
        "balance_delta_health" => "sender spends nearly all balance on this tx".into(),
        "fee_burden" if value >= 0.8 => "fee is minor relative to amount".into(),
        "fee_burden" if value >= 0.4 => "fee is a moderate share of amount".into(),
        "fee_burden" => "fee dominates the tx amount".into(),
        "mempool_pressure" if value >= 0.7 => "mempool has plenty of headroom".into(),
        "mempool_pressure" if value >= 0.3 => "mempool is partially backlogged".into(),
        "mempool_pressure" => "mempool is heavily backlogged — inclusion may delay".into(),
        "reserve_utilization" if value >= 0.7 => "node reserves are healthy".into(),
        "reserve_utilization" if value >= 0.3 => "node reserves are partly utilised".into(),
        "reserve_utilization" => "node reserves are stressed".into(),
        _ => format!("{name} = {:.2}", value),
    }
}

/// Human-readable explanation per swap component.
fn swap_explanation(name: &str, value: f64) -> String {
    match name {
        "execution_quality" if value >= 0.95 => "got virtually the expected output".into(),
        "execution_quality" if value >= 0.8 => "got close to the expected output".into(),
        "execution_quality" => "received significantly less than expected".into(),
        "fee_efficiency" if value >= 0.9 => "0.3% AMM fee is the only cost".into(),
        "fee_efficiency" if value >= 0.5 => "fee is a noticeable share of trade".into(),
        "fee_efficiency" => "fee is high relative to trade size".into(),
        "price_impact" if value >= 0.95 => "price barely moved (deep pool relative to trade)".into(),
        "price_impact" if value >= 0.7 => "moderate price movement".into(),
        "price_impact" => "large price impact — trade ate the pool".into(),
        "reserve_depth" if value >= 0.95 => "trade is small relative to pool depth".into(),
        "reserve_depth" if value >= 0.6 => "trade meaningfully utilises pool depth".into(),
        "reserve_depth" => "trade is large relative to pool depth".into(),
        _ => format!("{name} = {:.2}", value),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dummy_tx() -> Transaction {
        Transaction {
            id: [0u8; 32],
            from: [0u8; 32],
            to: [1u8; 32],
            amount: 0,
            fee: 0,
            nonce: 0,
            signature: vec![],
            timestamp: chrono::Utc::now(),
            data: vec![],
            token_type: q_types::TokenType::Qug,
            fee_token_type: q_types::TokenType::Qug,
            tx_type: q_types::TransactionType::Transfer,
            pqc_signature: None,
            signature_phase: q_types::TxSignaturePhase::Phase0Ed25519,
            pqc_public_key: None,
        }
    }

    #[test]
    fn tx_score_in_0_10_range() {
        let ctx = TxContext {
            sender_balance_before: 100.0,
            sender_balance_after: 99.5,
            fee_paid: 0.001,
            amount: 0.5,
            mempool_backlog_ratio: 0.1,
            reserve_utilization_ratio: 0.2,
        };
        let report = TxScorer::score_tx(&dummy_tx(), &ctx);
        assert!((0.0..=10.0).contains(&report.total), "total {} out of range", report.total);
        assert_eq!(report.components.len(), 4);
        for c in &report.components {
            assert!((0.0..=10.0).contains(&c.value), "component {} value {} out of range", c.name, c.value);
        }
    }

    #[test]
    fn tx_score_drains_sender_low_health() {
        let ctx = TxContext {
            sender_balance_before: 1.0,
            sender_balance_after: 0.0,
            fee_paid: 0.0,
            amount: 1.0,
            mempool_backlog_ratio: 0.0,
            reserve_utilization_ratio: 0.0,
        };
        let report = TxScorer::score_tx(&dummy_tx(), &ctx);
        let bdh = report.components.iter().find(|c| c.name == "balance_delta_health").unwrap();
        // Sender drained — balance_delta_health should be < 5.0 (out of 10).
        assert!(bdh.value < 5.0, "expected low balance_delta_health, got {}", bdh.value);
    }

    #[test]
    fn swap_score_high_quality_low_impact() {
        let swap = SwapRecord {
            amount_in: 1.0,
            amount_out: 1.99,
            expected_out: 2.0,
            fee_paid: 0.003,
        };
        let ctx = SwapContext {
            reserve_in: 1000.0,
            reserve_out: 2000.0,
            pre_swap_price: 2.0,
            post_swap_price: 1.998,
            volatility_ratio: 0.0,
        };
        let report = SwapScorer::score_swap(&swap, &ctx);
        assert!(report.total >= 7.0, "deep-pool low-volatility swap should score ≥ 7, got {}", report.total);
        let pi = report.components.iter().find(|c| c.name == "price_impact").unwrap();
        assert!(pi.value >= 8.0, "expected high price_impact score, got {}", pi.value);
    }

    #[test]
    fn swap_score_shallow_pool_low() {
        let swap = SwapRecord {
            amount_in: 100.0,
            amount_out: 50.0,
            expected_out: 90.0,
            fee_paid: 1.0,
        };
        let ctx = SwapContext {
            reserve_in: 50.0,
            reserve_out: 50.0,
            pre_swap_price: 1.0,
            post_swap_price: 0.5,
            volatility_ratio: 0.7,
        };
        let report = SwapScorer::score_swap(&swap, &ctx);
        assert!(report.total < 5.0, "high-impact shallow-pool swap should score < 5, got {}", report.total);
    }
}
