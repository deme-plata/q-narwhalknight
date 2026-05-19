use q_types::Transaction;

#[derive(Debug, Clone)]
pub struct ScoreReport {
    pub total: f64,
    pub components: Vec<ScoreComponent>,
}

#[derive(Debug, Clone)]
pub struct ScoreComponent {
    pub name: &'static str,
    pub value: f64,
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
        let mut total = 0.0;

        for ((name, weight), value) in COMPONENTS.into_iter().zip(values) {
            total += value * weight;
            components.push(ScoreComponent {
                name,
                value,
                weight,
                explanation: tx_explanation(name, value),
            });
        }

        ScoreReport {
            total: unit_interval(total),
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
        let mut total = 0.0;

        for ((name, weight), value) in COMPONENTS.into_iter().zip(values) {
            total += value * weight;
            components.push(ScoreComponent {
                name,
                value,
                weight,
                explanation: swap_explanation(name, value),
            });
        }

        ScoreReport {
            total: unit_interval(total),
            components,
        }
    }
}

#[inline]
fn unit_interval(v: f64) -> f64 {
    v.clamp(0.0, 1.0)
}

fn tx_explanation(name: &'static str, value: f64) -> String {
    format!("{name} component evaluated deterministically at {:.3}", value)
}

fn swap_explanation(name: &'static str, value: f64) -> String {
    format!("{name} component evaluated deterministically at {:.3}", value)
}
