use super::*;
use rust_decimal::Decimal;

pub struct GridStrategy {
    name: String,
    pair: TradingPair,
    grid_levels: usize,
    price_range_percent: Decimal,
}

impl GridStrategy {
    pub fn new(name: String, pair: TradingPair, grid_levels: usize, price_range: Decimal) -> Self {
        Self { name, pair, grid_levels, price_range_percent: price_range }
    }
}

#[async_trait]
impl Strategy for GridStrategy {
    fn name(&self) -> &str { &self.name }
    async fn analyze(&mut self, ticker: &Ticker, _balance: &WalletBalance) -> Result<TradingSignal> {
        // Grid trading logic - buy low, sell high at regular intervals
        Ok(TradingSignal::Hold)
    }
    async fn on_order_filled(&mut self, _order: &Order) -> Result<()> { Ok(()) }
}
