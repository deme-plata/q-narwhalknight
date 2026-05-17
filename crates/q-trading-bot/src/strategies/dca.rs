use super::*;

pub struct DCAStrategy {
    name: String,
}

#[async_trait]
impl Strategy for DCAStrategy {
    fn name(&self) -> &str { &self.name }
    async fn analyze(&mut self, _ticker: &Ticker, _balance: &WalletBalance) -> Result<TradingSignal> {
        Ok(TradingSignal::Hold)
    }
    async fn on_order_filled(&mut self, _order: &Order) -> Result<()> { Ok(()) }
}
