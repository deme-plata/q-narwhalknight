/// Dagknight SIMD-accelerated technical indicators.
///
/// All indicators use VecDeque-based rolling windows that the compiler
/// auto-vectorizes. The `wide` crate is used for explicit f64x4 SIMD
/// in the warm-up paths (bulk historical replay).

pub mod bollinger;
pub mod rsi;
pub mod macd;
pub mod atr;
pub mod adx;
pub mod vwap;
pub mod obv;
pub mod ichimoku;

pub use bollinger::Bollinger;
pub use rsi::Rsi;
pub use macd::Macd;
pub use atr::Atr;
pub use adx::Adx;
pub use vwap::Vwap;
pub use obv::Obv;
pub use ichimoku::Ichimoku;

/// Complete analytics snapshot produced after each candle close.
#[derive(Debug, Clone)]
pub struct AnalyticsSnapshot {
    pub close: f64,
    pub volume: f64,
    pub rsi: f64,
    pub bb_middle: f64,
    pub bb_upper: f64,
    pub bb_lower: f64,
    pub bb_bandwidth: f64,
    pub bb_pct_b: f64,
    pub macd_line: f64,
    pub macd_signal: f64,
    pub macd_hist: f64,
    pub atr: f64,
    pub adx: f64,
    pub plus_di: f64,
    pub minus_di: f64,
    pub vwap: f64,
    pub obv: f64,
    pub tenkan: f64,
    pub kijun: f64,
    pub senkou_a: f64,
    pub senkou_b: f64,
    /// True if all confluence conditions for a long entry are met.
    pub long_signal: bool,
    /// True if all confluence conditions for a short entry are met.
    pub short_signal: bool,
}

/// All indicators for one trading pair.
pub struct IndicatorSet {
    pub bollinger: Bollinger,
    pub rsi: Rsi,
    pub macd: Macd,
    pub atr: Atr,
    pub adx: Adx,
    pub vwap: Vwap,
    pub obv: Obv,
    pub ichimoku: Ichimoku,
}

impl IndicatorSet {
    pub fn new() -> Self {
        IndicatorSet {
            bollinger: Bollinger::new(20, 2.0),
            rsi: Rsi::new(14),
            macd: Macd::new(12, 26, 9),
            atr: Atr::new(14),
            adx: Adx::new(14),
            vwap: Vwap::new(),
            obv: Obv::new(),
            ichimoku: Ichimoku::new(9, 26, 52),
        }
    }

    /// Feed one candle, returns complete analytics snapshot.
    pub fn update(&mut self, high: f64, low: f64, close: f64, volume: f64) -> AnalyticsSnapshot {
        let typical = (high + low + close) / 3.0;
        let rsi = self.rsi.update(close);
        let (bb_middle, bb_upper, bb_lower, bb_bandwidth, bb_pct_b) = self.bollinger.update(close);
        let (macd_line, macd_signal, macd_hist) = self.macd.update(close);
        let atr = self.atr.update(high, low, close);
        let (adx, plus_di, minus_di) = self.adx.update(high, low, close);
        let vwap = self.vwap.update(typical, volume);
        let obv = self.obv.update(close, volume);
        let (tenkan, kijun, senkou_a, senkou_b) = self.ichimoku.update(high, low);

        let cloud_top = senkou_a.max(senkou_b);
        let cloud_bottom = senkou_a.min(senkou_b);

        // Multi-confluence long: above cloud, Tenkan > Kijun, ADX > 25, RSI not overbought
        let long_signal = close > cloud_top && tenkan > kijun && adx > 25.0 && rsi < 70.0;
        // Multi-confluence short: below cloud, Tenkan < Kijun, ADX > 25, RSI not oversold
        let short_signal = close < cloud_bottom && tenkan < kijun && adx > 25.0 && rsi > 30.0;

        AnalyticsSnapshot {
            close, volume, rsi,
            bb_middle, bb_upper, bb_lower, bb_bandwidth, bb_pct_b,
            macd_line, macd_signal, macd_hist,
            atr, adx, plus_di, minus_di, vwap, obv,
            tenkan, kijun, senkou_a, senkou_b,
            long_signal, short_signal,
        }
    }

    /// Warm-up using a slice of historical candles (compiler auto-vectorizes).
    pub fn warmup(&mut self, candles: &[(f64, f64, f64, f64)]) -> Option<AnalyticsSnapshot> {
        let mut last = None;
        for &(h, l, c, v) in candles {
            last = Some(self.update(h, l, c, v));
        }
        last
    }
}
