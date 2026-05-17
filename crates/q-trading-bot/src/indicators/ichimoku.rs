use std::collections::VecDeque;

/// Ichimoku Kinko Hyo — Tenkan, Kijun, Senkou A, Senkou B.
pub struct Ichimoku {
    tenkan_period: usize,
    kijun_period: usize,
    senkou_b_period: usize,
    highs: VecDeque<f64>,
    lows: VecDeque<f64>,
    /// Senkou A values (displaced forward, buffered here)
    senkou_a_buf: VecDeque<f64>,
    /// Senkou B values
    senkou_b_buf: VecDeque<f64>,
}

impl Ichimoku {
    pub fn new(tenkan: usize, kijun: usize, senkou_b: usize) -> Self {
        Ichimoku {
            tenkan_period: tenkan,
            kijun_period: kijun,
            senkou_b_period: senkou_b,
            highs: VecDeque::with_capacity(senkou_b + 1),
            lows: VecDeque::with_capacity(senkou_b + 1),
            senkou_a_buf: VecDeque::with_capacity(kijun + 1),
            senkou_b_buf: VecDeque::with_capacity(kijun + 1),
        }
    }

    /// Returns (tenkan, kijun, senkou_a, senkou_b).
    pub fn update(&mut self, high: f64, low: f64) -> (f64, f64, f64, f64) {
        self.highs.push_back(high);
        self.lows.push_back(low);
        if self.highs.len() > self.senkou_b_period {
            self.highs.pop_front();
            self.lows.pop_front();
        }

        let tenkan = donchian_mid(&self.highs, &self.lows, self.tenkan_period);
        let kijun = donchian_mid(&self.highs, &self.lows, self.kijun_period);
        let senkou_a_current = (tenkan + kijun) / 2.0;

        let senkou_b_high = max_n(&self.highs, self.senkou_b_period);
        let senkou_b_low = min_n(&self.lows, self.senkou_b_period);
        let senkou_b_current = (senkou_b_high + senkou_b_low) / 2.0;

        // Buffer displaced-forward cloud (use current as best proxy — correct in replay)
        self.senkou_a_buf.push_back(senkou_a_current);
        self.senkou_b_buf.push_back(senkou_b_current);
        if self.senkou_a_buf.len() > self.kijun_period {
            self.senkou_a_buf.pop_front();
            self.senkou_b_buf.pop_front();
        }

        let senkou_a = *self.senkou_a_buf.front().unwrap_or(&senkou_a_current);
        let senkou_b = *self.senkou_b_buf.front().unwrap_or(&senkou_b_current);

        (tenkan, kijun, senkou_a, senkou_b)
    }

    pub fn warmup(&mut self, candles: &[(f64, f64)]) {
        for &(h, l) in candles { self.update(h, l); }
    }
}

fn donchian_mid(highs: &VecDeque<f64>, lows: &VecDeque<f64>, period: usize) -> f64 {
    let h = max_n(highs, period);
    let l = min_n(lows, period);
    (h + l) / 2.0
}

fn max_n(vals: &VecDeque<f64>, n: usize) -> f64 {
    let start = vals.len().saturating_sub(n);
    vals.iter().skip(start).cloned().fold(f64::NEG_INFINITY, f64::max)
}

fn min_n(vals: &VecDeque<f64>, n: usize) -> f64 {
    let start = vals.len().saturating_sub(n);
    vals.iter().skip(start).cloned().fold(f64::INFINITY, f64::min)
}
