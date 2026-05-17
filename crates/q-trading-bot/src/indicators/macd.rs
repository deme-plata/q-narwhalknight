pub struct Macd {
    fast_ema: f64,
    slow_ema: f64,
    signal_ema: f64,
    fast_k: f64,
    slow_k: f64,
    signal_k: f64,
    count: usize,
}

impl Macd {
    pub fn new(fast: usize, slow: usize, signal: usize) -> Self {
        Macd {
            fast_ema: 0.0,
            slow_ema: 0.0,
            signal_ema: 0.0,
            fast_k: 2.0 / (fast as f64 + 1.0),
            slow_k: 2.0 / (slow as f64 + 1.0),
            signal_k: 2.0 / (signal as f64 + 1.0),
            count: 0,
        }
    }

    /// Returns (macd_line, signal_line, histogram).
    pub fn update(&mut self, price: f64) -> (f64, f64, f64) {
        if self.count == 0 {
            self.fast_ema = price;
            self.slow_ema = price;
        } else {
            self.fast_ema = price * self.fast_k + self.fast_ema * (1.0 - self.fast_k);
            self.slow_ema = price * self.slow_k + self.slow_ema * (1.0 - self.slow_k);
        }
        self.count += 1;
        let macd_line = self.fast_ema - self.slow_ema;
        if self.count == 1 {
            self.signal_ema = macd_line;
        } else {
            self.signal_ema = macd_line * self.signal_k + self.signal_ema * (1.0 - self.signal_k);
        }
        let hist = macd_line - self.signal_ema;
        (macd_line, self.signal_ema, hist)
    }

    pub fn warmup(&mut self, prices: &[f64]) {
        for &p in prices { self.update(p); }
    }
}
