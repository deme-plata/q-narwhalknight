use std::collections::VecDeque;
use wide::f64x4;

pub struct Bollinger {
    period: usize,
    multiplier: f64,
    values: VecDeque<f64>,
    sum: f64,
    sum_sq: f64,
}

impl Bollinger {
    pub fn new(period: usize, multiplier: f64) -> Self {
        Bollinger { period, multiplier, values: VecDeque::with_capacity(period + 1), sum: 0.0, sum_sq: 0.0 }
    }

    /// Returns (middle, upper, lower, bandwidth, %b).
    pub fn update(&mut self, price: f64) -> (f64, f64, f64, f64, f64) {
        self.values.push_back(price);
        self.sum += price;
        self.sum_sq += price * price;

        if self.values.len() > self.period {
            let old = self.values.pop_front().unwrap();
            self.sum -= old;
            self.sum_sq -= old * old;
        }

        let n = self.values.len() as f64;
        if n < self.period as f64 {
            return (price, price, price, 0.0, 0.5);
        }

        let middle = self.sum / n;
        let variance = ((self.sum_sq / n) - (middle * middle)).max(0.0);
        let std_dev = variance.sqrt();
        let upper = middle + self.multiplier * std_dev;
        let lower = middle - self.multiplier * std_dev;
        let bandwidth = upper - lower;
        let pct_b = if bandwidth == 0.0 { 0.5 } else { (price - lower) / bandwidth };
        (middle, upper, lower, bandwidth, pct_b)
    }

    /// SIMD warm-up: process 4 prices per iteration using f64x4.
    pub fn warmup_simd(&mut self, prices: &[f64]) {
        let chunks = prices.chunks_exact(4);
        let remainder = chunks.remainder();
        for chunk in chunks {
            let v = f64x4::from([chunk[0], chunk[1], chunk[2], chunk[3]]);
            let arr = v.to_array();
            for &p in &arr { self.update(p); }
        }
        for &p in remainder { self.update(p); }
    }
}
