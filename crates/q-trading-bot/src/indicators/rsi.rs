pub struct Rsi {
    period: usize,
    avg_gain: f64,
    avg_loss: f64,
    prev: Option<f64>,
    count: usize,
}

impl Rsi {
    pub fn new(period: usize) -> Self {
        Rsi { period, avg_gain: 0.0, avg_loss: 0.0, prev: None, count: 0 }
    }

    pub fn update(&mut self, price: f64) -> f64 {
        if let Some(prev) = self.prev {
            let change = price - prev;
            let gain = change.max(0.0);
            let loss = (-change).max(0.0);
            let p = self.period as f64;
            if self.count < self.period {
                self.avg_gain += gain / p;
                self.avg_loss += loss / p;
            } else {
                self.avg_gain = (self.avg_gain * (p - 1.0) + gain) / p;
                self.avg_loss = (self.avg_loss * (p - 1.0) + loss) / p;
            }
            self.count += 1;
        }
        self.prev = Some(price);
        if self.avg_loss == 0.0 { return 100.0; }
        100.0 - 100.0 / (1.0 + self.avg_gain / self.avg_loss)
    }

    pub fn warmup(&mut self, prices: &[f64]) {
        for &p in prices { self.update(p); }
    }
}
