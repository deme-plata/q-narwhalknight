pub struct Atr {
    period: f64,
    atr: f64,
    prev_close: Option<f64>,
    count: usize,
}

impl Atr {
    pub fn new(period: usize) -> Self {
        Atr { period: period as f64, atr: 0.0, prev_close: None, count: 0 }
    }

    pub fn update(&mut self, high: f64, low: f64, close: f64) -> f64 {
        let tr = match self.prev_close {
            Some(pc) => (high - low).max((high - pc).abs()).max((low - pc).abs()),
            None => high - low,
        };
        self.prev_close = Some(close);
        if self.count == 0 {
            self.atr = tr;
        } else {
            self.atr = (self.atr * (self.period - 1.0) + tr) / self.period;
        }
        self.count += 1;
        self.atr
    }

    pub fn warmup(&mut self, candles: &[(f64, f64, f64)]) {
        for &(h, l, c) in candles { self.update(h, l, c); }
    }
}
