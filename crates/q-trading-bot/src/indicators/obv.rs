pub struct Obv {
    obv: f64,
    prev_close: Option<f64>,
}

impl Obv {
    pub fn new() -> Self { Obv { obv: 0.0, prev_close: None } }

    pub fn update(&mut self, close: f64, volume: f64) -> f64 {
        if let Some(pc) = self.prev_close {
            if close > pc { self.obv += volume; }
            else if close < pc { self.obv -= volume; }
        }
        self.prev_close = Some(close);
        self.obv
    }
}
