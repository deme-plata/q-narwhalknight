/// Intra-day VWAP. Reset by calling `reset()` at session start.
pub struct Vwap {
    cum_pv: f64,
    cum_vol: f64,
}

impl Vwap {
    pub fn new() -> Self { Vwap { cum_pv: 0.0, cum_vol: 0.0 } }

    pub fn update(&mut self, typical_price: f64, volume: f64) -> f64 {
        self.cum_pv += typical_price * volume;
        self.cum_vol += volume;
        if self.cum_vol == 0.0 { return typical_price; }
        self.cum_pv / self.cum_vol
    }

    pub fn reset(&mut self) {
        self.cum_pv = 0.0;
        self.cum_vol = 0.0;
    }
}
