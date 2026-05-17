/// Wilder's ADX with +DI/-DI.
pub struct Adx {
    period: f64,
    prev_high: Option<f64>,
    prev_low: Option<f64>,
    prev_close: Option<f64>,
    smooth_tr: f64,
    smooth_dm_plus: f64,
    smooth_dm_minus: f64,
    adx: f64,
    count: usize,
}

impl Adx {
    pub fn new(period: usize) -> Self {
        Adx {
            period: period as f64,
            prev_high: None,
            prev_low: None,
            prev_close: None,
            smooth_tr: 0.0,
            smooth_dm_plus: 0.0,
            smooth_dm_minus: 0.0,
            adx: 0.0,
            count: 0,
        }
    }

    /// Returns (adx, +di, -di).
    pub fn update(&mut self, high: f64, low: f64, close: f64) -> (f64, f64, f64) {
        if self.prev_high.is_none() {
            self.prev_high = Some(high);
            self.prev_low = Some(low);
            self.prev_close = Some(close);
            return (0.0, 0.0, 0.0);
        }
        let ph = self.prev_high.unwrap();
        let pl = self.prev_low.unwrap();
        let pc = self.prev_close.unwrap();

        let tr = (high - low).max((high - pc).abs()).max((low - pc).abs());
        let dm_plus = if high - ph > pl - low { (high - ph).max(0.0) } else { 0.0 };
        let dm_minus = if pl - low > high - ph { (pl - low).max(0.0) } else { 0.0 };

        let p = self.period;
        if self.count == 0 {
            self.smooth_tr = tr;
            self.smooth_dm_plus = dm_plus;
            self.smooth_dm_minus = dm_minus;
        } else {
            self.smooth_tr = self.smooth_tr - self.smooth_tr / p + tr;
            self.smooth_dm_plus = self.smooth_dm_plus - self.smooth_dm_plus / p + dm_plus;
            self.smooth_dm_minus = self.smooth_dm_minus - self.smooth_dm_minus / p + dm_minus;
        }
        self.count += 1;

        let plus_di = if self.smooth_tr == 0.0 { 0.0 } else { 100.0 * self.smooth_dm_plus / self.smooth_tr };
        let minus_di = if self.smooth_tr == 0.0 { 0.0 } else { 100.0 * self.smooth_dm_minus / self.smooth_tr };
        let dx = if plus_di + minus_di == 0.0 { 0.0 } else { 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di) };

        if self.count <= p as usize {
            self.adx = (self.adx * (self.count as f64 - 1.0) + dx) / self.count as f64;
        } else {
            self.adx = (self.adx * (p - 1.0) + dx) / p;
        }

        self.prev_high = Some(high);
        self.prev_low = Some(low);
        self.prev_close = Some(close);
        (self.adx, plus_di, minus_di)
    }

    pub fn warmup(&mut self, candles: &[(f64, f64, f64)]) {
        for &(h, l, c) in candles { self.update(h, l, c); }
    }
}
