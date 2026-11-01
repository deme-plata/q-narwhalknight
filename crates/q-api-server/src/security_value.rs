//! Money figures for the security panel (`GET /api/v1/security/hashpower`), v10.11.114i.
//!
//! Until 114i the panel's pool value added BOTH reserves of every pool and multiplied the sum by the
//! QUG price, so every QUGUSD and every custom token counted as one whole QUG. Live on 2026-11-01:
//! "TVL" $32,169.08 B (the QUG/QUGUSD pool alone: 135 M QUGUSD × $22,870 = $3,097 B); the same 31
//! pools valued here hold about $1.65 B. That inflated figure × 5 % was then shown as the cost of an
//! attack ("Total Security $1,615 B") while one GPU outmined the whole network (hardware + 24 h ≈ $67).
//!
//! Dependency-free on purpose, so it can be tested without building the node:
//! `rustc --edition 2021 --test crates/q-api-server/src/security_value.rs -o /tmp/sv && /tmp/sv`

/// QUG and QUGUSD reserves carry 24 decimals.
const NATIVE_UNIT: f64 = 1e24;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NativeSide {
    Qug,
    Qugusd,
}

/// The native token a pool side names, in every spelling the node accepts ("QUG" / "native-qug",
/// "QUGUSD" / "qugusd-stable"). A custom-token address is never native, even one whose hex spells
/// "QUG" (`qnk515547…`, the QUG/QCREDIT bootstrap pool).
pub fn native_side(token: &str) -> Option<NativeSide> {
    match token.trim().to_ascii_lowercase().as_str() {
        "qug" | "native-qug" => Some(NativeSide::Qug),
        "qugusd" | "qugusd-stable" => Some(NativeSide::Qugusd),
        _ => None,
    }
}

/// USD value of one constant-product pool: twice its QUG side at `qug_price_usd`, else twice its
/// QUGUSD side at $1 (both sides of such a pool hold equal value). `None` when neither side is
/// native: the node has no price for that pool, and guessing one is how the old figure went wrong.
pub fn pool_value_usd(token0: &str, token1: &str, reserve0: u128, reserve1: u128, qug_price_usd: f64) -> Option<f64> {
    let sides = [(native_side(token0), reserve0), (native_side(token1), reserve1)];
    let amount = |want: NativeSide| {
        sides.iter().find(|(s, _)| *s == Some(want)).map(|(_, r)| *r as f64 / NATIVE_UNIT)
    };
    match (amount(NativeSide::Qug), amount(NativeSide::Qugusd)) {
        (Some(qug), _) => Some(2.0 * qug * qug_price_usd),
        (None, Some(usd)) => Some(2.0 * usd),
        (None, None) => None,
    }
}

/// Total over all pools: (USD value, pools valued, pools without a price).
pub fn total_pool_value_usd<'a, I>(pools: I, qug_price_usd: f64) -> (f64, usize, usize)
where
    I: IntoIterator<Item = (&'a str, &'a str, u128, u128)>,
{
    let (mut total, mut valued, mut unpriced) = (0.0f64, 0usize, 0usize);
    for (t0, t1, r0, r1) in pools {
        match pool_value_usd(t0, t1, r0, r1, qug_price_usd) {
            Some(v) => {
                total += v;
                valued += 1;
            }
            None => unpriced += 1,
        }
    }
    (total, valued, unpriced)
}

#[cfg(test)]
mod tests {
    use super::*;

    const PRICE: f64 = 22_870.0; // QUG in QUGUSD on 2026-11-01

    fn units(whole: u128, frac_millionths: u128) -> u128 {
        whole * 10u128.pow(24) + frac_millionths * 10u128.pow(18)
    }

    #[test]
    fn the_qug_qugusd_pool_of_2026_11_01_is_worth_millions_not_trillions() {
        let (qug, usd) = (units(5_920, 800_000), units(135_407_302, 960_000));
        let v = pool_value_usd("QUG", "QUGUSD", qug, usd, PRICE).unwrap();
        assert!((v - 270_812_000.0).abs() < 1_000_000.0, "got {v}");
        // the pre-114i formula: (r0 + r1) × price
        let old = (qug as f64 + usd as f64) / NATIVE_UNIT * PRICE;
        assert!(old > 3.0e12 && old / v > 10_000.0, "old {old}");
    }

    #[test]
    fn a_custom_token_pool_is_valued_by_its_native_side_only() {
        let v = pool_value_usd("QUG", "qnk6a600aa", units(657, 0), units(763_727_269, 0), PRICE).unwrap();
        assert_eq!(v, 2.0 * 657.0 * PRICE);
        let w = pool_value_usd("qnkabc", "qugusd-stable", units(1, 0), units(500, 0), PRICE).unwrap();
        assert_eq!(w, 1_000.0);
    }

    #[test]
    fn no_native_side_means_no_price() {
        let qug_lookalike = "qnk5155470000000000000000000000000000000000000000000000000000000000";
        let qcredit = "qnk5143524544495400000000000000000000000000000000000000000000000000";
        assert_eq!(native_side(qug_lookalike), None);
        assert_eq!(pool_value_usd(qug_lookalike, qcredit, units(10_000, 0), units(10_000, 0), PRICE), None);
    }

    #[test]
    fn every_native_spelling_is_recognised() {
        for t in ["QUG", "qug", " native-qug ", "Native-QUG"] {
            assert_eq!(native_side(t), Some(NativeSide::Qug), "{t}");
        }
        for t in ["QUGUSD", "qugusd", "qugusd-stable"] {
            assert_eq!(native_side(t), Some(NativeSide::Qugusd), "{t}");
        }
    }

    #[test]
    fn the_total_counts_valued_and_unpriced_pools() {
        let pools = [
            ("QUG", "QUGUSD", units(1, 0), units(22_870, 0)),
            ("QUGUSD", "qnkx", units(50, 0), units(9, 0)),
            ("qnka", "qnkb", units(1, 0), units(1, 0)),
        ];
        let (usd, valued, unpriced) = total_pool_value_usd(pools.iter().map(|p| (p.0, p.1, p.2, p.3)), PRICE);
        assert_eq!((valued, unpriced), (2, 1));
        assert_eq!(usd, 2.0 * PRICE + 100.0);
    }
}
