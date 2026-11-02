//! v10.11.114j — exact money amounts for the Quillon Bank.
//!
//! The bank stores QUGUSD as `u128` base units (24 decimals). Two things still turned
//! amounts into binary floating point on the way:
//!
//! 1. **Reading a request.** `q_types::u128_serde::deserialize` reads JSON through
//!    `deserialize_any`, and serde_json (built without `arbitrary_precision`) parses any
//!    integer literal above `u64::MAX` as an `f64`. A loan of 1,140 QUGUSD sent as the bare
//!    number `1140000000000000000000000000` therefore arrived as
//!    `1140000000000000064625836032`, the exact value of the nearest `f64`. Measured on the
//!    live bank 2026-11-02: two loans, 1,140 and 1,500 QUGUSD, were recorded and paid out as
//!    `…064625836032` and `1,499.999999999999882…`. Amounts sent as strings were always exact.
//! 2. **Interest.** `principal as f64 * rate * months / 12` cast back to `u128`, so the amount
//!    owed carried float noise and paying the shown amount could leave a few base units open.
//!
//! This module reads a number by its DECIMAL DIGITS. A string or an integer the deserializer
//! hands over intact is exact. A value serde_json already turned into an `f64` is read back
//! through its shortest round-trip decimal (Rust's `Display`), which is the number the client
//! wrote whenever that number has at most 15 significant digits (and usually up to 17). A
//! client that needs more digits than that must send a string. Nothing here rounds a request
//! amount: a fraction of a base unit is refused, not rounded.
//!
//! Collateral (`collateral_amount: f64` QUG) is NOT changed here on purpose: loans approved
//! before this version locked `(c * 1e24) as u128` and must get back exactly that amount.

use serde::de::{self, Deserializer, Visitor};
use std::fmt;

/// A decimal number as `mantissa × 10^exp10`, read from text without any float step.
fn decimal_parts(text: &str) -> Result<(u128, i64), String> {
    let t = text.trim();
    let t = t.strip_prefix('+').unwrap_or(t);
    if t.is_empty() {
        return Err("empty amount".into());
    }
    if t.starts_with('-') {
        return Err(format!("negative amount {t:?}"));
    }
    let (mant, exp_txt) = match t.find(['e', 'E']) {
        Some(i) => (&t[..i], Some(&t[i + 1..])),
        None => (t, None),
    };
    let (int_part, frac_part) = match mant.split_once('.') {
        Some((a, b)) => (a, b),
        None => (mant, ""),
    };
    if int_part.is_empty() && frac_part.is_empty() {
        return Err(format!("no digits in {t:?}"));
    }
    if !int_part.bytes().all(|b| b.is_ascii_digit()) || !frac_part.bytes().all(|b| b.is_ascii_digit()) {
        return Err(format!("not a decimal number: {t:?}"));
    }
    let exp: i64 = match exp_txt {
        None => 0,
        Some(e) => {
            let e2 = e.strip_prefix('+').unwrap_or(e);
            let (neg, digits) = match e2.strip_prefix('-') {
                Some(d) => (true, d),
                None => (false, e2),
            };
            if digits.is_empty() || digits.len() > 4 || !digits.bytes().all(|b| b.is_ascii_digit()) {
                return Err(format!("bad exponent in {t:?}"));
            }
            let v: i64 = digits.parse().map_err(|_| format!("bad exponent in {t:?}"))?;
            if neg { -v } else { v }
        }
    };
    // Digits without leading zeros; trailing zeros of the fraction just lower the exponent.
    let all: String = format!("{int_part}{frac_part}");
    let all = all.trim_start_matches('0');
    let mut exp10 = exp - frac_part.len() as i64;
    let mut digits = all;
    while digits.len() > 1 && digits.ends_with('0') {
        digits = &digits[..digits.len() - 1];
        exp10 += 1;
    }
    if digits.is_empty() {
        return Ok((0, 0));
    }
    let mut m: u128 = 0;
    for b in digits.bytes() {
        m = m
            .checked_mul(10)
            .and_then(|x| x.checked_add(u128::from(b - b'0')))
            .ok_or_else(|| format!("too many digits in {t:?}"))?;
    }
    Ok((m, exp10))
}

fn pow10(n: u32) -> Option<u128> {
    10u128.checked_pow(n)
}

/// Text → exact integer count of `10^-scale` units: `("1140", 24)` → 1140 × 10^24.
/// Accepts a fraction and an exponent (`"1.5"`, `"1.14e+27"`); refuses a negative number,
/// anything finer than one unit (no rounding), and anything above `u128::MAX`.
pub fn parse_decimal_units(text: &str, scale: u32) -> Result<u128, String> {
    let (m, exp10) = decimal_parts(text)?;
    if m == 0 {
        return Ok(0);
    }
    let shift = exp10 + i64::from(scale);
    if shift < 0 {
        return Err(format!("{text:?} is finer than one base unit"));
    }
    let p = u32::try_from(shift).ok().and_then(pow10).ok_or_else(|| format!("{text:?} is too large"))?;
    m.checked_mul(p).ok_or_else(|| format!("{text:?} is too large"))
}

/// An `f64` that already left the JSON parser → the whole number its shortest decimal names.
/// `1.14e27_f64` → 1140 × 10^24 exactly (an `as u128` cast gives …064625836032).
pub fn f64_units_by_digits(v: f64) -> Result<u128, String> {
    if !v.is_finite() {
        return Err(format!("amount {v} is not a number"));
    }
    // `{}` on f64 prints the shortest round-trip digits, never an exponent.
    parse_decimal_units(&format!("{v}"), 0)
}

fn gcd(mut a: u128, mut b: u128) -> u128 {
    while b != 0 {
        let r = a % b;
        a = b;
        b = r;
    }
    a
}

/// `a × b / d`, rounded half up, without forming `a × b`.
fn mul_div_round(a: u128, b: u128, d: u128) -> Option<u128> {
    if d == 0 {
        return None;
    }
    let (q, r) = (a / d, a % d);
    let hi = q.checked_mul(b)?;
    let rb = r.checked_mul(b)?;
    let (lo, rem) = (rb / d, rb % d);
    let mut out = hi.checked_add(lo)?;
    if rem.checked_mul(2)? >= d {
        out = out.checked_add(1)?;
    }
    Some(out)
}

/// Interest on `principal` base units at `rate_percent` % a year for `term_months` months:
/// principal × rate/100 × months/12, computed from the rate's decimal digits in whole numbers
/// and rounded half up to one base unit. The rate is the `f64` percent the loan record holds,
/// read by its shortest decimal (4.582856905903846 → exactly that decimal).
pub fn interest_units(principal: u128, rate_percent: f64, term_months: u32) -> Result<u128, String> {
    if !rate_percent.is_finite() || rate_percent < 0.0 {
        return Err(format!("bad interest rate {rate_percent}"));
    }
    let (m, exp10) = decimal_parts(&format!("{rate_percent}"))?;
    if m == 0 || principal == 0 || term_months == 0 {
        return Ok(0);
    }
    // rate_percent = m × 10^exp10 ; interest = principal × m × 10^exp10 × months / 1200
    let mut num = m.checked_mul(u128::from(term_months)).ok_or("interest overflow")?;
    let mut den: u128 = 1200;
    if exp10 >= 0 {
        let p = u32::try_from(exp10).ok().and_then(pow10).ok_or("interest overflow")?;
        num = num.checked_mul(p).ok_or("interest overflow")?;
    } else {
        let p = u32::try_from(-exp10).ok().and_then(pow10).ok_or("interest overflow")?;
        den = den.checked_mul(p).ok_or("interest overflow")?;
    }
    let g = gcd(num, den);
    let (num, den) = (num / g, den / g);
    mul_div_round(principal, num, den).ok_or_else(|| "interest overflow".to_string())
}

/// Principal + interest for a loan record: what closes the loan.
pub fn total_owed_units(principal: u128, rate_percent: f64, term_months: u32) -> Result<u128, String> {
    let interest = interest_units(principal, rate_percent, term_months)?;
    principal.checked_add(interest).ok_or_else(|| "total owed overflow".to_string())
}

/// `#[serde(deserialize_with = "crate::bank_amounts::exact_u128")]` — a whole number of base
/// units from a JSON string or number, read by its digits (see the module comment).
pub fn exact_u128<'de, D>(deserializer: D) -> Result<u128, D::Error>
where
    D: Deserializer<'de>,
{
    if deserializer.is_human_readable() {
        deserializer.deserialize_any(ExactU128)
    } else {
        deserializer.deserialize_str(ExactU128)
    }
}

struct ExactU128;

impl<'de> Visitor<'de> for ExactU128 {
    type Value = u128;

    fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("a non-negative whole number of base units, as a string or a number")
    }
    fn visit_u64<E: de::Error>(self, v: u64) -> Result<u128, E> {
        Ok(u128::from(v))
    }
    fn visit_u128<E: de::Error>(self, v: u128) -> Result<u128, E> {
        Ok(v)
    }
    fn visit_i64<E: de::Error>(self, v: i64) -> Result<u128, E> {
        u128::try_from(v).map_err(|_| E::custom("negative amount"))
    }
    fn visit_i128<E: de::Error>(self, v: i128) -> Result<u128, E> {
        u128::try_from(v).map_err(|_| E::custom("negative amount"))
    }
    fn visit_f64<E: de::Error>(self, v: f64) -> Result<u128, E> {
        f64_units_by_digits(v).map_err(E::custom)
    }
    fn visit_str<E: de::Error>(self, v: &str) -> Result<u128, E> {
        parse_decimal_units(v, 0).map_err(E::custom)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const UNIT: u128 = 1_000_000_000_000_000_000_000_000; // 10^24

    #[derive(serde::Deserialize)]
    struct Req {
        #[serde(deserialize_with = "exact_u128")]
        amount: u128,
    }
    fn read(json: &str) -> Result<u128, String> {
        serde_json::from_str::<Req>(json).map(|r| r.amount).map_err(|e| e.to_string())
    }

    #[test]
    fn the_two_live_loans_come_out_exact_when_sent_as_bare_numbers() {
        // What the wallet tool sent on 2026-10-31 and 2026-11-02.
        assert_eq!(read(r#"{"amount":1140000000000000000000000000}"#).unwrap(), 1140 * UNIT);
        assert_eq!(read(r#"{"amount":1500000000000000000000000000}"#).unwrap(), 1500 * UNIT);
        // The old path: the same bare number read as f64 and cast — the noise we saw live.
        assert_eq!(1140000000000000000000000000f64 as u128, 1140000000000000064625836032);
        assert_eq!(1500000000000000000000000000f64 as u128, 1499999999999999882492379136);
    }

    #[test]
    fn strings_and_small_numbers_are_exact_and_scientific_text_too() {
        assert_eq!(read(r#"{"amount":"1140000000000000064625836032"}"#).unwrap(), 1140000000000000064625836032);
        assert_eq!(read(r#"{"amount":42}"#).unwrap(), 42);
        assert_eq!(read(r#"{"amount":"1.14e+27"}"#).unwrap(), 1140 * UNIT);
        assert_eq!(read(r#"{"amount":1.14e27}"#).unwrap(), 1140 * UNIT); // JSON.stringify(1140*1e24)
        assert_eq!(read(r#"{"amount":"0"}"#).unwrap(), 0);
    }

    #[test]
    fn negative_fractional_and_oversized_amounts_are_refused_not_rounded() {
        assert!(read(r#"{"amount":-5}"#).is_err());
        assert!(read(r#"{"amount":"-5"}"#).is_err());
        assert!(read(r#"{"amount":0.5}"#).is_err());
        assert!(read(r#"{"amount":"12.5"}"#).is_err());
        assert!(read(r#"{"amount":"1e39"}"#).is_err());
        assert!(read(r#"{"amount":"abc"}"#).is_err());
        assert!(read(r#"{"amount":""}"#).is_err());
    }

    #[test]
    fn display_amounts_convert_to_base_units_exactly() {
        assert_eq!(parse_decimal_units("1140", 24).unwrap(), 1140 * UNIT);
        assert_eq!(parse_decimal_units("0.1", 24).unwrap(), UNIT / 10);
        assert_eq!(parse_decimal_units("1144.3537140606086", 24).unwrap(), 1144353714060608600000000000);
        assert!(parse_decimal_units("0.0000000000000000000000001", 24).is_err());
        assert_eq!(f64_units_by_digits(1e23).unwrap(), UNIT / 10);
        // Reading digits cannot undo arithmetic a client already did in floats:
        // 0.1 * 1e24 is the f64 1.0000000000000001e23. Clients must scale by text.
        assert_eq!(f64_units_by_digits(0.1 * 1e24).unwrap(), 100000000000000010000000);
    }

    #[test]
    fn interest_is_whole_numbers_and_matches_the_rate_as_written() {
        // 1,200 QUGUSD at 5 % for 12 months = 60 QUGUSD, exactly.
        assert_eq!(interest_units(1200 * UNIT, 5.0, 12).unwrap(), 60 * UNIT);
        // 1,000 at 6 % for 1 month = 5 QUGUSD.
        assert_eq!(interest_units(1000 * UNIT, 6.0, 1).unwrap(), 5 * UNIT);
        // Rocky's 1,140 loan (rate as stored): 1140 × 4.582856905903846 / 1200, half up.
        let i = interest_units(1140 * UNIT, 4.582856905903846, 1).unwrap();
        assert_eq!(i, 4353714060608653700000000);
        // Half-up rounding at one base unit: 1 unit at 50 % for 12 months = 0.5 → 1.
        assert_eq!(interest_units(1, 50.0, 12).unwrap(), 1);
        assert_eq!(interest_units(1, 49.0, 12).unwrap(), 0);
        assert_eq!(interest_units(1140 * UNIT, 0.0, 1).unwrap(), 0);
        assert!(interest_units(UNIT, f64::NAN, 1).is_err());
    }

    #[test]
    fn paying_the_total_owed_exactly_closes_the_loan_with_no_dust() {
        // The stored (noisy) principal of the live 1,140 loan.
        let p = 1140000000000000064625836032u128;
        let owed = total_owed_units(p, 4.582856905903846, 1).unwrap();
        assert_eq!(owed, p + interest_units(p, 4.582856905903846, 1).unwrap());
        // Large principals do not overflow: 10^12 QUGUSD at 99.999 % for 120 months.
        assert!(total_owed_units(1_000_000_000_000 * UNIT, 99.999, 120).is_ok());
    }
}
