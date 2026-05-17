//! Privacy-preserving log redaction for Q-NarwhalKnight.
//!
//! Controlled by `Q_LOG_PRIVACY` environment variable:
//! - `full` (default): addresses masked, amounts bucketed, hashes truncated
//! - `partial`: addresses shortened, amounts approximated, hashes full
//! - `none`: raw values (debugging only)

use once_cell::sync::Lazy;
use regex::Regex;
use std::fmt;
use std::sync::atomic::{AtomicU8, Ordering};

// ---------------------------------------------------------------------------
// Privacy level
// ---------------------------------------------------------------------------

const LEVEL_FULL: u8 = 2;
const LEVEL_PARTIAL: u8 = 1;
const LEVEL_NONE: u8 = 0;

static PRIVACY_LEVEL: AtomicU8 = AtomicU8::new(LEVEL_FULL);

/// Read `Q_LOG_PRIVACY` and set the global privacy level.
/// Call once at startup, before tracing initialisation.
pub fn init_privacy_level() {
    let level = match std::env::var("Q_LOG_PRIVACY")
        .unwrap_or_default()
        .to_lowercase()
        .as_str()
    {
        "none" | "off" | "0" => LEVEL_NONE,
        "partial" | "1" => LEVEL_PARTIAL,
        _ => LEVEL_FULL, // default
    };
    PRIVACY_LEVEL.store(level, Ordering::Relaxed);
    if level == LEVEL_NONE {
        tracing::warn!("Q_LOG_PRIVACY=none — logs will contain raw addresses and amounts (debug only)");
    }
}

#[inline(always)]
fn privacy() -> u8 {
    PRIVACY_LEVEL.load(Ordering::Relaxed)
}

// ---------------------------------------------------------------------------
// Mask helpers
// ---------------------------------------------------------------------------

/// Mask a hex-encoded wallet address (e.g. 64-char hex string).
///
/// - `full`:    `"qnkefca…0723"` (first 7 + last 4)
/// - `partial`: first 8 + `"…"` + last 5
/// - `none`:    unchanged
pub fn mask_addr(addr: &str) -> String {
    match privacy() {
        LEVEL_FULL => {
            let clean = addr.trim_start_matches("0x");
            if clean.len() <= 12 {
                return "qnk***".to_string();
            }
            format!("qnk{}…{}", &clean[..4], &clean[clean.len() - 4..])
        }
        LEVEL_PARTIAL => {
            let clean = addr.trim_start_matches("0x");
            if clean.len() <= 14 {
                return format!("{}…", &clean[..clean.len().min(8)]);
            }
            format!("{}…{}", &clean[..8], &clean[clean.len() - 5..])
        }
        _ => addr.to_string(),
    }
}

/// Mask raw address bytes.
pub fn mask_addr_bytes(bytes: &[u8]) -> String {
    mask_addr(&hex::encode(bytes))
}

/// Mask a base-unit amount (u128, typically 24-decimal fixed-point).
///
/// - `full`:    bucket like `[1K-10K]`
/// - `partial`: approximate like `~1.2K`
/// - `none`:    raw integer
pub fn mask_amt(amount: u128) -> String {
    match privacy() {
        LEVEL_FULL => bucket_amount(amount),
        LEVEL_PARTIAL => approx_amount(amount),
        _ => amount.to_string(),
    }
}

/// Mask a display-unit amount (f64, human-readable QUG).
pub fn mask_amt_display(amount: f64) -> String {
    match privacy() {
        LEVEL_FULL => bucket_display(amount),
        LEVEL_PARTIAL => approx_display(amount),
        _ => format!("{amount}"),
    }
}

/// Mask a transaction or block hash.
///
/// - `full`:    first 4 + `"…"` + last 4
/// - `partial`: full hash
/// - `none`:    full hash
pub fn mask_hash(hash: &str) -> String {
    match privacy() {
        LEVEL_FULL => {
            if hash.len() <= 10 {
                return hash.to_string();
            }
            format!("{}…{}", &hash[..4], &hash[hash.len() - 4..])
        }
        _ => hash.to_string(),
    }
}

// -- internal helpers -------------------------------------------------------

/// 24-decimal base-unit → display QUG.
const DECIMALS_24: f64 = 1e24;

fn to_display(amount: u128) -> f64 {
    amount as f64 / DECIMALS_24
}

fn bucket_amount(amount: u128) -> String {
    bucket_display(to_display(amount))
}

fn approx_amount(amount: u128) -> String {
    approx_display(to_display(amount))
}

fn bucket_display(v: f64) -> String {
    if v <= 0.0 {
        "[0]".to_string()
    } else if v < 0.01 {
        "[<0.01]".to_string()
    } else if v < 1.0 {
        "[0.01-1]".to_string()
    } else if v < 10.0 {
        "[1-10]".to_string()
    } else if v < 100.0 {
        "[10-100]".to_string()
    } else if v < 1_000.0 {
        "[100-1K]".to_string()
    } else if v < 10_000.0 {
        "[1K-10K]".to_string()
    } else if v < 100_000.0 {
        "[10K-100K]".to_string()
    } else if v < 1_000_000.0 {
        "[100K-1M]".to_string()
    } else {
        "[>1M]".to_string()
    }
}

fn approx_display(v: f64) -> String {
    if v <= 0.0 {
        "~0".to_string()
    } else if v < 1.0 {
        format!("~{:.2}", v)
    } else if v < 1_000.0 {
        format!("~{:.1}", v)
    } else if v < 1_000_000.0 {
        format!("~{:.1}K", v / 1_000.0)
    } else {
        format!("~{:.1}M", v / 1_000_000.0)
    }
}

// ---------------------------------------------------------------------------
// Safety-net RedactedFormatter — post-processes formatted log output
// ---------------------------------------------------------------------------

// Lazy-compiled regexes for catching sensitive data that slipped past manual masking.
static RE_HEX64: Lazy<Regex> = Lazy::new(|| {
    // 64-char hex strings (wallet addresses, tx hashes)
    Regex::new(r"\b[0-9a-fA-F]{64}\b").unwrap()
});
static RE_HEX40: Lazy<Regex> = Lazy::new(|| {
    // 40-char hex strings (shorter addresses)
    Regex::new(r"\b[0-9a-fA-F]{40}\b").unwrap()
});
static RE_QUG_AMOUNT: Lazy<Regex> = Lazy::new(|| {
    // Patterns like "1234.56 QUG" or "1234.56QUG"
    Regex::new(r"\b\d+\.?\d*\s?QUG\b").unwrap()
});
static RE_LARGE_NUMBER: Lazy<Regex> = Lazy::new(|| {
    // Very large integers (>= 15 digits) that look like base-unit amounts
    Regex::new(r"\b\d{15,}\b").unwrap()
});

/// Apply safety-net regex redaction to a formatted log line.
/// Returns the redacted string. Cheap no-op when privacy == none.
pub fn redact_line(line: &str) -> String {
    if privacy() == LEVEL_NONE {
        return line.to_string();
    }

    let mut out = line.to_string();

    // Replace 64-char hex (addresses/hashes)
    out = RE_HEX64.replace_all(&out, "qnk[REDACTED]").to_string();
    // Replace 40-char hex
    out = RE_HEX40.replace_all(&out, "qnk[REDACTED]").to_string();
    // Replace QUG amounts
    out = RE_QUG_AMOUNT.replace_all(&out, "[AMOUNT] QUG").to_string();
    // Replace very large integers (likely base-unit amounts)
    out = RE_LARGE_NUMBER.replace_all(&out, "[AMOUNT]").to_string();

    out
}

// ---------------------------------------------------------------------------
// RedactedFormatter — wraps tracing_subscriber's FormatEvent
// ---------------------------------------------------------------------------

/// A `FormatEvent` wrapper that applies safety-net regex redaction after the
/// inner formatter has written the log line.
pub struct RedactedFormatter<F> {
    inner: F,
}

impl<F> RedactedFormatter<F> {
    pub fn new(inner: F) -> Self {
        Self { inner }
    }
}

impl<S, N, F> tracing_subscriber::fmt::FormatEvent<S, N> for RedactedFormatter<F>
where
    S: tracing::Subscriber + for<'a> tracing_subscriber::registry::LookupSpan<'a>,
    N: for<'writer> tracing_subscriber::fmt::FormatFields<'writer> + 'static,
    F: tracing_subscriber::fmt::FormatEvent<S, N>,
{
    fn format_event(
        &self,
        ctx: &tracing_subscriber::fmt::FmtContext<'_, S, N>,
        mut writer: tracing_subscriber::fmt::format::Writer<'_>,
        event: &tracing::Event<'_>,
    ) -> fmt::Result {
        if privacy() == LEVEL_NONE {
            return self.inner.format_event(ctx, writer, event);
        }

        // Format into a temporary buffer, then redact.
        let mut buf = String::with_capacity(256);
        let buf_writer = tracing_subscriber::fmt::format::Writer::new(&mut buf);
        self.inner.format_event(ctx, buf_writer, event)?;
        let redacted = redact_line(&buf);
        writer.write_str(&redacted)
    }
}

// ---------------------------------------------------------------------------
// PrivacyRedactionLayer — convenience for the TUI log path
// ---------------------------------------------------------------------------

/// Standalone redaction utility for non-fmt subscriber paths (e.g. TUI ring buffer).
pub struct PrivacyRedactionLayer;

impl PrivacyRedactionLayer {
    /// Redact a single log message string. Zero-overhead when `Q_LOG_PRIVACY=none`.
    #[inline]
    pub fn redact(msg: &str) -> String {
        redact_line(msg)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn set_level(level: u8) {
        PRIVACY_LEVEL.store(level, Ordering::Relaxed);
    }

    #[test]
    fn test_mask_addr_full() {
        set_level(LEVEL_FULL);
        let addr = "efca1234567890abcdef1234567890abcdef1234567890abcdef1234567890ab0723";
        let masked = mask_addr(addr);
        assert!(masked.starts_with("qnk"));
        assert!(masked.contains('…'));
        assert!(masked.ends_with("0723"));
        assert!(masked.len() < addr.len());
    }

    #[test]
    fn test_mask_addr_partial() {
        set_level(LEVEL_PARTIAL);
        let addr = "efca1234567890abcdef1234567890abcdef1234567890abcdef1234567890ab0723";
        let masked = mask_addr(addr);
        assert!(masked.starts_with("efca1234"));
        assert!(masked.contains('…'));
        assert!(masked.ends_with("b0723"));
    }

    #[test]
    fn test_mask_addr_none() {
        set_level(LEVEL_NONE);
        let addr = "efca1234567890abcdef";
        assert_eq!(mask_addr(addr), addr);
    }

    #[test]
    fn test_mask_amt_full() {
        set_level(LEVEL_FULL);
        // 5000 QUG in 24-decimal base units
        let amount: u128 = 5_000 * 10u128.pow(24);
        let masked = mask_amt(amount);
        assert_eq!(masked, "[1K-10K]");
    }

    #[test]
    fn test_mask_amt_partial() {
        set_level(LEVEL_PARTIAL);
        let amount: u128 = 5_000 * 10u128.pow(24);
        let masked = mask_amt(amount);
        assert!(masked.starts_with('~'));
        assert!(masked.contains('K'));
    }

    #[test]
    fn test_mask_amt_none() {
        set_level(LEVEL_NONE);
        let amount: u128 = 12345;
        assert_eq!(mask_amt(amount), "12345");
    }

    #[test]
    fn test_mask_amt_display_full() {
        set_level(LEVEL_FULL);
        assert_eq!(mask_amt_display(0.005), "[<0.01]");
        assert_eq!(mask_amt_display(0.5), "[0.01-1]");
        assert_eq!(mask_amt_display(5.0), "[1-10]");
        assert_eq!(mask_amt_display(50.0), "[10-100]");
        assert_eq!(mask_amt_display(500.0), "[100-1K]");
        assert_eq!(mask_amt_display(5000.0), "[1K-10K]");
    }

    #[test]
    fn test_mask_hash_full() {
        set_level(LEVEL_FULL);
        let hash = "a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6f3d4";
        let masked = mask_hash(hash);
        assert_eq!(masked, "a1b2…f3d4");
    }

    #[test]
    fn test_mask_hash_partial() {
        set_level(LEVEL_PARTIAL);
        let hash = "a1b2c3d4e5f6a1b2c3d4e5f6";
        assert_eq!(mask_hash(hash), hash);
    }

    #[test]
    fn test_redact_line_catches_hex64() {
        set_level(LEVEL_FULL);
        let line = "Transfer from efca1234567890abcdef1234567890abcdef1234567890abcdef1234567890ab to abc";
        let redacted = redact_line(line);
        assert!(redacted.contains("qnk[REDACTED]"), "redacted = {redacted}");
        assert!(!redacted.contains("efca1234567890abcdef1234567890abcdef1234567890abcdef1234567890ab"));
    }

    #[test]
    fn test_redact_line_catches_hex40() {
        set_level(LEVEL_FULL);
        // exactly 40 hex chars
        let line = "Transfer from efca1234567890abcdef12345678abcdef123456 to abc";
        let redacted = redact_line(line);
        assert!(redacted.contains("qnk[REDACTED]"), "redacted = {redacted}");
    }

    #[test]
    fn test_redact_line_catches_qug() {
        set_level(LEVEL_FULL);
        let line = "Balance: 1234.56 QUG credited";
        let redacted = redact_line(line);
        assert!(redacted.contains("[AMOUNT] QUG"));
        assert!(!redacted.contains("1234.56"));
    }

    #[test]
    fn test_redact_line_noop_when_none() {
        set_level(LEVEL_NONE);
        let line = "Transfer from efca1234567890abcdef1234567890abcdef1234567890abcdef12345678 of 1234.56 QUG";
        assert_eq!(redact_line(line), line);
    }

    #[test]
    fn test_mask_addr_bytes() {
        set_level(LEVEL_FULL);
        let bytes = [0xef, 0xca, 0x12, 0x34, 0x56, 0x78, 0x90, 0xab,
                     0xcd, 0xef, 0x12, 0x34, 0x56, 0x78, 0x90, 0xab,
                     0xcd, 0xef, 0x12, 0x34, 0x56, 0x78, 0x90, 0xab,
                     0xcd, 0xef, 0x12, 0x34, 0x56, 0x78, 0x07, 0x23];
        let masked = mask_addr_bytes(&bytes);
        assert!(masked.starts_with("qnk"));
        assert!(masked.ends_with("0723"));
    }

    #[test]
    fn test_short_addr_handled() {
        set_level(LEVEL_FULL);
        assert_eq!(mask_addr("abcd"), "qnk***");
        set_level(LEVEL_PARTIAL);
        assert_eq!(mask_addr("abcd"), "abcd…");
    }
}
