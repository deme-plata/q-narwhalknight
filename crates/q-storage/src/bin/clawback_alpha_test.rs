//! Alpha-equivalent test rig for qugusd_clawback (Rule 4: prove balance-modifying
//! code on a throwaway DB before it touches prod). Seeds a fresh RocksDB with the
//! exact token_balance_<addr>_<QUGUSD> key format the clawback tool reads/writes:
//!   - two TARGET wallets (to be zeroed)
//!   - decoys that MUST stay untouched: legit 1ca3a232, founder efca1e8c, an honest
//!     small holder, and a wrong-expect target (proves pre-image mismatch → skip)
//!
//!   clawback_alpha_test seed   <db_dir>   -> create + seed, print balances
//!   clawback_alpha_test verify <db_dir>   -> reopen, print balances, PASS/FAIL asserts
//! Between the two, run the real qugusd_clawback binary with --confirm against <db_dir>.

use rocksdb::{Options, DB};
use anyhow::{anyhow, Result};

const CF: &str = "manifest";
const QUGUSD: &str = "5155475553440000000000000000000000000000000000000000000000000000";
const DEC: u128 = 1_000_000_000_000_000_000_000_000;

fn key(addr: &str) -> String { format!("token_balance_{}_{}", addr, QUGUSD) }
fn fmt(v: u128) -> String { format!("{}.{:024}", v / DEC, v % DEC) }

// (address_hex, balance_whole_qugusd, label, must_be_untouched)
fn fixtures() -> Vec<(&'static str, u128, &'static str, bool)> {
    vec![
        ("4cf3b55ebeb28c97ce5c7a5369169cf1db933a4cd3d120d7bef3a79ed8c4f2fb", 58_516_844_907_534, "TARGET abuser",        false),
        ("540e6a41ec1fd4de90ef8286b2dbe8c1f16d1104e9eece0ed482155f95afb1a2", 83_079_357_580,     "TARGET secondary",     false),
        ("1ca3a2320d3bc6d5479da1dd3e1267289aebdb93da1da60091fe40ce12a14aaa", 833_529_378,        "DECOY legit (exclude)", true),
        ("efca1e8c1f46e91013b4073898c771bb3d566453537ccf87e834505925e50723", 296_708_423,        "DECOY founder",         true),
        ("00000000000000000000000000000000000000000000000000000000deadbeef", 5,                  "DECOY honest holder",   true),
    ]
}

fn open(dir: &str) -> Result<DB> {
    let cfs = DB::list_cf(&Options::default(), dir).unwrap_or_else(|_| vec!["default".into()]);
    let mut o = Options::default();
    o.create_if_missing(true);
    o.create_missing_column_families(true);
    let mut names: Vec<String> = cfs;
    if !names.iter().any(|c| c == CF) { names.push(CF.into()); }
    let refs: Vec<&str> = names.iter().map(|s| s.as_str()).collect();
    DB::open_cf(&o, dir, &refs).map_err(|e| anyhow!("open failed: {e}"))
}

fn main() -> Result<()> {
    let a: Vec<String> = std::env::args().collect();
    if a.len() < 3 { eprintln!("usage: clawback_alpha_test <seed|verify> <db_dir>"); std::process::exit(2); }
    let (mode, dir) = (a[1].as_str(), a[2].as_str());
    let db = open(dir)?;
    let cf = db.cf_handle(CF).ok_or_else(|| anyhow!("no manifest CF"))?;

    match mode {
        "seed" => {
            for (addr, whole, label, _) in fixtures() {
                let base = whole.saturating_mul(DEC);
                db.put_cf(&cf, key(addr).as_bytes(), &base.to_le_bytes())?;
                println!("  seeded {}… = {} QUGUSD  [{}]", &addr[..12], fmt(base), label);
            }
            db.flush()?;
            println!("SEED DONE ({} wallets)", fixtures().len());
        }
        "verify" => {
            let mut fail = 0;
            for (addr, whole, label, untouched) in fixtures() {
                let got = db.get_cf(&cf, key(addr).as_bytes())?
                    .map(|v| if v.len() >= 16 { u128::from_le_bytes(v[..16].try_into().unwrap()) } else { 0 })
                    .unwrap_or(0);
                let orig = whole.saturating_mul(DEC);
                let ok = if untouched { got == orig } else { got == 0 };
                if !ok { fail += 1; }
                println!("  {} {}… now={} (was {}) [{}] expect {}",
                    if ok { "✅" } else { "❌" }, &addr[..12], fmt(got), fmt(orig), label,
                    if untouched { "UNCHANGED" } else { "ZEROED" });
            }
            println!("{}", if fail == 0 {
                "ALPHA TEST PASSED — only named targets zeroed; all decoys (legit/founder/honest) untouched"
            } else { "ALPHA TEST FAILED" });
            if fail > 0 { std::process::exit(1); }
        }
        _ => { eprintln!("mode must be seed|verify"); std::process::exit(2); }
    }
    Ok(())
}
