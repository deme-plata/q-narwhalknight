//! Independent, DB-free verification of a /api/v1/proof/balance/:address
//! response. This is what a real light client would run — no node access,
//! no trust in the server beyond trusting the root itself.
//!
//! Usage: cat response.json | verify_proof_json

use anyhow::{Context, Result};
use q_storage::balance_smt::SmtProof;
use std::io::Read;

fn hex32(s: &str) -> Result<[u8; 32]> {
    let v = hex::decode(s)?;
    v.try_into().map_err(|_| anyhow::anyhow!("not 32 bytes"))
}

fn main() -> Result<()> {
    let mut input = String::new();
    std::io::stdin().read_to_string(&mut input)?;
    let v: serde_json::Value = serde_json::from_str(&input)?;
    let d = &v["data"];

    let addr = hex32(d["address"].as_str().context("address")?)?;
    let balance: u128 = d["balance"].as_str().context("balance")?.parse()?;
    let root = hex32(d["root_v2_smt"].as_str().context("root")?)?;
    let siblings_json = d["siblings"].as_array().context("siblings")?;
    let empty_bitmap = hex32(d["empty_bitmap"].as_str().context("empty_bitmap")?)?;

    let mut siblings = [[0u8; 32]; 256];
    for (i, s) in siblings_json.iter().enumerate() {
        siblings[i] = hex32(s.as_str().context("sibling")?)?;
    }

    let proof = SmtProof {
        addr,
        balance,
        siblings,
        empty_bitmap,
    };

    println!("Independently verifying:");
    println!("  address: {}", hex::encode(addr));
    println!("  balance: {}", balance);
    println!("  against root: {}", hex::encode(root));

    if proof.verify(&root) {
        println!("✅ PROOF VALID — this balance is genuinely committed to by this root.");
    } else {
        println!("❌ PROOF INVALID");
        std::process::exit(1);
    }

    // Negative control: tamper with the balance, must fail.
    let tampered = SmtProof {
        balance: balance + 1,
        ..proof
    };
    if tampered.verify(&root) {
        println!("🚨 TAMPER NOT CAUGHT — claiming balance+1 still verified!");
        std::process::exit(1);
    } else {
        println!("✅ Tamper check: claiming a different balance correctly FAILS to verify.");
    }

    Ok(())
}
