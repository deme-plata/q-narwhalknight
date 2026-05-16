//! AI TRADE — bridge between the slint wallet and the user's Claude Code
//! installation via the existing `quillon-wallet-mcp` MCP server.
//!
//! v11.5.0 — first slice: emit setup instructions to a file in `~/.quillon/`
//! and fire a desktop notification telling the user where to look. Auto-write
//! to `~/.claude.json` is deferred to a follow-up.
//!
//! The wallet does not execute trades on the user's behalf. The MCP server is
//! suggested with read-only scopes so an AI client can analyse the portfolio
//! and draft trades; actual swaps still require an explicit confirm in the
//! wallet's send/DEX UI.

use std::fs;
use std::io::Write;
use std::path::PathBuf;

use crate::notifications;

const INSTRUCTIONS_FILENAME: &str = "ai-trade-setup.txt";

/// Write setup instructions to `~/.quillon/ai-trade-setup.txt` and fire a
/// notification with the path. Returns the absolute path on success.
pub fn emit_setup_instructions() -> Option<PathBuf> {
    let home = std::env::var("HOME").ok()?;
    let dir: PathBuf = [&home, ".quillon"].iter().collect();
    if let Err(e) = fs::create_dir_all(&dir) {
        eprintln!("[ai-trade] WARN: cannot create {:?}: {}", dir, e);
        return None;
    }
    let path = dir.join(INSTRUCTIONS_FILENAME);

    let body = build_instructions();
    if let Err(e) = write_atomic(&path, body.as_bytes()) {
        eprintln!("[ai-trade] WARN: cannot write {:?}: {}", path, e);
        return None;
    }

    notifications::notify(
        notifications::Category::Info,
        &format!(
            "AI TRADE setup written to {}. Open it for the Claude Code MCP command + starter prompt.",
            path.display()
        ),
    );
    eprintln!(
        "[ai-trade] OK — instructions at {} (also fired a desktop notification)",
        path.display()
    );

    Some(path)
}

fn build_instructions() -> String {
    let mcp_path_guess = guess_mcp_path();
    format!(
        "# Quillon Wallet — AI TRADE setup\n\
         #\n\
         # Connect your Claude Code installation to the Quillon wallet MCP\n\
         # server. The MCP exposes read-only portfolio + trading tools so\n\
         # Claude can analyse your wallet and draft trade ideas. Swaps still\n\
         # require an explicit confirm in the slint wallet's DEX screen.\n\
         \n\
         ## 1. Register the MCP server\n\
         \n\
         If you have the q-narwhalknight repo on this machine:\n\
         \n\
             claude mcp add quillon-wallet -- node {mcp}\n\
         \n\
         If you do not have the repo, install the MCP package globally first:\n\
         \n\
             npm install -g @modelcontextprotocol/sdk\n\
             # then clone q-narwhalknight and build tools/quillon-wallet-mcp\n\
         \n\
         ## 2. Confirm Claude sees it\n\
         \n\
             claude mcp list\n\
         \n\
         You should see `quillon-wallet ... ✓ Connected`.\n\
         \n\
         ## 3. Starter prompts\n\
         \n\
         Open Claude Code and try:\n\
         \n\
             Analyse my Quillon portfolio and suggest 3 trades for the next week.\n\
         \n\
             What's my current QUG balance and recent transaction history?\n\
         \n\
             Quote a swap of 100 QUG to QUGUSD. Don't execute it.\n\
         \n\
         ## Scopes\n\
         \n\
         The MCP currently runs unauthenticated against the public API. A\n\
         follow-up commit will mint a scoped OAuth2 Bearer (read:balance,\n\
         read:history, read:tokens, dex:quote) and inject it as\n\
         Authorization: Bearer ... so the AI cannot execute trades on its own.\n",
        mcp = mcp_path_guess,
    )
}

fn guess_mcp_path() -> String {
    let candidates = [
        "/opt/orobit/shared/q-narwhalknight/tools/quillon-wallet-mcp/build/index.js",
        "/home/orobit/q-narwhalknight/tools/quillon-wallet-mcp/build/index.js",
    ];
    for c in &candidates {
        if std::path::Path::new(c).exists() {
            return c.to_string();
        }
    }
    // Fallback: relative path the user must adjust.
    "/path/to/q-narwhalknight/tools/quillon-wallet-mcp/build/index.js".to_string()
}

fn write_atomic(path: &PathBuf, bytes: &[u8]) -> std::io::Result<()> {
    let tmp = path.with_extension("tmp");
    {
        let mut f = fs::File::create(&tmp)?;
        f.write_all(bytes)?;
        f.sync_all()?;
    }
    fs::rename(&tmp, path)
}
