# Claude Code + Q-NarwhalKnight MCP Setup Guide

Set up Claude Code with the Q-NarwhalKnight MCP server for AI-powered codebase exploration, bug hunting, and contribution submission. Zero to contributing in under 10 minutes.

---

## 1. Prerequisites

Before starting, make sure you have:

- **Node.js 18+** -- check with `node --version`
- **An Anthropic API key** -- get one at [console.anthropic.com](https://console.anthropic.com)
- **A terminal** -- Linux, macOS, or WSL on Windows

If you need Node.js:

```bash
# Ubuntu/Debian
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo bash -
sudo apt-get install -y nodejs

# macOS (Homebrew)
brew install node

# Or use nvm (any platform)
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.1/install.sh | bash
nvm install 20
```

---

## 2. Install Claude Code

Install the CLI globally:

```bash
npm install -g @anthropic-ai/claude-code
```

Set your Anthropic API key:

```bash
export ANTHROPIC_API_KEY=sk-ant-api03-XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
```

To persist it across sessions, add the export line to your shell profile (`~/.bashrc`, `~/.zshrc`, etc.).

Verify the installation:

```bash
claude --version
```

You should see a version string like `1.x.x`. If the command is not found, ensure `$(npm prefix -g)/bin` is in your `PATH`.

---

## 3. Connect to the Q-NarwhalKnight MCP Server

The MCP (Model Context Protocol) server at `code.quillon.xyz` provides read-only access to the entire Q-NarwhalKnight codebase. Claude Code connects to it over SSE and can read files, search code, view diffs, and submit contribution proposals -- all without needing a local clone of the repository.

### Option A: Project-level config (recommended for team use)

Create a `.mcp.json` file in your working directory:

```json
{
  "mcpServers": {
    "quillon-code": {
      "type": "sse",
      "url": "https://code.quillon.xyz/mcp/sse"
    }
  }
}
```

### Option B: User-level config (applies to all projects)

Edit `~/.claude/settings.json`:

```json
{
  "mcpServers": {
    "quillon-code": {
      "type": "sse",
      "url": "https://code.quillon.xyz/mcp/sse"
    }
  }
}
```

Create the directory if it does not exist:

```bash
mkdir -p ~/.claude
```

### Verify the connection

Launch Claude Code in interactive mode:

```bash
claude
```

Then type:

```
List the files in the Q-NarwhalKnight repository
```

If the MCP server is connected, Claude will call the `list_files` tool and display the repository's file tree.

---

## 4. Available Tools

Once connected, Claude Code has access to the following MCP tools:

| Tool | Description |
|------|-------------|
| `read_file` | Read any git-tracked file from the repository. Pass a relative path like `crates/q-types/src/lib.rs`. |
| `search_code` | Search for patterns across the codebase. Supports regex. Filter by file type: `rs`, `tsx`, `ts`, `toml`, `md`, `json`, `css`, `html`. |
| `list_files` | List all git-tracked files (capped at 500 results). |
| `list_branches` | List all local and remote branches with latest commit info. |
| `view_diff` | View the diff between a branch and `main` (or any other base branch). Useful for merge request previews. |
| `submit_contribution` | Submit a code proposal (unified diff format) for maintainer review. Requires a title, description, and diff. |
| `list_contributions` | List all pending contribution proposals and their review status. |

You do not need to call these tools manually. Ask Claude naturally and it will select the appropriate tool.

---

## 5. Example Session

Here is a realistic session showing how to explore the codebase, find a bug, and submit a fix:

```
you> Read the emission controller code

Claude> [calls read_file with path "crates/q-storage/src/emission_controller.rs"]
       Here's the emission controller. It manages per-block mining rewards
       using an adaptive rate based on network block production...

you> Search for where block rewards are calculated

Claude> [calls search_code with query "block_reward" and file_type "rs"]
       Found 14 matches across these files:
         crates/q-storage/src/emission_controller.rs (5 matches)
         crates/q-storage/src/balance_consensus.rs (3 matches)
         crates/q-api-server/src/block_producer.rs (4 matches)
         ...

you> Read the balance_consensus.rs file to see how rewards are applied

Claude> [calls read_file with path "crates/q-storage/src/balance_consensus.rs"]
       The process_block_mining_rewards function credits the miner address
       and tracks the block for emission rate adjustment...

you> I see a potential integer overflow on line 847 where the reward is
     multiplied by the era multiplier without checked arithmetic. Can you
     submit a fix that uses checked_mul with a fallback to the base reward?

Claude> [calls submit_contribution with title, description, and unified diff]
       Contribution submitted!
       Patch ID: a3f7c12e
       Status: pending_review
       Maintainers will review your proposal.

you> List all pending contributions

Claude> [calls list_contributions]
       [pending_review] a3f7c12e: Fix unchecked multiplication in reward
       calculation (by anonymous-claude-code-user, 2026-02-19T14:23:00Z)
```

### More examples

**Explore the project structure:**
```
you> What crates are in this workspace and what does each one do?
```

**Understand a specific subsystem:**
```
you> How does the turbo sync protocol work? Walk me through the code flow.
```

**Review a feature branch:**
```
you> Show me what changed in the feature/safe-batched-sync-v1.0.2 branch compared to main
```

**Search for security patterns:**
```
you> Search for all uses of "unsafe" in the Rust code
```

---

## 6. Security Notes

The MCP server is designed with defense-in-depth. Here is what you need to know:

- **All tools are read-only.** No MCP tool can modify files in the repository, run git commands that alter state, or execute arbitrary code on the server.

- **`read_file` only serves git-committed content.** It uses `git show HEAD:<path>` internally, so files that are untracked, gitignored, or only on disk (`.env`, private keys, `node-data/`, signing keys) are never accessible.

- **Path traversal is blocked.** Paths containing `..`, null bytes, or absolute prefixes are rejected. A whitelist blocks known sensitive patterns (`.env`, `.key`, `.keys`, `encryption`, `node_signing_key`, `libp2p_identity`, `.git/`).

- **Contributions are stored outside the repository.** The `submit_contribution` tool writes proposal files to an isolated directory (`/opt/orobit/contributions/`), completely separate from the git tree. Proposals are plain text (JSON metadata + `.patch` files) and are never auto-applied.

- **Rate limited.** 60 requests per minute per IP address. Exceeding the limit returns HTTP 429.

- **Sessions auto-expire.** SSE sessions are closed after 1 hour of inactivity. Maximum 20 concurrent sessions.

- **No authentication required.** The server is intentionally public and read-only. There is nothing to steal -- all content is from the public git history.

---

## 7. Troubleshooting

### "Session not found"

SSE sessions expire after 1 hour. Claude Code will automatically reconnect on the next request. If you see this repeatedly, restart Claude Code:

```bash
# Exit and relaunch
claude
```

### "Rate limited" / HTTP 429

You have exceeded 60 requests per minute. Wait 60 seconds and try again. If you are running automated scripts, add delays between requests.

### "File not found in git"

The file must be committed to git on the `HEAD` branch. Files that exist only on disk (untracked, staged but not committed) are not served. Check if the file is tracked:

```
you> Search for "filename" to see if it exists in the repo
```

### Connection refused or timeout

1. Verify the MCP server is running:

```bash
curl -s https://code.quillon.xyz/health
```

Expected response:

```json
{"status":"ok","repo_path":"/opt/orobit/shared/q-narwhalknight","mcp":true}
```

2. Check your network can reach `code.quillon.xyz` on port 443 (HTTPS).

3. If behind a corporate proxy, ensure the proxy allows SSE (long-lived HTTP connections with `text/event-stream` content type).

### Claude Code does not use MCP tools

Make sure your config file is in the right location and has correct JSON syntax:

```bash
# Check project-level config
cat .mcp.json

# Check user-level config
cat ~/.claude/settings.json
```

The `url` must be exactly `https://code.quillon.xyz/mcp/sse` (with the `/mcp/sse` path).

### "Unknown tool" errors

This means Claude tried to call a tool that does not exist on the server. The available tools are: `read_file`, `search_code`, `list_files`, `list_branches`, `view_diff`, `submit_contribution`, `list_contributions`. If you see this, it is likely a transient issue -- retry the request.

---

## 8. For Bounty Hunters

Q-NarwhalKnight runs a bug bounty program. Here is how to use Claude Code to find and report bugs:

### Step 1: Explore the codebase

```
you> Search for "unwrap()" in Rust files -- find places where panics could occur
you> Read the block validation logic in q-types/src/block.rs
you> Search for "unsafe" blocks and review each one
```

### Step 2: Identify a bug

Look for:
- Unchecked arithmetic (overflow/underflow in balance or reward calculations)
- Missing input validation in API handlers
- Race conditions in concurrent code
- Panic paths (`unwrap()`, `expect()`) in consensus-critical code
- Logic errors in sync, fork detection, or balance propagation

### Step 3: Submit your fix

```
you> I found a bug in emission_controller.rs line 203. The era_multiplier
     calculation can overflow for era > 50. Submit a fix using checked_mul
     with a fallback to era 50's multiplier.
```

Claude will generate a unified diff and submit it via `submit_contribution`. You will receive a patch ID.

### Step 4: File a bug report

Go to [quillon.xyz](https://quillon.xyz) and file a bug report with:
- The **patch ID** from your submission
- A description of the vulnerability and its severity
- Steps to reproduce (if applicable)

### Step 5: Earn bounty points

Accepted fixes earn bounty points based on severity:
- **Critical** (consensus, funds at risk): highest reward
- **High** (data corruption, sync issues): high reward
- **Medium** (API bugs, display errors): moderate reward
- **Low** (typos, minor UX issues): base reward

Check your bounty status at the Leaderboard page on quillon.xyz.

---

## Quick Reference

```bash
# Install
npm install -g @anthropic-ai/claude-code
export ANTHROPIC_API_KEY=sk-ant-api03-...

# Configure MCP (create .mcp.json in your project directory)
echo '{
  "mcpServers": {
    "quillon-code": {
      "type": "sse",
      "url": "https://code.quillon.xyz/mcp/sse"
    }
  }
}' > .mcp.json

# Launch
claude

# Health check
curl -s https://code.quillon.xyz/health
```
