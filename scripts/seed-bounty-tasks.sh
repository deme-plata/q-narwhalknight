#!/bin/bash
# Seed bounty tasks based on Operation Twelve Leagues Deep + ongoing endeavours
# Usage: ./scripts/seed-bounty-tasks.sh <admin_wallet>

ADMIN_WALLET="${1:-efca1e8c1f46e91013b4073898c771bb3d566453537ccf87e834505925e50723}"
BASE_URL="${2:-https://bounty.quillon.xyz}"

create_task() {
  local title="$1"
  local desc="$2"
  local reward_qug="$3"
  local reward_score="$4"
  local difficulty="$5"
  local category="$6"
  local proof_req="$7"

  curl -sk -X POST "$BASE_URL/v1/admin/task/create" \
    -H "Content-Type: application/json" \
    -H "X-Wallet-Auth: $ADMIN_WALLET" \
    -d "{
      \"title\": \"$title\",
      \"description\": \"$desc\",
      \"reward_qug\": $reward_qug,
      \"reward_score\": $reward_score,
      \"difficulty\": \"$difficulty\",
      \"category\": \"$category\",
      \"proof_requirements\": \"$proof_req\"
    }" | python3 -m json.tool 2>/dev/null || echo "{}"
  echo ""
}

echo "Seeding Operation Twelve Leagues Deep tasks..."

create_task \
  "Run a full node for 30 days" \
  "Operate a Q-NarwhalKnight full node continuously for 30 days. Your node must stay synced within 100 blocks of network tip and maintain uptime > 95%. This directly contributes to network decentralisation." \
  500 250 "medium" "node_operation" \
  "Submit your node's peer ID, public IP, and a screenshot of journalctl logs showing 30+ days of uptime with current block height."

create_task \
  "Verify safe shutdown procedure on your node" \
  "Test the safe shutdown vs kill -9 difference documented in Operation Twelve Leagues Deep. Run your node, use SIGTERM (not kill -9), verify no database corruption, then restart and confirm block height is intact. This tests lesson #1 from the April 5 incident." \
  150 75 "easy" "testing" \
  "Submit before/after block heights, the shutdown command used, and confirmation the node restarted cleanly at the same height."

create_task \
  "Reproduce and document VDF cap safety test" \
  "Based on Bug #14 (VDF cap 10K vs 134K), verify your miner and server agree on VDF iteration count. Check the miner's output shows matching iterations to what the server expects. Report any mismatches." \
  200 100 "medium" "bug_hunting" \
  "Submit miner log output showing VDF iteration count, server acceptance rate screenshot, and q-miner version used."

create_task \
  "Write a technical blog post about the April 5 incident" \
  "Write a detailed technical blog post (min 800 words) explaining the cascading failure documented in Operation Twelve Leagues Deep. Cover at least 3 of the 6 bug layers, the lessons learned, and what it means for blockchain reliability." \
  300 150 "medium" "documentation" \
  "Submit published URL (Medium, personal blog, Substack, etc.) with wallet address in the post to verify authorship."

create_task \
  "Stress test block sync with a fresh node" \
  "Start a fresh node (empty database) and sync to network tip. Record your sync speed (blocks/sec), peak memory usage, and total time. This helps identify sync bottlenecks and reproduce issues like Bug #4 (0-block responses = success)." \
  250 125 "hard" "testing" \
  "Submit sync logs showing blocks/sec at various heights, final height reached, peak RSS memory, and total sync duration. Include the node version."

create_task \
  "Translate Operation Twelve Leagues Deep to another language" \
  "Translate the full Operation Twelve Leagues Deep PDF narrative into Chinese, Spanish, Russian, Arabic, or another major language. High-quality translation only — machine translation alone does not qualify." \
  400 200 "medium" "community" \
  "Submit a link to the published translation (GitHub, docs site, etc.) and the language translated to."

create_task \
  "Find and report a new sync or storage bug" \
  "Find a reproducible bug in the Q-NarwhalKnight sync or storage subsystems. Must be a NEW issue not already in the known bug list. Provide full reproduction steps, logs, and a theory on root cause." \
  1000 500 "expert" "bug_hunting" \
  "Submit via the Bug Report form with CRITICAL or HIGH severity. Include reproduction steps, relevant logs, and analysis. Reference this task ID in your report."

create_task \
  "Set up a Tor-accessible node" \
  "Configure and run a Q-NarwhalKnight node accessible via Tor onion service using the --tor flag. Document the setup process for other operators." \
  350 175 "hard" "node_operation" \
  "Submit your .onion address, a connectivity test showing the node is reachable, and the setup guide you wrote."

create_task \
  "GPU mining test report" \
  "Download and run the new q-miner-v10.3.14 with GPU/OpenCL support. Report your GPU model, hashrate achieved, GPU temperature, power usage, and comparison with CPU-only mode." \
  200 100 "easy" "testing" \
  "Submit a screenshot of q-miner output showing GPU mining stats, your GPU model, and the platform (Debian 12, Ubuntu, etc.)."

create_task \
  "Create a mining monitoring dashboard" \
  "Build a monitoring dashboard (Grafana, custom web app, or similar) that tracks Q-NarwhalKnight mining metrics: hashrate, solution acceptance rate, block height, and rewards. Share it publicly." \
  600 300 "expert" "development" \
  "Submit link to the live dashboard and source code repository. Must show real-time data from at least one mining node."

echo "Done seeding tasks."
echo "View at: $BASE_URL/v1/tasks"
