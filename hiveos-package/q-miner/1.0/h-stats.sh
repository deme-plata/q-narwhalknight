#!/usr/bin/env bash

# Q-NarwhalKnight Miner Stats Script for HiveOS

# Get miner stats from log file
stats_raw=$(tail -100 $MINER_LOG_BASENAME.log 2>/dev/null | grep -oP '(?<=Hashrate: )[0-9.]+(?= KH/s)')

# Calculate total hashrate
khs=0
for rate in $stats_raw; do
    khs=$(echo "$khs + $rate" | bc 2>/dev/null)
done

# Get accepted shares
accepted=$(tail -100 $MINER_LOG_BASENAME.log 2>/dev/null | grep -c "✅ Solution accepted")

# Get rejected shares
rejected=$(tail -100 $MINER_LOG_BASENAME.log 2>/dev/null | grep -c "❌ Solution rejected")

# Get uptime
uptime=$(ps -p $(pgrep -f q-miner | head -1) -o etime= 2>/dev/null | tr -d ' ')
[[ -z $uptime ]] && uptime="0"

# Output stats in HiveOS format (JSON)
stats=$(jq -nc \
    --arg hs "$khs" \
    --arg ac "$accepted" \
    --arg rj "$rejected" \
    --arg uptime "$uptime" \
    '{
        hs: [$hs],
        hs_units: "khs",
        temp: [],
        fan: [],
        uptime: $uptime,
        ar: [$ac, $rj],
        algo: "dag-knight"
    }')

echo "$stats"
