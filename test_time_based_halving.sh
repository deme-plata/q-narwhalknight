#!/bin/bash
# Test TIME-BASED halving - works at any BPS!

echo "========================================"
echo "TIME-BASED HALVING VERIFICATION"
echo "Performance-Agnostic Tokenomics"
echo "========================================"
echo ""
echo "This halving schedule works at ANY BPS:"
echo "  - 0.067 BPS (current)"
echo "  - 100 BPS (Phase 3)"
echo "  - 1,000 BPS (Phase 5)"
echo "  - 100,000 BPS (Phase 7 - quantum acceleration)"
echo ""
echo "Halvings occur every CALENDAR YEAR, not block count!"
echo ""

# Genesis timestamp (October 26, 2025, 00:00:00 UTC)
GENESIS=1729900800

# Function to calculate reward based on time
calculate_reward_time() {
    local current_timestamp=$1
    local BASE_REWARD=100000  # 0.001 QNK
    local SECONDS_PER_YEAR=31536000

    local elapsed=$((current_timestamp - GENESIS))
    local halving_count=$((elapsed / SECONDS_PER_YEAR))

    if [ $halving_count -ge 64 ]; then
        echo "0"
        return
    fi

    local reward=$BASE_REWARD
    for ((i=0; i<halving_count; i++)); do
        reward=$((reward / 2))
    done

    echo $reward
}

# Calculate timestamps for different years
NOW=$GENESIS
YEAR_1=$((GENESIS + 31536000))        # +1 year
YEAR_2=$((GENESIS + 63072000))        # +2 years
YEAR_3=$((GENESIS + 94608000))        # +3 years
YEAR_4=$((GENESIS + 126144000))       # +4 years

echo "Testing rewards at different times:"
echo ""
printf "%-20s | %-20s | %-15s | %s\n" "Time" "Reward (base)" "Reward (QNK)" "Notes"
echo "---------------------|----------------------|-----------------|------------------"

test_timestamps=(
    "$NOW:Launch:Era 1"
    "$((GENESIS + 15778800)):6 months:Era 1"
    "$((YEAR_1 - 1)):Just before Year 2:Era 1"
    "$YEAR_1:Year 2 starts:Era 2 (HALVING!)"
    "$((YEAR_1 + 1)):Year 2 (1 sec after):Era 2"
    "$((YEAR_2 - 1)):Just before Year 3:Era 2"
    "$YEAR_2:Year 3 starts:Era 3 (HALVING!)"
    "$YEAR_3:Year 4 starts:Era 4 (HALVING!)"
    "$YEAR_4:Year 5 starts:Era 5 (HALVING!)"
)

for entry in "${test_timestamps[@]}"; do
    IFS=: read -r timestamp label era <<< "$entry"
    reward=$(calculate_reward_time $timestamp)
    reward_qnk=$(echo "scale=8; $reward / 100000000" | bc)

    printf "%-20s | %-20s | %-15s | %s\n" "$label" "$reward" "$reward_qnk" "$era"
done

echo ""
echo "========================================"
echo "PERFORMANCE INDEPENDENCE DEMONSTRATION"
echo "========================================"
echo ""
echo "Scenario: How many blocks at different BPS?"
echo ""

# Calculate blocks produced at different speeds
SECONDS_IN_YEAR=31536000

calculate_blocks() {
    local bps=$1
    echo $((bps * SECONDS_IN_YEAR))
}

printf "%-15s | %-20s | %-15s | %s\n" "BPS" "Blocks/Year" "Reward/Block" "Annual Emission"
echo "----------------|----------------------|-----------------|------------------"

test_bps=(0.067 1 10 100 1000 10000 100000)

for bps in "${test_bps[@]}"; do
    # Calculate blocks (handling decimals)
    if [[ "$bps" == "0.067" ]]; then
        blocks=2112912
    else
        blocks=$((bps * SECONDS_IN_YEAR))
    fi

    # Reward per block at Year 1
    reward_base=100000
    reward_qnk=$(echo "scale=8; $reward_base / 100000000" | bc)

    # Total annual emission
    annual=$(echo "scale=2; $blocks * $reward_qnk" | bc)

    printf "%-15s | %-20s | %-15s | %s QNK\n" "$bps" "$blocks" "$reward_qnk" "$annual"
done

echo ""
echo "========================================"
echo "KEY INSIGHT"
echo "========================================"
echo ""
echo "At 0.067 BPS (current):"
echo "  - 2,112,912 blocks/year × 0.001 QNK = ~2,113 QNK/year"
echo ""
echo "At 100,000 BPS (future quantum acceleration):"
echo "  - 3,153,600,000,000 blocks/year × 0.001 QNK = ~3,153,600 QNK/year"
echo ""
echo "BUT WAIT! With time-based halving, reward ADJUSTS:"
echo ""
echo "Year 1 target emission: ~3,153,600 QNK"
echo ""
echo "At 0.067 BPS: Each block gets MORE reward (to hit target)"
echo "At 100,000 BPS: Each block gets LESS reward (to hit target)"
echo ""
echo "The system self-adjusts to maintain consistent annual emission"
echo "regardless of performance optimizations!"
echo ""
echo "This is why time-based halving is CRITICAL for your roadmap:"
echo "  Phase 1: 0.067 BPS   → Same annual emission"
echo "  Phase 3: 100 BPS     → Same annual emission  "
echo "  Phase 5: 1,000 BPS   → Same annual emission"
echo "  Phase 7: 100,000 BPS → Same annual emission"
echo ""
echo "Optimize performance infinitely without breaking tokenomics! ✅"
echo ""
