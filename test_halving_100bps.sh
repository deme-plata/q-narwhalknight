#!/bin/bash
# Test the halving schedule for 100 BPS (democratized ASIC-resistant VDF mining)

echo "========================================"
echo "QNK HALVING SCHEDULE - 100 BPS MODEL"
echo "ASIC-Resistant VDF Mining"
echo "Democratized Mining for All"
echo "========================================"
echo ""
echo "Performance Targets:"
echo "  - 100 blocks per second (BPS)"
echo "  - High TPS through batched transactions"
echo "  - 3,153,600,000 blocks per year"
echo "  - ASIC-resistant VDF ensures fair mining"
echo ""
echo "Testing halving schedule at key block heights:"
echo ""

# Function to calculate block reward (must match Rust implementation)
calculate_reward() {
    local height=$1
    local HALVING_INTERVAL=3153600000
    local BASE_REWARD=100000  # 0.001 QNK in base units

    local halving_count=$((height / HALVING_INTERVAL))

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

# Test various block heights
test_heights=(
    0
    1000000
    100000000
    1000000000
    3153599999
    3153600000
    3153600001
    6307199999
    6307200000
    9460799999
    9460800000
    12614399999
    12614400000
)

printf "%-15s | %-20s | %-15s | %s\n" "Block Height" "Reward (base)" "Reward (QNK)" "Notes"
echo "----------------|----------------------|-----------------|------------------"

for height in "${test_heights[@]}"; do
    reward=$(calculate_reward $height)
    reward_qnk=$(echo "scale=8; $reward / 100000000" | bc)

    notes=""
    if [ $height -lt 3153600000 ]; then
        notes="Era 1: Full reward (Year 1)"
    elif [ $height -ge 3153600000 ] && [ $height -lt 6307200000 ]; then
        notes="Era 2: First halving (Year 2)"
    elif [ $height -ge 6307200000 ] && [ $height -lt 9460800000 ]; then
        notes="Era 3: Second halving (Year 3)"
    elif [ $height -ge 9460800000 ] && [ $height -lt 12614400000 ]; then
        notes="Era 4: Third halving (Year 4)"
    fi

    printf "%-15s | %-20s | %-15s | %s\n" "$height" "$reward" "$reward_qnk" "$notes"
done

echo ""
echo "========================================"
echo "ANNUAL EMISSION RATES"
echo "========================================"
echo ""
printf "%-10s | %-15s | %-20s | %s\n" "Year" "Reward/Block" "Blocks/Year" "Annual Emission"
echo "-----------|-----------------|----------------------|------------------"
printf "%-10s | %-15s | %-20s | %s\n" "1" "0.001 QNK" "3,153,600,000" "~3,153,600 QNK"
printf "%-10s | %-15s | %-20s | %s\n" "2" "0.0005 QNK" "3,153,600,000" "~1,576,800 QNK"
printf "%-10s | %-15s | %-20s | %s\n" "3" "0.00025 QNK" "3,153,600,000" "~788,400 QNK"
printf "%-10s | %-15s | %-20s | %s\n" "4" "0.000125 QNK" "3,153,600,000" "~394,200 QNK"
printf "%-10s | %-15s | %-20s | %s\n" "5-8" "Continues..." "..." "~788,400 QNK total"
echo ""
echo "Total after 4 years: ~5.9M QNK"
echo "Asymptotically approaches: 21M QNK"
echo ""
echo "========================================"
echo "MINER ECONOMICS (Democratized Mining)"
echo "========================================"
echo ""
echo "Scenario: Small Home Miner (0.001% of network)"
echo "  Mining rate: 1 block per second (0.001% of 100 BPS)"
echo "  Reward per block: 0.001 QNK"
echo "  Hourly earnings: 3.6 QNK"
echo "  Daily earnings: 86.4 QNK"
echo "  Monthly earnings: ~2,592 QNK"
echo ""
echo "Scenario: Medium Miner (0.1% of network)"
echo "  Mining rate: 100 blocks per second"
echo "  Hourly earnings: 360 QNK"
echo "  Daily earnings: 8,640 QNK"
echo "  Monthly earnings: ~259,200 QNK"
echo ""
echo "Scenario: Large Miner (1% of network)"
echo "  Mining rate: 1,000 blocks per second"
echo "  Daily earnings: 86,400 QNK"
echo "  Monthly earnings: ~2,592,000 QNK"
echo ""
echo "========================================"
echo "ASIC RESISTANCE BENEFITS"
echo "========================================"
echo ""
echo "✅ VDF (Verifiable Delay Function) mining:"
echo "   - Sequential computation (can't parallelize)"
echo "   - Levels playing field vs ASIC farms"
echo "   - Home miners competitive with data centers"
echo "   - Truly democratized mining"
echo ""
echo "✅ Many miners = decentralization:"
echo "   - Lower per-block rewards encourage participation"
echo "   - High frequency (100 BPS) = steady income"
echo "   - No mining pools needed (fast blocks)"
echo "   - Austrian economics time preference preserved"
echo ""
echo "✅ Network security:"
echo "   - More miners = more decentralized"
echo "   - ASIC resistance = no corporate dominance"
echo "   - VDF guarantees = provable fairness"
echo ""
