#!/bin/bash
# Test the halving schedule implementation

echo "========================================"
echo "QNK HALVING SCHEDULE VERIFICATION"
echo "Based on Austrian Economics Time Preference"
echo "========================================"
echo ""
echo "Testing halving schedule at key block heights:"
echo ""

# Function to calculate block reward (must match Rust implementation)
calculate_reward() {
    local height=$1
    local HALVING_INTERVAL=4200000
    local BASE_REWARD=50000000  # 0.5 QNK in base units

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
    1000
    100000
    4199999
    4200000
    4200001
    8399999
    8400000
    12599999
    12600000
    16799999
    16800000
)

printf "%-12s | %-20s | %-15s | %s\n" "Block Height" "Reward (base units)" "Reward (QNK)" "Notes"
echo "-------------|----------------------|-----------------|------------------"

for height in "${test_heights[@]}"; do
    reward=$(calculate_reward $height)
    reward_qnk=$(echo "scale=8; $reward / 100000000" | bc)

    notes=""
    if [ $height -lt 4200000 ]; then
        notes="Era 1: Full reward"
    elif [ $height -ge 4200000 ] && [ $height -lt 8400000 ]; then
        notes="Era 2: First halving"
    elif [ $height -ge 8400000 ] && [ $height -lt 12600000 ]; then
        notes="Era 3: Second halving"
    elif [ $height -ge 12600000 ] && [ $height -lt 16800000 ]; then
        notes="Era 4: Third halving"
    fi

    printf "%-12s | %-20s | %-15s | %s\n" "$height" "$reward" "$reward_qnk" "$notes"
done

echo ""
echo "========================================"
echo "ANNUAL EMISSION RATES"
echo "========================================"
echo ""
echo "Assuming ~10 second block time:"
echo "- Blocks per year: ~4,200,000"
echo ""
printf "%-15s | %-15s | %s\n" "Block Range" "Reward/Block" "Annual Emission"
echo "----------------|-----------------|------------------"
printf "%-15s | %-15s | %s\n" "0 - 4.2M" "0.5 QNK" "~2,100,000 QNK"
printf "%-15s | %-15s | %s\n" "4.2M - 8.4M" "0.25 QNK" "~1,050,000 QNK"
printf "%-15s | %-15s | %s\n" "8.4M - 12.6M" "0.125 QNK" "~525,000 QNK"
printf "%-15s | %-15s | %s\n" "12.6M - 16.8M" "0.0625 QNK" "~262,500 QNK"
echo ""
echo "This halving schedule reflects Austrian economics"
echo "time preference: present goods valued more than future goods"
echo ""
