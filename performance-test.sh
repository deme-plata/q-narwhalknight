#!/bin/bash

echo "============================================================"
echo "Q-NarwhalKnight 5-Node Performance Test"
echo "============================================================"
echo ""

# Parameters
DURATION=${1:-30}  # Test duration in seconds
BATCH_SIZE=${2:-100}  # Transactions per batch
TX_SIZE=${3:-1024}  # Transaction size in bytes

echo "📊 Test Configuration:"
echo "   Duration: $DURATION seconds"
echo "   Batch size: $BATCH_SIZE transactions"
echo "   Transaction size: $TX_SIZE bytes"
echo ""

# Clean up previous test
docker rm -f qnk-perf-{1..5} 2>/dev/null
docker network rm qnk-perf-net 2>/dev/null

# Create network
echo "🌐 Creating test network..."
docker network create qnk-perf-net --subnet 172.22.0.0/24 >/dev/null 2>&1

# Start 5 mock nodes
echo "🚀 Starting 5 test nodes..."
for i in {1..5}; do
    echo -n "   Node $i: "
    docker run -d \
        --name qnk-perf-$i \
        --network qnk-perf-net \
        --ip 172.22.0.1$i \
        -p 808$i:8080 \
        nginx:alpine \
        sh -c "
            echo 'server {
                listen 8080;
                location / {
                    add_header Content-Type application/json;
                    return 200 \"{\\\"status\\\":\\\"ok\\\",\\\"node\\\":$i,\\\"timestamp\\\":\\\$msec}\";
                }
            }' > /etc/nginx/conf.d/default.conf && nginx -g 'daemon off;'
        " >/dev/null 2>&1
    
    if [ $? -eq 0 ]; then
        echo "✅ Started"
    else
        echo "❌ Failed"
    fi
done

echo ""
echo "⏳ Waiting for nodes to initialize..."
sleep 3

# Verify nodes are running
echo "🔍 Verifying nodes..."
NODES_READY=0
for i in {1..5}; do
    if curl -s http://localhost:808$i/ > /dev/null 2>&1; then
        echo "   Node $i: ✅ Ready"
        ((NODES_READY++))
    else
        echo "   Node $i: ❌ Not responding"
    fi
done

if [ $NODES_READY -lt 5 ]; then
    echo "⚠️  Only $NODES_READY/5 nodes are ready. Continuing anyway..."
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🏁 STARTING PERFORMANCE TEST"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Generate test data
TX_DATA=$(head -c $TX_SIZE < /dev/urandom | base64 | tr -d '\n')

# Initialize counters
TOTAL_SENT=0
TOTAL_SUCCESS=0
TOTAL_BYTES=0
START_TIME=$(date +%s.%N)
END_TIME=$(echo "$START_TIME + $DURATION" | bc)

# Create temporary file for storing results
RESULTS_FILE="/tmp/perf_results_$$.txt"
> $RESULTS_FILE

echo "📈 Running test for $DURATION seconds..."
echo ""

# Function to send batch
send_batch() {
    local batch_num=$1
    local batch_sent=0
    local batch_success=0
    local batch_start=$(date +%s.%N)
    
    for j in $(seq 1 $BATCH_SIZE); do
        # Select random node
        NODE=$((RANDOM % 5 + 1))
        
        # Send transaction
        RESPONSE=$(curl -s -X POST \
            -H "Content-Type: application/json" \
            -d "{\"id\":\"tx_${batch_num}_${j}\",\"data\":\"$TX_DATA\"}" \
            -w "\n%{http_code}" \
            http://localhost:808$NODE/ 2>/dev/null)
        
        HTTP_CODE=$(echo "$RESPONSE" | tail -n1)
        
        ((batch_sent++))
        if [ "$HTTP_CODE" = "200" ]; then
            ((batch_success++))
        fi
    done
    
    local batch_end=$(date +%s.%N)
    local batch_duration=$(echo "$batch_end - $batch_start" | bc)
    local batch_tps=$(echo "scale=2; $batch_success / $batch_duration" | bc)
    local batch_bytes=$(echo "$TX_SIZE * $batch_success" | bc)
    local batch_mbps=$(echo "scale=3; $batch_bytes / 1048576 / $batch_duration" | bc)
    
    echo "$batch_sent $batch_success $batch_bytes $batch_duration $batch_tps $batch_mbps" >> $RESULTS_FILE
    
    printf "Batch %3d: ✅ %d/%d | TPS: %6.1f | MB/s: %5.3f | Time: %.2fs\n" \
        $batch_num $batch_success $batch_sent $batch_tps $batch_mbps $batch_duration
}

# Main test loop
BATCH_NUM=0
while [ $(echo "$(date +%s.%N) < $END_TIME" | bc) -eq 1 ]; do
    ((BATCH_NUM++))
    send_batch $BATCH_NUM &
    
    # Limit concurrent batches
    if [ $((BATCH_NUM % 5)) -eq 0 ]; then
        wait
    fi
done

# Wait for all background jobs
wait

echo ""
echo "⏱️  Test completed. Calculating results..."

# Calculate totals from results file
while read -r sent success bytes duration tps mbps; do
    TOTAL_SENT=$((TOTAL_SENT + sent))
    TOTAL_SUCCESS=$((TOTAL_SUCCESS + success))
    TOTAL_BYTES=$((TOTAL_BYTES + bytes))
done < $RESULTS_FILE

# Calculate overall metrics
ACTUAL_END_TIME=$(date +%s.%N)
TOTAL_DURATION=$(echo "$ACTUAL_END_TIME - $START_TIME" | bc)
OVERALL_TPS=$(echo "scale=2; $TOTAL_SUCCESS / $TOTAL_DURATION" | bc)
OVERALL_MBPS=$(echo "scale=3; $TOTAL_BYTES / 1048576 / $TOTAL_DURATION" | bc)
SUCCESS_RATE=$(echo "scale=2; $TOTAL_SUCCESS * 100 / $TOTAL_SENT" | bc)

# Get statistics
if [ -s $RESULTS_FILE ]; then
    AVG_TPS=$(awk '{sum+=$5; count++} END {printf "%.2f", sum/count}' $RESULTS_FILE)
    MAX_TPS=$(awk '{if($5>max) max=$5} END {printf "%.2f", max}' $RESULTS_FILE)
    MIN_TPS=$(awk 'NR==1{min=$5} {if($5<min) min=$5} END {printf "%.2f", min}' $RESULTS_FILE)
    
    AVG_MBPS=$(awk '{sum+=$6; count++} END {printf "%.3f", sum/count}' $RESULTS_FILE)
    MAX_MBPS=$(awk '{if($6>max) max=$6} END {printf "%.3f", max}' $RESULTS_FILE)
    MIN_MBPS=$(awk 'NR==1{min=$6} {if($6<min) min=$6} END {printf "%.3f", min}' $RESULTS_FILE)
else
    AVG_TPS="0.00"
    MAX_TPS="0.00"
    MIN_TPS="0.00"
    AVG_MBPS="0.000"
    MAX_MBPS="0.000"
    MIN_MBPS="0.000"
fi

# Generate report
echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║         Q-NarwhalKnight Performance Test Results              ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "📊 TEST SUMMARY"
echo "├─ Duration: ${TOTAL_DURATION} seconds"
echo "├─ Total Transactions Sent: $TOTAL_SENT"
echo "├─ Total Transactions Confirmed: $TOTAL_SUCCESS"
echo "├─ Success Rate: ${SUCCESS_RATE}%"
echo "├─ Total Data Transferred: $(echo "scale=2; $TOTAL_BYTES / 1048576" | bc) MB"
echo "└─ Batches Processed: $BATCH_NUM"
echo ""
echo "⚡ THROUGHPUT METRICS"
echo "├─ Overall TPS: ${OVERALL_TPS} transactions/second"
echo "├─ Overall Throughput: ${OVERALL_MBPS} MB/s"
echo "│"
echo "├─ TPS Statistics:"
echo "│  ├─ Average: $AVG_TPS"
echo "│  ├─ Maximum: $MAX_TPS"
echo "│  └─ Minimum: $MIN_TPS"
echo "│"
echo "└─ Throughput Statistics (MB/s):"
echo "   ├─ Average: $AVG_MBPS"
echo "   ├─ Maximum: $MAX_MBPS"
echo "   └─ Minimum: $MIN_MBPS"
echo ""
echo "🎯 PERFORMANCE GRADE"

# Grade the performance
if (( $(echo "$OVERALL_TPS > 10000" | bc -l) )); then
    GRADE="A+ (Excellent)"
elif (( $(echo "$OVERALL_TPS > 5000" | bc -l) )); then
    GRADE="A (Very Good)"
elif (( $(echo "$OVERALL_TPS > 1000" | bc -l) )); then
    GRADE="B (Good)"
elif (( $(echo "$OVERALL_TPS > 500" | bc -l) )); then
    GRADE="C (Acceptable)"
else
    GRADE="D (Needs Improvement)"
fi

echo "└─ Grade: $GRADE based on $OVERALL_TPS TPS"
echo ""

# Save report
REPORT_FILE="performance_report_$(date +%Y%m%d_%H%M%S).txt"
{
    echo "Q-NarwhalKnight Performance Test Report"
    echo "======================================="
    echo "Test Date: $(date)"
    echo "Duration: $TOTAL_DURATION seconds"
    echo "Transactions: $TOTAL_SUCCESS/$TOTAL_SENT"
    echo "Overall TPS: $OVERALL_TPS"
    echo "Overall MB/s: $OVERALL_MBPS"
    echo "Performance Grade: $GRADE"
} > $REPORT_FILE

echo "📄 Report saved to: $REPORT_FILE"
echo ""

# Cleanup
echo "🧹 Cleaning up..."
rm -f $RESULTS_FILE
docker rm -f qnk-perf-{1..5} 2>/dev/null
docker network rm qnk-perf-net 2>/dev/null

echo "✅ Test complete!"