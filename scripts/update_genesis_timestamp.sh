#!/bin/bash
# Update Genesis Timestamp for Mainnet Launch
# Usage: ./scripts/update_genesis_timestamp.sh [timestamp]

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

HANDLERS_FILE="crates/q-api-server/src/handlers.rs"
GENESIS_LINE=190

# Predefined timestamps
TESTNET_TIMESTAMP=1761436800  # Oct 26, 2025
MAINNET_DEC1_TIMESTAMP=1764547200   # Dec 1, 2025
MAINNET_DEC15_TIMESTAMP=1765756800  # Dec 15, 2025 (recommended)
MAINNET_DEC20_TIMESTAMP=1766188800  # Dec 20, 2025

echo -e "${BLUE}======================================================================${NC}"
echo -e "${BLUE}          Q-NarwhalKnight Genesis Timestamp Updater${NC}"
echo -e "${BLUE}======================================================================${NC}"
echo ""

# Show current timestamp
echo -e "${YELLOW}Current Genesis Timestamp:${NC}"
grep "GENESIS_TIMESTAMP" "$HANDLERS_FILE" || {
    echo -e "${RED}Error: Could not find GENESIS_TIMESTAMP in $HANDLERS_FILE${NC}"
    exit 1
}
echo ""

# Menu selection
if [ -z "$1" ]; then
    echo -e "${YELLOW}Select target network:${NC}"
    echo "  1) Testnet  (Oct 26, 2025)  - Timestamp: $TESTNET_TIMESTAMP"
    echo "  2) Mainnet  (Dec 1, 2025)   - Timestamp: $MAINNET_DEC1_TIMESTAMP"
    echo "  3) Mainnet  (Dec 15, 2025)  - Timestamp: $MAINNET_DEC15_TIMESTAMP ⭐ RECOMMENDED"
    echo "  4) Mainnet  (Dec 20, 2025)  - Timestamp: $MAINNET_DEC20_TIMESTAMP"
    echo "  5) Custom timestamp"
    echo ""
    read -p "Enter choice (1-5): " choice

    case $choice in
        1)
            NEW_TIMESTAMP=$TESTNET_TIMESTAMP
            NEW_DATE="Oct 26, 2025 00:00:00 UTC"
            ;;
        2)
            NEW_TIMESTAMP=$MAINNET_DEC1_TIMESTAMP
            NEW_DATE="Dec 1, 2025 00:00:00 UTC"
            ;;
        3)
            NEW_TIMESTAMP=$MAINNET_DEC15_TIMESTAMP
            NEW_DATE="Dec 15, 2025 00:00:00 UTC"
            ;;
        4)
            NEW_TIMESTAMP=$MAINNET_DEC20_TIMESTAMP
            NEW_DATE="Dec 20, 2025 00:00:00 UTC"
            ;;
        5)
            read -p "Enter custom Unix timestamp: " NEW_TIMESTAMP
            NEW_DATE=$(date -u -d @"$NEW_TIMESTAMP" 2>/dev/null || echo "Invalid timestamp")
            ;;
        *)
            echo -e "${RED}Invalid choice${NC}"
            exit 1
            ;;
    esac
else
    NEW_TIMESTAMP=$1
    NEW_DATE=$(date -u -d @"$NEW_TIMESTAMP" 2>/dev/null || echo "Invalid timestamp")
fi

echo ""
echo -e "${YELLOW}New Genesis Configuration:${NC}"
echo "  Timestamp: $NEW_TIMESTAMP"
echo "  Date: $NEW_DATE"
echo ""

# Verify timestamp is valid
CURRENT_TIME=$(date +%s)
if [ "$NEW_TIMESTAMP" -lt 1700000000 ]; then
    echo -e "${RED}Error: Timestamp seems too old (before 2023)${NC}"
    exit 1
fi

if [ "$NEW_TIMESTAMP" -gt $((CURRENT_TIME + 31536000)) ]; then
    echo -e "${YELLOW}Warning: Genesis timestamp is more than 1 year in the future${NC}"
fi

# Confirmation
read -p "Continue with update? (y/N): " confirm
if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
    echo -e "${YELLOW}Update cancelled${NC}"
    exit 0
fi

echo ""
echo -e "${BLUE}Updating genesis timestamp...${NC}"

# Backup original file
cp "$HANDLERS_FILE" "$HANDLERS_FILE.backup.$(date +%s)"
echo -e "${GREEN}✓${NC} Created backup: $HANDLERS_FILE.backup.*"

# Update the timestamp
sed -i "${GENESIS_LINE}s/pub const GENESIS_TIMESTAMP: u64 = [0-9]*;/pub const GENESIS_TIMESTAMP: u64 = ${NEW_TIMESTAMP};/" "$HANDLERS_FILE"

# Update the comment too
sed -i "${GENESIS_LINE}s|// .*|// ${NEW_DATE}|" "$HANDLERS_FILE"

echo -e "${GREEN}✓${NC} Updated genesis timestamp in $HANDLERS_FILE line $GENESIS_LINE"
echo ""

# Show the change
echo -e "${YELLOW}New configuration:${NC}"
grep "GENESIS_TIMESTAMP" "$HANDLERS_FILE"
echo ""

# Verify the change
echo -e "${BLUE}Verifying emission schedule...${NC}"
python3 << EOF
import time

GENESIS_TIMESTAMP = $NEW_TIMESTAMP
SECONDS_PER_YEAR = 31_536_000
BASE_REWARD = 100_000

current_timestamp = int(time.time())
elapsed_seconds = max(0, current_timestamp - GENESIS_TIMESTAMP)
halving_count = elapsed_seconds // SECONDS_PER_YEAR
reward = BASE_REWARD >> halving_count if halving_count < 64 else 0

print(f"Current time: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime(current_timestamp))}")
print(f"Genesis time: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime(GENESIS_TIMESTAMP))}")

if current_timestamp < GENESIS_TIMESTAMP:
    days_until = (GENESIS_TIMESTAMP - current_timestamp) / 86400
    print(f"Status: Genesis is {days_until:.1f} days in the future ✓")
else:
    days_since = elapsed_seconds / 86400
    print(f"Status: {days_since:.1f} days since genesis")

print(f"Halving count: {halving_count}")
print(f"Block reward: {reward:,} base units = {reward / 100_000_000:.9f} QUG")

if halving_count == 0 and reward == 100_000:
    print("✅ Emission schedule looks correct!")
elif current_timestamp < GENESIS_TIMESTAMP:
    print("✅ Genesis in future - ready for launch!")
else:
    print(f"⚠️  Halving count: {halving_count} (check if this is expected)")
EOF

echo ""
echo -e "${BLUE}Next steps:${NC}"
echo "  1. Review the changes above"
echo "  2. Build the project:"
echo "     ${GREEN}timeout 36000 cargo build --release --package q-api-server${NC}"
echo "  3. Test locally before deploying"
echo "  4. Deploy to production when ready"
echo ""
echo -e "${GREEN}✓ Genesis timestamp updated successfully!${NC}"
echo -e "${BLUE}======================================================================${NC}"
