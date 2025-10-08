#!/bin/bash
# Demo script for Quillon Bank CLI

CLI="./target/x86_64-unknown-linux-gnu/release/quillon-bank"

echo "🏦 QUILLON BANK CLI DEMONSTRATION"
echo "=================================="
echo ""

# Test 1: Bank Status
echo "📊 Test 1: Bank Status"
echo "Command: quillon-bank status"
echo ""
$CLI status
echo ""
echo "---"
echo ""

# Test 2: Natural Language - Good morning
echo "🤖 Test 2: Natural Language Query"
echo 'Question: "Good morning, what needs my attention?"'
echo ""
echo "You: Good morning, what needs my attention?" | $CLI claude-mode 2>/dev/null || {
    # Fallback to ask command
    $CLI ask "status"
}
echo ""
echo "---"
echo ""

# Test 3: Stablecoin Collateral Status
echo "💰 Test 3: QNKUSD Collateral Status"
echo "Command: quillon-bank stablecoin collateral status"
echo ""
$CLI stablecoin collateral status
echo ""
echo "---"
echo ""

# Test 4: Loans at Risk
echo "⚠️  Test 4: Loans at Risk"
echo "Command: quillon-bank lending at-risk"
echo ""
$CLI lending at-risk
echo ""
echo "---"
echo ""

# Test 5: Risk Assessment
echo "🎯 Test 5: Risk Assessment"
echo "Command: quillon-bank risk assessment daily"
echo ""
$CLI risk assessment daily
echo ""
echo "---"
echo ""

# Test 6: Analytics
echo "📈 Test 6: Daily Summary"
echo "Command: quillon-bank analytics daily-summary"
echo ""
$CLI analytics daily-summary
echo ""

echo "=================================="
echo "✅ Demo Complete!"
echo ""
echo "Try these commands yourself:"
echo "  $CLI status --full"
echo "  $CLI lending at-risk --collateral-ratio-below 110%"
echo "  $CLI stablecoin peg status"
echo "  $CLI treasury reserves status"
echo ""
echo "Or start interactive mode:"
echo "  $CLI claude-mode"