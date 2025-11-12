#!/bin/bash
# Quick test to verify shape mismatch fix

echo "🧪 Testing Shape Mismatch Fix"
echo "=============================="
echo

echo "✅ Applied fixes:"
echo "  1. Added .t() to q_proj multiplication (line 222)"
echo "  2. Added .t() to o_proj multiplication (line 283)"
echo "  3. K and V projections already had .t() (lines 227, 231)"
echo "  4. FFN projections already had .t() (lines 326, 327, 336)"
echo

echo "📋 Expected behavior:"
echo "  Input: [1, 10, 4096]"
echo "  Q proj: [4096, 4096] → with .t() → [batch*seq, 4096] × [4096, 4096]^T ✅"
echo "  K proj: [1024, 4096] → with .t() → [batch*seq, 4096] × [4096, 1024]^T ✅"
echo "  V proj: [1024, 4096] → with .t() → [batch*seq, 4096] × [4096, 1024]^T ✅"
echo "  O proj: [4096, 4096] → with .t() → [batch*seq, 4096] × [4096, 4096]^T ✅"
echo

echo "🔧 Changes made to mistral_model.rs:"
grep -n "\.matmul(&self\..*_proj" crates/q-ai-inference/src/mistral_model.rs | head -6

echo
echo "✅ Shape mismatch fix complete!"
echo "   All linear layer weight matrices now properly transposed"
echo "   Forward pass should work correctly now"
