#!/bin/bash

# Fix consensus latency wording
sed -i 's/Global consensus latency:/Global message propagation latency (P2P overlay):/g' PRIVACY_AS_A_SERVICE_WHITEPAPER.md

# Fix RingCT mention - change to amount bucketing
sed -i 's/Confidential transactions (RingCT)/Amount bucketing + STARK range proofs/g' PRIVACY_AS_A_SERVICE_WHITEPAPER.md

# Fix MEV section status markers
sed -i 's/| \*\*bloXroute\*\* | ✅ Enterprise | \~300ms | \$99\/mo | 🔄 Q1 2026 |/| **bloXroute** | ✅ Enterprise | ~300ms | $99\/mo | 📋 Planned Q1 2026 |/g' PRIVACY_AS_A_SERVICE_WHITEPAPER.md
sed -i 's/| \*\*Blocknative\*\* | ✅ Gas optimization | \~350ms | Custom | 🔄 Q1 2026 |/| **Blocknative** | ✅ Gas optimization | ~350ms | Custom | 📋 Planned Q1 2026 |/g' PRIVACY_AS_A_SERVICE_WHITEPAPER.md

echo "✅ Whitepaper fixes applied"
