# Reply to Brandon Ramsay (DSM, metzdowd)

**To:** cryptskii@proton.me (cc: info@irrefutablelabs.org)
**From:** _you_
**Subject:** Re: DSM beta — interest from Quillon Graph

---

Hi Brandon,

Thanks for putting DSM in front of the metzdowd list — strong primer, and the
beta-3 APK actually runs. Skimmed the repo and read the README + onboarding
walkthrough. The structure of the bilateral chain, the Lean4 work
(`DSMOfflineFinality.lean`, `DSM_dBTC_Conservation.lean` in particular), and
the BLE Coordinator on Android together look like a serious shot at
offline-capable settlement done properly.

A bit about us: we run **Quillon Graph** (Q-NarwhalKnight), a live mainnet
quantum-enhanced DAG-BFT chain with sub-3-second finality, Dilithium5 +
Kyber1024 PQ stack, and ~$2B market cap on the QUG token. The repo is at
`code.quillon.xyz` (self-hosted, not GitHub). We have the global-ledger
anchor side of the equation that your design needs for dispute resolution
of last resort.

We see a real fit for DSM's primitive — specifically, a Bluetooth-offline
QUG transfer path between two wallets out of range of internet. Festivals,
transit, remote areas, anywhere connectivity is unreliable. Our planned
adaptation:

- Lift the bilateral hash-chain transaction manager and the BLE Android stack
- Swap dBTC primitives for QUG balances
- Anchor disputes to a Quillon Graph smart contract (collateral + dispute
  window) rather than Bitcoin Signet
- Adapt the Lean4 proofs to our anchor model

We're targeting two-phase rollout: Web Bluetooth API in the desktop wallet
first, then a port to our Expo / React Native mobile app for iOS + Android.

We're not picking this up immediately — we have a production sync stability
issue to land first — but it's queued as a real follow-on initiative.

A few questions when you have a moment:

1. Is the bilateral_transaction_manager.rs design abstracted over the anchor
   chain, or is there Bitcoin-specific coupling I'd need to refactor?
2. Are the Lean4 theorems stated in terms generic to the anchor (i.e.,
   could `DSMOfflineFinality.lean` re-state cleanly with `QUG` replacing
   `dBTC`), or are they coupled to Bitcoin script primitives?
3. Are you open to a collaboration where we contribute back: an
   alternate-anchor adapter, an iOS BLE counterpart to your Android
   Coordinator, and (potentially) co-authored proof extensions?

Happy to email more concretely once we've completed our spike. If you'd
prefer, I can also open a discussion thread on the GitHub repo.

Keep building. Critical-feedback-encouraged is the right disposition.

Best,
_<your name>_

---

## Tone notes for whoever sends this

- Keep it concise. Brandon is on metzdowd; he respects directness.
- Lead with the concrete signal that we *read his work* (specific file names).
- Don't oversell the timeline. We said "not immediately" because that's
  honest.
- The three questions are useful both as research input AND as conversation
  starters that demonstrate technical engagement.
- Don't send commitments we can't keep ("we'll definitely ship X by Y").
- If you want to soften the "$2B market cap" line, drop it — Brandon likely
  doesn't care about caps, but technically-strong contributors do tend to
  prefer protocols with skin in the game.
