# Dragon Ball — Reply to Chong, 2026-05-12

Replies to Chong's two follow-up questions (pre-mining and miner pricing economics) plus his cooperation-model proposal (Dragon Ball does FPGA + chip design + fab + assembly with upfront payment, sells finished miners to Quillon at cost + markup).

## Facts verified in the codebase before drafting

- **Maximum supply**: `QUG_MAX_SUPPLY = 21,000,000` (q-storage/src/emission_controller.rs:143)
- **Era 0 emission**: `BASE_ANNUAL_EMISSION = 2,625,000 QUG/yr` (q-storage/src/emission_controller.rs:151)
- **Halving period**: `SECONDS_PER_HALVING = 126,230,400` (~4 years exactly), q-storage/src/emission_controller.rs:132. **Emission is fully time-scheduled, not block-count-scheduled** — `target_cumulative_emission(elapsed_secs)` and `emission_rate_per_second(k)` are pure functions of wall-clock time since genesis. Block reward = QUG allocated to this time window / blocks produced in that window. Fast blocks → smaller per-block reward; slow blocks → larger. Daily total fixed by era.
- **Dev fee**: **1.9% (190 bps)**, set at q-api-server/src/lib.rs:2796 and :4180 (comment `v8.8.1: 190 bps = 1.9% mainnet dev fee`). Block-producer emits the dev-fee tx with comment "1.9% + rounding remainder" at q-api-server/src/block_producer.rs:1473. (Note: the `DEV_FEE_PERCENT = 0.01` in q-types/src/lib.rs:3263 is a *validation tolerance bound* with 10% slack, not the runtime rate.)
- **No genesis pre-mine** — all supply comes from coinbase emission via the time-based schedule. Current circulating supply at the checkpoint is approximately 497,000 QUG, or ~2.4% of maximum.

## The reply

---

**Hi Chong,**

Thank you for the internal meeting and for laying out a complete cooperation model. Let me answer both of your questions first, then respond to the proposal.

**On pre-mining.** There is no genesis pre-mine. Every QUG in existence has been produced by proof-of-work mining since the chain launched. The consensus constants are: 21,000,000 QUG maximum supply, 2,625,000 QUG/year base emission in Era 0, 4-year Bitcoin-style halvings. Of each block's reward, 98.1% goes to the miner that produced the block and 1.9% goes to a development fund (this is a continuous tax on emissions, not a pre-allocation). As of today approximately 497,000 QUG have been mined, which is ~2.4% of the maximum supply. The remaining 97.6% is unminted and will be released to miners over the coming decades through the halving schedule. We can share the on-chain wallet table and the block-by-block emission curve for independent verification.

**On miner selling price.** Your assessment is correct — ASIC selling price is not determined by BOM cost. It is determined by expected mining profitability over a payback horizon (typically 6-12 months for the industry). The honest math for QUG today:

Our emission is time-scheduled, not block-count-scheduled. Era 0 produces exactly 2,625,000 QUG over a 4-year window, which is approximately 7,180 QUG per day distributed across all blocks the network produces in that day. The per-block reward shrinks or grows depending on block rate, but the daily total is fixed by the protocol. With 293 active miners today that is an average of ~24 QUG per miner per day before electricity. The USD value of that depends entirely on the real external exchange price of QUG, which does not yet exist. Our internal DEX is a price discovery mechanism but it is not the same as listed liquidity. Until QUG trades against USDT on HiBT or another external venue, no rational miner buyer can compute a real ROI, and we are not going to pretend otherwise.

**This connects directly to your proposal.** You are right that the full FPGA → chip design → fab → assembly path with upfront payment is the professional structure. It is also a structure that requires capital we do not yet have. Quillon is community-funded, our exchange listing is targeted for Q4 2026/Q1 2027, and the funding for ASIC tape-out is something we will only have the credibility to raise from external investors *after* a working FPGA validation is in hand. Asking Dragon Ball to front the tape-out cost was never our intention.

**Here is what we are able to commit to today.**

We can sign a formal cooperation agreement covering the full path you described — FPGA validation, chip design, tape-out, manufacturing, and assembly. The agreement specifies your milestone pricing exactly as you laid out: $20,000 initial manpower, $10,000 on BLAKE3 pipeline validation, $10,000 on Genus-2 VDF validation, $15,000 for Virtex UltraScale+ board if multi-core verification is needed, $10,000 for ASIC evaluation. Payments in USDT, funded from accumulated mining revenue and the founder development fund (the 1.9% protocol allocation, which has been compounding since genesis).

We can proceed to the FPGA phases under your existing XC7K355T (no hardware investment) immediately upon signing. We will pay the FPGA milestone amounts on schedule in USDT regardless of whether the project ever proceeds to tape-out. This way Dragon Ball's engineering time is compensated in fiat as required, and your work is not at risk if our investor round does not materialize.

Tape-out and manufacturing only proceed if and when external investor funding is secured — using your validated FPGA results as the de-risking artifact that makes that conversation possible. If the investor round closes, we proceed with your full contract-manufacturing model exactly as you proposed (you produce, we buy at cost + markup, we distribute). If it does not, the FPGA work still has independent value and you have been paid for it.

We understand this is a phased commitment rather than the full upfront structure you described. We believe it is the only honest path given our current capital position, and we want to enter this partnership with both sides understanding the risk profile clearly.

Please let us know if this is workable. If yes, we will draft the cooperation agreement covering all phases with clear milestone definitions, and arrange the first USDT payment upon signing.

Best regards,
Demetri / Quillon Team
