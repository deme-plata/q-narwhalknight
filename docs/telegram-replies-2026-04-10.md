# Telegram Community Replies — 2026-04-10
Copy-paste ready. English first, then Chinese below each.

---

## Reply 1: Balance display issue (for 九九艳阳天, jooosen, zise)

**English:**
We understand the concern. This was a wallet display issue — not a balance rollback on the blockchain. Your actual on-chain balance was never modified. The bug was in the wallet frontend: it was incorrectly filtering out some balance values. This has been fixed in v10.2.9. 

Please hard refresh your wallet page (Ctrl+Shift+R or Cmd+Shift+R on Mac). If your balance still looks wrong after refreshing, please share your wallet address here and we will verify it directly on-chain within minutes.

**中文:**
理解大家的担心。这是钱包前端的显示问题，不是链上余额回档。您的链上真实余额从未被修改。这个bug在钱包前端：它错误地过滤掉了一些余额数值。已在v10.2.9中修复。

请硬刷新钱包页面（Ctrl+Shift+R，Mac用Cmd+Shift+R）。如果刷新后余额仍然不正确，请在这里发送您的钱包地址，我们会在几分钟内直接在链上验证。

---

## Reply 2: Server stability (for yyds)

**English:**
The primary server is a 48-core, 64GB RAM, 10Gbit dedicated machine — not a budget server. The interruptions were caused by software bugs in the block producer, not hardware limitations. We've deployed 15+ fixes in the past 48 hours. The network gets stronger with every incident we fix. We appreciate your patience while we harden the system.

**中文:**
主服务器是48核、64GB内存、10Gbit专线的高性能服务器，不是小霸王。中断是区块生产器的软件bug导致的，不是硬件问题。过去48小时我们已部署了15+个修复。网络在每次修复中都变得更强壮。感谢大家在我们加固系统期间的耐心。

---

## Reply 3: Exchange listing (for MOJO NATIONS, ouke, 九九艳阳天)

**English:**
We agree — network maturity comes before exchange listing. That's exactly what we're doing this week: hardening block production, fixing display bugs, and adding watchdog monitoring. We are in active discussions with an exchange, but listing will only proceed after our reliability milestones are met. We will not rush a premature listing that could damage trust. When the time is right, the community will be the first to know.

**中文:**
我们完全同意——网络成熟度在交易所上市之前。这正是我们本周在做的事情：加固区块生产、修复显示bug、添加监控看门狗。我们正在与交易所积极讨论，但只有在可靠性里程碑达标后才会进行上市。我们不会仓促上市损害信任。时机成熟时，社区会第一时间知道。

---

## Reply 4: MetaMask support (for Polar Point)

**English:**
Not yet — QUG is a native Layer 1 blockchain with its own address format (qnk...), similar to Bitcoin. Connecting to MetaMask requires building an EVM-compatible bridge, which is on our roadmap. Current priorities are: network stability → exchange listing → Android wallet → MetaMask bridge. We'll announce each milestone as it's reached.

**中文:**
目前还不行——QUG是原生Layer 1区块链，有自己的地址格式（qnk...），类似比特币。连接MetaMask需要开发EVM兼容的桥接，已列入我们的路线图。当前优先级：网络稳定性 → 交易所上市 → 安卓钱包 → MetaMask桥接。每个里程碑达成时我们都会公告。

---

## Reply 5: GPU mining (for Polar Point, Liss)

**English:**
Yes, QUG supports both CPU and GPU mining. CPU mining is currently more efficient for most setups. The mining algorithm uses VDF (Verifiable Delay Function) which is sequential by nature, so raw GPU parallelism doesn't provide as large an advantage as in other coins. CPU miners with high single-thread performance do well. Download the miner at: https://quillon.xyz/downloads/

**中文:**
是的，QUG支持CPU和GPU挖矿。对大多数配置来说，CPU挖矿目前更高效。挖矿算法使用VDF（可验证延迟函数），本质上是顺序的，所以GPU的并行优势不像其他币那么大。单线程性能高的CPU表现很好。矿工下载：https://quillon.xyz/downloads/

---

## Reply 6: Network status update (general, pin-worthy)

**English:**
📢 **Network Status Update — v10.2.9**

✅ Block production: Normal (2-3 blocks/second)
✅ Mining: Active, 100+ miners connected
✅ Balances: All on-chain balances are correct and safe
✅ Display bug: Fixed — please hard refresh (Ctrl+Shift+R)

Fixes deployed in the past 48 hours:
• Block producer watchdog — detects and recovers from stalls automatically
• Balance display — corrected validation that was filtering some values
• Real-time updates — transfer recipients now see balance changes instantly via SSE
• Connection stability — reverse proxy limits tuned to prevent overload

The network is young and actively being hardened. Every incident makes it stronger. Thank you for your support and patience. 🙏

**中文:**
📢 **网络状态更新 — v10.2.9**

✅ 出块：正常（每秒2-3个区块）
✅ 挖矿：活跃，100+矿工在线
✅ 余额：所有链上余额正确且安全
✅ 显示bug：已修复——请硬刷新（Ctrl+Shift+R）

过去48小时部署的修复：
• 区块生产看门狗——自动检测和恢复停顿
• 余额显示——修正了错误过滤部分数值的验证逻辑
• 实时更新——转账接收方现在通过SSE即时看到余额变化
• 连接稳定性——反向代理限制调优防止过载

网络还年轻，正在积极加固中。每次事件都让它更强壮。感谢大家的支持和耐心。🙏
