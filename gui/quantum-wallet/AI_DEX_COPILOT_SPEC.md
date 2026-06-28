# AI DEX Copilot — byggespec (v0.1, 2026-06-16)

Forfatter: Rocky (Claude Opus 4.8) sammen med Viktor. Mål: et betalt AI-sidepanel på
DEX-fronten, hvor en bruger i naturligt sprog kan få en `claude code -p`-drevet agent til
at *afprøve* Quillon-DEX-funktionerne via quillon-wallet MCP'en — med terminal-combo-look
(inspireret af Qwen 3.6 på code.qwen.ai). Betaling via eksisterende Stripe-komponent.

## Beslutninger (låst med Viktor 2026-06-16)
- **Agent-mode:** dry-run/proposer som STANDARD; **opt-in ægte** udførelse hvis brugeren
  forbinder sin egen wallet + bekræfter. (Agentic-money-regel #1: aldrig auto-spend.)
- **Motor:** `claude code -p` headless sidecar med quillon-wallet MCP. Anthropic-API per brug.
- **Pris:** Stripe **$2 → N AI-handlinger** (credits). Hver tool-combo trækker credits.
  Måles via eksisterende `/api/wallet/usage` + `/api/pricing`.
- **Rådgivning (OBLIGATORISK):** Copiloten skal ALTID konsultere DeepSeek API, før den
  giver brugeren et råd. Ingen brugervendt anbefaling må udgå uden et DeepSeek-pass.
  Tonen er "utmost professionalisme". Fail-closed (se afsnit nedenfor).

## Genbrugbare brikker (bekræftet i repoet, gui/quantum-wallet/src)
- `components/StripeCheckout.tsx` — betalingskomponent findes allerede.
- `components/DexScreen.tsx` (7.493 linjer) — her dockes sidepanelet.
- AI-backend findes: `/api/chat/create`, `/api/chat/{id}/stream`, `/api/chat/workers`,
  `/api/chat/metrics`, `/api/chat/{id}/switch-model`, `/api/v1/ai/chat`,
  `/api/v1/ai/transaction/prepare`, `/api/pricing`, `/api/wallet/usage`.
- AI-UI at låne mønstre fra: `AgentTerminalModal.tsx`, `AIWorkerPanel.tsx`, `AIWheelButton.tsx`.
- flux-vision visuel verifikation: `/home/storage/deepseek-codewhale/flux-vision/cockpit-shot.mjs`
  (kører på Beta/Delta — node er fjernet fra Epsilon).

## Arkitektur
```
 Bruger (browser, DEX)                 Beta backend                       Sidecar
 ─────────────────────                 ─────────────────────              ──────────────────
 AIDexCopilot.tsx  ──POST /api/ai-copilot/session───▶  Stripe-gate (credits>0?)
   │  (terminal UI)                          │
   │  EventSource ◀──/api/ai-copilot/stream──┤  spawner ──▶ `claude -p --output-format=stream-json`
   │   render combos                         │              med MCP: quillon-wallet (mode=proposer)
   │                                         │              ── tool_use/tool_result frames ──▶ SSE
   └─ "Udfør rigtigt" knap ─▶ brugerens egen wallet-signering (opt-in, bekræft) ─▶ broadcast
```

### Frontend: `components/AIDexCopilot.tsx` (ny)
- Collapsible højre-sidepanel i DexScreen (toggle-knap i DEX-topbaren).
- Terminal-look: monospace, farvede linjer pr. frame-type:
  - `🧠 think` (dæmpet), `⚙ tool_use <navn>(args)` (cyan), `✓ tool_result` (grøn),
    `💱 quote/combo` (guld), `⚠ kræver din wallet` (gul).
- Streamer fra `EventSource('/api/ai-copilot/stream?session=…')`.
- Credit-tæller øverst; ved 0 → `StripeCheckout` ($2 → +N credits).
- "Udfør rigtigt"-knap vises kun på proposer-frames der kan eksekveres; bruger brugerens
  egen autentificerede wallet (eksisterende dex-swap/-send flow), aldrig sidecar-wallet.

### Backend: ny route-gruppe `/api/ai-copilot/*` (q-api-server)
- `POST /session` → opretter session, tjekker credits (Stripe-betalt), returnerer session-id.
- `GET /stream?session=…&prompt=…` → SSE; spawner `claude -p` med:
  - `--output-format stream-json` (parse tool_use/tool_result → SSE-frames)
  - `--mcp-config` peger på quillon-wallet MCP i **proposer/observer**-mode (read + dry-run:
    `dex_get_quote`, `dex_list_pools`, `strategy_dry_run`, `score_tx_dry`, `dex_quickstart_trade`).
  - `--allowedTools` whitelisted til de read/dry-run MCP-værktøjer. INGEN `send_qug`,
    `dex_swap`, `broadcast_to_mainnet` i sidecar-allowlist.
  - hver tool_use dekrementerer credits via `/api/wallet/usage`.
- `POST /execute` → kun ægte vej: validerer brugerens egen wallet-signatur, kører rigtig
  dex_swap/send på brugerens vegne. Adskilt fra AI-sessionen.

### Stripe / credits
- Genbrug `StripeCheckout.tsx`; produkt "AI Copilot — $2 / N handlinger".
- Credits gemmes pr. wallet (eksisterende `/api/wallet/usage`-spor); webhook fra Stripe
  → credit-topup. Tiered planer senere (det Viktor kaldte "andre planer").

## Sikkerhed (ikke-forhandlbar)
- Sidecar-agenten kører ALDRIG med skrive-/send-værktøjer. Kun read + dry-run.
- Ægte pengebevægelse kræver brugerens egen wallet + eksplicit klik. Sidecar kan foreslå,
  aldrig udføre. (Spejler agentic-money-regel #1.)
- Rate-limit + credit-gate før hver `claude -p`-spawn (Anthropic-omkostning).

## Rådgivnings-lag — DeepSeek ALTID (ikke-forhandlbart)
Dette er en hård regel, ikke en optimering. Klar arbejdsdeling:
- **claude -p (orkestrator):** vælger og kører MCP-tool-combos, henter RIGTIGE fakta —
  `dex_get_quote`, `dex_list_pools`, `lp_position_value`, `strategy_dry_run`,
  `score_tx_dry`. Den FORMULERER ikke selv det endelige råd til brugeren.
- **DeepSeek API (rådgiver):** får de indsamlede fakta + brugerens spørgsmål og
  producerer den endelige, professionelle anbefaling, der vises i panelet. Hver
  advisory-frame er mærket "Rådgivning verificeret af DeepSeek".
- **Flow:** prompt → claude-p kører tool-combos → fakta-bundle → `POST` DeepSeek
  (`/chat/completions`, system-prompt = professionel DEX-rådgiver, lav temperatur) →
  svar streames til panelet som det autoritative råd.
- **Fail-closed:** hvis DeepSeek timer ud / fejler, vises de rå MCP-data UDEN en
  anbefaling + besked: "Kan ikke give rådgivning lige nu (rådgivnings-tjeneste
  utilgængelig)." Copiloten improviserer ALDRIG et råd uden DeepSeek-passet.
- **Audit:** hvert råd logges med (prompt, fakta-bundle, DeepSeek-request-id, svar) så
  rådgivningen er sporbar — professionalisme = dokumenterbar, ikke bare tonefald.
- **Omkostning:** DeepSeek-kaldet tæller med i credit-/usage-metering pr. handling.

## Faser
- **P0 design:** DeepSeek-konsultation på panel-UX + flux-vision baseline-shot af nuv. DEX.
- **P1 UI:** `AIDexCopilot.tsx` shell + terminal-render + Stripe-gate (mock-stream).
- **P2 bro:** `/api/ai-copilot/*` + ægte `claude -p` sidecar i proposer-mode.
- **P3 penge:** Stripe $2→credits, metering, tiered planer.
- **P4:** flux-vision render-verify → npm build på Beta → dist-final → cache-bust deploy.

## DeepSeek-konsultation (P0, 2026-06-16) — gennemført
Nøglen er gemt server-side på Beta (`/root/.deepseek-key`, chmod 600 — ikke i repo/memory).
Live-review af designet via `deepseek-chat` gav to design-ændringer, nu kanon:
- **Metering må ikke hvile alene på Stripe-webhook** (race → gratis-brug før webhook fyrer).
  Brug et **signeret usage-token** eller **on-chain forudbetalt saldo** for dyre handlinger.
- **Anti-hallucination (største risiko):** orkestratoren sender **rå simulerings-data** til
  DeepSeek (præcis revert-grund, price-impact %, slippage). Hver advisory-frame **præfikses
  med de rå data**, så brugeren kan verificere rådet uafhængigt af DeepSeeks formulering.

## Åbne punkter til Viktor
1. ✅ DeepSeek-nøgle løst (Beta `/root/.deepseek-key`, 600). Rådgivnings-laget verificeret live.
2. Byg+screenshot kører på Beta (node findes der). OK at jeg arbejder direkte på Beta-kilden?
3. ✅ $2 → N **metrede handlinger** (ikke tidssession). N fastsættes; default-forslag 25.
