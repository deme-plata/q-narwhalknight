# QUILLON VAULT — Full Manufacturing Pipeline

## From Whitepaper to Working Product

**Status:** Planning Phase
**Estimated Timeline:** 9–12 months
**Estimated Total Investment:** $150–400K (for first 1,000 units)
**Target Retail Price:** $149 (Titanium) / $199 (Ceramic)
**Target BOM Cost:** ~$60/unit @ 10K volume

---

## PHASE 0: DESIGN & DOCUMENTATION (COMPLETE)

### Deliverables Completed
- [x] Product concept and industrial design specification (`docs/HARDWARE_WALLET_DESIGN.md`)
- [x] Whitepaper with full security architecture (`papers/quillon-vault-whitepaper.pdf`)
- [x] Product render / concept image (`docs/vault.png`)
- [x] Component BOM with specific part numbers
- [x] Q-NarwhalKnight blockchain integration target

### Deliverables — Claude Can Produce Next
- [ ] KiCad PCB schematic (`.kicad_sch`) with actual component footprints
- [ ] KiCad PCB layout (`.kicad_pcb`) — 4-layer, 0.8mm, 82mm x 50mm
- [ ] Complete BOM in JLCPCB/PCBWay import format (MPN, Mouser/Digikey links, quantities)
- [ ] Mechanical dimension specification with GD&T tolerances
- [ ] STM32 firmware project (complete C codebase)
- [ ] Dilithium5 signing library ported to ARM Cortex-M4
- [ ] USB HID protocol specification (byte-level command/response format)
- [ ] Desktop wallet application (Tauri)
- [ ] Browser WebHID integration for quillon.xyz
- [ ] Factory test firmware and QC scripts
- [ ] Common Criteria Security Target document

---

## PHASE 1: ELECTRICAL ENGINEERING

**Timeline:** Weeks 1–6
**Budget:** $5–15K

### What Happens

1. **Schematic Review** — Hired EE verifies component selections work together:
   - STM32L4S5: 1.71–3.6V operating range
   - ATECC608B: 2.0–5.5V
   - SSD1306 OLED: 3.3V
   - All powered from USB 5V/500mA bus power (no battery)
   - Power tree: USB 5V → LDO → 3.3V rail → all components

2. **PCB Layout** — EE takes schematic and creates 4-layer, 0.8mm board:
   - Internal dimensions: 82mm x 50mm (shell wall thickness eats ~1.8mm/side)
   - USB-C connector at right edge (slide mechanism access)
   - OLED centered under sapphire window
   - Secure Element physically isolated traces (security routing)
   - All components within 2.0mm height budget
   - Impedance-controlled USB differential pair

3. **DFM Review** — Design for Manufacturability check:
   - 4-layer 0.8mm is aggressive but doable at any major fab
   - Minimum trace/space: 4mil/4mil (standard)
   - Component placement compatible with pick-and-place machines
   - Thermal relief on ground pads

4. **Prototype Order**:
   - 5–10 bare PCBs: ~$50 (JLCPCB)
   - SMT assembly with components: ~$500–2,000 for 5 boards
   - QRNG chip ($12 each) is most expensive per-board component
   - Lead time: 2–3 weeks for PCB + assembly

### Who to Hire
- **Freelance EE** (Upwork/Toptal) with STM32 + secure element experience: $3–8K
- **Turnkey firms**: Altium Designer Services, Seeed Studio Fusion, MacroFab
- **PCB fabrication**: JLCPCB, PCBWay (China), OSH Park (US, higher quality)

### Claude's Role in Phase 1
- Write complete KiCad schematic netlist
- Generate BOM in exact PCBWay/JLCPCB format
- Write test firmware (blink LED, drive OLED, talk to SE) for board bring-up
- Write automated test scripts to validate each component

---

## PHASE 2: MECHANICAL ENGINEERING

**Timeline:** Weeks 2–8 (parallel with Phase 1)
**Budget:** $10–25K

### What Happens

1. **3D CAD Modeling** (SolidWorks / Fusion360):
   - Top titanium shell: OLED window cutout, logo pocket, chamfered edges
   - Bottom titanium shell: slide rail channels, screw bosses
   - Sliding frame: magnet pockets, spring contact housing
   - Sapphire glass window (sourced from watch crystal supplier)
   - Full assembly with PCB clearance verification

2. **Tolerance Analysis**:
   - Slide frame: 8mm travel with <0.05mm lateral play
   - Ceramic ball bearings: channels concentric to ±0.02mm
   - Total stack-up: verify 3.8mm thickness with real machining tolerances
   - Shell-to-PCB clearance: 0.1mm minimum each side

3. **Magnet Detent Simulation**:
   - N52 neodymium magnets: ~1–2N pull force at detent positions
   - Two energy wells: LOCKED and UNLOCKED positions
   - Shallow energy barrier between them (frame doesn't stop midway)
   - Magnet size: 2mm x 1mm x 0.5mm discs

4. **Hardware Switch Design**:
   - Miniature slide switch embedded in rail
   - Electrical: 3.3V/100mA (USB data), <50mΩ closed, >10MΩ open
   - Spring-loaded gold-plated contact pin
   - Actuated by frame position, not user force

5. **CNC Prototyping**:
   - First prototype in 6061 aluminum: ~$200–500/unit (faster, cheaper)
   - Second prototype in Grade 5 titanium (Ti-6Al-4V): ~$500–1,500/unit
   - 3–5 shell sets for PCB fit testing
   - PVD coating samples: TiN (gold), TiAlN (black), rose gold

### Who to Hire
- **ME / Industrial Designer** with consumer electronics enclosures: $5–15K
- **CNC prototyping**: Fictiv ($300–800/unit Ti), Xometry, Protolabs, Star Rapid
- **Sapphire windows**: Stettler Sapphire (Swiss), Alibaba suppliers (~$3–8/piece)
- **Magnets**: K&J Magnetics, first4magnets (N52 micro-disc)

### Claude's Role in Phase 2
- Write exact dimensional specification for ME to work from
- Calculate magnet force curves and detent spacing
- Spec the spring contact mechanism with electrical requirements
- Write PVD coating specification (material, thickness, color targets)

---

## PHASE 3: FIRMWARE DEVELOPMENT

**Timeline:** Weeks 4–14
**Budget:** $0 (Claude writes it) to $5–10K (if hiring contractor to review/polish)

### Architecture

```
┌─────────────────────────────────────────────────────┐
│                   STM32L4S5 MCU                      │
│                                                      │
│  ┌─────────────────┐  ┌──────────────────────────┐  │
│  │  SECURE WORLD   │  │    NON-SECURE WORLD      │  │
│  │  (TrustZone)    │  │                          │  │
│  │                 │  │  USB HID Protocol        │  │
│  │  SE (I2C) comm  │  │  OLED Display Driver     │  │
│  │  Dilithium5 sig │  │  Button Handler          │  │
│  │  Ed25519 sig    │  │  Transaction Parser      │  │
│  │  Key derivation │  │  Firmware Update Logic   │  │
│  │  QRNG interface │  │  LED/Haptic Control      │  │
│  │                 │  │                          │  │
│  └────────┬────────┘  └────────────┬─────────────┘  │
│           │ NSC gateway             │                 │
│           └─────────────────────────┘                 │
└─────────────────────────────────────────────────────┘
         │           │          │           │
    ┌────┴──┐  ┌────┴───┐  ┌──┴──┐  ┌────┴─────┐
    │ATECC  │  │SSD1306 │  │QRNG │  │ USB-C    │
    │608B   │  │OLED    │  │chip │  │ (HID)    │
    │(I2C)  │  │(I2C)   │  │(SPI)│  │          │
    └───────┘  └────────┘  └─────┘  └──────────┘
```

### Firmware Modules (Claude Writes All)

1. **Boot Sequence**
   - Secure boot validation (verify firmware signature in SE)
   - Peripheral initialization: I2C (SE, OLED), SPI (Flash, QRNG), USB
   - TrustZone configuration (SAU, IDAU regions)
   - Display "QUILLON VAULT" splash on OLED
   - Self-test all components

2. **USB HID Interface**
   - Custom HID device class (not mass storage)
   - Kyber-1024 handshake on connection establishment
   - AES-256-GCM encrypted command/response channel
   - Command set:
     - `GET_ADDRESS` — return wallet address
     - `SIGN_TX` — sign a transaction (requires OLED confirmation)
     - `GET_VERSION` — firmware version
     - `UPDATE_FIRMWARE` — enter DFU mode (requires OLED confirmation)
     - `GET_PUBLIC_KEY` — return Ed25519 + Dilithium5 public keys
     - `VERIFY_DEVICE` — device certificate attestation

3. **Secure Element Communication**
   - ATECC608B I2C protocol implementation
   - Key slot management (up to 16 key slots)
   - ECDSA/EdDSA signing within SE
   - Monotonic counter increment per signature
   - Secure boot attestation chain

4. **Dilithium5 Signing Engine**
   - Ported from pqcrypto reference implementation to ARM Cortex-M4
   - Optimized for 120MHz M4 with DSP instructions
   - Key material held in TrustZone secure memory
   - Signing time target: <500ms per signature
   - 4,627-byte signature output

5. **OLED Display Driver**
   - SSD1306 I2C driver (400kHz fast mode)
   - 5x7 monospace font for address display
   - 8x12 font for amounts and labels
   - Screen states:
     - SPLASH: "QUILLON VAULT" logo
     - ADDRESS: truncated wallet address
     - TX_CONFIRM: "SEND 50.0 QUG / To: qnk48cc... / [✓] [✗]"
     - PIN_ENTRY: "Enter PIN: ****"
     - FW_UPDATE: "Update v1.0 → v1.1? [✓] [✗]"
     - ERROR: error message display

6. **QRNG Interface**
   - SPI read from Quantis chip (up to 4 Mbps)
   - NIST SP 800-90B health checks (repetition count, adaptive proportion)
   - XOR conditioning with SE internal TRNG
   - Entropy pool management (512-bit buffer)

7. **Transaction Parser**
   - Q-NarwhalKnight transaction deserialization
   - Display fields: amount, recipient, token type, fee
   - SHA-3-256 hash computation
   - Support for: QUG, QUGUSD, custom tokens, DEX swaps, RWA tokens

### Development Tools
- **Toolchain**: ARM GCC (`arm-none-eabi-gcc`)
- **Build system**: CMake
- **Debug**: OpenOCD + ST-Link V2 (~$20)
- **Testing**: Unity test framework (embedded) + host-side simulation
- **CI**: GitHub Actions with QEMU ARM emulation

---

## PHASE 4: HOST APPLICATION

**Timeline:** Weeks 8–14
**Budget:** $0 (Claude writes it) to $5K (if hiring for UI polish)

### Desktop App (Tauri — Rust + Web Frontend)

```
┌────────────────────────────────────────────────┐
│            QUILLON VAULT Manager               │
│                                                │
│  ┌──────────────────────────────────────────┐  │
│  │  Device: QUILLON VAULT v1.0              │  │
│  │  Status: Connected ● (USB-C)             │  │
│  │  Firmware: 1.0.0                         │  │
│  │  Serial: QV-2026-00042                   │  │
│  └──────────────────────────────────────────┘  │
│                                                │
│  ┌─ Wallets ──────────────────────────────┐    │
│  │  QUG:     1,250.00 QUG ($53,125)       │    │
│  │  QUGUSD:  5,000.00 QUGUSD              │    │
│  │  BRC:     100,000 BRC ($2,450)         │    │
│  └────────────────────────────────────────┘    │
│                                                │
│  [ Send ]  [ Receive ]  [ Settings ]           │
│                                                │
└────────────────────────────────────────────────┘
```

Features:
- USB HID detection and Kyber-1024 session establishment
- Wallet balance display (queries Q-NarwhalKnight node)
- Transaction construction and device signing
- Dual-signature broadcast to network
- Firmware update interface
- PIN management
- BIP-39 backup initiation (mnemonic displayed on device OLED only)
- Device health monitoring

### Browser WebHID Integration

Adds hardware wallet support to the existing quillon.xyz wallet frontend:
- WebHID API connection to QUILLON VAULT
- "Sign with Hardware Wallet" button on send/swap/deploy screens
- Transaction constructed in browser, sent to device, signature returned
- Works in Chrome, Edge, Opera (WebHID support)

---

## PHASE 5: FIRST WORKING PROTOTYPE

**Timeline:** Weeks 14–16
**Budget:** $2–5K

### Assembly Steps (3–5 units, hand-assembled)

1. Place PCB into bottom titanium shell
2. Connect OLED flex cable, position under sapphire window
3. Insert slide mechanism assembly (rail + bearings + magnets + spring contact)
4. Solder spring contact wires to PCB USB D+/D- breakout pads
5. Close top shell, secure with M1.2 micro-screws (4x)
6. Flash firmware via SWD debug port (ST-Link programmer)
7. First boot — OLED shows "QUILLON VAULT"
8. Run factory self-test (all components verified)
9. Generate first key pair (QRNG entropy → SE)
10. Connect to host app, perform first transaction signing

### Validation Checklist
- [ ] OLED displays correctly through sapphire window
- [ ] Slide mechanism has clean LOCKED/UNLOCKED detent clicks
- [ ] USB-C enumerates only when frame is in UNLOCKED position
- [ ] USB-C is electrically dead when frame is LOCKED (oscilloscope verify)
- [ ] SE generates key pair successfully
- [ ] Ed25519 signature verifies on host
- [ ] Dilithium5 signature verifies on host
- [ ] Transaction round-trip: host → device → OLED → confirm → sign → host
- [ ] QRNG entropy passes NIST health checks
- [ ] Haptic feedback works on both buttons
- [ ] Device draws <150mA from USB at peak (during signing)

---

## PHASE 6: SECURITY AUDIT & CERTIFICATION

**Timeline:** Weeks 16–24
**Budget:** $15–50K

### Security Audit ($10–30K)

**Firms to hire:**
- **Ledger Donjon** (Paris) — World's top hardware wallet security lab
- **NCC Group** (UK/US) — Embedded security specialists
- **Trail of Bits** (NYC) — Smart contract + hardware audit
- **Riscure** (Netherlands) — Side-channel analysis specialists

**What they test:**
- Fault injection: voltage glitching, clock glitching, laser fault injection
- Side-channel analysis: power traces (DPA/SPA), electromagnetic emanation
- Firmware extraction: JTAG/SWD lockout verification, flash readout protection
- USB protocol fuzzing: malformed packets, buffer overflows
- Tamper mesh bypass: can the mesh be defeated without triggering wipe?
- Secure boot bypass: can unsigned firmware be loaded?
- Key extraction: can SE keys be read through any side channel?

### Regulatory Certification ($3–8K)

- **FCC Part 15** (US) — Unintentional radiator, required for US sales
- **CE Mark** (EU) — EMC Directive 2014/30/EU
- **Test labs**: TUV, Intertek, SGS, UL
- Timeline: 4–6 weeks after submission

### Optional: Common Criteria ($20–100K)

- EAL4+ or EAL5+ evaluation
- Required for government/institutional customers
- Ledger Nano X has CC EAL5+ (via ST33 SE)
- Our ATECC608B is not CC certified, but the system can be evaluated

---

## PHASE 7: PRODUCTION

**Timeline:** Weeks 24–36
**Budget:** $100–300K for first 1,000 units

### Tooling ($20–30K)
- Injection molds for slide rail plastic components: $5–15K
- CNC fixtures for titanium shells: $3–5K per fixture
- Pick-and-place programming and stencils: $1–2K
- Test jig for factory QC: $2–5K

### Component Sourcing (Lead Times)
| Component | Supplier | Lead Time |
|-----------|----------|-----------|
| STM32L4S5 | Mouser/Digikey/ST Direct | 8 weeks |
| ATECC608B | Microchip Direct | 12 weeks |
| IDQ Quantis QRNG | ID Quantique (Switzerland) | 16 weeks |
| SSD1306 OLED | Alibaba/Solomon Systech | 4 weeks |
| USB-C connector | Molex/JAE | 6 weeks |
| Sapphire glass | Watch crystal supplier | 8 weeks |
| Ti-6Al-4V bar stock | Titanium distributor | 4 weeks |
| N52 magnets | K&J Magnetics | 2 weeks |
| Ceramic ball bearings | Boca Bearings | 3 weeks |

**Critical path: QRNG chip at 16 weeks. Order first.**

### Contract Manufacturers
- **Flex Ltd** (Singapore) — Built hardware wallets before, can do full stack
- **Pegatron** (Taiwan) — Apple-tier quality, higher MOQ
- **Shenzhen CM** (various) — Lower cost, need more QC oversight
- **MacroFab** (US) — Good for <1000 units, US-based

### Per-Unit Production Cost @ 1,000 units
| Item | Cost |
|------|------|
| BOM (all components) | $60 |
| PCB assembly (SMT) | $5 |
| CNC titanium shells | $18 |
| Slide mechanism assembly | $5 |
| Final assembly + test | $8 |
| Packaging (box, cable, seed card) | $6 |
| **Total per unit** | **~$102** |

### Per-Unit Production Cost @ 10,000 units
| Item | Cost |
|------|------|
| BOM (volume pricing) | $48 |
| PCB assembly | $3 |
| CNC titanium shells (batch) | $12 |
| Slide mechanism (injection molded) | $2 |
| Final assembly + test | $5 |
| Packaging | $4 |
| **Total per unit** | **~$74** |

### Quality Control (Every Unit)
1. Automated electrical test via factory test firmware
2. Slide mechanism force test (detent strength within spec)
3. OLED visual inspection
4. USB enumeration verification
5. SE key generation test
6. QRNG entropy quality validation
7. Physical inspection (cosmetic defects, alignment)

---

## PHASE 8: LAUNCH

**Timeline from start:** ~9–12 months
**First batch:** 1,000 units

### Pricing
| Variant | Retail | Margin |
|---------|--------|--------|
| Titanium (Natural/Obsidian/Rose) | $149 | ~46% |
| Ceramic (Stealth/Arctic) | $199 | ~49% |
| Carbon Fiber | $179 | ~43% |

### Sales Channels
- quillon.xyz direct sales (highest margin)
- Amazon (visibility, lower margin)
- Crypto hardware resellers

### Marketing Assets
- Whitepaper (done)
- Product renders (done)
- Video: slide mechanism in action
- Comparison table vs Ledger/Trezor/Coldcard
- Security audit report (published)

---

## COST SUMMARY

| Phase | Budget Range | Timeline |
|-------|-------------|----------|
| Phase 0: Design (DONE) | $0 | Complete |
| Phase 1: Electrical Engineering | $5–15K | Weeks 1–6 |
| Phase 2: Mechanical Engineering | $10–25K | Weeks 2–8 |
| Phase 3: Firmware Development | $0–10K | Weeks 4–14 |
| Phase 4: Host Application | $0–5K | Weeks 8–14 |
| Phase 5: First Prototype | $2–5K | Weeks 14–16 |
| Phase 6: Audit + Certification | $15–50K | Weeks 16–24 |
| Phase 7: Production (1K units) | $100–300K | Weeks 24–36 |
| **TOTAL** | **$132–410K** | **9–12 months** |

### Revenue Projection (1,000 units @ $149 avg)
- Revenue: $149,000
- COGS: ~$102,000
- Gross profit: ~$47,000
- Breakeven on $150K investment: ~3,200 units

### Revenue Projection (10,000 units @ $149 avg)
- Revenue: $1,490,000
- COGS: ~$740,000
- Gross profit: ~$750,000

---

## RWA TOKENIZATION — QUILLON VAULT ON-CHAIN

The QUILLON VAULT can be tokenized as a Real World Asset (RWA) on Q-NarwhalKnight using the existing PhysicalGood smart contract in VittuaVM:

### Token: $VAULT

- **Type**: PhysicalGood RWA token
- **Backing**: Each token represents 1 physical QUILLON VAULT device
- **Supply**: Matches production quantity (e.g., 1,000 tokens for first batch)
- **Redemption**: Token holder can burn $VAULT to receive a physical device shipped to their address
- **Price oracle**: Pegged to retail price ($149 per token)

### Use Cases
1. **Pre-sale funding**: Sell $VAULT tokens before production to fund manufacturing
2. **Secondary market**: Holders can trade $VAULT on the DEX before/after redemption
3. **Proof of ownership**: On-chain proof of QUILLON VAULT purchase
4. **Batch tracking**: Each token linked to a production batch / serial range
5. **Warranty NFT**: Burned on redemption, replaced with a device-specific warranty token

### Smart Contract Integration
- Deploy via VittuaVM `PhysicalGood` contract type
- Set `totalSupply` = production batch size
- Set `redemptionEnabled` = true after production complete
- Revenue auto-distribution to project treasury
- Compliance: KYC for physical shipment (address collection on redemption)

---

*Document generated February 2026. QUILLON VAULT is a product of the Q-NarwhalKnight Project.*
*https://quillon.xyz*
