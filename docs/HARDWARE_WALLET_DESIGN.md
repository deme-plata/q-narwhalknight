# Q-NarwhalKnight Hardware Wallet: "QUILLON VAULT"

## Design Philosophy
**As thin as a credit card. As secure as a vault. As elegant as jewelry.**

---

## FORM FACTOR

```
Dimensions: 85.6mm x 54mm x 3.8mm (ISO/IEC 7810 credit card footprint)
Weight: 28g (titanium frame) / 22g (aluminum frame)
Profile: Thinner than 2 stacked credit cards
```

---

## EXTERIOR DESIGN — EXPLODED VIEW (Side Profile)

```
                    3.8mm total thickness
                 |<========================>|

    ┌─TITANIUM TOP SHELL (0.6mm)────────────────────────────────┐
    │  Brushed titanium / PVD black / rose gold anodized        │
    │  Laser-etched Q logo + "QUILLON VAULT" wordmark           │
    │  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓ │
    ├────────────────────────────────────────────────────────────┤
    │  OLED WINDOW (sapphire glass, 0.3mm)                      │
    │  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ │
    ├────────────────────────────────────────────────────────────┤
    │  PCB + COMPONENTS (2.0mm)                                 │
    │  ████████████████████████████████████████████████████████ │
    │  [SE] [MCU] [OLED] [USB-C] [BTN] [FLASH] [QRNG]         │
    ├────────────────────────────────────────────────────────────┤
    │  TITANIUM BOTTOM SHELL (0.6mm)                            │
    │  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓ │
    ├────────────────────────────────────────────────────────────┤
    │  MAGNETIC SLIDE RAIL (0.3mm)                              │
    └───────────────────────────────────────────────────────────┘
```

---

## TOP VIEW — CLOSED STATE

```
    ╔══════════════════════════════════════════════════════════╗
    ║                                                          ║
    ║          ┌──────────────────────────────────┐            ║
    ║          │    ◆  Q U I L L O N  V A U L T   │            ║
    ║          │         ─────────────             │            ║
    ║          │     Post-Quantum Secure           │            ║
    ║          └──────────────────────────────────┘            ║
    ║                                                          ║
    ║   ┌─────────────────────────────────────────────────┐    ║
    ║   │░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░│    ║
    ║   │░░░░░░░░░░░░░ OLED DISPLAY ░░░░░░░░░░░░░░░░░░░░│    ║
    ║   │░░░░░░░░░░░░░ (sapphire) ░░░░░░░░░░░░░░░░░░░░░░│    ║
    ║   │░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░│    ║
    ║   └─────────────────────────────────────────────────┘    ║
    ║                                                          ║
    ║      [ CONFIRM ]              [ REJECT ]                 ║
    ║       (haptic)                 (haptic)                  ║
    ║                                                          ║
    ╠══════════════════════════════════╦═══════════════════════╣
    ║  PROTECTIVE METAL FRAME ◄────── ║ ◄── SLIDE RAIL        ║
    ╚══════════════════════════════════╩═══════════════════════╝
                                              ↑
                                        USB-C HIDDEN
                                      (frame covers it)
```

---

## THE SLIDING MECHANISM — "VAULT LOCK"

The key design feature. A machined titanium frame slides along
precision rails to expose/protect the USB-C port.

```
    ═══════════════════════════════════════════════════
    STATE 1: LOCKED (USB-C Protected)
    ═══════════════════════════════════════════════════

    ┌────────────────────────────────────────────┬────┐
    │                                            │████│ ← Frame end cap
    │           WALLET BODY                      │████│   (covers USB-C)
    │                                            │████│
    └────────────────────────────────────────────┴────┘
     ▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔
     ◄──── Magnetic detent holds frame CLOSED ────►


    ═══════════════════════════════════════════════════
    STATE 2: UNLOCKED (Slide frame 8mm to expose USB-C)
    ═══════════════════════════════════════════════════

         8mm
         ←──→
    ┌────┬────────────────────────────────────────────┐
    │████│                                            │
    │████│           WALLET BODY                ╔═══╗ │
    │████│                                      ║USB║ │ ← USB-C exposed
    └────┴────────────────────────────────────────────┘
     ▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔
          ◄── Satisfying click at open position ──►


    ═══════════════════════════════════════════════════
    CROSS-SECTION OF SLIDE RAIL
    ═══════════════════════════════════════════════════

            Frame (slides)
            ┌──────────┐
            │ Titanium  │
    ┌───────┤          ├───────┐
    │ ◉ ◉ ◉│  Rail    │◉ ◉ ◉ │  ← Ceramic ball bearings
    │       │ Channel  │       │     (smooth, zero wobble)
    └───────┤          ├───────┘
            │ Titanium  │
            └──────────┘
               Body

    ◉ = N52 neodymium magnets (detent positions: LOCKED / UNLOCKED)
        Provides tactile "click" at both positions
```

---

## EDGE PROFILE — ALL 4 SIDES

```
    ╔═══════════════════════════ TOP EDGE ════════════════════════════╗
    ║  Chamfered 45° edge, mirror-polished                           ║
    ║  Contrast: brushed face + polished chamfer (Apple Watch style) ║
    ╚════════════════════════════════════════════════════════════════╝

        LEFT EDGE                              RIGHT EDGE
    ┌──────────────┐                      ┌──────────────┐
    │              │                      │              │
    │  (blank,     │                      │   USB-C      │
    │   seamless)  │                      │   port       │
    │              │                      │   cutout     │
    │              │                      │   (when      │
    │              │                      │    open)     │
    └──────────────┘                      └──────────────┘

    ╔════════════════════════ BOTTOM EDGE ════════════════════════════╗
    ║  Slide rail groove (0.3mm deep, barely visible)                ║
    ║  Two tiny dots: ● LOCKED position   ● UNLOCKED position       ║
    ╚════════════════════════════════════════════════════════════════╝
```

---

## INTERNAL COMPONENT LAYOUT (PCB Top View)

```
    ┌─────────────────────────────────────────────────────────┐
    │                                                         │
    │  ┌─────────┐  ┌─────────────────────────────────────┐  │
    │  │ ATECC   │  │                                     │  │
    │  │ 608B    │  │     0.42" OLED Display              │  │
    │  │ Secure  │  │     72 x 40 px monochrome           │  │
    │  │ Element │  │     (Tx confirmation + addresses)   │  │
    │  │ (2x3mm) │  │                                     │  │
    │  └─────────┘  └─────────────────────────────────────┘  │
    │                                                         │
    │  ┌──────────┐  ┌──────┐  ┌──────┐  ┌───────────────┐  │
    │  │ STM32    │  │ 2MB  │  │ QRNG │  │               │  │
    │  │ L4S5VI   │  │ Flash│  │ Chip │  │   USB-C       │  │
    │  │ (MCU)    │  │      │  │      │  │   Receptacle  │  │
    │  │ ARM M4   │  │      │  │      │  │   (mid-mount) │  │
    │  │ 120MHz   │  │      │  │      │  │               │  │
    │  └──────────┘  └──────┘  └──────┘  └───────────────┘  │
    │                                                         │
    │  ┌──────────────────┐  ┌──────────────────┐            │
    │  │ [■] CONFIRM BTN  │  │ [■] REJECT BTN   │            │
    │  │  Capacitive touch │  │  Capacitive touch │            │
    │  │  + LRA haptic     │  │  + LRA haptic     │            │
    │  └──────────────────┘  └──────────────────┘            │
    │                                                         │
    └─────────────────────────────────────────────────────────┘
```

---

## KEY COMPONENTS

| Component | Part | Size | Purpose |
|-----------|------|------|---------|
| **Secure Element** | ATECC608B-TNGTLS | 2x3mm UDFN | Private key storage, ECDSA/EdDSA, tamper-resistant |
| **MCU** | STM32L4S5VIT6 | 10x10mm LQFP | ARM Cortex-M4, 120MHz, 2MB Flash, TrustZone |
| **Display** | SSD1306 0.42" OLED | 11.6x12mm | 72x40 monochrome, Tx verification |
| **USB** | USB-C mid-mount | 8.9x2.5mm | Data + power (no battery needed) |
| **QRNG** | IDQ Quantis QRNG | 3x3mm QFN | True quantum random number generation |
| **Flash** | W25Q16JV | 3x4mm SOIC-8 | 2MB firmware storage |
| **Haptic** | LRA vibration motor | 4x4x1.5mm | Tactile feedback for button presses |
| **Buttons** | Capacitive touch pads | Flush-mount | Zero mechanical protrusion |

---

## COLOR & MATERIAL OPTIONS

```
    ┌──────────────────────────────────────────────────────┐
    │                                                      │
    │   OBSIDIAN                TITANIUM              ROSE │
    │   ▓▓▓▓▓▓▓▓              ░░░░░░░░            ▒▒▒▒▒▒ │
    │   ▓▓▓▓▓▓▓▓              ░░░░░░░░            ▒▒▒▒▒▒ │
    │   ▓▓▓▓▓▓▓▓              ░░░░░░░░            ▒▒▒▒▒▒ │
    │                                                      │
    │   PVD Black              Natural              Rose   │
    │   Titanium               Grade 5              Gold   │
    │   + Gold logo            Titanium              PVD   │
    │                          + Silver logo        + White│
    │                                                logo  │
    │                                                      │
    │   ─────────────────────────────────────────────────  │
    │                                                      │
    │   STEALTH               ARCTIC               CARBON │
    │   ████████              ████████             ▓▓▓▓▓▓ │
    │   ████████              ████████             ▓▓▓▓▓▓ │
    │   ████████              ████████             ▓▓▓▓▓▓ │
    │                                                      │
    │   Matte Black            Ceramic              Carbon │
    │   Ceramic                White                Fiber  │
    │   (stealth,              Zirconia             Inlay  │
    │    no logo)              + Titanium           + Ti   │
    │                          frame                frame  │
    └──────────────────────────────────────────────────────┘
```

---

## USB-C INTERACTION FLOW

```
    ┌─────────────────────────────────────────────────┐
    │                                                 │
    │  1. SLIDE FRAME  ──►  Click! (magnetic detent)  │
    │     to expose USB-C                             │
    │                                                 │
    │  2. PLUG IN USB-C  ──►  OLED wakes up           │
    │     to computer          Shows: "QUILLON VAULT" │
    │                          Then: wallet address   │
    │                                                 │
    │  3. SIGN TRANSACTION ──►  OLED shows:           │
    │     from computer         ┌─────────────────┐   │
    │                           │ SEND 50.0 QUG   │   │
    │                           │ To: qnk48cc...  │   │
    │                           │                 │   │
    │                           │ [✓]       [✗]   │   │
    │                           └─────────────────┘   │
    │                                                 │
    │  4. TAP CONFIRM ──►  Haptic buzz                │
    │     or REJECT         Dilithium5 signature      │
    │                       sent over USB             │
    │                                                 │
    │  5. UNPLUG  ──►  OLED sleeps                    │
    │                                                 │
    │  6. SLIDE FRAME BACK  ──►  Click! (locked)      │
    │     USB-C protected                             │
    │                                                 │
    └─────────────────────────────────────────────────┘
```

---

## SECURITY ARCHITECTURE

```
    ┌─────────────────────────────────────────────────────────────┐
    │                    QUILLON VAULT SECURITY                    │
    ├─────────────────────────────────────────────────────────────┤
    │                                                             │
    │  LAYER 1: PHYSICAL                                          │
    │  ├─ Titanium shell (drill-resistant)                        │
    │  ├─ Epoxy-potted PCB (no probe access)                      │
    │  ├─ Tamper-evident holographic seal                          │
    │  └─ Mesh tamper detection layer (triggers key wipe)         │
    │                                                             │
    │  LAYER 2: SECURE ELEMENT (ATECC608B)                        │
    │  ├─ Private keys NEVER leave the SE                         │
    │  ├─ Side-channel attack resistant (DPA/SPA)                 │
    │  ├─ Monotonic counter (replay protection)                   │
    │  └─ Secure boot chain verification                          │
    │                                                             │
    │  LAYER 3: FIRMWARE (STM32 TrustZone)                        │
    │  ├─ Signed firmware updates only                            │
    │  ├─ Memory isolation (secure/non-secure worlds)             │
    │  ├─ Stack canaries + ASLR                                   │
    │  └─ Watchdog timer (fault injection protection)             │
    │                                                             │
    │  LAYER 4: CRYPTOGRAPHY                                      │
    │  ├─ Dilithium5 (ML-DSA) post-quantum signatures             │
    │  ├─ Ed25519 classical signatures (dual-sign mode)           │
    │  ├─ SHA-3-256 for all hashing                               │
    │  ├─ QRNG for key generation (true quantum randomness)       │
    │  └─ Kyber-1024 for encrypted USB communication              │
    │                                                             │
    │  LAYER 5: USER VERIFICATION                                 │
    │  ├─ OLED shows EXACT transaction details                    │
    │  ├─ Physical button required (no remote signing)            │
    │  ├─ PIN entry via host computer (SE-locked)                 │
    │  └─ 3 wrong PINs = 24h lockout, 10 = permanent wipe        │
    │                                                             │
    └─────────────────────────────────────────────────────────────┘
```

---

## WHAT MAKES IT EXTRAORDINARY

```
    ┌─────────────────────────────────────────────────────────┐
    │                                                         │
    │  1. QUANTUM RANDOM KEY GENERATION                       │
    │     Hardware QRNG chip generates private keys from      │
    │     quantum shot noise — not PRNGs, not entropy pools.  │
    │     Physically impossible to predict, even with a       │
    │     quantum computer.                                   │
    │                                                         │
    │  2. POST-QUANTUM SIGNATURES ON-DEVICE                   │
    │     Dilithium5 runs on the secure element itself.       │
    │     Private key is generated, stored, and used for      │
    │     signing WITHOUT EVER BEING EXPOSED — not to the     │
    │     host, not to the MCU, not to firmware updates.      │
    │                                                         │
    │  3. DUAL-SIGNATURE MODE                                 │
    │     Every transaction is signed with BOTH:              │
    │       - Ed25519 (classical, fast, compact)              │
    │       - Dilithium5 (post-quantum, NIST Level 5)        │
    │     If quantum computers break Ed25519, the Dilithium   │
    │     signature remains valid. Belt AND suspenders.       │
    │                                                         │
    │  4. ENCRYPTED USB CHANNEL                               │
    │     Communication between wallet and host uses          │
    │     Kyber-1024 key encapsulation + AES-256-GCM.         │
    │     Even a compromised USB stack can't extract keys.    │
    │                                                         │
    │  5. THE SLIDE MECHANISM                                 │
    │     Not just aesthetic — it's a physical security gate.  │
    │     The USB-C data lines are ELECTRICALLY DISCONNECTED   │
    │     when the frame is in LOCKED position. A hardware     │
    │     switch on the slide rail physically breaks the       │
    │     D+/D- traces. No software exploit can bypass a      │
    │     broken circuit.                                     │
    │                                                         │
    └─────────────────────────────────────────────────────────┘
```

---

## SLIDE MECHANISM — ELECTRICAL DISCONNECT

```
    LOCKED (frame closed):

    USB-C Pin ──────┤ ├────── MCU        (circuit BROKEN)
                    GAP
                  (switch open)


    UNLOCKED (frame open):

    USB-C Pin ──────────────── MCU        (circuit COMPLETE)
                  (switch closed,
                   frame pushes
                   contact spring)
```

This means: even if malware on your computer tries to communicate
with the wallet, NOTHING can reach it when the frame is closed.
The air gap is physical, not software.

---

## MANUFACTURING NOTES

### Bill of Materials (Estimated per unit @ 10K volume)

| Component | Est. Cost |
|-----------|-----------|
| Titanium shells (CNC + PVD) | $18 |
| PCB + assembly (4-layer, 0.8mm) | $8 |
| ATECC608B Secure Element | $1.20 |
| STM32L4S5 MCU | $6 |
| 0.42" OLED + sapphire cover | $4 |
| USB-C mid-mount connector | $0.80 |
| QRNG chip (IDQ Quantis) | $12 |
| Slide mechanism (bearings + magnets) | $5 |
| LRA haptic motors (x2) | $1.50 |
| Flash + passives + assembly | $3 |
| **Total BOM** | **~$60** |
| **Target Retail** | **$149 (Ti) / $199 (Ceramic)** |

---

## PACKAGING CONCEPT

```
    ┌─────────────────────────────────────────┐
    │                                         │
    │   Magnetic-close black box              │
    │   (soft-touch matte, foil-stamped Q)    │
    │                                         │
    │   ┌─────────────────────────────────┐   │
    │   │                                 │   │
    │   │    Wallet nestled in molded     │   │
    │   │    microfiber tray              │   │
    │   │                                 │   │
    │   │    ◆ QUILLON VAULT              │   │
    │   │                                 │   │
    │   └─────────────────────────────────┘   │
    │                                         │
    │   Under tray:                           │
    │   - USB-C cable (braided, 30cm)         │
    │   - Recovery seed card (titanium!)      │
    │   - Quick start guide                   │
    │                                         │
    └─────────────────────────────────────────┘
```

The recovery seed card is also titanium — fireproof, waterproof,
stamp your 24 words into metal. Your seed survives what paper won't.

---

*Designed for Q-NarwhalKnight. The first hardware wallet where*
*the USB port has a physical air gap when you're not using it.*
