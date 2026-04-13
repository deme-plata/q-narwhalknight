# Technical Review v4: SystemVerilog RTL Update Roadmap for Dragon Ball ASIC/FPGA

**Date:** 2026-04-13
**Reviewer:** Quillon Foundation Engineering
**Status:** v3 RTL delivered to Dragon Ball, synthesis pending. This document identifies required code changes.
**Audience:** Dragon Ball Miner engineering team + Quillon RTL developers
**Previous:** v3 (audit-clean delivery), v2 (6 gaps closed), v1 (initial)

---

## Executive Summary

The v3 RTL delivery (6,779 lines, 25 SystemVerilog files) is architecturally complete and audit-clean. However, code review against the live mining protocol (BLAKE3 x100, LWMA difficulty, difficulty-weighted rewards) reveals **8 functional gaps** that must be closed before the design can mine on mainnet. None require architectural changes — all are localized additions to existing modules.

This document specifies exactly what to change, in which file, at which line, and why. Dragon Ball can use this as a work order for their FPGA bring-up.

**Priority ranking:**

| Priority | Gap | Impact if not fixed |
|----------|-----|-------------------|
| P0 | Xcrypto scratchpad stub | BLAKE3 chain produces wrong hashes |
| P0 | Nonce generator + target comparator | Cannot mine at all |
| P1 | Best-hash tracker | Loses ~16x revenue from difficulty-weighted rewards |
| P1 | Variable difficulty target input | Cannot adapt to LWMA difficulty changes |
| P2 | Chain loop dead-cycle elimination | ~12.5% hashrate loss (2 wasted cycles per iteration) |
| P2 | Host interface (SPI/PCIe) | Cannot communicate with miner software |
| P3 | Multi-tile memory arbiter | Single-tile only (FPGA prototype limitation) |
| P3 | ASIC-specific replacements | BRAM→SRAM, MMCM→PLL, clock tree |

---

## 1. P0: Xcrypto Message Block Interface (CRITICAL — Wrong Hashes)

### Problem

In `qug_tile.sv` lines 117-139, the Xcrypto message block memory interface is **stubbed to zero**:

```systemverilog
// Zero-fill message block -- firmware must pre-load message words into
// the state via blake3.init / data memory before invoking blake3.round.
generate
    for (genvar i = 0; i < 16; i++) begin : gen_msg_zero
        assign xc_mem_block[i] = 32'd0;
    end
endgenerate
```

When `xcrypto_unit.sv` enters `S_FETCH_MSG` (line 204), it reads 16 message words from this interface. All 16 words are zero. The first BLAKE3 compression in the chain gets `BLAKE3(0x00...00)` instead of `BLAKE3(challenge || nonce || address)`.

The `blake3.chain` loop then feeds each hash back as the chaining value for the next compression — but the message block stays zero for all 100 iterations. **The entire chain computes hashes of the wrong data.**

### Fix

Replace the zero-fill stub with a **512-bit tightly-coupled scratchpad SRAM** (or connect the xcrypto memory port to the data BRAM via a dedicated read port).

**Option A: Dedicated 512-bit scratchpad (recommended for ASIC)**

Add a new file `rtl/memory/xcrypto_scratchpad.sv`:

```systemverilog
// 512-bit (16 x 32-bit) scratchpad for BLAKE3 message blocks.
// Firmware writes 16 words via 32-bit data memory, then xcrypto reads
// all 16 in a single cycle via the wide port.
module xcrypto_scratchpad (
    input  logic        clk,
    input  logic        rst_n,

    // Narrow write port (from data memory bus, 32-bit)
    input  logic [3:0]  wr_idx,        // Word index 0..15
    input  logic [31:0] wr_data,
    input  logic        wr_en,

    // Wide read port (to xcrypto_unit, 512-bit single-cycle)
    input  logic        rd_en,
    output logic [31:0] rd_data [0:15],
    output logic        rd_valid
);
    logic [31:0] mem [0:15];
    logic        rd_valid_r;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (int i = 0; i < 16; i++) mem[i] <= 32'd0;
            rd_valid_r <= 1'b0;
        end else begin
            if (wr_en) mem[wr_idx] <= wr_data;
            rd_valid_r <= rd_en;
        end
    end

    always_comb begin
        for (int i = 0; i < 16; i++) rd_data[i] = mem[i];
    end

    assign rd_valid = rd_valid_r;
endmodule
```

**Then in `qug_tile.sv`, replace lines 117-139:**

```systemverilog
// Instantiate scratchpad
xcrypto_scratchpad u_xc_scratch (
    .clk      (clk),
    .rst_n    (rst_n),
    .wr_idx   (dmem_addr[5:2]),      // Byte address → word index
    .wr_data  (dmem_wdata),
    .wr_en    (xc_scratch_wr_en),    // Decode from memory map
    .rd_en    (xc_mem_rd_en),
    .rd_data  (xc_mem_block),
    .rd_valid (xc_mem_valid)
);
```

The firmware writes 16 words to the scratchpad (memory-mapped addresses), then issues `blake3.chain rd, rs1, 100`. The scratchpad delivers all 16 words in one cycle.

**Resource cost:** 16 x 32-bit FFs = 512 FFs + ~50 LUTs for write decode. Negligible.

**Option B: Dual-port BRAM connection (quick FPGA fix)**

If adding a new module is undesirable for the first FPGA test, modify `mem_subsystem.sv` to expose a second read port from the data BRAM, and wire it to `xc_mem_block`. This reuses existing BRAM (Kintex-7 BRAMs are true dual-port). But the BRAM returns 32 bits per cycle, not 512 — you'd need 16 sequential reads + a shift register, adding 16 cycles of latency per chain start. Not ideal.

**Recommendation:** Option A for both FPGA and ASIC. It's 50 lines of code and gives single-cycle message delivery.

---

## 2. P0: Nonce Generator + Target Comparator (CRITICAL — Cannot Mine)

### Problem

The current SoC has no hardware nonce generator and no difficulty target comparator. The RISC-V core must iterate nonces in a software loop, calling `blake3.chain` for each one and checking the result in software. This wastes CPU cycles on trivial counting and comparison.

For the ASIC, the nonce loop must be in hardware — the CPU is too slow.

### Fix

Add a new top-level module `rtl/mining/mining_controller.sv`:

```systemverilog
module mining_controller (
    input  logic         clk,
    input  logic         rst_n,

    // Configuration (written by host / CPU)
    input  logic         start,              // Begin mining
    output logic         busy,               // Mining in progress
    input  logic [255:0] challenge_hash,     // 32 bytes
    input  logic [255:0] miner_address,      // 32 bytes
    input  logic [255:0] difficulty_target,  // Variable per block
    input  logic [63:0]  nonce_start,
    input  logic [63:0]  nonce_end,

    // BLAKE3 chain interface (to xcrypto_unit or standalone pipeline)
    output logic         chain_start,
    output logic [31:0]  chain_msg [0:15],   // 512-bit message block
    output logic [6:0]   chain_length,       // 100 for mining
    input  logic [255:0] chain_result,       // 256-bit hash output
    input  logic         chain_done,

    // Results (read by host / CPU)
    output logic         solution_found,
    output logic [63:0]  solution_nonce,
    output logic [255:0] solution_hash,
    output logic         best_updated,       // New best hash found
    output logic [63:0]  best_nonce,
    output logic [255:0] best_hash,          // For difficulty-weighted rewards
    output logic [63:0]  hashes_computed     // Performance counter
);

    // FSM: IDLE → PREPARE → CHAIN → CHECK → (loop or DONE)
    // PREPARE: construct msg = challenge[0:31] || nonce[0:7] || address[0:23]
    //          (72 bytes = 18 words, padded to 64 bytes for BLAKE3 block)
    // CHAIN: issue blake3.chain with length=100, wait for done
    // CHECK: compare chain_result against difficulty_target (lexicographic)
    //        also compare against best_hash (track best for weighted rewards)
    // Loop: increment nonce, go back to PREPARE

    // Target comparison: big-endian byte-by-byte
    // hash[0] is most significant byte (BLAKE3 output order)
    // solution_valid = (chain_result <= difficulty_target)
    //
    // Best-hash tracking: if chain_result < best_hash, update best_*
    // This maximizes difficulty-weighted reward (2^leading_zeros weighting)
endmodule
```

**Key design decisions:**

1. **Nonce increment is a 64-bit counter** — at 1M nonces/sec per core, it takes 584,000 years to wrap. No range check needed in practice.

2. **Message construction** — The mining input is 72 bytes (challenge 32B + nonce 8B + address 32B). BLAKE3 compresses 64 bytes per block. So the first compression takes bytes 0-63, and a second compression takes bytes 64-71 (padded to 64). For the sequential chain, only the first compression uses the actual message — subsequent 99 hashes just chain the previous output as the new chaining value with an empty message. This means the message block only needs to be constructed once per nonce, and the chain loop reuses the latched message (which `xcrypto_unit.sv` already does at line 312-324, `msg_block_lat`).

3. **Target comparator** — Simple 256-bit unsigned comparison. The BLAKE3 output is big-endian (byte 0 = MSB). A valid solution has `hash <= target`. This is ~256 LUTs for a carry chain.

4. **Best-hash tracker** — A second comparator checking `hash < best_hash`. If true, latch the new best nonce and hash. This costs another ~256 LUTs + 320 FFs (nonce + hash registers). The host reads `best_nonce`/`best_hash` when submitting to the pool — submitting a harder solution than required earns proportionally more reward.

**Resource cost:** ~1,000 LUTs + ~800 FFs. Trivial on FPGA or ASIC.

---

## 3. P1: Best-Hash Tracker for Difficulty-Weighted Rewards

### Problem

Quillon uses **difficulty-weighted rewards** (Phase A, deployed on mainnet). Each miner's share of the block reward is proportional to `2^(leading_zeros)` of their submitted hash:

- Hash with 16 leading zero bits → weight = 2^16 = 65,536
- Hash with 20 leading zero bits → weight = 2^20 = 1,048,576

Submitting a hash that is harder than the minimum difficulty earns 16x more reward per submission. An ASIC that only reports the first valid solution (meeting minimum difficulty) throws away revenue.

### Fix

Included in the `mining_controller.sv` design above. The best-hash tracker runs a second comparator on every nonce result. The host reads the best hash periodically (e.g., every 10ms) and submits it. This way, even if the ASIC finds multiple valid solutions, it submits the hardest one.

**Implementation detail in the CHECK state:**

```systemverilog
// In the CHECK state of mining_controller FSM:
logic hash_below_target;
logic hash_below_best;

assign hash_below_target = (chain_result < difficulty_target);
                            // Note: <= for "meets difficulty"
assign hash_below_best   = (chain_result < best_hash);

always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        best_hash  <= {256{1'b1}};  // Start at max (worst)
        best_nonce <= 64'd0;
    end else if (state == S_CHECK) begin
        if (hash_below_target) begin
            solution_found <= 1'b1;
            solution_nonce <= current_nonce;
            solution_hash  <= chain_result;
        end
        if (hash_below_best) begin
            best_hash    <= chain_result;
            best_nonce   <= current_nonce;
            best_updated <= 1'b1;
        end
    end
end
```

---

## 4. P1: Variable Difficulty Target Input

### Problem

After LWMA activation, the difficulty target changes every block (~1 second at target rate). The current design has no mechanism for the host to update the target mid-operation.

### Fix

The `mining_controller.sv` above accepts `difficulty_target` as an input. Two implementation approaches:

**4a. Memory-mapped CSR (for CPU-controlled mining):**

Add a 256-bit CSR in the memory map (address `0x2000_0000 - 0x2000_001F`). The CPU writes the new target before starting each mining round. Simple, works now.

**4b. Host-pushed via SPI/PCIe (for standalone ASIC):**

The host interface (Section 7) writes the target directly. The mining controller has a `target_valid` signal that gates `start`. New challenge + target must be written atomically before starting.

**Critical: Do NOT hardcode the comparator.** The target changes every block. The design already supports this (it's a plain input port), but the firmware and host interface must be designed to update it before each challenge.

---

## 5. P2: Chain Loop Dead-Cycle Elimination (12.5% Hashrate Boost)

### Problem

In `xcrypto_unit.sv`, the `blake3.chain` loop goes through these states per iteration:

```
S_WAIT_PIPELINE (pipe_out_valid) → S_CHAIN_WRITEBACK (1 cycle) → S_COMPRESS (1 cycle) → S_WAIT_PIPELINE (14 cycles)
```

That's 16 cycles per chain iteration: 14 productive (pipeline) + 2 dead (writeback + compress launch). Over 100 iterations: 1,600 total cycles, 200 wasted = **12.5% overhead**.

### Fix

Merge `S_CHAIN_WRITEBACK` and `S_COMPRESS` into a single cycle by doing the CV writeback and pipeline launch simultaneously.

**In `xcrypto_unit.sv`, modify the FSM at line 217:**

```systemverilog
// BEFORE (2 dead cycles):
S_WAIT_PIPELINE: begin
    if (pipe_out_valid) begin
        if (lat_funct7 == F7_CHAIN && chain_count < chain_target) begin
            fsm_next = S_CHAIN_WRITEBACK;  // 1 cycle
        end else begin
            fsm_next = S_IDLE;
        end
    end
end

S_CHAIN_WRITEBACK: begin
    fsm_next = S_COMPRESS;  // 1 cycle
end

// AFTER (0 dead cycles):
S_WAIT_PIPELINE: begin
    if (pipe_out_valid) begin
        if (lat_funct7 == F7_CHAIN && chain_count < chain_target) begin
            fsm_next = S_COMPRESS;  // Go directly to compress
            // CV writeback happens combinationally in this same cycle
        end else begin
            fsm_next = S_IDLE;
        end
    end
end

// Delete S_CHAIN_WRITEBACK entirely
```

**The CV update (state_cv_in) must be changed from registered to combinational:**

Modify the pipeline input control (line 398-420) to read `pipe_hash_out` directly instead of `hash_latched` when transitioning from `S_WAIT_PIPELINE` to `S_COMPRESS`:

```systemverilog
if (fsm_state == S_COMPRESS) begin
    pipe_in_valid = 1'b1;
    if (lat_funct7 == F7_CHAIN && chain_count > 7'd0) begin
        // Use pipe_hash_out directly (bypasses hash_latched register)
        for (int i = 0; i < 8; i++) begin
            pipe_cv[i] = pipe_hash_out[i];  // Was: hash_latched[i]
        end
    end
end
```

**Timing concern:** This adds `pipe_hash_out → pipe_cv → pipeline input latch` to the combinational path. At 100 MHz FPGA this is fine (the hash_out is a simple XOR). At 1.5 GHz ASIC, may need a retiming register — but that's a 1-cycle hit vs 2-cycle hit, still a net gain.

**Result:** 100 iterations × 14 cycles = 1,400 cycles per nonce (was 1,600). **14.3% hashrate improvement**.

Also update `chain_count` increment: move it from `S_CHAIN_WRITEBACK` (line 296) to the `S_WAIT_PIPELINE → S_COMPRESS` transition.

---

## 6. P2: Host Interface (SPI or PCIe)

### Problem

The current SoC communicates via UART at 115,200 baud. This is fine for debugging but far too slow for production mining:

- Mining challenge = 32B challenge + 32B address + 32B target = 96 bytes
- At 115,200 baud (8N1): 96 × 10 bits / 115,200 = 8.3ms per challenge
- New challenges arrive every ~1 second — UART is sufficient for challenge delivery
- But reporting best-hash every 10ms at 115,200 baud is marginal (64B × 10 / 115,200 = 5.6ms)

For a production ASIC, SPI or PCIe is needed.

### Fix

**Option A: SPI slave (simplest, recommended for Dragon Ball's first board)**

Add `rtl/periph/spi_slave.sv` — a standard SPI slave with a register-mapped interface. The host MCU (e.g., STM32 on the miner board) acts as SPI master. Register map:

| Address | Width | R/W | Name | Description |
|---------|-------|-----|------|-------------|
| 0x00 | 256-bit | W | CHALLENGE | Challenge hash |
| 0x20 | 256-bit | W | ADDRESS | Miner address |
| 0x40 | 256-bit | W | TARGET | Difficulty target |
| 0x60 | 64-bit | W | NONCE_START | Starting nonce |
| 0x68 | 64-bit | W | NONCE_END | Ending nonce |
| 0x70 | 8-bit | W | CONTROL | bit 0 = start, bit 1 = stop |
| 0x80 | 8-bit | R | STATUS | bit 0 = busy, bit 1 = solution_found |
| 0x88 | 64-bit | R | SOL_NONCE | Solution nonce |
| 0x90 | 256-bit | R | SOL_HASH | Solution hash |
| 0xB0 | 64-bit | R | BEST_NONCE | Best (hardest) nonce found |
| 0xB8 | 256-bit | R | BEST_HASH | Best hash found |
| 0xD8 | 64-bit | R | HASH_COUNT | Total hashes computed |
| 0xE0 | 32-bit | R | HASHRATE | Hashes/sec (computed by hardware) |

SPI clock: 10 MHz is typical. At 10 MHz, writing a full challenge (96 bytes) takes 76.8 us — negligible vs the 1-second challenge interval.

**Resource cost:** ~500 LUTs + ~300 FFs for SPI slave + register file.

**Option B: PCIe (for rack-mounted ASIC cards)**

Requires a PCIe hard IP block (available in TSMC 7nm/12nm libraries). Dragon Ball has done this before (their KAS/ALPH miners use PCIe or USB). This is Phase 2 — after FPGA validation proves the mining core works.

---

## 7. P3: Multi-Tile Memory Arbiter Improvement

### Problem

The multi-tile path in `qug_soc_top.sv` (lines 312-366, inside `gen_multi_tile`) uses a simplistic round-robin arbiter on a shared memory bus. This creates contention when N tiles compete for BRAM access. Each tile stalls while waiting for its turn.

For FPGA with NUM_TILES=1, this doesn't matter. For ASIC with 8-16 tiles, it becomes the bottleneck.

### Fix

Two approaches, depending on the ASIC tile count:

**7a. Per-tile private SRAM (recommended for ASIC)**

Each tile gets its own 64KB instruction SRAM + 64KB data SRAM. No sharing, no contention. The firmware image is loaded into each tile's IMEM at boot (via SPI or JTAG scan chain). Cost: 128KB SRAM per tile = 2MB for 16 tiles. At 12nm, 2MB SRAM ≈ 2-3 mm². Acceptable.

Modify `qug_soc_top.sv` to instantiate per-tile `mem_subsystem` instead of a shared one:

```systemverilog
generate
    for (genvar t = 0; t < NUM_TILES; t++) begin : gen_tile_mem
        mem_subsystem u_mem_t (
            .clk       (clk),
            .rst_n     (rst_n),
            .req_valid (tile_dmem_req[t]),
            .req_ready (tile_dmem_gnt[t]),
            .req_addr  (tile_dmem_addr[t]),
            .req_wdata (tile_dmem_wdata[t]),
            .req_we    (|tile_dmem_we[t]),
            .resp_rdata(tile_dmem_rdata[t]),
            .resp_valid(/* unused */),
            .uart_tx_data (/* only tile 0 gets UART */),
            ...
        );
    end
endgenerate
```

**7b. Crossbar interconnect (more complex, better for many tiles)**

Replace round-robin with an N×M crossbar (N tiles, M memory banks). Each bank serves one request per cycle. If N=16 and M=4, contention drops to 25% of round-robin. This is standard NoC design — Dragon Ball likely has crossbar IP from their KAS/ALPH ASICs.

**Recommendation for Dragon Ball:** Start with per-tile private SRAM (7a). It's simpler, has zero contention, and the extra SRAM area is negligible at 12nm. Only move to a crossbar if die area is a binding constraint.

---

## 8. P3: ASIC-Specific Replacements

### Problem

Several modules use FPGA-specific constructs that must be replaced for ASIC:

### 8a. BRAM → SRAM Macros

`bram_sp.sv` and `bram_dp.sv` use Xilinx inference attributes (`ram_style = "block"`). For ASIC, replace with foundry-provided SRAM compiler macros.

**Change:** Replace the body of `bram_sp.sv` with an SRAM macro wrapper:

```systemverilog
// FPGA version (current):
(* ram_style = "block" *) logic [DATA_WIDTH-1:0] mem [0:DEPTH-1];

// ASIC version:
// Instantiate TSMC 12nm SRAM compiler output
tsmc12_sp_sram #(.WORDS(DEPTH), .BITS(DATA_WIDTH)) u_sram (...);
```

Use `ifdef SYNTHESIS_ASIC` / `ifdef SYNTHESIS_FPGA` guards so both versions coexist.

### 8b. MMCM → PLL

The `fpga_top.sv` wrapper instantiates a Xilinx `MMCME2_BASE` primitive for clock generation. For ASIC, replace with the foundry PLL macro. For Dragon Ball's FPGA prototype, no change needed — the MMCM is correct for Kintex-7.

### 8c. Clock Tree

FPGA uses global clock buffers (`BUFG`). ASIC needs a proper clock tree synthesized by the place-and-route tool. No RTL change — this is handled by the ASIC backend flow. Dragon Ball's ASIC team knows this.

### 8d. Reset Synchronizer

The `fpga_top.sv` wrapper has a 4-stage reset synchronizer. This is good practice for both FPGA and ASIC. No change needed, but the ASIC backend should verify the reset tree meets timing for all flip-flops.

---

## 9. Xlattice Memory Interface (For VDF Lane — Lower Priority)

### Problem

In `qug_tile.sv` lines 150-181, the Xlattice 256-bit SRAM interface is stubbed:

```systemverilog
assign xl_mem_rd_data_a = 256'd0;
assign xl_mem_rd_data_b = 256'd0;
```

The `xlattice_unit.sv` reads operands from SRAM addresses (rs1, rs2), but gets zeros. Modular multiplication of `0 * 0 = 0`. The VDF lane produces no useful output.

### Fix

Add a 256-bit-wide scratchpad (similar to Section 1). The VDF workload stores intermediate Jacobian coordinates (X, Y, Z — three 256-bit values = 96 bytes) and reads two operands per multiplication.

**Minimum scratchpad size:** 16 x 256-bit words = 512 bytes. Stores all VDF working registers (3 Jacobian coordinates + temporaries).

```systemverilog
module xlattice_scratchpad (
    input  logic         clk,
    input  logic         rst_n,
    // Read port A (256-bit, 1-cycle latency)
    input  logic [3:0]   rd_addr_a,
    input  logic         rd_en_a,
    output logic [255:0] rd_data_a,
    output logic         rd_valid_a,
    // Read port B (256-bit, 1-cycle latency)
    input  logic [3:0]   rd_addr_b,
    input  logic         rd_en_b,
    output logic [255:0] rd_data_b,
    output logic         rd_valid_b,
    // Write port (256-bit)
    input  logic [3:0]   wr_addr,
    input  logic         wr_en,
    input  logic [255:0] wr_data
);
    logic [255:0] mem [0:15];
    // ... standard dual-read, single-write register file
endmodule
```

**However:** Dragon Ball should NOT implement the VDF lane in their ASIC. The ASIC advantage over GPUs for VDF is only ~2-5x (clock speed), not the 100x+ for BLAKE3. This fix is only needed if the QUG-V1 card is intended to also run a full node with VDF verification capability.

**Recommendation:** Skip for ASIC. Fix only if Quillon decides the QUG-V1 card must verify VDF proofs (full-node mode).

---

## 10. NTT Stubs (Phase 1B — Not Needed for Mining)

### Problem

`xlattice_unit.sv` lines 93-97 define NTT instructions as stubs:

```systemverilog
localparam logic [6:0] F7_NTT_FWD = 7'd0;   // Forward NTT (stub)
localparam logic [6:0] F7_NTT_INV = 7'd1;    // Inverse NTT (stub)
localparam logic [6:0] F7_POLY_RED = 7'd4;   // Barrett reduction (stub)
```

These return zero immediately (line 189: `fsm_next = S_STUB_RESPOND`).

### Fix

NTT is for post-quantum lattice cryptography (Dilithium5 signature verification, Kyber1024 key exchange). It's used for full-node block validation, not mining.

**For Dragon Ball's mining ASIC:** No fix needed. Leave as stubs.

**For the QUG-V1 programmable SoC (full-node card):** Implement NTT butterfly unit in Phase 1B. Estimated 256-point NTT over Z_q: ~512 cycles using the existing mod_mul_256 for coefficient multiplies. Resource: ~4,000 LUTs + 16 DSPs for the butterfly array. This is future work and not blocking for the ASIC mining product.

---

## 11. FPGA Target Clarification: XC7K355T vs XC7A355T

### Problem

Dragon Ball mentioned two different FPGA parts in Discord:
- "we use **XC7K355T**" (Kintex-7, FFG901 package)
- "we have **Artix-7 XC7A355T** for ALPH" (Artix-7, FFG901 package)

These share the same package (FFG901) but have very different resources:

| Resource | XC7K355T (Kintex-7) | XC7A355T (Artix-7) |
|----------|--------------------|--------------------|
| LUTs | 226,800 | 224,080 |
| DSP48E1 | **1,440** | **740** |
| Block RAM (36Kb) | **445** | **303** |
| Speed grade | -2 (faster) | -2 (slower fabric) |

For a single tile (~20,700 LUTs, 12 DSPs, 34 BRAMs), either chip works.

For multi-tile scaling:
- Kintex-7: 10 tiles (120 DSPs, 340 BRAMs) — comfortable
- Artix-7: 6 tiles max (72 DSPs, but only 303 BRAMs limits to 8 tiles)

### Fix

We created `fpga/constraints/kintex7_355t.xdc` for the XC7K355T (FFG901). If Dragon Ball is actually using the Artix-7 XC7A355T, we need:

1. A new constraint file `fpga/constraints/artix7_355t.xdc` — same FFG901 ball map but different internal routing and speed constraints
2. Update `synth_vivado.tcl` to accept `xc7a355tffg901-2` as a part option
3. **Reduce target clock** from 100 MHz to 75-80 MHz — Artix-7 has slower fabric than Kintex-7

**Action needed from Dragon Ball:** Confirm which chip is on their board. If Artix-7, provide the board schematic so we can map the correct pins.

---

## 12. VCS Simulation Compatibility

### Problem

Dragon Ball uses Synopsys VCS. Our testbenches were written for Vivado Simulator / Verilator. Some constructs may not compile in VCS without flags.

### Potential issues

1. **`always_comb` with function calls** — VCS is stricter about automatic functions in `always_comb` sensitivity lists. Our `quarter_round` function in `blake3_round.sv` (line 116) is declared `automatic` which is correct for VCS.

2. **Unpacked array ports** — `logic [31:0] state_in [0:15]` (unpacked arrays as module ports). VCS supports this in SV-2012 mode. Compile with `-sverilog` flag.

3. **`genvar` in `for` loops** — `for (genvar i = ...)` inside `generate` blocks. VCS requires `genvar` to be declared outside the `for` if using an older VCS version. Our code declares `genvar` inline which requires VCS 2019.06+.

4. **Package imports** — Our modules use `import qug_pkg::*` at the module level (after the `module` keyword). VCS requires this to be either inside the port list or the module uses `import` before any declarations. Our style is correct.

### Fix

Add a VCS compile script `sim/Makefile.vcs`:

```makefile
VCS = vcs
VCS_FLAGS = -sverilog -full64 +v2k -timescale=1ns/1ps \
            -assert svaext \
            +incdir+../rtl/pkg

RTL_SRCS = ../rtl/pkg/qug_pkg.sv \
           ../rtl/pkg/xcrypto_pkg.sv \
           ../rtl/core/qug_core_pkg.sv \
           $(wildcard ../rtl/core/*.sv) \
           $(wildcard ../rtl/memory/*.sv) \
           $(wildcard ../rtl/xcrypto/*.sv) \
           $(wildcard ../rtl/xlattice/*.sv) \
           $(wildcard ../rtl/top/*.sv)

TB_SRCS = ../tb/blake3_tb.sv \
          ../tb/core_tb.sv \
          ../tb/xlattice_tb.sv

all: simv
	./simv

simv: $(RTL_SRCS) $(TB_SRCS)
	$(VCS) $(VCS_FLAGS) -o simv $(RTL_SRCS) $(TB_SRCS)

clean:
	rm -rf simv simv.daidir csrc *.log *.vpd
```

**Important:** Package files (`qug_pkg.sv`, `xcrypto_pkg.sv`, `qug_core_pkg.sv`) must be compiled first. The order in `RTL_SRCS` above is correct.

---

## 13. File-by-File Change Summary

| File | Changes Needed | Priority | Est. Lines |
|------|---------------|----------|-----------|
| `rtl/memory/xcrypto_scratchpad.sv` | **NEW FILE** — 512-bit scratchpad | P0 | ~50 |
| `rtl/top/qug_tile.sv` | Replace zero-fill stub with scratchpad instance | P0 | ~20 |
| `rtl/mining/mining_controller.sv` | **NEW FILE** — nonce gen + comparator + best-hash | P0 | ~250 |
| `rtl/top/qug_soc_top.sv` | Instantiate mining_controller, wire to Xcrypto | P0 | ~40 |
| `rtl/xcrypto/xcrypto_unit.sv` | Merge S_CHAIN_WRITEBACK + S_COMPRESS | P2 | ~30 (net delete) |
| `rtl/periph/spi_slave.sv` | **NEW FILE** — SPI slave + register map | P2 | ~300 |
| `rtl/memory/xlattice_scratchpad.sv` | **NEW FILE** — 256-bit scratchpad (VDF) | P3 | ~60 |
| `sim/Makefile.vcs` | **NEW FILE** — VCS compile script | P2 | ~25 |
| `fpga/constraints/artix7_355t.xdc` | **NEW FILE** (if Artix-7 confirmed) | P1 | ~120 |
| `fpga/scripts/synth_vivado.tcl` | Add xc7a355t part support | P1 | ~5 |
| `rtl/pkg/qug_pkg.sv` | Add mining_controller params | P0 | ~10 |

**Total new code:** ~910 lines across 5 new files + ~105 lines of modifications to 5 existing files.

**Total after changes:** ~7,800 lines (was 6,779).

---

## 14. Recommended Implementation Order

```
Week 1:  P0 — Xcrypto scratchpad + mining controller + target comparator
         These make the design functionally capable of mining.

Week 2:  P1 — FPGA target clarification (wait for Dragon Ball's board info)
         Best-hash tracker (part of mining controller)
         Variable difficulty target wiring

Week 3:  P2 — Chain loop optimization (14.3% hashrate boost)
         VCS simulation script
         SPI host interface

Week 4+: P3 — Multi-tile memory (only needed for ASIC, not FPGA prototype)
         ASIC-specific replacements (after FPGA validation)
```

---

## 15. What Doesn't Need Changing

These modules are correct and complete for both FPGA and ASIC:

| Module | Why it's fine |
|--------|--------------|
| `blake3_round.sv` | SIGMA tables verified, 2-stage pipeline meets 100 MHz |
| `blake3_pipeline.sv` | 14-stage pipeline, CV delay chain correct, finalization XOR correct |
| `blake3_state.sv` | All 6 operations implemented, bulk write priority correct |
| `mod_mul_256.sv` | Double-fold Barrett reduction correct, DSP inference verified |
| `mod_add_256.sv` | Single-cycle combinational add with conditional subtract |
| `mod_inv_256.sv` | Fermat's method, correct exponent (p-2), ~6084 cycles |
| `qug_pipeline.sv` | 5-way forwarding, load-use detection, ext_stall handshake |
| `qug_decoder.sv` | Full RV32IMC decode (not compliance-tested but structurally correct) |
| `qug_alu.sv` | Standard ALU, all M-extension ops |
| `qug_regfile.sv` | 32x32-bit, x0 hardwired to zero |
| `mem_subsystem.sv` | Address decode, BRAM + UART mux, single-cycle response |
| `fpga_top.sv` | MMCM + reset sync + LED passthrough |
| `xcrypto_pkg.sv` | BLAKE3 IV constants match NIST reference |

**Do not modify these files** unless Dragon Ball's synthesis reveals a timing violation.

---

## 16. Questions for Dragon Ball (Updated)

1. **Which FPGA chip?** XC7K355T (Kintex-7) or XC7A355T (Artix-7)?  We need this to deliver the correct constraint file.

2. **Board schematic?** We need pin assignments for: clock oscillator, reset, UART TX/RX, LEDs, SPI (if available).

3. **VCS version?** Inline `genvar` requires VCS 2019.06+. If older, we can refactor.

4. **Host MCU on the board?** Does the Dragon Ball FPGA board have an onboard MCU (STM32, etc.) for SPI communication, or is everything via UART/USB?

5. **ASIC process confirmed?** If Dragon Ball agrees on 12nm, we can start adding `ifdef SYNTHESIS_ASIC` guards to the BRAM modules now.

6. **Multi-tile target?** How many mining cores does Dragon Ball want on the ASIC die? This determines memory architecture (private SRAM vs crossbar).

---

*v4 is a code-change roadmap, not a delivery. The v3 RTL is architecturally correct and ready for Dragon Ball's synthesis. The changes described here make it functionally capable of mining on Quillon mainnet. Implementation begins upon Dragon Ball's confirmation of FPGA target and board pinout.*
