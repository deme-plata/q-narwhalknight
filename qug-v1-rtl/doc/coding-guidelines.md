# QUG-V1 RTL -- SystemVerilog Coding Guidelines

This document defines the mandatory coding standards for all RTL contributed
to the QUG-V1 Mining SoC project. Consistency across the codebase is critical
for synthesis reliability, simulation reproducibility, and team collaboration.

## 1. Language and Tool Requirements

- **Language**: SystemVerilog (IEEE 1800-2017)
- **Synthesis**: Xilinx Vivado 2023.2+ (Kintex-7 target)
- **Simulation**: Verilator 5.006+ (lint and cycle-accurate)
- **Formal**: Jasper or SymbiYosys (for SVA property verification)

## 2. File Organization

### One Module Per File

Each `.sv` file contains exactly one module (or one package/interface).
The filename must match the module name:

```
  qug_core.sv        -->  module qug_core (...);
  xcrypto_engine.sv   -->  module xcrypto_engine (...);
  xlattice_ntt.sv     -->  module xlattice_ntt (...);
```

### File Header

Every file begins with a standard header comment:

```systemverilog
// =============================================================================
// Module : <module_name>
// Project: QUG-V1 Mining SoC
// Author : <author or team>
// License: MIT
// =============================================================================
// Description:
//   <Brief description of what the module does.>
// =============================================================================
```

### Include Guards for Packages

Packages do not need include guards (SystemVerilog handles duplicate imports),
but each package file should have a clear `package ... endpackage` scope with
a label:

```systemverilog
package qug_pkg;
  // ...
endpackage : qug_pkg
```

## 3. Type System

### Use `logic`, Never `reg` or `wire`

The `logic` type is the universal 4-state type in SystemVerilog. It works in
both procedural and continuous contexts. Never use `reg` (legacy Verilog) or
bare `wire` (except in port declarations where required by tool).

```systemverilog
// CORRECT
logic [31:0] data_out;
logic        valid;

// WRONG
reg [31:0] data_out;    // Do not use reg
wire       valid;       // Do not use wire for internal signals
```

### Port Declarations

Use ANSI-style port declarations. Group ports by function with comments:

```systemverilog
module example_module
  import qug_pkg::*;
#(
  parameter int unsigned DATA_W = 32
) (
  // Clock and reset
  input  logic              clk,
  input  logic              rst_n,      // Active-low async reset

  // Data input
  input  logic [DATA_W-1:0] din,
  input  logic              din_valid,
  output logic              din_ready,

  // Data output
  output logic [DATA_W-1:0] dout,
  output logic              dout_valid,
  input  logic              dout_ready
);
```

### Typedef and Struct Usage

Use `typedef` for all custom types. Prefer `struct packed` for synthesizable
register bundles:

```systemverilog
typedef struct packed {
  logic [31:0] addr;
  logic [31:0] data;
  logic        valid;
} request_t;
```

## 4. Sequential and Combinational Logic

### `always_ff` for Registers

All flip-flop inference must use `always_ff`. Include the active-low async
reset in the sensitivity list:

```systemverilog
always_ff @(posedge clk or negedge rst_n) begin : ff_counter
  if (!rst_n) begin
    count_q <= '0;
  end else begin
    count_q <= count_d;
  end
end : ff_counter
```

### `always_comb` for Combinational Logic

All combinational logic must use `always_comb`. Never use `always @(*)`:

```systemverilog
always_comb begin : comb_next_state
  count_d = count_q;
  if (enable) begin
    count_d = count_q + 1'b1;
  end
end : comb_next_state
```

### Never Use `always_latch`

Latches are forbidden in this design. If a tool infers a latch, it is a bug.
Ensure every signal assigned in `always_comb` has a default value.

## 5. Naming Conventions

### Signals and Variables: `snake_case`

```systemverilog
logic [31:0] write_data;
logic        fifo_empty;
logic [7:0]  byte_count_q;   // _q suffix for registered version
logic [7:0]  byte_count_d;   // _d suffix for next-state (combinational)
```

### Parameters and Constants: `UPPER_SNAKE_CASE`

```systemverilog
localparam int unsigned FIFO_DEPTH  = 16;
localparam int unsigned ADDR_WIDTH  = 32;
parameter  int unsigned NUM_ENTRIES = 8;
```

### Modules and Packages: `snake_case`

```
qug_core, xcrypto_engine, xlattice_ntt, qug_pkg
```

### Interfaces: `snake_case` with `_if` suffix

```systemverilog
interface axi4l_if #(parameter int ADDR_W = 32, parameter int DATA_W = 32);
  // ...
endinterface : axi4l_if
```

### Signal Suffixes

| Suffix | Meaning                               |
|--------|---------------------------------------|
| `_q`   | Registered (flip-flop output)         |
| `_d`   | Next-state (combinational, drives _q) |
| `_n`   | Active-low                            |
| `_en`  | Enable                                |
| `_we`  | Write enable                          |
| `_re`  | Read enable                           |
| `_sel` | Select / mux control                  |

## 6. Reset Convention

- **Active-low asynchronous reset**: `rst_n`
- All flip-flops must be resettable
- Reset value is `'0` unless a specific non-zero reset is required
  (document the reason in a comment)
- Use the same reset signal name (`rst_n`) throughout the hierarchy.
  Domain-specific resets use a prefix: `rst_ddr_n`, `rst_jtag_n`

```systemverilog
always_ff @(posedge clk or negedge rst_n) begin : ff_state
  if (!rst_n) begin
    state_q <= IDLE;
  end else begin
    state_q <= state_d;
  end
end : ff_state
```

## 7. Parameters

- Declare parameters at the top of the module, before ports
- Use `parameter` for values configurable by instantiation
- Use `localparam` for derived or internal-only constants
- Always specify the type (`int unsigned`, `logic [N:0]`, etc.)

```systemverilog
module fifo
#(
  parameter  int unsigned DEPTH    = 16,
  parameter  int unsigned DATA_W   = 32,
  localparam int unsigned ADDR_W   = $clog2(DEPTH)
) (
  // ...
);
```

## 8. Named Begin/End Blocks

Every `begin/end` block must have a label. This improves waveform debugging
and lint clarity:

```systemverilog
always_ff @(posedge clk or negedge rst_n) begin : ff_output
  if (!rst_n) begin : ff_output_reset
    data_q <= '0;
  end : ff_output_reset
  else begin : ff_output_update
    data_q <= data_d;
  end : ff_output_update
end : ff_output
```

For short blocks (single assignment), the label on inner begin/end may be
omitted, but the outer block label is mandatory.

## 9. Assertions (SVA)

### Immediate Assertions

Use for simple sanity checks in simulation:

```systemverilog
// Verify FIFO never overflows
always_ff @(posedge clk) begin : check_no_overflow
  if (rst_n) begin
    assert (!wr_en || !full)
      else $error("%m: FIFO write while full at time %0t", $time);
  end
end : check_no_overflow
```

### Concurrent Assertions

Use `property` and `assert property` for temporal checks:

```systemverilog
// AXI handshake: valid must not deassert before ready
property p_valid_stable;
  @(posedge clk) disable iff (!rst_n)
    (valid && !ready) |=> valid;
endproperty : p_valid_stable

assert property (p_valid_stable)
  else $error("%m: valid deasserted before ready handshake");
```

### Assertion Naming

Prefix assertion labels with `a_` and property labels with `p_`:

```systemverilog
a_no_overflow : assert property (p_no_overflow);
```

### Synthesis Guards

Wrap assertions in synthesis-off pragmas so they do not affect synthesis:

```systemverilog
`ifndef SYNTHESIS
  a_fifo_check : assert property (p_no_overflow)
    else $fatal(1, "FIFO overflow detected");
`endif
```

## 10. Clock Domain Crossings

- All CDC crossings must use a dedicated synchronizer module
  (`cdc_sync_2ff.sv` for single-bit, `cdc_async_fifo.sv` for buses)
- Never connect signals across clock domains without a synchronizer
- Document every CDC path with a comment: `// CDC: clk_a -> clk_b`
- Use Vivado `set_false_path` or `set_max_delay` constraints for CDC paths

## 11. Synthesis Directives

Use Xilinx-compatible attributes sparingly and only when necessary:

```systemverilog
(* dont_touch = "true" *)  logic sync_ff1, sync_ff2;   // CDC synchronizer
(* ram_style = "block" *)  logic [31:0] mem [0:1023];   // Force BRAM
(* use_dsp = "yes" *)      logic [63:0] product;        // Force DSP48
```

## 12. Prohibited Constructs

| Construct            | Reason                                          | Alternative             |
|---------------------|-------------------------------------------------|-------------------------|
| `reg`               | Legacy Verilog                                  | `logic`                 |
| `wire` (internal)   | Unnecessary in SystemVerilog                    | `logic`                 |
| `always @(*)`       | Ambiguous sensitivity                           | `always_comb`           |
| `always_latch`      | Latches are forbidden                           | Fix the combinational logic |
| `initial` (synth)   | Not synthesizable                               | Use reset               |
| `#delay`            | Not synthesizable                               | Testbench only          |
| `force/release`     | Dangerous, non-portable                         | Use proper test harness |
| Tri-state (`inout`) | Not available on FPGA fabric                    | Bidirectional mux       |
| `casex`             | Treats X/Z as don't-care (dangerous)            | `casez` or `case inside`|

## 13. Code Review Checklist

Before submitting RTL for review, verify:

- [ ] One module per file, filename matches module name
- [ ] File header with description present
- [ ] All `always_ff` blocks have `rst_n` in sensitivity list
- [ ] All `always_comb` blocks have default assignments (no latches)
- [ ] All `begin/end` blocks are labeled
- [ ] No `reg`, `wire`, `always @(*)` usage
- [ ] Parameters at top of module with explicit types
- [ ] Signal names follow `snake_case` convention
- [ ] Constants follow `UPPER_SNAKE_CASE` convention
- [ ] Active-low reset named `rst_n`
- [ ] Assertions present for critical invariants
- [ ] No lint warnings from Verilator `--lint-only`
- [ ] CDC crossings use synchronizer modules
