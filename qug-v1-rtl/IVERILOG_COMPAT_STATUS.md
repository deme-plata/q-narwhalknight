# iverilog 11 Compatibility Fix — Status & Next Steps

## What we're doing

Making `make blake3` pass in `qug-v1-rtl/sim/`. The RTL is a 14-stage pipelined
BLAKE3 compression unit for the QUG-V1 FPGA mining SoC. Tests run under iverilog 11
which has multiple known limitations we've been working around.

## What was already fixed (this session)

All fixes applied and committed to git:

| File | Fix |
|------|-----|
| `rtl/xcrypto/blake3_round.sv` | Replaced `sigma[round_idx][i]` array with case statement; moved `quarter_round` calls to `assign` statements (not `always_comb`); replaced `always_comb` state_out with 16 individual `assign` statements |
| `rtl/xcrypto/blake3_state.sv` | `localparam` IV array → `reg+initial`; `always_comb` state_out → 16 individual `assign` statements |
| `rtl/xcrypto/blake3_pipeline.sv` | Unrolled 7-round generate loop into explicit instantiations; replaced 2D msg_pipe with 12 explicit 1D delay arrays; `localparam` IV → `reg+initial`; `always_comb` init_state → 16 individual `assign` statements; `(* dont_touch *)` before generate commented out |
| `rtl/xcrypto/xcrypto_unit.sv` | `localparam` BLAKE3_IV array → `reg+initial`; added `lat_rs1_lo4` wire for `lat_rs1[3:0]` used in `always_comb`; inlined BLAKE3_IV constants directly in `always_comb` pipe_cv driver |
| `rtl/pkg/qug_pkg.sv` | `localparam int unsigned` → `localparam int` |
| `rtl/pkg/xcrypto_pkg.sv` | `localparam int unsigned` → `localparam int` |
| `tb/blake3_tb.sv` | Full rewrite: `reg+initial` for IV; tasks instead of functions for G/compress_ref; all module-level buffers for array args; removed `return` from tasks; added `xc_hash_words_out` port |

Compilation is now **clean** (only harmless "empty port list" warnings for no-arg tasks).

## Current failure — ROOT CAUSE IDENTIFIED

**iverilog 11 does not propagate unpacked-array OUTPUT port connections.**

When a module has `output logic [31:0] state_out [0:15]` driven by assign statements
inside, and the parent connects `.state_out(state_r0)`, the parent wire `state_r0`
stays X — it never receives the driven values.

This means ALL 7 inter-round state wires (`state_r0..state_r6`) in `blake3_pipeline`
are permanently X. Each round receives X as its `state_in`, computes X results, writes
X into `s2_state`. Final `hash_out` = X.

Evidence:
- `pipe_out_valid` fires correctly (valid chain works — valid_rN are scalar, not arrays)
- Hierarchical reference `u_r6.s2_state[0]` is ALSO X (confirming s2_state itself is X
  from X inputs, not just a broken output port read)
- cv0 (chaining_value[0]) correctly reads `6a09e667` in the debug output, confirming
  the scalar INPUT port works fine

**Also suspected: input unpacked-array ports may also be unreliable** (msg[0:15]
inside blake3_round may read as X even when msg_s0 is 0). This hasn't been confirmed
separately since the state_in issue already causes X propagation.

## The fix needed

**Convert all inter-module unpacked-array state/message connections to packed vectors.**

Replace:
```systemverilog
input  logic [31:0] state_in  [0:15],   // broken in iverilog 11
output logic [31:0] state_out [0:15],   // broken in iverilog 11
input  logic [31:0] msg       [0:15],   // may be broken
```

With:
```systemverilog
input  logic [511:0] state_in,          // 16×32 packed — works fine
output logic [511:0] state_out,         // packed
input  logic [511:0] msg_packed,        // packed
```

### Files to change

#### `rtl/xcrypto/blake3_round.sv`
- Change `state_in`, `state_out`, `msg` ports to packed `[511:0]`
- Update all internal references:
  - `state_in[N]` → `state_in[N*32 +: 32]`  (N = 0..15)
  - `msg[N]` → `msg_packed[N*32 +: 32]`
  - `state_out` driven by: `assign state_out = {s2_state[15], s2_state[14], ..., s2_state[0]}`
    (word 0 at [31:0], word 15 at [511:480])
  - NOTE: `state_in[N*32 +: 32]` part-selects — may hit iverilog "constant select not supported" in always_comb. Workaround: copy to local logic array at top of module:
    ```
    logic [31:0] s_in [0:15];
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) for(j=0;j<16;j++) s_in[j] <= 0;
        else if (in_valid) for(j=0;j<16;j++) s_in[j] <= state_in[j*32 +: 32];
    end
    ```
    Use `s_in` internally instead of `state_in[N]`.
  - Similarly copy `msg_packed` to local `m_in[0:15]` at the stage-1 latch.

#### `rtl/xcrypto/blake3_pipeline.sv`
- Change inter-round wires to packed: `logic [511:0] state_r0..state_r6`
- Change msg delay chain to packed: `logic [511:0] msg_d1..msg_d12`
- Keep msg_s0 as `logic [31:0] msg_s0[0:15]` internally, pack it as:
  `logic [511:0] msg_s0_packed;`
  `assign msg_s0_packed[j*32 +: 32] = msg_s0[j];`  (16 assigns)
- Pack state_s0 similarly.
- Valid signals stay as scalar (they work fine).
- Output: `assign hash_out[i] = state_r6[(i)*32 +: 32] ^ state_r6[(i+8)*32 +: 32]`
  → revert to individual word assigns (no part-selects in always_comb).

#### `rtl/xcrypto/xcrypto_unit.sv`
- `pipe_cv [0:7]` and `pipe_block [0:15]` ports to blake3_pipeline → pack these too
- The always_comb that drives pipe_cv/pipe_block currently uses for-loops reading
  arrays — these will need to become per-element assigns or packed vector assigns.
- The hash_latched and hash_words_for_lzc arrays in always_comb — same issue.
- `pipe_hash_out` (from blake3_pipeline output) → will be packed after change.

#### `tb/blake3_tb.sv`
- `pipe_cv`, `pipe_block`, `pipe_hash_out` → pack to match new blake3_pipeline ports
- Setup loops in test tasks → individual word assignments

### Word packing convention (consistent across all files)
- Word 0 at bits [31:0] (LSW)
- Word N at bits [N*32+31:N*32]
- Word 15 at bits [511:480]

Example: `state_packed[0*32 +: 32]` = word 0 = `state[0]`

## Iverilog 11 limitations catalogue (comprehensive)

1. `localparam logic [W:0] arr [N]` in module/package scope → `reg + initial`
2. `(expression)[N:M]` part-select of expression result in functions → use tmp var
3. `inout`/`output` function arguments → use tasks
4. Unpacked array subroutine ports → module-level buffers
5. Dynamic 2D array indexing `arr[var][i]` in `always_*` → case statement
6. Function calls + part-selects on return value in `always_comb` → use `assign`
7. `(* dont_touch = "true" *)` before `generate` → comment out
8. Genvar-indexed 2D unpacked array slice as module port arg → unroll generate loop
9. `return` in tasks → restructure with flag variable
10. `always_comb` for-loop doesn't track array elements in sensitivity list → individual `assign`
11. **[NEW/ROOT CAUSE]** Unpacked-array output port connections don't propagate changes to parent wires → convert to packed vectors

## Commit state

All changes from #1-#10 are uncommitted (modified files). After fixing #11 and
getting all 4 tests to pass, commit everything as:
`feat(rtl): fix all iverilog 11 compatibility issues in BLAKE3 pipeline`

## Quick restart checklist

1. `cd /opt/orobit/shared/q-narwhalknight/qug-v1-rtl/sim && make blake3`
   → Should show compile OK + 4 tests failing with X (confirms current state)
2. Fix blake3_round.sv: pack state_in/state_out/msg ports → verify with single round
3. Fix blake3_pipeline.sv: pack all inter-round wires
4. Fix xcrypto_unit.sv: pack pipe_cv/pipe_block/pipe_hash_out
5. Fix tb/blake3_tb.sv: update testbench connections
6. `make blake3` → all 4 tests should PASS
7. Git commit
