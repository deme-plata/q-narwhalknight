// =============================================================================
// xlattice_tb.sv -- Testbench for Xlattice 256-bit Modular Arithmetic
// QUG-V1 Mining SoC -- Genus-2 VDF Field Arithmetic Verification
// =============================================================================
// Project  : QUG-V1 RISC-V Mining SoC
// Target   : Simulation (Verilator / Vivado XSIM / ModelSim)
// Author   : Quillon Foundation / Dragon Ball Miner
// License  : MIT
// =============================================================================
//
// Test cases:
//   1. mod_add_256: Basic addition, addition wrapping around p, edge cases
//   2. mod_mul_256: Small values, large values near p, multiply by 0/1
//   3. mod_inv_256: Verify a * a^(-1) = 1 (mod p) for a = 7
//   4. xlattice_unit: poly.add and poly.mul instruction decode and execution
//
// All tests use p = 2^255 - 19 (Curve25519 prime).
// =============================================================================

`timescale 1ns / 1ps

module xlattice_tb;

    // =========================================================================
    // Parameters
    // =========================================================================
    localparam logic [255:0] P = 256'h7FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFED;
    localparam CLK_PERIOD = 10;  // 100 MHz

    // =========================================================================
    // Signals
    // =========================================================================
    logic        clk;
    logic        rst_n;

    // --- mod_add_256 ---
    logic        add_start;
    logic [255:0] add_a, add_b;
    logic [255:0] add_modulus;
    logic [255:0] add_result;
    logic        add_done;

    // --- mod_mul_256 ---
    logic        mul_start;
    logic [255:0] mul_a, mul_b;
    logic [255:0] mul_modulus;
    logic [255:0] mul_result;
    logic        mul_done;

    // --- mod_inv_256 ---
    logic        inv_start;
    logic [255:0] inv_a;
    logic [255:0] inv_modulus;
    logic [255:0] inv_result;
    logic        inv_done;

    // --- xlattice_unit ---
    logic        xl_req_valid;
    logic        xl_req_ready;
    logic [6:0]  xl_funct7;
    logic [2:0]  xl_funct3;
    logic [31:0] xl_rs1, xl_rs2;
    logic [4:0]  xl_rd_addr;
    logic        xl_resp_valid;
    logic [4:0]  xl_resp_rd_addr;
    logic [31:0] xl_resp_data;
    logic        xl_resp_wr_en;

    // SRAM interface
    logic [31:0]  xl_mem_rd_addr_a, xl_mem_rd_addr_b;
    logic         xl_mem_rd_en_a, xl_mem_rd_en_b;
    logic [255:0] xl_mem_rd_data_a, xl_mem_rd_data_b;
    logic         xl_mem_rd_valid_a, xl_mem_rd_valid_b;
    logic [31:0]  xl_mem_wr_addr;
    logic         xl_mem_wr_en;
    logic [255:0] xl_mem_wr_data;

    // Test counters
    int pass_count = 0;
    int fail_count = 0;
    int test_num   = 0;

    // =========================================================================
    // Clock generation
    // =========================================================================
    initial clk = 1'b0;
    always #(CLK_PERIOD / 2) clk = ~clk;

    // =========================================================================
    // DUT instantiation: mod_add_256
    // =========================================================================
    mod_add_256 u_add (
        .clk     (clk),
        .rst_n   (rst_n),
        .start   (add_start),
        .op_a    (add_a),
        .op_b    (add_b),
        .modulus (add_modulus),
        .result  (add_result),
        .done    (add_done)
    );

    // =========================================================================
    // DUT instantiation: mod_mul_256
    // =========================================================================
    mod_mul_256 u_mul (
        .clk     (clk),
        .rst_n   (rst_n),
        .start   (mul_start),
        .op_a    (mul_a),
        .op_b    (mul_b),
        .modulus (mul_modulus),
        .result  (mul_result),
        .done    (mul_done)
    );

    // =========================================================================
    // DUT instantiation: mod_inv_256
    // =========================================================================
    mod_inv_256 u_inv (
        .clk     (clk),
        .rst_n   (rst_n),
        .start   (inv_start),
        .op_a    (inv_a),
        .modulus (inv_modulus),
        .result  (inv_result),
        .done    (inv_done)
    );

    // =========================================================================
    // DUT instantiation: xlattice_unit
    // =========================================================================
    xlattice_unit u_xlattice (
        .clk            (clk),
        .rst_n          (rst_n),
        .req_valid      (xl_req_valid),
        .req_ready      (xl_req_ready),
        .req_funct7     (xl_funct7),
        .req_funct3     (xl_funct3),
        .req_rs1        (xl_rs1),
        .req_rs2        (xl_rs2),
        .req_rd_addr    (xl_rd_addr),
        .resp_valid     (xl_resp_valid),
        .resp_rd_addr   (xl_resp_rd_addr),
        .resp_data      (xl_resp_data),
        .resp_wr_en     (xl_resp_wr_en),
        .mem_rd_addr_a  (xl_mem_rd_addr_a),
        .mem_rd_en_a    (xl_mem_rd_en_a),
        .mem_rd_data_a  (xl_mem_rd_data_a),
        .mem_rd_valid_a (xl_mem_rd_valid_a),
        .mem_rd_addr_b  (xl_mem_rd_addr_b),
        .mem_rd_en_b    (xl_mem_rd_en_b),
        .mem_rd_data_b  (xl_mem_rd_data_b),
        .mem_rd_valid_b (xl_mem_rd_valid_b),
        .mem_wr_addr    (xl_mem_wr_addr),
        .mem_wr_en      (xl_mem_wr_en),
        .mem_wr_data    (xl_mem_wr_data)
    );

    // =========================================================================
    // Simple SRAM model for xlattice_unit
    // =========================================================================
    // 4 entries of 256 bits, addressed by word address (0, 32, 64, 96)
    logic [255:0] sram [0:3];

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            xl_mem_rd_valid_a <= 1'b0;
            xl_mem_rd_valid_b <= 1'b0;
            xl_mem_rd_data_a  <= 256'd0;
            xl_mem_rd_data_b  <= 256'd0;
        end else begin
            // Read port A: 1-cycle latency
            xl_mem_rd_valid_a <= xl_mem_rd_en_a;
            if (xl_mem_rd_en_a) begin
                xl_mem_rd_data_a <= sram[xl_mem_rd_addr_a[6:5]];
            end

            // Read port B: 1-cycle latency
            xl_mem_rd_valid_b <= xl_mem_rd_en_b;
            if (xl_mem_rd_en_b) begin
                xl_mem_rd_data_b <= sram[xl_mem_rd_addr_b[6:5]];
            end

            // Write port
            if (xl_mem_wr_en) begin
                sram[xl_mem_wr_addr[6:5]] <= xl_mem_wr_data;
            end
        end
    end

    // =========================================================================
    // Helper tasks
    // =========================================================================
    task automatic check_result(
        input string test_name,
        input logic [255:0] actual,
        input logic [255:0] expected
    );
        test_num++;
        if (actual === expected) begin
            pass_count++;
            $display("[PASS] Test %0d: %s", test_num, test_name);
        end else begin
            fail_count++;
            $display("[FAIL] Test %0d: %s", test_num, test_name);
            $display("  Expected: 0x%064h", expected);
            $display("  Actual:   0x%064h", actual);
        end
    endtask

    task automatic wait_cycles(input int n);
        repeat(n) @(posedge clk);
    endtask

    // =========================================================================
    // Main test sequence
    // =========================================================================
    initial begin
        // Initialize all inputs
        rst_n        = 1'b0;
        add_start    = 1'b0;
        add_a        = 256'd0;
        add_b        = 256'd0;
        add_modulus  = P;
        mul_start    = 1'b0;
        mul_a        = 256'd0;
        mul_b        = 256'd0;
        mul_modulus  = P;
        inv_start    = 1'b0;
        inv_a        = 256'd0;
        inv_modulus  = P;
        xl_req_valid = 1'b0;
        xl_funct7    = 7'd0;
        xl_funct3    = 3'd0;
        xl_rs1       = 32'd0;
        xl_rs2       = 32'd0;
        xl_rd_addr   = 5'd0;

        // Reset
        wait_cycles(5);
        rst_n = 1'b1;
        wait_cycles(2);

        $display("=============================================================");
        $display("  Xlattice 256-bit Modular Arithmetic Testbench");
        $display("  Prime p = 2^255 - 19 (Curve25519)");
        $display("=============================================================");
        $display("");

        // =====================================================================
        // TEST GROUP 1: mod_add_256
        // =====================================================================
        $display("--- mod_add_256 Tests ---");

        // Test 1.1: Simple addition: 3 + 5 = 8 (mod p)
        add_a = 256'd3;
        add_b = 256'd5;
        add_start = 1'b1;
        @(posedge clk);
        add_start = 1'b0;
        @(posedge clk);
        wait(add_done);
        @(posedge clk);
        check_result("3 + 5 = 8 (mod p)", add_result, 256'd8);

        wait_cycles(2);

        // Test 1.2: Wrap-around: (p - 1) + 1 = 0 (mod p)
        add_a = P - 256'd1;  // p - 1
        add_b = 256'd1;
        add_start = 1'b1;
        @(posedge clk);
        add_start = 1'b0;
        @(posedge clk);
        wait(add_done);
        @(posedge clk);
        check_result("(p-1) + 1 = 0 (mod p)", add_result, 256'd0);

        wait_cycles(2);

        // Test 1.3: Wrap-around: (p - 20) + 1 = (p - 19) (mod p)
        // Wait: (p-20) + 1 = p - 19. Since p - 19 < p, result is p - 19.
        add_a = P - 256'd20;
        add_b = 256'd1;
        add_start = 1'b1;
        @(posedge clk);
        add_start = 1'b0;
        @(posedge clk);
        wait(add_done);
        @(posedge clk);
        check_result("(p-20) + 1 = p-19", add_result, P - 256'd19);

        wait_cycles(2);

        // Test 1.4: Large wrap: (p-1) + (p-1) = p - 2 (mod p)
        // (p-1) + (p-1) = 2p - 2. mod p = p - 2.
        add_a = P - 256'd1;
        add_b = P - 256'd1;
        add_start = 1'b1;
        @(posedge clk);
        add_start = 1'b0;
        @(posedge clk);
        wait(add_done);
        @(posedge clk);
        check_result("(p-1) + (p-1) = p-2 (mod p)", add_result, P - 256'd2);

        wait_cycles(2);

        // Test 1.5: Add zero: a + 0 = a
        add_a = 256'd42;
        add_b = 256'd0;
        add_start = 1'b1;
        @(posedge clk);
        add_start = 1'b0;
        @(posedge clk);
        wait(add_done);
        @(posedge clk);
        check_result("42 + 0 = 42", add_result, 256'd42);

        wait_cycles(2);

        // =====================================================================
        // TEST GROUP 2: mod_mul_256
        // =====================================================================
        $display("");
        $display("--- mod_mul_256 Tests ---");

        // Test 2.1: Simple: 2 * 3 = 6 (mod p)
        mul_a = 256'd2;
        mul_b = 256'd3;
        mul_start = 1'b1;
        @(posedge clk);
        mul_start = 1'b0;
        wait(mul_done);
        @(posedge clk);
        check_result("2 * 3 = 6 (mod p)", mul_result, 256'd6);

        wait_cycles(2);

        // Test 2.2: Multiply by 0
        mul_a = 256'hDEADBEEFDEADBEEFDEADBEEFDEADBEEFDEADBEEFDEADBEEFDEADBEEFDEADBEEF;
        mul_b = 256'd0;
        mul_start = 1'b1;
        @(posedge clk);
        mul_start = 1'b0;
        wait(mul_done);
        @(posedge clk);
        check_result("large * 0 = 0 (mod p)", mul_result, 256'd0);

        wait_cycles(2);

        // Test 2.3: Multiply by 1
        mul_a = 256'd12345678;
        mul_b = 256'd1;
        mul_start = 1'b1;
        @(posedge clk);
        mul_start = 1'b0;
        wait(mul_done);
        @(posedge clk);
        check_result("12345678 * 1 = 12345678 (mod p)", mul_result, 256'd12345678);

        wait_cycles(2);

        // Test 2.4: Small squaring: 7 * 7 = 49 (mod p)
        mul_a = 256'd7;
        mul_b = 256'd7;
        mul_start = 1'b1;
        @(posedge clk);
        mul_start = 1'b0;
        wait(mul_done);
        @(posedge clk);
        check_result("7 * 7 = 49 (mod p)", mul_result, 256'd49);

        wait_cycles(2);

        // Test 2.5: (p-1) * (p-1) = 1 (mod p)
        // Because (p-1) = -1 mod p, so (-1)*(-1) = 1
        mul_a = P - 256'd1;
        mul_b = P - 256'd1;
        mul_start = 1'b1;
        @(posedge clk);
        mul_start = 1'b0;
        wait(mul_done);
        @(posedge clk);
        check_result("(p-1) * (p-1) = 1 (mod p)", mul_result, 256'd1);

        wait_cycles(2);

        // Test 2.6: (p-1) * 2 = p - 2 (mod p)
        // (-1) * 2 = -2 = p - 2
        mul_a = P - 256'd1;
        mul_b = 256'd2;
        mul_start = 1'b1;
        @(posedge clk);
        mul_start = 1'b0;
        wait(mul_done);
        @(posedge clk);
        check_result("(p-1) * 2 = p-2 (mod p)", mul_result, P - 256'd2);

        wait_cycles(2);

        // =====================================================================
        // TEST GROUP 3: mod_inv_256
        // =====================================================================
        $display("");
        $display("--- mod_inv_256 Tests ---");
        $display("  (Inversion takes ~6000 cycles per operation, please wait...)");

        // Test 3.1: inv(7) -- verify 7 * inv(7) = 1 (mod p)
        inv_a = 256'd7;
        inv_start = 1'b1;
        @(posedge clk);
        inv_start = 1'b0;

        // Wait for inversion to complete (up to 100K cycles timeout)
        fork
            begin
                wait(inv_done);
            end
            begin
                wait_cycles(100000);
                $display("[TIMEOUT] mod_inv_256 did not complete within 100K cycles");
            end
        join_any
        disable fork;

        @(posedge clk);

        if (inv_done) begin
            $display("  inv(7) = 0x%064h", inv_result);

            // Verify: 7 * inv(7) should equal 1 (mod p)
            // Use the multiplier to check
            mul_a = 256'd7;
            mul_b = inv_result;
            mul_start = 1'b1;
            @(posedge clk);
            mul_start = 1'b0;
            wait(mul_done);
            @(posedge clk);
            check_result("7 * inv(7) = 1 (mod p)", mul_result, 256'd1);
        end else begin
            test_num++;
            fail_count++;
            $display("[FAIL] Test %0d: inv(7) timed out", test_num);
        end

        wait_cycles(5);

        // Test 3.2: inv(1) should equal 1
        inv_a = 256'd1;
        inv_start = 1'b1;
        @(posedge clk);
        inv_start = 1'b0;

        fork
            begin
                wait(inv_done);
            end
            begin
                wait_cycles(100000);
            end
        join_any
        disable fork;

        @(posedge clk);

        if (inv_done) begin
            check_result("inv(1) = 1", inv_result, 256'd1);
        end else begin
            test_num++;
            fail_count++;
            $display("[FAIL] Test %0d: inv(1) timed out", test_num);
        end

        wait_cycles(5);

        // Test 3.3: inv(p-1) should equal p-1 (since (p-1)^2 = 1 mod p)
        inv_a = P - 256'd1;
        inv_start = 1'b1;
        @(posedge clk);
        inv_start = 1'b0;

        fork
            begin
                wait(inv_done);
            end
            begin
                wait_cycles(100000);
            end
        join_any
        disable fork;

        @(posedge clk);

        if (inv_done) begin
            check_result("inv(p-1) = p-1", inv_result, P - 256'd1);
        end else begin
            test_num++;
            fail_count++;
            $display("[FAIL] Test %0d: inv(p-1) timed out", test_num);
        end

        wait_cycles(5);

        // =====================================================================
        // TEST GROUP 4: xlattice_unit (poly.add and poly.mul instructions)
        // =====================================================================
        $display("");
        $display("--- xlattice_unit Instruction Tests ---");

        // Initialize SRAM
        // Address 0x00 (index 0): operand A = 100
        // Address 0x20 (index 1): operand B = 200
        sram[0] = 256'd100;
        sram[1] = 256'd200;
        sram[2] = 256'd0;
        sram[3] = 256'd0;

        wait_cycles(2);

        // Test 4.1: poly.add (funct7 = 2)
        // 100 + 200 = 300 (mod p)
        xl_funct7    = 7'd2;   // poly.add
        xl_funct3    = 3'd0;
        xl_rs1       = 32'h00; // Address of A (SRAM index 0)
        xl_rs2       = 32'h20; // Address of B (SRAM index 1)
        xl_rd_addr   = 5'd10;
        xl_req_valid = 1'b1;
        @(posedge clk);
        xl_req_valid = 1'b0;

        // Wait for response
        wait(xl_resp_valid);
        @(posedge clk);
        check_result("poly.add: 100 + 200 = 300 (via xlattice)", sram[0], 256'd300);

        wait_cycles(5);

        // Test 4.2: poly.mul (funct7 = 3)
        // Set up: A = 7, B = 6
        sram[0] = 256'd7;
        sram[1] = 256'd6;

        wait_cycles(2);

        xl_funct7    = 7'd3;   // poly.mul
        xl_funct3    = 3'd0;
        xl_rs1       = 32'h00; // Address of A
        xl_rs2       = 32'h20; // Address of B
        xl_rd_addr   = 5'd11;
        xl_req_valid = 1'b1;
        @(posedge clk);
        xl_req_valid = 1'b0;

        // Wait for response
        fork
            begin
                wait(xl_resp_valid);
            end
            begin
                wait_cycles(100);
            end
        join_any
        disable fork;

        @(posedge clk);

        if (xl_resp_valid) begin
            check_result("poly.mul: 7 * 6 = 42 (via xlattice)", sram[0], 256'd42);
        end else begin
            test_num++;
            fail_count++;
            $display("[FAIL] Test %0d: poly.mul timed out", test_num);
        end

        wait_cycles(5);

        // Test 4.3: ntt.fwd stub (funct7 = 0) -- should return immediately
        xl_funct7    = 7'd0;   // ntt.fwd (stub)
        xl_funct3    = 3'd0;
        xl_rs1       = 32'h00;
        xl_rs2       = 32'h20;
        xl_rd_addr   = 5'd12;
        xl_req_valid = 1'b1;
        @(posedge clk);
        xl_req_valid = 1'b0;

        wait(xl_resp_valid);
        @(posedge clk);
        test_num++;
        if (xl_resp_data == 32'd0) begin
            pass_count++;
            $display("[PASS] Test %0d: ntt.fwd stub returns 0", test_num);
        end else begin
            fail_count++;
            $display("[FAIL] Test %0d: ntt.fwd stub returned %0d (expected 0)", test_num, xl_resp_data);
        end

        wait_cycles(5);

        // Test 4.4: ntt.inv stub (funct7 = 1) -- should return immediately
        xl_funct7    = 7'd1;   // ntt.inv (stub)
        xl_funct3    = 3'd0;
        xl_rs1       = 32'h00;
        xl_rs2       = 32'h20;
        xl_rd_addr   = 5'd13;
        xl_req_valid = 1'b1;
        @(posedge clk);
        xl_req_valid = 1'b0;

        wait(xl_resp_valid);
        @(posedge clk);
        test_num++;
        if (xl_resp_data == 32'd0) begin
            pass_count++;
            $display("[PASS] Test %0d: ntt.inv stub returns 0", test_num);
        end else begin
            fail_count++;
            $display("[FAIL] Test %0d: ntt.inv stub returned %0d (expected 0)", test_num, xl_resp_data);
        end

        wait_cycles(5);

        // Test 4.5: poly.reduce stub (funct7 = 4)
        xl_funct7    = 7'd4;   // poly.reduce (stub)
        xl_funct3    = 3'd0;
        xl_rs1       = 32'h00;
        xl_rs2       = 32'h20;
        xl_rd_addr   = 5'd14;
        xl_req_valid = 1'b1;
        @(posedge clk);
        xl_req_valid = 1'b0;

        wait(xl_resp_valid);
        @(posedge clk);
        test_num++;
        if (xl_resp_data == 32'd0) begin
            pass_count++;
            $display("[PASS] Test %0d: poly.reduce stub returns 0", test_num);
        end else begin
            fail_count++;
            $display("[FAIL] Test %0d: poly.reduce stub returned %0d (expected 0)", test_num, xl_resp_data);
        end

        // =====================================================================
        // Summary
        // =====================================================================
        wait_cycles(10);
        $display("");
        $display("=============================================================");
        $display("  Test Summary: %0d passed, %0d failed out of %0d tests",
                 pass_count, fail_count, test_num);
        $display("=============================================================");

        if (fail_count == 0) begin
            $display("  ALL TESTS PASSED");
        end else begin
            $display("  SOME TESTS FAILED -- review output above");
        end

        $display("");
        $finish;
    end

    // =========================================================================
    // Timeout watchdog (prevent simulation from hanging)
    // =========================================================================
    initial begin
        #(CLK_PERIOD * 500000);  // 5ms simulation time
        $display("[WATCHDOG] Global timeout reached -- aborting simulation");
        $finish;
    end

    // =========================================================================
    // Optional waveform dump
    // =========================================================================
    initial begin
        if ($test$plusargs("dump")) begin
            $dumpfile("xlattice_tb.vcd");
            $dumpvars(0, xlattice_tb);
        end
    end

endmodule
