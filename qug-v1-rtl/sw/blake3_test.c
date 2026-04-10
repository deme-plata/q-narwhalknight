/* =============================================================================
 * blake3_test.c -- Bare-Metal BLAKE3 Xcrypto Test
 * QUG-V1 Mining SoC -- RISC-V Firmware
 * =============================================================================
 * Project  : QUG-V1 RISC-V Mining SoC
 * Target   : QUG-V1 RV32IMC + Xcrypto Extension
 * Author   : Quillon Foundation / Dragon Ball Miner
 * License  : MIT
 * =============================================================================
 *
 * Tests the Xcrypto hardware BLAKE3 pipeline by hashing a known test vector
 * and comparing the result against the expected digest.
 *
 * Xcrypto instructions (custom-0 opcode 0x0B, R-type):
 *   blake3.init     rd, rs1       -- funct7=0: Initialize state from IV
 *   blake3.round    rd, rs1, rs2  -- funct7=1: Run 7-round compression
 *   blake3.chain    rd, rs1, rs2  -- funct7=2: Chain hash (VDF), rs2=count
 *   blake3.finalize rd, rs1       -- funct7=3: Read hash word, rs1=index
 *
 * UART is memory-mapped at 0x10000000:
 *   0x10000000: TX data  (write byte to send)
 *   0x10000004: TX status (bit 0 = ready)
 *   0x10000008: RX data  (read received byte)
 *   0x1000000C: RX status (bit 0 = valid)
 *
 * Build:
 *   riscv32-unknown-elf-gcc -march=rv32imc -mabi=ilp32 -nostdlib \
 *       -T linker.ld -O2 -o blake3_test.elf blake3_test.c
 *
 * ========================================================================== */

#include <stdint.h>

/* =========================================================================
 * Memory-mapped UART registers
 * ========================================================================= */
#define UART_BASE       ((volatile uint32_t *)0x10000000)
#define UART_TX_DATA    (UART_BASE[0])
#define UART_TX_STATUS  (UART_BASE[1])
#define UART_RX_DATA    (UART_BASE[2])
#define UART_RX_STATUS  (UART_BASE[3])

/* =========================================================================
 * Xcrypto inline assembly macros
 * ========================================================================= */

/*
 * R-type custom-0 encoding:
 *   [31:25] funct7 | [24:20] rs2 | [19:15] rs1 | [14:12] funct3=000
 *   | [11:7] rd | [6:0] opcode=0001011
 *
 * We encode instructions as .word directives since the assembler does not
 * know the custom ISA.  Register mapping:
 *   a0 = x10, a1 = x11, a2 = x12, ... a5 = x15
 *   zero = x0
 */

/* blake3.init: funct7=0, rs2=x0, rs1=x0, rd=x0, opcode=0x0B */
#define BLAKE3_INIT()                                                        \
    __asm__ volatile (                                                       \
        ".word 0x0000000B"  /* funct7=0000000 rs2=00000 rs1=00000 f3=000    \
                               rd=00000 opc=0001011 */                      \
        ::: "memory"                                                         \
    )

/*
 * blake3.round: funct7=1, rs2=a1(counter), rs1=a0(msg_addr), rd=a0
 * Encoding: 0000001 | rs2 | rs1 | 000 | rd | 0001011
 *
 * rs1 (a0 = x10): base address of 64-byte message block
 * rs2 (a1 = x11): counter value (upper 32 bits)
 * rd  (a0 = x10): receives status (first hash word)
 */
#define BLAKE3_ROUND(msg_addr, counter)                                      \
    do {                                                                     \
        register uint32_t _a0 __asm__("a0") = (uint32_t)(msg_addr);         \
        register uint32_t _a1 __asm__("a1") = (uint32_t)(counter);          \
        __asm__ volatile (                                                   \
            ".word 0x02B5050B"                                               \
            /* funct7=0000001 rs2=01011(a1) rs1=01010(a0) f3=000            \
               rd=01010(a0) opc=0001011 */                                  \
            : "+r"(_a0)                                                      \
            : "r"(_a1)                                                       \
            : "memory"                                                       \
        );                                                                   \
    } while (0)

/*
 * blake3.chain: funct7=2, rs2=a1(chain_count), rs1=a0(msg_addr), rd=a0
 * Encoding: 0000010 | rs2 | rs1 | 000 | rd | 0001011
 */
#define BLAKE3_CHAIN(msg_addr, count)                                        \
    do {                                                                     \
        register uint32_t _a0 __asm__("a0") = (uint32_t)(msg_addr);         \
        register uint32_t _a1 __asm__("a1") = (uint32_t)(count);            \
        __asm__ volatile (                                                   \
            ".word 0x04B5050B"                                               \
            /* funct7=0000010 rs2=01011(a1) rs1=01010(a0) f3=000            \
               rd=01010(a0) opc=0001011 */                                  \
            : "+r"(_a0)                                                      \
            : "r"(_a1)                                                       \
            : "memory"                                                       \
        );                                                                   \
    } while (0)

/*
 * blake3.finalize: funct7=3, rs2=x0, rs1=a0(word_index), rd=a0
 * Encoding: 0000011 | rs2 | rs1 | 000 | rd | 0001011
 * Returns the hash word at the given index (0..7) in rd.
 */
static inline uint32_t blake3_finalize(uint32_t word_index) {
    register uint32_t _a0 __asm__("a0") = word_index;
    __asm__ volatile (
        ".word 0x0600050B"
        /* funct7=0000011 rs2=00000(x0) rs1=01010(a0) f3=000
           rd=01010(a0) opc=0001011 */
        : "+r"(_a0)
        :
        : "memory"
    );
    return _a0;
}

/* =========================================================================
 * UART output routines
 * ========================================================================= */

static void uart_putc(char c) {
    /* Wait until TX is ready */
    while (!(UART_TX_STATUS & 1))
        ;
    UART_TX_DATA = (uint32_t)c;
}

static void uart_puts(const char *s) {
    while (*s)
        uart_putc(*s++);
}

static void uart_put_hex8(uint32_t val) {
    static const char hex[] = "0123456789abcdef";
    for (int i = 28; i >= 0; i -= 4) {
        uart_putc(hex[(val >> i) & 0xF]);
    }
}

static void uart_put_hex_byte(uint8_t val) {
    static const char hex[] = "0123456789abcdef";
    uart_putc(hex[(val >> 4) & 0xF]);
    uart_putc(hex[val & 0xF]);
}

static void uart_newline(void) {
    uart_putc('\r');
    uart_putc('\n');
}

/* =========================================================================
 * BLAKE3 reference test vector
 * ========================================================================= */

/*
 * Test vector: BLAKE3 hash of empty input (0 bytes).
 *
 * Expected output (256-bit, from BLAKE3 reference implementation):
 *   af1349b9f5f9a1a6a0404dea36dcc9499bcb25c9adc112b7cc9a93cae41f3262
 *
 * For the hardware test, we initialize the state (blake3.init), then run
 * one compression round with an all-zero message block (blake3.round),
 * with flags = CHUNK_START | CHUNK_END | ROOT = 0x0B, block_len = 0.
 *
 * Note: The hardware pipeline initializes the state with the standard IV
 * and the init instruction sets counter=0, block_len=0, flags=0.
 * The firmware must configure flags via rs2 on the round instruction.
 */

static const uint32_t expected_hash[8] = {
    0xaf1349b9, 0xf5f9a1a6, 0xa0404dea, 0x36dcc949,
    0x9bcb25c9, 0xadc112b7, 0xcc9a93ca, 0xe41f3262
};

/* All-zero message block (64 bytes) in data memory */
static volatile uint32_t msg_block[16] __attribute__((aligned(64))) = {
    0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0
};

/* =========================================================================
 * Main test
 * ========================================================================= */

void main(void) {
    uint32_t hash[8];
    int pass = 1;

    uart_puts("QUG-V1 BLAKE3 Xcrypto Test\r\n");
    uart_puts("==========================\r\n");

    /* Step 1: Initialize BLAKE3 state (loads IV, resets pipeline) */
    uart_puts("INIT... ");
    BLAKE3_INIT();
    uart_puts("OK\r\n");

    /* Step 2: Run one compression round on the zero message block */
    uart_puts("ROUND... ");
    BLAKE3_ROUND((uint32_t)msg_block, 0);
    uart_puts("OK\r\n");

    /* Step 3: Read the 8 hash output words via blake3.finalize */
    uart_puts("FINALIZE:\r\n");
    for (int i = 0; i < 8; i++) {
        hash[i] = blake3_finalize((uint32_t)i);
        uart_puts("  hash[");
        uart_putc('0' + i);
        uart_puts("] = 0x");
        uart_put_hex8(hash[i]);
        uart_newline();
    }

    /* Step 4: Compare against expected hash */
    uart_puts("\r\nExpected:\r\n");
    for (int i = 0; i < 8; i++) {
        uart_puts("  hash[");
        uart_putc('0' + i);
        uart_puts("] = 0x");
        uart_put_hex8(expected_hash[i]);
        uart_newline();

        if (hash[i] != expected_hash[i]) {
            pass = 0;
        }
    }

    /* Step 5: Report result */
    uart_newline();
    if (pass) {
        uart_puts("*** PASS ***\r\n");
    } else {
        uart_puts("*** FAIL ***\r\n");
        uart_puts("Hash mismatch detected!\r\n");
    }

    /* Step 6: Test VDF chain (optional -- run 4 chain iterations) */
    uart_puts("\r\nVDF Chain Test (4 iterations):\r\n");
    BLAKE3_INIT();
    BLAKE3_CHAIN((uint32_t)msg_block, 4);
    uart_puts("  chain hash[0] = 0x");
    uart_put_hex8(blake3_finalize(0));
    uart_newline();
    uart_puts("  chain hash[1] = 0x");
    uart_put_hex8(blake3_finalize(1));
    uart_newline();
    uart_puts("VDF chain complete.\r\n");

    /* Halt -- infinite loop */
    uart_puts("\r\nTest complete. Halting.\r\n");
    while (1)
        __asm__ volatile ("wfi");
}

/* =========================================================================
 * Startup code (_start entry point)
 * ========================================================================= */
void _start(void) __attribute__((naked, section(".text.init")));

void _start(void) {
    __asm__ volatile (
        /* Set up stack pointer (top of data memory) */
        "lui  sp, 0x00020\n"   /* sp = 0x00020000 (top of 128KB) */
        /* Clear BSS (simplified -- assume small BSS) */
        "la   t0, __bss_start\n"
        "la   t1, __bss_end\n"
        "1:\n"
        "bge  t0, t1, 2f\n"
        "sw   zero, 0(t0)\n"
        "addi t0, t0, 4\n"
        "j    1b\n"
        "2:\n"
        /* Jump to main */
        "call main\n"
        /* If main returns, loop forever */
        "3: wfi\n"
        "j 3b\n"
    );
}
