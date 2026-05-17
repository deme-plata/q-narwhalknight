/* =============================================================================
 * mining_loop.c -- Mining Firmware for QUG-V1 RISC-V SoC
 * QUG-V1 Mining SoC -- RISC-V Firmware
 * =============================================================================
 * Project  : QUG-V1 RISC-V Mining SoC
 * Target   : QUG-V1 RV32IMC + Xcrypto Extension
 * Author   : Quillon Foundation / Dragon Ball Miner
 * License  : MIT
 * =============================================================================
 *
 * Mining loop firmware:
 *   1. Read mining challenge from memory-mapped register
 *   2. For each nonce:
 *      a. Prepare message block (challenge + nonce)
 *      b. Run 100 BLAKE3 hashes via Xcrypto VDF chain
 *      c. Check final hash against difficulty target
 *      d. If solution found: write nonce to UART
 *   3. Increment nonce, repeat
 *
 * Memory-mapped I/O:
 *   0x10000000: UART TX data
 *   0x10000004: UART TX status (bit 0 = ready)
 *   0x10000008: UART RX data
 *   0x1000000C: UART RX status (bit 0 = valid)
 *
 * Build:
 *   riscv32-unknown-elf-gcc -march=rv32imc -mabi=ilp32 -nostdlib \
 *       -T linker.ld -O2 -o mining_loop.elf mining_loop.c
 *
 * ========================================================================== */

#include <stdint.h>

/* =========================================================================
 * Memory-mapped I/O
 * ========================================================================= */
#define UART_BASE       ((volatile uint32_t *)0x10000000)
#define UART_TX_DATA    (UART_BASE[0])
#define UART_TX_STATUS  (UART_BASE[1])
#define UART_RX_DATA    (UART_BASE[2])
#define UART_RX_STATUS  (UART_BASE[3])

/* =========================================================================
 * Mining configuration
 * ========================================================================= */

/* Number of BLAKE3 hash iterations per nonce (VDF chain length) */
#define VDF_CHAIN_LENGTH    100

/* Difficulty: number of leading zero bits required in final hash.
 * Start easy for testing; real difficulty set by network consensus. */
#define DIFFICULTY_BITS     16

/* =========================================================================
 * Xcrypto inline assembly
 * ========================================================================= */

/* blake3.init: Initialize BLAKE3 state with standard IV */
#define BLAKE3_INIT()                                                        \
    __asm__ volatile (".word 0x0000000B" ::: "memory")

/*
 * blake3.chain: Hash message block with VDF chain of `count` iterations.
 * rs1 (a0) = message block base address
 * rs2 (a1) = chain iteration count
 */
#define BLAKE3_CHAIN(msg_addr, count)                                        \
    do {                                                                     \
        register uint32_t _a0 __asm__("a0") = (uint32_t)(msg_addr);         \
        register uint32_t _a1 __asm__("a1") = (uint32_t)(count);            \
        __asm__ volatile (                                                   \
            ".word 0x04B5050B"                                               \
            : "+r"(_a0)                                                      \
            : "r"(_a1)                                                       \
            : "memory"                                                       \
        );                                                                   \
    } while (0)

/*
 * blake3.finalize: Read hash output word at index (0..7).
 * rs1 (a0) = word index
 * rd  (a0) = hash word value
 */
static inline uint32_t blake3_finalize(uint32_t word_index) {
    register uint32_t _a0 __asm__("a0") = word_index;
    __asm__ volatile (
        ".word 0x0600050B"
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
    for (int i = 28; i >= 0; i -= 4)
        uart_putc(hex[(val >> i) & 0xF]);
}

static void uart_newline(void) {
    uart_putc('\r');
    uart_putc('\n');
}

/* =========================================================================
 * UART input (check for new challenge)
 * ========================================================================= */

/* Non-blocking: returns 1 if a byte is available, 0 otherwise */
static inline int uart_rx_ready(void) {
    return (UART_RX_STATUS & 1);
}

static inline uint8_t uart_getc(void) {
    return (uint8_t)(UART_RX_DATA & 0xFF);
}

/* =========================================================================
 * Difficulty check
 * ========================================================================= */

/*
 * Check if the hash meets the difficulty target.
 * For DIFFICULTY_BITS leading zeros, we check that the first
 * (DIFFICULTY_BITS / 32) words are zero and the next word has
 * the required number of leading zero bits.
 */
static int check_difficulty(uint32_t hash0) {
#if DIFFICULTY_BITS <= 32
    /* Check leading zeros in the first hash word */
    uint32_t mask = 0xFFFFFFFF << (32 - DIFFICULTY_BITS);
    return (hash0 & mask) == 0;
#else
    /* For higher difficulty, would need to check multiple words.
     * Simplified: just check first word is zero. */
    return hash0 == 0;
#endif
}

/* =========================================================================
 * Mining message block
 * ========================================================================= */

/* 64-byte message block in data memory (word-aligned for Xcrypto DMA).
 *
 * Layout:
 *   msg[0..7]:   32-byte mining challenge (from network)
 *   msg[8..9]:   64-bit nonce (little-endian)
 *   msg[10..15]: padding (zeros)
 */
static volatile uint32_t msg_block[16] __attribute__((aligned(64)));

/* Default challenge (overwritten when a real challenge arrives via UART) */
static uint32_t challenge[8] = {
    0xDEADBEEF, 0xCAFEBABE, 0x12345678, 0x9ABCDEF0,
    0x0BADF00D, 0xFEEDFACE, 0xC0FFEE00, 0xBAAAAAAD
};

/* =========================================================================
 * Mining loop
 * ========================================================================= */

void main(void) {
    uint32_t nonce_lo = 0;
    uint32_t nonce_hi = 0;
    uint32_t solutions = 0;
    uint32_t hash0;

    uart_puts("QUG-V1 Mining Firmware v1.0\r\n");
    uart_puts("==========================\r\n");
    uart_puts("VDF chain length: ");
    uart_put_hex8(VDF_CHAIN_LENGTH);
    uart_newline();
    uart_puts("Difficulty bits:  ");
    uart_put_hex8(DIFFICULTY_BITS);
    uart_newline();
    uart_puts("Starting mining loop...\r\n\r\n");

    /* Load initial challenge into message block */
    for (int i = 0; i < 8; i++)
        msg_block[i] = challenge[i];

    /* Clear padding words */
    for (int i = 10; i < 16; i++)
        msg_block[i] = 0;

    /* Main mining loop */
    while (1) {
        /* Check for new challenge from UART (non-blocking).
         * Protocol: 32 bytes (8 words) sent MSB-first. If RX has data
         * and first byte is 0xFF (challenge header), read 32 bytes. */
        if (uart_rx_ready()) {
            uint8_t header = uart_getc();
            if (header == 0xFF) {
                /* Read 32-byte challenge */
                for (int i = 0; i < 8; i++) {
                    uint32_t word = 0;
                    for (int b = 0; b < 4; b++) {
                        while (!uart_rx_ready())
                            ;
                        word = (word << 8) | uart_getc();
                    }
                    challenge[i] = word;
                    msg_block[i] = word;
                }
                /* Reset nonce on new challenge */
                nonce_lo = 0;
                nonce_hi = 0;
                uart_puts("New challenge received.\r\n");
            }
        }

        /* Set nonce in message block */
        msg_block[8] = nonce_lo;
        msg_block[9] = nonce_hi;

        /* Initialize BLAKE3 state */
        BLAKE3_INIT();

        /* Run VDF chain: 100 sequential BLAKE3 compressions.
         * The hardware blake3.chain instruction runs the entire chain
         * without round-tripping through the register file. */
        BLAKE3_CHAIN((uint32_t)msg_block, VDF_CHAIN_LENGTH);

        /* Read first hash word to check difficulty */
        hash0 = blake3_finalize(0);

        /* Check if hash meets difficulty target */
        if (check_difficulty(hash0)) {
            solutions++;

            /* Solution found -- output via UART */
            uart_puts("SOLUTION FOUND!\r\n");
            uart_puts("  nonce_hi: 0x");
            uart_put_hex8(nonce_hi);
            uart_newline();
            uart_puts("  nonce_lo: 0x");
            uart_put_hex8(nonce_lo);
            uart_newline();
            uart_puts("  hash[0]:  0x");
            uart_put_hex8(hash0);
            uart_newline();

            /* Output full 256-bit hash */
            uart_puts("  full hash: ");
            for (int i = 0; i < 8; i++) {
                uart_put_hex8(blake3_finalize((uint32_t)i));
            }
            uart_newline();

            uart_puts("  solutions: ");
            uart_put_hex8(solutions);
            uart_newline();
            uart_newline();
        }

        /* Increment 64-bit nonce */
        nonce_lo++;
        if (nonce_lo == 0)
            nonce_hi++;

        /* Periodic status output every 2^20 (~1M) nonces */
        if ((nonce_lo & 0x000FFFFF) == 0) {
            uart_puts("nonce: 0x");
            uart_put_hex8(nonce_hi);
            uart_put_hex8(nonce_lo);
            uart_puts("  hash0: 0x");
            uart_put_hex8(hash0);
            uart_puts("  solutions: ");
            uart_put_hex8(solutions);
            uart_newline();
        }
    }
}

/* =========================================================================
 * Startup code (_start entry point)
 * ========================================================================= */
void _start(void) __attribute__((naked, section(".text.init")));

void _start(void) {
    __asm__ volatile (
        /* Set up stack pointer at top of data memory */
        "lui  sp, 0x00020\n"   /* sp = 0x00020000 */
        /* Clear BSS */
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
        /* Should never return, but just in case */
        "3: wfi\n"
        "j 3b\n"
    );
}
