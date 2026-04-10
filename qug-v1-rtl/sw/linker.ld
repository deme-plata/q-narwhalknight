/* =============================================================================
 * linker.ld -- RISC-V Bare-Metal Linker Script
 * QUG-V1 Mining SoC -- RISC-V Firmware
 * =============================================================================
 * Project  : QUG-V1 RISC-V Mining SoC
 * Target   : QUG-V1 RV32IMC
 * Author   : Quillon Foundation / Dragon Ball Miner
 * License  : MIT
 * =============================================================================
 *
 * Memory map (matching mem_subsystem.sv):
 *   0x0000_0000 - 0x0000_FFFF : Instruction BRAM (64KB, read-only from CPU)
 *   0x0001_0000 - 0x0001_FFFF : Data BRAM        (64KB, read-write)
 *   0x1000_0000 - 0x1000_000F : UART I/O         (memory-mapped peripherals)
 *
 * Stack: grows downward from top of data memory (0x0001_FFFC).
 * Entry point: _start (at address 0x0000_0000).
 *
 * ========================================================================== */

OUTPUT_ARCH("riscv")
ENTRY(_start)

MEMORY
{
    IMEM (rx)  : ORIGIN = 0x00000000, LENGTH = 64K
    DMEM (rwx) : ORIGIN = 0x00010000, LENGTH = 64K
}

SECTIONS
{
    /* ==================================================================== */
    /* Text section -- instruction memory (BRAM, read-only from CPU)        */
    /* ==================================================================== */
    .text : ALIGN(4)
    {
        /* Reset vector at address 0 -- _start must be first */
        *(.text.init)
        *(.text.init.*)

        /* Remaining code */
        *(.text)
        *(.text.*)

        /* Read-only data that can live in instruction memory */
        *(.rodata)
        *(.rodata.*)
        *(.rodata.str1.*)

        . = ALIGN(4);
        __text_end = .;
    } > IMEM

    /* ==================================================================== */
    /* Initialized data -- loaded into data memory                          */
    /* ==================================================================== */
    /* The .data section is placed in DMEM.  If booting from flash,         */
    /* a bootloader would copy this from flash to DMEM.  For BRAM           */
    /* prototype, Vivado pre-loads both BRAMs from the ELF.                 */
    .data : ALIGN(4)
    {
        __data_start = .;
        *(.data)
        *(.data.*)
        *(.sdata)
        *(.sdata.*)
        . = ALIGN(4);
        __data_end = .;
    } > DMEM

    /* ==================================================================== */
    /* BSS -- zero-initialized data in data memory                          */
    /* ==================================================================== */
    .bss (NOLOAD) : ALIGN(4)
    {
        __bss_start = .;
        *(.bss)
        *(.bss.*)
        *(.sbss)
        *(.sbss.*)
        *(COMMON)
        . = ALIGN(4);
        __bss_end = .;
    } > DMEM

    /* ==================================================================== */
    /* Stack -- grows downward from top of data memory                      */
    /* ==================================================================== */
    /* Stack pointer initialized in _start to __stack_top.                  */
    /* Reserve at least 4KB for stack.                                      */

    __stack_size = 4K;
    __stack_top  = ORIGIN(DMEM) + LENGTH(DMEM);
    __stack_bottom = __stack_top - __stack_size;

    /* Ensure BSS does not collide with stack */
    ASSERT(__bss_end <= __stack_bottom,
           "ERROR: BSS overlaps stack region!")

    /* ==================================================================== */
    /* Heap (optional) -- between BSS end and stack bottom                  */
    /* ==================================================================== */
    __heap_start = __bss_end;
    __heap_end   = __stack_bottom;

    /* ==================================================================== */
    /* Discard debug and comment sections to reduce binary size             */
    /* ==================================================================== */
    /DISCARD/ :
    {
        *(.comment)
        *(.comment.*)
        *(.note)
        *(.note.*)
        *(.eh_frame)
        *(.eh_frame_hdr)
    }
}
