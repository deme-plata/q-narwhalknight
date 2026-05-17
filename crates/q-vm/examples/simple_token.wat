;; Simple Token Smart Contract in WebAssembly Text Format (WAT)
;;
;; This contract implements a basic ERC20-like token with:
;; - Balance tracking
;; - Transfer functionality
;; - Total supply management

(module
  ;; Import host functions for storage
  (import "env" "storage_read" (func $storage_read (param i32 i32) (result i32)))
  (import "env" "storage_write" (func $storage_write (param i32 i32 i32)))
  (import "env" "get_caller" (func $get_caller (param i32)))

  ;; Memory for contract state
  (memory (export "memory") 1)

  ;; Contract functions

  ;; Initialize contract with total supply
  ;; Args: total_supply (u64)
  (func (export "init") (param $total_supply i64)
    (local $creator_addr i32)

    ;; Get creator address
    (local.set $creator_addr (i32.const 0))
    (call $get_caller (local.get $creator_addr))

    ;; Store total supply at address 0x100
    (i64.store (i32.const 0x100) (local.get $total_supply))

    ;; Give all tokens to creator
    ;; Balance key = address + offset 0x1000
    (i64.store (i32.const 0x1000) (local.get $total_supply))
  )

  ;; Get balance of an account
  ;; Args: account_addr (ptr to 32 bytes)
  ;; Returns: balance (u64)
  (func (export "balance_of") (param $account i32) (result i64)
    (local $balance_addr i32)

    ;; Calculate balance storage location
    ;; Simple hash: account_addr + 0x1000
    (local.set $balance_addr (i32.add (local.get $account) (i32.const 0x1000)))

    ;; Load balance from memory
    (i64.load (local.get $balance_addr))
  )

  ;; Transfer tokens
  ;; Args: to_addr (ptr), amount (u64)
  ;; Returns: success (i32)
  (func (export "transfer") (param $to i32) (param $amount i64) (result i32)
    (local $from i32)
    (local $from_balance i64)
    (local $to_balance i64)
    (local $from_addr i32)
    (local $to_addr i32)

    ;; Get caller address
    (local.set $from (i32.const 0))
    (call $get_caller (local.get $from))

    ;; Calculate storage addresses
    (local.set $from_addr (i32.add (local.get $from) (i32.const 0x1000)))
    (local.set $to_addr (i32.add (local.get $to) (i32.const 0x1000)))

    ;; Load sender balance
    (local.set $from_balance (i64.load (local.get $from_addr)))

    ;; Check sufficient balance
    (if (i64.lt_u (local.get $from_balance) (local.get $amount))
      (then
        ;; Insufficient balance, return 0 (failure)
        (return (i32.const 0))
      )
    )

    ;; Load recipient balance
    (local.set $to_balance (i64.load (local.get $to_addr)))

    ;; Update balances
    (i64.store (local.get $from_addr)
               (i64.sub (local.get $from_balance) (local.get $amount)))

    (i64.store (local.get $to_addr)
               (i64.add (local.get $to_balance) (local.get $amount)))

    ;; Return 1 (success)
    (i32.const 1)
  )

  ;; Get total supply
  ;; Returns: total_supply (u64)
  (func (export "total_supply") (result i64)
    (i64.load (i32.const 0x100))
  )

  ;; Mint new tokens (only creator can call)
  ;; Args: amount (u64)
  ;; Returns: success (i32)
  (func (export "mint") (param $amount i64) (result i32)
    (local $caller i32)
    (local $creator i32)
    (local $total_supply i64)
    (local $caller_balance i64)
    (local $caller_addr i32)

    ;; Get caller
    (local.set $caller (i32.const 0))
    (call $get_caller (local.get $caller))

    ;; For simplicity, creator is always at address 0
    ;; In production, verify caller == creator

    ;; Load total supply
    (local.set $total_supply (i64.load (i32.const 0x100)))

    ;; Update total supply
    (i64.store (i32.const 0x100)
               (i64.add (local.get $total_supply) (local.get $amount)))

    ;; Calculate caller balance address
    (local.set $caller_addr (i32.add (local.get $caller) (i32.const 0x1000)))

    ;; Load caller balance
    (local.set $caller_balance (i64.load (local.get $caller_addr)))

    ;; Mint tokens to caller
    (i64.store (local.get $caller_addr)
               (i64.add (local.get $caller_balance) (local.get $amount)))

    ;; Return success
    (i32.const 1)
  )
)
