(module
  ;; Import memory and environment functions
  (memory (export "memory") 1)
  (import "env" "read_state" (func  (param i32 i32 i32 i32) (result i32)))
  (import "env" "write_state" (func  (param i32 i32 i32 i32) (result i32)))

  ;; Data section for string constants
  (data (i32.const 0) "owner")
  (data (i32.const 6) "total_supply")
  (data (i32.const 19) "balance_")
  
  ;; Function to initialize the token
  (func  (param  i32) (result i32)
    ;; Store the caller as owner
    (i32.store (i32.const 1000) (i32.const 0)) ;; "owner" key offset
    (i32.store (i32.const 1004) (i32.const 5)) ;; "owner" key length
    
    ;; Store owner address at memory position 2000
    ;; In a real implementation, this would be the caller's address from environment
    (i32.store (i32.const 2000) (i32.const 123456789)) ;; Mock owner address
    
    ;; Write owner to state
    (call  
      (i32.const 1000) ;; key_ptr
      (i32.const 5)    ;; key_len
      (i32.const 2000) ;; value_ptr
      (i32.const 4)    ;; value_len
    )
    drop
    
    ;; Store total_supply key
    (i32.store (i32.const 1000) (i32.const 6)) ;; "total_supply" key offset
    (i32.store (i32.const 1004) (i32.const 12)) ;; "total_supply" key length
    
    ;; Store total_supply value
    (i32.store (i32.const 2000) (local.get ))
    
    ;; Write total_supply to state
    (call  
      (i32.const 1000) ;; key_ptr
      (i32.const 12)   ;; key_len
      (i32.const 2000) ;; value_ptr
      (i32.const 4)    ;; value_len
    )
    drop
    
    ;; Set initial balance for owner
    ;; Construct balance_<owner> key
    (i32.store (i32.const 1000) (i32.const 19)) ;; "balance_" key offset
    (i32.store (i32.const 1004) (i32.const 8))  ;; "balance_" key length
    
    ;; Store owner address after "balance_" prefix
    (i32.store (i32.const 1008) (i32.const 123456789)) ;; Mock owner address
    
    ;; Store total_supply value as owner's balance
    (i32.store (i32.const 2000) (local.get ))
    
    ;; Write owner's balance to state
    (call  
      (i32.const 1000) ;; key_ptr
      (i32.const 12)   ;; key_len (balance_ + address)
      (i32.const 2000) ;; value_ptr
      (i32.const 4)    ;; value_len
    )
    drop
    
    ;; Return success
    (i32.const 1)
  )
  
  ;; Function to transfer tokens
  (func  (param  i32) (param  i32) (result i32)
    (local  i32)
    (local  i32)
    
    ;; Get sender's balance
    ;; Construct balance_<sender> key
    (i32.store (i32.const 1000) (i32.const 19)) ;; "balance_" key offset
    (i32.store (i32.const 1004) (i32.const 8))  ;; "balance_" key length
    
    ;; Store sender address after "balance_" prefix
    (i32.store (i32.const 1008) (i32.const 123456789)) ;; Mock sender address
    
    ;; Read sender's balance
    (call  
      (i32.const 1000) ;; key_ptr
      (i32.const 12)   ;; key_len (balance_ + address)
      (i32.const 2000) ;; value_ptr
      (i32.const 3000) ;; value_len_ptr
    )
    drop
    
    ;; Load sender balance
    (local.set  (i32.load (i32.const 2000)))
    
    ;; Check if sender has enough balance
    (if (i32.lt_u (local.get ) (local.get ))
        (then
            ;; Return failure
            (return (i32.const 0))
        )
    )
    
    ;; Get receiver's balance
    ;; Construct balance_<receiver> key
    (i32.store (i32.const 1000) (i32.const 19)) ;; "balance_" key offset
    (i32.store (i32.const 1004) (i32.const 8))  ;; "balance_" key length
    
    ;; Store receiver address after "balance_" prefix
    (i32.store (i32.const 1008) (local.get ))
    
    ;; Read receiver's balance
    (call  
      (i32.const 1000) ;; key_ptr
      (i32.const 12)   ;; key_len (balance_ + address)
      (i32.const 2000) ;; value_ptr
      (i32.const 3000) ;; value_len_ptr
    )
    
    ;; If receiver has no balance yet, set it to 0
    (if (i32.eq (i32.const 0)) 
        (then
            (local.set  (i32.const 0))
        )
        (else
            ;; Load receiver balance
            (local.set  (i32.load (i32.const 2000)))
        )
    )
    
    ;; Update sender's balance
    ;; Construct balance_<sender> key
    (i32.store (i32.const 1000) (i32.const 19)) ;; "balance_" key offset
    (i32.store (i32.const 1004) (i32.const 8))  ;; "balance_" key length
    
    ;; Store sender address after "balance_" prefix
    (i32.store (i32.const 1008) (i32.const 123456789)) ;; Mock sender address
    
    ;; Calculate new sender balance
    (i32.store (i32.const 2000) 
        (i32.sub (local.get ) (local.get ))
    )
    
    ;; Write sender's updated balance to state
    (call  
      (i32.const 1000) ;; key_ptr
      (i32.const 12)   ;; key_len (balance_ + address)
      (i32.const 2000) ;; value_ptr
      (i32.const 4)    ;; value_len
    )
    drop
    
    ;; Update receiver's balance
    ;; Construct balance_<receiver> key
    (i32.store (i32.const 1000) (i32.const 19)) ;; "balance_" key offset
    (i32.store (i32.const 1004) (i32.const 8))  ;; "balance_" key length
    
    ;; Store receiver address after "balance_" prefix
    (i32.store (i32.const 1008) (local.get ))
    
    ;; Calculate new receiver balance
    (i32.store (i32.const 2000) 
        (i32.add (local.get ) (local.get ))
    )
    
    ;; Write receiver's updated balance to state
    (call  
      (i32.const 1000) ;; key_ptr
      (i32.const 12)   ;; key_len (balance_ + address)
      (i32.const 2000) ;; value_ptr
      (i32.const 4)    ;; value_len
    )
    drop
    
    ;; Return success
    (i32.const 1)
  )
  
  ;; Function to get a balance
  (func  (param  i32) (result i32)
    ;; Construct balance_<address> key
    (i32.store (i32.const 1000) (i32.const 19)) ;; "balance_" key offset
    (i32.store (i32.const 1004) (i32.const 8))  ;; "balance_" key length
    
    ;; Store address after "balance_" prefix
    (i32.store (i32.const 1008) (local.get ))
    
    ;; Read balance
    (call  
      (i32.const 1000) ;; key_ptr
      (i32.const 12)   ;; key_len (balance_ + address)
      (i32.const 2000) ;; value_ptr
      (i32.const 3000) ;; value_len_ptr
    )
    
    ;; If user has no balance, return 0
    (if (i32.eq (i32.const 0)) 
        (then
            (return (i32.const 0))
        )
    )
    
    ;; Return balance
    (i32.load (i32.const 2000))
  )
  
  ;; Function to get total supply
  (func  (result i32)
    ;; Store total_supply key
    (i32.store (i32.const 1000) (i32.const 6)) ;; "total_supply" key offset
    (i32.store (i32.const 1004) (i32.const 12)) ;; "total_supply" key length
    
    ;; Read total supply
    (call  
      (i32.const 1000) ;; key_ptr
      (i32.const 12)   ;; key_len
      (i32.const 2000) ;; value_ptr
      (i32.const 3000) ;; value_len_ptr
    )
    
    ;; Return total supply
    (i32.load (i32.const 2000))
  )
  
  ;; Export functions
  (export "init" (func ))
  (export "transfer" (func ))
  (export "balanceOf" (func ))
  (export "totalSupply" (func ))
)
