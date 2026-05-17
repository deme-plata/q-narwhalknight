#!/bin/bash

# Script to fix syntax errors in models/mod.rs
echo "Fixing syntax errors in models/mod.rs..."

PROJECT_DIR=$(pwd)
MODELS_FILE="${PROJECT_DIR}/src/models/mod.rs"

if [ -f "$MODELS_FILE" ]; then
  # Make a backup
  cp "$MODELS_FILE" "${MODELS_FILE}.bak"
  
  # Fix RwLock::new(HashMap::new(), (extra comma)
  sed -i 's/RwLock::new(HashMap::new(),/RwLock::new(HashMap::new())/g' "$MODELS_FILE"
  
  # Fix registration.clone(, (extra opening parenthesis without closing)
  sed -i 's/registration.clone(/registration.clone()/g' "$MODELS_FILE"
  
  # Fix registration.model_id.clone(, (extra opening parenthesis without closing)
  sed -i 's/registration.model_id.clone(/registration.model_id.clone()/g' "$MODELS_FILE"
  
  # Fix capability.clone(, (extra opening parenthesis without closing)
  sed -i 's/capability.clone(/capability.clone()/g' "$MODELS_FILE"
  
  # Fix "system".to_string(, (extra opening parenthesis without closing)
  sed -i 's/"system".to_string(/\"system\".to_string()/g' "$MODELS_FILE"
  
  # Fix "2.0".to_string(, (extra opening parenthesis without closing)
  sed -i 's/"2.0".to_string(/\"2.0\".to_string()/g' "$MODELS_FILE"
  
  # Fix "1.0".to_string(, (extra opening parenthesis without closing)
  sed -i 's/"1.0".to_string(/\"1.0\".to_string()/g' "$MODELS_FILE"
  
  echo "  Fixed syntax errors in models/mod.rs"
else
  echo "  Could not find models/mod.rs file"
fi

echo "Syntax fixes applied! Try building your project again with 'cargo build'"
