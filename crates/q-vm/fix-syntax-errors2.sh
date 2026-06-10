#!/bin/bash

# Script to fix more syntax errors in models/mod.rs
echo "Fixing more syntax errors in models/mod.rs..."

PROJECT_DIR=$(pwd)
MODELS_FILE="${PROJECT_DIR}/src/models/mod.rs"

if [ -f "$MODELS_FILE" ]; then
  # Make a backup
  cp "$MODELS_FILE" "${MODELS_FILE}.bak"
  
  # Fix all unclosed to_string() calls for model_id
  sed -i 's/\.to_string(/\.to_string()/g' "$MODELS_FILE"
  
  echo "  Fixed more syntax errors in models/mod.rs"
else
  echo "  Could not find models/mod.rs file"
fi

echo "More syntax fixes applied! Try building your project again with 'cargo build'"
