#!/bin/bash

# Fix Slint UI syntax errors across all UI files

UI_DIR="/opt/orobit/shared/q-narwhalknight/gui/ui"

echo "🎨 Fixing Slint UI syntax errors..."

# Fix font-family with multiple fonts
find "$UI_DIR" -name "*.slint" -exec sed -i 's/font-family: "Monaco", "Cascadia Code", monospace;/font-family: "Monaco";/g' {} \;

# Fix linear-gradient syntax - add @ prefix
find "$UI_DIR" -name "*.slint" -exec sed -i 's/background: linear-gradient(/background: @linear-gradient(/g' {} \;
find "$UI_DIR" -name "*.slint" -exec sed -i 's/: linear-gradient(/: @linear-gradient(/g' {} \;

# Fix radial-gradient syntax - simplify
find "$UI_DIR" -name "*.slint" -exec sed -i 's/background: radial-gradient(circle at [^,]*, /background: @radial-gradient(circle, /g' {} \;
find "$UI_DIR" -name "*.slint" -exec sed -i 's/background: radial-gradient(circle, /background: @radial-gradient(circle, /g' {} \;
find "$UI_DIR" -name "*.slint" -exec sed -i 's/: radial-gradient(circle, /: @radial-gradient(circle, /g' {} \;

# Fix percentage in gradients - remove percentages from colors
find "$UI_DIR" -name "*.slint" -exec sed -i 's/ 0%,/, /g' {} \;
find "$UI_DIR" -name "*.slint" -exec sed -i 's/ [0-9][0-9]%)/)/g' {} \;
find "$UI_DIR" -name "*.slint" -exec sed -i 's/ 100%)/)/g' {} \;

# Fix modulo operator - replace with Math.mod
find "$UI_DIR" -name "*.slint" -exec sed -i 's/(\([^)]*\)) % /Math.mod(\1, /g' {} \;

# Fix 'as float' type casting - remove as it's not supported in Slint
find "$UI_DIR" -name "*.slint" -exec sed -i 's/ as float//g' {} \;

# Fix color transparency functions - remove .transparentize() calls
find "$UI_DIR" -name "*.slint" -exec sed -i 's/[a-zA-Z-]*\.transparentize([^)]*)/#ffffff44/g' {} \;

echo "✅ Fixed Slint syntax issues in all UI files!"