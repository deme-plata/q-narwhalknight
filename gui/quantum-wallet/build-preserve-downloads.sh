#!/bin/bash
# Build script that preserves the downloads folder
# Usage: ./build-preserve-downloads.sh

set -e

echo "🔨 Building quantum-wallet frontend..."
echo "================================================"

# Step 1: Clean old assets but preserve downloads
echo "📦 Cleaning old assets (preserving downloads)..."
if [ -d "dist-final/assets" ]; then
    rm -rf dist-final/assets
    echo "   ✅ Removed old assets"
fi

if [ -f "dist-final/index.html" ]; then
    rm -f dist-final/index.html
    echo "   ✅ Removed old index.html"
fi

if [ -d "dist-final/downloads" ]; then
    echo "   ✅ Preserved downloads folder ($(du -sh dist-final/downloads | cut -f1))"
else
    echo "   ⚠️  No downloads folder found (will be created)"
fi

# Step 2: Run the build
echo ""
echo "🏗️  Running npm build..."
npm run build

# Step 3: Ensure downloads folder exists
echo ""
echo "📁 Ensuring downloads folder exists..."
mkdir -p dist-final/downloads
echo "   ✅ downloads folder ready"

# Step 4: Show results
echo ""
echo "✅ Build complete!"
echo "================================================"
echo ""
echo "📊 Build output:"
ls -lh dist-final/ | grep -v "^total"
echo ""
echo "📦 Downloads available:"
ls -lh dist-final/downloads/ 2>/dev/null | grep -v "^total" || echo "   (empty)"
echo ""
echo "🎉 Frontend ready to serve via nginx!"
