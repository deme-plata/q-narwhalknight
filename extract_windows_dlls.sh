#!/bin/bash
# Script to extract Windows DLLs from cross-compilation Docker container

set -e

echo "🔍 Extracting required Windows DLLs from cross container..."

# Create temporary container to extract DLLs
CONTAINER_ID=$(docker create ghcr.io/cross-rs/x86_64-pc-windows-gnu:main)

echo "📦 Container ID: $CONTAINER_ID"

# Create output directory
mkdir -p windows-dlls
cd windows-dlls

# Extract required DLLs
echo "📥 Extracting libgfortran-5.dll..."
docker cp "$CONTAINER_ID:/usr/lib/gcc/x86_64-w64-mingw32/10-posix/libgfortran-5.dll" . 2>/dev/null || \
docker cp "$CONTAINER_ID:/usr/x86_64-w64-mingw32/lib/libgfortran-5.dll" . 2>/dev/null || \
echo "⚠️  libgfortran-5.dll not found in expected locations"

echo "📥 Extracting libgcc_s_seh-1.dll..."
docker cp "$CONTAINER_ID:/usr/lib/gcc/x86_64-w64-mingw32/10-posix/libgcc_s_seh-1.dll" . 2>/dev/null || \
docker cp "$CONTAINER_ID:/usr/x86_64-w64-mingw32/lib/libgcc_s_seh-1.dll" . 2>/dev/null || \
echo "⚠️  libgcc_s_seh-1.dll not found"

echo "📥 Extracting libwinpthread-1.dll..."
docker cp "$CONTAINER_ID:/usr/lib/gcc/x86_64-w64-mingw32/10-posix/libwinpthread-1.dll" . 2>/dev/null || \
docker cp "$CONTAINER_ID:/usr/x86_64-w64-mingw32/lib/libwinpthread-1.dll" . 2>/dev/null || \
echo "⚠️  libwinpthread-1.dll not found"

echo "📥 Extracting libquadmath-0.dll..."
docker cp "$CONTAINER_ID:/usr/lib/gcc/x86_64-w64-mingw32/10-posix/libquadmath-0.dll" . 2>/dev/null || \
docker cp "$CONTAINER_ID:/usr/x86_64-w64-mingw32/lib/libquadmath-0.dll" . 2>/dev/null || \
echo "⚠️  libquadmath-0.dll not found"

# Cleanup
docker rm "$CONTAINER_ID" > /dev/null

echo ""
echo "✅ DLL extraction complete!"
echo "📊 Extracted DLLs:"
ls -lh *.dll 2>/dev/null || echo "⚠️  No DLLs found"

cd ..
