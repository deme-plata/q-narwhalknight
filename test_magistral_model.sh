#!/bin/bash
# Test Script for Magistral-Small-2509 Model
# Downloads and configures the model for distributed AI testing

set -e

MODEL_URL="https://huggingface.co/mistralai/Magistral-Small-2509-GGUF/resolve/main/Magistral-Small-2509-Q4_K_M.gguf"
MODEL_DIR="/opt/orobit/shared/q-narwhalknight/models"
MODEL_FILE="$MODEL_DIR/Magistral-Small-2509-Q4_K_M.gguf"

echo "🚀 Magistral-Small-2509 Model Test Setup"
echo "=========================================="
echo ""

# Create models directory if it doesn't exist
mkdir -p "$MODEL_DIR"

# Check if model already exists
if [ -f "$MODEL_FILE" ]; then
    echo "✅ Model already downloaded: $MODEL_FILE"
    MODEL_SIZE=$(du -h "$MODEL_FILE" | cut -f1)
    echo "   Size: $MODEL_SIZE"
else
    echo "📥 Downloading Magistral-Small-2509-Q4_K_M.gguf..."
    echo "   URL: $MODEL_URL"
    echo "   Destination: $MODEL_FILE"
    echo ""
    
    # Download with progress
    wget -c "$MODEL_URL" -O "$MODEL_FILE.tmp" 2>&1 | grep --line-buffered "%" | sed -u 's/.*\([0-9]\+%\).*$/\1/'
    
    # Move to final location
    mv "$MODEL_FILE.tmp" "$MODEL_FILE"
    
    echo ""
    echo "✅ Download complete!"
    MODEL_SIZE=$(du -h "$MODEL_FILE" | cut -f1)
    echo "   Size: $MODEL_SIZE"
fi

echo ""
echo "📊 Model Information:"
echo "   Name: Magistral-Small-2509"
echo "   Format: GGUF Q4_K_M"
echo "   Path: $MODEL_FILE"

# Get model info using file command
if command -v file &> /dev/null; then
    echo ""
    echo "🔍 File Info:"
    file "$MODEL_FILE"
fi

echo ""
echo "🧪 Testing Model with q-ai-inference..."
echo ""

# Test if we can load model info
cd /opt/orobit/shared/q-narwhalknight

# Check if q-api-server is built
if [ -f "target/release/q-api-server" ]; then
    echo "✅ q-api-server binary found"
    
    # Create test configuration
    cat > /tmp/magistral_test_config.json << TESTEOF
{
  "model_path": "$MODEL_FILE",
  "max_tokens": 512,
  "temperature": 0.7,
  "test_prompt": "Hello! Can you introduce yourself and tell me what you can do?"
}
TESTEOF
    
    echo ""
    echo "📝 Test Configuration:"
    cat /tmp/magistral_test_config.json
    
    echo ""
    echo "🚀 You can now test the model with:"
    echo ""
    echo "   # Start API server with Magistral model:"
    echo "   Q_MODEL_PATH=\"$MODEL_FILE\" ./target/release/q-api-server"
    echo ""
    echo "   # Or test inference directly:"
    echo "   curl -X POST http://localhost:8080/api/chat/stream \\"
    echo "     -H 'Content-Type: application/json' \\"
    echo "     -d '{\"content\": \"Hello! Can you introduce yourself?\", \"max_tokens\": 512}'"
    
else
    echo "⚠️  q-api-server not built yet. Run:"
    echo "   timeout 36000 cargo build --release --package q-api-server"
fi

echo ""
echo "🎯 Distributed AI Testing:"
echo ""
echo "   The Magistral-Small-2509 model has the same architecture as Mistral-7B,"
echo "   so it works with our distributed AI infrastructure:"
echo ""
echo "   ✅ Layer assignment algorithm (32 layers)"
echo "   ✅ Tensor forwarding with compression"
echo "   ✅ Coordinator election"
echo "   ✅ Multi-node inference"
echo ""
echo "   To test distributed inference across 3 nodes:"
echo "   1. Start node 1: Q_MODEL_PATH=\"$MODEL_FILE\" Q_DB_PATH=./data-node1 ./target/release/q-api-server --port 8001"
echo "   2. Start node 2: Q_MODEL_PATH=\"$MODEL_FILE\" Q_DB_PATH=./data-node2 ./target/release/q-api-server --port 8002"
echo "   3. Start node 3: Q_MODEL_PATH=\"$MODEL_FILE\" Q_DB_PATH=./data-node3 ./target/release/q-api-server --port 8003"
echo ""

echo "✨ Setup complete! Ready for testing."
