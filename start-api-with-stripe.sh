#!/bin/bash
# Start Q-NarwhalKnight API Server with Stripe Configuration

# Load environment variables from .env file
if [ -f "crates/q-api-server/.env" ]; then
    export $(cat crates/q-api-server/.env | grep -v '^#' | xargs)
    echo "✅ Loaded Stripe configuration from .env"
    echo "📊 STRIPE_SECRET_KEY is set: ${STRIPE_SECRET_KEY:0:20}..."
else
    echo "⚠️  No .env file found. Stripe payments will not work."
fi

# Start the API server
echo "🚀 Starting Q-NarwhalKnight API Server..."
./target/x86_64-unknown-linux-gnu/release/q-api-server
