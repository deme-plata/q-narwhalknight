#!/bin/bash

# Setup script for ShadowChain Writer CLI
# This script initializes the story database and imports Chapter One

echo "🖋️  ShadowChain Writer - Story Setup"
echo "======================================"

# Check if CLI tool is built
if [ ! -f "./target/release/shadowchain-writer" ]; then
    echo "❌ CLI tool not found. Please run 'cargo build --release' first."
    exit 1
fi

CLI_TOOL="./target/release/shadowchain-writer"

echo "✅ CLI tool found"
echo ""

# Initialize the story with pre-loaded entities
echo "📚 Initializing story database..."
$CLI_TOOL init "Shadows in the Chain" --author "Q-NarwhalKnight Authors"

if [ $? -eq 0 ]; then
    echo "✅ Story database initialized successfully"
else
    echo "❌ Failed to initialize story database"
    exit 1
fi

echo ""

# Create Chapter One
echo "📖 Creating Chapter One..."
CHAPTER_ID=$($CLI_TOOL story chapter create "Digital Shadows" --description "Opening chapter introducing Elena Voss in Berlin's underground and her first encounter with Marcus Hale's surveillance operation.")

if [ $? -eq 0 ]; then
    echo "✅ Chapter One created successfully"
else
    echo "❌ Failed to create Chapter One"
    exit 1
fi

echo ""

# Create scenes for Chapter One
echo "🎬 Creating scenes for Chapter One..."

# Scene 1: Ghost Protocol
$CLI_TOOL story scene create "Digital Shadows" "Ghost Protocol"
if [ $? -eq 0 ]; then
    echo "✅ Scene 1 'Ghost Protocol' created"
else
    echo "❌ Failed to create Scene 1"
fi

# Scene 2: The Hunt Begins
$CLI_TOOL story scene create "Digital Shadows" "The Hunt Begins"
if [ $? -eq 0 ]; then
    echo "✅ Scene 2 'The Hunt Begins' created"
else
    echo "❌ Failed to create Scene 2"
fi

# Scene 3: Quantum Entanglement
$CLI_TOOL story scene create "Digital Shadows" "Quantum Entanglement"
if [ $? -eq 0 ]; then
    echo "✅ Scene 3 'Quantum Entanglement' created"
else
    echo "❌ Failed to create Scene 3"
fi

echo ""

# Show story overview
echo "📊 Story Overview:"
echo "=================="
$CLI_TOOL story overview

echo ""

# Show entity list
echo "🎭 Entity List:"
echo "==============="
$CLI_TOOL entity list

echo ""

# Show analytics
echo "📈 Story Analytics:"
echo "=================="
$CLI_TOOL analyze stats

echo ""

# Generate LaTeX export
echo "📝 Generating LaTeX export..."
$CLI_TOOL export latex --output shadows_in_the_chain.tex

if [ $? -eq 0 ]; then
    echo "✅ LaTeX file generated: shadows_in_the_chain.tex"
else
    echo "❌ Failed to generate LaTeX file"
fi

echo ""

# Attempt PDF generation (optional, requires LaTeX installation)
echo "📄 Attempting PDF generation..."
$CLI_TOOL export pdf --output shadows_in_the_chain.pdf

if [ $? -eq 0 ]; then
    echo "✅ PDF generated: shadows_in_the_chain.pdf"
else
    echo "ℹ️  PDF generation requires LaTeX installation (texlive-full)"
    echo "   LaTeX source is available in shadows_in_the_chain.tex"
fi

echo ""
echo "🎉 Story setup completed!"
echo ""
echo "Next steps:"
echo "- Edit scene content with: $CLI_TOOL story scene edit \"Chapter Name\" \"Scene Name\""
echo "- Add more entities with: $CLI_TOOL entity create --entity-type [type] \"Name\""
echo "- Launch interactive dashboard: $CLI_TOOL dashboard"
echo "- Get AI assistance: $CLI_TOOL ai character \"Character Name\""
echo ""
echo "Happy writing! 🖋️✨"