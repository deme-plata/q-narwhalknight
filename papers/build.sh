#!/bin/bash
# Build LaTeX paper to PDF

set -e

echo "Building Q-NarwhalKnight Quantum Aesthetics Paper..."

# Check if pdflatex is available
if ! command -v pdflatex &> /dev/null; then
    echo "Error: pdflatex is not installed. Please install a LaTeX distribution."
    echo "Ubuntu/Debian: sudo apt-get install texlive-latex-base texlive-latex-extra"
    echo "macOS: brew install mactex"
    exit 1
fi

# Create output directory
mkdir -p output

# Build the PDF (run multiple times for references)
echo "Running pdflatex (pass 1/3)..."
pdflatex -output-directory=output -interaction=nonstopmode quantum-aesthetics.tex

echo "Running pdflatex (pass 2/3)..."
pdflatex -output-directory=output -interaction=nonstopmode quantum-aesthetics.tex

echo "Running pdflatex (pass 3/3)..."
pdflatex -output-directory=output -interaction=nonstopmode quantum-aesthetics.tex

# Copy final PDF to main directory
cp output/quantum-aesthetics.pdf quantum-aesthetics.pdf

echo "✓ Paper built successfully!"
echo "📄 Output: quantum-aesthetics.pdf"
echo "📊 Size: $(du -h quantum-aesthetics.pdf | cut -f1)"
echo ""
echo "Open with:"
echo "  macOS:   open quantum-aesthetics.pdf"
echo "  Linux:   xdg-open quantum-aesthetics.pdf"
echo "  Windows: start quantum-aesthetics.pdf"