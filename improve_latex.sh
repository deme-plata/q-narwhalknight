#!/bin/bash

# Re-convert with XeLaTeX support for Unicode
pandoc ./PRIVACY_AS_A_SERVICE_WHITEPAPER.md \
  -o ./PRIVACY_AS_A_SERVICE_WHITEPAPER_v2.tex \
  --from markdown \
  --to latex \
  --standalone \
  --toc \
  --toc-depth=3 \
  --number-sections \
  --highlight-style=tango \
  -V documentclass=article \
  -V geometry:margin=1in \
  -V fontsize=11pt \
  -V colorlinks=true \
  -V linkcolor=blue \
  -V urlcolor=blue \
  -V citecolor=blue \
  -V mainfont="DejaVu Sans" \
  --pdf-engine=xelatex

echo "✓ Improved LaTeX with XeLaTeX generated"
