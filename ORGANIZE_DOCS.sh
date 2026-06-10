#!/bin/bash
# Organize documentation into structured folders

echo "Creating documentation structure..."
mkdir -p docs/{session-notes,announcements,technical,whitepapers,releases,guides}

echo "Moving session notes and progress reports..."
mv *SESSION*.md docs/session-notes/ 2>/dev/null
mv *PROGRESS*.md docs/session-notes/ 2>/dev/null
mv *COMPLETE*.md docs/session-notes/ 2>/dev/null
mv *STATUS*.md docs/session-notes/ 2>/dev/null

echo "Moving announcements..."
mv *BITCOINTALK*.md docs/announcements/ 2>/dev/null
mv *BITCOINTALK*.bbcode docs/announcements/ 2>/dev/null
mv *DISCORD*.md docs/announcements/ 2>/dev/null
mv *ANNOUNCEMENT*.md docs/announcements/ 2>/dev/null
mv *SOCIAL*.md docs/announcements/ 2>/dev/null
mv *POST*.bbcode docs/announcements/ 2>/dev/null

echo "Moving technical documentation..."
mv *FIX*.md docs/technical/ 2>/dev/null
mv *BUG*.md docs/technical/ 2>/dev/null
mv *IMPLEMENTATION*.md docs/technical/ 2>/dev/null
mv *INTEGRATION*.md docs/technical/ 2>/dev/null
mv *ANALYSIS*.md docs/technical/ 2>/dev/null
mv *DESIGN*.md docs/technical/ 2>/dev/null
mv *PLAN*.md docs/technical/ 2>/dev/null

echo "Moving guides..."
mv *GUIDE*.md docs/guides/ 2>/dev/null
mv *QUICK_START*.md docs/guides/ 2>/dev/null
mv *DEVELOPER*.md docs/guides/ 2>/dev/null

echo "Moving release notes..."
mv *RELEASE*.md docs/releases/ 2>/dev/null
mv *V0.*.md docs/releases/ 2>/dev/null
mv *ROADMAP*.md docs/releases/ 2>/dev/null

echo "Moving whitepapers and research..."
mv *WHITEPAPER*.md docs/whitepapers/ 2>/dev/null
mv *HALVING*.md docs/whitepapers/ 2>/dev/null
mv *ECONOMICS*.md docs/whitepapers/ 2>/dev/null

echo "✅ Documentation organized!"
echo ""
echo "Summary:"
echo "  Session Notes: docs/session-notes/"
echo "  Announcements: docs/announcements/"
echo "  Technical: docs/technical/"
echo "  Guides: docs/guides/"
echo "  Releases: docs/releases/"
echo "  Whitepapers: docs/whitepapers/"
