#!/usr/bin/env python3
"""Comprehensive fix for all LaTeX formatting issues in chapter5_content.tex"""

import re

def fix_latex_comprehensive(content):
    """Fix all LaTeX formatting issues comprehensively"""

    # Fix trailing \textit{ at end of lines (should be removed)
    content = re.sub(r'\\textit\{\s*$', '', content, flags=re.MULTILINE)

    # Fix broken \begin{center}...\end{center\textit{
    content = content.replace(r'\end{center\textit{', r'\end{center}')

    # Fix broken subsection labels with trailing \textit{
    content = re.sub(r'(\\subsection\{[^}]+\}\\label\{[^}]+\})\\textit\{', r'\1', content)

    # Fix patterns like: }text\textit{ -> \textit{text}
    content = re.sub(r'\}([^}]+?)\\textit\{', r'\\textit{\1}', content)

    # Fix patterns like: }text} -> \textit{text}
    content = re.sub(r'\}([^}]+?\.)$', r'\1', content, flags=re.MULTILINE)

    # Fix line-final patterns: text.\textit{ -> \textit{text.}
    content = re.sub(r'([A-Z][^.]+\.)\\textit\{$', r'\\textit{\1}', content, flags=re.MULTILINE)

    # Fix blockquote > symbols (remove them, they're not LaTeX)
    lines = content.split('\n')
    fixed_lines = []
    for line in lines:
        # Remove standalone > at start of line
        if line.strip() == '>':
            continue
        fixed_lines.append(line)
    content = '\n'.join(fixed_lines)

    # Fix any remaining malformed italic patterns at line start
    content = re.sub(r'^([A-Z][^:]+:.*?)\\textit\{$', r'\\textit{\1}', content, flags=re.MULTILINE)

    # Fix patterns where text should be italic but closing brace is wrong
    # Pattern: PHOENIX: text.\textit{ -> \textit{PHOENIX: text.}
    content = re.sub(r'^([A-Z]+:.*?)\\textit\{$', r'\\textit{\1}', content, flags=re.MULTILINE)

    # Fix standalone words at start of line followed by \textit{
    content = re.sub(r'^([A-Z][a-z]+)\\textit\{', r'\\textit{\1}', content, flags=re.MULTILINE)

    return content

def main():
    input_file = '/opt/orobit/shared/q-narwhalknight/shadowchain-writer/chapter5_content.tex'

    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()

    print(f"Original length: {len(content)} bytes")

    fixed_content = fix_latex_comprehensive(content)

    with open(input_file, 'w', encoding='utf-8') as f:
        f.write(fixed_content)

    print(f"Fixed length: {len(fixed_content)} bytes")
    print("LaTeX formatting fixes applied")

if __name__ == '__main__':
    main()
