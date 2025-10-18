#!/usr/bin/env python3
"""Fix broken LaTeX italic tags in chapter5_content.tex"""

import re

def fix_latex_italics(content):
    """Fix all broken italic tag patterns"""

    # Pattern 1: Fix }text\textit{ -> \textit{text}
    content = re.sub(r'\}([^}]+?)\\textit\{', r'\\textit{\1}', content)

    # Pattern 2: Fix }\textit{Text}\textit{ -> \textit{Text}
    content = re.sub(r'\}\\textit\{([^}]+?)\}\\textit\{', r'\\textit{\1}', content)

    # Pattern 3: Convert ## Epilogue to \subsection
    content = content.replace('\\#\\# Epilogue: The First Domino',
                              '\\subsection{Epilogue: The First Domino}\\label{epilogue-first-domino}')

    # Pattern 4: Remove blockquote > markers that shouldn't be in LaTeX
    content = re.sub(r'\\textit\{([^}]*?)\}\\textit\{\n>', r'\\textit{\1}\n\n', content)

    # Pattern 5: Fix standalone } at start of lines
    content = re.sub(r'^}\s*([A-Z])', r'\1', content, flags=re.MULTILINE)

    return content

def main():
    input_file = '/opt/orobit/shared/q-narwhalknight/shadowchain-writer/chapter5_content.tex'
    output_file = input_file

    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()

    original_length = len(content)
    fixed_content = fix_latex_italics(content)

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(fixed_content)

    print(f"Fixed LaTeX italics in {input_file}")
    print(f"Original length: {original_length} bytes")
    print(f"Fixed length: {len(fixed_content)} bytes")

if __name__ == '__main__':
    main()
