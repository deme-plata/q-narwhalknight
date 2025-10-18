#!/usr/bin/env python3
"""Convert Chapter 5 markdown to LaTeX format matching existing chapters."""

import re

def convert_markdown_to_latex(md_file, tex_file):
    with open(md_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # Remove the statistics section at the end
    content = re.sub(r'\*\*Chapter 5 Statistics:.*$', '', content, flags=re.DOTALL)

    # Convert chapter title
    content = re.sub(r'^# (Chapter \d+:.*?)$', r'\\hypertarget{chapter-5-the-moscow-gambit}{%\n\\section{\1}\\label{chapter-5-the-moscow-gambit}}', content, flags=re.MULTILINE)

    # Convert scene headings (## Scene X: Title)
    def scene_replacer(match):
        scene_num = match.group(1)
        title = match.group(2)
        label = title.lower().replace(' ', '-').replace('\'', '')
        return f'\\hypertarget{{scene-{scene_num}-{label}}}{{%\n\\subsection{{Scene {scene_num}: {title}}}\\label{{scene-{scene_num}-{label}}}}}'

    content = re.sub(r'^## (Scene \d+): (.*?)$', scene_replacer, content, flags=re.MULTILINE)

    # Convert horizontal rules to LaTeX
    content = re.sub(r'^---+$', r'\\begin{center}\\rule{0.5\\linewidth}{0.5pt}\\end{center}', content, flags=re.MULTILINE)

    # Convert *italic* to \textit{italic}
    content = re.sub(r'\*([^*]+?)\*', r'\\textit{\1}', content)

    # Convert **bold** to \textbf{bold}
    content = re.sub(r'\*\*([^*]+?)\*\*', r'\\textbf{\1}', content)

    # Convert blockquotes (>) to quoted blocks
    content = re.sub(r'^> \*\*(.*?):\*\*(.*)$', r'\\textbf{\1}:\2', content, flags=re.MULTILINE)
    content = re.sub(r'^> (.*)$', r'\1', content, flags=re.MULTILINE)

    # Escape special LaTeX characters (but not backslashes we just added)
    def escape_latex(text):
        # Only escape if not already escaped
        text = re.sub(r'(?<!\\)&', r'\\&', text)
        text = re.sub(r'(?<!\\)%', r'\\%', text)
        text = re.sub(r'(?<!\\)\$', r'\\$', text)
        text = re.sub(r'(?<!\\)#', r'\\#', text)
        text = re.sub(r'(?<!\\)_', r'\\_', text)
        return text

    # Apply escaping carefully
    lines = content.split('\n')
    escaped_lines = []
    for line in lines:
        # Don't escape lines that are LaTeX commands
        if line.strip().startswith('\\'):
            escaped_lines.append(line)
        else:
            escaped_lines.append(escape_latex(line))

    content = '\n'.join(escaped_lines)

    # Convert em dashes (—) to LaTeX
    content = content.replace('—', '---')
    content = content.replace('–', '--')

    # Convert quotes
    content = re.sub(r'"([^"]+)"', r'``\1''', content)

    # Remove *End of Scene* markers
    content = re.sub(r'\n\n\*End of .*?\*\n', '', content)

    # Clean up multiple blank lines
    content = re.sub(r'\n{3,}', '\n\n', content)

    with open(tex_file, 'w', encoding='utf-8') as f:
        f.write(content)

    print(f"Converted {md_file} -> {tex_file}")

if __name__ == '__main__':
    convert_markdown_to_latex('chapter_five.md', 'chapter5_content.tex')
