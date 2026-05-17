#!/bin/bash

# Define variables
BASE_DIR="/home/myuser/viper/dagknight-vm"
OUTPUT_FILE="source_code_compilation.md"
TIMESTAMP=$(date "+%Y-%m-%d %H:%M:%S")

# Check if base directory exists
if [ ! -d "$BASE_DIR" ]; then
    echo "Error: Directory $BASE_DIR does not exist."
    exit 1
fi

# Check if src directory exists
if [ ! -d "$BASE_DIR/src" ]; then
    echo "Error: Directory $BASE_DIR/src does not exist."
    exit 1
fi

# Check if Cargo.toml exists
if [ ! -f "$BASE_DIR/Cargo.toml" ]; then
    echo "Error: File $BASE_DIR/Cargo.toml does not exist."
    exit 1
fi

# Create or overwrite output file with header
cat > "$OUTPUT_FILE" << EOF
# Source Code Compilation from $BASE_DIR
Generated on: $TIMESTAMP

## Table of Contents

EOF

# Function to get appropriate language for syntax highlighting
get_language() {
    local file_ext="${1##*.}"
    case "$file_ext" in
        py) echo "python" ;;
        js) echo "javascript" ;;
        java) echo "java" ;;
        cpp|cc) echo "cpp" ;;
        c) echo "c" ;;
        h) echo "c" ;;
        hpp) echo "cpp" ;;
        sh) echo "bash" ;;
        rb) echo "ruby" ;;
        pl) echo "perl" ;;
        php) echo "php" ;;
        html) echo "html" ;;
        css) echo "css" ;;
        xml) echo "xml" ;;
        json) echo "json" ;;
        md) echo "markdown" ;;
        go) echo "go" ;;
        rs) echo "rust" ;;
        ts) echo "typescript" ;;
        hs) echo "haskell" ;;
        lua) echo "lua" ;;
        sql) echo "sql" ;;
        swift) echo "swift" ;;
        kt|kts) echo "kotlin" ;;
        scala) echo "scala" ;;
        yaml|yml) echo "yaml" ;;
        toml) echo "toml" ;;
        *) echo "text" ;;  # Default to plain text
    esac
}

# First pass: create table of contents
echo "Building table of contents..."
file_count=0

# Only process src folder and Cargo.toml
(find "$BASE_DIR/src" -type f -not -path "*/\.*"; echo "$BASE_DIR/Cargo.toml") | sort | while read -r file; do
    # Skip binary files and large files (>1MB)
    if file "$file" | grep -q "binary"; then
        continue
    fi
    
    file_size=$(stat -c %s "$file" 2>/dev/null || stat -f %z "$file" 2>/dev/null)
    if [ "$file_size" -gt 1048576 ]; then
        continue
    fi
    
    # Get relative path for cleaner display
    rel_path="${file#$BASE_DIR/}"
    
    # Add to table of contents
    echo "- [${rel_path}](#${rel_path//\//-})" >> "$OUTPUT_FILE"
    file_count=$((file_count + 1))
done

# Second pass: add file contents
echo "Adding source code from $file_count files..."

# Only process src folder and Cargo.toml
(find "$BASE_DIR/src" -type f -not -path "*/\.*"; echo "$BASE_DIR/Cargo.toml") | sort | while read -r file; do
    # Skip binary files and large files (>1MB)
    if file "$file" | grep -q "binary"; then
        continue
    fi
    
    file_size=$(stat -c %s "$file" 2>/dev/null || stat -f %z "$file" 2>/dev/null)
    if [ "$file_size" -gt 1048576 ]; then
        continue
    fi
    
    # Get relative path for display
    rel_path="${file#$BASE_DIR/}"
    
    # Determine language for syntax highlighting
    lang=$(get_language "$file")
    
    # Add file header and content to markdown
    echo -e "\n## ${rel_path}\n" >> "$OUTPUT_FILE"
    echo "### File path: \`$file\`" >> "$OUTPUT_FILE"
    echo -e "\n\`\`\`$lang" >> "$OUTPUT_FILE"
    cat "$file" >> "$OUTPUT_FILE"
    echo -e "\`\`\`\n" >> "$OUTPUT_FILE"
    
    # Add a separator
    echo -e "---\n" >> "$OUTPUT_FILE"
done

echo "Markdown file created: $OUTPUT_FILE"
echo "Contains source code from $file_count files in $BASE_DIR"
