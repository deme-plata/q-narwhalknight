#!/bin/bash
# LLD-Link wrapper that uses response files to avoid Windows command-line length limits

# Create a temporary response file
RSP_FILE=$(mktemp /tmp/lld-link-args.XXXXXX.rsp)

# Write all arguments to the response file, one per line
for arg in "$@"; do
    echo "$arg" >> "$RSP_FILE"
done

# Call lld-link with the response file
lld-link "@$RSP_FILE"
RESULT=$?

# Clean up
rm -f "$RSP_FILE"

exit $RESULT
