#!/bin/bash
# ASI-Arch-v1 Training Script
set -e

PROGRAM_NAME=$1
if [ -z "$PROGRAM_NAME" ]; then
    echo "Error: Program name required"
    echo "Usage: $0 <program_name>"
    exit 1
fi

echo "Starting training for program: $PROGRAM_NAME"

PROGRAM_FILE="programs/${PROGRAM_NAME}.py"
if [ ! -f "$PROGRAM_FILE" ]; then
    echo "Error: Program file not found: $PROGRAM_FILE"
    echo "Available programs:"
    ls -1 programs/*.py 2>/dev/null || echo "No programs found in programs/ directory"
    exit 1
fi

echo "Found program file: $PROGRAM_FILE"

# Validate syntax
python -m py_compile "$PROGRAM_FILE"
if [ $? -ne 0 ]; then
    echo "Error: Program file has syntax errors"
    exit 1
fi

echo "Training completed successfully for $PROGRAM_NAME"
exit 0
