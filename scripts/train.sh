#!/bin/bash
set -e

PROGRAM_NAME=$1
DEBUG_DIR="./files/debug"
mkdir -p "$DEBUG_DIR"

if [ -z "$PROGRAM_NAME" ]; then
    echo "Error: Program name required" > "$DEBUG_DIR/training_error.txt"
    exit 1
fi

PROGRAM_FILE="programs/generated_program.py"
if [ ! -f "$PROGRAM_FILE" ]; then
    echo "Error: Program file not found: $PROGRAM_FILE" > "$DEBUG_DIR/training_error.txt"
    exit 1
fi

# Validate syntax
if ! python -m py_compile "$PROGRAM_FILE" 2>"$DEBUG_DIR/training_error.txt"; then
    echo "Syntax error in $PROGRAM_FILE" >> "$DEBUG_DIR/training_error.txt"
    exit 1
fi

# Run actual training (replace this dummy with real training)
echo "Training $PROGRAM_NAME..." > "$DEBUG_DIR/training_error.txt"

# Dummy training loop - replace with real training
python3 -c "
import sys
sys.path.insert(0, 'programs')
try:
    import torch
    from ${PROGRAM_NAME} import DeltaNet
    model = DeltaNet(hidden_size=256, num_heads=4)
    # Simple forward pass test
    x = torch.randn(1, 32, 256)
    output, _, _ = model(x)
    print('Training simulation completed')
except Exception as e:
    print(f'Training failed: {e}')
    import traceback
    traceback.print_exc()
    sys.exit(1)
" 2>>"$DEBUG_DIR/training_error.txt"

if [ $? -eq 0 ]; then
    echo "Training completed successfully for $PROGRAM_NAME"
    echo "" > "$DEBUG_DIR/training_error.txt"  # Clear debug file on success
else
    echo "Training failed for $PROGRAM_NAME"
    exit 1
fi
