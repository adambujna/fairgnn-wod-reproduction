#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"


if command -v wget &> /dev/null; then
    CMD() { wget "$1" -O "$2"; }
elif command -v curl &> /dev/null; then
    CMD() { curl -L "$1" -o "$2"; }
else
    echo "Please install wget or curl to download the checkpoints."
    exit 1
fi

ZIP_FILE="https://huggingface.co/AdamB2/fact-gnn-wod-ckpts/resolve/main/checkpoints.zip"


echo "Downloading checkpoints..."
CMD "$ZIP_FILE" "checkpoints.zip" || { echo "Failed to download dataset from $ZIP_FILE"; exit 1; }

echo "Successfully downloaded checkpoints."
