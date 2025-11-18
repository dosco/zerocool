#!/bin/bash

# Download TinyLLaMA-1.1B weights from HuggingFace
# This script downloads the safetensors format weights

set -e  # Exit on error

echo "========================================="
echo "Downloading TinyLLaMA-1.1B-Chat-v1.0"
echo "========================================="
echo ""

# Create models directory
MODEL_DIR="models/tinyllama"
mkdir -p "$MODEL_DIR"

echo "Model will be saved to: $MODEL_DIR"
echo ""

# Check if huggingface is installed
if command -v huggingface &> /dev/null; then
    echo "Using huggingface CLI..."
    echo ""

    # Download using HuggingFace CLI (fastest method)
    huggingface download TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
        --to "$MODEL_DIR" \
        --filter "*.safetensors" \
        --filter "config.json" \
        --filter "tokenizer.json" \
        --filter "tokenizer_config.json" \
        --filter "tokenizer.model"

    echo ""
    echo "✓ Download complete!"

else
    echo "huggingface CLI not found."
    echo "Installing huggingface_hub..."
    echo ""

    pip3 install --break-system-packages huggingface-hub

    echo ""
    echo "Downloading model files..."
    echo ""

    # Try again with newly installed CLI
    huggingface download TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
        --to "$MODEL_DIR" \
        --filter "*.safetensors" \
        --filter "config.json" \
        --filter "tokenizer.json" \
        --filter "tokenizer_config.json" \
        --filter "tokenizer.model"

    echo ""
    echo "✓ Download complete!"
fi

echo ""
echo "========================================="
echo "Downloaded files:"
echo "========================================="
ls -lh "$MODEL_DIR"

echo ""
echo "Model ready at: $MODEL_DIR"
echo ""
echo "You can now run: ./build/bin/freellm"
