#!/bin/bash
set -e

echo "========================================"
echo "FreeLLM Continuous Batching Demo (Lifecycle)"
echo "========================================"
echo "Building..."

# Build the project (including tests)
./build.sh

echo ""
echo "Running Paged KV Cache Lifecycle Test..."
echo "This demonstrates the core memory management for continuous batching:"
echo "1. Dynamic Block Allocation"
echo "2. Sequence-based Ownership"
echo "3. Automatic Block Reuse upon Sequence Completion"
echo "----------------------------------------"

# Run the lifecycle test
./build/tests/test_paged_kv_lifecycle

echo ""
echo "========================================"
echo "Demo Complete"
echo "========================================"
