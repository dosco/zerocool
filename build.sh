#!/bin/bash

# Build script for freellm C++23 project

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}Building freellm with C++23...${NC}"

# Create build directory if it doesn't exist
cd "$(dirname "$0")"
TASK_BUILD_DIR="${FREELLM_BUILD_DIR:-build/qwen}"
mkdir -p "$TASK_BUILD_DIR"

# Configure with Release mode for optimizations
# Dependencies (like RE2) are automatically downloaded and built via FetchContent
# This ensures ABI compatibility since everything uses the same compiler
echo -e "${BLUE}Configuring CMake (Release mode)...${NC}"
cmake -S . -B "$TASK_BUILD_DIR" -DCMAKE_BUILD_TYPE=Release "$@"

# Build
echo -e "${BLUE}Building project...${NC}"
cmake --build "$TASK_BUILD_DIR" --parallel "${FREELLM_BUILD_JOBS:-4}"

# Success message
echo -e "${GREEN}Build successful!${NC}"
echo -e "${GREEN}Run with: $TASK_BUILD_DIR/bin/freellm${NC}"
