#!/bin/bash

# Build script for freellm C++23 project

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}Building freellm with C++23...${NC}"

# Create build directory if it doesn't exist
mkdir -p build
cd build

# Configure with Release mode for optimizations
# Dependencies (like RE2) are automatically downloaded and built via FetchContent
# This ensures ABI compatibility since everything uses the same compiler
echo -e "${BLUE}Configuring CMake (Release mode)...${NC}"
cmake .. -DCMAKE_BUILD_TYPE=Release

# Build
echo -e "${BLUE}Building project...${NC}"
cmake --build .

# Success message
echo -e "${GREEN}Build successful!${NC}"
echo -e "${GREEN}Run with: ./build/bin/freellm${NC}"
