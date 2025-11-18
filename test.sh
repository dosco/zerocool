#!/bin/bash

# Exit on error
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}FreeLLM Test Runner${NC}"
echo -e "${BLUE}===================${NC}\n"

# Check if build directory exists
if [ ! -d "build" ]; then
    echo -e "${YELLOW}Build directory not found. Running initial build...${NC}\n"
    ./build.sh
fi

# Build tests
echo -e "${BLUE}Building tests...${NC}"
cd build
cmake --build . --target test_main test_quantization

echo -e "\n${BLUE}Running tests with CTest...${NC}"
echo -e "${BLUE}============================${NC}\n"

# Run tests with output on failure
if ctest --output-on-failure; then
    echo -e "\n${GREEN}✓ All tests passed!${NC}\n"

    # Show test summary
    echo -e "${BLUE}Test Summary:${NC}"
    echo -e "${BLUE}=============${NC}"
    ctest --quiet

    echo -e "\n${GREEN}Success! All tests are working correctly.${NC}\n"

    # Show additional options
    echo -e "${BLUE}Additional test commands:${NC}"
    echo -e "  Run quantization tests:  ${YELLOW}./build/tests/test_quantization${NC}"
    echo -e "  Run main tests:          ${YELLOW}./build/tests/test_main${NC}"
    echo -e "  List all test cases:     ${YELLOW}./build/tests/test_main --list-test-cases${NC}"
    echo -e "  Run specific test:       ${YELLOW}./build/tests/test_main --test-case=\"KV Cache\"${NC}"
    echo -e "  Show detailed output:    ${YELLOW}./build/tests/test_main --success${NC}"
    echo ""

    exit 0
else
    echo -e "\n${RED}✗ Some tests failed!${NC}\n"
    echo -e "${YELLOW}To debug:${NC}"
    echo -e "  Run tests individually:  ${YELLOW}./build/tests/test_main${NC}"
    echo -e "  Run with verbose output: ${YELLOW}./build/tests/test_main --success${NC}"
    echo ""
    exit 1
fi
