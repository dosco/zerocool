# FreeLLM Tests

This directory contains all tests for the FreeLLM project using the [doctest](https://github.com/doctest/doctest) framework.

## Test Files

- **test_main.cpp** - Core functionality tests (tensor operations, model components, generation, etc.)
- **test_quantization.cpp** - Quantization-specific tests (Q8_0, Q4_K, linear layers, etc.)

## Running Tests

### Run all tests with CTest
```bash
cd build
ctest --output-on-failure
```

### Run tests individually
```bash
# Run quantization tests
./build/tests/test_quantization

# Run main tests
./build/tests/test_main
```

### Run specific test cases
```bash
# Run only Q8_0 tests
./build/tests/test_quantization --test-case="Q8_0 Quantization"

# Run only attention tests
./build/tests/test_main --test-case="Attention Mechanism"
```

### List all available tests
```bash
# List quantization test cases
./build/tests/test_quantization --list-test-cases

# List main test cases
./build/tests/test_main --list-test-cases
```

### Useful doctest options
```bash
# Show detailed output for each test
./build/tests/test_main --success

# Run tests matching a pattern
./build/tests/test_main --test-case="*Quantiz*"

# Get help on all options
./build/tests/test_main --help

# Run tests in a specific subcase
./build/tests/test_quantization --subcase="Q8_0 quantized linear"
```

## Test Organization

Tests are organized using doctest's `TEST_CASE` and `SUBCASE` macros:

- **TEST_CASE**: Top-level test grouping
- **SUBCASE**: Sub-tests within a TEST_CASE

Example:
```cpp
TEST_CASE("Basic Tensor Operations") {
    SUBCASE("Element-wise operations") {
        // Test add, multiply, etc.
    }

    SUBCASE("Activation functions") {
        // Test relu, silu, etc.
    }
}
```

## Adding New Tests

1. Add your test to either `test_main.cpp` or `test_quantization.cpp`
2. Use doctest macros: `TEST_CASE`, `SUBCASE`, `CHECK`, `REQUIRE`
3. Rebuild: `./build.sh`
4. Run: `cd build && ctest`

Example:
```cpp
TEST_CASE("My New Feature") {
    // Setup
    Tensor x({10}, 1.0f);

    // Test
    Tensor y = my_new_function(x);

    // Verify
    CHECK(y.size() == 10);
    CHECK(y[0] == doctest::Approx(expected_value));
}
```

## Migration from Old Test System

The tests have been migrated from the old custom testing framework to doctest:

- **Old**: `./build/bin/freellm --tests` (removed)
- **New**: `./build/tests/test_main` and `./build/tests/test_quantization`

All functionality has been preserved, but now uses a proper testing framework with better:
- Test organization and filtering
- Assertion messages
- Integration with CTest
- Standard test runner features
