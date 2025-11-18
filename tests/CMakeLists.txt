# Tests configuration using doctest framework

# Quantization API unit tests
add_executable(test_quantization
    test_quantization.cpp
    ${CMAKE_SOURCE_DIR}/src/tensor.cpp
)

# Link required libraries
target_link_libraries(test_quantization PRIVATE sentencepiece-static)

# Include directories (include external for doctest.h)
target_include_directories(test_quantization PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/external
    ${sentencepiece_SOURCE_DIR}/src
)

# Set C++ standard for test
set_target_properties(test_quantization PROPERTIES
    CXX_STANDARD 23
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
)

# Main tests (from main.cpp)
add_executable(test_main
    test_main.cpp
    ${CMAKE_SOURCE_DIR}/src/tensor.cpp
)

# Link required libraries
target_link_libraries(test_main PRIVATE sentencepiece-static)

# Include directories (include external for doctest.h)
target_include_directories(test_main PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/external
    ${sentencepiece_SOURCE_DIR}/src
)

# Set C++ standard for test
set_target_properties(test_main PROPERTIES
    CXX_STANDARD 23
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
)

# Add tests to CTest
enable_testing()
add_test(NAME QuantizationTests COMMAND test_quantization)
add_test(NAME MainTests COMMAND test_main)

message(STATUS "Configured tests with doctest framework:")
message(STATUS "  - test_quantization")
message(STATUS "  - test_main")
