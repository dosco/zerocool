# Tests configuration using doctest framework

# Quantization API unit tests
add_executable(test_quantization
    test_quantization.cpp
    ${CMAKE_SOURCE_DIR}/src/core/tensor.cpp
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
    ${CMAKE_SOURCE_DIR}/src/core/tensor.cpp
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

# Paged KV Cache Test
add_executable(test_paged_kv_cache
    test_paged_kv_cache.cpp
    ${CMAKE_SOURCE_DIR}/src/core/tensor.cpp
)

target_link_libraries(test_paged_kv_cache PRIVATE sentencepiece-static)

target_include_directories(test_paged_kv_cache PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/external
    ${sentencepiece_SOURCE_DIR}/src
)

set_target_properties(test_paged_kv_cache PROPERTIES
    CXX_STANDARD 23
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
)

add_test(NAME PagedKVCacheTests COMMAND test_paged_kv_cache)
message(STATUS "  - test_paged_kv_cache")

# Paged KV Lifecycle Test
add_executable(test_paged_kv_lifecycle
    test_paged_kv_lifecycle.cpp
    ${CMAKE_SOURCE_DIR}/src/core/tensor.cpp
)

target_link_libraries(test_paged_kv_lifecycle PRIVATE freellm_lib)

target_include_directories(test_paged_kv_lifecycle PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/external
    ${sentencepiece_SOURCE_DIR}/src
)

set_target_properties(test_paged_kv_lifecycle PROPERTIES
    CXX_STANDARD 23
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
)

add_test(NAME PagedKVLifecycleTests COMMAND test_paged_kv_lifecycle)
message(STATUS "  - test_paged_kv_lifecycle")

# Backend tests
add_executable(test_backends test_backends.cpp)
target_link_libraries(test_backends PRIVATE freellm_lib)
target_include_directories(test_backends PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/external
)
set_target_properties(test_backends PROPERTIES
    CXX_STANDARD 23
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
)
add_test(NAME BackendTests COMMAND test_backends)
message(STATUS "  - test_backends")

# Q4_0 Implementation Unit Test (My implementation)
add_executable(test_q4_0_impl
    test_q4_0_impl.cpp
)
target_link_libraries(test_q4_0_impl PRIVATE sentencepiece-static)
target_include_directories(test_q4_0_impl PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/external
)
set_target_properties(test_q4_0_impl PROPERTIES
    CXX_STANDARD 23
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
)
add_test(NAME Q4_0ImplTests COMMAND test_q4_0_impl)
message(STATUS "  - test_q4_0_impl")

# Debug Q4_0 Numerical Correctness (Metal vs CPU)
add_executable(debug_q4_0
    debug_q4_0.cpp
)
target_link_libraries(debug_q4_0 PRIVATE freellm_lib)
target_include_directories(debug_q4_0 PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/external
)
set_target_properties(debug_q4_0 PROPERTIES
    CXX_STANDARD 23
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
)
# Note: debug_q4_0 is run manually, not part of CTest suite usually, unless we want it to fail build?
# Let's add it to CTest for convenience.
add_test(NAME DebugQ4_0 COMMAND debug_q4_0)
message(STATUS "  - debug_q4_0")

# Debug RoPE correctness
add_executable(debug_rope
    debug_rope.cpp
)
target_link_libraries(debug_rope PRIVATE freellm_lib)
target_include_directories(debug_rope PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/external
)
set_target_properties(debug_rope PROPERTIES
    CXX_STANDARD 23
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
)
add_test(NAME DebugRoPE COMMAND debug_rope)
message(STATUS "  - debug_rope")

# Int8 KV Cache Quantization
add_executable(test_kv_quantization
    test_kv_quantization.cpp
)
target_link_libraries(test_kv_quantization PRIVATE freellm_lib)
target_include_directories(test_kv_quantization PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/external
)
set_target_properties(test_kv_quantization PROPERTIES
    CXX_STANDARD 23
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
)
add_test(NAME KVQuantizationTests COMMAND test_kv_quantization)
message(STATUS "  - test_kv_quantization")

# BPE Tokenizer Test
add_executable(test_bpe_tokenizer
    test_bpe_tokenizer.cpp
)
target_link_libraries(test_bpe_tokenizer PRIVATE freellm_lib)
target_include_directories(test_bpe_tokenizer PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/external
)
set_target_properties(test_bpe_tokenizer PROPERTIES
    CXX_STANDARD 23
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
)
add_test(NAME BPETokenizerTests COMMAND test_bpe_tokenizer)
message(STATUS "  - test_bpe_tokenizer")
