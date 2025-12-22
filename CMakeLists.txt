cmake_minimum_required(VERSION 3.20)

# Project configuration
project(freellm
    VERSION 0.1.0
    DESCRIPTION "A modern C++ project"
    LANGUAGES CXX
)

# Set C++ standard
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Export compile commands for tools like clangd
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Add compiler warnings
# Note: We'll add these to our source files only, not to third-party libraries
set(PROJECT_WARNINGS)
if(MSVC)
    list(APPEND PROJECT_WARNINGS /W4 /WX)
else()
    list(APPEND PROJECT_WARNINGS -Wall -Wextra -Wpedantic -Werror)
endif()

# SIMD optimizations
if(NOT MSVC)
    # Enable AVX2 and FMA for x86/x64 targets
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "(x86)|(X86)|(amd64)|(AMD64)")
        add_compile_options(-mavx2 -mfma)
        message(STATUS "Enabling AVX2 and FMA optimizations")
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64)")
        message(STATUS "Enabling ARM64 NEON optimizations")
        # NEON is enabled by default on ARM64, but -mcpu=native optimizes for the specific chip
        if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|AppleClang")
            add_compile_options(-mcpu=native)
        elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
            add_compile_options(-march=native)
        endif()
    endif()

    # Enable C++20 experimental library features (needed for jthread on libc++ sometimes)
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|AppleClang")
        add_compile_options(-fexperimental-library)
    endif()
endif()

# Include directories
include_directories(${PROJECT_SOURCE_DIR}/include)

# Use FetchContent to download and build RE2 automatically
# This avoids requiring users to install RE2 manually and ensures ABI compatibility
# NOTE: Currently commented out as RE2 is not used by SentencePiece tokenizer.
# Uncomment if implementing HuggingFace tokenizers in the future (they require RE2 for regex pre-tokenization)
include(FetchContent)

# FetchContent_Declare(
#     re2
#     GIT_REPOSITORY https://github.com/google/re2.git
#     GIT_TAG        2024-07-02
#     GIT_SHALLOW    TRUE
# )
#
# # Disable RE2 tests to speed up build
# set(RE2_BUILD_TESTING OFF CACHE BOOL "" FORCE)
#
# # Download and build RE2 (only downloads/builds once, then caches)
# FetchContent_MakeAvailable(re2)
#
# message(STATUS "RE2 will be built from source (no manual installation needed)")

FetchContent_Declare(
    re2
    GIT_REPOSITORY https://github.com/google/re2.git
    GIT_TAG        2024-07-02
    GIT_SHALLOW    TRUE
)

# Disable RE2 tests to speed up build
set(RE2_BUILD_TESTING OFF CACHE BOOL "" FORCE)

# Download and build RE2
FetchContent_MakeAvailable(re2)

# Fetch nlohmann/json
FetchContent_Declare(
    json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG        v3.11.3
    GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(json)

# Fetch SentencePiece for tokenization
FetchContent_Declare(
    sentencepiece
    GIT_REPOSITORY https://github.com/google/sentencepiece.git
    GIT_TAG        v0.2.0
    GIT_SHALLOW    TRUE
    PATCH_COMMAND
        ${CMAKE_COMMAND}
            -Dsentencepiece_SOURCE_DIR=<SOURCE_DIR>
            -P ${CMAKE_SOURCE_DIR}/cmake/patch_sentencepiece.cmake
)

# Disable SentencePiece features we don't need
set(SPM_ENABLE_SHARED OFF CACHE BOOL "" FORCE)
set(SPM_ENABLE_TCMALLOC OFF CACHE BOOL "" FORCE)
set(SPM_ENABLE_TENSORFLOW OFF CACHE BOOL "" FORCE)
set(SPM_ENABLE_NFKC_COMPILE OFF CACHE BOOL "" FORCE)

# Download, patch, and add SentencePiece in a single step
FetchContent_MakeAvailable(sentencepiece)

message(STATUS "SentencePiece will be built from source (no manual installation needed)")

# Source files
set(SOURCES
    # src/main.cpp - Moved to executable target
    src/core/tensor.cpp
    src/infra/parallel_tensor_loader.cpp
    src/infra/compute_backend.cpp
    src/infra/cuda_backend.cpp
    src/infra/pjrt_backend.cpp
    src/core/scheduler.cpp
    src/core/generation_engine.cpp
    src/core/bpe_tokenizer.cpp
)

if(APPLE)
    list(APPEND SOURCES src/infra/metal_backend.mm)
    set_source_files_properties(src/infra/metal_backend.mm PROPERTIES COMPILE_FLAGS "-fobjc-arc")
    message(STATUS "Metal backend enabled (macOS detected)")
endif()

# Find CUDA Toolkit
find_package(CUDAToolkit)
if(CUDAToolkit_FOUND)
    message(STATUS "CUDA Toolkit found")
    add_compile_definitions(FREELM_ENABLE_CUDA)
else()
    message(WARNING "CUDA Toolkit not found - CUDA backend will be disabled")
endif()

# Create executable
# Create library
add_library(freellm_lib STATIC ${SOURCES})

# Apply warnings only to our source files, not third-party libraries
target_compile_options(freellm_lib PRIVATE ${PROJECT_WARNINGS})

# Link libraries - CMake handles include dirs automatically
target_link_libraries(freellm_lib PRIVATE
    re2::re2
    nlohmann_json::nlohmann_json
    sentencepiece-static
    ${CMAKE_DL_LIBS}
)

if(APPLE)
    find_library(METAL_LIB Metal)
    find_library(MPS_LIB MetalPerformanceShaders)
    find_library(FOUNDATION_LIB Foundation)
    target_link_libraries(freellm_lib PRIVATE ${METAL_LIB} ${MPS_LIB} ${FOUNDATION_LIB})
endif()

if(CUDAToolkit_FOUND)
    target_link_libraries(freellm_lib PRIVATE CUDA::cudart CUDA::cuda_driver)
endif()
message(STATUS "Linking SentencePiece library for tokenizer")

# Ensure SentencePiece headers are available to this target and consumers
target_include_directories(freellm_lib PUBLIC
    ${sentencepiece_SOURCE_DIR}/src
)

# Create executable
add_executable(${PROJECT_NAME} src/main.cpp)
target_link_libraries(${PROJECT_NAME} PRIVATE freellm_lib)

# Set output directory
set_target_properties(${PROJECT_NAME} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# Optional: Enable testing
enable_testing()
add_subdirectory(tests)

# Metal-specific examples
if(APPLE)


    # Continuous Batching Example
    add_executable(continuous_batching src/examples/continuous_batching.cpp)
    target_link_libraries(continuous_batching PRIVATE freellm_lib)
    set_target_properties(continuous_batching PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    )

    # Simple Chat Example (using GenerationEngine)
    add_executable(simple_chat src/examples/simple_chat.cpp)
    target_link_libraries(simple_chat PRIVATE freellm_lib)
    set_target_properties(simple_chat PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    )
endif()
