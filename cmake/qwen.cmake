set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
if(NOT APPLE)
    message(FATAL_ERROR "ZeroCool's native engine requires Apple Silicon/macOS")
endif()
enable_language(OBJCXX)
set(CMAKE_OBJCXX_STANDARD 23)
set(CMAKE_OBJCXX_STANDARD_REQUIRED ON)

# One warning policy for every target built from this repository, including
# tests and diagnostics. Fetched dependencies keep their own settings.
add_library(zerocool_warnings INTERFACE)
target_compile_options(zerocool_warnings INTERFACE -Wall -Wextra -Wpedantic -Werror)

# Arithmetic must stay reproducible; -ffast-math would silently change results.
add_library(zerocool_arithmetic INTERFACE)
target_compile_options(zerocool_arithmetic INTERFACE -fno-fast-math)

set(ZEROCOOL_SANITIZE "none" CACHE STRING "Sanitizer for local checks: none|address|undefined|thread")
set_property(CACHE ZEROCOOL_SANITIZE PROPERTY STRINGS none address undefined thread)
if(NOT ZEROCOOL_SANITIZE STREQUAL "none")
    # Sanitized builds are for tests and diagnosis; they never produce evidence.
    add_compile_options(-fsanitize=${ZEROCOOL_SANITIZE} -fno-omit-frame-pointer -g)
    add_link_options(-fsanitize=${ZEROCOOL_SANITIZE})
    message(STATUS "ZeroCool: ${ZEROCOOL_SANITIZE} sanitizer enabled; timings from this build are not evidence.")
endif()

include(FetchContent)
# Commit pins, not tags: a moved tag would change the engine without changing
# the recorded build identity.
FetchContent_Declare(json GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG 9cca280a4d0ccf0c08f47a99aa71d1b0e52f8d03) # v3.11.3
FetchContent_MakeAvailable(json)
set(MINJA_TEST_ENABLED OFF CACHE BOOL "" FORCE)
set(MINJA_EXAMPLE_ENABLED OFF CACHE BOOL "" FORCE)
set(MINJA_FUZZTEST_ENABLED OFF CACHE BOOL "" FORCE)
set(MINJA_USE_VENV OFF CACHE BOOL "" FORCE)
FetchContent_Declare(minja GIT_REPOSITORY https://github.com/google/minja.git GIT_TAG 021c2293c187789ef13d56c6cfd89c9b134fd80f)
FetchContent_MakeAvailable(minja)

file(READ "${CMAKE_SOURCE_DIR}/kernels/metal/qwen.metal" ZEROCOOL_METAL_SOURCE)
file(READ "${CMAKE_SOURCE_DIR}/models.lock.json" ZEROCOOL_MODEL_LOCK)
file(READ "${CMAKE_SOURCE_DIR}/mixed-models.lock.json" ZEROCOOL_MIXED_MODEL_LOCK)
file(READ "${CMAKE_SOURCE_DIR}/mixed-payload-reuse.lock.json" ZEROCOOL_MIXED_REUSE_LOCK)
file(GLOB ZEROCOOL_BUILD_INPUTS CONFIGURE_DEPENDS "${CMAKE_SOURCE_DIR}/src/engine/*" "${CMAKE_SOURCE_DIR}/include/engine/*")
list(APPEND ZEROCOOL_BUILD_INPUTS "${CMAKE_SOURCE_DIR}/kernels/metal/qwen.metal" "${CMAKE_SOURCE_DIR}/models.lock.json"
    "${CMAKE_SOURCE_DIR}/mixed-models.lock.json" "${CMAKE_SOURCE_DIR}/mixed-payload-reuse.lock.json"
    "${CMAKE_SOURCE_DIR}/cmake/qwen.cmake" "${CMAKE_SOURCE_DIR}/cmake/qwen_embedded.hpp.in" "${CMAKE_SOURCE_DIR}/CMakeLists.txt")
set(ZEROCOOL_BUILD_IDENTITY "")
foreach(input IN LISTS ZEROCOOL_BUILD_INPUTS)
    file(SHA256 "${input}" digest)
    file(RELATIVE_PATH label "${CMAKE_SOURCE_DIR}" "${input}")
    string(APPEND ZEROCOOL_BUILD_IDENTITY "${label}:${digest}\n")
endforeach()
# How the sources were compiled is part of the identity: an unoptimized or
# sanitized build must never report the same fingerprint as a measured one.
string(TOUPPER "${CMAKE_BUILD_TYPE}" ZEROCOOL_BUILD_TYPE_UPPER)
string(APPEND ZEROCOOL_BUILD_IDENTITY
    "toolchain:${CMAKE_CXX_COMPILER_ID}-${CMAKE_CXX_COMPILER_VERSION}\n"
    "build_type:${CMAKE_BUILD_TYPE}\n"
    "flags:${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_${ZEROCOOL_BUILD_TYPE_UPPER}}\n"
    "sanitizer:${ZEROCOOL_SANITIZE}\n")
string(SHA256 ZEROCOOL_BUILD_FINGERPRINT "${ZEROCOOL_BUILD_IDENTITY}")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${ZEROCOOL_BUILD_INPUTS})
configure_file(cmake/qwen_embedded.hpp.in generated/qwen_embedded.hpp @ONLY)

find_package(CURL REQUIRED)

add_library(zerocool_lib STATIC
    src/engine/fetch.cpp src/engine/storage.cpp src/engine/metal.mm src/engine/pressure.mm src/engine/model.cpp src/engine/prefill.cpp
    src/engine/pipeline.cpp src/engine/bench.cpp src/engine/kernel_bench.cpp src/engine/cached_replay.cpp
    src/engine/tokenizer.mm src/engine/session.cpp src/engine/server.cpp src/engine/chat_executor.cpp src/engine/cli.cpp)
target_include_directories(zerocool_lib PUBLIC include PRIVATE ${minja_SOURCE_DIR}/include ${CMAKE_BINARY_DIR}/generated)
target_link_libraries(zerocool_lib PUBLIC nlohmann_json::nlohmann_json PRIVATE zerocool_warnings zerocool_arithmetic CURL::libcurl
    "-framework Metal" "-framework Foundation" "-framework IOKit")
set_source_files_properties(src/engine/metal.mm src/engine/pressure.mm src/engine/tokenizer.mm PROPERTIES COMPILE_FLAGS "-fobjc-arc")

add_executable(zerocool src/engine/main.cpp)
target_link_libraries(zerocool PRIVATE zerocool_lib zerocool_warnings zerocool_arithmetic)

option(ZEROCOOL_BUILD_TUI "Build the local terminal chat client" ON)
if(ZEROCOOL_BUILD_TUI)
    set(FTXUI_BUILD_DOCS OFF CACHE BOOL "" FORCE)
    set(FTXUI_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(FTXUI_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(FTXUI_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
    FetchContent_Declare(ftxui GIT_REPOSITORY https://github.com/ArthurSonzogni/FTXUI.git
        GIT_TAG f921fad208912747c17d129a8ef75ec7624b6eec) # v7.0.3
    FetchContent_MakeAvailable(ftxui)
    add_library(zerocool_chat_client STATIC src/engine/chat_client.cpp)
    target_include_directories(zerocool_chat_client PUBLIC include)
    target_link_libraries(zerocool_chat_client PUBLIC nlohmann_json::nlohmann_json PRIVATE zerocool_warnings CURL::libcurl)
    target_sources(zerocool PRIVATE src/engine/chat.cpp)
    target_compile_definitions(zerocool PRIVATE ZEROCOOL_WITH_TUI=1)
    target_link_libraries(zerocool PRIVATE zerocool_chat_client ftxui::component ftxui::dom ftxui::screen)
endif()
set_target_properties(zerocool PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
install(TARGETS zerocool zerocool_lib RUNTIME DESTINATION bin ARCHIVE DESTINATION lib)
install(DIRECTORY include/engine DESTINATION include)

enable_testing()
add_executable(test_qwen tests/test_qwen.cpp)
target_include_directories(test_qwen PRIVATE external)
target_link_libraries(test_qwen PRIVATE zerocool_lib zerocool_warnings zerocool_arithmetic)
add_test(NAME QwenNative COMMAND test_qwen)
set_tests_properties(QwenNative PROPERTIES TIMEOUT 900)

if(ZEROCOOL_BUILD_TUI)
    add_executable(test_chat_transport tests/test_chat_transport.cpp)
    target_include_directories(test_chat_transport PRIVATE external)
    target_link_libraries(test_chat_transport PRIVATE zerocool_lib zerocool_chat_client zerocool_warnings)
    add_test(NAME ChatTransport COMMAND test_chat_transport)
    set_tests_properties(ChatTransport PROPERTIES TIMEOUT 120)
    add_executable(chat_fixture tests/chat_fixture.cpp src/engine/chat.cpp)
    target_link_libraries(chat_fixture PRIVATE zerocool_lib zerocool_chat_client zerocool_warnings ftxui::component ftxui::dom ftxui::screen)
endif()

# The model-free Python checks are part of the same suite; without them a green
# ctest would cover only the native half of the project.
# The checks import NumPy, so prefer the repository environment created by
# scripts/qwen/setup_reference.sh over whatever interpreter is on PATH.
set(ZEROCOOL_PYTHON "${CMAKE_SOURCE_DIR}/.venv/bin/python" CACHE FILEPATH "Interpreter for the model-free Python checks")
if(NOT EXISTS "${ZEROCOOL_PYTHON}")
    find_package(Python3 COMPONENTS Interpreter)
    set(ZEROCOOL_PYTHON "${Python3_EXECUTABLE}")
endif()
if(EXISTS "${ZEROCOOL_PYTHON}")
    add_test(NAME PythonTooling
        COMMAND ${ZEROCOOL_PYTHON} -m unittest discover -s ${CMAKE_SOURCE_DIR}/scripts/qwen -p "test_*.py"
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
    set_tests_properties(PythonTooling PROPERTIES TIMEOUT 900)
else()
    message(WARNING "No Python interpreter found; the model-free Python checks are not registered.")
endif()

# Developer replay and probe executables, built on request so an ordinary build
# stays the engine, the client and the tests.
option(ZEROCOOL_BUILD_DIAGNOSTICS "Build the replay, probe and check executables" ON)
if(ZEROCOOL_BUILD_DIAGNOSTICS)
    foreach(diagnostic
        qwen_moe_replay:replay_moe qwen_attention_replay:replay_attention qwen_storage_check:check_storage
        qwen_q8_check:check_q8 qwen_q4_check:check_q4 qwen_route_check:check_route qwen_panel_check:check_panel
        qwen_memory_check:check_memory qwen_q3_probe:probe_q3 qwen_submission_probe:probe_submission
        qwen_sparse_replay:replay_sparse qwen_cached_recovery:check_cached_recovery qwen_ple_replay:replay_ple)
        string(REPLACE ":" ";" parts ${diagnostic})
        list(GET parts 0 target)
        list(GET parts 1 source)
        add_executable(${target} scripts/qwen/${source}.cpp)
        target_link_libraries(${target} PRIVATE zerocool_lib zerocool_arithmetic)
    endforeach()
endif()

option(ZEROCOOL_REAL_MODEL_TESTS "Require the pinned model for explicit integration checks" OFF)
set(ZEROCOOL_MODEL_DIR "${CMAKE_SOURCE_DIR}/.cache/models/qwen38-flash-next" CACHE PATH "Pinned model directory")
if(ZEROCOOL_REAL_MODEL_TESTS)
    add_test(NAME QwenRealModel COMMAND zerocool bench --model ${ZEROCOOL_MODEL_DIR} --prompt "The capital of France is" --max-tokens 8 --repetitions 1)
    set_tests_properties(QwenRealModel PROPERTIES LABELS "real-model" TIMEOUT 1200)
endif()

install(FILES THIRD_PARTY_NOTICES.md DESTINATION share/zerocool)
install(DIRECTORY docs/licenses/ DESTINATION share/zerocool/licenses)
install(DIRECTORY "${json_SOURCE_DIR}/include/nlohmann" DESTINATION include)
install(FILES "${json_SOURCE_DIR}/LICENSE.MIT" DESTINATION share/zerocool/licenses RENAME nlohmann-json.txt)
