set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
if(NOT APPLE)
    message(FATAL_ERROR "FreeLLM's native engine requires Apple Silicon/macOS")
endif()
enable_language(OBJCXX)
set(CMAKE_OBJCXX_STANDARD 23)
set(CMAKE_OBJCXX_STANDARD_REQUIRED ON)

# One warning policy for every target built from this repository, including
# tests and diagnostics. Fetched dependencies keep their own settings.
add_library(freellm_warnings INTERFACE)
target_compile_options(freellm_warnings INTERFACE -Wall -Wextra -Wpedantic -Werror)

# Arithmetic must stay reproducible; -ffast-math would silently change results.
add_library(freellm_arithmetic INTERFACE)
target_compile_options(freellm_arithmetic INTERFACE -fno-fast-math)

set(FREELLM_SANITIZE "none" CACHE STRING "Sanitizer for local checks: none|address|undefined|thread")
set_property(CACHE FREELLM_SANITIZE PROPERTY STRINGS none address undefined thread)
if(NOT FREELLM_SANITIZE STREQUAL "none")
    # Sanitized builds are for tests and diagnosis; they never produce evidence.
    add_compile_options(-fsanitize=${FREELLM_SANITIZE} -fno-omit-frame-pointer -g)
    add_link_options(-fsanitize=${FREELLM_SANITIZE})
    message(STATUS "FreeLLM: ${FREELLM_SANITIZE} sanitizer enabled; timings from this build are not evidence.")
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

file(READ "${CMAKE_SOURCE_DIR}/kernels/metal/qwen.metal" FREELLM_METAL_SOURCE)
file(READ "${CMAKE_SOURCE_DIR}/models.lock.json" FREELLM_MODEL_LOCK)
file(READ "${CMAKE_SOURCE_DIR}/mixed-models.lock.json" FREELLM_MIXED_MODEL_LOCK)
file(READ "${CMAKE_SOURCE_DIR}/mixed-payload-reuse.lock.json" FREELLM_MIXED_REUSE_LOCK)
file(GLOB FREELLM_BUILD_INPUTS CONFIGURE_DEPENDS "${CMAKE_SOURCE_DIR}/src/qwen/*" "${CMAKE_SOURCE_DIR}/include/qwen/*")
list(APPEND FREELLM_BUILD_INPUTS "${CMAKE_SOURCE_DIR}/kernels/metal/qwen.metal" "${CMAKE_SOURCE_DIR}/models.lock.json"
    "${CMAKE_SOURCE_DIR}/mixed-models.lock.json" "${CMAKE_SOURCE_DIR}/mixed-payload-reuse.lock.json"
    "${CMAKE_SOURCE_DIR}/cmake/qwen.cmake" "${CMAKE_SOURCE_DIR}/cmake/qwen_embedded.hpp.in" "${CMAKE_SOURCE_DIR}/CMakeLists.txt")
set(FREELLM_BUILD_IDENTITY "")
foreach(input IN LISTS FREELLM_BUILD_INPUTS)
    file(SHA256 "${input}" digest)
    file(RELATIVE_PATH label "${CMAKE_SOURCE_DIR}" "${input}")
    string(APPEND FREELLM_BUILD_IDENTITY "${label}:${digest}\n")
endforeach()
# How the sources were compiled is part of the identity: an unoptimized or
# sanitized build must never report the same fingerprint as a measured one.
string(APPEND FREELLM_BUILD_IDENTITY
    "toolchain:${CMAKE_CXX_COMPILER_ID}-${CMAKE_CXX_COMPILER_VERSION}\n"
    "build_type:${CMAKE_BUILD_TYPE}\n"
    "flags:${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_${CMAKE_BUILD_TYPE}}\n"
    "sanitizer:${FREELLM_SANITIZE}\n")
string(SHA256 FREELLM_BUILD_FINGERPRINT "${FREELLM_BUILD_IDENTITY}")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${FREELLM_BUILD_INPUTS})
configure_file(cmake/qwen_embedded.hpp.in generated/qwen_embedded.hpp @ONLY)

add_library(freellm_lib STATIC
    src/qwen/storage.cpp src/qwen/metal.mm src/qwen/pressure.mm src/qwen/model.cpp src/qwen/prefill.cpp
    src/qwen/pipeline.cpp src/qwen/bench.cpp src/qwen/kernel_bench.cpp src/qwen/cached_replay.cpp
    src/qwen/tokenizer.mm src/qwen/session.cpp src/qwen/server.cpp src/qwen/chat_executor.cpp src/qwen/cli.cpp)
target_include_directories(freellm_lib PUBLIC include PRIVATE ${minja_SOURCE_DIR}/include ${CMAKE_BINARY_DIR}/generated)
target_link_libraries(freellm_lib PUBLIC nlohmann_json::nlohmann_json PRIVATE freellm_warnings freellm_arithmetic
    "-framework Metal" "-framework Foundation" "-framework IOKit")
set_source_files_properties(src/qwen/metal.mm src/qwen/pressure.mm src/qwen/tokenizer.mm PROPERTIES COMPILE_FLAGS "-fobjc-arc")

add_executable(freellm src/qwen/main.cpp)
target_link_libraries(freellm PRIVATE freellm_lib freellm_warnings freellm_arithmetic)

option(FREELLM_BUILD_TUI "Build the local terminal chat client" ON)
if(FREELLM_BUILD_TUI)
    find_package(CURL REQUIRED)
    set(FTXUI_BUILD_DOCS OFF CACHE BOOL "" FORCE)
    set(FTXUI_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(FTXUI_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(FTXUI_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
    FetchContent_Declare(ftxui GIT_REPOSITORY https://github.com/ArthurSonzogni/FTXUI.git
        GIT_TAG f921fad208912747c17d129a8ef75ec7624b6eec) # v7.0.3
    FetchContent_MakeAvailable(ftxui)
    add_library(freellm_chat_client STATIC src/qwen/chat_client.cpp)
    target_include_directories(freellm_chat_client PUBLIC include)
    target_link_libraries(freellm_chat_client PUBLIC nlohmann_json::nlohmann_json PRIVATE freellm_warnings CURL::libcurl)
    target_sources(freellm PRIVATE src/qwen/chat.cpp)
    target_compile_definitions(freellm PRIVATE FREELLM_WITH_TUI=1)
    target_link_libraries(freellm PRIVATE freellm_chat_client ftxui::component ftxui::dom ftxui::screen)
endif()
set_target_properties(freellm PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
install(TARGETS freellm freellm_lib RUNTIME DESTINATION bin ARCHIVE DESTINATION lib)
install(DIRECTORY include/qwen DESTINATION include)

enable_testing()
add_executable(test_qwen tests/test_qwen.cpp)
target_include_directories(test_qwen PRIVATE external)
target_link_libraries(test_qwen PRIVATE freellm_lib freellm_warnings freellm_arithmetic)
add_test(NAME QwenNative COMMAND test_qwen)
set_tests_properties(QwenNative PROPERTIES TIMEOUT 900)

if(FREELLM_BUILD_TUI)
    add_executable(test_chat_transport tests/test_chat_transport.cpp)
    target_include_directories(test_chat_transport PRIVATE external)
    target_link_libraries(test_chat_transport PRIVATE freellm_lib freellm_chat_client freellm_warnings)
    add_test(NAME ChatTransport COMMAND test_chat_transport)
    set_tests_properties(ChatTransport PROPERTIES TIMEOUT 120)
    add_executable(chat_fixture tests/chat_fixture.cpp src/qwen/chat.cpp)
    target_link_libraries(chat_fixture PRIVATE freellm_lib freellm_chat_client freellm_warnings ftxui::component ftxui::dom ftxui::screen)
endif()

# The model-free Python checks are part of the same suite; without them a green
# ctest would cover only the native half of the project.
# The checks import NumPy, so prefer the repository environment created by
# scripts/qwen/setup_reference.sh over whatever interpreter is on PATH.
set(FREELLM_PYTHON "${CMAKE_SOURCE_DIR}/.venv/bin/python" CACHE FILEPATH "Interpreter for the model-free Python checks")
if(NOT EXISTS "${FREELLM_PYTHON}")
    find_package(Python3 COMPONENTS Interpreter)
    set(FREELLM_PYTHON "${Python3_EXECUTABLE}")
endif()
if(EXISTS "${FREELLM_PYTHON}")
    add_test(NAME PythonTooling
        COMMAND ${FREELLM_PYTHON} -m unittest discover -s ${CMAKE_SOURCE_DIR}/scripts/qwen -p "test_*.py"
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
    set_tests_properties(PythonTooling PROPERTIES TIMEOUT 900)
else()
    message(WARNING "No Python interpreter found; the model-free Python checks are not registered.")
endif()

# Developer replay and probe executables, built on request so an ordinary build
# stays the engine, the client and the tests.
option(FREELLM_BUILD_DIAGNOSTICS "Build the replay, probe and check executables" ON)
if(FREELLM_BUILD_DIAGNOSTICS)
    foreach(diagnostic
        qwen_moe_replay:replay_moe qwen_attention_replay:replay_attention qwen_storage_check:check_storage
        qwen_q8_check:check_q8 qwen_q4_check:check_q4 qwen_route_check:check_route qwen_panel_check:check_panel
        qwen_memory_check:check_memory qwen_q3_probe:probe_q3 qwen_submission_probe:probe_submission
        qwen_sparse_replay:replay_sparse qwen_cached_recovery:check_cached_recovery qwen_ple_replay:replay_ple)
        string(REPLACE ":" ";" parts ${diagnostic})
        list(GET parts 0 target)
        list(GET parts 1 source)
        add_executable(${target} scripts/qwen/${source}.cpp)
        target_link_libraries(${target} PRIVATE freellm_lib freellm_arithmetic)
    endforeach()
endif()

option(FREELLM_REAL_MODEL_TESTS "Require the pinned model for explicit integration checks" OFF)
set(FREELLM_MODEL_DIR "${CMAKE_SOURCE_DIR}/.cache/models/qwen38-flash-next" CACHE PATH "Pinned model directory")
if(FREELLM_REAL_MODEL_TESTS)
    add_test(NAME QwenRealModel COMMAND freellm bench --model ${FREELLM_MODEL_DIR} --prompt "The capital of France is" --max-tokens 8 --repetitions 1)
    set_tests_properties(QwenRealModel PROPERTIES LABELS "real-model" TIMEOUT 1200)
endif()

install(FILES THIRD_PARTY_NOTICES.md DESTINATION share/freellm)
install(DIRECTORY docs/licenses/ DESTINATION share/freellm/licenses)
install(DIRECTORY "${json_SOURCE_DIR}/include/nlohmann" DESTINATION include)
install(FILES "${json_SOURCE_DIR}/LICENSE.MIT" DESTINATION share/freellm/licenses RENAME nlohmann-json.txt)
