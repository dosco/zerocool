if(NOT DEFINED sentencepiece_SOURCE_DIR)
    message(FATAL_ERROR "sentencepiece_SOURCE_DIR is not defined for patching.")
endif()

set(SPM_CMAKE_FILE "${sentencepiece_SOURCE_DIR}/CMakeLists.txt")
if(NOT EXISTS "${SPM_CMAKE_FILE}")
    message(FATAL_ERROR "SentencePiece CMakeLists.txt not found at ${SPM_CMAKE_FILE}")
endif()

file(READ "${SPM_CMAKE_FILE}" SPM_CMAKE_CONTENTS)
if(SPM_CMAKE_CONTENTS MATCHES "cmake_minimum_required\\(VERSION 3\\.20\\)")
    return()
endif()

string(REGEX REPLACE "cmake_minimum_required\\(VERSION [0-9.]+[^)]*\\)"
                     "cmake_minimum_required(VERSION 3.20)"
                     SPM_CMAKE_CONTENTS
                     "${SPM_CMAKE_CONTENTS}")
file(WRITE "${SPM_CMAKE_FILE}" "${SPM_CMAKE_CONTENTS}")
message(STATUS "Patched SentencePiece CMakeLists.txt to require CMake 3.20")

