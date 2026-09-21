cmake_minimum_required(VERSION 3.20)

project(zerocool
    VERSION 0.1.0
    DESCRIPTION "Qwen3.8-Flash-Next inference beyond RAM on a 32GB Apple Silicon Mac"
    LANGUAGES CXX
)

# Measurements and the shipped engine are Release builds. An unset build type
# would otherwise produce an unoptimized engine that still reports a normal
# build identity, so choose explicitly and record the choice in the fingerprint.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS Debug Release RelWithDebInfo MinSizeRel)
    message(STATUS "No CMAKE_BUILD_TYPE given; defaulting to Release.")
endif()

include(cmake/qwen.cmake)
