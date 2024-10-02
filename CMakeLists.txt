cmake_minimum_required(VERSION 3.22)

project(
    KokkosExecutionPolicyCreator
    LANGUAGES CXX
    VERSION 0.0.0
)

list(PREPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake/modules")

set(CMAKE_CXX_STANDARD 20)

include(options)
include(dependencies)

add_subdirectory(include)

if(POLK_ENABLE_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

if(POLK_ENABLE_EXAMPLES)
    add_subdirectory(examples)
endif()

if(POLK_ENABLE_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()
