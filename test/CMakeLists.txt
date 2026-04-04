# =============================================================================
# heimdall unit tests
#
# These tests validate CPU-testable logic without CUDA or GPU libraries.
# Instead of linking against the full module libraries (which may require
# CUDA, Protobuf, gRPC, etc.), we compile the specific .cpp source files
# needed for each test directly into the test binary.
#
# This makes the test suite self-contained and buildable on any machine
# with a C++17 compiler.
# =============================================================================

cmake_minimum_required(VERSION 3.16)
project(heimdall_tests LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---------------------------------------------------------------------------
# Module source files compiled directly into the test binary.
# Only pure-CPU .cpp files — no .cu files, no GPU dependencies.
# ---------------------------------------------------------------------------

set(MODULE_SOURCES
    # capture
    ${CMAKE_CURRENT_SOURCE_DIR}/../capture/src/frame_sync.cpp

    # gaussian — camera selector (no CUDA, no ZLIB needed)
    ${CMAKE_CURRENT_SOURCE_DIR}/../gaussian/src/camera_selector.cpp

    # texture — visibility / BVH (CPU-only)
    ${CMAKE_CURRENT_SOURCE_DIR}/../texture/src/visibility.cpp

    # stream — multiplexer (CPU-only, depends on spz_writer.h types only)
    ${CMAKE_CURRENT_SOURCE_DIR}/../stream/src/gs_multiplexer.cpp
)

# ---------------------------------------------------------------------------
# Test source files.
# ---------------------------------------------------------------------------

set(TEST_SOURCES
    test_main.cpp
    test_frame_sync.cpp
    test_camera_selector.cpp
    test_mesh_types.cpp
    test_mesh_pipeline.cpp
    test_visibility.cpp
    test_multiplexer.cpp
)

# ---------------------------------------------------------------------------
# Check for meshoptimizer availability.
# If found, we can compile the simplification tests; otherwise they are
# skipped at runtime.
# ---------------------------------------------------------------------------

find_package(meshoptimizer QUIET)

set(MESH_PIPELINE_SOURCES "")
if(TARGET meshoptimizer OR meshoptimizer_FOUND)
    set(HEIMDALL_HAS_MESHOPTIMIZER ON)
    list(APPEND MESH_PIPELINE_SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/../mesh/src/mesh_simplifier.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/../mesh/src/mesh_pipeline.cpp
    )
    # Check for xatlas too (needed by mesh_unwrapper which is called by pipeline).
    find_package(xatlas QUIET)
    if(TARGET xatlas OR xatlas_FOUND)
        list(APPEND MESH_PIPELINE_SOURCES
            ${CMAKE_CURRENT_SOURCE_DIR}/../mesh/src/mesh_unwrapper.cpp
        )
    endif()
endif()

# ---------------------------------------------------------------------------
# Test executable.
# ---------------------------------------------------------------------------

add_executable(heimdall_tests
    ${TEST_SOURCES}
    ${MODULE_SOURCES}
    ${MESH_PIPELINE_SOURCES}
)

target_include_directories(heimdall_tests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../capture/src
    ${CMAKE_CURRENT_SOURCE_DIR}/../gaussian/src
    ${CMAKE_CURRENT_SOURCE_DIR}/../mesh/src
    ${CMAKE_CURRENT_SOURCE_DIR}/../texture/src
    ${CMAKE_CURRENT_SOURCE_DIR}/../stream/src
    ${CMAKE_CURRENT_SOURCE_DIR}/../encode/src
)

target_compile_features(heimdall_tests PRIVATE cxx_std_17)

# Link meshoptimizer if available.
if(HEIMDALL_HAS_MESHOPTIMIZER)
    target_compile_definitions(heimdall_tests PRIVATE HEIMDALL_HAS_MESHOPTIMIZER)
    target_link_libraries(heimdall_tests PRIVATE meshoptimizer)
    if(TARGET xatlas)
        target_link_libraries(heimdall_tests PRIVATE xatlas)
    endif()
endif()

# pthreads (needed for std::mutex in frame_sync and multiplexer).
find_package(Threads REQUIRED)
target_link_libraries(heimdall_tests PRIVATE Threads::Threads)

# Register with CTest.
add_test(NAME heimdall_tests COMMAND heimdall_tests)
