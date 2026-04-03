# =============================================================================
# heimdall::mesh — Mesh simplification, UV unwrapping, and processing pipeline.
#
# Dependencies:
#   - meshoptimizer (v1.0+)  — triangle mesh simplification & GPU optimization
#   - xatlas                  — automatic UV atlas generation
#
# Both can be provided via:
#   (a) System install / find_package()
#   (b) Git submodules in third_party/
#   (c) FetchContent (shown below, commented out)
#
# To use FetchContent, uncomment the block below and remove the find_package
# calls.  FetchContent will download and build the dependencies at configure
# time.
# =============================================================================

# ---- Option (c): FetchContent (uncomment to use) ----------------------------
#
# include(FetchContent)
#
# FetchContent_Declare(
#     meshoptimizer
#     GIT_REPOSITORY https://github.com/zeux/meshoptimizer.git
#     GIT_TAG        v0.22
# )
# FetchContent_MakeAvailable(meshoptimizer)
#
# FetchContent_Declare(
#     xatlas
#     GIT_REPOSITORY https://github.com/jpcy/xatlas.git
#     GIT_TAG        master
# )
# FetchContent_GetProperties(xatlas)
# if(NOT xatlas_POPULATED)
#     FetchContent_Populate(xatlas)
#     # xatlas does not ship a CMakeLists.txt for a library target, so we
#     # create one manually.
#     add_library(xatlas STATIC ${xatlas_SOURCE_DIR}/source/xatlas/xatlas.cpp)
#     target_include_directories(xatlas PUBLIC ${xatlas_SOURCE_DIR}/source/xatlas)
# endif()
#
# ---- End FetchContent -------------------------------------------------------

# ---- Option (a)/(b): Assume targets already exist ---------------------------
# If using git submodules under third_party/, add them in the top-level
# CMakeLists.txt before this subdirectory:
#
#   add_subdirectory(third_party/meshoptimizer)
#   add_subdirectory(third_party/xatlas)  # with a wrapper CMakeLists.txt
#
# If using system installs:
#
#   find_package(meshoptimizer REQUIRED)
#   find_package(xatlas REQUIRED)
#
# For now we assume the targets `meshoptimizer` and `xatlas` are available
# by the time this file is processed (via any of the methods above).
# -----------------------------------------------------------------------------

add_library(heimdall_mesh STATIC
    src/mesh_simplifier.cpp
    src/mesh_unwrapper.cpp
    src/mesh_pipeline.cpp
)

target_include_directories(heimdall_mesh
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_compile_features(heimdall_mesh PUBLIC cxx_std_17)

target_link_libraries(heimdall_mesh
    PUBLIC
        meshoptimizer
        xatlas
)

# Alias for use with target_link_libraries(... heimdall::mesh).
add_library(heimdall::mesh ALIAS heimdall_mesh)
