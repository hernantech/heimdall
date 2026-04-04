# =============================================================================
# heimdall::texture — Multi-view texture atlas generation.
#
# Projects camera images onto a UV-unwrapped mesh to produce a texture atlas.
# CPU-only offline quality path — no CUDA, no external dependencies beyond the
# C++ standard library.
#
# Dependencies:
#   - heimdall::mesh  (for TriMesh / Vec3 types — header-only usage)
#   - heimdall_fusion (for CameraCalibration struct — header-only usage)
#
# These are optional link-time dependencies; the texture module only uses
# types from their public headers and can be built stand-alone.
# =============================================================================

add_library(heimdall_texture STATIC
    src/texturing.cpp
    src/visibility.cpp
)

target_include_directories(heimdall_texture
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_compile_features(heimdall_texture PUBLIC cxx_std_17)

# Alias for use with target_link_libraries(... heimdall::texture).
add_library(heimdall::texture ALIAS heimdall_texture)
