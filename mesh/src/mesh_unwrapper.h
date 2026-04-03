#pragma once

#include "mesh_types.h"

#include <cstddef>

namespace heimdall::mesh {

// ---------------------------------------------------------------------------
// Configuration for UV unwrapping / atlas generation.
// ---------------------------------------------------------------------------

struct UnwrapperConfig {
    /// Atlas dimensions in pixels.  Both width and height must be powers of
    /// two (or at least multiples of 4) for texture compression.
    int atlas_width  = 2048;
    int atlas_height = 2048;

    /// Maximum chart stretch (0 = no stretch allowed, 0.5 = moderate).
    /// Controls the trade-off between atlas utilization and geometric
    /// distortion in each UV chart.  xatlas default is ~0.5.
    float max_chart_stretch = 0.5f;

    /// Gutter (padding) in pixels between charts to prevent texture bleeding
    /// during mip-mapping.  2-4 pixels is typical for 2K atlases.
    int gutter_pixels = 4;

    /// Maximum number of charts that xatlas will produce.  0 = unlimited.
    int max_charts = 0;

    /// Resolution for xatlas rasterization during packing.
    /// Higher = tighter packing but slower.  0 = use atlas_width as default.
    int pack_resolution = 0;
};

// ---------------------------------------------------------------------------
// Result metrics from UV unwrapping.
// ---------------------------------------------------------------------------

struct UnwrapperResult {
    size_t input_vertices  = 0;
    size_t output_vertices = 0;  // may be larger due to UV seam splits
    size_t input_faces     = 0;
    size_t output_faces    = 0;
    int    num_charts      = 0;
    float  utilization_pct = 0.0f;  // atlas space utilization (0-100)
    int    atlas_width     = 0;
    int    atlas_height    = 0;
};

// ---------------------------------------------------------------------------
// MeshUnwrapper — wraps xatlas for automatic UV atlas generation.
//
// xatlas may split vertices along UV seams.  After unwrapping, the mesh's
// vertex count may increase.  All vertex attributes (positions, normals,
// colors) are duplicated along seams so that the vertex and index buffers
// remain self-consistent.
//
// Usage:
//     MeshUnwrapper unwrapper(config);
//     auto [mesh, result] = unwrapper.unwrap(input_mesh);
// ---------------------------------------------------------------------------

class MeshUnwrapper {
public:
    explicit MeshUnwrapper(const UnwrapperConfig& config = {});

    struct Output {
        TriMesh          mesh;
        UnwrapperResult  metrics;
    };

    /// Generate UV coordinates for the input mesh.
    /// The input mesh does not need to have existing texcoords.
    /// The output mesh will have texcoords in [0,1] range.
    Output unwrap(const TriMesh& input) const;

private:
    UnwrapperConfig config_;
};

} // namespace heimdall::mesh
