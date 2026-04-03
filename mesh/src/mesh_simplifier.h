#pragma once

#include "mesh_types.h"

#include <cstddef>

namespace heimdall::mesh {

// ---------------------------------------------------------------------------
// Configuration for mesh simplification.
// ---------------------------------------------------------------------------

struct SimplifierConfig {
    /// Target number of output faces.
    size_t target_faces = 25000;

    /// Absolute minimum face count — simplification will stop here even if
    /// the target_error budget is not yet exhausted.
    size_t min_faces = 5000;

    /// Maximum geometric error allowed by meshoptimizer (relative to mesh
    /// bounding sphere radius).  0.01 = 1% of bounding sphere.
    float target_error = 0.01f;

    /// If true, boundary (border) vertices are locked in place so open
    /// edges remain clean.  Useful for tiled meshes but usually not needed
    /// for closed human-body meshes from TSDF fusion.
    bool lock_boundary = false;

    /// Run post-simplification GPU-friendly optimizations:
    ///   - Vertex-cache optimization (reorder indices for hardware cache).
    ///   - Overdraw optimization (reorder for front-to-back rendering).
    bool optimize_vertex_cache = true;
    bool optimize_overdraw     = true;
};

// ---------------------------------------------------------------------------
// Result metrics from simplification.
// ---------------------------------------------------------------------------

struct SimplifierResult {
    size_t input_faces      = 0;
    size_t output_faces     = 0;
    size_t input_vertices   = 0;
    size_t output_vertices  = 0;
    float  result_error     = 0.0f;  // meshoptimizer reported error
};

// ---------------------------------------------------------------------------
// MeshSimplifier — wraps meshoptimizer for triangle-mesh decimation.
//
// Usage:
//     MeshSimplifier simplifier(config);
//     auto [mesh, result] = simplifier.simplify(input_mesh);
// ---------------------------------------------------------------------------

class MeshSimplifier {
public:
    explicit MeshSimplifier(const SimplifierConfig& config = {});

    /// Simplify a triangle mesh to approximately `config.target_faces` faces.
    ///
    /// The returned mesh has compacted vertex/index buffers (unused vertices
    /// are removed) and optionally GPU-optimized index ordering.
    ///
    /// The input mesh is not modified.
    struct Output {
        TriMesh           mesh;
        SimplifierResult  metrics;
    };

    Output simplify(const TriMesh& input) const;

private:
    SimplifierConfig config_;
};

} // namespace heimdall::mesh
