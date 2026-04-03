#pragma once

#include "mesh_simplifier.h"
#include "mesh_types.h"
#include "mesh_unwrapper.h"

#include <cstddef>
#include <string>

namespace heimdall::mesh {

// ---------------------------------------------------------------------------
// Top-level mesh processing pipeline configuration.
//
// This encapsulates the full simplify -> optimize -> unwrap pipeline that
// takes a raw mesh from fusion/marching-cubes and produces a mesh ready
// for texturing and glTF export.
// ---------------------------------------------------------------------------

struct MeshPipelineConfig {
    // -- Simplification ------------------------------------------------
    SimplifierConfig simplifier;

    // -- UV Unwrapping -------------------------------------------------
    UnwrapperConfig unwrapper;

    // -- Pre-processing flags ------------------------------------------

    /// Remove degenerate (zero-area) triangles before simplification.
    bool remove_degenerates = true;

    /// Recompute area-weighted per-vertex normals if the input mesh has
    /// no normals (or if force_recompute_normals is set).
    bool compute_normals_if_missing = true;

    /// Force recomputation of normals even if the input already has them.
    /// Useful after simplification changes the geometry.
    bool force_recompute_normals = false;

    /// Recompute normals after simplification (recommended — simplification
    /// can distort the original normals).
    bool recompute_normals_after_simplify = true;

    // -- Pipeline stage toggles ----------------------------------------

    /// Skip simplification (pass mesh through to unwrapping as-is).
    bool skip_simplification = false;

    /// Skip UV unwrapping (e.g. when only a simplified proxy mesh is needed).
    bool skip_unwrap = false;
};

// ---------------------------------------------------------------------------
// Per-stage metrics collected during pipeline execution.
// ---------------------------------------------------------------------------

struct MeshPipelineMetrics {
    // Input stats.
    size_t input_vertices  = 0;
    size_t input_faces     = 0;
    size_t degenerate_faces_removed = 0;

    // Simplification.
    SimplifierResult simplifier;

    // Unwrapping.
    UnwrapperResult  unwrapper;

    // Output stats.
    size_t output_vertices = 0;
    size_t output_faces    = 0;

    // Timing (milliseconds).
    double preprocess_ms   = 0.0;
    double simplify_ms     = 0.0;
    double unwrap_ms       = 0.0;
    double total_ms        = 0.0;
};

// ---------------------------------------------------------------------------
// MeshPipeline — orchestrates the full mesh processing workflow.
//
// Pipeline stages:
//   1. Validate input
//   2. Pre-process (remove degenerates, compute normals)
//   3. Simplify to target face count
//   4. Recompute normals (post-simplification)
//   5. UV unwrap
//   6. Return processed mesh
//
// The output TriMesh can be directly converted to heimdall::encode::MeshFrame
// via its flat_*() accessors.
//
// Usage:
//     MeshPipeline pipeline(config);
//     auto [mesh, metrics] = pipeline.process(raw_mesh);
//
//     // Convert to MeshFrame for the glTF writer:
//     encode::MeshFrame frame;
//     frame.positions = mesh.flat_positions();
//     frame.normals   = mesh.flat_normals();
//     frame.texcoords = mesh.flat_texcoords();
//     frame.indices   = mesh.indices;
// ---------------------------------------------------------------------------

class MeshPipeline {
public:
    explicit MeshPipeline(const MeshPipelineConfig& config = {});

    struct Output {
        TriMesh              mesh;
        MeshPipelineMetrics  metrics;
    };

    /// Process a raw mesh through the full pipeline.
    ///
    /// @param input  Raw triangle mesh (e.g. from marching cubes).
    /// @return       Processed mesh ready for texturing + export, plus metrics.
    ///
    /// @throws std::invalid_argument if the input mesh fails validation.
    Output process(const TriMesh& input) const;

private:
    MeshPipelineConfig config_;
};

} // namespace heimdall::mesh
