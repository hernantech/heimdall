#include "mesh_simplifier.h"

#include <meshoptimizer.h>

#include <algorithm>
#include <cassert>
#include <cstring>
#include <numeric>
#include <vector>

namespace heimdall::mesh {

MeshSimplifier::MeshSimplifier(const SimplifierConfig& config)
    : config_(config) {}

MeshSimplifier::Output MeshSimplifier::simplify(const TriMesh& input) const {
    Output out;
    out.metrics.input_faces    = input.face_count();
    out.metrics.input_vertices = input.vertex_count();

    if (input.empty()) {
        return out;
    }

    const size_t index_count  = input.indices.size();
    const size_t vertex_count = input.vertex_count();
    const size_t stride       = sizeof(float) * 3; // position stride

    // If the mesh is already at or below the target, skip simplification
    // but still run the optional GPU optimizations.
    const size_t target_index_count = config_.target_faces * 3;
    const size_t min_index_count    = config_.min_faces * 3;

    std::vector<uint32_t> result_indices(index_count);
    size_t result_index_count = 0;
    float  result_error       = 0.0f;

    if (input.face_count() <= config_.target_faces) {
        // Nothing to simplify — keep the original indices.
        result_indices     = input.indices;
        result_index_count = index_count;
    } else {
        // -----------------------------------------------------------------
        // meshopt_simplify
        //
        // Flags:
        //   meshopt_SimplifyLockBorder — prevent boundary vertices from moving.
        //   meshopt_SimplifyErrorAbsolute is NOT set, so target_error is
        //   relative to the mesh bounding sphere.
        // -----------------------------------------------------------------
        unsigned int options = 0;
        if (config_.lock_boundary) {
            options |= meshopt_SimplifyLockBorder;
        }

        result_index_count = meshopt_simplify(
            result_indices.data(),
            input.indices.data(),
            index_count,
            reinterpret_cast<const float*>(input.positions.data()),
            vertex_count,
            stride,
            std::max(target_index_count, min_index_count),
            config_.target_error,
            options,
            &result_error);

        result_indices.resize(result_index_count);
    }

    // -----------------------------------------------------------------
    // GPU-friendly index reordering (optional).
    // -----------------------------------------------------------------

    if (config_.optimize_vertex_cache && result_index_count > 0) {
        meshopt_optimizeVertexCache(
            result_indices.data(),
            result_indices.data(),
            result_index_count,
            vertex_count);
    }

    if (config_.optimize_overdraw && result_index_count > 0) {
        // Overdraw threshold of 1.05 means we accept up to 5% worse ACMR
        // to improve overdraw.  This is the meshoptimizer recommended default.
        meshopt_optimizeOverdraw(
            result_indices.data(),
            result_indices.data(),
            result_index_count,
            reinterpret_cast<const float*>(input.positions.data()),
            vertex_count,
            stride,
            1.05f);
    }

    // -----------------------------------------------------------------
    // Compact the vertex buffer — remove unreferenced vertices and
    // build a new contiguous vertex buffer + remapped index buffer.
    // -----------------------------------------------------------------

    // Build remap table: old vertex index -> new vertex index.
    std::vector<uint32_t> remap(vertex_count, UINT32_MAX);
    uint32_t new_vertex_count = 0;

    for (size_t i = 0; i < result_index_count; ++i) {
        uint32_t old_idx = result_indices[i];
        if (remap[old_idx] == UINT32_MAX) {
            remap[old_idx] = new_vertex_count++;
        }
        result_indices[i] = remap[old_idx];
    }

    // Build the output mesh.
    TriMesh& om = out.mesh;
    om.positions.resize(new_vertex_count);
    if (input.has_normals())   om.normals.resize(new_vertex_count);
    if (input.has_texcoords()) om.texcoords.resize(new_vertex_count);
    if (input.has_colors())    om.vertex_colors.resize(new_vertex_count);

    for (size_t v = 0; v < vertex_count; ++v) {
        if (remap[v] == UINT32_MAX) continue;
        uint32_t nv = remap[v];
        om.positions[nv] = input.positions[v];
        if (input.has_normals())   om.normals[nv]       = input.normals[v];
        if (input.has_texcoords()) om.texcoords[nv]      = input.texcoords[v];
        if (input.has_colors())    om.vertex_colors[nv]  = input.vertex_colors[v];
    }

    om.indices = std::move(result_indices);

    // -----------------------------------------------------------------
    // Vertex-fetch optimization: reorder the vertex buffer to match
    // the index access pattern for better GPU vertex-fetch locality.
    //
    // We use meshopt_optimizeVertexFetchRemap (not meshopt_optimizeVertexFetch)
    // so that we can apply the same permutation to all vertex attributes.
    // -----------------------------------------------------------------

    if (config_.optimize_vertex_cache && om.face_count() > 0) {
        size_t nv = om.positions.size();
        std::vector<unsigned int> fetch_remap(nv);
        size_t unique = meshopt_optimizeVertexFetchRemap(
            fetch_remap.data(),
            om.indices.data(),
            om.indices.size(),
            nv);

        // Remap indices.
        for (auto& idx : om.indices) {
            idx = fetch_remap[idx];
        }

        // Apply remap to all vertex attributes.
        auto remap_attribute = [&](auto& vec) {
            using T = typename std::decay_t<decltype(vec)>::value_type;
            std::vector<T> tmp(unique);
            for (size_t i = 0; i < nv; ++i) {
                tmp[fetch_remap[i]] = vec[i];
            }
            vec = std::move(tmp);
        };

        remap_attribute(om.positions);
        if (!om.normals.empty())       remap_attribute(om.normals);
        if (!om.texcoords.empty())     remap_attribute(om.texcoords);
        if (!om.vertex_colors.empty()) remap_attribute(om.vertex_colors);
    }

    out.metrics.output_faces    = om.face_count();
    out.metrics.output_vertices = om.vertex_count();
    out.metrics.result_error    = result_error;

    return out;
}

} // namespace heimdall::mesh
