#include "mesh_unwrapper.h"

#include <xatlas.h>

#include <cassert>
#include <cstring>
#include <vector>

namespace heimdall::mesh {

MeshUnwrapper::MeshUnwrapper(const UnwrapperConfig& config)
    : config_(config) {}

MeshUnwrapper::Output MeshUnwrapper::unwrap(const TriMesh& input) const {
    Output out;
    out.metrics.input_vertices = input.vertex_count();
    out.metrics.input_faces    = input.face_count();

    if (input.empty()) {
        return out;
    }

    // -----------------------------------------------------------------
    // 1. Create xatlas atlas.
    // -----------------------------------------------------------------

    xatlas::Atlas* atlas = xatlas::Create();

    // -----------------------------------------------------------------
    // 2. Add the input mesh to xatlas.
    // -----------------------------------------------------------------

    xatlas::MeshDecl mesh_decl;

    mesh_decl.vertexCount          = static_cast<uint32_t>(input.vertex_count());
    mesh_decl.vertexPositionData   = input.positions.data();
    mesh_decl.vertexPositionStride = sizeof(Vec3);

    if (input.has_normals()) {
        mesh_decl.vertexNormalData   = input.normals.data();
        mesh_decl.vertexNormalStride = sizeof(Vec3);
    }

    // If the mesh already has UVs (e.g. from a previous pass), we can feed
    // them as a hint, but typically post-simplification meshes do not.
    if (input.has_texcoords()) {
        mesh_decl.vertexUvData   = input.texcoords.data();
        mesh_decl.vertexUvStride = sizeof(Vec2);
    }

    mesh_decl.indexCount  = static_cast<uint32_t>(input.indices.size());
    mesh_decl.indexData   = input.indices.data();
    mesh_decl.indexFormat = xatlas::IndexFormat::UInt32;

    xatlas::AddMeshError error = xatlas::AddMesh(atlas, mesh_decl);
    if (error != xatlas::AddMeshError::Success) {
        xatlas::Destroy(atlas);
        // Return the input mesh unchanged on error.
        out.mesh = input;
        out.metrics.output_vertices = input.vertex_count();
        out.metrics.output_faces    = input.face_count();
        return out;
    }

    // -----------------------------------------------------------------
    // 3. Configure chart and pack options.
    // -----------------------------------------------------------------

    xatlas::ChartOptions chart_options;
    chart_options.maxIterations = 1;  // single pass is sufficient for real-time
    if (config_.max_charts > 0) {
        chart_options.maxCost = 2.0f; // higher cost = fewer charts
    }

    xatlas::PackOptions pack_options;
    pack_options.padding    = config_.gutter_pixels;
    pack_options.texelsPerUnit = 0.0f; // auto
    pack_options.resolution = config_.pack_resolution > 0
                                  ? config_.pack_resolution
                                  : config_.atlas_width;
    pack_options.bilinear   = true;
    pack_options.blockAlign = true;

    // -----------------------------------------------------------------
    // 4. Generate the atlas.
    // -----------------------------------------------------------------

    xatlas::Generate(atlas, chart_options, pack_options);

    // -----------------------------------------------------------------
    // 5. Extract results.
    //
    // xatlas may split vertices along UV seams.  The output mesh from
    // xatlas has its own index buffer and a per-output-vertex table
    // giving the original vertex index + the new UV.
    // -----------------------------------------------------------------

    assert(atlas->meshCount == 1);
    const xatlas::Mesh& xmesh = atlas->meshes[0];

    const uint32_t new_vertex_count = xmesh.vertexCount;
    const uint32_t new_index_count  = xmesh.indexCount;

    TriMesh& om = out.mesh;

    // Rebuild vertex attributes: for each xatlas output vertex, copy
    // the original vertex attributes and assign the new UV.
    om.positions.resize(new_vertex_count);
    if (input.has_normals())  om.normals.resize(new_vertex_count);
    if (input.has_colors())   om.vertex_colors.resize(new_vertex_count);
    om.texcoords.resize(new_vertex_count);

    for (uint32_t v = 0; v < new_vertex_count; ++v) {
        const xatlas::Vertex& xv = xmesh.vertexArray[v];
        uint32_t orig = xv.xref;  // index into the original vertex buffer

        om.positions[v] = input.positions[orig];
        if (input.has_normals())  om.normals[v]       = input.normals[orig];
        if (input.has_colors())   om.vertex_colors[v]  = input.vertex_colors[orig];

        // xatlas UVs are in texel space (0..atlas_width/height).
        // Normalize to [0, 1] for standard texcoord convention.
        float atlas_w = static_cast<float>(atlas->width);
        float atlas_h = static_cast<float>(atlas->height);
        om.texcoords[v].x = (atlas_w > 0.0f) ? xv.uv[0] / atlas_w : 0.0f;
        om.texcoords[v].y = (atlas_h > 0.0f) ? xv.uv[1] / atlas_h : 0.0f;
    }

    // Copy the new index buffer.
    om.indices.resize(new_index_count);
    for (uint32_t i = 0; i < new_index_count; ++i) {
        om.indices[i] = xmesh.indexArray[i];
    }

    // -----------------------------------------------------------------
    // 6. Populate metrics.
    // -----------------------------------------------------------------

    out.metrics.output_vertices = new_vertex_count;
    out.metrics.output_faces    = new_index_count / 3;
    out.metrics.num_charts      = static_cast<int>(atlas->chartCount);
    out.metrics.atlas_width     = static_cast<int>(atlas->width);
    out.metrics.atlas_height    = static_cast<int>(atlas->height);

    // Compute atlas utilization.
    // xatlas provides per-chart utilization; we can also read atlas->utilization
    // which gives an array of utilization per atlas page.
    if (atlas->atlasCount > 0 && atlas->utilization != nullptr) {
        out.metrics.utilization_pct = atlas->utilization[0] * 100.0f;
    }

    // -----------------------------------------------------------------
    // 7. Cleanup.
    // -----------------------------------------------------------------

    xatlas::Destroy(atlas);

    return out;
}

} // namespace heimdall::mesh
