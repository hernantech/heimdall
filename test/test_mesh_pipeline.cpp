// ============================================================================
// Tests for heimdall::mesh::MeshPipeline
//
// The pipeline links against meshoptimizer (simplification) and xatlas
// (UV unwrapping).  Tests that exercise the full pipeline are conditionally
// compiled behind HEIMDALL_HAS_MESHOPTIMIZER.  When that define is absent,
// all pipeline tests are marked SKIP.
//
// Preprocessing logic (degenerate removal, normal computation) is tested
// directly via TriMesh in test_mesh_types.cpp and does not require
// meshoptimizer.
// ============================================================================

#include "test_helpers.h"
#include "../mesh/src/mesh_types.h"

#ifdef HEIMDALL_HAS_MESHOPTIMIZER
#include "../mesh/src/mesh_pipeline.h"
#endif

#include <cmath>
#include <unordered_map>
#include <vector>

using namespace heimdall::mesh;

// ---------------------------------------------------------------------------
// Helpers -- icosphere generation (always available for mesh_types tests too)
// ---------------------------------------------------------------------------

static constexpr float kPi = 3.14159265358979323846f;

// Generate an icosphere by recursive subdivision of an icosahedron.
// Returns a TriMesh with approximately 20 * 4^subdivisions faces.
static TriMesh make_icosphere(int subdivisions = 3) {
    // Golden ratio.
    const float phi = (1.0f + std::sqrt(5.0f)) / 2.0f;

    // 12 vertices of an icosahedron.
    std::vector<Vec3> verts = {
        {-1,  phi, 0}, { 1,  phi, 0}, {-1, -phi, 0}, { 1, -phi, 0},
        { 0, -1,  phi}, { 0,  1,  phi}, { 0, -1, -phi}, { 0,  1, -phi},
        { phi, 0, -1}, { phi, 0,  1}, {-phi, 0, -1}, {-phi, 0,  1},
    };

    // Normalize to unit sphere.
    for (auto& v : verts) v = v.normalized();

    // 20 faces of the icosahedron.
    std::vector<uint32_t> indices = {
        0, 11, 5,   0, 5, 1,    0, 1, 7,    0, 7, 10,   0, 10, 11,
        1, 5, 9,    5, 11, 4,   11, 10, 2,  10, 7, 6,   7, 1, 8,
        3, 9, 4,    3, 4, 2,    3, 2, 6,    3, 6, 8,    3, 8, 9,
        4, 9, 5,    2, 4, 11,   6, 2, 10,   8, 6, 7,    9, 8, 1,
    };

    // Subdivision loop.
    for (int s = 0; s < subdivisions; ++s) {
        struct PairHash {
            size_t operator()(std::pair<uint32_t, uint32_t> p) const {
                return std::hash<uint64_t>{}(
                    (static_cast<uint64_t>(p.first) << 32) | p.second);
            }
        };
        std::unordered_map<std::pair<uint32_t, uint32_t>, uint32_t, PairHash> mid_cache;

        auto get_midpoint = [&](uint32_t a, uint32_t b) -> uint32_t {
            auto key = std::make_pair(std::min(a, b), std::max(a, b));
            auto it = mid_cache.find(key);
            if (it != mid_cache.end()) return it->second;
            Vec3 mid = (verts[a] + verts[b]) * 0.5f;
            mid = mid.normalized();
            uint32_t idx = static_cast<uint32_t>(verts.size());
            verts.push_back(mid);
            mid_cache[key] = idx;
            return idx;
        };

        std::vector<uint32_t> new_indices;
        new_indices.reserve(indices.size() * 4);

        for (size_t f = 0; f < indices.size(); f += 3) {
            uint32_t i0 = indices[f + 0];
            uint32_t i1 = indices[f + 1];
            uint32_t i2 = indices[f + 2];

            uint32_t m01 = get_midpoint(i0, i1);
            uint32_t m12 = get_midpoint(i1, i2);
            uint32_t m02 = get_midpoint(i0, i2);

            new_indices.insert(new_indices.end(), {i0, m01, m02});
            new_indices.insert(new_indices.end(), {i1, m12, m01});
            new_indices.insert(new_indices.end(), {i2, m02, m12});
            new_indices.insert(new_indices.end(), {m01, m12, m02});
        }

        indices = std::move(new_indices);
    }

    TriMesh m;
    m.positions = std::move(verts);
    m.indices   = std::move(indices);
    return m;
}

// ---------------------------------------------------------------------------
// Tests -- pipeline (requires meshoptimizer + xatlas)
// ---------------------------------------------------------------------------

#ifdef HEIMDALL_HAS_MESHOPTIMIZER

// A cube with a degenerate face appended.
static TriMesh make_cube_with_degenerate() {
    TriMesh m;
    m.positions = {
        {0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0},
        {0, 0, 1}, {1, 0, 1}, {1, 1, 1}, {0, 1, 1},
    };
    m.indices = {
        0, 3, 2,  0, 2, 1,
        4, 5, 6,  4, 6, 7,
        0, 4, 7,  0, 7, 3,
        1, 2, 6,  1, 6, 5,
        0, 1, 5,  0, 5, 4,
        3, 7, 6,  3, 6, 2,
        // Degenerate: two identical indices.
        0, 0, 1,
    };
    return m;
}

TEST(mesh_pipeline_rejects_empty_mesh) {
    MeshPipelineConfig config;
    config.skip_simplification = true;
    config.skip_unwrap = true;
    MeshPipeline pipeline(config);

    TriMesh empty;
    ASSERT_THROWS(pipeline.process(empty), std::invalid_argument);
}

TEST(mesh_pipeline_rejects_invalid_mesh) {
    MeshPipelineConfig config;
    config.skip_simplification = true;
    config.skip_unwrap = true;
    MeshPipeline pipeline(config);

    TriMesh bad;
    bad.positions = {{0, 0, 0}};
    bad.indices = {0, 1, 2}; // index 1, 2 out of range
    ASSERT_THROWS(pipeline.process(bad), std::invalid_argument);
}

TEST(mesh_pipeline_removes_degenerates) {
    MeshPipelineConfig config;
    config.skip_simplification = true;
    config.skip_unwrap = true;
    config.remove_degenerates = true;
    MeshPipeline pipeline(config);

    auto m = make_cube_with_degenerate();
    ASSERT_EQ(m.face_count(), static_cast<size_t>(13)); // 12 cube + 1 degenerate

    auto out = pipeline.process(m);
    ASSERT_EQ(out.metrics.degenerate_faces_removed, static_cast<size_t>(1));
    ASSERT_EQ(out.mesh.face_count(), static_cast<size_t>(12));
}

TEST(mesh_pipeline_computes_normals_if_missing) {
    MeshPipelineConfig config;
    config.skip_simplification = true;
    config.skip_unwrap = true;
    config.compute_normals_if_missing = true;
    MeshPipeline pipeline(config);

    auto m = make_icosphere(1);
    ASSERT_FALSE(m.has_normals());

    auto out = pipeline.process(m);
    ASSERT_TRUE(out.mesh.has_normals());
    ASSERT_EQ(out.mesh.normals.size(), out.mesh.positions.size());

    // All normals should be unit length.
    for (auto& n : out.mesh.normals) {
        ASSERT_NEAR(n.length(), 1.0f, 1e-4f);
    }
}

TEST(mesh_pipeline_skip_simplification_preserves_mesh) {
    MeshPipelineConfig config;
    config.skip_simplification = true;
    config.skip_unwrap = true;
    MeshPipeline pipeline(config);

    auto m = make_icosphere(2); // 320 faces
    size_t original_faces = m.face_count();

    auto out = pipeline.process(m);
    ASSERT_EQ(out.mesh.face_count(), original_faces);
    ASSERT_EQ(out.metrics.simplifier.output_faces, original_faces);
}

TEST(mesh_pipeline_metrics_populated) {
    MeshPipelineConfig config;
    config.skip_simplification = true;
    config.skip_unwrap = true;
    MeshPipeline pipeline(config);

    auto m = make_icosphere(2);
    auto out = pipeline.process(m);

    ASSERT_EQ(out.metrics.input_vertices, m.vertex_count());
    ASSERT_EQ(out.metrics.input_faces, m.face_count());
    ASSERT_GT(out.metrics.output_vertices, static_cast<size_t>(0));
    ASSERT_GT(out.metrics.output_faces, static_cast<size_t>(0));
    ASSERT_GE(out.metrics.total_ms, 0.0);
}

TEST(mesh_pipeline_simplify_icosphere) {
    // Generate an icosphere with ~1280 faces (3 subdivisions of icosahedron).
    auto m = make_icosphere(3);
    ASSERT_GT(m.face_count(), static_cast<size_t>(1000));

    MeshPipelineConfig config;
    config.simplifier.target_faces = 100;
    config.simplifier.min_faces = 50;
    config.simplifier.target_error = 0.05f;
    config.skip_unwrap = true;

    MeshPipeline pipeline(config);
    auto out = pipeline.process(m);

    // Face count should be reduced.
    ASSERT_LE(out.mesh.face_count(), static_cast<size_t>(150));
    ASSERT_GE(out.mesh.face_count(), config.simplifier.min_faces);

    // Mesh should still be valid.
    ASSERT_TRUE(out.mesh.is_valid());

    // Metrics should report simplification.
    ASSERT_GT(out.metrics.simplifier.input_faces,
              out.metrics.simplifier.output_faces);
}

TEST(mesh_pipeline_simplify_face_count_in_range) {
    auto m = make_icosphere(4); // ~5120 faces

    MeshPipelineConfig config;
    config.simplifier.target_faces = 500;
    config.simplifier.min_faces = 200;
    config.skip_unwrap = true;

    MeshPipeline pipeline(config);
    auto out = pipeline.process(m);

    ASSERT_GE(out.mesh.face_count(), config.simplifier.min_faces);
    // meshoptimizer may overshoot the target slightly, but should be
    // in the right ballpark.
    ASSERT_LE(out.mesh.face_count(), config.simplifier.target_faces * 2);
}

#else // !HEIMDALL_HAS_MESHOPTIMIZER

TEST_SKIP(mesh_pipeline_rejects_empty_mesh) {}
TEST_SKIP(mesh_pipeline_rejects_invalid_mesh) {}
TEST_SKIP(mesh_pipeline_removes_degenerates) {}
TEST_SKIP(mesh_pipeline_computes_normals_if_missing) {}
TEST_SKIP(mesh_pipeline_skip_simplification_preserves_mesh) {}
TEST_SKIP(mesh_pipeline_metrics_populated) {}
TEST_SKIP(mesh_pipeline_simplify_icosphere) {}
TEST_SKIP(mesh_pipeline_simplify_face_count_in_range) {}

#endif // HEIMDALL_HAS_MESHOPTIMIZER
