// ============================================================================
// Tests for heimdall::mesh::TriMesh, Vec3, and related helpers
// ============================================================================

#include "test_helpers.h"
#include "../mesh/src/mesh_types.h"

#include <cmath>
#include <vector>

using namespace heimdall::mesh;

// ---------------------------------------------------------------------------
// Helpers — generate simple meshes
// ---------------------------------------------------------------------------

// A single triangle in the XY plane.
static TriMesh make_single_triangle() {
    TriMesh m;
    m.positions = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}};
    m.indices   = {0, 1, 2};
    return m;
}

// A simple unit cube (8 vertices, 12 triangles).
// CCW winding with outward-facing normals.
static TriMesh make_cube() {
    TriMesh m;
    m.positions = {
        {0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0},  // front face (z=0)
        {0, 0, 1}, {1, 0, 1}, {1, 1, 1}, {0, 1, 1},  // back face (z=1)
    };

    m.indices = {
        // Front face (z=0, normal toward -Z)
        0, 3, 2,  0, 2, 1,
        // Back face (z=1, normal toward +Z)
        4, 5, 6,  4, 6, 7,
        // Left face (x=0, normal toward -X)
        0, 4, 7,  0, 7, 3,
        // Right face (x=1, normal toward +X)
        1, 2, 6,  1, 6, 5,
        // Bottom face (y=0, normal toward -Y)
        0, 1, 5,  0, 5, 4,
        // Top face (y=1, normal toward +Y)
        3, 7, 6,  3, 6, 2,
    };

    return m;
}

// ---------------------------------------------------------------------------
// Vec3 tests
// ---------------------------------------------------------------------------

TEST(vec3_addition) {
    Vec3 a{1, 2, 3};
    Vec3 b{4, 5, 6};
    Vec3 c = a + b;
    ASSERT_NEAR(c.x, 5.0f, 1e-6f);
    ASSERT_NEAR(c.y, 7.0f, 1e-6f);
    ASSERT_NEAR(c.z, 9.0f, 1e-6f);
}

TEST(vec3_subtraction) {
    Vec3 a{5, 3, 1};
    Vec3 b{1, 2, 3};
    Vec3 c = a - b;
    ASSERT_NEAR(c.x, 4.0f, 1e-6f);
    ASSERT_NEAR(c.y, 1.0f, 1e-6f);
    ASSERT_NEAR(c.z, -2.0f, 1e-6f);
}

TEST(vec3_scalar_multiply) {
    Vec3 a{2, 3, 4};
    Vec3 b = a * 3.0f;
    ASSERT_NEAR(b.x, 6.0f, 1e-6f);
    ASSERT_NEAR(b.y, 9.0f, 1e-6f);
    ASSERT_NEAR(b.z, 12.0f, 1e-6f);
}

TEST(vec3_length) {
    Vec3 a{3, 4, 0};
    ASSERT_NEAR(a.length(), 5.0f, 1e-6f);

    Vec3 b{1, 1, 1};
    ASSERT_NEAR(b.length(), std::sqrt(3.0f), 1e-6f);
}

TEST(vec3_normalize) {
    Vec3 a{3, 0, 0};
    Vec3 n = a.normalized();
    ASSERT_NEAR(n.x, 1.0f, 1e-6f);
    ASSERT_NEAR(n.y, 0.0f, 1e-6f);
    ASSERT_NEAR(n.z, 0.0f, 1e-6f);

    Vec3 b{1, 1, 1};
    Vec3 nb = b.normalized();
    ASSERT_NEAR(nb.length(), 1.0f, 1e-6f);
}

TEST(vec3_normalize_zero) {
    Vec3 z{0, 0, 0};
    Vec3 n = z.normalized();
    ASSERT_NEAR(n.x, 0.0f, 1e-6f);
    ASSERT_NEAR(n.y, 0.0f, 1e-6f);
    ASSERT_NEAR(n.z, 0.0f, 1e-6f);
}

TEST(vec3_dot_product) {
    Vec3 a{1, 0, 0};
    Vec3 b{0, 1, 0};
    ASSERT_NEAR(dot(a, b), 0.0f, 1e-6f);  // perpendicular

    Vec3 c{1, 2, 3};
    Vec3 d{4, 5, 6};
    ASSERT_NEAR(dot(c, d), 32.0f, 1e-6f);  // 1*4 + 2*5 + 3*6
}

TEST(vec3_cross_product) {
    Vec3 x{1, 0, 0};
    Vec3 y{0, 1, 0};
    Vec3 z = cross(x, y);
    ASSERT_NEAR(z.x, 0.0f, 1e-6f);
    ASSERT_NEAR(z.y, 0.0f, 1e-6f);
    ASSERT_NEAR(z.z, 1.0f, 1e-6f);

    // Cross product is anti-commutative: a x b = -(b x a).
    Vec3 z2 = cross(y, x);
    ASSERT_NEAR(z2.z, -1.0f, 1e-6f);
}

// ---------------------------------------------------------------------------
// TriMesh tests
// ---------------------------------------------------------------------------

TEST(trimesh_vertex_face_count) {
    auto m = make_cube();
    ASSERT_EQ(m.vertex_count(), static_cast<size_t>(8));
    ASSERT_EQ(m.face_count(), static_cast<size_t>(12));
}

TEST(trimesh_is_valid) {
    auto m = make_cube();
    ASSERT_TRUE(m.is_valid());

    // Invalid: index out of range.
    TriMesh bad = m;
    bad.indices.push_back(100); // index > vertex count
    bad.indices.push_back(0);
    bad.indices.push_back(1);
    ASSERT_FALSE(bad.is_valid());
}

TEST(trimesh_is_valid_bad_index_count) {
    // indices.size() not a multiple of 3.
    TriMesh m;
    m.positions = {{0, 0, 0}, {1, 0, 0}};
    m.indices   = {0, 1}; // only 2 indices
    ASSERT_FALSE(m.is_valid());
}

TEST(trimesh_is_valid_inconsistent_normals) {
    auto m = make_single_triangle();
    m.normals = {{0, 0, 1}}; // 1 normal for 3 vertices -> invalid
    ASSERT_FALSE(m.is_valid());
}

TEST(trimesh_from_flat_roundtrip) {
    // Build a mesh via from_flat, extract flat arrays, compare.
    std::vector<float> flat_pos = {0, 0, 0,  1, 0, 0,  0, 1, 0,  1, 1, 0};
    std::vector<uint32_t> tri_idx = {0, 1, 2,  1, 3, 2};
    std::vector<float> flat_norms = {0, 0, 1,  0, 0, 1,  0, 0, 1,  0, 0, 1};
    std::vector<float> flat_uv = {0, 0,  1, 0,  0, 1,  1, 1};
    std::vector<uint8_t> flat_col = {255, 0, 0,  0, 255, 0,  0, 0, 255,  255, 255, 0};

    auto m = TriMesh::from_flat(flat_pos, tri_idx, flat_norms, flat_uv, flat_col);

    ASSERT_EQ(m.vertex_count(), static_cast<size_t>(4));
    ASSERT_EQ(m.face_count(), static_cast<size_t>(2));
    ASSERT_TRUE(m.has_normals());
    ASSERT_TRUE(m.has_texcoords());
    ASSERT_TRUE(m.has_colors());
    ASSERT_TRUE(m.is_valid());

    // Round-trip: flat_positions
    auto rt_pos = m.flat_positions();
    ASSERT_EQ(rt_pos.size(), flat_pos.size());
    for (size_t i = 0; i < flat_pos.size(); ++i) {
        ASSERT_NEAR(rt_pos[i], flat_pos[i], 1e-6f);
    }

    // Round-trip: flat_normals
    auto rt_norms = m.flat_normals();
    ASSERT_EQ(rt_norms.size(), flat_norms.size());
    for (size_t i = 0; i < flat_norms.size(); ++i) {
        ASSERT_NEAR(rt_norms[i], flat_norms[i], 1e-6f);
    }

    // Round-trip: flat_texcoords
    auto rt_uv = m.flat_texcoords();
    ASSERT_EQ(rt_uv.size(), flat_uv.size());
    for (size_t i = 0; i < flat_uv.size(); ++i) {
        ASSERT_NEAR(rt_uv[i], flat_uv[i], 1e-6f);
    }

    // Round-trip: flat_colors
    auto rt_col = m.flat_colors();
    ASSERT_EQ(rt_col.size(), flat_col.size());
    for (size_t i = 0; i < flat_col.size(); ++i) {
        ASSERT_EQ(rt_col[i], flat_col[i]);
    }

    // Indices preserved.
    ASSERT_EQ(m.indices.size(), tri_idx.size());
    for (size_t i = 0; i < tri_idx.size(); ++i) {
        ASSERT_EQ(m.indices[i], tri_idx[i]);
    }
}

TEST(trimesh_from_flat_positions_only) {
    // from_flat with no optional attributes.
    std::vector<float> flat_pos = {0, 0, 0,  1, 0, 0,  0, 1, 0};
    std::vector<uint32_t> tri_idx = {0, 1, 2};

    auto m = TriMesh::from_flat(flat_pos, tri_idx);

    ASSERT_EQ(m.vertex_count(), static_cast<size_t>(3));
    ASSERT_FALSE(m.has_normals());
    ASSERT_FALSE(m.has_texcoords());
    ASSERT_FALSE(m.has_colors());
    ASSERT_TRUE(m.is_valid());
}

TEST(trimesh_remove_degenerate_faces_duplicate_indices) {
    TriMesh m;
    m.positions = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {1, 1, 0}};
    m.indices = {
        0, 1, 2,   // good
        0, 0, 1,   // degenerate (two identical indices)
        1, 3, 2,   // good
    };
    ASSERT_EQ(m.face_count(), static_cast<size_t>(3));

    size_t removed = m.remove_degenerate_faces();
    ASSERT_EQ(removed, static_cast<size_t>(1));
    ASSERT_EQ(m.face_count(), static_cast<size_t>(2));
}

TEST(trimesh_remove_degenerate_faces_zero_area) {
    // Three distinct vertices, but collinear -> zero-area triangle.
    TriMesh m;
    m.positions = {{0, 0, 0}, {1, 0, 0}, {2, 0, 0}, {0, 1, 0}};
    m.indices = {
        0, 1, 2,   // degenerate: collinear
        0, 1, 3,   // good
    };

    size_t removed = m.remove_degenerate_faces();
    ASSERT_EQ(removed, static_cast<size_t>(1));
    ASSERT_EQ(m.face_count(), static_cast<size_t>(1));
}

TEST(trimesh_remove_degenerate_faces_none) {
    auto m = make_single_triangle();
    size_t removed = m.remove_degenerate_faces();
    ASSERT_EQ(removed, static_cast<size_t>(0));
    ASSERT_EQ(m.face_count(), static_cast<size_t>(1));
}

TEST(trimesh_recompute_normals_single_triangle) {
    auto m = make_single_triangle();
    m.recompute_normals();

    ASSERT_TRUE(m.has_normals());
    ASSERT_EQ(m.normals.size(), m.positions.size());

    // For a CCW triangle in the XY plane, the normal should point +Z.
    for (auto& n : m.normals) {
        ASSERT_NEAR(n.length(), 1.0f, 1e-5f);
        ASSERT_NEAR(n.z, 1.0f, 1e-5f);
    }
}

TEST(trimesh_recompute_normals_cube_unit_normals) {
    auto m = make_cube();
    m.recompute_normals();

    ASSERT_TRUE(m.has_normals());

    // All vertex normals should be unit length.
    for (auto& n : m.normals) {
        ASSERT_NEAR(n.length(), 1.0f, 1e-4f);
    }
}

TEST(trimesh_recompute_normals_cube_outward) {
    auto m = make_cube();
    m.recompute_normals();

    // For each vertex, the normal should generally point away from the
    // center of the cube (0.5, 0.5, 0.5).
    Vec3 center{0.5f, 0.5f, 0.5f};
    for (size_t i = 0; i < m.vertex_count(); ++i) {
        Vec3 to_vertex = m.positions[i] - center;
        float d = dot(to_vertex, m.normals[i]);
        ASSERT_GT(d, 0.0f);  // normal points outward
    }
}

TEST(trimesh_empty) {
    TriMesh m;
    ASSERT_TRUE(m.empty());
    ASSERT_EQ(m.vertex_count(), static_cast<size_t>(0));
    ASSERT_EQ(m.face_count(), static_cast<size_t>(0));
    ASSERT_TRUE(m.is_valid());
}
