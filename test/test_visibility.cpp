// ============================================================================
// Tests for heimdall::texture::VisibilityTester (BVH + ray casting)
// ============================================================================

#include "test_helpers.h"
#include "../texture/src/visibility.h"

#include <cmath>
#include <vector>

using namespace heimdall::texture;

// ---------------------------------------------------------------------------
// Helpers — simple triangle scenes
// ---------------------------------------------------------------------------

// A single triangle in the XY plane at z=0:  (0,0,0), (1,0,0), (0,1,0).
struct SimpleScene {
    std::vector<float>    positions;
    std::vector<uint32_t> indices;

    size_t num_vertices()  const { return positions.size() / 3; }
    size_t num_triangles() const { return indices.size() / 3; }
};

static SimpleScene make_single_triangle_scene() {
    SimpleScene s;
    s.positions = {0, 0, 0,   1, 0, 0,   0, 1, 0};
    s.indices   = {0, 1, 2};
    return s;
}

// Two triangles: one at z=0, one at z=1 (parallel, both in the XY plane).
// The z=1 triangle occludes the z=0 triangle when viewed from z > 1.
static SimpleScene make_two_parallel_triangles() {
    SimpleScene s;
    s.positions = {
        // Triangle 0 at z=0
        0, 0, 0,   1, 0, 0,   0, 1, 0,
        // Triangle 1 at z=1
        0, 0, 1,   1, 0, 1,   0, 1, 1,
    };
    s.indices = {0, 1, 2,  3, 4, 5};
    return s;
}

// 4-triangle scene forming a simple "room": floor + 3 walls.
static SimpleScene make_4tri_scene() {
    SimpleScene s;
    s.positions = {
        // Floor triangle 0
        -1, 0, -1,   1, 0, -1,   1, 0,  1,
        // Floor triangle 1
        -1, 0, -1,   1, 0,  1,  -1, 0,  1,
        // Wall at x=1 (vertical)
         1, 0, -1,   1, 2, -1,   1, 0,  1,
        // Wall at z=-1 (vertical)
        -1, 0, -1,  -1, 2, -1,   1, 0, -1,
    };
    s.indices = {
        0, 1, 2,
        3, 4, 5,
        6, 7, 8,
        9, 10, 11,
    };
    return s;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

TEST(visibility_bvh_construction) {
    auto scene = make_4tri_scene();
    VisibilityTester vt;
    vt.build(scene.positions.data(), scene.num_vertices(),
             scene.indices.data(), scene.num_triangles());

    ASSERT_TRUE(vt.is_built());
    ASSERT_EQ(vt.num_triangles(), static_cast<size_t>(4));
}

TEST(visibility_bvh_empty_scene) {
    VisibilityTester vt;
    // Not built yet.
    ASSERT_FALSE(vt.is_built());

    // is_visible on an unbuilt BVH should return true (no occluders).
    bool vis = vt.is_visible({0, 0, 0}, {0, 0, 5}, 0);
    ASSERT_TRUE(vis);
}

TEST(visibility_ray_hits_triangle_center) {
    // Ray from z=5 toward origin, hitting the triangle at z=0.
    auto scene = make_single_triangle_scene();
    VisibilityTester vt;
    vt.build(scene.positions.data(), scene.num_vertices(),
             scene.indices.data(), scene.num_triangles());

    // Surface point at the centroid of the triangle.
    Vec3 centroid{1.0f / 3.0f, 1.0f / 3.0f, 0.0f};
    Vec3 camera{1.0f / 3.0f, 1.0f / 3.0f, 5.0f};

    // The point is ON face 0 — it should be visible (self-face excluded).
    bool vis = vt.is_visible(centroid, camera, /*surface_face_idx=*/0);
    ASSERT_TRUE(vis);
}

TEST(visibility_ray_misses_triangle) {
    // Ray from a point far outside the triangle's footprint.
    auto scene = make_single_triangle_scene();
    VisibilityTester vt;
    vt.build(scene.positions.data(), scene.num_vertices(),
             scene.indices.data(), scene.num_triangles());

    // Surface point at (5, 5, 0) — far from the triangle.
    // Camera at (5, 5, 3).
    // No triangle is in the way (the single triangle is near origin).
    Vec3 surface{5.0f, 5.0f, 0.0f};
    Vec3 camera{5.0f, 5.0f, 3.0f};

    bool vis = vt.is_visible(surface, camera, /*surface_face_idx=*/0);
    ASSERT_TRUE(vis);
}

TEST(visibility_ray_occluded_by_intervening_triangle) {
    // Two parallel triangles at z=0 and z=1.
    // Surface point on the z=0 triangle, camera at z=5.
    // The z=1 triangle occludes the line of sight.
    auto scene = make_two_parallel_triangles();
    VisibilityTester vt;
    vt.build(scene.positions.data(), scene.num_vertices(),
             scene.indices.data(), scene.num_triangles());

    Vec3 surface{0.25f, 0.25f, 0.0f};
    Vec3 camera{0.25f, 0.25f, 5.0f};

    // Surface is on face 0 (z=0 triangle). Face 1 (z=1) is in the way.
    bool vis = vt.is_visible(surface, camera, /*surface_face_idx=*/0);
    ASSERT_FALSE(vis);
}

TEST(visibility_not_occluded_when_camera_between_triangles) {
    // Camera between the two triangles (z=0.5), surface on z=0.
    // The z=1 triangle is behind the camera, so it should not occlude.
    auto scene = make_two_parallel_triangles();
    VisibilityTester vt;
    vt.build(scene.positions.data(), scene.num_vertices(),
             scene.indices.data(), scene.num_triangles());

    Vec3 surface{0.25f, 0.25f, 0.0f};
    Vec3 camera{0.25f, 0.25f, 0.5f};

    bool vis = vt.is_visible(surface, camera, /*surface_face_idx=*/0);
    ASSERT_TRUE(vis);
}

TEST(visibility_self_occlusion_excluded) {
    // A surface point on the triangle itself should not self-occlude.
    auto scene = make_single_triangle_scene();
    VisibilityTester vt;
    vt.build(scene.positions.data(), scene.num_vertices(),
             scene.indices.data(), scene.num_triangles());

    // Point exactly on the triangle surface (face 0), looking from below.
    Vec3 surface{0.1f, 0.1f, 0.0f};
    Vec3 camera{0.1f, 0.1f, -5.0f};

    bool vis = vt.is_visible(surface, camera, /*surface_face_idx=*/0);
    ASSERT_TRUE(vis);
}

TEST(visibility_4tri_scene_floor_visible_from_above) {
    // Camera above the floor, looking at a floor point.
    // No walls in the way.
    auto scene = make_4tri_scene();
    VisibilityTester vt;
    vt.build(scene.positions.data(), scene.num_vertices(),
             scene.indices.data(), scene.num_triangles());

    Vec3 floor_point{0.0f, 0.0f, 0.0f};
    Vec3 camera_above{0.0f, 5.0f, 0.0f};

    // Floor is face 0.
    bool vis = vt.is_visible(floor_point, camera_above, /*surface_face_idx=*/0);
    ASSERT_TRUE(vis);
}

TEST(visibility_4tri_scene_floor_occluded_by_wall) {
    // Camera behind the wall at x=1. A floor point at x=0.5 should be
    // occluded by the wall triangle at x=1.
    auto scene = make_4tri_scene();
    VisibilityTester vt;
    vt.build(scene.positions.data(), scene.num_vertices(),
             scene.indices.data(), scene.num_triangles());

    // Floor point at the center of the floor.
    Vec3 floor_point{0.0f, 0.0f, 0.0f};
    // Camera behind the wall at x=1, slightly elevated.
    Vec3 camera{5.0f, 0.5f, 0.0f};

    // The wall triangle at x=1 (face 2) blocks the view.
    bool vis = vt.is_visible(floor_point, camera, /*surface_face_idx=*/0);
    ASSERT_FALSE(vis);
}
