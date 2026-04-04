// ============================================================================
// Tests for heimdall::gaussian::CameraSelection / select_cameras
// ============================================================================

#include "test_helpers.h"
#include "../gaussian/src/camera_selector.h"

#include <array>
#include <cmath>
#include <numeric>
#include <set>
#include <vector>

using namespace heimdall::gaussian;

// ---------------------------------------------------------------------------
// Helpers — generate synthetic camera rigs
// ---------------------------------------------------------------------------

static constexpr float kPi = 3.14159265358979323846f;

// Build N cameras evenly spaced on a horizontal circle of given radius,
// all at y=0, looking inward toward the origin.
static std::vector<CameraInfo> make_ring(int n, float radius = 3.0f) {
    std::vector<CameraInfo> cameras(n);
    for (int i = 0; i < n; ++i) {
        float angle = 2.0f * kPi * static_cast<float>(i) / static_cast<float>(n);
        float cx = radius * std::cos(angle);
        float cz = radius * std::sin(angle);

        cameras[i].index            = i;
        cameras[i].serial_number    = 1000 + i;
        cameras[i].position[0]      = cx;
        cameras[i].position[1]      = 0.0f;
        cameras[i].position[2]      = cz;
        // Forward direction: toward origin.
        float len = std::sqrt(cx * cx + cz * cz);
        cameras[i].forward[0]       = -cx / len;
        cameras[i].forward[1]       = 0.0f;
        cameras[i].forward[2]       = -cz / len;
        cameras[i].fov_horizontal   = 1.22f; // ~70 degrees
        cameras[i].has_valid_frame  = true;
    }
    return cameras;
}

// Angular distance between two camera positions on the unit sphere
// (using the -normalize(position) convention from camera_selector.cpp).
static float cam_angular_dist(const CameraInfo& a, const CameraInfo& b) {
    auto norm = [](float x, float y, float z) -> std::array<float, 3> {
        float len = std::sqrt(x * x + y * y + z * z);
        if (len < 1e-12f) return {0, 0, 0};
        return {-x / len, -y / len, -z / len};
    };
    auto da = norm(a.position[0], a.position[1], a.position[2]);
    auto db = norm(b.position[0], b.position[1], b.position[2]);
    float d = da[0] * db[0] + da[1] * db[1] + da[2] * db[2];
    d = std::max(-1.0f, std::min(1.0f, d));
    return std::acos(d);
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

TEST(camera_select_k4_from_8_evenly_spaced) {
    // 8 cameras evenly spaced on a circle. Selecting K=4 should give
    // cameras approximately 90 degrees apart (indices {0,2,4,6} or a
    // rotated variant).

    auto cameras = make_ring(8);
    auto sel = select_cameras(cameras, 4);

    ASSERT_EQ(static_cast<int>(sel.selected_indices.size()), 4);

    // Compute the minimum angular separation between any two selected cameras.
    float min_sep = 1e30f;
    for (int i = 0; i < 4; ++i) {
        for (int j = i + 1; j < 4; ++j) {
            float d = cam_angular_dist(cameras[sel.selected_indices[i]],
                                       cameras[sel.selected_indices[j]]);
            min_sep = std::min(min_sep, d);
        }
    }

    // For 4 cameras from a ring of 8, the ideal minimum separation is
    // 90 degrees (pi/2). Allow a tolerance since the farthest-point
    // sampling is greedy and the exact result depends on seed choice.
    // The key check: min_sep should be much larger than for 4 adjacent cameras
    // (which would be ~45 degrees).
    float expected_min = kPi / 2.0f;  // 90 degrees
    ASSERT_GT(min_sep, expected_min * 0.85f);  // at least ~76 degrees
}

TEST(camera_select_skips_invalid_frames) {
    // Mark some cameras as having dropped frames. Those must not appear
    // in the selection.

    auto cameras = make_ring(8);
    cameras[0].has_valid_frame = false;
    cameras[1].has_valid_frame = false;
    cameras[2].has_valid_frame = false;

    auto sel = select_cameras(cameras, 4);

    ASSERT_EQ(static_cast<int>(sel.selected_indices.size()), 4);

    // None of the selected should be an invalid camera.
    for (int idx : sel.selected_indices) {
        ASSERT_TRUE(cameras[idx].has_valid_frame);
    }
}

TEST(camera_select_returns_fewer_if_not_enough_valid) {
    // Request K=6 but only 3 cameras have valid frames.
    auto cameras = make_ring(8);
    for (int i = 3; i < 8; ++i) {
        cameras[i].has_valid_frame = false;
    }

    auto sel = select_cameras(cameras, 6);

    // Should get at most 3.
    ASSERT_EQ(static_cast<int>(sel.selected_indices.size()), 3);
}

TEST(camera_select_viewer_bias_selects_nearest) {
    // Place the viewer at position (3, 0, 0) — on the +X axis.
    // Camera 0 is at angle 0 (also on +X axis at radius 3).
    // With viewer bias, camera 0 should always be selected (as the seed).

    auto cameras = make_ring(8);
    float viewer_pos[3] = {3.0f, 0.0f, 0.0f};

    auto sel = select_cameras(cameras, 4, viewer_pos);

    ASSERT_EQ(static_cast<int>(sel.selected_indices.size()), 4);

    // Camera 0 should be in the selection (it is the nearest to the viewer).
    bool found_cam0 = false;
    for (int idx : sel.selected_indices) {
        if (idx == 0) found_cam0 = true;
    }
    ASSERT_TRUE(found_cam0);
}

TEST(camera_select_viewer_bias_changes_seed) {
    // Two different viewer positions should (potentially) give different seeds
    // and thus different first selected cameras.

    auto cameras = make_ring(8);

    // Viewer on +X axis -> seed near camera 0.
    float viewer_a[3] = {5.0f, 0.0f, 0.0f};
    auto sel_a = select_cameras(cameras, 2, viewer_a);

    // Viewer on -X axis -> seed near camera 4.
    float viewer_b[3] = {-5.0f, 0.0f, 0.0f};
    auto sel_b = select_cameras(cameras, 2, viewer_b);

    // The first selected camera (the seed) should differ.
    ASSERT_NE(sel_a.selected_indices[0], sel_b.selected_indices[0]);
}

TEST(camera_select_weights_sum_to_one) {
    auto cameras = make_ring(8);
    auto sel = select_cameras(cameras, 4);

    ASSERT_EQ(static_cast<int>(sel.weights.size()),
              static_cast<int>(sel.selected_indices.size()));

    float sum = 0.0f;
    for (float w : sel.weights) {
        ASSERT_GT(w, 0.0f);  // all weights positive
        sum += w;
    }
    ASSERT_NEAR(sum, 1.0f, 1e-5f);
}

TEST(camera_select_weights_uniform_for_symmetric_rig) {
    // For K=4 from a ring of 8, if the 4 selected cameras are evenly
    // spaced, their weights should be approximately equal (all ~0.25).

    auto cameras = make_ring(8);
    auto sel = select_cameras(cameras, 4);

    float expected_w = 1.0f / 4.0f;
    for (float w : sel.weights) {
        ASSERT_NEAR(w, expected_w, 0.05f);
    }
}

TEST(camera_precompute_fixed_selection_consistent) {
    // precompute_fixed_selection should return the same result each time.
    auto cameras = make_ring(8);

    auto sel1 = precompute_fixed_selection(cameras, 4);
    auto sel2 = precompute_fixed_selection(cameras, 4);

    ASSERT_EQ(sel1.selected_indices.size(), sel2.selected_indices.size());
    for (size_t i = 0; i < sel1.selected_indices.size(); ++i) {
        ASSERT_EQ(sel1.selected_indices[i], sel2.selected_indices[i]);
    }

    ASSERT_EQ(sel1.weights.size(), sel2.weights.size());
    for (size_t i = 0; i < sel1.weights.size(); ++i) {
        ASSERT_NEAR(sel1.weights[i], sel2.weights[i], 1e-6f);
    }
}

TEST(camera_precompute_ignores_invalid_frames) {
    // precompute_fixed_selection treats all cameras as valid (copies with
    // has_valid_frame=true), so even cameras marked invalid should be
    // considered for precomputation.

    auto cameras = make_ring(8);
    cameras[0].has_valid_frame = false;
    cameras[4].has_valid_frame = false;

    auto sel = precompute_fixed_selection(cameras, 4);

    ASSERT_EQ(static_cast<int>(sel.selected_indices.size()), 4);

    // The result should be identical to precomputing with all valid,
    // because precompute_fixed_selection forces all cameras valid.
    auto cameras_all_valid = make_ring(8);
    auto sel_ref = precompute_fixed_selection(cameras_all_valid, 4);

    for (size_t i = 0; i < sel.selected_indices.size(); ++i) {
        ASSERT_EQ(sel.selected_indices[i], sel_ref.selected_indices[i]);
    }
}

TEST(camera_select_empty_input) {
    std::vector<CameraInfo> empty;
    auto sel = select_cameras(empty, 4);
    ASSERT_TRUE(sel.selected_indices.empty());
    ASSERT_TRUE(sel.weights.empty());
}

TEST(camera_select_k_zero) {
    auto cameras = make_ring(8);
    auto sel = select_cameras(cameras, 0);
    ASSERT_TRUE(sel.selected_indices.empty());
}
