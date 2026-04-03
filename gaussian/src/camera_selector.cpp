#include "camera_selector.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

namespace heimdall::gaussian {
namespace {

// ---------------------------------------------------------------------------
// Vector math helpers (3-component, stack-only)
// ---------------------------------------------------------------------------

struct Vec3 {
    float x, y, z;
};

inline float dot(Vec3 a, Vec3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

inline float length(Vec3 v) {
    return std::sqrt(dot(v, v));
}

inline Vec3 normalize(Vec3 v) {
    float len = length(v);
    if (len < 1e-12f) return {0.0f, 0.0f, 0.0f};
    float inv = 1.0f / len;
    return {v.x * inv, v.y * inv, v.z * inv};
}

// Angular distance between two unit vectors, in radians [0, pi].
inline float angular_distance(Vec3 a, Vec3 b) {
    float d = dot(a, b);
    // Clamp to [-1, 1] to guard against floating-point overshoot.
    d = std::max(-1.0f, std::min(1.0f, d));
    return std::acos(d);
}

// ---------------------------------------------------------------------------
// Direction from camera position toward the rig center (origin).
//
// In a volumetric capture rig the subject stands at the origin and cameras
// surround them. The "direction" of a camera on the unit sphere is the
// normalized vector from the camera position toward the origin, which is
// simply -normalize(position). This gives us a point on the unit sphere that
// represents where the camera is looking *from*, not where it is looking *at*.
//
// Why not use CameraInfo::forward directly?
//   forward is the optical-axis direction (camera-to-scene), which is useful
//   for FOV calculations but not for angular-coverage sampling. Two cameras
//   at 90 degrees apart on the rig circle could have identical forward
//   vectors if they are both aimed at the same point. We want the bearing
//   *from* the subject to the camera, which captures geometric diversity.
// ---------------------------------------------------------------------------
inline Vec3 camera_direction_on_sphere(const CameraInfo& cam) {
    // direction = normalize(origin - position) = normalize(-position)
    return normalize({-cam.position[0], -cam.position[1], -cam.position[2]});
}

// ---------------------------------------------------------------------------
// Greedy farthest-point sampling on the unit sphere.
//
// Algorithm:
//   1. Start with a seed camera (the one closest to a desired seed direction,
//      or the first valid camera if no preference).
//   2. Maintain a "min distance to selected set" array over all candidates.
//   3. At each step pick the candidate with the largest min-distance — this
//      is the point farthest from everything already selected, maximizing
//      angular spread.
//   4. Repeat until K cameras are selected.
//
// Complexity: O(K * N), which is negligible for typical rigs (N <= 100).
// ---------------------------------------------------------------------------

CameraSelection farthest_point_sample(
    const std::vector<CameraInfo>& cameras,
    int k,
    int seed_index,
    const std::vector<Vec3>& directions)
{
    const int n = static_cast<int>(cameras.size());

    // min_dist[i] = angular distance from camera i to the nearest selected camera.
    // Initialized to infinity; updated each time we select a new camera.
    std::vector<float> min_dist(n, std::numeric_limits<float>::infinity());

    // Track which cameras are selected.
    std::vector<bool> selected(n, false);
    std::vector<int> result;
    result.reserve(k);

    // --- Select the seed ---
    selected[seed_index] = true;
    result.push_back(seed_index);

    // Update min_dist with respect to the seed.
    for (int i = 0; i < n; ++i) {
        if (!cameras[i].has_valid_frame) continue;
        float d = angular_distance(directions[seed_index], directions[i]);
        min_dist[i] = std::min(min_dist[i], d);
    }

    // --- Greedily select remaining K-1 cameras ---
    for (int step = 1; step < k; ++step) {
        // Find the unselected, valid camera with the largest min_dist.
        int best = -1;
        float best_dist = -1.0f;
        for (int i = 0; i < n; ++i) {
            if (selected[i] || !cameras[i].has_valid_frame) continue;
            if (min_dist[i] > best_dist) {
                best_dist = min_dist[i];
                best = i;
            }
        }
        if (best < 0) break;  // Fewer valid cameras than K.

        selected[best] = true;
        result.push_back(best);

        // Update min_dist w.r.t. newly selected camera.
        for (int i = 0; i < n; ++i) {
            if (selected[i] || !cameras[i].has_valid_frame) continue;
            float d = angular_distance(directions[best], directions[i]);
            min_dist[i] = std::min(min_dist[i], d);
        }
    }

    return CameraSelection{std::move(result), {}};
}

// ---------------------------------------------------------------------------
// Compute per-camera weights proportional to angular isolation.
//
// A camera that is far from all its selected neighbors contributes a more
// unique viewpoint and should receive higher weight in blending.
//
// weight_i = min angular distance from camera i to any other selected camera.
// Weights are then normalized to sum to 1.
// ---------------------------------------------------------------------------
std::vector<float> compute_weights(
    const std::vector<int>& selected_indices,
    const std::vector<Vec3>& directions)
{
    const int k = static_cast<int>(selected_indices.size());
    std::vector<float> weights(k);

    for (int i = 0; i < k; ++i) {
        float min_d = std::numeric_limits<float>::infinity();
        for (int j = 0; j < k; ++j) {
            if (i == j) continue;
            float d = angular_distance(
                directions[selected_indices[i]],
                directions[selected_indices[j]]);
            min_d = std::min(min_d, d);
        }
        // For a single selected camera, give it weight 1.
        weights[i] = (k == 1) ? 1.0f : min_d;
    }

    // Normalize to sum to 1.
    float sum = std::accumulate(weights.begin(), weights.end(), 0.0f);
    if (sum > 1e-12f) {
        float inv = 1.0f / sum;
        for (float& w : weights) w *= inv;
    } else {
        // Degenerate case (all cameras co-located): uniform weights.
        float uniform = 1.0f / static_cast<float>(k);
        std::fill(weights.begin(), weights.end(), uniform);
    }

    return weights;
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// select_cameras — main entry point
//
// If viewer_position is provided, we bias the seed toward the viewer so that
// the selected set includes cameras near the viewer's viewpoint. This improves
// rendering quality for the viewer's particular angle.
//
// The bias is applied by choosing the seed camera to be the one whose
// direction on the unit sphere is closest to the viewer's direction. The
// farthest-point sampling then fills in coverage around that seed, so the
// result is a set that (a) includes the viewer's neighborhood and (b) still
// has good overall angular spread.
// ---------------------------------------------------------------------------
CameraSelection select_cameras(
    const std::vector<CameraInfo>& cameras,
    int k,
    const float* viewer_position)
{
    if (cameras.empty() || k <= 0) {
        return {{}, {}};
    }

    const int n = static_cast<int>(cameras.size());

    // Precompute unit-sphere directions for all cameras.
    std::vector<Vec3> directions(n);
    for (int i = 0; i < n; ++i) {
        directions[i] = camera_direction_on_sphere(cameras[i]);
    }

    // Count valid cameras and clamp k.
    int valid_count = 0;
    for (int i = 0; i < n; ++i) {
        if (cameras[i].has_valid_frame) ++valid_count;
    }
    k = std::min(k, valid_count);
    if (k <= 0) {
        return {{}, {}};
    }

    // --- Choose the seed camera ---
    int seed = -1;

    if (viewer_position) {
        // Viewer direction on the unit sphere: the direction from the viewer
        // toward the origin (subject). This matches our camera direction
        // convention so we can compare them directly.
        Vec3 viewer_dir = normalize({
            -viewer_position[0],
            -viewer_position[1],
            -viewer_position[2]
        });

        // Find the valid camera whose direction is closest to the viewer.
        float best_dist = std::numeric_limits<float>::infinity();
        for (int i = 0; i < n; ++i) {
            if (!cameras[i].has_valid_frame) continue;
            float d = angular_distance(directions[i], viewer_dir);
            if (d < best_dist) {
                best_dist = d;
                seed = i;
            }
        }
    } else {
        // No viewer preference — pick the first valid camera as an arbitrary
        // but deterministic seed.
        for (int i = 0; i < n; ++i) {
            if (cameras[i].has_valid_frame) {
                seed = i;
                break;
            }
        }
    }

    // Perform farthest-point sampling starting from the seed.
    CameraSelection selection = farthest_point_sample(cameras, k, seed, directions);

    // Compute isolation-based weights.
    selection.weights = compute_weights(selection.selected_indices, directions);

    return selection;
}

// ---------------------------------------------------------------------------
// precompute_fixed_selection — for fixed rigs, compute once at startup
//
// Identical to select_cameras with no viewer bias. The result can be cached
// and reused for every frame, with per-frame validity checks applied
// afterwards by the caller (skip cameras whose has_valid_frame flipped to
// false due to a dropped frame).
// ---------------------------------------------------------------------------
CameraSelection precompute_fixed_selection(
    const std::vector<CameraInfo>& cameras,
    int k)
{
    // For precomputation we treat all cameras as valid — the rig layout is
    // fixed and frame drops are transient. Create a copy with all frames
    // marked valid so that farthest-point sampling considers the full
    // geometric coverage.
    std::vector<CameraInfo> all_valid = cameras;
    for (auto& cam : all_valid) {
        cam.has_valid_frame = true;
    }

    return select_cameras(all_valid, k, /*viewer_position=*/nullptr);
}

} // namespace heimdall::gaussian
