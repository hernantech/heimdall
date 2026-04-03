#pragma once

#include <cstdint>
#include <vector>

namespace heimdall::gaussian {

struct CameraInfo {
    int index;
    int serial_number;
    float position[3];     // world position
    float forward[3];      // viewing direction (unit vector)
    float fov_horizontal;  // radians
    bool has_valid_frame;
};

// Select the K best cameras for Gaussian inference.
//
// Strategy for fixed-rig volumetric capture:
//   - Maximize angular coverage around the subject
//   - Ensure baseline diversity (no two cameras too close in angle)
//   - Prefer cameras with valid frames (no dropped frames)
//   - If viewer position known, bias toward cameras near viewer's viewpoint
//
// For live capture without viewer tracking, uses a fixed
// evenly-spaced selection that maximizes coverage.
struct CameraSelection {
    std::vector<int> selected_indices;
    std::vector<float> weights;     // confidence weight per selected camera
};

CameraSelection select_cameras(
    const std::vector<CameraInfo>& cameras,
    int k,
    const float* viewer_position = nullptr  // optional, xyz or null
);

// Precompute a fixed selection for a known rig.
// Call once at startup; result is valid for all frames.
CameraSelection precompute_fixed_selection(
    const std::vector<CameraInfo>& cameras,
    int k
);

} // namespace heimdall::gaussian
