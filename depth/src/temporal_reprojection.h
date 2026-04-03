#pragma once

#include <cuda_runtime.h>
#include <cstdint>

namespace heimdall::depth {

// Reproject the previous frame's depth map into the current frame's
// camera view to seed the depth solver with a temporal prior.
// This enables reducing PatchMatch / learned stereo iterations
// from ~16 to ~4-8 without quality loss.

struct ReprojectionParams {
    int width;
    int height;
    float fx, fy, cx, cy;           // current frame intrinsics
    float prev_fx, prev_fy, prev_cx, prev_cy;  // previous frame intrinsics

    // 4x4 transforms (row-major)
    float prev_cam_to_world[16];    // previous camera → world
    float world_to_curr_cam[16];    // world → current camera

    float depth_min_m;              // minimum valid depth (meters)
    float depth_max_m;              // maximum valid depth (meters)
    float confidence_threshold;     // discard low-confidence reprojections
    float margin_m;                 // search range around reprojected depth
};

// Launch the reprojection kernel.
// prev_depth: CV_32FC1 depth map from frame N-1 (meters)
// prev_confidence: CV_32FC1 confidence from frame N-1 (0-1)
// out_seed_depth: CV_32FC1 reprojected depth seed for frame N
// out_seed_confidence: CV_32FC1 reprojected confidence
// All pointers are CUDA device memory.
void launch_temporal_reprojection(
    const float* prev_depth,
    const float* prev_confidence,
    float* out_seed_depth,
    float* out_seed_confidence,
    const ReprojectionParams& params,
    cudaStream_t stream = nullptr
);

} // namespace heimdall::depth
