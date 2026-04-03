#pragma once

#include "tsdf_volume.h"   // FusionCameraParams, ExtractedPoints
#include <cuda_runtime.h>
#include <cstdint>
#include <vector>

namespace heimdall::fusion {

// Configuration for multi-view point cloud fusion.
struct PointCloudFusionConfig {
    float depth_min = 0.2f;              // minimum valid depth (meters)
    float depth_max = 5.0f;              // maximum valid depth (meters)
    int min_consistent_views = 2;        // minimum agreeing views to keep a point
    float consistency_threshold = 0.01f; // depth agreement threshold (meters)
    float normal_consistency_deg = 60.0f;// max angle between normals to count as consistent
    bool weight_by_angle = true;         // weight contributions by view-surface angle
    int max_points = 4'000'000;          // hard cap on output points
};

// Per-camera depth + color data living on the GPU.
struct DepthViewGpu {
    int camera_index;
    int width;
    int height;
    const float* depth_gpu;              // float32 meters, width*height
    const uint8_t* color_gpu;            // RGBA uint8, width*height*4 (may be nullptr)
    const uint8_t* alpha_gpu;            // uint8 foreground mask, width*height (may be nullptr)
    FusionCameraParams camera;
};

// Direct multi-view point cloud fusion on GPU.
//
// Unlike TSDF, this does not discretize into a voxel grid. Instead:
//   1. For each camera, unproject foreground depth pixels to 3D world coordinates.
//   2. Compute per-pixel normals from depth gradients.
//   3. For each candidate 3D point, reproject into all other cameras and check
//      depth agreement (multi-view consistency filter).
//   4. Points surviving the consistency check are output with averaged normals/colors.
//
// This approach preserves more fine detail than TSDF at the cost of producing
// a less clean surface. Suitable for point-cloud-based rendering or as input
// to Poisson surface reconstruction.
class PointCloudFusion {
public:
    explicit PointCloudFusion(const PointCloudFusionConfig& config);
    ~PointCloudFusion();

    PointCloudFusion(const PointCloudFusion&) = delete;
    PointCloudFusion& operator=(const PointCloudFusion&) = delete;

    // Fuse multiple depth views into a single point cloud.
    // All depth/color/alpha pointers in views must be valid GPU device memory.
    ExtractedPoints fuse(const std::vector<DepthViewGpu>& views,
                         cudaStream_t stream = nullptr);

private:
    PointCloudFusionConfig config_;

    // GPU scratch buffers
    float* d_candidate_positions_ = nullptr;    // Nx3
    float* d_candidate_normals_ = nullptr;      // Nx3
    uint8_t* d_candidate_colors_ = nullptr;     // Nx3
    int* d_candidate_views_ = nullptr;          // per-point source camera index
    int* d_candidate_count_ = nullptr;          // atomic counter
    int* d_consistency_counts_ = nullptr;        // per-candidate agreeing view count

    // Flattened camera params for GPU access during reprojection
    FusionCameraParams* d_cameras_ = nullptr;
    int max_cameras_allocated_ = 0;

    // Output buffers
    float* d_out_positions_ = nullptr;
    float* d_out_normals_ = nullptr;
    uint8_t* d_out_colors_ = nullptr;
    int* d_out_count_ = nullptr;

    static constexpr int kMaxCandidatePoints = 8'000'000;

    void ensure_gpu_memory(int num_cameras);
    void free_gpu_memory();
};

} // namespace heimdall::fusion
