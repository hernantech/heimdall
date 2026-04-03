#pragma once

#include "tsdf_volume.h"
#include "point_cloud_fusion.h"
#include <cuda_runtime.h>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace heimdall::fusion {

// ── Output data structures ──────────────────────────────────────────────

// Fused point cloud — host-side, ready for downstream consumption.
struct FusedPointCloud {
    std::vector<float> positions;     // x, y, z interleaved (N * 3)
    std::vector<float> normals;       // nx, ny, nz interleaved (N * 3)
    std::vector<uint8_t> colors;      // r, g, b interleaved (N * 3)
    int num_points = 0;
};

// Fused triangle mesh — host-side, ready for the mesh module.
struct FusedMesh {
    std::vector<float> vertices;      // x, y, z interleaved (V * 3)
    std::vector<float> normals;       // nx, ny, nz interleaved (V * 3)
    std::vector<uint8_t> colors;      // r, g, b interleaved (V * 3)
    std::vector<uint32_t> indices;    // triangle list (T * 3 indices)
    int num_vertices = 0;
    int num_triangles = 0;
};

// ── Configuration ───────────────────────────────────────────────────────

enum class FusionMethod {
    TSDF,           // Volumetric TSDF → marching cubes mesh
    POINT_CLOUD     // Direct multi-view point cloud fusion
};

struct FusionConfig {
    FusionMethod method = FusionMethod::TSDF;

    // Volume bounds in world space (meters).
    // Default: 2.5m cube centered on the origin — typical for single-person capture.
    float volume_min_x = -1.25f;
    float volume_min_y = -1.25f;
    float volume_min_z = -1.25f;
    float volume_max_x =  1.25f;
    float volume_max_y =  1.25f;
    float volume_max_z =  1.25f;

    // Voxel resolution (meters). 4mm is a good balance of quality vs memory.
    // At 4mm over a 2.5m cube: 625^3 = ~244M voxels = ~2.9 GB GPU memory.
    float voxel_size = 0.004f;

    // TSDF truncation distance (meters).
    // Typical: 3-5x voxel_size. Larger = smoother but loses thin features.
    float tsdf_truncation = 0.02f;

    // Valid depth range (meters).
    float min_depth = 0.2f;
    float max_depth = 5.0f;

    // Multi-view consistency: minimum number of cameras that must agree
    // on a surface point for it to be kept. Applies to both methods.
    int min_views = 2;

    // Weight depth contributions by cos(angle) between surface normal
    // and view direction. Grazing-angle observations get lower weight.
    bool weight_by_angle = true;

    // Point cloud fusion specific
    float consistency_threshold = 0.01f;  // depth agreement threshold (meters)
};

// ── Camera calibration for fusion ───────────────────────────────────────

// Matches the calibration schema (rig.schema.json) with OpenCV convention.
struct CameraCalibration {
    int camera_index;
    int width;
    int height;
    float fx, fy, cx, cy;

    // 3x3 rotation matrix (world → camera), row-major
    float rotation[9];

    // Translation vector (world → camera), meters
    float translation[3];
};

// ── Depth input ─────────────────────────────────────────────────────────

// One camera's depth + color + alpha for a single timestep.
struct DepthInput {
    int camera_index;
    CameraCalibration calibration;

    // All pointers are CUDA device memory.
    const float* depth_gpu;         // float32 meters, width*height
    const uint8_t* color_gpu;       // RGBA uint8, width*height*4 (may be nullptr)
    const uint8_t* alpha_gpu;       // uint8 foreground mask, width*height (may be nullptr)
};

// ── FusionEngine ────────────────────────────────────────────────────────

// Top-level orchestrator for depth fusion.
//
// Typical per-frame workflow:
//   1. engine.reset()                   — clear from previous frame
//   2. for each camera:
//        engine.integrate_depth(input)  — accumulate into volume/point set
//   3. auto mesh = engine.extract_mesh()  — get fused mesh (TSDF)
//      auto pts  = engine.extract_point_cloud()  — or get point cloud
//
// The engine internally delegates to TsdfVolume or PointCloudFusion
// based on the configured method.
class FusionEngine {
public:
    explicit FusionEngine(const FusionConfig& config);
    ~FusionEngine();

    FusionEngine(const FusionEngine&) = delete;
    FusionEngine& operator=(const FusionEngine&) = delete;

    // Reconfigure the engine. Reallocates GPU memory if volume size changes.
    void configure(const FusionConfig& config);

    // Integrate one camera's depth map into the fusion volume.
    // May be called multiple times per frame (once per camera).
    void integrate_depth(const DepthInput& input, cudaStream_t stream = nullptr);

    // Extract the fused point cloud (works for both TSDF and POINT_CLOUD methods).
    // For TSDF, extracts zero-crossing points from the volume.
    // For POINT_CLOUD, runs multi-view consistency and returns filtered points.
    FusedPointCloud extract_point_cloud(cudaStream_t stream = nullptr);

    // Extract a triangle mesh via marching cubes (TSDF method only).
    // Throws std::runtime_error if method is POINT_CLOUD.
    FusedMesh extract_mesh(cudaStream_t stream = nullptr);

    // Clear the volume / accumulated data for the next frame.
    void reset(cudaStream_t stream = nullptr);

    // Current configuration
    const FusionConfig& config() const { return config_; }

    // Number of depth maps integrated in the current frame
    int integration_count() const { return integration_count_; }

private:
    FusionConfig config_;

    // Backend implementations
    std::unique_ptr<TsdfVolume> tsdf_volume_;
    std::unique_ptr<PointCloudFusion> pc_fusion_;

    // Accumulated depth views for point cloud fusion (fused on extract)
    std::vector<DepthViewGpu> accumulated_views_;

    int integration_count_ = 0;

    // Convert our public CameraCalibration to internal FusionCameraParams.
    static FusionCameraParams make_camera_params(const CameraCalibration& calib);

    // Build VolumeParams from FusionConfig.
    static VolumeParams make_volume_params(const FusionConfig& config);
};

} // namespace heimdall::fusion
