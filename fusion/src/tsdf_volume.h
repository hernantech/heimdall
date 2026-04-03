#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include <vector>

namespace heimdall::fusion {

// A single TSDF voxel: signed distance, accumulated weight, and color.
// Packed to 8 bytes for GPU cache efficiency.
struct __align__(8) TsdfVoxel {
    float tsdf;         // truncated signed distance (meters)
    float weight;       // running weighted average denominator
    uint8_t r, g, b;    // accumulated color (weighted average)
    uint8_t pad;        // alignment padding
};

static_assert(sizeof(TsdfVoxel) == 12 || sizeof(TsdfVoxel) <= 16,
              "TsdfVoxel should stay small for GPU cache efficiency");

// Camera intrinsics + extrinsics for TSDF integration.
// All transforms are row-major 4x4.
struct FusionCameraParams {
    int width;
    int height;
    float fx, fy, cx, cy;

    // world → camera (for depth lookup during integration)
    float world_to_cam[16];

    // camera → world (for unprojection)
    float cam_to_world[16];
};

// Volume configuration passed to CUDA kernels.
struct VolumeParams {
    // Volume bounds in world space (meters)
    float origin_x, origin_y, origin_z;   // min corner
    float voxel_size;                      // meters per voxel

    int dim_x, dim_y, dim_z;              // voxel grid dimensions
    int64_t total_voxels;                  // dim_x * dim_y * dim_z

    float truncation_distance;            // TSDF truncation distance (meters)
    float depth_min;                      // minimum valid depth (meters)
    float depth_max;                      // maximum valid depth (meters)

    bool weight_by_angle;                 // weight by surface-normal vs view-dir angle
};

// Triangle vertex for marching cubes output.
struct MeshVertex {
    float x, y, z;       // position in world space
    float nx, ny, nz;    // normal
    uint8_t r, g, b;     // color
    uint8_t pad;
};

// Extracted triangle mesh from marching cubes.
struct ExtractedMesh {
    std::vector<MeshVertex> vertices;
    std::vector<uint32_t> indices;    // triangle list (3 indices per triangle)
};

// Extracted point cloud from TSDF zero-crossings.
struct ExtractedPoints {
    std::vector<float> positions;     // x, y, z interleaved (N * 3)
    std::vector<float> normals;       // nx, ny, nz interleaved (N * 3)
    std::vector<uint8_t> colors;      // r, g, b interleaved (N * 3)
    int num_points = 0;
};

// GPU-resident TSDF volume with CUDA integration and marching cubes extraction.
//
// Workflow:
//   1. Construct with VolumeParams
//   2. Call integrate() for each depth map (from each camera view)
//   3. Call extract_mesh() or extract_points() to get fused geometry
//   4. Call reset() to clear for the next frame/timestep
//
// All GPU memory is allocated in the constructor and reused across frames.
class TsdfVolume {
public:
    explicit TsdfVolume(const VolumeParams& params);
    ~TsdfVolume();

    // Not copyable, but movable.
    TsdfVolume(const TsdfVolume&) = delete;
    TsdfVolume& operator=(const TsdfVolume&) = delete;
    TsdfVolume(TsdfVolume&&) noexcept;
    TsdfVolume& operator=(TsdfVolume&&) noexcept;

    // Integrate a single depth map into the TSDF volume.
    // depth_gpu: device pointer, float32, meters, width*height elements
    // color_gpu: device pointer, RGBA uint8, width*height*4 elements (may be nullptr)
    // alpha_gpu: device pointer, uint8, width*height elements (foreground mask, may be nullptr)
    // camera: intrinsics + extrinsics for this view
    void integrate(
        const float* depth_gpu,
        const uint8_t* color_gpu,
        const uint8_t* alpha_gpu,
        const FusionCameraParams& camera,
        cudaStream_t stream = nullptr
    );

    // Extract triangle mesh via marching cubes.
    // Runs entirely on GPU, then copies results to host.
    ExtractedMesh extract_mesh(cudaStream_t stream = nullptr);

    // Extract point cloud from TSDF zero-crossings.
    // Faster than full marching cubes when you only need points.
    ExtractedPoints extract_points(cudaStream_t stream = nullptr);

    // Clear volume to empty state (tsdf=1, weight=0).
    void reset(cudaStream_t stream = nullptr);

    // Accessors
    const VolumeParams& params() const { return params_; }
    int64_t total_voxels() const { return params_.total_voxels; }
    int integration_count() const { return integration_count_; }

private:
    VolumeParams params_;
    TsdfVoxel* d_voxels_ = nullptr;         // GPU voxel grid

    // Scratch buffers for marching cubes
    MeshVertex* d_mc_vertices_ = nullptr;    // output vertices
    uint32_t* d_mc_indices_ = nullptr;       // output indices
    int* d_mc_count_ = nullptr;              // atomic counter for output triangles

    // Scratch buffers for point extraction
    float* d_pt_positions_ = nullptr;
    float* d_pt_normals_ = nullptr;
    uint8_t* d_pt_colors_ = nullptr;
    int* d_pt_count_ = nullptr;

    int integration_count_ = 0;

    // Maximum output sizes (conservative upper bounds)
    static constexpr int kMaxMarchingCubesTriangles = 8'000'000;
    static constexpr int kMaxExtractedPoints = 4'000'000;

    void allocate_gpu_memory();
    void free_gpu_memory();
};

} // namespace heimdall::fusion
