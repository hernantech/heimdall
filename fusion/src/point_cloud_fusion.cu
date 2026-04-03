#include "point_cloud_fusion.h"

#include <algorithm>
#include <cstdio>
#include <stdexcept>

namespace heimdall::fusion {

// ── Helper macros ───────────────────────────────────────────────────────

#define CUDA_CHECK(call)                                                      \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            char msg[256];                                                     \
            snprintf(msg, sizeof(msg), "CUDA error at %s:%d — %s",            \
                     __FILE__, __LINE__, cudaGetErrorString(err));             \
            throw std::runtime_error(msg);                                     \
        }                                                                      \
    } while (0)

// ── Device helpers ──────────────────────────────────────────────────────

// Transform a point by a row-major 4x4 matrix.
__device__ __forceinline__ void transform_point(
    const float* __restrict__ m,
    float ix, float iy, float iz,
    float& ox, float& oy, float& oz
) {
    ox = m[0]*ix + m[1]*iy + m[2]*iz  + m[3];
    oy = m[4]*ix + m[5]*iy + m[6]*iz  + m[7];
    oz = m[8]*ix + m[9]*iy + m[10]*iz + m[11];
}

// ── Kernel 1: Unproject depth pixels to 3D candidates ───────────────────

// For a single camera view, unproject each foreground depth pixel to world space.
// Also compute per-pixel normals from depth gradients.
__global__ void unproject_depth_kernel(
    const float* __restrict__ depth,
    const uint8_t* __restrict__ color,       // RGBA, may be nullptr
    const uint8_t* __restrict__ alpha,       // may be nullptr
    FusionCameraParams camera,
    float depth_min,
    float depth_max,
    int source_camera_index,
    float* __restrict__ out_positions,       // Nx3
    float* __restrict__ out_normals,         // Nx3
    uint8_t* __restrict__ out_colors,        // Nx3
    int* __restrict__ out_camera_indices,    // N
    int* __restrict__ out_count,
    int max_points
) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= camera.width || y >= camera.height) return;

    const int pixel_idx = y * camera.width + x;

    // Alpha mask check
    if (alpha != nullptr && alpha[pixel_idx] < 128) return;

    float d = depth[pixel_idx];
    if (d < depth_min || d > depth_max) return;

    // Unproject to camera space
    float cam_x = (x - camera.cx) / camera.fx * d;
    float cam_y = (y - camera.cy) / camera.fy * d;
    float cam_z = d;

    // Camera space to world space
    float wx, wy, wz;
    transform_point(camera.cam_to_world, cam_x, cam_y, cam_z, wx, wy, wz);

    // Compute normal from depth gradient (screen-space finite differences)
    float nx = 0.0f, ny = 0.0f, nz = 0.0f;
    bool has_normal = false;

    if (x > 0 && x < camera.width - 1 && y > 0 && y < camera.height - 1) {
        float dl = depth[(y) * camera.width + (x - 1)];
        float dr = depth[(y) * camera.width + (x + 1)];
        float du = depth[(y - 1) * camera.width + x];
        float dd = depth[(y + 1) * camera.width + x];

        if (dl > depth_min && dr > depth_min && du > depth_min && dd > depth_min &&
            dl < depth_max && dr < depth_max && du < depth_max && dd < depth_max) {
            // Unproject neighbors to camera space
            float lx = ((x - 1) - camera.cx) / camera.fx * dl;
            float ly = (y - camera.cy) / camera.fy * dl;
            float rx = ((x + 1) - camera.cx) / camera.fx * dr;
            float ry = (y - camera.cy) / camera.fy * dr;
            float ux = (x - camera.cx) / camera.fx * du;
            float uy = ((y - 1) - camera.cy) / camera.fy * du;
            float dx_p = (x - camera.cx) / camera.fx * dd;
            float dy_p = ((y + 1) - camera.cy) / camera.fy * dd;

            // Horizontal tangent (right - left) in camera space
            float tx = rx - lx;
            float ty = ry - ly;
            float tz = dr - dl;

            // Vertical tangent (down - up) in camera space
            float sx = dx_p - ux;
            float sy = dy_p - uy;
            float sz = dd - du;

            // Cross product (tangent x bitangent) = normal in camera space
            float cn_x = ty * sz - tz * sy;
            float cn_y = tz * sx - tx * sz;
            float cn_z = tx * sy - ty * sx;

            float len = sqrtf(cn_x*cn_x + cn_y*cn_y + cn_z*cn_z);
            if (len > 1e-8f) {
                cn_x /= len; cn_y /= len; cn_z /= len;

                // Transform normal to world space (rotation only — use 3x3 part of cam_to_world)
                const float* m = camera.cam_to_world;
                nx = m[0]*cn_x + m[1]*cn_y + m[2]*cn_z;
                ny = m[4]*cn_x + m[5]*cn_y + m[6]*cn_z;
                nz = m[8]*cn_x + m[9]*cn_y + m[10]*cn_z;
                has_normal = true;
            }
        }
    }

    if (!has_normal) {
        // Fallback: normal pointing from surface to camera
        float cam_wx, cam_wy, cam_wz;
        transform_point(camera.cam_to_world, 0.0f, 0.0f, 0.0f, cam_wx, cam_wy, cam_wz);
        nx = cam_wx - wx;
        ny = cam_wy - wy;
        nz = cam_wz - wz;
        float len = sqrtf(nx*nx + ny*ny + nz*nz);
        if (len > 1e-8f) { nx /= len; ny /= len; nz /= len; }
    }

    // Atomically allocate output slot
    int pt_idx = atomicAdd(out_count, 1);
    if (pt_idx >= max_points) return;

    out_positions[pt_idx * 3 + 0] = wx;
    out_positions[pt_idx * 3 + 1] = wy;
    out_positions[pt_idx * 3 + 2] = wz;

    out_normals[pt_idx * 3 + 0] = nx;
    out_normals[pt_idx * 3 + 1] = ny;
    out_normals[pt_idx * 3 + 2] = nz;

    out_camera_indices[pt_idx] = source_camera_index;

    if (color != nullptr) {
        int cidx = pixel_idx * 4;
        out_colors[pt_idx * 3 + 0] = color[cidx + 0];
        out_colors[pt_idx * 3 + 1] = color[cidx + 1];
        out_colors[pt_idx * 3 + 2] = color[cidx + 2];
    } else {
        out_colors[pt_idx * 3 + 0] = 128;
        out_colors[pt_idx * 3 + 1] = 128;
        out_colors[pt_idx * 3 + 2] = 128;
    }
}

// ── Kernel 2: Multi-view consistency check ──────────────────────────────

// For each candidate 3D point, reproject into every other camera and check
// depth agreement. Count how many cameras agree.
__global__ void consistency_check_kernel(
    const float* __restrict__ positions,     // Nx3 candidate positions (world space)
    const int* __restrict__ camera_indices,  // N, source camera for each point
    int num_candidates,
    const FusionCameraParams* __restrict__ cameras,
    const float* const* __restrict__ depth_maps,  // per-camera device pointers
    const uint8_t* const* __restrict__ alpha_maps, // per-camera alpha pointers (may be null)
    int num_cameras,
    float consistency_threshold,
    float normal_consistency_rad,
    const float* __restrict__ normals,       // Nx3
    int* __restrict__ out_consistent_count   // N
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_candidates) return;

    float wx = positions[idx * 3 + 0];
    float wy = positions[idx * 3 + 1];
    float wz = positions[idx * 3 + 2];
    int source_cam = camera_indices[idx];

    float pt_nx = normals[idx * 3 + 0];
    float pt_ny = normals[idx * 3 + 1];
    float pt_nz = normals[idx * 3 + 2];

    int consistent = 1;  // source camera counts as 1

    for (int c = 0; c < num_cameras; c++) {
        if (c == source_cam) continue;

        const FusionCameraParams& cam = cameras[c];

        // Project to this camera
        float cx, cy, cz;
        transform_point(cam.world_to_cam, wx, wy, wz, cx, cy, cz);

        if (cz <= 0.0f) continue;

        float u = cam.fx * (cx / cz) + cam.cx;
        float v = cam.fy * (cy / cz) + cam.cy;

        int px = __float2int_rn(u);
        int py = __float2int_rn(v);

        if (px < 0 || px >= cam.width || py < 0 || py >= cam.height) continue;

        int pixel_idx = py * cam.width + px;

        // Alpha mask check
        if (alpha_maps[c] != nullptr && alpha_maps[c][pixel_idx] < 128) continue;

        // Depth agreement check
        float measured_depth = depth_maps[c][pixel_idx];
        float depth_diff = fabsf(measured_depth - cz);

        if (depth_diff < consistency_threshold) {
            // Optional normal consistency: check if the view direction is within
            // a reasonable angle of the surface normal
            float view_dx = cam.cam_to_world[3] - wx;  // camera position - point
            float view_dy = cam.cam_to_world[7] - wy;
            float view_dz = cam.cam_to_world[11] - wz;
            float view_len = sqrtf(view_dx*view_dx + view_dy*view_dy + view_dz*view_dz);

            if (view_len > 1e-8f) {
                view_dx /= view_len; view_dy /= view_len; view_dz /= view_len;
                float cos_angle = pt_nx * view_dx + pt_ny * view_dy + pt_nz * view_dz;
                if (cos_angle < cosf(normal_consistency_rad)) continue;
            }

            consistent++;
        }
    }

    out_consistent_count[idx] = consistent;
}

// ── Kernel 3: Filter by consistency and compact output ──────────────────

__global__ void compact_consistent_kernel(
    const float* __restrict__ positions,
    const float* __restrict__ normals,
    const uint8_t* __restrict__ colors,
    const int* __restrict__ consistent_counts,
    int num_candidates,
    int min_consistent_views,
    float* __restrict__ out_positions,
    float* __restrict__ out_normals,
    uint8_t* __restrict__ out_colors,
    int* __restrict__ out_count,
    int max_output
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_candidates) return;

    if (consistent_counts[idx] >= min_consistent_views) {
        int out_idx = atomicAdd(out_count, 1);
        if (out_idx >= max_output) return;

        out_positions[out_idx * 3 + 0] = positions[idx * 3 + 0];
        out_positions[out_idx * 3 + 1] = positions[idx * 3 + 1];
        out_positions[out_idx * 3 + 2] = positions[idx * 3 + 2];

        out_normals[out_idx * 3 + 0] = normals[idx * 3 + 0];
        out_normals[out_idx * 3 + 1] = normals[idx * 3 + 1];
        out_normals[out_idx * 3 + 2] = normals[idx * 3 + 2];

        out_colors[out_idx * 3 + 0] = colors[idx * 3 + 0];
        out_colors[out_idx * 3 + 1] = colors[idx * 3 + 1];
        out_colors[out_idx * 3 + 2] = colors[idx * 3 + 2];
    }
}

// ── PointCloudFusion implementation ─────────────────────────────────────

PointCloudFusion::PointCloudFusion(const PointCloudFusionConfig& config)
    : config_(config)
{
    // Candidate buffers
    CUDA_CHECK(cudaMalloc(&d_candidate_positions_, kMaxCandidatePoints * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_candidate_normals_, kMaxCandidatePoints * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_candidate_colors_, kMaxCandidatePoints * 3 * sizeof(uint8_t)));
    CUDA_CHECK(cudaMalloc(&d_candidate_views_, kMaxCandidatePoints * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_candidate_count_, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_consistency_counts_, kMaxCandidatePoints * sizeof(int)));

    // Output buffers
    CUDA_CHECK(cudaMalloc(&d_out_positions_, config_.max_points * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out_normals_, config_.max_points * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out_colors_, config_.max_points * 3 * sizeof(uint8_t)));
    CUDA_CHECK(cudaMalloc(&d_out_count_, sizeof(int)));
}

PointCloudFusion::~PointCloudFusion() {
    free_gpu_memory();
}

void PointCloudFusion::ensure_gpu_memory(int num_cameras) {
    if (num_cameras > max_cameras_allocated_) {
        if (d_cameras_) cudaFree(d_cameras_);
        CUDA_CHECK(cudaMalloc(&d_cameras_, num_cameras * sizeof(FusionCameraParams)));
        max_cameras_allocated_ = num_cameras;
    }
}

void PointCloudFusion::free_gpu_memory() {
    auto safe_free = [](auto*& ptr) {
        if (ptr) { cudaFree(ptr); ptr = nullptr; }
    };
    safe_free(d_candidate_positions_);
    safe_free(d_candidate_normals_);
    safe_free(d_candidate_colors_);
    safe_free(d_candidate_views_);
    safe_free(d_candidate_count_);
    safe_free(d_consistency_counts_);
    safe_free(d_out_positions_);
    safe_free(d_out_normals_);
    safe_free(d_out_colors_);
    safe_free(d_out_count_);
    safe_free(d_cameras_);
}

ExtractedPoints PointCloudFusion::fuse(
    const std::vector<DepthViewGpu>& views,
    cudaStream_t stream
) {
    if (views.empty()) return {};

    int num_cameras = static_cast<int>(views.size());
    ensure_gpu_memory(num_cameras);

    // ── Phase 1: Unproject all cameras' depth to 3D candidates ──────────

    CUDA_CHECK(cudaMemsetAsync(d_candidate_count_, 0, sizeof(int), stream));

    for (int c = 0; c < num_cameras; c++) {
        const auto& view = views[c];
        dim3 block(16, 16);
        dim3 grid(
            (view.width + block.x - 1) / block.x,
            (view.height + block.y - 1) / block.y
        );

        unproject_depth_kernel<<<grid, block, 0, stream>>>(
            view.depth_gpu, view.color_gpu, view.alpha_gpu,
            view.camera,
            config_.depth_min, config_.depth_max,
            c,
            d_candidate_positions_, d_candidate_normals_,
            d_candidate_colors_, d_candidate_views_,
            d_candidate_count_, kMaxCandidatePoints
        );
    }

    // Read candidate count
    int num_candidates = 0;
    CUDA_CHECK(cudaMemcpyAsync(&num_candidates, d_candidate_count_, sizeof(int),
                                cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    if (num_candidates == 0) return {};
    num_candidates = std::min(num_candidates, kMaxCandidatePoints);

    // ── Phase 2: Multi-view consistency check ───────────────────────────

    // Upload camera params to GPU
    std::vector<FusionCameraParams> host_cameras(num_cameras);
    for (int c = 0; c < num_cameras; c++) {
        host_cameras[c] = views[c].camera;
    }
    CUDA_CHECK(cudaMemcpyAsync(d_cameras_, host_cameras.data(),
                                num_cameras * sizeof(FusionCameraParams),
                                cudaMemcpyHostToDevice, stream));

    // Build device pointer arrays for depth and alpha maps
    std::vector<const float*> h_depth_ptrs(num_cameras);
    std::vector<const uint8_t*> h_alpha_ptrs(num_cameras);
    for (int c = 0; c < num_cameras; c++) {
        h_depth_ptrs[c] = views[c].depth_gpu;
        h_alpha_ptrs[c] = views[c].alpha_gpu;
    }

    const float** d_depth_ptrs = nullptr;
    const uint8_t** d_alpha_ptrs = nullptr;
    CUDA_CHECK(cudaMalloc(&d_depth_ptrs, num_cameras * sizeof(const float*)));
    CUDA_CHECK(cudaMalloc(&d_alpha_ptrs, num_cameras * sizeof(const uint8_t*)));
    CUDA_CHECK(cudaMemcpyAsync(d_depth_ptrs, h_depth_ptrs.data(),
                                num_cameras * sizeof(const float*),
                                cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_alpha_ptrs, h_alpha_ptrs.data(),
                                num_cameras * sizeof(const uint8_t*),
                                cudaMemcpyHostToDevice, stream));

    float normal_consistency_rad = config_.normal_consistency_deg * 3.14159265f / 180.0f;

    {
        const int threads = 256;
        const int blocks = (num_candidates + threads - 1) / threads;

        consistency_check_kernel<<<blocks, threads, 0, stream>>>(
            d_candidate_positions_, d_candidate_views_,
            num_candidates,
            d_cameras_, d_depth_ptrs, d_alpha_ptrs, num_cameras,
            config_.consistency_threshold,
            normal_consistency_rad,
            d_candidate_normals_,
            d_consistency_counts_
        );
    }

    // ── Phase 3: Compact consistent points ──────────────────────────────

    CUDA_CHECK(cudaMemsetAsync(d_out_count_, 0, sizeof(int), stream));

    {
        const int threads = 256;
        const int blocks = (num_candidates + threads - 1) / threads;

        compact_consistent_kernel<<<blocks, threads, 0, stream>>>(
            d_candidate_positions_, d_candidate_normals_, d_candidate_colors_,
            d_consistency_counts_, num_candidates,
            config_.min_consistent_views,
            d_out_positions_, d_out_normals_, d_out_colors_,
            d_out_count_, config_.max_points
        );
    }

    // Read output count
    int num_output = 0;
    CUDA_CHECK(cudaMemcpyAsync(&num_output, d_out_count_, sizeof(int),
                                cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    num_output = std::min(num_output, config_.max_points);

    // Copy to host
    ExtractedPoints result;
    result.num_points = num_output;

    if (num_output > 0) {
        result.positions.resize(num_output * 3);
        result.normals.resize(num_output * 3);
        result.colors.resize(num_output * 3);

        CUDA_CHECK(cudaMemcpy(result.positions.data(), d_out_positions_,
                              num_output * 3 * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(result.normals.data(), d_out_normals_,
                              num_output * 3 * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(result.colors.data(), d_out_colors_,
                              num_output * 3 * sizeof(uint8_t), cudaMemcpyDeviceToHost));
    }

    // Free temporary pointer arrays
    cudaFree(d_depth_ptrs);
    cudaFree(d_alpha_ptrs);

    return result;
}

} // namespace heimdall::fusion
