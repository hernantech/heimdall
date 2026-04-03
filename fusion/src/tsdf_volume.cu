#include "tsdf_volume.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
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

__device__ __forceinline__ int voxel_index(int x, int y, int z,
                                            int dim_x, int dim_y) {
    return z * dim_x * dim_y + y * dim_x + x;
}

// World position of voxel center.
__device__ __forceinline__ void voxel_world_pos(
    int vx, int vy, int vz,
    float origin_x, float origin_y, float origin_z,
    float voxel_size,
    float& wx, float& wy, float& wz
) {
    wx = origin_x + (vx + 0.5f) * voxel_size;
    wy = origin_y + (vy + 0.5f) * voxel_size;
    wz = origin_z + (vz + 0.5f) * voxel_size;
}

// Transform a world-space point by a row-major 4x4 matrix.
__device__ __forceinline__ void transform_point(
    const float* __restrict__ m,
    float wx, float wy, float wz,
    float& cx, float& cy, float& cz
) {
    cx = m[0]*wx + m[1]*wy + m[2]*wz  + m[3];
    cy = m[4]*wx + m[5]*wy + m[6]*wz  + m[7];
    cz = m[8]*wx + m[9]*wy + m[10]*wz + m[11];
}

// ── TSDF Integration Kernel ─────────────────────────────────────────────

// Each thread processes one voxel.
// For each voxel: project to camera image, look up depth, compute TSDF, update.
__global__ void tsdf_integrate_kernel(
    TsdfVoxel* __restrict__ voxels,
    const float* __restrict__ depth,
    const uint8_t* __restrict__ color,      // RGBA, may be nullptr
    const uint8_t* __restrict__ alpha,      // foreground mask, may be nullptr
    FusionCameraParams camera,
    VolumeParams volume
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= volume.total_voxels) return;

    // Decompose linear index into 3D voxel coordinates
    const int vz = idx / (volume.dim_x * volume.dim_y);
    const int vy = (idx - vz * volume.dim_x * volume.dim_y) / volume.dim_x;
    const int vx = idx - vz * volume.dim_x * volume.dim_y - vy * volume.dim_x;

    // World position of voxel center
    float wx, wy, wz;
    voxel_world_pos(vx, vy, vz,
                    volume.origin_x, volume.origin_y, volume.origin_z,
                    volume.voxel_size,
                    wx, wy, wz);

    // Transform to camera space
    float cx, cy, cz;
    transform_point(camera.world_to_cam, wx, wy, wz, cx, cy, cz);

    // Must be in front of the camera
    if (cz <= 0.0f) return;

    // Project to pixel coordinates
    float u = camera.fx * (cx / cz) + camera.cx;
    float v = camera.fy * (cy / cz) + camera.cy;

    int px = __float2int_rn(u);
    int py = __float2int_rn(v);

    if (px < 0 || px >= camera.width || py < 0 || py >= camera.height) return;

    int pixel_idx = py * camera.width + px;

    // Check alpha mask — only integrate foreground
    if (alpha != nullptr && alpha[pixel_idx] < 128) return;

    // Look up measured depth
    float measured_depth = depth[pixel_idx];

    // Validity check
    if (measured_depth < volume.depth_min || measured_depth > volume.depth_max) return;

    // Signed distance: positive = in front of surface, negative = behind
    float sdf = measured_depth - cz;

    // Skip voxels that are too far behind the surface
    if (sdf < -volume.truncation_distance) return;

    // Truncate the SDF value
    float tsdf = fminf(1.0f, sdf / volume.truncation_distance);

    // Compute integration weight
    float w = 1.0f;

    if (volume.weight_by_angle) {
        // Approximate surface normal direction using the view ray.
        // Weight by cos(angle) ≈ z_cam / dist, which downweights grazing angles.
        float dist = sqrtf(cx*cx + cy*cy + cz*cz);
        if (dist > 1e-6f) {
            w = cz / dist;   // cos(angle between view ray and camera z-axis)
            w = fmaxf(w, 0.1f); // clamp so grazing angles still contribute a little
        }
    }

    // Running weighted average update
    TsdfVoxel& voxel = voxels[idx];
    float old_tsdf = voxel.tsdf;
    float old_weight = voxel.weight;
    float new_weight = old_weight + w;

    // Cap weight to prevent stale data from dominating
    const float max_weight = 128.0f;
    new_weight = fminf(new_weight, max_weight);

    voxel.tsdf = (old_tsdf * old_weight + tsdf * w) / new_weight;
    voxel.weight = new_weight;

    // Update color (weighted average)
    if (color != nullptr) {
        int color_base = pixel_idx * 4;  // RGBA layout
        uint8_t cr = color[color_base + 0];
        uint8_t cg = color[color_base + 1];
        uint8_t cb = color[color_base + 2];

        float blend = w / new_weight;
        voxel.r = static_cast<uint8_t>(voxel.r * (1.0f - blend) + cr * blend);
        voxel.g = static_cast<uint8_t>(voxel.g * (1.0f - blend) + cg * blend);
        voxel.b = static_cast<uint8_t>(voxel.b * (1.0f - blend) + cb * blend);
    }
}

// ── Reset Kernel ────────────────────────────────────────────────────────

__global__ void tsdf_reset_kernel(TsdfVoxel* __restrict__ voxels,
                                   int64_t total_voxels) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_voxels) return;

    voxels[idx].tsdf = 1.0f;    // far from any surface
    voxels[idx].weight = 0.0f;
    voxels[idx].r = 0;
    voxels[idx].g = 0;
    voxels[idx].b = 0;
    voxels[idx].pad = 0;
}

// ── Marching Cubes ──────────────────────────────────────────────────────

// Marching cubes edge table and triangle table.
// These are the standard tables — 256 entries for cube configurations.
// Stored in constant memory for fast GPU access.

// Edge table: for each of 256 cube configs, a 12-bit mask of which edges
// are intersected by the isosurface.
__device__ __constant__ int mc_edge_table[256] = {
    0x000, 0x109, 0x203, 0x30a, 0x406, 0x50f, 0x605, 0x70c,
    0x80c, 0x905, 0xa0f, 0xb06, 0xc0a, 0xd03, 0xe09, 0xf00,
    0x190, 0x099, 0x393, 0x29a, 0x596, 0x49f, 0x795, 0x69c,
    0x99c, 0x895, 0xb9f, 0xa96, 0xd9a, 0xc93, 0xf99, 0xe90,
    0x230, 0x339, 0x033, 0x13a, 0x636, 0x73f, 0x435, 0x53c,
    0xa3c, 0xb35, 0x83f, 0x936, 0xe3a, 0xf33, 0xc39, 0xd30,
    0x3a0, 0x2a9, 0x1a3, 0x0aa, 0x7a6, 0x6af, 0x5a5, 0x4ac,
    0xbac, 0xaa5, 0x9af, 0x8a6, 0xfaa, 0xea3, 0xda9, 0xca0,
    0x460, 0x569, 0x663, 0x76a, 0x066, 0x16f, 0x265, 0x36c,
    0xc6c, 0xd65, 0xe6f, 0xf66, 0x86a, 0x963, 0xa69, 0xb60,
    0x5f0, 0x4f9, 0x7f3, 0x6fa, 0x1f6, 0x0ff, 0x3f5, 0x2fc,
    0xdfc, 0xcf5, 0xfff, 0xef6, 0x9fa, 0x8f3, 0xbf9, 0xaf0,
    0x650, 0x759, 0x453, 0x55a, 0x256, 0x35f, 0x055, 0x15c,
    0xe5c, 0xf55, 0xc5f, 0xd56, 0xa5a, 0xb53, 0x859, 0x950,
    0x7c0, 0x6c9, 0x5c3, 0x4ca, 0x3c6, 0x2cf, 0x1c5, 0x0cc,
    0xfcc, 0xec5, 0xdcf, 0xcc6, 0xbca, 0xac3, 0x9c9, 0x8c0,
    0x8c0, 0x9c9, 0xac3, 0xbca, 0xcc6, 0xdcf, 0xec5, 0xfcc,
    0x0cc, 0x1c5, 0x2cf, 0x3c6, 0x4ca, 0x5c3, 0x6c9, 0x7c0,
    0x950, 0x859, 0xb53, 0xa5a, 0xd56, 0xc5f, 0xf55, 0xe5c,
    0x15c, 0x055, 0x35f, 0x256, 0x55a, 0x453, 0x759, 0x650,
    0xaf0, 0xbf9, 0x8f3, 0x9fa, 0xef6, 0xfff, 0xcf5, 0xdfc,
    0x2fc, 0x3f5, 0x0ff, 0x1f6, 0x6fa, 0x7f3, 0x4f9, 0x5f0,
    0xb60, 0xa69, 0x963, 0x86a, 0xf66, 0xe6f, 0xd65, 0xc6c,
    0x36c, 0x265, 0x16f, 0x066, 0x76a, 0x663, 0x569, 0x460,
    0xca0, 0xda9, 0xea3, 0xfaa, 0x8a6, 0x9af, 0xaa5, 0xbac,
    0x4ac, 0x5a5, 0x6af, 0x7a6, 0x0aa, 0x1a3, 0x2a9, 0x3a0,
    0xd30, 0xc39, 0xf33, 0xe3a, 0x936, 0x83f, 0xb35, 0xa3c,
    0x53c, 0x435, 0x73f, 0x636, 0x13a, 0x033, 0x339, 0x230,
    0xe90, 0xf99, 0xc93, 0xd9a, 0xa96, 0xb9f, 0x895, 0x99c,
    0x69c, 0x795, 0x49f, 0x596, 0x29a, 0x393, 0x099, 0x190,
    0xf00, 0xe09, 0xd03, 0xc0a, 0xb06, 0xa0f, 0x905, 0x80c,
    0x70c, 0x605, 0x50f, 0x406, 0x30a, 0x203, 0x109, 0x000
};

// Triangle table: for each of 256 cube configs, up to 5 triangles (15 edge indices).
// -1 terminates the list.
// This is the standard Paul Bourke marching cubes triangle table.
__device__ __constant__ int mc_tri_table[256][16] = {
    {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 0,  8,  3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 0,  1,  9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 1,  8,  3,  9,  8,  1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 1,  2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 0,  8,  3,  1,  2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 9,  2, 10,  0,  2,  9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 2,  8,  3,  2, 10,  8, 10,  9,  8, -1, -1, -1, -1, -1, -1, -1},
    { 3, 11,  2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 0, 11,  2,  8, 11,  0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 1,  9,  0,  2,  3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 1, 11,  2,  1,  9, 11,  9,  8, 11, -1, -1, -1, -1, -1, -1, -1},
    { 3, 10,  1, 11, 10,  3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 0, 10,  1,  0,  8, 10,  8, 11, 10, -1, -1, -1, -1, -1, -1, -1},
    { 3,  9,  0,  3, 11,  9, 11, 10,  9, -1, -1, -1, -1, -1, -1, -1},
    { 9,  8, 10, 10,  8, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 4,  7,  8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 4,  3,  0,  7,  3,  4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 0,  1,  9,  8,  4,  7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 4,  1,  9,  4,  7,  1,  7,  3,  1, -1, -1, -1, -1, -1, -1, -1},
    { 1,  2, 10,  8,  4,  7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 3,  4,  7,  3,  0,  4,  1,  2, 10, -1, -1, -1, -1, -1, -1, -1},
    { 9,  2, 10,  9,  0,  2,  8,  4,  7, -1, -1, -1, -1, -1, -1, -1},
    { 2, 10,  9,  2,  9,  7,  2,  7,  3,  7,  9,  4, -1, -1, -1, -1},
    { 8,  4,  7,  3, 11,  2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {11,  4,  7, 11,  2,  4,  2,  0,  4, -1, -1, -1, -1, -1, -1, -1},
    { 9,  0,  1,  8,  4,  7,  2,  3, 11, -1, -1, -1, -1, -1, -1, -1},
    { 4,  7, 11,  9,  4, 11,  9, 11,  2,  9,  2,  1, -1, -1, -1, -1},
    { 3, 10,  1,  3, 11, 10,  7,  8,  4, -1, -1, -1, -1, -1, -1, -1},
    { 1, 11, 10,  1,  4, 11,  1,  0,  4,  7, 11,  4, -1, -1, -1, -1},
    { 4,  7,  8,  9,  0, 11,  9, 11, 10, 11,  0,  3, -1, -1, -1, -1},
    { 4,  7, 11,  4, 11,  9,  9, 11, 10, -1, -1, -1, -1, -1, -1, -1},
    { 9,  5,  4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 9,  5,  4,  0,  8,  3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 0,  5,  4,  1,  5,  0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 8,  5,  4,  8,  3,  5,  3,  1,  5, -1, -1, -1, -1, -1, -1, -1},
    { 1,  2, 10,  9,  5,  4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 3,  0,  8,  1,  2, 10,  4,  9,  5, -1, -1, -1, -1, -1, -1, -1},
    { 5,  2, 10,  5,  4,  2,  4,  0,  2, -1, -1, -1, -1, -1, -1, -1},
    { 2, 10,  5,  3,  2,  5,  3,  5,  4,  3,  4,  8, -1, -1, -1, -1},
    { 9,  5,  4,  2,  3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 0, 11,  2,  0,  8, 11,  4,  9,  5, -1, -1, -1, -1, -1, -1, -1},
    { 0,  5,  4,  0,  1,  5,  2,  3, 11, -1, -1, -1, -1, -1, -1, -1},
    { 2,  1,  5,  2,  5,  8,  2,  8, 11,  4,  8,  5, -1, -1, -1, -1},
    {10,  3, 11, 10,  1,  3,  9,  5,  4, -1, -1, -1, -1, -1, -1, -1},
    { 4,  9,  5,  0,  8,  1,  8, 10,  1,  8, 11, 10, -1, -1, -1, -1},
    { 5,  4,  0,  5,  0, 11,  5, 11, 10, 11,  0,  3, -1, -1, -1, -1},
    { 5,  4,  8,  5,  8, 10, 10,  8, 11, -1, -1, -1, -1, -1, -1, -1},
    { 9,  7,  8,  5,  7,  9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 9,  3,  0,  9,  5,  3,  5,  7,  3, -1, -1, -1, -1, -1, -1, -1},
    { 0,  7,  8,  0,  1,  7,  1,  5,  7, -1, -1, -1, -1, -1, -1, -1},
    { 1,  5,  3,  3,  5,  7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 9,  7,  8,  9,  5,  7, 10,  1,  2, -1, -1, -1, -1, -1, -1, -1},
    {10,  1,  2,  9,  5,  0,  5,  3,  0,  5,  7,  3, -1, -1, -1, -1},
    { 8,  0,  2,  8,  2,  5,  8,  5,  7, 10,  5,  2, -1, -1, -1, -1},
    { 2, 10,  5,  2,  5,  3,  3,  5,  7, -1, -1, -1, -1, -1, -1, -1},
    { 7,  9,  5,  7,  8,  9,  3, 11,  2, -1, -1, -1, -1, -1, -1, -1},
    { 9,  5,  7,  9,  7,  2,  9,  2,  0,  2,  7, 11, -1, -1, -1, -1},
    { 2,  3, 11,  0,  1,  8,  1,  7,  8,  1,  5,  7, -1, -1, -1, -1},
    {11,  2,  1, 11,  1,  7,  7,  1,  5, -1, -1, -1, -1, -1, -1, -1},
    { 9,  5,  8,  8,  5,  7, 10,  1,  3, 10,  3, 11, -1, -1, -1, -1},
    { 5,  7,  0,  5,  0,  9,  7, 11,  0,  1,  0, 10, 11, 10,  0, -1},
    {11, 10,  0, 11,  0,  3, 10,  5,  0,  8,  0,  7,  5,  7,  0, -1},
    {11, 10,  5,  7, 11,  5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {10,  6,  5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 0,  8,  3,  5, 10,  6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 9,  0,  1,  5, 10,  6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 1,  8,  3,  1,  9,  8,  5, 10,  6, -1, -1, -1, -1, -1, -1, -1},
    { 1,  6,  5,  2,  6,  1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 1,  6,  5,  1,  2,  6,  3,  0,  8, -1, -1, -1, -1, -1, -1, -1},
    { 9,  6,  5,  9,  0,  6,  0,  2,  6, -1, -1, -1, -1, -1, -1, -1},
    { 5,  9,  8,  5,  8,  2,  5,  2,  6,  3,  2,  8, -1, -1, -1, -1},
    { 2,  3, 11, 10,  6,  5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {11,  0,  8, 11,  2,  0, 10,  6,  5, -1, -1, -1, -1, -1, -1, -1},
    { 0,  1,  9,  2,  3, 11,  5, 10,  6, -1, -1, -1, -1, -1, -1, -1},
    { 5, 10,  6,  1,  9,  2,  9, 11,  2,  9,  8, 11, -1, -1, -1, -1},
    { 6,  3, 11,  6,  5,  3,  5,  1,  3, -1, -1, -1, -1, -1, -1, -1},
    { 0,  8, 11,  0, 11,  5,  0,  5,  1,  5, 11,  6, -1, -1, -1, -1},
    { 3, 11,  6,  0,  3,  6,  0,  6,  5,  0,  5,  9, -1, -1, -1, -1},
    { 6,  5,  9,  6,  9, 11, 11,  9,  8, -1, -1, -1, -1, -1, -1, -1},
    { 5, 10,  6,  4,  7,  8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 4,  3,  0,  4,  7,  3,  6,  5, 10, -1, -1, -1, -1, -1, -1, -1},
    { 1,  9,  0,  5, 10,  6,  8,  4,  7, -1, -1, -1, -1, -1, -1, -1},
    {10,  6,  5,  1,  9,  7,  1,  7,  3,  7,  9,  4, -1, -1, -1, -1},
    { 6,  1,  2,  6,  5,  1,  4,  7,  8, -1, -1, -1, -1, -1, -1, -1},
    { 1,  2,  5,  5,  2,  6,  3,  0,  4,  3,  4,  7, -1, -1, -1, -1},
    { 8,  4,  7,  9,  0,  5,  0,  6,  5,  0,  2,  6, -1, -1, -1, -1},
    { 7,  3,  9,  7,  9,  4,  3,  2,  9,  5,  9,  6,  2,  6,  9, -1},
    { 3, 11,  2,  7,  8,  4, 10,  6,  5, -1, -1, -1, -1, -1, -1, -1},
    { 5, 10,  6,  4,  7,  2,  4,  2,  0,  2,  7, 11, -1, -1, -1, -1},
    { 0,  1,  9,  4,  7,  8,  2,  3, 11,  5, 10,  6, -1, -1, -1, -1},
    { 9,  2,  1,  9, 11,  2,  9,  4, 11,  7, 11,  4,  5, 10,  6, -1},
    { 8,  4,  7,  3, 11,  5,  3,  5,  1,  5, 11,  6, -1, -1, -1, -1},
    { 5,  1, 11,  5, 11,  6,  1,  0, 11,  7, 11,  4,  0,  4, 11, -1},
    { 0,  5,  9,  0,  6,  5,  0,  3,  6, 11,  6,  3,  8,  4,  7, -1},
    { 6,  5,  9,  6,  9, 11,  4,  7,  9,  7, 11,  9, -1, -1, -1, -1},
    {10,  4,  9,  6,  4, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 4, 10,  6,  4,  9, 10,  0,  8,  3, -1, -1, -1, -1, -1, -1, -1},
    {10,  0,  1, 10,  6,  0,  6,  4,  0, -1, -1, -1, -1, -1, -1, -1},
    { 8,  3,  1,  8,  1,  6,  8,  6,  4,  6,  1, 10, -1, -1, -1, -1},
    { 1,  4,  9,  1,  2,  4,  2,  6,  4, -1, -1, -1, -1, -1, -1, -1},
    { 3,  0,  8,  1,  2,  9,  2,  4,  9,  2,  6,  4, -1, -1, -1, -1},
    { 0,  2,  4,  4,  2,  6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 8,  3,  2,  8,  2,  4,  4,  2,  6, -1, -1, -1, -1, -1, -1, -1},
    {10,  4,  9, 10,  6,  4, 11,  2,  3, -1, -1, -1, -1, -1, -1, -1},
    { 0,  8,  2,  2,  8, 11,  4,  9, 10,  4, 10,  6, -1, -1, -1, -1},
    { 3, 11,  2,  0,  1,  6,  0,  6,  4,  6,  1, 10, -1, -1, -1, -1},
    { 6,  4,  1,  6,  1, 10,  4,  8,  1,  2,  1, 11,  8, 11,  1, -1},
    { 9,  6,  4,  9,  3,  6,  9,  1,  3, 11,  6,  3, -1, -1, -1, -1},
    { 8, 11,  1,  8,  1,  0, 11,  6,  1,  9,  1,  4,  6,  4,  1, -1},
    { 3, 11,  6,  3,  6,  0,  0,  6,  4, -1, -1, -1, -1, -1, -1, -1},
    { 6,  4,  8, 11,  6,  8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 7, 10,  6,  7,  8, 10,  8,  9, 10, -1, -1, -1, -1, -1, -1, -1},
    { 0,  7,  3,  0, 10,  7,  0,  9, 10,  6,  7, 10, -1, -1, -1, -1},
    {10,  6,  7,  1, 10,  7,  1,  7,  8,  1,  8,  0, -1, -1, -1, -1},
    {10,  6,  7, 10,  7,  1,  1,  7,  3, -1, -1, -1, -1, -1, -1, -1},
    { 1,  2,  6,  1,  6,  8,  1,  8,  9,  8,  6,  7, -1, -1, -1, -1},
    { 2,  6,  9,  2,  9,  1,  6,  7,  9,  0,  9,  3,  7,  3,  9, -1},
    { 7,  8,  0,  7,  0,  6,  6,  0,  2, -1, -1, -1, -1, -1, -1, -1},
    { 7,  3,  2,  6,  7,  2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 2,  3, 11, 10,  6,  8, 10,  8,  9,  8,  6,  7, -1, -1, -1, -1},
    { 2,  0,  7,  2,  7, 11,  0,  9,  7,  6,  7, 10,  9, 10,  7, -1},
    { 1,  8,  0,  1,  7,  8,  1, 10,  7,  6,  7, 10,  2,  3, 11, -1},
    {11,  2,  1, 11,  1,  7, 10,  6,  1,  6,  7,  1, -1, -1, -1, -1},
    { 8,  9,  6,  8,  6,  7,  9,  1,  6, 11,  6,  3,  1,  3,  6, -1},
    { 0,  9,  1, 11,  6,  7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 7,  8,  0,  7,  0,  6,  3, 11,  0, 11,  6,  0, -1, -1, -1, -1},
    { 7, 11,  6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 7,  6, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 3,  0,  8, 11,  7,  6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 0,  1,  9, 11,  7,  6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 8,  1,  9,  8,  3,  1, 11,  7,  6, -1, -1, -1, -1, -1, -1, -1},
    {10,  1,  2,  6, 11,  7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 1,  2, 10,  3,  0,  8,  6, 11,  7, -1, -1, -1, -1, -1, -1, -1},
    { 2,  9,  0,  2, 10,  9,  6, 11,  7, -1, -1, -1, -1, -1, -1, -1},
    { 6, 11,  7,  2, 10,  3, 10,  8,  3, 10,  9,  8, -1, -1, -1, -1},
    { 7,  2,  3,  6,  2,  7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 7,  0,  8,  7,  6,  0,  6,  2,  0, -1, -1, -1, -1, -1, -1, -1},
    { 2,  7,  6,  2,  3,  7,  0,  1,  9, -1, -1, -1, -1, -1, -1, -1},
    { 1,  6,  2,  1,  8,  6,  1,  9,  8,  8,  7,  6, -1, -1, -1, -1},
    {10,  7,  6, 10,  1,  7,  1,  3,  7, -1, -1, -1, -1, -1, -1, -1},
    {10,  7,  6,  1,  7, 10,  1,  8,  7,  1,  0,  8, -1, -1, -1, -1},
    { 0,  3,  7,  0,  7, 10,  0, 10,  9,  6, 10,  7, -1, -1, -1, -1},
    { 7,  6, 10,  7, 10,  8,  8, 10,  9, -1, -1, -1, -1, -1, -1, -1},
    { 6,  8,  4, 11,  8,  6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 3,  6, 11,  3,  0,  6,  0,  4,  6, -1, -1, -1, -1, -1, -1, -1},
    { 8,  6, 11,  8,  4,  6,  9,  0,  1, -1, -1, -1, -1, -1, -1, -1},
    { 9,  4,  6,  9,  6,  3,  9,  3,  1, 11,  3,  6, -1, -1, -1, -1},
    { 6,  8,  4,  6, 11,  8,  2, 10,  1, -1, -1, -1, -1, -1, -1, -1},
    { 1,  2, 10,  3,  0, 11,  0,  6, 11,  0,  4,  6, -1, -1, -1, -1},
    { 4, 11,  8,  4,  6, 11,  0,  2,  9,  2, 10,  9, -1, -1, -1, -1},
    {10,  9,  3, 10,  3,  2,  9,  4,  3, 11,  3,  6,  4,  6,  3, -1},
    { 8,  2,  3,  8,  4,  2,  4,  6,  2, -1, -1, -1, -1, -1, -1, -1},
    { 0,  4,  2,  4,  6,  2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 1,  9,  0,  2,  3,  4,  2,  4,  6,  4,  3,  8, -1, -1, -1, -1},
    { 1,  9,  4,  1,  4,  2,  2,  4,  6, -1, -1, -1, -1, -1, -1, -1},
    { 8,  1,  3,  8,  6,  1,  8,  4,  6,  6, 10,  1, -1, -1, -1, -1},
    {10,  1,  0, 10,  0,  6,  6,  0,  4, -1, -1, -1, -1, -1, -1, -1},
    { 4,  6,  3,  4,  3,  8,  6, 10,  3,  0,  3,  9, 10,  9,  3, -1},
    {10,  9,  4,  6, 10,  4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 4,  9,  5,  7,  6, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 0,  8,  3,  4,  9,  5, 11,  7,  6, -1, -1, -1, -1, -1, -1, -1},
    { 5,  0,  1,  5,  4,  0,  7,  6, 11, -1, -1, -1, -1, -1, -1, -1},
    {11,  7,  6,  8,  3,  4,  3,  5,  4,  3,  1,  5, -1, -1, -1, -1},
    { 9,  5,  4, 10,  1,  2,  7,  6, 11, -1, -1, -1, -1, -1, -1, -1},
    { 6, 11,  7,  1,  2, 10,  0,  8,  3,  4,  9,  5, -1, -1, -1, -1},
    { 7,  6, 11,  5,  4, 10,  4,  2, 10,  4,  0,  2, -1, -1, -1, -1},
    { 3,  4,  8,  3,  5,  4,  3,  2,  5, 10,  5,  2, 11,  7,  6, -1},
    { 7,  2,  3,  7,  6,  2,  5,  4,  9, -1, -1, -1, -1, -1, -1, -1},
    { 9,  5,  4,  0,  8,  6,  0,  6,  2,  6,  8,  7, -1, -1, -1, -1},
    { 3,  6,  2,  3,  7,  6,  1,  5,  0,  5,  4,  0, -1, -1, -1, -1},
    { 6,  2,  8,  6,  8,  7,  2,  1,  8,  4,  8,  5,  1,  5,  8, -1},
    { 9,  5,  4, 10,  1,  6,  1,  7,  6,  1,  3,  7, -1, -1, -1, -1},
    { 1,  6, 10,  1,  7,  6,  1,  0,  7,  8,  7,  0,  9,  5,  4, -1},
    { 4,  0, 10,  4, 10,  5,  0,  3, 10,  6, 10,  7,  3,  7, 10, -1},
    { 7,  6, 10,  7, 10,  8,  5,  4, 10,  4,  8, 10, -1, -1, -1, -1},
    { 6,  9,  5,  6, 11,  9, 11,  8,  9, -1, -1, -1, -1, -1, -1, -1},
    { 3,  6, 11,  0,  6,  3,  0,  5,  6,  0,  9,  5, -1, -1, -1, -1},
    { 0, 11,  8,  0,  5, 11,  0,  1,  5,  5,  6, 11, -1, -1, -1, -1},
    { 6, 11,  3,  6,  3,  5,  5,  3,  1, -1, -1, -1, -1, -1, -1, -1},
    { 1,  2, 10,  9,  5, 11,  9, 11,  8, 11,  5,  6, -1, -1, -1, -1},
    { 0, 11,  3,  0,  6, 11,  0,  9,  6,  5,  6,  9,  1,  2, 10, -1},
    {11,  8,  5, 11,  5,  6,  8,  0,  5, 10,  5,  2,  0,  2,  5, -1},
    { 6, 11,  3,  6,  3,  5,  2, 10,  3, 10,  5,  3, -1, -1, -1, -1},
    { 5,  8,  9,  5,  2,  8,  5,  6,  2,  3,  8,  2, -1, -1, -1, -1},
    { 9,  5,  6,  9,  6,  0,  0,  6,  2, -1, -1, -1, -1, -1, -1, -1},
    { 1,  5,  8,  1,  8,  0,  5,  6,  8,  3,  8,  2,  6,  2,  8, -1},
    { 1,  5,  6,  2,  1,  6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 1,  3,  6,  1,  6, 10,  3,  8,  6,  5,  6,  9,  8,  9,  6, -1},
    {10,  1,  0, 10,  0,  6,  9,  5,  0,  5,  6,  0, -1, -1, -1, -1},
    { 0,  3,  8,  5,  6, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {10,  5,  6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {11,  5, 10,  7,  5, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {11,  5, 10, 11,  7,  5,  8,  3,  0, -1, -1, -1, -1, -1, -1, -1},
    { 5, 11,  7,  5, 10, 11,  1,  9,  0, -1, -1, -1, -1, -1, -1, -1},
    {10,  7,  5, 10, 11,  7,  9,  8,  1,  8,  3,  1, -1, -1, -1, -1},
    {11,  1,  2, 11,  7,  1,  7,  5,  1, -1, -1, -1, -1, -1, -1, -1},
    { 0,  8,  3,  1,  2,  7,  1,  7,  5,  7,  2, 11, -1, -1, -1, -1},
    { 9,  7,  5,  9,  2,  7,  9,  0,  2,  2, 11,  7, -1, -1, -1, -1},
    { 7,  5,  2,  7,  2, 11,  5,  9,  2,  3,  2,  8,  9,  8,  2, -1},
    { 2,  5, 10,  2,  3,  5,  3,  7,  5, -1, -1, -1, -1, -1, -1, -1},
    { 8,  2,  0,  8,  5,  2,  8,  7,  5, 10,  2,  5, -1, -1, -1, -1},
    { 9,  0,  1,  5, 10,  3,  5,  3,  7,  3, 10,  2, -1, -1, -1, -1},
    { 9,  8,  2,  9,  2,  1,  8,  7,  2, 10,  2,  5,  7,  5,  2, -1},
    { 1,  3,  5,  3,  7,  5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 0,  8,  7,  0,  7,  1,  1,  7,  5, -1, -1, -1, -1, -1, -1, -1},
    { 9,  0,  3,  9,  3,  5,  5,  3,  7, -1, -1, -1, -1, -1, -1, -1},
    { 9,  8,  7,  5,  9,  7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 5,  8,  4,  5, 10,  8, 10, 11,  8, -1, -1, -1, -1, -1, -1, -1},
    { 5,  0,  4,  5, 11,  0,  5, 10, 11, 11,  3,  0, -1, -1, -1, -1},
    { 0,  1,  9,  8,  4, 10,  8, 10, 11, 10,  4,  5, -1, -1, -1, -1},
    {10, 11,  4, 10,  4,  5, 11,  3,  4,  9,  4,  1,  3,  1,  4, -1},
    { 2,  5,  1,  2,  8,  5,  2, 11,  8,  4,  5,  8, -1, -1, -1, -1},
    { 0,  4, 11,  0, 11,  3,  4,  5, 11,  2, 11,  1,  5,  1, 11, -1},
    { 0,  2,  5,  0,  5,  9,  2, 11,  5,  4,  5,  8, 11,  8,  5, -1},
    { 9,  4,  5,  2, 11,  3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 2,  5, 10,  3,  5,  2,  3,  4,  5,  3,  8,  4, -1, -1, -1, -1},
    { 5, 10,  2,  5,  2,  4,  4,  2,  0, -1, -1, -1, -1, -1, -1, -1},
    { 3, 10,  2,  3,  5, 10,  3,  8,  5,  4,  5,  8,  0,  1,  9, -1},
    { 5, 10,  2,  5,  2,  4,  1,  9,  2,  9,  4,  2, -1, -1, -1, -1},
    { 8,  4,  5,  8,  5,  3,  3,  5,  1, -1, -1, -1, -1, -1, -1, -1},
    { 0,  4,  5,  1,  0,  5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 8,  4,  5,  8,  5,  3,  9,  0,  5,  0,  3,  5, -1, -1, -1, -1},
    { 9,  4,  5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 4, 11,  7,  4,  9, 11,  9, 10, 11, -1, -1, -1, -1, -1, -1, -1},
    { 0,  8,  3,  4,  9,  7,  9, 11,  7,  9, 10, 11, -1, -1, -1, -1},
    { 1, 10, 11,  1, 11,  4,  1,  4,  0,  7,  4, 11, -1, -1, -1, -1},
    { 3,  1,  4,  3,  4,  8,  1, 10,  4,  7,  4, 11, 10, 11,  4, -1},
    { 4, 11,  7,  9, 11,  4,  9,  2, 11,  9,  1,  2, -1, -1, -1, -1},
    { 9,  7,  4,  9, 11,  7,  9,  1, 11,  2, 11,  1,  0,  8,  3, -1},
    {11,  7,  4, 11,  4,  2,  2,  4,  0, -1, -1, -1, -1, -1, -1, -1},
    {11,  7,  4, 11,  4,  2,  8,  3,  4,  3,  2,  4, -1, -1, -1, -1},
    { 2,  9, 10,  2,  7,  9,  2,  3,  7,  7,  4,  9, -1, -1, -1, -1},
    { 9, 10,  7,  9,  7,  4, 10,  2,  7,  8,  7,  0,  2,  0,  7, -1},
    { 3,  7, 10,  3, 10,  2,  7,  4, 10,  1, 10,  0,  4,  0, 10, -1},
    { 1, 10,  2,  8,  7,  4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 4,  9,  1,  4,  1,  7,  7,  1,  3, -1, -1, -1, -1, -1, -1, -1},
    { 4,  9,  1,  4,  1,  7,  0,  8,  1,  8,  7,  1, -1, -1, -1, -1},
    { 4,  0,  3,  7,  4,  3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 4,  8,  7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 9, 10,  8, 10, 11,  8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 3,  0,  9,  3,  9, 11, 11,  9, 10, -1, -1, -1, -1, -1, -1, -1},
    { 0,  1, 10,  0, 10,  8,  8, 10, 11, -1, -1, -1, -1, -1, -1, -1},
    { 3,  1, 10, 11,  3, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 1,  2, 11,  1, 11,  9,  9, 11,  8, -1, -1, -1, -1, -1, -1, -1},
    { 3,  0,  9,  3,  9, 11,  1,  2,  9,  2, 11,  9, -1, -1, -1, -1},
    { 0,  2, 11,  8,  0, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 3,  2, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 2,  3,  8,  2,  8, 10, 10,  8,  9, -1, -1, -1, -1, -1, -1, -1},
    { 9, 10,  2,  0,  9,  2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 2,  3,  8,  2,  8, 10,  0,  1,  8,  1, 10,  8, -1, -1, -1, -1},
    { 1, 10,  2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 1,  3,  8,  9,  1,  8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 0,  9,  1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 0,  3,  8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1}
};

// Edge endpoint vertex indices (which two corners form each of the 12 edges).
__device__ __constant__ int mc_edge_vertices[12][2] = {
    {0, 1}, {1, 2}, {2, 3}, {3, 0},  // bottom face edges
    {4, 5}, {5, 6}, {6, 7}, {7, 4},  // top face edges
    {0, 4}, {1, 5}, {2, 6}, {3, 7}   // vertical edges
};

// Corner offsets: 8 corners of a unit cube in (dx, dy, dz).
__device__ __constant__ int mc_corner_offsets[8][3] = {
    {0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0},
    {0, 0, 1}, {1, 0, 1}, {1, 1, 1}, {0, 1, 1}
};

// Interpolate vertex position along an edge based on TSDF values at the two corners.
__device__ __forceinline__ void interpolate_vertex(
    float x0, float y0, float z0, float v0,
    float x1, float y1, float z1, float v1,
    float& ox, float& oy, float& oz
) {
    // Guard against division by zero
    float denom = v1 - v0;
    float t = (fabsf(denom) > 1e-8f) ? (-v0 / denom) : 0.5f;
    t = fmaxf(0.0f, fminf(1.0f, t));
    ox = x0 + t * (x1 - x0);
    oy = y0 + t * (y1 - y0);
    oz = z0 + t * (z1 - z0);
}

__global__ void marching_cubes_kernel(
    const TsdfVoxel* __restrict__ voxels,
    VolumeParams volume,
    MeshVertex* __restrict__ out_vertices,
    uint32_t* __restrict__ out_indices,
    int* __restrict__ out_tri_count,
    int max_triangles
) {
    // Each thread handles one cube (voxel at lower-left corner).
    // Grid dimensions are (dim_x-1) * (dim_y-1) * (dim_z-1).
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int num_cubes_x = volume.dim_x - 1;
    const int num_cubes_y = volume.dim_y - 1;
    const int num_cubes_z = volume.dim_z - 1;
    const int total_cubes = num_cubes_x * num_cubes_y * num_cubes_z;
    if (idx >= total_cubes) return;

    const int cz = idx / (num_cubes_x * num_cubes_y);
    const int cy = (idx - cz * num_cubes_x * num_cubes_y) / num_cubes_x;
    const int cx = idx - cz * num_cubes_x * num_cubes_y - cy * num_cubes_x;

    // Read TSDF values at 8 corners
    float corner_tsdf[8];
    float corner_pos[8][3];
    uint8_t corner_color[8][3];

    const float min_weight = 1.0f;  // require at least some observations

    bool all_valid = true;
    for (int c = 0; c < 8; c++) {
        int vx = cx + mc_corner_offsets[c][0];
        int vy = cy + mc_corner_offsets[c][1];
        int vz = cz + mc_corner_offsets[c][2];

        int vidx = voxel_index(vx, vy, vz, volume.dim_x, volume.dim_y);
        const TsdfVoxel& v = voxels[vidx];

        if (v.weight < min_weight) {
            all_valid = false;
            break;
        }

        corner_tsdf[c] = v.tsdf;
        corner_color[c][0] = v.r;
        corner_color[c][1] = v.g;
        corner_color[c][2] = v.b;

        voxel_world_pos(vx, vy, vz,
                        volume.origin_x, volume.origin_y, volume.origin_z,
                        volume.voxel_size,
                        corner_pos[c][0], corner_pos[c][1], corner_pos[c][2]);
    }

    if (!all_valid) return;

    // Determine cube configuration index (8-bit)
    int cube_index = 0;
    for (int c = 0; c < 8; c++) {
        if (corner_tsdf[c] < 0.0f) {
            cube_index |= (1 << c);
        }
    }

    // Look up which edges are intersected
    int edges = mc_edge_table[cube_index];
    if (edges == 0) return;  // no triangles — entirely inside or outside

    // Compute intersection points on each active edge
    float edge_verts[12][3];
    uint8_t edge_colors[12][3];

    for (int e = 0; e < 12; e++) {
        if (edges & (1 << e)) {
            int c0 = mc_edge_vertices[e][0];
            int c1 = mc_edge_vertices[e][1];
            interpolate_vertex(
                corner_pos[c0][0], corner_pos[c0][1], corner_pos[c0][2], corner_tsdf[c0],
                corner_pos[c1][0], corner_pos[c1][1], corner_pos[c1][2], corner_tsdf[c1],
                edge_verts[e][0], edge_verts[e][1], edge_verts[e][2]
            );

            // Interpolate color
            float denom = corner_tsdf[c1] - corner_tsdf[c0];
            float t = (fabsf(denom) > 1e-8f) ? (-corner_tsdf[c0] / denom) : 0.5f;
            t = fmaxf(0.0f, fminf(1.0f, t));
            edge_colors[e][0] = static_cast<uint8_t>(corner_color[c0][0] * (1.0f - t) + corner_color[c1][0] * t);
            edge_colors[e][1] = static_cast<uint8_t>(corner_color[c0][1] * (1.0f - t) + corner_color[c1][1] * t);
            edge_colors[e][2] = static_cast<uint8_t>(corner_color[c0][2] * (1.0f - t) + corner_color[c1][2] * t);
        }
    }

    // Emit triangles
    for (int t = 0; mc_tri_table[cube_index][t] != -1; t += 3) {
        int tri_idx = atomicAdd(out_tri_count, 1);
        if (tri_idx >= max_triangles) return;

        int base_vert = tri_idx * 3;
        int base_idx = tri_idx * 3;

        for (int v = 0; v < 3; v++) {
            int edge = mc_tri_table[cube_index][t + v];
            MeshVertex& mv = out_vertices[base_vert + v];
            mv.x = edge_verts[edge][0];
            mv.y = edge_verts[edge][1];
            mv.z = edge_verts[edge][2];
            mv.r = edge_colors[edge][0];
            mv.g = edge_colors[edge][1];
            mv.b = edge_colors[edge][2];

            // Normal will be computed from the triangle after extraction
            mv.nx = 0.0f;
            mv.ny = 0.0f;
            mv.nz = 0.0f;
        }

        // Compute face normal from triangle vertices
        float ax = out_vertices[base_vert + 1].x - out_vertices[base_vert].x;
        float ay = out_vertices[base_vert + 1].y - out_vertices[base_vert].y;
        float az = out_vertices[base_vert + 1].z - out_vertices[base_vert].z;
        float bx = out_vertices[base_vert + 2].x - out_vertices[base_vert].x;
        float by = out_vertices[base_vert + 2].y - out_vertices[base_vert].y;
        float bz = out_vertices[base_vert + 2].z - out_vertices[base_vert].z;

        float nx = ay * bz - az * by;
        float ny = az * bx - ax * bz;
        float nz = ax * by - ay * bx;
        float len = sqrtf(nx*nx + ny*ny + nz*nz);
        if (len > 1e-8f) {
            nx /= len; ny /= len; nz /= len;
        }

        for (int v = 0; v < 3; v++) {
            out_vertices[base_vert + v].nx = nx;
            out_vertices[base_vert + v].ny = ny;
            out_vertices[base_vert + v].nz = nz;
        }

        // Indices: simple sequential (no vertex sharing — could be optimized later)
        out_indices[base_idx + 0] = base_vert + 0;
        out_indices[base_idx + 1] = base_vert + 1;
        out_indices[base_idx + 2] = base_vert + 2;
    }
}

// ── Point extraction kernel ─────────────────────────────────────────────

// Extract points from zero-crossings of the TSDF field.
// For each voxel, check its 3 positive-direction neighbors; if there is
// a sign change, emit a point at the interpolated position.
__global__ void extract_points_kernel(
    const TsdfVoxel* __restrict__ voxels,
    VolumeParams volume,
    float* __restrict__ out_positions,    // Nx3
    float* __restrict__ out_normals,      // Nx3
    uint8_t* __restrict__ out_colors,     // Nx3
    int* __restrict__ out_count,
    int max_points
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= volume.total_voxels) return;

    const int vz = idx / (volume.dim_x * volume.dim_y);
    const int vy = (idx - vz * volume.dim_x * volume.dim_y) / volume.dim_x;
    const int vx = idx - vz * volume.dim_x * volume.dim_y - vy * volume.dim_x;

    // Need room for neighbors
    if (vx >= volume.dim_x - 1 || vy >= volume.dim_y - 1 || vz >= volume.dim_z - 1) return;

    const TsdfVoxel& v0 = voxels[idx];
    if (v0.weight < 1.0f) return;
    if (fabsf(v0.tsdf) > 0.99f) return;  // far from surface

    // Check 3 neighbors for zero-crossing
    int neighbors[3] = {
        voxel_index(vx + 1, vy, vz, volume.dim_x, volume.dim_y),
        voxel_index(vx, vy + 1, vz, volume.dim_x, volume.dim_y),
        voxel_index(vx, vy, vz + 1, volume.dim_x, volume.dim_y)
    };

    for (int n = 0; n < 3; n++) {
        const TsdfVoxel& v1 = voxels[neighbors[n]];
        if (v1.weight < 1.0f) continue;

        // Zero-crossing check
        if ((v0.tsdf > 0.0f) != (v1.tsdf > 0.0f)) {
            int pt_idx = atomicAdd(out_count, 1);
            if (pt_idx >= max_points) return;

            // Interpolate position
            float p0x, p0y, p0z, p1x, p1y, p1z;
            voxel_world_pos(vx, vy, vz,
                            volume.origin_x, volume.origin_y, volume.origin_z,
                            volume.voxel_size, p0x, p0y, p0z);

            int nx = vx + (n == 0 ? 1 : 0);
            int ny = vy + (n == 1 ? 1 : 0);
            int nz = vz + (n == 2 ? 1 : 0);
            voxel_world_pos(nx, ny, nz,
                            volume.origin_x, volume.origin_y, volume.origin_z,
                            volume.voxel_size, p1x, p1y, p1z);

            float ox, oy, oz;
            interpolate_vertex(p0x, p0y, p0z, v0.tsdf,
                               p1x, p1y, p1z, v1.tsdf,
                               ox, oy, oz);

            out_positions[pt_idx * 3 + 0] = ox;
            out_positions[pt_idx * 3 + 1] = oy;
            out_positions[pt_idx * 3 + 2] = oz;

            // Approximate normal from TSDF gradient (central differences)
            float gx = 0.0f, gy = 0.0f, gz = 0.0f;
            if (vx > 0 && vx < volume.dim_x - 1) {
                int im = voxel_index(vx - 1, vy, vz, volume.dim_x, volume.dim_y);
                int ip = voxel_index(vx + 1, vy, vz, volume.dim_x, volume.dim_y);
                gx = voxels[ip].tsdf - voxels[im].tsdf;
            }
            if (vy > 0 && vy < volume.dim_y - 1) {
                int im = voxel_index(vx, vy - 1, vz, volume.dim_x, volume.dim_y);
                int ip = voxel_index(vx, vy + 1, vz, volume.dim_x, volume.dim_y);
                gy = voxels[ip].tsdf - voxels[im].tsdf;
            }
            if (vz > 0 && vz < volume.dim_z - 1) {
                int im = voxel_index(vx, vy, vz - 1, volume.dim_x, volume.dim_y);
                int ip = voxel_index(vx, vy, vz + 1, volume.dim_x, volume.dim_y);
                gz = voxels[ip].tsdf - voxels[im].tsdf;
            }
            float glen = sqrtf(gx*gx + gy*gy + gz*gz);
            if (glen > 1e-8f) {
                gx /= glen; gy /= glen; gz /= glen;
            }

            out_normals[pt_idx * 3 + 0] = gx;
            out_normals[pt_idx * 3 + 1] = gy;
            out_normals[pt_idx * 3 + 2] = gz;

            // Interpolate color
            float denom = v1.tsdf - v0.tsdf;
            float t = (fabsf(denom) > 1e-8f) ? (-v0.tsdf / denom) : 0.5f;
            t = fmaxf(0.0f, fminf(1.0f, t));

            out_colors[pt_idx * 3 + 0] = static_cast<uint8_t>(v0.r * (1.0f - t) + v1.r * t);
            out_colors[pt_idx * 3 + 1] = static_cast<uint8_t>(v0.g * (1.0f - t) + v1.g * t);
            out_colors[pt_idx * 3 + 2] = static_cast<uint8_t>(v0.b * (1.0f - t) + v1.b * t);
        }
    }
}

// ── TsdfVolume implementation ───────────────────────────────────────────

TsdfVolume::TsdfVolume(const VolumeParams& params)
    : params_(params)
{
    allocate_gpu_memory();
    reset();
}

TsdfVolume::~TsdfVolume() {
    free_gpu_memory();
}

TsdfVolume::TsdfVolume(TsdfVolume&& other) noexcept
    : params_(other.params_)
    , d_voxels_(other.d_voxels_)
    , d_mc_vertices_(other.d_mc_vertices_)
    , d_mc_indices_(other.d_mc_indices_)
    , d_mc_count_(other.d_mc_count_)
    , d_pt_positions_(other.d_pt_positions_)
    , d_pt_normals_(other.d_pt_normals_)
    , d_pt_colors_(other.d_pt_colors_)
    , d_pt_count_(other.d_pt_count_)
    , integration_count_(other.integration_count_)
{
    other.d_voxels_ = nullptr;
    other.d_mc_vertices_ = nullptr;
    other.d_mc_indices_ = nullptr;
    other.d_mc_count_ = nullptr;
    other.d_pt_positions_ = nullptr;
    other.d_pt_normals_ = nullptr;
    other.d_pt_colors_ = nullptr;
    other.d_pt_count_ = nullptr;
}

TsdfVolume& TsdfVolume::operator=(TsdfVolume&& other) noexcept {
    if (this != &other) {
        free_gpu_memory();
        params_ = other.params_;
        d_voxels_ = other.d_voxels_;
        d_mc_vertices_ = other.d_mc_vertices_;
        d_mc_indices_ = other.d_mc_indices_;
        d_mc_count_ = other.d_mc_count_;
        d_pt_positions_ = other.d_pt_positions_;
        d_pt_normals_ = other.d_pt_normals_;
        d_pt_colors_ = other.d_pt_colors_;
        d_pt_count_ = other.d_pt_count_;
        integration_count_ = other.integration_count_;

        other.d_voxels_ = nullptr;
        other.d_mc_vertices_ = nullptr;
        other.d_mc_indices_ = nullptr;
        other.d_mc_count_ = nullptr;
        other.d_pt_positions_ = nullptr;
        other.d_pt_normals_ = nullptr;
        other.d_pt_colors_ = nullptr;
        other.d_pt_count_ = nullptr;
    }
    return *this;
}

void TsdfVolume::allocate_gpu_memory() {
    // Voxel grid
    CUDA_CHECK(cudaMalloc(&d_voxels_, params_.total_voxels * sizeof(TsdfVoxel)));

    // Marching cubes output buffers
    size_t mc_vert_bytes = kMaxMarchingCubesTriangles * 3 * sizeof(MeshVertex);
    size_t mc_idx_bytes = kMaxMarchingCubesTriangles * 3 * sizeof(uint32_t);
    CUDA_CHECK(cudaMalloc(&d_mc_vertices_, mc_vert_bytes));
    CUDA_CHECK(cudaMalloc(&d_mc_indices_, mc_idx_bytes));
    CUDA_CHECK(cudaMalloc(&d_mc_count_, sizeof(int)));

    // Point extraction output buffers
    CUDA_CHECK(cudaMalloc(&d_pt_positions_, kMaxExtractedPoints * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_pt_normals_, kMaxExtractedPoints * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_pt_colors_, kMaxExtractedPoints * 3 * sizeof(uint8_t)));
    CUDA_CHECK(cudaMalloc(&d_pt_count_, sizeof(int)));
}

void TsdfVolume::free_gpu_memory() {
    if (d_voxels_)       { cudaFree(d_voxels_);       d_voxels_ = nullptr; }
    if (d_mc_vertices_)  { cudaFree(d_mc_vertices_);  d_mc_vertices_ = nullptr; }
    if (d_mc_indices_)   { cudaFree(d_mc_indices_);   d_mc_indices_ = nullptr; }
    if (d_mc_count_)     { cudaFree(d_mc_count_);     d_mc_count_ = nullptr; }
    if (d_pt_positions_) { cudaFree(d_pt_positions_);  d_pt_positions_ = nullptr; }
    if (d_pt_normals_)   { cudaFree(d_pt_normals_);   d_pt_normals_ = nullptr; }
    if (d_pt_colors_)    { cudaFree(d_pt_colors_);    d_pt_colors_ = nullptr; }
    if (d_pt_count_)     { cudaFree(d_pt_count_);     d_pt_count_ = nullptr; }
}

void TsdfVolume::reset(cudaStream_t stream) {
    const int threads = 256;
    const int blocks = (static_cast<int>(params_.total_voxels) + threads - 1) / threads;
    tsdf_reset_kernel<<<blocks, threads, 0, stream>>>(d_voxels_, params_.total_voxels);
    integration_count_ = 0;
}

void TsdfVolume::integrate(
    const float* depth_gpu,
    const uint8_t* color_gpu,
    const uint8_t* alpha_gpu,
    const FusionCameraParams& camera,
    cudaStream_t stream
) {
    const int threads = 256;
    const int blocks = (static_cast<int>(params_.total_voxels) + threads - 1) / threads;

    tsdf_integrate_kernel<<<blocks, threads, 0, stream>>>(
        d_voxels_, depth_gpu, color_gpu, alpha_gpu, camera, params_
    );

    integration_count_++;
}

ExtractedMesh TsdfVolume::extract_mesh(cudaStream_t stream) {
    // Reset triangle counter
    CUDA_CHECK(cudaMemsetAsync(d_mc_count_, 0, sizeof(int), stream));

    int num_cubes_x = params_.dim_x - 1;
    int num_cubes_y = params_.dim_y - 1;
    int num_cubes_z = params_.dim_z - 1;
    int total_cubes = num_cubes_x * num_cubes_y * num_cubes_z;

    const int threads = 256;
    const int blocks = (total_cubes + threads - 1) / threads;

    marching_cubes_kernel<<<blocks, threads, 0, stream>>>(
        d_voxels_, params_,
        d_mc_vertices_, d_mc_indices_, d_mc_count_,
        kMaxMarchingCubesTriangles
    );

    // Read triangle count
    int num_triangles = 0;
    CUDA_CHECK(cudaMemcpyAsync(&num_triangles, d_mc_count_, sizeof(int),
                                cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    num_triangles = std::min(num_triangles, kMaxMarchingCubesTriangles);

    // Copy results to host
    ExtractedMesh mesh;
    int num_verts = num_triangles * 3;
    mesh.vertices.resize(num_verts);
    mesh.indices.resize(num_verts);

    if (num_verts > 0) {
        CUDA_CHECK(cudaMemcpy(mesh.vertices.data(), d_mc_vertices_,
                              num_verts * sizeof(MeshVertex), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(mesh.indices.data(), d_mc_indices_,
                              num_verts * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    }

    return mesh;
}

ExtractedPoints TsdfVolume::extract_points(cudaStream_t stream) {
    // Reset point counter
    CUDA_CHECK(cudaMemsetAsync(d_pt_count_, 0, sizeof(int), stream));

    const int threads = 256;
    const int blocks = (static_cast<int>(params_.total_voxels) + threads - 1) / threads;

    extract_points_kernel<<<blocks, threads, 0, stream>>>(
        d_voxels_, params_,
        d_pt_positions_, d_pt_normals_, d_pt_colors_, d_pt_count_,
        kMaxExtractedPoints
    );

    // Read point count
    int num_points = 0;
    CUDA_CHECK(cudaMemcpyAsync(&num_points, d_pt_count_, sizeof(int),
                                cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    num_points = std::min(num_points, kMaxExtractedPoints);

    // Copy results to host
    ExtractedPoints pts;
    pts.num_points = num_points;

    if (num_points > 0) {
        pts.positions.resize(num_points * 3);
        pts.normals.resize(num_points * 3);
        pts.colors.resize(num_points * 3);

        CUDA_CHECK(cudaMemcpy(pts.positions.data(), d_pt_positions_,
                              num_points * 3 * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(pts.normals.data(), d_pt_normals_,
                              num_points * 3 * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(pts.colors.data(), d_pt_colors_,
                              num_points * 3 * sizeof(uint8_t), cudaMemcpyDeviceToHost));
    }

    return pts;
}

} // namespace heimdall::fusion
