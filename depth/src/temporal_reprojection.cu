#include "temporal_reprojection.h"

namespace heimdall::depth {

__global__ void temporal_reprojection_kernel(
    const float* __restrict__ prev_depth,
    const float* __restrict__ prev_confidence,
    float* __restrict__ out_depth,
    float* __restrict__ out_confidence,
    ReprojectionParams params
) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= params.width || y >= params.height) return;

    const int idx = y * params.width + x;
    out_depth[idx] = 0.0f;
    out_confidence[idx] = 0.0f;

    // Unproject pixel (x, y) from previous frame using previous intrinsics
    float d = prev_depth[idx];
    if (d < params.depth_min_m || d > params.depth_max_m) return;

    float conf = prev_confidence[idx];
    if (conf < params.confidence_threshold) return;

    // Previous camera space
    float pc_x = (x - params.prev_cx) / params.prev_fx * d;
    float pc_y = (y - params.prev_cy) / params.prev_fy * d;
    float pc_z = d;

    // Previous camera → world
    const float* m = params.prev_cam_to_world;
    float pw_x = m[0]*pc_x + m[1]*pc_y + m[2]*pc_z  + m[3];
    float pw_y = m[4]*pc_x + m[5]*pc_y + m[6]*pc_z  + m[7];
    float pw_z = m[8]*pc_x + m[9]*pc_y + m[10]*pc_z + m[11];

    // World → current camera
    const float* n = params.world_to_curr_cam;
    float cc_x = n[0]*pw_x + n[1]*pw_y + n[2]*pw_z  + n[3];
    float cc_y = n[4]*pw_x + n[5]*pw_y + n[6]*pw_z  + n[7];
    float cc_z = n[8]*pw_x + n[9]*pw_y + n[10]*pw_z + n[11];

    if (cc_z < params.depth_min_m) return;

    // Project into current frame
    float proj_x = params.fx * (cc_x / cc_z) + params.cx;
    float proj_y = params.fy * (cc_y / cc_z) + params.cy;

    int px = __float2int_rn(proj_x);
    int py = __float2int_rn(proj_y);
    if (px < 0 || px >= params.width || py < 0 || py >= params.height) return;

    int out_idx = py * params.width + px;

    // Atomic max by confidence — highest confidence write wins
    // Using atomicCAS pattern for float comparison
    float existing_conf = out_confidence[out_idx];
    if (conf > existing_conf) {
        out_depth[out_idx] = cc_z;
        out_confidence[out_idx] = conf;
    }
}

void launch_temporal_reprojection(
    const float* prev_depth,
    const float* prev_confidence,
    float* out_seed_depth,
    float* out_seed_confidence,
    const ReprojectionParams& params,
    cudaStream_t stream
) {
    // Zero the output buffers
    size_t buf_size = params.width * params.height * sizeof(float);
    cudaMemsetAsync(out_seed_depth, 0, buf_size, stream);
    cudaMemsetAsync(out_seed_confidence, 0, buf_size, stream);

    dim3 block(16, 16);
    dim3 grid(
        (params.width  + block.x - 1) / block.x,
        (params.height + block.y - 1) / block.y
    );

    temporal_reprojection_kernel<<<grid, block, 0, stream>>>(
        prev_depth, prev_confidence,
        out_seed_depth, out_seed_confidence,
        params
    );
}

} // namespace heimdall::depth
