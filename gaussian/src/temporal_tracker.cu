#include "temporal_tracker.h"
#include <cstring>
#include <algorithm>
#include <stdexcept>

namespace heimdall::gaussian {

// --- CUDA Kernels ---

__device__ float gaussian_distance(const Gaussian& a, const Gaussian& b, float color_weight) {
    // Euclidean distance in position
    float dx = a.position[0] - b.position[0];
    float dy = a.position[1] - b.position[1];
    float dz = a.position[2] - b.position[2];
    float pos_dist = sqrtf(dx*dx + dy*dy + dz*dz);

    // L1 distance in first 3 SH coefficients (DC term = base color)
    float dc0 = fabsf(a.sh[0] - b.sh[0]);
    float dc1 = fabsf(a.sh[1] - b.sh[1]);
    float dc2 = fabsf(a.sh[2] - b.sh[2]);
    float color_dist = (dc0 + dc1 + dc2) / 3.0f;

    return pos_dist + color_weight * color_dist;
}

__global__ void match_kernel(
    const Gaussian* __restrict__ new_gs, int num_new,
    const Gaussian* __restrict__ old_gs, int num_old,
    int* __restrict__ match_indices,
    float match_threshold,
    float color_weight
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_new) return;

    float best_dist = match_threshold;
    int best_idx = -1;

    // Brute-force nearest neighbor. For production, replace with
    // spatial hash grid or BVH for O(N log N) instead of O(N*M).
    for (int j = 0; j < num_old; j++) {
        float d = gaussian_distance(new_gs[i], old_gs[j], color_weight);
        if (d < best_dist) {
            best_dist = d;
            best_idx = j;
        }
    }

    match_indices[i] = best_idx;
}

__global__ void blend_kernel(
    Gaussian* __restrict__ persistent,
    const Gaussian* __restrict__ inferred,
    const int* __restrict__ match_indices,
    int num_inferred,
    float decay
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_inferred) return;

    int old_idx = match_indices[i];
    if (old_idx < 0) return;

    Gaussian& dst = persistent[old_idx];
    const Gaussian& src = inferred[i];
    float w_new = 1.0f - decay;

    // Position: full EMA (allow movement)
    dst.position[0] = decay * dst.position[0] + w_new * src.position[0];
    dst.position[1] = decay * dst.position[1] + w_new * src.position[1];
    dst.position[2] = decay * dst.position[2] + w_new * src.position[2];

    // Scale: EMA
    dst.scale[0] = decay * dst.scale[0] + w_new * src.scale[0];
    dst.scale[1] = decay * dst.scale[1] + w_new * src.scale[1];
    dst.scale[2] = decay * dst.scale[2] + w_new * src.scale[2];

    // Rotation: SLERP approximation via normalized EMA
    for (int q = 0; q < 4; q++) {
        dst.rotation[q] = decay * dst.rotation[q] + w_new * src.rotation[q];
    }
    // Renormalize quaternion
    float norm = sqrtf(
        dst.rotation[0]*dst.rotation[0] + dst.rotation[1]*dst.rotation[1] +
        dst.rotation[2]*dst.rotation[2] + dst.rotation[3]*dst.rotation[3]
    );
    if (norm > 1e-6f) {
        for (int q = 0; q < 4; q++) dst.rotation[q] /= norm;
    }

    // Opacity: EMA
    dst.opacity = decay * dst.opacity + w_new * src.opacity;

    // SH coefficients: slower blending for appearance stability
    float sh_decay = decay * 0.5f + 0.5f; // bias toward keeping old appearance
    float sh_new = 1.0f - sh_decay;
    for (int s = 0; s < 48; s++) {
        dst.sh[s] = sh_decay * dst.sh[s] + sh_new * src.sh[s];
    }
}

__global__ void ramp_kernel(
    Gaussian* __restrict__ persistent,
    int* __restrict__ ages,
    int num_persistent,
    int ramp_in_frames,
    int ramp_out_frames,
    float min_opacity
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_persistent) return;

    int age = ages[i];

    if (age > 0 && age <= ramp_in_frames) {
        // Ramping in: scale opacity by age/ramp_in
        float t = (float)age / (float)ramp_in_frames;
        persistent[i].opacity *= t;
    } else if (age < 0) {
        // Ramping out: age is negative (frames since last match)
        float t = 1.0f + (float)age / (float)ramp_out_frames; // goes from 1 to 0
        if (t < 0.0f) t = 0.0f;
        persistent[i].opacity *= t;
    }

    // Mark for removal if below threshold
    if (persistent[i].opacity < min_opacity) {
        persistent[i].opacity = 0.0f;
    }
}

// --- TemporalTracker Implementation ---

TemporalTracker::TemporalTracker(const TrackerConfig& config)
    : config_(config) {}

TemporalTracker::~TemporalTracker() {
    reset();
}

void TemporalTracker::reset() {
    if (d_persistent_) { cudaFree(d_persistent_); d_persistent_ = nullptr; }
    if (d_ages_) { cudaFree(d_ages_); d_ages_ = nullptr; }
    num_persistent_ = 0;
    last_matched_ = last_added_ = last_removed_ = 0;
}

void TemporalTracker::match_gaussians(
    const Gaussian* new_gs, int num_new,
    const Gaussian* old_gs, int num_old,
    int* match_indices,
    cudaStream_t stream
) {
    int block = 256;
    int grid = (num_new + block - 1) / block;
    match_kernel<<<grid, block, 0, stream>>>(
        new_gs, num_new, old_gs, num_old, match_indices,
        config_.match_distance_threshold, config_.match_color_weight
    );
}

void TemporalTracker::blend_matched(
    Gaussian* persistent,
    const Gaussian* inferred,
    const int* match_indices,
    int num_inferred,
    float decay,
    cudaStream_t stream
) {
    int block = 256;
    int grid = (num_inferred + block - 1) / block;
    blend_kernel<<<grid, block, 0, stream>>>(
        persistent, inferred, match_indices, num_inferred, decay
    );
}

void TemporalTracker::update_ramps(
    Gaussian* persistent,
    int* ages,
    int num_persistent,
    cudaStream_t stream
) {
    int block = 256;
    int grid = (num_persistent + block - 1) / block;
    ramp_kernel<<<grid, block, 0, stream>>>(
        persistent, ages, num_persistent,
        config_.ramp_in_frames, config_.ramp_out_frames,
        config_.min_opacity_threshold
    );
}

GaussianFrame TemporalTracker::process(const GaussianFrame& inferred) {
    // First frame: initialize persistent buffer directly
    if (num_persistent_ == 0) {
        num_persistent_ = inferred.num_gaussians;
        size_t gs_bytes = num_persistent_ * sizeof(Gaussian);
        size_t age_bytes = num_persistent_ * sizeof(int);
        cudaMalloc(&d_persistent_, gs_bytes);
        cudaMalloc(&d_ages_, age_bytes);
        cudaMemcpy(d_persistent_, inferred.gaussians.data(), gs_bytes, cudaMemcpyHostToDevice);
        cudaMemset(d_ages_, 1, age_bytes); // age=1 for all (just added)

        last_matched_ = 0;
        last_added_ = num_persistent_;
        last_removed_ = 0;

        // Return as-is for first frame
        return inferred;
    }

    // Upload inferred to device
    int num_new = inferred.num_gaussians;
    Gaussian* d_inferred;
    cudaMalloc(&d_inferred, num_new * sizeof(Gaussian));
    cudaMemcpy(d_inferred, inferred.gaussians.data(), num_new * sizeof(Gaussian), cudaMemcpyHostToDevice);

    // Allocate match indices
    int* d_match_indices;
    cudaMalloc(&d_match_indices, num_new * sizeof(int));

    // Step 1: match new to persistent
    match_gaussians(d_inferred, num_new, d_persistent_, num_persistent_, d_match_indices, nullptr);

    // Step 2: blend matched
    blend_matched(d_persistent_, d_inferred, d_match_indices, num_new, config_.ema_decay, nullptr);

    // Step 3: update ramp in/out
    update_ramps(d_persistent_, d_ages_, num_persistent_, nullptr);

    cudaDeviceSynchronize();

    // Download result
    GaussianFrame result;
    result.frame_id = inferred.frame_id;
    result.timestamp_ns = inferred.timestamp_ns;
    result.num_gaussians = num_persistent_;
    result.gaussians.resize(num_persistent_);
    result.is_keyframe = inferred.is_keyframe;
    cudaMemcpy(result.gaussians.data(), d_persistent_, num_persistent_ * sizeof(Gaussian), cudaMemcpyDeviceToHost);

    // Cleanup
    cudaFree(d_inferred);
    cudaFree(d_match_indices);

    // TODO: compact buffer (remove opacity=0 Gaussians) and
    // append unmatched new Gaussians. For now this is a simplified
    // version that only blends matched Gaussians.

    return result;
}

int TemporalTracker::persistent_count() const { return num_persistent_; }
int TemporalTracker::matched_count() const { return last_matched_; }
int TemporalTracker::added_count() const { return last_added_; }
int TemporalTracker::removed_count() const { return last_removed_; }

} // namespace heimdall::gaussian
