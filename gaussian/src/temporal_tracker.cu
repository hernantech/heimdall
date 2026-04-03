#include "temporal_tracker.h"
#include "spatial_hash.h"
#include <cstring>
#include <algorithm>
#include <stdexcept>

#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/remove.h>
#include <thrust/execution_policy.h>
#include <thrust/count.h>
#include <thrust/sequence.h>
#include <thrust/functional.h>

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

// Mark which persistent Gaussians were matched and update ages accordingly.
// matched_flags[j] = 1 if persistent Gaussian j was matched, 0 otherwise.
// Ages: matched -> reset to max(age, 1) (positive); unmatched -> decrement toward negative.
__global__ void mark_matched_kernel(
    const int* __restrict__ match_indices,
    int num_new,
    int* __restrict__ matched_flags, // [num_persistent], zeroed beforehand
    int num_persistent
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_new) return;

    int idx = match_indices[i];
    if (idx >= 0 && idx < num_persistent) {
        matched_flags[idx] = 1;
    }
}

__global__ void update_ages_kernel(
    int* __restrict__ ages,
    const int* __restrict__ matched_flags,
    int num_persistent
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_persistent) return;

    if (matched_flags[i]) {
        // Matched: keep positive age, increment (more established)
        int a = ages[i];
        ages[i] = (a > 0) ? a + 1 : 1;
    } else {
        // Unmatched: push age negative (toward removal)
        int a = ages[i];
        ages[i] = (a > 0) ? -1 : a - 1;
    }
}

// Gather unmatched new Gaussians into a contiguous append buffer.
// Uses atomic counter to get compact write position.
__global__ void gather_unmatched_kernel(
    const Gaussian* __restrict__ new_gs,
    const int* __restrict__ match_indices,
    int num_new,
    Gaussian* __restrict__ out_buffer,
    int* __restrict__ out_ages,
    int* __restrict__ write_counter // single int, zeroed beforehand
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_new) return;
    if (match_indices[i] != -1) return;

    int slot = atomicAdd(write_counter, 1);
    out_buffer[slot] = new_gs[i];
    out_ages[slot] = 1; // newly added
}

// For budget enforcement: write (opacity, index) pairs for sorting.
__global__ void write_opacity_index_kernel(
    const Gaussian* __restrict__ gaussians,
    float* __restrict__ opacities,
    int* __restrict__ indices,
    int num
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num) return;
    opacities[i] = gaussians[i].opacity;
    indices[i] = i;
}

// Compact selected Gaussians into a new buffer given a sorted index list.
__global__ void scatter_by_index_kernel(
    const Gaussian* __restrict__ src_gs,
    const int* __restrict__ src_ages,
    const int* __restrict__ keep_indices,
    int num_keep,
    Gaussian* __restrict__ dst_gs,
    int* __restrict__ dst_ages
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_keep) return;
    int j = keep_indices[i];
    dst_gs[i] = src_gs[j];
    dst_ages[i] = src_ages[j];
}

// --- TemporalTracker Implementation ---

// Persistent spatial hash storage — lives across frames, reallocated only if capacity grows.
// Note: file-static; assumes a single TemporalTracker instance per process.
// If multiple instances are needed, move this into the class (requires adding
// SpatialHashStorage to the header or using a pimpl).
static SpatialHashStorage s_hash_storage;

static constexpr int kBlockSize = 256;

static int div_ceil(int n, int d) {
    return (n + d - 1) / d;
}

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
    s_hash_storage.free();
}

void TemporalTracker::match_gaussians(
    const Gaussian* new_gs, int num_new,
    const Gaussian* old_gs, int num_old,
    int* match_indices,
    cudaStream_t stream
) {
    if (num_old == 0 || num_new == 0) {
        // No persistent Gaussians -> all unmatched.
        cudaMemsetAsync(match_indices, 0xFF, num_new * sizeof(int), stream); // sets to -1 (all bits)
        return;
    }

    float cell_size = config_.match_distance_threshold;

    // Ensure hash storage is large enough.
    s_hash_storage.allocate(num_old);

    // Step 1: Hash persistent Gaussians into cells.
    hash_gaussians_kernel<<<div_ceil(num_old, kBlockSize), kBlockSize, 0, stream>>>(
        old_gs, s_hash_storage.cell_hashes, s_hash_storage.sorted_indices,
        num_old, cell_size
    );

    // Step 2: Sort (cell_hash, index) pairs by cell hash.
    thrust::device_ptr<uint32_t> d_hashes(s_hash_storage.cell_hashes);
    thrust::device_ptr<int> d_indices(s_hash_storage.sorted_indices);
    thrust::sort_by_key(thrust::cuda::par.on(stream), d_hashes, d_hashes + num_old, d_indices);

    // Step 3: Clear and build cell start/end tables.
    clear_cell_table_kernel<<<div_ceil(kHashTableSize, kBlockSize), kBlockSize, 0, stream>>>(
        s_hash_storage.cell_start, s_hash_storage.cell_end, kHashTableSize
    );

    build_cell_table_kernel<<<div_ceil(num_old, kBlockSize), kBlockSize, 0, stream>>>(
        s_hash_storage.cell_hashes, s_hash_storage.cell_start, s_hash_storage.cell_end, num_old
    );

    // Step 4: Query — find nearest persistent Gaussian for each new Gaussian.
    SpatialHashGrid grid = s_hash_storage.make_grid(cell_size, num_old);
    hash_match_kernel<<<div_ceil(num_new, kBlockSize), kBlockSize, 0, stream>>>(
        new_gs, num_new, old_gs, grid, match_indices,
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
    if (num_inferred == 0) return;
    blend_kernel<<<div_ceil(num_inferred, kBlockSize), kBlockSize, 0, stream>>>(
        persistent, inferred, match_indices, num_inferred, decay
    );
}

void TemporalTracker::update_ramps(
    Gaussian* persistent,
    int* ages,
    int num_persistent,
    cudaStream_t stream
) {
    if (num_persistent == 0) return;
    ramp_kernel<<<div_ceil(num_persistent, kBlockSize), kBlockSize, 0, stream>>>(
        persistent, ages, num_persistent,
        config_.ramp_in_frames, config_.ramp_out_frames,
        config_.min_opacity_threshold
    );
}

GaussianFrame TemporalTracker::process(const GaussianFrame& inferred) {
    // Handle empty inferred frame.
    if (inferred.num_gaussians == 0 && num_persistent_ == 0) {
        GaussianFrame result;
        result.frame_id = inferred.frame_id;
        result.timestamp_ns = inferred.timestamp_ns;
        result.num_gaussians = 0;
        result.is_keyframe = inferred.is_keyframe;
        last_matched_ = last_added_ = last_removed_ = 0;
        return result;
    }

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // --- First frame: initialize persistent buffer directly ---
    if (num_persistent_ == 0) {
        num_persistent_ = inferred.num_gaussians;
        if (num_persistent_ > config_.max_gaussians) {
            num_persistent_ = config_.max_gaussians;
        }
        size_t gs_bytes = num_persistent_ * sizeof(Gaussian);
        size_t age_bytes = num_persistent_ * sizeof(int);
        cudaMalloc(&d_persistent_, gs_bytes);
        cudaMalloc(&d_ages_, age_bytes);
        cudaMemcpyAsync(d_persistent_, inferred.gaussians.data(), gs_bytes, cudaMemcpyHostToDevice, stream);

        // Set all ages to 1 (one int at a time via a tiny kernel or memset-then-fix).
        // cudaMemset sets bytes, not ints. Use a fill via thrust.
        thrust::device_ptr<int> d_ages_ptr(d_ages_);
        thrust::fill(thrust::cuda::par.on(stream), d_ages_ptr, d_ages_ptr + num_persistent_, 1);

        last_matched_ = 0;
        last_added_ = num_persistent_;
        last_removed_ = 0;

        // Download result.
        GaussianFrame result;
        result.frame_id = inferred.frame_id;
        result.timestamp_ns = inferred.timestamp_ns;
        result.num_gaussians = num_persistent_;
        result.gaussians.resize(num_persistent_);
        result.is_keyframe = inferred.is_keyframe;
        cudaMemcpyAsync(result.gaussians.data(), d_persistent_,
                         num_persistent_ * sizeof(Gaussian), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        cudaStreamDestroy(stream);
        return result;
    }

    // --- Subsequent frames ---

    int num_new = inferred.num_gaussians;

    // Upload inferred Gaussians to device.
    Gaussian* d_inferred = nullptr;
    if (num_new > 0) {
        cudaMalloc(&d_inferred, num_new * sizeof(Gaussian));
        cudaMemcpyAsync(d_inferred, inferred.gaussians.data(),
                         num_new * sizeof(Gaussian), cudaMemcpyHostToDevice, stream);
    }

    // Allocate match indices.
    int* d_match_indices = nullptr;
    if (num_new > 0) {
        cudaMalloc(&d_match_indices, num_new * sizeof(int));
    }

    // Step 1: Match new Gaussians to persistent buffer via spatial hash.
    match_gaussians(
        d_inferred, num_new,
        d_persistent_, num_persistent_,
        d_match_indices, stream
    );

    // Step 2: Mark which persistent Gaussians were matched and update ages.
    int* d_matched_flags = nullptr;
    cudaMalloc(&d_matched_flags, num_persistent_ * sizeof(int));
    cudaMemsetAsync(d_matched_flags, 0, num_persistent_ * sizeof(int), stream);

    if (num_new > 0) {
        mark_matched_kernel<<<div_ceil(num_new, kBlockSize), kBlockSize, 0, stream>>>(
            d_match_indices, num_new, d_matched_flags, num_persistent_
        );
    }

    update_ages_kernel<<<div_ceil(num_persistent_, kBlockSize), kBlockSize, 0, stream>>>(
        d_ages_, d_matched_flags, num_persistent_
    );

    // Step 3: Blend matched Gaussians (EMA).
    blend_matched(d_persistent_, d_inferred, d_match_indices, num_new, config_.ema_decay, stream);

    // Step 4: Update ramp in/out and mark dead Gaussians (opacity < threshold -> 0).
    update_ramps(d_persistent_, d_ages_, num_persistent_, stream);

    // Step 5: Count matched persistent Gaussians (for stats).
    // thrust::count returns a host-side value (internally synchronizes the reduction).
    {
        thrust::device_ptr<int> flags_ptr(d_matched_flags);
        last_matched_ = static_cast<int>(thrust::count(
            thrust::cuda::par.on(stream),
            flags_ptr, flags_ptr + num_persistent_, 1));
    }
    cudaFree(d_matched_flags);

    // Step 6: Gather unmatched NEW Gaussians to append.
    int* d_append_count = nullptr;
    Gaussian* d_append_gs = nullptr;
    int* d_append_ages = nullptr;
    int h_append_count = 0;

    if (num_new > 0) {
        cudaMalloc(&d_append_count, sizeof(int));
        cudaMemsetAsync(d_append_count, 0, sizeof(int), stream);

        // Upper bound: all new Gaussians could be unmatched.
        cudaMalloc(&d_append_gs, num_new * sizeof(Gaussian));
        cudaMalloc(&d_append_ages, num_new * sizeof(int));

        gather_unmatched_kernel<<<div_ceil(num_new, kBlockSize), kBlockSize, 0, stream>>>(
            d_inferred, d_match_indices, num_new,
            d_append_gs, d_append_ages, d_append_count
        );

        // Download append count.
        cudaMemcpyAsync(&h_append_count, d_append_count, sizeof(int),
                         cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream); // need count on host for allocation decisions
    }

    last_added_ = h_append_count;

    // Step 7: Stream compaction — remove dead Gaussians (opacity == 0).
    //
    // Strategy: build a flag array (alive = opacity > 0), prefix-sum to get compact
    // positions, then scatter. Using thrust::copy_if with a zip iterator is simpler.
    //
    // We need to compact both d_persistent_ and d_ages_ in tandem.
    // Approach: create an index array, remove indices where opacity==0, then scatter.

    int* d_compact_indices = nullptr;
    cudaMalloc(&d_compact_indices, num_persistent_ * sizeof(int));
    thrust::device_ptr<int> compact_ptr(d_compact_indices);
    thrust::sequence(thrust::cuda::par.on(stream), compact_ptr, compact_ptr + num_persistent_);

    // Use a lambda-like approach: copy_if indices where the Gaussian is alive.
    // We need the persistent Gaussian opacities accessible from thrust.
    // Simplest: create a temporary opacity array and use stencil.
    float* d_opacities = nullptr;
    cudaMalloc(&d_opacities, num_persistent_ * sizeof(float));
    write_opacity_index_kernel<<<div_ceil(num_persistent_, kBlockSize), kBlockSize, 0, stream>>>(
        d_persistent_, d_opacities, d_compact_indices, num_persistent_
    );

    // Remove indices where opacity == 0 (dead Gaussians).
    thrust::device_ptr<float> opacity_ptr(d_opacities);

    // Partition: keep indices with opacity > 0.
    // We'll use thrust::copy_if on the index array with opacity as stencil.
    int* d_alive_indices = nullptr;
    cudaMalloc(&d_alive_indices, num_persistent_ * sizeof(int));
    thrust::device_ptr<int> alive_ptr(d_alive_indices);

    // copy_if: keep index i where opacity[i] > 0.
    struct IsAlive {
        __host__ __device__ bool operator()(float opacity) const {
            return opacity > 0.0f;
        }
    };
    auto end_it = thrust::copy_if(
        thrust::cuda::par.on(stream),
        compact_ptr, compact_ptr + num_persistent_,
        opacity_ptr, // stencil
        alive_ptr,
        IsAlive{}
    );
    int num_alive = static_cast<int>(end_it - alive_ptr);
    int num_removed = num_persistent_ - num_alive;
    last_removed_ = num_removed;

    // Step 8: Build new persistent buffer = compacted alive + appended unmatched new.
    int new_total = num_alive + h_append_count;

    // Enforce max_gaussians cap.
    // If over budget after append, we must trim the lowest-opacity Gaussians.
    bool needs_budget_trim = (new_total > config_.max_gaussians);

    // Allocate new buffers large enough for the combined result (alive + appended).
    // If over budget, we'll trim after; allocate for full new_total now.
    Gaussian* d_new_persistent = nullptr;
    int* d_new_ages = nullptr;
    if (new_total > 0) {
        cudaMalloc(&d_new_persistent, new_total * sizeof(Gaussian));
        cudaMalloc(&d_new_ages, new_total * sizeof(int));
    }

    // Scatter alive Gaussians into new buffer.
    if (num_alive > 0) {
        scatter_by_index_kernel<<<div_ceil(num_alive, kBlockSize), kBlockSize, 0, stream>>>(
            d_persistent_, d_ages_, d_alive_indices, num_alive,
            d_new_persistent, d_new_ages
        );
    }

    // Append unmatched new Gaussians after the alive ones.
    if (h_append_count > 0) {
        cudaMemcpyAsync(
            d_new_persistent + num_alive,
            d_append_gs,
            h_append_count * sizeof(Gaussian),
            cudaMemcpyDeviceToDevice, stream
        );
        cudaMemcpyAsync(
            d_new_ages + num_alive,
            d_append_ages,
            h_append_count * sizeof(int),
            cudaMemcpyDeviceToDevice, stream
        );
    }

    // Step 9: Budget enforcement — if over max_gaussians, keep the highest-opacity ones.
    int final_count = new_total;
    if (needs_budget_trim && new_total > config_.max_gaussians) {
        // Sort by opacity descending, keep top max_gaussians.
        float* d_trim_opacities = nullptr;
        int* d_trim_indices = nullptr;
        cudaMalloc(&d_trim_opacities, new_total * sizeof(float));
        cudaMalloc(&d_trim_indices, new_total * sizeof(int));

        write_opacity_index_kernel<<<div_ceil(new_total, kBlockSize), kBlockSize, 0, stream>>>(
            d_new_persistent, d_trim_opacities, d_trim_indices, new_total
        );

        // Sort by opacity descending (negate to use ascending sort).
        // Thrust doesn't have a simple descending sort_by_key, so use greater<float>.
        thrust::device_ptr<float> trim_op_ptr(d_trim_opacities);
        thrust::device_ptr<int> trim_idx_ptr(d_trim_indices);
        thrust::sort_by_key(
            thrust::cuda::par.on(stream),
            trim_op_ptr, trim_op_ptr + new_total,
            trim_idx_ptr,
            thrust::greater<float>()
        );

        // Keep only the top max_gaussians.
        int keep = config_.max_gaussians;
        Gaussian* d_trimmed_gs = nullptr;
        int* d_trimmed_ages = nullptr;
        cudaMalloc(&d_trimmed_gs, keep * sizeof(Gaussian));
        cudaMalloc(&d_trimmed_ages, keep * sizeof(int));

        scatter_by_index_kernel<<<div_ceil(keep, kBlockSize), kBlockSize, 0, stream>>>(
            d_new_persistent, d_new_ages, d_trim_indices, keep,
            d_trimmed_gs, d_trimmed_ages
        );

        // Swap trimmed buffers in.
        cudaFree(d_new_persistent);
        cudaFree(d_new_ages);
        d_new_persistent = d_trimmed_gs;
        d_new_ages = d_trimmed_ages;
        final_count = keep;
        last_removed_ += (new_total - keep);

        cudaFree(d_trim_opacities);
        cudaFree(d_trim_indices);
    }

    // Step 10: Swap new buffers into persistent state.
    cudaFree(d_persistent_);
    cudaFree(d_ages_);
    d_persistent_ = d_new_persistent;
    d_ages_ = d_new_ages;
    num_persistent_ = final_count;

    // Step 11: Download result to host.
    GaussianFrame result;
    result.frame_id = inferred.frame_id;
    result.timestamp_ns = inferred.timestamp_ns;
    result.num_gaussians = num_persistent_;
    result.gaussians.resize(num_persistent_);
    result.is_keyframe = inferred.is_keyframe;

    if (num_persistent_ > 0) {
        cudaMemcpyAsync(result.gaussians.data(), d_persistent_,
                         num_persistent_ * sizeof(Gaussian), cudaMemcpyDeviceToHost, stream);
    }

    // Synchronize before returning — the host buffer must be valid.
    cudaStreamSynchronize(stream);

    // Cleanup per-frame allocations.
    if (d_inferred) cudaFree(d_inferred);
    if (d_match_indices) cudaFree(d_match_indices);
    if (d_compact_indices) cudaFree(d_compact_indices);
    if (d_opacities) cudaFree(d_opacities);
    if (d_alive_indices) cudaFree(d_alive_indices);
    if (d_append_count) cudaFree(d_append_count);
    if (d_append_gs) cudaFree(d_append_gs);
    if (d_append_ages) cudaFree(d_append_ages);

    cudaStreamDestroy(stream);

    return result;
}

int TemporalTracker::persistent_count() const { return num_persistent_; }
int TemporalTracker::matched_count() const { return last_matched_; }
int TemporalTracker::added_count() const { return last_added_; }
int TemporalTracker::removed_count() const { return last_removed_; }

} // namespace heimdall::gaussian
