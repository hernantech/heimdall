#pragma once

#include "pipeline.h"
#include <cuda_runtime.h>

namespace heimdall::gaussian {

// Temporal tracker for Gaussian splats.
//
// Problem: frame-by-frame feed-forward inference produces "boiling" —
// splats pop in/out and jitter spatially between frames.
//
// Solution: maintain a persistent Gaussian buffer. Each frame:
//   1. Infer new Gaussians from feed-forward model
//   2. Match new Gaussians to previous frame via nearest-neighbor (position + color)
//   3. Matched: EMA blend attributes with configurable decay
//   4. Unmatched new: add with ramped-up opacity over N frames
//   5. Unmatched old: decay opacity over N frames, then remove
//
// Appearance (SH coefficients) is blended slowly; position/scale can move freely.
// This trades ~3ms of CUDA compute for dramatically smoother output.

struct TrackerConfig {
    float ema_decay = 0.85f;                // blend weight: result = decay * old + (1-decay) * new
    float match_distance_threshold = 0.02f; // meters; max distance for nearest-neighbor match
    float match_color_weight = 0.3f;        // weight of color similarity in matching cost
    int ramp_in_frames = 3;                 // new Gaussians fade in over N frames
    int ramp_out_frames = 3;                // unmatched old Gaussians fade out over N frames
    float min_opacity_threshold = 0.01f;    // remove Gaussians below this opacity
    int max_gaussians = 500000;             // hard cap on persistent buffer size
};

class TemporalTracker {
public:
    explicit TemporalTracker(const TrackerConfig& config);
    ~TemporalTracker();

    // Process a new frame of inferred Gaussians.
    // Returns the temporally-smoothed Gaussian frame.
    GaussianFrame process(const GaussianFrame& inferred);

    // Reset persistent state (e.g., on scene change).
    void reset();

    // Stats
    int persistent_count() const;   // current number of tracked Gaussians
    int matched_count() const;      // how many matched in last frame
    int added_count() const;        // how many new in last frame
    int removed_count() const;      // how many removed in last frame

private:
    // CUDA kernels
    void match_gaussians(
        const Gaussian* new_gaussians, int num_new,
        const Gaussian* old_gaussians, int num_old,
        int* match_indices,           // output: old index per new, or -1
        cudaStream_t stream
    );

    void blend_matched(
        Gaussian* persistent,
        const Gaussian* inferred,
        const int* match_indices,
        int num_inferred,
        float decay,
        cudaStream_t stream
    );

    void update_ramps(
        Gaussian* persistent,
        int* ages,                    // frames since added (positive) or unmatched (negative)
        int num_persistent,
        cudaStream_t stream
    );

    TrackerConfig config_;

    // Persistent GPU buffers
    Gaussian* d_persistent_ = nullptr;
    int* d_ages_ = nullptr;
    int num_persistent_ = 0;

    // Per-frame stats
    int last_matched_ = 0;
    int last_added_ = 0;
    int last_removed_ = 0;
};

} // namespace heimdall::gaussian
