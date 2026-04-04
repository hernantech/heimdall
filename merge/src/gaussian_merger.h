#pragma once

#include "frame_aggregator.h"
#include "../../gaussian/src/pipeline.h"
#include "../../gaussian/src/spatial_hash.h"
#include <cuda_runtime.h>
#include <cstdint>
#include <memory>
#include <vector>

namespace heimdall::merge {

// ---------------------------------------------------------------------------
// GaussianMergerConfig
// ---------------------------------------------------------------------------
struct GaussianMergerConfig {
    // Spatial de-duplication threshold (meters). Gaussians from DIFFERENT
    // workers that are within this distance are considered overlapping and
    // merged via weighted average.
    float dedup_distance = 0.015f;

    // Minimum opacity after merge to keep a Gaussian (prunes ghosts).
    float min_opacity = 0.01f;

    // Maximum number of Gaussians in the merged output. Excess are pruned
    // by lowest opacity first.
    int max_output_gaussians = 500000;

    // Weight strategy for overlapping Gaussians.
    //   0 = simple average
    //   1 = opacity-weighted average
    //   2 = confidence-weighted (uses per-worker confidence if available)
    int weight_mode = 1;
};

// ---------------------------------------------------------------------------
// MergedFrame: the result of spatial de-duplication across workers.
// ---------------------------------------------------------------------------
struct MergedFrame {
    int64_t frame_id;
    int64_t timestamp_ns;
    int num_gaussians;

    // Device pointer to the merged Gaussians (owned by GaussianMerger,
    // valid until the next call to merge()).
    gaussian::Gaussian* d_gaussians = nullptr;

    // Stats from the merge pass.
    int total_input_gaussians = 0;   // sum of all partials
    int duplicates_removed = 0;
};

// ---------------------------------------------------------------------------
// GaussianMerger (CUDA)
//
// Receives partial Gaussian frames from multiple workers, uploads them to the
// GPU, uses a spatial hash grid to find overlapping Gaussians from DIFFERENT
// workers, and performs a weighted merge + compaction.
//
// Algorithm:
//   1. Concatenate all partials into a single GPU buffer, tagging each
//      Gaussian with its source worker_id.
//   2. Build spatial hash grid over the concatenated buffer.
//   3. For each Gaussian, probe neighboring cells. If an overlapping
//      Gaussian from a DIFFERENT worker is found within dedup_distance,
//      mark the lower-opacity one for removal and accumulate its
//      attributes into the survivor (weighted average).
//   4. Compact: stream-compact surviving Gaussians into a dense output
//      buffer.
//
// The spatial hash grid reuses heimdall::gaussian::SpatialHashGrid from
// gaussian/src/spatial_hash.h.
//
// Thread safety: NOT thread-safe. Called from the merge pipeline's single
// processing thread.
// ---------------------------------------------------------------------------
class GaussianMerger {
public:
    explicit GaussianMerger(const GaussianMergerConfig& config);
    ~GaussianMerger();

    // Merge partials from an aggregated frame.
    // Returns a MergedFrame with d_gaussians pointing to device memory
    // owned by this merger (valid until the next merge() call).
    MergedFrame merge(const AggregatedFrame& aggregated);

    // Free all GPU resources.
    void reset();

    // Stats
    int last_input_count() const;
    int last_output_count() const;
    int last_duplicates_removed() const;

private:
    // Upload all partials into a contiguous GPU buffer with worker tags.
    void upload_partials(const AggregatedFrame& aggregated);

    // Build spatial hash grid over the concatenated Gaussians.
    void build_hash_grid();

    // Run the dedup + weighted merge kernel.
    void dedup_and_merge();

    // Compact surviving Gaussians into the output buffer.
    void compact_output();

    GaussianMergerConfig config_;
    cudaStream_t stream_ = nullptr;

    // --- GPU buffers ---

    // Concatenated input buffer (all partials).
    gaussian::Gaussian* d_input_ = nullptr;
    int* d_worker_ids_ = nullptr;     // worker_id per Gaussian
    int input_capacity_ = 0;
    int input_count_ = 0;

    // Spatial hash storage (reused across frames).
    gaussian::SpatialHashStorage hash_storage_;

    // Per-Gaussian flags: 0 = keep, 1 = remove (duplicate).
    int* d_remove_flags_ = nullptr;

    // Accumulated merge weights (for weighted average).
    float* d_merge_weights_ = nullptr;

    // Compacted output buffer.
    gaussian::Gaussian* d_output_ = nullptr;
    int output_capacity_ = 0;
    int output_count_ = 0;

    // Prefix-sum scratch for stream compaction.
    int* d_compact_indices_ = nullptr;
    int* d_compact_count_ = nullptr;  // single int on device

    // Stats
    int stat_last_input_ = 0;
    int stat_last_output_ = 0;
    int stat_last_dupes_ = 0;
};

} // namespace heimdall::merge
