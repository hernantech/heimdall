#pragma once

#include "frame_aggregator.h"
#include "gaussian_merger.h"
#include "../../gaussian/src/pipeline.h"
#include "../../gaussian/src/temporal_tracker.h"
#include "../../gaussian/src/spz_writer.h"
#include "../../stream/src/gs_multiplexer.h"
#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace heimdall::merge {

// ---------------------------------------------------------------------------
// MergePipelineConfig
// ---------------------------------------------------------------------------
struct MergePipelineConfig {
    // Aggregator
    FrameAggregatorConfig aggregator;

    // Gaussian merger (CUDA dedup)
    GaussianMergerConfig merger;

    // Temporal tracker (runs on merge machine)
    gaussian::TrackerConfig tracker;

    // SPZ compression
    gaussian::SpzWriterConfig spz;

    // Streaming
    stream::MuxConfig mux;

    // If true, dequantize partials before merging (workers may send
    // quantized positions/rotations to save bandwidth).
    bool dequantize_partials = true;

    // Maximum latency target (ms). If pipeline is behind, skip frames.
    int max_latency_ms = 2000;
};

// ---------------------------------------------------------------------------
// MergePipelineStats
// ---------------------------------------------------------------------------
struct MergePipelineStats {
    int64_t frames_processed = 0;
    int64_t frames_dropped = 0;
    double avg_merge_ms = 0.0;      // GPU dedup time
    double avg_track_ms = 0.0;      // temporal tracker time
    double avg_compress_ms = 0.0;   // SPZ compression time
    double avg_total_ms = 0.0;      // end-to-end per frame
    int active_workers = 0;
    int persistent_gaussians = 0;
    int last_merge_input = 0;
    int last_merge_output = 0;
    int last_duplicates = 0;
};

// ---------------------------------------------------------------------------
// MergePipeline
//
// The SINGLE stateful component in the Heimdall volumetric capture pipeline.
// Runs on a dedicated merge machine (one GPU). Orchestrates:
//
//   1. Aggregate: collect partial Gaussian frames from N workers (quorum-based)
//   2. Dequantize: if workers sent quantized data, restore full precision
//   3. Merge: CUDA spatial hash dedup across workers, weighted average
//   4. Track: temporal tracker (persistent buffer + EMA smoothing)
//   5. Compress: SPZ keyframe/delta encoding
//   6. Stream: push to GaussianStreamMultiplexer for WebRTC delivery
//
// Graceful degradation: if a worker is slow or dead, the pipeline proceeds
// with whatever partials arrived (quorum or timeout).
//
// Thread model: single processing thread pulls from the aggregator callback
// and runs the GPU pipeline sequentially. Network threads push partials.
// ---------------------------------------------------------------------------
class MergePipeline {
public:
    explicit MergePipeline(const MergePipelineConfig& config);
    ~MergePipeline();

    // Start the pipeline (aggregator + processing thread).
    void start();

    // Stop gracefully, draining in-flight frames.
    void stop();

    // Push a partial result from a worker (thread-safe, called from network threads).
    void push_partial(PartialResult partial);

    // Access the multiplexer for WebRTC integration.
    stream::GaussianStreamMultiplexer& multiplexer();

    // Reset all persistent state (e.g., on scene change).
    void reset();

    // Stats
    MergePipelineStats stats() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace heimdall::merge
