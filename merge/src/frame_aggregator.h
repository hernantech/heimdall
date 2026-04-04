#pragma once

#include "../../gaussian/src/pipeline.h"
#include <chrono>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace heimdall::merge {

// ---------------------------------------------------------------------------
// PartialResult: one worker's contribution for a single frame.
//
// Workers run feed-forward Gaussian inference on a subset of cameras and send
// their partial GaussianFrame to the merge machine over the network. Positions
// may be quantized for bandwidth; the merge pipeline dequantizes on arrival.
// ---------------------------------------------------------------------------
struct PartialResult {
    int64_t frame_id;
    int64_t timestamp_ns;
    int worker_id;                              // 0..N-1
    gaussian::GaussianFrame gaussians;
    bool is_quantized = false;                  // positions/rotations compressed for transit
    std::chrono::steady_clock::time_point received_at;
};

// ---------------------------------------------------------------------------
// AggregatedFrame: collected partials for one frame_id, ready for merging.
// ---------------------------------------------------------------------------
struct AggregatedFrame {
    int64_t frame_id;
    int64_t timestamp_ns;
    std::vector<PartialResult> partials;        // one per contributing worker
    int expected_workers;                       // N
    bool quorum_met;                            // true if K-of-N arrived
    bool timed_out;                             // true if we proceeded on timeout
    std::chrono::steady_clock::time_point first_arrival;
    std::chrono::steady_clock::time_point release_time;
};

// ---------------------------------------------------------------------------
// FrameAggregatorConfig
// ---------------------------------------------------------------------------
struct FrameAggregatorConfig {
    int num_workers = 4;                        // N — total expected workers
    int quorum = 3;                             // K — proceed after this many arrive

    // Maximum time to wait for stragglers after first partial arrives.
    std::chrono::milliseconds timeout{200};

    // Maximum number of in-flight frames buffered simultaneously.
    // Older incomplete frames are force-released if this is exceeded.
    int max_in_flight = 4;

    // If a worker misses this many consecutive frames, mark it dead
    // and lower the effective quorum until it comes back.
    int worker_dead_threshold = 10;
};

// Callback fired when a frame is released (quorum met or timeout).
using FrameReadyCallback = std::function<void(AggregatedFrame&&)>;

// ---------------------------------------------------------------------------
// FrameAggregator
//
// Collects partial Gaussian results from N workers per frame_id. A frame is
// released to the merge pipeline when either:
//   (a) K-of-N partials have arrived (quorum), or
//   (b) 200ms have elapsed since the first partial for that frame_id.
//
// This allows graceful degradation: if a worker is slow or dead the pipeline
// proceeds with whatever data is available, rather than stalling.
//
// Thread safety: push_partial may be called concurrently from multiple
// network receive threads. The ready callback fires on an internal timer
// thread (not a caller thread).
// ---------------------------------------------------------------------------
class FrameAggregator {
public:
    explicit FrameAggregator(const FrameAggregatorConfig& config);
    ~FrameAggregator();

    // Register the callback invoked when a frame is ready.
    void set_ready_callback(FrameReadyCallback cb);

    // Start the internal timeout-check thread.
    void start();

    // Stop accepting partials and drain pending frames.
    void stop();

    // Called from network receive threads when a worker's result arrives.
    // Thread-safe.
    void push_partial(PartialResult partial);

    // Mark a worker as alive (e.g. on heartbeat). Restores quorum if it
    // was previously considered dead.
    void worker_heartbeat(int worker_id);

    // --- stats ---------------------------------------------------------------
    int frames_released() const;
    int frames_timed_out() const;
    int active_workers() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace heimdall::merge
