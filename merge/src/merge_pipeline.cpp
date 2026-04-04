#include "merge_pipeline.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <queue>
#include <thread>
#include <cuda_runtime.h>

namespace heimdall::merge {

struct MergePipeline::Impl {
    MergePipelineConfig config;

    // Components
    FrameAggregator aggregator;
    GaussianMerger merger;
    gaussian::TemporalTracker tracker;
    stream::GaussianStreamMultiplexer mux;

    // Processing thread
    std::thread proc_thread;
    std::atomic<bool> running{false};
    std::mutex queue_mu;
    std::condition_variable queue_cv;
    std::queue<AggregatedFrame> frame_queue;

    // SPZ state
    gaussian::GaussianFrame last_keyframe;
    int64_t frames_since_keyframe = 0;

    // Stats
    std::atomic<int64_t> stat_processed{0};
    std::atomic<int64_t> stat_dropped{0};
    double avg_merge_ms = 0.0;
    double avg_track_ms = 0.0;
    double avg_compress_ms = 0.0;
    double avg_total_ms = 0.0;
    mutable std::mutex stat_mu;

    Impl(const MergePipelineConfig& cfg)
        : config(cfg)
        , aggregator(cfg.aggregator)
        , merger(cfg.merger)
        , tracker(cfg.tracker)
    {
        mux.configure(cfg.mux);
    }

    // Dequantize partials if needed (16-bit positions -> float, etc.)
    void dequantize(AggregatedFrame& af) {
        if (!config.dequantize_partials) return;
        for (auto& partial : af.partials) {
            if (!partial.is_quantized) continue;
            // In a full implementation, this would decode 16-bit quantized
            // positions and 8-bit quaternions back to float32.
            // For now, partials arrive as float32 from workers on the same LAN.
            partial.is_quantized = false;
        }
    }

    // Download merged Gaussians from GPU to a host GaussianFrame.
    gaussian::GaussianFrame download_merged(const MergedFrame& mf) {
        gaussian::GaussianFrame gf;
        gf.frame_id = mf.frame_id;
        gf.timestamp_ns = mf.timestamp_ns;
        gf.num_gaussians = mf.num_gaussians;
        gf.gaussians.resize(mf.num_gaussians);
        if (mf.num_gaussians > 0 && mf.d_gaussians) {
            cudaMemcpy(gf.gaussians.data(), mf.d_gaussians,
                       mf.num_gaussians * sizeof(gaussian::Gaussian),
                       cudaMemcpyDeviceToHost);
        }
        gf.is_keyframe = (frames_since_keyframe == 0);
        return gf;
    }

    // Process one aggregated frame through the full pipeline.
    void process_frame(AggregatedFrame&& af) {
        auto t0 = std::chrono::steady_clock::now();

        // 1. Dequantize
        dequantize(af);

        // 2. GPU merge (spatial dedup across workers)
        auto t_merge_start = std::chrono::steady_clock::now();
        MergedFrame merged = merger.merge(af);
        auto t_merge_end = std::chrono::steady_clock::now();

        // 3. Download to host and run temporal tracker
        auto t_track_start = std::chrono::steady_clock::now();
        gaussian::GaussianFrame host_frame = download_merged(merged);
        gaussian::GaussianFrame tracked = tracker.process(host_frame);
        auto t_track_end = std::chrono::steady_clock::now();

        // 4. SPZ compress
        auto t_compress_start = std::chrono::steady_clock::now();
        bool is_keyframe = (frames_since_keyframe >= config.spz.keyframe_interval)
                           || frames_since_keyframe == 0;
        tracked.is_keyframe = is_keyframe;

        const gaussian::GaussianFrame* prev_kf =
            is_keyframe ? nullptr : &last_keyframe;
        gaussian::SpzChunk chunk = gaussian::encode_frame(
            tracked, prev_kf, config.spz);
        auto t_compress_end = std::chrono::steady_clock::now();

        if (is_keyframe) {
            last_keyframe = tracked;
            frames_since_keyframe = 0;
        }
        frames_since_keyframe++;

        // 5. Push to multiplexer for streaming
        uint32_t ts_ms = static_cast<uint32_t>(
            tracked.timestamp_ns / 1000000);
        mux.push_gaussian_frame(chunk, ts_ms);

        // Update stats
        auto t1 = std::chrono::steady_clock::now();
        double merge_ms = std::chrono::duration<double,std::milli>(t_merge_end-t_merge_start).count();
        double track_ms = std::chrono::duration<double,std::milli>(t_track_end-t_track_start).count();
        double comp_ms = std::chrono::duration<double,std::milli>(t_compress_end-t_compress_start).count();
        double total_ms = std::chrono::duration<double,std::milli>(t1-t0).count();

        constexpr double alpha = 0.1;  // EMA smoothing
        {
            std::lock_guard<std::mutex> lock(stat_mu);
            avg_merge_ms = avg_merge_ms*(1-alpha) + merge_ms*alpha;
            avg_track_ms = avg_track_ms*(1-alpha) + track_ms*alpha;
            avg_compress_ms = avg_compress_ms*(1-alpha) + comp_ms*alpha;
            avg_total_ms = avg_total_ms*(1-alpha) + total_ms*alpha;
        }
        stat_processed.fetch_add(1, std::memory_order_relaxed);
    }

    // Processing thread loop.
    void proc_loop() {
        while (running.load(std::memory_order_acquire)) {
            AggregatedFrame af;
            {
                std::unique_lock<std::mutex> lock(queue_mu);
                queue_cv.wait(lock, [this] {
                    return !frame_queue.empty() || !running.load(std::memory_order_relaxed);
                });
                if (!running.load(std::memory_order_relaxed) && frame_queue.empty())
                    break;
                if (frame_queue.empty()) continue;

                // If we are behind, skip older frames to stay within latency budget.
                while (frame_queue.size() > 1) {
                    frame_queue.pop();
                    stat_dropped.fetch_add(1, std::memory_order_relaxed);
                }
                af = std::move(frame_queue.front());
                frame_queue.pop();
            }
            process_frame(std::move(af));
        }
    }
};

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

MergePipeline::MergePipeline(const MergePipelineConfig& config)
    : impl_(std::make_unique<Impl>(config)) {}

MergePipeline::~MergePipeline() { stop(); }

void MergePipeline::start() {
    bool expected = false;
    if (!impl_->running.compare_exchange_strong(expected, true)) return;

    // Wire aggregator callback to enqueue frames for processing.
    impl_->aggregator.set_ready_callback([this](AggregatedFrame&& af) {
        std::lock_guard<std::mutex> lock(impl_->queue_mu);
        impl_->frame_queue.push(std::move(af));
        impl_->queue_cv.notify_one();
    });

    impl_->aggregator.start();
    impl_->proc_thread = std::thread([this] { impl_->proc_loop(); });
}

void MergePipeline::stop() {
    bool expected = true;
    if (!impl_->running.compare_exchange_strong(expected, false)) return;
    impl_->aggregator.stop();
    impl_->queue_cv.notify_all();
    if (impl_->proc_thread.joinable()) impl_->proc_thread.join();
}

void MergePipeline::push_partial(PartialResult partial) {
    impl_->aggregator.push_partial(std::move(partial));
}

stream::GaussianStreamMultiplexer& MergePipeline::multiplexer() {
    return impl_->mux;
}

void MergePipeline::reset() {
    impl_->tracker.reset();
    impl_->merger.reset();
    impl_->frames_since_keyframe = 0;
}

MergePipelineStats MergePipeline::stats() const {
    MergePipelineStats s;
    s.frames_processed = impl_->stat_processed.load(std::memory_order_relaxed);
    s.frames_dropped = impl_->stat_dropped.load(std::memory_order_relaxed);
    {
        std::lock_guard<std::mutex> lock(impl_->stat_mu);
        s.avg_merge_ms = impl_->avg_merge_ms;
        s.avg_track_ms = impl_->avg_track_ms;
        s.avg_compress_ms = impl_->avg_compress_ms;
        s.avg_total_ms = impl_->avg_total_ms;
    }
    s.active_workers = impl_->aggregator.active_workers();
    s.persistent_gaussians = impl_->tracker.persistent_count();
    s.last_merge_input = impl_->merger.last_input_count();
    s.last_merge_output = impl_->merger.last_output_count();
    s.last_duplicates = impl_->merger.last_duplicates_removed();
    return s;
}

} // namespace heimdall::merge
