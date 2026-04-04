#include "frame_aggregator.h"

#include <algorithm>
#include <atomic>
#include <cassert>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <vector>

namespace heimdall::merge {

struct PendingFrame {
    int64_t frame_id = -1;
    int64_t timestamp_ns = 0;
    std::vector<PartialResult> partials;
    int expected_workers = 0;
    std::chrono::steady_clock::time_point first_arrival;
    bool released = false;
};

struct FrameAggregator::Impl {
    FrameAggregatorConfig config;
    FrameReadyCallback ready_cb;

    mutable std::mutex mu;
    std::condition_variable cv;

    std::unordered_map<int64_t, PendingFrame> pending;
    std::vector<int> worker_miss_count;
    std::vector<bool> worker_alive;
    std::vector<int64_t> frame_order;

    std::atomic<bool> running{false};
    std::thread timer_thread;

    std::atomic<int> stat_released{0};
    std::atomic<int> stat_timed_out{0};

    int effective_quorum() const {
        int alive = 0;
        for (bool a : worker_alive) if (a) ++alive;
        return std::min(config.quorum, alive);
    }

    void release_frame(PendingFrame& pf, bool timed_out,
                       std::unique_lock<std::mutex>& lock) {
        if (pf.released) return;
        pf.released = true;

        AggregatedFrame af;
        af.frame_id = pf.frame_id;
        af.timestamp_ns = pf.timestamp_ns;
        af.partials = std::move(pf.partials);
        af.expected_workers = pf.expected_workers;
        af.quorum_met =
            static_cast<int>(af.partials.size()) >= effective_quorum();
        af.timed_out = timed_out;
        af.first_arrival = pf.first_arrival;
        af.release_time = std::chrono::steady_clock::now();

        stat_released.fetch_add(1, std::memory_order_relaxed);
        if (timed_out)
            stat_timed_out.fetch_add(1, std::memory_order_relaxed);

        update_worker_health(af);

        int64_t fid = pf.frame_id;
        pending.erase(fid);
        frame_order.erase(
            std::remove(frame_order.begin(), frame_order.end(), fid),
            frame_order.end());

        // Fire callback outside the lock to avoid deadlocks.
        lock.unlock();
        if (ready_cb) ready_cb(std::move(af));
        lock.lock();
    }

    void check_timeouts() {
        std::unique_lock<std::mutex> lock(mu);
        auto now = std::chrono::steady_clock::now();
        auto order_copy = frame_order;
        for (int64_t fid : order_copy) {
            auto it = pending.find(fid);
            if (it == pending.end()) continue;
            auto& pf = it->second;
            if (pf.released) continue;
            auto elapsed =
                std::chrono::duration_cast<std::chrono::milliseconds>(
                    now - pf.first_arrival);
            if (elapsed >= config.timeout)
                release_frame(pf, true, lock);
        }
        // Enforce max_in_flight: force-release oldest.
        while (static_cast<int>(frame_order.size()) > config.max_in_flight) {
            int64_t oldest = frame_order.front();
            auto it = pending.find(oldest);
            if (it != pending.end() && !it->second.released)
                release_frame(it->second, true, lock);
            else
                frame_order.erase(frame_order.begin());
        }
    }

    void timer_loop() {
        while (running.load(std::memory_order_acquire)) {
            {
                std::unique_lock<std::mutex> lock(mu);
                cv.wait_for(lock, std::chrono::milliseconds(10), [this] {
                    return !running.load(std::memory_order_relaxed);
                });
            }
            if (!running.load(std::memory_order_acquire)) break;
            check_timeouts();
        }
    }

    void update_worker_health(const AggregatedFrame& af) {
        std::vector<bool> contributed(config.num_workers, false);
        for (const auto& p : af.partials) {
            if (p.worker_id >= 0 && p.worker_id < config.num_workers)
                contributed[p.worker_id] = true;
        }
        for (int w = 0; w < config.num_workers; ++w) {
            if (contributed[w]) {
                worker_miss_count[w] = 0;
                worker_alive[w] = true;
            } else {
                worker_miss_count[w]++;
                if (worker_miss_count[w] >= config.worker_dead_threshold)
                    worker_alive[w] = false;
            }
        }
    }
};

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

FrameAggregator::FrameAggregator(const FrameAggregatorConfig& config)
    : impl_(std::make_unique<Impl>()) {
    impl_->config = config;
    impl_->worker_miss_count.resize(config.num_workers, 0);
    impl_->worker_alive.resize(config.num_workers, true);
}

FrameAggregator::~FrameAggregator() { stop(); }

void FrameAggregator::set_ready_callback(FrameReadyCallback cb) {
    std::lock_guard<std::mutex> lock(impl_->mu);
    impl_->ready_cb = std::move(cb);
}

void FrameAggregator::start() {
    bool expected = false;
    if (!impl_->running.compare_exchange_strong(expected, true)) return;
    impl_->timer_thread = std::thread([this] { impl_->timer_loop(); });
}

void FrameAggregator::stop() {
    bool expected = true;
    if (!impl_->running.compare_exchange_strong(expected, false)) return;
    impl_->cv.notify_all();
    if (impl_->timer_thread.joinable()) impl_->timer_thread.join();

    std::unique_lock<std::mutex> lock(impl_->mu);
    auto order_copy = impl_->frame_order;
    for (int64_t fid : order_copy) {
        auto it = impl_->pending.find(fid);
        if (it != impl_->pending.end() && !it->second.released)
            impl_->release_frame(it->second, true, lock);
    }
}

void FrameAggregator::push_partial(PartialResult partial) {
    if (!impl_->running.load(std::memory_order_acquire)) return;
    std::unique_lock<std::mutex> lock(impl_->mu);

    int64_t fid = partial.frame_id;
    auto it = impl_->pending.find(fid);

    if (it == impl_->pending.end()) {
        PendingFrame pf;
        pf.frame_id = fid;
        pf.timestamp_ns = partial.timestamp_ns;
        pf.expected_workers = impl_->config.num_workers;
        pf.first_arrival = std::chrono::steady_clock::now();
        pf.partials.push_back(std::move(partial));
        impl_->pending.emplace(fid, std::move(pf));
        impl_->frame_order.push_back(fid);
        auto& inserted = impl_->pending[fid];
        if (static_cast<int>(inserted.partials.size()) >=
            impl_->effective_quorum())
            impl_->release_frame(inserted, false, lock);
    } else {
        auto& pf = it->second;
        if (pf.released) return;
        for (const auto& existing : pf.partials)
            if (existing.worker_id == partial.worker_id) return;
        pf.partials.push_back(std::move(partial));
        if (static_cast<int>(pf.partials.size()) >=
            impl_->effective_quorum())
            impl_->release_frame(pf, false, lock);
    }
    impl_->cv.notify_one();
}

void FrameAggregator::worker_heartbeat(int worker_id) {
    std::lock_guard<std::mutex> lock(impl_->mu);
    if (worker_id >= 0 && worker_id < impl_->config.num_workers) {
        impl_->worker_miss_count[worker_id] = 0;
        impl_->worker_alive[worker_id] = true;
    }
}

int FrameAggregator::frames_released() const {
    return impl_->stat_released.load(std::memory_order_relaxed);
}

int FrameAggregator::frames_timed_out() const {
    return impl_->stat_timed_out.load(std::memory_order_relaxed);
}

int FrameAggregator::active_workers() const {
    std::lock_guard<std::mutex> lock(impl_->mu);
    int alive = 0;
    for (bool a : impl_->worker_alive) if (a) ++alive;
    return alive;
}

} // namespace heimdall::merge
