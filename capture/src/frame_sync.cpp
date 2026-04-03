#include "frame_sync.h"

namespace heimdall::capture {

FrameSync::FrameSync(int num_cameras, int max_pending)
    : num_cameras_(num_cameras), max_pending_(max_pending) {}

void FrameSync::set_callback(SyncCallback cb) {
    std::lock_guard<std::mutex> lock(mu_);
    callback_ = std::move(cb);
}

void FrameSync::push(std::shared_ptr<FrameData> frame) {
    std::lock_guard<std::mutex> lock(mu_);

    pending_[frame->frame_id].push_back(std::move(frame));
    try_complete(pending_.rbegin()->first);

    while (static_cast<int>(pending_.size()) > max_pending_) {
        evict_oldest();
    }
}

void FrameSync::try_complete(int64_t frame_id) {
    auto it = pending_.find(frame_id);
    if (it == pending_.end()) return;

    if (static_cast<int>(it->second.size()) == num_cameras_) {
        if (callback_) {
            callback_(it->second);
        }
        pending_.erase(it);
    }
}

void FrameSync::evict_oldest() {
    if (pending_.empty()) return;
    dropped_ += static_cast<int>(pending_.begin()->second.size());
    pending_.erase(pending_.begin());
}

int FrameSync::pending_count() const {
    std::lock_guard<std::mutex> lock(mu_);
    return static_cast<int>(pending_.size());
}

int FrameSync::dropped_count() const {
    return dropped_;
}

} // namespace heimdall::capture
