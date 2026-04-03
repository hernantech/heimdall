#pragma once

#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <vector>

namespace heimdall::capture {

struct FrameData {
    int64_t frame_id;
    int64_t timestamp_ns;
    int camera_index;
    void* gpu_ptr;          // CUDA device pointer to RGBA F32
    int width;
    int height;
    int channels;
};

using SyncedFrameSet = std::vector<std::shared_ptr<FrameData>>;
using SyncCallback = std::function<void(const SyncedFrameSet&)>;

// Synchronizes frames from N cameras by frame_id.
// When all cameras have delivered a frame for the same ID,
// fires the callback with the complete set.
class FrameSync {
public:
    explicit FrameSync(int num_cameras, int max_pending = 25);

    void push(std::shared_ptr<FrameData> frame);
    void set_callback(SyncCallback cb);

    int pending_count() const;
    int dropped_count() const;

private:
    void try_complete(int64_t frame_id);
    void evict_oldest();

    int num_cameras_;
    int max_pending_;
    int dropped_ = 0;

    mutable std::mutex mu_;
    SyncCallback callback_;
    std::map<int64_t, std::vector<std::shared_ptr<FrameData>>> pending_;
};

} // namespace heimdall::capture
