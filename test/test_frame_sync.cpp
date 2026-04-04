// ============================================================================
// Tests for heimdall::capture::FrameSync
// ============================================================================

#include "test_helpers.h"
#include "../capture/src/frame_sync.h"

#include <algorithm>
#include <memory>
#include <vector>

using namespace heimdall::capture;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static std::shared_ptr<FrameData> make_frame(int64_t frame_id, int camera_index) {
    auto f = std::make_shared<FrameData>();
    f->frame_id     = frame_id;
    f->timestamp_ns  = frame_id * 33'000'000; // ~30fps
    f->camera_index  = camera_index;
    f->gpu_ptr       = nullptr;
    f->width         = 1920;
    f->height        = 1080;
    f->channels      = 4;
    return f;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

TEST(frame_sync_all_cameras_fire_callback) {
    // Push frames from 4 cameras for the same frame_id.
    // The callback should fire exactly once with all 4 frames.

    constexpr int kNumCameras = 4;
    FrameSync sync(kNumCameras);

    int callback_count = 0;
    SyncedFrameSet received;

    sync.set_callback([&](const SyncedFrameSet& frames) {
        callback_count++;
        received = frames;
    });

    for (int cam = 0; cam < kNumCameras; ++cam) {
        sync.push(make_frame(/*frame_id=*/1, cam));
    }

    ASSERT_EQ(callback_count, 1);
    ASSERT_EQ(static_cast<int>(received.size()), kNumCameras);

    // Verify all camera indices are present.
    std::vector<int> cam_indices;
    for (auto& f : received) {
        cam_indices.push_back(f->camera_index);
    }
    std::sort(cam_indices.begin(), cam_indices.end());
    for (int i = 0; i < kNumCameras; ++i) {
        ASSERT_EQ(cam_indices[i], i);
    }

    // After completion, the pending set for frame 1 should be removed.
    ASSERT_EQ(sync.pending_count(), 0);
}

TEST(frame_sync_out_of_order_arrival) {
    // Frames arrive in non-sequential camera order.
    // Sync should still fire correctly when all cameras are present.

    constexpr int kNumCameras = 4;
    FrameSync sync(kNumCameras);

    int callback_count = 0;
    int64_t completed_frame_id = -1;

    sync.set_callback([&](const SyncedFrameSet& frames) {
        callback_count++;
        completed_frame_id = frames[0]->frame_id;
    });

    // Push cameras in reverse order.
    sync.push(make_frame(10, 3));
    sync.push(make_frame(10, 1));
    sync.push(make_frame(10, 0));
    ASSERT_EQ(callback_count, 0);  // not yet complete
    sync.push(make_frame(10, 2));
    ASSERT_EQ(callback_count, 1);
    ASSERT_EQ(completed_frame_id, 10);
}

TEST(frame_sync_multiple_frame_ids) {
    // Interleave frames from two different frame_ids.
    // Each should fire its own callback when complete.

    constexpr int kNumCameras = 2;
    FrameSync sync(kNumCameras);

    std::vector<int64_t> completed_ids;

    sync.set_callback([&](const SyncedFrameSet& frames) {
        completed_ids.push_back(frames[0]->frame_id);
    });

    // frame_id=1 camera 0
    sync.push(make_frame(1, 0));
    // frame_id=2 camera 0
    sync.push(make_frame(2, 0));
    // frame_id=2 camera 1 -- completes frame 2 first
    sync.push(make_frame(2, 1));

    ASSERT_EQ(static_cast<int>(completed_ids.size()), 1);
    ASSERT_EQ(completed_ids[0], 2);

    // frame_id=1 camera 1 -- completes frame 1
    sync.push(make_frame(1, 1));
    ASSERT_EQ(static_cast<int>(completed_ids.size()), 2);
    ASSERT_EQ(completed_ids[1], 1);
}

TEST(frame_sync_eviction_of_old_frames) {
    // max_pending=3, so when a 4th unique frame_id is pushed,
    // the oldest incomplete frame is evicted.

    constexpr int kNumCameras = 4;
    constexpr int kMaxPending = 3;
    FrameSync sync(kNumCameras, kMaxPending);

    sync.set_callback([](const SyncedFrameSet&) {
        // no-op — we don't expect completion in this test
    });

    // Push one frame from camera 0 for frame_ids 1, 2, 3 (each incomplete).
    sync.push(make_frame(1, 0));
    sync.push(make_frame(2, 0));
    sync.push(make_frame(3, 0));
    ASSERT_EQ(sync.pending_count(), 3);
    ASSERT_EQ(sync.dropped_count(), 0);

    // Push frame_id=4 — this should evict frame_id=1.
    sync.push(make_frame(4, 0));
    ASSERT_LE(sync.pending_count(), kMaxPending);
    ASSERT_GT(sync.dropped_count(), 0);
}

TEST(frame_sync_dropped_frame_count_tracking) {
    // Verify that dropped_count accurately reflects evicted frames.

    constexpr int kNumCameras = 3;
    constexpr int kMaxPending = 2;
    FrameSync sync(kNumCameras, kMaxPending);

    sync.set_callback([](const SyncedFrameSet&) {});

    // Push 2 frames for frame_id=1 (incomplete, needs 3).
    sync.push(make_frame(1, 0));
    sync.push(make_frame(1, 1));
    ASSERT_EQ(sync.pending_count(), 1);

    // Push frame_id=2 camera 0.
    sync.push(make_frame(2, 0));
    ASSERT_EQ(sync.pending_count(), 2);

    // Push frame_id=3 camera 0 — evicts frame_id=1 (had 2 frames).
    sync.push(make_frame(3, 0));
    ASSERT_LE(sync.pending_count(), kMaxPending);

    // frame_id=1 had 2 frames that were dropped.
    ASSERT_EQ(sync.dropped_count(), 2);
}

TEST(frame_sync_no_callback_if_not_set) {
    // Pushing a complete set without a callback should not crash.

    constexpr int kNumCameras = 2;
    FrameSync sync(kNumCameras);

    // No callback set — just push frames.
    sync.push(make_frame(1, 0));
    sync.push(make_frame(1, 1));

    // The frame set should be completed and cleaned up.
    ASSERT_EQ(sync.pending_count(), 0);
}

TEST(frame_sync_callback_not_fired_for_partial) {
    // If only some cameras deliver, no callback fires.

    constexpr int kNumCameras = 4;
    FrameSync sync(kNumCameras);

    int callback_count = 0;
    sync.set_callback([&](const SyncedFrameSet&) {
        callback_count++;
    });

    sync.push(make_frame(5, 0));
    sync.push(make_frame(5, 1));
    sync.push(make_frame(5, 2));
    // Only 3 of 4 — callback should NOT fire.
    ASSERT_EQ(callback_count, 0);
    ASSERT_EQ(sync.pending_count(), 1);
}
