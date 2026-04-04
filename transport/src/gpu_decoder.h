#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

namespace heimdall::transport {

// NVDEC-based GPU decoder for H.265 streams received via SRT.
// Decodes directly into CUDA device memory — no CPU copy.
//
// On a 4-GPU RunPod instance, GPU 0 runs NVDEC for all 20 cameras.
// Decoded frames are shared with other GPUs via CUDA IPC.
//
// NVDEC supports up to 32 concurrent decode sessions on modern GPUs
// (Ada Lovelace / Hopper), so 20 cameras is well within limits.

struct DecoderConfig {
    int gpu_device = 0;
    int max_concurrent_sessions = 20;
    int output_width = 0;           // 0 = native resolution
    int output_height = 0;
    bool output_nv12 = false;       // false = RGBA F32, true = NV12
};

struct DecodedFrame {
    int camera_index;
    int64_t frame_id;
    int64_t timestamp_ns;
    void* gpu_ptr;                  // CUDA device pointer (RGBA F32 or NV12)
    int width;
    int height;
    int pitch;                      // bytes per row
    cudaStream_t stream;            // CUDA stream this decode was submitted on
};

using DecodedFrameCallback = std::function<void(const DecodedFrame&)>;

class GpuDecoder {
public:
    explicit GpuDecoder(const DecoderConfig& config);
    ~GpuDecoder();

    // Create a decode session for a camera.
    // Returns session ID (used to submit NAL units).
    int create_session(int camera_index, int width, int height);

    // Submit encoded NAL units for decoding.
    // Non-blocking — decoded frames arrive via callback.
    void submit(int session_id, const uint8_t* nal_data, size_t nal_size,
                int64_t frame_id, int64_t timestamp_ns);

    // Register callback for decoded frames.
    void set_callback(DecodedFrameCallback cb);

    // Destroy a decode session.
    void destroy_session(int session_id);

    // Stats
    int active_sessions() const;
    double decode_fps() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace heimdall::transport
