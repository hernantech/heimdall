#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

namespace heimdall::gaussian {

// Gaussian splat attributes for a single splat.
struct Gaussian {
    float position[3];      // xyz world coordinates
    float scale[3];         // scale per axis
    float rotation[4];      // quaternion (wxyz)
    float opacity;          // alpha, 0-1
    float sh[48];           // spherical harmonics (degree 3 = 16 coeffs * 3 channels)
};

// A frame of Gaussians produced by the feed-forward model.
struct GaussianFrame {
    int64_t frame_id;
    int64_t timestamp_ns;
    int num_gaussians;
    std::vector<Gaussian> gaussians;
    bool is_keyframe;       // full inference vs delta update
};

// Input: a set of camera images for one timestep.
struct CameraInput {
    int camera_index;
    int serial_number;
    int width;
    int height;
    void* gpu_rgba_f32;     // CUDA device pointer
    void* gpu_alpha_u8;     // CUDA device pointer (matting result)
};

struct PipelineConfig {
    // Camera selection
    int max_cameras_per_inference = 6;

    // Feed-forward model
    std::string model_path;                 // path to TensorRT engine or ONNX model
    int inference_width = 512;
    int inference_height = 960;

    // Temporal tracker
    float temporal_ema_decay = 0.85f;       // blending weight for matched Gaussians
    float match_distance_threshold = 0.02f; // meters, for nearest-neighbor matching
    int ramp_in_frames = 3;                 // new Gaussians fade in over N frames
    int ramp_out_frames = 3;                // unmatched Gaussians fade out over N frames

    // Output
    int keyframe_interval = 30;             // full SPZ every N frames
    bool enable_delta_compression = true;
};

// Callback when a new Gaussian frame is ready for streaming.
using GaussianFrameCallback = std::function<void(const GaussianFrame&)>;

// The real-time Gaussian preview pipeline.
//
// Latency target: 1-2 seconds from capture to viewer.
// This allows using DepthSplat (0.6s on A100 for 12 views)
// rather than requiring sub-33ms inference.
//
// Pipeline:
//   camera images (CUDA IPC)
//     -> camera selector (pick K best views)
//     -> feed-forward Gaussian inference (DepthSplat/MVSplat via TensorRT)
//     -> temporal tracker (match + EMA blend across frames)
//     -> SPZ compression (keyframe + deltas)
//     -> stream callback
//
// The mesh + texture path runs separately (offline, Argo Workflows)
// and is NOT on this latency-critical path.
class GaussianPipeline {
public:
    explicit GaussianPipeline(const PipelineConfig& config);
    ~GaussianPipeline();

    // Submit a new set of camera images for Gaussian inference.
    // Non-blocking — queues work for the inference thread.
    void submit_frame(int64_t frame_id,
                      int64_t timestamp_ns,
                      const std::vector<CameraInput>& cameras);

    // Register callback for completed Gaussian frames.
    void set_output_callback(GaussianFrameCallback cb);

    // Start/stop the inference thread.
    void start();
    void stop();

    // Stats
    double average_latency_ms() const;
    int64_t frames_processed() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace heimdall::gaussian
