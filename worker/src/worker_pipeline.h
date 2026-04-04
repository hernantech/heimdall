#pragma once

#include "../../gaussian/src/pipeline.h"
#include "quantizer.h"
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace heimdall::worker {

struct StereoPairInput {
    int camera_a_index;
    int camera_b_index;
    std::vector<uint8_t> camera_a_nal;  // H.265 NAL units
    std::vector<uint8_t> camera_b_nal;
    // Calibration (OpenCV convention)
    float intrinsics_a[9];   // 3x3 row-major
    float extrinsics_a[16];  // 4x4 world-to-camera row-major
    float intrinsics_b[9];
    float extrinsics_b[16];
    int image_width;
    int image_height;
};

struct FrameRequest {
    int64_t frame_id;
    int64_t timestamp_ns;
    std::vector<StereoPairInput> stereo_pairs;
};

struct StereoPairResult {
    int camera_a_index;
    int camera_b_index;
    int num_gaussians;
    double processing_time_ms;
};

struct FrameResponse {
    int64_t frame_id;
    int num_total_gaussians;
    std::vector<uint8_t> quantized_gaussians;  // quantizer output
    std::vector<StereoPairResult> pair_results;
    double total_processing_time_ms;
};

struct WorkerPipelineConfig {
    // Model paths
    std::string gps_gaussian_model_path;   // TensorRT engine for GPS-Gaussian
    std::string matting_model_path;         // TensorRT engine for RVM/BMV2
    std::string matting_model_type = "rvm"; // "rvm" or "bmv2"

    // Inference resolution
    int inference_width = 1024;
    int inference_height = 1024;
    int matting_width = 512;
    int matting_height = 288;

    // Quantizer
    QuantizerConfig quantizer_config;

    // GPU
    int gpu_device = 0;
};

// Stateless worker pipeline.
//
// Processes a single frame: for each assigned stereo pair,
// decode H.265 → segment (matting) → GPS-Gaussian inference → quantize.
//
// No temporal state — every frame is independent.
// Temporal tracking happens on the merge machine.
class WorkerPipeline {
public:
    explicit WorkerPipeline(const WorkerPipelineConfig& config);
    ~WorkerPipeline();

    // Process a complete frame request. Thread-safe.
    FrameResponse process_frame(const FrameRequest& request);

    // Check if models are loaded and ready.
    bool is_ready() const;

    // Stats
    int64_t frames_processed() const;
    double average_processing_time_ms() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace heimdall::worker
