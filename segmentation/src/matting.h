#pragma once

// Background segmentation / matting engine for multi-camera volumetric capture.
//
// Supports two model architectures:
//   - RVM  (Robust Video Matting):    input [B, 3, H, W] RGB
//                                      output [B, 1, H, W] alpha
//   - BMV2 (BackgroundMattingV2):     input [B, 6, H, W] RGB + background RGB
//                                      output [B, 1, H, W] alpha + [B, 3, H, W] foreground
//
// Models are loaded from ONNX (with optional TensorRT engine) and run on GPU.
// The engine handles preprocessing (resize + color conversion), inference,
// and postprocessing (resize back, threshold, morphology, temporal EMA, F32->U8).
//
// Required link libraries:
//   -lnvinfer -lcudart
//
// Required headers:
//   NvInfer.h, NvInferRuntime.h, cuda_runtime.h

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

// Forward declarations — avoids requiring TensorRT headers for downstream consumers.
namespace nvinfer1 {
class ICudaEngine;
class IExecutionContext;
class IRuntime;
class ILogger;
} // namespace nvinfer1

struct CUstream_st;
using cudaStream_t = CUstream_st*;

namespace heimdall::segmentation {

// Supported matting model architectures.
enum class MattingModelType {
    RVM,   // Robust Video Matting: 3-channel RGB input
    BMV2   // BackgroundMattingV2: 6-channel (RGB + background) input
};

// Configuration for the matting engine.
struct MattingConfig {
    // Path to ONNX model (.onnx) or serialized TensorRT engine (.trt).
    std::string model_path;

    // Model architecture.
    MattingModelType model_type = MattingModelType::RVM;

    // Model input resolution. The source image is resized to this before inference.
    // Typical values: 512x288 (fast), 1024x576 (quality).
    int inference_width  = 512;
    int inference_height = 288;

    // Alpha binarization threshold.
    //   0.0 = soft matte (no thresholding — continuous alpha values)
    //   0.5 = typical hard mask threshold
    float threshold = 0.0f;

    // Temporal EMA blending factor for alpha stability across frames.
    //   0.0 = no temporal smoothing (each frame independent)
    //   0.3 = moderate smoothing (recommended)
    //   Higher values = more smoothing but increased latency to motion.
    float temporal_alpha = 0.0f;

    // Morphological post-processing to clean alpha edges (erode then dilate).
    bool enable_morphology = false;

    // Size of square structuring element for morphological ops (must be odd, >= 3).
    int morphology_kernel_size = 5;

    // Use an external CUDA stream instead of creating one internally.
    // If nullptr, a dedicated stream is created.
    cudaStream_t external_stream = nullptr;
};

// Input for a single camera frame to be matted.
struct MattingInput {
    int camera_index;       // Identifier for this camera in the rig.
    int width;              // Source image width (pixels).
    int height;             // Source image height (pixels).
    void* gpu_rgba_f32;     // CUDA device pointer to [H, W, 4] RGBA float32.

    // Optional: clean plate (background-only image) for BMV2 model.
    // Required when model_type == BMV2, ignored for RVM.
    // CUDA device pointer to [bg_h, bg_w, 4] RGBA float32, or nullptr.
    void* gpu_background_rgba_f32 = nullptr;
    int bg_width  = 0;
    int bg_height = 0;
};

// Output for a single camera frame after matting.
struct MattingOutput {
    int camera_index;
    int width;              // Same as input width.
    int height;             // Same as input height.
    void* gpu_alpha_u8;     // CUDA device pointer to [H, W] uint8 alpha (0-255).
                            // Owned by MattingEngine — valid until next call for this camera.
};

// Background segmentation / matting engine.
//
// Lifecycle:
//   1. Construct with config (loads model, allocates GPU buffers).
//   2. Call process_frame() or process_batch() each frame.
//   3. Destroy when done — GPU buffers are freed automatically.
//
// Thread safety: NOT thread-safe. The caller must ensure that processing
// methods are not called concurrently from multiple threads. Typical usage
// is from a single worker thread in the capture pipeline.
class MattingEngine {
public:
    // Load model and allocate all GPU buffers.
    // Throws std::runtime_error on failure.
    explicit MattingEngine(const MattingConfig& config);

    ~MattingEngine();

    // Non-copyable, movable.
    MattingEngine(const MattingEngine&) = delete;
    MattingEngine& operator=(const MattingEngine&) = delete;
    MattingEngine(MattingEngine&&) noexcept;
    MattingEngine& operator=(MattingEngine&&) noexcept;

    // Process a single camera frame.
    //
    // The input gpu_rgba_f32 must point to a valid CUDA device allocation.
    // For BMV2, gpu_background_rgba_f32 must also be valid.
    //
    // Returns a MattingOutput whose gpu_alpha_u8 is a device pointer to [H, W] uint8.
    // The output buffer is owned by MattingEngine and remains valid until the next
    // call to process_frame() or process_batch() for the same camera_index.
    //
    // Throws std::runtime_error on failure.
    MattingOutput process_frame(const MattingInput& input);

    // Process a batch of camera frames.
    //
    // Processes each camera sequentially through the TensorRT engine.
    // Batched inference (single forward pass with B > 1) is used when the engine
    // was built with dynamic batch support; otherwise falls back to sequential.
    //
    // Returns one MattingOutput per input, in the same order.
    // Output buffers are owned by MattingEngine and valid until the next batch call.
    //
    // Throws std::runtime_error on failure.
    std::vector<MattingOutput> process_batch(const std::vector<MattingInput>& inputs);

    // Accessors.
    int inference_width()  const;
    int inference_height() const;
    MattingModelType model_type() const;

    // Reset temporal state (e.g., on scene change or camera reconfiguration).
    // Clears all per-camera previous-frame alpha buffers.
    void reset_temporal_state();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace heimdall::segmentation
