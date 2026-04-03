#pragma once

// TensorRT inference wrapper for feed-forward Gaussian splatting models
// (DepthSplat, MVSplat exported via ONNX -> TensorRT).
//
// Loads a serialized TensorRT engine (.trt), pre-allocates GPU buffers,
// and runs inference: K camera images + calibration -> Gaussian splat attributes.
//
// Required link libraries (not linked at compile time — link at integration):
//   -lnvinfer -lnvonnxparser -lcudart -lnppig -lnppidei
//
// Required headers (TensorRT SDK):
//   NvInfer.h, NvInferRuntime.h, cuda_runtime.h, nppi_geometry_transforms.h

#include "pipeline.h"

#include <cstddef>
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

namespace heimdall::gaussian {

// Camera calibration data fed alongside images.
struct CameraCalibration {
    // 3x3 intrinsic matrix, row-major: [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
    float intrinsic[9];

    // 4x4 world-to-camera extrinsic matrix, row-major.
    // Top-left 3x3 is the rotation matrix; rightmost column is translation.
    float extrinsic[16];
};

// Configuration for the TensorRT inference wrapper.
struct TrtInferenceConfig {
    std::string engine_path;    // Path to serialized .trt engine file.

    int num_views       = 6;    // K: number of input camera views.
    int input_height    = 512;  // Model input height (pixels).
    int input_width     = 960;  // Model input width (pixels).

    // Maximum number of output Gaussians the model can produce.
    // Must match the engine's output binding shape. For DepthSplat at
    // 512x960 with 6 views, the default is H*W*K = 512*960*6 = 2,949,120
    // but most models produce far fewer (e.g. 100-300k after pruning).
    int max_gaussians   = 500000;

    // Use the default CUDA stream (nullptr) or provide an external one.
    // If nullptr, a dedicated stream is created internally.
    cudaStream_t external_stream = nullptr;
};

// TensorRT inference wrapper for feed-forward Gaussian splatting.
//
// Lifecycle:
//   1. Construct with engine path and config.
//   2. Call infer() each frame with camera images + calibration.
//   3. Destroy when done — GPU buffers are freed automatically.
//
// Thread safety: NOT thread-safe. The caller must ensure that infer()
// is not called concurrently from multiple threads. Typical usage is
// from a single inference worker thread in GaussianPipeline.
class TrtGaussianInference {
public:
    // Load a TensorRT engine and allocate all GPU buffers.
    // Throws std::runtime_error on failure.
    explicit TrtGaussianInference(const TrtInferenceConfig& config);

    ~TrtGaussianInference();

    // Non-copyable, movable.
    TrtGaussianInference(const TrtGaussianInference&) = delete;
    TrtGaussianInference& operator=(const TrtGaussianInference&) = delete;
    TrtGaussianInference(TrtGaussianInference&&) noexcept;
    TrtGaussianInference& operator=(TrtGaussianInference&&) noexcept;

    // Run inference on a set of camera views.
    //
    // cameras:       Must contain exactly config.num_views entries.
    //                Each CameraInput::gpu_rgba_f32 must point to a valid
    //                CUDA device allocation of [H, W, 4] float32.
    //
    // calibrations:  Per-camera intrinsics + extrinsics, same order as cameras.
    //
    // Returns a GaussianFrame with host-side Gaussian vector populated.
    // The frame_id and timestamp_ns are set to 0; the caller should overwrite them.
    //
    // Throws std::runtime_error if inference fails.
    GaussianFrame infer(const std::vector<CameraInput>& cameras,
                        const std::vector<CameraCalibration>& calibrations);

    // Accessors
    int num_views()     const;
    int input_height()  const;
    int input_width()   const;
    int max_gaussians() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace heimdall::gaussian
