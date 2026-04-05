#pragma once

// Production GPS-Gaussian inference via TensorRT.
//
// Requires: sm_75+ (Turing or newer), TensorRT 8.6+
// For: RTX 2080+, RTX 3060+, RTX 4060+, A100, H100, B200
//
// This is a thin wrapper that delegates to gaussian/src/trt_inference.h
// with GPS-Gaussian-specific configuration (2 input views, stereo pair mode).
//
// The PyTorch fallback (inference/pytorch/gps_gaussian_pt.py) provides
// the same interface for dev GPUs that can't run TensorRT (Maxwell/Pascal).

#include "../../gaussian/src/trt_inference.h"

namespace heimdall::inference::tensorrt {

// GPS-Gaussian-specific TRT config defaults.
// Wraps TrtGaussianInference with stereo pair settings.
inline gaussian::TrtInferenceConfig gps_gaussian_defaults() {
    gaussian::TrtInferenceConfig cfg;
    cfg.num_views = 2;           // GPS-Gaussian operates on stereo pairs
    cfg.input_height = 1024;
    cfg.input_width = 1024;
    return cfg;
}

// For the production worker pipeline:
//   auto cfg = gps_gaussian_defaults();
//   cfg.engine_path = "/models/gps_gaussian_sm86.trt";
//   auto engine = std::make_unique<gaussian::TrtGaussianInference>(cfg);

} // namespace heimdall::inference::tensorrt
