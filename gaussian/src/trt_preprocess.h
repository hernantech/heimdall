#pragma once

// GPU preprocessing kernels for TensorRT inference input preparation.
// Implementation lives in trt_preprocess.cu (compiled with nvcc).

struct CUstream_st;
using cudaStream_t = CUstream_st*;

namespace heimdall::gaussian {

// Resize an RGBA float32 image and convert to planar RGB float32 on the GPU.
//
// src_rgba: device pointer, [src_h, src_w, 4] interleaved RGBA float32
// dst_rgb:  device pointer, [3, dst_h, dst_w] planar RGB float32 (CHW layout)
//
// Uses bilinear interpolation. Alpha channel is discarded.
// Launches asynchronously on the given CUDA stream.
void launch_rgba_to_rgb_resize(
    const float* src_rgba,
    float* dst_rgb,
    int src_h, int src_w,
    int dst_h, int dst_w,
    cudaStream_t stream
);

} // namespace heimdall::gaussian
