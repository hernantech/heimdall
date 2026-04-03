#pragma once

// GPU preprocessing and postprocessing kernels for matting inference.
// Implementation lives in matting_preprocess.cu (compiled with nvcc).

struct CUstream_st;
using cudaStream_t = CUstream_st*;

namespace heimdall::segmentation {

// Resize an RGBA float32 image and convert to planar RGB float32 for matting.
//
// src_rgba: device pointer, [src_h, src_w, 4] interleaved RGBA float32
// dst_rgb:  device pointer, [3, dst_h, dst_w] planar RGB float32 (CHW layout)
//
// Values are clamped to [0, 1] range. Alpha channel is discarded.
// Uses bilinear interpolation. Launches asynchronously on the given stream.
void launch_matting_preprocess(
    const float* src_rgba,
    float* dst_rgb,
    int src_h, int src_w,
    int dst_h, int dst_w,
    cudaStream_t stream
);

// Concatenate foreground RGB and background RGB into a 6-channel planar tensor.
// Used by BackgroundMattingV2 which takes [B, 6, H, W] input.
//
// src_fg_rgba:  device pointer, [src_h, src_w, 4] interleaved RGBA float32
// src_bg_rgba:  device pointer, [bg_h, bg_w, 4] interleaved RGBA float32
// dst_concat:   device pointer, [6, dst_h, dst_w] planar float32 (CHW layout)
//                channels 0-2 = foreground RGB, channels 3-5 = background RGB
//
// Both inputs are resized to (dst_h, dst_w) with bilinear interpolation.
// Values are clamped to [0, 1]. Alpha channels are discarded.
void launch_matting_preprocess_with_bg(
    const float* src_fg_rgba,
    int fg_h, int fg_w,
    const float* src_bg_rgba,
    int bg_h, int bg_w,
    float* dst_concat,
    int dst_h, int dst_w,
    cudaStream_t stream
);

// Resize alpha output from model resolution back to original camera resolution
// using bilinear interpolation.
//
// src_alpha: device pointer, [src_h, src_w] float32 (single channel, range [0,1])
// dst_alpha: device pointer, [dst_h, dst_w] float32 (single channel)
void launch_alpha_resize(
    const float* src_alpha,
    float* dst_alpha,
    int src_h, int src_w,
    int dst_h, int dst_w,
    cudaStream_t stream
);

// Apply binary threshold to alpha matte.
// Pixels >= threshold become 1.0, pixels < threshold become 0.0.
// If threshold <= 0.0, this is a no-op (soft matte passthrough).
//
// alpha: device pointer, [h, w] float32, modified in-place
void launch_alpha_threshold(
    float* alpha,
    int h, int w,
    float threshold,
    cudaStream_t stream
);

// Apply morphological erosion followed by dilation (opening) to clean alpha edges.
// Uses a square structuring element of the given kernel_size.
//
// alpha:        device pointer, [h, w] float32
// temp:         device pointer, [h, w] float32 (scratch buffer, same size as alpha)
// kernel_size:  side length of square structuring element (must be odd, >= 3)
void launch_morphology_open(
    float* alpha,
    float* temp,
    int h, int w,
    int kernel_size,
    cudaStream_t stream
);

// Exponential moving average blend of current alpha with previous frame's alpha.
//
// current:  device pointer, [h, w] float32 (modified in-place to blended result)
// previous: device pointer, [h, w] float32 (read-only, previous frame's alpha)
// blend:    blending factor in [0, 1]. Result = blend * previous + (1 - blend) * current.
//           0.0 = no temporal smoothing (use current only),
//           1.0 = full freeze (use previous only).
void launch_temporal_ema(
    float* current,
    const float* previous,
    int h, int w,
    float blend,
    cudaStream_t stream
);

// Convert float32 alpha [0.0, 1.0] to uint8 alpha [0, 255].
//
// src_f32: device pointer, [h, w] float32
// dst_u8:  device pointer, [h, w] uint8
void launch_alpha_f32_to_u8(
    const float* src_f32,
    unsigned char* dst_u8,
    int h, int w,
    cudaStream_t stream
);

} // namespace heimdall::segmentation
