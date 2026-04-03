// GPU preprocessing and postprocessing kernels for matting inference.
//
// Compile with nvcc:
//   nvcc -std=c++17 -c matting_preprocess.cu -o matting_preprocess.o

#include "matting_preprocess.h"
#include <cuda_runtime.h>

namespace heimdall::segmentation {

// ============================================================================
// Helper: bilinear sample from RGBA interleaved image, single channel
// ============================================================================

__device__ inline float bilinear_sample_rgba(
    const float* __restrict__ rgba,
    int h, int w, int channel,
    float sx, float sy
) {
    int x0 = static_cast<int>(floorf(sx));
    int y0 = static_cast<int>(floorf(sy));
    int x1 = x0 + 1;
    int y1 = y0 + 1;

    float fx = sx - static_cast<float>(x0);
    float fy = sy - static_cast<float>(y0);

    x0 = max(0, min(x0, w - 1));
    x1 = max(0, min(x1, w - 1));
    y0 = max(0, min(y0, h - 1));
    y1 = max(0, min(y1, h - 1));

    float v00 = rgba[(y0 * w + x0) * 4 + channel];
    float v01 = rgba[(y0 * w + x1) * 4 + channel];
    float v10 = rgba[(y1 * w + x0) * 4 + channel];
    float v11 = rgba[(y1 * w + x1) * 4 + channel];

    float top    = v00 * (1.0f - fx) + v01 * fx;
    float bottom = v10 * (1.0f - fx) + v11 * fx;
    return top * (1.0f - fy) + bottom * fy;
}

// ============================================================================
// Helper: bilinear sample from single-channel float image
// ============================================================================

__device__ inline float bilinear_sample_f32(
    const float* __restrict__ src,
    int h, int w,
    float sx, float sy
) {
    int x0 = static_cast<int>(floorf(sx));
    int y0 = static_cast<int>(floorf(sy));
    int x1 = x0 + 1;
    int y1 = y0 + 1;

    float fx = sx - static_cast<float>(x0);
    float fy = sy - static_cast<float>(y0);

    x0 = max(0, min(x0, w - 1));
    x1 = max(0, min(x1, w - 1));
    y0 = max(0, min(y0, h - 1));
    y1 = max(0, min(y1, h - 1));

    float v00 = src[y0 * w + x0];
    float v01 = src[y0 * w + x1];
    float v10 = src[y1 * w + x0];
    float v11 = src[y1 * w + x1];

    float top    = v00 * (1.0f - fx) + v01 * fx;
    float bottom = v10 * (1.0f - fx) + v11 * fx;
    return top * (1.0f - fy) + bottom * fy;
}

// ============================================================================
// Preprocessing: RGBA F32 -> planar RGB F32 with resize
// ============================================================================

__global__ void matting_preprocess_kernel(
    const float* __restrict__ src_rgba,  // [src_h, src_w, 4]
    float* __restrict__ dst_rgb,         // [3, dst_h, dst_w]
    int src_h, int src_w,
    int dst_h, int dst_w
) {
    int ox = blockIdx.x * blockDim.x + threadIdx.x;
    int oy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ox >= dst_w || oy >= dst_h) return;

    // Map output pixel center to source coordinates.
    float sx = (static_cast<float>(ox) + 0.5f) * static_cast<float>(src_w)
               / static_cast<float>(dst_w) - 0.5f;
    float sy = (static_cast<float>(oy) + 0.5f) * static_cast<float>(src_h)
               / static_cast<float>(dst_h) - 0.5f;

    for (int c = 0; c < 3; c++) {
        float val = bilinear_sample_rgba(src_rgba, src_h, src_w, c, sx, sy);
        // Clamp to [0, 1] for model input normalization.
        val = fminf(1.0f, fmaxf(0.0f, val));
        // Write in CHW planar layout: [c, oy, ox]
        dst_rgb[c * dst_h * dst_w + oy * dst_w + ox] = val;
    }
}

void launch_matting_preprocess(
    const float* src_rgba,
    float* dst_rgb,
    int src_h, int src_w,
    int dst_h, int dst_w,
    cudaStream_t stream
) {
    dim3 block(16, 16);
    dim3 grid(
        (static_cast<unsigned>(dst_w) + block.x - 1) / block.x,
        (static_cast<unsigned>(dst_h) + block.y - 1) / block.y
    );
    matting_preprocess_kernel<<<grid, block, 0, stream>>>(
        src_rgba, dst_rgb, src_h, src_w, dst_h, dst_w
    );
}

// ============================================================================
// Preprocessing with background: FG RGBA + BG RGBA -> 6-channel planar
// ============================================================================

__global__ void matting_preprocess_with_bg_kernel(
    const float* __restrict__ src_fg_rgba,  // [fg_h, fg_w, 4]
    int fg_h, int fg_w,
    const float* __restrict__ src_bg_rgba,  // [bg_h, bg_w, 4]
    int bg_h, int bg_w,
    float* __restrict__ dst_concat,         // [6, dst_h, dst_w]
    int dst_h, int dst_w
) {
    int ox = blockIdx.x * blockDim.x + threadIdx.x;
    int oy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ox >= dst_w || oy >= dst_h) return;

    // Source coordinates for foreground.
    float fg_sx = (static_cast<float>(ox) + 0.5f) * static_cast<float>(fg_w)
                  / static_cast<float>(dst_w) - 0.5f;
    float fg_sy = (static_cast<float>(oy) + 0.5f) * static_cast<float>(fg_h)
                  / static_cast<float>(dst_h) - 0.5f;

    // Source coordinates for background.
    float bg_sx = (static_cast<float>(ox) + 0.5f) * static_cast<float>(bg_w)
                  / static_cast<float>(dst_w) - 0.5f;
    float bg_sy = (static_cast<float>(oy) + 0.5f) * static_cast<float>(bg_h)
                  / static_cast<float>(dst_h) - 0.5f;

    int pixel_idx = oy * dst_w + ox;
    int plane_size = dst_h * dst_w;

    // Channels 0-2: foreground RGB
    for (int c = 0; c < 3; c++) {
        float val = bilinear_sample_rgba(src_fg_rgba, fg_h, fg_w, c, fg_sx, fg_sy);
        val = fminf(1.0f, fmaxf(0.0f, val));
        dst_concat[c * plane_size + pixel_idx] = val;
    }

    // Channels 3-5: background RGB
    for (int c = 0; c < 3; c++) {
        float val = bilinear_sample_rgba(src_bg_rgba, bg_h, bg_w, c, bg_sx, bg_sy);
        val = fminf(1.0f, fmaxf(0.0f, val));
        dst_concat[(3 + c) * plane_size + pixel_idx] = val;
    }
}

void launch_matting_preprocess_with_bg(
    const float* src_fg_rgba,
    int fg_h, int fg_w,
    const float* src_bg_rgba,
    int bg_h, int bg_w,
    float* dst_concat,
    int dst_h, int dst_w,
    cudaStream_t stream
) {
    dim3 block(16, 16);
    dim3 grid(
        (static_cast<unsigned>(dst_w) + block.x - 1) / block.x,
        (static_cast<unsigned>(dst_h) + block.y - 1) / block.y
    );
    matting_preprocess_with_bg_kernel<<<grid, block, 0, stream>>>(
        src_fg_rgba, fg_h, fg_w,
        src_bg_rgba, bg_h, bg_w,
        dst_concat, dst_h, dst_w
    );
}

// ============================================================================
// Alpha resize (single-channel bilinear)
// ============================================================================

__global__ void alpha_resize_kernel(
    const float* __restrict__ src_alpha,  // [src_h, src_w]
    float* __restrict__ dst_alpha,        // [dst_h, dst_w]
    int src_h, int src_w,
    int dst_h, int dst_w
) {
    int ox = blockIdx.x * blockDim.x + threadIdx.x;
    int oy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ox >= dst_w || oy >= dst_h) return;

    float sx = (static_cast<float>(ox) + 0.5f) * static_cast<float>(src_w)
               / static_cast<float>(dst_w) - 0.5f;
    float sy = (static_cast<float>(oy) + 0.5f) * static_cast<float>(src_h)
               / static_cast<float>(dst_h) - 0.5f;

    float val = bilinear_sample_f32(src_alpha, src_h, src_w, sx, sy);
    dst_alpha[oy * dst_w + ox] = fminf(1.0f, fmaxf(0.0f, val));
}

void launch_alpha_resize(
    const float* src_alpha,
    float* dst_alpha,
    int src_h, int src_w,
    int dst_h, int dst_w,
    cudaStream_t stream
) {
    dim3 block(16, 16);
    dim3 grid(
        (static_cast<unsigned>(dst_w) + block.x - 1) / block.x,
        (static_cast<unsigned>(dst_h) + block.y - 1) / block.y
    );
    alpha_resize_kernel<<<grid, block, 0, stream>>>(
        src_alpha, dst_alpha, src_h, src_w, dst_h, dst_w
    );
}

// ============================================================================
// Alpha threshold (binarize)
// ============================================================================

__global__ void alpha_threshold_kernel(
    float* __restrict__ alpha,  // [h, w]
    int h, int w,
    float threshold
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = h * w;
    if (idx >= total) return;

    alpha[idx] = (alpha[idx] >= threshold) ? 1.0f : 0.0f;
}

void launch_alpha_threshold(
    float* alpha,
    int h, int w,
    float threshold,
    cudaStream_t stream
) {
    // Skip if threshold is non-positive (soft matte passthrough).
    if (threshold <= 0.0f) return;

    int total = h * w;
    int block_size = 256;
    int grid_size = (total + block_size - 1) / block_size;
    alpha_threshold_kernel<<<grid_size, block_size, 0, stream>>>(
        alpha, h, w, threshold
    );
}

// ============================================================================
// Morphological opening (erode then dilate)
// ============================================================================

__global__ void erode_kernel(
    const float* __restrict__ src,
    float* __restrict__ dst,
    int h, int w,
    int radius  // half-size: kernel_size = 2 * radius + 1
) {
    int ox = blockIdx.x * blockDim.x + threadIdx.x;
    int oy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ox >= w || oy >= h) return;

    float min_val = 1.0f;
    for (int dy = -radius; dy <= radius; dy++) {
        int sy = oy + dy;
        if (sy < 0 || sy >= h) { min_val = 0.0f; continue; }
        for (int dx = -radius; dx <= radius; dx++) {
            int sx = ox + dx;
            if (sx < 0 || sx >= w) { min_val = 0.0f; continue; }
            float v = src[sy * w + sx];
            min_val = fminf(min_val, v);
        }
    }
    dst[oy * w + ox] = min_val;
}

__global__ void dilate_kernel(
    const float* __restrict__ src,
    float* __restrict__ dst,
    int h, int w,
    int radius
) {
    int ox = blockIdx.x * blockDim.x + threadIdx.x;
    int oy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ox >= w || oy >= h) return;

    float max_val = 0.0f;
    for (int dy = -radius; dy <= radius; dy++) {
        int sy = oy + dy;
        if (sy < 0 || sy >= h) continue;
        for (int dx = -radius; dx <= radius; dx++) {
            int sx = ox + dx;
            if (sx < 0 || sx >= w) continue;
            float v = src[sy * w + sx];
            max_val = fmaxf(max_val, v);
        }
    }
    dst[oy * w + ox] = max_val;
}

void launch_morphology_open(
    float* alpha,
    float* temp,
    int h, int w,
    int kernel_size,
    cudaStream_t stream
) {
    if (kernel_size < 3) return;
    int radius = kernel_size / 2;

    dim3 block(16, 16);
    dim3 grid(
        (static_cast<unsigned>(w) + block.x - 1) / block.x,
        (static_cast<unsigned>(h) + block.y - 1) / block.y
    );

    // Erode: alpha -> temp
    erode_kernel<<<grid, block, 0, stream>>>(alpha, temp, h, w, radius);

    // Dilate: temp -> alpha
    dilate_kernel<<<grid, block, 0, stream>>>(temp, alpha, h, w, radius);
}

// ============================================================================
// Temporal EMA blending
// ============================================================================

__global__ void temporal_ema_kernel(
    float* __restrict__ current,
    const float* __restrict__ previous,
    int total,
    float blend
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    current[idx] = blend * previous[idx] + (1.0f - blend) * current[idx];
}

void launch_temporal_ema(
    float* current,
    const float* previous,
    int h, int w,
    float blend,
    cudaStream_t stream
) {
    if (blend <= 0.0f) return;  // No blending requested.

    int total = h * w;
    int block_size = 256;
    int grid_size = (total + block_size - 1) / block_size;
    temporal_ema_kernel<<<grid_size, block_size, 0, stream>>>(
        current, previous, total, blend
    );
}

// ============================================================================
// Float32 alpha -> uint8 alpha
// ============================================================================

__global__ void alpha_f32_to_u8_kernel(
    const float* __restrict__ src_f32,
    unsigned char* __restrict__ dst_u8,
    int total
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    float val = fminf(1.0f, fmaxf(0.0f, src_f32[idx]));
    dst_u8[idx] = static_cast<unsigned char>(val * 255.0f + 0.5f);
}

void launch_alpha_f32_to_u8(
    const float* src_f32,
    unsigned char* dst_u8,
    int h, int w,
    cudaStream_t stream
) {
    int total = h * w;
    int block_size = 256;
    int grid_size = (total + block_size - 1) / block_size;
    alpha_f32_to_u8_kernel<<<grid_size, block_size, 0, stream>>>(
        src_f32, dst_u8, total
    );
}

} // namespace heimdall::segmentation
