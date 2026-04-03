// GPU preprocessing kernels for TensorRT inference input preparation.
//
// Compile with nvcc:
//   nvcc -std=c++17 -c trt_preprocess.cu -o trt_preprocess.o

#include "trt_preprocess.h"
#include <cuda_runtime.h>

namespace heimdall::gaussian {

// Convert RGBA float32 [H_src, W_src, 4] to RGB float32 [3, H_dst, W_dst]
// with bilinear resize.  One thread per output pixel.
__global__ void rgba_to_rgb_resize_kernel(
    const float* __restrict__ src_rgba,   // [H_src, W_src, 4] RGBA float32
    float* __restrict__ dst_rgb,          // [3, H_dst, W_dst] planar RGB float32
    int src_h, int src_w,
    int dst_h, int dst_w
) {
    int ox = blockIdx.x * blockDim.x + threadIdx.x; // output x
    int oy = blockIdx.y * blockDim.y + threadIdx.y; // output y
    if (ox >= dst_w || oy >= dst_h) return;

    // Map output pixel to source coordinates (bilinear sampling).
    float sx = (static_cast<float>(ox) + 0.5f) * static_cast<float>(src_w)
               / static_cast<float>(dst_w) - 0.5f;
    float sy = (static_cast<float>(oy) + 0.5f) * static_cast<float>(src_h)
               / static_cast<float>(dst_h) - 0.5f;

    int x0 = static_cast<int>(floorf(sx));
    int y0 = static_cast<int>(floorf(sy));
    int x1 = x0 + 1;
    int y1 = y0 + 1;

    float fx = sx - static_cast<float>(x0);
    float fy = sy - static_cast<float>(y0);

    // Clamp to source bounds.
    x0 = max(0, min(x0, src_w - 1));
    x1 = max(0, min(x1, src_w - 1));
    y0 = max(0, min(y0, src_h - 1));
    y1 = max(0, min(y1, src_h - 1));

    // Bilinear interpolation for each of R, G, B (skip A channel).
    for (int c = 0; c < 3; c++) {
        float v00 = src_rgba[(y0 * src_w + x0) * 4 + c];
        float v01 = src_rgba[(y0 * src_w + x1) * 4 + c];
        float v10 = src_rgba[(y1 * src_w + x0) * 4 + c];
        float v11 = src_rgba[(y1 * src_w + x1) * 4 + c];

        float top    = v00 * (1.0f - fx) + v01 * fx;
        float bottom = v10 * (1.0f - fx) + v11 * fx;
        float val    = top * (1.0f - fy) + bottom * fy;

        // Write in CHW planar layout: [c, oy, ox]
        dst_rgb[c * dst_h * dst_w + oy * dst_w + ox] = val;
    }
}

void launch_rgba_to_rgb_resize(
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
    rgba_to_rgb_resize_kernel<<<grid, block, 0, stream>>>(
        src_rgba, dst_rgb, src_h, src_w, dst_h, dst_w
    );
    // Error check is deferred to the next cudaStreamSynchronize.
}

} // namespace heimdall::gaussian
