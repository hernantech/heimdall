// TensorRT inference wrapper for feed-forward Gaussian splatting models.
//
// Build:
//   # 1. Compile the CUDA preprocessing kernel (nvcc):
//   nvcc -std=c++17 -c trt_preprocess.cu -o trt_preprocess.o \
//        -I/usr/local/cuda/include
//
//   # 2. Compile this file (g++ or nvcc — no CUDA syntax here):
//   g++ -std=c++17 -c trt_inference.cpp -o trt_inference.o \
//       -I/usr/local/cuda/include -I/path/to/TensorRT/include
//
//   # 3. Link:
//   g++ trt_inference.o trt_preprocess.o -o ... \
//       -L/path/to/TensorRT/lib -lnvinfer \
//       -L/usr/local/cuda/lib64 -lcudart
//
// The TensorRT engine file (.trt) is produced by export_tensorrt.py.

#include "trt_inference.h"
#include "trt_preprocess.h"

#include <NvInfer.h>
#include <NvInferRuntime.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cassert>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace heimdall::gaussian {

// ============================================================================
// TensorRT logger (nvinfer1::ILogger implementation)
// ============================================================================

class TrtLogger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        // Suppress verbose and info to avoid log spam during normal operation.
        if (severity == Severity::kINFO || severity == Severity::kVERBOSE) return;

        const char* prefix = "";
        switch (severity) {
            case Severity::kINTERNAL_ERROR: prefix = "[TRT INTERNAL ERROR] "; break;
            case Severity::kERROR:          prefix = "[TRT ERROR] ";          break;
            case Severity::kWARNING:        prefix = "[TRT WARNING] ";        break;
            case Severity::kINFO:           prefix = "[TRT INFO] ";           break;
            case Severity::kVERBOSE:        prefix = "[TRT VERBOSE] ";        break;
        }
        std::cerr << prefix << msg << std::endl;
    }
};

// ============================================================================
// CUDA error checking
// ============================================================================

#define CUDA_CHECK(call)                                                       \
    do {                                                                        \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            std::ostringstream oss;                                             \
            oss << "CUDA error at " << __FILE__ << ":" << __LINE__              \
                << " -- " << cudaGetErrorString(err);                           \
            throw std::runtime_error(oss.str());                                \
        }                                                                       \
    } while (0)

// ============================================================================
// GpuBuffer — RAII wrapper for a CUDA device allocation
// ============================================================================

class GpuBuffer {
public:
    GpuBuffer() = default;

    explicit GpuBuffer(size_t bytes) : size_(bytes) {
        if (bytes > 0) {
            CUDA_CHECK(cudaMalloc(&ptr_, bytes));
        }
    }

    ~GpuBuffer() { release(); }

    // Non-copyable.
    GpuBuffer(const GpuBuffer&) = delete;
    GpuBuffer& operator=(const GpuBuffer&) = delete;

    // Movable.
    GpuBuffer(GpuBuffer&& other) noexcept
        : ptr_(other.ptr_), size_(other.size_) {
        other.ptr_ = nullptr;
        other.size_ = 0;
    }

    GpuBuffer& operator=(GpuBuffer&& other) noexcept {
        if (this != &other) {
            release();
            ptr_ = other.ptr_;
            size_ = other.size_;
            other.ptr_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }

    void* get() const { return ptr_; }
    size_t size() const { return size_; }

private:
    void release() {
        if (ptr_) {
            cudaFree(ptr_); // Intentionally ignoring error in destructor path.
            ptr_ = nullptr;
            size_ = 0;
        }
    }

    void* ptr_ = nullptr;
    size_t size_ = 0;
};

// ============================================================================
// Engine binding index map
// ============================================================================

struct BindingIndices {
    // Inputs
    int images     = -1;  // [1, K, 3, H, W]
    int intrinsics = -1;  // [1, K, 3, 3]
    int extrinsics = -1;  // [1, K, 4, 4]

    // Outputs
    int positions  = -1;  // [N, 3]
    int scales     = -1;  // [N, 3]
    int rotations  = -1;  // [N, 4]   quaternion wxyz
    int opacities  = -1;  // [N, 1]
    int sh_coeffs  = -1;  // [N, 48]  degree-3 SH, 16 coefficients * 3 channels
};

// ============================================================================
// Impl (pimpl idiom)
// ============================================================================

struct TrtGaussianInference::Impl {
    TrtInferenceConfig config;

    // TensorRT objects (owned).
    TrtLogger                    logger;
    nvinfer1::IRuntime*          runtime = nullptr;
    nvinfer1::ICudaEngine*       engine  = nullptr;
    nvinfer1::IExecutionContext* context = nullptr;

    // CUDA stream (owned only if we created it internally).
    cudaStream_t stream      = nullptr;
    bool         owns_stream = false;

    BindingIndices bindings;

    // --- Pre-allocated GPU buffers (reused every frame) ---
    // Inputs
    GpuBuffer buf_images;       // [1, K, 3, H, W]  float32
    GpuBuffer buf_intrinsics;   // [1, K, 3, 3]     float32
    GpuBuffer buf_extrinsics;   // [1, K, 4, 4]     float32

    // Outputs
    GpuBuffer buf_positions;    // [max_gaussians, 3]   float32
    GpuBuffer buf_scales;       // [max_gaussians, 3]   float32
    GpuBuffer buf_rotations;    // [max_gaussians, 4]   float32
    GpuBuffer buf_opacities;    // [max_gaussians, 1]   float32
    GpuBuffer buf_sh_coeffs;    // [max_gaussians, 48]  float32

    // Scratch: per-view preprocessed image (reused across K views in one frame).
    GpuBuffer buf_view_rgb;     // [3, H, W]  float32

    // Host staging for calibration upload (avoids per-frame allocation).
    std::vector<float> host_intrinsics;
    std::vector<float> host_extrinsics;

    // Binding pointer array for enqueueV2(), indexed by TensorRT binding index.
    std::vector<void*> binding_ptrs;

    // Release all TensorRT objects and the stream.
    void destroy() {
        // Destroy in reverse creation order.
        if (context) { context->destroy(); context = nullptr; }
        if (engine)  { engine->destroy();  engine  = nullptr; }
        if (runtime) { runtime->destroy(); runtime = nullptr; }

        if (owns_stream && stream) {
            cudaStreamDestroy(stream);
            stream = nullptr;
        }
    }
};

// ============================================================================
// File I/O helper
// ============================================================================

static std::vector<char> read_engine_file(const std::string& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open TensorRT engine file: " + path);
    }

    auto size = file.tellg();
    if (size <= 0) {
        throw std::runtime_error("TensorRT engine file is empty or unreadable: " + path);
    }

    file.seekg(0, std::ios::beg);
    std::vector<char> buffer(static_cast<size_t>(size));
    if (!file.read(buffer.data(), static_cast<std::streamsize>(size))) {
        throw std::runtime_error("Failed to read TensorRT engine file: " + path);
    }

    return buffer;
}

// ============================================================================
// Constructor
// ============================================================================

TrtGaussianInference::TrtGaussianInference(const TrtInferenceConfig& config)
    : impl_(std::make_unique<Impl>())
{
    impl_->config = config;

    const int K = config.num_views;
    const int H = config.input_height;
    const int W = config.input_width;
    const int N = config.max_gaussians;

    // ---- Deserialize the TensorRT engine ----

    std::vector<char> engine_data = read_engine_file(config.engine_path);

    impl_->runtime = nvinfer1::createInferRuntime(impl_->logger);
    if (!impl_->runtime) {
        throw std::runtime_error("Failed to create TensorRT runtime");
    }

    impl_->engine = impl_->runtime->deserializeCudaEngine(
        engine_data.data(), engine_data.size());
    if (!impl_->engine) {
        impl_->destroy();
        throw std::runtime_error(
            "Failed to deserialize TensorRT engine from: " + config.engine_path);
    }

    impl_->context = impl_->engine->createExecutionContext();
    if (!impl_->context) {
        impl_->destroy();
        throw std::runtime_error("Failed to create TensorRT execution context");
    }

    // ---- Resolve binding indices by name ----

    auto resolve_binding = [&](const char* name) -> int {
        int idx = impl_->engine->getBindingIndex(name);
        if (idx < 0) {
            impl_->destroy();
            throw std::runtime_error(
                std::string("TensorRT engine missing expected binding: ") + name);
        }
        return idx;
    };

    impl_->bindings.images     = resolve_binding("images");
    impl_->bindings.intrinsics = resolve_binding("intrinsics");
    impl_->bindings.extrinsics = resolve_binding("extrinsics");
    impl_->bindings.positions  = resolve_binding("positions");
    impl_->bindings.scales     = resolve_binding("scales");
    impl_->bindings.rotations  = resolve_binding("rotations");
    impl_->bindings.opacities  = resolve_binding("opacities");
    impl_->bindings.sh_coeffs  = resolve_binding("sh_coeffs");

    // ---- Set dynamic input dimensions ----
    //
    // The batch dimension in the ONNX export is marked dynamic.
    // We fix it to 1 here.  num_views, H, W come from config.

    {
        nvinfer1::Dims dims_images;
        dims_images.nbDims = 5;
        dims_images.d[0] = 1;
        dims_images.d[1] = K;
        dims_images.d[2] = 3;
        dims_images.d[3] = H;
        dims_images.d[4] = W;
        impl_->context->setBindingDimensions(impl_->bindings.images, dims_images);
    }
    {
        nvinfer1::Dims dims_intr;
        dims_intr.nbDims = 4;
        dims_intr.d[0] = 1;
        dims_intr.d[1] = K;
        dims_intr.d[2] = 3;
        dims_intr.d[3] = 3;
        impl_->context->setBindingDimensions(impl_->bindings.intrinsics, dims_intr);
    }
    {
        nvinfer1::Dims dims_extr;
        dims_extr.nbDims = 4;
        dims_extr.d[0] = 1;
        dims_extr.d[1] = K;
        dims_extr.d[2] = 4;
        dims_extr.d[3] = 4;
        impl_->context->setBindingDimensions(impl_->bindings.extrinsics, dims_extr);
    }

    // Verify that all input dimensions are now fully specified.
    if (!impl_->context->allInputDimensionsSpecified()) {
        impl_->destroy();
        throw std::runtime_error(
            "TensorRT: not all input dimensions specified after setting shapes. "
            "Ensure the engine was built with matching dynamic axis configuration.");
    }

    // ---- Create CUDA stream ----

    if (config.external_stream) {
        impl_->stream = config.external_stream;
        impl_->owns_stream = false;
    } else {
        CUDA_CHECK(cudaStreamCreate(&impl_->stream));
        impl_->owns_stream = true;
    }

    // ---- Allocate GPU buffers ----

    auto sz = [](auto... dims) -> size_t {
        return (static_cast<size_t>(dims) * ... * sizeof(float));
    };

    impl_->buf_images     = GpuBuffer(sz(1, K, 3, H, W));
    impl_->buf_intrinsics = GpuBuffer(sz(1, K, 3, 3));
    impl_->buf_extrinsics = GpuBuffer(sz(1, K, 4, 4));

    impl_->buf_positions  = GpuBuffer(sz(N,  3));
    impl_->buf_scales     = GpuBuffer(sz(N,  3));
    impl_->buf_rotations  = GpuBuffer(sz(N,  4));
    impl_->buf_opacities  = GpuBuffer(sz(N,  1));
    impl_->buf_sh_coeffs  = GpuBuffer(sz(N, 48));

    impl_->buf_view_rgb   = GpuBuffer(sz(3, H, W));

    // ---- Build the binding pointer array ----

    int num_bindings = impl_->engine->getNbBindings();
    impl_->binding_ptrs.resize(static_cast<size_t>(num_bindings), nullptr);

    impl_->binding_ptrs[impl_->bindings.images]     = impl_->buf_images.get();
    impl_->binding_ptrs[impl_->bindings.intrinsics] = impl_->buf_intrinsics.get();
    impl_->binding_ptrs[impl_->bindings.extrinsics] = impl_->buf_extrinsics.get();
    impl_->binding_ptrs[impl_->bindings.positions]  = impl_->buf_positions.get();
    impl_->binding_ptrs[impl_->bindings.scales]     = impl_->buf_scales.get();
    impl_->binding_ptrs[impl_->bindings.rotations]  = impl_->buf_rotations.get();
    impl_->binding_ptrs[impl_->bindings.opacities]  = impl_->buf_opacities.get();
    impl_->binding_ptrs[impl_->bindings.sh_coeffs]  = impl_->buf_sh_coeffs.get();

    // ---- Pre-allocate host staging for calibration upload ----

    impl_->host_intrinsics.resize(static_cast<size_t>(K) * 9);
    impl_->host_extrinsics.resize(static_cast<size_t>(K) * 16);
}

// ============================================================================
// Destructor / move operations
// ============================================================================

TrtGaussianInference::~TrtGaussianInference() {
    if (impl_) {
        impl_->destroy();
    }
}

TrtGaussianInference::TrtGaussianInference(TrtGaussianInference&&) noexcept = default;
TrtGaussianInference& TrtGaussianInference::operator=(TrtGaussianInference&&) noexcept = default;

// ============================================================================
// Inference
// ============================================================================

GaussianFrame TrtGaussianInference::infer(
    const std::vector<CameraInput>& cameras,
    const std::vector<CameraCalibration>& calibrations)
{
    const int K = impl_->config.num_views;
    const int H = impl_->config.input_height;
    const int W = impl_->config.input_width;
    const int N = impl_->config.max_gaussians;

    // ---- Validate inputs ----

    if (static_cast<int>(cameras.size()) != K) {
        std::ostringstream oss;
        oss << "Expected " << K << " camera views, got " << cameras.size();
        throw std::runtime_error(oss.str());
    }
    if (static_cast<int>(calibrations.size()) != K) {
        std::ostringstream oss;
        oss << "Expected " << K << " calibrations, got " << calibrations.size();
        throw std::runtime_error(oss.str());
    }

    cudaStream_t stream = impl_->stream;

    // ---- Stage 1: Preprocess images ----
    //
    // For each of K views, convert the source RGBA float32 image from CameraInput
    // to planar RGB float32 at the model's input resolution, writing directly into
    // the contiguous images tensor [1, K, 3, H, W].
    //
    // Tensor memory layout (float32, row-major):
    //   view k occupies offsets [k*3*H*W .. (k+1)*3*H*W).

    const size_t view_floats = static_cast<size_t>(3) * H * W;
    const size_t view_bytes  = view_floats * sizeof(float);

    for (int k = 0; k < K; k++) {
        const CameraInput& cam = cameras[k];

        if (!cam.gpu_rgba_f32) {
            std::ostringstream oss;
            oss << "Camera " << k << " (index " << cam.camera_index
                << ") has null gpu_rgba_f32 pointer";
            throw std::runtime_error(oss.str());
        }

        // Resize RGBA -> planar RGB into the scratch buffer.
        launch_rgba_to_rgb_resize(
            static_cast<const float*>(cam.gpu_rgba_f32),
            static_cast<float*>(impl_->buf_view_rgb.get()),
            cam.height, cam.width,
            H, W,
            stream
        );

        // Copy the result into the correct slice of the images tensor (D2D, async).
        float* dst = static_cast<float*>(impl_->buf_images.get())
                     + static_cast<size_t>(k) * view_floats;
        CUDA_CHECK(cudaMemcpyAsync(
            dst,
            impl_->buf_view_rgb.get(),
            view_bytes,
            cudaMemcpyDeviceToDevice,
            stream
        ));
    }

    // ---- Stage 2: Upload calibration data ----
    //
    // Pack intrinsics [1, K, 3, 3] and extrinsics [1, K, 4, 4] into contiguous
    // host staging buffers, then async-copy to device.

    for (int k = 0; k < K; k++) {
        std::memcpy(
            impl_->host_intrinsics.data() + static_cast<size_t>(k) * 9,
            calibrations[k].intrinsic,
            9 * sizeof(float)
        );
        std::memcpy(
            impl_->host_extrinsics.data() + static_cast<size_t>(k) * 16,
            calibrations[k].extrinsic,
            16 * sizeof(float)
        );
    }

    CUDA_CHECK(cudaMemcpyAsync(
        impl_->buf_intrinsics.get(),
        impl_->host_intrinsics.data(),
        static_cast<size_t>(K) * 9 * sizeof(float),
        cudaMemcpyHostToDevice,
        stream
    ));
    CUDA_CHECK(cudaMemcpyAsync(
        impl_->buf_extrinsics.get(),
        impl_->host_extrinsics.data(),
        static_cast<size_t>(K) * 16 * sizeof(float),
        cudaMemcpyHostToDevice,
        stream
    ));

    // ---- Stage 3: Run TensorRT inference ----

    bool ok = impl_->context->enqueueV2(
        impl_->binding_ptrs.data(), stream, nullptr);
    if (!ok) {
        throw std::runtime_error(
            "TensorRT enqueueV2 failed. Check engine compatibility and input shapes.");
    }

    // ---- Stage 4: Determine actual output count ----
    //
    // If the engine uses a dynamic output dimension (e.g. data-dependent pruning),
    // the execution context reports the realized shape after inference. Otherwise
    // this returns the static shape, which we clamp to max_gaussians.

    int num_output = N;
    {
        nvinfer1::Dims pos_dims = impl_->context->getBindingDimensions(
            impl_->bindings.positions);
        if (pos_dims.nbDims >= 1 && pos_dims.d[0] > 0) {
            num_output = pos_dims.d[0];
        }
    }
    num_output = std::min(num_output, N);

    if (num_output <= 0) {
        GaussianFrame frame{};
        frame.is_keyframe = true;
        return frame;
    }

    // ---- Stage 5: Synchronize and copy outputs to host ----

    CUDA_CHECK(cudaStreamSynchronize(stream));

    const auto n = static_cast<size_t>(num_output);

    std::vector<float> h_positions (n * 3);
    std::vector<float> h_scales    (n * 3);
    std::vector<float> h_rotations (n * 4);
    std::vector<float> h_opacities (n);
    std::vector<float> h_sh_coeffs (n * 48);

    CUDA_CHECK(cudaMemcpy(h_positions.data(),  impl_->buf_positions.get(),
                           n *  3 * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_scales.data(),     impl_->buf_scales.get(),
                           n *  3 * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_rotations.data(),  impl_->buf_rotations.get(),
                           n *  4 * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_opacities.data(),  impl_->buf_opacities.get(),
                           n *  1 * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_sh_coeffs.data(),  impl_->buf_sh_coeffs.get(),
                           n * 48 * sizeof(float), cudaMemcpyDeviceToHost));

    // ---- Stage 6: Pack SoA output tensors into AoS GaussianFrame ----

    GaussianFrame frame;
    frame.frame_id = 0;        // Caller should overwrite with actual frame id.
    frame.timestamp_ns = 0;    // Caller should overwrite with actual timestamp.
    frame.num_gaussians = num_output;
    frame.is_keyframe = true;  // Feed-forward output is always a full reconstruction.
    frame.gaussians.resize(n);

    for (size_t i = 0; i < n; i++) {
        Gaussian& g = frame.gaussians[i];

        g.position[0] = h_positions[i * 3 + 0];
        g.position[1] = h_positions[i * 3 + 1];
        g.position[2] = h_positions[i * 3 + 2];

        g.scale[0] = h_scales[i * 3 + 0];
        g.scale[1] = h_scales[i * 3 + 1];
        g.scale[2] = h_scales[i * 3 + 2];

        g.rotation[0] = h_rotations[i * 4 + 0]; // w
        g.rotation[1] = h_rotations[i * 4 + 1]; // x
        g.rotation[2] = h_rotations[i * 4 + 2]; // y
        g.rotation[3] = h_rotations[i * 4 + 3]; // z

        g.opacity = h_opacities[i];

        std::memcpy(g.sh, &h_sh_coeffs[i * 48], 48 * sizeof(float));
    }

    return frame;
}

// ============================================================================
// Accessors
// ============================================================================

int TrtGaussianInference::num_views()     const { return impl_->config.num_views; }
int TrtGaussianInference::input_height()  const { return impl_->config.input_height; }
int TrtGaussianInference::input_width()   const { return impl_->config.input_width; }
int TrtGaussianInference::max_gaussians() const { return impl_->config.max_gaussians; }

} // namespace heimdall::gaussian
