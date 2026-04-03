// Background segmentation / matting engine implementation.
//
// Build:
//   # 1. Compile the CUDA preprocessing kernel (nvcc):
//   nvcc -std=c++17 -c matting_preprocess.cu -o matting_preprocess.o \
//        -I/usr/local/cuda/include
//
//   # 2. Compile this file (g++ or nvcc):
//   g++ -std=c++17 -c matting.cpp -o matting.o \
//       -I/usr/local/cuda/include -I/path/to/TensorRT/include
//
//   # 3. Link:
//   g++ matting.o matting_preprocess.o -o ... \
//       -L/path/to/TensorRT/lib -lnvinfer \
//       -L/usr/local/cuda/lib64 -lcudart

#include "matting.h"
#include "matting_preprocess.h"

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
#include <unordered_map>
#include <vector>

namespace heimdall::segmentation {

// ============================================================================
// TensorRT logger
// ============================================================================

class TrtLogger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity == Severity::kINFO || severity == Severity::kVERBOSE) return;

        const char* prefix = "";
        switch (severity) {
            case Severity::kINTERNAL_ERROR: prefix = "[MATTING TRT INTERNAL ERROR] "; break;
            case Severity::kERROR:          prefix = "[MATTING TRT ERROR] ";          break;
            case Severity::kWARNING:        prefix = "[MATTING TRT WARNING] ";        break;
            case Severity::kINFO:           prefix = "[MATTING TRT INFO] ";           break;
            case Severity::kVERBOSE:        prefix = "[MATTING TRT VERBOSE] ";        break;
        }
        std::cerr << prefix << msg << std::endl;
    }
};

// ============================================================================
// CUDA error checking
// ============================================================================

#define CUDA_CHECK(call)                                                        \
    do {                                                                         \
        cudaError_t err = (call);                                                \
        if (err != cudaSuccess) {                                                \
            std::ostringstream oss;                                              \
            oss << "CUDA error at " << __FILE__ << ":" << __LINE__               \
                << " -- " << cudaGetErrorString(err);                            \
            throw std::runtime_error(oss.str());                                 \
        }                                                                        \
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

    GpuBuffer(const GpuBuffer&) = delete;
    GpuBuffer& operator=(const GpuBuffer&) = delete;

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
            cudaFree(ptr_);
            ptr_ = nullptr;
            size_ = 0;
        }
    }

    void* ptr_ = nullptr;
    size_t size_ = 0;
};

// ============================================================================
// Per-camera state (temporal alpha + output buffer)
// ============================================================================

struct CameraState {
    GpuBuffer prev_alpha_f32;   // [cam_h, cam_w] float32, previous frame's alpha
    GpuBuffer output_alpha_u8;  // [cam_h, cam_w] uint8, final output
    int width  = 0;
    int height = 0;
    bool has_prev = false;      // Whether prev_alpha_f32 has valid data.

    // Ensure buffers are allocated for the given resolution.
    void ensure_allocated(int w, int h) {
        if (w == width && h == height) return;

        size_t pixels = static_cast<size_t>(w) * h;
        prev_alpha_f32  = GpuBuffer(pixels * sizeof(float));
        output_alpha_u8 = GpuBuffer(pixels * sizeof(unsigned char));
        width = w;
        height = h;
        has_prev = false;
    }
};

// ============================================================================
// Binding indices
// ============================================================================

struct MattingBindings {
    int input     = -1;   // [B, C, H, W] where C=3 (RVM) or C=6 (BMV2)
    int alpha_out = -1;   // [B, 1, H, W]
    int fgr_out   = -1;   // [B, 3, H, W] (BMV2 only, may be -1 for RVM)
};

// ============================================================================
// Impl (pimpl idiom)
// ============================================================================

struct MattingEngine::Impl {
    MattingConfig config;

    // TensorRT objects.
    TrtLogger                    logger;
    nvinfer1::IRuntime*          runtime = nullptr;
    nvinfer1::ICudaEngine*       engine  = nullptr;
    nvinfer1::IExecutionContext* context = nullptr;

    // CUDA stream.
    cudaStream_t stream      = nullptr;
    bool         owns_stream = false;

    MattingBindings bindings;

    // Number of input channels: 3 for RVM, 6 for BMV2.
    int input_channels = 3;

    // Pre-allocated GPU buffers (reused every frame).
    GpuBuffer buf_input;        // [1, C, H, W] float32 — model input tensor
    GpuBuffer buf_alpha_out;    // [1, 1, H, W] float32 — model alpha output
    GpuBuffer buf_fgr_out;      // [1, 3, H, W] float32 — model foreground output (BMV2)

    // Intermediate: full-resolution alpha in float32 before U8 conversion.
    GpuBuffer buf_alpha_fullres;  // [cam_h, cam_w] float32 (resized to camera resolution)

    // Morphology scratch buffer.
    GpuBuffer buf_morph_temp;     // [cam_h, cam_w] float32

    // Track the current full-res allocation size to avoid re-allocating when
    // the same camera resolution is used repeatedly.
    int fullres_w = 0;
    int fullres_h = 0;

    // Per-camera temporal state and output buffers, keyed by camera_index.
    std::unordered_map<int, CameraState> camera_states;

    // Binding pointer array for enqueueV2().
    std::vector<void*> binding_ptrs;

    // Ensure full-resolution scratch buffers are large enough.
    void ensure_fullres_buffers(int w, int h) {
        if (w == fullres_w && h == fullres_h) return;

        size_t pixels = static_cast<size_t>(w) * h;
        buf_alpha_fullres = GpuBuffer(pixels * sizeof(float));
        buf_morph_temp    = GpuBuffer(pixels * sizeof(float));
        fullres_w = w;
        fullres_h = h;
    }

    void destroy() {
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
        throw std::runtime_error("Failed to open model file: " + path);
    }

    auto size = file.tellg();
    if (size <= 0) {
        throw std::runtime_error("Model file is empty or unreadable: " + path);
    }

    file.seekg(0, std::ios::beg);
    std::vector<char> buffer(static_cast<size_t>(size));
    if (!file.read(buffer.data(), static_cast<std::streamsize>(size))) {
        throw std::runtime_error("Failed to read model file: " + path);
    }

    return buffer;
}

// ============================================================================
// Detect file type by extension
// ============================================================================

static bool ends_with(const std::string& str, const std::string& suffix) {
    if (suffix.size() > str.size()) return false;
    return str.compare(str.size() - suffix.size(), suffix.size(), suffix) == 0;
}

// ============================================================================
// Constructor
// ============================================================================

MattingEngine::MattingEngine(const MattingConfig& config)
    : impl_(std::make_unique<Impl>())
{
    impl_->config = config;

    const int H = config.inference_height;
    const int W = config.inference_width;

    if (H <= 0 || W <= 0) {
        throw std::runtime_error(
            "Invalid inference resolution: " + std::to_string(W) + "x" + std::to_string(H));
    }

    impl_->input_channels = (config.model_type == MattingModelType::BMV2) ? 6 : 3;

    // ---- Load the TensorRT engine ----
    //
    // We expect a pre-serialized .trt engine file. If the user provides an
    // .onnx file, the expectation is that it has been pre-converted to TRT
    // using trtexec or the export script. We detect by extension and provide
    // a helpful error message.

    if (ends_with(config.model_path, ".onnx")) {
        // Build engine from ONNX at runtime using TensorRT's ONNX parser.
        // This is slower to start but convenient during development.
        //
        // For production, pre-convert with:
        //   trtexec --onnx=model.onnx --saveEngine=model.trt --fp16
        //
        // Runtime ONNX parsing requires linking libnvonnxparser.
        // We defer to the dedicated builder to keep this code simpler.
        throw std::runtime_error(
            "Direct ONNX loading is not supported at runtime. "
            "Please convert to a TensorRT engine first:\n"
            "  trtexec --onnx=" + config.model_path + " --saveEngine=model.trt --fp16\n"
            "Or use export_matting_model.py --export-trt");
    }

    std::vector<char> engine_data = read_engine_file(config.model_path);

    impl_->runtime = nvinfer1::createInferRuntime(impl_->logger);
    if (!impl_->runtime) {
        throw std::runtime_error("Failed to create TensorRT runtime");
    }

    impl_->engine = impl_->runtime->deserializeCudaEngine(
        engine_data.data(), engine_data.size());
    if (!impl_->engine) {
        impl_->destroy();
        throw std::runtime_error(
            "Failed to deserialize TensorRT engine from: " + config.model_path);
    }

    impl_->context = impl_->engine->createExecutionContext();
    if (!impl_->context) {
        impl_->destroy();
        throw std::runtime_error("Failed to create TensorRT execution context");
    }

    // ---- Resolve binding indices ----
    //
    // Naming conventions:
    //   RVM:  input = "input", output alpha = "alpha" or "pha"
    //   BMV2: input = "src" (6-channel), output alpha = "pha", foreground = "fgr"
    //
    // We try multiple names to be robust across model export variations.

    auto try_binding = [&](const std::vector<std::string>& names) -> int {
        for (const auto& name : names) {
            int idx = impl_->engine->getBindingIndex(name.c_str());
            if (idx >= 0) return idx;
        }
        return -1;
    };

    impl_->bindings.input = try_binding({"input", "src", "images"});
    if (impl_->bindings.input < 0) {
        impl_->destroy();
        throw std::runtime_error(
            "TensorRT engine missing input binding. "
            "Expected one of: 'input', 'src', 'images'");
    }

    impl_->bindings.alpha_out = try_binding({"alpha", "pha", "output", "alpha_out"});
    if (impl_->bindings.alpha_out < 0) {
        impl_->destroy();
        throw std::runtime_error(
            "TensorRT engine missing alpha output binding. "
            "Expected one of: 'alpha', 'pha', 'output', 'alpha_out'");
    }

    if (config.model_type == MattingModelType::BMV2) {
        impl_->bindings.fgr_out = try_binding({"fgr", "foreground", "fgr_out"});
        // Foreground output is optional — some BMV2 variants only output alpha.
    }

    // ---- Set dynamic input dimensions [1, C, H, W] ----

    {
        nvinfer1::Dims dims_input;
        dims_input.nbDims = 4;
        dims_input.d[0] = 1;  // batch size = 1
        dims_input.d[1] = impl_->input_channels;
        dims_input.d[2] = H;
        dims_input.d[3] = W;
        impl_->context->setBindingDimensions(impl_->bindings.input, dims_input);
    }

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

    auto float_bytes = [](auto... dims) -> size_t {
        return (static_cast<size_t>(dims) * ... * sizeof(float));
    };

    const int C = impl_->input_channels;

    impl_->buf_input     = GpuBuffer(float_bytes(1, C, H, W));
    impl_->buf_alpha_out = GpuBuffer(float_bytes(1, 1, H, W));

    if (config.model_type == MattingModelType::BMV2 && impl_->bindings.fgr_out >= 0) {
        impl_->buf_fgr_out = GpuBuffer(float_bytes(1, 3, H, W));
    }

    // ---- Build the binding pointer array ----

    int num_bindings = impl_->engine->getNbBindings();
    impl_->binding_ptrs.resize(static_cast<size_t>(num_bindings), nullptr);

    impl_->binding_ptrs[impl_->bindings.input]     = impl_->buf_input.get();
    impl_->binding_ptrs[impl_->bindings.alpha_out] = impl_->buf_alpha_out.get();

    if (impl_->bindings.fgr_out >= 0) {
        impl_->binding_ptrs[impl_->bindings.fgr_out] = impl_->buf_fgr_out.get();
    }
}

// ============================================================================
// Destructor / move operations
// ============================================================================

MattingEngine::~MattingEngine() {
    if (impl_) {
        impl_->destroy();
    }
}

MattingEngine::MattingEngine(MattingEngine&&) noexcept = default;
MattingEngine& MattingEngine::operator=(MattingEngine&&) noexcept = default;

// ============================================================================
// process_frame
// ============================================================================

MattingOutput MattingEngine::process_frame(const MattingInput& input) {
    const int H = impl_->config.inference_height;
    const int W = impl_->config.inference_width;
    const int cam_w = input.width;
    const int cam_h = input.height;
    cudaStream_t stream = impl_->stream;

    // ---- Validate input ----

    if (!input.gpu_rgba_f32) {
        std::ostringstream oss;
        oss << "Camera " << input.camera_index << " has null gpu_rgba_f32 pointer";
        throw std::runtime_error(oss.str());
    }

    if (cam_w <= 0 || cam_h <= 0) {
        std::ostringstream oss;
        oss << "Camera " << input.camera_index << " has invalid dimensions: "
            << cam_w << "x" << cam_h;
        throw std::runtime_error(oss.str());
    }

    if (impl_->config.model_type == MattingModelType::BMV2) {
        if (!input.gpu_background_rgba_f32) {
            std::ostringstream oss;
            oss << "Camera " << input.camera_index
                << ": BMV2 model requires a background image (gpu_background_rgba_f32 is null)";
            throw std::runtime_error(oss.str());
        }
        if (input.bg_width <= 0 || input.bg_height <= 0) {
            std::ostringstream oss;
            oss << "Camera " << input.camera_index
                << ": BMV2 background has invalid dimensions: "
                << input.bg_width << "x" << input.bg_height;
            throw std::runtime_error(oss.str());
        }
    }

    // ---- Ensure per-camera state and full-res scratch buffers ----

    auto& cam_state = impl_->camera_states[input.camera_index];
    cam_state.ensure_allocated(cam_w, cam_h);
    impl_->ensure_fullres_buffers(cam_w, cam_h);

    // ---- Stage 1: Preprocess ----
    //
    // Resize source image to model resolution and convert RGBA -> planar RGB/6ch.
    // Write directly into the model input buffer.

    if (impl_->config.model_type == MattingModelType::BMV2) {
        launch_matting_preprocess_with_bg(
            static_cast<const float*>(input.gpu_rgba_f32),
            cam_h, cam_w,
            static_cast<const float*>(input.gpu_background_rgba_f32),
            input.bg_height, input.bg_width,
            static_cast<float*>(impl_->buf_input.get()),
            H, W,
            stream
        );
    } else {
        launch_matting_preprocess(
            static_cast<const float*>(input.gpu_rgba_f32),
            static_cast<float*>(impl_->buf_input.get()),
            cam_h, cam_w,
            H, W,
            stream
        );
    }

    // ---- Stage 2: TensorRT inference ----

    bool ok = impl_->context->enqueueV2(
        impl_->binding_ptrs.data(), stream, nullptr);
    if (!ok) {
        throw std::runtime_error(
            "Matting TensorRT inference failed for camera " +
            std::to_string(input.camera_index));
    }

    // ---- Stage 3: Postprocess ----
    //
    // The model output is [1, 1, H, W] alpha at model resolution.
    // We need to:
    //   a) Resize to original camera resolution
    //   b) Apply threshold (if configured)
    //   c) Apply morphological opening (if configured)
    //   d) EMA blend with previous frame (if configured)
    //   e) Convert F32 -> U8

    float* alpha_model = static_cast<float*>(impl_->buf_alpha_out.get());
    float* alpha_full  = static_cast<float*>(impl_->buf_alpha_fullres.get());

    // (a) Resize alpha from model resolution to camera resolution.
    launch_alpha_resize(
        alpha_model, alpha_full,
        H, W,
        cam_h, cam_w,
        stream
    );

    // (b) Threshold.
    if (impl_->config.threshold > 0.0f) {
        launch_alpha_threshold(
            alpha_full, cam_h, cam_w,
            impl_->config.threshold,
            stream
        );
    }

    // (c) Morphological opening (erode then dilate).
    if (impl_->config.enable_morphology && impl_->config.morphology_kernel_size >= 3) {
        launch_morphology_open(
            alpha_full,
            static_cast<float*>(impl_->buf_morph_temp.get()),
            cam_h, cam_w,
            impl_->config.morphology_kernel_size,
            stream
        );
    }

    // (d) Temporal EMA blending.
    if (impl_->config.temporal_alpha > 0.0f && cam_state.has_prev) {
        launch_temporal_ema(
            alpha_full,
            static_cast<const float*>(cam_state.prev_alpha_f32.get()),
            cam_h, cam_w,
            impl_->config.temporal_alpha,
            stream
        );
    }

    // Save current alpha as the previous frame for next temporal blend.
    if (impl_->config.temporal_alpha > 0.0f) {
        size_t alpha_bytes = static_cast<size_t>(cam_w) * cam_h * sizeof(float);
        CUDA_CHECK(cudaMemcpyAsync(
            cam_state.prev_alpha_f32.get(),
            alpha_full,
            alpha_bytes,
            cudaMemcpyDeviceToDevice,
            stream
        ));
        cam_state.has_prev = true;
    }

    // (e) Convert F32 [0,1] -> U8 [0,255].
    launch_alpha_f32_to_u8(
        alpha_full,
        static_cast<unsigned char*>(cam_state.output_alpha_u8.get()),
        cam_h, cam_w,
        stream
    );

    // Synchronize to ensure the output is ready before returning.
    CUDA_CHECK(cudaStreamSynchronize(stream));

    MattingOutput output;
    output.camera_index = input.camera_index;
    output.width  = cam_w;
    output.height = cam_h;
    output.gpu_alpha_u8 = cam_state.output_alpha_u8.get();

    return output;
}

// ============================================================================
// process_batch
// ============================================================================

std::vector<MattingOutput> MattingEngine::process_batch(
    const std::vector<MattingInput>& inputs)
{
    std::vector<MattingOutput> outputs;
    outputs.reserve(inputs.size());

    // Process each camera sequentially through the TensorRT engine.
    //
    // True batched inference (B > 1 in a single enqueueV2 call) would require:
    //   1. All cameras at the same resolution
    //   2. Engine built with dynamic batch dimension
    //   3. Packing all preprocessed inputs into a single contiguous tensor
    //
    // For typical volumetric capture rigs (8-64 cameras at the same resolution),
    // sequential per-camera inference is simpler and avoids the large memory
    // allocation for the batched input tensor. The TensorRT engine execution
    // itself is the bottleneck, and the GPU is fully utilized per-camera.
    //
    // If profiling shows batch inference would help, we can extend this to
    // group cameras by resolution and batch within each group.

    for (const auto& input : inputs) {
        outputs.push_back(process_frame(input));
    }

    return outputs;
}

// ============================================================================
// Accessors
// ============================================================================

int MattingEngine::inference_width() const {
    return impl_->config.inference_width;
}

int MattingEngine::inference_height() const {
    return impl_->config.inference_height;
}

MattingModelType MattingEngine::model_type() const {
    return impl_->config.model_type;
}

// ============================================================================
// reset_temporal_state
// ============================================================================

void MattingEngine::reset_temporal_state() {
    for (auto& [cam_idx, state] : impl_->camera_states) {
        state.has_prev = false;
    }
}

} // namespace heimdall::segmentation
