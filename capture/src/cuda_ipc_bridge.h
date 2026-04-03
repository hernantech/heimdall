#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include <string>

namespace heimdall::capture {

// Zero-copy GPU memory sharing between processes on the same host.
// Allocates CUDA memory in one process and exports an IPC handle
// that another process can import to access the same GPU buffer
// without any CPU-side copy.
class CudaIpcBridge {
public:
    struct SharedBuffer {
        void* device_ptr;
        size_t size_bytes;
        cudaIpcMemHandle_t ipc_handle;
    };

    // Allocate GPU memory and export an IPC handle.
    static SharedBuffer allocate(size_t size_bytes);

    // Import a shared buffer from another process's IPC handle.
    static void* import_handle(const cudaIpcMemHandle_t& handle);

    // Release an imported pointer.
    static void close_imported(void* imported_ptr);

    // Free an allocated shared buffer.
    static void free_buffer(SharedBuffer& buf);

    // Serialize IPC handle to a hex string (for passing over gRPC/shared memory).
    static std::string handle_to_string(const cudaIpcMemHandle_t& handle);

    // Deserialize hex string back to IPC handle.
    static cudaIpcMemHandle_t string_to_handle(const std::string& s);
};

} // namespace heimdall::capture
