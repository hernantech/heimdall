#include "cuda_ipc_bridge.h"
#include <stdexcept>
#include <sstream>
#include <iomanip>
#include <cstring>

namespace heimdall::capture {

CudaIpcBridge::SharedBuffer CudaIpcBridge::allocate(size_t size_bytes) {
    SharedBuffer buf{};
    buf.size_bytes = size_bytes;

    cudaError_t err = cudaMalloc(&buf.device_ptr, size_bytes);
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("cudaMalloc failed: ") + cudaGetErrorString(err));
    }

    err = cudaIpcGetMemHandle(&buf.ipc_handle, buf.device_ptr);
    if (err != cudaSuccess) {
        cudaFree(buf.device_ptr);
        throw std::runtime_error(std::string("cudaIpcGetMemHandle failed: ") + cudaGetErrorString(err));
    }

    return buf;
}

void* CudaIpcBridge::import_handle(const cudaIpcMemHandle_t& handle) {
    void* ptr = nullptr;
    cudaError_t err = cudaIpcOpenMemHandle(&ptr, handle, cudaIpcMemLazyEnablePeerAccess);
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("cudaIpcOpenMemHandle failed: ") + cudaGetErrorString(err));
    }
    return ptr;
}

void CudaIpcBridge::close_imported(void* imported_ptr) {
    cudaIpcCloseMemHandle(imported_ptr);
}

void CudaIpcBridge::free_buffer(SharedBuffer& buf) {
    if (buf.device_ptr) {
        cudaFree(buf.device_ptr);
        buf.device_ptr = nullptr;
    }
}

std::string CudaIpcBridge::handle_to_string(const cudaIpcMemHandle_t& handle) {
    std::ostringstream oss;
    for (size_t i = 0; i < sizeof(handle.reserved); ++i) {
        oss << std::hex << std::setfill('0') << std::setw(2)
            << static_cast<unsigned>(static_cast<unsigned char>(handle.reserved[i]));
    }
    return oss.str();
}

cudaIpcMemHandle_t CudaIpcBridge::string_to_handle(const std::string& s) {
    cudaIpcMemHandle_t handle;
    if (s.size() != sizeof(handle.reserved) * 2) {
        throw std::runtime_error("Invalid IPC handle string length");
    }
    for (size_t i = 0; i < sizeof(handle.reserved); ++i) {
        unsigned val;
        std::istringstream(s.substr(i * 2, 2)) >> std::hex >> val;
        handle.reserved[i] = static_cast<char>(val);
    }
    return handle;
}

} // namespace heimdall::capture
