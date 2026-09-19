#ifndef VISIONFLOW_CUDA_INTERNAL_CUH
#define VISIONFLOW_CUDA_INTERNAL_CUH

#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>
#include "visionflow_cuda_errors.h"

namespace visionflow_cuda {

inline int runtime_error(cudaError_t error) {
    if (error == cudaSuccess) return VF_CUDA_OK;
    // A failed runtime call (for example cudaMalloc OOM) also stores a thread-local
    // last error. Consume it here so the next kernel launch check cannot report the
    // already-returned failure again; sticky context errors still persist.
    (void)cudaGetLastError();
    return VF_CUDA_RUNTIME_ERROR_BASE + static_cast<int>(error);
}

inline bool valid_image(
    const uint8_t* src,
    int width,
    int height,
    int stride,
    int channels) {
    return src != nullptr && width > 0 && height > 0 && channels > 0 &&
        stride >= width * channels;
}

inline int allocate_bytes(uint8_t** device, std::size_t bytes) {
    if (device == nullptr || bytes == 0) return VF_CUDA_INVALID_ARGUMENT;
    cudaError_t error = cudaMalloc(device, bytes);
    return error == cudaSuccess ? VF_CUDA_OK : runtime_error(error);
}

inline int allocate_and_upload(
    const uint8_t* host,
    int width,
    int height,
    int stride,
    int channels,
    uint8_t** device) {
    if (!valid_image(host, width, height, stride, channels) || device == nullptr) {
        return VF_CUDA_INVALID_ARGUMENT;
    }
    const std::size_t row_bytes = static_cast<std::size_t>(width) * channels;
    int result = allocate_bytes(device, row_bytes * height);
    if (result != VF_CUDA_OK) return result;
    cudaError_t error = cudaMemcpy2D(
        *device, row_bytes, host, stride, row_bytes, height, cudaMemcpyHostToDevice);
    if (error != cudaSuccess) {
        cudaFree(*device);
        *device = nullptr;
        return runtime_error(error);
    }
    return VF_CUDA_OK;
}

inline int download_and_free(
    uint8_t* host,
    int stride,
    int width,
    int height,
    int channels,
    uint8_t* device) {
    if (!valid_image(host, width, height, stride, channels) || device == nullptr) {
        if (device != nullptr) cudaFree(device);
        return VF_CUDA_INVALID_ARGUMENT;
    }
    const std::size_t row_bytes = static_cast<std::size_t>(width) * channels;
    cudaError_t error = cudaMemcpy2D(
        host, stride, device, row_bytes, row_bytes, height, cudaMemcpyDeviceToHost);
    cudaFree(device);
    return runtime_error(error);
}

inline int kernel_result() {
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) return runtime_error(error);
    return runtime_error(cudaDeviceSynchronize());
}

inline int kernel_launch_result() {
    return runtime_error(cudaGetLastError());
}

inline int stream_result(cudaStream_t stream) {
    return runtime_error(cudaStreamSynchronize(stream));
}

inline void free_device(void* pointer) {
    if (pointer != nullptr) cudaFree(pointer);
}

}  // namespace visionflow_cuda

#endif
