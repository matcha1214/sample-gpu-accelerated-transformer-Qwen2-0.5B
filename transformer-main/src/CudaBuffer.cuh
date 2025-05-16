#pragma once

class HostBuffer;

// Fixed-size GPU device buffer, automatically free'd on destruction.
// Uses cudaMallocManaged, so that memory is also accessible from the host.
class CudaBuffer {
public:
    /// data previously allocated with cudaMallocManaged
    CudaBuffer(void *data, size_t size);
    /// Upload from host memory
    explicit CudaBuffer(const HostBuffer &host_buffer);
    /// Allocate new buffer
    explicit CudaBuffer(size_t size);
    ~CudaBuffer();

    void* data{};
    size_t size{};
};