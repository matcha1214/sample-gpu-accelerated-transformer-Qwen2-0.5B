#include "CudaBuffer.cuh"
#include "ErrorCheck.h"
#include "HostBuffer.h"

CudaBuffer::CudaBuffer(void *data, size_t size): data(data), size(size) {}

CudaBuffer::CudaBuffer(const HostBuffer &host_buffer): CudaBuffer(host_buffer.size) {
    checkCuda(cudaMemcpy(data, host_buffer.data, size, cudaMemcpyDefault));
}

CudaBuffer::CudaBuffer(size_t size) : size(size) {
    checkCuda(cudaMallocManaged(&data, size));
}

CudaBuffer::~CudaBuffer() {
    checkCuda(cudaFree(data));
}