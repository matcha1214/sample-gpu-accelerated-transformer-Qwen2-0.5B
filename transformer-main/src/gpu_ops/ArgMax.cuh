#pragma once

#include "../CudaBuffer.cuh"
#include <memory>

/**
 * Returns the output index of the maximum value of a bfloat16 GPU array.
 * If there are multiple maximum values, return the one with lower index.
 */
class ArgMax {
    std::shared_ptr<CudaBuffer> temp_space;
public:
    /**
     * Initialize temporary space
     */
    explicit ArgMax(int32_t len);

    /**
     * Queues the argmax kernels on the stream.
     * @param bf16_data GPU bf16 values
     * @param stream CUDA stream to execute kernels on asynchronously
     * @return pointer in GPU memory to the index. Not valid until cudaStreamSynchronize() is called later.
     */
    int32_t *bf16_argmax(const std::shared_ptr<CudaBuffer> &bf16_data, cudaStream_t stream);
};