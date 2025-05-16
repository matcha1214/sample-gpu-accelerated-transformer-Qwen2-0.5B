#pragma once

#include <memory>
#include "../CudaBuffer.cuh"

class SiLUMult {
public:
    /**
     * Fused sigmoid linear unit and element-wise multiplication. Part of Swish Gate Linear Unit (SwiGLU), see https://arxiv.org/pdf/2002.05202.
     * Writes result in-place over x, such that:
     * x = x / (1 + exp(-x)) * y
     * @param x bf16 vector, calculated with gate_proj(ffn_input)
     * @param y bf16 vector, calculated with up_proj(ffn_input)
     * @param stream CUDA stream for asynchronous operation
     */
    static void silu_mult_in_place(const std::shared_ptr<CudaBuffer> &x, const std::shared_ptr<CudaBuffer> &y, cudaStream_t stream);
};