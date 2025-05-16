#pragma once
#include <memory>
#include <cuda_bf16.h>

/**
 *  GPT-NeoX Style Rotary Positional Embeddings, see https://nn.labml.ai/transformers/rope/index.html
 */
class RoPE {
public:
    /**
     * Apply RoPE in-place
     * @param x queries or keys, of shape (num_heads, head_dim)
     * @param num_heads Number of query/key heads
     * @param head_dim Elements per query/key
     * @param position_idx Current position in sequence
     * @param theta_base RoPE parameter
     * @param stream CUDA stream for asynchronous operation
     */
    static void apply_rope_to_qk(__nv_bfloat16 *x, int32_t num_heads, int32_t head_dim,
                                 int32_t position_idx, float theta_base, cudaStream_t stream);
};
