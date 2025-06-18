#pragma once

#include "../qwen2/Qwen2Config.h"
#include "../CudaBuffer.cuh"
#include <cuda_bf16.h>
#include <memory>
#include "../ErrorCheck.h"

template<Qwen2Size QWEN2_SIZE>
class GroupQueryAttention {
public:
    using Qwen2Config = Qwen2Config<QWEN2_SIZE>;

    /**
     * Allocate temporary space
     */
    explicit GroupQueryAttention(int32_t max_seq_len) {
        // Allocate space for attention weights (seq_len floats per query head)
        // and intermediate computations
        size_t temp_size = max_seq_len * Qwen2Config::num_query_heads() * sizeof(float);
        temp_space = std::make_shared<CudaBuffer>(temp_size);
    }

    /**
     * Scaled dot product attention with grouped queries, see https://arxiv.org/abs/2305.13245.
     * Performs softmax((QK^T)/sqrt(d_k))*V for all queries Q and their associated K and V
     * - dot product each query with its target value throughout the sequence
     * - numerically stable softmax
     * - save a weighted sum of values
     * Does not perform the output projection.
     *
     * All inputs and outputs are row-major
     *
     * @param queries (num_query_heads, head_size)
     * @param k_cache (seq_len, num_layers, num_kv_heads, key_size)
     * @param v_cache (seq_len, num_layers, num_kv_heads, value_size)
     * @param weighted_values (num_query_heads, value_size) outputs
     * @param layer_num layer index, starting at 0
     * @param seq_len current sequence length
     * @param stream CUDA stream for asynchronous operation
     */
    void sdpa(__nv_bfloat16 *queries, __nv_bfloat16 *k_cache, __nv_bfloat16 *v_cache, float *weighted_values, int32_t layer_num, int32_t seq_len, cudaStream_t stream);

private:
    std::shared_ptr<CudaBuffer> temp_space;
};

// Kernel for computing attention for a single query head
template<Qwen2Size QWEN2_SIZE>
__global__ void gqa_kernel(
    __nv_bfloat16 *queries,
    __nv_bfloat16 *k_cache, 
    __nv_bfloat16 *v_cache,
    float *weighted_values,
    float *temp_attention_weights,
    int32_t layer_num,
    int32_t seq_len
) {
    using Qwen2Config = Qwen2Config<QWEN2_SIZE>;
    
    int head_idx = blockIdx.x;
    int tid = threadIdx.x;
    int block_size = blockDim.x;
    
    if (head_idx >= Qwen2Config::num_query_heads()) return;
    
    // Determine which key-value head this query head uses
    // With 14 query heads and 2 KV heads: heads 0-6 use KV head 0, heads 7-13 use KV head 1
    int kv_head_idx = head_idx * Qwen2Config::num_kv_heads() / Qwen2Config::num_query_heads();
    
    // Use global memory for attention weights to avoid shared memory limitations
    float *attention_weights = temp_attention_weights + head_idx * seq_len;
    
    // Shared memory for reduction operations only
    extern __shared__ float sdata[];
    
    __nv_bfloat16 *query_head = queries + head_idx * Qwen2Config::head_size();
    
    // Each thread processes multiple sequence positions
    for (int pos = tid; pos < seq_len; pos += block_size) {
        // Get pointer to this key vector in the cache
        // Layout: k_cache[seq_pos][layer][kv_head][head_dim]
        __nv_bfloat16 *key_vec = k_cache + 
            pos * (Qwen2Config::num_layers() * Qwen2Config::keys_size()) +
            layer_num * Qwen2Config::keys_size() +
            kv_head_idx * Qwen2Config::head_size();
        
        // Compute dot product
        float dot_product = 0.0f;
        for (int i = 0; i < Qwen2Config::head_size(); i++) {
            dot_product += __bfloat162float(query_head[i]) * __bfloat162float(key_vec[i]);
        }
        
        // Scale by 1/sqrt(head_size) for attention
        float scaled_dot = dot_product / sqrtf(static_cast<float>(Qwen2Config::head_size()));
        attention_weights[pos] = scaled_dot;
    }
    __syncthreads();
    
    // Numerically stable softmax using online algorithm
    // Step 1: Find maximum value
    float thread_max = -INFINITY;
    for (int pos = tid; pos < seq_len; pos += block_size) {
        thread_max = fmaxf(thread_max, attention_weights[pos]);
    }
    
    // Reduction to find global maximum
    sdata[tid] = thread_max;
    __syncthreads();
    
    for (int stride = block_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + stride]);
        }
        __syncthreads();
    }
    
    float global_max = sdata[0];
    __syncthreads();
    
    // Step 2: Compute sum of exponentials
    float thread_sum = 0.0f;
    for (int pos = tid; pos < seq_len; pos += block_size) {
        thread_sum += expf(attention_weights[pos] - global_max);
    }
    
    // Reduction to find global sum
    sdata[tid] = thread_sum;
    __syncthreads();
    
    for (int stride = block_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }
    
    float global_sum = sdata[0];
    __syncthreads();
    
    // Step 3: Normalize to get probabilities
    for (int pos = tid; pos < seq_len; pos += block_size) {
        attention_weights[pos] = expf(attention_weights[pos] - global_max) / global_sum;
    }
    __syncthreads();
    
    // Each thread computes part of the output vector
    for (int dim = tid; dim < Qwen2Config::value_size(); dim += block_size) {
        float weighted_sum = 0.0f;
        
        for (int pos = 0; pos < seq_len; pos++) {
            // Get pointer to this value vector in the cache
            // Layout: v_cache[seq_pos][layer][kv_head][value_dim]
            __nv_bfloat16 *value_vec = v_cache + 
                pos * (Qwen2Config::num_layers() * Qwen2Config::values_size()) +
                layer_num * Qwen2Config::values_size() +
                kv_head_idx * Qwen2Config::value_size();
            
            weighted_sum += attention_weights[pos] * __bfloat162float(value_vec[dim]);
        }
        
        // Store result
        weighted_values[head_idx * Qwen2Config::value_size() + dim] = weighted_sum;
    }
}

template<Qwen2Size QWEN2_SIZE>
void GroupQueryAttention<QWEN2_SIZE>::sdpa(__nv_bfloat16 *queries, __nv_bfloat16 *k_cache, __nv_bfloat16 *v_cache, float *weighted_values, int32_t layer_num, int32_t seq_len, cudaStream_t stream) {
    // Launch one block per query head
    int num_blocks = Qwen2Config::num_query_heads();
    int block_size = 256;  // threads per block
    
    // Calculate shared memory requirements - only for reduction operations now
    size_t shared_mem_size = block_size * sizeof(float);
    
    // Get temporary space pointer
    float *temp_weights = static_cast<float*>(temp_space->data);
    
    gqa_kernel<QWEN2_SIZE><<<num_blocks, block_size, shared_mem_size, stream>>>(
        queries, k_cache, v_cache, weighted_values, temp_weights, layer_num, seq_len
    );
}