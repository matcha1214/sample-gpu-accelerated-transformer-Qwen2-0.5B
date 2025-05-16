#include "LayerNorm.cuh"
#include <cuda_bf16.h>
#include "../ErrorCheck.h"

LayerNorm::LayerNorm(int32_t len) {
    // No additional temporary space needed for this implementation
}

__global__ void layer_norm_kernel(__nv_bfloat16 *input, __nv_bfloat16 *weights, 
                                  __nv_bfloat16 *output, int32_t len, float eps) {
    extern __shared__ float sdata[];
    
    int tid = threadIdx.x;
    int block_size = blockDim.x;
    
    // Phase 1: Each thread accumulates sum of squares for multiple elements
    float thread_sum_sq = 0.0f;
    for (int i = tid; i < len; i += block_size) {
        float val = __bfloat162float(input[i]);
        thread_sum_sq += val * val;
    }
    
    // Store in shared memory for reduction
    sdata[tid] = thread_sum_sq;
    __syncthreads();
    
    // Reduction in shared memory to calculate total sum of squares
    for (int s = block_size / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    // Calculate variance (mean of squares, no mean subtraction)
    // T5-style: variance = mean(x^2) 
    __shared__ float sqrt_var;
    if (tid == 0) {
        float variance = sdata[0] / static_cast<float>(len);
        sqrt_var = sqrtf(variance + eps);
    }
    __syncthreads();
    
    // Phase 2: Each thread applies normalization to multiple elements
    // T5 formula: output[i] = weight[i] * input[i] / sqrt(variance + eps)
    for (int i = tid; i < len; i += block_size) {
        float input_val = __bfloat162float(input[i]);
        float weight_val = __bfloat162float(weights[i]);
        float normalized = weight_val * input_val / sqrt_var;
        output[i] = __float2bfloat16(normalized);
    }
}

void LayerNorm::normalize_hidden_state(const std::shared_ptr<CudaBuffer> &hidden_state, 
                                      const std::shared_ptr<CudaBuffer> &output, 
                                      cudaStream_t stream) {
    __nv_bfloat16 *input_ptr = static_cast<__nv_bfloat16*>(hidden_state->data);
    __nv_bfloat16 *weights_ptr = static_cast<__nv_bfloat16*>(weights->data);
    __nv_bfloat16 *output_ptr = static_cast<__nv_bfloat16*>(output->data);
    int32_t len = hidden_state->size / sizeof(__nv_bfloat16);
    
    // Use a single block with 512 threads
    // Each thread will process multiple elements in a strided pattern
    int block_size = 512;
    size_t shared_mem_size = block_size * sizeof(float);
    
    layer_norm_kernel<<<1, block_size, shared_mem_size, stream>>>(
        input_ptr, weights_ptr, output_ptr, len, EPS);
}
