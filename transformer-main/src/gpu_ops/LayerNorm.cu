#include "LayerNorm.cuh"
#include <cuda_bf16.h>
#include "../ErrorCheck.h"

LayerNorm::LayerNorm(int32_t len) {
}

// Kernel 1: Compute partial sums of squares across multiple blocks
__global__ void layer_norm_partial_sums(__nv_bfloat16 *input, float *partial_sums, 
                                        int32_t len, int32_t num_blocks) {
    extern __shared__ float sdata[];
    
    int tid = threadIdx.x;
    int block_size = blockDim.x;
    int global_idx = blockIdx.x * block_size + tid;
    
    float thread_sum_sq = 0.0f;
    
    // Each thread accumulates its portion with strided access
    for (int i = global_idx; i < len; i += gridDim.x * block_size) {
        float val = __bfloat162float(input[i]);
        thread_sum_sq += val * val;
    }
    
    // Store in shared memory for block-level reduction
    sdata[tid] = thread_sum_sq;
    __syncthreads();
    
    // Block-level reduction in shared memory
    for (int s = block_size / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    // Block leader stores partial sum
    if (tid == 0) {
        partial_sums[blockIdx.x] = sdata[0];
    }
}

// Kernel 2: Final reduction and normalization
__global__ void layer_norm_finalize(__nv_bfloat16 *input, __nv_bfloat16 *weights, 
                                   __nv_bfloat16 *output, float *partial_sums, 
                                   int32_t num_partial_sums, int32_t len, float eps) {
    extern __shared__ float sdata[];
    
    int tid = threadIdx.x;
    int block_size = blockDim.x;
    
    // Block 0 performs final reduction to get global variance
    __shared__ float global_sqrt_var;
    
    if (blockIdx.x == 0) {
        // Load partial sums into shared memory for reduction
        float thread_sum = 0.0f;
        for (int i = tid; i < num_partial_sums; i += block_size) {
            thread_sum += partial_sums[i];
        }
        
        sdata[tid] = thread_sum;
        __syncthreads();
        
        // Reduce to get total sum of squares
        for (int s = block_size / 2; s > 0; s >>= 1) {
            if (tid < s) {
                sdata[tid] += sdata[tid + s];
            }
            __syncthreads();
        }
        
        // Calculate final variance and store it back to global memory
        if (tid == 0) {
            float total_sum_sq = sdata[0];
            float variance = total_sum_sq / static_cast<float>(len);
            global_sqrt_var = sqrtf(variance + eps);
            partial_sums[num_partial_sums] = global_sqrt_var; // Store for other blocks
        }
    }
    __syncthreads();
    
    // All blocks wait for variance calculation using simple spin lock
    if (blockIdx.x != 0) {
        if (tid == 0) {
            // Wait for block 0 to finish variance calculation
            volatile float *variance_ptr = &partial_sums[num_partial_sums];
            while (*variance_ptr == 0.0f) { /* spin wait */ }
            global_sqrt_var = *variance_ptr;
        }
        __syncthreads();
    }
    
    // All blocks normalize their assigned portions
    int global_idx = blockIdx.x * block_size + tid;
    for (int i = global_idx; i < len; i += gridDim.x * block_size) {
        float input_val = __bfloat162float(input[i]);
        float weight_val = __bfloat162float(weights[i]);
        float normalized = weight_val * input_val / global_sqrt_var;
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
    
    // **NEW: Multi-block implementation for better GPU utilization**
    int block_size = 512;
    int max_blocks = (len + block_size - 1) / block_size;
    int num_blocks = min(max_blocks, 108);  // Limit to available SMs on A100
    
    // Allocate temporary memory for partial sums dynamically
    float *partial_sums_ptr;
    size_t temp_size = (num_blocks + 1) * sizeof(float); // +1 for storing final variance
    cudaMallocAsync((void**)&partial_sums_ptr, temp_size, stream);
    
    // Initialize the variance storage location to 0
    cudaMemsetAsync(&partial_sums_ptr[num_blocks], 0, sizeof(float), stream);
    
    size_t shared_mem_size = block_size * sizeof(float);
    
    // Two-pass approach
    // Pass 1: Compute partial sums of squares
    layer_norm_partial_sums<<<num_blocks, block_size, shared_mem_size, stream>>>(
        input_ptr, partial_sums_ptr, len, num_blocks);
    
    // Pass 2: Final reduction and normalization  
    layer_norm_finalize<<<num_blocks, block_size, shared_mem_size, stream>>>(
        input_ptr, weights_ptr, output_ptr, partial_sums_ptr, num_blocks, len, EPS);
    
    // Free temporary memory
    cudaFreeAsync(partial_sums_ptr, stream);
}
