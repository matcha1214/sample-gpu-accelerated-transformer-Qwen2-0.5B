#include "LayerNorm.cuh"
#include <cuda_bf16.h>
#include "../ErrorCheck.h"

LayerNorm::LayerNorm(int32_t len) {
    // Allocate temporary space for reduction
    // We need space for partial sums from each block, plus space for final result
    int32_t block_size = 256;
    int32_t grid_size = (len + block_size - 1) / block_size;
    size_t temp_size = grid_size * sizeof(float) + sizeof(float);
    temp_space = std::make_shared<CudaBuffer>(temp_size);
}

__global__ void layernorm_variance_kernel(__nv_bfloat16 *input, float *partial_sums, int32_t len) {
    extern __shared__ float sdata[];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load data and compute squared values
    if (idx < len) {
        float val = __bfloat162float(input[idx]);
        sdata[tid] = val * val;
    } else {
        sdata[tid] = 0.0f;
    }
    
    __syncthreads();
    
    // Reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    // Write block result to global memory
    if (tid == 0) {
        partial_sums[blockIdx.x] = sdata[0];
    }
}

__global__ void layernorm_variance_final_kernel(float *partial_sums, float *variance, int32_t num_blocks, int32_t len) {
    float sum = 0.0f;
    
    for (int i = 0; i < num_blocks; i++) {
        sum += partial_sums[i];
    }
    
    // Compute variance = sum(x^2) / n
    *variance = sum / static_cast<float>(len);
}

__global__ void layernorm_normalize_kernel(__nv_bfloat16 *input, __nv_bfloat16 *weights, 
                                          __nv_bfloat16 *output, float variance, int32_t len) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < len) {
        float input_val = __bfloat162float(input[idx]);
        float weight_val = __bfloat162float(weights[idx]);
        
        // Normalize: val / sqrt(variance + EPS) * weight
        float normalized = input_val / sqrtf(variance + LayerNorm::EPS) * weight_val;
        
        output[idx] = __float2bfloat16(normalized);
    }
}

void LayerNorm::normalize_hidden_state(const std::shared_ptr<CudaBuffer> &hidden_state, 
                                      const std::shared_ptr<CudaBuffer> &output, 
                                      cudaStream_t stream) {
    __nv_bfloat16 *input = static_cast<__nv_bfloat16*>(hidden_state->data);
    __nv_bfloat16 *out = static_cast<__nv_bfloat16*>(output->data);
    __nv_bfloat16 *weights_ptr = static_cast<__nv_bfloat16*>(weights->data);
    int32_t len = hidden_state->size / sizeof(__nv_bfloat16);
    
    // Calculate grid and block dimensions for variance computation
    int32_t block_size = 256;
    int32_t grid_size = (len + block_size - 1) / block_size;
    
    // Get pointers to temporary space
    float *partial_sums = static_cast<float*>(temp_space->data);
    float *variance = partial_sums + grid_size;
    
    // Launch variance computation kernel
    size_t shared_mem_size = block_size * sizeof(float);
    layernorm_variance_kernel<<<grid_size, block_size, shared_mem_size, stream>>>(
        input, partial_sums, len);
    
    // Launch final variance reduction kernel
    layernorm_variance_final_kernel<<<1, 1, 0, stream>>>(
        partial_sums, variance, grid_size, len);
    
    // Launch normalization kernel
    grid_size = (len + block_size - 1) / block_size;
    layernorm_normalize_kernel<<<grid_size, block_size, 0, stream>>>(
        input, weights_ptr, out, *variance, len);
}
