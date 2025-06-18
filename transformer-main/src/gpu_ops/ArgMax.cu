#include "ArgMax.cuh"
#include <cuda_bf16.h>
#include "../ErrorCheck.h"

ArgMax::ArgMax(int32_t len) {
    size_t temp_size = len * sizeof(float) + sizeof(int32_t);
    temp_space = std::make_shared<CudaBuffer>(temp_size);
}

__global__ void argmax_kernel(__nv_bfloat16 *data, float *temp_values, int32_t *temp_indices, int32_t *result, int32_t len) {
    extern __shared__ float sdata[];
    int32_t *sindices = (int32_t*)&sdata[blockDim.x];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load data into shared memory
    if (idx < len) {
        sdata[tid] = __bfloat162float(data[idx]);
        sindices[tid] = idx;
    } else {
        sdata[tid] = -INFINITY;
        sindices[tid] = -1;
    }
    
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && idx + s < len) {
            if (sdata[tid + s] > sdata[tid] || 
                (sdata[tid + s] == sdata[tid] && sindices[tid + s] < sindices[tid])) {
                sdata[tid] = sdata[tid + s];
                sindices[tid] = sindices[tid + s];
            }
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        temp_values[blockIdx.x] = sdata[0];
        temp_indices[blockIdx.x] = sindices[0];
    }
}

__global__ void argmax_final_kernel(float *temp_values, int32_t *temp_indices, int32_t *result, int32_t num_blocks) {
    float max_val = -INFINITY;
    int32_t max_idx = -1;
    
    for (int i = 0; i < num_blocks; i++) {
        if (temp_values[i] > max_val || 
            (temp_values[i] == max_val && temp_indices[i] < max_idx)) {
            max_val = temp_values[i];
            max_idx = temp_indices[i];
        }
    }
    
    *result = max_idx;
}

int32_t *ArgMax::bf16_argmax(const std::shared_ptr<CudaBuffer> &bf16_data, cudaStream_t stream) {
    __nv_bfloat16 *data = static_cast<__nv_bfloat16*>(bf16_data->data);
    int32_t len = bf16_data->size / sizeof(__nv_bfloat16);
    
    int32_t block_size = 256;
    int32_t grid_size = (len + block_size - 1) / block_size;
    
    float *temp_values = static_cast<float*>(temp_space->data);
    int32_t *temp_indices = reinterpret_cast<int32_t*>(temp_values + grid_size);
    int32_t *result = temp_indices + grid_size;
    
    size_t shared_mem_size = block_size * (sizeof(float) + sizeof(int32_t));
    argmax_kernel<<<grid_size, block_size, shared_mem_size, stream>>>(
        data, temp_values, temp_indices, result, len);
    
    argmax_final_kernel<<<1, 1, 0, stream>>>(temp_values, temp_indices, result, grid_size);
    
    return result;
}
