#include "LayerNorm.cuh"
#include <cuda_bf16.h>
#include "../ErrorCheck.h"

/**
 * First stage kernel: Compute sum of squares within each block
 * This kernel implements the first step of RMSNorm: calculating Σ(a_i²)
 * Each block processes a portion of the data using parallel reduction
 */
__global__ void rmsnorm_sum_squares_kernel(__nv_bfloat16 *input, float *block_sum_squares, int32_t len) {
    // Use dynamic shared memory to store each thread's squared value
    extern __shared__ float sdata[];
    
    int tid = threadIdx.x;  // Thread ID within the block
    int idx = blockIdx.x * blockDim.x + threadIdx.x;  // Global thread ID
    
    // Each thread loads one element and computes its square
    // If beyond array bounds, set to 0 (won't affect the sum)
    if (idx < len) {
        float val = __bfloat162float(input[idx]);  // Convert to float for precision
        sdata[tid] = val * val;  // Compute square - this is the key RMSNorm step
    } else {
        sdata[tid] = 0.0f;  // Out-of-bounds elements set to 0
    }
    
    __syncthreads();  // Ensure all threads completed data loading
    
    // Parallel reduction: sum all squared values within the block
    // This is a classic tree-based reduction pattern with O(log n) complexity
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];  // Add adjacent elements
        }
        __syncthreads();  // Synchronize after each reduction step
    }
    
    // Only thread 0 writes this block's result to global memory
    // This gives us the sum of squares for each block
    if (tid == 0) {
        block_sum_squares[blockIdx.x] = sdata[0];
    }
}

/**
 * Second stage kernel: Compute final RMS from all block results
 * This kernel completes the RMSNorm calculation: RMS = sqrt(mean of squares)
 * Since the number of blocks is typically small, we use a single thread
 */
__global__ void rmsnorm_compute_rms_kernel(float *block_sum_squares, float *global_rms, 
                                          int32_t num_blocks, int32_t total_len) {
    // Only one thread executes this since the number of blocks is relatively small
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        float total_sum_squares = 0.0f;
        
        // Sum all block results to get the global sum of squares
        for (int i = 0; i < num_blocks; i++) {
            total_sum_squares += block_sum_squares[i];
        }
        
        // Apply RMSNorm formula: RMS = sqrt(mean(squares)) = sqrt(Σ(a_i²)/n)
        float mean_squares = total_sum_squares / total_len;  // Mean of squared values
        float rms = sqrtf(mean_squares + EPS);  // Add epsilon to prevent division by zero
        *global_rms = rms;  // Store the final RMS value
    }
}

/**
 * Third stage kernel: Apply RMSNorm normalization
 * This kernel implements the final RMSNorm formula: ā_i = (a_i / RMS(a)) * g_i
 * Each thread processes one element, performing normalization and weight scaling
 */
__global__ void rmsnorm_normalize_kernel(__nv_bfloat16 *input, __nv_bfloat16 *weights, 
                                        __nv_bfloat16 *output, float rms, int32_t len) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < len) {
        // Read input value and corresponding weight
        float val = __bfloat162float(input[idx]);
        float weight = __bfloat162float(weights[idx]);
        
        // Apply RMSNorm formula: (a_i / RMS) * g_i
        // Note: No mean subtraction here - this is the key difference from LayerNorm
        float normalized = (val / rms) * weight;
        
        // Convert result back to bfloat16 and store
        output[idx] = __float2bfloat16(normalized);
    }
}

LayerNorm::LayerNorm(int32_t len) {
    // Allocate temporary space for RMSNorm's reduction operation
    // We need space for each block's sum of squares, plus the final RMS value
    int32_t block_size = 256;  // Process 256 elements per block - good balance between occupancy and shared memory
    int32_t num_blocks = (len + block_size - 1) / block_size;
    
    // Calculate temporary storage requirements:
    // 1. num_blocks floats to store each block's sum of squares
    // 2. 1 float to store the final RMS value
    size_t temp_size = num_blocks * sizeof(float) + sizeof(float);
    temp_space = std::make_shared<CudaBuffer>(temp_size);
}

void LayerNorm::normalize_hidden_state(const std::shared_ptr<CudaBuffer> &hidden_state, 
                                      const std::shared_ptr<CudaBuffer> &output, cudaStream_t stream) {
    // Get pointers to input data and calculate length
    __nv_bfloat16 *input = static_cast<__nv_bfloat16*>(hidden_state->data);
    __nv_bfloat16 *out = static_cast<__nv_bfloat16*>(output->data);
    __nv_bfloat16 *weight_ptr = static_cast<__nv_bfloat16*>(weights->data);
    int32_t len = hidden_state->size / sizeof(__nv_bfloat16);
    
    // Configure kernel launch parameters
    int32_t block_size = 256;  // Number of threads per block
    int32_t grid_size = (len + block_size - 1) / block_size;  // Number of blocks
    
    // Get pointers to temporary storage space
    float *block_sum_squares = static_cast<float*>(temp_space->data);
    float *global_rms = block_sum_squares + grid_size;  // RMS stored after block results
    
    // Stage 1: Compute sum of squares within each block
    // Shared memory size equals one block's worth of threads
    size_t shared_mem_size = block_size * sizeof(float);
    rmsnorm_sum_squares_kernel<<<grid_size, block_size, shared_mem_size, stream>>>(
        input, block_sum_squares, len);
    
    // Stage 2: Compute final RMS from all block results
    // This kernel needs only one thread, so we use <<<1, 1>>>
    rmsnorm_compute_rms_kernel<<<1, 1, 0, stream>>>(
        block_sum_squares, global_rms, grid_size, len);
    
    // Stage 3: Apply normalization
    // We need to ensure RMS calculation is complete before normalization
    // In production, this could be optimized using CUDA events or callbacks
    // But for clarity, we use explicit synchronization here
    checkCuda(cudaStreamSynchronize(stream));
    
    // Read RMS value from GPU to CPU, then pass to normalization kernel
    // Alternative: Keep RMS on GPU and pass pointer to kernel (more efficient)
    float rms_value;
    checkCuda(cudaMemcpy(&rms_value, global_rms, sizeof(float), cudaMemcpyDeviceToHost));
    
    // Stage 3: Apply RMSNorm normalization
    // Each thread processes one element, performing (a_i / RMS) * g_i calculation
    rmsnorm_normalize_kernel<<<grid_size, block_size, 0, stream>>>(
        input, weight_ptr, out, rms_value, len);
}
