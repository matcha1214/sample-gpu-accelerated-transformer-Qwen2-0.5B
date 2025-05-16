#include "MatrixVectorMultiply.cuh"
#include "../ErrorCheck.h"

template<typename input_float_t>
__global__ void matmul_kernel_block_iteration(int32_t m, int32_t k, __nv_bfloat16 *mat, 
                                             __nv_bfloat16 *bias, input_float_t *vec, 
                                             __nv_bfloat16 *out) {
    extern __shared__ char shared_mem[];
    __nv_bfloat16 *s_vec = (__nv_bfloat16*)shared_mem;
    float *s_partial_sums = (float*)&s_vec[k];
    
    int tid = threadIdx.x;
    int block_id = blockIdx.x;
    
    // Step 1: Cooperatively load vector into shared memory
    // Each thread loads k/blockDim.x elements
    for (int i = tid; i < k; i += blockDim.x) {
        if constexpr (std::is_same_v<input_float_t, __nv_bfloat16>) {
            s_vec[i] = vec[i];
        } else {
            s_vec[i] = __float2bfloat16(vec[i]);
        }
    }
    __syncthreads();
    
    // Step 2: Each block iterates over multiple rows
    // Calculate how many rows this block will process
    int rows_per_block = (m + gridDim.x - 1) / gridDim.x;
    int start_row = block_id * rows_per_block;
    int end_row = min(start_row + rows_per_block, m);
    
    // Process each row assigned to this block
    for (int row = start_row; row < end_row; row++) {
        // Step 3: Initialize shared memory for partial sums
        s_partial_sums[tid] = 0.0f;
        
        // Step 4: Each thread processes k/blockDim.x elements of this row
        for (int col = tid; col < k; col += blockDim.x) {
            // Load matrix element
            __nv_bfloat16 mat_val = mat[row * k + col];
            
            // Read corresponding vector value from shared memory
            __nv_bfloat16 vec_val = s_vec[col];
            
            // Perform elementwise multiplication and accumulate
            s_partial_sums[tid] += __bfloat162float(mat_val) * __bfloat162float(vec_val);
        }
        
        __syncthreads();
        
        // Step 5: Reduce across the block
        // Parallel reduction to sum all partial results
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                s_partial_sums[tid] += s_partial_sums[tid + stride];
            }
            __syncthreads();
        }
        
        // Step 6: Write the output sum to global memory
        if (tid == 0) {
            float result = s_partial_sums[0];
            // Add bias if provided
            if (bias != nullptr) {
                result += __bfloat162float(bias[row]);
            }
            out[row] = __float2bfloat16(result);
        }
        
        __syncthreads(); // Ensure write completes before next iteration
    }
}

// Optimized version for smaller matrices (simpler but less following the pattern)
template<typename input_float_t>
__global__ void matmul_kernel_simple(int32_t m, int32_t k, __nv_bfloat16 *mat, __nv_bfloat16 *bias, 
                                     input_float_t *vec, __nv_bfloat16 *out) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < m) {
        float sum = 0.0f;
        
        // Add bias if provided
        if (bias != nullptr) {
            sum = __bfloat162float(bias[row]);
        }
        
        // Compute dot product
        for (int col = 0; col < k; col++) {
            float mat_val = __bfloat162float(mat[row * k + col]);
            float vec_val;
            
            if constexpr (std::is_same_v<input_float_t, __nv_bfloat16>) {
                vec_val = __bfloat162float(vec[col]);
            } else {
                vec_val = vec[col];
            }
            
            sum += mat_val * vec_val;
        }
        
        out[row] = __float2bfloat16(sum);
    }
}

template<typename input_float_t>
void MatrixVectorMultiply::bf16_matmul(int32_t m, int32_t k, __nv_bfloat16 *mat, __nv_bfloat16 *bias, 
                                       input_float_t *vec, __nv_bfloat16 *out, cudaStream_t stream) {
    // Choose kernel based on problem size and characteristics
    
    if (k > 1024 && m > 1024) {
        // Use the block-iteration approach for larger matrices
        // This follows the suggested pattern exactly
        int32_t block_size = 256;
        int32_t grid_size = min((m + block_size - 1) / block_size, 65535); // Max grid size limit
        
        // Calculate shared memory requirements
        size_t vector_size = k * sizeof(__nv_bfloat16);
        size_t partial_sums_size = block_size * sizeof(float);
        size_t shared_mem_size = vector_size + partial_sums_size;
        
        // Check shared memory limits (typically 48KB per block)
        if (shared_mem_size <= 48 * 1024) {
            matmul_kernel_block_iteration<<<grid_size, block_size, shared_mem_size, stream>>>(
                m, k, mat, bias, vec, out);
        } else {
            // Fall back to simple version if vector too large for shared memory
            grid_size = (m + block_size - 1) / block_size;
            matmul_kernel_simple<<<grid_size, block_size, 0, stream>>>(
                m, k, mat, bias, vec, out);
        }
    } else {
        // Use simple version for smaller matrices
        int32_t block_size = 256;
        int32_t grid_size = (m + block_size - 1) / block_size;
        matmul_kernel_simple<<<grid_size, block_size, 0, stream>>>(
            m, k, mat, bias, vec, out);
    }
}

// explicit instantiations
template void MatrixVectorMultiply::bf16_matmul<__nv_bfloat16>(int32_t m, int32_t k, __nv_bfloat16 *mat, __nv_bfloat16* bias, __nv_bfloat16 *vec, __nv_bfloat16 *out, cudaStream_t stream);
template void MatrixVectorMultiply::bf16_matmul<float>(int32_t m, int32_t k, __nv_bfloat16 *mat, __nv_bfloat16* bias, float *vec, __nv_bfloat16 *out, cudaStream_t stream);
