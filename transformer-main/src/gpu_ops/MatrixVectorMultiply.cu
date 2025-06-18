#include "MatrixVectorMultiply.cuh"
#include "../ErrorCheck.h"

// First kernel: Each block calculates partial sums for a set of rows
template<typename input_float_t>
__global__ void matmul_kernel_partial_sums(int32_t m, int32_t k, __nv_bfloat16 *mat,
                                           input_float_t *vec, float *partial_results) {
    extern __shared__ char shared_mem[];
    __nv_bfloat16 *s_vec = (__nv_bfloat16*)shared_mem;
    
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int num_blocks = gridDim.x;
    
    // Step 1: Cooperatively load vector into shared memory
    for (int i = tid; i < k; i += blockDim.x) {
        if constexpr (std::is_same_v<input_float_t, __nv_bfloat16>) {
            s_vec[i] = vec[i];
        } else {
            s_vec[i] = __float2bfloat16(vec[i]);
        }
    }
    __syncthreads();
    
    // Each block processes a strided set of rows for better load balancing
    for (int row = bid; row < m; row += num_blocks) {
        float sum = 0.0f;
        
        // Each thread processes k/blockDim.x elements of this row
        for (int col = tid; col < k; col += blockDim.x) {
            // Load matrix element
            __nv_bfloat16 mat_val = mat[row * k + col];
            
            // Read corresponding vector value from shared memory
            __nv_bfloat16 vec_val = s_vec[col];
            
            // Perform elementwise multiplication and accumulate
            sum += __bfloat162float(mat_val) * __bfloat162float(vec_val);
        }
        
        // Use shared memory for block-level reduction
        __shared__ float s_reduction[1024]; // Assuming max 1024 threads per block
        s_reduction[tid] = sum;
        __syncthreads();
        
        // Parallel reduction within block
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                s_reduction[tid] += s_reduction[tid + stride];
            }
            __syncthreads();
        }
        
        // Write partial sum to global memory
        if (tid == 0) {
            partial_results[row * num_blocks + bid] = s_reduction[0];
        }
    }
}

// Second kernel: Reduce partial sums and apply bias
__global__ void reduce_partial_sums_kernel(int32_t m, int32_t num_blocks, 
                                           float *partial_results, __nv_bfloat16 *bias, 
                                           __nv_bfloat16 *out) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < m) {
        float sum = 0.0f;
        
        // Sum all partial results for this row
        for (int b = 0; b < num_blocks; b++) {
            sum += partial_results[row * num_blocks + b];
        }
        
        // Add bias if provided
        if (bias != nullptr) {
            sum += __bfloat162float(bias[row]);
        }
        
        // Write final result
        out[row] = __float2bfloat16(sum);
    }
}

// Simplified version for small matrices (unchanged)
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
    if (k > 1024 && m > 1024) {
        // Use the two-phase approach with cross-block reduction for larger matrices
        int32_t block_size = 256;
        
        // Limit the number of blocks for better occupancy
        int32_t num_blocks = min(32, (m + 31)/32); // Use at most 32 blocks per row
        
        // Calculate shared memory size for vector
        size_t shared_mem_size = k * sizeof(__nv_bfloat16);
        
        // Check shared memory limits (typically 48KB per block)
        if (shared_mem_size <= 48 * 1024) {
            // Allocate temporary storage for partial results
            float *partial_results;
            cudaMalloc(&partial_results, m * num_blocks * sizeof(float));
            
            // Launch first kernel to compute partial sums
            matmul_kernel_partial_sums<<<num_blocks, block_size, shared_mem_size, stream>>>(
                m, k, mat, vec, partial_results);
            checkCuda(cudaPeekAtLastError());
            
            // Launch second kernel to reduce partial sums and apply bias
            int reduce_block_size = 256;
            int reduce_grid_size = (m + reduce_block_size - 1) / reduce_block_size;
            
            reduce_partial_sums_kernel<<<reduce_grid_size, reduce_block_size, 0, stream>>>(
                m, num_blocks, partial_results, bias, out);
            checkCuda(cudaPeekAtLastError());
            
            // Free temporary storage
            cudaFree(partial_results);
        } else {
            // Fall back to simple version if vector too large for shared memory
            int32_t grid_size = (m + block_size - 1) / block_size;
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
