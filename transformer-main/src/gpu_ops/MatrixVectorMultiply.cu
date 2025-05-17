#include "MatrixVectorMultiply.cuh"
#include "../ErrorCheck.h"

template<typename input_float_t>
__global__ void matmul_kernel_block_reduce(
    int32_t m, int32_t k, __nv_bfloat16 *mat, __nv_bfloat16 *bias,
    input_float_t *vec, float *partial_out
) {
    extern __shared__ char shared_mem[];
    __nv_bfloat16* s_vec = (__nv_bfloat16*)shared_mem;
    float* s_partial = (float*)&s_vec[k];

    int tid = threadIdx.x;
    int block_id = blockIdx.x;

    // 1. Load vector into shared memory (cooperatively)
    for (int i = tid; i < k; i += blockDim.x) {
        if constexpr (std::is_same_v<input_float_t, __nv_bfloat16>) {
            s_vec[i] = vec[i];
        } else {
            s_vec[i] = __float2bfloat16(vec[i]);
        }
    }
    __syncthreads();

    // 2. Each block processes a subset of rows
    int rows_per_block = (m + gridDim.x - 1) / gridDim.x;
    int start_row = block_id * rows_per_block;
    int end_row = min(start_row + rows_per_block, m);

    for (int row = start_row; row < end_row; row++) {
        float thread_sum = 0.0f;

        // 3. Each thread processes a chunk of 'k'
        for (int col = tid; col < k; col += blockDim.x) {
            __nv_bfloat16 mat_val = mat[row * k + col];
            __nv_bfloat16 vec_val = s_vec[col];
            thread_sum += __bfloat162float(mat_val) * __bfloat162float(vec_val);
        }

        // 4. Store thread_sum into shared memory for block reduction
        s_partial[tid] = thread_sum;
        __syncthreads();

        // 5. Parallel reduction within block
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                s_partial[tid] += s_partial[tid + stride];
            }
            __syncthreads();
        }

        // 6. Write partial sum to global memory
        if (tid == 0) {
            // Each block writes its partial sum for this row
            partial_out[row * gridDim.x + block_id] = s_partial[0];
        }
        __syncthreads();
    }
}

// Stage 2: Final reduction kernel (sum across blocks for each row)
__global__ void matmul_kernel_block_reduce_finalize(
    int32_t m, int32_t num_blocks, float *partial_out, __nv_bfloat16 *bias, __nv_bfloat16 *out
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < m) {
        float sum = 0.0f;
        for (int blk = 0; blk < num_blocks; ++blk) {
            sum += partial_out[row * num_blocks + blk];
        }
        if (bias != nullptr) {
            sum += __bfloat162float(bias[row]);
        }
        out[row] = __float2bfloat16(sum);
    }
}

template<typename input_float_t>
void MatrixVectorMultiply::bf16_matmul(
    int32_t m, int32_t k, __nv_bfloat16 *mat, __nv_bfloat16 *bias,
    input_float_t *vec, __nv_bfloat16 *out, cudaStream_t stream
) {
    // Use two-stage reduction for large matrices where more blocks are needed
    int block_size = 256;
    int max_blocks = 65535; // CUDA limit
    int grid_size = min((m + block_size - 1) / block_size, max_blocks);

    size_t vector_size = k * sizeof(__nv_bfloat16);
    size_t partial_sums_size = block_size * sizeof(float);
    size_t shared_mem_size = vector_size + partial_sums_size;

    if (k > 1024 && m > 1024 && shared_mem_size <= 48 * 1024) {
        // Stage 1: Compute partial sums
        float *partial_out;
        cudaMalloc(&partial_out, sizeof(float) * m * grid_size);

        matmul_kernel_block_reduce<input_float_t><<<grid_size, block_size, shared_mem_size, stream>>>(
            m, k, mat, bias, vec, partial_out
        );

        // Stage 2: Final reduction
        int red_grid = (m + block_size - 1) / block_size;
        matmul_kernel_block_reduce_finalize<<<red_grid, block_size, 0, stream>>>(
            m, grid_size, partial_out, bias, out
        );

        cudaFree(partial_out);
    } else {
        // Fallback to simple kernel for smaller matrices or memory constraints
        int grid = (m + block_size - 1) / block_size;
        matmul_kernel_simple<input_float_t><<<grid, block_size, 0, stream>>>(
            m, k, mat, bias, vec, out
        );
    }
}

// explicit instantiations
template void MatrixVectorMultiply::bf16_matmul<__nv_bfloat16>(int32_t m, int32_t k, __nv_bfloat16 *mat, __nv_bfloat16* bias, __nv_bfloat16 *vec, __nv_bfloat16 *out, cudaStream_t stream);
template void MatrixVectorMultiply::bf16_matmul<float>(int32_t m, int32_t k, __nv_bfloat16 *mat, __nv_bfloat16* bias, float *vec, __nv_bfloat16 *out, cudaStream_t stream);
