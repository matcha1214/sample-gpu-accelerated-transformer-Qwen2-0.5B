#pragma once

#include <cuda_bf16.h>

class MatrixVectorMultiply {
public:
    /**
     * BF16 Matrix vector multiplication, with optional bias vector to add to the result.
     * @tparam input_float_t Input vector type, either float or __nv_bfloat16. Only affects `vec`, other parameters are still bf16.
     * @param m Rows in matrix/number of elements in output vector
     * @param k Columns in matrix/number of elements in input vector
     * @param mat Row-major bf16 matrix
     * @param bias Bias bf16 vector, null for 0 bias
     * @param vec Input vector
     * @param out Output bf16 vector
     * @param stream CUDA stream
     */
    template<typename input_float_t>
    static void bf16_matmul(int32_t m, int32_t k, __nv_bfloat16 *mat, __nv_bfloat16 *bias, input_float_t *vec, __nv_bfloat16 *out, cudaStream_t stream);
};

// explicitly instantiate in cu file
extern template void MatrixVectorMultiply::bf16_matmul<__nv_bfloat16>(int32_t m, int32_t k, __nv_bfloat16 *mat, __nv_bfloat16 *bias, __nv_bfloat16 *vec, __nv_bfloat16 *out, cudaStream_t stream);
extern template void MatrixVectorMultiply::bf16_matmul<float>(int32_t m, int32_t k, __nv_bfloat16 *mat, __nv_bfloat16 *bias, float *vec, __nv_bfloat16 *out, cudaStream_t stream);
