#include <memory>
#include "../CudaBuffer.cuh"
#include <random>
#include <cuda_bf16.h>
#include "../gpu_ops/MatrixVectorMultiply.cuh"
#include "TestUtils.cuh"

int main() {
    // when debugging, it may be helpful to lower dimensions
    int32_t m = 12345;
    int32_t k = 800;
    auto mat = std::make_shared<CudaBuffer>(m * k * sizeof(__nv_bfloat16));
    __nv_bfloat16 *mat_bf16 = static_cast<__nv_bfloat16*>(mat->data);
    auto in_vec = std::make_shared<CudaBuffer>(k * sizeof(__nv_bfloat16));
    __nv_bfloat16 *in_vec_bf16 = static_cast<__nv_bfloat16*>(in_vec->data);
    auto bias_vec = std::make_shared<CudaBuffer>(m * sizeof(__nv_bfloat16));
    __nv_bfloat16 *bias_vec_bf16 = static_cast<__nv_bfloat16*>(bias_vec->data);
    auto out_vec = std::make_shared<CudaBuffer>(m * sizeof(__nv_bfloat16));
    __nv_bfloat16 *out_vec_bf16 = static_cast<__nv_bfloat16*>(out_vec->data);

    // seeded random
    std::mt19937 generator{123};
    std::normal_distribution distribution(0.0f, 1.0f);
    for (int32_t i = 0; i < m * k; i++) {
        mat_bf16[i] = distribution(generator);
    }
    for (int32_t i = 0; i < k; i++) {
        in_vec_bf16[i] = distribution(generator);
    }

    __nv_bfloat16 cpu_out[m];
    for (int i = 0; i < m; i++) {
        __nv_bfloat16 *row = mat_bf16 + (i * k);
        // summing one by one will accumulate more floating point error than a tree-based reduction,
        // which makes the outputs between GPU and GPU differ very slightly in some cases
        float sum = bias_vec_bf16[i];
        for (int j = 0; j < k; j++) {
            sum += static_cast<float>(row[j]) * static_cast<float>(in_vec_bf16[j]);
        }
        cpu_out[i] = static_cast<__nv_bfloat16>(sum);
    }

    for (int32_t i = 0; i < m; i++) {
        // output should be overwritten
        out_vec_bf16[i] = __float2bfloat16(-1);
    }
    MatrixVectorMultiply::bf16_matmul(m, k, mat_bf16, bias_vec_bf16, in_vec_bf16, out_vec_bf16, cudaStreamPerThread);
    cudaStreamSynchronize(cudaStreamPerThread);

    check_bf16_allclose(out_vec_bf16, cpu_out, m);
}
