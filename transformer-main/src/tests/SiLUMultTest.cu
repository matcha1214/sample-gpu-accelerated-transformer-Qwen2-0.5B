#include "../CudaBuffer.cuh"
#include <memory>
#include <cuda_bf16.h>
#include <random>

#include "TestUtils.cuh"
#include "../gpu_ops/SiLUMult.cuh"
#include "../HostBuffer.h"

int main() {
    int32_t len = 12345;
    auto x = std::make_shared<CudaBuffer>(len * sizeof(__nv_bfloat16));
    __nv_bfloat16 *x_bf16 = static_cast<__nv_bfloat16*>(x->data);
    auto y = std::make_shared<CudaBuffer>(len * sizeof(__nv_bfloat16));
    __nv_bfloat16 *y_bf16 = static_cast<__nv_bfloat16*>(y->data);
    auto out_cpu = std::make_shared<HostBuffer>(len * sizeof(__nv_bfloat16));
    __nv_bfloat16 *out_cpu_bf16 = static_cast<__nv_bfloat16*>(out_cpu->data);

    std::mt19937 generator{123};
    std::normal_distribution distribution(0.0f, 10.0f);
    for (int32_t i = 0; i < len; i++) {
        __nv_bfloat16 x_val = distribution(generator);
        __nv_bfloat16 y_val = distribution(generator);
        x_bf16[i] = x_val;
        y_bf16[i] = y_val;
        float x_val_fp32 = x_val;
        float y_val_fp32 = y_val;
        out_cpu_bf16[i] = x_val_fp32 / (1.0f + expf(-x_val_fp32)) * y_val_fp32;
    }

    SiLUMult::silu_mult_in_place(x, y, cudaStreamPerThread);
    cudaStreamSynchronize(cudaStreamPerThread);

    check_bf16_allclose(x_bf16, out_cpu_bf16, len);
}
