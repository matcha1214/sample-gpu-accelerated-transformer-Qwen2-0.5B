#include "../CudaBuffer.cuh"
#include <cuda_bf16.h>
#include <memory>
#include <random>
#include "../gpu_ops/LayerNorm.cuh"
#include "TestUtils.cuh"

int main() {
    int32_t hidden_dim = 8000;
    auto weights_vec = std::make_shared<CudaBuffer>(hidden_dim * sizeof(__nv_bfloat16));
    __nv_bfloat16 *weights_bf16 = static_cast<__nv_bfloat16*>(weights_vec->data);
    auto in_vec = std::make_shared<CudaBuffer>(hidden_dim * sizeof(__nv_bfloat16));
    __nv_bfloat16 *in_vec_bf16 = static_cast<__nv_bfloat16*>(in_vec->data);
    auto out_vec = std::make_shared<CudaBuffer>(hidden_dim * sizeof(__nv_bfloat16));
    __nv_bfloat16 *out_vec_bf16 = static_cast<__nv_bfloat16*>(out_vec->data);

    // seeded random
    std::mt19937 generator{123};
    std::normal_distribution distribution(0.0f, 1.0f);
    for (int32_t i = 0; i < hidden_dim; i++) {
        weights_bf16[i] = distribution(generator);
    }
    float cpu_variance_sum = 0.0f;
    for (int32_t i = 0; i < hidden_dim; i++) {
        in_vec_bf16[i] = distribution(generator);
        // cast back to float for proper precision loss
        float val_f{in_vec_bf16[i]};
        cpu_variance_sum += val_f * val_f;
    }
    // std::cerr << "cpu variance sum " << cpu_variance_sum << std::endl;
    float cpu_variance = sqrtf(cpu_variance_sum / static_cast<float>(hidden_dim) + LayerNorm::EPS);

    __nv_bfloat16 cpu_out[hidden_dim];
    for (int i = 0; i < hidden_dim; i++) {
        cpu_out[i] = float(weights_bf16[i]) * float(in_vec_bf16[i]) / cpu_variance;
    }

    for (int32_t i = 0; i < hidden_dim; i++) {
        // output should be overwritten
        out_vec_bf16[i] = __float2bfloat16(-1);
    }
    LayerNorm layer_norm(hidden_dim);
    layer_norm.weights = weights_vec;

    for (int run = 0; run < 2; run++) {
        // run twice to make sure that there are no issues with retaining state in temp storage
        layer_norm.normalize_hidden_state(in_vec, out_vec, cudaStreamPerThread);
        cudaStreamSynchronize(cudaStreamPerThread);
        check_bf16_allclose(out_vec_bf16, cpu_out, hidden_dim);
    }
}