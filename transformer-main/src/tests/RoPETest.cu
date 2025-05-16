#include <memory>
#include "../CudaBuffer.cuh"
#include <random>
#include <cuda_bf16.h>
#include "../gpu_ops/RoPE.cuh"
#include "TestUtils.cuh"

int main() {
    int32_t num_heads = 4;
    int32_t head_dim = 256; // head_dim must be even
    int32_t position_idx = 13;
    float theta_base = 1e6f;

    auto queries = std::make_shared<CudaBuffer>(num_heads * head_dim * sizeof(__nv_bfloat16));
    __nv_bfloat16 *queries_bf16 = static_cast<__nv_bfloat16*>(queries->data);

    // seeded random
    std::mt19937 generator{123};
    std::normal_distribution distribution(0.0f, 1.0f);
    for (int32_t i = 0; i < num_heads * head_dim; i++) {
        // queries_bf16[i] = i;
        queries_bf16[i] = distribution(generator);
    }

    float cos_vals[head_dim / 2];
    float sin_vals[head_dim / 2];
    for (int32_t theta_idx = 0; theta_idx < head_dim / 2; theta_idx++) {
        float theta_idx_frac = static_cast<float>(theta_idx) / static_cast<float>(head_dim / 2);
        float theta = powf(theta_base, -theta_idx_frac);
        float angle = theta * static_cast<float>(position_idx);
        cos_vals[theta_idx] = cosf(angle);
        sin_vals[theta_idx] = sinf(angle);
    }

    __nv_bfloat16 cpu_out[head_dim * num_heads];
    for (int32_t head = 0; head < num_heads; head++) {
        __nv_bfloat16 *cpu_out_row = cpu_out + head * head_dim;
        __nv_bfloat16 *in_row = queries_bf16 + head * head_dim;

        float in_row_rotated_half[head_dim];
        for (int32_t i = 0; i < head_dim / 2; i++) {
            in_row_rotated_half[i] = -in_row[i + head_dim / 2];
        }
        for (int32_t i = head_dim / 2; i < head_dim; i++) {
            in_row_rotated_half[i] = in_row[i - head_dim / 2];
        }

        for (int32_t i = 0; i < head_dim; i++) {
            float cos_val = cos_vals[i % (head_dim / 2)];
            float sin_val = sin_vals[i % (head_dim / 2)];
            cpu_out_row[i] = float(in_row[i]) * cos_val + float(in_row_rotated_half[i]) * sin_val;
        }
    }

    RoPE::apply_rope_to_qk(queries_bf16, num_heads, head_dim, position_idx, theta_base, cudaStreamPerThread);
    cudaStreamSynchronize(cudaStreamPerThread);

    check_bf16_allclose(queries_bf16, cpu_out, num_heads * head_dim);
}