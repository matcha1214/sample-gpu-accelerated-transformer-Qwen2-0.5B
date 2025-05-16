#include "../CudaBuffer.cuh"
#include "../HostBuffer.h"
#include "../qwen2/Qwen2Config.h"
#include <random>
#include <cuda_bf16.h>
#include "TestUtils.cuh"
#include "../gpu_ops/GroupQueryAttention.cuh"

void test_gqa(int32_t max_seq_len, int32_t seq_len, int32_t layer_num) {
    constexpr Qwen2Size QWEN2_SIZE = QWEN2_0_5B;
    using Qwen2Config = Qwen2Config<QWEN2_SIZE>;

    CudaBuffer k_cache{max_seq_len * Qwen2Config::num_layers() * Qwen2Config::keys_size() * sizeof(__nv_bfloat16)};
    CudaBuffer v_cache{max_seq_len * Qwen2Config::num_layers() * Qwen2Config::values_size() * sizeof(__nv_bfloat16)};
    CudaBuffer queries{Qwen2Config::queries_size() * sizeof(__nv_bfloat16)};
    CudaBuffer out{Qwen2Config::num_query_heads() * Qwen2Config::value_size() * sizeof(float)};
    HostBuffer out_cpu{Qwen2Config::num_query_heads() * Qwen2Config::value_size() * sizeof(float)};

    std::mt19937 generator{123};
    std::normal_distribution distribution(0.0f, 100.0f);
    fill_random_bf16(k_cache, distribution, generator);
    fill_random_bf16(v_cache, distribution, generator);
    fill_random_bf16(queries, distribution, generator);

    // group query attention
    for (int32_t head_idx = 0; head_idx < Qwen2Config::num_query_heads(); head_idx++) {
        __nv_bfloat16 *head_query = static_cast<__nv_bfloat16*>(queries.data) + head_idx * Qwen2Config::head_size();
        // each key is used multiple times for different queries
        int32_t key_idx = head_idx * Qwen2Config::num_kv_heads() / Qwen2Config::num_query_heads();

        // one scaled dot product for each token in the sequence
        std::vector<float> scaled_dot_products;
        float scaled_dot_products_max = -INFINITY;

        // QK^T/sqrt(d_k)
        for (int32_t sequence_pos = 0; sequence_pos < seq_len; sequence_pos++) {
            float sum = 0.0f;
            for (int32_t el_idx = 0; el_idx < Qwen2Config::head_size(); el_idx++) {
                __nv_bfloat16 key_el = *(static_cast<__nv_bfloat16*>(k_cache.data) +
                    sequence_pos * (Qwen2Config::num_layers() * Qwen2Config::keys_size()) +
                    layer_num * Qwen2Config::keys_size() +
                    key_idx * Qwen2Config::head_size() +
                    el_idx
                );
                sum += float(key_el) * float(head_query[el_idx]);
            }
            float scaled_dot_product = sum / sqrtf(Qwen2Config::head_size());
            scaled_dot_products.push_back(scaled_dot_product);
            scaled_dot_products_max = fmaxf(scaled_dot_products_max, scaled_dot_product);
        }

        // numerically stable softmax denominator
        float d = 0.0f;
        for (float sdp : scaled_dot_products) {
            d += expf(sdp - scaled_dot_products_max);
        }

        // weighted sum of values
        float *out_row = static_cast<float*>(out_cpu.data) + head_idx * Qwen2Config::value_size();
        for (int32_t el_idx = 0; el_idx < Qwen2Config::value_size(); el_idx++) {
            float out_el_sum = 0.0f;
            for (int32_t sequence_pos = 0; sequence_pos < seq_len; sequence_pos++) {
                float val_scaling_factor = expf(scaled_dot_products[sequence_pos] - scaled_dot_products_max) / d;
                __nv_bfloat16 val_el = *(static_cast<__nv_bfloat16*>(v_cache.data) +
                    sequence_pos * (Qwen2Config::num_layers() * Qwen2Config::values_size()) +
                    layer_num * Qwen2Config::values_size() +
                    key_idx * Qwen2Config::value_size() +
                    el_idx
                );
                out_el_sum += val_scaling_factor * float(val_el);
            }
            out_row[el_idx] = out_el_sum;
        }
    }

    for (int run = 0; run < 2; run++) {
        // run twice to ensure that temp space is reused correctly
        GroupQueryAttention<QWEN2_SIZE> gqa{max_seq_len};
        gqa.sdpa(static_cast<__nv_bfloat16*>(queries.data),
            static_cast<__nv_bfloat16*>(k_cache.data),
            static_cast<__nv_bfloat16*>(v_cache.data),
            static_cast<float*>(out.data),
            layer_num, seq_len, cudaStreamPerThread);
        cudaStreamSynchronize(cudaStreamPerThread);

        check_fp32_allclose(static_cast<float*>(out.data), static_cast<float*>(out_cpu.data),
            Qwen2Config::num_query_heads() * Qwen2Config::value_size());
    }
}

int main() {
    test_gqa(1000, 732, 5);
    test_gqa(1000, 1, 0);
    test_gqa(100, 2, 23);
}
