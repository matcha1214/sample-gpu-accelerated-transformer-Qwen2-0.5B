#pragma once

#include <cuda_bf16.h>
#include <iostream>
#include <random>

static bool bf16close(__nv_bfloat16 a, __nv_bfloat16 b, float rtol, float atol) {
    float a_f = a;
    float b_f = b;
    return fabsf(a_f - b_f) <= atol + rtol * fabsf(b_f);
}

static void check_bf16_allclose(__nv_bfloat16 *gpu_vals, __nv_bfloat16 *cpu_vals, int32_t len, float rtol=1e-2, float atol=1e-4) {
    bool all_close = true;
    for (int32_t i = 0; i < len; i++) {
        if (!bf16close(gpu_vals[i], cpu_vals[i], rtol, atol)) {
            std::cerr << "difference at index " << i << ": GPU calculated " << float(gpu_vals[i])
                << ", CPU calculated " << float(cpu_vals[i]) << std::endl;
            all_close = false;
        }
    }
    if (!all_close) {
        std::exit(1);
    }
}

static bool fp32close(__nv_bfloat16 a, __nv_bfloat16 b, float rtol, float atol) {
    return fabsf(a - b) <= atol + rtol * fabsf(b);
}

static void check_fp32_allclose(float *gpu_vals, float *cpu_vals, int32_t len, float rtol=1e-5, float atol=1e-8) {
    bool all_close = true;
    for (int32_t i = 0; i < len; i++) {
        if (!fp32close(gpu_vals[i], cpu_vals[i], rtol, atol)) {
            std::cerr << "difference at index " << i << ": GPU calculated " << gpu_vals[i]
                << ", CPU calculated " << cpu_vals[i] << std::endl;
            all_close = false;
        }
    }
    if (!all_close) {
        std::exit(1);
    }
}

static void fill_random_bf16(CudaBuffer &buf, std::normal_distribution<float> &distribution, std::mt19937 &generator) {
    for (size_t i = 0; i < buf.size / sizeof(__nv_bfloat16); i++) {
        static_cast<__nv_bfloat16*>(buf.data)[i] = distribution(generator);
    }
}