#include "../ErrorCheck.h"
#include "../gpu_ops/ArgMax.cuh"
#include <random>
#include <cuda_bf16.h>

void test_argmax(int32_t num_els) {
    auto buf = std::make_shared<CudaBuffer>(num_els * sizeof(__nv_bfloat16));
    __nv_bfloat16* bf16s = static_cast<__nv_bfloat16*>(buf->data);

    // seeded random
    std::mt19937 generator{42};
    std::normal_distribution distribution(0.0f, 100.0f);
    int32_t max_index = -1;
    float max_val = -INFINITY;
    for (int32_t i = 0; i < num_els; i++) {
        __nv_bfloat16 val = __float2bfloat16(distribution(generator));
        bf16s[i] = val;
        // cast back to float incorporates rounding errors
        float val_f = __bfloat162float(val);
        if (val_f > max_val) {
            max_index = i;
            max_val = val_f;
        }
    }

    ArgMax argmax(num_els);
    cudaStream_t stream{};
    checkCuda(cudaStreamCreate(&stream));
    int32_t *calculated_index_ptr = argmax.bf16_argmax(buf, stream);
    checkCuda(cudaStreamSynchronize(stream));
    int32_t calculated_index = *calculated_index_ptr;
    checkCuda(cudaStreamDestroy(stream));

    if (calculated_index != max_index) {
        if (calculated_index < 0 || calculated_index >= num_els) {
            std::cerr << "got index " << calculated_index << " (out of range), "
                << "expected index " << max_index << " (value " << __bfloat162float(bf16s[max_index]) << ")" << std::endl;
        } else {
            std::cerr << "got index " << calculated_index << " (value " << __bfloat162float(bf16s[calculated_index])
                << "), expected index " << max_index << " (value " << __bfloat162float(bf16s[max_index]) << ")" << std::endl;
        }
        std::exit(1);
    }
}

int main() {
    test_argmax(1);
    test_argmax(1234);
    test_argmax(321234);
}