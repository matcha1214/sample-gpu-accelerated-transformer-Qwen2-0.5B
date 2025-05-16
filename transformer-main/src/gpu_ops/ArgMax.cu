#include "ArgMax.cuh"
#include <cuda_bf16.h>
#include "../ErrorCheck.h"

ArgMax::ArgMax(int32_t len) {
    // TODO
}

int32_t *ArgMax::bf16_argmax(const std::shared_ptr<CudaBuffer> &bf16_data, cudaStream_t stream) {
    // TODO
    return nullptr;
}
