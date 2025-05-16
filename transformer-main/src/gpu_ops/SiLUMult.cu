#include "SiLUMult.cuh"
#include <cuda_bf16.h>
#include "../ErrorCheck.h"

void SiLUMult::silu_mult_in_place(const std::shared_ptr<CudaBuffer> &x, const std::shared_ptr<CudaBuffer> &y, cudaStream_t stream) {
    // TODO
}