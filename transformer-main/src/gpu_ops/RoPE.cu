#include "RoPE.cuh"
#include <cuda_bf16.h>
#include "../ErrorCheck.h"

void RoPE::apply_rope_to_qk(__nv_bfloat16 *x, int32_t num_heads, int32_t head_dim,
        int32_t position_idx, float theta_base, cudaStream_t stream) {
    // TODO
}
