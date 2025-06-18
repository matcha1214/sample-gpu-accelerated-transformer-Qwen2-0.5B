#include "SiLUMult.cuh"
#include <cuda_bf16.h>
#include "../ErrorCheck.h"
#include <device_functions.h> // Required for __expf() and other intrinsics

/**
 * SiLUMult kernel: Fused SiLU activation and element-wise multiplication
 * * This kernel implements the core of SwiGLU activation used in modern transformers */
__global__ void silu_mult_kernel(__nv_bfloat16 *gate, __nv_bfloat16 *up, int32_t len) {
    // Calculate global thread index
    // This ensures coalesced memory access when threads in a warp
    // access consecutive elements gate[0], gate[1], gate[2], etc.
    int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Boundary check to prevent out-of-bounds access
    if (idx < len) {
        // Since threads in a warp have consecutive idx values,
        // they access consecutive memory locations, ensuring coalescing
        float gate_val = __bfloat162float(gate[idx]);
        float up_val = __bfloat162float(up[idx]);
        
        // SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
        // Using __expf for potentially lower latency.
        float silu_gate = gate_val / (1.0f + __expf(-gate_val)); // MODIFIED LINE
        
        // This is the core operation: activated_gate * up_value
        float result = silu_gate * up_val;
        
        // Overwrite the gate vector to save memory
        // Again, consecutive threads write to consecutive locations
        gate[idx] = __float2bfloat16(result);
    }
}

void SiLUMult::silu_mult_in_place(const std::shared_ptr<CudaBuffer> &gate_proj_output, 
                                  const std::shared_ptr<CudaBuffer> &up_proj_output, 
                                  cudaStream_t stream) {
    // Extract raw pointers and calculate vector length
    __nv_bfloat16 *gate_ptr = static_cast<__nv_bfloat16*>(gate_proj_output->data);
    __nv_bfloat16 *up_ptr = static_cast<__nv_bfloat16*>(up_proj_output->data); // Corrected to use up_proj_output
    int32_t len = gate_proj_output->size / sizeof(__nv_bfloat16);

    if (len == 0) return; // Handle empty input
    
    // Configure kernel launch parameters
    int32_t block_size = 256;
    // Calculate grid size ensuring all elements are covered
    int32_t grid_size = (len + block_size - 1) / block_size;
    
    // Launch the kernel
    silu_mult_kernel<<<grid_size, block_size, 0, stream>>>(gate_ptr, up_ptr, len);
}
