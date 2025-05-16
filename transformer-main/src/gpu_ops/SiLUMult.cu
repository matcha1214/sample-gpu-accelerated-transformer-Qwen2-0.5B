#include "SiLUMult.cuh"
#include <cuda_bf16.h>
#include "../ErrorCheck.h"

/**
 * SiLUMult kernel: Fused SiLU activation and element-wise multiplication
 * 
 * This kernel implements the core of SwiGLU activation used in modern transformers:
 * result = SiLU(gate) * up = (gate / (1 + exp(-gate))) * up
 * 
 * Key optimizations:
 * 1. Coalesced memory access: threads in a warp access consecutive elements
 * 2. Fused operations: SiLU and multiplication in one kernel
 * 3. In-place computation: result overwrites the gate vector to save memory
 */
__global__ void silu_mult_kernel(__nv_bfloat16 *gate, __nv_bfloat16 *up, int32_t len) {
    // Calculate global thread index
    // This ensures coalesced memory access when threads in a warp
    // access consecutive elements gate[0], gate[1], gate[2], etc.
    int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Boundary check to prevent out-of-bounds access
    if (idx < len) {
        // Step 1: Load elements from both vectors (coalesced access)
        // Since threads in a warp have consecutive idx values,
        // they access consecutive memory locations, ensuring coalescing
        float gate_val = __bfloat162float(gate[idx]);
        float up_val = __bfloat162float(up[idx]);
        
        // Step 2: Apply SiLU (Swish) activation to the gate value
        // SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
        // This is mathematically equivalent to x * (1 / (1 + exp(-x)))
        // We compute it as x / (1 + exp(-x)) for better numerical stability
        float silu_gate = gate_val / (1.0f + expf(-gate_val));
        
        // Step 3: Element-wise multiplication with the up vector
        // This is the core operation: activated_gate * up_value
        float result = silu_gate * up_val;
        
        // Step 4: Write result back to memory (coalesced access)
        // We overwrite the gate vector to save memory
        // Again, consecutive threads write to consecutive locations
        gate[idx] = __float2bfloat16(result);
    }
}

void SiLUMult::silu_mult_in_place(const std::shared_ptr<CudaBuffer> &gate_proj_output, 
                                  const std::shared_ptr<CudaBuffer> &up_proj_output, 
                                  cudaStream_t stream) {
    // Extract raw pointers and calculate vector length
    __nv_bfloat16 *gate_ptr = static_cast<__nv_bfloat16*>(gate_proj_output->data);
    __nv_bfloat16 *up_ptr = static_cast<__nv_bfloat16*>(up_proj_output->data);
    int32_t len = gate_proj_output->size / sizeof(__nv_bfloat16);
    
    // Configure kernel launch parameters
    // Block size of 256 is chosen for several reasons:
    // 1. Good occupancy on most GPU architectures
    // 2. Allows multiple warps per block for better instruction-level parallelism
    // 3. Efficient use of shared memory and registers
    int32_t block_size = 256;
    int32_t grid_size = (len + block_size - 1) / block_size;
    
    // Launch the kernel
    // No shared memory needed since this is a straightforward elementwise operation
    silu_mult_kernel<<<grid_size, block_size, 0, stream>>>(gate_ptr, up_ptr, len);
}
