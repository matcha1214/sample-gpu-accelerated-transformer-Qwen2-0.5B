#include "RoPE.cuh"
#include <cuda_bf16.h>
#include "../ErrorCheck.h"

/**
 * RoPE kernel implementing GPT-NeoX style rotation
 * 
 * Mathematical background:
 * RoPE applies rotation to query/key vectors such that:
 * - Each dimension pair (i, i+d/2) is rotated by angle θᵢ * position
 * - θᵢ = base^(-2i/d) where base is typically 10000 or 1000000
 * - This creates position-dependent similarities in attention calculation
 */
__global__ void rope_kernel(__nv_bfloat16 *x, int32_t num_heads, int32_t head_dim,
                           int32_t position_idx, float theta_base) {
    // Each block processes one head, each thread processes one dimension pair
    int head_idx = blockIdx.x;
    int dim_pair_idx = threadIdx.x;  // which pair of dimensions (0, head_dim/2), (1, head_dim/2+1), etc.
    
    // Boundary check - ensure we don't exceed the number of heads or dimension pairs
    if (head_idx >= num_heads || dim_pair_idx >= head_dim / 2) return;
    
    // Calculate the base frequency for this dimension pair
    // GPT-NeoX style: θᵢ = theta_base^(-2*i/head_dim)
    // This makes lower dimensions rotate faster than higher dimensions
    float theta_idx_frac = (2.0f * static_cast<float>(dim_pair_idx)) / static_cast<float>(head_dim);
    float theta = powf(theta_base, -theta_idx_frac);
    
    // Calculate the actual rotation angle for this position
    // angle = θᵢ * position_idx
    float angle = theta * static_cast<float>(position_idx);
    
    // Compute rotation coefficients on-the-fly (following the slide's guidance)
    // This saves memory bandwidth compared to pre-computing and storing these values
    float cos_val = cosf(angle);
    float sin_val = sinf(angle);
    
    // Calculate memory indices for the dimension pair
    // GPT-NeoX style rotates (dim_i, dim_i+head_dim/2) pairs
    int base_idx = head_idx * head_dim;
    int idx1 = base_idx + dim_pair_idx;              // First dimension of the pair
    int idx2 = base_idx + dim_pair_idx + head_dim / 2; // Second dimension of the pair
    
    // Load the values to be rotated
    float x1 = __bfloat162float(x[idx1]);
    float x2 = __bfloat162float(x[idx2]);
    
    // Apply the 2D rotation transformation
    // This is the core RoPE operation: rotating in the (x1, x2) plane
    float rotated_x1 = x1 * cos_val - x2 * sin_val;
    float rotated_x2 = x1 * sin_val + x2 * cos_val;
    
    // Store the rotated values back
    x[idx1] = __float2bfloat16(rotated_x1);
    x[idx2] = __float2bfloat16(rotated_x2);
}

void RoPE::apply_rope_to_qk(__nv_bfloat16 *x, int32_t num_heads, int32_t head_dim,
                           int32_t position_idx, float theta_base, cudaStream_t stream) {
    // Grid configuration: one block per head
    // Block configuration: one thread per dimension pair
    // This gives us optimal parallelism - each head is processed independently,
    // and within each head, all dimension pairs are rotated simultaneously
    dim3 grid_dim(num_heads);
    dim3 block_dim(head_dim / 2);
    
    // Launch the kernel
    // Note: No shared memory needed since each thread works on independent data
    rope_kernel<<<grid_dim, block_dim, 0, stream>>>(
        x, num_heads, head_dim, position_idx, theta_base);
}
