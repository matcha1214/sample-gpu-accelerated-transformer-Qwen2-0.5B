#pragma once

#include <cuda_bf16.h>

#include "Qwen2Config.h"
#include "../CudaBuffer.cuh"
#include <memory>

#include "../gpu_ops/MatrixVectorMultiply.cuh"
#include "../gpu_ops/LayerNorm.cuh"
#include "../ErrorCheck.h"
#include "../gpu_ops/RoPE.cuh"
#include "../gpu_ops/GroupQueryAttention.cuh"
#include "../gpu_ops/SiLUMult.cuh"

// Custom kernel for element-wise addition (since we don't have an explicit MatrixVectorMultiply::add)
// Make it inline to avoid multiple definition errors
__global__ inline void elementwise_add_kernel(__nv_bfloat16* a, __nv_bfloat16* b, __nv_bfloat16* result, int32_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float a_val = __bfloat162float(a[idx]);
        float b_val = __bfloat162float(b[idx]);
        float sum = a_val + b_val;
        result[idx] = __float2bfloat16(sum);
    }
}

template<Qwen2Size QWEN2_SIZE>
class Qwen2Layer {
public:
    using Qwen2Config = Qwen2Config<QWEN2_SIZE>;

    Qwen2Layer(uint32_t layer_num, uint32_t max_seq_len):
    layer_num(layer_num), input_layernorm(Qwen2Config::hidden_size()), post_attention_layernorm(Qwen2Config::hidden_size()) {
        // Allocate memory for all the layer's weights and biases
        q_proj_weight = std::make_shared<CudaBuffer>(Qwen2Config::queries_size() * Qwen2Config::hidden_size() * sizeof(__nv_bfloat16));
        q_proj_bias = std::make_shared<CudaBuffer>(Qwen2Config::queries_size() * sizeof(__nv_bfloat16));
        
        k_proj_weight = std::make_shared<CudaBuffer>(Qwen2Config::keys_size() * Qwen2Config::hidden_size() * sizeof(__nv_bfloat16));
        k_proj_bias = std::make_shared<CudaBuffer>(Qwen2Config::keys_size() * sizeof(__nv_bfloat16));
        
        v_proj_weight = std::make_shared<CudaBuffer>(Qwen2Config::values_size() * Qwen2Config::hidden_size() * sizeof(__nv_bfloat16));
        v_proj_bias = std::make_shared<CudaBuffer>(Qwen2Config::values_size() * sizeof(__nv_bfloat16));
        
        o_proj_weight = std::make_shared<CudaBuffer>(Qwen2Config::hidden_size() * Qwen2Config::queries_size() * sizeof(__nv_bfloat16));
        
        // MLP weights
        up_proj_weight = std::make_shared<CudaBuffer>(Qwen2Config::intermediate_size() * Qwen2Config::hidden_size() * sizeof(__nv_bfloat16));
        gate_proj_weight = std::make_shared<CudaBuffer>(Qwen2Config::intermediate_size() * Qwen2Config::hidden_size() * sizeof(__nv_bfloat16));
        down_proj_weight = std::make_shared<CudaBuffer>(Qwen2Config::hidden_size() * Qwen2Config::intermediate_size() * sizeof(__nv_bfloat16));
        
        // Allocate temporary buffers for computation
        norm_output = std::make_shared<CudaBuffer>(Qwen2Config::hidden_size() * sizeof(__nv_bfloat16));
        query_buffer = std::make_shared<CudaBuffer>(Qwen2Config::queries_size() * sizeof(__nv_bfloat16));
        key_buffer = std::make_shared<CudaBuffer>(Qwen2Config::keys_size() * sizeof(__nv_bfloat16));
        value_buffer = std::make_shared<CudaBuffer>(Qwen2Config::values_size() * sizeof(__nv_bfloat16));
        attention_output = std::make_shared<CudaBuffer>(Qwen2Config::queries_size() * sizeof(float));
        ffn_gate_output = std::make_shared<CudaBuffer>(Qwen2Config::intermediate_size() * sizeof(__nv_bfloat16));
        ffn_up_output = std::make_shared<CudaBuffer>(Qwen2Config::intermediate_size() * sizeof(__nv_bfloat16));
        ffn_silu_out = std::make_shared<CudaBuffer>(Qwen2Config::intermediate_size() * sizeof(__nv_bfloat16));
        residual = std::make_shared<CudaBuffer>(Qwen2Config::hidden_size() * sizeof(__nv_bfloat16));
        
        // Initialize group query attention
        gqa = std::make_shared<GroupQueryAttention<QWEN2_SIZE>>(max_seq_len);
        
        // Initialize rotary position embeddings
        rope = std::make_shared<RoPE>();
        
        // Initialize SiLUMult
        silu_mult = std::make_shared<SiLUMult>();
        
        // Initialize MatrixVectorMultiply
        matmul = std::make_shared<MatrixVectorMultiply>();
    }

    uint32_t layer_num;
    LayerNorm input_layernorm;                              // (hidden_size,)
    std::shared_ptr<CudaBuffer> q_proj_weight;              // (queries_size, hidden_size)
    std::shared_ptr<CudaBuffer> q_proj_bias;                // (queries_size,)
    std::shared_ptr<CudaBuffer> k_proj_weight;              // (keys_size, hidden_size)
    std::shared_ptr<CudaBuffer> k_proj_bias;                // (keys_size,)
    std::shared_ptr<CudaBuffer> v_proj_weight;              // (values_size, hidden_size)
    std::shared_ptr<CudaBuffer> v_proj_bias;                // (values_size,)
    std::shared_ptr<CudaBuffer> o_proj_weight;              // (hidden_size, queries_size)
    LayerNorm post_attention_layernorm;                     // (hidden_size,)
    std::shared_ptr<CudaBuffer> up_proj_weight;             // (intermediate_size, intermediate_size)
    std::shared_ptr<CudaBuffer> gate_proj_weight;           // (intermediate_size, hidden_size)
    std::shared_ptr<CudaBuffer> down_proj_weight;           // (hidden_size, intermediate_size)

    // Additional member variables for computation
    std::shared_ptr<CudaBuffer> norm_output;                // (hidden_size,)
    std::shared_ptr<CudaBuffer> query_buffer;               // (queries_size,)
    std::shared_ptr<CudaBuffer> key_buffer;                 // (keys_size,)
    std::shared_ptr<CudaBuffer> value_buffer;               // (values_size,)
    std::shared_ptr<CudaBuffer> attention_output;           // (queries_size,)
    std::shared_ptr<CudaBuffer> ffn_gate_output;            // (intermediate_size,)
    std::shared_ptr<CudaBuffer> ffn_up_output;              // (intermediate_size,)
    std::shared_ptr<CudaBuffer> ffn_silu_out;               // (intermediate_size,)
    std::shared_ptr<CudaBuffer> residual;                   // (hidden_size,)
    
    // Components
    std::shared_ptr<GroupQueryAttention<QWEN2_SIZE>> gqa;
    std::shared_ptr<RoPE> rope;
    std::shared_ptr<SiLUMult> silu_mult;
    std::shared_ptr<MatrixVectorMultiply> matmul;

    /**
     * Pass the hidden state through this layer. Modifies the hidden state in-place.
     * @param k_cache bf16 keys (seq_len, num_layers, num_kv_heads, key_size)
     * @param v_cache bf16 values (seq_len, num_layers, num_kv_heads, value_size)
     * @param hidden_state current hidden state bf16 (hidden_size,)
     * @param seq_len current sequence length
     * @param stream CUDA stream for asynchronous operation
     */
    void forward(const std::shared_ptr<CudaBuffer>& k_cache, const std::shared_ptr<CudaBuffer> &v_cache, const std::shared_ptr<CudaBuffer> &hidden_state, int32_t seq_len, cudaStream_t stream) {
        // Save the input for residual connection
        checkCuda(cudaMemcpyAsync(
            residual->data, 
            hidden_state->data, 
            Qwen2Config::hidden_size() * sizeof(__nv_bfloat16), 
            cudaMemcpyDeviceToDevice, 
            stream
        ));
        
        // Apply input layer normalization
        input_layernorm.normalize_hidden_state(
            hidden_state,
            norm_output,
            stream
        );
        
        // Project to query, key and value
        // Query projection: (queries_size, hidden_size) x (hidden_size,) -> (queries_size,)
        MatrixVectorMultiply::bf16_matmul(
            Qwen2Config::queries_size(),
            Qwen2Config::hidden_size(),
            static_cast<__nv_bfloat16*>(q_proj_weight->data), 
            static_cast<__nv_bfloat16*>(q_proj_bias->data), 
            static_cast<__nv_bfloat16*>(norm_output->data),
            static_cast<__nv_bfloat16*>(query_buffer->data),
            stream
        );
        
        // Key projection: (keys_size, hidden_size) x (hidden_size,) -> (keys_size,)
        MatrixVectorMultiply::bf16_matmul(
            Qwen2Config::keys_size(),
            Qwen2Config::hidden_size(),
            static_cast<__nv_bfloat16*>(k_proj_weight->data), 
            static_cast<__nv_bfloat16*>(k_proj_bias->data), 
            static_cast<__nv_bfloat16*>(norm_output->data),
            static_cast<__nv_bfloat16*>(key_buffer->data),
            stream
        );
        
        // Value projection: (values_size, hidden_size) x (hidden_size,) -> (values_size,)
        MatrixVectorMultiply::bf16_matmul(
            Qwen2Config::values_size(),
            Qwen2Config::hidden_size(),
            static_cast<__nv_bfloat16*>(v_proj_weight->data), 
            static_cast<__nv_bfloat16*>(v_proj_bias->data), 
            static_cast<__nv_bfloat16*>(norm_output->data),
            static_cast<__nv_bfloat16*>(value_buffer->data),
            stream
        );
        
        // Apply rotary position embeddings to query and key vectors
        RoPE::apply_rope_to_qk(
            static_cast<__nv_bfloat16*>(query_buffer->data), 
            Qwen2Config::num_query_heads(), 
            Qwen2Config::head_size(), 
            seq_len - 1,
            10000.0f, // Standard theta_base value 
            stream
        );
        
        RoPE::apply_rope_to_qk(
            static_cast<__nv_bfloat16*>(key_buffer->data), 
            Qwen2Config::num_kv_heads(), 
            Qwen2Config::head_size(), 
            seq_len - 1,
            10000.0f, // Standard theta_base value
            stream
        );
        
        // Save key and value to cache
        // Calculate pointers to the current position in the cache
        __nv_bfloat16* k_cache_ptr = static_cast<__nv_bfloat16*>(k_cache->data) + 
            (seq_len - 1) * (Qwen2Config::num_layers() * Qwen2Config::keys_size()) +
            layer_num * Qwen2Config::keys_size();
            
        __nv_bfloat16* v_cache_ptr = static_cast<__nv_bfloat16*>(v_cache->data) + 
            (seq_len - 1) * (Qwen2Config::num_layers() * Qwen2Config::values_size()) +
            layer_num * Qwen2Config::values_size();
        
        // Copy key and value to cache
        checkCuda(cudaMemcpyAsync(
            k_cache_ptr, 
            key_buffer->data, 
            Qwen2Config::keys_size() * sizeof(__nv_bfloat16), 
            cudaMemcpyDeviceToDevice, 
            stream
        ));
        
        checkCuda(cudaMemcpyAsync(
            v_cache_ptr, 
            value_buffer->data, 
            Qwen2Config::values_size() * sizeof(__nv_bfloat16), 
            cudaMemcpyDeviceToDevice, 
            stream
        ));
        
        // Perform grouped query attention (GQA)
        gqa->sdpa(
            static_cast<__nv_bfloat16*>(query_buffer->data), 
            static_cast<__nv_bfloat16*>(k_cache->data), 
            static_cast<__nv_bfloat16*>(v_cache->data), 
            static_cast<float*>(attention_output->data), 
            layer_num, 
            seq_len, 
            stream
        );
        
        // Apply output projection: (hidden_size, queries_size) x (queries_size,) -> (hidden_size,)
        MatrixVectorMultiply::bf16_matmul<float>(
            Qwen2Config::hidden_size(),
            Qwen2Config::queries_size(),
            static_cast<__nv_bfloat16*>(o_proj_weight->data), 
            nullptr, // No bias
            static_cast<float*>(attention_output->data),
            static_cast<__nv_bfloat16*>(hidden_state->data),
            stream
        );
        
        // Add residual connection
        // Custom kernel for element-wise addition needed
        // For now, using a temporary workaround
        size_t grid_size = (Qwen2Config::hidden_size() + 255) / 256;
        elementwise_add_kernel<<<grid_size, 256, 0, stream>>>(
            static_cast<__nv_bfloat16*>(hidden_state->data),
            static_cast<__nv_bfloat16*>(residual->data),
            static_cast<__nv_bfloat16*>(hidden_state->data),
            Qwen2Config::hidden_size()
        );
        
        // Save hidden state for next residual connection
        checkCuda(cudaMemcpyAsync(
            residual->data, 
            hidden_state->data, 
            Qwen2Config::hidden_size() * sizeof(__nv_bfloat16), 
            cudaMemcpyDeviceToDevice, 
            stream
        ));
        
        // Apply post-attention layer normalization
        post_attention_layernorm.normalize_hidden_state(
            hidden_state,
            norm_output,
            stream
        );
        
        // Feed-forward network
        // Gate projection: (intermediate_size, hidden_size) x (hidden_size,) -> (intermediate_size,)
        MatrixVectorMultiply::bf16_matmul(
            Qwen2Config::intermediate_size(),
            Qwen2Config::hidden_size(),
            static_cast<__nv_bfloat16*>(gate_proj_weight->data), 
            nullptr, // No bias
            static_cast<__nv_bfloat16*>(norm_output->data),
            static_cast<__nv_bfloat16*>(ffn_gate_output->data),
            stream
        );
        
        // Up projection: (intermediate_size, hidden_size) x (hidden_size,) -> (intermediate_size,)
        MatrixVectorMultiply::bf16_matmul(
            Qwen2Config::intermediate_size(),
            Qwen2Config::hidden_size(),
            static_cast<__nv_bfloat16*>(up_proj_weight->data), 
            nullptr, // No bias
            static_cast<__nv_bfloat16*>(norm_output->data),
            static_cast<__nv_bfloat16*>(ffn_up_output->data),
            stream
        );
        
        // Apply SiLU activation to gate_projection and multiply with up_projection
        SiLUMult::silu_mult_in_place(
            ffn_gate_output,
            ffn_up_output,
            stream
        );
        
        // Down projection: (hidden_size, intermediate_size) x (intermediate_size,) -> (hidden_size,)
        MatrixVectorMultiply::bf16_matmul(
            Qwen2Config::hidden_size(),
            Qwen2Config::intermediate_size(),
            static_cast<__nv_bfloat16*>(down_proj_weight->data), 
            nullptr, // No bias
            static_cast<__nv_bfloat16*>(ffn_gate_output->data), // Using gate_output which now contains SiLU result
            static_cast<__nv_bfloat16*>(hidden_state->data),
            stream
        );
        
        // Add residual connection
        // Custom kernel for element-wise addition
        elementwise_add_kernel<<<grid_size, 256, 0, stream>>>(
            static_cast<__nv_bfloat16*>(hidden_state->data),
            static_cast<__nv_bfloat16*>(residual->data),
            static_cast<__nv_bfloat16*>(hidden_state->data),
            Qwen2Config::hidden_size()
        );
    }
};
