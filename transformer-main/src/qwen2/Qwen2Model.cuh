#pragma once

#include <memory>

#include "Qwen2Layer.cuh"
#include "Qwen2Config.h"
#include "../ErrorCheck.h"
#include "../gpu_ops/LayerNorm.cuh"
#include "../gpu_ops/ArgMax.cuh"

template<Qwen2Size QWEN2_SIZE>
class Qwen2Model {
    cudaStream_t stream;
public:
    using Qwen2Config = Qwen2Config<QWEN2_SIZE>;
    using Qwen2Layer = Qwen2Layer<QWEN2_SIZE>;

    Qwen2Model() {
        checkCuda(cudaStreamCreate(&stream));
        
        // Allocate buffers for computations
        hidden_states = std::make_shared<CudaBuffer>(Qwen2Config::hidden_size() * sizeof(__nv_bfloat16));
        norm_output = std::make_shared<CudaBuffer>(Qwen2Config::hidden_size() * sizeof(__nv_bfloat16));
        logits = std::make_shared<CudaBuffer>(Qwen2Config::vocab_size() * sizeof(__nv_bfloat16));
        
        // Initialize argmax with maximum size needed
        argmax = std::make_shared<ArgMax>(Qwen2Config::vocab_size());
    }

    ~Qwen2Model() {
        checkCuda(cudaStreamDestroy(stream));
    }

    std::shared_ptr<CudaBuffer> embedding_weight; // (vocab_size, hidden_size)
    std::shared_ptr<Qwen2Layer> layers[Qwen2Config::num_layers()];
    LayerNorm final_layernorm{Qwen2Config::hidden_size()}; // (hidden_size,)
    
    // Buffers for computations
    std::shared_ptr<CudaBuffer> hidden_states;   // (hidden_size,)
    std::shared_ptr<CudaBuffer> norm_output;     // (hidden_size,)
    std::shared_ptr<CudaBuffer> logits;          // (vocab_size,)
    
    // Components
    std::shared_ptr<ArgMax> argmax;

    /**
     *
     * @param k_cache bf16 keys (seq_len, num_layers, num_kv_heads, key_size)
     * @param v_cache bf16 values (seq_len, num_layers, num_kv_heads, value_size)
     * @param seq_len current sequence length
     * @param input_tok_id last token in the sequence
     * @param temperature Sampling parameter. Always set to 0, for deterministic (greedy) decoding, see https://www.ibm.com/docs/en/watsonx/saas?topic=lab-model-parameters-prompting.
     *                    You do not need to implement any other sampling methods.
     * @return The ID of the next token
     */
    int32_t forward(const std::shared_ptr<CudaBuffer> &k_cache, const std::shared_ptr<CudaBuffer> &v_cache, int32_t seq_len, int32_t input_tok_id, float temperature) {
        // Embedding lookup (extract the embedding for the input token)
        size_t embedding_offset = input_tok_id * Qwen2Config::hidden_size() * sizeof(__nv_bfloat16);
        checkCuda(cudaMemcpyAsync(
            hidden_states->data,
            static_cast<uint8_t*>(embedding_weight->data) + embedding_offset,
            Qwen2Config::hidden_size() * sizeof(__nv_bfloat16),
            cudaMemcpyDeviceToDevice,
            stream
        ));
        
        // Forward through each transformer layer
        for (int layer_idx = 0; layer_idx < Qwen2Config::num_layers(); ++layer_idx) {
            layers[layer_idx]->forward(k_cache, v_cache, hidden_states, seq_len, stream);
        }
        
        // Apply final layer normalization
        final_layernorm.normalize_hidden_state(
            hidden_states,
            norm_output,
            stream
        );
        
        // Compute logits through the output embedding matrix (transpose of the input embedding matrix)
        MatrixVectorMultiply::bf16_matmul(
            Qwen2Config::vocab_size(),
            Qwen2Config::hidden_size(),
            static_cast<__nv_bfloat16*>(embedding_weight->data),
            nullptr, // No bias for logits computation
            static_cast<__nv_bfloat16*>(norm_output->data),
            static_cast<__nv_bfloat16*>(logits->data),
            stream
        );
        
        // Apply temperature scaling (if temperature > 0)
        // When temperature is 0, we just do greedy decoding (argmax)
        int32_t next_token;
        
        if (temperature > 0.0f) {
            // Apply temperature scaling to logits
            int32_t *next_token_ptr = argmax->bf16_argmax(
                logits,
                stream
            );
            // Synchronize to get the result
            checkCuda(cudaStreamSynchronize(stream));
            checkCuda(cudaMemcpy(&next_token, next_token_ptr, sizeof(int32_t), cudaMemcpyDeviceToHost));
        } else {
            // Greedy decoding (argmax)
            int32_t *next_token_ptr = argmax->bf16_argmax(
                logits,
                stream
            );
            // Synchronize to get the result
            checkCuda(cudaStreamSynchronize(stream));
            checkCuda(cudaMemcpy(&next_token, next_token_ptr, sizeof(int32_t), cudaMemcpyDeviceToHost));
        }
        
        return next_token;
    }
};