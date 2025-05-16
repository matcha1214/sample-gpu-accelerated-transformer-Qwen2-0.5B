#pragma once

#include <stdexcept>

enum Qwen2Size {
    QWEN2_0_5B,
};

template<Qwen2Size size>
struct Qwen2Config {
    // Token latent space representation size
    static constexpr size_t hidden_size() {
        if constexpr (size == QWEN2_0_5B) return 896;
        throw std::logic_error("Unknown size");
    }
    // Number of decoder layers
    static constexpr size_t num_layers() {
        if constexpr (size == QWEN2_0_5B) return 24;
        throw std::logic_error("Unknown size");
    }
    // Number of queries per token per layer, see Group Query Attention
    static constexpr size_t num_query_heads() {
        if constexpr (size == QWEN2_0_5B) return 14;
        throw std::logic_error("Unknown size");
    }
    // Number of key-value pairs per token per layer
    static constexpr size_t num_kv_heads() {
        if constexpr (size == QWEN2_0_5B) return 2;
        throw std::logic_error("Unknown size");
    }
    // Size of a single query or key
    static constexpr size_t head_size() {
        if constexpr (size == QWEN2_0_5B) return 64;
        throw std::logic_error("Unknown size");
    }
    // Intermediate dimension of the post-attention feedforward network
    static constexpr size_t intermediate_size() {
        if constexpr (size == QWEN2_0_5B) return 4864;
        throw std::logic_error("Unknown size");
    }
    // If the input embedding matrix is the same as the output logit matrix
    // This is a trick to reduce size in lower parameter count models
    static constexpr bool embedding_tying() {
        if constexpr (size == QWEN2_0_5B) return true;
        throw std::logic_error("Unknown size");
    }

    static constexpr size_t vocab_size() {
        return 151936;
    }

    static constexpr float rope_theta_base() {
        return 1e6f;
    }

    // Computed values based on constants

    /**
     * Size of all queries grouped together.
     * In Qwen2, same as hidden_size.
     * 896 on 0.5B.
     */
    static constexpr int queries_size() {
        return head_size() * num_query_heads();
    }

    /**
     * Size of all key vectors for a token, when grouped together.
     * Intentionally much smaller than hidden_size, to reduce KV cache size.
     * See Group Query Attention paper.
     * 128 for 0.5B.
     */
    static constexpr int keys_size() {
        return head_size() * num_kv_heads();
    }

    /**
     * For Qwen2, value size is the same as query/key size.
     * 64 on 0.5B.
     */
    static constexpr int value_size() {
        return head_size();
    }

    /**
     * Size of all value vectors for a token, when grouped together.
     * For Qwen2, values size is the same as keys size.
     * 128 for 0.5B.
     */
    static constexpr int values_size() {
        return value_size() * num_kv_heads();
    }
};