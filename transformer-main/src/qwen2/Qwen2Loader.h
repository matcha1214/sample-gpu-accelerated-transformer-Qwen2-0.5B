#pragma once

#include <string>
#include <memory>
#include "../CudaBuffer.cuh"
#include "Qwen2Config.h"
#include "Qwen2Model.cuh"
#include "../vendor/safetensors.hh"
#include <iostream>

class Qwen2Loader {
public:
    /**
     * Get the path of the Qwen2 model, either from TRANSFORMER_MODEL_DIR environment variable.
     * Huggingface snapshot directory, which contains model.safetensors, config.json, and vocab.json.
     * Defaults to /cs179/Qwen2.5-0.5B-Instruct (which is copied from ~/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775)
     */
    static std::string get_model_dir();

    /**
     * Load a tensor from the safetensors file, in bfloat16 format, and upload to GPU.
     */
    static std::shared_ptr<CudaBuffer> load_bf16_tensor(safetensors::safetensors_t &st, const std::string &name, size_t expected_dim_0, size_t expected_dim_1 = 0);

    template<Qwen2Size QWEN2_SIZE>
    static std::shared_ptr<Qwen2Model<QWEN2_SIZE>> load_qwen2(const std::string &safetensors_file, int32_t max_seq_len) {
        using Qwen2Config = Qwen2Config<QWEN2_SIZE>;
        auto model = std::make_shared<Qwen2Model<QWEN2_SIZE>>();

        // open safetensors file
        std::string warn, err;
        safetensors::safetensors_t st;
        bool ret = safetensors::mmap_from_file(safetensors_file, &st, &warn, &err);
        if (!warn.empty()) {
            std::cerr << "safetensors warning: " << warn << std::endl;
        }
        if (!ret) {
            throw std::runtime_error("safetensors error: " + err);
        }
        if (!safetensors::validate_data_offsets(st, err)) {
            throw std::runtime_error("safetensors: invalid data offsets: " + err);
        }

        // load all tensors
        model->embedding_weight = load_bf16_tensor(st, "model.embed_tokens.weight", Qwen2Config::vocab_size(), Qwen2Config::hidden_size());
        for (uint32_t layer_idx = 0; layer_idx < Qwen2Config::num_layers(); layer_idx++) {
            std::string layer_prefix = "model.layers." + std::to_string(layer_idx) + ".";
            model->layers[layer_idx] = std::make_shared<Qwen2Layer<QWEN2_SIZE>>(layer_idx, max_seq_len);
            model->layers[layer_idx]->input_layernorm.weights = load_bf16_tensor(st, layer_prefix + "input_layernorm.weight", Qwen2Config::hidden_size());
            model->layers[layer_idx]->q_proj_weight = load_bf16_tensor(st, layer_prefix + "self_attn.q_proj.weight", Qwen2Config::queries_size(), Qwen2Config::hidden_size());
            model->layers[layer_idx]->q_proj_bias = load_bf16_tensor(st, layer_prefix + "self_attn.q_proj.bias", Qwen2Config::queries_size());
            model->layers[layer_idx]->k_proj_weight = load_bf16_tensor(st, layer_prefix + "self_attn.k_proj.weight", Qwen2Config::keys_size(), Qwen2Config::hidden_size());
            model->layers[layer_idx]->k_proj_bias = load_bf16_tensor(st, layer_prefix + "self_attn.k_proj.bias", Qwen2Config::keys_size());
            model->layers[layer_idx]->v_proj_weight = load_bf16_tensor(st, layer_prefix + "self_attn.v_proj.weight", Qwen2Config::values_size(), Qwen2Config::hidden_size());
            model->layers[layer_idx]->v_proj_bias = load_bf16_tensor(st, layer_prefix + "self_attn.v_proj.bias", Qwen2Config::values_size());
            model->layers[layer_idx]->o_proj_weight = load_bf16_tensor(st, layer_prefix + "self_attn.o_proj.weight", Qwen2Config::hidden_size(), Qwen2Config::queries_size());
            model->layers[layer_idx]->post_attention_layernorm.weights = load_bf16_tensor(st, layer_prefix + "post_attention_layernorm.weight", Qwen2Config::hidden_size());
            model->layers[layer_idx]->up_proj_weight = load_bf16_tensor(st, layer_prefix + "mlp.up_proj.weight", Qwen2Config::intermediate_size(), Qwen2Config::hidden_size());
            model->layers[layer_idx]->gate_proj_weight = load_bf16_tensor(st, layer_prefix + "mlp.gate_proj.weight", Qwen2Config::intermediate_size(), Qwen2Config::hidden_size());
            model->layers[layer_idx]->down_proj_weight = load_bf16_tensor(st, layer_prefix + "mlp.down_proj.weight", Qwen2Config::hidden_size(), Qwen2Config::intermediate_size());
        }
        model->final_layernorm.weights = load_bf16_tensor(st, "model.norm.weight", Qwen2Config::hidden_size());

        return model;
    }
};
