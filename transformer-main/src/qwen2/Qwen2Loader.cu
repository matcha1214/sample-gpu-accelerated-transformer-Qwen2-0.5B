#include "Qwen2Loader.h"

#include "../ErrorCheck.h"
#include "../HostBuffer.h"

std::string Qwen2Loader::get_model_dir() {
    const char *model_dir_env = std::getenv("TRANSFORMER_MODEL_DIR");
    if (model_dir_env) {
        return model_dir_env;
    } else {
        return "/cs179/Qwen2.5-0.5B-Instruct";
    }
}

std::shared_ptr<CudaBuffer> Qwen2Loader::load_bf16_tensor(safetensors::safetensors_t &st, const std::string &name, size_t expected_dim_0, size_t expected_dim_1) {
    safetensors::tensor_t tensor;
    bool found = false;

    // search for key
    for (size_t i = 0; i < st.tensors.size(); i++) {
        std::string current_name = st.tensors.keys()[i];
        if (current_name == name) {
            // found key
            found = true;
            st.tensors.at(i, &tensor);
            break;
        }
    }
    if (!found) {
        throw std::runtime_error("failed to find tensor: " + name);
    }

    if (tensor.shape[0] != expected_dim_0 ||
        (tensor.shape.size() == 2 && tensor.shape[1] != expected_dim_1)) {
        throw std::runtime_error("unexpected tensor shape for tensor: " + name + ", got shape (" + std::to_string(tensor.shape[0]) + ", " + std::to_string(tensor.shape[1]) + ")");
    }
    if (tensor.dtype != safetensors::kBFLOAT16) {
        throw std::runtime_error("unexpected dtype, can only handle bfloat16");
    }

    size_t num_els = tensor.shape[0];
    if (tensor.shape.size() == 2 && tensor.shape[1] != 0) {
        // 2D
        num_els *= tensor.shape[1];
    }
    size_t len_bytes = num_els * sizeof(__nv_bfloat16);

    const uint8_t *tensor_data_host = st.databuffer_addr + tensor.data_offsets[0];
    // upload to GPU
    auto tensor_data_gpu = std::make_shared<CudaBuffer>(len_bytes);
    checkCuda(cudaMemcpy(tensor_data_gpu->data, tensor_data_host, len_bytes, cudaMemcpyDefault));

    return tensor_data_gpu;
}
