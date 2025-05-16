import safetensors
from pathlib import Path
import json
from dataclasses import dataclass
import torch
import math
from rope import RoPE


@dataclass
class Qwen2Config:
    # Token latent space representation size
    hidden_size: int
    # Number of decoder layers
    num_layers: int
    # Number of queries per token per layer, see Group Query Attention
    num_query_heads: int
    # Number of key-value pairs per token per layer
    num_kv_heads: int
    # Size of a single query or key
    head_size: int
    # Intermediate dimension of the post-attention feedforward network
    intermediate_size: int
    # If the input embedding matrix is the same as the output logit matrix
    # This is a trick to reduce size in lower parameter count models
    embedding_tying: bool
    
    def queries_size(self) -> int:
        """
        Size of all queries grouped together.
        In Qwen2, same as hidden_size.
        896 on 0.5B.
        """
        return self.head_size * self.num_query_heads

    def keys_size(self) -> int:
        """
        Size of keys vector for each token.
        Intentionally much smaller than hidden_size, to reduce KV cache size.
        See Group Query Attention paper.
        128 for 0.5B.
        """
        return self.head_size * self.num_kv_heads
    
    def value_size(self) -> int:
        """
        For Qwen2, value size is the same as query/key size.
        64 on 0.5B.
        """
        return self.head_size

    def values_size(self) -> int:
        """
        For Qwen2, values size is the same as keys size.
        128 for 0.5B.
        """
        return self.value_size() * self.num_kv_heads


def t5_layer_norm(weight: torch.Tensor, hidden_state: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Layer normalization without bias, as used in the T5 paper https://arxiv.org/pdf/1910.10683"""
    variance = hidden_state.pow(2).mean()
    # scale to unit variance (eps for numerical stability)
    hidden_state = hidden_state / (variance + eps).sqrt()
    # scale to learned variance
    return weight * hidden_state


def silu(x: torch.Tensor) -> torch.Tensor:
    """Sigmoid linear unit"""
    return x / (1 + torch.exp(-x))


@dataclass
class Qwen2Layer:
    config: Qwen2Config

    layer_num: int
    input_layernorm: torch.Tensor  # (hidden_size,)
    q_proj_weight: torch.Tensor  # (queries_size, hidden_size)
    q_proj_bias: torch.Tensor  # (queries_size,)
    k_proj_weight: torch.Tensor  # (keys_size, hidden_size)
    k_proj_bias: torch.Tensor  # (keys_size,)
    v_proj_weight: torch.Tensor  # (values_size, hidden_size)
    v_proj_bias: torch.Tensor  # (values_size,)
    o_proj_weight: torch.Tensor  # (hidden_size, queries_size)
    post_attention_layernorm: torch.Tensor  # (hidden_size,)
    up_proj_weight: torch.Tensor  # (intermediate_size, intermediate_size)
    gate_proj_weight: torch.Tensor  # (intermediate_size, hidden_size)
    down_proj_weight: torch.Tensor  # (hidden_size, intermediate_size)

    def forward(self, keys: torch.Tensor, values: torch.Tensor, hidden_state: torch.Tensor) -> torch.Tensor:
        """
        keys: (seq_len, num_layers, keys_size). However, only seq_len - 1 are initially populated, this function call will fill in the last token.
        values: (seq_len, num_layers, values_size). Only seq_len - 1 are initially populated.
        hidden_state: (hidden_size). Last token hidden state.
        """
        seq_len = keys.shape[0]

        # pre-attention layernorm
        attn_input = t5_layer_norm(self.input_layernorm, hidden_state)  # (hidden_size,)

        # self attention
        queries = self.q_proj_weight @ attn_input + self.q_proj_bias  # (num_query_heads * head_size,)
        # separate into heads
        queries = queries.view((self.config.num_query_heads, self.config.head_size))  # (num_query_heads, head_size)
        queries = RoPE.apply_rope_to_qk(queries, seq_len - 1)

        new_keys = self.k_proj_weight @ attn_input + self.k_proj_bias  # (keys_size,)
        new_keys = new_keys.view((self.config.num_kv_heads, self.config.head_size))  # (num_kv_heads, head_size)
        new_keys = RoPE.apply_rope_to_qk(new_keys, seq_len - 1)
        new_keys = new_keys.view((self.config.num_kv_heads * self.config.head_size))  # (keys_size,)
        # add last token to KV cache
        keys[-1, self.layer_num, :] = new_keys
        # separate into heads
        keys = keys.view((seq_len, self.config.num_layers, self.config.num_kv_heads, self.config.head_size))

        new_values = self.v_proj_weight @ attn_input + self.v_proj_bias  # (values_size,)
        values[-1, self.layer_num, :] = new_values
        values = values.view((seq_len, self.config.num_layers, self.config.num_kv_heads, self.config.value_size()))

        weighted_values = torch.zeros((self.config.num_query_heads, self.config.value_size()))  # (num_query_heads, value_size)

        # group query attention
        for head_idx in range(self.config.num_query_heads):
            head_query = queries[head_idx]

            # each key is used multiple times for different queries
            key_idx = head_idx * self.config.num_kv_heads // self.config.num_query_heads
            head_keys = keys[:, self.layer_num, key_idx, :]  # (seq_len, head_size)

            # dot product of query with all keys, scaled by 1/sqrt(d_k)
            attn_vals_raw = (head_query @ head_keys.T) / math.sqrt(self.config.head_size)  # (seq_len,)

            # numerically stable softmax
            attn_max = torch.max(attn_vals_raw)
            attn_vals_exp = torch.exp(attn_vals_raw - attn_max)
            attn_vals_exp_sum = attn_vals_exp.sum()
            attn_vals_softmax = attn_vals_exp / attn_vals_exp_sum

            # weighted sum of values
            head_values = values[:, self.layer_num, key_idx, :]  # (seq_len, value_size)
            weighted_values[head_idx] = attn_vals_softmax @ head_values
        
        # view contiguously
        weighted_values = weighted_values.view((self.config.num_query_heads * self.config.value_size()))
        attn_output = self.o_proj_weight @ weighted_values

        # residual connection
        hidden_state += attn_output

        # pre-FFN layernorm
        ffn_input = t5_layer_norm(self.post_attention_layernorm, hidden_state)
        # feed forward network with gated linear unit
        ffn_output = self.down_proj_weight @ (silu(self.gate_proj_weight @ ffn_input) * (self.up_proj_weight @ ffn_input))

        # residual connection
        hidden_state += ffn_output

        return hidden_state


class Qwen2Model:
    config: Qwen2Config
    # embedding weight: (vocab_size, hidden_size)
    embedding_weight: torch.Tensor
    layers: list[Qwen2Layer]
    # final layernorm: (hidden_size)
    final_layernorm: torch.Tensor

    def __init__(self, model_dir: Path, dtype: torch.dtype=torch.float32):
        with open(model_dir / "config.json", "r") as f:
            config_obj = json.load(f)
        self.config = Qwen2Config(
            hidden_size=config_obj["hidden_size"],
            num_layers=config_obj["num_hidden_layers"],
            num_query_heads=config_obj["num_attention_heads"], 
            num_kv_heads=config_obj["num_key_value_heads"],
            head_size=config_obj["hidden_size"] // config_obj["num_attention_heads"],
            intermediate_size=config_obj["intermediate_size"],
            embedding_tying=config_obj["tie_word_embeddings"]
        )

        with safetensors.safe_open(model_dir / "model.safetensors", framework="pt") as tensors:
            self.embedding_weight = tensors.get_tensor("model.embed_tokens.weight").to(dtype)
            self.layers = []
            for i in range(self.config.num_layers):
                self.layers.append(Qwen2Layer(
                    config=self.config,
                    layer_num=i,
                    input_layernorm=tensors.get_tensor(f"model.layers.{i}.input_layernorm.weight").to(dtype),
                    q_proj_weight=tensors.get_tensor(f"model.layers.{i}.self_attn.q_proj.weight").to(dtype),
                    q_proj_bias=tensors.get_tensor(f"model.layers.{i}.self_attn.q_proj.bias").to(dtype),
                    k_proj_weight=tensors.get_tensor(f"model.layers.{i}.self_attn.k_proj.weight").to(dtype), 
                    k_proj_bias=tensors.get_tensor(f"model.layers.{i}.self_attn.k_proj.bias").to(dtype),
                    v_proj_weight=tensors.get_tensor(f"model.layers.{i}.self_attn.v_proj.weight").to(dtype),
                    v_proj_bias=tensors.get_tensor(f"model.layers.{i}.self_attn.v_proj.bias").to(dtype),
                    o_proj_weight=tensors.get_tensor(f"model.layers.{i}.self_attn.o_proj.weight").to(dtype),
                    post_attention_layernorm=tensors.get_tensor(f"model.layers.{i}.post_attention_layernorm.weight").to(dtype),
                    up_proj_weight=tensors.get_tensor(f"model.layers.{i}.mlp.up_proj.weight").to(dtype),
                    gate_proj_weight=tensors.get_tensor(f"model.layers.{i}.mlp.gate_proj.weight").to(dtype),
                    down_proj_weight=tensors.get_tensor(f"model.layers.{i}.mlp.down_proj.weight").to(dtype)
                ))
            self.final_layernorm = tensors.get_tensor("model.norm.weight").to(dtype)

    def forward(self, keys: torch.Tensor, values: torch.Tensor, input_tok_id: int, temperature: float) -> int:
        """
        keys: (seq_len, num_layers, keys_size). Only seq_len - 1 are initially populated, this function call will fill in the last token.
        values: (seq_len, num_layers, values_size)
        input_tok_id: last token in the sequence
        temperature: sampling parameter, deterministic=0
        """

        # take initial hidden state from embedding table
        hidden_state = self.embedding_weight[input_tok_id].squeeze().detach().clone()

        for layer in self.layers:
            hidden_state = layer.forward(keys, values, hidden_state)

        # final layernorm
        hidden_state = t5_layer_norm(self.final_layernorm, hidden_state)

        assert self.config.embedding_tying, "current implementation only support embedding tying"
        output_scores = hidden_state @ self.embedding_weight.T
        if temperature == 0:
            # softmax is monotonic, so we can sample the largest value directly
            new_token = torch.argmax(output_scores)
        else:
            raise ValueError('temperature > 0 not supported')

        return int(new_token)
