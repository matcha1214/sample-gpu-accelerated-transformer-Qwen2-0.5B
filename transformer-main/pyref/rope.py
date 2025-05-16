import torch


class RoPE:
    """
    GPT-NeoX Style Rotary Positional Embeddings, see https://nn.labml.ai/transformers/rope/index.html
    Implementation based on transformers _compute_default_rope_parameters, apply_rotary_pos_emb, and rotate_half 
    """

    @staticmethod
    def rotate_half(x):
        x1 = x[..., :x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2:]
        return torch.cat((-x2, x1), dim=-1)
    
    @staticmethod
    def apply_rope_to_qk(x: torch.Tensor, position_idx: int, theta_base: float=1e6) -> torch.Tensor:
        """
        x: (num_heads, query_size) Can be queries or keys.
        position_idx: Token index in the sequence.
        """
        head_dim = x.shape[-1]
        
        thetas = 1.0 / (theta_base ** (torch.arange(0, head_dim, 2, dtype=torch.float, device=x.device) / head_dim))  # shape=(head_dim / 2,)
        angles = position_idx * thetas
        cos = angles.cos()
        sin = angles.sin()

        # each theta value used twice
        cos = torch.concat([cos, cos])  # shape=(head_dim,)
        sin = torch.concat([sin, sin])
        
        # Apply rotary embeddings
        # Similar to equation 34 in RoPE paper, except repeat theta values are concatenated (stride of head_dim / 2) instead of repeat_interleaved (stride of 1)
        return (x * cos) + (RoPE.rotate_half(x) * sin)
