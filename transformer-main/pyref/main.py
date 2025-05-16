from qwen2 import Qwen2Model
import torch
import os
from pathlib import Path
from transformers import AutoTokenizer

def main():
    # ~/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775
    model_dir = Path(os.getenv("TRANSFORMER_MODEL_DIR") or "/cs179/Qwen2.5-0.5B-Instruct")
    model = Qwen2Model(model_dir)

    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    # autoregressively generate 100 tokens
    max_seq_len = 100
    latest_token = 64  # sequence begins with "a" token

    k_cache = torch.zeros((max_seq_len, model.config.num_layers, model.config.keys_size()))
    v_cache = torch.zeros((max_seq_len, model.config.num_layers, model.config.values_size()))

    for i in range(max_seq_len):
        seq_len = i + 1
        keys_seq = k_cache[:seq_len, :]
        values_seq = v_cache[:seq_len, :]
        new_token = model.forward(keys_seq, values_seq, latest_token, 0.0)
        print(tokenizer.decode([new_token]), end='', flush=True)
        latest_token = new_token


if __name__ == '__main__':
    main()
