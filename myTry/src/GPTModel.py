import torch
import torch.nn as nn

GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False
}


class GPTModel(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()

        # token embedding layer
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])

        # position embedding layer
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"]) 

        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )