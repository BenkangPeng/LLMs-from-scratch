import torch
import torch.nn as nn
from MultiHeadAttention import MultiHeadAttention

class TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = 