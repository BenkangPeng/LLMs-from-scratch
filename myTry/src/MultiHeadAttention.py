import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, drop_rate, num_heads, qkv_bias = False):
        """
        Args:
            d_in(int): input dimension, the embedding dimentions of the token
            d_out(int): output dimension, the embedding dimentions of the output token from attention head
            context_length(int): context length
            drop_rate(float): dropout rate
            num_heads(int): number of heads
        """

        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"

        self.d_in = d_in
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads # the output dimension of each head

        self.W_query = nn.Linear(d_in, d_out, bias = qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias = qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias = qkv_bias)

        # optional projection  
        self.out_proj = nn.Linear(d_out, d_out)

        self.dropout = nn.Dropout(drop_rate)
        self.register_buffer(
            'mask',
            torch.triu(torch.ones(context_length, context_length), diagonal=1)

            # atten_socores' dimension is context_length x context_length
            # so as the mask matrix
        )

    def forward(self, x):
        """
        Args:
            x(torch.Tensor): input tensor of shape (batch_size, context_length, d_in(emb_dim))

        Returns:

        """

        batch_size, num_tokens, d_in = x.shape
        # dimension of the Weight matrix is (d_in x d_out)
        # x @ W
        # dimension of Q,K,V is batch_size x context_length x d_out
        Q, K, V = self.W_query(x), self.W_key(x), self.W_value(x)

        # We implicitly split the matrix by adding a `num_heads` dimension
        # Unroll the last dim: (b, num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim)
        Q = Q.view(batch_size, num_tokens, self.num_heads, self.head_dim)
        K = K.view(batch_size, num_tokens, self.num_heads, self.head_dim)
        V = V.view(batch_size, num_tokens, self.num_heads, self.head_dim)

        # Transpose: (b, num_tokens, num_heads, head_dim) -> (b, num_heads, num_tokens, head_dim)
        Q, K, V = Q.transpose(1, 2), K.transpose(1, 2), V.transpose(1, 2)

        attn_scores = Q @ K.transpose(2, 3) # batch_size x num_heads x num_tokens x num_tokens

        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

        attn_scores.masked_fill_(mask_bool, -torch.inf)

        attn_weights = torch.softmax(attn_scores / K.shape[-1] ** 0.5, dim = -1)
        attn_weights = self.dropout(attn_weights)
        # batch_size x num_heads x num_tokens x num_tokens


        context_vec = (attn_weights @ V).transpose(1, 2)
        # batch_size x num_tokens x num_heads x head_dim

        # `context_vec` has been transposed before, and the elements aren't 
        # arranged continuously in memory
        # so before call .view(), we need to call contiguous() to ensure the memory continuous
        context_vec = context_vec.contiguous().view(batch_size, num_tokens, self.d_out)

        # optional projection
        # 在多头注意力中，输入被拆分到多个头（num_heads）独立计算，每个头生成 head_dim 维的特征。
        # 计算完成后，context_vec(batch_size, num_tokens, num_heads, head_dim)中包括
        # num_heads个维度为(batch_size, num_tokens, 1, head_dim)的特征向量(一个头产生一个特征向量)
        # 乘上一个可学习的out_proj矩阵，将这些特征向量进行线性组合，增强不同头之间的信息交互
        context_vec = self.out_proj(context_vec)

        return context_vec

