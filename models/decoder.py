import torch
import torch.nn as nn
from models.attention import MultiHeadAttention
from models.fnn import FeedForward
import copy
from models.embeddings import PositionalEncoding, Embeddings

class DecoderLayer(nn.Module):
    def __init__(self, d_model, heads):
        super(DecoderLayer, self).__init__()
        
        # 三个子层：Self-Attention, Encoder-Decoder Attention 和 FFN
        self.self_attn = MultiHeadAttention(heads, d_model)
        self.enc_dec_attn = MultiHeadAttention(heads, d_model)
        self.feed_forward = FeedForward(d_model)
        
        # 三个归一化层 (LayerNorm)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        # Dropout 用于防止过拟合
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, enc_output, src_mask, tgt_mask):
        # --- 子层 1: Self-Attention ---
        residual = x 
        x = self.self_attn(x, x, x, tgt_mask)
        x = self.dropout(x)
        x = self.norm1(x + residual)
        
        # --- 子层 2: Encoder-Decoder Attention ---
        residual = x
        x = self.enc_dec_attn(x, enc_output, enc_output, src_mask)
        x = self.dropout(x)
        x = self.norm2(x + residual)
        
        # --- 子层 3: FFN ---
        residual = x
        x = self.feed_forward(x)
        x = self.dropout(x)
        x = self.norm3(x + residual)
        
        return x
    
if __name__ == "__main__":
    # 测试代码
    d_model = 512
    seq_length = 20
    batch_size = 32
    heads = 8

    decoder_layer = DecoderLayer(d_model, heads)

    sample_input = torch.randn(batch_size, seq_length, d_model)
    enc_output = torch.randn(batch_size, seq_length, d_model)
    src_mask = None
    tgt_mask = None

    output = decoder_layer(sample_input, enc_output, src_mask, tgt_mask)

    print("Output shape:", output.shape)  # 应该是 [32, 20, 512]