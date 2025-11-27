import torch
import torch.nn as nn
from models.attention import MultiHeadAttention
from models.fnn import FeedForward
import copy
from models.embeddings import PositionalEncoding, Embeddings

class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads):
        super(EncoderLayer, self).__init__()
        
        # 两个子层：Attention 和 FFN
        self.self_attn = MultiHeadAttention(heads, d_model)
        self.feed_forward = FeedForward(d_model)
        
        # 两个归一化层 (LayerNorm)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout 用于防止过拟合
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, mask):
        # --- 子层 1: Attention ---
        # 原始输入 x 存起来做残差
        residual = x 
        
        # 跑 Attention
        x = self.self_attn(x, x, x, mask)
        x = self.dropout(x)
        
        # Add & Norm: 原始 x + 新 x，再归一化
        x = self.norm1(x + residual)
        
        # --- 子层 2: FFN ---
        # 现在的 x 存起来做残差
        residual = x
        
        # 跑 FFN
        x = self.feed_forward(x)
        x = self.dropout(x)
        
        # Add & Norm
        x = self.norm2(x + residual)
        
        return x
    
# 一个小工具函数，用来克隆 N 层相同的网络
def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, N, heads):
        super(Encoder, self).__init__()
        
        # 1. 嵌入层
        self.embed = Embeddings(d_model, vocab_size)
        
        # 2. 位置编码
        self.pe = PositionalEncoding(d_model)
        
        # 3. 堆叠 N 层 EncoderLayer (比如 6 层)
        self.layers = clones(EncoderLayer(d_model, heads), N)
        
        # 4. 最终的归一化
        self.norm = nn.LayerNorm(d_model)

    def forward(self, src, mask):
        # src 是输入的 token IDs
        
        # 先做 Embedding + 位置编码
        x = self.embed(src)
        x = self.pe(x)
        
        # 依次穿过 N 层 EncoderLayer
        for layer in self.layers:
            x = layer(x, mask)
            
        # 最后再做一次 Norm
        return self.norm(x)
    
if __name__ == "__main__":
    # 假设参数
   vocab_size = 1000   # 词表大小
   d_model = 512       # 向量维度
   heads = 8           # 8个头
   N = 6               # 堆叠6层

# 实例化模型
   model = Encoder(vocab_size, d_model, N, heads)

# 模拟输入数据
# Batch size = 2, 句子长度 = 10
   src = torch.randint(0, vocab_size, (2, 10)) 

# 这里的 mask 设为 None，假装没有 padding
   output = model(src, None)

   print("输入形状:", src.shape)   # [2, 10]
   print("输出形状:", output.shape) # [2, 10, 512]