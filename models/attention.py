import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model):
        super(MultiHeadAttention, self).__init__()
        
        self.d_model = d_model
        self.h = heads  # 头数 (比如 8)
        self.d_k = d_model // heads # 每个头的维度 (512 // 8 = 64)
        
        # 定义四个线性层 (矩阵乘法)
        # W_q, W_k, W_v 负责把输入投影成 Q, K, V
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        
        # W_o 负责最后把大家的结果融合
        self.out = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        bs = q.size(0) # Batch Size (一次处理多少句话)
        
        # 1. 线性投影 (Linear Projections)
        # 此时形状: [Batch, Seq_Len, d_model]
        k = self.k_linear(k)
        q = self.q_linear(q)
        v = self.v_linear(v)
        
        # 2. 切分多头 (Split Heads)
        # view: 把 d_model 拆成 (heads, d_k)
        # transpose: 把 heads 维度移到前面，方便并行计算
        # 变换后形状: [Batch, Heads, Seq_Len, d_k]
        k = k.view(bs, -1, self.h, self.d_k).transpose(1, 2)
        q = q.view(bs, -1, self.h, self.d_k).transpose(1, 2)
        v = v.view(bs, -1, self.h, self.d_k).transpose(1, 2)

        # 3. 缩放点积注意力 (Scaled Dot-Product Attention)
        # Q 乘 K 的转置
        scores = torch.matmul(q, k.transpose(-2, -1)) 
        
        # 除以根号 d_k (缩放)
        scores = scores / math.sqrt(self.d_k)
        
        # 如果有 mask (掩码)，把不该看的地方设为负无穷
        if mask is not None:
             scores = scores.masked_fill(mask == 0, -1e9)
        
        # 计算概率分布
        attn = torch.softmax(scores, dim=-1)
        
        # 乘 V (提取信息)
        output = torch.matmul(attn, v)
        
        # 4. 合并多头 (Concat)
        # transpose: 把 heads 移回去
        # contiguous: 内存连续化(技术细节)
        # view: 把 (heads, d_k) 拼回 d_model
        # 形状变回: [Batch, Seq_Len, d_model]
        output = output.transpose(1, 2).contiguous().view(bs, -1, self.d_model)
        
        # 5. 最后的线性层 (Final Linear)
        output = self.out(output)
        
        return output
    
if __name__ == "__main__":
    # 测试代码
    d_model = 512
    heads = 8
    seq_length = 20
    batch_size = 32

    mha = MultiHeadAttention(heads, d_model)

    sample_q = torch.rand(batch_size, seq_length, d_model)
    sample_k = torch.rand(batch_size, seq_length, d_model)
    sample_v = torch.rand(batch_size, seq_length, d_model)

    output = mha(sample_q, sample_k, sample_v)

    print("Output shape:", output.shape)  # 应该是 [32, 20, 512]