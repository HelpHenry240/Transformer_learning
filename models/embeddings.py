import torch
import torch.nn as nn
import math

class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        # 1. 词嵌入层：把 ID 查表变成向量
        # d_model: 向量维度 (比如 512)
        # vocab: 词表大小 (比如 10000)
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        # x 是输入的一串数字 ID
        # 这里的 math.sqrt(self.d_model) 是论文中的一个小细节
        # 作用是放大 embedding 的数值，为了和后面加上的 Positional Encoding 保持量级一致
        return self.lut(x) * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        # 创建一个矩阵 pe，用来存位置编码
        # max_len 是句子最大长度，d_model 是维度
        pe = torch.zeros(max_len, d_model)
        
        # 下面是论文中复杂的数学公式 (sin/cos)，你只需要知道它生成了一组固定的波形数字
        # 就像给每个位置发了一个独特的“纹身”
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term) # 偶数位置用 sin
        pe[:, 1::2] = torch.cos(position * div_term) # 奇数位置用 cos
        
        pe = pe.unsqueeze(0) # 增加一个 Batch 维度
        # register_buffer 告诉 PyTorch：这不是需要学习的参数，但这属于模型的一部分
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [Batch, Sequence_Length, Embedding_Dim]
        # 把 Embedding 和 位置编码 直接相加
        x = x + self.pe[:, :x.size(1)]
        return x
    
if __name__ == "__main__":
    # 测试代码
    vocab_size = 10000
    d_model = 512
    seq_length = 20
    batch_size = 32

    embeddings = Embeddings(d_model, vocab_size)
    pos_encoding = PositionalEncoding(d_model)

    sample_input = torch.randint(0, vocab_size, (batch_size, seq_length))
    embedded = embeddings(sample_input)
    positioned = pos_encoding(embedded)

    print("Embedded shape:", embedded.shape)  # 应该是 [32, 20, 512]
    print("Positioned shape:", positioned.shape)  # 应该是 [32, 20, 512]