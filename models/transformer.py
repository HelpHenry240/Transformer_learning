import torch
import torch.nn as nn
from models.attention import MultiHeadAttention
from models.fnn import FeedForward
import copy
from models.embeddings import PositionalEncoding, Embeddings
from models.encoder import EncoderLayer
from models.decoder import DecoderLayer

class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, N, heads):
        super(Transformer, self).__init__()
        
        # 编码器和解码器各 N 层
        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, heads) for _ in range(N)])
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, heads) for _ in range(N)])
        
        # 嵌入层和位置编码
        self.src_embed = Embeddings(d_model, src_vocab_size)
        self.tgt_embed = Embeddings(d_model, tgt_vocab_size)
        self.src_pe = PositionalEncoding(d_model)
        self.tgt_pe = PositionalEncoding(d_model)
        
        # 最后的线性层，把 d_model 映射到词表大小
        self.out = nn.Linear(d_model, tgt_vocab_size)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        # 编码器部分
        x = self.src_embed(src)
        x = self.src_pe(x)
        for layer in self.encoder_layers:
            x = layer(x, src_mask)
        enc_output = x
        
        # 解码器部分
        y = self.tgt_embed(tgt)
        y = self.tgt_pe(y)
        for layer in self.decoder_layers:
            y = layer(y, enc_output, src_mask, tgt_mask)
        
        # 输出层
        output = self.out(y)
        return output
    
if __name__ == "__main__":
    # 测试代码
    src_vocab_size = 10000
    tgt_vocab_size = 10000
    d_model = 512
    N = 6
    heads = 8
    seq_length = 20
    batch_size = 32

    model = Transformer(src_vocab_size, tgt_vocab_size, d_model, N, heads)

    sample_src = torch.randint(0, src_vocab_size, (batch_size, seq_length))
    sample_tgt = torch.randint(0, tgt_vocab_size, (batch_size, seq_length))

    output = model(sample_src, sample_tgt)

    print("Output shape:", output.shape)  # 应该是 [32, 20, 10000]