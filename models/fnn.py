import torch
import torch.nn as nn


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=2048):
        super(FeedForward, self).__init__()
        # d_ff 通常是 d_model 的 4 倍
        self.linear1 = nn.Linear(d_model, d_ff)   # 膨胀
        self.dropout = nn.Dropout(0.1)
        self.linear2 = nn.Linear(d_ff, d_model)   # 压缩

    def forward(self, x):
        # 也就是公式: ReLU(xW1 + b1)W2 + b2
        x = self.dropout(torch.relu(self.linear1(x)))
        x = self.linear2(x)
        return x

if __name__ == "__main__":
    # 测试代码
    d_model = 512
    seq_length = 20
    batch_size = 32

    ffn = FeedForward(d_model)

    sample_input = torch.randn(batch_size, seq_length, d_model)
    output = ffn(sample_input)

    print("Output shape:", output.shape)  # 应该是 [32, 20, 512]