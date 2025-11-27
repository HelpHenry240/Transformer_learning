import torch
import torch.nn as nn
# 确保你的文件夹结构是正确的，且包含 __init__.py
from models.transformer import Transformer 

def make_dummy_mask(src, tgt):
    """
    创建一个简易的 Mask 用于测试
    src_mask: [Batch, 1, 1, SrcLen] - 遮挡 padding
    tgt_mask: [Batch, 1, TgtLen, TgtLen] - 遮挡未来时刻 (look-ahead)
    """
    # 这里为了跑通代码，我们暂时创建全 1 的 mask (即不遮挡任何东西，除了 pad)
    # 实际项目中你需要根据 padding value (比如 0) 来生成 mask
    
    # Src Mask: 假设 src 中为 0 的是 padding
    src_mask = (src != 0).unsqueeze(1).unsqueeze(2) 
    
    # Tgt Mask: 生成一个下三角矩阵，防止看到未来
    tgt_len = tgt.size(1)
    tgt_mask = torch.tril(torch.ones((tgt_len, tgt_len))).expand(
        tgt.size(0), 1, tgt_len, tgt_len
    ).type_as(src_mask)
    
    return src_mask, tgt_mask

def run_debug():
    print("----- 开始 Transformer 本地逻辑验证 -----")

    # 1. 定义超参数 (使用小参数以适应 CPU)
    BATCH_SIZE = 2
    SEQ_LEN = 10
    SRC_VOCAB_SIZE = 100
    TGT_VOCAB_SIZE = 120
    D_MODEL = 64       # 正常是 512，本地用 64 够了
    N_LAYERS = 2       # 层数
    HEADS = 4          # 注意：D_MODEL 必须能被 HEADS 整除 (64/4=16)
    
    device = torch.device('cpu') # 强制使用 CPU

    # 2. 实例化模型
    try:
        model = Transformer(
            src_vocab_size=SRC_VOCAB_SIZE, 
            tgt_vocab_size=TGT_VOCAB_SIZE, 
            d_model=D_MODEL, 
            N=N_LAYERS, 
            heads=HEADS
        ).to(device)
        print("✅ 模型实例化成功")
    except Exception as e:
        print(f"❌ 模型实例化失败: {e}")
        return

    # 3. 构造伪数据 (Dummy Data)
    # 输入: [Batch, SeqLen] 的整数索引
    src = torch.randint(1, SRC_VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN)).to(device)
    tgt = torch.randint(1, TGT_VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN)).to(device)

    # 构造简单的 Mask
    src_mask, tgt_mask = make_dummy_mask(src, tgt)
    
    print(f"输入 Src 形状: {src.shape}")
    print(f"输入 Tgt 形状: {tgt.shape}")

    # 4. 前向传播 (Forward Pass)
    try:
        # 注意：这里调用的是 model.forward(src, tgt, src_mask, tgt_mask)
        output = model(src, tgt, src_mask, tgt_mask)
        print("✅ 前向传播成功")
        print(f"输出 Output 形状: {output.shape}") 
        
        # 验证输出维度：应该是 [Batch, SeqLen, Tgt_Vocab_Size]
        expected_shape = (BATCH_SIZE, SEQ_LEN, TGT_VOCAB_SIZE)
        assert output.shape == expected_shape, f"维度错误，期望 {expected_shape}，实际 {output.shape}"
        
    except Exception as e:
        print(f"❌ 前向传播失败: {e}")
        # 打印详细错误栈对于 debug 很有用
        import traceback
        traceback.print_exc()
        return

    # 5. 反向传播测试 (Backward Pass)
    # 这一步是为了确保梯度链没有断，没有出现 inplace操作错误
    try:
        # 假设标签是 tgt 向后移一位 (这里为了测试直接用随机数)
        # 展平 output 以计算 CrossEntropy: [Batch*SeqLen, Vocab]
        output_flat = output.view(-1, TGT_VOCAB_SIZE)
        target_flat = torch.randint(0, TGT_VOCAB_SIZE, (BATCH_SIZE * SEQ_LEN,)).to(device)
        
        criterion = nn.CrossEntropyLoss()
        loss = criterion(output_flat, target_flat)
        
        model.zero_grad()
        loss.backward()
        print(f"✅ 反向传播成功 (Loss: {loss.item():.4f})")
        print("🎉 恭喜！Transformer 模型核心逻辑通过测试！")
        
    except RuntimeError as e:
        print(f"❌ 反向传播失败 (通常是维度不匹配或 inplace 错误): {e}")

if __name__ == '__main__':
    run_debug()