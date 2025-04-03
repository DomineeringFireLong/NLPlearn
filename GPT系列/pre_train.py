import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

class SimpleGPT(nn.Module):
    def __init__(self, vocab_size, embed_size, num_heads, num_layers, context_length):
        super(SimpleGPT, self).__init__()
        self.context_length = context_length  # 最大上下文长度
        
        # 词嵌入层：将token索引映射为embed_size维向量
        # 输入: (batch_size, seq_len) 输出: (batch_size, seq_len, embed_size)
        self.token_embedding = nn.Embedding(vocab_size, embed_size)#x*w  seq_len*embed_size
        #嵌入层通过查表操作将输入的索引映射到嵌入向量，而不是直接进行矩阵乘法

        # 位置嵌入层：给每个位置生成embed_size维编码
        # 输入: (1, seq_len) 输出: (1, seq_len, embed_size)
        self.position_embedding = nn.Embedding(context_length, embed_size)
        
        # Transformer编码层堆叠
        # 每层的输入输出都是(batch_size, seq_len, embed_size)
        self.transformer_blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_size,          # 输入输出维度
                nhead=num_heads,            # 注意力头数
                dim_feedforward=4*embed_size, # 前馈网络隐藏层维度
                dropout=0.1,                # 防止过拟合
                activation='gelu'           # 激活函数
            ) for _ in range(num_layers)
        ])
        
        # 输出层
        self.ln_f = nn.LayerNorm(embed_size)  # 层归一化
        self.head = nn.Linear(embed_size, vocab_size, bias=False)  # 输出词汇表概率
        
    def forward(self, x):
        # x形状: (batch_size, seq_len)
        B, T = x.shape
        
        # 生成位置编码 [0,1,2,...,T-1]
        pos = torch.arange(0, T, dtype=torch.long, device=x.device).unsqueeze(0)
        # pos形状: (1, seq_len)
        
        # 获取词嵌入和位置嵌入
        tok_emb = self.token_embedding(x)  # (B,T,embed_size)
        pos_emb = self.position_embedding(pos)  # (1,T,embed_size)
        
        # 合并嵌入 (广播机制使pos_emb自动扩展到B个样本)
        x = tok_emb + pos_emb  # (B,T,embed_size)
        
        # 通过所有Transformer层
        for block in self.transformer_blocks:
            x = block(x)  # 每层保持(B,T,embed_size)形状
            
        # 最终输出处理
        x = self.ln_f(x)  # 层归一化 (B,T,embed_size)
        logits = self.head(x)  # (B,T,vocab_size)
        # 最后输出的logits，维度(B,T,vocab_size)，b=1，就是序列长度个词汇量大小的向量。
        # 就是每个词生成的词汇量大小维度的概率，就是当前位置预测的下一个token属于词汇量中每个词的概率。
        return logits

def pretrain_gpt_with_visualization():
    # 超参数设置
    vocab_size = 5000    # 词汇表大小
    embed_size = 128     # 词向量维度
    num_heads = 4        # 注意力头数
    num_layers = 4       # Transformer层数
    context_length = 64  # 上下文长度
    batch_size = 32      # 批量大小
    lr = 3e-4           # 学习率
    num_steps = 1000     # 训练步数
    
    # 设备选择
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 初始化模型并移至设备
    model = SimpleGPT(vocab_size, embed_size, num_heads, num_layers, context_length).to(device)
    
    # 使用AdamW优化器
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    
    # 记录训练指标
    losses = []  # 损失值
    ppls = []    # 困惑度
    
    # 训练循环
    progress_bar = tqdm(range(num_steps), desc="Training")
    for step in progress_bar:
        # 生成随机输入数据 (模拟文本)
        # inputs形状: (batch_size, context_length)
        inputs = torch.randint(0, vocab_size, (batch_size, context_length), device=device)
        
        # 前向传播
        # 输入: inputs[:, :-1] 形状 (batch_size, context_length-1)
        # 输出: logits 形状 (batch_size, context_length-1, vocab_size)
        logits = model(inputs[:, :-1])
        
        # 计算损失
        # 将logits展平为 (batch_size, seq_len-1, vocab_size)
        # 目标展平为 (batch_size, seq_len-1)
        # 目标函数本质上是最大化似然估计（Maximum Likelihood Estimation, MLE），但具体实现是通过最小化预测值与输入序列右移的交叉熵来完成的。
        # 交叉熵损失直接衡量模型预测分布与真实分布之间的差异
        # 因为真实分布p(x)是一个one-hot分布，所以交叉熵损失可以看作是对数似然的负值。
        loss = F.cross_entropy(
            logits.reshape(-1, vocab_size), 
            inputs[:, 1:].reshape(-1)
        )
        # 自监督：目标 token（即输入序列右移一位）
        # 反向传播
        optimizer.zero_grad()  # 清空梯度
        loss.backward()        # 计算梯度
        optimizer.step()       # 更新参数
        
        # 记录指标
        losses.append(loss.item())
        ppl = torch.exp(loss).item()  # 计算困惑度
        ppls.append(ppl)
        
        # 更新进度条显示
        progress_bar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'ppl': f"{ppl:.2f}"
        })
    
    # 可视化训练过程
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(losses)
    plt.title("Training Loss")
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    
    plt.subplot(1, 2, 2)
    plt.plot(ppls)
    plt.title("Perplexity")
    plt.xlabel("Steps")
    plt.ylabel("PPL")
    
    plt.tight_layout()
    plt.show()
    # 保存预训练模型
    torch.save(model.state_dict(), './pretrained_gpt.pth')
    print("Pretrained model saved to pretrained_gpt.pth")
    
    return model

def generate_text(model, start_text, max_length=50, temperature=1.0, top_k=40):
    # 切换到评估模式
    model.eval()
    
    # 将起始文本转换为token序列
    # 简单实现：使用ASCII码模vocab_size作为token
    tokens = [ord(c) % model.token_embedding.num_embeddings for c in start_text]
    
    for _ in range(max_length):
        # 准备输入：取最后context_length个token
        input_tensor = torch.tensor(
            tokens[-model.context_length:], 
            dtype=torch.long,
            device=next(model.parameters()).device
        ).unsqueeze(0)  # 增加batch维度 -> (1, seq_len)
        
        # 获取预测 (不计算梯度)
        with torch.no_grad():
            logits = model(input_tensor)  # (1, seq_len, vocab_size)
        
        # 取最后一个位置的logits并应用温度调节
        logits = logits[0, -1, :] / temperature  # (vocab_size,)
        
        # Top-k过滤：只保留概率最高的k个token
        if top_k is not None:
            topk_values, _ = torch.topk(logits, top_k)
            logits[logits < topk_values[-1]] = -float('Inf')
        
        # 计算概率分布
        probs = F.softmax(logits, dim=-1)  # (vocab_size,)
        
        # 从分布中采样下一个token
        next_token = torch.multinomial(probs, num_samples=1).item()
        tokens.append(next_token)
    
    # 将token转换回文本 (仅显示可打印ASCII字符)
    return ''.join([chr(t) if 32 <= t < 127 else '?' for t in tokens])

def test_generation(model, test_cases):
    print("\nTesting Generation:")
    for i, (prompt, temp) in enumerate(test_cases):
        print(f"\nTest Case {i+1}:")
        print(f"Prompt: '{prompt}'")
        print(f"Temperature: {temp}")
        
        # 每个测试用例生成3个样本
        for _ in range(3):
            generated = generate_text(model, prompt, temperature=temp)
            print(f"Generated: {generated}")

# 训练模型并可视化
gpt_model = pretrain_gpt_with_visualization()

# 定义测试用例 (提示文本, 温度)
test_cases = [
    ("The quick brown fox", 0.7),
    ("Once upon a time", 1.0),
    ("Artificial intelligence", 1.2),
    ("To be or not to be", 0.5)
]




if __name__ == '__main__':
    model = pretrain_gpt_with_visualization()
    # 测试生成效果
    # test_generation(gpt_model, test_cases)