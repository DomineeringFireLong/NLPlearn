import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# ======================
# 数据准备 (示例：简单序列预测)
# ======================
# 生成一个简单的时间序列数据集：输入为0-3的4种状态，输出为下一状态
text = "0 1 2 3 0 1 2 3"  # 示例序列
sequence = text.split()

# 创建词汇表映射
vocab = {'0':0, '1':1, '2':2, '3':3}
int2vocab = {v:k for k,v in vocab.items()}

# 将序列转换为数字张量
X = [vocab[char] for char in sequence[:-1]]  # 输入：前n-1个元素
y = [vocab[char] for char in sequence[1:]]   # 目标：后n-1个元素

# 转换为PyTorch张量 (batch_size=1, seq_length=7, input_size=4)
X = torch.tensor(X).long()
X_onehot = nn.functional.one_hot(X, num_classes=4).float().unsqueeze(0)  # 添加batch维度
y = torch.tensor(y).long().unsqueeze(0)  # 形状：(1,7)

# ======================
# RNN模型定义
# ======================
class CustomRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        """
        Args:
            input_size:  输入特征维度 (e.g., 4)
            hidden_size: 隐藏层维度
            output_size: 输出类别数 (e.g., 4)
        """
        super(CustomRNN, self).__init__()
        self.hidden_size = hidden_size
        
        # 权重矩阵 (使用PyTorch Parameter注册可训练参数)
        self.Wxh = nn.Parameter(torch.randn(input_size, hidden_size))  # 输入到隐藏层
        self.Whh = nn.Parameter(torch.randn(hidden_size, hidden_size))  # 隐藏层到隐藏层
        self.Why = nn.Parameter(torch.randn(hidden_size, output_size))  # 隐藏层到输出
        
        # 偏置项
        self.bh = nn.Parameter(torch.zeros(hidden_size))
        self.by = nn.Parameter(torch.zeros(output_size))
        
    def forward(self, x, h_prev=None):
        """
        Args:
            x: 输入序列张量，形状 (batch_size, seq_len, input_size)
            h_prev: 初始隐藏状态，形状 (batch_size, hidden_size)
        
        Returns:
            outputs: 每个时间步的输出，形状 (batch_size, seq_len, output_size)
            h_final: 最后时间步的隐藏状态
        """
        batch_size, seq_len, _ = x.shape
        
        # 初始化隐藏状态
        if h_prev is None:
            h = torch.zeros(batch_size, self.hidden_size)
        else:
            h = h_prev.clone()
        
        outputs = []
        
        # 遍历序列的每个时间步
        for t in range(seq_len):
            # 当前时间步的输入 (batch_size, input_size)
            xt = x[:, t, :]
            
            # RNN核心计算
            h = torch.tanh(xt @ self.Wxh + h @ self.Whh + self.bh)
            
            # 计算输出 (未应用softmax，后续用CrossEntropyLoss会自动处理)
            yt = h @ self.Why + self.by
            outputs.append(yt)
        
        # 堆叠所有时间步的输出 (batch_size, seq_len, output_size)
        outputs = torch.stack(outputs, dim=1)
        return outputs, h

# ======================
# 实例化模型
# ======================
input_size = 4  # 输入为one-hot编码的4维向量
hidden_size = 8
output_size = 4

model = CustomRNN(input_size, hidden_size, output_size)
print(model)

# ======================
# 训练配置
# ======================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.05)

# ======================
# 训练循环
# ======================
num_epochs = 100
for epoch in range(num_epochs):
    # 前向传播
    outputs, _ = model(X_onehot)
    loss = criterion(outputs.view(-1, output_size), y.view(-1))
    
    # 反向传播与优化
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # 打印训练信息
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# ======================
# 测试推理
# ======================
with torch.no_grad():
    test_outputs, _ = model(X_onehot)
    predicted = test_outputs.argmax(dim=-1)
    print("\n原始序列:", sequence)
    print("预测序列:", [int2vocab[i] for i in predicted.squeeze().tolist()])