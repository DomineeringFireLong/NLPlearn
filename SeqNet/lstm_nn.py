import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# 设置随机种子以确保可重复性
torch.manual_seed(42)

# 1. 生成数据
def generate_sine_wave(seq_length, num_samples):
    """生成正弦波时间序列"""
    time = np.linspace(0, 4 * np.pi, seq_length + num_samples)
    data = np.sin(time)
    return data

seq_length = 50  # 输入序列长度
num_samples = 200  # 样本数量
data = generate_sine_wave(seq_length, num_samples)

# 2. 准备数据集
def create_sequences(data, seq_length):
    """将时间序列分割为输入-输出对"""
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:i + seq_length]
        y = data[i + seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

X, y = create_sequences(data, seq_length)
X = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)  # 形状: (num_samples, seq_length, 1)
y = torch.tensor(y, dtype=torch.float32).unsqueeze(-1)  # 形状: (num_samples, 1)

# 3. 定义 LSTM 模型
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # 初始化隐藏状态和细胞状态
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # LSTM 前向传播
        out, _ = self.lstm(x, (h0, c0))  # out 形状: (batch_size, seq_length, hidden_size)

        # 只取最后一个时间步的输出
        out = self.fc(out[:, -1, :])  # 形状: (batch_size, output_size)
        return out

# 4. 初始化模型、损失函数和优化器
input_size = 1  # 输入特征维度
hidden_size = 50  # 隐藏层维度
output_size = 1  # 输出维度
num_layers = 1  # LSTM 层数

model = LSTMModel(input_size, hidden_size, output_size, num_layers)
criterion = nn.MSELoss()  # 均方误差损失
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# 5. 训练模型
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()

    # 前向传播
    outputs = model(X)
    loss = criterion(outputs, y)

    # 反向传播和优化
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# 6. 预测
model.eval()
with torch.no_grad():
    test_input = X[0].unsqueeze(0)  # 取第一个样本作为测试输入
    predicted = model(test_input)

# 7. 可视化结果
plt.figure(figsize=(10, 6))
plt.plot(data, label='True Data')
plt.plot(np.arange(seq_length, len(data)), y.numpy(), label='Target')
plt.scatter(seq_length, predicted.item(), color='red', label='Prediction')
plt.legend()
plt.show()