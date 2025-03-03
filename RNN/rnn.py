import torch
import torch.nn as nn
import torch.optim as optim

# ======================
# 数据准备
# ======================
text = "0 1 2 3 0 1 2 3"
sequence = text.split()

vocab = {'0': 0, '1': 1, '2': 2, '3': 3}
int2vocab = {v: k for k, v in vocab.items()}

X = [vocab[char] for char in sequence[:-1]]
y = [vocab[char] for char in sequence[1:]]

X = torch.tensor(X).long()
X_onehot = nn.functional.one_hot(X, num_classes=4).float().unsqueeze(0)  # (1, 7, 4)
y = torch.tensor(y).long()  # (7,)

# ======================
# RNN模型定义
# ======================
class CustomRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(CustomRNN, self).__init__()
        self.hidden_size = hidden_size
        self.Wxh = nn.Parameter(torch.randn(input_size, hidden_size))
        self.Whh = nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.Why = nn.Parameter(torch.randn(hidden_size, output_size))
        self.bh = nn.Parameter(torch.zeros(hidden_size))
        self.by = nn.Parameter(torch.zeros(output_size))

    def forward(self, x, h_prev=None):
        batch_size, seq_len, _ = x.shape
        if h_prev is None:
            h = torch.zeros(batch_size, self.hidden_size)
        else:
            h = h_prev.clone()

        outputs = []
        for t in range(seq_len):
            xt = x[:, t, :]
            h = torch.tanh(xt @ self.Wxh + h @ self.Whh + self.bh)
            yt = torch.softmax(h @ self.Why + self.by, dim=-1)  # 对 logits 应用 softmax
            outputs.append(yt)

        outputs = torch.stack(outputs, dim=1)  # (batch_size, seq_len, output_size)
        return outputs, h

# ======================
# 自定义 NLLLoss 函数
# ======================
def custom_nllloss(output, target):
    """
    自定义负对数似然损失函数。

    Args:
        output: 模型输出，形状为 (batch_size * seq_len, output_size)，已经是 log-probabilities。
        target: 目标标签，形状为 (batch_size * seq_len,)。

    Returns:
        损失值（标量）。
    """
    # 根据目标标签选择对应的 log-probabilities
    log_probs = output[range(len(target)), target]
    
    # 计算负对数似然损失
    loss = -log_probs.mean()#让对应标签的概率最大
    return loss

# ======================
# 实例化模型
# ======================
input_size = 4
hidden_size = 8
output_size = 4
model = CustomRNN(input_size, hidden_size, output_size)

# ======================
# 训练配置
# ======================
optimizer = optim.Adam(model.parameters(), lr=0.01)

# ======================
# 训练循环
# ======================
num_epochs = 100
for epoch in range(num_epochs):
    outputs, _ = model(X_onehot)
    
    # 将输出和目标调整为 NLLLoss 的输入格式
    # outputs 是 (1, 7, 4)，y 是 (7,)，需要将 outputs 展平为 (7, 4)
    log_probs = torch.log(outputs.view(-1, output_size))  # 计算 log-probabilities
    loss = custom_nllloss(log_probs, y.view(-1))  # 使用自定义 NLLLoss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# ======================
# 测试推理
# ======================
with torch.no_grad():
    test_outputs, _ = model(X_onehot)
    predicted = test_outputs.argmax(dim=-1)
    print("\n原始序列:", sequence)
    print("预测序列:", [int2vocab[i] for i in predicted.squeeze().tolist()])