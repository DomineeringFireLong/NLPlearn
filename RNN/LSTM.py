import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt

# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size

        # 输入门
        self.W_xi = nn.Linear(input_size, hidden_size)
        self.W_hi = nn.Linear(hidden_size, hidden_size)
        self.b_i = nn.Parameter(torch.zeros(hidden_size))

        # 遗忘门
        self.W_xf = nn.Linear(input_size, hidden_size)
        self.W_hf = nn.Linear(hidden_size, hidden_size)
        self.b_f = nn.Parameter(torch.zeros(hidden_size))

        # 细胞状态更新门
        self.W_xc = nn.Linear(input_size, hidden_size)
        self.W_hc = nn.Linear(hidden_size, hidden_size)
        self.b_c = nn.Parameter(torch.zeros(hidden_size))

        # 输出门
        self.W_xo = nn.Linear(input_size, hidden_size)
        self.W_ho = nn.Linear(hidden_size, hidden_size)
        self.b_o = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x, init_states=None):
        seq_len, batch_size, _ = x.size()

        # 初始化状态
        h, c = (torch.zeros(batch_size, self.hidden_size).to(x.device),
                torch.zeros(batch_size, self.hidden_size).to(x.device)) \
               if init_states is None else init_states

        outputs = []
        for t in range(seq_len):
            x_t = x[t]  # (batch_size, input_size)

            # 输入门
            i_t = torch.sigmoid(self.W_xi(x_t) + self.W_hi(h) + self.b_i)
            # 遗忘门
            f_t = torch.sigmoid(self.W_xf(x_t) + self.W_hf(h) + self.b_f)
            # 细胞状态更新门
            c_t = torch.tanh(self.W_xc(x_t) + self.W_hc(h) + self.b_c)
            # 输出门
            o_t = torch.sigmoid(self.W_xo(x_t) + self.W_ho(h) + self.b_o)

            # 更新细胞状态和隐藏状态
            c = f_t * c + i_t * c_t
            h = o_t * torch.tanh(c)

            outputs.append(h.unsqueeze(0))

        outputs = torch.cat(outputs, dim=0)  # (seq_len, batch, hidden)
        return outputs, (h, c)

class LSTMPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMPredictor, self).__init__()
        self.lstm = LSTM(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, init_states=None):
        outputs, _ = self.lstm(x, init_states)
        return self.fc(outputs[-1])  # (batch_size, output_size)

def generate_data(seq_length=500, lookback=10, noise=0.1,input_size=1,output_size=1):
    """生成20维特征的数据，输出5维"""
    x = np.linspace(0, 8*np.pi, seq_length)#这是时间自变量，不是特征
    # 生成20个特征
    y = np.zeros((seq_length, input_size))
    for i in range(input_size):
        y[:, i] = np.sin(x + i * np.pi / 10) + noise * np.random.randn(seq_length)

    # 输出只保留前output_size个特征
    y_output = y[:, :output_size]

    sequences = []
    targets = []
    for i in range(len(y) - lookback):
        sequences.append(y[i:i + lookback])  # 输入是20维
        targets.append(y_output[i + lookback])  # 输出是5维

    X = torch.FloatTensor(np.array(sequences))  # (seq_length-lookback, lookback, 20)
    y = torch.FloatTensor(np.array(targets))    # (seq_length-lookback, 5)
    return X, y

def train_model():
    seq_length = 500
    lookback = 10
    input_size = 20  # 输入维度改为20
    hidden_size = 64
    output_size = 5  # 输出维度改为5
    epochs = 200
    batch_size = 32

    X, y = generate_data(seq_length=seq_length, lookback=lookback, noise=0.05,input_size=input_size,output_size=output_size)
    train_size = int(0.8 * len(X))
    train_X, test_X = X[:train_size], X[train_size:]
    train_y, test_y = y[:train_size], y[train_size:]

    train_dataset = TensorDataset(train_X, train_y)
    test_dataset = TensorDataset(test_X, test_y)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    model = LSTMPredictor(input_size=input_size, hidden_size=hidden_size, output_size=output_size).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    best_loss = float('inf')
    train_losses, val_losses = [], []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0

        for inputs, targets in train_loader:
            inputs = inputs.permute(1, 0, 2).to(device)  # (seq, batch, features)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs = inputs.permute(1, 0, 2).to(device)
                targets = targets.to(device)
                outputs = model(inputs)
                val_loss += criterion(outputs, targets).item()

        avg_train = epoch_loss / len(train_loader)
        avg_val = val_loss / len(test_loader)
        train_losses.append(avg_train)
        val_losses.append(avg_val)

        if avg_val < best_loss:
            best_loss = avg_val
            torch.save(model.state_dict(), 'best_lstm.pth')

        if (epoch + 1) % 20 == 0:
            print(f'Epoch {epoch + 1}/{epochs} | Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f}')

    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Loss Curve')
    plt.xlabel('Epochs')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.show()

    return model

def visualize_predictions(model, lookback=10, predict_steps=50):
    """可视化预测效果（仅展示前2个输出特征）"""
    x = np.linspace(0, 2*np.pi, lookback)
    test_input = np.zeros((lookback, 20))
    for i in range(20):
        test_input[:, i] = np.sin(x + i * np.pi / 10)
    test_tensor = torch.FloatTensor(test_input).unsqueeze(1).to(device)  # (seq, 1, 20)

    model.eval()
    predictions = []
    current_seq = test_tensor.clone()

    with torch.no_grad():
        for _ in range(predict_steps):
            pred = model(current_seq)  # pred shape: (1, 5)
            predictions.append(pred[0].cpu().numpy())  # 保存所有5个特征的预测值

            # 将 pred 扩展为 (1, 1, 20) 以便与 current_seq 的特征维度匹配
            # 这里假设我们只关心前5个特征，其余特征填充为0
            pred_expanded = torch.zeros(1, 1, 20).to(device)
            pred_expanded[0, 0, :5] = pred

            # 更新 current_seq
            current_seq = torch.cat([current_seq[1:], pred_expanded])

    predictions = np.array(predictions)  # (predict_steps, 5)

    # 生成真实值
    x_true = np.linspace(0, 2*np.pi + 2*np.pi/predict_steps, lookback + predict_steps)
    true_values = np.zeros((lookback + predict_steps, 5))
    for i in range(5):
        true_values[:, i] = np.sin(x_true + i * np.pi / 10)

    # 仅绘制前2个输出特征
    plt.figure(figsize=(12, 6))
    for i in range(2):
        plt.plot(range(lookback), test_input[:, i], label=f'Initial Sequence {i + 1}')
        plt.plot(range(lookback - 1, lookback + predict_steps),
                 true_values[lookback - 1:, i],
                 '--', label=f'True Values {i + 1}')
        plt.plot(range(lookback, lookback + predict_steps),
                 predictions[:, i],
                 '-', label=f'Predictions {i + 1}')

    plt.title('Multi-step Prediction (First 2 Output Features)')
    plt.xlabel('Time Steps')
    plt.ylabel('Value')
    plt.legend()
    plt.show()
if __name__ == "__main__":
    trained_model = train_model()
    visualize_predictions(trained_model)