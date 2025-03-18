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
            #因为这里索引方便，所以x的维度是(seq_len，batch_size, input_size)

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

def generate_data(seq_length=500, input_seq_length=20, output_seq_length=5, input_size=1, output_size=1, noise=0.1):
    """
    生成时序数据。
    参数:
        seq_length (int): 总时间序列长度。
        input_seq_length (int): 输入序列的长度（时间步数）。
        output_seq_length (int): 输出序列的长度（时间步数）。
        input_size (int): 输入特征的维度。
        output_size (int): 输出特征的维度。
        noise (float): 噪声强度。
    返回:
        x (torch.Tensor): 输入数据，形状为 (input_seq_length, batch, input_size)。
        y (torch.Tensor): 输出数据，形状为 (output_seq_length,batch,  output_size)。
    """
    t = np.linspace(0, 8 * np.pi, seq_length)  # 生成时间自变量
    # 生成 input_size 个输入特征和 output_size 个输出特征
    data = np.zeros((seq_length, input_size))
    for i in range(input_size):
        data[:, i] = np.sin(t + i * np.pi / 10) + noise * np.random.randn(seq_length)

    # 生成输入和输出序列
    x, y = [], []
    #从0到seq_length - input_seq_length - output_seq_length，不包含最后一个,python的切片是左闭右开
    for i in range(seq_length - input_seq_length - output_seq_length+1):
        # 输入是 input_seq_length 个时间步的 input_size 个特征
        x.append(data[i:i + input_seq_length, :input_size])
        # 输出是 output_seq_length 个时间步的 output_size 个特征
        y.append(data[i + input_seq_length:i + input_seq_length + output_seq_length, [0,-1]])
        #x和y是时间前后关系

    # 转换为 PyTorch 张量
    x = torch.tensor(np.array(x), dtype=torch.float32)  
    y = torch.tensor(np.array(y), dtype=torch.float32)
    x = x.permute(1, 0, 2)  # (input_seq_length, num_samples, input_size)
    y = y.permute(1, 0, 2)  # (output_seq_length, num_samples, output_size)

    return x, y

# LSTM的数据可以是：
# 同一个变量的不同时间前后的数据（单变量时间序列）。（如股票价格、温度、销售量等）在不同时间点的历史数据。
# 多个不同物理属性特征和输出变量（多变量时间序列）。（如在气象预测中，输入可以包括温度、湿度、风速、气压等多个特征的时间序列。输出：可以是单个变量（如预测未来温度），也可以是多个变量（如同时预测温度和湿度）。

#序列数据的特点：跟维度没有关系，输入和输出可以是任意维度，而序列数据的意思是数据之间有时间上的关系，
#从 输入序列长度个元素 -> 输出序列长度个元素。 是一个sequence to sequence的模型。
#LSTM的输入是一个序列，输出也是一个序列，它们之间的长度都是任意的，可以设置的。


if __name__ == "__main__":
    
    seq_length = 500
    input_seq_length=20
    output_seq_length=5
    input_size = 5  # 输入维度改为5
    hidden_size = 64
    output_size = 2  # 输出维度改为2
    epochs = 200
    batch_size = 32

    X, y = generate_data(seq_length=seq_length, input_seq_length=input_seq_length, output_seq_length=output_seq_length, input_size=input_size, output_size=output_size, noise=0.01)
    X = X.permute(1, 0, 2)  # (batch_size, input_seq_length, input_size)
    y = y.permute(1, 0, 2) 
    train_size = int(0.8 * len(X))
    train_X, test_X = X[:train_size], X[train_size:]
    train_y, test_y = y[:train_size], y[train_size:]

    train_dataset = TensorDataset(train_X, train_y)
    test_dataset = TensorDataset(test_X, test_y)

    #dataloader必须(batch_size, input_seq_length, input_size)
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
            inputs = inputs.permute(1, 0, 2).to(device)  # (seq_len, batch_size, input_size)
            targets = targets.permute(1, 0, 2).to(device)  # (seq_len, batch_size, output_size)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)  # 只计算最后一个时间步的损失
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs = inputs.permute(1, 0, 2).to(device)
                targets = targets.permute(1, 0, 2).to(device)
                outputs = model(inputs)
                val_loss += criterion(outputs, targets[-1]).item()

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

    # 预测并绘制结果
    model.load_state_dict(torch.load('best_lstm.pth'))
    model.eval()
    true_labels = np.zeros((len(test_loader)*batch_size, 2))
    predictions = np.zeros((len(test_loader)*batch_size, 2))
    count=0
    with torch.no_grad(): 
        for inputs, targets in test_loader:#(batch_size, input_seq_length, input_size)
            inputs = inputs.permute(1, 0, 2).to(device)  # (input_seq_length, batch_size, input_size)
            targets = targets.permute(1, 0, 2).to(device)  # (output_seq_length, batch_size, output_size)
            for i in range(len(inputs[1])):         
                print(inputs[:,i:i+1, :])
                print(inputs[:,i:i+1, :].shape) 
                predict = model(inputs[:,i:i+1, :])
                true_labels[count,0]=targets[0,i:i+1,0].cpu().numpy()
                true_labels[count,1]=targets[0,i:i+1,1].cpu().numpy()
                predictions[count,0]=predict[0,0].cpu().numpy()
                predictions[count,1]=predict[0,1].cpu().numpy()
                count+=1


    plt.figure(figsize=(10, 5))
    plt.plot(true_labels[:, 0], label='Actual_1')
    plt.plot(predictions[:, 0], label='Predicted_1')
    plt.plot(true_labels[:, 1], label='Actual_2')
    plt.plot(predictions[:, 1], label='Predicted_2')
    plt.title('Actual vs Predicted')
    plt.xlabel('Time Steps')
    plt.ylabel('Value')
    plt.legend()
    plt.show()



